#!/usr/bin/env python3
"""
Live 5-minute trading signal engine.

Every 5 minutes:
  1. Fetch latest 5-min bars for the active pairs
  2. Recompute z-scores
  3. Generate entry/exit signals
  4. Output trade recommendations with position sizing
  5. Log signals and push to GitHub

Usage:
    python -m live_feed.trader
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from live_feed.alpaca_client import (
    execute_trade,
    fetch_5min_data_alpaca,
    get_account_info,
    get_positions,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INTERVAL_SECONDS = 60  # 1 minute, matches 1-min bar frequency
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_FILE = PROJECT_ROOT / "live_feed" / "signals.csv"
POSITIONS_FILE = PROJECT_ROOT / "live_feed" / "positions.csv"
OUTPUT_LOG = PROJECT_ROOT / "live_feed" / "trader_output.log"

# --- Strategy parameters ---
ZSCORE_LOOKBACK = 60       # 60 x 5min = 5 hours rolling window
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
TOTAL_CAPITAL = 100_000    # total account size (Alpaca paper)
MAX_PAIRS = 25
MAX_EXPOSURE_PER_PAIR = 5_000  # $5,000 per leg = $10,000 gross per pair
WATCHLIST_THRESHOLD = 1.75  # only show pairs with |z| >= 1.75
ZSCORE_HARD_STOP = 3.25    # force exit if |z| blows out past this
TIME_STOP_BARS = 390       # 5 trading days * 78 bars/day (5-min bars, 6.5hr session)
COOLDOWN_BARS = 78         # 1 trading day cooldown after hard/time stop


class PairPosition:
    """Track state for a single pair."""

    def __init__(self, ticker_a: str, ticker_b: str, hedge_ratio: float, sector: str):
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.hedge_ratio = hedge_ratio
        self.sector = sector
        self.signal = 0  # +1 long spread, -1 short spread, 0 flat
        self.entry_z = None
        self.entry_time = None
        self.bars_held = 0
        self.cooldown_remaining = 0

    def _force_exit(self, z_score: float, reason: str) -> dict:
        """Build a forced exit action and reset state."""
        action = {
            "action": "EXIT",
            "prev_signal": "LONG_SPREAD" if self.signal == 1 else "SHORT_SPREAD",
            "entry_z": self.entry_z,
            "exit_z": z_score,
            "bars_held": self.bars_held,
            "exit_reason": reason,
        }
        self.signal = 0
        self.entry_z = None
        self.entry_time = None
        self.bars_held = 0
        if reason in ("HARD_STOP", "TIME_STOP"):
            self.cooldown_remaining = COOLDOWN_BARS
        return action

    def update(self, z_score: float, timestamp: str) -> dict | None:
        """
        Update position based on z-score.
        Returns a trade action dict if a signal fires, else None.
        """
        prev_signal = self.signal
        action = None

        if self.signal == 0:
            # Tick down cooldown
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                return None

            # Look for entry
            if z_score <= -ZSCORE_ENTRY:
                self.signal = 1  # long spread: long A, short B
                self.entry_z = z_score
                self.entry_time = timestamp
                self.bars_held = 0
                action = {
                    "action": "ENTER_LONG_SPREAD",
                    "long": self.ticker_a,
                    "short": self.ticker_b,
                }
            elif z_score >= ZSCORE_ENTRY:
                self.signal = -1  # short spread: short A, long B
                self.entry_z = z_score
                self.entry_time = timestamp
                self.bars_held = 0
                action = {
                    "action": "ENTER_SHORT_SPREAD",
                    "long": self.ticker_b,
                    "short": self.ticker_a,
                }
        else:
            self.bars_held += 1

            # Hard stop: z-score blew out
            if abs(z_score) >= ZSCORE_HARD_STOP:
                action = self._force_exit(z_score, "HARD_STOP")

            # Time stop: held too long
            elif self.bars_held >= TIME_STOP_BARS:
                action = self._force_exit(z_score, "TIME_STOP")

            # Normal mean-reversion exit
            elif abs(z_score) <= ZSCORE_EXIT:
                action = self._force_exit(z_score, "PROFIT_EXIT")

        if action:
            action.update({
                "pair": f"{self.ticker_a}/{self.ticker_b}",
                "hedge_ratio": self.hedge_ratio,
                "z_score": z_score,
                "timestamp": timestamp,
                "sector": self.sector,
            })

        return action


def load_pairs() -> list[dict]:
    """
    Load pairs from top_pairs.txt or run scanner if not found.
    Returns list of pair dicts with ticker_a, ticker_b, hedge_ratio, sector.
    """
    pairs_file = PROJECT_ROOT / "top_pairs.txt"
    pairs_csv = PROJECT_ROOT / "live_feed" / "active_pairs.csv"

    if pairs_csv.exists():
        df = pd.read_csv(pairs_csv)
        return df.to_dict("records")

    # If no pre-computed pairs, use these well-known stat-arb pairs as defaults
    # (scanner should be run first for proper pair selection)
    print("WARNING: No active_pairs.csv found. Run 'python -m strategy.scanner' first.")
    print("Using fallback pairs...\n")
    return [
        {"ticker_a": "KO", "ticker_b": "PEP", "hedge_ratio": 1.0, "sector": "Consumer Staples"},
        {"ticker_a": "XOM", "ticker_b": "CVX", "hedge_ratio": 1.0, "sector": "Energy"},
        {"ticker_a": "GS", "ticker_b": "MS", "hedge_ratio": 1.0, "sector": "Financials"},
        {"ticker_a": "JPM", "ticker_b": "BAC", "hedge_ratio": 1.0, "sector": "Financials"},
        {"ticker_a": "HD", "ticker_b": "LOW", "hedge_ratio": 1.0, "sector": "Consumer Discretionary"},
    ]


def fetch_5min_data(tickers: list[str]) -> pd.DataFrame:
    """Fetch recent 5-min bars for the given tickers via Alpaca."""
    return fetch_5min_data_alpaca(tickers)


def compute_zscore(spread: pd.Series, lookback: int = ZSCORE_LOOKBACK) -> float:
    """Compute current z-score of the spread."""
    if len(spread) < lookback:
        lookback = len(spread)
    recent = spread.iloc[-lookback:]
    mean = recent.mean()
    std = recent.std()
    if std < 1e-8:
        return 0.0
    return (spread.iloc[-1] - mean) / std


def compute_shares(price: float, dollar_amount: float) -> int:
    """Convert dollar amount to whole shares."""
    if price <= 0:
        return 0
    return int(dollar_amount / price)


def format_signal_table(
    positions: list[PairPosition],
    z_scores: dict,
    latest_prices: dict,
) -> str:
    """Format current pair status with exact share counts and dollar amounts."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"\n{'='*70}")
    lines.append(f"  LIVE SIGNALS — {now}")
    lines.append(f"{'='*70}")

    total_long_dollars = 0
    total_short_dollars = 0

    hidden = 0
    for pos in positions:
        label = f"{pos.ticker_a}/{pos.ticker_b}"
        z = z_scores.get(label, 0)
        price_a = latest_prices.get(pos.ticker_a, 0)
        price_b = latest_prices.get(pos.ticker_b, 0)

        # Skip flat pairs with z-score below watchlist threshold
        if pos.signal == 0 and abs(z) < WATCHLIST_THRESHOLD:
            hidden += 1
            continue

        lines.append(f"\n  PAIR: {pos.ticker_a} / {pos.ticker_b}  ({pos.sector})")
        lines.append(f"  Z-Score: {z:+.2f}")
        lines.append(f"  Prices: {pos.ticker_a} = ${price_a:.2f}  |  {pos.ticker_b} = ${price_b:.2f}")

        if pos.signal == 0:
            # Watching — show projected trade
            shares_a = compute_shares(price_a, MAX_EXPOSURE_PER_PAIR)
            shares_b = compute_shares(price_b, MAX_EXPOSURE_PER_PAIR)
            pct = abs(z) / ZSCORE_ENTRY * 100
            lines.append(f"  Status: WATCHING ({pct:.0f}% to entry)")
            if z > 0:
                lines.append(f"  If z hits +{ZSCORE_ENTRY:.1f} → Short {shares_a} shares {pos.ticker_a} (${shares_a * price_a:,.2f})")
                lines.append(f"                    Buy {shares_b} shares {pos.ticker_b} (${shares_b * price_b:,.2f})")
            else:
                lines.append(f"  If z hits -{ZSCORE_ENTRY:.1f} → Buy {shares_a} shares {pos.ticker_a} (${shares_a * price_a:,.2f})")
                lines.append(f"                    Short {shares_b} shares {pos.ticker_b} (${shares_b * price_b:,.2f})")
        else:
            # Active position — show exact shares
            shares_a = compute_shares(price_a, MAX_EXPOSURE_PER_PAIR)
            shares_b = compute_shares(price_b, MAX_EXPOSURE_PER_PAIR)
            long_dollars = shares_a * price_a if pos.signal == 1 else shares_b * price_b
            short_dollars = shares_b * price_b if pos.signal == 1 else shares_a * price_a

            if pos.signal == 1:
                lines.append(f"  ACTION: LONG SPREAD")
                lines.append(f"    BUY  {shares_a} shares of {pos.ticker_a} @ ${price_a:.2f} = ${shares_a * price_a:,.2f}")
                lines.append(f"    SHORT {shares_b} shares of {pos.ticker_b} @ ${price_b:.2f} = ${shares_b * price_b:,.2f}")
            else:
                lines.append(f"  ACTION: SHORT SPREAD")
                lines.append(f"    SHORT {shares_a} shares of {pos.ticker_a} @ ${price_a:.2f} = ${shares_a * price_a:,.2f}")
                lines.append(f"    BUY  {shares_b} shares of {pos.ticker_b} @ ${price_b:.2f} = ${shares_b * price_b:,.2f}")

            lines.append(f"    Long:  ${long_dollars:,.2f}")
            lines.append(f"    Short: ${short_dollars:,.2f}")
            lines.append(f"    Net:   ${long_dollars - short_dollars:,.2f}")
            lines.append(f"    Entry Z: {pos.entry_z:+.2f}  |  Entry time: {pos.entry_time}")

            total_long_dollars += long_dollars
            total_short_dollars += short_dollars

    active = sum(1 for p in positions if p.signal != 0)
    watching = len(positions) - active - hidden
    gross = total_long_dollars + total_short_dollars
    net = total_long_dollars - total_short_dollars
    lines.append(f"\n{'='*70}")
    lines.append(f"  PORTFOLIO SUMMARY")
    lines.append(f"{'='*70}")
    lines.append(f"  Active: {active}  |  Watching: {watching}  |  Quiet: {hidden}")
    lines.append(f"  Total long:   ${total_long_dollars:,.2f}")
    lines.append(f"  Total short:  ${total_short_dollars:,.2f}")
    lines.append(f"  Gross exposure: ${gross:,.2f} / $10,000 max")
    lines.append(f"  Net exposure:   ${net:,.2f} (target: $0)")
    lines.append(f"  Account size:   ${TOTAL_CAPITAL:,}")
    lines.append(f"{'='*70}\n")
    return "\n".join(lines)


def log_signal(action: dict) -> None:
    """Append trade signal to signals CSV."""
    row = pd.DataFrame([action])
    if SIGNALS_FILE.exists():
        row.to_csv(SIGNALS_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(SIGNALS_FILE, index=False)


def log(text: str) -> None:
    """Write to both console and the output log file."""
    print(text, flush=True)
    with open(OUTPUT_LOG, "a") as f:
        f.write(text + "\n")


def git_push(msg: str) -> None:
    """Commit and push signal/position data."""
    try:
        files = [
            "live_feed/signals.csv",
            "live_feed/positions.csv",
            "live_feed/trader_output.log",
        ]
        cmds = [
            ["git", "add"] + files,
            ["git", "commit", "-m", msg],
            ["git", "push", "origin", "main"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, timeout=30)
    except Exception:
        pass


def run_trader() -> None:
    """Main 5-minute trading loop."""
    # Clear output log on startup
    with open(OUTPUT_LOG, "w") as f:
        f.write("")

    log("=" * 60)
    log("  LIVE STAT-ARB TRADER (5-min)")
    log(f"  Output log: {OUTPUT_LOG}")
    log("=" * 60)

    pair_configs = load_pairs()
    log(f"Loaded {len(pair_configs)} pairs.\n")

    # Initialize position trackers
    positions = []
    all_tickers = set()
    for p in pair_configs:
        pos = PairPosition(p["ticker_a"], p["ticker_b"], p["hedge_ratio"], p["sector"])
        positions.append(pos)
        all_tickers.update([p["ticker_a"], p["ticker_b"]])

    all_tickers = sorted(all_tickers)
    log(f"Tracking {len(all_tickers)} unique tickers across {len(positions)} pairs.\n")

    # Position reconciliation: sync Alpaca positions into local state
    try:
        alpaca_positions = get_positions()
        alpaca_symbols = {p["symbol"]: p for p in alpaca_positions}
        if alpaca_symbols:
            log(f"Alpaca has open positions in: {', '.join(sorted(alpaca_symbols))}")
        else:
            log("Alpaca: no open positions.")

        # Restore local signal state from Alpaca positions
        for pos in positions:
            a_pos = alpaca_symbols.get(pos.ticker_a)
            b_pos = alpaca_symbols.get(pos.ticker_b)
            if a_pos and b_pos:
                a_qty = float(a_pos["qty"])
                b_qty = float(b_pos["qty"])
                if a_qty > 0 and b_qty < 0:
                    pos.signal = 1  # long A, short B = long spread
                    pos.entry_time = "reconciled"
                    pos.entry_z = 0.0
                    log(f"  Reconciled {pos.ticker_a}/{pos.ticker_b} → LONG SPREAD")
                elif a_qty < 0 and b_qty > 0:
                    pos.signal = -1  # short A, long B = short spread
                    pos.entry_time = "reconciled"
                    pos.entry_z = 0.0
                    log(f"  Reconciled {pos.ticker_a}/{pos.ticker_b} → SHORT SPREAD")

        account = get_account_info()
        log(f"Alpaca account — equity: ${float(account['equity']):,.2f}, "
            f"cash: ${float(account['cash']):,.2f}\n")
    except Exception as e:
        log(f"Warning: could not reconcile Alpaca positions: {e}\n")

    tick = 0
    consecutive_errors = 0
    while True:
        tick += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(f"[{now}] Tick #{tick}")

        try:
            # Fetch latest 5-min data
            prices = fetch_5min_data(all_tickers)

            if prices.empty:
                log("  No data (market may be closed)")
                time.sleep(INTERVAL_SECONDS)
                consecutive_errors = 0
                continue

            # Compute z-scores and update signals
            z_scores = {}
            actions = []
            for pos in positions:
                if pos.ticker_a not in prices.columns or pos.ticker_b not in prices.columns:
                    continue

                spread = prices[pos.ticker_a] - pos.hedge_ratio * prices[pos.ticker_b]
                z = compute_zscore(spread)
                label = f"{pos.ticker_a}/{pos.ticker_b}"
                z_scores[label] = z

                action = pos.update(z, now)
                if action:
                    # Add share counts to the action
                    price_a = prices[pos.ticker_a].iloc[-1] if pos.ticker_a in prices.columns else 0
                    price_b = prices[pos.ticker_b].iloc[-1] if pos.ticker_b in prices.columns else 0
                    shares_a = compute_shares(price_a, MAX_EXPOSURE_PER_PAIR)
                    shares_b = compute_shares(price_b, MAX_EXPOSURE_PER_PAIR)
                    action["shares_a"] = shares_a
                    action["shares_b"] = shares_b
                    action["price_a"] = price_a
                    action["price_b"] = price_b

                    actions.append(action)
                    log_signal(action)

                    # Execute paper trade via Alpaca
                    try:
                        execute_trade(action)
                    except Exception as e:
                        log(f"  [ALPACA] Trade execution error: {e}")

                    if action["action"] == "EXIT":
                        reason = action.get("exit_reason", "UNKNOWN")
                        bars = action.get("bars_held", "?")
                        log(f"  >>> EXIT ({reason}): {action['pair']} @ z={z:+.2f} after {bars} bars")
                        log(f"      Sell {shares_a} shares {pos.ticker_a}, Cover {shares_b} shares {pos.ticker_b}")
                    else:
                        long_tk = action.get("long", "")
                        short_tk = action.get("short", "")
                        long_sh = shares_a if long_tk == pos.ticker_a else shares_b
                        short_sh = shares_b if short_tk == pos.ticker_b else shares_a
                        long_px = price_a if long_tk == pos.ticker_a else price_b
                        short_px = price_b if short_tk == pos.ticker_b else price_a
                        log(f"  >>> {action['action']}: {action['pair']} @ z={z:+.2f}")
                        log(f"      BUY  {long_sh} shares of {long_tk} @ ${long_px:.2f} = ${long_sh * long_px:,.2f}")
                        log(f"      SHORT {short_sh} shares of {short_tk} @ ${short_px:.2f} = ${short_sh * short_px:,.2f}")

            # Get latest prices for share calculations
            latest_prices = {}
            for t in all_tickers:
                if t in prices.columns:
                    latest_prices[t] = prices[t].iloc[-1]

            # Print status table
            table = format_signal_table(positions, z_scores, latest_prices)
            log(table)

            # Save current positions
            pos_data = []
            for pos in positions:
                label = f"{pos.ticker_a}/{pos.ticker_b}"
                pos_data.append({
                    "pair": label,
                    "signal": pos.signal,
                    "z_score": z_scores.get(label, 0),
                    "hedge_ratio": pos.hedge_ratio,
                    "sector": pos.sector,
                    "entry_time": pos.entry_time,
                })
            pd.DataFrame(pos_data).to_csv(POSITIONS_FILE, index=False)

            # Push to GitHub
            if actions:
                git_push(f"trade signal tick #{tick} — {now}")
            else:
                git_push(f"position update #{tick} — {now}")

            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            log(f"  ERROR (attempt {consecutive_errors}): {e}")
            if consecutive_errors >= 5:
                log(f"  {consecutive_errors} consecutive errors — sleeping 5 min before retry")
                time.sleep(300)
            # Keep the loop alive regardless

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run_trader()
