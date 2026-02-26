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

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd

from live_feed.alpaca_client import (
    execute_trade,
    fetch_5min_data_alpaca,
    get_account_info,
    get_positions,
    _close_position,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INTERVAL_SECONDS = 60  # 1 minute, matches 1-min bar frequency
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_FILE = PROJECT_ROOT / "live_feed" / "signals.csv"
POSITIONS_FILE = PROJECT_ROOT / "live_feed" / "positions.csv"
OUTPUT_LOG = PROJECT_ROOT / "live_feed" / "trader_output.log"
STATE_FILE = PROJECT_ROOT / "live_feed" / "position_state.json"

# --- Strategy parameters ---
ZSCORE_LOOKBACK = 60       # 60 x 5min = 5 hours rolling window
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
TOTAL_CAPITAL = 100_000    # total account size (Alpaca paper)
MAX_PAIRS = 25
BASE_EXPOSURE_PER_PAIR = 5_000  # base $5,000 per leg, adjusted by volatility
WATCHLIST_THRESHOLD = 1.75  # only show pairs with |z| >= 1.75
ZSCORE_HARD_STOP = 3.25    # force exit if |z| blows out past this
TIME_STOP_BARS = 390       # 5 trading days * 78 bars/day (5-min bars, 6.5hr session)
COOLDOWN_BARS = 78         # 1 trading day cooldown after hard/time stop
OPEN_COOLDOWN_MINUTES = 15 # skip new entries for first 15 min after market open (9:30 ET)
MAX_ENTRY_FAILURES = 3     # disable pair after this many consecutive failed entries

# --- Risk management ---
MAX_GROSS_EXPOSURE = 100_000   # cap total gross exposure at 1.0x capital
MAX_SECTOR_ACTIVE = 3          # max active positions per sector
MAX_SECTOR_LOSING = 2          # block new entries if this many in same sector are losing
VOL_LOOKBACK = 60              # bars for spread volatility calculation
PNL_FILE = Path(__file__).resolve().parent / "pair_pnl.csv"
LOSS_STREAK_CUTOFF = 3         # reduce size after this many consecutive losses
LOSS_STREAK_SCALE = 0.5        # scale factor when on a loss streak

# --- Telegram alerts ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
SLIPPAGE_FILE = Path(__file__).resolve().parent / "slippage.csv"
DAILY_SUMMARY_SENT = {}        # tracks if summary sent today {date_str: True}


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
        self.entry_shares_a = 0
        self.entry_shares_b = 0
        self.consecutive_entry_failures = 0
        # Risk management
        self.spread_vol = 1.0          # rolling spread volatility (updated each tick)
        self.entry_price_a = 0.0       # price at entry for P&L tracking
        self.entry_price_b = 0.0
        self.consecutive_losses = 0    # consecutive losing trades
        self.allocated_exposure = BASE_EXPOSURE_PER_PAIR  # vol-adjusted per-leg $

    def _force_exit(self, z_score: float, reason: str) -> dict:
        """Build a forced exit action and reset state."""
        action = {
            "action": "EXIT",
            "prev_signal": "LONG_SPREAD" if self.signal == 1 else "SHORT_SPREAD",
            "signal": self.signal,
            "entry_z": self.entry_z,
            "exit_z": z_score,
            "bars_held": self.bars_held,
            "exit_reason": reason,
            "exit_shares_a": self.entry_shares_a,
            "exit_shares_b": self.entry_shares_b,
        }
        self.signal = 0
        self.entry_z = None
        self.entry_time = None
        self.bars_held = 0
        self.entry_shares_a = 0
        self.entry_shares_b = 0
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

            # Look for entry (skip if already past hard stop — would exit immediately)
            if abs(z_score) >= ZSCORE_HARD_STOP:
                pass  # z-score too extreme, don't enter
            elif z_score <= -ZSCORE_ENTRY:
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


def compute_vol_adjusted_exposure(
    spread: pd.Series,
    all_spread_vols: list[float],
    lookback: int = VOL_LOOKBACK,
) -> float:
    """
    Compute per-leg dollar exposure inversely proportional to spread volatility.
    Higher vol → smaller allocation. Returns dollar amount per leg.
    """
    if len(spread) < lookback:
        lookback = max(len(spread), 2)
    vol = spread.iloc[-lookback:].std()
    if vol < 1e-8:
        return BASE_EXPOSURE_PER_PAIR

    # Median vol across all pairs as anchor
    if not all_spread_vols or all(v < 1e-8 for v in all_spread_vols):
        return BASE_EXPOSURE_PER_PAIR

    median_vol = np.median([v for v in all_spread_vols if v > 1e-8])
    if median_vol < 1e-8:
        return BASE_EXPOSURE_PER_PAIR

    # Scale inversely: if this pair has 2x median vol, get 0.5x allocation
    scale = median_vol / vol
    # Clamp between 0.25x and 2.0x base exposure
    scale = max(0.25, min(2.0, scale))
    return BASE_EXPOSURE_PER_PAIR * scale


def get_current_gross_exposure(positions: list, latest_prices: dict) -> float:
    """Calculate total gross dollar exposure across all active positions."""
    gross = 0.0
    for pos in positions:
        if pos.signal == 0:
            continue
        price_a = latest_prices.get(pos.ticker_a, 0)
        price_b = latest_prices.get(pos.ticker_b, 0)
        gross += pos.entry_shares_a * price_a + pos.entry_shares_b * price_b
    return gross


def count_sector_active(positions: list, sector: str) -> int:
    """Count active positions in a given sector."""
    return sum(1 for p in positions if p.signal != 0 and p.sector == sector)


def count_sector_losing(
    positions: list, sector: str, z_scores: dict,
) -> int:
    """Count active positions in a sector where z-score moved against the entry."""
    count = 0
    for p in positions:
        if p.signal == 0 or p.sector != sector:
            continue
        label = f"{p.ticker_a}/{p.ticker_b}"
        z = z_scores.get(label, 0)
        # Losing = z moved further from zero since entry (wrong direction)
        if p.entry_z is not None and p.entry_z != 0:
            if p.signal == 1 and z < p.entry_z:  # long spread, z went more negative
                count += 1
            elif p.signal == -1 and z > p.entry_z:  # short spread, z went more positive
                count += 1
    return count


def load_pair_pnl() -> dict:
    """Load per-pair P&L tracking from disk. Returns {pair_label: consecutive_losses}."""
    if not PNL_FILE.exists():
        return {}
    try:
        df = pd.read_csv(PNL_FILE)
        return dict(zip(df["pair"], df["consecutive_losses"]))
    except Exception:
        return {}


def save_pair_pnl(positions: list) -> None:
    """Save per-pair consecutive loss counts to disk."""
    rows = []
    for pos in positions:
        label = f"{pos.ticker_a}/{pos.ticker_b}"
        rows.append({"pair": label, "consecutive_losses": pos.consecutive_losses})
    pd.DataFrame(rows).to_csv(PNL_FILE, index=False)


def save_position_state(positions: list) -> None:
    """Persist full PairPosition state to disk as JSON."""
    state = {}
    for pos in positions:
        label = f"{pos.ticker_a}/{pos.ticker_b}"
        state[label] = {
            "signal": pos.signal,
            "entry_z": pos.entry_z,
            "entry_time": pos.entry_time,
            "bars_held": pos.bars_held,
            "cooldown_remaining": pos.cooldown_remaining,
            "entry_shares_a": pos.entry_shares_a,
            "entry_shares_b": pos.entry_shares_b,
            "entry_price_a": pos.entry_price_a,
            "entry_price_b": pos.entry_price_b,
            "consecutive_losses": pos.consecutive_losses,
            "consecutive_entry_failures": pos.consecutive_entry_failures,
        }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_position_state() -> dict:
    """Load persisted PairPosition state from disk. Returns {pair_label: state_dict}."""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def restore_position_from_state(pos, saved: dict) -> None:
    """Restore a PairPosition's fields from saved state."""
    pos.signal = saved.get("signal", 0)
    pos.entry_z = saved.get("entry_z")
    pos.entry_time = saved.get("entry_time")
    pos.bars_held = saved.get("bars_held", 0)
    pos.cooldown_remaining = saved.get("cooldown_remaining", 0)
    pos.entry_shares_a = saved.get("entry_shares_a", 0)
    pos.entry_shares_b = saved.get("entry_shares_b", 0)
    pos.entry_price_a = saved.get("entry_price_a", 0.0)
    pos.entry_price_b = saved.get("entry_price_b", 0.0)
    pos.consecutive_losses = saved.get("consecutive_losses", 0)
    pos.consecutive_entry_failures = saved.get("consecutive_entry_failures", 0)


def send_telegram(message: str) -> None:
    """Send a message via Telegram bot. Silently fails if not configured."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  [TELEGRAM] Failed to send: {e}", flush=True)


def alert_trade(action: dict, exposure: float = 0, pnl: float = None) -> None:
    """Send Telegram alert for a trade entry or exit."""
    act = action.get("action", "")
    pair = action.get("pair", "")
    z = action.get("z_score", 0)

    if act == "EXIT":
        reason = action.get("exit_reason", "UNKNOWN")
        bars = action.get("bars_held", "?")
        emoji = "🛑" if reason == "HARD_STOP" else "⏰" if reason == "TIME_STOP" else "✅"
        pnl_line = ""
        if pnl is not None:
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            entry_cost = action.get("entry_cost", 0)
            pct = (pnl / entry_cost * 100) if entry_cost > 0 else 0
            pnl_line = f"\n{pnl_emoji} P&L: ${pnl:+,.2f} ({pct:+.2f}%)"
        msg = (
            f"{emoji} <b>EXIT {pair}</b>\n"
            f"Reason: {reason}\n"
            f"Z-score: {z:+.2f} | Bars held: {bars}"
            f"{pnl_line}"
        )
    else:
        direction = "LONG" if "LONG" in act else "SHORT"
        long_tk = action.get("long", "")
        short_tk = action.get("short", "")
        msg = (
            f"📈 <b>{direction} SPREAD {pair}</b>\n"
            f"Buy {long_tk} / Short {short_tk}\n"
            f"Z-score: {z:+.2f} | Exposure: ${exposure:,.0f}/leg"
        )
    send_telegram(msg)


def alert_risk_block(pair: str, reason: str) -> None:
    """Log risk block (no Telegram — too noisy)."""
    pass  # logged to console/file only


def alert_disabled(pair: str) -> None:
    """Send Telegram alert when a pair is disabled."""
    send_telegram(f"🚫 <b>DISABLED</b> {pair}\nToo many consecutive entry failures")


def record_slippage(
    action: dict, signal_price_a: float, signal_price_b: float,
) -> None:
    """Query Alpaca for fill price and record slippage."""
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
            paper=True,
        )
        # Get most recent orders
        orders = client.get_orders(filter={"status": "filled", "limit": 10})
        pair = action.get("pair", "")
        tickers = pair.split("/")
        if len(tickers) != 2:
            return

        fills = {}
        for order in orders:
            if order.symbol in tickers and order.filled_avg_price:
                fills[order.symbol] = float(order.filled_avg_price)

        if len(fills) < 2:
            return

        fill_a = fills.get(tickers[0], signal_price_a)
        fill_b = fills.get(tickers[1], signal_price_b)
        slip_a = fill_a - signal_price_a
        slip_b = fill_b - signal_price_b

        row = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "action": action.get("action", ""),
            "signal_price_a": signal_price_a,
            "fill_price_a": fill_a,
            "slippage_a": slip_a,
            "signal_price_b": signal_price_b,
            "fill_price_b": fill_b,
            "slippage_b": slip_b,
        }])
        if SLIPPAGE_FILE.exists():
            row.to_csv(SLIPPAGE_FILE, mode="a", header=False, index=False)
        else:
            row.to_csv(SLIPPAGE_FILE, index=False)
    except Exception as e:
        print(f"  [SLIPPAGE] Failed to record: {e}", flush=True)


def send_daily_summary(
    positions: list, z_scores: dict, latest_prices: dict,
) -> None:
    """Send end-of-day summary via Telegram at 4:00 PM ET."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    today = now_et.strftime("%Y-%m-%d")

    # Only send once per day, between 4:00-4:05 PM ET
    if not (dtime(16, 0) <= now_et.time() <= dtime(16, 5)):
        return
    if DAILY_SUMMARY_SENT.get(today):
        return
    DAILY_SUMMARY_SENT[today] = True

    # Compute portfolio stats
    active = []
    total_unrealized = 0.0
    for pos in positions:
        if pos.signal == 0:
            continue
        label = f"{pos.ticker_a}/{pos.ticker_b}"
        z = z_scores.get(label, 0)
        price_a = latest_prices.get(pos.ticker_a, 0)
        price_b = latest_prices.get(pos.ticker_b, 0)

        # Rough unrealized P&L
        if pos.entry_price_a > 0:
            if pos.signal == 1:
                pnl = (price_a - pos.entry_price_a) * pos.entry_shares_a \
                    - (price_b - pos.entry_price_b) * pos.entry_shares_b
            else:
                pnl = -(price_a - pos.entry_price_a) * pos.entry_shares_a \
                    + (price_b - pos.entry_price_b) * pos.entry_shares_b
        else:
            pnl = 0
        total_unrealized += pnl
        active.append(f"  {label}: z={z:+.2f} P&L=${pnl:+,.0f}")

    gross = get_current_gross_exposure(positions, latest_prices)

    # Get account info
    try:
        account = get_account_info()
        equity = float(account["equity"])
        cash = float(account["cash"])
    except Exception:
        equity = 0
        cash = 0

    # Avg slippage today
    slip_note = ""
    if SLIPPAGE_FILE.exists():
        try:
            sdf = pd.read_csv(SLIPPAGE_FILE)
            today_slips = sdf[sdf["timestamp"].str.startswith(today)]
            if not today_slips.empty:
                avg_slip = today_slips[["slippage_a", "slippage_b"]].abs().mean().mean()
                slip_note = f"\nAvg slippage: ${avg_slip:.4f}"
        except Exception:
            pass

    active_count = len(active)
    positions_text = "\n".join(active[:10]) if active else "  None"
    if active_count > 10:
        positions_text += f"\n  ... +{active_count - 10} more"

    msg = (
        f"📊 <b>DAILY SUMMARY — {today}</b>\n\n"
        f"Equity: ${equity:,.2f}\n"
        f"Cash: ${cash:,.2f}\n"
        f"Unrealized P&L: ${total_unrealized:+,.2f}\n"
        f"Gross exposure: ${gross:,.0f} / ${MAX_GROSS_EXPOSURE:,.0f}\n"
        f"Active positions: {active_count}\n\n"
        f"<b>Positions:</b>\n{positions_text}"
        f"{slip_note}"
    )
    send_telegram(msg)


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
            # Watching — show projected trade with vol-adjusted exposure
            exp = pos.allocated_exposure
            shares_a = compute_shares(price_a, exp)
            shares_b = compute_shares(price_b, exp)
            pct = abs(z) / ZSCORE_ENTRY * 100
            streak_note = f" [loss streak: {pos.consecutive_losses}]" if pos.consecutive_losses >= LOSS_STREAK_CUTOFF else ""
            lines.append(f"  Status: WATCHING ({pct:.0f}% to entry, ${exp:,.0f}/leg){streak_note}")
            if z > 0:
                lines.append(f"  If z hits +{ZSCORE_ENTRY:.1f} → Short {shares_a} shares {pos.ticker_a} (${shares_a * price_a:,.2f})")
                lines.append(f"                    Buy {shares_b} shares {pos.ticker_b} (${shares_b * price_b:,.2f})")
            else:
                lines.append(f"  If z hits -{ZSCORE_ENTRY:.1f} → Buy {shares_a} shares {pos.ticker_a} (${shares_a * price_a:,.2f})")
                lines.append(f"                    Short {shares_b} shares {pos.ticker_b} (${shares_b * price_b:,.2f})")
        else:
            # Active position — show actual entry shares
            shares_a = pos.entry_shares_a if pos.entry_shares_a > 0 else compute_shares(price_a, pos.allocated_exposure)
            shares_b = pos.entry_shares_b if pos.entry_shares_b > 0 else compute_shares(price_b, pos.allocated_exposure)
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
    lines.append(f"  Gross exposure: ${gross:,.2f} / ${MAX_GROSS_EXPOSURE:,.0f} cap")
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

    # Restore position state: prefer saved state file, fall back to Alpaca reconciliation
    saved_state = load_position_state()
    restored_from_file = 0

    if saved_state:
        log(f"Found saved state for {len(saved_state)} pairs.")
        for pos in positions:
            label = f"{pos.ticker_a}/{pos.ticker_b}"
            if label in saved_state:
                restore_position_from_state(pos, saved_state[label])
                if pos.signal != 0:
                    direction = "LONG SPREAD" if pos.signal == 1 else "SHORT SPREAD"
                    log(f"  Restored {label} → {direction} (entry_z={pos.entry_z:+.2f}, bars={pos.bars_held})")
                    restored_from_file += 1

    # Cross-check with Alpaca positions for consistency
    try:
        alpaca_positions = get_positions()
        alpaca_symbols = {p["symbol"]: p for p in alpaca_positions}
        if alpaca_symbols:
            log(f"Alpaca has open positions in: {', '.join(sorted(alpaca_symbols))}")
        else:
            log("Alpaca: no open positions.")

        # If no saved state, fall back to Alpaca reconciliation
        reconciled_symbols = set()
        if restored_from_file == 0:
            log("No saved state — falling back to Alpaca reconciliation.")
            for pos in positions:
                a_pos = alpaca_symbols.get(pos.ticker_a)
                b_pos = alpaca_symbols.get(pos.ticker_b)
                if a_pos and b_pos:
                    a_qty = float(a_pos["qty"])
                    b_qty = float(b_pos["qty"])
                    if a_qty > 0 and b_qty < 0:
                        pos.signal = 1
                        pos.entry_time = "reconciled"
                        pos.entry_z = 0.0
                        log(f"  Reconciled {pos.ticker_a}/{pos.ticker_b} → LONG SPREAD")
                        reconciled_symbols.update([pos.ticker_a, pos.ticker_b])
                    elif a_qty < 0 and b_qty > 0:
                        pos.signal = -1
                        pos.entry_time = "reconciled"
                        pos.entry_z = 0.0
                        log(f"  Reconciled {pos.ticker_a}/{pos.ticker_b} → SHORT SPREAD")
                        reconciled_symbols.update([pos.ticker_a, pos.ticker_b])
        else:
            # Mark symbols from restored positions
            for pos in positions:
                if pos.signal != 0:
                    reconciled_symbols.update([pos.ticker_a, pos.ticker_b])

        # Close orphaned positions (from removed pairs after a rescan)
        orphaned = set(alpaca_symbols.keys()) - reconciled_symbols
        if orphaned:
            log(f"  Closing {len(orphaned)} orphaned positions: {', '.join(sorted(orphaned))}")
            for symbol in orphaned:
                _close_position(symbol)
                log(f"    Closed orphaned position: {symbol}")

        account = get_account_info()
        log(f"Alpaca account — equity: ${float(account['equity']):,.2f}, "
            f"cash: ${float(account['cash']):,.2f}\n")
    except Exception as e:
        log(f"Warning: could not reconcile Alpaca positions: {e}\n")

    tick = 0
    consecutive_errors = 0
    last_trading_date = None  # track date to reset entry failures daily
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

            # Check market hours — only trade during regular session (9:30-16:00 ET)
            now_et = datetime.now(ZoneInfo("America/New_York"))
            market_open_time = dtime(9, 30)
            market_close_time = dtime(16, 0)
            market_is_open = market_open_time <= now_et.time() <= market_close_time and now_et.weekday() < 5

            if not market_is_open:
                log("  Market closed — skipping signal processing")
                save_position_state(positions)
                time.sleep(INTERVAL_SECONDS)
                consecutive_errors = 0
                continue

            # Reset consecutive entry failures at start of each trading day
            today_str = now_et.strftime("%Y-%m-%d")
            if last_trading_date != today_str:
                last_trading_date = today_str
                for pos in positions:
                    if pos.consecutive_entry_failures > 0:
                        log(f"  [RESET] {pos.ticker_a}/{pos.ticker_b} entry failures reset (was {pos.consecutive_entry_failures})")
                        pos.consecutive_entry_failures = 0

            # Check if we're in the opening cooldown period (first 15 min after 9:30 ET)
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            cooldown_end = now_et.replace(hour=9, minute=30 + OPEN_COOLDOWN_MINUTES, second=0, microsecond=0)
            in_open_cooldown = market_open <= now_et < cooldown_end
            if in_open_cooldown:
                log(f"  Opening cooldown active until 9:{30 + OPEN_COOLDOWN_MINUTES} ET — no new entries")

            # First pass: compute all spread vols for vol-weighting
            spreads = {}
            all_spread_vols = []
            for pos in positions:
                if pos.ticker_a not in prices.columns or pos.ticker_b not in prices.columns:
                    continue
                spread = prices[pos.ticker_a] - pos.hedge_ratio * prices[pos.ticker_b]
                label = f"{pos.ticker_a}/{pos.ticker_b}"
                spreads[label] = spread
                lookback = min(VOL_LOOKBACK, max(len(spread), 2))
                vol = spread.iloc[-lookback:].std()
                pos.spread_vol = vol
                all_spread_vols.append(vol)

            # Compute z-scores and update signals
            z_scores = {}
            actions = []
            for pos in positions:
                label = f"{pos.ticker_a}/{pos.ticker_b}"
                if label not in spreads:
                    continue

                spread = spreads[label]
                z = compute_zscore(spread)
                z_scores[label] = z

                price_a = prices[pos.ticker_a].iloc[-1]
                price_b = prices[pos.ticker_b].iloc[-1]

                # Vol-adjusted exposure per leg
                exposure = compute_vol_adjusted_exposure(spread, all_spread_vols)
                # Scale down on loss streak
                if pos.consecutive_losses >= LOSS_STREAK_CUTOFF:
                    exposure *= LOSS_STREAK_SCALE
                pos.allocated_exposure = exposure

                shares_a = compute_shares(price_a, exposure)
                shares_b = compute_shares(price_b, exposure)

                # During opening cooldown, skip flat positions (no new entries)
                # but still allow exits for active positions
                if in_open_cooldown and pos.signal == 0:
                    continue

                # Skip pairs that have failed too many consecutive entries
                if pos.signal == 0 and pos.consecutive_entry_failures >= MAX_ENTRY_FAILURES:
                    continue

                # --- Risk gates (only for new entries, not exits) ---
                if pos.signal == 0 and abs(z) >= ZSCORE_ENTRY:
                    # 1. Gross exposure cap
                    latest_prices_snap = {}
                    for t in all_tickers:
                        if t in prices.columns:
                            latest_prices_snap[t] = prices[t].iloc[-1]
                    current_gross = get_current_gross_exposure(positions, latest_prices_snap)
                    new_gross = shares_a * price_a + shares_b * price_b
                    if current_gross + new_gross > MAX_GROSS_EXPOSURE:
                        log(f"  [RISK] Skipping {label}: gross exposure would be ${current_gross + new_gross:,.0f} > ${MAX_GROSS_EXPOSURE:,.0f} cap")
                        alert_risk_block(label, f"Gross exposure ${current_gross + new_gross:,.0f} > ${MAX_GROSS_EXPOSURE:,.0f} cap")
                        continue

                    # 2. Sector concentration limit
                    sector_active = count_sector_active(positions, pos.sector)
                    if sector_active >= MAX_SECTOR_ACTIVE:
                        log(f"  [RISK] Skipping {label}: {sector_active} active in {pos.sector} (max {MAX_SECTOR_ACTIVE})")
                        alert_risk_block(label, f"{sector_active} active in {pos.sector} (max {MAX_SECTOR_ACTIVE})")
                        continue

                    # 3. Sector losing check
                    sector_losing = count_sector_losing(positions, pos.sector, z_scores)
                    if sector_losing >= MAX_SECTOR_LOSING:
                        log(f"  [RISK] Skipping {label}: {sector_losing} losing positions in {pos.sector}")
                        alert_risk_block(label, f"{sector_losing} losing in {pos.sector}")
                        continue

                action = pos.update(z, now)
                if action:
                    if action["action"] != "EXIT":
                        # Store entry shares and prices on the position
                        pos.entry_shares_a = shares_a
                        pos.entry_shares_b = shares_b
                        pos.entry_price_a = price_a
                        pos.entry_price_b = price_b
                    action["shares_a"] = shares_a
                    action["shares_b"] = shares_b
                    action["price_a"] = price_a
                    action["price_b"] = price_b

                    # Execute paper trade via Alpaca
                    trade_ok = False
                    try:
                        trade_ok = execute_trade(action)
                    except Exception as e:
                        log(f"  [ALPACA] Trade execution error: {e}")

                    if not trade_ok and action["action"] != "EXIT":
                        # Entry failed — roll back PairPosition state
                        pos.consecutive_entry_failures += 1
                        if pos.consecutive_entry_failures >= MAX_ENTRY_FAILURES:
                            log(f"  [DISABLED] {pos.ticker_a}/{pos.ticker_b} disabled after {MAX_ENTRY_FAILURES} consecutive entry failures")
                            alert_disabled(f"{pos.ticker_a}/{pos.ticker_b}")
                        else:
                            log(f"  [ROLLBACK] Entry failed for {pos.ticker_a}/{pos.ticker_b} ({pos.consecutive_entry_failures}/{MAX_ENTRY_FAILURES}), resetting to flat")
                        pos.signal = 0
                        pos.entry_z = None
                        pos.entry_time = None
                        pos.bars_held = 0
                        pos.entry_shares_a = 0
                        pos.entry_shares_b = 0
                        continue  # skip logging this as a successful action

                    # Successful entry — reset failure counter
                    if action["action"] != "EXIT":
                        pos.consecutive_entry_failures = 0

                    # Track P&L on exits
                    trade_pnl = None
                    if action["action"] == "EXIT" and pos.entry_price_a > 0:
                        # Compute rough P&L from entry vs exit prices
                        if action.get("signal") == 1:  # was long A, short B
                            trade_pnl = (price_a - pos.entry_price_a) * pos.entry_shares_a \
                                - (price_b - pos.entry_price_b) * pos.entry_shares_b
                        else:  # was short A, long B
                            trade_pnl = -(price_a - pos.entry_price_a) * pos.entry_shares_a \
                                + (price_b - pos.entry_price_b) * pos.entry_shares_b
                        # Store entry cost for % calculation
                        entry_cost = pos.entry_price_a * pos.entry_shares_a \
                            + pos.entry_price_b * pos.entry_shares_b
                        action["entry_cost"] = entry_cost
                        if trade_pnl < 0:
                            pos.consecutive_losses += 1
                            log(f"  [P&L] {label}: loss (${trade_pnl:+,.2f}), streak: {pos.consecutive_losses}")
                        else:
                            pos.consecutive_losses = 0
                            log(f"  [P&L] {label}: win (${trade_pnl:+,.2f}), streak reset")
                        pos.entry_price_a = 0.0
                        pos.entry_price_b = 0.0
                        save_pair_pnl(positions)

                    actions.append(action)
                    log_signal(action)
                    alert_trade(action, exposure, pnl=trade_pnl)
                    record_slippage(action, price_a, price_b)

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
                        log(f"  >>> {action['action']}: {action['pair']} @ z={z:+.2f} (exposure: ${exposure:,.0f}/leg)")
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

            # Send daily summary at market close
            send_daily_summary(positions, z_scores, latest_prices)

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

            # Persist full position state for restart recovery
            save_position_state(positions)

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
