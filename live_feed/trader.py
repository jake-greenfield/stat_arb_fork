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
    cancel_all_orders,
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
MAX_PAIRS = 8
BASE_EXPOSURE_PER_PAIR = 10_000  # base $10,000 per leg, adjusted by volatility
WATCHLIST_THRESHOLD = 1.75  # only show pairs with |z| >= 1.75
ZSCORE_HARD_STOP = 3.25    # force exit if |z| blows out past this
TIME_STOP_BARS = 78        # 1 trading day (6.5hr session at 5-min bars) — intraday only
COOLDOWN_BARS = 78         # 1 trading day cooldown after hard/time stop
OPEN_COOLDOWN_MINUTES = 30 # skip new entries for first 30 min after market open (9:30 ET)
EOD_CLOSE_TIME = dtime(15, 45)   # force exit all positions at 3:45 PM ET
EOD_NO_ENTRY_TIME = dtime(15, 30) # block new entries after 3:30 PM ET
MAX_ENTRY_FAILURES = 3     # disable pair after this many consecutive failed entries
TRAILING_STOP_PCT = 0.02   # exit if P&L drops 2% of entry cost from peak

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
DAILY_TRADES = []              # accumulates trade dicts for daily recap
MORNING_CHECK_SENT = {}        # tracks if morning health check sent today
HOURLY_PULSE_SENT = {}         # tracks hourly pulse {hour_key: True}
WEEKLY_REPORT_SENT = {}        # tracks if weekly report sent {week_key: True}
POSITION_IMBALANCE_PCT = 0.15  # alert if long/short legs differ by >15%


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
        self.peak_pnl = 0.0            # peak unrealized P&L since entry (trailing stop)
        self._trailing_stop_triggered = False

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
        self.peak_pnl = 0.0
        self._trailing_stop_triggered = False
        if reason in ("HARD_STOP", "TIME_STOP", "TRAILING_STOP"):
            self.cooldown_remaining = COOLDOWN_BARS
        return action

    def compute_unrealized_pnl(self, price_a: float, price_b: float) -> float:
        """Compute current unrealized P&L for an active position."""
        if self.signal == 0 or self.entry_price_a == 0:
            return 0.0
        if self.signal == 1:  # long A, short B
            return (price_a - self.entry_price_a) * self.entry_shares_a \
                - (price_b - self.entry_price_b) * self.entry_shares_b
        else:  # short A, long B
            return -(price_a - self.entry_price_a) * self.entry_shares_a \
                + (price_b - self.entry_price_b) * self.entry_shares_b

    def update_trailing_stop(self, price_a: float, price_b: float) -> None:
        """Update peak P&L and set trailing stop flag if drawdown exceeds threshold."""
        if self.signal == 0 or self.entry_price_a == 0:
            return
        pnl = self.compute_unrealized_pnl(price_a, price_b)
        self.peak_pnl = max(self.peak_pnl, pnl)
        entry_cost = self.entry_price_a * self.entry_shares_a \
            + self.entry_price_b * self.entry_shares_b
        if entry_cost <= 0:
            return
        drawdown = self.peak_pnl - pnl
        if drawdown > entry_cost * TRAILING_STOP_PCT:
            self._trailing_stop_triggered = True

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
                self.peak_pnl = 0.0
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
                self.peak_pnl = 0.0
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

            # Trailing stop: P&L dropped too far from peak (checked via update_trailing_stop)
            elif self._trailing_stop_triggered:
                action = self._force_exit(z_score, "TRAILING_STOP")

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
            "peak_pnl": pos.peak_pnl,
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
    pos.peak_pnl = saved.get("peak_pnl", 0.0)


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
    """Send Telegram alert for a trade entry or exit with full dollar details."""
    act = action.get("action", "")
    pair = action.get("pair", "")
    z = action.get("z_score", 0)

    if act == "EXIT":
        reason = action.get("exit_reason", "UNKNOWN")
        bars = action.get("bars_held", "?")
        emoji = "🛑" if reason == "HARD_STOP" else "⏰" if reason == "TIME_STOP" else "📉" if reason == "TRAILING_STOP" else "🕐" if reason == "EOD_CLOSE" else "✅"
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
    elif act in ("ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD"):
        direction = "LONG" if "LONG" in act else "SHORT"
        long_tk = action.get("long", "")
        short_tk = action.get("short", "")
        shares_long = action.get("shares_long", 0)
        shares_short = action.get("shares_short", 0)
        # Get fill or signal prices for dollar calculations
        price_a = action.get("price_a", 0)
        price_b = action.get("price_b", 0)
        if long_tk == pair.split("/")[0] if "/" in pair else "":
            long_px, short_px = price_a, price_b
        else:
            long_px, short_px = price_b, price_a
        long_dollars = shares_long * long_px
        short_dollars = shares_short * short_px

        msg = (
            f"📈 <b>{direction} SPREAD {pair}</b>\n"
            f"BUY {shares_long} {long_tk} @ ${long_px:.2f} = ${long_dollars:,.0f}\n"
            f"SHORT {shares_short} {short_tk} @ ${short_px:.2f} = ${short_dollars:,.0f}\n"
            f"Z-score: {z:+.2f} | Exposure: ${exposure:,.0f}/leg"
        )
        # Balance check: alert if legs are imbalanced
        if long_dollars > 0 and short_dollars > 0:
            imbalance = abs(long_dollars - short_dollars) / max(long_dollars, short_dollars)
            if imbalance > POSITION_IMBALANCE_PCT:
                msg += f"\n⚠️ IMBALANCED: long ${long_dollars:,.0f} vs short ${short_dollars:,.0f} ({imbalance:.0%} diff)"
    else:
        # Fallback for EOD_CLOSE batch alert etc.
        msg = f"📋 <b>{act}</b> {pair}"

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
    """Send comprehensive end-of-day summary via Telegram at 4:00 PM ET."""
    global DAILY_TRADES
    now_et = datetime.now(ZoneInfo("America/New_York"))
    today = now_et.strftime("%Y-%m-%d")

    # Only send once per day, between 4:00-4:05 PM ET
    if not (dtime(16, 0) <= now_et.time() <= dtime(16, 5)):
        return
    if DAILY_SUMMARY_SENT.get(today):
        return
    DAILY_SUMMARY_SENT[today] = True

    # --- Account info ---
    try:
        account = get_account_info()
        equity = float(account["equity"])
        cash = float(account["cash"])
    except Exception:
        equity = 0
        cash = 0

    # --- Realized P&L from today's trades ---
    exits = [t for t in DAILY_TRADES if t["action"] == "EXIT" and t.get("pnl") is not None]
    entries = [t for t in DAILY_TRADES if t["action"] != "EXIT"]
    total_realized_pnl = sum(t["pnl"] for t in exits)
    wins = [t for t in exits if t["pnl"] >= 0]
    losses = [t for t in exits if t["pnl"] < 0]
    win_rate = (len(wins) / len(exits) * 100) if exits else 0

    # Breakdown by exit reason
    reason_counts = {}
    for t in exits:
        reason = t.get("exit_reason", "UNKNOWN")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Per-pair breakdown
    pair_pnl = {}
    for t in exits:
        pair = t.get("pair", "?")
        if pair not in pair_pnl:
            pair_pnl[pair] = 0.0
        pair_pnl[pair] += t["pnl"]

    # --- Slippage ---
    total_slippage = 0.0
    slip_count = 0
    if SLIPPAGE_FILE.exists():
        try:
            sdf = pd.read_csv(SLIPPAGE_FILE)
            today_slips = sdf[sdf["timestamp"].str.startswith(today)]
            if not today_slips.empty:
                total_slippage = today_slips[["slippage_a", "slippage_b"]].abs().sum().sum()
                slip_count = len(today_slips)
        except Exception:
            pass
    # Also compute slippage from DAILY_TRADES fill vs signal prices
    fill_slippage = 0.0
    for t in DAILY_TRADES:
        fill_a = t.get("fill_price_a", 0)
        fill_b = t.get("fill_price_b", 0)
        sig_a = t.get("signal_price_a", 0)
        sig_b = t.get("signal_price_b", 0)
        if fill_a > 0 and sig_a > 0:
            fill_slippage += abs(fill_a - sig_a)
        if fill_b > 0 and sig_b > 0:
            fill_slippage += abs(fill_b - sig_b)

    # --- Active positions (should be 0 after EOD close) ---
    active_positions = [p for p in positions if p.signal != 0]

    # --- Build message ---
    pnl_emoji = "🟢" if total_realized_pnl >= 0 else "🔴"

    msg = f"📊 <b>DAILY RECAP — {today}</b>\n\n"

    # Account
    msg += f"<b>Account</b>\n"
    msg += f"  Equity: ${equity:,.2f}\n"
    msg += f"  Cash: ${cash:,.2f}\n\n"

    # P&L
    msg += f"<b>Realized P&L</b>\n"
    msg += f"  {pnl_emoji} Total: ${total_realized_pnl:+,.2f}\n"
    msg += f"  Trades: {len(exits)} exits, {len(entries)} entries\n"
    msg += f"  Win rate: {win_rate:.0f}% ({len(wins)}W / {len(losses)}L)\n"

    # Exit reasons
    if reason_counts:
        reasons_str = ", ".join(f"{r}: {c}" for r, c in sorted(reason_counts.items()))
        msg += f"  Exit types: {reasons_str}\n"
    msg += "\n"

    # Per-pair breakdown
    if pair_pnl:
        msg += f"<b>Per-Pair P&L</b>\n"
        for pair, pnl in sorted(pair_pnl.items(), key=lambda x: x[1]):
            pair_emoji = "🟢" if pnl >= 0 else "🔴"
            msg += f"  {pair_emoji} {pair}: ${pnl:+,.2f}\n"
        msg += "\n"

    # Slippage
    msg += f"<b>Slippage</b>\n"
    msg += f"  Total fill vs signal: ${fill_slippage:,.2f} ({len(DAILY_TRADES)} orders)\n"
    if slip_count > 0:
        msg += f"  Avg per order: ${total_slippage / slip_count:,.4f}\n"
    msg += "\n"

    # Remaining positions (should be empty after EOD)
    if active_positions:
        msg += f"⚠️ <b>{len(active_positions)} positions still open!</b>\n"
        for pos in active_positions:
            label = f"{pos.ticker_a}/{pos.ticker_b}"
            msg += f"  {label} (signal={pos.signal}, bars={pos.bars_held})\n"
        msg += "\n"

    # Diagnostic flags
    diagnostics = []
    if len(losses) > len(wins) and len(exits) >= 3:
        diagnostics.append("⚠️ More losses than wins today")
    hard_stops = reason_counts.get("HARD_STOP", 0)
    if hard_stops >= 2:
        diagnostics.append(f"⚠️ {hard_stops} hard stops — check pair selection")
    trailing_stops = reason_counts.get("TRAILING_STOP", 0)
    if trailing_stops >= 3:
        diagnostics.append(f"⚠️ {trailing_stops} trailing stops — possible choppy market")
    if fill_slippage > 50:
        diagnostics.append(f"⚠️ High slippage: ${fill_slippage:,.2f}")
    if total_realized_pnl < -200:
        diagnostics.append(f"🔴 Significant loss day — review pair performance")

    if diagnostics:
        msg += "<b>Diagnostics</b>\n"
        for d in diagnostics:
            msg += f"  {d}\n"

    send_telegram(msg)

    # Reset daily trades for next day
    DAILY_TRADES = []


def send_morning_health_check(positions: list) -> None:
    """Send morning health check at 10:00 AM ET after cooldown ends."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    today = now_et.strftime("%Y-%m-%d")

    # Send once per day, between 10:00-10:05 AM ET
    if not (dtime(10, 0) <= now_et.time() <= dtime(10, 5)):
        return
    if MORNING_CHECK_SENT.get(today):
        return
    MORNING_CHECK_SENT[today] = True

    # Account info
    try:
        account = get_account_info()
        equity = float(account["equity"])
        cash = float(account["cash"])
    except Exception:
        equity = 0
        cash = 0

    # Cross-check internal state vs Alpaca
    mismatches = []
    try:
        alpaca_positions = get_positions()
        alpaca_symbols = {p["symbol"]: p for p in alpaca_positions}

        # Check each pair
        internal_symbols = set()
        for pos in positions:
            if pos.signal != 0:
                internal_symbols.update([pos.ticker_a, pos.ticker_b])
                # Verify Alpaca has these positions
                if pos.ticker_a not in alpaca_symbols:
                    mismatches.append(f"⚠️ {pos.ticker_a}: internal=active, Alpaca=missing")
                if pos.ticker_b not in alpaca_symbols:
                    mismatches.append(f"⚠️ {pos.ticker_b}: internal=active, Alpaca=missing")

        # Check for orphaned Alpaca positions
        orphaned = set(alpaca_symbols.keys()) - internal_symbols
        for sym in orphaned:
            p = alpaca_symbols[sym]
            mismatches.append(f"⚠️ {sym}: Alpaca has {p['qty']} shares, internal=flat (orphaned)")
    except Exception as e:
        mismatches.append(f"⚠️ Could not check Alpaca: {e}")

    # Position summary
    active = [p for p in positions if p.signal != 0]
    on_cooldown = [p for p in positions if p.signal == 0 and p.cooldown_remaining > 0]

    msg = f"🌅 <b>MORNING CHECK — {today}</b>\n\n"
    msg += f"Equity: ${equity:,.2f} | Cash: ${cash:,.2f}\n"
    msg += f"Active: {len(active)} | Cooldown: {len(on_cooldown)} | Flat: {len(positions) - len(active) - len(on_cooldown)}\n\n"

    if active:
        msg += "<b>Active Positions</b>\n"
        for pos in active:
            label = f"{pos.ticker_a}/{pos.ticker_b}"
            direction = "LONG" if pos.signal == 1 else "SHORT"
            long_val = pos.entry_shares_a * pos.entry_price_a if pos.signal == 1 else pos.entry_shares_b * pos.entry_price_b
            short_val = pos.entry_shares_b * pos.entry_price_b if pos.signal == 1 else pos.entry_shares_a * pos.entry_price_a
            msg += f"  {label}: {direction} (z={pos.entry_z:+.2f}, bars={pos.bars_held})\n"
            msg += f"    Long ${long_val:,.0f} / Short ${short_val:,.0f}\n"
    else:
        msg += "No active positions\n"

    if on_cooldown:
        msg += f"\n<b>On Cooldown</b>\n"
        for pos in on_cooldown:
            label = f"{pos.ticker_a}/{pos.ticker_b}"
            bars_left = pos.cooldown_remaining
            msg += f"  {label}: {bars_left} bars remaining\n"

    if mismatches:
        msg += f"\n<b>State Mismatches</b>\n"
        for m in mismatches:
            msg += f"  {m}\n"
    else:
        msg += "\n✅ Internal state matches Alpaca"

    send_telegram(msg)


def send_hourly_pulse(positions: list, z_scores: dict, latest_prices: dict) -> None:
    """Send hourly one-line portfolio pulse during market hours."""
    now_et = datetime.now(ZoneInfo("America/New_York"))

    # Only during market hours, at the top of each hour (XX:00 - XX:02)
    if not (dtime(10, 0) <= now_et.time() <= dtime(15, 45)):
        return
    if now_et.minute > 2:
        return

    hour_key = now_et.strftime("%Y-%m-%d-%H")
    if HOURLY_PULSE_SENT.get(hour_key):
        return
    HOURLY_PULSE_SENT[hour_key] = True

    active = [p for p in positions if p.signal != 0]
    total_unrealized = 0.0
    worst_pair = ""
    worst_pnl = 0.0
    best_pair = ""
    best_pnl = 0.0

    for pos in active:
        label = f"{pos.ticker_a}/{pos.ticker_b}"
        price_a = latest_prices.get(pos.ticker_a, 0)
        price_b = latest_prices.get(pos.ticker_b, 0)
        pnl = pos.compute_unrealized_pnl(price_a, price_b)
        total_unrealized += pnl
        if pnl < worst_pnl:
            worst_pnl = pnl
            worst_pair = label
        if pnl > best_pnl:
            best_pnl = pnl
            best_pair = label

    hour_label = now_et.strftime("%-I%p").lower()
    pnl_emoji = "🟢" if total_unrealized >= 0 else "🔴"

    msg = f"⏱ <b>{hour_label}</b> — {len(active)} active, {pnl_emoji} unrealized: ${total_unrealized:+,.2f}"

    # Add today's realized from DAILY_TRADES
    exits_today = [t for t in DAILY_TRADES if t["action"] == "EXIT" and t.get("pnl") is not None]
    if exits_today:
        realized = sum(t["pnl"] for t in exits_today)
        msg += f" | realized: ${realized:+,.2f} ({len(exits_today)} trades)"

    if worst_pair and worst_pnl < -20:
        msg += f"\n  Worst: {worst_pair} ${worst_pnl:+,.2f}"
    if best_pair and best_pnl > 20:
        msg += f"\n  Best: {best_pair} ${best_pnl:+,.2f}"

    send_telegram(msg)


def send_weekly_report(positions: list) -> None:
    """Send weekly performance report on Friday at 4:10 PM ET."""
    now_et = datetime.now(ZoneInfo("America/New_York"))

    # Friday only (weekday=4), between 4:10-4:15 PM ET (after daily recap)
    if now_et.weekday() != 4:
        return
    if not (dtime(16, 10) <= now_et.time() <= dtime(16, 15)):
        return

    week_key = now_et.strftime("%Y-W%U")
    if WEEKLY_REPORT_SENT.get(week_key):
        return
    WEEKLY_REPORT_SENT[week_key] = True

    # Read signals.csv for this week's trades
    if not SIGNALS_FILE.exists():
        return

    try:
        sdf = pd.read_csv(SIGNALS_FILE)
        if "timestamp" not in sdf.columns:
            return

        # Filter to this week (Monday through Friday)
        monday = now_et - pd.Timedelta(days=now_et.weekday())
        monday_str = monday.strftime("%Y-%m-%d")
        week_trades = sdf[sdf["timestamp"] >= monday_str]

        exits = week_trades[week_trades["action"] == "EXIT"]
        entries = week_trades[week_trades["action"] != "EXIT"]

        if exits.empty and entries.empty:
            send_telegram(f"📊 <b>WEEKLY REPORT — {week_key}</b>\n\nNo trades this week.")
            return

        msg = f"📊 <b>WEEKLY REPORT — {week_key}</b>\n\n"

        # Trade counts
        msg += f"<b>Activity</b>\n"
        msg += f"  Entries: {len(entries)} | Exits: {len(exits)}\n"

        # Exit reason breakdown
        if "exit_reason" in exits.columns and not exits.empty:
            reason_counts = exits["exit_reason"].value_counts()
            reasons = ", ".join(f"{r}: {c}" for r, c in reason_counts.items())
            msg += f"  Exit types: {reasons}\n"

        # Per-pair stats
        if "pair" in exits.columns and not exits.empty:
            msg += f"\n<b>Per-Pair (exits)</b>\n"
            pair_groups = exits.groupby("pair")
            for pair_name, group in pair_groups:
                count = len(group)
                # Count exit reasons for this pair
                if "exit_reason" in group.columns:
                    reasons = group["exit_reason"].value_counts()
                    reason_str = ", ".join(f"{r}:{c}" for r, c in reasons.items())
                else:
                    reason_str = ""
                avg_bars = group["bars_held"].mean() if "bars_held" in group.columns else 0
                msg += f"  {pair_name}: {count} trades, avg hold {avg_bars:.0f} bars"
                if reason_str:
                    msg += f" ({reason_str})"
                msg += "\n"

        # Slippage summary for the week
        if SLIPPAGE_FILE.exists():
            try:
                slip_df = pd.read_csv(SLIPPAGE_FILE)
                week_slips = slip_df[slip_df["timestamp"] >= monday_str]
                if not week_slips.empty:
                    total_slip = week_slips[["slippage_a", "slippage_b"]].abs().sum().sum()
                    avg_slip = week_slips[["slippage_a", "slippage_b"]].abs().mean().mean()
                    msg += f"\n<b>Slippage</b>\n"
                    msg += f"  Total: ${total_slip:,.2f} | Avg per leg: ${avg_slip:,.4f}\n"
            except Exception:
                pass

        # Pair health: consecutive losses
        msg += f"\n<b>Pair Health</b>\n"
        for pos in positions:
            label = f"{pos.ticker_a}/{pos.ticker_b}"
            status = "🟢" if pos.consecutive_losses < LOSS_STREAK_CUTOFF else "🔴"
            notes = []
            if pos.consecutive_losses > 0:
                notes.append(f"losses: {pos.consecutive_losses}")
            if pos.cooldown_remaining > 0:
                notes.append(f"cooldown: {pos.cooldown_remaining}")
            if pos.consecutive_entry_failures > 0:
                notes.append(f"entry fails: {pos.consecutive_entry_failures}")
            note_str = f" ({', '.join(notes)})" if notes else ""
            msg += f"  {status} {label}{note_str}\n"

        # Account
        try:
            account = get_account_info()
            equity = float(account["equity"])
            msg += f"\nEquity: ${equity:,.2f}"
        except Exception:
            pass

        send_telegram(msg)
    except Exception as e:
        print(f"  [WEEKLY] Failed to send report: {e}", flush=True)


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


def _seconds_until_market_open() -> int:
    """Calculate seconds until next market open (9:30 ET, next weekday)."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    target = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    # If it's before 9:30 on a weekday, open is today
    if now_et.weekday() < 5 and now_et < target:
        delta = (target - now_et).total_seconds()
        return max(int(delta), 60)

    # Otherwise, next weekday
    days_ahead = 1
    next_day = now_et + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:  # skip Saturday/Sunday
        next_day += pd.Timedelta(days=1)
        days_ahead += 1

    target = (now_et + pd.Timedelta(days=days_ahead)).replace(
        hour=9, minute=30, second=0, microsecond=0
    )
    delta = (target - now_et).total_seconds()
    return max(int(delta), 60)


def _maybe_run_weekly_rescan(last_rescan_date: str | None) -> str | None:
    """Run pair rescan on Sunday night (20:00 ET) or if never run. Returns new date or same."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    today_str = now_et.strftime("%Y-%m-%d")

    if last_rescan_date == today_str:
        return last_rescan_date  # already ran today

    # Run on Sunday evening (day 6) after 20:00 ET
    if now_et.weekday() == 6 and now_et.hour >= 20:
        log("=== WEEKLY PAIR RESCAN (Sunday night) ===")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "strategy.scanner"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                log("Rescan completed successfully.")
                if result.stdout:
                    # Log last few lines of output
                    for line in result.stdout.strip().split("\n")[-5:]:
                        log(f"  {line}")
            else:
                log(f"Rescan failed: {result.stderr[:500]}")
        except Exception as e:
            log(f"Rescan error: {e}")
        return today_str

    return last_rescan_date


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

    # Cancel any stale orders from previous session before reconciling
    try:
        cancel_all_orders()
        log("Cancelled all stale open orders.")
    except Exception as e:
        log(f"Warning: could not cancel stale orders: {e}")

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
    last_rescan_date = None   # track weekly rescan
    while True:
        tick += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(f"[{now}] Tick #{tick}")

        try:
            # Fetch latest 5-min data
            prices = fetch_5min_data(all_tickers)

            if prices.empty:
                log("  No data (market may be closed)")
                # If outside market hours, sleep until next open
                now_et_check = datetime.now(ZoneInfo("America/New_York"))
                if not (dtime(9, 30) <= now_et_check.time() <= dtime(16, 0) and now_et_check.weekday() < 5):
                    sleep_secs = _seconds_until_market_open()
                    hours = sleep_secs // 3600
                    mins = (sleep_secs % 3600) // 60
                    log(f"  Sleeping {hours}h {mins}m until next market open")
                    time.sleep(sleep_secs)
                else:
                    time.sleep(INTERVAL_SECONDS)
                consecutive_errors = 0
                continue

            # Check market hours — only trade during regular session (9:30-16:00 ET)
            now_et = datetime.now(ZoneInfo("America/New_York"))
            market_open_time = dtime(9, 30)
            market_close_time = dtime(16, 0)
            market_is_open = market_open_time <= now_et.time() <= market_close_time and now_et.weekday() < 5

            if not market_is_open:
                # Check for weekly rescan (Sunday 20:00 ET)
                last_rescan_date = _maybe_run_weekly_rescan(last_rescan_date)
                if last_rescan_date and last_rescan_date == now_et.strftime("%Y-%m-%d"):
                    # Rescan just ran — reload pairs
                    new_configs = load_pairs()
                    if new_configs and len(new_configs) != len(pair_configs):
                        log(f"Rescan produced {len(new_configs)} pairs (was {len(pair_configs)}). Will reload on next startup.")

                # Smart sleep: wait until next market open instead of 60s ticks
                sleep_secs = _seconds_until_market_open()
                hours = sleep_secs // 3600
                mins = (sleep_secs % 3600) // 60
                log(f"  Market closed — sleeping {hours}h {mins}m until next open")
                save_position_state(positions)
                time.sleep(sleep_secs)
                consecutive_errors = 0
                continue

            # Reset consecutive entry failures and daily trades at start of each trading day
            today_str = now_et.strftime("%Y-%m-%d")
            if last_trading_date != today_str:
                last_trading_date = today_str
                DAILY_TRADES.clear()
                for pos in positions:
                    if pos.consecutive_entry_failures > 0:
                        log(f"  [RESET] {pos.ticker_a}/{pos.ticker_b} entry failures reset (was {pos.consecutive_entry_failures})")
                        pos.consecutive_entry_failures = 0

            # End-of-day: force exit all positions at 3:45 PM ET to avoid overnight risk
            if now_et.time() >= EOD_CLOSE_TIME:
                eod_exits = 0
                for pos in positions:
                    if pos.signal != 0:
                        label = f"{pos.ticker_a}/{pos.ticker_b}"
                        z = 0.0  # z-score irrelevant for EOD forced close
                        # Save entry prices before _force_exit resets shares
                        saved_entry_price_a = pos.entry_price_a
                        saved_entry_price_b = pos.entry_price_b
                        saved_signal = pos.signal
                        exit_shares_a = pos.entry_shares_a
                        exit_shares_b = pos.entry_shares_b

                        action = pos._force_exit(z, "EOD_CLOSE")
                        action.update({
                            "pair": label,
                            "hedge_ratio": pos.hedge_ratio,
                            "z_score": z,
                            "timestamp": now,
                            "sector": pos.sector,
                        })
                        # Add current IEX prices for limit orders
                        if pos.ticker_a in prices.columns and pos.ticker_b in prices.columns:
                            action["price_a"] = prices[pos.ticker_a].iloc[-1]
                            action["price_b"] = prices[pos.ticker_b].iloc[-1]

                        trade_result = {"success": False, "fill_price_a": 0.0, "fill_price_b": 0.0}
                        try:
                            trade_result = execute_trade(action)
                        except Exception as e:
                            log(f"  [ALPACA] EOD close error for {label}: {e}")

                        # Compute P&L using actual fill prices
                        trade_pnl = None
                        if saved_entry_price_a > 0:
                            fill_a = trade_result.get("fill_price_a", 0.0) or action.get("price_a", 0.0)
                            fill_b = trade_result.get("fill_price_b", 0.0) or action.get("price_b", 0.0)
                            if saved_signal == 1:  # was long A, short B
                                trade_pnl = (fill_a - saved_entry_price_a) * exit_shares_a \
                                    - (fill_b - saved_entry_price_b) * exit_shares_b
                            else:  # was short A, long B
                                trade_pnl = -(fill_a - saved_entry_price_a) * exit_shares_a \
                                    + (fill_b - saved_entry_price_b) * exit_shares_b
                            entry_cost = saved_entry_price_a * exit_shares_a \
                                + saved_entry_price_b * exit_shares_b
                            action["entry_cost"] = entry_cost
                            pnl_emoji = "win" if trade_pnl >= 0 else "loss"
                            log(f"  [P&L] {label}: {pnl_emoji} (${trade_pnl:+,.2f}) [EOD CLOSE]")
                            if trade_pnl < 0:
                                pos.consecutive_losses += 1
                            else:
                                pos.consecutive_losses = 0
                            pos.entry_price_a = 0.0
                            pos.entry_price_b = 0.0
                            save_pair_pnl(positions)

                        # Track for daily recap
                        DAILY_TRADES.append({
                            "pair": label,
                            "action": "EXIT",
                            "exit_reason": "EOD_CLOSE",
                            "pnl": trade_pnl,
                            "entry_cost": action.get("entry_cost", 0),
                            "fill_price_a": trade_result.get("fill_price_a", 0),
                            "fill_price_b": trade_result.get("fill_price_b", 0),
                            "signal_price_a": action.get("price_a", 0),
                            "signal_price_b": action.get("price_b", 0),
                        })

                        log(f"  >>> EOD CLOSE: {label}")
                        alert_trade(action, 0, pnl=trade_pnl)
                        eod_exits += 1
                if eod_exits > 0:
                    log(f"  [EOD] Force-exited {eod_exits} positions before market close")
                    save_position_state(positions)
                # Sleep until next market open
                sleep_secs = _seconds_until_market_open()
                hours = sleep_secs // 3600
                mins = (sleep_secs % 3600) // 60
                log(f"  EOD — sleeping {hours}h {mins}m until next open")
                save_position_state(positions)
                time.sleep(sleep_secs)
                consecutive_errors = 0
                continue

            # Check if we're in the opening cooldown period
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            cooldown_end = market_open + pd.Timedelta(minutes=OPEN_COOLDOWN_MINUTES)
            in_open_cooldown = market_open <= now_et < cooldown_end
            if in_open_cooldown:
                log(f"  Opening cooldown active until {cooldown_end.strftime('%-I:%M %p')} ET — no new entries")

            # Block new entries after 3:30 PM ET (too close to EOD close)
            in_eod_no_entry = now_et.time() >= EOD_NO_ENTRY_TIME

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

                # Update trailing stop tracking for active positions
                if pos.signal != 0:
                    pos.update_trailing_stop(price_a, price_b)

                # Vol-adjusted exposure per leg
                exposure = compute_vol_adjusted_exposure(spread, all_spread_vols)
                # Scale down on loss streak
                if pos.consecutive_losses >= LOSS_STREAK_CUTOFF:
                    exposure *= LOSS_STREAK_SCALE
                pos.allocated_exposure = exposure

                shares_a = compute_shares(price_a, exposure)
                shares_b = compute_shares(price_b, exposure)

                # During opening cooldown or EOD wind-down, skip flat positions (no new entries)
                # but still allow exits for active positions
                if (in_open_cooldown or in_eod_no_entry) and pos.signal == 0:
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

                    # Map shares correctly to long/short based on action type
                    if action["action"] == "ENTER_LONG_SPREAD":
                        # long=A, short=B
                        action["shares_long"] = shares_a
                        action["shares_short"] = shares_b
                    elif action["action"] == "ENTER_SHORT_SPREAD":
                        # long=B, short=A — must swap so execute_trade gets correct counts
                        action["shares_long"] = shares_b
                        action["shares_short"] = shares_a

                    # Execute paper trade via Alpaca
                    trade_result = {"success": False, "fill_price_a": 0.0, "fill_price_b": 0.0}
                    try:
                        trade_result = execute_trade(action)
                    except Exception as e:
                        log(f"  [ALPACA] Trade execution error: {e}")

                    trade_ok = trade_result.get("success", False)

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

                    # Successful entry — store actual fill prices and reset failure counter
                    if action["action"] != "EXIT":
                        pos.consecutive_entry_failures = 0
                        # Use actual fill prices if available, otherwise keep IEX prices
                        fill_a = trade_result.get("fill_price_a", 0.0)
                        fill_b = trade_result.get("fill_price_b", 0.0)
                        if fill_a > 0:
                            pos.entry_price_a = fill_a
                        if fill_b > 0:
                            pos.entry_price_b = fill_b

                    # Track P&L on exits using actual fill prices
                    trade_pnl = None
                    if action["action"] == "EXIT" and pos.entry_price_a > 0:
                        # Use shares from action dict (pos fields already reset by _force_exit)
                        exit_shares_a = action.get("exit_shares_a", 0)
                        exit_shares_b = action.get("exit_shares_b", 0)
                        # Use actual fill prices from Alpaca, fall back to IEX quotes
                        exit_fill_a = trade_result.get("fill_price_a", 0.0) or price_a
                        exit_fill_b = trade_result.get("fill_price_b", 0.0) or price_b
                        if action.get("signal") == 1:  # was long A, short B
                            trade_pnl = (exit_fill_a - pos.entry_price_a) * exit_shares_a \
                                - (exit_fill_b - pos.entry_price_b) * exit_shares_b
                        else:  # was short A, long B
                            trade_pnl = -(exit_fill_a - pos.entry_price_a) * exit_shares_a \
                                + (exit_fill_b - pos.entry_price_b) * exit_shares_b
                        # Store entry cost for % calculation
                        entry_cost = pos.entry_price_a * exit_shares_a \
                            + pos.entry_price_b * exit_shares_b
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

                    # Track for daily recap
                    DAILY_TRADES.append({
                        "pair": action.get("pair", label),
                        "action": action["action"],
                        "exit_reason": action.get("exit_reason", ""),
                        "pnl": trade_pnl,
                        "entry_cost": action.get("entry_cost", 0),
                        "fill_price_a": trade_result.get("fill_price_a", 0),
                        "fill_price_b": trade_result.get("fill_price_b", 0),
                        "signal_price_a": price_a,
                        "signal_price_b": price_b,
                    })

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

            # Periodic Telegram alerts
            send_morning_health_check(positions)
            send_hourly_pulse(positions, z_scores, latest_prices)
            send_daily_summary(positions, z_scores, latest_prices)
            send_weekly_report(positions)

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
