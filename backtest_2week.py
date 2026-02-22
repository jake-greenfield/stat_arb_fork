#!/usr/bin/env python3
"""
2-week backtest using the live trader's exact logic with 5-min bars.
Simulates continuous operation during all market hours.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from live_feed.alpaca_client import _get_data_client
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# ---- Strategy params (same as live trader) ----
ZSCORE_LOOKBACK = 60
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
ZSCORE_HARD_STOP = 3.25
TIME_STOP_BARS = 390       # 5 trading days
COOLDOWN_BARS = 78          # 1 trading day
TOTAL_CAPITAL = 100_000
MAX_EXPOSURE_PER_PAIR = 5_000


def fetch_2week_data(tickers, days=18):
    """Fetch 5-min bars for past 2+ weeks."""
    end = datetime.now() - timedelta(minutes=16)
    start = end - timedelta(days=days)

    print(f"Fetching 5-min data from {start.date()} to {end.date()}...")
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.SIP,
    )
    bars = _get_data_client().get_stock_bars(request)
    bar_df = bars.df
    if bar_df.empty:
        return pd.DataFrame()

    bar_df = bar_df.reset_index()
    pivot = bar_df.pivot_table(index="timestamp", columns="symbol", values="close")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index().ffill()
    print(f"Got {len(pivot)} bars across {len(pivot.columns)} tickers")
    return pivot


def compute_zscore(spread, lookback=ZSCORE_LOOKBACK):
    if len(spread) < 10:
        return 0.0
    lb = min(lookback, len(spread))
    recent = spread.iloc[-lb:]
    mean = recent.mean()
    std = recent.std()
    if std < 1e-8:
        return 0.0
    return (spread.iloc[-1] - mean) / std


def run_backtest():
    # Load pairs
    pairs_csv = Path("live_feed/active_pairs.csv")
    pairs = pd.read_csv(pairs_csv).to_dict("records")
    print(f"Loaded {len(pairs)} pairs\n")

    # Get unique tickers
    all_tickers = sorted(set(
        t for p in pairs for t in [p["ticker_a"], p["ticker_b"]]
    ))

    # Fetch data
    prices = fetch_2week_data(all_tickers)
    if prices.empty:
        print("No data fetched!")
        sys.exit(1)

    # Filter to market hours only (9:30-16:00 ET)
    prices.index = prices.index.tz_convert("US/Eastern") if prices.index.tz else prices.index.tz_localize("UTC").tz_convert("US/Eastern")
    prices = prices.between_time("09:30", "16:00")
    # Only keep last 2 trading weeks (~10 trading days)
    trading_days = prices.index.normalize().unique()
    if len(trading_days) > 10:
        cutoff = trading_days[-10]
        prices = prices[prices.index >= cutoff]
    trading_days = prices.index.normalize().unique()
    print(f"Trading days: {len(trading_days)} ({trading_days[0].date()} to {trading_days[-1].date()})")
    print(f"Total bars: {len(prices)}\n")

    # Initialize pair state
    class PairState:
        def __init__(self, pair):
            self.ticker_a = pair["ticker_a"]
            self.ticker_b = pair["ticker_b"]
            self.hedge_ratio = pair["hedge_ratio"]
            self.sector = pair["sector"]
            self.signal = 0
            self.entry_z = None
            self.entry_price_a = None
            self.entry_price_b = None
            self.entry_shares_a = 0
            self.entry_shares_b = 0
            self.bars_held = 0
            self.cooldown = 0

    pair_states = [PairState(p) for p in pairs]

    # Track P&L
    trades = []
    equity_curve = [TOTAL_CAPITAL]
    running_pnl = 0.0

    # Simulate bar by bar
    for bar_idx in range(ZSCORE_LOOKBACK, len(prices)):
        window = prices.iloc[:bar_idx + 1]
        ts = prices.index[bar_idx]

        for ps in pair_states:
            if ps.ticker_a not in prices.columns or ps.ticker_b not in prices.columns:
                continue

            spread = window[ps.ticker_a] - ps.hedge_ratio * window[ps.ticker_b]
            z = compute_zscore(spread)

            price_a = prices[ps.ticker_a].iloc[bar_idx]
            price_b = prices[ps.ticker_b].iloc[bar_idx]

            if pd.isna(price_a) or pd.isna(price_b):
                continue

            if ps.signal == 0:
                if ps.cooldown > 0:
                    ps.cooldown -= 1
                    continue

                shares_a = int(MAX_EXPOSURE_PER_PAIR / price_a) if price_a > 0 else 0
                shares_b = int(MAX_EXPOSURE_PER_PAIR / price_b) if price_b > 0 else 0

                if z <= -ZSCORE_ENTRY:
                    ps.signal = 1
                    ps.entry_z = z
                    ps.entry_price_a = price_a
                    ps.entry_price_b = price_b
                    ps.entry_shares_a = shares_a
                    ps.entry_shares_b = shares_b
                    ps.bars_held = 0
                elif z >= ZSCORE_ENTRY:
                    ps.signal = -1
                    ps.entry_z = z
                    ps.entry_price_a = price_a
                    ps.entry_price_b = price_b
                    ps.entry_shares_a = shares_a
                    ps.entry_shares_b = shares_b
                    ps.bars_held = 0
            else:
                ps.bars_held += 1
                exit_reason = None

                if abs(z) >= ZSCORE_HARD_STOP:
                    exit_reason = "HARD_STOP"
                elif ps.bars_held >= TIME_STOP_BARS:
                    exit_reason = "TIME_STOP"
                elif abs(z) <= ZSCORE_EXIT:
                    exit_reason = "PROFIT_EXIT"

                if exit_reason:
                    # Calculate P&L
                    if ps.signal == 1:  # long A, short B
                        pnl_a = (price_a - ps.entry_price_a) * ps.entry_shares_a
                        pnl_b = (ps.entry_price_b - price_b) * ps.entry_shares_b
                    else:  # short A, long B
                        pnl_a = (ps.entry_price_a - price_a) * ps.entry_shares_a
                        pnl_b = (price_b - ps.entry_price_b) * ps.entry_shares_b

                    total_pnl = pnl_a + pnl_b
                    running_pnl += total_pnl

                    trades.append({
                        "pair": f"{ps.ticker_a}/{ps.ticker_b}",
                        "sector": ps.sector,
                        "direction": "LONG" if ps.signal == 1 else "SHORT",
                        "entry_z": ps.entry_z,
                        "exit_z": z,
                        "bars_held": ps.bars_held,
                        "exit_reason": exit_reason,
                        "pnl": total_pnl,
                        "exit_time": str(ts),
                    })

                    ps.signal = 0
                    ps.entry_z = None
                    ps.bars_held = 0
                    if exit_reason in ("HARD_STOP", "TIME_STOP"):
                        ps.cooldown = COOLDOWN_BARS

        equity_curve.append(TOTAL_CAPITAL + running_pnl)

    # ---- Results ----
    print("=" * 60)
    print("  2-WEEK BACKTEST RESULTS")
    print("=" * 60)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("\nNo trades executed during this period.")
        return

    total_pnl = trades_df["pnl"].sum()
    n_trades = len(trades_df)
    winners = trades_df[trades_df["pnl"] > 0]
    losers = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0

    print(f"\n  Total trades:     {n_trades}")
    print(f"  Winners:          {len(winners)} ({win_rate:.1f}%)")
    print(f"  Losers:           {len(losers)} ({100 - win_rate:.1f}%)")
    print(f"  Total P&L:        ${total_pnl:,.2f}")
    print(f"  Avg P&L/trade:    ${trades_df['pnl'].mean():,.2f}")
    print(f"  Best trade:       ${trades_df['pnl'].max():,.2f}")
    print(f"  Worst trade:      ${trades_df['pnl'].min():,.2f}")
    print(f"  Final equity:     ${equity_curve[-1]:,.2f}")
    print(f"  Return:           {(equity_curve[-1] / TOTAL_CAPITAL - 1) * 100:.2f}%")

    # Max drawdown
    eq = pd.Series(equity_curve)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    print(f"  Max drawdown:     {dd.min() * 100:.2f}%")

    # By exit reason
    print(f"\n  Exit breakdown:")
    for reason, group in trades_df.groupby("exit_reason"):
        print(f"    {reason:12s}: {len(group):3d} trades, P&L ${group['pnl'].sum():,.2f}")

    # Individual trades
    print(f"\n  {'Pair':<12s} {'Dir':<6s} {'Entry Z':>8s} {'Exit Z':>8s} {'Bars':>5s} {'Reason':<12s} {'P&L':>10s}")
    print(f"  {'-'*67}")
    for _, t in trades_df.iterrows():
        print(f"  {t['pair']:<12s} {t['direction']:<6s} {t['entry_z']:>+8.2f} {t['exit_z']:>+8.2f} {t['bars_held']:>5d} {t['exit_reason']:<12s} ${t['pnl']:>9,.2f}")

    print(f"\n  {'='*60}")

    # Save results
    trades_df.to_csv("backtest_2week_trades.csv", index=False)
    pd.DataFrame({"equity": equity_curve}).to_csv("backtest_2week_equity.csv", index=False)
    print(f"\n  Saved: backtest_2week_trades.csv, backtest_2week_equity.csv")


if __name__ == "__main__":
    run_backtest()
