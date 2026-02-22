#!/usr/bin/env python3
"""
2-month backtest using the live trader's exact logic with 5-min bars.
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


def fetch_data(tickers, days=70):
    """Fetch 5-min bars in weekly batches to avoid API limits."""
    end = datetime.now() - timedelta(minutes=16)
    start = end - timedelta(days=days)

    print(f"Fetching 5-min data from {start.date()} to {end.date()}...")

    all_frames = []
    chunk_start = start
    chunk_days = 7  # fetch 1 week at a time

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        try:
            request = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=chunk_start,
                end=chunk_end,
                feed=DataFeed.SIP,
            )
            bars = _get_data_client().get_stock_bars(request)
            bar_df = bars.df
            if not bar_df.empty:
                all_frames.append(bar_df)
                print(f"  {chunk_start.date()} to {chunk_end.date()}: {len(bar_df)} bars")
            else:
                print(f"  {chunk_start.date()} to {chunk_end.date()}: no data")
        except Exception as e:
            print(f"  {chunk_start.date()} to {chunk_end.date()}: ERROR - {e}")

        chunk_start = chunk_end

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    combined = combined.reset_index()
    pivot = combined.pivot_table(index="timestamp", columns="symbol", values="close")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index().ffill()
    print(f"\nTotal: {len(pivot)} bars across {len(pivot.columns)} tickers")
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

    # Fetch data (70 calendar days ≈ 2 months)
    prices = fetch_data(all_tickers, days=70)
    if prices.empty:
        print("No data fetched!")
        sys.exit(1)

    # Filter to market hours only (9:30-16:00 ET)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        prices.index = prices.index.tz_convert("US/Eastern")
    prices = prices.between_time("09:30", "16:00")

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
    daily_pnl = {}

    # Simulate bar by bar
    for bar_idx in range(ZSCORE_LOOKBACK, len(prices)):
        window = prices.iloc[:bar_idx + 1]
        ts = prices.index[bar_idx]
        day = ts.normalize()

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
                    if ps.signal == 1:
                        pnl_a = (price_a - ps.entry_price_a) * ps.entry_shares_a
                        pnl_b = (ps.entry_price_b - price_b) * ps.entry_shares_b
                    else:
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

                    day_key = str(day.date())
                    daily_pnl[day_key] = daily_pnl.get(day_key, 0) + total_pnl

                    ps.signal = 0
                    ps.entry_z = None
                    ps.bars_held = 0
                    if exit_reason in ("HARD_STOP", "TIME_STOP"):
                        ps.cooldown = COOLDOWN_BARS

        equity_curve.append(TOTAL_CAPITAL + running_pnl)

    # ---- Results ----
    print("=" * 60)
    print("  2-MONTH BACKTEST RESULTS")
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

    # Annualized metrics
    n_trading_days = len(trading_days)
    daily_return = total_pnl / TOTAL_CAPITAL / n_trading_days
    ann_return = daily_return * 252 * 100
    daily_returns = pd.Series(equity_curve).pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
    sharpe = (daily_return * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    print(f"\n  Period:           {trading_days[0].date()} to {trading_days[-1].date()} ({n_trading_days} trading days)")
    print(f"  Total trades:     {n_trades}")
    print(f"  Winners:          {len(winners)} ({win_rate:.1f}%)")
    print(f"  Losers:           {len(losers)} ({100 - win_rate:.1f}%)")
    print(f"  Total P&L:        ${total_pnl:,.2f}")
    print(f"  Return:           {total_pnl / TOTAL_CAPITAL * 100:.2f}%")
    print(f"  Ann. return:      {ann_return:.1f}%")
    print(f"  Ann. volatility:  {ann_vol:.1f}%")
    print(f"  Sharpe ratio:     {sharpe:.2f}")
    print(f"  Avg P&L/trade:    ${trades_df['pnl'].mean():,.2f}")
    print(f"  Median P&L/trade: ${trades_df['pnl'].median():,.2f}")
    print(f"  Best trade:       ${trades_df['pnl'].max():,.2f}")
    print(f"  Worst trade:      ${trades_df['pnl'].min():,.2f}")
    print(f"  Final equity:     ${equity_curve[-1]:,.2f}")

    # Max drawdown
    eq = pd.Series(equity_curve)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    print(f"  Max drawdown:     {dd.min() * 100:.2f}%")

    # By exit reason
    print(f"\n  Exit breakdown:")
    for reason, group in trades_df.groupby("exit_reason"):
        avg = group['pnl'].mean()
        print(f"    {reason:12s}: {len(group):4d} trades, P&L ${group['pnl'].sum():>10,.2f}, avg ${avg:>8,.2f}")

    # By sector
    print(f"\n  Sector breakdown:")
    for sector, group in trades_df.groupby("sector"):
        print(f"    {sector:28s}: {len(group):4d} trades, P&L ${group['pnl'].sum():>10,.2f}, win rate {len(group[group['pnl']>0])/len(group)*100:.0f}%")

    # Top/bottom pairs
    pair_pnl = trades_df.groupby("pair")["pnl"].agg(["sum", "count", "mean"])
    pair_pnl = pair_pnl.sort_values("sum", ascending=False)
    print(f"\n  Top 5 pairs:")
    for pair, row in pair_pnl.head(5).iterrows():
        print(f"    {pair:12s}: ${row['sum']:>10,.2f} ({int(row['count'])} trades, avg ${row['mean']:>8,.2f})")
    print(f"\n  Bottom 5 pairs:")
    for pair, row in pair_pnl.tail(5).iterrows():
        print(f"    {pair:12s}: ${row['sum']:>10,.2f} ({int(row['count'])} trades, avg ${row['mean']:>8,.2f})")

    # Weekly P&L
    print(f"\n  Weekly P&L:")
    trades_df["week"] = pd.to_datetime(trades_df["exit_time"]).dt.isocalendar().week
    for week, group in trades_df.groupby("week"):
        print(f"    Week {int(week):2d}: ${group['pnl'].sum():>10,.2f} ({len(group)} trades)")

    print(f"\n  {'='*60}")

    # Save results
    trades_df.to_csv("backtest_2month_trades.csv", index=False)
    pd.DataFrame({"equity": equity_curve}).to_csv("backtest_2month_equity.csv", index=False)
    print(f"\n  Saved: backtest_2month_trades.csv, backtest_2month_equity.csv")


if __name__ == "__main__":
    run_backtest()
