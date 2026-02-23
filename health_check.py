#!/usr/bin/env python3
"""
Health check for the stat-arb algo.
Run anytime to catch issues before they become losses.

Usage:
    python health_check.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_feed.alpaca_client import get_account_info, get_positions

TOTAL_CAPITAL = 100_000
MAX_EXPOSURE_PER_PAIR = 5_000
POSITIONS_FILE = Path("live_feed/positions.csv")

errors = []
warnings = []


def check_account():
    """Check account health."""
    print("=" * 60)
    print("  STAT-ARB HEALTH CHECK")
    print("=" * 60)

    info = get_account_info()
    equity = float(info["equity"])
    cash = float(info["cash"])

    print(f"\n  Account equity:  ${equity:,.2f}")
    print(f"  Cash:            ${cash:,.2f}")
    print(f"  P&L from start:  ${equity - TOTAL_CAPITAL:,.2f} ({(equity/TOTAL_CAPITAL - 1)*100:+.2f}%)")

    if equity < TOTAL_CAPITAL * 0.90:
        errors.append(f"Account down >10% (${equity:,.2f})")
    elif equity < TOTAL_CAPITAL * 0.95:
        warnings.append(f"Account down >5% (${equity:,.2f})")


def check_position_sync():
    """Check if local state matches Alpaca positions."""
    print(f"\n  --- Position Sync ---")

    alpaca_positions = get_positions()
    alpaca_map = {}
    for p in alpaca_positions:
        alpaca_map[p["symbol"]] = {
            "qty": float(p["qty"]),
            "market_value": float(p["market_value"]),
            "unrealized_pl": float(p["unrealized_pl"]),
        }

    if not POSITIONS_FILE.exists():
        errors.append("positions.csv not found")
        return

    local_df = pd.read_csv(POSITIONS_FILE)
    active_local = local_df[local_df["signal"] != 0]

    # Build expected Alpaca positions from local state
    expected_tickers = set()
    for _, row in active_local.iterrows():
        a, b = row["pair"].split("/")
        expected_tickers.add(a)
        expected_tickers.add(b)

    alpaca_tickers = set(alpaca_map.keys())

    # Check for orphaned Alpaca positions (in Alpaca but not in local state)
    orphaned = alpaca_tickers - expected_tickers
    if orphaned:
        errors.append(f"Orphaned Alpaca positions (no local pair): {', '.join(sorted(orphaned))}")
        for t in sorted(orphaned):
            p = alpaca_map[t]
            print(f"    ORPHANED: {t} qty={p['qty']:.0f} P&L=${p['unrealized_pl']:.2f}")

    # Check for missing Alpaca positions (in local state but not in Alpaca)
    missing = expected_tickers - alpaca_tickers
    if missing:
        errors.append(f"Missing Alpaca positions (local says active): {', '.join(sorted(missing))}")

    print(f"  Local active pairs:    {len(active_local)}")
    print(f"  Alpaca positions:      {len(alpaca_positions)}")
    print(f"  Orphaned positions:    {len(orphaned)}")
    print(f"  Missing positions:     {len(missing)}")


def check_exposure():
    """Check for excessive exposure or imbalance."""
    print(f"\n  --- Exposure ---")

    alpaca_positions = get_positions()
    total_long = 0
    total_short = 0

    for p in alpaca_positions:
        mv = float(p["market_value"])
        if mv > 0:
            total_long += mv
        else:
            total_short += abs(mv)

    gross = total_long + total_short
    net = total_long - total_short
    net_pct = abs(net) / gross * 100 if gross > 0 else 0

    print(f"  Total long:      ${total_long:,.2f}")
    print(f"  Total short:     ${total_short:,.2f}")
    print(f"  Gross exposure:  ${gross:,.2f}")
    print(f"  Net exposure:    ${net:,.2f} ({net_pct:.1f}% of gross)")

    if net_pct > 20:
        errors.append(f"Net exposure is {net_pct:.0f}% of gross — not market neutral")
    elif net_pct > 10:
        warnings.append(f"Net exposure is {net_pct:.0f}% of gross — drifting from neutral")

    # Check for oversized positions
    for p in alpaca_positions:
        mv = abs(float(p["market_value"]))
        if mv > MAX_EXPOSURE_PER_PAIR * 2:
            errors.append(f"{p['symbol']} position ${mv:,.0f} is >2x max exposure (${MAX_EXPOSURE_PER_PAIR})")


def check_shared_tickers():
    """Check for pairs that share tickers (risk of exit interference)."""
    print(f"\n  --- Shared Tickers ---")

    if not POSITIONS_FILE.exists():
        return

    local_df = pd.read_csv(POSITIONS_FILE)
    active = local_df[local_df["signal"] != 0]

    ticker_pairs = {}
    for _, row in active.iterrows():
        a, b = row["pair"].split("/")
        for t in [a, b]:
            if t not in ticker_pairs:
                ticker_pairs[t] = []
            ticker_pairs[t].append(row["pair"])

    shared = {t: pairs for t, pairs in ticker_pairs.items() if len(pairs) > 1}
    if shared:
        for t, pairs in shared.items():
            warnings.append(f"{t} is active in {len(pairs)} pairs: {', '.join(pairs)}")
            print(f"    {t} shared across: {', '.join(pairs)}")
    else:
        print("  No shared tickers among active pairs")


def check_unrealized_pnl():
    """Flag positions with large unrealized losses."""
    print(f"\n  --- Position P&L ---")

    alpaca_positions = get_positions()
    for p in alpaca_positions:
        pl = float(p["unrealized_pl"])
        sym = p["symbol"]
        if pl < -200:
            errors.append(f"{sym} unrealized P&L ${pl:.2f}")
        elif pl < -100:
            warnings.append(f"{sym} unrealized P&L ${pl:.2f}")
        print(f"    {sym:6s} P&L: ${pl:>8.2f}")


def print_summary():
    print(f"\n{'='*60}")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    [!] {e}")
    if warnings:
        print(f"  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    [~] {w}")
    if not errors and not warnings:
        print("  ALL CHECKS PASSED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    check_account()
    check_position_sync()
    check_exposure()
    check_shared_tickers()
    check_unrealized_pnl()
    print_summary()
