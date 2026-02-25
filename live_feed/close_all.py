#!/usr/bin/env python3
"""
Close all Alpaca positions and reset local state.

Usage:
    python -m live_feed.close_all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_feed.alpaca_client import get_positions, _close_position, get_account_info


def close_all():
    positions = get_positions()
    if not positions:
        print("No open positions.")
        return

    print(f"Closing {len(positions)} positions...")
    for p in positions:
        symbol = p["symbol"]
        qty = p["qty"]
        pl = float(p["unrealized_pl"])
        print(f"  Closing {symbol} (qty={qty}, unrealized=${pl:+,.2f})...", end=" ")
        if _close_position(symbol):
            print("OK")
        else:
            print("FAILED")

    # Clear saved state
    state_file = Path(__file__).resolve().parent / "position_state.json"
    if state_file.exists():
        state_file.unlink()
        print("Cleared position_state.json")

    pnl_file = Path(__file__).resolve().parent / "pair_pnl.csv"
    if pnl_file.exists():
        pnl_file.unlink()
        print("Cleared pair_pnl.csv")

    account = get_account_info()
    print(f"\nAccount — equity: ${float(account['equity']):,.2f}, "
          f"cash: ${float(account['cash']):,.2f}")


if __name__ == "__main__":
    close_all()
