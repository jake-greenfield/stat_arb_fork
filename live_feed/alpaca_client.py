#!/usr/bin/env python3
"""
Alpaca client for paper trading and market data.

Provides helpers to:
  - Fetch 5-min bar data (replacing yfinance)
  - Execute paper trades (market orders)
  - Query account info and positions
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

_trading_client = None
_data_client = None


def _get_trading_client() -> TradingClient:
    global _trading_client
    if _trading_client is None:
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        _trading_client = TradingClient(api_key, secret_key, paper=True)
    return _trading_client


def _get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        _data_client = StockHistoricalDataClient(api_key, secret_key)
    return _data_client


def fetch_5min_data_alpaca(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch recent 1-min bars for given tickers via Alpaca.

    Returns a DataFrame with DatetimeIndex and one column per ticker (close prices),
    matching the format previously returned by yfinance.
    """
    end = datetime.now() - timedelta(minutes=16)  # 15-min delay for free SIP access
    start = end - timedelta(days=5)

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.SIP,
    )

    bars = _get_data_client().get_stock_bars(request)
    bar_df = bars.df  # MultiIndex: (symbol, timestamp)

    if bar_df.empty:
        return pd.DataFrame()

    # Pivot to get one column per ticker with close prices
    bar_df = bar_df.reset_index()
    pivot = bar_df.pivot_table(index="timestamp", columns="symbol", values="close")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    return pivot.ffill().dropna(how="all")


def fetch_latest_prices_alpaca(tickers: list[str], batch_size: int = 100) -> dict:
    """
    Fetch the latest price for each ticker via Alpaca 1-min bars.

    Returns dict of {ticker: price}.
    """
    all_prices = {}
    end = datetime.now()
    start = end - timedelta(minutes=30)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            bars = _get_data_client().get_stock_bars(request)
            bar_df = bars.df

            if bar_df.empty:
                continue

            bar_df = bar_df.reset_index()
            for ticker in batch:
                ticker_data = bar_df[bar_df["symbol"] == ticker]
                if not ticker_data.empty:
                    all_prices[ticker] = ticker_data["close"].iloc[-1]
        except Exception as e:
            print(f"  Warning: batch {i}-{i+len(batch)} failed: {e}", flush=True)

    return all_prices


def execute_trade(action: dict) -> bool:
    """
    Execute a paper trade via Alpaca based on the action dict from PairPosition.

    Supported actions:
      - ENTER_LONG_SPREAD / ENTER_SHORT_SPREAD: buy 'long' ticker, sell short 'short' ticker
      - EXIT: close both legs of the pair

    Returns True if ALL orders succeeded, False if any failed.
    """
    act = action.get("action", "")
    pair = action.get("pair", "")

    if act in ("ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD"):
        long_ticker = action["long"]
        short_ticker = action["short"]
        shares_long = action.get("shares_long", action.get("shares_a", 0))
        shares_short = action.get("shares_short", action.get("shares_b", 0))

        ok_long = True
        ok_short = True
        if shares_long > 0:
            ok_long = _submit_order(long_ticker, shares_long, OrderSide.BUY)
        if shares_short > 0:
            ok_short = _submit_order(short_ticker, shares_short, OrderSide.SELL)

        success = ok_long and ok_short
        if success:
            print(f"  [ALPACA] Entered {act}: BUY {shares_long} {long_ticker}, "
                  f"SELL {shares_short} {short_ticker}", flush=True)
        else:
            print(f"  [ALPACA] FAILED to enter {act}: BUY {long_ticker} ({'OK' if ok_long else 'FAIL'}), "
                  f"SELL {short_ticker} ({'OK' if ok_short else 'FAIL'})", flush=True)
        return success

    elif act == "EXIT":
        # Exit by reversing the exact entry shares (not closing entire position)
        tickers = pair.split("/")
        if len(tickers) == 2:
            ticker_a, ticker_b = tickers
            shares_a = action.get("exit_shares_a", 0)
            shares_b = action.get("exit_shares_b", 0)
            signal = action.get("signal", 0)

            if shares_a > 0 and shares_b > 0 and signal != 0:
                if signal == 1:
                    # Was long A, short B → sell A, buy-to-cover B
                    ok_a = _submit_order(ticker_a, shares_a, OrderSide.SELL)
                    ok_b = _submit_order(ticker_b, shares_b, OrderSide.BUY)
                else:
                    # Was short A, long B → buy-to-cover A, sell B
                    ok_a = _submit_order(ticker_a, shares_a, OrderSide.BUY)
                    ok_b = _submit_order(ticker_b, shares_b, OrderSide.SELL)
                print(f"  [ALPACA] Exited pair {pair}: reversed {shares_a} {ticker_a}, "
                      f"{shares_b} {ticker_b}", flush=True)
                return ok_a and ok_b
            else:
                # Fallback: close entire position (legacy behavior)
                for t in tickers:
                    for attempt in range(3):
                        if _close_position(t):
                            break
                        time.sleep(1)
                print(f"  [ALPACA] Exited pair {pair}: closed positions in "
                      f"{ticker_a} and {ticker_b}", flush=True)
                return True
        return False


def _submit_order(ticker: str, qty: int, side: OrderSide) -> bool:
    """Submit a market order. Returns True if order succeeded, False otherwise."""
    try:
        order = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        _get_trading_client().submit_order(order)
        return True
    except Exception as e:
        print(f"  [ALPACA] Order failed ({side.name} {qty} {ticker}): {e}", flush=True)
        return False


def _close_position(ticker: str) -> bool:
    """Close any existing position in the given ticker. Returns True if closed or no position."""
    try:
        _get_trading_client().close_position(ticker)
        return True
    except Exception as e:
        if "position does not exist" in str(e).lower():
            return True  # already closed
        print(f"  [ALPACA] Close position failed ({ticker}): {e}", flush=True)
        return False


def get_account_info() -> dict:
    """Fetch paper account balance and key info."""
    account = _get_trading_client().get_account()
    return {
        "equity": account.equity,
        "cash": account.cash,
        "buying_power": account.buying_power,
        "portfolio_value": account.portfolio_value,
        "status": account.status,
    }


def get_all_tradeable_tickers() -> list[str]:
    """Get all active, tradeable US equity tickers from Alpaca (NASDAQ + NYSE)."""
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    request = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )
    assets = _get_trading_client().get_all_assets(request)
    tickers = [
        a.symbol for a in assets
        if a.tradable and a.exchange in ("NASDAQ", "NYSE")
        and a.shortable  # need shortable for stat arb
        and not a.symbol.isdigit()  # skip weird tickers
        and "." not in a.symbol  # skip preferred shares etc.
        and len(a.symbol) <= 5  # skip long tickers (warrants, units)
    ]
    return sorted(tickers)


def fetch_5min_data_alpaca_batch(
    tickers: list[str], days: int = 5, batch_size: int = 500,
) -> pd.DataFrame:
    """
    Fetch 5-min bars for a large set of tickers in batches.
    Returns DataFrame with DatetimeIndex x ticker columns (close prices).
    """
    end = datetime.now() - timedelta(minutes=16)  # 15-min delay for free tier
    start = end - timedelta(days=days)
    all_frames = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start,
                end=end,
                feed=DataFeed.SIP,
            )
            bars = _get_data_client().get_stock_bars(request)
            bar_df = bars.df
            if not bar_df.empty:
                all_frames.append(bar_df)
            print(f"  Fetched batch {i}-{i+len(batch)} "
                  f"({len(bar_df) if not bar_df.empty else 0} bars)", flush=True)
        except Exception as e:
            print(f"  Warning: batch {i}-{i+len(batch)} failed: {e}", flush=True)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    combined = combined.reset_index()
    pivot = combined.pivot_table(index="timestamp", columns="symbol", values="close")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    return pivot


def cancel_all_orders() -> int:
    """Cancel all open orders. Returns number of orders cancelled."""
    try:
        client = _get_trading_client()
        client.cancel_orders()
        # Brief pause for cancellations to process
        time.sleep(1)
        orders = client.get_orders()
        # If there are still open orders, they're being cancelled
        return len(orders)
    except Exception as e:
        print(f"  [ALPACA] Cancel all orders failed: {e}", flush=True)
        return -1


def get_positions() -> list[dict]:
    """Fetch all current Alpaca positions."""
    positions = _get_trading_client().get_all_positions()
    return [
        {
            "symbol": p.symbol,
            "qty": p.qty,
            "side": p.side,
            "market_value": p.market_value,
            "unrealized_pl": p.unrealized_pl,
        }
        for p in positions
    ]
