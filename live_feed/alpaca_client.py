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
from alpaca.data.enums import DataFeed  # IEX = free real-time, SIP = 15-min delayed on free tier
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

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
    end = datetime.now()
    start = end - timedelta(days=5)

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,
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


def execute_trade(action: dict) -> dict:
    """
    Execute a paper trade via Alpaca based on the action dict from PairPosition.

    Supported actions:
      - ENTER_LONG_SPREAD / ENTER_SHORT_SPREAD: buy 'long' ticker, sell short 'short' ticker
      - EXIT: close both legs of the pair

    Returns dict with:
      - "success": bool
      - "fill_price_a": float (actual fill for ticker_a, 0 if unavailable)
      - "fill_price_b": float (actual fill for ticker_b, 0 if unavailable)
    """
    result = {"success": False, "fill_price_a": 0.0, "fill_price_b": 0.0}
    act = action.get("action", "")
    pair = action.get("pair", "")

    if act in ("ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD"):
        long_ticker = action["long"]
        short_ticker = action["short"]
        shares_long = action.get("shares_long", action.get("shares_a", 0))
        shares_short = action.get("shares_short", action.get("shares_b", 0))
        # Get prices for limit orders
        price_a = action.get("price_a", 0.0)
        price_b = action.get("price_b", 0.0)
        long_price = price_b if act == "ENTER_SHORT_SPREAD" else price_a
        short_price = price_a if act == "ENTER_SHORT_SPREAD" else price_b

        ok_long, fill_long = True, 0.0
        ok_short, fill_short = True, 0.0
        if shares_long > 0:
            ok_long, fill_long = _submit_order(long_ticker, shares_long, OrderSide.BUY, long_price)
        if shares_short > 0:
            ok_short, fill_short = _submit_order(short_ticker, shares_short, OrderSide.SELL, short_price)

        success = ok_long and ok_short
        # Map fills back to ticker_a and ticker_b
        tickers = pair.split("/") if pair else []
        if len(tickers) == 2:
            ticker_a, ticker_b = tickers
            if long_ticker == ticker_a:
                result["fill_price_a"] = fill_long
                result["fill_price_b"] = fill_short
            else:
                result["fill_price_a"] = fill_short
                result["fill_price_b"] = fill_long

        if success:
            print(f"  [ALPACA] Entered {act}: BUY {shares_long} {long_ticker} @ ${fill_long:.2f}, "
                  f"SELL {shares_short} {short_ticker} @ ${fill_short:.2f}", flush=True)
        else:
            print(f"  [ALPACA] FAILED to enter {act}: BUY {long_ticker} ({'OK' if ok_long else 'FAIL'}), "
                  f"SELL {short_ticker} ({'OK' if ok_short else 'FAIL'})", flush=True)
        result["success"] = success
        return result

    elif act == "EXIT":
        # Exit by reversing the exact entry shares (not closing entire position)
        tickers = pair.split("/")
        if len(tickers) == 2:
            ticker_a, ticker_b = tickers
            shares_a = action.get("exit_shares_a", 0)
            shares_b = action.get("exit_shares_b", 0)
            signal = action.get("signal", 0)

            if shares_a > 0 and shares_b > 0 and signal != 0:
                exit_price_a = action.get("price_a", 0.0)
                exit_price_b = action.get("price_b", 0.0)
                if signal == 1:
                    # Was long A, short B → sell A, buy-to-cover B
                    ok_a, fill_a = _submit_order(ticker_a, shares_a, OrderSide.SELL, exit_price_a)
                    ok_b, fill_b = _submit_order(ticker_b, shares_b, OrderSide.BUY, exit_price_b)
                else:
                    # Was short A, long B → buy-to-cover A, sell B
                    ok_a, fill_a = _submit_order(ticker_a, shares_a, OrderSide.BUY, exit_price_a)
                    ok_b, fill_b = _submit_order(ticker_b, shares_b, OrderSide.SELL, exit_price_b)
                print(f"  [ALPACA] Exited pair {pair}: reversed {shares_a} {ticker_a} @ ${fill_a:.2f}, "
                      f"{shares_b} {ticker_b} @ ${fill_b:.2f}", flush=True)
                result["success"] = ok_a and ok_b
                result["fill_price_a"] = fill_a
                result["fill_price_b"] = fill_b
                return result
            else:
                # Fallback: close entire position (legacy behavior)
                for t in tickers:
                    for attempt in range(3):
                        if _close_position(t):
                            break
                        time.sleep(1)
                print(f"  [ALPACA] Exited pair {pair}: closed positions in "
                      f"{ticker_a} and {ticker_b}", flush=True)
                result["success"] = True
                return result
        return result


LIMIT_ORDER_BUFFER_PCT = 0.0005  # 0.05% buffer on limit price
LIMIT_ORDER_TIMEOUT_SECS = 30   # wait this long for limit fill
LIMIT_ORDER_POLL_SECS = 2       # poll interval for fill check


def _submit_order(ticker: str, qty: int, side: OrderSide, price: float = 0.0) -> tuple[bool, float]:
    """
    Submit a limit order with market fallback.

    Returns (success: bool, fill_price: float).
    fill_price is the actual average fill price from Alpaca, or 0.0 if failed.

    1. Submit limit order at current price + 0.05% buffer
    2. Poll for up to 30 seconds for fill
    3. If not filled, cancel and submit market order as fallback
    4. If market order also fails, return (False, 0.0)

    If price is 0 or not provided, falls back to market order immediately.
    """
    if price <= 0:
        return _submit_market_order(ticker, qty, side)

    # Calculate limit price with buffer
    if side == OrderSide.BUY:
        limit_price = round(price * (1 + LIMIT_ORDER_BUFFER_PCT), 2)
    else:
        limit_price = round(price * (1 - LIMIT_ORDER_BUFFER_PCT), 2)

    # Try limit order first
    try:
        order_req = LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
        )
        order = _get_trading_client().submit_order(order_req)
        order_id = order.id

        # Poll for fill
        elapsed = 0
        while elapsed < LIMIT_ORDER_TIMEOUT_SECS:
            time.sleep(LIMIT_ORDER_POLL_SECS)
            elapsed += LIMIT_ORDER_POLL_SECS
            try:
                updated = _get_trading_client().get_order_by_id(order_id)
                if updated.status == OrderStatus.FILLED:
                    fill = float(updated.filled_avg_price) if updated.filled_avg_price else limit_price
                    saved = abs(fill - price) * qty
                    print(f"  [ALPACA] Limit FILLED: {side.name} {qty} {ticker} @ ${fill:.2f} "
                          f"(limit ${limit_price:.2f}, saved ~${saved:.2f})", flush=True)
                    return (True, fill)
                if updated.status in (OrderStatus.CANCELED, OrderStatus.EXPIRED,
                                      OrderStatus.REJECTED):
                    print(f"  [ALPACA] Limit {updated.status.name}: {side.name} {qty} {ticker}, "
                          f"falling back to market", flush=True)
                    return _submit_market_order(ticker, qty, side)
                if updated.status == OrderStatus.PARTIALLY_FILLED:
                    filled_qty = int(float(updated.filled_qty)) if updated.filled_qty else 0
                    remaining = qty - filled_qty
                    partial_fill = float(updated.filled_avg_price) if updated.filled_avg_price else limit_price
                    if elapsed >= LIMIT_ORDER_TIMEOUT_SECS and remaining > 0:
                        # Cancel remaining and fill rest at market
                        try:
                            _get_trading_client().cancel_order_by_id(order_id)
                            time.sleep(1)
                        except Exception:
                            pass
                        print(f"  [ALPACA] Limit partial fill: {filled_qty}/{qty} {ticker}, "
                              f"sending market for remaining {remaining}", flush=True)
                        ok, market_fill = _submit_market_order(ticker, remaining, side)
                        if ok and market_fill > 0:
                            # Weighted average of partial limit fill + market fill
                            avg_fill = (partial_fill * filled_qty + market_fill * remaining) / qty
                            return (True, avg_fill)
                        return (ok, partial_fill)  # at least got partial
            except Exception as e:
                print(f"  [ALPACA] Error checking order status: {e}", flush=True)

        # Timeout — cancel limit order and fall back to market
        try:
            _get_trading_client().cancel_order_by_id(order_id)
            time.sleep(1)
            # Check if it filled during cancellation
            final = _get_trading_client().get_order_by_id(order_id)
            if final.status == OrderStatus.FILLED:
                fill = float(final.filled_avg_price) if final.filled_avg_price else limit_price
                print(f"  [ALPACA] Limit FILLED (during cancel): {side.name} {qty} {ticker} @ ${fill:.2f}", flush=True)
                return (True, fill)
            filled_qty = int(float(final.filled_qty)) if final.filled_qty else 0
            partial_fill = float(final.filled_avg_price) if final.filled_avg_price and filled_qty > 0 else 0.0
            remaining = qty - filled_qty
            if filled_qty > 0:
                print(f"  [ALPACA] Limit partial: {filled_qty}/{qty} {ticker}, "
                      f"market for remaining {remaining}", flush=True)
            else:
                print(f"  [ALPACA] Limit timeout ({LIMIT_ORDER_TIMEOUT_SECS}s): {side.name} {qty} {ticker}, "
                      f"falling back to market", flush=True)
            if remaining > 0:
                ok, market_fill = _submit_market_order(ticker, remaining, side)
                if ok and market_fill > 0 and filled_qty > 0 and partial_fill > 0:
                    avg_fill = (partial_fill * filled_qty + market_fill * remaining) / qty
                    return (True, avg_fill)
                elif ok and market_fill > 0:
                    return (True, market_fill)
                return (ok, partial_fill if partial_fill > 0 else 0.0)
            return (True, partial_fill)
        except Exception as e:
            print(f"  [ALPACA] Error cancelling limit order: {e}, falling back to market", flush=True)
            return _submit_market_order(ticker, qty, side)

    except Exception as e:
        print(f"  [ALPACA] Limit order submit failed ({side.name} {qty} {ticker}): {e}, "
              f"falling back to market", flush=True)
        return _submit_market_order(ticker, qty, side)


def _submit_market_order(ticker: str, qty: int, side: OrderSide) -> tuple[bool, float]:
    """Submit a market order as fallback. Returns (success, fill_price)."""
    try:
        order_req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = _get_trading_client().submit_order(order_req)
        print(f"  [ALPACA] Market order: {side.name} {qty} {ticker}", flush=True)
        # Wait briefly for fill, then check fill price
        time.sleep(2)
        try:
            filled_order = _get_trading_client().get_order_by_id(order.id)
            if filled_order.filled_avg_price:
                fill = float(filled_order.filled_avg_price)
                print(f"  [ALPACA] Market fill: {side.name} {qty} {ticker} @ ${fill:.2f}", flush=True)
                return (True, fill)
        except Exception:
            pass
        return (True, 0.0)  # succeeded but couldn't get fill price
    except Exception as e:
        print(f"  [ALPACA] Market order FAILED ({side.name} {qty} {ticker}): {e}", flush=True)
        return (False, 0.0)


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
    end = datetime.now()
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
                feed=DataFeed.IEX,
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
