# Stat Arb Trading System

## Overview
Intraday pairs trading algo running on Alpaca paper trading. Trades 8 cointegrated pairs on 5-min bars using z-score mean reversion. All positions close at 3:45 PM ET, no overnight holds.

## Server
- **Host**: 167.99.116.104 (DigitalOcean, Ubuntu)
- **SSH**: `ssh root@167.99.116.104` password: `6100SigmaChi`
- **Service**: `statarb.service` — restart with `systemctl restart statarb`
- **Logs**: `journalctl -u statarb -f` or `/root/stat_arb/live_feed/trader_output.log`
- **Project path on server**: `/root/stat_arb`

## API Keys
- **Alpaca API Key**: `PKVHIIJQX2QWCX7SYLQMY3NXCS`
- **Alpaca Secret**: `GCPvVEosUs724SnEKJLZS6p8KTsCSeQMEaygUCtWw3mP`
- **Alpaca**: Paper trading account, free tier, IEX data feed (real-time)
- **Telegram Bot Token**: `8524701532:AAGiISCA7mjfUn69E-H52z5WmyHqovjvcCg`
- **Telegram Chat ID**: `8795063486`

## Git Remotes
- **origin**: `https://github.com/AidanJaghab/stat_arb.git` (main repo)
- **jake**: `https://github.com/jake-greenfield/stat_arb_fork.git` (fork)
- Always push to BOTH remotes: `git push origin main && git push jake main`

## Deploy Workflow
```bash
# 1. Make changes locally
# 2. Commit
git add <files> && git commit -m "message"
# 3. Push to both remotes
git push origin main && git push jake main
# 4. Deploy to server
ssh root@167.99.116.104 "cd /root/stat_arb && git pull origin main && systemctl restart statarb"
# 5. Verify
ssh root@167.99.116.104 "systemctl status statarb --no-pager -l | head -20"
```

## Key Files
| File | What it does |
|------|-------------|
| `live_feed/trader.py` | Main trading loop, PairPosition class, all strategy logic, Telegram alerts |
| `live_feed/alpaca_client.py` | Alpaca API: data fetching, order execution (limit w/ market fallback), positions |
| `live_feed/active_pairs.csv` | Current 8 pairs with ticker_a, ticker_b, hedge_ratio, sector |
| `live_feed/position_state.json` | Persisted position state for restart recovery |
| `live_feed/signals.csv` | Trade signal log |
| `live_feed/slippage.csv` | Fill price vs signal price tracking |
| `live_feed/pair_pnl.csv` | Per-pair consecutive loss counts |
| `strategy/scanner.py` | Pair scanner — runs weekly Sunday 8PM ET. Filters: coint p<0.02, ADF p<0.02, R²>0.60, half-life<30 |

## Strategy Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Entry z-score | 2.0 | Enter when \|z\| >= 2.0 |
| Exit z-score | 0.5 | Exit when \|z\| <= 0.5 (mean reversion) |
| Hard stop | 3.25 | Force exit if \|z\| blows past 3.25 |
| Trailing stop | 2% of entry cost | Exit if P&L drops 2% from peak |
| Time stop | 78 bars (1 day) | Force exit if held full session |
| Cooldown | 78 bars | After hard/time/trailing stop |
| Opening cooldown | 30 min | No entries until 10:00 AM ET |
| EOD no entry | 3:30 PM ET | Block new entries |
| EOD close | 3:45 PM ET | Force exit all positions |
| Base exposure | $10,000/leg | Vol-adjusted per pair |
| Max gross exposure | $100,000 | Cap across all positions |
| Loss streak cutoff | 3 | Scale to 50% after 3 consecutive losses |
| Limit order buffer | 0.05% | Limit price cushion |
| Limit order timeout | 30s | Then falls back to market |

## Telegram Alerts
| Alert | When | What |
|-------|------|------|
| Entry | On trade entry | Pair, shares, dollar amounts per leg, z-score, balance warning if >15% imbalanced |
| Exit | On trade exit | Pair, reason, bars held, P&L from actual fill prices |
| Morning check | 10:00 AM ET | Account, positions, Alpaca vs internal state, cooldowns |
| Hourly pulse | Every hour 10AM-3:45PM | Active count, unrealized P&L, best/worst pair, today's realized |
| Daily recap | 4:00 PM ET | Total realized P&L, win rate, per-pair breakdown, exit types, slippage, diagnostics |
| Weekly report | Friday 4:10 PM ET | Week's trades, per-pair stats, avg hold time, slippage, pair health |

## Architecture Notes
- **Data feed**: IEX (free real-time, ~2-3% of volume). Switched from SIP (15-min delayed).
- **Order execution**: Limit orders first (0.05% buffer), 30s timeout, then market fallback. Returns actual fill prices.
- **P&L**: Computed from actual Alpaca fill prices, not IEX quotes. Entry fill prices stored on position.
- **State persistence**: `position_state.json` saves all position fields on every tick. On restart, restores from file, then cross-checks with Alpaca positions. Orphaned positions auto-closed.
- **Weekly rescan**: Scanner runs Sunday 8PM ET automatically, produces new `active_pairs.csv`.
- **Shares mapping**: LONG_SPREAD: long=A, short=B. SHORT_SPREAD: long=B, short=A. Critical — was a bug source.

## Common Tasks
- **Check if running**: `ssh root@167.99.116.104 "systemctl status statarb --no-pager"`
- **View live logs**: `ssh root@167.99.116.104 "journalctl -u statarb -f"`
- **Check positions**: `ssh root@167.99.116.104 "cat /root/stat_arb/live_feed/position_state.json | python3 -m json.tool"`
- **Manual pair rescan**: `ssh root@167.99.116.104 "cd /root/stat_arb && /root/stat_arb/venv/bin/python -m strategy.scanner"`
- **Cancel all orders**: Done automatically on startup, or via Alpaca dashboard
