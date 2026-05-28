#!/usr/bin/env python3
"""
Alpaca paper-trading bot for a portfolio of trained DQN agents.

Supports two modes:
    --portfolio  : Trades the configured WaveNet portfolio (default)
  --symbol XYZ : Trades a single stock (legacy mode)

Runs once per trading day (designed to be invoked by cron / systemd / scheduler).
Each invocation per stock:
  1. Fetches recent daily OHLCV bars from Alpaca.
  2. Computes 150 alpha features (matching training pipeline exactly).
  3. Normalises with the saved StandardScaler from training.
  4. Constructs a 30-day sliding-window observation.
  5. Runs the DQN agent to get an action (SHORT / CLOSE / LONG).
  6. Translates the action into Alpaca orders with position sizing.
  7. Logs everything.

Usage:
    # Portfolio mode (all Tier 1 stocks, equal weight):
    python alpaca_trader.py --portfolio

    # Single stock:
    python alpaca_trader.py --symbol MSFT

    # Continuous daemon:
    python alpaca_trader.py --portfolio --daemon

    # Dry run (no orders):
    python alpaca_trader.py --portfolio --dry-run
"""
import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
import urllib.error
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Ensure project root is importable ──────────────────────────────────────
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

from live_trading import config
from live_trading.features import cal_factor
from live_trading.agent_loader import load_agent, load_scaler, predict_action

# ── Logging setup ──────────────────────────────────────────────────────────
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(config.LOG_DIR, f"trader_{dt.date.today()}.log")
        ),
    ],
)
log = logging.getLogger("alpaca_trader")
DRY_RUN = False


# ═══════════════════════════════════════════════════════════════════════════
#  Alpaca API helpers (thin wrappers – avoids hard alpaca-py dependency)
# ═══════════════════════════════════════════════════════════════════════════
def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
        "accept": "application/json",
    }


def _alpaca_get(endpoint: str, params: dict | None = None) -> dict | list:
    """GET request to Alpaca REST API."""
    import urllib.request
    import urllib.parse

    url = f"{config.ALPACA_BASE_URL}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_alpaca_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _alpaca_post(endpoint: str, body: dict) -> dict:
    """POST request to Alpaca REST API."""
    import urllib.request

    url = f"{config.ALPACA_BASE_URL}{endpoint}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={**_alpaca_headers(), "content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _alpaca_delete(endpoint: str) -> dict | None:
    """DELETE request to Alpaca REST API."""
    import urllib.request

    url = f"{config.ALPACA_BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers=_alpaca_headers(), method="DELETE")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
        return json.loads(body) if body else None


def _data_get(endpoint: str, params: dict | None = None) -> dict | list:
    """GET from Alpaca *data* API (different base URL)."""
    import urllib.request
    import urllib.parse

    base = "https://data.alpaca.markets"
    url = f"{base}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_alpaca_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


# ═══════════════════════════════════════════════════════════════════════════
#  Market-data fetching
# ═══════════════════════════════════════════════════════════════════════════

def fetch_daily_bars(symbol: str, limit: int = config.MIN_HISTORY_BARS) -> pd.DataFrame:
    """Fetch *limit* most-recent daily bars from Alpaca data API v2.

    Returns DataFrame indexed by datetime with columns:
        open, high, low, close, adj_close, volume
    """
    # Alpaca returns bars oldest-first; fetch generously and trim to last *limit*.
    start_date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=limit * 2)).strftime("%Y-%m-%d")

    all_bars = []
    page_token = None
    while True:
        params = {
            "timeframe": "1Day",
            "adjustment": "all",
            "start": start_date,
            "limit": "1000",        # max per page
        }
        if page_token:
            params["page_token"] = page_token

        resp = _data_get(f"/v2/stocks/{symbol}/bars", params=params)
        bars = resp.get("bars") if isinstance(resp, dict) else resp
        if bars:
            all_bars.extend(bars)
        page_token = resp.get("next_page_token") if isinstance(resp, dict) else None
        if not page_token:
            break

    if not all_bars:
        raise ValueError(f"No bars returned for {symbol} (start={start_date})")

    rows = []
    for bar in all_bars:
        rows.append({
            "timestamp": bar["t"][:10],   # "2024-06-20T04:00:00Z" → "2024-06-20"
            "open": float(bar["o"]),
            "high": float(bar["h"]),
            "low": float(bar["l"]),
            "close": float(bar["c"]),
            "volume": float(bar["v"]),
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    # Keep only the most recent *limit* bars
    df = df.iloc[-limit:]
    # Alpaca split-adjusted bars – treat close as adj_close
    df["adj_close"] = df["close"]
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Observation builder
# ═══════════════════════════════════════════════════════════════════════════

def build_observation(
    bars_df: pd.DataFrame,
    scaler,
    timestamps: int = config.TIMESTAMPS,
) -> np.ndarray:
    """From raw OHLCV bars, compute features, normalise, and return
    the latest *timestamps*-row observation window.

    Returns ndarray of shape (timestamps, 153).
    """
    feat_df = cal_factor(bars_df, real_correlation=config.REAL_CORRELATION)

    # Select exactly the 150 feature columns + 3 temporals
    features_cols = config.FEATURES_NAME
    temporals_cols = config.TEMPORALS_NAME

    # Normalise the 150 features (NOT temporals) using the training scaler
    feat_df[features_cols] = scaler.transform(feat_df[features_cols])

    # Stack features + temporals → (N, 153)
    obs_array = feat_df[features_cols + temporals_cols].values

    # Take the last `timestamps` rows
    if obs_array.shape[0] < timestamps:
        raise ValueError(
            f"Not enough data: have {obs_array.shape[0]} rows, need {timestamps}"
        )
    return obs_array[-timestamps:]


# ═══════════════════════════════════════════════════════════════════════════
#  Risk management helpers
# ═══════════════════════════════════════════════════════════════════════════

_trade_count = 0   # reset at the start of each cycle


def check_drawdown_breaker() -> bool:
    """Return True if the portfolio has breached the max drawdown limit.
    Compares current equity to INITIAL_CAPITAL (high-water mark on paper)."""
    account = get_account_info()
    equity = float(account["equity"])
    drawdown = 1.0 - (equity / config.INITIAL_CAPITAL)
    if drawdown >= config.MAX_DRAWDOWN_PCT:
        log.critical(
            "CIRCUIT BREAKER: equity=$%.2f, drawdown=%.1f%% >= limit %.1f%%. "
            "ALL TRADING HALTED. Manual intervention required.",
            equity, drawdown * 100, config.MAX_DRAWDOWN_PCT * 100,
        )
        return True
    return False


def check_daily_loss_limit(start_equity: float) -> bool:
    """Return True if intraday loss exceeds the daily limit."""
    account = get_account_info()
    equity = float(account["equity"])
    daily_pnl_pct = (equity - start_equity) / start_equity
    if daily_pnl_pct <= -config.DAILY_LOSS_LIMIT_PCT:
        log.warning(
            "DAILY LOSS LIMIT: equity=$%.2f, day PnL=%.2f%% <= -%.1f%%. "
            "Skipping remaining trades for today.",
            equity, daily_pnl_pct * 100, config.DAILY_LOSS_LIMIT_PCT * 100,
        )
        return True
    return False


def check_data_staleness(bars_df: pd.DataFrame, symbol: str) -> bool:
    """Return True if the most recent bar is too old (stale data)."""
    last_bar_date = bars_df.index[-1].date()
    today = dt.date.today()
    gap = (today - last_bar_date).days
    if gap > config.MAX_DATA_STALENESS_DAYS:
        log.error(
            "STALE DATA for %s: last bar=%s, today=%s, gap=%d days > limit %d. Skipping.",
            symbol, last_bar_date, today, gap, config.MAX_DATA_STALENESS_DAYS,
        )
        return True
    return False


def check_trade_cap() -> bool:
    """Return True if we've hit the per-cycle trade cap."""
    global _trade_count
    if _trade_count >= config.MAX_TRADES_PER_CYCLE:
        log.warning(
            "TRADE CAP: %d trades reached (limit=%d). No more orders this cycle.",
            _trade_count, config.MAX_TRADES_PER_CYCLE,
        )
        return True
    return False


def get_gross_exposure() -> float:
    """Sum of |market_value| across all positions / equity."""
    account = get_account_info()
    equity = float(account["equity"])
    if equity <= 0:
        return float("inf")
    try:
        positions = _alpaca_get("/v2/positions")
    except Exception:
        return 0.0
    total_abs = sum(abs(float(p["market_value"])) for p in positions)
    return total_abs / equity


def get_available_cash() -> float:
    """Return the actual available buying power, respecting the cash reserve."""
    account = get_account_info()
    equity = float(account["equity"])
    cash = float(account["cash"])
    reserve = equity * config.MIN_CASH_RESERVE_PCT
    return max(cash - reserve, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Position management
# ═══════════════════════════════════════════════════════════════════════════

def get_account_info() -> dict:
    return _alpaca_get("/v2/account")


def get_position(symbol: str) -> dict | None:
    """Return the current position dict, or None if flat."""
    try:
        return _alpaca_get(f"/v2/positions/{symbol}")
    except Exception:
        return None


def close_position(symbol: str) -> None:
    """Liquidate the entire position for *symbol*."""
    global _trade_count
    if DRY_RUN:
        _trade_count += 1
        log.info("[DRY-RUN] Would close position for %s (trade #%d)", symbol, _trade_count)
        return
    try:
        _alpaca_delete(f"/v2/positions/{symbol}")
        _trade_count += 1
        log.info("Closed position for %s (trade #%d)", symbol, _trade_count)
    except Exception as e:
        log.warning("close_position(%s) error: %s", symbol, e)


def _wait_positions_flat(symbols: list[str], timeout: int = 30) -> None:
    """Poll until all *symbols* show no position (or timeout)."""
    deadline = time.monotonic() + timeout
    remaining = set(symbols)
    while remaining and time.monotonic() < deadline:
        time.sleep(1)
        still_open = set()
        for sym in remaining:
            if get_position(sym) is not None:
                still_open.add(sym)
        remaining = still_open
        if remaining:
            log.info("  Still settling: %s", ", ".join(sorted(remaining)))
    if remaining:
        log.warning("  Timeout waiting for positions to close: %s", ", ".join(sorted(remaining)))


def submit_market_order(symbol: str, qty: int, side: str) -> dict | None:
    """Submit a market order with pre-flight checks.

    Returns the order response, or None if rejected by safety checks.
    """
    global _trade_count

    if check_trade_cap():
        return None

    # Cash check for buy orders. The training env sizes all-in positions with
    # a (1 + transaction_cost_pct) denominator, so use the same buffer here.
    if side == "buy" and not DRY_RUN:
        available = get_available_cash()
        price = _get_last_price(symbol)
        estimated_cost = qty * price * (1 + config.TRANSACTION_COST_PCT)
        if estimated_cost > available:
            old_qty = qty
            qty = _target_qty_for_allocation(available, price)
            log.warning(
                "CASH CHECK: %s buy %d shares ($%.0f incl cost) > available $%.0f. "
                "Reduced to %d shares.",
                symbol, old_qty, estimated_cost, available, qty,
            )
            if qty <= 0:
                log.warning("CASH CHECK: cannot buy any %s, skipping order.", symbol)
                return None

    # Gross exposure check
    exposure = get_gross_exposure() if not DRY_RUN else 0.0
    if not DRY_RUN and exposure >= config.MAX_GROSS_EXPOSURE:
        log.warning(
            "EXPOSURE LIMIT: gross exposure=%.2f >= limit %.2f. Skipping %s %s %d.",
            exposure, config.MAX_GROSS_EXPOSURE, side, symbol, qty,
        )
        return None

    if DRY_RUN:
        _trade_count += 1
        log.info("[DRY-RUN] Would submit order: %s %d %s (trade #%d)",
                 side, qty, symbol, _trade_count)
        return {"id": "dry-run", "symbol": symbol, "qty": str(abs(qty)), "side": side}

    body = {
        "symbol": symbol,
        "qty": str(abs(qty)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    try:
        resp = _alpaca_post("/v2/orders", body)
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode()
        except Exception:
            pass
        log.error(
            "ORDER REJECTED: %s %d %s → HTTP %d: %s",
            side, qty, symbol, e.code, error_body,
        )
        return None
    _trade_count += 1
    log.info("Order submitted: %s %d %s → %s (trade #%d)",
             side, qty, symbol, resp.get("id", "?"), _trade_count)
    return resp


def _target_qty_for_allocation(allocated_equity: float, price: float) -> int:
    """Size like EnvironmentRET.trade(): notional divided by price plus fee."""
    if price <= 0:
        return 0
    return int(allocated_equity / (price * (1 + config.TRANSACTION_COST_PCT)))


def execute_action(action_label: str, symbol: str, allocation_pct: float = 1.0) -> None:
    """Translate SHORT / CLOSE / LONG into Alpaca orders.

    Position semantics match training env: same-action = no-op.
    Sizing respects allocation_pct, cash reserve, per-stock cap, and exposure limit.
    """
    pos = get_position(symbol)
    current_qty = int(pos["qty"]) if pos else 0
    current_side = pos["side"] if pos else None  # "long" or "short"

    account = get_account_info()
    equity = float(account["equity"])
    # Cap at the per-stock maximum
    effective_alloc = min(allocation_pct, config.MAX_SINGLE_STOCK_PCT)
    allocated_equity = equity * effective_alloc
    last_price = float(pos["current_price"]) if pos else _get_last_price(symbol)

    if action_label == "CLOSE":
        if current_qty != 0:
            close_position(symbol)
        else:
            log.info("[%s] Action=CLOSE but already flat; no-op", symbol)

    elif action_label == "LONG":
        if current_side == "long":
            log.info("[%s] Action=LONG, already long %d shares; no-op", symbol, current_qty)
            return
        # Close any existing short first
        if current_qty != 0:
            close_position(symbol)
            if not DRY_RUN:
                time.sleep(1)
        target_qty = _target_qty_for_allocation(allocated_equity, last_price)
        if target_qty > 0:
            submit_market_order(symbol, target_qty, "buy")

    elif action_label == "SHORT":
        if current_side == "short":
            log.info("[%s] Action=SHORT, already short %d shares; no-op", symbol, current_qty)
            return
        # Close any existing long first
        if current_qty != 0:
            close_position(symbol)
            if not DRY_RUN:
                time.sleep(1)
        target_qty = _target_qty_for_allocation(allocated_equity, last_price)
        if target_qty > 0:
            submit_market_order(symbol, target_qty, "sell")

    else:
        log.error("Unknown action: %s", action_label)


def _get_last_price(symbol: str) -> float:
    """Fetch the latest trade price."""
    resp = _data_get(f"/v2/stocks/{symbol}/trades/latest")
    trade = resp.get("trade", resp)
    return float(trade["p"])


# ═══════════════════════════════════════════════════════════════════════════
#  Main trading loop
# ═══════════════════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    """Check if the US stock market is currently open."""
    clock = _alpaca_get("/v2/clock")
    return clock.get("is_open", False)


def _trade_single_stock(
    symbol: str,
    agent,
    device,
    scaler,
    allocation_pct: float = 1.0,
    execute: bool = True,
) -> dict:
    """Run fetch→compute→decide for one stock. Optionally execute.

    Returns action info dict; raises on data problems.
    """
    log.info("── %s (allocation=%.1f%%) ──", symbol, allocation_pct * 100)

    bars_df = fetch_daily_bars(symbol)
    log.info("  Got %d bars: %s → %s", len(bars_df), bars_df.index[0].date(), bars_df.index[-1].date())

    # Data staleness check
    if check_data_staleness(bars_df, symbol):
        return {"symbol": symbol, "action": "SKIP", "q_values": [0, 0, 0],
                "reason": "stale_data"}

    observation = build_observation(bars_df, scaler)
    action_id, action_label, q_values = predict_action(observation, agent, device)
    log.info(
        "  Action: %s (id=%d) | Q-values: short=%.4f close=%.4f long=%.4f",
        action_label, action_id, q_values[0], q_values[1], q_values[2],
    )

    if execute:
        execute_action(action_label, symbol, allocation_pct)

    return {"symbol": symbol, "action": action_label, "q_values": q_values.tolist()}


def run_portfolio() -> None:
    """Execute one trading cycle for the full Tier 1 portfolio.

    Safety protocol:
      1. Check drawdown circuit breaker before anything.
      2. Load all agents, compute all signals (no orders yet).
      3. Phase 1: execute all CLOSE orders and close legs of reversals.
      4. Wait for settlement.
      5. Check daily loss limit.
      6. Phase 2: execute all BUY/SELL (new position) orders, cash-aware.
      7. Log portfolio summary.
    """
    global _trade_count
    _trade_count = 0

    portfolio = config.PORTFOLIO
    n_stocks = len(portfolio)
    allocation_pct = 1.0 / n_stocks

    log.info("=" * 60)
    log.info("PORTFOLIO MODE: %d stocks, %.1f%% each (cap %.1f%%)",
             n_stocks, allocation_pct * 100, config.MAX_SINGLE_STOCK_PCT * 100)
    log.info("Stocks: %s", ", ".join(portfolio.keys()))

    # ── Pre-flight: drawdown breaker ──────────────────────────────────────
    if check_drawdown_breaker():
        return

    market_open = is_market_open()
    if not market_open:
        log.warning("Market is closed — logging actions but NOT executing orders")

    # Record start-of-cycle equity for daily loss limit
    account = get_account_info()
    start_equity = float(account["equity"])
    log.info("Start-of-cycle equity: $%.2f (initial capital: $%.2f)",
             start_equity, config.INITIAL_CAPITAL)

    # ── Signal generation: compute all actions first ──────────────────────
    log.info("─ PHASE 0: Computing signals for all stocks ─")
    agents = {}
    signals = {}  # symbol → {"action": str, "q_values": list}
    for symbol, paths in portfolio.items():
        try:
            agent, device = load_agent(checkpoint_path=paths["checkpoint_path"])
            scaler = load_scaler(scaler_path=paths["scaler_path"])
            agents[symbol] = (agent, device, scaler)

            bars_df = fetch_daily_bars(symbol)
            log.info("  %s: %d bars (%s → %s)",
                     symbol, len(bars_df), bars_df.index[0].date(), bars_df.index[-1].date())

            if check_data_staleness(bars_df, symbol):
                signals[symbol] = {"action": "SKIP", "q_values": [0, 0, 0]}
                continue

            observation = build_observation(bars_df, scaler)
            action_id, action_label, q_values = predict_action(observation, agent, device)
            signals[symbol] = {"action": action_label, "q_values": q_values.tolist()}
            log.info("  %s → %s (Q: [%.4f, %.4f, %.4f])",
                     symbol, action_label, q_values[0], q_values[1], q_values[2])
        except Exception:
            log.exception("Error computing signal for %s", symbol)
            signals[symbol] = {"action": "SKIP", "q_values": [0, 0, 0]}

    if not market_open:
        log.info("─ Market closed; signals computed but no orders placed ─")
        _log_portfolio_summary(signals)
        return

    # ── Classify actions into sells/closes (phase 1) and buys (phase 2) ──
    phase1_symbols = []  # symbols that need closing (CLOSE, or reversal close leg)
    phase2_symbols = []  # symbols that need new position entry

    for symbol, sig in signals.items():
        action = sig["action"]
        if action == "SKIP":
            continue
        pos = get_position(symbol)
        current_side = pos["side"] if pos else None

        if action == "CLOSE":
            if pos and int(float(pos["qty"])) != 0:
                phase1_symbols.append(symbol)
        elif action == "LONG":
            if current_side == "long":
                pass  # no-op
            elif current_side == "short":
                phase1_symbols.append(symbol)  # close short first
                phase2_symbols.append(symbol)  # then buy
            else:
                phase2_symbols.append(symbol)  # flat → buy
        elif action == "SHORT":
            if current_side == "short":
                pass  # no-op
            elif current_side == "long":
                phase1_symbols.append(symbol)  # close long first
                phase2_symbols.append(symbol)  # then sell short
            else:
                phase2_symbols.append(symbol)  # flat → sell short

    log.info("─ PHASE 1: Closing/reducing positions (%d orders) ─", len(phase1_symbols))
    for symbol in phase1_symbols:
        if check_trade_cap():
            break
        close_position(symbol)

    # Wait until all Phase 1 closes have settled (position actually gone)
    if phase1_symbols and DRY_RUN:
        log.info("  [DRY-RUN] Skipping settlement wait for Phase 1 closes")
    elif phase1_symbols:
        log.info("  Waiting for Phase 1 closes to settle...")
        _wait_positions_flat(phase1_symbols, timeout=30)

    # ── Daily loss check between phases ───────────────────────────────────
    if check_daily_loss_limit(start_equity):
        _log_portfolio_summary(signals)
        return

    # ── Phase 2: Open new positions (cash-aware, sell-first already done) ─
    log.info("─ PHASE 2: Opening new positions (%d orders) ─", len(phase2_symbols))
    for symbol in phase2_symbols:
        if check_trade_cap():
            break
        if check_daily_loss_limit(start_equity):
            break
        action = signals[symbol]["action"]
        last_price = _get_last_price(symbol)

        account = get_account_info()
        equity = float(account["equity"])
        effective_alloc = min(allocation_pct, config.MAX_SINGLE_STOCK_PCT)
        allocated_equity = equity * effective_alloc
        target_qty = _target_qty_for_allocation(allocated_equity, last_price)

        if target_qty <= 0:
            log.warning("  [%s] target_qty=0 (alloc=$%.0f, price=$%.2f). Skipping.",
                        symbol, allocated_equity, last_price)
            continue

        if action == "LONG":
            log.info("  [%s] LONG → buy %d shares ($%.0f)", symbol, target_qty,
                     target_qty * last_price)
            submit_market_order(symbol, target_qty, "buy")
        elif action == "SHORT":
            log.info("  [%s] SHORT → sell %d shares ($%.0f)", symbol, target_qty,
                     target_qty * last_price)
            submit_market_order(symbol, target_qty, "sell")

    # ── Final summary ─────────────────────────────────────────────────────
    _log_portfolio_summary(signals)


def _log_portfolio_summary(signals: dict) -> None:
    """Log the end-of-cycle portfolio state."""
    account = get_account_info()
    log.info("─" * 40)
    log.info("PORTFOLIO SUMMARY")
    log.info("  Equity: $%.2f | Cash: $%.2f | Gross Exposure: %.2f",
             float(account["equity"]), float(account["cash"]), get_gross_exposure())
    for symbol, sig in signals.items():
        pos = get_position(symbol)
        pos_str = f"{pos['side']} {pos['qty']} @ ${pos['avg_entry_price']}" if pos else "FLAT"
        log.info("  %s: signal=%s | position=%s", symbol, sig["action"], pos_str)
    log.info("  Total trades this cycle: %d / %d", _trade_count, config.MAX_TRADES_PER_CYCLE)
    log.info("Cycle complete")


def run_once() -> None:
    """Execute one trading cycle for a single stock (legacy mode)."""
    global _trade_count
    _trade_count = 0

    log.info("=" * 60)
    log.info("Starting trading cycle for %s", config.SYMBOL)

    # Pre-flight: drawdown breaker
    if check_drawdown_breaker():
        return

    # 1. Load agent & scaler
    log.info("Loading agent from %s", config.CHECKPOINT_PATH)
    agent, device = load_agent()
    scaler = load_scaler()
    log.info("Agent loaded on %s", device)

    # 2. Fetch market data
    log.info("Fetching %d daily bars for %s", config.MIN_HISTORY_BARS, config.SYMBOL)
    bars_df = fetch_daily_bars(config.SYMBOL)
    log.info("Got %d bars: %s → %s", len(bars_df), bars_df.index[0].date(), bars_df.index[-1].date())

    # Data staleness check
    if check_data_staleness(bars_df, config.SYMBOL):
        return

    # 3. Build observation
    observation = build_observation(bars_df, scaler)
    log.info("Observation shape: %s", observation.shape)

    # 4. Get action
    action_id, action_label, q_values = predict_action(observation, agent, device)
    log.info(
        "Action: %s (id=%d) | Q-values: short=%.4f close=%.4f long=%.4f",
        action_label, action_id, q_values[0], q_values[1], q_values[2],
    )

    # 5. Execute
    if not is_market_open():
        log.warning("Market is closed — logging action but NOT executing")
    else:
        account = get_account_info()
        start_equity = float(account["equity"])
        if check_daily_loss_limit(start_equity):
            return
        execute_action(action_label, config.SYMBOL)

    # 6. Log portfolio state
    account = get_account_info()
    pos = get_position(config.SYMBOL)
    log.info(
        "Portfolio: equity=$%.2f cash=$%.2f | Gross exposure: %.2f | Position: %s",
        float(account["equity"]),
        float(account["cash"]),
        get_gross_exposure(),
        f"{pos['side']} {pos['qty']} @ ${pos['avg_entry_price']}" if pos else "FLAT",
    )
    log.info("  Trades this cycle: %d / %d", _trade_count, config.MAX_TRADES_PER_CYCLE)
    log.info("Cycle complete")


def run_daemon(portfolio_mode: bool = False) -> None:
    """Run continuously, executing once per trading day at the configured time."""
    tz = ZoneInfo(config.TIMEZONE)
    mode_str = "PORTFOLIO" if portfolio_mode else f"SINGLE ({config.SYMBOL})"
    log.info("Daemon mode [%s] — will trade at %02d:%02d %s daily",
             mode_str, config.TRADE_HOUR, config.TRADE_MINUTE, config.TIMEZONE)

    traded_today = False

    while True:
        now = dt.datetime.now(tz)

        # Reset flag at midnight
        if now.hour == 0 and now.minute == 0:
            traded_today = False

        if (
            not traded_today
            and now.hour == config.TRADE_HOUR
            and now.minute >= config.TRADE_MINUTE
        ):
            try:
                if not is_market_open():
                    log.info("Trade time but market closed (holiday/weekend); skipping")
                else:
                    if portfolio_mode:
                        run_portfolio()
                    else:
                        run_once()
                traded_today = True
            except Exception:
                log.exception("Error during trading cycle")

        time.sleep(30)  # poll every 30 seconds


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global DRY_RUN
    parser = argparse.ArgumentParser(description="DQN Alpaca Paper Trader")
    parser.add_argument("--portfolio", action="store_true",
                        help="Trade the configured WaveNet clean28 portfolio with equal-weight allocation")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Trade a single stock (overrides --portfolio)")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute action but do not execute orders")
    args = parser.parse_args()

    # Validate credentials
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        log.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        sys.exit(1)

    DRY_RUN = args.dry_run
    if DRY_RUN:
        log.info("Dry-run mode enabled: signals and intended orders will be logged only")

    # Single-stock override
    if args.symbol:
        config.SYMBOL = args.symbol
        if args.symbol in config.PORTFOLIO:
            paths = config.PORTFOLIO[args.symbol]
            config.CHECKPOINT_PATH = paths["checkpoint_path"]
            config.SCALER_PATH = paths["scaler_path"]

    portfolio_mode = args.portfolio and not args.symbol

    if args.daemon:
        run_daemon(portfolio_mode=portfolio_mode)
    elif portfolio_mode:
        run_portfolio()
    else:
        run_once()


if __name__ == "__main__":
    main()
