#!/usr/bin/env python3
"""
Alpaca paper-trading bot for the trained DQN agent.

Runs once per trading day (designed to be invoked by cron / systemd / scheduler).
Each invocation:
  1. Fetches recent daily OHLCV bars from Alpaca.
  2. Computes 150 alpha features (matching training pipeline exactly).
  3. Normalises with the saved StandardScaler from training.
  4. Constructs a 30-day sliding-window observation.
  5. Runs the DQN agent to get an action (SHORT / CLOSE / LONG).
  6. Translates the action into Alpaca orders.
  7. Logs everything.

Usage:
    # One-shot (e.g. from cron at 15:55 ET on market days):
    python alpaca_trader.py

    # Continuous daemon (blocks, sleeps until next trade window):
    python alpaca_trader.py --daemon
"""
import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
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
    resp = _data_get(
        f"/v2/stocks/{symbol}/bars",
        params={"timeframe": "1Day", "limit": str(limit), "adjustment": "all", "feed": "sip"},
    )
    bars = resp.get("bars", resp) if isinstance(resp, dict) else resp

    rows = []
    for bar in bars:
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
    feat_df = cal_factor(bars_df)

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
    try:
        _alpaca_delete(f"/v2/positions/{symbol}")
        log.info("Closed position for %s", symbol)
    except Exception as e:
        log.warning("close_position(%s) error: %s", symbol, e)


def submit_market_order(symbol: str, qty: int, side: str) -> dict:
    """Submit a market order. *side* is 'buy' or 'sell'."""
    body = {
        "symbol": symbol,
        "qty": str(abs(qty)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    resp = _alpaca_post("/v2/orders", body)
    log.info("Order submitted: %s %d %s → %s", side, qty, symbol, resp.get("id", "?"))
    return resp


def execute_action(action_label: str, symbol: str) -> None:
    """Translate SHORT / CLOSE / LONG into Alpaca orders.

    Position semantics (matches training env):
        LONG  → 100% of equity in stock
        SHORT → -100% of equity (short position)
        CLOSE → flat (no position)
    """
    pos = get_position(symbol)
    current_qty = int(pos["qty"]) if pos else 0
    current_side = pos["side"] if pos else None  # "long" or "short"

    account = get_account_info()
    equity = float(account["equity"])
    last_price = float(pos["current_price"]) if pos else _get_last_price(symbol)

    if action_label == "CLOSE":
        if current_qty != 0:
            close_position(symbol)
        else:
            log.info("Action=CLOSE but already flat; no-op")

    elif action_label == "LONG":
        if current_side == "long":
            log.info("Action=LONG, already long %d shares; no-op", current_qty)
            return
        # Close any existing short first
        if current_qty != 0:
            close_position(symbol)
            time.sleep(1)  # allow settlement
            account = get_account_info()
            equity = float(account["equity"])
        # Go long with full equity
        target_qty = int(equity / last_price)
        if target_qty > 0:
            submit_market_order(symbol, target_qty, "buy")

    elif action_label == "SHORT":
        if current_side == "short":
            log.info("Action=SHORT, already short %d shares; no-op", current_qty)
            return
        # Close any existing long first
        if current_qty != 0:
            close_position(symbol)
            time.sleep(1)
            account = get_account_info()
            equity = float(account["equity"])
        # Go short with full equity
        target_qty = int(equity / last_price)
        if target_qty > 0:
            submit_market_order(symbol, target_qty, "sell")

    else:
        log.error("Unknown action: %s", action_label)


def _get_last_price(symbol: str) -> float:
    """Fetch the latest trade price."""
    resp = _data_get(f"/v2/stocks/{symbol}/trades/latest", params={"feed": "sip"})
    trade = resp.get("trade", resp)
    return float(trade["p"])


# ═══════════════════════════════════════════════════════════════════════════
#  Main trading loop
# ═══════════════════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    """Check if the US stock market is currently open."""
    clock = _alpaca_get("/v2/clock")
    return clock.get("is_open", False)


def run_once() -> None:
    """Execute one trading cycle: fetch → compute → decide → trade."""
    log.info("=" * 60)
    log.info("Starting trading cycle for %s", config.SYMBOL)

    # 1. Load agent & scaler
    log.info("Loading agent from %s", config.CHECKPOINT_PATH)
    agent, device = load_agent()
    scaler = load_scaler()
    log.info("Agent loaded on %s", device)

    # 2. Fetch market data
    log.info("Fetching %d daily bars for %s", config.MIN_HISTORY_BARS, config.SYMBOL)
    bars_df = fetch_daily_bars(config.SYMBOL)
    log.info("Got %d bars: %s → %s", len(bars_df), bars_df.index[0].date(), bars_df.index[-1].date())

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
        execute_action(action_label, config.SYMBOL)

    # 6. Log portfolio state
    account = get_account_info()
    pos = get_position(config.SYMBOL)
    log.info(
        "Portfolio: equity=$%.2f cash=$%.2f | Position: %s",
        float(account["equity"]),
        float(account["cash"]),
        f"{pos['side']} {pos['qty']} @ ${pos['avg_entry_price']}" if pos else "FLAT",
    )
    log.info("Cycle complete")


def run_daemon() -> None:
    """Run continuously, executing once per trading day at the configured time."""
    tz = ZoneInfo(config.TIMEZONE)
    log.info("Daemon mode — will trade at %02d:%02d %s daily",
             config.TRADE_HOUR, config.TRADE_MINUTE, config.TIMEZONE)

    # Pre-load heavy objects once
    agent, device = load_agent()
    scaler = load_scaler()
    log.info("Agent pre-loaded on %s", device)

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
                    bars_df = fetch_daily_bars(config.SYMBOL)
                    observation = build_observation(bars_df, scaler)
                    action_id, action_label, q_values = predict_action(
                        observation, agent, device
                    )
                    log.info(
                        "Action: %s | Q: [%.4f, %.4f, %.4f]",
                        action_label, *q_values,
                    )
                    execute_action(action_label, config.SYMBOL)

                    account = get_account_info()
                    pos = get_position(config.SYMBOL)
                    log.info(
                        "Equity=$%.2f | Pos=%s",
                        float(account["equity"]),
                        f"{pos['side']} {pos['qty']}" if pos else "FLAT",
                    )
                traded_today = True
            except Exception:
                log.exception("Error during trading cycle")

        time.sleep(30)  # poll every 30 seconds


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DQN Alpaca Paper Trader")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute action but do not execute orders")
    args = parser.parse_args()

    # Validate credentials
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        log.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        sys.exit(1)

    if args.dry_run:
        # Override execute_action to no-op
        global execute_action
        _original = execute_action
        def execute_action(action_label, symbol):
            log.info("[DRY-RUN] Would execute %s on %s", action_label, symbol)

    if args.daemon:
        run_daemon()
    else:
        run_once()


if __name__ == "__main__":
    main()
