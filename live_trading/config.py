"""Configuration for the Alpaca paper-trading bot."""
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = str(Path(__file__).resolve().parents[1])

# Load .env from project root (won't overwrite existing env vars)
load_dotenv(os.path.join(ROOT, ".env"))

# ── Portfolio: Tier 1 stocks with best checkpoint per ticker ───────────────
# Each entry: ticker → (checkpoint_number, test_sharpe)
# Selected by best val_SR from sweep; only tickers with test_SR >= 1.0
TIER1_STOCKS = {
    "PG":   {"ckpt": 2,  "test_SR": 2.58},
    "MRK":  {"ckpt": 2,  "test_SR": 1.90},
    "BA":   {"ckpt": 32, "test_SR": 1.59},
    "MCD":  {"ckpt": 9,  "test_SR": 1.55},
    "JNJ":  {"ckpt": 4,  "test_SR": 1.51},
    "MSFT": {"ckpt": 18, "test_SR": 1.51},
    "KO":   {"ckpt": 10, "test_SR": 1.35},
    "NKE":  {"ckpt": 12, "test_SR": 1.11},
}

def _ckpt_path(ticker: str, ckpt: int) -> str:
    return os.path.join(
        ROOT,
        f"downstream_tasks/rl/trading/workdir/exp/trading/{ticker}/dqn/exp001_aug/saved_model/{ckpt}.pth",
    )

def _scaler_path(ticker: str) -> str:
    return os.path.join(ROOT, f"live_trading/artifacts/{ticker}_scaler.pkl")

# Build per-ticker config dict used by the trader
PORTFOLIO = {}
for _ticker, _info in TIER1_STOCKS.items():
    PORTFOLIO[_ticker] = {
        "checkpoint_path": _ckpt_path(_ticker, _info["ckpt"]),
        "scaler_path": _scaler_path(_ticker),
    }

# Legacy single-stock config (still used by single-stock mode)
SYMBOL = "AAPL"
CHECKPOINT_PATH = _ckpt_path("AAPL", 12)
SCALER_PATH = _scaler_path("AAPL")

# ── Agent architecture (must match training config AAPL_aug.py) ───────────
INPUT_DIM = 153          # 150 features + 3 temporals
TIMESTAMPS = 30          # sliding-window length
EMBED_DIM = 64
DEPTH = 1
ACTION_DIM = 3           # short / close / long
TEMPORALS_NAME = ["day", "weekday", "month"]
USE_QUANTILE_BELIEF = True
QUANTILE_HEADS = [0.05, 0.25, 0.5, 0.75, 0.95]
USE_NFSP = True

# ── Trading parameters ────────────────────────────────────────────────────
SYMBOL = "AAPL"
INITIAL_CAPITAL = 100_000.0       # paper-trading starting capital
TRANSACTION_COST_PCT = 1e-3       # must match training
POSITION_LOWERBOUND = -1          # allow short

# ── Risk management ───────────────────────────────────────────────────────
# Portfolio-level drawdown circuit breaker: if equity drops below this
# fraction of INITIAL_CAPITAL, halt ALL trading until manual reset.
MAX_DRAWDOWN_PCT = 0.20           # 20% drawdown → full stop

# Daily loss limit: if intraday PnL drops below this, skip remaining trades.
DAILY_LOSS_LIMIT_PCT = 0.05       # 5% daily loss → stop for the day

# Per-stock max allocation (hard cap regardless of equal-weight math).
MAX_SINGLE_STOCK_PCT = 0.15       # never more than 15% in one name

# Minimum cash reserve: always keep this fraction in cash (never go all-in).
MIN_CASH_RESERVE_PCT = 0.05       # keep 5% cash buffer

# Maximum trades per cycle (prevents runaway loops).
MAX_TRADES_PER_CYCLE = 16         # 8 stocks × 2 (close + open) max

# Data staleness: reject bars older than this many calendar days.
MAX_DATA_STALENESS_DAYS = 5       # accounts for weekends + holidays

# Gross exposure limit: sum of |position_value| / equity.
# 1.0 = fully invested long-only; 2.0 = allows 100% long + 100% short.
MAX_GROSS_EXPOSURE = 1.5

# ── Feature engineering ───────────────────────────────────────────────────
# Minimum bars needed: 60 (max rolling window) + 30 (observation window) + margin
MIN_HISTORY_BARS = 120
ROLLING_WINDOWS = [5, 10, 20, 30, 60]

FEATURES_NAME = [
    'open', 'high', 'low', 'close', 'adj_close',
    'kmid', 'kmid2', 'klen', 'kup', 'kup2', 'klow', 'klow2', 'ksft', 'ksft2',
    'roc_5', 'roc_10', 'roc_20', 'roc_30', 'roc_60',
    'ma_5', 'ma_10', 'ma_20', 'ma_30', 'ma_60',
    'std_5', 'std_10', 'std_20', 'std_30', 'std_60',
    'beta_5', 'beta_10', 'beta_20', 'beta_30', 'beta_60',
    'max_5', 'max_10', 'max_20', 'max_30', 'max_60',
    'min_5', 'min_10', 'min_20', 'min_30', 'min_60',
    'qtlu_5', 'qtlu_10', 'qtlu_20', 'qtlu_30', 'qtlu_60',
    'qtld_5', 'qtld_10', 'qtld_20', 'qtld_30', 'qtld_60',
    'rank_5', 'rank_10', 'rank_20', 'rank_30', 'rank_60',
    'imax_5', 'imax_10', 'imax_20', 'imax_30', 'imax_60',
    'imin_5', 'imin_10', 'imin_20', 'imin_30', 'imin_60',
    'imxd_5', 'imxd_10', 'imxd_20', 'imxd_30', 'imxd_60',
    'rsv_5', 'rsv_10', 'rsv_20', 'rsv_30', 'rsv_60',
    'cntp_5', 'cntp_10', 'cntp_20', 'cntp_30', 'cntp_60',
    'cntn_5', 'cntn_10', 'cntn_20', 'cntn_30', 'cntn_60',
    'cntd_5', 'cntd_10', 'cntd_20', 'cntd_30', 'cntd_60',
    'corr_5', 'corr_10', 'corr_20', 'corr_30', 'corr_60',
    'cord_5', 'cord_10', 'cord_20', 'cord_30', 'cord_60',
    'sump_5', 'sump_10', 'sump_20', 'sump_30', 'sump_60',
    'sumn_5', 'sumn_10', 'sumn_20', 'sumn_30', 'sumn_60',
    'sumd_5', 'sumd_10', 'sumd_20', 'sumd_30', 'sumd_60',
    'vma_5', 'vma_10', 'vma_20', 'vma_30', 'vma_60',
    'vstd_5', 'vstd_10', 'vstd_20', 'vstd_30', 'vstd_60',
    'wvma_5', 'wvma_10', 'wvma_20', 'wvma_30', 'wvma_60',
    'vsump_5', 'vsump_10', 'vsump_20', 'vsump_30', 'vsump_60',
    'vsumn_5', 'vsumn_10', 'vsumn_20', 'vsumn_30', 'vsumn_60',
    'vsumd_5', 'vsumd_10', 'vsumd_20', 'vsumd_30', 'vsumd_60',
    'log_volume',
]  # 150 total

LABELS_NAME = ['ret1', 'mov1']

# ── Alpaca ─────────────────────────────────────────────────────────────────
# Set via environment variables — never hard-code secrets
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.environ.get(
    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
)

# ── Scheduling ─────────────────────────────────────────────────────────────
# Run daily at 15:55 ET (5 min before close) to capture the day's bar
TRADE_HOUR = 15
TRADE_MINUTE = 55
TIMEZONE = "US/Eastern"

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(ROOT, "live_trading/logs")
