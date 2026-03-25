"""Configuration for the Alpaca paper-trading bot."""
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = str(Path(__file__).resolve().parents[1])

# Load .env from project root (won't overwrite existing env vars)
load_dotenv(os.path.join(ROOT, ".env"))

# Best checkpoint from sweep (ckpt 13 = 130K steps, SR=1.576)
CHECKPOINT_PATH = os.path.join(
    ROOT,
    "downstream_tasks/rl/trading/workdir/exp/trading/AAPL/dqn/exp001_aug/saved_model/13.pth",
)
SCALER_PATH = os.path.join(ROOT, "live_trading/artifacts/scaler.pkl")

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
