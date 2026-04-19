"""
Standalone feature engineering for the DQN adapter.

Computes 150 alpha factors + 3 temporals from OHLCV data, exactly
matching the training pipeline.

Input:  DataFrame with columns [open, high, low, close, adj_close, volume]
Output: DataFrame with 150 features + 3 temporals (day, weekday, month).
"""
import numpy as np
import pandas as pd

# Ordered feature names (150 alpha + 3 temporal) matching training.
FEATURES_NAME = [
    "open", "high", "low", "close", "adj_close",
    "kmid", "kmid2", "klen", "kup", "kup2", "klow", "klow2", "ksft", "ksft2",
    "roc_5", "roc_10", "roc_20", "roc_30", "roc_60",
    "ma_5", "ma_10", "ma_20", "ma_30", "ma_60",
    "std_5", "std_10", "std_20", "std_30", "std_60",
    "beta_5", "beta_10", "beta_20", "beta_30", "beta_60",
    "max_5", "max_10", "max_20", "max_30", "max_60",
    "min_5", "min_10", "min_20", "min_30", "min_60",
    "qtlu_5", "qtlu_10", "qtlu_20", "qtlu_30", "qtlu_60",
    "qtld_5", "qtld_10", "qtld_20", "qtld_30", "qtld_60",
    "rank_5", "rank_10", "rank_20", "rank_30", "rank_60",
    "imax_5", "imax_10", "imax_20", "imax_30", "imax_60",
    "imin_5", "imin_10", "imin_20", "imin_30", "imin_60",
    "imxd_5", "imxd_10", "imxd_20", "imxd_30", "imxd_60",
    "rsv_5", "rsv_10", "rsv_20", "rsv_30", "rsv_60",
    "cntp_5", "cntp_10", "cntp_20", "cntp_30", "cntp_60",
    "cntn_5", "cntn_10", "cntn_20", "cntn_30", "cntn_60",
    "cntd_5", "cntd_10", "cntd_20", "cntd_30", "cntd_60",
    "corr_5", "corr_10", "corr_20", "corr_30", "corr_60",
    "cord_5", "cord_10", "cord_20", "cord_30", "cord_60",
    "sump_5", "sump_10", "sump_20", "sump_30", "sump_60",
    "sumn_5", "sumn_10", "sumn_20", "sumn_30", "sumn_60",
    "sumd_5", "sumd_10", "sumd_20", "sumd_30", "sumd_60",
    "vma_5", "vma_10", "vma_20", "vma_30", "vma_60",
    "vstd_5", "vstd_10", "vstd_20", "vstd_30", "vstd_60",
    "wvma_5", "wvma_10", "wvma_20", "wvma_30", "wvma_60",
    "vsump_5", "vsump_10", "vsump_20", "vsump_30", "vsump_60",
    "vsumn_5", "vsumn_10", "vsumn_20", "vsumn_30", "vsumn_60",
    "vsumd_5", "vsumd_10", "vsumd_20", "vsumd_30", "vsumd_60",
    "log_volume",
]

TEMPORALS_NAME = ["day", "weekday", "month"]
ALL_COLUMNS = FEATURES_NAME + TEMPORALS_NAME


def _my_rank(x: np.ndarray) -> float:
    return pd.Series(x).rank(pct=True).iloc[-1]


def cal_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 150 alpha factors + temporals from OHLCV data.

    Parameters
    ----------
    df : DataFrame with columns: open, high, low, close, adj_close, volume.

    Returns
    -------
    DataFrame with 153 columns (150 features + 3 temporals).
    """
    df = df.copy()

    max_oc = df[["open", "close"]].max(axis=1)
    min_oc = df[["open", "close"]].min(axis=1)
    hl_range = df["high"] - df["low"] + 1e-12

    df["kmid"] = (df["close"] - df["open"]) / df["close"]
    df["kmid2"] = (df["close"] - df["open"]) / hl_range
    df["klen"] = (df["high"] - df["low"]) / df["open"]
    df["kup"] = (df["high"] - max_oc) / df["open"]
    df["kup2"] = (df["high"] - max_oc) / hl_range
    df["klow"] = (min_oc - df["low"]) / df["open"]
    df["klow2"] = (min_oc - df["low"]) / hl_range
    df["ksft"] = (2 * df["close"] - df["high"] - df["low"]) / df["open"]
    df["ksft2"] = (2 * df["close"] - df["high"] - df["low"]) / hl_range

    windows = [5, 10, 20, 30, 60]
    cols = {}

    for w in windows:
        cols[f"roc_{w}"] = df["close"].shift(w) / df["close"]
    for w in windows:
        cols[f"ma_{w}"] = df["close"].rolling(w).mean() / df["close"]
    for w in windows:
        cols[f"std_{w}"] = df["close"].rolling(w).std() / df["close"]
    for w in windows:
        cols[f"beta_{w}"] = (df["close"].shift(w) - df["close"]) / (w * df["close"])
    for w in windows:
        cols[f"max_{w}"] = df["close"].rolling(w).max() / df["close"]
    for w in windows:
        cols[f"min_{w}"] = df["close"].rolling(w).min() / df["close"]
    for w in windows:
        cols[f"qtlu_{w}"] = df["close"].rolling(w).quantile(0.8) / df["close"]
    for w in windows:
        cols[f"qtld_{w}"] = df["close"].rolling(w).quantile(0.2) / df["close"]
    for w in windows:
        cols[f"rank_{w}"] = df["close"].rolling(w).apply(_my_rank, raw=True) / w
    for w in windows:
        cols[f"imax_{w}"] = df["high"].rolling(w).apply(np.argmax, raw=True) / w
    for w in windows:
        cols[f"imin_{w}"] = df["low"].rolling(w).apply(np.argmin, raw=True) / w
    for w in windows:
        cols[f"imxd_{w}"] = (
            df["high"].rolling(w).apply(np.argmax, raw=True)
            - df["low"].rolling(w).apply(np.argmin, raw=True)
        ) / w

    for w in windows:
        shift = df["close"].shift(w)
        low_or_shift = df["low"].where(df["low"] < shift, shift)
        high_or_shift = df["high"].where(df["high"] > shift, shift)
        cols[f"rsv_{w}"] = (df["close"] - low_or_shift) / (high_or_shift - low_or_shift + 1e-12)

    ret1 = df["close"].pct_change(1)
    for w in windows:
        cols[f"cntp_{w}"] = ret1.gt(0).rolling(w).sum() / w
    for w in windows:
        cols[f"cntn_{w}"] = ret1.lt(0).rolling(w).sum() / w
    for w in windows:
        cols[f"cntd_{w}"] = cols[f"cntp_{w}"] - cols[f"cntn_{w}"]

    # Correlation features: training bug replication (always 1.0)
    for w in windows:
        cols[f"corr_{w}"] = df["close"].rolling(w).apply(lambda x: 1.0, raw=True)
    close_ret = df["close"] / df["close"].shift(1)
    for w in windows:
        cols[f"cord_{w}"] = close_ret.rolling(w).apply(lambda x: 1.0, raw=True)

    abs_ret1 = ret1.abs()
    pos_ret1 = ret1.clip(lower=0)
    for w in windows:
        cols[f"sump_{w}"] = pos_ret1.rolling(w).sum() / (abs_ret1.rolling(w).sum() + 1e-12)
    for w in windows:
        cols[f"sumn_{w}"] = 1 - cols[f"sump_{w}"]
    for w in windows:
        cols[f"sumd_{w}"] = 2 * cols[f"sump_{w}"] - 1

    for w in windows:
        cols[f"vma_{w}"] = df["volume"].rolling(w).mean() / (df["volume"] + 1e-12)
    for w in windows:
        cols[f"vstd_{w}"] = df["volume"].rolling(w).std() / (df["volume"] + 1e-12)
    weighted = np.abs(df["close"] / df["close"].shift(1) - 1) * df["volume"]
    for w in windows:
        cols[f"wvma_{w}"] = weighted.rolling(w).std() / (weighted.rolling(w).mean() + 1e-12)

    vchg = df["volume"] - df["volume"].shift(1)
    abs_vchg = vchg.abs()
    pos_vchg = vchg.clip(lower=0)
    for w in windows:
        cols[f"vsump_{w}"] = pos_vchg.rolling(w).sum() / (abs_vchg.rolling(w).sum() + 1e-12)
    for w in windows:
        cols[f"vsumn_{w}"] = 1 - cols[f"vsump_{w}"]
    for w in windows:
        cols[f"vsumd_{w}"] = 2 * cols[f"vsump_{w}"] - 1

    cols["log_volume"] = np.log(df["volume"] + 1)

    # Temporal constants (training artefact — see docstring in live_trading/features.py)
    cols["day"] = pd.Series(1, index=df.index)
    cols["weekday"] = pd.Series(3, index=df.index)
    cols["month"] = pd.Series(1, index=df.index)

    df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
