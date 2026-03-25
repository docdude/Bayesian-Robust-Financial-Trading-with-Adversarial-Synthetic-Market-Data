"""
Feature engineering for live trading.

Replicates module/processor/processor.py::cal_factor() exactly so that
live OHLCV data produces the same 150 alpha features + 3 temporals the
DQN agent was trained on.

Input: a pandas DataFrame with columns [open, high, low, close, adj_close, volume]
       indexed by datetime.
Output: DataFrame with 150 features + 3 temporals (day, weekday, month).
"""
import numpy as np
import pandas as pd


def _my_rank(x: np.ndarray) -> float:
    """Percentile rank of the last element in the window.
    Mirrors the original `my_rank` used by processor.py and API.py:
        pd.Series(x).rank(pct=True).iloc[-1]
    """
    return pd.Series(x).rank(pct=True).iloc[-1]


def cal_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 150 alpha factors + temporals from OHLCV data.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: open, high, low, close, adj_close, volume.
        Index must be a DatetimeIndex.

    Returns
    -------
    DataFrame with columns matching config.FEATURES_NAME + temporals.
    First ~60 rows will have NaN (filled with 0) due to rolling windows.
    """
    df = df.copy()

    # ── Keltner-channel-like features (9) ──────────────────────────────────
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

    # ── Price-based rolling features ───────────────────────────────────────
    for w in windows:
        df[f"roc_{w}"] = df["close"].shift(w) / df["close"]

    for w in windows:
        df[f"ma_{w}"] = df["close"].rolling(w).mean() / df["close"]

    for w in windows:
        df[f"std_{w}"] = df["close"].rolling(w).std() / df["close"]

    for w in windows:
        df[f"beta_{w}"] = (df["close"].shift(w) - df["close"]) / (w * df["close"])

    for w in windows:
        df[f"max_{w}"] = df["close"].rolling(w).max() / df["close"]

    for w in windows:
        df[f"min_{w}"] = df["close"].rolling(w).min() / df["close"]

    for w in windows:
        df[f"qtlu_{w}"] = df["close"].rolling(w).quantile(0.8) / df["close"]

    for w in windows:
        df[f"qtld_{w}"] = df["close"].rolling(w).quantile(0.2) / df["close"]

    for w in windows:
        df[f"rank_{w}"] = df["close"].rolling(w).apply(_my_rank, raw=True) / w

    for w in windows:
        df[f"imax_{w}"] = df["high"].rolling(w).apply(np.argmax, raw=True) / w

    for w in windows:
        df[f"imin_{w}"] = df["low"].rolling(w).apply(np.argmin, raw=True) / w

    for w in windows:
        df[f"imxd_{w}"] = (
            df["high"].rolling(w).apply(np.argmax, raw=True)
            - df["low"].rolling(w).apply(np.argmin, raw=True)
        ) / w

    # ── RSV ────────────────────────────────────────────────────────────────
    for w in windows:
        shift = df["close"].shift(w)
        low_or_shift = df["low"].where(df["low"] < shift, shift)
        high_or_shift = df["high"].where(df["high"] > shift, shift)
        df[f"rsv_{w}"] = (df["close"] - low_or_shift) / (high_or_shift - low_or_shift + 1e-12)

    # ── Return-count features ──────────────────────────────────────────────
    ret1 = df["close"].pct_change(1)
    for w in windows:
        df[f"cntp_{w}"] = ret1.gt(0).rolling(w).sum() / w

    for w in windows:
        df[f"cntn_{w}"] = ret1.lt(0).rolling(w).sum() / w

    for w in windows:
        df[f"cntd_{w}"] = df[f"cntp_{w}"] - df[f"cntn_{w}"]

    # ── Correlation features ───────────────────────────────────────────────
    # NOTE: The original processor.py had a bug: it called
    #   df1.corr(pairwise=df2)  where df2 is a Rolling object.
    # This passes `other=None, pairwise=<rolling obj>` (truthy), computing
    # self-correlation = 1.0 (NaN where window insufficient). The training
    # data reflects this bug, and the StandardScaler was fit on it, so we
    # must replicate the behaviour for consistent normalisation.
    for w in windows:
        df[f"corr_{w}"] = df["close"].rolling(w).apply(lambda x: 1.0, raw=True)

    close_ret = df["close"] / df["close"].shift(1)
    for w in windows:
        df[f"cord_{w}"] = close_ret.rolling(w).apply(lambda x: 1.0, raw=True)

    # ── Sum-positive / sum-negative return features ────────────────────────
    abs_ret1 = ret1.abs()
    pos_ret1 = ret1.clip(lower=0)
    for w in windows:
        df[f"sump_{w}"] = pos_ret1.rolling(w).sum() / (abs_ret1.rolling(w).sum() + 1e-12)

    for w in windows:
        df[f"sumn_{w}"] = 1 - df[f"sump_{w}"]

    for w in windows:
        df[f"sumd_{w}"] = 2 * df[f"sump_{w}"] - 1

    # ── Volume-based features ──────────────────────────────────────────────
    for w in windows:
        df[f"vma_{w}"] = df["volume"].rolling(w).mean() / (df["volume"] + 1e-12)

    for w in windows:
        df[f"vstd_{w}"] = df["volume"].rolling(w).std() / (df["volume"] + 1e-12)

    for w in windows:
        weighted = np.abs(df["close"] / df["close"].shift(1) - 1) * df["volume"]
        df[f"wvma_{w}"] = weighted.rolling(w).std() / (weighted.rolling(w).mean() + 1e-12)

    vchg = df["volume"] - df["volume"].shift(1)
    abs_vchg = vchg.abs()
    pos_vchg = vchg.clip(lower=0)
    for w in windows:
        df[f"vsump_{w}"] = pos_vchg.rolling(w).sum() / (abs_vchg.rolling(w).sum() + 1e-12)
    for w in windows:
        df[f"vsumn_{w}"] = 1 - df[f"vsump_{w}"]
    for w in windows:
        df[f"vsumd_{w}"] = 2 * df[f"vsump_{w}"] - 1

    df["log_volume"] = np.log(df["volume"] + 1)

    # ── Temporal features ─────────────────────────────────────────────────
    # NOTE: The training parquet was saved with a RangeIndex (0..N), so the
    # processor's pd.to_datetime(df.index) converted row numbers to
    # 1970-01-01 nanosecond offsets, yielding day=1, weekday=3, month=1
    # for EVERY row. The agent was trained on these constants, so we must
    # replicate them. (Temporals are NOT normalised by the scaler.)
    df["day"] = 1
    df["weekday"] = 3
    df["month"] = 1

    # ── Clean up ───────────────────────────────────────────────────────────
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df
