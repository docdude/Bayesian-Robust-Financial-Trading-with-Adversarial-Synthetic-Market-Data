"""Manifold-overlap sanity check for the ETF WaveNet Lambert GAN.

Compares the *reconstructed* synthetic Alpha158 feature manifold (what the RL
agent actually consumes) against the real GLD Alpha158 manifold from the
processed parquet. Mirrors the GLD_aug.py config exactly so the API runs with
the same paths / feature_method / correlation settings as training.

Run from repo root on a GPU host (the generator won't load on small GPUs):
    python generator/WAVENET_LAMBERT_GAN/manifold_check_etf.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from generator.WAVENET_LAMBERT_GAN.models.API import GeneratorAPI  # noqa: E402

# ---- GLD_aug.py settings (kept in sync with the RL config) ----
GAN_MODEL_PATH = "generator/WAVENET_LAMBERT_GAN/output/futures_etf_lambert_derived"
GAN_DATA_PATH = "datasets/output_data_lambert_future_etfs_derived"
GAN_FEATURE_METHOD = "derived"   # API also autoselects from the data dir
GAN_REAL_CORRELATION = True
GAN_CHECKPOINT_EPOCH = 4000
REAL_PARQUET_TMPL = "datasets/processd_day_future_etfs/features/{ticker}.parquet"
TEMPORALS = ['day', 'weekday', 'month']

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
]

N_WINDOWS = 200          # number of generated windows to pool
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='GLD')
    parser.add_argument('--windows', type=int, default=N_WINDOWS)
    parser.add_argument('--tail', type=int, default=30,
                        help='rows kept from the end of each window (mirrors '
                             'the RL agent which consumes result[-timestamps:]). '
                             '0 = keep the whole window.')
    # Optional overrides so the same check can target the Dow derived model
    # (defaults preserve the ETF log_returns behaviour).
    parser.add_argument('--model-path', default=GAN_MODEL_PATH)
    parser.add_argument('--data-dir', default=GAN_DATA_PATH)
    parser.add_argument('--feature-method', default=GAN_FEATURE_METHOD)
    parser.add_argument('--real-parquet-tmpl', default=REAL_PARQUET_TMPL)
    parser.add_argument('--checkpoint-epoch', default=GAN_CHECKPOINT_EPOCH,
                        type=lambda v: None if str(v).lower() == 'none' else int(v))
    parser.add_argument('--no-real-correlation', dest='real_correlation',
                        action='store_false', default=GAN_REAL_CORRELATION)
    args = parser.parse_args()
    ticker = args.ticker
    real_parquet = args.real_parquet_tmpl.format(ticker=ticker)

    np.random.seed(SEED)

    api = GeneratorAPI(
        model_path=args.model_path,
        ticker_name=ticker,
        obs_features=FEATURES_NAME,
        temporal_features=TEMPORALS,
        feature_method=args.feature_method,
        data_dir=args.data_dir,
        checkpoint_epoch=args.checkpoint_epoch,
        real_correlation=args.real_correlation,
    )

    dates = pd.DatetimeIndex(api._date_array)
    n = min(args.windows, len(dates))
    pick = np.random.choice(len(dates), n, replace=False)
    sample_dates = dates[np.sort(pick)]
    print(f"\nGenerating {n} synthetic windows over "
          f"{sample_dates[0].date()} → {sample_dates[-1].date()} ...")

    eps = np.zeros((n, api.macro_dim), dtype=np.float32)   # real macro conditioning
    synth_dfs = api.call_batch(list(sample_dates), eps)
    if args.tail and args.tail > 0:
        synth_dfs = [d.iloc[-args.tail:] for d in synth_dfs]
    synth = pd.concat(synth_dfs, axis=0)[FEATURES_NAME].apply(
        pd.to_numeric, errors='coerce')
    synth = synth.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Rows kept per window (tail): {args.tail if args.tail else 'all'}")
    print(f"Synthetic pooled rows: {synth.shape}")

    # ---- Real manifold over the same date span ----
    real = pd.read_parquet(real_parquet)
    real['timestamp'] = pd.to_datetime(real['timestamp'])
    lo, hi = sample_dates.min(), sample_dates.max()
    real = real[(real['timestamp'] >= lo) & (real['timestamp'] <= hi)]
    real = real[FEATURES_NAME].apply(pd.to_numeric, errors='coerce')
    real = real.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Real pooled rows:      {real.shape}")

    # ---- Per-feature overlap metrics (standardized by real moments) ----
    rows = []
    for f in FEATURES_NAME:
        r = real[f].values
        s = synth[f].values
        if len(r) < 10 or len(s) < 10:
            continue
        mu, sd = r.mean(), r.std() + 1e-12
        rs, ss = (r - mu) / sd, (s - mu) / sd
        ks = ks_2samp(rs, ss).statistic
        w = wasserstein_distance(rs, ss)
        # fraction of synthetic inside the real 1-99 pct band
        lo_b, hi_b = np.percentile(r, 1), np.percentile(r, 99)
        inside = np.mean((s >= lo_b) & (s <= hi_b))
        rows.append((f, ks, w, inside, ss.mean(), ss.std()))

    res = pd.DataFrame(rows, columns=[
        'feature', 'ks', 'wass_std', 'inside_1_99', 'synth_z_mean', 'synth_z_std'])

    print("\n" + "=" * 78)
    print(f"  ETF GAN Manifold-Overlap Check (synthetic Alpha158 vs real {ticker})")
    print("=" * 78)
    print(f"  Features compared:                 {len(res)}")
    print(f"  Mean KS (standardized):            {res['ks'].mean():.4f}")
    print(f"  Median KS:                         {res['ks'].median():.4f}")
    print(f"  Mean Wasserstein (std units):      {res['wass_std'].mean():.4f}")
    print(f"  Mean synthetic mass inside 1-99%:  {res['inside_1_99'].mean():.3f}")
    print(f"  Features with KS > 0.30 (poor):    {(res['ks'] > 0.30).sum()}")
    print(f"  Features with <80% inside band:    {(res['inside_1_99'] < 0.80).sum()}")
    print("=" * 78)

    worst = res.sort_values('ks', ascending=False).head(12)
    print("\n  Worst-overlapping features (by KS):")
    print(worst.to_string(index=False,
                          float_format=lambda x: f"{x:.4f}"))

    best = res.sort_values('ks').head(8)
    print("\n  Best-overlapping features (by KS):")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out = f"generator/WAVENET_LAMBERT_GAN/manifold_check_etf_{ticker}_result.csv"
    res.to_csv(out, index=False)
    print(f"\n  Per-feature results written to {out}")


if __name__ == "__main__":
    main()
