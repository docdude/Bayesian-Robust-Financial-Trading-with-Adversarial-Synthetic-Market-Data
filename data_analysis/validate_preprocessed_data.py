"""
Validate preprocessed GAN training data (.npy files) for data quality issues.

Checks:
  1. Shape and dtype consistency
  2. NaN / Inf detection
  3. Global and per-feature z-score statistics
  4. Outlier counts at multiple thresholds
  5. Per-stock variance and extreme-value detection
  6. NaN-mask coverage per ticker
  7. Generates a summary report (CSV + console)

Usage:
    python data_analysis/validate_preprocessed_data.py [--data-dir datasets/output_data]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

FEAT_NAMES = ["close_ret", "oc_ret", "high_ratio", "low_ratio", "log_vol"]


def load_data(data_dir):
    data = np.load(os.path.join(data_dir, "output_data.npy"))
    mask = np.load(os.path.join(data_dir, "output_mask_data.npy"))
    tickers = np.load(os.path.join(data_dir, "ticker_list.npy"), allow_pickle=True)
    return data, mask, [str(t) for t in tickers]


def check_basics(data, mask, tickers):
    n_samples, seq_len, n_cols = data.shape
    n_stocks = len(tickers)
    expected_cols = n_stocks * 5

    issues = []
    if n_cols != expected_cols:
        issues.append(f"Column mismatch: {n_cols} != {n_stocks}*5={expected_cols}")
    if np.isnan(data).any():
        issues.append(f"NaN count: {np.isnan(data).sum()}")
    if np.isinf(data).any():
        issues.append(f"Inf count: {np.isinf(data).sum()}")

    print(f"Shape: {data.shape}  ({n_samples} samples, {seq_len} seq_len, {n_cols} features)")
    print(f"Tickers ({n_stocks}): {tickers}")
    print(f"Global: mean={np.nanmean(data):.4f}  std={np.nanstd(data):.4f}")
    print(f"Range: [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")
    if issues:
        print(f"ISSUES: {issues}")
    else:
        print("Basic checks: PASS")
    return issues


def check_outliers(data):
    flat = data.reshape(-1)
    thresholds = [5, 10, 20, 50]
    print("\n--- Outlier counts ---")
    results = {}
    for t in thresholds:
        n = int((np.abs(flat) > t).sum())
        pct = 100 * n / flat.size
        status = "OK" if pct < 0.01 else ("WARN" if pct < 0.1 else "FAIL")
        print(f"  |x| > {t:3d}: {n:>10d} ({pct:.4f}%)  [{status}]")
        results[f"outlier_gt_{t}"] = n
    return results


def check_per_feature(data, n_stocks):
    print("\n--- Per-feature statistics ---")
    rows = []
    for fi, fn in enumerate(FEAT_NAMES):
        cols = [s * 5 + fi for s in range(n_stocks)]
        fd = data[:, :, cols].reshape(-1)
        row = {
            "feature": fn,
            "mean": fd.mean(),
            "std": fd.std(),
            "min": fd.min(),
            "max": fd.max(),
            "abs_max": np.abs(fd).max(),
        }
        status = "OK" if row["abs_max"] <= 10 else ("WARN" if row["abs_max"] <= 50 else "FAIL")
        print(f"  {fn:12s}: mean={row['mean']:8.4f}  std={row['std']:8.4f}  "
              f"range=[{row['min']:8.4f}, {row['max']:8.4f}]  [{status}]")
        rows.append(row)
    return pd.DataFrame(rows)


def check_per_stock(data, tickers):
    print("\n--- Per-stock diagnostics ---")
    rows = []
    problems = []
    for s, ticker in enumerate(tickers):
        sd = data[:, :, s * 5:(s + 1) * 5].reshape(-1)
        std = sd.std()
        abs_max = np.abs(sd).max()
        n10 = int((np.abs(sd) > 10).sum())
        n50 = int((np.abs(sd) > 50).sum())

        row = {"ticker": ticker, "std": std, "abs_max": abs_max, "gt_10": n10, "gt_50": n50}
        rows.append(row)

        is_problem = n50 > 0 or std > 5
        if is_problem or n10 > 500:
            flag = " *** PROBLEM" if is_problem else ""
            print(f"  {ticker:6s}: std={std:.3f}  max|x|={abs_max:.2f}  "
                  f"|x|>10: {n10}  |x|>50: {n50}{flag}")
            if is_problem:
                problems.append(ticker)

    if not problems:
        print("  All stocks: PASS (no |x|>50 or std>5)")
    else:
        print(f"\n  PROBLEM STOCKS: {problems}")
    return pd.DataFrame(rows), problems


def check_mask(mask, tickers):
    print("\n--- NaN-mask coverage ---")
    rows = []
    for s, ticker in enumerate(tickers):
        sm = mask[:, :, s * 5:(s + 1) * 5]
        nan_pct = 100 * sm.sum() / sm.size
        rows.append({"ticker": ticker, "nan_pct": nan_pct})
        if nan_pct > 0.1:
            print(f"  {ticker:6s}: {nan_pct:.1f}% NaN-filled")

    if all(r["nan_pct"] <= 0.1 for r in rows):
        print("  All stocks: <0.1% NaN — PASS")
    return pd.DataFrame(rows)


def summary_verdict(basic_issues, outlier_results, stock_problems):
    print("\n" + "=" * 60)
    if not basic_issues and outlier_results.get("outlier_gt_50", 0) == 0 and not stock_problems:
        print("VERDICT: PASS — data is clean for GAN training")
        return True
    else:
        print("VERDICT: FAIL — data quality issues remain")
        if basic_issues:
            print(f"  Basic: {basic_issues}")
        if outlier_results.get("outlier_gt_50", 0) > 0:
            print(f"  Outliers |x|>50: {outlier_results['outlier_gt_50']}")
        if stock_problems:
            print(f"  Problem stocks: {stock_problems}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed GAN data")
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "datasets", "output_data"))
    parser.add_argument("--save-csv", default=None, help="Save per-stock report to CSV")
    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}\n")
    data, mask, tickers = load_data(args.data_dir)
    n_stocks = len(tickers)

    basic_issues = check_basics(data, mask, tickers)
    outlier_results = check_outliers(data)
    check_per_feature(data, n_stocks)
    stock_df, stock_problems = check_per_stock(data, tickers)
    check_mask(mask, tickers)
    passed = summary_verdict(basic_issues, outlier_results, stock_problems)

    if args.save_csv:
        stock_df.to_csv(args.save_csv, index=False)
        print(f"\nPer-stock report saved to: {args.save_csv}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
