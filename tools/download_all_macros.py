"""
Download all 46 macro variables needed for the GAN macro conditioning.

Sources:
  1. FRED (9 vars)         — via fredapi
  2. Fed EBP (3 vars)      — direct CSV from Fed Board
  3. Fed FCI-G (7 vars)    — direct CSV from Fed Board
  4. Fed SOFR (3 vars)     — direct CSV from Fed Board
  5. Fed CIE (2 vars)      — direct CSV from Fed Board
  6. Fed LMCI (1 var)      — via FRED (FRBLMCI, discontinued 2017)
  7. Fed DKW (21 vars)     — direct CSV from Fed Board (TIPS decomposition)

Output: datasets/macro_raw/  (individual source CSVs)
        datasets/macro_processed/macro_data.csv  (merged, monthly, forward-filled)
        datasets/macro_processed/macro_data_resampled.csv  (daily, forward-filled)

Usage:
  python tools/download_all_macros.py [--fred-key YOUR_KEY] [--output-dir datasets/macro_raw]
"""

import os
import sys
import io
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = str(Path(__file__).resolve().parents[1])

# ---------------------------------------------------------------------------
# 1. FRED series (monthly + quarterly)
# ---------------------------------------------------------------------------
FRED_MONTHLY = ["CPIAUCSL", "FEDFUNDS", "GS10", "M1NS", "M2SL", "T10YIEM", "UNRATE"]
FRED_QUARTERLY = ["A191RL1Q225SBEA", "A191RP1Q027SBEA"]
FRED_LMCI = "FRBLMCI"  # Board LMCI (discontinued 2017-06, 491 obs)


def download_fred(api_key: str, output_dir: str) -> dict[str, pd.DataFrame]:
    """Download all FRED series and return {filename: DataFrame}."""
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    frames = {}

    # Monthly series  -> macro1_Monthly.txt equivalent
    monthly_dfs = []
    for sid in FRED_MONTHLY:
        logger.info(f"  FRED: {sid}")
        s = fred.get_series(sid)
        s.name = sid
        monthly_dfs.append(s)
    monthly = pd.concat(monthly_dfs, axis=1)
    monthly.index.name = "DATE"
    monthly.to_csv(os.path.join(output_dir, "macro1_Monthly.txt"), sep="\t")
    frames["macro1_Monthly.txt"] = monthly
    logger.info(f"  -> macro1_Monthly.txt  ({len(monthly)} rows, {len(FRED_MONTHLY)} cols)")

    # Quarterly series -> macro1_Quarterly.txt equivalent
    quarterly_dfs = []
    for sid in FRED_QUARTERLY:
        logger.info(f"  FRED: {sid}")
        s = fred.get_series(sid)
        s.name = sid
        quarterly_dfs.append(s)
    quarterly = pd.concat(quarterly_dfs, axis=1)
    quarterly.index.name = "DATE"
    quarterly.to_csv(os.path.join(output_dir, "macro1_Quarterly.txt"), sep="\t")
    frames["macro1_Quarterly.txt"] = quarterly
    logger.info(f"  -> macro1_Quarterly.txt  ({len(quarterly)} rows, {len(FRED_QUARTERLY)} cols)")

    # LMCI -> lmci_feds.csv equivalent
    logger.info(f"  FRED: {FRED_LMCI}")
    lmci = fred.get_series(FRED_LMCI)
    lmci_df = lmci.to_frame(name="lmci")
    lmci_df.index.name = "date"
    lmci_df.to_csv(os.path.join(output_dir, "lmci_feds.csv"))
    frames["lmci_feds.csv"] = lmci_df
    logger.info(f"  -> lmci_feds.csv  ({len(lmci_df)} rows)")

    return frames


# ---------------------------------------------------------------------------
# 2. Direct Fed Board CSV downloads
# ---------------------------------------------------------------------------
FED_SOURCES = {
    "ebp_csv.csv": {
        "url": "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv",
        "skiprows": 0,
        "date_col": "date",
        "desc": "Gilchrist-Zakrajšek credit spreads & EBP (3 vars)",
    },
    "fci_g_public_monthly_3yr.csv": {
        "url": "https://www.federalreserve.gov/econres/notes/feds-notes/fci_g_public_monthly_3yr.csv",
        "skiprows": 0,
        "date_col": "date",
        "desc": "FCI-G Financial Conditions Impulse on Growth (7+ vars)",
    },
    "FED_Note_Term_SOFR.csv": {
        "url": "https://www.federalreserve.gov/econres/notes/feds-notes/FED_Note_Term_SOFR.csv",
        "skiprows": 10,  # Has disclaimer header (10 lines before column names)
        "date_col": "DATE",
        "date_format": "%m-%d-%Y",  # Live file uses MM-DD-YYYY format
        "desc": "SOFR term rates (6 vars, 3 realized used)",
    },
    "FEDS-Note-2873-cie-data.csv": {
        "url": "https://www.federalreserve.gov/econres/notes/feds-notes/FEDS-Note-2873-cie-data.csv",
        "skiprows": 0,
        "date_col": "period",
        "desc": "Common Inflation Expectations (2 vars)",
    },
}

# DKW TIPS Yield Curve Decomposition
# D'Amico, Kim, and Wei (2018), "Tips from TIPS", updated by Kim, Walsh, and Wei (2019).
# Published as a FEDS Note with a permanent CSV download URL.
DKW_SOURCE = {
    "url": "https://www.federalreserve.gov/econres/notes/feds-notes/DKW_updates.csv",
    "skiprows": 11,  # 11 header/comment lines before the data
    "date_col": "date",
    "desc": "DKW TIPS Yield Curve Decomposition (21 vars) — D'Amico, Kim, Wei",
}

# Expected DKW column names from macro_list.txt (21 variables):
DKW_EXPECTED_COLS = [
    "exp.real.short.rate.5", "exp.inflation.5", "real.term.prem.5",
    "inflation.risk.prem.5", "tips.liq.prem.5", "nominal.yield.raw.5",
    "nominal.yield.fitted.5", "ic.raw.5", "ic.fitted.5",
    "exp.real.short.rate.10", "exp.inflation.10", "tips.liq.prem.10",
    "nominal.yield.raw.10", "nominal.yield.fitted.10", "ic.raw.10",
    "ic.fitted.10",
    "exp.real.short.rate.5f5", "exp.inflation.5f5", "tips.liq.prem.5f5",
    "nominal.yield.raw.5f5", "nominal.yield.fitted.5f5",
]


def download_fed_csv(name: str, info: dict, output_dir: str) -> pd.DataFrame:
    """Download a Fed Board CSV and save locally."""
    logger.info(f"  Downloading {name}: {info['desc']}")
    r = requests.get(info["url"], timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text), skiprows=info.get("skiprows", 0))

    # SOFR: live file is daily with FORWARD+REALIZED columns and MM-DD-YYYY dates.
    # We only need the monthly REALIZED columns (matching local format).
    if name == "FED_Note_Term_SOFR.csv":
        realized_cols = [c for c in df.columns if c.startswith("REALIZED_")]
        df["DATE"] = pd.to_datetime(df["DATE"].str.strip(), format="%m-%d-%Y", errors="coerce")
        df = df[["DATE"] + realized_cols].copy()
        # Convert whitespace/empty strings to numeric
        for col in realized_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["DATE"])
        # Keep only rows that have REALIZED data (not just FORWARD)
        df = df.dropna(subset=realized_cols, how="all")
        # Aggregate to monthly (mean) to match original local format
        df = df.set_index("DATE").resample("MS").mean(numeric_only=True).reset_index()
        df = df.sort_values("DATE")
        logger.info(f"  SOFR: extracted {len(realized_cols)} REALIZED columns, aggregated to {len(df)} monthly rows")

    outpath = os.path.join(output_dir, name)
    df.to_csv(outpath, index=False)
    logger.info(f"  -> {name}  ({len(df)} rows, {len(df.columns)} cols)")
    return df


def download_dkw(output_dir: str) -> pd.DataFrame:
    """
    Download the DKW model data from the Fed's FEDS Notes.

    Source: D'Amico, Kim, and Wei (2018), "Tips from TIPS: The Informational
    Content of Treasury Inflation-Protected Security Prices", updated by
    Kim, Walsh, and Wei (2019), FEDS Notes.

    URL: https://www.federalreserve.gov/econres/notes/feds-notes/DKW_updates.csv
    """
    logger.info(f"  Downloading DKW: {DKW_SOURCE['desc']}")
    r = requests.get(DKW_SOURCE["url"], timeout=60)
    r.raise_for_status()

    df = pd.read_csv(
        io.StringIO(r.text),
        skiprows=DKW_SOURCE["skiprows"],
        na_values="NA",
    )
    logger.info(f"  DKW: {len(df)} rows, {len(df.columns)} cols")
    logger.info(f"  Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")

    # Verify expected columns
    found = [c for c in DKW_EXPECTED_COLS if c in df.columns]
    missing = [c for c in DKW_EXPECTED_COLS if c not in df.columns]
    if missing:
        logger.warning(f"  Missing {len(missing)} DKW columns: {missing}")
    else:
        logger.info(f"  All {len(DKW_EXPECTED_COLS)} DKW columns present")

    outpath = os.path.join(output_dir, "DKW_updates.csv")
    df.to_csv(outpath, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. Merge all sources into unified macro dataset
# ---------------------------------------------------------------------------
MACRO_LIST_PATH = os.path.join(ROOT, "generator", "GRT_GAN", "data", "macro_list.txt")

DATE_FORMATS = ["%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%Ym%m"]


def unify_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convert various date formats to datetime, then to month start."""
    col = df[date_col]
    converted = None
    for fmt in DATE_FORMATS:
        try:
            converted = pd.to_datetime(col, format=fmt, errors="coerce")
            if not converted.isna().all():
                break
        except Exception:
            continue
    if converted is None or converted.isna().any():
        converted = pd.to_datetime(col, errors="coerce")

    df = df.copy()
    df["date"] = converted.dt.to_period("M").dt.to_timestamp()
    if date_col != "date":
        df.drop(columns=[date_col], inplace=True, errors="ignore")
    return df


def merge_and_process(raw_dir: str, processed_dir: str):
    """Merge all downloaded CSVs, forward-fill, resample to daily."""
    logger.info("Merging all macro sources...")

    os.makedirs(processed_dir, exist_ok=True)

    # Read all files
    dataframes = {}
    for fname in os.listdir(raw_dir):
        fpath = os.path.join(raw_dir, fname)
        if fname.endswith(".csv"):
            df = pd.read_csv(fpath)
        elif fname.endswith(".txt"):
            df = pd.read_table(fpath)
        else:
            continue
        dataframes[fname] = df
        logger.info(f"  Loaded {fname}: {df.shape}")

    if not dataframes:
        logger.error("No data files found to merge!")
        return

    # Unify date columns
    date_col_map = {
        "macro1_Monthly.txt": "DATE",
        "macro1_Quarterly.txt": "DATE",
        "lmci_feds.csv": "date",
        "ebp_csv.csv": "date",
        "FED_Note_Term_SOFR.csv": "DATE",
        "fci_g_public_monthly_3yr.csv": "date",
        "FEDS-Note-2873-cie-data.csv": "period",
        "DKW_updates.csv": "date",
    }

    for fname, df in dataframes.items():
        date_col = date_col_map.get(fname)
        if date_col and date_col in df.columns:
            dataframes[fname] = unify_dates(df, date_col)
        elif "date" in df.columns:
            dataframes[fname] = unify_dates(df, "date")
        elif "DATE" in df.columns:
            dataframes[fname] = unify_dates(df, "DATE")
        else:
            logger.warning(f"  No date column found in {fname}, skipping")

    # Cast non-date columns to numeric
    for fname, df in dataframes.items():
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Group mean by date (handles duplicate months)
    for fname in dataframes:
        df = dataframes[fname]
        if "date" in df.columns:
            dataframes[fname] = df.groupby("date").mean().reset_index()

    # Outer-merge all on date
    merged = None
    for fname, df in dataframes.items():
        if "date" not in df.columns:
            continue
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="date", how="outer")

    # Filter to 1990-01 through current (stock data starts 1993-12, most macro sources start 1990)
    merged = merged[(merged["date"] >= "1990-01") & (merged["date"] <= str(datetime.now().year + 1))]
    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Forward fill NaN, then backfill leading NaNs (matches data_preparation_2.ipynb)
    merged.ffill(inplace=True)
    merged.bfill(inplace=True)

    # Save monthly merged data
    merged.rename(columns={"date": "Date"}, inplace=True)
    merged.to_csv(os.path.join(processed_dir, "macro_data.csv"), index=False)
    logger.info(f"  Saved macro_data.csv: {merged.shape}")

    # Select only the 46 macro variables from macro_list.txt
    if os.path.exists(MACRO_LIST_PATH):
        with open(MACRO_LIST_PATH, "r") as f:
            macro_features = [line.strip() for line in f if line.strip()]
        logger.info(f"  Selecting {len(macro_features)} features from macro_list.txt")

        available = [c for c in macro_features if c in merged.columns]
        missing = [c for c in macro_features if c not in merged.columns]
        if missing:
            logger.warning(f"  Missing {len(missing)} features: {missing}")
        selected = merged[["Date"] + available].copy()
    else:
        logger.warning(f"  macro_list.txt not found at {MACRO_LIST_PATH}, using all columns")
        selected = merged.copy()

    # Resample to daily frequency (forward fill)
    selected["Date"] = pd.to_datetime(selected["Date"])
    selected.set_index("Date", inplace=True)
    daily = selected.resample("D").ffill()
    daily.reset_index(inplace=True)

    daily.to_csv(os.path.join(processed_dir, "macro_data_resampled.csv"), index=False)
    logger.info(f"  Saved macro_data_resampled.csv: {daily.shape}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"  Total columns (merged): {len(merged.columns) - 1}")
    logger.info(f"  Selected features:      {len(available) if 'available' in dir() else '?'}")
    if 'missing' in dir() and missing:
        logger.info(f"  Missing features:       {len(missing)}")
        for m in missing:
            logger.info(f"    - {m}")
    logger.info(f"  Date range:             {merged['Date'].min()} to {merged['Date'].max()}")
    logger.info(f"  Monthly rows:           {len(merged)}")
    logger.info(f"  Daily rows:             {len(daily)}")
    logger.info(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download all 46 macro variables for GAN conditioning")
    parser.add_argument("--fred-key", type=str, default="5ce3e0cd1cb5b158b0c0f000d194ebd4",
                        help="FRED API key")
    parser.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "datasets", "macro_raw"),
                        help="Directory for raw downloaded files")
    parser.add_argument("--processed-dir", type=str,
                        default=os.path.join(ROOT, "datasets", "macro_processed"),
                        help="Directory for merged/processed output")
    parser.add_argument("--skip-dkw", action="store_true",
                        help="Skip DKW download (use existing DKW_updates.csv if available)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Only download, don't merge/process")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Downloading all macro data sources")
    logger.info("=" * 60)

    # 1. FRED series
    logger.info("\n[1/3] FRED series (LMCI + 7 monthly + 2 quarterly = 10 vars)")
    download_fred(args.fred_key, args.output_dir)

    # 2. Fed Board direct CSVs
    logger.info("\n[2/3] Fed Board CSV downloads (EBP + FCI-G + SOFR + CIE = 15 vars)")
    for name, info in FED_SOURCES.items():
        try:
            download_fed_csv(name, info, args.output_dir)
        except requests.HTTPError as e:
            logger.error(f"  Failed to download {name}: {e}")
            logger.error(f"  URL: {info['url']}")

    # 3. DKW TIPS decomposition
    if not args.skip_dkw:
        logger.info("\n[3/3] DKW TIPS Yield Curve Decomposition (21 vars)")
        download_dkw(args.output_dir)
    else:
        # Copy existing DKW if available
        existing = os.path.join(ROOT, "datasets", "macro", "DKW_updates.csv")
        if os.path.exists(existing):
            import shutil
            dst = os.path.join(args.output_dir, "DKW_updates.csv")
            shutil.copy2(existing, dst)
            logger.info(f"  Copied existing DKW_updates.csv from {existing}")
        else:
            logger.warning("  --skip-dkw set but no existing DKW_updates.csv found")

    # 4. Merge and process
    if not args.skip_merge:
        logger.info("\n" + "=" * 60)
        logger.info("Processing & merging all sources")
        logger.info("=" * 60)
        merge_and_process(args.output_dir, args.processed_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
