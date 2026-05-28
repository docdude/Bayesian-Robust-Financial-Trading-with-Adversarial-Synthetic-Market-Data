"""
Download all 46 macro variables needed for the GAN macro conditioning.

Sources:
  1. FRED (9 vars)         — via fredapi
  2. Fed EBP (3 vars)      — direct CSV from Fed Board
  3. Fed FCI-G (7 vars)    — direct CSV from Fed Board
    4. SOFR proxy (3 vars)   — NY Fed primary-dealer repo survey + indicative/official SOFR
  5. Fed CIE (2 vars)      — direct CSV from Fed Board
  6. Fed LMCI (1 var)      — via FRED (FRBLMCI, discontinued 2017)
  7. Fed DKW (21 vars)     — direct CSV from Fed Board (TIPS decomposition)

Output: datasets/macro_raw/  (individual source CSVs)
        datasets/macro_processed/macro_data.csv  (merged, monthly, forward-filled only)
        datasets/macro_processed/macro_data_resampled.csv  (daily, forward-filled only)

Usage:
  python tools/download_all_macros.py [--fred-key YOUR_KEY] [--output-dir datasets/macro_raw]
"""

import os
import sys
import io
import argparse
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta

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
        "desc": "Proxy-based realized SOFR averages (3 vars)",
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

T10YIEM_DKW_PROXY_COLS = [
    "exp.inflation.10",
    "inflation.risk.prem.10",
    "tips.liq.prem.10",
]

SOFR_PROXY_SOURCES = {
    "primary_dealer_survey": "https://www.newyorkfed.org/medialibrary/media/markets/HistoricalOvernightTreasGCRepoPriDealerSurvRate.xlsx",
    "indicative_repo_rates": "https://www.newyorkfed.org/medialibrary/media/markets/Data%20Release.xlsx",
    "official_sofr": "https://markets.newyorkfed.org/read?productCode=50&eventCodes=520&limit=10000&startPosition=0&sort=postDt:1&format=csv",
}

XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def xlsx_column_index(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def read_xlsx_rows(content: bytes, sheet_name: str | None = None) -> list[list[object]]:
    """Read rows from a simple XLSX worksheet without requiring openpyxl."""
    with zipfile.ZipFile(io.BytesIO(content)) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", XLSX_NS):
                parts = [node.text or "" for node in item.findall(".//main:t", XLSX_NS)]
                shared_strings.append("".join(parts))

        workbook_xml = ET.fromstring(workbook.read("xl/workbook.xml"))
        rels_xml = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        rels = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_xml.findall("pkgrel:Relationship", XLSX_NS)}

        selected_target = None
        for sheet in workbook_xml.findall("main:sheets/main:sheet", XLSX_NS):
            name = sheet.attrib["name"]
            if sheet_name is None or name == sheet_name:
                rel_id = sheet.attrib[f"{{{XLSX_NS['rel']}}}id"]
                selected_target = rels[rel_id]
                break

        if selected_target is None:
            raise ValueError(f"Sheet not found: {sheet_name}")

        if not selected_target.startswith("worksheets/"):
            selected_target = "worksheets/" + selected_target.split("/")[-1]
        sheet_xml = ET.fromstring(workbook.read("xl/" + selected_target))

        rows = []
        for row in sheet_xml.findall(".//main:sheetData/main:row", XLSX_NS):
            values = []
            for cell in row.findall("main:c", XLSX_NS):
                col_index = xlsx_column_index(cell.attrib["r"])
                while len(values) <= col_index:
                    values.append(None)

                value = None
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//main:t", XLSX_NS))
                else:
                    raw_value = cell.find("main:v", XLSX_NS)
                    if raw_value is not None:
                        value = raw_value.text
                        if cell_type == "s":
                            value = shared_strings[int(value)]
                values[col_index] = value
            rows.append(values)

    return rows


def excel_serial_to_datetime(value: object) -> pd.Timestamp:
    return pd.Timestamp(datetime(1899, 12, 30) + timedelta(days=float(value)))


def download_primary_dealer_sofr_proxy() -> pd.DataFrame:
    """NY Fed primary-dealer overnight Treasury GC repo survey proxy, available from 1998."""
    url = SOFR_PROXY_SOURCES["primary_dealer_survey"]
    logger.info("  SOFR proxy: downloading NY Fed primary-dealer Treasury GC repo survey")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    rows = read_xlsx_rows(response.content)
    records = []
    for row in rows:
        if len(row) < 2 or row[0] in (None, "") or row[1] in (None, ""):
            continue
        try:
            records.append((excel_serial_to_datetime(row[0]), float(row[1])))
        except (TypeError, ValueError):
            continue

    df = pd.DataFrame(records, columns=["DATE", "SOFR_PROXY"])
    df["source_priority"] = 1
    df["source"] = "primary_dealer_survey"
    logger.info(f"  SOFR proxy: primary-dealer survey {df['DATE'].min().date()} to {df['DATE'].max().date()} ({len(df)} rows)")
    return df


def download_indicative_sofr() -> pd.DataFrame:
    """NY Fed pre-production indicative SOFR, available from 2014-08 through 2018-03."""
    url = SOFR_PROXY_SOURCES["indicative_repo_rates"]
    logger.info("  SOFR proxy: downloading NY Fed indicative TGCR/BGCR/SOFR release")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    rows = read_xlsx_rows(response.content, sheet_name="VWM Rates")
    records = []
    for row in rows:
        if len(row) < 4 or row[0] in (None, "") or row[3] in (None, ""):
            continue
        try:
            date = excel_serial_to_datetime(row[0])
            rate_percent = float(row[3]) / 100.0  # Workbook reports basis points.
            records.append((date, rate_percent))
        except (TypeError, ValueError):
            continue

    df = pd.DataFrame(records, columns=["DATE", "SOFR_PROXY"])
    df["source_priority"] = 2
    df["source"] = "indicative_sofr"
    logger.info(f"  SOFR proxy: indicative SOFR {df['DATE'].min().date()} to {df['DATE'].max().date()} ({len(df)} rows)")
    return df


def download_official_sofr() -> pd.DataFrame:
    """Official NY Fed SOFR, available from 2018-04 onward."""
    url = SOFR_PROXY_SOURCES["official_sofr"]
    logger.info("  SOFR proxy: downloading official NY Fed SOFR")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df = df[df["Rate Type"].eq("SOFR")].copy()
    df["DATE"] = pd.to_datetime(df["Effective Date"], format="%m/%d/%Y", errors="coerce")
    df["SOFR_PROXY"] = pd.to_numeric(df["Rate (%)"], errors="coerce")
    df = df.dropna(subset=["DATE", "SOFR_PROXY"])[["DATE", "SOFR_PROXY"]]
    df["source_priority"] = 3
    df["source"] = "official_sofr"
    logger.info(f"  SOFR proxy: official SOFR {df['DATE'].min().date()} to {df['DATE'].max().date()} ({len(df)} rows)")
    return df


def compound_realized_average(rate_percent: pd.Series, window_days: int) -> pd.Series:
    daily_growth = 1.0 + rate_percent.astype(float) / 100.0 / 360.0
    compounded = daily_growth.rolling(window_days, min_periods=window_days).apply(np.prod, raw=True)
    return (compounded - 1.0) * 360.0 / window_days * 100.0


def build_sofr_proxy_realized_averages() -> pd.DataFrame:
    """
    Build no-lookahead REALIZED_1M/3M/6M from a daily SOFR-like rate stack.

    Source priority is official SOFR > NY Fed indicative SOFR > NY Fed primary-dealer
    overnight Treasury GC repo survey, matching the Fed's recommended historical proxy.
    """
    primary_dealer = download_primary_dealer_sofr_proxy()
    indicative = download_indicative_sofr()
    official = download_official_sofr()

    indicative_start = indicative["DATE"].min()
    official_start = official["DATE"].min()
    primary_dealer = primary_dealer[primary_dealer["DATE"] < indicative_start]
    indicative = indicative[
        (indicative["DATE"] >= indicative_start)
        & (indicative["DATE"] < official_start)
    ]

    daily_sources = [primary_dealer, indicative, official]
    daily = pd.concat(daily_sources, ignore_index=True)
    daily = daily.sort_values(["DATE", "source_priority"]).drop_duplicates("DATE", keep="last")
    daily = daily.sort_values("DATE").reset_index(drop=True)
    observation_dates = daily["DATE"].copy()

    logger.info("  SOFR proxy: chosen source segments")
    for source, group in daily.groupby("source", sort=False):
        logger.info(f"    {source}: {group['DATE'].min().date()} to {group['DATE'].max().date()} ({len(group)} rows)")

    calendar = pd.DataFrame({"DATE": pd.date_range(daily["DATE"].min(), daily["DATE"].max(), freq="D")})
    calendar = calendar.merge(daily[["DATE", "SOFR_PROXY"]], on="DATE", how="left")
    calendar["SOFR_PROXY"] = calendar["SOFR_PROXY"].ffill()
    calendar["REALIZED_1M"] = compound_realized_average(calendar["SOFR_PROXY"], 30)
    calendar["REALIZED_3M"] = compound_realized_average(calendar["SOFR_PROXY"], 90)
    calendar["REALIZED_6M"] = compound_realized_average(calendar["SOFR_PROXY"], 180)

    realized_cols = ["REALIZED_1M", "REALIZED_3M", "REALIZED_6M"]
    observed = calendar[calendar["DATE"].isin(observation_dates)]
    monthly = observed.set_index("DATE")[realized_cols].resample("MS").mean(numeric_only=True).reset_index()
    monthly = monthly.dropna(subset=realized_cols, how="all")
    logger.info(
        "  SOFR proxy: realized monthly averages "
        f"{monthly['DATE'].min().date()} to {monthly['DATE'].max().date()} ({len(monthly)} rows)"
    )
    return monthly


def download_fed_csv(name: str, info: dict, output_dir: str) -> pd.DataFrame:
    """Download a Fed Board CSV and save locally."""
    logger.info(f"  Downloading {name}: {info['desc']}")

    if name == "FED_Note_Term_SOFR.csv":
        df = build_sofr_proxy_realized_averages()
        outpath = os.path.join(output_dir, name)
        df.to_csv(outpath, index=False)
        logger.info(f"  -> {name}  ({len(df)} rows, {len(df.columns)} cols)")
        return df

    r = requests.get(info["url"], timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text), skiprows=info.get("skiprows", 0))

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


def build_t10yiem_dkw_proxy(dkw: pd.DataFrame) -> pd.DataFrame:
    """Build monthly 10-year inflation compensation proxy from DKW components."""
    missing = [col for col in T10YIEM_DKW_PROXY_COLS if col not in dkw.columns]
    if missing:
        raise ValueError(f"Missing DKW columns for T10YIEM proxy: {missing}")

    proxy = dkw[["date"] + T10YIEM_DKW_PROXY_COLS].copy()
    for col in T10YIEM_DKW_PROXY_COLS:
        proxy[col] = pd.to_numeric(proxy[col], errors="coerce")
    proxy["T10YIEM_DKW_PROXY"] = (
        proxy["exp.inflation.10"]
        + proxy["inflation.risk.prem.10"]
        - proxy["tips.liq.prem.10"]
    )
    return proxy[["date", "T10YIEM_DKW_PROXY"]].dropna()


def fill_t10yiem_with_dkw_proxy(
    monthly: pd.DataFrame,
    dkw: pd.DataFrame,
) -> tuple[pd.DataFrame, int, pd.Timestamp | None, pd.Timestamp | None]:
    """Fill missing pre-FRED T10YIEM values with the DKW proxy only."""
    if "date" not in monthly.columns or "T10YIEM" not in monthly.columns:
        return monthly, 0, None, None
    if "date" not in dkw.columns:
        return monthly, 0, None, None

    real_t10 = pd.to_numeric(monthly["T10YIEM"], errors="coerce")
    first_real_date = monthly.loc[real_t10.notna(), "date"].min()
    if pd.isna(first_real_date):
        return monthly, 0, None, None

    proxy = build_t10yiem_dkw_proxy(dkw)
    patched = monthly.merge(proxy, on="date", how="left")
    fill_mask = (
        patched["date"].lt(first_real_date)
        & pd.to_numeric(patched["T10YIEM"], errors="coerce").isna()
        & patched["T10YIEM_DKW_PROXY"].notna()
    )
    fill_count = int(fill_mask.sum())
    if fill_count:
        patched.loc[fill_mask, "T10YIEM"] = patched.loc[
            fill_mask,
            "T10YIEM_DKW_PROXY",
        ]
        start = patched.loc[fill_mask, "date"].min()
        end = patched.loc[fill_mask, "date"].max()
    else:
        start = end = None

    patched = patched.drop(columns=["T10YIEM_DKW_PROXY"])
    return patched, fill_count, start, end


def patch_t10yiem_raw_with_dkw_proxy(output_dir: str) -> None:
    """Patch macro1_Monthly.txt with DKW proxy values before FRED T10YIEM starts."""
    macro_path = os.path.join(output_dir, "macro1_Monthly.txt")
    dkw_path = os.path.join(output_dir, "DKW_updates.csv")
    if not os.path.exists(macro_path) or not os.path.exists(dkw_path):
        logger.warning("  T10YIEM proxy skipped: macro1_Monthly.txt or DKW missing")
        return

    monthly = pd.read_table(macro_path)
    dkw = pd.read_csv(dkw_path)
    monthly = unify_dates(monthly, "DATE")
    dkw = unify_dates(dkw, "date")

    for df in (monthly, dkw):
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

    monthly = monthly.groupby("date").mean().reset_index()
    dkw = dkw.groupby("date").mean().reset_index()
    patched, fill_count, start, end = fill_t10yiem_with_dkw_proxy(monthly, dkw)

    if fill_count:
        out = patched.rename(columns={"date": "DATE"})
        out.to_csv(macro_path, sep="\t", index=False)
        logger.info(
            "  T10YIEM proxy: filled "
            f"{fill_count} monthly rows from {start.date()} to {end.date()}"
        )
    else:
        logger.info("  T10YIEM proxy: no missing pre-FRED rows to fill")


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

    if "macro1_Monthly.txt" in dataframes and "DKW_updates.csv" in dataframes:
        monthly, fill_count, start, end = fill_t10yiem_with_dkw_proxy(
            dataframes["macro1_Monthly.txt"],
            dataframes["DKW_updates.csv"],
        )
        dataframes["macro1_Monthly.txt"] = monthly
        if fill_count:
            logger.info(
                "  T10YIEM proxy: filled "
                f"{fill_count} monthly rows from {start.date()} to {end.date()}"
            )

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

    # Forward fill only. Leading NaNs are left missing to avoid using future values.
    merged.ffill(inplace=True)

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

    patch_t10yiem_raw_with_dkw_proxy(args.output_dir)

    # 4. Merge and process
    if not args.skip_merge:
        logger.info("\n" + "=" * 60)
        logger.info("Processing & merging all sources")
        logger.info("=" * 60)
        merge_and_process(args.output_dir, args.processed_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
