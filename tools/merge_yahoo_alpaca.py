"""
Merge Yahoo pre-2016 + Alpaca 2016-2024 daily data into a single CSV per ticker,
then combine all tickers into one merged_stock_data.csv matching the format
expected by the GAN preprocessing notebook.
"""
import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

YAHOO_DIR = ROOT / "workdir" / "yahoo_day_prices_dj30_pre2016"
ALPACA_DIR = ROOT / "workdir" / "alpaca_day_prices_dj30"
OUTPUT_DIR = ROOT / "workdir" / "merged_dj30"
TICKER_FILE = ROOT / "configs" / "_asset_list_" / "dj30_clean25.txt"

# Columns expected by downstream: Date, Open, High, Low, Close, Adj Close, Volume, ticker
FINAL_COLS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker"]


def load_yahoo(ticker: str) -> pd.DataFrame:
    path = YAHOO_DIR / f"{ticker}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"Adj Close": "Adj Close"})  # already correct
    df["ticker"] = ticker
    # Standardize column order
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df[FINAL_COLS]


def load_alpaca(ticker: str) -> pd.DataFrame:
    path = ALPACA_DIR / f"{ticker}.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["ticker"] = ticker
    return df[FINAL_COLS]


def main():
    with open(TICKER_FILE) as f:
        tickers = [line.strip() for line in f if line.strip()]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_dfs = []
    for ticker in tickers:
        yahoo_df = load_yahoo(ticker)
        alpaca_df = load_alpaca(ticker)

        # Yahoo ends 2015-12-31, Alpaca starts 2016-01-04 — no overlap expected
        # But verify and drop any overlap just in case
        yahoo_max_date = yahoo_df["Date"].max()
        alpaca_min_date = alpaca_df["Date"].min()

        # Keep Yahoo up to end of 2015, Alpaca from 2016 onward
        yahoo_df = yahoo_df[yahoo_df["Date"] < "2016-01-01"]
        alpaca_df = alpaca_df[alpaca_df["Date"] >= "2016-01-01"]

        merged = pd.concat([yahoo_df, alpaca_df], ignore_index=True)
        merged = merged.sort_values("Date").reset_index(drop=True)

        # Save per-ticker
        merged.to_csv(OUTPUT_DIR / f"{ticker}.csv", index=False)
        all_dfs.append(merged)

        print(f"{ticker}: Yahoo {len(yahoo_df)} + Alpaca {len(alpaca_df)} = {len(merged)} rows  "
              f"({merged['Date'].min()} to {merged['Date'].max()})")

    # Combine all tickers
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["ticker", "Date"]).reset_index(drop=True)

    output_path = OUTPUT_DIR / "merged_stock_data.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nCombined: {len(combined)} rows, {len(tickers)} tickers -> {output_path}")

    # Sanity checks
    print("\n--- Sanity Checks ---")
    for ticker in tickers:
        tdf = combined[combined["ticker"] == ticker]
        dates = pd.to_datetime(tdf["Date"])
        gaps = dates.diff().dt.days
        max_gap = gaps.max()
        nulls = tdf.isnull().sum().sum()
        print(f"  {ticker}: {len(tdf)} rows, {dates.min().date()} to {dates.max().date()}, "
              f"max_gap={max_gap} days, nulls={nulls}")


if __name__ == "__main__":
    main()
