import os
import time
import pandas as pd
import requests
from tqdm.auto import tqdm
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(verbose=True)

from module.registry import DOWNLOADER
from module.downloader.custom import Downloader


@DOWNLOADER.register_module(force=True)
class TiingoDayPriceDownloader(Downloader):
    """Download daily adjusted OHLCV bars from Tiingo's EOD endpoint.

    Tiingo returns a single JSON array per ticker for the full date range
    with both raw and fully-adjusted (split + dividend) OHLCV. We keep the
    adjusted series as the canonical source so there are no splice artifacts
    across history. Output CSV columns match the project convention:
        Date, Open, High, Low, Close, Adj Close, Volume, ticker

    Requires TIINGO_API_KEY in env (or pass via `token=` in config).
    """

    BASE_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

    def __init__(self,
                 root: str = "",
                 token: str = None,
                 delay: float = 0.0,
                 start_date: str = "1993-12-01",
                 end_date: str = "2024-12-31",
                 interval: str = "1d",
                 stocks_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 max_retry: int = 3,
                 **kwargs):

        self.root = root
        self.api_key = token if token is not None else os.environ.get("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY not set (pass `token=` or export env var).")

        self.delay = delay
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.stocks_path = os.path.join(root, stocks_path)
        self.tag = tag
        self.workdir = os.path.join(root, workdir, tag)
        self.max_retry = max_retry

        os.makedirs(self.workdir, exist_ok=True)
        self.log_path = os.path.join(self.workdir, "{}.txt".format(tag))
        with open(self.log_path, "w") as op:
            op.write("")

        self.stocks = self._init_stocks()

        # Avoid passing proxy kwargs into parent — we don't need them for Tiingo
        super().__init__(use_proxy=None, max_retry=max_retry, **kwargs)

    def _init_stocks(self):
        with open(self.stocks_path) as op:
            stocks = [line.strip() for line in op.readlines()
                      if line.strip() and not line.strip().startswith('#')]
        return stocks

    def _ticker_csv_path(self, stock: str) -> str:
        return os.path.join(self.workdir, "{}.csv".format(stock))

    def check_download(self,
                       stocks=None,
                       start_date=None,
                       end_date=None):
        """Return the list of tickers still missing a per-ticker CSV."""
        stocks = stocks if stocks else self.stocks
        failed = []
        count = 0
        for stock in stocks:
            if os.path.exists(self._ticker_csv_path(stock)):
                count += 1
                print("{}: OK".format(stock))
            else:
                failed.append(stock)
                print("{}: MISSING".format(stock))
        print("Total: {}/{}, failed {}/{}".format(
            count, len(stocks), len(stocks) - count, len(stocks)))
        return failed

    def _fetch_ticker(self, stock: str) -> pd.DataFrame:
        url = self.BASE_URL.format(ticker=stock)
        params = {
            "startDate": self.start_date,
            "endDate": self.end_date,
            "format": "json",
            "resampleFreq": "daily",
            "token": self.api_key,
        }
        headers = {"Content-Type": "application/json"}

        last_err = None
        for attempt in range(1, self.max_retry + 1):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=60)
                if r.status_code == 429:
                    # Rate limited — back off and retry
                    wait = 2 ** attempt
                    print("Rate limited on {} (attempt {}), sleeping {}s".format(stock, attempt, wait))
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                if not data:
                    return pd.DataFrame()
                return pd.DataFrame(data)
            except Exception as e:
                last_err = e
                if attempt < self.max_retry:
                    time.sleep(2 ** attempt)
                    continue
                raise
        if last_err:
            raise last_err
        return pd.DataFrame()

    def download(self,
                 stocks=None,
                 start_date=None,
                 end_date=None):
        if start_date is not None:
            self.start_date = start_date
        if end_date is not None:
            self.end_date = end_date
        stocks = stocks if stocks else self.stocks

        for stock in tqdm(stocks, desc="Tiingo EOD"):
            out_path = self._ticker_csv_path(stock)
            if os.path.exists(out_path):
                continue

            try:
                raw = self._fetch_ticker(stock)
            except Exception as e:
                print("Error downloading {}: {}".format(stock, e))
                with open(self.log_path, "a") as op:
                    op.write("{},{}\n".format(stock, str(e)))
                continue

            if raw.empty:
                print("No data returned for {}".format(stock))
                with open(self.log_path, "a") as op:
                    op.write("{},empty\n".format(stock))
                continue

            # Tiingo fields: date, open, high, low, close, volume,
            # adjOpen, adjHigh, adjLow, adjClose, adjVolume, divCash, splitFactor
            df = pd.DataFrame()
            df["Date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
            df["Open"] = raw["adjOpen"]
            df["High"] = raw["adjHigh"]
            df["Low"] = raw["adjLow"]
            df["Close"] = raw["adjClose"]
            df["Adj Close"] = raw["adjClose"]
            df["Volume"] = raw["adjVolume"]
            df["ticker"] = stock

            df = df.sort_values("Date").reset_index(drop=True)
            df.to_csv(out_path, index=False)

            if self.delay > 0:
                time.sleep(self.delay)
