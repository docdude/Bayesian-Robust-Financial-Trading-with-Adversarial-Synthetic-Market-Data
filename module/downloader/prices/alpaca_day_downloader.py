import os
import pandas as pd
from module.registry import DOWNLOADER
from module.downloader.custom import Downloader
from module.utils.misc import generate_intervals
from tqdm.auto import tqdm
from datetime import datetime
import time
from pandas_market_calendars import get_calendar
from dotenv import load_dotenv

load_dotenv(verbose=True)

NYSE = get_calendar('XNYS')


@DOWNLOADER.register_module(force=True)
class AlpacaDayPriceDownloader(Downloader):
    """Download daily OHLCV bars from Alpaca SIP consolidated feed.

    Follows the same per-ticker, per-year CSV caching pattern as FMPDayPriceDownloader.
    Requires ALPACA_API_KEY and ALPACA_API_SECRET env vars (or pass via config).
    """

    def __init__(self,
                 root: str = "",
                 token: str = None,
                 api_secret: str = None,
                 delay: int = 0,
                 start_date: str = "1993-12-01",
                 end_date: str = "2024-01-01",
                 interval: str = "1d",
                 stocks_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 feed: str = "sip",
                 **kwargs):

        self.root = root
        self.api_key = token if token is not None else os.environ.get("ALPACA_API_KEY_PAID")
        self.api_secret = api_secret if api_secret is not None else os.environ.get("ALPACA_API_SECRET_PAID")
        self.delay = delay
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.stocks_path = os.path.join(root, stocks_path)
        self.tag = tag
        self.workdir = os.path.join(root, workdir, tag)
        self.feed = feed

        os.makedirs(self.workdir, exist_ok=True)
        self.log_path = os.path.join(self.workdir, "{}.txt".format(tag))

        with open(self.log_path, "w") as op:
            op.write("")

        self.stocks = self._init_stocks()

        # Lazy import to avoid import errors if alpaca-py isn't installed
        self._client = None

        super().__init__(**kwargs)

    @property
    def client(self):
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(self.api_key, self.api_secret)
        return self._client

    def _init_stocks(self):
        with open(self.stocks_path) as op:
            stocks = [line.strip() for line in op.readlines() if line.strip() and not line.strip().startswith('#')]
        return stocks

    def check_download(self,
                       stocks=None,
                       start_date=None,
                       end_date=None):
        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")
        stocks = stocks if stocks else self.stocks

        intervals = generate_intervals(start_date, end_date, "year")

        failed_stocks = []
        total_count = 0
        total_stock_count = 0

        for stock in stocks:
            count = 0
            stock_count = 0

            for (start, end) in intervals:
                csv_path = os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d")))
                if os.path.exists(csv_path):
                    count += 1
                    total_count += 1
                stock_count += 1
                total_stock_count += 1

            if count != stock_count:
                failed_stocks.append(stock)

            print("{}: {}/{}".format(stock, count, stock_count))

        print("Total: {}/{}, failed {}/{}".format(
            total_count, total_stock_count,
            total_stock_count - total_count, total_stock_count))

        return failed_stocks

    def download(self,
                 stocks=None,
                 start_date=None,
                 end_date=None):
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed

        feed_map = {"sip": DataFeed.SIP, "iex": DataFeed.IEX}
        data_feed = feed_map.get(self.feed.lower(), DataFeed.SIP)

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")
        stocks = stocks if stocks else self.stocks

        intervals = generate_intervals(start_date, end_date, "year")

        for stock in stocks:
            os.makedirs(os.path.join(self.workdir, stock), exist_ok=True)

            df = pd.DataFrame()

            for (start, end) in tqdm(
                intervals,
                bar_format="Download {} Prices:".format(stock) + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"
            ):
                csv_path = os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d")))

                is_trading_day = NYSE.valid_days(start_date=start, end_date=end).size > 0
                if not is_trading_day:
                    continue

                if os.path.exists(csv_path):
                    chunk_df = pd.read_csv(csv_path)
                else:
                    try:
                        if self.delay > 0:
                            time.sleep(self.delay)

                        request_params = StockBarsRequest(
                            symbol_or_symbols=stock,
                            timeframe=TimeFrame.Day,
                            start=start,
                            end=end,
                            feed=data_feed,
                        )
                        bars = self.client.get_stock_bars(request_params).df

                        if bars.empty:
                            with open(self.log_path, "a") as op:
                                op.write("{},{}\n".format(stock, start.strftime("%Y-%m-%d")))
                            continue

                        bars = bars.reset_index()

                        # Normalize column names to match FMP output format
                        col_map = {}
                        if "symbol" in bars.columns:
                            col_map["symbol"] = "ticker"
                        if "timestamp" in bars.columns:
                            col_map["timestamp"] = "Date"

                        bars = bars.rename(columns=col_map)

                        # Rename to standard OHLCV columns matching the merged CSV format
                        rename = {
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        }
                        bars = bars.rename(columns=rename)

                        # Format Date as string
                        if "Date" in bars.columns:
                            bars["Date"] = pd.to_datetime(bars["Date"]).dt.strftime("%Y-%m-%d")

                        # Alpaca daily bars don't have a separate Adj Close — close IS adjusted
                        if "Adj Close" not in bars.columns:
                            bars["Adj Close"] = bars["Close"]

                        # Keep columns consistent
                        keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker",
                                                  "vwap", "trade_count"] if c in bars.columns]
                        chunk_df = bars[keep_cols]

                        chunk_df.to_csv(csv_path, index=False)

                    except Exception as e:
                        print("Error downloading {} for {}: {}".format(stock, start.strftime("%Y-%m-%d"), e))
                        with open(self.log_path, "a") as op:
                            op.write("{},{},{}\n".format(stock, start.strftime("%Y-%m-%d"), str(e)))
                        continue

                df = pd.concat([df, chunk_df], axis=0)

            if not df.empty:
                df = df.sort_values(by="Date", ascending=True)
                df.to_csv(os.path.join(self.workdir, "{}.csv".format(stock)), index=False)
