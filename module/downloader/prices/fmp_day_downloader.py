import os
import pandas as pd
from module.registry import DOWNLOADER
from module.downloader.custom import Downloader
from module.utils.misc import generate_intervals
from tqdm.auto import tqdm
from datetime import datetime
import time
import signal
from pandas_market_calendars import get_calendar
import requests as _requests
import json
from dotenv import load_dotenv

load_dotenv(verbose=True)

class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Time out")

def get_jsonparsed_data(url, max_retries=5):
    for attempt in range(max_retries):
        r = _requests.get(url, timeout=60)
        if r.status_code in (429, 402):
            wait = 15 * (attempt + 1)
            print(f"  {r.status_code} rate-limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()

NYSE = get_calendar('XNYS')

@DOWNLOADER.register_module(force=True)
class FMPDayPriceDownloader(Downloader):
    def __init__(self,
                 root: str = "",
                 token: str = None,
                 delay: int = 1,
                 start_date: str = "2023-04-01",
                 end_date: str = "2023-04-01",
                 interval: str = "minute",
                 stocks_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 **kwargs):

        self.root = root
        self.token = token if token is not None else os.environ.get("OA_FMP_KEY")
        self.delay = delay
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.stocks_path = os.path.join(root, stocks_path)
        self.tag = tag
        self.workdir = os.path.join(root, workdir, tag)

        self.log_path = os.path.join(self.workdir, "{}.txt".format(tag))

        with open(self.log_path, "w") as op:
            op.write("")

        self.stocks = self._init_stocks()

        self.request_url_full = "https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={}&from={}&to={}&apikey={}"
        self.request_url_adj = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted?symbol={}&from={}&to={}&apikey={}"

        super().__init__(**kwargs)

    def _init_stocks(self):
        with open(self.stocks_path) as op:
            stocks = [line.strip() for line in op.readlines()]
        return stocks

    def check_download(self,
                       stocks = None,
                       start_date = None,
                       end_date = None):
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
                if os.path.exists(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d")))):
                    count += 1
                    total_count += 1
                stock_count += 1
                total_stock_count += 1

            if count != stock_count:
                failed_stocks.append(stock)

            print("{}: {}/{}".format(stock, count, stock_count))

        print("Total: {}/{}, failed {}/{}".format(total_count, total_stock_count, total_stock_count - total_count, total_stock_count))

        return failed_stocks

    def download(self,
                 stocks = None,
                 start_date = None,
                 end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        stocks = stocks if stocks else self.stocks

        intervals = generate_intervals(start_date, end_date, "year")

        for stock in stocks:

            os.makedirs(os.path.join(self.workdir, stock), exist_ok=True)

            df = pd.DataFrame()

            for (start, end) in tqdm(intervals, bar_format="Download {} Prices:".format(stock) + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):

                is_trading_day = NYSE.valid_days(start_date=start, end_date=end).size > 0
                if is_trading_day:
                    if os.path.exists(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d")))):
                        chunk_df = pd.read_csv(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d"))))
                    else:
                        chunk_df = {
                            "open": [],
                            "high": [],
                            "low": [],
                            "close": [],
                            "volume": [],
                            "timestamp": [],
                            "adjClose": [],
                            "change": [],
                            "changePercent": [],
                            "vwap": [],
                        }

                        url_full = self.request_url_full.format(
                            stock,
                            start.strftime("%Y-%m-%d"),
                            end.strftime("%Y-%m-%d"),
                            self.token)
                        url_adj = self.request_url_adj.format(
                            stock,
                            start.strftime("%Y-%m-%d"),
                            end.strftime("%Y-%m-%d"),
                            self.token)

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(120)

                        try:
                            time.sleep(self.delay)
                            aggs = get_jsonparsed_data(url_full)
                            if not isinstance(aggs, list):
                                aggs = []
                            time.sleep(self.delay)
                            adj_aggs = get_jsonparsed_data(url_adj)
                            if not isinstance(adj_aggs, list):
                                adj_aggs = []
                            signal.alarm(0)
                        except TimeoutException:
                            print("Time out")
                            aggs = []
                            adj_aggs = []

                        # Build adjClose lookup by date
                        adj_map = {}
                        for aa in adj_aggs:
                            adj_map[aa["date"]] = aa.get("adjClose", None)

                        if len(aggs) == 0:
                            with open(self.log_path, "a") as op:
                                op.write("{},{}\n".format(stock, start.strftime("%Y-%m-%d")))
                            continue

                        for a in aggs:

                            chunk_df["open"].append(a["open"])
                            chunk_df["high"].append(a["high"])
                            chunk_df["low"].append(a["low"])
                            chunk_df["close"].append(a["close"])
                            chunk_df["volume"].append(a["volume"])
                            chunk_df["timestamp"].append(a["date"])
                            chunk_df["adjClose"].append(adj_map.get(a["date"], a["close"]))
                            chunk_df["change"].append(a.get("change", 0))
                            chunk_df["changePercent"].append(a.get("changePercent", 0))
                            chunk_df["vwap"].append(a.get("vwap", 0))

                        chunk_df = pd.DataFrame(chunk_df,index=range(len(chunk_df["timestamp"])))
                        chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"]).apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

                        chunk_df.to_csv(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d"))), index=False)

                    df = pd.concat([df, chunk_df], axis=0)

            df = df.sort_values(by="timestamp", ascending=True)
            df.to_csv(os.path.join(self.workdir, "{}.csv".format(stock)), index=False)

if __name__ == '__main__':

    request_url = "https://financialmodelingprep.com/api/v3/stock_news?tickers=AAPL&page=218&apikey=yh3YH3bjNMYn8GkLf6saTvKu8sJo7397"
    aggs = get_jsonparsed_data(request_url)
    print(aggs)

