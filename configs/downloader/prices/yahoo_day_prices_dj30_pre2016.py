root = None
workdir = "workdir"
tag = "yahoo_day_prices_dj30_pre2016"
batch_size = 5
max_workers = 5

downloader = dict(
    type = "YahooFinanceDayPriceDownloader",
    root = root,
    token = "",
    start_date = "1993-12-01",
    end_date = "2016-01-01",
    interval = "day",
    delay = 2,
    stocks_path = "configs/_asset_list_/dj30_clean25.txt",
    workdir = workdir,
    tag = tag
)
