root = None
workdir = "workdir"
tag = "alpaca_day_prices_dj30"
batch_size = 5

downloader = dict(
    type = "AlpacaDayPriceDownloader",
    root = root,
    token = None,
    api_secret = None,
    start_date = "1993-12-01",
    end_date = "2024-01-01",
    interval = "1d",
    delay = 0,
    stocks_path = "configs/_asset_list_/dj30_clean25.txt",
    workdir = workdir,
    tag = tag,
    feed = "sip",
)
