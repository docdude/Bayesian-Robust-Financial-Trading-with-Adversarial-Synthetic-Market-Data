root = None
workdir = "workdir"
tag = "tiingo_day_prices_dj30"
batch_size = 5

downloader = dict(
    type = "TiingoDayPriceDownloader",
    root = root,
    token = None,
    start_date = "1993-12-01",
    end_date = "2024-12-31",
    interval = "1d",
    delay = 0.0,
    stocks_path = "configs/_asset_list_/dj30_clean28.txt",
    workdir = workdir,
    tag = tag,
)
