root = None
workdir = "workdir"
tag = "fmp_day_prices_dj30_pre2016"
batch_size = 1
max_workers = 1

downloader = dict(
    type = "FMPDayPriceDownloader",
    root = root,
    token = None,
    start_date = "1993-12-01",
    end_date = "2016-01-01",
    interval = "1d",
    delay = 3,
    stocks_path = "configs/_asset_list_/dj30_clean25.txt",
    workdir = workdir,
    tag = tag
)
