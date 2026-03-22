# BRFT Repository Guide

Complete breakdown of every runnable module in the repo, with commands and expected outputs.

> **Prerequisites:** All commands assume you're in the repo root with the venv activated:
> ```bash
> cd Bayesian-Robust-Financial-Trading-with-Adversarial-Synthetic-Market-Data
> source .venv/Scripts/activate   # Windows/Git Bash
> # or: .venv\Scripts\activate    # Windows CMD
> ```

---

## Table of Contents

1. [Data Download (Prices)](#1-data-download-prices)
2. [Data Download (News)](#2-data-download-news)
3. [Data Download (Sentiment)](#3-data-download-sentiment)
4. [Data Download (Economic Indicators)](#4-data-download-economic-indicators)
5. [Data Download (Macro from FRED)](#5-data-download-macro-from-fred)
6. [Data Download (Expert Knowledge)](#6-data-download-expert-knowledge)
7. [Stock Info Lookup](#7-stock-info-lookup)
8. [Data Processing (Feature Engineering)](#8-data-processing-feature-engineering)
9. [GAN Training (Synthetic Data Generation)](#9-gan-training-synthetic-data-generation)
10. [RL Trading — DQN](#10-rl-trading--dqn)
11. [RL Trading — PPO](#11-rl-trading--ppo)
12. [RL Trading — SAC](#12-rl-trading--sac)
13. [RL Inference & Metrics](#13-rl-inference--metrics)
14. [RL Trading Plots](#14-rl-trading-plots)
15. [Rule-Based Strategy Backtesting](#15-rule-based-strategy-backtesting)
16. [Time Series Forecasting](#16-time-series-forecasting)
17. [Data Analysis — Qualitative](#17-data-analysis--qualitative)
18. [Data Analysis — Quantitative](#18-data-analysis--quantitative)
19. [Metrics Module](#19-metrics-module)
20. [Pipeline Summary](#20-pipeline-summary)

---

## 1. Data Download (Prices)

**Script:** `tools/download_prices.py`
**Purpose:** Download OHLCV price data from Yahoo Finance, FMP, or Polygon.
**API Keys:** Yahoo = none needed. FMP = `OA_FMP_KEY` env var. Polygon = `OA_POLYGON_KEY` env var.

```bash
# Download 18 future ETFs from Yahoo Finance (no API key needed)
python tools/download_prices.py \
  --config configs/downloader/prices/yahoofinance_day_prices_future_etfs.py

# Download DJ30 stocks from Yahoo
python tools/download_prices.py \
  --config configs/downloader/prices/yahoofinance_day_prices_dj30.py

# Download DJ30 from FMP (requires OA_FMP_KEY)
python tools/download_prices.py \
  --config configs/downloader/prices/fmp_day_prices_dj30.py

# Download with multiprocessing
python tools/download_prices.py \
  --config configs/downloader/prices/yahoofinance_day_prices_futures.py \
  --batch_size 4
```

**Available configs:** (`configs/downloader/prices/`)
| Config | Source | Assets |
|--------|--------|--------|
| `yahoofinance_day_prices_future_etfs.py` | Yahoo | 18 commodity/currency ETFs |
| `yahoofinance_day_prices_futures.py` | Yahoo | US futures |
| `yahoofinance_day_prices_dj30.py` | Yahoo | Dow Jones 30 |
| `fmp_day_prices_dj30.py` | FMP | Dow Jones 30 |
| `fmp_day_prices_exp_stocks.py` | FMP | Experimental stocks |
| `fmp_day_prices_exp_cryptos.py` | FMP | Crypto pairs |
| `fmp_day_prices_exp_forexs.py` | FMP | Forex pairs |
| `fmp_day_prices_sp500.py` | FMP | S&P 500 |
| `polygon_day_prices_dj30.py` | Polygon | Dow Jones 30 |

**Output:** `workdir/{tag}/{SYMBOL}.csv` — one CSV per symbol with date, open, high, low, close, volume columns.

**Useful for:** Getting raw price data for any asset class. Yahoo configs require no API key. Start here.

---

## 2. Data Download (News)

**Script:** `tools/download_news.py`
**Purpose:** Download financial news articles from FMP, Polygon, or Yahoo Finance.
**API Keys:** FMP = `OA_FMP_KEY`. Polygon = `OA_POLYGON_KEY`. Yahoo = none.

```bash
# Download news for experimental stocks (FMP)
python tools/download_news.py \
  --config configs/downloader/news/fmp_news_exp_stocks.py

# Download crypto news (FMP)
python tools/download_news.py \
  --config configs/downloader/news/fmp_news_exp_cryptos.py

# Download DJ30 news from Yahoo (no API key)
python tools/download_news.py \
  --config configs/downloader/news/yahoofinance_news_dj30.py
```

**Available configs:** (`configs/downloader/news/`)
| Config | Source | Assets |
|--------|--------|--------|
| `fmp_news_dj30.py` | FMP | DJ30 |
| `fmp_news_exp_stocks.py` | FMP | Experimental stocks |
| `fmp_news_exp_cryptos.py` | FMP | Crypto |
| `fmp_news_exp_forexs.py` | FMP | Forex |
| `polygon_news_dj30.py` | Polygon | DJ30 |
| `polygon_news_exp.py` | Polygon | Experimental |
| `yahoofinance_news_dj30.py` | Yahoo | DJ30 |
| `yahoofinance_news_exp.py` | Yahoo | Experimental |

**Output:** `workdir/{tag}/{SYMBOL}.csv` — columns: timestamp, title, text, source, url, image.

**Status:** ⚠️ **Downloaded but not integrated** into the training pipeline. News data sits on disk — the processor never merges it into feature parquets. Useful only if you build your own NLP feature pipeline on top.

---

## 3. Data Download (Sentiment)

**Script:** `tools/download_sentiment.py`
**Purpose:** Download social media sentiment from FMP (StockTwits + Twitter metrics).
**API Key:** `OA_FMP_KEY` env var required.

```bash
python tools/download_sentiment.py \
  --config configs/downloader/tools/fmp_sentiment_exp.py
```

**Output:** `workdir/fmp_sentiment_exp/{SYMBOL}.csv` — columns: stocktwitsPosts, twitterPosts, stocktwitsComments, twitterComments, stocktwitsLikes, twitterLikes, stocktwitsImpressions, twitterImpressions, stocktwitsSentiment, twitterSentiment.

**Status:** ⚠️ **Downloaded but not integrated.** `cal_sentiment()` aggregation function exists in `module/processor/processor.py` but is never called. Data never reaches the RL agents or forecasting models.

---

## 4. Data Download (Economic Indicators)

**Script:** `tools/download_economic.py`
**Purpose:** Download economic indicator data from FMP.
**API Key:** `OA_FMP_KEY` env var required.

```bash
python tools/download_economic.py \
  --config configs/downloader/tools/fmp_economic_exp.py
```

**Output:** `workdir/fmp_economic_exp/` — economic indicator CSVs.

**Status:** ⚠️ **Downloaded but not integrated** into the feature pipeline.

---

## 5. Data Download (Macro from FRED)

**Script:** `tools/download_macro.py`
**Purpose:** Download macroeconomic data from the Federal Reserve (FRED).
**API Key:** Hardcoded FRED API key in script.

```bash
python tools/download_macro.py
```

**No arguments.** Downloads 5 series:
| Series | Description |
|--------|-------------|
| FEDFUNDS | Federal Funds Rate |
| CPIAUCSL | Consumer Price Index |
| PAYEMS | Nonfarm Payrolls |
| GDP | Gross Domestic Product |
| NETFI | Net Foreign Investment |

**Output:** `datasets/macro/{SERIES}.csv`

**Status:** ⚠️ **Downloaded but not integrated.** Same gap as sentiment/news.

---

## 6. Data Download (Expert Knowledge)

**Script:** `tools/download_expert_knowledege.py`
**Purpose:** Download analyst reports from Seeking Alpha via RapidAPI.
**API Key:** RapidAPI key required.

```bash
python tools/download_expert_knowledege.py \
  --config configs/downloader/tools/rapidapi_seekingalpha_analysis_exp.py \
  --quota 5
```

**Extra args:**
- `--quota` (int, default: 5) — download quota per symbol
- `--tool` (str, default: "seekingalpha_analysis") — tool name

**Output:** `workdir/rapidapi_seekingalpha_analysis_exp/{SYMBOL}.csv`

**Status:** ⚠️ **Downloaded but not integrated.**

---

## 7. Stock Info Lookup

**Script:** `tools/get_stock_infos.py`
**Purpose:** Fetch company profiles from FMP for DJ30 stocks.
**API Key:** `OA_FMP_KEY` env var.

```bash
python tools/get_stock_infos.py
```

**No arguments.** Hardcoded to DJ30.

**Output:** `dj30.json` — company name, sector, industry, description, etc.

**Useful for:** Quick reference. Minimal utility for trading pipeline.

---

## 8. Data Processing (Feature Engineering)

**Script:** `tools/data_process.py`
**Purpose:** Transform raw OHLCV prices → 150+ technical indicator features. This is the critical step that creates the parquet files used by all downstream tasks.

```bash
# Process future ETFs (what we've been using)
python tools/data_process.py \
  --config configs/processor/processor_day_future_etfs.py

# Process DJ30
python tools/data_process.py \
  --config configs/processor/processor_day_dj30.py

# Process experimental stocks
python tools/data_process.py \
  --config configs/processor/processor_day_exp_stocks.py
```

**Available configs:** (`configs/processor/`)
| Config | Asset Class |
|--------|-------------|
| `processor_day_future_etfs.py` | 18 commodity/currency ETFs |
| `processor_day_dj30.py` | Dow Jones 30 stocks |
| `processor_day_exp_stocks.py` | Experimental stocks |
| `processor_day_exp_cryptos.py` | Crypto pairs |
| `processor_day_exp_forexs.py` | Forex pairs |
| `processor_day_futures.py` | US futures |

**Output:** `datasets/processd_day_{asset_class}/features/{SYMBOL}.parquet` — ~155 columns per symbol:
- Raw prices: open, high, low, close, adj_close
- Keltner/candlestick: kmid, kmid2, klen, kup, kup2, klow, klow2, ksft, ksft2
- Momentum: roc_5/10/20/30/60
- Moving averages: ma_5/10/20/30/60
- Volatility: std_5/10/20/30/60
- Beta: beta_5/10/20/30/60
- Extremes: max, min, qtlu, qtld, rank, imax, imin, imxd (each at 5/10/20/30/60)
- Stochastic: rsv_5/10/20/30/60
- Counting: cntp, cntn, cntd (each at 5/10/20/30/60)
- Correlation: corr, cord (each at 5/10/20/30/60)
- Summation: sump, sumn, sumd (each at 5/10/20/30/60)
- Volume: vma, vstd, wvma, vsump, vsumn, vsumd (each at 5/10/20/30/60), log_volume
- Labels: ret1 (1-day return), mov1 (binary movement)

**Useful for:** Essential step. Must run after downloading prices and before any training.

---

## 9. GAN Training (Synthetic Data Generation)

**Script:** `generator/GRT_GAN/main.py`
**Purpose:** Train a TimeGAN to generate synthetic market data for RL data augmentation.

```bash
# Train GAN on processed ETF data
python generator/GRT_GAN/main.py --exp etf_18_120

# Quick smoke test (10 epochs)
python generator/GRT_GAN/main.py --exp etf_18_120_test --epochs 10
```

**Prerequisites:**
- Processed data must exist at `generator/GRT_GAN/datasets/output_data/` (generated by preprocessing notebooks or manually prepared)
- Files needed: `output_adj_factor.npy`, `output_data.npy`, `output_label.npy`, `output_time.npy`, `ticker_list.npy`

**Output:**
- Trained model: `generator/GRT_GAN/output/{exp}/` (checkpoint files)
- TensorBoard logs: `generator/GRT_GAN/tensorboard/`
- Evaluation metrics: feature prediction, one-step-ahead prediction, re-identification score

**Related notebooks:**
- `GeneratorAPI_tuning.ipynb` — hyperparameter tuning
- `visualization.ipynb` — compare real vs synthetic data

**Useful for:** Required only if using `generator_adv_agent` or `generator_noise` augmentation in DQN training. The GAN generates synthetic market sequences that the adversarial agent uses to create worst-case training scenarios.

---

## 10. RL Trading — DQN

**Script:** `downstream_tasks/rl/trading/dqn/train.py`
**Purpose:** Train a Deep Q-Network trading agent with optional data augmentation, NFSP, and quantile belief.

```bash
# Train with full config (1M steps, all features enabled)
python downstream_tasks/rl/trading/dqn/train.py \
  --config downstream_tasks/rl/trading/dqn/configs/DBB.py

# Train and remove previous run
python downstream_tasks/rl/trading/dqn/train.py \
  --config downstream_tasks/rl/trading/dqn/configs/DBB.py \
  --if_remove

# Resume from checkpoint
python downstream_tasks/rl/trading/dqn/train.py \
  --config downstream_tasks/rl/trading/dqn/configs/DBB.py \
  --resume

# Override config values inline
python downstream_tasks/rl/trading/dqn/train.py \
  --config downstream_tasks/rl/trading/dqn/configs/DBB.py \
  --cfg-options total_timesteps=500000 tag=my_experiment
```

**Available DQN configs:** (`downstream_tasks/rl/trading/dqn/configs/`)
| Config | Asset | Notes |
|--------|-------|-------|
| `DBB.py` | DBB ETF | **Full BRFT config**: 1M steps, augmentation + NFSP + quantile belief |
| `DBB_baseline_200k.py` | DBB ETF | No augmentation, 200k steps |
| `DBB_aug_200k.py` | DBB ETF | With augmentation, 200k steps |
| `SPY.py` | SPY ETF | S&P 500 tracker |
| `AAPL.py` | Apple | Individual stock |
| `CORN.py` | CORN ETF | Corn commodity |
| `GLD.py` | GLD ETF | Gold |
| `QQQ.py` | QQQ ETF | NASDAQ 100 |
| `IWM.py` | IWM ETF | Russell 2000 |
| `USO.py` | USO ETF | Oil |
| `UNG.py` | UNG ETF | Natural gas |
| `UGA.py` | UGA ETF | Gasoline |
| `DBC.py` | DBC ETF | Commodity index |
| `CYB.py` | CYB ETF | Chinese yuan |
| `FXB.py` | FXB ETF | British pound |
| `FXC.py` | FXC ETF | Canadian dollar |
| `FXE.py` | FXE ETF | Euro |
| `FXY.py` | FXY ETF | Japanese yen |

**Key config parameters (DBB.py — full BRFT):**
```python
total_timesteps = 1_000_000      # Training steps
check_steps = 10_000             # Checkpoint interval (100 checkpoints)
num_envs = 4                     # Parallel environments
use_data_augmentation = True     # GAN-based augmentation
augmentation_method = 'generator_adv_agent'  # Adversarial augmentation
use_nfsp = True                  # Nash Feedback on Self-Play
use_quantile_belief = True       # Bayesian quantile heads
```

**Output:** `downstream_tasks/rl/trading/workdir/exp/trading/{STOCK}/dqn/{tag}/`
- `saved_model/{N}.pth` — checkpoints every `check_steps`
- `events.out.tfevents.*` — TensorBoard logs

**Training speed:** ~300 steps/sec on CPU (no augmentation), ~60 steps/sec (with augmentation).

**Useful for:** Core of the paper. The DQN with all features (augmentation + NFSP + quantile belief) is the full BRFT agent.

---

## 11. RL Trading — PPO

**Script:** `downstream_tasks/rl/trading/ppo/train.py`
**Purpose:** Train a Proximal Policy Optimization trading agent.

```bash
python downstream_tasks/rl/trading/ppo/train.py \
  --config downstream_tasks/rl/trading/ppo/configs/SPY.py

python downstream_tasks/rl/trading/ppo/train.py \
  --config downstream_tasks/rl/trading/ppo/configs/AAPL.py
```

**Available PPO configs:** `AAPL.py`, `SPY.py` only.

**Architecture:** Separate actor-critic networks with shared embedding. On-policy with advantage estimation. No data augmentation support.

**Output:** Same structure as DQN under `.../ppo/{tag}/`

**Useful for:** Comparison baseline. Only 2 asset configs exist — less developed than DQN.

---

## 12. RL Trading — SAC

**Script:** `downstream_tasks/rl/trading/sac/train.py`
**Purpose:** Train a Soft Actor-Critic trading agent.

```bash
python downstream_tasks/rl/trading/sac/train.py \
  --config downstream_tasks/rl/trading/sac/configs/AAPL.py
```

**Available SAC configs:** `AAPL.py` only.

**Architecture:** Dual Q-networks with entropy regularization. Off-policy, maximum entropy framework.

**Output:** Same structure under `.../sac/{tag}/`

**Useful for:** Another comparison baseline. Only 1 config — least developed RL agent.

---

## 13. RL Inference & Metrics

### Inference

**Script:** `downstream_tasks/rl/trading/dqn/inference.py`
**Purpose:** Run a trained DQN agent on train/valid/test splits and save trading records.

```bash
python downstream_tasks/rl/trading/dqn/inference.py \
  --config downstream_tasks/rl/trading/dqn/configs/DBB.py
```

Loads the **last checkpoint** (`total_timesteps // check_steps`) and runs it through all three data splits.

**Output:** In the experiment directory:
- `train_records.json` — trading actions/values on training data
- `valid_records.json` — trading actions/values on validation data
- `test_records.json` — trading actions/values on test data

Each JSON contains: timestamp, value (portfolio), cash, position, ret (daily return), price, discount, total_profit, total_return, action.

### Metrics

**Script:** `downstream_tasks/rl/trading/dqn/calculate_metrics.py`
**Purpose:** Compute standard trading performance metrics from inference records.

```bash
# Compute metrics for a specific config
python downstream_tasks/rl/trading/dqn/calculate_metrics.py \
  --config downstream_tasks/rl/trading/dqn/configs/DBB.py

# Also available for other agents:
python downstream_tasks/rl/trading/ppo/calculate_metrics.py \
  --config downstream_tasks/rl/trading/ppo/configs/SPY.py

python downstream_tasks/rl/trading/sac/calculate_metrics.py \
  --config downstream_tasks/rl/trading/sac/configs/AAPL.py
```

**Output (console):**
```
Stage train:
ARR%, SR, CR, SOR, MDD%, VOL: -2.9704, -0.1260, -0.0157, -0.9338, 56.6375, 0.0146
Stage valid:
ARR%, SR, CR, SOR, MDD%, VOL: -1.6646, -0.0611, -0.0230, -0.8630, 23.8661, 0.0096
Stage test:
ARR%, SR, CR, SOR, MDD%, VOL: -5.4926, -0.2591, -0.1116, -3.8686, 30.9129, 0.0142
```

Metrics explained:
| Metric | Description | Good Values |
|--------|-------------|-------------|
| ARR% | Annualized Return Rate | Higher = better |
| SR | Sharpe Ratio (risk-adjusted return) | > 1.0 is good, > 2.0 is excellent |
| CR | Calmar Ratio (return / max drawdown) | Higher = better |
| SOR | Sortino Ratio (return / downside deviation) | Higher = better |
| MDD% | Maximum Drawdown | Lower = better |
| VOL | Daily Volatility | Context-dependent |

There's also a generic `tools/calculate_metrics.py` that works with strategy configs from `configs/exp/`.

---

## 14. RL Trading Plots

**Script:** `downstream_tasks/rl/trading/plot.py`
**Purpose:** Generate portfolio value charts from inference records.

```bash
python downstream_tasks/rl/trading/plot.py \
  --stock DBB \
  --algo dqn

# Custom directory
python downstream_tasks/rl/trading/plot.py \
  --dir_path downstream_tasks/rl/trading/workdir/exp/trading/ \
  --stock AAPL \
  --algo ppo
```

**Args:**
- `--dir_path` — base directory (default: `downstream_tasks/rl/trading/workdir/exp/trading/`)
- `--stock` — ticker symbol (default: AAPL)
- `--algo` — algorithm name: dqn, ppo, sac (default: ppo)

**Reads:** `{dir_path}/{stock}/{algo}/{tag}/{stage}_records.json`
**Output:** `downstream_tasks/rl/trading/workdir/fig/{algo}_{stock}_{tag}_{stage}.png`

**Prerequisite:** Must run inference first to generate `*_records.json` files.

---

## 15. Rule-Based Strategy Backtesting

**Script:** `downstream_tasks/strategy/trading/evaluation.py`
**Purpose:** Backtest 6 built-in technical analysis strategies with optional Optuna hyperparameter optimization.

```bash
# Backtest AAPL with all 6 strategies
python downstream_tasks/strategy/trading/evaluation.py \
  --config downstream_tasks/strategy/trading/configs/AAPL.py

# Other assets
python downstream_tasks/strategy/trading/evaluation.py \
  --config downstream_tasks/strategy/trading/configs/TSLA.py
```

**Available strategy configs:** `AAPL.py`, `AMZN.py`, `ETHUSD.py`, `GOOGL.py`, `MSFT.py`, `TSLA.py`

**Strategies evaluated:**
| # | Strategy | Key Parameters |
|---|----------|----------------|
| 0 | Buy & Hold | None (baseline) |
| 1 | SMA Crossover | short_window=5 |
| 2 | MACD | ilong=9, isig=3, rsiOverbought=60, rsiOversold=40 |
| 3 | Bollinger Bands | std_dev=2.0, overbought=80, oversold=20 |
| 4 | Z-Score Mean Reversion | z_score_threshold=1.0 |
| 5 | ATR Volatility Stop | atr_length=7, atr_multiplier=2.0 |

**Output:** Trading records + performance metrics for each strategy. Also supports Optuna-based parameter optimization.

**Useful for:** Quick baseline comparison without training any models. Good for evaluating whether ML agents beat simple rules.

---

## 16. Time Series Forecasting

**Script:** `downstream_tasks/forecasting/run.py`
**Purpose:** Train and evaluate 28 time series forecasting models on financial data.

```bash
# Train TimesNet for long-term forecasting
python downstream_tasks/forecasting/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TimesNet \
  --data ETTm1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 32 \
  --train_epochs 10

# Train on stock data
python downstream_tasks/forecasting/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model Transformer \
  --data custom \
  --root_path datasets/processd_day_future_etfs/features/ \
  --stock_ticker DBB \
  --features_name open high low close adj_close \
  --labels_name ret1 \
  --seq_len 30 \
  --pred_len 5
```

**Available models (28):**
| Category | Models |
|----------|--------|
| Attention | Autoformer, Transformer, Reformer, Informer, iTransformer, Crossformer |
| Temporal Conv | TCN, LightTS, TiDE |
| Frequency | FreTS, FEDformer, ETSformer |
| Novel Arch | TimesNet, DLinear, FiLM, Pyraformer, PatchTST, Koopa, S4, MICN, KAN_ |
| Baselines | GRU, LSTM, RNN, MLP |

**Task types:**
| Task | Description |
|------|-------------|
| `long_term_forecast` | Multi-step ahead prediction |
| `short_term_forecast` | Next-step prediction (M4 dataset) |
| `imputation` | Fill missing values |
| `classification` | Market regime classification |
| `anomaly_detection` | Detect anomalous patterns |

**Key arguments:**
```
--seq_len         Input sequence length (default: 96)
--pred_len        Prediction horizon (default: 96)
--label_len       Decoder start token length (default: 48)
--d_model         Model dimension (default: 512)
--n_heads         Attention heads (default: 8)
--e_layers        Encoder layers (default: 2)
--d_layers        Decoder layers (default: 1)
--batch_size      Batch size (default: 32)
--learning_rate   LR (default: 0.0001)
--train_epochs    Epochs (default: 10)
--patience        Early stopping patience (default: 3)
--features        M=multivariate, S=univariate, MS=multi→uni
```

**Output:** `checkpoints/` directory with trained models + evaluation metrics.

**Useful for:** Price prediction, volume forecasting, regime detection. Completely independent from the RL pipeline — can be used standalone. 28 models is a strong forecasting benchmarking suite.

---

## 17. Data Analysis — Qualitative

**Script:** `data_analysis/qualitative_analysis.py`
**Purpose:** Visualize market data with t-SNE and PCA dimensionality reduction.

```bash
python data_analysis/qualitative_analysis.py
```

**No arguments** — hardcoded to scan `datasets/` for all `.parquet` and `.csv` files.

**Output:** `workdir/qualitative_analysis/`
- `binary/TSNE/{filename}.png` — t-SNE plots (perplexity=30)
- `binary/MarketDynamics/{filename}.png` — temporal market dynamics

**Useful for:** Visual exploration of feature distributions and market regime clustering. Good for presentations / paper figures.

---

## 18. Data Analysis — Quantitative

**Script:** `data_analysis/quantitative_analysis.py`
**Purpose:** Statistical tests on time series data.

```bash
python data_analysis/quantitative_analysis.py
```

**Provides 3 functions** (call programmatically or via the script):

| Function | Test | What It Tells You |
|----------|------|-------------------|
| `kpss_test(series)` | KPSS Stationarity | Is the series stationary? (p < 0.05 = non-stationary) |
| `hurst_exponent(series)` | Hurst Exponent | H ≈ 0.5 = random walk, H < 0.5 = mean-reverting, H > 0.5 = trending |
| `variance_ratio(series, lag)` | Variance Ratio | VR ≈ 1 = random walk, VR > 1 = trending, VR < 1 = mean-reverting |

**Useful for:** Understanding whether a given asset's price series is suitable for momentum vs. mean-reversion strategies. Helps choose which strategy/model to apply.

---

## 19. Metrics Module

**Location:** `module/metrics/metrics.py`
**Purpose:** Shared trading performance metrics used by all calculate_metrics scripts.

```python
from module.metrics import ARR, SR, CR, SOR, MDD, VOL, DD

# Example usage with daily returns array
import numpy as np
rets = np.array([0.01, -0.005, 0.003, ...])  # daily returns

print(f"Annualized Return: {ARR(rets)*100:.2f}%")
print(f"Sharpe Ratio: {SR(rets):.4f}")
print(f"Max Drawdown: {MDD(rets)*100:.2f}%")
print(f"Calmar Ratio: {CR(rets, mdd=MDD(rets)):.4f}")
print(f"Sortino Ratio: {SOR(rets, dd=DD(rets)):.4f}")
print(f"Volatility: {VOL(rets):.4f}")
```

---

## 20. Pipeline Summary

### Full Pipeline (what we ran)
```bash
# Step 1: Download prices (no API key needed)
python tools/download_prices.py \
  --config configs/downloader/prices/yahoofinance_day_prices_future_etfs.py

# Step 2: Process features
python tools/data_process.py \
  --config configs/processor/processor_day_future_etfs.py

# Step 3: Train GAN (optional, needed for augmentation)
python generator/GRT_GAN/main.py --exp etf_18_120

# Step 4: Train DQN agent
python downstream_tasks/rl/trading/dqn/train.py \
  --config downstream_tasks/rl/trading/dqn/configs/DBB.py

# Step 5: Run inference
python downstream_tasks/rl/trading/dqn/inference.py \
  --config downstream_tasks/rl/trading/dqn/configs/DBB.py

# Step 6: Calculate metrics
python downstream_tasks/rl/trading/dqn/calculate_metrics.py \
  --config downstream_tasks/rl/trading/dqn/configs/DBB.py

# Step 7: Generate plots
python downstream_tasks/rl/trading/plot.py --stock DBB --algo dqn
```

### What's Actually Connected End-to-End
```
✅ download_prices → data_process → DQN/PPO/SAC train → inference → metrics/plots
✅ download_prices → data_process → strategy evaluation → metrics
✅ download_prices → data_process → GAN train → DQN augmented training
✅ forecasting/run.py (standalone with its own data loading)
✅ data_analysis scripts (standalone)

⚠️ NOT CONNECTED:
   download_news      → sits on disk, never merged into features
   download_sentiment → sits on disk, never merged into features
   download_economic  → sits on disk, never merged into features
   download_macro     → sits on disk, never merged into features
   download_expert    → sits on disk, never merged into features
```

### Asset Coverage by Module
| Module | Assets Available |
|--------|-----------------|
| DQN | 18 configs (ETFs + stocks) |
| PPO | 2 configs (AAPL, SPY) |
| SAC | 1 config (AAPL) |
| Strategies | 6 configs (AAPL, AMZN, ETHUSD, GOOGL, MSFT, TSLA) |
| Forecasting | 28 models, any data |
| Data Download | DJ30, S&P500, NASDAQ100, Russell1K, crypto, forex, futures, ETFs |

### What's Worth Evaluating

| Module | Effort | Value |
|--------|--------|-------|
| **DQN full (1M steps)** | High (hours of training) | Core paper claim — does augmentation help? |
| **Strategy backtesting** | Low (runs in seconds) | Quick baselines, Optuna tuning |
| **Forecasting models** | Medium (minutes per model) | 28-model benchmarking suite |
| **PPO/SAC** | Medium | RL comparison baselines |
| **Data analysis** | Low | Statistical insights on assets |
| **Sentiment/news pipeline** | High (needs code changes) | Incomplete — requires integration work |
| **GAN visualization** | Low | Synthetic data quality check |
