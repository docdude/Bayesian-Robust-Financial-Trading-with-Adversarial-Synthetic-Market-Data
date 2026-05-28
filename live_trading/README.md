# Live Trading — DQN Agent on Alpaca Paper Trading

Deploy the trained Bayesian-robust DQN agents to Alpaca paper trading for live
out-of-sample generalization testing. The default model path targets the
WaveNet full-run tag `{TICKER}_aug_wavenet_dj30_v6_full` and uses the
validation-selected checkpoint map in `config.py`. The selection rationale and
holdout audit are recorded in `CHECKPOINT_SELECTION.md`.

## Architecture

```
live_trading/
├── config.py          # Settings (paths, agent arch, Alpaca creds, scheduling)
├── features.py        # Replicates cal_factor() — 150 alpha factors + 3 temporals
├── agent_loader.py    # Load checkpoint, scaler, run inference
├── export_scaler.py   # One-time: export StandardScaler from training env
├── CHECKPOINT_SELECTION.md  # Validation-selected checkpoint audit
├── alpaca_trader.py   # Main trading bot (one-shot or daemon)
├── artifacts/
│   └── <TICKER>_scaler.pkl  # Persisted StandardScaler per ticker
└── logs/              # Daily trading logs
```

## Setup

### 1. Export the scaler (one-time)

```bash
cd /opt/Bayesian-Robust-Financial-Trading-with-Adversarial-Synthetic-Market-Data
source .venv/bin/activate
python live_trading/export_scaler.py
```

Creates `live_trading/artifacts/AAPL_scaler.pkl` fitted on the training split.
For the full clean28 WaveNet universe, run:

```bash
python live_trading/export_scaler.py --stock all
```

### 2. Set Alpaca credentials

Add your keys to the project-root `.env` file (already gitignored):

```env
ALPACA_API_KEY=your-paper-api-key
ALPACA_SECRET_KEY=your-paper-secret-key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Get paper trading keys at <https://app.alpaca.markets/paper/dashboard/overview>.

`config.py` calls `load_dotenv()` automatically — no manual `export` needed.

### 3. Run

**One-shot** (e.g. from cron):
```bash
python live_trading/alpaca_trader.py
```

**Dry run** (compute actions and intended orders but skip order execution):
```bash
python live_trading/alpaca_trader.py --dry-run
```

**Daemon mode** (runs continuously, trades daily at 15:55 ET):
```bash
python live_trading/alpaca_trader.py --daemon
```

### 4. Cron example (recommended)

```cron
# Run at 3:55 PM ET on weekdays
55 15 * * 1-5 cd /opt/Bayesian-Robust-Financial-Trading-with-Adversarial-Synthetic-Market-Data && .venv/bin/python live_trading/alpaca_trader.py >> live_trading/logs/cron.log 2>&1
```

## Pipeline per trading cycle

Portfolio mode trades the default candidate basket `AXP`, `JPM`, `MRK`, and
`PG`. The full corrected WaveNet universe is kept for scaler export and audit.
Set `LIVE_TRADING_ACTIVE_STOCKS=all` to intentionally trade all audited names,
or provide a comma-separated ticker list to run a custom basket.

1. **Fetch** 120 daily OHLCV bars from Alpaca Data API (split-adjusted)
2. **Compute** 150 alpha factors via `cal_factor()` (rolling windows 5/10/20/30/60)
3. **Normalize** the 150 features using the saved per-ticker StandardScaler (temporals are NOT normalized)
4. **Window** the last 30 rows → observation tensor `(1, 30, 153)`
5. **Infer** action via quantile-belief + NFSP Q-network → `argmax(Q-values)` → SHORT/CLOSE/LONG
6. **Execute** market order on Alpaca with the same transaction-cost sizing buffer used by `EnvironmentRET.trade()`
7. **Log** action, Q-values, portfolio state

## Action mapping

| Action ID | Label | Position |
|-----------|-------|----------|
| 0 | SHORT | -100% equity (short) |
| 1 | CLOSE | Flat (no position) |
| 2 | LONG  | +100% equity (long) |

## Known data quirks (replicated for consistency)

- **Correlation features** (`corr_*`, `cord_*`): Training processor had a bug that
  computed self-correlation (≈1.0) instead of price–volume correlation. Replicated.
- **Temporal features** (`day`, `weekday`, `month`): Training parquet saved with
  RangeIndex, so `pd.to_datetime()` converted to 1970-01-01 → all THREE are constant
  (1, 3, 1). Replicated.
- **Rank features** (`rank_*`): Uses `pd.Series(x).rank(pct=True).iloc[-1] / w`,
  matching the original `my_rank()` function.
- **Lambert scaling**: Only applies to WaveNet synthetic generation during
  adversarial training. Live broker observations use real OHLCV bars and the
  processed-parquet feature/scaler path.

## Validation

Features were validated against the training parquet — all 150 features match within
1e-6 tolerance. Full inference pipeline tested across 6 date windows spanning
2000–2023, producing diverse actions (SHORT, CLOSE, LONG) consistent with market
conditions.
