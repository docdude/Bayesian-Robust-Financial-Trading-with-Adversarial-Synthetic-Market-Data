# Live Trading Checkpoint Selection

This file records the checkpoint defaults used by `live_trading/config.py` for
the corrected WaveNet Lambert full-run artifacts. These are not shell export
overrides; the checkpoint map is encoded directly in `BEST_CHECKPOINTS`.

## Artifact Set

- Model tag: `{TICKER}_aug_wavenet_dj30_v6_full`
- Sweep path: `downstream_tasks/rl/trading/workdir/exp/trading/<TICKER>/dqn/<TICKER>_aug_wavenet_dj30_v6_full/sweep_results/sweep_<TICKER>.csv`
- Checkpoint path: `downstream_tasks/rl/trading/workdir/exp/trading/<TICKER>/dqn/<TICKER>_aug_wavenet_dj30_v6_full/saved_model/<CKPT>.pth`
- Current live universe: `V` disabled, `SHW` included.
- Old `exp001_aug` artifacts are excluded because they were trained against the erroneous environment.

## Selection Rule

Checkpoint selection uses validation metrics only:

1. Keep rows with finite validation metrics, `val_ARR > 0`, `val_SR > 0`, and nonzero `val_VOL`.
2. Select the row with the highest `val_SR`.
3. Use `val_ARR` and lower `val_MDD` only as tie-breakers.
4. Do not select on test metrics.
5. Do not use Sortino as a primary selector; it is recorded only as a diagnostic.

Test metrics are holdout diagnostics for deployability, not checkpoint selection.

## Holdout Tiers

- Candidate: `val_SR >= 1.0`, `val_ARR > 0`, `test_SR >= 0.5`, `test_ARR > 0`, and `test_MDD <= 0.30`.
- Watchlist: `val_SR >= 0.5`, `val_ARR > 0`, `test_SR > 0`, `test_ARR > 0`, and `test_MDD <= 0.50`.
- Reject: selected checkpoint does not pass the holdout gate. It may still be listed in config for audit completeness, but should not be promoted to active paper allocation without a later decision.

## Selected Checkpoints

| Ticker | Checkpoint | Steps | Val SR | Val ARR | Val MDD | Test SR | Test ARR | Test MDD | Tier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AAPL | 2 | 20000 | 1.935 | 0.511 | 0.306 | 0.069 | -0.016 | 0.321 | Reject |
| AMGN | 38 | 380000 | 2.567 | 0.518 | 0.230 | 0.253 | 0.011 | 0.207 | Watchlist |
| AXP | 9 | 90000 | 1.634 | 0.315 | 0.278 | 0.536 | 0.048 | 0.218 | Candidate |
| BA | 14 | 140000 | 1.602 | 0.541 | 0.318 | -1.162 | -0.216 | 0.644 | Reject |
| CAT | 15 | 150000 | 1.389 | 0.295 | 0.293 | -1.338 | -0.178 | 0.660 | Reject |
| CSCO | 4 | 40000 | 1.241 | 0.235 | 0.344 | -0.530 | -0.090 | 0.416 | Reject |
| CVX | 29 | 290000 | 1.394 | 0.322 | 0.443 | -1.011 | -0.126 | 0.423 | Reject |
| DIS | 1 | 10000 | 1.298 | 0.273 | 0.431 | -1.224 | -0.183 | 0.617 | Reject |
| GS | 3 | 30000 | 0.494 | 0.045 | 0.320 | -0.768 | -0.095 | 0.397 | Reject |
| HD | 2 | 20000 | 1.639 | 0.344 | 0.364 | -0.203 | -0.045 | 0.358 | Reject |
| HON | 1 | 10000 | 1.158 | 0.205 | 0.430 | -0.203 | -0.040 | 0.345 | Reject |
| IBM | 40 | 400000 | 1.595 | 0.309 | 0.199 | -0.372 | -0.053 | 0.318 | Reject |
| INTC | 12 | 120000 | 1.896 | 0.579 | 0.295 | -0.844 | -0.170 | 0.704 | Reject |
| JNJ | 40 | 400000 | 0.937 | 0.095 | 0.248 | -1.821 | -0.136 | 0.476 | Reject |
| JPM | 31 | 310000 | 1.920 | 0.502 | 0.334 | 0.750 | 0.082 | 0.294 | Candidate |
| KO | 35 | 350000 | 1.810 | 0.230 | 0.217 | 0.351 | 0.019 | 0.172 | Watchlist |
| MCD | 34 | 340000 | 1.725 | 0.211 | 0.163 | -0.367 | -0.030 | 0.233 | Reject |
| MMM | 32 | 320000 | 1.147 | 0.182 | 0.365 | -1.214 | -0.131 | 0.520 | Reject |
| MRK | 25 | 250000 | 1.741 | 0.223 | 0.120 | 0.921 | 0.103 | 0.191 | Candidate |
| MSFT | 28 | 280000 | 1.649 | 0.253 | 0.266 | -0.254 | -0.073 | 0.340 | Reject |
| NKE | 2 | 20000 | 1.630 | 0.365 | 0.398 | -0.648 | -0.135 | 0.536 | Reject |
| PG | 29 | 290000 | 2.369 | 0.385 | 0.155 | 0.823 | 0.077 | 0.298 | Candidate |
| SHW | 19 | 190000 | 1.283 | 0.213 | 0.251 | -1.218 | -0.143 | 0.478 | Reject |
| TRV | 20 | 200000 | 1.571 | 0.313 | 0.230 | -1.370 | -0.133 | 0.392 | Reject |
| UNH | 1 | 10000 | 1.476 | 0.317 | 0.310 | -1.042 | -0.088 | 0.361 | Reject |
| VZ | 40 | 400000 | 2.160 | 0.336 | 0.241 | 0.202 | 0.004 | 0.250 | Watchlist |
| WBA | 9 | 90000 | 0.882 | 0.130 | 0.429 | -0.363 | -0.075 | 0.395 | Reject |
| WMT | 28 | 280000 | 1.194 | 0.166 | 0.178 | -0.736 | -0.094 | 0.358 | Reject |

## Promotion Set

Initial paper-trading candidates from this checkpoint rule are `AXP`, `JPM`, `MRK`, and `PG`.
Watchlist names are `AMGN`, `KO`, and `VZ`.

The default portfolio mode in `config.py` trades only the candidate basket:
`AXP`, `JPM`, `MRK`, and `PG`. The full corrected universe remains available
for scaler export, audit, and deliberate custom runs.