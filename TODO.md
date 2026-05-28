# TODO — Next Iteration

## Current Artifact Guardrails
- [ ] Do not change `corr_*/cord_*` semantics for current Dow checkpoints: `module/processor/processor.py`, `live_trading/features.py`, `dqn_adapter/features.py`, and the generator APIs must stay mutually consistent with the already processed feature parquet and exported live `_scaler.pkl` files.
- [ ] Treat existing live `_scaler.pkl` files as DQN `StandardScaler` artifacts fitted on real processed training features; these are separate from WaveNet Lambert `lambert_fit_params.pkl`, which is generator-internal.
- [ ] If using current checkpoints live, keep `live_trading/features.py` and `dqn_adapter/features.py` on the replicated self-correlation behavior (`lambda x: 1.0`) so inference remains parity with training.

## Breaking Retrain Branch
- [x] Fix `corr_*/cord_*` in `module/processor/processor.py` (real price-volume correlation). Gated behind a `real_correlation` flag (default False = legacy Dow-safe); the ETF processor config sets it True.
- [x] Regenerate processed feature parquet after the processor fix. Ran `processor_day_future_etfs.py` (`real_correlation=True`); 18 ETF feature/price parquets regenerated in `workdir/processd_day_future_etfs` and synced to `datasets/processd_day_future_etfs`. Verified `corr_*/cord_*` are now genuine (nunique ≈ row count, range ≈ [-0.97, 0.94]).
- [x] Mirror the new real-correlation feature logic in `live_trading/features.py`, `dqn_adapter/features.py`, `generator/WAVENET_LAMBERT_GAN/models/API.py`, and `generator/GRT_GAN/models/API.py` (all via the `real_correlation` flag; default preserves the legacy 1.0 behaviour for Dow checkpoints).
- [x] WaveNet GAN does NOT need retraining for the corr fix: the generator only emits the 5 PV/OHLCV channels per ticker; `corr_*/cord_*` (and all 153 alpha features) are derived deterministically afterward in `transform_data_to_feature`. Epoch-4000 checkpoint reused as-is with `real_correlation=True` at inference.
- [x] Set training window to match available ETF data (ends 2023-12-29): train 2000–2019, valid 2020–2021, test 2022–2023. Applied to all 15 ETF `_aug.py` configs (CORN, CYB, DBB, DBC, FXB, FXC, FXE, FXY, GLD, IWM, QQQ, SPY, UGA, UNG, USO), each with `tag="aug_wavenet_gen_adv"`, `gan_checkpoint_epoch=4000`, `gan_real_correlation=True`, `gan_feature_method="log_returns"`, `total_timesteps=400k`. Old bare-name ETF configs deleted; `train.py` default updated to `CORN_aug.py`.
- [ ] Refit DQN `StandardScaler` on the extended regenerated training data, export new live `_scaler.pkl` files, and pair them only with checkpoints trained on the same feature semantics.
- [ ] If retraining WaveNet, regenerate/retrain its Lambert preprocessing artifacts separately (`datasets/output_data_lambert*`, `lambert_fit_params.pkl`) so generator scaling matches the new generator model.
- [ ] Re-sweep all 30 DJ30 stocks, re-select Tier 1 (test_SR >= 1.0).

## GAN Upgrade
- [ ] Evaluate alternative GAN architecture (replace GRT_GAN)
- [ ] Compare generated data quality: correlation difference, inter-instrument fidelity
- [ ] Retrain DQN agents with new GAN adversarial augmentation
- [ ] Benchmark new aug models vs current `exp001_aug` on same test window
- [ ] Goal: more stocks reaching Tier 1, higher Sharpe across the board

## Monitoring (Current Models)
- [ ] Track paper trading Sharpe over 1–2 months (current 8 Tier 1 stocks)
- [ ] Flag if any stock's rolling Sharpe drops below 0.5
- [ ] Compare live signals vs backtest signals for drift detection
