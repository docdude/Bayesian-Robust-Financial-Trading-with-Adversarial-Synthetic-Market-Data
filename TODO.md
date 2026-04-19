# TODO — Next Iteration

## Retraining
- [ ] Fix `corr_*/cord_*` bug in `module/processor/processor.py` (change `pairwise=df2` → positional `df2`)
- [ ] Mirror fix in `live_trading/features.py` and `dqn_adapter/features.py` (replace `lambda x: 1.0` with real correlation)
- [ ] Extend training window (train through 2022, val 2023, test 2024–2025)
- [ ] Refit StandardScaler on extended training data, export new `_scaler.pkl` files
- [ ] Re-sweep all 30 DJ30 stocks, re-select Tier 1 (test_SR >= 1.0)

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
