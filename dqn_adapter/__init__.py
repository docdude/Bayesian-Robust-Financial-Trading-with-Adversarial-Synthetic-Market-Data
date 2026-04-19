"""
Standalone DQN trading adapter.

Loads trained DQN state dicts and runs inference without the full repo.
Minimal dependencies: torch, einops, timm, scikit-learn, numpy, pandas.

Usage:
    from dqn_adapter import DQNAdapter

    adapter = DQNAdapter.from_checkpoint(
        checkpoint_path="path/to/2.pth",
        scaler_path="path/to/PG_scaler.pkl",
    )

    # From OHLCV DataFrame:
    action, label, q_values = adapter.predict(ohlcv_df)

    # From raw numpy observation (30, 153):
    action, label, q_values = adapter.predict_from_observation(obs_array)
"""

from dqn_adapter.adapter import DQNAdapter, PredictionResult, ACTION_LABELS

__all__ = ["DQNAdapter", "PredictionResult", "ACTION_LABELS"]
