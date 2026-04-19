"""
DQNAdapter — standalone interface for loading trained DQN agents and
running inference on OHLCV market data.

No dependency on the training repo. Only needs: torch, timm, einops,
scikit-learn, numpy, pandas.

Example
-------
    from dqn_adapter import DQNAdapter

    adapter = DQNAdapter.from_checkpoint("PG_2.pth", "PG_scaler.pkl")
    action, label, q_values = adapter.predict(ohlcv_df)
    # action: 0=SHORT, 1=CLOSE, 2=LONG
    # label: "SHORT" / "CLOSE" / "LONG"
    # q_values: np.ndarray of shape (3,)

    # Backtest a full history:
    signals = adapter.backtest(ohlcv_df)
    # returns DataFrame with columns: date, action, label, q_short, q_close, q_long
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dqn_adapter.features import FEATURES_NAME, TEMPORALS_NAME, ALL_COLUMNS, cal_factor
from dqn_adapter.model import Agent

ACTION_LABELS = {0: "SHORT", 1: "CLOSE", 2: "LONG"}

# Default architecture (matches all DJ30 Tier 1 models)
_DEFAULT_ARCH = dict(
    input_dim=153,
    timestamps=30,
    embed_dim=64,
    depth=1,
    action_dim=3,
    temporals_name=("day", "weekday", "month"),
    use_quantile_belief=True,
    quantile_heads_num=5,
    use_nfsp=True,
)

# Minimum bars needed: 60 (max rolling window) + 30 (observation window) + margin
MIN_HISTORY_BARS = 120


@dataclass
class PredictionResult:
    """Result of a single inference step."""
    action: int             # 0=SHORT, 1=CLOSE, 2=LONG
    label: str              # "SHORT", "CLOSE", "LONG"
    q_values: np.ndarray    # shape (3,) — raw Q-values


class DQNAdapter:
    """Stateless adapter for DQN agent inference.

    Holds a loaded model + scaler.  Feed it OHLCV data and get
    trading signals back.  No Alpaca / broker dependency.
    """

    def __init__(
        self,
        agent: Agent,
        scaler,
        device: torch.device,
        timestamps: int = 30,
        use_nfsp: bool = True,
        use_quantile_belief: bool = True,
    ):
        self.agent = agent
        self.scaler = scaler
        self.device = device
        self.timestamps = timestamps
        self.use_nfsp = use_nfsp
        self.use_quantile_belief = use_quantile_belief

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        scaler_path: str | Path,
        device: str | torch.device | None = None,
        arch: dict | None = None,
    ) -> DQNAdapter:
        """Load a trained model from a state-dict file and a scaler pickle.

        Parameters
        ----------
        checkpoint_path : path to the .pth state-dict file.
        scaler_path     : path to the sklearn StandardScaler pickle.
        device          : "cpu", "cuda", or None (auto-detect).
        arch            : dict overriding default architecture params.
                          Keys: input_dim, timestamps, embed_dim, depth,
                          action_dim, temporals_name, use_quantile_belief,
                          quantile_heads_num, use_nfsp.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        cfg = {**_DEFAULT_ARCH, **(arch or {})}

        agent = Agent(
            input_dim=cfg["input_dim"],
            timestamps=cfg["timestamps"],
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            action_dim=cfg["action_dim"],
            temporals_name=cfg["temporals_name"],
            device=device,
            use_quantile_belief=cfg["use_quantile_belief"],
            quantile_heads_num=cfg["quantile_heads_num"],
            use_nfsp=cfg["use_nfsp"],
        )

        state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
        agent.load_state_dict(state_dict)
        agent.eval()

        with open(str(scaler_path), "rb") as f:
            scaler = pickle.load(f)

        return cls(
            agent=agent,
            scaler=scaler,
            device=device,
            timestamps=cfg["timestamps"],
            use_nfsp=cfg["use_nfsp"],
            use_quantile_belief=cfg["use_quantile_belief"],
        )

    # ── Observation building ───────────────────────────────────────────────

    def build_observation(self, bars_df: pd.DataFrame) -> np.ndarray:
        """From raw OHLCV bars, compute features, normalise, return the
        latest *timestamps*-row observation window.

        Parameters
        ----------
        bars_df : DataFrame with columns [open, high, low, close, adj_close, volume],
                  DatetimeIndex, sorted ascending.  Needs >= MIN_HISTORY_BARS rows.

        Returns
        -------
        ndarray of shape (timestamps, 153).
        """
        feat_df = cal_factor(bars_df)

        # Select the 150 feature columns (no temporals) for scaling
        feature_cols = FEATURES_NAME
        temporal_cols = TEMPORALS_NAME

        features = feat_df[feature_cols].values
        temporals = feat_df[temporal_cols].values

        # Normalise features only (temporals stay raw)
        features_scaled = self.scaler.transform(features)

        # Recombine: features + temporals
        combined = np.concatenate([features_scaled, temporals], axis=1)

        # Last *timestamps* rows
        observation = combined[-self.timestamps:]
        return observation.astype(np.float32)

    # ── Inference ──────────────────────────────────────────────────────────

    def _get_quantile_belief(self, obs: torch.Tensor) -> torch.Tensor | None:
        """Compute quantile belief from observation tensor."""
        if not self.use_quantile_belief:
            return None
        quantile_logits = self.agent.quantile_belief_network(obs[:, :-1, :])[:, -1]
        current_price = obs[:, -1, 19]  # index 19 = ma_5 (matches training)
        diff = (quantile_logits - current_price.unsqueeze(-1)) ** 2
        return torch.argmin(diff, dim=-1)

    def predict_from_observation(self, observation: np.ndarray) -> PredictionResult:
        """Run inference on a pre-built observation array.

        Parameters
        ----------
        observation : ndarray of shape (timestamps, 153).

        Returns
        -------
        PredictionResult with action, label, q_values.
        """
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            belief = self._get_quantile_belief(obs)
            if self.use_nfsp:
                q_values = self.agent.q_network_nfsp(obs, belief)
            else:
                q_values = self.agent.target_network(obs, belief)
            action = torch.argmax(q_values, dim=1).item()

        q_np = q_values.cpu().numpy().flatten()
        return PredictionResult(action=action, label=ACTION_LABELS[action], q_values=q_np)

    def predict(self, bars_df: pd.DataFrame) -> PredictionResult:
        """End-to-end: OHLCV DataFrame → trading signal.

        Parameters
        ----------
        bars_df : DataFrame with >= MIN_HISTORY_BARS rows of
                  [open, high, low, close, adj_close, volume].

        Returns
        -------
        PredictionResult with action (0/1/2), label, q_values.
        """
        observation = self.build_observation(bars_df)
        return self.predict_from_observation(observation)

    # ── Backtesting ────────────────────────────────────────────────────────

    def backtest(
        self,
        bars_df: pd.DataFrame,
        start_index: int | None = None,
    ) -> pd.DataFrame:
        """Roll through a full OHLCV history and produce signals at each step.

        Parameters
        ----------
        bars_df     : Full OHLCV history (needs >= MIN_HISTORY_BARS rows).
        start_index : Row index to start from (default: MIN_HISTORY_BARS).

        Returns
        -------
        DataFrame with columns: date, action, label, q_short, q_close, q_long
        """
        if start_index is None:
            start_index = MIN_HISTORY_BARS

        records = []
        for i in range(start_index, len(bars_df)):
            window = bars_df.iloc[: i + 1]
            try:
                result = self.predict(window)
                records.append({
                    "date": bars_df.index[i],
                    "action": result.action,
                    "label": result.label,
                    "q_short": result.q_values[0],
                    "q_close": result.q_values[1],
                    "q_long": result.q_values[2],
                })
            except Exception as e:
                records.append({
                    "date": bars_df.index[i],
                    "action": 1,  # CLOSE on error
                    "label": "ERROR",
                    "q_short": 0.0,
                    "q_close": 0.0,
                    "q_long": 0.0,
                })

        return pd.DataFrame(records)

    # ── Multi-model portfolio ──────────────────────────────────────────────

    @classmethod
    def load_portfolio(
        cls,
        portfolio: dict[str, dict],
        device: str | torch.device | None = None,
        arch: dict | None = None,
    ) -> dict[str, DQNAdapter]:
        """Load multiple models for a portfolio of tickers.

        Parameters
        ----------
        portfolio : {ticker: {"checkpoint_path": ..., "scaler_path": ...}}
        device    : shared device for all models.
        arch      : shared architecture overrides.

        Returns
        -------
        {ticker: DQNAdapter}
        """
        adapters = {}
        for ticker, paths in portfolio.items():
            adapters[ticker] = cls.from_checkpoint(
                checkpoint_path=paths["checkpoint_path"],
                scaler_path=paths["scaler_path"],
                device=device,
                arch=arch,
            )
        return adapters
