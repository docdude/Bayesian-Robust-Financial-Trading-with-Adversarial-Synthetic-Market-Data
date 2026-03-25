"""
Load a trained DQN agent checkpoint and run inference on a live observation.
"""
import os
import sys
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make sure the DQN modules are importable
ROOT = str(Path(__file__).resolve().parents[1])
DQN_DIR = os.path.join(ROOT, "downstream_tasks/rl/trading/dqn")
sys.path.insert(0, ROOT)
sys.path.insert(0, DQN_DIR)

from policy import Agent          # noqa: E402
from live_trading import config   # noqa: E402

ACTION_LABELS = {0: "SHORT", 1: "CLOSE", 2: "LONG"}


def load_agent(
    checkpoint_path: str = config.CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[Agent, torch.device]:
    """Instantiate the Agent and load checkpoint weights.

    Returns (agent, device).
    """
    if device is None:
        if torch.cuda.is_available():
            try:
                # Quick sanity check; cuDNN version mismatches only surface at use-time
                torch.zeros(1, device="cuda")
                device = torch.device("cuda")
            except RuntimeError:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

    quantile_heads_num = len(config.QUANTILE_HEADS) if config.USE_QUANTILE_BELIEF else 0

    agent = Agent(
        input_dim=config.INPUT_DIM,
        timestamps=config.TIMESTAMPS,
        embed_dim=config.EMBED_DIM,
        depth=config.DEPTH,
        cls_embed=False,
        action_dim=config.ACTION_DIM,
        temporals_name=config.TEMPORALS_NAME,
        device=device,
        use_quantile_belief=config.USE_QUANTILE_BELIEF,
        quantile_heads_num=quantile_heads_num,
        use_nfsp=config.USE_NFSP,
    )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent, device


def load_scaler(scaler_path: str = config.SCALER_PATH):
    """Load the persisted StandardScaler (fitted on training data)."""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def get_quantile_belief(obs: torch.Tensor, agent: Agent) -> torch.Tensor | None:
    """Compute quantile belief from observation.

    Parameters
    ----------
    obs : Tensor of shape (1, timestamps, input_dim)
    agent : trained Agent

    Returns
    -------
    Tensor of shape (1,) – argmin quantile index, or None.
    """
    if not config.USE_QUANTILE_BELIEF:
        return None
    quantile_logits = agent.quantile_belief_network(obs[:, :-1, :])[:, -1]
    # Feature index 19 = 'close' after normalization
    # (open=0, high=1, low=2, close=3, adj_close=4, kmid=5, ...,
    #  roc_5=14, roc_10=15, roc_20=16, roc_30=17, roc_60=18, ma_5=19)
    # Actually in the training code it is hardcoded as index 19
    current_price = obs[:, -1, 19]
    diff = (quantile_logits - current_price.unsqueeze(-1)) ** 2
    return torch.argmin(diff, dim=-1)


def predict_action(
    observation: np.ndarray,
    agent: Agent,
    device: torch.device,
) -> tuple[int, str, np.ndarray]:
    """Run one inference step.

    Parameters
    ----------
    observation : ndarray of shape (timestamps, input_dim)
        The 30-day sliding window of 150 normalised features + 3 raw temporals.
    agent : loaded Agent in eval mode.
    device : torch device.

    Returns
    -------
    (action_id, action_label, q_values)
        action_id : 0=short, 1=close, 2=long
        action_label : human-readable string
        q_values : raw Q-values array of shape (3,)
    """
    obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        belief = get_quantile_belief(obs, agent)
        if config.USE_NFSP:
            q_values = agent.q_network_nfsp(obs, belief)
        else:
            q_values = agent.target_network(obs, belief)
        action = torch.argmax(q_values, dim=1).item()

    q_np = q_values.cpu().numpy().flatten()
    return action, ACTION_LABELS[action], q_np
