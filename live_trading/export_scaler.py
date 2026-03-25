#!/usr/bin/env python3
"""
One-time utility: export the StandardScaler fitted on training data.

This recreates the *exact* training environment setup, fits the scaler,
and serialises it to live_trading/artifacts/scaler.pkl for live inference.

Usage:
    python export_scaler.py
"""
import os
import sys
import pickle
from pathlib import Path
from copy import deepcopy

ROOT = str(Path(__file__).resolve().parents[1])
DQN_DIR = os.path.join(ROOT, "downstream_tasks/rl/trading/dqn")
sys.path.insert(0, ROOT)
sys.path.insert(0, DQN_DIR)

from mmengine.config import Config                                   # noqa: E402
from downstream_tasks.dataset import AugmentatedDatasetStocks as DS  # noqa: E402
from environment import EnvironmentRET                               # noqa: E402


def main():
    config_path = os.path.join(DQN_DIR, "configs", "AAPL_aug.py")
    cfg = Config.fromfile(config_path)
    cfg.merge_from_dict({"root": ROOT})
    cfg.dataset.root = ROOT
    for key, value in cfg.items():
        if isinstance(value, dict) and "root" in value:
            cfg[key]["root"] = ROOT

    abs_data_path = os.path.join(ROOT, cfg.dataset.data_path)

    dataset = DS(
        root_path=ROOT,
        data_path=abs_data_path,
        train_stock_ticker=cfg.select_stock,
        test_stock_ticker=cfg.select_stock,
        features_name=cfg.dataset.features_name,
        temporals_name=cfg.dataset.temporals_name,
        target=cfg.dataset.labels_name,
        flag="RL",
    )

    # Build training env — this fits the StandardScaler
    train_env = EnvironmentRET(
        mode="train",
        dataset=dataset,
        select_stock=cfg.select_stock,
        timestamps=cfg.env.timestamps,
        if_norm=cfg.env.if_norm,
        if_norm_temporal=cfg.env.if_norm_temporal,
        scaler=cfg.env.scaler,
        start_date=cfg.train_start_date,
        end_date=cfg.train_end_date,
        initial_amount=cfg.env.initial_amount,
        transaction_cost_pct=cfg.env.transaction_cost_pct,
        level=cfg.level,
    )

    # The scaler list has one entry per stock in dj30.txt;
    # we need the one for select_stock
    stock_id = train_env.stocks2id[cfg.select_stock]
    scaler = train_env.scaler[stock_id]

    out_dir = os.path.join(ROOT, "live_trading", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scaler.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Scaler saved to {out_path}")
    print(f"  n_features_in_ = {scaler.n_features_in_}")
    print(f"  mean_[:5]      = {scaler.mean_[:5]}")
    print(f"  scale_[:5]     = {scaler.scale_[:5]}")


if __name__ == "__main__":
    main()
