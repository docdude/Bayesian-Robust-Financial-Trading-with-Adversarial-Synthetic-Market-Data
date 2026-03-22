"""Sweep all checkpoints and evaluate on valid/test sets."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
from copy import deepcopy
import torch
import numpy as np
import random
import gymnasium as gym

ROOT = str(Path(__file__).resolve().parents[4])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)

from mmengine.config import Config
from downstream_tasks.dataset import AugmentatedDatasetStocks as Dataset_Stocks
from environment import EnvironmentRET
from wrapper import make_env
from policy import Agent
from module.metrics import ARR, SR, CR, SOR, MDD, VOL


def get_quantile_belief(cfg, obs, quantile_belief_network):
    quantile_logits = quantile_belief_network(obs[:, :-1, :])[:, -1]
    current_price = obs[:, -1, 19]
    difference = (quantile_logits - current_price.unsqueeze(-1)) ** 2
    return torch.argmin(difference, dim=-1)


def evaluate(cfg, agent, envs, device):
    """Run one episode, return (total_return, total_profit, final_value, SR, MDD)."""
    state, info = envs.reset()
    rets = [info["ret"]]
    next_obs = torch.Tensor(state).to(device)

    while True:
        with torch.no_grad():
            belief = get_quantile_belief(cfg, next_obs, agent.quantile_belief_network) if cfg.use_quantile_belief else None
            if cfg.use_nfsp:
                q_values = agent.q_network_nfsp(next_obs, belief)
            else:
                q_values = agent.target_network(next_obs, belief)
            action = torch.argmax(q_values, dim=1)

        next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
        record_info = info
        if 'final_info' in info:
            fi = info['final_info']
            if isinstance(fi, dict):
                record_info = fi
            else:
                record_info = {k: np.array([item[k] for item in fi]) for k in fi[0].keys()}
        rets.append(record_info["ret"])
        next_obs = torch.Tensor(next_obs).to(device)

        if "final_info" in info:
            fi = info["final_info"]
            if isinstance(fi, dict):
                tr = fi['total_return'][0]
                tp = fi['total_profit'][0]
                val = fi['value'][0]
            else:
                tr = fi[0]['total_return']
                tp = fi[0]['total_profit']
                val = fi[0]['value']
            break

    rets = np.array(rets)
    sr = SR(rets)
    mdd = MDD(rets)
    return tr, tp, val, sr, mdd


def main():
    config_path = os.path.join(CURRENT, "configs", "DBB.py")
    cfg = Config.fromfile(config_path)
    cfg.merge_from_dict({"root": ROOT})

    # Update data root
    cfg.dataset.root = ROOT
    for key, value in cfg.items():
        if isinstance(value, dict) and "root" in value:
            cfg[key]["root"] = ROOT

    exp_path = os.path.join(ROOT, cfg.workdir, cfg.tag)
    save_dir = os.path.join(exp_path, cfg.save_path)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cpu")

    # Load dataset once
    dataset = Dataset_Stocks(root_path=ROOT, data_path=cfg.dataset.data_path,
                             train_stock_ticker=cfg.select_stock, test_stock_ticker=cfg.select_stock,
                             features_name=cfg.dataset.features_name, temporals_name=cfg.dataset.temporals_name,
                             target=cfg.dataset.labels_name, flag="RL")

    train_env = EnvironmentRET(mode="train", dataset=dataset, select_stock=cfg.select_stock,
                               timestamps=cfg.env.timestamps, if_norm=cfg.env.if_norm,
                               if_norm_temporal=cfg.env.if_norm_temporal, scaler=cfg.env.scaler,
                               start_date=cfg.train_start_date, end_date=cfg.train_end_date,
                               initial_amount=cfg.env.initial_amount, transaction_cost_pct=cfg.env.transaction_cost_pct,
                               level=cfg.level)
    train_env.mode = "val"

    val_env = EnvironmentRET(mode="val", dataset=dataset, select_stock=cfg.select_stock,
                             timestamps=cfg.env.timestamps, if_norm=cfg.env.if_norm,
                             if_norm_temporal=cfg.env.if_norm_temporal, scaler=train_env.scaler,
                             start_date=cfg.valid_start_date, end_date=cfg.valid_end_date,
                             initial_amount=cfg.env.initial_amount, transaction_cost_pct=cfg.env.transaction_cost_pct,
                             level=cfg.level)

    test_env = EnvironmentRET(mode="test", dataset=dataset, select_stock=cfg.select_stock,
                              timestamps=cfg.env.timestamps, if_norm=cfg.env.if_norm,
                              if_norm_temporal=cfg.env.if_norm_temporal, scaler=train_env.scaler,
                              start_date=cfg.test_start_date, end_date=cfg.test_end_date,
                              initial_amount=cfg.env.initial_amount, transaction_cost_pct=cfg.env.transaction_cost_pct,
                              level=cfg.level)

    valid_envs = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(val_env),
                 transition_shape=cfg.transition_shape, seed=cfg.seed)) for _ in range(1)
    ], autoreset_mode="SameStep")
    test_envs = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(test_env),
                 transition_shape=cfg.transition_shape, seed=cfg.seed)) for _ in range(1)
    ], autoreset_mode="SameStep")

    quantile_heads_num = len(cfg.quantile_heads) if cfg.use_quantile_belief else 0
    agent = Agent(input_dim=cfg.num_features, timestamps=cfg.timestamps, embed_dim=cfg.embed_dim,
                  depth=cfg.depth, cls_embed=False, action_dim=train_env.action_dim,
                  temporals_name=cfg.temporals_name, device=device,
                  use_quantile_belief=cfg.use_quantile_belief, quantile_heads_num=quantile_heads_num,
                  use_nfsp=cfg.use_nfsp)

    # Find all checkpoints
    ckpts = sorted([int(f.replace('.pth', '')) for f in os.listdir(save_dir)
                    if f.endswith('.pth') and not f.endswith('_adv.pth')])

    print(f"{'Ckpt':>4s} {'Steps':>7s} | {'Val Return':>10s} {'Val Profit':>10s} {'Val Value':>12s} {'Val SR':>7s} {'Val MDD':>8s} | {'Test Return':>11s} {'Test Profit':>11s} {'Test Value':>12s} {'Test SR':>7s} {'Test MDD':>8s}")
    print("-" * 130)

    best_val_profit = -999
    best_ckpt = None

    for ckpt_num in ckpts:
        steps = ckpt_num * cfg.check_steps
        ckpt_path = os.path.join(save_dir, f"{ckpt_num}.pth")
        agent.load_state_dict(torch.load(ckpt_path, map_location=device))

        v_ret, v_prof, v_val, v_sr, v_mdd = evaluate(cfg, agent, valid_envs, device)
        t_ret, t_prof, t_val, t_sr, t_mdd = evaluate(cfg, agent, test_envs, device)

        marker = ""
        if v_prof > best_val_profit:
            best_val_profit = v_prof
            best_ckpt = ckpt_num
            marker = " <-- best val"

        print(f"{ckpt_num:4d} {steps:7d} | {v_ret:+10.4f} {v_prof:+10.2f}% {v_val:12,.0f} {v_sr:7.3f} {v_mdd:8.4f} | {t_ret:+11.4f} {t_prof:+11.2f}% {t_val:12,.0f} {t_sr:7.3f} {t_mdd:8.4f}{marker}")

    print(f"\nBest validation checkpoint: {best_ckpt} ({best_ckpt * cfg.check_steps} steps) with profit {best_val_profit:+.2f}%")

    valid_envs.close()
    test_envs.close()


if __name__ == '__main__':
    main()
