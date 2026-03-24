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
from module.metrics import ARR, SR, CR, SOR, MDD, VOL, DD


def get_quantile_belief(cfg, obs, quantile_belief_network):
    quantile_logits = quantile_belief_network(obs[:, :-1, :])[:, -1]
    current_price = obs[:, -1, 19]
    difference = (quantile_logits - current_price.unsqueeze(-1)) ** 2
    return torch.argmin(difference, dim=-1)


def evaluate(cfg, agent, envs, device):
    """Run one episode, return dict with all paper-comparable metrics."""
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
    mdd = MDD(rets)
    dd = DD(rets)
    return dict(
        total_return=tr, profit=tp, value=val,
        ARR=ARR(rets), SR=SR(rets), CR=CR(rets, mdd=mdd),
        SOR=SOR(rets, dd=dd), MDD=mdd, VOL=VOL(rets),
    )


def main():
    config_path = os.path.join(CURRENT, "configs", "AAPL_aug.py")
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

    # Load dataset once — data_path in config is relative, make it absolute
    abs_data_path = os.path.join(ROOT, cfg.dataset.data_path)
    dataset = Dataset_Stocks(root_path=ROOT, data_path=abs_data_path,
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

    hdr = (f"{'Ckpt':>4s} {'Steps':>7s} | "
           f"{'V-ARR%':>7s} {'V-SR':>6s} {'V-CR':>7s} {'V-SOR':>7s} {'V-MDD%':>7s} {'V-VOL':>6s} | "
           f"{'T-ARR%':>7s} {'T-SR':>6s} {'T-CR':>7s} {'T-SOR':>7s} {'T-MDD%':>7s} {'T-VOL':>6s}")
    print(hdr)
    print("-" * len(hdr))

    best_val_sr = -999
    best_ckpt = None
    results = []

    for ckpt_num in ckpts:
        steps = ckpt_num * cfg.check_steps
        ckpt_path = os.path.join(save_dir, f"{ckpt_num}.pth")
        agent.load_state_dict(torch.load(ckpt_path, map_location=device))

        v = evaluate(cfg, agent, valid_envs, device)
        t = evaluate(cfg, agent, test_envs, device)
        results.append((ckpt_num, steps, v, t))

        marker = ""
        if v['SR'] > best_val_sr:
            best_val_sr = v['SR']
            best_ckpt = ckpt_num
            marker = " <-- best val"

        print(f"{ckpt_num:4d} {steps:7d} | "
              f"{v['ARR']*100:+7.2f} {v['SR']:6.3f} {v['CR']:+7.3f} {v['SOR']:+7.3f} {v['MDD']*100:7.2f} {v['VOL']:6.4f} | "
              f"{t['ARR']*100:+7.2f} {t['SR']:6.3f} {t['CR']:+7.3f} {t['SOR']:+7.3f} {t['MDD']*100:7.2f} {t['VOL']:6.4f}{marker}")

    # Print top-5 by test SR
    by_test_sr = sorted(results, key=lambda x: x[3]['SR'], reverse=True)[:5]
    print(f"\n=== Top 5 by Test Sharpe Ratio ===")
    print(f"{'Ckpt':>4s} {'Steps':>7s} | {'ARR%':>7s} {'SR':>6s} {'CR':>7s} {'SOR':>7s} {'MDD%':>7s} {'VOL':>6s}")
    for ckpt_num, steps, v, t in by_test_sr:
        print(f"{ckpt_num:4d} {steps:7d} | {t['ARR']*100:+7.2f} {t['SR']:6.3f} {t['CR']:+7.3f} {t['SOR']:+7.3f} {t['MDD']*100:7.2f} {t['VOL']:6.4f}")

    print(f"\nBest validation checkpoint: {best_ckpt} ({best_ckpt * cfg.check_steps} steps) with SR {best_val_sr:.3f}")

    valid_envs.close()
    test_envs.close()


if __name__ == '__main__':
    main()
