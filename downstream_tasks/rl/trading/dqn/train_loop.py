import importlib

import numpy as np
import torch

import downstream_tasks.rl.trading.dqn.train as batch_train

from generator.WAVENET_LAMBERT_GAN.models.API_loop import GeneratorAPI as LoopGeneratorAPI

ROOT = batch_train.ROOT
CURRENT = batch_train.CURRENT

Dataset_Stocks = batch_train.Dataset_Stocks
EnvironmentRET = batch_train.EnvironmentRET
make_env = batch_train.make_env
Agent = batch_train.Agent
ActorContinuous = batch_train.ActorContinuous
ReplayBuffer = batch_train.ReplayBuffer
ReservoirReplayBuffer = batch_train.ReservoirReplayBuffer
save_json = batch_train.save_json
linear_schedule = batch_train.linear_schedule
update_data_root = batch_train.update_data_root
build_storage = batch_train.build_storage
compute_values = batch_train.compute_values
validate_agent = batch_train.validate_agent
get_quantile_belief = batch_train.get_quantile_belief

__all__ = [
    "ROOT",
    "CURRENT",
    "Dataset_Stocks",
    "EnvironmentRET",
    "make_env",
    "Agent",
    "ActorContinuous",
    "ReplayBuffer",
    "ReservoirReplayBuffer",
    "save_json",
    "linear_schedule",
    "update_data_root",
    "build_storage",
    "compute_values",
    "validate_agent",
    "get_quantile_belief",
    "data_augmentation_function",
    "main",
]


def data_augmentation_function(data: np.ndarray, cfg,
                               method: str = 'random', agent: Agent = None,
                               device=None, adv_agent: ActorContinuous = None, generator=None, timestamp=None, scaler=None):
    epsilon = cfg.epsilon
    data_std = data.std(axis=(0, 1))
    if method == 'random':
        noise = np.random.normal(loc=0.0, scale=epsilon * data_std, size=data.shape)
        return (noise + data, None)
    elif method == 'min_q':
        alpha = cfg.alpha
        iterations = cfg.iterations
        obs_ori = torch.Tensor(data).to(device)
        obs_std = torch.Tensor(data_std).to(device)
        quantile_belief_ori = get_quantile_belief(cfg, obs_ori, agent.quantile_belief_network)
        q_values = agent.q_network(torch.Tensor(data).to(device), quantile_belief_ori)
        actions_ori = torch.argmax(q_values, dim=1)

        obs = obs_ori.clone()
        for _ in range(iterations):
            with torch.no_grad():
                quantile_belief = get_quantile_belief(cfg, obs, agent.quantile_belief_network)
            obs.requires_grad_(True)
            q_values = agent.q_network(obs, quantile_belief)
            cost = torch.gather(q_values, -1, actions_ori.unsqueeze(-1)).sum()
            cost.backward()
            noise = -obs.grad.sgn() * alpha * obs_std
            obs = obs_ori + torch.clamp(obs.detach() + noise - obs_ori, -epsilon * obs_std, epsilon * obs_std)
        return (obs.detach().cpu().numpy(), None)
    elif method == 'adv_agent':
        data_tensor = torch.Tensor(data).to(device)
        noise = adv_agent(data_tensor).detach().cpu().numpy()
        return ((data + epsilon * noise * data_std), noise)
    elif method == 'generator_noise':
        num_envs = data.shape[0]
        macro_dim = getattr(generator, "macro_dim", 46)
        noise = np.random.normal(loc=0.0, scale=1, size=(num_envs, macro_dim))
        new_obs = np.zeros(data.shape)
        for i in range(num_envs):
            generated_data = generator.call(timestamp[i], noise[i])
            generated_feature = generated_data[-data.shape[1]:]
            generated_feature = np.concatenate((generated_feature, data[i][:, len(cfg.dataset.features_name):]), axis=1)
            if cfg.env.if_norm_temporal:
                new_obs[i] = scaler.transform(generated_feature)
            else:
                generated_feature_norm = scaler.transform(generated_feature[:, :len(cfg.dataset.features_name)])
                new_obs[i] = np.concatenate((generated_feature_norm, generated_feature[:, len(cfg.dataset.features_name):]), axis=1)
        return (new_obs, noise)
    elif method == 'generator_adv_agent':
        data_tensor = torch.Tensor(data).to(device)
        noise = adv_agent(data_tensor).detach().cpu().numpy()
        noise_output = noise.copy()
        if '0.3' in cfg.tag:
            noise = noise * 0.3
        num_envs = data.shape[0]
        new_obs = np.zeros(data.shape)
        for i in range(num_envs):
            generated_data = generator.call(timestamp[i], noise[i])
            generated_feature = generated_data[-data.shape[1]:]
            generated_feature = np.concatenate((generated_feature, data[i][:, len(cfg.dataset.features_name):]), axis=1)
            if cfg.env.if_norm_temporal:
                new_obs[i] = scaler.transform(generated_feature)
            else:
                generated_feature_norm = scaler.transform(generated_feature[:, :len(cfg.dataset.features_name)])
                new_obs[i] = np.concatenate((generated_feature_norm, generated_feature[:, len(cfg.dataset.features_name):]), axis=1)
        return (new_obs, noise_output)
    else:
        raise NotImplementedError


def main():
    api_module = importlib.import_module("generator.WAVENET_LAMBERT_GAN.models.API")
    original_generator_api = api_module.GeneratorAPI
    original_augmentation = batch_train.data_augmentation_function
    try:
        api_module.GeneratorAPI = LoopGeneratorAPI
        batch_train.data_augmentation_function = data_augmentation_function
        batch_train.main()
    finally:
        batch_train.data_augmentation_function = original_augmentation
        api_module.GeneratorAPI = original_generator_api


if __name__ == '__main__':
    main()