#!/usr/bin/env python3
# =========================================================
# ReMetaBayes — Probabilistic Meta-RL Training Script
# =========================================================

import argparse
import os

import yaml
from tqdm import trange

import numpy as np
import torch

from core.envs.gridworld_shift import GridWorldShift
from core.envs.contextual_bandit import ContextualBandit
from core.utils.logger import Logger
from core.agents.meta_agent import MetaAgent


def make_env(env_cfg, seed_offset: int = 0):
    name = env_cfg["name"].lower()
    seed = env_cfg.get("seed", 0) + seed_offset

    if name == "gridworldshift":
        return GridWorldShift(
            size=env_cfg.get("size", 5),
            shift_interval=env_cfg.get("shift_interval", 100),
            seed=seed,
        )
    elif name == "contextualbandit":
        return ContextualBandit(
            n_actions=env_cfg.get("n_actions", 5),
            n_contexts=env_cfg.get("n_contexts", 10),
            reward_shift_interval=env_cfg.get("reward_shift_interval", 50),
            reward_noise_std=env_cfg.get("reward_noise_std", 0.05),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown environment name: {env_cfg['name']}")


def get_obs_dim(env, env_cfg):
    state = env.reset()
    return int(np.array(state).shape[-1])


def collect_trajectory(env, agent, max_steps: int):
    """
    Roll out one episode (or fixed horizon if env has no terminal) tanpa z eksplisit.
    z akan di-infer di dalam MetaAgent.meta_update() dari support.
    """
    states, actions, rewards = [], [], []

    state = env.reset()
    done = False
    step = 0

    while step < max_steps and not done:
        action, logp, entropy = agent.act(state)  # z=None
        next_state, reward, done, _ = env.step(action)

        states.append(np.array(state, dtype=np.float32))
        actions.append(int(action))
        rewards.append(float(reward))

        state = next_state
        step += 1

    states_t = torch.tensor(np.stack(states, axis=0), dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)

    return {
        "states": states_t,
        "actions": actions_t,
        "rewards": rewards_t,
        "total_reward": float(sum(rewards)),
    }


def train(config_path: str):
    # 1) Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    env_cfg = cfg["environment"]
    agent_cfg = cfg["agent"]
    meta_cfg = cfg["meta_learning"]
    log_cfg = cfg.get("logging", {})

    seed = exp_cfg.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2) env prototype → obs_dim
    env_proto = make_env(env_cfg, seed_offset=0)
    obs_dim = get_obs_dim(env_proto, env_cfg)
    if env_cfg["name"].lower() == "contextualbandit":
        action_dim = env_cfg.get("n_actions", 5)
    else:
        action_dim = 4  # GridWorldShift: up/right/down/left
    del env_proto

    # 3) MetaAgent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = MetaAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=agent_cfg.get("latent_dim", 16),
        hidden_dim=agent_cfg.get("hidden_dim", 128),
        lr=agent_cfg.get("lr", 3e-4),
        beta_kl=agent_cfg.get("beta_kl", 1e-3),
        beta_kl_z=meta_cfg.get("beta_kl_z", 1e-3),
        gamma=agent_cfg.get("gamma", 0.99),
        device=device,
    )

    # 4) Logger
    save_path = log_cfg.get("save_path", "results/")
    os.makedirs(save_path, exist_ok=True)
    logger = Logger(log_dir=save_path, filename=f"{exp_cfg['name']}_metrics.json")

    total_meta_iters = exp_cfg.get("total_episodes", 1000)
    max_steps_per_episode = exp_cfg.get("max_steps_per_episode", 50)

    n_tasks = meta_cfg.get("n_tasks", 5)
    adaptation_episodes = meta_cfg.get("adaptation_episodes", 2)

    print(f"🚀 Starting probabilistic meta-training for {total_meta_iters} iterations on {env_cfg['name']}")

    for meta_iter in trange(total_meta_iters, desc="Meta-iter"):
        tasks = []
        mean_query_return = 0.0

        for task_idx in range(n_tasks):
            env = make_env(env_cfg, seed_offset=meta_iter * 1000 + task_idx * 10)

            # 1) support trajectories
            support_trajs = []
            for _ in range(adaptation_episodes):
                traj = collect_trajectory(env, agent, max_steps_per_episode)
                support_trajs.append(traj)

            support_states = torch.cat([t["states"] for t in support_trajs], dim=0)
            support_actions = torch.cat([t["actions"] for t in support_trajs], dim=0)
            support_rewards = torch.cat([t["rewards"] for t in support_trajs], dim=0)

            support = {
                "states": support_states,
                "actions": support_actions,
                "rewards": support_rewards,
            }

            # 2) query trajectory
            query_traj = collect_trajectory(env, agent, max_steps_per_episode)
            query = {
                "states": query_traj["states"],
                "actions": query_traj["actions"],
                "rewards": query_traj["rewards"],
            }
            mean_query_return += query_traj["total_reward"] / float(n_tasks)

            tasks.append({"support": support, "query": query})

        # 3) meta-update
        info = agent.meta_update(tasks)

        # 4) log
        logger.log(
            episode=meta_iter,
            reward=mean_query_return,
            uncertainty=None,
            kl=None,  # bisa diperluas untuk log KL z
        )

        if (meta_iter + 1) % exp_cfg.get("log_interval", 20) == 0:
            print(
                f"[Iter {meta_iter+1}] "
                f"mean_query_return={mean_query_return:.2f} "
                f"meta_loss={info['meta_loss']:.4f}"
            )

    logger.save()
    print(f"✅ Training complete. Metrics saved to {logger.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ReMetaBayes Probabilistic Meta-RL Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/run_gridworld.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    train(args.config)
