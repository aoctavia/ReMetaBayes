#!/usr/bin/env python3
# =========================================================
# ReMetaBayes — Probabilistic Meta-RL Training Script (Family Version)
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


# =========================================================
# MAKE ENVIRONMENT
# =========================================================
def make_env(env_cfg, seed_offset=0):
    name = env_cfg["name"].lower()
    seed = env_cfg.get("seed", 0) + seed_offset

    if name == "gridworldshift":
        return GridWorldShift(
            size=env_cfg.get("size", 5),
            shift_interval=env_cfg.get("shift_interval", 100),
            reward_noise=env_cfg.get("reward_noise", 0.0),
            reward_scale=env_cfg.get("reward_scale", 1.0),
            max_steps=env_cfg.get("max_steps", 50),
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
        raise ValueError(f"Unknown environment name: {name}")


# =========================================================
# GET OBSERVATION DIMENSION
# =========================================================
def get_obs_dim(env):
    state = env.reset()
    return int(np.array(state).shape[-1])


# =========================================================
# TRAJECTORY COLLECTION
# =========================================================
def collect_trajectory(env, agent, max_steps, z=None):
    states, actions, rewards = [], [], []

    state = env.reset()
    done = False
    step = 0

    while step < max_steps and not done:
        action, logp, entropy = agent.act(state, z=z)
        next_state, reward, done, _ = env.step(action)

        states.append(np.array(state, dtype=np.float32))
        actions.append(int(action))
        rewards.append(float(reward))

        state = next_state
        step += 1

    return {
        "states": torch.tensor(np.stack(states), dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.long),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "total_reward": float(sum(rewards)),
    }


# =========================================================
# TRAINING LOOP
# =========================================================
def train(config_path: str):
    # load config
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

    # ===============================================
    # PROTOTYPE ENV FOR DIMENSIONS
    # ===============================================
    first_family = env_cfg["task_families"][0]
    env_proto = make_env(first_family)
    obs_dim = get_obs_dim(env_proto)
    del env_proto

    # determine action space
    if first_family["name"].lower() == "contextualbandit":
        action_dim = first_family.get("n_actions", 5)
    else:
        action_dim = 4  # GridWorldShift uses 4 actions

    # ===============================================
    # META AGENT
    # ===============================================
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

    # ===============================================
    # LOGGER
    # ===============================================
    save_path = log_cfg.get("save_path", "results_meta/")
    os.makedirs(save_path, exist_ok=True)
    logger = Logger(log_dir=save_path, filename=f"{exp_cfg['name']}_metrics.json")

    total_meta_iters = exp_cfg.get("total_episodes", 1000)
    max_steps = exp_cfg.get("max_steps_per_episode", 50)
    n_tasks = meta_cfg.get("n_tasks", 5)
    adaptation_episodes = meta_cfg.get("adaptation_episodes", 2)

    print(f"🚀 Starting probabilistic meta-training for {total_meta_iters} iterations")

    # ===============================================
    # META-TRAINING LOOP
    # ===============================================
    for meta_iter in trange(total_meta_iters, desc="Meta-iter"):
        tasks = []
        mean_query_return = 0.0

        for task_idx in range(n_tasks):

            # PICK A RANDOM TASK FAMILY
            family_cfg = dict(np.random.choice(env_cfg["task_families"]))

            env = make_env(
                family_cfg,
                seed_offset=meta_iter * 1000 + task_idx * 10
            )

            # SUPPORT SET (ADAPTATION)
            support_trajs = []
            for _ in range(adaptation_episodes):
                traj = collect_trajectory(env, agent, max_steps, z=None)
                support_trajs.append(traj)

            support_states = torch.cat([t["states"] for t in support_trajs])
            support_actions = torch.cat([t["actions"] for t in support_trajs])
            support_rewards = torch.cat([t["rewards"] for t in support_trajs])

            support = {
                "states": support_states,
                "actions": support_actions,
                "rewards": support_rewards,
            }

            # INFER Z
            z_mean, z_logvar, z = agent.encode_support(support)

            # QUERY TRAJECTORY
            query_traj = collect_trajectory(env, agent, max_steps, z=z)
            mean_query_return += query_traj["total_reward"] / n_tasks

            query = {
                "states": query_traj["states"],
                "actions": query_traj["actions"],
                "rewards": query_traj["rewards"],
            }

            tasks.append({"support": support, "query": query})

        # META-UPDATE
        info = agent.meta_update(tasks)

        logger.log(
            episode=meta_iter,
            reward=mean_query_return
        )

        if (meta_iter + 1) % exp_cfg.get("log_interval", 50) == 0:
            print(f"[Iter {meta_iter+1}] mean_return={mean_query_return:.2f} meta_loss={info['meta_loss']:.4f}")

    logger.save()
    print(f"✅ Training complete. Results saved to {logger.path}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    train(args.config)
