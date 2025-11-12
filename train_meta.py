#!/usr/bin/env python3
# =========================================================
# ReMetaBayes — Training Script for Adaptive Meta-RL
# Author: Aulia Octaviani
# =========================================================

import argparse
import yaml
import json
import os
from tqdm import trange

import torch
import numpy as np

# Import project modules
from core.agents.meta_agent import MetaAgent
from core.agents.bayesian_agent import BayesianAgent
from core.envs.gridworld_shift import GridWorldShift
from core.utils.logger import Logger
from core.utils.helpers import set_seed


# =========================================================
# 1️⃣ Load configuration
# =========================================================
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# =========================================================
# 2️⃣ Environment factory
# =========================================================
def make_env(env_cfg):
    name = env_cfg.get("name", "GridWorldShift")
    if name.lower() == "gridworldshift":
        env = GridWorldShift(size=env_cfg["size"],
                             shift_interval=env_cfg["shift_interval"],
                             seed=env_cfg.get("seed", None))
    else:
        raise ValueError(f"Unsupported environment: {name}")
    return env


# =========================================================
# 3️⃣ Agent factory
# =========================================================
def make_agent(agent_cfg):
    agent_type = agent_cfg.get("type", "MetaAgent").lower()
    obs_dim = agent_cfg["obs_dim"]
    action_dim = agent_cfg["action_dim"]

    if agent_type == "metaagent":
        agent = MetaAgent(obs_dim=obs_dim,
                          action_dim=action_dim,
                          lr_inner=agent_cfg["lr_inner"],
                          lr_outer=agent_cfg["lr_outer"],
                          beta=agent_cfg["beta"],
                          device=agent_cfg.get("device", "cpu"))
    elif agent_type == "bayesianagent":
        agent = BayesianAgent(obs_dim=obs_dim,
                              action_dim=action_dim,
                              lr=agent_cfg["lr"],
                              beta=agent_cfg["beta"],
                              device=agent_cfg.get("device", "cpu"))
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    return agent


# =========================================================
# 4️⃣ Run single episode
# =========================================================
def run_episode(env, agent, max_steps=50, evaluate=False):
    state = env.reset()
    total_reward = 0
    log_probs, rewards = [], []

    for _ in range(max_steps):
        action, log_prob = agent.act(state, evaluate=evaluate)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
        if done:
            break

    # Optional simple policy update (for BayesianAgent)
    if not evaluate and hasattr(agent, "update"):
        old_means, old_logvars = torch.zeros(1), torch.zeros(1)
        agent.update(torch.stack(log_probs), rewards, old_means, old_logvars)

    return total_reward


# =========================================================
# 5️⃣ Training loop
# =========================================================
def train(cfg_path):
    # --- Load configuration ---
    cfg = load_config(cfg_path)
    set_seed(cfg["experiment"]["seed"])

    # --- Setup environment ---
    env = make_env(cfg["environment"])

    # --- Setup agent ---
    agent = make_agent(cfg["agent"])
    logger = Logger(log_dir=cfg["logging"]["save_path"])

    total_episodes = cfg["experiment"]["total_episodes"]
    max_steps = cfg["experiment"]["max_steps_per_episode"]
    eval_interval = cfg["experiment"]["eval_interval"]

    print(f"🚀 Starting training for {total_episodes} episodes on {env.__class__.__name__}...")

    # --- Main training loop ---
    for episode in trange(total_episodes, desc="Training"):
        reward = run_episode(env, agent, max_steps)
        logger.log(episode, reward)

        # Optional evaluation phase
        if (episode + 1) % eval_interval == 0:
            eval_reward = np.mean([run_episode(env, agent, max_steps, evaluate=True)
                                   for _ in range(5)])
            print(f"Episode {episode+1}/{total_episodes} | Eval Reward: {eval_reward:.3f}")

        # Periodically save metrics
        if (episode + 1) % cfg["experiment"]["log_interval"] == 0:
            logger.save()

    # --- Final save ---
    logger.save()
    print(f"✅ Training complete. Metrics saved to {logger.path}")


# =========================================================
# 6️⃣ Entry point
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ReMetaBayes Meta-Agent")
    parser.add_argument("--config", type=str, default="experiments/run_gridworld.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    os.makedirs("results/", exist_ok=True)
    train(args.config)
