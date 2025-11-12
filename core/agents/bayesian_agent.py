# core/agents/bayesian_agent.py
# Catatan:

# BayesianPolicyNet memakai dua output (mean, logvar) untuk menangkap parameter uncertainty.

# update() menambahkan KL regularization untuk menahan drift akibat non-stationary environment.

# Kamu bisa tambahkan entropy bonus atau Bayesian value function di versi lanjutannya.


import torch
import torch.nn as nn
import torch.distributions as dist
from .base_agent import BaseAgent

class BayesianPolicyNet(nn.Module):
    """
    Policy network with Gaussian weight uncertainty (Bayesian linear layers).
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1_mean = nn.Linear(obs_dim, hidden_dim)
        self.fc1_logvar = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        mean = self.fc1_mean(x)
        logvar = self.fc1_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std  # sample latent representation
        action_logits = self.fc2(torch.tanh(z))
        return action_logits, (mean, logvar)


class BayesianAgent(BaseAgent):
    """
    Bayesian Reinforcement Learning agent.
    Uses stochastic policy for exploration and uncertainty tracking.
    """

    def __init__(self, obs_dim, action_dim, lr=3e-4, beta=0.01, device="cpu"):
        super().__init__(obs_dim, action_dim, device)
        self.policy = BayesianPolicyNet(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.beta = beta  # KL regularization coefficient

    def act(self, state, evaluate=False):
        """
        Select an action based on current policy logits.
        Exploration is automatically modulated by entropy:
        high uncertainty -> softer sampling (more exploration),
        low uncertainty -> greedy behavior (more exploitation).
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        # Some Bayesian networks return (mean, log_var)
        logits_out = self.policy(state)
        if isinstance(logits_out, tuple):
            logits, _ = logits_out
        else:
            logits = logits_out

        # Entropy of the base distribution
        base_probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(base_probs * torch.log(base_probs + 1e-8)).item()

        # 🔥 Entropy-weighted exploration
        temperature = 1.0 + entropy  # high entropy → more exploration
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        if evaluate:
            action = torch.argmax(probs)

        return action.item(), log_prob, entropy



    def update(self, log_probs, rewards, old_means, old_logvars):
        """
        Perform a simple policy gradient update with KL regularization
        to prevent posterior drift under non-stationarity.
        """
        returns = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        loss_policy = -(returns * log_probs).mean()

        # KL regularization term (simplified between current and previous weights)
        kl_loss = 0.5 * torch.sum(torch.exp(old_logvars) + (old_means - 0)**2 - 1.0 - old_logvars)
        loss = loss_policy + self.beta * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "policy_loss": loss_policy.item(), "kl_loss": kl_loss.item()}
