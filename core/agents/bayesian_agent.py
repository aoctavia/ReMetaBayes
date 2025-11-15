# core/agents/bayesian_agent.py
# Catatan:
# - BayesianPolicyNet mengeluarkan mean & log-var dari logits aksi.
# - BayesianAgent: policy-gradient + KL regularizer untuk menjaga stabilitas
#   di lingkungan non-stationary.
import torch
import torch.nn as nn
import torch.distributions as dist

from .base_agent import BaseAgent


class BayesianPolicyNet(nn.Module):
    """
    Simple Bayesian-style policy network:
    - takes state (and optional latent context z)
    - outputs mean and log-variance of action logits
    """

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 0, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        input_dim = obs_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.out_mean = nn.Linear(hidden_dim, action_dim)
        self.out_logvar = nn.Linear(hidden_dim, action_dim)

    def forward(self, states: torch.Tensor, z: torch.Tensor | None = None):
        """
        states: [B, obs_dim]
        z:      [B, latent_dim] or None (treated as zeros)
        returns:
            logits_mean, logits_logvar  (both [B, action_dim])
        """
        if z is None:
            if self.latent_dim > 0:
                z = torch.zeros(states.shape[0], self.latent_dim, device=states.device)
            else:
                z = None

        if z is not None:
            x = torch.cat([states, z], dim=-1)
        else:
            x = states

        h = self.net(x)
        mean = self.out_mean(h)
        logvar = self.out_logvar(h).clamp(-10.0, 2.0)  # prevent extreme values
        return mean, logvar


class BayesianAgent(BaseAgent):
    """
    Single-task Bayesian policy gradient agent.
    - Uses BayesianPolicyNet
    - Samples action logits from Normal(mean, var)
    - Uses entropy-weighted temperature for exploration
    - Adds a simple KL regularizer to keep logits near a zero-mean prior.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 0,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        beta_kl: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        super().__init__(obs_dim, action_dim, device=device)
        self.gamma = gamma
        self.beta_kl = beta_kl

        self.policy = BayesianPolicyNet(obs_dim, action_dim, latent_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state, z: torch.Tensor | None = None):
        """
        Choose an action for a single state.
        Returns action (int), log_prob (tensor), entropy (float).
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0)  # [1, obs_dim]

        mean, logvar = self.policy(state, z)
        std = torch.exp(0.5 * logvar)
        logits_dist = dist.Normal(mean, std)
        logits = logits_dist.rsample()  # [1, action_dim]

        base_probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(base_probs * torch.log(base_probs + 1e-8), dim=-1)  # [1]

        # entropy-weighted temperature
        temperature = 1.0 + entropy
        scaled_logits = logits / temperature.unsqueeze(-1)
        probs = torch.softmax(scaled_logits, dim=-1)  # [1, action_dim]

        m = dist.Categorical(probs)
        action = m.sample()             # [1]
        log_prob = m.log_prob(action)   # [1]

        return int(action.item()), log_prob.squeeze(0), float(entropy.item())

    def _compute_returns(self, rewards: list[float]) -> torch.Tensor:
        """Compute discounted returns."""
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = float(r) + self.gamma * R
            returns.append(R)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(
        self,
        log_probs: torch.Tensor,
        rewards: list[float],
        mean_history: torch.Tensor | None = None,
        logvar_history: torch.Tensor | None = None,
    ):
        """
        Simple policy gradient update with KL regularization.

        log_probs: [T] tensor of log π(a_t | s_t)
        rewards:   list of T rewards
        mean_history, logvar_history:
            optional tensors aggregated during the episode for KL regularization.
            If None, KL is computed w.r.t. N(0, I) prior.
        """
        returns = self._compute_returns(rewards)

        # policy gradient
        loss_policy = -(returns * log_probs).mean()

        # KL between current logits distribution and zero-mean unit-variance prior
        if mean_history is not None and logvar_history is not None:
            prior_logvar = torch.zeros_like(logvar_history)
            prior_mean = torch.zeros_like(mean_history)

            kl_per_t = 0.5 * (
                prior_logvar
                - logvar_history
                + (torch.exp(logvar_history) + (mean_history - prior_mean) ** 2)
                / torch.exp(prior_logvar + 1e-8)
                - 1.0
            )
            kl_loss = kl_per_t.sum(dim=-1).mean()
        else:
            kl_loss = torch.tensor(0.0, device=self.device)

        loss = loss_policy + self.beta_kl * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(loss_policy.item()),
            "kl_loss": float(kl_loss.item()),
        }
