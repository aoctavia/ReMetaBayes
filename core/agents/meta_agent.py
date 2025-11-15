# core/agents/meta_agent.py
# Probabilistic meta-RL ala Probabilistic MAML (amortized latent context)
import torch
import torch.nn as nn
import torch.distributions as dist

from .bayesian_agent import BayesianAgent


class ContextEncoder(nn.Module):
    """
    Simple amortized encoder q_phi(z | support set)
    Input: aggregated features from support trajectories
    Output: mean and logvar of latent context z
    """

    def __init__(self, obs_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        # we aggregate: mean_state [obs_dim], mean_reward [1]
        enc_in_dim = obs_dim + 1
        self.net = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.out_mean = nn.Linear(hidden_dim, latent_dim)
        self.out_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, mean_state: torch.Tensor, mean_reward: torch.Tensor):
        """
        mean_state: [B, obs_dim]
        mean_reward: [B, 1]
        returns: z_mean [B, latent_dim], z_logvar [B, latent_dim]
        """
        x = torch.cat([mean_state, mean_reward], dim=-1)
        h = self.net(x)
        mean = self.out_mean(h)
        logvar = self.out_logvar(h).clamp(-10.0, 2.0)
        return mean, logvar


class MetaAgent(BayesianAgent):
    """
    Probabilistic meta-RL agent (Probabilistic MAML style, but amortized):
    - Inherits Bayesian policy network (stochastic logits)
    - Adds a latent task variable z ~ q_phi(z | support trajectories)
    - Policy is π(a | s, z)
    - Meta-update optimizes both θ (policy) and φ (encoder)
      to perform well on query trajectories after observing support.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        beta_kl: float = 1e-3,
        beta_kl_z: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        # parent will create Bayesian policy with latent_dim input
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            beta_kl=beta_kl,
            gamma=gamma,
            device=device,
        )
        self.latent_dim = latent_dim
        self.beta_kl_z = beta_kl_z

        self.encoder = ContextEncoder(obs_dim, latent_dim, hidden_dim).to(self.device)

        # shared optimizer for policy + encoder (meta-update)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.encoder.parameters()),
            lr=lr,
        )

    # ---------- helper: encode support set into z ----------

    def encode_support(self, support):
        """
        support: dict with keys 'states', 'rewards'
            states:  [T, obs_dim]  (torch.Tensor)
            rewards: [T]           (torch.Tensor or list)
        returns:
            z_mean [1, latent_dim], z_logvar [1, latent_dim], z_sample [1, latent_dim]
        """
        states = support["states"].to(self.device)  # [T, obs_dim]
        rewards = support["rewards"]
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = rewards.to(self.device)

        mean_state = states.mean(dim=0, keepdim=True)  # [1, obs_dim]
        mean_reward = rewards.mean().view(1, 1)        # [1, 1]

        z_mean, z_logvar = self.encoder(mean_state, mean_reward)
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + std * eps  # reparameterization

        return z_mean, z_logvar, z

    # ---------- override act() to optionally use z ----------
    @torch.no_grad()
    def act(self, state, z: torch.Tensor | None = None):
        return super().act(state, z=z)

    # ---------- meta-update: core probabilistic meta-RL ----------
    def meta_update(self, tasks):
        """
        Perform a single meta-update over a batch of tasks.
        Each task is a dict:
            {
                "support": { "states", "actions", "rewards" },
                "query":   { "states", "actions", "rewards" }
            }
        """
        meta_loss = torch.tensor(0.0, device=self.device)
        n_tasks = len(tasks)

        for task in tasks:
            support = task["support"]
            query = task["query"]

            # 1) infer latent context z from support trajectories
            z_mean, z_logvar, z = self.encode_support(support)  # [1, latent_dim]

            # broadcast z over query length
            states_q = query["states"].to(self.device)          # [T_q, obs_dim]
            T_q = states_q.shape[0]
            z_q = z.expand(T_q, -1)                             # [T_q, latent_dim]

            actions_q = query["actions"]
            if not torch.is_tensor(actions_q):
                actions_q = torch.tensor(actions_q, dtype=torch.long, device=self.device)
            else:
                actions_q = actions_q.to(self.device)

            rewards_q = query["rewards"]
            if not torch.is_tensor(rewards_q):
                rewards_q = torch.tensor(rewards_q, dtype=torch.float32, device=self.device)
            else:
                rewards_q = rewards_q.to(self.device)

            # 2) stochastic logits from Bayesian policy πθ(a|s,z)
            mean_logits, logvar_logits = self.policy(states_q, z_q)
            std_logits = torch.exp(0.5 * logvar_logits)
            logits_dist = dist.Normal(mean_logits, std_logits)
            logits = logits_dist.rsample()  # [T_q, action_dim]

            base_probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(base_probs * torch.log(base_probs + 1e-8), dim=-1)  # [T_q]
            temperature = 1.0 + entropy
            scaled_logits = logits / temperature.unsqueeze(-1)
            probs = torch.softmax(scaled_logits, dim=-1)

            m = dist.Categorical(probs)
            log_probs = m.log_prob(actions_q)  # [T_q]

            # 3) compute discounted returns on query set
            returns = self._compute_returns(rewards_q.tolist())

            # 4) policy gradient loss on query
            task_policy_loss = -(returns * log_probs).mean()

            # 5) KL(q(z|support) || N(0, I))
            prior_mean = torch.zeros_like(z_mean)
            prior_logvar = torch.zeros_like(z_logvar)
            kl_z = 0.5 * (
                prior_logvar
                - z_logvar
                + (torch.exp(z_logvar) + (z_mean - prior_mean) ** 2)
                / torch.exp(prior_logvar + 1e-8)
                - 1.0
            ).sum(dim=-1).mean()

            task_loss = task_policy_loss + self.beta_kl_z * kl_z
            meta_loss = meta_loss + task_loss

        meta_loss = meta_loss / max(1, n_tasks)

        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()

        return {"meta_loss": float(meta_loss.item())}
