# core/agents/meta_agent.py
# Catatan:

# inner_update() meniru pembaruan MAML cepat berdasarkan support set.

# meta_update() menggabungkan beberapa tugas (task batches) → outer update.

# Kombinasi dengan BayesianAgent membuatnya menjadi Bayesian-MAML, sangat sesuai dengan tema probabilistic meta-learning.

import torch
import copy
from .bayesian_agent import BayesianAgent

class MetaAgent(BayesianAgent):
    """
    Meta-learning agent using Bayesian MAML approach.
    Performs inner-loop adaptation and outer-loop meta-update.
    """

    def __init__(self, obs_dim, action_dim, lr_inner=1e-3, lr_outer=1e-4, beta=0.01, device="cpu"):
        super().__init__(obs_dim, action_dim, lr=lr_outer, beta=beta, device=device)
        self.lr_inner = lr_inner

    def inner_update(self, support_data):
        """
        Adapt policy to new environment/task using support set (fast adaptation).
        """
        states, actions, rewards = support_data
        new_policy = copy.deepcopy(self.policy)

        logits, (mean, logvar) = new_policy(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        inner_loss = -(rewards * log_probs).mean()

        grads = torch.autograd.grad(inner_loss, new_policy.parameters(), create_graph=True)
        updated_params = [p - self.lr_inner * g for p, g in zip(new_policy.parameters(), grads)]

        # Apply updated parameters
        for p, new_p in zip(new_policy.parameters(), updated_params):
            p.data = new_p.data

        return new_policy, inner_loss.item()

    def meta_update(self, tasks):
        """
        Outer loop: aggregate gradients from multiple tasks (meta-level learning).
        """
        meta_loss = 0.0
        for task in tasks:
            adapted_policy, inner_loss = self.inner_update(task["support"])
            states, actions, rewards = task["query"]
            logits, _ = adapted_policy(states)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
            meta_loss += -(rewards * log_probs).mean()

        meta_loss /= len(tasks)
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        return {"meta_loss": meta_loss.item()}
