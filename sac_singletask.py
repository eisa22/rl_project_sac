"""
Clean SAC Implementation (Single Task) - NO Task Embeddings

For debugging and tuning on a single task (reach-v3) before adding complexity.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -20
LOG_STD_MAX = 2


# ============================================================
# Replay Buffer (Single Task)
# ============================================================

class ReplayBuffer:
    """Simple replay buffer for single-task learning."""
    
    def __init__(self, obs_dim, act_dim, buffer_size=1_000_000):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rews = np.zeros(buffer_size, dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.done = np.zeros(buffer_size, dtype=np.float32)
    
    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample_batch(self, batch_size=256):
        if self.size < batch_size:
            return None
        
        idx = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'obs': torch.as_tensor(self.obs[idx], device=device),
            'acts': torch.as_tensor(self.acts[idx], device=device),
            'rews': torch.as_tensor(self.rews[idx], device=device),
            'next_obs': torch.as_tensor(self.next_obs[idx], device=device),
            'done': torch.as_tensor(self.done[idx], device=device),
        }
    
    def __len__(self):
        return self.size


# ============================================================
# Network Utilities
# ============================================================

def weight_init(m):
    """Orthogonal initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(sizes, activation=nn.ReLU, output_activation=None):
    """Build MLP with proper initialization."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
        if act is not None:
            layers += [act()]
    net = nn.Sequential(*layers)
    net.apply(weight_init)
    return net


# ============================================================
# Actor
# ============================================================

class GaussianActor(nn.Module):
    """Gaussian actor with tanh squashing."""
    
    def __init__(self, obs_dim, act_dim, act_limit, 
                 hidden_sizes=(256, 256), log_std_init=-3.0):
        super().__init__()
        
        self.act_limit = act_limit
        
        self.trunk = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        
        self.mu_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], act_dim)
        
        self.apply(weight_init)
        with torch.no_grad():
            self.log_std_head.bias.fill_(log_std_init)
    
    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        Args:
            obs: [batch, obs_dim]
            deterministic: if True, return mean action
            with_logprob: if True, return log probability
            
        Returns:
            action: [batch, act_dim]
            log_prob: [batch, 1] (if with_logprob)
        """
        net_out = self.trunk(obs)
        
        mu = self.mu_head(net_out)
        log_std = self.log_std_head(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        if with_logprob:
            # Stable tanh correction (denisyarats formula)
            log_prob = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
            log_prob -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1, keepdim=True)
        else:
            log_prob = None
        
        action = torch.tanh(pi_action) * self.act_limit
        
        return action, log_prob
    
    def act(self, obs, deterministic=False):
        """Get action for environment (numpy API)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.forward(obs_t, deterministic, with_logprob=False)
        return action.cpu().numpy()[0]


# ============================================================
# Critic
# ============================================================

class DoubleQCritic(nn.Module):
    """Twin Q-networks."""
    
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512, 512, 512)):
        super().__init__()
        
        input_dim = obs_dim + act_dim
        self.q1 = mlp([input_dim] + list(hidden_sizes) + [1], nn.ReLU, None)
        self.q2 = mlp([input_dim] + list(hidden_sizes) + [1], nn.ReLU, None)
    
    def forward(self, obs, act):
        """
        Args:
            obs: [batch, obs_dim]
            act: [batch, act_dim]
            
        Returns:
            q1: [batch]
            q2: [batch]
        """
        x = torch.cat([obs, act], dim=-1)
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1)
        return q1, q2


# ============================================================
# Single-Task SAC Agent
# ============================================================

class SingleTaskSAC:
    """SAC for single task (reach-v3)."""
    
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha_lr=1e-4,
        hidden_actor=(256, 256),
        hidden_critic=(512, 512, 512),
        target_entropy=None,
        buffer_size=1_000_000,
    ):
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit
        
        # Networks
        self.actor = GaussianActor(
            obs_dim, act_dim, act_limit,
            hidden_sizes=hidden_actor
        ).to(device)
        
        self.critic = DoubleQCritic(
            obs_dim, act_dim,
            hidden_sizes=hidden_critic
        ).to(device)
        
        self.critic_target = DoubleQCritic(
            obs_dim, act_dim,
            hidden_sizes=hidden_critic
        ).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Entropy tuning
        self.target_entropy = target_entropy if target_entropy is not None else -act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)
    
    @property
    def alpha(self):
        return self.log_alpha.exp().item()
    
    def act(self, obs, deterministic=False):
        return self.actor.act(obs, deterministic)
    
    def add_experience(self, obs, action, reward, next_obs, done):
        self.replay_buffer.add(obs, action, reward, next_obs, done)
    
    def update(self, batch_size=256):
        """One SAC update step."""
        batch = self.replay_buffer.sample_batch(batch_size)
        if batch is None:
            return {}
        
        obs = batch['obs']
        next_obs = batch['next_obs']
        acts = batch['acts']
        rews = batch['rews']
        done = batch['done']
        
        # ========== Critic Update ==========
        with torch.no_grad():
            next_acts, next_log_probs = self.actor(next_obs, deterministic=False, with_logprob=True)
            q1_target, q2_target = self.critic_target(next_obs, next_acts)
            q_target = torch.min(q1_target, q2_target)
            
            alpha_detached = self.log_alpha.exp().detach()
            backup = rews + self.gamma * (1 - done) * (q_target - alpha_detached * next_log_probs.squeeze(-1))
            backup = torch.clamp(backup, -1e3, 1e3)
        
        q1, q2 = self.critic(obs, acts)
        q1_loss = F.mse_loss(q1, backup)
        q2_loss = F.mse_loss(q2, backup)
        q_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        
        # ========== Actor Update ==========
        for p in self.critic.parameters():
            p.requires_grad = False
        
        new_acts, log_probs = self.actor(obs, deterministic=False, with_logprob=True)
        q1_new, q2_new = self.critic(obs, new_acts)
        q_new = torch.min(q1_new, q2_new)
        
        alpha_detached = self.log_alpha.exp().detach()
        actor_loss = (alpha_detached * log_probs.squeeze(-1) - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()
        
        for p in self.critic.parameters():
            p.requires_grad = True
        
        # ========== Entropy Tuning ==========
        alpha_loss = (self.log_alpha.exp() * (-log_probs.squeeze(-1) - self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ========== Target Update ==========
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_target.data.mul_(1 - self.tau)
                p_target.data.add_(self.tau * p.data)
        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss.item(),
        }
