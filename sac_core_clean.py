"""
Clean SAC Implementation with Task Embeddings

Based on best practices from:
- denisyarats/pytorch_sac (clean, minimal reference)
- Haarnoja et al. 2018: Soft Actor-Critic
- SpinningUp SAC

Key improvements over original implementation:
1. Numerically stable tanh log-prob correction
2. Correct entropy tuning and target handling
3. Clean separation of concerns (actor/critic/alpha)
4. Task embeddings integrated cleanly
5. Proper initialization (log_std=-3.0, orthogonal weights)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -20
LOG_STD_MAX = 2


# ============================================================
# Per-Task Replay Buffer
# ============================================================

class PerTaskReplayBuffer:
    """Per-task replay buffers with balanced sampling."""
    
    def __init__(self, obs_dim, act_dim, num_tasks, buffer_size_per_task=100_000):
        self.num_tasks = num_tasks
        self.buffer_size_per_task = buffer_size_per_task
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Separate buffer for each task
        self.buffers = {}
        for task_id in range(num_tasks):
            self.buffers[task_id] = {
                'obs': np.zeros((buffer_size_per_task, obs_dim), dtype=np.float32),
                'acts': np.zeros((buffer_size_per_task, act_dim), dtype=np.float32),
                'rews': np.zeros(buffer_size_per_task, dtype=np.float32),
                'next_obs': np.zeros((buffer_size_per_task, obs_dim), dtype=np.float32),
                'done': np.zeros(buffer_size_per_task, dtype=np.float32),
                'ptr': 0,
                'size': 0
            }
    
    def add(self, obs, act, rew, next_obs, done, task_id):
        """Add transition to the appropriate task buffer."""
        if task_id not in self.buffers:
            print(f"Warning: task_id {task_id} not in buffers, skipping")
            return
        
        buf = self.buffers[task_id]
        ptr = buf['ptr']
        
        buf['obs'][ptr] = obs
        buf['acts'][ptr] = act
        buf['rews'][ptr] = rew
        buf['next_obs'][ptr] = next_obs
        buf['done'][ptr] = done
        
        buf['ptr'] = (ptr + 1) % self.buffer_size_per_task
        buf['size'] = min(buf['size'] + 1, self.buffer_size_per_task)
    
    def sample_batch(self, batch_size=256):
        """Sample equal number of transitions from each active task."""
        # Determine active tasks (with at least 1 sample)
        active_tasks = [tid for tid, buf in self.buffers.items() if buf['size'] > 0]
        if not active_tasks:
            return None
        
        # Balance samples over active tasks (important during curriculum)
        k = len(active_tasks)
        samples_per_task = max(1, batch_size // k)
        
        obs_list, next_obs_list, acts_list, rews_list, done_list, task_id_list = [], [], [], [], [], []
        
        for task_id in active_tasks:
            buf = self.buffers[task_id]
            n = min(samples_per_task, buf['size'])
            idx = np.random.randint(0, buf['size'], size=n)
            
            obs_list.append(buf['obs'][idx])
            next_obs_list.append(buf['next_obs'][idx])
            acts_list.append(buf['acts'][idx])
            rews_list.append(buf['rews'][idx])
            done_list.append(buf['done'][idx])
            task_id_list.append(np.full(n, task_id, dtype=np.int64))
        
        if not obs_list:
            return None
        
        # Concatenate all task samples
        batch = {
            'obs': torch.as_tensor(np.concatenate(obs_list, axis=0), device=device),
            'next_obs': torch.as_tensor(np.concatenate(next_obs_list, axis=0), device=device),
            'acts': torch.as_tensor(np.concatenate(acts_list, axis=0), device=device),
            'rews': torch.as_tensor(np.concatenate(rews_list, axis=0), device=device),
            'done': torch.as_tensor(np.concatenate(done_list, axis=0), device=device),
            'task_ids': torch.as_tensor(np.concatenate(task_id_list, axis=0), dtype=torch.long, device=device),
        }
        return batch
    
    def __len__(self):
        return sum(buf['size'] for buf in self.buffers.values())


# ============================================================
# Network Building Blocks
# ============================================================

def weight_init(m):
    """Orthogonal initialization for better gradient flow."""
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
# Actor with Task Embeddings
# ============================================================

class SquashedGaussianActor(nn.Module):
    """
    Gaussian actor with tanh squashing and task embeddings.
    
    Improvements:
    - Stable tanh log-prob correction (Spinningup formula)
    - log_std initialized to -3.0 (SB3 default)
    - Orthogonal weight initialization
    """
    
    def __init__(self, obs_dim, act_dim, act_limit, num_tasks, 
                 hidden_sizes=(256, 256), embedding_dim=16, log_std_init=-3.0):
        super().__init__()
        
        self.act_limit = act_limit
        
        # Task embedding layer
        self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
        
        # Trunk network
        self.trunk = mlp([obs_dim + embedding_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        
        # Output heads
        self.mu_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], act_dim)
        
        # Initialize
        self.apply(weight_init)
        with torch.no_grad():
            self.log_std_head.bias.fill_(log_std_init)
    
    def forward(self, obs, task_id, deterministic=False, with_logprob=True):
        """
        Forward pass.
        
        Args:
            obs: [batch, obs_dim]
            task_id: [batch] (long)
            deterministic: if True, return mean action
            with_logprob: if True, return log probability
            
        Returns:
            action: [batch, act_dim]
            log_prob: [batch, 1] (if with_logprob)
        """
        # Embed task
        task_emb = self.task_embedding(task_id)
        
        # Concatenate obs with task embedding
        x = torch.cat([obs, task_emb], dim=-1)
        
        # Forward through trunk
        net_out = self.trunk(x)
        
        # Get mu and log_std
        mu = self.mu_head(net_out)
        log_std = self.log_std_head(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # Sample or use mean
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        if with_logprob:
            # Compute log prob with stable tanh correction
            # Reference: SpinningUp SAC
            log_prob = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
            # Correction for tanh squashing: sum_i log(1 - tanh^2(a_i))
            log_prob -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1, keepdim=True)
        else:
            log_prob = None
        
        # Squash to action limits
        action = torch.tanh(pi_action) * self.act_limit
        
        return action, log_prob
    
    def act(self, obs, task_id, deterministic=False):
        """Get action for environment interaction (numpy API)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        task_id_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
        
        with torch.no_grad():
            action, _ = self.forward(obs_t, task_id_t, deterministic, with_logprob=False)
        
        return action.cpu().numpy()[0]


# ============================================================
# Critic with Task Embeddings
# ============================================================

class TwinQCritic(nn.Module):
    """
    Twin Q-networks with task embeddings.
    
    Uses two independent Q-networks for double Q-learning.
    Each Q takes (obs, action, task_embedding) as input.
    """
    
    def __init__(self, obs_dim, act_dim, num_tasks, 
                 hidden_sizes=(512, 512, 512), embedding_dim=16):
        super().__init__()
        
        # Task embeddings (separate from actor)
        self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
        
        # Twin Q-networks
        input_dim = obs_dim + act_dim + embedding_dim
        self.q1 = mlp([input_dim] + list(hidden_sizes) + [1], nn.ReLU, None)
        self.q2 = mlp([input_dim] + list(hidden_sizes) + [1], nn.ReLU, None)
    
    def forward(self, obs, act, task_id):
        """
        Forward pass through both Q-networks.
        
        Returns:
            q1: [batch]
            q2: [batch]
        """
        # Embed task
        task_emb = self.task_embedding(task_id)
        
        # Concatenate inputs
        x = torch.cat([obs, act, task_emb], dim=-1)
        
        # Forward through both Q-networks
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1)
        
        return q1, q2


# ============================================================
# Clean SAC Agent
# ============================================================

class CleanSACAgent:
    """
    Clean SAC implementation with task embeddings.
    
    Key features:
    - Automatic entropy tuning
    - Twin Q-networks with Polyak averaging
    - Stable tanh squashing
    - Task embeddings for multi-task learning
    """
    
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        num_tasks=3,
        gamma=0.99,
        tau=0.005,
        lr=1e-4,
        alpha_lr=1e-4,
        hidden_actor=(256, 256),
        hidden_critic=(512, 512, 512),
        embedding_dim=16,
        target_entropy=None,
        buffer_size_per_task=1_000_000,
    ):
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit
        self.num_tasks = num_tasks
        
        # Networks
        self.actor = SquashedGaussianActor(
            obs_dim, act_dim, act_limit, num_tasks,
            hidden_sizes=hidden_actor,
            embedding_dim=embedding_dim
        ).to(device)
        
        self.critic = TwinQCritic(
            obs_dim, act_dim, num_tasks,
            hidden_sizes=hidden_critic,
            embedding_dim=embedding_dim
        ).to(device)
        
        self.critic_target = TwinQCritic(
            obs_dim, act_dim, num_tasks,
            hidden_sizes=hidden_critic,
            embedding_dim=embedding_dim
        ).to(device)
        
        # Initialize target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Freeze target network
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
        self.replay_buffer = PerTaskReplayBuffer(
            obs_dim, act_dim, num_tasks, buffer_size_per_task
        )
    
    @property
    def alpha(self):
        return self.log_alpha.exp().item()
    
    def act(self, obs, task_id, deterministic=False):
        """Get action for environment interaction."""
        return self.actor.act(obs, task_id, deterministic)
    
    def add_experience(self, obs, action, reward, next_obs, done, task_id):
        """Add transition to replay buffer."""
        self.replay_buffer.add(obs, action, reward, next_obs, done, task_id)
    
    def update(self, batch_size=256):
        """
        Perform one SAC update step.
        
        Returns dict of losses for logging.
        """
        batch = self.replay_buffer.sample_batch(batch_size)
        if batch is None:
            return {}
        
        obs = batch['obs']
        next_obs = batch['next_obs']
        acts = batch['acts']
        rews = batch['rews']
        done = batch['done']
        task_ids = batch['task_ids']
        
        # ========== Critic Update ==========
        with torch.no_grad():
            # Sample next actions from current policy
            next_acts, next_log_probs = self.actor(next_obs, task_ids, deterministic=False, with_logprob=True)
            
            # Compute target Q-values (min of twin Q)
            q1_target, q2_target = self.critic_target(next_obs, next_acts, task_ids)
            q_target = torch.min(q1_target, q2_target)
            
            # Bellman backup (subtract entropy bonus)
            alpha_detached = self.log_alpha.exp().detach()
            backup = rews + self.gamma * (1 - done) * (q_target - alpha_detached * next_log_probs.squeeze(-1))
            # Clip targets to curb exploding Q estimates
            backup = torch.clamp(backup, -1e3, 1e3)
        
        # Current Q estimates
        q1, q2 = self.critic(obs, acts, task_ids)
        
        # MSE loss for both Q-networks
        q1_loss = F.mse_loss(q1, backup)
        q2_loss = F.mse_loss(q2, backup)
        q_loss = q1_loss + q2_loss
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        
        # ========== Actor Update ==========
        # Freeze critic so we don't update it here
        for p in self.critic.parameters():
            p.requires_grad = False
        
        # Sample actions from current policy
        new_acts, log_probs = self.actor(obs, task_ids, deterministic=False, with_logprob=True)
        
        # Compute Q-values for new actions
        q1_new, q2_new = self.critic(obs, new_acts, task_ids)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss: maximize Q(s,a) - alpha * log_prob
        alpha_detached = self.log_alpha.exp().detach()
        actor_loss = (alpha_detached * log_probs.squeeze(-1) - q_new).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()
        
        # Unfreeze critic
        for p in self.critic.parameters():
            p.requires_grad = True
        
        # ========== Entropy Tuning ==========
        # denisyarats formula: minimize (alpha * (-log_prob - target_entropy))
        # This makes alpha increase when entropy too low, decrease when too high
        alpha_loss = (self.log_alpha.exp() * (-log_probs.squeeze(-1) - self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ========== Soft Update Target Network ==========
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_target.data.mul_(1 - self.tau)
                p_target.data.add_(self.tau * p.data)
        
        # Return metrics
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss.item(),
        }
