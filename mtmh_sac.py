"""
Multi-Task Multi-Head SAC Implementation

Based on:
- MTRL paper (https://github.com/rainx0r/mtrl)
- Winner: MTMH-SAC with task-specific heads and per-task entropy tuning

Architecture:
- Shared trunk encoder for feature extraction
- Task-specific heads for actor and critic
- Task-specific alpha (entropy coefficient) per task
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import defaultdict

# Mixed Precision Training - compatible with both old and new PyTorch versions
try:
    # PyTorch >= 2.0
    from torch.amp import autocast, GradScaler
    AMP_DEVICE = 'cuda'
except ImportError:
    # PyTorch < 2.0 (cluster compatibility)
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE = None  # old API doesn't need device parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -20
LOG_STD_MAX = 2


# ============================================================
# Per-Task Replay Buffer (reuse from clean SAC)
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
        
        # Balance samples over active tasks
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
# Multi-Head Actor
# ============================================================

class MultiHeadActor(nn.Module):
    """
    Multi-Head Gaussian Actor with shared trunk and task-specific heads.
    
    Architecture:
    - Shared trunk: obs -> trunk_hidden_dim
    - Task heads: trunk_hidden_dim -> (mu, log_std) per task
    """
    
    def __init__(self, obs_dim, act_dim, act_limit, num_tasks,
                 trunk_hidden=(256, 256), head_hidden=(256,), log_std_init=-3.0):
        super().__init__()
        
        self.act_limit = act_limit
        self.num_tasks = num_tasks
        
        # Shared trunk encoder
        trunk_sizes = [obs_dim] + list(trunk_hidden)
        self.trunk = mlp(trunk_sizes, nn.ReLU, nn.ReLU)
        
        # Task-specific heads
        self.task_heads = nn.ModuleList()
        for _ in range(num_tasks):
            head_sizes = [trunk_hidden[-1]] + list(head_hidden)
            head_trunk = mlp(head_sizes, nn.ReLU, nn.ReLU) if head_hidden else nn.Identity()
            
            mu_head = nn.Linear(head_sizes[-1] if head_hidden else trunk_hidden[-1], act_dim)
            log_std_head = nn.Linear(head_sizes[-1] if head_hidden else trunk_hidden[-1], act_dim)
            
            # Initialize log_std
            with torch.no_grad():
                log_std_head.bias.fill_(log_std_init)
            
            self.task_heads.append(nn.ModuleDict({
                'trunk': head_trunk,
                'mu': mu_head,
                'log_std': log_std_head
            }))
        
        self.apply(weight_init)
    
    def forward(self, obs, task_id, deterministic=False, with_logprob=True):
        """
        Forward pass through shared trunk + task-specific head.
        
        Args:
            obs: [batch, obs_dim]
            task_id: [batch] (long) or int
            deterministic: if True, return mean action
            with_logprob: if True, return log probability
            
        Returns:
            action: [batch, act_dim]
            log_prob: [batch, 1] (if with_logprob)
        """
        # Forward through shared trunk
        trunk_out = self.trunk(obs)
        
        # Handle scalar task_id
        if isinstance(task_id, int):
            task_id = torch.full((obs.shape[0],), task_id, dtype=torch.long, device=obs.device)
        
        # Batch processing: group by task_id
        unique_tasks = torch.unique(task_id)

        # Use same dtype as trunk_out (important for Mixed Precision Training)
        actions = torch.zeros((obs.shape[0], self.task_heads[0]['mu'].out_features), device=obs.device, dtype=trunk_out.dtype)
        log_probs = torch.zeros((obs.shape[0], 1), device=obs.device, dtype=trunk_out.dtype) if with_logprob else None
        
        for tid in unique_tasks:
            mask = (task_id == tid)
            trunk_masked = trunk_out[mask]
            
            # Task-specific head
            head = self.task_heads[tid]
            head_out = head['trunk'](trunk_masked)
            
            mu = head['mu'](head_out)
            log_std = head['log_std'](head_out)
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
                log_prob = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
                log_prob -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1, keepdim=True)
                log_probs[mask] = log_prob
            
            # Squash to action limits
            action = torch.tanh(pi_action) * self.act_limit
            actions[mask] = action
        
        return actions, log_probs
    
    def act(self, obs, task_id, deterministic=False):
        """Get action for environment interaction (numpy API)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self.forward(obs_t, task_id, deterministic, with_logprob=False)
        
        return action.cpu().numpy()[0]


# ============================================================
# Multi-Head Critic
# ============================================================

class MultiHeadCritic(nn.Module):
    """
    Multi-Head Twin Q-Critic with shared trunk and task-specific heads.
    
    Architecture:
    - Shared trunk: (obs, act) -> trunk_hidden_dim
    - Twin Q heads per task: trunk_hidden_dim -> 1
    """
    
    def __init__(self, obs_dim, act_dim, num_tasks,
                 trunk_hidden=(512, 512), head_hidden=(512,)):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Shared trunk encoder (takes obs + act)
        trunk_sizes = [obs_dim + act_dim] + list(trunk_hidden)
        self.trunk = mlp(trunk_sizes, nn.ReLU, nn.ReLU)
        
        # Task-specific Q1 and Q2 heads
        self.q1_heads = nn.ModuleList()
        self.q2_heads = nn.ModuleList()
        
        for _ in range(num_tasks):
            head_sizes = [trunk_hidden[-1]] + list(head_hidden) + [1]
            self.q1_heads.append(mlp(head_sizes, nn.ReLU, None))
            self.q2_heads.append(mlp(head_sizes, nn.ReLU, None))
        
        self.apply(weight_init)
    
    def forward(self, obs, act, task_id):
        """
        Forward pass through shared trunk + task-specific Q-heads.
        
        Returns:
            q1: [batch]
            q2: [batch]
        """
        # Forward through shared trunk
        x = torch.cat([obs, act], dim=-1)
        trunk_out = self.trunk(x)
        
        # Handle scalar task_id
        if isinstance(task_id, int):
            task_id = torch.full((obs.shape[0],), task_id, dtype=torch.long, device=obs.device)
        
        # Batch processing: group by task_id
        unique_tasks = torch.unique(task_id)

        # Use same dtype as trunk_out (important for Mixed Precision Training)
        q1_vals = torch.zeros(obs.shape[0], device=obs.device, dtype=trunk_out.dtype)
        q2_vals = torch.zeros(obs.shape[0], device=obs.device, dtype=trunk_out.dtype)
        
        for tid in unique_tasks:
            mask = (task_id == tid)
            trunk_masked = trunk_out[mask]
            
            q1_vals[mask] = self.q1_heads[tid](trunk_masked).squeeze(-1)
            q2_vals[mask] = self.q2_heads[tid](trunk_masked).squeeze(-1)
        
        return q1_vals, q2_vals


# ============================================================
# MTMH SAC Agent
# ============================================================

class MTMHSACAgent:
    """
    Multi-Task Multi-Head SAC Agent.
    
    Key features:
    - Shared trunk + task-specific heads for actor and critic
    - Task-specific alpha (entropy coefficient) tuning
    - Twin Q-networks with Polyak averaging
    """
    
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        num_tasks=3,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha_lr=3e-4,
        trunk_hidden_actor=(256, 256),
        head_hidden_actor=(256,),
        trunk_hidden_critic=(512, 512),
        head_hidden_critic=(512,),
        target_entropy=None,
        buffer_size_per_task=1_000_000,
    ):
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit
        self.num_tasks = num_tasks
        
        # Networks
        self.actor = MultiHeadActor(
            obs_dim, act_dim, act_limit, num_tasks,
            trunk_hidden=trunk_hidden_actor,
            head_hidden=head_hidden_actor
        ).to(device)
        
        self.critic = MultiHeadCritic(
            obs_dim, act_dim, num_tasks,
            trunk_hidden=trunk_hidden_critic,
            head_hidden=head_hidden_critic
        ).to(device)
        
        self.critic_target = MultiHeadCritic(
            obs_dim, act_dim, num_tasks,
            trunk_hidden=trunk_hidden_critic,
            head_hidden=head_hidden_critic
        ).to(device)
        
        # Initialize target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Freeze target network
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Per-task entropy tuning (MTMH key feature!)
        self.target_entropy = target_entropy if target_entropy is not None else -act_dim
        self.log_alphas = nn.Parameter(torch.zeros(num_tasks, device=device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alphas], lr=alpha_lr)
        
        # Replay buffer
        self.replay_buffer = PerTaskReplayBuffer(
            obs_dim, act_dim, num_tasks, buffer_size_per_task
        )

        # Mixed Precision Training (GPU optimization)
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            # Use device parameter only for PyTorch >= 2.0
            self.scaler = GradScaler(AMP_DEVICE) if AMP_DEVICE else GradScaler()
        else:
            self.scaler = None
    
    def get_alpha(self, task_id):
        """Get alpha for specific task."""
        if isinstance(task_id, int):
            return self.log_alphas[task_id].exp().item()
        else:
            return self.log_alphas[task_id].exp()
    
    def act(self, obs, task_id, deterministic=False):
        """Get action for environment interaction (single observation)."""
        return self.actor.act(obs, task_id, deterministic)

    def act_batch(self, obs_batch, task_ids, deterministic=False):
        """
        Batch action inference for multiple environments (GPU optimization).

        Args:
            obs_batch: numpy array [num_envs, obs_dim]
            task_ids: numpy array [num_envs] - task IDs for each observation
            deterministic: bool - if True, return mean actions

        Returns:
            actions: numpy array [num_envs, act_dim]
        """
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        task_ids_t = torch.as_tensor(task_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            actions, _ = self.actor.forward(obs_t, task_ids_t, deterministic=deterministic, with_logprob=False)

        return actions.cpu().numpy()
    
    def add_experience(self, obs, action, reward, next_obs, done, task_id):
        """Add transition to replay buffer."""
        self.replay_buffer.add(obs, action, reward, next_obs, done, task_id)
    
    def update(self, batch_size=256):
        """
        Perform one SAC update step with Mixed Precision Training.

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
        # Compute target (no_grad, so no autocast needed)
        with torch.no_grad():
            # Sample next actions from current policy
            next_acts, next_log_probs = self.actor(next_obs, task_ids, deterministic=False, with_logprob=True)

            # Compute target Q-values (min of twin Q)
            q1_target, q2_target = self.critic_target(next_obs, next_acts, task_ids)
            q_target = torch.min(q1_target, q2_target)

            # Bellman backup (subtract entropy bonus, per-task alpha!)
            alphas = self.get_alpha(task_ids)
            backup = rews + self.gamma * (1 - done) * (q_target - alphas * next_log_probs.squeeze(-1))
            backup = torch.clamp(backup, -1e3, 1e3)

        # Forward pass with mixed precision
        self.critic_optimizer.zero_grad()

        if self.use_amp:
            # Use device parameter only for PyTorch >= 2.0
            autocast_ctx = autocast(AMP_DEVICE) if AMP_DEVICE else autocast()
            with autocast_ctx:
                # Current Q estimates
                q1, q2 = self.critic(obs, acts, task_ids)

                # MSE loss for both Q-networks (loss computed in float32 automatically)
                q1_loss = F.mse_loss(q1, backup)
                q2_loss = F.mse_loss(q2, backup)
                q_loss = q1_loss + q2_loss

            # Backward with gradient scaling
            self.scaler.scale(q_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            # Standard float32 training (fallback)
            q1, q2 = self.critic(obs, acts, task_ids)
            q1_loss = F.mse_loss(q1, backup)
            q2_loss = F.mse_loss(q2, backup)
            q_loss = q1_loss + q2_loss

            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
            self.critic_optimizer.step()
        
        # ========== Actor Update ==========
        # Freeze critic so we don't update it here
        for p in self.critic.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()

        if self.use_amp:
            # Use device parameter only for PyTorch >= 2.0
            autocast_ctx = autocast(AMP_DEVICE) if AMP_DEVICE else autocast()
            with autocast_ctx:
                # Sample actions from current policy
                new_acts, log_probs = self.actor(obs, task_ids, deterministic=False, with_logprob=True)

                # Compute Q-values for new actions
                q1_new, q2_new = self.critic(obs, new_acts, task_ids)
                q_new = torch.min(q1_new, q2_new)

                # Actor loss: maximize Q(s,a) - alpha * log_prob (per-task alpha!)
                alphas = self.get_alpha(task_ids)
                actor_loss = (alphas.detach() * log_probs.squeeze(-1) - q_new).mean()

            # Backward with gradient scaling
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            # Standard float32 training
            new_acts, log_probs = self.actor(obs, task_ids, deterministic=False, with_logprob=True)
            q1_new, q2_new = self.critic(obs, new_acts, task_ids)
            q_new = torch.min(q1_new, q2_new)
            alphas = self.get_alpha(task_ids)
            actor_loss = (alphas.detach() * log_probs.squeeze(-1) - q_new).mean()

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_optimizer.step()

        # Unfreeze critic
        for p in self.critic.parameters():
            p.requires_grad = True
        
        # ========== Per-Task Entropy Tuning ==========
        # Compute alpha loss per task
        unique_tasks = torch.unique(task_ids)
        alpha_losses = []
        
        for tid in unique_tasks:
            mask = (task_ids == tid)
            log_probs_task = log_probs[mask]
            
            # Alpha loss for this task
            alpha_loss = (self.log_alphas[tid].exp() * 
                         (-log_probs_task.squeeze(-1) - self.target_entropy).detach()).mean()
            alpha_losses.append(alpha_loss)
        
        total_alpha_loss = torch.stack(alpha_losses).mean()
        
        self.alpha_optimizer.zero_grad()
        total_alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ========== Soft Update Target Network ==========
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_target.data.mul_(1 - self.tau)
                p_target.data.add_(self.tau * p.data)
        
        # Return metrics (aggregate alphas)
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.log_alphas.exp().mean().item(),  # mean across tasks
            "alpha_loss": total_alpha_loss.item(),
        }
