"""
SAC Agent with Task Embeddings for Multi-Task RL

Key Differences from sac_agent.py:
1. Uses learned task embeddings instead of one-hot encoding
2. Actor/Critic take task_id as separate input instead of concatenated one-hot
3. Embeddings are trained jointly with the policy
4. More compact representation: embedding_dim (e.g. 16) vs num_tasks (e.g. 10)

Based on:
- McLean et al. 2025: Multi-Task RL Enables Parameter Scaling
- Haarnoja et al. 2018: Soft Actor-Critic
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------
# Per-Task Replay Buffer (unchanged from sac_agent.py)
# --------------------------------------------------------
class PerTaskReplayBuffer:
    """
    Per-task replay buffers with equal sampling per task.
    Paper: "per-task replay buffers, with an equal number of samples per task"
    """
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
        """
        Sample equal number of transitions from each task.
        Paper requirement: "equal number of samples per task for each update"
        
        NEW: Returns task_ids in batch for embedding lookup
        """
        # Determine active tasks (with at least 1 sample)
        active_tasks = [tid for tid, buf in self.buffers.items() if buf['size'] > 0]
        if not active_tasks:
            return None

        # Balance samples over active tasks (important during curriculum gating)
        k = len(active_tasks)
        samples_per_task = max(1, batch_size // k)

        obs_list, next_obs_list, acts_list, rews_list, done_list, task_id_list = [], [], [], [], [], []

        for task_id in active_tasks:
            buf = self.buffers.get(task_id)
            if buf is None or buf['size'] <= 0:
                continue

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
            'obs2': torch.as_tensor(np.concatenate(next_obs_list, axis=0), device=device),
            'acts': torch.as_tensor(np.concatenate(acts_list, axis=0), device=device),
            'rews': torch.as_tensor(np.concatenate(rews_list, axis=0), device=device),
            'done': torch.as_tensor(np.concatenate(done_list, axis=0), device=device),
            'task_ids': torch.as_tensor(np.concatenate(task_id_list, axis=0), dtype=torch.long, device=device),  # NEW!
        }
        return batch
    
    def __len__(self):
        return sum(buf['size'] for buf in self.buffers.values())


# --------------------------------------------------------
# Helper: MLP
# --------------------------------------------------------
def mlp(sizes, activation, output_activation=nn.Identity):
    """Multi-layer perceptron."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# --------------------------------------------------------
# Task Embedding Gaussian Policy (NEW!)
# --------------------------------------------------------
class TaskEmbeddingGaussianPolicy(nn.Module):
    """
    Gaussian stochastic policy with learned task embeddings.
    
    Key Changes from GaussianPolicy:
    1. Takes task_id (integer) instead of one-hot vector
    2. Learns task embeddings during training
    3. Concatenates obs with task embedding internally
    
    Architecture:
        Input: obs (obs_dim) + task_id (scalar)
        ↓
        Task Embedding Layer: task_id → embedding_dim
        ↓
        Concatenate: [obs, task_embedding]
        ↓
        MLP Trunk: hidden_sizes (e.g. 256, 256)
        ↓
        Split to mu and log_std heads
    """
    def __init__(self, obs_dim, act_dim, act_limit, num_tasks,
                 hidden_sizes=(256, 256), 
                 embedding_dim=16,  # NEW: Embedding dimension
                 log_std_min=-20, log_std_max=2,
                 log_std_init=-3.0):
        super().__init__()
        
        # NEW: Learned task embeddings (trainable lookup table)
        # Shape: [num_tasks, embedding_dim]
        # Each task gets a unique embedding_dim-dimensional vector
        self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
        
        # Network takes obs + task_embedding as input
        # Instead of obs_dim + num_tasks (one-hot), now obs_dim + embedding_dim
        self.net = mlp([obs_dim + embedding_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_limit = act_limit

        # Initialize log_std bias towards SB3 default for stability
        with torch.no_grad():
            self.log_std_layer.bias.fill_(log_std_init)

    def forward(self, obs, task_id):
        """
        Forward pass through policy network.
        
        Args:
            obs: [batch, obs_dim] - Robot state WITHOUT one-hot encoding
            task_id: [batch] - Integer task IDs (0 to num_tasks-1)
        
        Returns:
            mu: [batch, act_dim] - Mean of action distribution
            std: [batch, act_dim] - Std of action distribution
        """
        # NEW: Lookup learned task embedding
        # task_embedding: [num_tasks, embedding_dim]
        # task_id: [batch] → task_emb: [batch, embedding_dim]
        task_emb = self.task_embedding(task_id)
        
        # Concatenate observation with task embedding
        # Instead of [obs, one_hot], now [obs, learned_embedding]
        combined = torch.cat([obs, task_emb], dim=-1)  # [batch, obs_dim + embedding_dim]
        
        # Rest is same as original GaussianPolicy
        net_out = self.net(combined)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs, task_id):
        """
        Sample action using reparameterization trick.
        
        Args:
            obs: [batch, obs_dim]
            task_id: [batch] - Integer task IDs
            
        Returns:
            action: [batch, act_dim] - Sampled action
            log_prob: [batch, 1] - Log probability of action
            mean_action: [batch, act_dim] - Deterministic (mean) action
        """
        mu, std = self.forward(obs, task_id)
        
        # Normal distribution
        pi_distribution = torch.distributions.Normal(mu, std)
        
        # Reparameterization trick
        pre_tanh_action = pi_distribution.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.act_limit * tanh_action
        
        # Compute log probability with numerically stable tanh correction
        # Reference: SAC stable formulation
        log_prob = pi_distribution.log_prob(pre_tanh_action).sum(-1, keepdim=True)
        correction = 2.0 * (np.log(2) - pre_tanh_action - F.softplus(-2.0 * pre_tanh_action))
        log_prob = log_prob - correction.sum(-1, keepdim=True)
        
        # Mean action (deterministic)
        mu_action = self.act_limit * torch.tanh(mu)
        
        return action, log_prob, mu_action

    def act(self, obs, task_id, deterministic=False):
        """
        Get action for environment interaction.
        
        Args:
            obs: numpy array [obs_dim] - Single observation
            task_id: int - Task ID (0 to num_tasks-1)
            deterministic: bool - If True, return mean action
            
        Returns:
            action: numpy array [act_dim]
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        task_id_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
        
        with torch.no_grad():
            if deterministic:
                _, _, action = self.sample(obs_t, task_id_t)
            else:
                action, _, _ = self.sample(obs_t, task_id_t)
        return action.cpu().numpy()[0]


# --------------------------------------------------------
# Task Embedding Q-Network (NEW!)
# --------------------------------------------------------
class TaskEmbeddingQNetwork(nn.Module):
    """
    Q-function network with learned task embeddings.
    
    Key Changes from QNetwork:
    1. Takes task_id instead of one-hot in observation
    2. Has separate task embedding layer
    3. Concatenates obs, action, and task embedding
    
    Architecture:
        Input: obs (obs_dim) + action (act_dim) + task_id (scalar)
        ↓
        Task Embedding Layer: task_id → embedding_dim
        ↓
        Concatenate: [obs, action, task_embedding]
        ↓
        MLP: hidden_sizes (e.g. 1024, 1024, 1024)
        ↓
        Output: Q-value (scalar)
    """
    def __init__(self, obs_dim, act_dim, num_tasks,
                 hidden_sizes=(1024, 1024, 1024),
                 embedding_dim=16):  # NEW: Embedding dimension
        super().__init__()
        
        # NEW: Learned task embeddings for critic
        # NOTE: These are SEPARATE from actor embeddings (not shared)
        # This allows actor and critic to learn different task representations
        self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
        
        # Q-network takes obs + action + task_embedding
        # Instead of obs_dim + act_dim + num_tasks (one-hot),
        # now obs_dim + act_dim + embedding_dim
        self.q = mlp([obs_dim + act_dim + embedding_dim] + list(hidden_sizes) + [1], 
                     nn.ReLU, nn.Identity)

    def forward(self, obs, act, task_id):
        """
        Forward pass through Q-network.
        
        Args:
            obs: [batch, obs_dim] - Robot state WITHOUT one-hot
            act: [batch, act_dim] - Action
            task_id: [batch] - Integer task IDs
            
        Returns:
            q: [batch] - Q-values
        """
        # NEW: Lookup learned task embedding
        task_emb = self.task_embedding(task_id)  # [batch, embedding_dim]
        
        # Concatenate observation, action, and task embedding
        combined = torch.cat([obs, act, task_emb], dim=-1)
        
        q = self.q(combined)
        return torch.squeeze(q, -1)


# --------------------------------------------------------
# SAC Agent with Task Embeddings
# --------------------------------------------------------
class SACAgentEmbedding:
    """
    Soft Actor-Critic agent with Task Embeddings for Multi-Task RL.
    
    Key Changes from SACAgent:
    1. Uses TaskEmbeddingGaussianPolicy instead of GaussianPolicy
    2. Uses TaskEmbeddingQNetwork instead of QNetwork
    3. obs_dim is now PURE observation dimension (no one-hot appended)
    4. All forward passes include task_id for embedding lookup
    
    Based on:
    - McLean et al. 2025: Multi-Task RL Enables Parameter Scaling
    - Haarnoja et al. 2018: Soft Actor-Critic
    - Task Embeddings: Similar to word embeddings in NLP
    """
    def __init__(
        self,
        obs_dim,           # NEW: Pure obs dimension (e.g. 39), NOT including one-hot
        act_dim,
        act_limit,
        num_tasks=10,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=1e-4,  # REDUCED: from 3e-4 to 1e-4 for stability (SAC can be sensitive)
        hidden_actor=(256, 256),
        hidden_critic=(1024, 1024, 1024),
        embedding_dim=16,  # NEW: Task embedding dimension (tunable!)
        target_entropy=None,
        automatic_entropy_tuning=True,
        buffer_size_per_task=100_000,
        log_std_min=-20,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.act_limit = act_limit
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim  # NEW: Store for logging
        
        # NEW: Actor with Task Embeddings (replaces GaussianPolicy)
        self.actor = TaskEmbeddingGaussianPolicy(
            obs_dim, act_dim, act_limit, num_tasks,
            hidden_sizes=hidden_actor,
            embedding_dim=embedding_dim,
            log_std_min=log_std_min
        ).to(device)
        
        # NEW: Critics with Task Embeddings (replaces QNetwork)
        self.q1 = TaskEmbeddingQNetwork(
            obs_dim, act_dim, num_tasks,
            hidden_sizes=hidden_critic,
            embedding_dim=embedding_dim
        ).to(device)
        
        self.q2 = TaskEmbeddingQNetwork(
            obs_dim, act_dim, num_tasks,
            hidden_sizes=hidden_critic,
            embedding_dim=embedding_dim
        ).to(device)
        
        # Target networks
        self.q1_target = TaskEmbeddingQNetwork(
            obs_dim, act_dim, num_tasks,
            hidden_sizes=hidden_critic,
            embedding_dim=embedding_dim
        ).to(device)
        
        self.q2_target = TaskEmbeddingQNetwork(
            obs_dim, act_dim, num_tasks,
            hidden_sizes=hidden_critic,
            embedding_dim=embedding_dim
        ).to(device)
        
        # Copy initial weights to targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Freeze target networks
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -act_dim  # Heuristic: -dim(A)
            else:
                self.target_entropy = target_entropy
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.log_alpha = None
        
        # Replay buffer (unchanged)
        self.replay_buffer = PerTaskReplayBuffer(
            obs_dim, act_dim, num_tasks, buffer_size_per_task
        )

    def update(self, batch_size=256):
        """
        Update actor and critic networks.
        
        Key Change: All network calls now include task_id for embedding lookup
        """
        batch = self.replay_buffer.sample_batch(batch_size)
        if batch is None:
            return {}
        
        obs = batch['obs']
        obs2 = batch['obs2']
        acts = batch['acts']
        rews = batch['rews']
        done = batch['done']
        task_ids = batch['task_ids']  # NEW: Task IDs for embedding lookup
        
        # ========== Critic Update ==========
        with torch.no_grad():
            # Sample actions from current policy
            # NEW: Pass task_ids to actor
            next_acts, next_log_probs, _ = self.actor.sample(obs2, task_ids)
            
            # Target Q-values
            # NEW: Pass task_ids to target critics
            q1_target = self.q1_target(obs2, next_acts, task_ids)
            q2_target = self.q2_target(obs2, next_acts, task_ids)
            q_target = torch.min(q1_target, q2_target)
            
            # Bellman backup
            backup = rews + self.gamma * (1 - done) * (q_target - self.alpha * next_log_probs.squeeze())
        
        # Q-function loss
        # NEW: Pass task_ids to critics
        q1_pred = self.q1(obs, acts, task_ids)
        q2_pred = self.q2(obs, acts, task_ids)
        q1_loss = F.mse_loss(q1_pred, backup)
        q2_loss = F.mse_loss(q2_pred, backup)
        
        # Update Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # Update Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # ========== Actor Update ==========
        # Sample new actions
        # NEW: Pass task_ids to actor
        new_acts, log_probs, _ = self.actor.sample(obs, task_ids)
        
        # Q-values for new actions
        # NEW: Pass task_ids to critics
        q1_new = self.q1(obs, new_acts, task_ids)
        q2_new = self.q2(obs, new_acts, task_ids)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ========== Alpha Update (unchanged) ==========
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # ========== Target Update (unchanged) ==========
        self._soft_update_target(self.q1, self.q1_target)
        self._soft_update_target(self.q2, self.q2_target)
        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha if isinstance(self.alpha, float) else self.alpha.item(),
        }
    
    def _soft_update_target(self, source, target):
        """Soft update of target network parameters (unchanged)."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def act(self, obs, task_id, deterministic=False):
        """
        Get action for environment interaction.
        
        Args:
            obs: numpy array [obs_dim] - Pure observation (NO one-hot!)
            task_id: int - Task ID (0 to num_tasks-1)
            deterministic: bool
        """
        return self.actor.act(obs, task_id, deterministic)
    
    def add_experience(self, obs, action, reward, next_obs, done, task_id):
        """Add experience to replay buffer (unchanged)."""
        self.replay_buffer.add(obs, action, reward, next_obs, done, task_id)
