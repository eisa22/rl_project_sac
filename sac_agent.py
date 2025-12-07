import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------
# Per-Task Replay Buffer (Paper: McLean et al. 2025)
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
                'next_obs': np.zeros((buffer_size_per_task, obs_dim), dtype=np.float32),
                'acts': np.zeros((buffer_size_per_task, act_dim), dtype=np.float32),
                'rews': np.zeros((buffer_size_per_task, 1), dtype=np.float32),
                'done': np.zeros((buffer_size_per_task, 1), dtype=np.float32),
                'ptr': 0,
                'size': 0
            }
    
    def add(self, obs, act, rew, next_obs, done, task_id):
        """Add transition to the appropriate task buffer."""
        if task_id not in self.buffers:
            return  # Skip if invalid task_id
        
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
        """
        samples_per_task = max(1, batch_size // self.num_tasks)
        
        obs_list, next_obs_list, acts_list, rews_list, done_list = [], [], [], [], []
        
        for task_id in range(self.num_tasks):
            buf = self.buffers[task_id]
            if buf['size'] == 0:
                continue  # Skip empty buffers
            
            # Sample from this task's buffer
            idxs = np.random.randint(0, buf['size'], size=samples_per_task)
            
            obs_list.append(buf['obs'][idxs])
            next_obs_list.append(buf['next_obs'][idxs])
            acts_list.append(buf['acts'][idxs])
            rews_list.append(buf['rews'][idxs])
            done_list.append(buf['done'][idxs])
        
        if not obs_list:
            return None  # No data yet
        
        # Concatenate all task samples
        batch = {
            'obs': torch.as_tensor(np.concatenate(obs_list, axis=0), device=device),
            'obs2': torch.as_tensor(np.concatenate(next_obs_list, axis=0), device=device),
            'acts': torch.as_tensor(np.concatenate(acts_list, axis=0), device=device),
            'rews': torch.as_tensor(np.concatenate(rews_list, axis=0), device=device),
            'done': torch.as_tensor(np.concatenate(done_list, axis=0), device=device),
        }
        return batch
    
    def __len__(self):
        return sum(buf['size'] for buf in self.buffers.values())


# --------------------------------------------------------
# Networks (following "Grokking Deep RL" structure)
# --------------------------------------------------------
def mlp(sizes, activation, output_activation=nn.Identity):
    """Multi-layer perceptron."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """
    Gaussian stochastic policy for SAC.
    Paper spec: log_std_min = -20
    """
    def __init__(self, obs_dim, act_dim, act_limit,
                 hidden_sizes=(256, 256), log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_limit = act_limit

    def forward(self, obs):
        """Forward pass through policy network."""
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        """
        Sample action using reparameterization trick.
        Returns: action, log_prob, mean_action
        """
        mu, std = self.forward(obs)
        # Normal distribution
        pi_distribution = torch.distributions.Normal(mu, std)
        # Reparameterization trick
        pre_tanh_action = pi_distribution.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.act_limit * tanh_action
        
        # Compute log probability with tanh correction
        log_prob = pi_distribution.log_prob(pre_tanh_action)
        log_prob -= torch.log(self.act_limit * (1 - tanh_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        # Mean action (deterministic)
        mu_action = self.act_limit * torch.tanh(mu)
        
        return action, log_prob, mu_action

    def act(self, obs, deterministic=False):
        """Get action for environment interaction."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.forward(obs_t)
                action = self.act_limit * torch.tanh(mu)
            else:
                action, _, _ = self.sample(obs_t)
        return action.cpu().numpy()[0]


class QNetwork(nn.Module):
    """
    Q-function network (critic).
    Paper: Critic scaling is critical (1024³ for MTRL)
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(1024, 1024, 1024)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU, nn.Identity)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


# --------------------------------------------------------
# SAC Agent (Paper-compliant + Grokking Deep RL)
# --------------------------------------------------------
class SACAgent:
    """
    Soft Actor-Critic agent for Multi-Task RL.
    
    Based on:
    - McLean et al. 2025: Multi-Task RL Enables Parameter Scaling
    - Haarnoja et al. 2018: Soft Actor-Critic
    - Grokking Deep Reinforcement Learning (Stevens & Casper)
    
    Key features:
    - Per-task replay buffers with equal sampling
    - Separate actor (256²) and critic (1024³) architectures
    - Automatic entropy tuning
    """
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        num_tasks=10,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        hidden_actor=(256, 256),
        hidden_critic=(1024, 1024, 1024),
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
        
        # Networks
        self.actor = GaussianPolicy(
            obs_dim, act_dim, act_limit, 
            hidden_sizes=hidden_actor,
            log_std_min=log_std_min
        ).to(device)
        
        self.q1 = QNetwork(obs_dim, act_dim, hidden_critic).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_critic).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_critic).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_critic).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        
        # Entropy tuning
        if target_entropy is None:
            target_entropy = -act_dim
        self.target_entropy = target_entropy
        
        if automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = None
        
        # Per-task replay buffers (Paper requirement)
        self.replay_buffer = PerTaskReplayBuffer(
            obs_dim, act_dim, num_tasks, buffer_size_per_task
        )
        
        # Training stats
        self.num_updates = 0

    def update(self, batch_size=256):
        """
        SAC optimization step (following Grokking Deep RL structure).
        
        Steps:
        1. Sample equal data from each task buffer
        2. Update critics (Q-functions)
        3. Update actor (policy)
        4. Update entropy coefficient (alpha)
        5. Soft-update target networks
        """
        batch = self.replay_buffer.sample_batch(batch_size)
        if batch is None:
            return  # Not enough data yet
        
        obs, next_obs = batch["obs"], batch["obs2"]
        acts, rews, done = batch["acts"], batch["rews"], batch["done"]
        
        # --------------------
        # 1. Critic Update
        # --------------------
        with torch.no_grad():
            # Sample next actions from current policy
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            # Target Q-values (minimum of two Q-functions)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob.squeeze(-1)
            # Bellman backup
            target_q = rews.squeeze(-1) + (1 - done.squeeze(-1)) * self.gamma * q_next
        
        # Current Q-value estimates
        q1_pred = self.q1(obs, acts)
        q2_pred = self.q2(obs, acts)
        
        # Q-function losses
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        
        # Update Q-function 1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # Update Q-function 2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # --------------------
        # 2. Actor Update
        # --------------------
        # Sample new actions from current policy
        new_actions, log_prob, _ = self.actor.sample(obs)
        # Q-values for new actions
        q1_new = self.q1(obs, new_actions)
        q2_new = self.q2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss: maximize Q - alpha * entropy
        actor_loss = (self.alpha * log_prob - q_new.unsqueeze(-1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --------------------
        # 3. Entropy Tuning (Alpha Update)
        # --------------------
        if self.automatic_entropy_tuning:
            # Alpha loss: match target entropy
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # --------------------
        # 4. Soft Update Target Networks
        # --------------------
        self._soft_update_target(self.q1, self.q1_target)
        self._soft_update_target(self.q2, self.q2_target)
        
        self.num_updates += 1
        
        # --------------------
        # Logging
        # --------------------
        if wandb.run is not None:
            wandb.log({
                "train/q1_loss": q1_loss.item(),
                "train/q2_loss": q2_loss.item(),
                "train/actor_loss": actor_loss.item(),
                "train/alpha": self.alpha,
                "train/q1_mean": q1_pred.mean().item(),
                "train/q2_mean": q2_pred.mean().item(),
                "train/target_q_mean": target_q.mean().item(),
                "train/log_prob_mean": log_prob.mean().item(),
            })
    
    def _soft_update_target(self, source, target):
        """Soft update: target = tau * source + (1 - tau) * target"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def act(self, obs, deterministic=False):
        """Get action for environment interaction."""
        return self.actor.act(obs, deterministic)
    
    def add_experience(self, obs, action, reward, next_obs, done, task_id):
        """Add transition to per-task replay buffer."""
        self.replay_buffer.add(obs, action, reward, next_obs, done, task_id)
