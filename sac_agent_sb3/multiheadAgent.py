"""
Multi-Task Multi-Head SAC (MTMH-SAC) implementation for Stable-Baselines3.

This module implements MTMH-SAC following the benchmark specifications from Meta-World+
and LEXPOL papers. Key features:
- Separate action heads per task (Multi-Head architecture)
- Multi-task temperature (separate alpha per task)
- Proper hyperparameters for Meta-World benchmarks
- Task-conditioned policy via one-hot encoding

References:
- Meta-World+: https://arxiv.org/abs/2310.16828
- LEXPOL: https://arxiv.org/abs/2310.08343
"""

from dataclasses import dataclass, field
from typing import Sequence, Union, Optional, Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.sac.policies import Actor, ContinuousCritic, SACPolicy
from stable_baselines3.common.callbacks import BaseCallback

# =============================================================================
# Constants following Meta-World benchmark recommendations
# =============================================================================
LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0
MAX_Q_VALUE = 5000.0  # Q-value clipping for stability (from Meta-World+ paper)


"""
class MultiHeadLinear(nn.Module):
    
    #Neural network module with a shared trunk and independent heads for each task.
    

    def __init__(self, input_dim, trunk_layers, head_layers, output_dim, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks

        # 1. Shared Trunk
        trunk = []
        last_dim = input_dim
        for hidden in trunk_layers:
            trunk.append(nn.Linear(last_dim, hidden))
            trunk.append(nn.ReLU())
            last_dim = hidden
        self.trunk = nn.Sequential(*trunk)

        # 2. Independent Heads
        self.heads = nn.ModuleList()
        for _ in range(num_tasks):
            head = []
            h_last = last_dim
            for h_hidden in head_layers:
                head.append(nn.Linear(h_last, h_hidden))
                head.append(nn.ReLU())
                h_last = h_hidden
            # Final output layer
            head.append(nn.Linear(h_last, output_dim))
            self.heads.append(nn.Sequential(*head))

    def forward(self, features, task_indices):
        
        #features: [batch_size, input_dim]
        #task_indices: [batch_size] (integers)
        
        shared_out = self.trunk(features)

        # Route each batch element to the correct head
        outputs = []
        for i in range(len(features)):
            idx = int(task_indices[i].item())
            # Safety clamp
            idx = max(0, min(idx, self.num_tasks - 1))
            out = self.heads[idx](shared_out[i])
            outputs.append(out)

        return torch.stack(outputs)

"""


class MultiHeadLinear(nn.Module):
    """
    Neural network module with a shared trunk and independent heads for each task.
    Optimized for batch processing.
    
    Architecture follows Meta-World+ recommendations:
    - Shared trunk: 3 layers with 400 units each (or configurable)
    - Task-specific heads: 1-2 layers per task
    - ReLU activation throughout
    """

    def __init__(self, input_dim: int, trunk_layers: list, head_layers: list, 
                 output_dim: int, num_tasks: int, use_layer_norm: bool = False):
        super().__init__()
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm

        # 1. Shared Trunk with optional LayerNorm
        trunk = []
        last_dim = input_dim
        for hidden in trunk_layers:
            trunk.append(nn.Linear(last_dim, hidden))
            if use_layer_norm:
                trunk.append(nn.LayerNorm(hidden))
            trunk.append(nn.ReLU())
            last_dim = hidden
        self.trunk = nn.Sequential(*trunk)
        self.trunk_output_dim = last_dim

        # 2. Independent Heads with proper initialization
        self.heads = nn.ModuleList()
        for _ in range(num_tasks):
            head = []
            h_last = last_dim
            for h_hidden in head_layers:
                linear = nn.Linear(h_last, h_hidden)
                # Xavier uniform initialization for hidden layers
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                head.append(linear)
                head.append(nn.ReLU())
                h_last = h_hidden
            # Final output layer with small initialization (like uniform(-1e-3, 1e-3))
            final_layer = nn.Linear(h_last, output_dim)
            nn.init.uniform_(final_layer.weight, -1e-3, 1e-3)
            nn.init.uniform_(final_layer.bias, -1e-3, 1e-3)
            head.append(final_layer)
            self.heads.append(nn.Sequential(*head))

    def forward(self, features: torch.Tensor, task_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with efficient task-based routing.
        
        Args:
            features: [batch_size, input_dim] - Input features (state or state+action)
            task_indices: [batch_size] - Integer task indices
            
        Returns:
            outputs: [batch_size, output_dim] - Task-specific network outputs
        """
        # 1. Run Shared Trunk (Vectorized)
        shared_out = self.trunk(features)

        # 2. Pre-allocate output tensor
        outputs = torch.zeros(
            (features.size(0), self.output_dim),
            device=features.device,
            dtype=features.dtype
        )

        # 3. Vectorized Head Execution - loop over tasks (typically 3-10) not batch
        for i in range(self.num_tasks):
            mask = (task_indices == i)
            if mask.any():
                head_input = shared_out[mask]
                head_output = self.heads[i](head_input)
                outputs[mask] = head_output

        return outputs
    
    def get_trunk_features(self, features: torch.Tensor) -> torch.Tensor:
        """Get shared trunk features for analysis/logging."""
        return self.trunk(features)


class MultiTaskTemperature(nn.Module):
    """
    Multi-task temperature module with separate log_alpha per task.
    
    This follows the benchmark implementation where each task has its own
    entropy coefficient, allowing task-specific exploration-exploitation trade-offs.
    """
    
    def __init__(self, num_tasks: int, initial_temperature: float = 1.0):
        super().__init__()
        self.num_tasks = num_tasks
        # Initialize log_alpha for each task
        self.log_alpha = nn.Parameter(
            torch.full((num_tasks,), np.log(initial_temperature), dtype=torch.float32)
        )
    
    def forward(self, task_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Get temperature values for the given task indices.
        
        Args:
            task_one_hot: [batch_size, num_tasks] - One-hot task encoding
            
        Returns:
            alpha: [batch_size, 1] - Temperature values per sample
        """
        # task_one_hot @ log_alpha gives log_alpha for each sample's task
        log_alpha_per_sample = task_one_hot @ self.log_alpha.unsqueeze(-1)
        return torch.exp(log_alpha_per_sample)
    
    def get_alpha_for_task(self, task_idx: int) -> torch.Tensor:
        """Get alpha for a specific task."""
        return torch.exp(self.log_alpha[task_idx])
    
    def get_all_alphas(self) -> torch.Tensor:
        """Get all alpha values."""
        return torch.exp(self.log_alpha)


class MultiHeadActor(Actor):
    """
    Custom Actor that expects the last `num_tasks` elements of the observation
    to be a one-hot task encoding. It strips this encoding for routing and
    uses a MultiHead network.
    
    Architecture:
    - Shared trunk processes state features
    - Task-specific heads output mean and log_std
    - Log_std is clamped to [-20, 2] following SAC conventions
    """

    def __init__(self, *args, trunk_layers=None, head_layers=None, num_tasks=1, 
                 use_layer_norm: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        trunk_arch = trunk_layers if trunk_layers else [400, 400, 400]
        head_arch = head_layers if head_layers else [128]

        # Input dim is original dim MINUS the one-hot encoding size
        input_dim = self.features_dim - num_tasks
        action_dim = get_action_dim(self.action_space)

        # Replace standard networks with MultiHead versions
        self.mu_multi = MultiHeadLinear(
            input_dim, trunk_arch, head_arch, action_dim, num_tasks, 
            use_layer_norm=use_layer_norm
        )
        self.log_std_multi = MultiHeadLinear(
            input_dim, trunk_arch, head_arch, action_dim, num_tasks,
            use_layer_norm=use_layer_norm
        )

        # Remove original layers to prevent unused parameter errors
        del self.latent_pi, self.mu, self.log_std
        
        # Store for diagnostics
        self._last_log_std_raw = None

    def get_action_dist_params(self, obs: torch.Tensor):
        """
        Get action distribution parameters (mean, log_std) for given observations.
        
        Args:
            obs: [batch_size, obs_dim + num_tasks] - Observations with task encoding
            
        Returns:
            mean_actions: [batch_size, action_dim]
            log_std: [batch_size, action_dim] - Clamped log standard deviation
            kwargs: Empty dict (for compatibility)
        """
        # 1. Extract Task Info (last N elements)
        state = obs[:, :-self.num_tasks]
        task_one_hot = obs[:, -self.num_tasks:]
        task_indices = task_one_hot.argmax(dim=1)

        # 2. Forward Pass via MultiHead
        mean_actions = self.mu_multi(state, task_indices)
        log_std_raw = self.log_std_multi(state, task_indices)

        # 3. Store raw log_std for diagnostics
        self._last_log_std_raw = log_std_raw.detach()

        # 4. Clamp log_std following benchmark conventions
        log_std = torch.clamp(log_std_raw, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean_actions, log_std, {}


class MultiHeadCritic(ContinuousCritic):
    """
    Custom Critic that uses MultiHead networks for Q-value estimation.
    
    Features:
    - Separate Q-networks per task (via multi-head architecture)
    - Q-value clipping for stability (optional, from Meta-World+ paper)
    - Double Q-learning with two separate networks
    """

    def __init__(self, *args, trunk_layers=None, head_layers=None, num_tasks=1,
                 use_layer_norm: bool = False, clip_q_values: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.clip_q_values = clip_q_values
        trunk_arch = trunk_layers if trunk_layers else [400, 400, 400]
        head_arch = head_layers if head_layers else [128]

        # Critic input: State (minus one-hot) + Action
        input_dim = (self.features_extractor.features_dim - num_tasks) + get_action_dim(self.action_space)

        # Replace qf0 and qf1 with MultiHead versions
        self.qf0 = MultiHeadLinear(
            input_dim, trunk_arch, head_arch, 1, num_tasks,
            use_layer_norm=use_layer_norm
        )
        self.qf1 = MultiHeadLinear(
            input_dim, trunk_arch, head_arch, 1, num_tasks,
            use_layer_norm=use_layer_norm
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple:
        """
        Compute Q-values for given observations and actions.
        
        Args:
            obs: [batch_size, obs_dim + num_tasks] - Observations with task encoding
            actions: [batch_size, action_dim] - Actions
            
        Returns:
            q1, q2: Tuple of [batch_size, 1] Q-values from both networks
        """
        # 1. Extract Task Info
        state = obs[:, :-self.num_tasks]
        task_one_hot = obs[:, -self.num_tasks:]
        task_indices = task_one_hot.argmax(dim=1)

        # 2. Prepare Input (State + Action)
        q_input = torch.cat([state, actions], dim=1)

        # 3. Forward through both Q-networks
        q1 = self.qf0(q_input, task_indices)
        q2 = self.qf1(q_input, task_indices)
        
        # 4. Optional Q-value clipping for stability (Meta-World+ recommendation)
        if self.clip_q_values:
            q1 = torch.clamp(q1, -MAX_Q_VALUE, MAX_Q_VALUE)
            q2 = torch.clamp(q2, -MAX_Q_VALUE, MAX_Q_VALUE)

        return q1, q2
    
    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward through Q1 only (for actor update)."""
        state = obs[:, :-self.num_tasks]
        task_one_hot = obs[:, -self.num_tasks:]
        task_indices = task_one_hot.argmax(dim=1)
        q_input = torch.cat([state, actions], dim=1)
        q1 = self.qf0(q_input, task_indices)
        if self.clip_q_values:
            q1 = torch.clamp(q1, -MAX_Q_VALUE, MAX_Q_VALUE)
        return q1


@dataclass
class MHSACAgentSB3Config:
    """
    Configuration for MTMH-SAC following Meta-World benchmark recommendations.
    
    Key hyperparameters based on Meta-World+ and LEXPOL papers:
    - learning_rate: 3e-4 (standard for SAC)
    - batch_size: 512-1280 (scaled with number of tasks)
    - gamma: 0.99
    - tau: 0.005 (soft update coefficient)
    - network: 3 layers with 400 units for trunk, 128 for heads
    - initial_temperature: 1.0 with auto-tuning or 0.1 fixed
    """

    # Core SAC parameters (Meta-World+ recommended values)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Target network soft update coefficient
    buffer_size: int = 1_000_000
    batch_size: int = 1280  # Scaled for multi-task: 128 * num_tasks (MT10)
    learning_starts: int = 4_000  # Warmstart steps
    train_freq: Union[int, tuple[int, str]] = 1
    gradient_steps: int = 1
    
    # Entropy / Temperature settings
    ent_coef: Union[str, float] = "auto"  # "auto" enables automatic tuning
    target_entropy: Union[str, float] = "auto"  # -dim(A) by default
    initial_temperature: float = 1.0  # Initial alpha value
    
    # Network architecture (Meta-World+ recommendations: asymmetric with larger critic)
    # Scientific rationale: Critic capacity is the bottleneck in multi-task RL.
    # The actor only needs to output actions, but the critic must model 
    # complex Q-functions across all tasks - requiring more capacity.
    actor_hidden: Sequence[int] = field(default_factory=lambda: [256, 256])
    critic_hidden: Sequence[int] = field(default_factory=lambda: [512, 512, 512])
    actor_head_hidden: Sequence[int] = field(default_factory=lambda: [128])
    critic_head_hidden: Sequence[int] = field(default_factory=lambda: [256])  # Larger heads for critic
    use_layer_norm: bool = False  # Optional: can improve stability
    clip_q_values: bool = True  # Clip Q-values for stability
    
    # Logging and device
    tensorboard_log: str | None = None
    verbose: int = 1
    device: str = "auto"
    policy_kwargs: dict | None = None
    seed: int = 0
    
    # Multi-Task specific
    num_tasks: int = 3

    def build_policy_kwargs(self) -> dict:
        """Build policy_kwargs dict for SB3 SAC."""
        if self.policy_kwargs is not None:
            return dict(self.policy_kwargs)
        return {
            "actor_hidden": list(self.actor_hidden),
            "critic_hidden": list(self.critic_hidden),
            "num_tasks": self.num_tasks,
            "actor_head_hidden": list(self.actor_head_hidden),
            "critic_head_hidden": list(self.critic_head_hidden),
            "use_layer_norm": self.use_layer_norm,
            "clip_q_values": self.clip_q_values,
        }
    
    def get_effective_batch_size(self) -> int:
        """Get batch size, scaled by num_tasks if using default."""
        # Common practice: 128 * num_tasks
        return self.batch_size if self.batch_size != 1280 else 128 * self.num_tasks


class AsymSACPolicy(SACPolicy):
    """
    Custom SAC policy with Multi-Head architecture for multi-task learning.
    
    Features:
    - Separate actor/critic hidden sizes
    - Multi-head architecture for task-specific processing
    - Optional layer normalization
    - Q-value clipping for stability
    """

    def __init__(self, *args, actor_hidden=None, critic_hidden=None, 
                 actor_head_hidden=None, critic_head_hidden=None, num_tasks=1,
                 use_layer_norm=False, clip_q_values=True, **kwargs):
        # Asymmetric architecture: smaller actor, larger critic
        self._actor_hidden = actor_hidden or [256, 256]
        self._critic_hidden = critic_hidden or [512, 512, 512]
        self.num_tasks = num_tasks
        self.actor_head_hidden = actor_head_hidden or [128]
        self.critic_head_hidden = critic_head_hidden or [256]
        self.use_layer_norm = use_layer_norm
        self.clip_q_values = clip_q_values
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs.update(net_arch=self._actor_hidden)
        return MultiHeadActor(
            **actor_kwargs, 
            trunk_layers=self._actor_hidden, 
            head_layers=self.actor_head_hidden, 
            num_tasks=self.num_tasks,
            use_layer_norm=self.use_layer_norm
        ).to(self.device)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update(net_arch=self._critic_hidden)
        return MultiHeadCritic(
            **critic_kwargs, 
            trunk_layers=self._critic_hidden, 
            head_layers=self.critic_head_hidden, 
            num_tasks=self.num_tasks,
            use_layer_norm=self.use_layer_norm,
            clip_q_values=self.clip_q_values
        ).to(self.device)


def _wrap_env(env: gym.Env | VecEnv) -> gym.Env | VecEnv:
    """Wrap environment if needed (currently pass-through)."""
    if isinstance(env, gym.Env) and not isinstance(env, VecEnv):
        return env
    return env


class MHSACAgentSB3:
    """
    Multi-Task Multi-Head SAC Agent wrapper around Stable-Baselines3.
    
    This agent implements MTMH-SAC following Meta-World benchmark protocols:
    - Multi-head architecture for task-specific processing
    - Proper hyperparameters for multi-task learning
    - Comprehensive logging for diagnostics
    
    Usage:
        config = MHSACAgentSB3Config(num_tasks=3)
        agent = MHSACAgentSB3(env, config)
        agent.learn(total_timesteps=1_000_000)
    """

    def __init__(self, env: gym.Env | VecEnv, config: MHSACAgentSB3Config):
        self.env = env
        self.config = config
        self.num_tasks = config.num_tasks
        
        policy_kwargs = config.build_policy_kwargs()
        
        # Compute effective batch size if using task-scaled default
        effective_batch_size = config.get_effective_batch_size()
        
        self.model = SAC(
            policy=AsymSACPolicy,
            env=_wrap_env(env),
            gamma=config.gamma,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=effective_batch_size,
            learning_starts=config.learning_starts,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            tau=config.tau,
            ent_coef=config.ent_coef,
            target_entropy=config.target_entropy,
            policy_kwargs=policy_kwargs,
            tensorboard_log=config.tensorboard_log,
            verbose=config.verbose,
            device=config.device,
            seed=config.seed,
        )
        
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary for logging."""
        return {
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "tau": self.config.tau,
            "buffer_size": self.config.buffer_size,
            "batch_size": self.config.get_effective_batch_size(),
            "learning_starts": self.config.learning_starts,
            "train_freq": self.config.train_freq,
            "gradient_steps": self.config.gradient_steps,
            "ent_coef": self.config.ent_coef,
            "target_entropy": self.config.target_entropy,
            "num_tasks": self.config.num_tasks,
            "actor_hidden": list(self.config.actor_hidden),
            "critic_hidden": list(self.config.critic_hidden),
            "actor_head_hidden": list(self.config.actor_head_hidden),
            "critic_head_hidden": list(self.config.critic_head_hidden),
            "use_layer_norm": self.config.use_layer_norm,
            "clip_q_values": self.config.clip_q_values,
        }

    def learn(self, total_timesteps: int, **learn_kwargs):
        """Train the agent."""
        return self.model.learn(total_timesteps=total_timesteps, **learn_kwargs)

    def predict(self, observation, deterministic: bool = False):
        """Get action for given observation."""
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        """Save model to disk."""
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env=None, config: MHSACAgentSB3Config = None, **kwargs):
        """
        Load a saved model from disk.
        
        Args:
            path: Path to the saved model file
            env: Environment to use (required for continued training)
            config: Optional config; if None, creates default with num_tasks=3
            **kwargs: Additional arguments passed to SAC.load()
            
        Returns:
            MHSACAgentSB3 instance with loaded model
        """
        if config is None:
            # Create default config - num_tasks will be inferred or use default
            config = MHSACAgentSB3Config(num_tasks=3)
        
        # Create a dummy instance
        instance = object.__new__(cls)
        instance.config = config
        instance.env = env
        instance.num_tasks = config.num_tasks
        
        # Load the underlying SAC model
        instance.model = SAC.load(path, env=_wrap_env(env) if env else None, **kwargs)
        
        return instance
        
    def get_alpha(self) -> float:
        """Get current entropy coefficient."""
        if hasattr(self.model, "log_ent_coef"):
            return float(torch.exp(self.model.log_ent_coef.detach()).item())
        return float(self.model.ent_coef)
    
    def get_q_values_stats(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Sample Q-values from replay buffer and return statistics.
        
        Useful for diagnosing Q-value explosion/collapse.
        """
        rb = getattr(self.model, "replay_buffer", None)
        if rb is None or rb.size() < batch_size:
            return {}
            
        batch = rb.sample(batch_size, env=self.model._vec_normalize_env)
        obs = torch.as_tensor(batch.observations, device=self.model.device)
        actions = torch.as_tensor(batch.actions, device=self.model.device)
        
        with torch.no_grad():
            q1, q2 = self.model.critic(obs, actions)
            
        return {
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "q1_std": float(q1.std().item()),
            "q2_std": float(q2.std().item()),
            "q1_min": float(q1.min().item()),
            "q1_max": float(q1.max().item()),
            "q2_min": float(q2.min().item()),
            "q2_max": float(q2.max().item()),
        }


# =============================================================================
# Enhanced Callbacks for Evaluation and Logging
# =============================================================================

class MTMHSACLoggingCallback(BaseCallback):
    """
    Comprehensive logging callback for MTMH-SAC training.
    
    Logs:
    - Per-task success rates and returns
    - Q-value statistics for stability monitoring
    - Alpha (entropy coefficient) values
    - Gradient magnitudes (if available)
    """
    
    def __init__(self, 
                 num_tasks: int,
                 task_names: list,
                 log_interval: int = 1000,
                 log_q_stats: bool = True,
                 verbose: int = 0):
        super().__init__(verbose)
        self.num_tasks = num_tasks
        self.task_names = task_names
        self.log_interval = log_interval
        self.log_q_stats = log_q_stats
        self.last_log_step = 0
        
        # Per-task metrics accumulators
        self.task_returns = {name: [] for name in task_names}
        self.task_successes = {name: [] for name in task_names}
        self.task_lengths = {name: [] for name in task_names}
        
    def _on_step(self) -> bool:
        # Collect per-task episode metrics from infos
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(len(infos), dtype=bool))
        
        for idx, info in enumerate(infos):
            if dones[idx] and isinstance(info, dict):
                task_name = info.get("task_name", "unknown")
                if task_name in self.task_names:
                    # Episode return from info if available
                    ep_return = info.get("episode", {}).get("r", 0)
                    ep_length = info.get("episode", {}).get("l", 0)
                    success = int(bool(info.get("success", False)))
                    
                    self.task_returns[task_name].append(ep_return)
                    self.task_successes[task_name].append(success)
                    self.task_lengths[task_name].append(ep_length)
        
        # Log at intervals
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self._log_metrics()
            self.last_log_step = self.num_timesteps
            
        return True
    
    def _log_metrics(self):
        """Log accumulated metrics."""
        import wandb
        
        logs = {}
        
        # Per-task metrics
        all_success_rates = []
        for task_name in self.task_names:
            if self.task_successes[task_name]:
                success_rate = float(np.mean(self.task_successes[task_name]))
                all_success_rates.append(success_rate)
                
                task_key = task_name.replace("-", "_")
                logs[f"eval/task/{task_key}/success_rate"] = success_rate
                
                if self.task_returns[task_name]:
                    logs[f"eval/task/{task_key}/return_mean"] = float(np.mean(self.task_returns[task_name]))
                    
                if self.task_lengths[task_name]:
                    logs[f"eval/task/{task_key}/length_mean"] = float(np.mean(self.task_lengths[task_name]))
        
        # Aggregate metrics
        if all_success_rates:
            logs["eval/mean_success_rate"] = float(np.mean(all_success_rates))
            
        # Q-value statistics
        if self.log_q_stats and hasattr(self.model, "replay_buffer"):
            rb = self.model.replay_buffer
            if rb is not None and rb.size() > self.model.batch_size:
                try:
                    batch = rb.sample(min(256, rb.size()), env=self.model._vec_normalize_env)
                    obs = torch.as_tensor(batch.observations, device=self.model.device)
                    actions = torch.as_tensor(batch.actions, device=self.model.device)
                    
                    with torch.no_grad():
                        q1, q2 = self.model.critic(obs, actions)
                        
                    logs["train/q1_mean"] = float(q1.mean().item())
                    logs["train/q2_mean"] = float(q2.mean().item())
                    logs["train/q_min"] = float(min(q1.min().item(), q2.min().item()))
                    logs["train/q_max"] = float(max(q1.max().item(), q2.max().item()))
                except Exception as e:
                    pass  # Skip Q-stats if sampling fails
        
        # Alpha (entropy coefficient)
        if hasattr(self.model, "log_ent_coef"):
            alpha = float(torch.exp(self.model.log_ent_coef.detach()).item())
            logs["train/alpha"] = alpha
        
        # Log to SB3 logger
        for key, value in logs.items():
            self.logger.record(key, value)
            
        # Log to wandb if active
        if wandb.run is not None:
            wandb.log(logs, step=self.num_timesteps)
            
        # Clear accumulators
        for task_name in self.task_names:
            self.task_returns[task_name].clear()
            self.task_successes[task_name].clear()
            self.task_lengths[task_name].clear()