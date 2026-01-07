"""
Meta-World MT10 Training with Task Embeddings (Custom SAC)

Key Differences from train_metaworld.py:
1. Uses sac_agent_embeddings.py (learned task embeddings)
2. Observation space is PURE 39D (no one-hot encoding appended)
3. Environment returns task_id in info dict (already does!)
4. Agent receives task_id separately, not in observation

Based on:
- McLean et al. 2025: Multi-Task RL Enables Parameter Scaling
- Task Embeddings: Learned representations instead of one-hot
"""

import argparse
import os
import numpy as np
import torch
import wandb
import gymnasium as gym
import metaworld
from tqdm import tqdm

from sac_agent_embeddings import SACAgentEmbedding


# ============================================================
#   MT10 Env Wrapper (Modified for Task Embeddings)
# ============================================================

class MetaWorldMT10EnvEmbedding(gym.Env):
    """
    MT10 wrapper for Task Embeddings.
    
    Key Changes from MetaWorldMT10Env:
    1. Observation space is PURE obs (39D), NO one-hot encoding
    2. _augment_obs() is REMOVED (no concatenation)
    3. Task ID is returned in info dict only
    4. Agent receives task_id separately from observation
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, seed=0, max_episode_steps=150, render_mode=None, fixed_task_name=None):
        super().__init__()
        self.mt10 = metaworld.MT10()
        self.task_envs = {name: cls(render_mode=render_mode) for name, cls in self.mt10.train_classes.items()}
        self.tasks = list(self.mt10.train_tasks)
        self.task_names = list(self.task_envs.keys())
        self.num_tasks = len(self.task_names)
        self.task_id_map = {name: i for i, name in enumerate(self.task_names)}
        self.render_mode = render_mode
        self.fixed_task_name = fixed_task_name

        # Reference space
        ref_env = self.task_envs[self.task_names[0]]
        base_obs_space = ref_env.observation_space

        # Action space
        self.action_space = ref_env.action_space

        # NEW: Observation space is PURE observation (39D)
        # NO one-hot encoding appended!
        # This is the key difference from train_metaworld.py
        self.observation_space = gym.spaces.Box(
            low=base_obs_space.low.astype(np.float32),
            high=base_obs_space.high.astype(np.float32),
            dtype=np.float32
        )

        self.max_episode_steps = max_episode_steps
        self._rng = np.random.default_rng(seed)
        self._env = None
        self._current_task = None
        self._tid = None
        self._step = 0

    def _sample_task(self):
        """Sample a random task (unchanged)."""
        if self.fixed_task_name and self.fixed_task_name in self.task_envs:
            env_name = self.fixed_task_name
            for task in self.tasks:
                if task.env_name == env_name:
                    self._current_task = task
                    break
        else:
            # Windows-safe task sampling
            task_idx = self._rng.integers(0, len(self.tasks))
            self._current_task = self.tasks[task_idx]
            env_name = self._current_task.env_name
        
        self._tid = self.task_id_map[env_name]
        self._env = self.task_envs[env_name]
        self._env.set_task(self._current_task)

    def reset(self, seed=None, options=None):
        """
        Reset environment.
        
        Returns:
            obs: np.ndarray [39] - PURE observation (NO one-hot!)
            info: dict - Contains task_name and task_id
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._sample_task()
        self._step = 0

        obs, info = self._env.reset()
        
        # NEW: Return PURE observation (no augmentation!)
        # Task ID is in info dict for agent to use
        info = {
            "task_name": self._current_task.env_name,
            "task_id": int(self._tid)  # Agent uses this for embedding lookup
        }
        return obs.astype(np.float32), info

    def step(self, action):
        """
        Step environment.
        
        Returns:
            obs: np.ndarray [39] - PURE observation (NO one-hot!)
            reward: float
            terminated: bool
            truncated: bool
            info: dict - Contains task_name, task_id, success
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1

        if self._step >= self.max_episode_steps:
            truncated = True

        # NEW: Return PURE observation (no augmentation!)
        # Task ID remains in info for next step
        info = {
            "task_name": self._current_task.env_name,
            "task_id": int(self._tid),  # Important: Agent needs this!
            "success": info.get("success", False)
        }
        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        if self._env is not None:
            return self._env.render()

    def close(self):
        pass


# ============================================================
#   MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="mt10_embeddings")
    parser.add_argument("--total_steps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_dim", type=int, default=16,
                       help="Task embedding dimension (8, 16, 32, 64)")
    args = parser.parse_args()

    RUN = args.run_name
    TOTAL_STEPS = args.total_steps
    SEED = args.seed
    MAX_STEPS = 150
    EMBEDDING_DIM = args.embedding_dim

    # --------------------- SAC Config with Task Embeddings ---------------------
    sac_config = {
        "policy": "MlpPolicy",
        "env": None,
        "learning_rate": 3e-4,
        "buffer_size": 2_000_000,          # 200k per task × 10 tasks
        "learning_starts": 10_000,
        "batch_size": 512,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": -1,
        "ent_coef": "auto",
        "target_entropy": "auto",
        "verbose": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": SEED,
        "total_steps": TOTAL_STEPS,
        "max_episode_steps": MAX_STEPS,
        "run_name": RUN,
        "actor_hidden_sizes": [256, 256],
        "critic_hidden_sizes": [1024, 1024, 1024],
        "embedding_dim": EMBEDDING_DIM,  # NEW!
        "embedding_type": "learned",      # NEW: To distinguish from one-hot
    }

    wandb.init(
        project="Robot_learning_2025",
        name=RUN,
        config=sac_config
    )

    os.makedirs("./models_mt10_embeddings", exist_ok=True)
    
    # Create run-specific model directory
    model_dir = f"./models_mt10_embeddings/{RUN}"
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 70)
    print("Meta-World MT10 Training with Task Embeddings")
    print("Paper: McLean et al. 2025 - Multi-Task RL Enables Parameter Scaling")
    print("NEW: Learned Task Embeddings instead of One-Hot Encoding")
    print(f"Run: {RUN}")
    print(f"Model directory: {model_dir}")
    print(f"Actor: {sac_config['actor_hidden_sizes']}, Critic: {sac_config['critic_hidden_sizes']}")
    print(f"Embedding Dimension: {EMBEDDING_DIM}D (learned)")
    print(f"Observation Dimension: 39D (pure, no one-hot)")
    print(f"Buffer: {sac_config['buffer_size'] // 10:,}k per task × 10 tasks")
    print("=" * 70)

    # --------------------- Env ---------------------
    # NEW: Uses MetaWorldMT10EnvEmbedding (no one-hot augmentation)
    base_env = MetaWorldMT10EnvEmbedding(seed=SEED, max_episode_steps=MAX_STEPS)
    
    # Get env dimensions
    # NEW: obs_dim is PURE dimension (39), not including one-hot
    obs_dim = base_env.observation_space.shape[0]  # 39 instead of 49
    act_dim = base_env.action_space.shape[0]
    act_limit = float(base_env.action_space.high[0])
    num_tasks = base_env.num_tasks

    print(f"\n📊 Environment Info:")
    print(f"  Observation Dimension: {obs_dim} (pure state, no task encoding)")
    print(f"  Action Dimension: {act_dim}")
    print(f"  Number of Tasks: {num_tasks}")
    print(f"  Task Names: {base_env.task_names}\n")

    # --------------------- SAC Agent with Embeddings ---------------------
    # NEW: Uses SACAgentEmbedding instead of SACAgent
    agent = SACAgentEmbedding(
        obs_dim=obs_dim,  # 39 (pure), not 49 (with one-hot)
        act_dim=act_dim,
        act_limit=act_limit,
        num_tasks=num_tasks,
        gamma=sac_config["gamma"],
        tau=sac_config["tau"],
        lr=sac_config["learning_rate"],
        hidden_actor=tuple(sac_config["actor_hidden_sizes"]),
        hidden_critic=tuple(sac_config["critic_hidden_sizes"]),
        embedding_dim=EMBEDDING_DIM,  # NEW: Embedding dimension
        buffer_size_per_task=sac_config["buffer_size"] // num_tasks,
        log_std_min=-20,
    )
    
    print(f"✓ SAC Agent with Task Embeddings initialized")
    print(f"  Embedding Dimension: {EMBEDDING_DIM}")
    print(f"  Actor Embeddings: {num_tasks} tasks × {EMBEDDING_DIM}D = {num_tasks * EMBEDDING_DIM} params")
    print(f"  Critic Embeddings: {num_tasks} tasks × {EMBEDDING_DIM}D = {num_tasks * EMBEDDING_DIM} params (×2)")
    
    # --------------------- Training Loop ---------------------
    print("\n🚀 Starting MT10 Training with Task Embeddings ...\n")
    
    obs, info = base_env.reset()
    task_id = info["task_id"]  # NEW: Extract task_id from info
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    # Per-task tracking
    from collections import defaultdict
    task_rewards = defaultdict(list)
    task_successes = defaultdict(list)
    task_lengths = defaultdict(list)
    
    with tqdm(total=TOTAL_STEPS, desc="Training", unit="step") as pbar:
        for step in range(TOTAL_STEPS):
            
            # NEW: Get action with task_id (not in observation!)
            # Agent uses task_id for embedding lookup
            if step < sac_config["learning_starts"]:
                action = base_env.action_space.sample()
            else:
                action = agent.act(obs, task_id, deterministic=False)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = base_env.step(action)
            next_task_id = info["task_id"]  # Should be same during episode
            done = terminated or truncated
            
            task_name = info["task_name"]
            episode_reward += reward
            episode_length += 1
            
            # NEW: Add experience with task_id
            # obs is 39D (pure), no one-hot included
            agent.add_experience(obs, action, reward, next_obs, done, task_id)
            
            obs = next_obs
            task_id = next_task_id  # Update task_id for next step
            
            # Update networks
            if step >= sac_config["learning_starts"] and step % sac_config["train_freq"] == 0:
                losses = agent.update(batch_size=sac_config["batch_size"])
                
                # Log training losses
                if step % 1000 == 0:
                    wandb.log({
                        "train/q1_loss": losses["q1_loss"],
                        "train/q2_loss": losses["q2_loss"],
                        "train/actor_loss": losses["actor_loss"],
                        "train/alpha": losses["alpha"],
                        "train/step": step
                    }, step=step)
            
            # Episode end
            if done or truncated:
                episode_count += 1
                
                # Track per-task metrics
                task_rewards[task_name].append(episode_reward)
                task_successes[task_name].append(info.get("success", False))
                task_lengths[task_name].append(episode_length)
                
                # Log episode
                pbar.set_postfix({
                    'ep': episode_count,
                    'task': task_name[:6],
                    'reward': f'{episode_reward:.1f}',
                    'success': '✓' if info.get("success", False) else '✗'
                })
                
                # Reset
                obs, info = base_env.reset()
                task_id = info["task_id"]  # NEW: Get task_id for new episode
                episode_reward = 0
                episode_length = 0
            
            pbar.update(1)
            
            # Periodic logging
            if step > 0 and step % 5000 == 0:
                # Per-task success rates
                task_stats = {}
                for task_name in base_env.task_names:
                    if len(task_successes[task_name]) > 0:
                        recent_successes = task_successes[task_name][-50:]
                        success_rate = np.mean(recent_successes)
                        mean_reward = np.mean(task_rewards[task_name][-50:]) if len(task_rewards[task_name]) > 0 else 0
                        
                        task_stats[f"train/task/{task_name}/success_rate"] = success_rate
                        task_stats[f"train/task/{task_name}/mean_reward"] = mean_reward
                
                # Overall metrics
                all_success_rates = [np.mean(task_successes[t][-50:]) 
                                   for t in base_env.task_names if len(task_successes[t]) > 0]
                if all_success_rates:
                    task_stats["train/mean_success_all_tasks"] = np.mean(all_success_rates)
                
                task_stats["train/step"] = step
                task_stats["train/episodes"] = episode_count
                wandb.log(task_stats, step=step)
                
                # Console output
                if all_success_rates:
                    print(f"\n[Step {step:,}] Mean Success Rate: {np.mean(all_success_rates):.1%}")
            
            # Save checkpoint
            if step > 0 and step % 100_000 == 0:
                checkpoint_path = f"{model_dir}/checkpoint_{step}.pt"
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'q1': agent.q1.state_dict(),
                    'q2': agent.q2.state_dict(),
                    'q1_target': agent.q1_target.state_dict(),
                    'q2_target': agent.q2_target.state_dict(),
                    'log_alpha': agent.log_alpha if agent.log_alpha is not None else None,
                    'step': step,
                    'config': sac_config,
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # --------------------- Final Save ---------------------
    final_path = f"{model_dir}/final_model.pt"
    print(f"\n💾 Saving final model to: {final_path}")
    torch.save({
        'actor': agent.actor.state_dict(),
        'q1': agent.q1.state_dict(),
        'q2': agent.q2.state_dict(),
        'q1_target': agent.q1_target.state_dict(),
        'q2_target': agent.q2_target.state_dict(),
        'log_alpha': agent.log_alpha if agent.log_alpha is not None else None,
        'step': TOTAL_STEPS,
        'config': sac_config,
        # NEW: Save task embeddings for analysis
        'actor_task_embeddings': agent.actor.task_embedding.weight.detach().cpu(),
        'q1_task_embeddings': agent.q1.task_embedding.weight.detach().cpu(),
        'q2_task_embeddings': agent.q2.task_embedding.weight.detach().cpu(),
    }, final_path)

    # --------------------- Final Statistics ---------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Total Steps: {TOTAL_STEPS:,}")
    print(f"Total Episodes: {episode_count}")
    print(f"\nPer-Task Performance (last 50 episodes):")
    for task_name in sorted(base_env.task_names):
        if len(task_successes[task_name]) >= 10:
            recent_success = np.mean(task_successes[task_name][-50:])
            recent_reward = np.mean(task_rewards[task_name][-50:])
            print(f"  {task_name:20s}: {recent_success:5.1%} success | {recent_reward:6.1f} reward")
    
    all_success_rates = [np.mean(task_successes[t][-50:]) 
                        for t in base_env.task_names if len(task_successes[t]) >= 10]
    if all_success_rates:
        print(f"\n🎯 Overall Mean Success Rate: {np.mean(all_success_rates):.1%}")
    print("=" * 70)

    wandb.finish()
    base_env.close()

    print("\n🎉 Training finished successfully!")
    print(f"\n💡 To analyze task embeddings, load '{final_path}' and visualize:")
    print("   embeddings = torch.load('final_model.pt')['actor_task_embeddings']")
    print("   # Use t-SNE or PCA to visualize task similarities!")


if __name__ == "__main__":
    main()
