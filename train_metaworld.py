import os
import argparse
import numpy as np
import gymnasium as gym
import metaworld
import torch
import wandb
from tqdm import tqdm

from sac_agent import SACAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback


# ============================================================
#   MT10 Env Wrapper (episodic)
# ============================================================

class MetaWorldMT10Env(gym.Env):
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

        # Obs + one-hot task
        low = np.concatenate([
            base_obs_space.low.astype(np.float32),
            np.zeros(self.num_tasks, dtype=np.float32)
        ])
        high = np.concatenate([
            base_obs_space.high.astype(np.float32),
            np.ones(self.num_tasks, dtype=np.float32)
        ])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_episode_steps = max_episode_steps
        self._rng = np.random.default_rng(seed)
        self._env = None
        self._current_task = None
        self._tid = None
        self._step = 0

    def _sample_task(self):
        import random
        if self.fixed_task_name and self.fixed_task_name in self.task_envs:
            env_name = self.fixed_task_name
            # select the first matching task object for the fixed env name
            self._current_task = next(t for t in self.tasks if t.env_name == env_name)
        else:
            self._current_task = random.choice(self.tasks)
            env_name = self._current_task.env_name
        self._tid = self.task_id_map[env_name]
        self._env = self.task_envs[env_name]
        self._env.set_task(self._current_task)

    def _augment_obs(self, obs):
        onehot = np.zeros(self.num_tasks, dtype=np.float32)
        onehot[self._tid] = 1.0
        return np.concatenate([obs.astype(np.float32), onehot])

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._sample_task()
        self._step = 0

        obs, info = self._env.reset()
        obs = self._augment_obs(obs)

        info = {
            "task_name": self._current_task.env_name,
            "task_id": int(self._tid)
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1

        if self._step >= self.max_episode_steps:
            truncated = True

        obs = self._augment_obs(obs)

        # Preserve success flag from the underlying Meta-World env for logging/eval
        success = False
        if isinstance(info, dict):
            success = bool(info.get("success", False))

        info = {
            "task_name": self._current_task.env_name,
            "task_id": int(self._tid),
            "success": success
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        pass


# ============================================================
#   W&B Callback
# ============================================================

class WandbCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_successes = []
        self.task_episode_rewards = {}  # {task_name: [rewards]}
        self.task_episode_successes = {}  # {task_name: [successes]}

    def _on_step(self):
        # Track episode rewards and successes from the environment's info
        # In DummyVecEnv, we get dones and infos for all environments
        if hasattr(self.model.env, 'buf_dones') and hasattr(self.model.env, 'buf_infos'):
            dones = self.model.env.buf_dones
            infos = self.model.env.buf_infos
            
            for i in range(len(dones)):
                if dones[i]:
                    # Episode finished for environment i
                    info = infos[i] if infos and i < len(infos) else {}
                    
                    # Get episode reward from Monitor wrapper
                    if hasattr(self.model.env, 'envs') and i < len(self.model.env.envs):
                        monitor_env = self.model.env.envs[i]
                        if hasattr(monitor_env, 'episode_returns'):
                            ep_reward = monitor_env.episode_returns[-1] if monitor_env.episode_returns else 0.0
                        else:
                            ep_reward = 0.0
                    else:
                        ep_reward = 0.0
                    
                    # Get success status
                    ep_success = info.get("success", False) if isinstance(info, dict) else False
                    
                    # Get task name
                    task_name = info.get("task_name", "unknown") if isinstance(info, dict) else "unknown"
                    
                    # Global tracking
                    self.episode_rewards.append(ep_reward)
                    self.episode_successes.append(int(ep_success))
                    
                    # Task-specific tracking
                    if task_name not in self.task_episode_rewards:
                        self.task_episode_rewards[task_name] = []
                        self.task_episode_successes[task_name] = []
                    
                    self.task_episode_rewards[task_name].append(ep_reward)
                    self.task_episode_successes[task_name].append(int(ep_success))
        
        # Log metrics every log_freq steps
        if self.n_calls % self.log_freq == 0:
            logs = {k: v for k, v in self.model.logger.name_to_value.items()}
            logs["global_step"] = self.num_timesteps
            
            # Log overall episode returns
            if self.episode_rewards:
                logs["train/episode_reward_mean"] = np.mean(self.episode_rewards)
                logs["train/episode_reward_std"] = np.std(self.episode_rewards)
                logs["train/episode_reward_max"] = np.max(self.episode_rewards)
                logs["train/episode_reward_min"] = np.min(self.episode_rewards)
                self.episode_rewards = []
            
            # Log overall success rate
            if self.episode_successes:
                logs["train/success_rate"] = np.mean(self.episode_successes)
                self.episode_successes = []
            
            # Log task-specific metrics
            for task_name in sorted(self.task_episode_rewards.keys()):
                if self.task_episode_rewards[task_name]:
                    task_rewards = self.task_episode_rewards[task_name]
                    task_successes = self.task_episode_successes[task_name]
                    
                    safe_task_name = task_name.replace("-", "_")
                    logs[f"train/task/{safe_task_name}/episode_reward_mean"] = np.mean(task_rewards)
                    logs[f"train/task/{safe_task_name}/episode_reward_std"] = np.std(task_rewards)
                    logs[f"train/task/{safe_task_name}/success_rate"] = np.mean(task_successes)
                    
                    # Clear task-specific buffers
                    self.task_episode_rewards[task_name] = []
                    self.task_episode_successes[task_name] = []
            
            wandb.log(logs)
        return True


# ============================================================
#   MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="mt10_run")
    parser.add_argument("--total_steps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RUN = args.run_name
    TOTAL_STEPS = args.total_steps
    SEED = args.seed
    MAX_STEPS = 150

    # --------------------- SAC Config (Paper: McLean et al. 2025) ---------------------
    # Allow parallel environments for GPU throughput (doesn't affect hyperparameters)
    num_parallel_envs = int(os.environ.get("NUM_PARALLEL_ENVS", "1"))
    
    # PAPER HYPERPARAMETERS (FIXED)
    sac_config = {
        "policy": "MlpPolicy",
        "env": None,
        "learning_rate": 3e-4,
        "buffer_size": 2_000_000,          # Paper: 200k × 10 tasks
        "learning_starts": 10_000,         # Paper standard
        "batch_size": 512,                 # Paper standard
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
        "parallel_envs": num_parallel_envs,  # For GPU throughput, not Paper constraint
    }

    wandb.init(
        project="Robot_learning_2025",
        name=RUN,
        config=sac_config
    )

    os.makedirs("./models_mt10", exist_ok=True)
    
    # Create run-specific model directory
    model_dir = f"./models_mt10/{RUN}"
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 70)
    print("Meta-World MT10 Training (Custom SAC)")
    print("Paper: McLean et al. 2025 - Multi-Task RL Enables Parameter Scaling")
    print(f"Run: {RUN}")
    print(f"Model directory: {model_dir}")
    print(f"Actor: {sac_config['actor_hidden_sizes']}, Critic: {sac_config['critic_hidden_sizes']}")
    print(f"Buffer: {sac_config['buffer_size'] // 10:,}k per task × 10 tasks (PAPER)")
    if num_parallel_envs > 1:
        print(f"⚡ GPU Optimization: {num_parallel_envs}× parallel environments")
    print("=" * 70)

    # --------------------- Env ---------------------
    base_env = MetaWorldMT10Env(seed=SEED, max_episode_steps=MAX_STEPS)
    
    # Get env dimensions
    obs_dim = base_env.observation_space.shape[0]
    act_dim = base_env.action_space.shape[0]
    act_limit = float(base_env.action_space.high[0])
    num_tasks = base_env.num_tasks

    # --------------------- SAC Agent ---------------------
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        num_tasks=num_tasks,
        gamma=sac_config["gamma"],
        tau=sac_config["tau"],
        lr=sac_config["learning_rate"],
        hidden_actor=tuple(sac_config["actor_hidden_sizes"]),
        hidden_critic=tuple(sac_config["critic_hidden_sizes"]),
        buffer_size_per_task=sac_config["buffer_size"] // num_tasks,
        log_std_min=-20,
    )
    
    print(f"✓ SAC Agent initialized with {num_tasks} per-task buffers")
    
    # --------------------- Training Loop ---------------------
    print("\n🚀 Starting MT10 Training ...\n")
    
    obs, info = base_env.reset()
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
            # Select action
            if step < sac_config["learning_starts"]:
                action = base_env.action_space.sample()
            else:
                action = agent.act(obs, deterministic=False)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = base_env.step(action)
            done = terminated or truncated
            task_id = info["task_id"]
            task_name = info["task_name"]
            
            # Store transition
            agent.add_experience(obs, action, reward, next_obs, done, task_id)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Train agent
            if step >= sac_config["learning_starts"] and step % sac_config["train_freq"] == 0:
                agent.update(batch_size=sac_config["batch_size"])
            
            # Episode end
            if done:
                episode_count += 1
                success = int(info.get("success", False))
                
                # Track per-task metrics
                task_rewards[task_name].append(episode_reward)
                task_successes[task_name].append(success)
                task_lengths[task_name].append(episode_length)
                
                # Log overall metrics
                wandb.log({
                    "train/episode_reward": episode_reward,
                    "train/episode_length": episode_length,
                    "train/success": success,
                    "global_step": step,
                })
                
                # Log per-task metrics (every 10 episodes per task)
                if len(task_rewards[task_name]) % 10 == 0:
                    safe_name = task_name.replace("-", "_")
                    wandb.log({
                        f"train/task/{safe_name}/reward_mean": np.mean(task_rewards[task_name][-10:]),
                        f"train/task/{safe_name}/reward_std": np.std(task_rewards[task_name][-10:]),
                        f"train/task/{safe_name}/success_rate": np.mean(task_successes[task_name][-10:]),
                        f"train/task/{safe_name}/length_mean": np.mean(task_lengths[task_name][-10:]),
                    })
                
                # Update progress bar
                pbar.set_postfix({
                    'ep': episode_count,
                    'task': task_name.split('-')[0][:6],
                    'rew': f'{episode_reward:.1f}',
                    'succ': success
                })
                
                # Reset
                obs, info = base_env.reset()
                episode_reward = 0
                episode_length = 0
            
            pbar.update(1)
            
            # Save checkpoint
            if step % 100_000 == 0 and step > 0:
                save_path = f"{model_dir}/checkpoint_{step}.pt"
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'q1': agent.q1.state_dict(),
                    'q2': agent.q2.state_dict(),
                    'step': step,
                }, save_path)
                pbar.write(f"💾 Checkpoint saved: {save_path}")
    
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
    }, final_path)

    wandb.finish()
    base_env.close()

    print("\n🎉 Training finished successfully!")


if __name__ == "__main__":
    main()
