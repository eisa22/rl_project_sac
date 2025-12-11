import os
import argparse
import numpy as np
import gymnasium as gym
import metaworld
import torch
import wandb

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


# ============================================================
#   MT10 Env Wrapper (episodic)
# ============================================================

class MetaWorldMT10Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, seed=0, max_episode_steps=150):
        super().__init__()
        self.mt10 = metaworld.MT10()
        self.task_envs = {name: cls() for name, cls in self.mt10.train_classes.items()}
        self.tasks = list(self.mt10.train_tasks)
        self.task_names = list(self.task_envs.keys())
        self.num_tasks = len(self.task_names)
        self.task_id_map = {name: i for i, name in enumerate(self.task_names)}

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
        # Windows-safe task sampling (avoid numpy pickle issues)
        task_idx = self._rng.integers(0, len(self.tasks))
        self._current_task = self.tasks[task_idx]
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

        info = {
            "task_name": self._current_task.env_name,
            "task_id": int(self._tid)
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

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            logs = {k: v for k, v in self.model.logger.name_to_value.items()}
            logs["global_step"] = self.num_timesteps
            wandb.log(logs)
        return True


# ============================================================
#   MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="mt10_run")
    parser.add_argument("--total_steps", type=int, default=5_000_000)  # Paper uses 20M for MT10
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RUN = args.run_name
    TOTAL_STEPS = args.total_steps
    SEED = args.seed
    MAX_STEPS = 150

    # --------------------- W&B ---------------------
    wandb.init(
        project="Robot_learning_2025",
        name=RUN,
        config={
            "algo": "SAC",
            "taskset": "MT10",
            "total_steps": TOTAL_STEPS,
            "seed": SEED,
            "max_episode_steps": MAX_STEPS,
            "batch_size": 512,
            "gamma": 0.99,
            "lr": 3e-4,
            "buffer_size": 1_000_000,
            "hidden_layers": [1024, 1024, 1024],
        }
    )

    os.makedirs("./models_mt10", exist_ok=True)

    print("=" * 70)
    print("Meta-World MT10 Training (SAC, Windows safe)")
    print(f"Run Name        : {RUN}")
    print(f"Total Steps     : {TOTAL_STEPS}")
    print(f"Seed            : {SEED}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    print("Using Device    :", "cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)

    # --------------------- Env ---------------------
    def make_env():
        return Monitor(MetaWorldMT10Env(seed=SEED, max_episode_steps=MAX_STEPS))

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # --------------------- SAC Model ---------------------
    # Hyperparameters based on arXiv:2503.05126v3 (best MT10 results)
    # Paper findings: Critic scaling is most important, use 3 hidden layers with width 1024
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128, 128], qf=[512, 512, 512])  # Small actor, large critic
    )
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=1_000_000,  # Paper uses 100k*N tasks = 1M for MT10
        learning_starts=10_000,
        batch_size=512,  # Paper uses 512 for multi-task
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=-1,  # Update after every step
        ent_coef="auto",  # Automatic entropy tuning (paper uses this)
        target_entropy="auto",
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=SEED,
    )

    # --------------------- Callbacks ---------------------
    checkpoint = CheckpointCallback(
        save_freq=200_000,  # Paper evaluates every 200k steps for MT10
        save_path="./models_mt10/",
        name_prefix="sac_mt10",
        verbose=1
    )

    wandb_cb = WandbCallback(log_freq=1000)

    print("\n🚀 Starting MT10 Training ...\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[checkpoint, wandb_cb],
        progress_bar=True,
    )

    final_path = "./models_mt10/sac_mt10_final"
    print("Saving final model to:", final_path)
    model.save(final_path)

    wandb.finish()
    env.close()
    eval_env.close()

    print("\n🎉 Training finished successfully!")


if __name__ == "__main__":
    main()
