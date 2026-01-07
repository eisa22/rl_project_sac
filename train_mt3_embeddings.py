"""
Meta-World MT3 Training with Task Embeddings (Custom SAC)

Trains 3 tasks jointly (reach-v3, push-v3, pick-place-v3) using learned task
embeddings instead of one-hot conditioning.

Key points:
- Environment returns PURE observations (no one-hot concatenation)
- `info['task_id']` identifies the task (0..2) for embedding lookup
- Uses `SACAgentEmbedding` from sac_agent_embeddings.py
- Per-task replay buffer ensures balanced sampling per task

Run examples:
    python train_mt3_embeddings.py --run_name mt3_emb_16d
    python train_mt3_embeddings.py --run_name mt3_emb_32d --embedding_dim 32
    python train_mt3_embeddings.py --total_steps 800000 --seed 123
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import gymnasium as gym
import metaworld
import numpy as np
import torch
import wandb
from tqdm import tqdm

from sac_agent_embeddings import SACAgentEmbedding


# ============================================================
#   MT3 Env Wrapper (Embeddings; reach, push, pick-place)
# ============================================================

class MetaWorldMT3EnvEmbedding(gym.Env):
    """
    MT3 wrapper for Task Embeddings.

    Differences vs one-hot wrappers:
    - observation_space = pure Meta-World obs (float32)
    - returns task_id in info for embedding lookup
    - samples tasks uniformly from the 3 selected tasks
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, seed: int = 0, max_episode_steps: int = 150, render_mode: str | None = None):
        super().__init__()
        # Define the 3 tasks (v3 variants)
        self.task_sequence = ["reach-v3", "push-v3", "pick-place-v3"]

        # Build MT10 once and map envs for our selected tasks
        self.mt10 = metaworld.MT10()
        train_classes = self.mt10.train_classes
        train_tasks = list(self.mt10.train_tasks)

        # Instantiate envs only for our 3 tasks
        self.task_envs = {
            name: train_classes[name](render_mode=render_mode)
            for name in self.task_sequence
        }
        # Keep only those tasks whose name we use
        self.tasks = [t for t in train_tasks if t.env_name in self.task_sequence]

        self.task_names = list(self.task_envs.keys())  # keep order from sequence
        self.num_tasks = len(self.task_names)
        self.task_id_map = {name: i for i, name in enumerate(self.task_names)}

        # Reference env for spaces
        ref_env = self.task_envs[self.task_names[0]]
        base_obs_space = ref_env.observation_space

        # Action space (unchanged)
        self.action_space = ref_env.action_space

        # PURE observation space (no one-hot)
        self.observation_space = gym.spaces.Box(
            low=base_obs_space.low.astype(np.float32),
            high=base_obs_space.high.astype(np.float32),
            dtype=np.float32,
        )

        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        self._env = None
        self._current_task = None
        self._tid = None
        self._step = 0

    def _sample_task(self):
        # Uniform sampling among the 3 tasks
        idx = int(self._rng.integers(low=0, high=len(self.tasks)))
        self._current_task = self.tasks[idx]
        env_name = self._current_task.env_name
        self._tid = self.task_id_map[env_name]
        self._env = self.task_envs[env_name]
        self._env.set_task(self._current_task)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._sample_task()

        obs, info = self._env.reset()
        info = {"task_name": self._current_task.env_name, "task_id": int(self._tid)}
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1
        if self._step >= self.max_episode_steps:
            truncated = True

        # Normalize info
        info = {
            "task_name": self._current_task.env_name,
            "task_id": int(self._tid),
            "success": bool(info.get("success", False)) if isinstance(info, dict) else False,
        }
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self._env is not None:
            return self._env.render()

    def close(self):
        for env in self.task_envs.values():
            env.close()


# ============================================================
#   MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MT3 (reach/push/pick-place) with Task Embeddings")
    parser.add_argument("--run_name", type=str, default="mt3_embeddings")
    parser.add_argument("--total_steps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_dim", type=int, default=16,
                        help="Task embedding dimension (e.g., 8, 16, 32)")
    parser.add_argument("--max_episode_steps", type=int, default=150)
    args = parser.parse_args()

    RUN = args.run_name
    TOTAL_STEPS = args.total_steps
    SEED = args.seed
    MAX_STEPS = args.max_episode_steps
    EMB_DIM = args.embedding_dim

    # Config (aligned with custom MT3 defaults; 200k per task → 600k)
    sac_config = {
        "learning_rate": 3e-4,
        "buffer_size": 600_000,
        "learning_starts": 5_000,
        "batch_size": 512,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": -1,
        "ent_coef": "auto",
        "target_entropy": "auto",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": SEED,
        "total_steps": TOTAL_STEPS,
        "max_episode_steps": MAX_STEPS,
        "actor_hidden_sizes": [256, 256],
        "critic_hidden_sizes": [512, 512, 512],
        "embedding_dim": EMB_DIM,
    }

    wandb.init(project="Robot_learning_2025", name=RUN, config=sac_config)

    os.makedirs("./models_mt3_embeddings", exist_ok=True)
    model_dir = f"./models_mt3_embeddings/{RUN}"
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 70)
    print("Meta-World MT3 Training with Task Embeddings (Custom SAC)")
    print("Tasks: reach-v3, push-v3, pick-place-v3")
    print(f"Run: {RUN}")
    print(f"Model dir: {model_dir}")
    print(f"Actor: {sac_config['actor_hidden_sizes']}, Critic: {sac_config['critic_hidden_sizes']}")
    print(f"Embedding dim: {EMB_DIM}")
    print(f"Buffer: {sac_config['buffer_size'] // 3:,} per task × 3 tasks")
    print("=" * 70)

    # --------------------- Env ---------------------
    env = MetaWorldMT3EnvEmbedding(seed=SEED, max_episode_steps=MAX_STEPS)

    # Shapes
    obs_dim = env.observation_space.shape[0]  # pure obs (39)
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    num_tasks = env.num_tasks  # 3

    print("\nEnvironment:")
    print(f"  obs_dim = {obs_dim} (pure)")
    print(f"  act_dim = {act_dim}")
    print(f"  num_tasks = {num_tasks} ({env.task_names})\n")

    # --------------------- Agent ---------------------
    agent = SACAgentEmbedding(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        num_tasks=num_tasks,
        gamma=sac_config["gamma"],
        tau=sac_config["tau"],
        lr=sac_config["learning_rate"],
        hidden_actor=tuple(sac_config["actor_hidden_sizes"]),
        hidden_critic=tuple(sac_config["critic_hidden_sizes"]),
        embedding_dim=EMB_DIM,
        buffer_size_per_task=sac_config["buffer_size"] // num_tasks,
        log_std_min=-20,
    )

    print("✓ Agent initialized with task embeddings")

    # --------------------- Training Loop ---------------------
    print("\n🚀 Starting MT3 training with embeddings...\n")
    obs, info = env.reset()
    task_id = info["task_id"]

    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    task_rewards: dict[str, list[float]] = defaultdict(list)
    task_successes: dict[str, list[bool]] = defaultdict(list)
    task_lengths: dict[str, list[int]] = defaultdict(list)

    with tqdm(total=TOTAL_STEPS, desc="Training", unit="step") as pbar:
        for step in range(TOTAL_STEPS):
            if step < sac_config["learning_starts"]:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, task_id, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_task_id = info["task_id"]  # stays same during episode
            task_name = info["task_name"]

            # Store
            agent.add_experience(obs, action, reward, next_obs, done, task_id)

            obs = next_obs
            task_id = next_task_id
            episode_reward += reward
            episode_length += 1

            # Update
            if step >= sac_config["learning_starts"] and (step % sac_config["train_freq"] == 0):
                losses = agent.update(batch_size=sac_config["batch_size"])
                if step % 1000 == 0:
                    wandb.log({
                        "train/q1_loss": losses.get("q1_loss", 0.0),
                        "train/q2_loss": losses.get("q2_loss", 0.0),
                        "train/actor_loss": losses.get("actor_loss", 0.0),
                        "train/alpha": losses.get("alpha", 0.0),
                        "train/step": step,
                    }, step=step)

            if done:
                episode_count += 1
                task_rewards[task_name].append(episode_reward)
                task_successes[task_name].append(info.get("success", False))
                task_lengths[task_name].append(episode_length)

                pbar.set_postfix({
                    "ep": episode_count,
                    "task": task_name[:10],
                    "rew": f"{episode_reward:.1f}",
                    "succ": "✓" if info.get("success", False) else "✗",
                })

                obs, info = env.reset()
                task_id = info["task_id"]
                episode_reward = 0.0
                episode_length = 0

            if step > 0 and step % 5000 == 0:
                log = {"train/step": step, "train/episodes": episode_count}
                for name in env.task_names:
                    if task_rewards[name]:
                        log[f"train/task/{name}/mean_reward"] = float(np.mean(task_rewards[name][-50:]))
                    if task_successes[name]:
                        log[f"train/task/{name}/success_rate"] = float(np.mean(task_successes[name][-50:]))
                rates = [np.mean(task_successes[n][-50:]) for n in env.task_names if task_successes[n]]
                if rates:
                    log["train/mean_success_all_tasks"] = float(np.mean(rates))
                wandb.log(log, step=step)

            pbar.update(1)

    # --------------------- Final Save ---------------------
    final_path = f"{model_dir}/final_model.pt"
    print(f"\n💾 Saving final model to: {final_path}")
    torch.save({
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_target": agent.q1_target.state_dict(),
        "q2_target": agent.q2_target.state_dict(),
        "log_alpha": agent.log_alpha if agent.log_alpha is not None else None,
        "step": TOTAL_STEPS,
        "config": sac_config,
        # Save embeddings for offline analysis
        "actor_task_embeddings": agent.actor.task_embedding.weight.detach().cpu(),
        "q1_task_embeddings": agent.q1.task_embedding.weight.detach().cpu(),
        "q2_task_embeddings": agent.q2.task_embedding.weight.detach().cpu(),
    }, final_path)

    wandb.finish()
    env.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Total steps: {TOTAL_STEPS:,}")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
