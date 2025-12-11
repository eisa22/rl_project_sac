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
#   MT3 Curriculum Env Wrapper (reach → push → pick-place)
# ============================================================

class MetaWorldMT3CurriculumEnv(gym.Env):
    """
    MT3 with Curriculum Learning:
    Phase 1: reach-v3 only (until 60% success)
    Phase 2: reach + push (until push reaches 50% success)
    Phase 3: all 3 tasks (reach + push + pick-place)
    
    Similar structure to MetaWorldMT10Env but with curriculum task unlocking.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, seed=0, max_episode_steps=150, render_mode=None, 
                 curriculum_thresholds=None, fixed_task_name=None):
        super().__init__()
        
        # MT3 task curriculum: reach → push → pick-place
        self.curriculum_tasks = ['reach-v3', 'push-v3', 'pick-place-v3']
        
        # Curriculum thresholds: success rate needed to unlock next task
        if curriculum_thresholds is None:
            self.curriculum_thresholds = [0.6, 0.5, 0.0]  # [reach, push, pick-place]
        else:
            self.curriculum_thresholds = curriculum_thresholds
        
        # Build environments for MT3 tasks only (from MT10)
        self.mt10 = metaworld.MT10()
        all_train_classes = self.mt10.train_classes
        all_train_tasks = list(self.mt10.train_tasks)
        
        self.task_envs = {name: all_train_classes[name](render_mode=render_mode) 
                         for name in self.curriculum_tasks}
        self.tasks = [t for t in all_train_tasks if t.env_name in self.curriculum_tasks]
        self.task_names = self.curriculum_tasks
        self.num_tasks = len(self.curriculum_tasks)  # Always 3 for one-hot encoding
        self.task_id_map = {name: i for i, name in enumerate(self.task_names)}
        self.render_mode = render_mode
        self.fixed_task_name = fixed_task_name
        
        # Curriculum state tracking
        self.active_tasks = [self.curriculum_tasks[0]]  # Start with reach only
        self.task_success_history = {name: [] for name in self.curriculum_tasks}
        self.curriculum_unlocked = [True, False, False]  # [reach, push, pick-place]

        # Reference space (same as MT10Env)
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
        """Sample from currently active tasks in curriculum."""
        import random
        if self.fixed_task_name and self.fixed_task_name in self.task_envs:
            env_name = self.fixed_task_name
            self._current_task = next(t for t in self.tasks if t.env_name == env_name)
        else:
            # Sample only from active curriculum tasks
            active_task_pool = [t for t in self.tasks if t.env_name in self.active_tasks]
            self._current_task = random.choice(active_task_pool)
            env_name = self._current_task.env_name
        self._tid = self.task_id_map[env_name]
        self._env = self.task_envs[env_name]
        self._env.set_task(self._current_task)
    
    def update_curriculum(self, task_name, success):
        """Track success and unlock next task if threshold met."""
        self.task_success_history[task_name].append(success)
        
        # Check if we should unlock next task
        for i, name in enumerate(self.curriculum_tasks):
            if self.curriculum_unlocked[i]:
                # Calculate recent success rate (last 50 episodes)
                recent_history = self.task_success_history[name][-50:]
                if len(recent_history) >= 20:  # Require minimum episodes
                    success_rate = np.mean(recent_history)
                    
                    # Unlock next task if threshold met
                    if i < len(self.curriculum_tasks) - 1 and not self.curriculum_unlocked[i + 1]:
                        if success_rate >= self.curriculum_thresholds[i]:
                            next_task = self.curriculum_tasks[i + 1]
                            self.curriculum_unlocked[i + 1] = True
                            self.active_tasks.append(next_task)
                            return True, next_task, success_rate
        return False, None, None
    
    def get_curriculum_status(self):
        """Return current curriculum state for logging."""
        status = {}
        for i, name in enumerate(self.curriculum_tasks):
            unlocked = self.curriculum_unlocked[i]
            history = self.task_success_history[name][-50:]
            success_rate = np.mean(history) if len(history) > 0 else 0.0
            status[name] = {
                'unlocked': unlocked,
                'success_rate': success_rate,
                'episodes': len(self.task_success_history[name])
            }
        return status

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
#   MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="mt3_curriculum_run")
    parser.add_argument("--total_steps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum_thresholds", type=float, nargs=3, 
                       default=[0.6, 0.5, 0.0],
                       help="Success thresholds for [reach, push, pick-place]")
    args = parser.parse_args()

    RUN = args.run_name
    TOTAL_STEPS = args.total_steps
    SEED = args.seed
    MAX_STEPS = 150
    CURRICULUM_THRESHOLDS = args.curriculum_thresholds

    # --------------------- SAC Config (Based on MT10 config) ---------------------
    num_parallel_envs = int(os.environ.get("NUM_PARALLEL_ENVS", "1"))
    
    sac_config = {
        "policy": "MlpPolicy",
        "env": None,
        "learning_rate": 3e-4,
        "buffer_size": 600_000,            # 200k × 3 tasks
        "learning_starts": 5_000,          # Lower for MT3 (vs 10k for MT10)
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
        "critic_hidden_sizes": [512, 512, 512],
        "parallel_envs": num_parallel_envs,
        "curriculum_thresholds": CURRICULUM_THRESHOLDS,
    }

    wandb.init(
        entity="Robot_learning_2025",
        project="Robot_learning_2025",
        name=RUN,
        config=sac_config
    )

    os.makedirs("./models_mt3", exist_ok=True)
    
    # Create run-specific model directory
    model_dir = f"./models_mt3/{RUN}"
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 70)
    print("Meta-World MT3 Curriculum Training (Custom SAC)")
    print("Curriculum: reach-v3 → push-v3 → pick-place-v3")
    print(f"Thresholds: reach={CURRICULUM_THRESHOLDS[0]:.0%}, push={CURRICULUM_THRESHOLDS[1]:.0%}")
    print(f"Run: {RUN}")
    print(f"Model directory: {model_dir}")
    print(f"Actor: {sac_config['actor_hidden_sizes']}, Critic: {sac_config['critic_hidden_sizes']}")
    print(f"Buffer: {sac_config['buffer_size'] // 3:,}k per task × 3 tasks")
    if num_parallel_envs > 1:
        print(f"⚡ GPU Optimization: {num_parallel_envs}× parallel environments")
    print("=" * 70)

    # --------------------- Env ---------------------
    base_env = MetaWorldMT3CurriculumEnv(
        seed=SEED, 
        max_episode_steps=MAX_STEPS,
        curriculum_thresholds=CURRICULUM_THRESHOLDS
    )
    
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
    print(f"📚 Curriculum starting with: {base_env.active_tasks}")
    
    # --------------------- Training Loop ---------------------
    print("\n🚀 Starting MT3 Curriculum Training ...\n")
    
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
                
                # Update curriculum (check for task unlocks)
                unlocked, new_task, unlock_success_rate = base_env.update_curriculum(task_name, success)
                if unlocked:
                    pbar.write(f"\n🎓 CURRICULUM UNLOCK: {new_task} added! ({task_name} reached {unlock_success_rate:.1%} success)\n")
                    wandb.log({
                        "curriculum/task_unlocked": len(base_env.active_tasks) - 1,
                        "curriculum/unlock_step": step,
                        f"curriculum/{task_name.replace('-', '_')}_unlock_success": unlock_success_rate,
                    })
                
                # Log overall metrics
                wandb.log({
                    "train/episode_reward": episode_reward,
                    "train/episode_length": episode_length,
                    "train/success": success,
                    "curriculum/num_active_tasks": len(base_env.active_tasks),
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
                
                # Update progress bar with curriculum info
                pbar.set_postfix({
                    'ep': episode_count,
                    'tasks': len(base_env.active_tasks),
                    'task': task_name.split('-')[0][:6],
                    'rew': f'{episode_reward:.1f}',
                    'succ': success
                })
                
                # Reset
                obs, info = base_env.reset()
                episode_reward = 0
                episode_length = 0
            
            pbar.update(1)
            
            # Save checkpoint (same as MT10 but with curriculum state)
            if step % 100_000 == 0 and step > 0:
                save_path = f"{model_dir}/checkpoint_{step}.pt"
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'q1': agent.q1.state_dict(),
                    'q2': agent.q2.state_dict(),
                    'step': step,
                    'curriculum_unlocked': base_env.curriculum_unlocked,
                    'active_tasks': base_env.active_tasks,
                }, save_path)
                
                # Log curriculum status
                curriculum_status = base_env.get_curriculum_status()
                pbar.write(f"\n💾 Checkpoint {step}: Active tasks = {base_env.active_tasks}")
                for task, status in curriculum_status.items():
                    if status['unlocked']:
                        pbar.write(f"  ✓ {task}: {status['success_rate']:.1%} success ({status['episodes']} eps)")
                pbar.write("")
    
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
        'curriculum_unlocked': base_env.curriculum_unlocked,
        'active_tasks': base_env.active_tasks,
    }, final_path)

    # Print final curriculum status
    print("\n" + "=" * 70)
    print("Final Curriculum Status:")
    curriculum_status = base_env.get_curriculum_status()
    for task, status in curriculum_status.items():
        unlock_status = "✓ UNLOCKED" if status['unlocked'] else "✗ LOCKED"
        print(f"  {unlock_status} {task}: {status['success_rate']:.1%} success ({status['episodes']} episodes)")
    print("=" * 70)

    wandb.finish()
    base_env.close()

    print("\n🎉 Training finished successfully!")


if __name__ == "__main__":
    main()
