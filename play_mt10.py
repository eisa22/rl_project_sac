"""
Meta-World MT10 Evaluation Script

Evaluates a trained MT10 SAC model on all 10 tasks.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import metaworld
from stable_baselines3 import SAC


# ============================================================
#   MT10 Env Wrapper (same as training)
# ============================================================

class MetaWorldMT10Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, seed=0, max_episode_steps=150, render_mode=None):
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
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        self._env = None
        self._current_task = None
        self._tid = None
        self._step = 0

    def set_task_by_name(self, task_name):
        """Set a specific task by name"""
        self._tid = self.task_id_map[task_name]
        # Find task with matching env_name
        for task in self.tasks:
            if task.env_name == task_name:
                self._current_task = task
                break
        self._env = self.task_envs[task_name]
        self._env.set_task(self._current_task)

    def _sample_task(self):
        # Windows-safe task sampling
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
            "task_id": int(self._tid),
            "success": info.get("success", False)
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._env is not None:
            return self._env.render()

    def close(self):
        pass


# ============================================================
#   Evaluation
# ============================================================

def evaluate_mt10_model(model_path, episodes_per_task=10, render=False):
    """Evaluate MT10 model on all 10 tasks"""
    
    print(f"\n{'='*70}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*70}\n")

    # Create environment
    env = MetaWorldMT10Env(seed=42, max_episode_steps=150, render_mode="human" if render else None)
    
    # Load model
    model = SAC.load(model_path, env=env)
    
    # Get all task names
    task_names = env.task_names
    
    print(f"Evaluating on {len(task_names)} tasks with {episodes_per_task} episodes each\n")
    
    results = {}
    
    for task_name in task_names:
        print(f"📝 Evaluating: {task_name}")
        
        # Set specific task
        env.set_task_by_name(task_name)
        
        successes = []
        total_rewards = []
        
        for ep in range(episodes_per_task):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            while not done and step < 150:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1
                
                if render:
                    env.render()
            
            success = info.get("success", False)
            successes.append(success)
            total_rewards.append(episode_reward)
            
            print(f"  Episode {ep+1}/{episodes_per_task}: Reward={episode_reward:.2f}, Success={success}")
        
        # Calculate statistics
        success_rate = np.mean(successes) * 100
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        results[task_name] = {
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "std_reward": std_reward
        }
        
        print(f"  ✅ Success Rate: {success_rate:.1f}%")
        print(f"  📊 Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
    
    # Overall statistics
    print(f"\n{'='*70}")
    print("OVERALL RESULTS:")
    print(f"{'='*70}\n")
    
    overall_success = np.mean([r["success_rate"] for r in results.values()])
    overall_reward = np.mean([r["mean_reward"] for r in results.values()])
    
    print(f"🎯 Average Success Rate across all tasks: {overall_success:.1f}%")
    print(f"📊 Average Reward across all tasks: {overall_reward:.2f}\n")
    
    # Print per-task summary
    print("Per-Task Summary:")
    for task_name, stats in results.items():
        print(f"  {task_name:20s} → {stats['success_rate']:5.1f}% success")
    
    print(f"\n{'='*70}\n")
    
    env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models_mt10/sac_mt10_final.zip",
                        help="Path to the trained MT10 model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per task")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        print("\nAvailable models:")
        model_dir = "./models_mt10"
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith(".zip"):
                    print(f"  - {os.path.join(model_dir, f)}")
        else:
            print(f"  Model directory {model_dir} does not exist")
    else:
        evaluate_mt10_model(args.model_path, episodes_per_task=args.episodes, render=args.render)
