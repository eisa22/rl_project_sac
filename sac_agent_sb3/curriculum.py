from typing import Dict, List, Tuple

import gymnasium as gym
import metaworld
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class MetaWorldCurriculumEnv(gym.Env):
    """Simple MT3 wrapper that supports custom task sequences."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        task_sequence: List[str],
        curriculum_thresholds: List[float],
        seed: int = 0,
        max_episode_steps: int = 150,
        render_mode: str | None = None,
        fixed_task_name: str | None = None,
    ):
        super().__init__()
        self.task_sequence = list(task_sequence)
        self.curriculum_thresholds = list(curriculum_thresholds)
        self.render_mode = render_mode
        self.fixed_task_name = fixed_task_name
        self.max_episode_steps = max_episode_steps

        self.mt10 = metaworld.MT10()
        train_classes = self.mt10.train_classes
        train_tasks = list(self.mt10.train_tasks)
        self.task_envs = {
            name: train_classes[name](render_mode=render_mode)
            for name in self.task_sequence
        }
        self.tasks = [t for t in train_tasks if t.env_name in self.task_sequence]
        self.num_tasks = len(self.task_sequence)
        self.task_id_map = {name: i for i, name in enumerate(self.task_sequence)}

        self.active_tasks = [self.task_sequence[0]]
        self.task_success_history = {name: [] for name in self.task_sequence}
        self.curriculum_unlocked = [name in self.active_tasks for name in self.task_sequence]

        ref_env = self.task_envs[self.task_sequence[0]]
        obs_low = ref_env.observation_space.low.astype(np.float32)
        obs_high = ref_env.observation_space.high.astype(np.float32)
        low = np.concatenate([obs_low, np.zeros(self.num_tasks, dtype=np.float32)])
        high = np.concatenate([obs_high, np.ones(self.num_tasks, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = ref_env.action_space

        self._rng = np.random.default_rng(seed)
        self._env = None
        self._current_task = None
        self._tid = None
        self._step = 0

    def _sample_task(self):
        if self.fixed_task_name and self.fixed_task_name in self.task_envs:
            env_name = self.fixed_task_name
            self._current_task = next(t for t in self.tasks if t.env_name == env_name)
        else:
            available = [t for t in self.tasks if t.env_name in self.active_tasks]
            idx = int(self._rng.integers(low=0, high=len(available)))
            self._current_task = available[idx]
            env_name = self._current_task.env_name
        self._tid = self.task_id_map[env_name]
        self._env = self.task_envs[env_name]
        self._env.set_task(self._current_task)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        onehot = np.zeros(self.num_tasks, dtype=np.float32)
        onehot[self._tid] = 1.0
        return np.concatenate([obs.astype(np.float32), onehot])

    def update_curriculum(self, task_name: str, success: bool) -> Tuple[bool, str, float]:
        self.task_success_history[task_name].append(success)
        for idx, name in enumerate(self.task_sequence):
            if not self.curriculum_unlocked[idx]:
                continue
            if idx >= len(self.task_sequence) - 1:
                continue
            history = self.task_success_history[name][-50:]
            if len(history) < 20:
                continue
            threshold = self.curriculum_thresholds[idx] if idx < len(self.curriculum_thresholds) else 0.0
            success_rate = float(np.mean(history))
            if success_rate >= threshold and not self.curriculum_unlocked[idx + 1]:
                next_task = self.task_sequence[idx + 1]
                self.curriculum_unlocked[idx + 1] = True
                if next_task not in self.active_tasks:
                    self.active_tasks.append(next_task)
                return True, next_task, success_rate
        return False, "", 0.0

    def update_active_tasks(self, tasks: List[str]):
        self.active_tasks = [t for t in self.task_sequence if t in tasks]
        self.curriculum_unlocked = [t in self.active_tasks for t in self.task_sequence]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._sample_task()
        self._step = 0
        obs, info = self._env.reset()
        obs = self._augment_obs(obs)
        info = {"task_name": self._current_task.env_name, "task_id": int(self._tid)}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1
        done = terminated or truncated
        if self._step >= self.max_episode_steps:
            truncated = True
            done = True
        obs = self._augment_obs(obs)
        success = bool(info.get("success", False)) if isinstance(info, dict) else False
        unlocked = False
        new_task = ""
        unlock_rate = 0.0
        if done:
            unlocked, new_task, unlock_rate = self.update_curriculum(
                self._current_task.env_name, success
            )
        info = {
            "task_name": self._current_task.env_name,
            "task_id": int(self._tid),
            "success": success,
        }
        if done and unlocked:
            info["curriculum_unlocked"] = new_task
            info["curriculum_unlock_rate"] = unlock_rate
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        for env in self.task_envs.values():
            env.close()


def make_curriculum_vec_env(
    num_envs: int,
    seed: int,
    task_sequence: List[str],
    curriculum_thresholds: List[float],
    max_episode_steps: int = 150,
    render_mode: str | None = None,
    fixed_task_name: str | None = None,
    force_subproc: bool = False,
):
    def make_env(rank: int):
        def _init():
            return MetaWorldCurriculumEnv(
                task_sequence=task_sequence,
                curriculum_thresholds=curriculum_thresholds,
                seed=seed + rank,
                max_episode_steps=max_episode_steps,
                render_mode=render_mode,
                fixed_task_name=fixed_task_name,
            )

        return _init

    if num_envs > 1 or force_subproc:
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method="spawn")
    else:
        env = DummyVecEnv([make_env(0)])
    task_id_map = {name: idx for idx, name in enumerate(task_sequence)}
    return env, task_id_map


class CurriculumTracker:
    def __init__(self, tasks: List[str], thresholds: List[float]):
        self.tasks = tasks
        self.thresholds = thresholds
        self.active_tasks = [tasks[0]]
        self.unlocked = [True] + [False] * (len(tasks) - 1)
        self.episode_history = {task: [] for task in tasks}
        self.min_episodes = 20
        self.window_size = 50

    def update(self, task_name: str, success: bool):
        self.episode_history[task_name].append(success)
        idx = self.tasks.index(task_name)
        if idx < len(self.tasks) - 1 and not self.unlocked[idx + 1]:
            history = self.episode_history[task_name]
            if len(history) >= self.min_episodes:
                recent = history[-self.window_size :]
                success_rate = float(np.mean(recent))
                threshold = self.thresholds[idx] if idx < len(self.thresholds) else 0.0
                if success_rate >= threshold:
                    next_task = self.tasks[idx + 1]
                    self.unlocked[idx + 1] = True
                    self.active_tasks.append(next_task)
                    return True, next_task, success_rate
        return False, "", 0.0

    def get_status(self) -> Dict[str, dict]:
        status = {}
        for task in self.tasks:
            history = self.episode_history[task]
            success_rate = float(np.mean(history[-self.window_size :])) if history else 0.0
            status[task] = {
                "unlocked": task in self.active_tasks,
                "success_rate": success_rate,
                "episodes": len(history),
            }
        return status
