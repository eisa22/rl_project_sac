import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback

from sac_agent_sb3.agent import SACAgentSB3, SACAgentSB3Config
from sac_agent_sb3.curriculum import CurriculumTracker, make_curriculum_vec_env


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        curriculum: CurriculumTracker,
        log_interval: int = 1000,
    ):
        super().__init__(verbose=0)
        self.curriculum = curriculum
        self.log_interval = log_interval
        self.last_log_step = 0
        self.episode_rewards = None
        self.episode_lengths = None
        self.episode_success = None
        self.total_env_steps = 0
        self.episode_count = 0
        self.task_metrics = defaultdict(lambda: defaultdict(list))

    def _on_training_start(self):
        self.num_envs = self.training_env.num_envs
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_success = np.zeros(self.num_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        terminated = self.locals.get("terminated")
        truncated = self.locals.get("truncated")

        if infos is None or rewards is None:
            return True

        if dones is None and terminated is not None and truncated is not None:
            dones = np.logical_or(terminated, truncated)
        elif dones is None:
            dones = np.zeros(len(infos), dtype=bool)

        rewards = np.asarray(rewards)
        self.total_env_steps += len(infos)

        for idx, info in enumerate(infos):
            self.episode_rewards[idx] += rewards[idx]
            self.episode_lengths[idx] += 1
            success = int(bool(info.get("success", False)))
            self.episode_success[idx] = max(self.episode_success[idx], success)

            if dones[idx]:
                task_name = info.get("task_name", "unknown")
                episode_reward = float(self.episode_rewards[idx])
                episode_length = int(self.episode_lengths[idx])
                episode_success = int(self.episode_success[idx])

                unlocked, new_task, success_rate = self.curriculum.update(task_name, episode_success)
                if unlocked:
                    self.training_env.env_method("update_active_tasks", self.curriculum.active_tasks)
                    print(f"Curriculum unlocked {new_task} at step {self.num_timesteps}")

                self.task_metrics[task_name]["reward"].append(episode_reward)
                self.task_metrics[task_name]["success"].append(episode_success)
                self.task_metrics[task_name]["length"].append(episode_length)

                self.logger.record("train/episode_reward", episode_reward)
                self.logger.record("train/episode_length", episode_length)
                self.logger.record("train/success", episode_success)
                self.logger.record("curriculum/active_tasks", len(self.curriculum.active_tasks))
                self.logger.record("curriculum/last_task", task_name)
                if wandb.run:
                    wandb.log(
                        {
                            "train/episode_reward": episode_reward,
                            "train/episode_length": episode_length,
                            "train/success": episode_success,
                            "curriculum/active_tasks": len(self.curriculum.active_tasks),
                            "curriculum/last_task": task_name,
                        },
                        step=self.num_timesteps,
                    )

                self.episode_rewards[idx] = 0
                self.episode_lengths[idx] = 0
                self.episode_success[idx] = 0
                self.episode_count += 1

        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self.last_log_step = self.num_timesteps
            wandb_payload = {}
            for task, metrics in self.task_metrics.items():
                if metrics["reward"]:
                    self.logger.record(
                        f"train/task/{task.replace('-', '_')}/reward_mean",
                        float(np.mean(metrics["reward"])),
                    )
                    if wandb.run:
                        wandb_payload[f"train/task/{task.replace('-', '_')}/reward_mean"] = float(
                            np.mean(metrics["reward"])
                        )
                if metrics["success"]:
                    self.logger.record(
                        f"train/task/{task.replace('-', '_')}/success",
                        float(np.mean(metrics["success"])),
                    )
                    if wandb.run:
                        wandb_payload[f"train/task/{task.replace('-', '_')}/success"] = float(
                            np.mean(metrics["success"])
                        )
                if metrics["length"]:
                    self.logger.record(
                        f"train/task/{task.replace('-', '_')}/length",
                        float(np.mean(metrics["length"])),
                    )
                    if wandb.run:
                        wandb_payload[f"train/task/{task.replace('-', '_')}/length"] = float(
                            np.mean(metrics["length"])
                        )
            if wandb.run and wandb_payload:
                wandb.log(wandb_payload, step=self.num_timesteps)
            self.task_metrics = defaultdict(lambda: defaultdict(list))
        return True


class AlphaQLoggingCallback(BaseCallback):
    """Logs alpha (entropy coeff) and approximate Q means from replay samples."""

    def __init__(self, log_interval: int = 1000):
        super().__init__(verbose=0)
        self.log_interval = log_interval
        self.last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step < self.log_interval:
            return True
        self.last_log_step = self.num_timesteps

        model = self.model
        # Log alpha
        if hasattr(model, "log_ent_coef"):
            alpha = float(torch.exp(model.log_ent_coef.detach()).item())
            self.logger.record("train/alpha", alpha)
            if wandb.run:
                wandb.log({"train/alpha": alpha}, step=self.num_timesteps)

        # Log Q means from a sampled batch if buffer is ready
        rb = getattr(model, "replay_buffer", None)
        if rb is not None and rb.size() > model.batch_size:
            batch = rb.sample(model.batch_size, env=model._vec_normalize_env)
            obs = torch.as_tensor(batch.observations, device=model.device)
            actions = torch.as_tensor(batch.actions, device=model.device)
            with torch.no_grad():
                q1, q2 = model.critic(obs, actions)
                q1_mean = float(q1.mean().item())
                q2_mean = float(q2.mean().item())
            self.logger.record("train/q1_mean", q1_mean)
            self.logger.record("train/q2_mean", q2_mean)
            if wandb.run:
                wandb.log(
                    {
                        "train/q1_mean": q1_mean,
                        "train/q2_mean": q2_mean,
                    },
                    step=self.num_timesteps,
                )
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="SB3 SAC MT3 curriculum runner")
    parser.add_argument("--run_name", type=str, default="mt3_sb3")
    parser.add_argument("--total_steps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=3_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_starts", type=int, default=5_000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--ent_coef", type=str, default="auto", help="Entropy coefficient; e.g., auto or auto_0.2")
    parser.add_argument(
        "--target_entropy",
        type=str,
        default="auto",
        help="Target entropy (auto or numeric, e.g., -2). Less negative keeps alpha higher",
    )
    parser.add_argument("--checkpoint_interval", type=int, default=100_000)
    parser.add_argument("--log_interval", type=int, default=1_000)
    parser.add_argument("--max_episode_steps", type=int, default=150)
    parser.add_argument(
        "--curriculum_thresholds",
        type=float,
        nargs=3,
        default=[0.6, 0.5, 0.4],
        help="Success rates required to unlock tasks (reach, pick-place, push)",
    )
    parser.add_argument("--model_dir", type=str, default="./models_sac_sb3")
    parser.add_argument("--wandb_project", type=str, default="Robot_learning_2025")
    parser.add_argument("--wandb_entity", type=str, default="Robot_learning_2025")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="WandB run mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    def maybe_float(val: str):
        try:
            return float(val)
        except ValueError:
            return val

    task_sequence = ["reach-v3", "push-v3", "pick-place-v3"]
    os.makedirs(args.model_dir, exist_ok=True)
    model_dir = os.path.join(args.model_dir, args.run_name)
    os.makedirs(model_dir, exist_ok=True)

    env, _ = make_curriculum_vec_env(
        num_envs=args.num_envs,
        seed=args.seed,
        task_sequence=task_sequence,
        curriculum_thresholds=args.curriculum_thresholds,
        max_episode_steps=args.max_episode_steps,
    )

    config = SACAgentSB3Config(
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        target_entropy=maybe_float(args.target_entropy),
        seed=args.seed,
    )

    agent = SACAgentSB3(env=env, config=config)
    wandb_config = {
        "total_steps": args.total_steps,
        "num_envs": args.num_envs,
        "max_episode_steps": args.max_episode_steps,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "curriculum_thresholds": args.curriculum_thresholds,
        "seed": args.seed,
        "task_sequence": task_sequence,
    }

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=wandb_config,
        mode=args.wandb_mode,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="sac_mt3",
        verbose=1,
    )

    # Per-task evaluation environments (deterministic, single-env) to mirror MT1 script behavior
    eval_callbacks = []
    for task in task_sequence:
        eval_env, _ = make_curriculum_vec_env(
            num_envs=1,
            seed=args.seed + 10_000,
            task_sequence=task_sequence,
            curriculum_thresholds=args.curriculum_thresholds,
            max_episode_steps=args.max_episode_steps,
            fixed_task_name=task,
            force_subproc=True,
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, f"best_{task}"),
            log_path=os.path.join(model_dir, f"eval_{task}"),
            eval_freq=args.checkpoint_interval,
            n_eval_episodes=5,
            deterministic=True,
            verbose=0,
        )
        eval_callbacks.append(eval_cb)

    curriculum = CurriculumTracker(tasks=task_sequence, thresholds=args.curriculum_thresholds)
    curriculum_callback = CurriculumCallback(curriculum=curriculum, log_interval=args.log_interval)
    alpha_q_callback = AlphaQLoggingCallback(log_interval=args.log_interval)
    callback = CallbackList([checkpoint_callback, curriculum_callback, alpha_q_callback, *eval_callbacks])

    agent.learn(
        total_timesteps=args.total_steps,
        callback=callback,
        log_interval=args.log_interval,
        progress_bar=True,
    )

    agent.save(os.path.join(model_dir, "final_sac_mt3"))
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
