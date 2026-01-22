"""
MTMH-SAC Training Script for Meta-World MT3 Benchmark.

This script implements training following Meta-World benchmark protocols:
- Multi-Task Multi-Head SAC (MTMH-SAC) architecture
- Optional curriculum learning
- Comprehensive logging (per-task success rates, Q-values, alpha)
- Proper evaluation during and after training

Usage:
    # Standard multi-task training (recommended)
    python train_mt3_curriculum_sb3.py --run_name mt3_run1 --enable_multihead
    
    # With curriculum learning
    python train_mt3_curriculum_sb3.py --run_name mt3_curriculum --enable_multihead --enable_curriculum
    
    # Custom hyperparameters
    python train_mt3_curriculum_sb3.py --run_name mt3_custom --batch_size 1280 --learning_rate 3e-4
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
import traceback
from stable_baselines3.common.vec_env import VecNormalize

from sac_agent_sb3.agent import SACAgentSB3, SACAgentSB3Config
from sac_agent_sb3.curriculum import CurriculumTracker, make_curriculum_vec_env
from sac_agent_sb3.multiheadAgent import MHSACAgentSB3, MHSACAgentSB3Config, MTMHSACLoggingCallback
from sac_agent_sb3.evaluation import OnlineEvaluationTracker


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning and comprehensive episode logging.
    
    Features:
    - Tracks per-task success rates for curriculum progression
    - Logs per-task metrics (success, return, length)
    - Supports both curriculum and uniform sampling modes
    """
    
    def __init__(
        self,
        curriculum: CurriculumTracker,
        task_names: List[str],
        log_interval: int = 1000,
        enable_curriculum: bool = True,
    ):
        super().__init__(verbose=0)
        self.curriculum = curriculum
        self.task_names = task_names
        self.log_interval = log_interval
        self.enable_curriculum = enable_curriculum
        self.last_log_step = 0
        self.episode_rewards = None
        self.episode_lengths = None
        self.episode_success = None
        self.total_env_steps = 0
        self.episode_count = 0
        self.task_metrics = defaultdict(lambda: defaultdict(list))
        
        # Online evaluation tracker
        self.eval_tracker = OnlineEvaluationTracker(task_names, window_size=100)

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
                if unlocked and self.enable_curriculum:
                    self.training_env.env_method("update_active_tasks", self.curriculum.active_tasks)
                    print(f"[Curriculum] Unlocked '{new_task}' at step {self.num_timesteps} (success rate: {success_rate:.2%})")

                # Track for evaluation metrics
                self.eval_tracker.add_episode(task_name, episode_reward, episode_success, episode_length)
                
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
            
            # Per-task metrics from recent episodes
            for task, metrics in self.task_metrics.items():
                task_key = task.replace('-', '_')
                if metrics["reward"]:
                    mean_reward = float(np.mean(metrics["reward"]))
                    self.logger.record(f"train/task/{task_key}/reward_mean", mean_reward)
                    wandb_payload[f"train/task/{task_key}/reward_mean"] = mean_reward
                if metrics["success"]:
                    mean_success = float(np.mean(metrics["success"]))
                    self.logger.record(f"train/task/{task_key}/success_rate", mean_success)
                    wandb_payload[f"train/task/{task_key}/success_rate"] = mean_success
                if metrics["length"]:
                    mean_length = float(np.mean(metrics["length"]))
                    self.logger.record(f"train/task/{task_key}/length", mean_length)
                    wandb_payload[f"train/task/{task_key}/length"] = mean_length
            
            # Aggregate metrics from evaluation tracker
            eval_metrics = self.eval_tracker.get_metrics()
            for key, value in eval_metrics.items():
                self.logger.record(key, value)
                wandb_payload[key] = value
            
            if wandb.run and wandb_payload:
                wandb.log(wandb_payload, step=self.num_timesteps)
            
            # Clear recent metrics
            self.task_metrics = defaultdict(lambda: defaultdict(list))
        return True


class AlphaQLoggingCallback(BaseCallback):
    """
    Logs entropy coefficient (alpha) and Q-value statistics for diagnostics.
    
    Monitoring these helps detect:
    - Alpha collapse (exploration issues)
    - Q-value explosion (training instability)
    - Learning progress
    """

    def __init__(self, log_interval: int = 1000, sample_batch_size: int = 256):
        super().__init__(verbose=0)
        self.log_interval = log_interval
        self.sample_batch_size = sample_batch_size
        self.last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step < self.log_interval:
            return True
        self.last_log_step = self.num_timesteps

        model = self.model
        wandb_payload = {}
        
        # Log alpha (entropy coefficient)
        if hasattr(model, "log_ent_coef"):
            alpha = float(torch.exp(model.log_ent_coef.detach()).item())
            self.logger.record("train/alpha", alpha)
            wandb_payload["train/alpha"] = alpha
            
            # Log target entropy for reference
            if hasattr(model, "target_entropy"):
                self.logger.record("train/target_entropy", float(model.target_entropy))
                wandb_payload["train/target_entropy"] = float(model.target_entropy)

        # Log Q-value statistics from a sampled batch
        rb = getattr(model, "replay_buffer", None)
        if rb is not None and rb.size() > self.sample_batch_size:
            try:
                batch = rb.sample(self.sample_batch_size, env=model._vec_normalize_env)
                obs = torch.as_tensor(batch.observations, device=model.device)
                actions = torch.as_tensor(batch.actions, device=model.device)
                
                with torch.no_grad():
                    q1, q2 = model.critic(obs, actions)
                    
                    q1_mean = float(q1.mean().item())
                    q2_mean = float(q2.mean().item())
                    q1_std = float(q1.std().item())
                    q2_std = float(q2.std().item())
                    q_min = float(min(q1.min().item(), q2.min().item()))
                    q_max = float(max(q1.max().item(), q2.max().item()))
                    
                self.logger.record("train/q1_mean", q1_mean)
                self.logger.record("train/q2_mean", q2_mean)
                self.logger.record("train/q1_std", q1_std)
                self.logger.record("train/q2_std", q2_std)
                self.logger.record("train/q_min", q_min)
                self.logger.record("train/q_max", q_max)
                
                wandb_payload.update({
                    "train/q1_mean": q1_mean,
                    "train/q2_mean": q2_mean,
                    "train/q1_std": q1_std,
                    "train/q2_std": q2_std,
                    "train/q_min": q_min,
                    "train/q_max": q_max,
                })
            except Exception as e:
                # Skip Q-stats if sampling fails
                pass
                
        if wandb.run and wandb_payload:
            wandb.log(wandb_payload, step=self.num_timesteps)
            
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="MTMH-SAC MT3 Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Run configuration
    parser.add_argument("--run_name", type=str, default="mt3_sb3",
                        help="Name for this training run")
    parser.add_argument("--total_steps", type=int, default=2_000_000,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_envs", type=int, default=8,
                        help="Number of parallel environments")
    
    # SAC Hyperparameters (Meta-World+ recommended values)
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate for actor and critic")
    parser.add_argument("--buffer_size", type=int, default=1_000_000,
                        help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=1280,
                        help="Batch size (scaled for multi-task: 128 * num_tasks)")
    parser.add_argument("--learning_starts", type=int, default=4_000,
                        help="Warmstart steps before training")
    parser.add_argument("--train_freq", type=int, default=1,
                        help="Training frequency (updates per env step)")
    parser.add_argument("--gradient_steps", type=int, default=1,
                        help="Gradient steps per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Target network soft update coefficient")
    parser.add_argument("--ent_coef", type=str, default="auto",
                        help="Entropy coefficient; 'auto' for automatic tuning, or float value")
    parser.add_argument("--target_entropy", type=str, default="auto",
                        help="Target entropy; 'auto' for -dim(A), or float value")
    parser.add_argument("--initial_temperature", type=float, default=1.0,
                        help="Initial temperature for entropy coefficient")
    
    # Network architecture (asymmetric: smaller actor, larger critic)
    # Meta-World+ finding: Critic capacity is the bottleneck in multi-task RL
    parser.add_argument("--actor_hidden", type=int, nargs="+", default=[256, 256],
                        help="Actor trunk hidden layer sizes")
    parser.add_argument("--critic_hidden", type=int, nargs="+", default=[512, 512, 512],
                        help="Critic trunk hidden layer sizes (larger for multi-task)")
    parser.add_argument("--actor_head_hidden", type=int, nargs="+", default=[128],
                        help="Actor head hidden layer sizes")
    parser.add_argument("--critic_head_hidden", type=int, nargs="+", default=[256],
                        help="Critic head hidden layer sizes")
    parser.add_argument("--use_layer_norm", action="store_true",
                        help="Use layer normalization in networks")
    parser.add_argument("--clip_q_values", action="store_true", default=True,
                        help="Clip Q-values for stability")
    
    # Environment settings
    parser.add_argument("--max_episode_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--curriculum_thresholds", type=float, nargs=3,
                        default=[0.6, 0.5, 0.4],
                        help="Success thresholds to unlock tasks (curriculum mode)")
    
    # Logging and checkpointing
    parser.add_argument("--checkpoint_interval", type=int, default=100_000,
                        help="Steps between checkpoints")
    parser.add_argument("--eval_interval", type=int, default=50_000,
                        help="Steps between evaluations")
    parser.add_argument("--log_interval", type=int, default=1_000,
                        help="Steps between logging")
    parser.add_argument("--n_eval_episodes", type=int, default=10,
                        help="Episodes per task for evaluation")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory to save models")
    
    # W&B configuration
    parser.add_argument("--wandb_project", type=str, default="Robot_learning_2025",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default="Robot_learning_2025",
                        help="Weights & Biases entity")
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"],
                        default="online", help="W&B run mode")
    
    # Training modes
    parser.add_argument("--enable_curriculum", action="store_true",
                        help="Enable Curriculum Learning (tasks unlocked progressively)")
    parser.add_argument("--enable_multihead", action="store_true",
                        help="Enable MultiHead Architecture (recommended)")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., ./models_sac_sb3/run_name/checkpoints/sac_mt3_500000_steps.zip)")
    parser.add_argument("--resume_steps", type=int, default=None,
                        help="Number of steps already completed (auto-detected from checkpoint name if not specified)")
    
    return parser.parse_args()


def main():
    args = parse_args()

    def maybe_float(val: str):
        """Convert string to float if possible, otherwise return as-is."""
        try:
            return float(val)
        except ValueError:
            return val

    # Task configuration
    task_sequence = ["reach-v3", "push-v3", "pick-place-v3"]
    num_tasks = len(task_sequence)
    
    # Setup directories
    os.makedirs(args.model_dir, exist_ok=True)
    model_dir = os.path.join(args.model_dir, args.run_name)
    os.makedirs(model_dir, exist_ok=True)

    # Create training environment
    env, _ = make_curriculum_vec_env(
        num_envs=args.num_envs,
        seed=args.seed,
        task_sequence=task_sequence,
        curriculum_thresholds=args.curriculum_thresholds,
        max_episode_steps=args.max_episode_steps,
    )
    # Note: norm_reward=True can help stability but makes returns harder to interpret
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Create agent based on mode
    if args.enable_multihead:
        print(f"[Config] Using Multi-Head Architecture (MTMH-SAC)")
        config = MHSACAgentSB3Config(
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            target_entropy=maybe_float(args.target_entropy),
            initial_temperature=args.initial_temperature,
            actor_hidden=args.actor_hidden,
            critic_hidden=args.critic_hidden,
            actor_head_hidden=args.actor_head_hidden,
            critic_head_hidden=args.critic_head_hidden,
            use_layer_norm=args.use_layer_norm,
            clip_q_values=args.clip_q_values,
            num_tasks=num_tasks,
            seed=args.seed,
        )
        if args.resume:
            print(f"[Resume] Loading checkpoint from: {args.resume}")
            agent = MHSACAgentSB3.load(args.resume, env=env)
            # Try to load replay buffer if it exists
            replay_buffer_path = args.resume.replace('.zip', '_replay_buffer.pkl')
            if os.path.exists(replay_buffer_path):
                print(f"[Resume] Loading replay buffer from: {replay_buffer_path}")
                agent.model.load_replay_buffer(replay_buffer_path)
            else:
                print(f"[Resume] Warning: No replay buffer found at {replay_buffer_path}")
                print(f"[Resume] Training will continue but buffer starts empty (first {args.learning_starts} steps will be warmup)")
        else:
            agent = MHSACAgentSB3(env=env, config=config)
    else:
        print(f"[Config] Using Standard SAC (shared network)")
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
        if args.resume:
            print(f"[Resume] Loading checkpoint from: {args.resume}")
            agent = SACAgentSB3.load(args.resume, env=env)
        else:
            agent = SACAgentSB3(env=env, config=config)
    
    # Calculate remaining steps if resuming
    completed_steps = 0
    if args.resume:
        if args.resume_steps is not None:
            completed_steps = args.resume_steps
        else:
            # Try to extract steps from checkpoint filename (e.g., sac_mt3_500000_steps.zip)
            import re
            match = re.search(r'_(\d+)_steps', args.resume)
            if match:
                completed_steps = int(match.group(1))
                print(f"[Resume] Auto-detected {completed_steps:,} completed steps from filename")
            else:
                print(f"[Resume] Warning: Could not detect steps from filename. Use --resume_steps to specify.")
        
        remaining_steps = max(0, args.total_steps - completed_steps)
        print(f"[Resume] Training for {remaining_steps:,} more steps (total target: {args.total_steps:,})")
    else:
        remaining_steps = args.total_steps

    # Build W&B configuration
    wandb_config = {
        # Training settings
        "total_steps": args.total_steps,
        "num_envs": args.num_envs,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
        "task_sequence": task_sequence,
        # SAC hyperparameters
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "tau": args.tau,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "ent_coef": args.ent_coef,
        "target_entropy": args.target_entropy,
        # Architecture
        "enable_multihead": args.enable_multihead,
        "actor_hidden": args.actor_hidden if args.enable_multihead else "default",
        "critic_hidden": args.critic_hidden if args.enable_multihead else "default",
        "actor_head_hidden": args.actor_head_hidden if args.enable_multihead else "N/A",
        "critic_head_hidden": args.critic_head_hidden if args.enable_multihead else "N/A",
        "use_layer_norm": args.use_layer_norm,
        "clip_q_values": args.clip_q_values,
        # Curriculum
        "enable_curriculum": args.enable_curriculum,
        "curriculum_thresholds": args.curriculum_thresholds,
    }

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=wandb_config,
        mode=args.wandb_mode,
    )
    
    print(f"\n{'=' * 60}")
    print(f"MTMH-SAC Training Configuration")
    print(f"{'=' * 60}")
    print(f"Run Name: {args.run_name}")
    print(f"Tasks: {task_sequence}")
    print(f"Total Steps: {args.total_steps:,}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Architecture: {'Multi-Head' if args.enable_multihead else 'Shared'}")
    print(f"Curriculum: {'Enabled' if args.enable_curriculum else 'Disabled (Uniform)'}")
    print(f"{'=' * 60}\n")


    class SafeCheckpointCallback(CheckpointCallback):
        def _on_step(self) -> bool:
            try:
                result = super()._on_step()
                # Also save replay buffer with each checkpoint
                if self.n_calls % self.save_freq == 0:
                    checkpoint_path = os.path.join(
                        self.save_path, 
                        f"{self.name_prefix}_{self.num_timesteps}_steps_replay_buffer.pkl"
                    )
                    try:
                        self.model.save_replay_buffer(checkpoint_path)
                        print(f"[Checkpoint] Saved replay buffer: {checkpoint_path}")
                    except Exception as e:
                        print(f"[Checkpoint] Warning: Could not save replay buffer: {e}")
                return result
            except Exception as e:
                print(f"[CheckpointCallback ERROR] Could not save checkpoint at step {self.num_timesteps}: {e}")
                traceback.print_exc()
                return True

    checkpoint_callback = SafeCheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="sac_mt3",
        verbose=1,
    )

    # Per-task evaluation environments
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
        vn_path = os.path.join(model_dir, "vecnormalize.pkl")
        if os.path.exists(vn_path):
            eval_env = VecNormalize.load(vn_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False   # raw returns in eval (recommended)
            eval_env.norm_obs = False      # you trained with norm_obs=False anyway
        else:
            # fallback: no saved stats -> just ensure eval is non-training normalize wrapper
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
        
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, f"best_{task}"),
            log_path=os.path.join(model_dir, f"eval_{task}"),
            eval_freq=args.eval_interval,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            verbose=0,
        )
        eval_callbacks.append(eval_cb)

    # Curriculum tracker and callbacks
    curriculum = CurriculumTracker(tasks=task_sequence, thresholds=args.curriculum_thresholds)
    alpha_q_callback = AlphaQLoggingCallback(log_interval=args.log_interval)
    curriculum_callback = CurriculumCallback(
        curriculum=curriculum, 
        task_names=task_sequence,
        log_interval=args.log_interval,
        enable_curriculum=args.enable_curriculum,
    )
    
    # Setup curriculum or uniform sampling
    if args.enable_curriculum:
        print(f"[Curriculum] Starting with task: {task_sequence[0]}")
    else:
        # Unlock all tasks immediately for uniform sampling
        env.env_method("update_active_tasks", task_sequence)
        curriculum.active_tasks = list(task_sequence)
        print(f"[Sampling] Uniform sampling over all tasks")

    # Combine all callbacks
    callback_list = [
        checkpoint_callback,
        curriculum_callback,
        alpha_q_callback,
        *eval_callbacks
    ]
    callback = CallbackList(callback_list)

    # Train
    print(f"\nStarting training...")
    agent.learn(
        total_timesteps=remaining_steps,
        callback=callback,
        log_interval=args.log_interval,
        progress_bar=True,
        reset_num_timesteps=not bool(args.resume),  # Don't reset step counter when resuming
    )

    # Save final model
    final_path = os.path.join(model_dir, "final_sac_mt3")
    agent.save(final_path)
    print(f"\nSaved final model to: {final_path}")
    
    # Save replay buffer for future resumption
    replay_buffer_path = final_path + "_replay_buffer.pkl"
    agent.model.save_replay_buffer(replay_buffer_path)
    print(f"Saved replay buffer to: {replay_buffer_path}")
    
    # Save VecNormalize statistics (important for evaluation)
    env.save(os.path.join(model_dir, "vecnormalize.pkl"))
    
    # Cleanup
    env.close()
    for cb in eval_callbacks:
        if hasattr(cb, 'eval_env') and cb.eval_env is not None:
            cb.eval_env.close()
    
    wandb.finish()
    print(f"\nTraining complete!")


if __name__ == "__main__":
    main()
