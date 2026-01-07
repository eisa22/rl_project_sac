import argparse
import os

import gymnasium as gym
import metaworld
import wandb
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sac_agent_sb3.agent import SACAgentSB3, SACAgentSB3Config


def make_mt1_env(task_name: str, rank: int, seed: int, max_episode_steps: int, normalize_reward: bool):
    def _init():
        if hasattr(metaworld, "register_all"):
            metaworld.register_all()
        env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            seed=seed + rank,
            reward_function_version="v3",
            max_episode_steps=max_episode_steps,
            terminate_on_success=False,
        )
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env)
        return Monitor(env)

    return _init


def make_vectorized_env(task_name: str, num_envs: int, seed: int, max_episode_steps: int, normalize_reward: bool):
    factories = [
        make_mt1_env(task_name, rank, seed, max_episode_steps, normalize_reward)
        for rank in range(num_envs)
    ]
    if num_envs > 1:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def parse_args():
    parser = argparse.ArgumentParser(description="SAC SB3 MT1 runner")
    parser.add_argument("--run_name", type=str, default="mt1_sb3")
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="reach-v3")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_episode_steps", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=3_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_starts", type=int, default=5_000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--checkpoint_interval", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=25_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--normalize_reward", action="store_true")
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

    os.makedirs(args.model_dir, exist_ok=True)

    env = make_vectorized_env(
        task_name=args.task,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        normalize_reward=args.normalize_reward,
    )

    eval_env = make_mt1_env(
        args.task,
        rank=0,
        seed=args.seed + 1_000,
        max_episode_steps=args.max_episode_steps,
        normalize_reward=False,
    )()

    model_dir = os.path.join(args.model_dir, args.run_name)
    os.makedirs(model_dir, exist_ok=True)

    config = SACAgentSB3Config(
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        seed=args.seed,
    )

    wandb_config = {
        "task": args.task,
        "total_steps": args.total_steps,
        "num_envs": args.num_envs,
        "max_episode_steps": args.max_episode_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "normalize_reward": args.normalize_reward,
        "seed": args.seed,
    }

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=wandb_config,
        mode=args.wandb_mode,
    )

    agent = SACAgentSB3(env=env, config=config)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix=f"sac_{args.task}",
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=os.path.join(model_dir, "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    agent.learn(
        total_timesteps=args.total_steps,
        callback=callback,
        log_interval=args.log_interval,
        progress_bar=True,
    )

    agent.save(os.path.join(model_dir, f"final_sac_{args.task}"))

    env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
