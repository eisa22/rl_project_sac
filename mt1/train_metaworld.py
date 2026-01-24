import os
import warnings
import gymnasium as gym
import metaworld
import numpy as np
import torch
import wandb  # Import WandB
import argparse

from stable_baselines3 import TD3, DDPG, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback  # Import WandB SB3 Integration


# 1. Standard Env Creator (Identical to Script A)
def make_env(task_name, rank, seed, max_episode_steps, normalize_reward=False):
    """
    Erstellt und wrappt die Meta-World Umgebung.
    """
    def _init():

        env = gym.make(
            'Meta-World/MT1',
            env_name=task_name,
            seed=seed + rank,
            reward_function_version='v3',
            max_episode_steps=max_episode_steps,
            terminate_on_success=False,
        )

        # Monitor Wrapper ist kritisch für korrekte 'rollout/ep_rew_mean' Kurven in WandB
        env = Monitor(env)
        return env

    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="pick-place-v3")  # Default to the harder task
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)  # Match Script A's steps
    args = parser.parse_args()

    config = {
        "task_name": args.task_name,
        "algorithm": "SAC",
        "total_timesteps": args.total_timesteps,
        "max_episode_steps": 500,
        "seed": 42,
        "n_envs": 12,
        "use_parallel": True,
        "normalize_reward": False,
        "eval_freq": 10000,
        "n_eval_episodes": 20,
        "policy_type": "MlpPolicy",
        # Algorithm Hyperparams
        "learning_rate": 3e-4,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "buffer_size": 1_000_000,
        "learning_starts": 5000,
        "net_arch": [256, 256, 256],
        "ent_coef": "auto",
        "train_freq": 1
    }

    # 2. WandB Init (MonitorGym=False, Sync=True)
    run = wandb.init(
        project="Robot_learning_2025",
        entity=None,
        config=config,
        sync_tensorboard=True,  # This effectively replaces WandbCallback for logging
        monitor_gym=False,  # Disabled to prevent video crashes
        save_code=True,
        name=f"{config['algorithm']}_{args.task_name}_sb3",
    )

    # Directories
    os.makedirs(f"./metaworld_models/checkpoints_{args.task_name}", exist_ok=True)
    os.makedirs(f"./metaworld_models/best_{args.task_name}", exist_ok=True)

    # Environments
    if config['use_parallel']:
        env = SubprocVecEnv(
            [make_env(args.task_name, i, config['seed'], config['max_episode_steps'], config['normalize_reward'])
             for i in range(config['n_envs'])],
            start_method='spawn'
        )
    else:
        env = make_env(args.task_name, 0, config['seed'], config['max_episode_steps'], config['normalize_reward'])()

    # Eval Env
    eval_env = make_env(args.task_name, 0, config['seed'] + 1000, config['max_episode_steps'], normalize_reward=False)()

    # Agent
    model = SAC(
        "MlpPolicy",
        env=env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        ent_coef=config['ent_coef'],
        target_entropy='auto',
        policy_kwargs=dict(
            net_arch=config['net_arch'],
            activation_fn=torch.nn.ReLU,
            log_std_init=-3,
        ),
        train_freq=config['train_freq'],
        gradient_steps=-1,
        tensorboard_log=f"runs/{run.id}",  # WandB syncs from here
        verbose=1,
        seed=config['seed'],
        device="auto",
        use_sde=False,
    )

    # 3. Callbacks: Use Standard SB3 Callbacks (Safest)
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=f"./metaworld_models/checkpoints_{args.task_name}/",
        name_prefix=f"sac_{args.task_name}",
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{args.task_name}/",
        log_path=f"./metaworld_logs/eval_{args.task_name}/",
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=True,
        render=False
    )

    # ==================== TRAINING ====================
    print(f"Starte Training für {config['total_timesteps']} Steps...")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training unterbrochen...")
    finally:
        # Sauber beenden
        model.save(f"./metaworld_models/{args.task_name}_{config['task_name']}_final")
        env.close()
        eval_env.close()
        run.finish()

    print("Training abgeschlossen & Run synchronisiert.")