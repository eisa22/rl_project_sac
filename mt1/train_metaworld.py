import os
import warnings
import gymnasium as gym
import metaworld
import numpy as np
import torch
import wandb  # Import WandB

from stable_baselines3 import TD3, DDPG, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback  # Import WandB SB3 Integration


def make_env(task_name, rank, seed, max_episode_steps, normalize_reward=False):
    """
    Erstellt und wrappt die Meta-World Umgebung.
    """
    def _init():
        env = gym.make(
            'Meta-World/MT1',
            env_name=task_name,
            seed=seed + rank,
            reward_function_version='v2', # v2 ist oft stabiler definiert
            max_episode_steps=max_episode_steps,
            terminate_on_success=False,
        )

        # Monitor Wrapper ist kritisch für korrekte 'rollout/ep_rew_mean' Kurven in WandB
        env = Monitor(env)
        return env

    return _init


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # Wir speichern alles in einem Dictionary für WandB
    config = {
        "task_name": "reach-v3",
        "algorithm": "SAC",
        "total_timesteps": 500_000,
        "max_episode_steps": 500,
        "seed": 42,
        "n_envs": 8,
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
    }

    # Verzeichnisse erstellen
    os.makedirs("./metaworld_models", exist_ok=True)

    # ==================== WANDB INIT ====================
    run = wandb.init(
        project="Robot_learning_2025",
        entity=None, # Hier dein Username/Teamname optional eintragen
        config=config,
        sync_tensorboard=True,  # WICHTIG: Liest SB3 Metriken automatisch
        monitor_gym=True,       # Versucht Videos zu speichern (wenn Render möglich)
        save_code=True,         # Speichert dieses Skript in der Cloud
        name=f"{config['algorithm']}_{config['task_name']}_seed{config['seed']}",
    )

    print(f"=" * 60)
    print(f"Meta-World MT1 Training: {config['task_name']} mit WandB Logging")
    print(f"Run ID: {run.id}")
    print(f"=" * 60)

    # ==================== ENV SETUP ====================
    if config['use_parallel']:
        env = SubprocVecEnv(
            [make_env(config['task_name'], i, config['seed'], config['max_episode_steps'], config['normalize_reward'])
             for i in range(config['n_envs'])],
            start_method='spawn'
        )
    else:
        env = make_env(config['task_name'], 0, config['seed'], config['max_episode_steps'], config['normalize_reward'])()

    # Eval Env
    eval_env = make_env(config['task_name'], 0, config['seed'] + 1000, config['max_episode_steps'], normalize_reward=False)()

    # ==================== AGENT SETUP ====================

    # Wir nutzen SAC als Standard, wie im Config definiert
    model = SAC(
        policy=config['policy_type'],
        env=env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        ent_coef=config['ent_coef'],
        policy_kwargs=dict(
            net_arch=config['net_arch'],
            activation_fn=torch.nn.ReLU,
        ),
        # Wir leiten Tensorboard logs in den WandB Ordner um (Sync)
        tensorboard_log=f"runs/{run.id}",
        verbose=1,
        device="auto",
        seed=config['seed'],
    )

    # ==================== CALLBACKS ====================

    # 1. WandB Callback (lädt Modelldateien und System-Metriken hoch)
    wandb_callback = WandbCallback(
        gradient_save_freq=0,  # Setze auf 100, wenn du Gradienten-Histogramme willst (teuer!)
        model_save_path=f"models/{run.id}",
        model_save_freq=25000,
        verbose=2,
    )

    # 2. Eval Callback (Loggt Validierungs-Erfolg zu WandB)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{config['task_name']}/",
        log_path=f"./metaworld_logs/eval_{config['task_name']}/",
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=True,
        render=False
    )

    # ==================== TRAINING ====================
    print(f"Starte Training für {config['total_timesteps']} Steps...")

    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=[wandb_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training unterbrochen...")
    finally:
        # Sauber beenden
        model.save(f"./metaworld_models/{config['algorithm']}_{config['task_name']}_final")
        env.close()
        eval_env.close()
        run.finish()

    print("Training abgeschlossen & Run synchronisiert.")