"""
Single-Task SAC Trainer for reach-v3

Simple baseline to debug hyperparameters and SAC implementation.
Without task embeddings or curriculum.
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import wandb

from sac_core_clean import CleanSACAgent


def main():
    parser = argparse.ArgumentParser(description="Single-Task SAC (reach-v3)")
    parser.add_argument("--run_name", type=str, default="reach_singletask_sac")
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for actor/critic")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="Reward scaling factor")
    args = parser.parse_args()

    RUN = args.run_name
    TOTAL_STEPS = args.total_steps
    SEED = args.seed
    LR = args.lr
    REWARD_SCALE = args.reward_scale

    # Config
    sac_config = {
        # Garage/Meta-World single-task SAC defaults (Table 3)
        "learning_rate": 3e-4,  # policy_lr and qf_lr
        "buffer_size": 1_000_000,
        "learning_starts": 0,
        "batch_size": 500,
        "tau": 0.005,  # target_update_tau
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,  # paper reports 500 grad steps/epoch; here 1 per env step (comparable over horizon)
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": SEED,
        "total_steps": TOTAL_STEPS,
        "reward_scale": REWARD_SCALE,
    }

    # Allow overriding lr via CLI, else use paper value
    sac_config["learning_rate"] = LR if LR is not None else sac_config["learning_rate"]

    wandb.init(project="Robot_learning_2025", name=RUN, config=sac_config)

    os.makedirs("./models_singletask", exist_ok=True)
    model_dir = f"./models_singletask/{RUN}"
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 70)
    print("Single-Task SAC: reach-v3 (NO embeddings, NO curriculum)")
    print(f"Run: {RUN}")
    print(f"Learning rate: {LR}")
    print(f"Reward scaling: {REWARD_SCALE}")
    print("=" * 70)

    # Environment via Meta-World ML1 (single task)
    import metaworld
    ml1 = metaworld.ML1('reach-v3', seed=SEED)
    env = ml1.train_classes['reach-v3']()
    env.set_task(ml1.train_tasks[0])  # critical: set task before reset to init _last_rand_vec
    # Optionally wrap with gymnasium RecordEpisodeStatistics if available for logging
    try:
        from gymnasium.wrappers import RecordEpisodeStatistics
        env = RecordEpisodeStatistics(env)
    except Exception:
        pass
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    print(f"\nEnvironment:")
    print(f"  obs_dim = {obs_dim}")
    print(f"  act_dim = {act_dim}")
    print(f"  act_limit = {act_limit}\n")

    # Agent
    agent = CleanSACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        num_tasks=1,
        gamma=sac_config["gamma"],
        tau=sac_config["tau"],
        lr=sac_config["learning_rate"],
        alpha_lr=sac_config["learning_rate"] * 0.1,
        hidden_actor=(256, 256),
        hidden_critic=(256, 256),
        embedding_dim=8,
        target_entropy=None,
        buffer_size_per_task=sac_config["buffer_size"],
    )

    print("✓ Agent initialized\n")
    print("🚀 Starting reach-v3 training...\n")

    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    task_id = 0  # single task
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    episode_rewards = []
    episode_successes = []
    episode_lengths = []

    with tqdm(total=TOTAL_STEPS, desc="Training", unit="step") as pbar:
        for step in range(TOTAL_STEPS):
            if step < sac_config["learning_starts"]:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, task_id=task_id, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # Scale reward
            scaled_reward = reward * REWARD_SCALE
            agent.add_experience(obs, action, scaled_reward, next_obs, done, task_id)

            obs = next_obs
            episode_reward += reward
            episode_length += 1

            # Update
            if step >= sac_config["learning_starts"] and (step % sac_config["train_freq"] == 0):
                losses = agent.update(batch_size=sac_config["batch_size"])
                if step % 5000 == 0:
                    wandb.log({
                        "train/q1_loss": losses.get("q1_loss", 0.0),
                        "train/q2_loss": losses.get("q2_loss", 0.0),
                        "train/q_loss": losses.get("q_loss", 0.0),
                        "train/actor_loss": losses.get("actor_loss", 0.0),
                        "train/alpha": losses.get("alpha", 0.0),
                        "train/alpha_loss": losses.get("alpha_loss", 0.0),
                        "train/step": step,
                    }, step=step)

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                # Only count final success status (standard Meta-World metric)
                episode_successes.append(info.get("success", False))
                episode_lengths.append(episode_length)

                pbar.set_postfix({
                    "ep": episode_count,
                    "rew": f"{episode_reward:.1f}",
                    "succ": "✓" if info.get("success", False) else "✗",
                })

                obs, info = env.reset()
                episode_reward = 0.0
                episode_length = 0

            if step > 0 and step % 10000 == 0:
                # Log statistics
                if episode_rewards:
                    mean_reward = float(np.mean(episode_rewards[-100:]))
                    mean_success = float(np.mean(episode_successes[-100:]))
                    mean_length = float(np.mean(episode_lengths[-100:]))
                    
                    wandb.log({
                        "train/mean_reward_100": mean_reward,
                        "train/success_rate_100": mean_success,
                        "train/episode_length_100": mean_length,
                        "train/total_episodes": episode_count,
                        "train/step": step,
                    }, step=step)

            pbar.update(1)

    # Save final model
    final_path = f"{model_dir}/final_model.pt"
    print(f"\n💾 Saving final model to: {final_path}")
    torch.save({
        "actor": agent.actor.state_dict(),
        "q1": agent.critic.q1.state_dict(),
        "q2": agent.critic.q2.state_dict(),
        "q1_target": agent.critic_target.q1.state_dict(),
        "q2_target": agent.critic_target.q2.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "config": sac_config,
    }, final_path)

    wandb.finish()
    env.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    if episode_rewards:
        print(f"Final 100 episodes - Reward: {np.mean(episode_rewards[-100:]):.1f}")
        print(f"Final 100 episodes - Success Rate: {np.mean(episode_successes[-100:]):.2%}")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
