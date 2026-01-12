"""
Curriculum Learning for MT3 MTMH-SAC

Phase 1 (0-500k):   reach-v3 only
Phase 2 (500k-2M):  reach-v3 + push-v3
Phase 3 (2M-6M):    reach-v3 + push-v3 + pick-place-v3
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import wandb

from mtmh_sac import MTMHSACAgent


def get_active_tasks(step, total_steps):
    """Return active task indices based on curriculum phase."""
    phase_1_end = total_steps * 0.1   # 600k for 6M
    phase_2_end = total_steps * 0.4   # 2.4M for 6M
    
    if step < phase_1_end:
        return [0]  # reach only
    elif step < phase_2_end:
        return [0, 1]  # reach + push
    else:
        return [0, 1, 2]  # all three


def main():
    parser = argparse.ArgumentParser(description="Curriculum MTMH-SAC for MT3")
    parser.add_argument("--run_name", type=str, default="mtmh_sac_curriculum_v2")
    parser.add_argument("--total_steps", type=int, default=6_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    
    # Architecture
    parser.add_argument("--trunk_hidden_actor", type=str, default="512,512")
    parser.add_argument("--head_hidden_actor", type=str, default="256")
    parser.add_argument("--trunk_hidden_critic", type=str, default="1024,1024")
    parser.add_argument("--head_hidden_critic", type=str, default="512,512")
    
    args = parser.parse_args()
    
    # Parse architecture
    trunk_hidden_actor = tuple(map(int, args.trunk_hidden_actor.split(',')))
    head_hidden_actor = tuple(map(int, args.head_hidden_actor.split(',')))
    trunk_hidden_critic = tuple(map(int, args.trunk_hidden_critic.split(',')))
    head_hidden_critic = tuple(map(int, args.head_hidden_critic.split(',')))
    
    sac_config = {
        "learning_rate": args.lr,
        "alpha_lr": args.alpha_lr,
        "buffer_size": 1_000_000,
        "learning_starts": 0,
        "batch_size": args.batch_size,
        "tau": args.tau,
        "gamma": 0.99,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": args.seed,
        "total_steps": args.total_steps,
        "reward_scale": args.reward_scale,
        "trunk_hidden_actor": trunk_hidden_actor,
        "head_hidden_actor": head_hidden_actor,
        "trunk_hidden_critic": trunk_hidden_critic,
        "head_hidden_critic": head_hidden_critic,
    }
    
    wandb.init(project="Robot_learning_2025", name=args.run_name, config=sac_config)
    
    os.makedirs("./models_mtmh", exist_ok=True)
    model_dir = f"./models_mtmh/{args.run_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 70)
    print("MTMH-SAC Curriculum Learning: MT3")
    print(f"Run: {args.run_name}")
    print(f"Phase 1 (0-10%):   reach-v3")
    print(f"Phase 2 (10-40%):  reach-v3 + push-v3")
    print(f"Phase 3 (40-100%): reach-v3 + push-v3 + pick-place-v3")
    print("=" * 70)
    
    import metaworld
    TASK_NAMES = ['reach-v3', 'push-v3', 'pick-place-v3']
    num_tasks = len(TASK_NAMES)
    
    # Create environments
    envs = []
    for task_name in TASK_NAMES:
        ml1 = metaworld.ML1(task_name, seed=args.seed)
        env = ml1.train_classes[task_name]()
        env.set_task(ml1.train_tasks[0])
        envs.append(env)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]
    act_limit = float(envs[0].action_space.high[0])
    
    print(f"\nEnvironment: obs_dim={obs_dim}, act_dim={act_dim}, num_tasks={num_tasks}\n")
    
    # Agent (always uses 3 tasks, but curriculum determines which are active)
    agent = MTMHSACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        num_tasks=num_tasks,
        gamma=sac_config["gamma"],
        tau=sac_config["tau"],
        lr=args.lr,
        alpha_lr=args.alpha_lr,
        trunk_hidden_actor=trunk_hidden_actor,
        head_hidden_actor=head_hidden_actor,
        trunk_hidden_critic=trunk_hidden_critic,
        head_hidden_critic=head_hidden_critic,
        target_entropy=None,
        buffer_size_per_task=sac_config["buffer_size"],
    )
    
    print("✓ MTMH-SAC Agent initialized\n🚀 Starting MT3 Curriculum Training...\n")
    
    # Training state
    task_obs = []
    task_episode_reward = []
    task_episode_length = []
    task_episode_count = []
    task_episode_rewards = []
    task_episode_successes = []
    task_episode_lengths = []
    
    for task_id in range(num_tasks):
        reset_out = envs[task_id].reset()
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out
        task_obs.append(obs)
        task_episode_reward.append(0.0)
        task_episode_length.append(0)
        task_episode_count.append(0)
        task_episode_rewards.append([])
        task_episode_successes.append([])
        task_episode_lengths.append([])
    
    step = 0
    with tqdm(total=args.total_steps, desc="Training", unit="step") as pbar:
        while step < args.total_steps:
            # Get active tasks for current phase
            active_tasks = get_active_tasks(step, args.total_steps)
            
            for task_id in active_tasks:
                if step >= args.total_steps:
                    break
                
                obs = task_obs[task_id]
                action = agent.act(obs, task_id=task_id, deterministic=False)
                
                next_obs, reward, terminated, truncated, info = envs[task_id].step(action)
                done = bool(terminated or truncated)
                
                scaled_reward = reward * args.reward_scale
                agent.add_experience(obs, action, scaled_reward, next_obs, done, task_id)
                
                task_obs[task_id] = next_obs
                task_episode_reward[task_id] += reward
                task_episode_length[task_id] += 1
                
                # Update networks
                losses = agent.update(batch_size=sac_config["batch_size"])
                
                # Handle episode end
                if done:
                    task_episode_count[task_id] += 1
                    task_episode_rewards[task_id].append(task_episode_reward[task_id])
                    task_episode_successes[task_id].append(info.get("success", False))
                    task_episode_lengths[task_id].append(task_episode_length[task_id])
                    
                    reset_out = envs[task_id].reset()
                    if isinstance(reset_out, tuple):
                        task_obs[task_id], _ = reset_out
                    else:
                        task_obs[task_id] = reset_out
                    task_episode_reward[task_id] = 0.0
                    task_episode_length[task_id] = 0
                
                step += 1
                pbar.update(1)
                
                # Logging
                if step > 0 and step % 10_000 == 0:
                    phase_name = "Phase 1 (Reach)" if step < args.total_steps * 0.1 else \
                                 "Phase 2 (Reach+Push)" if step < args.total_steps * 0.4 else \
                                 "Phase 3 (All)"
                    
                    for tid in range(num_tasks):
                        if task_episode_rewards[tid]:
                            mean_reward = float(np.mean(task_episode_rewards[tid][-100:]))
                            mean_success = float(np.mean(task_episode_successes[tid][-100:]))
                            
                            wandb.log({
                                f"train/{TASK_NAMES[tid]}/mean_reward_100": mean_reward,
                                f"train/{TASK_NAMES[tid]}/success_rate_100": mean_success,
                                f"train/{TASK_NAMES[tid]}/total_episodes": task_episode_count[tid],
                                f"train/{TASK_NAMES[tid]}/alpha": agent.get_alpha(tid),
                                "train/phase": phase_name,
                                "train/step": step,
                            }, step=step)
    
    # Save final model
    final_path = f"{model_dir}/final_model.pt"
    print(f"\n💾 Saving final model to: {final_path}")
    
    save_dict = {
        "actor_trunk": agent.actor.trunk.state_dict(),
        "critic_trunk": agent.critic.trunk.state_dict(),
        "critic_target_trunk": agent.critic_target.trunk.state_dict(),
        "log_alphas": agent.log_alphas.detach().cpu(),
        "config": sac_config,
    }
    
    for tid in range(num_tasks):
        save_dict[f"actor_head_{tid}"] = agent.actor.task_heads[tid].state_dict()
        save_dict[f"critic_q1_head_{tid}"] = agent.critic.q1_heads[tid].state_dict()
        save_dict[f"critic_q2_head_{tid}"] = agent.critic.q2_heads[tid].state_dict()
        save_dict[f"critic_target_q1_head_{tid}"] = agent.critic_target.q1_heads[tid].state_dict()
        save_dict[f"critic_target_q2_head_{tid}"] = agent.critic_target.q2_heads[tid].state_dict()
    
    torch.save(save_dict, final_path)
    
    wandb.finish()
    for env in envs:
        env.close()
    
    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
