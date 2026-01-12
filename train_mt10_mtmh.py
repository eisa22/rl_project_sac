"""
Multi-Task Multi-Head SAC Training Script for MT10

Trains MTMH-SAC on 10 Meta-World tasks with configurable architecture.
Based on MTRL winner architecture.
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import wandb

from mtmh_sac import MTMHSACAgent


def main():
    parser = argparse.ArgumentParser(description="MTMH-SAC for MT10")
    parser.add_argument("--run_name", type=str, default="mtmh_sac_mt10")
    parser.add_argument("--total_steps", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--reward_scale", type=float, default=5.0)
    
    # Architecture config
    parser.add_argument("--trunk_hidden_actor", type=str, default="512,512",
                        help="Actor trunk hidden sizes (comma-separated)")
    parser.add_argument("--head_hidden_actor", type=str, default="256",
                        help="Actor head hidden sizes (comma-separated)")
    parser.add_argument("--trunk_hidden_critic", type=str, default="1024,1024",
                        help="Critic trunk hidden sizes (comma-separated)")
    parser.add_argument("--head_hidden_critic", type=str, default="512,512",
                        help="Critic head hidden sizes (comma-separated)")
    
    # Training config
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--log_freq", type=int, default=10_000)
    
    args = parser.parse_args()
    
    # Parse architecture
    trunk_hidden_actor = tuple(map(int, args.trunk_hidden_actor.split(',')))
    head_hidden_actor = tuple(map(int, args.head_hidden_actor.split(',')))
    trunk_hidden_critic = tuple(map(int, args.trunk_hidden_critic.split(',')))
    head_hidden_critic = tuple(map(int, args.head_hidden_critic.split(',')))
    
    # Config
    sac_config = {
        "learning_rate": args.lr,
        "alpha_lr": args.alpha_lr,
        "buffer_size": 1_000_000,
        "learning_starts": args.learning_starts,
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
    print("MTMH-SAC: Multi-Task Multi-Head (MT10)")
    print(f"Run: {args.run_name}")
    print(f"Trunk (actor): {trunk_hidden_actor}")
    print(f"Heads (actor): {head_hidden_actor}")
    print(f"Trunk (critic): {trunk_hidden_critic}")
    print(f"Heads (critic): {head_hidden_critic}")
    print("=" * 70)
    
    # MT10 Tasks
    import metaworld
    TASK_NAMES = [
        'reach-v3', 
        'push-v3', 
        'pick-place-v3',
        'door-open-v3',
        'drawer-close-v3',
        'button-press-topdown-v3',
        'peg-insert-side-v3',
        'window-open-v3',
        'drawer-open-v3',
        'door-close-v3'
    ]
    num_tasks = len(TASK_NAMES)
    
    print(f"\nMT10 Tasks:")
    for i, task_name in enumerate(TASK_NAMES):
        print(f"  {i}: {task_name}")
    print()
    
    # Create environments
    envs = []
    for task_name in TASK_NAMES:
        ml1 = metaworld.ML1(task_name, seed=args.seed)
        env = ml1.train_classes[task_name]()
        env.set_task(ml1.train_tasks[0])
        envs.append(env)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get dimensions from first env
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]
    act_limit = float(envs[0].action_space.high[0])
    
    print(f"Environment:")
    print(f"  obs_dim = {obs_dim}")
    print(f"  act_dim = {act_dim}")
    print(f"  act_limit = {act_limit}")
    print(f"  num_tasks = {num_tasks}\n")
    
    # Agent
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
    
    print("✓ MTMH-SAC Agent initialized\n")
    print("🚀 Starting MT10 training...\n")
    
    # Training state per task
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
    
    # Round-robin task sampling
    step = 0
    with tqdm(total=args.total_steps, desc="Training", unit="step") as pbar:
        while step < args.total_steps:
            for task_id in range(num_tasks):
                if step >= args.total_steps:
                    break
                
                obs = task_obs[task_id]
                
                # Select action
                if step < sac_config["learning_starts"]:
                    action = envs[task_id].action_space.sample()
                else:
                    action = agent.act(obs, task_id=task_id, deterministic=False)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = envs[task_id].step(action)
                done = bool(terminated or truncated)
                
                # Scale reward and add to buffer
                scaled_reward = reward * args.reward_scale
                agent.add_experience(obs, action, scaled_reward, next_obs, done, task_id)
                
                # Update state
                task_obs[task_id] = next_obs
                task_episode_reward[task_id] += reward
                task_episode_length[task_id] += 1
                
                # Update networks
                if step >= sac_config["learning_starts"]:
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
                
                # Handle episode end
                if done:
                    task_episode_count[task_id] += 1
                    task_episode_rewards[task_id].append(task_episode_reward[task_id])
                    task_episode_successes[task_id].append(info.get("success", False))
                    task_episode_lengths[task_id].append(task_episode_length[task_id])
                    
                    # Reset
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
                if step > 0 and step % args.log_freq == 0:
                    # Log per-task metrics
                    for tid in range(num_tasks):
                        if task_episode_rewards[tid]:
                            mean_reward = float(np.mean(task_episode_rewards[tid][-100:]))
                            mean_success = float(np.mean(task_episode_successes[tid][-100:]))
                            mean_length = float(np.mean(task_episode_lengths[tid][-100:]))
                            
                            wandb.log({
                                f"train/{TASK_NAMES[tid]}/mean_reward_100": mean_reward,
                                f"train/{TASK_NAMES[tid]}/success_rate_100": mean_success,
                                f"train/{TASK_NAMES[tid]}/episode_length_100": mean_length,
                                f"train/{TASK_NAMES[tid]}/total_episodes": task_episode_count[tid],
                                f"train/{TASK_NAMES[tid]}/alpha": agent.get_alpha(tid),
                                "train/step": step,
                            }, step=step)
                    
                    # Aggregate metrics
                    all_successes = [s for tid_successes in task_episode_successes for s in tid_successes[-100:]]
                    if all_successes:
                        wandb.log({
                            "train/aggregate_success_rate_100": float(np.mean(all_successes)),
                            "train/step": step,
                        }, step=step)
    
    # Save final model
    final_path = f"{model_dir}/final_model.pt"
    print(f"\n💾 Saving final model to: {final_path}")
    
    # Save actor heads and critic heads separately
    save_dict = {
        "actor_trunk": agent.actor.trunk.state_dict(),
        "critic_trunk": agent.critic.trunk.state_dict(),
        "critic_target_trunk": agent.critic_target.trunk.state_dict(),
        "log_alphas": agent.log_alphas.detach().cpu(),
        "config": sac_config,
    }
    
    # Save task heads
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
    print("TRAINING COMPLETED")
    print("=" * 70)
    for tid in range(num_tasks):
        if task_episode_rewards[tid]:
            print(f"{TASK_NAMES[tid]}:")
            print(f"  Final 100 episodes - Reward: {np.mean(task_episode_rewards[tid][-100:]):.1f}")
            print(f"  Final 100 episodes - Success Rate: {np.mean(task_episode_successes[tid][-100:]):.2%}")
    print(f"\nModel saved to: {final_path}")


if __name__ == "__main__":
    main()
