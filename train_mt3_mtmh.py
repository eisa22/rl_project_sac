"""
Multi-Task Multi-Head SAC Training Script for MT3

Trains MTMH-SAC on 3 Meta-World tasks with configurable architecture.
Based on MTRL winner architecture.
"""

import argparse
import os
import copy
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import wandb

from mtmh_sac import MTMHSACAgent


def main():
    parser = argparse.ArgumentParser(description="MTMH-SAC for MT3")
    parser.add_argument("--run_name", type=str, default="mtmh_sac_mt3")
    parser.add_argument("--total_steps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--pick_place_base_scale", type=float, default=1.0,
                        help="Optional initial multiplier for pick-place rewards")
    parser.add_argument("--ars_enable", action="store_true",
                        help="Enable adaptive reward scaling per task")
    parser.add_argument("--ars_update_freq", type=int, default=50_000,
                        help="How often (steps) to recompute ARS scales")
    parser.add_argument("--ars_bootstrap_steps", type=int, default=10_000,
                        help="Start ARS only after this many steps (for buffer stats)")
    parser.add_argument("--ars_min_scale", type=float, default=1.0,
                        help="Lower clamp for ARS scales")
    parser.add_argument("--ars_max_scale", type=float, default=200.0,
                        help="Upper clamp for ARS scales")
    parser.add_argument("--reset_enable", action="store_true",
                        help="Periodically reset actor/critic weights (buffer retained)")
    parser.add_argument("--reset_every", type=int, default=5_000_000,
                        help="Steps between resets if enabled")
    parser.add_argument("--num_envs_per_task", type=int, default=1,
                        help="Number of parallel environments per task")
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="If >0, save checkpoint every N steps (per-task heads + trunks)")
    
    # Architecture config
    parser.add_argument("--trunk_hidden_actor", type=str, default="256,256",
                        help="Actor trunk hidden sizes (comma-separated)")
    parser.add_argument("--head_hidden_actor", type=str, default="256",
                        help="Actor head hidden sizes (comma-separated)")
    parser.add_argument("--trunk_hidden_critic", type=str, default="512,512",
                        help="Critic trunk hidden sizes (comma-separated)")
    parser.add_argument("--head_hidden_critic", type=str, default="512",
                        help="Critic head hidden sizes (comma-separated)")
    
    # Training config
    parser.add_argument("--learning_starts", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--log_freq", type=int, default=10_000)
    parser.add_argument("--update_every", type=int, default=1,
                        help="Perform network update every N steps (GPU optimization)")
    
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
        "update_every": args.update_every,
        "checkpoint_every": args.checkpoint_every,
        "trunk_hidden_actor": trunk_hidden_actor,
        "head_hidden_actor": head_hidden_actor,
        "trunk_hidden_critic": trunk_hidden_critic,
        "head_hidden_critic": head_hidden_critic,
        "ars_enable": args.ars_enable,
        "ars_update_freq": args.ars_update_freq,
        "ars_bootstrap_steps": args.ars_bootstrap_steps,
        "ars_min_scale": args.ars_min_scale,
        "ars_max_scale": args.ars_max_scale,
        "pick_place_base_scale": args.pick_place_base_scale,
        "reset_enable": args.reset_enable,
        "reset_every": args.reset_every,
        "num_envs_per_task": args.num_envs_per_task,
    }
    
    # Log to team Robot_learning_2025, project Robot_learning_2025
    wandb.init(
        entity="Robot_learning_2025",
        project="Robot_learning_2025",
        name=args.run_name,
        config=sac_config
    )
    
    os.makedirs("./models_mtmh", exist_ok=True)
    model_dir = f"./models_mtmh/{args.run_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 70)
    print("MTMH-SAC: Multi-Task Multi-Head (MT3)")
    print(f"Run: {args.run_name}")
    print(f"Tasks: reach-v3, push-v3, pick-place-v3")
    print(f"Trunk (actor): {trunk_hidden_actor}")
    print(f"Heads (actor): {head_hidden_actor}")
    print(f"Trunk (critic): {trunk_hidden_critic}")
    print(f"Heads (critic): {head_hidden_critic}")
    print("=" * 70)
    
    # MT3 Tasks
    import metaworld
    TASK_NAMES = ['reach-v3', 'push-v3', 'pick-place-v3']
    num_tasks = len(TASK_NAMES)
    num_envs_per_task = args.num_envs_per_task
    
    # Create parallel environments: envs[task_id][env_idx]
    envs = []
    tasks = []
    for task_name in TASK_NAMES:
        task_envs = []
        for env_idx in range(num_envs_per_task):
            ml1 = metaworld.ML1(task_name, seed=args.seed + env_idx)
            env = ml1.train_classes[task_name]()
            env.set_task(ml1.train_tasks[0])
            task_envs.append(env)
        envs.append(task_envs)
        tasks.append(task_name)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Enable cuDNN benchmarking for GPU optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Get dimensions from first env
    obs_dim = envs[0][0].observation_space.shape[0]
    act_dim = envs[0][0].action_space.shape[0]
    act_limit = float(envs[0][0].action_space.high[0])
    
    print(f"\nEnvironment:")
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

    # Save initial weights for periodic resets (actor/critic/target/log_alphas)
    initial_actor_state = copy.deepcopy(agent.actor.state_dict())
    initial_critic_state = copy.deepcopy(agent.critic.state_dict())
    initial_log_alphas = agent.log_alphas.detach().cpu().clone()

    def reset_agent_weights():
        agent.actor.load_state_dict(initial_actor_state)
        agent.critic.load_state_dict(initial_critic_state)
        agent.critic_target.load_state_dict(agent.critic.state_dict())
        with torch.no_grad():
            agent.log_alphas.copy_(initial_log_alphas.to(agent.log_alphas.device))
        agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=sac_config["learning_rate"])
        agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=sac_config["learning_rate"])
        agent.alpha_optimizer = torch.optim.Adam([agent.log_alphas], lr=sac_config["alpha_lr"])
        print("[Reset] Actor/Critic/Alpha weights reset; buffer retained")

    def save_checkpoint(path):
        save_dict = {
            "actor_trunk": agent.actor.trunk.state_dict(),
            "critic_trunk": agent.critic.trunk.state_dict(),
            "critic_target_trunk": agent.critic_target.trunk.state_dict(),
            "log_alphas": agent.log_alphas.detach().cpu(),
            "config": sac_config,
            "step": step,
        }

        for tid in range(num_tasks):
            save_dict[f"actor_head_{tid}"] = agent.actor.task_heads[tid].state_dict()
            save_dict[f"critic_q1_head_{tid}"] = agent.critic.q1_heads[tid].state_dict()
            save_dict[f"critic_q2_head_{tid}"] = agent.critic.q2_heads[tid].state_dict()
            save_dict[f"critic_target_q1_head_{tid}"] = agent.critic_target.q1_heads[tid].state_dict()
            save_dict[f"critic_target_q2_head_{tid}"] = agent.critic_target.q2_heads[tid].state_dict()

        torch.save(save_dict, path)
        print(f"[Checkpoint] Saved to {path}")

    # ARS state
    ars_scales = np.ones(num_tasks, dtype=np.float32)
    # Optional initial boost for pick-place
    pick_place_tid = TASK_NAMES.index('pick-place-v3')
    ars_scales[pick_place_tid] *= args.pick_place_base_scale
    next_ars_update = args.ars_bootstrap_steps if args.ars_enable else np.inf
    next_reset_step = args.reset_every if args.reset_enable else np.inf
    
    print("✓ MTMH-SAC Agent initialized\n")
    print("🚀 Starting MT3 training...\n")
    
    # Training state per task and env: [task_id][env_idx]
    task_env_obs = []
    task_env_episode_reward = []
    task_env_episode_length = []
    task_episode_count = []
    task_episode_rewards = []
    task_episode_successes = []
    task_episode_lengths = []
    
    for task_id in range(num_tasks):
        env_obs_list = []
        env_reward_list = []
        env_length_list = []
        
        for env_idx in range(num_envs_per_task):
            reset_out = envs[task_id][env_idx].reset()
            if isinstance(reset_out, tuple):
                obs, _ = reset_out
            else:
                obs = reset_out
            env_obs_list.append(obs)
            env_reward_list.append(0.0)
            env_length_list.append(0)
        
        task_env_obs.append(env_obs_list)
        task_env_episode_reward.append(env_reward_list)
        task_env_episode_length.append(env_length_list)
        task_episode_count.append(0)
        task_episode_rewards.append([])
        task_episode_successes.append([])
        task_episode_lengths.append([])
    
    # Vectorized training loop (GPU optimization)
    step = 0
    total_envs = num_tasks * num_envs_per_task  # 24 environments for MT3

    with tqdm(total=args.total_steps, desc="Training", unit="step") as pbar:
        while step < args.total_steps:
            # Collect all observations for batch inference
            all_obs = []
            all_task_ids = []
            env_indices = []  # Track which (task_id, env_idx) each obs belongs to

            for task_id in range(num_tasks):
                for env_idx in range(num_envs_per_task):
                    all_obs.append(task_env_obs[task_id][env_idx])
                    all_task_ids.append(task_id)
                    env_indices.append((task_id, env_idx))

            # Batch action selection (1 GPU call instead of 24!)
            if step < sac_config["learning_starts"]:
                # Random actions during warmup
                all_actions = np.array([envs[tid][eidx].action_space.sample()
                                       for tid, eidx in env_indices])
            else:
                # Batch inference (GPU optimization)
                all_actions = agent.act_batch(np.array(all_obs), np.array(all_task_ids), deterministic=False)

            # Step all environments and collect transitions
            for idx, (task_id, env_idx) in enumerate(env_indices):
                if step >= args.total_steps:
                    break

                obs = all_obs[idx]
                action = all_actions[idx]

                # Step environment
                next_obs, reward, terminated, truncated, info = envs[task_id][env_idx].step(action)
                done = bool(terminated or truncated)

                # Adaptive reward scaling per task
                task_scale = ars_scales[task_id] if args.ars_enable else 1.0
                scaled_reward = reward * args.reward_scale * task_scale
                agent.add_experience(obs, action, scaled_reward, next_obs, done, task_id)

                # Update state
                task_env_obs[task_id][env_idx] = next_obs
                task_env_episode_reward[task_id][env_idx] += reward
                task_env_episode_length[task_id][env_idx] += 1

                # Update networks (with reduced frequency for GPU efficiency)
                if step >= sac_config["learning_starts"] and step % args.update_every == 0:
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
                    task_episode_rewards[task_id].append(task_env_episode_reward[task_id][env_idx])
                    task_episode_successes[task_id].append(info.get("success", False))
                    task_episode_lengths[task_id].append(task_env_episode_length[task_id][env_idx])

                    # Reset
                    reset_out = envs[task_id][env_idx].reset()
                    if isinstance(reset_out, tuple):
                        task_env_obs[task_id][env_idx], _ = reset_out
                    else:
                        task_env_obs[task_id][env_idx] = reset_out
                    task_env_episode_reward[task_id][env_idx] = 0.0
                    task_env_episode_length[task_id][env_idx] = 0

                step += 1
                pbar.update(1)

                if step >= args.total_steps:
                    break
                
                if step >= args.total_steps:
                    break

                # ARS update (based on replay buffer means)
                if args.ars_enable and step >= next_ars_update:
                    buf = agent.replay_buffer.buffers
                    means = []
                    for tid in range(num_tasks):
                        size = buf[tid]['size']
                        if size == 0:
                            means.append(None)
                        else:
                            means.append(float(buf[tid]['rews'][:size].mean()))
                    valid_means = [m for m in means if m is not None and m != 0.0]
                    if valid_means:
                        max_mean = max(abs(m) for m in valid_means)
                        eps = 1e-6
                        new_scales = []
                        for tid, m in enumerate(means):
                            if m is None or abs(m) < eps:
                                new_scales.append(ars_scales[tid])
                            else:
                                raw = max_mean / abs(m)
                                raw = np.clip(raw, args.ars_min_scale, args.ars_max_scale)
                                # KEEP pick-place base boost: multiply ARS scale with base
                                if tid == pick_place_tid:
                                    raw *= args.pick_place_base_scale
                                new_scales.append(raw)
                        ars_scales = np.array(new_scales, dtype=np.float32)
                        print(f"[ARS] step={step} scales={ars_scales} means={means}")
                    next_ars_update += args.ars_update_freq

                # Periodic reset of networks (keep buffer)
                if args.reset_enable and step >= next_reset_step:
                    reset_agent_weights()
                    next_reset_step += args.reset_every
                
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
                                f"train/{TASK_NAMES[tid]}/ars_scale": float(ars_scales[tid]) if args.ars_enable else 1.0,
                                "train/step": step,
                            }, step=step)
                    
                    # Aggregate metrics
                    all_successes = [s for tid_successes in task_episode_successes for s in tid_successes[-100:]]
                    if all_successes:
                        wandb.log({
                            "train/aggregate_success_rate_100": float(np.mean(all_successes)),
                            "train/step": step,
                        }, step=step)

                # Periodic checkpointing
                if args.checkpoint_every > 0 and step > 0 and step % args.checkpoint_every == 0:
                    ckpt_path = f"{model_dir}/checkpoint_step{step}.pt"
                    save_checkpoint(ckpt_path)
    
    # Save final model
    final_path = f"{model_dir}/final_model.pt"
    print(f"\n💾 Saving final model to: {final_path}")
    save_checkpoint(final_path)
    
    wandb.finish()
    for task_envs in envs:
        for env in task_envs:
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
