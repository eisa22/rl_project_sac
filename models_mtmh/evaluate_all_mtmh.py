#!/usr/bin/env python3
"""
Batch Evaluation Script for MTMH-SAC Checkpoints

Automatically evaluates all downloaded model checkpoints and generates a comparison report.
Works with PyTorch .pt checkpoints from MTMH-SAC training.

Usage:
    # Evaluate all checkpoints
    python evaluate_all_mtmh.py
    
    # Quick test (fewer episodes/seeds)
    python evaluate_all_mtmh.py --episodes 10 --n_seeds 1
    
    # Export results to CSV
    python evaluate_all_mtmh.py --output_csv results.csv
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch
import metaworld

# Add parent directory to path to import mtmh_sac
sys.path.insert(0, str(Path(__file__).parent.parent))
from mtmh_sac import MTMHSACAgent


# Model checkpoint configurations
CHECKPOINTS = [
    {
        "name": "MT10_params_v3_longtime_11.5M",
        "path": "mt10_mtmh_params_v3_longtime_checkpoints_500k_seed1/checkpoint_step11500000.pt",
        "tasks": ["reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-close-v3",
                  "button-press-topdown-v3", "peg-insert-side-v3", "window-open-v3", "drawer-open-v3", "door-close-v3"],
        "type": "mt10"
    },
    {
        "name": "MT10_gpu_optimized_v3_13M",
        "path": "mt10_mtmh_gpu_optimized_params_v3_longtime_checkpoints_seed1/checkpoint_step13000000.pt",
        "tasks": ["reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-close-v3",
                  "button-press-topdown-v3", "peg-insert-side-v3", "window-open-v3", "drawer-open-v3", "door-close-v3"],
        "type": "mt10"
    },
    {
        "name": "MT10_gpu_optimized_v5_fast_3M",
        "path": "mt10_mtmh_gpu_optimized_params_v5_fast_seed1/checkpoint_step3000000.pt",
        "tasks": ["reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-close-v3",
                  "button-press-topdown-v3", "peg-insert-side-v3", "window-open-v3", "drawer-open-v3", "door-close-v3"],
        "type": "mt10"
    },
    {
        "name": "MT3_gpu_optimized_v5_fast_3M",
        "path": "mt3_mtmh_gpu_optimized_params_v5_fast_seed1/checkpoint_step3000000.pt",
        "tasks": ["reach-v3", "push-v3", "pick-place-v3"],
        "type": "mt3"
    },
    {
        "name": "MT3_gpu_optimized_final",
        "path": "mt3_mtmh_gpu_optimized_seed1/final_model.pt",
        "tasks": ["reach-v3", "push-v3", "pick-place-v3"],
        "type": "mt3"
    },
    {
        "name": "MT3_longtime_1M",
        "path": "mt3_mtmh_longtime_checkpoints_500k_seed1/checkpoint_step1000000.pt",
        "tasks": ["reach-v3", "push-v3", "pick-place-v3"],
        "type": "mt3"
    },
    {
        "name": "MT3_params_v3_longtime_3M",
        "path": "mt3_mtmh_params_v3_longtime_checkpoints_500k_seed1/checkpoint_step3000000.pt",
        "tasks": ["reach-v3", "push-v3", "pick-place-v3"],
        "type": "mt3"
    },
]


def load_checkpoint(checkpoint_path: str, num_tasks: int, obs_dim: int = 39, act_dim: int = 4):
    """Load MTMH-SAC agent from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract architecture from checkpoint if available
    config = checkpoint.get('config', {})
    trunk_hidden_actor = config.get('trunk_hidden_actor', (512, 512))
    head_hidden_actor = config.get('head_hidden_actor', (256,))
    trunk_hidden_critic = config.get('trunk_hidden_critic', (1024, 1024))
    head_hidden_critic = config.get('head_hidden_critic', (512, 512))
    
    # Create agent with matching architecture
    agent = MTMHSACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=1.0,
        num_tasks=num_tasks,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha_lr=3e-4,
        trunk_hidden_actor=trunk_hidden_actor,
        head_hidden_actor=head_hidden_actor,
        trunk_hidden_critic=trunk_hidden_critic,
        head_hidden_critic=head_hidden_critic,
    )
    
    # Load weights - checkpoints save trunk and heads separately
    # Actor
    agent.actor.trunk.load_state_dict(checkpoint['actor_trunk'])
    for task_id in range(num_tasks):
        agent.actor.task_heads[task_id].load_state_dict(checkpoint[f'actor_head_{task_id}'])
    
    # Critic
    agent.critic.trunk.load_state_dict(checkpoint['critic_trunk'])
    for task_id in range(num_tasks):
        agent.critic.q1_heads[task_id].load_state_dict(checkpoint[f'critic_q1_head_{task_id}'])
        agent.critic.q2_heads[task_id].load_state_dict(checkpoint[f'critic_q2_head_{task_id}'])
    
    # Alpha parameters
    agent.log_alphas.data = checkpoint['log_alphas']
    
    return agent


def evaluate_task(agent: MTMHSACAgent, task_name: str, task_id: int, 
                 episodes: int = 50, seed: int = 42, max_steps: int = 500) -> Dict:
    """Evaluate agent on a single task."""
    
    # Create environment
    ml1 = metaworld.ML1(task_name, seed=seed)
    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[0])
    
    successes = []
    returns = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0
        success = False
        
        while not done and steps < max_steps:
            # Get action from agent (deterministic evaluation)
            action = agent.act(obs, task_id, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Track success
            if info.get('success', False):
                success = True
        
        successes.append(1.0 if success else 0.0)
        returns.append(total_reward)
    
    env.close()
    
    return {
        'task': task_name,
        'success_rate': np.mean(successes),
        'success_std': np.std(successes),
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'episodes': episodes
    }


def evaluate_checkpoint(checkpoint_info: Dict, base_dir: Path, 
                       episodes: int = 50, n_seeds: int = 3) -> Dict:
    """Evaluate a single checkpoint across all its tasks."""
    
    checkpoint_path = base_dir / checkpoint_info['path']
    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {checkpoint_info['name']}")
    print(f"Path: {checkpoint_path}")
    print(f"Type: {checkpoint_info['type'].upper()}")
    print(f"{'=' * 70}")
    
    tasks = checkpoint_info['tasks']
    num_tasks = len(tasks)
    
    # Load agent
    try:
        agent = load_checkpoint(str(checkpoint_path), num_tasks)
        print(f"✓ Loaded checkpoint")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None
    
    # Evaluate each task across multiple seeds
    all_task_results = []
    
    for task_idx, task_name in enumerate(tasks):
        print(f"\n  Task {task_idx+1}/{num_tasks}: {task_name}")
        
        seed_results = []
        for seed in range(n_seeds):
            result = evaluate_task(agent, task_name, task_idx, episodes, seed=42 + seed)
            seed_results.append(result)
            print(f"    Seed {seed+1}/{n_seeds}: Success={result['success_rate']:.1%}, "
                  f"Return={result['mean_return']:.1f}")
        
        # Aggregate across seeds
        task_result = {
            'task': task_name,
            'success_rate': np.mean([r['success_rate'] for r in seed_results]),
            'success_std': np.std([r['success_rate'] for r in seed_results]),
            'mean_return': np.mean([r['mean_return'] for r in seed_results]),
            'std_return': np.mean([r['std_return'] for r in seed_results]),
        }
        all_task_results.append(task_result)
    
    # Compute aggregate metrics
    avg_success = np.mean([r['success_rate'] for r in all_task_results])
    avg_return = np.mean([r['mean_return'] for r in all_task_results])
    
    print(f"\n  {'─' * 66}")
    print(f"  AGGREGATE: Success={avg_success:.1%}, Mean Return={avg_return:.1f}")
    print(f"  {'─' * 66}")
    
    return {
        'name': checkpoint_info['name'],
        'type': checkpoint_info['type'],
        'path': str(checkpoint_path),
        'task_results': all_task_results,
        'avg_success_rate': avg_success,
        'avg_return': avg_return,
        'num_tasks': num_tasks,
        'episodes_per_task': episodes,
        'n_seeds': n_seeds,
    }


def print_comparison_table(results: List[Dict]):
    """Print a comparison table of all evaluated checkpoints."""
    
    if not results:
        print("No results to display")
        return
    
    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY - ALL CHECKPOINTS")
    print("=" * 100)
    
    # Header
    print(f"{'Model':<45} {'Type':<6} {'Avg Success':<13} {'Avg Return':<12} {'Tasks':<6}")
    print("-" * 100)
    
    # Sort by type and success rate
    sorted_results = sorted(results, key=lambda x: (x['type'], -x['avg_success_rate']))
    
    for res in sorted_results:
        print(f"{res['name']:<45} {res['type'].upper():<6} "
              f"{res['avg_success_rate']:>6.1%} ± {np.std([r['success_rate'] for r in res['task_results']]):>4.1%}   "
              f"{res['avg_return']:>6.1f} ± {np.std([r['mean_return'] for r in res['task_results']]):>4.1f}   "
              f"{res['num_tasks']:<6}")
    
    print("=" * 100)
    
    # Best models
    print("\n🏆 BEST MODELS:")
    mt3_best = max([r for r in results if r['type'] == 'mt3'], 
                   key=lambda x: x['avg_success_rate'], default=None)
    mt10_best = max([r for r in results if r['type'] == 'mt10'], 
                    key=lambda x: x['avg_success_rate'], default=None)
    
    if mt3_best:
        print(f"  MT3:  {mt3_best['name']:<45} Success={mt3_best['avg_success_rate']:.1%}")
    if mt10_best:
        print(f"  MT10: {mt10_best['name']:<45} Success={mt10_best['avg_success_rate']:.1%}")


def export_to_csv(results: List[Dict], output_path: str):
    """Export results to CSV file."""
    
    rows = []
    for res in results:
        # Overall metrics
        base_row = {
            'model': res['name'],
            'type': res['type'],
            'avg_success_rate': res['avg_success_rate'],
            'avg_return': res['avg_return'],
            'num_tasks': res['num_tasks'],
            'episodes': res['episodes_per_task'],
            'seeds': res['n_seeds'],
        }
        
        # Add per-task metrics
        for task_res in res['task_results']:
            row = base_row.copy()
            row['task'] = task_res['task']
            row['task_success_rate'] = task_res['success_rate']
            row['task_success_std'] = task_res['success_std']
            row['task_mean_return'] = task_res['mean_return']
            row['task_std_return'] = task_res['std_return']
            rows.append(row)
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['model', 'type', 'task', 'task_success_rate', 'task_success_std',
                     'task_mean_return', 'task_std_return', 'avg_success_rate', 'avg_return',
                     'num_tasks', 'episodes', 'seeds']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of MTMH-SAC checkpoints")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Episodes per task per seed (default: 50)")
    parser.add_argument("--n_seeds", type=int, default=3,
                       help="Number of random seeds for evaluation (default: 3)")
    parser.add_argument("--output_csv", type=str, default=None,
                       help="Export results to CSV file")
    parser.add_argument("--base_dir", type=str, default=None,
                       help="Base directory containing checkpoints (default: script directory)")
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(__file__).parent
    
    print(f"Base directory: {base_dir}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Random seeds: {args.n_seeds}")
    
    # Evaluate all checkpoints
    results = []
    for checkpoint_info in CHECKPOINTS:
        result = evaluate_checkpoint(checkpoint_info, base_dir, args.episodes, args.n_seeds)
        if result:
            results.append(result)
    
    # Print comparison
    print_comparison_table(results)
    
    # Export to CSV
    if args.output_csv and results:
        export_to_csv(results, args.output_csv)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
