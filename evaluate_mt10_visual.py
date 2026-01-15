"""
MTMH-SAC Evaluation Script with Simulator Visualization

Evaluates trained MTMH-SAC models on MT10 Meta-World tasks with live visualization.

Usage:
    python evaluate_mt10_visual.py --model_path models_mtmh/dummy_model/final_model.pt --task reach-v3
    python evaluate_mt10_visual.py --model_path models_mtmh/dummy_model/final_model.pt --task all --num_episodes 5
"""

import argparse
import numpy as np
import torch
import time
from collections import defaultdict

from mtmh_sac import MTMHSACAgent, MultiHeadActor, MultiHeadCritic

# MT10 Task definitions
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


def load_model(model_path, device='cuda'):
    """Load MTMH-SAC model from checkpoint."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config
    config = checkpoint.get('config', {})
    
    # Default architecture (from train_mt10_mtmh.py defaults)
    trunk_hidden_actor = config.get('trunk_hidden_actor', (512, 512))
    head_hidden_actor = config.get('head_hidden_actor', (256,))
    trunk_hidden_critic = config.get('trunk_hidden_critic', (1024, 1024))
    head_hidden_critic = config.get('head_hidden_critic', (512, 512))
    
    # MT10 dimensions
    obs_dim = 39  # Meta-World observation space
    act_dim = 4   # Meta-World action space
    act_limit = 1.0
    num_tasks = 10
    
    # Create actor (we only need actor for evaluation)
    actor = MultiHeadActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        num_tasks=num_tasks,
        trunk_hidden=trunk_hidden_actor,
        head_hidden=head_hidden_actor
    ).to(device)
    
    # Load trunk weights
    actor.trunk.load_state_dict(checkpoint['actor_trunk'])
    
    # Load task-specific head weights
    for tid in range(num_tasks):
        key = f'actor_head_{tid}'
        if key in checkpoint:
            actor.task_heads[tid].load_state_dict(checkpoint[key])
    
    print(f"✓ Model loaded (step {checkpoint.get('step', 'unknown')})")
    print(f"  Actor trunk: {trunk_hidden_actor}")
    print(f"  Actor heads: {head_hidden_actor}")
    
    return actor, config


def create_env(task_name, render_mode='human', seed=42):
    """Create a Meta-World environment for a specific task."""
    import metaworld
    
    ml1 = metaworld.ML1(task_name, seed=seed)
    env_class = ml1.train_classes[task_name]
    env = env_class(render_mode=render_mode)
    env.set_task(ml1.train_tasks[0])
    
    return env


def evaluate_task(actor, task_name, task_id, num_episodes=5, render=True, delay=0.02, device='cuda'):
    """Evaluate actor on a single task with optional rendering."""
    render_mode = 'human' if render else None
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {task_name} (task_id={task_id})")
    print(f"{'='*60}")
    
    try:
        env = create_env(task_name, render_mode=render_mode)
    except Exception as e:
        print(f"Error creating environment for {task_name}: {e}")
        return None
    
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
        episode_reward = 0.0
        episode_success = False
        step_count = 0
        max_steps = 500  # Meta-World default
        
        while step_count < max_steps:
            # Get action from actor
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, _ = actor.forward(obs_tensor, task_id, deterministic=True, with_logprob=False)
            
            action = action.cpu().numpy()[0]
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            if info.get('success', False):
                episode_success = True
            
            obs = next_obs
            
            # Rendering delay for visualization
            if render and delay > 0:
                time.sleep(delay)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_successes.append(episode_success)
        episode_lengths.append(step_count)
        
        status = "✓ SUCCESS" if episode_success else "✗ FAILED"
        print(f"  Episode {ep+1}/{num_episodes}: {status} | Reward: {episode_reward:.2f} | Steps: {step_count}")
    
    env.close()
    
    # Compute statistics
    results = {
        'task': task_name,
        'task_id': task_id,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': np.mean(episode_successes),
        'mean_length': np.mean(episode_lengths),
        'num_episodes': num_episodes
    }
    
    print(f"\n  Summary:")
    print(f"    Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"    Success Rate: {results['success_rate']*100:.1f}%")
    print(f"    Mean Episode Length: {results['mean_length']:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MTMH-SAC Evaluation with Visualization")
    parser.add_argument("--model_path", type=str, default="models_mtmh/dummy_model/final_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--task", type=str, default="reach-v3",
                        help="Task to evaluate (task name or 'all')")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes per task")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable rendering (faster evaluation)")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Delay between steps for visualization (seconds)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model
    actor, config = load_model(args.model_path, device=device)
    actor.eval()
    
    # Determine tasks to evaluate
    if args.task.lower() == 'all':
        tasks_to_eval = list(enumerate(TASK_NAMES))
    else:
        if args.task not in TASK_NAMES:
            print(f"Error: Unknown task '{args.task}'")
            print(f"Available tasks: {TASK_NAMES}")
            return
        task_id = TASK_NAMES.index(args.task)
        tasks_to_eval = [(task_id, args.task)]
    
    # Run evaluation
    all_results = []
    
    for task_id, task_name in tasks_to_eval:
        result = evaluate_task(
            actor=actor,
            task_name=task_name,
            task_id=task_id,
            num_episodes=args.num_episodes,
            render=not args.no_render,
            delay=args.delay,
            device=device
        )
        if result:
            all_results.append(result)
    
    # Print final summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("FINAL SUMMARY (All Tasks)")
        print(f"{'='*60}")
        
        total_success = np.mean([r['success_rate'] for r in all_results])
        total_reward = np.mean([r['mean_reward'] for r in all_results])
        
        print(f"\n{'Task':<30} {'Success Rate':<15} {'Mean Reward':<15}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['task']:<30} {r['success_rate']*100:>10.1f}% {r['mean_reward']:>15.2f}")
        print("-" * 60)
        print(f"{'AVERAGE':<30} {total_success*100:>10.1f}% {total_reward:>15.2f}")


if __name__ == "__main__":
    main()
