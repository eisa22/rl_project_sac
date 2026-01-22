"""
Final Evaluation Script for MTMH-SAC on Meta-World MT1/MT3/MT10.

This script provides proper scientific evaluation following Meta-World benchmarks:
- Per-task success rates with bootstrap confidence intervals
- Aggregate metrics: Mean and IQM (Interquartile Mean)
- Multi-seed evaluation for statistical robustness
- Stability metrics (mean reward and standard deviation)
- Grading criteria check (Pass/Fail against project requirements)
- CSV export for paper-style reporting

Grading Criteria (Project Requirements):
- MT1 Single-task: reach > 90%, push > 30%, pick-place > 30%
- MT3 Multi-task: average success rate > 40%
- MT10 Multi-task: average success rate > 30%

Usage:
    # Evaluate MT3 final model
    python evaluate.py --run_name my_run --which final --episodes 100 --n_seeds 5
    
    # Evaluate MT1 single-task models
    python evaluate.py --run_name my_run --eval_mode mt1 --episodes 100
    
    # Evaluate MT10 model
    python evaluate.py --run_name my_run --eval_mode mt10 --episodes 100
    
    # Export results to CSV
    python evaluate.py --run_name my_run --output_csv results.csv

References:
- Statistical Best Practices for RL: https://arxiv.org/abs/2108.13264
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
from stable_baselines3 import SAC

# Import custom classes for SB3 deserialization
import sac_agent_sb3.multiheadAgent  # noqa: F401

from sac_agent_sb3.curriculum import make_curriculum_vec_env
from sac_agent_sb3.evaluation import (
    TaskEvalResult,              # Contains returns, successes, lengths for single task
    MultiTaskEvalResult,         # Contains results for multiple tasks in one seed
    MultiSeedEvalResult,         # Contains results across all seeds
    interquartile_mean,          # Robust aggregate metric (less sensitive to outliers)
    bootstrap_ci,                # Bootstrap confidence intervals for robust statistics
    stratified_bootstrap_ci,     # Stratified bootstrap for more stable CI estimates
)

# ============================================================================
# TASK DEFINITIONS - Different Meta-World benchmark levels
# ============================================================================
# MT1: Single-task evaluation - evaluate each task separately
#      Tests if the agent is specialist at each individual task
MT1_TASKS = ["reach-v3", "push-v3", "pick-place-v3"]

# MT3: Multi-task learning with 3 tasks - tests generalization
#      Agent trained on all 3 tasks must perform well on all
MT3_TASKS = ["reach-v3", "push-v3", "pick-place-v3"]

# MT10: Full Meta-World benchmark with 10 diverse tasks
#       Most challenging - requires good generalization to diverse tasks
MT10_TASKS = [
    "reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-close-v3",
    "button-press-topdown-v3", "peg-insert-side-v3", "window-open-v3", 
    "sweep-v3", "basketball-v3"
]

# ============================================================================
# GRADING THRESHOLDS - Project submission requirements
# ============================================================================
# These thresholds define what constitutes a passing evaluation result.
# They come from the project specification, not from the Meta-World paper.
GRADING_THRESHOLDS = {
    # MT1: Single-task specialization - higher bar (already trained on single task)
    #      reach: very easy (manipulation primitives), push/pick-place: harder
    "mt1": {
        "reach-v3": 0.90,       # > 90% - Most basic skill, should be easiest
        "push-v3": 0.30,        # > 30% - Harder skill, lower threshold
        "pick-place-v3": 0.30,  # > 30% - Most complex, lowest threshold
    },
    # MT3: Multi-task generalization across 3 related tasks
    #      Average > 40% - easier than single-task (agent sees all tasks during training)
    "mt3": {
        "average": 0.40,        # Average across 3 tasks > 40%
    },
    # MT10: Full benchmark with 10 diverse tasks (hardest)
    #       Average > 30% - significantly harder, lower bar due to task diversity
    "mt10": {
        "average": 0.30,        # Average across 10 tasks > 30%
    },
}

DEFAULT_TASKS = MT3_TASKS  # Default to MT3 (standard benchmark)


def find_models(model_root: str, which: str, tasks: List[str]) -> Dict[str, str]:
    """
    Discover model checkpoints saved by the training script.

    Returns a dict: {label -> model_path}

    Options:
    - final: uses final*.zip saved at the end of training
    - best:  uses per-task "best_model.zip" saved by SB3's EvalCallback
    - all:   evaluates both final and per-task best models
    """
    model_root = os.path.abspath(model_root)
    out: Dict[str, str] = {}

    # Final checkpoint(s)
    if which in ("final", "all"):
        finals = sorted(glob.glob(os.path.join(model_root, "final*.zip")))
        if finals:
            out["final"] = finals[-1]

    # Best-per-task checkpoints from EvalCallback
    if which in ("best", "all"):
        for t in tasks:
            p = os.path.join(model_root, f"best_{t}", "best_model.zip")
            if os.path.exists(p):
                out[f"best_{t}"] = p
            else:
                d = os.path.join(model_root, f"best_{t}")
                if os.path.isdir(d):
                    z = sorted(glob.glob(os.path.join(d, "*.zip")))
                    if z:
                        out[f"best_{t}"] = z[-1]

    if not out:
        raise FileNotFoundError(f"No models found in {model_root} for which='{which}'")

    return out


def is_vec_env(env) -> bool:
    """Check if env is a VecEnv (has num_envs attribute)."""
    return hasattr(env, "num_envs")


def reset_env(env):
    """Reset wrapper that works across Gymnasium and SB3 VecEnv conventions."""
    res = env.reset()
    if isinstance(res, tuple):
        obs = res[0]
    else:
        obs = res
    return obs


def step_env(env, action):
    """
    Step wrapper that normalizes different step() signatures into:
      obs, reward(float), done(bool), info(dict)
    """
    out = env.step(action)

    if not isinstance(out, tuple):
        raise ValueError(f"Unexpected step return: {out}")

    if len(out) == 5:
        if is_vec_env(env):
            obs, rewards, terminated, truncated, infos = out
            reward = float(rewards[0])
            done = bool(terminated[0] or truncated[0])
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            return obs, reward, done, info
        else:
            obs, reward, terminated, truncated, info = out
            return obs, float(reward), bool(terminated or truncated), info

    if len(out) == 4:
        if is_vec_env(env):
            obs, rewards, dones, infos = out
            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            return obs, reward, done, info
        else:
            obs, reward, done, info = out
            return obs, float(reward), bool(done), info

    raise ValueError(f"Unrecognized step tuple length: {len(out)}")


def eval_one_task(
    model: SAC,
    task_name: str,
    episodes: int,
    seed: int,
    task_sequence: List[str],
    max_episode_steps: int,
) -> TaskEvalResult:
    """
    Evaluate model on a single task for multiple episodes.
    
    This function:
    1. Creates a deterministic environment (no randomness except initial state)
    2. Runs N independent episodes
    3. Collects per-episode metrics: returns (rewards), success/failure, episode length
    4. Returns aggregated statistics for this task
    
    Why deterministic evaluation?
    - Policy is fixed (no training), so determinism lets us see raw performance
    - Reproducibility: same seed gives same trajectory
    - Comparison: different random seeds test generalization to different environments
    
    Returns:
        TaskEvalResult with:
            - returns: np.array of episode returns (sum of rewards per episode)
            - successes: np.array of binary success indicators (task success/failure)
            - lengths: np.array of episode lengths (steps taken)
    """
    # Create deterministic single-env evaluation environment
    # - num_envs=1: Single environment (not vectorized)
    # - fixed_task_name: Only evaluate on this specific task (not curriculum)
    # - curriculum_thresholds=[0.0]: No curriculum - task always available
    # - different seed per task to avoid correlated initial states
    env, _ = make_curriculum_vec_env(
        num_envs=1,
        seed=seed,
        task_sequence=task_sequence,
        curriculum_thresholds=[0.0] * len(task_sequence),  # No curriculum
        max_episode_steps=max_episode_steps,
        fixed_task_name=task_name,  # Fix to single task
        force_subproc=False,  # Don't use subprocess for single env
    )

    # Accumulate episode metrics across multiple evaluation episodes
    ep_returns = []   # Sum of rewards per episode
    ep_success = []   # Binary success indicator (0 or 1)
    ep_lengths = []   # Number of steps taken

    # Run multiple episodes to get robust statistics
    for _ in range(episodes):
        # Initialize episode
        obs = reset_env(env)
        done = False
        total_return = 0.0  # Accumulate rewards
        succeeded = 0       # Track if episode succeeded
        length = 0          # Count steps

        # Run single episode until completion or max steps
        while not done and length < max_episode_steps:
            # Get deterministic action (no exploration noise)
            action, _ = model.predict(obs, deterministic=True)
            # Step environment
            obs, r, done, info = step_env(env, action)
            # Accumulate metrics
            total_return += r
            length += 1

            # Extract success indicator from environment info
            # Different Meta-World environments may use different keys
            if isinstance(info, dict):
                succeeded = max(
                    succeeded,
                    int(bool(info.get("is_success", info.get("success", False))))
                )

        # Record this episode's metrics
        ep_returns.append(total_return)
        ep_success.append(succeeded)
        ep_lengths.append(length)

    env.close()
    
    return TaskEvalResult(
        task_name=task_name,
        returns=np.array(ep_returns),
        successes=np.array(ep_success),
        lengths=np.array(ep_lengths)
    )


def evaluate_model_multiseed(
    model: SAC,
    model_label: str,
    tasks: List[str],
    episodes_per_task: int,
    n_seeds: int,
    seed_base: int,
    max_episode_steps: int,
) -> MultiSeedEvalResult:
    """
    Evaluate model across multiple random seeds for robust statistics.
    
    Why multiple seeds?
    - Different seeds = different random environment initializations
    - Allows estimation of performance variance (std dev)
    - Matches Meta-World benchmark evaluation protocol
    - Typical practice: 5-10 seeds for publication-quality results
    
    Structure:
    1. For each seed:
       a. For each task:
          - Run eval_one_task() with different random seed
          - Collect success rates and returns for that task
       b. Aggregate all tasks for this seed
    2. Return all seeds' results for downstream statistical analysis
    
    Returns:
        MultiSeedEvalResult containing results from all seeds
    """
    per_seed_results = []  # Accumulate results from each seed
    
    # Evaluate model under multiple random seeds
    for s in range(n_seeds):
        task_results = {}  # Results for all tasks under this seed
        
        # Evaluate on all tasks (with different random seed for each)
        for task_i, task in enumerate(tasks):
            # Generate unique seed for each (seed, task) pair
            # This ensures:
            # 1. Different environments for different seeds (s varies)
            # 2. Different environments for different tasks within same seed (task_i varies)
            # 3. Reproducibility: same inputs always produce same randomness
            seed = seed_base + s + 10_000 * task_i
            
            # Evaluate this task under this seed
            result = eval_one_task(
                model=model,
                task_name=task,
                episodes=episodes_per_task,  # e.g., 100 episodes per task
                seed=seed,
                task_sequence=tasks,
                max_episode_steps=max_episode_steps,
            )
            task_results[task] = result
        
        # Aggregate all tasks for this seed
        per_seed_results.append(MultiTaskEvalResult(task_results=task_results, seed=s))
    
    # Return results from all seeds for statistical analysis
    return MultiSeedEvalResult(per_seed_results=per_seed_results)


def check_grading_criteria(result: MultiSeedEvalResult, eval_mode: str, tasks: List[str]) -> Dict:
    """
    Check if evaluation results meet the grading criteria and compute stability metrics.
    
    ========================================================================
    GRADING CRITERIA (from project specification):
    ========================================================================
    - MT1: Individual success rate thresholds (reach>90%, push>30%, pick-place>30%)
    - MT3: Average success rate > 40% across 3 tasks
    - MT10: Average success rate > 30% across 10 tasks (harder -> lower threshold)
    
    ========================================================================
    STABILITY METRICS (from project requirement - why these metrics?):
    ========================================================================
    Mean Reward:
    - What: Average return (sum of rewards) across all episodes and tasks
    - Why: Shows expected cumulative reward - overall agent performance
    - Formula: mean([task1_mean_return, task2_mean_return, ...])
    - Unit: Reward value (environment-dependent, typically -500 to +100 for Meta-World)
    - Interpretation: Higher = better performance; relative metric
    
    Standard Deviation of Rewards:
    - What: Variability of mean returns across tasks
    - Why: Shows consistency/stability - low variance = consistent across tasks
    - Formula: std([task1_mean_return, task2_mean_return, ...])
    - Unit: Same as mean reward
    - Interpretation: Lower = more stable; indicates specialization vs generalization
    - Example: 
      * 103.35 ± 5.45 = reaches ~103 on average, varies by ~5 across tasks (stable)
      * 100 ± 20 = reaches ~100 on average, varies by ~20 across tasks (unstable)
    
    Why these metrics matter:
    1. Success rate (%) = binary outcome: success or failure
    2. Mean reward = continuous signal: HOW WELL you succeed (or gracefully fail)
    3. Std dev = robustness: does agent work equally well on all tasks?
    
    Together they answer:
    - Success rate: Can agent complete the task? (binary)
    - Mean reward: How effectively does it complete? (continuous)
    - Std dev: Is this performance consistent? (reliability)
    
    Returns:
        Dict with per-task and aggregate results:
        - success rates (%)
        - mean rewards (continuous)
        - pass/fail indicators
        - stability metrics (mean ± std)
    """
    grading = {}
    
    # ========================================================================
    # STEP 1: Aggregate per-task statistics across all seeds
    # ========================================================================
    # For each task, collect data from all seeds' evaluations
    task_success_rates = {}     # Aggregated success rate per task (0.0-1.0)
    task_mean_rewards = {}      # Mean episode return per task (continuous)
    task_reward_stds = {}       # Std dev of returns per task (variability measure)
    
    # Loop over each task to extract and aggregate its statistics
    for task in tasks:
        task_successes = []     # All success indicators from all seeds
        task_rewards = []       # All episode returns from all episodes across all seeds
        
        # Collect data from each seed's evaluation
        for seed_result in result.per_seed_results:
            if task in seed_result.task_results:
                task_eval = seed_result.task_results[task]
                # task_eval contains returns and successes from this seed for this task
                task_successes.append(task_eval.success_rate)
                # Collect all episode returns for this task-seed combination
                # This allows us to compute mean/std of actual episode rewards
                task_rewards.extend(task_eval.returns.tolist())
        
        # Compute per-task success rate (average across all seeds)
        if task_successes:
            task_success_rates[task] = np.mean(task_successes)
        
        # Compute per-task reward statistics (mean and std across all episodes)
        # This is the core of stability metrics calculation
        if task_rewards:
            # Mean return: what's the expected episode reward for this task?
            task_mean_rewards[task] = np.mean(task_rewards)
            # Std dev: how much variation is there in episode returns?
            # High variance = some episodes much better than others (inconsistency)
            # Low variance = consistent performance across episodes
            task_reward_stds[task] = np.std(task_rewards)
    
    # ========================================================================
    # STEP 2: Compute aggregate stability metrics
    # ========================================================================
    # These metrics answer: how stable is the agent across different tasks?
    
    # Mean Reward Across Tasks (required metric #1)
    # Takes the mean from each task and computes mean across all tasks
    # Result: single value representing expected reward across all tasks
    if task_mean_rewards:
        # Aggregate: mean of the task means
        # Example: if reach gives 104.83, push gives 96.06, pick-place gives 109.16
        #          then mean_reward_across_tasks = (104.83 + 96.06 + 109.16) / 3 = 103.35
        mean_reward_across_tasks = np.mean(list(task_mean_rewards.values()))
        # Standard Deviation Across Tasks (required metric #2)
        # How much do the task means differ from each other?
        # Shows if agent specializes in some tasks more than others
        # Example: std([104.83, 96.06, 109.16]) = 5.45
        std_reward_across_tasks = np.std(list(task_mean_rewards.values()))
    else:
        # No reward data available (shouldn't happen in normal operation)
        mean_reward_across_tasks = 0.0
        std_reward_across_tasks = 0.0
    
    # Average Success Rate (for grading pass/fail)
    # What fraction of episodes succeeded on average?
    if task_success_rates:
        avg_success = np.mean(list(task_success_rates.values()))
        std_success = np.std(list(task_success_rates.values()))
    else:
        avg_success = 0.0
        std_success = 0.0
    
    # ========================================================================
    # STEP 3: Store all metrics in output dictionary
    # ========================================================================
    # Store the aggregate stability metrics
    grading["avg_success_rate"] = avg_success
    grading["std_success_rate"] = std_success
    grading["mean_reward_across_tasks"] = mean_reward_across_tasks  # Project requirement
    grading["std_reward_across_tasks"] = std_reward_across_tasks    # Project requirement
    
    # Store per-task metrics for detailed breakdown
    for task in tasks:
        if task in task_mean_rewards:
            grading[f"{task}_mean_reward"] = task_mean_rewards[task]
            grading[f"{task}_std_reward"] = task_reward_stds[task]
        if task in task_success_rates:
            grading[f"{task}_success"] = task_success_rates[task]
    
    # ========================================================================
    # STEP 4: Apply mode-specific grading criteria
    # ========================================================================
    
    if eval_mode == "mt1":
        # MT1 MODE: Single-task evaluation - need to pass individual task thresholds
        # Each task has its own threshold (easier tasks need higher success rate)
        thresholds = GRADING_THRESHOLDS["mt1"]
        all_pass = True
        
        # Check each task independently
        for task, threshold in thresholds.items():
            if task in task_success_rates:
                # Binary pass/fail for this task: is success_rate > threshold?
                passed = task_success_rates[task] > threshold
                grading[f"{task}_success"] = task_success_rates[task]
                grading[f"{task}_threshold"] = threshold
                grading[f"{task}_pass"] = passed
                if not passed:
                    all_pass = False
            else:
                # Task not evaluated - automatic fail
                grading[f"{task}_pass"] = False
                all_pass = False
        
        # Overall result: pass only if ALL tasks pass
        grading["all_pass"] = all_pass
        
    elif eval_mode == "mt3":
        # MT3 MODE: Multi-task learning with 3 tasks
        # Easier criterion than MT1 (40% vs 30-90%)
        # Reason: agent trains on all tasks, so seeing all during training
        threshold = GRADING_THRESHOLDS["mt3"]["average"]
        grading["threshold"] = threshold
        # Pass if average success > threshold (not all-or-nothing)
        grading["pass"] = avg_success > threshold
        
        # Also track per-task for reference
        for task, rate in task_success_rates.items():
            grading[f"{task}_success"] = rate
            
    elif eval_mode == "mt10":
        # MT10 MODE: Full Meta-World benchmark with 10 diverse tasks
        # Lowest criterion (30%) - most challenging benchmark
        # Reason: 10 diverse tasks, harder to generalize to all
        threshold = GRADING_THRESHOLDS["mt10"]["average"]
        grading["threshold"] = threshold
        # Pass if average success > threshold
        grading["pass"] = avg_success > threshold
        
        # Track per-task for reference
        for task, rate in task_success_rates.items():
            grading[f"{task}_success"] = rate
    
    return grading


def print_grading_summary(grading_results: List[Dict], eval_mode: str):
    """
    Print a formatted grading summary with all evaluation metrics.
    
    This function formats the grading dictionary into human-readable output,
    displaying both success rates and stability metrics as required by the project.
    
    Displays:
    - Grading criteria thresholds for the selected mode
    - Per-model results including pass/fail status
    - Stability metrics (Mean Reward ± Std Dev)
    - Per-task breakdown
    
    Format varies by eval_mode:
    - MT1: Individual task results (must pass all)
    - MT3/MT10: Average across tasks + detailed per-task breakdown
    """
    print(f"\n{'=' * 60}")
    print(f"GRADING SUMMARY ({eval_mode.upper()})")
    print(f"{'=' * 60}")
    
    if eval_mode == "mt1":
        # MT1 OUTPUT: Single-task evaluation results
        # Each task evaluated independently with its own threshold
        print("\nSingle-Task Evaluation Criteria:")
        print("  - reach-v3:      > 90%")
        print("  - push-v3:       > 30%")
        print("  - pick-place-v3: > 30%")
        print("-" * 60)
        
        for result in grading_results:
            model = result.get("model", "unknown")
            print(f"\nModel: {model}")
            
            # Display each task's results
            for task in ["reach-v3", "push-v3", "pick-place-v3"]:
                success = result.get(f"{task}_success", 0) * 100
                threshold = result.get(f"{task}_threshold", 0) * 100
                passed = result.get(f"{task}_pass", False)
                status = "✅ PASS" if passed else "❌ FAIL"
                
                # Stability metrics: per-task mean reward
                mean_reward = result.get(f"{task}_mean_reward", 0.0)
                std_reward = result.get(f"{task}_std_reward", 0.0)
                
                # Show success rate with pass/fail
                print(f"  {task}: {success:.1f}% (threshold: >{threshold:.0f}%) {status}")
                # Show stability metrics (why this task succeeded/failed)
                # Lower returns might indicate the agent found a policy but it's suboptimal
                # High variance in returns might indicate policy is unstable for this task
                print(f"    Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Overall status for this model
            all_pass = result.get("all_pass", False)
            print(f"\n  Overall: {'✅ ALL PASS' if all_pass else '❌ NOT ALL PASS'}")
            
    elif eval_mode in ["mt3", "mt10"]:
        # MT3/MT10 OUTPUT: Multi-task evaluation results
        # Display average success rate and stability metrics
        threshold = GRADING_THRESHOLDS[eval_mode]["average"] * 100
        print(f"\nMulti-Task Evaluation Criteria:")
        print(f"  - Average success rate > {threshold:.0f}%")
        print("-" * 60)
        
        for result in grading_results:
            model = result.get("model", "unknown")
            # Aggregate success rate across all tasks
            avg = result.get("avg_success_rate", 0) * 100
            std = result.get("std_success_rate", 0) * 100
            # Pass/fail status
            passed = result.get("pass", False)
            status = "✅ PASS" if passed else "❌ FAIL"
            
            print(f"\nModel: {model}")
            # Show success rates: binary indicator of task completion
            print(f"  Average Success Rate: {avg:.1f}% ± {std:.1f}%")
            print(f"  Threshold: >{threshold:.0f}%")
            print(f"  Status: {status}")
            
            # ================================================================
            # STABILITY METRICS: The new required project metrics
            # ================================================================
            # These show the QUALITY of performance, not just success/failure
            # Mean Reward: How well does the agent perform on average?
            # Std Dev: How consistent is performance across different tasks?
            # ================================================================
            mean_reward = result.get("mean_reward_across_tasks", 0.0)
            std_reward = result.get("std_reward_across_tasks", 0.0)
            print(f"\n  Stability Metrics:")
            # This is the metric required by project specification
            # "Mean Reward ± Std Dev" format shows:
            # - First number: expected cumulative reward per episode
            # - ± Second number: how much it varies across tasks
            print(f"    Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Per-task breakdown: why does the overall metric look this way?
            print(f"\n  Per-Task Breakdown (Success Rate | Mean Reward):")
            for key, value in result.items():
                # Find all per-task success metrics
                if key.endswith("_success") and key != "avg_success_rate":
                    task = key.replace("_success", "")
                    task_success = value * 100
                    # Get this task's mean and std reward
                    task_mean_reward = result.get(f"{task}_mean_reward", 0.0)
                    task_std_reward = result.get(f"{task}_std_reward", 0.0)
                    # Display: how successful was this task | how good were the trajectories?
                    print(f"    {task}: {task_success:.1f}% | {task_mean_reward:.2f} ± {task_std_reward:.2f}")
    
    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Final Evaluation for MTMH-SAC (MT1/MT3/MT10)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name of the training run to evaluate")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Base directory containing model checkpoints")
    parser.add_argument("--which", type=str, default="final", 
                        choices=["final", "best", "all"],
                        help="Which model(s) to evaluate")
    parser.add_argument("--eval_mode", type=str, default="mt3",
                        choices=["mt1", "mt3", "mt10"],
                        help="Evaluation mode: mt1 (single-task), mt3 (3-task), mt10 (10-task)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per task per seed")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of environment seeds for variance estimation")
    parser.add_argument("--seed_base", type=int, default=0,
                        help="Base seed for evaluation")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Tasks to evaluate (auto-set based on eval_mode if not specified)")
    parser.add_argument("--max_episode_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for model inference")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save CSV results")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for intervals")
    args = parser.parse_args()

    # Set tasks based on eval_mode if not explicitly provided
    if args.tasks is None:
        if args.eval_mode == "mt1":
            args.tasks = MT1_TASKS
        elif args.eval_mode == "mt3":
            args.tasks = MT3_TASKS
        elif args.eval_mode == "mt10":
            args.tasks = MT10_TASKS
        else:
            args.tasks = DEFAULT_TASKS

    # Resolve the run directory
    model_root = os.path.join(os.path.abspath(args.model_dir), args.run_name)
    if not os.path.isdir(model_root):
        raise FileNotFoundError(f"Model directory not found: {model_root}")

    # Locate candidate models
    models = find_models(model_root, args.which, args.tasks)
    print(f"\nFound {len(models)} model(s) to evaluate")
    print(f"Evaluation Mode: {args.eval_mode.upper()}")
    print(f"Tasks: {args.tasks}")

    # Load each model
    loaded: Dict[str, SAC] = {}
    for label, path in models.items():
        print(f"Loading: {label} -> {path}")
        loaded[label] = SAC.load(path, device=args.device)

    # Collect results
    all_results = []
    all_grading_results = []

    for label, model in loaded.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating model: {label}")
        print(f"{'=' * 60}")
        
        result = evaluate_model_multiseed(
            model=model,
            model_label=label,
            tasks=args.tasks,
            episodes_per_task=args.episodes,
            n_seeds=args.n_seeds,
            seed_base=args.seed_base,
            max_episode_steps=args.max_episode_steps,
        )
        
        # Print detailed report
        result.print_report(confidence=args.confidence)
        
        # Store for CSV export
        summary = result.summary_dict(confidence=args.confidence)
        summary["model"] = label
        all_results.append(summary)
        
        # Grading check
        grading_result = check_grading_criteria(result, args.eval_mode, args.tasks)
        all_grading_results.append({"model": label, **grading_result})

    # Print grading summary
    print_grading_summary(all_grading_results, args.eval_mode)

    # Optional CSV export
    if args.output_csv and all_results:
        # Flatten results for CSV
        rows = []
        for summary in all_results:
            model_name = summary.pop("model")
            for key, value in summary.items():
                rows.append({
                    "model": model_name,
                    "metric": key,
                    "value": value
                })
        
        # Add grading results
        for grading in all_grading_results:
            model_name = grading.pop("model")
            for key, value in grading.items():
                rows.append({
                    "model": model_name,
                    "metric": f"grading_{key}",
                    "value": value
                })
        
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved results to: {args.output_csv}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
