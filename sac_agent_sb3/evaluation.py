"""
Evaluation utilities for MTMH-SAC following Meta-World benchmark protocols.

This module provides:
1. Online evaluation during training (per-task success rates)
2. Final evaluation with proper statistical measures (IQM, confidence intervals)
3. Multi-seed aggregation for robust performance estimates

Scientific Context:
- Success Rate is the primary metric for Meta-World benchmarks
- IQM (Interquartile Mean) is more robust than arithmetic mean for RL
- 95% confidence intervals via bootstrap for reliable comparisons
- Per-task breakdown helps identify which tasks are learned/forgotten

References:
- Meta-World: https://arxiv.org/abs/1910.10897
- Statistical Best Practices: https://arxiv.org/abs/2108.13264 (Agarwal et al.)
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
from scipy import stats


# =============================================================================
# Statistical Utilities
# =============================================================================

def interquartile_mean(data: np.ndarray) -> float:
    """
    Compute the Interquartile Mean (IQM) of data.
    
    IQM is more robust to outliers than arithmetic mean and is recommended
    for RL benchmarks by Agarwal et al. (2021).
    
    Args:
        data: 1D array of values
        
    Returns:
        IQM value
    """
    if len(data) == 0:
        return 0.0
    data = np.asarray(data).flatten()
    q25, q75 = np.percentile(data, [25, 75])
    mask = (data >= q25) & (data <= q75)
    return float(np.mean(data[mask])) if mask.any() else float(np.mean(data))


def bootstrap_ci(data: np.ndarray, 
                 statistic: Callable = np.mean,
                 confidence: float = 0.95,
                 n_bootstrap: int = 10000,
                 seed: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: 1D array of values
        statistic: Function to compute (e.g., np.mean, interquartile_mean)
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data).flatten()
    n = len(data)
    
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        bootstrap_stats.append(statistic(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentile CI
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_stats, 100 * alpha)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha))
    point_estimate = statistic(data)
    
    return float(point_estimate), float(lower), float(upper)


def stratified_bootstrap_ci(per_task_data: Dict[str, np.ndarray],
                            statistic: Callable = np.mean,
                            confidence: float = 0.95,
                            n_bootstrap: int = 10000,
                            seed: int = 42) -> Tuple[float, float, float]:
    """
    Compute stratified bootstrap CI for multi-task metrics.
    
    This bootstraps within each task then aggregates, which is more
    appropriate for multi-task evaluation.
    
    Args:
        per_task_data: Dict mapping task names to arrays of metrics
        statistic: Function to compute per-task (e.g., np.mean)
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        Tuple of (aggregate_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    task_names = list(per_task_data.keys())
    
    if not task_names:
        return 0.0, 0.0, 0.0
    
    # Bootstrap: resample within each task, then compute task means, then aggregate
    bootstrap_aggregates = []
    for _ in range(n_bootstrap):
        task_stats = []
        for task in task_names:
            data = np.asarray(per_task_data[task]).flatten()
            if len(data) == 0:
                continue
            sample = data[rng.randint(0, len(data), size=len(data))]
            task_stats.append(statistic(sample))
        if task_stats:
            bootstrap_aggregates.append(np.mean(task_stats))
    
    bootstrap_aggregates = np.array(bootstrap_aggregates)
    
    # Point estimate: mean of per-task means
    point_estimate = np.mean([
        statistic(np.asarray(per_task_data[task]).flatten()) 
        for task in task_names 
        if len(per_task_data[task]) > 0
    ])
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_aggregates, 100 * alpha)
    upper = np.percentile(bootstrap_aggregates, 100 * (1 - alpha))
    
    return float(point_estimate), float(lower), float(upper)


# =============================================================================
# Evaluation Data Structures
# =============================================================================

@dataclass
class TaskEvalResult:
    """Evaluation results for a single task."""
    task_name: str
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    successes: np.ndarray = field(default_factory=lambda: np.array([]))
    lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @property
    def success_rate(self) -> float:
        """Mean success rate."""
        return float(np.mean(self.successes)) if len(self.successes) > 0 else 0.0
    
    @property
    def mean_return(self) -> float:
        """Mean episodic return."""
        return float(np.mean(self.returns)) if len(self.returns) > 0 else 0.0
    
    @property
    def mean_length(self) -> float:
        """Mean episode length."""
        return float(np.mean(self.lengths)) if len(self.lengths) > 0 else 0.0
    
    def success_rate_with_ci(self, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Success rate with bootstrap confidence interval."""
        return bootstrap_ci(self.successes, np.mean, confidence)
    
    def return_with_ci(self, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Return with bootstrap confidence interval."""
        return bootstrap_ci(self.returns, np.mean, confidence)


@dataclass
class MultiTaskEvalResult:
    """Aggregated evaluation results across tasks."""
    task_results: Dict[str, TaskEvalResult] = field(default_factory=dict)
    seed: Optional[int] = None
    
    @property
    def task_names(self) -> List[str]:
        return list(self.task_results.keys())
    
    @property
    def mean_success_rate(self) -> float:
        """Mean success rate across all tasks."""
        rates = [r.success_rate for r in self.task_results.values()]
        return float(np.mean(rates)) if rates else 0.0
    
    @property
    def iqm_success_rate(self) -> float:
        """IQM success rate across tasks."""
        rates = [r.success_rate for r in self.task_results.values()]
        return interquartile_mean(np.array(rates)) if rates else 0.0
    
    @property
    def mean_return(self) -> float:
        """Mean return across all tasks."""
        returns = [r.mean_return for r in self.task_results.values()]
        return float(np.mean(returns)) if returns else 0.0
    
    def get_per_task_success_dict(self) -> Dict[str, float]:
        """Get success rates per task as dict."""
        return {name: r.success_rate for name, r in self.task_results.items()}
    
    def success_rate_with_stratified_ci(self, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Mean success rate with stratified bootstrap CI."""
        per_task_data = {
            name: r.successes for name, r in self.task_results.items()
        }
        return stratified_bootstrap_ci(per_task_data, np.mean, confidence)


@dataclass
class MultiSeedEvalResult:
    """Aggregated evaluation results across multiple seeds."""
    per_seed_results: List[MultiTaskEvalResult] = field(default_factory=list)
    
    @property
    def n_seeds(self) -> int:
        return len(self.per_seed_results)
    
    @property
    def task_names(self) -> List[str]:
        if self.per_seed_results:
            return self.per_seed_results[0].task_names
        return []
    
    def get_per_task_success_rates(self) -> Dict[str, np.ndarray]:
        """Get success rates per task across seeds."""
        result = defaultdict(list)
        for seed_result in self.per_seed_results:
            for task_name, task_result in seed_result.task_results.items():
                result[task_name].append(task_result.success_rate)
        return {k: np.array(v) for k, v in result.items()}
    
    def get_aggregate_success_rates(self) -> np.ndarray:
        """Get aggregate (mean over tasks) success rate per seed."""
        return np.array([r.mean_success_rate for r in self.per_seed_results])
    
    def mean_success_rate_with_ci(self, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Mean success rate across seeds with CI."""
        rates = self.get_aggregate_success_rates()
        return bootstrap_ci(rates, np.mean, confidence)
    
    def iqm_success_rate_with_ci(self, confidence: float = 0.95) -> Tuple[float, float, float]:
        """IQM success rate across seeds with CI."""
        rates = self.get_aggregate_success_rates()
        return bootstrap_ci(rates, interquartile_mean, confidence)
    
    def per_task_success_with_ci(self, confidence: float = 0.95) -> Dict[str, Tuple[float, float, float]]:
        """Per-task success rates with CI across seeds."""
        per_task = self.get_per_task_success_rates()
        return {
            task: bootstrap_ci(rates, np.mean, confidence)
            for task, rates in per_task.items()
        }
    
    def summary_dict(self, confidence: float = 0.95) -> Dict[str, Any]:
        """Generate summary dictionary for logging/reporting."""
        mean_sr, mean_lower, mean_upper = self.mean_success_rate_with_ci(confidence)
        iqm_sr, iqm_lower, iqm_upper = self.iqm_success_rate_with_ci(confidence)
        per_task = self.per_task_success_with_ci(confidence)
        
        summary = {
            "n_seeds": self.n_seeds,
            "mean_success_rate": mean_sr,
            "mean_success_rate_ci_lower": mean_lower,
            "mean_success_rate_ci_upper": mean_upper,
            "iqm_success_rate": iqm_sr,
            "iqm_success_rate_ci_lower": iqm_lower,
            "iqm_success_rate_ci_upper": iqm_upper,
        }
        
        for task_name, (rate, lower, upper) in per_task.items():
            task_key = task_name.replace("-", "_")
            summary[f"{task_key}_success_rate"] = rate
            summary[f"{task_key}_success_rate_ci_lower"] = lower
            summary[f"{task_key}_success_rate_ci_upper"] = upper
            
        return summary
    
    def print_report(self, confidence: float = 0.95):
        """Print formatted evaluation report."""
        print("\n" + "=" * 60)
        print("MTMH-SAC Evaluation Report")
        print("=" * 60)
        print(f"Number of seeds: {self.n_seeds}")
        print(f"Confidence level: {confidence * 100:.0f}%")
        print("-" * 60)
        
        mean_sr, mean_lower, mean_upper = self.mean_success_rate_with_ci(confidence)
        iqm_sr, iqm_lower, iqm_upper = self.iqm_success_rate_with_ci(confidence)
        
        print(f"\nAggregate Success Rate:")
        print(f"  Mean: {mean_sr * 100:.2f}% [{mean_lower * 100:.2f}%, {mean_upper * 100:.2f}%]")
        print(f"  IQM:  {iqm_sr * 100:.2f}% [{iqm_lower * 100:.2f}%, {iqm_upper * 100:.2f}%]")
        
        print(f"\nPer-Task Success Rates:")
        per_task = self.per_task_success_with_ci(confidence)
        for task_name in sorted(per_task.keys()):
            rate, lower, upper = per_task[task_name]
            print(f"  {task_name}: {rate * 100:.2f}% [{lower * 100:.2f}%, {upper * 100:.2f}%]")
        
        print("=" * 60 + "\n")


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_single_task(model, 
                         env,
                         task_name: str,
                         n_episodes: int = 10,
                         deterministic: bool = True,
                         max_steps: int = 500) -> TaskEvalResult:
    """
    Evaluate model on a single task for multiple episodes.
    
    Args:
        model: Trained model with .predict() method
        env: Environment for the task
        task_name: Name of the task
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        max_steps: Maximum steps per episode
        
    Returns:
        TaskEvalResult with returns, successes, and lengths
    """
    returns = []
    successes = []
    lengths = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        done = False
        episode_return = 0.0
        episode_success = 0
        episode_length = 0
        
        while not done and episode_length < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                
            episode_return += float(reward)
            episode_length += 1
            
            # Check success
            if isinstance(info, dict):
                success = info.get("success", info.get("is_success", False))
            elif isinstance(info, (list, tuple)) and len(info) > 0:
                success = info[0].get("success", info[0].get("is_success", False))
            else:
                success = False
            episode_success = max(episode_success, int(bool(success)))
        
        returns.append(episode_return)
        successes.append(episode_success)
        lengths.append(episode_length)
    
    return TaskEvalResult(
        task_name=task_name,
        returns=np.array(returns),
        successes=np.array(successes),
        lengths=np.array(lengths)
    )


def evaluate_multi_task(model,
                        env_factory: Callable,
                        task_names: List[str],
                        n_episodes_per_task: int = 10,
                        deterministic: bool = True,
                        seed: int = 42) -> MultiTaskEvalResult:
    """
    Evaluate model on multiple tasks.
    
    Args:
        model: Trained model
        env_factory: Function(task_name, seed) -> env
        task_names: List of task names to evaluate
        n_episodes_per_task: Episodes per task
        deterministic: Use deterministic policy
        seed: Random seed for environment
        
    Returns:
        MultiTaskEvalResult with per-task results
    """
    task_results = {}
    
    for task_name in task_names:
        env = env_factory(task_name, seed)
        try:
            result = evaluate_single_task(
                model, env, task_name, n_episodes_per_task, deterministic
            )
            task_results[task_name] = result
        finally:
            env.close()
    
    return MultiTaskEvalResult(task_results=task_results, seed=seed)


def evaluate_multi_seed(model_loader: Callable,
                        env_factory: Callable,
                        task_names: List[str],
                        seeds: List[int],
                        n_episodes_per_task: int = 10,
                        deterministic: bool = True) -> MultiSeedEvalResult:
    """
    Evaluate model across multiple seeds.
    
    Args:
        model_loader: Function() -> model (loads/creates model)
        env_factory: Function(task_name, seed) -> env
        task_names: List of task names
        seeds: List of random seeds
        n_episodes_per_task: Episodes per task per seed
        deterministic: Use deterministic policy
        
    Returns:
        MultiSeedEvalResult with per-seed results
    """
    model = model_loader()
    per_seed_results = []
    
    for seed in seeds:
        result = evaluate_multi_task(
            model, env_factory, task_names, n_episodes_per_task, deterministic, seed
        )
        per_seed_results.append(result)
    
    return MultiSeedEvalResult(per_seed_results=per_seed_results)


# =============================================================================
# Callback for Online Evaluation During Training
# =============================================================================

class OnlineEvaluationTracker:
    """
    Track evaluation metrics during training for logging.
    
    Usage:
        tracker = OnlineEvaluationTracker(task_names, window_size=50)
        
        # In training loop:
        tracker.add_episode("reach-v3", return=5.0, success=1, length=100)
        metrics = tracker.get_metrics()
    """
    
    def __init__(self, task_names: List[str], window_size: int = 50):
        self.task_names = task_names
        self.window_size = window_size
        
        # Circular buffers for each task
        self.returns = {name: [] for name in task_names}
        self.successes = {name: [] for name in task_names}
        self.lengths = {name: [] for name in task_names}
        
    def add_episode(self, task_name: str, 
                    episode_return: float = 0.0,
                    success: int = 0,
                    length: int = 0):
        """Add episode result for a task."""
        if task_name not in self.task_names:
            return
            
        self.returns[task_name].append(episode_return)
        self.successes[task_name].append(success)
        self.lengths[task_name].append(length)
        
        # Keep only window_size most recent
        if len(self.returns[task_name]) > self.window_size:
            self.returns[task_name] = self.returns[task_name][-self.window_size:]
            self.successes[task_name] = self.successes[task_name][-self.window_size:]
            self.lengths[task_name] = self.lengths[task_name][-self.window_size:]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics for all tasks."""
        metrics = {}
        
        all_success_rates = []
        all_returns = []
        
        for task_name in self.task_names:
            task_key = task_name.replace("-", "_")
            
            if self.successes[task_name]:
                sr = float(np.mean(self.successes[task_name]))
                metrics[f"eval/{task_key}/success_rate"] = sr
                all_success_rates.append(sr)
                
            if self.returns[task_name]:
                ret = float(np.mean(self.returns[task_name]))
                metrics[f"eval/{task_key}/return"] = ret
                all_returns.append(ret)
                
            if self.lengths[task_name]:
                metrics[f"eval/{task_key}/length"] = float(np.mean(self.lengths[task_name]))
        
        # Aggregate metrics
        if all_success_rates:
            metrics["eval/mean_success_rate"] = float(np.mean(all_success_rates))
            metrics["eval/iqm_success_rate"] = interquartile_mean(np.array(all_success_rates))
            
        if all_returns:
            metrics["eval/mean_return"] = float(np.mean(all_returns))
            
        return metrics
    
    def reset(self):
        """Clear all stored data."""
        for task_name in self.task_names:
            self.returns[task_name].clear()
            self.successes[task_name].clear()
            self.lengths[task_name].clear()
