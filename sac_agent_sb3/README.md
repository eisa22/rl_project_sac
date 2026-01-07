# SAC Agent (Stable-Baselines3)

Multi-task SAC implementation for Meta-World using Stable-Baselines3, supporting curriculum learning and asymmetric actor-critic architectures.

## Overview

This package provides a lightweight wrapper around SB3's SAC algorithm with:
- **Asymmetric networks**: Larger critic networks (3×512) than actor networks (2×256) for improved value estimation
- **Curriculum learning**: Progressive task unlocking based on success rates
- **Multi-task support**: MT3 training with task one-hot encoding
- **WandB integration**: Comprehensive logging including alpha, Q-values, per-task metrics
- **Parallel environments**: SubprocVecEnv for efficient data collection

## Architecture

### AsymSACPolicy
Custom policy extending SB3's `SACPolicy` with separate actor/critic hidden sizes:
- **Actor**: 2 layers × 256 units (default)
- **Critic**: 3 layers × 512 units (default)
- **Motivation**: Larger critics improve Q-value estimation stability; smaller actors prevent overfitting

### Key Components
- `agent.py`: `SACAgentSB3` wrapper and `AsymSACPolicy` implementation
- `curriculum.py`: `MetaWorldCurriculumEnv` wrapper with task progression logic
- Training scripts: MT1 (`train_mt1_sb3.py`) and MT3 curriculum (`train_mt3_curriculum_sb3.py`)

## Default Parameters

Based on stable SAC hyperparameters for continuous control:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 3e-4 | Standard for Adam with SAC |
| `buffer_size` | 3M | Large enough for multi-task replay |
| `batch_size` | 256 | Balance sample efficiency and GPU utilization |
| `gamma` | 0.99 | Standard discount for episodic tasks |
| `tau` | 0.005 | Slow target network updates for stability |
| `learning_starts` | 5000 | Warm-up before training |
| `train_freq` | 1 | Update after every env step |
| `gradient_steps` | 1 | One gradient step per env step |
| `ent_coef` | auto | Automatic entropy tuning |
| `target_entropy` | auto | `-dim(A)` by default |
| `log_std_init` | -3.0 | Conservative exploration init |

### Curriculum Defaults
- **Task sequence**: reach → push → pick-place
- **Thresholds**: [0.6, 0.4, 0.0] (success rates to unlock next task)
- **Min episodes**: 20 per task before unlock check
- **Window size**: Last 50 episodes for success rate

## Installation

```bash
# Inside rl_project_sac/
pip install stable-baselines3[extra] metaworld gymnasium wandb torch
```

## Usage

### MT3 Curriculum Training

```bash
python rl_project_sac/train_mt3_curriculum_sb3.py \
    --run_name my_mt3_run \
    --total_steps 2000000 \
    --num_envs 8 \
    --curriculum_thresholds 0.6 0.4 0.0
```

**Key flags:**
- `--run_name`: WandB run name and model save dir
- `--total_steps`: Total environment steps (default 1.5M)
- `--num_envs`: Parallel envs (default 8; uses SubprocVecEnv if >1)
- `--curriculum_thresholds`: Success rates [reach, push, pick] to unlock next task
- `--ent_coef`: Entropy coefficient (e.g., `auto`, `auto_0.2`)
- `--target_entropy`: Target entropy (e.g., `auto`, `-2`)
- `--checkpoint_interval`: Steps between checkpoints (default 100k)
- `--log_interval`: Steps between metric logs (default 1k)
- `--max_episode_steps`: Episode length (default 150)

### MT1 Single-Task Training

```bash
python rl_project_sac/train_mt1_sb3.py \
    --run_name my_mt1_run \
    --task reach-v3 \
    --total_steps 1000000 \
    --num_envs 4
```

### Playing Back a Trained Model

```bash
python rl_project_sac/play_metaworld_sb3.py \
    --model_path models_sac_sb3/my_mt3_run/final_sac_mt3.zip \
    --task reach-v3 \
    --num_episodes 10
```

## Monitoring

### WandB Metrics
- **Training**: `train/episode_reward`, `train/success`, `train/alpha`, `train/q1_mean`, `train/q2_mean`
- **Per-task**: `train/task/{task}/reward_mean`, `train/task/{task}/success`
- **Curriculum**: `curriculum/active_tasks`, `curriculum/last_task`
- **Evaluation**: Per-task EvalCallbacks log to disk at checkpoint intervals

### Checkpoints
Saved every 100k steps (configurable) to `models_sac_sb3/{run_name}/checkpoints/`.

## Tuning Tips

### If alpha collapses too fast:
```bash
--ent_coef auto_0.2 --target_entropy -2
```
(Less negative target entropy or higher initial ent_coef keeps exploration higher)

### If tasks don't unlock:
- Lower thresholds: `--curriculum_thresholds 0.5 0.3 0.0`
- Check WandB `train/task/{task}/success` to see current rates

### If training is slow:
- Increase `--num_envs` (scales linearly with CPU cores)
- Reduce `--checkpoint_interval` and `--log_interval` overhead

### If you want symmetric networks:
Edit `actor_hidden` and `critic_hidden` in `sac_agent_sb3/agent.py` to match sizes.

## Design Rationale

### Why Asymmetric Networks?
Standard SAC often uses identical actor/critic architectures. We hypothesize that:
- Critics benefit from extra capacity for accurate Q-value bootstrapping
- Actors can remain compact to avoid overfitting sparse rewards
- This mirrors findings in discrete action spaces (e.g., DQN with larger networks)

### Why Curriculum?
Meta-World pick-place is challenging with sparse success signals. Progressive unlocking:
- Stabilizes early training on easier tasks (reach)
- Transfers representations to harder tasks (push, pick-place)
- Reduces catastrophic forgetting via continued multi-task replay

### Why Task One-Hot?
Appending a one-hot task ID to observations:
- Simple, no architectural changes required
- Allows single policy to condition on task context
- Compatible with standard SB3 policies

## Files

```
sac_agent_sb3/
├── README.md              # This file
├── __init__.py
├── agent.py               # SACAgentSB3 + AsymSACPolicy
└── curriculum.py          # MetaWorldCurriculumEnv + CurriculumTracker
```

## References

- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [Meta-World Benchmark](https://meta-world.github.io/)
- [Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1812.05905)

---

**Author**: Robot Learning Team  
**Date**: January 2026  
**License**: MIT
