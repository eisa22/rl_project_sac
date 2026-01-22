# MTMH-SAC: Multi-Task Multi-Head Soft Actor-Critic for Meta-World

This repository contains the implementation of **MTMH-SAC** (Multi-Task Multi-Head SAC) for the Meta-World MT3 benchmark. The implementation supports three SAC variants with increasing complexity:

1. **SAC** - Standard Soft Actor-Critic with shared networks
2. **MT-SAC** - Multi-Task SAC with task conditioning (one-hot encoding)
3. **MHMT-SAC** - Multi-Head Multi-Task SAC with separate action heads per task

Additionally, **Curriculum Learning** can be enabled to progressively unlock harder tasks based on success rate thresholds.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pre-trained Models](#pre-trained-models)
- [Implementation Details](#implementation-details)

---

## Architecture Overview

### From SAC to MHMT-SAC

The implementation builds from the standard SAC algorithm to the full multi-head multi-task variant in three levels:

#### Level 1: SAC (Soft Actor-Critic)

Standard off-policy actor-critic algorithm with entropy regularization:

- **Actor**: Learns a stochastic policy π(a|s) that outputs mean and log_std of a Gaussian distribution
- **Critic**: Two Q-networks (Q1, Q2) to mitigate overestimation bias
- **Entropy Tuning**: Automatic temperature (α) adjustment to balance exploration vs. exploitation
- **Target Networks**: Soft updates for stable learning

**Code location**: `sac_agent_sb3/agent.py` → `SACAgentSB3` class

```
┌─────────────────────────────────────────────────────────────┐
│                    SAC Architecture                         │
├─────────────────────────────────────────────────────────────┤
│  State s ──→ [Actor Network] ──→ μ, log_σ ──→ Action a     │
│  (s, a) ──→ [Critic Network Q1] ──→ Q1(s,a)                │
│  (s, a) ──→ [Critic Network Q2] ──→ Q2(s,a)                │
│  α (temperature) ──→ Auto-tuned based on target entropy    │
└─────────────────────────────────────────────────────────────┘
```

#### Level 2: MT-SAC (Multi-Task SAC)

Extends SAC to handle multiple tasks by conditioning on task identity:

- **Task Conditioning**: One-hot task encoding appended to observations
- **Shared Networks**: Single actor and critic handle all tasks
- **Task Sampling**: Uniform or curriculum-based task selection during training

**Code location**: `sac_agent_sb3/curriculum.py` → `MetaWorldCurriculumEnv` class

```
┌─────────────────────────────────────────────────────────────┐
│                   MT-SAC Architecture                       │
├─────────────────────────────────────────────────────────────┤
│  Original obs (39 dims) + One-hot task ID (3 dims) = 42    │
│                                                             │
│  State s' = [s, task_one_hot] ──→ Shared Actor ──→ Action  │
│  State s' = [s, task_one_hot] ──→ Shared Critic ──→ Q(s,a) │
└─────────────────────────────────────────────────────────────┘
```

#### Level 3: MHMT-SAC (Multi-Head Multi-Task SAC)

The full implementation with task-specific action heads:

- **Shared Trunk**: Common feature extraction layers for all tasks
- **Multi-Head Actor**: Separate action output heads per task
- **Multi-Head Critic**: Separate value heads per task
- **Per-Task Temperature**: Individual entropy coefficients (α) per task
- **Q-Value Clipping**: Prevents critic divergence (common in multi-task RL)

**Code location**: `sac_agent_sb3/multiheadAgent.py` → `MHSACAgentSB3` class

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MHMT-SAC Architecture                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Observation ──→ [Shared Trunk: 3×400 units] ──→ Shared Features   │
│                           │                                         │
│         ┌─────────────────┼─────────────────┐                       │
│         ▼                 ▼                 ▼                       │
│   [Head: reach]     [Head: push]    [Head: pick-place]             │
│         │                 │                 │                       │
│         ▼                 ▼                 ▼                       │
│   μ₁, σ₁ ──→ a₁     μ₂, σ₂ ──→ a₂    μ₃, σ₃ ──→ a₃                │
│                                                                     │
│  Per-Task Temperature: α₁, α₂, α₃ (auto-tuned independently)       │
│                                                                     │
│  Task Routing: task_idx determines which head processes action     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Components in `multiheadAgent.py`:**

| Class | Purpose |
|-------|---------|
| `MultiHeadLinear` | Shared trunk + task-specific heads network |
| `MultiHeadActor` | Actor with per-task action distribution heads |
| `MultiHeadCritic` | Critic with per-task Q-value heads |
| `MultiTaskTemperature` | Per-task learnable entropy coefficients |
| `AsymSACPolicy` | Custom SB3 policy integrating multi-head architecture |
| `MHSACAgentSB3` | Main agent class wrapping everything |

### Curriculum Learning

Curriculum learning progressively unlocks harder tasks as the agent masters easier ones:

```
Task Sequence: reach-v3 → push-v3 → pick-place-v3
                  ↓            ↓           ↓
Thresholds:      60%         50%         40%
                  ↓            ↓           ↓
              (unlock       (unlock    (always
               push)      pick-place)  available)
```

**Implementation** (`sac_agent_sb3/curriculum.py`):

1. **CurriculumTracker**: Monitors success rates per task over a sliding window
2. **MetaWorldCurriculumEnv**: Gymnasium wrapper that:
   - Tracks which tasks are currently "active" (unlocked)
   - Samples tasks according to curriculum state
   - Appends one-hot task encoding to observations
   - Reports task completion to tracker

3. **Task Sampling with Curriculum**:
   - When `pick-place-v3` is unlocked: 80% pick-place, 20% other tasks (to focus on hardest)
   - Otherwise: Uniform sampling over unlocked tasks

4. **Unlock Logic**:
   ```python
   for i, task in enumerate(task_sequence):
       if success_rate(task) > thresholds[i]:
           unlock(task_sequence[i + 1])
   ```

---

## File Structure

```
.
├── README.md                      # This documentation
├── requirements.txt               # Python dependencies
├── train_mt3_curriculum_sb3.py    # Main training script
├── evaluate.py                    # Evaluation and grading script
├── models/                        # Pre-trained model checkpoints
│   ├── mt3_optimized_run2/        # Run 2: Uniform sampling + multi-head
│   ├── mt3_optimized_run2+/       # Run 2+: Curriculum + multi-head
│   ├── mt3_optimized_run3/        # Run 3: Uniform sampling + multi-head
│   └── mt3_optimized_run3+/       # Run 3+: Curriculum + multi-head
└── sac_agent_sb3/                 # Core implementation package
    ├── __init__.py                # Package exports
    ├── agent.py                   # Standard SAC agent (SACAgentSB3)
    ├── multiheadAgent.py          # Multi-head SAC agent (MHSACAgentSB3)
    ├── curriculum.py              # Curriculum learning and environment wrapper
    ├── evaluation.py              # Evaluation utilities (IQM, bootstrap CI)
    └── README.md                  # Package documentation
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- Conda or venv for environment management

### Setup

```bash
# Create a conda environment (recommended)
conda create -n metaworld python=3.10
conda activate metaworld

# Install PyTorch (adjust for CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import metaworld; import stable_baselines3; print('Installation successful!')"
```

---

## Training

### Basic Training (MHMT-SAC without Curriculum)

Train a multi-head multi-task SAC agent with uniform task sampling:

```bash
python train_mt3_curriculum_sb3.py \
    --run_name my_mt3_run \
    --enable_multihead \
    --total_steps 3000000 \
    --batch_size 1280 \
    --num_envs 10
```

### Training with Curriculum Learning

Enable curriculum learning to progressively unlock tasks:

```bash
python train_mt3_curriculum_sb3.py \
    --run_name my_curriculum_run \
    --enable_multihead \
    --enable_curriculum \
    --curriculum_thresholds 0.6 0.5 0.4 \
    --total_steps 3000000
```

### Training Standard SAC (Shared Network)

For comparison, train without multi-head architecture:

```bash
python train_mt3_curriculum_sb3.py \
    --run_name my_sac_run \
    --total_steps 3000000
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--run_name` | Required | Name for this training run (used for saving) |
| `--enable_multihead` | False | Use multi-head architecture (MHMT-SAC) |
| `--enable_curriculum` | False | Enable curriculum learning |
| `--total_steps` | 3000000 | Total environment steps for training |
| `--batch_size` | 256 | Batch size for SAC updates |
| `--num_envs` | 10 | Number of parallel environments |
| `--learning_rate` | 3e-4 | Learning rate for all networks |
| `--buffer_size` | 3000000 | Replay buffer size |
| `--curriculum_thresholds` | 0.6 0.5 0.4 | Success rate thresholds to unlock tasks |
| `--model_dir` | ./models | Directory to save checkpoints |
| `--wandb_mode` | online | WandB logging mode (online/offline/disabled) |

---

## Evaluation

### Evaluate Pre-trained Models

```bash
# Evaluate run2 (uniform sampling + multi-head)
python evaluate.py --run_name mt3_optimized_run2 --which final --episodes 100 --n_seeds 5

# Evaluate run2+ (curriculum + multi-head)
python evaluate.py --run_name mt3_optimized_run2+ --which final --episodes 100 --n_seeds 5

# Evaluate run3 (uniform sampling + multi-head)
python evaluate.py --run_name mt3_optimized_run3 --which final --episodes 100 --n_seeds 5

# Evaluate run3+ (curriculum + multi-head)
python evaluate.py --run_name mt3_optimized_run3+ --which final --episodes 100 --n_seeds 5
```

### Key Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--run_name` | Required | Name of the training run to evaluate |
| `--model_dir` | ./models | Base directory containing model checkpoints |
| `--which` | final | Which model to evaluate: `final`, `best`, or `all` |
| `--eval_mode` | mt3 | Evaluation mode: `mt1`, `mt3`, or `mt10` |
| `--episodes` | 100 | Episodes per task per seed |
| `--n_seeds` | 5 | Number of random seeds for statistical robustness |

### Grading Criteria

| Benchmark | Criterion | Threshold |
|-----------|-----------|-----------|
| MT1 | reach-v3 | > 90% |
| MT1 | push-v3 | > 30% |
| MT1 | pick-place-v3 | > 30% |
| MT3 | Average success rate | > 40% |
| MT10 | Average success rate | > 30% |

---

## Pre-trained Models

### Included Runs

| Run Name | Configuration | Description |
|----------|---------------|-------------|
| `mt3_optimized_run2` | MHMT-SAC + Uniform | Multi-head with uniform task sampling |
| `mt3_optimized_run2+` | MHMT-SAC + Curriculum | Multi-head with curriculum learning |
| `mt3_optimized_run3` | MHMT-SAC + Uniform | Multi-head with uniform task sampling (different seed) |
| `mt3_optimized_run3+` | MHMT-SAC + Curriculum | Multi-head with curriculum learning (different seed) |

### Model Files

Each run directory contains:

- `final_sac_mt3.zip` - Final model checkpoint (required for evaluation)
- `best_*/best_model.zip` - Best models per task during training
- `eval_*/evaluations.npz` - Evaluation logs

---

## Implementation Details

### Network Architecture

| Component | Architecture |
|-----------|--------------|
| Shared Trunk | 3 × 400 units with ReLU |
| Actor Head | 1 × 128 units per task |
| Critic Head | 1 × 256 units per task |
| Action Space | 4-dimensional continuous |
| Observation Space | 39 (raw) + 3 (task one-hot) = 42 |

### Asymmetric Actor-Critic Capacity (Larger Critic)

We use asymmetric networks where the critic is larger than the actor. Concretely,
the default multi-head configuration uses a smaller actor trunk and head
(`--actor_hidden 256 256`, `--actor_head_hidden 128`) and a deeper, wider critic
(`--critic_hidden 512 512 512`, `--critic_head_hidden 256`).

Rationale:
- In off-policy actor-critic methods, policy updates depend on accurate Q-value
  estimates. Under multi-task training, the value function must model a broader
  distribution of states and rewards, which can make the critic the bottleneck.
- Empirically, giving the critic more capacity is a common stabilization
  heuristic and can improve sample efficiency for multi-task settings.

References:
- Haarnoja et al., 2018. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL."
- Fujimoto et al., 2018. "Addressing Function Approximation Error in Actor-Critic Methods."
- Yu et al., 2019. "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta RL."

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 3e-4 | Standard for Adam with SAC |
| Batch size | 256–1280 | Larger batches for multi-task |
| Buffer size | 3M | Large enough for multi-task replay |
| γ (discount) | 0.99 | Standard for episodic tasks |
| τ (soft update) | 0.005 | Slow target updates for stability |
| Entropy coef | auto | Automatic entropy tuning |
