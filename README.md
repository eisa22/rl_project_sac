# Multi-Task Multi-Head SAC (MTMH-SAC)

Efficient multi-task reinforcement learning on Meta-World robotics benchmarks using Soft Actor-Critic with task-specific heads, adaptive reward scaling, and periodic resets.

## Architecture

**Core Algorithm**: Multi-Task Multi-Head SAC
- Shared trunk encoder with task-specific actor/critic heads
- Per-task entropy coefficients (α) for adaptive exploration
- Per-task replay buffers for balanced sampling
- GPU-optimized with mixed precision training (PyTorch AMP)

**Key Features**:
- **Adaptive Reward Scaling (ARS)**: Equalizes reward magnitudes across tasks using buffer statistics
- **Periodic Resets (PR)**: Resets network weights every N steps to prevent plasticity loss
- **Batch Inference**: Vectorized action selection across parallel environments
- **Vectorized Training Loop**: Efficient data collection with minimal CPU-GPU sync

## Benchmarks

**MT3**: 3 Meta-World tasks (reach, push, pick-place)
- Default: 20M steps, 16 envs/task (48 parallel), batch 2048
- Training time: ~24-36h on A40 GPU

**MT10**: 10 Meta-World tasks
- Default: 20M steps, 8 envs/task (80 parallel), batch 2048
- Training time: ~48-72h on A40 GPU

## Installation

```bash
# Clone repository
git clone <repo-url>
cd rl_project_sac

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Local Training

```bash
# MT3 (3 tasks)
python train_mt3_mtmh.py \
    --run_name mt3_test \
    --total_steps 3000000 \
    --batch_size 2048 \
    --num_envs_per_task 16 \
    --ars_enable \
    --reset_enable

# MT10 (10 tasks)
python train_mt10_mtmh.py \
    --run_name mt10_test \
    --total_steps 20000000 \
    --batch_size 2048 \
    --num_envs_per_task 8 \
    --ars_enable \
    --reset_enable
```

## Cluster Training

**Prerequisites**: SLURM cluster with Singularity/Apptainer support

```bash
# Build Docker image and convert to Singularity
cd docker/cluster
./build_docker.sh
./convert_to_singularity.sh

# Deploy to cluster and submit job
./deploy_to_cluster.sh [seed]
```

See [CLUSTER_SETUP.md](CLUSTER_SETUP.md) for detailed cluster configuration.

## Key Hyperparameters

| Parameter | MT3 | MT10 | Description |
|-----------|-----|------|-------------|
| `--total_steps` | 20M | 20M | Total environment steps |
| `--batch_size` | 2048 | 2048 | SAC training batch size |
| `--num_envs_per_task` | 16 | 8 | Parallel envs per task |
| `--learning_starts` | 30k | 40k | Warmup steps before learning |
| `--update_every` | 1 | 1 | SAC update frequency (UTD 1:1) |
| `--ars_enable` | ✓ | ✓ | Adaptive reward scaling |
| `--ars_update_freq` | 50k | 50k | ARS recomputation interval |
| `--pick_place_base_scale` | 100 | 1 | Initial boost for hard tasks |
| `--reset_enable` | ✓ | ✓ | Periodic network resets |
| `--reset_every` | 5M | 5M | Steps between resets |

## Monitoring

Training logs to [Weights & Biases](https://wandb.ai):
- Per-task success rates, rewards, episode lengths
- Alpha values (entropy coefficients)
- ARS scales (reward multipliers)
- Actor/Critic losses

## Implementation Details

**Algorithm**: Soft Actor-Critic (SAC) with:
- Twin Q-networks (clipped double Q-learning)
- Automatic entropy tuning per task
- Polyak-averaged target networks

**ARS (Adaptive Reward Scaling)**:
- Formula: `c_i = max(r̄₁, ..., r̄ₙ) / r̄ᵢ` (equalizer rule)
- Applied at data collection time (transparent to critic)
- Updates every 50k steps based on buffer statistics

**PR (Periodic Resets)**:
- Resets actor, critic, target networks, and alpha
- Retains replay buffer for continued sample efficiency
- Prevents loss of plasticity in long training runs

**GPU Optimizations**:
- Mixed precision training (float16 forward, float32 gradients)
- cuDNN benchmarking for faster convolutions
- Batch action inference across all environments
- Vectorized environment stepping

## File Structure

```
.
├── mtmh_sac.py                 # MTMH-SAC agent implementation
├── train_mt3_mtmh.py           # MT3 training script
├── train_mt10_mtmh.py          # MT10 training script
├── deploy_to_cluster.sh        # Cluster deployment orchestration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
├── docker/cluster/             # SLURM batch scripts
│   ├── train_mt3_mtmh.sh       # MT3 cluster job
│   ├── train_mt10_mtmh.sh      # MT10 cluster job
│   └── ...                     # Build/deployment utilities
└── CLUSTER_SETUP.md            # Cluster setup guide
```

## Citation

Based on the MTRL winner architecture with enhancements:
- Adaptive Reward Scaling for multi-task balance
- Periodic Resets for maintaining plasticity
- GPU optimizations for 5-8x training speedup

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)
- Meta-World 2.0+
- Weights & Biases (for logging)

See [requirements.txt](requirements.txt) for full dependencies.
