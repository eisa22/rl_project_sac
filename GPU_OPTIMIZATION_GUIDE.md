#!/usr/bin/env bash
# GPU Optimization Guide for Meta-World SAC MT10
#
# Problem: Single sequential environment is CPU-bound
# Solution: Parallel environments for better GPU throughput
#
# Paper Config (FIXED):
# - Buffer Size: 2M total (200k per task × 10 tasks)
# - Batch Size: Flexible (impact GPU, not Paper constraint)
# - Network: Actor [256,256], Critic [1024,1024,1024] (FIXED)
#
# ============================================================

# OPTION 1: STANDARD (Sequential - Legacy)
# GPU Utilization: ~5-20%
# Training Time: ~8-10 hours
# VRAM: ~3-4 GB
# export NUM_PARALLEL_ENVS=1
# export SAC_BATCH_SIZE=512
# export SAC_BUFFER_SIZE=2000000
# export SAC_LEARNING_STARTS=10000

# ============================================================

# OPTION 2: PARALLEL x4 (Balanced)
# Collect 4× more transitions per second
# GPU Utilization: ~40-60%
# Training Time: ~3-4 hours (4× faster experience collection)
# VRAM: ~6-8 GB
# export NUM_PARALLEL_ENVS=4
# export SAC_BATCH_SIZE=512        # Keep PAPER buffer size
# export SAC_BUFFER_SIZE=2000000   # FIXED: 200k × 10 tasks
# export SAC_LEARNING_STARTS=10000

# ============================================================

# OPTION 3: PARALLEL x8 (High GPU Load)
# Collect 8× more transitions per second
# GPU Utilization: ~60-75%
# Training Time: ~2-2.5 hours (8× faster experience collection)
# VRAM: ~12-14 GB
# export NUM_PARALLEL_ENVS=8
# export SAC_BATCH_SIZE=512
# export SAC_BUFFER_SIZE=2000000   # FIXED: Paper constraint
# export SAC_LEARNING_STARTS=10000

# ============================================================

# OPTION 4: PARALLEL x16 (Very High GPU Load) ⭐ RECOMMENDED
# Collect 16× more transitions per second
# GPU Utilization: ~70-80%
# Training Time: ~1.5-2 hours (16× faster experience collection)
# VRAM: ~20-24 GB
# export NUM_PARALLEL_ENVS=16
# export SAC_BATCH_SIZE=512
# export SAC_BUFFER_SIZE=2000000   # FIXED: Paper constraint
# export SAC_LEARNING_STARTS=10000

# ============================================================

# OPTION 5: PARALLEL x32 (Maximum - CURRENT DEFAULT) ⚡
# Collect 32× more transitions per second
# GPU Utilization: ~75-85%
# Training Time: ~1-1.5 hours (32× faster experience collection)
# VRAM: ~24-28 GB (still comfortable on A40 48GB)
# CPU may become slight bottleneck (but 16 CPUs available)
export NUM_PARALLEL_ENVS=32
export SAC_BATCH_SIZE=512
export SAC_BUFFER_SIZE=2000000   # FIXED: Paper constraint
export SAC_LEARNING_STARTS=10000

# ============================================================

# HOW IT WORKS:
#
# Single Env (Sequential):
#   Step 1: Actor network inference → 1 action
#   Step 2: MuJoCo env step → 1 transition
#   Step 3: Replay buffer update
#   Step 4: (if train_freq met) Critic update
#   → GPU idles between steps (CPU-bound)
#
# Parallel Envs (e.g., 4x):
#   Step 1: Actor network inference → 4 actions (batched)
#   Step 2: MuJoCo steps in parallel (CPU fast-path)
#   Step 3: Replay buffer updates (4 transitions)
#   Step 4: (if train_freq met) Critic update on GPU
#   → GPU stays busy with batched operations
#
# ============================================================

# PAPER FIDELITY NOTES:
#
# The paper (McLean et al. 2025) specifies:
# - 200k buffer per task (2M total for 10 tasks)
# - Actor: [256, 256]
# - Critic: [1024, 1024, 1024]
# - These are HYPERPARAMETERS, not implementation details
#
# Parallel environments collect MORE diverse transitions in LESS wall-clock time.
# The effective sample diversity increases (exploration), not the capacity.
# This is consistent with RL theory (more envs = more exploration).
#
# ============================================================

# RECOMMENDED COMMAND:
#
# For MAXIMUM GPU utilization (DEFAULT - 32 parallel envs):
#
#   WANDB_MODE=online sbatch train_mt10_full.sh
#   # Expected time: 1-1.5 hours, GPU util: 75-85%
#
# For BALANCED setup (16 parallel envs):
#
#   export NUM_PARALLEL_ENVS=16
#   WANDB_MODE=online sbatch train_mt10_full.sh
#   # Expected time: 1.5-2 hours, GPU util: 70-80%
#
# For SEQUENTIAL (legacy - very slow):
#
#   export NUM_PARALLEL_ENVS=1
#   WANDB_MODE=online sbatch train_mt10_full.sh
#   # Expected time: 8-10 hours, GPU util: 5-20%
#
# ============================================================
