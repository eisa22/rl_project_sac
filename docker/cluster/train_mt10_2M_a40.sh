#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a40:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --job-name="sac-mt10-2M"
#SBATCH --output=/home/%u/metaworld_project/logs/train_2M_%j.log
#SBATCH --error=/home/%u/metaworld_project/logs/train_2M_%j.err

# =============================================================================
# MT10 Training (2M Timesteps) - A40 Optimized
# =============================================================================
# Paper: McLean et al. 2025 - "Multi-Task RL Enables Parameter Scaling"
# Hardware: A40 (48GB VRAM, 16 CPU cores)
# Expected Runtime: 1-1.5 hours
# GPU Utilization: 75-85% (32 parallel environments)
# =============================================================================

START_TIME=$(date +%s)
echo "========================================="
echo "Meta-World SAC - MT10 Training (2M Steps)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "========================================="

# Load cluster config
ACTUAL_SCRIPT_DIR="/home/${USER}/metaworld_project/source/rl_project_sac/docker/cluster"
set -a
source "${ACTUAL_SCRIPT_DIR}/.env.cluster"
set +a

# =============================================================================
# PAPER HYPERPARAMETERS (McLean et al. 2025)
# =============================================================================
export SAC_BATCH_SIZE=512          # Paper: Standard SAC batch
export SAC_BUFFER_SIZE=2000000     # Paper: 200k × 10 tasks
export SAC_LEARNING_STARTS=10000   # Paper: Standard
export SAC_LEARNING_RATE=0.0003    # Paper: 3e-4
export SAC_ACTOR_HIDDEN="256,256"  # Paper: [256, 256]
export SAC_CRITIC_HIDDEN="1024,1024,1024"  # Paper: [1024, 1024, 1024]

# =============================================================================
# GPU OPTIMIZATION (A40-Specific)
# =============================================================================
export NUM_PARALLEL_ENVS=32        # 32× parallel environments
export CUDA_VISIBLE_DEVICES=0

# W&B Configuration
export WANDB_MODE=online
export WANDB_PROJECT="Robot_learning_2025"
RUN_NAME="MT10_2M_A40_${SLURM_JOB_ID}"
TOTAL_STEPS=2000000

echo ""
echo "========================================="
echo "TRAINING CONFIGURATION"
echo "========================================="
echo "Paper: McLean et al. 2025 (MT10 Benchmark)"
echo "Total Timesteps: 2,000,000"
echo "Batch Size: $SAC_BATCH_SIZE (Paper Standard)"
echo "Buffer Size: 2,000,000 (200k × 10 tasks)"
echo "Actor: [$SAC_ACTOR_HIDDEN]"
echo "Critic: [$SAC_CRITIC_HIDDEN]"
echo ""
echo "GPU OPTIMIZATION:"
echo "Hardware: A40 (48GB VRAM)"
echo "Parallel Envs: $NUM_PARALLEL_ENVS× (32× faster)"
echo "Expected GPU Util: 75-85%"
echo "Expected Training Time: 1-1.5 hours"
echo "========================================="
echo ""

# GPU Check
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
echo ""

# =============================================================================
# RUN TRAINING
# =============================================================================
echo "🚀 Starting MT10 Training (2M timesteps)..."
echo ""

"${ACTUAL_SCRIPT_DIR}/run_singularity.sh" \
    "${PROJECT_DIR}" \
    base \
    train_metaworld.py \
    --run_name "$RUN_NAME" \
    --total_steps "$TOTAL_STEPS" \
    --seed 42

TRAIN_EXIT_CODE=$?

# End timing
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================="
echo "TRAINING COMPLETED"
echo "========================================="
echo "Exit Code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo "Elapsed Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "========================================="

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: MT10 Training completed!"
    echo "📊 Check W&B: https://wandb.ai/Robot_learning_2025/Robot_learning_2025"
else
    echo ""
    echo "❌ ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
fi

exit $TRAIN_EXIT_CODE
