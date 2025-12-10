#!/bin/bash
#SBATCH --job-name=mt10_20M_a100
#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/e11704784/metaworld_project/logs/mt10_20M_%j.log
#SBATCH --error=/home/e11704784/metaworld_project/logs/mt10_20M_%j.err

# =============================================================================
# MT10 Full Training (20M Timesteps) - A100 Optimized
# =============================================================================
# Paper: McLean et al. 2025 - "Multi-Task RL Enables Parameter Scaling"
# Hardware: A100 (80GB VRAM, 16 CPU cores)
# Expected Runtime: 50-60 hours (drawer tasks are slow)
# GPU Utilization: 80-90% (48 parallel environments)
# =============================================================================

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

# Load environment config with absolute path
CLUSTER_DIR="/home/e11704784/metaworld_project/source/rl_project_sac/docker/cluster"
PROJECT_DIR="/home/e11704784/metaworld_project/source/rl_project_sac"
set -a
source "${CLUSTER_DIR}/.env.cluster"
set +a

# =============================================================================
# PAPER HYPERPARAMETERS (McLean et al. 2025)
# =============================================================================
# These are FIXED and match the Paper's MT10 configuration
export TOTAL_TIMESTEPS=20000000          # Paper: 20M timesteps (2M per task)
export SAC_BATCH_SIZE=512                # Paper: Standard SAC batch size
export SAC_BUFFER_SIZE=2000000           # 200k × 10 tasks (larger = better for MTRL)
export SAC_LEARNING_RATE=0.0003          # Paper: 3e-4 (standard SAC)
export SAC_ACTOR_HIDDEN="256,256"        # Paper: [256, 256] baseline
export SAC_CRITIC_HIDDEN="1024,1024,1024" # Paper: [1024, 1024, 1024] (critic scaling!)

# =============================================================================
# GPU OPTIMIZATION (A100-Specific)
# =============================================================================
# This does NOT affect Paper hyperparameters, only GPU throughput
export NUM_PARALLEL_ENVS=48              # A100: 48× parallel MuJoCo instances
export CUDA_VISIBLE_DEVICES=0

# W&B Configuration
export WANDB_MODE=online
export WANDB_PROJECT="Robot_learning_2025"
export RUN_NAME="johannes_MT10_20M_A100_${SLURM_JOB_ID}"

echo ""
echo "========================================="
echo "TRAINING CONFIGURATION"
echo "========================================="
echo "Paper: McLean et al. 2025 (MT10 Benchmark)"
echo "Total Timesteps: $(printf '%s' "$TOTAL_TIMESTEPS" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
echo "Batch Size: $SAC_BATCH_SIZE (Paper Standard)"
echo "Buffer Size: $(printf '%s' "$SAC_BUFFER_SIZE" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta') (200k × 10 tasks)"
echo "Actor: [$SAC_ACTOR_HIDDEN]"
echo "Critic: [$SAC_CRITIC_HIDDEN] (Paper: Critic scaling > Actor scaling!)"
echo ""
echo "GPU OPTIMIZATION:"
echo "Hardware: A100 (80GB VRAM)"
echo "Parallel Envs: $NUM_PARALLEL_ENVS× (8-10× faster)"
echo "Expected GPU Util: 80-90%"
echo "Expected VRAM: ~45-50GB / 80GB"
echo "Expected Training Time: 50-60 hours (drawer tasks: ~55 steps/sec)"
echo "========================================="
echo ""

# GPU Check
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv,noheader
echo ""

# Start timing
START_TIME=$(date +%s)

# =============================================================================
# RUN TRAINING
# =============================================================================
echo "🚀 Starting MT10 Training (20M timesteps)..."
echo ""

"${CLUSTER_DIR}/run_singularity.sh" \
    "${PROJECT_DIR}" \
    base \
    train_metaworld.py \
    --run_name "$RUN_NAME" \
    --total_steps "$TOTAL_TIMESTEPS" \
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

# GPU summary
echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: MT10 Training completed!"
    echo "📊 Check W&B: https://wandb.ai/Robot_learning_2025/Robot_learning_2025"
    echo "💾 Models saved to: ${CLUSTER_MODELS_DIR}/${RUN_NAME}/"
else
    echo ""
    echo "❌ ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "📋 Check logs: ~/metaworld_project/logs/mt10_20M_${SLURM_JOB_ID}.log"
fi

exit $TRAIN_EXIT_CODE
