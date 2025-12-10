#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a40:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --job-name="sac-mt10-full"
#SBATCH --output=/home/%u/metaworld_project/logs/train_full_%j.log
#SBATCH --error=/home/%u/metaworld_project/logs/train_full_%j.err

# =============================================================================
# MT10 Full Training - Meta-World SAC
# Full training run: 2M steps (~8-10 hours on A40)
# GPU-optimized for A40 48GB VRAM
# =============================================================================

START_TIME=$(date +%s)
echo "========================================="
echo "Meta-World SAC - MT10 Full Training (GPU-OPTIMIZED)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "========================================="

# Load cluster config
ACTUAL_SCRIPT_DIR="/home/${USER}/metaworld_project/source/rl_project_sac/docker/cluster"
set -a
source "${ACTUAL_SCRIPT_DIR}/.env.cluster"
set +a

# GPU Optimization: Parallel Environments (NOT aggressive batch/buffer)
# Key: Keep PAPER hyperparameters, use parallel envs for GPU throughput
export NUM_PARALLEL_ENVS=32        # 32 parallel environments (extreme GPU load)
export SAC_BATCH_SIZE=512          # PAPER: Standard batch size
export SAC_BUFFER_SIZE=2000000     # PAPER: 200k × 10 tasks (FIXED!)
export SAC_LEARNING_STARTS=10000   # PAPER: Standard starts

echo "GPU Config: PARALLEL ENVIRONMENTS (MAXIMUM)"
echo "  Parallel Envs: 32 (32× faster experience collection)"
echo "  Batch Size: 512 (PAPER standard)"
echo "  Buffer Size: 2M (PAPER: 200k × 10 tasks)"
echo "  Expected GPU Util: 75-85% (vs 5-20% sequential)"
echo "  Expected Training Time: 1-1.5 hours (32× faster)"
echo ""

# Training parameters (FULL 2M steps)
RUN_NAME="Cluster_mt10_full_${SLURM_JOB_ID}"
TOTAL_STEPS=2000000
SEED=${SLURM_JOB_ID}

echo ""
echo "Run name: ${RUN_NAME}"
echo "Total steps: ${TOTAL_STEPS:,}"
echo "Seed: ${SEED}"
echo ""
echo "========================================="
echo "GPU INFO:"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv
nvidia-smi dmon -s pucm -c 1
echo ""

# Start GPU monitoring in background
GPU_LOG="${CLUSTER_LOGS_DIR}/gpu_full_${SLURM_JOB_ID}.log"
bash "${ACTUAL_SCRIPT_DIR}/monitor_gpu.sh" "${GPU_LOG}" 1 &
GPU_MONITOR_PID=$!

echo "🚀 Starting training..."
echo ""

# Run training with environment overrides
bash "${CLUSTER_SAC_DIR}/docker/cluster/run_singularity.sh" \
    "${CLUSTER_SAC_DIR}" \
    "base" \
    "train_metaworld.py" \
    --run_name "${RUN_NAME}" \
    --total_steps ${TOTAL_STEPS} \
    --seed ${SEED}

EXIT_CODE=$?

# Stop GPU monitoring
kill ${GPU_MONITOR_PID} 2>/dev/null || true

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================="
echo "TRAINING COMPLETED"
echo "========================================="
echo "Elapsed Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Total Seconds: ${ELAPSED}"
echo "Finished: $(date)"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Status: SUCCESS"
    echo ""
    echo "📊 Results Location:"
    echo "  Final Model: ${CLUSTER_MODELS_DIR}/${RUN_NAME}/final_model.pt"
    echo "  Training Log: ${CLUSTER_LOGS_DIR}/train_full_${SLURM_JOB_ID}.log"
    echo "  GPU Metrics: ${GPU_LOG}"
    echo ""
    echo "📈 W&B Dashboard:"
    echo "  Run: ${RUN_NAME}"
    echo "  Project: Robot_learning_2025"
    echo "  URL: https://wandb.ai/Robot_learning_2025/Robot_learning_2025"
    echo ""
    echo "📥 Download Results (on your local PC):"
    echo "  rsync -avP datalab:/home/${USER}/metaworld_project/models/${RUN_NAME}/ ./models_cluster/${RUN_NAME}/"
    echo "  rsync -avP datalab:/home/${USER}/metaworld_project/logs/train_full_${SLURM_JOB_ID}.* ./logs_cluster/"
else
    echo "❌ Status: FAILED (exit code ${EXIT_CODE})"
fi
echo "========================================="

exit ${EXIT_CODE}
