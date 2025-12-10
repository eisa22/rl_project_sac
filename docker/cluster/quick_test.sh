#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a40:1
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --job-name="sac-quick-test"
#SBATCH --output=/home/%u/metaworld_project/logs/quick_test_%j.log
#SBATCH --error=/home/%u/metaworld_project/logs/quick_test_%j.err

# =============================================================================
# Quick Test - Meta-World SAC with GPU Monitoring
# Very short training run: 10k steps (~2-3 min)
# Tests: Setup, GPU utilization, basic training loop
# =============================================================================

echo "========================================="
echo "Meta-World SAC - Quick Test with GPU Monitoring"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "========================================="

# Load cluster config
ACTUAL_SCRIPT_DIR="/home/${USER}/metaworld_project/source/rl_project_sac/docker/cluster"
set -a
source "${ACTUAL_SCRIPT_DIR}/.env.cluster"
set +a

# Training parameters (VERY SHORT test)
# Use a clear, online-visible name
RUN_NAME="Cluster_test_${SLURM_JOB_ID}"
TOTAL_STEPS=10000
SEED=${SLURM_JOB_ID}

# Force W&B online logging for this run
export WANDB_MODE=online

echo "Run name: ${RUN_NAME}"
echo "Total steps: ${TOTAL_STEPS}"
echo "Seed: ${SEED}"
echo ""

# Start GPU monitoring in background
GPU_LOG="${CLUSTER_LOGS_DIR}/gpu_${SLURM_JOB_ID}.log"
echo "Starting GPU monitoring: ${GPU_LOG}"
bash "${ACTUAL_SCRIPT_DIR}/monitor_gpu.sh" "${GPU_LOG}" 2 &
GPU_MONITOR_PID=$!

# Run training
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

echo ""
echo "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Quick test completed successfully"
    echo ""
    echo "Check results:"
    echo "  Training log: ${CLUSTER_LOGS_DIR}/quick_test_${SLURM_JOB_ID}.log"
    echo "  GPU monitoring: ${GPU_LOG}"
    echo "  Models: ${CLUSTER_MODELS_DIR}/${RUN_NAME}/"
    echo ""
    echo "GPU utilization summary:"
    tail -n 20 "${GPU_LOG}" || echo "  (log not available)"
    echo ""
    echo "To sync W&B:"
    echo "  wandb sync ${CLUSTER_WANDB_DIR}/wandb/run-*"
else
    echo "❌ Quick test failed with exit code ${EXIT_CODE}"
fi
echo "========================================="

exit ${EXIT_CODE}
