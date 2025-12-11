#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a40:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --job-name="sac-mt10-test"
#SBATCH --output=/home/%u/metaworld_project/logs/train_test_%j.log
#SBATCH --error=/home/%u/metaworld_project/logs/train_test_%j.err

# =============================================================================
# MT10 Training Test - Meta-World SAC
# Short training run: 100k steps (~15-20 min)
# Tests: Full training pipeline, GPU utilization, W&B logging
# =============================================================================

echo "========================================="
echo "Meta-World SAC - MT10 Training Test"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "========================================="

# Load cluster config
ACTUAL_SCRIPT_DIR="/home/${USER}/metaworld_project/source/rl_project_sac/docker/cluster"
set -a
source "${ACTUAL_SCRIPT_DIR}/.env.cluster"
set +a

# Training parameters (SHORT test run)
RUN_NAME="Cluster_test_long_${SLURM_JOB_ID}"
TOTAL_STEPS=100000
SEED=${SLURM_JOB_ID}

# Enable W&B online logging for this run
export WANDB_MODE=online

echo "Run name: ${RUN_NAME}"
echo "Total steps: ${TOTAL_STEPS}"
echo "Seed: ${SEED}"
echo ""

# Run training
bash "${CLUSTER_SAC_DIR}/docker/cluster/run_singularity.sh" \
    "${CLUSTER_SAC_DIR}" \
    "base" \
    "train_metaworld.py" \
    --run_name "${RUN_NAME}" \
    --total_steps ${TOTAL_STEPS} \
    --seed ${SEED}

EXIT_CODE=$?

echo ""
echo "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Training test completed successfully"
    echo ""
    echo "Check results:"
    echo "  Logs: ${CLUSTER_LOGS_DIR}/"
    echo "  Models: ${CLUSTER_MODELS_DIR}/${RUN_NAME}/"
    echo "  W&B: wandb sync ${CLUSTER_WANDB_DIR}"
else
    echo "❌ Training test failed with exit code ${EXIT_CODE}"
fi
echo "========================================="

exit ${EXIT_CODE}
