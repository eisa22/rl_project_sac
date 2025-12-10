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

echo "========================================="
echo "Meta-World SAC - MT10 Full Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "========================================="

# Load cluster config
ACTUAL_SCRIPT_DIR="/home/${USER}/metaworld_project/source/rl_project_sac/docker/cluster"
set -a
source "${ACTUAL_SCRIPT_DIR}/.env.cluster"
set +a

# Training parameters (FULL run)
RUN_NAME="${USER}_mt10_full_${SLURM_JOB_ID}"
TOTAL_STEPS=2000000
SEED=${SLURM_JOB_ID}

echo "Run name: ${RUN_NAME}"
echo "Total steps: ${TOTAL_STEPS}"
echo "Seed: ${SEED}"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
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
echo "Finished: $(date)"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Training completed successfully"
    echo ""
    echo "Results:"
    echo "  Final model: ${CLUSTER_MODELS_DIR}/${RUN_NAME}/final_model.pt"
    echo "  Logs: ${CLUSTER_LOGS_DIR}/"
    echo ""
    echo "To sync W&B results:"
    echo "  cd ${CLUSTER_WANDB_DIR} && wandb sync"
else
    echo "❌ Training failed with exit code ${EXIT_CODE}"
fi
echo "========================================="

exit ${EXIT_CODE}
