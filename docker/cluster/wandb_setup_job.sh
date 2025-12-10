#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a40:1
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --job-name="wandb-setup"
#SBATCH --output=/home/%u/metaworld_project/logs/wandb_setup_%j.log
#SBATCH --error=/home/%u/metaworld_project/logs/wandb_setup_%j.err

# =============================================================================
# W&B Setup Job - Initialize wandb in container
# This job will log in to wandb and cache the credentials
# =============================================================================

echo "========================================="
echo "W&B Setup - Container Login"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "========================================="
echo ""

# Load cluster config
ACTUAL_SCRIPT_DIR="/home/${USER}/metaworld_project/source/rl_project_sac/docker/cluster"
set -a
source "${ACTUAL_SCRIPT_DIR}/.env.cluster"
set +a

# W&B API Key (SET THIS!)
WANDB_API_KEY="${1:-}"

if [ -z "${WANDB_API_KEY}" ]; then
    echo "❌ ERROR: No W&B API key provided!"
    echo ""
    echo "Usage:"
    echo "  sbatch wandb_setup_job.sh YOUR_WANDB_API_KEY"
    echo ""
    echo "Get your API key from: https://wandb.ai/authorize"
    exit 1
fi

echo "Setting up W&B with provided API key..."
echo ""

# Create wandb config directory in container-accessible location
mkdir -p "${CLUSTER_WANDB_DIR}/.netrc_dir"

# Run wandb login in container
${CLUSTER_SINGULARITY_BIN} exec \
    --nv \
    --bind "${CLUSTER_WANDB_DIR}:/root/.config/wandb" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env WANDB_DIR="${CLUSTER_WANDB_DIR}" \
    --env HOME=/root \
    "${CLUSTER_FULL_SIF_PATH}" \
    bash -c "wandb login ${WANDB_API_KEY} && wandb status"

EXIT_CODE=$?

echo ""
echo "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ W&B setup completed successfully!"
    echo ""
    echo "W&B is now configured for all future jobs."
    echo "Credentials stored in: ${CLUSTER_WANDB_DIR}"
    echo ""
    echo "Next step: Run a test job"
    echo "  sbatch quick_test.sh"
else
    echo "❌ W&B setup failed!"
    echo ""
    echo "Check the log for details:"
    echo "  cat ~/metaworld_project/logs/wandb_setup_${SLURM_JOB_ID}.log"
fi
echo "========================================="

exit ${EXIT_CODE}
