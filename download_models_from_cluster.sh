#!/bin/bash
# Download trained models from cluster to local machine
# Usage: ./download_models_from_cluster.sh [job_name]

set -e

CLUSTER_USER="e11704784"
CLUSTER_HOST="datalab"
LOCAL_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_MODELS_DIR="${LOCAL_BASE}/models_cluster"
REMOTE_MODELS_DIR="/home/${CLUSTER_USER}/metaworld_project/models"

# Optional: specify job name as argument
JOB_NAME="${1:-}"

echo "======================================"
echo "Download Models from Cluster"
echo "======================================"
echo ""

# Create local models directory
mkdir -p "${LOCAL_MODELS_DIR}"

if [ -n "${JOB_NAME}" ]; then
    echo "Downloading models for job: ${JOB_NAME}"
    REMOTE_PATH="${REMOTE_MODELS_DIR}/${JOB_NAME}/"
    LOCAL_PATH="${LOCAL_MODELS_DIR}/${JOB_NAME}/"
    
    rsync -avP --include='*.pt' --include='*/' --exclude='*' \
        "${CLUSTER_HOST}:${REMOTE_PATH}" \
        "${LOCAL_PATH}"
else
    echo "Downloading all models from cluster"
    echo "Remote: ${CLUSTER_HOST}:${REMOTE_MODELS_DIR}"
    echo "Local:  ${LOCAL_MODELS_DIR}"
    echo ""
    
    rsync -avP --include='*.pt' --include='*/' --exclude='*' \
        "${CLUSTER_HOST}:${REMOTE_MODELS_DIR}/" \
        "${LOCAL_MODELS_DIR}/"
fi

echo ""
echo "✓ Download complete!"
echo ""
echo "Models saved to: ${LOCAL_MODELS_DIR}"
echo ""
echo "To download a specific job's models:"
echo "  ./download_models_from_cluster.sh SAC_mt10_test"
echo ""
echo "To download logs and wandb data:"
echo "  rsync -avP ${CLUSTER_HOST}:/home/${CLUSTER_USER}/metaworld_project/logs/ ./logs_cluster/"
echo "  rsync -avP ${CLUSTER_HOST}:/home/${CLUSTER_USER}/metaworld_project/wandb_cache/ ./wandb_cluster/"
