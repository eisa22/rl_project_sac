#!/bin/bash
# Upload code to datalab cluster
# Syncs the project code to ~/metaworld_project/source/rl_project_sac

set -e

CLUSTER_USER="e11704784"
CLUSTER_HOST="datalab"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_BASE="/home/${CLUSTER_USER}/metaworld_project/source"
REMOTE_DIR="${REMOTE_BASE}/rl_project_sac"

echo "======================================"
echo "Uploading code to cluster"
echo "======================================"
echo "Local:  ${LOCAL_DIR}"
echo "Remote: ${CLUSTER_HOST}:${REMOTE_DIR}"
echo ""

# Create remote directory structure if it doesn't exist
ssh ${CLUSTER_HOST} "mkdir -p ${REMOTE_BASE}"

# Sync code (exclude large files, caches, logs, models)
rsync -avP --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude 'wandb/' \
  --exclude 'logs*/' \
  --exclude 'models*/' \
  --exclude '*_logs/' \
  --exclude '*.sif' \
  --exclude '*.pt' \
  --exclude 'sac_metaworld.sif' \
  "${LOCAL_DIR}/" \
  "${CLUSTER_HOST}:${REMOTE_DIR}/"

echo ""
echo "✓ Code upload complete!"
echo ""
echo "Next steps:"
echo "1. SSH to cluster: ssh ${CLUSTER_HOST}"
echo "2. Navigate to: cd ${REMOTE_DIR}/docker/cluster"
echo "3. Setup W&B: wandb login"
echo "4. Run test job: sbatch test_simple.sh"
