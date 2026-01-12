#!/bin/bash
# Upload code + .sif to DataLAB cluster
# Usage: ./upload_to_cluster.sh e11704784

set -e

CLUSTER_HOST="datalab"
CLUSTER_USER="${1:-e11704784}"
CLUSTER_SIF_DIR="/share/${CLUSTER_USER}/containers"
CLUSTER_PROJECT_DIR="/home/${CLUSTER_USER}/metaworld_project"

echo "=========================================="
echo "Uploading to DataLAB Cluster"
echo "=========================================="
echo "Cluster User: $CLUSTER_USER"
echo "Cluster Host: $CLUSTER_HOST"
echo "SIF Location: $CLUSTER_SIF_DIR (containers)"
echo "Project Location: $CLUSTER_PROJECT_DIR (metaworld_project)"
echo ""

# # Check .sif exists
# if [ ! -f "sac_mtmh.sif" ]; then
#     echo "❌ Error: sac_mtmh.sif not found!"
#     echo "   Run: bash docker/cluster/convert_to_singularity.sh"
#     exit 1
# fi

# echo "📤 Step 1: Uploading Singularity image (8GB)..."
# scp -C sac_mtmh.sif ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_SIF_DIR}/

# if [ $? -ne 0 ]; then
#     echo "❌ Failed to upload .sif file"
#     exit 1
# fi

# echo "✅ .sif uploaded"
# echo ""

echo "📤 Step 2: Syncing code to cluster..."
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.sif' \
    --exclude='logs/' \
    --exclude='models/' \
    --exclude='wandb/' \
    --exclude='.wandb_api_key' \
    ./ ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PROJECT_DIR}/

if [ $? -ne 0 ]; then
    echo "❌ Failed to sync code"
    exit 1
fi

echo "✅ Code synced"
echo ""

# Create directories on cluster
echo "📁 Creating/verifying directories on cluster..."
ssh ${CLUSTER_USER}@${CLUSTER_HOST} << EOF
set -e
mkdir -p ${CLUSTER_PROJECT_DIR}/logs
mkdir -p ${CLUSTER_PROJECT_DIR}/models
mkdir -p ${CLUSTER_PROJECT_DIR}/wandb_cache
mkdir -p ${CLUSTER_SIF_DIR}
chmod -R u+rwx ${CLUSTER_PROJECT_DIR}
echo "✅ Directories ready"
EOF

echo ""
echo "=========================================="
echo "✅ Upload Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. SSH to cluster:"
echo "     ssh ${CLUSTER_USER}@${CLUSTER_HOST}"
echo ""
echo "  2. Setup W&B API key:"
echo "     bash /home/${CLUSTER_USER}/metaworld_project/docker/cluster/setup_wandb.sh"
echo ""
echo "  3. Submit training job:"
echo "     sbatch /home/${CLUSTER_USER}/metaworld_project/docker/cluster/train_mt3_mtmh.sh"
echo ""
