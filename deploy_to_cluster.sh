#!/bin/bash
# Quick deployment to DataLAB Cluster (code + .sif if newer)
# Usage: ./deploy_to_cluster.sh [seed]

set -e

CLUSTER_HOST="datalab"
CLUSTER_USER="e11704784"
CLUSTER_SIF_DIR="/share/${CLUSTER_USER}/containers"
SEED="${1:-1}"

echo "=========================================="
echo "Deploying MTMH-SAC to DataLAB"
echo "=========================================="
echo "Cluster: ${CLUSTER_HOST}"
echo "User: ${CLUSTER_USER}"
echo "Seed: ${SEED}"
echo ""

# # Check if .sif exists locally
# if [ -f "sac_mtmh.sif" ]; then
#     echo "📤 Step 1: Uploading Singularity image..."
#     scp -C sac_mtmh.sif ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_SIF_DIR}/
    
#     if [ $? -ne 0 ]; then
#         echo "❌ Failed to upload .sif"
#         exit 1
#     fi
#     echo "✅ .sif uploaded"
#     echo ""
# else
#     echo "⚠️  No .sif found locally - skipping .sif upload"
#     echo "   If you built a new image, run:"
#     echo "   ./docker/cluster/convert_to_singularity.sh"
#     echo ""
# fi

echo "📦 Cleaning Python cache locally..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✅ Local cache cleaned"
echo ""

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
    ./ ${CLUSTER_USER}@${CLUSTER_HOST}:~/metaworld_project/

if [ $? -ne 0 ]; then
    echo "❌ Failed to sync code"
    exit 1
fi

echo ""
echo "✅ Code synced successfully!"
echo ""

echo "📤 Step 3: Submitting training job..."
echo ""

# Submit job via SSH
ssh ${CLUSTER_USER}@${CLUSTER_HOST} << ENDSSH
set -e

# Navigate to project
cd ~/metaworld_project

# Clean remote Python cache
echo "Cleaning remote Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "Remote cache cleaned"
echo ""

# Make scripts executable
chmod +x docker/cluster/*.sh

# Submit SLURM job
echo "Submitting SLURM job with seed ${SEED}..."
sbatch docker/cluster/train_mt3_mtmh.sh ${SEED}

# Wait a moment for job submission
sleep 2

# Show job status
echo ""
echo "Job submitted! Current queue status:"
squeue -u ${CLUSTER_USER}

ENDSSH

if [ $? -ne 0 ]; then
    echo "❌ Failed to submit job"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Deployment & Job Submission Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Monitor: ssh ${CLUSTER_USER}@${CLUSTER_HOST} 'squeue -u \$USER'"
echo "  2. Logs: ssh ${CLUSTER_HOST} 'cd ~/metaworld_project && tail -f logs/*.log'"
echo "  3. WandB: https://wandb.ai/Robot_learning_2025/Robot_learning_2025"
echo ""
