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
    --exclude='.mt10_checkpoints_aggregate_success_05' \
    --exclude='ARCHIVE/' \
    --exclude='models/' \
    --exclude='.claude/' \
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

# Submit both MT3 and MT10 jobs
echo "Submitting MT3 training job with seed ${SEED}..."
MT3_JOB=\$(sbatch docker/cluster/train_mt3_mtmh.sh ${SEED} | awk '{print \$4}')
echo "MT3 Job ID: \$MT3_JOB"

echo ""
#echo "Submitting MT10 training job with seed ${SEED}..."
#MT10_JOB=\$(sbatch docker/cluster/train_mt10_mtmh.sh ${SEED} | awk '{print \$4}')
#echo "MT10 Job ID: \$MT10_JOB"

# Wait a moment for jobs to appear in queue
sleep 2

# Show job status
echo ""
echo "Jobs submitted! Current queue status:"
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
echo "Submitted Jobs:"  
#echo "  - MT3:  Job \$MT3_JOB (3 tasks, 16 envs/task, batch 2048)"
#echo "  - MT10: Job \$MT10_JOB (10 tasks, 8 envs/task, batch 2048)"
echo ""
echo "Next steps:"
echo "  1. Monitor: ssh ${CLUSTER_USER}@${CLUSTER_HOST} 'squeue -u \$USER'"
echo "  2. MT3 logs:  ssh ${CLUSTER_HOST} 'tail -f ~/metaworld_project/logs/mt3_mtmh_*.log'"
# echo "  3. MT10 logs: ssh ${CLUSTER_HOST} 'tail -f ~/metaworld_project/logs/mt10_mtmh_*.log'"
echo "  4. WandB: https://wandb.ai/Robot_learning_2025/Robot_learning_2025"
echo ""
