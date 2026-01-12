#!/bin/bash
# MTMH-SAC MT3 Training on DataLAB Cluster
# Trains on: reach-v3, push-v3, pick-place-v3
# Duration: ~12-16 hours
# GPU: A40 recommended

#SBATCH --job-name=mt3_mtmh_sac
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/home/%u/metaworld_project/logs/mt3_mtmh_%j.log
#SBATCH --error=/home/%u/metaworld_project/logs/mt3_mtmh_%j.err

set -e

# === Configuration ===
CLUSTER_USER="${USER}"
PROJECT_DIR="/home/${CLUSTER_USER}/metaworld_project"
SIF_PATH="/share/${CLUSTER_USER}/containers/sac_mtmh.sif"
SEED="${1:-1}"

echo "=========================================="
echo "MTMH-SAC Training: MT3 (reach, push, pick-place)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "User: $CLUSTER_USER"
echo "Seed: $SEED"
echo "GPU: $SLURM_GPUS"
echo "Time Limit: ${SLURM_TIME}"
echo ""

# Check .sif exists
if [ ! -f "$SIF_PATH" ]; then
    echo "❌ Singularity image not found: $SIF_PATH"
    echo "   Make sure to run: bash docker/cluster/upload_to_cluster.sh $CLUSTER_USER"
    echo "   .sif must be at: /share/$CLUSTER_USER/containers/sac_mtmh.sif"
    exit 1
fi

# Check W&B API key
if [ ! -f ~/.wandb_api_key ]; then
    echo "⚠️  W&B API key not found at ~/.wandb_api_key"
    echo "   Run: bash ${PROJECT_DIR}/docker/cluster/setup_wandb.sh"
    echo "   Continuing without W&B logging..."
    WANDB_MODE="disabled"
else
    WANDB_API_KEY=$(cat ~/.wandb_api_key)
    export WANDB_API_KEY
    WANDB_MODE="online"
fi

export WANDB_MODE

# Create output directories if needed
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/models"

echo "🚀 Starting training in Singularity container..."
echo "Container: $SIF_PATH"
echo "Command: python train_mt3_mtmh.py --seed $SEED"
echo ""

# Run training in Singularity
apptainer exec --nv \
    --bind ${PROJECT_DIR}:/workspace \
    --bind /share/${CLUSTER_USER}/containers:/containers_ro:ro \
    "$SIF_PATH" \
    bash -c "
        cd /workspace
        
        # Export W&B settings
        export WANDB_MODE='${WANDB_MODE}'
        
        # Run training
        python train_mt3_mtmh.py \
            --run_name mt3_mtmh_highexplore_seed${SEED} \
            --total_steps 6000000 \
            --seed ${SEED} \
            --lr 1e-3 \
            --alpha_lr 1e-3 \
            --batch_size 256 \
            --tau 0.01 \
            --learning_starts 10000 \
            --trunk_hidden_actor 512,512 \
            --head_hidden_actor 256 \
            --trunk_hidden_critic 1024,1024 \
            --head_hidden_critic 512,512 \
            --reward_scale 5.0
    "

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training Completed Successfully!"
else
    echo "❌ Training Failed (Exit Code: $EXIT_CODE)"
fi
echo "=========================================="
echo "Project: /home/${CLUSTER_USER}/metaworld_project/"
echo "Log file: ${PROJECT_DIR}/logs/mt3_mtmh_${SLURM_JOB_ID}.log"
echo "Models: ${PROJECT_DIR}/models/"
echo ""

exit $EXIT_CODE
