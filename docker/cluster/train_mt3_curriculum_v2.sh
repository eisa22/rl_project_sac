#!/bin/bash
# Curriculum Learning for MT3 with Stronger Architecture
# Phase 1: reach-only
# Phase 2: reach + push
# Phase 3: reach + push + pick-place

#SBATCH --job-name=mt3_curriculum_v2
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/metaworld_project/logs/mt3_curriculum_v2_%j.log
#SBATCH --error=/home/%u/metaworld_project/logs/mt3_curriculum_v2_%j.err

set -e

CLUSTER_USER="${USER}"
PROJECT_DIR="/home/${CLUSTER_USER}/metaworld_project"
SIF_PATH="/share/${CLUSTER_USER}/containers/sac_mtmh.sif"
SEED="${1:-1}"

echo "=========================================="
echo "Curriculum Learning: MT3 (MTMH-SAC v2)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Seed: $SEED"
echo ""

if [ ! -f "$SIF_PATH" ]; then
    echo "❌ Singularity image not found: $SIF_PATH"
    exit 1
fi

if [ ! -f ~/.wandb_api_key ]; then
    WANDB_MODE="disabled"
else
    WANDB_API_KEY=$(cat ~/.wandb_api_key)
    export WANDB_API_KEY
    WANDB_MODE="online"
fi

export WANDB_MODE

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/models"

echo "🚀 Starting Curriculum Training..."
echo ""

# Run with stronger architecture + higher reward scaling for pick-place
apptainer exec --nv \
    --bind ${PROJECT_DIR}:/workspace \
    "$SIF_PATH" \
    bash -c "
        cd /workspace
        export WANDB_MODE='${WANDB_MODE}'
        
        python train_mt3_curriculum_v2.py \
            --run_name mt3_curriculum_v2_seed${SEED} \
            --total_steps 6000000 \
            --seed ${SEED} \
            --lr 3e-4 \
            --batch_size 256 \
            --tau 0.005 \
            --trunk_hidden_actor 512,512 \
            --head_hidden_actor 256 \
            --trunk_hidden_critic 2048,2048 \
            --head_hidden_critic 1024,1024 \
            --reward_scale 2.0
    "

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Curriculum Training Completed!"
else
    echo "❌ Training Failed (Exit Code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
