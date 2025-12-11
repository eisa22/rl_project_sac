#!/bin/bash
# Setup W&B on datalab cluster
# Run this script once on the cluster to configure wandb

set -e

echo "======================================"
echo "W&B Setup for Cluster"
echo "======================================"
echo ""
echo "This script will:"
echo "1. Initialize wandb login"
echo "2. Configure offline mode for cluster runs"
echo "3. Set up wandb cache directory"
echo ""

# Setup wandb cache directory
WANDB_CACHE="${HOME}/metaworld_project/wandb_cache"
mkdir -p "${WANDB_CACHE}"

echo "W&B cache directory: ${WANDB_CACHE}"
echo ""

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "ERROR: wandb is not installed in the container or environment!"
    echo "The container should have wandb pre-installed."
    exit 1
fi

# Login to wandb
echo "Please login to W&B:"
echo "(You'll need your API key from https://wandb.ai/authorize)"
echo ""
wandb login

echo ""
echo "✓ W&B setup complete!"
echo ""
echo "Configuration:"
echo "- W&B cache: ${WANDB_CACHE}"
echo "- Offline mode: enabled in training scripts"
echo "- Runs will be synced after job completion with: wandb sync <run_dir>"
echo ""
echo "Next steps:"
echo "1. Update .env.cluster with your paths"
echo "2. Run test job: sbatch test_simple.sh"
