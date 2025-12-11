#!/bin/bash
# Simple W&B setup for offline mode (no login required)
# W&B runs will be stored locally and can be synced later

set -e

echo "======================================"
echo "W&B Offline Mode Setup"
echo "======================================"
echo ""

WANDB_DIR="${HOME}/metaworld_project/wandb_cache"
mkdir -p "${WANDB_DIR}"

echo "✅ W&B directory created: ${WANDB_DIR}"
echo ""
echo "Configuration:"
echo "  Mode: OFFLINE (no login required)"
echo "  Storage: ${WANDB_DIR}"
echo ""
echo "How it works:"
echo "  1. Training runs save W&B data locally"
echo "  2. After training completes, sync runs manually:"
echo "     cd ${WANDB_DIR}"
echo "     wandb login  # only needed once for sync"
echo "     wandb sync wandb/run-*"
echo ""
echo "✅ Setup complete! No further action needed."
echo ""
echo "Start training with:"
echo "  sbatch quick_test.sh"
