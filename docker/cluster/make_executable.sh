#!/bin/bash
# Make all scripts executable

set -e

echo "🔧 Making scripts executable..."

cd "$(dirname "$0")"

chmod +x build_docker.sh
chmod +x convert_to_singularity.sh
chmod +x upload_to_cluster.sh
chmod +x setup_wandb.sh
chmod +x train_mt3_mtmh.sh

echo "✅ All scripts are now executable"
echo ""
echo "You can now run:"
echo "  1. bash docker/cluster/build_docker.sh"
echo "  2. bash docker/cluster/convert_to_singularity.sh"
echo "  3. bash docker/cluster/upload_to_cluster.sh <tu-username>"
echo ""
