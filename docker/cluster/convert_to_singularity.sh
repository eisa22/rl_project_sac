#!/bin/bash
# Convert Docker image to Singularity
# Creates: sac_mtmh.sif (~8GB)

set -e

echo "=========================================="
echo "Converting Docker → Singularity"
echo "=========================================="
echo ""

# Check docker image exists
if ! docker images | grep -q "sac_mtmh"; then
    echo "❌ Docker image sac_mtmh:latest not found!"
    echo "   Run: bash docker/cluster/build_docker.sh"
    exit 1
fi

# Check apptainer/singularity installed
if ! command -v apptainer &> /dev/null; then
    echo "❌ apptainer not found!"
    echo "   Install: apt-get install apptainer"
    exit 1
fi

echo "🔄 Converting sac_mtmh:latest → sac_mtmh.sif..."
apptainer build --fakeroot sac_mtmh.sif docker-daemon://sac_mtmh:latest

if [ $? -ne 0 ]; then
    echo "❌ Conversion failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Conversion Complete!"
echo "=========================================="
echo ""

# Show file info
ls -lh sac_mtmh.sif
echo ""
echo "Next step: bash docker/cluster/upload_to_cluster.sh e11704784"
echo ""
