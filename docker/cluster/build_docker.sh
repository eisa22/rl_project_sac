#!/bin/bash
# Build Docker image for MTMH-SAC Training
# Creates image: sac_mtmh:latest (~8GB)

set -e

echo "=========================================="
echo "Building Docker Image: sac_mtmh:latest"
echo "=========================================="
echo ""

# Check Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Dockerfile not found!"
    echo "   Run this script from repository root"
    exit 1
fi

# Build
echo "🐳 Building Docker image..."
docker build -t sac_mtmh:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -f Dockerfile .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Build Complete!"
echo "=========================================="
echo ""

# Show image info
IMAGE_ID=$(docker images -q sac_mtmh:latest)
IMAGE_SIZE=$(docker images sac_mtmh:latest --format "{{.Size}}")

echo "Image ID: $IMAGE_ID"
echo "Size: $IMAGE_SIZE"
echo ""
echo "Next step: bash docker/cluster/convert_to_singularity.sh"
echo ""
