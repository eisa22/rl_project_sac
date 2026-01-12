#!/bin/bash
# W&B API Key Setup (One-time)
# Saves API key to ~/.wandb_api_key (600 permissions)

set -e

echo "=========================================="
echo "W&B API Key Setup"
echo "=========================================="
echo ""
echo "1. Go to: https://wandb.ai/authorize"
echo "2. Copy your API key"
echo "3. Paste below (will be hidden)"
echo ""

read -sp "Enter W&B API Key: " WANDB_API_KEY
echo ""

# Save to home directory with restricted permissions
echo "$WANDB_API_KEY" > ~/.wandb_api_key
chmod 600 ~/.wandb_api_key

echo ""
echo "✅ W&B API key saved to ~/.wandb_api_key"
echo ""
echo "You can now run training jobs with W&B logging enabled."
echo ""
