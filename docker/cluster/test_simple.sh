#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a40:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --job-name="sac-test"
#SBATCH --output=/home/%u/metaworld_project/logs/test_%j.log
#SBATCH --error=/home/%u/metaworld_project/logs/test_%j.err

# =============================================================================
# Simple Test Job - Meta-World SAC
# Tests: Container, GPU, Python imports, MuJoCo, Basic environment
# =============================================================================

echo "========================================="
echo "Meta-World SAC - Simple Test Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "========================================="

# Load cluster config
ACTUAL_SCRIPT_DIR="/home/${USER}/metaworld_project/source/rl_project_sac/docker/cluster"
set -a
source "${ACTUAL_SCRIPT_DIR}/.env.cluster"
set +a

# Create simple test script
TEST_SCRIPT="${CLUSTER_SAC_DIR}/test_cluster.py"

cat > "${TEST_SCRIPT}" << 'EOF'
"""Simple test script for cluster validation"""
import sys
print("[TEST] Python executable:", sys.executable, flush=True)
print("[TEST] Python version:", sys.version, flush=True)

print("\n[TEST] Testing imports...", flush=True)
import torch
print(f"[TEST] PyTorch: {torch.__version__}", flush=True)
print(f"[TEST] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[TEST] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[TEST] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

import numpy as np
print(f"[TEST] NumPy: {np.__version__}", flush=True)

import metaworld
print(f"[TEST] Meta-World: {metaworld.__version__}", flush=True)

import gymnasium as gym
print(f"[TEST] Gymnasium: {gym.__version__}", flush=True)

print("\n[TEST] Creating Meta-World MT10 environment...", flush=True)
mt10 = metaworld.MT10()
print(f"[TEST] MT10 tasks available: {len(mt10.train_classes)}", flush=True)
print(f"[TEST] Task names: {list(mt10.train_classes.keys())}", flush=True)

print("\n[TEST] Testing single environment...", flush=True)
env_name = 'reach-v2'
env_cls = mt10.train_classes[env_name]
env = env_cls(render_mode=None)
env.reset()
print(f"[TEST] Environment '{env_name}' created successfully", flush=True)
print(f"[TEST] Observation space: {env.observation_space.shape}", flush=True)
print(f"[TEST] Action space: {env.action_space.shape}", flush=True)

print("\n[TEST] Running 10 simulation steps...", flush=True)
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 5 == 0:
        print(f"[TEST] Step {i}: reward={reward:.3f}", flush=True)

env.close()

print("\n✅ [TEST] All tests passed successfully!", flush=True)
EOF

# Run test
bash "${CLUSTER_SAC_DIR}/docker/cluster/run_singularity.sh" \
    "${CLUSTER_SAC_DIR}" \
    "base" \
    "test_cluster.py"

EXIT_CODE=$?

echo ""
echo "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Test job completed successfully"
else
    echo "❌ Test job failed with exit code ${EXIT_CODE}"
fi
echo "========================================="

exit ${EXIT_CODE}
