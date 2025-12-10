#!/usr/bin/env bash
# =============================================================================
# Meta-World SAC Singularity Runner
# Based on Isaac Lab cluster deployment (mit allen Fixes!)
# =============================================================================
#
# Usage: run_singularity.sh <project_dir> <profile> <python_script> [args...]
#
# Example:
#   bash run_singularity.sh \
#       /home/e11704784/metaworld_project/source/rl_project_sac \
#       base \
#       train_metaworld.py \
#       --run_name cluster_test \
#       --total_steps 100000
#
# KRITISCH: Argument-Passing mit printf '%q' (siehe Isaac Lab Lessons Learned)
# =============================================================================

set -e

# Arguments
PROJECT_DIR="${1}"
PROFILE="${2}"
PYTHON_SCRIPT="${3}"
shift 3  # Remaining args are Python script arguments

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${SCRIPT_DIR}/.env.cluster"
source "${SCRIPT_DIR}/.env.base"
set +a

echo "========================================"
echo "Meta-World SAC Singularity Runner"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-LOCAL}"
echo "Project Dir: ${PROJECT_DIR}"
echo "Profile: ${PROFILE}"
echo "Python Script: ${PYTHON_SCRIPT}"
echo "Arguments: $@"
echo "Container: ${CLUSTER_FULL_SIF_PATH}"
echo "========================================"

# Validate container exists
if [ ! -f "${CLUSTER_FULL_SIF_PATH}" ]; then
    echo "❌ Error: Container not found at ${CLUSTER_FULL_SIF_PATH}"
    exit 1
fi

# Create Job Work Directory (Critical: auf /share für Space!)
if [ -n "${SLURM_JOB_ID}" ]; then
    JOB_WORK_DIR="${CLUSTER_WORK_ROOT}/${SLURM_JOB_ID}"
else
    JOB_WORK_DIR="${CLUSTER_WORK_ROOT}/local_$(date +%s)"
fi

echo "Job Work Dir: ${JOB_WORK_DIR}"
mkdir -p "${JOB_WORK_DIR}"

# Replicate code to job work dir
echo "Replicating code..."
rsync -a --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='wandb' \
    --exclude='logs' \
    --exclude='models_mt10' \
    "${PROJECT_DIR}/" "${JOB_WORK_DIR}/rl_project_sac/"

# Create necessary directories
mkdir -p "${JOB_WORK_DIR}/logs"
mkdir -p "${JOB_WORK_DIR}/models"
mkdir -p "${JOB_WORK_DIR}/wandb_cache"
# Ensure bind targets exist inside the container workspace
mkdir -p "${JOB_WORK_DIR}/rl_project_sac/logs" "${JOB_WORK_DIR}/rl_project_sac/models_mt10"

# Critical: Apptainer TMP auf /share (Isaac Lab Lesson!)
export APPTAINER_TMPDIR="${JOB_WORK_DIR}/apptainer_tmp"
export APPTAINER_CACHEDIR="${JOB_WORK_DIR}/apptainer_cache"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

echo "Apptainer TMP: ${APPTAINER_TMPDIR}"
echo "Apptainer Cache: ${APPTAINER_CACHEDIR}"

# Create MuJoCo cache directory (persistent)
mkdir -p "${CLUSTER_CACHE_ROOT}/mujoco_cache"

# Bind Mounts
BIND_MOUNTS=""
BIND_MOUNTS="${BIND_MOUNTS} -B ${JOB_WORK_DIR}/rl_project_sac:${DOCKER_WORKSPACE_PATH}"
BIND_MOUNTS="${BIND_MOUNTS} -B ${CLUSTER_LOGS_DIR}:${DOCKER_WORKSPACE_PATH}/logs"
BIND_MOUNTS="${BIND_MOUNTS} -B ${CLUSTER_MODELS_DIR}:${DOCKER_WORKSPACE_PATH}/models_mt10"
BIND_MOUNTS="${BIND_MOUNTS} -B ${JOB_WORK_DIR}/wandb_cache:/root/.cache/wandb"
BIND_MOUNTS="${BIND_MOUNTS} -B ${CLUSTER_CACHE_ROOT}/mujoco_cache:/root/.mujoco"

echo "Bind Mounts configured"

# Build Python command with properly quoted arguments (KRITISCH!)
PYTHON_CMD="${DOCKER_PYTHON_EXECUTABLE} ${PYTHON_SCRIPT}"
for arg in "$@"; do
    PYTHON_CMD="${PYTHON_CMD} $(printf '%q' "$arg")"
done

echo "Python Command: ${PYTHON_CMD}"
echo "========================================"
echo ""

# Environment variables for container
ENV_VARS=""
ENV_VARS="${ENV_VARS} PYTHONUNBUFFERED=1"
ENV_VARS="${ENV_VARS} MUJOCO_GL=${MUJOCO_GL}"
ENV_VARS="${ENV_VARS} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
ENV_VARS="${ENV_VARS} WANDB_MODE=${WANDB_MODE}"
ENV_VARS="${ENV_VARS} WANDB_DIR=/root/.cache/wandb"

# Run container
echo "🚀 Starting container..."
${CLUSTER_SINGULARITY_BIN} exec \
    --nv \
    ${BIND_MOUNTS} \
    --pwd ${DOCKER_WORKSPACE_PATH} \
    --cleanenv \
    ${CLUSTER_FULL_SIF_PATH} \
    bash -c "
        export ${ENV_VARS}
        
        # Set additional TMP vars inside container
        export TMPDIR=/tmp
        export TEMP=/tmp
        export TMP=/tmp
        
        echo \"[Container] Environment ready\"
        echo \"[Container] Python: \$(which python)\"
        echo \"[Container] CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')\"
        echo \"[Container] Working directory: \$(pwd)\"
        echo \"[Container] Executing: ${PYTHON_CMD}\"
        echo \"\"
        
        # Execute Python script
        ${PYTHON_CMD}
        exit_code=\$?
        
        echo \"\"
        echo \"[Container] Script finished with exit code: \${exit_code}\"
        exit \${exit_code}
    "

EXIT_CODE=$?

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Job succeeded. Exit code: ${EXIT_CODE}"
else
    echo "❌ Job failed. Exit code: ${EXIT_CODE}"
fi

# Optionally backup logs/models from work dir to project dir
if [ -n "${SLURM_JOB_ID}" ] && [ ${EXIT_CODE} -eq 0 ]; then
    echo "Backing up results..."
    rsync -a "${JOB_WORK_DIR}/logs/" "${CLUSTER_LOGS_DIR}/" 2>/dev/null || true
    rsync -a "${JOB_WORK_DIR}/models/" "${CLUSTER_MODELS_DIR}/" 2>/dev/null || true
fi

# Cleanup (optional: comment out for debugging)
# rm -rf "${JOB_WORK_DIR}"

echo "(run_singularity.sh): Finished with exit code ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
