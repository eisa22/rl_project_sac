# DataLAB Cluster Deployment Quick Start

**Status:** ✅ Ready for DataLAB  
**Target:** `/home/e11704784/metaworld_project/` (overwrite old MTRL)  
**Containers:** `/share/e11704784/containers/`  
**IsaacLAB:** Protected at `/share/isaaclab/` + `/home/e11704784/isaaclab_project/`

---

## 🚀 3-Minute Quick Start

### Step 1: Build & Convert (Local Machine - 30 min)
```bash
cd /path/to/rl_project_sac

# Make scripts executable
bash docker/cluster/make_executable.sh

# Build Docker image
bash docker/cluster/build_docker.sh

# Convert to Singularity
bash docker/cluster/convert_to_singularity.sh

# Output: sac_mtmh.sif (~8GB)
```

### Step 2: Upload to Cluster (10 min)
```bash
# Upload .sif + code to DataLAB
bash docker/cluster/upload_to_cluster.sh e11704784

# This will:
# - Upload sac_mtmh.sif to /share/e11704784/containers/
# - Sync code to /home/e11704784/metaworld_project/
# - Create logs, models, wandb_cache directories
```

### Step 3: Cluster Setup (SSH, 5 min)
```bash
ssh e11704784@datalab

# Setup W&B API key (one-time)
bash /home/e11704784/metaworld_project/docker/cluster/setup_wandb.sh
```

### Step 4: Submit Training Job
```bash
# Submit MT3 MTMH-SAC training
sbatch /home/e11704784/metaworld_project/docker/cluster/train_mt3_mtmh.sh

# Check status
squeue -u e11704784
tail -f /home/e11704784/metaworld_project/logs/mt3_mtmh_*.log
```

---

## 📁 Cluster Directory Structure

```
/home/e11704784/
├── isaaclab_project/          ← PROTECTED (Do not touch!)
├── metaworld_project/         ← YOUR PROJECT (can overwrite)
│   ├── logs/                  ← SLURM output logs
│   ├── models/                ← Trained models
│   ├── wandb_cache/           ← W&B offline cache
│   ├── source/                ← (old MTRL code - gets overwritten)
│   ├── docker/cluster/        ← Training scripts
│   ├── train_mt3_mtmh.py      ← Main training script
│   ├── mtmh_sac.py            ← Algorithm implementation
│   └── ...

/share/e11704784/
├── containers/                ← Singularity images
│   ├── isaaclab.sif           ← PROTECTED (Do not touch!)
│   ├── sac_mtmh.sif           ← YOUR IMAGE (uploaded here)
│   └── ...
└── isaaclab_work/             ← PROTECTED (Do not touch!)
```

---

## 🎯 Commands Reference

### Local
```bash
# Make scripts executable
bash docker/cluster/make_executable.sh

# Build Docker
bash docker/cluster/build_docker.sh

# Convert to Singularity
bash docker/cluster/convert_to_singularity.sh

# Upload to cluster (overwrites old MTRL setup)
bash docker/cluster/upload_to_cluster.sh e11704784
```

### On Cluster
```bash
# Setup W&B (one-time)
ssh e11704784@datalab
bash ~/metaworld_project/docker/cluster/setup_wandb.sh

# Submit job
sbatch ~/metaworld_project/docker/cluster/train_mt3_mtmh.sh

# Monitor
squeue -u e11704784
tail -f ~/metaworld_project/logs/mt3_mtmh_*.log

# View results
https://wandb.ai/e11704784/
```

---

## 📊 Training Configuration

**Default Settings (from Paper):**
- Total Steps: 6M (2M per task × 3 tasks)
- Learning Rate: 3e-4
- Batch Size: 500
- Reward Scale: 1.0
- Gamma: 0.99
- Tau: 0.005
- GPU: A40 (12GB VRAM)
- Time Limit: 20 hours

**Customization:**
Edit `docker/cluster/train_mt3_mtmh.sh` to change any parameters before submitting.

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| "sac_mtmh.sif not found" | Run `convert_to_singularity.sh` locally first |
| "Permission denied" | `bash docker/cluster/make_executable.sh` |
| "W&B API key not found" | SSH to cluster, run `setup_wandb.sh` |
| "Job pending" | Check cluster load: `sinfo` |
| "CUDA not available" | Verify GPU partition: `sinfo -p GPU-a40` |
| "Old MTRL stuff still there" | `rm -rf /home/e11704784/metaworld_project/source` before upload |

---

## ⚠️ Important Notes

1. **IsaacLAB Protection:** Completely separate at `/share/isaaclab` + `/home/e11704784/isaaclab_project` - **Never touched**
2. **.sif Location:** Goes to `/share/e11704784/containers/` (DataLAB policy)
3. **Project Root:** `/home/e11704784/metaworld_project/` (old MTRL code gets overwritten)
4. **W&B Integration:** Logs automatically to your W&B project during training
5. **Code Mounting:** Code is bind-mounted read-only in container for safety

---

**Created:** January 9, 2026  
**For:** RL Project SAC MTMH Training on DataLAB  
**Overwrites:** Old MTRL setup in metaworld_project/  
**Protects:** IsaacLAB at /share/isaaclab and /home/e11704784/isaaclab_project
