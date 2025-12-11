# 🚀 SAC Meta-World Training - DataLAB Cluster Deployment Guide

Complete guide to deploy your SAC Meta-World training on the DataLAB cluster using Docker/Singularity.

---

## 📋 Prerequisites

### On Your Local Machine:
- Docker installed
- Apptainer (Singularity) installed
- SSH access to DataLAB cluster configured

### On DataLAB Cluster:
- SSH key setup
- W&B account with API key

---

## 🔧 Step 1: Build Docker Image (Local)

```bash
cd /home/johannes/Documents/01_Master_Robotik/4_Semester/VU_Robot_Learning/Course_Project/rl_project_sac

# Make build script executable
chmod +x build_docker.sh

# Build Docker image
./build_docker.sh
```

**Expected output:** Docker image `sac_metaworld:latest` (~8-10 GB)

### Test locally (optional):
```bash
docker run --gpus all -it sac_metaworld:latest bash
# Inside container:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import metaworld; print('Meta-World OK')"
exit
```

---

## 🔄 Step 2: Convert to Singularity (Local)

```bash
# Make conversion script executable
chmod +x convert_to_singularity.sh

# Convert Docker → Singularity
./convert_to_singularity.sh
```

**Expected output:** `sac_metaworld.sif` file (~8-10 GB)

**Note:** This step takes 10-20 minutes depending on your machine.

---

## 📤 Step 3: Upload to Cluster

```bash
# Replace 'username' with your DataLAB username
scp sac_metaworld.sif username@cluster.datalab.tuwien.ac.at:~/

# Also upload your code (if not using git)
scp -r . username@cluster.datalab.tuwien.ac.at:~/rl_project_sac/
```

---

## 🔑 Step 4: Configure W&B on Cluster

SSH into the cluster:
```bash
ssh username@cluster.datalab.tuwien.ac.at
```

Get your W&B API key from: https://wandb.ai/authorize

Edit the submit script:
```bash
cd ~/rl_project_sac
nano submit_cluster_job.sh
```

Replace this line:
```bash
WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
```

With your actual key:
```bash
WANDB_API_KEY="abc123your-actual-key-here"
```

Save and exit (Ctrl+X, Y, Enter)

---

## 🚀 Step 5: Submit Training Job

```bash
# Make submit script executable
chmod +x submit_cluster_job.sh

# Create logs directory
mkdir -p logs

# Submit job to SLURM
sbatch submit_cluster_job.sh
```

**Monitor your job:**
```bash
# Check job status
squeue -u $USER

# View output (replace JOBID with actual job ID)
tail -f logs/slurm_JOBID.out

# Check GPU usage
ssh compute-node-name  # Use node from squeue
nvidia-smi
```

---

## 📊 Step 6: Monitor Training

### Weights & Biases Dashboard:
- Go to: https://wandb.ai/your-team/Robot_learning_2025
- Find your run: `cluster_run_JOBID`
- Monitor: loss curves, rewards, success rates

### Check logs on cluster:
```bash
# View latest log
ls -lt logs/
tail -f logs/slurm_XXXXX.out
```

---

## 📥 Step 7: Download Results

After training completes:

```bash
# From your local machine:
scp -r username@cluster.datalab.tuwien.ac.at:~/rl_project_sac/logs ./logs_from_cluster
scp -r username@cluster.datalab.tuwien.ac.at:~/rl_project_sac/models_mt10 ./models_from_cluster
```

---

## ⚙️ Customizing Training Parameters

Edit `submit_cluster_job.sh` to change:

```bash
# Training duration
TOTAL_STEPS=2000000  # Change to 5000000 for longer training

# GPU allocation
#SBATCH --gres=gpu:1  # Change to gpu:2 for multi-GPU

# Memory
#SBATCH --mem=32G  # Increase if needed

# Time limit
#SBATCH --time=24:00:00  # Increase for longer runs
```

Or pass arguments directly:
```bash
singularity exec --nv \
    --bind $PWD:/workspace/rl_project_sac \
    $SINGULARITY_IMAGE \
    python /workspace/rl_project_sac/train_metaworld.py \
    --run_name my_custom_run \
    --total_steps 5000000 \
    --seed 123
```

---

## 🐛 Troubleshooting

### Issue: "CUDA not available"
**Solution:** Make sure `--nv` flag is used in singularity exec

### Issue: "No module named 'metaworld'"
**Solution:** Rebuild Docker image and reconvert to Singularity

### Issue: "W&B login failed"
**Solution:** Check WANDB_API_KEY in submit script

### Issue: "Permission denied"
**Solution:** Make scripts executable: `chmod +x *.sh`

### Issue: Job pending for long time
**Solution:** Check cluster load: `squeue`, may need to wait for resources

### Issue: "MuJoCo rendering error"
**Solution:** Already handled by MUJOCO_GL=osmesa in Dockerfile

---

## 📂 Directory Structure on Cluster

```
~/
├── sac_metaworld.sif          # Singularity image (~10GB)
└── rl_project_sac/            # Your code
    ├── train_metaworld.py
    ├── sac_agent.py
    ├── submit_cluster_job.sh
    ├── logs/                   # SLURM outputs
    │   └── slurm_XXXXX.out
    └── models_mt10/            # Trained models
        └── cluster_run_XXXXX/
            └── final_model.pt
```

---

## 🔄 Updating Code on Cluster

If you modify your Python code:

```bash
# Option 1: Use git (recommended)
ssh username@cluster.datalab.tuwien.ac.at
cd ~/rl_project_sac
git pull

# Option 2: Copy specific files
scp train_metaworld.py username@cluster.datalab.tuwien.ac.at:~/rl_project_sac/
scp sac_agent.py username@cluster.datalab.tuwien.ac.at:~/rl_project_sac/
```

**Note:** You don't need to rebuild the Docker/Singularity image for code changes!
The `--bind` flag in the submit script uses your local code.

---

## 📊 Expected Training Times

On DataLAB cluster with 1 GPU:

| Steps | Time (approx) |
|-------|---------------|
| 100k  | ~30 min       |
| 500k  | ~2.5 hours    |
| 1M    | ~5 hours      |
| 2M    | ~10 hours     |

---

## 🎯 Quick Reference Commands

```bash
# Check job status
squeue -u $USER

# Cancel job
scancel JOBID

# View job details
scontrol show job JOBID

# Check disk usage
du -sh ~/*

# Clean up old logs
rm logs/slurm_*.out

# Monitor live training
tail -f logs/slurm_$(squeue -u $USER -h -o %i).out
```

---

## 📞 Support

- **DataLAB Issues:** Matrix channel #gpu:tuwien.ac.at
- **Code Issues:** Check logs in `logs/slurm_*.out`
- **W&B Issues:** https://docs.wandb.ai/

---

## ✅ Checklist

Before submitting job:
- [ ] Docker image built successfully
- [ ] Singularity .sif file created
- [ ] .sif uploaded to cluster
- [ ] W&B API key set in submit script
- [ ] logs/ directory exists
- [ ] submit_cluster_job.sh is executable

After job starts:
- [ ] Job appears in squeue
- [ ] GPU is allocated (check squeue)
- [ ] Output appears in logs/slurm_*.out
- [ ] W&B run is visible in dashboard

---

Good luck with your training! 🚀
