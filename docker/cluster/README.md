# Meta-World SAC - Cluster Deployment Files

Vollständige SLURM/Singularity Konfiguration für TU Wien dataLAB Cluster.

## 📁 Dateien

| Datei | Beschreibung |
|-------|-------------|
| `.env.cluster` | Cluster-Pfade, User-Config |
| `.env.base` | Container-Profil (base) |
| `.env.gpu_config` | GPU-Tuning Parameter für A40 |
| `run_singularity.sh` | Universal Container Runner (mit Isaac Lab Fixes!) |
| `test_simple.sh` | SLURM Job: Import & Environment Test (~5 min) |
| `train_mt10_test.sh` | SLURM Job: Short Training (100k steps, ~20 min) |
| `train_mt10_full.sh` | SLURM Job: Full Training (2M steps, ~10h) |
| `CLUSTER_DEPLOYMENT.md` | Vollständige Dokumentation |

## 🚀 Quick Start

### 1. Docker → Singularity

```bash
# Im Projekt-Root:
cd /path/to/rl_project_sac

# Konvertieren
./convert_to_singularity.sh

# Upload
scp sac_metaworld.sif datalab:/share/e11704784/containers/
```

### 2. Code hochladen

```bash
rsync -avP \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='wandb' \
    --exclude='logs' \
    . datalab:/home/e11704784/metaworld_project/source/rl_project_sac/
```

### 3. Cluster-Setup

```bash
ssh datalab

# Verzeichnisse erstellen
mkdir -p /home/e11704784/metaworld_project/{logs,models,wandb_cache}
mkdir -p /share/e11704784/{containers,metaworld_cache,metaworld_work}

# Config anpassen
cd /home/e11704784/metaworld_project/source/rl_project_sac/docker/cluster
nano .env.cluster
# WICHTIG: CLUSTER_USER="e11704784" → dein User!
```

### 4. Test

```bash
sbatch test_simple.sh
tail -f /home/e11704784/metaworld_project/logs/test_*.log
```

### 5. Training

```bash
# Short test (100k steps)
sbatch train_mt10_test.sh

# Full training (2M steps)
sbatch train_mt10_full.sh
```

## ⚙️ Konfiguration

### User-Anpassung

In `.env.cluster` anpassen:
```bash
export CLUSTER_USER="e11704784"  # ← DEIN USER!
```

### GPU-Tuning

In SLURM-Skripten oder via `.env.gpu_config`:
```bash
# Conservative (Standard)
--batch_size 512
--buffer_size 2000000

# Aggressive (A40 voll ausnutzen)
--batch_size 2048
--buffer_size 10000000
```

## 📊 Monitoring

```bash
# Job Status
squeue -u $USER

# Live Logs
tail -f /home/e11704784/metaworld_project/logs/train_*.log

# GPU Usage (auf Compute Node)
ssh a-a40-o-X
nvidia-smi -l 1
```

## 📥 Results

```bash
# Von lokalem Rechner:
rsync -avP datalab:/home/e11704784/metaworld_project/models/ ./models/
rsync -avP datalab:/home/e11704784/metaworld_project/wandb_cache/ ./wandb/
```

## 🔍 Troubleshooting

Siehe `CLUSTER_DEPLOYMENT.md` für detailliertes Troubleshooting.

**Häufige Probleme:**
- ✅ "No space left" → Bereits gefixt (APPTAINER_TMPDIR auf /share)
- ✅ "Arguments not passed" → Bereits gefixt (printf '%q')
- ✅ "CUDA OOM" → Batch Size reduzieren
- ✅ "W&B login failed" → Offline mode (kein Login nötig)

## 📚 Dokumentation

**Vollständige Anleitung:** `CLUSTER_DEPLOYMENT.md`

**Basiert auf:** Isaac Lab Cluster Setup (alle Lessons Learned übernommen!)

---

**Status:** ✅ Deployment-Ready  
**Getestet:** Docker Build erfolgreich (15GB Image)  
**Nächster Schritt:** Singularity Conversion & Upload
