# Deployment Checklist - Meta-World SAC auf dataLAB Cluster

## ✅ Phase 1: Lokale Vorbereitung

- [x] Docker Image gebaut (`sac_metaworld:latest`, 15GB)
- [ ] Docker Image getestet lokal
- [ ] Singularity Conversion (`./convert_to_singularity.sh`)
- [ ] .sif File validiert (apptainer exec --nv sac_metaworld.sif python --version)

**Commands:**
```bash
cd /home/johannes/Documents/01_Master_Robotik/4_Semester/VU_Robot_Learning/Course_Project/rl_project_sac

# Optional: Lokal testen
docker run --gpus all -it sac_metaworld:latest bash
python -c "import torch, metaworld; print(torch.cuda.is_available())"

# Singularity Conversion
./convert_to_singularity.sh
# Output: sac_metaworld.sif (~15GB)

# Validieren
apptainer exec --nv sac_metaworld.sif python --version
```

---

## ✅ Phase 2: Upload zum Cluster

- [ ] Container hochgeladen (`/share/e11704784/containers/sac_metaworld.sif`)
- [ ] Code hochgeladen (`/home/e11704784/metaworld_project/source/rl_project_sac/`)
- [ ] Cluster-Skripte vorhanden (`docker/cluster/*.sh`)
- [ ] Permissions gesetzt (chmod +x *.sh)

**Commands:**
```bash
# Container Upload
scp sac_metaworld.sif datalab:/share/e11704784/containers/

# Code Upload
rsync -avP \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='wandb' \
    --exclude='logs' \
    --exclude='models_mt10' \
    . datalab:/home/e11704784/metaworld_project/source/rl_project_sac/

# Validieren
ssh datalab "ls -lh /share/e11704784/containers/sac_metaworld.sif"
ssh datalab "ls /home/e11704784/metaworld_project/source/rl_project_sac/docker/cluster/"
```

---

## ✅ Phase 3: Cluster-Setup

- [ ] Verzeichnisstruktur erstellt
- [ ] `.env.cluster` angepasst (CLUSTER_USER)
- [ ] Permissions validiert
- [ ] Container existiert und ist lesbar

**Commands:**
```bash
ssh datalab

# Verzeichnisse erstellen
mkdir -p /home/e11704784/metaworld_project/{logs,models,wandb_cache}
mkdir -p /share/e11704784/{containers,metaworld_cache,metaworld_work}
mkdir -p /share/e11704784/metaworld_cache/mujoco_cache

# Config anpassen
cd /home/e11704784/metaworld_project/source/rl_project_sac/docker/cluster
nano .env.cluster
# Ändern: export CLUSTER_USER="e11704784" → DEIN USER!

# Permissions
chmod -R 755 /home/e11704784/metaworld_project
chmod +x *.sh

# Validieren
ls -lh /share/e11704784/containers/sac_metaworld.sif
cat .env.cluster | grep CLUSTER_USER
```

---

## ✅ Phase 4: Test-Jobs

- [ ] Simple Test erfolgreich (test_simple.sh)
- [ ] Imports funktionieren (PyTorch, Meta-World, MuJoCo)
- [ ] GPU erkannt (CUDA available: True)
- [ ] Environment Creation erfolgreich

**Commands:**
```bash
cd /home/e11704784/metaworld_project/source/rl_project_sac/docker/cluster

# Test starten
sbatch test_simple.sh

# Job Status checken
squeue -u $USER

# Logs live folgen
tail -f /home/e11704784/metaworld_project/logs/test_*.log

# Erwartete Ausgabe:
# [TEST] CUDA available: True
# [TEST] CUDA device: NVIDIA A40
# [TEST] MT10 tasks available: 10
# ✅ [TEST] All tests passed successfully!
```

**Validierung:**
- Exit Code: 0
- Alle Imports erfolgreich
- GPU erkannt
- MT10 Environment erstellt

---

## ✅ Phase 5: Short Training Test

- [ ] Training-Test erfolgreich (train_mt10_test.sh)
- [ ] 100k steps durchlaufen (~15-20 min)
- [ ] Model gespeichert
- [ ] W&B Logs erstellt
- [ ] Keine CUDA OOM Errors

**Commands:**
```bash
sbatch train_mt10_test.sh

# Monitoring
watch -n 5 'squeue -u $USER'
tail -f /home/e11704784/metaworld_project/logs/train_test_*.log

# GPU Usage checken (nach Job-Start)
COMPUTE_NODE=$(squeue -u $USER -h -o %N)
ssh ${COMPUTE_NODE} nvidia-smi
```

**Validierung:**
- Training läuft durch (100k steps)
- Progress Bar updates
- Model saved: `/home/e11704784/metaworld_project/models/cluster_test_*/final_model.pt`
- W&B logs: `/home/e11704784/metaworld_project/wandb_cache/`

---

## ✅ Phase 6: Full Training

- [ ] Full Training gestartet (train_mt10_full.sh)
- [ ] 2M steps Parameter korrekt
- [ ] GPU-Auslastung optimal (>80%)
- [ ] Checkpoints werden gespeichert
- [ ] Kein Memory Leak

**Commands:**
```bash
sbatch train_mt10_full.sh

# Long-running Monitoring
watch -n 30 'squeue -u $USER'

# Periodisch GPU checken
COMPUTE_NODE=$(squeue -u $USER -h -o %N)
ssh ${COMPUTE_NODE} "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 10"

# Checkpoints validieren (nach 1-2h)
ls -lh /home/e11704784/metaworld_project/models/*/checkpoint_*.pt
```

**Erwartete Timeline:**
- 0-5 min: Container startup
- 5-10h: Training (2M steps)
- ~10h: Total

---

## ✅ Phase 7: Results & Cleanup

- [ ] Training erfolgreich abgeschlossen
- [ ] Final Model existiert
- [ ] Logs vollständig
- [ ] W&B Sync erfolgreich
- [ ] Lokales Backup erstellt

**Commands:**
```bash
# Results downloaden
rsync -avP datalab:/home/e11704784/metaworld_project/models/ ./trained_models_from_cluster/
rsync -avP datalab:/home/e11704784/metaworld_project/logs/ ./training_logs_from_cluster/
rsync -avP datalab:/home/e11704784/metaworld_project/wandb_cache/ ./wandb_from_cluster/

# W&B Sync (lokal)
cd wandb_from_cluster
wandb sync

# Optional: Cleanup auf Cluster
ssh datalab "rm -rf /share/e11704784/metaworld_work/*"
ssh datalab "rm /home/e11704784/metaworld_project/logs/train_*.log"
```

---

## 🚨 Troubleshooting Checklist

Falls Probleme auftreten:

### Container Issues
- [ ] Container existiert? `ls -lh /share/e11704784/containers/sac_metaworld.sif`
- [ ] Container lesbar? `apptainer exec ... sac_metaworld.sif echo OK`
- [ ] CUDA im Container? `apptainer exec --nv ... python -c "import torch; print(torch.cuda.is_available())"`

### Path Issues
- [ ] CLUSTER_USER korrekt in `.env.cluster`?
- [ ] Alle Verzeichnisse existieren?
- [ ] Permissions gesetzt (755)?

### GPU Issues
- [ ] GPU-Queue verfügbar? `sinfo -p GPU-a40`
- [ ] CUDA verfügbar im Container? `--nv` Flag gesetzt?
- [ ] Memory ausreichend? `nvidia-smi` auf Compute Node

### Training Issues
- [ ] Arguments werden übergeben? Check `Python Command` in Logs
- [ ] W&B offline mode? `WANDB_MODE=offline` in env?
- [ ] Checkpoints gespeichert? `ls models/*/`

---

## 📊 Success Criteria

### Minimal Success (Phase 4-5)
- ✅ Container startet
- ✅ Imports funktionieren
- ✅ GPU erkannt
- ✅ Short training (100k steps) läuft durch

### Full Success (Phase 6-7)
- ✅ Full training (2M steps) erfolgreich
- ✅ Checkpoints gespeichert
- ✅ W&B Logs vollständig
- ✅ Final Model funktioniert

---

## 📝 Notizen

**Lessons Learned from Isaac Lab:**
- ✅ APPTAINER_TMPDIR auf /share (nicht /tmp)
- ✅ Argument-Passing mit `printf '%q'`
- ✅ PYTHONUNBUFFERED=1 für Live-Output
- ✅ Bind-Mounts für Logs/Models

**Meta-World Specifics:**
- MuJoCo Cache ist klein (~100 MB)
- Leichtgewichtiger als Isaac Sim
- Mehr parallele Envs möglich
- Schnelleres Startup

---

**Deployment Date:** 2025-12-09  
**Status:** Ready for Execution  
**Next Action:** Run Phase 1 (Singularity Conversion)
