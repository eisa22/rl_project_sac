# Meta-World SAC Cluster Quick Start Guide

## 🚀 Setup auf dem Cluster (einmalig)

### 1. Code hochladen

```bash
# Lokal ausführen (auf deinem PC)
./upload_code_to_cluster.sh
```

### 2. Auf Cluster einloggen und W&B einrichten

```bash
# SSH zum Cluster
ssh datalab

# Navigiere zum Projekt
cd ~/metaworld_project/source/rl_project_sac/docker/cluster

# W&B Login (einmalig)
wandb login
# API Key von: https://wandb.ai/authorize

# Alternativ: setup script verwenden
bash setup_wandb.sh
```

### 3. Verifikation

```bash
# Prüfe, dass Container vorhanden ist
ls -lh /share/e11704784/containers/sac_metaworld.sif

# Prüfe Projekt-Struktur
ls -la ~/metaworld_project/
# Sollte zeigen: logs/ models/ wandb_cache/ source/
```

---

## 🧪 Test-Training durchführen

### Quick Test (10k steps, ~2-3 min)

```bash
cd ~/metaworld_project/source/rl_project_sac/docker/cluster
sbatch quick_test.sh
```

**Monitoring während des Jobs:**

```bash
# Job-Status prüfen
squeue -u e11704784

# Live-Log verfolgen
tail -f ~/metaworld_project/logs/quick_test_<JOB_ID>.log

# GPU-Auslastung ansehen (während Job läuft)
tail -f ~/metaworld_project/logs/gpu_<JOB_ID>.log
```

**Nach Job-Abschluss:**

```bash
# Vollständiges Log ansehen
cat ~/metaworld_project/logs/quick_test_<JOB_ID>.log

# GPU-Statistiken zusammenfassen
tail -n 50 ~/metaworld_project/logs/gpu_<JOB_ID>.log
```

### Standard Test (100k steps, ~15-20 min)

```bash
sbatch train_mt10_test.sh
```

---

## 🏋️ Vollständiges Training starten

### MT10 Full Training (2M steps, ~6-8 Stunden)

```bash
cd ~/metaworld_project/source/rl_project_sac/docker/cluster
sbatch train_mt10_full.sh
```

**Job-Management:**

```bash
# Alle deine Jobs anzeigen
squeue -u e11704784

# Job abbrechen (falls nötig)
scancel <JOB_ID>

# Job-Informationen
scontrol show job <JOB_ID>

# Ressourcennutzung nach Abschluss
sacct -j <JOB_ID> --format=JobID,JobName,Elapsed,State,MaxRSS,MaxVMSize
```

---

## 📊 W&B Sync (nach Job-Abschluss)

W&B läuft im Offline-Mode auf dem Cluster. Nach dem Training musst du die Runs manuell hochladen:

```bash
# Auf dem Cluster: alle offline runs syncen
cd ~/metaworld_project/wandb_cache
wandb sync wandb/run-*

# Oder spezifischen Run syncen
wandb sync wandb/run-<RUN_ID>

# Alle Runs auflisten
ls -la wandb/
```

---

## 💾 Modelle herunterladen (lokal ausführen)

### Alle Modelle runterladen

```bash
# Auf deinem PC
./download_models_from_cluster.sh
```

### Spezifischen Job herunterladen

```bash
./download_models_from_cluster.sh quick_test_12345
```

### Logs und W&B-Daten herunterladen

```bash
# Logs
rsync -avP datalab:/home/e11704784/metaworld_project/logs/ ./logs_cluster/

# W&B offline runs
rsync -avP datalab:/home/e11704784/metaworld_project/wandb_cache/ ./wandb_cluster/

# Dann lokal syncen
cd wandb_cluster
wandb sync wandb/run-*
```

---

## 🔍 GPU-Nutzung prüfen

### Während des Trainings

```bash
# Live GPU monitoring (auf Compute-Node während Job läuft)
watch -n 2 nvidia-smi

# Oder: GPU-Log-Datei verfolgen
tail -f ~/metaworld_project/logs/gpu_<JOB_ID>.log
```

### Nach dem Training

```bash
# GPU-Log auswerten
cat ~/metaworld_project/logs/gpu_<JOB_ID>.log

# Durchschnittliche Auslastung berechnen
awk -F',' 'NR>1 {sum+=$3; count++} END {print "Avg GPU Util: " sum/count "%"}' \
    ~/metaworld_project/logs/gpu_<JOB_ID>.log
```

---

## ⚙️ GPU-Tuning anpassen

Wenn die GPU-Auslastung niedrig ist (<50%), kannst du die Batch-Größen erhöhen:

```bash
cd ~/metaworld_project/source/rl_project_sac/docker/cluster

# Bearbeite .env.gpu_config
nano .env.gpu_config

# Ändere z.B.:
# SAC_BATCH_SIZE=1024 → 2048
# SAC_BUFFER_SIZE=2000000 → 3000000

# Dann Job neu starten
sbatch train_mt10_test.sh
```

---

## 🐛 Troubleshooting

### Job startet nicht

```bash
# Prüfe Job-Queue und Grund
squeue -u e11704784 -o "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %.20R"

# Prüfe Partition-Status
sinfo -p GPU-a40
```

### Container nicht gefunden

```bash
# Prüfe Container-Pfad
ls -lh /share/e11704784/containers/sac_metaworld.sif

# Falls fehlt: erneut hochladen (lokal)
rsync -avP sac_metaworld.sif datalab:/share/e11704784/containers/
```

### W&B funktioniert nicht

```bash
# Prüfe Login-Status
wandb status

# Erneut einloggen
wandb login

# Prüfe Offline-Runs
ls -la ~/metaworld_project/wandb_cache/wandb/
```

### Speicherplatz voll

```bash
# Speichernutzung prüfen
du -sh ~/metaworld_project/*
du -sh /share/e11704784/*

# Alte Logs/Models löschen
rm -rf ~/metaworld_project/logs/old_*
rm -rf ~/metaworld_project/models/old_*
```

---

## 📝 Wichtige Pfade

| Typ | Pfad |
|-----|------|
| Container | `/share/e11704784/containers/sac_metaworld.sif` |
| Projekt-Code | `~/metaworld_project/source/rl_project_sac` |
| Logs | `~/metaworld_project/logs/` |
| Modelle | `~/metaworld_project/models/` |
| W&B Cache | `~/metaworld_project/wandb_cache/` |
| Job Scripts | `~/metaworld_project/source/rl_project_sac/docker/cluster/` |

---

## 🎯 Typischer Workflow

```bash
# 1. Code-Änderungen lokal machen
# ... edit train_metaworld.py, sac_agent.py, etc. ...

# 2. Code hochladen
./upload_code_to_cluster.sh

# 3. SSH zum Cluster
ssh datalab

# 4. Schnellen Test starten
cd ~/metaworld_project/source/rl_project_sac/docker/cluster
sbatch quick_test.sh

# 5. Job-Status prüfen
squeue -u e11704784

# 6. Log verfolgen
tail -f ~/metaworld_project/logs/quick_test_*.log

# 7. Falls Test erfolgreich: Full Training
sbatch train_mt10_full.sh

# 8. Nach Abschluss: W&B syncen
cd ~/metaworld_project/wandb_cache
wandb sync wandb/run-*

# 9. Modelle runterladen (zurück auf deinem PC)
./download_models_from_cluster.sh

# 10. Lokal evaluieren
python play_metaworld.py --model_path models_cluster/cluster_test_12345/final_model.pt
```

---

## 🚀 Performance-Tipps

1. **GPU-Auslastung optimieren:**
   - Prüfe GPU-Log: sollte >70% sein
   - Falls niedrig: erhöhe Batch-Size in `.env.gpu_config`
   - Falls Out-of-Memory: reduziere Batch-Size oder Buffer-Size

2. **Parallele Umgebungen:**
   - Standardwert: 8 parallel envs
   - Bei niedriger CPU-Last: erhöhe auf 16 (falls genug CPUs)

3. **Mixed Precision:**
   - Aktiviere in `.env.gpu_config`: `USE_MIXED_PRECISION=true`
   - Kann Training um ~20-30% beschleunigen

4. **Checkpoint-Intervalle:**
   - Standard: alle 50k steps
   - Für lange Runs: reduziere auf 100k (spart I/O)

---

Viel Erfolg beim Training! 🎉
