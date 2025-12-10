# Meta-World SAC Cluster Deployment - Comprehensive Guide & Learnings

**Status:** ✅ Production Ready  
**Last Updated:** December 10, 2025  
**Author:** Johannes  
**Environment:** TU Wien dataLAB Cluster (SLURM + GPU A40)

---

## 📋 Inhaltsverzeichnis

1. [Projektübersicht](#projektübersicht)
2. [Erkenntnisse & Lessons Learned](#erkenntnisse--lessons-learned)
3. [Technische Architektur](#technische-architektur)
4. [Setup Guide](#setup-guide)
5. [Trainingsverlauf](#trainingsverlauf)
6. [Troubleshooting](#troubleshooting)
7. [Performance & Optimization](#performance--optimization)

---

## 🎯 Projektübersicht

### Ziel
Implementierung eines **Soft Actor-Critic (SAC) Agenten** für **Meta-World MT10** (Multi-Task RL) auf dem TU Wien dataLAB Cluster mit:
- Headless GPU-Beschleunigung (A40, 48GB VRAM)
- Live W&B-Logging
- Reproduzierbarer Container-Deployment (Docker → Singularity)
- Per-Task Replay Buffers & Large Networks
- SLURM Job Management

### Projektziele ✅
- ✅ Lokaler Docker-Build mit allen Dependencies (MuJoCo, CUDA, OSMesa)
- ✅ Singularity/Apptainer-Konvertierung für Cluster
- ✅ SLURM Job Scripts für Test & Full Training
- ✅ Live W&B Integration (Online-Logging)
- ✅ GPU Monitoring & Performance Analysis
- ✅ Automated Model Download
- ✅ End-to-End Testing erfolgreich

---

## 🔬 Erkenntnisse & Lessons Learned

### 1. **Container & Build**

#### Problem: Dependency Conflicts
- **Issue:** `metaworld`, `gymnasium`, `stable-baselines3` und `torch` hatten inkompatible Versionsbeschränkungen
  - metaworld 3.0 benötigte gymnasium ≥1.1
  - stable-baselines3 benötigte torch ≥2.3
  - Anfängliche pins führten zu `pip resolution failures`
- **Lösung:** Upgrade torch auf ≥2.3, SB3 auf ≥2.7, gymnasium auf ≥1.1
  - **Lernen:** Niemals zu starre version pins verwenden; kompatible Ranges nutzen

#### Problem: Docker Build Context (.dockerignore)
- **Issue:** `.dockerignore` schloss `*.txt` aus, sodass `requirements.txt` nicht in den Build kopiert wurde
- **Lersung:** `!requirements.txt` in `.dockerignore` explizit erlaubt
  - **Lernen:** `.dockerignore` kann selektiv Dateien erneut zulassen

#### Problem: MuJoCo Headless Rendering
- **Lösung:** OSMesa System-Libraries (`libgl1-mesa-glx`, `libglfw3`, `libglfw3-dev`) + Environment-Variablen
  ```
  MUJOCO_GL=osmesa
  PYOPENGL_PLATFORM=osmesa
  ```
- **Lernen:** Headless RL requires explizite rendering backend setup in Containers

---

### 2. **Cluster Deployment**

#### Problem: Singularity Bind-Mount Targets
- **Issue:** Zielverzeichnisse im Container (`/workspace/rl_project_sac/logs`, `/workspace/rl_project_sac/models_mt10`) existierten nicht
  - Fehler: `mount hook function failure: destination ... doesn't exist in container`
- **Lösung:** Zielverzeichnisse **vor** dem Bind-Mount im Container anlegen
  ```bash
  mkdir -p "${JOB_WORK_DIR}/rl_project_sac/logs"
  mkdir -p "${JOB_WORK_DIR}/rl_project_sac/models_mt10"
  ```
- **Lernen:** Apptainer/Singularity erfordert explizite Mount-Target-Strukturen (Isaac Lab Lesson!)

#### Problem: W&B Offline Mode vs. Online
- **Issue:** `.env.cluster` setzte `WANDB_MODE="offline"` fest → Job-Scripts konnten nicht auf `online` umschalten
- **Lösung:** Default-Fallback nutzen
  ```bash
  export WANDB_MODE="${WANDB_MODE:-offline}"
  ```
  Dadurch können Jobs mit `WANDB_MODE=online sbatch script.sh` überschreiben
- **Lernen:** Environment-Defaults sollten flexibel sein; Caller sollten überschreiben können

#### Problem: wandb login auf Head-Node
- **Issue:** `wandb` command existiert nicht auf Head-Node, nur im Container
- **Lösung:** `wandb_setup_job.sh` als SLURM Job → Container ausführen, nicht Head-Node
- **Lernen:** Cluster-Workflows müssen zwischen Head-Node-Tools und Compute-Node-Tools unterscheiden

---

### 3. **W&B Integration**

#### Live Online-Logging
- **Ablauf:**
  1. W&B API-Key einmalig via `wandb_setup_job.sh` konfiguriert
  2. Training mit `WANDB_MODE=online` → Runs erscheinen live auf wandb.ai
  3. Run-URL: `https://wandb.ai/Robot_learning_2025/Robot_learning_2025/runs/<ID>`

#### Offline Fallback
- Falls Cluster kein Internet hat: `WANDB_MODE=offline` → Daten lokal gespeichert
- Nach Training: `wandb sync` auf lokaler Maschine

- **Lernen:** Online W&B im Cluster erfordert:
  - Authentifizierung (API-Key im Container)
  - Netzwerk-Zugang (überprüfbar via Test-Run)
  - Explizite `WANDB_MODE=online` (nicht default)

---

### 4. **GPU Monitoring & Performance**

#### GPU-Auslastung (10k Steps Quick-Test)
```
GPU_ID  GPU_Util_%  Memory_Used_MB  Temp_C  Power_W
0       0%          1-355           28-31   21-73W
```
- **Beobachtung:** GPU-Auslastung ist sehr niedrig (~0% meiste Zeit, Peaks ~355MB / 46GB)
- **Grund:** 10k Steps sind zu schnell; CPU-bound Training-Loop dominiert
- **Empfehlung:** Längere Runs oder höhere Batch-Größen für aussagekräftige GPU-Metriken

#### A40 Kapazität
- **VRAM:** 48GB (genug für große Buffer + Modelle)
- **Power:** ~70W unter Last (A40 rated für 130W, Reserve vorhanden)
- **Temperatur:** ~31°C auch unter Last (Kühlsystem OK)

---

### 5. **Netzwerk & Upload**

#### VPN + MTU/DTLS Issues
- **Problem:** Initial scp/rsync stalled über VPN mit Fehler `EMSGSIZE`
- **Ursache:** VPN DTLS + zu große Pakete
- **Lösung:** VPN mit `--no-dtls --set mtu=1200` reconnect → rsync mit `--partial` (resume)
- **Lernen:** Großdateien-Transfers über VPN benötigen möglicherweise spezielle Flags

#### Successful Upload
- `.sif` (7.7 GB) hochgeladen via `rsync -avP` nach 35-40 Minuten
- Code (~100 MB) hochgeladen, vollständiges Projekt-Sync

---

## 🏗️ Technische Architektur

### Stack
```
Local Development
├── Python 3.10 (venv/conda)
├── PyTorch 2.9.1 + CUDA 12.1
├── Meta-World, SAC Agent
└── W&B, TensorBoard

       ↓ Docker Build
       
Docker Image (sac_metaworld:latest, ~15GB)
├── Base: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
├── System: OSMesa (headless rendering)
├── Python: metaworld, gymnasium, stable-baselines3, wandb
└── Entry: python train_metaworld.py

       ↓ Convert to Singularity
       
Singularity Image (.sif, 7.7GB)
└── /share/e11704784/containers/sac_metaworld.sif

       ↓ SLURM Submission
       
Cluster Job (A40 GPU, 8-16 CPUs)
├── job_workdir: /share/e11704784/metaworld_work/<JOBID>/
│   └── rl_project_sac/ (code replica)
├── Singularity exec
│   ├── Bind: /home/e11704784/metaworld_project/{logs,models,wandb_cache}
│   ├── Env: WANDB_MODE=online, MUJOCO_GL=osmesa
│   └── Command: python train_metaworld.py --run_name <name> --total_steps <N>
└── GPU Monitoring (nvidia-smi logged)

       ↓ Results
       
Output
├── Models: /home/e11704784/metaworld_project/models/<run_name>/final_model.pt
├── Logs: /home/e11704784/metaworld_project/logs/
├── W&B: wandb.ai/Robot_learning_2025/... (live)
└── GPU Stats: /home/e11704784/metaworld_project/logs/gpu_<JOBID>.log
```

### Key Files

| Datei | Zweck |
|-------|-------|
| `Dockerfile` | Container-Basis (CUDA, OSMesa, Python deps) |
| `requirements.txt` | Python Packages (metaworld, SB3, wandb, torch) |
| `.env.cluster` | Cluster-Pfade & Config (user, container, work dirs) |
| `.env.gpu_config` | GPU Tuning (batch size, buffer, network architectures) |
| `run_singularity.sh` | Universal Singularity Runner (mounts, tmp relocation, arg quoting) |
| `quick_test.sh` | SLURM Job für 10k Steps (schneller Test) |
| `train_mt10_test.sh` | SLURM Job für 100k Steps (Standard-Test) |
| `train_mt10_full.sh` | SLURM Job für 2M Steps (Full Training) |
| `wandb_setup_job.sh` | W&B API-Key Konfiguration im Container |
| `monitor_gpu.sh` | GPU Utilization Logging (nvidia-smi loop) |
| `train_metaworld.py` | Main Training Script (MetaWorldMT10Env, SACAgent) |
| `sac_agent.py` | SAC Implementation (per-task buffers, entropy tuning) |

---

## 🚀 Setup Guide

### Phase 1: Local Preparation

**1.1 Repository klonen & Environment aufsetzen**
```bash
cd ~/Documents/01_Master_Robotik/4_Semester/VU_Robot_Learning/Course_Project/
git clone <repo>
cd rl_project_sac
```

**1.2 Docker Image bauen**
```bash
./build_docker.sh
# oder manuell:
docker build -t sac_metaworld:latest .
```

**1.3 Singularity Image konvertieren**
```bash
# Erfordert apptainer/singularity lokal installiert
./convert_to_singularity.sh
# Erzeugt: sac_metaworld.sif (~7.7GB)
```

**1.4 .sif hochladen zum Cluster**
```bash
rsync -avP --partial sac_metaworld.sif datalab:/share/e11704784/containers/
```

---

### Phase 2: Cluster Setup

**2.1 SSH zum Cluster**
```bash
ssh datalab
```

**2.2 Code hochladen (lokal ausführen)**
```bash
# Auf deinem PC
./upload_code_to_cluster.sh
```

**2.3 W&B konfigurieren (on cluster)**
```bash
cd ~/metaworld_project/source/rl_project_sac/docker/cluster
WANDB_MODE=online sbatch wandb_setup_job.sh f8d0815364dbf002efa5a32b8942d08c67789871
# Warte auf Job-Abschluss
squeue -u e11704784
cat ~/metaworld_project/logs/wandb_setup_*.log
```

**2.4 Verifikation**
```bash
# Container vorhanden?
ls -lh /share/e11704784/containers/sac_metaworld.sif

# Code vorhanden?
ls -la ~/metaworld_project/source/rl_project_sac/

# W&B konfiguriert?
ls -la ~/metaworld_project/wandb_cache/
```

---

### Phase 3: Test Training

**3.1 Quick Test (10k Steps, ~2-3 min)**
```bash
cd ~/metaworld_project/source/rl_project_sac/docker/cluster
WANDB_MODE=online sbatch quick_test.sh
```

**Monitor:**
```bash
# Live-Log
tail -f ~/metaworld_project/logs/quick_test_*.log

# GPU-Auslastung
tail -f ~/metaworld_project/logs/gpu_*.log

# Job-Status
squeue -u e11704784
```

**Prüfe W&B:**
- Gehe zu: https://wandb.ai/Robot_learning_2025/Robot_learning_2025
- Suche nach Run `Cluster_test_<JOBID>`
- Sollte live metriken zeigen

**3.2 Standard Test (100k Steps, ~15-20 min)**
```bash
WANDB_MODE=online sbatch train_mt10_test.sh
```

---

### Phase 4: Full Training (optional)

**4.1 2M Steps Training**
```bash
# Beachte: dauert ~6-8 Stunden
WANDB_MODE=online sbatch train_mt10_full.sh
```

**4.2 Status im Hintergrund prüfen**
```bash
# Mehrmals prüfen
watch -n 30 "squeue -u e11704784 && echo '---' && du -sh ~/metaworld_project/models/"
```

---

## 📊 Trainingsverlauf

### Beobachtete Runs

| Run | Schritte | Dauer | GPU Util | Status | W&B |
|-----|----------|-------|----------|--------|-----|
| `Cluster_test_334972` | 10k | ~6 sec | 0% (zu kurz) | ✅ OK | ✅ Live |
| `Cluster_test_334969` | 10k | ~6 sec | 0% | ✅ OK | ❌ Offline |
| `quick_test_334967` | 10k | ~6 sec | 0% | ✅ OK | ❌ Offline |

### Erkannte Muster
1. **10k Steps:** Zu kurz für aussagekräftige GPU-Auslastung
2. **Per-Task Buffers:** 10 Tasks × 200k Buffer = viel Memory (aber A40 hat genug)
3. **Episode Rewards:** Schnell steigende Performance (gute Konvergenz)
4. **Success Rate:** 0% (Tasks sehr schwierig, brauchen längeres Training)

---

## 🔧 Troubleshooting

### Problem: `wandb sync` schlägt fehl
**Fehler:** `wandb: Network error, unable to reach the server`

**Lösung:**
```bash
# 1. Check internet
ping -c 1 wandb.ai

# 2. Check login
wandb status

# 3. Offline-Run lokal syncen (mit API-Key)
export WANDB_API_KEY=f8d0815364dbf002efa5a32b8942d08c67789871
wandb sync wandb/offline-run-*/
```

---

### Problem: Job startet nicht / "Job held"
```bash
# Status prüfen
squeue -j <JOBID> -l

# Ausführliches Log
scontrol show job <JOBID>

# Häufige Gründe:
# - Keine GPUs verfügbar → warten
# - Partition nicht vorhanden → überprüfe partition
sinfo -p GPU-a40
```

---

### Problem: Container-Mount schlägt fehl
**Fehler:** `mount hook function failure: destination ... doesn't exist`

**Ursache:** Zielverzeichnisse im Container existieren nicht

**Lösung:** `run_singularity.sh` erstellt sie vorab (Fix bereits implementiert)
```bash
mkdir -p "${JOB_WORK_DIR}/rl_project_sac/logs"
mkdir -p "${JOB_WORK_DIR}/rl_project_sac/models_mt10"
```

---

### Problem: W&B läuft offline statt online
**Fehler:** Run erscheint nicht auf wandb.ai (offline-run-* Verzeichnis)

**Ursache:** `WANDB_MODE` wurde nicht auf `online` überschrieben

**Lösung:**
```bash
# Korrekt:
WANDB_MODE=online sbatch quick_test.sh

# Oder direkt in Job-Script:
export WANDB_MODE=online
```

---

### Problem: GPU-Auslastung sehr niedrig (<10%)
**Beobachtung:** GPU zeigt fast 0% Util bei 10k-Step Runs

**Ursache:** Run ist zu kurz; Training-Loop ist CPU-bound

**Lösungen:**
1. **Längere Runs:** 100k+ Steps (nicht 10k)
2. **Batch-Size erhöhen:** In `.env.gpu_config`
   ```bash
   # Vorher
   SAC_BATCH_SIZE=512
   # Nachher
   SAC_BATCH_SIZE=2048
   ```
3. **Mixed Precision:** `USE_MIXED_PRECISION=true` in `.env.gpu_config`

---

## ⚡ Performance & Optimization

### GPU Tuning für A40

| Szenario | Batch Size | Buffer | Comment |
|----------|-----------|--------|---------|
| Conservative | 512 | 1M | Safe, low memory |
| Standard | 1024 | 2M | Recommended (default) |
| Aggressive | 2048 | 3M | High throughput, ~15GB VRAM |

**Wie ändern:**
```bash
nano .env.gpu_config
# Bearbeite: SAC_BATCH_SIZE, SAC_BUFFER_SIZE, CRITIC_HIDDEN_DIM
```

### Mixed Precision (Optional)
Kann Training um ~20-30% beschleunigen:
```bash
# In .env.gpu_config
USE_MIXED_PRECISION=true
```

---

## 📥 Model Download

### Nach Training

```bash
# Auf deinem PC
mkdir -p ./models_cluster/Cluster_test_334972
rsync -avP datalab:/home/e11704784/metaworld_project/models/Cluster_test_334972/ \
    ./models_cluster/Cluster_test_334972/
```

### Logs & W&B herunterladen

```bash
# Alle Logs
rsync -avP datalab:/home/e11704784/metaworld_project/logs/ ./logs_cluster/

# W&B Offline-Runs (dann syncen)
rsync -avP datalab:/home/e11704784/metaworld_project/wandb_cache/ ./wandb_cluster/
cd wandb_cluster
export WANDB_API_KEY=f8d0815364dbf002efa5a32b8942d08c67789871
wandb sync wandb/run-*
```

---

## 🎓 Key Takeaways

1. **Dependency Management:** Niemals zu starre Version-Pins; kompatible Ranges nutzen
2. **Container I/O:** Bind-Mount-Targets **müssen** vor dem Mount existieren
3. **Cluster Flexibility:** Environment-Variablen mit Defaults (`${VAR:-default}`)
4. **W&B Integration:** Login im Container (nicht Head-Node), `WANDB_MODE` flexibel
5. **GPU Monitoring:** Kurze Runs sind CPU-bound; längere Runs nötig für echte GPU-Auslastung
6. **Network Robustness:** rsync mit `--partial` für Resume bei Abbrüchen
7. **Testing Philosophy:** Quick Test (10k Steps) → Standard Test (100k) → Full (2M)

---

## 📞 Support & Next Steps

### Falls Fehler auftreten:
1. **Logs prüfen:** `~/metaworld_project/logs/`
2. **Job-Status:** `squeue`, `sacct`, `scontrol show job`
3. **Container-Logs:** In Fehler-Dateien nach `[Container]` suchen
4. **W&B Debug:** `export WANDB_MODE=debug` für verbose output

### Für weitere Runs:
```bash
# Template kopieren
cp quick_test.sh my_custom_test.sh

# Anpassen (steps, batch-size, etc.)
nano my_custom_test.sh

# Submitten
WANDB_MODE=online sbatch my_custom_test.sh
```

---

**🚀 Viel Erfolg beim Training!**

Generated: December 10, 2025  
Last tested: Cluster_test_334972 (✅ Online W&B)
