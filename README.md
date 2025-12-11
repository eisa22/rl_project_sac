# Meta-World SAC (Multi-Task + Single-Task)

Vollständige Implementierung des **Soft Actor-Critic (SAC)** Algorithmus mit **CUDA-Support**, **Weights & Biases Logging** und **TU Wien dataLAB Cluster Deployment**.

Unterstützt **Single-Task (ML1)** und **Multi-Task (MT10)** Reinforcement Learning für Meta-World Umgebungen.

## 🚀 Features

- ✅ Custom SAC Implementation (McLean et al. 2025 spec)
- ✅ Multi-Task MT10 Training (10 Tasks parallel)
- ✅ Per-Task Replay Buffers mit Equal Sampling
- ✅ Large Critic Networks (1024³) für Multi-Task Scaling
- ✅ Weights & Biases Integration (Online + Offline Mode)
- ✅ **Cluster Deployment Ready** (SLURM/Singularity)
- ✅ Docker/Singularity Container (~15GB)
- ✅ GPU-Optimized für NVIDIA A40 (48GB VRAM)

---

## 📁 Repository-Struktur

```
rl_project_sac/
├── train_metaworld.py          # MT10 Training Script
├── sac_agent.py                # Custom SAC Implementation
├── play_metaworld.py           # Evaluation Script
├── requirements.txt            # Python Dependencies
├── Dockerfile                  # Container Build
├── docker/
│   └── cluster/               # ⭐ Cluster Deployment
│       ├── .env.cluster       # Cluster Configuration
│       ├── run_singularity.sh # Container Runner
│       ├── test_simple.sh     # Test Job
│       ├── train_mt10_test.sh # Short Training
│       ├── train_mt10_full.sh # Full Training
│       ├── CLUSTER_DEPLOYMENT.md  # Full Documentation
│       └── README.md
├── DEPLOYMENT_CHECKLIST.md    # Step-by-Step Deployment Guide
└── README.md                  # This file
```

---

## 🚀 Quick Start (Lokal)

### 1. Python-Umgebung erstellen
```bash
conda create -n metaworld_rl python=3.10
conda activate metaworld_rl
```

### 2. Benötigte Libraries installieren
```bash
pip install metaworld==2.* gymnasium wandb
```

### 3. PyTorch mit CUDA installieren

Wählen Sie die korrekte Version für Ihre GPU unter: https://pytorch.org/get-started/locally/

Beispiel für CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```


### 🧠 Training starten (Lokal)

**Test Command:**
```bash
python train_metaworld.py --run_name local_test_tiny --total_steps 10000
```

Das Training wird über `train_metaworld.py` gesteuert:

---

#### MT10 Multi-Task Training

Trainiert SAC auf **10 Tasks gleichzeitig**:

```bash
python train_metaworld.py \
    --run_name my_mt10_run \
    --total_steps 2000000 \
    --seed 42
```

**Verfügbare MT10 Tasks:**
- reach-v2, push-v2, pick-place-v2, door-open-v2, drawer-open-v2
- drawer-close-v2, button-press-topdown-v2, peg-insert-side-v2
- window-open-v2, window-close-v2

---

## 🖥️ Cluster Deployment (TU Wien dataLAB)

**Vollständige Anleitung:** [`docker/cluster/CLUSTER_DEPLOYMENT.md`](docker/cluster/CLUSTER_DEPLOYMENT.md)  
**Deployment Checklist:** [`DEPLOYMENT_CHECKLIST.md`](DEPLOYMENT_CHECKLIST.md)

### Quick Start (Cluster)

```bash
# 1. Container bauen & konvertieren
./build_docker.sh
./convert_to_singularity.sh

# 2. Upload
scp sac_metaworld.sif datalab:/share/e11704784/containers/
rsync -avP . datalab:/home/e11704784/metaworld_project/source/rl_project_sac/

# 3. Setup
ssh datalab
mkdir -p /home/e11704784/metaworld_project/{logs,models,wandb_cache}

# 4. Test
cd /home/e11704784/metaworld_project/source/rl_project_sac/docker/cluster
sbatch test_simple.sh

# 5. Training
sbatch train_mt10_full.sh  # 2M steps, ~10h on A40
```

**Features:**
- ✅ SLURM Integration
- ✅ Singularity/Apptainer Container
- ✅ GPU-optimiert für A40 (48GB VRAM)
- ✅ W&B Offline Mode
- ✅ Automatic Checkpointing
- ✅ Based on Isaac Lab Lessons Learned

---

### 📊 Weights & Biases Setup

```bash
# Lokal
wandb login

# Cluster (offline mode)
# → Kein Login nötig!
# Nach Training: wandb sync
```

## 📁 Projektdateien – Übersicht

### `train_metaworld.py`
Das Haupt-Trainingsskript.  
Es ermöglicht:

- **Single-Task Training (ML1)** z. B. `reach-v3`, `push-v3`
- **Multi-Task Training (MT10)** mit 10 Tasks gleichzeitig
- automatisches Logging in **Weights & Biases**
- Ausführen von SAC-Updates und regelmäßiger Evaluation

Dieses Skript wird genutzt, um neue Modelle zu trainieren.

---

### `sac_agent.py`
Implementiert den eigentlichen **Soft Actor-Critic (SAC)** Algorithmus:

- Actor-Netzwerk (Policy)
- zwei große Critic-Netzwerke (Q-Funktionen)
- Target Networks
- Replay Buffer
- Entropy-Tuning
- CUDA-Support
- Logging der Trainingsmetriken

Dieses File enthält die lernenden Komponenten des Agents.

---

### `play_metaworld.py` 
Skript zur **Evaluation eines trainierten Modells**:

- lädt ein gespeichertes SB3-Modell (SAC/TD3/DDPG)
- führt mehrere Episoden im ausgewählten Meta-World Task aus
- zeigt das Verhalten im **Rendering-Fenster**
- misst Erfolgsrate, Rewards und Steps

Perfekt, um schnell zu testen, wie gut ein Modell gelernt hat.


### 2. Konfiguration

    Projektname: Robot_learning_2025

    Run-Name: Wird über das Argument --run_name gesetzt. Bitte nutzen Sie Ihren eigenen, eindeutigen Run-Namen!

        Beispiele: --run_name samuel_bigcritic_test, --run_name lukas_actor_small
        beispiel mt10: python train_metaworld.py --run_name samuel_mt10_run

### 3. Geloggte Metriken
Kategorie	Metriken
Trainingsmetriken	q1_loss, q2_loss, actor_loss, alpha
Single-Task Eval	eval_avg_return, eval_success_rate
Multi-Task Eval	task_name_avg_return, task_name_success_rate (für jeden Task separat) und mean_success_all_tasks

Gerne anpassen :)