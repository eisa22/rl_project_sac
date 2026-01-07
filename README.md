# Meta-World SAC (Multi-Task + Single-Task)

Dieses Repository enthält eine vollständige Implementierung des **Soft Actor-Critic (SAC)** Algorithmus mit **CUDA-Support** und **Weights & Biases (W&B) Logging**.

Es unterstützt sowohl **Single-Task Reinforcement Learning (ML1)** als auch **Multi-Task Reinforcement Learning (MT3)** für die Meta-World Umgebungen.

### test Command zum starten: 
```bash
python train_metaworld.py --mode single --env reach-v3 --run_name [euer_name]_test_tiny
```

---

## 🚀 Installation

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


### 🧠 Training starten

Das Training wird über `train_metaworld.py` oder `train_mt3_curriculum.py` gesteurt.  
Es gibt drei Modi:

---

#### 1️⃣ Single-Task Training (ML1)

Trainiert SAC auf **einem einzelnen Task**.

Verfügbare Meta-World Tasks:

- `reach-v2`
- `push-v2`
- `pick-place-v2`

> **Hinweis:**  
> Der Parameter `--run_name samuel_reach_bigcritic` ist **nur ein Beispiel**.  
> Bitte tragt **euren eigenen Namen** ein. 
> Dadurch können die Runs eindeutig zugeordnet und korrekt in **Weights & Biases** getrackt werden.

Beispiel:
```bash
python train_metaworld.py --mode single --env reach-v2 --run_name samuel_reach_bigcritic
```

---

#### 2️⃣ MT3 Curriculum Learning

Trainiert SAC auf **3 Tasks mit Curriculum Learning**: `reach-v2 → push-v2 → pick-place-v2`

Das Training beginnt nur mit `reach-v2`. Wenn eine Task einen Erfolgs-Schwellenwert erreicht, wird die nächste Task freigeschaltet.

**Beispiele:**

```bash
# Standard-Training (1.5M steps, 60% reach / 50% push thresholds)
python train_mt3_curriculum.py --run_name thomas_mt3_test

# Custom Thresholds (strengere Anforderungen)
python train_mt3_curriculum.py --run_name mt3_strict --curriculum_thresholds 0.7 0.6 0.0

# Schneller Test mit lockeren Thresholds
python train_mt3_curriculum.py --run_name mt3_quick --total_steps 500_000 --curriculum_thresholds 0.5 0.4 0.0
```

**Parameter:**
- `--run_name`: Experiment-Name (erscheint in W&B)
- `--total_steps`: Gesamtzahl Trainingsschritte (default: 1,500,000)
- `--seed`: Random seed (default: 42)
- `--curriculum_thresholds`: Drei Werte für Success-Thresholds `[reach, push, pick-place]` (default: 0.6 0.5 0.0)

**Curriculum-Ablauf:**
1. **Phase 1**: Training nur auf `reach-v2` bis 60% Erfolgsrate
2. **Phase 2**: `reach + push` trainieren bis push 50% erreicht
3. **Phase 3**: Alle 3 Tasks (`reach + push + pick-place`)

**W&B Metriken:**
- `curriculum/num_active_tasks`: Anzahl aktiver Tasks
- `curriculum/task_unlocked`: Welche Task wurde freigeschaltet
- `curriculum/unlock_step`: Schritt bei dem Unlock erfolgte
- `train/task/{task_name}/success_rate`: Per-Task Erfolgsraten

### 📊 Weights & Biases (W&B) Setup
1. Login

In Weights and Biases (wandb) einloggen --> API Key ist im Browser im Projekt:
```bash

wandb login
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

### `train_mt3_curriculum.py`
Curriculum Learning für MT3 (reach → push → pick-place):

- beginnt mit nur einem Task (`reach-v2`)
- schaltet automatisch neue Tasks frei basierend auf Erfolgsrate
- trackt Per-Task Erfolgsraten über die letzten 50 Episoden
- nutzt gleiche SAC-Architektur wie `train_metaworld.py`
- speichert Curriculum-Status in Checkpoints
- 200k Buffer pro Task (600k total)

Ideal für **schrittweises Multi-Task Learning**.

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
---

## 🆕 Task Embeddings (NEU!)

**Alternative zu One-Hot Encoding**: Learned Task Embeddings für besseren Transfer zwischen Tasks.

### Quick Start
```bash
# 1. Tests laufen lassen
python test_embeddings.py

# 2. Training mit Embeddings
python train_metaworld_embeddings.py --run_name my_embedding_run --embedding_dim 16

# 3. Embeddings analysieren
python analyze_embeddings.py --model_path ./models_mt10_embeddings/my_embedding_run/final_model.pt
```

**Siehe [TASK_EMBEDDINGS_README.md](TASK_EMBEDDINGS_README.md) für vollständige Dokumentation!**

**Vorteile:**
- 📦 Kompakter als One-Hot (16D statt 10D für MT10)
- 🧠 Lernt Task-Ähnlichkeiten automatisch
- 🔄 Besserer Transfer zwischen verwandten Tasks  
- 📈 Skaliert besser zu vielen Tasks (MT50: 16D statt 50D)
- 🔍 Visualisierbar und interpretierbar

**Neue Dateien:**
- `sac_agent_embeddings.py` - SAC mit Task Embeddings
- `train_metaworld_embeddings.py` - MT10 Training
- `test_embeddings.py` - Test-Suite
- `analyze_embeddings.py` - Embedding-Visualisierung
