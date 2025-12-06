# Meta-World SAC (Multi-Task + Single-Task)

Dieses Repository enthält eine vollständige Implementierung des **Soft Actor-Critic (SAC)** Algorithmus mit **CUDA-Support** und **Weights & Biases (W&B) Logging**.

Es unterstützt sowohl **Single-Task Reinforcement Learning (ML1)** als auch **Multi-Task Reinforcement Learning (MT3)** für die Meta-World Umgebungen.

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

Das Training wird über `train_metaworld.py` gesteuert.  
Es gibt zwei Modi:

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

- **Single-Task Training (ML1)** z. B. `reach-v2`, `push-v2`
- **Multi-Task Training (MT3)** mit 3 Tasks gleichzeitig
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

    Projektname: metaworld-sac-mtrl

    Run-Name: Wird über das Argument --run_name gesetzt. Bitte nutzen Sie Ihren eigenen, eindeutigen Run-Namen!

        Beispiele: --run_name samuel_bigcritic_test, --run_name lukas_actor_small

### 3. Geloggte Metriken
Kategorie	Metriken
Trainingsmetriken	q1_loss, q2_loss, actor_loss, alpha
Single-Task Eval	eval_avg_return, eval_success_rate
Multi-Task Eval	task_name_avg_return, task_name_success_rate (für jeden Task separat) und mean_success_all_tasks

Gerne anpassen :)