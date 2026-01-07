# Task Embeddings Implementation

Diese Implementierung erweitert den SAC Agent um **learned task embeddings** als Alternative zu One-Hot Encoding.

## 📁 Neue Dateien

| Datei | Beschreibung |
|-------|--------------|
| `sac_agent_embeddings.py` | SAC Agent mit Task Embedding Layers |
| `train_metaworld_embeddings.py` | MT10 Training mit Task Embeddings |
| `test_embeddings.py` | Test-Suite zur Validierung |
| `analyze_embeddings.py` | Visualisierung der gelernten Embeddings |

---

## 🎯 Was ist neu?

### **Vorher: One-Hot Encoding**
```python
# Observation: 39D state + 10D one-hot = 49D
reach     → [robot_state (39D), [1,0,0,0,0,0,0,0,0,0]]
push      → [robot_state (39D), [0,1,0,0,0,0,0,0,0,0]]
pick-place → [robot_state (39D), [0,0,1,0,0,0,0,0,0,0]]
```

### **Nachher: Task Embeddings**
```python
# Observation: 39D state (pure!)
# Task ID: Integer (0, 1, 2, ...)
# Embedding: Learned 16D vector per task

reach     → obs (39D) + task_id=0 → embedding_layer[0] → [0.8, 0.1, ..., 0.5] (16D)
push      → obs (39D) + task_id=1 → embedding_layer[1] → [0.7, 0.3, ..., 0.4] (16D)
pick-place → obs (39D) + task_id=2 → embedding_layer[2] → [-0.2, 0.9, ..., 0.1] (16D)

# Ähnliche Tasks bekommen ähnliche Embeddings! (gelernt)
```

---

## 🚀 Quick Start

### 1. Test Installation
```bash
python test_embeddings.py
```
Das sollte ausgeben: `ALL TESTS PASSED! ✅`

### 2. Training starten
```bash
# Standard (16D Embeddings)
python train_metaworld_embeddings.py --run_name my_embedding_run

# Mit verschiedenen Embedding-Dimensionen
python train_metaworld_embeddings.py --run_name emb_8d --embedding_dim 8
python train_metaworld_embeddings.py --run_name emb_32d --embedding_dim 32

# Kurzer Test-Run
python train_metaworld_embeddings.py --run_name test --total_steps 100000
```

### 3. Embeddings analysieren
```bash
# Nach Training
python analyze_embeddings.py --model_path ./models_mt10_embeddings/my_embedding_run/final_model.pt

# Plots speichern statt anzeigen
python analyze_embeddings.py --model_path ./models_mt10_embeddings/*/final_model.pt --save_plots
```

---

## 📊 Vergleich: One-Hot vs Embeddings

| Aspekt | One-Hot | Embeddings |
|--------|---------|-----------|
| **Obs Dimension** | 49D (39 + 10) | 39D (pure) |
| **Task Encoding** | 10D fest | 16D gelernt |
| **Parameter** | Keine extra | 10×16×3 = 480 (actor + 2 critics) |
| **Skalierung** | O(n) tasks | O(1) - fest embedding_dim |
| **Transfer** | Kein | Ja - zwischen ähnlichen Tasks |
| **Interpretierbar** | Nein | Ja - via t-SNE/PCA |

---

## 🧠 Technische Details

### **Embedding Layer**

```python
# Initialisierung
task_embedding = nn.Embedding(
    num_embeddings=10,  # MT10 tasks
    embedding_dim=16    # Kompakte Repräsentation
)

# Shape der Weight Matrix: [10, 16]
# Jede Zeile ist die Repräsentation einer Task

# Forward Pass
task_id = torch.tensor([3])  # z.B. "door-open-v2"
embedding = task_embedding(task_id)  # → [16] dimensional vector

# Das ist ein LOOKUP, kein Dense Layer!
# embedding = task_embedding.weight[task_id]
```

### **Gradient Flow**

```python
# Nur die Embeddings der GENUTZTEN Tasks werden upgedated!

# Batch mit task_ids = [0, 1, 2, 1, 0, 3, ...]
loss.backward()

# Gradient nur für benutzte Task IDs:
# task_embedding.weight.grad[0] ≠ 0  (genutzt)
# task_embedding.weight.grad[1] ≠ 0  (genutzt)
# task_embedding.weight.grad[2] ≠ 0  (genutzt)
# task_embedding.weight.grad[3] ≠ 0  (genutzt)
# task_embedding.weight.grad[4] = 0  (nicht genutzt)
# ...
```

---

## 🔬 Was die Embeddings lernen

Nach Training repräsentieren die Embedding-Dimensionen verschiedene Task-Properties:

**Hypothetische gelernte Features:**
- **Dimension 0:** "Benötigt Greifen?" (hoch für pick-place, niedrig für reach)
- **Dimension 1:** "Horizontale Bewegung?" (hoch für push, niedrig für vertical tasks)
- **Dimension 2:** "Objekt-Manipulation?" (hoch für alle außer reach)
- **Dimension 3:** "Präzision erforderlich?" (hoch für peg-insert, niedrig für push)
- ...

**Das Netzwerk entscheidet selbst, was wichtig ist!**

---

## 📈 Erwartete Ergebnisse

### **Task Clustering**

Nach Training sollten ähnliche Tasks in der Visualisierung nahe beieinander sein:

**Gruppe 1: Simple Reaching**
- `reach-v2`

**Gruppe 2: Push/Pull**
- `push-v2`
- `door-open-v2`
- `drawer-open-v2`

**Gruppe 3: Precision Manipulation**
- `pick-place-v2`
- `peg-insert-side-v2`

**Gruppe 4: Window Tasks**
- `window-open-v2`
- `window-close-v2`

### **Performance**

- **Ähnlich zu One-Hot** bei ausreichender Embedding-Dimension (≥16)
- **Potentiell besser** durch Transfer-Learning zwischen ähnlichen Tasks
- **Schnellere Konvergenz** möglich bei verwandten Tasks

---

## 🎨 Visualisierungen

`analyze_embeddings.py` erstellt:

1. **t-SNE Plot** (2D): Non-lineare Projektion der Embeddings
   - Zeigt Task-Clustering
   - Nahe Punkte = ähnliche Tasks

2. **PCA Plot** (2D): Lineare Projektion
   - Zeigt Hauptvarianzrichtungen
   - Erste Komponente = wichtigste Unterscheidung

3. **Distance Matrix** (Heatmap): Paarweise Distanzen
   - Dunkle Werte = ähnlich
   - Helle Werte = unterschiedlich

4. **Dimension Analysis** (Bar Chart): Varianz pro Dimension
   - Hohe Varianz = informative Dimension
   - Niedrige Varianz = wenig genutzt

---

## 🔧 Hyperparameter Tuning

### **Embedding Dimension**

```bash
# Zu klein (8): Eventuell zu wenig Kapazität
python train_metaworld_embeddings.py --embedding_dim 8

# Sweet Spot (16): Gute Balance
python train_metaworld_embeddings.py --embedding_dim 16

# Größer (32): Mehr Kapazität
python train_metaworld_embeddings.py --embedding_dim 32

# Maximal (64): Wie One-Hot, aber learned
python train_metaworld_embeddings.py --embedding_dim 64
```

**Faustregel:** `embedding_dim ≈ sqrt(num_tasks) * 4`
- MT3: 8-16D
- MT10: 12-24D
- MT50: 28-48D

### **Learning Rate für Embeddings**

Aktuell: **Gleiche LR wie Actor/Critic** (3e-4)

Optional: Separate LR für Embeddings (meist nicht nötig)

---

## 🆚 Wann welche Methode?

### **One-Hot (train_metaworld.py)**
✅ Einfacher, bewährte Methode  
✅ Keine extra Hyperparameter  
✅ Für wenige Tasks (≤10)  
❌ Skaliert schlecht (50 Tasks = 50D)  
❌ Kein Transfer-Learning  

### **Embeddings (train_metaworld_embeddings.py)**
✅ Skaliert gut (konstante Dimension)  
✅ Lernt Task-Similarities  
✅ Transfer zwischen verwandten Tasks  
✅ Visualisierbar und interpretierbar  
❌ Ein Hyperparameter mehr (embedding_dim)  
❌ Leicht komplexere Implementierung  

---

## 📝 Code-Änderungen Übersicht

### **Environment**
```python
# Vorher
obs = concat([robot_state, one_hot])  # 49D
return obs, info

# Nachher
obs = robot_state  # 39D (pure!)
info["task_id"] = task_id  # Separat
return obs, info
```

### **Agent**
```python
# Vorher
policy(obs)  # obs enthält one-hot

# Nachher
policy(obs, task_id)  # task_id separat, lookup embedding intern
```

### **Training Loop**
```python
# Vorher
obs, info = env.reset()
action = agent.act(obs)  # obs ist 49D

# Nachher
obs, info = env.reset()
task_id = info["task_id"]  # Extract task_id
action = agent.act(obs, task_id)  # obs ist 39D, task_id separat
```

---

## 🐛 Troubleshooting

### **"RuntimeError: Expected tensor for argument #1"**
→ Stelle sicher, dass `task_id` ein **Integer** oder **torch.Long** Tensor ist, nicht Float!

### **"IndexError: index out of range"**
→ `task_id` muss zwischen 0 und `num_tasks-1` liegen

### **Embeddings ändern sich nicht**
→ Prüfe, ob `task_ids` korrekt im Batch sind: `batch['task_ids']`

### **Performance schlechter als One-Hot**
→ Versuche größere `embedding_dim` (16 → 32 → 64)

---

## 📚 Weiterführende Ideen

### **1. Shared Embeddings**
Actor und Critic nutzen aktuell **separate** Embeddings. Man könnte sie **teilen**:
```python
shared_embedding = nn.Embedding(num_tasks, embedding_dim)
actor.task_embedding = shared_embedding
critic.task_embedding = shared_embedding
```

### **2. Pre-trained Embeddings**
Embeddings aus einem vortrainierten Modell initialisieren:
```python
pretrained_emb = torch.load('pretrained_embeddings.pt')
actor.task_embedding.weight.data = pretrained_emb
```

### **3. Conditional Layer Normalization**
Statt Concat, Task-Embeddings für Layer Norm nutzen (FiLM):
```python
gamma = task_embedding_gamma(task_id)
beta = task_embedding_beta(task_id)
normalized = gamma * layer_norm(x) + beta
```

### **4. Meta-Learning**
Embedding als Meta-Information für Few-Shot Learning nutzen

---

## 📖 Literatur

**Task Embeddings in RL:**
- Schmidhuber (1997): Task Complexity and Transferability
- Rusu et al. (2016): Policy Distillation - Multi-Task Learning
- Teh et al. (2017): Distral: Multi-Task Reinforcement Learning

**Word Embeddings (Inspiration):**
- Mikolov et al. (2013): Word2Vec
- Pennington et al. (2014): GloVe

**Multi-Task RL:**
- McLean et al. (2025): Multi-Task RL Enables Parameter Scaling (euer Paper!)

---

## ✅ Checkliste für eigene Experimente

- [ ] Test-Suite läuft durch (`test_embeddings.py`)
- [ ] Baseline One-Hot trainiert (zum Vergleich)
- [ ] Embeddings mit verschiedenen Dimensionen getestet (8, 16, 32)
- [ ] Embeddings visualisiert und analysiert
- [ ] Task Clustering macht Sinn (ähnliche Tasks nahe beieinander)
- [ ] Performance dokumentiert (W&B)
- [ ] Ablation Study: Embedding vs One-Hot

**Viel Erfolg! 🚀**
