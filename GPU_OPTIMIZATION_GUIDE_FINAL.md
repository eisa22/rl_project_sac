# GPU Optimization Guide - Meta-World SAC Training

**Paper:** McLean et al. 2025 - "Multi-Task RL Enables Parameter Scaling"

---

## 📋 **Paper Hyperparameters (FIXED)**

These parameters are from the Paper and **must not be changed**:

```python
# MT10 Configuration (McLean et al. 2025)
buffer_size = 2_000_000        # 200k × 10 tasks
batch_size = 512               # Standard SAC
learning_rate = 3e-4           # Standard SAC
actor_hidden = [256, 256]      # Baseline architecture
critic_hidden = [1024, 1024, 1024]  # Critic scaling!
```

**Key Insight from Paper:** Critic scaling > Actor scaling in MTRL!

---

## 🚀 **Training Strategies**

### **Strategy 1: Short Run (2M Steps)**
- **Purpose:** Quick experiments, hyperparameter tuning
- **Hardware:** A40 (48GB VRAM)
- **Script:** `train_mt10_2M_a40.sh`
- **Config:**
  - Parallel Envs: 32
  - Expected Time: 1-1.5h
  - GPU Util: 75-85%

### **Strategy 2: Full Run (20M Steps)** ⭐
- **Purpose:** Paper reproduction, final results
- **Hardware:** A100 (80GB VRAM)
- **Script:** `train_mt10_20M_a100.sh`
- **Config:**
  - Parallel Envs: 48
  - Expected Time: 8-10h
  - GPU Util: 80-90%

---

## 📊 **GPU Hardware Comparison**

| GPU | VRAM | Best For | Parallel Envs | Training Time (20M) |
|-----|------|----------|---------------|---------------------|
| A40 | 48GB | Short runs (2M) | 32 | 40-50h (nicht empfohlen!) |
| **A100** | **80GB** | **Full runs (20M)** | **48** | **8-10h** ✅ |
| L40S | 48GB | Alternative zu A40 | 32-40 | 35-45h |

---

## ⚙️ **Parallel Environment Options**

The **ONLY** tunable parameter for GPU optimization (Paper hyperparameters stay FIXED):

### **OPTION 1: Sequential (Baseline)**
```bash
export NUM_PARALLEL_ENVS=1
```
- GPU Util: 5-10%
- Training Time (20M): 80-100h
- **Use case:** Never (wasteful)

### **OPTION 2: Moderate (A40)**
```bash
export NUM_PARALLEL_ENVS=16
```
- GPU Util: 60-70%
- Training Time (20M): 40-50h
- **Use case:** Resource-conscious

### **OPTION 3: High (A40)** ⚡
```bash
export NUM_PARALLEL_ENVS=32
```
- GPU Util: 75-85%
- Training Time (2M): 1-1.5h
- Training Time (20M): 25-30h
- **Use case:** **2M steps on A40** ← CURRENT DEFAULT

### **OPTION 4: Maximum (A100)** 🔥 **← RECOMMENDED FOR 20M**
```bash
export NUM_PARALLEL_ENVS=48
```
- GPU Util: 80-90%
- VRAM: ~45-50GB / 80GB
- Training Time (20M): 8-10h ✅
- **Use case:** **20M steps on A100**

---

## 🎓 **Paper Insights (McLean et al. 2025)**

### **Key Finding: Critic Scaling > Actor Scaling**
- Scaling the **critic** network provides greater benefits than scaling the **actor**
- Reason: Critic must approximate multiple state-action value functions across diverse tasks
- **Recommendation:** Use [1024, 1024, 1024] critic (as in Paper)

### **Plasticity Loss Mitigation**
- **Problem:** Large models can suffer plasticity loss (neurons become dormant)
- **Solution:** Increase number of tasks **AND** parameters simultaneously
- MT10 (10 tasks): Some plasticity loss with large models
- MT50 (50 tasks): Almost no plasticity loss even with 4096-width networks

### **Optimal Architecture (from Paper)**
- Actor: [256, 256] (3 hidden layers)
- Critic: [1024, 1024, 1024] (3 hidden layers)
- Width 1024 provides best performance/speed trade-off

---

## 🚀 **Quick Start**

### **For 2M Steps (Fast Experiment):**
```bash
cd ~/metaworld_project/source/rl_project_sac/docker/cluster
WANDB_MODE=online sbatch train_mt10_2M_a40.sh
```
- Hardware: A40
- Time: 1-1.5h
- Result: Quick validation

### **For 20M Steps (Paper Reproduction):**
```bash
cd ~/metaworld_project/source/rl_project_sac/docker/cluster
WANDB_MODE=online sbatch train_mt10_20M_a100.sh
```
- Hardware: A100
- Time: 8-10h
- Result: Final Paper-quality results

---

## 📈 **Expected Performance (from Paper)**

### **MT10 (20M timesteps):**
- Success Rate: ~90-95% (averaged across 10 tasks)
- Most tasks reach >95% success by 10M steps
- Some tasks (e.g., assembly) may need full 20M

### **Evaluation:**
- Every 200k steps (every 40 episodes)
- 50 episodes per evaluation
- Metrics: Success rate, episode reward

---

## 💡 **Troubleshooting**

### **Problem: Low GPU Utilization (<30%)**
- **Cause:** Too few parallel environments
- **Solution:** Increase `NUM_PARALLEL_ENVS` (32 for A40, 48 for A100)

### **Problem: OOM Error**
- **Cause:** Too many parallel environments for GPU VRAM
- **Solution:** Reduce `NUM_PARALLEL_ENVS` by 25-50%

### **Problem: Training slower than expected**
- **Cause:** CPU bottleneck (MuJoCo physics)
- **Solution:** Check `--cpus-per-task` is set to 16

### **Problem: Plasticity loss (many dormant neurons)**
- **Cause:** Large model with too few tasks
- **Solution:** Either reduce model size OR train on MT50 instead of MT10

---

## 📚 **References**

- **Paper:** McLean et al. 2025 - "Multi-Task RL Enables Parameter Scaling"
- **Benchmark:** Meta-World (Yu et al. 2020)
- **Algorithm:** Soft Actor-Critic (Haarnoja et al. 2018)
