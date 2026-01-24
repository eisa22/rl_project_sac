# From Single-Task to Multi-Task Reinforcement Learning in Meta-World Using SAC

This repository documents the progressive transition from single-task to multi-task reinforcement learning (MTRL) using the Meta-World benchmark and the Soft Actor-Critic (SAC) algorithm.

**Paper:** "From Single-Task to Multi-Task Reinforcement Learning in Meta-World Using Soft Actor-Critic"  
**Authors:** Einspieler Samuel, Kestler Johannes, Loibelsberger Thomas, Müller Markus  
**Institution:** Institute of Computer Technology, TU Wien

---

## 📂 Branch Overview

This repository contains **5 branches**, each representing a stage in the methodological progression described in the paper:

| Branch | Description | Paper Section |
|--------|-------------|---------------|
| `main` | Final production-ready codebase | - |
| `feature/mt1` | Single-Task SAC baseline | III-A |
| `feature/mt3` | Multi-Task Multi-Head SAC (3 tasks) | III-B, III-D |
| `feature/mt10-mt3_ars_pr` | MTMH-SAC with ARS + Periodic Resets | IV-A, IV-B |
| `evaluation_visualization` | Evaluation and visualization tools | V |

---

## 🎯 Branch Details

### `main`
**Production-ready codebase**

The main branch contains the stable, documented version of the project with basic training infrastructure. This serves as the integration point for all feature branches.

**Contents:**
- `CLUSTER_DEPLOYMENT.md` / `QUICKSTART.md` - Documentation
- `logs_multitask/` - Training logs
- `metaworld_multitask_logs/` - Meta-World specific logs
- `train_mt10` - MT10 training utilities

---

### `feature/mt1`
**Stage 1: Single-Task SAC Baseline**

Implementation of standard Soft Actor-Critic for a single Meta-World task. This establishes the baseline performance before transitioning to multi-task learning.

**Concept (Paper Section III-A):**
- Training one task in isolation (e.g., `reach-v2`)
- Standard SAC with automatic entropy tuning
- Uses Stable Baselines 3 as foundation
- WandB logging for experiment tracking

**Key Files:**
```
mt1/
├── train_metaworld.py    # Single-task SAC training script
├── play_metaworld.py     # Policy visualization/evaluation
├── models/               # Saved model checkpoints
└── metaworld_logs/       # Training logs
```

**Hyperparameters:**
- Learning rate: 3e-4
- Batch size: 256
- Evaluation episodes: 20

---

### `feature/mt3`
**Stage 2: Multi-Task Multi-Head SAC (MT3)**

First extension to multi-task learning with 3 related Meta-World tasks: **Reach**, **Push**, and **Pick-Place**. Introduces the Multi-Head architecture with task-specific actor and critic heads.

**Concept (Paper Section III-B, III-D):**
- **Multi-Head Actor**: Separate policy heads per task
- **Multi-Head Critic**: Separate Q-value heads per task
- **Shared Trunk**: Common feature encoder across tasks
- **Curriculum Learning**: Progressive task introduction

**Key Files:**
```
├── train_mt3_curriculum_sb3.py    # MT3 training with curriculum
├── evaluate.py                     # Evaluation script
├── sac_agent_sb3/
│   ├── agent.py                   # Base SAC agent
│   ├── multiheadAgent.py          # Multi-head architecture
│   ├── curriculum.py              # Curriculum learning logic
│   └── evaluation.py              # Evaluation utilities
├── models/                         # Saved checkpoints
└── requirements.txt
```

**Architecture:**
- Shared feature extractor (trunk)
- Task-specific policy heads (actor)
- Task-specific Q-function heads (critic)
- Per-task entropy coefficients (α)

---

### `feature/mt10-mt3_ars_pr`
**Stage 3: MTMH-SAC with Advanced Techniques**

Extended implementation adding **Adaptive Reward Scaling (ARS)** and **Periodic Resets (PR)** to handle the challenges of multi-task learning at scale (MT3 and MT10).

**Concept (Paper Section IV):**

**Adaptive Reward Scaling (ARS) - Section IV-A:**
- Equalizes reward magnitudes across tasks with different reward scales
- Formula: `c_i = max(r̄₁, ..., r̄ₙ) / r̄ᵢ` (equalizer rule)
- Recomputed every 50k steps based on replay buffer statistics
- Prevents tasks with larger rewards from dominating learning

**Periodic Resets (PR) - Section IV-B:**
- Resets actor, critic, target networks, and entropy coefficient α
- Replay buffer is **retained** for continued sample efficiency
- Prevents "loss of plasticity" in long training runs
- Reset interval: every 5M steps

**Key Files:**
```
├── mtmh_sac.py              # Full MTMH-SAC implementation with ARS/PR
├── train_mt3_mtmh.py        # MT3 training script
├── train_mt10_mtmh.py       # MT10 training script (10 tasks)
├── deploy_to_cluster.sh     # HPC cluster deployment
├── docker/cluster/          # SLURM job scripts
│   ├── build_docker.sh
│   ├── convert_to_singularity.sh
│   ├── train_mt3_mtmh.sh
│   └── train_mt10_mtmh.sh
├── models_mtmh/             # Model checkpoints
├── CLUSTER_SETUP.md         # Cluster configuration guide
└── Dockerfile
```

**Training Configuration:**
| Parameter | MT3 | MT10 |
|-----------|-----|------|
| Total Steps | 20M | 20M |
| Batch Size | 2048 | 2048 |
| Envs per Task | 16 | 8 |
| Learning Starts | 30k | 40k |
| ARS Update Freq | 50k | 50k |
| Reset Interval | 5M | 5M |

---

### `evaluation_visualization`
**Stage 4: Evaluation and Visualization**

Tools for evaluating trained models and visualizing robot manipulation behavior in the Meta-World simulator.

**Key Files:**
```
├── evaluate_mt3_visual.py    # MT3 visual evaluation
├── evaluate_mt10_visual.py   # MT10 visual evaluation
├── EVALUATION_README.md      # Usage documentation
├── mtmh_sac.py               # Agent implementation
├── train_mt3_mtmh.py         # Training script
└── train_mt10_mtmh.py        # Training script
```

**Features:**
- Load pre-trained MTMH-SAC models
- Visual rendering of robot manipulation
- Success rate computation per task
- Episode recording capabilities

---

## 📊 Methodological Progression

The branches follow the paper's progression from simple to complex:

```
feature/mt1 (Single-Task SAC)
    ↓
feature/mt3 (Multi-Head Architecture + Curriculum)
    ↓
feature/mt10-mt3_ars_pr (+ ARS + Periodic Resets)
    ↓
evaluation_visualization (Evaluation Tools)
```

| Paper Section | Technique | Branch |
|---------------|-----------|--------|
| III-A | Single-Task SAC Baseline | `feature/mt1` |
| III-B | Task Conditioning (One-Hot) | `feature/mt3` |
| III-D | Multi-Head Actor-Critic | `feature/mt3` |
| IV-A | Adaptive Reward Scaling | `feature/mt10-mt3_ars_pr` |
| IV-B | Periodic Network Resets | `feature/mt10-mt3_ars_pr` |
| V | Evaluation Protocol | `evaluation_visualization` |

---

## 🚀 Quick Start

### Single-Task Training (MT1)
```bash
git checkout feature/mt1
cd mt1
python train_metaworld.py
```

### Multi-Task Training (MT3)
```bash
git checkout feature/mt3
python train_mt3_curriculum_sb3.py
```

### Multi-Task with ARS/PR (MT3 or MT10)
```bash
git checkout feature/mt10-mt3_ars_pr
python train_mt3_mtmh.py --ars_enable --reset_enable
python train_mt10_mtmh.py --ars_enable --reset_enable
```

### Evaluation
```bash
git checkout evaluation_visualization
python evaluate_mt3_visual.py --model_path <path_to_model>
```

---

## 📖 References

- **Paper:** "From Single-Task to Multi-Task Reinforcement Learning in Meta-World Using Soft Actor-Critic"
- **Authors:** Einspieler Samuel, Kestler Johannes, Loibelsberger Thomas, Müller Markus
- **Institution:** Institute of Computer Technology, TU Wien, Vienna, Austria
- **Meta-World Benchmark:** Yu et al., "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning"
- **SAC Algorithm:** Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"

---

## 📁 Repository Structure

```
rl_project_sac/
├── main                        # Production codebase
├── feature/mt1                 # Single-task baseline
├── feature/mt3                 # Multi-head MT3
├── feature/mt10-mt3_ars_pr     # Full MTMH-SAC with ARS/PR
└── evaluation_visualization    # Evaluation tools
```
