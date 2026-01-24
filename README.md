# Git Branches Overview

This repository documents the transition from single-task to multi-task reinforcement learning (MTRL) using the Meta-World benchmark and the Soft Actor-Critic (SAC) algorithm, as described in the paper **"From Single-Task to Multi-Task Reinforcement Learning in Meta-World Using Soft Actor-Critic"**.

## Branch Overview

The branches are categorized by their functionality:

---

## 🎯 Main Branches

### `main`
**Main branch with the final MTMH-SAC implementation**

Contains the complete Multi-Task Multi-Head SAC (MTMH-SAC) implementation with all features:
- Shared trunk encoder with task-specific actor/critic heads
- Adaptive Reward Scaling (ARS)
- Periodic Resets (PR)
- GPU optimizations with mixed precision training
- Training for MT3 (3 tasks) and MT10 (10 tasks)

**Main files:**
- `mtmh_sac.py` - MTMH-SAC agent implementation
- `train_mt3_mtmh.py` - MT3 training script
- `train_mt10_mtmh.py` - MT10 training script

---

### `evaluation_visualization`
**Evaluation and visualization of trained models**

This branch contains tools for evaluation and visual representation of trained MTMH-SAC models:
- Evaluation scripts for MT3 and MT10
- Visualization of robot manipulation tasks
- Success rate metrics and performance analysis

---

## 📚 Feature Branches (Methodological Development)

### `feature/mt1`
**Single-Task SAC Baseline (MT1)**

Implementation of single-task Soft Actor-Critic as the starting point for the transition to multi-task learning. Corresponds to Section III-A of the paper.

**Concept:** 
- Training a single task in isolation
- Basic SAC implementation without multi-task extensions
- Serves as baseline for performance comparisons

**Structure:**
```
mt1/
├── train.py        # Single-Task Training
└── sac_agent.py    # Basic SAC implementation
```

---

### `feature/mt3`
**Multi-Task SAC for 3 Tasks (Reach, Push, Pick-Place)**

First extension to multi-task learning with 3 related Meta-World tasks. Implements the basic MTMH-SAC architecture as described in Section III-B of the paper.

**Concept:**
- Multi-head architecture with task-specific heads
- Curriculum learning approach
- Uses Stable Baselines 3 (SB3) as foundation

**Contains:**
- `train_mt3_curriculum_sb3.py` - Training with curriculum learning
- `sac_agent_sb3/` - SAC agent based on SB3
- `evaluate.py` - Evaluation script

---

### `feature/mt10-mt3_ars_pr`
**MT10/MT3 with Adaptive Reward Scaling and Periodic Resets**

Extended implementation with advanced techniques from Section IV of the paper:

**Adaptive Reward Scaling (ARS):**
- Equalizes reward magnitudes across different tasks
- Formula: `c_i = max(r̄₁, ..., r̄ₙ) / r̄ᵢ`
- Updates every 50k steps based on buffer statistics

**Periodic Resets (PR):**
- Resets actor, critic, target networks, and alpha
- Replay buffer is retained for sample efficiency
- Prevents "loss of plasticity" during long training runs

---

## � Development Branches

### `mtmhsac-markus`
**Multi-Task Multi-Head SAC Development (Markus)**

Development branch for the MTMH-SAC architecture:
- SB3-based MT-SAC implementation
- Cluster setup for training on GPU nodes
- Task-specific configurations

---

### `mt3_markus_final`
**Finalized MT3 Implementation**

Final version of the MT3 implementation with:
- `metaworld_mt_env.py` - Multi-task environment wrapper
- `make_vec_envs.py` - Vectorized environment creation
- `evaluate_mt3.py` - Final evaluation scripts

---

### `dev-johannes`
**Development Branch (Johannes)**

Experimental developments and tests.

---

### `dev-markus`
**Development Branch (Markus)**

Experimental implementations and architecture tests.

---

### `dev-thomas`
**Development Branch (Thomas)**

Development and experiments.

---

### `dev-thomas-mtrl-cluster`
**MTRL Cluster Deployment (Thomas)**

Cluster-specific configurations and SLURM job scripts for training on HPC clusters.

---

## 🧪 Experimental Branches

### `MT_SAC_task_embedding`
**Task Embeddings Experiment**

Implementation of learned task representations as an alternative to one-hot task conditioning (Section III-C of the paper):

**Concept:**
- Replacing explicit task IDs with learned embeddings
- Dense representations can capture task similarities
- Potential for better transfer between related tasks

---

### `in_dev_sb3_*`
**Stable Baselines 3 Development**

Experimental implementations with the Stable Baselines 3 framework.

---

## 📊 Methodological Overview (Paper Reference)

The branches reflect the progressive development described in the paper:

| Section | Branch(es) | Description |
|---------|-----------|-------------|
| III-A: Single-Task SAC | `feature/mt1` | Baseline SAC training |
| III-B: Task Conditioning | `feature/mt3` | One-hot task IDs |
| III-C: Learned Embeddings | `MT_SAC_task_embedding` | Dense task representations |
| III-D: Multi-Head Architecture | `mtmhsac-markus`, `main` | Task-specific heads |
| IV-A: Adaptive Reward Scaling | `feature/mt10-mt3_ars_pr` | ARS implementation |
| IV-B: Periodic Resets | `feature/mt10-mt3_ars_pr`, `main` | Network resets |
| V: GPU Optimizations | `main` | Mixed precision, batch inference |

---

## 🚀 Recommended Workflow

1. **Understand baseline:** `feature/mt1` for single-task SAC
2. **Multi-task fundamentals:** `feature/mt3` for first MT experiments
3. **Advanced techniques:** `feature/mt10-mt3_ars_pr` for ARS/PR
4. **Final implementation:** `main` for complete MTMH-SAC

---

## 📖 References

- Paper: "From Single-Task to Multi-Task Reinforcement Learning in Meta-World Using Soft Actor-Critic"
- Authors: Einspieler Samuel, Kestler Johannes, Loibelsberger Thomas, Müller Markus
- Institution: Institute of Computer Technology, TU Wien
- Code: https://github.com/eisa22/rl_project_sac