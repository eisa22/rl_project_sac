# MTMH-SAC Evaluation Scripts

Evaluates trained MTMH-SAC models on Meta-World tasks with optional live visualization.

## Requirements

```bash
pip install torch metaworld mujoco numpy
```

---

## MT10 Evaluation (10 Tasks)

### Quick Start

```bash
# Single task with visualization
python evaluate_mt10_visual.py --model_path models_mtmh/checkpoint_step4000000.pt --task reach-v3

# All 10 tasks
python evaluate_mt10_visual.py --model_path models_mtmh/checkpoint_step4000000.pt --task all

# Fast (no rendering)
python evaluate_mt10_visual.py --model_path models_mtmh/checkpoint_step4000000.pt --task all --no_render
```

### MT10 Tasks
reach-v3, push-v3, pick-place-v3, door-open-v3, drawer-close-v3, button-press-topdown-v3, peg-insert-side-v3, window-open-v3, drawer-open-v3, door-close-v3

---

## MT3 Evaluation (3 Tasks)

### Quick Start

```bash
# Single task with visualization
python evaluate_mt3_visual.py --model_path models_mtmh/mt3/checkpoint_step3000000.pt --task reach-v3

# All 3 tasks
python evaluate_mt3_visual.py --model_path models_mtmh/mt3/checkpoint_step3000000.pt --task all

# Fast (no rendering)
python evaluate_mt3_visual.py --model_path models_mtmh/mt3/checkpoint_step3000000.pt --task all --no_render
```

### MT3 Tasks
reach-v3, push-v3, pick-place-v3

---

## Command Line Arguments (both scripts)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (see defaults) | Path to model checkpoint |
| `--task` | `reach-v3` | Task name or `all` |
| `--num_episodes` | `5` | Episodes per task |
| `--no_render` | `false` | Disable visualization |
| `--delay` | `0.02` | Step delay (seconds) |
| `--device` | auto | Force `cuda` or `cpu` |
| `--seed` | `42` | Random seed |

The script outputs:
- Per-episode results (SUCCESS/FAILED, Reward, Steps)
- Per-task summary (Mean Reward, Success Rate, Mean Episode Length)
- Final summary table when evaluating all tasks
