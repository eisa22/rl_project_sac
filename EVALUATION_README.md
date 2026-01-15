# MTMH-SAC Evaluation Script

Evaluates trained MTMH-SAC models on MT10 Meta-World tasks with optional live visualization.

## Requirements

```bash
pip install torch metaworld mujoco numpy
```

## Quick Start

```bash
cd c:\Users\samue\Documents\TU-Wien\RobotLearning\code-review\rl_project_sac

# Single task with visualization
python evaluate_mt10_visual.py --model_path models_mtmh/checkpoint_step4000000.pt --task reach-v3

# All 10 MT10 tasks (with visualization)
python evaluate_mt10_visual.py --model_path models_mtmh/checkpoint_step4000000.pt --task all

# Fast evaluation without rendering
python evaluate_mt10_visual.py --model_path models_mtmh/checkpoint_step4000000.pt --task all --no_render
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `models_mtmh/dummy_model/final_model.pt` | Path to model checkpoint |
| `--task` | `reach-v3` | Task name or `all` for all 10 tasks |
| `--num_episodes` | `5` | Number of episodes per task |
| `--no_render` | `false` | Disable MuJoCo visualization |
| `--delay` | `0.02` | Delay between steps (seconds) |
| `--device` | auto | Force `cuda` or `cpu` |
| `--seed` | `42` | Random seed |

## Available MT10 Tasks

1. `reach-v3` - Reach to target position
2. `push-v3` - Push object to goal
3. `pick-place-v3` - Pick up and place object
4. `door-open-v3` - Open a door
5. `drawer-close-v3` - Close a drawer
6. `button-press-topdown-v3` - Press button from above
7. `peg-insert-side-v3` - Insert peg into hole
8. `window-open-v3` - Slide window open
9. `drawer-open-v3` - Open a drawer
10. `door-close-v3` - Close a door

## Examples

```bash
# Evaluate push task with 10 episodes
python evaluate_mt10_visual.py --model_path models_mtmh/checkpoint_step4000000.pt --task push-v3 --num_episodes 10

# Fast benchmark all tasks on GPU
python evaluate_mt10_visual.py --model_path models_mtmh/checkpoint_step4000000.pt --task all --no_render --device cuda --num_episodes 20

# Slow visualization for demos
python evaluate_mt10_visual.py --model_path models_mtmh/checkpoint_step4000000.pt --task door-open-v3 --delay 0.05
```

## Output

The script outputs:
- Per-episode results (SUCCESS/FAILED, Reward, Steps)
- Per-task summary (Mean Reward, Success Rate, Mean Episode Length)
- Final summary table when evaluating all tasks
