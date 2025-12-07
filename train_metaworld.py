import argparse
import random
import numpy as np
import metaworld
import wandb
import torch

from sac_agent import SACAgent, ReplayBuffer, device


# ============================================================
# MT10 Task Setup laut Paper
# ============================================================
MT10_TASKS = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "door-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
]


# ============================================================
# Environment Helpers
# ============================================================
def make_mt1_env(task_name):
    """Creates single-task MetaWorld MT1 environment for evaluation."""
    ml1 = metaworld.ML1(task_name)
    env_cls = ml1.train_classes[task_name]
    env = env_cls()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)
    return env


def load_mt10_envs():
    """Creates all MT10 training environments."""
    mt10 = metaworld.MT10()
    envs = {}
    tasks = []

    for name in MT10_TASKS:
        envs[name] = mt10.train_classes[name]()
        tasks.extend([t for t in mt10.train_tasks if t.env_name == name])

    return envs, tasks


# ============================================================
# Evaluation of a single task
# ============================================================
def evaluate_single(task_name, agent, episodes=5):
    env = make_mt1_env(task_name)
    returns = []
    successes = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0
        info = {}

        for _ in range(150):
            action = agent.act(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            if done or truncated:
                break

        returns.append(ep_ret)

        if "success" in info and info["success"]:
            successes += 1

    return np.mean(returns), successes / episodes


# ============================================================
# MT10 Training Loop (Paper Version)
# ============================================================
def train_mt10(total_steps=2_000_000, start_steps=3000, batch_size=512):
    print("\n🚀 Starting MT10 Training (Paper Setup)")

    envs, tasks = load_mt10_envs()

    # Observation / Action Dimensions
    ref_env = envs[MT10_TASKS[0]]
    obs_dim = ref_env.observation_space.shape[0]
    act_dim = ref_env.action_space.shape[0]
    act_limit = float(ref_env.action_space.high[0])

    num_tasks = len(MT10_TASKS)
    task_id_map = {name: i for i, name in enumerate(MT10_TASKS)}

    # Replay Buffer w/ task one-hot appended
    replay_buffer = ReplayBuffer(obs_dim, act_dim, task_dim=num_tasks, size=int(1e6))

    # SAC Agent w/ large critics per paper
    agent = SACAgent(
        obs_dim=obs_dim + num_tasks,
        act_dim=act_dim,
        act_limit=act_limit,
        hidden_actor=(256, 256),              # small actor (paper)
        hidden_critic=(2048, 2048, 2048),     # scaled critic (paper)
        gamma=0.99,
        lr=3e-4,
        tau=0.005,
    )

    # Initial Task
    cur_task = random.choice(tasks)
    env = envs[cur_task.env_name]
    env.set_task(cur_task)
    tid = task_id_map[cur_task.env_name]

    obs_raw, _ = env.reset()
    obs = np.concatenate([obs_raw, np.eye(num_tasks)[tid]])

    ep_len = 0

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for t in range(1, total_steps + 1):

        # Early random exploration
        if t < start_steps:
            action = ref_env.action_space.sample()
        else:
            action = agent.act(obs)

        next_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_obs = np.concatenate([next_raw, np.eye(num_tasks)[tid]])

        replay_buffer.add(obs, action, reward, next_obs, float(done))
        obs = next_obs
        ep_len += 1

        # Episode finished → sample new task (MT10 mixing)
        if done or ep_len >= 150:
            cur_task = random.choice(tasks)
            env = envs[cur_task.env_name]
            env.set_task(cur_task)
            tid = task_id_map[cur_task.env_name]
            o_raw, _ = env.reset()
            obs = np.concatenate([o_raw, np.eye(num_tasks)[tid]])
            ep_len = 0

        # Update SAC
        if t >= start_steps:
            agent.update(replay_buffer, batch_size=batch_size)

        # ------------------------------------------------------------
        # Periodic Evaluation (every 50k steps)
        # ------------------------------------------------------------
        if t % 50_000 == 0:
            print(f"\n📈 Eval @ step {t}")
            for task in MT10_TASKS:
                avg_ret, succ = evaluate_single(task, agent)
                wandb.log({
                    f"{task}_avg_return": avg_ret,
                    f"{task}_success_rate": succ,
                    "global_step": t
                })
                print(f"  {task}: return={avg_ret:.2f}, success={succ*100:.1f}%")

    print("\n🎉 Training Finished!")
    return agent


# ============================================================
# Main Entry
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="mt10_2mio_run")
    args = parser.parse_args()

    wandb.init(
        project="Robot_learning_2025",       # <--- YOUR PROJECT
        name=args.run_name,
        config={
            "tasks": MT10_TASKS,
            "total_steps": 2_000_000,
            "batch_size": 512,
            "actor": [256, 256],
            "critic": [2048, 2048, 2048]
        }
    )

    train_mt10(
        total_steps=2_000_000,
        start_steps=3000,
        batch_size=512
    )


if __name__ == "__main__":
    main()
