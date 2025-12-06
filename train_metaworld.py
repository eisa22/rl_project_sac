import argparse
import random
import numpy as np
import metaworld
import wandb
import torch

from sac_agent import SACAgent, ReplayBuffer, device


# --------------------------------------------------------
# Env Setup
# --------------------------------------------------------
def make_single_env(env_name):
    ml1 = metaworld.ML1(env_name)
    env_cls = ml1.train_classes[env_name]
    env = env_cls()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)
    return env


def make_multitask_envs(selected_env_names):
    mt10 = metaworld.MT10()
    envs = {}
    for name, env_cls in mt10.train_classes.items():
        if name in selected_env_names:
            envs[name] = env_cls()

    tasks = [t for t in mt10.train_tasks if t.env_name in selected_env_names]
    return envs, tasks


# --------------------------------------------------------
# Evaluation
# --------------------------------------------------------
def evaluate_single(env_name, agent, episodes=10):
    env = make_single_env(env_name)
    returns = []
    successes = 0

    success_threshold = 0.05

    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret = 0
        info = {}  # IMPORTANT FIX

        for _ in range(150):
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            if terminated or truncated:
                break

        returns.append(ep_ret)

        if "success" in info:
            successes += float(info["success"])
        elif "distance" in info:
            successes += float(info["distance"] < success_threshold)

    return np.mean(returns), successes / episodes


# --------------------------------------------------------
# Training Single Task
# --------------------------------------------------------
def train_single_task(env_name, total_steps=300000, start_steps=1000, batch_size=256, eval_interval=10000):
    env = make_single_env(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(obs_dim, act_dim, size=int(1e6))
    agent = SACAgent(obs_dim, act_dim, act_limit)

    obs, _ = env.reset()
    ep_len = 0

    for t in range(1, total_steps + 1):
        if t < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.add(obs, action, reward, next_obs, float(done))
        obs = next_obs

        ep_len += 1
        if done or ep_len >= 150:
            obs, _ = env.reset()
            ep_len = 0

        if t >= start_steps:
            agent.update(replay_buffer, batch_size)

        if t % eval_interval == 0:
            avg_ret, succ = evaluate_single(env_name, agent)
            wandb.log({
                "eval_avg_return": avg_ret,
                "eval_success_rate": succ,
                "global_step": t
            })
            print(f"[{env_name}] {t}: return={avg_ret:.2f}, success={succ:.2f}")


# --------------------------------------------------------
# Training Multi Task (3 tasks)
# --------------------------------------------------------
def train_multitask(env_names, total_steps=800000, start_steps=2000, batch_size=512, eval_interval=20000):
    envs, tasks = make_multitask_envs(env_names)

    ref_env = list(envs.values())[0]
    obs_dim = ref_env.observation_space.shape[0]
    act_dim = ref_env.action_space.shape[0]
    act_limit = float(ref_env.action_space.high[0])
    num_tasks = len(env_names)

    task_id_map = {name: i for i, name in enumerate(env_names)}

    replay_buffer = ReplayBuffer(obs_dim, act_dim, task_dim=num_tasks, size=int(1e6))
    agent = SACAgent(obs_dim + num_tasks, act_dim, act_limit)

    cur_task = random.choice(tasks)
    env = envs[cur_task.env_name]
    env.set_task(cur_task)
    tid = task_id_map[cur_task.env_name]

    obs_raw, _ = env.reset()
    obs = np.concatenate([obs_raw, np.eye(num_tasks)[tid]])

    ep_len = 0

    for t in range(1, total_steps + 1):
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

        if done or ep_len >= 150:
            cur_task = random.choice(tasks)
            env = envs[cur_task.env_name]
            env.set_task(cur_task)
            tid = task_id_map[cur_task.env_name]

            o_raw, _ = env.reset()
            obs = np.concatenate([o_raw, np.eye(num_tasks)[tid]])
            ep_len = 0

        if t >= start_steps:
            agent.update(replay_buffer, batch_size)

        if t % eval_interval == 0:
            for name in env_names:
                avg_ret, succ = evaluate_single(name, agent)
                wandb.log({
                    f"{name}_avg_return": avg_ret,
                    f"{name}_success_rate": succ
                })

            print(f"[MT] Eval step {t}")


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--env", type=str, default="reach-v2")
    parser.add_argument("--run_name", type=str, default="experiment_1")

    args = parser.parse_args()

    wandb.init(
        project="metaworld-sac-mtrl",
        name=args.run_name,
        config={
            "mode": args.mode,
            "env": args.env
        }
    )

    if args.mode == "single":
        train_single_task(args.env)
    else:
        train_multitask(["reach-v2", "push-v2", "pick-place-v2"])


if __name__ == "__main__":
    main()
