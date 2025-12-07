"""
Meta-World MT1 Evaluation Script (Stable-Baselines3)

A safe and modern evaluation script compatible with:
- Meta-World MT1 environments (v2 or v3)
- Stable-Baselines3 algorithms: SAC, TD3, DDPG

Steps:
1. Set TASK_NAME to match your trained task (e.g. 'reach-v2', 'push-v2', 'pick-place-v2')
2. Set ALGORITHM to the algorithm you trained ('SAC', 'TD3', or 'DDPG')
3. Run: python evaluate_metaworld_sb3.py
"""

import os
import numpy as np
from stable_baselines3 import SAC

from train_metaworld import MetaWorldMT10Env


# ================================
# User Configuration (MT10)
# ================================
ALGORITHM = "SAC"  # align with training algo
MODEL_PATH = "./models_mt10/sac_mt10_final.zip"
TASK_NAME = "reach-v3"  # choose one MT10 task; matches training env_name
SEED = 42
EPISODES = 10
MAX_EPISODE_STEPS = 150  # match training
RENDER = True            # set False for headless evaluation
# ================================


if __name__ == "__main__":

    # -------------------------------------------------------
    # Build Meta-World MT10 environment (one-hot task id)
    # -------------------------------------------------------
    print("\nCreating Meta-World MT10 environment (random task per episode)")

    env = MetaWorldMT10Env(
        seed=SEED,
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode="human" if RENDER else None,
        fixed_task_name=TASK_NAME,
    )

    # -------------------------------------------------------
    # Locate trained model
    # -------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print("\n❌ No trained model found!")
        print(f"Expected: {MODEL_PATH}")
        exit(1)

    print(f"\nLoading model from: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH, env=env)


    # -------------------------------------------------------
    # Evaluation Loop
    # -------------------------------------------------------
    print("\n===========================================================")
    print(f"Running evaluation on task: {TASK_NAME}")
    print("===========================================================\n")

    total_rewards = []
    successes = 0

    for ep in range(EPISODES):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        ep_success = False
        steps = 0
        task_name = info.get("task_name", "unknown") if isinstance(info, dict) else "unknown"

        print(f"\n--- Episode {ep+1}/{EPISODES} ---")
        print(f"Task: {task_name}")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            ep_reward += reward
            steps += 1

            # Meta-World provides success flag
            if isinstance(info, dict) and info.get("success", False):
                ep_success = True

            if RENDER:
                env.render()

        print(f"Steps: {steps}   |   Reward: {ep_reward:.2f}   |   Success: {ep_success}")

        total_rewards.append(ep_reward)
        if ep_success:
            successes += 1


    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print("===========================================================")
    print("EVALUATION SUMMARY")
    print("===========================================================")
    print(f"Episodes: {EPISODES}")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Std reward: {np.std(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    print(f"Success rate: {successes}/{EPISODES} ({100 * successes/EPISODES:.1f}%)")

    print("===========================================================\n")

    env.close()
