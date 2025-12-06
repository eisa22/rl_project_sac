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
import gymnasium as gym
import metaworld
from stable_baselines3 import SAC, TD3, DDPG


# ================================
# User Configuration
# ================================
TASK_NAME = "reach-v2"      # must match the task used during training
ALGORITHM = "SAC"           # one of: "SAC", "TD3", "DDPG"
MODEL_DIR = "./metaworld_models"
SEED = 42
EPISODES = 10
MAX_EPISODE_STEPS = 200     # should match your training setting
RENDER = True               # set False for headless evaluation
# ================================


def load_model(algo, model_path, env):
    if algo == "SAC":
        return SAC.load(model_path, env=env)
    elif algo == "TD3":
        return TD3.load(model_path, env=env)
    elif algo == "DDPG":
        return DDPG.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


if __name__ == "__main__":

    # -------------------------------------------------------
    # Build Meta-World MT1 environment
    # -------------------------------------------------------
    print(f"\nCreating Meta-World MT1 environment for: {TASK_NAME}")

    env = gym.make(
        "Meta-World/MT1",
        env_name=TASK_NAME,
        seed=SEED,
        render_mode="human" if RENDER else None,
        max_episode_steps=MAX_EPISODE_STEPS,
        terminate_on_success=False,
    )

    # -------------------------------------------------------
    # Locate trained model
    # -------------------------------------------------------
    preferred_dir = os.path.join(MODEL_DIR, f"best_{TASK_NAME}")
    preferred_model = os.path.join(preferred_dir, "best_model.zip")

    fallback_model = os.path.join(
        MODEL_DIR,
        f"{ALGORITHM.lower()}_{TASK_NAME}_final.zip"
    )

    if os.path.exists(preferred_model):
        model_path = preferred_model
    elif os.path.exists(fallback_model):
        model_path = fallback_model
    else:
        print("\n❌ No trained model found!")
        print(f"Expected: {preferred_model}")
        print(f"Or:      {fallback_model}")
        exit(1)

    print(f"\nLoading model from: {model_path}")
    model = load_model(ALGORITHM, model_path, env)


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

        print(f"\n--- Episode {ep+1}/{EPISODES} ---")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            ep_reward += reward
            steps += 1

            # Meta-World provides success flag
            if "success" in info and info["success"]:
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
    print("\n===========================================================")
    print("EVALUATION SUMMARY")
    print("===========================================================")
    print(f"Task: {TASK_NAME}")
    print(f"Episodes: {EPISODES}")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Std reward: {np.std(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    print(f"Success rate: {successes}/{EPISODES} ({100 * successes/EPISODES:.1f}%)")
    print("===========================================================\n")

    env.close()
