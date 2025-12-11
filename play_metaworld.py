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
import torch
from stable_baselines3 import SAC

from train_metaworld import MetaWorldMT10Env
from sac_agent import SACAgent


# ================================
# User Configuration (MT10)
# ================================
ALGORITHM = "SAC"  # align with training algo
MODEL_PATH = "./models_mt10/SAC_Agent_test_3/final_model.pt"  # supports .pt (custom) or .zip (SB3)
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

    def load_custom_pt(path, env):
        """Load custom SACAgent weights saved via torch.save(.pt)."""
        ckpt = torch.load(path, map_location=torch.device("cpu"))

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        act_limit = float(env.action_space.high[0])
        num_tasks = env.num_tasks if hasattr(env, "num_tasks") else 10

        agent = SACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_limit=act_limit,
            num_tasks=num_tasks,
            hidden_actor=(256, 256),
            hidden_critic=(1024, 1024, 1024),
            log_std_min=-20,
        )

        agent.actor.load_state_dict(ckpt["actor"])
        agent.q1.load_state_dict(ckpt["q1"])
        agent.q2.load_state_dict(ckpt["q2"])
        if "q1_target" in ckpt:
            agent.q1_target.load_state_dict(ckpt["q1_target"])
        if "q2_target" in ckpt:
            agent.q2_target.load_state_dict(ckpt["q2_target"])
        if "log_alpha" in ckpt and ckpt["log_alpha"] is not None:
            agent.log_alpha = ckpt["log_alpha"]
            agent.alpha = agent.log_alpha.exp().item()
        return agent

    if MODEL_PATH.endswith(".pt"):
        model = load_custom_pt(MODEL_PATH, env)
        predict_fn = lambda obs: (model.act(obs, deterministic=True), None)
    else:
        sb3_model = SAC.load(MODEL_PATH, env=env)
        predict_fn = lambda obs: sb3_model.predict(obs, deterministic=True)


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
            action, _ = predict_fn(obs)
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
