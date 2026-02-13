import os
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # IMPORTANT: reliable PNG saving everywhere
import matplotlib.pyplot as plt

from tqdm import trange

from environment import OrionReachEnv
from ppo_agent import PPO


def encode_obs(obs):
    """
    Tiny feature encoder:
      - proprio (7)
      - RGB channel mean (3)
      - RGB channel std (3)
    Total = 13 dims. Very lightweight, no dataset needed.
    """
    rgb = obs["rgb"]
    proprio = obs["proprio"]
    ch_mean = rgb.mean(axis=(0, 1))
    ch_std = rgb.std(axis=(0, 1))
    return np.concatenate([proprio, ch_mean, ch_std], axis=0).astype(np.float32)


def compute_returns_adv(rewards, values, gamma=0.99):
    ret = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        ret.append(g)
    ret = list(reversed(ret))
    ret = np.array(ret, dtype=np.float32)
    adv = ret - np.array(values, dtype=np.float32)
    return ret, adv


def main(
    out_dir="docs/orion",
    episodes=80,
    steps_per_update=1024,
    render=False,
    seed=0,
):
    print("✅ Starting Orion training...")
    print(f"   out_dir={out_dir}, episodes={episodes}, steps_per_update={steps_per_update}, render={render}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ensure docs/ exists too (for safety)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = OrionReachEnv(render=render, seed=seed)

    obs, info = env.reset()
    obs_dim = encode_obs(obs).shape[0]
    act_dim = 7

    agent = PPO(obs_dim, act_dim, lr=3e-4)

    logs = []
    reward_hist = []
    success_hist = []

    buffer = {"obs": [], "act": [], "logp": [], "rew": [], "val": []}

    for ep in trange(episodes, desc="Training"):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_success = 0

        while True:
            x = encode_obs(obs)
            a, logp, v = agent.act(x)

            next_obs, r, done, step_info = env.step(a)

            buffer["obs"].append(x)
            buffer["act"].append(a)
            buffer["logp"].append(logp)
            buffer["rew"].append(r)
            buffer["val"].append(v)

            ep_reward += float(r)
            ep_success = max(ep_success, int(step_info["success"]))
            obs = next_obs

            if len(buffer["obs"]) >= steps_per_update:
                ret, adv = compute_returns_adv(buffer["rew"], buffer["val"], gamma=agent.gamma)
                batch = {
                    "obs": np.array(buffer["obs"], dtype=np.float32),
                    "act": np.array(buffer["act"], dtype=np.float32),
                    "logp": np.array(buffer["logp"], dtype=np.float32),
                    "ret": ret,
                    "adv": adv,
                }
                agent.update(batch)
                buffer = {"obs": [], "act": [], "logp": [], "rew": [], "val": []}

            if done:
                break

        reward_hist.append(ep_reward)
        success_hist.append(ep_success)

        logs.append({
            "episode": ep + 1,
            "episode_reward": float(ep_reward),
            "success": int(ep_success),
        })

    # Save JSONL logs
    log_path = out / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")

    # Save model
    import torch
    model_path = out / "ppo_policy.pth"
    torch.save(agent.model.state_dict(), model_path)

    # Plot reward curve
    xs = np.arange(1, len(reward_hist) + 1)
    y = np.array(reward_hist, dtype=np.float32)

    plt.figure()
    plt.plot(xs, y, label="Episode reward")
    if len(y) >= 10:
        y_smooth = np.convolve(y, np.ones(10) / 10, mode="valid")
        plt.plot(xs[9:], y_smooth, label="Smoothed (w=10)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Orion: PPO Training Reward")
    plt.legend()
    plt.tight_layout()
    reward_fig = out / "reward_curve.png"
    plt.savefig(reward_fig, dpi=200)

    # Plot success curve
    s = np.array(success_hist, dtype=np.float32)
    plt.figure()
    plt.plot(xs, s, label="Success (0/1)")
    if len(s) >= 10:
        s_smooth = np.convolve(s, np.ones(10) / 10, mode="valid")
        plt.plot(xs[9:], s_smooth, label="Smoothed (w=10)")
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.title("Orion: Success Rate")
    plt.legend()
    plt.tight_layout()
    succ_fig = out / "success_curve.png"
    plt.savefig(succ_fig, dpi=200)

    env.close()

    print("\n✅ Finished. Saved artifacts:")
    print(f"- {reward_fig}")
    print(f"- {succ_fig}")
    print(f"- {log_path}")
    print(f"- {model_path}")


if __name__ == "__main__":
    main()
