"""Minimal training bootstrap to verify task discovery and env stepping."""

from __future__ import annotations

import argparse

from isaacgymenvs.tasks import make_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task name, e.g. Go2Bridge")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=8)
    args = parser.parse_args()

    cfg = {
        "num_envs": args.num_envs,
        "num_observations": 8,
        "num_actions": 12,
        "max_episode_length": 500,
        "bridge_type": "narrow_static_bridge",
    }

    env = make_task(args.task, cfg)
    obs = env.reset()
    print(f"[train] task={args.task} initialized, obs_shape=({len(obs)}, {len(obs[0]) if obs else 0})")

    for i in range(args.max_iters):
        actions = [[0.0 for _ in range(env.num_actions)] for _ in range(env.num_envs)]
        step = env.step(actions)
        reward_mean = sum(step.rewards) / len(step.rewards)
        done_count = sum(1 for d in step.dones if d)
        print(f"[train] iter={i} reward_mean={reward_mean:.3f} done_count={done_count}")


if __name__ == "__main__":
    main()
