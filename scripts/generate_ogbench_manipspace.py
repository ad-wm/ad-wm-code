#!/usr/bin/env python3
"""Generate LeWM-ready OGBench ManipSpace datasets with SWM record_dataset.

Default Scene command:
  python scripts/generate_ogbench_manipspace.py \
    --env_name scene-v0 --dataset_type play --num_episodes 10000 \
    --max_episode_steps 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench.expert_policy import ExpertPolicy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ogbench_scene import install_scene_goal_state_patch  # noqa: E402


def resolve_env_name(env_name: str) -> str:
    aliases = {
        "cube-v0": "swm/OGBCube-v0",
        "scene-v0": "swm/OGBScene-v0",
    }
    return aliases.get(env_name, env_name)


def add_episode_idx_alias(path: Path) -> None:
    if not path.exists():
        return
    with h5py.File(path, "a") as f:
        if "ep_idx" in f and "episode_idx" not in f:
            f.create_dataset("episode_idx", data=f["ep_idx"][:])


def record_split(args, split_name: str, episodes: int, seed: int) -> None:
    if episodes <= 0:
        return

    env_name = resolve_env_name(args.env_name)
    dataset_name = args.dataset_name if split_name == "train" else f"{args.dataset_name}_val"
    print(f"Recording {episodes} {split_name} episodes to {dataset_name}.h5")

    world = swm.World(
        env_name=env_name,
        num_envs=args.num_envs,
        image_shape=(args.height, args.width),
        seed=seed,
        history_size=1,
        frame_skip=1,
        max_episode_steps=args.max_episode_steps,
        env_type=args.env_type,
        ob_type=args.ob_type,
        mode=args.mode,
        width=args.width,
        height=args.height,
        multiview=args.multiview,
        terminate_at_goal=args.terminate_at_goal,
        verbose=args.verbose,
    )
    policy = ExpertPolicy(
        policy_type=args.policy_type,
        action_noise=args.action_noise,
        p_random_action=args.p_random_action,
        noise_smoothing=args.noise_smoothing,
        min_norm=args.min_norm,
        seed=seed,
    )
    world.set_policy(policy)
    world.record_dataset(
        dataset_name=dataset_name,
        episodes=episodes,
        seed=seed,
        cache_dir=args.cache_dir,
    )
    add_episode_idx_alias(Path(args.cache_dir or swm.data.utils.get_cache_dir()) / f"{dataset_name}.h5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="scene-v0")
    parser.add_argument("--dataset_type", default="play", choices=["play"])
    parser.add_argument("--dataset_name", default="ogbench/scene_play_lewm_10k_200")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--cache_dir", default=None,
                        help="Dataset root; defaults to STABLEWM_HOME (or SWM's default cache)")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--env_type", default="scene")
    parser.add_argument("--ob_type", default="states")
    parser.add_argument("--mode", default="data_collection")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--terminate_at_goal", action="store_true")
    parser.add_argument("--policy_type", default="markov_oracle", choices=["markov_oracle", "plan_oracle"])
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--p_random_action", type=float, default=0.0)
    parser.add_argument("--noise_smoothing", type=float, default=0.5)
    parser.add_argument("--min_norm", type=float, default=0.4)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    install_scene_goal_state_patch()
    args = parse_args()
    record_split(args, "train", args.num_episodes, args.seed)
    if not args.no_val:
        val_episodes = int(round(args.num_episodes * args.val_fraction))
        record_split(args, "val", val_episodes, args.seed + args.num_episodes)


if __name__ == "__main__":
    main()
