#!/usr/bin/env python3
"""Convert an OGBench ManipSpace NPZ replay into an SWM HDF5 dataset."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ogbench_scene import install_scene_goal_state_patch  # noqa: E402


def resolve_env_name(env_name: str) -> str:
    return {"cube-v0": "swm/OGBCube-v0", "scene-v0": "swm/OGBScene-v0"}.get(env_name, env_name)


def sanitize_key(key: str) -> str:
    return key.replace("/", "_")


def get_button_states(raw, i: int, info: dict) -> np.ndarray | None:
    if "button_states" in raw:
        return np.asarray(raw["button_states"][i]).reshape(-1)

    values = []
    j = 0
    while f"button_state_{j}" in raw:
        values.append(raw[f"button_state_{j}"][i])
        j += 1
    if values:
        return np.asarray(values).reshape(-1)

    if "button_states" in info:
        return np.asarray(info["button_states"]).reshape(-1)
    return None


def episode_boundaries(raw, n_steps: int, max_episode_steps: int | None):
    done = np.zeros(n_steps, dtype=bool)
    for key in ("terminals", "terminateds", "truncations", "truncateds", "dones"):
        if key in raw:
            done |= np.asarray(raw[key]).reshape(-1).astype(bool)

    if max_episode_steps:
        done[np.arange(max_episode_steps - 1, n_steps, max_episode_steps)] = True

    starts = [0]
    ends = []
    for i, is_done in enumerate(done):
        if is_done:
            ends.append(i + 1)
            if i + 1 < n_steps:
                starts.append(i + 1)
    if len(ends) < len(starts):
        ends.append(n_steps)
    return list(zip(starts, ends))


def append_row(rows, key: str, value) -> None:
    value = np.asarray(value)
    if value.dtype.kind in {"O", "U", "S"}:
        return
    rows[sanitize_key(key)].append(value.copy())


def write_hdf5(rows: dict[str, list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", libver="latest") as f:
        for key, values in rows.items():
            array = np.asarray(values)
            f.create_dataset(key, data=array)


def convert(args: argparse.Namespace) -> None:
    install_scene_goal_state_patch()
    raw = np.load(args.input, allow_pickle=True)
    if "qpos" not in raw or "qvel" not in raw:
        raise ValueError("Input NPZ must contain qpos and qvel.")

    env = gym.make(
        resolve_env_name(args.env_name),
        max_episode_steps=args.max_episode_steps,
        env_type=args.env_type,
        ob_type="states",
        mode=args.mode,
        width=args.width,
        height=args.height,
        terminate_at_goal=False,
    )
    env.reset(seed=args.seed)
    unwrapped = env.unwrapped

    n_steps = len(raw["qpos"])
    rows = defaultdict(list)
    ep_lengths = []
    ep_offsets = []
    global_idx = 0

    for ep_idx, (start, end) in enumerate(episode_boundaries(raw, n_steps, args.max_episode_steps)):
        ep_len = end - start
        ep_lengths.append(ep_len)
        ep_offsets.append(global_idx)
        for step_idx, i in enumerate(range(start, end)):
            button_states = get_button_states(raw, i, {})
            kwargs = {}
            if button_states is not None:
                for button_id, state in enumerate(button_states):
                    kwargs[f"button_state_{button_id}"] = state
            unwrapped.set_state(raw["qpos"][i], raw["qvel"][i], **kwargs)

            info = unwrapped.get_step_info()
            action = raw["actions"][i] if "actions" in raw else raw["action"][i]

            append_row(rows, "id", global_idx)
            append_row(rows, "ep_idx", ep_idx)
            append_row(rows, "episode_idx", ep_idx)
            append_row(rows, "step_idx", step_idx)
            append_row(rows, "action", action)
            append_row(rows, "pixels", unwrapped.get_pixel_observation())
            append_row(rows, "observation", unwrapped.compute_observation())

            if button_states is not None:
                append_row(rows, "button_states", button_states)
            for key, value in info.items():
                append_row(rows, key, value)
            for key in ("rewards", "reward", "terminals", "terminateds", "truncations", "truncateds"):
                if key in raw:
                    append_row(rows, key.rstrip("s"), raw[key][i])

            global_idx += 1

    rows["ep_len"] = ep_lengths
    rows["ep_offset"] = ep_offsets
    write_hdf5(rows, args.output)
    env.close()
    print(f"Wrote {global_idx} rows to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--env_name", default="scene-v0")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--env_type", default="scene")
    parser.add_argument("--mode", default="data_collection")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
