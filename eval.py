import os

os.environ.setdefault("MUJOCO_GL", "egl")


# MUJOCO_EGL_DEVICE_ID, if set, indexes EGL devices independently of CUDA visibility.


import time
from pathlib import Path
import re
import json
import shutil
from collections import defaultdict
from copy import deepcopy

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from ogbench_scene import install_scene_goal_state_patch
from eval_video_filter import video_reference_mask, video_output_indices

install_scene_goal_state_patch()

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "result"


def serialize_metric_value(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    if value is None:
        return np.array([], dtype=np.float32)
    return np.asarray(value)


def save_structured_results(
    base_path: Path,
    cfg: DictConfig,
    metrics: dict,
    evaluation_time: float,
    eval_episodes,
    eval_start_idx,
    extra_payload: dict | None = None,
):
    structured_name = cfg.output.get("structured_filename")
    if structured_name:
        structured_path = base_path.parent / structured_name
    else:
        stem = Path(cfg.output.filename).stem
        structured_path = base_path.parent / f"{stem}__{sanitize_filename(cfg.policy)}.npz"

    payload = {
        "policy": np.array([cfg.policy]),
        "success_rate": np.array([metrics.get("success_rate", np.nan)], dtype=np.float32),
        "evaluation_time": np.array([evaluation_time], dtype=np.float32),
        "eval_episodes": np.asarray(eval_episodes),
        "eval_start_idx": np.asarray(eval_start_idx),
    }
    for key, value in metrics.items():
        payload[key] = serialize_metric_value(value)
    for key, value in (extra_payload or {}).items():
        payload[key] = serialize_metric_value(value)

    structured_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(structured_path, **payload)
    return structured_path


def get_eval_settings(cfg: DictConfig):
    start_mode = OmegaConf.select(cfg, "eval.start_selection.mode", default="random")
    start_filter_mode = OmegaConf.select(cfg, "eval.start_filter.mode", default="any")
    perturb_enabled = bool(
        OmegaConf.select(cfg, "eval.ood.initial_perturb.enabled", default=False)
    )
    visual_variation_enabled = bool(
        OmegaConf.select(cfg, "eval.ood.visual_variation.enabled", default=False)
    )
    return start_mode, start_filter_mode, perturb_enabled, visual_variation_enabled


def get_scene_component_info(cfg: DictConfig, dataset, valid_indices):
    goal_row_offset = max(int(cfg.eval.goal_offset_steps) - 1, 0)
    goal_indices = valid_indices + goal_row_offset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_col = dataset.get_col_data(col_name)
    same_episode = episode_col[goal_indices] == episode_col[valid_indices]
    if not np.all(same_episode):
        raise ValueError("Goal indices crossed episode boundary while applying Scene filters.")

    masks = {}
    magnitudes = {}
    distances = {}

    if "privileged_block_0_pos" in dataset.column_names:
        block_pos = np.asarray(dataset.get_col_data("privileged_block_0_pos"))
        block_tol = float(
            OmegaConf.select(cfg, "eval.start_filter.block_pos_tolerance", default=0.04)
        )
        block_distance = np.linalg.norm(
            block_pos[goal_indices, :3] - block_pos[valid_indices, :3], axis=1
        )
        masks["cube"] = block_distance > block_tol
        magnitudes["cube"] = block_distance

    if "button_states" in dataset.column_names:
        button_states = np.asarray(dataset.get_col_data("button_states"))
        masks["button"] = np.any(
            button_states[goal_indices] != button_states[valid_indices], axis=1
        )
        magnitudes["button"] = masks["button"].astype(np.float32)
    else:
        button_keys = [
            key
            for key in sorted(dataset.column_names)
            if re.fullmatch(r"privileged_button_\d+_state", key)
        ]
        if button_keys:
            button_changed = np.zeros(len(valid_indices), dtype=bool)
            for key in button_keys:
                button_state = np.asarray(dataset.get_col_data(key)).reshape(
                    len(dataset.get_col_data(key)), -1
                )
                button_changed |= np.any(
                    button_state[goal_indices] != button_state[valid_indices], axis=1
                )
            masks["button"] = button_changed
            magnitudes["button"] = button_changed.astype(np.float32)

    if "privileged_drawer_pos" in dataset.column_names:
        drawer_pos = np.asarray(dataset.get_col_data("privileged_drawer_pos"))
        drawer_tol = float(
            OmegaConf.select(cfg, "eval.start_filter.drawer_pos_tolerance", default=0.04)
        )
        drawer_delta = np.abs(
            drawer_pos[goal_indices].reshape(len(valid_indices), -1)[:, 0]
            - drawer_pos[valid_indices].reshape(len(valid_indices), -1)[:, 0]
        )
        masks["drawer"] = drawer_delta > drawer_tol
        magnitudes["drawer"] = drawer_delta

    if "privileged_window_pos" in dataset.column_names:
        window_pos = np.asarray(dataset.get_col_data("privileged_window_pos"))
        window_tol = float(
            OmegaConf.select(cfg, "eval.start_filter.window_pos_tolerance", default=0.04)
        )
        window_delta = np.abs(
            window_pos[goal_indices].reshape(len(valid_indices), -1)[:, 0]
            - window_pos[valid_indices].reshape(len(valid_indices), -1)[:, 0]
        )
        masks["window"] = window_delta > window_tol
        magnitudes["window"] = window_delta

    if not masks:
        raise ValueError(
            "Scene filters require at least one Scene goal column "
            "(privileged_block_0_pos, button_states, privileged_drawer_pos, "
            "privileged_window_pos)."
        )

    if "proprio_effector_pos" in dataset.column_names:
        effector_pos = np.asarray(dataset.get_col_data("proprio_effector_pos")[valid_indices])
        affordance_columns = {
            "button": "privileged_target_button_top_pos",
            "cube": "privileged_block_0_pos",
            "drawer": "privileged_drawer_handle_pos",
            "window": "privileged_window_handle_pos",
        }
        for component, column in affordance_columns.items():
            if component in masks and column in dataset.column_names:
                target_pos = np.asarray(dataset.get_col_data(column)[valid_indices])
                distances[component] = np.linalg.norm(
                    effector_pos[:, :3] - target_pos[:, :3], axis=1
                )

    changed_count = np.stack(list(masks.values())).sum(axis=0).astype(np.int32)
    return {
        "masks": masks,
        "magnitudes": magnitudes,
        "distances": distances,
        "changed_count": changed_count,
    }


def build_start_filter_mask(cfg: DictConfig, dataset, valid_indices):
    mode = OmegaConf.select(cfg, "eval.start_filter.mode", default="any")
    if mode not in {"any", "table_non_contact", "scene_goal_mismatch"}:
        raise ValueError(f"Unsupported eval.start_filter.mode: {mode}")

    mask = np.ones(len(valid_indices), dtype=bool)
    goal_row_offset = max(int(cfg.eval.goal_offset_steps) - 1, 0)
    goal_indices = valid_indices + goal_row_offset

    pos = None
    if mode == "table_non_contact":
        required = {"privileged_block_0_pos", "proprio_gripper_contact"}
        missing = required.difference(dataset.column_names)
        if missing:
            raise ValueError(
                "table_non_contact requires dataset columns: "
                + ", ".join(sorted(missing))
            )

        pos = np.asarray(dataset.get_col_data("privileged_block_0_pos")[valid_indices])
        contact = np.asarray(
            dataset.get_col_data("proprio_gripper_contact")[valid_indices]
        ).reshape(-1)
        z_threshold = float(
            OmegaConf.select(cfg, "eval.start_filter.table_z_threshold", default=0.03)
        )
        contact_threshold = float(
            OmegaConf.select(cfg, "eval.start_filter.contact_threshold", default=0.5)
        )
        mask &= (pos[:, 2].reshape(-1) < z_threshold) & (contact < contact_threshold)
    elif mode == "scene_goal_mismatch":
        scene_info = get_scene_component_info(cfg, dataset, valid_indices)
        component_masks = scene_info["masks"]
        changed_count = scene_info["changed_count"]
        magnitudes = scene_info["magnitudes"]
        distances = scene_info["distances"]

        mask &= changed_count > 0

        min_changed = int(
            OmegaConf.select(cfg, "eval.start_filter.min_changed_components", default=1)
        )
        max_changed = OmegaConf.select(
            cfg, "eval.start_filter.max_changed_components", default=None
        )
        if min_changed > 1:
            mask &= changed_count >= min_changed
        if max_changed is not None:
            mask &= changed_count <= int(max_changed)

        min_component_delta = {
            "cube": float(
                OmegaConf.select(
                    cfg, "eval.start_filter.min_block_goal_distance", default=0.0
                )
            ),
            "drawer": float(
                OmegaConf.select(
                    cfg, "eval.start_filter.min_drawer_goal_delta", default=0.0
                )
            ),
            "window": float(
                OmegaConf.select(
                    cfg, "eval.start_filter.min_window_goal_delta", default=0.0
                )
            ),
        }
        for component, min_delta in min_component_delta.items():
            if min_delta > 0.0 and component in component_masks:
                mask &= (~component_masks[component]) | (magnitudes[component] >= min_delta)

        min_effector_distance = float(
            OmegaConf.select(
                cfg,
                "eval.start_filter.min_effector_changed_target_distance",
                default=0.0,
            )
        )
        if min_effector_distance > 0.0:
            for component, component_mask in component_masks.items():
                if component not in distances:
                    if np.any(component_mask):
                        raise ValueError(
                            "min_effector_changed_target_distance requires an "
                            f"affordance position column for Scene component {component}."
                        )
                    continue
                mask &= (~component_mask) | (distances[component] >= min_effector_distance)

    min_goal_distance = float(
        OmegaConf.select(cfg, "eval.start_filter.min_goal_distance", default=0.0)
    )
    if min_goal_distance > 0.0:
        if "privileged_block_0_pos" not in dataset.column_names:
            raise ValueError(
                "min_goal_distance requires dataset column privileged_block_0_pos."
            )
        if pos is None:
            pos = np.asarray(dataset.get_col_data("privileged_block_0_pos")[valid_indices])
        col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        episode_col = dataset.get_col_data(col_name)
        same_episode = episode_col[goal_indices] == episode_col[valid_indices]
        if not np.all(same_episode):
            raise ValueError(
                "Goal indices crossed episode boundary while applying min_goal_distance."
            )
        goal_pos = np.asarray(
            dataset.get_col_data("privileged_block_0_pos")[goal_indices]
        )
        start_goal_distance = np.linalg.norm(goal_pos[:, :2] - pos[:, :2], axis=1)
        mask &= start_goal_distance > min_goal_distance

    min_effector_block_distance = float(
        OmegaConf.select(
            cfg, "eval.start_filter.min_effector_block_distance", default=0.0
        )
    )
    if min_effector_block_distance > 0.0:
        required = {"proprio_effector_pos", "privileged_block_0_pos"}
        missing = required.difference(dataset.column_names)
        if missing:
            raise ValueError(
                "min_effector_block_distance requires dataset columns: "
                + ", ".join(sorted(missing))
            )
        if pos is None:
            pos = np.asarray(dataset.get_col_data("privileged_block_0_pos")[valid_indices])
        effector_pos = np.asarray(
            dataset.get_col_data("proprio_effector_pos")[valid_indices]
        )
        effector_block_distance = np.linalg.norm(
            effector_pos[:, :3] - pos[:, :3], axis=1
        )
        mask &= effector_block_distance > min_effector_block_distance

    return mask


def get_task_goal_distance(cfg: DictConfig, dataset, indices):
    """Return task-native normalized start--goal distance and initial success."""
    indices = np.asarray(indices, dtype=np.int64)
    goal_offset = max(int(cfg.eval.goal_offset_steps) - 1, 0)
    goal_indices = indices + goal_offset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_col = np.asarray(dataset.get_col_data(col_name))
    if not np.all(episode_col[goal_indices] == episode_col[indices]):
        raise ValueError("Goal indices crossed episode boundary in task-distance selection.")

    metric = str(
        OmegaConf.select(cfg, "eval.start_selection.distance_metric", default="")
    )
    if metric == "pusht_state":
        if "state" not in dataset.column_names:
            raise ValueError("pusht_state distance requires dataset column state.")
        state = np.asarray(dataset.get_col_data("state"))
        start = state[indices]
        goal = state[goal_indices]
        if start.ndim != 2 or start.shape[1] < 5:
            raise ValueError(f"Expected PushT state with at least 5 dims, got {start.shape}.")

        position_threshold = float(
            OmegaConf.select(
                cfg,
                "eval.start_selection.position_threshold",
                default=20.0,
            )
        )
        angle_threshold = float(
            OmegaConf.select(
                cfg,
                "eval.start_selection.angle_threshold",
                default=np.pi / 9,
            )
        )
        position_distance = np.linalg.norm(goal[:, :4] - start[:, :4], axis=1)
        angle_delta = np.abs(goal[:, 4] - start[:, 4])
        angle_distance = np.abs((angle_delta + np.pi) % (2 * np.pi) - np.pi)
        initial_success = (position_distance < position_threshold) & (
            angle_distance < angle_threshold
        )
        distance = np.maximum(
            position_distance / position_threshold,
            angle_distance / angle_threshold,
        )
    elif metric == "reacher_qpos":
        if "qpos" not in dataset.column_names:
            raise ValueError("reacher_qpos distance requires dataset column qpos.")
        qpos = np.asarray(dataset.get_col_data("qpos"))
        start = qpos[indices]
        goal = qpos[goal_indices]
        qpos_threshold = float(
            OmegaConf.select(
                cfg,
                "eval.start_selection.qpos_threshold",
                default=0.05,
            )
        )
        normalized_difference = np.abs(goal - start) / qpos_threshold
        initial_success = np.all(normalized_difference < 1.0, axis=1)
        distance = np.max(normalized_difference, axis=1)
    else:
        raise ValueError(
            "top_task_goal_distance requires distance_metric in "
            "{pusht_state, reacher_qpos}."
        )

    return distance.astype(np.float32), initial_success.astype(np.bool_)


def select_eval_indices(cfg: DictConfig, dataset, ep_indices):
    episode_len = get_episodes_length(dataset, ep_indices)
    # Horizon-scaling runs need every goal distance to use the same candidate
    # pool.  Validate starts against the largest requested offset while still
    # passing the horizon-specific goal_offset_steps to the evaluator below.
    sampling_goal_offset_steps = int(
        OmegaConf.select(
            cfg,
            "eval.sample_goal_offset_steps",
            default=cfg.eval.goal_offset_steps,
        )
    )
    max_start_idx = episode_len - sampling_goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_col = dataset.get_col_data(col_name)
    step_col = dataset.get_col_data("step_idx")
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in episode_col])

    valid_mask = step_col <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    manifest_name = OmegaConf.select(cfg, "eval.start_manifest", default=None)
    if manifest_name:
        manifest_path = Path(str(manifest_name))
        if not manifest_path.is_absolute():
            manifest_path = Path(__file__).resolve().parent / manifest_path
        with np.load(manifest_path, allow_pickle=False) as manifest:
            requested_episodes = np.asarray(manifest["eval_episodes"], dtype=np.int64)
            requested_steps = np.asarray(manifest["eval_start_idx"], dtype=np.int64)
            dataset_name = str(manifest["dataset_name"])
            goal_offset = int(manifest["goal_offset_steps"])
            eval_seed = int(manifest["eval_seed"])
        if dataset_name != str(cfg.eval.dataset_name):
            raise ValueError(f"Start manifest dataset {dataset_name} differs from {cfg.eval.dataset_name}")
        if goal_offset != int(cfg.eval.goal_offset_steps) or eval_seed != int(cfg.seed):
            raise ValueError("Start manifest goal offset or evaluation seed differs from config")
        if len(requested_episodes) != int(cfg.eval.num_eval) or len(requested_steps) != len(requested_episodes):
            raise ValueError("Start manifest episode count differs from eval.num_eval")
        # Look up each stored (episode, step) pair in the current HDF5. This also
        # detects a dataset whose episode indexing changed since the paper run.
        multiplier = int(np.max(step_col)) + 1
        row_keys = episode_col.astype(np.int64) * multiplier + step_col.astype(np.int64)
        order = np.argsort(row_keys)
        sorted_keys = row_keys[order]
        requested_keys = requested_episodes * multiplier + requested_steps
        locations = np.searchsorted(sorted_keys, requested_keys)
        if np.any(locations >= len(sorted_keys)) or not np.array_equal(sorted_keys[locations], requested_keys):
            raise ValueError("Start manifest contains episode/step pairs missing from this dataset")
        selected_indices = order[locations]
        if not np.all(valid_mask[selected_indices]):
            raise ValueError("Start manifest contains starts too close to an episode boundary")
        filter_mask = build_start_filter_mask(cfg, dataset, selected_indices)
        failed = int((~filter_mask).sum())
        if failed:
            raise ValueError(f"{failed} manifest starts fail the configured start filter")
        if not np.array_equal(np.sort(selected_indices), selected_indices):
            raise ValueError("Start manifest is not in dataset order")
        print(f"Loaded {len(selected_indices)} starts from {manifest_path}.")
        return selected_indices, dataset.get_row_data(selected_indices)

    start_filter_mask = build_start_filter_mask(cfg, dataset, valid_indices)
    valid_indices = valid_indices[start_filter_mask]
    print(
        len(valid_indices),
        "starting points remain after",
        OmegaConf.select(cfg, "eval.start_filter.mode", default="any"),
        "filter.",
    )
    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f"Only {len(valid_indices)} starting points remain after filtering; "
            f"need {cfg.eval.num_eval}."
        )

    g = np.random.default_rng(cfg.seed)
    mode = OmegaConf.select(cfg, "eval.start_selection.mode", default="random")
    if mode == "random":
        sampled = g.choice(len(valid_indices), size=cfg.eval.num_eval, replace=False)
        selected_indices = valid_indices[sampled]
    elif mode == "scene_balanced_components":
        scene_info = get_scene_component_info(cfg, dataset, valid_indices)
        component_masks = scene_info["masks"]
        components = OmegaConf.select(
            cfg,
            "eval.start_selection.components",
            default=["button", "cube", "drawer", "window"],
        )
        components = list(components)
        if not components:
            raise ValueError("scene_balanced_components requires at least one component.")

        base_quota = int(cfg.eval.num_eval) // len(components)
        remainder = int(cfg.eval.num_eval) % len(components)
        selected = []
        used = set()
        for i, component in enumerate(components):
            if component not in component_masks:
                raise ValueError(
                    f"scene_balanced_components requested {component}, "
                    "but the dataset has no matching Scene goal column."
                )
            quota = base_quota + (1 if i < remainder else 0)
            candidates = valid_indices[component_masks[component]]
            candidates = np.asarray([idx for idx in candidates if int(idx) not in used])
            if len(candidates) < quota:
                raise ValueError(
                    f"Only {len(candidates)} candidates for Scene component "
                    f"{component}; need {quota}."
                )
            sampled = g.choice(len(candidates), size=quota, replace=False)
            sampled_indices = candidates[sampled]
            selected.extend(sampled_indices.tolist())
            used.update(int(idx) for idx in sampled_indices)
            print(
                f"scene_balanced_components selected {quota} / {len(candidates)} "
                f"candidates for {component}."
            )
        selected_indices = np.asarray(selected, dtype=valid_indices.dtype)
    elif mode == "top_goal_distance":
        if "privileged_block_0_pos" not in dataset.column_names:
            raise ValueError("top_goal_distance requires privileged_block_0_pos.")
        pos = dataset.get_col_data("privileged_block_0_pos")
        goal_indices = valid_indices + max(int(cfg.eval.goal_offset_steps) - 1, 0)
        same_episode = episode_col[goal_indices] == episode_col[valid_indices]
        candidate_indices = valid_indices[same_episode]
        candidate_goal_indices = goal_indices[same_episode]
        goal_distance = np.linalg.norm(
            pos[candidate_goal_indices, :2] - pos[candidate_indices, :2],
            axis=1,
        )
        quantile = float(
            OmegaConf.select(cfg, "eval.start_selection.quantile", default=0.70)
        )
        threshold = np.quantile(goal_distance, quantile)
        hard_indices = candidate_indices[goal_distance >= threshold]
        if len(hard_indices) < cfg.eval.num_eval:
            raise ValueError(
                f"Only {len(hard_indices)} starts satisfy top_goal_distance "
                f"quantile={quantile}; need {cfg.eval.num_eval}."
            )
        sampled = g.choice(len(hard_indices), size=cfg.eval.num_eval, replace=False)
        selected_indices = hard_indices[sampled]
        print(
            f"top_goal_distance selected {len(hard_indices)} candidates "
            f"at quantile>={quantile:.2f} threshold={threshold:.4f}."
        )
    elif mode == "top_task_goal_distance":
        task_distance, initial_success = get_task_goal_distance(
            cfg, dataset, valid_indices
        )
        exclude_initial_success = bool(
            OmegaConf.select(
                cfg,
                "eval.start_selection.exclude_initial_success",
                default=True,
            )
        )
        candidate_mask = ~initial_success if exclude_initial_success else np.ones_like(
            initial_success, dtype=bool
        )
        candidate_indices = valid_indices[candidate_mask]
        candidate_distance = task_distance[candidate_mask]
        if len(candidate_indices) < cfg.eval.num_eval:
            raise ValueError(
                f"Only {len(candidate_indices)} task-mismatch starts remain; "
                f"need {cfg.eval.num_eval}."
            )

        quantile = float(
            OmegaConf.select(cfg, "eval.start_selection.quantile", default=0.70)
        )
        threshold = float(np.quantile(candidate_distance, quantile))
        hard_mask = candidate_distance >= threshold
        hard_indices = candidate_indices[hard_mask]
        if len(hard_indices) < cfg.eval.num_eval:
            raise ValueError(
                f"Only {len(hard_indices)} starts satisfy top_task_goal_distance "
                f"quantile={quantile}; need {cfg.eval.num_eval}."
            )
        sampled = g.choice(len(hard_indices), size=cfg.eval.num_eval, replace=False)
        selected_indices = hard_indices[sampled]
        print(
            f"top_task_goal_distance excluded {int(initial_success.sum())} initially "
            f"successful starts and selected from {len(hard_indices)} / "
            f"{len(candidate_indices)} candidates at quantile>={quantile:.2f} "
            f"threshold={threshold:.4f}."
        )
    else:
        raise ValueError(f"Unsupported eval.start_selection.mode: {mode}")

    selected_indices = np.sort(selected_indices)
    print(selected_indices)
    rows = dataset.get_row_data(selected_indices)
    return selected_indices, rows


def collect_step_arrays(env_unwrapped):
    info = env_unwrapped.get_step_info()
    arrays = {
        "qpos": np.asarray(info["qpos"]).copy(),
        "qvel": np.asarray(info["qvel"]).copy(),
        "observation": np.asarray(env_unwrapped.compute_observation()).copy(),
        "pixels": np.asarray(env_unwrapped.get_pixel_observation()).copy(),
    }
    for key, value in info.items():
        sanitized_key = key.replace("/", "_")
        if sanitized_key in arrays:
            continue
        if (
            key.startswith("privileged/")
            or key.startswith("proprio/")
            or key in {"button_states", "prev_button_states", "target"}
        ):
            try:
                arrays[sanitized_key] = np.asarray(value).copy()
            except Exception:
                continue
    return arrays


def set_env_state(env_unwrapped, qpos, qvel, button_states=None):
    kwargs = {}
    if button_states is not None:
        kwargs["button_states"] = button_states
    env_unwrapped.set_state(np.asarray(qpos), np.asarray(qvel), **kwargs)


def find_stack_wrapper(env):
    current = env
    while current is not None:
        if hasattr(current, "init_buffer") and hasattr(current, "buffers"):
            return current
        current = getattr(current, "env", None)
    return None


def extract_single_env_value(value, env_idx, num_envs):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim >= 2 and value.shape[0] == num_envs:
            return value[env_idx, -1].clone()
        if value.ndim >= 1 and value.shape[0] == num_envs:
            return value[env_idx].clone()
        return value.clone()
    if isinstance(value, np.ndarray):
        if value.ndim >= 2 and value.shape[0] == num_envs:
            return value[env_idx, -1].copy()
        if value.ndim >= 1 and value.shape[0] == num_envs:
            return value[env_idx].copy()
        return value.copy()
    return deepcopy(value)


def refresh_wrapper_buffers(world, base_infos, init_step, goal_step):
    for i, env in enumerate(world.envs.unwrapped.envs):
        stack_wrapper = find_stack_wrapper(env)
        if stack_wrapper is None:
            continue

        single_info = {
            key: extract_single_env_value(value, i, world.num_envs)
            for key, value in base_infos.items()
        }
        for key, value in init_step.items():
            if key in single_info:
                single_info[key] = deepcopy(value[i])
        for key, value in goal_step.items():
            if key in single_info:
                single_info[key] = deepcopy(value[i])
        stack_wrapper.init_buffer(single_info)


def render_goal_step_with_current_variation(world, init_step, goal_step):
    rendered_goals = []
    rendered_goal_observations = []

    has_goal_state = "goal_qpos" in goal_step and "goal_qvel" in goal_step
    if not has_goal_state:
        return {}, init_step

    for i, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        current_qpos = np.asarray(init_step["qpos"][i]).copy()
        current_qvel = np.asarray(init_step["qvel"][i]).copy()
        current_button_states = (
            np.asarray(init_step["button_states"][i]).copy()
            if "button_states" in init_step
            else None
        )
        goal_button_states = (
            np.asarray(goal_step["goal_button_states"][i]).copy()
            if "goal_button_states" in goal_step
            else None
        )

        set_env_state(
            env_unwrapped,
            goal_step["goal_qpos"][i],
            goal_step["goal_qvel"][i],
            goal_button_states,
        )
        rendered_goals.append(np.asarray(env_unwrapped.get_pixel_observation()).copy())
        rendered_goal_observations.append(
            np.asarray(env_unwrapped.compute_observation()).copy()
        )

        set_env_state(env_unwrapped, current_qpos, current_qvel, current_button_states)

    init_refresh = defaultdict(list)
    for env in world.envs.unwrapped.envs:
        for key, value in collect_step_arrays(env.unwrapped).items():
            init_refresh[key].append(value)

    refreshed_init_step = deepcopy(init_step)
    for key, values in init_refresh.items():
        refreshed_init_step[key] = np.stack(values)

    return {
        "goal": np.stack(rendered_goals),
        "goal_observation": np.stack(rendered_goal_observations),
    }, refreshed_init_step


def apply_initial_perturbations(cfg: DictConfig, init_step: dict):
    perturb_cfg = OmegaConf.select(cfg, "eval.ood.initial_perturb", default={})
    if not perturb_cfg or not perturb_cfg.get("enabled", False):
        return np.zeros((len(init_step["qpos"]), 2), dtype=np.float32)

    qpos_key = perturb_cfg.get("qpos_key", "qpos")
    pos_key = perturb_cfg.get("position_key", "privileged_block_0_pos")
    qpos_start = int(perturb_cfg.get("qpos_start", 14))
    radius = float(perturb_cfg.get("xy_radius", 0.04))
    x_bounds = np.asarray(perturb_cfg.get("x_bounds", [0.30, 0.55]), dtype=np.float64)
    y_bounds = np.asarray(perturb_cfg.get("y_bounds", [-0.30, 0.30]), dtype=np.float64)

    if qpos_key not in init_step or pos_key not in init_step:
        raise ValueError(f"Initial perturbation requires {qpos_key} and {pos_key}.")

    qpos = init_step[qpos_key].copy()
    pos = init_step[pos_key].copy()
    if qpos.shape[1] < qpos_start + 3:
        raise ValueError(
            f"{qpos_key} shape {qpos.shape} is too small for qpos_start={qpos_start}."
        )
    if not np.allclose(qpos[:, qpos_start : qpos_start + 3], pos[:, :3], atol=1e-5):
        raise ValueError(
            f"{qpos_key}[{qpos_start}:{qpos_start + 3}] does not match {pos_key}."
        )

    perturb_xy = np.zeros((qpos.shape[0], 2), dtype=np.float32)
    for i in range(qpos.shape[0]):
        rng = np.random.default_rng(int(cfg.seed) + i)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        delta = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        old_xy = qpos[i, qpos_start : qpos_start + 2].copy()
        new_xy = old_xy + delta
        new_xy[0] = np.clip(new_xy[0], x_bounds[0], x_bounds[1])
        new_xy[1] = np.clip(new_xy[1], y_bounds[0], y_bounds[1])
        applied = new_xy - old_xy
        qpos[i, qpos_start : qpos_start + 2] = new_xy
        pos[i, :2] = new_xy
        perturb_xy[i] = applied.astype(np.float32)

    init_step[qpos_key] = qpos
    init_step[pos_key] = pos
    return perturb_xy


def build_scene_component_payload(init_step: dict, goal_step: dict, final_step: dict | None = None):
    if "qpos" not in init_step:
        return {}

    num_envs = len(init_step["qpos"])
    payload = {}
    changed_masks = []

    if (
        "privileged_block_0_pos" in init_step
        and "goal_privileged_block_0_pos" in goal_step
    ):
        block_goal_distance = np.linalg.norm(
            goal_step["goal_privileged_block_0_pos"][:, :3]
            - init_step["privileged_block_0_pos"][:, :3],
            axis=1,
        ).astype(np.float32)
        payload["scene_goal_block_distance"] = block_goal_distance
        changed_masks.append(block_goal_distance > 0.04)
        if final_step and "privileged_block_0_pos" in final_step:
            payload["scene_final_block_distance"] = np.linalg.norm(
                goal_step["goal_privileged_block_0_pos"][:, :3]
                - final_step["privileged_block_0_pos"][:, :3],
                axis=1,
            ).astype(np.float32)

    if "button_states" in init_step and "goal_button_states" in goal_step:
        button_changed = np.any(
            init_step["button_states"] != goal_step["goal_button_states"], axis=1
        )
        payload["scene_goal_button_changed"] = button_changed
        changed_masks.append(button_changed)
        if final_step and "button_states" in final_step:
            payload["scene_final_button_match"] = np.all(
                final_step["button_states"] == goal_step["goal_button_states"], axis=1
            )

    if (
        "privileged_drawer_pos" in init_step
        and "goal_privileged_drawer_pos" in goal_step
    ):
        drawer_delta = np.abs(
            goal_step["goal_privileged_drawer_pos"].reshape(num_envs, -1)[:, 0]
            - init_step["privileged_drawer_pos"].reshape(num_envs, -1)[:, 0]
        ).astype(np.float32)
        payload["scene_goal_drawer_delta"] = drawer_delta
        changed_masks.append(drawer_delta > 0.04)
        if final_step and "privileged_drawer_pos" in final_step:
            payload["scene_final_drawer_delta"] = np.abs(
                goal_step["goal_privileged_drawer_pos"].reshape(num_envs, -1)[:, 0]
                - final_step["privileged_drawer_pos"].reshape(num_envs, -1)[:, 0]
            ).astype(np.float32)

    if (
        "privileged_window_pos" in init_step
        and "goal_privileged_window_pos" in goal_step
    ):
        window_delta = np.abs(
            goal_step["goal_privileged_window_pos"].reshape(num_envs, -1)[:, 0]
            - init_step["privileged_window_pos"].reshape(num_envs, -1)[:, 0]
        ).astype(np.float32)
        payload["scene_goal_window_delta"] = window_delta
        changed_masks.append(window_delta > 0.04)
        if final_step and "privileged_window_pos" in final_step:
            payload["scene_final_window_delta"] = np.abs(
                goal_step["goal_privileged_window_pos"].reshape(num_envs, -1)[:, 0]
                - final_step["privileged_window_pos"].reshape(num_envs, -1)[:, 0]
            ).astype(np.float32)

    if changed_masks:
        payload["scene_goal_changed_component_count"] = np.stack(changed_masks).sum(
            axis=0
        ).astype(np.int32)

    return payload


def build_reset_options(
    cfg: DictConfig,
    world,
    init_step: dict,
):
    variation_prefix = "variation."
    dataset_variations = {
        k.removeprefix(variation_prefix): v
        for k, v in init_step.items()
        if k.startswith(variation_prefix)
    }

    visual_cfg = OmegaConf.select(cfg, "eval.ood.visual_variation", default={})
    configured_values = {}
    if visual_cfg and visual_cfg.get("enabled", False):
        configured_values = OmegaConf.to_container(
            visual_cfg.get("values", {}), resolve=True
        )

    options = [{} for _ in range(world.num_envs)]
    if not dataset_variations and not configured_values:
        return options, configured_values

    for key, value in dataset_variations.items():
        value_array = np.asarray(value)
        for i in range(world.num_envs):
            options[i].setdefault("variation", []).append(key)
            options[i].setdefault("variation_values", {})[key] = value_array[i]

    for key, value in configured_values.items():
        value_array = np.asarray(value)
        for i in range(world.num_envs):
            options[i].setdefault("variation", []).append(key)
            options[i].setdefault("variation_values", {})[key] = value_array

    return options, configured_values


def extract_goal_step_at(data, goal_keys, offset: int):
    goal_step = {}
    for goal_key in goal_keys:
        source_key = "pixels" if goal_key == "goal" else goal_key[len("goal_") :]
        values = []
        for ep in data:
            value = ep[source_key][offset]
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            values.append(value)
        goal_step[goal_key] = np.stack(values)
    return goal_step


def broadcast_step(step: dict, shape_prefix):
    return {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
        for k, v in step.items()
    }


def get_rolling_goal_offset(cfg: DictConfig, step_idx: int, target_len: int) -> int:
    mode = str(OmegaConf.select(cfg, "eval.rolling_goal.mode", default="rolling"))
    lookahead_steps = int(
        OmegaConf.select(cfg, "eval.rolling_goal.lookahead_steps", default=25)
    )
    if mode == "piecewise":
        goal_steps = config_value_to_container(
            OmegaConf.select(cfg, "eval.rolling_goal.goal_steps", default=[25, 50])
        )
        switch_steps = config_value_to_container(
            OmegaConf.select(cfg, "eval.rolling_goal.switch_steps", default=[50])
        )
        if len(goal_steps) != len(switch_steps) + 1:
            raise ValueError(
                "piecewise rolling_goal requires len(goal_steps) == "
                "len(switch_steps) + 1."
            )
        phase = 0
        for switch_step in switch_steps:
            if step_idx < int(switch_step):
                break
            phase += 1
        goal_step = int(goal_steps[phase])
        return min(max(goal_step - 1, 0), target_len - 1)
    if mode == "two_phase":
        switch_step = int(
            OmegaConf.select(
                cfg,
                "eval.rolling_goal.switch_step",
                default=lookahead_steps * 2,
            )
        )
        if step_idx < switch_step:
            return min(max(lookahead_steps - 1, 0), target_len - 1)
        return target_len - 1
    if mode != "rolling":
        raise ValueError(f"Unsupported eval.rolling_goal.mode={mode!r}")
    # Dataset chunks use frame 0 as the initial state, so a 25-step goal is
    # represented by offset 24, matching the fixed-goal convention.
    offset = step_idx + max(lookahead_steps - 1, 0)
    return min(offset, target_len - 1)


def config_value_to_container(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def get_handoff_diagnostic_steps(
    cfg: DictConfig,
    rolling_goal_enabled: bool,
    eval_budget: int,
) -> list[int]:
    enabled = bool(
        OmegaConf.select(
            cfg,
            "eval.handoff_diagnostics.enabled",
            default=rolling_goal_enabled,
        )
    )
    if not enabled:
        return []

    configured_steps = OmegaConf.select(
        cfg, "eval.handoff_diagnostics.steps", default=None
    )
    if configured_steps is not None:
        steps = config_value_to_container(configured_steps)
    else:
        mode = str(OmegaConf.select(cfg, "eval.rolling_goal.mode", default="rolling"))
        if mode == "two_phase":
            steps = [
                int(
                    OmegaConf.select(
                        cfg, "eval.rolling_goal.switch_step", default=-1
                    )
                )
            ]
        elif mode == "piecewise":
            steps = config_value_to_container(
                OmegaConf.select(cfg, "eval.rolling_goal.switch_steps", default=[])
            )
        else:
            steps = []

    valid_steps = []
    for step in steps:
        step = int(step)
        if 0 < step < eval_budget:
            valid_steps.append(step)
    return sorted(set(valid_steps))


def collect_world_step_arrays(world):
    collected = defaultdict(list)
    for env in world.envs.unwrapped.envs:
        for key, value in collect_step_arrays(env.unwrapped).items():
            collected[key].append(value)
    return {key: np.stack(values) for key, values in collected.items()}


def build_handoff_diagnostic_payload(
    cfg: DictConfig,
    handoff_steps: list[int],
    handoff_snapshots: dict[int, dict],
    data,
    goal_keys: list[str],
    final_goal_step: dict,
    target_len: int,
):
    payload = {}
    if not handoff_steps:
        return payload

    payload["handoff_diagnostic_steps"] = np.asarray(handoff_steps, dtype=np.int32)

    block_key = "privileged_block_0_pos"
    effector_key = "proprio_effector_pos"
    contact_key = "proprio_gripper_contact"
    if block_key in final_goal_step:
        final_goal_block_pos = final_goal_step[block_key][:, :3]
    elif f"goal_{block_key}" in final_goal_step:
        final_goal_block_pos = final_goal_step[f"goal_{block_key}"][:, :3]
    else:
        final_goal_block_pos = None

    block_z = []
    contact = []
    effector_block_distance = []
    block_to_phase_goal_distance = []
    block_to_final_goal_distance = []
    previous_goal_steps = []
    next_goal_steps = []

    for step in handoff_steps:
        snapshot = handoff_snapshots[step]
        block_pos = snapshot.get(block_key)
        effector_pos = snapshot.get(effector_key)
        contact_value = snapshot.get(contact_key)
        previous_goal_offset = get_rolling_goal_offset(cfg, step - 1, target_len)
        next_goal_offset = get_rolling_goal_offset(cfg, step, target_len)
        previous_goal_step = extract_goal_step_at(data, goal_keys, previous_goal_offset)
        previous_goal_steps.append(previous_goal_offset + 1)
        next_goal_steps.append(next_goal_offset + 1)

        if block_pos is not None:
            block_z.append(block_pos[:, 2].astype(np.float32))
            phase_goal_pos = previous_goal_step.get(f"goal_{block_key}")
            if phase_goal_pos is not None:
                block_to_phase_goal_distance.append(
                    np.linalg.norm(phase_goal_pos[:, :3] - block_pos[:, :3], axis=1)
                    .astype(np.float32)
                )
            else:
                block_to_phase_goal_distance.append(
                    np.full(len(block_pos), np.nan, dtype=np.float32)
                )
            if final_goal_block_pos is not None:
                block_to_final_goal_distance.append(
                    np.linalg.norm(final_goal_block_pos - block_pos[:, :3], axis=1)
                    .astype(np.float32)
                )
            else:
                block_to_final_goal_distance.append(
                    np.full(len(block_pos), np.nan, dtype=np.float32)
                )
        else:
            num_envs = len(next(iter(snapshot.values())))
            block_z.append(np.full(num_envs, np.nan, dtype=np.float32))
            block_to_phase_goal_distance.append(
                np.full(num_envs, np.nan, dtype=np.float32)
            )
            block_to_final_goal_distance.append(
                np.full(num_envs, np.nan, dtype=np.float32)
            )

        if block_pos is not None and effector_pos is not None:
            effector_block_distance.append(
                np.linalg.norm(effector_pos[:, :3] - block_pos[:, :3], axis=1)
                .astype(np.float32)
            )
        else:
            effector_block_distance.append(
                np.full(len(block_z[-1]), np.nan, dtype=np.float32)
            )

        if contact_value is not None:
            contact.append(np.asarray(contact_value, dtype=np.float32).reshape(len(block_z[-1]), -1))
        else:
            contact.append(
                np.full((len(block_z[-1]), 1), np.nan, dtype=np.float32)
            )

    payload["handoff_previous_goal_steps"] = np.asarray(
        previous_goal_steps, dtype=np.int32
    )
    payload["handoff_next_goal_steps"] = np.asarray(next_goal_steps, dtype=np.int32)
    payload["handoff_block_z"] = np.stack(block_z)
    payload["handoff_gripper_contact"] = np.stack(contact)
    payload["handoff_effector_block_distance"] = np.stack(effector_block_distance)
    payload["handoff_block_to_phase_goal_distance"] = np.stack(
        block_to_phase_goal_distance
    )
    payload["handoff_block_to_final_goal_distance"] = np.stack(
        block_to_final_goal_distance
    )
    return payload


def evaluate_from_dataset_with_ood(
    cfg: DictConfig,
    world,
    dataset,
    episodes_idx,
    start_steps,
    callables,
    video_path,
):
    ep_idx_arr = np.asarray(episodes_idx)
    start_steps_arr = np.asarray(start_steps)
    video_filter = str(OmegaConf.select(cfg, "output.video_filter", default="all"))
    if video_filter not in {"all", "success", "failure"}:
        raise ValueError(f"Unknown video filter: {video_filter}")
    video_candidates = video_reference_mask(
        ep_idx_arr,
        start_steps_arr,
        OmegaConf.select(cfg, "output.video_reference_successes", default=None),
        settings={
            "eval_seed": cfg.seed,
            "num_eval": cfg.eval.num_eval,
            "goal_offset_steps": cfg.eval.goal_offset_steps,
            "eval_budget": cfg.eval.eval_budget,
            "plan_horizon": cfg.plan_config.horizon,
            "receding_horizon": cfg.plan_config.receding_horizon,
            "action_block": cfg.plan_config.action_block,
        },
    )
    end_steps = start_steps_arr + cfg.eval.goal_offset_steps
    data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)
    columns = dataset.column_names

    init_step_per_env = defaultdict(list)
    goal_step_per_env = defaultdict(list)
    for ep in data:
        for col in columns:
            if col.startswith("goal"):
                continue
            if col.startswith("pixels"):
                ep[col] = ep[col].permute(0, 2, 3, 1)
            if not isinstance(ep[col], (torch.Tensor | np.ndarray)):
                continue
            init_data = ep[col][0]
            goal_data = ep[col][-1]
            if not isinstance(init_data, (np.ndarray | torch.Tensor)):
                continue
            if isinstance(init_data, torch.Tensor):
                init_data = init_data.numpy()
            if isinstance(goal_data, torch.Tensor):
                goal_data = goal_data.numpy()
            init_step_per_env[col].append(init_data)
            goal_step_per_env[col].append(goal_data)

    init_step = {k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()}
    goal_step = {}
    for key, value in goal_step_per_env.items():
        key = "goal" if key == "pixels" else f"goal_{key}"
        goal_step[key] = np.stack(value)

    if (
        "goal_privileged_block_0_pos" in goal_step
        and "privileged_block_0_pos" in init_step
    ):
        ood_goal_distance = np.linalg.norm(
            goal_step["goal_privileged_block_0_pos"][:, :2]
            - init_step["privileged_block_0_pos"][:, :2],
            axis=1,
        ).astype(np.float32)
    else:
        ood_goal_distance = np.full(len(episodes_idx), np.nan, dtype=np.float32)
    if "proprio_effector_pos" in init_step and "privileged_block_0_pos" in init_step:
        ood_effector_block_distance = np.linalg.norm(
            init_step["proprio_effector_pos"][:, :3]
            - init_step["privileged_block_0_pos"][:, :3],
            axis=1,
        ).astype(np.float32)
    else:
        ood_effector_block_distance = np.full(
            len(episodes_idx), np.nan, dtype=np.float32
        )
    if "privileged_block_0_pos" in init_step:
        ood_start_block_z = init_step["privileged_block_0_pos"][:, 2].astype(np.float32)
    else:
        ood_start_block_z = np.full(len(episodes_idx), np.nan, dtype=np.float32)
    ood_start_gripper_contact = np.asarray(
        init_step.get(
            "proprio_gripper_contact",
            np.full(len(episodes_idx), np.nan, dtype=np.float32),
        ),
        dtype=np.float32,
    )
    ood_initial_perturb_xy = apply_initial_perturbations(cfg, init_step)

    seeds = init_step.get("seed")
    options, ood_visual_variation_values = build_reset_options(cfg, world, init_step)

    init_with_goal = deepcopy(init_step)
    init_with_goal.update(deepcopy(goal_step))
    world.reset(seed=seeds, options=options)
    reset_infos = deepcopy(world.infos)

    callables = callables or []
    for i, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                continue
            method = getattr(env_unwrapped, method_name)
            prepared_args = {}
            for args_name, args_data in spec.get("args", spec).items():
                value = args_data.get("value", None)
                is_in_dataset = args_data.get("in_dataset", True)
                if is_in_dataset:
                    if value not in init_with_goal:
                        continue
                    prepared_args[args_name] = deepcopy(init_with_goal[value][i])
                else:
                    prepared_args[args_name] = args_data.get("value")
            method(**prepared_args)

    refresh_pixels = bool(
        OmegaConf.select(cfg, "eval.ood.visual_variation.enabled", default=False)
    )
    if bool(OmegaConf.select(cfg, "eval.ood.initial_perturb.enabled", default=False)) or refresh_pixels:
        refreshed = defaultdict(list)
        for env in world.envs.unwrapped.envs:
            for key, value in collect_step_arrays(env.unwrapped).items():
                refreshed[key].append(value)
        for key, values in refreshed.items():
            stacked = np.stack(values)
            init_step[key] = stacked

    expert_target_frames = torch.stack([ep["pixels"] for ep in data]).numpy()
    target_len = expert_target_frames.shape[1]
    rolling_goal_enabled = bool(
        OmegaConf.select(cfg, "eval.rolling_goal.enabled", default=False)
    )
    handoff_steps = get_handoff_diagnostic_steps(
        cfg, rolling_goal_enabled, int(cfg.eval.eval_budget)
    )
    handoff_snapshots = {}

    if refresh_pixels:
        if rolling_goal_enabled:
            raise NotImplementedError(
                "rolling_goal is not implemented with visual_variation refresh."
            )
        rendered_goal_step, init_step = render_goal_step_with_current_variation(
            world, init_step, goal_step
        )
        goal_step.update(rendered_goal_step)

    flat_init_step = deepcopy(init_step)
    flat_goal_step = deepcopy(goal_step)

    refresh_wrapper_buffers(world, reset_infos, init_step, goal_step)

    shape_prefix = world.infos["pixels"].shape[:2]
    init_step = broadcast_step(init_step, shape_prefix)
    final_goal_step = broadcast_step(goal_step, shape_prefix)
    goal_keys = list(goal_step.keys())
    world.infos.update(deepcopy(init_step))
    world.infos.update(deepcopy(final_goal_step))

    results = {
        "success_rate": 0.0,
        "episode_successes": np.zeros(len(episodes_idx)),
        "seeds": seeds,
    }
    save_video = bool(OmegaConf.select(cfg, "output.save_video", default=True))
    if save_video:
        import imageio

        target_frames = expert_target_frames
        video_frames = np.empty(
            (world.num_envs, cfg.eval.eval_budget, *world.infos["pixels"].shape[-3:]),
            dtype=np.uint8,
        )

    for i in range(cfg.eval.eval_budget):
        if i in handoff_steps:
            handoff_snapshots[i] = collect_world_step_arrays(world)
        if save_video:
            video_frames[:, i] = world.infos["pixels"][:, -1]
        if rolling_goal_enabled:
            goal_offset = get_rolling_goal_offset(cfg, i, target_len)
            planner_goal_step = broadcast_step(
                extract_goal_step_at(data, goal_keys, goal_offset),
                shape_prefix,
            )
        else:
            planner_goal_step = final_goal_step
        world.infos.update(deepcopy(planner_goal_step))
        world.step()
        results["episode_successes"] = np.logical_or(
            results["episode_successes"], world.terminateds
        )
        world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))

    final_step = defaultdict(list)
    for env in world.envs.unwrapped.envs:
        for key, value in collect_step_arrays(env.unwrapped).items():
            final_step[key].append(value)
    final_step = {key: np.stack(values) for key, values in final_step.items()}

    if save_video:
        video_frames[:, -1] = world.infos["pixels"][:, -1]
        video_path_obj = Path(video_path)
        video_path_obj.mkdir(parents=True, exist_ok=True)
        saved_indices = video_output_indices(
            results["episode_successes"], video_candidates, video_filter
        )
        results["video_saved_indices"] = saved_indices
        print(f"Saving {len(saved_indices)}/{world.num_envs} videos ({video_filter})")
        for i in saved_indices:
            out = imageio.get_writer(
                video_path_obj / f"rollout_{i}.mp4",
                fps=15,
                codec="libx264",
            )
            goal_panel = final_goal_step["goal"][i, -1]
            goals = np.vstack([goal_panel, goal_panel])
            for t in range(cfg.eval.eval_budget):
                stacked_frame = np.vstack(
                    [video_frames[i, t], target_frames[i, t % target_len]]
                )
                out.append_data(np.hstack([stacked_frame, goals]))
            out.close()

    results["success_rate"] = (
        float(np.sum(results["episode_successes"])) / len(episodes_idx) * 100.0
    )
    scene_component_payload = build_scene_component_payload(
        flat_init_step,
        flat_goal_step,
        final_step,
    )
    handoff_diagnostic_payload = build_handoff_diagnostic_payload(
        cfg,
        handoff_steps,
        handoff_snapshots,
        data,
        goal_keys,
        flat_goal_step,
        target_len,
    )
    return results, {
        "ood_goal_distance": ood_goal_distance,
        "ood_start_filter_mode": np.array(
            [OmegaConf.select(cfg, "eval.start_filter.mode", default="any")]
        ),
        "ood_start_block_z": ood_start_block_z,
        "ood_start_gripper_contact": ood_start_gripper_contact,
        "ood_start_effector_block_distance": ood_effector_block_distance,
        "ood_initial_perturb_xy": ood_initial_perturb_xy,
        "ood_visual_variation_values": np.array(
            [
                json.dumps(
                    {
                        key: np.asarray(value).tolist()
                        for key, value in ood_visual_variation_values.items()
                    },
                    sort_keys=True,
                )
            ]
        ),
        "rolling_goal_enabled": np.array([rolling_goal_enabled], dtype=np.bool_),
        "rolling_goal_lookahead_steps": np.array(
            [
                int(
                    OmegaConf.select(
                        cfg, "eval.rolling_goal.lookahead_steps", default=25
                    )
                )
            ],
            dtype=np.int32,
        ),
        "rolling_goal_mode": np.array(
            [str(OmegaConf.select(cfg, "eval.rolling_goal.mode", default="rolling"))]
        ),
        "rolling_goal_switch_step": np.array(
            [
                int(
                    OmegaConf.select(
                        cfg, "eval.rolling_goal.switch_step", default=-1
                    )
                )
            ],
            dtype=np.int32,
        ),
        "rolling_goal_goal_steps": np.array(
            [
                json.dumps(
                    config_value_to_container(
                        OmegaConf.select(
                            cfg, "eval.rolling_goal.goal_steps", default=[]
                        )
                    )
                )
            ]
        ),
        "rolling_goal_switch_steps": np.array(
            [
                json.dumps(
                    config_value_to_container(
                        OmegaConf.select(
                            cfg, "eval.rolling_goal.switch_steps", default=[]
                        )
                    )
                )
            ]
        ),
        **handoff_diagnostic_payload,
        **scene_component_payload,
    }


def evaluate_policy(cfg: DictConfig):
    """Run planning evaluation and return raw metrics plus metadata."""
    policy = cfg.get("policy", "random")
    if policy != "random":
        assert (
            cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
        ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world_height = int(OmegaConf.select(cfg, "world.height", default=cfg.eval.img_size))
    world_width = int(OmegaConf.select(cfg, "world.width", default=cfg.eval.img_size))
    world = swm.World(
        **cfg.world,
        image_shape=(world_height, world_width),
    )

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    if policy != "random":
        device_name = OmegaConf.select(cfg, "solver.device", default="cuda")
        device = torch.device(device_name)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(device.index if device.index is not None else 0)

        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_dir = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )
    video_path = results_dir
    video_subdir = OmegaConf.select(cfg, "output.video_subdir", default=None)
    if video_subdir:
        video_path = results_dir / str(video_subdir)

    selected_indices, eval_rows = select_eval_indices(cfg, dataset, ep_indices)
    eval_episodes = eval_rows[col_name]
    eval_start_idx = eval_rows["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    (
        start_mode,
        start_filter_mode,
        perturb_enabled,
        visual_variation_enabled,
    ) = get_eval_settings(cfg)
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)

    start_time = time.time()
    extra_payload = {
        "eval_seed": np.asarray([int(cfg.seed)], dtype=np.int64),
        "num_eval": np.asarray([int(cfg.eval.num_eval)], dtype=np.int64),
        "goal_offset_steps": np.asarray(
            [int(cfg.eval.goal_offset_steps)], dtype=np.int64
        ),
        "sample_goal_offset_steps": np.asarray(
            [
                int(
                    OmegaConf.select(
                        cfg,
                        "eval.sample_goal_offset_steps",
                        default=cfg.eval.goal_offset_steps,
                    )
                )
            ],
            dtype=np.int64,
        ),
        "eval_budget": np.asarray([int(cfg.eval.eval_budget)], dtype=np.int64),
        "plan_horizon": np.asarray(
            [int(cfg.plan_config.horizon)], dtype=np.int64
        ),
        "receding_horizon": np.asarray(
            [int(cfg.plan_config.receding_horizon)], dtype=np.int64
        ),
        "action_block": np.asarray(
            [int(cfg.plan_config.action_block)], dtype=np.int64
        ),
    }
    if start_mode == "top_task_goal_distance":
        selected_task_distance, selected_initial_success = get_task_goal_distance(
            cfg, dataset, selected_indices
        )
        extra_payload.update(
            {
                "hardstart_task_goal_distance": selected_task_distance,
                "hardstart_initial_success": selected_initial_success,
            }
        )
    selection_only = bool(
        OmegaConf.select(cfg, "eval.selection_only", default=False)
    )
    if (
        (
            selection_only
            or (start_mode == "random" and start_filter_mode == "any")
        )
        and not perturb_enabled
        and not visual_variation_enabled
    ):
        eval_kwargs = dict(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=callables,
            save_video=bool(OmegaConf.select(cfg, "output.save_video", default=True)),
            video_path=video_path,
        )
        if len(eval_episodes) == world.num_envs:
            metrics = world.evaluate_from_dataset(**eval_kwargs)
        else:
            metrics = evaluate_from_dataset_serial(world, **eval_kwargs)
    else:
        metrics, ood_payload = evaluate_from_dataset_with_ood(
            cfg,
            world,
            dataset,
            episodes_idx=eval_episodes.tolist(),
            start_steps=eval_start_idx.tolist(),
            callables=callables,
            video_path=video_path,
        )
        extra_payload.update(ood_payload)
    end_time = time.time()

    print(metrics)
    return {
        "metrics": metrics,
        "evaluation_time": end_time - start_time,
        "results_dir": results_dir,
        "eval_episodes": eval_episodes,
        "eval_start_idx": eval_start_idx,
        "extra_payload": extra_payload,
    }


def evaluate_from_dataset_serial(
    world,
    dataset,
    *,
    start_steps,
    goal_offset_steps,
    eval_budget,
    episodes_idx,
    callables,
    save_video,
    video_path,
) -> dict:
    """Evaluate more dataset starts than env instances by reusing envs serially."""

    if world.num_envs != 1:
        raise ValueError(
            "Serial dataset evaluation currently expects world.num_envs=1. "
            f"Got {world.num_envs}."
        )

    successes = []
    seeds = []
    video_root = Path(video_path)
    serial_tmp_root = video_root / "_serial_tmp"
    for idx, (episode_idx, start_step) in enumerate(zip(episodes_idx, start_steps)):
        run_video_path = serial_tmp_root / f"{idx:04d}"
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=[int(start_step)],
            goal_offset_steps=goal_offset_steps,
            eval_budget=eval_budget,
            episodes_idx=[int(episode_idx)],
            callables=callables,
            save_video=save_video,
            video_path=run_video_path,
        )
        if save_video:
            src_video = run_video_path / "rollout_0.mp4"
            dst_video = video_root / f"rollout_{idx}.mp4"
            if src_video.exists():
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                if dst_video.exists():
                    dst_video.unlink()
                shutil.move(str(src_video), str(dst_video))
                try:
                    run_video_path.rmdir()
                except OSError:
                    pass
        successes.append(np.asarray(metrics.get("episode_successes", [False])).reshape(-1)[0])
        metric_seeds = metrics.get("seeds")
        if metric_seeds is not None:
            seeds.append(np.asarray(metric_seeds).reshape(-1)[0])

    if save_video:
        try:
            serial_tmp_root.rmdir()
        except OSError:
            pass

    successes_arr = np.asarray(successes, dtype=np.bool_)
    out = {
        "success_rate": float(np.sum(successes_arr)) / len(successes_arr) * 100.0,
        "episode_successes": successes_arr,
    }
    if seeds:
        out["seeds"] = np.asarray(seeds)
    else:
        out["seeds"] = None
    return out

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of a policy against the dataset-backed planning protocol."""
    result = evaluate_policy(cfg)

    results_path = result["results_dir"] / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {result['metrics']}\n")
        f.write(f"evaluation_time: {result['evaluation_time']} seconds\n")

    if cfg.output.get("save_structured", True):
        save_structured_results(
            results_path,
            cfg,
            result["metrics"],
            result["evaluation_time"],
            result["eval_episodes"],
            result["eval_start_idx"],
            result.get("extra_payload"),
        )


if __name__ == "__main__":
    run()
