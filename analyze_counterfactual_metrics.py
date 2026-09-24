import gc
import hashlib
import json
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/lewm-matplotlib-{os.getuid()}")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


# MUJOCO_EGL_DEVICE_ID, if set, indexes EGL devices independently of CUDA visibility.


import hydra
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict

from analyze_planner_alignment import (
    TracingCEMSolver,
    build_base_info,
    build_eval_case,
    build_plan_namespace,
    build_process_and_transform,
    build_world_from_eval_config,
    clip_flat_candidates,
    compute_model_costs,
    evaluate_env_costs,
    format_duration,
    get_flat_action_bounds,
    get_solver_kwargs,
    load_eval_config,
    move_info_to_device,
    prepare_info_dict,
    reset_world_to_case,
    sample_eval_rows,
    spearman_corr,
    step_world_with_blocks,
    unflatten_action_candidates,
)
from analyze_residual_diagnostics import (
    collect_checkpoint_paths,
    get_cache_dir,
    load_model_object,
    move_batch_to_device,
    normalize_train_cfg,
    parse_epoch,
)
from train import compute_model_outputs
from utils import get_column_normalizer, get_img_preprocessor


def suppress_gym_box_warnings():
    warning_patterns = [
        r"^WARN: Casting input x to numpy array\.$",
        r"^WARN: Box low's precision lowered by casting to float32, current low\.dtype=float64$",
        r"^WARN: Box high's precision lowered by casting to float32, current high\.dtype=float64$",
    ]
    for pattern in warning_patterns:
        warnings.filterwarnings("ignore", message=pattern, category=UserWarning)


suppress_gym_box_warnings()


def build_val_dataset_and_split(cfg: DictConfig):
    dataset = swm.data.HDF5Dataset(
        **cfg.data.dataset,
        transform=None,
        cache_dir=get_cache_dir(cfg),
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    for col in cfg.data.dataset.keys_to_cache:
        if col.startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    _, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    return dataset, val_set


def build_loader(dataset, batch_size: int, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )


def select_subset_indices(total: int, count: int, seed: int):
    if count >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total, size=count, replace=False))
    return selected.astype(np.int64)


def split_eval_and_bank_indices(total: int, eval_count: int, bank_count: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    eval_count = min(eval_count, total)
    eval_local = np.sort(perm[:eval_count]).astype(np.int64)
    remaining = perm[eval_count:]
    if remaining.size >= bank_count:
        bank_local = np.sort(remaining[:bank_count]).astype(np.int64)
    elif total >= bank_count:
        bank_local = np.sort(perm[:bank_count]).astype(np.int64)
    else:
        bank_local = np.arange(total, dtype=np.int64)
    return eval_local, bank_local


def collect_action_bank(bank_loader, history_size: int, rollout_horizon: int):
    last_actions = []
    suffix_actions = []
    for batch in bank_loader:
        action = torch.nan_to_num(batch["action"], 0.0)
        last_actions.append(action[:, history_size - 1].cpu())
        suffix_actions.append(action[:, history_size - 1 : history_size - 1 + rollout_horizon].cpu())
    return {
        "last_actions": torch.cat(last_actions, dim=0),
        "suffix_actions": torch.cat(suffix_actions, dim=0),
    }


def safe_mean(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.nanmean(array))


def safe_std(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.nanstd(array))


def safe_corr(x, y, eps: float = 1.0e-8):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x ** 2).sum()) * float((y ** 2).sum()))
    if denom <= eps:
        return float("nan")
    return float((x * y).sum() / denom)


def rankdata(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def safe_spearman(x, y, eps: float = 1.0e-8):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return safe_corr(rankdata(x[mask]), rankdata(y[mask]), eps=eps)


def zscore(values: np.ndarray, eps: float):
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if std <= eps:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def sample_with_padding(indices: np.ndarray, count: int, mode: str):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.full(count, -1, dtype=np.int64)
    if indices.size >= count:
        if mode == "front":
            return indices[:count]
        if mode == "back":
            return indices[-count:]
        picks = np.linspace(0, indices.size - 1, count)
        return indices[np.round(picks).astype(np.int64)]
    pad = np.full(count - indices.size, int(indices[-1]), dtype=np.int64)
    return np.concatenate([indices, pad], axis=0)


def select_easy_medium_hard_indices(distance_matrix: np.ndarray, count_per_bucket: int, exclusion_eps: float):
    batch_size, _ = distance_matrix.shape
    easy = np.empty((batch_size, count_per_bucket), dtype=np.int64)
    medium = np.empty((batch_size, count_per_bucket), dtype=np.int64)
    hard = np.empty((batch_size, count_per_bucket), dtype=np.int64)

    for row in range(batch_size):
        distances = distance_matrix[row]
        sorted_idx = np.argsort(distances)
        sorted_idx = sorted_idx[distances[sorted_idx] > exclusion_eps]
        if sorted_idx.size == 0:
            sorted_idx = np.argsort(distances)

        buckets = np.array_split(sorted_idx, 3)
        hard_bucket = buckets[0] if len(buckets) > 0 else sorted_idx
        medium_bucket = buckets[1] if len(buckets) > 1 else sorted_idx
        easy_bucket = buckets[2] if len(buckets) > 2 else sorted_idx

        hard[row] = sample_with_padding(hard_bucket, count_per_bucket, mode="front")
        medium[row] = sample_with_padding(medium_bucket, count_per_bucket, mode="spread")
        easy[row] = sample_with_padding(easy_bucket, count_per_bucket, mode="back")

    return easy, medium, hard


def select_hard_and_random_indices(distance_matrix: np.ndarray, num_hard: int, num_random: int, exclusion_eps: float, seed: int):
    batch_size, num_bank = distance_matrix.shape
    hard = np.empty((batch_size, num_hard), dtype=np.int64)
    random_idx = np.empty((batch_size, num_random), dtype=np.int64)

    for row in range(batch_size):
        distances = distance_matrix[row]
        valid = np.flatnonzero(distances > exclusion_eps)
        if valid.size == 0:
            valid = np.arange(num_bank, dtype=np.int64)

        ordered = valid[np.argsort(distances[valid])]
        hard[row] = sample_with_padding(ordered, num_hard, mode="front")

        remaining = np.setdiff1d(valid, hard[row], assume_unique=False)
        if remaining.size == 0:
            remaining = valid
        rng = np.random.default_rng(seed + row)
        if remaining.size >= num_random:
            sampled = rng.choice(remaining, size=num_random, replace=False)
        else:
            sampled = rng.choice(remaining, size=num_random, replace=True)
        random_idx[row] = np.asarray(sampled, dtype=np.int64)

    return hard, random_idx


def predict_one_step_candidates(model, emb_history: torch.Tensor, fixed_prefix_actions: torch.Tensor, candidate_last_actions: torch.Tensor):
    batch_size, num_candidates, _, action_dim = candidate_last_actions.shape
    history_size = emb_history.size(1)
    flat_emb_history = emb_history.unsqueeze(1).expand(-1, num_candidates, -1, -1).reshape(
        batch_size * num_candidates, history_size, emb_history.size(-1)
    )
    flat_prefix = fixed_prefix_actions.unsqueeze(1).expand(-1, num_candidates, -1, -1).reshape(
        batch_size * num_candidates, history_size - 1, action_dim
    )
    flat_last = candidate_last_actions.reshape(batch_size * num_candidates, 1, action_dim)
    act_history = torch.cat([flat_prefix, flat_last], dim=1)
    with torch.inference_mode():
        pred_next = model.predict(flat_emb_history, model.action_encoder(act_history))[:, -1]
    return pred_next.reshape(batch_size, num_candidates, -1)


def rollout_counterfactual_suffix(
    model,
    emb_history: torch.Tensor,
    fixed_prefix_actions: torch.Tensor,
    candidate_suffix_actions: torch.Tensor,
):
    batch_size, num_candidates, horizon, action_dim = candidate_suffix_actions.shape
    history_size = emb_history.size(1)
    flat_emb_history = emb_history.unsqueeze(1).expand(-1, num_candidates, -1, -1).reshape(
        batch_size * num_candidates, history_size, emb_history.size(-1)
    )
    flat_prefix = fixed_prefix_actions.unsqueeze(1).expand(-1, num_candidates, -1, -1).reshape(
        batch_size * num_candidates, history_size - 1, action_dim
    )
    flat_suffix = candidate_suffix_actions.reshape(batch_size * num_candidates, horizon, action_dim)

    emb_hist = flat_emb_history.clone()
    act_prefix = flat_prefix.clone()
    predictions = []
    for step in range(horizon):
        current_action = flat_suffix[:, step : step + 1]
        act_hist = torch.cat([act_prefix, current_action], dim=1)[:, -history_size:]
        with torch.inference_mode():
            pred_next = model.predict(emb_hist[:, -history_size:], model.action_encoder(act_hist))[:, -1:]
        predictions.append(pred_next)
        emb_hist = torch.cat([emb_hist, pred_next], dim=1)
        act_prefix = torch.cat([act_prefix, current_action], dim=1)

    pred_future = torch.cat(predictions, dim=1)
    return pred_future.reshape(batch_size, num_candidates, horizon, -1)


def pairwise_action_distance(flat_candidates: np.ndarray):
    diff = flat_candidates[:, None, :] - flat_candidates[None, :, :]
    return np.sqrt(np.maximum((diff ** 2).sum(axis=-1), 0.0))


def pairwise_cost_difference(costs: np.ndarray):
    return np.abs(costs[:, None] - costs[None, :])


def compute_neighbor_smoothness(flat_candidates: np.ndarray, model_costs: np.ndarray):
    action_dist = pairwise_action_distance(flat_candidates)
    cost_diff = pairwise_cost_difference(model_costs)
    triu = np.triu_indices_from(action_dist, k=1)
    return spearman_corr(action_dist[triu], cost_diff[triu])


def compute_topk_overlap(model_costs: np.ndarray, env_costs: np.ndarray, topk: int):
    topk = min(int(topk), model_costs.size, env_costs.size)
    if topk <= 0:
        return float("nan")
    model_best = set(np.argsort(model_costs)[:topk].tolist())
    env_best = set(np.argsort(env_costs)[:topk].tolist())
    return float(len(model_best.intersection(env_best)) / topk)


def topk_good_hit(model_costs: np.ndarray, env_costs: np.ndarray, topk: int, good_quantile: float):
    topk = min(int(topk), model_costs.size, env_costs.size)
    if topk <= 0:
        return float("nan")
    threshold = float(np.quantile(env_costs, np.clip(float(good_quantile), 0.0, 1.0)))
    model_topk = np.argsort(model_costs)[:topk]
    return float(np.any(env_costs[model_topk] <= threshold))


def topk_good_precision(model_costs: np.ndarray, env_costs: np.ndarray, topk: int, good_quantile: float):
    topk = min(int(topk), model_costs.size, env_costs.size)
    if topk <= 0:
        return float("nan")
    threshold = float(np.quantile(env_costs, np.clip(float(good_quantile), 0.0, 1.0)))
    model_topk = np.argsort(model_costs)[:topk]
    return float(np.mean(env_costs[model_topk] <= threshold))


def pairwise_preference_metrics(
    model_costs: np.ndarray,
    env_costs: np.ndarray,
    *,
    gap_quantile: float,
    min_gap: float,
    eps: float,
):
    model_costs = np.asarray(model_costs, dtype=np.float64).reshape(-1)
    env_costs = np.asarray(env_costs, dtype=np.float64).reshape(-1)
    tri = np.triu_indices(model_costs.size, k=1)
    env_diff = env_costs[tri[0]] - env_costs[tri[1]]
    model_diff = model_costs[tri[0]] - model_costs[tri[1]]
    env_gap = np.abs(env_diff)
    finite = np.isfinite(env_gap) & np.isfinite(model_diff)
    if not np.any(finite):
        return {
            "pairwise_preference_acc": float("nan"),
            "pairwise_preference_weighted_acc": float("nan"),
            "pairwise_preference_pair_ratio": 0.0,
            "pairwise_preference_gap_threshold": float("nan"),
        }

    finite_gaps = env_gap[finite]
    positive_gaps = finite_gaps[finite_gaps > eps]
    if positive_gaps.size == 0:
        return {
            "pairwise_preference_acc": float("nan"),
            "pairwise_preference_weighted_acc": float("nan"),
            "pairwise_preference_pair_ratio": 0.0,
            "pairwise_preference_gap_threshold": float("nan"),
        }

    threshold = max(float(min_gap), float(np.quantile(positive_gaps, np.clip(float(gap_quantile), 0.0, 1.0))))
    mask = finite & (env_gap >= threshold)
    if not np.any(mask):
        return {
            "pairwise_preference_acc": float("nan"),
            "pairwise_preference_weighted_acc": float("nan"),
            "pairwise_preference_pair_ratio": 0.0,
            "pairwise_preference_gap_threshold": threshold,
        }

    correct = np.sign(model_diff[mask]) == np.sign(env_diff[mask])
    weights = env_gap[mask]
    weighted = float(np.sum(correct.astype(np.float64) * weights) / max(np.sum(weights), eps))
    return {
        "pairwise_preference_acc": float(correct.mean()),
        "pairwise_preference_weighted_acc": weighted,
        "pairwise_preference_pair_ratio": float(mask.mean()),
        "pairwise_preference_gap_threshold": threshold,
    }


def shared_planner_value_metrics(model_costs: np.ndarray, env_costs: np.ndarray, topk: int, cfg: DictConfig):
    model_costs = np.asarray(model_costs, dtype=np.float64).reshape(-1)
    env_costs = np.asarray(env_costs, dtype=np.float64).reshape(-1)
    topk = min(int(topk), model_costs.size, env_costs.size)
    model_order = np.argsort(model_costs)
    env_order = np.argsort(env_costs)
    model_best = int(model_order[0])
    env_best = int(env_order[0])
    model_topk = model_order[:topk]

    env_best_cost = float(env_costs[env_best])
    env_worst_cost = float(np.max(env_costs))
    denom = max(env_worst_cost - env_best_cost, float(cfg.analysis.eps))
    model_best_env_cost = float(env_costs[model_best])
    model_topk_env_min = float(np.min(env_costs[model_topk])) if topk > 0 else float("nan")
    model_topk_env_mean = float(np.mean(env_costs[model_topk])) if topk > 0 else float("nan")
    regret = model_best_env_cost - env_best_cost
    topk_regret = model_topk_env_min - env_best_cost
    good_quantile = float(getattr(cfg.analysis.shared_planner, "good_quantile", 0.2))
    pairwise = pairwise_preference_metrics(
        model_costs,
        env_costs,
        gap_quantile=float(getattr(cfg.analysis.shared_planner, "pairwise_gap_quantile", 0.5)),
        min_gap=float(getattr(cfg.analysis.shared_planner, "pairwise_min_env_gap", 0.0)),
        eps=float(cfg.analysis.eps),
    )
    return {
        "model_best_env_cost": model_best_env_cost,
        "env_best_cost": env_best_cost,
        "model_topk_env_min": model_topk_env_min,
        "model_topk_env_mean": model_topk_env_mean,
        "model_best_env_regret": float(regret),
        "model_topk_env_regret": float(topk_regret),
        "model_best_env_regret_norm": float(regret / denom),
        "model_topk_env_regret_norm": float(topk_regret / denom),
        "env_best_model_rank": rank_of_index(model_costs, env_best),
        "env_best_model_rank_norm": rank_of_index(model_costs, env_best) / max(model_costs.size - 1, 1),
        "model_topk_good_hit": topk_good_hit(model_costs, env_costs, topk=topk, good_quantile=good_quantile),
        "model_top1_good_hit": topk_good_hit(model_costs, env_costs, topk=1, good_quantile=good_quantile),
        "model_topk_good_precision": topk_good_precision(model_costs, env_costs, topk=topk, good_quantile=good_quantile),
        **pairwise,
    }


def sample_local_plan_candidates(center: np.ndarray, scale: np.ndarray, low: np.ndarray, high: np.ndarray, num_candidates: int, seed: int, noise_scale: float):
    rng = np.random.default_rng(seed)
    flat_dim = center.size
    candidates = np.empty((num_candidates, flat_dim), dtype=np.float32)
    candidates[0] = center.astype(np.float32)
    if num_candidates > 1:
        noise = rng.normal(size=(num_candidates - 1, flat_dim)).astype(np.float32)
        candidates[1:] = center[None, :] + noise_scale * noise * scale[None, :]
    return clip_flat_candidates(candidates, low=low, high=high).astype(np.float32)


def fit_response_curve_values(action_distance, prediction_shift, eps: float):
    action_np = np.asarray(action_distance, dtype=np.float64)
    shift_np = np.asarray(prediction_shift, dtype=np.float64)
    slopes = []
    r2_values = []

    for x, y in zip(action_np, shift_np):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            slopes.append(np.nan)
            r2_values.append(np.nan)
            continue
        x = x[mask]
        y = y[mask]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        x_var = float((x_centered ** 2).sum())
        y_var = float((y_centered ** 2).sum())
        if x_var <= eps:
            slopes.append(np.nan)
            r2_values.append(np.nan)
            continue
        slope = float((x_centered * y_centered).sum() / x_var)
        intercept = float(y.mean() - slope * x.mean())
        residual = y - (slope * x + intercept)
        slopes.append(slope)
        if y_var <= eps:
            r2_values.append(np.nan)
        else:
            r2_values.append(float(1.0 - (residual ** 2).sum() / y_var))

    return np.asarray(slopes, dtype=np.float32), np.asarray(r2_values, dtype=np.float32)


def response_curve_metrics(prefix: str, action_distance, prediction_shift, eps: float):
    slopes, r2_values = fit_response_curve_values(action_distance, prediction_shift, eps=eps)
    return {
        f"{prefix}_response_slope_values": np.asarray(slopes, dtype=np.float32),
        f"{prefix}_response_r2_values": np.asarray(r2_values, dtype=np.float32),
    }


def response_curve_at_h(action_distance, prediction_shift_at_h, eps: float):
    action_np = np.asarray(action_distance, dtype=np.float64)
    shift_np = np.asarray(prediction_shift_at_h, dtype=np.float64)
    horizon = shift_np.shape[-1]
    slope_at_h = []
    r2_at_h = []
    for horizon_idx in range(horizon):
        slopes, r2_values = fit_response_curve_values(action_np, shift_np[..., horizon_idx], eps=eps)
        slope_at_h.append(slopes)
        r2_at_h.append(r2_values)
    return np.stack(slope_at_h, axis=1).astype(np.float32), np.stack(r2_at_h, axis=1).astype(np.float32)


def sample_rollout_perturbations(factual_suffix: torch.Tensor, action_bank_suffix: torch.Tensor, cfg: DictConfig, sample_offset: int):
    rollout_cfg = cfg.analysis.rollout
    device = factual_suffix.device
    batch_size, horizon, action_dim = factual_suffix.shape
    bank_device = action_bank_suffix.to(device)
    action_scale = bank_device.reshape(-1, action_dim).std(dim=0).clamp_min(float(cfg.analysis.eps))
    action_low = bank_device.reshape(-1, action_dim).amin(dim=0)
    action_high = bank_device.reshape(-1, action_dim).amax(dim=0)

    candidates = []
    labels = []
    for label, count_key, scale_key in [
        ("near", "near_count", "near_scale"),
        ("mid", "mid_count", "mid_scale"),
        ("far", "far_count", "far_scale"),
    ]:
        count = int(getattr(rollout_cfg, count_key))
        if count <= 0:
            continue
        per_row = []
        for row in range(batch_size):
            gen = torch.Generator(device=device).manual_seed(int(cfg.seed) + 30_003 + sample_offset + row * 101 + len(labels))
            noise = torch.randn(
                count,
                horizon,
                action_dim,
                generator=gen,
                device=device,
                dtype=factual_suffix.dtype,
            )
            suffix = factual_suffix[row : row + 1] + float(getattr(rollout_cfg, scale_key)) * noise * action_scale.view(1, 1, -1)
            if bool(getattr(rollout_cfg, "clip_to_bank_range", True)):
                suffix = suffix.clamp(action_low.view(1, 1, -1), action_high.view(1, 1, -1))
            per_row.append(suffix)
        candidates.append(torch.stack(per_row, dim=0))
        labels.extend([label] * count)

    main_suffix = torch.cat(candidates, dim=1) if candidates else factual_suffix.new_empty(batch_size, 0, horizon, action_dim)

    random_count = int(getattr(rollout_cfg, "dataset_random_count", 0))
    random_suffix = factual_suffix.new_empty(batch_size, 0, horizon, action_dim)
    if random_count > 0:
        factual_flat = factual_suffix.reshape(batch_size, -1)
        bank_flat = bank_device.reshape(bank_device.size(0), -1)
        dist = torch.cdist(factual_flat, bank_flat)
        _, random_idx = select_hard_and_random_indices(
            dist.detach().cpu().numpy(),
            num_hard=0,
            num_random=random_count,
            exclusion_eps=float(rollout_cfg.exclusion_eps),
            seed=int(cfg.seed) + sample_offset,
        )
        random_suffix = bank_device[random_idx]

    return main_suffix, np.asarray(labels), random_suffix


def rank_of_index(costs: np.ndarray, index: int):
    costs = np.asarray(costs, dtype=np.float64)
    if index < 0 or index >= costs.size:
        return float("nan")
    order = np.argsort(costs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(costs.size)
    return float(ranks[index])


def extract_expert_action_blocks(dataset, episode_idx: int, start_step: int, horizon: int, action_block: int, action_dim: int, goal_offset_steps: int):
    min_steps = horizon * action_block
    end_step = int(start_step) + max(int(goal_offset_steps), min_steps)
    data = dataset.load_chunk(
        np.array([int(episode_idx)]),
        np.array([int(start_step)]),
        np.array([end_step]),
    )[0]
    actions = data["action"]
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions, dtype=np.float32)
    actions = actions.reshape(actions.shape[0], -1)
    if actions.shape[0] < min_steps:
        if actions.shape[0] == 0:
            actions = np.zeros((min_steps, action_dim), dtype=np.float32)
        else:
            pad = np.repeat(actions[-1:], min_steps - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)
    actions = actions[:min_steps, :action_dim]
    return actions.reshape(horizon, action_block, action_dim).astype(np.float32)


def build_shared_action_bank(dataset, eval_cfg: DictConfig, rows: tuple[np.ndarray, np.ndarray], horizon: int, action_block: int, action_dim: int):
    episodes, start_idx = rows
    bank = []
    for episode_idx, start_step in zip(episodes, start_idx):
        bank.append(
            extract_expert_action_blocks(
                dataset,
                episode_idx=int(episode_idx),
                start_step=int(start_step),
                horizon=horizon,
                action_block=action_block,
                action_dim=action_dim,
                goal_offset_steps=int(eval_cfg.eval.goal_offset_steps),
            )
        )
    if not bank:
        return np.empty((0, horizon * action_block * action_dim), dtype=np.float32)
    return np.stack(bank).reshape(len(bank), -1).astype(np.float32)


def shared_planner_env_cost_mode(cfg: DictConfig):
    mode = str(getattr(cfg.analysis.shared_planner, "env_cost_mode", "visual_latent")).lower()
    if mode == "bisual_latent":
        mode = "visual_latent"
    if mode not in {"state_mse", "visual_latent"}:
        raise ValueError(f"Unsupported analysis.shared_planner.env_cost_mode={mode!r}")
    return mode


def sample_shared_planner_candidates(
    expert_flat: np.ndarray,
    shared_bank: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    cfg: DictConfig,
    seed: int,
    plan_shape: tuple[int, int],
):
    shared_cfg = cfg.analysis.shared_planner
    rng = np.random.default_rng(seed)
    scale = np.maximum(high - low, float(cfg.analysis.eps))
    flat_dim = expert_flat.size
    pieces = [expert_flat[None, :].astype(np.float32)]
    labels = ["expert"]

    if bool(getattr(shared_cfg, "include_zero", True)):
        pieces.append(np.zeros((1, flat_dim), dtype=np.float32))
        labels.append("zero")
    if bool(getattr(shared_cfg, "include_negated", True)):
        pieces.append((-expert_flat[None, :]).astype(np.float32))
        labels.append("negated_expert")
    if bool(getattr(shared_cfg, "include_reversed", True)):
        if len(plan_shape) == 2 and int(np.prod(plan_shape)) == flat_dim:
            reversed_plan = expert_flat.reshape(plan_shape)[::-1].reshape(1, flat_dim)
            pieces.append(reversed_plan.astype(np.float32))
            labels.append("reversed_expert")

    for name, count_key, scale_key in [
        ("near", "near_count", "near_scale"),
        ("mid", "mid_count", "mid_scale"),
        ("far", "far_count", "far_scale"),
    ]:
        count = int(getattr(shared_cfg, count_key))
        if count <= 0:
            continue
        noise = rng.normal(size=(count, flat_dim)).astype(np.float32)
        candidates = expert_flat[None, :] + float(getattr(shared_cfg, scale_key)) * noise * scale[None, :]
        pieces.append(candidates.astype(np.float32))
        labels.extend([name] * count)

    random_count = int(shared_cfg.random_count)
    if random_count > 0:
        if shared_bank.size > 0:
            replace = shared_bank.shape[0] < random_count
            random_idx = rng.choice(shared_bank.shape[0], size=random_count, replace=replace)
            random_candidates = shared_bank[random_idx]
        else:
            random_candidates = rng.uniform(low, high, size=(random_count, flat_dim)).astype(np.float32)
        pieces.append(random_candidates.astype(np.float32))
        labels.extend(["dataset_random"] * random_count)

    flat_candidates = np.concatenate(pieces, axis=0)
    return clip_flat_candidates(flat_candidates, low=low, high=high).astype(np.float32), np.asarray(labels)


def hash_array(array: np.ndarray):
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def shared_planner_cache_path(cfg: DictConfig):
    explicit = getattr(cfg.analysis.shared_planner, "cache_path", None)
    if explicit:
        return Path(explicit)
    shared_cfg = cfg.analysis.shared_planner
    name = (
        f"{cfg.eval_config_name}_seed{int(cfg.seed)}"
        f"_n{int(shared_cfg.num_eval)}"
        f"_near{int(shared_cfg.near_count)}x{float(shared_cfg.near_scale):g}"
        f"_mid{int(shared_cfg.mid_count)}x{float(shared_cfg.mid_scale):g}"
        f"_far{int(shared_cfg.far_count)}x{float(shared_cfg.far_scale):g}"
        f"_rand{int(shared_cfg.random_count)}"
        f"_zero{int(bool(getattr(shared_cfg, 'include_zero', True)))}"
        f"_neg{int(bool(getattr(shared_cfg, 'include_negated', True)))}"
        f"_rev{int(bool(getattr(shared_cfg, 'include_reversed', True)))}"
        f"_{shared_planner_env_cost_mode(cfg)}"
        "_v4.npz"
    )
    return get_cache_dir(cfg) / "counterfactual_shared_planner_cache" / name


def build_or_load_shared_planner_cache(
    dataset,
    eval_cfg: DictConfig,
    cfg: DictConfig,
    eval_episodes: np.ndarray,
    eval_start_idx: np.ndarray,
    shared_action_bank: np.ndarray,
):
    cache_path = shared_planner_cache_path(cfg)
    if cache_path.exists() and not bool(getattr(cfg.analysis.shared_planner, "overwrite_cache", False)):
        cached = np.load(cache_path, allow_pickle=True)
        return {key: cached[key] for key in cached.files}

    cost_mode = shared_planner_env_cost_mode(cfg)
    env_world = build_world_from_eval_config(eval_cfg, num_envs=int(cfg.analysis.shared_planner.env_batch_size))
    candidate_records = []
    label_records = []
    env_cost_records = []

    try:
        action_block = int(eval_cfg.plan_config.action_block)
        horizon = int(eval_cfg.plan_config.horizon)
        action_dim = int(np.asarray(env_world.single_action_space.low).reshape(-1).shape[0])
        low, high = get_flat_action_bounds(env_world.single_action_space, horizon=horizon, action_block=action_block)

        for case_idx, (episode_idx, start_step) in enumerate(zip(eval_episodes, eval_start_idx), start=1):
            case = build_eval_case(
                dataset,
                episode_idx=int(episode_idx),
                start_step=int(start_step),
                goal_offset_steps=eval_cfg.eval.goal_offset_steps,
            )
            expert_blocks = extract_expert_action_blocks(
                dataset,
                episode_idx=int(episode_idx),
                start_step=int(start_step),
                horizon=horizon,
                action_block=action_block,
                action_dim=action_dim,
                goal_offset_steps=int(eval_cfg.eval.goal_offset_steps),
            )
            flat_candidates, labels = sample_shared_planner_candidates(
                expert_flat=expert_blocks.reshape(-1).astype(np.float32),
                shared_bank=shared_action_bank,
                low=low,
                high=high,
                cfg=cfg,
                seed=int(cfg.seed) + 100_003 + case_idx,
                plan_shape=(horizon, action_block * action_dim),
            )
            action_blocks = unflatten_action_candidates(
                flat_candidates,
                horizon=horizon,
                action_block=action_block,
                action_dim=action_dim,
            )
            if cost_mode == "state_mse":
                env_costs = evaluate_env_costs(
                    env_world,
                    case,
                    eval_cfg.eval.callables,
                    action_blocks,
                    eval_cfg=eval_cfg,
                )
                env_cost_records.append(env_costs.astype(np.float32))
            candidate_records.append(flat_candidates.astype(np.float32))
            label_records.append(labels.astype(str))

    finally:
        try:
            env_world.close()
        except Exception:
            pass
        del env_world
        gc.collect()

    flat_candidates = np.stack(candidate_records).astype(np.float32)
    payload = {
        "shared_planner_eval_episodes": np.asarray(eval_episodes, dtype=np.int64),
        "shared_planner_eval_start_idx": np.asarray(eval_start_idx, dtype=np.int64),
        "shared_planner_flat_candidates": flat_candidates,
        "shared_planner_candidate_labels": np.stack(label_records),
        "shared_planner_env_cost_mode": np.asarray([cost_mode]),
        "shared_planner_candidate_hash": np.asarray([hash_array(flat_candidates)]),
    }
    if env_cost_records:
        env_costs = np.stack(env_cost_records).astype(np.float32)
        payload["shared_planner_env_costs"] = env_costs
        payload["shared_planner_env_cost_hash"] = np.asarray([hash_array(env_costs)])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **payload)
    return payload


def bucket_response_metrics(prefix: str, action_distance: torch.Tensor, prediction_shift: torch.Tensor, bucket_size: int, eps: float):
    easy_dist = action_distance[:, :bucket_size]
    medium_dist = action_distance[:, bucket_size : 2 * bucket_size]
    hard_dist = action_distance[:, 2 * bucket_size :]
    easy_shift = prediction_shift[:, :bucket_size]
    medium_shift = prediction_shift[:, bucket_size : 2 * bucket_size]
    hard_shift = prediction_shift[:, 2 * bucket_size :]

    easy_shift_mean = easy_shift.mean(dim=1)
    medium_shift_mean = medium_shift.mean(dim=1)
    hard_shift_mean = hard_shift.mean(dim=1)
    easy_gain = (easy_shift / easy_dist.clamp_min(eps)).mean(dim=1)
    medium_gain = (medium_shift / medium_dist.clamp_min(eps)).mean(dim=1)
    hard_gain = (hard_shift / hard_dist.clamp_min(eps)).mean(dim=1)
    monotonicity = ((hard_shift_mean <= medium_shift_mean) & (medium_shift_mean <= easy_shift_mean)).float()

    action_np = action_distance.detach().cpu().numpy()
    shift_np = prediction_shift.detach().cpu().numpy()
    rank_corr = np.asarray(
        [safe_spearman(action_np[row], shift_np[row], eps=eps) for row in range(action_np.shape[0])],
        dtype=np.float32,
    )

    metrics = {
        f"{prefix}_pred_shift_easy_values": easy_shift_mean.detach().cpu().numpy(),
        f"{prefix}_pred_shift_medium_values": medium_shift_mean.detach().cpu().numpy(),
        f"{prefix}_pred_shift_hard_values": hard_shift_mean.detach().cpu().numpy(),
        f"{prefix}_pred_shift_far_values": easy_shift_mean.detach().cpu().numpy(),
        f"{prefix}_pred_shift_mid_values": medium_shift_mean.detach().cpu().numpy(),
        f"{prefix}_pred_shift_near_values": hard_shift_mean.detach().cpu().numpy(),
        f"{prefix}_response_gain_easy_values": easy_gain.detach().cpu().numpy(),
        f"{prefix}_response_gain_medium_values": medium_gain.detach().cpu().numpy(),
        f"{prefix}_response_gain_hard_values": hard_gain.detach().cpu().numpy(),
        f"{prefix}_response_gain_far_values": easy_gain.detach().cpu().numpy(),
        f"{prefix}_response_gain_mid_values": medium_gain.detach().cpu().numpy(),
        f"{prefix}_response_gain_near_values": hard_gain.detach().cpu().numpy(),
        f"{prefix}_pred_shift_monotonicity_values": monotonicity.detach().cpu().numpy(),
        f"{prefix}_response_rank_corr_values": rank_corr,
    }
    metrics.update(response_curve_metrics(prefix, action_np, shift_np, eps=eps))
    return metrics


def build_plot(output_path: Path, labels: list[str], means: list[float], stds: list[float], title: str):
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=stds, color=["#c8d5b9", "#8fc0a9", "#3b6064"], capsize=6)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Best Negative Error - Factual Error")
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def upsert_summary_record(summary_path: Path, record: dict):
    existing = []
    if summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("checkpoint") == record.get("checkpoint") and payload.get("epoch") == record.get("epoch"):
                continue
            existing.append(payload)
    existing.append(record)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        for item in existing:
            f.write(json.dumps(item, sort_keys=True) + "\n")


def output_path_for_checkpoint(run_dir: Path, output_subdir: str, checkpoint_path: Path):
    epoch = parse_epoch(checkpoint_path)
    return run_dir / output_subdir / f"{checkpoint_path.stem}_counterfactual_metrics_epoch_{epoch:03d}.npz"


def run_one_step_counterfactual(
    model,
    outputs,
    batch,
    action_bank_last: torch.Tensor,
    cfg: DictConfig,
):
    count_per_bucket = int(cfg.analysis.one_step.num_negatives_per_bucket)
    target_next = outputs["emb"][:, cfg.wm.history_size]
    factual_action = batch["action"][:, cfg.wm.history_size - 1]
    emb_history = outputs["emb"][:, : cfg.wm.history_size]
    fixed_prefix = batch["action"][:, : cfg.wm.history_size - 1]

    dist = torch.cdist(factual_action, action_bank_last.to(factual_action.device))
    easy_idx, medium_idx, hard_idx = select_easy_medium_hard_indices(
        dist.detach().cpu().numpy(),
        count_per_bucket=count_per_bucket,
        exclusion_eps=float(cfg.analysis.one_step.exclusion_eps),
    )

    bank_device = action_bank_last.to(factual_action.device)
    easy_actions = bank_device[easy_idx]
    medium_actions = bank_device[medium_idx]
    hard_actions = bank_device[hard_idx]
    negative_actions = torch.cat([easy_actions, medium_actions, hard_actions], dim=1)

    candidate_last = torch.cat([factual_action.unsqueeze(1), negative_actions], dim=1).unsqueeze(2)
    pred_next = predict_one_step_candidates(model, emb_history, fixed_prefix, candidate_last)
    target = target_next.unsqueeze(1)
    error = (pred_next - target).pow(2).mean(dim=-1)
    factual_error = error[:, 0]
    easy_error = error[:, 1 : 1 + count_per_bucket]
    medium_error = error[:, 1 + count_per_bucket : 1 + 2 * count_per_bucket]
    hard_error = error[:, 1 + 2 * count_per_bucket :]
    best_negative_error = error[:, 1:].min(dim=1).values

    factual_pred = pred_next[:, :1]
    negative_pred = pred_next[:, 1:]
    pred_shift = (negative_pred - factual_pred).norm(dim=-1)
    action_distance = (negative_actions - factual_action.unsqueeze(1)).norm(dim=-1)

    if cfg.analysis.one_step.include_identity_candidate:
        identity_last = factual_action.unsqueeze(1).unsqueeze(2)
        identity_pred = predict_one_step_candidates(model, emb_history, fixed_prefix, identity_last)
        identity_error = (identity_pred.squeeze(1) - target_next).pow(2).mean(dim=-1)
        identity_pred_gap = (identity_pred.squeeze(1) - pred_next[:, 0]).norm(dim=-1)
    else:
        identity_error = torch.full_like(factual_error, float("nan"))
        identity_pred_gap = torch.full_like(factual_error, float("nan"))

    batch_metrics = {
        "one_step_cf_factual_top1_values": (error.argmin(dim=1) == 0).float().cpu().numpy(),
        "one_step_cf_factual_margin_values": (best_negative_error - factual_error).cpu().numpy(),
        "one_step_cf_easy_margin_values": (easy_error.min(dim=1).values - factual_error).cpu().numpy(),
        "one_step_cf_medium_margin_values": (medium_error.min(dim=1).values - factual_error).cpu().numpy(),
        "one_step_cf_hard_margin_values": (hard_error.min(dim=1).values - factual_error).cpu().numpy(),
        "one_step_cf_hard_negative_top1_values": (factual_error < hard_error.min(dim=1).values).float().cpu().numpy(),
        "one_step_cf_identity_error_gap_values": (identity_error - factual_error).cpu().numpy(),
        "one_step_cf_identity_prediction_gap_values": identity_pred_gap.cpu().numpy(),
        "one_step_cf_action_distance_values": action_distance.cpu().numpy(),
        "one_step_cf_prediction_shift_values": pred_shift.cpu().numpy(),
        "one_step_cf_factual_error_values": factual_error.cpu().numpy(),
        "one_step_cf_best_negative_error_values": best_negative_error.cpu().numpy(),
    }
    batch_metrics.update(
        bucket_response_metrics(
            "one_step_cf",
            action_distance=action_distance,
            prediction_shift=pred_shift,
            bucket_size=count_per_bucket,
            eps=float(cfg.analysis.eps),
        )
    )
    batch_metrics["one_step_cf_action_distance_vs_prediction_shift_corr_values"] = np.asarray(
        [safe_corr(action_distance[row].cpu().numpy(), pred_shift[row].cpu().numpy(), eps=float(cfg.analysis.eps))
         for row in range(action_distance.size(0))],
        dtype=np.float32,
    )
    return batch_metrics


def run_rollout_counterfactual(
    model,
    outputs,
    batch,
    action_bank_suffix: torch.Tensor,
    cfg: DictConfig,
    sample_offset: int,
):
    horizon = int(cfg.analysis.rollout.horizon)
    target_future = outputs["emb"][:, cfg.wm.history_size : cfg.wm.history_size + horizon]
    emb_history = outputs["emb"][:, : cfg.wm.history_size]
    fixed_prefix = batch["action"][:, : cfg.wm.history_size - 1]
    factual_suffix = batch["action"][:, cfg.wm.history_size - 1 : cfg.wm.history_size - 1 + horizon]

    main_suffix, main_labels, random_suffix = sample_rollout_perturbations(
        factual_suffix=factual_suffix,
        action_bank_suffix=action_bank_suffix,
        cfg=cfg,
        sample_offset=sample_offset,
    )
    negative_suffix = torch.cat([main_suffix, random_suffix], dim=1)
    candidate_suffix = torch.cat([factual_suffix.unsqueeze(1), negative_suffix], dim=1)
    pred_future = rollout_counterfactual_suffix(model, emb_history, fixed_prefix, candidate_suffix)

    target = target_future.unsqueeze(1)
    per_h_error = (pred_future - target).pow(2).mean(dim=-1)
    aggregate_error = per_h_error.mean(dim=-1)
    factual_error = aggregate_error[:, 0]
    best_negative_error = aggregate_error[:, 1:].min(dim=1).values

    factual_pred = pred_future[:, :1]
    negative_pred = pred_future[:, 1:]
    main_count = main_suffix.size(1)
    main_pred = pred_future[:, 1 : 1 + main_count]
    random_pred = pred_future[:, 1 + main_count :]
    main_per_candidate_separation = (main_pred - factual_pred).norm(dim=-1)
    random_per_candidate_separation = (random_pred - factual_pred).norm(dim=-1) if random_suffix.size(1) > 0 else None
    separation = main_per_candidate_separation.mean(dim=1)
    aggregate_prediction_shift = main_per_candidate_separation.mean(dim=-1)
    candidate_action_distance = (main_suffix - factual_suffix.unsqueeze(1)).reshape(
        main_suffix.size(0),
        main_suffix.size(1),
        -1,
    ).norm(dim=-1)

    near_count = int(cfg.analysis.rollout.near_count)
    mid_count = int(cfg.analysis.rollout.mid_count)
    far_count = int(cfg.analysis.rollout.far_count)
    near_slice = slice(0, near_count)
    mid_slice = slice(near_count, near_count + mid_count)
    far_slice = slice(near_count + mid_count, near_count + mid_count + far_count)
    near_shift = aggregate_prediction_shift[:, near_slice]
    mid_shift = aggregate_prediction_shift[:, mid_slice]
    far_shift = aggregate_prediction_shift[:, far_slice]
    near_distance = candidate_action_distance[:, near_slice]
    mid_distance = candidate_action_distance[:, mid_slice]
    far_distance = candidate_action_distance[:, far_slice]
    near_shift_mean = near_shift.mean(dim=1)
    mid_shift_mean = mid_shift.mean(dim=1)
    far_shift_mean = far_shift.mean(dim=1)
    near_distance_mean = near_distance.mean(dim=1)
    mid_distance_mean = mid_distance.mean(dim=1)
    far_distance_mean = far_distance.mean(dim=1)
    eps = float(cfg.analysis.eps)
    near_gain = (near_shift / near_distance.clamp_min(eps)).mean(dim=1)
    mid_gain = (mid_shift / mid_distance.clamp_min(eps)).mean(dim=1)
    far_gain = (far_shift / far_distance.clamp_min(eps)).mean(dim=1)
    shift_monotonicity = ((near_shift_mean <= mid_shift_mean) & (mid_shift_mean <= far_shift_mean)).float()
    action_np = candidate_action_distance.detach().cpu().numpy()
    shift_np = aggregate_prediction_shift.detach().cpu().numpy()
    shift_rank_corr = np.asarray(
        [safe_spearman(action_np[row], shift_np[row], eps=eps) for row in range(action_np.shape[0])],
        dtype=np.float32,
    )
    slope_at_h, r2_at_h = response_curve_at_h(
        action_np,
        main_per_candidate_separation.detach().cpu().numpy(),
        eps=eps,
    )
    response_auc = torch.trapz(separation, dim=1) / max(separation.size(1) - 1, 1)

    pre_invariance = torch.zeros(pred_future.size(0), device=pred_future.device)
    if emb_history.size(1) > 0:
        candidate_hist = emb_history.unsqueeze(1).expand(-1, candidate_suffix.size(1), -1, -1)
        pre_invariance = (candidate_hist - candidate_hist[:, :1]).abs().amax(dim=(-1, -2, -3))

    margin_at_h = per_h_error[:, 1:].min(dim=1).values - per_h_error[:, 0]
    batch_metrics = {
        "rollout_cf_suffix_factual_top1_values": (aggregate_error.argmin(dim=1) == 0).float().cpu().numpy(),
        "rollout_cf_suffix_margin_values": (best_negative_error - factual_error).cpu().numpy(),
        "rollout_cf_suffix_margin_at_h_values": margin_at_h.cpu().numpy(),
        "rollout_cf_cf_separation_curve_values": separation.cpu().numpy(),
        "rollout_cf_cf_separation_growth_values": (separation[:, -1] - separation[:, 0]).cpu().numpy(),
        "rollout_cf_prediction_shift_near_values": near_shift_mean.cpu().numpy(),
        "rollout_cf_prediction_shift_mid_values": mid_shift_mean.cpu().numpy(),
        "rollout_cf_prediction_shift_far_values": far_shift_mean.cpu().numpy(),
        "rollout_cf_action_distance_near_values": near_distance_mean.cpu().numpy(),
        "rollout_cf_action_distance_mid_values": mid_distance_mean.cpu().numpy(),
        "rollout_cf_action_distance_far_values": far_distance_mean.cpu().numpy(),
        "rollout_cf_response_gain_near_values": near_gain.cpu().numpy(),
        "rollout_cf_response_gain_mid_values": mid_gain.cpu().numpy(),
        "rollout_cf_response_gain_far_values": far_gain.cpu().numpy(),
        "rollout_cf_prediction_shift_hard_values": near_shift_mean.cpu().numpy(),
        "rollout_cf_prediction_shift_random_values": far_shift_mean.cpu().numpy(),
        "rollout_cf_action_distance_hard_values": near_distance_mean.cpu().numpy(),
        "rollout_cf_action_distance_random_values": far_distance_mean.cpu().numpy(),
        "rollout_cf_response_gain_hard_values": near_gain.cpu().numpy(),
        "rollout_cf_response_gain_random_values": far_gain.cpu().numpy(),
        "rollout_cf_shift_monotonicity_values": shift_monotonicity.cpu().numpy(),
        "rollout_cf_shift_rank_corr_values": shift_rank_corr,
        "rollout_cf_response_slope_at_h_values": slope_at_h,
        "rollout_cf_response_r2_at_h_values": r2_at_h,
        "rollout_cf_response_auc_values": response_auc.cpu().numpy(),
        "rollout_cf_pre_intervention_invariance_values": pre_invariance.cpu().numpy(),
        "rollout_cf_factual_error_values": factual_error.cpu().numpy(),
        "rollout_cf_best_negative_error_values": best_negative_error.cpu().numpy(),
        "rollout_cf_candidate_labels": np.tile(main_labels[None, :], (factual_suffix.size(0), 1)),
        "rollout_cf_action_distance_values": action_np,
        "rollout_cf_prediction_shift_values": shift_np,
    }
    if random_per_candidate_separation is not None:
        random_shift = random_per_candidate_separation.mean(dim=-1)
        random_distance = (random_suffix - factual_suffix.unsqueeze(1)).reshape(
            random_suffix.size(0),
            random_suffix.size(1),
            -1,
        ).norm(dim=-1)
        batch_metrics.update(
            {
                "rollout_cf_dataset_random_prediction_shift_values": random_shift.mean(dim=1).cpu().numpy(),
                "rollout_cf_dataset_random_action_distance_values": random_distance.mean(dim=1).cpu().numpy(),
                "rollout_cf_dataset_random_response_gain_values": (random_shift / random_distance.clamp_min(eps)).mean(dim=1).cpu().numpy(),
            }
        )
    batch_metrics.update(response_curve_metrics("rollout_cf", action_np, shift_np, eps=eps))
    return batch_metrics


def run_planner_counterfactual(
    model,
    dataset,
    process,
    transform,
    eval_cfg: DictConfig,
    cfg: DictConfig,
    eval_episodes: np.ndarray,
    eval_start_idx: np.ndarray,
):
    cem_world = build_world_from_eval_config(eval_cfg, num_envs=1)
    env_world = build_world_from_eval_config(eval_cfg, num_envs=int(cfg.analysis.planner.env_batch_size))

    spearman_values = []
    overlap_values = []
    hit_values = []
    gap_values = []
    smoothness_values = []
    model_cost_records = []
    env_cost_records = []

    try:
        for case_idx, (episode_idx, start_step) in enumerate(zip(eval_episodes, eval_start_idx), start=1):
            case = build_eval_case(
                dataset,
                episode_idx=int(episode_idx),
                start_step=int(start_step),
                goal_offset_steps=eval_cfg.eval.goal_offset_steps,
            )
            reset_world_to_case(cem_world, case, eval_cfg.eval.callables)
            base_info = build_base_info(cem_world)
            prepared = move_info_to_device(
                prepare_info_dict(base_info, process=process, transform=transform),
                cfg.analysis.device,
            )

            solver_kwargs = get_solver_kwargs(eval_cfg)
            solver_kwargs["device"] = str(cfg.analysis.device)
            for key, value in OmegaConf.to_container(cfg.analysis.planner.solver_overrides, resolve=True).items():
                if value is not None:
                    solver_kwargs[key] = value
            solver = TracingCEMSolver(
                model=model,
                store_full_samples=False,
                **solver_kwargs,
            )
            solver.configure(
                action_space=cem_world.single_action_space,
                n_envs=1,
                config=build_plan_namespace(eval_cfg.plan_config),
            )
            cem_output = solver.solve(prepared)

            center = cem_output["actions"][0].numpy().astype(np.float32)
            scale = cem_output["var"][-1][0].numpy().astype(np.float32)
            center_flat = center.reshape(-1)
            scale_flat = np.maximum(scale.reshape(-1), float(cfg.analysis.eps))
            action_block = int(eval_cfg.plan_config.action_block)
            horizon = int(eval_cfg.plan_config.horizon)
            action_dim = int(np.asarray(cem_world.single_action_space.low).reshape(-1).shape[0])
            low, high = get_flat_action_bounds(cem_world.single_action_space, horizon=horizon, action_block=action_block)

            flat_candidates = sample_local_plan_candidates(
                center=center_flat,
                scale=scale_flat,
                low=low,
                high=high,
                num_candidates=int(cfg.analysis.planner.num_candidates),
                seed=cfg.seed + case_idx,
                noise_scale=float(cfg.analysis.planner.noise_scale),
            )
            action_candidates = flat_candidates.reshape(flat_candidates.shape[0], horizon, action_block * action_dim)
            action_blocks = unflatten_action_candidates(
                flat_candidates,
                horizon=horizon,
                action_block=action_block,
                action_dim=action_dim,
            )

            model_costs = compute_model_costs(
                model,
                base_info=base_info,
                action_candidates=action_candidates,
                process=process,
                transform=transform,
                device=cfg.analysis.device,
            )
            env_costs = evaluate_env_costs(
                env_world,
                case,
                eval_cfg.eval.callables,
                action_blocks,
                eval_cfg=eval_cfg,
            )

            spearman_values.append(spearman_corr(model_costs, env_costs))
            overlap_values.append(compute_topk_overlap(model_costs, env_costs, topk=int(cfg.analysis.planner.topk)))
            hit_values.append(float(np.argmin(model_costs) == np.argmin(env_costs)))
            gap_values.append(float(np.mean(np.abs(zscore(model_costs, eps=float(cfg.analysis.eps)) - zscore(env_costs, eps=float(cfg.analysis.eps))))))
            smoothness_values.append(compute_neighbor_smoothness(flat_candidates, model_costs))
            model_cost_records.append(model_costs.astype(np.float32))
            env_cost_records.append(env_costs.astype(np.float32))

    finally:
        try:
            cem_world.close()
        except Exception:
            pass
        try:
            env_world.close()
        except Exception:
            pass
        del cem_world
        del env_world
        gc.collect()

    return {
        "planner_cf_spearman_model_env_rank_values": np.asarray(spearman_values, dtype=np.float32),
        "planner_cf_topk_overlap_model_env_values": np.asarray(overlap_values, dtype=np.float32),
        "planner_cf_best_action_hit_rate_values": np.asarray(hit_values, dtype=np.float32),
        "planner_cf_local_cost_gap_mean_values": np.asarray(gap_values, dtype=np.float32),
        "planner_cf_neighbor_smoothness_values": np.asarray(smoothness_values, dtype=np.float32),
        "planner_cf_model_costs": np.stack(model_cost_records).astype(np.float32),
        "planner_cf_env_costs": np.stack(env_cost_records).astype(np.float32),
        "planner_cf_eval_episodes": np.asarray(eval_episodes, dtype=np.int64),
        "planner_cf_eval_start_idx": np.asarray(eval_start_idx, dtype=np.int64),
    }


def encode_pixels_with_model(model, pixels: np.ndarray, transform: dict, device: str):
    info = prepare_info_dict({"pixels": pixels}, process={}, transform=transform)
    info = move_info_to_device(info, device)
    with torch.inference_mode():
        encoded = model.encode(info)
    return encoded["emb"][:, -1].detach()


def compute_visual_latent_costs(model, final_pixels: np.ndarray, goal_pixels: np.ndarray, transform: dict, device: str):
    final_emb = encode_pixels_with_model(model, final_pixels, transform=transform, device=device)
    goal_emb = encode_pixels_with_model(model, goal_pixels, transform=transform, device=device)
    costs = (final_emb - goal_emb.detach()).pow(2).sum(dim=-1)
    return costs.detach().cpu().numpy().astype(np.float32)


def evaluate_visual_latent_env_costs_batched(
    model,
    world,
    case: dict,
    callables,
    action_blocks: np.ndarray,
    transform: dict,
    device: str,
):
    costs = []
    total = action_blocks.shape[0]
    batch_size = world.num_envs
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk = action_blocks[start:end]
        actual = chunk.shape[0]
        if actual < batch_size:
            padding = np.repeat(chunk[-1:][:, ...], batch_size - actual, axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)
        goal_broadcast = reset_world_to_case(world, case, callables)
        step_world_with_blocks(world, chunk, goal_broadcast)
        final_pixels = np.asarray(world.infos["pixels"][:actual, -1])
        goal_pixels = np.asarray(goal_broadcast["goal"][:actual, -1])
        costs.append(
            compute_visual_latent_costs(
                model,
                final_pixels[:, None, ...],
                goal_pixels[:, None, ...],
                transform=transform,
                device=device,
            )
        )
    return np.concatenate(costs, axis=0).astype(np.float32)


def evaluate_visual_latent_env_costs(
    model,
    world,
    case: dict,
    callables,
    action_blocks: np.ndarray,
    transform: dict,
    device: str,
    eval_cfg: DictConfig,
):
    try:
        return evaluate_visual_latent_env_costs_batched(
            model,
            world,
            case,
            callables,
            action_blocks,
            transform=transform,
            device=device,
        )
    except Exception as exc:
        if world.num_envs <= 1 or "Offscreen framebuffer is not complete" not in str(exc):
            raise
        print(
            f"  framebuffer allocation failed with env_batch_size={world.num_envs}; "
            "falling back to serial visual-latent env-cost evaluation"
        )
        fallback_world = build_world_from_eval_config(eval_cfg, num_envs=1)
        try:
            return evaluate_visual_latent_env_costs_batched(
                model,
                fallback_world,
                case,
                callables,
                action_blocks,
                transform=transform,
                device=device,
            )
        finally:
            del fallback_world
            gc.collect()


def run_shared_planner_counterfactual(
    model,
    dataset,
    process,
    transform,
    eval_cfg: DictConfig,
    cfg: DictConfig,
    eval_episodes: np.ndarray,
    eval_start_idx: np.ndarray,
    shared_planner_cache: dict,
):
    model_world = build_world_from_eval_config(eval_cfg, num_envs=1)
    env_cost_mode = shared_planner_env_cost_mode(cfg)
    visual_env_world = None
    if env_cost_mode == "visual_latent":
        visual_env_world = build_world_from_eval_config(eval_cfg, num_envs=int(cfg.analysis.shared_planner.env_batch_size))

    spearman_values = []
    overlap_values = []
    expert_model_rank_values = []
    expert_env_rank_values = []
    model_best_env_regret_values = []
    expert_env_regret_values = []
    model_best_env_cost_values = []
    env_best_cost_values = []
    model_topk_env_min_values = []
    model_topk_env_mean_values = []
    model_topk_env_regret_values = []
    model_best_env_regret_norm_values = []
    model_topk_env_regret_norm_values = []
    env_best_model_rank_values = []
    env_best_model_rank_norm_values = []
    model_topk_good_hit_values = []
    model_top1_good_hit_values = []
    model_topk_good_precision_values = []
    pairwise_preference_acc_values = []
    pairwise_preference_weighted_acc_values = []
    pairwise_preference_pair_ratio_values = []
    pairwise_preference_gap_threshold_values = []
    model_cost_records = []
    env_cost_model_records = []

    try:
        action_block = int(eval_cfg.plan_config.action_block)
        horizon = int(eval_cfg.plan_config.horizon)
        action_dim = int(np.asarray(model_world.single_action_space.low).reshape(-1).shape[0])
        flat_candidate_records = np.asarray(shared_planner_cache["shared_planner_flat_candidates"], dtype=np.float32)
        env_cost_records = None
        if env_cost_mode == "state_mse":
            env_cost_records = np.asarray(shared_planner_cache["shared_planner_env_costs"], dtype=np.float32)

        for case_idx, (episode_idx, start_step) in enumerate(zip(eval_episodes, eval_start_idx), start=1):
            case = build_eval_case(
                dataset,
                episode_idx=int(episode_idx),
                start_step=int(start_step),
                goal_offset_steps=eval_cfg.eval.goal_offset_steps,
            )
            reset_world_to_case(model_world, case, eval_cfg.eval.callables)
            base_info = build_base_info(model_world)

            record_idx = case_idx - 1
            flat_candidates = flat_candidate_records[record_idx]
            action_candidates = flat_candidates.reshape(flat_candidates.shape[0], horizon, action_block * action_dim)
            action_blocks = unflatten_action_candidates(
                flat_candidates,
                horizon=horizon,
                action_block=action_block,
                action_dim=action_dim,
            )
            if env_cost_mode == "visual_latent":
                env_costs = evaluate_visual_latent_env_costs(
                    model,
                    visual_env_world,
                    case,
                    eval_cfg.eval.callables,
                    action_blocks,
                    transform=transform,
                    device=cfg.analysis.device,
                    eval_cfg=eval_cfg,
                )
            else:
                env_costs = env_cost_records[record_idx]

            model_costs = compute_model_costs(
                model,
                base_info=base_info,
                action_candidates=action_candidates,
                process=process,
                transform=transform,
                device=cfg.analysis.device,
            )

            model_best = int(np.argmin(model_costs))
            env_best = int(np.argmin(env_costs))
            spearman_values.append(spearman_corr(model_costs, env_costs))
            overlap_values.append(compute_topk_overlap(model_costs, env_costs, topk=int(cfg.analysis.shared_planner.topk)))
            expert_model_rank_values.append(rank_of_index(model_costs, 0))
            expert_env_rank_values.append(rank_of_index(env_costs, 0))
            model_best_env_regret_values.append(float(env_costs[model_best] - env_costs[env_best]))
            expert_env_regret_values.append(float(env_costs[0] - env_costs[env_best]))
            value_metrics = shared_planner_value_metrics(
                model_costs,
                env_costs,
                topk=int(cfg.analysis.shared_planner.topk),
                cfg=cfg,
            )
            model_best_env_cost_values.append(value_metrics["model_best_env_cost"])
            env_best_cost_values.append(value_metrics["env_best_cost"])
            model_topk_env_min_values.append(value_metrics["model_topk_env_min"])
            model_topk_env_mean_values.append(value_metrics["model_topk_env_mean"])
            model_topk_env_regret_values.append(value_metrics["model_topk_env_regret"])
            model_best_env_regret_norm_values.append(value_metrics["model_best_env_regret_norm"])
            model_topk_env_regret_norm_values.append(value_metrics["model_topk_env_regret_norm"])
            env_best_model_rank_values.append(value_metrics["env_best_model_rank"])
            env_best_model_rank_norm_values.append(value_metrics["env_best_model_rank_norm"])
            model_topk_good_hit_values.append(value_metrics["model_topk_good_hit"])
            model_top1_good_hit_values.append(value_metrics["model_top1_good_hit"])
            model_topk_good_precision_values.append(value_metrics["model_topk_good_precision"])
            pairwise_preference_acc_values.append(value_metrics["pairwise_preference_acc"])
            pairwise_preference_weighted_acc_values.append(value_metrics["pairwise_preference_weighted_acc"])
            pairwise_preference_pair_ratio_values.append(value_metrics["pairwise_preference_pair_ratio"])
            pairwise_preference_gap_threshold_values.append(value_metrics["pairwise_preference_gap_threshold"])
            model_cost_records.append(model_costs.astype(np.float32))
            env_cost_model_records.append(env_costs.astype(np.float32))

    finally:
        try:
            model_world.close()
        except Exception:
            pass
        if visual_env_world is not None:
            try:
                visual_env_world.close()
            except Exception:
                pass
        del model_world
        del visual_env_world
        gc.collect()

    output = {
        "shared_planner_spearman_values": np.asarray(spearman_values, dtype=np.float32),
        "shared_planner_topk_overlap_values": np.asarray(overlap_values, dtype=np.float32),
        "shared_planner_expert_model_rank_values": np.asarray(expert_model_rank_values, dtype=np.float32),
        "shared_planner_expert_env_rank_values": np.asarray(expert_env_rank_values, dtype=np.float32),
        "shared_planner_model_best_env_regret_values": np.asarray(model_best_env_regret_values, dtype=np.float32),
        "shared_planner_expert_env_regret_values": np.asarray(expert_env_regret_values, dtype=np.float32),
        "shared_planner_model_best_env_cost_values": np.asarray(model_best_env_cost_values, dtype=np.float32),
        "shared_planner_env_best_cost_values": np.asarray(env_best_cost_values, dtype=np.float32),
        "shared_planner_model_topk_env_min_values": np.asarray(model_topk_env_min_values, dtype=np.float32),
        "shared_planner_model_topk_env_mean_values": np.asarray(model_topk_env_mean_values, dtype=np.float32),
        "shared_planner_model_topk_env_regret_values": np.asarray(model_topk_env_regret_values, dtype=np.float32),
        "shared_planner_model_best_env_regret_norm_values": np.asarray(model_best_env_regret_norm_values, dtype=np.float32),
        "shared_planner_model_topk_env_regret_norm_values": np.asarray(model_topk_env_regret_norm_values, dtype=np.float32),
        "shared_planner_env_best_model_rank_values": np.asarray(env_best_model_rank_values, dtype=np.float32),
        "shared_planner_env_best_model_rank_norm_values": np.asarray(env_best_model_rank_norm_values, dtype=np.float32),
        "shared_planner_model_topk_good_hit_values": np.asarray(model_topk_good_hit_values, dtype=np.float32),
        "shared_planner_model_top1_good_hit_values": np.asarray(model_top1_good_hit_values, dtype=np.float32),
        "shared_planner_model_topk_good_precision_values": np.asarray(model_topk_good_precision_values, dtype=np.float32),
        "shared_planner_pairwise_preference_acc_values": np.asarray(pairwise_preference_acc_values, dtype=np.float32),
        "shared_planner_pairwise_preference_weighted_acc_values": np.asarray(pairwise_preference_weighted_acc_values, dtype=np.float32),
        "shared_planner_pairwise_preference_pair_ratio_values": np.asarray(pairwise_preference_pair_ratio_values, dtype=np.float32),
        "shared_planner_pairwise_preference_gap_threshold_values": np.asarray(pairwise_preference_gap_threshold_values, dtype=np.float32),
        "shared_planner_eval_episodes": np.asarray(eval_episodes, dtype=np.int64),
        "shared_planner_eval_start_idx": np.asarray(eval_start_idx, dtype=np.int64),
        "shared_planner_candidate_hash": np.asarray(shared_planner_cache["shared_planner_candidate_hash"]),
        "shared_planner_env_cost_mode": np.asarray([env_cost_mode]),
    }
    if "shared_planner_env_cost_hash" in shared_planner_cache:
        output["shared_planner_env_cost_hash"] = np.asarray(shared_planner_cache["shared_planner_env_cost_hash"])
    if model_cost_records:
        output["shared_planner_model_costs"] = np.stack(model_cost_records).astype(np.float32)
        output["shared_planner_env_costs"] = np.stack(env_cost_model_records).astype(np.float32)
        output["shared_planner_flat_candidates"] = np.asarray(shared_planner_cache["shared_planner_flat_candidates"], dtype=np.float32)
        output["shared_planner_candidate_labels"] = np.asarray(shared_planner_cache["shared_planner_candidate_labels"])
    return output


def finalize_summary(payload: dict):
    summary = {}
    scalar_keys = [
        "one_step_cf_factual_top1_values",
        "one_step_cf_factual_margin_values",
        "one_step_cf_easy_margin_values",
        "one_step_cf_medium_margin_values",
        "one_step_cf_hard_margin_values",
        "one_step_cf_hard_negative_top1_values",
        "one_step_cf_identity_error_gap_values",
        "one_step_cf_identity_prediction_gap_values",
        "one_step_cf_action_distance_vs_prediction_shift_corr_values",
        "one_step_cf_pred_shift_easy_values",
        "one_step_cf_pred_shift_medium_values",
        "one_step_cf_pred_shift_hard_values",
        "one_step_cf_pred_shift_far_values",
        "one_step_cf_pred_shift_mid_values",
        "one_step_cf_pred_shift_near_values",
        "one_step_cf_response_gain_easy_values",
        "one_step_cf_response_gain_medium_values",
        "one_step_cf_response_gain_hard_values",
        "one_step_cf_response_gain_far_values",
        "one_step_cf_response_gain_mid_values",
        "one_step_cf_response_gain_near_values",
        "one_step_cf_pred_shift_monotonicity_values",
        "one_step_cf_response_rank_corr_values",
        "one_step_cf_response_slope_values",
        "one_step_cf_response_r2_values",
        "rollout_cf_suffix_factual_top1_values",
        "rollout_cf_suffix_margin_values",
        "rollout_cf_cf_separation_growth_values",
        "rollout_cf_prediction_shift_near_values",
        "rollout_cf_prediction_shift_mid_values",
        "rollout_cf_prediction_shift_far_values",
        "rollout_cf_action_distance_near_values",
        "rollout_cf_action_distance_mid_values",
        "rollout_cf_action_distance_far_values",
        "rollout_cf_response_gain_near_values",
        "rollout_cf_response_gain_mid_values",
        "rollout_cf_response_gain_far_values",
        "rollout_cf_prediction_shift_hard_values",
        "rollout_cf_prediction_shift_random_values",
        "rollout_cf_action_distance_hard_values",
        "rollout_cf_action_distance_random_values",
        "rollout_cf_response_gain_hard_values",
        "rollout_cf_response_gain_random_values",
        "rollout_cf_shift_monotonicity_values",
        "rollout_cf_shift_rank_corr_values",
        "rollout_cf_response_slope_values",
        "rollout_cf_response_r2_values",
        "rollout_cf_response_auc_values",
        "rollout_cf_dataset_random_prediction_shift_values",
        "rollout_cf_dataset_random_action_distance_values",
        "rollout_cf_dataset_random_response_gain_values",
        "rollout_cf_pre_intervention_invariance_values",
        "planner_cf_spearman_model_env_rank_values",
        "planner_cf_topk_overlap_model_env_values",
        "planner_cf_best_action_hit_rate_values",
        "planner_cf_local_cost_gap_mean_values",
        "planner_cf_neighbor_smoothness_values",
        "shared_planner_spearman_values",
        "shared_planner_topk_overlap_values",
        "shared_planner_expert_model_rank_values",
        "shared_planner_expert_env_rank_values",
        "shared_planner_model_best_env_regret_values",
        "shared_planner_expert_env_regret_values",
        "shared_planner_model_best_env_cost_values",
        "shared_planner_env_best_cost_values",
        "shared_planner_model_topk_env_min_values",
        "shared_planner_model_topk_env_mean_values",
        "shared_planner_model_topk_env_regret_values",
        "shared_planner_model_best_env_regret_norm_values",
        "shared_planner_model_topk_env_regret_norm_values",
        "shared_planner_env_best_model_rank_values",
        "shared_planner_env_best_model_rank_norm_values",
        "shared_planner_model_topk_good_hit_values",
        "shared_planner_model_top1_good_hit_values",
        "shared_planner_model_topk_good_precision_values",
        "shared_planner_pairwise_preference_acc_values",
        "shared_planner_pairwise_preference_weighted_acc_values",
        "shared_planner_pairwise_preference_pair_ratio_values",
        "shared_planner_pairwise_preference_gap_threshold_values",
    ]
    for key in scalar_keys:
        if key in payload:
            summary[key.removesuffix("_values")] = np.array([safe_mean(payload[key])], dtype=np.float32)

    if "one_step_cf_action_distance_values" in payload and "one_step_cf_prediction_shift_values" in payload:
        summary["one_step_cf_action_distance_vs_prediction_shift_corr"] = np.array(
            [
                safe_corr(
                    payload["one_step_cf_action_distance_values"].reshape(-1),
                    payload["one_step_cf_prediction_shift_values"].reshape(-1),
                )
            ],
            dtype=np.float32,
        )
    if "rollout_cf_suffix_margin_at_h_values" in payload:
        summary["rollout_cf_suffix_margin_at_h"] = np.asarray(
            payload["rollout_cf_suffix_margin_at_h_values"].mean(axis=0),
            dtype=np.float32,
        )
    if "rollout_cf_cf_separation_curve_values" in payload:
        summary["rollout_cf_cf_separation_curve"] = np.asarray(
            payload["rollout_cf_cf_separation_curve_values"].mean(axis=0),
            dtype=np.float32,
        )
        summary["rollout_cf_separation_curve"] = summary["rollout_cf_cf_separation_curve"]
    if "rollout_cf_response_slope_at_h_values" in payload:
        summary["rollout_cf_response_slope_at_h"] = np.asarray(
            payload["rollout_cf_response_slope_at_h_values"].mean(axis=0),
            dtype=np.float32,
        )
    if "rollout_cf_response_r2_at_h_values" in payload:
        summary["rollout_cf_response_r2_at_h"] = np.asarray(
            payload["rollout_cf_response_r2_at_h_values"].mean(axis=0),
            dtype=np.float32,
        )
    if "rollout_cf_cf_separation_growth" in summary:
        summary["rollout_cf_separation_growth"] = summary["rollout_cf_cf_separation_growth"]
    return summary


@hydra.main(version_base=None, config_path="./config/counterfactual", config_name="cube")
def run(cfg: DictConfig):
    train_dataset, val_set = build_val_dataset_and_split(cfg)
    eval_local_idx, bank_local_idx = split_eval_and_bank_indices(
        total=len(val_set),
        eval_count=int(cfg.analysis.eval_states),
        bank_count=int(cfg.analysis.action_bank_size),
        seed=int(cfg.seed),
    )
    eval_subset = torch.utils.data.Subset(val_set, eval_local_idx.tolist())
    bank_subset = torch.utils.data.Subset(val_set, bank_local_idx.tolist())
    eval_loader = build_loader(eval_subset, batch_size=int(cfg.analysis.batch_size), num_workers=int(cfg.analysis.num_workers))
    bank_loader = build_loader(bank_subset, batch_size=int(cfg.analysis.batch_size), num_workers=int(cfg.analysis.num_workers))

    action_bank = collect_action_bank(
        bank_loader,
        history_size=int(cfg.wm.history_size),
        rollout_horizon=int(cfg.analysis.rollout.horizon),
    )
    del bank_loader
    gc.collect()

    eval_cfg = load_eval_config(cfg.eval_config_name)
    with open_dict(eval_cfg):
        eval_cfg.cache_dir = str(get_cache_dir(cfg))
        eval_cfg.world.max_episode_steps = 2 * int(eval_cfg.eval.eval_budget)

    planner_dataset = None
    planner_process = None
    planner_transform = None
    planner_eval_episodes = None
    planner_eval_start_idx = None
    shared_planner_eval_episodes = None
    shared_planner_eval_start_idx = None
    shared_action_bank = None
    shared_planner_cache = None
    if cfg.analysis.planner.enabled or cfg.analysis.shared_planner.enabled:
        from eval import get_dataset

        planner_dataset = get_dataset(eval_cfg, eval_cfg.eval.dataset_name)
        planner_process, planner_transform = build_process_and_transform(eval_cfg, planner_dataset)
    if cfg.analysis.planner.enabled:
        planner_eval_episodes, planner_eval_start_idx = sample_eval_rows(
            eval_cfg,
            planner_dataset,
            seed=cfg.seed,
            override_num_eval=int(cfg.analysis.planner.num_eval),
        )
    if cfg.analysis.shared_planner.enabled:
        shared_planner_eval_episodes, shared_planner_eval_start_idx = sample_eval_rows(
            eval_cfg,
            planner_dataset,
            seed=cfg.seed + 10_003,
            override_num_eval=int(cfg.analysis.shared_planner.num_eval),
        )
        raw_action = np.asarray(planner_dataset.get_col_data("action"))
        action_dim = int(raw_action.reshape(raw_action.shape[0], -1).shape[-1])
        shared_action_bank = np.empty(
            (0, int(eval_cfg.plan_config.horizon) * int(eval_cfg.plan_config.action_block) * action_dim),
            dtype=np.float32,
        )
        if int(cfg.analysis.shared_planner.random_count) > 0:
            shared_bank_rows = sample_eval_rows(
                eval_cfg,
                planner_dataset,
                seed=cfg.seed + 20_003,
                override_num_eval=int(cfg.analysis.shared_planner.random_bank_size),
            )
            shared_action_bank = build_shared_action_bank(
                planner_dataset,
                eval_cfg=eval_cfg,
                rows=shared_bank_rows,
                horizon=int(eval_cfg.plan_config.horizon),
                action_block=int(eval_cfg.plan_config.action_block),
                action_dim=action_dim,
            )
        shared_planner_cache = build_or_load_shared_planner_cache(
            planner_dataset,
            eval_cfg=eval_cfg,
            cfg=cfg,
            eval_episodes=shared_planner_eval_episodes,
            eval_start_idx=shared_planner_eval_start_idx,
            shared_action_bank=shared_action_bank,
        )

    run_dir, checkpoint_paths = collect_checkpoint_paths(cfg)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found under {run_dir} matching {cfg.checkpoints.pattern}")

    train_cfg_path = run_dir / "config.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Missing training config at {train_cfg_path}")
    train_cfg = normalize_train_cfg(OmegaConf.load(train_cfg_path))
    if int(train_cfg.wm.history_size) != int(cfg.wm.history_size):
        raise ValueError(
            f"Counterfactual config history_size={cfg.wm.history_size} does not match training config history_size={train_cfg.wm.history_size}"
        )

    output_dir = run_dir / cfg.analysis.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"

    total_checkpoints = len(checkpoint_paths)
    total_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and str(cfg.analysis.device).startswith("cuda") else None
    wall_start = torch.cuda.Event(enable_timing=True) if total_start is not None else None
    if total_start is not None:
        total_start.record()
        wall_start.record()
    total_wall_time = 0.0

    print(
        f"Counterfactual run: {total_checkpoints} checkpoint(s), "
        f"{len(eval_subset)} eval state(s), "
        f"bank_size={len(bank_subset)}, "
        f"planner_eval={int(cfg.analysis.planner.num_eval) if cfg.analysis.planner.enabled else 0}, "
        f"shared_planner_eval={int(cfg.analysis.shared_planner.num_eval) if cfg.analysis.shared_planner.enabled else 0}"
    )

    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths, start=1):
        output_path = output_path_for_checkpoint(run_dir, cfg.analysis.output_subdir, checkpoint_path)
        if output_path.exists() and not cfg.analysis.overwrite:
            print(f"Skipping existing counterfactual file: {output_path}")
            continue

        checkpoint_start = __import__("time").time()
        print(f"[checkpoint {checkpoint_idx}/{total_checkpoints}] Analyzing {checkpoint_path.name}")
        model = load_model_object(checkpoint_path, cfg.analysis.device)

        aggregated = {
            "epoch": np.array([parse_epoch(checkpoint_path)], dtype=np.int32),
            "checkpoint_path": np.asarray([str(checkpoint_path)]),
            "seed": np.array([int(cfg.seed)], dtype=np.int32),
            "eval_local_indices": eval_local_idx.astype(np.int64),
            "bank_local_indices": bank_local_idx.astype(np.int64),
            "one_step_cf_bucket_order_legacy": np.asarray(["easy", "medium", "hard"]),
            "one_step_cf_bucket_order": np.asarray(["far", "mid", "near"]),
        }

        one_step_chunks = []
        rollout_chunks = []

        sample_offset = 0
        for batch_idx, batch in enumerate(eval_loader, start=1):
            batch = move_batch_to_device(batch, cfg.analysis.device)
            outputs = compute_model_outputs(model, batch, train_cfg)

            if cfg.analysis.one_step.enabled:
                one_step_chunks.append(
                    run_one_step_counterfactual(
                        model=model,
                        outputs=outputs,
                        batch=batch,
                        action_bank_last=action_bank["last_actions"],
                        cfg=cfg,
                    )
                )

            if cfg.analysis.rollout.enabled:
                rollout_chunks.append(
                    run_rollout_counterfactual(
                        model=model,
                        outputs=outputs,
                        batch=batch,
                        action_bank_suffix=action_bank["suffix_actions"],
                        cfg=cfg,
                        sample_offset=sample_offset,
                    )
                )

            sample_offset += batch["action"].size(0)
            if batch_idx % max(int(cfg.analysis.progress_every), 1) == 0 or batch_idx == len(eval_loader):
                elapsed = __import__("time").time() - checkpoint_start
                avg_batch_time = elapsed / max(batch_idx, 1)
                eta = avg_batch_time * max(len(eval_loader) - batch_idx, 0)
                print(
                    f"  batch {batch_idx}/{len(eval_loader)} "
                    f"({avg_batch_time:.2f}s avg) "
                    f"ckpt ETA {format_duration(eta)}"
                )

        def _merge_chunks(chunks):
            merged = {}
            if not chunks:
                return merged
            for key in chunks[0]:
                merged[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
            return merged

        aggregated.update(_merge_chunks(one_step_chunks))
        aggregated.update(_merge_chunks(rollout_chunks))

        if cfg.analysis.planner.enabled:
            aggregated.update(
                run_planner_counterfactual(
                    model=model,
                    dataset=planner_dataset,
                    process=planner_process,
                    transform=planner_transform,
                    eval_cfg=eval_cfg,
                    cfg=cfg,
                    eval_episodes=planner_eval_episodes,
                    eval_start_idx=planner_eval_start_idx,
                )
            )

        if cfg.analysis.shared_planner.enabled:
            aggregated.update(
                run_shared_planner_counterfactual(
                    model=model,
                    dataset=planner_dataset,
                    process=planner_process,
                    transform=planner_transform,
                    eval_cfg=eval_cfg,
                    cfg=cfg,
                    eval_episodes=shared_planner_eval_episodes,
                    eval_start_idx=shared_planner_eval_start_idx,
                    shared_planner_cache=shared_planner_cache,
                )
            )

        aggregated.update(finalize_summary(aggregated))

        if cfg.analysis.save_plots and "one_step_cf_easy_margin_values" in aggregated:
            plot_path = output_dir / "plots" / f"{checkpoint_path.stem}_one_step_margin_bars.png"
            build_plot(
                plot_path,
                labels=["Far", "Mid", "Near"],
                means=[
                    safe_mean(aggregated["one_step_cf_easy_margin_values"]),
                    safe_mean(aggregated["one_step_cf_medium_margin_values"]),
                    safe_mean(aggregated["one_step_cf_hard_margin_values"]),
                ],
                stds=[
                    safe_std(aggregated["one_step_cf_easy_margin_values"]),
                    safe_std(aggregated["one_step_cf_medium_margin_values"]),
                    safe_std(aggregated["one_step_cf_hard_margin_values"]),
                ],
                title=f"{checkpoint_path.stem} One-Step Factual Margin",
            )

        np.savez(output_path, **aggregated)

        summary_record = {
            "checkpoint": checkpoint_path.name,
            "epoch": int(aggregated["epoch"][0]),
        }
        for key in [
            "one_step_cf_factual_top1",
            "one_step_cf_factual_margin",
            "one_step_cf_easy_margin",
            "one_step_cf_medium_margin",
            "one_step_cf_hard_margin",
            "one_step_cf_hard_negative_top1",
            "one_step_cf_action_distance_vs_prediction_shift_corr",
            "one_step_cf_pred_shift_easy",
            "one_step_cf_pred_shift_medium",
            "one_step_cf_pred_shift_hard",
            "one_step_cf_pred_shift_far",
            "one_step_cf_pred_shift_mid",
            "one_step_cf_pred_shift_near",
            "one_step_cf_response_gain_easy",
            "one_step_cf_response_gain_medium",
            "one_step_cf_response_gain_hard",
            "one_step_cf_response_gain_far",
            "one_step_cf_response_gain_mid",
            "one_step_cf_response_gain_near",
            "one_step_cf_pred_shift_monotonicity",
            "one_step_cf_response_rank_corr",
            "one_step_cf_response_slope",
            "one_step_cf_response_r2",
            "rollout_cf_suffix_factual_top1",
            "rollout_cf_suffix_margin",
            "rollout_cf_cf_separation_growth",
            "rollout_cf_separation_growth",
            "rollout_cf_prediction_shift_near",
            "rollout_cf_prediction_shift_mid",
            "rollout_cf_prediction_shift_far",
            "rollout_cf_action_distance_near",
            "rollout_cf_action_distance_mid",
            "rollout_cf_action_distance_far",
            "rollout_cf_response_gain_near",
            "rollout_cf_response_gain_mid",
            "rollout_cf_response_gain_far",
            "rollout_cf_prediction_shift_hard",
            "rollout_cf_prediction_shift_random",
            "rollout_cf_action_distance_hard",
            "rollout_cf_action_distance_random",
            "rollout_cf_response_gain_hard",
            "rollout_cf_response_gain_random",
            "rollout_cf_shift_monotonicity",
            "rollout_cf_shift_rank_corr",
            "rollout_cf_response_slope",
            "rollout_cf_response_r2",
            "rollout_cf_response_auc",
            "rollout_cf_dataset_random_prediction_shift",
            "rollout_cf_dataset_random_action_distance",
            "rollout_cf_dataset_random_response_gain",
            "rollout_cf_pre_intervention_invariance",
            "planner_cf_spearman_model_env_rank",
            "planner_cf_topk_overlap_model_env",
            "planner_cf_best_action_hit_rate",
            "planner_cf_local_cost_gap_mean",
            "planner_cf_neighbor_smoothness",
            "shared_planner_spearman",
            "shared_planner_topk_overlap",
            "shared_planner_expert_model_rank",
            "shared_planner_expert_env_rank",
            "shared_planner_model_best_env_regret",
            "shared_planner_expert_env_regret",
            "shared_planner_model_best_env_cost",
            "shared_planner_env_best_cost",
            "shared_planner_model_topk_env_min",
            "shared_planner_model_topk_env_mean",
            "shared_planner_model_topk_env_regret",
            "shared_planner_model_best_env_regret_norm",
            "shared_planner_model_topk_env_regret_norm",
            "shared_planner_env_best_model_rank",
            "shared_planner_env_best_model_rank_norm",
            "shared_planner_model_topk_good_hit",
            "shared_planner_model_top1_good_hit",
            "shared_planner_model_topk_good_precision",
            "shared_planner_pairwise_preference_acc",
            "shared_planner_pairwise_preference_weighted_acc",
            "shared_planner_pairwise_preference_pair_ratio",
            "shared_planner_pairwise_preference_gap_threshold",
        ]:
            if key in aggregated:
                summary_record[key] = float(np.asarray(aggregated[key]).reshape(-1)[0])
        upsert_summary_record(summary_path, summary_record)

        checkpoint_time = __import__("time").time() - checkpoint_start
        total_wall_time += checkpoint_time
        avg_ckpt = total_wall_time / checkpoint_idx
        total_eta = avg_ckpt * max(total_checkpoints - checkpoint_idx, 0)
        print(
            f"[checkpoint {checkpoint_idx}/{total_checkpoints}] done in "
            f"{format_duration(checkpoint_time)}; total ETA {format_duration(total_eta)}"
        )

        del model
        gc.collect()


if __name__ == "__main__":
    run()
