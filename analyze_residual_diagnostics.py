from pathlib import Path
import json
import os
import re
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/lewm-matplotlib-{os.getuid()}")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


# MUJOCO_EGL_DEVICE_ID, if set, indexes EGL devices independently of CUDA visibility.


import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict

from eval import evaluate_policy, save_structured_results
from train import (
    compute_inverse_outputs,
    compute_model_outputs,
)
from utils import get_column_normalizer, get_img_preprocessor


def load_model_object(path: Path, device: str):
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    return model


def parse_epoch(path: Path):
    match = re.search(r"_epoch_(\d+)_object\.ckpt$", path.name)
    return int(match.group(1)) if match else -1


def get_cache_dir(cfg: DictConfig):
    return Path(cfg.cache_dir or swm.data.utils.get_cache_dir())


def resolve_run_dir(cfg: DictConfig):
    run_dir = Path(cfg.checkpoints.run_dir)
    if run_dir.is_absolute():
        return run_dir
    return get_cache_dir(cfg) / run_dir


def collect_checkpoint_paths(cfg: DictConfig):
    run_dir = resolve_run_dir(cfg)
    paths = sorted(run_dir.glob(cfg.checkpoints.pattern), key=parse_epoch)
    requested_epochs = set(OmegaConf.to_container(cfg.checkpoints.epochs, resolve=True) or [])
    if requested_epochs:
        paths = [path for path in paths if parse_epoch(path) in requested_epochs]
    max_count = cfg.checkpoints.max_count
    if max_count:
        paths = paths[:max_count]
    return run_dir, paths


def format_duration(seconds: float):
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(int(round(seconds)), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def normalize_train_cfg(train_cfg: DictConfig):
    with open_dict(train_cfg):
        if "loss" not in train_cfg:
            train_cfg.loss = {}
        if "inverse" not in train_cfg.loss:
            train_cfg.loss.inverse = {}
        train_cfg.loss.inverse.setdefault("enabled", False)
        train_cfg.loss.inverse.setdefault("weight", 0.1)
        train_cfg.loss.inverse.setdefault("grad_mode", "end_to_end")
        if "diagnostics" not in train_cfg:
            train_cfg.diagnostics = {}
        train_cfg.diagnostics.setdefault("eps", 1.0e-8)
    return train_cfg


def build_val_loader(cfg: DictConfig):
    dataset = swm.data.HDF5Dataset(
        **cfg.data.dataset,
        transform=None,
        cache_dir=get_cache_dir(cfg),
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    transforms.append(get_column_normalizer(dataset, "action", "action"))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    _, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    return torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.analysis.batch_size,
        num_workers=cfg.analysis.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )


def move_batch_to_device(batch, device: str):
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    output["action"] = torch.nan_to_num(output["action"], 0.0)
    return output


def teacher_forced_metrics(model, emb, act_emb, history_size: int, eps: float):
    pred_delta_norms = []
    target_delta_norms = []
    delta_error_norms = []
    relative_delta_error_norms = []
    pred_target_delta_cos = []
    delta_residual_sq_sum = 0.0
    delta_target_sum = 0.0
    delta_target_sq_sum = 0.0
    delta_target_numel = 0

    total_steps = emb.size(1)
    for end_idx in range(history_size - 1, total_steps - 1):
        ctx_emb = emb[:, end_idx - history_size + 1 : end_idx + 1]
        ctx_act = act_emb[:, end_idx - history_size + 1 : end_idx + 1]
        pred_next = model.predict(ctx_emb, ctx_act)[:, -1]
        z_t = ctx_emb[:, -1]
        z_next = emb[:, end_idx + 1]
        pred_delta = pred_next - z_t
        target_delta = z_next - z_t
        delta_error = pred_delta - target_delta

        pred_delta_norms.append(pred_delta.norm(dim=-1).cpu())
        target_delta_norms.append(target_delta.norm(dim=-1).cpu())
        delta_error_norms.append(delta_error.norm(dim=-1).cpu())
        relative_delta_error_norms.append(
            (delta_error.norm(dim=-1) / (target_delta.norm(dim=-1) + eps)).cpu()
        )
        pred_target_delta_cos.append(
            F.cosine_similarity(pred_delta, target_delta, dim=-1, eps=eps).cpu()
        )
        delta_residual_sq_sum += float(delta_error.pow(2).sum().item())
        delta_target_sum += float(target_delta.sum().item())
        delta_target_sq_sum += float(target_delta.pow(2).sum().item())
        delta_target_numel += int(target_delta.numel())

    return {
        "pred_delta_norms": torch.cat(pred_delta_norms).numpy(),
        "target_delta_norms": torch.cat(target_delta_norms).numpy(),
        "delta_error_norms": torch.cat(delta_error_norms).numpy(),
        "relative_delta_error_norms": torch.cat(relative_delta_error_norms).numpy(),
        "pred_target_delta_cos": torch.cat(pred_target_delta_cos).numpy(),
        "delta_residual_sq_sum": np.array([delta_residual_sq_sum], dtype=np.float64),
        "delta_target_sum": np.array([delta_target_sum], dtype=np.float64),
        "delta_target_sq_sum": np.array([delta_target_sq_sum], dtype=np.float64),
        "delta_target_numel": np.array([delta_target_numel], dtype=np.int64),
    }


def free_rollout_metrics(model, emb, action, history_size: int, horizon: int, eps: float):
    emb_history = emb[:, :history_size].clone()
    action_history = action[:, :history_size].clone()
    future_actions = action[:, history_size : history_size + horizon]
    predictions = []

    for step in range(future_actions.size(1)):
        act_emb = model.action_encoder(action_history)
        pred_next = model.predict(
            emb_history[:, -history_size:],
            act_emb[:, -history_size:],
        )[:, -1:]
        emb_history = torch.cat([emb_history, pred_next], dim=1)
        action_history = torch.cat([action_history, future_actions[:, step : step + 1]], dim=1)
        predictions.append(pred_next.squeeze(1))

    if not predictions:
        empty = np.empty((0,), dtype=np.float32)
        return {
            "rollout_mse": empty.reshape(0, 0),
            "rollout_cos": empty.reshape(0, 0),
            "rollout_velocity_mse": empty.reshape(0, 0),
            "rollout_velocity_cos": empty.reshape(0, 0),
            "delta_cos_consecutive": empty,
        }

    pred_future = torch.stack(predictions, dim=1)
    target_future = emb[:, history_size : history_size + pred_future.size(1)]
    rollout_error = pred_future - target_future
    rollout_mse = rollout_error.pow(2).mean(dim=-1).cpu().numpy()
    rollout_cos = F.cosine_similarity(pred_future, target_future, dim=-1, eps=eps).cpu().numpy()

    rollout_states = torch.cat([emb[:, history_size - 1 : history_size], pred_future], dim=1)
    deltas = rollout_states[:, 1:] - rollout_states[:, :-1]
    if deltas.size(1) > 1:
        delta_cos = F.cosine_similarity(deltas[:, :-1], deltas[:, 1:], dim=-1, eps=eps).cpu().numpy()
    else:
        delta_cos = np.empty((rollout_states.size(0), 0), dtype=np.float32)

    if pred_future.size(1) > 1:
        pred_deltas = pred_future[:, 1:] - pred_future[:, :-1]
        target_deltas = target_future[:, 1:] - target_future[:, :-1]
        rollout_velocity_mse = (pred_deltas - target_deltas).pow(2).mean(dim=-1).cpu().numpy()
        rollout_velocity_cos = F.cosine_similarity(
            pred_deltas,
            target_deltas,
            dim=-1,
            eps=eps,
        ).cpu().numpy()
    else:
        rollout_velocity_mse = np.empty((pred_future.size(0), 0), dtype=np.float32)
        rollout_velocity_cos = np.empty((pred_future.size(0), 0), dtype=np.float32)

    return {
        "rollout_mse": rollout_mse,
        "rollout_cos": rollout_cos,
        "rollout_velocity_mse": rollout_velocity_mse,
        "rollout_velocity_cos": rollout_velocity_cos,
        "delta_cos_consecutive": delta_cos,
    }


def summarize_rollout(rollout_values: np.ndarray):
    if rollout_values.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return rollout_values.mean(axis=0), rollout_values.std(axis=0)


def summarize_local_rollout(rollout_mse_mean: np.ndarray, rollout_cos_mean: np.ndarray, local_horizon: int):
    local_horizon = int(max(local_horizon, 0))
    usable = min(local_horizon, rollout_mse_mean.size, rollout_cos_mean.size)
    if usable == 0:
        empty = np.empty((0,), dtype=np.float32)
        scalar_nan = np.array([np.nan], dtype=np.float32)
        return {
            "local_horizons": empty.astype(np.int32),
            "rollout_mse_local": empty,
            "rollout_cos_local": empty,
            "rollout_mse_auc_local": scalar_nan,
            "rollout_mse_mean_local": scalar_nan,
            "rollout_cos_mean_local": scalar_nan,
            "rollout_mse_growth_local": scalar_nan,
            "rollout_cos_drop_local": scalar_nan,
        }

    local_mse = rollout_mse_mean[:usable].astype(np.float32)
    local_cos = rollout_cos_mean[:usable].astype(np.float32)
    return {
        "local_horizons": np.arange(1, usable + 1, dtype=np.int32),
        "rollout_mse_local": local_mse,
        "rollout_cos_local": local_cos,
        "rollout_mse_auc_local": np.array([local_mse.sum()], dtype=np.float32),
        "rollout_mse_mean_local": np.array([local_mse.mean()], dtype=np.float32),
        "rollout_cos_mean_local": np.array([local_cos.mean()], dtype=np.float32),
        "rollout_mse_growth_local": np.array([local_mse[-1] - local_mse[0]], dtype=np.float32),
        "rollout_cos_drop_local": np.array([local_cos[0] - local_cos[-1]], dtype=np.float32),
    }


def checkpoint_output_path(run_dir: Path, output_subdir: str, checkpoint_path: Path):
    epoch = parse_epoch(checkpoint_path)
    return run_dir / output_subdir / f"{checkpoint_path.stem}_diagnostics_epoch_{epoch:03d}.npz"


def infer_model_type(cfg: DictConfig, model):
    if cfg.analysis.model_type:
        return cfg.analysis.model_type
    if getattr(model, "has_inverse_head", lambda: False)():
        return "res_inv" if getattr(model, "residual_target", False) else "abs_inv"
    return "res" if getattr(model, "residual_target", False) else "abs"


def to_policy_name(cfg: DictConfig, checkpoint_path: Path):
    cache_dir = get_cache_dir(cfg)
    relative = checkpoint_path.relative_to(cache_dir)
    return str(relative).removesuffix("_object.ckpt")


def run_planning_eval(cfg: DictConfig, checkpoint_path: Path):
    eval_cfg_path = Path(__file__).parent / "config" / "eval" / f"{cfg.planning_eval.config_name}.yaml"
    eval_cfg = OmegaConf.load(eval_cfg_path)
    with open_dict(eval_cfg):
        eval_cfg.cache_dir = str(get_cache_dir(cfg))
        eval_cfg.policy = to_policy_name(cfg, checkpoint_path)
        eval_cfg.output.save_structured = cfg.planning_eval.save_structured
    result = evaluate_policy(eval_cfg)
    if cfg.planning_eval.save_structured:
        save_structured_results(
            result["results_dir"] / eval_cfg.output.filename,
            eval_cfg,
            result["metrics"],
            result["evaluation_time"],
            result["eval_episodes"],
            result["eval_start_idx"],
        )
    return result


def compute_inverse_analysis(model, outputs, train_cfg):
    pred_loss = (outputs["pred_emb"] - outputs["tgt_emb"]).pow(2).mean()
    inverse_outputs = compute_inverse_outputs(
        model,
        outputs["ctx_emb"],
        outputs["pred_emb"],
        outputs["tgt_emb"],
        outputs["ctx_act"],
        outputs["ctx_act"],
        train_cfg,
        pred_loss=pred_loss,
    )
    pred_act_emb = inverse_outputs.pop("pred_act_emb", None)
    tgt_act_emb_used = inverse_outputs.pop("tgt_act_emb_used", None)
    if pred_act_emb is None or tgt_act_emb_used is None:
        return None
    return inverse_outputs, pred_act_emb, tgt_act_emb_used


def compute_one_step_action_separation(model, ctx_emb, ctx_act, eps: float):
    if ctx_emb.size(0) < 2 or ctx_act.size(1) == 0:
        return None

    perm_a = torch.randperm(ctx_emb.size(0), device=ctx_emb.device)
    perm_b = torch.randperm(ctx_emb.size(0), device=ctx_emb.device)
    act_a = ctx_act.clone()
    act_b = ctx_act.clone()
    act_a[:, -1] = ctx_act[perm_a, -1]
    act_b[:, -1] = ctx_act[perm_b, -1]

    pred_a = model.predict(ctx_emb, act_a)[:, -1]
    pred_b = model.predict(ctx_emb, act_b)[:, -1]
    pred_dist = (pred_a - pred_b).norm(dim=-1)
    act_dist = (act_a[:, -1] - act_b[:, -1]).norm(dim=-1)
    gain = pred_dist / (act_dist + eps)

    centered_pred = pred_dist - pred_dist.mean()
    centered_act = act_dist - act_dist.mean()
    denom = torch.sqrt(centered_pred.pow(2).sum() * centered_act.pow(2).sum()).clamp_min(eps)
    corr = (centered_pred * centered_act).sum() / denom

    return {
        "action_separation_pred_dist": pred_dist.detach().cpu().numpy(),
        "action_separation_act_dist": act_dist.detach().cpu().numpy(),
        "action_separation_gain": gain.detach().cpu().numpy(),
        "action_separation_corr": np.array([float(corr.detach().cpu().item())], dtype=np.float32),
    }


def compute_action_conditioned_margin(pred_act_emb, tgt_act_emb, eps: float):
    if pred_act_emb.size(0) < 2:
        margin = pred_act_emb.new_zeros(pred_act_emb.size(0))
    else:
        perm = torch.randperm(pred_act_emb.size(0), device=pred_act_emb.device)
        neg_target = tgt_act_emb[perm]
        pos_err = (pred_act_emb - tgt_act_emb).pow(2).mean(dim=(-1, -2))
        neg_err = (pred_act_emb - neg_target).pow(2).mean(dim=(-1, -2))
        margin = neg_err - pos_err
    return {
        "action_conditioned_margin": margin.detach().cpu().numpy(),
    }


@hydra.main(version_base=None, config_path="./config/diagnostics", config_name="cube")
def run(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    run_dir, checkpoint_paths = collect_checkpoint_paths(cfg)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found under {run_dir} matching {cfg.checkpoints.pattern}")

    train_cfg_path = run_dir / "config.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Missing training config at {train_cfg_path}")
    train_cfg = normalize_train_cfg(OmegaConf.load(train_cfg_path))

    val_loader = build_val_loader(cfg)
    output_dir = run_dir / cfg.analysis.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.jsonl"
    total_checkpoints = len(checkpoint_paths)
    total_val_batches = len(val_loader)
    effective_total_batches = (
        min(total_val_batches, cfg.analysis.limit_batches)
        if cfg.analysis.limit_batches is not None
        else total_val_batches
    )
    progress_every = max(int(cfg.analysis.get("progress_every", 1)), 1)
    analysis_start_time = time.time()

    print(
        f"Diagnostics run: {total_checkpoints} checkpoint(s), "
        f"{effective_total_batches} batch(es)/checkpoint, "
        f"rollout_horizon={cfg.analysis.rollout_horizon}, "
        f"planning_eval={cfg.planning_eval.enabled}"
    )

    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths, start=1):
        output_path = checkpoint_output_path(run_dir, cfg.analysis.output_subdir, checkpoint_path)
        if output_path.exists() and not cfg.analysis.overwrite:
            print(f"Skipping existing diagnostics file: {output_path}")
            continue

        checkpoint_start_time = time.time()
        print(
            f"[checkpoint {checkpoint_idx}/{total_checkpoints}] "
            f"Analyzing {checkpoint_path.name}"
        )
        model = load_model_object(checkpoint_path, cfg.analysis.device)
        history_size = getattr(model.predictor, "pos_embedding").size(1)

        teacher_stats = {
            "pred_delta_norms": [],
            "target_delta_norms": [],
            "delta_error_norms": [],
            "relative_delta_error_norms": [],
            "pred_target_delta_cos": [],
            "delta_residual_sq_sum": [],
            "delta_target_sum": [],
            "delta_target_sq_sum": [],
            "delta_target_numel": [],
        }
        inverse_stats = {
            "inverse_loss_values": [],
            "inverse_cos_values": [],
            "inverse_to_pred_loss_ratio_values": [],
            "action_conditioned_margin": [],
        }
        action_separation_stats = {
            "action_separation_pred_dist": [],
            "action_separation_act_dist": [],
            "action_separation_gain": [],
            "action_separation_corr": [],
        }
        rollout_mse_batches = []
        rollout_cos_batches = []
        rollout_velocity_mse_batches = []
        rollout_velocity_cos_batches = []
        delta_cos_batches = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if cfg.analysis.limit_batches is not None and batch_idx >= cfg.analysis.limit_batches:
                    break

                batch_start_time = time.time()

                batch = move_batch_to_device(batch, cfg.analysis.device)
                outputs = compute_model_outputs(model, batch, train_cfg)
                emb = outputs["emb"]
                act_emb = outputs["act_emb"]

                tf_metrics = teacher_forced_metrics(
                    model,
                    emb,
                    act_emb,
                    history_size=history_size,
                    eps=cfg.analysis.eps,
                )
                for key, value in tf_metrics.items():
                    teacher_stats[key].append(value)

                action_separation = compute_one_step_action_separation(
                    model,
                    outputs["ctx_emb"],
                    outputs["ctx_act"],
                    cfg.analysis.eps,
                )
                if action_separation is not None:
                    for key, value in action_separation.items():
                        action_separation_stats[key].append(value)

                inverse_analysis = compute_inverse_analysis(model, outputs, train_cfg)
                if inverse_analysis is not None:
                    inverse_metrics, pred_act_emb, tgt_act_emb_used = inverse_analysis
                    inverse_stats["inverse_loss_values"].append(
                        np.array([float(inverse_metrics["inverse_loss"].cpu().item())], dtype=np.float32)
                    )
                    inverse_stats["inverse_cos_values"].append(
                        np.array([float(inverse_metrics["inverse_cos"].cpu().item())], dtype=np.float32)
                    )
                    inverse_stats["inverse_to_pred_loss_ratio_values"].append(
                        np.array(
                            [float(inverse_metrics["inverse_to_pred_loss_ratio"].cpu().item())],
                            dtype=np.float32,
                        )
                    )
                    inverse_stats["action_conditioned_margin"].append(
                        compute_action_conditioned_margin(pred_act_emb, tgt_act_emb_used, cfg.analysis.eps)[
                            "action_conditioned_margin"
                        ]
                    )

                rollout_metrics = free_rollout_metrics(
                    model,
                    emb,
                    batch["action"],
                    history_size=history_size,
                    horizon=cfg.analysis.rollout_horizon,
                    eps=cfg.analysis.eps,
                )
                if rollout_metrics["rollout_mse"].size:
                    rollout_mse_batches.append(rollout_metrics["rollout_mse"])
                    rollout_cos_batches.append(rollout_metrics["rollout_cos"])
                if rollout_metrics["rollout_velocity_mse"].size:
                    rollout_velocity_mse_batches.append(rollout_metrics["rollout_velocity_mse"])
                    rollout_velocity_cos_batches.append(rollout_metrics["rollout_velocity_cos"])
                if rollout_metrics["delta_cos_consecutive"].size:
                    delta_cos_batches.append(rollout_metrics["delta_cos_consecutive"])

                batches_done = batch_idx + 1
                if (
                    batches_done == effective_total_batches
                    or batches_done % progress_every == 0
                ):
                    elapsed = time.time() - checkpoint_start_time
                    avg_batch_time = elapsed / max(batches_done, 1)
                    checkpoint_eta = avg_batch_time * max(effective_total_batches - batches_done, 0)
                    total_elapsed = time.time() - analysis_start_time
                    completed_before_current = checkpoint_idx - 1
                    fractional_progress = completed_before_current + (
                        batches_done / max(effective_total_batches, 1)
                    )
                    avg_checkpoint_time = total_elapsed / max(fractional_progress, 1e-8)
                    total_eta = avg_checkpoint_time * max(total_checkpoints - fractional_progress, 0.0)
                    print(
                        f"  batch {batches_done}/{effective_total_batches} "
                        f"({avg_batch_time:.2f}s avg, last {time.time() - batch_start_time:.2f}s) "
                        f"ckpt ETA {format_duration(checkpoint_eta)}, "
                        f"total ETA {format_duration(total_eta)}"
                    )

        teacher_stats = {
            key: np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)
            for key, values in teacher_stats.items()
        }
        inverse_stats = {
            key: np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)
            for key, values in inverse_stats.items()
        }
        action_separation_stats = {
            key: np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)
            for key, values in action_separation_stats.items()
        }
        rollout_mse = (
            np.concatenate(rollout_mse_batches, axis=0)
            if rollout_mse_batches
            else np.empty((0, 0), dtype=np.float32)
        )
        rollout_cos = (
            np.concatenate(rollout_cos_batches, axis=0)
            if rollout_cos_batches
            else np.empty((0, 0), dtype=np.float32)
        )
        rollout_velocity_mse = (
            np.concatenate(rollout_velocity_mse_batches, axis=0)
            if rollout_velocity_mse_batches
            else np.empty((0, 0), dtype=np.float32)
        )
        rollout_velocity_cos = (
            np.concatenate(rollout_velocity_cos_batches, axis=0)
            if rollout_velocity_cos_batches
            else np.empty((0, 0), dtype=np.float32)
        )
        delta_cos_consecutive = (
            np.concatenate(delta_cos_batches, axis=0)
            if delta_cos_batches
            else np.empty((0,), dtype=np.float32)
        )
        delta_residual_sq_sum = float(teacher_stats.pop("delta_residual_sq_sum").sum())
        delta_target_sum = float(teacher_stats.pop("delta_target_sum").sum())
        delta_target_sq_sum = float(teacher_stats.pop("delta_target_sq_sum").sum())
        delta_target_numel = int(teacher_stats.pop("delta_target_numel").sum())
        if delta_target_numel > 0:
            delta_target_mean = delta_target_sum / delta_target_numel
            delta_target_sst = delta_target_sq_sum - delta_target_numel * (delta_target_mean ** 2)
            delta_r2_score = (
                1.0 - (delta_residual_sq_sum / delta_target_sst)
                if delta_target_sst > cfg.analysis.eps
                else np.nan
            )
        else:
            delta_r2_score = np.nan
        rollout_mse_mean, rollout_mse_std = summarize_rollout(rollout_mse)
        rollout_cos_mean, rollout_cos_std = summarize_rollout(rollout_cos)
        rollout_velocity_mse_mean, rollout_velocity_mse_std = summarize_rollout(rollout_velocity_mse)
        rollout_velocity_cos_mean, rollout_velocity_cos_std = summarize_rollout(rollout_velocity_cos)
        local_rollout_summary = summarize_local_rollout(
            rollout_mse_mean,
            rollout_cos_mean,
            cfg.analysis.local_rollout_horizon,
        )
        local_rollout_velocity_summary = summarize_local_rollout(
            rollout_velocity_mse_mean,
            rollout_velocity_cos_mean,
            max(cfg.analysis.local_rollout_horizon - 1, 0),
        )

        static_metrics = {}
        if (
            cfg.analysis.static_preservation.enabled
            and teacher_stats["target_delta_norms"].size
        ):
            target_delta_norms = teacher_stats["target_delta_norms"]
            pred_delta_norms = teacher_stats["pred_delta_norms"]
            relative_delta_error_norms = teacher_stats["relative_delta_error_norms"]
            if cfg.analysis.static_preservation.threshold_mode == "quantile":
                static_threshold = float(
                    np.quantile(target_delta_norms, cfg.analysis.static_preservation.quantile)
                )
            else:
                static_threshold = float(cfg.analysis.static_preservation.threshold)
            static_mask = target_delta_norms <= static_threshold
            static_metrics = {
                "static_target_threshold": np.array([static_threshold], dtype=np.float32),
                "static_sample_ratio": np.array([static_mask.mean()], dtype=np.float32),
                "static_sample_count": np.array([static_mask.sum()], dtype=np.int32),
                "static_pred_delta_norm_mean": np.array(
                    [pred_delta_norms[static_mask].mean()] if static_mask.any() else [np.nan],
                    dtype=np.float32,
                ),
                "static_target_delta_norm_mean": np.array(
                    [target_delta_norms[static_mask].mean()] if static_mask.any() else [np.nan],
                    dtype=np.float32,
                ),
                "static_relative_delta_error_mean": np.array(
                    [relative_delta_error_norms[static_mask].mean()] if static_mask.any() else [np.nan],
                    dtype=np.float32,
                ),
            }

        planning_metrics = {}
        if cfg.planning_eval.enabled:
            print("  running planning eval...")
            planning_result = run_planning_eval(cfg, checkpoint_path)
            planning_metrics = {
                "planning_success_rate": np.array(
                    [planning_result["metrics"].get("success_rate", np.nan)],
                    dtype=np.float32,
                ),
                "planning_evaluation_time": np.array(
                    [planning_result["evaluation_time"]],
                    dtype=np.float32,
                ),
                "planning_episode_successes": np.asarray(
                    planning_result["metrics"].get("episode_successes", [])
                ),
                "planning_eval_episodes": np.asarray(planning_result["eval_episodes"]),
                "planning_eval_start_idx": np.asarray(planning_result["eval_start_idx"]),
            }

        payload = {
            "checkpoint_path": np.array([str(checkpoint_path)]),
            "model_type": np.array([infer_model_type(cfg, model)]),
            "epoch": np.array([parse_epoch(checkpoint_path)], dtype=np.int32),
            "seed": np.array([cfg.seed], dtype=np.int32),
            "history_size": np.array([history_size], dtype=np.int32),
            "horizons": np.arange(1, rollout_mse_mean.size + 1, dtype=np.int32),
            "rollout_mse_mean": rollout_mse_mean.astype(np.float32),
            "rollout_mse_std": rollout_mse_std.astype(np.float32),
            "rollout_cos_mean": rollout_cos_mean.astype(np.float32),
            "rollout_cos_std": rollout_cos_std.astype(np.float32),
            "rollout_velocity_horizons": np.arange(2, rollout_velocity_mse_mean.size + 2, dtype=np.int32),
            "rollout_velocity_mse_mean": rollout_velocity_mse_mean.astype(np.float32),
            "rollout_velocity_mse_std": rollout_velocity_mse_std.astype(np.float32),
            "rollout_velocity_cos_mean": rollout_velocity_cos_mean.astype(np.float32),
            "rollout_velocity_cos_std": rollout_velocity_cos_std.astype(np.float32),
            "local_horizons": local_rollout_summary["local_horizons"],
            "rollout_mse_local": local_rollout_summary["rollout_mse_local"],
            "rollout_cos_local": local_rollout_summary["rollout_cos_local"],
            "rollout_mse_auc_local": local_rollout_summary["rollout_mse_auc_local"],
            "rollout_mse_mean_local": local_rollout_summary["rollout_mse_mean_local"],
            "rollout_cos_mean_local": local_rollout_summary["rollout_cos_mean_local"],
            "rollout_mse_growth_local": local_rollout_summary["rollout_mse_growth_local"],
            "rollout_cos_drop_local": local_rollout_summary["rollout_cos_drop_local"],
            "rollout_velocity_local_horizons": (
                local_rollout_velocity_summary["local_horizons"] + 1
                if local_rollout_velocity_summary["local_horizons"].size
                else local_rollout_velocity_summary["local_horizons"]
            ),
            "rollout_velocity_mse_local": local_rollout_velocity_summary["rollout_mse_local"],
            "rollout_velocity_cos_local": local_rollout_velocity_summary["rollout_cos_local"],
            "rollout_velocity_mse_auc_local": local_rollout_velocity_summary["rollout_mse_auc_local"],
            "rollout_velocity_mse_mean_local": local_rollout_velocity_summary["rollout_mse_mean_local"],
            "rollout_velocity_cos_mean_local": local_rollout_velocity_summary["rollout_cos_mean_local"],
            "rollout_velocity_mse_growth_local": local_rollout_velocity_summary["rollout_mse_growth_local"],
            "rollout_velocity_cos_drop_local": local_rollout_velocity_summary["rollout_cos_drop_local"],
            "delta_cos_consecutive": delta_cos_consecutive.astype(np.float32),
            "delta_cos_consecutive_mean": np.array(
                [delta_cos_consecutive.mean()] if delta_cos_consecutive.size else [np.nan],
                dtype=np.float32,
            ),
            "delta_cos_consecutive_std": np.array(
                [delta_cos_consecutive.std()] if delta_cos_consecutive.size else [np.nan],
                dtype=np.float32,
            ),
            "delta_cos_negative_ratio": np.array(
                [(delta_cos_consecutive < 0).mean()] if delta_cos_consecutive.size else [np.nan],
                dtype=np.float32,
            ),
            "delta_r2_score": np.array([delta_r2_score], dtype=np.float32),
            "inverse_loss_mean": np.array(
                [inverse_stats["inverse_loss_values"].mean()] if inverse_stats["inverse_loss_values"].size else [np.nan],
                dtype=np.float32,
            ),
            "inverse_cos_mean": np.array(
                [inverse_stats["inverse_cos_values"].mean()] if inverse_stats["inverse_cos_values"].size else [np.nan],
                dtype=np.float32,
            ),
            "inverse_to_pred_loss_ratio_mean": np.array(
                [inverse_stats["inverse_to_pred_loss_ratio_values"].mean()]
                if inverse_stats["inverse_to_pred_loss_ratio_values"].size
                else [np.nan],
                dtype=np.float32,
            ),
            "action_conditioned_margin": inverse_stats["action_conditioned_margin"].astype(np.float32),
            "action_conditioned_margin_mean": np.array(
                [inverse_stats["action_conditioned_margin"].mean()]
                if inverse_stats["action_conditioned_margin"].size
                else [np.nan],
                dtype=np.float32,
            ),
            "action_separation_pred_dist": action_separation_stats["action_separation_pred_dist"].astype(np.float32),
            "action_separation_act_dist": action_separation_stats["action_separation_act_dist"].astype(np.float32),
            "action_separation_gain": action_separation_stats["action_separation_gain"].astype(np.float32),
            "action_separation_corr": action_separation_stats["action_separation_corr"].astype(np.float32),
            "action_separation_pred_dist_mean": np.array(
                [action_separation_stats["action_separation_pred_dist"].mean()]
                if action_separation_stats["action_separation_pred_dist"].size
                else [np.nan],
                dtype=np.float32,
            ),
            "action_separation_act_dist_mean": np.array(
                [action_separation_stats["action_separation_act_dist"].mean()]
                if action_separation_stats["action_separation_act_dist"].size
                else [np.nan],
                dtype=np.float32,
            ),
            "action_separation_gain_mean": np.array(
                [action_separation_stats["action_separation_gain"].mean()]
                if action_separation_stats["action_separation_gain"].size
                else [np.nan],
                dtype=np.float32,
            ),
            "action_separation_corr_mean": np.array(
                [action_separation_stats["action_separation_corr"].mean()]
                if action_separation_stats["action_separation_corr"].size
                else [np.nan],
                dtype=np.float32,
            ),
        }
        payload.update({key: value.astype(np.float32) for key, value in teacher_stats.items()})
        payload.update(static_metrics)
        payload.update(planning_metrics)

        np.savez(output_path, **payload)
        summary_record = {
            "checkpoint_path": str(checkpoint_path),
            "epoch": parse_epoch(checkpoint_path),
            "model_type": infer_model_type(cfg, model),
            "pred_delta_norm_mean": float(payload["pred_delta_norms"].mean()) if payload["pred_delta_norms"].size else float("nan"),
            "target_delta_norm_mean": float(payload["target_delta_norms"].mean()) if payload["target_delta_norms"].size else float("nan"),
            "relative_delta_error_mean": float(payload["relative_delta_error_norms"].mean()) if payload["relative_delta_error_norms"].size else float("nan"),
            "pred_target_delta_cos_mean": float(payload["pred_target_delta_cos"].mean()) if payload["pred_target_delta_cos"].size else float("nan"),
            "delta_r2_score": float(payload["delta_r2_score"][0]),
            "inverse_loss_mean": float(payload["inverse_loss_mean"][0]),
            "inverse_cos_mean": float(payload["inverse_cos_mean"][0]),
            "action_conditioned_margin_mean": float(payload["action_conditioned_margin_mean"][0]),
            "action_separation_pred_dist_mean": float(payload["action_separation_pred_dist_mean"][0]),
            "action_separation_act_dist_mean": float(payload["action_separation_act_dist_mean"][0]),
            "action_separation_gain_mean": float(payload["action_separation_gain_mean"][0]),
            "action_separation_corr_mean": float(payload["action_separation_corr_mean"][0]),
            "rollout_mse_final": float(rollout_mse_mean[-1]) if rollout_mse_mean.size else float("nan"),
            "rollout_cos_final": float(rollout_cos_mean[-1]) if rollout_cos_mean.size else float("nan"),
            "rollout_velocity_mse_final": float(rollout_velocity_mse_mean[-1]) if rollout_velocity_mse_mean.size else float("nan"),
            "rollout_velocity_cos_final": float(rollout_velocity_cos_mean[-1]) if rollout_velocity_cos_mean.size else float("nan"),
            "rollout_mse_auc_local": float(payload["rollout_mse_auc_local"][0]),
            "rollout_mse_mean_local": float(payload["rollout_mse_mean_local"][0]),
            "rollout_cos_mean_local": float(payload["rollout_cos_mean_local"][0]),
            "rollout_mse_growth_local": float(payload["rollout_mse_growth_local"][0]),
            "rollout_cos_drop_local": float(payload["rollout_cos_drop_local"][0]),
            "rollout_velocity_mse_auc_local": float(payload["rollout_velocity_mse_auc_local"][0]),
            "rollout_velocity_mse_mean_local": float(payload["rollout_velocity_mse_mean_local"][0]),
            "rollout_velocity_cos_mean_local": float(payload["rollout_velocity_cos_mean_local"][0]),
            "rollout_velocity_mse_growth_local": float(payload["rollout_velocity_mse_growth_local"][0]),
            "rollout_velocity_cos_drop_local": float(payload["rollout_velocity_cos_drop_local"][0]),
            "delta_cos_negative_ratio": float(payload["delta_cos_negative_ratio"][0]),
            "planning_success_rate": float(planning_metrics["planning_success_rate"][0]) if planning_metrics else float("nan"),
        }
        if static_metrics:
            summary_record.update(
                {
                    "static_target_threshold": float(payload["static_target_threshold"][0]),
                    "static_sample_ratio": float(payload["static_sample_ratio"][0]),
                    "static_sample_count": int(payload["static_sample_count"][0]),
                    "static_pred_delta_norm_mean": float(payload["static_pred_delta_norm_mean"][0]),
                    "static_target_delta_norm_mean": float(payload["static_target_delta_norm_mean"][0]),
                    "static_relative_delta_error_mean": float(payload["static_relative_delta_error_mean"][0]),
                }
            )
        for horizon, mse_value, cos_value in zip(
            payload["local_horizons"],
            payload["rollout_mse_local"],
            payload["rollout_cos_local"],
        ):
            summary_record[f"rollout_mse_h{int(horizon)}"] = float(mse_value)
            summary_record[f"rollout_cos_h{int(horizon)}"] = float(cos_value)
        for horizon, mse_value, cos_value in zip(
            payload["rollout_velocity_local_horizons"],
            payload["rollout_velocity_mse_local"],
            payload["rollout_velocity_cos_local"],
        ):
            summary_record[f"rollout_velocity_mse_h{int(horizon)}"] = float(mse_value)
            summary_record[f"rollout_velocity_cos_h{int(horizon)}"] = float(cos_value)
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary_record, sort_keys=True) + "\n")

        checkpoint_elapsed = time.time() - checkpoint_start_time
        total_elapsed = time.time() - analysis_start_time
        avg_checkpoint_time = total_elapsed / max(checkpoint_idx, 1)
        total_eta = avg_checkpoint_time * max(total_checkpoints - checkpoint_idx, 0)
        print(
            f"Saved diagnostics to {output_path} "
            f"(checkpoint {format_duration(checkpoint_elapsed)}, "
            f"total elapsed {format_duration(total_elapsed)}, "
            f"remaining ETA {format_duration(total_eta)})"
        )


if __name__ == "__main__":
    run()
