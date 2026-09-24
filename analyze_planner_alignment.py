import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/lewm-matplotlib-{os.getuid()}")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


# MUJOCO_EGL_DEVICE_ID, if set, indexes EGL devices independently of CUDA visibility.


import gc
import json
import math
import re
import time
import warnings
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import hydra
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn import preprocessing
from torchvision import tv_tensors

from eval import get_dataset, get_episodes_length, img_transform


def suppress_gym_box_warnings():
    warning_patterns = [
        r"^WARN: Casting input x to numpy array\.$",
        r"^WARN: Box low's precision lowered by casting to float32, current low\.dtype=float64$",
        r"^WARN: Box high's precision lowered by casting to float32, current high\.dtype=float64$",
    ]
    for pattern in warning_patterns:
        warnings.filterwarnings(
            "ignore",
            message=pattern,
            category=UserWarning,
        )


suppress_gym_box_warnings()


class TracingCEMSolver:
    """Analysis-only CEM solver that records the optimization trace."""

    def __init__(
        self,
        model,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        store_full_samples: bool = False,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.store_full_samples = store_full_samples
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

    def configure(self, *, action_space, n_envs: int, config) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        # `model.get_cost()` expects each planning step to contain one flattened
        # action block, i.e. `action_block * base_action_dim`. `single_action_space`
        # is typically a plain Box(shape=(base_action_dim,)), so `shape[1:]`
        # collapses to 1 and underestimates the true action width.
        self._action_dim = int(np.asarray(action_space.low).reshape(-1).shape[0])
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def init_action_distrib(self, actions: torch.Tensor | None = None):
        sigma = self.var_scale * torch.ones([self.n_envs, self.horizon, self.action_dim], device=self.device)
        mean = torch.zeros([self.n_envs, 0, self.action_dim], device=self.device) if actions is None else actions
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            new_mean = torch.zeros([self.n_envs, remaining, self.action_dim], device=self.device)
            mean = torch.cat([mean, new_mean], dim=1)
        return mean, sigma

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        outputs = {
            "costs": [],
            "mean": [],
            "var": [],
            "trace": [],
        }

        mean, sigma = self.init_action_distrib(init_action)
        total_envs = self.n_envs

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_sigma = sigma[start_idx:end_idx]

            expanded_infos = {}
            for key, value in info_dict.items():
                value_batch = value[start_idx:end_idx]
                if torch.is_tensor(value_batch):
                    value_batch = value_batch.unsqueeze(1)
                    value_batch = value_batch.expand(current_bs, self.num_samples, *value_batch.shape[2:])
                elif isinstance(value_batch, np.ndarray):
                    value_batch = np.repeat(value_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[key] = value_batch

            final_batch_cost = None
            batch_trace = []
            for _ in range(self.n_steps):
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                )
                candidates = candidates * batch_sigma.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean

                costs = self.model.get_cost(expanded_infos.copy(), candidates)
                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]

                batch_mean = topk_candidates.mean(dim=1)
                batch_sigma = topk_candidates.std(dim=1)
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

                step_record = {
                    "topk_inds": topk_inds.detach().cpu(),
                    "topk_costs": topk_vals.detach().cpu(),
                    "topk_candidates": topk_candidates.detach().cpu(),
                    "mean": batch_mean.detach().cpu(),
                    "var": batch_sigma.detach().cpu(),
                }
                if self.store_full_samples:
                    step_record["candidates"] = candidates.detach().cpu()
                    step_record["costs"] = costs.detach().cpu()
                batch_trace.append(step_record)

            mean[start_idx:end_idx] = batch_mean
            sigma[start_idx:end_idx] = batch_sigma
            outputs["costs"].extend(final_batch_cost)
            outputs["trace"].extend(batch_trace)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [sigma.detach().cpu()]
        return outputs


def suppress_egl_teardown_noise():
    try:
        from mujoco import egl as mujoco_egl
    except Exception:
        mujoco_egl = None

    if mujoco_egl is not None:
        gl_context_cls = getattr(mujoco_egl, "GLContext", None)
        if gl_context_cls is not None and not getattr(gl_context_cls, "_safe_del_installed", False):
            original_del = getattr(gl_context_cls, "__del__", None)

            def _safe_gl_del(self):
                try:
                    if original_del is not None:
                        original_del(self)
                except Exception:
                    pass

            gl_context_cls.__del__ = _safe_gl_del
            gl_context_cls._safe_del_installed = True

    try:
        from mujoco.rendering.classic.renderer import Renderer
    except Exception:
        return

    if getattr(Renderer, "_safe_del_installed", False):
        return

    original_renderer_del = getattr(Renderer, "__del__", None)

    def _safe_renderer_del(self):
        try:
            if original_renderer_del is not None:
                original_renderer_del(self)
        except Exception:
            pass

    Renderer.__del__ = _safe_renderer_del
    Renderer._safe_del_installed = True


suppress_egl_teardown_noise()


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


def load_model_object(path: Path, device: str):
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    return model


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


def load_eval_config(config_name: str):
    base_dir = Path(__file__).parent / "config" / "eval"
    cfg_path = base_dir / f"{config_name}.yaml"
    cfg = OmegaConf.load(cfg_path)
    defaults = OmegaConf.to_container(cfg.get("defaults", []), resolve=False) or []
    for entry in defaults:
        if not isinstance(entry, dict):
            continue
        for group, value in entry.items():
            if group == "_self_" or value in (None, "null"):
                continue
            if group in cfg:
                continue
            group_path = base_dir / group / f"{value}.yaml"
            if group_path.exists():
                cfg = OmegaConf.merge(cfg, OmegaConf.create({group: OmegaConf.load(group_path)}))
    return cfg


def build_process_and_transform(eval_cfg: DictConfig, dataset):
    transform = {
        "pixels": img_transform(eval_cfg),
        "goal": img_transform(eval_cfg),
    }

    process = {}
    for col in eval_cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process, transform


def build_world_from_eval_config(eval_cfg: DictConfig, num_envs: int):
    world_kwargs = OmegaConf.to_container(eval_cfg.world, resolve=True)
    world_kwargs["num_envs"] = int(num_envs)
    world_kwargs["image_shape"] = (eval_cfg.eval.img_size, eval_cfg.eval.img_size)
    return swm.World(**world_kwargs)


def prepare_info_dict(info_dict: dict, process: dict, transform: dict):
    output = {}
    for key, value in info_dict.items():
        v = value
        is_numpy = isinstance(v, (np.ndarray, np.generic))

        if key in process:
            if not is_numpy:
                raise ValueError(f"Expected numpy array for key '{key}', got {type(v)}")
            shape = v.shape
            if len(shape) > 2:
                v = v.reshape(-1, *shape[2:])
            v = process[key].transform(v)
            v = v.reshape(shape)

        if key in transform:
            shape = None
            if is_numpy or torch.is_tensor(v):
                if v.ndim > 2:
                    shape = v.shape
                    v = v.reshape(-1, *shape[2:])
            if key.startswith("pixels") or key.startswith("goal"):
                if is_numpy:
                    v = np.transpose(v, (0, 3, 1, 2))
                else:
                    v = v.permute(0, 3, 1, 2)
            v = torch.stack([transform[key](tv_tensors.Image(x)) for x in v])
            if shape is not None:
                v = v.reshape(*shape[:2], *v.shape[1:])
            is_numpy = False

        if is_numpy and getattr(v, "dtype", None) is not None and v.dtype.kind not in "USO":
            v = torch.from_numpy(v)

        output[key] = v
    return output


def expand_for_candidates(info_dict: dict, num_candidates: int, device: str):
    expanded = {}
    for key, value in info_dict.items():
        if torch.is_tensor(value):
            value = value.to(device)
            expanded[key] = value.unsqueeze(1).expand(value.size(0), num_candidates, *value.shape[1:])
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(value[:, None, ...], num_candidates, axis=1)
        else:
            expanded[key] = value
    return expanded


def rankdata(values: np.ndarray):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray):
    if x.size < 2 or y.size < 2:
        return np.nan
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom <= 0:
        return np.nan
    return float((rx * ry).sum() / denom)


def sample_eval_rows(eval_cfg: DictConfig, dataset, seed: int, override_num_eval=None):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - eval_cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    num_eval = min(int(override_num_eval or eval_cfg.eval.num_eval), len(valid_indices))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(valid_indices), size=num_eval, replace=False)
    selected = np.sort(valid_indices[selected])
    rows = dataset.get_row_data(selected)
    return rows[col_name], rows["step_idx"]


def build_eval_case(dataset, episode_idx: int, start_step: int, goal_offset_steps: int):
    data = dataset.load_chunk(
        np.array([episode_idx]),
        np.array([start_step]),
        np.array([start_step + goal_offset_steps]),
    )[0]
    columns = dataset.column_names

    init_step = {}
    goal_step = {}
    variation_values = {}

    for col in columns:
        if col.startswith("goal"):
            continue
        col_data = data[col]
        if col.startswith("pixels"):
            col_data = col_data.permute(0, 2, 3, 1)
        if not isinstance(col_data, (torch.Tensor, np.ndarray)):
            continue

        init_value = col_data[0]
        goal_value = col_data[-1]
        if isinstance(init_value, torch.Tensor):
            init_value = init_value.numpy()
        if isinstance(goal_value, torch.Tensor):
            goal_value = goal_value.numpy()

        init_step[col] = init_value
        goal_key = "goal" if col == "pixels" else f"goal_{col}"
        goal_step[goal_key] = goal_value

        if col.startswith("variation."):
            variation_values[col.removeprefix("variation.")] = init_value

    return {
        "init_step": init_step,
        "goal_step": goal_step,
        "variation_values": variation_values,
        "seed": init_step.get("seed", None),
    }


def reset_world_to_case(world, case: dict, callables):
    options = [{} for _ in range(world.num_envs)]
    if case["variation_values"]:
        keys = list(case["variation_values"].keys())
        for i in range(world.num_envs):
            options[i]["variation"] = keys
            options[i]["variation_values"] = {
                key: deepcopy(value) for key, value in case["variation_values"].items()
            }

    seeds = None
    if case["seed"] is not None:
        seeds = [int(case["seed"])] * world.num_envs

    world.reset(seed=seeds, options=options)

    for env in world.envs.unwrapped.envs:
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                continue
            method = getattr(env_unwrapped, method_name)
            args = spec.get("args", spec)
            prepared_args = {}
            for arg_name, arg_cfg in args.items():
                if arg_name == "method":
                    continue
                value_key = arg_cfg.get("value")
                in_dataset = arg_cfg.get("in_dataset", True)
                if in_dataset:
                    if value_key in case["init_step"]:
                        prepared_args[arg_name] = deepcopy(case["init_step"][value_key])
                    elif value_key in case["goal_step"]:
                        prepared_args[arg_name] = deepcopy(case["goal_step"][value_key])
                    else:
                        continue
                else:
                    prepared_args[arg_name] = deepcopy(value_key)
            method(**prepared_args)

    shape_prefix = world.infos["pixels"].shape[:2]
    init_broadcast = {
        key: np.broadcast_to(value[None, None, ...], shape_prefix + value.shape)
        for key, value in case["init_step"].items()
    }
    goal_broadcast = {
        key: np.broadcast_to(value[None, None, ...], shape_prefix + value.shape)
        for key, value in case["goal_step"].items()
    }
    world.infos.update(deepcopy(init_broadcast))
    world.infos.update(deepcopy(goal_broadcast))
    world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))
    return goal_broadcast


def step_world_with_blocks(world, action_blocks: np.ndarray, goal_broadcast: dict):
    horizon = action_blocks.shape[1]
    action_block = action_blocks.shape[2]
    for plan_step in range(horizon):
        for block_idx in range(action_block):
            actions = action_blocks[:, plan_step, block_idx]
            (
                world.states,
                world.rewards,
                world.terminateds,
                world.truncateds,
                world.infos,
            ) = world.envs.step(actions)
            world.infos.update(deepcopy(goal_broadcast))
            world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))


def compute_oracle_goal_costs(world):
    costs = []
    for env_idx in range(world.num_envs):
        current = world.infos["observation"][env_idx]
        target = world.infos["target"][env_idx]
        if current.ndim > 1:
            current = current[-1]
        if target.ndim > 1:
            target = target[-1]
        costs.append(float(np.mean((current - target) ** 2)))
    return np.asarray(costs, dtype=np.float32)


def compute_model_costs(model, base_info: dict, action_candidates: np.ndarray, process: dict, transform: dict, device: str):
    prepared = prepare_info_dict(base_info, process=process, transform=transform)
    expanded = expand_for_candidates(prepared, num_candidates=action_candidates.shape[0], device=device)
    candidates = torch.from_numpy(action_candidates).unsqueeze(0).to(device)
    with torch.inference_mode():
        costs = model.get_cost(expanded, candidates)
    return costs.squeeze(0).detach().cpu().numpy().astype(np.float32)


def build_base_info(world):
    return {key: deepcopy(value[:1]) for key, value in world.infos.items() if isinstance(value, np.ndarray)}


def move_info_to_device(info_dict: dict, device: str):
    output = {}
    for key, value in info_dict.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def build_plan_namespace(plan_cfg: DictConfig, horizon_override: int | None = None):
    payload = dict(OmegaConf.to_container(plan_cfg, resolve=True))
    if horizon_override is not None:
        payload["horizon"] = int(horizon_override)
    return SimpleNamespace(**payload)


def get_solver_kwargs(eval_cfg: DictConfig):
    solver_kwargs = dict(OmegaConf.to_container(eval_cfg.solver, resolve=True))
    solver_kwargs.pop("_target_", None)
    solver_kwargs.pop("model", None)
    return solver_kwargs


def get_structured_eval_path(run_dir: Path, checkpoint_path: Path):
    policy_stem = checkpoint_path.stem.removesuffix("_object")
    return run_dir / f"ogb_cube_results__{run_dir.name}_{policy_stem}.npz"


def load_structured_eval(run_dir: Path, checkpoint_path: Path):
    path = get_structured_eval_path(run_dir, checkpoint_path)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "path": path,
        "eval_episodes": data["eval_episodes"].astype(np.int64),
        "eval_start_idx": data["eval_start_idx"].astype(np.int64),
        "episode_successes": data["episode_successes"].astype(bool),
    }


def find_matching_checkpoint(run_dir: Path, epoch: int):
    matches = sorted(run_dir.glob(f"*_epoch_{epoch}_object.ckpt"), key=lambda path: path.name)
    return matches[0] if matches else None


def select_landscape_cases(cfg: DictConfig, current_eval: dict | None, compare_eval: dict | None):
    target_count = (
        int(cfg.analysis.landscape.representative.current_only)
        + int(cfg.analysis.landscape.representative.compare_only)
        + int(cfg.analysis.landscape.representative.both_success)
        + int(cfg.analysis.landscape.representative.both_fail)
    )
    if current_eval is None:
        return []

    current_eps = current_eval["eval_episodes"]
    current_starts = current_eval["eval_start_idx"]
    current_success = current_eval["episode_successes"]

    if compare_eval is None:
        return [
            {
                "episode_idx": int(ep),
                "start_step": int(start),
                "category": "current_eval",
            }
            for ep, start in zip(current_eps[:target_count], current_starts[:target_count])
        ]

    if not (
        np.array_equal(current_eps, compare_eval["eval_episodes"])
        and np.array_equal(current_starts, compare_eval["eval_start_idx"])
    ):
        raise ValueError("Current and comparison structured eval files must share eval_episodes/eval_start_idx.")

    compare_success = compare_eval["episode_successes"]
    masks = {
        "current_only": current_success & ~compare_success,
        "compare_only": ~current_success & compare_success,
        "both_success": current_success & compare_success,
        "both_fail": ~current_success & ~compare_success,
    }
    quotas = {
        "current_only": int(cfg.analysis.landscape.representative.current_only),
        "compare_only": int(cfg.analysis.landscape.representative.compare_only),
        "both_success": int(cfg.analysis.landscape.representative.both_success),
        "both_fail": int(cfg.analysis.landscape.representative.both_fail),
    }

    selected = []
    used = np.zeros_like(current_success, dtype=bool)
    for category in ["current_only", "compare_only", "both_success", "both_fail"]:
        idxs = np.where(masks[category])[0][: quotas[category]]
        for idx in idxs:
            selected.append(
                {
                    "episode_idx": int(current_eps[idx]),
                    "start_step": int(current_starts[idx]),
                    "category": category,
                }
            )
            used[idx] = True

    if len(selected) < target_count:
        remaining = np.where(~used)[0][: target_count - len(selected)]
        for idx in remaining:
            selected.append(
                {
                    "episode_idx": int(current_eps[idx]),
                    "start_step": int(current_starts[idx]),
                    "category": "fallback",
                }
            )
    return selected


def flatten_plan(plan: np.ndarray):
    return plan.reshape(-1)


def unflatten_action_candidates(flat_candidates: np.ndarray, horizon: int, action_block: int, action_dim: int):
    return flat_candidates.reshape(flat_candidates.shape[0], horizon, action_block, action_dim).astype(np.float32)


def clip_flat_candidates(flat_candidates: np.ndarray, low: np.ndarray, high: np.ndarray):
    return np.clip(flat_candidates, low[None, :], high[None, :])


def get_flat_action_bounds(action_space, horizon: int, action_block: int):
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    low = np.tile(low, horizon * action_block)
    high = np.tile(high, horizon * action_block)
    return low, high


def project_std_along_axis(std_flat: np.ndarray, axis: np.ndarray):
    return float(np.sqrt(np.sum((std_flat ** 2) * (axis ** 2))))


def compute_landscape_axes(elite_flat: np.ndarray, std_flat: np.ndarray, eps: float):
    centered = elite_flat - elite_flat.mean(axis=0, keepdims=True)
    if centered.shape[0] >= 2 and np.linalg.norm(centered) > eps:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_a = vh[0]
        if vh.shape[0] > 1 and np.linalg.norm(vh[1]) > eps:
            axis_b = vh[1]
        else:
            axis_b = None
    else:
        axis_a = None
        axis_b = None

    if axis_a is None or np.linalg.norm(axis_a) <= eps:
        order = np.argsort(-(std_flat ** 2))
        axis_a = np.zeros_like(std_flat)
        axis_a[order[0]] = 1.0
    axis_a = axis_a / max(np.linalg.norm(axis_a), eps)

    if axis_b is None or np.linalg.norm(axis_b) <= eps:
        order = np.argsort(-(std_flat ** 2))
        axis_b = np.zeros_like(std_flat)
        fallback_idx = order[1] if order.size > 1 else order[0]
        axis_b[fallback_idx] = 1.0
        axis_b = axis_b - axis_b.dot(axis_a) * axis_a
    axis_b = axis_b / max(np.linalg.norm(axis_b), eps)
    return axis_a.astype(np.float32), axis_b.astype(np.float32)


def resolve_scan_steps(horizon: int, scan_step: int, scan_num_steps: int):
    scan_num_steps = max(1, int(scan_num_steps))
    if scan_step < 0:
        end = horizon + int(scan_step) + 1
    else:
        end = int(scan_step) + 1
    end = min(max(end, 1), horizon)
    start = max(end - scan_num_steps, 0)
    return list(range(start, end))


def compute_step_action_axes(plan_shape: tuple[int, ...], dims: list[int], scan_step: int, scan_num_steps: int, eps: float):
    horizon, flat_action_dim = int(plan_shape[0]), int(plan_shape[1])
    if horizon < 1:
        raise ValueError(f"Expected non-empty plan horizon, got shape={plan_shape}")
    if len(dims) != 2:
        raise ValueError(f"landscape.scan_dims must contain exactly two dims, got {dims}")
    axis_a = np.zeros(horizon * flat_action_dim, dtype=np.float32)
    axis_b = np.zeros_like(axis_a)
    dim_a, dim_b = int(dims[0]), int(dims[1])
    if not (0 <= dim_a < flat_action_dim and 0 <= dim_b < flat_action_dim):
        raise ValueError(
            f"scan_dims={dims} out of range for flattened action dim {flat_action_dim}"
        )
    if dim_a == dim_b:
        raise ValueError(f"scan_dims must be different, got {dims}")
    for step_idx in resolve_scan_steps(horizon, scan_step=scan_step, scan_num_steps=scan_num_steps):
        base = step_idx * flat_action_dim
        axis_a[base + dim_a] = 1.0
        axis_b[base + dim_b] = 1.0
    axis_a = axis_a / max(np.linalg.norm(axis_a), eps)
    axis_b = axis_b / max(np.linalg.norm(axis_b), eps)
    return axis_a, axis_b


def compute_suffix_elite_pca_axes(final_plan: np.ndarray, final_elites: np.ndarray, scan_step: int, scan_num_steps: int, eps: float):
    horizon, flat_action_dim = int(final_plan.shape[0]), int(final_plan.shape[1])
    scan_steps = resolve_scan_steps(horizon, scan_step=scan_step, scan_num_steps=scan_num_steps)
    suffix_elites = final_elites[:, scan_steps, :].reshape(final_elites.shape[0], -1)
    centered = suffix_elites - suffix_elites.mean(axis=0, keepdims=True)
    if centered.shape[0] >= 2 and np.linalg.norm(centered) > eps:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        local_a = vh[0]
        local_b = vh[1] if vh.shape[0] > 1 and np.linalg.norm(vh[1]) > eps else None
    else:
        local_a = None
        local_b = None
    if local_a is None or np.linalg.norm(local_a) <= eps:
        local_a = np.zeros(len(scan_steps) * flat_action_dim, dtype=np.float32)
        local_a[0] = 1.0
    if local_b is None or np.linalg.norm(local_b) <= eps:
        local_b = np.zeros(len(scan_steps) * flat_action_dim, dtype=np.float32)
        local_b[min(1, local_b.size - 1)] = 1.0
        local_b = local_b - local_b.dot(local_a) * local_a

    axis_a = np.zeros(horizon * flat_action_dim, dtype=np.float32)
    axis_b = np.zeros_like(axis_a)
    local_a = local_a / max(np.linalg.norm(local_a), eps)
    local_b = local_b / max(np.linalg.norm(local_b), eps)
    local_a = local_a.reshape(len(scan_steps), flat_action_dim)
    local_b = local_b.reshape(len(scan_steps), flat_action_dim)
    for local_idx, step_idx in enumerate(scan_steps):
        start = step_idx * flat_action_dim
        axis_a[start : start + flat_action_dim] = local_a[local_idx]
        axis_b[start : start + flat_action_dim] = local_b[local_idx]
    axis_a = axis_a / max(np.linalg.norm(axis_a), eps)
    axis_b = axis_b / max(np.linalg.norm(axis_b), eps)
    return axis_a, axis_b


def compute_landscape_axes_for_cfg(final_plan: np.ndarray, final_elites: np.ndarray, std_flat: np.ndarray, cfg: DictConfig):
    axis_mode = str(cfg.analysis.landscape.get("axis_mode", "elite_pca"))
    if axis_mode in {"elite_pca", "reference_elite_pca"}:
        elite_flat = final_elites.reshape(final_elites.shape[0], -1)
        return compute_landscape_axes(elite_flat, std_flat, eps=cfg.analysis.eps)
    scan_step = int(cfg.analysis.landscape.get("scan_step", -1))
    scan_num_steps = int(cfg.analysis.landscape.get("scan_num_steps", 1))
    if axis_mode in {"last_action", "step_action"}:
        dims = list(cfg.analysis.landscape.get("scan_dims", cfg.analysis.landscape.get("last_action_dims", [0, 1])))
        return compute_step_action_axes(
            final_plan.shape,
            dims=dims,
            scan_step=scan_step,
            scan_num_steps=scan_num_steps,
            eps=cfg.analysis.eps,
        )
    if axis_mode == "suffix_elite_pca":
        return compute_suffix_elite_pca_axes(
            final_plan=final_plan,
            final_elites=final_elites,
            scan_step=scan_step,
            scan_num_steps=scan_num_steps,
            eps=cfg.analysis.eps,
        )
    raise ValueError(f"Unknown analysis.landscape.axis_mode={axis_mode!r}")


def load_reference_landscape_geometry(cfg: DictConfig):
    reference_npz = cfg.analysis.landscape.get("reference_npz", None)
    if reference_npz in (None, "", "null"):
        return None
    path = Path(str(reference_npz)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"analysis.landscape.reference_npz does not exist: {path}")

    data = np.load(path, allow_pickle=True)
    case_idx = int(cfg.analysis.landscape.get("reference_case_idx", 0))
    mode = str(cfg.analysis.landscape.get("reference_mode", "full"))
    prefix = "one_step_landscape" if mode in {"one_step", "one"} else "full_landscape"
    required = {
        "center": f"{prefix}_center_actions",
        "std": f"{prefix}_center_std",
        "elite": f"{prefix}_elite_trace",
    }
    missing = [key for key in required.values() if key not in data]
    if missing:
        raise KeyError(f"Reference landscape file {path} is missing keys: {missing}")

    center_actions = np.asarray(data[required["center"]][case_idx], dtype=np.float32)
    center_std = np.asarray(data[required["std"]][case_idx], dtype=np.float32)
    elite_trace = np.asarray(data[required["elite"]][case_idx], dtype=np.float32)
    final_elites = elite_trace[-1]

    model_grid_key = f"{prefix}_model_grids"
    axis_a_key = f"{prefix}_axis_a"
    axis_b_key = f"{prefix}_axis_b"
    grid_best_key = f"{prefix}_grid_best_actions"
    grid_best_action = None
    reference_axis_a = None
    reference_axis_b = None
    if grid_best_key in data:
        grid_best_action = np.asarray(data[grid_best_key][case_idx], dtype=np.float32)
    if axis_a_key in data and axis_b_key in data:
        reference_axis_a = np.asarray(data[axis_a_key][case_idx], dtype=np.float32)
        reference_axis_b = np.asarray(data[axis_b_key][case_idx], dtype=np.float32)
    if (
        grid_best_action is None
        and model_grid_key in data
        and reference_axis_a is not None
        and reference_axis_b is not None
        and "landscape_alphas" in data
        and "landscape_betas" in data
    ):
        model_grid = np.asarray(data[model_grid_key][case_idx], dtype=np.float32)
        alphas = np.asarray(data["landscape_alphas"], dtype=np.float32)
        betas = np.asarray(data["landscape_betas"], dtype=np.float32)
        best_row, best_col = np.unravel_index(int(np.argmin(model_grid)), model_grid.shape)
        center_flat = flatten_plan(center_actions)
        grid_best_flat = center_flat + float(alphas[best_col]) * reference_axis_a + float(betas[best_row]) * reference_axis_b
        grid_best_action = grid_best_flat.reshape(center_actions.shape).astype(np.float32)

    best_elite_idx = int(cfg.analysis.landscape.get("reference_best_elite_idx", 0))
    if not (0 <= best_elite_idx < final_elites.shape[0]):
        raise IndexError(
            f"reference_best_elite_idx={best_elite_idx} out of range for "
            f"{final_elites.shape[0]} final elites"
        )
    return {
        "path": path,
        "case_idx": case_idx,
        "mode": mode,
        "center_actions": center_actions,
        "center_std": center_std,
        "final_elites": final_elites,
        "best_elite_action": final_elites[best_elite_idx],
        "best_elite_idx": best_elite_idx,
        "grid_best_action": grid_best_action,
        "axis_a": reference_axis_a,
        "axis_b": reference_axis_b,
    }


def project_to_axes(points_flat: np.ndarray, center_flat: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray):
    centered = points_flat - center_flat[None, :]
    return np.stack([centered @ axis_a, centered @ axis_b], axis=-1).astype(np.float32)


def evaluate_env_costs_batched(world, case: dict, callables, action_blocks: np.ndarray):
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
        costs.append(compute_oracle_goal_costs(world)[:actual])
    return np.concatenate(costs, axis=0).astype(np.float32)


def evaluate_env_costs(world, case: dict, callables, action_blocks: np.ndarray, eval_cfg: DictConfig):
    try:
        return evaluate_env_costs_batched(world, case, callables, action_blocks)
    except Exception as exc:
        if world.num_envs <= 1 or "Offscreen framebuffer is not complete" not in str(exc):
            raise
        print(
            f"  framebuffer allocation failed with env_batch_size={world.num_envs}; "
            "falling back to serial env-cost evaluation"
        )
        fallback_world = build_world_from_eval_config(eval_cfg, num_envs=1)
        try:
            return evaluate_env_costs_batched(fallback_world, case, callables, action_blocks)
        finally:
            del fallback_world
            gc.collect()


def build_landscape_grid(center_flat: np.ndarray, std_flat: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray, cfg: DictConfig, low: np.ndarray, high: np.ndarray):
    fixed_radius = cfg.analysis.landscape.get("fixed_radius", None)
    if fixed_radius is None:
        radius_a = cfg.analysis.landscape.radius_scale * max(project_std_along_axis(std_flat, axis_a), cfg.analysis.eps)
        radius_b = cfg.analysis.landscape.radius_scale * max(project_std_along_axis(std_flat, axis_b), cfg.analysis.eps)
    else:
        radius_a = float(fixed_radius)
        radius_b = float(fixed_radius)
    alphas = np.linspace(-radius_a, radius_a, int(cfg.analysis.landscape.grid_points), dtype=np.float32)
    betas = np.linspace(-radius_b, radius_b, int(cfg.analysis.landscape.grid_points), dtype=np.float32)
    grid_points = []
    for beta in betas:
        for alpha in alphas:
            grid_points.append(center_flat + alpha * axis_a + beta * axis_b)
    flat_candidates = np.stack(grid_points).astype(np.float32)
    flat_candidates = clip_flat_candidates(flat_candidates, low=low, high=high)
    return alphas, betas, flat_candidates


def compute_total_variation(grid: np.ndarray):
    if grid.size == 0:
        return float("nan")
    diff_x = np.abs(np.diff(grid, axis=1)).mean() if grid.shape[1] > 1 else 0.0
    diff_y = np.abs(np.diff(grid, axis=0)).mean() if grid.shape[0] > 1 else 0.0
    return float(diff_x + diff_y)


def compute_landscape_metrics(model_grid: np.ndarray, env_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray, mean_proj: np.ndarray, elite_proj: np.ndarray):
    center_row = len(betas) // 2
    center_col = len(alphas) // 2
    center_model = float(model_grid[center_row, center_col])
    center_env = float(env_grid[center_row, center_col])

    def _grid_metrics(grid: np.ndarray):
        best_idx = int(np.argmin(grid))
        best_row, best_col = np.unravel_index(best_idx, grid.shape)
        row_lo = max(center_row - 1, 0)
        row_hi = min(center_row + 2, grid.shape[0])
        col_lo = max(center_col - 1, 0)
        col_hi = min(center_col + 2, grid.shape[1])
        neighbor = grid[row_lo:row_hi, col_lo:col_hi]
        return {
            "center_to_best_gap": float(grid[center_row, center_col] - grid.min()),
            "local_min_offset": float(math.sqrt((alphas[best_col] ** 2) + (betas[best_row] ** 2))),
            "neighbor_std": float(neighbor.std()),
            "total_variation": compute_total_variation(grid),
        }

    model_metrics = _grid_metrics(model_grid)
    env_metrics = _grid_metrics(env_grid)
    elite_spread = np.sqrt((elite_proj ** 2).sum(axis=-1)).mean(axis=-1) if elite_proj.size else np.empty((0,), dtype=np.float32)
    mean_step_distance = (
        np.sqrt(((mean_proj[1:] - mean_proj[:-1]) ** 2).sum(axis=-1)).mean()
        if mean_proj.shape[0] > 1
        else float("nan")
    )
    return {
        "cem_model_center_to_best_gap": model_metrics["center_to_best_gap"],
        "cem_env_center_to_best_gap": env_metrics["center_to_best_gap"],
        "cem_model_local_min_offset": model_metrics["local_min_offset"],
        "cem_env_local_min_offset": env_metrics["local_min_offset"],
        "cem_model_neighbor_std": model_metrics["neighbor_std"],
        "cem_env_neighbor_std": env_metrics["neighbor_std"],
        "cem_model_total_variation": model_metrics["total_variation"],
        "cem_env_total_variation": env_metrics["total_variation"],
        "cem_landscape_rank_spearman": spearman_corr(model_grid.reshape(-1), env_grid.reshape(-1)),
        "cem_terminal_model_env_gap": float(abs(center_model - center_env)),
        "cem_trace_mean_step_distance": float(mean_step_distance),
        "cem_trace_elite_spread_final": float(elite_spread[-1]) if elite_spread.size else float("nan"),
        "cem_trace_elite_spread_decay": float(elite_spread[0] - elite_spread[-1]) if elite_spread.size else float("nan"),
    }


def render_landscape_plot(output_path: Path, title: str, alphas: np.ndarray, betas: np.ndarray, model_grid: np.ndarray, env_grid: np.ndarray, mean_proj: np.ndarray, elite_proj: np.ndarray):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    extent = [float(alphas[0]), float(alphas[-1]), float(betas[0]), float(betas[-1])]
    diff_grid = model_grid - env_grid

    panels = [
        (axes[0], model_grid, "Model Cost"),
        (axes[1], env_grid, "Env Cost"),
        (axes[2], diff_grid, "Model - Env"),
    ]
    for axis, grid, subtitle in panels:
        image = axis.imshow(grid, origin="lower", extent=extent, aspect="auto")
        axis.scatter([0.0], [0.0], c="white", s=40, marker="x", label="CEM Final")
        best_row, best_col = np.unravel_index(np.argmin(grid), grid.shape)
        axis.scatter([alphas[best_col]], [betas[best_row]], c="red", s=35, marker="o", label="Grid Best")
        axis.set_title(subtitle)
        axis.set_xlabel("Axis 1")
        axis.set_ylabel("Axis 2")
        fig.colorbar(image, ax=axis, shrink=0.8)

    axes[3].imshow(model_grid, origin="lower", extent=extent, aspect="auto")
    if elite_proj.size:
        colors = plt.cm.viridis(np.linspace(0.1, 0.95, elite_proj.shape[0]))
        for step_idx in range(elite_proj.shape[0]):
            axes[3].scatter(
                elite_proj[step_idx, :, 0],
                elite_proj[step_idx, :, 1],
                s=10,
                alpha=0.25 + 0.6 * (step_idx + 1) / elite_proj.shape[0],
                color=colors[step_idx],
            )
    if mean_proj.size:
        axes[3].plot(mean_proj[:, 0], mean_proj[:, 1], color="white", linewidth=2.0, marker="o", markersize=3)
    axes[3].scatter([0.0], [0.0], c="red", s=45, marker="x")
    axes[3].set_title("CEM Elite / Mean Trace")
    axes[3].set_xlabel("Axis 1")
    axes[3].set_ylabel("Axis 2")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_cem_landscape_mode(model, cem_world, env_world, case: dict, process: dict, transform: dict, device: str, cfg: DictConfig, eval_cfg: DictConfig, horizon: int):
    reset_world_to_case(cem_world, case, eval_cfg.eval.callables)
    base_info = build_base_info(cem_world)
    prepared = move_info_to_device(prepare_info_dict(base_info, process=process, transform=transform), device)

    solver = TracingCEMSolver(
        model=model,
        store_full_samples=bool(cfg.analysis.landscape.store_full_samples),
        **get_solver_kwargs(eval_cfg),
    )
    solver.configure(
        action_space=cem_world.single_action_space,
        n_envs=1,
        config=build_plan_namespace(eval_cfg.plan_config, horizon_override=horizon),
    )
    cem_output = solver.solve(prepared)

    final_plan = cem_output["actions"][0].numpy().astype(np.float32)
    final_std = cem_output["var"][-1][0].numpy().astype(np.float32)
    final_elites = cem_output["trace"][-1]["topk_candidates"][0].numpy().astype(np.float32)

    reference = load_reference_landscape_geometry(cfg)
    center_mode = str(cfg.analysis.landscape.get("center_mode", "cem_final"))
    if center_mode == "cem_final":
        center_plan = final_plan
        std_plan = final_std
    elif reference is None:
        raise ValueError(f"center_mode={center_mode!r} requires analysis.landscape.reference_npz")
    elif center_mode == "reference":
        center_plan = reference["center_actions"]
        std_plan = reference["center_std"]
    elif center_mode == "reference_best_elite":
        center_plan = reference["best_elite_action"]
        std_plan = reference["center_std"]
    elif center_mode == "reference_grid_best":
        if reference["grid_best_action"] is None:
            raise KeyError(
                "center_mode='reference_grid_best' requires the reference npz to contain "
                "model_grids, landscape_alphas/betas, and saved axis_a/axis_b. "
                "Run a first landscape pass with the current code, then use that npz as reference."
            )
        center_plan = reference["grid_best_action"]
        std_plan = reference["center_std"]
    else:
        raise ValueError(f"Unknown analysis.landscape.center_mode={center_mode!r}")

    axis_mode = str(cfg.analysis.landscape.get("axis_mode", "elite_pca"))
    if axis_mode in {"reference_saved_axes", "reference_grid_axes"} and reference is not None:
        if reference["axis_a"] is None or reference["axis_b"] is None:
            raise KeyError(
                f"axis_mode={axis_mode!r} requires saved axis_a/axis_b in the reference npz."
            )
        axis_plan = center_plan
        axis_elites = final_elites
        axis_std = std_plan
        axis_a = reference["axis_a"]
        axis_b = reference["axis_b"]
    elif axis_mode.startswith("reference_") and reference is not None:
        axis_plan = reference["center_actions"]
        axis_elites = reference["final_elites"]
        axis_std = reference["center_std"]
    else:
        axis_plan = center_plan
        axis_elites = final_elites
        axis_std = std_plan

    center_flat = flatten_plan(center_plan)
    std_flat = flatten_plan(axis_std)
    if "axis_a" not in locals() or "axis_b" not in locals():
        axis_a, axis_b = compute_landscape_axes_for_cfg(
            final_plan=axis_plan,
            final_elites=axis_elites,
            std_flat=std_flat,
            cfg=cfg,
        )

    action_block = int(eval_cfg.plan_config.action_block)
    action_dim = int(np.asarray(cem_world.single_action_space.low).reshape(-1).shape[0])
    low, high = get_flat_action_bounds(cem_world.single_action_space, horizon=horizon, action_block=action_block)
    alphas, betas, flat_candidates = build_landscape_grid(
        center_flat=center_flat,
        std_flat=std_flat,
        axis_a=axis_a,
        axis_b=axis_b,
        cfg=cfg,
        low=low,
        high=high,
    )
    action_candidates = flat_candidates.reshape(flat_candidates.shape[0], horizon, action_block * action_dim)
    model_costs = compute_model_costs(
        model,
        base_info=base_info,
        action_candidates=action_candidates,
        process=process,
        transform=transform,
        device=device,
    )
    grid_size = len(alphas)
    model_grid = model_costs.reshape(len(betas), grid_size).astype(np.float32)
    grid_best_flat = flat_candidates[int(np.argmin(model_costs))]
    grid_best_action = grid_best_flat.reshape(horizon, action_block * action_dim).astype(np.float32)
    if bool(cfg.analysis.landscape.get("eval_env_costs", True)):
        action_blocks = unflatten_action_candidates(flat_candidates, horizon=horizon, action_block=action_block, action_dim=action_dim)
        env_costs = evaluate_env_costs(env_world, case, eval_cfg.eval.callables, action_blocks, eval_cfg=eval_cfg)
        env_grid = env_costs.reshape(len(betas), grid_size).astype(np.float32)
    else:
        env_grid = np.full_like(model_grid, np.nan, dtype=np.float32)

    mean_trace = np.stack([step["mean"][0].numpy().astype(np.float32) for step in cem_output["trace"]], axis=0)
    elite_trace = np.stack([step["topk_candidates"][0].numpy().astype(np.float32) for step in cem_output["trace"]], axis=0)
    var_trace = np.stack([step["var"][0].numpy().astype(np.float32) for step in cem_output["trace"]], axis=0)
    mean_proj = project_to_axes(mean_trace.reshape(mean_trace.shape[0], -1), center_flat, axis_a, axis_b)
    elite_proj = project_to_axes(
        elite_trace.reshape(elite_trace.shape[0] * elite_trace.shape[1], -1),
        center_flat,
        axis_a,
        axis_b,
    ).reshape(elite_trace.shape[0], elite_trace.shape[1], 2)

    metrics = compute_landscape_metrics(model_grid, env_grid, alphas, betas, mean_proj, elite_proj)
    return {
        "model_grid": model_grid,
        "env_grid": env_grid,
        "alphas": alphas,
        "betas": betas,
        "mean_proj": mean_proj,
        "elite_proj": elite_proj,
        "center_action": final_plan,
        "landscape_center_action": center_plan,
        "center_std": final_std,
        "landscape_center_std": std_plan,
        "landscape_axis_a": axis_a,
        "landscape_axis_b": axis_b,
        "landscape_grid_best_action": grid_best_action,
        "mean_trace": mean_trace,
        "elite_trace": elite_trace,
        "var_trace": var_trace,
        "metrics": metrics,
    }


def output_path_for_checkpoint(run_dir: Path, output_subdir: str, checkpoint_path: Path):
    epoch = parse_epoch(checkpoint_path)
    return run_dir / output_subdir / f"{checkpoint_path.stem}_planner_alignment_epoch_{epoch:03d}.npz"


@hydra.main(version_base=None, config_path="./config/planner_alignment", config_name="cube")
def run(cfg: DictConfig):
    eval_cfg = load_eval_config(cfg.eval_config_name)
    with open_dict(eval_cfg):
        eval_cfg.cache_dir = str(get_cache_dir(cfg))
        eval_cfg.world.max_episode_steps = 2 * int(eval_cfg.eval.eval_budget)
        eval_cfg.solver.device = str(cfg.analysis.device)

    run_dir, checkpoint_paths = collect_checkpoint_paths(cfg)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found under {run_dir} matching {cfg.checkpoints.pattern}")

    dataset = get_dataset(eval_cfg, eval_cfg.eval.dataset_name)
    process, transform = build_process_and_transform(eval_cfg, dataset)
    eval_episodes, eval_start_idx = sample_eval_rows(
        eval_cfg,
        dataset,
        seed=cfg.seed,
        override_num_eval=cfg.analysis.num_eval,
    )

    output_dir = run_dir / cfg.analysis.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"

    landscape_cem_world = build_world_from_eval_config(eval_cfg, num_envs=1)
    landscape_env_world = build_world_from_eval_config(
        eval_cfg, num_envs=int(cfg.analysis.landscape.env_batch_size)
    )

    total_start_time = time.time()
    total_checkpoints = len(checkpoint_paths)
    print(
        f"Planner alignment run: {total_checkpoints} checkpoint(s), "
        f"{len(eval_episodes)} eval case(s), "
        f"landscape={cfg.analysis.landscape.enabled}"
    )

    try:
        for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths, start=1):
            output_path = output_path_for_checkpoint(run_dir, cfg.analysis.output_subdir, checkpoint_path)
            if output_path.exists() and not cfg.analysis.overwrite:
                print(f"Skipping existing planner-alignment file: {output_path}")
                continue

            checkpoint_start = time.time()
            print(f"[checkpoint {checkpoint_idx}/{total_checkpoints}] Analyzing {checkpoint_path.name}")
            model = load_model_object(checkpoint_path, cfg.analysis.device)

            current_structured = load_structured_eval(run_dir, checkpoint_path)
            compare_structured = None
            if cfg.analysis.landscape.compare_run_dir:
                compare_run_dir = Path(cfg.analysis.landscape.compare_run_dir)
                compare_checkpoint = find_matching_checkpoint(compare_run_dir, parse_epoch(checkpoint_path))
                if compare_checkpoint is not None:
                    compare_structured = load_structured_eval(compare_run_dir, compare_checkpoint)
            landscape_cases = (
                select_landscape_cases(cfg, current_eval=current_structured, compare_eval=compare_structured)
                if cfg.analysis.landscape.enabled
                else []
            )

            landscape_payload = {}
            landscape_summary = {}
            if cfg.analysis.landscape.enabled and landscape_cases:
                landscape_dir = output_dir / "landscape" / checkpoint_path.stem
                if cfg.analysis.landscape.save_plots:
                    landscape_dir.mkdir(parents=True, exist_ok=True)
                store_landscape_arrays = bool(cfg.analysis.landscape.save_npz)
                selected_eps = []
                selected_starts = []
                selected_categories = []
                one_step_landscape_metrics = []
                full_landscape_metrics = []
                one_step_model_grids = []
                one_step_center_actions = []
                one_step_center_std = []
                one_step_axis_a = []
                one_step_axis_b = []
                one_step_grid_best_actions = []
                one_step_mean_trace = []
                one_step_elite_trace = []
                one_step_var_trace = []
                one_step_mean_proj = []
                one_step_elite_proj = []
                full_model_grids = []
                full_center_actions = []
                full_center_std = []
                full_axis_a = []
                full_axis_b = []
                full_grid_best_actions = []
                full_mean_trace = []
                full_elite_trace = []
                full_var_trace = []
                full_mean_proj = []
                full_elite_proj = []
                grid_alphas = None
                grid_betas = None
                landscape_start = time.time()
                total_landscape_cases = len(landscape_cases)
                landscape_modes = int(bool(cfg.analysis.landscape.one_step)) + int(
                    bool(cfg.analysis.landscape.full_horizon)
                )
                total_landscape_tasks = total_landscape_cases * max(landscape_modes, 1)
                completed_landscape_tasks = 0

                for selected_idx, selected_case in enumerate(landscape_cases, start=1):
                    case = build_eval_case(
                        dataset,
                        episode_idx=int(selected_case["episode_idx"]),
                        start_step=int(selected_case["start_step"]),
                        goal_offset_steps=eval_cfg.eval.goal_offset_steps,
                    )
                    selected_eps.append(int(selected_case["episode_idx"]))
                    selected_starts.append(int(selected_case["start_step"]))
                    selected_categories.append(selected_case["category"])

                    if cfg.analysis.landscape.one_step:
                        print(
                            f"  landscape case {selected_idx}/{total_landscape_cases} "
                            f"[one-step] category={selected_case['category']} "
                            f"ep={int(selected_case['episode_idx'])} "
                            f"step={int(selected_case['start_step'])}"
                        )
                        mode_start = time.time()
                        one_landscape = run_cem_landscape_mode(
                            model=model,
                            cem_world=landscape_cem_world,
                            env_world=landscape_env_world,
                            case=case,
                            process=process,
                            transform=transform,
                            device=cfg.analysis.device,
                            cfg=cfg,
                            eval_cfg=eval_cfg,
                            horizon=1,
                        )
                        completed_landscape_tasks += 1
                        one_step_landscape_metrics.append(one_landscape["metrics"])
                        if store_landscape_arrays:
                            one_step_model_grids.append(one_landscape["model_grid"])
                            one_step_center_actions.append(one_landscape["landscape_center_action"])
                            one_step_center_std.append(one_landscape["landscape_center_std"])
                            one_step_axis_a.append(one_landscape["landscape_axis_a"])
                            one_step_axis_b.append(one_landscape["landscape_axis_b"])
                            one_step_grid_best_actions.append(one_landscape["landscape_grid_best_action"])
                            one_step_mean_trace.append(one_landscape["mean_trace"])
                            one_step_elite_trace.append(one_landscape["elite_trace"])
                            one_step_var_trace.append(one_landscape["var_trace"])
                            one_step_mean_proj.append(one_landscape["mean_proj"])
                            one_step_elite_proj.append(one_landscape["elite_proj"])
                            grid_alphas = one_landscape["alphas"]
                            grid_betas = one_landscape["betas"]
                        elapsed = time.time() - landscape_start
                        avg_task_time = elapsed / max(completed_landscape_tasks, 1)
                        remaining = total_landscape_tasks - completed_landscape_tasks
                        print(
                            f"    done in {time.time() - mode_start:.2f}s; "
                            f"landscape ETA {format_duration(avg_task_time * remaining)}"
                        )
                        if cfg.analysis.landscape.save_plots:
                            render_landscape_plot(
                                landscape_dir / (
                                    f"case_{selected_idx:02d}_ep_{int(selected_case['episode_idx'])}"
                                    f"_step_{int(selected_case['start_step'])}_one_step.png"
                                ),
                                title=(
                                    f"{checkpoint_path.stem} | one-step | {selected_case['category']} | "
                                    f"ep={int(selected_case['episode_idx'])} step={int(selected_case['start_step'])}"
                                ),
                                alphas=one_landscape["alphas"],
                                betas=one_landscape["betas"],
                                model_grid=one_landscape["model_grid"],
                                env_grid=one_landscape["env_grid"],
                                mean_proj=one_landscape["mean_proj"],
                                elite_proj=one_landscape["elite_proj"],
                            )

                    if cfg.analysis.landscape.full_horizon:
                        print(
                            f"  landscape case {selected_idx}/{total_landscape_cases} "
                            f"[full-horizon] category={selected_case['category']} "
                            f"ep={int(selected_case['episode_idx'])} "
                            f"step={int(selected_case['start_step'])}"
                        )
                        mode_start = time.time()
                        full_landscape = run_cem_landscape_mode(
                            model=model,
                            cem_world=landscape_cem_world,
                            env_world=landscape_env_world,
                            case=case,
                            process=process,
                            transform=transform,
                            device=cfg.analysis.device,
                            cfg=cfg,
                            eval_cfg=eval_cfg,
                            horizon=int(eval_cfg.plan_config.horizon),
                        )
                        completed_landscape_tasks += 1
                        full_landscape_metrics.append(full_landscape["metrics"])
                        if store_landscape_arrays:
                            full_model_grids.append(full_landscape["model_grid"])
                            full_center_actions.append(full_landscape["landscape_center_action"])
                            full_center_std.append(full_landscape["landscape_center_std"])
                            full_axis_a.append(full_landscape["landscape_axis_a"])
                            full_axis_b.append(full_landscape["landscape_axis_b"])
                            full_grid_best_actions.append(full_landscape["landscape_grid_best_action"])
                            full_mean_trace.append(full_landscape["mean_trace"])
                            full_elite_trace.append(full_landscape["elite_trace"])
                            full_var_trace.append(full_landscape["var_trace"])
                            full_mean_proj.append(full_landscape["mean_proj"])
                            full_elite_proj.append(full_landscape["elite_proj"])
                            grid_alphas = full_landscape["alphas"]
                            grid_betas = full_landscape["betas"]
                        elapsed = time.time() - landscape_start
                        avg_task_time = elapsed / max(completed_landscape_tasks, 1)
                        remaining = total_landscape_tasks - completed_landscape_tasks
                        print(
                            f"    done in {time.time() - mode_start:.2f}s; "
                            f"landscape ETA {format_duration(avg_task_time * remaining)}"
                        )
                        if cfg.analysis.landscape.save_plots:
                            render_landscape_plot(
                                landscape_dir / (
                                    f"case_{selected_idx:02d}_ep_{int(selected_case['episode_idx'])}"
                                    f"_step_{int(selected_case['start_step'])}_full_horizon.png"
                                ),
                                title=(
                                    f"{checkpoint_path.stem} | full-horizon | {selected_case['category']} | "
                                    f"ep={int(selected_case['episode_idx'])} step={int(selected_case['start_step'])}"
                                ),
                                alphas=full_landscape["alphas"],
                                betas=full_landscape["betas"],
                                model_grid=full_landscape["model_grid"],
                                env_grid=full_landscape["env_grid"],
                                mean_proj=full_landscape["mean_proj"],
                                elite_proj=full_landscape["elite_proj"],
                            )

                landscape_payload = {
                    "landscape_eval_episodes": np.asarray(selected_eps, dtype=np.int64),
                    "landscape_eval_start_idx": np.asarray(selected_starts, dtype=np.int64),
                    "landscape_case_categories": np.asarray(selected_categories),
                    "landscape_center_mode": np.asarray([str(cfg.analysis.landscape.get("center_mode", "cem_final"))]),
                    "landscape_axis_mode": np.asarray([str(cfg.analysis.landscape.get("axis_mode", "elite_pca"))]),
                    "landscape_reference_best_elite_idx": np.asarray(
                        [int(cfg.analysis.landscape.get("reference_best_elite_idx", 0))], dtype=np.int64
                    ),
                    "landscape_scan_step": np.asarray([int(cfg.analysis.landscape.get("scan_step", -1))], dtype=np.int64),
                    "landscape_scan_num_steps": np.asarray(
                        [int(cfg.analysis.landscape.get("scan_num_steps", 1))], dtype=np.int64
                    ),
                    "landscape_scan_dims": np.asarray(
                        list(cfg.analysis.landscape.get("scan_dims", cfg.analysis.landscape.get("last_action_dims", [0, 1]))),
                        dtype=np.int64,
                    ),
                }
                if store_landscape_arrays and grid_alphas is not None and grid_betas is not None:
                    landscape_payload["landscape_alphas"] = np.asarray(grid_alphas, dtype=np.float32)
                    landscape_payload["landscape_betas"] = np.asarray(grid_betas, dtype=np.float32)

                def _aggregate_metrics(records, prefix):
                    if not records:
                        return {}
                    keys = [
                        "cem_model_center_to_best_gap",
                        "cem_model_local_min_offset",
                        "cem_model_neighbor_std",
                        "cem_model_total_variation",
                        "cem_terminal_model_env_gap",
                        "cem_trace_mean_step_distance",
                        "cem_trace_elite_spread_final",
                    ]
                    agg = {}
                    for key in keys:
                        if key not in records[0]:
                            continue
                        values = np.asarray([record[key] for record in records], dtype=np.float32)
                        agg[f"{prefix}_{key}_values"] = values
                        agg[f"{prefix}_{key}"] = np.array([np.nanmean(values)], dtype=np.float32)
                    return agg

                landscape_payload.update(_aggregate_metrics(one_step_landscape_metrics, "one_step_landscape"))
                landscape_payload.update(_aggregate_metrics(full_landscape_metrics, "full_landscape"))

                if cfg.analysis.landscape.save_npz and one_step_model_grids:
                    landscape_payload["one_step_landscape_model_grids"] = np.stack(one_step_model_grids).astype(np.float32)
                    landscape_payload["one_step_landscape_center_actions"] = np.stack(one_step_center_actions).astype(np.float32)
                    landscape_payload["one_step_landscape_center_std"] = np.stack(one_step_center_std).astype(np.float32)
                    landscape_payload["one_step_landscape_axis_a"] = np.stack(one_step_axis_a).astype(np.float32)
                    landscape_payload["one_step_landscape_axis_b"] = np.stack(one_step_axis_b).astype(np.float32)
                    landscape_payload["one_step_landscape_grid_best_actions"] = np.stack(one_step_grid_best_actions).astype(np.float32)
                    landscape_payload["one_step_landscape_mean_trace"] = np.stack(one_step_mean_trace).astype(np.float32)
                    landscape_payload["one_step_landscape_elite_trace"] = np.stack(one_step_elite_trace).astype(np.float32)
                    landscape_payload["one_step_landscape_var_trace"] = np.stack(one_step_var_trace).astype(np.float32)
                    landscape_payload["one_step_landscape_mean_proj"] = np.stack(one_step_mean_proj).astype(np.float32)
                    landscape_payload["one_step_landscape_elite_proj"] = np.stack(one_step_elite_proj).astype(np.float32)
                if cfg.analysis.landscape.save_npz and full_model_grids:
                    landscape_payload["full_landscape_model_grids"] = np.stack(full_model_grids).astype(np.float32)
                    landscape_payload["full_landscape_center_actions"] = np.stack(full_center_actions).astype(np.float32)
                    landscape_payload["full_landscape_center_std"] = np.stack(full_center_std).astype(np.float32)
                    landscape_payload["full_landscape_axis_a"] = np.stack(full_axis_a).astype(np.float32)
                    landscape_payload["full_landscape_axis_b"] = np.stack(full_axis_b).astype(np.float32)
                    landscape_payload["full_landscape_grid_best_actions"] = np.stack(full_grid_best_actions).astype(np.float32)
                    landscape_payload["full_landscape_mean_trace"] = np.stack(full_mean_trace).astype(np.float32)
                    landscape_payload["full_landscape_elite_trace"] = np.stack(full_elite_trace).astype(np.float32)
                    landscape_payload["full_landscape_var_trace"] = np.stack(full_var_trace).astype(np.float32)
                    landscape_payload["full_landscape_mean_proj"] = np.stack(full_mean_proj).astype(np.float32)
                    landscape_payload["full_landscape_elite_proj"] = np.stack(full_elite_proj).astype(np.float32)

                def _summary_from_payload(prefix):
                    summary = {}
                    for metric_name in [
                        "cem_model_center_to_best_gap",
                        "cem_model_local_min_offset",
                        "cem_model_neighbor_std",
                        "cem_model_total_variation",
                        "cem_terminal_model_env_gap",
                        "cem_trace_mean_step_distance",
                        "cem_trace_elite_spread_final",
                    ]:
                        key = f"{prefix}_{metric_name}"
                        if key in landscape_payload:
                            summary[key] = float(landscape_payload[key][0])
                    return summary

                landscape_summary.update(_summary_from_payload("one_step_landscape"))
                landscape_summary.update(_summary_from_payload("full_landscape"))

            payload = {
                "epoch": np.array([parse_epoch(checkpoint_path)], dtype=np.int32),
                "eval_episodes": np.asarray(eval_episodes),
                "eval_start_idx": np.asarray(eval_start_idx),
            }
            payload.update(landscape_payload)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, **payload)

            summary_record = {
                "checkpoint": checkpoint_path.name,
                "epoch": int(payload["epoch"][0]),
            }
            summary_record.update(landscape_summary)
            with summary_path.open("a") as f:
                f.write(json.dumps(summary_record, sort_keys=True) + "\n")

            print(
                f"[checkpoint {checkpoint_idx}/{total_checkpoints}] done in "
                f"{format_duration(time.time() - checkpoint_start)}"
            )
    finally:
        try:
            landscape_cem_world.close()
        except Exception:
            pass
        try:
            landscape_env_world.close()
        except Exception:
            pass
        del landscape_cem_world
        del landscape_env_world
        gc.collect()


if __name__ == "__main__":
    run()
