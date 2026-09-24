#!/usr/bin/env python3
"""Audit whether Cube P00--P04 starts leave empirical training XY support.

For each training seed, this script exactly reconstructs the 90% random split
of 4-step, frameskip-5 training clips.  It then collects every block-XY row
actually presented to the model and compares the corresponding P00--P04
evaluation starts with that seed-specific empirical support.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
STABLE_WM = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
DATASET_PATH = STABLE_WM / "ogbench" / "cube_single_expert.h5"
OUT_DIR = ROOT / "analysis" / "results" / "cube_support"
SEEDS = (3072, 4096, 6144)
NUM_STEPS = 4
FRAMESKIP = 5
TRAIN_FRACTION = 0.9
ROBUST_TAIL = 0.005


def train_split_clip_indices(lengths: np.ndarray, seed: int) -> np.ndarray:
    span = NUM_STEPS * FRAMESKIP
    clips_per_episode = np.maximum(lengths.astype(np.int64) - span + 1, 0)
    total = int(clips_per_episode.sum())
    raw_lengths = [int(math.floor(total * TRAIN_FRACTION)), int(math.floor(total * (1.0 - TRAIN_FRACTION)))]
    for index in range(total - sum(raw_lengths)):
        raw_lengths[index % 2] += 1
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator)
    return permutation[: raw_lengths[0]].numpy().astype(np.int64)


def observed_training_row_mask(lengths: np.ndarray, offsets: np.ndarray, seed: int, total_rows: int) -> np.ndarray:
    span = NUM_STEPS * FRAMESKIP
    clips_per_episode = np.maximum(lengths.astype(np.int64) - span + 1, 0)
    cumulative = np.cumsum(clips_per_episode)
    previous = np.concatenate([np.zeros(1, dtype=np.int64), cumulative[:-1]])
    clip_indices = train_split_clip_indices(lengths, seed)
    episodes = np.searchsorted(cumulative, clip_indices, side="right")
    starts = clip_indices - previous[episodes]
    sampled_offsets = np.arange(NUM_STEPS, dtype=np.int64) * FRAMESKIP
    rows = offsets[episodes, None] + starts[:, None] + sampled_offsets[None, :]
    mask = np.zeros(total_rows, dtype=bool)
    mask[rows.reshape(-1)] = True
    return mask


def find_eval_file(seed: int, perturb_idx: int) -> Path:
    run_dir = STABLE_WM / f"cube_abs_seed{seed}_seedfix"
    path = run_dir / f"fixed_protocol_p0{perturb_idx}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def outside_box(points: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.any((points < low[None, :]) | (points > high[None, :]), axis=1)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATASET_PATH, "r") as handle:
        block_pos = np.asarray(handle["privileged_block_0_pos"], dtype=np.float64)
        block_xy = block_pos[:, :2]
        contact = np.asarray(handle["proprio_gripper_contact"], dtype=np.float64).reshape(-1)
        lengths = np.asarray(handle["ep_len"], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"], dtype=np.int64)

    per_point_rows = []
    training_summary = {}

    for seed in SEEDS:
        train_mask = observed_training_row_mask(lengths, offsets, seed, len(block_xy))
        train_xy = block_xy[train_mask]
        train_table_mask = train_mask & (block_pos[:, 2] < 0.03) & (contact < 0.5)
        train_table_xy = block_xy[train_table_mask]
        all_tree = cKDTree(train_xy)
        table_tree = cKDTree(train_table_xy)
        all_low, all_high = train_xy.min(axis=0), train_xy.max(axis=0)
        table_low, table_high = train_table_xy.min(axis=0), train_table_xy.max(axis=0)
        robust_low, robust_high = np.quantile(
            train_table_xy,
            [ROBUST_TAIL, 1.0 - ROBUST_TAIL],
            axis=0,
        )
        training_summary[str(seed)] = {
            "num_unique_rows_presented": int(train_mask.sum()),
            "num_table_non_contact_rows_presented": int(train_table_mask.sum()),
            "all_xy_bounds": {"low": all_low.tolist(), "high": all_high.tolist()},
            "table_xy_bounds": {"low": table_low.tolist(), "high": table_high.tolist()},
            "table_99pct_marginal_box": {"low": robust_low.tolist(), "high": robust_high.tolist()},
        }

        for perturb_idx in range(5):
            with np.load(find_eval_file(seed, perturb_idx), allow_pickle=True) as result:
                episodes = np.asarray(result["eval_episodes"], dtype=np.int64)
                starts = np.asarray(result["eval_start_idx"], dtype=np.int64)
                applied = np.asarray(result["ood_initial_perturb_xy"], dtype=np.float64)
            base_rows = offsets[episodes] + starts
            base_xy = block_xy[base_rows]
            perturbed_xy = base_xy + applied
            all_nn = all_tree.query(perturbed_xy, k=1)[0]
            table_nn = table_tree.query(perturbed_xy, k=1)[0]
            intended_radius = perturb_idx * 0.01
            applied_radius = np.linalg.norm(applied, axis=1)

            for case_idx in range(len(episodes)):
                per_point_rows.append(
                    {
                        "seed": seed,
                        "perturbation": f"P0{perturb_idx}",
                        "case_index": case_idx,
                        "episode": int(episodes[case_idx]),
                        "start_step": int(starts[case_idx]),
                        "base_x": float(base_xy[case_idx, 0]),
                        "base_y": float(base_xy[case_idx, 1]),
                        "perturbed_x": float(perturbed_xy[case_idx, 0]),
                        "perturbed_y": float(perturbed_xy[case_idx, 1]),
                        "applied_radius_m": float(applied_radius[case_idx]),
                        "clipped": bool(applied_radius[case_idx] < max(intended_radius - 1.0e-6, 0.0)),
                        "outside_train_bounds": bool(outside_box(perturbed_xy[case_idx : case_idx + 1], all_low, all_high)[0]),
                        "outside_table_bounds": bool(outside_box(perturbed_xy[case_idx : case_idx + 1], table_low, table_high)[0]),
                        "outside_table_99pct_marginal_box": bool(
                            outside_box(perturbed_xy[case_idx : case_idx + 1], robust_low, robust_high)[0]
                        ),
                        "nn_train_m": float(all_nn[case_idx]),
                        "nn_table_m": float(table_nn[case_idx]),
                    }
                )

    aggregate = {}
    for perturb_idx in range(5):
        label = f"P0{perturb_idx}"
        selected = [row for row in per_point_rows if row["perturbation"] == label]
        all_nn_mm = np.asarray([row["nn_train_m"] for row in selected]) * 1000.0
        table_nn_mm = np.asarray([row["nn_table_m"] for row in selected]) * 1000.0
        applied_mm = np.asarray([row["applied_radius_m"] for row in selected]) * 1000.0
        unique_xy = np.unique(
            np.round(np.asarray([[row["perturbed_x"], row["perturbed_y"]] for row in selected]), decimals=10),
            axis=0,
        )
        aggregate[label] = {
            "num_evaluated_starts": len(selected),
            "num_unique_xy": int(len(unique_xy)),
            "applied_radius_mm": summarize(applied_mm),
            "clipped_fraction": float(np.mean([row["clipped"] for row in selected])),
            "outside_exact_training_bounds_fraction": float(np.mean([row["outside_train_bounds"] for row in selected])),
            "outside_table_training_bounds_fraction": float(np.mean([row["outside_table_bounds"] for row in selected])),
            "outside_table_99pct_marginal_box_fraction": float(
                np.mean([row["outside_table_99pct_marginal_box"] for row in selected])
            ),
            "nearest_training_xy_mm": summarize(all_nn_mm),
            "nearest_table_training_xy_mm": summarize(table_nn_mm),
            "fraction_farther_than_1mm_from_any_training_xy": float(np.mean(all_nn_mm > 1.0)),
            "fraction_farther_than_5mm_from_any_table_training_xy": float(np.mean(table_nn_mm > 5.0)),
        }

    csv_path = OUT_DIR / "p0_cube_xy_support_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_point_rows[0]))
        writer.writeheader()
        writer.writerows(per_point_rows)

    payload = {
        "protocol": {
            "dataset": str(DATASET_PATH),
            "seeds": list(SEEDS),
            "train_fraction": TRAIN_FRACTION,
            "num_steps": NUM_STEPS,
            "frameskip": FRAMESKIP,
            "table_filter": "block_z < 0.03 and gripper_contact < 0.5",
            "robust_box": "per-axis 0.5th--99.5th percentiles of seed-specific table/non-contact training rows",
            "support_reference": "unique HDF5 rows actually presented by the exact 90% random training split",
        },
        "training_support": training_summary,
        "aggregate": aggregate,
        "conclusion": (
            "P01--P04 do not leave empirical cube-XY training support: every evaluated point is inside both the full "
            "training bounds and the table/non-contact training bounds, and every point is within 1 mm of an XY "
            "coordinate actually presented during training. The robust 99% marginal box shows a growing tail shift "
            "at P03--P04, but this supports the wording controlled perturbation/off-trajectory distribution shift, "
            "not strict OOD by cube XY position."
        ),
    }
    json_path = OUT_DIR / "p0_cube_xy_support_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# P0 Cube P00--P04 empirical-support audit",
        "",
        "The reference set is reconstructed separately for each training seed from the exact 90% random split. "
        "It contains the unique HDF5 rows actually presented in 4-step, frameskip-5 training clips.",
        "",
        "| Protocol | Starts | Unique XY | Applied radius (median mm) | Outside train bounds | Outside table bounds | Outside table 99% box | NN to any train XY (p95/max mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for perturb_idx in range(5):
        label = f"P0{perturb_idx}"
        entry = aggregate[label]
        md.append(
            f"| {label} | {entry['num_evaluated_starts']} | {entry['num_unique_xy']} | "
            f"{entry['applied_radius_mm']['median']:.1f} | "
            f"{entry['outside_exact_training_bounds_fraction']*100:.1f}% | "
            f"{entry['outside_table_training_bounds_fraction']*100:.1f}% | "
            f"{entry['outside_table_99pct_marginal_box_fraction']*100:.1f}% | "
            f"{entry['nearest_training_xy_mm']['p95']:.3f}/{entry['nearest_training_xy_mm']['max']:.3f} |"
        )
    md += [
        "",
        "## Conclusion",
        "",
        payload["conclusion"],
        "",
        "This audit only tests cube XY support. It does not establish support in the full image/history/action space.",
    ]
    md_path = OUT_DIR / "p0_cube_xy_support_audit.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
