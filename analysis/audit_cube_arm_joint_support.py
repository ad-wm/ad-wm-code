#!/usr/bin/env python3
"""Audit Cube perturbations in joint cube/arm and relative-pose spaces."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

import audit_cube_perturbation_support as xy_audit


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = xy_audit.OUT_DIR
CALIBRATION_SAMPLES = 20_000
NEIGHBORS = 64
CALIBRATION_SEED = 20260806

SPACES = {
    "cube_arm_joint": "[cube_x, cube_y, end_effector_x, end_effector_y, end_effector_z]",
    "arm_cube_relative": "end_effector_xyz - cube_xyz",
}


def cross_episode_nn(
    tree: cKDTree,
    train_episode: np.ndarray,
    query: np.ndarray,
    query_episode: np.ndarray,
) -> np.ndarray:
    distances, indices = tree.query(query, k=NEIGHBORS)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    valid = train_episode[indices] != query_episode[:, None]
    if not np.all(valid.any(axis=1)):
        distances, indices = tree.query(query, k=NEIGHBORS * 4)
        valid = train_episode[indices] != query_episode[:, None]
    if not np.all(valid.any(axis=1)):
        raise RuntimeError("failed to find a cross-episode neighbor")
    first = valid.argmax(axis=1)
    return distances[np.arange(len(query)), first]


def summary_mm(values_m: np.ndarray) -> dict[str, float]:
    values = np.asarray(values_m, dtype=np.float64) * 1000.0
    return {
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(xy_audit.DATASET_PATH, "r") as handle:
        block = np.asarray(handle["privileged_block_0_pos"], dtype=np.float64)
        effector = np.asarray(handle["proprio_effector_pos"], dtype=np.float64)
        contact = np.asarray(handle["proprio_gripper_contact"], dtype=np.float64).reshape(-1)
        episode = np.asarray(handle["ep_idx"], dtype=np.int64)
        lengths = np.asarray(handle["ep_len"], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"], dtype=np.int64)

    rng = np.random.default_rng(CALIBRATION_SEED)
    calibration = {}
    point_rows = []

    for seed in xy_audit.SEEDS:
        train_mask = xy_audit.observed_training_row_mask(lengths, offsets, seed, len(block))
        train_mask &= (block[:, 2] < 0.03) & (contact < 0.5)
        train_episode = episode[train_mask]
        representations = {
            "cube_arm_joint": np.concatenate([block[train_mask, :2], effector[train_mask, :3]], axis=1),
            "arm_cube_relative": effector[train_mask, :3] - block[train_mask, :3],
        }

        trees = {name: cKDTree(values) for name, values in representations.items()}
        sample_idx = rng.choice(len(train_episode), size=min(CALIBRATION_SAMPLES, len(train_episode)), replace=False)
        calibration[str(seed)] = {}
        thresholds = {}
        for name, values in representations.items():
            distances = cross_episode_nn(
                trees[name],
                train_episode,
                values[sample_idx],
                train_episode[sample_idx],
            )
            p95, p99 = np.quantile(distances, [0.95, 0.99])
            thresholds[name] = (float(p95), float(p99))
            calibration[str(seed)][name] = {
                "description": SPACES[name],
                "num_training_rows": int(len(values)),
                "cross_episode_nn_mm": summary_mm(distances),
                "p95_mm": float(p95 * 1000.0),
                "p99_mm": float(p99 * 1000.0),
            }

        for perturb_idx in range(5):
            with np.load(xy_audit.find_eval_file(seed, perturb_idx), allow_pickle=True) as result:
                eval_episode = np.asarray(result["eval_episodes"], dtype=np.int64)
                starts = np.asarray(result["eval_start_idx"], dtype=np.int64)
                applied = np.asarray(result["ood_initial_perturb_xy"], dtype=np.float64)
            source_rows = offsets[eval_episode] + starts
            perturbed_block = block[source_rows].copy()
            perturbed_block[:, :2] += applied
            eval_representations = {
                "cube_arm_joint": np.concatenate([perturbed_block[:, :2], effector[source_rows, :3]], axis=1),
                "arm_cube_relative": effector[source_rows, :3] - perturbed_block[:, :3],
            }

            distances = {}
            for name, values in eval_representations.items():
                distances[name] = {
                    "any": trees[name].query(values, k=1)[0],
                    "cross": cross_episode_nn(trees[name], train_episode, values, eval_episode),
                }

            for case_idx in range(len(eval_episode)):
                row = {
                    "seed": seed,
                    "perturbation": f"P0{perturb_idx}",
                    "case_index": case_idx,
                    "episode": int(eval_episode[case_idx]),
                    "start_step": int(starts[case_idx]),
                }
                for name in SPACES:
                    p95, p99 = thresholds[name]
                    row[f"{name}_nn_any_m"] = float(distances[name]["any"][case_idx])
                    row[f"{name}_nn_cross_episode_m"] = float(distances[name]["cross"][case_idx])
                    row[f"{name}_above_natural_p95"] = bool(distances[name]["cross"][case_idx] > p95)
                    row[f"{name}_above_natural_p99"] = bool(distances[name]["cross"][case_idx] > p99)
                point_rows.append(row)

        del trees

    aggregate = {}
    for perturb_idx in range(5):
        label = f"P0{perturb_idx}"
        selected = [row for row in point_rows if row["perturbation"] == label]
        aggregate[label] = {}
        for name in SPACES:
            any_values = np.asarray([row[f"{name}_nn_any_m"] for row in selected])
            cross_values = np.asarray([row[f"{name}_nn_cross_episode_m"] for row in selected])
            aggregate[label][name] = {
                "description": SPACES[name],
                "nearest_any_training_state_mm": summary_mm(any_values),
                "nearest_cross_episode_state_mm": summary_mm(cross_values),
                "fraction_above_natural_p95": float(np.mean([row[f"{name}_above_natural_p95"] for row in selected])),
                "fraction_above_natural_p99": float(np.mean([row[f"{name}_above_natural_p99"] for row in selected])),
            }

    csv_path = OUT_DIR / "supplement_cube_arm_joint_support_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)

    payload = {
        "protocol": {
            "seeds": list(xy_audit.SEEDS),
            "training_reference": "exact 90% split; table/non-contact rows actually presented to each model",
            "spaces": SPACES,
            "natural_distance_calibration": (
                f"{CALIBRATION_SAMPLES} training states per seed; nearest training neighbor from a different episode"
            ),
            "calibration_neighbors_searched": NEIGHBORS,
            "calibration_seed": CALIBRATION_SEED,
        },
        "calibration": calibration,
        "aggregate": aggregate,
        "interpretation": (
            "Cube XY remains marginally inside training support, but moving the cube while holding the arm fixed creates "
            "a strong joint/relational shift. The joint cube/end-effector distance grows monotonically, and the relative "
            "end-effector-minus-cube geometry is outside the natural cross-episode p95 neighborhood for many P02 and most "
            "P03/P04 starts. This supports an off-trajectory joint-state or relational OOD claim, not a marginal cube-XY OOD claim."
        ),
    }
    json_path = OUT_DIR / "supplement_cube_arm_joint_support.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Supplement: cube–arm joint-support audit",
        "",
        "The arm is not independently perturbed, but the cube perturbation changes the joint cube/arm state and "
        "the relative end-effector-to-cube geometry. Natural-distance thresholds are calibrated using 20,000 "
        "table/non-contact training states per seed and the nearest training state from a different episode.",
        "",
        "| Protocol | Joint NN any train, p95 (mm) | Joint > natural p95 | Joint > p99 | Relative NN any train, p95 (mm) | Relative > natural p95 | Relative > p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for perturb_idx in range(5):
        label = f"P0{perturb_idx}"
        joint = aggregate[label]["cube_arm_joint"]
        relative = aggregate[label]["arm_cube_relative"]
        md.append(
            f"| {label} | {joint['nearest_any_training_state_mm']['p95']:.2f} | "
            f"{joint['fraction_above_natural_p95']*100:.1f}% | {joint['fraction_above_natural_p99']*100:.1f}% | "
            f"{relative['nearest_any_training_state_mm']['p95']:.2f} | "
            f"{relative['fraction_above_natural_p95']*100:.1f}% | {relative['fraction_above_natural_p99']*100:.1f}% |"
        )
    md += ["", "## Interpretation", "", payload["interpretation"]]
    md_path = OUT_DIR / "supplement_cube_arm_joint_support.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
