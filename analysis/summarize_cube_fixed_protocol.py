#!/usr/bin/env python3
"""Validate and summarize fixed-protocol Cube P00--P04 results.

Evaluator success_rate is a percentage; episode_successes is binary.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cube_fixed_protocol import ALL, SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path,
                        default=Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm")))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "analysis" / "results" / "cube_fixed_protocol.json")
    args = parser.parse_args()
    manifest_path = ROOT / "protocols" / "cube_hardstart.npz"
    with np.load(manifest_path, allow_pickle=False) as manifest:
        expected_ep = manifest["eval_episodes"]
        expected_step = manifest["eval_start_idx"]
    expected_metadata = {
        "eval_seed": 42,
        "num_eval": 50,
        "goal_offset_steps": 25,
        "sample_goal_offset_steps": 25,
        "eval_budget": 50,
        "plan_horizon": 5,
        "receding_horizon": 5,
        "action_block": 5,
        "ood_start_filter_mode": "table_non_contact",
        "rolling_goal_enabled": False,
    }
    perturbations = {}
    rows = []
    validated_files = 0
    for variant in ALL:
        for seed in SEEDS:
            run = f"{variant}_seed{seed}_seedfix"
            rates = []
            missing = []
            for radius in range(5):
                path = args.cache_root / run / f"fixed_protocol_p0{radius}.npz"
                if not path.is_file():
                    missing.append(radius)
                    rates.append(None)
                    continue
                with np.load(path, allow_pickle=False) as result:
                    if not (np.array_equal(result["eval_episodes"], expected_ep) and
                            np.array_equal(result["eval_start_idx"], expected_step)):
                        raise RuntimeError(f"Unmatched starts: {path}")
                    successes = np.asarray(result["episode_successes"])
                    if successes.shape != (50,) or not np.all(np.isin(successes, (0, 1))):
                        raise RuntimeError(f"Invalid per-episode successes: {path}")
                    rate = float(result["success_rate"])
                    if not np.isfinite(rate) or not np.isclose(rate, successes.mean() * 100):
                        raise RuntimeError(f"Success percentage disagrees with episodes: {path}")
                    policy = f"{run}/{run}_epoch_10"
                    if result["policy"].tolist() != [policy]:
                        raise RuntimeError(f"Wrong policy: {path}")
                    for key, value in expected_metadata.items():
                        if result[key].tolist() != [value]:
                            raise RuntimeError(f"Wrong {key}: {path}")
                    perturb = np.asarray(result["ood_initial_perturb_xy"])
                    if perturb.shape != (50, 2) or not np.all(np.isfinite(perturb)):
                        raise RuntimeError(f"Invalid perturbations: {path}")
                    if np.any(np.linalg.norm(perturb, axis=1) > radius * 0.01 + 1e-6):
                        raise RuntimeError(f"Perturbation exceeds radius: {path}")
                    if radius in perturbations and not np.array_equal(perturbations[radius], perturb):
                        raise RuntimeError(f"Perturbations differ between policies: {path}")
                    perturbations.setdefault(radius, perturb.copy())
                    rates.append(rate)
                    validated_files += 1
            rows.append({"variant": variant, "seed": seed,
                         **{f"p0{i}": rates[i] for i in range(5)},
                         "hs": float(np.mean(rates)) if not missing else None,
                         "missing": missing})
    aggregate = {}
    metrics = [f"p0{i}" for i in range(5)] + ["hs"]
    for variant in ALL:
        selected = [row for row in rows if row["variant"] == variant and not row["missing"]]
        aggregate[variant] = {"complete_seeds": len(selected)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            aggregate[variant][metric] = {
                "mean": float(values.mean()) if len(values) else None,
                "population_sd": float(values.std(ddof=0)) if len(values) else None,
                "values": values.tolist(),
            }
    payload = {"manifest": "protocols/cube_hardstart.npz", "seeds": list(SEEDS),
               "validated_files": validated_files,
               "expected_files": len(ALL) * len(SEEDS) * 5,
               "rate_unit": "percent", "sd_definition": "population",
               "per_seed": rows, "aggregate": aggregate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "seed", "p00", "p01", "p02",
                                                   "p03", "p04", "hs", "missing"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "missing": ",".join(map(str, row["missing"]))})
    with args.output.with_name(args.output.stem + "_aggregate.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "complete_seeds"] +
                        [f"{metric}_{field}" for metric in metrics
                         for field in ("mean", "population_sd")])
        for variant, data in aggregate.items():
            writer.writerow([variant, data["complete_seeds"]] +
                            [data[metric][field] for metric in metrics
                             for field in ("mean", "population_sd")])
    print(f"Validated {validated_files}/{len(ALL) * len(SEEDS) * 5} result files")
    print(json.dumps({variant: {"complete_seeds": data["complete_seeds"],
                                "hs": data["hs"]} for variant, data in aggregate.items()}, indent=2))


if __name__ == "__main__":
    main()
