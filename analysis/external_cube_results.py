#!/usr/bin/env python3
"""Package or verify the paper's external Cube per-episode results.

No third-party model code or checkpoints are copied into this release.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("original", "p00", "p01", "p02", "p03", "p04")
METHODS = ("fast_lewm_cem", "subjepa_cem", "intact_direct",
           "intact_pure_cem", "intact_actor_cem")
EPISODE_FIELDS = ("method", "training_seed", "eval_seed", "protocol",
                  "case_index", "episode", "start_idx", "success",
                  "perturb_x", "perturb_y")
AGGREGATE_FIELDS = ("method", "protocol", "checkpoints", "mean_percent",
                    "population_sd_percent")
SOURCE_FIELDS = ("method", "training_seed", "protocol", "source_path", "sha256")


def source_files(root: Path):
    legacy = root / "results/cube_hard_p00_p04_all_seed42"
    originals = root / "results/cube_original_p00_p04_intact3_fast_sub"
    paper = root / "results/cube_paper_intact_e5_3train"
    for method in METHODS[:2]:
        for protocol in PROTOCOLS:
            path = (originals / method / "seed42/original.npz" if protocol == "original"
                    else legacy / method / f"{protocol}.npz")
            yield method, "released", protocol, path
    for method in METHODS[2:]:
        for seed in (0, 42, 3072):
            for protocol in PROTOCOLS:
                yield method, str(seed), protocol, (
                    paper / method / f"train{seed}" / "eval42" / f"{protocol}.npz"
                )


def manifests():
    result = {}
    for protocol in PROTOCOLS:
        name = "cube_original.npz" if protocol == "original" else "cube_hardstart.npz"
        with np.load(ROOT / "protocols" / name, allow_pickle=False) as data:
            result[protocol] = (data["eval_episodes"], data["eval_start_idx"])
    return result


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator=chr(10))
        writer.writeheader()
        writer.writerows(rows)


def validate_and_aggregate(rows, expected):
    groups = defaultdict(list)
    perturbations = {}
    for row in rows:
        key = (row["method"], row["training_seed"], row["protocol"])
        groups[key].append(row)
    expected_keys = {(method, seed, protocol)
                     for method, seed, protocol, _ in source_files(Path("."))}
    if set(groups) != expected_keys:
        raise RuntimeError(f"Expected {len(expected_keys)} method/seed/protocol groups; "
                           f"found {len(groups)}")
    rates = defaultdict(list)
    for (method, seed, protocol), cases in groups.items():
        if len(cases) != 50:
            raise RuntimeError(f"Expected 50 cases in {(method, seed, protocol)}")
        cases = sorted(cases, key=lambda row: int(row["case_index"]))
        if [int(row["case_index"]) for row in cases] != list(range(50)):
            raise RuntimeError(f"Missing/duplicate case indices in {(method, seed, protocol)}")
        episodes, starts = expected[protocol]
        for idx, row in enumerate(cases):
            if (int(row["episode"]), int(row["start_idx"])) != (int(episodes[idx]), int(starts[idx])):
                raise RuntimeError(f"Unmatched start in {(method, seed, protocol, idx)}")
            if int(row["eval_seed"]) != 42 or int(row["success"]) not in (0, 1):
                raise RuntimeError(f"Invalid seed/success in {(method, seed, protocol, idx)}")
            perturb = (float(row["perturb_x"]), float(row["perturb_y"]))
            if protocol != "original":
                radius = int(protocol[-1]) * 0.01
                if np.hypot(*perturb) > radius + 1e-6:
                    raise RuntimeError(f"Perturbation exceeds {protocol} radius")
                pkey = (protocol, idx)
                if pkey in perturbations and not np.allclose(perturbations[pkey], perturb, atol=1e-7):
                    raise RuntimeError(f"Unmatched perturbation in {(method, seed, protocol, idx)}")
                perturbations.setdefault(pkey, perturb)
        rates[(method, protocol)].append(sum(int(row["success"]) for row in cases) * 2.0)
    aggregates = []
    for method in METHODS:
        for protocol in PROTOCOLS:
            values = np.asarray(rates[(method, protocol)], dtype=float)
            expected_n = 1 if method in METHODS[:2] else 3
            if len(values) != expected_n:
                raise RuntimeError(f"Wrong checkpoint count in {(method, protocol)}")
            aggregates.append({
                "method": method, "protocol": protocol, "checkpoints": expected_n,
                "mean_percent": f"{values.mean():.10g}",
                "population_sd_percent": f"{values.std(ddof=0):.10g}",
            })
    return aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        help="Import saved NPZ results from the local evaluation workspace")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    expected = manifests()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    episode_path = args.results_dir / "external_cube_episodes.csv"
    aggregate_path = args.results_dir / "external_cube_aggregate.csv"
    source_path = args.results_dir / "external_cube_sources.csv"
    if args.source_root:
        rows, sources = [], []
        for method, seed, protocol, path in source_files(args.source_root):
            with np.load(path, allow_pickle=False) as data:
                successes = np.asarray(data["episode_successes"])
                episodes, starts = expected[protocol]
                if (successes.shape != (50,) or
                    not np.array_equal(data["eval_episodes"], episodes) or
                    not np.array_equal(data["eval_start_idx"], starts) or
                    not np.isclose(float(data["success_rate"]), successes.mean() * 100)):
                    raise RuntimeError(f"Invalid source result: {path}")
                perturb = (np.asarray(data["ood_initial_perturb_xy"])
                           if protocol != "original" else np.zeros((50, 2)))
                if perturb.shape != (50, 2):
                    raise RuntimeError(f"Invalid perturbations: {path}")
                for idx in range(50):
                    rows.append(dict(method=method, training_seed=seed, eval_seed=42,
                                     protocol=protocol, case_index=idx,
                                     episode=int(episodes[idx]), start_idx=int(starts[idx]),
                                     success=int(successes[idx]),
                                     perturb_x=float(perturb[idx, 0]),
                                     perturb_y=float(perturb[idx, 1])))
            sources.append(dict(method=method, training_seed=seed, protocol=protocol,
                                source_path=str(path.relative_to(args.source_root)),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        aggregates = validate_and_aggregate(rows, expected)
        write_csv(episode_path, EPISODE_FIELDS, rows)
        write_csv(aggregate_path, AGGREGATE_FIELDS, aggregates)
        write_csv(source_path, SOURCE_FIELDS, sources)
    else:
        with episode_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        aggregates = validate_and_aggregate(rows, expected)
        with aggregate_path.open(newline="") as handle:
            saved = list(csv.DictReader(handle))
        if saved != [{key: str(value) for key, value in row.items()} for row in aggregates]:
            raise RuntimeError(f"Aggregate does not match episode records: {aggregate_path}")
    print(f"Validated {len(rows)} external episode records across "
          f"{len(rows) // 50} result files; summary: {aggregate_path}")


if __name__ == "__main__":
    main()
