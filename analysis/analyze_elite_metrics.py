#!/usr/bin/env python3
"""Recompute CEM-facing metrics from saved shared-bank raw costs.

The primary good-hit definition is rank based: the predicted elite set must
intersect the best 1% of candidates under environment-realized visual cost.
This avoids cost-scale-dependent thresholds and has an exact random baseline.
Best-hit (the elite contains the single realized-best candidate) is also
reported.  Neither threshold is selected from model outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats


REPO_DIR = Path(__file__).resolve().parents[1]
STABLE_WM = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
SEEDS = (3072, 4096, 6144)
MODELS = {
    "LeWM": "cube_abs_seed{seed}_seedfix",
    "Res": "cube_res_seed{seed}_seedfix",
    "Res+Inv": "cube_res_inv_detachact_seed{seed}_seedfix",
    "Res+MI (.01)": "cube_res_mifixedvar_seed{seed}_seedfix",
    "AD-WM (.01)": "cube_res_inv_detachact_mifixedvar_seed{seed}_seedfix",
}
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260905


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-subdir",
        default="counterfactual_metrics_visual_latent_n300",
        help="Diagnostic subdirectory inside each stable-wm run.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=(15, 30, 60),
        help="Elite sizes to audit for the paper's N=300 candidate bank.",
    )
    parser.add_argument(
        "--good-fraction",
        type=float,
        default=0.01,
        help="Realized-best fraction defining good candidates (default: 1%%).",
    )
    parser.add_argument(
        "--success-pattern",
        default="fixed_protocol_p0{radius}.npz",
        help="Per-radius result filename pattern below each run directory.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def metric_path(run_dir: Path, output_subdir: str) -> Path:
    matches = sorted((run_dir / output_subdir).glob("*.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one .npz under {run_dir / output_subdir}, found {matches}"
        )
    return matches[0]


def random_hit_probability(num_candidates: int, elite_size: int, good_count: int) -> float:
    """P(a uniformly random k-subset intersects a fixed m-subset)."""
    n = int(num_candidates)
    k = min(int(elite_size), n)
    m = min(max(int(good_count), 0), n)
    if m == 0 or k == 0:
        return 0.0
    if n - m < k:
        return 1.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def load_raw(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        model_costs = np.asarray(data["shared_planner_model_costs"], dtype=np.float64)
        env_costs = np.asarray(data["shared_planner_env_costs"], dtype=np.float64)
        candidates = np.asarray(data["shared_planner_flat_candidates"], dtype=np.float32)
        labels = np.asarray(data["shared_planner_candidate_labels"]).astype(str)
        episodes = np.asarray(data["shared_planner_eval_episodes"], dtype=np.int64)
        starts = np.asarray(data["shared_planner_eval_start_idx"], dtype=np.int64)

    if not (model_costs.shape == env_costs.shape == labels.shape):
        raise RuntimeError(
            f"cost/label shape mismatch in {path}: "
            f"{model_costs.shape}, {env_costs.shape}, {labels.shape}"
        )
    if candidates.shape[:2] != model_costs.shape:
        raise RuntimeError(f"candidate shape mismatch in {path}: {candidates.shape}")

    if not np.all(np.isfinite(model_costs)) or not np.all(np.isfinite(env_costs)):
        raise RuntimeError(f"non-finite costs in {path}")

    protocol = {
        "episodes": episodes.tobytes(),
        "starts": starts.tobytes(),
        "candidate_hash": hashlib.sha256(candidates.tobytes()).hexdigest(),
        "shape": candidates.shape,
    }
    return model_costs, env_costs, protocol


def load_hardstart_success(run_dir: Path, filename_pattern: str) -> float:
    rates = []
    for perturb_idx in range(5):
        pattern = filename_pattern.format(radius=perturb_idx)
        matches = sorted(run_dir.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one P0{perturb_idx} result under {run_dir}, found {matches}"
            )
        with np.load(matches[0], allow_pickle=True) as data:
            rates.append(float(np.mean(np.asarray(data["episode_successes"], dtype=np.float64))))
    return float(np.mean(rates))


def case_metrics(
    model_costs: np.ndarray,
    env_costs: np.ndarray,
    elite_size: int,
    good_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    regrets = []
    elite_mean_regrets = []
    good_hits = []
    best_hits = []
    cad_values = []
    for predicted, realized in zip(model_costs, env_costs):
        predicted_elite = np.argsort(predicted, kind="stable")[:elite_size]
        realized_order = np.argsort(realized, kind="stable")
        realized_best = int(realized_order[0])
        realized_good = realized_order[:good_count]
        denom = max(float(np.max(realized) - realized[realized_best]), 1.0e-8)
        regret = (float(np.min(realized[predicted_elite])) - float(realized[realized_best])) / denom
        ideal_elite = realized_order[:elite_size]
        elite_mean_regret = (
            float(np.mean(realized[predicted_elite])) - float(np.mean(realized[ideal_elite]))
        ) / denom
        regrets.append(regret)
        elite_mean_regrets.append(elite_mean_regret)
        good_hits.append(float(np.intersect1d(predicted_elite, realized_good).size > 0))
        best_hits.append(float(realized_best in predicted_elite))
        cad_values.append(float(stats.spearmanr(predicted, realized).statistic))
    return tuple(
        np.asarray(x, dtype=np.float64)
        for x in (regrets, elite_mean_regrets, good_hits, best_hits, cad_values)
    )


def hierarchical_paired_reduction_ci(
    baseline_by_seed: dict[int, np.ndarray],
    target_by_seed: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap the reduction baseline-target, pairing models by case."""
    reductions = {
        seed: np.asarray(baseline_by_seed[seed]) - np.asarray(target_by_seed[seed])
        for seed in SEEDS
    }
    point = float(np.mean([np.mean(reductions[seed]) for seed in SEEDS]))
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for boot_idx in range(BOOTSTRAP_SAMPLES):
        selected_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        seed_means = []
        for seed in selected_seeds:
            values = reductions[int(seed)]
            case_indices = rng.integers(0, values.size, size=values.size)
            seed_means.append(float(np.mean(values[case_indices])))
        samples[boot_idx] = float(np.mean(seed_means))
    low, high = np.quantile(samples, [0.025, 0.975])
    return point, float(low), float(high)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.good_fraction <= 1.0:
        raise ValueError("--good-fraction must be in (0, 1]")
    if not args.topk or any(k <= 0 for k in args.topk):
        raise ValueError("all --topk values must be positive")

    rows: list[dict] = []
    case_values: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    reference_protocol = None
    num_candidates = None

    for seed in SEEDS:
        for model, pattern in MODELS.items():
            run_name = pattern.format(seed=seed)
            path = metric_path(STABLE_WM / run_name, args.output_subdir)
            model_costs, env_costs, protocol = load_raw(path)
            if model_costs.shape != (64, 300):
                raise RuntimeError(
                    f"paper shared bank must have shape (64, 300), got {model_costs.shape} in {path}"
                )
            hardstart_success = load_hardstart_success(
                STABLE_WM / run_name, args.success_pattern
            )
            if reference_protocol is None:
                reference_protocol = protocol
                num_candidates = int(model_costs.shape[1])
            elif protocol != reference_protocol:
                raise RuntimeError(f"shared-bank protocol mismatch: {run_name}")

            assert num_candidates is not None
            good_count = max(1, int(math.ceil(args.good_fraction * num_candidates)))
            for elite_size in args.topk:
                if elite_size > num_candidates:
                    raise ValueError(f"top-k {elite_size} exceeds N={num_candidates}")
                regret, elite_mean_regret, good_hit, best_hit, cad = case_metrics(
                    model_costs, env_costs, elite_size, good_count
                )
                case_values[(model, seed, elite_size)] = {
                    "regret": regret,
                    "elite_mean_regret": elite_mean_regret,
                }
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "num_candidates": num_candidates,
                        "elite_size": elite_size,
                        "elite_fraction": elite_size / num_candidates,
                        "good_fraction_requested": args.good_fraction,
                        "good_count": good_count,
                        "random_good_hit": random_hit_probability(
                            num_candidates, elite_size, good_count
                        ),
                        "random_best_hit": elite_size / num_candidates,
                        "hardstart_success": hardstart_success,
                        "cad_mean": float(np.mean(cad)),
                        "regret_mean": float(np.mean(regret)),
                        "elite_mean_regret_mean": float(np.mean(elite_mean_regret)),
                        "good_hit_mean": float(np.mean(good_hit)),
                        "best_hit_mean": float(np.mean(best_hit)),
                    }
                )

    assert num_candidates is not None
    aggregates = []
    for model in MODELS:
        for elite_size in args.topk:
            selected = [
                row for row in rows if row["model"] == model and row["elite_size"] == elite_size
            ]
            entry = {
                "model": model,
                "num_candidates": num_candidates,
                "elite_size": elite_size,
                "elite_fraction": elite_size / num_candidates,
                "good_count": selected[0]["good_count"],
                "random_good_hit": selected[0]["random_good_hit"],
                "random_best_hit": selected[0]["random_best_hit"],
            }
            for metric in (
                "hardstart_success",
                "cad_mean",
                "regret_mean",
                "elite_mean_regret_mean",
                "good_hit_mean",
                "best_hit_mean",
            ):
                seed_values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                entry[f"{metric}_across_seed_mean"] = float(np.mean(seed_values))
                entry[f"{metric}_across_seed_pop_std"] = float(np.std(seed_values, ddof=0))
            aggregates.append(entry)

    cad_rows = [row for row in rows if row["elite_size"] == args.topk[0]]
    cad_correlation = stats.spearmanr(
        [row["cad_mean"] for row in cad_rows],
        [row["hardstart_success"] for row in cad_rows],
    )
    global_cad = {
        "num_model_seed_points": len(cad_rows),
        "spearman_rho": float(cad_correlation.statistic),
        "spearman_p": float(cad_correlation.pvalue),
    }

    correlations = {}
    for elite_size in args.topk:
        selected = [row for row in rows if row["elite_size"] == elite_size]
        success = np.asarray([row["hardstart_success"] for row in selected])
        correlations[str(elite_size)] = {}
        for metric in ("regret_mean", "elite_mean_regret_mean"):
            oriented = -np.asarray([row[metric] for row in selected])
            spearman = stats.spearmanr(oriented, success)
            kendall = stats.kendalltau(oriented, success)
            correlations[str(elite_size)][metric] = {
                "orientation": "negative regret (higher is better)",
                "num_model_seed_points": len(selected),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_p": float(kendall.pvalue),
            }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paired_bootstrap = {}
    for elite_size in args.topk:
        paired_bootstrap[str(elite_size)] = {}
        for target in tuple(MODELS)[1:]:
            paired_bootstrap[str(elite_size)][target] = {}
            for metric in ("regret", "elite_mean_regret"):
                baseline = {
                    seed: case_values[("LeWM", seed, elite_size)][metric] for seed in SEEDS
                }
                comparison = {
                    seed: case_values[(target, seed, elite_size)][metric] for seed in SEEDS
                }
                point, low, high = hierarchical_paired_reduction_ci(
                    baseline, comparison, rng
                )
                paired_bootstrap[str(elite_size)][target][metric] = {
                    "reduction_LeWM_minus_target": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_samples": BOOTSTRAP_SAMPLES,
                    "bootstrap_unit": "seeds, then paired cases within seed",
                }

    adjacent_bootstrap = {}
    adjacent_pairs = tuple(zip(tuple(MODELS)[:-1], tuple(MODELS)[1:]))
    for elite_size in args.topk:
        adjacent_bootstrap[str(elite_size)] = {}
        for source, target in adjacent_pairs:
            comparison_name = f"{source} -> {target}"
            adjacent_bootstrap[str(elite_size)][comparison_name] = {}
            for metric in ("regret", "elite_mean_regret"):
                source_values = {
                    seed: case_values[(source, seed, elite_size)][metric] for seed in SEEDS
                }
                target_values = {
                    seed: case_values[(target, seed, elite_size)][metric] for seed in SEEDS
                }
                point, low, high = hierarchical_paired_reduction_ci(
                    source_values, target_values, rng
                )
                adjacent_bootstrap[str(elite_size)][comparison_name][metric] = {
                    "reduction_source_minus_target": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_samples": BOOTSTRAP_SAMPLES,
                    "bootstrap_unit": "seeds, then paired cases within seed",
                }

    args.result_dir.mkdir(parents=True, exist_ok=True)
    tag = f"n{num_candidates}_{args.output_subdir}"
    csv_path = args.result_dir / f"elite_metric_audit_{tag}.csv"
    json_path = args.result_dir / f"elite_metric_audit_{tag}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "definition": {
                    "good_hit": "predicted top-k intersects realized top-ceil(q*N)",
                    "good_fraction": args.good_fraction,
                    "best_hit": "predicted top-k contains the realized-best candidate",
                    "regret": "normalized realized-cost loss of the best predicted-elite candidate",
                    "elite_mean_regret": "normalized mean-cost gap between predicted and realized top-k sets",
                    "random_baseline": "exact uniform-subset hypergeometric probability",
                },
                "per_seed": rows,
                "aggregate": aggregates,
                "global_cad": global_cad,
                "correlations": correlations,
                "paired_bootstrap": paired_bootstrap,
                "adjacent_bootstrap": adjacent_bootstrap,
            },
            handle,
            indent=2,
        )

    print(f"N={num_candidates}; wrote {csv_path} and {json_path}")
    for elite_size in args.topk:
        first = next(row for row in aggregates if row["elite_size"] == elite_size)
        print(
            f"k={elite_size} ({elite_size / num_candidates:.1%}); "
            f"good=top-{first['good_count']}; "
            f"random good-hit={first['random_good_hit']:.3f}; "
            f"random best-hit={first['random_best_hit']:.3f}"
        )
        for row in (item for item in aggregates if item["elite_size"] == elite_size):
            print(
                f"  {row['model']:<14} "
                f"regret={row['regret_mean_across_seed_mean']:.4f}"
                f"+/-{row['regret_mean_across_seed_pop_std']:.4f}  "
                f"elite-mean={row['elite_mean_regret_mean_across_seed_mean']:.4f}  "
                f"good-hit={row['good_hit_mean_across_seed_mean']:.3f}  "
                f"best-hit={row['best_hit_mean_across_seed_mean']:.3f}"
            )
        corr = correlations[str(elite_size)]
        print(
            f"  {corr['regret_mean']['num_model_seed_points']}-point Spearman: "
            f"best-in-elite={corr['regret_mean']['spearman_rho']:.3f} "
            f"(p={corr['regret_mean']['spearman_p']:.4g}); "
            f"elite-mean={corr['elite_mean_regret_mean']['spearman_rho']:.3f} "
            f"(p={corr['elite_mean_regret_mean']['spearman_p']:.4g})"
        )
        for target in tuple(MODELS)[1:]:
            stat = paired_bootstrap[str(elite_size)][target]["regret"]
            print(
                f"  LeWM->{target} regret reduction={stat['reduction_LeWM_minus_target']:.4f} "
                f"95% CI [{stat['ci95_low']:.4f}, {stat['ci95_high']:.4f}]"
            )


if __name__ == "__main__":
    main()
