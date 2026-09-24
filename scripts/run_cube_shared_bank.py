#!/usr/bin/env python3
"""Build and score the shared Cube N=300, k=30 candidate bank for 15 models."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_paper_simulation import ROOT, SEEDS


MODELS = (
    "cube_abs", "cube_res", "cube_res_inv_detachact",
    "cube_res_mifixedvar", "cube_res_inv_detachact_mifixedvar",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cache_root = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    bank = cache_root / "counterfactual_shared_planner_cache" / "cube_seed3072_n64_balanced300_visual_latent_v1.npz"
    for model in MODELS:
        for seed in SEEDS:
            name = f"{model}_seed{seed}_seedfix"
            checkpoint = cache_root / name / f"{name}_epoch_10_object.ckpt"
            if not args.dry_run and not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            command = [
                sys.executable, "analyze_counterfactual_metrics.py",
                f"checkpoints.run_dir={name}",
                "analysis.output_subdir=counterfactual_metrics_visual_latent_n300",
                "analysis.one_step.enabled=false", "analysis.rollout.enabled=false",
                "analysis.planner.enabled=false", "analysis.shared_planner.enabled=true",
                "analysis.shared_planner.num_eval=64", "analysis.shared_planner.topk=30",
                "analysis.shared_planner.env_cost_mode=visual_latent",
                "analysis.shared_planner.near_count=99", "analysis.shared_planner.mid_count=99",
                "analysis.shared_planner.far_count=98", "analysis.shared_planner.random_count=0",
                f"analysis.shared_planner.cache_path={bank}",
            ]
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
    summary = [sys.executable, "analysis/analyze_elite_metrics.py",
               "--output-subdir", "counterfactual_metrics_visual_latent_n300",
               "--topk", "15", "30", "60"]
    print(" ".join(summary), flush=True)
    if not args.dry_run:
        subprocess.run(summary, cwd=ROOT, check=True)
        subprocess.run([sys.executable, "analysis/bootstrap_regret_chain.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
