#!/usr/bin/env python3
"""Run the paper's simulation training or evaluation matrix.

Examples:
  python scripts/run_paper_simulation.py train --group main --env cube
  python scripts/run_paper_simulation.py eval --group main --env cube --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3072, 4096, 6144)
MAIN = {
    "cube": "cube_res_inv_detachact_mifixedvar",
    "reacher": "reacher_res_inv_detachact_mifixedvar",
    "tworoom": "tworoom_res_inv_detachact_mifixedvar",
    "pusht": "pusht_res_inv_detachact_mifixedvar",
    "scene": "scene_res_inv_detachact_mifixedvar_w0001",
}
BASELINE = {env: f"{env}_abs" for env in MAIN}
CUBE_ABLATIONS = (
    "cube_abs", "cube_inv_detachact_mifixedvar_w01", "cube_res",
    "cube_res_inv_detachact", "cube_res_mifixedvar",
    "cube_res_inv_detachact_mifixedvar",
    "cube_res_inv_trueconcat_detachact",
    "cube_res_inv_trueconcat_detachact_mifixedvar",
    "cube_res_inv_disp_detachact",
    "cube_res_inv_disp_detachact_mifixedvar",
    "cube_res_inv_detachact_mifixedvar_w005_kl01",
    "cube_res_inv_detachact_mifixedvar_w015",
    "cube_res_inv_detachact_mifixedvar_w03",
    "cube_res_inv_detachact_mifixedvar_w05",
    "cube_res_inv_detachact_mifixedvar_invw005",
    "cube_res_inv_detachact_mifixedvar_invw02",
)
EVAL_CONFIGS = {
    "cube": ("cube_original_benchmark",) + tuple(
        f"cube_ood_table_perturb0{i}_hardstart" for i in range(5)),
    "reacher": ("reacher",),
    "tworoom": ("tworoom",),
    "pusht": ("pusht",),
    "scene": ("scene_hardstart_balanced",),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--group", choices=("main", "baseline", "cube-ablations"), default="main")
    parser.add_argument("--env", choices=tuple(MAIN) + ("all",), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.group == "cube-ablations" and args.env not in ("cube", "all"):
        parser.error("cube-ablations applies only to Cube")
    if any(seed not in SEEDS for seed in args.seeds):
        parser.error(f"paper seeds are {SEEDS}")
    environments = list(MAIN) if args.env == "all" else [args.env]
    if args.group == "cube-ablations":
        environments = ["cube"]
    cache = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))

    for env in environments:
        patterns = (CUBE_ABLATIONS if args.group == "cube-ablations" else
                    (MAIN[env] if args.group == "main" else BASELINE[env],))
        for pattern in patterns:
            for seed in args.seeds:
                name = f"{pattern}_seed{seed}_seedfix"
                if args.mode == "train":
                    cmd = [sys.executable, "train.py", f"experiment={name}"]
                    if not (ROOT / "config/train/experiment" / f"{name}.yaml").is_file():
                        raise FileNotFoundError(f"Missing experiment config: {name}")
                    print(" ".join(cmd), flush=True)
                    if not args.dry_run:
                        subprocess.run(cmd, cwd=ROOT, check=True)
                else:
                    checkpoint = cache / name / f"{name}_epoch_10_object.ckpt"
                    if not args.dry_run and not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                    for config in EVAL_CONFIGS[env]:
                        cmd = [sys.executable, "eval.py", f"--config-name={config}",
                               f"policy={name}/{name}_epoch_10"]
                        if env == "cube":
                            if config == "cube_original_benchmark":
                                cmd.append("eval.start_manifest=protocols/cube_original.npz")
                                cmd.extend(("output.filename=fixed_protocol_original.txt",
                                            "output.structured_filename=fixed_protocol_original.npz"))
                            else:
                                cmd.append("eval.start_manifest=protocols/cube_hardstart.npz")
                                radius = config.split("perturb0", 1)[1][0]
                                cmd.extend((f"output.filename=fixed_protocol_p0{radius}.txt",
                                            f"output.structured_filename=fixed_protocol_p0{radius}.npz"))
                        elif env == "scene":
                            cmd.append("eval.start_manifest=protocols/scene_hardstart_balanced.npz")
                            cmd.extend(("output.filename=fixed_protocol_scene.txt",
                                        "output.structured_filename=fixed_protocol_scene.npz"))
                        print(" ".join(cmd), flush=True)
                        if not args.dry_run:
                            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
