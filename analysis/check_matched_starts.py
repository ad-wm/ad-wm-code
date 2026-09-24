#!/usr/bin/env python3
"""Compare structured evaluation starts with a fixed paper protocol manifest."""

import argparse
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3072, 4096, 6144)
COMPONENTS = (
    "cube_abs", "cube_inv_detachact_mifixedvar_w01",
    "cube_res", "cube_res_inv_detachact",
    "cube_res_mifixedvar", "cube_res_inv_detachact_mifixedvar",
)
ALL = COMPONENTS + (
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path,
                        default=Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm")))
    parser.add_argument("--group", choices=("components", "all"), default="components")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on missing or mismatched results")
    args = parser.parse_args()
    variants = COMPONENTS if args.group == "components" else ALL
    manifest = ROOT / "protocols" / "cube_hardstart.npz"
    with np.load(manifest, allow_pickle=False) as data:
        expected_episodes = data["eval_episodes"]
        expected_steps = data["eval_start_idx"]

    failures = 0
    for variant in variants:
        for seed in SEEDS:
            run = f"{variant}_seed{seed}_seedfix"
            for radius in range(5):
                path = args.cache_root / run / f"fixed_protocol_p0{radius}.npz"
                if not path.is_file():
                    print(f"MISSING P0{radius}: {run}")
                    failures += 1
                    continue
                with np.load(path, allow_pickle=False) as data:
                    episodes = data["eval_episodes"]
                    steps = data["eval_start_idx"]
                    successes = data["episode_successes"]
                if len(episodes) != len(expected_episodes):
                    print(f"COUNT P0{radius}: {run} has {len(episodes)} starts")
                    failures += 1
                    continue
                if successes.shape != (50,) or not np.all(np.isfinite(successes)):
                    print(f"INVALID P0{radius}: {run} has invalid success values")
                    failures += 1
                    continue
                different = int(np.count_nonzero((episodes != expected_episodes) |
                                                 (steps != expected_steps)))
                if different:
                    print(f"MISMATCH P0{radius}: {run} differs on {different}/50 starts")
                    failures += 1
    print(f"Fixed Cube protocol: {failures} missing/mismatched run-protocol results")
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
