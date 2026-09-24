#!/usr/bin/env python3
"""Rerun Cube P00--P04 on one fixed, protocol-valid set of 50 starts.

Jobs are restart safe and write `fixed_protocol_*` results beside checkpoints,
without modifying historical result files. Headline jobs are ordered first.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3072, 4096, 6144)
HEADLINE = (
    "cube_abs",
    "cube_res_inv_detachact_mifixedvar",
)
COMPONENTS = (
    "cube_abs",
    "cube_inv_detachact_mifixedvar_w01",
    "cube_res",
    "cube_res_inv_detachact",
    "cube_res_mifixedvar",
    "cube_res_inv_detachact_mifixedvar",
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


def output_path(cache: Path, run: str, radius: int) -> Path:
    return cache / run / f"fixed_protocol_p0{radius}.npz"


def validate(path: Path, manifest: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(manifest, allow_pickle=False) as expected, np.load(path, allow_pickle=False) as result:
            return (
                result["episode_successes"].shape == (50,)
                and np.array_equal(result["eval_episodes"], expected["eval_episodes"])
                and np.array_equal(result["eval_start_idx"], expected["eval_start_idx"])
                and np.isfinite(float(result["success_rate"]))
            )
    except Exception:
        return False


def run_job(job, gpu: int, args, manifest: Path, cache: Path, log_dir: Path):
    variant, seed, radius = job
    run = f"{variant}_seed{seed}_seedfix"
    checkpoint = cache / run / f"{run}_epoch_10_object.ckpt"
    result = output_path(cache, run, radius)
    if validate(result, manifest):
        return {"run": run, "radius": radius, "status": "existing", "gpu": gpu}
    if not checkpoint.is_file():
        return {"run": run, "radius": radius, "status": "missing_checkpoint", "gpu": gpu}

    command = [
        sys.executable, "eval.py", f"--config-name=cube_ood_table_perturb0{radius}_hardstart",
        f"cache_dir={cache}", f"policy={run}/{run}_epoch_10",
        f"eval.start_manifest={manifest}", "output.save_video=false",
        f"output.filename=fixed_protocol_p0{radius}.txt",
        f"output.structured_filename=fixed_protocol_p0{radius}.npz",
    ]
    log_path = log_dir / f"{run}.p0{radius}.log"
    if args.dry_run:
        print(f"GPU {gpu}: {' '.join(command)}")
        return {"run": run, "radius": radius, "status": "dry_run", "gpu": gpu}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),
               MPLCONFIGDIR=str(log_dir / f"mpl-gpu{gpu}"), MUJOCO_GL="egl",
               PYTHONDONTWRITEBYTECODE="1")
    if args.egl_devices is not None:
        env["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_devices[args.gpu_ids.index(gpu)])
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
    status = "complete" if completed.returncode == 0 and validate(result, manifest) else "failed"
    return {"run": run, "radius": radius, "status": status, "gpu": gpu,
            "returncode": completed.returncode, "log": str(log_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("headline", "components", "all"), default="headline")
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU IDs")
    parser.add_argument("--egl-devices", help="Optional comma-separated EGL device IDs, one per GPU; EGL IDs are independent of CUDA IDs")
    parser.add_argument(
        "--radii",
        default="0,1,2,3,4",
        help="Comma-separated Cube perturbation IDs from 0 through 4",
    )
    parser.add_argument("--cache-root", type=Path,
                        default=Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    variants = {"headline": HEADLINE, "components": COMPONENTS, "all": ALL}[args.group]
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)):
        parser.error("--gpus must contain distinct IDs")
    args.gpu_ids = gpus
    if args.egl_devices is not None:
        try:
            args.egl_devices = [int(value) for value in args.egl_devices.split(",")]
        except ValueError:
            parser.error("--egl-devices must contain integers")
        if len(args.egl_devices) != len(gpus) or any(x < 0 for x in args.egl_devices):
            parser.error("--egl-devices requires one nonnegative EGL ID per GPU")
    try:
        radii = [int(value) for value in args.radii.split(",")]
    except ValueError:
        parser.error("--radii must be a comma-separated list of integers")
    if not radii or len(radii) != len(set(radii)) or not set(radii) <= set(range(5)):
        parser.error("--radii must contain distinct values from 0 through 4")
    manifest = (ROOT / "protocols" / "cube_hardstart.npz").resolve()
    log_dir = args.cache_root / "fixed_protocol_logs"
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(variant, seed, radius) for variant in variants for seed in SEEDS for radius in radii]
    # One serial worker per GPU. Each worker receives every Nth task.
    shards = [jobs[index::len(gpus)] for index in range(len(gpus))]

    def worker(gpu, shard):
        return [run_job(job, gpu, args, manifest, args.cache_root, log_dir) for job in shard]

    records = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu, shard) for gpu, shard in zip(gpus, shards)]
        for future in as_completed(futures):
            for record in future.result():
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    summary = {status: sum(record["status"] == status for record in records)
               for status in sorted({record["status"] for record in records})}
    print(json.dumps({"group": args.group, "jobs": len(jobs), "summary": summary}, indent=2))
    if any(record["status"] in {"failed", "missing_checkpoint"} for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
