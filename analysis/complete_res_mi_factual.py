#!/usr/bin/env python3
"""Complete default-weight Res+MI factual MSE on the original diagnostic split.

Keeps the original 28-step clip index and seed-3072 validation split (61,999
clips), then reads only the first 8 frames needed for the 3-frame context and
5-step factual rollout. Uses the original preprocessing and rollout routine.
Recomputes LeWM seed 3072 as a full-split numerical check against its cache.
No training or planning evaluation is performed.
"""
import os
import sys
import time
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-factual-completion-mpl")

import numpy as np
import torch
from omegaconf import OmegaConf
from analyze_residual_diagnostics import build_val_loader, free_rollout_metrics, load_model_object


def main():
    torch.set_num_threads(4)
    cfg = OmegaConf.create(dict(
        seed=3072, cache_dir=os.environ.get("STABLEWM_HOME"), img_size=224, train_split=.9,
        analysis=dict(batch_size=64, num_workers=4),
        data=dict(dataset=dict(name="ogbench/cube_single_expert", num_steps=28,
                               frameskip=5, keys_to_load=["pixels", "action"],
                               keys_to_cache=["action"])),
    ))
    loader = build_val_loader(cfg)
    assert len(loader.dataset) == 61999
    # clip_indices and the already-created Subset stay exactly unchanged.
    ds = loader.dataset.dataset
    ds.num_steps = 8
    ds.span = 40
    index_hash = hashlib.sha256(np.asarray(loader.dataset.indices, dtype=np.int64).tobytes()).hexdigest()
    names = ["cube_abs_seed3072_seedfix"] + [f"cube_res_mifixedvar_seed{s}_seedfix" for s in [3072, 4096, 6144]]
    models = {}
    for name in names:
        path = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm")) / name / f"{name}_epoch_10_object.ckpt"
        models[name] = load_model_object(path, "cuda")
    values = {name: [] for name in names}
    started = time.time()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            pixels = batch["pixels"].cuda(non_blocking=True)
            action = torch.nan_to_num(batch["action"].cuda(non_blocking=True), 0.)
            for name, model in models.items():
                emb = model.encode({"pixels": pixels})["emb"]
                metrics = free_rollout_metrics(model, emb, action, 3, 5, 1e-8)
                values[name].append(metrics["rollout_mse"])
            if i % 25 == 0 or i + 1 == len(loader):
                elapsed = time.time() - started
                eta = elapsed / (i + 1) * (len(loader) - i - 1)
                print(f"{i+1}/{len(loader)} batches; elapsed {elapsed:.0f}s; ETA {eta:.0f}s", flush=True)
    out = Path(__file__).resolve().parent / "results/res_mi_factual_completion"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"validation_split_seed": 3072, "validation_clips": 61999,
               "validation_index_sha256": index_hash, "original_clip_steps": 28,
               "loaded_clip_steps": 8, "frameskip": 5, "history_size": 3,
               "rollout_steps": 5, "precision": "float32", "models": {}}
    for name in names:
        mse = np.concatenate(values[name], axis=0)
        assert mse.shape == (61999, 5) and np.isfinite(mse).all()
        by_horizon = mse.mean(axis=0)
        summary["models"][name] = dict(one_step_mse=float(by_horizon[0]),
                                        local_mse=float(by_horizon.mean()),
                                        mse_by_horizon=by_horizon.tolist())
        np.savez_compressed(out / f"{name}.npz", rollout_mse=mse, rollout_mse_mean=by_horizon)
    reference = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm")) / names[0] / "residual_diagnostics" / f"{names[0]}_epoch_10_object_diagnostics_epoch_010.npz"
    with np.load(reference) as old:
        expected = old["rollout_mse_mean"][:5]
    actual = np.asarray(summary["models"][names[0]]["mse_by_horizon"])
    summary["reference_max_absolute_error"] = float(np.max(np.abs(actual - expected)))
    assert np.allclose(actual, expected, rtol=1e-4, atol=1e-7), (actual, expected)
    for metric in ["one_step_mse", "local_mse"]:
        x = np.array([summary["models"][n][metric] for n in names[1:]]) * 1000
        summary[metric + "_x1000"] = dict(mean=float(x.mean()), population_sd=float(x.std(ddof=0)))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
