# AD-WM: action-discriminative world models for counterfactual MPC

This directory is the **simulation release** for the ICRA paper. It is based on [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm), retaining its encoder, predictor backbone, data interface, and CEM planner. AD-WM trains a residual latent predictor with inverse dynamics and normalized action recovery. The two auxiliary heads are used during training; evaluation plans with the encoder, action encoder, and residual predictor. The LeWM code is MIT licensed; see [LICENSE](LICENSE).

The main simulation setting is **Res + Inv (weight 0.1) + MI (weight 0.01)**. Scene uses MI weight **0.0001**. The paper's Cube ablations, sensitivity sweeps, cross-environment results, hard-start protocols, and planning diagnostics also require the baseline/control configurations included here. Only the 15 main-model checkpoints (five environments × three seeds) are staged locally. Control models are reproducible from their training configurations.

Real-robot deployment, Franka evaluation, and V-JEPA 2/DROID post-training code and weights are outside this release. External Fast-LeWM, Sub-JEPA, and INTACT implementations and weights belong to their respective authors; the paper uses their native inference interfaces. Their 66 Cube evaluation files are represented here as validated per-episode records, with source hashes and table statistics in [external/README.md](external/README.md). Re-running those external models requires the authors' code and weights. Real-robot experiments require the separate robot setup.

## Contents

- `train.py`, `jepa.py`, `module.py`, `utils.py`: simulation training and world model. Training logs only epoch-level total, prediction, SIGReg, inverse, and MI losses; it does not compute the abandoned action-NCE/counterfactual objectives or per-epoch inverse-gradient probes.
- `config/train/experiment/`: the three-seed main, baseline, Cube ablation, and weight-sensitivity configurations used for paper tables. Names match the original run IDs.
- `eval.py`, `config/eval/`: dataset-backed closed-loop evaluation, including Cube Original/P00–P04 and balanced Scene starts. `.npz` results retain selected episode/start indices and per-episode success.
- `protocols/`: fixed episode/start manifests used to keep all matched evaluations on identical cases.
- `scripts/generate_ogbench_manipspace.py`, `ogbench_scene.py`: Scene collection and state/goal integration. `scripts/convert_ogbench_manipspace_to_swm_hdf5.py` converts an existing OGBench NPZ replay if needed.
- `analyze_*.py`, `analysis/`, `scripts/run_cube_shared_bank.py`: factual-rollout, CEM landscape, 300-candidate shared-bank, bootstrap, and configuration-shift diagnostics.
- `results/external_cube_*.csv`, `analysis/external_cube_results.py`, `external/README.md`: external Cube per-episode records, exact-start/perturbation checks, paper-table statistics, and upstream provenance.
- `checkpoints/MANIFEST.csv`: names, sizes, and SHA-256 hashes of the 15 staged main-model object checkpoints. Each run also includes a resolved `config.yaml` for the diagnostic tools. Binary checkpoints are excluded from Git and are distributed in the [AD-WM model repository](https://huggingface.co/ad-wm/ad-wm).

## Environment

Follow the [official LeWM installation](https://github.com/lucas-maes/le-wm#using-the-code) for the base environment. The simulation runs here used Python 3.10, PyTorch 2.6.0+cu124, `stable-worldmodel==0.0.6`, and `stable-pretraining==0.1.6`; `requirements.txt` pins the other key versions, including `datasets==4.0.0` required by the training dependency. Install a PyTorch build appropriate to the host first, then:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt
export MUJOCO_GL=egl
export STABLEWM_HOME="$PWD/checkpoints"
```

`STABLEWM_HOME` must contain both datasets and checkpoint run directories. It defaults to `~/.stable-wm` if unset. Pin model computation with `CUDA_VISIBLE_DEVICES=<physical GPU>`. EGL enumerates rendering devices independently: leave `MUJOCO_EGL_DEVICE_ID` unset for automatic selection, or set an EGL ID verified on your host. The Cube batch runner accepts `--egl-devices` with one EGL ID per listed GPU. The checkpoint objects use Python pickle; load only files from trusted sources.

## Data

Download the original LeWM datasets from the [official data collection](https://huggingface.co/collections/quentinll/lewm) and extract their `.h5` files under `$STABLEWM_HOME` as described in the [upstream README](https://github.com/lucas-maes/le-wm#data). The paths expected here are:

| Environment | HDF5 path below `$STABLEWM_HOME` |
| --- | --- |
| Cube | `ogbench/cube_single_expert.h5` |
| Reacher | `dmc/reacher.h5` |
| TwoRoom | `tworooms/tworoom.h5` |
| PushT | `pusht/pusht_expert_train.h5` |
| Scene | `ogbench/scene_play_lewm_10k_200.h5` (generate below) |

Scene is an addition to the official LeWM benchmark and needs its own dataset. The collection script records 10,000 200-step play episodes at 224×224 with the Markov oracle, action noise 0.1, seed 0, and an optional 1,000-episode validation file. Training uses a seeded 90/10 split of the main HDF5, not the extra `_val.h5` file.

```bash
python scripts/generate_ogbench_manipspace.py \
  --env_name scene-v0 --dataset_type play \
  --num_episodes 10000 --max_episode_steps 200 \
  --dataset_name ogbench/scene_play_lewm_10k_200 \
  --cache_dir "$STABLEWM_HOME"
```

Keep the generated `qpos`, `qvel`, button state, and privileged component columns: balanced Scene evaluation uses them to restore and select starts/goals. The script supplies an `episode_idx` alias when the recorder writes `ep_idx`.

## Train

The paper uses 224×224 images, history 3, frame skip 5, a trainable ViT-tiny encoder with latent dimension 192, a depth-6/16-head predictor, bf16, 10 epochs, AdamW, batch size 128, learning rate `5e-5`, weight decay `1e-3`, and SIGReg weight `0.09`. MI uses a detached, batch-standardized action embedding target and fixed unit-variance posterior/prior with KL coefficient `0.01`.

Run one exact paper configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py experiment=cube_res_inv_detachact_mifixedvar_seed3072_seedfix
CUDA_VISIBLE_DEVICES=0 python train.py experiment=scene_res_inv_detachact_mifixedvar_w0001_seed3072_seedfix
```

Or run a complete matrix (sequentially; use a scheduler or separate GPUs for parallel jobs):

```bash
python scripts/run_paper_simulation.py train --group main
python scripts/run_paper_simulation.py train --group baseline
python scripts/run_paper_simulation.py train --group cube-ablations --env cube
```

Training defaults to one GPU per run, matching the paper's simulation setup and SIGReg implementation. Use `CUDA_VISIBLE_DEVICES` to assign different runs to different GPUs.

Use `--dry-run` to print commands. Each run writes `config.yaml`, five epoch-level training/validation losses, the final epoch-10 object checkpoint, and the final weights checkpoint under `$STABLEWM_HOME/<run_name>/`. The control configurations explicitly disable missing objectives, so their results are not affected by the default main-model settings.

## Closed-loop evaluation

The checkpoint policy is a path **relative to `$STABLEWM_HOME` without `_object.ckpt`**, following the [official LeWM evaluation convention](https://github.com/lucas-maes/le-wm#planning). For example:

```bash
python eval.py --config-name=cube_original_benchmark \
  policy=cube_res_inv_detachact_mifixedvar_seed3072_seedfix/cube_res_inv_detachact_mifixedvar_seed3072_seedfix_epoch_10 \
  eval.start_manifest=protocols/cube_original.npz
python eval.py --config-name=cube_ood_table_perturb04_hardstart \
  policy=cube_res_inv_detachact_mifixedvar_seed3072_seedfix/cube_res_inv_detachact_mifixedvar_seed3072_seedfix_epoch_10 \
  eval.start_manifest=protocols/cube_hardstart.npz
python eval.py --config-name=scene_hardstart_balanced \
  policy=scene_res_inv_detachact_mifixedvar_w0001_seed3072_seedfix/scene_res_inv_detachact_mifixedvar_w0001_seed3072_seedfix_epoch_10 \
  eval.start_manifest=protocols/scene_hardstart_balanced.npz
```

The evaluator uses seed 42. Each Cube checkpoint has 50 episodes for Original and each P00–P04 protocol. Original samples dataset states; the hard-start protocol requires tabletop/no contact, goal distance greater than 0.05, and end-effector–cube distance greater than 0.02. P00–P04 apply cube XY radii 0/.01/.02/.03/.04, clipped to X [0.30,0.55] and Y [-0.30,0.30]. HS is the mean of those five success rates. Scene balances 200 episodes across Button, Cube, Drawer, Window (50 each), with one changed component, tolerance .04, and minimum geometry filters .08. Other environments use the original LeWM protocols. CEM uses 300 samples, 30 elites, 30 iterations, horizon 5, receding horizon 5, action block 5, terminal latent distance, and a 50-step budget for Cube/Scene.

The matrix runner evaluates all required protocols for the requested models:

```bash
python scripts/run_paper_simulation.py eval --group main
python scripts/run_paper_simulation.py eval --group baseline
python scripts/run_paper_simulation.py eval --group cube-ablations --env cube
```

The runner fixes Cube and Scene starts with `protocols/` manifests. The Cube hard-start manifest satisfies the configured filters and is reused unchanged for every model, training seed, and perturbation radius.

To run or resume the fixed-manifest Cube P00–P04 matrix, use one serial worker per listed GPU. Results are written as `fixed_protocol_p00.npz` through `fixed_protocol_p04.npz`, so existing evaluation files remain available.

```bash
python scripts/run_cube_fixed_protocol.py --group all --gpus 0,1,2,3
python analysis/summarize_cube_fixed_protocol.py
```

The `headline` group contains LeWM and AD-WM, `components` contains the six component-ablation variants, and `all` additionally contains the inverse-input and loss-weight sweeps. Completed valid files are skipped when a run is resumed. The summarizer validates the episode/start pairs and perturbations and writes per-seed and aggregate results under the ignored `analysis/results/` directory. Rates are percentages; SD is the population SD across three training seeds.

For the external Cube comparison, run `python analysis/external_cube_results.py` to validate the 3,300 packaged per-episode records and rebuild the 30 table cells; see [external/README.md](external/README.md) for upstream versions and model sources.

The run directories contain text results and structured `.npz` files. The evaluator stores per-episode success, selected dataset episode/start, seed, perturbation, and Scene component information. For new publication statistics, take the arithmetic mean and **population** SD across the three training seeds, and verify episode/start arrays are identical across models.

```bash
python analysis/check_matched_starts.py --strict
# After running the full sensitivity matrix:
python analysis/check_matched_starts.py --group all --strict
```

## Diagnostics and paper tables

- **Factual prediction and local rollouts:** `python analyze_residual_diagnostics.py checkpoints.run_dir=<run_name>`. The Cube config uses a three-frame context, 25-step rollout, and first-five-step local MSE. Run for the five default-weight Cube variants and three seeds. `analysis/complete_res_mi_factual.py` contains the full 61,999-clip Res+MI completion used in the paper; it expects the LeWM reference diagnostic `.npz` first.
- **Shared candidate bank:** `python scripts/run_cube_shared_bank.py`. The script evaluates 64 matched Cube cases for LeWM, Res, Res+Inv, Res+MI, and AD-WM, each at three seeds. Each bank has expert, zero, negated, reversed, plus 99 near, 99 medium, and 98 far candidates. It reuses one saved candidate cache across all models and scores environment-realized endpoints in each model's encoder space. `analysis/analyze_elite_metrics.py` recomputes CAD, normalized best-in-elite and elite-mean regret at k=15/30/60 and 20,000 paired hierarchical bootstrap resamples; `analysis/bootstrap_regret_chain.py` summarizes the paper's k=30 chain.
- **Local CEM landscape:** `python analyze_planner_alignment.py checkpoints.run_dir=<run_name>` uses 20 cases and a 31×31 grid in the final elite PCA plane. Select the five default-weight Cube runs for comparison.
- **Configuration-shift support:** after baseline Cube P00–P04 `.npz` results exist, run `python analysis/audit_cube_perturbation_support.py` then `python analysis/audit_cube_arm_joint_support.py`. The latter uses 20,000 tabletop/non-contact training states per seed and cross-episode nearest neighbors for joint cube/arm and relative-pose distances.

These diagnostic tools are offline analyses. They do not alter training or the deployed MPC solver.

## Release artifacts and limits

The staged checkpoint manifest covers the five environment-specific main models at seeds 3072, 4096, and 6144. The 15 object checkpoints (about 1.16 GB) are hosted at [ad-wm/ad-wm](https://huggingface.co/ad-wm/ad-wm), not in Git. Download them with `hf download ad-wm/ad-wm --local-dir "$STABLEWM_HOME"`, then run `python scripts/verify_checkpoint_manifest.py --root "$STABLEWM_HOME"` to check all hashes and resolved configs. Train or obtain the ablation checkpoints before rerunning their tables and the shared-bank analysis. The release has no real-robot code or laboratory data, and no external-method checkpoints. Hardware/library variation and regenerated Scene trajectories may change exact percentages.
