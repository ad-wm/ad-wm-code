# AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control

Official simulation code for **AD-WM**, an action-discriminative latent world model for counterfactual model predictive control (MPC).

AD-WM builds on [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm). It combines residual latent dynamics with inverse dynamics and a normalized action-recovery objective motivated by conditional mutual information. These objectives train the predictor to retain action-dependent differences between candidate futures. Their auxiliary heads are discarded at test time, so planning uses the standard latent-space CEM interface.

**Links:** [Project page](https://ad-wm.github.io/) · [Pretrained checkpoints](https://huggingface.co/ad-wm/ad-wm) · [LeWM](https://github.com/lucas-maes/le-wm)

The main simulation setting is **Res + Inv (weight 0.1) + MI (weight 0.01)**. Scene uses MI weight **0.0001**. The model repository provides the 15 main-model checkpoints (five environments × three training seeds). This code repository also contains the baseline and ablation configurations, evaluation protocols, analysis tools, and external Cube evaluation records. Real-robot code and weights are outside this release.

## Repository structure

```text
.
├── train.py, jepa.py, module.py     # Training and world-model components
├── eval.py                           # Closed-loop MPC evaluation
├── config/
│   ├── train/experiment/            # Main, baseline, ablation, and sweep runs
│   └── eval/                         # Environment and protocol settings
├── protocols/                         # Fixed evaluation start sets
├── analysis/, analyze_*.py           # Diagnostics and result summaries
├── scripts/                           # Data, training, and evaluation utilities
├── checkpoints/                       # Manifest and resolved model configs
├── results/                           # External Cube evaluation records
└── external/                          # External-baseline provenance
```

## Installation

AD-WM follows the LeWM environment and data interface. The simulation runs used Python 3.10, PyTorch 2.6.0+cu124, `stable-worldmodel==0.0.6`, and `stable-pretraining==0.1.6`. Install a PyTorch build appropriate for your CUDA setup first; `requirements.txt` pins the other key versions, including `datasets==4.0.0`.

```bash
git clone https://github.com/ad-wm/ad-wm-code.git
cd ad-wm-code
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt

export MUJOCO_GL=egl
export STABLEWM_HOME="$PWD/checkpoints"
```

`STABLEWM_HOME` must contain both datasets and checkpoint run directories. If unset, LeWM defaults to `~/.stable-wm`. For GPU selection, use `CUDA_VISIBLE_DEVICES=<physical GPU>`. EGL enumerates rendering devices separately: leave `MUJOCO_EGL_DEVICE_ID` unset for automatic selection or set an EGL ID verified on your host. The Cube batch runner accepts one EGL ID per listed GPU through `--egl-devices`. See the [upstream installation notes](https://github.com/lucas-maes/le-wm#using-the-code) for the base LeWM environment.

## Data

Download the original LeWM datasets from the [official collection](https://huggingface.co/collections/quentinll/lewm) and extract their HDF5 files under `$STABLEWM_HOME` as described in the [LeWM data instructions](https://github.com/lucas-maes/le-wm#data).

| Environment | HDF5 path below `$STABLEWM_HOME` |
| --- | --- |
| Cube | `ogbench/cube_single_expert.h5` |
| Reacher | `dmc/reacher.h5` |
| TwoRoom | `tworooms/tworoom.h5` |
| PushT | `pusht/pusht_expert_train.h5` |
| Scene | `ogbench/scene_play_lewm_10k_200.h5` (generate below) |

Scene is an addition to the LeWM benchmark. The collection script records 10,000 200-step play episodes at 224×224 with the Markov oracle, action noise 0.1, and seed 0. An optional 1,000-episode validation file can also be generated. Training uses a seeded 90/10 split of the main HDF5, rather than that extra validation file.

```bash
python scripts/generate_ogbench_manipspace.py \
  --env_name scene-v0 --dataset_type play \
  --num_episodes 10000 --max_episode_steps 200 \
  --dataset_name ogbench/scene_play_lewm_10k_200 \
  --cache_dir "$STABLEWM_HOME"
```

Keep the generated `qpos`, `qvel`, button state, and privileged component columns: balanced Scene evaluation uses them to restore and select starts and goals. The script supplies an `episode_idx` alias when the recorder writes `ep_idx`. If you already have an OGBench NPZ replay, `scripts/convert_ogbench_manipspace_to_swm_hdf5.py` converts it to the expected HDF5 layout.

## Checkpoints

Download the [AD-WM main-model checkpoints](https://huggingface.co/ad-wm/ad-wm) into the same root as the datasets, then verify their sizes, SHA-256 hashes, and resolved configs against `checkpoints/MANIFEST.csv`:

```bash
hf download ad-wm/ad-wm --local-dir "$STABLEWM_HOME"
python scripts/verify_checkpoint_manifest.py --root "$STABLEWM_HOME"
```

The archive contains Cube, Reacher, TwoRoom, PushT, and Scene models at seeds 3072, 4096, and 6144. The object checkpoints contain Python pickle data; load files only from a trusted source.

## Training

The simulation setup uses 224×224 images, history 3, frame skip 5, a trainable ViT-tiny encoder with latent dimension 192, a depth-6/16-head predictor, bf16, 10 epochs, AdamW, batch size 128, learning rate `5e-5`, weight decay `1e-3`, and SIGReg weight `0.09`. The MI objective uses a detached, batch-standardized action embedding target and fixed unit-variance posterior and prior with KL coefficient `0.01`.

Train an individual main model:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py experiment=cube_res_inv_detachact_mifixedvar_seed3072_seedfix
CUDA_VISIBLE_DEVICES=0 python train.py experiment=scene_res_inv_detachact_mifixedvar_w0001_seed3072_seedfix
```

Run configuration groups with the supplied runner:

```bash
python scripts/run_paper_simulation.py train --group main
python scripts/run_paper_simulation.py train --group baseline
python scripts/run_paper_simulation.py train --group cube-ablations --env cube
```

Use `--dry-run` to inspect commands. The runner uses one GPU per run; assign separate GPUs or a scheduler for parallel runs. Each run writes its resolved `config.yaml`, epoch-level training and validation losses, and final epoch-10 object and weights checkpoints under `$STABLEWM_HOME/<run_name>/`. The control configurations explicitly disable objectives absent from those models.

## Evaluation

A policy path is **relative to `$STABLEWM_HOME` and omits the `_object.ckpt` suffix**, following the [LeWM evaluation convention](https://github.com/lucas-maes/le-wm#planning). Examples:

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

The evaluator uses seed 42. Each Cube checkpoint is evaluated on 50 episodes for Original and each P00–P04 setting. Original samples dataset states. The Cube hard-start selection requires tabletop/no contact, goal distance above 0.05, and end-effector–cube distance above 0.02. P00–P04 apply cube XY radii 0/.01/.02/.03/.04, clipped to X [0.30, 0.55] and Y [-0.30, 0.30]; HS is their mean success rate. Balanced Scene evaluation uses 200 episodes across Button, Cube, Drawer, and Window (50 each), one changed component, tolerance 0.04, and minimum geometry filters 0.08. Other environments use the LeWM protocols.

CEM uses 300 samples, 30 elites, 30 iterations, horizon 5, receding horizon 5, action block 5, terminal latent distance, and a 50-step budget for Cube and Scene. Evaluation `.npz` files retain each selected episode and start index, per-episode success, perturbation, and Scene component information.

Run the evaluation groups:

```bash
python scripts/run_paper_simulation.py eval --group main
python scripts/run_paper_simulation.py eval --group baseline
python scripts/run_paper_simulation.py eval --group cube-ablations --env cube
```

The runner reuses the fixed start manifests in `protocols/` across models and seeds. To run or resume the Cube P00–P04 matrix with one serial worker per listed GPU:

```bash
python scripts/run_cube_fixed_protocol.py --group all --gpus 0,1,2,3
python analysis/summarize_cube_fixed_protocol.py
```

`headline` evaluates LeWM and AD-WM; `components` adds the component ablations; `all` includes inverse-input and loss-weight sweeps. Completed valid files are skipped on resume. The summarizer checks episode/start pairs and perturbations and writes per-seed and aggregate summaries under ignored `analysis/results/`. Rates are percentages, and SD is the population SD across three training seeds. Check start matching with:

```bash
python analysis/check_matched_starts.py --strict
# After running the full sensitivity matrix:
python analysis/check_matched_starts.py --group all --strict
```

## Analysis

The repository includes the diagnostics used to study factual prediction, counterfactual ranking, CEM candidate selection, and configuration shift:

- **Factual prediction and local rollouts:** `python analyze_residual_diagnostics.py checkpoints.run_dir=<run_name>`. The Cube config uses a three-frame context, 25-step rollout, and first-five-step local MSE. `analysis/complete_res_mi_factual.py` contains the 61,999-clip Res+MI completion; it expects the LeWM reference diagnostic `.npz` first.
- **Shared candidate bank:** `python scripts/run_cube_shared_bank.py` evaluates 64 matched Cube cases for LeWM, Res, Res+Inv, Res+MI, and AD-WM at three seeds. Each bank contains expert, zero, negated, and reversed actions plus 99 near, 99 medium, and 98 far candidates. `analysis/analyze_elite_metrics.py` recomputes CAD and normalized elite regret at k=15/30/60 with 20,000 paired hierarchical bootstrap resamples; `analysis/bootstrap_regret_chain.py` summarizes the k=30 chain.
- **Local CEM landscape:** `python analyze_planner_alignment.py checkpoints.run_dir=<run_name>` uses 20 cases and a 31×31 grid in the final elite PCA plane.
- **Configuration-shift support:** after baseline Cube P00–P04 results exist, run `python analysis/audit_cube_perturbation_support.py` and `python analysis/audit_cube_arm_joint_support.py`. The latter uses 20,000 tabletop/non-contact training states per seed and cross-episode nearest neighbors.

These scripts perform offline analysis and do not change training or the deployed MPC solver. The required baseline and ablation checkpoints can be trained from their included configurations.

## External baselines and release scope

[External-baseline notes](external/README.md) document the Fast-LeWM, Sub-JEPA, and INTACT sources and versions. The released `results/external_cube_*.csv` files contain 3,300 validated per-episode Cube records from 66 evaluation files, with source hashes and table statistics. Run `python analysis/external_cube_results.py` to validate the records and rebuild the 30 comparison cells. Re-running the external models requires their authors' implementations and weights.

This release covers simulation training, evaluation, and analysis. Real-robot deployment, Franka evaluation, and V-JEPA 2/DROID post-training code and weights are outside its scope. The ablation checkpoints are not hosted with the 15 main-model checkpoints. Hardware and library differences, and regenerated Scene trajectories, may change exact percentages.

## Acknowledgements

This codebase builds on [LeWorldModel](https://github.com/lucas-maes/le-wm). We thank its authors for releasing their code, datasets, and evaluation framework. We also thank the authors of the external methods used in the comparison.

## Citation

The arXiv citation will be added here when the preprint is available.

## License

This repository is released under the [MIT License](LICENSE). Code derived from LeWorldModel retains the corresponding copyright and license notices.
