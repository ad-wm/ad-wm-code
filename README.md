# AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control

Official simulation code for **AD-WM**, an action-discriminative latent world model for counterfactual model predictive control (MPC).

AD-WM builds on [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm). It combines residual latent dynamics with inverse dynamics and a normalized action-recovery objective motivated by conditional mutual information. These objectives train the predictor to retain action-dependent differences between candidate futures. Their auxiliary heads are discarded at test time, so planning uses the standard latent-space CEM interface.

**Links:** [Project page](https://ad-wm.github.io/) · [Pretrained checkpoints](https://huggingface.co/ad-wm/ad-wm) · [LeWM](https://github.com/lucas-maes/le-wm)

## Repository structure

```text
.
├── train.py, jepa.py, module.py     # Training and world-model components
├── eval.py                           # Closed-loop MPC evaluation
├── config/train/experiment/          # Paper training configurations
├── config/eval/                      # Evaluation configurations
├── protocols/                        # Fixed evaluation starts
├── scripts/                          # Data and experiment runners
├── analysis/, analyze_*.py           # Analysis tools
├── checkpoints/                      # Checkpoint manifest
└── external/                         # External-baseline provenance
```

## Installation

Follow the [LeWM installation instructions](https://github.com/lucas-maes/le-wm#using-the-code) for the base environment. This release uses Python 3.10; `requirements.txt` pins the additional dependencies. Install a PyTorch build appropriate for your CUDA setup first.

```bash
git clone https://github.com/ad-wm/ad-wm-code.git
cd ad-wm-code
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt

export MUJOCO_GL=egl
export STABLEWM_HOME="$PWD/checkpoints"
```

`STABLEWM_HOME` contains both datasets and checkpoint run directories. Use `CUDA_VISIBLE_DEVICES` to select a GPU.

## Data

Download the original LeWM datasets from the [official collection](https://huggingface.co/collections/quentinll/lewm) and extract their HDF5 files under `$STABLEWM_HOME` as described in the [LeWM data instructions](https://github.com/lucas-maes/le-wm#data).

| Environment | HDF5 path below `$STABLEWM_HOME` |
| --- | --- |
| Cube | `ogbench/cube_single_expert.h5` |
| Reacher | `dmc/reacher.h5` |
| TwoRoom | `tworooms/tworoom.h5` |
| PushT | `pusht/pusht_expert_train.h5` |
| Scene | `ogbench/scene_play_lewm_10k_200.h5` (generate below) |

Generate the additional Scene dataset with the included collection script:

```bash
python scripts/generate_ogbench_manipspace.py \
  --env_name scene-v0 --dataset_type play \
  --num_episodes 10000 --max_episode_steps 200 \
  --dataset_name ogbench/scene_play_lewm_10k_200 \
  --cache_dir "$STABLEWM_HOME"
```

## Pretrained checkpoints

Download the [main-model checkpoints](https://huggingface.co/ad-wm/ad-wm) into the same root as the datasets. The manifest verifies the downloaded files:

```bash
hf download ad-wm/ad-wm --local-dir "$STABLEWM_HOME"
python scripts/verify_checkpoint_manifest.py --root "$STABLEWM_HOME"
```

## Training

Paper configurations are in `config/train/experiment/`. Train one model or run a configuration group:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py experiment=cube_res_inv_detachact_mifixedvar_seed3072_seedfix

python scripts/run_paper_simulation.py train --group main
python scripts/run_paper_simulation.py train --group baseline
python scripts/run_paper_simulation.py train --group cube-ablations --env cube
```

Use `--dry-run` with the runner to inspect its commands. Resolved configurations and checkpoints are written under `$STABLEWM_HOME`.

## Evaluation

Fixed manifests under `protocols/` reproduce the Cube Original, Cube hard-start perturbation, and balanced Scene evaluation settings. The matching settings are in `config/eval/`. A policy path is relative to `$STABLEWM_HOME` and omits the `_object.ckpt` suffix.

```bash
python eval.py --config-name=cube_original_benchmark \
  policy=cube_res_inv_detachact_mifixedvar_seed3072_seedfix/cube_res_inv_detachact_mifixedvar_seed3072_seedfix_epoch_10 \
  eval.start_manifest=protocols/cube_original.npz
```

Run the paper evaluation groups with the supplied runner:

```bash
python scripts/run_paper_simulation.py eval --group main
python scripts/run_paper_simulation.py eval --group baseline
python scripts/run_paper_simulation.py eval --group cube-ablations --env cube
```

The runner uses the fixed manifests for matched starts across models and seeds. Evaluation outputs include per-episode results.

## Analysis

The repository includes tools for the paper's simulation analyses:

- **Factual prediction:** `analyze_residual_diagnostics.py`
- **Counterfactual candidate analysis:** `scripts/run_cube_shared_bank.py` and `analysis/analyze_elite_metrics.py`
- **Planner landscape:** `analyze_planner_alignment.py`
- **Configuration shift:** `analysis/audit_cube_perturbation_support.py` and `analysis/audit_cube_arm_joint_support.py`

The required baseline and ablation models can be trained from the included configurations.

## External baselines

[External-baseline notes](external/README.md) document the Fast-LeWM, Sub-JEPA, and INTACT sources, released Cube evaluation records, and how to rebuild their comparison results. Re-running those models requires their authors' code and weights. Real-robot code and weights are outside this simulation release.

## Acknowledgements

This codebase builds on [LeWorldModel](https://github.com/lucas-maes/le-wm). We thank its authors for releasing their code, datasets, and evaluation framework. We also thank the authors of the external methods used in the comparison.

## Citation

The arXiv citation will be added here when the preprint is available.

## License

This repository is released under the [MIT License](LICENSE). Code derived from LeWorldModel retains the corresponding copyright and license notices.
