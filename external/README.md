# External Cube baselines

The Cube comparison uses Fast-LeWM, Sub-JEPA, and INTACT with their original implementations and weights. This repository releases the evaluation records, not third-party model code or checkpoints.

## Released results

- [Per-episode outcomes](../results/external_cube_episodes.csv)
- [Aggregate results](../results/external_cube_aggregate.csv)
- [Saved evaluation file hashes](../results/external_cube_sources.csv)
- [Code revisions and checkpoint hashes](../results/external_cube_model_sources.csv)

From the repository root, validate the packaged results and rebuild the aggregate table:

```bash
python analysis/external_cube_results.py
```

## Methods

| Method | Code | Checkpoint | Inference |
| --- | --- | --- | --- |
| Fast-LeWM | [Official repository](https://github.com/Yuntian-Gao/Fast-LeWorldModel) | [Released weights](https://huggingface.co/naiverer/fast-leworldmodel) | CEM |
| Sub-JEPA | [Official repository](https://github.com/intcomp/Sub-JEPA) | [Released weights](https://huggingface.co/intcomp/sub-jepa) | CEM |
| INTACT | [Official repository](https://github.com/zju3dv/INTACT-JEPA) | [Paper E5 weights](https://huggingface.co/INTACT-JEPA/INTACT/tree/paper-e5-goal-v1) | Direct, Pure CEM, Actor CEM |

## Evaluation protocol

All methods use the released [Cube Original](../protocols/cube_original.npz) and [hard-start](../protocols/cube_hardstart.npz) start manifests. P00–P04 use matched perturbations. The checker validates starts and perturbations against the per-episode records.

## Scope

Re-running these baselines requires the authors' repositories, weights, and matching evaluation adapters. No third-party source code or weights are vendored here.
