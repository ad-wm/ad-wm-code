# External Cube comparison

This directory documents the external methods in the paper's Cube table. Their
training code and model weights stay with the original authors. This release
contains our **evaluation outputs** as per-episode records, the shared start
manifests, and a checker that rebuilds the reported means and population SDs.

## Check the released results

From the repository root:

```bash
python analysis/external_cube_results.py
```

The checker reads [per-episode results](../results/external_cube_episodes.csv)
and recomputes [the table values](../results/external_cube_aggregate.csv).
[Source hashes](../results/external_cube_sources.csv) identify the 66 saved
evaluation files used to build the CSV. There are 50 episodes in every file:
one released Cube checkpoint each for Fast-LeWM and Sub-JEPA, and three E5
training checkpoints for each INTACT inference mode. All use evaluation seed
42, goal offset 25, budget 50, and the exact
[`cube_original.npz`](../protocols/cube_original.npz) or
[`cube_hardstart.npz`](../protocols/cube_hardstart.npz) episode/start list.
P00–P04 also have identical applied XY perturbations across methods.
The checker verifies these facts from the packaged records.

To regenerate these CSVs from the saved NPZ evaluation workspace:

```bash
python analysis/external_cube_results.py --source-root /path/to/evaluation-workspace
```

This import mode expects the layout recorded in the script. It verifies each
NPZ success percentage against its 50 per-episode outcomes before packaging.
The resulting CSVs contain evaluation records only, no observations or weights.

## Original code and checkpoints

| Method | Original implementation and checkpoint source | Evaluated inference |
| --- | --- | --- |
| Fast-LeWM | [code](https://github.com/Yuntian-Gao/Fast-LeWorldModel), [weights](https://huggingface.co/naiverer/fast-leworldmodel) | Released Cube checkpoint, CEM with horizon 1 and action block 25 |
| Sub-JEPA | [code](https://github.com/intcomp/Sub-JEPA), [weights](https://huggingface.co/intcomp/sub-jepa) | Released Cube checkpoint, CEM with 300 samples, 30 elites, 30 iterations, horizon 5, action block 5 |
| INTACT | [code](https://github.com/zju3dv/INTACT-JEPA), [paper E5 weights](https://huggingface.co/INTACT-JEPA/INTACT/tree/paper-e5-goal-v1) | Three official E5 training checkpoints (0, 42, 3072), each under Direct, Pure CEM, and Actor CEM; one shared evaluation seed |

Evaluated local source revisions were
Fast-LeWM `492752d96b2a11ec802c55322c46bdc87885da09`,
Sub-JEPA `ef945ed434ce529bc7c5f1995f2e1cf173954843`, and
INTACT `235b6a3a92db4d0f1b3a40597ab1f407db4fd15b`.
The released Fast-LeWM Cube object checkpoint had SHA-256
`e7b43b8dea338b7c65d81fd59c7aeba7fa8ca6e2124cca8f18885a982a2fa1`;
the Sub-JEPA Cube object checkpoint had SHA-256
`05a269ab6da3be48748db77f878442852d2af18dc7693ebb2fd3220d4760263c`.
INTACT's three E5 weight hashes are
`96a4d525e996c2b78fa384873496396024364850ad4af0003274b8c1426ea270`,
`d155d9d0c71cc48614d822b14456f341748a25524c7a899b959a54312755e94a`,
and `d2abfd5d907491cf770cc9d3f5bd003f53d0d16f01ac4732111b9dc459895f07`
for seeds 0/42/3072. INTACT E5 weights require its frozen
`paper_runtime/` checkpoint grammar; the current root runtime is
incompatible with those weights. Its [paper checkpoint notes](https://github.com/zju3dv/INTACT-JEPA/blob/main/docs/PAPER_CHECKPOINTS.md)
explain this boundary.

For a new model run, use each method's original environment and inference
interface, select the released Cube checkpoint above, and pass the shared
episode/start manifests to a dataset-backed evaluator. Save the policy name,
training/evaluation seed, protocol, 50 episode/start pairs, applied perturbation,
and 50 binary successes. Rebuild the comparison with the checker after the new
run. The archived numerical table can be independently verified from the
packaged CSVs; running the external models again requires their original code,
weights, and the matching evaluation adapter. No third-party source or weights
are vendored here.
