---
license: mit
library_name: pytorch
tags:
- robotics
- world-model
- model-predictive-control
---

# AD-WM simulation checkpoints

This model repository contains the 15 main AD-WM simulation object checkpoints:
Cube, Reacher, TwoRoom, PushT, and Scene at training seeds 3072, 4096, and
6144. Cube/Reacher/TwoRoom/PushT use residual prediction with inverse weight
0.1 and normalized action-recovery weight 0.01. Scene uses 0.0001 for the
action-recovery weight. Each run directory includes the resolved `config.yaml`
needed by the diagnostic tools.

The code repository's `checkpoints/MANIFEST.csv` records the exact relative
paths, byte sizes, and SHA-256 hashes. After download, from the code root run:

```bash
python scripts/verify_checkpoint_manifest.py --root /path/to/downloaded/models
```

Set `STABLEWM_HOME` to a cache root containing these run directories and the
datasets in the code repository's documented layout. The evaluation loader
uses a policy path relative to `STABLEWM_HOME`, without the `_object.ckpt`
suffix. For example:

```bash
python eval.py --config-name=cube_ood_table_perturb04_hardstart \
  policy=cube_res_inv_detachact_mifixedvar_seed3072_seedfix/cube_res_inv_detachact_mifixedvar_seed3072_seedfix_epoch_10 \
  eval.start_manifest=protocols/cube_hardstart.npz
```

These are PyTorch object checkpoints containing Python pickle data. Load only
from a trusted source and verify the hashes before loading. The original LeWM
code and data interface are acknowledged in the companion code repository.
Real-robot weights are outside this release.
