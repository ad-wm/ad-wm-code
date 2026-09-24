# Checkpoints

The pretrained AD-WM checkpoints and their model card are hosted on [Hugging Face](https://huggingface.co/ad-wm/ad-wm).

[`MANIFEST.csv`](MANIFEST.csv) records the expected checkpoint paths, sizes, and SHA-256 hashes. The resolved `config.yaml` files in this directory support evaluation and analysis.

After downloading the models to `$STABLEWM_HOME`, verify them from the repository root:

```bash
python scripts/verify_checkpoint_manifest.py --root "$STABLEWM_HOME"
```
