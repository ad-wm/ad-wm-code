#!/usr/bin/env python3
"""Verify the 15 released main-model checkpoints against MANIFEST.csv."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "checkpoints")
    args = parser.parse_args()
    with (args.root / "MANIFEST.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 15:
        raise SystemExit(f"Expected 15 checkpoints, found {len(rows)} manifest entries")
    for row in rows:
        path = args.root / row["relative_checkpoint"]
        if not path.is_file():
            raise SystemExit(f"Missing checkpoint: {path}")
        if path.stat().st_size != int(row["bytes"]):
            raise SystemExit(f"Size mismatch: {path}")
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != row["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {path}")
        if not (path.parent / "config.yaml").is_file():
            raise SystemExit(f"Missing resolved config: {path.parent}")
    print("Verified 15 main-model checkpoints, sizes, SHA-256 hashes, and configs")


if __name__ == "__main__":
    main()
