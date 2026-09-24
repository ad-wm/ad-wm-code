"""Select videos without changing the evaluated batch or episode numbering."""

import numpy as np


def video_reference_mask(episodes, start_steps, reference_path=None, settings=None):
    episodes = np.asarray(episodes)
    start_steps = np.asarray(start_steps)
    if episodes.ndim != 1 or start_steps.shape != episodes.shape:
        raise ValueError("Expected aligned one-dimensional episode/start arrays")
    if not reference_path:
        return np.ones(len(episodes), dtype=bool)
    with np.load(reference_path, allow_pickle=False) as reference:
        for key, expected in (
            ("eval_episodes", episodes), ("eval_start_idx", start_steps)
        ):
            if not np.array_equal(reference[key], expected):
                raise ValueError(f"Video reference has different {key}: {reference_path}")
        for key, expected in (settings or {}).items():
            if key not in reference or not np.array_equal(
                reference[key].reshape(-1), np.asarray(expected).reshape(-1)
            ):
                raise ValueError(f"Video reference has different/missing {key}")
        mask = reference["episode_successes"].astype(bool)
        if mask.shape != episodes.shape:
            raise ValueError("Video reference success array has incorrect shape")
        return mask


def video_output_indices(successes, reference_mask, mode="all"):
    successes = np.asarray(successes, dtype=bool)
    mask = np.asarray(reference_mask, dtype=bool).copy()
    if mask.shape != successes.shape:
        raise ValueError("Video mask and success array have different shapes")
    if mode == "success":
        mask &= successes
    elif mode == "failure":
        mask &= ~successes
    elif mode != "all":
        raise ValueError(f"Unknown video filter: {mode}")
    return np.flatnonzero(mask)
