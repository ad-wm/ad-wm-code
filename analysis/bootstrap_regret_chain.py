#!/usr/bin/env python3
"""Audit consecutive ablation steps from saved N=300 costs (no inference)."""
import json
from pathlib import Path
import numpy as np
from analyze_elite_metrics import (
    MODELS, SEEDS, STABLE_WM, BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES,
    metric_path, load_raw, case_metrics, hierarchical_paired_reduction_ci,
)

def main():
    values = {}
    reference = None
    for model, pattern in MODELS.items():
        values[model] = {}
        for seed in SEEDS:
            path = metric_path(STABLE_WM / pattern.format(seed=seed),
                               'counterfactual_metrics_visual_latent_n300')
            predicted, realized, protocol = load_raw(path)
            assert predicted.shape == (64, 300)
            if reference is None: reference = protocol
            assert protocol == reference, f'Unmatched bank: {path}'
            values[model][seed] = case_metrics(predicted, realized, 30, 3)[0]
    pairs = [('LeWM', 'Res'), ('Res', 'Res+Inv'),
             ('Res+Inv', 'AD-WM (.01)'), ('Res+MI (.01)', 'AD-WM (.01)')]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = {}
    for source, target in pairs:
        point, low, high = hierarchical_paired_reduction_ci(values[source], values[target], rng)
        label = f'{source} -> {target}'.replace('AD-WM (.01)', 'AD-WM')
        results[label] = dict(reduction=point, ci95=[low, high])
    out = Path(__file__).parent / 'results/regret_chain_bootstrap_n300_k30.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(metric='normalized best-in-elite regret R30', candidate_count=300,
                  elite_size=30, case_count=64, seeds=list(SEEDS),
                  bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
                  bootstrap_unit='training seeds, then paired cases within seed',
                  positive_direction='source minus target: reduction in regret', results=results)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
