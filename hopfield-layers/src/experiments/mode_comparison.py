"""
Mode Comparison Experiment
==========================
Compare baseline (no diffusion) against all three diffusion modes
(simple, iterative, spectral) at a fixed noise level.

Runs a noise sweep for each mode to show accuracy-vs-noise curves
and reports overall accuracy at a target noise level.

Outputs
-------
* results/mode_comparison.csv
* results/plots/mode_comparison.png
* results/plots/noise_multi_mode.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path

import torch
import numpy as np
import pandas as pd

from hflayers import Hopfield, DiffusedHopfield
from src.utils.data_gen import generate_patterns, generate_clustered_patterns, add_noise
from src.utils.metrics import accuracy


def _build_model(mode: str, beta: float, eta: float, k: int,
                 steps: int, device: torch.device):
    common = dict(
        input_size=None,
        stored_pattern_as_static=True,
        state_pattern_as_static=True,
        pattern_projection_as_static=True,
        disable_out_projection=True,
        normalize_stored_pattern=False,
        normalize_stored_pattern_affine=False,
        normalize_state_pattern=False,
        normalize_state_pattern_affine=False,
        normalize_pattern_projection=False,
        normalize_pattern_projection_affine=False,
        batch_first=True,
        scaling=beta,
    )
    if mode == "baseline":
        return Hopfield(**common).to(device).eval()
    return DiffusedHopfield(
        **common, eta=eta, k_neighbors=k,
        use_normalized_laplacian=True,
        diffuse_query=False, diffuse_key=True,
        diffusion_mode=mode, diffusion_steps=steps,
    ).to(device).eval()


@torch.no_grad()
def _run_retrieval(model, patterns, noisy_queries):
    M, d = noisy_queries.shape
    N = patterns.shape[0]
    stored = patterns.unsqueeze(0).expand(M, N, d)
    query = noisy_queries.unsqueeze(1)
    attn = model.get_association_matrix(input=(stored, query, stored))
    attn = attn[:, 0, 0, :]
    return attn.argmax(dim=-1)


def run_mode_comparison(
    N: int = 200,
    d: int = 64,
    beta: float = 12.0,
    eta: float = 0.10,
    k: int = 7,
    steps: int = 3,
    M: int = 500,
    noise_levels: list = None,
    n_clusters: int = 20,
    seed: int = 42,
    results_dir: str = "results",
) -> tuple:
    """
    Compare diffusion modes across noise levels.

    Returns:
        (df_sweep, df_summary): Full noise-sweep DataFrame and per-mode summary.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    modes = ["baseline", "simple", "iterative", "spectral"]
    device = torch.device("cpu")
    patterns = generate_clustered_patterns(
        N, d, n_clusters=n_clusters, seed=seed
    ).to(device)

    models = {m: _build_model(m, beta, eta, k, steps, device) for m in modes}

    sweep_rows = []
    for p in noise_levels:
        rng = torch.Generator()
        rng.manual_seed(seed + int(p * 1000))
        target_idx = torch.randint(0, N, (M,), generator=rng)
        clean_queries = patterns[target_idx]
        noisy_queries = add_noise(clean_queries, p, seed=seed + int(p * 100))

        for mode in modes:
            pred = _run_retrieval(models[mode], patterns, noisy_queries)
            acc = accuracy(pred, target_idx)
            sweep_rows.append({
                "noise_level": round(p, 4),
                "mode": mode,
                "accuracy": round(acc, 4),
            })

        # Print progress for the last noise level processed
        line = f"  p={p:.2f}"
        for mode in modes:
            row = [r for r in sweep_rows if r["noise_level"] == round(p, 4)
                   and r["mode"] == mode][0]
            line += f"  {mode}={row['accuracy']:.3f}"
        print(line)

    df_sweep = pd.DataFrame(sweep_rows)

    # Summary at a mid-range noise level
    target_p = 0.35
    df_summary = df_sweep[df_sweep["noise_level"] == target_p][
        ["mode", "accuracy"]
    ].reset_index(drop=True)
    if df_summary.empty:
        # fallback to closest
        closest_p = min(noise_levels, key=lambda x: abs(x - target_p))
        df_summary = df_sweep[df_sweep["noise_level"] == closest_p][
            ["mode", "accuracy"]
        ].reset_index(drop=True)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_sweep.to_csv(out_dir / "mode_comparison_sweep.csv", index=False)
    df_summary.to_csv(out_dir / "mode_comparison.csv", index=False)
    print(f"\nSaved → {out_dir / 'mode_comparison_sweep.csv'}")
    print(f"Saved → {out_dir / 'mode_comparison.csv'}")

    return df_sweep, df_summary
