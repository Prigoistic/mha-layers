"""
Diffusion Steps Sweep Experiment
=================================
How many diffusion steps are optimal? Sweeps steps ∈ {0,1,2,3,5,7,10}
for each diffusion mode (simple, iterative, spectral) plus energy tracking.

Also tracks Hopfield energy at each step count to visualise the energy
landscape as diffusion depth increases.

Outputs
-------
* results/steps_sweep.csv
* results/energy_vs_steps.csv
* results/plots/steps_sweep.png
* results/plots/energy_vs_steps.png
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
from src.utils.metrics import accuracy, hopfield_energy
from hflayers.graph.build_graph import build_similarity_matrix, build_knn_graph
from hflayers.graph.laplacian import compute_normalized_laplacian
from hflayers.diffusion import apply_diffusion


def _build_model(mode: str, steps: int, N: int, d: int,
                 beta: float, eta: float, k: int,
                 device: torch.device):
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
    if steps == 0:
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


def run_steps_sweep(
    N: int = 200,
    d: int = 64,
    beta: float = 12.0,
    eta: float = 0.10,
    k: int = 7,
    M: int = 500,
    noise_level: float = 0.30,
    step_values: list = None,
    modes: list = None,
    n_clusters: int = 20,
    seed: int = 42,
    results_dir: str = "results",
) -> tuple:
    """
    Sweep diffusion steps for each mode.

    Returns:
        (df_acc, df_energy): DataFrames for accuracy and energy results.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if step_values is None:
        step_values = [0, 1, 2, 3, 5, 7, 10]
    if modes is None:
        modes = ["simple", "iterative", "spectral"]

    device = torch.device("cpu")
    patterns = generate_clustered_patterns(
        N, d, n_clusters=n_clusters, seed=seed
    ).to(device)

    rng = torch.Generator()
    rng.manual_seed(seed + 20)
    target_idx = torch.randint(0, N, (M,), generator=rng)
    clean_queries = patterns[target_idx]
    noisy_queries = add_noise(clean_queries, noise_level, seed=seed + 21)

    # Build Laplacian once for energy computation
    S_mat = build_similarity_matrix(patterns)
    A = build_knn_graph(S_mat, k)
    L = compute_normalized_laplacian(A)

    acc_rows = []
    energy_rows = []

    for mode in modes:
        for steps in step_values:
            model = _build_model(mode, steps, N, d, beta, eta, k, device)
            pred = _run_retrieval(model, patterns, noisy_queries)
            acc = accuracy(pred, target_idx)

            # Energy on diffused patterns: apply diffusion manually and measure
            if steps == 0:
                K_eff = patterns
            else:
                K_eff = apply_diffusion(patterns, L, eta,
                                        mode=mode, steps=steps)
            energy = hopfield_energy(noisy_queries[:N], K_eff, L, eta, beta)

            acc_rows.append({"mode": mode, "steps": steps,
                             "accuracy": round(acc, 4)})
            energy_rows.append({"mode": mode, "steps": steps,
                                "energy": round(energy, 4)})
            print(f"  {mode:10s} steps={steps:2d}  acc={acc:.3f}  energy={energy:.2f}")

    df_acc = pd.DataFrame(acc_rows)
    df_energy = pd.DataFrame(energy_rows)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_acc.to_csv(out_dir / "steps_sweep.csv", index=False)
    df_energy.to_csv(out_dir / "energy_vs_steps.csv", index=False)
    print(f"Saved → {out_dir / 'steps_sweep.csv'}")
    print(f"Saved → {out_dir / 'energy_vs_steps.csv'}")

    return df_acc, df_energy
