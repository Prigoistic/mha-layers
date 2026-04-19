"""
Noise Robustness Experiment
===========================
Tests H1: Diffusion improves recall under noise.

Setup
-----
* N stored binary patterns of dimension d.
* For each noise level p ∈ [0.0 → 0.5]:
    - M noisy queries are generated (one per stored pattern, randomly selected).
    - Both baseline Hopfield and DiffusedHopfield retrieve the closest stored
      pattern using static attention (β = beta, no learned projections).
    - Retrieval accuracy = fraction of queries where the argmax attention weight
      points to the correct stored pattern.

Outputs
-------
* results/noise_vs_accuracy.csv
* results/plots/noise_vs_accuracy.png  (via visualization module)
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


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------

def _build_retrieval_model(diffused: bool, N: int, d: int, beta: float,
                            eta: float, k: int, device: torch.device,
                            diffusion_mode: str = "iterative",
                            diffusion_steps: int = 3):
    """
    Return a (Diffused)Hopfield module configured for static pattern retrieval.

    Inputs / outputs:
        stored  (key + value) : (batch=M, seq=N, d)
        query   (state)       : (batch=M, seq=1, d)
        output                : (batch=M, seq=1, d)
    """
    common = dict(
        input_size=None,          # static mode — no projection
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
    if diffused:
        model = DiffusedHopfield(**common, eta=eta, k_neighbors=k,
                                 use_normalized_laplacian=True,
                                 diffuse_query=False, diffuse_key=True,
                                 diffusion_mode=diffusion_mode,
                                 diffusion_steps=diffusion_steps)
    else:
        model = Hopfield(**common)
    return model.to(device).eval()


@torch.no_grad()
def _run_retrieval(model, patterns: torch.Tensor,
                   noisy_queries: torch.Tensor) -> torch.Tensor:
    """
    Retrieve patterns for a batch of noisy queries.

    Args:
        patterns:      (N, d) stored patterns.
        noisy_queries: (M, d) noisy query vectors.

    Returns:
        pred_idx: (M,) tensor of predicted pattern indices (argmax attention).
    """
    M = noisy_queries.shape[0]
    N = patterns.shape[0]
    d = patterns.shape[1]

    # (M, N, d) — same stored patterns for every query in the batch
    stored = patterns.unsqueeze(0).expand(M, N, d)           # (M, N, d)
    query  = noisy_queries.unsqueeze(1)                      # (M, 1, d)

    # get_association_matrix returns (batch=M, heads=1, L_query=1, S_stored=N)
    attn = model.get_association_matrix(
        input=(stored, query, stored)
    )                                                         # (M, 1, 1, N)
    attn = attn[:, 0, 0, :]                                   # (M, N)
    pred_idx = attn.argmax(dim=-1)                            # (M,)
    return pred_idx


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_noise_robustness(
    N: int = 200,
    d: int = 64,
    beta: float = 12.0,
    eta: float = 0.10,
    k: int = 7,
    M: int = 500,
    noise_levels: list = None,
    diffusion_mode: str = "spectral",
    diffusion_steps: int = 3,
    n_clusters: int = 20,
    seed: int = 42,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Run the noise-robustness experiment and return a results DataFrame.

    Args:
        N:            Number of stored patterns.
        d:            Pattern dimensionality.
        beta:         Hopfield scaling (β).
        eta:          Diffusion strength for DiffusedHopfield.
        k:            kNN neighbours in similarity graph.
        M:            Number of noisy queries generated per noise level.
        noise_levels: List of flip probabilities to sweep.
        seed:         Master random seed.
        results_dir:  Directory where CSV and plots are saved.

    Returns:
        df: DataFrame with columns
            [noise_level, baseline_accuracy, diffused_accuracy,
             baseline_hamming, diffused_hamming].
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]

    device = torch.device("cpu")
    patterns = generate_clustered_patterns(
        N, d, n_clusters=n_clusters, seed=seed
    ).to(device)   # (N, d)

    baseline_model  = _build_retrieval_model(False, N, d, beta, eta, k, device)
    diffused_model  = _build_retrieval_model(True,  N, d, beta, eta, k, device,
                                              diffusion_mode, diffusion_steps)

    rows = []
    for p in noise_levels:
        # Generate M noisy queries: pick M random target pattern indices, add noise
        rng = torch.Generator()
        rng.manual_seed(seed + int(p * 1000))
        target_idx = torch.randint(0, N, (M,), generator=rng)        # (M,)
        clean_queries = patterns[target_idx]                          # (M, d)
        noisy_queries = add_noise(clean_queries, p, seed=seed + int(p * 100))

        pred_base = _run_retrieval(baseline_model, patterns, noisy_queries)
        pred_diff = _run_retrieval(diffused_model, patterns, noisy_queries)

        # Hamming distance to true pattern
        true_patterns = patterns[target_idx]                          # (M, d)
        retr_base = patterns[pred_base]                               # (M, d)
        retr_diff = patterns[pred_diff]                               # (M, d)
        ham_base = ((true_patterns.sign() != retr_base.sign()).float().sum(-1) / d).mean().item()
        ham_diff = ((true_patterns.sign() != retr_diff.sign()).float().sum(-1) / d).mean().item()

        rows.append({
            "noise_level": round(p, 4),
            "baseline_accuracy": accuracy(pred_base, target_idx),
            "diffused_accuracy": accuracy(pred_diff, target_idx),
            "baseline_hamming": round(ham_base, 4),
            "diffused_hamming": round(ham_diff, 4),
        })
        print(f"  p={p:.2f}  baseline={rows[-1]['baseline_accuracy']:.3f}"
              f"  diffused={rows[-1]['diffused_accuracy']:.3f}")

    df = pd.DataFrame(rows)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "noise_vs_accuracy.csv", index=False)
    print(f"Saved → {out_dir / 'noise_vs_accuracy.csv'}")
    return df
