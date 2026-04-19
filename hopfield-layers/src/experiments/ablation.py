"""
Ablation Study
==============
Tests H2: Diffusion reduces spurious retrievals.

Four configurations are evaluated at a fixed medium noise level:
    none     — standard Hopfield (no diffusion)
    Q_only   — diffuse query patterns only
    K_only   — diffuse key   patterns only  (primary contribution)
    both     — diffuse both Q and K

For Q_only and both to show a meaningful effect, the M queries are
presented as a *sequence* (L=M, batch=1) so that a graph can be
constructed over the M query positions.

Outputs
-------
* results/ablation.csv
* results/plots/ablation.png
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
# Helpers
# ---------------------------------------------------------------------------

def _build_model(diffuse_q: bool, diffuse_k: bool,
                 beta: float, eta: float, k: int,
                 device: torch.device,
                 diffusion_mode: str = "iterative",
                 diffusion_steps: int = 3):
    """Build a (Diffused)Hopfield in static mode for retrieval."""
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
        batch_first=False,   # use (seq, batch, d) layout for sequence-wise diffusion
        scaling=beta,
    )
    if diffuse_q or diffuse_k:
        model = DiffusedHopfield(**common, eta=eta, k_neighbors=k,
                                 use_normalized_laplacian=True,
                                 diffuse_query=diffuse_q,
                                 diffuse_key=diffuse_k,
                                 diffusion_mode=diffusion_mode,
                                 diffusion_steps=diffusion_steps)
    else:
        model = Hopfield(**common)
    return model.to(device).eval()


@torch.no_grad()
def _retrieve_sequence(model, patterns: torch.Tensor,
                       queries: torch.Tensor) -> torch.Tensor:
    """
    Retrieve patterns for a batch of queries presented as a sequence.

    Uses batch_first=False layout:
        stored (key + value) : (S=N, batch=1, d)
        query  (state)       : (L=M, batch=1, d)
        output               : (L=M, batch=1, d)  — not used directly

    Association matrix shape: (batch=1, L=M, S=N)

    Returns:
        pred_idx: (M,) predicted pattern indices.
    """
    N, d = patterns.shape
    M = queries.shape[0]

    stored = patterns.unsqueeze(1)   # (N, 1, d)
    query  = queries.unsqueeze(1)    # (M, 1, d)

    # Returns (batch=1, heads=1, L=M, S=N)
    attn = model.get_association_matrix(
        input=(stored, query, stored)
    )                                # (1, 1, M, N)
    attn = attn[0, 0, :, :]         # (M, N)
    return attn.argmax(dim=-1)       # (M,)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_ablation(
    N: int = 200,
    d: int = 64,
    beta: float = 12.0,
    eta: float = 0.10,
    k: int = 7,
    M: int = 500,
    noise_level: float = 0.30,
    diffusion_mode: str = "spectral",
    diffusion_steps: int = 3,
    n_clusters: int = 20,
    seed: int = 42,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Ablation over four diffusion configurations at a fixed noise level.

    Args:
        N:           Number of stored patterns.
        d:           Pattern dimensionality.
        beta:        Hopfield scaling β.
        eta:         Diffusion strength.
        k:           kNN graph neighbours.
        M:           Number of noisy queries.
        noise_level: Bit-flip probability for noise corruption.
        seed:        Master random seed.
        results_dir: Output directory.

    Returns:
        df: DataFrame with columns [config, accuracy, hamming].
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    patterns = generate_clustered_patterns(
        N, d, n_clusters=n_clusters, seed=seed
    ).to(device)  # (N, d)

    rng = torch.Generator()
    rng.manual_seed(seed + 1)
    target_idx   = torch.randint(0, N, (M,), generator=rng)
    clean_queries = patterns[target_idx]
    noisy_queries = add_noise(clean_queries, noise_level, seed=seed + 2)

    configs = [
        ("none",   False, False),
        ("Q_only", True,  False),
        ("K_only", False, True),
        ("both",   True,  True),
    ]

    rows = []
    for name, dq, dk in configs:
        model = _build_model(dq, dk, beta, eta, k, device,
                             diffusion_mode, diffusion_steps)
        pred = _retrieve_sequence(model, patterns, noisy_queries)

        true_pats = patterns[target_idx]
        retr_pats = patterns[pred]
        ham = ((true_pats.sign() != retr_pats.sign()).float().sum(-1) / d).mean().item()

        acc = accuracy(pred, target_idx)
        rows.append({"config": name, "accuracy": round(acc, 4),
                     "hamming": round(ham, 4)})
        print(f"  {name:<8}  accuracy={acc:.3f}  hamming={ham:.3f}")

    df = pd.DataFrame(rows)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ablation.csv", index=False)
    print(f"Saved → {out_dir / 'ablation.csv'}")
    return df
