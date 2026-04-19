"""
Attention Analysis Experiment
==============================
Tests H3 and H4:
    H3: There exists an optimal η (non-monotonic effect).
    H4: Diffusion smooths the attention distribution (lower entropy variance /
        controlled entropy shift).

Two sub-experiments are run:

A. η sweep at fixed noise
   For η ∈ [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
       measure retrieval accuracy, mean attention entropy, mean sparsity.

B. Attention distribution comparison (baseline vs best-η)
   Collect per-query entropy and sparsity for both models; save histograms.

Outputs
-------
* results/attention_analysis.csv
* results/plots/eta_sweep.png
* results/plots/attention_entropy.png
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
from src.utils.metrics import accuracy, attention_entropy, attention_sparsity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(eta: float, beta: float, k: int, device: torch.device,
                 diffusion_mode: str = "iterative",
                 diffusion_steps: int = 3):
    """Build (Diffused)Hopfield for static retrieval; eta=0 → baseline."""
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
    if eta > 0.0:
        return DiffusedHopfield(**common, eta=eta, k_neighbors=k,
                                use_normalized_laplacian=True,
                                diffuse_query=False, diffuse_key=True,
                                diffusion_mode=diffusion_mode,
                                diffusion_steps=diffusion_steps).to(device).eval()
    else:
        return Hopfield(**common).to(device).eval()


@torch.no_grad()
def _get_attention_and_pred(model, patterns: torch.Tensor,
                            queries: torch.Tensor):
    """
    Returns:
        pred_idx: (M,) argmax predictions.
        attn:     (M, N) attention weight matrix.
    """
    M, d = queries.shape
    N = patterns.shape[0]

    stored = patterns.unsqueeze(0).expand(M, N, d)  # (M, N, d)
    query  = queries.unsqueeze(1)                   # (M, 1, d)

    # Returns (batch=M, heads=1, L=1, S=N)
    attn = model.get_association_matrix(input=(stored, query, stored))  # (M, 1, 1, N)
    attn = attn[:, 0, 0, :]                         # (M, N)
    return attn.argmax(dim=-1), attn


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_attention_analysis(
    N: int = 200,
    d: int = 64,
    beta: float = 12.0,
    k: int = 7,
    M: int = 500,
    noise_level: float = 0.30,
    eta_values: list = None,
    diffusion_mode: str = "spectral",
    diffusion_steps: int = 3,
    n_clusters: int = 20,
    seed: int = 42,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Run η sweep + attention distribution analysis.

    Returns:
        df: DataFrame with columns
            [eta, accuracy, mean_entropy, std_entropy, mean_sparsity].
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if eta_values is None:
        eta_values = [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]

    device = torch.device("cpu")
    patterns = generate_clustered_patterns(
        N, d, n_clusters=n_clusters, seed=seed
    ).to(device)

    rng = torch.Generator()
    rng.manual_seed(seed + 7)
    target_idx    = torch.randint(0, N, (M,), generator=rng)
    clean_queries = patterns[target_idx]
    noisy_queries = add_noise(clean_queries, noise_level, seed=seed + 8)

    rows = []
    entropy_per_eta = {}      # stored for histogram plot

    for eta in eta_values:
        model = _build_model(eta, beta, k, device,
                             diffusion_mode, diffusion_steps)
        pred, attn = _get_attention_and_pred(model, patterns, noisy_queries)

        H = attention_entropy(attn).cpu().numpy()        # (M,)
        sp = attention_sparsity(attn).cpu().numpy()      # (M,)

        acc = accuracy(pred, target_idx)
        entropy_per_eta[f"η={eta}"] = H

        rows.append({
            "eta":           round(eta, 4),
            "accuracy":      round(acc, 4),
            "mean_entropy":  round(float(H.mean()), 4),
            "std_entropy":   round(float(H.std()),  4),
            "mean_sparsity": round(float(sp.mean()), 4),
        })
        print(f"  η={eta:.3f}  acc={acc:.3f}  mean_H={H.mean():.3f}  sparsity={sp.mean():.3f}")

    df = pd.DataFrame(rows)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "attention_analysis.csv", index=False)
    print(f"Saved → {out_dir / 'attention_analysis.csv'}")

    # Store entropy distributions for visualization later
    return df, entropy_per_eta
