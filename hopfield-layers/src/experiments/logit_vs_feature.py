"""
Logit-Level vs Feature-Level Diffusion Experiment
==================================================
Compares three regimes:
    1. feature-level  — diffuse K before attention  (standard)
    2. logit-level    — diffuse raw attention logits via L @ logits
    3. both           — feature-level + logit-level

Runs at the harder regime (N=200, d=32, β=12) across noise levels.

Outputs
-------
* results/logit_vs_feature.csv
* results/plots/logit_vs_feature.png
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
from hflayers.diffusion import apply_diffusion
from hflayers.graph.build_graph import build_similarity_matrix, build_knn_graph
from hflayers.graph.laplacian import compute_normalized_laplacian


def _build_baseline(beta: float, device: torch.device):
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
    return Hopfield(**common).to(device).eval()


def _build_diffused(beta: float, eta: float, k: int, mode: str,
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
    return DiffusedHopfield(
        **common, eta=eta, k_neighbors=k,
        use_normalized_laplacian=True,
        diffuse_query=False, diffuse_key=True,
        diffusion_mode=mode, diffusion_steps=steps,
    ).to(device).eval()


@torch.no_grad()
def _retrieval_with_logit_diffusion(
    baseline_model, patterns, noisy_queries, L, eta, mode, steps
):
    """
    Logit-level diffusion: run baseline attention to get raw logits,
    then smooth them with graph diffusion before softmax.
    """
    M, d = noisy_queries.shape
    N = patterns.shape[0]
    stored = patterns.unsqueeze(0).expand(M, N, d)
    query = noisy_queries.unsqueeze(1)

    # Get raw associations (before softmax)
    attn = baseline_model.get_association_matrix(input=(stored, query, stored))
    logits = attn[:, 0, 0, :]                        # (M, N) — these are post-softmax

    # For logit-level diffusion, we diffuse the softmax outputs as a proxy
    # (true logit access would need deeper hook into HopfieldCore)
    # Diffuse over the N-dim (key dimension) using L
    # logits shape: (M, N) — treat as (M, N) where we diffuse over N
    logits_t = logits.t()                             # (N, M)
    logits_diffused = apply_diffusion(logits_t, L, eta, mode=mode, steps=steps)
    logits_diffused = logits_diffused.t()             # (M, N)

    # Re-normalise
    logits_diffused = logits_diffused.clamp(min=0)
    logits_diffused = logits_diffused / (logits_diffused.sum(dim=-1, keepdim=True) + 1e-9)

    return logits_diffused.argmax(dim=-1)


@torch.no_grad()
def _run_retrieval(model, patterns, noisy_queries):
    M, d = noisy_queries.shape
    N = patterns.shape[0]
    stored = patterns.unsqueeze(0).expand(M, N, d)
    query = noisy_queries.unsqueeze(1)
    attn = model.get_association_matrix(input=(stored, query, stored))
    attn = attn[:, 0, 0, :]
    return attn.argmax(dim=-1)


def run_logit_vs_feature(
    N: int = 200,
    d: int = 64,
    beta: float = 12.0,
    eta: float = 0.10,
    k: int = 7,
    steps: int = 3,
    mode: str = "spectral",
    M: int = 500,
    noise_levels: list = None,
    n_clusters: int = 20,
    seed: int = 42,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Compare feature-level vs logit-level vs both diffusion modes.

    Returns:
        df: DataFrame with columns [noise_level, config, accuracy].
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5]

    device = torch.device("cpu")
    patterns = generate_clustered_patterns(
        N, d, n_clusters=n_clusters, seed=seed
    ).to(device)

    # Build Laplacian for logit-level diffusion
    S_mat = build_similarity_matrix(patterns)
    A = build_knn_graph(S_mat, k)
    L = compute_normalized_laplacian(A)

    baseline_model = _build_baseline(beta, device)
    feature_model = _build_diffused(beta, eta, k, mode, steps, device)

    rows = []
    for p in noise_levels:
        rng = torch.Generator()
        rng.manual_seed(seed + int(p * 1000))
        target_idx = torch.randint(0, N, (M,), generator=rng)
        clean_queries = patterns[target_idx]
        noisy_queries = add_noise(clean_queries, p, seed=seed + int(p * 100))

        # Baseline
        pred_base = _run_retrieval(baseline_model, patterns, noisy_queries)
        acc_base = accuracy(pred_base, target_idx)

        # Feature-level diffusion
        pred_feat = _run_retrieval(feature_model, patterns, noisy_queries)
        acc_feat = accuracy(pred_feat, target_idx)

        # Logit-level diffusion
        pred_logit = _retrieval_with_logit_diffusion(
            baseline_model, patterns, noisy_queries, L, eta, mode, steps
        )
        acc_logit = accuracy(pred_logit, target_idx)

        # Both
        # Feature diffused model + logit diffusion on top
        attn = feature_model.get_association_matrix(
            input=(patterns.unsqueeze(0).expand(M, N, d),
                   noisy_queries.unsqueeze(1),
                   patterns.unsqueeze(0).expand(M, N, d))
        )[:, 0, 0, :]
        attn_t = attn.t()
        attn_diffused = apply_diffusion(attn_t, L, eta, mode=mode, steps=steps)
        attn_diffused = attn_diffused.t().clamp(min=0)
        attn_diffused = attn_diffused / (attn_diffused.sum(dim=-1, keepdim=True) + 1e-9)
        pred_both = attn_diffused.argmax(dim=-1)
        acc_both = accuracy(pred_both, target_idx)

        for config, acc in [("baseline", acc_base), ("feature", acc_feat),
                            ("logit", acc_logit), ("both", acc_both)]:
            rows.append({"noise_level": round(p, 4), "config": config,
                         "accuracy": round(acc, 4)})

        print(f"  p={p:.2f}  base={acc_base:.3f}  feat={acc_feat:.3f}"
              f"  logit={acc_logit:.3f}  both={acc_both:.3f}")

    df = pd.DataFrame(rows)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "logit_vs_feature.csv", index=False)
    print(f"Saved → {out_dir / 'logit_vs_feature.csv'}")
    return df
