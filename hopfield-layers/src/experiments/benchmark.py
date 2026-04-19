"""
Runtime Benchmark Experiment
=============================
Compares wall-clock time and (optionally) peak memory for:
    1. Dense vs sparse Laplacian storage and matmul
    2. Diffusion mode runtime (simple / iterative / spectral)
    3. DiffusionOperator reuse (precomputed D) vs rebuilt every call
    4. DynamicsEngine T-step sweep at T ∈ {1, 2, 3, 5}

All timings are mean ± std over ``n_repeats`` runs after one warmup.

Outputs
-------
* results/benchmark_diffusion_modes.csv
* results/benchmark_sparse_dense.csv
* results/benchmark_op_reuse.csv
* results/benchmark_steps.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from hflayers.diffusion import DiffusionOperator, apply_diffusion
from hflayers.dynamics_engine import DiffusionConfig, DynamicsEngine, GraphCache
from hflayers.graph.build_graph import build_similarity_matrix, build_knn_graph
from hflayers.graph.laplacian import compute_normalized_laplacian
from src.utils.data_gen import generate_clustered_patterns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, n_repeats: int = 20):
    """Return (mean_ms, std_ms) over n_repeats calls after one warmup."""
    fn()   # warmup
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def _build_base(N, d, k, seed=42):
    """Return (patterns, L_dense) for timing experiments."""
    patterns = generate_clustered_patterns(N, d, n_clusters=max(4, N // 10), seed=seed)
    S = build_similarity_matrix(patterns)
    A = build_knn_graph(S, k)
    L = compute_normalized_laplacian(A)
    return patterns, L


# ---------------------------------------------------------------------------
# Benchmark 1: Diffusion mode comparison
# ---------------------------------------------------------------------------

def bench_diffusion_modes(
    N: int = 200, d: int = 64, k: int = 7, eta: float = 0.1,
    n_repeats: int = 20, seed: int = 42,
) -> pd.DataFrame:
    """Time simple vs iterative vs spectral for equal number of total steps=3."""
    patterns, L = _build_base(N, d, k, seed)
    X = patterns.clone()

    rows = []
    for mode in ("simple", "iterative", "spectral"):
        steps = 3
        op = DiffusionOperator.create(mode=mode, eta=eta, steps=steps)
        op.precompute(L)

        mean_ms, std_ms = _time_fn(lambda: op(X), n_repeats)
        rows.append({
            "mode": mode,
            "N": N,
            "d": d,
            "steps": steps,
            "mean_ms": round(mean_ms, 4),
            "std_ms": round(std_ms, 4),
        })
        print(f"  {mode:12s}  N={N:4d}  {mean_ms:.3f} ± {std_ms:.3f} ms")

    # functional API for reference (rebuilds operator every call)
    for mode in ("simple", "iterative", "spectral"):
        mean_ms, std_ms = _time_fn(
            lambda: apply_diffusion(X, L, eta, mode=mode, steps=3), n_repeats
        )
        rows.append({
            "mode": f"{mode}_functional",
            "N": N, "d": d, "steps": 3,
            "mean_ms": round(mean_ms, 4),
            "std_ms": round(std_ms, 4),
        })
        print(f"  {mode+'(fn)':12s}  N={N:4d}  {mean_ms:.3f} ± {std_ms:.3f} ms")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Benchmark 2: Sparse vs dense
# ---------------------------------------------------------------------------

def bench_sparse_vs_dense(
    N: int = 200, d: int = 64, k: int = 7, eta: float = 0.1,
    n_repeats: int = 20, seed: int = 42,
) -> pd.DataFrame:
    """Compare O(kN) torch.sparse.mm vs dense matmul inside SimpleDiffusion."""
    patterns = generate_clustered_patterns(N, d, n_clusters=max(4, N // 10), seed=seed)
    S = build_similarity_matrix(patterns)

    rows = []
    for as_sparse in (False, True):
        A = build_knn_graph(S, k, as_sparse=as_sparse)
        L = compute_normalized_laplacian(A)   # always dense
        op = DiffusionOperator.create("simple", eta=eta).precompute(L)
        X = patterns.clone()
        mean_ms, std_ms = _time_fn(lambda: op(X), n_repeats)
        label = "sparse" if as_sparse else "dense"
        rows.append({
            "storage": label,
            "N": N, "d": d, "k": k,
            "mean_ms": round(mean_ms, 4),
            "std_ms": round(std_ms, 4),
        })
        print(f"  {label:8s}  N={N}  k={k}  {mean_ms:.3f} ± {std_ms:.3f} ms")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Benchmark 3: Operator reuse vs rebuild
# ---------------------------------------------------------------------------

def bench_op_reuse(
    N: int = 200, d: int = 64, k: int = 7, eta: float = 0.1,
    n_repeats: int = 20, seed: int = 42,
) -> pd.DataFrame:
    """Confirm that precomputed DiffusionOperator is faster than functional API."""
    patterns, L = _build_base(N, d, k, seed)
    X = patterns.clone()

    op = DiffusionOperator.create("spectral", eta=eta).precompute(L)

    mean_reuse, std_reuse = _time_fn(lambda: op(X), n_repeats)
    mean_rebuild, std_rebuild = _time_fn(
        lambda: apply_diffusion(X, L, eta, mode="spectral"), n_repeats
    )

    rows = [
        {"strategy": "reuse (precomputed)", "mean_ms": round(mean_reuse, 4),
         "std_ms": round(std_reuse, 4)},
        {"strategy": "rebuild (functional)", "mean_ms": round(mean_rebuild, 4),
         "std_ms": round(std_rebuild, 4)},
    ]
    for r in rows:
        print(f"  {r['strategy']:26s}  {r['mean_ms']:.3f} ± {r['std_ms']:.3f} ms")
    speedup = mean_rebuild / (mean_reuse + 1e-9)
    print(f"  Speedup from reuse: {speedup:.2f}x")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Benchmark 4: T-step sweep (DynamicsEngine)
# ---------------------------------------------------------------------------

def bench_steps(
    N: int = 200, d: int = 64, k: int = 7, eta: float = 0.1,
    step_values: List[int] = None,
    n_repeats: int = 20, seed: int = 42,
) -> pd.DataFrame:
    """Time DynamicsEngine.run_diffusion for varying step counts."""
    if step_values is None:
        step_values = [1, 2, 3, 5, 7, 10]

    cfg = DiffusionConfig(
        eta=eta, steps=1, diffusion_mode="iterative",
        k_neighbors=k, cache_graph=True,
    )
    cache = GraphCache(cfg)
    patterns, L = _build_base(N, d, k, seed)
    _W, _deg, _adj, _L, op = cache.get(patterns.float())
    X = patterns.clone()

    rows = []
    for steps in step_values:
        engine = DynamicsEngine(op, steps=steps)
        mean_ms, std_ms = _time_fn(lambda: engine.run_diffusion(X), n_repeats)
        rows.append({
            "steps": steps, "N": N, "d": d,
            "mean_ms": round(mean_ms, 4), "std_ms": round(std_ms, 4),
        })
        print(f"  T={steps:2d}  {mean_ms:.3f} ± {std_ms:.3f} ms")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    N: int = 200, d: int = 64, k: int = 7, eta: float = 0.1,
    n_repeats: int = 20, seed: int = 42, results_dir: str = "results",
) -> None:
    """Run all benchmarks and save CSVs."""
    print("\n" + "=" * 60)
    print("BENCHMARK — Diffusion Modes")
    print("=" * 60)
    df_modes = bench_diffusion_modes(N, d, k, eta, n_repeats, seed)

    print("\n" + "=" * 60)
    print("BENCHMARK — Sparse vs Dense")
    print("=" * 60)
    df_sparse = bench_sparse_vs_dense(N, d, k, eta, n_repeats, seed)

    print("\n" + "=" * 60)
    print("BENCHMARK — Operator Reuse")
    print("=" * 60)
    df_reuse = bench_op_reuse(N, d, k, eta, n_repeats, seed)

    print("\n" + "=" * 60)
    print("BENCHMARK — T-step Sweep (DynamicsEngine)")
    print("=" * 60)
    df_steps = bench_steps(N, d, k, eta, None, n_repeats, seed)

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_modes.to_csv(out / "benchmark_diffusion_modes.csv", index=False)
    df_sparse.to_csv(out / "benchmark_sparse_dense.csv", index=False)
    df_reuse.to_csv(out / "benchmark_op_reuse.csv", index=False)
    df_steps.to_csv(out / "benchmark_steps.csv", index=False)

    for name in ("benchmark_diffusion_modes", "benchmark_sparse_dense",
                 "benchmark_op_reuse", "benchmark_steps"):
        print(f"Saved → {results_dir}/{name}.csv")
