"""
Visualization utilities for graph-regularized Hopfield attention experiments.

All functions save plots to a specified path and also return the Figure object.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. Noise vs Accuracy
# ---------------------------------------------------------------------------

def plot_noise_vs_accuracy(df: pd.DataFrame, save_path: str) -> plt.Figure:
    """
    Line plot comparing baseline vs diffused accuracy across noise levels.

    Expected columns: ['noise_level', 'baseline_accuracy', 'diffused_accuracy'].
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["noise_level"], df["baseline_accuracy"],
            marker="o", label="Baseline Hopfield", linewidth=2)
    ax.plot(df["noise_level"], df["diffused_accuracy"],
            marker="s", label="Diffused Hopfield", linewidth=2, linestyle="--")
    ax.set_xlabel("Noise level p (bit-flip probability)")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("H1: Noise Robustness — Baseline vs Diffused Hopfield")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 2. η sweep (diffusion strength)
# ---------------------------------------------------------------------------

def plot_eta_sweep(df: pd.DataFrame, save_path: str) -> plt.Figure:
    """
    Plot accuracy as a function of eta for a fixed noise level.

    Expected columns: ['eta', 'accuracy'].
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["eta"], df["accuracy"], marker="D", color="darkorange", linewidth=2)
    ax.axhline(df["accuracy"].iloc[0], color="steelblue",
               linestyle=":", linewidth=1.5, label="Baseline (η=0)")
    ax.set_xlabel("Diffusion strength η")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("H3: Optimal η — Non-Monotonic Effect of Diffusion Strength")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 3. Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation(df: pd.DataFrame, save_path: str) -> plt.Figure:
    """
    Bar chart comparing ablation configurations.

    Expected columns: ['config', 'accuracy'].
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(df["config"], df["accuracy"],
                  color=colors[: len(df)], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Diffusion configuration")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("H2: Ablation — Where Does Diffusion Help?")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 4. Attention entropy histogram / distribution plot
# ---------------------------------------------------------------------------

def plot_attention_entropy(
    results_dict: dict,
    save_path: str,
    title: str = "H4: Attention Entropy — Baseline vs Diffused",
) -> plt.Figure:
    """
    Histogram (or KDE) comparing attention entropy distributions.

    Args:
        results_dict: dict mapping label string → 1-D numpy array of entropy values.
        save_path:    Output file path.
        title:        Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, vals in results_dict.items():
        ax.hist(vals, bins=30, alpha=0.6, label=label, density=True)
    ax.set_xlabel("Attention entropy (nats)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 5. Optional: PCA of embeddings before / after diffusion
# ---------------------------------------------------------------------------

def plot_pca_embeddings(
    X_before: np.ndarray,
    X_after: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: str = "results/plots/pca_embeddings.png",
) -> plt.Figure:
    """
    Side-by-side PCA scatter of stored patterns before and after diffusion.

    Args:
        X_before: (N, d) array — pre-diffusion.
        X_after:  (N, d) array — post-diffusion.
        labels:   Optional (N,) integer cluster labels for colouring.
        save_path: Output path.
    """
    from sklearn.decomposition import PCA  # optional dependency

    pca = PCA(n_components=2)
    pca.fit(X_before)
    Z_before = pca.transform(X_before)
    Z_after = pca.transform(X_after)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, Z, ttl in zip(axes, (Z_before, Z_after), ("Before diffusion", "After diffusion")):
        sc = ax.scatter(Z[:, 0], Z[:, 1],
                        c=labels if labels is not None else "steelblue",
                        cmap="tab10", s=30, alpha=0.7)
        ax.set_title(ttl)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(True, alpha=0.3)
    fig.suptitle("PCA of Stored Patterns Before / After Graph Diffusion")
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 6. Steps sweep — accuracy vs diffusion steps for each mode
# ---------------------------------------------------------------------------

def plot_steps_sweep(df: pd.DataFrame, save_path: str) -> plt.Figure:
    """
    Line plot of accuracy as a function of diffusion steps, one line per mode.

    Expected columns: ['steps', 'mode', 'accuracy'].
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = {"simple": "o", "iterative": "s", "spectral": "D"}
    for mode, grp in df.groupby("mode"):
        ax.plot(grp["steps"], grp["accuracy"],
                marker=markers.get(mode, "^"), label=mode, linewidth=2)
    ax.set_xlabel("Diffusion steps")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("Diffusion Steps Sweep — Accuracy vs Steps by Mode")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 7. Mode comparison bar chart
# ---------------------------------------------------------------------------

def plot_mode_comparison(df: pd.DataFrame, save_path: str) -> plt.Figure:
    """
    Bar chart comparing diffusion modes at a fixed noise level.

    Expected columns: ['mode', 'accuracy'].
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(df["mode"], df["accuracy"],
                  color=colors[: len(df)], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Diffusion mode")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("Mode Comparison — Baseline vs Simple vs Iterative vs Spectral")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 8. Energy tracking — energy vs diffusion steps
# ---------------------------------------------------------------------------

def plot_energy_vs_steps(df: pd.DataFrame, save_path: str) -> plt.Figure:
    """
    Line plot of Hopfield energy as a function of diffusion steps.

    Expected columns: ['steps', 'energy']. Optionally 'mode' for per-mode lines.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    if "mode" in df.columns:
        for mode, grp in df.groupby("mode"):
            ax.plot(grp["steps"], grp["energy"],
                    marker="o", label=mode, linewidth=2)
    else:
        ax.plot(df["steps"], df["energy"],
                marker="o", color="darkorange", linewidth=2)
    ax.set_xlabel("Diffusion steps")
    ax.set_ylabel("Hopfield energy")
    ax.set_title("Energy Landscape — Hopfield Energy vs Diffusion Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 9. Noise sweep with multiple modes
# ---------------------------------------------------------------------------

def plot_noise_multi_mode(df: pd.DataFrame, save_path: str) -> plt.Figure:
    """
    Noise sweep comparing multiple diffusion modes.

    Expected columns: ['noise_level', 'mode', 'accuracy'].
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "baseline": ("-", "o"),
        "simple": ("--", "s"),
        "iterative": ("-.", "D"),
        "spectral": (":", "^"),
    }
    for mode, grp in df.groupby("mode"):
        ls, mk = styles.get(mode, ("-", "x"))
        ax.plot(grp["noise_level"], grp["accuracy"],
                linestyle=ls, marker=mk, label=mode, linewidth=2)
    ax.set_xlabel("Noise level p (bit-flip probability)")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("Noise Robustness — All Diffusion Modes")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150)
    return fig
