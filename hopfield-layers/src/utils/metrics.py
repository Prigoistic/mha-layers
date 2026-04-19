"""
Evaluation metrics for graph-regularized Hopfield attention experiments.
"""

import torch
from torch import Tensor


def accuracy(pred: Tensor, target: Tensor) -> float:
    """
    Fraction of examples where pred == target (exact match).

    Args:
        pred:   (N,) integer class predictions.
        target: (N,) integer ground-truth labels.

    Returns:
        acc: scalar float in [0, 1].
    """
    return (pred == target).float().mean().item()


def hamming_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Normalised per-element Hamming distance between two binary {-1, +1} tensors.

    Args:
        x: (..., d) tensor.
        y: (..., d) tensor, same shape as x.

    Returns:
        dist: (...) tensor of per-sample normalised Hamming distances in [0, 1].
    """
    d = x.shape[-1]
    return (x.sign() != y.sign()).float().sum(dim=-1) / d


def attention_entropy(weights: Tensor, eps: float = 1e-9) -> Tensor:
    """
    Shannon entropy of each attention distribution.

    Args:
        weights: (..., S) attention weight matrix (each row sums to ~1).
        eps:     Small constant for numerical stability inside log.

    Returns:
        H: (...) per-query entropy values (in nats).
    """
    return -(weights * (weights + eps).log()).sum(dim=-1)


def attention_sparsity(weights: Tensor, threshold: float = 0.01) -> Tensor:
    """
    Fraction of attention weights that are effectively zero (< threshold).

    Args:
        weights:   (..., S) attention weights.
        threshold: Values below this are considered near-zero.

    Returns:
        sparsity: (...) per-query sparsity in [0, 1].
    """
    return (weights < threshold).float().mean(dim=-1)


def hopfield_energy(Q: Tensor, K: Tensor, L: Tensor,
                    eta: float, beta: float = 1.0) -> float:
    """
    Compute the Hopfield energy with graph-regularisation penalty.

    E = -(β * Q @ K^T).mean() + η * trace(K^T L K) / N

    The first term is the (negative) mean scaled dot-product affinity — lower
    is better (stronger pattern alignment).  The second term is the graph
    smoothness penalty (Dirichlet energy) measuring feature variation across
    graph edges.

    Args:
        Q: (N, d) query patterns.
        K: (N, d) key patterns.
        L: (N, N) graph Laplacian.
        eta: Diffusion / regularisation strength.
        beta: Hopfield scaling (temperature).

    Returns:
        energy: Scalar energy value.
    """
    affinity = -(beta * Q @ K.t()).mean()
    smoothness = eta * torch.trace(K.t() @ L @ K) / K.shape[0]
    return (affinity + smoothness).item()
