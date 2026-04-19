"""
Synthetic data generation utilities for graph-regularized Hopfield experiments.
"""

import torch
import numpy as np
from torch import Tensor


def generate_patterns(N: int, d: int, seed: int = 42) -> Tensor:
    """
    Generate N random binary patterns of dimension d with values in {-1, +1}.

    Patterns are drawn i.i.d. from a Rademacher distribution and L2-normalised
    to unit vectors (makes cosine-similarity well defined).

    Args:
        N: Number of patterns.
        d: Pattern dimension.
        seed: Random seed for reproducibility.

    Returns:
        patterns: (N, d) float32 tensor with values approximately ±1/√d after
                  normalization.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    bits = torch.randint(0, 2, (N, d), generator=rng).float() * 2 - 1
    patterns = bits / (d ** 0.5)
    return patterns


def generate_clustered_patterns(N: int, d: int, n_clusters: int = 10,
                                 intra_noise: float = 0.15,
                                 seed: int = 42) -> Tensor:
    """
    Generate N patterns organised into clusters.

    Each cluster has a random centroid (Rademacher ±1) and member patterns
    are noisy copies of the centroid.  This creates patterns with real graph
    structure — nearby patterns in the kNN graph genuinely share features,
    making graph diffusion more effective than on fully random patterns.

    Args:
        N:           Total number of patterns.
        d:           Pattern dimension.
        n_clusters:  Number of clusters.
        intra_noise: Per-bit flip probability within a cluster (lower = tighter clusters).
        seed:        Random seed for reproducibility.

    Returns:
        patterns: (N, d) float32 tensor, L2-normalised.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    per_cluster = N // n_clusters
    remainder = N - per_cluster * n_clusters

    centroids = torch.randint(0, 2, (n_clusters, d), generator=rng).float() * 2 - 1

    patterns_list = []
    for i in range(n_clusters):
        n_i = per_cluster + (1 if i < remainder else 0)
        centroid = centroids[i].unsqueeze(0).expand(n_i, d)
        flip = torch.bernoulli(
            torch.full((n_i, d), intra_noise), generator=rng
        ).bool()
        noisy = torch.where(flip, -centroid, centroid)
        patterns_list.append(noisy)

    patterns = torch.cat(patterns_list, dim=0)
    patterns = patterns / (d ** 0.5)
    return patterns


def add_noise(x: Tensor, p: float, seed: int = 0) -> Tensor:
    """
    Corrupt a pattern by independently flipping each dimension's sign with
    probability p (bit-flip noise on the binary ±1 encoding).

    Args:
        x: (..., d) input pattern(s).
        p: Flip probability in [0, 1].
        seed: Random seed for reproducibility.

    Returns:
        x_noisy: Same shape as x with approximately p*d bits flipped.
    """
    if p == 0.0:
        return x.clone()
    rng = torch.Generator()
    rng.manual_seed(seed)
    flip_mask = torch.bernoulli(
        torch.full(x.shape, p, dtype=x.dtype, device=x.device),
        generator=rng,
    ).bool()
    return torch.where(flip_mask, -x, x)
