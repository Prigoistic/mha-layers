"""
Laplacian builder for Graph-Regularized Hopfield attention.

Responsibility: Compute the graph Laplacian L from adjacency W.
This is the *only* class that calls ``compute_laplacian`` or
``compute_normalized_laplacian``; it is used exclusively when an explicit
Laplacian is required — i.e. for ``SpectralDiffusion`` (eigendecomposition)
and energy tracking (``EnergyTracker``).

For ``FactoredDiffusion`` the Laplacian is never formed; that mode uses
(W, deg) directly, so ``LaplacianBuilder`` is not needed there.

Usage::

    builder = LaplacianBuilder(normalized=True)
    L = builder.build(W)          # W: (N, N) dense or sparse

Complexity:
    normalized=False : O(N²) time, O(N²) space
    normalized=True  : O(N²) time, O(N²) space   (D^{-1/2} L D^{-1/2})

Memory: O(N²) — always returns a dense Laplacian
"""

from __future__ import annotations

from torch import Tensor

from .laplacian import compute_laplacian, compute_normalized_laplacian


class LaplacianBuilder:
    """
    Computes the graph Laplacian L from adjacency W.

    Two variants:
      * ``normalized=False`` — unnormalized L = D - A.
        Eigenvalues in ``[0, max_degree]``.
      * ``normalized=True``  — symmetric-normalized L_norm = D^{-1/2}(D-A)D^{-1/2}.
        Eigenvalues in ``[0, 2]``; stable for η ∈ (0, 0.5).

    Args:
        normalized: If True (default), produce the symmetric-normalized
                    Laplacian.  Recommended for diffusion stability.
    """

    def __init__(self, normalized: bool = True) -> None:
        self.normalized = normalized

    def build(self, W: Tensor) -> Tensor:
        """
        Compute the graph Laplacian from adjacency W.

        Args:
            W: (N, N) adjacency matrix — dense or sparse_coo.

        Returns:
            L: (N, N) dense graph Laplacian (float32).

        Complexity: O(N²) time, O(N²) space.
        """
        if self.normalized:
            return compute_normalized_laplacian(W)
        return compute_laplacian(W)
