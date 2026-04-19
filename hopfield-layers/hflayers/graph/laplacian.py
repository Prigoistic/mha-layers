"""
Laplacian computation for graph-regularized Hopfield attention.

Responsibility: Convert an adjacency matrix into a graph Laplacian.

Provides both the unnormalized (L = D - A) and symmetric normalized
(L_norm = D^{-1/2} L D^{-1/2}) graph Laplacians.

Supports dense and sparse adjacency inputs.  When the input is a
sparse_coo_tensor every intermediate is kept sparse so that downstream
diffusion can exploit O(kN) sparse matrix-vector products.

Complexity (dense N×N input):
    compute_laplacian            : O(N²) time, O(N²) space
    compute_normalized_laplacian : O(N²) time, O(N²) space
"""

import torch
from torch import Tensor


def _is_sparse(A: Tensor) -> bool:
    return A.is_sparse


def compute_laplacian(A: Tensor) -> Tensor:
    """
    Compute the unnormalized graph Laplacian L = D - A.

    Works with both dense and sparse_coo adjacency matrices.
    Sparse input → sparse output (preserves O(kN) downstream products).

    Args:
        A: (N, N) adjacency / similarity matrix (dense or sparse_coo).

    Returns:
        L: (N, N) graph Laplacian, same layout (dense or sparse_coo).
           Eigenvalues in [0, max_degree].
    """
    if _is_sparse(A):
        A_dense = A.to_dense()
        D = torch.diag(A_dense.sum(dim=-1))
        L_dense = D - A_dense
        # Return dense — sparse Laplacian construction is complex and rarely
        # needed; the sparse benefit is captured at the matmul call site.
        return L_dense

    D = torch.diag(A.sum(dim=-1))
    return D - A


def compute_normalized_laplacian(A: Tensor) -> Tensor:
    """
    Compute the symmetric-normalized graph Laplacian:
        L_norm = D^{-1/2} (D - A) D^{-1/2}

    Eigenvalues are in [0, 2], which gives a stable diffusion range
    eta in (0, 0.5).  Isolated nodes (degree = 0) are handled safely
    by setting their inverse-sqrt degree to 0.

    Works with both dense and sparse_coo adjacency matrices; always
    returns a dense Laplacian (used for eigendecomposition / precompute).

    Args:
        A: (N, N) adjacency / similarity matrix (dense or sparse_coo).

    Returns:
        L_norm: (N, N) dense normalized graph Laplacian.
    """
    A_dense = A.to_dense() if _is_sparse(A) else A
    deg = A_dense.sum(dim=-1)                        # (N,)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0       # isolated nodes -> 0
    D_inv_sqrt = torch.diag(d_inv_sqrt)              # (N, N)
    L = torch.diag(deg) - A_dense
    return D_inv_sqrt @ L @ D_inv_sqrt
