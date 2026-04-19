"""
Graph construction utilities for Graph-Regularized Hopfield attention.

Responsibility: Build a kNN similarity graph from pattern embeddings.

Builds similarity graphs over pattern sets (stored patterns / queries) used
to construct the graph Laplacian for diffusion pre-processing.

Supports dense output (default) and torch.sparse_coo output for O(kN)
Laplacian-vector products at inference time.

Complexity:
    build_similarity_matrix : O(N²d)
    build_knn_graph (dense)  : O(N²)
    build_knn_graph (sparse) : O(kN) storage
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def build_similarity_matrix(X: Tensor) -> Tensor:
    """
    Compute pairwise cosine similarity matrix, clamping negatives to zero
    and zeroing out the diagonal (no self-loops).

    Args:
        X: (N, d) float32 embedding matrix.

    Returns:
        S: (N, N) dense, non-negative cosine similarity matrix.

    Complexity: O(N²d) time, O(N²) space.
    """
    X_norm = F.normalize(X.float(), p=2, dim=-1)   # (N, d)
    S = X_norm @ X_norm.t()                         # (N, N)
    S = S.clamp(min=0.0)
    S.fill_diagonal_(0.0)
    return S


def build_knn_graph(S: Tensor, k: int, as_sparse: bool = False) -> Tensor:
    """
    Sparsify a similarity matrix so that each node retains only its top-k
    neighbors; adjacency is symmetrized (undirected graph).

    Args:
        S:         (N, N) similarity matrix.
        k:         Number of neighbors to keep per node.
        as_sparse: If True, return a torch.sparse_coo_tensor instead of
                   a dense tensor.  Enables O(kN) sparse matmuls downstream.

    Returns:
        A: (N, N) symmetric adjacency matrix, dense or sparse_coo.

    Complexity: O(N²) time for topk; O(kN) storage when as_sparse=True.
    """
    N = S.shape[0]
    k = min(k, N - 1)

    # Select top-k similarities per row — O(N²) due to full topk scan.
    topk_vals, topk_idx = torch.topk(S, k=k, dim=-1)   # (N, k)

    mask = torch.zeros_like(S)
    mask.scatter_(dim=-1, index=topk_idx, src=topk_vals)

    # Symmetrize: A_ij = max(A_ij, A_ji) — preserves all chosen edges.
    A = torch.max(mask, mask.t())

    if as_sparse:
        # Convert to COO sparse for efficient downstream L @ X products.
        indices = A.nonzero(as_tuple=False).t().contiguous()  # (2, nnz)
        values = A[indices[0], indices[1]]
        return torch.sparse_coo_tensor(indices, values, A.shape,
                                       dtype=A.dtype, device=A.device).coalesce()
    return A
