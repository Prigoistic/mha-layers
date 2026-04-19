"""
Graph builder for Graph-Regularized Hopfield attention.

Responsibility: Build the kNN similarity graph (adjacency W and degree vector)
from raw pattern embeddings.  This is the *only* class that calls
``build_similarity_matrix`` and ``build_knn_graph``.

Separating graph construction from Laplacian computation and diffusion
satisfies Single-Responsibility (SOLID) and avoids duplicated graph logic (DRY).

Usage::

    builder = GraphBuilder(k=5, use_sparse=True)
    W, deg, adj_idx = builder.build(X)    # X: (N, d)

Complexity:
    build_similarity_matrix : O(N²d)
    build_knn_graph (dense) : O(N²)  — topk + symmetrize
    build_knn_graph (sparse): O(kN) storage
    degree computation      : O(N)
    adj_indices extraction  : O(kN)

Memory:
    dense  : O(N²)
    sparse : O(kN) for W; O(N) for deg; O(kN) for adj_indices
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from .build_graph import build_knn_graph, build_similarity_matrix


class GraphBuilder:
    """
    Builds kNN adjacency matrix W, degree vector, and neighbor-index table
    from pattern embeddings.

    The neighbor-index table (adj_indices) is the (N, k) integer tensor
    required by graph-constrained attention (``AttentionOperator`` in graph
    mode) to avoid forming the full N×N weight matrix.

    Args:
        k:          Number of nearest neighbors per node.
        use_sparse: Return W as ``torch.sparse_coo_tensor`` (O(kN) storage).
                    Enables O(kN) sparse matmuls in ``FactoredDiffusion``.
    """

    def __init__(self, k: int = 5, use_sparse: bool = False) -> None:
        self.k = k
        self.use_sparse = use_sparse

    def build(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build adjacency W, degree deg, and neighbor index table from X.

        Args:
            X: (N, d) float32 pattern embeddings.

        Returns:
            W:          (N, N) adjacency matrix — dense float or sparse_coo.
            deg:        (N,)   degree vector, float32 (row sums of dense W).
            adj_indices:(N, k) LongTensor of kNN neighbor indices (dense W).
                        Required by ``AttentionOperator(mode='graph')``.

        Complexity: O(N²d) + O(N²) + O(kN).
        """
        S = build_similarity_matrix(X)                          # O(N²d)
        W_dense = build_knn_graph(S, k=self.k, as_sparse=False) # always dense for indices

        # Degree: row-sum of dense adjacency
        deg = W_dense.sum(dim=1)                                 # (N,)

        # Neighbor index table: top-k by row (symmetric W, so any nonzero works).
        # Use topk on dense W for exact kNN indices.
        k_actual = min(self.k, W_dense.shape[0] - 1)
        adj_indices = torch.topk(W_dense, k=k_actual, dim=1).indices  # (N, k)

        # Optionally convert W to sparse AFTER extracting dense indices
        if self.use_sparse:
            nz = W_dense.nonzero(as_tuple=False).t().contiguous()
            vals = W_dense[nz[0], nz[1]]
            W: Tensor = torch.sparse_coo_tensor(
                nz, vals, W_dense.shape,
                dtype=W_dense.dtype, device=W_dense.device,
            ).coalesce()
        else:
            W = W_dense

        return W, deg, adj_indices
