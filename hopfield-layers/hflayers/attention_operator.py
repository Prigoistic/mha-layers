"""
Attention operator for the diffusion-attention dynamical memory system.

Responsibility: Apply scaled-dot-product attention in exactly two modes.
This is the *only* place that computes attention — no other module does it.

Modes
-----
dense  (default)
    logits  = beta * Q @ K.T        — (N, N)
    weights = softmax(logits)       — (N, N)
    output  = weights @ V           — (N, d)
    O(N²d) time, O(N²) space.
    Exact match to the Hopfield baseline.

graph
    For each query i, attend *only* to its kNN neighbors.
    Requires adj_indices (N, k) LongTensor from ``GraphBuilder``.
    O(kNd) time, O(kN) space.
    Strictly faster than dense when k ≪ N.

API
---
    op = AttentionOperator(beta=10.0, mode="dense")
    out = op(Q, K, V)                           # dense
    out = op(Q, K, V, adj_indices=adj_idx)      # graph

Constraints
-----------
* Modes are implemented in separate methods (_dense / _graph).
  No conditional logic inside a shared inner loop.
* Dense mode never uses adj_indices; graph mode never builds N×N logit matrix.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class AttentionOperator:
    """
    Scaled-dot-product attention with dense or graph-constrained mode.

    Args:
        beta: Scaling / inverse-temperature factor. Default: 1.0.
        mode: ``'dense'`` (O(N²)) or ``'graph'`` (O(kN)).

    Time complexity:
        dense : O(N²d)
        graph : O(kNd)

    Space complexity:
        dense : O(N²)
        graph : O(kN)
    """

    def __init__(self, beta: float = 1.0, mode: str = "dense") -> None:
        if mode not in ("dense", "graph"):
            raise ValueError(
                f"AttentionOperator: mode must be 'dense' or 'graph', got '{mode}'."
            )
        self.beta = beta
        self.mode = mode

    def __call__(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        adj_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply attention.

        Args:
            Q:           (N, d) query patterns.
            K:           (N, d) key patterns.
            V:           (N, d) value patterns.
            adj_indices: (N, k) LongTensor of neighbor indices.
                         Required for graph mode; ignored in dense mode.

        Returns:
            output: (N, d) attended result.
        """
        if self.mode == "dense":
            return self._dense(Q, K, V)
        return self._graph(Q, K, V, adj_indices)

    # ------------------------------------------------------------------
    # Dense  O(N²d)
    # ------------------------------------------------------------------

    def _dense(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """
        Standard full-rank attention.

        2D: logits = beta * Q @ K.T (N, N) → softmax → weights @ V.
        3D: (S, B, d) → batched bmm over B.

        Complexity: O(N²d) time, O(N²) space per batch element.
        """
        if Q.dim() == 2:
            logits = self.beta * (Q @ K.t())          # (N, N)
            weights = F.softmax(logits, dim=-1)        # (N, N)
            return weights @ V                         # (N, d)
        # 3D: (S, B, d) — transpose to (B, S, d) for batched matmul
        Q_b, K_b, V_b = Q.permute(1, 0, 2), K.permute(1, 0, 2), V.permute(1, 0, 2)
        logits = self.beta * torch.bmm(Q_b, K_b.transpose(1, 2))  # (B, S, S)
        weights = F.softmax(logits, dim=-1)                         # (B, S, S)
        return torch.bmm(weights, V_b).permute(1, 0, 2)            # (S, B, d)

    # ------------------------------------------------------------------
    # Graph-constrained  O(kNd)
    # ------------------------------------------------------------------

    def _graph(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        adj_indices: Optional[Tensor],
    ) -> Tensor:
        """
        Attend only to kNN neighbors of each query node.

        2D: adj_indices (N, k) → gather → local softmax → weighted sum.
        3D: (S, B, d) → batched gather over B.
        No N×N logit matrix is formed.

        Complexity: O(kNd) time, O(kN) space per batch element.
        """
        if adj_indices is None:
            raise ValueError(
                "AttentionOperator(mode='graph') requires adj_indices (N, k)."
            )

        if Q.dim() == 2:
            K_nbrs = K[adj_indices]                                       # (N, k, d)
            V_nbrs = V[adj_indices]                                       # (N, k, d)
            logits = self.beta * (Q.unsqueeze(1) * K_nbrs).sum(dim=-1)   # (N, k)
            weights = F.softmax(logits, dim=-1)                           # (N, k)
            return (weights.unsqueeze(-1) * V_nbrs).sum(dim=1)           # (N, d)

        # 3D: (S, B, d) — transpose to (B, S, d) for batched gather
        Q_b = Q.permute(1, 0, 2)                                         # (B, S, d)
        K_b = K.permute(1, 0, 2)                                         # (B, S, d)
        V_b = V.permute(1, 0, 2)                                         # (B, S, d)
        K_nbrs = K_b[:, adj_indices]                                      # (B, S, k, d)
        V_nbrs = V_b[:, adj_indices]                                      # (B, S, k, d)
        logits = self.beta * (Q_b.unsqueeze(2) * K_nbrs).sum(dim=-1)    # (B, S, k)
        weights = F.softmax(logits, dim=-1)                               # (B, S, k)
        output = (weights.unsqueeze(-1) * V_nbrs).sum(dim=2)             # (B, S, d)
        return output.permute(1, 0, 2)                                    # (S, B, d)
