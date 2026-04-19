"""
Dynamical Memory Engine for iterative diffusion-attention systems.

Responsibility: Orchestrate the interleaved diffusion → attention loop.

This module provides four focused classes following Single Responsibility:

    GraphCache      — builds and caches (W, deg, adj_indices, L, op) once per input
    DynamicsEngine  — runs T steps of diffusion → attention on Q, K, V patterns
    EnergyTracker   — computes and tracks Hopfield energy across steps
    DiffusionConfig — single serialisable config object shared by all classes

Design notes
------------
* ``GraphCache`` is the single source of truth for all graph objects.
  No other module rebuilds W, deg, L, or the diffusion operator.
* ``DynamicsEngine`` holds references to a ``DiffusionOperator`` and an
  ``AttentionOperator``; it does NOT rebuild either per step or per call.
* ``FactoredDiffusion`` is the default for sparse mode — it never forms L.
  ``SpectralDiffusion`` and ``SimpleDiffusion`` use L when required.
* ``EnergyTracker`` is optional and zero-cost when disabled.

Full dynamics loop (Section 4.1 of spec)
-----------------------------------------
    for t in range(T):
        Q = diffusion(Q)
        K = diffusion(K)
        Q = attention(Q, K, V)

Each iteration costs:
    diffusion : O(kNd) with FactoredDiffusion + sparse W
    attention : O(N²d) dense  OR  O(kNd) graph mode

Complexity (N patterns, d features, T steps, k graph neighbours)
-----------------------------------------------------------------
    GraphCache.get   build  : O(N²d) sim-matrix + O(N²) kNN + O(N³) if spectral
    GraphCache.get   hit    : O(1)
    DynamicsEngine.run      : O(T * kNd)  sparse factored diffusion + graph attn
                              O(T * N^2d)  dense diffusion + dense attn
    EnergyTracker.step      : O(N²)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .attention_operator import AttentionOperator
from .diffusion import DiffusionOperator, FactoredDiffusion
from .graph.builder import GraphBuilder
from .graph.laplacian_builder import LaplacianBuilder


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    """
    Unified, serialisable configuration for the dynamical diffusion system.

    Attributes:
        eta:                  Diffusion strength eta.
        beta:                 Hopfield scaling / attention inverse-temperature.
        steps:                Number of diffusion+attention iterations (T).
        diffusion_mode:       'simple', 'iterative', 'spectral', or 'factored'.
        attention_mode:       'dense' (O(N^2)) or 'graph' (O(kN)).
        k_neighbors:          kNN graph degree.
        use_normalized_laplacian: Use symmetric-normalised L (recommended).
        use_sparse:           Store adjacency as sparse_coo (O(kN) storage).
        diffuse_key:          Apply diffusion to stored patterns (keys).
        diffuse_query:        Apply diffusion to state patterns (queries).
        use_logit_diffusion:  Also smooth post-softmax attention weights.
        logit_eta:            eta for logit-level diffusion; defaults to eta.
        adaptive_eta:         Scale eta by attention entropy.
        adaptive_temperature: Sigmoid temperature for adaptive eta.
        adaptive_threshold:   Entropy midpoint for adaptive eta.
        cache_graph:          Cache graph across forward passes.
        energy_stop_tol:      Early-stop if |Delta E| < tol (0 = disabled).
    """
    eta: float = 0.1
    beta: float = 1.0
    steps: int = 3
    diffusion_mode: str = "factored"
    attention_mode: str = "dense"
    k_neighbors: int = 5
    use_normalized_laplacian: bool = True
    use_sparse: bool = False
    diffuse_key: bool = True
    diffuse_query: bool = False
    use_logit_diffusion: bool = False
    logit_eta: Optional[float] = None
    adaptive_eta: bool = False
    adaptive_temperature: float = 5.0
    adaptive_threshold: float = 1.0
    cache_graph: bool = True
    energy_stop_tol: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable dict."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Graph Cache
# ---------------------------------------------------------------------------

class GraphCache:
    """
    Builds and caches all graph objects for a given pattern tensor.

    Responsibility: graph construction and operator lifecycle — nothing else.

    Caches per unique input (keyed by tensor data pointer):
        * W           — (N, N) adjacency (dense or sparse_coo)
        * deg         — (N,)   degree vector
        * adj_indices — (N, k) kNN neighbor indices (needed by graph attention)
        * L           — (N, N) Laplacian (only when required by diffusion mode)
        * op          — Precomputed DiffusionOperator

    Call ``invalidate()`` to force a full rebuild on the next ``get()``.

    Args:
        config: DiffusionConfig controlling graph and diffusion settings.
    """

    def __init__(self, config: DiffusionConfig) -> None:
        self._cfg = config
        self._graph_builder = GraphBuilder(
            k=config.k_neighbors, use_sparse=config.use_sparse
        )
        self._lap_builder = LaplacianBuilder(
            normalized=config.use_normalized_laplacian
        )
        self._reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, patterns: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], DiffusionOperator]:
        """
        Return (W, deg, adj_indices, L, op) for the given patterns.

        Fast path (O(1)): returns cached objects when patterns are unchanged.
        Slow path (O(N^2 d)): builds everything from scratch.

        Args:
            patterns: (N, d) float32 representative patterns.

        Returns:
            W:           (N, N) adjacency.
            deg:         (N,)   degree vector.
            adj_indices: (N, k) neighbor indices for graph attention.
            L:           (N, N) Laplacian — None for 'factored' mode.
            op:          Precomputed DiffusionOperator.
        """
        ptr = patterns.data_ptr()
        if self._cfg.cache_graph and ptr == self._cached_ptr:
            return (
                self._cached_W,
                self._cached_deg,
                self._cached_adj,
                self._cached_L,
                self._cached_op,
            )

        W, deg, adj_idx, L, op = self._build(patterns)

        if self._cfg.cache_graph:
            self._cached_ptr = ptr
            self._cached_W   = W
            self._cached_deg = deg
            self._cached_adj = adj_idx
            self._cached_L   = L
            self._cached_op  = op

        return W, deg, adj_idx, L, op

    def invalidate(self) -> None:
        """Force a full rebuild on the next call to ``get``."""
        self._reset()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._cached_ptr: Optional[int]               = None
        self._cached_W:   Optional[Tensor]            = None
        self._cached_deg: Optional[Tensor]            = None
        self._cached_adj: Optional[Tensor]            = None
        self._cached_L:   Optional[Tensor]            = None
        self._cached_op:  Optional[DiffusionOperator] = None

    def _build(
        self, patterns: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], DiffusionOperator]:
        cfg = self._cfg
        X = patterns.detach().float()                               # (N, d)

        # All graph construction through GraphBuilder (single responsibility)
        W, deg, adj_idx = self._graph_builder.build(X)

        # Move to target device/dtype
        W       = W.to(device=patterns.device)
        deg     = deg.to(dtype=patterns.dtype, device=patterns.device)
        adj_idx = adj_idx.to(device=patterns.device)

        # FactoredDiffusion: no L required — avoids dense N*N matrix
        if cfg.diffusion_mode == "factored":
            op = FactoredDiffusion(eta=cfg.eta, steps=cfg.steps)
            op.precompute_from_graph(W, deg)
            return W, deg, adj_idx, None, op

        # All other modes need L (LaplacianBuilder — single responsibility)
        L = self._lap_builder.build(W).to(
            dtype=patterns.dtype, device=patterns.device
        )
        op = DiffusionOperator.create(
            mode=cfg.diffusion_mode,
            eta=cfg.eta,
            steps=cfg.steps,
        )
        op.precompute(L)
        return W, deg, adj_idx, L, op


# ---------------------------------------------------------------------------
# Energy Tracker
# ---------------------------------------------------------------------------

class EnergyTracker:
    """
    Tracks Hopfield energy across diffusion steps.

    E = -(beta * Q @ K^T).mean() + eta * trace(K^T L K) / N

    Responsibility: measuring energy; providing early-stop signal.

    Args:
        beta: Hopfield scaling factor.
        eta:  Diffusion regularisation strength.
        tol:  Stop if |E_t - E_{t-1}| < tol. 0 disables early stopping.
    """

    def __init__(self, beta: float, eta: float, tol: float = 0.0) -> None:
        self.beta = beta
        self.eta  = eta
        self.tol  = tol
        self._history: List[float] = []

    @property
    def history(self) -> List[float]:
        """List of per-step energy values."""
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()

    @torch.no_grad()
    def step(self, Q: Tensor, K: Tensor, L: Tensor) -> Tuple[float, bool]:
        """
        Compute energy for the current step; return (energy, should_stop).

        Args:
            Q: (N, d) or (S, B, d) query patterns.
            K: (N, d) or (S, B, d) key patterns (after diffusion at this step).
            L: (N, N) graph Laplacian.

        Returns:
            energy:      Scalar energy value.
            should_stop: True if early-stop criterion is met.
        """
        Q_2d = Q.mean(dim=1) if Q.dim() == 3 else Q
        K_2d = K.mean(dim=1) if K.dim() == 3 else K
        affinity   = -(self.beta * Q_2d @ K_2d.t()).mean()
        smoothness = self.eta * torch.trace(K_2d.t() @ L @ K_2d) / K_2d.shape[0]
        energy     = (affinity + smoothness).item()
        self._history.append(energy)

        if self.tol > 0.0 and len(self._history) >= 2:
            if abs(self._history[-1] - self._history[-2]) < self.tol:
                return energy, True

        return energy, False

    @torch.no_grad()
    def step_factored(
        self, Q: Tensor, K: Tensor, W: Tensor, deg: Tensor,
    ) -> Tuple[float, bool]:
        """
        Compute energy without L, using the factored identity:

            tr(K^T L K) = (deg ⊙ ||k_i||²).sum() - (W@K ⊙ K).sum()

        Enables energy tracking with FactoredDiffusion (L=None).

        Args:
            Q:   (N, d) or (S, B, d) query patterns.
            K:   (N, d) or (S, B, d) key patterns.
            W:   (N, N) adjacency (dense or sparse).
            deg: (N,) degree vector.

        Returns:
            energy:      Scalar energy value.
            should_stop: True if early-stop criterion is met.
        """
        Q_2d = Q.mean(dim=1) if Q.dim() == 3 else Q
        K_2d = K.mean(dim=1) if K.dim() == 3 else K

        affinity = -(self.beta * Q_2d @ K_2d.t()).mean()
        K_norms_sq = (K_2d * K_2d).sum(dim=-1)                  # (N,)
        deg_term = (deg * K_norms_sq).sum()
        WK = torch.sparse.mm(W, K_2d) if W.is_sparse else W @ K_2d
        smoothness = self.eta * (deg_term - (WK * K_2d).sum()) / K_2d.shape[0]
        energy = (affinity + smoothness).item()
        self._history.append(energy)

        if self.tol > 0.0 and len(self._history) >= 2:
            if abs(self._history[-1] - self._history[-2]) < self.tol:
                return energy, True
        return energy, False


# ---------------------------------------------------------------------------
# Dynamics Engine
# ---------------------------------------------------------------------------

class DynamicsEngine:
    """
    Core iterative dynamical system: x_{t+1} = Attention(D * x_t).

    Responsibility: run the T-step loop, alternating diffusion and attention.
    This class NEVER builds or rebuilds the graph, Laplacian, or operators.
    All precomputed objects are injected at construction time.

    Full loop (spec Section 4.1)::

        for t in range(T):
            Q = diffusion(Q)
            K = diffusion(K)
            Q = attention(Q, K, V)

    Each step costs:
        diffusion : O(kNd) factored/sparse  or  O(N^2 d) dense
        attention : O(kNd) graph mode       or  O(N^2 d) dense mode

    The engine also exposes ``run_diffusion`` for single-tensor use (keys or
    queries independently) to maintain backward compatibility with
    ``DiffusedHopfield._associate``.

    Args:
        diffusion_op:   Precomputed DiffusionOperator (must be callable).
        attention_op:   AttentionOperator in 'dense' or 'graph' mode.
                        Required only for ``run_dynamics``; optional for
                        ``run_diffusion`` (backward-compat path).
        steps:          Number of dynamics iterations T.
        energy_tracker: Optional EnergyTracker; enables per-step early-stop.
    """

    def __init__(
        self,
        diffusion_op: DiffusionOperator,
        attention_op: Optional[AttentionOperator] = None,
        steps: Optional[int] = None,
        energy_tracker: Optional[EnergyTracker] = None,
    ) -> None:
        self._diff_op = diffusion_op
        self._attn_op = attention_op
        self._steps   = steps if steps is not None else diffusion_op.steps
        self._tracker = energy_tracker

    # ------------------------------------------------------------------
    # Full dynamics loop  Q_T = run_dynamics(Q, K, V)
    # ------------------------------------------------------------------

    def run_dynamics(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        adj_indices: Optional[Tensor] = None,
        L: Optional[Tensor] = None,
        W: Optional[Tensor] = None,
        deg: Optional[Tensor] = None,
        diffuse_query: bool = True,
        diffuse_key: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Run the full T-step diffusion-attention loop.

        Inside the loop — ZERO redundant computation:
            * No graph rebuild
            * No Laplacian recompute
            * No new memory allocation (reuses Q, K in-place conceptually)

        Energy tracking works with both L-based and factored (W, deg) forms.

        Args:
            Q:             (N, d) or (S, B, d) query patterns.
            K:             (N, d) or (S, B, d) key patterns.
            V:             (N, d) or (S, B, d) value patterns.
            adj_indices:   (N, k) neighbor indices — required for graph attention.
            L:             (N, N) Laplacian — for L-based energy tracking.
            W:             (N, N) adjacency — for factored energy tracking.
            deg:           (N,) degree vector — for factored energy tracking.
            diffuse_query: Whether to diffuse Q each iteration.
            diffuse_key:   Whether to diffuse K each iteration.

        Returns:
            (Q, K): Tuple of updated queries and diffused keys after T steps.

        Complexity per step:
            diffusion : O(kNd) factored sparse  or  O(N^2 d) dense
            attention : O(kNd) graph            or  O(N^2 d) dense
        """
        if self._attn_op is None:
            raise RuntimeError(
                "DynamicsEngine.run_dynamics requires an AttentionOperator. "
                "Pass attention_op= at construction time."
            )

        # Clean slate for energy tracking each dynamics run
        if self._tracker is not None:
            self._tracker.reset()

        use_energy_L = self._tracker is not None and L is not None
        use_energy_fac = (
            self._tracker is not None
            and W is not None and deg is not None
            and not use_energy_L
        )

        for _ in range(self._steps):
            if diffuse_key:
                K = self._diff_op(K)
            if diffuse_query:
                Q = self._diff_op(Q)

            # Attention update — dense O(N^2) or graph O(kN)
            Q = self._attn_op(Q, K, V, adj_indices=adj_indices)

            # Optional: energy check for early stop
            if use_energy_L:
                _, stop = self._tracker.step(Q, K, L)
                if stop:
                    break
            elif use_energy_fac:
                _, stop = self._tracker.step_factored(Q, K, W, deg)
                if stop:
                    break

        return Q, K

    # ------------------------------------------------------------------
    # Single-tensor diffusion (backward-compatible)
    # ------------------------------------------------------------------

    def run_diffusion(
        self,
        X: Tensor,
        L: Optional[Tensor] = None,
        Q_ref: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply the diffusion operator for ``steps`` iterations to a single tensor.

        Preserves the original API used by ``DiffusedHopfield._associate``
        for feature-level key/query diffusion before Hopfield attention.

        No graph rebuild per step.

        Args:
            X:     (N, d) or (S, B, d) patterns to diffuse.
            L:     (N, N) Laplacian — required only for energy tracking.
            Q_ref: (N, d) query reference — required only for energy tracking.

        Returns:
            X': Diffused patterns, same shape as X.
        """
        use_tracking = (
            self._tracker is not None
            and L is not None
            and Q_ref is not None
            and X.dim() == 2
        )

        if use_tracking:
            for _ in range(self._steps):
                X = self._diff_op(X)
                _, stop = self._tracker.step(Q_ref, X, L)
                if stop:
                    break
        else:
            X = self._diff_op(X)

        return X

    # ------------------------------------------------------------------
    # Adaptive eta utility
    # ------------------------------------------------------------------

    def compute_adaptive_eta(
        self, logits: Tensor, base_eta: float,
        temperature: float = 5.0, threshold: float = 1.0,
    ) -> float:
        """
        Compute entropy-gated adaptive eta.

        eta_eff = base_eta * sigmoid(temperature * (H(attn) - threshold))

        Args:
            logits:      (..., S) raw attention logits or weights.
            base_eta:    Maximum eta value.
            temperature: Sigmoid steepness.
            threshold:   Entropy midpoint.

        Returns:
            eta_eff: Scalar float in [0, base_eta].
        """
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            H     = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
            scale = torch.sigmoid(temperature * (H - threshold))
        return base_eta * scale.item()
