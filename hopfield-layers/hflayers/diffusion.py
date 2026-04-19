"""
Graph diffusion operators for Graph-Regularized Hopfield attention.

Responsibility: Apply graph diffusion to pattern tensors.

Architecture: Strategy pattern — each mode is a subclass of
``DiffusionOperator`` that precomputes an operator from (L, eta) and
applies it in __call__.  The operator D = I - eta*L is precomputed once
and reused across all diffusion steps and forward passes.

Four diffusion modes:

1. **simple**    — One explicit Euler step using precomputed D = I - eta*L.
2. **iterative** — T applications of D, giving (I - eta*L)^T X.
3. **spectral**  — Exact heat-kernel via eigendecomposition:
                   X' = U exp(-eta*Λ) U^T X.
4. **factored**  — Memory-optimal Laplacian-free form:
                   x' = (1 - η·deg) ⊙ x + η · W @ x
                   Stores only (W_sparse, deg). O(kN) memory.

All operators support 2-D (N, d) and 3-D (S, B, d) inputs.

Backward-compatible functional API (``apply_diffusion``) is preserved.

Complexity:
    DiffusionOperator.precompute : O(N²)  [simple/iterative]
                                   O(N³)  [spectral — eigendecomp]
    DiffusionOperator.__call__   : O(N²)  [dense matmul, all modes]
    With sparse L and sparse matmul: O(kN) per step [simple/iterative]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _matmul(M: Tensor, X: Tensor) -> Tensor:
    """
    M @ X for 2-D or 3-D X, supporting sparse M.

    Args:
        M: (N, N) dense or sparse_coo operator.
        X: (N, d) or (S, B, d) input.

    Returns:
        Result of M @ X, same shape as X.
    """
    if X.dim() == 2:
        if M.is_sparse:
            return torch.sparse.mm(M, X)
        return M @ X
    # 3-D: (S, B, d) — reshape to (S, B*d), matmul, reshape back.
    S, B, d = X.shape
    X_flat = X.reshape(S, B * d)
    if M.is_sparse:
        out = torch.sparse.mm(M, X_flat)
    else:
        out = M @ X_flat
    return out.reshape(S, B, d)


# ---------------------------------------------------------------------------
# Strategy base
# ---------------------------------------------------------------------------

class DiffusionOperator(ABC):
    """
    Abstract base for graph diffusion operators.

    Subclasses must implement ``precompute`` (build the operator from L and
    eta) and ``__call__`` (apply the operator to X).

    The precomputed operator is stored in ``self._op`` and is never
    rebuilt unless ``precompute`` is called again — satisfying the
    no-recomputation-per-step requirement.

    Args:
        eta:   Diffusion strength / time.
        steps: Number of iterations (relevant for iterative mode only).
    """

    def __init__(self, eta: float, steps: int = 1) -> None:
        self.eta = eta
        self.steps = steps
        self._op: Optional[Tensor] = None    # precomputed operator

    def precompute(self, L: Tensor) -> "DiffusionOperator":
        """
        Build and cache the diffusion operator from the Laplacian L.

        Must be called once per unique (L, eta) pair before __call__.

        Args:
            L: (N, N) graph Laplacian (dense or sparse_coo).

        Returns:
            self — enables chaining: op = SimpleDiffusion(eta).precompute(L)
        """
        self._op = self._build_operator(L)
        return self

    @abstractmethod
    def _build_operator(self, L: Tensor) -> Tensor:
        """Construct the precomputed operator from L."""

    @abstractmethod
    def __call__(self, X: Tensor) -> Tensor:
        """
        Apply the diffusion operator to X.

        Args:
            X: (N, d) or (S, B, d) input patterns, float32.

        Returns:
            X': Diffused tensor, same shape as X.
        """

    def _check_ready(self) -> None:
        if self._op is None:
            raise RuntimeError(
                f"{type(self).__name__}.precompute(L) must be called before __call__."
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create(mode: str, eta: float, steps: int = 3) -> "DiffusionOperator":
        """
        Factory method — returns the correct DiffusionOperator subclass.

        Args:
            mode:  'simple', 'iterative', or 'spectral'.
            eta:   Diffusion strength.
            steps: Iterations for 'iterative' mode.

        Returns:
            Un-precomputed DiffusionOperator instance.
        """
        _MAP = {
            "simple": SimpleDiffusion,
            "iterative": IterativeDiffusion,
            "spectral": SpectralDiffusion,
            "factored": FactoredDiffusion,
        }
        cls = _MAP.get(mode)
        if cls is None:
            raise ValueError(
                f"Unknown diffusion mode '{mode}'. Choose from {list(_MAP.keys())}."
            )
        return cls(eta=eta, steps=steps)


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class SimpleDiffusion(DiffusionOperator):
    """
    One-step explicit Euler diffusion: X' = D @ X, D = I - eta*L.

    Precomputes D once; applies a single dense (or sparse) matmul per call.

    Complexity:
        precompute : O(N²) — dense; O(kN) if L is sparse
        __call__   : O(N²d) — dense; O(kNd) if D stored sparse
    """

    def _build_operator(self, L: Tensor) -> Tensor:
        # D = I - eta * L  (always dense — sparse I - eta*L is dense anyway)
        L_dense = L.to_dense() if L.is_sparse else L
        N = L_dense.shape[0]
        I = torch.eye(N, dtype=L_dense.dtype, device=L_dense.device)
        return I - self.eta * L_dense     # (N, N) dense

    def __call__(self, X: Tensor) -> Tensor:
        self._check_ready()
        if self.eta == 0.0:
            return X
        return _matmul(self._op, X)


class IterativeDiffusion(DiffusionOperator):
    """
    Multi-step explicit Euler: X' = D^steps @ X, D = I - eta*L.

    Precomputes D once; applies it ``steps`` times per call.
    Equivalent to (I - eta*L)^steps X without re-recomputing L.

    Over-smoothing guard: stops early if ||X_t - X_{t-1}||_F < tol.

    Complexity:
        precompute : O(N²)
        __call__   : O(steps * N²d)
    """

    def __init__(self, eta: float, steps: int = 3,
                 early_stop_tol: float = 1e-6) -> None:
        super().__init__(eta=eta, steps=steps)
        self.early_stop_tol = early_stop_tol

    def _build_operator(self, L: Tensor) -> Tensor:
        L_dense = L.to_dense() if L.is_sparse else L
        N = L_dense.shape[0]
        I = torch.eye(N, dtype=L_dense.dtype, device=L_dense.device)
        return I - self.eta * L_dense

    def __call__(self, X: Tensor) -> Tensor:
        self._check_ready()
        if self.eta == 0.0 or self.steps == 0:
            return X
        for _ in range(self.steps):
            X_new = _matmul(self._op, X)
            # Early stopping: over-smoothing guard.
            if torch.norm(X_new - X).item() < self.early_stop_tol:
                break
            X = X_new
        return X


class SpectralDiffusion(DiffusionOperator):
    """
    Exact heat-kernel diffusion: X' = H @ X,
    H = U exp(-eta*Λ) U^T  (eigendecomposition of L).

    Unconditionally stable for all eta > 0.  More expensive than Euler
    methods to precompute but cheap to apply (single matmul).

    Complexity:
        precompute : O(N³) — full symmetric eigendecomposition
        __call__   : O(N²d)
    """

    def _build_operator(self, L: Tensor) -> Tensor:
        L_dense = (L.to_dense() if L.is_sparse else L).float()
        eigenvalues, U = torch.linalg.eigh(L_dense)            # (N,), (N, N)
        H = U @ torch.diag(torch.exp(-self.eta * eigenvalues)) @ U.t()
        return H                                                # (N, N)

    def __call__(self, X: Tensor) -> Tensor:
        self._check_ready()
        if self.eta == 0.0:
            return X
        H = self._op.to(dtype=X.dtype, device=X.device)
        return _matmul(H, X)


class FactoredDiffusion(DiffusionOperator):
    """
    Memory-optimal diffusion using factored form — no explicit L matrix.

    Instead of precomputing and storing the dense D = I - η*L (O(N²) memory),
    this operator stores only:
        * W   — (N, N) sparse adjacency  → O(kN) when sparse
        * deg — (N,) degree vector       → O(N)

    and applies the identity-expanded form of one diffusion step:

        x' = (1 - η·deg) ⊙ x + η · W @ x

    which is algebraically identical to (I - η*L) @ x when L = D - A,
    i.e. for the unnormalised Laplacian.

    Initialisation
    --------------
    Preferred path — call ``precompute_from_graph(W, deg)`` directly
    (provided by ``GraphCache``) to bypass L-construction entirely.

    Fallback path — call ``precompute(L)`` as with other operators;
    the class then recovers (W, deg) from L.

    Args:
        eta:   Diffusion strength η.
        steps: Number of applications per forward call.

    Time complexity:  O(kNd) per step (sparse W) or O(N²d) (dense W)
    Memory:          O(kN) + O(N)  — NO full N×N matrix stored
    """

    def __init__(self, eta: float, steps: int = 1) -> None:
        super().__init__(eta=eta, steps=steps)
        self._W: Optional[Tensor] = None     # sparse or dense adjacency
        self._deg: Optional[Tensor] = None   # (N,) degree

    def precompute_from_graph(
        self, W: Tensor, deg: Tensor
    ) -> "FactoredDiffusion":
        """
        Initialise directly from adjacency W and degree vector — no L needed.

        Args:
            W:   (N, N) adjacency matrix (dense or sparse_coo).
            deg: (N,) float32 degree vector.

        Returns:
            self — enables chaining.
        """
        self._W = W
        self._deg = deg
        self._op = True    # sentinel: marks precomputed
        return self

    def _build_operator(self, L: Tensor) -> bool:
        """
        Fallback: reconstruct (W, deg) from an unnormalised Laplacian.

        For L = D - A:  deg = diag(D) = diag(L)  and  A = diag(deg) - L.
        NOTE: this identity holds ONLY for the unnormalised Laplacian.
        For normalised L, call ``precompute_from_graph`` instead.
        """
        L_dense = L.to_dense() if L.is_sparse else L
        self._deg = L_dense.diagonal().clone()                # (N,)
        self._W = torch.diag(self._deg) - L_dense             # A = D - L
        return True   # sentinel — _op not used as a matrix here

    def __call__(self, X: Tensor) -> Tensor:
        if self._W is None:
            raise RuntimeError(
                "FactoredDiffusion: call precompute(L) or "
                "precompute_from_graph(W, deg) before __call__."
            )
        if self.eta == 0.0 or self.steps == 0:
            return X

        W = self._W
        eta = self.eta

        for _ in range(self.steps):
            # x' = (1 - η·deg) ⊙ x  +  η · W @ x
            if X.dim() == 2:
                scale = (1.0 - eta * self._deg).unsqueeze(-1)  # (N, 1)
                Wx = torch.sparse.mm(W, X) if W.is_sparse else W @ X
                X = scale * X + eta * Wx
            else:
                # 3-D: (S, B, d)  where S = N patterns
                S, B, d = X.shape
                scale = (1.0 - eta * self._deg).view(S, 1, 1)  # (S, 1, 1)
                X_flat = X.reshape(S, B * d)
                Wx_flat = (
                    torch.sparse.mm(W, X_flat) if W.is_sparse else W @ X_flat
                )
                X = scale * X + eta * Wx_flat.reshape(S, B, d)

        return X


# ---------------------------------------------------------------------------
# Backward-compatible functional API
# ---------------------------------------------------------------------------

def apply_diffusion(X: Tensor, L: Tensor, eta: float,
                    mode: str = "simple", steps: int = 3) -> Tensor:
    """
    Backward-compatible functional diffusion dispatch.

    Builds and immediately applies a DiffusionOperator.  For repeated
    use on the same graph, prefer the class-based API which amortises
    the precompute cost.

    Args:
        X:     (N, d) or (S, B, d) input patterns.
        L:     (N, N) graph Laplacian.
        eta:   Diffusion strength / time.
        mode:  'simple', 'iterative', or 'spectral'.
        steps: Iterations (iterative mode only).

    Returns:
        X': Diffused tensor, same shape as X.
    """
    if eta == 0.0:
        return X
    op = DiffusionOperator.create(mode=mode, eta=eta, steps=steps)
    op.precompute(L)
    return op(X)
