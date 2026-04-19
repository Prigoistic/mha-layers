"""
Graph-Regularized (Diffusion-Augmented) Hopfield Attention.

Core idea
---------
Core model / dynamics loop
--------------------------

    x_{t+1} = Attention(D · x_t)    — spec Section 0

In practice the stored key patterns (and optionally queries) are diffused
before the Hopfield attention layer:

    for t in range(T):             # via DynamicsEngine.run_diffusion
        K' = D @ K                 # D = I - η*L  (or factored form)
        Q' = D @ Q
    Attention = softmax(β Q' K'ᵀ) V   — dense O(N²) or graph O(kN)

Optionally, post-softmax attention weights are also smoothed over the graph
(logit-level diffusion):
    weights' = diffuse(weights, L_K, η_logit)
    output   = weights' @ V

Four diffusion modes (implemented in ``DiffusionOperator`` subclasses):
    * **factored**  — x' = (1-η·deg)⊙x + η·W@x. O(kNd), no L formed.
    * **simple**    — D = I - η*L, applied once. O(N²d).
    * **iterative** — D^T X. Deeper smoothing; early-stop guard. O(T·N²d).
    * **spectral**  — H = U exp(-η Λ) U^T. Exact heat kernel. O(N³) precompute.

Two attention modes (implemented in ``AttentionOperator``):
    * **dense** (default) — full softmax(β Q Kᵀ) V. O(N²d). Exact baseline.
    * **graph**           — attend only to kNN neighbors. O(kNd). Faster.

Design
------
``DiffusedHopfield`` is a drop-in replacement for ``Hopfield``.  It inherits
the full constructor and only overrides ``_associate`` to splice in diffusion.

Internally it delegates all graph/diffusion work to:
    * ``GraphCache``     — builds and caches (W, deg, adj_idx, L, op) once.
    * ``DynamicsEngine`` — runs T-step loop; no rebuild inside loop.
    * ``AttentionOperator`` — dense or graph-constrained attention.
    * ``EnergyTracker``  — (optional) per-step energy + early-stop.

This satisfies Open-Closed: new diffusion modes are added by subclassing
``DiffusionOperator`` in ``diffusion.py``; no changes needed here.
"""

from typing import Dict, Optional, Tuple, Union

import copy

import torch
import torch.nn.functional as F
from torch import Tensor

from . import Hopfield
from .attention_operator import AttentionOperator
from .diffusion import DiffusionOperator
from .dynamics_engine import DiffusionConfig, DynamicsEngine, EnergyTracker, GraphCache


class DiffusedHopfield(Hopfield):
    """
    Hopfield association module augmented with Laplacian graph diffusion.

    Adds diffusion-specific parameters on top of the full ``Hopfield`` API.
    All graph construction, caching, and diffusion are delegated to
    ``GraphCache`` and ``DynamicsEngine``; this class only orchestrates the
    pre-processing hook inside ``_associate``.

    New parameters (beyond the standard Hopfield signature)
    -------------------------------------------------------
    eta                  : float  — Diffusion strength η. Default: 0.1.
    k_neighbors          : int    — kNN graph degree. Default: 5.
    use_normalized_laplacian : bool — Symmetric-normalised L. Default: True.
    diffuse_query        : bool   — Diffuse query patterns. Default: False.
    diffuse_key          : bool   — Diffuse key patterns. Default: True.
    diffusion_mode       : str    — 'factored'|'simple'|'iterative'|'spectral'.
    diffusion_steps      : int    — Iterations for iterative/spectral mode.
    attention_mode       : str    — 'dense' (O(N²)) or 'graph' (O(kN)).
    use_sparse           : bool   — Sparse adjacency for O(kN) products.
    use_logit_diffusion  : bool   — Smooth post-softmax weights.
    logit_eta            : float  — η for logit-level diffusion.
    adaptive_eta         : bool   — Entropy-gated η scaling.
    adaptive_temperature : float  — Sigmoid temperature for adaptive η.
    adaptive_threshold   : float  — Entropy midpoint for adaptive η.
    cache_graph          : bool   — Cache graph/operator between passes.
    energy_stop_tol      : float  — Early-stop tolerance; 0 = off.

    All remaining kwargs are forwarded unchanged to ``Hopfield``.
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        output_size: Optional[int] = None,
        pattern_size: Optional[int] = None,
        num_heads: int = 1,
        scaling: Optional[Union[float, Tensor]] = None,
        update_steps_max: Optional[Union[int, Tensor]] = 0,
        update_steps_eps: Union[float, Tensor] = 1e-4,

        normalize_stored_pattern: bool = True,
        normalize_stored_pattern_affine: bool = True,
        normalize_stored_pattern_eps: float = 1e-5,
        normalize_state_pattern: bool = True,
        normalize_state_pattern_affine: bool = True,
        normalize_state_pattern_eps: float = 1e-5,
        normalize_pattern_projection: bool = True,
        normalize_pattern_projection_affine: bool = True,
        normalize_pattern_projection_eps: float = 1e-5,
        normalize_hopfield_space: bool = False,
        normalize_hopfield_space_affine: bool = False,
        normalize_hopfield_space_eps: float = 1e-5,
        stored_pattern_as_static: bool = False,
        state_pattern_as_static: bool = False,
        pattern_projection_as_static: bool = False,
        pattern_projection_as_connected: bool = False,
        stored_pattern_size: Optional[int] = None,
        pattern_projection_size: Optional[int] = None,

        batch_first: bool = True,
        association_activation: Optional[str] = None,
        dropout: float = 0.0,
        input_bias: bool = True,
        concat_bias_pattern: bool = False,
        add_zero_association: bool = False,
        disable_out_projection: bool = False,

        # --- Diffusion-specific parameters ---
        eta: float = 0.1,
        k_neighbors: int = 5,
        use_normalized_laplacian: bool = True,
        diffuse_query: bool = False,
        diffuse_key: bool = True,
        diffusion_mode: str = "factored",
        diffusion_steps: int = 3,
        attention_mode: str = "dense",
        use_sparse: bool = False,
        use_logit_diffusion: bool = False,
        logit_eta: Optional[float] = None,
        adaptive_eta: bool = False,
        adaptive_temperature: float = 5.0,
        adaptive_threshold: float = 1.0,
        cache_graph: bool = True,
        energy_stop_tol: float = 0.0,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            pattern_size=pattern_size,
            num_heads=num_heads,
            scaling=scaling,
            update_steps_max=update_steps_max,
            update_steps_eps=update_steps_eps,
            normalize_stored_pattern=normalize_stored_pattern,
            normalize_stored_pattern_affine=normalize_stored_pattern_affine,
            normalize_stored_pattern_eps=normalize_stored_pattern_eps,
            normalize_state_pattern=normalize_state_pattern,
            normalize_state_pattern_affine=normalize_state_pattern_affine,
            normalize_state_pattern_eps=normalize_state_pattern_eps,
            normalize_pattern_projection=normalize_pattern_projection,
            normalize_pattern_projection_affine=normalize_pattern_projection_affine,
            normalize_pattern_projection_eps=normalize_pattern_projection_eps,
            normalize_hopfield_space=normalize_hopfield_space,
            normalize_hopfield_space_affine=normalize_hopfield_space_affine,
            normalize_hopfield_space_eps=normalize_hopfield_space_eps,
            stored_pattern_as_static=stored_pattern_as_static,
            state_pattern_as_static=state_pattern_as_static,
            pattern_projection_as_static=pattern_projection_as_static,
            pattern_projection_as_connected=pattern_projection_as_connected,
            stored_pattern_size=stored_pattern_size,
            pattern_projection_size=pattern_projection_size,
            batch_first=batch_first,
            association_activation=association_activation,
            dropout=dropout,
            input_bias=input_bias,
            concat_bias_pattern=concat_bias_pattern,
            add_zero_association=add_zero_association,
            disable_out_projection=disable_out_projection,
        )

        # Build unified config
        _beta = float(scaling) if isinstance(scaling, (int, float)) else 1.0
        self._diff_cfg = DiffusionConfig(
            eta=eta,
            beta=_beta,
            steps=diffusion_steps,
            diffusion_mode=diffusion_mode,
            attention_mode=attention_mode,
            k_neighbors=k_neighbors,
            use_normalized_laplacian=use_normalized_laplacian,
            use_sparse=use_sparse,
            diffuse_key=diffuse_key,
            diffuse_query=diffuse_query,
            use_logit_diffusion=use_logit_diffusion,
            logit_eta=logit_eta if logit_eta is not None else eta,
            adaptive_eta=adaptive_eta,
            adaptive_temperature=adaptive_temperature,
            adaptive_threshold=adaptive_threshold,
            cache_graph=cache_graph,
            energy_stop_tol=energy_stop_tol,
        )

        # Separate cache per role (key vs query patterns may differ in shape)
        self._key_cache = GraphCache(self._diff_cfg)
        self._query_cache = GraphCache(self._diff_cfg)

        # Attention operator — dense O(N²) or graph-constrained O(kN)
        self._attn_op = AttentionOperator(beta=_beta, mode=attention_mode)

        # Energy tracker (shared across key/query for the last step)
        self._energy_tracker: Optional[EnergyTracker] = (
            EnergyTracker(
                beta=_beta,
                eta=eta,
                tol=energy_stop_tol,
            )
            if energy_stop_tol > 0.0 else None
        )

        # Expose scalar hypers for backward compat property access
        self.eta = eta
        self.k_neighbors = k_neighbors
        self.use_normalized_laplacian = use_normalized_laplacian
        self.diffuse_query = diffuse_query
        self.diffuse_key = diffuse_key
        self.diffusion_mode = diffusion_mode
        self.diffusion_steps = diffusion_steps
        self.attention_mode = attention_mode
        self.use_logit_diffusion = use_logit_diffusion
        self.logit_eta = logit_eta if logit_eta is not None else eta
        self.adaptive_eta = adaptive_eta
        self.adaptive_temperature = adaptive_temperature
        self.adaptive_threshold = adaptive_threshold
        self.cache_graph = cache_graph

        # Cache Hopfield name-mangled attrs used in association_core call
        self._d_update_steps_max = update_steps_max
        self._d_update_steps_eps = update_steps_eps

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Clear cached Laplacians and operators. Call when patterns change."""
        self._key_cache.invalidate()
        self._query_cache.invalidate()

    # ------------------------------------------------------------------
    # Override _associate to inject diffusion
    # ------------------------------------------------------------------

    def _associate(
        self,
        data: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
        return_raw_associations: bool = False,
        return_projected_patterns: bool = False,
        stored_pattern_padding_mask: Optional[Tensor] = None,
        association_mask: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], ...]:
        """
        Hopfield association with graph-diffusion pre-processing.

        Mirrors ``Hopfield._associate`` exactly; inserts a DynamicsEngine
        diffusion pass on stored / state patterns after optional LayerNorm
        but before HopfieldCore attention.

        No graph is rebuilt if the same patterns are passed again
        (GraphCache returns the cached DiffusionOperator in O(1)).
        """
        assert (type(data) == Tensor) or (
            (type(data) == tuple) and (len(data) == 3)
        ), (
            "either one tensor or a 3-tuple "
            "(stored_pattern, state_pattern, pattern_projection) must be provided."
        )

        if type(data) == Tensor:
            stored_pattern = state_pattern = pattern_projection = data
        else:
            stored_pattern, state_pattern, pattern_projection = data

        # --- batch_first transpose ---
        stored_pattern, state_pattern, pattern_projection = self._maybe_transpose(
            stored_pattern, state_pattern, pattern_projection
        )

        # --- Optional LayerNorm (mirroring Hopfield._associate) ---
        if self.norm_stored_pattern is not None:
            stored_pattern = self.norm_stored_pattern(
                input=stored_pattern.reshape(-1, stored_pattern.shape[2])
            ).reshape(stored_pattern.shape)

        if self.norm_state_pattern is not None:
            state_pattern = self.norm_state_pattern(
                input=state_pattern.reshape(-1, state_pattern.shape[2])
            ).reshape(state_pattern.shape)

        if self.norm_pattern_projection is not None:
            pattern_projection = self.norm_pattern_projection(
                input=pattern_projection.reshape(-1, pattern_projection.shape[2])
            ).reshape(pattern_projection.shape)

        # --- Full dynamics loop: interleaved diffusion + attention (§0, §4) ---
        cfg = self._diff_cfg
        _W_k = _deg_k = _adj_k = L_k = op_k = None
        if cfg.eta > 0.0 and (cfg.diffuse_key or cfg.diffuse_query):
            key_repr = stored_pattern.detach().mean(dim=1).float()
            _W_k, _deg_k, _adj_k, L_k, op_k = self._key_cache.get(key_repr)

            # Adaptive η: scale by attention entropy before dynamics loop
            eta_eff = cfg.eta
            if cfg.adaptive_eta:
                raw_logits = torch.bmm(
                    state_pattern.permute(1, 0, 2),
                    stored_pattern.permute(1, 2, 0),
                )   # (B, L, S)
                engine_tmp = DynamicsEngine(op_k)
                eta_eff = engine_tmp.compute_adaptive_eta(
                    raw_logits, cfg.eta,
                    cfg.adaptive_temperature, cfg.adaptive_threshold,
                )
                if abs(eta_eff - cfg.eta) / (cfg.eta + 1e-9) > 0.05:
                    from .diffusion import FactoredDiffusion
                    if cfg.diffusion_mode == "factored":
                        op_k = FactoredDiffusion(
                            eta=eta_eff, steps=cfg.steps
                        ).precompute_from_graph(_W_k, _deg_k)
                    else:
                        op_k = DiffusionOperator.create(
                            cfg.diffusion_mode, eta_eff, cfg.steps
                        ).precompute(L_k)

            # Single-step copy for dynamics (outer loop controls iteration count)
            op_dyn = copy.copy(op_k)
            op_dyn.steps = 1

            engine = DynamicsEngine(
                diffusion_op=op_dyn,
                attention_op=self._attn_op,
                steps=self.diffusion_steps,
                energy_tracker=self._energy_tracker,
            )
            state_pattern, stored_pattern = engine.run_dynamics(
                Q=state_pattern, K=stored_pattern, V=pattern_projection,
                adj_indices=_adj_k, L=L_k, W=_W_k, deg=_deg_k,
                diffuse_query=cfg.diffuse_query, diffuse_key=cfg.diffuse_key,
            )

        # --- Logit-level diffusion (post-softmax weight smoothing) ---
        # Injected after core association when requested.
        # We apply it on the raw-association output before returning.
        result = self.association_core(
            query=state_pattern,
            key=stored_pattern,
            value=pattern_projection,
            key_padding_mask=stored_pattern_padding_mask,
            need_weights=cfg.use_logit_diffusion,   # need weights for smoothing
            attn_mask=association_mask,
            scaling=self.scaling,
            update_steps_max=self._d_update_steps_max,
            update_steps_eps=self._d_update_steps_eps,
            return_raw_associations=return_raw_associations,
            return_pattern_projections=return_projected_patterns,
        )

        if cfg.use_logit_diffusion and cfg.logit_eta > 0.0:
            # result[0] = output, result[1] = attn_weights (B, H, L, S)
            # Smooth weights over the key graph and re-normalise.
            attn_weights = result[1]   # (B, H, L, S) or None
            if attn_weights is not None:
                # Reuse cached graph if available; otherwise build from keys
                if op_k is None:
                    key_repr = stored_pattern.detach().mean(dim=1).float()
                    _W_k, _deg_k, _adj_k, L_k, op_k = self._key_cache.get(key_repr)
                op_logit = op_k
                # Treat (S,) distribution per query as the diffusion signal.
                # Flatten to (S, B*H*L), diffuse, reshape and renormalise.
                B, H, L_q_len, S = attn_weights.shape
                w_flat = attn_weights.permute(3, 0, 1, 2).reshape(S, B * H * L_q_len)
                w_diff = op_logit(w_flat)          # diffuse along S
                w_diff = w_diff.reshape(S, B, H, L_q_len).permute(1, 2, 3, 0)
                w_diff = w_diff.clamp(min=0.0)
                w_diff = w_diff / (w_diff.sum(dim=-1, keepdim=True) + 1e-9)
                # Recompute output with smoothed weights.
                V = pattern_projection.permute(1, 0, 2)   # (B, S, d)
                out_smooth = torch.einsum("bhls,bsd->bld", w_diff, V)
                out_smooth = out_smooth.permute(1, 0, 2)  # (L, B, d)
                result = (out_smooth,) + result[1:]

        return result

    # ------------------------------------------------------------------
    # Config dict API
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, object]:
        """Return a JSON-serialisable dict of all diffusion hyperparameters."""
        return self._diff_cfg.to_dict()

