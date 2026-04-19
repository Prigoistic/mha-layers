# Spec Alignment Audit: Diffusion-Attention Dynamical Memory System

**Scope:** Cross-reference of all implementation files against the 10-section engineering spec  
**Date:** Audit of current codebase state

---

## Summary

| Verdict        | Count | Details                                    |
|----------------|-------|--------------------------------------------|
| **Fully Aligned**  | 7 / 10 | §1, §2, §3, §5, §7, §8, §9           |
| **Partially Aligned** | 2 / 10 | §4 (dynamics loop), §6 (performance)  |
| **Major Gap**        | 1 / 10 | §0 (core model not wired end-to-end)  |

---

## Section-by-Section Analysis

### §0 — Core Model: `x_{t+1} = Attention(D · x_t)` — ⚠️ MAJOR GAP

**Spec requirement:** The full interleaved loop must be the actual forward path of `DiffusedHopfield`.

**Implementation:**
- `DynamicsEngine.run_dynamics()` correctly implements the loop:
  ```python
  for t in range(T):
      Q = diffusion(Q)
      K = diffusion(K)
      Q = attention(Q, K, V)
  ```
- **BUT `DiffusedHopfield._associate()` does NOT call `run_dynamics`.** It calls:
  1. `engine_k.run_diffusion(stored_pattern)` — single-tensor diffusion on keys
  2. `engine_q.run_diffusion(state_pattern)` — single-tensor diffusion on queries
  3. `self.association_core(...)` — original Hopfield attention (NOT `AttentionOperator`)

  This means the *actual forward path* is: `Attention(D·K, D·Q, V)` — diffusion and attention are sequential, not interleaved over T steps.

**Impact:** The `run_dynamics` loop exists as dead code (never invoked from the model). The `AttentionOperator` instance (`self._attn_op`) is constructed in `__init__` but never called in `_associate`. Graph-mode attention (`O(kN)`) is unreachable during inference.

---

### §1 — Module Separation (6 classes, single responsibility) — ✅ ALIGNED

| Required Class     | File                          | SRP Satisfied? |
|--------------------|-------------------------------|----------------|
| `GraphBuilder`     | `hflayers/graph/builder.py`   | ✅ Only builds (W, deg, adj_indices) |
| `LaplacianBuilder` | `hflayers/graph/laplacian_builder.py` | ✅ Only builds L from W |
| `GraphCache`       | `hflayers/dynamics_engine.py` | ✅ Only caches graph objects |
| `DiffusionOperator`| `hflayers/diffusion.py`       | ✅ ABC + 4 strategies |
| `AttentionOperator`| `hflayers/attention_operator.py` | ✅ Dense/graph methods isolated |
| `DynamicsEngine`   | `hflayers/dynamics_engine.py` | ✅ Runs loop, no rebuild |

All 6 classes exist with clear single-responsibility boundaries.

---

### §2 — Graph + Diffusion Optimization (FactoredDiffusion) — ✅ ALIGNED

**Spec requirement:** `x' = (1 - η·deg) ⊙ x + η · W @ x` — no L formed, O(kN) memory.

**Implementation in `FactoredDiffusion`:**
```python
scale = (1.0 - eta * self._deg).unsqueeze(-1)  # (N, 1)
Wx = torch.sparse.mm(W, X) if W.is_sparse else W @ X
X = scale * X + eta * Wx
```
- Formula matches spec exactly ✅
- `precompute_from_graph(W, deg)` bypasses L entirely ✅
- Sparse W support via `torch.sparse.mm` ✅
- Handles both 2D `(N, d)` and 3D `(S, B, d)` inputs ✅
- Fallback `precompute(L)` recovers `(W, deg)` from `L = D - A` ✅
- Registered in factory as `mode="factored"` ✅

---

### §3 — Attention Dual Mode (dense / graph) — ✅ ALIGNED

**Spec requirement:** Two cleanly separated modes; no conditional logic inside shared inner loops.

**Implementation in `AttentionOperator`:**
- `_dense(Q, K, V)`: `logits = beta * Q @ K.T → softmax → weights @ V`. O(N²d). ✅
- `_graph(Q, K, V, adj_indices)`: Gather `K[adj_indices]` → per-neighbor dot products → softmax → weighted sum. O(kNd). ✅
- **No N×N matrix formed in graph mode** ✅
- **Dense mode never touches adj_indices** ✅
- Modes dispatched by `__call__` to separate methods (§3.4 satisfied) ✅

**Caveat:** Graph-mode attention is currently unreachable via `DiffusedHopfield` (see §0).

---

### §4 — Dynamics Loop — ⚠️ PARTIALLY ALIGNED

**What's aligned:**
- `DynamicsEngine.run_dynamics()` implements the exact spec loop ✅
- No graph rebuild inside loop ✅
- No L recompute inside loop ✅
- Early-stop via `EnergyTracker.step()` when `|E_t - E_{t-1}| < tol` ✅

**What's NOT aligned:**
1. **`run_dynamics` is not wired into `DiffusedHopfield._associate`** (see §0).
2. **Dimensionality mismatch:** `run_dynamics` expects 2D `(N, d)` tensors. `DiffusedHopfield._associate` works with 3D `(S, B, d)`. Wiring requires per-batch iteration or 3D support in `AttentionOperator`.
3. **`run_diffusion` step asymmetry:** Without energy tracking, `run_diffusion` calls `self._diff_op(X)` exactly once. The operator's internal `steps` loop runs. But with energy tracking, the engine's `self._steps` outer loop multiplies with the operator's inner steps, potentially over-diffusing.

---

### §5 — Energy Tracking — ✅ ALIGNED (with caveat)

**Spec requirement:** `E = -(β · Q @ K^T).mean() + η · tr(K^T L K) / N`

**Implementation in `EnergyTracker.step()`:**
```python
affinity   = -(self.beta * Q @ K.t()).mean()
smoothness = self.eta * torch.trace(K.t() @ L @ K) / K.shape[0]
energy     = (affinity + smoothness).item()
```
- Formula matches spec exactly ✅
- Early-stop: `|E_t - E_{t-1}| < tol` ✅
- History tracked per step ✅

**Caveat:** Energy tracking requires `L`, but the default `factored` mode sets `L=None`. The engine silently disables tracking when `L is None`. This means **energy tracking + factored mode are mutually exclusive** — the two recommended defaults are incompatible.

---

### §6 — Performance Rules — ⚠️ PARTIALLY ALIGNED

| Rule                      | Status      | Notes |
|---------------------------|-------------|-------|
| Sparse-first storage      | ✅          | `GraphBuilder(use_sparse=True)` returns sparse W |
| O(1) cache hit            | ✅          | `GraphCache` keyed by `data_ptr()` |
| No redundant computation  | ✅          | Operator precomputed once, reused |
| Batched ops               | ✅          | torch matmuls throughout |
| Memory reuse inside loop  | ⚠️ Partial | `Q = self._diff_op(Q)` allocates new tensor each step (Python semantics — not truly in-place). Minor. |
| `adj_indices` usage       | ⚠️ Waste   | `_adj_k` from cache is computed but never consumed in `_associate` when `attention_mode='graph'`. |

---

### §7 — Config System — ✅ ALIGNED

**`DiffusionConfig` fields present:**

| Field                       | Present? | Default       |
|-----------------------------|----------|---------------|
| `eta`                       | ✅       | 0.1           |
| `beta`                      | ✅       | 1.0           |
| `steps`                     | ✅       | 3             |
| `diffusion_mode`            | ✅       | "factored"    |
| `attention_mode`            | ✅       | "dense"       |
| `k_neighbors`               | ✅       | 5             |
| `use_normalized_laplacian`  | ✅       | True          |
| `use_sparse`                | ✅       | False         |
| `diffuse_key`               | ✅       | True          |
| `diffuse_query`             | ✅       | False         |
| `use_logit_diffusion`       | ✅       | False         |
| `logit_eta`                 | ✅       | None          |
| `adaptive_eta`              | ✅       | False         |
| `adaptive_temperature`      | ✅       | 5.0           |
| `adaptive_threshold`        | ✅       | 1.0           |
| `cache_graph`               | ✅       | True          |
| `energy_stop_tol`           | ✅       | 0.0           |

`to_dict()` serialization via `dataclasses.asdict` ✅

---

### §8 — Code Quality (SOLID, DRY, types, docs) — ✅ ALIGNED

- **Single Responsibility:** Each class has one job ✅
- **Open-Closed:** New diffusion modes via subclassing, factory registration ✅
- **DRY:** Graph logic in `GraphBuilder` only; Laplacian in `LaplacianBuilder` only ✅
- **Type annotations:** All public methods annotated ✅
- **Docstrings:** All classes and methods documented with args/returns/complexity ✅
- **Strategy pattern:** `DiffusionOperator` ABC with factory ✅

---

### §9 — Validation Requirements — ✅ ALIGNED

| Requirement                    | Status |
|--------------------------------|--------|
| Graph built once, cached       | ✅     |
| L reused (not recomputed)      | ✅     |
| Sparse ops where enabled       | ✅     |
| No redundant compute in loop   | ✅     |
| Backward compatibility         | ✅ (noise experiment confirmed) |

---

## Top 3 Issues (Ranked by Severity)

### 1. CRITICAL — Full dynamics loop not connected to forward path

`DynamicsEngine.run_dynamics()` (the interleaved diffusion→attention loop) is never called from `DiffusedHopfield._associate()`. The model's actual forward path does sequential diffusion-then-attention, not the iterative `for t: Q=diff(Q); K=diff(K); Q=attn(Q,K,V)` required by §0 and §4.

**Root cause:** `_associate` delegates attention to `self.association_core` (Hopfield's internal `HopfieldCore`), which handles multi-head projections, masking, dropout, etc. Replacing it with `AttentionOperator._dense` would lose those features. The design tension is: full dynamics loop vs. Hopfield-compatible attention pipeline.

**Fix options:**
- **Option A (minimal):** Call `run_dynamics` on 2D representative patterns as a pre-processing step before `association_core`. The dynamics loop refines Q/K in embedding space; `association_core` then handles the final multi-head attention.
- **Option B (full):** Replace `association_core` with `AttentionOperator` inside the loop, re-implementing multi-head projection, masking, and dropout within `AttentionOperator`.

### 2. MODERATE — `AttentionOperator` is dead code in model forward path

`self._attn_op = AttentionOperator(...)` is constructed in `__init__` but never invoked in `_associate`. Setting `attention_mode='graph'` has no effect during inference.

**Fix:** Wire `self._attn_op` call into `_associate` (tied to Issue #1).

### 3. MODERATE — Energy tracking incompatible with factored mode

Factored mode (`L=None`) silently disables `EnergyTracker`. Since factored is the default and recommended mode, energy-based early stopping is effectively unusable out of the box.

**Fix options:**
- Compute energy using factored form: `tr(K^T L K) = deg ⊙ ||k_i||² - K^T W K`. Avoids forming L.
- Or: build L lazily only when energy tracking is requested, while keeping factored diffusion.

---

## Minor Issues

| # | Issue | Severity |
|---|-------|----------|
| 4 | `adj_indices` computed by cache but never consumed in `_associate` when `attention_mode='graph'` | Low (wasted O(kN) compute) |
| 5 | `run_diffusion` without tracking applies op once; with tracking applies `engine_steps` times — step count is asymmetric | Low (confusing API) |
| 6 | `run_dynamics` only handles 2D; forward path uses 3D `(S,B,d)` tensors — blocks wiring | Blocks fix for Issue #1 |

---

## Conclusion

The **module architecture, diffusion operators, attention operators, config system, and code quality** are all fully spec-aligned. The individual components are correct and well-designed.

The **critical gap** is the wiring: `DiffusedHopfield._associate` uses the backward-compatible `run_diffusion` + `association_core` path instead of the full `run_dynamics` loop. This makes `AttentionOperator` and the interleaved dynamics loop effectively dead code during model inference.

Fixing this requires either (a) a 2D pre-processing `run_dynamics` call before `association_core`, or (b) a deeper refactor to replace `association_core` with `AttentionOperator` inside the loop while preserving multi-head support.
