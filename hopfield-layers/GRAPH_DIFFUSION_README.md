# Graph-Regularized Hopfield Attention (Diffusion-Augmented)

## Method Summary

Standard scaled-dot-product / Hopfield attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}(\beta\, Q K^\top)\, V$$

This project augments the attention with **one-step graph Laplacian diffusion** applied to the stored patterns (keys) before computing the attention:

$$K' = (I - \eta L)\, K, \quad Q' = (I - \eta L)\, Q$$
$$\text{Attention}_\text{diffused} = \text{softmax}(\beta\, Q' K^{\prime\top})\, V$$

where:

| Symbol | Meaning |
|--------|---------|
| $L$ | Symmetric-normalised graph Laplacian built from the cosine-similarity kNN graph of the stored patterns |
| $\eta$ | Diffusion strength (stable range: $0 < \eta < 0.5$ with normalised $L$) |

The intuition is that diffusion smooths noisy or spurious pattern activations along the similarity structure of the stored patterns, making retrieval more robust under input corruption.

---

## Project Structure

```
hopfield-layers/
├── hflayers/
│   ├── diffused_attention.py   ← DiffusedHopfield (drop-in Hopfield replacement)
│   ├── diffusion.py            ← diffuse(X, L, eta) operator
│   ├── graph/
│   │   ├── build_graph.py      ← cosine-similarity + kNN graph
│   │   └── laplacian.py        ← unnormalised and sym-normalised Laplacians
│   ├── activation.py           ← HopfieldCore (unchanged)
│   ├── functional.py           ← hopfield_core_forward (unchanged)
│   └── __init__.py             ← exports Hopfield, DiffusedHopfield, ...
│
├── src/
│   ├── experiments/
│   │   ├── noise_robustness.py
│   │   ├── ablation.py
│   │   └── attention_analysis.py
│   └── utils/
│       ├── data_gen.py
│       ├── metrics.py
│       └── visualization.py
│
├── results/           ← CSV logs + plots (generated at runtime)
├── main.py            ← experiment runner
└── GRAPH_DIFFUSION_README.md  (this file)
```

---

## Installation

```bash
# From the repo root
cd hopfield-layers
pip install -e .
pip install matplotlib pandas numpy
```

Python >= 3.8, PyTorch >= 1.5 required.

---

## How to Run

```bash
cd hopfield-layers

# Run all experiments (noise robustness + ablation + attention analysis)
python main.py

# Run a single experiment
python main.py --exp noise
python main.py --exp ablation
python main.py --exp attention

# Customise hyper-parameters
python main.py --exp all --N 50 --d 64 --beta 8.0 --eta 0.1 --k 5 --M 200 --seed 42
```

Results are written to:

| File | Contents |
|------|---------|
| `results/noise_vs_accuracy.csv` | Accuracy & Hamming distance vs noise level |
| `results/ablation.csv` | Accuracy for {none, Q_only, K_only, both} diffusion |
| `results/attention_analysis.csv` | η sweep: accuracy, entropy, sparsity |
| `results/plots/noise_vs_accuracy.png` | H1 visualisation |
| `results/plots/ablation.png` | H2 visualisation |
| `results/plots/eta_sweep.png` | H3 visualisation |
| `results/plots/attention_entropy.png` | H4 visualisation |

---

## Using `DiffusedHopfield` in Your Code

`DiffusedHopfield` is a drop-in replacement for `Hopfield` with three extra arguments:

```python
from hflayers import DiffusedHopfield

model = DiffusedHopfield(
    input_size=64,
    hidden_size=64,
    output_size=64,
    num_heads=1,
    batch_first=True,
    # --- diffusion params ---
    eta=0.1,                       # diffusion strength
    k_neighbors=5,                 # kNN graph degree
    use_normalized_laplacian=True, # sym-norm Laplacian (recommended)
    diffuse_query=False,           # smooth query patterns too?
    diffuse_key=True,              # smooth stored patterns (primary effect)
)

# forward -- same interface as Hopfield
output = model(input_tensor)                               # single tensor
output = model((stored_patterns, state_patterns, values))  # tuple of three
```

---

## Key Hypotheses

| ID | Statement | Validated by |
|----|-----------|--------------|
| H1 | Diffusion improves recall under noise | `noise_vs_accuracy.csv` |
| H2 | K-diffusion is the dominant contributor; both >= K_only > Q_only > none | `ablation.csv` |
| H3 | Optimal η exists; accuracy is non-monotonic in η | `attention_analysis.csv` |
| H4 | Diffusion shifts/controls attention entropy distribution | `attention_analysis.csv` + entropy plot |

---

## Key Results (fill in after running)

Run `python main.py` and paste the printed SUMMARY here.
