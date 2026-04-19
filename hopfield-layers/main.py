"""
Graph-Regularized Hopfield Attention — Experiment Runner
=========================================================

Usage
-----
Run all experiments:
    python main.py

Run a specific experiment:
    python main.py --exp noise
    python main.py --exp ablation
    python main.py --exp attention
    python main.py --exp steps
    python main.py --exp modes
    python main.py --exp logit
    python main.py --exp all       (default)

Key flags
---------
    --N       Number of stored patterns     (default: 200)
    --d       Pattern dimensionality        (default: 32)
    --beta    Hopfield scaling β            (default: 12.0)
    --eta     Diffusion strength η          (default: 0.08)
    --k       kNN graph neighbours          (default: 7)
    --M       Queries per noise level       (default: 300)
    --mode    Diffusion mode                (default: iterative)
    --steps   Diffusion steps               (default: 3)
    --seed    Global random seed            (default: 42)
    --results Output directory              (default: results)

Results are saved under <results>/  as CSV files and plots.
"""

import argparse
import sys
import os
import random

import numpy as np
import torch


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args():
    p = argparse.ArgumentParser(description="Graph-Regularized Hopfield Experiments")
    p.add_argument("--exp",     default="all",
                   choices=["all", "noise", "ablation", "attention",
                            "steps", "modes", "logit", "bench"],
                   help="Which experiment(s) to run")
    p.add_argument("--N",       type=int,   default=200,   help="Number of stored patterns")
    p.add_argument("--d",       type=int,   default=64,    help="Pattern dimensionality")
    p.add_argument("--beta",    type=float, default=12.0,  help="Hopfield scaling beta")
    p.add_argument("--eta",     type=float, default=0.10,  help="Diffusion strength eta")
    p.add_argument("--k",       type=int,   default=7,     help="kNN graph neighbours")
    p.add_argument("--M",       type=int,   default=500,   help="Queries per noise level")
    p.add_argument("--mode",    default="spectral",
                   choices=["simple", "iterative", "spectral"],
                   help="Diffusion mode")
    p.add_argument("--steps",   type=int,   default=3,     help="Diffusion steps (iterative mode)")
    p.add_argument("--clusters", type=int,  default=20,    help="Number of pattern clusters")
    p.add_argument("--seed",    type=int,   default=42,    help="Global random seed")
    p.add_argument("--results", default="results",         help="Output directory")
    return p.parse_args()


def run_noise(args):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 — Noise Robustness (H1)")
    print("=" * 60)
    from src.experiments.noise_robustness import run_noise_robustness
    from src.utils.visualization import plot_noise_vs_accuracy
    import os

    df = run_noise_robustness(
        N=args.N, d=args.d, beta=args.beta, eta=args.eta, k=args.k,
        M=args.M, seed=args.seed, results_dir=args.results,
        diffusion_mode=args.mode, diffusion_steps=args.steps,
        n_clusters=args.clusters,
    )
    plot_noise_vs_accuracy(
        df, save_path=os.path.join(args.results, "plots", "noise_vs_accuracy.png")
    )
    print("Plot saved → results/plots/noise_vs_accuracy.png")
    return df


def run_ablation(args):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — Ablation Study (H2)")
    print("=" * 60)
    from src.experiments.ablation import run_ablation as _run
    from src.utils.visualization import plot_ablation
    import os

    df = _run(
        N=args.N, d=args.d, beta=args.beta, eta=args.eta, k=args.k,
        M=args.M, noise_level=0.30, seed=args.seed, results_dir=args.results,
        diffusion_mode=args.mode, diffusion_steps=args.steps,
        n_clusters=args.clusters,
    )
    plot_ablation(
        df, save_path=os.path.join(args.results, "plots", "ablation.png")
    )
    print("Plot saved → results/plots/ablation.png")
    return df


def run_attention(args):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 — η Sweep + Attention Analysis (H3, H4)")
    print("=" * 60)
    from src.experiments.attention_analysis import run_attention_analysis
    from src.utils.visualization import plot_eta_sweep, plot_attention_entropy
    import os

    df, entropy_per_eta = run_attention_analysis(
        N=args.N, d=args.d, beta=args.beta, k=args.k, M=args.M,
        noise_level=0.30, seed=args.seed, results_dir=args.results,
        diffusion_mode=args.mode, diffusion_steps=args.steps,
        n_clusters=args.clusters,
    )
    plot_eta_sweep(
        df, save_path=os.path.join(args.results, "plots", "eta_sweep.png")
    )
    print("Plot saved → results/plots/eta_sweep.png")

    # Entropy histogram: baseline vs best-η model
    best_row = df.loc[df["accuracy"].idxmax()]
    best_eta = best_row["eta"]
    baseline_key = "η=0.0"
    best_key = f"η={best_eta}"
    subset = {k: v for k, v in entropy_per_eta.items()
              if k in (baseline_key, best_key)}
    plot_attention_entropy(
        subset,
        save_path=os.path.join(args.results, "plots", "attention_entropy.png"),
        title=f"H4: Attention Entropy — Baseline vs Best η={best_eta}",
    )
    print("Plot saved → results/plots/attention_entropy.png")
    return df


def run_steps(args):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4 — Diffusion Steps Sweep")
    print("=" * 60)
    from src.experiments.steps_sweep import run_steps_sweep
    from src.utils.visualization import plot_steps_sweep, plot_energy_vs_steps
    import os

    df_acc, df_energy = run_steps_sweep(
        N=args.N, d=args.d, beta=args.beta, eta=args.eta, k=args.k,
        M=args.M, noise_level=0.35, seed=args.seed, results_dir=args.results,
    )
    plot_steps_sweep(
        df_acc, save_path=os.path.join(args.results, "plots", "steps_sweep.png")
    )
    print("Plot saved → results/plots/steps_sweep.png")
    plot_energy_vs_steps(
        df_energy, save_path=os.path.join(args.results, "plots", "energy_vs_steps.png")
    )
    print("Plot saved → results/plots/energy_vs_steps.png")
    return df_acc


def run_modes(args):
    print("\n" + "=" * 60)
    print("EXPERIMENT 5 — Mode Comparison")
    print("=" * 60)
    from src.experiments.mode_comparison import run_mode_comparison
    from src.utils.visualization import plot_mode_comparison, plot_noise_multi_mode
    import os

    df_sweep, df_summary = run_mode_comparison(
        N=args.N, d=args.d, beta=args.beta, eta=args.eta, k=args.k,
        steps=args.steps, M=args.M, seed=args.seed, results_dir=args.results,
    )
    plot_mode_comparison(
        df_summary,
        save_path=os.path.join(args.results, "plots", "mode_comparison.png"),
    )
    print("Plot saved → results/plots/mode_comparison.png")
    plot_noise_multi_mode(
        df_sweep,
        save_path=os.path.join(args.results, "plots", "noise_multi_mode.png"),
    )
    print("Plot saved → results/plots/noise_multi_mode.png")
    return df_summary


def run_logit(args):
    print("\n" + "=" * 60)
    print("EXPERIMENT 6 — Logit vs Feature Diffusion")
    print("=" * 60)
    from src.experiments.logit_vs_feature import run_logit_vs_feature
    from src.utils.visualization import plot_noise_multi_mode
    import os

    df = run_logit_vs_feature(
        N=args.N, d=args.d, beta=args.beta, eta=args.eta, k=args.k,
        steps=args.steps, mode=args.mode, M=args.M,
        seed=args.seed, results_dir=args.results,
    )
    # Rename 'config' to 'mode' for the generic multi-mode plot
    df_plot = df.rename(columns={"config": "mode"})
    plot_noise_multi_mode(
        df_plot,
        save_path=os.path.join(args.results, "plots", "logit_vs_feature.png"),
    )
    print("Plot saved → results/plots/logit_vs_feature.png")
    return df


def _print_summary(noise_df, ablation_df, attn_df, steps_df, modes_df, logit_df):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if noise_df is not None:
        noise_df = noise_df.copy()
        noise_df["gain"] = noise_df["diffused_accuracy"] - noise_df["baseline_accuracy"]
        best_row = noise_df.loc[noise_df["gain"].idxmax()]
        gain = best_row["gain"]
        print(f"H1 (noise={best_row['noise_level']:.2f}): "
              f"baseline={best_row['baseline_accuracy']:.3f} "
              f"diffused={best_row['diffused_accuracy']:.3f}  Δ={gain:+.3f}")

    if ablation_df is not None:
        best = ablation_df.loc[ablation_df["accuracy"].idxmax(), "config"]
        print(f"H2 (ablation):  best config = {best}")

    if attn_df is not None:
        best_eta = attn_df.loc[attn_df["accuracy"].idxmax(), "eta"]
        print(f"H3 (eta sweep): best η = {best_eta}")
        baseline_H = attn_df.loc[attn_df["eta"] == 0.0, "mean_entropy"].values
        best_H     = attn_df.loc[attn_df["eta"] == best_eta, "mean_entropy"].values
        if len(baseline_H) and len(best_H):
            print(f"H4 (entropy):   baseline mean H={baseline_H[0]:.3f}"
                  f"  best-η mean H={best_H[0]:.3f}")

    if steps_df is not None:
        for mode, grp in steps_df.groupby("mode"):
            best_s = grp.loc[grp["accuracy"].idxmax()]
            print(f"Steps sweep ({mode}): best steps={int(best_s['steps'])}"
                  f"  acc={best_s['accuracy']:.3f}")

    if modes_df is not None:
        best_mode = modes_df.loc[modes_df["accuracy"].idxmax()]
        print(f"Mode comparison: best={best_mode['mode']}"
              f"  acc={best_mode['accuracy']:.3f}")

    if logit_df is not None:
        at_noise = logit_df[logit_df["noise_level"] == 0.35]
        if at_noise.empty:
            at_noise = logit_df[logit_df["noise_level"] == logit_df["noise_level"].max()]
        for _, row in at_noise.iterrows():
            print(f"Logit/Feature ({row['config']}): acc={row['accuracy']:.3f}")

    print("\nAll CSV results → results/")
    print("All plots       → results/plots/")


def main():
    args = _parse_args()
    _set_seeds(args.seed)

    os.makedirs(os.path.join(args.results, "plots"), exist_ok=True)

    noise_df = ablation_df = attn_df = steps_df = modes_df = logit_df = None

    if args.exp in ("all", "noise"):
        noise_df = run_noise(args)

    if args.exp in ("all", "ablation"):
        ablation_df = run_ablation(args)

    if args.exp in ("all", "attention"):
        attn_df = run_attention(args)

    if args.exp in ("all", "steps"):
        steps_df = run_steps(args)

    if args.exp in ("all", "modes"):
        modes_df = run_modes(args)

    if args.exp in ("all", "logit"):
        logit_df = run_logit(args)

    if args.exp == "bench":
        from src.experiments.benchmark import run_benchmark
        run_benchmark(N=args.N, d=args.d, k=args.k, eta=args.eta,
                      seed=args.seed, results_dir=args.results)

    if args.exp == "all":
        _print_summary(noise_df, ablation_df, attn_df,
                       steps_df, modes_df, logit_df)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
