from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sae_cbm_eval.constants import CLIP_MODEL_ID, SAE_REPO_ID
from sae_cbm_eval.runtime import (
    configure_runtime_logging,
    ensure_dir,
    load_project_env,
    project_path,
    write_json,
    write_run_manifest,
)


SCRIPT_NAME = "13_plot_all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all final paper figures.")
    parser.add_argument("--results-dir", type=Path, default=project_path("results"))
    parser.add_argument("--figures-dir", type=Path, default=project_path("figures"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text())


def plot_enhanced_pruning_curve(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Pruning curve with random baseline and CLIP baseline overlay."""
    curve = np.load(results_dir / "pruning_curve.npy")
    l1_results = load_json(results_dir / "l1_baseline.json")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        curve[:, 0], curve[:, 1] * 100.0,
        "o-", markersize=2, linewidth=1.5, color="tab:blue",
        label="Iterative pruning (SAE)",
    )

    for r in l1_results:
        ax.plot(float(r["nonzero_features"]), float(r["val_acc"]) * 100.0,
                "s", color="red", markersize=7, zorder=5)
    ax.plot([], [], "s", color="red", markersize=7, label="L1 baseline")

    random_path = results_dir / "random_baseline.json"
    if random_path.exists():
        random_results = load_json(random_path)
        for r in random_results:
            ax.errorbar(
                r["k"], r["random_mean_acc"] * 100.0,
                yerr=r["random_std_acc"] * 100.0,
                fmt="^", color="green", markersize=8, capsize=3, zorder=5,
            )
        ax.plot([], [], "^", color="green", markersize=8, label="Random subset (mean +/- std)")

    clip_path = results_dir / "clip_baseline.json"
    if clip_path.exists():
        clip_results = load_json(clip_path)
        clip_curve = clip_results["pruning_curve"]
        n_feats = [p["n_features"] for p in clip_curve]
        accs = [p["val_acc"] * 100.0 for p in clip_curve]
        ax.plot(n_feats, accs, "d--", markersize=2, linewidth=1.2, color="purple",
                label="Iterative pruning (raw CLIP)")

    baseline_acc_pct = float(curve[0, 1]) * 100.0
    ax.axhline(baseline_acc_pct, color="gray", linestyle="--", alpha=0.5,
               label=f"Full SAE ({baseline_acc_pct:.1f}%)")

    ax.set_xlabel("Number of features")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.set_title("Iterative Feature Pruning on CUB-200")
    fig.tight_layout()
    fig.savefig(figures_dir / "pruning_curve_full.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_histogram(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Histogram of best-match AUROC for retained features."""
    alignment_path = results_dir / "attribute_alignment.json"
    if not alignment_path.exists():
        logging.warning("No alignment results found, skipping histogram.")
        return

    alignment_results = load_json(alignment_path)
    if not alignment_results:
        return

    data = alignment_results[0]
    matches = data["best_matches"]
    aurocs = [m["best_auroc"] for m in matches if not (m["best_auroc"] is None or np.isnan(m["best_auroc"]))]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(aurocs, bins=30, color="steelblue", edgecolor="white", alpha=0.8, label="Pruned features")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (0.5)")
    ax.axvline(data.get("perm_baseline_mean", 0.5), color="orange", linestyle="--",
               alpha=0.7, label=f"Permutation baseline ({data.get('perm_baseline_mean', 0):.3f})")
    ax.set_xlabel("Best-match AUROC")
    ax.set_ylabel("Number of features")
    ax.set_title(f"Feature-Attribute Alignment (delta={data['delta']}, n={data['n_features']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "alignment_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_semantic_scatter(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Scatter plot of AUROC vs CLIP text similarity for labeled features."""
    agreement_path = results_dir / "semantic_agreement.json"
    if not agreement_path.exists():
        logging.warning("No semantic agreement results found, skipping scatter plot.")
        return

    agreement_results = load_json(agreement_path)
    if not agreement_results:
        return

    data = agreement_results[0]
    features = data["feature_agreements"]

    aurocs = []
    sims = []
    for f in features:
        if f.get("best_auroc") is not None and f.get("clip_sim") is not None:
            aurocs.append(f["best_auroc"])
            sims.append(f["clip_sim"])

    if not aurocs:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["tab:green" if (a >= data["auroc_threshold"] and s >= data["sim_threshold"])
               else "tab:gray" for a, s in zip(aurocs, sims)]
    ax.scatter(aurocs, sims, c=colors, alpha=0.6, edgecolors="white", linewidths=0.5, s=40)
    ax.axvline(data["auroc_threshold"], color="orange", linestyle="--", alpha=0.5)
    ax.axhline(data["sim_threshold"], color="orange", linestyle="--", alpha=0.5)
    ax.set_xlabel("Best-match AUROC (activation vs attribute)")
    ax.set_ylabel("CLIP text similarity (MLLM label vs attribute name)")
    ax.set_title(f"Joint Concept Quality (delta={data['delta']})")

    n_hq = data["n_high_quality"]
    n_total = data["n_labeled"]
    ax.text(0.95, 0.05, f"High-quality: {n_hq}/{n_total} ({100*n_hq/n_total:.0f}%)",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(figures_dir / "semantic_agreement_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    results_dir = ensure_dir(args.results_dir)
    figures_dir = ensure_dir(args.figures_dir)

    try:
        load_project_env()

        plot_enhanced_pruning_curve(results_dir, figures_dir)
        logging.info("Wrote pruning_curve_full.pdf")

        plot_alignment_histogram(results_dir, figures_dir)
        logging.info("Wrote alignment_histogram.pdf")

        plot_semantic_scatter(results_dir, figures_dir)
        logging.info("Wrote semantic_agreement_scatter.pdf")

        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={"status": "ok", "figures_dir": str(figures_dir)},
        )
        return 0
    except Exception as exc:
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={"status": "error", "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
