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
    parser.add_argument("--exemplars-dir", type=Path, default=project_path("exemplars"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text())


def plot_pruning_curve_headline(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Stripped-down pruning curve: SAE only, with test-accuracy markers at operating points."""
    curve = np.load(results_dir / "pruning_curve.npy")
    final_results = load_json(results_dir / "final_test_results.json")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        curve[:, 0], curve[:, 1] * 100.0,
        "o-", markersize=3, linewidth=1.8, color="tab:blue",
        label="Iterative pruning (val)",
    )

    baseline_acc_pct = float(curve[0, 1]) * 100.0
    ax.axhline(baseline_acc_pct, color="gray", linestyle="--", alpha=0.6,
               label=f"Full SAE (n={int(curve[0, 0])}, {baseline_acc_pct:.1f}%)")

    for entry in final_results:
        k = entry["n_features_final"]
        test_acc = entry["test_acc"] * 100.0
        ax.plot(k, test_acc, "D", color="tab:red", markersize=9, zorder=6,
                markeredgecolor="white", markeredgewidth=1)
    ax.plot([], [], "D", color="tab:red", markersize=9,
            label="Test accuracy at δ-operating points")

    table_lines = [" δ       k     test "]
    table_lines.append(" " + "─" * 22)
    for entry in final_results:
        table_lines.append(
            f" {entry['delta']}   {entry['n_features_final']:>4}   {entry['test_acc']*100:5.2f}%"
        )
    ax.text(
        0.02, 0.04, "\n".join(table_lines),
        transform=ax.transAxes, fontsize=9, family="monospace",
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#888", alpha=0.95),
    )

    ax.set_xlabel("Number of features (log scale, pruning direction →)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("SAE Feature Pruning on CUB-200: 10906 → 22 features")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "pruning_curve_headline.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_random_baseline_bar(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Grouped bar chart: pruned vs random subset at matched k."""
    random_path = results_dir / "random_baseline.json"
    if not random_path.exists():
        logging.warning("No random baseline results found, skipping bar chart.")
        return

    results = load_json(random_path)
    results = sorted(results, key=lambda r: -r["k"])

    ks = [r["k"] for r in results]
    pruned = [r["pruned_val_acc"] * 100.0 for r in results]
    random_mean = [r["random_mean_acc"] * 100.0 for r in results]
    random_std = [r["random_std_acc"] * 100.0 for r in results]
    n_trials = results[0]["n_trials"]

    x = np.arange(len(ks))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars_pruned = ax.bar(
        x - width / 2, pruned, width,
        color="tab:blue", edgecolor="white", label="Pruned (selected features)",
    )
    bars_random = ax.bar(
        x + width / 2, random_mean, width, yerr=random_std,
        color="tab:green", edgecolor="white", capsize=4,
        label=f"Random subset (mean ± std, n={n_trials})",
    )

    for bar, val in zip(bars_pruned, pruned):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars_random, random_mean):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 3.0,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#444")

    for i, (p, r) in enumerate(zip(pruned, random_mean)):
        gap = p - r
        ax.annotate(
            f"Δ = +{gap:.1f}pp",
            xy=(i, (p + r) / 2), ha="center", va="center",
            fontsize=9, fontweight="bold", color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#888", alpha=0.9),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Pruning vs Random Subset at Matched Sparsity")
    ax.set_ylim(0, max(pruned) * 1.2)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_dir / "random_baseline_bar.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_exemplar_panels(
    results_dir: Path,
    figures_dir: Path,
    exemplars_dir: Path,
    delta: str = "0.05",
    n_per_panel: int = 3,
) -> None:
    """Two side-by-side panels: clean-match vs semantic-mismatch exemplars at one delta."""
    agreement_path = results_dir / "semantic_agreement.json"
    if not agreement_path.exists():
        logging.warning("No semantic agreement results, skipping exemplar panels.")
        return

    delta_exemplar_dir = exemplars_dir / f"delta_{delta}"
    if not delta_exemplar_dir.exists():
        logging.warning("No exemplar dir for delta=%s, skipping.", delta)
        return

    agreement_results = load_json(agreement_path)
    d_data = next((d for d in agreement_results if d["delta"] == delta), None)
    if d_data is None:
        logging.warning("No agreement data for delta=%s", delta)
        return

    exemplar_files = sorted(delta_exemplar_dir.glob("feature_*.jpg"))
    rank_to_path = {}
    for p in exemplar_files:
        try:
            rank = int(p.stem.split("_")[1])
            rank_to_path[rank] = p
        except (IndexError, ValueError):
            continue

    feats = d_data["feature_agreements"]
    valid = [f for f in feats if f.get("best_auroc") is not None]
    clean = sorted([f for f in valid if f.get("high_quality")],
                   key=lambda f: -f["best_auroc"])[:n_per_panel]
    mismatch = sorted([f for f in valid
                       if f["best_auroc"] >= 0.65 and not f.get("llm_agreement")],
                      key=lambda f: -f["best_auroc"])[:n_per_panel]

    def _render(panel_entries, title, out_name):
        if not panel_entries:
            return
        import matplotlib.image as mpimg
        n = len(panel_entries)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.8))
        if n == 1:
            axes = [axes]
        for ax, entry in zip(axes, panel_entries):
            rank = entry["rank"]
            img_path = rank_to_path.get(rank)
            if img_path is None:
                ax.set_axis_off()
                continue
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            label = entry["label"]
            if len(label) > 42:
                label = label[:42] + "…"
            attr = entry["best_attr"].replace("_", " ").replace("::", ": ")
            if len(attr) > 42:
                attr = attr[:42] + "…"
            caption = (
                f"VLM label: \"{label}\"\n"
                f"Best CUB attr: {attr}\n"
                f"AUROC = {entry['best_auroc']:.3f}"
            )
            ax.set_xlabel(caption, fontsize=8.5, labelpad=6, family="sans-serif")
        fig.suptitle(title, fontsize=11, y=1.0)
        fig.tight_layout()
        fig.savefig(figures_dir / out_name, dpi=200, bbox_inches="tight")
        plt.close(fig)

    _render(
        clean,
        f"Clean matches at δ={delta} (high AUROC ∧ LLM agrees)",
        "exemplars_clean.pdf",
    )
    _render(
        mismatch,
        f"Semantic mismatches at δ={delta} (high AUROC ∧ LLM disagrees)",
        "exemplars_mismatch.pdf",
    )


def plot_sae_vs_clip(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """SAE pruning curve vs raw-CLIP top-k pruning curve with matched-k gap table."""
    curve = np.load(results_dir / "pruning_curve.npy")
    clip_path = results_dir / "clip_baseline.json"
    if not clip_path.exists():
        logging.warning("No CLIP baseline results, skipping SAE-vs-CLIP plot.")
        return
    clip_results = load_json(clip_path)
    clip_curve = clip_results["pruning_curve"]

    sae_full_acc = float(curve[0, 1]) * 100.0
    clip_full_acc = float(clip_results["full_val_acc"]) * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))

    clip_n = [p["n_features"] for p in clip_curve]
    clip_acc = [p["val_acc"] * 100.0 for p in clip_curve]
    ax.plot(clip_n, clip_acc, "d--", markersize=3, linewidth=1.6,
            color="tab:purple", label="Raw CLIP top-k (val)")

    ax.plot(curve[:, 0], curve[:, 1] * 100.0,
            "o-", markersize=3, linewidth=1.8, color="tab:blue",
            label="SAE pruned (val)")

    ax.axhline(clip_full_acc, color="tab:purple", linestyle=":", alpha=0.6,
               label=f"Raw CLIP full (d=768, {clip_full_acc:.1f}%)")
    ax.axhline(sae_full_acc, color="tab:blue", linestyle=":", alpha=0.6,
               label=f"SAE full (n=10906, {sae_full_acc:.1f}%)")

    final_results = load_json(results_dir / "final_test_results.json")
    final_results = sorted(final_results, key=lambda r: -r["n_features_final"])

    table_lines = [" k     SAE    CLIP    Δ   "]
    table_lines.append(" " + "─" * 26)
    for entry in final_results:
        k = entry["n_features_final"]
        sae_val = float(np.interp(k, curve[::-1, 0], curve[::-1, 1])) * 100.0
        clip_pt = min(clip_curve, key=lambda p: abs(p["n_features"] - k))
        clip_val = clip_pt["val_acc"] * 100.0
        gap = clip_val - sae_val
        table_lines.append(f" {k:>4}  {sae_val:5.1f}%  {clip_val:5.1f}%  +{gap:3.1f}pp")

    ax.text(
        0.02, 0.04, "\n".join(table_lines),
        transform=ax.transAxes, fontsize=9, family="monospace",
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#888", alpha=0.95),
    )

    ax.set_xlabel("Number of features (log scale, pruning direction →)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("SAE vs Raw CLIP at Matched Sparsity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "sae_vs_clip.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    """Three-panel histogram of best-match AUROC per operating point."""
    alignment_path = results_dir / "attribute_alignment.json"
    if not alignment_path.exists():
        logging.warning("No alignment results found, skipping histogram.")
        return

    alignment_results = load_json(alignment_path)
    if not alignment_results:
        return

    n = len(alignment_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    bins = np.linspace(0.4, 1.0, 25)
    for ax, data in zip(axes, alignment_results):
        matches = data["best_matches"]
        aurocs = [m["best_auroc"] for m in matches
                  if not (m["best_auroc"] is None or np.isnan(m["best_auroc"]))]

        ax.hist(aurocs, bins=bins, color="steelblue", edgecolor="white",
                alpha=0.85, label="Pruned features")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (0.5)")

        perm_mean = data.get("perm_baseline_mean", 0.5)
        perm_std = data.get("perm_baseline_std", 0.0)
        ax.axvline(perm_mean, color="orange", linestyle="--", alpha=0.8,
                   label=f"Permutation null ({perm_mean:.3f})")
        ax.axvspan(perm_mean - perm_std, perm_mean + perm_std,
                   color="orange", alpha=0.15)

        ax.axvline(data.get("auroc_threshold", 0.65), color="tab:red",
                   linestyle=":", alpha=0.7,
                   label=f"Threshold ({data.get('auroc_threshold', 0.65):.2f})")

        frac_above = data.get("frac_above_threshold", 0.0) * 100.0
        mean_auroc = data.get("mean_best_auroc", 0.0)
        ax.set_title(
            f"δ={data['delta']}, k={data['n_features']}\n"
            f"mean={mean_auroc:.3f} · {frac_above:.1f}% above threshold",
            fontsize=10,
        )
        ax.set_xlabel("Best-match AUROC")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Number of features")
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Feature-Attribute Alignment (test split, min_certainty=3)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "alignment_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_high_quality_bar(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Bar chart of high-quality concept fractions across operating points."""
    agreement_path = results_dir / "semantic_agreement.json"
    if not agreement_path.exists():
        logging.warning("No semantic agreement results found, skipping bar chart.")
        return

    agreement_results = load_json(agreement_path)
    if not agreement_results:
        return

    deltas = []
    fracs = []
    labels = []
    for data in agreement_results:
        n_hq = data["n_high_quality"]
        n_total = data["n_labeled"]
        deltas.append(data["delta"])
        fracs.append(data["frac_high_quality"] * 100.0)
        labels.append(f"{n_hq}/{n_total}")

    fig, ax = plt.subplots(figsize=(max(4, len(deltas) * 1.5), 4))
    bars = ax.bar(range(len(deltas)), fracs, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([f"delta={d}" for d in deltas], fontsize=9)
    ax.set_ylabel("High-quality concepts (%)")
    ax.set_title("Fraction of High-Quality Concepts by Operating Point")
    ax.set_ylim(0, 100)

    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(figures_dir / "high_quality_bar.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    results_dir = ensure_dir(args.results_dir)
    figures_dir = ensure_dir(args.figures_dir)

    try:
        load_project_env()

        output_files = [
            figures_dir / "pruning_curve_headline.pdf",
            figures_dir / "random_baseline_bar.pdf",
            figures_dir / "sae_vs_clip.pdf",
            figures_dir / "pruning_curve_full.pdf",
            figures_dir / "alignment_histogram.pdf",
            figures_dir / "high_quality_bar.pdf",
            figures_dir / "exemplars_clean.pdf",
            figures_dir / "exemplars_mismatch.pdf",
        ]
        existing = [p for p in output_files if p.exists()]
        if existing and not args.overwrite:
            joined = ", ".join(str(p) for p in existing)
            raise FileExistsError(
                f"Output files already exist: {joined}. Use --overwrite."
            )

        plot_pruning_curve_headline(results_dir, figures_dir)
        logging.info("Wrote pruning_curve_headline.pdf")

        plot_random_baseline_bar(results_dir, figures_dir)
        logging.info("Wrote random_baseline_bar.pdf")

        plot_sae_vs_clip(results_dir, figures_dir)
        logging.info("Wrote sae_vs_clip.pdf")

        plot_enhanced_pruning_curve(results_dir, figures_dir)
        logging.info("Wrote pruning_curve_full.pdf")

        plot_alignment_histogram(results_dir, figures_dir)
        logging.info("Wrote alignment_histogram.pdf")

        plot_high_quality_bar(results_dir, figures_dir)
        logging.info("Wrote high_quality_bar.pdf")

        plot_exemplar_panels(results_dir, figures_dir, args.exemplars_dir)
        logging.info("Wrote exemplars_clean.pdf / exemplars_mismatch.pdf")

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
