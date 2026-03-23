from __future__ import annotations

import argparse
import json
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


SCRIPT_NAME = "06_plot_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the pruning curve with the L1 baseline overlay."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_path("results"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=project_path("figures"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting figures/pruning_curve.pdf.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for the plotting script itself.",
    )
    return parser.parse_args()


def check_overwrite(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist: {joined}. Rerun with --overwrite to replace them."
        )


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    results_dir = ensure_dir(args.results_dir)
    figures_dir = ensure_dir(args.figures_dir)
    figure_path = figures_dir / "pruning_curve.pdf"
    summary_path = figures_dir / "plot_summary.json"

    try:
        check_overwrite([figure_path, summary_path], overwrite=args.overwrite)
        load_project_env()

        curve_path = results_dir / "pruning_curve.npy"
        l1_path = results_dir / "l1_baseline.json"
        if not curve_path.exists():
            raise FileNotFoundError(
                f"Required pruning curve not found: {curve_path}. Run script 03 first."
            )
        if not l1_path.exists():
            raise FileNotFoundError(
                f"Required L1 baseline results not found: {l1_path}. Run script 04 first."
            )

        curve = np.load(curve_path)
        l1_results = load_json(l1_path)

        if curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError(
                f"Expected pruning_curve.npy to have shape (n_rounds, 2), observed {curve.shape}"
            )
        if not isinstance(l1_results, list) or not l1_results:
            raise ValueError("results/l1_baseline.json must contain a non-empty list.")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            curve[:, 0],
            curve[:, 1] * 100.0,
            "o-",
            markersize=2,
            linewidth=1.5,
            label="Iterative pruning",
        )

        for result in l1_results:
            ax.plot(
                float(result["nonzero_features"]),
                float(result["val_acc"]) * 100.0,
                "s",
                color="red",
                markersize=7,
                zorder=5,
            )
        ax.plot([], [], "s", color="red", markersize=7, label="L1 baseline")

        baseline_acc_pct = float(curve[0, 1]) * 100.0
        ax.axhline(
            baseline_acc_pct,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Full ({baseline_acc_pct:.1f}%)",
        )

        for delta in [0.01, 0.02, 0.05, 0.10]:
            ax.axhline(
                (float(curve[0, 1]) - delta) * 100.0,
                color="orange",
                linestyle=":",
                alpha=0.3,
            )

        ax.set_xlabel("Number of SAE features")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.legend()
        ax.set_title("Iterative Feature Pruning on CUB-200")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        best_l1 = max(l1_results, key=lambda r: float(r["val_acc"]))
        plot_summary = {
            "curve_rounds": int(curve.shape[0]),
            "full_validation_accuracy": float(curve[0, 1]),
            "best_l1_C": float(best_l1["C"]),
            "best_l1_val_acc": float(best_l1["val_acc"]),
            "figure_path": str(figure_path),
        }
        write_json(summary_path, plot_summary)
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={
                "status": "ok",
                "results_dir": str(results_dir),
                "figures_dir": str(figures_dir),
                "figure_path": str(figure_path),
                "plot_summary_path": str(summary_path),
                "curve_rounds": int(curve.shape[0]),
                "n_l1_points": int(len(l1_results)),
            },
        )
        return 0
    except Exception as exc:
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={
                "status": "error",
                "results_dir": str(results_dir),
                "figures_dir": str(figures_dir),
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
