from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sae_cbm_eval.constants import (
    CLIP_MODEL_ID,
    DEFAULT_LOGREG_MAX_ITER,
    DEFAULT_PRUNE_FRACTION,
    EXPECTED_SAE_DIM,
    EXPECTED_TEST_IMAGES,
    EXPECTED_TRAIN_IMAGES,
    RANDOM_SEED,
    SAE_REPO_ID,
)
from sae_cbm_eval.pruning import compute_sigma_train, prune_to_k
from sae_cbm_eval.runtime import (
    configure_runtime_logging,
    ensure_dir,
    load_project_env,
    project_path,
    set_reproducibility,
    write_hardware_report,
    write_json,
    write_run_manifest,
)


SCRIPT_NAME = "05_final_test"
DEFAULT_FINAL_TEST_DELTAS = ("0.01", "0.02", "0.05")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected pruning operating points on the held-out test set."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=project_path("features"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_path("results"),
    )
    parser.add_argument(
        "--stage2-results-dir",
        type=Path,
        default=project_path("results"),
        help="Directory containing baseline_summary.json from script 2.",
    )
    parser.add_argument(
        "--stage3-results-dir",
        type=Path,
        default=project_path("results"),
        help="Directory containing k_delta_table.json from script 3.",
    )
    parser.add_argument(
        "--deltas",
        nargs="+",
        type=str,
        default=list(DEFAULT_FINAL_TEST_DELTAS),
    )
    parser.add_argument(
        "--prune-fraction",
        type=float,
        default=DEFAULT_PRUNE_FRACTION,
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_LOGREG_MAX_ITER,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting results/final_test_results.json.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for the final-test script itself.",
    )
    return parser.parse_args()


def check_overwrite(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist: {joined}. Rerun with --overwrite to replace them."
        )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text())


def load_inputs(
    *,
    features_dir: Path,
    stage2_results_dir: Path,
    stage3_results_dir: Path,
) -> tuple[np.memmap, np.ndarray, np.memmap, np.ndarray, dict[str, Any], dict[str, Any]]:
    z_train_path = features_dir / "Z_train.npy"
    y_train_path = features_dir / "y_train.npy"
    z_test_path = features_dir / "Z_test.npy"
    y_test_path = features_dir / "y_test.npy"
    baseline_summary_path = stage2_results_dir / "baseline_summary.json"
    k_delta_table_path = stage3_results_dir / "k_delta_table.json"

    missing = [
        path
        for path in [
            z_train_path,
            y_train_path,
            z_test_path,
            y_test_path,
            baseline_summary_path,
            k_delta_table_path,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 5 inputs: "
            + ", ".join(str(path) for path in missing)
            + ". Run scripts/02_train_baseline.py and scripts/03_run_pruning.py first."
        )

    return (
        np.load(z_train_path, mmap_mode="r"),
        np.load(y_train_path),
        np.load(z_test_path, mmap_mode="r"),
        np.load(y_test_path),
        load_json(baseline_summary_path),
        load_json(k_delta_table_path),
    )


def validate_inputs(
    *,
    Z_train: np.memmap,
    y_train: np.ndarray,
    Z_test: np.memmap,
    y_test: np.ndarray,
    baseline_summary: dict[str, Any],
    k_delta_table: dict[str, Any],
    deltas: list[str],
    prune_fraction: float,
    max_iter: int,
) -> float:
    if Z_train.shape != (EXPECTED_TRAIN_IMAGES, EXPECTED_SAE_DIM):
        raise ValueError(
            f"Expected Z_train shape {(EXPECTED_TRAIN_IMAGES, EXPECTED_SAE_DIM)}, "
            f"observed {tuple(Z_train.shape)}"
        )
    if y_train.shape != (EXPECTED_TRAIN_IMAGES,):
        raise ValueError(
            f"Expected y_train shape {(EXPECTED_TRAIN_IMAGES,)}, observed {tuple(y_train.shape)}"
        )
    if Z_test.shape != (EXPECTED_TEST_IMAGES, EXPECTED_SAE_DIM):
        raise ValueError(
            f"Expected Z_test shape {(EXPECTED_TEST_IMAGES, EXPECTED_SAE_DIM)}, "
            f"observed {tuple(Z_test.shape)}"
        )
    if y_test.shape != (EXPECTED_TEST_IMAGES,):
        raise ValueError(
            f"Expected y_test shape {(EXPECTED_TEST_IMAGES,)}, observed {tuple(y_test.shape)}"
        )
    if baseline_summary.get("best_C") is None:
        raise ValueError("results/baseline_summary.json did not contain best_C.")
    for delta in deltas:
        if delta not in k_delta_table:
            raise ValueError(f"k_delta_table.json did not contain a '{delta}' entry.")
    if not 0.0 < prune_fraction < 1.0:
        raise ValueError("--prune-fraction must lie in (0, 1).")
    if max_iter <= 0:
        raise ValueError("--max-iter must be positive.")
    return float(baseline_summary["best_C"])


def build_progress_logger(*, delta: str, k_target: int, max_iter: int):
    def _callback(step: dict[str, Any]) -> None:
        logging.info(
            "delta=%s target=%s phase=%s round=%s n_features=%s n_iter=%s converged=%s",
            delta,
            k_target,
            step["phase"],
            step["round"],
            step["n_features"],
            step["n_iter"],
            step["converged"],
        )
        if int(step["n_iter"]) >= max_iter:
            logging.warning(
                "delta=%s target=%s phase=%s hit max_iter=%s",
                delta,
                k_target,
                step["phase"],
                max_iter,
            )

    return _callback


def select_train_active_features(Z_train_full: np.ndarray) -> np.ndarray:
    active_mask = np.any(Z_train_full != 0.0, axis=0)
    if not np.any(active_mask):
        raise ValueError("All full-training features are zero; cannot run final test stage.")
    return np.flatnonzero(active_mask)


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    features_dir = ensure_dir(args.features_dir)
    results_dir = ensure_dir(args.results_dir)
    final_test_results_path = results_dir / "final_test_results.json"

    try:
        check_overwrite([final_test_results_path], overwrite=args.overwrite)
        load_project_env()
        set_reproducibility(RANDOM_SEED)
        write_hardware_report(results_dir=results_dir)

        Z_train, y_train, Z_test, y_test, baseline_summary, k_delta_table = load_inputs(
            features_dir=features_dir,
            stage2_results_dir=args.stage2_results_dir,
            stage3_results_dir=args.stage3_results_dir,
        )
        best_C = validate_inputs(
            Z_train=Z_train,
            y_train=y_train,
            Z_test=Z_test,
            y_test=y_test,
            baseline_summary=baseline_summary,
            k_delta_table=k_delta_table,
            deltas=args.deltas,
            prune_fraction=args.prune_fraction,
            max_iter=args.max_iter,
        )

        Z_train_full = np.ascontiguousarray(Z_train, dtype=np.float32)
        y_train_full = np.asarray(y_train, dtype=np.int64)
        y_test_full = np.asarray(y_test, dtype=np.int64)
        active_full_feature_ids = select_train_active_features(Z_train_full)
        logging.info(
            "Reduced Stage 5 full-train design matrix from %s to %s active features",
            EXPECTED_SAE_DIM,
            int(len(active_full_feature_ids)),
        )
        Z_train_reduced = np.ascontiguousarray(
            Z_train_full[:, active_full_feature_ids],
            dtype=np.float32,
        )
        sigma_reduced = compute_sigma_train(Z_train_reduced)

        final_test_results: list[dict[str, Any]] = []
        selected_targets: dict[str, int] = {}

        for delta in args.deltas:
            k_target_raw = k_delta_table.get(delta)
            if k_target_raw is None:
                logging.warning("Skipping delta=%s because k_delta_table entry is null.", delta)
                continue
            k_target = int(k_target_raw)
            selected_targets[delta] = k_target
            logging.info(
                "Running final test evaluation for delta=%s with k_target=%s and C=%s",
                delta,
                k_target,
                best_C,
            )
            active_reduced, clf_final = prune_to_k(
                Z_fit=Z_train_reduced,
                y_fit=y_train_full,
                sigma_fit=sigma_reduced,
                C=best_C,
                k_target=k_target,
                prune_fraction=args.prune_fraction,
                max_iter=args.max_iter,
                random_state=RANDOM_SEED,
                progress_callback=build_progress_logger(
                    delta=delta,
                    k_target=k_target,
                    max_iter=args.max_iter,
                ),
            )
            active_full = active_full_feature_ids[active_reduced]
            test_acc = float(
                clf_final.score(
                    np.ascontiguousarray(Z_test[:, active_full], dtype=np.float32),
                    y_test_full,
                )
            )
            result = {
                "delta": delta,
                "k_target": k_target,
                "n_features_final": int(len(active_full)),
                "test_acc": test_acc,
                "active_feature_indices": active_full.tolist(),
            }
            final_test_results.append(result)
            logging.info(
                "Completed delta=%s: n_features_final=%s test_acc=%.4f",
                delta,
                len(active_full),
                test_acc,
            )

        write_json(final_test_results_path, final_test_results)
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={
                "status": "ok",
                "features_dir": str(features_dir),
                "results_dir": str(results_dir),
                "stage2_results_dir": str(args.stage2_results_dir),
                "stage3_results_dir": str(args.stage3_results_dir),
                "deltas": list(args.deltas),
                "selected_targets": selected_targets,
                "best_C": best_C,
                "max_iter": args.max_iter,
                "prune_fraction": args.prune_fraction,
                "n_evaluated_points": int(len(final_test_results)),
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
                "features_dir": str(features_dir),
                "results_dir": str(results_dir),
                "stage2_results_dir": str(args.stage2_results_dir),
                "stage3_results_dir": str(args.stage3_results_dir),
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
