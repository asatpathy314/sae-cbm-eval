from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sae_cbm_eval.constants import (
    CLIP_MODEL_ID,
    DEFAULT_K_DELTA_DELTAS,
    DEFAULT_LOGREG_MAX_ITER,
    DEFAULT_PRUNE_FRACTION,
    DEFAULT_PRUNING_K_MIN,
    DEFAULT_PRUNING_MAX_ROUNDS,
    DEFAULT_SENSITIVITY_CV_FOLDS,
    DEFAULT_SENSITIVITY_C_MULTIPLIERS,
    DEFAULT_SENSITIVITY_TARGETS,
    EXPECTED_SAE_DIM,
    EXPECTED_TRAIN_IMAGES,
    RANDOM_SEED,
    SAE_REPO_ID,
)
from sae_cbm_eval.pruning import (
    build_pruning_curve,
    compute_k_delta_table,
    compute_sigma_train,
    iterative_pruning,
    run_sensitivity_check,
    serialize_pruning_results,
)
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


SCRIPT_NAME = "03_run_pruning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run iterative pruning on the Stage 2 train/validation split."
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
        help="Directory containing split_indices.json and baseline_summary.json from script 2.",
    )
    parser.add_argument(
        "--prune-fraction",
        type=float,
        default=DEFAULT_PRUNE_FRACTION,
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=DEFAULT_PRUNING_K_MIN,
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_LOGREG_MAX_ITER,
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_PRUNING_MAX_ROUNDS,
        help="Hard cap for smoke tests; default is the full schedule.",
    )
    parser.add_argument(
        "--k-deltas",
        nargs="+",
        type=float,
        default=list(DEFAULT_K_DELTA_DELTAS),
    )
    parser.add_argument(
        "--skip-sensitivity-check",
        action="store_true",
        help="Skip the lightweight fixed-C sensitivity pass.",
    )
    parser.add_argument(
        "--sensitivity-targets",
        nargs="+",
        type=int,
        default=list(DEFAULT_SENSITIVITY_TARGETS),
    )
    parser.add_argument(
        "--sensitivity-c-multipliers",
        nargs="+",
        type=float,
        default=list(DEFAULT_SENSITIVITY_C_MULTIPLIERS),
    )
    parser.add_argument(
        "--sensitivity-cv-folds",
        type=int,
        default=DEFAULT_SENSITIVITY_CV_FOLDS,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Used only for the sensitivity CV pass.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing pruning artifacts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for the pruning script itself.",
    )
    return parser.parse_args()


def check_overwrite(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist: {joined}. Rerun with --overwrite to replace them."
        )


def load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        return {}
    return json.loads(path.read_text())


def load_stage2_inputs(
    *,
    features_dir: Path,
    stage2_results_dir: Path,
) -> tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    z_train_path = features_dir / "Z_train.npy"
    y_train_path = features_dir / "y_train.npy"
    split_indices_path = stage2_results_dir / "split_indices.json"
    baseline_summary_path = stage2_results_dir / "baseline_summary.json"

    missing = [
        path
        for path in [z_train_path, y_train_path, split_indices_path, baseline_summary_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 2 inputs: "
            + ", ".join(str(path) for path in missing)
            + ". Run scripts/02_train_baseline.py first."
        )

    Z_train = np.load(z_train_path, mmap_mode="r")
    y_train = np.load(y_train_path)
    split_indices = load_json(split_indices_path)
    baseline_summary = load_json(baseline_summary_path)
    return (
        Z_train,
        y_train,
        np.asarray(split_indices["train_idx"], dtype=np.int64),
        np.asarray(split_indices["val_idx"], dtype=np.int64),
        baseline_summary,
    )


def validate_inputs(
    *,
    Z_train: np.memmap,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    baseline_summary: dict[str, Any],
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
    if len(train_idx) + len(val_idx) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(
            "Train/validation split indices do not cover the expected 5,994 training rows."
        )

    best_C = baseline_summary.get("best_C")
    if best_C is None:
        raise ValueError("results/baseline_summary.json did not contain best_C.")
    return float(best_C)


def build_output_paths(*, features_dir: Path, results_dir: Path) -> dict[str, Path]:
    return {
        "sigma_train": features_dir / "sigma_train.npy",
        "pruning_curve": results_dir / "pruning_curve.npy",
        "pruning_results": results_dir / "pruning_results.json",
        "k_delta_table": results_dir / "k_delta_table.json",
        "pruning_sensitivity": results_dir / "pruning_sensitivity.json",
    }


def build_progress_logger(max_iter: int):
    def _callback(result: dict[str, Any]) -> None:
        logging.info(
            "Round %s: n_features=%s val_acc=%.4f n_iter=%s converged=%s",
            result["round"],
            result["n_features"],
            result["val_acc"],
            result["n_iter"],
            result["converged"],
        )
        if int(result["n_iter"]) >= max_iter:
            logging.warning(
                "Round %s hit max_iter=%s; later runs may need a larger limit.",
                result["round"],
                max_iter,
            )

    return _callback


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    features_dir = ensure_dir(args.features_dir)
    results_dir = ensure_dir(args.results_dir)
    output_paths = build_output_paths(features_dir=features_dir, results_dir=results_dir)

    tracked_outputs = [
        output_paths["sigma_train"],
        output_paths["pruning_curve"],
        output_paths["pruning_results"],
        output_paths["k_delta_table"],
    ]
    if not args.skip_sensitivity_check:
        tracked_outputs.append(output_paths["pruning_sensitivity"])

    try:
        if not 0.0 < args.prune_fraction < 1.0:
            raise ValueError("--prune-fraction must be strictly between 0 and 1.")
        if args.k_min <= 0:
            raise ValueError("--k-min must be positive.")
        if args.max_iter <= 0:
            raise ValueError("--max-iter must be positive.")
        if args.max_rounds <= 0:
            raise ValueError("--max-rounds must be positive.")
        if args.sensitivity_cv_folds < 2:
            raise ValueError("--sensitivity-cv-folds must be at least 2.")
        if args.n_jobs <= 0:
            raise ValueError("--n-jobs must be positive.")
        if any(delta <= 0 for delta in args.k_deltas):
            raise ValueError("--k-deltas must all be positive.")
        if any(target <= 0 for target in args.sensitivity_targets):
            raise ValueError("--sensitivity-targets must all be positive.")
        if any(multiplier <= 0 for multiplier in args.sensitivity_c_multipliers):
            raise ValueError("--sensitivity-c-multipliers must all be positive.")

        check_overwrite(tracked_outputs, overwrite=args.overwrite)
        load_project_env()
        set_reproducibility(RANDOM_SEED)
        write_hardware_report(results_dir=results_dir)

        Z_train, y_train, train_idx, val_idx, baseline_summary = load_stage2_inputs(
            features_dir=features_dir,
            stage2_results_dir=args.stage2_results_dir,
        )
        best_C = validate_inputs(
            Z_train=Z_train,
            y_train=y_train,
            train_idx=train_idx,
            val_idx=val_idx,
            baseline_summary=baseline_summary,
        )

        Z_tr = np.ascontiguousarray(Z_train[train_idx], dtype=np.float32)
        Z_val = np.ascontiguousarray(Z_train[val_idx], dtype=np.float32)
        y_tr = np.asarray(y_train[train_idx], dtype=np.int64)
        y_val = np.asarray(y_train[val_idx], dtype=np.int64)

        sigma_tr = compute_sigma_train(Z_tr)
        np.save(output_paths["sigma_train"], sigma_tr)

        logging.info(
            "Running pruning with best_C=%s, prune_fraction=%s, k_min=%s",
            best_C,
            args.prune_fraction,
            args.k_min,
        )
        results = iterative_pruning(
            Z_tr=Z_tr,
            y_tr=y_tr,
            Z_val=Z_val,
            y_val=y_val,
            sigma_tr=sigma_tr,
            C=best_C,
            prune_fraction=args.prune_fraction,
            k_min=args.k_min,
            max_iter=args.max_iter,
            max_rounds=args.max_rounds,
            random_state=RANDOM_SEED,
            progress_callback=build_progress_logger(args.max_iter),
        )

        curve = build_pruning_curve(results)
        np.save(output_paths["pruning_curve"], curve)
        write_json(output_paths["pruning_results"], serialize_pruning_results(results))

        k_delta_table = compute_k_delta_table(results, args.k_deltas)
        write_json(output_paths["k_delta_table"], k_delta_table)

        sensitivity_results: list[dict[str, Any]] | None = None
        if not args.skip_sensitivity_check:
            logging.info(
                "Running fixed-C sensitivity check at %s targets",
                len(args.sensitivity_targets),
            )
            sensitivity_results = run_sensitivity_check(
                Z_tr=Z_tr,
                y_tr=y_tr,
                results=results,
                best_C=best_C,
                targets=args.sensitivity_targets,
                c_multipliers=args.sensitivity_c_multipliers,
                cv_folds=args.sensitivity_cv_folds,
                max_iter=args.max_iter,
                random_state=RANDOM_SEED,
                n_jobs=args.n_jobs,
            )
            write_json(output_paths["pruning_sensitivity"], sensitivity_results)

        completed_schedule = bool(results and results[-1]["n_features"] <= args.k_min)
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
                "best_C": best_C,
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "n_rounds": int(len(results)),
                "completed_schedule": completed_schedule,
                "final_n_features": int(results[-1]["n_features"]),
                "initial_val_acc": float(results[0]["val_acc"]),
                "best_pruned_val_acc": float(max(r["val_acc"] for r in results)),
                "prune_fraction": args.prune_fraction,
                "k_min": args.k_min,
                "max_iter": args.max_iter,
                "max_rounds": args.max_rounds,
                "ran_sensitivity_check": not args.skip_sensitivity_check,
                "k_delta_table": k_delta_table,
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
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
