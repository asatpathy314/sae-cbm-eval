from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sae_cbm_eval.classification import (
    build_multinomial_logreg,
    cross_validate_regularization,
    split_train_val_indices,
    summarize_iterations,
)
from sae_cbm_eval.constants import (
    CLIP_MODEL_ID,
    DEFAULT_CV_FOLDS,
    DEFAULT_C_CANDIDATES,
    DEFAULT_LOGREG_MAX_ITER,
    DEFAULT_TRAIN_VAL_TEST_SIZE,
    EXPECTED_NUM_CLASSES,
    EXPECTED_SAE_DIM,
    EXPECTED_TRAIN_IMAGES,
    HEURISTIC_MIN_BASELINE_VAL_ACC,
    RANDOM_SEED,
    SAE_REPO_ID,
    TOKEN_POLICY_CLS_ONLY,
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


SCRIPT_NAME = "02_train_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Stage 2 multinomial logistic regression baseline."
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
        "--test-size",
        type=float,
        default=DEFAULT_TRAIN_VAL_TEST_SIZE,
        help="Fraction of Z_train reserved for validation.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
    )
    parser.add_argument(
        "--c-candidates",
        nargs="+",
        type=float,
        default=list(DEFAULT_C_CANDIDATES),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_LOGREG_MAX_ITER,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Passed to sklearn cross_validate; keep at 1 unless RAM headroom is clear.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing Stage 2 result files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for the baseline script itself.",
    )
    return parser.parse_args()


def check_overwrite(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist: {joined}. Rerun with --overwrite to replace them."
        )


def load_required_feature_artifacts(features_dir: Path) -> tuple[np.memmap, np.ndarray, dict[str, Any]]:
    z_train_path = features_dir / "Z_train.npy"
    y_train_path = features_dir / "y_train.npy"
    extraction_meta_path = features_dir / "extraction_meta.json"

    missing = [path for path in [z_train_path, y_train_path, extraction_meta_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 1 feature artifacts: "
            + ", ".join(str(path) for path in missing)
            + ". Run scripts/01_extract_features.py first."
        )

    Z_train = np.load(z_train_path, mmap_mode="r")
    y_train = np.load(y_train_path)
    extraction_meta = json.loads(extraction_meta_path.read_text())
    return Z_train, y_train, extraction_meta


def validate_stage1_outputs(
    *,
    Z_train: np.memmap,
    y_train: np.ndarray,
    extraction_meta: dict[str, Any],
) -> int:
    if Z_train.shape != (EXPECTED_TRAIN_IMAGES, EXPECTED_SAE_DIM):
        raise ValueError(
            f"Expected Z_train shape {(EXPECTED_TRAIN_IMAGES, EXPECTED_SAE_DIM)}, "
            f"observed {tuple(Z_train.shape)}"
        )
    if y_train.shape != (EXPECTED_TRAIN_IMAGES,):
        raise ValueError(
            f"Expected y_train shape {(EXPECTED_TRAIN_IMAGES,)}, observed {tuple(y_train.shape)}"
        )

    if extraction_meta.get("token_policy") != TOKEN_POLICY_CLS_ONLY:
        raise ValueError(
            "Stage 1 extraction metadata did not record the required CLS-only token policy."
        )

    n_classes = int(np.unique(y_train).size)
    if n_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CLASSES} classes in y_train, observed {n_classes}"
        )

    dead_features = extraction_meta.get("post_cache_sanity", {}).get("dead_features")
    if dead_features is None:
        dead_features = int((np.asarray(Z_train.max(axis=0)) == 0).sum())
    return int(dead_features)


def build_split_payload(train_idx: np.ndarray, val_idx: np.ndarray) -> dict[str, Any]:
    return {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
    }


def build_baseline_summary(
    *,
    best_C: float,
    val_accuracy: float,
    n_iter: int,
    max_iter: int,
    dead_features: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    Z_train_shape: tuple[int, int],
    cv_results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "best_C": best_C,
        "val_accuracy": val_accuracy,
        "n_iter": n_iter,
        "max_iter": max_iter,
        "converged": bool(n_iter < max_iter),
        "dead_features": dead_features,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "n_features": int(Z_train_shape[1]),
        "cv_candidates": [float(k) for k in cv_results.keys()],
        "cv_best_mean_acc": float(cv_results[str(best_C)]["mean_acc"]),
    }


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    features_dir = ensure_dir(args.features_dir)
    results_dir = ensure_dir(args.results_dir)

    split_indices_path = results_dir / "split_indices.json"
    cv_results_path = results_dir / "cv_results.json"
    baseline_summary_path = results_dir / "baseline_summary.json"
    output_paths = [split_indices_path, cv_results_path, baseline_summary_path]

    try:
        if not 0.0 < args.test_size < 1.0:
            raise ValueError("--test-size must be strictly between 0 and 1.")
        if args.cv_folds < 2:
            raise ValueError("--cv-folds must be at least 2.")
        if args.max_iter <= 0:
            raise ValueError("--max-iter must be positive.")
        if args.n_jobs <= 0:
            raise ValueError("--n-jobs must be positive.")
        if not args.c_candidates:
            raise ValueError("--c-candidates must contain at least one positive value.")
        if any(c <= 0 for c in args.c_candidates):
            raise ValueError("--c-candidates must all be positive.")

        check_overwrite(output_paths, overwrite=args.overwrite)
        load_project_env()
        set_reproducibility(RANDOM_SEED)
        write_hardware_report(results_dir=results_dir)

        Z_train, y_train, extraction_meta = load_required_feature_artifacts(features_dir)
        dead_features = validate_stage1_outputs(
            Z_train=Z_train,
            y_train=y_train,
            extraction_meta=extraction_meta,
        )

        train_idx, val_idx = split_train_val_indices(
            n_samples=len(y_train),
            y=y_train,
            test_size=args.test_size,
            random_state=RANDOM_SEED,
        )
        write_json(split_indices_path, build_split_payload(train_idx, val_idx))

        Z_tr_full = np.ascontiguousarray(Z_train[train_idx], dtype=np.float32)
        Z_val_full = np.ascontiguousarray(Z_train[val_idx], dtype=np.float32)
        y_tr = np.asarray(y_train[train_idx], dtype=np.int64)
        y_val = np.asarray(y_train[val_idx], dtype=np.int64)

        # Drop dead (all-zero) features to speed up CV and fitting
        live_mask = np.any(Z_tr_full != 0.0, axis=0)
        live_indices = np.flatnonzero(live_mask)
        Z_tr = Z_tr_full[:, live_indices]
        Z_val = Z_val_full[:, live_indices]
        logging.info(
            "Dropped %s dead features, fitting on %s live features",
            int((~live_mask).sum()), len(live_indices),
        )

        logging.info(
            "Running %s-fold CV on %s train rows with %s candidate C values",
            args.cv_folds,
            len(train_idx),
            len(args.c_candidates),
        )
        cv_results, best_C = cross_validate_regularization(
            Z=Z_tr,
            y=y_tr,
            c_candidates=args.c_candidates,
            n_splits=args.cv_folds,
            max_iter=args.max_iter,
            random_state=RANDOM_SEED,
            n_jobs=args.n_jobs,
        )
        write_json(cv_results_path, cv_results)

        unconverged = [c for c, payload in cv_results.items() if not payload["converged"]]
        if unconverged:
            logging.warning(
                "Some CV folds hit max_iter=%s for C values: %s",
                args.max_iter,
                ", ".join(unconverged),
            )

        clf_full = build_multinomial_logreg(
            C=best_C,
            max_iter=args.max_iter,
            random_state=RANDOM_SEED,
        )
        logging.info("Fitting baseline classifier with best_C=%s", best_C)
        clf_full.fit(Z_tr, y_tr)
        val_accuracy = float(clf_full.score(Z_val, y_val))
        n_iter = summarize_iterations(clf_full)

        if n_iter >= args.max_iter:
            logging.warning(
                "Final baseline fit hit max_iter=%s; consider rerunning with a larger value.",
                args.max_iter,
            )
        if val_accuracy < HEURISTIC_MIN_BASELINE_VAL_ACC:
            logging.warning(
                "Validation accuracy %.4f is below the spec heuristic floor %.2f; "
                "this may indicate a pipeline issue.",
                val_accuracy,
                HEURISTIC_MIN_BASELINE_VAL_ACC,
            )

        baseline_summary = build_baseline_summary(
            best_C=best_C,
            val_accuracy=val_accuracy,
            n_iter=n_iter,
            max_iter=args.max_iter,
            dead_features=dead_features,
            train_idx=train_idx,
            val_idx=val_idx,
            Z_train_shape=tuple(Z_train.shape),
            cv_results=cv_results,
        )
        write_json(baseline_summary_path, baseline_summary)

        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={
                "status": "ok",
                "features_dir": str(features_dir),
                "results_dir": str(results_dir),
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "best_C": best_C,
                "val_accuracy": val_accuracy,
                "n_iter": n_iter,
                "max_iter": args.max_iter,
                "cv_folds": args.cv_folds,
                "n_jobs": args.n_jobs,
                "dead_features": dead_features,
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
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
