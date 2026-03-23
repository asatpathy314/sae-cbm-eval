from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sae_cbm_eval.classification import summarize_iterations
from sae_cbm_eval.constants import (
    CLIP_MODEL_ID,
    DEFAULT_L1_C_CANDIDATES,
    DEFAULT_L1_MAX_ITER,
    DEFAULT_L1_TOL,
    EXPECTED_SAE_DIM,
    EXPECTED_TRAIN_IMAGES,
    RANDOM_SEED,
    SAE_REPO_ID,
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


SCRIPT_NAME = "04_l1_baseline"
_LOGREG_SUPPORTS_MULTI_CLASS = "multi_class" in inspect.signature(LogisticRegression).parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage 4 standardized L1 logistic regression baseline."
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
        help="Directory containing split_indices.json from script 2.",
    )
    parser.add_argument(
        "--c-candidates",
        nargs="+",
        type=float,
        default=list(DEFAULT_L1_C_CANDIDATES),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_L1_MAX_ITER,
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_L1_TOL,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting results/l1_baseline.json.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for the L1 baseline script itself.",
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


def load_stage2_inputs(
    *,
    features_dir: Path,
    stage2_results_dir: Path,
) -> tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray]:
    z_train_path = features_dir / "Z_train.npy"
    y_train_path = features_dir / "y_train.npy"
    split_indices_path = stage2_results_dir / "split_indices.json"

    missing = [
        path for path in [z_train_path, y_train_path, split_indices_path] if not path.exists()
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
    return (
        Z_train,
        y_train,
        np.asarray(split_indices["train_idx"], dtype=np.int64),
        np.asarray(split_indices["val_idx"], dtype=np.int64),
    )


def validate_inputs(
    *,
    Z_train: np.memmap,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    c_candidates: list[float],
    max_iter: int,
    tol: float,
) -> None:
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
    if not c_candidates or any(c <= 0 for c in c_candidates):
        raise ValueError("--c-candidates must contain at least one positive value.")
    if max_iter <= 0:
        raise ValueError("--max-iter must be positive.")
    if tol <= 0:
        raise ValueError("--tol must be positive.")


def build_l1_model(*, C: float, max_iter: int, tol: float) -> LogisticRegression:
    logistic_kwargs: dict[str, Any] = {
        "penalty": "l1",
        "C": C,
        "solver": "saga",
        "max_iter": max_iter,
        "tol": tol,
        "random_state": RANDOM_SEED,
        "warm_start": True,
    }
    if _LOGREG_SUPPORTS_MULTI_CLASS:
        logistic_kwargs["multi_class"] = "multinomial"

    return LogisticRegression(**logistic_kwargs)


def select_train_active_features(
    Z_tr: np.ndarray,
    Z_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Dead train-split features are identically zero after scaling as well, so
    # dropping them preserves the exact optimization problem while shrinking the
    # dense L1 solve dramatically.
    active_mask = np.any(Z_tr != 0.0, axis=0)
    if not np.any(active_mask):
        raise ValueError("All training features are zero; cannot fit Stage 4 baseline.")
    return Z_tr[:, active_mask], Z_val[:, active_mask], active_mask


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    features_dir = ensure_dir(args.features_dir)
    results_dir = ensure_dir(args.results_dir)
    l1_results_path = results_dir / "l1_baseline.json"

    try:
        check_overwrite([l1_results_path], overwrite=args.overwrite)
        load_project_env()
        set_reproducibility(RANDOM_SEED)
        write_hardware_report(results_dir=results_dir)

        Z_train, y_train, train_idx, val_idx = load_stage2_inputs(
            features_dir=features_dir,
            stage2_results_dir=args.stage2_results_dir,
        )
        validate_inputs(
            Z_train=Z_train,
            y_train=y_train,
            train_idx=train_idx,
            val_idx=val_idx,
            c_candidates=args.c_candidates,
            max_iter=args.max_iter,
            tol=args.tol,
        )

        Z_tr = np.ascontiguousarray(Z_train[train_idx], dtype=np.float32)
        Z_val = np.ascontiguousarray(Z_train[val_idx], dtype=np.float32)
        y_tr = np.asarray(y_train[train_idx], dtype=np.int64)
        y_val = np.asarray(y_train[val_idx], dtype=np.int64)

        Z_tr, Z_val, active_mask = select_train_active_features(Z_tr, Z_val)
        logging.info(
            "Reduced Stage 4 design matrix from %s to %s train-active features",
            EXPECTED_SAE_DIM,
            int(active_mask.sum()),
        )

        # For logistic regression with a free intercept, feature centering can
        # be absorbed into the intercept term. Scaling by variance only is thus
        # equivalent to the centered parameterization while preserving sparsity.
        Z_tr_sparse = sparse.csr_matrix(Z_tr)
        Z_val_sparse = sparse.csr_matrix(Z_val)

        # Scale once and reuse the same standardized matrices across the C path.
        scaler = StandardScaler(copy=True, with_mean=False)
        Z_tr_scaled = scaler.fit_transform(Z_tr_sparse)
        Z_val_scaled = scaler.transform(Z_val_sparse)

        warnings.filterwarnings(
            "ignore",
            message=r".*'penalty' was deprecated.*",
            category=FutureWarning,
            module=r"sklearn\.linear_model\._logistic",
        )

        l1_results: list[dict[str, Any]] = []
        clf_l1 = build_l1_model(
            C=float(args.c_candidates[0]),
            max_iter=args.max_iter,
            tol=args.tol,
        )
        for C_l1 in args.c_candidates:
            logging.info("Fitting L1 baseline with C=%s", C_l1)
            clf_l1.set_params(C=float(C_l1))
            fit_started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                clf_l1.fit(Z_tr_scaled, y_tr)
            fit_seconds = float(time.perf_counter() - fit_started)
            val_acc = float(clf_l1.score(Z_val_scaled, y_val))

            W_l1 = clf_l1.coef_
            nonzero = int((np.abs(W_l1).sum(axis=0) > 1e-10).sum())
            convergence_warnings = sum(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )
            n_iter = summarize_iterations(clf_l1)

            result = {
                "C": float(C_l1),
                "val_acc": val_acc,
                "nonzero_features": nonzero,
                "convergence_warnings": int(convergence_warnings),
                "n_iter": n_iter,
                "converged": bool(convergence_warnings == 0 and n_iter < args.max_iter),
                "fit_seconds": fit_seconds,
            }
            l1_results.append(result)
            logging.info(
                "L1 C=%s: val_acc=%.4f nonzero=%s n_iter=%s warnings=%s fit_seconds=%.1f",
                C_l1,
                val_acc,
                nonzero,
                n_iter,
                convergence_warnings,
                fit_seconds,
            )

        write_json(l1_results_path, l1_results)
        best_result = max(l1_results, key=lambda r: r["val_acc"])

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
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "train_active_features": int(active_mask.sum()),
                "c_candidates": [float(c) for c in args.c_candidates],
                "max_iter": args.max_iter,
                "tol": args.tol,
                "best_C": float(best_result["C"]),
                "best_val_acc": float(best_result["val_acc"]),
                "best_nonzero_features": int(best_result["nonzero_features"]),
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
