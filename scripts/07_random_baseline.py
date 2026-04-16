from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sae_cbm_eval.classification import build_multinomial_logreg, summarize_iterations
from sae_cbm_eval.constants import (
    CLIP_MODEL_ID,
    DEFAULT_LOGREG_MAX_ITER,
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


SCRIPT_NAME = "07_random_baseline"
DEFAULT_N_RANDOM_TRIALS = 50
DEFAULT_FEATURE_COUNTS = [139, 103, 52]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pruned feature subsets against random subsets of matched size."
    )
    parser.add_argument("--features-dir", type=Path, default=project_path("features"))
    parser.add_argument("--results-dir", type=Path, default=project_path("results"))
    parser.add_argument(
        "--stage2-results-dir",
        type=Path,
        default=project_path("results"),
    )
    parser.add_argument(
        "--stage3-results-dir",
        type=Path,
        default=project_path("results"),
    )
    parser.add_argument(
        "--feature-counts",
        nargs="+",
        type=int,
        default=None,
        help="Feature counts to evaluate. Auto-derived from final_test_results.json if omitted.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_RANDOM_TRIALS,
    )
    parser.add_argument("--max-iter", type=int, default=DEFAULT_LOGREG_MAX_ITER)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    features_dir = ensure_dir(args.features_dir)
    results_dir = ensure_dir(args.results_dir)
    output_path = results_dir / "random_baseline.json"

    try:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
        load_project_env()
        set_reproducibility(RANDOM_SEED)
        write_hardware_report(results_dir=results_dir)

        Z_train = np.load(features_dir / "Z_train.npy", mmap_mode="r")
        y_train = np.load(features_dir / "y_train.npy")
        split_indices = load_json(args.stage2_results_dir / "split_indices.json")
        baseline_summary = load_json(args.stage2_results_dir / "baseline_summary.json")
        pruning_results = load_json(args.stage3_results_dir / "pruning_results.json")

        train_idx = np.asarray(split_indices["train_idx"], dtype=np.int64)
        val_idx = np.asarray(split_indices["val_idx"], dtype=np.int64)
        best_C = float(baseline_summary["best_C"])

        Z_tr = np.ascontiguousarray(Z_train[train_idx], dtype=np.float32)
        Z_val = np.ascontiguousarray(Z_train[val_idx], dtype=np.float32)
        y_tr = np.asarray(y_train[train_idx], dtype=np.int64)
        y_val = np.asarray(y_train[val_idx], dtype=np.int64)

        if args.feature_counts is None:
            ftr_path = args.stage3_results_dir / "final_test_results.json"
            if ftr_path.exists():
                ftr_data = load_json(ftr_path)
                args.feature_counts = [
                    len(op["active_feature_indices"]) for op in ftr_data
                ]
                logging.info(
                    "Auto-derived feature counts from final_test_results.json: %s",
                    args.feature_counts,
                )
            else:
                args.feature_counts = DEFAULT_FEATURE_COUNTS
                logging.info(
                    "final_test_results.json not found, using defaults: %s",
                    args.feature_counts,
                )

        active_mask = np.any(Z_tr != 0.0, axis=0)
        active_indices = np.flatnonzero(active_mask)
        logging.info("Active (non-dead) features: %s", len(active_indices))

        pruned_acc_lookup = {}
        for r in pruning_results:
            pruned_acc_lookup[int(r["n_features"])] = float(r["val_acc"])

        rng = np.random.default_rng(RANDOM_SEED)
        all_results: list[dict[str, Any]] = []

        for k in args.feature_counts:
            nearest_k = min(pruned_acc_lookup.keys(), key=lambda x: abs(x - k))
            pruned_acc = pruned_acc_lookup[nearest_k]

            trial_accs = []
            for trial in range(args.n_trials):
                sampled = rng.choice(active_indices, size=min(k, len(active_indices)), replace=False)
                clf = build_multinomial_logreg(
                    C=best_C,
                    max_iter=args.max_iter,
                    random_state=RANDOM_SEED,
                )
                clf.fit(Z_tr[:, sampled], y_tr)
                acc = float(clf.score(Z_val[:, sampled], y_val))
                trial_accs.append(acc)

                if (trial + 1) % 10 == 0:
                    logging.info(
                        "k=%s trial %s/%s: acc=%.4f",
                        k, trial + 1, args.n_trials, acc,
                    )

            trial_accs_arr = np.array(trial_accs)
            result = {
                "k": k,
                "nearest_pruned_k": nearest_k,
                "pruned_val_acc": pruned_acc,
                "random_mean_acc": float(trial_accs_arr.mean()),
                "random_std_acc": float(trial_accs_arr.std()),
                "random_median_acc": float(np.median(trial_accs_arr)),
                "random_min_acc": float(trial_accs_arr.min()),
                "random_max_acc": float(trial_accs_arr.max()),
                "n_trials": args.n_trials,
                "trial_accs": [float(a) for a in trial_accs],
                "pruned_minus_random_mean": float(pruned_acc - trial_accs_arr.mean()),
            }
            all_results.append(result)
            logging.info(
                "k=%s: pruned=%.4f random_mean=%.4f (+/- %.4f) gap=%.4f",
                k, pruned_acc, result["random_mean_acc"],
                result["random_std_acc"], result["pruned_minus_random_mean"],
            )

        write_json(output_path, all_results)
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={
                "status": "ok",
                "feature_counts": args.feature_counts,
                "n_trials": args.n_trials,
                "n_active_features": int(len(active_indices)),
            },
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
