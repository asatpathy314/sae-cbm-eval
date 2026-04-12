from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sae_cbm_eval.alignment import (
    best_matched_attributes,
    compute_feature_attribute_auroc,
    permutation_baseline,
)
from sae_cbm_eval.attributes import build_attribute_matrix
from sae_cbm_eval.constants import (
    CLIP_MODEL_ID,
    CUB_DIR_NAME,
    RANDOM_SEED,
    SAE_REPO_ID,
)
from sae_cbm_eval.cub import parse_cub_metadata, split_cub_metadata, validate_cub_metadata
from sae_cbm_eval.runtime import (
    configure_runtime_logging,
    ensure_dir,
    load_project_env,
    project_path,
    set_reproducibility,
    write_json,
    write_run_manifest,
)


SCRIPT_NAME = "11_attribute_alignment"
DEFAULT_N_PERMUTATIONS = 100
DEFAULT_AUROC_THRESHOLD = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute feature-attribute AUROC alignment for retained features."
    )
    parser.add_argument("--dataset-root", type=Path, default=project_path("data", CUB_DIR_NAME))
    parser.add_argument("--features-dir", type=Path, default=project_path("features"))
    parser.add_argument("--results-dir", type=Path, default=project_path("results"))
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    parser.add_argument("--auroc-threshold", type=float, default=DEFAULT_AUROC_THRESHOLD)
    parser.add_argument(
        "--operating-points",
        nargs="+",
        default=["0.01", "0.02", "0.05"],
    )
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
    results_dir = ensure_dir(args.results_dir)
    output_path = results_dir / "attribute_alignment.json"

    try:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
        load_project_env()
        set_reproducibility(RANDOM_SEED)

        dataset_root = Path(args.dataset_root)
        metadata = parse_cub_metadata(dataset_root)
        validate_cub_metadata(metadata)
        train_df, _, _ = split_cub_metadata(metadata)

        train_image_ids = train_df["image_id"].values
        A_train, attr_names = build_attribute_matrix(dataset_root, train_image_ids)
        logging.info("Attribute matrix shape: %s, %s attributes", A_train.shape, len(attr_names))

        Z_train = np.load(args.features_dir / "Z_train.npy", mmap_mode="r")
        final_test_results = load_json(args.results_dir / "final_test_results.json")

        rng = np.random.default_rng(RANDOM_SEED)
        all_alignment_results: list[dict[str, Any]] = []

        for op_result in final_test_results:
            delta = op_result["delta"]
            if delta not in args.operating_points:
                continue

            feature_indices = np.array(op_result["active_feature_indices"], dtype=np.int64)
            logging.info(
                "Computing alignment for delta=%s (%s features)", delta, len(feature_indices),
            )

            auroc_matrix = compute_feature_attribute_auroc(Z_train, A_train, feature_indices)
            logging.info("AUROC matrix shape: %s", auroc_matrix.shape)

            matches = best_matched_attributes(auroc_matrix, attr_names)

            best_scores = [m["best_auroc"] for m in matches if not np.isnan(m["best_auroc"])]
            best_scores_arr = np.array(best_scores)

            frac_above = float((best_scores_arr >= args.auroc_threshold).mean()) if len(best_scores_arr) > 0 else 0.0

            logging.info("Running %s permutation trials for delta=%s", args.n_permutations, delta)
            perm_aurocs = permutation_baseline(
                Z_train, A_train, feature_indices, args.n_permutations, rng,
            )
            perm_best_mean = float(np.nanmean(perm_aurocs))
            perm_best_std = float(np.nanstd(perm_aurocs))

            attr_coverage: dict[str, int] = {}
            for m in matches:
                name = m["best_attr_name"]
                if name != "none":
                    attr_coverage[name] = attr_coverage.get(name, 0) + 1
            top_attrs = sorted(attr_coverage.items(), key=lambda x: x[1], reverse=True)[:20]

            alignment_result = {
                "delta": delta,
                "n_features": len(feature_indices),
                "mean_best_auroc": float(best_scores_arr.mean()) if len(best_scores_arr) > 0 else None,
                "median_best_auroc": float(np.median(best_scores_arr)) if len(best_scores_arr) > 0 else None,
                "std_best_auroc": float(best_scores_arr.std()) if len(best_scores_arr) > 0 else None,
                "frac_above_threshold": frac_above,
                "auroc_threshold": args.auroc_threshold,
                "perm_baseline_mean": perm_best_mean,
                "perm_baseline_std": perm_best_std,
                "best_matches": matches,
                "top_covered_attributes": [{"name": n, "count": c} for n, c in top_attrs],
                "auroc_matrix": auroc_matrix.tolist(),
            }
            all_alignment_results.append(alignment_result)

            logging.info(
                "delta=%s: mean_best_auroc=%.4f, frac_above_%.2f=%.3f, perm_mean=%.4f",
                delta,
                alignment_result["mean_best_auroc"] or 0,
                args.auroc_threshold,
                frac_above,
                perm_best_mean,
            )

        write_json(output_path, all_alignment_results)

        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={
                "status": "ok",
                "operating_points": args.operating_points,
                "n_permutations": args.n_permutations,
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
