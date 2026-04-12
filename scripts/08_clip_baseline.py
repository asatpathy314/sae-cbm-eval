from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message=r".*Plotly version 5\.19\.0, which is not compatible with this version of Kaleido.*",
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"kaleido\._sync_server")

from sae_cbm_eval.classification import (
    build_multinomial_logreg,
    cross_validate_regularization,
    summarize_iterations,
)
from sae_cbm_eval.constants import (
    CLIP_MODEL_ID,
    CUB_DIR_NAME,
    DEFAULT_CV_FOLDS,
    DEFAULT_C_CANDIDATES,
    DEFAULT_LOGREG_MAX_ITER,
    DEFAULT_PRUNE_FRACTION,
    DEFAULT_PRUNING_K_MIN,
    EXPECTED_HOOK_NAME,
    EXPECTED_INPUT_DIM,
    RANDOM_SEED,
    SAE_REPO_ID,
)
from sae_cbm_eval.cub import (
    CUBImageDataset,
    ensure_cub_dataset,
    parse_cub_metadata,
    split_cub_metadata,
    validate_cub_metadata,
)
from sae_cbm_eval.extraction import build_preprocess, load_clip_model_for_extraction
from sae_cbm_eval.pruning import compute_feature_importance, compute_sigma_train
from sae_cbm_eval.runtime import (
    configure_runtime_logging,
    ensure_dir,
    load_project_env,
    project_path,
    resolve_device,
    set_reproducibility,
    write_hardware_report,
    write_json,
    write_run_manifest,
)


SCRIPT_NAME = "08_clip_baseline"
CLIP_PRUNING_K_MIN = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract raw CLIP features and run iterative pruning on them."
    )
    parser.add_argument("--dataset-root", type=Path, default=project_path("data", CUB_DIR_NAME))
    parser.add_argument("--clip-model-id", default=CLIP_MODEL_ID)
    parser.add_argument("--features-dir", type=Path, default=project_path("features"))
    parser.add_argument("--results-dir", type=Path, default=project_path("results"))
    parser.add_argument(
        "--stage2-results-dir", type=Path, default=project_path("results"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_LOGREG_MAX_ITER)
    parser.add_argument("--prune-fraction", type=float, default=DEFAULT_PRUNE_FRACTION)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text())


def extract_clip_features(
    model, dataloader, hook_name: str, n_rows: int, d_model: int, device: str,
) -> np.ndarray:
    features = np.zeros((n_rows, d_model), dtype=np.float32)
    with torch.inference_mode():
        for images, row_indices, _class_ids in dataloader:
            images = images.to(device=device, dtype=torch.float32)
            _, cache = model.run_with_cache(images, names_filter=hook_name)
            activations = cache[hook_name]
            h_cls = activations[:, 0, :].detach().cpu().numpy().astype(np.float32)
            features[row_indices.numpy()] = h_cls
            del cache
    return features


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    features_dir = ensure_dir(args.features_dir)
    results_dir = ensure_dir(args.results_dir)
    device = resolve_device(args.device)

    clip_train_path = features_dir / "H_train_clip.npy"
    clip_test_path = features_dir / "H_test_clip.npy"
    output_path = results_dir / "clip_baseline.json"

    try:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
        load_project_env()
        set_reproducibility(RANDOM_SEED)
        write_hardware_report(results_dir=results_dir)

        split_indices = load_json(args.stage2_results_dir / "split_indices.json")
        train_idx = np.asarray(split_indices["train_idx"], dtype=np.int64)
        val_idx = np.asarray(split_indices["val_idx"], dtype=np.int64)

        if not clip_train_path.exists() or args.overwrite:
            logging.info("Extracting raw CLIP features (device=%s)", device)
            dataset_root = ensure_cub_dataset(args.dataset_root, download_if_missing=False)
            metadata = parse_cub_metadata(dataset_root)
            validate_cub_metadata(metadata)
            train_df, test_df, _ = split_cub_metadata(metadata)

            preprocess = build_preprocess(args.clip_model_id)
            model = load_clip_model_for_extraction(
                clip_model_id=args.clip_model_id, device=device,
            )

            train_ds = CUBImageDataset(dataset_root, train_df, preprocess)
            test_ds = CUBImageDataset(dataset_root, test_df, preprocess)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
            )

            logging.info("Extracting %s training images", len(train_df))
            H_train = extract_clip_features(
                model, tqdm(train_loader, desc="clip-train"),
                EXPECTED_HOOK_NAME, len(train_df), EXPECTED_INPUT_DIM, device,
            )
            logging.info("Extracting %s test images", len(test_df))
            H_test = extract_clip_features(
                model, tqdm(test_loader, desc="clip-test"),
                EXPECTED_HOOK_NAME, len(test_df), EXPECTED_INPUT_DIM, device,
            )
            np.save(clip_train_path, H_train)
            np.save(clip_test_path, H_test)
            del model
        else:
            logging.info("Loading cached raw CLIP features")
            H_train = np.load(clip_train_path)
            H_test = np.load(clip_test_path)

        H_tr = np.ascontiguousarray(H_train[train_idx], dtype=np.float32)
        H_val = np.ascontiguousarray(H_train[val_idx], dtype=np.float32)
        y_train = np.load(features_dir / "y_train.npy")
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        cv_results, best_C = cross_validate_regularization(
            Z=H_tr, y=y_tr,
            c_candidates=DEFAULT_C_CANDIDATES,
            n_splits=DEFAULT_CV_FOLDS,
            max_iter=args.max_iter,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        logging.info("CLIP baseline best_C=%s", best_C)

        clf_full = build_multinomial_logreg(C=best_C, max_iter=args.max_iter, random_state=RANDOM_SEED)
        clf_full.fit(H_tr, y_tr)
        full_val_acc = float(clf_full.score(H_val, y_val))
        logging.info("CLIP baseline full val_acc=%.4f (%s dims)", full_val_acc, H_tr.shape[1])

        sigma_tr = compute_sigma_train(H_tr)
        active = np.arange(H_tr.shape[1], dtype=np.int64)
        pruning_curve: list[dict[str, Any]] = []

        while len(active) > CLIP_PRUNING_K_MIN:
            clf = build_multinomial_logreg(C=best_C, max_iter=args.max_iter, random_state=RANDOM_SEED)
            clf.fit(H_tr[:, active], y_tr)
            val_acc = float(clf.score(H_val[:, active], y_val))
            n_iter = summarize_iterations(clf)
            pruning_curve.append({
                "n_features": int(len(active)),
                "val_acc": val_acc,
                "n_iter": n_iter,
            })
            logging.info("CLIP pruning: n_features=%s val_acc=%.4f", len(active), val_acc)

            importance = compute_feature_importance(clf.coef_, sigma_tr[active])
            prune_count = max(int(np.floor(args.prune_fraction * len(active))), 1)
            prune_count = min(prune_count, len(active) - CLIP_PRUNING_K_MIN)
            if prune_count <= 0:
                break
            prune_idx = np.argsort(importance, kind="stable")[:prune_count]
            keep_mask = np.ones(len(active), dtype=bool)
            keep_mask[prune_idx] = False
            active = active[keep_mask]

        if len(active) <= CLIP_PRUNING_K_MIN:
            clf = build_multinomial_logreg(C=best_C, max_iter=args.max_iter, random_state=RANDOM_SEED)
            clf.fit(H_tr[:, active], y_tr)
            val_acc = float(clf.score(H_val[:, active], y_val))
            pruning_curve.append({
                "n_features": int(len(active)),
                "val_acc": val_acc,
                "n_iter": summarize_iterations(clf),
            })

        output = {
            "best_C": best_C,
            "full_val_acc": full_val_acc,
            "d_model": EXPECTED_INPUT_DIM,
            "cv_results": cv_results,
            "pruning_curve": pruning_curve,
        }
        write_json(output_path, output)

        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={
                "status": "ok",
                "full_val_acc": full_val_acc,
                "best_C": best_C,
                "n_pruning_rounds": len(pruning_curve),
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
