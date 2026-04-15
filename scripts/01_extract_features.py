from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sae_cbm_eval.constants import (
    CLIP_MODEL_ID,
    CUB_DIR_NAME,
    CUB_DOWNLOAD_URL,
    EXPECTED_CONTEXT_SIZE,
    EXPECTED_HOOK_LAYER,
    EXPECTED_HOOK_NAME,
    EXPECTED_INPUT_DIM,
    EXPECTED_SAE_DIM,
    PREPROCESS_ID,
    RANDOM_SEED,
    SAE_CONFIG_FILENAME,
    SAE_REPO_ID,
    SAE_WEIGHT_FILENAME,
    TOKEN_POLICY_ALL_MAXPOOL,
)
from sae_cbm_eval.cub import (
    CUBImageDataset,
    ensure_cub_dataset,
    parse_cub_metadata,
    split_cub_metadata,
    validate_cub_metadata,
)
from sae_cbm_eval.extraction import (
    build_preprocess,
    extract_split_features,
    load_clip_model_for_extraction,
    load_sae_for_extraction,
)
from sae_cbm_eval.runtime import (
    configure_runtime_logging,
    ensure_dir,
    get_package_version,
    load_project_env,
    project_path,
    resolve_device,
    set_reproducibility,
    sha256_file,
    write_hardware_report,
    write_json,
    write_run_manifest,
)

warnings.filterwarnings(
    "ignore",
    message=r".*Plotly version 5\.19\.0, which is not compatible with this version of Kaleido.*",
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"kaleido\._sync_server")

SCRIPT_NAME = "01_extract_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SAE features for CUB-200-2011."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_path("data", CUB_DIR_NAME),
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=project_path("data", "CUB_200_2011.tgz"),
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download and extract CUB-200-2011 if dataset-root does not exist.",
    )
    parser.add_argument(
        "--download-url",
        default=CUB_DOWNLOAD_URL,
    )
    parser.add_argument("--repo-id", default=SAE_REPO_ID)
    parser.add_argument("--weight-filename", default=SAE_WEIGHT_FILENAME)
    parser.add_argument("--config-filename", default=SAE_CONFIG_FILENAME)
    parser.add_argument("--clip-model-id", default=CLIP_MODEL_ID)
    parser.add_argument(
        "--verify-report-path",
        type=Path,
        default=project_path("results", "verify_sae.json"),
    )
    parser.add_argument(
        "--skip-verify-check",
        action="store_true",
        help="Skip checking that results/verify_sae.json exists and passed.",
    )
    parser.add_argument(
        "--skip-count-validation",
        action="store_true",
        help="Allow non-standard CUB metadata counts, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional cap per split for smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use: auto, cpu, cuda, or mps.",
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
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing feature/output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for the extractor itself.",
    )
    return parser.parse_args()


def ensure_verify_report_ok(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Verification report not found at {path}. Run scripts/00_verify_sae.py first."
        )
    report = json.loads(path.read_text())
    if not report.get("ok"):
        raise ValueError(
            f"Verification report at {path} did not pass. Fix script 0 before extraction."
        )
    return report


def check_overwrite(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist: {joined}. Rerun with --overwrite to replace them."
        )


def build_dataloader(
    dataset_root: Path,
    records: pd.DataFrame,
    transform,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> DataLoader:
    dataset = CUBImageDataset(dataset_root, records, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )


def save_labels(
    features_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    np.save(features_dir / "y_train.npy", train_df["class_id"].to_numpy(dtype=np.int64))
    np.save(features_dir / "y_test.npy", test_df["class_id"].to_numpy(dtype=np.int64))


def compute_post_cache_sanity(
    z_path: Path,
    y_path: Path,
    *,
    strict_counts: bool,
) -> dict[str, Any]:
    Z = np.load(z_path, mmap_mode="r")
    y = np.load(y_path)

    sigma = np.asarray(Z.std(axis=0))
    dead = int((np.asarray(Z.max(axis=0)) == 0).sum())
    counts = Counter(y.tolist())

    if strict_counts:
        if len(counts) != 200:
            raise ValueError(
                f"Expected 200 classes in training labels, observed {len(counts)}"
            )
        if not all(29 <= c <= 30 for c in counts.values()):
            raise ValueError(
                "Expected per-class training counts to fall in [29, 30], "
                f"observed range [{min(counts.values())}, {max(counts.values())}]"
            )

    return {
        "z_shape": list(Z.shape),
        "y_shape": list(y.shape),
        "dead_features": dead,
        "sigma_zero": int((sigma == 0).sum()),
        "median_sigma": float(np.median(sigma)),
        "n_classes": len(counts),
        "min_class_count": int(min(counts.values())) if counts else 0,
        "max_class_count": int(max(counts.values())) if counts else 0,
    }


def build_extraction_meta(
    *,
    args: argparse.Namespace,
    sae_config_path: Path,
    observed_hook_shape: list[int],
    first_batch_stats: dict[str, Any],
    metadata_counts: dict[str, int],
    sanity_checks: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    return {
        "repo_id": args.repo_id,
        "weight_filename": args.weight_filename,
        "config_filename": args.config_filename,
        "clip_model_name": args.clip_model_id,
        "preprocess_id": PREPROCESS_ID,
        "image_size": 224,
        "hook_name": EXPECTED_HOOK_NAME,
        "layer": EXPECTED_HOOK_LAYER,
        "token_policy": TOKEN_POLICY_ALL_MAXPOOL,
        "n_features": EXPECTED_SAE_DIM,
        "input_dim": EXPECTED_INPUT_DIM,
        "vit_prisma_version": get_package_version("vit-prisma"),
        "torch_version": torch.__version__,
        "sae_config_path": str(sae_config_path),
        "sae_config_hash": sha256_file(sae_config_path),
        "dtype_policy": {
            "model_dtype": "torch.float32",
            "sae_dtype": "torch.float32",
            "feature_cache_dtype": "float32",
        },
        "observed_hook_shape": observed_hook_shape,
        "context_size": EXPECTED_CONTEXT_SIZE,
        "device": device,
        "dataset_root": str(args.dataset_root),
        "metadata_counts": metadata_counts,
        "first_batch_stats": first_batch_stats,
        "post_cache_sanity": sanity_checks,
    }


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    device = resolve_device(args.device)

    features_dir = ensure_dir(args.features_dir)
    results_dir = ensure_dir(args.results_dir)

    output_paths = [
        features_dir / "Z_train.npy",
        features_dir / "Z_test.npy",
        features_dir / "y_train.npy",
        features_dir / "y_test.npy",
        features_dir / "row_mapping.csv",
        features_dir / "extraction_meta.json",
    ]

    try:
        if args.max_images_per_split is not None and args.max_images_per_split <= 0:
            raise ValueError("--max-images-per-split must be positive when provided.")
        check_overwrite(output_paths, overwrite=args.overwrite)
        hf_token = load_project_env()
        set_reproducibility(RANDOM_SEED)
        write_hardware_report(results_dir=results_dir)

        verify_report = None
        if not args.skip_verify_check:
            verify_report = ensure_verify_report_ok(args.verify_report_path)

        dataset_root = ensure_cub_dataset(
            args.dataset_root,
            download_if_missing=args.download_if_missing,
            archive_path=args.archive_path,
            download_url=args.download_url,
        )

        metadata = parse_cub_metadata(dataset_root)
        metadata_counts = validate_cub_metadata(
            metadata,
            strict_counts=not args.skip_count_validation,
        )
        train_df, test_df, row_mapping = split_cub_metadata(
            metadata,
            max_images_per_split=args.max_images_per_split,
        )

        row_mapping.to_csv(features_dir / "row_mapping.csv", index=False)
        save_labels(features_dir, train_df, test_df)

        preprocess = build_preprocess(args.clip_model_id)
        model = load_clip_model_for_extraction(
            clip_model_id=args.clip_model_id,
            device=device,
        )
        sae, sae_config_path, _weights_path = load_sae_for_extraction(
            repo_id=args.repo_id,
            weight_filename=args.weight_filename,
            config_filename=args.config_filename,
            device=device,
            hf_token=hf_token,
        )

        train_loader = build_dataloader(
            dataset_root,
            train_df,
            preprocess,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        test_loader = build_dataloader(
            dataset_root,
            test_df,
            preprocess,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        logging.info("Extracting %s training images", len(train_df))
        train_stats = extract_split_features(
            model=model,
            sae=sae,
            dataloader=tqdm(train_loader, desc="train", leave=False),
            hook_name=EXPECTED_HOOK_NAME,
            output_path=features_dir / "Z_train.npy",
            n_rows=len(train_df),
            n_features=EXPECTED_SAE_DIM,
            device=device,
        )
        logging.info("Extracting %s test images", len(test_df))
        test_stats = extract_split_features(
            model=model,
            sae=sae,
            dataloader=tqdm(test_loader, desc="test", leave=False),
            hook_name=EXPECTED_HOOK_NAME,
            output_path=features_dir / "Z_test.npy",
            n_rows=len(test_df),
            n_features=EXPECTED_SAE_DIM,
            device=device,
        )

        sanity_checks = compute_post_cache_sanity(
            features_dir / "Z_train.npy",
            features_dir / "y_train.npy",
            strict_counts=not args.skip_count_validation,
        )
        extraction_meta = build_extraction_meta(
            args=args,
            sae_config_path=sae_config_path,
            observed_hook_shape=train_stats["observed_hook_shape"],
            first_batch_stats=train_stats["first_batch_stats"],
            metadata_counts=metadata_counts,
            sanity_checks=sanity_checks,
            device=device,
        )
        extraction_meta["test_first_batch_stats"] = test_stats["first_batch_stats"]
        extraction_meta["verify_report_path"] = (
            str(args.verify_report_path) if verify_report is not None else None
        )
        write_json(features_dir / "extraction_meta.json", extraction_meta)

        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=args.repo_id,
            clip_model_id=args.clip_model_id,
            results_dir=results_dir,
            extra={
                "status": "ok",
                "dataset_root": str(dataset_root),
                "features_dir": str(features_dir),
                "results_dir": str(results_dir),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "device": device,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "max_images_per_split": args.max_images_per_split,
            },
        )
    except Exception as exc:
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=args.repo_id,
            clip_model_id=args.clip_model_id,
            results_dir=results_dir,
            extra={
                "status": "error",
                "error": str(exc),
                "dataset_root": str(args.dataset_root),
                "features_dir": str(features_dir),
                "results_dir": str(results_dir),
            },
        )
        logging.exception("Feature extraction failed")
        return 1

    logging.info(
        "Feature extraction completed. Wrote caches to %s and metadata to %s",
        features_dir,
        features_dir / "extraction_meta.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
