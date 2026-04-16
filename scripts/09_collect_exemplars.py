from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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


SCRIPT_NAME = "09_collect_exemplars"
DEFAULT_N_POSITIVE = 15
DEFAULT_N_NEGATIVE = 15
DEFAULT_MAX_PER_CLASS = 3
DEFAULT_OPERATING_POINTS = ["0.01", "0.02", "0.05"]
THUMB_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect top/bottom activating images for retained features."
    )
    parser.add_argument("--dataset-root", type=Path, default=project_path("data", CUB_DIR_NAME))
    parser.add_argument("--features-dir", type=Path, default=project_path("features"))
    parser.add_argument("--results-dir", type=Path, default=project_path("results"))
    parser.add_argument("--output-dir", type=Path, default=project_path("exemplars"))
    parser.add_argument("--n-positive", type=int, default=DEFAULT_N_POSITIVE)
    parser.add_argument("--n-negative", type=int, default=DEFAULT_N_NEGATIVE)
    parser.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS)
    parser.add_argument(
        "--operating-points",
        nargs="+",
        default=DEFAULT_OPERATING_POINTS,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text())


def select_diverse_top_k(
    activations: np.ndarray,
    class_ids: np.ndarray,
    k: int,
    max_per_class: int,
) -> np.ndarray:
    """Select top-k activating image indices with class diversity."""
    sorted_idx = np.argsort(activations)[::-1]
    selected = []
    class_counts: dict[int, int] = {}

    for idx in sorted_idx:
        if len(selected) >= k:
            break
        cls = int(class_ids[idx])
        if class_counts.get(cls, 0) >= max_per_class:
            continue
        if activations[idx] <= 0:
            break
        selected.append(int(idx))
        class_counts[cls] = class_counts.get(cls, 0) + 1

    return np.array(selected, dtype=np.int64)


def select_bottom_k(
    activations: np.ndarray,
    class_ids: np.ndarray,
    k: int,
    max_per_class: int,
    exclude: set[int] | None = None,
) -> np.ndarray:
    """Select bottom-k activating image indices (low or zero activation)."""
    sorted_idx = np.argsort(activations)
    selected = []
    class_counts: dict[int, int] = {}
    exclude = exclude or set()

    for idx in sorted_idx:
        if len(selected) >= k:
            break
        if int(idx) in exclude:
            continue
        cls = int(class_ids[idx])
        if class_counts.get(cls, 0) >= max_per_class:
            continue
        selected.append(int(idx))
        class_counts[cls] = class_counts.get(cls, 0) + 1

    return np.array(selected, dtype=np.int64)


def create_montage(
    image_paths: list[Path],
    n_cols: int = 5,
    thumb_size: int = THUMB_SIZE,
) -> Image.Image:
    """Create a grid montage from a list of image paths."""
    n = len(image_paths)
    n_rows = (n + n_cols - 1) // n_cols
    montage = Image.new("RGB", (n_cols * thumb_size, n_rows * thumb_size), (255, 255, 255))

    for i, path in enumerate(image_paths):
        row, col = divmod(i, n_cols)
        with Image.open(path) as img:
            img = img.convert("RGB").resize((thumb_size, thumb_size), Image.LANCZOS)
            montage.paste(img, (col * thumb_size, row * thumb_size))

    return montage


def build_labeling_prompt(feature_rank: int) -> str:
    return (
        "You are shown two sets of bird images.\n\n"
        "TOP ROW(S): These images strongly activate a particular visual feature.\n"
        "BOTTOM ROW(S): These images do NOT activate this feature.\n\n"
        "Describe the visual property that is present in the top images but absent "
        "in the bottom images. Use a short phrase (3-10 words). "
        "Focus only on visible properties: color, texture, shape, pattern, or body part. "
        "Do NOT name any bird species. Do NOT mention non-visual properties.\n\n"
        f"Feature #{feature_rank}"
    )


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    output_dir = ensure_dir(args.output_dir)

    try:
        load_project_env()
        set_reproducibility(RANDOM_SEED)

        dataset_root = Path(args.dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"CUB dataset not found: {dataset_root}")

        metadata = parse_cub_metadata(dataset_root)
        validate_cub_metadata(metadata)
        train_df, _, _ = split_cub_metadata(metadata)

        Z_train = np.load(args.features_dir / "Z_train.npy", mmap_mode="r")
        y_train = np.load(args.features_dir / "y_train.npy")

        final_test_results = load_json(args.results_dir / "final_test_results.json")

        images_root = dataset_root / "images"

        for op_result in final_test_results:
            delta = op_result["delta"]
            if delta not in args.operating_points:
                continue

            op_dir = output_dir / f"delta_{delta}"
            prompts_output = op_dir / "prompts.json"
            if prompts_output.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output exists: {prompts_output}. Use --overwrite."
                )

            feature_indices = np.array(op_result["active_feature_indices"], dtype=np.int64)
            op_dir = ensure_dir(op_dir)
            logging.info(
                "Processing delta=%s with %s features", delta, len(feature_indices),
            )

            prompts_data: list[dict[str, Any]] = []

            for rank, feat_idx in enumerate(feature_indices):
                z_col = np.asarray(Z_train[:, feat_idx], dtype=np.float32)

                top_idx = select_diverse_top_k(
                    z_col, y_train, args.n_positive, args.max_per_class,
                )
                bottom_idx = select_bottom_k(
                    z_col, y_train, args.n_negative, args.max_per_class,
                    exclude=set(top_idx.tolist()),
                )

                top_paths = [
                    images_root / train_df.iloc[int(i)]["relative_path"]
                    for i in top_idx
                ]
                bottom_paths = [
                    images_root / train_df.iloc[int(i)]["relative_path"]
                    for i in bottom_idx
                ]

                if len(top_paths) > 0 and len(bottom_paths) > 0:
                    top_montage = create_montage(top_paths)
                    bottom_montage = create_montage(bottom_paths)

                    sep_height = 4
                    combined = Image.new(
                        "RGB",
                        (top_montage.width, top_montage.height + sep_height + bottom_montage.height),
                        (128, 128, 128),
                    )
                    combined.paste(top_montage, (0, 0))
                    combined.paste(bottom_montage, (0, top_montage.height + sep_height))

                    montage_path = op_dir / f"feature_{rank:04d}_idx{feat_idx}.jpg"
                    combined.save(montage_path, quality=90)

                prompt = build_labeling_prompt(rank)
                prompts_data.append({
                    "rank": rank,
                    "feature_index": int(feat_idx),
                    "montage_file": f"feature_{rank:04d}_idx{feat_idx}.jpg",
                    "prompt": prompt,
                    "top_activations": [float(z_col[i]) for i in top_idx],
                    "bottom_activations": [float(z_col[i]) for i in bottom_idx],
                    "top_image_indices": top_idx.tolist(),
                    "bottom_image_indices": bottom_idx.tolist(),
                })

            write_json(op_dir / "prompts.json", prompts_data)
            logging.info("Wrote %s montages to %s", len(prompts_data), op_dir)

        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=args.results_dir,
            extra={
                "status": "ok",
                "operating_points": args.operating_points,
                "output_dir": str(output_dir),
            },
        )
        return 0
    except Exception as exc:
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=args.results_dir,
            extra={"status": "error", "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
