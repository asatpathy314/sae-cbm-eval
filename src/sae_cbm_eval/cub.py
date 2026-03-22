from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from sae_cbm_eval.constants import (
    CUB_ARCHIVE_NAME,
    CUB_DOWNLOAD_URL,
    EXPECTED_NUM_CLASSES,
    EXPECTED_TEST_IMAGES,
    EXPECTED_TOTAL_IMAGES,
    EXPECTED_TRAIN_IMAGES,
)


def ensure_cub_dataset(
    dataset_root: Path,
    *,
    download_if_missing: bool = False,
    archive_path: Path | None = None,
    download_url: str = CUB_DOWNLOAD_URL,
) -> Path:
    if dataset_root.exists():
        return dataset_root

    if not download_if_missing:
        raise FileNotFoundError(
            f"CUB dataset not found at {dataset_root}. "
            f"Download it manually from {download_url} or rerun with --download-if-missing."
        )

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    archive_path = archive_path or (dataset_root.parent / CUB_ARCHIVE_NAME)
    if not archive_path.exists():
        urllib.request.urlretrieve(download_url, archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dataset_root.parent)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Expected extracted dataset at {dataset_root}, but it was not created."
        )
    return dataset_root


def _read_two_column_table(
    path: Path,
    *,
    names: list[str],
    dtype: dict[str, str | type],
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=names,
        dtype=dtype,
        engine="python",
    )


def parse_cub_metadata(dataset_root: Path) -> pd.DataFrame:
    images = _read_two_column_table(
        dataset_root / "images.txt",
        names=["image_id", "relative_path"],
        dtype={"image_id": np.int64, "relative_path": "string"},
    )
    labels = _read_two_column_table(
        dataset_root / "image_class_labels.txt",
        names=["image_id", "class_id_raw"],
        dtype={"image_id": np.int64, "class_id_raw": np.int64},
    )
    split = _read_two_column_table(
        dataset_root / "train_test_split.txt",
        names=["image_id", "is_training"],
        dtype={"image_id": np.int64, "is_training": np.int64},
    )

    metadata = images.merge(labels, on="image_id", how="inner").merge(
        split, on="image_id", how="inner"
    )
    metadata["class_id"] = metadata["class_id_raw"] - 1
    metadata = metadata.sort_values("image_id").reset_index(drop=True)
    metadata["relative_path"] = metadata["relative_path"].astype(str)
    metadata["class_id"] = metadata["class_id"].astype(np.int64)
    metadata["is_training"] = metadata["is_training"].astype(np.int64)
    return metadata[
        ["image_id", "relative_path", "class_id_raw", "class_id", "is_training"]
    ]


def validate_cub_metadata(metadata: pd.DataFrame, *, strict_counts: bool = True) -> dict[str, int]:
    required_columns = {
        "image_id",
        "relative_path",
        "class_id_raw",
        "class_id",
        "is_training",
    }
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)!r}")

    if metadata["image_id"].duplicated().any():
        raise ValueError("image_id values must be unique after parsing.")

    if metadata["class_id"].min() < 0:
        raise ValueError("class_id remapping produced a negative value.")
    if not set(metadata["is_training"].unique().tolist()).issubset({0, 1}):
        raise ValueError("is_training values must be binary 0/1.")

    train_count = int((metadata["is_training"] == 1).sum())
    test_count = int((metadata["is_training"] == 0).sum())
    class_count = int(metadata["class_id"].nunique())
    total_count = int(len(metadata))

    if strict_counts:
        problems: list[str] = []
        if total_count != EXPECTED_TOTAL_IMAGES:
            problems.append(
                f"expected total image count {EXPECTED_TOTAL_IMAGES}, observed {total_count}"
            )
        if train_count != EXPECTED_TRAIN_IMAGES:
            problems.append(
                f"expected train image count {EXPECTED_TRAIN_IMAGES}, observed {train_count}"
            )
        if test_count != EXPECTED_TEST_IMAGES:
            problems.append(
                f"expected test image count {EXPECTED_TEST_IMAGES}, observed {test_count}"
            )
        if class_count != EXPECTED_NUM_CLASSES:
            problems.append(
                f"expected class count {EXPECTED_NUM_CLASSES}, observed {class_count}"
            )
        if problems:
            raise ValueError("CUB metadata validation failed: " + "; ".join(problems))

    return {
        "total_count": total_count,
        "train_count": train_count,
        "test_count": test_count,
        "class_count": class_count,
    }


def split_cub_metadata(
    metadata: pd.DataFrame,
    *,
    max_images_per_split: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = metadata.loc[metadata["is_training"] == 1].copy().reset_index(drop=True)
    test_df = metadata.loc[metadata["is_training"] == 0].copy().reset_index(drop=True)

    if max_images_per_split is not None:
        train_df = train_df.iloc[:max_images_per_split].copy().reset_index(drop=True)
        test_df = test_df.iloc[:max_images_per_split].copy().reset_index(drop=True)

    train_df["row_index"] = np.arange(len(train_df), dtype=np.int64)
    test_df["row_index"] = np.arange(len(test_df), dtype=np.int64)

    train_mapping = train_df.assign(split="train")[
        ["split", "row_index", "image_id", "relative_path", "class_id"]
    ]
    test_mapping = test_df.assign(split="test")[
        ["split", "row_index", "image_id", "relative_path", "class_id"]
    ]
    row_mapping = pd.concat([train_mapping, test_mapping], ignore_index=True)

    return train_df, test_df, row_mapping


class CUBImageDataset(Dataset):
    def __init__(self, dataset_root: Path, records: pd.DataFrame, transform) -> None:
        self.images_root = dataset_root / "images"
        self.records = list(
            records[["row_index", "relative_path", "class_id"]].itertuples(index=False, name=None)
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row_index, relative_path, class_id = self.records[index]
        image_path = self.images_root / relative_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        return tensor, int(row_index), int(class_id)
