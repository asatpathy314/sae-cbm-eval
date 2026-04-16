from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_attribute_file(dataset_root: Path, filename: str) -> Path:
    """Find a CUB attribute file, checking standard location and dataset-root fallback."""
    standard = dataset_root / "attributes" / filename
    if standard.exists():
        return standard
    fallback = dataset_root.parent / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"{filename} not found at {standard} or fallback {fallback}"
    )


def parse_attribute_names(dataset_root: Path) -> pd.DataFrame:
    path = _resolve_attribute_file(dataset_root, "attributes.txt")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        attr_id_str, name = line.split(" ", 1)
        rows.append({"attribute_id": int(attr_id_str), "attribute_name": name.strip()})
    return pd.DataFrame(rows)


def parse_image_attribute_labels(dataset_root: Path) -> pd.DataFrame:
    path = _resolve_attribute_file(dataset_root, "image_attribute_labels.txt")
    # A handful of rows in the CUB attribute labels file have stray extra
    # fields; skip them rather than failing the whole pipeline.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
        dtype={
            "image_id": np.int64,
            "attribute_id": np.int64,
            "is_present": np.int64,
            "certainty_id": np.int64,
            "time": float,
        },
        engine="python",
        on_bad_lines="skip",
    )
    return df


def build_attribute_matrix(
    dataset_root: Path,
    image_ids: np.ndarray,
    *,
    min_certainty: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a (n_images, 312) binary attribute matrix aligned with image_ids."""
    attr_names_df = parse_attribute_names(dataset_root)
    labels_df = parse_image_attribute_labels(dataset_root)

    if min_certainty is not None:
        labels_df = labels_df[labels_df["certainty_id"] >= min_certainty]

    n_attrs = len(attr_names_df)

    id_to_row = {int(img_id): i for i, img_id in enumerate(image_ids)}

    matrix = np.zeros((len(image_ids), n_attrs), dtype=np.int8)
    for _, row in labels_df.iterrows():
        img_id = int(row["image_id"])
        if img_id not in id_to_row:
            continue
        attr_idx = int(row["attribute_id"]) - 1
        if 0 <= attr_idx < n_attrs:
            matrix[id_to_row[img_id], attr_idx] = int(row["is_present"])

    attr_names = attr_names_df["attribute_name"].tolist()
    return matrix, attr_names
