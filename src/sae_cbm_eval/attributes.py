from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_attributes_dir(dataset_root: Path) -> Path:
    """Return the directory containing attribute files, with fallback."""
    standard = dataset_root / "attributes"
    if (standard / "attributes.txt").exists():
        return standard
    fallback = dataset_root.parent
    if (fallback / "attributes.txt").exists():
        return fallback
    raise FileNotFoundError(
        f"attributes.txt not found at {standard / 'attributes.txt'} "
        f"or fallback {fallback / 'attributes.txt'}"
    )


def parse_attribute_names(dataset_root: Path) -> pd.DataFrame:
    attrs_dir = _resolve_attributes_dir(dataset_root)
    path = attrs_dir / "attributes.txt"
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        attr_id_str, name = line.split(" ", 1)
        rows.append({"attribute_id": int(attr_id_str), "attribute_name": name.strip()})
    return pd.DataFrame(rows)


def parse_image_attribute_labels(dataset_root: Path) -> pd.DataFrame:
    attrs_dir = _resolve_attributes_dir(dataset_root)
    path = attrs_dir / "image_attribute_labels.txt"
    if not path.exists():
        raise FileNotFoundError(f"Image attribute labels file not found: {path}")
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
