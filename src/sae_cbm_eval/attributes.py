from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def parse_attribute_names(dataset_root: Path) -> pd.DataFrame:
    path = dataset_root / "attributes" / "attributes.txt"
    if not path.exists():
        raise FileNotFoundError(f"Attribute names file not found: {path}")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        attr_id_str, name = line.split(" ", 1)
        rows.append({"attribute_id": int(attr_id_str), "attribute_name": name.strip()})
    return pd.DataFrame(rows)


def parse_image_attribute_labels(dataset_root: Path) -> pd.DataFrame:
    path = dataset_root / "attributes" / "image_attribute_labels.txt"
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
) -> tuple[np.ndarray, list[str]]:
    """Build a (n_images, 312) binary attribute matrix aligned with image_ids."""
    attr_names_df = parse_attribute_names(dataset_root)
    labels_df = parse_image_attribute_labels(dataset_root)

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
