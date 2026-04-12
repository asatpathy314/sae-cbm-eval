from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_feature_attribute_auroc(
    Z: np.ndarray,
    A: np.ndarray,
    feature_indices: np.ndarray,
) -> np.ndarray:
    """Compute AUROC between each retained feature and each attribute.

    Returns shape (n_retained_features, n_attributes).
    """
    n_features = len(feature_indices)
    n_attrs = A.shape[1]
    auroc_matrix = np.full((n_features, n_attrs), np.nan, dtype=np.float64)

    for fi, feat_idx in enumerate(feature_indices):
        z_col = Z[:, feat_idx].astype(np.float64)
        if z_col.std() < 1e-12:
            continue
        for ai in range(n_attrs):
            a_col = A[:, ai]
            n_pos = int(a_col.sum())
            n_neg = len(a_col) - n_pos
            if n_pos < 5 or n_neg < 5:
                continue
            try:
                auroc_matrix[fi, ai] = roc_auc_score(a_col, z_col)
            except ValueError:
                continue

    return auroc_matrix


def best_matched_attributes(
    auroc_matrix: np.ndarray,
    attr_names: list[str],
) -> list[dict[str, Any]]:
    """For each feature, find the attribute with highest AUROC."""
    results = []
    for fi in range(auroc_matrix.shape[0]):
        row = auroc_matrix[fi]
        valid_mask = ~np.isnan(row)
        if not valid_mask.any():
            results.append({
                "feature_rank": fi,
                "best_attr_idx": -1,
                "best_attr_name": "none",
                "best_auroc": float("nan"),
            })
            continue
        best_idx = int(np.nanargmax(row))
        results.append({
            "feature_rank": fi,
            "best_attr_idx": best_idx,
            "best_attr_name": attr_names[best_idx],
            "best_auroc": float(row[best_idx]),
        })
    return results


def permutation_baseline(
    Z: np.ndarray,
    A: np.ndarray,
    feature_indices: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute best-match AUROC distribution under shuffled attribute labels."""
    n_features = len(feature_indices)
    best_aurocs = np.full((n_permutations, n_features), np.nan, dtype=np.float64)

    for pi in range(n_permutations):
        perm = rng.permutation(A.shape[0])
        A_shuffled = A[perm]
        auroc_mat = compute_feature_attribute_auroc(Z, A_shuffled, feature_indices)
        for fi in range(n_features):
            row = auroc_mat[fi]
            if np.any(~np.isnan(row)):
                best_aurocs[pi, fi] = float(np.nanmax(row))

    return best_aurocs


def random_feature_baseline_auroc(
    Z: np.ndarray,
    A: np.ndarray,
    n_features_to_sample: int,
    n_total_features: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute best-match AUROC distribution for randomly chosen feature sets."""
    best_aurocs_all = []

    for si in range(n_samples):
        rand_indices = rng.choice(n_total_features, size=n_features_to_sample, replace=False)
        auroc_mat = compute_feature_attribute_auroc(Z, A, rand_indices)
        per_feature_best = np.nanmax(auroc_mat, axis=1)
        best_aurocs_all.append(per_feature_best)

    return np.array(best_aurocs_all, dtype=np.float64)
