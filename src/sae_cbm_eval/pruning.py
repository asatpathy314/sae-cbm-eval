from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from sae_cbm_eval.classification import (
    build_multinomial_logreg,
    cross_validate_regularization,
    summarize_iterations,
)


def compute_sigma_train(Z_tr: np.ndarray) -> np.ndarray:
    return np.asarray(Z_tr.std(axis=0), dtype=np.float32)


def compute_feature_importance(W: np.ndarray, sigma_active: np.ndarray) -> np.ndarray:
    return np.linalg.norm(W, axis=0) * sigma_active


def compute_prune_count(
    *,
    n_active: int,
    prune_fraction: float,
    min_remaining: int | None = None,
) -> int:
    count = max(int(np.floor(prune_fraction * n_active)), 1)
    if min_remaining is not None:
        count = min(count, n_active - min_remaining)
    return int(count)


def iterative_pruning(
    *,
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    sigma_tr: np.ndarray,
    C: float,
    prune_fraction: float,
    k_min: int,
    max_iter: int,
    max_rounds: int,
    random_state: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    active = np.arange(Z_tr.shape[1], dtype=np.int64)
    results: list[dict[str, Any]] = []

    for round_idx in range(max_rounds):
        clf = build_multinomial_logreg(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        )
        clf.fit(Z_tr[:, active], y_tr)
        val_acc = float(clf.score(Z_val[:, active], y_val))
        n_iter = summarize_iterations(clf)

        round_result: dict[str, Any] = {
            "round": round_idx,
            "n_features": int(len(active)),
            "val_acc": val_acc,
            "feature_indices": active.copy(),
            "n_iter": n_iter,
            "converged": bool(n_iter < max_iter),
        }
        results.append(round_result)
        if progress_callback is not None:
            progress_callback(round_result)

        if len(active) <= k_min:
            break

        importance = compute_feature_importance(clf.coef_, sigma_tr[active])
        prune_count = compute_prune_count(
            n_active=len(active),
            prune_fraction=prune_fraction,
        )
        prune_idx = np.argsort(importance, kind="stable")[:prune_count]
        keep_mask = np.ones(len(active), dtype=bool)
        keep_mask[prune_idx] = False
        active = active[keep_mask]

    return results


def build_pruning_curve(results: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [(r["n_features"], r["val_acc"]) for r in results],
        dtype=np.float64,
    )


def serialize_pruning_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for result in results:
        payload.append(
            {
                "round": int(result["round"]),
                "n_features": int(result["n_features"]),
                "val_acc": float(result["val_acc"]),
                "n_iter": int(result["n_iter"]),
                "converged": bool(result["converged"]),
                "feature_indices": result["feature_indices"].tolist(),
            }
        )
    return payload


def compute_k_delta_table(
    results: list[dict[str, Any]],
    deltas: Iterable[float],
) -> dict[str, int | None]:
    baseline_acc = float(results[0]["val_acc"])
    table: dict[str, int | None] = {}
    for delta in deltas:
        threshold = baseline_acc - float(delta)
        valid = [result for result in results if float(result["val_acc"]) >= threshold]
        table[str(delta)] = min((int(r["n_features"]) for r in valid), default=None)
    return table


def select_nearest_rounds(
    results: list[dict[str, Any]],
    targets: Iterable[int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for target in targets:
        nearest = min(
            results,
            key=lambda result: (abs(int(result["n_features"]) - int(target)), int(result["round"])),
        )
        selected.append(nearest)
    return selected


def run_sensitivity_check(
    *,
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    results: list[dict[str, Any]],
    best_C: float,
    targets: Iterable[int],
    c_multipliers: Iterable[float],
    cv_folds: int,
    max_iter: int,
    random_state: int,
    n_jobs: int,
) -> list[dict[str, Any]]:
    sensitivity_results: list[dict[str, Any]] = []
    selected_rounds = select_nearest_rounds(results, targets)

    for requested_target, round_result in zip(targets, selected_rounds, strict=True):
        active = round_result["feature_indices"]
        c_candidates = [float(multiplier) * best_C for multiplier in c_multipliers]
        candidate_results, selected_best_C = cross_validate_regularization(
            Z=np.ascontiguousarray(Z_tr[:, active], dtype=np.float32),
            y=y_tr,
            c_candidates=c_candidates,
            n_splits=cv_folds,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        sensitivity_results.append(
            {
                "requested_n_features": int(requested_target),
                "actual_n_features": int(round_result["n_features"]),
                "round": int(round_result["round"]),
                "best_C": float(selected_best_C),
                "candidate_results": candidate_results,
            }
        )
    return sensitivity_results


def prune_to_k(
    *,
    Z_fit: np.ndarray,
    y_fit: np.ndarray,
    sigma_fit: np.ndarray,
    C: float,
    k_target: int,
    prune_fraction: float,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, Any]:
    active = np.arange(Z_fit.shape[1], dtype=np.int64)

    while len(active) > k_target:
        clf = build_multinomial_logreg(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        )
        clf.fit(Z_fit[:, active], y_fit)
        importance = compute_feature_importance(clf.coef_, sigma_fit[active])
        prune_count = compute_prune_count(
            n_active=len(active),
            prune_fraction=prune_fraction,
            min_remaining=k_target,
        )
        prune_idx = np.argsort(importance, kind="stable")[:prune_count]
        keep_mask = np.ones(len(active), dtype=bool)
        keep_mask[prune_idx] = False
        active = active[keep_mask]

    clf_final = build_multinomial_logreg(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
    )
    clf_final.fit(Z_fit[:, active], y_fit)
    return active, clf_final
