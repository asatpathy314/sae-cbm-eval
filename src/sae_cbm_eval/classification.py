from __future__ import annotations

import inspect
from typing import Any, Iterable

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


_LOGREG_SUPPORTS_MULTI_CLASS = "multi_class" in inspect.signature(LogisticRegression).parameters


def build_multinomial_logreg(
    *,
    C: float,
    max_iter: int,
    random_state: int,
    verbose: int = 0,
) -> LogisticRegression:
    kwargs: dict[str, Any] = {
        "C": C,
        "solver": "lbfgs",
        "max_iter": max_iter,
        "random_state": random_state,
        "verbose": verbose,
    }
    # sklearn 1.8 removed the explicit `multi_class` kwarg. With lbfgs and a
    # multiclass target, it now selects multinomial behavior implicitly.
    if _LOGREG_SUPPORTS_MULTI_CLASS:
        kwargs["multi_class"] = "multinomial"

    return LogisticRegression(
        **kwargs,
    )


def split_train_val_indices(
    *,
    n_samples: int,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_samples, dtype=np.int64)
    train_idx, val_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
    )


def summarize_iterations(estimator: LogisticRegression) -> int:
    return int(np.max(np.asarray(estimator.n_iter_)))


def cross_validate_regularization(
    *,
    Z: np.ndarray,
    y: np.ndarray,
    c_candidates: Iterable[float],
    n_splits: int,
    max_iter: int,
    random_state: int,
    n_jobs: int,
) -> tuple[dict[str, Any], float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: dict[str, Any] = {}
    ordered_candidates = [float(c) for c in c_candidates]

    for ci, C in enumerate(ordered_candidates, 1):
        logging.info(
            "CV candidate %s/%s: C=%s (%s-fold)",
            ci, len(ordered_candidates), C, n_splits,
        )
        clf = build_multinomial_logreg(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            verbose=1,
        )
        result = cross_validate(
            clf,
            Z,
            y,
            cv=cv,
            scoring="accuracy",
            return_estimator=True,
            n_jobs=n_jobs,
        )
        fold_iters = [summarize_iterations(est) for est in result["estimator"]]
        mean_acc = float(result["test_score"].mean())
        results[str(C)] = {
            "mean_acc": mean_acc,
            "std_acc": float(result["test_score"].std()),
            "fold_accs": [float(x) for x in result["test_score"].tolist()],
            "fold_iters": fold_iters,
            "converged": bool(all(it < max_iter for it in fold_iters)),
        }
        logging.info(
            "  C=%s done: mean_acc=%.4f, iters=%s",
            C, mean_acc, fold_iters,
        )

    best_C = max(ordered_candidates, key=lambda c: results[str(c)]["mean_acc"])
    return results, float(best_C)
