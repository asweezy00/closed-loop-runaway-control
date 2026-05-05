"""Train and select ML seizure-like event detectors."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, WindowedFeatures
from .metrics import BinaryMetrics, binary_metrics, choose_threshold_youden


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class FittedModel:
    name: str
    estimator: object
    threshold: float
    metrics: BinaryMetrics
    model_rows: list[dict[str, float | str]]
    test_scores: np.ndarray
    test_predictions: np.ndarray
    test_scores_by_model: dict[str, np.ndarray]
    split: SplitData


def make_group_splits(groups: list[str], random_state: int = 7) -> SplitData:
    """Split by record id to avoid window leakage across train/test."""
    groups_arr = np.asarray(groups)
    all_idx = np.arange(groups_arr.size)
    first_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_val_idx, test_idx = next(first_split.split(all_idx, groups=groups_arr))

    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + 1)
    train_rel, val_rel = next(second.split(train_val_idx, groups=groups_arr[train_val_idx]))
    return SplitData(
        train_idx=train_val_idx[train_rel],
        val_idx=train_val_idx[val_rel],
        test_idx=test_idx,
    )


def candidate_models(random_state: int = 7) -> dict[str, object]:
    """Models compared before selecting the validation ROC-AUC winner."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def score_estimator(estimator: object, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    raise TypeError("estimator must expose predict_proba or decision_function")


def fit_best_model(dataset: WindowedFeatures, random_state: int = 7) -> FittedModel:
    """Fit candidate models and keep the validation ROC-AUC winner."""
    split = make_group_splits(dataset.record_ids, random_state=random_state)
    x_train, y_train = dataset.x[split.train_idx], dataset.y[split.train_idx]
    x_val, y_val = dataset.x[split.val_idx], dataset.y[split.val_idx]
    x_test, y_test = dataset.x[split.test_idx], dataset.y[split.test_idx]

    rows: list[dict[str, float | str]] = []
    fitted: dict[str, object] = {}
    val_scores: dict[str, np.ndarray] = {}
    test_scores_by_model: dict[str, np.ndarray] = {}
    for name, estimator in candidate_models(random_state=random_state).items():
        estimator.fit(x_train, y_train)
        fitted[name] = estimator
        val_score = score_estimator(estimator, x_val)
        test_score = score_estimator(estimator, x_test)
        val_scores[name] = val_score
        test_scores_by_model[name] = test_score
        rows.append(
            {
                "model": name,
                "validation_roc_auc": float(roc_auc_score(y_val, val_score)),
                "validation_pr_auc": float(average_precision_score(y_val, val_score)),
                "test_roc_auc_before_refit": float(roc_auc_score(y_test, test_score)),
                "test_pr_auc_before_refit": float(average_precision_score(y_test, test_score)),
            }
        )

    best_name = max(rows, key=lambda row: float(row["validation_roc_auc"]))["model"]
    best_estimator = candidate_models(random_state=random_state)[str(best_name)]
    train_val_idx = np.concatenate([split.train_idx, split.val_idx])
    best_estimator.fit(dataset.x[train_val_idx], dataset.y[train_val_idx])

    threshold = choose_threshold_youden(dataset.y[split.val_idx], val_scores[str(best_name)])
    test_scores = score_estimator(best_estimator, x_test)
    test_scores_by_model[str(best_name)] = test_scores
    metrics = binary_metrics(y_test, test_scores, threshold)
    test_predictions = (test_scores >= threshold).astype(int)

    return FittedModel(
        name=str(best_name),
        estimator=best_estimator,
        threshold=threshold,
        metrics=metrics,
        model_rows=rows,
        test_scores=test_scores,
        test_predictions=test_predictions,
        test_scores_by_model=test_scores_by_model,
        split=split,
    )


def prediction_rows(dataset: WindowedFeatures, fitted: FittedModel) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for out_idx, idx in enumerate(fitted.split.test_idx):
        rows.append(
            {
                "row": int(idx),
                "record_id": dataset.record_ids[idx],
                "record_type": dataset.record_types[idx],
                "seed": int(dataset.seeds[idx]),
                "center_s": float(dataset.centers_s[idx]),
                "true_label": int(dataset.y[idx]),
                "score": float(fitted.test_scores[out_idx]),
                "predicted_label": int(fitted.test_predictions[out_idx]),
            }
        )
    return rows


def feature_importance_rows(
    dataset: WindowedFeatures,
    fitted: FittedModel,
    random_state: int = 7,
) -> list[dict[str, float | str]]:
    """Permutation importance on the held-out test split."""
    x_test = dataset.x[fitted.split.test_idx]
    y_test = dataset.y[fitted.split.test_idx]
    result = permutation_importance(
        fitted.estimator,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=8,
        random_state=random_state,
        n_jobs=1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    rows: list[dict[str, float | str]] = []
    for idx in order:
        rows.append(
            {
                "feature": FEATURE_NAMES[int(idx)],
                "importance_mean_auc_drop": float(result.importances_mean[int(idx)]),
                "importance_std": float(result.importances_std[int(idx)]),
            }
        )
    return rows
