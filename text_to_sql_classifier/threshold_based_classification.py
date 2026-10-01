"""
Threshold selection algorithms
"""
import math
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ThresholdAlgorithm = Literal[
    "roc_based_threshold",
    "f_1_score_based_threshold",
    "f_0_5_score_based_threshold",
    "database_specific_roc_based_threshold",
    "database_specific_f_0_5_score_based_threshold",
    "accuracy_restriction_based_threshold",
]


def accuracy_restriction_based_threshold_selection(scores: np.ndarray, labels: np.ndarray, target_accuracy: float) -> float:
    """
    Returns the threshold value that maximizes the coverage while keeping the accuracy above the target_accuracy.

    Args:
        - scores: a list with the uncertainty scores.
        - labels: a list with boolean values. True/1 means that the point should be accepted.
        - target_accuracy: the minimum accuracy that should be achieved.

    Returns:
        The threshold value that maximizes the coverage while keeping the accuracy above the target_accuracy.

    """
    candidates = np.unique(scores)

    best_threshold = float(candidates[0])
    best_coverage = 0.0
    for candidate in candidates:
        rejected = scores > candidate
        accepted = ~rejected

        tp = int((accepted & labels).sum())
        fn = int((rejected & labels).sum())
        fp = int((accepted & ~labels).sum())
        tn = int((rejected & ~labels).sum())

        selective_accuracy = tp / (tp+fp) if (tp + fp) > 0 else 100.0
        coverage = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if selective_accuracy >= target_accuracy and coverage > best_coverage:
            best_coverage = coverage
            best_threshold = float(candidate)

    return best_threshold


def roc_based_threshold_selection(scores: np.ndarray, labels: np.ndarray) -> float:
    """ 
    Returns the threshold value that maximizes the TPR - FPR.

    Args:
        - scores: a list with the uncertainty scores.
        - labels: a list with boolean values. True/1 means that the point should be accepted.

    Returns:
        The threshold value that maximizes the TPR - FPR.

    """

    candidates = np.unique(scores)

    best_threshold = float(candidates[0])
    best_score = float("-inf")
    for candidate in candidates:
        rejected = scores > candidate
        accepted = ~rejected

        tp = int((accepted & labels).sum())
        fn = int((rejected & labels).sum())
        fp = int((accepted & ~labels).sum())
        tn = int((rejected & ~labels).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        score = tpr - fpr
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)

    return best_threshold


def f_score_based_threshold_selection(scores: np.ndarray, labels: np.ndarray, beta: float) -> float:
    """
    Returns the threshold value that maximizes the F-score defined as

    (1+beta^2) * (precision*recall) / (b^2*precision + recall)

    precision or selective_accuracy: the ratio of the correct answers over the non-rejected set
    recall - coverage_over_correct: the ratio of the correct answers that were not rejected

    Args:
        - scores: a list with the uncertainty scores
        - labels: a list with boolean values. True/1 means that the point should be accepted.
        - beta: beta=1.0 weights precision and recall equally (F1); beta=0.5 weights precision
          (selective_accuracy) more heavily than recall (coverage) - see
          "f_1_score_based_threshold"/"f_0_5_score_based_threshold" in ThresholdAlgorithm.

    Returns:
        The threshold value that maximizes the F-score.

    """
    candidates = np.unique(scores)

    best_threshold = float(candidates[0])
    best_score = float("-inf")
    for candidate in candidates:
        rejected = scores > candidate
        accepted = ~rejected

        tp = int((accepted & labels).sum())
        fp = int((accepted & ~labels).sum())
        fn = int((rejected & labels).sum())

        if tp + fp == 0 or tp + fn == 0:
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        denom = beta**2 * precision + recall
        f_score = (1 + beta**2) * precision * recall / denom if denom > 0 else 0.0

        if f_score > best_score:
            best_score = f_score
            best_threshold = float(candidate)

    return best_threshold


def database_specific_roc_based_threshold_selection(scores: np.ndarray, labels: np.ndarray, db_ids: list[str]) -> dict[str, float]:
    """Returns the threshold value that maximizes the TPR - FPR for each database"""
    
    db_ids_arr = np.asarray(db_ids)

    thresholds_by_db: dict[str, float] = {}
    for db_id in np.unique(db_ids_arr):
        db_mask = db_ids_arr == db_id
        thresholds_by_db[str(db_id)] = roc_based_threshold_selection(scores[db_mask], labels[db_mask])

    return thresholds_by_db


def database_specific_f_0_5_score_threshold_selection(scores: np.ndarray, labels: np.ndarray, db_ids: list[str]) -> dict[str, float]:
    """Returns the threshold value that maximizes the F-0.5 score for each database"""

    db_ids_arr = np.asarray(db_ids)

    thresholds_by_db: dict[str, float] = {}
    for db_id in np.unique(db_ids_arr):
        db_mask = db_ids_arr == db_id
        thresholds_by_db[str(db_id)] = f_score_based_threshold_selection(scores[db_mask], labels[db_mask], beta=0.5)

    return thresholds_by_db


def _subsampling(x, y, size, sample_distribution: Literal["initial", "equal"] = "initial", random_state=None):
    """
    Returns a subsample of the given pair lists.

    Args:
        - size: how much of the data to keep for threshold selection. A float in (0, 1] is a
          fraction of the full set; an int > 1 is an absolute number of points.
        - sample_distribution: "initial" stratifies by label so the subsample preserves the
          correct/incorrect ratio of the full set (default). "equal" draws an equal number of
          points from each label instead.
        - random_state: seed for reproducibility.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    n_total = len(x_arr)

    if isinstance(size, float) and 0.0 < size <= 1.0:
        n_sample = round(n_total * size)
    else:
        if size > n_total:
            raise ValueError(f"Requested sample size {size} is larger than the total number of points {n_total}.")
        n_sample = int(size)

    if sample_distribution == "equal":
        rng = np.random.default_rng(random_state)
        categories = np.unique(y_arr)
        n_per_category = max(1, n_sample // len(categories))

        sample_indices = np.concatenate([
            rng.choice(np.flatnonzero(y_arr == category), size=min(n_per_category, (y_arr == category).sum()), replace=False)
            for category in categories
        ])
        rng.shuffle(sample_indices)
        return x_arr[sample_indices], y_arr[sample_indices]

    x_sample, _, y_sample, _ = train_test_split(
        x_arr, y_arr, train_size=n_sample, stratify=y_arr, random_state=random_state,
    )
    return x_sample, y_sample


def classification(df: pd.DataFrame, threshold_alg: ThresholdAlgorithm, selection_pool_size: float = 1,
                   sample_distribution: Optional[Literal["initial", "equal"]] = "initial",
                   random_state: Optional[int] = None,
                   **kwargs) -> tuple[list[bool], float | dict[str, float]]:
    """
    Returns a boolean array with the results of the classification based on the threshold, and the threshold(s)
    that were selected. If the resulting value is True/1 means that the score is below or equal to the threshold,
    and therefore the point is accepted.

    Args:
        df: a DataFrame with a "scores" column (uncertainty scores) and a "labels" column (target labels;
          1 means the point should be accepted, 0 the point should be rejected). A "db_ids" column (one
          database identifier per row) is required for the database-specific algorithms.
        threshold_alg: the algorithm used to select the threshold.
        selection_pool_size: how much uncertainties to keep for threshold selection. A float in (0, 1] is a
          fraction of the full set; an int > 1 is an absolute number of points.
        sample_distribution: passed to the subsampling used for the threshold selection pool. "initial" (default)
          preserves the correct/incorrect ratio of the full set, "equal" draws an equal number of points per label.
        random_state: seed for the subsampling of the threshold selection pool. None (default) => non-reproducible.

    Returns:
        (accepted, threshold): `accepted` is the boolean accept/reject decision per point. `threshold` is the
        single scalar threshold that was selected, except for the database-specific algorithms, where it is a
        dict mapping each db_id to its own threshold.
    """
    scores = df["scores"].to_numpy()
    labels = df["labels"].to_numpy(dtype=bool)
    db_ids: Optional[list[str]] = df["db_ids"].tolist() if "db_ids" in df.columns else None

    if selection_pool_size != 1:
        if threshold_alg in ("database_specific_roc_based_threshold", "database_specific_f_0_5_score_based_threshold"):
            raise ValueError(f"selection_pool_size is not supported for {threshold_alg}!")
        pool_scores, pool_labels = _subsampling(
            scores, labels, selection_pool_size, sample_distribution=sample_distribution, random_state=random_state,
        )
    else:
        pool_scores, pool_labels = scores, labels

    match threshold_alg:
        case "database_specific_roc_based_threshold":
            if db_ids is None:
                raise ValueError("db_ids is required for the 'database_specific_threshold' algorithm")
            dbs = np.asarray(db_ids)

            thresholds_by_db = database_specific_roc_based_threshold_selection(scores, labels, dbs)
            thresholds = np.array([thresholds_by_db[str(db)] for db in dbs])
            return (scores <= thresholds).tolist(), thresholds_by_db
        case "database_specific_f_0_5_score_based_threshold":
            if db_ids is None:
                raise ValueError("db_ids is required for the 'database_specific_f_0_5_score_based_threshold' algorithm")
            dbs = np.asarray(db_ids)

            thresholds_by_db = database_specific_f_0_5_score_threshold_selection(scores, labels, dbs)
            thresholds = np.array([thresholds_by_db[str(db)] for db in dbs])
            return (scores <= thresholds).tolist(), thresholds_by_db
        case "accuracy_restriction_based_threshold":
            try:
                target_accuracy = kwargs["target_accuracy"]
            except KeyError:
                raise ValueError("target_accuracy is required for the 'accuracy_restriction_based_threshold' algorithm")
            threshold = accuracy_restriction_based_threshold_selection(pool_scores, pool_labels, target_accuracy=target_accuracy)
            return (scores <= threshold).tolist(), threshold
        case "roc_based_threshold":
            threshold = roc_based_threshold_selection(pool_scores, pool_labels)
            return (scores <= threshold).tolist(), threshold
        case "f_1_score_based_threshold":
            threshold = f_score_based_threshold_selection(pool_scores, pool_labels, beta=1.0)
            return (scores <= threshold).tolist(), threshold
        case "f_0_5_score_based_threshold":
            threshold = f_score_based_threshold_selection(pool_scores, pool_labels, beta=0.5)
            return (scores <= threshold).tolist(), threshold
        case _:
            raise ValueError(f"Unknown threshold_alg: {threshold_alg}")

