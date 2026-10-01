"""
Computes the selective-prediction metrics of the uncertainty-based classifiers: for every uncertainty method of an
uncertainties file, a threshold is selected with each of the configured algorithms (config.ORACLE_ALGORITHMS /
config.SAMPLING_ALGORITHMS) and the classifier "accept a prediction iff its uncertainty is <= the threshold" is scored.
"""
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from pipeline import config
from pipeline.common import role_of_method, uncertainties_file_paths, uncertainty_method_columns
from text_to_sql_classifier.threshold_based_classification import (
    ThresholdAlgorithm,
    _subsampling,
    accuracy_restriction_based_threshold_selection,
    f_score_based_threshold_selection,
    roc_based_threshold_selection,
)


# =============================================================================================
# Metrics of an accept/reject decision
# =============================================================================================

def _f_beta_score(accuracy: float, coverage: float, beta: float) -> float:
    """Weighted harmonic mean of `accuracy` and `coverage` (F-beta score); beta < 1 weights
    `accuracy` more heavily than `coverage`. NaN (e.g. no queries were accepted, so
    selective_accuracy is undefined) propagates through rather than being treated as 0."""
    if np.isnan(accuracy) or np.isnan(coverage):
        return float("nan")
    denominator = beta ** 2 * accuracy + coverage
    if denominator <= 0:
        return 0.0
    return (1 + beta ** 2) * accuracy * coverage / denominator


@dataclass
class ThresholdMetrics:
    overall_accuracy: float
    selective_accuracy: float
    true_acceptance_rate: float
    false_acceptance_rate: float
    true_rejection_rate: float
    false_rejection_rate: float
    coverage_over_correct_answers: float
    n_total: int
    n_accepted: int
    selective_accuracy_correct_coverage_score: float
    overall_accuracy_coverage_score: float
    selective_accuracy_correct_coverage_0_5_score: float
    overall_accuracy_coverage_0_5_score: float


def _metrics_from_accepted(accepted: np.ndarray, labels: np.ndarray) -> ThresholdMetrics:
    """Given a boolean accept/reject decision per query and its label (1 = incorrect, 0 = correct),
    compute the selective-prediction metrics shared by every algorithm."""
    n_total = len(labels)
    overall_accuracy = float(1 - labels.mean()) if n_total else float("nan")

    n_accepted = int(accepted.sum())
    selective_accuracy = float(1 - labels[accepted].mean()) if n_accepted else float("nan")

    is_correct = labels == 0
    is_incorrect = labels == 1

    true_acceptance_num = (is_correct & accepted).sum()
    true_acceptance_rate = (
        float(true_acceptance_num / n_total) if n_total else float("nan")
    )
    false_acceptance_rate = (
        float((is_incorrect & accepted).sum() / n_total) if n_total else float("nan")
    )
    true_rejection_rate = (
        float((is_incorrect & ~accepted).sum() / n_total) if n_total else float("nan")
    )
    false_rejection_rate = (
        float((is_correct & ~accepted).sum() / n_total) if n_total else float("nan")
    )

    coverage_over_correct_answers = float(true_acceptance_num / is_correct.sum())

    return ThresholdMetrics(
        overall_accuracy=overall_accuracy,
        selective_accuracy=selective_accuracy,
        coverage_over_correct_answers=coverage_over_correct_answers,
        true_acceptance_rate=true_acceptance_rate,
        false_acceptance_rate=false_acceptance_rate,
        true_rejection_rate=true_rejection_rate,
        false_rejection_rate=false_rejection_rate,
        n_total=n_total,
        n_accepted=n_accepted,
        selective_accuracy_correct_coverage_score=_f_beta_score(selective_accuracy, coverage_over_correct_answers, beta=1.0),
        overall_accuracy_coverage_score=_f_beta_score(overall_accuracy, 1, beta=1.0),
        selective_accuracy_correct_coverage_0_5_score=_f_beta_score(selective_accuracy, coverage_over_correct_answers, beta=0.5),
        overall_accuracy_coverage_0_5_score=_f_beta_score(overall_accuracy, 1, beta=0.5)
    )


def metrics_for_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> ThresholdMetrics:
    """Applies one already-chosen `threshold` to a dataset (a point is accepted iff its score is <= the threshold) and
    returns the resulting ThresholdMetrics. `labels`: one binary label per query (1 = incorrect, 0 = correct)."""
    return _metrics_from_accepted(np.asarray(scores, dtype=float) <= threshold, np.asarray(labels))


def aggregate_metrics(runs: list[ThresholdMetrics]) -> dict[str, float]:
    """Averages each ThresholdMetrics field across repeated runs, adding `{field}_std`,
    `{field}_min` and `{field}_max` entries whenever there is more than one run (i.e. the
    algorithm actually subsamples) - the min/max are the actual best/worst repeat's own
    value, for callers that want the real range across repeats rather than a std-derived
    estimate of it."""
    aggregated: dict[str, float] = {}
    for field in fields(ThresholdMetrics):
        values = np.array([getattr(run, field.name) for run in runs], dtype=float)
        aggregated[field.name] = float(np.mean(values))
        if len(runs) > 1:
            aggregated[f"{field.name}_std"] = float(np.std(values))
            aggregated[f"{field.name}_min"] = float(np.min(values))
            aggregated[f"{field.name}_max"] = float(np.max(values))
    return aggregated


# =============================================================================================
# Threshold selection
# =============================================================================================

def select_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    spec: dict,
    random_state: Optional[int] = config.CLASSIFICATION_RANDOM_STATE,
) -> list[float]:
    """Runs the threshold selection algorithm described by `spec` (a config.ORACLE_ALGORITHMS /
    config.SAMPLING_ALGORITHMS entry) over one dataset `n_repeats` times and returns the selected threshold of each
    run.

    Args:
        scores: one uncertainty score per query.
        labels: one binary label per query (1 = incorrect, 0 = correct), aligned by position with `scores`. Converted
            internally to the "True = accept" boolean the selection algorithms expect.
        spec: `threshold_alg`, `selection_pool_size` (how much of the data is kept for the threshold selection: a
            float in (0, 1] is a fraction, an int > 1 an absolute count, 1.0 the full data - an "oracle" threshold),
            and optionally `n_repeats` (how many times the pool is resampled and the threshold reselected),
            `sample_distribution` ("initial" preserves the correct/incorrect ratio of the full data in the pool,
            "equal" draws an equal number per label) and `target_accuracy`.
        random_state: seed of the sub-sampling. Each repeat draws its own distinct subsample (a per-repeat seed is
            derived from this one), so re-running gives the same thresholds. None => non-reproducible.

    Returns:
        A list of length `n_repeats` with the selected thresholds.
    """
    scores = np.asarray(scores, dtype=float)
    labels_bool = ~np.asarray(labels).astype(bool)

    selection_pool_size = spec["selection_pool_size"]
    subsamples = not (isinstance(selection_pool_size, float) and selection_pool_size == 1.0) \
        and selection_pool_size != 1

    seed_rng = np.random.default_rng(random_state) if random_state is not None else None

    thresholds = []
    for _ in range(spec.get("n_repeats", 1)):
        repeat_seed = None if seed_rng is None else int(seed_rng.integers(0, 2**32))

        if subsamples:
            pool_scores, pool_labels = _subsampling(
                scores, labels_bool, selection_pool_size,
                sample_distribution=spec.get("sample_distribution", "initial"), random_state=repeat_seed,
            )
        else:
            pool_scores, pool_labels = scores, labels_bool

        thresholds.append(_select_one(spec, pool_scores, pool_labels))
    return thresholds


def _select_one(spec: dict, pool_scores: np.ndarray, pool_labels: np.ndarray) -> float:
    """The threshold the algorithm of `spec` selects on the given selection pool."""
    threshold_alg: ThresholdAlgorithm = spec["threshold_alg"]
    match threshold_alg:
        case "roc_based_threshold":
            return roc_based_threshold_selection(pool_scores, pool_labels)
        case "f_1_score_based_threshold":
            return f_score_based_threshold_selection(pool_scores, pool_labels, beta=1.0)
        case "f_0_5_score_based_threshold":
            return f_score_based_threshold_selection(pool_scores, pool_labels, beta=0.5)
        case "accuracy_restriction_based_threshold":
            try:
                target_accuracy = spec["target_accuracy"]
            except KeyError:
                raise ValueError("target_accuracy is required for 'accuracy_restriction_based_threshold'")
            return accuracy_restriction_based_threshold_selection(
                pool_scores, pool_labels, target_accuracy=target_accuracy,
            )
        case _:
            raise ValueError(f"Unknown threshold_alg: {threshold_alg}")


def _mean_threshold_metrics(scores: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    """Applies the mean of the per-repeat selected thresholds to the *full* `scores` (not a
    subsample) and computes ThresholdMetrics once from that single accept/reject decision -
    "what if this cheap algorithm always used its average threshold instead of resampling".
    Only called for algorithms with more than one repeat. Unlike the per-repeat runs, this
    is a single deterministic computation - there's no distribution here to take a std across."""
    mean_threshold = float(np.mean(thresholds))
    metrics = metrics_for_threshold(scores, labels, mean_threshold)
    result = {f"mean_threshold_{field.name}": getattr(metrics, field.name) for field in fields(ThresholdMetrics)}
    result["mean_threshold_value"] = mean_threshold
    return result


def compute_classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    spec: dict,
    random_state: Optional[int] = config.CLASSIFICATION_RANDOM_STATE,
) -> tuple[dict[str, float], dict]:
    """Selects the thresholds of the algorithm `spec` on (scores, labels) and returns the metrics of the classifiers
    they define, evaluated on the full dataset (see select_thresholds for the arguments).

    Returns:
        (metrics, extra_fields): ThresholdMetrics, or the mean of each ThresholdMetrics value plus its `_std`, `_min`
        and `_max` in case of multiple repeats. extra_fields carries `n_repeats`, the selected `thresholds` and, for
        algorithms with more than one repeat, a `mean_threshold_{field}` entry per ThresholdMetrics field (the metrics
        of applying the mean of the repeats' own selected thresholds to the full, non-subsampled data - see
        _mean_threshold_metrics) and `mean_threshold_value` (that mean threshold itself).
    """
    thresholds = select_thresholds(scores, labels, spec, random_state)
    metrics = aggregate_metrics([metrics_for_threshold(scores, labels, threshold) for threshold in thresholds])

    extra_fields = {
        "n_repeats": len(thresholds),
        "thresholds": ", ".join(f"{threshold:.6g}" for threshold in thresholds),
    }
    if len(thresholds) > 1:
        extra_fields.update(_mean_threshold_metrics(scores, labels, thresholds))

    return metrics, extra_fields


# =============================================================================================
# Scores and labels of an uncertainties file
# =============================================================================================

def _filter_rows_without_predicted_sql(results_df: pd.DataFrame, predicted_sql_col: str) -> pd.DataFrame:
    """Drops rows where the given run has no predicted_sql (e.g. parsing/majority-voting failed) - there's no
    execution-accuracy label to evaluate an uncertainty method against for these rows, so they are excluded from
    the classification metrics entirely rather than just masked out."""
    return results_df[results_df[predicted_sql_col].notna()]


def exec_correctness_labels(df: pd.DataFrame, exec_col: str) -> np.ndarray:
    """Labels for the classification metrics: 1 = incorrect, 0 = correct. `exec_col` is True if the
    execution was successful (correct), False if it failed (incorrect). If the dataset row is
    ambiguous or unanswerable, the label is set to 1 (incorrect) since there is no correct answer to
    these questions."""
    exec_labels = 1 - df[exec_col].to_numpy(dtype=float)
    if "is_ambiguous" in df.columns:
        no_correct_answer = df["is_ambiguous"].fillna(False).to_numpy(dtype=bool)
    elif "unanswerable" in df.columns:
        no_correct_answer = df["unanswerable"].fillna(False).to_numpy(dtype=bool)
    else:
        no_correct_answer = np.zeros(len(df), dtype=bool)
    return np.where(no_correct_answer, 1.0, exec_labels)


def scores_and_labels_for_method(results_df: pd.DataFrame, method: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """One uncertainty method's scores and its binary labels (1 = incorrect, 0 = correct), aligned by position.
    The method is evaluated against its own role's predicted_sql/exec (see config.UNCERTAINTY_COLUMN_ROLE), and
    ambiguous/unanswerable rows are forced incorrect since there's no correct answer to accept. Returns None if every
    value for this method is NaN."""
    role = role_of_method(method)
    filtered_df = _filter_rows_without_predicted_sql(results_df, f"{role}_predicted_sql")

    method_mask = filtered_df[method].notna().to_numpy()
    if not method_mask.any():
        return None

    scores = filtered_df.loc[method_mask, method].to_numpy(dtype=float)
    labels = exec_correctness_labels(filtered_df, f"{role}_exec")[method_mask]
    return scores, labels


def compute_classification_results(
    results_df: pd.DataFrame,
    algorithms: dict[str, dict],
    random_state: Optional[int] = config.CLASSIFICATION_RANDOM_STATE,
) -> pd.DataFrame:
    """Returns the metrics of every uncertainty method of results_df for every algorithm of `algorithms`.

    results_df: one row per query, as produced by the uncertainty stage's "results" sheet (question,
    db_path, {role}_predicted_sql/{role}_exec/{role}_exec_error columns, and one column per uncertainty method),
    where role corresponds to the default run, the vanilla verbalized run and the cot verbalized run.
    """
    rows = []
    for method in uncertainty_method_columns(results_df):
        scores_labels = scores_and_labels_for_method(results_df, method)
        if scores_labels is None:
            logger.warning(f"All the uncertainties were None. No calculation can be done for the {method}...")
            continue
        scores, labels = scores_labels

        role = role_of_method(method)
        n_none_uncertainties = int(
            _filter_rows_without_predicted_sql(results_df, f"{role}_predicted_sql")[method].isna().sum()
        )

        for algorithm, spec in algorithms.items():
            # An absolute selection pool size (an int > 1) larger than the available points can't be sampled
            pool_size = spec["selection_pool_size"]
            if not isinstance(pool_size, float) and pool_size > len(scores):
                logger.warning(
                    f"  {method}/{algorithm}: selection pool size {pool_size} is larger than the "
                    f"{len(scores)} available points - skipping."
                )
                continue

            metrics, extra_fields = compute_classification_metrics(scores, labels, spec, random_state)
            rows.append({
                "uncertainty_method": method,
                "algorithm": algorithm,
                **metrics,
                **extra_fields,
                "n_none_uncertainties": n_none_uncertainties,
            })

    return pd.DataFrame(rows)


# =============================================================================================
# Stage entry point
# =============================================================================================

def run(
    experiments: dict[str, list[dict]],
    uncertainty_dir: Path = config.UNCERTAINTY_RESULTS_DIR,
    output_dir: Path = config.CLASSIFICATION_RESULTS_DIR,
) -> list[str]:
    """
    Computes the classification results of every evaluation dataset (config.CLASSIFICATION_DATASETS) of every
    experiment group, saving each as `{uncertainties file name}_classification_results.xlsx` in output_dir.

    Returns:
        The names of the uncertainties files that failed or have no usable results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for group, group_experiments in experiments.items():
        algorithms = config.classification_algorithms(group)
        for dataset, path in uncertainties_file_paths(group, group_experiments, uncertainty_dir).items():
            logger.info(f"[{group}/{dataset}] Computing classification results...")
            try:
                report_df = compute_classification_results(pd.read_excel(path, sheet_name="results"), algorithms)
            except Exception as e:
                failed.append(path.name)
                logger.error(f"[{group}/{dataset}] Failed: {e}")
                continue
            if report_df.empty:
                failed.append(path.name)
                logger.warning(f"[{group}/{dataset}] no usable uncertainty-method columns - skipping.")
                continue

            out_path = output_dir / path.name.replace("_uncertainties.xlsx", "_classification_results.xlsx")
            report_df.to_excel(out_path, sheet_name="classification_results", index=False)
            logger.info(f"  -> {out_path.name} ({len(report_df)} rows)")

    return failed
