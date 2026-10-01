"""
Computes the cross-dataset selective-prediction results: for every uncertainty method, the thresholds selected by a
resampling algorithm on a source dataset ("foreign" thresholds) are applied to every target dataset of the same
experiment group, including the source itself.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from pipeline import config
from pipeline.common import uncertainties_file_paths, uncertainty_method_columns
from pipeline.stages.classification import (
    aggregate_metrics,
    metrics_for_threshold,
    scores_and_labels_for_method,
    select_thresholds,
)


def compute_cross_dataset_results(
    datasets: dict[str, pd.DataFrame],
    algorithms: list[str] = config.CROSS_DATASET_ALGORITHMS,
    random_state: Optional[int] = config.CROSS_DATASET_RANDOM_STATE,
) -> pd.DataFrame:
    """For every (uncertainty method, resampling algorithm, source dataset, target dataset), tune the algorithm's
    thresholds on the source and score them on the target.

    Args:
        datasets: {dataset_name: "results" sheet DataFrame} of one experiment group.
        algorithms: names of the algorithms of config.SAMPLING_ALGORITHMS to evaluate.
        random_state: seed of the sub-sampling (see select_thresholds), so re-running this over the same input files
            reproduces the same numbers.

    Returns:
        One row per (uncertainty_method, algorithm, source_dataset, target_dataset) with the aggregated
        ThresholdMetrics (mean + `_std`, `_min`, `_max` across the algorithm's repeats), `n_repeats` and
        `is_same_dataset`.
    """
    # method -> {dataset_name: (scores, labels)}
    per_method: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for name, df in datasets.items():
        for method in uncertainty_method_columns(df):
            scores_labels = scores_and_labels_for_method(df, method)
            if scores_labels is None:
                continue
            _, labels = scores_labels
            # A dataset with only one label class can't have a meaningful threshold selected or scored against it -
            # skip it as a source/target for this method.
            if np.count_nonzero(labels == 0) == 0 or np.count_nonzero(labels == 1) == 0:
                logger.warning(f"[{method}] dataset={name}: only one label class present - skipping.")
                continue
            per_method.setdefault(method, {})[name] = scores_labels

    rows = []
    for method, by_dataset in per_method.items():
        for algorithm in algorithms:
            spec = config.SAMPLING_ALGORITHMS[algorithm]
            for source, (src_scores, src_labels) in by_dataset.items():
                try:
                    thresholds = select_thresholds(src_scores, src_labels, spec, random_state)
                except ValueError as exc:
                    logger.warning(f"[{method}/{algorithm}] source={source}: {exc}")
                    continue

                for target, (tgt_scores, tgt_labels) in by_dataset.items():
                    runs = [metrics_for_threshold(tgt_scores, tgt_labels, threshold) for threshold in thresholds]
                    rows.append({
                        "uncertainty_method": method,
                        "algorithm": algorithm,
                        "source_dataset": source,
                        "target_dataset": target,
                        "is_same_dataset": source == target,
                        "n_repeats": len(thresholds),
                        **aggregate_metrics(runs),
                    })
    return pd.DataFrame(rows)


def run(
    experiments: dict[str, list[dict]],
    uncertainty_dir: Path = config.UNCERTAINTY_RESULTS_DIR,
    output_dir: Path = config.CROSS_DATASET_RESULTS_DIR,
) -> list[str]:
    """
    Computes the cross-dataset results of every experiment group of config.CROSS_DATASET_GROUPS, saving each as
    `{group}_cross_dataset_classification_results.xlsx` in output_dir.

    Returns:
        The names of the groups that failed or have no usable results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for group, group_experiments in experiments.items():
        if group not in config.CROSS_DATASET_GROUPS:
            continue

        paths = uncertainties_file_paths(group, group_experiments, uncertainty_dir)
        if len(paths) < 2:
            failed.append(group)
            logger.warning(f"[{group}] only {len(paths)} dataset(s) - skipping cross-dataset run.")
            continue

        logger.info(f"[{group}] Computing cross-dataset classification over {sorted(paths)}...")
        datasets = {name: pd.read_excel(path, sheet_name="results") for name, path in paths.items()}
        report_df = compute_cross_dataset_results(datasets)
        if report_df.empty:
            failed.append(group)
            logger.warning(f"[{group}] no usable uncertainty-method columns - skipping.")
            continue

        out_path = output_dir / f"{group}_cross_dataset_classification_results.xlsx"
        report_df.to_excel(out_path, sheet_name="cross_dataset_results", index=False)
        logger.info(f"  -> {out_path.name} ({len(report_df)} rows)")

    return failed
