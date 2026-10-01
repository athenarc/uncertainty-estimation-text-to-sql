"""
Computes selective-prediction metrics using the execution-results-based classifier
(text_to_sql_classifier/execution_results_based_classification.py) as a baseline: a prediction is accepted iff its
query executes without error and returns a non-empty result set - no uncertainty score or threshold involved.

The baseline is computed on the default run of every experiment of the experiments file (the same runs the uncertainty methods
are scored on) and on the ambiqt_plus_spider composite, whose rows are rebuilt from the per-dataset rows exactly as
composite_datasets.py builds them for the uncertainty scores.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from pipeline import config
from pipeline.common import processed_run_path, resolve_db_path
from pipeline.composite_datasets import build_composite_row_sets
from pipeline.stages.classification import (
    _filter_rows_without_predicted_sql,
    _metrics_from_accepted,
    exec_correctness_labels,
)
from metrics.utils.execution_accuracy_calculator import _get_results
from text_to_sql.sqlite_db import DatabaseSqlite
from text_to_sql_classifier.execution_results_based_classification import (
    classification as execution_results_classification,
)

# The columns of a processed run the baseline and the composite datasets need.
_RUN_COLUMNS = {
    "question", "db_path", "sql_query", "predicted_sql", "exec", "exec_error", "is_ambiguous", "unanswerable",
    "empty_results",
}


def query_with_empty_results(row: pd.Series) -> bool:
    """Executes the row's predicted query and returns True if its result set is empty."""
    # A prediction that raised an execution error/timeout has no result set to be empty - skip it (and avoid paying
    # to re-execute a query that is already known to fail).
    exec_error = row.get("exec_error")
    if isinstance(exec_error, str) and exec_error.strip():
        return False

    sql = row.get("predicted_sql")
    if sql is None or (isinstance(sql, float) and math.isnan(sql)) or not str(sql).strip():
        return False

    try:
        results, _ = _get_results(
            sql=str(sql),
            db=DatabaseSqlite(resolve_db_path(str(row["db_path"]))),
            query_type="prediction",
            timeout_seconds=config.QUERY_TIMEOUT_SECONDS,
        )
    except Exception as e:
        # SyntaxError (execution error), TimeoutError, RuntimeError (worker killed) - none of these are an
        # "empty result set".
        logger.debug(f"Could not re-execute prediction to check for empty results ({e}).")
        return False

    return len(results) == 0


def _load_default_run(file_path: Path) -> pd.DataFrame:
    """Loads the columns of a processed default run the baseline needs. If the run has no 'empty_results' column yet
    (the released runs do), it is computed by re-executing the predicted queries. An existing column is reused as it
    is: whether a slow query times out depends on the machine, so re-executing could flip a few rows."""
    results_df = pd.read_excel(file_path, sheet_name="results", usecols=lambda column: column in _RUN_COLUMNS)
    if "empty_results" in results_df.columns:
        return results_df

    results_df["empty_results"] = [
        query_with_empty_results(row)
        for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc=f"[{file_path.name}] Empty results")
    ]
    return results_df


def compute_execution_classification_metrics(results_df: pd.DataFrame) -> dict:
    """Metrics of the execution-based baseline for one dataset's (or composite's) rows."""
    filtered_df = _filter_rows_without_predicted_sql(results_df, "predicted_sql")
    labels = exec_correctness_labels(filtered_df, "exec")
    accepted = np.array(execution_results_classification(filtered_df), dtype=bool)

    return vars(_metrics_from_accepted(accepted, labels))


def run(
    experiments: dict[str, list[dict]],
    processed_dir: Path = config.PROCESSED_RUNS_DIR,
    raw_dir: Path = config.RAW_RUNS_DIR,
    output_dir: Path = config.EXECUTION_CLASSIFICATION_RESULTS_DIR,
) -> list[str]:
    """
    Computes the baseline of every evaluation dataset (config.CLASSIFICATION_DATASETS) of every experiment group and
    saves them in output_dir/execution_results_based_classification_results.xlsx, one row per default run file (or per
    composite, named `{composite}_{group}.xlsx`).

    Returns:
        The "group/dataset" names that failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, failed = [], []
    for group, group_experiments in experiments.items():
        dataset_results: dict[str, pd.DataFrame] = {}
        for experiment in group_experiments:
            name = f"{group}/{experiment['dataset']}"
            logger.info(f"[{name}] Computing execution-results-based classification results...")
            try:
                dataset_results[experiment["dataset"]] = _load_default_run(processed_run_path(experiment["default"], processed_dir, raw_dir))
            except Exception as e:
                failed.append(name)
                logger.error(f"[{name}] Failed: {e}")
                continue
            if experiment["dataset"] in config.CLASSIFICATION_DATASETS:
                rows.append({
                    "file_name": experiment["default"],
                    **compute_execution_classification_metrics(dataset_results[experiment["dataset"]]),
                })

        composites = build_composite_row_sets(dataset_results, [config.AMBIQT_PLUS_SPIDER])
        for name, (combined_df, _sources) in composites.items():
            rows.append({"file_name": f"{name}_{group}.xlsx", **compute_execution_classification_metrics(combined_df)})

    if rows:
        out_path = output_dir / "execution_results_based_classification_results.xlsx"
        pd.DataFrame(rows).to_excel(out_path, sheet_name="execution_classification_results", index=False)
        logger.info(f"-> {out_path.name} ({len(rows)} rows)")

    return failed
