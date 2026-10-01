"""
Enriches the run model results excel files with the information the uncertainty methods are computed from:
  - predicted_sql_i is parsed from every generated_output_i,
  - for runs with multiple sampled generations, the self-consistency majority vote selects one of them
    (predicted_sql_index, see compute_self_consistency_majority_voting),
  - the execution accuracy of the selected generation is computed against the gold query (predicted_sql, exec,
    exec_error).

Steps whose columns are already in a run file are not recomputed (unless `reprocess` is set): the run files shipped
with the paper's experiments are already enriched, and recomputing them from scratch can change a few predicted SQLs
(the SQL parsing of the prompts was fixed after they were processed). Only the files that need work are written, to
config.PROCESSED_RUNS_DIR under the same name; the raw run files are never modified.
"""
import contextlib
import io
import random
import time
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from pipeline import config
from pipeline.common import (
    fix_column_types,
    generation_indices,
    is_n_results_run,
    load_run_results,
    log_table,
    reattach_heavy_columns,
    resolve_db_path,
    resolve_prompt_for_run,
    save_run_results,
    spill_heavy_columns,
)
from metrics.execution_accuracy import ExecutionAccuracy
from text_to_sql.majority_voting import _majority_vote
from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC


# =============================================================================================
# Self-consistency majority voting
# =============================================================================================

def seed_majority_voting() -> None:
    """Seeds the random choice made when none of a row's generations executes. Called once per pipeline run, before
    the first file is processed: the choices of a file depend on the order the files are processed in."""
    random.seed(config.MAJORITY_VOTING_SEED)


def compute_self_consistency_majority_voting(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get as input a dataframe with multiple sampled generations per question (generated_output_0 .. generated_output_{n-1})
    and compute the majority-voted SQL for each row based on execution-result equivalence.

    results_df: DataFrame containing the results of the model run. The following columns are expected:
        - db_path: The path to the database file.
        - predicted_sql_i: The generated SQL query for the i-th generation.

    Returns:
        The DataFrame with the additional columns:
        - predicted_sql_index: The majority-voted SQL index for each row.
        - majority_vote_count: The number of generations that agreed with the majority-voted SQL.
        - n_clusters: The total number of clusters formed based on execution-result equivalence.
        - n_valid_generations: The number of generations that executed successfully.
    """
    results_df = fix_column_types(results_df)

    # Remap server db_paths to local paths if needed
    results_df["db_path"] = results_df["db_path"].apply(resolve_db_path)

    gen_indices = generation_indices(results_df)
    if not gen_indices:
        raise ValueError("No 'generated_output_i' columns found in the results sheet.")
    logger.info(f"Found {len(gen_indices)} generation column(s): {gen_indices}")

    predicted_sqls_index = []
    majority_counts = []
    n_valids = []
    n_clusters = []

    for row_idx in tqdm(range(len(results_df)), desc="Majority voting", unit="row"):
        row = results_df.iloc[row_idx]
        sqls = [row[f"predicted_sql_{i}"] for i in gen_indices]
        try:
            predicted_sql_index, majority_count, n_cluster_count, n_valid = _majority_vote(sqls, row["db_path"])
        except Exception as e:
            logger.warning(f"Row {row_idx}: majority vote failed: {e}")
            predicted_sql_index, majority_count, n_valid, n_cluster_count = None, 0, 0, 0

        predicted_sqls_index.append(predicted_sql_index)
        majority_counts.append(majority_count)
        n_valids.append(n_valid)
        n_clusters.append(n_cluster_count)

    results_df["predicted_sql_index"] = predicted_sqls_index
    results_df["majority_vote_count"] = majority_counts
    results_df["n_valid_generations"] = n_valids
    results_df["n_clusters"] = n_clusters

    # Rows with a None predicted_sql_index had an exception or a memory limit error during the calculation.
    logger.info(f"Majority voting completed. There were {sum(1 for idx in predicted_sqls_index if idx is None)} rows "
                f"with None predicted_sql_index (majority voting failed).")

    return results_df


# =============================================================================================
# Processing of a run file
# =============================================================================================

def _predicted_sqls_exist(results_df: pd.DataFrame) -> bool:
    """Returns True if predicted_sql_i has already been parsed for every generation column (generated_output_i)."""
    gen_indices = generation_indices(results_df)
    if not gen_indices:
        raise ValueError("No 'generated_output_i' columns found in the results sheet.")

    return all(
        f"predicted_sql_{i}" in results_df.columns and results_df[f"predicted_sql_{i}"].notna().any()
        for i in gen_indices
    )


def _majority_voting_exists(results_df: pd.DataFrame) -> bool:
    """Returns True if majority voting has already been computed. Checks for at least one non-NaN
    predicted_sql_index rather than all rows: a row can legitimately end up without one (all its generations failed
    to parse/execute, so there was nothing to vote on)."""
    return "predicted_sql_index" in results_df.columns and results_df["predicted_sql_index"].notna().any()


def _execution_accuracy_exists(results_df: pd.DataFrame) -> bool:
    """Returns True if execution accuracy has already been computed for every row."""
    return "exec" in results_df.columns and results_df["exec"].notna().all()


def _parse_predicted_sqls(results_df: pd.DataFrame, prompt: PromptABC) -> pd.DataFrame:
    """Parses predicted_sql_i from generated_output_i for every generation column of the run."""
    results_df = fix_column_types(results_df)

    gen_indices = generation_indices(results_df)
    if not gen_indices:
        raise ValueError("No 'generated_output_i' columns found in the results sheet.")

    for i in gen_indices:
        results_df[f"predicted_sql_{i}"] = results_df[f"generated_output_{i}"].apply(
            lambda x: prompt.get_predicted_sql(str(x)) if pd.notna(x) else None
        )

    return results_df


def _selected_predicted_sql(results_df: pd.DataFrame) -> pd.Series:
    """
    Picks, for each row, the predicted_sql of the selected generation: predicted_sql_0 for greedy runs (a single
    generation), or predicted_sql_{predicted_sql_index} (the majority-voted generation) for n_results runs.
    """
    if not is_n_results_run(results_df):
        return results_df["predicted_sql_0"]

    return results_df.apply(
        lambda row: row[f"predicted_sql_{int(row['predicted_sql_index'])}"]
        if pd.notna(row["predicted_sql_index"]) else None,
        axis=1,
    )


def _compute_selected_execution_accuracy(results_df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """
    Computes the execution accuracy of the selected generation of each row against the gold sql_query, and adds the
    predicted_sql/exec/exec_error columns to the dataframe.
    """
    results_df["predicted_sql"] = _selected_predicted_sql(results_df)
    results_df["db_path"] = results_df["db_path"].apply(resolve_db_path)

    execs, exec_errors = [], []
    for row_idx in tqdm(range(len(results_df)), desc=f"[{file_name}] Execution accuracy", unit="row"):
        row = results_df.iloc[row_idx]
        ea = ExecutionAccuracy()

        # update() prints its own per-call tqdm bar and logs on every failed query; since this runs once per row
        # that would otherwise flood the console. redirect_std* catches the tqdm bar, but loguru's default sink is
        # bound directly to the real stderr at import time and ignores sys.stderr reassignment, so logging needs
        # logger.disable() instead. disable("") disables every module (the root), not just "metrics".
        logger.disable("")
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ea.update(
                    preds=[row["predicted_sql"]],
                    targets=[row["sql_query"]],
                    db_paths=[row["db_path"]],
                )
                result = ea.compute(aggregated=False)[0]
        finally:
            logger.enable("")

        execs.append(result["exec"])
        exec_errors.append(result.get("exec_error"))

    results_df["exec"] = execs
    results_df["exec_error"] = exec_errors

    return results_df


def _n_rows_without_prediction(results_df: pd.DataFrame) -> int:
    """Rows with no usable prediction: no majority vote for n_results runs, or SQL parsing failed for greedy runs."""
    column = "predicted_sql_index" if is_n_results_run(results_df) else "predicted_sql_0"
    return int(results_df[column].isna().sum())


def _run_summary(file_name: str, results_df: pd.DataFrame) -> dict:
    """The row count and correct/missing/error counts of a processed run, for the stage's summary table."""
    n_rows = len(results_df)
    return {
        "file_name": file_name,
        "n_rows": n_rows,
        "n_correct": int(results_df["exec"].sum()) if n_rows else 0,
        "n_missing_predicted_sql": _n_rows_without_prediction(results_df) if n_rows else 0,
        "n_exec_errors": int(results_df["exec_error"].notna().sum()) if n_rows else 0,
    }


def process_run_model_results(raw_path: Path, processed_path: Path, reprocess: bool = False) -> dict:
    """
    Enriches the run model results file raw_path with whatever of the parsed SQLs, the majority vote and the execution
    accuracy it is missing, and writes the result to processed_path. A file that is already complete is not written.

    Args:
        raw_path: The run file.
        processed_path: Where the enriched file is written.
        reprocess: Recompute every step even if its columns are already in the file.

    Returns:
        A dict with the file's row count and its correct/missing/error counts, for the stage's summary table.
    """
    start_time = time.perf_counter()
    file_name = raw_path.name

    results_df, config_df = load_run_results(raw_path)
    logger.info(f"[{file_name}] Loaded {len(results_df)} row(s).")

    needs_parsing = reprocess or not _predicted_sqls_exist(results_df)
    needs_majority_voting = is_n_results_run(results_df) and (reprocess or not _majority_voting_exists(results_df))
    # A new majority vote can select different generations, so the execution accuracy of the selection is recomputed.
    needs_execution_accuracy = reprocess or not _execution_accuracy_exists(results_df) or needs_majority_voting
    if not (needs_parsing or needs_majority_voting or needs_execution_accuracy):
        logger.info(f"[{file_name}] Already processed (parsed SQLs, majority vote and execution accuracy present).")
        return _run_summary(file_name, results_df)

    # The heavy per-generation columns (top_tokens_probs_i/log_probabilities_i/generated_tokens_i - can be multiple
    # GB on a 10-generation file) are not read by anything below, so they are spilled out of memory for the (slow)
    # majority voting and execution accuracy passes and reattached right before saving.
    results_df, heavy_columns_path = spill_heavy_columns(results_df, raw_path)
    try:
        if needs_parsing:
            logger.info(f"[{file_name}] Parsing the predicted SQL of every generation...")
            results_df = _parse_predicted_sqls(results_df, resolve_prompt_for_run(config_df))

        if needs_majority_voting:
            logger.info(f"[{file_name}] Running self-consistency majority voting over {len(results_df)} row(s)...")
            results_df = compute_self_consistency_majority_voting(results_df)

        if needs_execution_accuracy:
            logger.info(f"[{file_name}] Computing execution accuracy for {len(results_df)} row(s)...")
            results_df = _compute_selected_execution_accuracy(results_df, file_name)

        save_run_results(reattach_heavy_columns(results_df, heavy_columns_path), config_df, processed_path)
    finally:
        if heavy_columns_path is not None:
            Path(heavy_columns_path).unlink(missing_ok=True)

    logger.success(f"[{file_name}] Processed in {time.perf_counter() - start_time:.1f}s.")
    return _run_summary(file_name, results_df)


def _log_summary_table(summaries: list[dict]) -> None:
    """Logs one row per processed file (row count, accuracy and error counts)."""
    if not summaries:
        return
    rows = [
        [
            s["file_name"], str(s["n_rows"]), str(s["n_correct"]),
            f"{s['n_correct'] / s['n_rows']:.1%}" if s["n_rows"] else "n/a",
            str(s["n_missing_predicted_sql"]), str(s["n_exec_errors"]),
        ]
        for s in summaries
    ]
    log_table("Processed runs:", ["File", "Rows", "Correct", "Accuracy", "Missing SQL", "Exec Errors"], rows)


def run(
    file_names: list[str],
    raw_dir: Path = config.RAW_RUNS_DIR,
    processed_dir: Path = config.PROCESSED_RUNS_DIR,
    reprocess: bool = False,
) -> list[str]:
    """
    Processes the given generation run files of raw_dir (see process_run_model_results below) into processed_dir, in the
    given order (the random choices of the majority voting depend on it).

    Returns:
        The names of the files that failed to process.
    """
    seed_majority_voting()

    summaries, failed = [], []
    for i, file_name in enumerate(file_names, start=1):
        logger.info(f"[{i}/{len(file_names)}] Processing '{file_name}'...")
        try:
            summaries.append(
                process_run_model_results(raw_dir / file_name, processed_dir / file_name, reprocess)
            )
        except Exception as e:
            failed.append(file_name)
            logger.error(f"[{i}/{len(file_names)}] Failed for '{file_name}': {e}")

    _log_summary_table(summaries)
    return failed
