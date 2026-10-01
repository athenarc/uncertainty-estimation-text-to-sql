"""
Computes the uncertainty scores of the uncertainty estimation methods of the paper for every (model, dataset)
experiment of the experiments file and saves them, with the metrics that evaluate them, in an excel file per
experiment (plus the composite datasets, see composite_datasets.py).

The methods and the run each one is computed from:
  - default run (processed generation run): logit-based mean / max-word / schema-linked, and - for runs with sampled
    generations - predictive entropy, execution entropy, global execution entropy and consistency.
  - vanilla / CoT verbalized runs (processed generation runs): vanilla verbalized and CoT verbalized.
  - p_true / self_probing runs (verification runs of a verifier model over the SQL of the default run): P(True) and
    self-probing. An existing verification run is reused; otherwise the verifier model is run (needs GPUs and vLLM).
"""
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from pipeline import config
from pipeline.common import (
    generation_indices,
    get_generation,
    load_run_results,
    log_table,
    existing_verification_run,
    generated_verification_run,
    processed_run_path,
    resolve_db_path,
    resolve_prompt_for_run,
    selected_generation_index,
)
from pipeline.composite_datasets import build_composite_datasets
from pipeline.overall_metrics import build_overall_metrics_sheet
from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC
from text_to_sql.prompt_templates.verification_task.SelfProbingPrompt import SelfProbingPrompt
from text_to_sql.sqlite_db import DatabaseSqlite
from uncertainty_methods.consistency.consistency_based_uncertainty import calculate_consistency_based_uncertainty
from uncertainty_methods.logit_based.execution_entropy_uncertainty import calculate_execution_entropy_from_logprobs
from uncertainty_methods.logit_based.logit_based_uncertainty import (
    AggregationMethod,
    ConsideredTokens,
    calculate_multiple_outputs_logit_based_uncertainty,
    calculate_single_output_logit_based_uncertainty,
)
from uncertainty_methods.logit_based.p_true_uncertainty import calculate_p_true_uncertainty
from uncertainty_methods.verbalized.self_probing_uncertainty import calculate_self_probing_uncertainty
from uncertainty_methods.verbalized.verbalized_based_uncertainty import calculate_verbalized_uncertainty


# =============================================================================================
# Loading
# =============================================================================================

def _load_processed_run(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads a processed generation run (see the process stage (pipeline/stages/process.py)). Uncertainty scores are only meaningful over
    a run that has its parsed predicted_sql_i, its selected predicted_sql and its execution accuracy.
    """
    results_df, config_df = load_run_results(file_path)
    missing = [c for c in ("predicted_sql_0", "predicted_sql", "exec") if c not in results_df.columns]
    if len(generation_indices(results_df)) > 1 and "predicted_sql_index" not in results_df.columns:
        missing.append("predicted_sql_index")
    if missing:
        raise ValueError(f"'{file_path.name}' has not been processed (missing columns {missing}).")

    # Remap server db_paths to local paths if needed.
    results_df["db_path"] = results_df["db_path"].apply(resolve_db_path)
    return results_df, config_df


# =============================================================================================
# Logit-based uncertainty (single-output variations, predictive entropy, execution entropy)
# =============================================================================================

def _execution_entropy_pair_for_row(row: pd.Series, n_generations: int) -> tuple[Optional[float], Optional[float]]:
    """
    Computes execution_entropy and global_execution_entropy together from a single execution pass over the row's
    candidates: both derive from the same calculate_execution_entropy_from_logprobs call (execution_entropy is
    the already-selected candidate's uncertainty score U(s_i); global_execution_entropy is H_exec(Q), the
    whole-question entropy over execution-result clusters shared by every candidate) - fusing them avoids
    executing every candidate against its db twice. Needs >= 2 parsed candidates and a resolvable db_path.

    :return: (execution_entropy, global_execution_entropy), either being None if not computable (e.g. the
        selected candidate isn't among the parsed ones, or execution/clustering fails).
    """

    # Get the SQLs and their logprobs, keeping track of each candidate's original generation index so the
    # selected candidate's position in this (possibly filtered) list can be recovered afterwards.
    indices, sqls, logprobs = [], [], []
    for i in range(n_generations):
        sql = row.get(f"predicted_sql_{i}")
        log_probs = row.get(f"log_probabilities_{i}")
        if pd.notna(sql) and isinstance(log_probs, list):
            indices.append(i)
            sqls.append(sql)
            logprobs.append(log_probs)

    db_path = row.get("db_path")
    if len(sqls) < 2 or not isinstance(db_path, str) or not Path(db_path).exists():
        return None, None

    selected_idx = selected_generation_index(row, n_generations)
    position = indices.index(selected_idx) if selected_idx in indices else None

    try:
        db = DatabaseSqlite(db_path)
        uncertainty_scores, cluster_info = calculate_execution_entropy_from_logprobs(sqls, logprobs, db)
        execution_entropy = uncertainty_scores[position] if position is not None else None
        # global_entropy is the same value in every cluster - any entry's value is H_exec(Q).
        global_execution_entropy = next(iter(cluster_info.values()))["global_entropy"] if cluster_info else None
        return execution_entropy, global_execution_entropy
    except Exception:
        return None, None


def _calculate_logit_based_uncertainty_results(results_df: pd.DataFrame, prompt: PromptABC) -> dict[str, list]:
    """
    Calculates the uncertainties of the logit-based methods and returns {column name: values}. The run must contain,
    for every generation i, generated_output_i, generated_tokens_i and log_probabilities_i.

    Predictive entropy, execution entropy and global execution entropy compare multiple candidates against each
    other, so they are only computed for runs with sampled generations.
    """
    n_generations = len(generation_indices(results_df))
    uncertainties = {}

    logger.info("Computing the single-logit-based uncertainties...")

    # Get the required information per row for the single logit-based calculations
    generations_info = []
    for _, row in results_df.iterrows():
        selected_idx = selected_generation_index(row, n_generations)
        generations_info.append(get_generation(row, selected_idx) if selected_idx is not None else None)

    for considered_tokens, aggregation in config.SINGLE_LOGIT_BASED_VARIATIONS:
        variation_name = f"{considered_tokens}_{aggregation}"
        uncertainty_method_results = []
        for generation in tqdm(generations_info, desc=f"  logit_based_{variation_name}"):
            if generation is None:
                uncertainty_method_results.append(None)
                continue
            uncertainty_method_results.append(calculate_single_output_logit_based_uncertainty(
                result=generation, prompt=prompt,
                considered_tokens=ConsideredTokens(considered_tokens), method=AggregationMethod(aggregation),
            ))
        uncertainties[f"logit_based_{variation_name}"] = uncertainty_method_results

    if n_generations > 1:
        logger.info("Computing the entropy-based uncertainties...")

        entropy_results = []
        for _, row in tqdm(results_df.iterrows(), desc="  entropy_based"):
            generations = [g for i in range(n_generations) if (g := get_generation(row, i)) is not None]
            entropy_results.append(
                calculate_multiple_outputs_logit_based_uncertainty(generations, prompt) if generations else None
            )
        uncertainties["entropy"] = entropy_results

        logger.info("Computing the execution_entropy/global_execution_entropy uncertainties...")

        # Each row executes every candidate against its db: logging is disabled so the failed executions don't flood
        # the console.
        pair_results = []
        logger.disable("")
        try:
            for row_idx in tqdm(range(len(results_df)), desc="  execution_entropy"):
                pair_results.append(_execution_entropy_pair_for_row(results_df.iloc[row_idx], n_generations))
        finally:
            logger.enable("")
        uncertainties["execution_entropy"] = [v[0] for v in pair_results]
        uncertainties["global_execution_entropy"] = [v[1] for v in pair_results]

    return uncertainties


# =============================================================================================
# Consistency-based uncertainty
# =============================================================================================

def _consistency_for_row(row: pd.Series, n_generations: int) -> Optional[float]:
    """
    Consistency-based uncertainty for one row: 1 - average similarity between the self-consistency majority-voted
    candidate (the anchor) and the other sampled candidates, based on execution-result equivalence.
    """
    indices, sqls = [], []
    for i in range(n_generations):
        sql = row.get(f"predicted_sql_{i}")
        if pd.notna(sql):
            indices.append(i)
            sqls.append(sql)

    db_path = row.get("db_path")
    if len(sqls) < 2 or not isinstance(db_path, str) or not Path(db_path).exists():
        return None

    selected_idx = selected_generation_index(row, n_generations)
    if selected_idx in indices:
        position = indices.index(selected_idx)
        initial_prediction = sqls[position]
        variations_sqls = sqls[:position] + sqls[position + 1:]
    else:
        return None

    try:
        return calculate_consistency_based_uncertainty(initial_prediction, variations_sqls, db_path)
    except Exception:
        return None


def _calculate_consistency_based_uncertainty_results(results_df: pd.DataFrame) -> dict[str, list]:
    """
    Calculates the consistency-based uncertainty of each row. Only meaningful for runs with sampled generations
    (requires at least 2 candidates per row to compare).
    """
    logger.info("Computing the consistency-based uncertainties...")

    n_generations = len(generation_indices(results_df))

    consistency_results = []
    logger.disable("")
    try:
        for row_idx in tqdm(range(len(results_df)), desc="  consistency_based"):
            consistency_results.append(_consistency_for_row(results_df.iloc[row_idx], n_generations))
    finally:
        logger.enable("")

    return {"consistency": consistency_results}


# =============================================================================================
# Verbalized uncertainty (vanilla and CoT - both read a verbalized confidence from the model's own
# generation, so share the same per-row logic below)
# =============================================================================================

def _verbalized_uncertainty_results(results_df: pd.DataFrame, config_df: pd.DataFrame, label: str) -> list:
    """Computes the verbalized uncertainty of each row of a verbalized run, using that run's own selected generation."""
    prompt = resolve_prompt_for_run(config_df)
    n_generations = len(generation_indices(results_df))

    logger.info(f"Computing the {label} uncertainties...")

    uncertainty_results = []
    for _, row in tqdm(results_df.iterrows(), desc=f"  {label}"):
        selected_idx = selected_generation_index(row, n_generations)
        gen = get_generation(row, selected_idx) if selected_idx is not None else None
        if gen is None:
            uncertainty_results.append(None)
            continue
        predicted_sql = row.get(f"predicted_sql_{selected_idx}")
        uncertainty_results.append(calculate_verbalized_uncertainty(gen, prompt, predicted_sql))

    return uncertainty_results


# =============================================================================================
# External-model methods: p_true and self-probing. A verifier model judges the SQL the default run selected; its
# response is the verification run the scores are computed from.
# =============================================================================================

def _obtain_verification_run(default_model_run: Path, prompt_name: str, raw_dir: Path) -> Path:
    """
    The verification run of the default run for `prompt_name` (p_true_unique or self_probing_unique): the run
    that already exists - the one registered in the experiments file, or one generated by an earlier pipeline run - or else
    the verifier model is run on the SQL the default run selected (needs GPUs and vLLM) and the run is saved for reuse.
    """
    existing = existing_verification_run(default_model_run.name, prompt_name, raw_dir)
    if existing is not None:
        logger.info(f"Reusing the {prompt_name} verification run {existing.name}.")
        return existing

    # Imported here: running the verifier needs vLLM, which is not needed when the verification runs already exist.
    from inference import run_model

    logger.info(f"No {prompt_name} verification run of {default_model_run.name} found - running the verifier model...")
    return run_model.run_verification(
        default_model_run, prompt_name, generated_verification_run(default_model_run.name, prompt_name, raw_dir),
    )


class _TokenLogprob:
    """The decoded token and logprob of a candidate token of a generation step, as calculate_p_true_uncertainty
    expects them."""

    def __init__(self, decoded_token: str, logprob: float):
        self.decoded_token = decoded_token
        self.logprob = logprob


def _p_true_uncertainty_for_row(row: pd.Series, n_steps: int = 3) -> Optional[float]:
    """
    Adapts a top_tokens_probs_0 cell (a list of {"top_tokens": [...], "top_log_probs": [...]} dicts, one per
    generated step) into the list[dict[int, <token, logprob>]] shape calculate_p_true_uncertainty expects.
    """
    if pd.isna(row.get("predicted_sql")):
        return None

    raw = row.get("top_tokens_probs_0")
    if not isinstance(raw, list) or not raw:
        return None

    steps = []
    for step in raw[:n_steps]:
        tokens = step.get("top_tokens", [])
        logprobs = step.get("top_log_probs", [])
        steps.append({i: _TokenLogprob(tok, lp) for i, (tok, lp) in enumerate(zip(tokens, logprobs))})
    if not steps:
        return None

    return calculate_p_true_uncertainty(steps)


def _calculate_p_true_uncertainty_results(model_run: Path) -> list:
    """Loads the p_true verification run and computes the p_true uncertainty of each row."""
    results_df, _ = load_run_results(model_run)

    logger.info("Computing the p_true uncertainties...")

    return [_p_true_uncertainty_for_row(row) for _, row in tqdm(results_df.iterrows(), desc="  p_true")]


def _self_probing_uncertainty_for_row(row: pd.Series, prompt: SelfProbingPrompt) -> Optional[float]:
    """
    self_probing files carry the verifier model's raw response in verification_output_0 (not generated_output_0 -
    there's no SQL generation happening here) and the SQL being probed in a plain predicted_sql column (not
    predicted_sql_0 - one row = one verification, no candidate index).
    """
    response = row.get("verification_output_0")
    predicted_sql = row.get("predicted_sql")
    if pd.isna(response):
        return None
    return calculate_self_probing_uncertainty(str(response), prompt, predicted_sql)


def _calculate_self_probing_uncertainty_results(model_run: Path) -> list:
    """Loads the self_probing verification run and computes the self-probing uncertainty of each row."""
    results_df, _ = load_run_results(model_run)
    # Only used to parse the confidence out of the response: without a dataset name the prompt doesn't load the
    # dataset to pre-compute the schema of its databases.
    prompt = SelfProbingPrompt()

    logger.info("Computing the self_probing uncertainties...")

    return [
        _self_probing_uncertainty_for_row(row, prompt)
        for _, row in tqdm(results_df.iterrows(), desc="  self_probing")
    ]


# =============================================================================================
# Output sheets
# =============================================================================================

def _prediction_info_for_run(
    results_df: pd.DataFrame, run_name: str, role: str, expected_n_rows: int,
) -> dict[str, pd.Series]:
    """
    A run's own predicted_sql/exec/exec_error columns (see the process stage (pipeline/stages/process.py)), each prefixed with its role
    - a run can select a different generation, and so a different predicted_sql, than another run over the same
    questions.

    Raises if this run's row count doesn't match expected_n_rows (the default run's): the returned columns are pandas
    Series that get assigned into the shared output DataFrame by index, so a length mismatch would otherwise silently
    misalign rows (NaN-fill) instead of raising.
    """
    if len(results_df) != expected_n_rows:
        raise ValueError(
            f"{role} run '{run_name}' has {len(results_df)} row(s), expected {expected_n_rows} (from the default run) "
            "- the runs must cover the same questions in the same order."
        )
    return {
        f"{role}_predicted_sql": results_df.get("predicted_sql"),
        f"{role}_exec": results_df.get("exec"),
        f"{role}_exec_error": results_df.get("exec_error"),
    }


def _build_files_used_sheet(runs: dict[str, Optional[Path]], n_rows: int) -> pd.DataFrame:
    """One row per role, recording which file (if any) was used for it and whether it was provided."""
    return pd.DataFrame([
        {
            "role": role,
            "path": path.name if path else None,
            "status": "found" if path else "not_provided",
            "n_rows": n_rows if path else None,
        }
        for role, path in runs.items()
    ])


def _log_none_report(uncertainties: dict[str, list], file_name: str) -> None:
    """Logs how many rows came out as None for each uncertainty method, so missing coverage is visible at a glance."""
    if not uncertainties:
        return
    rows = []
    for column, values in uncertainties.items():
        n_none = sum(1 for v in values if v is None or pd.isna(v))
        rows.append([column, str(n_none), str(len(values)), f"{n_none / len(values):.1%}" if values else "n/a"])
    log_table(f"[{file_name}] None report:", ["Uncertainty method", "None", "Total", "None %"], rows)


# =============================================================================================
# One experiment
# =============================================================================================

def calculate_uncertainty_results(
    default_model_run: Path,
    vanilla_verbalized_model_run: Optional[Path],
    cot_verbalized_model_run: Optional[Path],
    output_path: Path,
    raw_dir: Path = config.RAW_RUNS_DIR,
) -> pd.DataFrame:
    """
    Computes the uncertainty scores of the methods of one (model, dataset) experiment and saves them in output_path.

    Args:
        default_model_run: The processed generation run the logit-based, entropy and consistency methods are
            computed from.
        vanilla_verbalized_model_run: The processed vanilla verbalized run (vanilla verbalized method), if any.
        cot_verbalized_model_run: The processed CoT verbalized run (CoT verbalized method), if any.
        output_path: Where the excel file with the "results", "overall_metrics" and "files_used" sheets is saved.
        raw_dir: Where the verification runs of P(True) and self-probing are looked for (and saved, if the verifier has
            to be run, see _obtain_verification_run).

    Returns:
        The DataFrame of the "results" sheet: the questions, each run's own predicted_sql/exec/exec_error and the
        score of every method.
    """
    default_df, default_config = _load_processed_run(default_model_run)
    n_rows = len(default_df)

    # entropy/execution entropy/global execution entropy/consistency need >= 2 sampled candidates per row to compare
    # against each other, so none of them is applicable to a greedy (single-generation) run.
    is_greedy_run = len(generation_indices(default_df)) <= 1
    if is_greedy_run:
        logger.info("Greedy run (single generation) - entropy, execution-entropy and consistency are not applicable.")

    uncertainties: dict[str, list] = {}
    uncertainties.update(_calculate_logit_based_uncertainty_results(default_df, resolve_prompt_for_run(default_config)))
    if not is_greedy_run:
        uncertainties.update(_calculate_consistency_based_uncertainty_results(default_df))

    # Output DataFrame: question/db_path/sql_query (sql_query - the gold SQL - lets composite datasets dedup on
    # question+sql_query, since the same question text can have different gold SQL across datasets), each run's own
    # predicted_sql/exec/exec_error (tagged with that run's role, since a run can select a different generation than
    # another), and every uncertainty.
    question_type_columns = [col for col in ("is_ambiguous", "unanswerable") if col in default_df.columns]
    output_df = default_df[["question", "db_path", "sql_query", *question_type_columns]].copy()
    for column, values in _prediction_info_for_run(default_df, default_model_run.name, "default", n_rows).items():
        output_df[column] = values

    for role, column, model_run in (
        ("vanilla", "vanilla_verbalized", vanilla_verbalized_model_run),
        ("cot", "cot_verbalized", cot_verbalized_model_run),
    ):
        if model_run is None:
            logger.info(f"No {role} verbalized run provided - skipping that uncertainty method.")
            continue
        # The vanilla run is usually the default run itself - not loaded twice.
        if model_run == default_model_run:
            run_df, run_config = default_df, default_config
        else:
            run_df, run_config = _load_processed_run(model_run)
        uncertainties[column] = _verbalized_uncertainty_results(run_df, run_config, column)
        for info_column, values in _prediction_info_for_run(run_df, model_run.name, role, n_rows).items():
            output_df[info_column] = values

    verification_runs: dict[str, Path] = {}
    for column, prompt_name, calculate in (
        ("p_true", "p_true_unique", _calculate_p_true_uncertainty_results),
        ("self_probing", "self_probing_unique", _calculate_self_probing_uncertainty_results),
    ):
        verification_runs[column] = _obtain_verification_run(default_model_run, prompt_name, raw_dir)
        uncertainties[column] = calculate(verification_runs[column])

    # Each uncertainty column must line up 1:1 with the rows assembled above.
    for column, values in uncertainties.items():
        if len(values) != n_rows:
            raise ValueError(
                f"Uncertainty column '{column}' has {len(values)} value(s), expected {n_rows} (the run's row "
                f"count) - refusing to write a misaligned uncertainties file."
            )
        output_df[column] = values

    overall_metrics_df = build_overall_metrics_sheet(output_df, list(uncertainties))
    files_used_df = _build_files_used_sheet(
        {
            "default": default_model_run, "vanilla_verbalized": vanilla_verbalized_model_run,
            "cot_verbalized": cot_verbalized_model_run, **verification_runs,
        },
        n_rows,
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        output_df.to_excel(writer, sheet_name="results", index=False)
        overall_metrics_df.to_excel(writer, sheet_name="overall_metrics", index=False)
        files_used_df.to_excel(writer, sheet_name="files_used", index=False)

    _log_none_report(uncertainties, default_model_run.name)

    return output_df


# =============================================================================================
# Stage entry point
# =============================================================================================

def _run_path(experiment: dict, role: str, processed_dir: Path, raw_dir: Path) -> Optional[Path]:
    """The processed generation run (`role`: default, vanilla or cot) of an experiment, or None if the experiment
    doesn't use that run."""
    return processed_run_path(experiment[role], processed_dir, raw_dir) if experiment[role] else None


def run(
    experiments: dict[str, list[dict]],
    raw_dir: Path = config.RAW_RUNS_DIR,
    processed_dir: Path = config.PROCESSED_RUNS_DIR,
    output_dir: Path = config.UNCERTAINTY_RESULTS_DIR,
) -> list[str]:
    """
    Computes the uncertainty results of every experiment of the experiments file (grouped by model and generation pipeline),
    then the composite datasets of each group. Generation runs are read from processed_dir (or raw_dir if they needed
    no processing), verification runs are looked for in raw_dir (and run if missing, see _obtain_verification_run),
    and the excel files are saved in output_dir as `{default run name}_uncertainties.xlsx`.

    Returns:
        The "group/dataset" names of the experiments that failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for group, group_experiments in experiments.items():
        dataset_results: dict[str, pd.DataFrame] = {}
        for experiment in group_experiments:
            name = f"{group}/{experiment['dataset']}"
            logger.info(f"[{name}] Calculating uncertainty results...")

            default_run = _run_path(experiment, "default", processed_dir, raw_dir)
            try:
                dataset_results[experiment["dataset"]] = calculate_uncertainty_results(
                    default_model_run=default_run,
                    vanilla_verbalized_model_run=_run_path(experiment, "vanilla", processed_dir, raw_dir),
                    cot_verbalized_model_run=_run_path(experiment, "cot", processed_dir, raw_dir),
                    output_path=output_dir / f"{default_run.stem}_uncertainties.xlsx",
                    raw_dir=raw_dir,
                )
            except Exception as e:
                failed.append(name)
                logger.error(f"[{name}] Failed: {e}")

        logger.info(f"[{group}] Building composite datasets...")
        try:
            build_composite_datasets(output_dir, group, dataset_results)
        except Exception as e:
            failed.append(f"{group}/composites")
            logger.error(f"[{group}] Failed to build composite datasets: {e}")

    return failed
