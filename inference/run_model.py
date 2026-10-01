"""
Step 1 of the reproduction: obtains the inference results. It runs the Text-to-SQL models of the paper (needs GPUs and
vLLM; see requirements-generation.txt) and saves the raw run results in config.RAW_RUNS_DIR as excel files with a
"results" and a "configuration" sheet. This step is separate from the pipeline of pipeline/run.py (step 2), which
starts from these files; the runs of the paper can be downloaded instead of regenerated (see the README).

  - generation runs (`generate`): a model produces the SQL of every question of a dataset, once greedily and once as
    10 sampled generations (the self-consistency pipeline), with the token logprobs the uncertainty methods need.
  - verification runs (`verify`): a verifier model judges the SQL that a generation run selected (p_true and
    self-probing prompts). They read a *processed* generation run, because they need its selected `predicted_sql`:
    process the generation runs first with
    `python -m pipeline --stages process --keep-intermediate`.

    python -m inference.run_model generate [--models MODEL_ID ...] [--datasets DATASET ...]
    python -m inference.run_model verify [--models GROUP ...] [--datasets DATASET ...]

The output file names are `{dataset}_{prompt}_{model}_{greedy|{n}_results}[_from_{prompt}_{model}]_{timestamp}.xlsx`.
After a re-run, the new file names must be registered in pipeline/experiments.json.
"""
import argparse
import importlib
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from pipeline import config
from pipeline.common import instantiate_prompt, processed_run_path, select_experiments

_SAFE_MODEL_IDS = sorted((m.replace(".", "_") for m in config.MODELS), key=len, reverse=True)
_SOURCE_RUN_NAME_RE = re.compile(
    rf"^[^_]+_(?P<prompt>.+)_(?P<model>{'|'.join(re.escape(m) for m in _SAFE_MODEL_IDS)})_(?:greedy|\d+_results)_\d+$"
)


def _rebase_db_path(db_path: str) -> str:
    """Re-anchor a `db_path` captured in a prior run's results file to this checkout's `evaluated_datasets/` tree, so
    a verification run can load a results file generated on a different machine.

    Generation runs snapshot an absolute `db_path` into the results xlsx. Loaded elsewhere, that prefix no longer
    resolves even though the same `evaluated_datasets/` layout exists under a different root - `DatabaseSqlite` then
    raises `ValueError` for every row, which `Model._build_prompts` silently drops, so the model gets called zero
    times and an empty results file is written without any error.
    """
    parts = Path(db_path).parts
    if "evaluated_datasets" not in parts:
        return db_path
    idx = parts.index("evaluated_datasets")
    return str(config.PROJECT_ROOT / Path(*parts[idx:]))


def _exec_columns_for(sql_column: str) -> tuple[Optional[str], Optional[str]]:
    """Map a predicted-SQL column name of a processed run to its matching execution-accuracy columns:
    predicted_sql_{i} -> exec_accuracy_{i} / exec_error_{i}, and the selected predicted_sql -> exec / exec_error.
    Returns (None, None) if `sql_column` doesn't match either convention."""
    if sql_column == "predicted_sql":
        return "exec", "exec_error"
    m = re.match(r"predicted_sql_(\d+)$", sql_column)
    if m:
        return f"exec_accuracy_{m.group(1)}", f"exec_error_{m.group(1)}"
    return None, None


def _default_sql_column(path: Path) -> str:
    """Pick the predicted-SQL column to verify when the caller didn't name one: greedy results files (named with a
    `_greedy_` segment) only ever have a single candidate, `predicted_sql_0`; any other results file is expected to
    carry the majority-voted `predicted_sql` column."""
    return "predicted_sql_0" if "greedy" in path.stem.split("_") else "predicted_sql"


def _source_run_identity(source_file: Path) -> tuple[Optional[str], Optional[str]]:
    """Pulls (prompt_name, safe_model_id) back out of `source_file`'s name, which `run()` itself built. Returns
    (None, None) if it doesn't match that convention."""
    match = _SOURCE_RUN_NAME_RE.match(source_file.stem)
    return (match["prompt"], match["model"]) if match else (None, None)


def _initialize_dataset(dataset_name: str, sql_column: Optional[str] = None):
    """Resolve `dataset_name` into a dataset ready for `run()`.

    `dataset_name` is either:
      - a dataset key of config.DATASETS ("bird", "spider", ...) -> the benchmark is loaded via the matching
        `Dataset.get_data()`.
      - a path to a processed generation run `.xlsx` file -> a verification dataset is built from it instead:
        `question`/`db_path`/gold `query` are read straight from the file, plus a `generated_sql` column taken from
        `sql_column` (the selected `predicted_sql` computed by the process stage (pipeline/stages/process.py); `predicted_sql_0` for
        the greedy candidate). If `sql_column` is `None`, it's picked via `_default_sql_column`.

    Returns:
        (dataset_df, dataset_key, dataset_class_name, source_file):
            - dataset_df: DataFrame with at least question/query/db_path (plus generated_sql for the file-based
              branch).
            - dataset_key: short slug used for prompt selection and the output filename (e.g. "bird").
            - dataset_class_name: e.g. "BirdDataset", written to the `configuration` sheet's `dataset` field.
            - source_file: the source results Path, or None for a live dataset - written to the `configuration`
              sheet as `source_results_file`.
    """
    path = Path(dataset_name)
    if path.suffix.lower() == ".xlsx":
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        results_df = pd.read_excel(path, sheet_name="results")
        if sql_column is None:
            sql_column = _default_sql_column(path)
        if sql_column not in results_df.columns:
            available = [c for c in results_df.columns if c.startswith("predicted_sql")]
            raise ValueError(
                f"Column '{sql_column}' not found in {path.name}. "
                f"Available predicted-SQL columns: {available}"
            )

        dataset_df = pd.DataFrame({
            "question": results_df["question"],
            "query": results_df["sql_query"],
            "db_path": results_df["db_path"].apply(_rebase_db_path),
            "generated_sql": results_df[sql_column],
        })
        for optional_field in ("is_ambiguous", "ambig_type", "unanswerable", "origin_dataset"):
            if optional_field in results_df.columns:
                dataset_df[optional_field] = results_df[optional_field]
        # BIRD's evidence ("hint") is part of the question the SQL was generated for, so the verifier
        # needs it too. Empty hints come back from Excel as NaN - blank them so the prompt doesn't read "nan".
        if "hint" in results_df.columns:
            dataset_df["hint"] = results_df["hint"].fillna("").astype(str)

        # Carry through execution accuracy for `generated_sql` if it was already computed upstream, so a
        # verification run doesn't need to re-execute every query to know whether the SQL it's verifying was correct.
        exec_col, exec_error_col = _exec_columns_for(sql_column)
        if exec_col and exec_col in results_df.columns:
            dataset_df["exec_accuracy"] = results_df[exec_col]
        if exec_error_col and exec_error_col in results_df.columns:
            dataset_df["exec_error"] = results_df[exec_error_col]

        config_df = pd.read_excel(path, sheet_name="configuration")
        dataset_class_name = str(config_df.iloc[0]["dataset"]) if not config_df.empty else "UnknownDataset"
        dataset_key = config.DATASET_CLASS_TO_KEY.get(dataset_class_name, path.name.split("_")[0])

        if dataset_class_name == "BirdDataset" and "hint" not in dataset_df.columns:
            raise ValueError(
                f"{path.name} is a BIRD results file without a 'hint' column - the verification "
                "prompt would be missing BIRD's evidence. Add the hints to the file first."
            )

        return dataset_df, dataset_key, dataset_class_name, path

    try:
        module_name, class_name = config.DATASETS[dataset_name]
    except KeyError:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    dataset_obj = getattr(importlib.import_module(module_name), class_name)()

    return dataset_obj.get_data(), dataset_name, class_name, None


def run(
    dataset_name: str,
    prompt_name: str,
    model_name_id: str,
    n_generations: int = 1,
    logprobs_num: int = 1,
    temperature: float = 1.0,
    model_args: Optional[dict] = None,
    sql_column: Optional[str] = None,
    output_dir: Path = config.RAW_RUNS_DIR,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run `prompt_name` against `model_name_id` and save the results in output_dir (under a name that ends with a
    timestamp), or in output_path if it is given.

    `dataset_name` is either a dataset key ("bird", "spider", ...) for a generation run, or a path to a processed
    generation run `.xlsx` for a verification run (`p_true_unique` / `self_probing_unique`) - see
    `_initialize_dataset`. `sql_column` only matters in the latter case: it picks which column of the source file
    holds the SQL to verify (see `_default_sql_column`).
    """
    # Imported here: the inference code needs vLLM, which the rest of the pipeline doesn't.
    from text_to_sql.inference.text_to_sql_inference import text_to_sql_inference

    dataset, dataset_key, dataset_class_name, source_file = _initialize_dataset(dataset_name, sql_column)
    model_name = config.MODELS[model_name_id]["hf_name"]
    prompt = instantiate_prompt(prompt_name, dataset_key)
    is_verification_prompt = config.PROMPTS[prompt_name][3]
    output_col_prefix = "verification_output" if is_verification_prompt else "generated_output"

    if model_args is None:
        model_args = dict(config.MODELS[model_name_id]["sampling_args"]) if temperature != 0 else {}

    all_generation_results = text_to_sql_inference(
        model_name=model_name,
        model_parameters={},
        prompt_template=prompt,
        dataset=dataset,
        dataset_name=dataset_class_name,
        n=n_generations,
        logprobs_num=logprobs_num,
        temperature=temperature,
        model_args=model_args,
    )

    rows = []
    for generations in all_generation_results:
        main = generations[0]
        row = {
            "model_input": main["model_input"],
            "question": main["question"],
            "sql_query": main["sql_query"],
            "db_path": main["db_path"],
        }
        if "generated_sql" in main:
            row["predicted_sql"] = main["generated_sql"]
        if "exec_accuracy" in main:
            row["exec_accuracy"] = main["exec_accuracy"]
        if "exec_error" in main:
            row["exec_error"] = main["exec_error"]
        if "question_id" in main:
            row["question_id"] = main["question_id"]
        if "hint" in main:
            row["hint"] = main["hint"]
        for i, gen in enumerate(generations):
            row[f"{output_col_prefix}_{i}"] = gen["generated_output"]
            row[f"generated_tokens_{i}"] = gen["generated_tokens"]
            row[f"log_probabilities_{i}"] = gen["token_log_probs"]
            row[f"top_tokens_probs_{i}"] = gen.get("top_tokens_probs")
        for field in ("is_ambiguous", "ambig_type", "unanswerable", "origin_dataset"):
            if field in main:
                row[field] = main[field]
        rows.append(row)

    results_df = pd.DataFrame(rows)

    configuration = pd.DataFrame([{
        "model_name": model_name,
        "prompt_name": prompt.__class__.__name__,
        "dataset": dataset_class_name,
        "n_generations": n_generations,
        "logprobs_num": logprobs_num,
        "temperature": temperature,
        "model_args": str(model_args),
        "source_results_file": str(source_file) if source_file else None,
    }])

    safe_model_id = model_name_id.replace(".", "_")
    results_type = "greedy" if temperature == 0 else f"{n_generations}_results"
    from_suffix = ""
    if source_file is not None:
        source_prompt_name, source_model_id = _source_run_identity(source_file)
        if source_prompt_name:
            from_suffix = f"_from_{source_prompt_name}_{source_model_id}"
    report_path = output_path or (
        output_dir / f"{dataset_key}_{prompt_name}_{safe_model_id}_{results_type}{from_suffix}_{int(time.time())}.xlsx"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="results", index=False)
        configuration.to_excel(writer, sheet_name="configuration", index=False)

    logger.info(f"Results saved to {report_path}")

    return results_df


def generate(models: Optional[list[str]] = None, datasets: Optional[list[str]] = None) -> None:
    """Runs every generation run of config.GENERATION_PLAN (optionally restricted to some model ids / dataset keys) on
    every dataset of config.DATASETS."""
    for model_name_id, prompt_name, decoding_configs in config.GENERATION_PLAN:
        if models and model_name_id not in models:
            continue
        for dataset_name in config.DATASETS:
            if datasets and dataset_name not in datasets:
                continue
            for n_generations, temperature in decoding_configs:
                logger.info(
                    f"Generating: {dataset_name} / {prompt_name} / {model_name_id} "
                    f"(n={n_generations}, temperature={temperature})"
                )
                # An explicit model_args bypasses the per-model sampling defaults in run(), so merge them back in for
                # sampled runs.
                model_args = dict(config.GENERATION_MODEL_ARGS)
                if temperature != 0:
                    model_args = {**config.MODELS[model_name_id]["sampling_args"], **model_args}
                run(
                    dataset_name=dataset_name,
                    prompt_name=prompt_name,
                    model_name_id=model_name_id,
                    n_generations=n_generations,
                    logprobs_num=config.GENERATION_LOGPROBS_NUM,
                    temperature=temperature,
                    model_args=model_args,
                )


def run_verification(processed_run: Path, prompt_name: str, output_path: Path) -> Path:
    """Runs the verification `prompt_name` (a key of config.VERIFICATION_RUN_ARGS) over the SQL that the processed
    generation run selected, judged greedily by config.VERIFIER_MODEL_ID, and saves the run in output_path."""
    run_args = config.VERIFICATION_RUN_ARGS[prompt_name]
    run(
        dataset_name=str(processed_run),
        prompt_name=prompt_name,
        model_name_id=config.VERIFIER_MODEL_ID,
        n_generations=1,
        logprobs_num=run_args["logprobs_num"],
        temperature=0,
        model_args=dict(run_args["model_args"]),
        sql_column="predicted_sql",
        output_path=output_path,
    )
    return output_path


def verify(processed_runs: list[Path]) -> None:
    """Runs the p_true and self-probing verification of every given processed generation run, always judged greedily
    by config.VERIFIER_MODEL_ID."""
    for processed_run in processed_runs:
        for prompt_name, run_args in config.VERIFICATION_RUN_ARGS.items():
            logger.info(f"Verifying: {processed_run.name} / {prompt_name} / {config.VERIFIER_MODEL_ID}")
            run(
                dataset_name=str(processed_run),
                prompt_name=prompt_name,
                model_name_id=config.VERIFIER_MODEL_ID,
                n_generations=1,
                logprobs_num=run_args["logprobs_num"],
                temperature=0,
                model_args=dict(run_args["model_args"]),
                sql_column="predicted_sql",
            )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Run the generation runs of config.GENERATION_PLAN.")
    generate_parser.add_argument(
        "--models", nargs="+", choices=list(config.MODELS), default=None, help="Only these model ids.",
    )
    generate_parser.add_argument(
        "--datasets", nargs="+", choices=list(config.DATASETS), default=None, help="Only these datasets.",
    )

    verify_parser = subparsers.add_parser(
        "verify", help="Run the p_true and self-probing verification of the processed default run of each experiment.",
    )
    verify_parser.add_argument(
        "--models", nargs="+", metavar="GROUP", default=None,
        help="Only these experiment groups (e.g. qwen2_5_coder_32b_10_results) or models (e.g. qwen2_5_coder_32b).",
    )
    verify_parser.add_argument(
        "--datasets", nargs="+", choices=list(config.DATASETS), default=None, help="Only these datasets.",
    )

    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.models, args.datasets)
    else:
        experiments = select_experiments(args.models, args.datasets)
        verify(sorted({
            processed_run_path(experiment["default"], config.PROCESSED_RUNS_DIR, config.RAW_RUNS_DIR)
            for group_experiments in experiments.values() for experiment in group_experiments
        }))


if __name__ == "__main__":
    main()
