"""
Helpers shared by the modules of pipeline/ and inference/: reading/writing the run model results excel files, resolving the
prompt of a run, picking the selected generation of a row, and a small table logger.
"""
import ast
import gc
import importlib
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from pipeline import config
from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC
from uncertainty_methods.logit_based.logit_based_uncertainty import Generation


# =============================================================================================
# Experiments file
# =============================================================================================

def _group_matches(group: str, name: str) -> bool:
    """Whether `name` is the group itself or a model name whose greedy / 10_results pipeline is `group`."""
    return group == name or group in (f"{name}_greedy", f"{name}_10_results")


def select_experiments(
    groups: Optional[list[str]] = None, datasets: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    """The experiments of config.EXPERIMENTS restricted to the given groups (e.g. "qwen2_5_coder_32b_10_results") or
    model names without their pipeline (e.g. "qwen2_5_coder_32b" selects its greedy and 10_results groups), and to
    the given datasets. None selects everything."""
    unknown = [name for name in groups or [] if not any(_group_matches(g, name) for g in config.EXPERIMENTS)]
    if unknown:
        raise ValueError(f"Unknown group/model(s): {unknown} - valid groups are {sorted(config.EXPERIMENTS)}.")

    selected = {}
    for group, experiments in config.EXPERIMENTS.items():
        if groups and not any(_group_matches(group, name) for name in groups):
            continue
        kept = [e for e in experiments if not datasets or e["dataset"] in datasets]
        if kept:
            selected[group] = kept
    return selected


def generation_run_files(experiments: dict[str, list[dict]]) -> list[str]:
    """File names of the generation runs (default/vanilla/cot) of the experiments, deduplicated, in the sorted order
    they are processed in. Verification runs (p_true/self_probing) are not processed."""
    files = {
        experiment[role] for group_experiments in experiments.values() for experiment in group_experiments
        for role in ("default", "vanilla", "cot") if experiment[role]
    }
    return sorted(files)


def processed_run_path(file_name: str, processed_dir: Path, raw_dir: Path) -> Path:
    """The enriched version of a generation run: the file written by the process stage if the run needed processing,
    otherwise the raw file itself (a run that is already complete is not copied)."""
    processed_path = processed_dir / file_name
    return processed_path if processed_path.exists() else raw_dir / file_name


def registered_verification_run(default_run_name: str, prompt_name: str, raw_dir: Path) -> Optional[Path]:
    """The verification run (`prompt_name`: a key of config.VERIFICATION_RUN_ARGS) that the experiments file registers for the
    default run `default_run_name`, or None if it registers none."""
    experiment_field = config.VERIFICATION_EXPERIMENT_FIELDS[prompt_name]
    for experiments in config.EXPERIMENTS.values():
        for experiment in experiments:
            if experiment["default"] == default_run_name and experiment[experiment_field]:
                return raw_dir / experiment[experiment_field]
    return None


def generated_verification_run(default_run_name: str, prompt_name: str, raw_dir: Path) -> Path:
    """Where the verification run of `default_run_name` is saved when the pipeline runs the verifier itself."""
    return raw_dir / f"{Path(default_run_name).stem}__{prompt_name}.xlsx"


def existing_verification_run(default_run_name: str, prompt_name: str, raw_dir: Path) -> Optional[Path]:
    """The verification run of `default_run_name` that is already on disk - the one registered in the experiments file, or one
    the pipeline generated before - or None if the verifier still has to be run."""
    candidates = (
        registered_verification_run(default_run_name, prompt_name, raw_dir),
        generated_verification_run(default_run_name, prompt_name, raw_dir),
    )
    return next((path for path in candidates if path is not None and path.exists()), None)


def uncertainties_file_paths(group: str, group_experiments: list[dict], uncertainty_dir: Path) -> dict[str, Path]:
    """
    The uncertainties files of the evaluation datasets (config.CLASSIFICATION_DATASETS) of an experiment group, as
    {dataset name: path}: `{default run name}_uncertainties.xlsx` for a dataset of the experiments file and
    `{composite}_{group}_uncertainties.xlsx` for a composite dataset. Datasets of the group without a file are skipped.
    """
    default_runs = {experiment["dataset"]: experiment["default"] for experiment in group_experiments}
    paths = {}
    for dataset in config.CLASSIFICATION_DATASETS:
        if dataset in (config.AMBIQT_PLUS_SPIDER, config.SPIDER_ALL):
            path = uncertainty_dir / f"{dataset}_{group}_uncertainties.xlsx"
        elif dataset in default_runs:
            path = uncertainty_dir / f"{Path(default_runs[dataset]).stem}_uncertainties.xlsx"
        else:
            continue
        if path.exists():
            paths[dataset] = path
        else:
            logger.warning(f"[{group}/{dataset}] {path.name} not found in {uncertainty_dir} - skipping.")
    return paths


# =============================================================================================
# Run model results files
# =============================================================================================

def fix_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse any column values that were serialised as list strings back to Python lists."""
    for col in df.columns:
        def _try_parse(x):
            if isinstance(x, str) and x.startswith("["):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    pass
            return x
        df[col] = df[col].apply(_try_parse)
    return df


def resolve_db_path(db_path: str) -> str:
    """Re-anchors a db path stored in a run file to PROJECT_ROOT. The run files store the absolute path of the machine
    that produced them, ending in `evaluated_datasets/<dataset>/storage/...`; everything before `evaluated_datasets`
    is dropped. A path without it is taken as relative to PROJECT_ROOT."""
    parts = Path(db_path).parts
    if "evaluated_datasets" in parts:
        parts = parts[parts.index("evaluated_datasets"):]
    return str(config.PROJECT_ROOT.joinpath(*parts))


def load_run_results(file_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the 'results' (with list-valued cells parsed back) and 'configuration' sheets of a run results file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # A large (100s of MB) xlsx file can take several minutes to parse - log it so a run doesn't look stalled.
    logger.info(f"Loading {file_path.name} ({file_path.stat().st_size / (1024 * 1024):.0f} MB)...")
    results_df = fix_column_types(pd.read_excel(file_path, sheet_name="results"))
    config_df = pd.read_excel(file_path, sheet_name="configuration")
    return results_df, config_df


def save_run_results(results_df: pd.DataFrame, config_df: pd.DataFrame, file_path: str | Path) -> None:
    """Writes results_df/config_df to the 'results'/'configuration' sheets of file_path."""
    if "exec_error" in results_df.columns:
        # exec_error can hold a parser's exception message with ANSI escape codes, which openpyxl rejects when
        # writing a cell (IllegalCharacterError).
        results_df = results_df.copy()
        results_df["exec_error"] = results_df["exec_error"].map(
            lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v
        )
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="results", index=False)
        config_df.to_excel(writer, sheet_name="configuration", index=False)
    logger.info(f"Saved to {file_path}")


def generation_indices(df: pd.DataFrame) -> list[int]:
    """Returns the sorted list of i's for every generated_output_i column found in df."""
    indices = []
    for col in df.columns:
        m = re.match(r"generated_output_(\d+)$", col)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def is_n_results_run(results_df: pd.DataFrame) -> bool:
    """Returns True if this run sampled more than one generation per question, i.e. requires majority voting."""
    return len(generation_indices(results_df)) > 1


def get_generation(row: pd.Series, idx: int) -> Optional[Generation]:
    """Builds a Generation dict for the idx-th sampled candidate of a row, or None if it is missing/incomplete."""
    output = row.get(f"generated_output_{idx}")
    tokens = row.get(f"generated_tokens_{idx}")
    log_probs = row.get(f"log_probabilities_{idx}")
    if pd.isna(output) or not isinstance(tokens, list) or not isinstance(log_probs, list):
        return None
    return {"generated_output": str(output), "generated_tokens": tokens, "token_log_probs": log_probs}


def selected_generation_index(row: pd.Series, n_generations: int) -> Optional[int]:
    """
    Which of generated_output_0..n-1 is the selected candidate: 0 for greedy runs (a single generation), or the
    self-consistency majority-voted predicted_sql_index for n_results runs.
    """
    if n_generations <= 1:
        return 0
    idx = row.get("predicted_sql_index")
    return int(idx) if pd.notna(idx) else None


# Per-token dumps of a generation run (top-k logprobs, token ids, per-step logprobs) - large (100s of MB per column on
# a 10-generation file) and never read by parsing/majority voting/execution accuracy. See spill_heavy_columns.
_HEAVY_COLUMN_RE = re.compile(r"^(top_tokens_probs|log_probabilities|generated_tokens)_\d+$")


def spill_heavy_columns(results_df: pd.DataFrame, file_path: str | Path) -> tuple[pd.DataFrame, Optional[Path]]:
    """Writes results_df's heavy per-generation columns to a sidecar pickle in WORK_DIR and drops them from the
    returned DataFrame, so processing a 10-generation file doesn't hold multiple GB in memory (which has pushed the
    process over its memory limit). Call reattach_heavy_columns right before saving. Returns
    (light_df, heavy_columns_path); heavy_columns_path is None if there was nothing to spill."""
    heavy_cols = [c for c in results_df.columns if _HEAVY_COLUMN_RE.match(c)]
    if not heavy_cols:
        return results_df, None

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    heavy_path = config.WORK_DIR / f"{Path(file_path).stem}.heavy_columns.pkl"
    results_df[heavy_cols].to_pickle(heavy_path)
    light_df = results_df.drop(columns=heavy_cols)
    del results_df
    gc.collect()
    return light_df, heavy_path


def reattach_heavy_columns(results_df: pd.DataFrame, heavy_columns_path: Optional[Path]) -> pd.DataFrame:
    """Merges back the columns spilled by spill_heavy_columns, if any. Row order/count must be unchanged since the
    spill, as the two frames are joined positionally."""
    if heavy_columns_path is None or not Path(heavy_columns_path).exists():
        return results_df

    heavy_df = pd.read_pickle(heavy_columns_path)
    if len(heavy_df) != len(results_df):
        raise ValueError(
            f"Row count mismatch reattaching heavy columns: {len(results_df)} rows in results_df, "
            f"{len(heavy_df)} in {heavy_columns_path} - row order must not change between "
            "spill_heavy_columns and reattach_heavy_columns."
        )
    return pd.concat([results_df.reset_index(drop=True), heavy_df.reset_index(drop=True)], axis=1)


# =============================================================================================
# Prompts
# =============================================================================================

def instantiate_prompt(prompt_name: str, dataset_name: Optional[str] = None):
    """Instantiates the prompt registered under `prompt_name` in config.PROMPTS (passing `dataset_name` to the prompts
    that take it)."""
    try:
        module_name, class_name, takes_dataset_name, _ = config.PROMPTS[prompt_name]
    except KeyError:
        raise ValueError(f"Unknown prompt name: {prompt_name}")
    prompt_class = getattr(importlib.import_module(module_name), class_name)
    return prompt_class(dataset_name=dataset_name) if takes_dataset_name else prompt_class()


def resolve_prompt_for_run(config_df: pd.DataFrame) -> PromptABC:
    """Resolves the prompt that produced a run from its 'configuration' sheet. Configuration sheets are inconsistent
    about which column holds the prompt class name - older files use 'prompt_type', newer ones 'prompt_name'."""
    run_config = config_df.iloc[0]
    for col in ("prompt_type", "prompt_name"):
        if col in run_config.index:
            prompt_class_name = str(run_config[col])
            break
    else:
        raise KeyError("Neither 'prompt_type' nor 'prompt_name' found in configuration sheet")

    try:
        prompt_name = config.PROMPT_NAME_BY_CLASS[prompt_class_name]
    except KeyError:
        raise ValueError(f"Unknown prompt class '{prompt_class_name}'. Add it to PROMPTS in pipeline/config.py.")
    return instantiate_prompt(prompt_name)


# =============================================================================================
# Uncertainty-method columns
# =============================================================================================

def uncertainty_method_columns(results_df: pd.DataFrame) -> list[str]:
    """The uncertainty-method columns of an uncertainties 'results' sheet that have at least one value."""
    return [
        col for col in results_df.columns
        if col not in config.NON_METHOD_COLUMNS
        and not col.endswith(config.NON_METHOD_COLUMN_SUFFIXES)
        and results_df[col].notna().any()
    ]


def role_of_method(method: str) -> str:
    """The run ('default', 'vanilla' or 'cot') whose predicted_sql/exec an uncertainty method is evaluated against."""
    return config.UNCERTAINTY_COLUMN_ROLE.get(method, config.DEFAULT_ROLE)


# =============================================================================================
# Logging
# =============================================================================================

def log_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Logs a left-aligned text table."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) + 2 for i, h in enumerate(headers)]
    separator = "-" * sum(widths)
    lines = [title, "".join(h.ljust(w) for h, w in zip(headers, widths)), separator]
    lines.extend("".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    lines.append(separator)
    logger.info("\n".join(lines))
