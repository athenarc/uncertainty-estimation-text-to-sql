"""
Single source of configuration for the results pipeline (see pipeline/run.py).

Everything that is shared across the modules of pipeline/ and inference/ lives here: paths, the model/dataset/
prompt registries, generation and verification parameters, the parameters of the uncertainty methods and of the
selective-prediction classifiers, and the experiments file. Only the experiments of the paper are configured.

This module must stay light (no heavy imports) because it is imported by every script and by metrics/.
"""
import json
import os
from pathlib import Path

# =============================================================================================
# Paths
# =============================================================================================

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
# All the inputs, outputs and temporary files of the pipeline live under STORAGE_DIR; the environment variable
# TEXT2SQL_UE_STORAGE_DIR points the pipeline to another folder (e.g. to try it on a few rows of the runs).
STORAGE_DIR = Path(os.environ.get("TEXT2SQL_UE_STORAGE_DIR", PROJECT_ROOT / "storage"))

# Input: the raw outputs of inference/run_model.py (generation runs and verification runs).
RAW_RUNS_DIR = STORAGE_DIR / "model_run_results"

# Final outputs (the files the paper's figures and tables are built from).
UNCERTAINTY_RESULTS_DIR = STORAGE_DIR / "uncertainty_results"
CLASSIFICATION_RESULTS_DIR = STORAGE_DIR / "classification_results"
CROSS_DATASET_RESULTS_DIR = STORAGE_DIR / "cross_dataset_classification_results"
EXECUTION_CLASSIFICATION_RESULTS_DIR = STORAGE_DIR / "execution_classification_results"

# Working directory of one pipeline run: the processed run files handed from stage to stage. Removed at the end of
# the pipeline unless --keep-intermediate is passed.
WORK_DIR = STORAGE_DIR / "_work"
PROCESSED_RUNS_DIR = WORK_DIR / "processed_runs"

# Persistent (database, sql) -> result cache of the query executions (see metrics/utils/query_execution_cache.py).
QUERY_EXECUTION_CACHE_DIR = STORAGE_DIR / "query_execution_cache"

EXPERIMENTS_FILE_PATH = PIPELINE_DIR / "experiments.json"

# The group of the main experiments of the paper (Qwen2.5-Coder-32B, self-consistency generations).
PRIMARY_GROUP = "qwen2_5_coder_32b_10_results"

# =============================================================================================
# Registries: datasets, models, prompts
# =============================================================================================

# dataset key -> (module, class); imported lazily by inference/run_model.py.
DATASETS: dict[str, tuple[str, str]] = {
    "spider": ("evaluated_datasets.spider.spider", "SpiderDataset"),
    "bird": ("evaluated_datasets.bird.bird", "BirdDataset"),
    "ambrosia": ("evaluated_datasets.ambrosia.ambrosia", "AmbrosiaDataset"),
    "ambiqt": ("evaluated_datasets.ambiqt.ambiqt", "AmbiQTDataset"),
    "trustsql": ("evaluated_datasets.trustsql.trustsql", "TrustSQLDataset"),
}
DATASET_CLASS_TO_KEY = {class_name: key for key, (_, class_name) in DATASETS.items()}

# =============================================================================================
# Experiments file
# =============================================================================================

EXPERIMENT_FIELDS = ("dataset", "default", "vanilla", "cot", "p_true", "self_probing")


def validate_experiments(experiments: dict) -> dict[str, list[dict]]:
    """Checks the structure of an experiments file (see the README) and returns it. File existence is not checked
    here: the pipeline reports the files it needs and does not find."""
    if not isinstance(experiments, dict):
        raise ValueError("The experiments file must be a JSON object {group: [experiment, ...]}.")
    for group, group_experiments in experiments.items():
        if not isinstance(group_experiments, list):
            raise ValueError(f"Group '{group}' must be a list of experiments.")
        datasets = []
        for experiment in group_experiments:
            if not isinstance(experiment, dict) or set(experiment) != set(EXPERIMENT_FIELDS):
                raise ValueError(f"An experiment of group '{group}' must have exactly the fields {EXPERIMENT_FIELDS}.")
            if experiment["dataset"] not in DATASETS:
                raise ValueError(f"Group '{group}': unknown dataset '{experiment['dataset']}', valid: {list(DATASETS)}.")
            if not isinstance(experiment["default"], str) or not experiment["default"]:
                raise ValueError(f"Group '{group}'/{experiment['dataset']}: 'default' must be a file name.")
            for field in EXPERIMENT_FIELDS[2:]:
                if experiment[field] is not None and not isinstance(experiment[field], str):
                    raise ValueError(f"Group '{group}'/{experiment['dataset']}: '{field}' must be a file name or null.")
            datasets.append(experiment["dataset"])
        if len(datasets) != len(set(datasets)):
            raise ValueError(f"Group '{group}' lists a dataset twice.")
    return experiments


# The experiments the pipeline runs, read from pipeline/experiments.json (see the README for its structure):
# {group: [{"dataset", "default", "vanilla", "cot", "p_true", "self_probing"}, ...]}, where a group is a model plus a
# generation pipeline (e.g. "qwen2_5_coder_32b_10_results"). The values are file names in RAW_RUNS_DIR (None if the
# run is not used): "default" is the generation run the logit-based, entropy and consistency methods and the
# classifiers are computed on, "vanilla"/"cot" the verbalized-confidence generation runs, and "p_true"/"self_probing"
# the verification runs (optional: a missing one is generated by running the verifier).
with open(EXPERIMENTS_FILE_PATH) as _f:
    EXPERIMENTS: dict[str, list[dict]] = validate_experiments(json.load(_f))

# model id -> HuggingFace name and the sampling arguments used for the sampled (temperature > 0) runs. The arguments
# are the ones recorded in the `configuration` sheet of the paper's runs.
MODELS: dict[str, dict] = {
    "qwen2.5_coder_32b": {
        "hf_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "sampling_args": {"top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05},
    },
    "qwen3-coder-32b": {"hf_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "sampling_args": {}},
    "omnisql_32b": {
        "hf_name": "seeklhy/OmniSQL-32B",
        "sampling_args": {"top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05},
    },
    "xiyan-32b": {"hf_name": "XGenerationLab/XiYanSQL-QwenCoder-32B-2504", "sampling_args": {}},
}

# prompt name -> (module, class, takes_dataset_name, is_verification_prompt). The vanilla/cot verbalized prompts are
# the ones of Maleki et al., "Confidence Estimation for Text-to-SQL in Large Language Models" (AAAI'26,
# https://arxiv.org/pdf/2508.14056). The class name is what the
# `configuration` sheet of a run file records, and is used to resolve the prompt of a run back from its file.
PROMPTS: dict[str, tuple[str, str, bool, bool]] = {
    "vanilla_verbalized_maleki": (
        "text_to_sql.prompt_templates.text_to_sql_task.VanillaVerbalizedConfidence_MalekiPrompt",
        "VanillaVerbalizedMalekiPrompt", True, False,
    ),
    "cot_verbalized_maleki": (
        "text_to_sql.prompt_templates.text_to_sql_task.COTVerbalizedConfidence_MalekiPrompt",
        "COTVerbalizedMalekiPrompt", True, False,
    ),
    "omnisql": ("text_to_sql.prompt_templates.text_to_sql_task.OmniSQLPrompt", "OmniSQLPrompt", True, False),
    "xiyan": ("text_to_sql.prompt_templates.text_to_sql_task.XiYanSQLPrompt", "XiYanSQLPrompt", False, False),
    "p_true_unique": (
        "text_to_sql.prompt_templates.verification_task.PTrueUniquePrompt", "PTrueUniquePrompt", True, True,
    ),
    "self_probing_unique": (
        "text_to_sql.prompt_templates.verification_task.SelfProbingUniquePrompt", "SelfProbingUniquePrompt", True, True,
    ),
}
PROMPT_NAME_BY_CLASS = {class_name: name for name, (_, class_name, _, _) in PROMPTS.items()}
# The task-model prompts of the paper come from Maleki et al. (https://arxiv.org/pdf/2508.14056). The `configuration`
# sheet of the released runs still records their former class names.
PROMPT_NAME_BY_CLASS |= {
    "VanillaVerbalizedPourrezaPrompt": "vanilla_verbalized_maleki",
    "COTVerbalizedPourrezaPrompt": "cot_verbalized_maleki",
}

# =============================================================================================
# Generation (inference/run_model.py)
# =============================================================================================

# (n_generations, temperature): the greedy pipeline and the self-consistency pipeline (10 sampled generations).
GREEDY_DECODING = (1, 0)
SELF_CONSISTENCY_DECODING = (10, 0.7)

GENERATION_LOGPROBS_NUM = 5
GENERATION_MODEL_ARGS = {"max_tokens": 2048}

# (model id, prompt name, decoding configs) - every generation run of the paper, executed on each of DATASETS.
GENERATION_PLAN: list[tuple[str, str, list[tuple[int, float]]]] = [
    ("qwen2.5_coder_32b", "vanilla_verbalized_maleki", [GREEDY_DECODING, SELF_CONSISTENCY_DECODING]),
    ("qwen2.5_coder_32b", "cot_verbalized_maleki", [GREEDY_DECODING, SELF_CONSISTENCY_DECODING]),
    ("qwen3-coder-32b", "vanilla_verbalized_maleki", [SELF_CONSISTENCY_DECODING]),
    ("omnisql_32b", "omnisql", [SELF_CONSISTENCY_DECODING]),
    ("xiyan-32b", "xiyan", [SELF_CONSISTENCY_DECODING]),
]

# P(True) only emits "(A)"/"(B)" and is read off the logprobs of that first token (short max_tokens, wide logprobs);
# self-probing emits a plain 0-100 integer parsed from the text.
VERIFICATION_RUN_ARGS = {
    "p_true_unique": {"logprobs_num": 10, "model_args": {"max_tokens": 3}},
    "self_probing_unique": {"logprobs_num": 10, "model_args": {"max_tokens": 10}},
}
# The experiment field (see EXPERIMENTS) that registers the run of each verification prompt.
VERIFICATION_EXPERIMENT_FIELDS = {"p_true_unique": "p_true", "self_probing_unique": "self_probing"}
# Both verification prompts judge the processed SQL of every generation run, always with this model, greedily.
VERIFIER_MODEL_ID = "qwen2.5_coder_32b"

# =============================================================================================
# Processing of the runs and uncertainty methods
# =============================================================================================

# Seed of the random choice of a candidate when none of the sampled generations executes (majority voting).
MAJORITY_VOTING_SEED = 35

# Timeout of the execution of a predicted query when checking for empty result sets.
QUERY_TIMEOUT_SECONDS = 180

# (considered tokens, aggregation) values of the single-output logit-based methods: mean, schema-linked and max-word.
# Column names are logit_based_{considered_tokens}_{aggregation}.
SINGLE_LOGIT_BASED_VARIATIONS = [
    ("all", "mean_nll"),
    ("schema_linked_only", "mean_nll"),
    ("all", "max_word_nll"),
]

# Which run's predicted_sql/exec an uncertainty column is evaluated against: the verbalized methods are tied to the
# run they were read from (it can select a different generation), every other method to the default run.
UNCERTAINTY_COLUMN_ROLE = {"vanilla_verbalized": "vanilla", "cot_verbalized": "cot"}
DEFAULT_ROLE = "default"

# Columns of an uncertainties "results" sheet that are not uncertainty-method scores: row identifiers and question
# labels, plus each run's own `{role}_predicted_sql` / `{role}_exec` / `{role}_exec_error`.
NON_METHOD_COLUMNS = (
    "question", "db_path", "sql_query", "is_ambiguous", "ambig_type", "unanswerable", "origin_dataset",
)
NON_METHOD_COLUMN_SUFFIXES = ("_predicted_sql", "_exec", "_exec_error")

# =============================================================================================
# Composite datasets
# =============================================================================================

AMBIQT_PLUS_SPIDER = "ambiqt_plus_spider"
SPIDER_ALL = "spider_all"
# Groups for which the spider_all composite (every failure case over the Spider databases) is built.
SPIDER_ALL_GROUPS = [PRIMARY_GROUP]
# TrustSQL keeps one sqlite db per origin dataset under storage/dataset/<origin>/ - this identifies its Spider rows.
TRUSTSQL_SPIDER_ORIGIN_RE = r"trustsql/storage/dataset/spider/"

# =============================================================================================
# Selective-prediction classifiers
# =============================================================================================

# The five evaluation datasets of the paper (ambiqt is only an ingredient of the ambiqt_plus_spider composite).
CLASSIFICATION_DATASETS = ["spider", "bird", "ambrosia", "trustsql", AMBIQT_PLUS_SPIDER]

# Seed of the sub-sampling of the sampling-based algorithms (each (method, algorithm) call re-seeds from it).
CLASSIFICATION_RANDOM_STATE = 0
SAMPLING_N_REPEATS = 50
SAMPLE_SIZES = [100, 200, 300, 400, 500]

# algorithm name -> (threshold_alg, selection_pool_size, n_repeats, sample_distribution, target_accuracy) as the
# keyword arguments of text_to_sql_classifier.threshold_based_classification.classification().
ORACLE_ALGORITHMS: dict[str, dict] = {
    "oracle_f_0_5_score_based_threshold": {"threshold_alg": "f_0_5_score_based_threshold", "selection_pool_size": 1.0},
    **{
        f"oracle_{str(accuracy).replace('.', '_')}_accuracy_based_threshold": {
            "threshold_alg": "accuracy_restriction_based_threshold", "selection_pool_size": 1.0,
            "target_accuracy": accuracy,
        }
        for accuracy in (0.9, 0.8, 0.7, 0.6)
    },
}
# The threshold is selected on a sample of `size` examples: drawn at random ("initial" distribution) or with an
# equal number of correct and wrong ones ("equal").
SAMPLING_ALGORITHMS: dict[str, dict] = {
    **{
        f"sample_{size}_examples_f_0_5_score_based_threshold": {
            "threshold_alg": "f_0_5_score_based_threshold", "selection_pool_size": size,
            "n_repeats": SAMPLING_N_REPEATS,
        }
        for size in SAMPLE_SIZES
    },
    **{
        f"sample_{size}_examples_equal_dist_f_0_5_score_based_threshold": {
            "threshold_alg": "f_0_5_score_based_threshold", "selection_pool_size": size,
            "n_repeats": SAMPLING_N_REPEATS, "sample_distribution": "equal",
        }
        for size in SAMPLE_SIZES
    },
}

# Sampling algorithms are only evaluated on the primary group; the oracle ones on every group.
SAMPLING_GROUPS = [PRIMARY_GROUP]


def classification_algorithms(group: str) -> dict[str, dict]:
    """The algorithms evaluated for the experiments of `group`."""
    if group in SAMPLING_GROUPS:
        return {**ORACLE_ALGORITHMS, **SAMPLING_ALGORITHMS}
    return dict(ORACLE_ALGORITHMS)


# Cross-dataset thresholds (selected on a source dataset, applied to a target one): groups and algorithms.
CROSS_DATASET_GROUPS = [PRIMARY_GROUP]
CROSS_DATASET_ALGORITHMS = ["sample_500_examples_equal_dist_f_0_5_score_based_threshold"]
CROSS_DATASET_RANDOM_STATE = 0
