# Systematic Analysis of Uncertainty Estimation for Text-to-SQL Reliability

Code and instructions to reproduce the results of the paper *Systematic Analysis of Uncertainty Estimation for
Text-to-SQL Reliability*.

## Contents

1. [Setup](#setup)
2. [Step 1: the inference results](#step-1-the-inference-results)
3. [Step 2: running the pipeline](#step-2-running-the-pipeline)
4. [The experiments file](#the-experiments-file)
5. [Configuration](#configuration)

## Setup

Python 3.11 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt            # everything needed by step 2 (the pipeline)
pip install -r requirements-generation.txt # additionally, only to run the Text-to-SQL models yourself (step 1)
```

### Third-party code to download (OmniSQL, XiYan M-Schema and RTS-SQL)

Some upstream files are not included in this repository:

- `text_to_sql/utils/omnisql_utils/` and `text_to_sql/utils/xiyan_utils/`: only needed to run the OmniSQL and XiYan
  models yourself (step 1). Follow [text_to_sql/utils/README.md](text_to_sql/utils/README.md).
- `uncertainty_methods/logit_based/rts_sql/execution_entropy.py`: needed for the execution entropy uncertainty
  (step 2). Follow [uncertainty_methods/logit_based/rts_sql/README.md](uncertainty_methods/logit_based/rts_sql/README.md).

### Download the datasets

Both steps execute SQL queries against the original benchmark databases. Follow the README of each dataset to download
it into its `storage/` folder: [Spider](evaluated_datasets/spider/README.md), [BIRD](evaluated_datasets/bird/README.md),
[AmbiQT](evaluated_datasets/ambiqt/README.md), [Ambrosia](evaluated_datasets/ambrosia/README.md),
[TrustSQL](evaluated_datasets/trustsql/README.md).

## Step 1: the inference results

The pipeline of step 2 starts from the *inference results*: the raw outputs of the Text-to-SQL models (the SQL
generations with their token logprobs). To reproduce them, create the
inference results with step 1 and the result files with step 2.

**Run the generation models** (GPUs and vLLM, `requirements-generation.txt`) with
[`inference/run_model.py`](inference/run_model.py):

```bash
python -m inference.run_model generate    # every generation run of GENERATION_PLAN (pipeline/config.py) on the 5 datasets
```

`--models` and `--datasets` run a part of the plan. Each run is saved to `storage/model_run_results/` under a name that
ends with a timestamp (see [The experiments file](#the-experiments-file) for the resulting layout).

**Register the runs** in [`pipeline/experiments.json`](pipeline/experiments.json), which is empty in the repository: one
entry per (model, pipeline, dataset) with the file names of its generation runs, as described in
[The experiments file](#the-experiments-file). The following pipeline only works on the experiments registered there.

## Step 2: running the pipeline

```bash
python -m pipeline
```

runs the following stages in order (`--stages` selects some of them):

| Stage | What it does | Writes to |
|---|---|---|
| `process` | Parses the SQL out of each generation, selects one generation per question by self-consistency majority voting, and computes its execution accuracy. | `storage/_work/processed_runs/` (temporary) |
| `uncertainty` | Computes the uncertainty scores of the 11 methods for every (model, dataset) experiment, their AUROC/AURC, and the composite datasets (Spider-all errors, AmbiQT+Spider). For P(True) and self-probing it reuses the verification run if it exists, and otherwise runs the verifier model (GPUs, see below). | `storage/uncertainty_results/` and, for the verifier, `storage/model_run_results/` |
| `classification` | Selects thresholds (oracle, or from a sample of the data) and scores the uncertainty-based classifiers. | `storage/classification_results/` |
| `cross_dataset` | Selects thresholds on one dataset and applies them to the others. | `storage/cross_dataset_classification_results/` |
| `execution_baseline` | Scores the baseline that rejects queries with execution errors or empty results. | `storage/execution_classification_results/` |

Options:

```bash
python -m pipeline --stages classification cross_dataset   # only some stages
python -m pipeline --models qwen2_5_coder_32b --datasets spider ambrosia
python -m pipeline --reprocess         # recompute parsing/majority voting/execution (see below)
python -m pipeline --keep-intermediate # keep storage/_work/
```

`--models` takes experiment groups (`qwen2_5_coder_32b_10_results`) or a model name (`qwen2_5_coder_32b`, which
selects its greedy and self-consistency groups). `--datasets` takes the datasets of the experiments file; the composite
datasets are only built when all their ingredients are selected.

**The verifier runs inside the `uncertainty` stage.** P(True) and self-probing need a verifier model (Qwen2.5-Coder-32B)
to judge the SQL each default run selected. The stage looks for the verification run registered in the experiments file (or one
it generated earlier, saved in `storage/model_run_results/` as `{default run}__{p_true_unique|self_probing_unique}.xlsx`)
and reuses it; only if none exists does it run the verifier (GPUs and vLLM). The pipeline warns before starting
how many verification runs are missing, so on a first run it needs a GPU. To run the verifier separately, for example
on a GPU server, before step 2:

```bash
python -m pipeline --stages process --keep-intermediate   # selects the SQL to verify
python -m inference.run_model verify                      # P(True) and self-probing, judged by Qwen2.5-Coder-32B
```

and register the resulting verification files in the experiments file.

## Configuration

Everything that can be changed is in [`pipeline/config.py`](pipeline/config.py):

* paths of the inputs, outputs and the temporary directory (`STORAGE_DIR`, overridable with the environment variable
  `TEXT2SQL_UE_STORAGE_DIR`);
* the registries of models, datasets and prompts (`MODELS`, `DATASETS`, `PROMPTS`) and the generation plan;
* the parameters of the methods (`SINGLE_LOGIT_BASED_VARIATIONS`, `MAJORITY_VOTING_SEED`);
* the threshold selection algorithms (`ORACLE_ALGORITHMS`, `SAMPLING_ALGORITHMS`), the groups they run on and the
  datasets of the classification stages (`CLASSIFICATION_DATASETS`).

The run files of each experiment are listed in
[`pipeline/experiments.json`](pipeline/experiments.json): for each group, one entry per
dataset with the `default` run (logit-based and consistency), the `vanilla` and `cot`
verbalized runs, and the `p_true` and `self_probing` verification runs.

## The experiments file

[`pipeline/experiments.json`](pipeline/experiments.json) tells the pipeline which runs exist and which experiments to
compute. It is empty (`{}`) in the repository: you fill it in once step 1 has produced the generation runs, and the
pipeline then works on the experiments listed there (and only those). Its structure:

```json
{
    "qwen2_5_coder_32b_10_results": [
        {
            "dataset": "spider",
            "default": "spider_vanilla_verbalized_maleki_qwen2_5_coder_32b_10_results_1700000000.xlsx",
            "vanilla": "spider_vanilla_verbalized_maleki_qwen2_5_coder_32b_10_results_1700000000.xlsx",
            "cot": "spider_cot_verbalized_maleki_qwen2_5_coder_32b_10_results_1700000001.xlsx",
            "p_true": null,
            "self_probing": null
        },
        {"dataset": "ambiqt", "default": "ambiqt_vanilla_verbalized_maleki_qwen2_5_coder_32b_10_results_1700000010.xlsx",
         "vanilla": null, "cot": null, "p_true": null, "self_probing": null}
    ],
    "xiyan_32b_10_results": [
        {"dataset": "spider", "default": "spider_xiyan_xiyan-32b_10_results_1700000020.xlsx",
         "vanilla": null, "cot": null, "p_true": null, "self_probing": null}
    ]
}
```

* **Group (the key).** A model plus a generation pipeline: the model id with `.` and `-` replaced by `_`, then `greedy`
  or `10_results`, e.g. `omnisql_32b_10_results`. A new group needs no code, but the sampling-based thresholds, the
  cross-dataset results and the `spider_all` composite are only computed for the groups listed in `SAMPLING_GROUPS`,
  `CROSS_DATASET_GROUPS` and `SPIDER_ALL_GROUPS` in `pipeline/config.py`.
* **Entry.** One per dataset (`spider`, `bird`, `ambrosia`, `trustsql` or `ambiqt`) of the group, with all six fields
  (the structure is validated when the pipeline starts). A group needs `spider` and `ambiqt` for the
  `ambiqt_plus_spider` composite.
* **`default`** (required): the generation run the logit-based and consistency methods and the classifiers are
  computed on. **`vanilla`** and **`cot`**: the generation runs of the verbalized-confidence prompts, each `null` if
  the method is not evaluated for the model; `vanilla` is usually the same file as `default`.
* **`p_true`** and **`self_probing`**: the verification runs of the default run. After step 1 they are `null`: the
  `uncertainty` stage then runs the verifier and saves the runs, and you can register them afterwards to skip the
  verifier on later runs.
* All values are file names in `storage/model_run_results/`.

### What `storage/` contains

After step 1 (generation runs, named `{dataset}_{prompt}_{model}_{greedy|10_results}_{timestamp}.xlsx`):

```
storage/model_run_results/
    spider_vanilla_verbalized_maleki_qwen2_5_coder_32b_10_results_1700000000.xlsx
    spider_cot_verbalized_maleki_qwen2_5_coder_32b_10_results_1700000001.xlsx
    ...
```

After step 2:

```
storage/
    model_run_results/              + {default run}__p_true_unique.xlsx, {default run}__self_probing_unique.xlsx
                                      (the verification runs, if the pipeline ran the verifier)
    uncertainty_results/            {default run}_uncertainties.xlsx for each experiment, plus
                                      ambiqt_plus_spider_{group}_uncertainties.xlsx (and spider_all_{group}_...)
    classification_results/         {default run}_classification_results.xlsx (and one per composite)
    cross_dataset_classification_results/   {group}_cross_dataset_classification_results.xlsx
    execution_classification_results/       execution_results_based_classification_results.xlsx
    query_execution_cache/          cache of the executed queries
```

`storage/_work/` (the processed runs) only exists while the pipeline runs, unless you pass `--keep-intermediate`.

## License

The code of this repository is released under the [MIT License](LICENSE).
