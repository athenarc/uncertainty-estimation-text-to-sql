"""
Step 2 of the reproduction: creates the result files of the paper from the inference results (the raw model runs in
storage/model_run_results/, see inference/run_model.py for how they are obtained):

    process            parse SQL, majority voting, execution acc.   -> storage/_work/processed_runs/ (temporary)
    uncertainty        uncertainty scores + overall metrics         -> storage/uncertainty_results/
    classification     uncertainty-based classifiers                -> storage/classification_results/
    cross_dataset      thresholds transferred across datasets       -> storage/cross_dataset_classification_results/
    execution_baseline execution-results-based classifier           -> storage/execution_classification_results/

Usage (from the repository root):

    python -m pipeline                         # all the stages
    python -m pipeline --stages classification cross_dataset
    python -m pipeline --models qwen2_5_coder_32b --datasets spider ambrosia

The experiments (model groups, datasets and the run files of each) are listed in pipeline/experiments.json
and everything else is configured in pipeline/config.py.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from pipeline import config
from pipeline.common import existing_verification_run, generation_run_files, select_experiments

STAGES = ["process", "uncertainty", "classification", "cross_dataset", "execution_baseline"]


def _missing_inputs(experiments: dict[str, list[dict]], stages: list[str]) -> list[Path]:
    """The raw input files the given stages need that are not in config.RAW_RUNS_DIR."""
    needed = set()
    if "process" in stages or "uncertainty" in stages:
        needed.update(generation_run_files(experiments))
    return sorted(config.RAW_RUNS_DIR / name for name in needed if not (config.RAW_RUNS_DIR / name).exists())


def _missing_verification_runs(experiments: dict[str, list[dict]]) -> list[str]:
    """The names of the default runs that have no P(True) or self-probing verification run yet: the uncertainty stage
    has to run the verifier model for them."""
    return sorted(
        f"{experiment['default']} ({prompt_name})"
        for group_experiments in experiments.values() for experiment in group_experiments
        for prompt_name in config.VERIFICATION_RUN_ARGS
        if existing_verification_run(experiment["default"], prompt_name, config.RAW_RUNS_DIR) is None
    )


def run_stage(stage: str, experiments: dict[str, list[dict]], reprocess: bool = False) -> list[str]:
    """Runs one stage of the pipeline over the selected experiments and returns the names of its failures."""
    # Imported here so that a stage only needs the modules it uses.
    match stage:
        case "process":
            from pipeline.stages import process
            return process.run(generation_run_files(experiments), reprocess=reprocess)
        case "uncertainty":
            from pipeline.stages import uncertainty
            return uncertainty.run(experiments)
        case "classification":
            from pipeline.stages import classification
            return classification.run(experiments)
        case "cross_dataset":
            from pipeline.stages import cross_dataset
            return cross_dataset.run(experiments)
        case "execution_baseline":
            from pipeline.stages import execution_baseline
            return execution_baseline.run(experiments)
        case _:
            raise ValueError(f"Unknown stage: {stage}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=None,
        help=f"Stages to run, in pipeline order (default: {' '.join(STAGES)}).",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL", default=None,
        help="Only these experiment groups (e.g. qwen2_5_coder_32b_10_results) or models (e.g. qwen2_5_coder_32b, "
             "which selects its greedy and 10_results groups).",
    )
    parser.add_argument(
        "--datasets", nargs="+", metavar="DATASET", default=None,
        help="Only these datasets of the experiments file (see pipeline/experiments.json). The composite datasets "
             "are skipped if their ingredients are not all selected.",
    )
    parser.add_argument(
        "--reprocess", action="store_true",
        help="Recompute the SQL parsing, majority voting and execution accuracy of the run files even if they are "
             "already in them. The run files of the paper are already processed: recomputing them can change a few "
             "predicted SQLs, as the SQL parsing of the prompts was fixed after they were processed.",
    )
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Keep the working directory with the processed runs (storage/_work) after the pipeline.",
    )
    args = parser.parse_args(argv)

    selected = set(args.stages or STAGES)
    stages = [stage for stage in STAGES if stage in selected]  # pipeline order

    experiments = select_experiments(args.models, args.datasets)
    if not experiments:
        logger.error(
            "No experiment selected: register the experiments to run in pipeline/experiments.json (see the README) "
            "and check the --models and --datasets options."
        )
        return 1

    missing = _missing_inputs(experiments, stages)
    if missing:
        logger.error(
            f"{len(missing)} input file(s) not found in {config.RAW_RUNS_DIR}, e.g. {missing[0].name}. "
            "See the README for how to obtain the inference results."
        )
        return 1

    if "uncertainty" in stages:
        missing_verification = _missing_verification_runs(experiments)
        if missing_verification:
            logger.warning(
                f"{len(missing_verification)} P(True)/self-probing verification run(s) not found in "
                f"{config.RAW_RUNS_DIR}, e.g. {missing_verification[0]}: the uncertainty stage will run the verifier "
                f"model ({config.VERIFIER_MODEL_ID}), which needs GPUs and vLLM."
            )

    logger.info(f"Running stages {stages} over {sum(len(v) for v in experiments.values())} experiment(s).")
    failures: dict[str, list[str]] = {}
    for stage in stages:
        start = time.perf_counter()
        logger.info(f"===== Stage: {stage} =====")
        failed = run_stage(stage, experiments, args.reprocess)
        logger.info(f"===== Stage {stage} finished in {time.perf_counter() - start:.1f}s =====")
        if failed:
            failures[stage] = failed

    if failures:
        for stage, failed in failures.items():
            logger.error(f"Stage {stage} failed for: {failed}")
        logger.warning(f"Intermediate files kept in {config.WORK_DIR} for debugging.")
        return 1

    if not args.keep_intermediate and config.WORK_DIR.exists():
        shutil.rmtree(config.WORK_DIR)
        logger.info(f"Removed {config.WORK_DIR}.")
    logger.success("Pipeline completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
