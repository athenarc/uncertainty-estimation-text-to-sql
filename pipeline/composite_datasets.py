"""
Composite evaluation datasets, built out of the per-dataset results of an experiment group:

- ambiqt_plus_spider: the AmbiQT dataset (ambiguous-only by construction, so on its own its exec-based target is
  degenerate - no negative/"correct" class) plus Spider's correctly-translated rows folded in as the negative class.
- spider_all: every Spider-derived question pooled across the three sources that touch Spider - the original Spider
  dataset, AmbiQT's ambiguous rewrites of Spider questions, and the subset of TrustSQL whose questions originate from
  Spider.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from pipeline import config
from pipeline.overall_metrics import build_overall_metrics_sheet

# {composite name: (combined DataFrame, {source name: the rows it contributed})}
CompositeRowSets = dict[str, tuple[pd.DataFrame, dict[str, pd.DataFrame]]]


def _trustsql_spider_origin_mask(output_df: pd.DataFrame) -> pd.Series:
    """Rows of a trustsql DataFrame whose db_path comes from TrustSQL's Spider-origin split."""
    return output_df["db_path"].astype(str).str.contains(config.TRUSTSQL_SPIDER_ORIGIN_RE, regex=True, na=False)


def build_composite_row_sets(
    dataset_results: dict[str, pd.DataFrame], composites: Optional[list[str]] = None,
) -> CompositeRowSets:
    """
    Combines the per-dataset DataFrames of one experiment group into the requested composite datasets (default: all of
    them).

    dataset_results maps each experiment's "dataset" name ("ambiqt", "spider", "trustsql") to its DataFrame. A
    composite whose prerequisite datasets are missing (not configured for the group, or their computation failed) is
    skipped.
    """
    composites = composites if composites is not None else [config.AMBIQT_PLUS_SPIDER, config.SPIDER_ALL]
    ambiqt_df = dataset_results.get("ambiqt")
    spider_df = dataset_results.get("spider")
    trustsql_df = dataset_results.get("trustsql")

    row_sets: CompositeRowSets = {}

    if config.AMBIQT_PLUS_SPIDER in composites:
        if ambiqt_df is not None and spider_df is not None:
            combined = pd.concat([ambiqt_df, spider_df], ignore_index=True, sort=False)
            row_sets[config.AMBIQT_PLUS_SPIDER] = (combined, {"ambiqt": ambiqt_df, "spider(correct)": spider_df})
        else:
            logger.info(f"ambiqt or spider dataset missing - skipping {config.AMBIQT_PLUS_SPIDER}.")

    if config.SPIDER_ALL in composites:
        if spider_df is not None and ambiqt_df is not None and trustsql_df is not None:
            trustsql_spider = trustsql_df[_trustsql_spider_origin_mask(trustsql_df)]
            sources = {"spider": spider_df, "ambiqt": ambiqt_df, "trustsql(spider-origin)": trustsql_spider}
            combined = pd.concat(list(sources.values()), ignore_index=True, sort=False)

            # All the ambiqt rows are unique, but the answerable part of trustsql and spider can contain duplicates:
            # remove them (rows whose gold sql_query is a list are not hashable and are kept as they are).
            is_list_valued = combined["sql_query"].map(lambda value: isinstance(value, list))
            deduped_non_list = combined[~is_list_valued].drop_duplicates(subset=["question", "sql_query"], keep="first")
            combined = pd.concat([deduped_non_list, combined[is_list_valued]], ignore_index=True, sort=False)
            row_sets[config.SPIDER_ALL] = (combined, sources)
        else:
            logger.info(f"spider, ambiqt or trustsql dataset missing - skipping {config.SPIDER_ALL}.")

    return row_sets


def _composite_names_for(group: str) -> list[str]:
    """The composites built for an experiment group."""
    names = [config.AMBIQT_PLUS_SPIDER]
    if group in config.SPIDER_ALL_GROUPS:
        names.append(config.SPIDER_ALL)
    return names


def build_composite_datasets(output_dir: Path, group: str, dataset_results: dict[str, pd.DataFrame]) -> None:
    """
    Builds the composite datasets of one experiment group out of its per-dataset uncertainties DataFrames, recomputes
    their overall metrics over the combined rows, and saves each as `{composite}_{group}_uncertainties.xlsx`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (combined_df, sources) in build_composite_row_sets(dataset_results, _composite_names_for(group)).items():
        uncertainty_columns = [
            column for column in combined_df.columns
            if column not in config.NON_METHOD_COLUMNS and not column.endswith(config.NON_METHOD_COLUMN_SUFFIXES)
        ]
        overall_metrics_df = build_overall_metrics_sheet(combined_df, uncertainty_columns)
        files_used_df = pd.DataFrame([
            {"source_dataset": dataset, "n_rows_contributed": len(df)} for dataset, df in sources.items()
        ])

        output_path = output_dir / f"{name}_{group}_uncertainties.xlsx"
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            combined_df.to_excel(writer, sheet_name="results", index=False)
            overall_metrics_df.to_excel(writer, sheet_name="overall_metrics", index=False)
            files_used_df.to_excel(writer, sheet_name="files_used", index=False)

        logger.info(f"[{group}/{name}] Saved composite dataset ({len(combined_df)} rows) to {output_path.name}")
