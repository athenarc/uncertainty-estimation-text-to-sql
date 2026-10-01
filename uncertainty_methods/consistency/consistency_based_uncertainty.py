"""Consistency-based uncertainty: calculation only.

Given an initial SQL prediction and a set of SQL variations (e.g. from
temperature or input-perturbation sampling), computes an uncertainty score
based on how (dis)similar the variations are to the initial prediction.
Does not run any model inference.
"""
from typing import Literal, Optional

import pandas as pd
from loguru import logger

from metrics.utils.execution_accuracy_calculator import _results_comparison, _get_results
from text_to_sql.sqlite_db import DatabaseSqlite


def _execute_query(query: str, db: DatabaseSqlite) -> Optional[pd.DataFrame]:
    """
    Executes a single SQL query against the given database, returning its results.

    Returns None if execution fails, so failures can be handled uniformly at
    comparison time instead of aborting the whole consistency calculation.
    """
    try:
        result, _ = _get_results(query, db, "reference")
        return result
    except Exception as e:
        return None


def _execute_queries(queries: list[str], db: DatabaseSqlite) -> dict[str, Optional[pd.DataFrame]]:
    """
    Executes each distinct query in ``queries`` exactly once against ``db``.
    """
    results: dict[str, Optional[pd.DataFrame]] = {}
    for query in queries:
        if query not in results:
            results[query] = _execute_query(query, db)
    return results


def _calculate_result_similarity(
    result1: Optional[pd.DataFrame],
    result2: Optional[pd.DataFrame],
    order_matters: bool = False,
) -> int:
    """
    Calculates binary similarity between two already-executed SQL query results.

    Returns:
        1 if results are considered equal/similar, 0 otherwise
    """
    if result1 is None or result2 is None:
        return 0

    try:
        comparison = _results_comparison(result1, result2, order_matters)
        return 1 if comparison == "equal" else 0
    except Exception as e:
        logger.warning(f"Error during result similarity calculation: {e}")
        return 0


def calculate_consistency_based_uncertainty(
    initial_prediction: str,
    variations_sqls: list[str],
    db_path: Optional[str],
) -> Optional[float]:
    """
    Calculate the consistency-based uncertainty score for a given initial prediction and its variations.

    The consistency-based uncertainty is calculated as 1 - average_similarity, where average_similarity is the average
    similarity between the initial prediction and the variations.

    Args:
        initial_prediction: SQL query predicted with the main (greedy)
            generation.
        variations_sqls: SQL queries predicted by the variation runs (e.g.
            repeated temperature sampling or alternative prompts) to compare
            against ``initial_prediction``.
        db_path: path to the SQLite database the queries run against.
            Required when ``similarity_method="results_comparison"``.

    Returns None when the initial prediction is None (cannot compute consistency).
    Returns:
        uncertainty_score: A float representing the consistency-based uncertainty score.
    """
    if initial_prediction is None:
        return None

    db = DatabaseSqlite(db_path)
    query_results = _execute_queries([initial_prediction] + variations_sqls, db)
    initial_result = query_results[initial_prediction]

    # If all query_results are None
    if all(x is None for x in query_results):
        return 1

    similarity_scores = [
        _calculate_result_similarity(initial_result, query_results[variation])
        for variation in variations_sqls
    ]

    average_similarity = sum(similarity_scores) / len(similarity_scores)

    return 1 - average_similarity
