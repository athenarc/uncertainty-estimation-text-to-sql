"""
Execution Entropy uncertainty, using the implementation of RTS-SQL (not included in this repository).

The unmodified RTS-SQL file `execution_entropy.py` is downloaded into `rts_sql/` (see README.md in that folder). It
clusters the candidates by execution result, computes the semantic consistency score P(r|Q) of each cluster, the
global execution entropy H_exec(Q) and the uncertainty score U(s_i) = H_exec(Q) + λ * (-log P(r|Q)) of each candidate.

This module adapts it to this project: queries are executed with the project's sandboxed executor on a
DatabaseSqlite, and the log probabilities are plain per-token floats.

Calculation only: candidates must already be generated and scored with per-token log probabilities. Does not run any
model inference.
"""
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
from loguru import logger

from metrics.utils.execution_accuracy_calculator import _get_results
from text_to_sql.sqlite_db import DatabaseSqlite

_UPSTREAM_PATH = Path(__file__).parent / "rts_sql" / "execution_entropy.py"


@lru_cache(maxsize=None)
def _upstream() -> ModuleType:
    """Load the RTS-SQL file. It imports `SQLRunner` from RTS-SQL's own `src` package only for a type hint, so an
    empty stand-in is provided while it loads."""
    if not _UPSTREAM_PATH.exists():
        raise FileNotFoundError(
            f"{_UPSTREAM_PATH} is missing. It is RTS-SQL code that is not included in this repository: download it "
            f"as described in uncertainty_methods/logit_based/rts_sql/README.md."
        )
    stand_ins = {name: ModuleType(name) for name in ("src", "src.sql_runner")}
    stand_ins["src.sql_runner"].SQLRunner = object
    saved = {name: sys.modules.get(name) for name in stand_ins}
    sys.modules.update(stand_ins)
    try:
        spec = importlib.util.spec_from_file_location("rts_sql_execution_entropy", _UPSTREAM_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


class _DatabaseRunner:
    """The `run_query(sql) -> (result, error)` runner that RTS-SQL's code expects, executing on a DatabaseSqlite.
    Keeps the result of every query, in order."""

    def __init__(self, db: DatabaseSqlite):
        self._db = db
        self.results: List[Optional[np.ndarray]] = []

    def run_query(self, sql: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        try:
            result_df, _ = _get_results(sql, self._db, "prediction")
            result, error = result_df.values, None
        except Exception as e:
            logger.warning(f"Error executing candidate {len(self.results)}: {e}")
            result, error = None, str(e)
        self.results.append(result)
        return result, error


def calculate_execution_entropy_from_logprobs(
    candidates: List[str],
    logprobs_list: List[List[float]],
    db: DatabaseSqlite,
    lambda_weight: float = 1.0
) -> Tuple[List[float], Dict[Hashable, Dict[str, Any]]]:
    """
    Calculate execution entropy from candidates and their per-token log probabilities.

    :param candidates: List of SQL query strings (all variations for one question).
    :param logprobs_list: List of token log-probability lists, one per candidate.
    :param db: DatabaseSqlite instance for executing queries.
    :param lambda_weight: Hyperparameter λ controlling the weight of result unlikelihood.
    :return: Tuple of (uncertainty_scores, cluster_info) where:
        - uncertainty_scores: List of U(s_i) for each candidate (all 1.0 if no candidate could be executed)
        - cluster_info: Dictionary mapping result hash to cluster information
    """
    runner = _DatabaseRunner(db)
    # RTS-SQL's code takes (token, log probability) pairs; the tokens themselves are not used.
    token_logprobs_list = [[("", logprob) for logprob in logprobs] for logprobs in logprobs_list]
    scores, cluster_info = _upstream().calculate_execution_entropy_from_logprobs(
        candidates, token_logprobs_list, runner, lambda_weight
    )
    if runner.results and all(result is None for result in runner.results):
        return [1.0] * len(candidates), {}
    return scores, cluster_info
