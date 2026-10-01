import itertools
import random
import threading
import time
from collections import Counter
from typing import Literal, Union

import pandas as pd

from metrics.utils.process_sql import exists_order_by
from metrics.utils.query_execution_cache import cached_query_execution, db_key_for
from metrics.utils.worker_pool import run_in_worker
from text_to_sql.sqlite_db import DatabaseSqlite
from loguru import logger


def _rename_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames duplicate column names to make them unique by appending suffixes.
    For example: ['name', 'age', 'name'] becomes ['name', 'age', 'name_2']
    
    Args:
        df: DataFrame with potentially duplicate column names
        
    Returns:
        DataFrame with unique column names
    """
    df = df.copy()
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    
    for col in cols:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            new_cols.append(col)
    
    df.columns = new_cols
    return df



def exponential_backoff_query_execution(sql: str, db: DatabaseSqlite) -> pd.DataFrame:
    """
    Executes the given query in the given db, transparently caching (and persisting to
    disk immediately, see metrics.utils.query_execution_cache) successful results so the
    same (db, sql) pair is never re-executed - within this run or a later one.

    Args:
        sql (str): The query to execute.
        db (Database): The database upon which the query will be executed.

    Returns:
        The Dataframe with the results and the execution time of the query.
    """
    return cached_query_execution(sql, db_key_for(db), lambda: _exponential_backoff_query_execution_uncached(sql, db))


def _exponential_backoff_query_execution_uncached(sql: str, db: DatabaseSqlite) -> pd.DataFrame:
    """
    Executes the given query in the given db with exponential backoff in case the database is in recovery mode.
    The waiting intervals are: 16, 64, 180, 600 seconds

    Args:
        sql (str): The query to execute.
        db (Database): The database upon which the query will be executed.

    Returns:
        The Dataframe with the results and the execution time of the query.
    """
    wait_intervals = [16, 64, 180, 600]
    for wait_interval in wait_intervals:

        if sql == "SELECT T1.player_name FROM Player AS T1 INNER JOIN Player_Attributes":
            return {"error": "The query is problematic. The execution accuracy will be set to 0."}

        # If there is a bind parameter in the sql do not execute the query and return an error message
        # if ":" in sql:
        #     return {"error": "The query contains bind parameters. The execution accuracy will be set to 0."}

        result = db.execute(sql=sql, limit=-1)
        if isinstance(result, pd.DataFrame):
            return result
        elif (
            "error" in result
            and "the database system is in recovery mode" in result["error"]
        ):
            logger.warning(
                f"The database is in recovery mode. Waiting {wait_interval} seconds before trying again."
            )
            time.sleep(wait_interval)
        else:
            return result


DEFAULT_QUERY_TIMEOUT_SECONDS = 3 * 60  # 3 minutes


def _execute_sqlite_query_worker(db_path: str, sql: str):
    """Picklable entry point for run_in_worker: reconnects to the sqlite file
    from scratch (a live DatabaseSqlite/engine isn't picklable) and runs the
    query. Executed inside the isolated worker process, not the caller."""
    return exponential_backoff_query_execution(sql=sql, db=DatabaseSqlite(db_path))


def _get_results(
    sql: str,
    db: DatabaseSqlite,
    query_type: Literal["reference", "prediction"],
    timeout_seconds: int = DEFAULT_QUERY_TIMEOUT_SECONDS,
    isolate: bool = True,
) -> (pd.DataFrame, float):
    """
    Executes the given query in the given db with a timeout.

    Args:
        sql (str): The query to execute.
        db (Database): The database upon which the query will be executed.
        query_type (str): The type of query to execute. THe possible options are: 'reference', 'prediction'.
            This parameter is used for the messages in case of an error.
        timeout_seconds (int): Maximum time to allow the query to run before giving up.
        isolate (bool): When True (default) and db is a DatabaseSqlite, the query runs in
            a persistent worker process (see metrics.utils.worker_pool) so a query that
            hangs, crashes, or gets OOM-killed cannot take the caller down with it (this is
            what causes a "Process finished with exit code 137 (SIGKILL)" crash) - only the
            worker is lost, and it is restarted transparently on the next call. Set to False
            when the caller is already running inside such a worker process itself, to avoid
            pointless nested process-spawning. Non-sqlite databases aren't picklable for
            worker isolation and always fall back to a thread-based timeout.

    Returns:
        The Dataframe with the results and the execution time of the query.

    Raises:
        TimeoutError: If the query execution exceeds timeout_seconds.
        RuntimeError: If the worker process executing the query was killed (e.g. OOM/SIGKILL).
        SyntaxError: If there is an error during query execution.
    """
    start_time = time.time()
    db_path = db.database if isolate and isinstance(db, DatabaseSqlite) else None

    if db_path is not None:
        result = run_in_worker(_execute_sqlite_query_worker, db_path, sql, timeout=timeout_seconds)
    else:
        result_container = {"result": None, "exception": None}

        def execute_query():
            try:
                result_container["result"] = exponential_backoff_query_execution(sql=sql, db=db)
            except Exception as e:
                result_container["exception"] = e

        # Create and start the execution thread
        execution_thread = threading.Thread(target=execute_query, daemon=False)
        execution_thread.start()

        # Wait for the thread to complete with timeout
        execution_thread.join(timeout=timeout_seconds)

        # Check if thread is still alive (timeout occurred)
        if execution_thread.is_alive():
            logger.error(
                f"Timeout occurred while executing the {query_type} query. "
                f"Query: {sql} | Timeout: {timeout_seconds}s"
            )
            raise TimeoutError(f"Query execution exceeded {timeout_seconds} seconds")

        # Check for exceptions during execution
        if result_container["exception"] is not None:
            raise result_container["exception"]

        result = result_container["result"]

    exec_time = time.time() - start_time

    if "error" in result:
        logger.warning(
            f"There was an error while executing the {query_type} query {sql}! The execution accuracy will be set "
            f"to 0."
        )
        raise SyntaxError(result["error"])

    return result, exec_time


def _constrain_mappings(
    mappings: dict, row: list, results: Union[pd.DataFrame, list]
) -> None:
    """
    Reduce the given mappings based on the constraints in order the given row to be equal with a result row.
    Args:
        mappings: A dictionary with the keys the row columns' indexes and values the results columns' indexes.
        row: A list of column values
        results: A dataframe or a list with which the row is compared.
    """
    row_column_num = len(row)

    if isinstance(results, pd.DataFrame):
        for i in range(row_column_num):
            i_mappings = mappings[i].copy()
            for j in i_mappings:
                if row[i] not in results[j].values:
                    mappings[i].remove(j)
    else:  # results is a list (order_matters=True case)
        results_len = len(results)
        for i in range(row_column_num):
            i_mappings = mappings[i].copy()
            for j in i_mappings:
                # Check if j is a valid index for results
                if j >= results_len:
                    mappings[i].remove(j)
                    continue

                # REMOVED STRICT TYPE CHECK - rely on value comparison
                # Add try-except for comparison issues if necessary
                try:
                    if row[i] != results[j]:
                        mappings[i].remove(j)
                except TypeError:  # Handle potential comparison errors between types
                    mappings[i].remove(j)


def _valid_final_mapping(possible_mappings: dict) -> bool:
    """
    Checks if a mapping dictionary is valid. A mapping is considered valid if all keys are mapped to 1 or 0 values and
    no value exists more than one times.
    """
    if not all([len(possible_mappings[i]) <= 1 for i in range(len(possible_mappings))]):
        return False

    # Check that there are no duplicates except for None
    values_count = Counter(
        list([value[0] if len(value) else None for value in possible_mappings.values()])
    )
    if None in values_count.keys():
        values_count.pop(None)

    if any(value_count > 1 for value_count in values_count.values()):
        return False

    return True


def _constrain_columns_mappings(
    result1: pd.DataFrame, result2: pd.DataFrame, order_matters: bool
) -> dict:
    """
    Creates a dictionary with the mapping of columns from result1 with result2.
    """

    # Create all possible mappings
    possible_mappings = {
        i: [j for j in range(result2.shape[1])] for i in range(result1.shape[1])
    }
    iterations = 0
    while not _valid_final_mapping(possible_mappings) and iterations < 10:
        stable_row_idx = random.randint(0, result1.shape[0] - 1)

        if order_matters:
            _constrain_mappings(
                possible_mappings,
                result1.iloc[stable_row_idx].tolist(),
                result2.iloc[stable_row_idx].tolist(),
            )

        else:
            _constrain_mappings(
                possible_mappings, result1.iloc[stable_row_idx].tolist(), result2
            )

        iterations += 1
    return possible_mappings


def _map_results(
    result1: pd.DataFrame, result2: pd.DataFrame, mappings: dict
) -> (pd.DataFrame, pd.DataFrame):
    # Remove columns that are not mapped
    mappings = {k: v for k, v in mappings.items() if v is not None}

    result1 = result1[list(mappings.keys())]
    result1 = result1.rename(columns=mappings)

    if len(result2.columns) != len(result1.columns):
        result2 = result2[list(mappings.values())]

    return result1, result2


def _valid_column_mapping(comb: dict) -> bool:
    # If the combination contains only one column and it is None or the combination contain only 1 value
    if (len(comb) == 1 and list(comb.values())[0] is None) or set(comb.values()) == {
        None
    }:
        return False

    # Check that each column is used at most once in the mapping values
    value_count = Counter(list(comb.values()))
    value_count.pop(None, None)

    if sum(value_count.values()) != len(value_count.values()):
        return False

    return True


def _get_valid_column_mappings(column_possible_mappings: dict) -> list[dict]:
    """Returns the column mapping combinations based on which 2 dataframes can be compared"""

    # Add None as possible mapping value to all values
    column_possible_mappings = {
        k: v + [None] for k, v in column_possible_mappings.items()
    }

    # Get all permutations
    columns_combinations = (
        list(itertools.product(*column_possible_mappings.values()))
        if len(column_possible_mappings) > 1
        else list(column_possible_mappings.values())
    )
    column_mapping_combs = [
        {
            k: v
            for k, v in zip(
                list(column_possible_mappings.keys()), list(columns_combination)
            )
        }
        for columns_combination in columns_combinations
    ]

    # Remove invalid combinations
    column_mapping_combs = [
        column_valid_comb
        for column_valid_comb in column_mapping_combs
        if _valid_column_mapping(column_valid_comb)
    ]

    # Order combinations with descending number of None values
    sorted_column_mapping_combs = sorted(
        column_mapping_combs, key=lambda x: Counter(x.values())[None]
    )

    return sorted_column_mapping_combs


def _dataframes_approx_equal(
    df1: pd.DataFrame, df2: pd.DataFrame, atol: float = 1e-5, rtol: float = 1e-5
) -> bool:
    """Check if two DataFrames are approximately equal within given tolerances."""
    try:
        pd.testing.assert_frame_equal(
            df1,
            df2,
            check_exact=False,
            atol=atol,
            rtol=rtol,
            check_dtype=False,
            check_column_type=False,
        )
        return True
    except AssertionError:
        return False


def _try_pivot_and_match(
    df1: pd.DataFrame, df2: pd.DataFrame, order_matters: bool
) -> (pd.DataFrame, pd.DataFrame):
    """
    Tries to reshape one dataframe to match the other by attempting both pivot (long-to-wide)
    and melt (wide-to-long) transformations.
    """
    if df1.shape == df2.shape:
        return df1, df2

    if df1.shape[0] == 1 and df2.shape[1] == 1 and df1.shape[1] == df2.shape[0]:
        df2_t = df2.T.reset_index(drop=True)
        df2_t.columns = df2.index
        return df1, df2_t
    if df2.shape[0] == 1 and df1.shape[1] == 1 and df2.shape[1] == df1.shape[0]:
        df1_t = df1.T.reset_index(drop=True)
        df1_t.columns = df1.index
        return df1_t, df2

    if df1.shape[0] == df2.shape[0]:
        return df1, df2

    if df1.shape[0] > df2.shape[0]:
        df_long, df_wide = df1.copy(), df2.copy()
        was_df1_long = True
    else:
        df_long, df_wide = df2.copy(), df1.copy()
        was_df1_long = False

    # try 1: pivot the long dataframe to match the WIDE one.
    if not isinstance(df_long.index, pd.RangeIndex):
        df_long.reset_index(inplace=True)
    numeric_cols_long = df_long.select_dtypes(include="number").columns.tolist()
    non_numeric_cols_long = df_long.select_dtypes(exclude="number").columns.tolist()

    if len(non_numeric_cols_long) >= 2 and numeric_cols_long:
        for pivot_col in non_numeric_cols_long:
            id_vars = [c for c in non_numeric_cols_long if c != pivot_col]
            for values_col in numeric_cols_long:
                try:
                    pivoted_df = df_long.pivot(
                        index=id_vars, columns=pivot_col, values=values_col
                    )
                    pivoted_df.reset_index(inplace=True)
                    pivoted_df.columns.name = None
                    if pivoted_df.shape[0] == df_wide.shape[0]:
                        pivoted_df_temp = pivoted_df.copy()
                        df_wide_temp = df_wide.copy()
                        pivoted_df_temp.columns = range(pivoted_df_temp.shape[1])
                        df_wide_temp.columns = range(df_wide_temp.shape[1])
                        mappings = _constrain_columns_mappings(
                            df_wide_temp, pivoted_df_temp, order_matters
                        )
                        if _get_valid_column_mappings(mappings):
                            return (
                                (pivoted_df, df_wide)
                                if was_df1_long
                                else (df_wide, pivoted_df)
                            )
                except Exception:
                    continue

    # try 2: melt the wide dataframe to match the long one.
    if not isinstance(df_wide.index, pd.RangeIndex):
        df_wide.reset_index(inplace=True)
    non_numeric_cols_wide = df_wide.select_dtypes(exclude="number").columns.tolist()

    # Iterate through all possible combinations of non-numeric columns to use as identifiers.
    for i in range(len(non_numeric_cols_wide) + 1):
        for id_vars_tuple in itertools.combinations(non_numeric_cols_wide, i):
            id_vars = list(id_vars_tuple)
            try:
                melted_df = pd.melt(df_wide, id_vars=id_vars if id_vars else None)

                # If the melt results in the same number of rows as the long df, itsa candidate.
                if melted_df.shape[0] == df_long.shape[0]:
                    melted_df_temp = melted_df.copy()
                    df_long_temp = df_long.copy()
                    melted_df_temp.columns = range(melted_df_temp.shape[1])
                    df_long_temp.columns = range(df_long_temp.shape[1])
                    mappings = _constrain_columns_mappings(
                        df_long_temp, melted_df_temp, order_matters
                    )
                    if _get_valid_column_mappings(mappings):
                        if was_df1_long:
                            return (df_long, melted_df)
                        else:
                            return (melted_df, df_long)
            except Exception:
                continue

    return df1, df2


def _results_comparison(
    result1: pd.DataFrame,
    result2: pd.DataFrame,
    order_matters: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Literal[
    "equal", "columns_subset", "columns_superset", "columns_intersect", "different"
]:
    # Rename duplicate columns to make them unique from the start
    result1 = _rename_duplicate_columns(result1)
    result2 = _rename_duplicate_columns(result2)
    
    result1 = result1.map(lambda x: x.lower() if isinstance(x, str) else x)
    result2 = result2.map(lambda x: x.lower() if isinstance(x, str) else x)

    # result1, result2 = _try_pivot_and_match(result1, result2, order_matters)
    if result1.shape[0] == 0 and result2.shape[0] == 0:
        return "equal"

    if result1.shape[0] != result2.shape[0]:
        return "different"
    for df in (result1, result2):
        for col_idx in range(df.shape[1]):
            col = df.columns[col_idx]
            # if categorical, cast to str so assignment of ints works
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype(str)
            orig = df.iloc[:, col_idx]
            # std = _try_convert_series_months_to_numeric(orig)
            # if std is not orig:
            #     df.iloc[:, col_idx] = std

    # Remove column names
    result1.columns = [i for i in range(result1.shape[1])]
    result2.columns = [i for i in range(result2.shape[1])]
    result1 = result1.reset_index(drop=True)
    result2 = result2.reset_index(drop=True)
    # Compare results
    if _dataframes_approx_equal(result1, result2, atol, rtol):
        return "equal"

    columns_num_difference = len(result1.columns) - len(result2.columns)

    if columns_num_difference > 0:
        columns_mappings = _constrain_columns_mappings(result1, result2, order_matters)
        reverse = False
    else:
        columns_mappings = _constrain_columns_mappings(result2, result1, order_matters)
        reverse = True

    for column_mapping_comb in _get_valid_column_mappings(columns_mappings):
        # Map results
        if not reverse:
            result1_edited, result2_edited = _map_results(
                result1, result2, column_mapping_comb
            )
        else:
            result2_edited, result1_edited = _map_results(
                result2, result1, column_mapping_comb
            )

        # Order the results to compare them
        if not order_matters:
            columns = list(result1_edited.columns)
            result1_edited = result1_edited.sort_values(by=columns).reset_index(
                drop=True
            )
            result2_edited = result2_edited.sort_values(by=columns).reset_index(
                drop=True
            )

        none_mappings_num = list(column_mapping_comb.values()).count(None)
        if _dataframes_approx_equal(result1_edited, result2_edited, atol, rtol):
            if none_mappings_num > 0:
                if columns_num_difference != 0:
                    return "columns_superset" if not reverse else "columns_subset"
                else:
                    return "columns_intersect"
            else:
                return "equal"

    return "different"


def exec_evaluator(db_name: str, pred: str, target: Union[str, list[str]], isolate: bool = True) -> dict:
    """
    Returns the execution accuracy result for the given prediction and target sql queries.
    Args:
        db_name: The name of the database upon which the queries will run. It can be one of the 3 cases below:
            * Path of a sqlite database ending in .sqlite or .db
            * A hosted database such as "fc4eosc". For available databases check the utils_configs component
            * A hosted database with a specified schema such as "fc4eosc.fc4eosc_subset"
        pred: The predicted sql query.
        target: The target sql query, or a list of valid target queries. When a list is given the prediction
            is considered correct if it matches any one of them.
        isolate: Passed through to _get_results - run each query in an isolated worker process
            (protects the caller from a query that hangs/crashes/gets OOM-killed). Set to False
            when this call is already running inside such a worker process (e.g. called from
            metrics.execution_accuracy's own worker), to avoid nested process-spawning.
    Returns: A dictionary with the execution accuracy results. (e.g., {"exec": 0, "exec-only_common_result_columns": 1})
    """

    # Connect to database
    if db_name.endswith(".sqlite") or db_name.endswith(".db"):
        db = DatabaseSqlite(db_name)
        db_dialect = "sqlite"
    else:
        raise ValueError("Not supported db type. Only sqlite databases are supported.")

    targets = target if isinstance(target, list) else [target]

    # Get the results of the predicted query
    pred_results, pred_exec_time = _get_results(
        sql=pred, db=db, query_type="prediction", isolate=isolate
    )

    best: dict | None = None
    for t in targets:
        target_results, target_exec_time = _get_results(
            sql=t, db=db, query_type="reference", isolate=isolate
        )
        order_matters = exists_order_by(t, db_dialect)
        comp = _results_comparison(pred_results, target_results, order_matters)

        candidate = {
            "exec": 1 if comp == "equal" else 0,
            "exec-only_common_result_columns": 0 if comp == "different" else 1,
            "exec-target_result_columns_subset": (
                1 if comp in ["equal", "columns_subset"] else 0
            ),
            "exec-target_result_columns_superset": (
                1 if comp in ["equal", "columns_superset"] else 0
            ),
            "target_exec_time": target_exec_time,
            "pred_exec_time": pred_exec_time,
        }

        if best is None or candidate["exec"] > best["exec"]:
            best = candidate

        if best["exec"] == 1:
            break

    return best
