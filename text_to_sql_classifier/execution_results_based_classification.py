import math

import pandas as pd


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip()


def _row_should_be_rejected(row: pd.Series) -> bool:
    """Executes row['predicted_sql'] against row['db_path'] and returns True if it errors/times out or
    returns an empty result set."""
    from metrics.utils.execution_accuracy_calculator import _get_results
    from text_to_sql.sqlite_db import DatabaseSqlite

    sql = row.get("predicted_sql")
    if _is_missing(sql):
        return True

    try:
        results, _ = _get_results(
            sql=str(sql),
            db=DatabaseSqlite(str(row["db_path"])),
            query_type="prediction",
        )
    except Exception:
        return True

    return len(results) == 0


def classification(results_df: pd.DataFrame) -> list[bool]:
    """
    Returns a boolean array with the results of the classification based on the execution results. If there is an
    execution error or the results are empty returns False, which means that the point is rejected.

    Args:
        results_df: a DataFrame with one row per prediction. If present, the "exec_error" column and the "empty_results" column are used directly. For rows where
          either column is missing, the "predicted_sql" query is executed against "db_path" to determine
          whether it errors or its results are empty.

    Returns:
        A boolean list, one per row of results_df. True means the point is accepted (the query executed
        successfully and returned a non-empty result set), False means it is rejected.
    """
    has_error_col = "exec_error" in results_df.columns
    has_empty_col = "empty_results" in results_df.columns

    # If empty_results is missing the dataframe must then include predicted_sql and db_path.
    if not has_empty_col and not ("predicted_sql" in results_df.columns and "db_path" in results_df.columns):
        raise ValueError("The dataframe must include the `predicted_sql` and `db_path` columns")

    accepted = []
    for _, row in results_df.iterrows():
        if has_error_col and not _is_missing(row.get("exec_error")):
            accepted.append(False)
            continue

        if has_empty_col:
            accepted.append(not bool(row["empty_results"]))
            continue

        # Missing exec_error and/or empty_results for this row - execute the query to find out.
        accepted.append(not _row_should_be_rejected(row))

    return accepted