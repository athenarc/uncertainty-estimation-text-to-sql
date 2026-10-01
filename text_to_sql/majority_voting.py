import contextlib
import io
import random
from loguru import logger

from metrics.utils.execution_accuracy_calculator import _get_results, _results_comparison
from text_to_sql.sqlite_db import DatabaseSqlite


def _majority_vote(sqls: list[str | None], db_path: str) -> tuple[int, int, int, int | None]:
    """
    Clusters the given SQLs by execution-result equivalence and
    returns the majority-voted SQL index, the size of its cluster, the number of
    generations that executed successfully and the total number of clusters

    Returns:
        (predicted_sql_index, majority_count, n_clusters, n_valid)
    """
    db = DatabaseSqlite(db_path)

    # Each cluster is represented by the index (in `sqls`) of the query that formed it, and
    # contains the execution results of that query and the number of queries in that cluster
    clusters: dict[int, list] = {}
    n_valid = 0

    # Cluster the SQL queries
    for idx, sql in enumerate(sqls):
        # Get the execution results of the query
        if not sql:
            continue

        logger.disable("")
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result_df, _ = _get_results(sql, db, "prediction")
        except Exception as e:
            logger.enable("")
            continue
        finally:
            logger.enable("")

        n_valid += 1

        # Check if the execution results match any existing cluster by comparing them with the execution accuracy calculator
        cluster_found = False
        for cluster in clusters.values():
            cluster_results, _ = cluster
            # TODO check what i have to add in order matters
            if _results_comparison(result_df, cluster_results, order_matters=True) == "equal":
                cluster[1] += 1  # Increment the count of queries in this cluster
                cluster_found = True
                break

        if not cluster_found:
            # If no matching cluster is found, create a new cluster with the current SQL's index and execution results
            clusters[idx] = [result_df, 1]

    # Select the winning sql's index
    if not clusters: # If all queries failed to execute, return a random query's index from the original list
        valid_indices = [i for i, sql in enumerate(sqls) if sql is not None]
        return random.choice(valid_indices), 1, 0, n_valid
    else:
        # Get the index of the sql from the cluster with the maximum number of queries
        best_cluster_idx = max(clusters.items(), key=lambda item: item[1][1])[0]
        return best_cluster_idx, int(clusters[best_cluster_idx][1]), len(clusters), n_valid