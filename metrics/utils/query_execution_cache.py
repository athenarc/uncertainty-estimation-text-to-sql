"""Persistent (db, sql) -> result cache for query execution.

Every caller of exponential_backoff_query_execution gets this cache
automatically.

The same query is often executed many times across a run (e.g. the same gold
query compared against dozens of candidate predictions). Caching avoids
redundant execution, and persisting each new entry to disk immediately means
already-computed results survive a crash/SIGKILL instead of having to be
recomputed from scratch on the next run.

One pickle file per database, keyed by a sanitized version of the db's
connection identity (e.g. "sqlite:///path/to/db.sqlite"). Each new entry is
*appended* to that file as its own pickle record (one `pickle.dump((sql, result), f)`
call per cache miss, under an flock so concurrent writers can't interleave
their bytes) - never rewriting what's already on disk. Rewriting the whole
accumulated dict on every miss (the previous approach) is O(1) per call in
isolation but O(n^2) in total over n distinct queries against the same db,
since dict i's rewrite re-serializes all i-1 earlier entries too; for a
database queried by every row in a run (e.g. trustsql's atis/advising/mimic_iv,
each a single sqlite file shared across the whole dataset, unlike spider's
one-tiny-file-per-question layout) that dict can reach 1-2GB+, making each
individual rewrite take 15-30+ seconds. Loading tolerates a truncated/corrupt
trailing record (a crash mid-append) by simply stopping there - only that one
unfinished entry is lost, not any of the earlier ones - and also reads the
legacy single-whole-dict format transparently, since a plain dict is just
another kind of record in the stream.

A run that touches many different databases (e.g. trustsql, which spans
dozens of distinct sqlite files) would otherwise keep every one of those
databases' full result caches in memory forever - some of those caches are
1GB+ (large source tables, e.g. mimic_iv), so that unbounded growth reliably
OOM-kills the process partway through a run. `_query_cache` is therefore
bounded to MAX_CACHE_BYTES total, evicting the least-recently-used database's
in-memory cache (never the one currently being queried) once the budget is
exceeded - evicted entries are already persisted on disk, so they're simply
reloaded if that database is queried again later.
"""
import fcntl
import hashlib
import os
import pickle
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import pandas as pd
from loguru import logger

from pipeline.config import QUERY_EXECUTION_CACHE_DIR

CACHE_DIR = QUERY_EXECUTION_CACHE_DIR

# Total in-memory budget across all loaded databases' caches. This cache lives inside the
# persistent worker process (see metrics.utils.worker_pool.run_in_worker), so the real ceiling
# is WORKER_MEMORY_LIMIT_MB (6GiB by default) rather than the host's total RAM - leave enough
# headroom under that limit for the in-flight query's own DataFrame plus Python/pandas overhead,
# or the memory watchdog will kill the worker instead of just evicting a cache entry. 4GiB
# comfortably fits a single large shared-db cache (e.g. TrustSQL's atis.sqlite cache, ~2.1GB)
# under the default 6GiB worker limit; raise WORKER_MEMORY_LIMIT_MB in tandem if multiple large
# caches need to be resident at once. Overridable via env var for machines with more/less RAM.
MAX_CACHE_BYTES = int(os.environ.get("QUERY_EXECUTION_CACHE_MAX_BYTES", 4 * 1024**3))  # 4 GiB

# db_key -> {sql: result}, populated lazily as databases are first queried, ordered
# least-to-most-recently-used for eviction.
_query_cache: "OrderedDict[str, dict]" = OrderedDict()
# db_key -> approximate byte size of _query_cache[db_key], kept in sync alongside it.
_query_cache_bytes: dict[str, int] = {}


def db_key_for(db) -> str:
    return getattr(db, "connection_uri", str(id(db)))


# Longest file name stem the cache uses: file systems allow 255 bytes per name, and the database paths of some datasets
# (e.g. Ambrosia's) are longer than that once sanitized. A write that fails is treated as a failed query execution.
_MAX_CACHE_FILE_STEM = 200


def _cache_path(db_key: str) -> Path:
    safe = re.sub(r"[^\w.-]", "_", db_key)
    if len(safe) > _MAX_CACHE_FILE_STEM:
        # Keep the end of the key (it names the database) and make it unique with a hash of the whole key.
        safe = f"{safe[-(_MAX_CACHE_FILE_STEM - 17):]}_{hashlib.sha1(db_key.encode()).hexdigest()[:16]}"
    return CACHE_DIR / f"{safe}.pkl"


def _estimate_bytes(value) -> int:
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(deep=True).sum())
    return sys.getsizeof(value)


def _evict_lru(keep: str) -> None:
    """Drop least-recently-used database caches from memory (but never `keep`, the database
    currently being queried) until the total is back under MAX_CACHE_BYTES."""
    total = sum(_query_cache_bytes.values())
    for db_key in list(_query_cache.keys()):
        if total <= MAX_CACHE_BYTES:
            return
        if db_key == keep:
            continue
        total -= _query_cache_bytes.pop(db_key, 0)
        del _query_cache[db_key]
        logger.debug(f"Evicted in-memory query cache for {db_key} (LRU, over {MAX_CACHE_BYTES / 1e9:.1f}GB budget)")


def _read_cache_records(path: Path) -> dict:
    """Reads the append-only record stream at `path` into a dict.

    Each record is either a plain dict (the legacy whole-cache snapshot format -
    at most the first record in a file written before this module switched to
    per-entry appends) or a (sql_key, result) tuple (one per cache miss, in the
    order they were appended - a later record for the same sql_key, e.g. from a
    retried write, simply overwrites the earlier one).

    A trailing record can be truncated/corrupt if the process was killed mid-append;
    that's treated as "nothing useful past this point" rather than an error - only
    that one unfinished entry is lost, every earlier record loaded fine.
    """
    cache: dict = {}
    with open(path, "rb") as f:
        while True:
            try:
                record = pickle.load(f)
            except EOFError:
                break
            except Exception as e:
                logger.warning(
                    f"Stopping read of {path} at a truncated/corrupt trailing record "
                    f"(likely a crash mid-append) - earlier entries are unaffected: {e}"
                )
                break

            if isinstance(record, dict):
                cache.update(record)
            else:
                sql_key, result = record
                cache[sql_key] = result
    return cache


def _load_cache(db_key: str) -> dict:
    if db_key in _query_cache:
        _query_cache.move_to_end(db_key)
        return _query_cache[db_key]

    path = _cache_path(db_key)
    if path.exists():
        cache = _read_cache_records(path)
        size_bytes = path.stat().st_size
        logger.debug(f"Loaded query execution cache for {db_key}: {len(cache)} entries ({size_bytes / 1e6:.0f}MB)")
    else:
        cache = {}
        size_bytes = 0

    _query_cache[db_key] = cache
    _query_cache_bytes[db_key] = size_bytes
    _query_cache.move_to_end(db_key)
    _evict_lru(keep=db_key)
    return cache


def _append_cache_entry(db_key: str, sql_key: str, result: "pd.DataFrame") -> None:
    """Appends a single (sql, result) record to db_key's cache file - O(1) per call,
    instead of re-serializing the whole accumulated cache on every miss (which made
    total cache I/O cost O(n^2) over n distinct queries against the same db - see the
    module docstring). flock serializes concurrent writers (e.g. two runs touching the
    same db at once) so their appends can't interleave and corrupt the pickle stream;
    a crash mid-append can only ever corrupt its own trailing record, which
    _read_cache_records discards on the next load without affecting earlier entries."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(db_key)
    with open(path, "ab") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            pickle.dump((sql_key, result), f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def cached_query_execution(sql: str, db_key: str, execute_fn: Callable[[], "pd.DataFrame | dict"]) -> "pd.DataFrame | dict":
    """Return the cached result for (db_key, sql); on a miss, run execute_fn(),
    cache the result, persist it to disk immediately, and return it.

    Only successful DataFrame results are cached - errors (e.g. malformed
    candidate SQL, or a transient "database is in recovery mode") propagate
    and are retried on the next call rather than being cached.
    """
    cache = _load_cache(db_key)
    sql_key = sql.strip()
    if sql_key in cache:
        return cache[sql_key]

    result = execute_fn()

    if isinstance(result, pd.DataFrame):
        cache[sql_key] = result
        _query_cache_bytes[db_key] = _query_cache_bytes.get(db_key, 0) + _estimate_bytes(result)
        _append_cache_entry(db_key, sql_key, result)
        _evict_lru(keep=db_key)

    return result
