"""Shared persistent-worker-process infrastructure for SQL query execution.

sqlite/pandas query execution runs in-process by default. A query that
balloons memory or crashes the interpreter takes the *whole* calling process
down with it (observed as exit code 137 / SIGKILL from the OS OOM killer).
Routing execution through a subprocess means only that subprocess is lost -
the caller sees a catchable RuntimeError/TimeoutError instead and can log a
failure and move on.

Any code that needs this isolation for a query should call `run_in_worker`
instead of spawning its own pool, so there is exactly one persistent worker
process shared across all callers.
"""
import ctypes
import ctypes.util
import os
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import Callable

from loguru import logger

# Per-query execution errors/timeouts logged by these modules are extremely
# common (malformed candidate SQL, slow queries, ...) and are almost always
# already surfaced by the caller as an aggregate failure count, so they are
# dropped instead of spamming the console per query.
NOISY_SQL_LOG_SOURCES = {
    "text_to_sql.sqlite_db",
    "text_to_sql.core",
    "metrics.utils.execution_accuracy_calculator",
    "uncertainty_methods.logit_based.execution_entropy_uncertainty",
    "uncertainty_methods.consistency.consistency_based_uncertainty",
}


def configure_quiet_sql_logging() -> None:
    """Drop NOISY_SQL_LOG_SOURCES log records from the default loguru sink."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, filter=lambda record: record["name"] not in NOISY_SQL_LOG_SOURCES)


# Process-wide cap on SQLite's own engine memory (sorters, temp b-trees for
# joins/GROUP BY/DISTINCT, ...) - not a row-count or query-text limit, so it
# doesn't affect query correctness. A candidate query with a pathological
# intermediate (e.g. a missing join condition) fails its own allocations past
# this cap instead of growing this process's RSS unboundedly; combined with
# PRAGMA temp_store=FILE (set per-connection in DatabaseSqlite), SQLite's
# sorter is designed to spill to disk once this limit is hit rather than
# erroring outright. Applies to every sqlite3 connection opened in this
# process for its lifetime (sqlite3_hard_heap_limit64 is process-global, not
# per-connection). Override via TEXT_TO_SQL_SQLITE_HEAP_LIMIT_MB; 0 disables.
SQLITE_HEAP_LIMIT_MB = int(os.environ.get("TEXT_TO_SQL_SQLITE_HEAP_LIMIT_MB", 512))


def set_sqlite_heap_limit() -> None:
    if SQLITE_HEAP_LIMIT_MB <= 0:
        return
    try:
        lib_path = ctypes.util.find_library("sqlite3")
        if not lib_path:
            return
        lib = ctypes.CDLL(lib_path)
        lib.sqlite3_hard_heap_limit64.restype = ctypes.c_int64
        lib.sqlite3_hard_heap_limit64.argtypes = [ctypes.c_int64]
        lib.sqlite3_hard_heap_limit64(SQLITE_HEAP_LIMIT_MB * 1024 * 1024)
    except (OSError, AttributeError):
        pass  # not available on this platform/sqlite build - no hard cap, still fine


def _worker_init() -> None:
    configure_quiet_sql_logging()
    set_sqlite_heap_limit()


# A single persistent worker process shared across all calls. Restarted only
# when the worker is killed (e.g. OOM); spawning once amortises the ~300 ms
# macOS "spawn" overhead over thousands of queries instead of paying it per row.
_executor: ProcessPoolExecutor | None = None


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=1, initializer=_worker_init)
    return _executor


class WorkerMemoryLimitExceeded(RuntimeError):
    """Raised when the worker process was pre-emptively killed for exceeding
    WORKER_MEMORY_LIMIT_MB, as opposed to being killed by the OS OOM killer."""


# A pathological query (e.g. a missing join condition producing a huge cross
# product) can grow the worker process's RSS well past what SQLITE_HEAP_LIMIT_MB
# covers (that only caps sqlite's own engine memory, not the pandas/python
# memory holding the result set) - large enough to make the OS start swapping
# or invoke its own OOM killer, which on a memory-constrained machine can
# destabilize the whole system (not just the worker) before the OS gets around
# to killing anything. The watchdog thread below polls the worker's RSS from
# the parent process and kills it itself once it crosses this cap, well
# before that happens. Default matches a 16GB machine with headroom for the
# OS and other apps; override via TEXT_TO_SQL_WORKER_MEMORY_LIMIT_MB, 0 disables.
WORKER_MEMORY_LIMIT_MB = int(os.environ.get("TEXT_TO_SQL_WORKER_MEMORY_LIMIT_MB", 6 * 1024))
_MEMORY_POLL_INTERVAL_SECONDS = 0.5


def _current_worker_pid(executor: ProcessPoolExecutor) -> int | None:
    processes = getattr(executor, "_processes", None) or {}
    return next(iter(processes), None)


# pids the watchdog has killed for exceeding WORKER_MEMORY_LIMIT_MB, so
# run_in_worker can tell "my future broke because the watchdog killed it"
# apart from "my future broke for some other reason (crash / OS OOM)".
# Plain set + GIL is enough synchronization for add/discard/contains here.
_memory_killed_pids: set[int] = set()

# One background thread for the whole process lifetime, not one per query -
# run_in_worker is the choke point for every SQL execution in the codebase
# (thousands+ calls per report), so spawning/joining a thread per call would
# add real overhead (and could block a fast call waiting to join a watchdog
# thread that was still mid-sleep). Poll from a single persistent thread instead.
_watchdog_thread: threading.Thread | None = None
_watchdog_lock = threading.Lock()


def _memory_watchdog_loop() -> None:
    import psutil

    watched_pid: int | None = None
    proc = None
    while True:
        time.sleep(_MEMORY_POLL_INTERVAL_SECONDS)
        try:
            if WORKER_MEMORY_LIMIT_MB <= 0:
                continue
            executor = _executor
            if executor is None:
                watched_pid, proc = None, None
                continue

            pid = _current_worker_pid(executor)
            if pid is None:
                watched_pid, proc = None, None
                continue
            if pid != watched_pid:
                # Only commit to watched_pid once psutil.Process() actually succeeds - on failure,
                # both must stay in sync (reset together) so the next iteration retries the lookup
                # instead of falling through to rss_mb = proc.memory_info() with a stale None proc
                # (an unguarded AttributeError there would silently kill this thread for the rest of
                # the process's lifetime, disabling worker memory protection with no indication).
                try:
                    proc = psutil.Process(pid)
                    watched_pid = pid
                except psutil.NoSuchProcess:
                    watched_pid, proc = None, None
                    continue

            try:
                rss_mb = proc.memory_info().rss / (1024 * 1024)
            except psutil.NoSuchProcess:
                watched_pid, proc = None, None
                continue

            if rss_mb > WORKER_MEMORY_LIMIT_MB:
                logger.warning(
                    f"Worker process (pid {pid}) hit {rss_mb:.0f}MB, over the "
                    f"{WORKER_MEMORY_LIMIT_MB}MB limit - killing it to protect the host."
                )
                _memory_killed_pids.add(pid)
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                watched_pid, proc = None, None
        except Exception as e:
            # This loop is the only thing standing between a runaway worker query and an OS-level
            # OOM kill of the whole process (the OS kill isn't necessarily confined to the worker -
            # see run_in_worker's docstring). Never let an unexpected error here kill the thread
            # silently - log it and keep polling instead.
            logger.warning(f"Memory watchdog loop hit an unexpected error, continuing: {e}")
            watched_pid, proc = None, None


def _ensure_watchdog_started() -> None:
    global _watchdog_thread
    if _watchdog_thread is not None or WORKER_MEMORY_LIMIT_MB <= 0:
        return
    with _watchdog_lock:
        if _watchdog_thread is not None:
            return
        try:
            import psutil  # noqa: F401 - just checking availability up front
        except ImportError:
            logger.debug("psutil not installed - worker memory watchdog disabled.")
            return
        _watchdog_thread = threading.Thread(target=_memory_watchdog_loop, daemon=True)
        _watchdog_thread.start()


def run_in_worker(fn: Callable, *args, timeout: int):
    """Run fn(*args) in the persistent worker process.

    The worker is reused across calls (and across callers - any code that needs
    SIGKILL/OOM isolation for a query can submit its own function here instead
    of spawning a second pool). If it is killed (OOM / SIGKILL) or hangs past
    `timeout`, the pool is restarted transparently and a RuntimeError /
    TimeoutError is raised so the caller can record a failure and continue.
    A background watchdog (see _memory_watchdog_loop) also pre-emptively kills
    the worker (raising WorkerMemoryLimitExceeded, a RuntimeError subclass) if
    its RSS crosses WORKER_MEMORY_LIMIT_MB, rather than waiting for the OS to
    do it.
    """
    global _executor
    _ensure_watchdog_started()
    try:
        executor = _get_executor()
        future = executor.submit(fn, *args)
        pid = _current_worker_pid(executor)
    except BrokenProcessPool:
        _executor = None
        raise RuntimeError("Worker process was killed (OOM/SIGKILL)")

    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        try:
            _executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _executor = None
        raise TimeoutError(f"Query execution timed out after {timeout}s")
    except BrokenProcessPool:
        _executor = None
        if pid is not None and pid in _memory_killed_pids:
            _memory_killed_pids.discard(pid)
            raise WorkerMemoryLimitExceeded(
                f"Worker process exceeded the {WORKER_MEMORY_LIMIT_MB}MB memory limit and was killed"
            )
        raise RuntimeError("Worker process was killed (OOM/SIGKILL)")
