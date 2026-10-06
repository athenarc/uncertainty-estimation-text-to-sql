"""
Adapter around the OmniSQL schema/value-linking code (not included in this repository).

The unmodified OmniSQL files `process_dataset.py` and `build_contents_index.py` are downloaded into `omnisql_utils/`
(see README.md in this folder). This module loads them lazily and adds the few things this project needs on top:
reading the schema straight from a SQLite file and reading BM25 hits with a newer pyserini.
"""
import importlib
import importlib.util
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

_UPSTREAM_DIR = Path(__file__).parent / "omnisql_utils"

# Functions of OmniSQL's process_dataset.py that are used as they are.
_UPSTREAM_FUNCTIONS = {
    "deduplicate_dicts",
    "obtain_db_details",
    "obtain_n_grams",
    "retrieve_question_related_db_values",
    "sample_table_values",
}


@contextmanager
def _stub_if_not_importable():
    """process_dataset.py imports pyserini (needs Java) and ijson at the top, but the functions used here do not need
    them. Provide empty stand-ins while it loads when they are not installed."""
    stubbed = []

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
        stubbed.append(name)

    try:
        importlib.import_module("pyserini.search.lucene")
    except Exception:
        for name in ("pyserini", "pyserini.search"):
            stub(name)
        stub("pyserini.search.lucene", LuceneSearcher=None)
    try:
        importlib.import_module("ijson")
    except ImportError:
        stub("ijson")
    try:
        yield
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)


@lru_cache(maxsize=None)
def load_upstream(module_name: str) -> ModuleType:
    """Load `omnisql_utils/<module_name>.py` (unmodified OmniSQL code)."""
    path = _UPSTREAM_DIR / f"{module_name}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It is OmniSQL code that is not included in this repository: download it as "
            f"described in text_to_sql/utils/README.md."
        )
    spec = importlib.util.spec_from_file_location(f"omnisql_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    with _stub_if_not_importable():
        spec.loader.exec_module(module)
    return module


def __getattr__(name: str):
    if name in _UPSTREAM_FUNCTIONS:
        return getattr(load_upstream("process_dataset"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _RawHitSearcher:
    """Gives the hits of a pyserini LuceneSearcher the `.raw` attribute that OmniSQL's retrieve_relevant_hits reads
    (newer pyserini versions expose it as `lucene_document.get('raw')`)."""

    def __init__(self, searcher):
        self._searcher = searcher

    def batch_search(self, queries, qids, **kwargs):
        results = self._searcher.batch_search(queries, qids, **kwargs)
        return {
            qid: [SimpleNamespace(raw=hit.lucene_document.get("raw")) for hit in hits]
            for qid, hits in results.items()
        }


def retrieve_relevant_hits(searcher, queries) -> dict:
    """OmniSQL's retrieve_relevant_hits on a pyserini LuceneSearcher: {query: [hit dict]}."""
    return load_upstream("process_dataset").retrieve_relevant_hits(_RawHitSearcher(searcher), queries)


def _normalize_sqlite_type(raw_type: str | None) -> str:
    """Return the raw SQLite column type as-is (lowercased, stripped of size specs).

    Preserves the original vocabulary (e.g. 'integer', 'real', 'varchar') so that
    prompts generated from the SQLite fallback path stay consistent with whatever
    types the database schema actually declares.
    """
    if not raw_type:
        return "text"
    return raw_type.split("(")[0].strip().lower()


def obtain_db_info_from_sqlite(db_path: str) -> dict:
    """
    Build a Spider-format db_info dict directly from a SQLite file using PRAGMA queries.
    No tables.json required — all tables and columns are read from the database itself.

    Since we have no manual annotations, column_names and column_names_original are set
    to the same values. obtain_db_details detects this equality and omits the comment
    annotation, so only the "-- example: [...]" value hints appear in the DDL.

    Spider convention used throughout obtain_db_details:
      - column_names_original[0] == (-1, "*")  (wildcard sentinel at index 0)
      - All other entries are (table_idx, column_name), 0-indexed by table order
      - primary_keys: list of global column indices (nested list for composite PKs)
      - foreign_keys: list of (source_global_idx, target_global_idx) pairs
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY rowid;")
    table_names = [row[0] for row in cursor.fetchall()]

    # Index 0 is always the wildcard sentinel — required by obtain_db_details
    column_names_original = [(-1, "*")]
    column_names = [(-1, "*")]
    column_types = ["text"]
    primary_keys = []
    col_index_map: dict[tuple[str, str], int] = {}  # (table_lower, col_lower) -> global idx

    for table_idx, table_name in enumerate(table_names):
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        # Row format: (cid, name, type, notnull, dflt_value, pk)
        # pk > 0 means the column is part of the PK; value is its 1-indexed position in the PK
        pk_order_pairs = []
        for _, col_name, col_type, _, _, pk_pos in cursor.fetchall():
            global_idx = len(column_names_original)
            col_index_map[(table_name.lower(), col_name.lower())] = global_idx
            column_names_original.append((table_idx, col_name))
            column_names.append((table_idx, col_name))  # same as original — no annotation available
            column_types.append(_normalize_sqlite_type(col_type))
            if pk_pos > 0:
                pk_order_pairs.append((pk_pos, global_idx))

        pk_order_pairs.sort(key=lambda x: x[0])
        pk_indices = [idx for _, idx in pk_order_pairs]
        if len(pk_indices) == 1:
            primary_keys.append(pk_indices[0])
        elif len(pk_indices) > 1:
            primary_keys.append(pk_indices)  # composite PK stored as a nested list

    foreign_keys = []
    for table_name in table_names:
        cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
        # Row format: (id, seq, ref_table, from_col, to_col, on_update, on_delete, match)
        for _, _, ref_table, from_col, to_col, *_ in cursor.fetchall():
            if from_col is None or to_col is None:
                continue
            src = col_index_map.get((table_name.lower(), from_col.lower()))
            tgt = col_index_map.get((ref_table.lower(), to_col.lower()))
            if src is not None and tgt is not None:
                foreign_keys.append((src, tgt))

    cursor.close()
    conn.close()

    return {
        "db_id": os.path.splitext(os.path.basename(db_path))[0],
        "table_names_original": table_names,
        "column_names_original": column_names_original,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }
