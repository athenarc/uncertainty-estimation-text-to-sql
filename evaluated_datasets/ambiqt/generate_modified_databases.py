#!/usr/bin/env python3
"""
Generate modified SQLite databases for all four AmbiQT benchmark types.

Each type gets its own directory under storage/modified-databases/:
  col-synonyms  – original tables with synonym columns added
  tbl-synonyms  – original tables with synonym tables added
  tbl-split     – original tables plus a split table (table_col)
  tbl-agg       – original tables plus a precomputed aggregate table

Run from the repo root:
  python evaluated_datasets/ambiqt/storage/generate_modified_databases.py
"""
import json
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

STORAGE = Path(__file__).parent
DB_ROOT = STORAGE / "db-content" / "database"
MODIFIED_ROOT = STORAGE / "modified-databases"
BENCHMARK = STORAGE / "benchmark"

SPIDER_TABLES = Path(__file__).parents[2] / "spider" / "storage" / "tables.json"

AGG_FUNCS = {"sum_": "SUM", "avg_": "AVG", "min_": "MIN", "max_": "MAX", "count_": "COUNT"}


# ── helpers ───────────────────────────────────────────────────────────────────

def copy_db(db_id: str, dest_dir: Path) -> Path:
    src = DB_ROOT / db_id / f"{db_id}.sqlite"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{db_id}.sqlite"
    shutil.copy2(src, dest)
    return dest


def table_cols(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    """Return [(name, type), ...] via PRAGMA table_info."""
    return [(r[1], r[2]) for r in conn.execute(f"PRAGMA table_info('{table}')").fetchall()]


def existing_tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0].lower() for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}


def parse_agg(col: str) -> tuple[str, str] | None:
    """'avg_foo' → ('AVG', 'foo'), 'number' → ('COUNT', '*'), else None."""
    cl = col.lower()
    if cl == "number":
        return "COUNT", "*"
    for prefix, fn in AGG_FUNCS.items():
        if cl.startswith(prefix):
            return fn, col[len(prefix):]
    return None


def build_fk_map(spider_tables_path: Path) -> dict[str, list[tuple]]:
    """Build {db_id: [(from_table, from_col, to_table, to_col), ...]}."""
    fk_map: dict[str, list] = {}
    for db in json.loads(spider_tables_path.read_text()):
        db_id = db["db_id"].lower()
        tnames = [t.lower() for t in db["table_names_original"]]
        col_entries = db["column_names_original"]  # [[tbl_idx, col_name], ...]
        fks = []
        for fi, ti in db.get("foreign_keys", []):
            fe = col_entries[fi]
            te = col_entries[ti]
            if fe[0] < 0 or te[0] < 0:
                continue
            fks.append((tnames[fe[0]], fe[1].lower(), tnames[te[0]], te[1].lower()))
        fk_map[db_id] = fks
    return fk_map


# ── col-synonyms ──────────────────────────────────────────────────────────────

def generate_col_synonyms() -> None:
    data = json.loads((BENCHMARK / "col-synonyms" / "validation.json").read_text())

    # Merge all extra_maps per db_id
    db_mods: dict[str, dict[str, dict[str, set]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for entry in data:
        for tbl, col_map in entry["extra_map"].items():
            for orig_col, synonyms in col_map.items():
                for syn in synonyms:
                    db_mods[entry["db_id"]][tbl.lower()][orig_col.lower()].add(syn.lower())

    for db_id, tbl_map in db_mods.items():
        dest = copy_db(db_id, MODIFIED_ROOT / "col-synonyms" / db_id)
        conn = sqlite3.connect(dest)
        for tbl, col_map in tbl_map.items():
            existing = {c.lower() for c, _ in table_cols(conn, tbl)}
            for orig_col, synonyms in col_map.items():
                col_type = next(
                    (t for c, t in table_cols(conn, tbl) if c.lower() == orig_col), "TEXT"
                )
                for syn in synonyms:
                    if syn in existing:
                        continue
                    conn.execute(f'ALTER TABLE "{tbl}" ADD COLUMN "{syn}" {col_type}')
                    conn.execute(f'UPDATE "{tbl}" SET "{syn}" = "{orig_col}"')
                    existing.add(syn)
        conn.commit()
        conn.close()
        print(f"  col-synonyms: {db_id}")


# ── tbl-synonyms ──────────────────────────────────────────────────────────────

def generate_tbl_synonyms() -> None:
    data = json.loads((BENCHMARK / "tbl-synonyms" / "validation.json").read_text())

    db_mods: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for entry in data:
        for orig_tbl, synonyms in entry["extra_table_map"].items():
            for syn in synonyms:
                db_mods[entry["db_id"]][orig_tbl.lower()].add(syn.lower())

    for db_id, tbl_map in db_mods.items():
        dest = copy_db(db_id, MODIFIED_ROOT / "tbl-synonyms" / db_id)
        conn = sqlite3.connect(dest)
        exists = existing_tables(conn)
        for orig_tbl, synonyms in tbl_map.items():
            for syn in synonyms:
                if syn in exists:
                    continue
                conn.execute(f'CREATE TABLE "{syn}" AS SELECT * FROM "{orig_tbl}"')
                exists.add(syn)
        conn.commit()
        conn.close()
        print(f"  tbl-synonyms: {db_id}")


# ── tbl-split ─────────────────────────────────────────────────────────────────

def generate_tbl_split() -> None:
    data = json.loads((BENCHMARK / "tbl-split" / "validation.json").read_text())

    # (table, split_col, pkey) per db_id
    db_mods: dict[str, set] = defaultdict(set)
    for entry in data:
        for tbl, split_col in entry["split_map"].items():
            pkey = entry["primary_key"].get(tbl)
            if pkey:
                db_mods[entry["db_id"]].add((tbl.lower(), split_col.lower(), pkey.lower()))

    for db_id, mods in db_mods.items():
        dest = copy_db(db_id, MODIFIED_ROOT / "tbl-split" / db_id)
        conn = sqlite3.connect(dest)
        exists = existing_tables(conn)
        for tbl, split_col, pkey in mods:
            new_tbl = f"{tbl}_{split_col}"
            if new_tbl in exists:
                continue
            conn.execute(
                f"CREATE TABLE {new_tbl} AS SELECT {pkey}, {split_col} FROM {tbl}"
            )
            exists.add(new_tbl)
        conn.commit()
        conn.close()
        print(f"  tbl-split: {db_id}")


# ── tbl-agg ───────────────────────────────────────────────────────────────────

def generate_tbl_agg(fk_map: dict) -> None:
    data = json.loads((BENCHMARK / "tbl-agg" / "validation.json").read_text())

    # Merge all entries for the same (db_id, new_table_name):
    # take the UNION of all_cols (different entries can reference different dimensions)
    # and keep the tables_with_pkeys from the first entry.
    merged: dict[tuple, tuple] = {}
    for entry in data:
        key = (entry["db_id"], entry["new_table_name"])
        if key not in merged:
            merged[key] = (
                list(entry["all_cols"]),
                list(entry["all_raw_cols"]),
                entry["tables_with_pkeys"],
            )
        else:
            existing_cols, existing_raw, twp = merged[key]
            for c in entry["all_cols"]:
                if c not in existing_cols:
                    existing_cols.append(c)
            for c in entry["all_raw_cols"]:
                if c not in existing_raw:
                    existing_raw.append(c)

    by_db: dict[str, list] = defaultdict(list)
    for (db_id, new_tbl), val in merged.items():
        by_db[db_id].append((new_tbl, *val))

    for db_id, new_tables in by_db.items():
        dest = copy_db(db_id, MODIFIED_ROOT / "tbl-agg" / db_id)
        conn = sqlite3.connect(dest)
        exists = existing_tables(conn)

        for new_tbl, all_cols, all_raw_cols, tables_with_pkeys in new_tables:
            if new_tbl.lower() in exists:
                continue

            src_tables = [t for t, _ in tables_with_pkeys]
            pkeys = {t.lower(): (pk.lower() if pk else None) for t, pk in tables_with_pkeys}

            # Always use a window-function approach so that every source column is
            # retained (different entries for the same table can filter/group on
            # different columns, so we can't commit to a single GROUP BY).
            agg_cols = [c for c in all_cols if parse_agg(c) is not None]

            # Build SELECT parts
            select_parts: list[str] = []
            src_tbl = src_tables[0]
            pkey = pkeys.get(src_tbl.lower())
            qualify = len(src_tables) > 1  # avoid ambiguity in multi-table joins
            seen_cols: set[str] = set()
            for col, _ in table_cols(conn, src_tbl):
                expr = f'"{src_tbl}"."{col}"' if qualify else f'"{col}"'
                select_parts.append(expr)
                seen_cols.add(col.lower())

            for ac in agg_cols:
                if ac.lower() in seen_cols:
                    continue  # already a source column with this name
                fn, src_col = parse_agg(ac)
                if src_col == "*":
                    select_parts.append(f'COUNT(*) OVER() AS "{ac}"')
                else:
                    select_parts.append(f'{fn}("{src_col}") OVER() AS "{ac}"')
                seen_cols.add(ac.lower())

            if not select_parts:
                continue

            # Build FROM / JOIN clause
            from_clause = _build_from(conn, src_tables, pkeys, fk_map.get(db_id.lower(), []))

            sql = (
                f'CREATE TABLE "{new_tbl}" AS '
                f"SELECT {', '.join(select_parts)} FROM {from_clause}"
            )

            try:
                conn.execute(sql)
                exists.add(new_tbl.lower())
            except Exception as exc:
                print(f"    ERROR {db_id}/{new_tbl}: {exc}")
                print(f"    SQL: {sql}")

        conn.commit()
        conn.close()
        print(f"  tbl-agg: {db_id}")


def _build_from(
    conn: sqlite3.Connection,
    src_tables: list[str],
    pkeys: dict[str, str | None],
    fks: list[tuple],
) -> str:
    if len(src_tables) == 1:
        return src_tables[0]

    clause = src_tables[0]
    joined: set[str] = {src_tables[0].lower()}

    for tbl in src_tables[1:]:
        tl = tbl.lower()
        if tl in joined:
            continue
        cond = None
        for ft, fc, tt, tc in fks:
            if ft in joined and tt == tl:
                cond = f"{ft}.{fc} = {tt}.{tc}"
                break
            if tt in joined and ft == tl:
                cond = f"{tt}.{tc} = {ft}.{fc}"
                break
        clause += f" JOIN {tbl} ON {cond}" if cond else f" JOIN {tbl}"
        joined.add(tl)

    return clause


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fk_map = build_fk_map(SPIDER_TABLES)

    print("Generating col-synonyms databases...")
    generate_col_synonyms()

    print("Generating tbl-synonyms databases...")
    generate_tbl_synonyms()

    print("Generating tbl-split databases...")
    generate_tbl_split()

    print("Generating tbl-agg databases...")
    generate_tbl_agg(fk_map)

    print("Done.")
