"""
Adapter around the XiYan M-Schema code (not included in this repository).

The unmodified M-Schema files `m_schema.py`, `schema_engine.py` and `utils.py` are downloaded into `xiyan_utils/`
(see README.md in this folder). They import each other with bare module names, so they are imported with that folder
temporarily on `sys.path`. `SchemaEngine` below adapts the schema to SQLite databases.
"""
import contextlib
import importlib
import io
import sys
from contextlib import contextmanager
from pathlib import Path

_UPSTREAM_DIR = Path(__file__).parent / "xiyan_utils"
_UPSTREAM_MODULES = ("utils", "m_schema", "schema_engine")


@contextmanager
def _upstream_on_path():
    """Make `import utils / m_schema / schema_engine` resolve to the downloaded files, without leaving these generic
    module names behind in sys.modules."""
    if not (_UPSTREAM_DIR / "schema_engine.py").exists():
        raise FileNotFoundError(
            f"{_UPSTREAM_DIR / 'schema_engine.py'} is missing. It is M-Schema code that is not included in this "
            f"repository: download it as described in text_to_sql/utils/README.md."
        )
    saved = {name: sys.modules.pop(name) for name in _UPSTREAM_MODULES if name in sys.modules}
    sys.path.insert(0, str(_UPSTREAM_DIR))
    try:
        yield
    finally:
        sys.path.remove(str(_UPSTREAM_DIR))
        for name in _UPSTREAM_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


with _upstream_on_path():
    _schema_engine = importlib.import_module("schema_engine")


class SchemaEngine(_schema_engine.SchemaEngine):
    """M-Schema SchemaEngine for SQLite: tables are named without the `main.` schema prefix, foreign keys carry no
    schema, and the debug prints of init_mschema are silenced."""

    def init_mschema(self):
        with contextlib.redirect_stdout(io.StringIO()):
            super().init_mschema()
        if self._engine.dialect.name != "sqlite":
            return
        mschema = self._mschema
        prefix = "main."
        mschema.tables = {
            name.removeprefix(prefix): table for name, table in mschema.tables.items()
        }
        for foreign_key in mschema.foreign_keys:
            # [table, column, referred schema, referred table, referred column]; to_mschema only lists the foreign
            # keys whose referred schema equals the M-Schema's schema, which is None for SQLite.
            foreign_key[0] = foreign_key[0].removeprefix(prefix)
            foreign_key[2] = None
