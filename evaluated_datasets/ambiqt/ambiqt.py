from pathlib import Path
from typing import Literal
import pandas as pd

from evaluated_datasets.DatasetABC import DatasetABC


_SUBDIR_TO_TYPE = {
    "col-synonyms": "C",
    "tbl-synonyms": "T",
    "tbl-split": "J",
    "tbl-agg": "P",
}

_TYPE_TO_SUBDIR = {v: k for k, v in _SUBDIR_TO_TYPE.items()}


class AmbiQTDataset(DatasetABC):
    """
    AmbiQT benchmark: ambiguous NL-to-SQL questions derived from Spider.
    Four ambiguity types:
      C (col-synonyms)  – column-level synonym ambiguity
      T (tbl-synonyms)  – table-level synonym ambiguity
      J (tbl-split)     – join / table-split ambiguity
      P (tbl-agg)       – precomputed-aggregate ambiguity
    Each row has two valid SQL interpretations (query1, query2) stored as a list in 'query'.
    Only the validation split is used.
    """

    def __init__(self):
        benchmark_path = Path(__file__).parent / "storage" / "benchmark"
        self._modified_root = Path(__file__).parent / "storage" / "modified-databases"

        frames = []
        for subdir, ambig_type in _SUBDIR_TO_TYPE.items():
            df = pd.read_json(benchmark_path / subdir / "validation.json")
            df["ambig_type"] = ambig_type
            frames.append(df)

        data = pd.concat(frames, ignore_index=True)

        data["query"] = data.apply(lambda row: [row["query1"], row["query2"]], axis=1)
        data["db_path"] = data.apply(
            lambda row: str(
                self._modified_root
                / _TYPE_TO_SUBDIR[row["ambig_type"]]
                / row["db_id"]
                / f"{row['db_id']}.sqlite"
            ),
            axis=1,
        )

        data["is_ambiguous"] = True  # All entries in AmbiQT are ambiguous by design

        self.data = data

    def get_data(self, ambig_type: Literal["C", "T", "J", "P"] = None) -> pd.DataFrame:
        if ambig_type is not None:
            return self.data[self.data["ambig_type"] == ambig_type].reset_index(drop=True)
        return self.data
