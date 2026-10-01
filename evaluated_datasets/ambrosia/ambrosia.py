import ast
from pathlib import Path
from typing import Literal

import pandas as pd

from evaluated_datasets.DatasetABC import DatasetABC

class AmbrosiaDataset(DatasetABC):
    """
    AMBROSIA: a benchmark of ambiguous NL-to-SQL questions.
    Each row has an ambiguous question with multiple valid SQL interpretations.
    Splits: 'test' (evaluation) and 'few_shot_examples'.
    """

    def __init__(self):
        storage_path = Path(__file__).parent / "storage"
        raw = pd.read_csv(storage_path / "data" / "ambrosia.csv", index_col=0)

        raw["db_path"] = raw["db_file"].apply(
            lambda p: str(storage_path / p)
        )
        raw["db_id"] = raw["db_file"].apply(
            lambda p: Path(p).stem
        )
        raw.rename(columns={
            "ambig_question": "question",
            "question": "unambiguous_version",
        }, inplace=True)
        raw["query"] = raw.apply(
            lambda row: ast.literal_eval(row["ambig_queries"]) if row["is_ambiguous"] else [row["gold_queries"]],
            axis=1,
        )
        raw.drop(columns=["ambig_queries", "gold_queries"], inplace=True)

        self.data = {
            "test": raw[raw["split"] == "test"].reset_index(drop=True),
            "few_shot_examples": raw[raw["split"] == "few_shot_examples"].reset_index(drop=True),
        }

        self._db_path_map = (
            raw[["db_id", "db_path"]]
            .drop_duplicates("db_id")
            .set_index("db_id")["db_path"]
            .to_dict()
        )

    def get_data(self, split: Literal["test", "few_shot_examples"] = "test") -> pd.DataFrame:
        return self.data[split]