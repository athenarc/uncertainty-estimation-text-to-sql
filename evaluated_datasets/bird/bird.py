from pathlib import Path

import pandas as pd

from evaluated_datasets.DatasetABC import DatasetABC


class BirdDataset(DatasetABC):
    def __init__(self):

        self.data = pd.read_json(Path(__file__).parent / "storage/dev.json")

        self.databases_path = Path(__file__).parent / "storage" / "dev_databases"

        self.data['db_path'] = self.data['db_id'].apply(
            lambda db_id: str(self.databases_path / db_id / f"{db_id}.sqlite")
        )
        self.data.rename(columns={"SQL": "query", "evidence": "hint"}, inplace=True)

    def get_data(self):
        return self.data

