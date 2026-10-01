
from pathlib import Path
from typing import Literal

import pandas as pd

from evaluated_datasets.DatasetABC import DatasetABC


class SpiderDataset(DatasetABC):
    def __init__(self):

        splits = {
            "train": {"data_file_name": "train_spider.json", "db_dir_name": "database"},
            "dev": {"data_file_name": "dev.json", "db_dir_name": "database"},
            "test": {"data_file_name": "test.json", "db_dir_name": "test_database"},
        }

        # Imported here: the evaluation script is downloaded into storage/ (see README.md), and the module must
        # stay importable without it for code that only needs the dataset's db paths.
        from evaluated_datasets.spider.storage.evaluation import Evaluator as _SpiderEvaluator

        self._evaluator = _SpiderEvaluator()
        self.data = {}
        for split_name, info in splits.items():
            self.data[split_name] = pd.read_json(Path(__file__).parent / "storage" / info["data_file_name"])
            self.data[split_name]["db_path"] = self.data[split_name]['db_id'].apply(
                lambda db_id: str(Path(__file__).parent / "storage" / info["db_dir_name"] / db_id / f"{db_id}.sqlite")
            )
            self.data[split_name]["hardness"] = self.data[split_name]['sql'].apply(
                self._evaluator.eval_hardness
            )

    def get_data(self, split: Literal["train", "dev", "test"] = None):
        if split:
            return self.data.get(split, None)
        else:
            # return pd.concat([self.data[split] for split in self.data], ignore_index=True)
            return self.data.get("test", None)