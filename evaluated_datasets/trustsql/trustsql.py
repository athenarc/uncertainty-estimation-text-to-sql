from pathlib import Path

import pandas as pd

from evaluated_datasets.DatasetABC import DatasetABC

_SINGLE_DB_FILE = {
    "atis": "atis.sqlite",
    "advising": "advising.sqlite",
    "ehrsql": "mimic_iv.sqlite",
}

_ORIGIN_DATASETS = ["spider", "atis", "advising", "ehrsql"]


def _spider_db_path(ds_dir: Path, db_id: str) -> str:
    return str(ds_dir / "database" / db_id / f"{db_id}.sqlite")


class TrustSQLDataset(DatasetABC):
    def __init__(self):
        storage = Path(__file__).parent / "storage" / "dataset"

        frames = []
        for origin in _ORIGIN_DATASETS:
            ds_dir = storage / origin

            feasible = pd.read_json(ds_dir / f"{origin}_test_feasible.json")
            feasible["unanswerable"] = False

            infeasible = pd.read_json(ds_dir / f"{origin}_test_infeasible.json")
            infeasible["unanswerable"] = True

            combined = pd.concat([feasible, infeasible], ignore_index=True)
            combined["origin_dataset"] = origin

            if origin == "spider":
                combined["db_path"] = combined["db_id"].apply(
                    lambda db_id: _spider_db_path(ds_dir, db_id)
                )
            else:
                combined["db_path"] = str(ds_dir / _SINGLE_DB_FILE[origin])

            frames.append(combined)

        self.data = pd.concat(frames, ignore_index=True)

    def get_data(self) -> pd.DataFrame:
        return self.data