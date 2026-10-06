"""
Builds the BM25 index of the database contents that OmniSQL uses for value linking, for every dataset of this project.

Run from the repository root: `python -m text_to_sql.utils.build_omnisql_indexes`.
Needs pyserini and Java; if JAVA_HOME is not set it falls back to the Homebrew OpenJDK 21 on macOS.
"""
import os
from pathlib import Path

from evaluated_datasets.ambiqt.ambiqt import AmbiQTDataset
from evaluated_datasets.ambrosia.ambrosia import AmbrosiaDataset
from evaluated_datasets.bird.bird import BirdDataset
from evaluated_datasets.spider.spider import SpiderDataset
from evaluated_datasets.trustsql.trustsql import TrustSQLDataset
from text_to_sql.utils.omnisql_adapter import load_upstream

_DATASETS_ROOT = Path(__file__).parent.parent.parent / "evaluated_datasets"

DATASET_CLASSES = {
    "spider": SpiderDataset,
    "bird": BirdDataset,
    "ambiqt": AmbiQTDataset,
    "ambrosia": AmbrosiaDataset,
    "trustsql": TrustSQLDataset,
}


def main():
    if not os.environ.get("JAVA_HOME"):
        os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"

    upstream = load_upstream("build_contents_index")

    for dataset_name, dataset_cls in DATASET_CLASSES.items():
        print(f"\n=== {dataset_name} ===")
        data = dataset_cls().get_data()

        index_path_prefix = str(_DATASETS_ROOT / dataset_name / "storage" / "db_contents_index")
        upstream.remove_contents_of_a_folder(index_path_prefix)

        for db_file_path in data["db_path"].unique():
            if os.path.isfile(db_file_path):
                db_id = os.path.splitext(os.path.basename(db_file_path))[0]
                print(f"Building index for {db_id}...")
                upstream.build_content_index(db_file_path, os.path.join(index_path_prefix, db_id))
            else:
                print(f"DB file does not exist: {db_file_path}")


if __name__ == "__main__":
    main()
