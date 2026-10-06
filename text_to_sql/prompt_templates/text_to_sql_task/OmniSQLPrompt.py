"""
Prompt of the OmniSQL model, used with its schema/value-linking component (see text_to_sql/utils/omnisql_adapter.py).

The prompt format is the one provided in the OmniSQL repository, from:
Haoyang Li, Shang Wu, Xiaokang Zhang, Xinmei Huang, Jing Zhang, Fuxin Jiang, Shuai Wang, Tieying Zhang, Jianjun Chen,
Rui Shi, Hong Chen, and Cuiping Li. 2025. OmniSQL: Synthesizing High-Quality Text-to-SQL Data at Scale. Proceedings of
the VLDB Endowment 18, 11 (2025), 4695-4709. https://doi.org/10.14778/3749646.3749723
"""
import inspect
import os
import re
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from evaluated_datasets.ambrosia.ambrosia import AmbrosiaDataset
from evaluated_datasets.ambiqt.ambiqt import AmbiQTDataset
from evaluated_datasets.bird.bird import BirdDataset
from evaluated_datasets.spider.spider import SpiderDataset
from evaluated_datasets.trustsql.trustsql import TrustSQLDataset
from text_to_sql.utils.omnisql_adapter import (
    deduplicate_dicts,
    obtain_db_details,
    obtain_db_info_from_sqlite,
    obtain_n_grams,
    retrieve_question_related_db_values,
    retrieve_relevant_hits,
    sample_table_values,
)
from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC

# try:
#     from pyserini.search.lucene import LuceneSearcher
#     _LUCENE_AVAILABLE = True
# except ImportError:

_LUCENE_AVAILABLE = False


_DATASETS_ROOT = Path(__file__).parent.parent.parent.parent / "evaluated_datasets"

# Default locations of the optional precomputed DB schema caches (one JSON per dataset); a dataset without a cache
# file gets its schemas built when the prompt is created.
# The cache stores the {db_schema} fragment (value-linked DDL), keyed by "<question>|||<db_id>",
# so it is shared by OmniSQLPrompt and any prompt built on top of it (e.g. the verbalized-
# confidence variants), which only differ in the wrapping prompt_template().
_DEFAULT_OMNISQL_SCHEMA_CACHE_PATHS: dict[str, Path] = {
    "bird":     _DATASETS_ROOT / "bird"     / "storage" / "omnisql_schema_cache.json",
    "spider":   _DATASETS_ROOT / "spider"   / "storage" / "omnisql_schema_cache.json",
    "ambrosia": _DATASETS_ROOT / "ambrosia" / "storage" / "omnisql_schema_cache.json",
    "ambiqt":   _DATASETS_ROOT / "ambiqt"   / "storage" / "omnisql_schema_cache.json",
    "trustsql": _DATASETS_ROOT / "trustsql" / "storage" / "omnisql_schema_cache.json",
}

# Maps dataset_name -> dataset class; used in __init__ to obtain all db_paths and questions.
_DATASET_CLASSES = {
    "bird":      BirdDataset,
    "spider":    SpiderDataset,
    "ambrosia":  AmbrosiaDataset,
    "ambiqt":    AmbiQTDataset,
    "trustsql":  TrustSQLDataset,
}


class OmniSQLPrompt(PromptABC):

    def __init__(
        self,
        dataset_name: str = None,
        value_limit_num: int = 2,
    ):
        """
        Args:
            dataset_name: key from _DATASET_CLASSES (e.g. "spider"). The matching
                dataset class is instantiated to discover all db_paths and questions,
                which are pre-loaded into the schema/value cache and Lucene hit cache.
                If a precomputed prompt cache exists for this dataset, it is loaded
                instead and no Lucene index is required.
            value_limit_num: number of distinct cell values sampled per column.
        """
        self._value_limit_num = value_limit_num
        self._schema_cache: dict[str, str] | None = None
        self._db_content_index_path: str | None = None

        self._db_id2db_info: dict = {}         # db_id -> Spider-format schema dict
        self._db_id2sampled_values: dict = {}  # db_id -> {table.column: [val, ...]}
        self._db_id2searcher: dict = {}        # db_id -> LuceneSearcher
        self._db_id2query2hits: dict = {}      # db_id -> {query_string -> [hit dicts]}

        if dataset_name:
            # If a precomputed schema cache exists, load it and skip all Lucene/schema work.
            cache_path = _DEFAULT_OMNISQL_SCHEMA_CACHE_PATHS.get(dataset_name)
            if cache_path and cache_path.exists():
                with open(cache_path, encoding="utf-8") as f:
                    self._schema_cache = json.load(f)
                logger.info(f"OmniSQLPrompt: loaded precomputed schema cache for '{dataset_name}' ({len(self._schema_cache)} entries).")
                return

            dataset_cls = _DATASET_CLASSES.get(dataset_name)
            if dataset_cls is None:
                raise ValueError(
                    f"Unknown dataset_name='{dataset_name}'. "
                    f"Supported: {list(_DATASET_CLASSES.keys())}"
                )

            candidate = Path(inspect.getfile(dataset_cls)).parent / "storage" / "db_contents_index"
            if candidate.exists():
                self._db_content_index_path = str(candidate)
                logger.info(f"OmniSQLPrompt: auto-detected db_content_index_path={self._db_content_index_path}")

            dataset = dataset_cls()
            data: pd.DataFrame = dataset.get_data()

            # --- Pre-process step: pre-load schema and sample values for the databases used ---
            # Iterates over all unique db_paths in the dataset and reads each SQLite file
            # directly via PRAGMA — no tables.json required.
            for db_path in tqdm(
                data["db_path"].unique(),
                desc=f"OmniSQLPrompt [{dataset_name}]: loading schema/values",
            ):
                db_id = os.path.splitext(os.path.basename(db_path))[0]
                self._db_id2db_info[db_id] = obtain_db_info_from_sqlite(db_path)
                self._db_id2sampled_values[db_id] = sample_table_values(
                    db_path,
                    self._db_id2db_info[db_id]["table_names_original"],
                    value_limit_num,
                )

            logger.info(
                f"OmniSQLPrompt: pre-loaded schema/values for {len(self._db_id2db_info)} databases."
            )

            # The remaining three steps require a Lucene index — skip if not provided.
            if self._db_content_index_path and _LUCENE_AVAILABLE:
                index_root = Path(self._db_content_index_path)

                # --- Retrieve question-relevant cell values from a Lucene index ---
                # Opens one LuceneSearcher per database that has a built index directory.
                for db_id in self._db_id2db_info:
                    index_dir = index_root / db_id
                    if index_dir.exists():
                        self._db_id2searcher[db_id] = LuceneSearcher(str(index_dir))
                logger.info(
                    f"OmniSQLPrompt: Lucene searchers loaded for {len(self._db_id2searcher)} databases."
                )

                # --- Collect all n-gram queries (up to 8-grams) for all the questions ---
                # grouped by db_id so we can batch-search each DB's index in one call.
                # Using n-grams (not just the full question) ensures short cell values
                # that appear as sub-phrases are also retrieved.
                hint_col = "hint" if "hint" in data.columns else None
                db_id2queries: dict[str, list[str]] = {}

                for _, row in data.iterrows():
                    db_id = os.path.splitext(os.path.basename(row["db_path"]))[0]
                    if db_id not in self._db_id2searcher:
                        continue  # no index for this DB — skip
                    hint = str(row[hint_col]) if hint_col and pd.notna(row[hint_col]) else ""
                    full_question = f"{hint}\n{row['question']}" if hint else str(row["question"])
                    queries = obtain_n_grams(full_question, 8) + [full_question]
                    db_id2queries.setdefault(db_id, []).extend(queries)

                # --- Run all queries against each DB's Lucene index in a single batch call ---
                # retrieve_relevant_hits deduplicates queries internally and issues one
                # batch_search per DB, returning {query_string -> [hit dicts]}.
                for db_id, queries in tqdm(
                    db_id2queries.items(),
                    desc=f"OmniSQLPrompt [{dataset_name}]: Lucene batch retrieval",
                ):
                    queries = list(dict.fromkeys(queries))  # deduplicate while preserving order
                    self._db_id2query2hits[db_id] = retrieve_relevant_hits(
                        self._db_id2searcher[db_id], queries
                    )

    def _load_db(self, db_path: str) -> None:
        """Lazy-load a database not seen during __init__ (e.g. when dataset_name=None)."""
        db_id = os.path.splitext(os.path.basename(db_path))[0]
        self._db_id2db_info[db_id] = obtain_db_info_from_sqlite(db_path)
        self._db_id2sampled_values[db_id] = sample_table_values(
            db_path, self._db_id2db_info[db_id]["table_names_original"], self._value_limit_num
        )
        if self._db_content_index_path and _LUCENE_AVAILABLE:
            index_dir = Path(self._db_content_index_path) / db_id
            if index_dir.exists():
                self._db_id2searcher[db_id] = LuceneSearcher(str(index_dir))

    def prompt_template(self) -> str:
        """
        Replace the placeholders for "db_schema" and "question" to get started. Note that "db_details" is formatted as
        CREATE TABLE statements (i.e., DDL) of tables in the database. You can add database values and column
        descriptions in DDLs with SQL comments. External knowledge can be concatenated with the natural language
        question and placed in the "question" placeholder. OmniSQL currently supports only SQLite, as the SQL queries in
        SynSQL-2.5M are synthesized using the SQLite dialect.
        """
        return """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{hint}{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
"""

    def get_omnisql_db_schema(self, question: str, db_path: str, hint: str = "") -> str:
        """Build the OmniSQL DDL schema string for a given question and database.

        Mirrors the logic from prepare_input_output_pairs in OmniSQL's process_dataset.py,
        adapted to use the caches pre-loaded in __init__ instead of the batch structures
        from the original __main__ block.

        Args:
            question: the natural language question.
            db_path: path to the SQLite file.
            hint: optional external knowledge / evidence to prepend to the question
                  when retrieving relevant cell values (same role as ek_key in __main__).

        Returns:
            A DDL schema string (CREATE TABLE statements with inline example values)
            ready to drop into the {db_schema} placeholder of the prompt template.
        """
        db_id = os.path.splitext(os.path.basename(db_path))[0]

        if self._schema_cache is not None:
            key = f"{question}|||{db_id}"
            if key in self._schema_cache:
                return self._schema_cache[key]

        # Lazy-load if this db_path was not covered by the dataset used in __init__
        if db_id not in self._db_id2db_info:
            self._load_db(db_path)

        db_info = self._db_id2db_info[db_id]
        sampled_values = self._db_id2sampled_values[db_id]

        # Combine hint + question the same way prepare_input_output_pairs does with ek_key
        full_question = f"{hint}\n{question}" if hint else question

        relavant_db_values_dict = {}
        if db_id in self._db_id2query2hits:
            # Use hits pre-computed at init time: look up each of this question's
            # n-gram queries in the cached {query_string -> [hits]} dict.
            queries = obtain_n_grams(full_question, 8) + [full_question]
            queries = list(dict.fromkeys(queries))
            query2hits = self._db_id2query2hits[db_id]
            hits = []
            for query in queries:
                if query in query2hits:
                    hits.extend(query2hits[query])
            hits = deduplicate_dicts(hits)
            relavant_db_values_dict = retrieve_question_related_db_values(hits, full_question)

        elif db_id in self._db_id2searcher:
            # Fallback: run Lucene search on the fly (e.g. for lazy-loaded databases)
            queries = obtain_n_grams(full_question, 8) + [full_question]
            queries = list(dict.fromkeys(queries))
            query2hits = retrieve_relevant_hits(self._db_id2searcher[db_id], queries)
            hits = []
            for h in query2hits.values():
                hits.extend(h)
            hits = deduplicate_dicts(hits)
            relavant_db_values_dict = retrieve_question_related_db_values(hits, full_question)

        # Build the DDL schema string with sample values embedded as SQL comments.
        # mode="test" includes all columns (training mode randomly drops some for diversity).
        # data_source="spider" uses standard non-synthetic comment formatting.
        return obtain_db_details(
            db_info=db_info,
            data_source="spider",
            sampled_db_values_dict=sampled_values,
            relavant_db_values_dict=relavant_db_values_dict,
            output_seq="",
            mode="test",
            question=question,
        )

    def fill_prompt(self, question: str, db_path: str, *args, **kwargs) -> str:
        hint = str(kwargs.get("hint", "") or "")
        return self.prompt_template().format(
            question=question,
            db_schema=self.get_omnisql_db_schema(question, db_path, hint=hint),
            hint=f"{hint}\n" if hint else "",
        )

    def chat_message(self, question: str, db_path: str, *args, **kwargs) -> list[dict]:
        """Generates a chat message for the given question and database path.

        Args:
            question (str): The question to be answered.
            db_path (str): The path to the database.
            args: Additional arguments passed to the template (e.g., the database schema).
            **kwargs: Additional keyword arguments passed to the template
                (e.g., the database schema).

        Returns:
            list[dict]: A list of dictionaries representing the chat message.
        """
        return [
            {"role": "user", "content": self.fill_prompt(question, db_path, **kwargs)}
        ]

    def get_predicted_sql(self, response: str) -> [str, None]:
        # Source: https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/infer.py
        # Copyright: <TODO>
        # Licensed under: <TODO> (full text in THIRD_PARTY_LICENSES)
        # Modified: <TODO>
        pattern = r"```sql\s*(.*?)\s*```"
    
        sql_blocks = re.findall(pattern, response, re.DOTALL)

        if sql_blocks:
            # Extract the last SQL query in the response text and remove extra whitespace characters
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            # print("No SQL blocks found.")
            return ""


if __name__ == "__main__":
    import json
    import time

    DATASET_NAME = "bird"
    prompt = OmniSQLPrompt(dataset_name=DATASET_NAME)

    dataset = BirdDataset().get_data()

    prompts = []
    for _, row in tqdm(dataset.iterrows()):
        question = row["question"]
        db_path = row["db_path"]

        prompts.append({
            "question": question,
            "db_path": db_path,
            "prompt": prompt.fill_prompt(question, db_path),
        })

    output_path = f"prompts_{DATASET_NAME}_{int(time.time())}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(prompts)} prompts to {output_path}")
