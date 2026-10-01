from abc import ABC, abstractmethod
from typing import Literal
from loguru import logger
import pandas as pd
from tqdm import tqdm
from text_to_sql.auto_db_schema import obtain_schema_from_db
from text_to_sql.db_schema_sequence import get_db_schema_sequence
from text_to_sql.sqlite_db import DatabaseSqlite

# Maps dataset_name -> dataset class; used by _build_db_schema_cache to discover
# all db_paths for a dataset so their schemas can be pre-computed once.
def _get_dataset_classes() -> dict:
    from evaluated_datasets.ambiqt.ambiqt import AmbiQTDataset
    from evaluated_datasets.ambrosia.ambrosia import AmbrosiaDataset
    from evaluated_datasets.bird.bird import BirdDataset
    from evaluated_datasets.spider.spider import SpiderDataset
    from evaluated_datasets.trustsql.trustsql import TrustSQLDataset

    return {
        "bird": BirdDataset,
        "spider": SpiderDataset,
        "ambrosia": AmbrosiaDataset,
        "ambiqt": AmbiQTDataset,
        "trustsql": TrustSQLDataset,
    }

def get_maleki_db_serialization(schema: list[dict], db_path: str) -> str:
    """Serializes a database schema in Maleki et al. format.
    
    Format:
        Table <table_name>
        columns = [*, col1, col2, col3, ...]
        sample rows:
        row1
        row2
        row3
    
    Args:
        schema (list[dict]): The database schema obtained from obtain_schema_from_db.
        db_path (str): The path to the database.
        
    Returns:
        str: The serialized schema in Maleki et al. format.
    """
    db = DatabaseSqlite(db_path)
    result = []
    
    for table_info in schema:
        table_name = table_info["table_name"]
        columns = table_info["columns"]
        
        # Add table header
        result.append(f"Table {table_name}")
        
        # Add columns line with * and all column names
        column_names = ["*"] + [col["column"] for col in columns]
        result.append(f"columns = {column_names}")
        
        # Add sample rows
        result.append("sample rows:")
        
        # Query sample rows from the table
        try:
            rows_df = db.execute(f"SELECT * FROM '{table_name}' LIMIT 3", limit=-1)
            if isinstance(rows_df, pd.DataFrame) and not rows_df.empty:
                for _, row in rows_df.iterrows():
                    result.append(str(row.to_dict()))
            else:
                logger.warning("No sample rows available")
                result.append("")
        except Exception as e:
            result.append("")
            logger.warning(f"Error retrieving sample rows: {str(e)}")
        
        result.append("")  # Add blank line between tables
    
    return "\n".join(result)


class PromptABC(ABC):

    @abstractmethod
    def prompt_template(self) -> str:
        """The template of the prompt, that contains the description."""
        pass

    @abstractmethod
    def fill_prompt(self, question: str, db_path: str, *args, **kwargs) -> str:
        """Fills the prompt template with the given arguments and returns a prompt.

        Args:
            question (str): The question that will be used to fill the template.
            db_path (str): The path to the database that will be used to fill the template.
            args: Additional arguments passed to the template (e.g., the database schema).
            **kwargs: Additional keyword arguments passed to the template
                (e.g., the database schema).

        Returns:
            str: The prompt template filled with the given arguments.
        """
        pass

    @abstractmethod
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
        pass

    def get_predicted_sql(self, response: str) -> [str, None]:
        """Extracts the SQL query from the model's response.

        Args:
            response (str): The model's response containing the SQL query.

        Returns:
            str: The extracted SQL query.
        """
        # Try to find ```sql first
        sql_start = response.rfind("```sql")
        if sql_start != -1:
            sql = response[sql_start+len("```sql"):].split("```")[0].strip()
            return self._remove_sql_comments(sql)
        
        sql_start = response.rfind("```")
        if sql_start != -1:
            # Look for another ``` before this one
            before_last = response.rfind("```", 0, sql_start)
            if before_last != -1:
                # There's a closing ``` before the last one, extract content between them
                sql = response[before_last+len("```"):sql_start].strip()
            else:
                # No closing ``` before, take everything after the last ```
                sql = response[sql_start+len("```"):].strip()
            return self._remove_sql_comments(sql)

        sql_start = response.rfind("`SELECT")
        if sql_start != -1:
            # Found a final inline SQL snippet starting with backtick
            # Return everything from that backtick until the next backtick (exclusive)
            sql_end = response.find("`", sql_start+1)
            if sql_end != -1:
                sql = response[sql_start+1:sql_end].strip()
            else:
                # No closing backtick, take everything after the last one
                sql = response[sql_start+1:].strip()
            return self._remove_sql_comments(sql)

        return None

    @staticmethod
    def _remove_sql_comments(sql: str) -> str:
        """Remove lines starting with -- (SQL comments) from SQL string.

        Args:
            sql (str): The SQL query string.

        Returns:
            str: The SQL query with comment lines removed.
        """
        lines = sql.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith('--')]
        return '\n'.join(cleaned_lines).strip()

    @staticmethod
    def _build_db_schema_cache(
        dataset_name: str | None,
        db_type: Literal["ddl", "compact", "m-schema", "maleki"] = "ddl",
        example_db_values: int = 3,
    ) -> dict[str, str] | None:
        """Pre-compute the formatted schema for every unique db_path in a dataset.

        Building a schema requires reading the SQLite file (and sampling rows), so doing
        this once per db_path at prompt-object initialization avoids repeating that work
        for every row of the dataset that shares the same database.

        Args:
            dataset_name: key from the dataset-class map (e.g. "spider"). If None, no
                cache is built and callers should fall back to computing schemas on demand.
            db_type: schema serialization format, passed through to _format_db_schema.
            example_db_values: number of sample values per column, passed through too.

        Returns:
            A dict mapping db_path -> formatted schema string, or None if dataset_name
            was not given.
        """
        if not dataset_name:
            return None

        dataset_classes = _get_dataset_classes()
        dataset_cls = dataset_classes.get(dataset_name)
        if dataset_cls is None:
            raise ValueError(
                f"Unknown dataset_name='{dataset_name}'. Supported: {list(dataset_classes.keys())}"
            )

        data: pd.DataFrame = dataset_cls().get_data()
        schema_cache: dict[str, str] = {}
        for db_path in tqdm(
            data["db_path"].unique(),
            desc=f"Pre-loading DB schemas for '{dataset_name}'",
        ):
            schema_cache[db_path] = PromptABC._format_db_schema(
                db_path, db_type=db_type, example_db_values=example_db_values
            )
        return schema_cache

    @staticmethod
    def _format_db_schema(db_path: str, db_type: Literal["ddl", "compact", "m-schema", "maleki"]="ddl", example_db_values: int = 3) -> str:
        # Get the tables and the columns of the database given the db_path
        db = DatabaseSqlite(db_path)
        schema = obtain_schema_from_db(db=db, sample_size=example_db_values)

        if db_type in ["ddl", "compact", "m-schema"]:
            return get_db_schema_sequence(
                schema=schema, type=db_type, include_notes=True, values_num=example_db_values,
                categorical_threshold=20
            )
        elif db_type == "maleki":
            return get_maleki_db_serialization(schema=schema, db_path=db_path)


