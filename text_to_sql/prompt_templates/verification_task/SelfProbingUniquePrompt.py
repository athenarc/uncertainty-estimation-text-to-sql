"""
Self-probing verification prompt: a verifier model is asked how likely the generated SQL query is to be correct, and
states its confidence (0-100); the uncertainty is one minus that confidence.

Self-probing is from:
Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. 2024. Can LLMs Express Their
Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. In International Conference on Learning
Representations (ICLR).
The prompt is the P(True) prompt (see PTruePrompt.py) extended with the request for a confidence score.
"""
import re
from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC

# Source: https://github.com/kckevinchen/RTS-SQL/blob/main/template/candidate_selection.txt
# Copyright: <TODO>
# Licensed under: <TODO> (full text in THIRD_PARTY_LICENSES)
# Modified: <TODO>

class SelfProbingUniquePrompt:
    """Ask the model to self-evaluate how confident it is that a SQL query answers a question.

    The model outputs a numeric confidence score (0–100).  The database schema
    is passed in as a pre-formatted string so the caller controls how it is
    serialised (DDL, compact, M-schema, etc.).
    """

    def __init__(self, dataset_name: str = None):
        """
        Args:
            dataset_name: key from PromptABC's dataset-class map (e.g. "spider"). When
                given, the schema for every db_path in the dataset is pre-computed once
                here instead of being recomputed on every fill_prompt call.
        """
        self._schema_cache = PromptABC._build_db_schema_cache(dataset_name)

    def prompt_template(self) -> str:
        return """Below, you are presented with a database schema, a question, and a SQL query that has been generated. Your task is to output the confidence score that the provided query is correct. To do so, carefully analyze the query against the schema and question, ensuring the following:

The query correctly addresses the question and retrieves the required information.
The query is the only correct interpretation for the given question.
The syntax is valid according to SQL standards and matches the expected format for the database (e.g., SQLite).
The query references the correct tables and columns as per the provided schema.
The query does not include unnecessary columns, tables, or joins, adhering strictly to what is asked in the question.
If the query uses any aggregate functions, ensure they are applied correctly with the appropriate GROUP BY clauses.
If the question specifies sorting or filtering criteria, check that these are properly implemented.
Ensure the query follows the instructions regarding table aliases, logical operations, and ordering.

Database Schema:
{schema}

Question:
{hint}
{input_query}

SQL Query:
{generated_sql}

Output a single integer between 0 and 100, where 0 means you are certain the SQL is wrong and 100 means you are certain the SQL is correct. Output the number only, with no explanation.

Confidence:"""

    def fill_prompt(self, question: str, db_path: str, *args, **kwargs) -> str:
        generated_sql = kwargs.get("generated_sql", "")
        hint = kwargs.get("hint", "")
        if self._schema_cache is not None and db_path in self._schema_cache:
            schema = self._schema_cache[db_path]
        else:
            schema = PromptABC._format_db_schema(db_path)
        return self.prompt_template().format(
            schema=schema,
            input_query=question,
            hint=hint,
            generated_sql=generated_sql
        )

    def chat_message(self, question: str, db_path: str, *args, **kwargs) -> list[dict]:
        return [
            {"role": "user", "content": self.fill_prompt(question, db_path, **kwargs)}
        ]

    def parse_confidence(self, response: str) -> float:
        """Extract the first integer or decimal from the model response.

        Returns a value in [0, 1] (confidence), or 0.5 on parse failure.
        """
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(100.0, score)) / 100.0
            except ValueError:
                pass
        return None
