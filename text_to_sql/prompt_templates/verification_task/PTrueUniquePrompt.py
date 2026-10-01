"""
P(True) verification prompt: a verifier model is asked whether the generated SQL query correctly answers the question,
and answers with option (A) True or (B) False; the uncertainty is read off the probability of the answer.

P(True) is from:
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac
Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. 2022. Language Models (Mostly) Know What They Know.
https://arxiv.org/abs/2207.05221
The prompt adapts it to Text-to-SQL (the question, the SQL query and the database schema), following the template
referenced in the comment below.
"""
from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC


# prompt similar to https://github.com/kckevinchen/RTS-SQL/blob/main/template/candidate_selection.txt

class PTrueUniquePrompt:

    def __init__(self, dataset_name: str = None):
        """
        Args:
            dataset_name: key from PromptABC's dataset-class map (e.g. "spider"). When
                given, the schema for every db_path in the dataset is pre-computed once
                here instead of being recomputed on every fill_prompt call.
        """
        self._schema_cache = PromptABC._build_db_schema_cache(dataset_name)

    def prompt_template(self) -> str:
        return """Below, you are presented with a database schema, a question, and a SQL query that has been generated. Your task is to verify whether the generated SQL query is correct. To do so, carefully analyze the query against the schema and question, ensuring the following:

The query correctly addresses the question and retrieves the required information.
The query is the only correct interpretation for the given question.
The syntax is valid according to SQL standards and matches the expected format for the database (e.g., SQLite).
The query references the correct tables and columns as per the provided schema.
The query does not include unnecessary columns, tables, or joins, adhering strictly to what is asked in the question.
If the query uses any aggregate functions, ensure they are applied correctly with the appropriate GROUP BY clauses.
If the question specifies sorting or filtering criteria, check that these are properly implemented.
Ensure the query follows the instructions regarding table aliases, logical operations, and ordering.

Database Schema

{schema}

This schema provides an in-depth description of the database's structure, detailing tables, columns, primary keys, foreign keys, and any relevant constraints. Pay close attention to the examples listed beside each column, as they provide important hints about which columns are relevant to the question.

Question:
{hint}
{input_query}

Generated SQL Query:
{generated_sql}

Is the generated SQL query:
(A) True
(B) False

The generated SQL query is:
"""

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
            generated_sql=generated_sql,
        )

    def chat_message(self, question: str, db_path: str, *args, **kwargs) -> list[dict]:
        return [
            {"role": "user", "content": self.fill_prompt(question, db_path, **kwargs)}
        ]
