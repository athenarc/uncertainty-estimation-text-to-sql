"""
Prompt of the XiYanSQL model (XiYanSQL-QwenCoder), with the M-Schema serialization of the database.

The prompt format is the one provided with the model, from:
Yifu Liu, Yin Zhu, Yingqi Gao, Zhiling Luo, Xiaoxia Li, Xiaorong Shi, Yuntao Hong, Jinyang Gao, Yu Li, Bolin Ding, and
Jingren Zhou. 2026. XiYan-SQL: A Novel Multi-Generator Framework for Text-to-SQL. IEEE Transactions on Knowledge and
Data Engineering 38, 4 (2026), 2474-2487. https://doi.org/10.1109/TKDE.2026.3657851
"""
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.exc import NoSuchTableError
from tqdm import tqdm

from text_to_sql.utils.xiyan_adapter import SchemaEngine
from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC
from evaluated_datasets.spider.spider import SpiderDataset


class XiYanSQLPrompt(PromptABC):

    def prompt_template(self) -> str:
        return """You are an expert in SQLite. Your task is to read and understand the following **database schema** description and any provided **reference information**, then use your knowledge of SQLite to generate an SQL query that answers the **user's question**.

**User Question**
{question}

**Database Schema**
{db_schema}

**Reference Information**
{evidence}

**User Question**
{question}

```sql"""

    def fill_prompt(self, question: str, db_path: str, *args, **kwargs) -> str:
        hint = str(kwargs.get("hint", "") or "")
        db_schema = self._format_db_schema(db_path)
        return self.prompt_template().format(
            question=question,
            db_schema=db_schema,
            evidence=f"{hint}\n" if hint else "",
        )

    def chat_message(self, question: str, db_path: str, *args, **kwargs) -> list[dict]:
        """Generates a chat message for the given question and database path.

        Args:
            question (str): The question to be answered.
            db_path (str): The path to the database.
            args: Additional arguments passed to the template.
            **kwargs: Additional keyword arguments passed to the template
                (e.g., hint).

        Returns:
            list[dict]: A list of dictionaries representing the chat message.
        """
        return [
            {"role": "user", "content": self.fill_prompt(question, db_path, **kwargs)}
        ]
    
    def get_predicted_sql(self, response: str) -> [str, None]:
        # The prompt template already ends with an open "```sql" fence, so the model's
        # response is just the SQL body (no fences at all in practice), optionally
        # followed by a closing "```" if the model happens to add one.
        if not response or not response.strip():
            return None
        sql = response.split("```")[0].strip()
        if not sql:
            return None
        return self._remove_sql_comments(sql)

    @staticmethod
    def _format_db_schema(db_path: str) -> str:
        db_id = os.path.splitext(os.path.basename(db_path))[0]
        abs_path = os.path.abspath(db_path)
        db_engine = create_engine(f'sqlite:///{abs_path}')
        try:
            schema_engine = SchemaEngine(engine=db_engine, db_name=db_id)
            return schema_engine.mschema.to_mschema()
        except NoSuchTableError:
            return PromptABC._format_db_schema(db_path, db_type="m-schema")


if __name__ == "__main__":

    prompt = XiYanSQLPrompt()

    dataset = SpiderDataset().get_data()

    prompts = []
    for _, row in tqdm(dataset.iterrows()):
        question = row["question"]
        db_path = row["db_path"]

        prompts.append(prompt.fill_prompt(question, db_path,))

    print(prompts)