"""
Vanilla verbalized confidence prompt: the model generates the SQL query and then states its confidence (0-100).

The prompt is taken from:
Sepideh Entezari Maleki, Mohammadreza Pourreza, and Davood Rafiei. 2026. Confidence Estimation for Text-to-SQL in
Large Language Models. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI'26).
https://arxiv.org/pdf/2508.14056
"""
import json
import re
from pathlib import Path

from evaluated_datasets.bird.bird import BirdDataset
from loguru import logger
from text_to_sql.prompt_templates.text_to_sql_task.VerbalizedConfidencePromptABC import VerbalizedConfidencePromptABC
from tqdm import tqdm

_DATASETS_ROOT = Path(__file__).parent.parent.parent.parent / "evaluated_datasets"

# Default locations of the optional precomputed schema caches, one per dataset ({db_path: serialized schema}). When a
# dataset has no cache file, the schemas are serialized on the fly.
_DEFAULT_SCHEMA_CACHE_PATHS: dict[str, Path] = {
    "bird": _DATASETS_ROOT / "bird" / "storage" / "maleki_db_schema_cache.json",
    "spider": _DATASETS_ROOT / "spider" / "storage" / "maleki_db_schema_cache.json",
    "ambrosia": _DATASETS_ROOT / "ambrosia" / "storage" / "maleki_db_schema_cache.json",
    "ambiqt": _DATASETS_ROOT / "ambiqt" / "storage" / "maleki_db_schema_cache.json",
    "trustsql": _DATASETS_ROOT / "trustsql" / "storage" / "maleki_db_schema_cache.json",
}

# Matches an inline cue immediately preceding a query the model states mid-sentence/mid-bullet
# instead of on its own line (e.g. "6. The query should be: SELECT Manager, Captain FROM club"),
# used only as a last resort once the label/line-anchor extraction below has failed outright.
_INLINE_SQL_CUE = re.compile(
    r"(?:should be|correct query(?: should be| is)|query (?:should be|is))\s*:?\s*[\"'`]?\s*(?=SELECT\b|WITH\b)",
    re.IGNORECASE,
)


def _extract_labeled_or_anchored_sql(region: str) -> str | None:
    """Extracts a query from region via the "SQL Query:" label (last occurrence, since a
    self-correction repeats it) or, failing that, the last standalone SELECT/WITH statement
    anchored to the start of a line (an indented SELECT is usually a nested subquery fragment,
    e.g. inside a WHERE ... IN (...), not the outer statement - preferred over an unanchored
    match for that reason). Returns None if neither is found, or if what follows the label isn't
    actually a query (the model sometimes echoes the label and then writes a meta-comment instead
    of continuing with SQL).
    """
    marker_matches = list(re.finditer(r"(?m)^[ \t]*SQL\s*Query\s*:\s*", region, flags=re.IGNORECASE))
    if marker_matches:
        start = marker_matches[-1].end()
        tail = region[start:]
        end_match = re.search(r"\n[ \t]*\n|Scratchpad:|Confidence:", tail, flags=re.IGNORECASE)
        end = start + end_match.start() if end_match else len(region)
        candidate = region[start:end].strip()
        candidate = re.sub(r"^```(?:sql)?\s*", "", candidate)
        candidate = re.sub(r"\s*```\s*$", "", candidate).strip()
        if candidate.upper().startswith(("SELECT", "WITH")):
            return candidate

    stmt_starts = list(re.finditer(r"(?m)^(?:SELECT|WITH)\b", region))
    if not stmt_starts:
        stmt_starts = list(re.finditer(r"(?im)^[ \t]*(?:SELECT|WITH)\b", region))
    if stmt_starts:
        start = stmt_starts[-1].start()
        tail = region[start:]
        end_match = re.search(r"\n[ \t]*\n|Scratchpad:|Confidence:", tail, flags=re.IGNORECASE)
        end = start + end_match.start() if end_match else len(region)
        return region[start:end].strip()

    return None


class VanillaVerbalizedMalekiPrompt(VerbalizedConfidencePromptABC):

    def __init__(self, dataset_name: str = None):
        self._schema_cache: dict[str, str] | None = None
        if dataset_name:
            path = _DEFAULT_SCHEMA_CACHE_PATHS.get(dataset_name)
            if path and path.exists():
                with open(path, encoding="utf-8") as f:
                    self._schema_cache = json.load(f)
                logger.info(f"Loaded Maleki schema cache for '{dataset_name}' ({len(self._schema_cache)} DBs).")

    def prompt_template(self) -> str:
        return """You are an agent designed to answer the given question by generating a SQL Query and provide your confidence level.
Note that the confidence level indicates the degree of certainty you have about the SQL Query (between 0 and 100).
The schema of the tables are provided with three samples rows from each table. Use this information for generating the SQL Query.

follow the below format:
Question: user's question
SQL Query: SQL query that can answer the question
Confidence: A score in range (0-100) which shows how confident your are about the answer

{db_schema}

Question: {hint}{question}
SQL Query:
"""

    def fill_prompt(self, question: str, db_path: str, *args, **kwargs) -> str:
        hint = str(kwargs.get("hint", "") or "")
        if self._schema_cache is not None and db_path in self._schema_cache:
            db_schema = self._schema_cache[db_path]
        else:
            db_schema = self._format_db_schema(db_path, example_db_values=0, db_type="maleki")
        return self.prompt_template().format(
            question=question,
            db_schema=db_schema,
            hint=f"{hint}\n" if hint else ""
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
        """Extracts the SQL query from the model's response.

        Args:
            response (str): The model's response containing the SQL query.

        Returns:
            str: The extracted SQL query, or None if extraction fails.
        """
        try:
            # The model sometimes prepends free-form reasoning before presenting the query in a
            # fenced ```sql ... ``` block, and/or revises the query across multiple fenced blocks
            # in the same response. Prefer the LAST fenced block anywhere in the response - it is
            # consistently the final, refined query.
            fence_matches = re.findall(r"```(?:sql)?\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
            if fence_matches:
                return fence_matches[-1].strip()

            # Restrict to the region before the LAST "Confidence:" - text after that is normally
            # the verbalized confidence explanation, not part of the query. Use the last
            # occurrence, not the first: when the model self-corrects ("SQL Query: <draft>\n
            # Confidence: 85\n...revised...\nSQL Query: <revised>\nConfidence: 90"), cutting at the
            # first "Confidence:" would hide the revised query entirely and resurrect the disavowed
            # draft instead.
            conf_matches = list(re.finditer(r"Confidence:", response))
            search_region = response if not conf_matches else response[:conf_matches[-1].start()]

            sql_text = _extract_labeled_or_anchored_sql(search_region)

            if sql_text is None and search_region is not response:
                # The model sometimes second-guesses a score it already gave and keeps writing,
                # producing the real, final query AFTER its last "Confidence:" mention (often
                # without a fresh score of its own, because generation got cut off by max_tokens
                # first). That query is invisible to the region above, which clipped everything
                # after the last "Confidence:" - retry unclipped, against the whole response.
                sql_text = _extract_labeled_or_anchored_sql(response)

            if sql_text is None:
                # Last resort: the model sometimes never gives the query its own line at all, and
                # instead states it inline mid-sentence/mid-bullet while critiquing a (nonexistent)
                # provided query, e.g. "6. The query should be: SELECT Manager, Captain FROM club".
                # Take the LAST such cue anywhere in the response - the final, corrected version.
                cue_matches = list(_INLINE_SQL_CUE.finditer(response))
                if cue_matches:
                    start = cue_matches[-1].end()
                    tail = response[start:]
                    end_match = re.search(r"\n|\. [A-Z]", tail)
                    end = start + end_match.start() if end_match else len(response)
                    candidate = response[start:end].strip()
                    if candidate.upper().startswith(("SELECT", "WITH")):
                        sql_text = candidate

            # Nothing recoverable - report "no prediction" rather than falling back to dumping raw
            # reasoning prose as if it were SQL, which just hands the execution engine a syntax
            # error instead of an honest missing-prediction signal.
            return sql_text
        except Exception:
            return None

    def get_predicted_uncertainty(self, response: str) -> float:
        """Extracts the confidence score from the model's response.

        Args:
            response (str): The model's response containing the confidence score.

        Returns:
            int: The extracted uncertainty score, or None if extraction fails.
        """
        # Find last occurrence of 'Confidence:' (case-sensitive) and extract integer following it
        matches = list(re.finditer(r"Confidence:\s*([0-9]+)", response))
        if not matches:
            return None
        last = matches[-1]
        try:
            return (100 - float(last.group(1))) / 100
        except ValueError:
            print("Error parsing confidence score.")
        return None


if __name__ == "__main__":
    prompt = VanillaVerbalizedMalekiPrompt()

    dataset = BirdDataset().get_data()

    prompts = []
    for _, row in tqdm(dataset.iterrows()):
        question = row["question"]
        db_path = row["db_path"]

        prompts.append(prompt.fill_prompt(question, db_path, **{"hint": row["hint"]}))

    print(prompts)
