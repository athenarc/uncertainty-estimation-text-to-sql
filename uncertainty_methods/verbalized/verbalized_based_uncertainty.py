"""Verbalized-confidence uncertainty: calculation only.

Given a generation in which the model was prompted to state its own
confidence in natural language/numeric form, extracts an uncertainty score.
Does not run any model inference.
"""
import math
from typing import Optional

from text_to_sql.prompt_templates.text_to_sql_task.VerbalizedConfidencePromptABC import VerbalizedConfidencePromptABC


def calculate_verbalized_uncertainty(
    result: dict,
    prompt: VerbalizedConfidencePromptABC,
    predicted_sql: str,
) -> Optional[float]:
    """Extract the verbalized uncertainty from a generation's output text.

    Args:
        result: generation dict. Required key: ``generated_output`` (the
            full model output text, expected to contain a verbalized
            confidence statement).
        prompt: prompt template used to parse the confidence/uncertainty out
            of ``result['generated_output']``.
        predicted_sql: the SQL query extracted from the same generation.
            When ``None`` or NaN (i.e. no SQL could be generated), the
            uncertainty is not computable and ``None`` is returned.

    Returns:
        Uncertainty score parsed from the output, or None if no SQL was
        predicted.
    """
    if predicted_sql is None or (type(predicted_sql) == float and math.isnan(predicted_sql)):
        return None
    return prompt.get_predicted_uncertainty(result['generated_output'])
