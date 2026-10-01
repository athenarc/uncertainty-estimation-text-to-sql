"""Token-probability ("logit-based") uncertainty.

Given the tokens and per-token log-probabilities produced, this module
scores how confident the model was in its own output.
"""

import logging
import re
from enum import Enum
from typing import NotRequired, Optional, TypedDict

import numpy as np

from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC

logger = logging.getLogger(__name__)


class ConsideredTokens(str, Enum):
    """Which tokens of a generation contribute to the uncertainty score."""

    ALL = "all"
    """Every generated token."""

    SQL_ONLY = "sql_only"
    """Only tokens that fall inside the predicted SQL substring."""

    SCHEMA_LINKED_ONLY = "schema_linked_only"
    """Only tokens of SQL that reference schema/literal content
    (table names, column names, string/numeric literals)"""


class AggregationMethod(str, Enum):
    """How the selected token log-probabilities are combined into one score."""

    MEAN_NLL = "mean_nll"
    """-mean(log_probs) — length-normalised NLL (default)."""

    MAX_NLL = "max_nll"
    """max(-log_probs) — NLL of the single least-probable token."""

    MAX_WORD_NLL = "max_word_nll"
    """Group tokens into words, take the min log-prob per word,
    then average and negate."""


class Generation(TypedDict):
    """Minimal shape of a single model generation required by this module."""

    generated_output: str
    generated_tokens: list[str]
    token_log_probs: list[float]
    question: NotRequired[str]  # only used to identify the row in log messages


# ---------------------------------------------------------------------------
# Word-level tokenisation
# ---------------------------------------------------------------------------
#
# General-purpose word tokeniser: it splits arbitrary text (natural-language
# preamble as well as embedded SQL) into words. It is SQL-aware — quoted
# identifiers, backtick-quoting, and dot-qualified names such as
# `table.column_name` are kept as a single word — but those cases only ever
# trigger where SQL syntax actually appears in the text; plain prose is
# split on whitespace/punctuation like any general-purpose tokeniser.

_WORD_TOKEN_RE = re.compile(
    r"""
    ([^\s,();[\]'"`]+`(?:[^`]|``)*`)  # unquoted prefix immediately followed by backtick-quoted
                                       # e.g.  f.`Charter School (Y/N)`, s.`FRPM Count (K-12)`
    |(`(?:[^`]|``)*`)                  # standalone backtick-quoted identifier
    |("(?:[^"\\]|\\.)*")               # double-quoted identifier
    |('(?:''|[^'])*')                  # single-quoted string  (' escape = '')
    |([,();[\]])                       # structural single-char punctuation
    |([^\s,();[\]'"`]+)                # unquoted: keywords, identifiers, operators, numbers
    """,
    re.VERBOSE | re.DOTALL,
)


def word_spans(text: str) -> list[tuple[str, int, int]]:
    """Return ``[(word_text, start, end), ...]`` for each word in ``text``.

    Whitespace between words is skipped. Structural punctuation characters
    ``( ) , ; [ ]`` each become their own single-character word. Quoted
    identifiers and string literals (as used in SQL) are returned as single
    words regardless of their content — this only matters where such syntax
    appears (i.e. inside embedded SQL); ordinary prose is unaffected.
    """
    return [(m.group(), m.start(), m.end()) for m in _WORD_TOKEN_RE.finditer(text)]


def compute_word_log_probs(
    text: str,
    tokens: list[str],
    token_log_probs: list[float],
    word_aggregation: str = "min",
) -> list[tuple[str, float]]:
    """Return ``[(word, log_prob), ...]`` for each word in ``text``.

    Args:
        text: the sequence to score.
        tokens: BPE token strings such that ``"".join(tokens) == text``.
        token_log_probs: per-token log-probabilities, aligned with ``tokens``.
        word_aggregation: how multiple BPE tokens attributed to the same
            word are combined — ``"sum"`` (joint log-probability of the
            full word), ``"min"`` (log-probability of the single most
            uncertain token in the word), or ``"none"`` (do not aggregate;
            return each word's raw per-token log-probs — used by callers,
            e.g. schema-linked filtering, that need individual token values
            rather than one score per word).
    """
    if word_aggregation not in ("sum", "min", "none"):
        raise ValueError(f"word_aggregation must be 'sum', 'min' or 'none', got {word_aggregation!r}")

    cursor = 0
    token_spans: list[tuple[int, int]] = []
    for tok in tokens:
        token_spans.append((cursor, cursor + len(tok)))
        cursor += len(tok)

    word_entries = word_spans(text)
    if not word_entries:
        return []

    word_lp_lists: list[list[float]] = [[] for _ in word_entries]

    # Attribute each BPE token to one word via its effective start position
    for (tok_start, _), tok_text, lp in zip(token_spans, tokens, token_log_probs):
        stripped = tok_text.lstrip()
        if not stripped:
            continue  # pure-whitespace token — not part of any word
        eff_start = tok_start + (len(tok_text) - len(stripped))

        for idx, (_, w_start, w_end) in enumerate(word_entries):
            if w_start <= eff_start < w_end:
                word_lp_lists[idx].append(lp)
                break

    words = [word for word, _, _ in word_entries]

    if word_aggregation == "none":
        return list(zip(words, word_lp_lists))
    if word_aggregation == "min":
        return [(word, min(lps) if lps else 0.0) for word, lps in zip(words, word_lp_lists)]
    return [(word, sum(lps)) for word, lps in zip(words, word_lp_lists)]


# ---------------------------------------------------------------------------
# Schema-linking classification
# ---------------------------------------------------------------------------
#
# Used by ConsideredTokens.SCHEMA_LINKED_ONLY to isolate the lexemes that
# actually reference the database schema or literal values.

_SQL_KEYWORDS = frozenset({
    "select", "from", "where", "join", "inner", "outer", "left", "right", "full",
    "cross", "on", "as", "with", "distinct", "all", "union", "intersect", "except",
    "group", "by", "order", "having", "limit", "offset",
    "and", "or", "not", "in", "exists", "between", "like", "ilike", "is", "null",
    "true", "false", "case", "when", "then", "else", "end",
    "asc", "desc",
})

# Aggregate functions - not a table/column identifier or literal, so excluded from
# schema-linked lexemes just like _SQL_KEYWORDS, even though they aren't SQL keywords.
_SQL_AGGREGATE_FUNCTIONS = frozenset({
    "count", "sum", "avg", "min", "max", "total", "group_concat",
})

_SQL_OPERATORS_AND_PUNCTUATION = frozenset({
    "(", ")", ",", ";", ".", "*", "!=", "<>", ">=", "<=", "<", ">", "=",
})


def _is_schema_linked_lexeme(token: str) -> bool:
    """Return True if a SQL lexeme is a table/column identifier or a literal value."""
    if not token:
        return False
    if token[0] in ("'", '"', "`", "["):
        return True
    try:
        float(token)
        return True
    except (ValueError, TypeError):
        pass
    if token in _SQL_OPERATORS_AND_PUNCTUATION:
        return False
    lowered = token.lower()
    return lowered not in _SQL_KEYWORDS and lowered not in _SQL_AGGREGATE_FUNCTIONS


def _filter_schema_linked_log_probs(
    text: str,
    tokens: list[str],
    token_log_probs: list[float],
) -> list[float]:
    """Keep only the log-probs of tokens belonging to schema-linked words in
    ``text`` (table names, column names, string/numeric literals), dropping
    keywords, operators, and punctuation.

    Reuses :func:`compute_word_log_probs` (with ``word_aggregation="none"``)
    for the word-attribution step; ``text``/``tokens`` must be the same kind
    of self-contained sequence that function expects.
    """
    grouped = compute_word_log_probs(text, tokens, token_log_probs, word_aggregation="none")
    return [lp for word, lps in grouped if _is_schema_linked_lexeme(word) for lp in lps]


# ---------------------------------------------------------------------------
# Calculate uncertainty
# ---------------------------------------------------------------------------

def _get_predicted_sql(result: Generation, prompt: PromptABC) -> Optional[str]:
    """Extract the predicted SQL substring from a generation, or None on failure."""
    return prompt.get_predicted_sql(result["generated_output"])


def _select_substring_tokens(
    output_text: str,
    tokens: list[str],
    token_log_probs: list[float],
    substring: str,
) -> tuple[str, list[str], list[float]]:
    """Narrow a generation down to just the tokens overlapping ``substring``.

    Returns:
        ``(text, tokens, token_log_probs)`` with ``text == "".join(tokens)``
        — ready to hand to :func:`compute_word_log_probs` or
        :func:`_filter_schema_linked_log_probs`. ``("", [], [])`` if
        ``substring`` is not found in ``output_text``.
    """
    if len(tokens) != len(token_log_probs):
        raise ValueError("tokens and token_log_probs must be aligned (same length)")

    start_char = output_text.find(substring)
    if start_char == -1:
        return "", [], []
    end_char = start_char + len(substring)

    cursor = 0
    selected_tokens: list[str] = []
    selected_log_probs: list[float] = []
    for tok, lp in zip(tokens, token_log_probs):
        tok_start = cursor
        tok_end = cursor + len(tok)
        cursor = tok_end
        if tok_end > start_char and tok_start < end_char:
            selected_tokens.append(tok)
            selected_log_probs.append(lp)

    return "".join(selected_tokens), selected_tokens, selected_log_probs


def calculate_single_output_logit_based_uncertainty(
    result: Generation,
    prompt: PromptABC,
    considered_tokens: ConsideredTokens | str = ConsideredTokens.ALL,
    method: AggregationMethod | str = AggregationMethod.MEAN_NLL,
) -> Optional[float]:
    """Score a single generation's uncertainty from its token log-probabilities.

    Args:
        result: the generation to score. Required keys: ``generated_output``
            (full model output text), ``generated_tokens`` (list of token
            strings) and ``token_log_probs`` (list of per-token
            log-probabilities, aligned with ``generated_tokens``). See
            :class:`Generation`.
        prompt: prompt template used to extract the SQL substring from
            ``generated_output`` (via ``prompt.get_predicted_sql``). Only
            needed when ``considered_tokens`` or ``method`` restricts
            scoring to the SQL substring.
        considered_tokens: which tokens of the generation to score. See
            :class:`ConsideredTokens`.
        method: how to aggregate the selected token log-probabilities. See
            :class:`AggregationMethod`.

    Returns:
        The uncertainty score (higher = more uncertain), or None when it
        cannot be computed (predicted SQL is None, token log-probs are
        missing/empty, or any other error).
    """
    considered = ConsideredTokens(considered_tokens)
    aggregation = AggregationMethod(method)

    try:
        # Step 1: Get the considered tokens from the result output.
        if considered is ConsideredTokens.ALL:
            text = result["generated_output"]
            tokens = result["generated_tokens"]
            token_log_probs = result["token_log_probs"]
        else:
            predicted_sql = _get_predicted_sql(result, prompt)
            if predicted_sql is None:
                return None
            text, tokens, token_log_probs = _select_substring_tokens(
                result["generated_output"],
                result["generated_tokens"],
                result["token_log_probs"],
                predicted_sql,
            )

        # Step 2: Aggregate the probabilities
        if aggregation is AggregationMethod.MAX_WORD_NLL:
            # Calculate the probabilies per word
            word_log_probs = compute_word_log_probs(text, tokens, token_log_probs, word_aggregation="min")
            if considered is ConsideredTokens.SCHEMA_LINKED_ONLY:
                word_log_probs = [
                    (word, lp) for word, lp in word_log_probs if _is_schema_linked_lexeme(word)
                ]
            if not word_log_probs:
                return None
            return float(-np.mean([lp for _, lp in word_log_probs]))

        if considered is ConsideredTokens.SCHEMA_LINKED_ONLY:
            token_log_probs = _filter_schema_linked_log_probs(text, tokens, token_log_probs)

        if not token_log_probs:
            return None

        if aggregation is AggregationMethod.MAX_NLL:
            return float(np.max(-np.array(token_log_probs)))
        return float(-np.mean(token_log_probs))  # MEAN_NLL

    except Exception:
        logger.exception(
            "Failed to compute logit-based uncertainty for question=%r",
            result.get("question", "?"),
        )
        return None


def calculate_multiple_outputs_logit_based_uncertainty(
    generations: list[Generation],
    prompt: PromptABC,
    considered_tokens: ConsideredTokens | str = ConsideredTokens.ALL,
    method: AggregationMethod | str = AggregationMethod.MEAN_NLL,
) -> Optional[float]:
    """Average the logit-based uncertainty over n generations.

    For each generation the uncertainty is computed exactly as in
    :func:`calculate_single_output_logit_based_uncertainty`. The final score
    is the mean across all valid (non-None) generations.

    Args:
        generations: independent samples for the same input (e.g. produced
            by repeated temperature sampling). Each must contain the keys
            required by :func:`calculate_single_output_logit_based_uncertainty`.
        prompt: prompt template used to extract the SQL substring from each
            generation's output.
        considered_tokens: which tokens of each generation to score. See
            :class:`ConsideredTokens`.
        method: how to aggregate each generation's token log-probabilities.
            See :class:`AggregationMethod`.

    Returns:
        The mean uncertainty across generations, or None if every generation
        produced None (i.e. no valid uncertainty could be computed).
    """
    per_generation = [
        calculate_single_output_logit_based_uncertainty(r, prompt, considered_tokens, method)
        for r in generations
    ]
    valid = [v for v in per_generation if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))