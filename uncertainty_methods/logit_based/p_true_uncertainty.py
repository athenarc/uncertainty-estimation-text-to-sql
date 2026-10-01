"""P(True) uncertainty: calculation only.

Given the per-step top-k log-probabilities from a verification model's
answer to "is this SQL correct? (A) yes (B) no", computes an uncertainty
score. Does not run any model inference.

Reference: Kadavath et al. 2022, Section 3.2 / 4.1 - P(True) is the raw
probability the model assigns to the "(A)"/True token, so uncertainty here
is 1 - exp(logprob_a), taken directly from that token's own logprob (not
renormalized against "(B)"'s logprob).
"""
import math
from typing import Optional


def _uncertainty_from_logprob_a(logprob_a: float) -> float:
    """Return uncertainty = 1 - P(True), with P(True) = exp(logprob_a) - the raw
    probability the model puts on the "(A)"/True token (Kadavath et al. 2022)."""
    return 1.0 - math.exp(logprob_a)


_TOKENS_A = {"(A", "(A)", "A", "A)"}
_TOKENS_B = {"(B", "(B)", "B", "B)"}
_TOKENS_OPEN_PAREN = {" (", "( "}


def _scan_step(step_logprobs: dict) -> tuple[Optional[float], Optional[float], bool]:
    """Scan one step's logprob dict for A/B signals.

    Returns:
        (logprob_a, logprob_b, has_open_paren)
    """
    lp_a = lp_b = None
    has_open_paren = False
    for logprob_obj in step_logprobs.values():
        token = logprob_obj.decoded_token
        stripped = token.strip()
        if stripped in _TOKENS_A and lp_a is None:
            lp_a = logprob_obj.logprob
        elif stripped in _TOKENS_B and lp_b is None:
            lp_b = logprob_obj.logprob
        if token in _TOKENS_OPEN_PAREN:
            has_open_paren = True
    return lp_a, lp_b, has_open_paren


def calculate_p_true_uncertainty(token_logprobs: list[dict]) -> Optional[float]:
    """Compute uncertainty as 1 - P(True) from the first up-to-3 generated tokens
    of a verification model's response.

    Because "(A)" is tokenised as ["(A", ")"] and " (A)" as [" (", "A", ")"]
    for Qwen tokenisers, a single token is not always enough to distinguish
    (A) from (B).  This function inspects up to 3 token steps:

    * Token 0 unambiguous  → 1 - exp(logprob_a) at step 0.
    * Token 0 is " ("     → both options share the same first token; look at
                             step 1 where "A" vs "B" are distinct.
    * Fallback             → return None (model output not parseable).

    Reference: Kadavath et al. 2022, Section 3.2 / 4.1.

    Args:
        token_logprobs: list of per-step logprob dicts (token_id → object with
            ``decoded_token``/``logprob`` attributes, e.g. vLLM's ``Logprob``)
            for the first up to 3 generated tokens of the verification
            response.

    Returns:
        Uncertainty in [0, 1], or None if neither (A) nor (B) appeared in top-k.
    """

    lp_a, lp_b, ambiguous = _scan_step(token_logprobs[0])

    # Unambiguous hit at position 0
    if lp_a is not None:
        return _uncertainty_from_logprob_a(lp_a)
    if lp_b is not None and not ambiguous:
        return 1.0  # A absent from top-k, only B found -> treat as certain False

    # First token was " (" — look at position 1 for "A" vs "B"
    if ambiguous and len(token_logprobs) > 1:
        lp_a, lp_b, _ = _scan_step(token_logprobs[1])
        if lp_a is not None:
            return _uncertainty_from_logprob_a(lp_a)
        if lp_b is not None:
            return 1.0

    return None  # neither token found in top-k → uncertainty not computable
