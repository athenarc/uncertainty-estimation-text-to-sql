import pandas as pd

from text_to_sql.inference.model import Model
from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC


def text_to_sql_inference(
    model_name: str,
    model_parameters: dict,
    prompt_template: PromptABC,
    dataset: pd.DataFrame,
    dataset_name: str,
    n: int = 5,
    logprobs_num: int = 1,
    temperature: float = 1.0,
    model_args: dict | None = None,
    model_parallelism: bool = True,
) -> list[list[dict]]:
    """Run vLLM inference and generate n SQL completions per question.

    Initializes a vLLM-backed model, builds prompts from the dataset using the
    provided prompt template, and generates n independent samples per question with
    per-token log probabilities.

    Args:
        model_name: HuggingFace model identifier or local path passed to vLLM.
        model_parameters: Model configuration dict; recognised keys include
            ``max_tokens`` (int, default 1024) and ``temperature`` (float, default 0.0).
        prompt_template: Prompt strategy that converts a dataset row into a
            chat-message list.
        dataset: DataFrame where each row is one NL-to-SQL example. Required columns:
            ``question``, ``query`` (ground-truth SQL), ``db_path``
        dataset_name: Dataset identifier forwarded to the prompt template (e.g.
            ``"spider"``, ``"bird"``).
        n: Number of independent completions to generate per question.
        logprobs_num: Number of top-token log-probability entries to return at each
            decoding step (passed as ``logprobs`` to vLLM ``SamplingParams``).
        temperature: Sampling temperature.
        model_args: Extra keyword arguments forwarded to vLLM ``SamplingParams``.
        model_parallelism: If ``True``, uses tensor parallelism across 2 GPUs
            (``tensor_parallel_size=2``); if ``False``, runs on a single GPU.

    Returns:
        A nested list ``all_results`` of shape ``[num_questions][n]``.
        Each element is a dict with the following keys:

        - ``question`` (str): Natural-language question.
        - ``sql_query`` (str): Ground-truth SQL from the dataset.
        - ``db_path`` (str): Path to the SQLite database file.
        - ``model_input`` (list[dict]): Chat-message list sent to the model.
        - ``generated_output`` (str): Raw text produced by the model.
        - ``generated_tokens`` (list[str]): Sequence of decoded tokens.
        - ``token_log_probs`` (list[float]): Per-token log probability of the
          chosen token at each decoding step.
        - ``top_tokens_probs`` (list[dict]): Per-step top-k alternatives, each
          dict containing ``"top_tokens"`` (list[str]) and
          ``"top_log_probs"`` (list[float]), sorted descending by log prob.
        - ``is_ambiguous``, ``ambig_type``, ``unanswerable``, ``origin_dataset``
          (optional): Copied from the dataset row when present.
    """
    model = Model(
        model_name=model_name,
        model_parameters=model_parameters,
        model_parallelism=model_parallelism,
    )
    model.initialize()
    return model.inference(
        prompt_template=prompt_template,
        dataset=dataset,
        dataset_name=dataset_name,
        n=n,
        logprobs_num=logprobs_num,
        temperature=temperature,
        model_args=model_args,
    )
