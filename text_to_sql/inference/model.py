import os
from loguru import logger

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC


class Model:

    def __init__(self, model_name: str, model_parameters: dict,
                 model_parallelism: bool = True):
        """
        Args:
            model_name: HuggingFace model identifier or local path forwarded to
                vLLM (e.g. ``"Qwen/Qwen2.5-Coder-7B-Instruct"``).
            model_parameters: Generation config dict.
            model_parallelism: When ``True``, uses ``tensor_parallel_size=2``
                to split the model across two GPUs; ``False`` runs on one GPU.
        """
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.batch_size = 4
        self.model_parallelism = model_parallelism

        self.model = None

    def _initialize_model_parameters(self):

        default_parameters = {
            "temperature": 0.0,
            "max_tokens": 1024
        }

        for default_parameter, default_value in default_parameters.items():
            if default_parameter not in self.model_parameters:
                self.model_parameters[default_parameter] = default_value

    def initialize(self):
        """Load the model into GPU memory via vLLM"""
        self._initialize_model_parameters()

        self.model = LLM(
            model=self.model_name,
            dtype="auto",
            tensor_parallel_size=2 if self.model_parallelism else 1,
            max_num_seqs=32,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            # Where the model weights are downloaded to (default: the Hugging Face cache).
            download_dir=os.environ.get("VLLM_DOWNLOAD_DIR"),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _build_prompts(self, prompt_template: PromptABC, dataset: pd.DataFrame, dataset_name: str):
        """Build chat-message lists for every row in the dataset.

        Iterates over the dataset and calls ``prompt_template.chat_message`` for
        each row. Rows that raise ``ValueError`` during prompt construction are
        silently dropped and a warning is logged.

        Args:
            prompt_template: Prompt to be used in the input.
            dataset: DataFrame with at least ``question``, ``db_path`` columns.
            dataset_name: Dataset identifier passed through to the prompt template.

        Returns:
            tuple[list[list[dict]], pd.DataFrame]:
                - ``model_inputs``: One chat-message list per row.
                - ``dataset``: The input DataFrame with error rows removed and
                  index reset so it aligns positionally with ``model_inputs``.
        """
        error_data_points = []
        model_inputs = []
        for i, row in tqdm(dataset.iterrows(), desc="Building prompts..."):
            try:
                model_inputs.append(prompt_template.chat_message(
                    question=row["question"],
                    db_path=row["db_path"],
                    question_id=row.get("question_id", None),
                    dataset_name=dataset_name,
                    hint=row.get("hint", ""),
                    generated_sql=row.get("generated_sql", ""),
                ))
            except ValueError:
                error_data_points.append(i)

        dataset = dataset.drop(index=error_data_points).reset_index(drop=True)
        logger.warning(f" {len(error_data_points)} errors found in building the prompt. The corresponding queries will be removed from the dataset.")

        if error_data_points and not model_inputs:
            raise RuntimeError(
                f"All {len(error_data_points)} rows failed to build a prompt (e.g. their "
                "db_path doesn't exist on this machine). Refusing to call the model with an "
                "empty batch, which would silently write an empty results file."
            )
        return model_inputs, dataset

    def _build_result_dict(self, row, model_input, generated_output, generated_tokens,
                           token_log_probs, top_tokens_probs) -> dict:
        """Create a result dict from a dataset row and its model output.

        Args:
            row: A single pandas Series from the dataset (one NL-to-SQL example).
            model_input: Chat-message list that was sent to the model.
            generated_output: Raw text string produced by the model.
            generated_tokens: Sequence of decoded token strings in generation order.
            token_log_probs: Per-token log probability of the chosen token at each
                decoding step.
            top_tokens_probs: Per-step top-k alternatives; each entry is a dict
                with ``"top_tokens"`` (list[str]) and ``"top_log_probs"``
                (list[float]), sorted descending by log prob.

        Returns:
            dict with keys ``question``, ``sql_query``, ``db_path``,
            ``model_input``, ``generated_output``, ``generated_tokens``,
            ``token_log_probs``, ``top_tokens_probs``, plus any of
            ``is_ambiguous``, ``ambig_type``, ``unanswerable``,
            ``origin_dataset`` that are present in ``row``.
        """
        result = {
            "question": row["question"],
            "sql_query": row["query"],
            "db_path": row["db_path"],
            "model_input": model_input,
            "generated_output": generated_output,
            "generated_tokens": generated_tokens,
            "token_log_probs": token_log_probs,
            "top_tokens_probs": top_tokens_probs,
        }
        for optional_field in (
            "is_ambiguous", "ambig_type", "unanswerable", "origin_dataset",
            "generated_sql", "exec_accuracy", "exec_error", "question_id", "hint"
        ):
            if optional_field in row.index:
                result[optional_field] = row[optional_field]
        return result

    def inference(
        self,
        prompt_template: PromptABC,
        dataset: pd.DataFrame,
        dataset_name: str,
        n: int = 5,
        logprobs_num: int = 1,
        temperature: float = 1.0,
        model_args: dict | None = None,
    ) -> list[list[dict]]:
        """Generate n SQL completions per question with per-token log probabilities.

        Builds prompts via :meth:`_build_prompts`, submits them to vLLM in a
        single batched call, and unpacks the ``n`` completions per question into
        result dicts via :meth:`_build_result_dict`.

        Args:
            prompt_template: Prompt format to be used in the inference input.
            dataset: DataFrame with at least ``question``, ``query``, ``db_path``
                columns.
            dataset_name: Dataset identifier forwarded to the prompt template.
            n: Number of independent completions to generate per question.
            logprobs_num: Number of top-token log-probability entries to capture
                at each decoding step (vLLM ``SamplingParams.logprobs``).
            temperature: Sampling temperature.
            model_args: Extra keyword arguments forwarded directly to vLLM
                ``SamplingParams`` (e.g. ``top_p``, ``top_k``).

        Returns:
            Nested list of shape ``[num_questions][n]``. Each element is a dict with keys: ``question``,
            ``sql_query``, ``db_path``, ``model_input``, ``generated_output``,
            ``generated_tokens``, ``token_log_probs``, ``top_tokens_probs``, and
            any optional dataset fields present in the row.
        """
        all_results: list[list[dict]] = []

        # Convert each dataset row into a chat-message list; rows that fail are dropped.
        model_inputs, dataset = self._build_prompts(prompt_template, dataset, dataset_name)

        # Build sampling params; pass any extra caller-supplied args through.
        _args = (model_args or {}).copy()

        if "max_tokens" not in _args:
            _args["max_tokens"] = 1024

        sampling_params = SamplingParams(
            temperature=temperature,
            n=n,
            logprobs=logprobs_num,
            **_args,
        )

        outputs = self.model.generate(
            [self.tokenizer.apply_chat_template(mi, add_generation_prompt=True, tokenize=False) for mi in model_inputs],
            sampling_params,
        )

        # For every question, unpack the n completions into result dicts and append to the output list.
        for i, output in enumerate(outputs):
            row = dataset.iloc[i]
            row_results = []

            # Each completion is one of the n independent samples for this question.
            for completion in output.outputs:
                token_log_probs = []
                generated_tokens = []
                top_tokens_probs = []

                for t, step_logprobs in enumerate(completion.logprobs):
                    token_id = completion.token_ids[t]
                    lp = step_logprobs[token_id]
                    token_log_probs.append(lp.logprob)
                    generated_tokens.append(lp.decoded_token)

                    # Sort all returned candidates by log prob (descending) so
                    # callers can inspect the top-k distribution at each step.
                    sorted_entries = sorted(
                        step_logprobs.values(),
                        key=lambda obj: obj.logprob,
                        reverse=True,
                    )
                    top_tokens_probs.append({
                        "top_tokens": [e.decoded_token for e in sorted_entries],
                        "top_log_probs": [e.logprob for e in sorted_entries],
                    })

                row_results.append(self._build_result_dict(
                    row, model_inputs[i], completion.text, generated_tokens,
                    token_log_probs, top_tokens_probs,
                ))

            all_results.append(row_results)

        return all_results
