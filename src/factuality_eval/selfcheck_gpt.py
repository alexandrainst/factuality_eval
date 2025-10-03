"""SelfCheckGPT prompt-based evaluation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from openai import OpenAI

from factuality_eval.train import PromptUtils

logger = logging.getLogger(__name__)


@dataclass
class PromptVerdict:
    """Result of querying the LLM for a context/answer pair."""

    sample_index: int
    context: str
    raw_response: str | None
    label: str
    score: float


class SelfCheckGPTEvaluator:
    """Convenience wrapper around the OpenAI Responses API for SelfCheckGPT prompts."""

    def __init__(self, client: OpenAI, model: str) -> None:
        """Initialize the evaluator.

        Args:
            client:
                An initialized OpenAI client.
            model:
                The model name to use for evaluation.
        """
        self._client = client
        self._model = model

    def score_answer_against_contexts(
        self, answer: str, contexts: Iterable[str]
    ) -> list[PromptVerdict]:
        """Evaluate ``answer`` against a collection of contexts.

        Args:
            answer:
                The model answer we want to verify.
            contexts:
                A sequence of context strings to use in individual prompts.

        Returns:
            List of verdicts with label and numeric score for each context.
        """
        verdicts: list[PromptVerdict] = []
        for idx, context in enumerate(contexts):
            trimmed_context = context.strip()
            prompt = PromptUtils.load_selfcheckgpt_prompt(trimmed_context, answer)
            raw_response = self._call(prompt)
            label, score = self._map_response_to_score(raw_response)
            verdicts.append(
                PromptVerdict(
                    sample_index=idx,
                    context=trimmed_context,
                    raw_response=raw_response,
                    label=label,
                    score=score,
                )
            )

        return verdicts

    def _call(self, prompt: str) -> str | None:
        request_options: dict[str, Any] = {}

        response = self._client.responses.create(
            model=self._model,
            input=prompt,
            temperature=0.0,
            max_output_tokens=16,
            **request_options,
        )
        return response.output_text.strip()

    @staticmethod
    def _map_response_to_score(response: str | None) -> tuple[str, float]:
        if response is None:
            return "N/A", 0.5

        normalized = response.strip().lower()
        if normalized.startswith("ja"):
            return "Ja", 0.0
        if normalized.startswith("nej"):
            return "Nej", 1.0

        return "N/A", 0.5
