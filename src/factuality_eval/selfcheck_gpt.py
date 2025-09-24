"""SelfCheckGPT prompt-based evaluation helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable

from openai import OpenAI

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Context: {context}\n"
    "Sentence: {sentence}\n"
    "Is the sentence supported by the context above? Answer Yes or No:"
)


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

    def __init__(
        self,
        client: OpenAI,
        model: str,
        *,
        max_retries: int = 3,
        request_timeout: float | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_retries = max(1, max_retries)
        self._request_timeout = request_timeout

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
            prompt = PROMPT_TEMPLATE.format(context=trimmed_context, sentence=answer)
            raw_response = self._call_with_retries(prompt)
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

    def _call_with_retries(self, prompt: str) -> str | None:
        last_error: Exception | None = None
        request_options: dict[str, Any] = {}
        if self._request_timeout is not None:
            request_options["timeout"] = self._request_timeout

        for attempt in range(self._max_retries):
            try:
                response = self._client.responses.create(
                    model=self._model,
                    input=prompt,
                    temperature=0.0,
                    max_output_tokens=8,
                    **request_options,
                )
                return response.output_text.strip()
            except Exception as exc:  # noqa: BLE001 - OpenAI raises multiple subclasses
                last_error = exc
                wait_seconds = min(2**attempt, 10)
                logger.warning(
                    "SelfCheckGPT prompt call failed (attempt %s/%s): %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                time.sleep(wait_seconds)

        logger.error(
            "SelfCheckGPT prompt call failed after %s attempts", self._max_retries
        )
        if last_error is not None:
            logger.debug("Last error: %s", last_error)
        return None

    @staticmethod
    def _map_response_to_score(response: str | None) -> tuple[str, float]:
        if response is None:
            return "N/A", 0.5

        normalized = response.strip().lower()
        if normalized.startswith("yes"):
            return "Yes", 0.0
        if normalized.startswith("no"):
            return "No", 1.0

        return "N/A", 0.5
