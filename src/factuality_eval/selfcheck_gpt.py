"""SelfCheckGPT prompt-based evaluation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from openai import OpenAI

from factuality_eval.train import PromptUtils

from factuality_eval.prompt_utils import YES_WORDS, NO_WORDS
logger = logging.getLogger(__name__)


@dataclass
class PromptVerdict:
    """Result of querying the LLM for a context/answer pair."""

    sample_index: int
    sentence: str
    sample_answer: str
    judge_response: str | None
    score: float

class SelfCheckGPTEvaluator:
    """Convenience wrapper around the OpenAI Responses API for SelfCheckGPT prompts."""

    def __init__(self, client: OpenAI, model: str, lang: str) -> None:
        """Initialize the evaluator.

        Args:
            client:
                An initialized OpenAI client.
            model:
                The model name to use for evaluation.
        """
        self._client = client
        self._model = model
        self.lang = lang
        
        
    def score_samples_against_reference(
        self, reference: dict[str, str], samples: list[dict[str, str]]
    ) -> list[PromptVerdict]:
        """Evaluate ``reference`` against a collection of samples.

        Args:
            reference:
                A dictionary with at least an "answer" key.
            samples:
                A list of dictionaries, each with at least an "answer" key.

        Returns:
            List of verdicts with label and numeric score for each context.
        """
        verdicts: list[PromptVerdict] = []
        
        sentences = reference["answer"].split('.')
        for idx, sentence in enumerate(sentences):
            
            for s_idx, sample in enumerate(samples):

                prompt = PromptUtils.load_selfcheckgpt_prompt(sentence, sample["answer"], self.lang)
                judge_response = self._call(prompt)
                score = self._map_response_to_score(judge_response)
                verdicts.append(
                    PromptVerdict(
                        sample_index=s_idx,
                        sentence=sentence,
                        sample_answer=sample["answer"],
                        judge_response=judge_response,
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

    def _map_response_to_score(self, response: str | None) -> float:
        if response is None:
            return 0.5

        yes_word = YES_WORDS.get(self.lang)
        no_word = NO_WORDS.get(self.lang)
        
        normalized = response.strip().lower()
        if normalized.startswith(yes_word):
            return 0.0
        if normalized.startswith(no_word):
            return 1.0

        return 0.5
