"""Build a human-annotatable ground-truth evaluation dataset.

For every (context, question, answer) triple this module emits one JSONL row
holding three columns of token-level verdicts on the answer:

* ``model_*``: the trained hallucination detector's per-token predictions.
* ``llm_*``: a language model's verdict (filled in by a separate
  ``llm_evaluate_ground_truth.py`` pass that calls the API).
* ``human_*``: empty slots for a human annotator to fill in afterwards.

A reviewer can then read each row and compare the three verdicts directly.
``build_ground_truth_evaluation_dataset`` itself never calls an external
API; ``evaluate_answer_with_llm`` is provided for the dedicated LLM pass.
"""

import json
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


_LLM_EVAL_SYSTEM_PROMPT = (
    "You are a careful annotator that detects hallucinations in question "
    "answering. Given a CONTEXT, a QUESTION, and an ANSWER, identify any "
    "spans of the ANSWER that are not supported by the CONTEXT or that "
    "contradict it. Respond with a single JSON object with two keys: "
    '"hallucinated_parts" (a list of exact substrings copied verbatim from '
    'the ANSWER) and "explanation" (a short string explaining your verdict). '
    "If nothing is hallucinated, return an empty list and an explanation."
)


def evaluate_answer_with_llm(
    client: OpenAI, model: str, context: list[str] | str, question: str, answer: str
) -> dict[str, Any]:
    """Ask an OpenAI model to flag hallucinated spans in an answer.

    Args:
        client: An initialised OpenAI client.
        model: Model name to call (e.g. ``"gpt-4.1-mini"``).
        context: The grounding context. Joined into a single string if a list.
        question: The question the answer is responding to.
        answer: The answer whose hallucinations we want flagged.

    Returns:
        A dict with ``hallucinated_parts`` (list of str) and ``explanation``
        (str). On parse failure the dict contains the raw response text under
        ``raw_response`` and empty defaults.
    """
    context_str = "\n".join(context) if isinstance(context, list) else context
    user_message = (
        f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _LLM_EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("LLM evaluator returned non-JSON content; storing raw text.")
        return {"hallucinated_parts": [], "explanation": "", "raw_response": content}

    return {
        "hallucinated_parts": list(parsed.get("hallucinated_parts", [])),
        "explanation": str(parsed.get("explanation", "")),
    }


def _hallucinated_text_from_tokens(tokens: list[dict[str, Any]]) -> str:
    """Concatenate the tokens that the detector flagged as hallucinated."""
    return "".join(token["token"] for token in tokens if token.get("pred") == 1)


def build_ground_truth_evaluation_dataset(
    hashes: list[str],
    contexts: list[Any],
    questions: list[str],
    answers: list[str],
    predictions: list[list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Write one JSONL row per sample combining detector, LLM, and human slots.

    Each row contains:
        - hash, context, question, answer
        - tokens: detector token-level predictions ({token, pred, prob})
        - model_predicted_hallucinated_text: tokens with pred==1 concatenated
        - llm_hallucinated_parts, llm_explanation: empty placeholders for a
          later LLM evaluation pass
        - human_annotation_labels: null list of token length, for human use
        - human_annotation_notes: empty string for human use
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample_hash, context, question, answer, pred_tokens in zip(
            hashes, contexts, questions, answers, predictions
        ):
            row = {
                "hash": sample_hash,
                "context": context,
                "question": question,
                "answer": answer,
                "tokens": pred_tokens,
                "model_predicted_hallucinated_text": _hallucinated_text_from_tokens(
                    pred_tokens
                ),
                "llm_hallucinated_parts": [],
                "llm_explanation": "",
                "human_annotation_labels": [None] * len(pred_tokens),
                "human_annotation_notes": "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Wrote ground-truth evaluation dataset to {output_path}")
