"""Generate hallucination-annotated dataset.

Usage:
    uv run src/scripts/generate_hallucination_dataset.py <config_key>=<config_value> ...

The script generates answers with the eval model and then tags hallucinated
segments using the hallucination classifier. The output JSONL contains per-row
question, context, answer, and hallucinated tokens with their character spans
(relative to the answer string).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from datasets import Dataset
from dotenv import load_dotenv
from lettucedetect.models.inference import HallucinationDetector
from omegaconf import DictConfig

from factuality_eval.dataset_generation import load_qa_data
from factuality_eval.model_generation import generate_answers_from_qa_data
from factuality_eval.prompt_utils import Lang, PromptUtils

load_dotenv()

logger = logging.getLogger(__name__)


def _build_detector_model_path(config: DictConfig) -> str:
    target_dataset_name = f"{config.base_dataset.id}-synthetic-hallucinations"
    return (
        f"{config.hub_organisation}/"
        f"{config.models.hallu_detect_model}-{target_dataset_name}-{config.language}"
    )


def _format_context(context: list | tuple | str) -> list[str]:
    """Normalize an arbitrary context value to a list of strings."""
    if isinstance(context, (list, tuple)):
        return [str(c) for c in context]
    return [str(context)]


def _extract_hallucinated_tokens(
    detector: HallucinationDetector,
    context: list[str],
    question: str,
    answer: str,
    lang: Lang,
) -> list[dict[str, Any]]:
    prompt = PromptUtils.format_context(context, question, lang)

    # Use detector's span output to avoid offset drift between our formatter and the model tokenizer.
    spans = detector.predict_prompt(prompt=prompt, answer=answer, output_format="spans")

    # Normalize span schema without confidence/probability.
    normalized: list[dict[str, Any]] = []
    for span in spans:
        normalized.append(
            {
                "text": span.get("text", ""),
                "start": span.get("start", 0),
                "end": span.get("end", 0),
            }
        )

    return normalized


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Run hallucination annotation over the configured QA dataset."""
    logging.getLogger("httpx").setLevel(logging.WARNING)

    target_dataset_name = (
        f"{config.base_dataset.id}-{config.language}-"
        f"{config.models.eval_model.split('/')[1]}"
    )

    logger.info(
        "Loading dataset %s for hallucination annotation...",
        f"{config.base_dataset.organisation}/{config.base_dataset.id}:{config.language}",
    )

    contexts, questions, answers = load_qa_data(
        base_dataset_id=f"{config.base_dataset.organisation}/{config.base_dataset.id}:{config.language}",
        split="test",
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
        max_examples=config.generation.max_examples,
    )

    generated_answers = generate_answers_from_qa_data(
        eval_model=config.models.eval_model,
        contexts=contexts,
        questions=questions,
        answers=answers,
        lang=config.language,
        max_new_tokens=config.generation.max_new_tokens,
        output_jsonl_path=Path("data", "final", f"{target_dataset_name}.jsonl"),
    )

    detector_model_path = _build_detector_model_path(config)
    logger.info("Loading hallucination detector: %s", detector_model_path)

    if len(generated_answers) == 0:
        logger.warning("No generated answers to annotate. Skipping.")
        return

    detector = HallucinationDetector(
        method="transformer",
        model_path=detector_model_path,
        device_map="auto",
        torch_dtype="auto",
    )

    # Build a lookup from question text to ground-truth answer so that
    # the ground truth stays aligned even when generate_answers_from_qa_data
    # reorders or skips entries (e.g. from caching or errors).
    gt_lookup: dict[str, str] = {q: a for q, a in zip(questions, answers)}

    records: list[dict[str, Any]] = []
    logger.info("Annotating %d generated answers...", len(generated_answers))

    for context, question, answer in zip(
        generated_answers["context"],
        generated_answers["question"],
        generated_answers["answer"],
    ):
        ground_truth_answer = gt_lookup.get(question, "")
        formatted_context = _format_context(context)
        hallucinated_tokens = _extract_hallucinated_tokens(
            detector=detector,
            context=formatted_context,
            question=question,
            answer=answer,
            lang=config.language,
        )

        records.append(
            {
                "context": formatted_context,
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "answer": answer,
                "hallucinated_tokens": hallucinated_tokens,
            }
        )

    output_path = Path("data", "final", f"{target_dataset_name}-hallucinations.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing hallucination-annotated dataset to %s", output_path)
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(str(output_path) + "_hf")

    logger.info("Done. Wrote %d records.", len(records))


if __name__ == "__main__":
    main()
