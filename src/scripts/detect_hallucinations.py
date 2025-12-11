"""Detect hallucinations.

Usage:
    uv run src/scripts/detect_hallucinations.py <config_key>=<config_value> ...
"""

import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from factuality_eval.dataset_generation import load_qa_data
from factuality_eval.hallucination_detection import (
    detect_hallucinations,
    evaluate_predicted_answers,
)
from factuality_eval.model_generation import generate_answers_from_qa_data

load_dotenv()

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)

    target_dataset_name = (
        f"{config.base_dataset.id}-{config.language}-"
        f"{config.models.eval_model.split('/')[1]}"
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

    hallucination_detector_hugging_face_path = (
        f"{config.hub_organisation}/"
        f"{config.models.hallu_detect_model}-{config.base_dataset.id}-{config.language}"
    )

    hallucinations = detect_hallucinations(
        generated_answers, model=hallucination_detector_hugging_face_path
    )

    evaluate_predicted_answers(hallucinations)


if __name__ == "__main__":
    main()
