"""Run a baseline hallucination check on gold answers.

Usage:
    uv run src/scripts/baseline.py <config_key>=<config_value> ...
"""

import logging

import hydra
from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import DictConfig

from factuality_eval.dataset_generation import load_qa_data
from factuality_eval.hallucination_detection import (
    detect_hallucinations,
    evaluate_predicted_answers,
)
from factuality_eval.prompt_utils import PromptUtils

load_dotenv()

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Run a baseline where gold answers are checked for hallucinations."""
    logging.getLogger("httpx").setLevel(logging.WARNING)

    base_dataset_id = (
        f"{config.base_dataset.organisation}/{config.base_dataset.id}:{config.language}"
    )

    contexts, questions, answers = load_qa_data(
        base_dataset_id=base_dataset_id,
        split="test",
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
        max_examples=config.generation.max_examples,
    )

    prompts = [
        PromptUtils.format_context(ctx, q, lang=config.language)
        for ctx, q in zip(contexts, questions)
    ]

    dataset = Dataset.from_dict(
        {
            "context": contexts,
            "question": questions,
            "answer": answers,
            "prompt": prompts,
        }
    )

    target_dataset_name = f"{config.base_dataset.id}-synthetic-hallucinations"
    hallucination_detector_hugging_face_path = (
        f"{config.hub_organisation}/"
        f"{config.models.hallu_detect_model}-{target_dataset_name}-{config.language}"
    )

    logger.info("Running hallucination baseline on gold answers...")
    hallucinations = detect_hallucinations(
        dataset, model=hallucination_detector_hugging_face_path
    )
    evaluate_predicted_answers(hallucinations)


if __name__ == "__main__":
    main()
