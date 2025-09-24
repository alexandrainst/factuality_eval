"""Detect hallucinations.

Usage:
    uv run src/scripts/detect_hallucinations.py <config_key>=<config_value> ...
"""

import json
import logging
import os
from pathlib import Path

from datasets.utils.extract import TarExtractor
import hydra
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from factuality_eval.dataset_generation import (
    generate_answers_from_qa_data,
    load_qa_data,
)
from factuality_eval.hallucination_detection import detect_hallucinations
from scripts.train_hallucination_detector import format_dataset_to_ragtruth

from factuality_eval.prompt_utils import PromptUtils

load_dotenv()


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
        f"{config.base_dataset.id}-{config.language}-{config.models.eval_model}"
    )

    # Load from hub and split into train/test
    contexts, questions, answers = load_qa_data(
        base_dataset_id=config.base_dataset.id,
        split="train",  # Load train split
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
    )

    # Convert to lists to avoid numpy type issues
    contexts = list(contexts)
    questions = list(questions)
    answers = list(answers)
    # Split into train/test (80/20) and use only test portion
    from sklearn.model_selection import train_test_split

    _, contexts, _, questions, _, answers = train_test_split(
        contexts, questions, answers, test_size=0.2, random_state=42, shuffle=False
    )

    for s in range(0, 20):
        # Generate answers from the model to be evaluated
        generated_dataset = generate_answers_from_qa_data(
            contexts=contexts,
            questions=questions,
            answers=answers,
            model=config.models.eval_model,
            temperature=1.0,
            output_jsonl_path=Path(
                "data",
                "final",
                "selfcheckgpt",
                f"{target_dataset_name.split('/')[1]}_sample_{s}.jsonl",
            ),
        )


if __name__ == "__main__":
    main()
