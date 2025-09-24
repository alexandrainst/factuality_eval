"""Detect hallucinations.

Usage:
    uv run src/scripts/detect_hallucinations.py <config_key>=<config_value> ...
"""

import logging
from pathlib import Path

import hydra
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig

from factuality_eval.hallucination_detection import (
    detect_hallucinations,
    evaluate_predicted_answers,
)
from factuality_eval.model_generation import (
    generate_answers_from_qa_data,
    load_model_for_generation,
)

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

    target_dataset_name = f"{config.base_dataset.id}-{config.language}-{config.models.eval_model.split('/')[1]}"

    dataset = load_dataset(
        path=f"{config.hub_organisation}/{config.base_dataset.id}",
        name=config.language,
        split="train",
    )
    test_dataset = dataset.train_test_split(test_size=0.2, seed=42)["test"]

    model, tokenizer = load_model_for_generation(config.models.eval_model)

    generated_dataset = generate_answers_from_qa_data(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        output_jsonl_path=Path("data", "final", f"{target_dataset_name}.jsonl"),
    )

    hallu_detector_hugging_face_path = (
        f"{config.hub_organisation}/"
        f"{config.models.hallu_detect_model}-{config.base_dataset.id}-{config.language}"
    )

    hallucinations = detect_hallucinations(
        generated_dataset, model=hallu_detector_hugging_face_path
    )
    evaluate_predicted_answers(hallucinations)
    return


if __name__ == "__main__":
    main()
