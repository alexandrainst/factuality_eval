"""Generate a hallucination dataset.

Usage:
    uv run src/scripts/generate_dataset.py <config_key>=<config_value> ...
"""

import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from factuality_eval.dataset_generation import (
    generate_hallucinations_from_qa_data,
    load_qa_data,
    sample_hallucination_intensities,
)

load_dotenv()


@hydra.main(
    config_path="../../config", config_name="dataset_generation", version_base=None
)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Generate the hallucination dataset
    contexts, questions, answers = load_qa_data(
        base_dataset_id=config.base_dataset.id,
        split=config.base_dataset.split,
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
    )
    intensities = sample_hallucination_intensities(
        mean=config.hallucination_intensity.mean,
        std=config.hallucination_intensity.std,
        size=len(answers),
    )
    dataset = generate_hallucinations_from_qa_data(
        contexts=contexts,
        questions=questions,
        answers=answers,
        intensities=intensities,
        model=config.model,
        temperature=config.temperature,
    )

    # Push the generated dataset to the Hugging Face Hub
    target_dataset_name = config.base_dataset.id.split("/")[-1].replace(":", "-")
    target_repo = f"{config.hub_organisation}/{target_dataset_name}-hallucinated"
    dataset.push_to_hub(repo_id=target_repo, private=config.private)


if __name__ == "__main__":
    main()
