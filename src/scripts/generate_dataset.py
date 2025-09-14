"""Generate a hallucination dataset.

Usage:
    uv run src/scripts/generate_dataset.py <config_key>=<config_value> ...
"""

import logging
from pathlib import Path

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
        f"{config.base_dataset.id.split('/')[-1].replace(':', '-')}-hallucinated"
    )

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
        mean=config.beta_distribution.mean,
        std=config.beta_distribution.std,
        size=len(answers),
    )
    dataset = generate_hallucinations_from_qa_data(
        contexts=contexts,
        questions=questions,
        answers=answers,
        intensities=intensities,
        model=config.model,
        temperature=config.temperature,
        output_jsonl_path=Path("data", "final", f"{target_dataset_name}.jsonl"),
    )

    # Push the generated dataset to the Hugging Face Hub
    dataset.push_to_hub(
        repo_id=f"{config.hub_organisation}/{target_dataset_name}",
        private=config.private,
    )


if __name__ == "__main__":
    main()
