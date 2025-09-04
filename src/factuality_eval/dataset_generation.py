"""Automatic generation of hallucination datasets."""

import logging
from collections import defaultdict

import numpy as np
from datasets import Dataset, load_dataset
from lettucedetect import HallucinationGenerator
from omegaconf import DictConfig
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def generate_hallucination_dataset(config: DictConfig) -> None:
    """An example function for your project.

    Args:
        config:
            The Hydra config for your project.
    """
    logger.info(f"Loading base dataset {config.base_dataset.id!r}...")
    dataset_id = config.base_dataset.id.split(":")[0]
    subset = (
        config.base_dataset.id.split(":")[1] if ":" in config.base_dataset.id else None
    )
    ds = load_dataset(path=dataset_id, name=subset, split=config.base_dataset.split)

    logger.info("Preparing dataset...")
    contexts: list[list[str]] = [[ctx] for ctx in ds[config.base_dataset.context_key]]
    questions: list[str] = ds[config.base_dataset.question_key]
    if config.base_dataset.squad_format:
        answers: list[str] = [
            dict(answer_dict)["text"][0]
            for answer_dict in ds[config.base_dataset.answer_key]
        ]
    else:
        answers = ds[config.base_dataset.answer_key]

    if config.testing:
        logger.info("Truncating dataset for testing...")
        contexts = contexts[:10]
        questions = questions[:10]
        answers = answers[:10]

    logger.info("Sampling hallucination intensities...")
    np.random.seed(42)
    mean = config.beta_distribution.mean
    std = config.beta_distribution.std
    n = mean * (1 - mean) / (std**2)
    alpha = mean * n
    beta = (1 - mean) * n
    intensities = np.clip(
        np.random.beta(a=alpha, b=beta, size=len(contexts)) + 0.1, a_min=0.1, a_max=1.0
    )

    logger.info("Generating hallucinations...")
    generator = HallucinationGenerator(
        model=config.model, temperature=config.temperature
    )
    data_dict: dict[str, list] = defaultdict(list)

    for context, question, answer, intensity in zip(
        tqdm(contexts), questions, answers, intensities
    ):
        # Generate hallucinated answer with specified intensity
        result = generator.generate(
            context=context, question=question, answer=answer, intensity=intensity
        )

        # Original non-hallucinated example
        data_dict["context"].append(context)
        data_dict["question"].append(question)
        data_dict["answer"].append(answer)
        data_dict["hallucination"].append(False)
        data_dict["intensity"].append(float("nan"))

        # Hallucinated example
        data_dict["context"].append(context)
        data_dict["question"].append(question)
        data_dict["answer"].append(result["hallucinated_answer"])
        data_dict["hallucination"].append(True)
        data_dict["intensity"].append(intensity)

    generated_dataset = Dataset.from_dict(mapping=data_dict)

    logger.info("Pushing dataset to the Hub...")
    target_dataset_name = config.base_dataset.id.split("/")[-1].replace(":", "-")
    target_repo = f"{config.hub_organisation}/{target_dataset_name}-hallucinated"
    generated_dataset.push_to_hub(repo_id=target_repo, private=config.private)
