"""Automatic generation of hallucination datasets."""

import logging
from collections import defaultdict

import numpy as np
from datasets import Dataset, load_dataset
from lettucedetect import HallucinationGenerator
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def load_qa_data(
    base_dataset_id: str,
    split: str,
    context_key: str,
    question_key: str,
    answer_key: str,
    squad_format: bool,
    testing: bool,
) -> tuple[list[list[str]], list[str], list[str]]:
    """Load the base dataset.

    Args:
        base_dataset_id:
            The dataset ID in the format "dataset_name:subset_name" or "dataset_name".
        split:
            The dataset split to load (e.g., "train", "validation", "test").
        context_key:
            The key in the dataset corresponding to the context.
        question_key:
            The key in the dataset corresponding to the question.
        answer_key:
            The key in the dataset corresponding to the answer.
        squad_format:
            Whether the answers are in SQuAD format.
        testing:
            If True, only load a small subset of the data for testing purposes.

    Returns:
        A tuple of (contexts, questions, answers).
    """
    logger.info(f"Loading base dataset {base_dataset_id!r}...")
    dataset_id = base_dataset_id.split(":")[0]
    subset = base_dataset_id.split(":")[1] if ":" in base_dataset_id else None
    ds = load_dataset(path=dataset_id, name=subset, split=split)

    logger.info("Preparing dataset...")
    contexts: list[list[str]] = [[ctx] for ctx in ds[context_key]]
    questions: list[str] = ds[question_key]
    if squad_format:
        answers: list[str] = [
            dict(answer_dict)["text"][0] for answer_dict in ds[answer_key]
        ]
    else:
        answers = ds[answer_key]

    if testing:
        logger.info("Truncating dataset for testing...")
        contexts = contexts[:10]
        questions = questions[:10]
        answers = answers[:10]

    return contexts, questions, answers


def sample_hallucination_intensities(mean: float, std: float, size: int) -> list[float]:
    """Sample hallucination intensities from a clipped Beta distribution.

    Args:
        mean:
            The mean of the Beta distribution.
        std:
            The standard deviation of the Beta distribution.
        size:
            The number of samples to generate.

    Returns:
        A list of sampled hallucination intensities.
    """
    logger.info(
        f"Sampling hallucination intensities with mean {mean:.2f} and standard "
        f"deviation {std:.2f}..."
    )

    # Compute the alpha and beta parameters of the Beta distribution
    n = mean * (1 - mean) / (std**2)
    alpha = mean * n
    beta = (1 - mean) * n

    # Add a small constant to avoid zero intensities
    epsilon = 1e-6
    alpha = max(alpha, epsilon)
    beta = max(beta, epsilon)

    # Sample from the Beta distribution. We add 0.1 as the minimum intensity is 0.1, and
    # the Beta distribution is defined on [0, 1].
    intensities = np.random.beta(a=alpha, b=beta, size=size) + 0.1

    # Clip the intensities to be in the range [0.1, 1.0], as that's the allowed range
    intensities = np.clip(intensities, a_min=0.1, a_max=1.0)

    return intensities.tolist()


def generate_hallucinations_from_qa_data(
    contexts: list[list[str]],
    questions: list[str],
    answers: list[str],
    intensities: list[float],
    model: str,
    temperature: float,
) -> Dataset:
    """Generate hallucinations from given QA data.

    Args:
        contexts:
            A list of contexts, where each context is a list of strings.
        questions:
            A list of questions corresponding to the contexts.
        answers:
            A list of answers corresponding to the questions.
        intensities:
            A list of hallucination intensities for each QA pair.
        model:
            The model name to use for hallucination generation.
        temperature:
            The temperature to use for the model during generation.

    Returns:
        A Dataset containing both original and hallucinated QA pairs.
    """
    logger.info("Generating hallucinations...")
    generator = HallucinationGenerator(model=model, temperature=temperature)
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
    return generated_dataset
