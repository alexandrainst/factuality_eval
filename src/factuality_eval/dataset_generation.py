"""Automatic generation of hallucination datasets."""

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

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
    output_jsonl_path: Path | None,
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
        output_jsonl_path:
            The path to save the generated dataset in JSONL format, or None to skip
            saving.

    Returns:
        A Dataset containing both original and hallucinated QA pairs.
    """
    logger.info("Generating hallucinations...")

    generator = HallucinationGenerator(model=model, temperature=temperature)
    records: list[dict] = list()

    # Load the existing dataset if it exists
    if output_jsonl_path is not None and output_jsonl_path.exists():
        logger.info(f"Loading existing dataset from {output_jsonl_path}...")
        with output_jsonl_path.open() as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

    # Extract the list of hashes for quick lookups
    hashes = {record["hash"] for record in records}

    for context, question, answer, intensity in zip(
        tqdm(contexts), questions, answers, intensities
    ):
        hash_ = generate_hash(context=context, question=question, answer=answer)
        if hash_ in hashes:
            continue

        # Generate hallucinated answer with specified intensity
        for _ in range(num_attempts := 100):
            try:
                result = generator.generate(
                    context=context,
                    question=question,
                    answer=answer,
                    intensity=intensity,
                )
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                continue
            break
        else:
            raise RuntimeError(
                f"Failed to generate hallucination after {num_attempts} attempts. "
                "Please try again."
            )

        # Save the record
        record = dict(
            hash=hash_,
            context=context,
            question=question,
            answer=answer,
            hallucinated_answer=result["hallucinated_answer"],
            intensity=intensity,
        )
        records.append(record)
        hashes.add(hash_)
        if output_jsonl_path is not None:
            with output_jsonl_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    # Convert records to a Dataset
    data_dict: dict[str, list] = defaultdict(list)
    for record in records:
        # Non-hallucinated example
        data_dict["context"].append(record["context"])
        data_dict["question"].append(record["question"])
        data_dict["answer"].append(record["answer"])
        data_dict["intensity"].append(float("nan"))
        data_dict["hallucination"].append(False)

        # Hallucinated example
        data_dict["context"].append(record["context"])
        data_dict["question"].append(record["question"])
        data_dict["answer"].append(record["hallucinated_answer"])
        data_dict["intensity"].append(record["intensity"])
        data_dict["hallucination"].append(True)
    generated_dataset = Dataset.from_dict(mapping=data_dict)

    return generated_dataset


def generate_hash(context: list[str], question: str, answer: str) -> str:
    """Generate a unique hash for a QA pair.

    Args:
        context:
            The context as a list of strings.
        question:
            The question string.
        answer:
            The answer string.

    Returns:
        A unique hash string for the QA pair.
    """
    return hashlib.md5((context[0] + question + answer).encode("utf-8")).hexdigest()
