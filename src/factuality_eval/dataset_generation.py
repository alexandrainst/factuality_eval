"""Automatic generation of hallucination datasets."""

import hashlib
import inspect
import json
import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from lettucedetect import HallucinationSample
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _patch_openai_create_with_reasoning_effort(
    create_fn: Callable, reasoning_effort: str
) -> Callable:
    """Wrap an OpenAI create callable to inject reasoning_effort by default."""
    if inspect.iscoroutinefunction(create_fn):

        @wraps(create_fn)
        async def _wrapped_async(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
            kwargs.setdefault("reasoning_effort", reasoning_effort)
            return await create_fn(*args, **kwargs)

        return _wrapped_async

    @wraps(create_fn)
    def _wrapped_sync(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
        kwargs.setdefault("reasoning_effort", reasoning_effort)
        return create_fn(*args, **kwargs)

    return _wrapped_sync


def _configure_generator_reasoning_effort(
    generator: object, reasoning_effort: str | None
) -> None:
    """Inject reasoning_effort into rag_fact_checker OpenAI calls from this repo.

    This avoids patching external packages while letting us control reasoning effort
    for GPT reasoning models used in hallucination generation.
    """
    if not reasoning_effort:
        return

    rag = getattr(generator, "rag", None)
    if rag is None:
        logger.warning(
            "Could not configure reasoning effort=%r because generator.rag is missing.",
            reasoning_effort,
        )
        return

    components = [
        getattr(rag, "answer_generator", None),
        getattr(rag, "reference_generator", None),
        getattr(rag, "fact_checker", None),
        getattr(rag, "triplet_generator", None),
    ]

    patched_count = 0
    for component in components:
        if component is None:
            continue
        for client_attr in ("model", "async_model"):
            client = getattr(component, client_attr, None)
            completions = getattr(getattr(client, "chat", None), "completions", None)
            create_fn = getattr(completions, "create", None)
            if create_fn is None:
                continue
            if getattr(create_fn, "_factuality_reasoning_patched", False):
                continue
            try:
                wrapped = _patch_openai_create_with_reasoning_effort(
                    create_fn=create_fn, reasoning_effort=reasoning_effort
                )
                setattr(wrapped, "_factuality_reasoning_patched", True)
                setattr(completions, "create", wrapped)
                patched_count += 1
            except Exception as e:
                logger.warning(
                    "Failed to patch %s.%s create() with reasoning effort %r: %s",
                    component.__class__.__name__,
                    client_attr,
                    reasoning_effort,
                    e,
                )

    if patched_count == 0:
        logger.warning(
            "No OpenAI create() call sites were patched for reasoning effort=%r.",
            reasoning_effort,
        )
    else:
        logger.info(
            "Configured reasoning effort=%r for %d OpenAI call sites.",
            reasoning_effort,
            patched_count,
        )


def load_qa_data(
    base_dataset_id: str,
    split: str,
    context_key: str,
    question_key: str,
    answer_key: str,
    squad_format: bool,
    testing: bool,
    max_examples: int = -1,
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
        max_examples:
            Maximum number of data samples. If -1, it will use all samples.

    Returns:
        A tuple of (contexts, questions, answers).
    """
    logger.info(f"Loading base dataset {base_dataset_id!r}...")
    dataset_id = base_dataset_id.split(":")[0]
    subset = base_dataset_id.split(":")[1] if ":" in base_dataset_id else None

    ds = load_dataset(path=dataset_id, name=subset)

    if len(ds.keys()) > 1:  # Dataset is already split
        ds = ds[split]
    elif "train" in ds:
        ds = ds["train"].train_test_split(test_size=0.2, seed=42)[split]
    else:
        raise ValueError(
            "Dataset cannot be split into test and train. Please check if "
            "'train' is a subset of the dataset."
        )

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
    elif max_examples != -1:
        logger.info(f"Truncating dataset to {max_examples} examples...")
        contexts = contexts[:max_examples]
        questions = questions[:max_examples]
        answers = answers[:max_examples]

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
    output_jsonl_path: Path | None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    max_workers: int = 8,
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
        output_jsonl_path:
            The path to save the generated dataset in JSONL format, or None to skip
            saving.
        temperature:
            The temperature to use for the model during generation. If None, the
            default temperature is used. Defaults to None.
        reasoning_effort:
            Optional OpenAI reasoning effort for reasoning-capable models
            (e.g. ``"low"``, ``"medium"``, ``"high"``). If None, API defaults
            are used.
        max_workers:
            Number of parallel threads to use for generation. Defaults to 8.

    Returns:
        A Dataset containing both original and hallucinated QA pairs.
    """
    logger.info("Generating hallucinations...")

    from lettucedetect import HallucinationGenerator

    generator = HallucinationGenerator(model=model, temperature=temperature)
    _configure_generator_reasoning_effort(
        generator=generator, reasoning_effort=reasoning_effort
    )
    records: list[dict] = list()

    # Load the existing dataset if it exists
    if output_jsonl_path is not None and output_jsonl_path.exists():
        logger.info(f"Loading existing dataset from {output_jsonl_path}...")
        with output_jsonl_path.open() as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

    # Extract the list of hashes for quick lookups
    hashes: set[str] = {record["hash"] for record in records}

    # Build the list of items that still need to be processed
    items_to_process = [
        (context, question, answer, intensity)
        for context, question, answer, intensity in zip(
            contexts, questions, answers, intensities
        )
        if generate_hash(context=context, question=question, answer=answer)
        not in hashes
    ]

    file_lock = threading.Lock()
    records_lock = threading.Lock()

    def _process_one(item: tuple[list[str], str, str, float]) -> dict | None:
        """Process a single QA pair and return a record or None to skip."""
        context, question, answer, intensity = item
        hash_ = generate_hash(context=context, question=question, answer=answer)

        # Generate hallucinated answer with specified intensity
        try:
            result = generator.generate(
                context=context, question=question, answer=answer, intensity=intensity
            )
        except Exception as e:
            logger.error(f"Error during generation: {e}. Skipping...")
            return None

        labels_result = get_hallucinated_labels(hallucinated_dict=result)

        # Skip samples where labels cannot be reliably determined
        if labels_result is None:
            return None

        hallucinated_labels, clean_parts = labels_result

        return dict(
            hash=hash_,
            context=context,
            question=question,
            answer=answer,
            hallucinated_answer=result["hallucinated_answer"],
            hallucinated_parts=clean_parts,
            hallucinated_labels=hallucinated_labels,
            intensity=intensity,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, item): item for item in items_to_process
        }
        with tqdm(total=len(items_to_process)) as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                record = future.result()
                if record is None:
                    continue
                with records_lock:
                    records.append(record)
                    hashes.add(record["hash"])
                if output_jsonl_path is not None:
                    with file_lock:
                        with output_jsonl_path.open("a") as f:
                            f.write(json.dumps(record) + "\n")

    # Remove records where the hallucinated answer is identical to the original answer
    records = [
        record
        for record in records
        if record["hallucinated_answer"].strip() != record["answer"].strip()
    ]

    # Convert records to a Dataset
    data_dict: dict[str, list] = defaultdict(list)
    for record in records:
        # Non-hallucinated example
        data_dict["context"].append(record["context"])
        data_dict["question"].append(record["question"])
        data_dict["answer"].append(record["answer"])
        data_dict["intensity"].append(float("nan"))
        data_dict["hallucination"].append(False)
        data_dict["hallucinated_parts"].append([])
        data_dict["hallucinated_labels"].append([])

        # Hallucinated example
        data_dict["context"].append(record["context"])
        data_dict["question"].append(record["question"])
        data_dict["answer"].append(record["hallucinated_answer"])
        data_dict["intensity"].append(record["intensity"])
        data_dict["hallucination"].append(True)
        data_dict["hallucinated_parts"].append(record["hallucinated_parts"])
        data_dict["hallucinated_labels"].append(record["hallucinated_labels"])

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


def get_hallucinated_labels(
    hallucinated_dict: dict,
) -> tuple[list[dict], list[str]] | None:
    """Get the hallucinated labels from the generation result.

    Filters out parts that are absent from the hallucinated answer, discards
    ambiguous parts that appear more than once, and removes redundant parts
    whose spans are fully overlapped by a longer accepted part (e.g. '2005'
    and '2008' when '2005/2008' is already labelled).

    Args:
        hallucinated_dict:
            The dictionary from the hallucination generator.

    Returns:
        A tuple of (labels, clean_parts) where labels is a list of dicts with
        start, end, and label for each accepted hallucinated span and clean_parts
        is the corresponding filtered list of hallucinated part strings, or None
        if the labels cannot be reliably determined.
    """
    answer = hallucinated_dict["hallucinated_answer"]
    raw_parts = hallucinated_dict["hallucinated_parts"]

    # First pass: drop parts absent from the answer or ambiguously duplicated.
    present_parts: list[str] = []
    for part in raw_parts:
        count = answer.count(part)
        if count == 0:
            logger.warning(
                f"Skipping hallucinated part {part!r} — not found in answer."
            )
        elif count > 1:
            logger.warning(
                f"Discarding sample - hallucinated part {part!r} appears {count} times "
                f"in answer, cannot determine which occurrence is hallucinated."
            )
            return None
        else:
            present_parts.append(part)

    # Second pass: sort by length descending so longer (more specific) spans take
    # priority, then skip any span that overlaps with an already-accepted span.
    present_parts.sort(key=len, reverse=True)

    accepted_spans: list[tuple[int, int]] = []
    hallucinated_labels: list[dict] = []
    clean_parts: list[str] = []

    for part in present_parts:
        start = answer.find(part)
        end = start + len(part)
        if any(not (end <= s or start >= e) for s, e in accepted_spans):
            logger.warning(
                f"Skipping redundant/overlapping hallucinated part {part!r}."
            )
            continue
        accepted_spans.append((start, end))
        hallucinated_labels.append(
            {"start": start, "end": end, "label": "hallucinated"}
        )
        clean_parts.append(part)

    # Re-sort labels and parts by position in the answer for a natural order.
    order = sorted(
        range(len(hallucinated_labels)), key=lambda i: hallucinated_labels[i]["start"]
    )
    hallucinated_labels = [hallucinated_labels[i] for i in order]
    clean_parts = [clean_parts[i] for i in order]

    return hallucinated_labels, clean_parts


def generate_lettucedetect_hallucination_samples(
    dataset_split: Dataset,
) -> list[HallucinationSample]:
    """Generate hallucination samples for the LettuceDetect model.

    Args:
        dataset_split: The dataset split to generate samples from.

    Returns:
        A list of hallucination samples.
    """
    samples = []
    for item in dataset_split:
        sample = HallucinationSample(
            prompt=item["prompt"],
            answer=item["answer"],
            labels=item["labels"],
            split=item["split"],
            task_type=item["task_type"],
            dataset=item["dataset"],
            language=item["language"],
        )
        samples.append(sample)
    return samples
