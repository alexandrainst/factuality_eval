"""Tests for the `generate_dataset` module."""

import pytest

from factuality_eval.dataset_generation import (
    generate_hallucinations_from_qa_data,
    load_qa_data,
    sample_hallucination_intensities,
)


@pytest.mark.parametrize(
    argnames=[
        "base_dataset_id",
        "split",
        "context_key",
        "question_key",
        "answer_key",
        "squad_format",
    ],
    argvalues=[
        ("rajpurkar/squad", "train", "context", "question", "answers", True),
        ("rajpurkar/squad", "validation", "context", "question", "answers", True),
        (
            "alexandrainst/multi-wiki-qa:da",
            "train",
            "context",
            "question",
            "answers",
            True,
        ),
    ],
    ids=["normal", "validation-split", "dataset-with-subset"],
)
def test_load_qa_data(
    base_dataset_id: str,
    split: str,
    context_key: str,
    question_key: str,
    answer_key: str,
    squad_format: bool,
) -> None:
    """Test the `load_qa_data` function."""
    contexts, questions, answers = load_qa_data(
        base_dataset_id=base_dataset_id,
        split=split,
        context_key=context_key,
        question_key=question_key,
        answer_key=answer_key,
        squad_format=squad_format,
        testing=True,
    )
    assert len(contexts) == len(questions) == len(answers), (
        f"Expected equal lengths, but got {len(contexts)}, {len(questions)}, and "
        f"{len(answers)}."
    )
    assert all(isinstance(ctx, list) for ctx in contexts), (
        "All contexts must be lists of strings."
    )
    assert all(isinstance(q, str) for q in questions), "All questions must be strings."
    assert all(isinstance(a, str) for a in answers), "All answers must be strings."


@pytest.mark.parametrize(
    argnames=["mean", "std", "size"],
    argvalues=[(0.5, 0.1, 10), (1.0, 0.2, 20), (0.0, 0.5, 5)],
    ids=["normal", "larger-size", "zero-mean"],
)
def test_sample_hallucination_intensities(mean: float, std: float, size: int) -> None:
    """Test the `sample_hallucination_intensities` function."""
    intensities = sample_hallucination_intensities(mean=mean, std=std, size=size)
    assert len(intensities) == size, (
        f"Expected {size} intensities, but got {len(intensities)}."
    )
    assert all(isinstance(i, float) for i in intensities), (
        "All intensities must be floats."
    )
    assert all(0.1 <= i <= 1.0 for i in intensities), (
        "All intensities must be between 0.1 and 1.0."
    )


@pytest.mark.parametrize(
    argnames=[
        "contexts",
        "questions",
        "answers",
        "intensities",
        "model",
        "temperature",
    ],
    argvalues=[
        (
            [["He was born in 1990."]],
            ["When was he born?"],
            ["1990"],
            [0.5],
            "gpt-4o-mini",
            0.7,
        ),
        (
            [["He was born in 1990."]],
            ["When was he born?"],
            ["1990"],
            [0.5],
            "gpt-4o-mini",
            0.0,
        ),
        (
            [["He was born in 1990."], ["The capital of France is Paris."]],
            ["When was he born?", "What is the capital of France?"],
            ["1990", "Paris"],
            [0.5, 0.8],
            "gpt-4o-mini",
            0.7,
        ),
    ],
    ids=["single-example", "temperature=0", "multiple-examples"],
)
def test_generate_hallucinations_from_qa_data(
    contexts: list[list[str]],
    questions: list[str],
    answers: list[str],
    intensities: list[float],
    model: str,
    temperature: float,
) -> None:
    """Test the `generate_hallucinations_from_qa_data` function."""
    assert len(contexts) == len(questions) == len(answers) == len(intensities), (
        "The lengths of contexts, questions, answers, and intensities must be equal."
    )
    dataset = generate_hallucinations_from_qa_data(
        contexts=contexts,
        questions=questions,
        answers=answers,
        intensities=intensities,
        model=model,
        temperature=temperature,
        output_jsonl_path=None,
    )
    assert len(dataset) == 2 * len(contexts), (
        f"Expected dataset length {2 * len(contexts)}, but got {len(dataset)}."
    )
    assert all("context" in item for item in dataset), (
        "Each item in the dataset must contain a 'context' field."
    )
    assert all("question" in item for item in dataset), (
        "Each item in the dataset must contain a 'question' field."
    )
    assert all("answer" in item for item in dataset), (
        "Each item in the dataset must contain an 'answer' field."
    )
    assert all("intensity" in item for item in dataset), (
        "Each item in the dataset must contain an 'intensity' field."
    )
    assert all("hallucination" in item for item in dataset), (
        "Each item in the dataset must contain a 'hallucination' field."
    )
