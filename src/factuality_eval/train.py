"""Train utilities for hallucination evaluation."""

import typing as t

from datasets import Dataset

from .prompt_utils import Lang, PromptUtils


def format_dataset_to_ragtruth(
    dataset: Dataset, language: Lang = "da", split: str = "train"
) -> Dataset:
    """Format the dataset to ragtruth format.

    Args:
        dataset:
            The dataset to format to ragtruth.

    Returns:
        The ragtruth formatted dataset.
    """

    def _format_row(x: dict[str, t.Any]) -> dict[str, t.Any]:
        return {
            "prompt": PromptUtils.format_context(
                x["context"], x["question"], lang=language
            ),
            "answer": x["answer"],
            "labels": x["hallucinated_labels"],
            "split": split,
            "task_type": "qa",
            "language": language,
            "dataset": "ragtruth",
        }

    return dataset.map(_format_row, remove_columns=["question", "context"])
