"""Train utilities for hallucination evaluation."""

from datasets import Dataset

from factuality_eval.prompt_utils import Lang, PromptUtils


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
    return dataset.map(
        lambda x: {
            "prompt": PromptUtils.format_context(
                x["context"], x["question"], lang=language
            ),
            "answer": x["answer"],
            "labels": x["hallucinated_labels"],
            "split": split,
            "task_type": "qa",
            "language": language,
            "dataset": "ragtruth",
        },
        remove_columns=["question", "context"],
    )
