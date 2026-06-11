"""Detect hallucinations.

Usage:
    uv run src/scripts/detect_hallucinations.py <config_key>=<config_value> ...
"""

import logging
from pathlib import Path

import hydra
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig

from factuality_eval.dataset_generation import load_qa_data
from factuality_eval.hallucination_detection import (
    detect_hallucinations,
    evaluate_predicted_answers,
)
from factuality_eval.model_generation import (
    generate_answers_from_prompts,
    generate_answers_from_qa_data,
)
from factuality_eval.train import format_dataset_to_ragtruth_without_labels

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
    if config.get("ragtruth", None) and config.ragtruth.get("enable", False):
        suffix = (
            "-with-ragtruth"
            if config.multiwikiqa.get("enable", False)
            else "-only-ragtruth"
        )
    else:
        suffix = ""

    eval_dataset_cfg = config.get("eval_dataset", None)
    use_eval_dataset = bool(eval_dataset_cfg and eval_dataset_cfg.get("id", None))

    if use_eval_dataset:
        rag_truth_dataset = _build_rag_truth_from_eval_dataset(config, eval_dataset_cfg)
    else:
        rag_truth_dataset = _build_rag_truth_from_multiwikiqa(config)

    hallucination_detector_hugging_face_path = (
        f"{config.hub_organisation}/"
        f"{config.models.hallu_detect_model}-"
        f"{config.base_dataset.id}-synthetic-hallucinations{suffix}-{config.language}"
    )
    hallucinations = detect_hallucinations(
        rag_truth_dataset, model=hallucination_detector_hugging_face_path
    )

    evaluate_predicted_answers(hallucinations)


def _build_rag_truth_from_eval_dataset(
    config: DictConfig, eval_dataset_cfg: DictConfig
) -> Dataset:
    """Build a RAGTruth-format dataset from a pre-formatted evaluation dataset.

    Loads a dataset whose rows already contain a complete RAG ``prompt`` and a
    reference ``answer`` (e.g. ``EuroEval/ragtruth-translated-hallucinations-da-mini``),
    feeds each prompt to the eval model verbatim, and returns the generated
    ``prompt``/``answer`` pairs for the hallucination detector. This mirrors how
    EuroEval evaluates ``--dataset ragtruth-da``.

    Args:
        config:
            The Hydra config.
        eval_dataset_cfg:
            The ``eval_dataset`` sub-config, with ``id``, ``split``,
            ``prompt_key`` and ``answer_key``.

    Returns:
        A dataset with ``prompt`` and ``answer`` columns.
    """
    dataset_id = eval_dataset_cfg.id
    subset = dataset_id.split(":")[1] if ":" in dataset_id else None
    ds = load_dataset(path=dataset_id.split(":")[0], name=subset)
    ds = ds[eval_dataset_cfg.split]

    prompt_key = eval_dataset_cfg.get("prompt_key", "prompt")
    answer_key = eval_dataset_cfg.get("answer_key", "answer")
    prompts = list(ds[prompt_key])
    answers = list(ds[answer_key])

    if config.testing:
        prompts, answers = prompts[:10], answers[:10]
    elif config.generation.max_examples != -1:
        prompts = prompts[: config.generation.max_examples]
        answers = answers[: config.generation.max_examples]

    safe_dataset_name = dataset_id.replace("/", "__").replace(":", "__")
    target_dataset_name = (
        f"{safe_dataset_name}-{config.models.eval_model.split('/')[1]}"
    )

    return generate_answers_from_prompts(
        eval_model=config.models.eval_model,
        prompts=prompts,
        answers=answers,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.get("temperature", None),
        top_p=config.generation.get("top_p", None),
        top_k=config.generation.get("top_k", None),
        output_jsonl_path=Path("data", "final", f"{target_dataset_name}.jsonl"),
    )


def _build_rag_truth_from_multiwikiqa(config: DictConfig) -> Dataset:
    """Build a RAGTruth-format dataset from the multi-wiki-qa base dataset.

    Args:
        config:
            The Hydra config.

    Returns:
        A dataset with ``prompt`` and ``answer`` columns.
    """
    target_dataset_name = (
        f"{config.base_dataset.id}-{config.language}-"
        f"{config.models.eval_model.split('/')[1]}"
    )

    contexts, questions, answers = load_qa_data(
        base_dataset_id=f"{config.base_dataset.organisation}/{config.base_dataset.id}:{config.language}",
        split="test",
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
        max_examples=config.generation.max_examples,
    )

    generated_answers = generate_answers_from_qa_data(
        eval_model=config.models.eval_model,
        contexts=contexts,
        questions=questions,
        answers=answers,
        lang=config.language,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.get("temperature", None),
        top_p=config.generation.get("top_p", None),
        top_k=config.generation.get("top_k", None),
        output_jsonl_path=Path("data", "final", f"{target_dataset_name}.jsonl"),
    )

    return format_dataset_to_ragtruth_without_labels(
        generated_answers, language=config.language, split="test"
    )


if __name__ == "__main__":
    main()
