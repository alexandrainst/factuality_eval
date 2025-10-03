"""Run the SelfCheckGPT prompt-based evaluation pipeline.

Usage:
    uv run src/scripts/selfCheckGPT.py <config_key>=<config_value> ...
"""

import json
import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI

from factuality_eval.dataset_generation import load_qa_data
from factuality_eval.model_generation import generate_answers_from_qa_data
from factuality_eval.prompt_utils import PromptUtils
from factuality_eval.selfcheck_gpt import PromptVerdict, SelfCheckGPTEvaluator

load_dotenv()

logger = logging.getLogger(__name__)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


class _SampleDatasetList(list):
    """List variant whose ``append`` accepts ``generated_answers`` keyword."""

    def append(self, *, generated_answers) -> None:  # type: ignore[override]
        super().append(generated_answers)


def _extract_generated_answers(dataset) -> list[str]:
    """Return the generated answer column from a dataset-like object."""
    if dataset is None:
        return []

    if hasattr(dataset, "column_names") and "answer" in dataset.column_names:
        return list(dataset["answer"])

    if isinstance(dataset, list):
        return list(dataset)

    return list(dataset)


def _prepare_context_prompts(
    *, base_prompt: str, sample_answers: list[list[str]], example_index: int
) -> list[str]:
    """Construct prompts combining base context with sampled answers."""
    contexts: list[str] = []
    for sample_idx, answers in enumerate(sample_answers):
        if example_index >= len(answers):
            continue
        sample_answer = answers[example_index]
        if not sample_answer:
            continue
        contexts.append(
            f"{base_prompt}\n\nSample answer {sample_idx + 1}:\n{sample_answer}"
        )

    return contexts


def _sanitize_temperature(value: float | None) -> float | None:
    """Map non-positive temperatures to ``None`` for greedy decoding."""
    if value is None:
        return None

    if value <= 0:
        return None

    return value


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)

    reference_dataset_name = (
        f"{config.base_dataset.id}-{config.language}-"
        f"{config.models.eval_model.split('/')[1]}-reference"
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

    reference_answers = generate_answers_from_qa_data(
        eval_model=config.models.eval_model,
        contexts=contexts,
        questions=questions,
        answers=answers,
        lang=config.language,
        max_new_tokens=config.generation.max_new_tokens,
        output_jsonl_path=Path("data", "final", f"{reference_dataset_name}.jsonl"),
        temperature=_sanitize_temperature(config.selfcheckgpt.reference_temperature),
    )

    sample_dataset_name = (
        f"{config.base_dataset.id}-{config.language}-"
        f"{config.models.eval_model.split('/')[1]}-sample"
    )
    sample_datasets = []
    try:
        sample_datasets.__class__ = _SampleDatasetList
    except TypeError:
        sample_datasets = _SampleDatasetList(sample_datasets)
    for sample_idx in range(config.selfcheckgpt.num_samples):
        sample_datasets.append(
            generated_answers=generate_answers_from_qa_data(
                eval_model=config.models.eval_model,
                contexts=contexts,
                questions=questions,
                answers=answers,
                lang=config.language,
                max_new_tokens=config.generation.max_new_tokens,
                output_jsonl_path=Path(
                    "data", "final", f"{sample_dataset_name}-{sample_idx}.jsonl"
                ),
                temperature=_sanitize_temperature(
                    config.selfcheckgpt.sampling_temperature
                ),
            )
        )

    evaluator = SelfCheckGPTEvaluator(
        client=OpenAI(),
        model=getattr(config.selfcheckgpt, "prompt_model", "gpt-4o-mini"),
    )

    reference_dataset = reference_answers
    reference_generated_answers = _extract_generated_answers(reference_dataset)[
        : len(contexts)
    ]
    sample_generated_answers = [
        _extract_generated_answers(dataset)[: len(contexts)]
        for dataset in sample_datasets
    ]

    results = []
    mean_scores = []

    for idx, (context_passages, question, ground_truth_answer) in enumerate(
        zip(contexts, questions, answers)
    ):
        if idx >= len(reference_generated_answers):
            logger.warning("Missing reference answer for index %s; skipping", idx)
            continue

        reference_answer = reference_generated_answers[idx]
        context_prompt = PromptUtils.format_context(
            context_passages, question, lang=config.language
        )

        contexts_for_scoring = _prepare_context_prompts(
            base_prompt=context_prompt,
            sample_answers=sample_generated_answers,
            example_index=idx,
        )

        if not contexts_for_scoring:
            logger.warning("No sampled contexts available for index %s", idx)
            continue

        verdicts: list[PromptVerdict] = evaluator.score_answer_against_contexts(
            reference_answer, contexts_for_scoring
        )
        valid_scores = [verdict.score for verdict in verdicts]
        mean_inconsistency = (
            sum(valid_scores) / len(valid_scores) if valid_scores else None
        )
        if mean_inconsistency is not None:
            mean_scores.append(mean_inconsistency)

        results.append(
            {
                "index": idx,
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "reference_answer": reference_answer,
                "contexts": contexts_for_scoring,
                "verdicts": [
                    {
                        "sample_index": verdict.sample_index,
                        "label": verdict.label,
                        "score": verdict.score,
                        "raw_response": verdict.raw_response,
                    }
                    for verdict in verdicts
                ],
                "mean_inconsistency": mean_inconsistency,
            }
        )

    output_dir = Path(
        getattr(config.selfcheckgpt, "output_dir", "data/final/selfcheckgpt")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name_safe = _safe_model_name(config.models.eval_model)
    output_path = output_dir / (
        f"{config.base_dataset.id}-{config.language}-{model_name_safe}-selfcheckgpt_prompt.jsonl"
    )

    with output_path.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    if mean_scores:
        overall_mean = sum(mean_scores) / len(mean_scores)
        logger.info("Average SelfCheckGPT inconsistency score: %.4f", overall_mean)
    else:
        logger.info("No SelfCheckGPT scores computed.")

    logger.info("Saved detailed results to %s", output_path)


if __name__ == "__main__":
    main()
