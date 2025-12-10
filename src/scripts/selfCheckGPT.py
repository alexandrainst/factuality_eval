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
from tqdm.auto import tqdm

from factuality_eval.dataset_generation import load_qa_data, generate_hash
from factuality_eval.model_generation import generate_answers_from_qa_data
from factuality_eval.prompt_utils import PromptUtils
from factuality_eval.selfcheck_gpt import PromptVerdict, SelfCheckGPTEvaluator

load_dotenv()

logger = logging.getLogger(__name__)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


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
        temperature=config.selfcheckgpt.reference_temperature,
    )

    sample_dataset_name = (
        f"{config.base_dataset.id}-{config.language}-"
        f"{config.models.eval_model.split('/')[1]}-sample"
    )
    sample_datasets = []
    for sample_idx in range(config.selfcheckgpt.num_samples):
        sample_datasets.append(
            generate_answers_from_qa_data(
                eval_model=config.models.eval_model,
                contexts=contexts,
                questions=questions,
                answers=answers,
                lang=config.language,
                max_new_tokens=config.generation.max_new_tokens,
                output_jsonl_path=Path(
                    "data", "final", f"{sample_dataset_name}-{sample_idx}.jsonl"
                ),
                temperature=config.selfcheckgpt.sampling_temperature
            )
        )

    evaluator = SelfCheckGPTEvaluator(
        client=OpenAI(),
        model="gpt-4o-mini",
        lang=config.language,
    )
    records: list[dict] = list()
    self_checkgpt_output_jsonl_path = Path(
        "data",
        "final",
        "selfcheckgpt",
        (
            f"{config.base_dataset.id}-{config.language}-"
            f"{config.models.eval_model.split('/')[1]}_ongoing_evaluation.jsonl"
        ),
    )
    # Load the existing dataset if it exists
    if self_checkgpt_output_jsonl_path is not None and self_checkgpt_output_jsonl_path.exists():
        logger.info(f"Loading existing selfcheckgpt evaluations from {self_checkgpt_output_jsonl_path}...")
        with self_checkgpt_output_jsonl_path.open() as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

    hashes = {record["hash"] for record in records}

    results = []
    sentence_scores = []

    for idx, reference in enumerate(tqdm(reference_answers, desc="Evaluating with SelfCheckGPT")):
        hash_ = generate_hash(context=reference["context"], question=reference["question"], answer=reference["answer"])
        if hash_ in hashes:
            # Read mean_selfcheckgpt_inconsistency from existing records
            existing_record = next((record for record in records if record["hash"] == hash_), None)
            if existing_record:
                sentence_scores.append(existing_record.get("mean_selfcheckgpt_inconsistency"))

            continue

        samples = [sample[idx] for sample in sample_datasets]

        verdicts: list[PromptVerdict] = evaluator.score_samples_against_reference(
            reference, samples
        )
        valid_scores = [verdict.score for verdict in verdicts]
        mean_inconsistency = (
            sum(valid_scores) / len(valid_scores) if valid_scores else None
        )

        sentence_scores.append(mean_inconsistency)
        record = dict(
            hash=hash_,
            context=reference["context"],
            question=reference["question"],
            answer=reference["answer"],
            mean_selfcheckgpt_inconsistency=mean_inconsistency,
        )
        records.append(record)
        hashes.add(hash_)

        if self_checkgpt_output_jsonl_path is not None:
            with self_checkgpt_output_jsonl_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    output_dir = Path(
        getattr(config.selfcheckgpt, "output_dir", "data/final/selfcheckgpt")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name_safe = _safe_model_name(config.models.eval_model)
    output_path = output_dir / (
        f"{config.base_dataset.id}-{config.language}-{model_name_safe}-selfcheckgpt_prompt.jsonl"
    )

    results = [
        {
            "Average_SelfCheckGPT_Inconsistency": sum(sentence_scores) / len(sentence_scores)
        }
    ]
    with output_path.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    logger.info("Average SelfCheckGPT inconsistency score: %.4f", sum(sentence_scores) / len(sentence_scores))

    logger.info("Saved detailed results to %s", output_path)


if __name__ == "__main__":
    main()
