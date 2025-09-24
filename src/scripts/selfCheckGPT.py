"""Run the SelfCheckGPT prompt-based evaluation pipeline.

Usage:
    uv run src/scripts/selfCheckGPT.py <config_key>=<config_value> ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    dataset = load_dataset(
        path=f"{config.hub_organisation}/{config.base_dataset.id}",
        name=config.language,
        split="train",
    )
    test_dataset = dataset.train_test_split(test_size=0.2, seed=42)["test"]

    tokenizer = AutoTokenizer.from_pretrained(config.models.eval_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.models.eval_model, torch_dtype="auto", device_map="auto"
    )

    max_new_tokens = getattr(config.selfcheckgpt, "max_new_tokens", 32768)

    reference_dataset_name = f"{config.base_dataset.id}-{config.language}-{config.models.eval_model.split('/')[1]}"
    generate_answers_from_qa_data(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        lang=config.language,
        max_new_tokens=max_new_tokens,
        output_jsonl_path=Path("data", "final", f"{reference_dataset_name}.jsonl"),
        max_examples=config.selfcheckgpt.max_examples,
    )

    sample_dataset_name = f"{config.base_dataset.id}-{config.language}-{config.models.eval_model.split('/')[1]}-"
    sample_datasets = []
    for sample_idx in range(config.selfcheckgpt.num_samples):
        sample_datasets.append(
            generate_answers_from_qa_data(
                model=model,
                tokenizer=tokenizer,
                dataset=test_dataset,
                lang=config.language,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=config.selfcheckgpt.sampling_temperature,
                max_examples=config.selfcheckgpt.max_examples,
                output_jsonl_path=Path(
                    "data", "final", f"{sample_dataset_name}{sample_idx}.jsonl"
                ),
            )
        )

    evaluator = SelfCheckGPTEvaluator(
        client=OpenAI(),
        model=getattr(config.selfcheckgpt, "prompt_model", "gpt-4o-mini"),
        max_retries=getattr(config.selfcheckgpt, "max_retries", 3),
        request_timeout=getattr(config.selfcheckgpt, "request_timeout", None),
    )

    results = []
    mean_scores = []
    for idx, (example, reference_answer) in enumerate(
        tqdm(
            zip(test_dataset, reference_outputs),
            total=len(test_dataset),
            desc="Running SelfCheckGPT scoring",
            unit="answer",
        )
    ):
        raw_context = example.get("context")
        question = example.get("question")
        ground_truth_raw = example.get("answer")

        if raw_context is None:
            context_passages: list[str] = []
        elif isinstance(raw_context, str):
            context_passages = [raw_context]
        else:
            context_passages = list(raw_context)

        if isinstance(ground_truth_raw, dict) and "text" in ground_truth_raw:
            ground_truth_answer = ground_truth_raw["text"]
        elif isinstance(ground_truth_raw, list):
            ground_truth_answer = ground_truth_raw[0] if ground_truth_raw else ""
        else:
            ground_truth_answer = ground_truth_raw

        context_prompt = PromptUtils.format_context(
            context_passages, question, lang=config.language
        )
        contexts_for_scoring = [
            f"{context_prompt}\n\nSample answer {sample_idx + 1}:\n{samples[idx]}"
            for sample_idx, samples in enumerate(sampled_outputs)
        ]

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
