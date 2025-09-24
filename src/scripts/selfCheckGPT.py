"""Run the SelfCheckGPT prompt-based evaluation pipeline.

Usage:
    uv run src/scripts/selfCheckGPT.py <config_key>=<config_value> ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from factuality_eval.dataset_generation import load_qa_data
from factuality_eval.model_generation import generate_single_answer, infer_model_device
from factuality_eval.prompt_utils import Lang, PromptUtils
from factuality_eval.selfcheck_gpt import PromptVerdict, SelfCheckGPTEvaluator


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _generate_model_outputs(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    contexts: Sequence[list[str]],
    questions: Sequence[str],
    *,
    lang: Lang,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    description: str,
) -> list[str]:
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    eos_token_id = tokenizer.eos_token_id
    generation_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": pad_token_id}
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if do_sample:
        generation_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generation_kwargs["do_sample"] = False
        if temperature is not None:
            generation_kwargs["temperature"] = temperature

    outputs: list[str] = []
    device = infer_model_device(model)

    for context, question in tqdm(
        zip(contexts, questions), total=len(questions), desc=description, unit="sample"
    ):
        answer_kwargs = dict(generation_kwargs)
        answer_kwargs.pop("max_new_tokens", None)
        generated_answer = generate_single_answer(
            tokenizer=tokenizer,
            model=model,
            context=context,
            question=question,
            lang=lang,
            max_new_tokens=max_new_tokens,
            device=device,
            enable_thinking=False,
            **answer_kwargs,
        )
        outputs.append(generated_answer)

    return outputs


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    load_dotenv()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    base_dataset_id = (
        f"{config.base_dataset.organisation}/{config.base_dataset.id}:{config.language}"
    )

    contexts, questions, answers = load_qa_data(
        base_dataset_id=base_dataset_id,
        split=config.base_dataset.split,
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
    )

    contexts = list(contexts)
    questions = list(questions)
    answers = list(answers)

    _, contexts, _, questions, _, answers = train_test_split(
        contexts, questions, answers, test_size=0.2, random_state=42, shuffle=False
    )

    max_examples = getattr(config.selfcheckgpt, "max_examples", None)
    if max_examples is not None:
        contexts = contexts[:max_examples]
        questions = questions[:max_examples]
        answers = answers[:max_examples]

    tokenizer = AutoTokenizer.from_pretrained(config.models.eval_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.models.eval_model, torch_dtype="auto", device_map="auto"
    )

    max_new_tokens = getattr(config.selfcheckgpt, "max_new_tokens", 1024)

    reference_outputs = _generate_model_outputs(
        tokenizer,
        model,
        contexts,
        questions,
        lang=config.language,
        max_new_tokens=max_new_tokens,
        do_sample=getattr(config.selfcheckgpt, "reference_do_sample", False),
        temperature=getattr(config.selfcheckgpt, "reference_temperature", 0.0),
        description="Generating reference answers",
    )

    sampled_outputs: list[list[str]] = []
    num_samples = getattr(config.selfcheckgpt, "num_samples", 20)
    for sample_idx in range(num_samples):
        sample_outputs = _generate_model_outputs(
            tokenizer,
            model,
            contexts,
            questions,
            lang=config.language,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=getattr(config.selfcheckgpt, "sampling_temperature", 1.0),
            description=f"Sampling answers {sample_idx + 1}/{num_samples}",
        )
        sampled_outputs.append(sample_outputs)

    evaluator = SelfCheckGPTEvaluator(
        client=OpenAI(),
        model=getattr(config.selfcheckgpt, "prompt_model", "gpt-4o-mini"),
        max_retries=getattr(config.selfcheckgpt, "max_retries", 3),
        request_timeout=getattr(config.selfcheckgpt, "request_timeout", None),
        context_char_limit=getattr(config.selfcheckgpt, "context_char_limit", None),
    )

    results = []
    mean_scores = []
    for idx, (context_passages, question, reference_answer, ground_truth) in enumerate(
        tqdm(
            zip(contexts, questions, reference_outputs, answers),
            total=len(questions),
            desc="Running SelfCheckGPT scoring",
            unit="answer",
        )
    ):
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
                "ground_truth_answer": ground_truth,
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
