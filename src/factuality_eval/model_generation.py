"""Shared utilities for generating model answers from QA-style prompts."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import Dataset
from openai import OpenAI
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from factuality_eval.dataset_generation import generate_hash
from factuality_eval.prompt_utils import Lang, PromptUtils

logger = logging.getLogger(__name__)


def generate_single_answer(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    context: Iterable[str],
    question: str | None,
    lang: Lang,
    max_new_tokens: int = 32768,
    temperature: float | None = None,
) -> str:
    """Generate a single answer from model for the given context and question.

    Args:
        tokenizer: The tokenizer paired with ``model``.
        model: A causal language model used for answer generation.
        context: The context to condition the generation on.
        question: The question to condition the generation on.
        lang: Language passed to the prompt formatter.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The temperature to use for generation. If None, the
            default temperature of the model is used.

    Returns:
        The generated answer.
    """
    prompt = PromptUtils.format_context(list(context), question, lang=lang)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Only include temperature in generation parameters if it's specified
    generation_kwargs: dict[str, int | float] = {"max_new_tokens": max_new_tokens}
    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    generated_ids = model.generate(**model_inputs, **generation_kwargs)
    output_ids = generated_ids[0].tolist()

    # parsing thinking content (from documentation of Qwen,
    # must be done even if thinking is disabled)
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content


def generate_single_answer_from_openai(
    client: OpenAI,
    eval_model: str,
    context: Iterable[str],
    question: str | None,
    lang: Lang,
    max_new_tokens: int = 32768,
    temperature: float | None = None,
) -> str:
    """Generate a single answer from OpenAI for the given context and question.

    Args:
        client: An OpenAI client used for answer generation.
        context: The context to condition the generation on.
        question: The question to condition the generation on.
        lang: Language passed to the prompt formatter.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The temperature to use for generation. If None, the
            default temperature of the model is used.

    Returns:
        The generated answer.
    """
    prompt = PromptUtils.format_context(list(context), question, lang=lang)
    messages = [{"role": "user", "content": prompt}]

    # Only include temperature in generation parameters if it's specified
    generation_kwargs: dict[str, int | float] = {"max_tokens": max_new_tokens}
    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    response = client.chat.completions.create(
        model=eval_model, messages=messages, **generation_kwargs
    )
    return response.choices[0].message.content.strip("\n")


def generate_answers_from_qa_data(
    eval_model: str,
    contexts: list[list[str]],
    questions: list[str],
    answers: list[str],
    output_jsonl_path: Path | None,
    max_new_tokens: int = 32768,
    temperature: float | None = None,
    lang: Lang = "da",
) -> Dataset:
    """Generate answers from a model for given QA data.

    Args:
        eval_model:
            The name of the model to use for generation.
        dataset:
            A Hugging Face ``Dataset`` containing ``context``, ``question`` and
            ``answer`` columns.
        output_jsonl_path:
            Optional path used to cache generations to disk.
        lang:
            Language passed to the prompt formatter.

    Returns:
        A Dataset containing both original and generated QA pairs.
    """
    logger.info("Generating answers from model to be evaluated...")

    records: list[dict] = list()

    if eval_model.startswith("openai/"):
        model = None
        tokenizer = None
    else:
        model, tokenizer = load_model_for_generation(eval_model)

    # Load the existing dataset if it exists
    if output_jsonl_path is not None and output_jsonl_path.exists():
        logger.info(f"Loading existing dataset from {output_jsonl_path}...")
        with output_jsonl_path.open() as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

    # Extract the list of hashes for quick lookups
    hashes = {record["hash"] for record in records}

    for context, question, _answer in zip(
        tqdm(contexts, desc="Generating answers"), questions, answers
    ):
        hash_ = generate_hash(context=context, question=question, answer=_answer)
        if hash_ in hashes:
            continue

        try:
            if tokenizer is None or model is None:
                answer = generate_single_answer_from_openai(
                    client=OpenAI(),
                    eval_model=eval_model.split("/")[1],
                    context=context,
                    question=question,
                    lang=lang,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            else:
                answer = generate_single_answer(
                    tokenizer=tokenizer,
                    model=model,
                    context=context,
                    question=question,
                    lang=lang,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
        except Exception as e:
            logger.error(f"Error during generation: {e}. Skipping...")
            continue

        record = dict(hash=hash_, context=context, question=question, answer=answer)
        records.append(record)
        hashes.add(hash_)
        if output_jsonl_path is not None:
            with output_jsonl_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    data_dict: dict[str, list] = defaultdict(list)
    for record in records:
        data_dict["context"].append(record["context"])
        data_dict["question"].append(record["question"])
        data_dict["answer"].append(record["answer"])

    generated_dataset = Dataset.from_dict(mapping=data_dict)

    return generated_dataset


def load_model_for_generation(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model for generation.

    Args:
        model_name:
            The name of the model to load.

    Returns:
        A tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    return model, tokenizer
