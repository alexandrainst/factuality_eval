"""Shared utilities for generating model answers from QA-style prompts."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset
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
    *,
    lang: Lang,
    max_new_tokens: int = 32768,
    enable_thinking: bool = False,
    **generation_kwargs: Any,
) -> str:
    """Generate a single answer from ``model`` for the given ``context``/``question``."""
    prompt = PromptUtils.format_context(list(context), question, lang=lang)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = model_inputs.to(model.device)

    generation_inputs = dict(generation_kwargs)
    generation_inputs.setdefault("max_new_tokens", max_new_tokens)

    generated_ids = model.generate(**model_inputs, **generation_inputs)
    prompt_length = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[0][prompt_length:].tolist()

    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def generate_answers_from_qa_data(
    model,
    tokenizer,
    dataset: Dataset,
    output_jsonl_path: Path | None,
    max_examples: int | None = None,
    *,
    lang: Lang = "da",
    **kwargs,
) -> Dataset:
    """Generate answers from a model for given QA data.

    Args:
        model:
            A causal language model used for answer generation.
        tokenizer:
            The tokenizer paired with ``model``.
        dataset:
            A Hugging Face ``Dataset`` containing ``context``, ``question`` and
            ``answer`` columns.
        output_jsonl_path:
            Optional path used to cache generations to disk.
        lang:
            Language passed to the prompt formatter.
        **kwargs:
            Extra keyword arguments forwarded to ``generate_single_answer`` (e.g.
            ``do_sample`` or ``temperature``).

    Returns:
        A Dataset containing both original and generated QA pairs.
    """
    logger.info("Generating answers from model to be evaluated...")

    records: list[dict] = list()

    # Load the existing dataset if it exists
    if output_jsonl_path is not None and output_jsonl_path.exists():
        logger.info(f"Loading existing dataset from {output_jsonl_path}...")
        with output_jsonl_path.open() as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

    # Extract the list of hashes for quick lookups
    hashes = {record["hash"] for record in records}

    contexts = dataset["context"]
    questions = dataset["question"]
    answers = dataset["answers"]

    for context, question, answer in zip(
        tqdm(contexts, desc="Generating answers"), questions, answers
    ):
        hash_ = generate_hash(context=context, question=question, answer=answer)
        if hash_ in hashes:
            continue

        generation_params = dict(kwargs)
        max_new_tokens = generation_params.pop("max_new_tokens", 32768)

        try:
            generated_answer = generate_single_answer(
                tokenizer=tokenizer,
                model=model,
                context=context,
                question=question,
                lang=lang,
                max_new_tokens=max_new_tokens,
                enable_thinking=False,
                **generation_params,
            )
        except Exception as e:
            logger.error(f"Error during generation: {e}. Skipping...")
            continue

        record = dict(
            hash=hash_,
            context=context,
            question=question,
            answer=answer,
            generated_answer=generated_answer,
            **kwargs,
        )
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
        data_dict["generated_answer"].append(record["generated_answer"])

        if "temperature" in kwargs.keys():
            data_dict["temperature"].append(record["temperature"])

    if not records:
        empty_columns = {
            "context": [],
            "question": [],
            "answer": [],
            "generated_answer": [],
        }
        if "temperature" in kwargs.keys():
            empty_columns["temperature"] = []
        return Dataset.from_dict(empty_columns)

    generated_dataset = Dataset.from_dict(mapping=data_dict)

    return generated_dataset


def load_model_for_generation(model_name: str):
    """Load a model for generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    return model, tokenizer
