"""Shared utilities for generating model answers from QA-style prompts."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import cast

import torch
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

REASONING_MAX_TOKENS = 8192

# Pattern to strip markdown bold/italic markers from model output
_MD_MARKERS_RE = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")


def _strip_markdown(text: str) -> str:
    """Remove markdown bold/italic markers from text.

    Replaces ``**bold**``, ``*italic*``, ``***both***`` (and underscore
    equivalents) with just the inner text.
    """
    return _MD_MARKERS_RE.sub(r"\2", text)


def generate_single_answer_from_prompt(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    prompt: str,
    max_new_tokens: int = 32768,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> str:
    """Generate a single answer from model for a fully-formed prompt.

    The ``prompt`` is fed to the model verbatim (only wrapping it in the chat
    template). For QA data, build the prompt with
    :meth:`PromptUtils.format_context` first; for pre-formatted RAGTruth prompts
    (as EuroEval uses), pass the prompt through unchanged.

    Args:
        tokenizer: The tokenizer paired with ``model``.
        model: A causal language model used for answer generation.
        prompt: The fully-formed prompt to condition the generation on.
        max_new_tokens (optional): The maximum number of new tokens to generate.
            Defaults to 32768.
        temperature (float, optional): The temperature to use for generation.
            Defaults to None (use the model's default temperature). When
            ``temperature`` is None or <= 0, decoding is greedy and ``top_p`` /
            ``top_k`` are ignored.
        top_p (float, optional): The nucleus sampling probability. Only applied
            when sampling. Defaults to None (use the model's default).
        top_k (int, optional): The top-k sampling cutoff. Only applied when
            sampling. Defaults to None (use the model's default).

    Returns:
        The generated answer.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError as e:
        if "enable_thinking" not in str(e):
            raise
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_length = model_inputs["input_ids"].shape[-1]
    if input_length > REASONING_MAX_TOKENS:
        return ""

    # Only include temperature when it is strictly positive; otherwise use greedy
    generation_kwargs: dict[str, int | float | bool] = {
        "max_new_tokens": max_new_tokens
    }
    if temperature is None or temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["do_sample"] = True
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if top_k is not None and top_k > 0:
            generation_kwargs["top_k"] = top_k

    generated_ids = model.generate(  # type: ignore[operator]
        **model_inputs, **generation_kwargs
    )

    # Only decode newly generated tokens, excluding the input prompt
    output_ids = cast(torch.Tensor, generated_ids)[0][input_length:].tolist()

    # Decode keeping special tokens so the end-of-reasoning token can be detected,
    # mirroring EuroEval's handling.
    content = tokenizer.decode(output_ids, skip_special_tokens=False)

    # Match EuroEval's reasoning handling: keep only the text after the
    # end-of-reasoning token. If the model started reasoning but never emitted the
    # closing token (i.e. it exhausted its token budget mid-reasoning), treat the
    # sample as empty so it is dropped rather than scoring raw chain-of-thought.
    if "</think>" in content:
        content = content.split("</think>")[-1]
    elif "<think>" in content:
        content = ""
    eos_token = tokenizer.eos_token
    if eos_token:
        content = content.replace(eos_token, "")
    for special_token in tokenizer.all_special_tokens:
        content = content.replace(special_token, "")
    content = content.strip()

    return content


def generate_single_answer_from_openai_prompt(
    client: OpenAI,
    eval_model: str,
    prompt: str,
    max_new_tokens: int = 32768,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Generate a single answer from OpenAI for a fully-formed prompt.

    Args:
        client: An OpenAI client used for answer generation.
        eval_model: The name of the OpenAI model to use.
        prompt: The fully-formed prompt to condition the generation on.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The temperature to use for generation. If None, the
            default temperature of the model is used.
        top_p: The nucleus sampling probability. If None, the model's default is
            used. (The OpenAI API does not expose a ``top_k`` parameter.)

    Returns:
        The generated answer.
    """
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

    create_kwargs: dict = dict(
        model=eval_model,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature if temperature is not None else 1.0,
    )
    if top_p is not None:
        create_kwargs["top_p"] = top_p

    response = client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]
    content = response.choices[0].message.content
    return content.strip("\n") if content else ""


def _generate_answers(
    eval_model: str,
    prompts: list[str],
    hashes: list[str],
    extra_fields: list[dict],
    output_jsonl_path: Path | None,
    max_new_tokens: int = 32768,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Generate answers for pre-built prompts, with on-disk caching.

    Shared core used by :func:`generate_answers_from_qa_data` and
    :func:`generate_answers_from_prompts`. Each prompt is fed to the model
    verbatim (wrapped only in the chat template). Previously generated records
    are loaded from ``output_jsonl_path`` and skipped via their ``hash``.

    Args:
        eval_model: The name of the model to use for generation. If the name
            starts with ``"openai/"``, the OpenAI API is used; otherwise, a local
            Hugging Face model is loaded.
        prompts: The fully-formed prompts to condition the generation on.
        hashes: A unique hash per prompt, used to deduplicate against the cache.
        extra_fields: Additional fields to store on each generated record,
            aligned with ``prompts`` (e.g. ``{"context": ..., "question": ...}``
            or ``{"prompt": ...}``).
        output_jsonl_path: Path to a JSONL file used to cache generations. If the
            file exists, previously generated samples are loaded and reused.
        max_new_tokens: The maximum number of new tokens to generate for each
            answer. Defaults to 32768.
        temperature: The temperature to use for generation. Defaults to None
            (use the model's default temperature).
        top_p: The nucleus sampling probability. Only applied when sampling.
            Defaults to None (use the model's default).
        top_k: The top-k sampling cutoff. Only applied when sampling. Defaults
            to None (use the model's default).

    Returns:
        A list of generated records, each containing ``hash``, the keys from the
        corresponding ``extra_fields`` entry, and ``answer``.
    """
    logger.info("Generating answers from model to be evaluated...")

    records: list[dict] = list()

    is_openai_model = eval_model.startswith("openai/")
    model = None
    tokenizer = None

    # Load the existing dataset if it exists
    if output_jsonl_path is not None and output_jsonl_path.exists():
        logger.info(f"Loading existing dataset from {output_jsonl_path}...")
        with output_jsonl_path.open() as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

    # Extract the list of hashes for quick lookups
    seen_hashes = {record["hash"] for record in records}

    for prompt, hash_, extra in zip(
        tqdm(prompts, desc="Generating answers"), hashes, extra_fields
    ):
        if hash_ in seen_hashes:
            continue

        try:
            if is_openai_model:
                answer = generate_single_answer_from_openai_prompt(
                    client=OpenAI(),
                    eval_model=eval_model.split("/")[1],
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                if tokenizer is None or model is None:
                    model, tokenizer = load_model_for_generation(eval_model)
                answer = generate_single_answer_from_prompt(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

        except Exception as e:
            logger.error(f"Error during generation: {e}. Skipping...")
            continue
        if len(answer) == 0:
            continue
        # Strip markdown bold/italic markers that models sometimes add
        answer = _strip_markdown(answer)

        record = dict(hash=hash_, **extra, answer=answer)
        records.append(record)
        seen_hashes.add(hash_)
        if output_jsonl_path is not None:
            with output_jsonl_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    return records


def generate_answers_from_qa_data(
    eval_model: str,
    contexts: list[list[str]],
    questions: list[str],
    answers: list[str],
    output_jsonl_path: Path | None,
    max_new_tokens: int = 32768,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    lang: Lang = "da",
) -> Dataset:
    """Generate answers from a model for given QA data.

    Each ``(context, question)`` pair is formatted into a prompt via
    :meth:`PromptUtils.format_context` before generation.

    Args:
        eval_model: The name of the model to use for generation. If the name
            starts with ``"openai/"``, the OpenAI API is used; otherwise, a local
            Hugging Face model is loaded.
        contexts: A list of contexts, where each context is a
            list of strings to condition the generation on.
        questions: A list of questions corresponding to each context.
        answers: The original reference answers, used only to compute
            cache hashes and avoid regenerating duplicates.
        output_jsonl_path: Path to a JSONL file used to cache
            generations. If the file exists, previously generated samples are
            loaded and reused. Defaults to None.
        max_new_tokens: The maximum number of new tokens to
            generate for each answer. Defaults to 32768.
        temperature: The temperature to use for generation.
            Defaults to None (use the model's default temperature).
        top_p: The nucleus sampling probability. Only applied when sampling.
            Defaults to None (use the model's default).
        top_k: The top-k sampling cutoff. Only applied when sampling. Defaults
            to None (use the model's default).
        lang: Language passed to the prompt formatter.
            Defaults to ``"da"``.

    Returns:
        A Hugging Face ``Dataset`` containing the generated QA pairs
        with columns ``"context"``, ``"question"``, and ``"answer"``.
    """
    prompts = [
        PromptUtils.format_context(list(context), question, lang=lang)
        for context, question in zip(contexts, questions)
    ]
    hashes = [
        generate_hash(context=context, question=question, answer=answer)
        for context, question, answer in zip(contexts, questions, answers)
    ]
    extra_fields = [
        dict(context=context, question=question)
        for context, question in zip(contexts, questions)
    ]

    records = _generate_answers(
        eval_model=eval_model,
        prompts=prompts,
        hashes=hashes,
        extra_fields=extra_fields,
        output_jsonl_path=output_jsonl_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    data_dict: dict[str, list] = defaultdict(list)
    for record in records:
        data_dict["context"].append(record["context"])
        data_dict["question"].append(record["question"])
        data_dict["answer"].append(record["answer"])

    return Dataset.from_dict(mapping=data_dict)


def generate_answers_from_prompts(
    eval_model: str,
    prompts: list[str],
    answers: list[str],
    output_jsonl_path: Path | None,
    max_new_tokens: int = 32768,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Dataset:
    """Generate answers from a model for fully-formed prompts.

    Unlike :func:`generate_answers_from_qa_data`, each prompt is fed to the model
    verbatim without re-wrapping it through :meth:`PromptUtils.format_context`.
    This mirrors how EuroEval evaluates pre-formatted RAGTruth prompts, so the
    resulting hallucination rate is directly comparable.

    Args:
        eval_model: The name of the model to use for generation. If the name
            starts with ``"openai/"``, the OpenAI API is used; otherwise, a local
            Hugging Face model is loaded.
        prompts: A list of fully-formed prompts to condition the generation on.
        answers: The original reference answers, used only to compute cache
            hashes and avoid regenerating duplicates.
        output_jsonl_path: Path to a JSONL file used to cache generations. If the
            file exists, previously generated samples are loaded and reused.
        max_new_tokens: The maximum number of new tokens to generate for each
            answer. Defaults to 32768.
        temperature: The temperature to use for generation. Defaults to None
            (use the model's default temperature).
        top_p: The nucleus sampling probability. Only applied when sampling.
            Defaults to None (use the model's default).
        top_k: The top-k sampling cutoff. Only applied when sampling. Defaults
            to None (use the model's default).

    Returns:
        A Hugging Face ``Dataset`` with columns ``"prompt"`` and ``"answer"``.
    """
    hashes = [
        generate_hash(context=[prompt], question="", answer=answer)
        for prompt, answer in zip(prompts, answers)
    ]
    extra_fields = [dict(prompt=prompt) for prompt in prompts]

    records = _generate_answers(
        eval_model=eval_model,
        prompts=prompts,
        hashes=hashes,
        extra_fields=extra_fields,
        output_jsonl_path=output_jsonl_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    data_dict: dict[str, list] = defaultdict(list)
    for record in records:
        data_dict["prompt"].append(record["prompt"])
        data_dict["answer"].append(record["answer"])

    return Dataset.from_dict(mapping=data_dict)


def _build_max_memory() -> dict[int, str] | None:
    if not torch.cuda.is_available():
        return None
    max_memory: dict[int, str] = {}
    for device_index in range(torch.cuda.device_count()):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
        free_gib = int((free_bytes / (1024**3)) * 0.95)
        if free_gib <= 0:
            continue
        max_memory[device_index] = f"{free_gib}GiB"
    return max_memory or None


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
    max_memory = _build_max_memory()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", max_memory=max_memory
    )

    return model, tokenizer
