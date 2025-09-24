"""Shared utilities for generating model answers from QA-style prompts."""

from __future__ import annotations

from typing import Any, Iterable

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from factuality_eval.prompt_utils import Lang, PromptUtils


def infer_model_device(model: PreTrainedModel) -> Any:
    """Infer the device a model currently lives on."""
    if hasattr(model, "device"):
        return model.device

    first_param = next(model.parameters(), None)
    if first_param is not None:
        return first_param.device

    return "cpu"


def generate_single_answer(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    context: Iterable[str],
    question: str | None,
    *,
    lang: Lang,
    max_new_tokens: int = 32768,
    device: Any | None = None,
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
    target_device = device if device is not None else infer_model_device(model)
    model_inputs = model_inputs.to(target_device)

    generation_inputs = dict(generation_kwargs)
    generation_inputs.setdefault("max_new_tokens", max_new_tokens)

    generated_ids = model.generate(**model_inputs, **generation_inputs)
    prompt_length = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[0][prompt_length:].tolist()

    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()
