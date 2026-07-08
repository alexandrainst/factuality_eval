"""Tests for the `model_generation` module."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Iterator

import pytest


@pytest.fixture()
def model_generation_module(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[types.ModuleType]:
    """Import ``model_generation`` with lightweight dependency stubs."""
    dataset_module = types.ModuleType("datasets")

    class Dataset:
        """Stub dataset type."""

    dataset_module.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", dataset_module)

    openai_module = types.ModuleType("openai")

    class OpenAI:
        """Stub OpenAI client type."""

    openai_module.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    torch_module = types.ModuleType("torch")

    class Tensor:
        """Stub tensor type."""

    torch_module.Tensor = Tensor
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    tqdm_module = types.ModuleType("tqdm")
    tqdm_auto_module = types.ModuleType("tqdm.auto")
    tqdm_auto_module.tqdm = lambda iterable, **_: iterable
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)
    monkeypatch.setitem(sys.modules, "tqdm.auto", tqdm_auto_module)

    transformers_module = types.ModuleType("transformers")

    class AutoModelForCausalLM:
        """Stub model loader type."""

    class AutoTokenizer:
        """Stub tokenizer loader type."""

    class PreTrainedModel:
        """Stub pretrained model type."""

    class PreTrainedTokenizerBase:
        """Stub pretrained tokenizer type."""

    transformers_module.AutoModelForCausalLM = AutoModelForCausalLM
    transformers_module.AutoTokenizer = AutoTokenizer
    transformers_module.PreTrainedModel = PreTrainedModel
    transformers_module.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    dataset_generation_module = types.ModuleType("factuality_eval.dataset_generation")
    dataset_generation_module.generate_hash = lambda **_: "hash"
    monkeypatch.setitem(
        sys.modules, "factuality_eval.dataset_generation", dataset_generation_module
    )

    prompt_utils_module = types.ModuleType("factuality_eval.prompt_utils")
    prompt_utils_module.Lang = str

    class PromptUtils:
        """Stub prompt utility type."""

    prompt_utils_module.PromptUtils = PromptUtils
    monkeypatch.setitem(
        sys.modules, "factuality_eval.prompt_utils", prompt_utils_module
    )

    sys.modules.pop("factuality_eval.model_generation", None)
    yield importlib.import_module("factuality_eval.model_generation")
    sys.modules.pop("factuality_eval.model_generation", None)


class _FakeModelInputs(dict):
    def to(self, _device: str) -> "_FakeModelInputs":
        return self


class _FakeInputIds:
    shape = (1, 1)


class _FakeGeneratedRow:
    def __getitem__(self, _slice: slice) -> "_FakeGeneratedRow":
        return self

    def tolist(self) -> list[int]:
        return [1]


class _FakeGeneratedIds:
    def __getitem__(self, _index: int) -> _FakeGeneratedRow:
        return _FakeGeneratedRow()


class _FakeTokenizer:
    eos_token = None
    all_special_tokens: list[str] = []

    def __init__(self) -> None:
        self.apply_chat_template_calls: list[dict] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> str:
        self.apply_chat_template_calls.append({"messages": messages, **kwargs})
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        return "prompt"

    def __call__(self, texts: list[str], return_tensors: str) -> _FakeModelInputs:
        assert texts == ["prompt"]
        assert return_tensors == "pt"
        return _FakeModelInputs(input_ids=_FakeInputIds())

    def decode(self, output_ids: list[int], skip_special_tokens: bool) -> str:
        assert output_ids == [1]
        assert skip_special_tokens is False
        return "answer"


class _ExplodingTokenizer(_FakeTokenizer):
    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> str:
        self.apply_chat_template_calls.append({"messages": messages, **kwargs})
        raise TypeError("broken template")


class _FakeModel:
    device = "cpu"

    def generate(self, **_kwargs: object) -> _FakeGeneratedIds:
        return _FakeGeneratedIds()


def test_generate_single_answer_falls_back_when_enable_thinking_is_unsupported(
    model_generation_module: types.ModuleType,
) -> None:
    """Unsupported ``enable_thinking`` falls back to the plain template call."""
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    answer = model_generation_module.generate_single_answer_from_prompt(
        tokenizer=tokenizer,
        model=model,
        prompt="Question?",
    )

    assert answer == "answer"
    assert tokenizer.apply_chat_template_calls == [
        {
            "messages": [{"role": "user", "content": "Question?"}],
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
        {
            "messages": [{"role": "user", "content": "Question?"}],
            "tokenize": False,
            "add_generation_prompt": True,
        },
    ]


def test_generate_single_answer_reraises_unrelated_chat_template_type_errors(
    model_generation_module: types.ModuleType,
) -> None:
    """Non-``enable_thinking`` type errors are propagated unchanged."""
    tokenizer = _ExplodingTokenizer()
    model = _FakeModel()

    with pytest.raises(TypeError, match="broken template"):
        model_generation_module.generate_single_answer_from_prompt(
            tokenizer=tokenizer,
            model=model,
            prompt="Question?",
        )
