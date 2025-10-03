"""Utilities for loading and formatting prompts. Adapted from LettuceDetect."""

from __future__ import annotations

import typing as t
from pathlib import Path
from string import Template

# Type for supported languages
Lang = t.Literal["en", "de", "fr", "es", "it", "pl", "cn", "da"]

LANG_TO_PASSAGE = {
    "da": "afsnit",
    "en": "passage",
    "de": "Passage",
    "fr": "passage",
    "es": "pasaje",
    "it": "brano",
    "pl": "fragment",
    "cn": "段落",
    "hu": "szövegrészlet",
}

LANG_TO_FULL_NAME = {
    "da": "Danish",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pl": "Polish",
    "cn": "Chinese",
    "hu": "Hungarian",
}

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


class PromptUtils:
    """Utility class for loading and formatting prompts."""

    @staticmethod
    def load_prompt(filename: str) -> Template:
        """Load a prompt template from the prompts directory.

        Args:
            filename:
                Name of the prompt file.

        Returns:
            Template object for the prompt.

        Raises:
            FileNotFoundError:
                If the prompt file doesn't exist.
        """
        path = PROMPT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return Template(path.read_text(encoding="utf-8"))

    @staticmethod
    def load_selfcheckgpt_prompt(lang: Lang) -> Template:
        """Load the SelfCheckGPT prompt template.

        Returns:
            Template object for the SelfCheckGPT prompt.
        """
        return PromptUtils.load_prompt(f"selfcheckgpt_prompt_{lang.lower()}.txt")

    @staticmethod
    def format_context(context: list[str], question: str | None, lang: Lang) -> str:
        """Format context and question into a prompt.

        Args:
            context:
                List of passages.
            question:
                The question, or None for summarization tasks.
            lang:
                The language code.

        Returns:
            Formatted prompt.
        """
        passage_word = LANG_TO_PASSAGE[lang]
        ctx_block = "\n".join(
            f"{passage_word} {i + 1}: {p}" for i, p in enumerate(context)
        )

        if question is None:
            tmpl = PromptUtils.load_prompt(f"summary_prompt_{lang.lower()}.txt")
            return tmpl.substitute(text=ctx_block)

        tmpl = PromptUtils.load_prompt(f"qa_prompt_{lang.lower()}.txt")
        return tmpl.substitute(
            question=question, num_passages=len(context), context=ctx_block
        )

    @staticmethod
    def get_full_language_name(lang: Lang) -> str:
        """Get the full language name for a language code.

        Args:
            lang: Language code.

        Returns:
            Full language name.
        """
        return LANG_TO_FULL_NAME.get(lang, "Unknown")
