"""Utilities for loading and formatting prompts. Adapted from LettuceDetect."""

from __future__ import annotations

import typing as t
from pathlib import Path
from string import Template

# Type for supported languages
Lang = t.Literal[
    "bs",
    "bg",
    "ca",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fo",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "is",
    "it",
    "lv",
    "lt",
    "no",
    "pl",
    "pt",
    "ro",
    "sr",
    "sk",
    "sl",
    "es",
    "sv",
    "uk",
]

LANG_TO_FULL_NAME = {
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fo": "Faroese",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "is": "Icelandic",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "uk": "Ukrainian",
}

YES_WORDS = {
    "bs": "da",
    "bg": "да",
    "ca": "sí",
    "hr": "da",
    "cs": "ano",
    "da": "ja",
    "nl": "ja",
    "en": "yes",
    "et": "jah",
    "fo": "ja",
    "fi": "kyllä",
    "fr": "oui",
    "de": "ja",
    "el": "ναι",
    "hu": "igen",
    "is": "já",
    "it": "sì",
    "lv": "jā",
    "lt": "taip",
    "no": "ja",
    "pl": "tak",
    "pt": "sim",
    "ro": "da",
    "sr": "да",
    "sk": "áno",
    "sl": "da",
    "es": "sí",
    "sv": "ja",
    "uk": "так",
}

NO_WORDS = {
    "bs": "ne",
    "bg": "не",
    "ca": "no",
    "hr": "ne",
    "cs": "ne",
    "da": "nej",
    "nl": "nee",
    "en": "no",
    "et": "ei",
    "fo": "nei",
    "fi": "ei",
    "fr": "non",
    "de": "nein",
    "el": "όχι",
    "hu": "nem",
    "is": "nei",
    "it": "no",
    "lv": "nē",
    "lt": "ne",
    "no": "nei",
    "pl": "nie",
    "pt": "não",
    "ro": "nu",
    "sr": "не",
    "sk": "nie",
    "sl": "ne",
    "es": "no",
    "sv": "nej",
    "uk": "ні",
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
    def load_selfcheckgpt_prompt(context: str, sentence: str, lang: Lang) -> str:
        """Load the SelfCheckGPT prompt template.

        Returns:
            Template object for the SelfCheckGPT prompt.
        """
        tmpl = PromptUtils.load_prompt(f"selfcheckgpt_prompt_{lang.lower()}.txt")
        return tmpl.substitute(context=context, sentence=sentence)

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
        ctx_block = "\n".join(context)

        if question is None:
            tmpl = PromptUtils.load_prompt(f"summary_prompt_{lang.lower()}.txt")
            return tmpl.substitute(text=ctx_block)

        tmpl = PromptUtils.load_prompt(f"qa_prompt_{lang.lower()}.txt")
        return tmpl.substitute(question=question, text=ctx_block)

    @staticmethod
    def get_full_language_name(lang: Lang) -> str:
        """Get the full language name for a language code.

        Args:
            lang: Language code.

        Returns:
            Full language name.
        """
        return LANG_TO_FULL_NAME.get(lang, "Unknown")
