"""Tests for the human-in-the-loop ground-truth evaluation utilities."""

from __future__ import annotations

from factuality_eval.ground_truth_eval import (
    HUMAN_STATUS_UNANNOTATED,
    build_char_mask,
    default_human_fields,
    ensure_schema,
    hallucinated_text_from_tokens,
    spans_to_token_labels,
)


def _make_tokens(words: list[tuple[str, int]]) -> list[dict]:
    """Build token records ``[{"token", "pred", "prob"}]`` from (text, pred) pairs."""
    return [{"token": t, "pred": p, "prob": 0.9 if p else 0.1} for t, p in words]


def test_build_char_mask_marks_only_matched_chars() -> None:
    """Spans found verbatim flip their characters; others stay False."""
    text = "Anne Marie døde i 1951."
    mask = build_char_mask(text, ["1951", "Marie", "missing"])
    # "Marie" at index 5..10, "1951" at index 18..22.
    assert sum(mask) == len("Marie") + len("1951")
    assert all(mask[5:10])
    assert all(mask[18:22])
    assert not mask[0]


def test_spans_to_token_labels_flips_overlapping_tokens() -> None:
    """Tokens whose chars overlap a span get label 1; others 0."""
    # Concatenated text: "Filmen foregår i 1951.<eos>"
    tokens = _make_tokens(
        [
            ("Filmen", 0),
            (" foregår", 0),
            (" i", 0),
            (" 1951", 1),
            (".", 0),
            ("<eos>", 0),
        ]
    )
    labels = spans_to_token_labels(tokens, ["1951"])
    assert labels == [0, 0, 0, 1, 0, 0]


def test_spans_to_token_labels_handles_subword_split() -> None:
    """A span covering multiple subword tokens flips them all."""
    tokens = _make_tokens([("Strøms", 0), ("godset", 0), (".", 0)])
    labels = spans_to_token_labels(tokens, ["Strømsgodset"])
    assert labels == [1, 1, 0]


def test_spans_to_token_labels_eos_always_zero() -> None:
    """The meta ``<eos>`` token is never marked even if the span string matches."""
    tokens = _make_tokens([("hello", 0), ("<eos>", 0)])
    labels = spans_to_token_labels(tokens, ["<eos>"])
    assert labels == [0, 0]


def test_hallucinated_text_from_tokens_concatenates_pred1() -> None:
    """Only tokens with ``pred == 1`` are concatenated, in order."""
    tokens = _make_tokens([("a", 1), ("b", 0), ("c", 1)])
    assert hallucinated_text_from_tokens(tokens) == "ac"


def test_ensure_schema_backfills_missing_fields() -> None:
    """A row missing human/llm fields gets them added; the call returns True."""
    row = {"tokens": _make_tokens([("a", 0), ("b", 0)])}
    mutated = ensure_schema(row)
    assert mutated is True
    assert row["human_annotation_status"] == HUMAN_STATUS_UNANNOTATED
    assert row["human_annotation_labels"] == [None, None]
    assert row["human_hallucinated_parts"] == []
    assert row["llm_hallucinated_parts"] == []
    assert row["llm_explanation"] == ""
    # Idempotent: a second call must not mutate.
    assert ensure_schema(row) is False


def test_default_human_fields_shape() -> None:
    """Default fields have the expected keys and lengths."""
    fields = default_human_fields(3)
    assert fields["human_annotation_labels"] == [None, None, None]
    assert fields["human_annotation_status"] == HUMAN_STATUS_UNANNOTATED
    assert fields["human_annotated_at"] is None
