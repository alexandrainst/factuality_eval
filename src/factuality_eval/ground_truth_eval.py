"""Shared utilities for the human-in-the-loop ground-truth evaluation pipeline.

Used by:
- ``src/scripts/ground_truth/build_ground_truth_dataset.py`` — writes detector
  predictions.
- ``src/scripts/ground_truth/llm_judge_ground_truth.py`` — fills in LLM-judge spans.
- ``src/scripts/ground_truth/upgrade_ground_truth_schema.py`` — backfills new schema
  fields.
- ``src/scripts/ground_truth/annotate_ground_truth.py`` — Streamlit annotator.
- ``src/scripts/ground_truth/evaluate_against_human.py`` — metrics vs human gold.

The annotation target is ``data/final/ground_truth_evaluation_dataset.jsonl``.
Each row holds detector, LLM-judge, and human verdicts on the same answer.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


HUMAN_STATUS_UNANNOTATED = "unannotated"
HUMAN_STATUS_ANNOTATED = "annotated"
HUMAN_STATUS_SKIPPED = "skipped"


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts to a JSONL file atomically."""
    _atomic_write(rows, path)


def _atomic_write(rows: list[dict[str, Any]], path: Path) -> None:
    """Write ``rows`` to ``path`` atomically via ``.tmp`` + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def hallucinated_text_from_tokens(tokens: list[dict[str, Any]]) -> str:
    """Concatenate tokens flagged by the detector (``pred == 1``)."""
    return "".join(t["token"] for t in tokens if int(t.get("pred", 0)) == 1)


def build_char_mask(text: str, spans: list[str]) -> list[bool]:
    """Mark every character of ``text`` that lies inside any occurrence of any span.

    Each span is searched verbatim (all overlapping start positions). Empty
    spans and spans not found in ``text`` are silently ignored.
    """
    mask = [False] * len(text)
    for span in spans:
        if not span:
            continue
        idx = text.find(span)
        while idx != -1:
            for k in range(idx, min(idx + len(span), len(text))):
                mask[k] = True
            idx = text.find(span, idx + 1)
    return mask


def spans_to_token_labels(
    tokens: list[dict[str, Any]], spans: list[str], *, skip_meta: bool = True
) -> list[int]:
    """Project a list of verbatim hallucinated spans onto per-token 0/1 labels.

    Token character offsets are derived by concatenating ``token["token"]`` in
    order (matching how the detector's tokens map back to ``answer``). A token
    is labelled 1 iff any of its character positions fall inside a span match.

    Args:
        tokens: Per-token records ``{"token": str, "pred": int, "prob": float}``
            as written by ``build_ground_truth_dataset.py``.
        spans: Verbatim hallucinated substrings (same format as
            ``llm_hallucinated_parts`` / ``human_hallucinated_parts``).
        skip_meta: If True, the synthetic ``"<eos>"`` token always gets label 0.

    Returns:
        A list of 0/1 ints with the same length as ``tokens``.
    """
    text = "".join(t["token"] for t in tokens)
    mask = build_char_mask(text, spans)

    labels: list[int] = []
    pos = 0
    for t in tokens:
        tt = t["token"]
        start, end = pos, pos + len(tt)
        pos = end
        if skip_meta and tt == "<eos>":
            labels.append(0)
            continue
        labels.append(1 if any(mask[start:end]) else 0)
    return labels


def default_human_fields(num_tokens: int) -> dict[str, Any]:
    """Return the default human-annotation slot values for a new row."""
    return {
        "human_hallucinated_parts": [],
        "human_annotation_labels": [None] * num_tokens,
        "human_annotation_notes": "",
        "human_annotation_status": HUMAN_STATUS_UNANNOTATED,
        "human_annotated_at": None,
    }


def char_label_spans_to_strings(answer: str, labels: list[dict[str, Any]]) -> list[str]:
    """Convert character-offset labels to verbatim substrings of ``answer``.

    Each ``label`` is expected to provide integer ``start`` and ``end`` keys
    (half-open offsets into ``answer``), as produced by RAGTruth. Labels with
    out-of-range offsets are clipped; empty resulting spans are dropped.
    """
    spans: list[str] = []
    n = len(answer)
    for label in labels or []:
        try:
            start = int(label["start"])
            end = int(label["end"])
        except (KeyError, TypeError, ValueError):
            continue
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        if end <= start:
            continue
        spans.append(answer[start:end])
    return spans


def port_annotations(prev_row: dict[str, Any], new_row: dict[str, Any]) -> bool:
    """Carry forward human + LLM-judge fields from ``prev_row`` onto ``new_row``.

    ``new_row`` is the freshly built row (with new detector ``tokens``); we copy
    the human spans / notes / status / timestamp and the LLM-judge spans &
    explanation from ``prev_row``, then re-project ``human_annotation_labels``
    against the new token grid via :func:`spans_to_token_labels` so the labels
    stay aligned with the new tokenisation.

    Returns ``True`` if anything was ported.
    """
    if not prev_row:
        return False

    ported = False

    if not new_row.get("gold_answer") and prev_row.get("gold_answer"):
        new_row["gold_answer"] = prev_row["gold_answer"]
        ported = True

    if not new_row.get("source") and prev_row.get("source"):
        new_row["source"] = prev_row["source"]
        ported = True

    human_status = prev_row.get("human_annotation_status", HUMAN_STATUS_UNANNOTATED)
    if human_status != HUMAN_STATUS_UNANNOTATED:
        spans = list(prev_row.get("human_hallucinated_parts") or [])
        new_row["human_hallucinated_parts"] = spans
        new_row["human_annotation_labels"] = spans_to_token_labels(
            new_row.get("tokens", []), spans
        )
        new_row["human_annotation_notes"] = prev_row.get("human_annotation_notes", "")
        new_row["human_annotation_status"] = human_status
        new_row["human_annotated_at"] = prev_row.get("human_annotated_at")
        ported = True

    llm_spans = prev_row.get("llm_hallucinated_parts") or []
    llm_explanation = prev_row.get("llm_explanation", "")
    if llm_spans or llm_explanation:
        new_row["llm_hallucinated_parts"] = list(llm_spans)
        new_row["llm_explanation"] = llm_explanation
        ported = True

    return ported


def ensure_schema(row: dict[str, Any]) -> bool:
    """Backfill any missing schema fields on ``row``. Returns True if mutated."""
    mutated = False
    num_tokens = len(row.get("tokens", []))

    defaults = default_human_fields(num_tokens)
    for key, value in defaults.items():
        if key not in row:
            row[key] = value
            mutated = True

    # Keep human_annotation_labels in sync with token count for unannotated rows.
    labels = row.get("human_annotation_labels")
    if (
        row.get("human_annotation_status") == HUMAN_STATUS_UNANNOTATED
        and isinstance(labels, list)
        and len(labels) != num_tokens
    ):
        row["human_annotation_labels"] = [None] * num_tokens
        mutated = True

    if "llm_hallucinated_parts" not in row:
        row["llm_hallucinated_parts"] = []
        mutated = True
    if "llm_explanation" not in row:
        row["llm_explanation"] = ""
        mutated = True
    if "gold_answer" not in row:
        row["gold_answer"] = ""
        mutated = True
    if "source" not in row:
        row["source"] = "multiwikiqa"
        mutated = True

    return mutated
