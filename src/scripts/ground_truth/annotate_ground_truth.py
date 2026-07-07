"""Streamlit UI for human annotation of hallucinations.

Run with::

    uv run streamlit run src/scripts/ground_truth/annotate_ground_truth.py

The app reads ``data/final/ground_truth_evaluation_dataset.jsonl``, lets a
single annotator mark verbatim hallucinated substrings on each answer, and
writes the result (spans, derived per-token labels, notes, status, timestamp)
back into the same file atomically.

Model and LLM-judge highlight overlays are hidden by default to reduce
anchoring bias — toggle them on in the sidebar when you want to compare.
"""

from __future__ import annotations

import datetime as dt
import html
import os
from pathlib import Path

import streamlit as st

from factuality_eval.ground_truth_eval import (
    HUMAN_STATUS_ANNOTATED,
    HUMAN_STATUS_SKIPPED,
    HUMAN_STATUS_UNANNOTATED,
    _atomic_write,
    build_char_mask,
    ensure_schema,
    hallucinated_text_from_tokens,
    read_rows,
    spans_to_token_labels,
)

DEFAULT_DATASET_PATH = Path(
    os.environ.get(
        "FACTUALITY_EVAL_DATASET",
        "data/final/ground_truth_evaluation_dataset.jsonl",
    )
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> list[dict]:
    """Load the dataset; ``mtime`` invalidates the cache when the file changes."""
    rows = read_rows(Path(path))
    for row in rows:
        ensure_schema(row)
    return rows


def _save(rows: list[dict], path: Path) -> None:
    _atomic_write(rows, path)
    # Invalidate the cached read so the next refresh sees the new file mtime.
    _load.clear()


def _filtered_indices(
    rows: list[dict], status_filter: str, language: str | None
) -> list[int]:
    out = []
    for i, row in enumerate(rows):
        if status_filter != "all" and row.get("human_annotation_status") != status_filter:
            continue
        if language and row.get("language", language) != language:
            # Most rows won't have a language field; assume they match.
            continue
        out.append(i)
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_answer(answer: str, spans: list[str], color: str) -> str:
    """Return HTML with ``spans`` highlighted in ``answer`` using ``color``."""
    mask = build_char_mask(answer, spans)
    out: list[str] = []
    inside = False
    buf: list[str] = []

    def _flush(escaped_inside: bool) -> None:
        if not buf:
            return
        text = html.escape("".join(buf))
        if escaped_inside:
            out.append(
                f'<mark style="background-color: {color}; padding: 0 2px;">{text}</mark>'
            )
        else:
            out.append(text)
        buf.clear()

    for i, ch in enumerate(answer):
        if mask[i] != inside:
            _flush(inside)
            inside = mask[i]
        buf.append(ch)
    _flush(inside)
    return (
        '<div style="font-family: serif; font-size: 1.05rem; '
        'line-height: 1.5; white-space: pre-wrap;">'
        + "".join(out)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Hallucination annotator", layout="wide", page_icon="📝"
    )

    dataset_path = Path(
        st.sidebar.text_input("Dataset path", str(DEFAULT_DATASET_PATH))
    )
    if not dataset_path.exists():
        st.error(f"Dataset not found: {dataset_path}")
        return

    rows = _load(str(dataset_path), dataset_path.stat().st_mtime)

    # ------------------------------------------------------------- Sidebar
    st.sidebar.header("Filter")
    status_filter = st.sidebar.selectbox(
        "Status",
        options=["unannotated", "annotated", "skipped", "all"],
        index=0,
    )

    indices = _filtered_indices(rows, status_filter, language=None)
    total = len(rows)
    n_annotated = sum(
        1 for r in rows if r.get("human_annotation_status") == HUMAN_STATUS_ANNOTATED
    )
    n_skipped = sum(
        1 for r in rows if r.get("human_annotation_status") == HUMAN_STATUS_SKIPPED
    )

    st.sidebar.markdown(
        f"**Progress**: {n_annotated} annotated · {n_skipped} skipped · {total} total"
    )
    st.sidebar.progress(min(n_annotated / max(total, 1), 1.0))

    if not indices:
        st.success("No rows match the current filter. 🎉")
        return

    # ------------------------------------------------------------- Navigator
    st.sidebar.header("Navigate")
    pos = st.session_state.get("pos", 0)
    pos = max(0, min(pos, len(indices) - 1))

    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button("← Prev", use_container_width=True):
        pos = max(0, pos - 1)
    if col_next.button("Next →", use_container_width=True):
        pos = min(len(indices) - 1, pos + 1)
    pos = st.sidebar.number_input(
        f"Position (1-{len(indices)})",
        min_value=1,
        max_value=len(indices),
        value=pos + 1,
        step=1,
    ) - 1
    st.session_state["pos"] = pos

    row_idx = indices[pos]
    row = rows[row_idx]

    st.sidebar.header("Overlays")
    show_model = st.sidebar.checkbox("Show model spans (yellow)", value=False)
    show_llm = st.sidebar.checkbox("Show LLM-judge spans (orange)", value=False)

    # ------------------------------------------------------------- Main
    st.markdown(
        f"### Row {row_idx + 1} / {total}  ·  `{row.get('hash', '')}`  ·  "
        f"source: **{row.get('source', 'multiwikiqa')}**  ·  "
        f"status: **{row.get('human_annotation_status')}**"
    )

    st.markdown("**Question**")
    st.write(row.get("question", ""))

    st.markdown("**QA gold answer**")
    st.write(row.get("gold_answer") or "(not present in this dataset)")

    with st.expander("Context", expanded=False):
        for i, ctx in enumerate(row.get("context", []) or []):
            st.markdown(f"*Passage {i + 1}*")
            st.write(ctx)

    answer = row.get("answer", "")

    st.markdown("**Answer (clean)**")
    st.markdown(_render_answer(answer, [], "transparent"), unsafe_allow_html=True)

    if show_model:
        st.markdown("**Answer with model spans**")
        model_spans = [hallucinated_text_from_tokens(row.get("tokens", []))]
        model_spans = [s for s in model_spans if s]
        st.markdown(
            _render_answer(answer, model_spans, "#fff59d"), unsafe_allow_html=True
        )

    if show_llm:
        st.markdown("**Answer with LLM-judge spans**")
        st.markdown(
            _render_answer(answer, row.get("llm_hallucinated_parts", []), "#ffb74d"),
            unsafe_allow_html=True,
        )
        with st.expander("LLM-judge explanation", expanded=False):
            st.write(row.get("llm_explanation", ""))

    # --------------------------------------------------------- Annotation
    st.divider()
    st.markdown("**Your annotation — hallucinated substrings (one per line)**")
    st.caption(
        "Paste verbatim substrings of the answer. Whitespace matters. "
        "Spans that don't occur in the answer will be ignored on save."
    )

    spans_text = st.text_area(
        "Hallucinated spans",
        value="\n".join(row.get("human_hallucinated_parts") or []),
        height=150,
        key=f"spans_{row_idx}",
    )
    notes = st.text_area(
        "Notes",
        value=row.get("human_annotation_notes", ""),
        height=80,
        key=f"notes_{row_idx}",
    )

    # Live preview of the spans the user typed in.
    typed_spans = [s for s in (spans_text.splitlines()) if s.strip()]
    valid_spans = [s for s in typed_spans if s in answer]
    invalid_spans = [s for s in typed_spans if s not in answer]
    if invalid_spans:
        st.warning(
            "These spans don't appear verbatim in the answer and will be dropped: "
            + ", ".join(repr(s) for s in invalid_spans)
        )
    if valid_spans:
        st.markdown("**Live preview (green)**")
        st.markdown(
            _render_answer(answer, valid_spans, "#a5d6a7"), unsafe_allow_html=True
        )

    col_save, col_skip, _ = st.columns([1, 1, 4])
    if col_save.button("💾 Save & next", type="primary"):
        row["human_hallucinated_parts"] = valid_spans
        row["human_annotation_labels"] = spans_to_token_labels(
            row.get("tokens", []), valid_spans
        )
        row["human_annotation_notes"] = notes
        row["human_annotation_status"] = HUMAN_STATUS_ANNOTATED
        row["human_annotated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _save(rows, dataset_path)
        st.session_state["pos"] = min(len(indices) - 1, pos + 1)
        st.rerun()

    if col_skip.button("⏭ Skip"):
        row["human_annotation_notes"] = notes or "(skipped)"
        row["human_annotation_status"] = HUMAN_STATUS_SKIPPED
        row["human_annotated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _save(rows, dataset_path)
        st.session_state["pos"] = min(len(indices) - 1, pos + 1)
        st.rerun()

    if row.get("human_annotation_status") != HUMAN_STATUS_UNANNOTATED:
        if st.button("↺ Reset this row to unannotated"):
            row["human_hallucinated_parts"] = []
            row["human_annotation_labels"] = [None] * len(row.get("tokens", []))
            row["human_annotation_notes"] = ""
            row["human_annotation_status"] = HUMAN_STATUS_UNANNOTATED
            row["human_annotated_at"] = None
            _save(rows, dataset_path)
            st.rerun()


if __name__ == "__main__":
    main()
