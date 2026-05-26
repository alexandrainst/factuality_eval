# Hallucination Annotation Guidelines (Danish)

These guidelines apply when annotating Danish QA answers in the Streamlit
annotator (`uv run streamlit run src/scripts/ground_truth/annotate_ground_truth.py`).

## Definition

A **hallucination** is any part of the model's `ANSWER` that is **not
supported** by the supplied `CONTEXT` or that **contradicts** it. Hallucination
is defined *relative to the context*, not relative to world truth or the gold
QA answer.

## What to mark

Mark hallucinations as **verbatim substrings of the ANSWER**, copied exactly
(including diacritics, casing, and punctuation that is part of the span).
Multiple spans per answer are fine; overlapping spans should be merged.

Concretely, mark a span when the answer:

- Introduces a **named entity** (person, place, organisation, work) that the
  context does not mention.
- Asserts a **number, date, or quantity** that does not appear in the context
  (or contradicts one that does).
- Claims a **relation** (X is part of Y, X happened in year Z) that the
  context does not state or implies otherwise.
- Adds **qualifiers** ("the first", "the only", "always") not grounded in the
  context.

## What NOT to mark

- **Question-echoing prefixes.** If the answer restates the question (e.g.
  "Filmen foregår i …") and the restated material is content-free framing,
  do not mark it. Only mark the asserted answer content.
- **Bare punctuation or whitespace.** Never mark a span that is only commas,
  periods, quotes, or spaces.
- **Subword fragments alone.** If the model emits a hallucinated word as
  multiple subword tokens, mark the whole word — the per-token projection
  handles the split automatically.
- **Incomplete but supported answers.** If the answer omits information from
  the context but everything it does say is supported, mark nothing.
- **Stylistic awkwardness.** Ungrammatical or oddly phrased Danish is not a
  hallucination if the claim itself is supported by the context.

## Edge cases

- **Paraphrase of context.** Supported, do not mark — even if the wording
  differs.
- **Partial match.** If the answer says "Anne Marie Petersen døde i januar
  1951" and the context says "død 9. januar 1951", the answer is supported.
- **Contradiction.** If the context says "Arsenal FC" and the answer says
  "Strømsgodset", mark `Strømsgodset`.
- **Unsupported but plausible.** If the context is silent on a claim that is
  probably true (world knowledge), still mark it — hallucination is judged
  against the context only.

## Workflow tips

- The model and LLM-judge highlight overlays are **hidden by default** in the
  UI. Form your own verdict first; reveal the overlays only as a final
  sanity check.
- Use the **notes** field to record borderline calls or guideline
  clarifications — useful when re-reading later.
- The **Skip** button is for rows that are unannotatable (empty answer,
  truncated context, language mismatch). Add a one-line reason in notes.
