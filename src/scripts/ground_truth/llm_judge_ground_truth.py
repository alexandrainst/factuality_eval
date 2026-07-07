"""Fill in the LLM-judge spans on the ground-truth evaluation dataset.

For every row of ``data/final/ground_truth_evaluation_dataset.jsonl`` that has
an empty ``llm_explanation``, this calls an OpenAI model to mark hallucinated
spans in the answer (verbatim substrings) and writes the verdict back. The
write is atomic and checkpointed every 25 rows so an interrupted run resumes
without losing progress.

When ``manual_annotation.port_from_backup=true`` (default), verdicts are first
copied from ``<dataset>.bak.jsonl`` whenever the ``(hash, answer)`` pair matches
— this avoids paying for the same judgements twice after a detector rebuild.

Usage:
    uv run src/scripts/ground_truth/llm_judge_ground_truth.py [<key>=<value> ...]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI
from tqdm.auto import tqdm

from factuality_eval.ground_truth_eval import _atomic_write, read_rows

load_dotenv()

logger = logging.getLogger("llm_judge_ground_truth")

CHECKPOINT_EVERY = 25

_SYSTEM_PROMPT = (
    "You are a careful annotator that detects hallucinations in question "
    "answering. Given a CONTEXT, a QUESTION, and an ANSWER, identify any "
    "spans of the ANSWER that are not supported by the CONTEXT or that "
    "contradict it. Respond with a single JSON object with two keys: "
    '"hallucinated_parts" (a list of exact substrings copied verbatim from '
    'the ANSWER) and "explanation" (a short string explaining your verdict). '
    "If nothing is hallucinated, return an empty list and an explanation."
)


def _judge_one(
    client: OpenAI,
    model: str,
    context: list[str] | str,
    question: str,
    answer: str,
) -> dict[str, Any]:
    """Call the LLM judge for a single row and parse the JSON response."""
    context_str = "\n".join(context) if isinstance(context, list) else context
    user_message = (
        f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("LLM judge returned non-JSON content; storing raw text.")
        return {"hallucinated_parts": [], "explanation": "", "raw_response": content}
    return {
        "hallucinated_parts": list(parsed.get("hallucinated_parts", [])),
        "explanation": str(parsed.get("explanation", "")),
    }


def _port_from_backup(rows: list[dict[str, Any]], backup_path: Path) -> int:
    """Copy ``llm_*`` fields from the backup wherever ``(hash, answer)`` matches.

    Returns the number of rows updated.
    """
    if not backup_path.exists():
        logger.info(f"No backup at {backup_path}; skipping port step.")
        return 0
    backup = {
        (r.get("hash"), r.get("answer")): r for r in read_rows(backup_path)
    }
    ported = 0
    for row in rows:
        key = (row.get("hash"), row.get("answer"))
        prior = backup.get(key)
        if prior is None:
            continue
        if not row.get("llm_explanation") and prior.get("llm_explanation"):
            row["llm_hallucinated_parts"] = list(prior.get("llm_hallucinated_parts", []))
            row["llm_explanation"] = prior.get("llm_explanation", "")
            ported += 1
    logger.info(f"Ported LLM verdicts from backup for {ported} rows.")
    return ported


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Run the LLM-judge pass over the ground-truth evaluation dataset."""
    logging.getLogger("httpx").setLevel(logging.WARNING)

    dataset_path = Path(config.manual_annotation.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} not found — run build_ground_truth_dataset.py first."
        )

    rows = read_rows(dataset_path)

    # Optional: recycle verdicts from the pre-rebuild backup before calling the API.
    if config.manual_annotation.port_from_backup:
        backup_path = dataset_path.with_suffix(".bak.jsonl")
        ported = _port_from_backup(rows, backup_path)
        if ported:
            _atomic_write(rows, dataset_path)

    pending = [i for i, r in enumerate(rows) if not r.get("llm_explanation")]
    logger.info(f"{len(pending)} of {len(rows)} rows still need LLM judgement.")
    if not pending:
        logger.info("Nothing to do.")
        return

    client = OpenAI()
    model = config.models.hallu_gen_model

    written_since_checkpoint = 0
    for idx in tqdm(pending, desc="LLM judge"):
        row = rows[idx]
        try:
            verdict = _judge_one(
                client=client,
                model=model,
                context=row["context"],
                question=row["question"],
                answer=row["answer"],
            )
        except Exception as exc:
            logger.warning(f"Row {idx} ({row.get('hash')}): judge call failed: {exc}")
            continue

        row["llm_hallucinated_parts"] = verdict.get("hallucinated_parts", [])
        row["llm_explanation"] = verdict.get("explanation", "")
        if "raw_response" in verdict:
            row["llm_raw_response"] = verdict["raw_response"]

        written_since_checkpoint += 1
        if written_since_checkpoint >= CHECKPOINT_EVERY:
            _atomic_write(rows, dataset_path)
            written_since_checkpoint = 0

    if written_since_checkpoint:
        _atomic_write(rows, dataset_path)
    logger.info("LLM-judge pass complete.")


if __name__ == "__main__":
    main()
