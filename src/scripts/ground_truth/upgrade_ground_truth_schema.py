"""Idempotently backfill the human-annotation schema on the ground-truth dataset.

Adds any missing schema fields to every row of
``data/final/ground_truth_evaluation_dataset.jsonl`` (and to the
``<name>.bak.jsonl`` backup, if present) and rewrites them atomically.

Also backfills ``gold_answer`` on rows where it's empty by reloading the source
QA data and matching by row hash.

Safe to re-run: a second invocation is a no-op.

Usage:
    uv run src/scripts/ground_truth/upgrade_ground_truth_schema.py [<key>=<value> ...]
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from factuality_eval.dataset_generation import load_qa_data
from factuality_eval.ground_truth_eval import ensure_schema, read_rows, write_rows

logger = logging.getLogger("upgrade_ground_truth_schema")


def _build_gold_lookup(config: DictConfig) -> dict[tuple[str, str], str]:
    """Reload the source QA data and map ``(context[0], question)`` to gold answer.

    We key on ``(context, question)`` rather than the row hash because the row
    hash also incorporates the gold answer text, which can drift across dataset
    revisions and break hash-based lookups for older rows.
    """
    contexts, questions, answers = load_qa_data(
        base_dataset_id=(
            f"{config.base_dataset.organisation}/{config.base_dataset.id}:"
            f"{config.language}"
        ),
        split="test",
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
        max_examples=config.generation.max_examples,
    )
    return {(ctx[0], q): a for ctx, q, a in zip(contexts, questions, answers)}


def _row_key(row: dict) -> tuple[str, str] | None:
    ctx = row.get("context") or []
    q = row.get("question")
    if not ctx or q is None:
        return None
    return (ctx[0], q)


def _backfill_gold(rows: list[dict], gold_lookup: dict[tuple[str, str], str]) -> int:
    """Fill ``gold_answer`` on rows where it's currently empty. Returns count filled."""
    filled = 0
    for row in rows:
        if row.get("gold_answer"):
            continue
        key = _row_key(row)
        gold = gold_lookup.get(key) if key else None
        if gold:
            row["gold_answer"] = gold
            filled += 1
    return filled


def _upgrade_file(path: Path, gold_lookup: dict[tuple[str, str], str]) -> None:
    if not path.exists():
        logger.info(f"{path} does not exist; skipping.")
        return
    rows = read_rows(path)
    schema_mutated = sum(1 for r in rows if ensure_schema(r))
    gold_filled = _backfill_gold(rows, gold_lookup)
    if schema_mutated or gold_filled:
        write_rows(rows, path)
        logger.info(
            f"Upgraded {schema_mutated} of {len(rows)} rows "
            f"and backfilled gold_answer on {gold_filled} rows in {path}"
        )
    else:
        logger.info(f"No changes needed in {path}")


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Run the schema upgrade on the main file and its backup, if any."""
    dataset_path = Path(config.manual_annotation.dataset_path)
    backup_path = dataset_path.with_suffix(".bak.jsonl")

    gold_lookup = _build_gold_lookup(config)

    _upgrade_file(dataset_path, gold_lookup)
    _upgrade_file(backup_path, gold_lookup)


if __name__ == "__main__":
    main()
