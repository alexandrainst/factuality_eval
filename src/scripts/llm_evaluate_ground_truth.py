"""Fill in the LLM verdict columns of the ground-truth evaluation dataset.

This is the API-calling pass that ``evaluate_ground_truth.py`` deliberately
skips. It reads ``data/final/ground_truth_evaluation_dataset.jsonl``, sends
each unevaluated row to an OpenAI model, and writes the verdict back into
``llm_hallucinated_parts`` / ``llm_explanation`` in place.

The pass is idempotent: rows with a non-empty ``llm_explanation`` are
skipped. The file is rewritten atomically every ``checkpoint_every`` calls
so an interrupted run can resume without losing progress.

Usage:
    uv run src/scripts/llm_evaluate_ground_truth.py <config_key>=<config_value> ...
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI
from tqdm import tqdm

from factuality_eval.ground_truth_eval import evaluate_answer_with_llm

load_dotenv()

logger = logging.getLogger("llm_evaluate_ground_truth")

DATASET_PATH = Path("data", "final", "ground_truth_evaluation_dataset.jsonl")
CHECKPOINT_EVERY = 25


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _atomic_write(rows: list[dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Run the LLM evaluation pass over the ground-truth dataset."""
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found — run evaluate_ground_truth.py first."
        )

    rows = _read_rows(DATASET_PATH)
    pending_indices = [
        i for i, row in enumerate(rows) if not row.get("llm_explanation")
    ]
    logger.info(f"Loaded {len(rows)} rows; {len(pending_indices)} need LLM evaluation")
    if not pending_indices:
        logger.info("Nothing to do.")
        return

    client = OpenAI()
    model_name = config.models.hallu_gen_model

    processed = 0
    for i in tqdm(pending_indices, desc=f"LLM eval ({model_name})"):
        row = rows[i]
        verdict = evaluate_answer_with_llm(
            client=client,
            model=model_name,
            context=row["context"],
            question=row["question"],
            answer=row["answer"],
        )
        row["llm_hallucinated_parts"] = verdict["hallucinated_parts"]
        row["llm_explanation"] = verdict["explanation"]
        if "raw_response" in verdict:
            row["llm_raw_response"] = verdict["raw_response"]

        processed += 1
        if processed % CHECKPOINT_EVERY == 0:
            _atomic_write(rows, DATASET_PATH)
            logger.debug(f"Checkpointed after {processed} rows")

    _atomic_write(rows, DATASET_PATH)
    logger.info(f"Wrote {processed} LLM verdicts back to {DATASET_PATH}")


if __name__ == "__main__":
    main()
