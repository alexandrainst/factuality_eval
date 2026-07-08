"""Build the ground-truth evaluation dataset by running the current detector.

For each cached QA answer this writes one JSONL row containing:

- ``hash``, ``context``, ``question``, ``answer``
- ``tokens``: per-token records ``{"token", "pred", "prob"}`` from the detector
- ``model_predicted_hallucinated_text``: concatenation of tokens with ``pred == 1``
- ``llm_hallucinated_parts``, ``llm_explanation``: empty placeholders to be filled
  by ``llm_judge_ground_truth.py``
- Human-annotation slots (see ``ground_truth_eval.default_human_fields``)

Existing file at the output path is backed up to ``<name>.bak.jsonl`` before being
overwritten (the prior LLM-judge verdicts can be recycled by re-running
``llm_judge_ground_truth.py``, which ports from the backup by default via
``manual_annotation.port_from_backup=true``).

Usage:
    uv run src/scripts/ground_truth/build_ground_truth_dataset.py [<key>=<value> ...]
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from lettucedetect.models.inference import HallucinationDetector
from omegaconf import DictConfig
from tqdm.auto import tqdm

from factuality_eval.dataset_generation import generate_hash, load_qa_data
from factuality_eval.ground_truth_eval import (
    HUMAN_STATUS_ANNOTATED,
    char_label_spans_to_strings,
    default_human_fields,
    hallucinated_text_from_tokens,
    port_annotations,
    read_rows,
    spans_to_token_labels,
    write_rows,
)
from factuality_eval.model_generation import generate_answers_from_qa_data
from factuality_eval.train import format_dataset_to_ragtruth_without_labels

load_dotenv()

logger = logging.getLogger("build_ground_truth_dataset")


def _resolve_detector_path(config: DictConfig) -> str:
    """Resolve the manual-annotation detector path.

    Prefers an explicit ``manual_annotation.detector_model_dir`` override, then a
    local checkpoint matching the current training-data configuration
    (mwqa-only / with-ragtruth / only-ragtruth), and finally the matching HF Hub
    repository.
    """
    override = (
        config.manual_annotation.detector_model_dir
        if "manual_annotation" in config and config.manual_annotation.detector_model_dir
        else None
    )
    if override:
        logger.info(f"Using detector override: {override}")
        return str(override)

    base_name = (
        f"{config.models.hallu_detect_model}-"
        f"{config.base_dataset.id}-synthetic-hallucinations-{config.language}"
    )

    ragtruth_enabled = bool(config.get("ragtruth", {}).get("enable", False))
    multiwikiqa_enabled = bool(config.get("multiwikiqa", {}).get("enable", False))
    if ragtruth_enabled and multiwikiqa_enabled:
        suffix = "-with-ragtruth"
    elif ragtruth_enabled and not multiwikiqa_enabled:
        suffix = "-only-ragtruth"
    else:
        suffix = ""

    preferred = f"{base_name}{suffix}"
    fallbacks = [preferred]
    if suffix:
        # Backward-compatible fallback if only a mwqa-only checkpoint exists.
        fallbacks.append(base_name)

    for model_name in fallbacks:
        local_path = f"{config.training.output_dir}/{model_name}"
        if os.path.isdir(local_path):
            logger.info(f"Using local detector checkpoint at {local_path}")
            return local_path

    hub_path = f"{config.hub_organisation}/{preferred}"
    logger.info(f"Local checkpoint not found; using Hub model {hub_path}")
    return hub_path


def _backup_existing(output_path: Path) -> None:
    """Copy an existing output file to ``<name>.bak.jsonl`` for verdict recycling."""
    if not output_path.exists():
        return
    backup = output_path.with_suffix(".bak.jsonl")
    shutil.copy2(output_path, backup)
    logger.info(f"Backed up existing dataset to {backup}")


def _run_detector(
    detector: HallucinationDetector, prompt: str, answer: str
) -> list[dict[str, Any]]:
    """Run the detector and return the per-token predictions."""
    return list(detector.predict_prompt(prompt=prompt, answer=answer))


def _load_ragtruth_samples(
    path: Path, language: str, splits: list[str] | None, task_types: list[str] | None
) -> list[dict[str, Any]]:
    """Load and filter RAGTruth samples produced by ``preprocess_ragtruth.py``."""
    if not path.exists():
        raise FileNotFoundError(f"RAGTruth dataset not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        # The exported HallucinationData uses ``{"samples": [...]}``.
        samples = samples.get("samples", [])

    def _keep(sample: dict[str, Any]) -> bool:
        if language and sample.get("language") not in (None, language):
            return False
        if splits and sample.get("split") not in splits:
            return False
        if task_types and sample.get("task_type") not in task_types:
            return False
        return True

    return [s for s in samples if _keep(s)]


def _build_ragtruth_rows(
    detector: HallucinationDetector,
    config: DictConfig,
    prev_by_hash: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Build ground-truth rows from RAGTruth samples.

    Each row uses the RAGTruth ``prompt`` (which already embeds context and
    instruction/question) as a single-element ``context`` list, with an empty
    ``question``. The RAGTruth char-offset ``labels`` are converted to verbatim
    substrings of the answer and stored as pre-populated human annotations.
    """
    cfg = config.manual_annotation.ragtruth
    path = Path(cfg.path)
    splits = list(cfg.splits) if cfg.get("splits") else None
    task_types = list(cfg.task_types) if cfg.get("task_types") else None
    samples = _load_ragtruth_samples(
        path=path, language=config.language, splits=splits, task_types=task_types
    )
    logger.info(
        f"Loaded {len(samples)} RAGTruth samples from {path} "
        f"(language={config.language}, splits={splits}, task_types={task_types})."
    )

    n_samples = int(cfg.get("n_samples", 0) or 0)
    if n_samples and n_samples < len(samples):
        rng = random.Random(int(cfg.get("seed", 42)))
        samples = rng.sample(samples, n_samples)
        logger.info(f"Subsampled to {len(samples)} RAGTruth rows.")

    preload_human = bool(cfg.get("preload_human_annotations", True))
    rows: list[dict[str, Any]] = []
    ported_human = 0
    ported_llm = 0
    for sample in tqdm(samples, desc="RAGTruth detector pass"):
        prompt = sample["prompt"]
        answer = sample["answer"]
        tokens = _run_detector(detector, prompt=prompt, answer=answer)
        row_hash = generate_hash(context=[prompt], question="", answer=answer)
        row: dict[str, Any] = {
            "hash": row_hash,
            "context": [prompt],
            "question": "",
            "answer": answer,
            "gold_answer": "",
            "tokens": tokens,
            "model_predicted_hallucinated_text": hallucinated_text_from_tokens(tokens),
            "llm_hallucinated_parts": [],
            "llm_explanation": "",
            "source": "ragtruth",
            "ragtruth_task_type": sample.get("task_type"),
            "ragtruth_split": sample.get("split"),
            **default_human_fields(num_tokens=len(tokens)),
        }
        if preload_human:
            human_spans = char_label_spans_to_strings(answer, sample.get("labels", []))
            row["human_hallucinated_parts"] = human_spans
            row["human_annotation_labels"] = spans_to_token_labels(tokens, human_spans)
            row["human_annotation_notes"] = "Pre-populated from RAGTruth gold labels."
            row["human_annotation_status"] = HUMAN_STATUS_ANNOTATED
            row["human_annotated_at"] = None
        prev = prev_by_hash.get(row_hash)
        if prev is not None and port_annotations(prev, row):
            if row.get("human_annotation_status") not in (None, "unannotated"):
                ported_human += 1
            if row.get("llm_hallucinated_parts") or row.get("llm_explanation"):
                ported_llm += 1
        rows.append(row)
    return rows, ported_human, ported_llm


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Build the 3-tier ground-truth evaluation dataset for the configured language."""
    logging.getLogger("httpx").setLevel(logging.WARNING)

    target_dataset_name = (
        f"{config.base_dataset.id}-{config.language}-"
        f"{config.models.eval_model.split('/')[1]}"
    )
    answers_path = Path("data", "final", f"{target_dataset_name}.jsonl")

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

    # Map each row hash back to its gold QA answer so annotators can see it.
    gold_by_hash: dict[str, str] = {
        generate_hash(context=ctx, question=q, answer=a): a
        for ctx, q, a in zip(contexts, questions, answers)
    }

    # Reuse cached answers (or generate any that are missing).
    generated_answers = generate_answers_from_qa_data(
        eval_model=config.models.eval_model,
        contexts=contexts,
        questions=questions,
        answers=answers,
        lang=config.language,
        max_new_tokens=config.generation.max_new_tokens,
        output_jsonl_path=answers_path,
    )

    if len(generated_answers) == 0:
        logger.warning("No generated answers available — nothing to predict on.")
        return

    rag_truth_dataset = format_dataset_to_ragtruth_without_labels(
        generated_answers, language=config.language, split="test"
    )

    detector_path = _resolve_detector_path(config)
    logger.info(f"Loading hallucination detector: {detector_path}")
    detector = HallucinationDetector(
        method="transformer",
        model_path=detector_path,
        device_map="auto",
        torch_dtype="auto",
    )

    output_path = Path(config.manual_annotation.dataset_path)

    # Load existing rows (if any) so we can carry forward human + LLM-judge
    # annotations across detector swaps. Re-projection onto the new token grid
    # happens inside ``port_annotations``.
    prev_by_hash: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for existing in read_rows(output_path):
            prev_by_hash[existing["hash"]] = existing
        logger.info(
            f"Loaded {len(prev_by_hash)} existing rows from {output_path} "
            "for annotation porting."
        )

    _backup_existing(output_path)

    rows: list[dict[str, Any]] = []
    ported_human = 0
    ported_llm = 0
    empty_token_rows = 0
    for context, question, answer, prompt in zip(
        tqdm(
            generated_answers["context"],
            desc="Detector pass",
            total=len(generated_answers),
        ),
        generated_answers["question"],
        generated_answers["answer"],
        rag_truth_dataset["prompt"],
    ):
        tokens = _run_detector(detector, prompt=prompt, answer=answer)
        if not tokens:
            empty_token_rows += 1
        row_hash = generate_hash(context=context, question=question, answer=answer)
        gold_answer = gold_by_hash.get(row_hash, "")
        if not gold_answer:
            logger.warning(f"No gold answer found for hash {row_hash}")
        if not tokens:
            logger.warning(
                "Detector returned 0 tokens for multiwikiqa row hash=%s", row_hash
            )
        row: dict[str, Any] = {
            "hash": row_hash,
            "context": context,
            "question": question,
            "answer": answer,
            "gold_answer": gold_answer,
            "tokens": tokens,
            "model_predicted_hallucinated_text": hallucinated_text_from_tokens(tokens),
            "llm_hallucinated_parts": [],
            "llm_explanation": "",
            "source": "multiwikiqa",
            **default_human_fields(num_tokens=len(tokens)),
        }
        prev = prev_by_hash.get(row_hash)
        if prev is not None and port_annotations(prev, row):
            if row.get("human_annotation_status") not in (None, "unannotated"):
                ported_human += 1
            if row.get("llm_hallucinated_parts") or row.get("llm_explanation"):
                ported_llm += 1
        rows.append(row)

    if prev_by_hash:
        logger.info(
            f"Ported human annotations on {ported_human} rows and "
            f"LLM-judge verdicts on {ported_llm} rows from the previous file."
        )

    ragtruth_cfg = (
        config.manual_annotation.ragtruth
        if "manual_annotation" in config and "ragtruth" in config.manual_annotation
        else None
    )
    if ragtruth_cfg is not None and bool(ragtruth_cfg.get("enable", False)):
        rt_rows, rt_ported_human, rt_ported_llm = _build_ragtruth_rows(
            detector=detector, config=config, prev_by_hash=prev_by_hash
        )
        empty_token_rows += sum(1 for r in rt_rows if not r.get("tokens"))
        for r in rt_rows:
            if not r.get("tokens"):
                logger.warning(
                    "Detector returned 0 tokens for ragtruth row hash=%s", r.get("hash")
                )
        # Deduplicate against MultiWikiQA rows on row hash (RAGTruth hashes are
        # generated from prompt+answer so collisions are essentially impossible,
        # but guard anyway so we never write duplicate rows).
        existing_hashes = {r["hash"] for r in rows}
        rt_new = [r for r in rt_rows if r["hash"] not in existing_hashes]
        rows.extend(rt_new)
        logger.info(
            f"Appended {len(rt_new)} RAGTruth rows "
            f"({len(rt_rows) - len(rt_new)} skipped as duplicates). "
            f"Ported human annotations on {rt_ported_human} rows and "
            f"LLM-judge verdicts on {rt_ported_llm} rows from the previous file."
        )

    write_rows(rows, output_path)
    logger.info(f"Wrote {len(rows)} rows to {output_path}")
    if empty_token_rows:
        logger.warning(
            "Detected %d rows with empty detector token output. "
            "These rows are kept for continuity but should be reviewed.",
            empty_token_rows,
        )

    # Round-trip sanity check.
    reloaded = read_rows(output_path)
    assert len(reloaded) == len(rows), "round-trip row count mismatch"
    assert all(
        len(r["tokens"]) == len(r["human_annotation_labels"]) for r in reloaded
    ), "human_annotation_labels length must match token count"
    logger.info("Round-trip sanity check passed.")


if __name__ == "__main__":
    main()
