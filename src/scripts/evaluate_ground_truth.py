"""Evaluate the hallucination detector against token-level ground truth.

Phase 1 loads the synthetic-hallucinations dataset from the HuggingFace Hub
(it is *not* regenerated) and uses its ``hallucinated_labels`` column as
ground truth, computing per-token precision / recall / F1 / AUROC via
``lettucedetect``'s ``evaluate_model``.

Phase 2 reads the existing Qwen3-0.6B answers from ``data/final/`` and runs
the detector on them to report the hallucination rate. It also dumps
token-level predictions to a JSONL file and writes a ground-truth
evaluation dataset that combines detector predictions with empty slots for
LLM evaluation and human annotation. The LLM evaluation is filled in by a
separate script later — this script never calls an external API.

Usage:
    uv run src/scripts/evaluate_ground_truth.py <config_key>=<config_value> ...
"""

import json
import logging
import os
from pathlib import Path

import hydra
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from lettucedetect import HallucinationDataset
from lettucedetect.models.evaluator import evaluate_model, print_metrics
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from factuality_eval.dataset_generation import (
    generate_lettucedetect_hallucination_samples,
)
from factuality_eval.ground_truth_eval import build_ground_truth_evaluation_dataset
from factuality_eval.hallucination_detection import (
    detect_hallucinations,
    evaluate_predicted_answers,
)
from factuality_eval.train import format_dataset_to_ragtruth

load_dotenv()

logger = logging.getLogger("evaluate_ground_truth")


def _resolve_model_path(config: DictConfig, target_dataset_name: str) -> str:
    """Return the local model directory if it exists, else the Hub repo id."""
    local_path = (
        f"{config.training.output_dir}/"
        f"{config.models.hallu_detect_model}-{target_dataset_name}-{config.language}"
    )
    if os.path.isdir(local_path):
        logger.info(f"Using local model checkpoint at {local_path}")
        return local_path

    hub_path = (
        f"{config.hub_organisation}/"
        f"{config.models.hallu_detect_model}-{target_dataset_name}-{config.language}"
    )
    logger.info(f"Local checkpoint not found; using Hub model {hub_path}")
    return hub_path


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Run ground-truth evaluation followed by inference on real model answers."""
    logging.getLogger("httpx").setLevel(logging.WARNING)

    target_dataset_name = f"{config.base_dataset.id}-synthetic-hallucinations"
    model_path = _resolve_model_path(config, target_dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Phase 1 — Token-level ground-truth evaluation on the synthetic set.
    # ------------------------------------------------------------------
    logger.info("Phase 1: token-level ground-truth evaluation")

    dataset = load_dataset(
        f"{config.hub_organisation}/{target_dataset_name}", name=config.language
    )
    test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)["test"]

    test_ragtruth = format_dataset_to_ragtruth(
        test_split, language=config.language, split="test"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.models.pretrained_model, trust_remote_code=True
    )
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )
    test_hallu_dataset = HallucinationDataset(
        generate_lettucedetect_hallucination_samples(test_ragtruth),
        tokenizer,
        max_length=config.training.max_length,
    )
    test_loader = DataLoader(
        test_hallu_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_path, trust_remote_code=True
    )
    model.to(device)

    metrics = evaluate_model(model, test_loader, device)
    print_metrics(metrics)

    # ------------------------------------------------------------------
    # Phase 2 — Hallucination rate on the existing Qwen3-0.6B answers.
    # ------------------------------------------------------------------
    logger.info("Phase 2: detection on existing model answers")

    answers_filename = (
        f"{config.base_dataset.id}-{config.language}-"
        f"{config.models.eval_model.split('/')[1]}.jsonl"
    )
    answers_path = Path("data", "final", answers_filename)
    if not answers_path.exists():
        raise FileNotFoundError(
            f"Expected pre-generated answers at {answers_path}, but file is missing."
        )

    model_answers = Dataset.from_json(str(answers_path))
    hallucinations = detect_hallucinations(model_answers, model=model_path)
    evaluate_predicted_answers(hallucinations)

    predictions_path = Path("data", "final", "evaluate_ground_truth_predictions.jsonl")
    with predictions_path.open("w", encoding="utf-8") as f:
        for sample_hash, predictions in zip(
            model_answers["hash"], hallucinations["predict_answers"]
        ):
            f.write(
                json.dumps(
                    {"hash": sample_hash, "predictions": predictions},
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info(f"Wrote token-level predictions to {predictions_path}")

    # ------------------------------------------------------------------
    # Phase 3 — Build a human-annotatable ground-truth evaluation dataset.
    # No external API is called here; the LLM column is left empty and
    # filled in by a separate script later.
    # ------------------------------------------------------------------
    logger.info("Phase 3: writing ground-truth evaluation dataset (no API calls)")

    eval_dataset_path = Path("data", "final", "ground_truth_evaluation_dataset.jsonl")
    build_ground_truth_evaluation_dataset(
        hashes=list(model_answers["hash"]),
        contexts=list(model_answers["context"]),
        questions=list(model_answers["question"]),
        answers=list(model_answers["answer"]),
        predictions=hallucinations["predict_answers"],
        output_path=eval_dataset_path,
    )


if __name__ == "__main__":
    main()
