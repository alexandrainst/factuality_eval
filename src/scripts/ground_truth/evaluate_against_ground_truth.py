"""Validate the trained hallucination detector against token-level labels.

This script loads the synthetic-hallucinations dataset from the HuggingFace Hub
(it is not regenerated), uses its ``hallucinated_labels`` column as ground truth,
and computes token-level metrics via ``lettucedetect``'s ``evaluate_model``. Its
purpose is to check whether the trained detection method can identify
hallucinated tokens.

Usage:
    uv run src/scripts/ground_truth/evaluate_ground_truth.py
        <config_key>=<config_value> ...
"""

import logging
import os

import hydra
import torch
from datasets import load_dataset
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
from factuality_eval.train import format_dataset_to_ragtruth

load_dotenv()

logger = logging.getLogger("evaluate_ground_truth")


def _training_sources_suffix(config: DictConfig) -> str:
    """Return the model-name suffix used by the training script."""
    sources = []
    if config.multiwikiqa.enable:
        sources.append("mwqa")
    if config.ragtruth.enable:
        sources.append("ragtruth")
    return f"-{'+'.join(sources)}" if sources else ""


def _resolve_model_path(config: DictConfig, target_dataset_name: str) -> str:
    """Return the local model directory if it exists, else the Hub repo id."""
    suffix = _training_sources_suffix(config)
    local_path = (
        f"{config.training.output_dir}/"
        f"{config.models.hallu_detect_model}-{target_dataset_name}-"
        f"{config.language}{suffix}"
    )
    if os.path.isdir(local_path):
        logger.info(f"Using local model checkpoint at {local_path}")
        return local_path

    hub_path = (
        f"{config.hub_organisation}/"
        f"{config.models.hallu_detect_model}-{target_dataset_name}-"
        f"{config.language}{suffix}"
    )
    logger.info(f"Local checkpoint not found; using Hub model {hub_path}")
    return hub_path


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Run token-level validation for the trained hallucination detector."""
    logging.getLogger("httpx").setLevel(logging.WARNING)

    target_dataset_name = f"{config.base_dataset.id}-synthetic-hallucinations"
    model_path = _resolve_model_path(config, target_dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Running token-level ground-truth evaluation")

    dataset = load_dataset(
        f"{config.hub_organisation}/{target_dataset_name}", name=config.language
    )
    test_split = dataset["train"].train_test_split(
        test_size=0.2, seed=42, shuffle=False
    )["test"]

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


if __name__ == "__main__":
    main()
