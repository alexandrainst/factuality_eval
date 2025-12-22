"""Train hallucination detector.

Usage:
    uv run src/scripts/train_hallucination_detector.py <config_key>=<config_value> ...
"""

import logging
import os

import hydra
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from lettucedetect import HallucinationDataset
from lettucedetect.models.evaluator import evaluate_model, print_metrics
from lettucedetect.models.trainer import Trainer
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
logger = logging.getLogger("train_hallucination_detector")


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    target_dataset_name = f"{config.base_dataset.id}-synthetic-hallucinations"

    # Load from hub
    dataset = load_dataset(
        f"{config.hub_organisation}/{target_dataset_name}", name=config.language
    )
    train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)

    # Process dataset to ragtruth format
    train_dataset = format_dataset_to_ragtruth(
        train_test_split["train"], language=config.language, split="train"
    )
    test_dataset = format_dataset_to_ragtruth(
        train_test_split["test"], language=config.language, split="test"
    )

    # Create tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.pretrained_model, trust_remote_code=True, use_safetensors=True
    )
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )

    # Create lettucedetect datasets
    train_hallu_dataset = HallucinationDataset(
        generate_lettucedetect_hallucination_samples(train_dataset), tokenizer
    )
    test_hallu_dataset = HallucinationDataset(
        generate_lettucedetect_hallucination_samples(test_dataset), tokenizer
    )

    # Create data loaders
    train_loader = DataLoader(
        train_hallu_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_hallu_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # Check if model already exists
    model_save_path = (
        f"{config.training.output_dir}/"
        f"{config.models.hallu_detect_model}-{config.base_dataset.id}-{config.language}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_save_path) and os.path.isdir(model_save_path):
        # Load existing model
        logging.info(f"Loading existing model from {model_save_path}")
        model = AutoModelForTokenClassification.from_pretrained(
            model_save_path, trust_remote_code=True, use_safetensors=True
        )
        model.to(device)

        logger.info("\nEvaluating...")
        metrics = evaluate_model(model, test_loader, device)
        print_metrics(metrics)

    else:
        model = AutoModelForTokenClassification.from_pretrained(
            config.models.pretrained_model,
            num_labels=2,
            trust_remote_code=True,
            use_safetensors=True,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            save_path=f"{config.training.output_dir}/{config.models.hallu_detect_model}-{config.base_dataset.id}-{config.language}",
        )

        logging.info("Starting training...")
        trainer.train()

        if config.training.push_to_hub:
            model.push_to_hub(
                repo_id=f"{config.hub_organisation}/{config.models.hallu_detect_model}-{config.base_dataset.id}-{config.language}",
                private=config.private,
            )
            tokenizer.push_to_hub(
                repo_id=f"{config.hub_organisation}/{config.models.hallu_detect_model}-{config.base_dataset.id}-{config.language}",
                private=config.private,
            )


if __name__ == "__main__":
    main()
