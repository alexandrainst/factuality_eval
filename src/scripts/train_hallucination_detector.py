"""Train hallucination detector.

Usage:
    uv run src/scripts/train_hallucination_detector.py <config_key>=<config_value> ...
"""

import logging
import os

import hydra
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from lettucedetect import HallucinationDataset, HallucinationSample
from lettucedetect.models.evaluator import evaluate_model, print_metrics

# from lettucedetect.models.evaluator import DataLoader
from lettucedetect.models.trainer import Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from factuality_eval.train import format_dataset_to_ragtruth

load_dotenv()


def generate_lettucedetect_hallucination_samples(
    dataset_split: Dataset,
) -> list[HallucinationSample]:
    """Generate hallucination samples for the LettuceDetect model.

    Args:
        dataset_split: The dataset split to generate samples from.

    Returns:
        A list of hallucination samples.
    """
    samples = []
    for item in dataset_split:
        sample = HallucinationSample(
            prompt=item["prompt"],
            answer=item["answer"],
            labels=item["labels"],
            split=item["split"],
            task_type=item["task_type"],
            dataset=item["dataset"],
            language=item["language"],
        )
        samples.append(sample)
    return samples


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)

    target_dataset_name = (
        f"{config.base_dataset.id.split('/')[-1].replace(':', '-')}-hallucinated"
    )

    # Load from hub
    dataset = load_dataset(f"{config.hub_organisation}/{target_dataset_name}")
    train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)

    # Process dataset to ragtruth format
    train_dataset = format_dataset_to_ragtruth(
        train_test_split["train"], language=config.training.language, split="train"
    )
    test_dataset = format_dataset_to_ragtruth(
        train_test_split["test"], language=config.training.language, split="test"
    )

    # Create tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.pretrained_model_name,
        trust_remote_code=True,
        use_safetensors=True,
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
    model_save_path = f"{config.training.output_dir}/{config.models.target_model_name}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_save_path) and os.path.isdir(model_save_path):
        # Load existing model
        logging.info(f"Loading existing model from {model_save_path}")
        model = AutoModelForTokenClassification.from_pretrained(
            model_save_path, trust_remote_code=True, use_safetensors=True
        )
        model.to(device)

        print("\nEvaluating...")
        metrics = evaluate_model(model, test_loader, device)
        print_metrics(metrics)

    else:
        model = AutoModelForTokenClassification.from_pretrained(
            config.models.pretrained_model_name,
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
            save_path=f"{config.training.output_dir}/{config.models.target_model_name}",
        )

        logging.info("Starting training...")
        trainer.train()

        if config.training.push_to_hub:
            model.push_to_hub(
                repo_id=f"{config.hub_organisation}/{config.models.target_model_name}",
                private=config.private,
            )
            tokenizer.push_to_hub(
                repo_id=f"{config.hub_organisation}/{config.models.target_model_name}",
                private=config.private,
            )

    # Detect hallucinations
    # hallucinations = detect_hallucinations(
    #     train_test_split["test"],
    #     model=f"{config.hub_organisation}/{config.models.target_model_name}",
    # )

    # Save to Hydra's output directory
    # predictions_file = os.path.join(
    #     hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    #     "predict_hallucinations.json",
    # )

    # if config.save_dataset_to_file:
    #     with open(predictions_file, "w") as f:
    #         json.dump(hallucinations, f, indent=4)


if __name__ == "__main__":
    main()
