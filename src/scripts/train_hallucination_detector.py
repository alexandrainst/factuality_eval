"""Train hallucination detector.

Usage:
    uv run src/scripts/train_hallucination_detector.py <config_key>=<config_value> ...
"""

import json
import logging
import os

import hydra
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from lettucedetect import HallucinationDataset
from lettucedetect.datasets.hallucination_dataset import HallucinationData
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


def load_ragtruth_translated(
    path: Path, language: str
) -> tuple[list, list]:
    """Load a translated RAGTruth file and split into train/test using RAGTruth's native splits.

    The file is expected to be a HallucinationData JSON produced by
    LettuceDetect's translate.py. Each sample has a `split` field
    ("train" or "test") inherited from the original RAGTruth.

    Args:
        path: Path to the translated ragtruth_data_<lang>.json file.
        language: Target language code, used to filter and tag samples.

    Returns:
        Tuple of (train_samples, test_samples), each a list of sample dicts
        in RAGTruth format compatible with generate_lettucedetect_hallucination_samples.
    """
    if not path.exists():
        raise FileNotFoundError(f"Translated RAGTruth file not found: {path}")

    data = HallucinationData.from_json(json.loads(path.read_text()))

    train_samples, test_samples = [], []
    for sample in data.samples:
        # Defensive: skip samples whose language doesn't match (mixed-language files)
        if sample.language and sample.language.lower() != language.lower():
            continue
        # Convert HallucinationSample to the dict shape your other code expects
        sample_dict = {
            "prompt": sample.prompt,
            "answer": sample.answer,
            "labels": sample.labels,
            "split": sample.split,
            "task_type": sample.task_type,
            "dataset": sample.dataset,
            "language": sample.language,
        }
        if sample.split == "train":
            train_samples.append(sample_dict)
        elif sample.split == "test":
            test_samples.append(sample_dict)
        else:
            logger.warning(f"Unknown split '{sample.split}' on sample, skipping.")

    logger.info(
        f"Loaded translated RAGTruth ({language}): "
        f"{len(train_samples)} train, {len(test_samples)} test"
    )
    return train_samples, test_samples


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


    # ------------------------------------------------------------------
    # 1. Load / generate synthetic hallucination dataset (existing flow)
    # ------------------------------------------------------------------
    try:
        dataset = load_dataset(
            f"{config.hub_organisation}/{target_dataset_name}", name=config.language
        )
    except ValueError:
        logger.info(
            f"Language '{config.language}' not found in hub dataset "
            f"'{config.hub_organisation}/{target_dataset_name}'. "
            "Generating dataset locally and pushing to hub..."
        )
        contexts, questions, answers = load_qa_data(
            base_dataset_id=(
                f"{config.base_dataset.organisation}/{config.base_dataset.id}"
                f":{config.language}"
            ),
            split=config.base_dataset.split,
            context_key=config.base_dataset.context_key,
            question_key=config.base_dataset.question_key,
            answer_key=config.base_dataset.answer_key,
            squad_format=config.base_dataset.squad_format,
            testing=config.testing,
        )
        intensities = sample_hallucination_intensities(
            mean=config.beta_distribution.mean,
            std=config.beta_distribution.std,
            size=len(answers),
        )
        generated = generate_hallucinations_from_qa_data(
            contexts=contexts,
            questions=questions,
            answers=answers,
            intensities=intensities,
            model=config.models.hallu_gen_model,
            output_jsonl_path=Path(
                "data", "final", f"{target_dataset_name}-{config.language}.jsonl"
            ),
            max_workers=config.max_workers,
        )
        generated.push_to_hub(
            repo_id=f"{config.hub_organisation}/{target_dataset_name}",
            config_name=config.language,
            private=config.private,
        )
        dataset = load_dataset(
            f"{config.hub_organisation}/{target_dataset_name}", name=config.language
        )
    train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)

    # Process synthetic dataset to ragtruth format
    synthetic_train = format_dataset_to_ragtruth(
        train_test_split["train"], language=config.language, split="train"
    )
    synthetic_test = format_dataset_to_ragtruth(
        train_test_split["test"], language=config.language, split="test"
    )
    logger.info(
        f"Synthetic dataset: {len(synthetic_train)} train, {len(synthetic_test)} test"
    )

    # ------------------------------------------------------------------
    # 2. Load translated RAGTruth and combine with synthetic
    # ------------------------------------------------------------------
    if config.get("ragtruth", None) and config.ragtruth.get("path", None):
        ragtruth_path = Path(config.ragtruth.path)
        ragtruth_train, ragtruth_test = load_ragtruth_translated(
            ragtruth_path, language=config.language
        )

        # Combine the two sources. Synthetic comes first; order doesn't matter
        # for training (DataLoader shuffles), but it's predictable for logging.
        ragtruth_train_ds = Dataset.from_list(ragtruth_train)
        ragtruth_test_ds = Dataset.from_list(ragtruth_test)
        train_dataset = concatenate_datasets([synthetic_train, ragtruth_train_ds])
        test_dataset = concatenate_datasets([synthetic_test, ragtruth_test_ds])
        logger.info(
            f"Combined dataset: {len(train_dataset)} train "
            f"({len(synthetic_train)} synthetic + {len(ragtruth_train)} ragtruth), "
            f"{len(test_dataset)} test "
            f"({len(synthetic_test)} synthetic + {len(ragtruth_test)} ragtruth)"
        )
    else:
        logger.info("No ragtruth.path in config; training on synthetic data only.")
        train_dataset = synthetic_train
        test_dataset = synthetic_test

    # ------------------------------------------------------------------
    # 3. Tokenize and train (existing flow, unchanged below)
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.pretrained_model,
        trust_remote_code=True,
    )
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )

    train_hallu_dataset = HallucinationDataset(
        generate_lettucedetect_hallucination_samples(train_dataset),
        tokenizer,
        max_length=config.training.max_length,
    )
    test_hallu_dataset = HallucinationDataset(
        generate_lettucedetect_hallucination_samples(test_dataset),
        tokenizer,
        max_length=config.training.max_length,
    )

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

    # Naming: include "+ragtruth" suffix so combined-vs-synthetic-only models
    # don't overwrite each other in the output dir / hub repo.
    suffix = "-with-ragtruth" if config.get("ragtruth", None) and config.ragtruth.get("path", None) else ""
    model_save_path = (
        f"{config.training.output_dir}/"
        f"{config.models.hallu_detect_model}-{target_dataset_name}-{config.language}{suffix}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_save_path) and os.path.isdir(model_save_path):
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
            config.models.pretrained_model, num_labels=2, trust_remote_code=True
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            save_path=model_save_path,
        )

        logging.info("Starting training...")
        trainer.train()

        if config.training.push_to_hub:
            hub_repo_id = (
                f"{config.hub_organisation}/"
                f"{config.models.hallu_detect_model}-{target_dataset_name}-{config.language}{suffix}"
            )
            model.push_to_hub(repo_id=hub_repo_id, private=config.private)
            tokenizer.push_to_hub(repo_id=hub_repo_id, private=config.private)


if __name__ == "__main__":
    main()