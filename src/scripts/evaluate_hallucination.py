"""Evaluate a hallucinations.

Usage:
    uv run src/scripts/evaluate_hallucination.py <config_key>=<config_value> ...
"""

import json
import logging
import os

import hydra
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig

from factuality_eval.detect_hallucinations import detect_hallucinations

load_dotenv()


@hydra.main(
    config_path="../../config", config_name="dataset_generation", version_base=None
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

    # Detect hallucinations
    hallucinations = detect_hallucinations(dataset)

    # Save to Hydra's output directory
    predictions_file = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        "predict_hallucinations.json",
    )

    with open(predictions_file, "w") as f:
        json.dump(hallucinations, f, indent=4)


if __name__ == "__main__":
    main()
