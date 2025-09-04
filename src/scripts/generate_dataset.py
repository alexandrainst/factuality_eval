"""Generate a hallucination dataset.

Usage:
    uv run src/scripts/generate_dataset.py <config_key>=<config_value> ...
"""

import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from factuality_eval.dataset_generation import generate_hallucination_dataset

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
    generate_hallucination_dataset(config=config)


if __name__ == "__main__":
    main()
