"""Run the HallBayes evaluation pipeline.

Usage:
    uv run src/scripts/hallbayes.py <config_key>=<config_value> ...

Authors
-------
https://github.com/leochlon/hallbayes
"""

import json
import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from factuality_eval.dataset_generation import load_qa_data
from factuality_eval.hallbayes.scripts.hallucination_toolkit import (
    OpenAIBackend,
    OpenAIItem,
    OpenAIPlanner,
    make_sla_certificate,
    save_sla_certificate_json,
)
from factuality_eval.prompt_utils import PromptUtils

load_dotenv()

logger = logging.getLogger(__name__)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Run HallBayes evaluation.

    Args:
        config:
            The Hydra config for your project.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Setup backend
    backend = OpenAIBackend(model="gpt-4o-mini")

    # Load QA data
    contexts, questions, answers = load_qa_data(
        base_dataset_id=f"{config.base_dataset.organisation}/{config.base_dataset.id}:{config.language}",
        split="test",
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
        max_examples=config.generation.max_examples,
    )

    # Prepare items
    items = []
    for context, question, answer in zip(contexts, questions, answers):
        prompt = PromptUtils.format_context(context, question, lang=config.language)
        items.append(
            OpenAIItem(
                prompt=prompt,
                n_samples=7,
                m=6,
                skeleton_policy="auto",
            )
        )

    logger.info(f"Evaluating {len(items)} items with HallBayes...")

    # Run evaluation
    planner = OpenAIPlanner(backend, temperature=0.3)
    metrics = planner.run(
        items,
        h_star=0.05,  # Target 5% hallucination max
        isr_threshold=1.0,  # Standard threshold
        margin_extra_bits=0.2,  # Safety margin
        B_clip=12.0,  # Clipping bound
        clip_mode="one-sided",  # Conservative mode
    )

    # Generate report and certificate
    report = planner.aggregate(items, metrics, alpha=0.05, h_star=0.05)

    # Calculate refusals
    refusals = sum(1 for m in metrics if not m.decision_answer)

    logger.info(f"Number of refusals: {refusals} out of {len(items)}")
    logger.info(f"Refusal rate: {refusals / len(items):.2%}")
    logger.info(f"Answer rate: {report.answer_rate:.2%}")

    # Save results
    output_dir = Path("data", "final", "hallbayes")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name_safe = _safe_model_name("gpt-4o-mini")

    # Save certificate
    cert = make_sla_certificate(report, model_name="GPT-4o-mini")
    cert_path = output_dir / (
        f"{config.base_dataset.id}-{config.language}-{model_name_safe}-certificate.json"
    )
    save_sla_certificate_json(cert, str(cert_path))
    logger.info(f"Saved certificate to {cert_path}")

    # Save summary results
    results = {
        "model": "gpt-4o-mini",
        "language": config.language,
        "total_items": len(items),
        "refusals": refusals,
        "refusal_rate": refusals / len(items),
        "answer_rate": report.answer_rate,
        # "abstention_rate": report.abstention_rate,
        # "wilson_upper_bound": report.wilson_upper,
        # "worst_item_roh_bound": report.worst_item_roh_bound,
        # "median_item_roh_bound": report.median_item_roh_bound,
        # "h_star": report.h_star,
        # "isr_threshold": report.isr_threshold,
    }

    results_path = output_dir / (
        f"{config.base_dataset.id}-{config.language}-{model_name_safe}-hallbayes.jsonl"
    )
    with results_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")

    logger.info(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
