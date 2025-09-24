"""Detect hallucinations.

Usage:
    uv run src/scripts/detect_hallucinations.py <config_key>=<config_value> ...
"""

import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from factuality_eval.dataset_generation import (
    generate_answers_from_qa_data,
    load_qa_data,
)
from factuality_eval.hallucination_detection import detect_hallucinations

load_dotenv()

logger = logging.getLogger(__name__)


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

    base_dataset_id = f"{config.base_dataset.id}-{config.language}"
    model_name = config.models.eval_model
    target_dataset_name = f"{base_dataset_id}-{model_name.split('/')[1]}"

    # Load from hub and split into train/test
    contexts, questions, answers = load_qa_data(
        base_dataset_id=f"{config.base_dataset.organisation}/{config.base_dataset.id}:{config.language}",
        split="train",
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
    )

    # Convert to lists to avoid numpy type issues
    contexts = list(contexts)
    questions = list(questions)
    answers = list(answers)

    from sklearn.model_selection import train_test_split

    _, contexts, _, questions, _, answers = train_test_split(
        contexts, questions, answers, test_size=0.2, random_state=42, shuffle=False
    )

    generated_dataset = generate_answers_from_qa_data(
        contexts=contexts,
        questions=questions,
        answers=answers,
        model=config.models.eval_model,
        output_jsonl_path=Path("data", "final", f"{target_dataset_name}.jsonl"),
    )
    hugging_face_path = (
        f"{config.hub_organisation}/"
        f"{config.models.hallu_detect_model}-{config.base_dataset.id}-{config.language}"
    )

    hallucinations = detect_hallucinations(generated_dataset, model=hugging_face_path)

    print(len(hallucinations))

    no_hallucination_in_answers = []
    no_tokens_in_answers = []

    hallucinated_tokens = 0
    total_tokens = 0
    for predict_answer in hallucinations["predict_answers"]:
        no_hallucination_in_answer = 0
        no_tokens_in_answer = 0
        for tokens in predict_answer:
            hallucinated_tokens += tokens["pred"]
            total_tokens += 1

            no_hallucination_in_answer += tokens["pred"]
            no_tokens_in_answer += 1
        no_hallucination_in_answers.append(no_hallucination_in_answer)
        no_tokens_in_answers.append(no_tokens_in_answer)

    logger.info("Evaluating model answers for hallucinations...")

    hallucination_rate = hallucinated_tokens / total_tokens
    logger.info(
        f"Hallucination rate (hallucinated_tokens/total_tokens) : "
        f"{hallucination_rate:.2f}"
    )

    avg_hallucinations = sum(no_hallucination_in_answers) / len(
        no_hallucination_in_answers
    )
    logger.info(f"Average hallucinations per answer: {avg_hallucinations:.2f}")

    answers_with_hallucinations = sum([1 for x in no_hallucination_in_answers if x > 0])
    rate_with_hallucinations = answers_with_hallucinations / len(
        no_hallucination_in_answers
    )
    logger.info(
        f"Rate of answers with at least one hallucination: "
        f"{rate_with_hallucinations:.2f}"
    )

    avg_tokens = sum(no_tokens_in_answers) / len(no_tokens_in_answers)
    logger.info(f"Average tokens per answer: {avg_tokens:.2f}")
    logger.info(f"Total answers: {len(no_hallucination_in_answers)}")
    logger.info(f"Total hallucinated tokens: {hallucinated_tokens}")
    logger.info(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
