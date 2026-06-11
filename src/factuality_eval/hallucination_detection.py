"""Detection of hallucinations in a dataset."""

import logging
from collections import defaultdict

from datasets import Dataset
from lettucedetect.models.inference import HallucinationDetector

logger = logging.getLogger(__name__)


def detect_hallucinations(
    dataset: Dataset, model: str = "KRLabsOrg/tinylettuce-ettin-17m-en"
) -> dict[str, list]:
    """Load tinylettuce model and detect hallucinations.

    Args:
        dataset: Hallucination dataset, generated with e.g. lettuce.
        model: Path to model.

    Returns:
        A dictionary with the predicted answers and ground truth hallucinated parts.
    """
    required_columns = {"prompt", "answer"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        logger.warning(
            "Skipping hallucination detection. Dataset missing columns: %s",
            ", ".join(sorted(missing_columns)),
        )
        return {"predict_answers": [], "ground_truth": []}

    detector = HallucinationDetector(
        method="transformer",
        model_path=model,
        device_map="auto",
        torch_dtype="auto",
        max_length=8192,
    )

    tokenizer = detector.detector.tokenizer
    max_length = detector.detector.max_length

    has_hallucinated_parts = "hallucinated_parts" in dataset.column_names
    hallucinated_parts = (
        dataset["hallucinated_parts"]
        if has_hallucinated_parts
        else [None] * len(dataset)
    )

    predict_answers = []
    all_hallucinated_parts = []
    for prompt, answer, hallucinated_part in zip(
        dataset["prompt"], dataset["answer"], hallucinated_parts
    ):
        answer_token_count = len(tokenizer.encode(answer, add_special_tokens=False))
        if answer_token_count >= max_length:
            logger.warning(
                "Skipping sample: answer has %d tokens, which exceeds the detector's "
                "max_length of %d.",
                answer_token_count,
                max_length,
            )
            continue

        # Use the detector to predict if the answer is hallucinated
        predict_answer = detector.predict_prompt(prompt=prompt, answer=answer)
        predict_answers.append(predict_answer)
        if has_hallucinated_parts:
            all_hallucinated_parts.append(hallucinated_part)

    data_dict: dict[str, list] = defaultdict(list)
    data_dict["predict_answers"] = predict_answers
    data_dict["ground_truth"] = all_hallucinated_parts

    return data_dict


def evaluate_predicted_answers(hallucinations: dict) -> None:
    """Evaluate the predicted answers for hallucinations.

    Args:
        hallucinations:
            A dictionary with the predicted answers and ground truth hallucinated parts.
    """
    logger.info("Evaluating model answers for hallucinations...")

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

    hallucination_rate = hallucinated_tokens / total_tokens if total_tokens > 0 else 0.0

    answers_with_hallucinations = sum([1 for x in no_hallucination_in_answers if x > 0])

    rate_with_hallucinations = (
        answers_with_hallucinations / len(no_hallucination_in_answers)
        if no_hallucination_in_answers
        else 0.0
    )
    logger.info("Results ________________________________________")
    logger.info(
        f"Hallucination rate (hallucinated_tokens/total_tokens) : "
        f"{hallucination_rate:.2f}"
    )
    logger.info(
        f"Rate of answers with at least one hallucination: "
        f"{rate_with_hallucinations:.2f}"
    )
    return
