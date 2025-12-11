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
    detector = HallucinationDetector(
        method="transformer", model_path=model, device_map="auto", torch_dtype="auto"
    )

    predict_answers = []
    all_hallucinated_parts = []
    for context, question, answer in zip(
        dataset["context"], dataset["question"], dataset["answer"]
    ):
        # Use the detector to predict if the answer is hallucinated
        try:
            predict_answer = detector.predict(
                context=context, question=question, answer=answer
            )
        except Exception as e:
            logger.error(f"Error during hallucination detection: {e}. Skipping...")
            continue
        predict_answers.append(predict_answer)

    if "hallucinated_parts" in dataset.column_names:
        for hallucinated_part in dataset["hallucinated_parts"]:
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

    Returns:
        None
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

    hallucination_rate = hallucinated_tokens / total_tokens

    answers_with_hallucinations = sum([1 for x in no_hallucination_in_answers if x > 0])

    rate_with_hallucinations = answers_with_hallucinations / len(
        no_hallucination_in_answers
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
