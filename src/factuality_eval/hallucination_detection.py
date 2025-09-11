"""Detection of hallucinations in a dataset."""

from collections import defaultdict

from datasets import Dataset
from lettucedetect.models.inference import HallucinationDetector


def detect_hallucinations(
    dataset: Dataset, model: str = "KRLabsOrg/tinylettuce-ettin-17m-en"
) -> dict[str, list]:
    """Load tinylettuce model and detect hallucinations.

    Args:
        dataset:
            Hallucination dataset, generated with e.g. lettuce.
        model:
            Path to model.

    """
    detector = HallucinationDetector(method="transformer", model_path=model)

    predict_answers = []
    all_hallucinated_parts = []
    for context, question, answer, hallucinated_parts in zip(
        dataset["train"]["context"],
        dataset["train"]["question"],
        dataset["train"]["answer"],
        dataset["train"]["hallucinated_parts"],
    ):
        # Use the detector to predict if the answer is hallucinated
        predict_answer = detector.predict(
            context=context, question=question, answer=answer
        )

        predict_answers.append(predict_answer)
        all_hallucinated_parts.append(hallucinated_parts)

    data_dict: dict[str, list] = defaultdict(list)
    data_dict["predict_answers"] = predict_answers
    data_dict["ground_truth"] = all_hallucinated_parts

    return data_dict
