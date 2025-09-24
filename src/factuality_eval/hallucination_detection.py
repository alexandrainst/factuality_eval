"""Detection of hallucinations in a dataset."""

from collections import defaultdict

from datasets import Dataset
from lettucedetect.models.inference import HallucinationDetector


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
        dataset["context"], dataset["question"], dataset["generated_answer"]
    ):
        # Use the detector to predict if the answer is hallucinated
        predict_answer = detector.predict(
            context=context, question=question, answer=answer
        )

        predict_answers.append(predict_answer)

    if "hallucinated_parts" in dataset.column_names:
        for hallucinated_part in dataset["hallucinated_parts"]:
            all_hallucinated_parts.append(hallucinated_part)

    data_dict: dict[str, list] = defaultdict(list)
    data_dict["predict_answers"] = predict_answers
    data_dict["ground_truth"] = all_hallucinated_parts

    return data_dict
