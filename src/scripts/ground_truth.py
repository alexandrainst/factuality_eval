"""Evaluate model answers against ground truth."""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig
from openai import OpenAI
from tqdm.auto import tqdm

from factuality_eval.dataset_generation import (
    load_qa_data,
)
from factuality_eval.model_generation import generate_answers_from_qa_data
from factuality_eval.prompt_utils import Lang

logger = logging.getLogger(__name__)

# Language-specific evaluation prompts
EVAL_PROMPTS: dict[Lang, str] = {
    "en": (
        "You are grading factual correctness. Compare the model answer to the ground truth. "
        "Respond with exactly one word: yes if the model answer explicitly contains, clearly states, or partially overlaps the ground truth, otherwise no. "
        "Model answer: '{generated}'. Ground truth: '{ground_truth}'. Reply: yes/no only."
    ),
    "de": (
        "Bewerte die faktische Korrektheit. Vergleiche Modellantwort und Referenz. "
        "Antworte mit genau einem Wort: ja, wenn die Modellantwort die Referenz ausdrücklich enthält, klar benennt oder teilweise überlappt, sonst nein. "
        "Modellantwort: '{generated}'. Referenz: '{ground_truth}'. Antwort: ja/nein."
    ),
    "da": (
        "Vurder faktuel korrekthed. Svar med præcis ét ord: ja, hvis modelsvaret udtrykkeligt indeholder, klart angiver eller delvist overlapper facit, ellers nej. "
        "Modelsvar: '{generated}'. Facit: '{ground_truth}'. Svar: ja/nej."
    ),
    "is": (
        "Metu staðreyndréttmæti. Svaraðu með nákvæmlega einu orði: já, ef svarið inniheldur, segir skýrt eða skarast að hluta við facit, annars nei. "
        "Svar: '{generated}'. Facit: '{ground_truth}'. Svar: ja/nei."
    ),
    "fo": (
        "Met um svarið er faktuelt rætt. Svara við einum orði: ja, um svarið inniheldur, sigur greitt ella samsvarar partvíst við facit, annars nei. "
        "Svar: '{generated}'. Facit: '{ground_truth}'. Svar: ja/nei."
    ),
}

# Language-specific positive responses
POSITIVE_RESPONSES: dict[Lang, list[str]] = {
    "en": ["yes"],
    "de": ["ja"],
    "da": ["ja"],
    "is": ["ja"],
    "fo": ["ja"],
}


@hydra.main(
    config_path="../../config",
    config_name="hallucination_detection",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Evaluate model answers against ground truth.

    Args:
        config:
            The Hydra config for your project.
    """
    base_dataset_id = (
        f"{config.base_dataset.organisation}/{config.base_dataset.id}"
        f":{config.language}"
    )

    target_dataset_name = (
        f"{config.base_dataset.id}-{config.language}-"
        f"{config.models.eval_model.split('/')[1]}"
    )

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

    generated_answers = generate_answers_from_qa_data(
        eval_model=config.models.eval_model,
        contexts=contexts,
        questions=questions,
        answers=answers,
        lang=config.language,
        max_new_tokens=config.generation.max_new_tokens,
        output_jsonl_path=Path("data", "final", f"{target_dataset_name}.jsonl"),
    )

    # Evaluate with GPT-4o-mini
    load_dotenv()  # Load OPENAI_API_KEY from .env if present
    client = OpenAI()
    lang: Lang = config.language
    eval_prompt_template = EVAL_PROMPTS[lang]
    positive_responses = POSITIVE_RESPONSES[lang]

    correct = 0
    wrong = 0

    for item, gt_answer in tqdm(
        zip(generated_answers, answers),
        desc="Evaluating",
        total=len(generated_answers),
    ):
        gen_answer = item["answer"]

        eval_prompt = eval_prompt_template.format(
            generated=gen_answer, ground_truth=gt_answer
        )
        eval_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0,
        )
        eval_result = eval_response.choices[0].message.content.strip().lower()
        print("")
        if any(pos in eval_result for pos in positive_responses):
            correct += 1
        else:
            wrong += 1

    # Write results
    total = len(generated_answers)
    results = {
        "model": config.models.eval_model,
        "language": config.language,
        "total": total,
        "correct": correct,
        "correct_pct": round(correct / total * 100, 2),
        "wrong": wrong,
        "wrong_pct": round(wrong / total * 100, 2),
    }

    output_dir = Path("data", "ground_truth")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.models.eval_model.split("/")[-1]
    results_file = output_dir / f"{model_name}_{config.language}_results.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results written to: {results_file}")


if __name__ == "__main__":
    main()
