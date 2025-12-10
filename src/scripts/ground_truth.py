"""Evaluate model answers against ground truth."""

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from openai import OpenAI
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from factuality_eval.dataset_generation import (
    load_qa_data
)
from factuality_eval.model_generation import (
    generate_answers_from_qa_data
)
from factuality_eval.prompt_utils import Lang
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

# Language-specific evaluation prompts
EVAL_PROMPTS: dict[Lang, str] = {
    "en": "Answer with only 'yes' or 'no': Is the text '{generated}' essentially the same as '{ground_truth}'?",
    "de": "Antworte nur mit 'ja' oder 'nein': Ist die Text '{generated}' im Wesentlichen dasselbe wie '{ground_truth}'?",
    "da": "Svar kun med 'ja' eller 'nej': Er teksten '{generated}' i det væsentlige det samme som '{ground_truth}'?",
}

# Language-specific refusal detection prompts
REFUSAL_PROMPTS: dict[Lang, str] = {
    "en": "Answer with only 'yes' or 'no': Does the answer '{generated}' indicate a refusal to answer the question?",
    "de": "Antworte nur mit 'ja' oder 'nein': Zeigt die Antwort '{generated}' eine Verweigerung die Frage zu beantworten?",
    "da": "Svar kun med 'ja' eller 'nej': Indikerer svaret '{generated}' en afvisning til at besvare spørgsmålet?",
}

# Language-specific positive responses
POSITIVE_RESPONSES: dict[Lang, list[str]] = {
    "en": ["yes"],
    "de": ["ja"],
    "da": ["ja"],
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
    ground_truth_dataset_name = (
        f"{config.base_dataset.id}-{config.language}-"
        f"{config.models.eval_model.split('/')[1]}"
    )
    
    # Load test data
    contexts, questions, answers = load_qa_data(
        base_dataset_id=f"{config.base_dataset.organisation}/{config.base_dataset.id}:{config.language}",
        split="test",
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
    )

    # Generate answers using existing function
    dataset = generate_answers_from_qa_data(
        contexts=contexts,
        questions=questions,
        answers=answers,
        eval_model=config.models.eval_model,
        output_jsonl_path=Path("data", "final", f"{ground_truth_dataset_name}-reference.jsonl"),
        lang=config.language,
    )

    # Evaluate with GPT-4o-mini
    client = OpenAI()
    lang: Lang = config.language
    eval_prompt_template = EVAL_PROMPTS[lang]
    refusal_prompt_template = REFUSAL_PROMPTS[lang]
    positive_responses = POSITIVE_RESPONSES[lang]

    correct = 0
    wrong = 0
    refusals = 0
    detailed_results = []

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        gen_answer = item["answer"]
        gt_answer = answers[idx]
        question = questions[idx]
        context = contexts[idx]

        # First check if it's a refusal
        refusal_prompt = refusal_prompt_template.format(generated=gen_answer)
        refusal_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": refusal_prompt}],
            temperature=0,
        )
        refusal_result = refusal_response.choices[0].message.content.strip().lower()

        result_entry = {
            #"index": idx,
            #"question": question,
            #"context": context,
            "ground_truth_answer": gt_answer,
            "generated_answer": gen_answer,
        }

        if any(pos in refusal_result for pos in positive_responses):
            refusals += 1
            result_entry["evaluation"] = "refused"
        else:
            # Check if answer is correct
            eval_prompt = eval_prompt_template.format(
                generated=gen_answer, ground_truth=gt_answer
            )
            eval_response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0,
            )
            eval_result = eval_response.choices[0].message.content.strip().lower()

            if any(pos in eval_result for pos in positive_responses):
                correct += 1
                result_entry["evaluation"] = "correct"
            else:
                wrong += 1
                result_entry["evaluation"] = "incorrect"
        
        detailed_results.append(result_entry)

    # Write summary results
    total = len(dataset)
    summary = {
        "model": config.models.eval_model,
        "language": config.language,
        "total": total,
        "correct": correct,
        "correct_pct": round(correct / total * 100, 2),
        "wrong": wrong,
        "wrong_pct": round(wrong / total * 100, 2),
        "refusals": refusals,
        "refusal_pct": round(refusals / total * 100, 2),
    }

    output_dir = Path("data", "ground_truth")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.models.eval_model.split("/")[-1]
    
    # Save summary
    summary_file = output_dir / f"{model_name}_{config.language}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save detailed results as JSONL
    detailed_file = output_dir / f"{model_name}_{config.language}_detailed.jsonl"
    with open(detailed_file, "w", encoding="utf-8") as f:
        for result in detailed_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(f"Summary written to: {summary_file}")
    logger.info(f"Detailed results written to: {detailed_file}")
    logger.info(f"Correct: {correct}/{total} ({summary['correct_pct']}%)")
    logger.info(f"Incorrect: {wrong}/{total} ({summary['wrong_pct']}%)")
    logger.info(f"Refusals: {refusals}/{total} ({summary['refusal_pct']}%)")


if __name__ == "__main__":
    main()