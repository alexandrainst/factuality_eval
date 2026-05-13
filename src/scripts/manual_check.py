"""Generate a manual-check report comparing ground truth and generated answers."""

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from factuality_eval.dataset_generation import load_qa_data

logger = logging.getLogger(__name__)


def _normalize_cell(text: str, max_chars: int) -> str:
    cleaned = text.replace("\n", "<br>").replace("|", "\\|").strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        return cleaned[: max_chars - 3].rstrip() + "..."
    return cleaned


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Create a Markdown report for manual checking."""
    manual_cfg = config.manual_check
    input_path = Path(manual_cfg.input_path)
    output_path = Path(manual_cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    contexts, questions, answers = load_qa_data(
        base_dataset_id=(
            f"{config.base_dataset.organisation}/{config.base_dataset.id}"
            f":{config.language}"
        ),
        split="test",
        context_key=config.base_dataset.context_key,
        question_key=config.base_dataset.question_key,
        answer_key=config.base_dataset.answer_key,
        squad_format=config.base_dataset.squad_format,
        testing=config.testing,
        max_examples=config.generation.max_examples,
    )

    gt_by_key = {
        (context[0], question): answer
        for context, question, answer in zip(contexts, questions, answers)
    }

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    max_rows = manual_cfg.max_rows
    if max_rows > 0:
        rows = rows[:max_rows]

    include_context = manual_cfg.include_context
    max_chars = int(manual_cfg.max_chars)
    context_chars = int(manual_cfg.context_chars)

    header = (
        "# Manual Check Report\n\n"
        f"Input: {input_path}\n\n"
        "| idx | question | ground_truth | generated | verdict | notes |"
        "\n| --- | --- | --- | --- | --- | --- |\n"
    )

    if include_context:
        header = (
            "# Manual Check Report\n\n"
            f"Input: {input_path}\n\n"
            "| idx | question | ground_truth | generated | context | verdict | notes |"
            "\n| --- | --- | --- | --- | --- | --- | --- |\n"
        )

    lines = [header]
    missing_gt = 0

    for idx, row in enumerate(rows, start=1):
        question = row.get("question", "")
        context = row.get("context", [""])
        context_text = context[0] if context else ""
        generated = row.get("answer", "")
        ground_truth = gt_by_key.get((context_text, question), "")
        if not ground_truth:
            missing_gt += 1

        question_cell = _normalize_cell(question, max_chars)
        gt_cell = _normalize_cell(ground_truth, max_chars)
        gen_cell = _normalize_cell(generated, max_chars)

        if include_context:
            context_cell = _normalize_cell(context_text, context_chars)
            line = (
                f"| {idx} | {question_cell} | {gt_cell} | {gen_cell} | "
                f"{context_cell} |  |  |\n"
            )
        else:
            line = f"| {idx} | {question_cell} | {gt_cell} | {gen_cell} |  |  |\n"
        lines.append(line)

    output_path.write_text("".join(lines), encoding="utf-8")

    if missing_gt:
        logger.warning("Missing ground truth for %s rows", missing_gt)
    logger.info("Manual check report written to: %s", output_path)


if __name__ == "__main__":
    main()
