# Factuality Evaluation of LLMs

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-48%25-orange.svg)](https://github.com/alexandrainst/factuality_eval/tree/main/tests)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/factuality_eval)
[![License](https://img.shields.io/github/license/alexandrainst/factuality_eval)](https://github.com/alexandrainst/factuality_eval/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/factuality_eval)](https://github.com/alexandrainst/factuality_eval/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/factuality_eval/blob/main/CODE_OF_CONDUCT.md)

This repository contains tools for evaluating factuality and hallucination
behaviour in large language models. The current focus is a multilingual
token-level hallucination detector trained with LettuceDetect on synthetic
MultiWikiQA hallucinations, optionally mixed with translated RAGTruth examples.

The project supports three main workflows:

- Generate synthetic hallucination-labelled QA data.
- Fine-tune a token-classification model to detect hallucinated spans.
- Generate answers from an evaluation model and estimate hallucination rates.

## Training Pipeline

The detector is trained as a token-classification model. MultiWikiQA question-answer
pairs are split into faithful and hallucinated examples. The hallucinated examples
are created with LettuceDetect's `HallucinationGenerator`, using hallucination
intensities sampled from a clipped Beta distribution. The resulting token-level
labels are used to fine-tune `jhu-clsp/mmBERT-small` with LettuceDetect's trainer.

Translated RAGTruth data can be added to the training mix with the Hydra config
flags `ragtruth.enable` and `multiwikiqa.enable`.

![Training pipeline: MultiWikiQA + Beta-sampled hallucination intensities feed LettuceDetect's HallucinationGenerator to produce a synthetic dataset, which fine-tunes mmBERT-small into a token-level hallucination classifier.](diagram.png)

Background reading and design notes are kept separately in
[`research/README.md`](research/README.md).

## Installation

The project uses Python 3.11 and `uv` for dependency management.

```bash
make install
```

After installation, activate the virtual environment:

```bash
source .venv/bin/activate
```

If you only need to install dependencies without the interactive setup, run:

```bash
uv sync --all-extras --python 3.11
```

Some workflows need external credentials:

- Set `OPENAI_API_KEY` when using OpenAI-backed generation models such as
  `gpt-5-mini` or `openai/<model-name>`.
- Log in to Hugging Face with `huggingface-cli login` when loading private
  datasets/models or pushing generated datasets and checkpoints.

## Configuration

The main Hydra config is [`config/hallucination_detection.yaml`](config/hallucination_detection.yaml).
Important settings include:

- `language`: target language code, for example `da`, `en`, or `de`.
- `base_dataset`: the Hugging Face QA dataset and field names.
- `models.hallu_gen_model`: model used to generate synthetic hallucinations.
- `models.pretrained_model`: base encoder used for token classification.
- `models.eval_model`: model whose answers should be evaluated.
- `ragtruth.enable` and `multiwikiqa.enable`: training data sources.
- `training`: output directory, batch size, epochs, learning rate, and hub upload settings.
- `generation.max_examples`: maximum number of examples to generate or evaluate.

Hydra values can be overridden from the command line:

```bash
uv run src/scripts/train_hallucination_detector.py language=en training.epochs=3 testing=true
```

## Common Workflows

Generate a synthetic hallucination dataset and push it to the configured Hugging
Face Hub repository:

```bash
uv run src/scripts/generate_dataset.py language=da
```

Train or evaluate the hallucination detector:

```bash
uv run src/scripts/train_hallucination_detector.py language=da
```

Run the detector on gold answers to establish a baseline hallucination rate:

```bash
uv run src/scripts/baseline.py language=da
```

Generate answers from `models.eval_model` and evaluate them with the trained
hallucination detector:

```bash
uv run src/scripts/detect_hallucinations.py language=da models.eval_model=Qwen/Qwen3-0.6B
```

Generate a JSONL dataset containing generated answers and hallucinated token spans:

```bash
uv run src/scripts/generate_hallucination_dataset.py language=da models.eval_model=Qwen/Qwen3-0.6B
```

Evaluate a trained hallucination detector against token-level ground-truth labels:

```bash
uv run src/scripts/evaluate_ground_truth.py language=da
```

## Data and Outputs

Generated files are written under `data/` when the relevant workflow is run. Common
outputs include:

- `data/final/*`: generated QA answers and hallucination-labelled JSONL files.
- `models/*`: locally trained hallucination detector checkpoints.
- Hydra run directories and logs for script executions.

Large generated datasets, model checkpoints, and local run outputs are not intended
to be edited by hand.

## Development

Run the test suite:

```bash
make test
```

Run linting, formatting, and type checks:

```bash
make check
```

Serve the documentation locally:

```bash
make docs
```

## Project Structure

- `config/`: Hydra configuration.
- `src/factuality_eval/`: reusable dataset generation, model generation, training,
  prompt, and hallucination-detection utilities.
- `src/scripts/`: executable workflows.
- `src/prompts/`: language-specific QA prompt templates.
- `tests/`: automated tests.
- `research/`: background notes and literature references.
- `models/`: local model checkpoints.
