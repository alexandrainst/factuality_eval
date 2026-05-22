# Human-grounded hallucination evaluation

Rows annotated: **185**.
Bootstrap 95% CIs over rows (iters = 1000).

## Token-level

| System | Precision | Recall | F1 (95% CI) |
| --- | --- | --- | --- |
| detector | 0.371 | 0.564 | 0.448 (0.240–0.604) |
| llm | 0.206 | 0.887 | 0.334 (0.209–0.446) |

## Span-level (RAGTruth char-overlap)

| System | Precision | Recall | F1 (95% CI) | Jaccard (mean) |
| --- | --- | --- | --- | --- |
| detector | 0.338 | 0.284 | 0.309 (0.170–0.414) | 0.076 |
| llm | 0.263 | 0.901 | 0.407 (0.265–0.525) | 0.204 |

## Example-level (any-token-hallucinated)

| System | Precision | Recall | F1 (95% CI) | AUROC |
| --- | --- | --- | --- | --- |
| detector | 0.353 | 0.375 | 0.364 (0.209–0.508) | 0.754 |
| llm | 0.378 | 0.969 | 0.544 (0.426–0.648) | n/a |

## Agreement with human (Cohen's kappa, token level)

- detector vs human: **0.420**
- LLM-judge vs human: **0.288**

## Data quality checks

- Rows with empty detector token output (all rows): **33**
- Rows with empty detector token output (annotated): **4**
- Rows dropped due to token/label length mismatch during scoring: **0**

## Source breakdown (annotated rows)

| Source | Rows | Token F1 (detector) | Token F1 (llm) | Span F1 (detector) | Span F1 (llm) | Example F1 (detector) | Example F1 (llm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multiwikiqa | 85 | 0.119 | 0.538 | 0.130 | 0.528 | 0.182 | 0.870 |
| ragtruth | 100 | 0.495 | 0.300 | 0.334 | 0.388 | 0.545 | 0.324 |
