# Detector Diagnosis & Cheap-Wins Report

Evaluated on 260 hand-annotated samples (5438 non-eos tokens).

## Baseline (token-level, model `pred` field as-is)

```
TP=  279 FP=  822 FN=  461 TN= 3876 | P=0.253 R=0.377 Spec=0.825 F1=0.303 Acc=0.764
```

## Probability calibration (per-bucket distribution of `prob`)

| Bucket | n | mean | median | p10 | p90 |
|--------|---|------|--------|-----|-----|
| TP | 279 | 0.879 | 0.936 | 0.648 | 0.998 |
| FP | 822 | 0.809 | 0.845 | 0.565 | 0.989 |
| FN | 461 | 0.085 | 0.016 | 0.000 | 0.301 |
| TN | 3876 | 0.070 | 0.008 | 0.000 | 0.270 |

Reading: if FP probabilities cluster low and TP probabilities cluster high, threshold tuning helps; if they overlap heavily, threshold tuning is limited.

## Threshold sweep on `prob`

| Threshold | TP | FP | FN | TN | Precision | Recall | Specificity | F1 |
|-----------|-----|-----|-----|-----|-----------|--------|-------------|----|
| 0.05 | 446 | 2010 | 294 | 2688 | 0.182 | 0.603 | 0.572 | 0.279 |
| 0.10 | 402 | 1686 | 338 | 3012 | 0.193 | 0.543 | 0.641 | 0.284 |
| 0.15 | 376 | 1487 | 364 | 3211 | 0.202 | 0.508 | 0.683 | 0.289 |
| 0.20 | 356 | 1354 | 384 | 3344 | 0.208 | 0.481 | 0.712 | 0.291 |
| 0.25 | 342 | 1243 | 398 | 3455 | 0.216 | 0.462 | 0.735 | 0.294 |
| 0.30 | 327 | 1145 | 413 | 3553 | 0.222 | 0.442 | 0.756 | 0.296 |
| 0.35 | 314 | 1040 | 426 | 3658 | 0.232 | 0.424 | 0.779 | 0.300 |
| 0.40 | 302 | 959 | 438 | 3739 | 0.239 | 0.408 | 0.796 | 0.302 |
| 0.45 | 292 | 889 | 448 | 3809 | 0.247 | 0.395 | 0.811 | 0.304 |
| 0.50 | 279 | 822 | 461 | 3876 | 0.253 | 0.377 | 0.825 | 0.303 |
| 0.55 | 271 | 749 | 469 | 3949 | 0.266 | 0.366 | 0.841 | 0.308 |
| 0.60 | 265 | 702 | 475 | 3996 | 0.274 | 0.358 | 0.851 | 0.310 |
| 0.65 | 252 | 641 | 488 | 4057 | 0.282 | 0.341 | 0.864 | 0.309 |
| 0.70 | 237 | 586 | 503 | 4112 | 0.288 | 0.320 | 0.875 | 0.303 |
| 0.75 | 224 | 526 | 516 | 4172 | 0.299 | 0.303 | 0.888 | 0.301 |
| 0.80 | 212 | 464 | 528 | 4234 | 0.314 | 0.286 | 0.901 | 0.299 |
| 0.85 | 193 | 406 | 547 | 4292 | 0.322 | 0.261 | 0.914 | 0.288 |
| 0.90 | 164 | 333 | 576 | 4365 | 0.330 | 0.222 | 0.929 | 0.265 |
| 0.95 | 131 | 222 | 609 | 4476 | 0.371 | 0.177 | 0.953 | 0.240 |

**Best F1 threshold:** 0.60 → F1 0.310 (P 0.274, R 0.358)
**Best F1 with recall ≥ 0.5:** thr 0.15 → F1 0.289 (P 0.202, R 0.508)

## Per-category token breakdown

Categorization heuristic: punct/eos/whitespace, numbers, entities (capitalized), short words, regular words; cross-tabulated by whether the token text appears in the context.

| Category | n | TP | FP | FN | TN | Prec | Rec | Spec | F1 |
|----------|---|-----|-----|-----|-----|------|-----|------|----|
| short_word | 2152 | 95 | 349 | 183 | 1525 | 0.214 | 0.342 | 0.814 | 0.263 |
| word_in_context | 849 | 44 | 98 | 84 | 623 | 0.310 | 0.344 | 0.864 | 0.326 |
| number_not_in_context | 653 | 77 | 178 | 47 | 351 | 0.302 | 0.621 | 0.664 | 0.406 |
| punct | 555 | 7 | 33 | 47 | 468 | 0.175 | 0.130 | 0.934 | 0.149 |
| entity_in_context | 520 | 12 | 38 | 62 | 408 | 0.240 | 0.162 | 0.915 | 0.194 |
| entity_not_in_context | 259 | 16 | 34 | 19 | 190 | 0.320 | 0.457 | 0.848 | 0.376 |
| whitespace | 237 | 9 | 57 | 12 | 159 | 0.136 | 0.429 | 0.736 | 0.207 |
| word_not_in_context | 213 | 19 | 35 | 7 | 152 | 0.352 | 0.731 | 0.813 | 0.475 |

## Stratified analysis

| Stratum | n_tokens | Prec | Rec | Spec | F1 | FPR |
|---------|----------|------|-----|------|----|-----|
| sample_gt_halluc | 1756 | 0.626 | 0.377 | 0.836 | 0.470 | 0.164 |
| sample_gt_faithful | 3682 | 0.000 | 0.000 | 0.822 | 0.000 | 0.178 |
| question_echo_prefix | 197 | 0.667 | 0.293 | 0.962 | 0.407 | 0.038 |
| no_question_echo_prefix | 5241 | 0.247 | 0.382 | 0.820 | 0.300 | 0.180 |
| high_q_overlap_>=0.5 | 2604 | 0.257 | 0.444 | 0.854 | 0.326 | 0.146 |
| short_answer_<=60c | 1860 | 0.213 | 0.337 | 0.808 | 0.261 | 0.192 |
| long_answer_>60c | 3578 | 0.276 | 0.397 | 0.834 | 0.326 | 0.166 |
| token_in_context | 2200 | 0.254 | 0.285 | 0.863 | 0.268 | 0.137 |
| token_not_in_context | 3238 | 0.253 | 0.443 | 0.799 | 0.322 | 0.201 |
| token_in_question | 1461 | 0.122 | 0.206 | 0.883 | 0.153 | 0.117 |

## Cheap-win interventions (token-level)

Each row applies a transformation on top of the model `pred` (or `prob` for thresholding) and recomputes confusion vs. ground truth.

| Rule | TP | FP | FN | TN | Prec | Rec | Spec | F1 |
|------|-----|-----|-----|-----|------|-----|------|----|
| 00_baseline_pred=1 | 279 | 822 | 461 | 3876 | 0.253 | 0.377 | 0.825 | 0.303 |
| 01_drop_punct_only | 263 | 732 | 477 | 3966 | 0.264 | 0.355 | 0.844 | 0.303 |
| 02_thr_best_f1 | 250 | 628 | 490 | 4070 | 0.285 | 0.338 | 0.866 | 0.309 |
| 03_thr_best_f1_recall>=0.5 | 353 | 1294 | 387 | 3404 | 0.214 | 0.477 | 0.725 | 0.296 |
| 04_kill_qecho_prefix | 251 | 728 | 489 | 3970 | 0.256 | 0.339 | 0.845 | 0.292 |
| 05_kill_token_in_question | 241 | 574 | 499 | 4124 | 0.296 | 0.326 | 0.878 | 0.310 |
| 06_kill_token_in_context | 175 | 473 | 565 | 4225 | 0.270 | 0.236 | 0.899 | 0.252 |
| 07_qecho+inquestion | 229 | 570 | 511 | 4128 | 0.287 | 0.309 | 0.879 | 0.298 |
| 08_qecho+inquestion+smooth | 199 | 442 | 541 | 4256 | 0.310 | 0.269 | 0.906 | 0.288 |
| 09_thr+qecho+inquestion | 221 | 498 | 519 | 4200 | 0.307 | 0.299 | 0.894 | 0.303 |
| 10_thr+qecho+inquestion+smooth | 196 | 390 | 544 | 4308 | 0.334 | 0.265 | 0.917 | 0.296 |

## Sample-level metrics under cheap wins

| Variant | TP | FP | FN | TN | Prec | Rec | Spec | F1 |
|---------|-----|-----|-----|-----|------|-----|------|----|
| baseline | 60 | 87 | 24 | 88 | 0.408 | 0.714 | 0.503 | 0.519 |
| thr_best_f1 | 59 | 81 | 25 | 94 | 0.421 | 0.702 | 0.537 | 0.527 |
| qecho+inq | 58 | 79 | 26 | 96 | 0.423 | 0.690 | 0.549 | 0.525 |
| combined | 57 | 74 | 27 | 101 | 0.435 | 0.679 | 0.577 | 0.530 |

## Key findings

- Baseline token F1 = 0.303. Best cheap-win combination: **05_kill_token_in_question** → F1 0.310 (P 0.296, R 0.326).
- Optimal probability threshold (F1) is 0.60, not 0.5; F1 lifts from 0.303 → 0.310.
- Question-echo-prefix samples (197 tokens): FPR 0.038 vs. 0.180 on non-echo samples — lower FPR there.
- Tokens whose surface form appears in context: FPR 0.137 vs. 0.201 for not-in-context — confirms context-copy whitelist intuition.
