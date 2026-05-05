# Detector Diagnosis: Findings & Cheap-Wins Summary

Evaluated on the first 260 hand-annotated samples of `data/final/ground_truth_evaluation_dataset.jsonl` (5,438 non-eos tokens).

## Headline findings

| Metric | Baseline | Best cheap win |
|---|---|---|
| Token F1 | 0.303 | 0.310 (`thr=0.60` or `kill_token_in_question`) |
| Token Precision | 0.253 | 0.296–0.334 |
| Token Recall | 0.377 | 0.265–0.358 |
| Sample F1 | 0.519 | 0.530 (combined rules) |

**Cheap wins recover only ~1 F1 point at the token level.** The detector's failure modes are not fixable by post-hoc rules — they require retraining.

## Why threshold tuning barely helps

The probability distributions are bimodal but FP and TP buckets *both* sit at high probability:

| Bucket | n | mean prob | median | p10 | p90 |
|---|---|---|---|---|---|
| TP | 279 | 0.879 | 0.936 | 0.648 | 0.998 |
| **FP** | **822** | **0.809** | **0.845** | **0.565** | **0.989** |
| FN | 461 | 0.085 | 0.016 | 0.000 | 0.301 |
| TN | 3876 | 0.070 | 0.008 | 0.000 | 0.270 |

When the detector flags a token, it does so with ~85% confidence whether it's right or wrong. A threshold sweep over 0.05–0.95 moves F1 only from 0.279 → 0.310. Optimal F1 threshold is **0.60** (F1 0.310, P 0.274, R 0.358); requiring recall ≥ 0.5 forces threshold down to 0.15 (F1 0.289, P 0.202, R 0.508).

## Per-category failure analysis

Tokens were categorized by surface form and context membership:

| Category | n | Prec | Rec | Spec | F1 |
|---|---|---|---|---|---|
| short_word | 2152 | 0.214 | 0.342 | 0.814 | 0.263 |
| word_in_context | 849 | 0.310 | 0.344 | 0.864 | 0.326 |
| **number_not_in_context** | **653** | **0.302** | **0.621** | 0.664 | **0.406** |
| punct | 555 | 0.175 | 0.130 | 0.934 | 0.149 |
| **entity_in_context** | **520** | 0.240 | **0.162** | 0.915 | 0.194 |
| entity_not_in_context | 259 | 0.320 | 0.457 | 0.848 | 0.376 |
| whitespace | 237 | 0.136 | 0.429 | 0.736 | 0.207 |
| **word_not_in_context** | 213 | 0.352 | **0.731** | 0.813 | **0.475** |

Reading:

- **Strongest category: invented numbers** (`number_not_in_context`, recall 0.62) — detector reliably catches fabricated dates/figures.
- **Weakest category: entity mis-attribution** (`entity_in_context`, recall 0.16) — when the answer assigns a context fact to the wrong entity (e.g., sample 217 mixing two birthdates from the same article), the detector almost always misses. This is the dominant failure family.
- **Punctuation tokens still get flagged** (33 FPs on 555 punct tokens) — pure noise the model should never produce.

## Corrected hypotheses

Several "obvious" cheap-win heuristics turned out to be net-negative on this data:

1. **Question-echo prefix samples are *not* over-flagged.** I expected high FPR there; actual FPR is 0.038 vs. 0.180 elsewhere. The detector correctly stays faithful on echoed-question text. The post-hoc rule `kill_qecho_prefix` loses 28 TPs without major FP gains.
2. **Context-copy whitelist** (`token_in_context`) modestly reduces FPR (0.137 vs. 0.201) but recall drops 0.377 → 0.236 because hallucinations frequently *do* reuse context tokens with new attributions. Net F1 -0.05.
3. **Span smoothing** (drop isolated single-token flags) drops FPs but loses more TPs proportionally. Net F1 -0.02.
4. **Only `kill_token_in_question`** gives a single-rule F1 win (0.303 → 0.310), because the detector has terrible precision on question-substring tokens (P 0.122 in that stratum).

## Cheap-win interventions evaluated (token-level)

| Rule | Prec | Rec | Spec | F1 |
|------|------|-----|------|-----|
| 00 baseline `pred=1` | 0.253 | 0.377 | 0.825 | 0.303 |
| 01 drop punct only | 0.264 | 0.355 | 0.844 | 0.303 |
| 02 thr = 0.60 | 0.285 | 0.338 | 0.866 | 0.309 |
| 03 thr = 0.15 (recall ≥ 0.5) | 0.214 | 0.477 | 0.725 | 0.296 |
| 04 kill qecho prefix | 0.256 | 0.339 | 0.845 | 0.292 |
| 05 kill token-in-question | 0.296 | 0.326 | 0.878 | 0.310 |
| 06 kill token-in-context | 0.270 | 0.236 | 0.899 | 0.252 |
| 07 qecho + inquestion | 0.287 | 0.309 | 0.879 | 0.298 |
| 08 qecho + inquestion + smooth | 0.310 | 0.269 | 0.906 | 0.288 |
| 09 thr + qecho + inquestion | 0.307 | 0.299 | 0.894 | 0.303 |
| 10 thr + qecho + inquestion + smooth | 0.334 | 0.265 | 0.917 | 0.296 |

## Sample-level metrics under cheap wins

| Variant | Prec | Rec | Spec | F1 |
|---|---|---|---|---|
| baseline | 0.408 | 0.714 | 0.503 | 0.519 |
| thr_best_f1 (0.60) | 0.421 | 0.702 | 0.537 | 0.527 |
| qecho + inq | 0.423 | 0.690 | 0.549 | 0.525 |
| **combined** | **0.435** | 0.679 | **0.577** | **0.530** |

Sample-level shows a small but real lift (F1 +0.011) from the combined rule set.

## Recommended cheap-win configuration

Apply at inference time:

1. Threshold = 0.60 (F1 ≥ baseline; modest precision gain).
2. Force `pred = False` for punctuation, whitespace, and `<eos>` tokens.
3. Force `pred = False` when the token's stripped form is a substring of the question.

Net token-level F1 stays ~baseline; sample-level F1 lifts 0.519 → 0.530. **This is the ceiling of post-hoc improvement on the current detector.**

## Implication for next steps

The detector's failure modes are structural: high-confidence FPs at prob > 0.8 and near-zero recall on entity mis-attributions within context. These live in the model's representations, not in its decision threshold or output post-processing.

Real leverage is in:

- **§3 Training-data fixes** — especially adding entity-substitution and mis-attribution examples (mis-assigning a context fact to a different entity in the same passage). This is the largest gap.
- **§4 Modeling fixes** — loss reweighting (13.6% / 86.4% class imbalance), stronger backbone, full-context (non-truncated) encoding, span-level head.

## Reproducibility

- Script: `analysis/diagnose_detector.py`
- Auto-generated full report: `analysis/detector_diagnosis.md`
- Raw numbers: `analysis/detector_diagnosis.json`
