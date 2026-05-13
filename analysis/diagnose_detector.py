"""Diagnose and cheap-wins evaluation of the token-level hallucination detector.

Operates on the first 260 samples of data/final/ground_truth_evaluation_dataset.jsonl
that have been hand-annotated with `llm_hallucinated_parts`.

Outputs:
  analysis/detector_diagnosis.md    - human-readable report
  analysis/detector_diagnosis.json  - raw numbers
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Callable

DATA_PATH = "/workspace/data/final/ground_truth_evaluation_dataset.jsonl"
OUT_MD = "/workspace/analysis/detector_diagnosis.md"
OUT_JSON = "/workspace/analysis/detector_diagnosis.json"
N_ANNOTATED = 260


def load_samples(path: str, n: int) -> list[dict]:
    """Load up to ``n`` JSONL samples from ``path``."""
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            out.append(json.loads(line))
    return out


def build_char_mask(s: str, parts: list[str]) -> list[bool]:
    """Return a per-character mask marking characters in ``s`` covered by ``parts``."""
    mask = [False] * len(s)
    for p in parts:
        if not p:
            continue
        idx = s.find(p)
        while idx != -1:
            for k in range(idx, idx + len(p)):
                if k < len(mask):
                    mask[k] = True
            idx = s.find(p, idx + 1)
    return mask


def confusion(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Compute confusion-matrix-derived metrics from raw counts."""
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    acc = (tp + tn) / total if total else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        total=total,
        precision=prec,
        recall=rec,
        specificity=spec,
        npv=npv,
        fpr=fpr,
        fnr=fnr,
        accuracy=acc,
        f1=f1,
    )


def is_punct_or_eos(tok: str) -> bool:
    """Return True if the token is punctuation, whitespace, or ``<eos>``."""
    if tok == "<eos>":
        return True
    s = tok.strip()
    if not s:
        return True
    return all(not c.isalnum() for c in s)


def tokenize_loose(s: str) -> list[str]:
    """Loosely tokenize ``s`` into lowercase word tokens."""
    return re.findall(r"\w+", s.lower())


def question_echo_overlap(answer: str, question: str) -> float:
    """Return the fraction of answer tokens that also appear in the question."""
    q_toks = set(tokenize_loose(question))
    a_toks = tokenize_loose(answer)
    if not a_toks or not q_toks:
        return 0.0
    overlap = sum(1 for t in a_toks if t in q_toks)
    return overlap / len(a_toks)


def question_prefix_match_len(answer: str, question: str) -> int:
    """Return number of leading chars of answer that match question (case-folded)."""
    a = answer.lstrip()
    q = question.strip()
    if not a or not q:
        return 0
    n = 0
    al, ql = a.lower(), q.lower()
    for i in range(min(len(al), len(ql))):
        if al[i] == ql[i]:
            n += 1
        else:
            break
    return n


def context_str(ctx: list[str] | str) -> str:
    """Join list-of-strings context into a single newline-separated string."""
    if isinstance(ctx, list):
        return "\n".join(ctx)
    return str(ctx)


def token_in_context(tok: str, ctx: str) -> bool:
    """Return True if a non-trivial token surface form appears in ``ctx``."""
    s = tok.strip()
    if len(s) < 3:
        return False
    return s.lower() in ctx.lower()


def token_in_question(tok: str, question: str) -> bool:
    """Return True if a non-trivial token surface form appears in ``question``."""
    s = tok.strip()
    if len(s) < 3:
        return False
    return s.lower() in question.lower()


def categorize_token(tok: str, ctx: str, question: str) -> str:
    """Assign a heuristic category to a token based on its surface form and context."""
    s = tok.strip()
    if not s:
        return "whitespace"
    if s == "<eos>":
        return "eos"
    if all(not c.isalnum() for c in s):
        return "punct"
    # Numbers/dates
    if re.fullmatch(r"\d+([.,]\d+)?", s) or re.search(r"\d", s):
        if token_in_context(tok, ctx):
            return "number_in_context"
        return "number_not_in_context"
    # Capitalized → likely entity
    if s[0].isupper():
        if token_in_context(tok, ctx):
            return "entity_in_context"
        return "entity_not_in_context"
    # Function/short words
    if len(s) <= 3:
        return "short_word"
    if token_in_context(tok, ctx):
        return "word_in_context"
    return "word_not_in_context"


# --- Load -----------------------------------------------------------------
samples = load_samples(DATA_PATH, N_ANNOTATED)
print(f"Loaded {len(samples)} samples")


# --- Per-token records ----------------------------------------------------
records = []  # one per non-meta token

for sample_idx, o in enumerate(samples):
    answer = o["answer"]
    parts = o.get("llm_hallucinated_parts", [])
    question = o.get("question", "")
    ctx = context_str(o.get("context", []))
    tokens = o["tokens"]

    joined = "".join(t["token"] for t in tokens)
    mask = build_char_mask(joined, parts)

    sample_gt = bool(parts)
    q_echo = question_echo_overlap(answer, question)
    q_prefix_chars = question_prefix_match_len(answer, question)
    has_q_echo_prefix = q_prefix_chars >= max(15, int(0.5 * len(question.strip())))
    ans_len_chars = len(answer)

    pos = 0
    for t in tokens:
        tt = t["token"]
        s_ = pos
        e_ = pos + len(tt)
        pos = e_

        if tt == "<eos>":
            continue

        gt = any(mask[k] for k in range(s_, e_))
        pred_label = bool(t.get("pred", 0))
        prob = float(t.get("prob", 0.0))
        category = categorize_token(tt, ctx, question)
        in_ctx = token_in_context(tt, ctx)
        in_q = token_in_question(tt, question)

        records.append(
            dict(
                sample_idx=sample_idx,
                token=tt,
                pred=pred_label,
                prob=prob,
                gt=gt,
                sample_gt=sample_gt,
                category=category,
                in_context=in_ctx,
                in_question=in_q,
                sample_qecho=q_echo,
                sample_qprefix_match=has_q_echo_prefix,
                sample_ans_len=ans_len_chars,
            )
        )

print(f"Total tokens (non-eos): {len(records)}")


# --- §1 Diagnostics -------------------------------------------------------


def confusion_for(records: list[dict], pred_key: str = "pred") -> dict:
    """Compute confusion metrics across ``records`` using ``pred_key`` as prediction."""
    tp = fp = fn = tn = 0
    for r in records:
        p = r[pred_key]
        g = r["gt"]
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif (not p) and g:
            fn += 1
        else:
            tn += 1
    return confusion(tp, fp, fn, tn)


def confusion_for_threshold(records: list[dict], thr: float) -> dict:
    """Compute confusion metrics by thresholding the ``prob`` field at ``thr``."""
    tp = fp = fn = tn = 0
    for r in records:
        p = r["prob"] >= thr
        g = r["gt"]
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif (not p) and g:
            fn += 1
        else:
            tn += 1
    return confusion(tp, fp, fn, tn)


# --- Probability calibration ---------------------------------------------
# Distribution of prob in TP/FP/FN/TN
prob_buckets: dict[str, list[float]] = {"TP": [], "FP": [], "FN": [], "TN": []}
for r in records:
    if r["pred"] and r["gt"]:
        prob_buckets["TP"].append(r["prob"])
    elif r["pred"] and not r["gt"]:
        prob_buckets["FP"].append(r["prob"])
    elif (not r["pred"]) and r["gt"]:
        prob_buckets["FN"].append(r["prob"])
    else:
        prob_buckets["TN"].append(r["prob"])


def stats(xs: list[float]) -> dict:
    """Compute n/mean/median/p10/p90/min/max for a numeric sequence."""
    if not xs:
        return dict(n=0)
    xs_sorted = sorted(xs)
    n = len(xs)
    mean = sum(xs) / n
    median = xs_sorted[n // 2]
    p10 = xs_sorted[max(0, int(0.1 * n) - 1)]
    p90 = xs_sorted[min(n - 1, int(0.9 * n))]
    return dict(
        n=n,
        mean=mean,
        median=median,
        p10=p10,
        p90=p90,
        min=xs_sorted[0],
        max=xs_sorted[-1],
    )


prob_stats = {k: stats(v) for k, v in prob_buckets.items()}


# --- Threshold sweep ------------------------------------------------------
thresholds = [round(x * 0.05, 2) for x in range(1, 20)]  # 0.05..0.95
sweep = []
for thr in thresholds:
    c = confusion_for_threshold(records, thr)
    sweep.append({"threshold": thr, **c})

# Best F1 threshold
best_f1 = max(sweep, key=lambda r: r["f1"])
# Best F1 with recall >= 0.5
recall_floor = [r for r in sweep if r["recall"] >= 0.5]
best_f1_recallfloor = max(recall_floor, key=lambda r: r["f1"]) if recall_floor else None


# --- Per-category breakdown ----------------------------------------------
cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: dict(tp=0, fp=0, fn=0, tn=0))
for r in records:
    c = cat_stats[r["category"]]
    if r["pred"] and r["gt"]:
        c["tp"] += 1
    elif r["pred"] and not r["gt"]:
        c["fp"] += 1
    elif (not r["pred"]) and r["gt"]:
        c["fn"] += 1
    else:
        c["tn"] += 1

cat_metrics = {}
for cat, d in cat_stats.items():
    cat_metrics[cat] = {**d, **confusion(d["tp"], d["fp"], d["fn"], d["tn"])}


# --- Stratified by sample type -------------------------------------------
def conf_subset(
    filter_fn: Callable[[dict], bool], pred_key: str = "pred"
) -> tuple[int, dict]:
    """Return (n_tokens, confusion) for the records selected by ``filter_fn``."""
    sub = [r for r in records if filter_fn(r)]
    return len(sub), confusion_for(sub, pred_key=pred_key)


strat = {}
strat["sample_gt_halluc"] = conf_subset(lambda r: r["sample_gt"])
strat["sample_gt_faithful"] = conf_subset(lambda r: not r["sample_gt"])
strat["question_echo_prefix"] = conf_subset(lambda r: r["sample_qprefix_match"])
strat["no_question_echo_prefix"] = conf_subset(lambda r: not r["sample_qprefix_match"])
strat["high_q_overlap_>=0.5"] = conf_subset(lambda r: r["sample_qecho"] >= 0.5)
strat["short_answer_<=60c"] = conf_subset(lambda r: r["sample_ans_len"] <= 60)
strat["long_answer_>60c"] = conf_subset(lambda r: r["sample_ans_len"] > 60)
strat["token_in_context"] = conf_subset(lambda r: r["in_context"])
strat["token_not_in_context"] = conf_subset(lambda r: not r["in_context"])
strat["token_in_question"] = conf_subset(lambda r: r["in_question"])


# --- §2 Cheap wins -------------------------------------------------------


def apply_rules(
    records: list[dict],
    *,
    threshold: float | None = None,
    kill_qecho_prefix: bool = False,
    kill_token_in_question: bool = False,
    kill_token_in_context: bool = False,
    drop_punct: bool = True,
    smooth_isolated: bool = False,
) -> dict:
    """Compute new pred per record according to rules. Return new confusion dict."""
    # First pass: assign initial pred
    preds = []
    for r in records:
        if drop_punct and r["category"] in ("punct", "whitespace", "eos"):
            preds.append(False)  # always faithful
            continue
        if threshold is not None:
            p = r["prob"] >= threshold
        else:
            p = r["pred"]

        if kill_qecho_prefix and r["sample_qprefix_match"]:
            # Override entire question-echo-prefix sample to faithful at token level
            p = False
        if kill_token_in_question and r["in_question"]:
            p = False
        if kill_token_in_context and r["in_context"]:
            p = False
        preds.append(p)

    if smooth_isolated:
        # Group by sample, flip isolated single-token flags
        new_preds = preds[:]
        # Build per-sample index
        sample_to_idx = defaultdict(list)
        for i, r in enumerate(records):
            sample_to_idx[r["sample_idx"]].append(i)
        for s_idx, idxs in sample_to_idx.items():
            for j, i in enumerate(idxs):
                if not preds[i]:
                    continue
                left = preds[idxs[j - 1]] if j > 0 else False
                right = preds[idxs[j + 1]] if j < len(idxs) - 1 else False
                if not left and not right:
                    new_preds[i] = False
        preds = new_preds

    tp = fp = fn = tn = 0
    for p, r in zip(preds, records):
        g = r["gt"]
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif (not p) and g:
            fn += 1
        else:
            tn += 1
    return confusion(tp, fp, fn, tn)


baseline = confusion_for(records, "pred")

cheap_wins = {}
cheap_wins["00_baseline_pred=1"] = baseline
cheap_wins["01_drop_punct_only"] = apply_rules(records)
cheap_wins["02_thr_best_f1"] = apply_rules(records, threshold=best_f1["threshold"])
if best_f1_recallfloor:
    cheap_wins["03_thr_best_f1_recall>=0.5"] = apply_rules(
        records, threshold=best_f1_recallfloor["threshold"]
    )
cheap_wins["04_kill_qecho_prefix"] = apply_rules(records, kill_qecho_prefix=True)
cheap_wins["05_kill_token_in_question"] = apply_rules(
    records, kill_token_in_question=True
)
cheap_wins["06_kill_token_in_context"] = apply_rules(
    records, kill_token_in_context=True
)
cheap_wins["07_qecho+inquestion"] = apply_rules(
    records, kill_qecho_prefix=True, kill_token_in_question=True
)
cheap_wins["08_qecho+inquestion+smooth"] = apply_rules(
    records, kill_qecho_prefix=True, kill_token_in_question=True, smooth_isolated=True
)
cheap_wins["09_thr+qecho+inquestion"] = apply_rules(
    records,
    threshold=best_f1["threshold"],
    kill_qecho_prefix=True,
    kill_token_in_question=True,
)
cheap_wins["10_thr+qecho+inquestion+smooth"] = apply_rules(
    records,
    threshold=best_f1["threshold"],
    kill_qecho_prefix=True,
    kill_token_in_question=True,
    smooth_isolated=True,
)

# --- Sample-level metrics (with cheap-win combo applied) -----------------


def sample_level_from_records(
    records: list[dict], rule_fn: Callable[[dict], bool]
) -> dict:
    """rule_fn(r) -> bool token-level pred. Sample positive if any token positive."""
    by_sample: dict[int, dict[str, bool]] = defaultdict(
        lambda: dict(any_pred=False, gt=False)
    )
    for r in records:
        s = by_sample[r["sample_idx"]]
        if rule_fn(r):
            s["any_pred"] = True
        if r["sample_gt"]:
            s["gt"] = True
    tp = fp = fn = tn = 0
    for v in by_sample.values():
        if v["any_pred"] and v["gt"]:
            tp += 1
        elif v["any_pred"] and not v["gt"]:
            fp += 1
        elif (not v["any_pred"]) and v["gt"]:
            fn += 1
        else:
            tn += 1
    return confusion(tp, fp, fn, tn)


def make_rule(
    threshold: float | None = None, kill_qecho: bool = False, kill_inq: bool = False
) -> Callable[[dict], bool]:
    """Build a token-level prediction rule with optional thresholding and overrides."""

    def rule(r: dict) -> bool:
        if r["category"] in ("punct", "whitespace", "eos"):
            return False
        if kill_qecho and r["sample_qprefix_match"]:
            return False
        if kill_inq and r["in_question"]:
            return False
        if threshold is not None:
            return r["prob"] >= threshold
        return r["pred"]

    return rule


sample_level = {
    "baseline": sample_level_from_records(records, make_rule()),
    "thr_best_f1": sample_level_from_records(
        records, make_rule(threshold=best_f1["threshold"])
    ),
    "qecho+inq": sample_level_from_records(
        records, make_rule(kill_qecho=True, kill_inq=True)
    ),
    "combined": sample_level_from_records(
        records,
        make_rule(threshold=best_f1["threshold"], kill_qecho=True, kill_inq=True),
    ),
}


# --- Save raw JSON -------------------------------------------------------
out = dict(
    n_samples=len(samples),
    n_tokens=len(records),
    baseline_token=baseline,
    prob_stats=prob_stats,
    threshold_sweep=sweep,
    best_f1=best_f1,
    best_f1_recall_floor_0_5=best_f1_recallfloor,
    category_metrics=cat_metrics,
    stratified={k: dict(n_tokens=v[0], **v[1]) for k, v in strat.items()},
    cheap_wins_token=cheap_wins,
    sample_level=sample_level,
)

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"Wrote {OUT_JSON}")


# --- Render markdown report ---------------------------------------------


def fmt_conf(c: dict) -> str:
    """Format a confusion dict as a single-line summary string."""
    return (
        f"TP={c['tp']:5d} FP={c['fp']:5d} FN={c['fn']:5d} TN={c['tn']:5d} | "
        f"P={c['precision']:.3f} R={c['recall']:.3f} Spec={c['specificity']:.3f} "
        f"F1={c['f1']:.3f} Acc={c['accuracy']:.3f}"
    )


lines = []
lines.append("# Detector Diagnosis & Cheap-Wins Report")
lines.append("")
lines.append(
    f"Evaluated on {len(samples)} hand-annotated samples ({len(records)} non-eos tokens)."
)
lines.append("")

lines.append("## Baseline (token-level, model `pred` field as-is)")
lines.append("")
lines.append("```")
lines.append(fmt_conf(baseline))
lines.append("```")
lines.append("")

lines.append("## Probability calibration (per-bucket distribution of `prob`)")
lines.append("")
lines.append("| Bucket | n | mean | median | p10 | p90 |")
lines.append("|--------|---|------|--------|-----|-----|")
for k in ["TP", "FP", "FN", "TN"]:
    s = prob_stats[k]
    if s["n"] == 0:
        lines.append(f"| {k} | 0 | - | - | - | - |")
    else:
        lines.append(
            f"| {k} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | {s['p10']:.3f} | {s['p90']:.3f} |"
        )
lines.append("")
lines.append(
    "Reading: if FP probabilities cluster low and TP probabilities cluster high, threshold tuning helps; if they overlap heavily, threshold tuning is limited."
)
lines.append("")

lines.append("## Threshold sweep on `prob`")
lines.append("")
lines.append(
    "| Threshold | TP | FP | FN | TN | Precision | Recall | Specificity | F1 |"
)
lines.append(
    "|-----------|-----|-----|-----|-----|-----------|--------|-------------|----|"
)
for r in sweep:
    lines.append(
        f"| {r['threshold']:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['specificity']:.3f} | {r['f1']:.3f} |"
    )
lines.append("")
lines.append(
    f"**Best F1 threshold:** {best_f1['threshold']:.2f} → F1 {best_f1['f1']:.3f} (P {best_f1['precision']:.3f}, R {best_f1['recall']:.3f})"
)
if best_f1_recallfloor:
    lines.append(
        f"**Best F1 with recall ≥ 0.5:** thr {best_f1_recallfloor['threshold']:.2f} → F1 {best_f1_recallfloor['f1']:.3f} (P {best_f1_recallfloor['precision']:.3f}, R {best_f1_recallfloor['recall']:.3f})"
    )
lines.append("")

lines.append("## Per-category token breakdown")
lines.append("")
lines.append(
    "Categorization heuristic: punct/eos/whitespace, numbers, entities (capitalized), short words, regular words; cross-tabulated by whether the token text appears in the context."
)
lines.append("")
lines.append("| Category | n | TP | FP | FN | TN | Prec | Rec | Spec | F1 |")
lines.append("|----------|---|-----|-----|-----|-----|------|-----|------|----|")
for cat in sorted(cat_metrics, key=lambda k: -cat_metrics[k]["total"]):
    m = cat_metrics[cat]
    lines.append(
        f"| {cat} | {m['total']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['specificity']:.3f} | {m['f1']:.3f} |"
    )
lines.append("")

lines.append("## Stratified analysis")
lines.append("")
lines.append("| Stratum | n_tokens | Prec | Rec | Spec | F1 | FPR |")
lines.append("|---------|----------|------|-----|------|----|-----|")
for k, (n, c) in strat.items():
    lines.append(
        f"| {k} | {n} | {c['precision']:.3f} | {c['recall']:.3f} | {c['specificity']:.3f} | {c['f1']:.3f} | {c['fpr']:.3f} |"
    )
lines.append("")

lines.append("## Cheap-win interventions (token-level)")
lines.append("")
lines.append(
    "Each row applies a transformation on top of the model `pred` (or `prob` for thresholding) and recomputes confusion vs. ground truth."
)
lines.append("")
lines.append("| Rule | TP | FP | FN | TN | Prec | Rec | Spec | F1 |")
lines.append("|------|-----|-----|-----|-----|------|-----|------|----|")
for k, c in cheap_wins.items():
    lines.append(
        f"| {k} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['specificity']:.3f} | {c['f1']:.3f} |"
    )
lines.append("")

lines.append("## Sample-level metrics under cheap wins")
lines.append("")
lines.append("| Variant | TP | FP | FN | TN | Prec | Rec | Spec | F1 |")
lines.append("|---------|-----|-----|-----|-----|------|-----|------|----|")
for k, c in sample_level.items():
    lines.append(
        f"| {k} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['specificity']:.3f} | {c['f1']:.3f} |"
    )
lines.append("")

# Top-line takeaways
lines.append("## Key findings")
lines.append("")
b_f1 = baseline["f1"]
best_token_rule = max(cheap_wins.items(), key=lambda kv: kv[1]["f1"])
lines.append(
    f"- Baseline token F1 = {b_f1:.3f}. Best cheap-win combination: **{best_token_rule[0]}** → F1 {best_token_rule[1]['f1']:.3f} (P {best_token_rule[1]['precision']:.3f}, R {best_token_rule[1]['recall']:.3f})."
)
lines.append(
    f"- Optimal probability threshold (F1) is {best_f1['threshold']:.2f}, not 0.5; F1 lifts from {baseline['f1']:.3f} → {best_f1['f1']:.3f}."
)
# Rate of FPs in question-echo-prefix samples
qe_n, qe_c = strat["question_echo_prefix"]
nqe_n, nqe_c = strat["no_question_echo_prefix"]
lines.append(
    f"- Question-echo-prefix samples ({qe_n} tokens): FPR {qe_c['fpr']:.3f} vs. {nqe_c['fpr']:.3f} on non-echo samples — {'higher' if qe_c['fpr'] > nqe_c['fpr'] else 'lower'} FPR there."
)
ic_n, ic_c = strat["token_in_context"]
nc_n, nc_c = strat["token_not_in_context"]
lines.append(
    f"- Tokens whose surface form appears in context: FPR {ic_c['fpr']:.3f} vs. {nc_c['fpr']:.3f} for not-in-context — confirms context-copy whitelist intuition."
)

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {OUT_MD}")
