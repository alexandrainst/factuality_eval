"""Evaluate the detector and the LLM-judge against the human gold annotations.

Reads ``data/final/ground_truth_evaluation_dataset.jsonl``, keeps rows where
``human_annotation_status == "annotated"``, then for each of two systems
(detector predictions, LLM-judge spans) reports — against the human spans:

- Token-level precision / recall / F1
- Span-level (RAGTruth char-overlap) precision / recall / F1
- Example-level precision / recall / F1 + AUROC (detector only, uses token probs)
- Cohen's kappa at token level
- Span-level Jaccard (char-set IoU)

Each F1 ships with a bootstrap 95% confidence interval (rows resampled).

Outputs ``analysis/human_evaluation.json`` and ``analysis/human_evaluation.md``.

Usage:
    uv run src/scripts/ground_truth/evaluate_against_human.py [<key>=<value> ...]
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from factuality_eval.ground_truth_eval import (
    HUMAN_STATUS_ANNOTATED,
    build_char_mask,
    read_rows,
    spans_to_token_labels,
)

logger = logging.getLogger("evaluate_against_human")

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Per-row scoring primitives
# ---------------------------------------------------------------------------


def _row_token_labels(row: dict[str, Any]) -> dict[str, list[int]]:
    """Return aligned per-token 0/1 labels for human / detector / LLM judge."""
    tokens = row["tokens"]
    human = spans_to_token_labels(tokens, row.get("human_hallucinated_parts", []))
    llm = spans_to_token_labels(tokens, row.get("llm_hallucinated_parts", []))
    detector = [int(t.get("pred", 0)) for t in tokens]
    # Skip <eos> meta token for everyone, just like spans_to_token_labels does
    # for human/llm: produce a mask of "scorable" indices.
    keep = [i for i, t in enumerate(tokens) if t["token"] != "<eos>"]
    return {
        "human": [human[i] for i in keep],
        "detector": [detector[i] for i in keep],
        "llm": [llm[i] for i in keep],
        "probs": [float(tokens[i].get("prob", 0.0)) for i in keep],
    }


def _confusion(pred: list[int], gold: list[int]) -> tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for p, g in zip(pred, gold):
        if p == 1 and g == 1:
            tp += 1
        elif p == 1 and g == 0:
            fp += 1
        elif p == 0 and g == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _char_overlap(pred_text: str, pred_spans: list[str], gold_spans: list[str]) -> tuple[int, int, int]:
    """Return (intersection chars, gold chars, pred chars) on the answer text."""
    pred_mask = build_char_mask(pred_text, pred_spans)
    gold_mask = build_char_mask(pred_text, gold_spans)
    inter = sum(1 for a, b in zip(pred_mask, gold_mask) if a and b)
    return inter, sum(gold_mask), sum(pred_mask)


def _cohen_kappa(pred: list[int], gold: list[int]) -> float:
    n = len(pred)
    if n == 0:
        return 0.0
    po = sum(1 for p, g in zip(pred, gold) if p == g) / n
    p_pred1 = sum(pred) / n
    p_gold1 = sum(gold) / n
    pe = p_pred1 * p_gold1 + (1 - p_pred1) * (1 - p_gold1)
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1 - pe)


# ---------------------------------------------------------------------------
# Aggregation across rows
# ---------------------------------------------------------------------------


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all per-system metrics over the supplied annotated rows."""
    # Token-level pooled counts.
    tok_counts = {
        sys: dict(tp=0, fp=0, fn=0, tn=0) for sys in ("detector", "llm")
    }
    # Span-level pooled char counts.
    span_counts = {sys: dict(inter=0, gold=0, pred=0) for sys in ("detector", "llm")}
    # Example-level bools.
    ex_human: list[int] = []
    ex_detector: list[int] = []
    ex_llm: list[int] = []
    ex_detector_score: list[float] = []
    # Token streams for kappa.
    pooled_human: list[int] = []
    pooled_detector: list[int] = []
    pooled_llm: list[int] = []
    # Span Jaccard.
    span_jaccard = {"detector": [], "llm": []}

    for row in rows:
        labels = _row_token_labels(row)
        h = labels["human"]
        d = labels["detector"]
        l = labels["llm"]
        probs = labels["probs"]

        pooled_human.extend(h)
        pooled_detector.extend(d)
        pooled_llm.extend(l)

        for sys, pred in (("detector", d), ("llm", l)):
            tp, fp, fn, _ = _confusion(pred, h)
            tn = len(h) - tp - fp - fn
            tok_counts[sys]["tp"] += tp
            tok_counts[sys]["fp"] += fp
            tok_counts[sys]["fn"] += fn
            tok_counts[sys]["tn"] += tn

        answer = row.get("answer", "")
        human_spans = row.get("human_hallucinated_parts", []) or []
        # Detector spans are reconstructed as contiguous positive-token runs.
        # Using one merged string across all positives can create impossible
        # substrings and severely underestimate overlap.
        detector_spans = _predicted_spans_from_labels(row["tokens"], d)
        llm_spans = row.get("llm_hallucinated_parts", []) or []

        for sys, spans in (("detector", detector_spans), ("llm", llm_spans)):
            inter, gold, pred = _char_overlap(answer, spans, human_spans)
            span_counts[sys]["inter"] += inter
            span_counts[sys]["gold"] += gold
            span_counts[sys]["pred"] += pred
            # Jaccard per row, ignoring rows where both sides are empty.
            union = gold + pred - inter
            if union > 0:
                span_jaccard[sys].append(inter / union)

        ex_human.append(1 if any(h) else 0)
        ex_detector.append(1 if any(d) else 0)
        ex_llm.append(1 if any(l) else 0)
        ex_detector_score.append(max(probs) if probs else 0.0)

    metrics: dict[str, Any] = {"n_rows": len(rows)}

    for sys in ("detector", "llm"):
        c = tok_counts[sys]
        p, r, f = _prf(c["tp"], c["fp"], c["fn"])
        metrics[f"token/{sys}"] = dict(precision=p, recall=r, f1=f, **c)

    for sys in ("detector", "llm"):
        s = span_counts[sys]
        prec = s["inter"] / s["pred"] if s["pred"] else 0.0
        rec = s["inter"] / s["gold"] if s["gold"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[f"span/{sys}"] = dict(
            precision=prec, recall=rec, f1=f1, **s,
            jaccard_mean=float(np.mean(span_jaccard[sys])) if span_jaccard[sys] else 0.0,
        )

    for sys, ex_pred in (("detector", ex_detector), ("llm", ex_llm)):
        tp, fp, fn, _ = _confusion(ex_pred, ex_human)
        p, r, f = _prf(tp, fp, fn)
        metrics[f"example/{sys}"] = dict(precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn)

    metrics["example/detector"]["auroc"] = _auroc(ex_detector_score, ex_human)

    metrics["kappa/detector_vs_human"] = _cohen_kappa(pooled_detector, pooled_human)
    metrics["kappa/llm_vs_human"] = _cohen_kappa(pooled_llm, pooled_human)

    return metrics


def _predicted_spans_from_labels(
    tokens: list[dict[str, Any]], labels: list[int]
) -> list[str]:
    """Reconstruct predicted-hallucinated spans as contiguous token runs.

    ``labels`` is expected to align with non-``<eos>`` tokens only, matching
    ``_row_token_labels``.
    """
    spans: list[str] = []
    current: list[str] = []
    keep_iter = iter(labels)
    for t in tokens:
        if t["token"] == "<eos>":
            continue
        if next(keep_iter, 0) == 1:
            current.append(t["token"])
        elif current:
            spans.append("".join(current))
            current = []
    if current:
        spans.append("".join(current))
    return spans


def _auroc(scores: list[float], labels: list[int]) -> float:
    """Standard ROC AUC (Mann-Whitney U formulation, ties get 0.5 credit)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


_F1_KEYS = (
    "token/detector",
    "token/llm",
    "span/detector",
    "span/llm",
    "example/detector",
    "example/llm",
)


def _bootstrap_f1_ci(
    rows: list[dict[str, Any]], iters: int = BOOTSTRAP_ITERS
) -> dict[str, tuple[float, float]]:
    """Bootstrap a 95% CI on each F1 metric by resampling rows."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(rows)
    if n == 0:
        return {k: (float("nan"), float("nan")) for k in _F1_KEYS}

    samples: dict[str, list[float]] = {k: [] for k in _F1_KEYS}
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        boot = [rows[i] for i in idx]
        m = _aggregate(boot)
        for k in _F1_KEYS:
            samples[k].append(m[k]["f1"])

    out: dict[str, tuple[float, float]] = {}
    for k, vals in samples.items():
        lo, hi = np.quantile(vals, [0.025, 0.975])
        out[k] = (float(lo), float(hi))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _render_md(metrics: dict[str, Any], ci: dict[str, tuple[float, float]]) -> str:
    def f1_with_ci(key: str) -> str:
        f1 = metrics[key]["f1"]
        lo, hi = ci[key]
        return f"{f1:.3f} ({lo:.3f}–{hi:.3f})"

    lines = [
        "# Human-grounded hallucination evaluation",
        "",
        f"Rows annotated: **{metrics['n_rows']}**.",
        f"Bootstrap 95% CIs over rows (iters = {BOOTSTRAP_ITERS}).",
        "",
        "## Token-level",
        "",
        "| System | Precision | Recall | F1 (95% CI) |",
        "| --- | --- | --- | --- |",
    ]
    for sys in ("detector", "llm"):
        m = metrics[f"token/{sys}"]
        lines.append(
            f"| {sys} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{f1_with_ci(f'token/{sys}')} |"
        )

    lines += [
        "",
        "## Span-level (RAGTruth char-overlap)",
        "",
        "| System | Precision | Recall | F1 (95% CI) | Jaccard (mean) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for sys in ("detector", "llm"):
        m = metrics[f"span/{sys}"]
        lines.append(
            f"| {sys} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{f1_with_ci(f'span/{sys}')} | {m['jaccard_mean']:.3f} |"
        )

    lines += [
        "",
        "## Example-level (any-token-hallucinated)",
        "",
        "| System | Precision | Recall | F1 (95% CI) | AUROC |",
        "| --- | --- | --- | --- | --- |",
    ]
    auroc = metrics["example/detector"].get("auroc", float("nan"))
    auroc_str = f"{auroc:.3f}" if not math.isnan(auroc) else "n/a"
    m = metrics["example/detector"]
    lines.append(
        f"| detector | {m['precision']:.3f} | {m['recall']:.3f} | "
        f"{f1_with_ci('example/detector')} | {auroc_str} |"
    )
    m = metrics["example/llm"]
    lines.append(
        f"| llm | {m['precision']:.3f} | {m['recall']:.3f} | "
        f"{f1_with_ci('example/llm')} | n/a |"
    )

    lines += [
        "",
        "## Agreement with human (Cohen's kappa, token level)",
        "",
        f"- detector vs human: **{metrics['kappa/detector_vs_human']:.3f}**",
        f"- LLM-judge vs human: **{metrics['kappa/llm_vs_human']:.3f}**",
        "",
    ]

    quality = metrics.get("quality", {})
    if quality:
        lines += [
            "## Data quality checks",
            "",
            f"- Rows with empty detector token output (all rows): **{quality.get('empty_tokens_total', 0)}**",
            f"- Rows with empty detector token output (annotated): **{quality.get('empty_tokens_annotated', 0)}**",
            f"- Rows dropped due to token/label length mismatch during scoring: **{quality.get('length_mismatch_annotated', 0)}**",
            "",
        ]

    by_source = metrics.get("by_source", {})
    if by_source:
        lines += [
            "## Source breakdown (annotated rows)",
            "",
            "| Source | Rows | Token F1 (detector) | Token F1 (llm) | Span F1 (detector) | Span F1 (llm) | Example F1 (detector) | Example F1 (llm) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for source, sm in sorted(by_source.items()):
            lines.append(
                "| "
                f"{source} | {sm['n_rows']} | "
                f"{sm['token/detector']['f1']:.3f} | {sm['token/llm']['f1']:.3f} | "
                f"{sm['span/detector']['f1']:.3f} | {sm['span/llm']['f1']:.3f} | "
                f"{sm['example/detector']['f1']:.3f} | {sm['example/llm']['f1']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Compute and write human-grounded evaluation metrics."""
    dataset_path = Path(config.manual_annotation.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found.")

    all_rows = read_rows(dataset_path)
    rows = [r for r in all_rows if r.get("human_annotation_status") == HUMAN_STATUS_ANNOTATED]
    logger.info(f"Using {len(rows)} of {len(all_rows)} rows (annotated only).")
    if not rows:
        raise SystemExit(
            "No annotated rows yet — run src/scripts/ground_truth/annotate_ground_truth.py first."
        )

    metrics = _aggregate(rows)
    ci = _bootstrap_f1_ci(rows)

    metrics["quality"] = {
        "empty_tokens_total": sum(1 for r in all_rows if not r.get("tokens")),
        "empty_tokens_annotated": sum(1 for r in rows if not r.get("tokens")),
        "length_mismatch_annotated": sum(
            1
            for r in rows
            if len(r.get("tokens", [])) != len(r.get("human_annotation_labels", []))
        ),
    }

    # Source-wise breakdown for diagnostic visibility while keeping the same
    # core metric definitions.
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row.get("source", "unknown")), []).append(row)
    metrics["by_source"] = {
        source: _aggregate(source_rows) for source, source_rows in by_source.items()
    }

    out_dir = Path("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "metrics": metrics,
        "f1_95ci": {k: list(v) for k, v in ci.items()},
        "bootstrap_iters": BOOTSTRAP_ITERS,
    }
    (out_dir / "human_evaluation.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "human_evaluation.md").write_text(
        _render_md(metrics, ci), encoding="utf-8"
    )
    logger.info(f"Wrote analysis/human_evaluation.{{json,md}} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
