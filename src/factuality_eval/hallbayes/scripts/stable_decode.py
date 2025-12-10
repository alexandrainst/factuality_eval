#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""stable_decode.py — Inflection‑Aware Generation using the **Completions API**.
=======================================================================================
Example Usage
------------
 python stable_decode.py
  --prompt "Evidence:
- Study A: Treatment improved outcomes by 52%
- Study B: Treatment improved outcomes by 48%
- Study C: No significant improvement observed
- Meta-analysis: Results are inconclusive
Question: Is the treatment effective?"
  --model gpt-3.5-turbo-instruct
  --use-pmd
  --max-tokens 100

Requirements
------------
- Python 3.9+
- openai >= 1.0.0
- numpy, scipy

Recommended models
------------------
Use a Completions‑capable model that supports `logprobs` (e.g., `gpt-3.5-turbo-instruct`).
Newer chat‑only models (e.g., many `gpt-4o*`) may not support Completions.

Usage (CLI)
-----------
python stable_decode.py \
  --prompt "Evidence:\n- Paris is in France\n\nQuestion: Capital of France?" \
  --model gpt-3.5-turbo-instruct \
  --use-pmd \
  --max-tokens 128

Authors
-------
https://github.com/leochlon/hallbayes
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as scipy_entropy

try:
    from openai import OpenAI  # pip install openai>=1.0.0
except Exception:
    OpenAI = None


# ====================================================================================
# Numeric utilities
# ====================================================================================

EPS = 1e-12


def _safe_log(x: float) -> float:
    return math.log(max(float(x), EPS))


def shannon_entropy(probs: Dict[str, float]) -> float:
    if not probs:
        return 0.0
    arr = np.array(list(probs.values()), dtype=float)
    s = float(arr.sum())
    if s <= 0:
        return 0.0
    arr = arr / s
    return float(scipy_entropy(arr, base=np.e))


def top2_margin(probs: Dict[str, float]) -> float:
    if not probs:
        return 0.0
    vals = sorted(probs.values(), reverse=True)
    if len(vals) == 1:
        return vals[0]
    return float(vals[0] - vals[1])


def js_divergence_over_dicts(dlist: List[Dict[str, float]]) -> float:
    if len(dlist) < 2:
        return 0.0
    keys = set().union(*[set(d.keys()) for d in dlist])
    mats = []
    for d in dlist:
        vec = np.array([d.get(k, 0.0) for k in keys], dtype=float)
        s = float(vec.sum())
        vec = vec / s if s > 0 else np.ones(len(keys), dtype=float) / len(keys)
        mats.append(vec)
    js_vals = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            js = float(
                jensenshannon(mats[i], mats[j]) ** 2
            )  # square to get JSD in nats
            js_vals.append(js)
    return float(sum(js_vals) / max(1, len(js_vals)))


# ====================================================================================
# Evidence parsing / permutation helpers (permute **only** the exchangeable evidence)
# ====================================================================================

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _extract_bullets(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        m = _BULLET.match(ln)
        if m:
            out.append(_BULLET.sub("", ln).strip())
    return [x for x in out if x]


def _split_user_text(user_text: str) -> Tuple[str, List[str], str]:
    """Heuristically split `user_text` into:
      header (instructions/context), evidence items (bullets or 'Evidence:' lines), footer (question).
    If no evidence block is found, returns (user_text, [], "").
    """
    lines = user_text.splitlines()
    if not lines:
        return "", [], ""

    start = None
    for i, ln in enumerate(lines):
        if (
            ln.strip()
            .lower()
            .startswith(("evidence:", "facts:", "context:", "examples:"))
        ):
            start = i
            break
        if _BULLET.match(ln):
            start = i
            break

    if start is None:
        return user_text.strip(), [], ""

    evid_lines = []
    j = start
    while j < len(lines) and (
        lines[j].strip() and (_BULLET.match(lines[j]) or j == start)
    ):
        evid_lines.append(lines[j])
        j += 1

    header = "\n".join(lines[:start]).strip()
    footer = "\n".join(lines[j:]).strip()

    items = _extract_bullets(evid_lines)
    if not items:
        # handle single‑line "Evidence: X"
        items = [ln.split(":", 1)[1].strip() for ln in evid_lines if ":" in ln]

    return header, items, footer


def _join_user_text(header: str, items: List[str], footer: str) -> str:
    parts: List[str] = []
    if header:
        parts.append(header)
    if items:
        parts.append("Evidence:")
        parts.extend([f"- {it}" for it in items])
    if footer:
        parts.append(footer)
    return "\n".join(parts).strip()


# ====================================================================================
# Completions backend
# ====================================================================================


class OpenAICompletionsBackend:
    """Minimal wrapper over `client.completions.create` for one‑token probing."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-instruct",
        api_key: Optional[str] = None,
        request_timeout: float = 60.0,
    ) -> None:
        if OpenAI is None:
            raise ImportError("Install `openai>=1.0.0` and set OPENAI_API_KEY.")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.request_timeout = float(request_timeout)

    @staticmethod
    def build_prompt(user_text: str, prefix: str) -> str:
        """Compose a simple Q&A style prompt that *naturally* continues after `prefix`."""
        return f"{user_text.strip()}\n\nAnswer:\n{prefix}"

    def next_token_distribution(
        self,
        user_text: str,
        prefix: str,
        top_logprobs: int = 20,
        temperature: float = 0.0,
        **kwargs,
    ) -> Tuple[str, Dict[str, float]]:
        """Request exactly one next token after `prefix`. Returns (chosen_token, {token: prob})."""
        prompt = self.build_prompt(user_text, prefix)
        params = dict(
            model=self.model,
            prompt=prompt,
            max_tokens=1,
            temperature=temperature,
            logprobs=top_logprobs,
        )
        params.update(kwargs)
        # The SDK uses `request_timeout` instead of `timeout`
        if "timeout" in params:
            params["request_timeout"] = params.pop("timeout")

        resp = self.client.completions.create(**params)

        chosen_token = ""
        probs: Dict[str, float] = {}

        try:
            # Chosen token is the **first** generated token
            chosen_token = resp.choices[0].logprobs.tokens[0]
            top_map = resp.choices[0].logprobs.top_logprobs[0]  # dict: token -> logprob
            for tok, lp in (top_map or {}).items():
                if tok is None or lp is None:
                    continue
                probs[str(tok)] = float(math.exp(float(lp)))
            # normalize
            Z = sum(probs.values())
            if Z > 0:
                for k in list(probs.keys()):
                    probs[k] /= Z
        except Exception:
            # Fallback: rely on text if logprobs missing (some models)
            try:
                chosen_token = resp.choices[0].text or ""
            except Exception:
                chosen_token = ""

        return chosen_token, probs


# ====================================================================================
# Inflection point detection
# ====================================================================================


@dataclass
class InflectionConfig:
    """Thresholds for deciding when to engage PMD."""

    tau_margin: float = 0.12  # trigger if (p1 - p2) < tau_margin
    tau_entropy: float = 2.0  # trigger if H(probs) > tau_entropy (nats)
    use_margin: bool = True
    use_entropy: bool = True
    # Optional JSD gate if you pre‑probe permutations before committing a token
    use_jsd: bool = False
    tau_jsd: float = 0.15


class InflectionDetector:
    """Detect high‑entropy (unstable) next‑token steps."""

    def __init__(self, config: Optional[InflectionConfig] = None) -> None:
        self.cfg = config or InflectionConfig()
        self.history: List[Dict] = []

    def is_inflection_point(
        self,
        base_probs: Dict[str, float],
        permutation_probs: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        metrics: Dict[str, float] = {}
        triggers: List[str] = []

        if self.cfg.use_entropy:
            H = shannon_entropy(base_probs)
            metrics["entropy"] = H
            if H > self.cfg.tau_entropy:
                triggers.append("entropy")

        if self.cfg.use_margin:
            m = top2_margin(base_probs)
            metrics["top2_margin"] = m
            if m < self.cfg.tau_margin:
                triggers.append("margin")

        if self.cfg.use_jsd and permutation_probs:
            jsd = js_divergence_over_dicts(permutation_probs)
            metrics["jsd"] = jsd
            if jsd > self.cfg.tau_jsd:
                triggers.append("jsd")

        metrics["triggers"] = ",".join(triggers)
        is_inflection = len(triggers) > 0

        self.history.append(
            {"timestamp": time.time(), "is_inflection": is_inflection, **metrics}
        )
        return is_inflection, metrics


# ====================================================================================
# PMD: Permutation‑Mixture Decoding (using Completions)
# ====================================================================================


@dataclass
class PMDConfig:
    k_permutations: int = 12
    aggregation: str = "expected_log"  # "expected_log" | "average" | "geometric"
    seed: Optional[int] = 42


class PMDStabilizer:
    """Applies permutation‑mixture decoding: shuffle only the **exchangeable evidence items**,
    probe next‑token distribution for each permutation (temperature=0), and aggregate
    distributions to pick a robust next token (default: highest expected log‑probability).
    """

    def __init__(
        self, backend: OpenAICompletionsBackend, config: Optional[PMDConfig] = None
    ) -> None:
        self.backend = backend
        self.cfg = config or PMDConfig()
        self.rng = random.Random(self.cfg.seed)

    def _aggregate(self, dists: List[Dict[str, float]]) -> Dict[str, float]:
        if not dists:
            return {}
        all_tokens = set().union(*[set(d.keys()) for d in dists])

        if self.cfg.aggregation == "expected_log":
            agg: Dict[str, float] = {}
            for t in all_tokens:
                lps = []
                for d in dists:
                    p = d.get(t, EPS)
                    lps.append(_safe_log(p))
                agg[t] = float(math.exp(sum(lps) / len(lps)))
        elif self.cfg.aggregation == "geometric":
            agg = {}
            for t in all_tokens:
                vals = [d.get(t, EPS) for d in dists]
                agg[t] = float(math.exp(sum(_safe_log(v) for v in vals) / len(vals)))
        else:  # "average"
            agg = {}
            for t in all_tokens:
                vals = [d.get(t, 0.0) for d in dists]
                agg[t] = float(sum(vals) / len(vals))

        Z = sum(agg.values())
        if Z > 0:
            for k in list(agg.keys()):
                agg[k] /= Z
        return agg

    def compute_pmd_distribution(
        self, user_text: str, assistant_prefix: str = "", **model_kwargs
    ) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict]:
        """Aggregate distributions across k permutations of the evidence block."""
        header, items, footer = _split_user_text(user_text)

        def assemble(h: str, its: List[str], f: str) -> str:
            return _join_user_text(h, its, f)

        # If there is nothing to permute, just probe once
        if not items:
            base_tok, base_probs = self.backend.next_token_distribution(
                user_text=user_text, prefix=assistant_prefix, **model_kwargs
            )
            return (
                base_probs,
                [base_probs],
                {
                    "k_permutations": 1,
                    "num_unique_tokens": len(base_probs),
                    "aggregation": self.cfg.aggregation,
                },
            )

        k = max(1, int(self.cfg.k_permutations))
        dists: List[Dict[str, float]] = []

        # Original order
        orig_text = assemble(header, items, footer)
        _, p = self.backend.next_token_distribution(
            user_text=orig_text, prefix=assistant_prefix, **model_kwargs
        )
        dists.append(p)

        # k-1 random permutations
        for _ in range(k - 1):
            perm = items[:]
            self.rng.shuffle(perm)
            perm_text = assemble(header, perm, footer)
            _, dp = self.backend.next_token_distribution(
                user_text=perm_text, prefix=assistant_prefix, **model_kwargs
            )
            dists.append(dp)

        agg = self._aggregate(dists)
        meta = {
            "k_permutations": k,
            "aggregation": self.cfg.aggregation,
            "num_unique_tokens": len(agg),
        }
        return agg, dists, meta


# ====================================================================================
# Token‑by‑token decoding with inflection & PMD (Completions)
# ====================================================================================


class InflectionAwareDecoder:
    """Token‑by‑token decoder that (a) detects inflection points from next‑token logprobs and
    (b) applies PMD only at those steps, using the **Completions** API.
    """

    def __init__(
        self,
        backend: OpenAICompletionsBackend,
        inflection_config: Optional[InflectionConfig] = None,
        pmd_config: Optional[PMDConfig] = None,
    ) -> None:
        self.backend = backend
        self.inflection = InflectionDetector(inflection_config or InflectionConfig())
        self.pmd = PMDStabilizer(backend, pmd_config or PMDConfig())

    def generate_with_inflection_control(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        use_pmd: bool = True,
        stop: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, List[Dict]]:
        """Sequential decoding with inflection detection and PMD.

        Inputs:
            prompt: the user text (may include an "Evidence:" block and a question).

        Returns:
            generated_text, trace (list of dicts with step metrics)
        """
        generated = ""
        trace: List[Dict] = []
        same_token_streak = 0
        last_tok = None

        for step in range(1, max_tokens + 1):
            tok, base_probs = self.backend.next_token_distribution(
                user_text=prompt, prefix=generated, temperature=temperature, **kwargs
            )

            if not tok:
                break

            is_inflect, metrics = self.inflection.is_inflection_point(base_probs)

            chosen = tok
            stage = "base"
            jsd = 0.0

            if use_pmd and is_inflect:
                agg, permlist, meta = self.pmd.compute_pmd_distribution(
                    user_text=prompt,
                    assistant_prefix=generated,
                    temperature=0.0,
                    **kwargs,
                )
                if agg:
                    chosen = max(agg.items(), key=lambda kv: kv[1])[0]
                    stage = "pmd"
                    jsd = js_divergence_over_dicts(permlist)
                    metrics = {
                        "entropy": shannon_entropy(agg),
                        "top2_margin": top2_margin(agg),
                        "triggers": metrics.get("triggers", ""),
                    }

            generated += chosen

            # optional stop sequence (string match on the running text)
            if stop and generated.endswith(stop):
                break

            # simple loop guard against pathological repetition of an identical token
            if chosen == last_tok:
                same_token_streak += 1
            else:
                same_token_streak = 0
            last_tok = chosen
            if same_token_streak >= 50:
                break

            trace.append(
                {
                    "step": step,
                    "stage": stage,
                    "token": chosen,
                    "entropy": round(metrics.get("entropy", 0.0), 4),
                    "top2_margin": round(metrics.get("top2_margin", 0.0), 4),
                    "jsd_perms": round(jsd, 4),
                    "triggers": metrics.get("triggers", ""),
                }
            )

        return generated, trace


# ====================================================================================
# CLI
# ====================================================================================


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Inflection‑Aware generation (Completions API)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt; include an 'Evidence:' block if you have exchangeable items.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-instruct",
        help="Completions‑capable model that supports logprobs.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key (or env OPENAI_API_KEY).",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--use-pmd",
        action="store_true",
        help="Apply permutation‑mixture decoding at inflection points.",
    )
    parser.add_argument(
        "--stop", type=str, default=None, help="Optional stop sequence (string suffix)."
    )
    args = parser.parse_args()

    backend = OpenAICompletionsBackend(model=args.model, api_key=args.api_key)
    decoder = InflectionAwareDecoder(backend)

    text, trace = decoder.generate_with_inflection_control(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_pmd=args.use_pmd,
        stop=args.stop,
    )
    print("\n=== OUTPUT ===\n")
    print(text)
    print("\n=== TRACE (first 30 steps) ===\n")
    for row in trace[:30]:
        print(row)


if __name__ == "__main__":
    _cli()
