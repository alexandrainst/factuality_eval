# Factuality Evaluation — Research Notes

Background reading and design notes that inform the `factuality_eval` project.
These are working notes, not project documentation; for usage docs see the top-level
[`README.md`](../README.md).

## Contents

- [Motivation](#motivation)
- [Research goals](#research-goals)
- [Challenges](#challenges)
- [Overall strategies](#overall-strategies)
- [Evaluation tools](#evaluation-tools)
- [Evaluation datasets](#evaluation-datasets)
- [Benchmarks](#benchmarks)
- [Further reading](#further-reading)

## Motivation

LLMs do not know what they do not know. They sometimes overestimate their own
capabilities and confidently emit unknown information, producing false responses.
A related open question is *the art of saying "I don't know"* — calibrating
abstention so the model declines when it should, without abstaining so often
that abstention becomes a cheap way to game any factuality metric.

## Research goals

We are less interested in *how factual* an LLM is in absolute terms, and more
interested in *whether it can be trusted*. The distinction matters:

- We are not trying to identify the LLM most likely to win at Jeopardy.
- We *are* trying to identify the LLM that would lie or cheat at Jeopardy.
- At the same time, the metric must not reward an LLM that always answers
  "I don't know."

Concrete questions guiding the work:

- What do we want to measure?
- Which methods will we use?
- Which datasets are relevant?

## Challenges

- **Metric bias.** Different studies use different datasets and metrics, which
  makes it hard to compare factuality across models — or to compare the
  effectiveness of different factuality-enhancement methods — under the same
  conditions. See Augenstein et al., *"Factuality challenges in the era of
  large language models and opportunities for fact-checking"*, Nature Machine
  Intelligence 6(8), 852–863, 2024.
- **Dependence on high-quality annotated evidence datasets.**

## Overall strategies

- **LLM-as-a-judge.** Using LLMs themselves as evaluators. Often nearly as
  good as humans.
- **Retrieval-augmented generation (RAG).**
- **Human evaluation.**

## Evaluation tools

| Tool | Paper | Authors | Affiliation | Year | Code | Languages | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenFactCheck | OpenFactCheck: A Unified Framework for Factuality Evaluation of LLMs | Iqbal, Wang, Wang, Georgiev, Geng, Gurevych, Nakov | MBZUAI | 2024-08 | [mbzuai-nlp/openfactcheck](https://github.com/mbzuai-nlp/openfactcheck) | English, Urdu | Three modules — `ResponseEval` (per-document fact-check), `LLMEval` (overall factuality of an LLM), `CheckerEval` (evaluate fact-checkers). Ships two datasets: [FactQA](https://raw.githubusercontent.com/hasaniqbal777/OpenFactCheck/main/src/openfactcheck/templates/llm/questions.csv) (6,480 questions) and [FactBench](https://raw.githubusercontent.com/hasaniqbal777/OpenFactCheck/main/src/openfactcheck/templates/factchecker/claims.jsonl) (4,507 claims). |
| Loki | Loki: An Open-Source Tool for Fact Verification | Li, Han, Wang, Wang, Wang, Xing, … Baldwin | LibrAI; MBZUAI; Monash; Univ. of Melbourne | 2024-10 | [Libr-AI/OpenFactVerification](https://github.com/Libr-AI/OpenFactVerification) | Multilingual | Hosted at <https://loki.librai.tech/>. |
| FactScore | — | — | — | — | — | — | To investigate. |
| SelfCheckGPT | — | — | — | — | — | — | To investigate. |
| LongForm SAFE | Long-form factuality in large language models | — | — | — | — | — | To investigate. |
| Perplexity fact-checker | — | — | — | — | — | — | Not open source. |

## Evaluation datasets

Candidate datasets to investigate further:

- Snowball
- SelfAware
- FreshQA
- FacTool
- FELM
- Factcheck-Bench
- FactScore-Bio

## Benchmarks

| Benchmark | Description |
| --- | --- |
| LLM-AGGREFACT | Human annotations. |
| ReaLMistake | Binary error detection. |
| **LEAF Fact-check Score** | Ratio of factually supported sentences to total response. |
| Knowledge F1 | Overlap between human-used and model-used knowledge. |
| Preservation score | How much original content remains intact after hallucination correction. |

## Further reading

Surveys and adjacent papers worth tracking:

- *Hallucination to Truth: A Review of Fact-Checking and Factuality Evaluation
  in Large Language Models.* Rahman et al., 2025-08. (United International
  University; Daffodil International University; Charles Darwin University.)
- *Survey on Factuality in Large Language Models.* ACM, 2024.
  <https://dl.acm.org/doi/10.1145/3742420>
- *Factuality of Large Language Models: A Survey.* <https://arxiv.org/abs/2402.02420>
- *Trustworthiness in Retrieval-Augmented Generation Systems: A Survey.*
  <https://arxiv.org/abs/2409.10102>
- *FactTest: Factuality Testing in Large Language Models with Finite-Sample
  and Distribution-Free Guarantees.* <https://arxiv.org/abs/2411.02603>
- *SciTrust: Evaluating the Trustworthiness of Large Language Models for
  Science.* <https://ieeexplore.ieee.org/document/10820709>
- *WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge
  Conflicts from Wikipedia.* <https://arxiv.org/abs/2406.13805>
- *Identifying Factual Inconsistencies in Summaries: Grounding LLM Inference
  via Task Taxonomy.* <https://arxiv.org/abs/2402.12821>
- *Factual Consistency Evaluation of Summarization in the Era of Large
  Language Models.* <https://arxiv.org/abs/2402.13758>
- *FENICE: Factuality Evaluation of Summarization based on Natural Language
  Inference and Claim Extraction.* <https://arxiv.org/abs/2403.02270>
- *SIFiD: Reassess Summary Factual Inconsistency Detection with LLM.*
  <https://arxiv.org/abs/2403.07557>
- *TofuEval: Evaluating Hallucinations of LLMs on Topic-Focused Dialogue
  Summarization.* <https://arxiv.org/abs/2402.13249>
- *FactPICO: Factuality Evaluation for Plain Language Summarization of
  Medical Evidence.* <https://arxiv.org/abs/2402.11456>
- *TrustScore: Reference-Free Evaluation of LLM Response Trustworthiness.*
  <https://arxiv.org/abs/2402.12545>
