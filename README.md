# Factuality Evaluation of LLMs

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-48%25-orange.svg)](https://github.com/alexandrainst/factuality_eval/tree/main/tests)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/factuality_eval)
[![License](https://img.shields.io/github/license/alexandrainst/factuality_eval)](https://github.com/alexandrainst/factuality_eval/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/factuality_eval)](https://github.com/alexandrainst/factuality_eval/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/factuality_eval/blob/main/CODE_OF_CONDUCT.md)

## Literature Review

### Evaluation Tools

| Paper title | Authors | Affiliation | Published | Code | Summary | Comments | Languages | Tool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenFactCheck: A Unified Framework for Factuality Evaluation of LLMs | Iqbal, H., Wang, Y., Wang, M., Georgiev, G., Geng, J., Gurevych, I., & Nakov, P.  | MBZUAI (Mohamed bin Zayed University of Artificial Intelligence) | 2024-08 | <https://github.com/mbzuai-nlp/openfactcheck> | OpenFactCheck has 3 modules: \n\n- RESPONSEEVAL: customize fact-checking system and assess the factuality of all claims in an input document\n- LLMEVAL: assess overall factuality of an LLM\n- CHECKEREVAL: evaluate automatic fact-checking systems | They created two datasets: [FactQA](https://raw.githubusercontent.com/hasaniqbal777/OpenFactCheck/main/src/openfactcheck/templates/llm/questions.csv) (6480 questions) and [FactBench](https://raw.githubusercontent.com/hasaniqbal777/OpenFactCheck/main/src/openfactcheck/templates/factchecker/claims.jsonl) (4507 claims).  | English, Urdu | OpenFactCheck |
| Loki: An Open-Source Tool for Fact Verification | Li, H., Han, X., Wang, H., Wang, Y., Wang, M., Xing, R., ... & Baldwin | LibrAI, MBZUAI, Monash University, The University of Melbourne | 2024-10 | <https://github.com/Libr-AI/OpenFactVerification> |  | <https://loki.librai.tech/> | Multilingual | Loki |
|  |  |  |  |  |  |  |  | FactScore |
|  |  |  |  | <https://www.comet.com/site/blog/selfcheckgpt-for-llm-evaluation/> |  | A blackbox hallucination detection method that relies solely on stochastic sampling of model responses. The core intuition of their method is that factually accurate responses are typically consistent and frequent, whereas hallucinated outputs tend to vary and contradict each other. |  | SelfCheckGPT |
| Long-form factuality in large language models |  |  |  |  |  |  |  | LongForm SAFE |
|  |  |  |  |  |  | Not open-source |  | Perplexity fact checker |
| Hallucination to Truth: A Review of Fact-Checking and Factuality\n\nEvaluation in Large Language Models | Rahman, S. S., Islam, M. A., Alam, M. M., Zeba, M., Rahman, M. A., Chowa, S. S., ... & Azam, S. | United International University (Bangladesh),  Daffodil International University (Bangladesh), Charles Darwin University (Australia) | 2025-08 |  |  |  |  |  |
| FACTTEST: FACTUALITY TESTING IN LARGE  LANGUAGE MODELS WITH FINITE-SAMPLE AND  DISTRIBUTION-FREE GUARANTEES | Fan Nie1 Xiaotian Hou2 Shuhang Lin2 James Zou1 Huaxiu Yao3 Linjun Zhang | Stanford University, 2Rutgers University, 3UNC-Chapel Hill | 2024-11 |  | Used to "finetune" models to not answer if the answer is likely to be false. |  |  |  |
| Seq vs Seq: An Open Suite of Paired Encoders and Decoders |  |  |  |  | TinyLettuce is used to have a dataset consisting of hallunications and correct responses.\n\n*"**The Problem**: Training robust hallucination detection models requires large datasets of both correct and hallucinated responses. Manually creating such datasets is expensive and time-consuming.*\n\n***Our Solution****: LettuceDetect's synthetic data generation pipeline can generate realistic hallucinations from factual content."* |  |  |  |
| Hallucination Risk Calculator & Prompt Re‑engineering Toolkit (OpenAI‑only) |  |  |  | <https://hassana.io/readme.html> | Calculate the risk of hallucination based on a prompt.\n\nBasically just entropy calculation?\n\nProblem is, which prompts should we supply? |  |  |  |
| (Im)possibility of Automated Hallucination Detection in\n\nLarge Language Models |  |  |  |  | Not possible if trained only on correct samples (duh) |  |  |  |
| HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models |  |  |  | <https://github.com/RUCAIBox/HaluEval> | Many citations |  |  |  |

### Evaluation Benchmarks and Datasets

| Paper title | Authors | Affiliation | Date | Code | Summary/comments | Dataset |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  | Snowball |
|  |  |  |  |  |  | SelfAware |
|  |  |  |  |  |  | FreshQA |
|  |  |  |  |  |  | FacTool |
|  |  |  |  |  |  | FELM |
|  |  |  |  |  |  | Factcheck-Bench |
|  |  |  |  |  |  | FactScore-Bio |
|  |  |  |  |  | Human annotations | LLM-AGGREFACT |
|  |  |  |  |  | Binary error detection | ReaLMistake |
|  |  |  |  |  | Compute the ratio of factually supported sentences to the total response | **LEAF Fact-check Score** |
|  |  |  |  |  | Measures the overlap between human-used and model-used knowledge | Knowledge F1 |
|  |  |  |  |  | evaluates how much original content remains intact after hallucination correction | Presevation score |
|  |  |  |  |  | Human annotations | LLM-AGGREFACT |
|  |  |  |  |  | Binary error detection | ReaLMistake |
|  |  |  |  |  | Compute the ratio of factually supported sentences to the total response | **LEAF Fact-check Score** |
|  |  |  |  |  | Measures the overlap between human-used and model-used knowledge | Knowledge F1 |
|  |  |  |  |  | evaluates how much original content remains intact after hallucination correction | Presevation score |
|  |  |  |  |  | HotpotQA is a  released question-answering dataset that involves multi-hop reasoning over\n\nmultiple paragraphs of information to produce an answer. A successful model must not only report\n\nanswers as yes/no or a span within the text but also identify supporting facts. | HotpotQA |
|  |  |  |  |  |  | SimpleQA |
|  |  |  |  |  | Possibly not public/open. | PersonQA |
| TRUSTSCORE: REFERENCE-FREE EVALUATION OFLLM RESPONSE TRUSTWORTHINESS | Danna Zheng, Danyang Liu, Mirella Lapata, Jeff Z. Pan | University of Edinburgh,\n\nHuawei Edinburgh Research Centre |  |  |  | TrustScore |
| Know What You Don't Know: Unanswerable Questions for SQuAD |  |  | 2018-11 | <https://rajpurkar.github.io/SQuAD-explorer/> | Many  | SQuAD |
|  |  |  |  |  | is **an automatic evaluation metric for factual precision in long-form text generation**. It uses large language models and retrieval to break down generations into atomic facts and then measure the correctness with respect to a knowledge source (like Wikipedia). | FactScore |

### Papers from Dan

[Survey on Factuality in Large Language Models](https://dl.acm.org/doi/10.1145/3742420 "https://dl.acm.org/doi/10.1145/3742420")

[Trustworthiness in Retrieval-Augmented Generation Systems: A Survey](http://arxiv.org/abs/2409.10102 "http://arxiv.org/abs/2409.10102")

[SciTrust: Evaluating the Trustworthiness of Large Language Models for Science](https://ieeexplore.ieee.org/document/10820709 "https://ieeexplore.ieee.org/document/10820709")

[WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia](http://arxiv.org/abs/2406.13805 "http://arxiv.org/abs/2406.13805")

[Identifying Factual Inconsistencies in Summaries: Grounding LLM Inference via Task Taxonomy](http://arxiv.org/abs/2402.12821 "http://arxiv.org/abs/2402.12821")

[Factual consistency evaluation of summarization in the Era of large language models](http://arxiv.org/abs/2402.13758 "http://arxiv.org/abs/2402.13758")

[FENICE: Factuality Evaluation of summarization based on Natural language Inference and Claim Extraction](http://arxiv.org/abs/2403.02270 "http://arxiv.org/abs/2403.02270")

[SIFiD: Reassess Summary Factual Inconsistency Detection with LLM](http://arxiv.org/abs/2403.07557 "http://arxiv.org/abs/2403.07557")

[TofuEval: Evaluating Hallucinations of LLMs on Topic-Focused Dialogue Summarization](http://arxiv.org/abs/2402.13249 "http://arxiv.org/abs/2402.13249")

[Factuality of Large Language Models: A Survey](http://arxiv.org/abs/2402.02420 "http://arxiv.org/abs/2402.02420")

[FactPICO: Factuality Evaluation for Plain Language Summarization of Medical Evidence](http://arxiv.org/abs/2402.11456 "http://arxiv.org/abs/2402.11456")

[TrustScore: Reference-Free Evaluation of LLM Response Trustworthiness](http://arxiv.org/abs/2402.12545 "http://arxiv.org/abs/2402.12545")

## Why are LLMs not factual?

- LLMs do not know what they do not know, sometimes overestimate their capacities and
  confidently output unknown information, leading to false responses.

## The art of saying "I don't know"

(Check paper from Friday on multiple questionnaires, where it sometimes says it doesn't
know the answer.)

## Difficulties with evaluating factuality

- Studies assessing language models’ factuality or evaluating whether the methods are
  effective to mitigate model hallucinations use different datasets and metrics.
- This makes it difficult to compare, in the same conditions, the factuality of
   different models as well as to compare the effectiveness of different factuality
   enhancement approaches.

## Research goals

- Hvad vil vi måle
- Hvilke metoder vil vi bruge
- Hvilke dataset er relevante

We don't really care about how factual the LLM's are, but we want to know can we trust
them. This means that we're not trying to test which LLM is most likeliy to win in
Jeapordy, however we are more interested in testing which one would lie or cheat in
Jeapordy. But at the same time, we do not want a metric that will be biased in a way
that the LLM always answers I dont know, and gets a good score for that.

## Challenges

Metric bias (**I. Augenstein, T. Baldwin, M. Cha, T. Chakraborty, G. L. Ciampaglia, D.
Corney, R. DiResta, E. Ferrara, S. Hale, A. Halevy et al., “Factuality challenges in the
era of large language models and opportunities for factchecking,” Nature Machine
Intelligence, vol. 6, no. 8, pp. 852–863, 2024.)**

Dependence on high quality annotated evidence datasets.

## Overall strategies

Using LLMs themselves as evaluators (LLM-as-a-judge). Often closely just as good as
humans.

RAG

Human evaluation

Detecting hallucinations in language models is challenging. There are three general
approaches:

- **Measuring token-level probability distributions** for indications that a model is
  “confused.” Though sometimes effective, these methods rely on model internals being
  accessible—which is often not the case when working with hosted LLMs.
- **Referencing external fact-verification systems**, like a database or document store.
  These methods are great for RAG-style use-cases, but they are only effective if you
  have a useful dataset and the infrastructure to use it.
- **Using LLM-as-a-judge techniques** to assess whether or not a model hallucinated.
  These techniques are becoming standard in the LLM ecosystem, but as I’ll explain
  throughout this piece, using them effectively requires a deceptive amount of work.

The problem with many LLM-as-a-Judge techniques is that they tend towards two
polarities: they are either too simple, using a basic zero-shot approach, or they are
wildly complex, involving multiple LLMs interacting via multi-turn reasoning.

## Datasets

- HotpotQA
- SimpleQA
- PersonQA (possibly not public)
- SQuAD

## Hallucination

### Definition of hallucinations

Hallucinations are a feature, not a bug. When is a LLM hallucinating, and when is it
creating?

### Hallucination theory

- Entropy measurements (need output probability distribution)
- Er der teoretisk grundlag for at man kan teste factuality?

### Hallucination detectors

- Paper: Not possible if trained only on correct samples (duh)
- SelfCheckGPT: Voting system

### SelfCheckGPT

Check for variance i output af ens model, er det meget stokastisk / random eller
konvergerer modellen mod det samme svar?
