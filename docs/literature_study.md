# Evaluation tools

| Paper title | Authors | Affiliation | Published | Code | Summary | Comments | Languages | Tool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenFactCheck: A Unified Framework for Factuality Evaluation of LLMs | Iqbal, H., Wang, Y., Wang, M., Georgiev, G., Geng, J., Gurevych, I., & Nakov, P.  | MBZUAI (Mohamed bin Zayed University of Artificial Intelligence) | 2024-08 | https://github.com/mbzuai-nlp/openfactcheck | OpenFactCheck has 3 modules: \n\n- RESPONSEEVAL: customize fact-checking system and assess the factuality of all claims in an input document\n- LLMEVAL: assess overall factuality of an LLM\n- CHECKEREVAL: evaluate automatic fact-checking systems | They created two datasets: [FactQA](https://raw.githubusercontent.com/hasaniqbal777/OpenFactCheck/main/src/openfactcheck/templates/llm/questions.csv) (6480 questions) and [FactBench](https://raw.githubusercontent.com/hasaniqbal777/OpenFactCheck/main/src/openfactcheck/templates/factchecker/claims.jsonl) (4507 claims).  | English, Urdu | OpenFactCheck |
| Loki: An Open-Source Tool for Fact Verification | Li, H., Han, X., Wang, H., Wang, Y., Wang, M., Xing, R., ... & Baldwin | LibrAI, MBZUAI, Monash University, The University of Melbourne | 2024-10 | https://github.com/Libr-AI/OpenFactVerification |  | https://loki.librai.tech/ | Multilingual | Loki |
|  |  |  |  |  |  |  |  | FactScore |
|  |  |  |  |  |  |  |  | SelfCheckGPT |
| Long-form factuality in large language models |  |  |  |  |  |  |  | LongForm SAFE |
|  |  |  |  |  |  | Not open-source |  | Perplexity fact checker |
| Hallucination to Truth: A Review of Fact-Checking and Factuality\n\nEvaluation in Large Language Models | Rahman, S. S., Islam, M. A., Alam, M. M., Zeba, M., Rahman, M. A., Chowa, S. S., ... & Azam, S. | United International University (Bangladesh),  Daffodil International University (Bangladesh), Charles Darwin University (Australia) | 2025-08 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |

# Evaluation datasets

| Paper title |  |  |  |  |  |  |  | Dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  | Snowball |
|  |  |  |  |  |  |  |  | SelfAware |
|  |  |  |  |  |  |  |  | FreshQA |
|  |  |  |  |  |  |  |  | FacTool |
|  |  |  |  |  |  |  |  | FELM |
|  |  |  |  |  |  |  |  | Factcheck-Bench |
|  |  |  |  |  |  |  |  | FactScore-Bio |

## Benchmarks

| Paper title |  |  |  |  |  |  | Description | Benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  | Human annotations | LLM-AGGREFACT |
|  |  |  |  |  |  |  | Binary error detection | ReaLMistake |
|  |  |  |  |  |  |  | Compute the ratio of factually supported sentences to the total response | **LEAF Fact-check Score** |
|  |  |  |  |  |  |  | Measures the overlap between human-used and model-used knowledge | Knowledge F1 |
|  |  |  |  |  |  |  | evaluates how much original content remains intact after hallucination correction | Presevation score |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  | 0 |  |  |

# Papers from Dan

[FactTest: Factuality Testing in Large Language Models with Finite-Sample and Distribution-Free Guarantees](http://arxiv.org/abs/2411.02603 "http://arxiv.org/abs/2411.02603")

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

# Why are LLMs not factual?

- LLMs do not know what they do not know, sometimes overestimate their capacities and confidently output unknown information, leading to false responses.

# The art of saying "I don't know"

(Check paper from Friday on multiple questionnaires, where it sometimes says it doesn't know the answer.)

# Difficulties with evaluating factuality

- Studies assessing language models’ factuality or evaluating whether the methods are effective to mitigate model hallucinations use different datasets and metrics.
    - This makes it difficult to compare, in the same conditions, the factuality of different models as well as to compare the effectiveness of different factuality enhancement approaches.

# Research goals

- Hvad vil vi måle

- Hvilke metoder vil vi bruge

- Hvilke dataset er relevante

We don't really care about how factual the LLM's are, but we want to know can we trust them. This means that we're not trying to test which LLM is most likeliy to win in Jeapordy, however we are more interested in testing which one would lie or cheat in Jeapordy. But at the same time, we do not want a metric that will be biased in a way that the LLM always answers I dont know, and gets a good score for that.

# Challenges

Metric bias (**I. Augenstein, T. Baldwin, M. Cha, T. Chakraborty, G. L. Ciampaglia, D. Corney, R. DiResta, E. Ferrara, S. Hale, A. Halevy et al., “Factuality challenges in the era of large language models and opportunities for factchecking,” Nature Machine Intelligence, vol. 6, no. 8, pp. 852–863, 2024.)**

Dependence on high quality annotated evidence datasets

# Overall strategies

Using LLMs themselves as evaluators (LLM-as-a-judge). Often closely just as good as humans.

RAG

Human evaluation
