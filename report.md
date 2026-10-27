# HybridRAG-Eval: Experiment Report

**Project:** hybrid retrieval (BM25 + vector) with cross-encoder reranking and cited answers
**Date:** October 2026
**Author:** Muhammad Faizan Farooq

## 1. Objective

Build a RAG system over at least 20 documents (PDF, DOCX, TXT) that uses hybrid retrieval and a reranking step, produces answers with source citations, and is evaluated on a set of at least 30 question-answer pairs. The evaluation must measure retrieval hit rate and answer correctness and must compare at least two chunking strategies and report which one performs better and why.

## 2. Setup

| Component | Choice |
|---|---|
| Corpus | 22 documents (10 PDF, 6 TXT, 6 DOCX), 337 pages: arXiv papers on Transformers, BERT, RAG, ResNet, LoRA, Adam, batch normalization, word2vec, dropout and DPR, plus Wikipedia articles on general ML topics |
| Chunking A: fixed | Sliding window of 150 words with 30 words of overlap, applied page by page. 1403 chunks, 136 words on average |
| Chunking B: recursive | Paragraphs split into sentences, whole sentences packed up to 150 words, last sentence carried over as overlap, words as a last resort. 1403 chunks, 126 words on average |
| Vector search | `all-MiniLM-L6-v2` embeddings, FAISS inner-product index |
| Keyword search | BM25 (Okapi) over lower-cased, stop-word-filtered, Porter-stemmed tokens |
| Fusion | Reciprocal Rank Fusion, k = 60, 20 candidates from each retriever |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 20 candidates reranked to top 5 |
| LLM | Gemini 3.5 Flash, temperature 0, answers only from the 5 retrieved chunks and cites `[source: file, page X]` |

Both strategies use the same maximum chunk size on purpose, so that the comparison isolates **where chunks are cut** rather than how large they are. Chunks never cross page boundaries so that every citation has an exact page (for TXT and DOCX files, which have no pages, a page is a block of about 3000 characters).

### Evaluation set

47 questions written for this corpus: 20 keyword-style (a rare term or number), 19 semantic (a concept described in different words) and 8 factual. Each has a ground-truth answer, a source document and a list of evidence phrases. A retrieved chunk is **relevant** when it comes from a ground-truth document and contains at least one evidence phrase. This definition does not depend on chunk boundaries, so the two strategies are scored on the same footing. A validation script confirmed that every question's source file and evidence phrase exist in the corpus (0 problems out of 47).

### Metrics

- **Hit@k:** at least one of the top-k chunks is relevant. **DocHit@k:** at least one top-k chunk comes from the correct document. **MRR@5:** mean reciprocal rank of the first relevant chunk in the top 5.
- **Correctness:** an LLM judge scores each answer 1, 0.5 or 0 against the ground truth. **Answer similarity:** cosine similarity between the answer and the ground truth embeddings.
- **Citation accuracy:** the answer cites at least one ground-truth source. **Citation grounding:** every cited (source, page) was among the retrieved chunks.

## 3. Results

### 3.1 Retrieval (47 questions)

| Chunking | Retrieval | Hit@1 | Hit@3 | Hit@5 | DocHit@5 | MRR@5 |
|---|---|---|---|---|---|---|
| fixed | Vector only | 0.511 | 0.787 | 0.830 | 0.979 | 0.644 |
| fixed | BM25 only | 0.638 | 0.787 | 0.809 | 1.000 | 0.718 |
| fixed | Hybrid (RRF) | 0.681 | 0.830 | 0.851 | 0.979 | 0.756 |
| fixed | Hybrid + Rerank | 0.681 | 0.872 | 0.915 | 0.979 | 0.782 |
| recursive | Vector only | 0.447 | 0.745 | 0.851 | 0.979 | 0.606 |
| recursive | BM25 only | 0.660 | 0.809 | 0.830 | 1.000 | 0.736 |
| recursive | Hybrid (RRF) | 0.638 | 0.809 | 0.851 | 0.979 | 0.729 |
| recursive | Hybrid + Rerank | 0.723 | 0.894 | 0.936 | 0.979 | 0.801 |

One question is worth 0.021 (1/47) on any Hit metric.

### 3.2 Hit@5 by question type, Hybrid + Rerank

| Chunking | Factual (8) | Keyword (20) | Semantic (19) |
|---|---|---|---|
| fixed | 0.875 (7/8) | 0.950 (19/20) | 0.895 (17/19) |
| recursive | 0.875 (7/8) | 1.000 (20/20) | 0.895 (17/19) |

### 3.3 Answer quality (pilot, first 5 questions only)

| Chunking | Questions graded | Correctness | Answer similarity | Citation accuracy | Citation grounding | Abstained |
|---|---|---|---|---|---|---|
| fixed | 5 | 1.000 | 0.724 | 0.800 | 1.000 | 0 |
| recursive | 5 | 1.000 | 0.738 | 0.800 | 1.000 | 0 |

All 10 pilot answers were judged fully correct, no cited page was invented (grounding 1.0), and in 1 of 5 answers per strategy none of the cited sources matched the ground-truth document (the cause was not investigated). With only 5 questions per strategy these figures show that the generation and citation path works, but they do not measure answer quality reliably (see Section 5).

## 4. Analysis

### 4.1 Which chunking strategy performs better?

With the full pipeline (Hybrid + Rerank), **recursive chunking scored higher**: Hit@5 0.936 against 0.915, Hit@1 0.723 against 0.681 and MRR@5 0.801 against 0.782. It missed 3 questions at Hit@5 and fixed chunking missed 4.

This advantage is **small and not statistically reliable**. The Hit@5 gap equals a single question. Comparing the two strategies question by question, the reciprocal rank differed on 16 questions: recursive was better on 9, fixed on 7, and the remaining 31 were tied. A sign test on those 16 discordant questions gives p ≈ 0.80, so there is no evidence that either strategy is better. The honest conclusion is: **recursive chunking was slightly ahead in this experiment, but the two strategies are practically equivalent at this corpus size and chunk size.**

The per-question table shows a pattern worth testing further, although it is only an observation. Recursive chunking won 6 of the 7 semantic questions on which the strategies differed (Q13, Q16, Q18, Q20, Q37, Q46; fixed won Q11). Fixed chunking won 5 of the 7 keyword questions on which they differed (Q2, Q3, Q9, Q10, Q35; recursive won Q29 and Q42). Of the 2 factual questions that differed, fixed won Q1 and recursive won Q6.

### 4.2 Possible reasons (hypotheses, not verified)

- **Recursive chunking keeps sentences and paragraphs whole**, so a concept that is explained over several sentences usually sits inside one chunk. That could help semantic questions, where the cross-encoder needs the full explanation to recognise the answer.
- **Fixed windows overlap by 30 words and ignore sentence boundaries**, so a specific number or name is almost always inside some chunk with a lot of its surrounding terms. That could help keyword questions, where BM25 only needs the term to be present.
- Fixed windows can cut a sentence in half, and the cut chunk may then rank lower even though the evidence is present.

These explanations were not tested. A chunk-size ablation, per-question inspection of the 16 differing questions (`python -m evaluation.failure_analysis` produces this) and a larger evaluation set would be needed to confirm or reject them.

### 4.3 Retrieval methods

- **The reranker gave the most consistent gain.** Moving from Hybrid (RRF) to Hybrid + Rerank raised Hit@5 from 0.851 to 0.915 for fixed chunks (+3 questions) and from 0.851 to 0.936 for recursive chunks (+4 questions). Hit@3 rose by 2 and 4 questions, and recursive Hit@1 rose from 0.638 to 0.723 (+4 questions).
- **Hybrid fusion alone was not always better than the best single retriever.** With fixed chunks it beat both vector and BM25 on MRR (0.756 against 0.644 and 0.718). With recursive chunks its MRR (0.729) was slightly below BM25 alone (0.736) and its Hit@5 tied with vector search (0.851). Fusion helps recall, but its benefit depended on the chunking, so the reranker is what made the final result robust.
- **Document-level retrieval is nearly solved** (DocHit@5 0.979 to 1.000). The remaining errors are mostly about picking the right chunk inside the right document.
- **Vector search was weaker than expected on semantic questions** (Hit@5 0.737 against 0.789 for BM25). Possible reasons are the small embedding model (it truncates input at about 256 tokens) and the fact that many questions share vocabulary with their source text. This was not investigated.

## 5. Issues encountered and limitations

1. **Free Gemini API quota (answer-quality evaluation limited to 5 questions).** A full answer-quality run needs about 190 LLM calls (47 questions, 2 chunking strategies, one generation call and one judge call each). The Gemini free-tier key allowed only the first 5 questions to be graded. The next run returned `429 RESOURCE_EXHAUSTED` ("You exceeded your current quota") at question 6, the client retried four times (waiting 8, 16, 32 and 64 seconds) and then stopped. A later attempt hit the same error. The free tier's limits are small and change over time, so it cannot cover this workload in a single day. Because of this, the answer-quality results in Section 3.3 cover **5 of the 47 questions** only. All retrieval metrics cover all 47 questions and need no LLM calls. The evaluation code caches graded answers, so the run can be completed later by re-running `python -m evaluation.run_evaluation` once a higher quota is available, or by switching to another provider (Groq, OpenRouter or a local Ollama model) through `.env`.
2. **Model change.** The originally planned `gemini-2.5-flash` model had been retired, so `gemini-3.5-flash` was used.
3. **Self-grading.** In the pilot the same model generated and judged the answers, which can inflate correctness. The code supports a separate judge model.
4. **Small sample.** With 47 retrieval questions and 5 graded answers, a difference of one or two questions (2 to 4 percentage points) is noise.
5. **Evaluation set authorship.** The questions and evidence phrases were written for this project and checked against the corpus text, but not independently reviewed. Evidence-phrase matching can miss a valid passage that expresses the answer differently.
6. **Page-bounded chunks.** Answers that span two pages can be cut. TXT and DOCX citations refer to blocks of about 3000 characters, not real pages.
7. **Narrow tuning.** One embedding model, one chunk size and one overlap setting were tested.

## 6. Conclusions and next steps

- The full pipeline (hybrid retrieval, reranking, cited generation) works end to end. Retrieval with Hybrid + Rerank finds relevant evidence in the top 5 for 91.5% (fixed) and 93.6% (recursive) of the 47 questions, and the pilot answers were correct and cited real retrieved pages.
- **Reranking is the most valuable component**, and **recursive chunking is slightly ahead but not significantly better than fixed-size chunking** on this corpus.
- To strengthen the conclusions: (a) finish the answer-quality run on all 47 questions with a larger quota and a separate judge model, (b) add bootstrap confidence intervals and a chunk-size ablation, (c) inspect the questions where the strategies disagree, (d) enlarge the evaluation set and have it reviewed by a second person, and (e) try a stronger embedding model.
