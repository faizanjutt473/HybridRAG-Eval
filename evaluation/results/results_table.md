| chunking | retrieval | Hit@1 | Hit@3 | Hit@5 | DocHit@5 | MRR@5 | answers_graded | correctness | fully_correct | answer_similarity | citation_correct | citation_grounded | abstain_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | Vector only | 0.467 | 0.767 | 0.767 | 0.967 | 0.606 | - | - | - | - | - | - | - |
| fixed | BM25 only | 0.567 | 0.733 | 0.733 | 1.000 | 0.650 | - | - | - | - | - | - | - |
| fixed | Hybrid (RRF) | 0.567 | 0.767 | 0.800 | 0.967 | 0.668 | - | - | - | - | - | - | - |
| fixed | Hybrid + Rerank | 0.667 | 0.833 | 0.900 | 0.967 | 0.763 | 5.000 | 1.000 | 1.000 | 0.724 | 0.800 | 1.000 | 0.000 |
| recursive | Vector only | 0.433 | 0.733 | 0.800 | 0.967 | 0.583 | - | - | - | - | - | - | - |
| recursive | BM25 only | 0.600 | 0.733 | 0.767 | 1.000 | 0.675 | - | - | - | - | - | - | - |
| recursive | Hybrid (RRF) | 0.500 | 0.767 | 0.800 | 0.967 | 0.634 | - | - | - | - | - | - | - |
| recursive | Hybrid + Rerank | 0.667 | 0.867 | 0.933 | 0.967 | 0.761 | 5.000 | 1.000 | 1.000 | 0.738 | 0.800 | 1.000 | 0.000 |
