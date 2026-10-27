# Failure analysis (Hybrid + Rerank)

## fixed: 4 question(s) missed at Hit@5

### Q4 [semantic] Why does the Transformer need positional encodings?
- expected: attention_is_all_you_need.pdf | evidence: ['positional encodings', 'sine and cosine functions']
- evidence is actually in: attention_is_all_you_need.pdf p6, attention_is_all_you_need.pdf p8
- retrieved top-5:
  - #1 wiki_transformer_model.docx p15: 'encoding methods than sinusoidal. The original transformer paper reported using a learned '
  - #2 wiki_transformer_model.docx p7: 'this positional encoding function is that using it, shifts are linear transformations: f ('
  - #3 wiki_transformer_model.docx p1: 'Transformer (deep learning architecture) In deep learning, the transformer is a family of '
  - #4 wiki_transformer_model.docx p7: '( x W + b ) {\\displaystyle \\mathrm {UnEmbed} (x)=\\mathrm {softmax} (xW+b)} The matrix has '
  - #5 wiki_transformer_model.docx p15: 'unlike the encoder, while the RNN decoder was much faster. Subsequent work Many improvemen'

### Q24 [factual] Which two model architectures are proposed in the word2vec paper?
- expected: word2vec.pdf | evidence: ['skip-gram']
- evidence is actually in: word2vec.pdf p4, word2vec.pdf p5, word2vec.pdf p7, word2vec.pdf p8, word2vec.pdf p9, word2vec.pdf p10, word2vec.pdf p11
- retrieved top-5:
  - #1 word2vec.pdf p4: '2.3 Parallel Training of Neural Networks To train models on huge data sets, we have implem'
  - #2 wiki_transformer_model.docx p3: 'of research from bag of words and word2vec. It was followed by BERT (2018), an encoder-onl'
  - #3 word2vec.pdf p4: 'and then the N-gram NNLM is trained on top of these distributed representations of words. '
  - #4 wiki_transformer_model.docx p3: 'This led to the introduction of a multi-head attention model that was easier to paralleliz'
  - #5 word2vec.pdf p1: 'words [10]. For example, neural network based language models signiﬁcantly outperform N-gr'

### Q29 [keyword] By how much does DPR outperform BM25 in top-20 passage retrieval accuracy?
- expected: dense_passage_retrieval.pdf | evidence: ['19%']
- evidence is actually in: dense_passage_retrieval.pdf p1
- retrieved top-5:
  - #1 dense_passage_retrieval.pdf p2: 'of careful ablation studies, our ﬁnal solution is surprisingly simple: the embedding is op'
  - #2 dense_passage_retrieval.pdf p6: 'affect the results much. More details can be found in Appendix B. Cross-dataset generaliza'
  - #3 dense_passage_retrieval.pdf p12: 'Table 5: Retrieval accuracy on the development set of Natural Questions, trained on passag'
  - #4 dense_passage_retrieval.pdf p12: 'which is documented in our code release. C Qualitative Analysis Although DPR performs bett'
  - #5 dense_passage_retrieval.pdf p5: 'than BM25 on all datasets. The gap is especially large whenk is small (e.g., 78.4% vs. 59.'

### Q36 [semantic] What is the goal of an agent in reinforcement learning?
- expected: wiki_reinforcement_learning.txt | evidence: ['cumulative reward']
- evidence is actually in: wiki_reinforcement_learning.txt p3, wiki_reinforcement_learning.txt p4
- retrieved top-5:
  - #1 wiki_reinforcement_learning.txt p2: '. The purpose of reinforcement learning is for the agent to learn an optimal (or near-opti'
  - #2 wiki_reinforcement_learning.txt p2: 'reward R t {\\displaystyle R_{t}} . It then chooses an action A t {\\displaystyle A_{t}} fro'
  - #3 wiki_reinforcement_learning.txt p1: 'Reinforcement learning In machine learning and optimal control, reinforcement learning (RL'
  - #4 wiki_reinforcement_learning.txt p3: "the value by 4, the transition will not be allowed. When the agent's performance is compar"
  - #5 wiki_machine_learning.txt p8: 'Reinforcement learning is an area of machine learning concerned with how software agents o'

## recursive: 3 question(s) missed at Hit@5

### Q4 [semantic] Why does the Transformer need positional encodings?
- expected: attention_is_all_you_need.pdf | evidence: ['positional encodings', 'sine and cosine functions']
- evidence is actually in: attention_is_all_you_need.pdf p6, attention_is_all_you_need.pdf p8
- retrieved top-5:
  - #1 wiki_transformer_model.docx p7: '. Some architectures use the transpose of the embedding matrix M {\\displaystyle M} as the '
  - #2 wiki_transformer_model.docx p1: 'Because self-attention alone is permutation-invariant, transformers inject positional info'
  - #3 wiki_transformer_model.docx p15: 'The original transformer paper reported using a learned positional encoding, but finding i'
  - #4 wiki_transformer_model.docx p1: 'Transformer (deep learning architecture) In deep learning, the transformer is a family of '
  - #5 wiki_transformer_model.docx p15: 'Many improvements have been added to the Transformer architecture. A standard improved con'

### Q24 [factual] Which two model architectures are proposed in the word2vec paper?
- expected: word2vec.pdf | evidence: ['skip-gram']
- evidence is actually in: word2vec.pdf p4, word2vec.pdf p5, word2vec.pdf p7, word2vec.pdf p8, word2vec.pdf p9, word2vec.pdf p10, word2vec.pdf p11
- retrieved top-5:
  - #1 word2vec.pdf p2: '2 Model Architectures Many different types of models were proposed for estimating continuo'
  - #2 word2vec.pdf p4: 'The new architectures directly follow those proposed in our earlier work [13, 14], where i'
  - #3 word2vec.pdf p4: '3.1 Continuous Bag-of-Words Model The ﬁrst proposed architecture is similar to the feedfor'
  - #4 word2vec.pdf p4: '2.3 Parallel Training of Neural Networks To train models on huge data sets, we have implem'
  - #5 word2vec.pdf p2: '1.2 Previous Work Representation of words as continuous vectors has a long history [10, 26'

### Q36 [semantic] What is the goal of an agent in reinforcement learning?
- expected: wiki_reinforcement_learning.txt | evidence: ['cumulative reward']
- evidence is actually in: wiki_reinforcement_learning.txt p3, wiki_reinforcement_learning.txt p4
- retrieved top-5:
  - #1 wiki_reinforcement_learning.txt p2: 'from the set of available actions, which is subsequently sent to the environment. The envi'
  - #2 wiki_reinforcement_learning.txt p2: '. The purpose of reinforcement learning is for the agent to learn an optimal (or near-opti'
  - #3 wiki_reinforcement_learning.txt p1: 'Reinforcement learning In machine learning and optimal control, reinforcement learning (RL'
  - #4 wiki_reinforcement_learning.txt p3: 'For example, the state of an account balance could be restricted to be positive; if the cu'
  - #5 wiki_machine_learning.txt p8: 'Reinforcement learning is an area of machine learning concerned with how software agents o'

## Fixed vs recursive: questions where the outcome differs
- by reciprocal rank: recursive better on 9, fixed better on 7, tie on 31 questions

| Q | type | fixed hit@5 | recursive hit@5 | fixed RR | recursive RR | question |
|---|---|---|---|---|---|---|
| 1 | factual | 1 | 1 | 1.00 | 0.50 | How many attention heads does the base Transformer model use, and what |
| 2 | keyword | 1 | 1 | 1.00 | 0.50 | What BLEU score did the big Transformer reach on the WMT 2014 English- |
| 3 | keyword | 1 | 1 | 1.00 | 0.33 | How many warmup steps were used in the Transformer's learning rate sch |
| 6 | factual | 1 | 1 | 0.20 | 0.33 | What two unsupervised pre-training tasks does BERT use? |
| 9 | keyword | 1 | 1 | 1.00 | 0.33 | Which corpora was BERT pre-trained on? |
| 10 | keyword | 1 | 1 | 0.50 | 0.25 | What are the two formulations of RAG models proposed in the retrieval- |
| 11 | semantic | 1 | 1 | 0.50 | 0.25 | What retriever and generator components make up the RAG model? |
| 13 | semantic | 1 | 1 | 0.50 | 1.00 | What problem does residual learning address in very deep networks? |
| 16 | semantic | 1 | 1 | 0.50 | 1.00 | How does LoRA reduce the number of trainable parameters when adapting  |
| 18 | semantic | 1 | 1 | 0.50 | 1.00 | Does LoRA add inference latency compared to a fully fine-tuned model? |
| 20 | semantic | 1 | 1 | 0.20 | 1.00 | What does the name Adam stand for and what is the method based on? |
| 29 | keyword | 0 | 1 | 0.00 | 0.33 | By how much does DPR outperform BM25 in top-20 passage retrieval accur |
| 35 | keyword | 1 | 1 | 0.50 | 0.33 | Which researcher introduced the neocognitron, an early architecture th |
| 37 | semantic | 1 | 1 | 0.50 | 1.00 | Which trade-off is central to reinforcement learning? |
| 42 | keyword | 1 | 1 | 0.33 | 0.50 | Which algorithm is the standard method for solving k-means clustering? |
| 46 | semantic | 1 | 1 | 0.50 | 1.00 | What problem are LSTM units designed to mitigate in recurrent neural n |
