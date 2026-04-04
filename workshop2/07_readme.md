# Workshop 2: Advanced Practical NLP for Archive Assistants

This folder contains a Colab-ready hands-on workshop for participants who already know the basics of ML and NLP and want to explore tokenization, retrieval, model comparison, evaluation, and lightweight adaptation in a retrieval-grounded archive assistant setting.

Each notebook is self-contained: the workshop framing, runtime setup, seed initialization, and device checks now live inside every notebook instead of in a separate setup notebook.

## Open In Colab

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop2/01_tokenization_playground.ipynb) `01_tokenization_playground.ipynb`
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop2/02_embeddings_and_similarity.ipynb) `02_embeddings_and_similarity.ipynb`
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop2/03_retriever_benchmarking_for_rag.ipynb) `03_retriever_benchmarking_for_rag.ipynb`
4. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop2/04_llm_comparison_in_same_rag_pipeline.ipynb) `04_llm_comparison_in_same_rag_pipeline.ipynb`
5. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop2/05_evaluating_rag_systems.ipynb) `05_evaluating_rag_systems.ipynb`
6. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop2/06_optional_lora_or_domain_adaptation.ipynb) `06_optional_lora_or_domain_adaptation.ipynb`

## Files

- `01_tokenization_playground.ipynb`: compare word, character, byte, WordPiece, byte-BPE, and SentencePiece tokenization
- `02_embeddings_and_similarity.ipynb`: encode archive-style passages and inspect similarity behavior
- `03_retriever_benchmarking_for_rag.ipynb`: benchmark lexical, dense, FAISS, and hybrid retrievers
- `04_llm_comparison_in_same_rag_pipeline.ipynb`: hold retrieval fixed and compare multiple open generators
- `05_evaluating_rag_systems.ipynb`: evaluate retrieval-grounded answers with both automatic and qualitative criteria
- `06_optional_lora_or_domain_adaptation.ipynb`: optional LoRA/PEFT demo on a tiny toy dataset

## Core versus optional

Core notebooks:
- `01_tokenization_playground.ipynb`
- `02_embeddings_and_similarity.ipynb`
- `03_retriever_benchmarking_for_rag.ipynb`
- `04_llm_comparison_in_same_rag_pipeline.ipynb`
- `05_evaluating_rag_systems.ipynb`

Optional notebook:
- `06_optional_lora_or_domain_adaptation.ipynb`

## Expected runtime

These are teaching notebooks, not large-scale benchmarks.

## Required libraries

Most notebooks use a subset of:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `transformers`
- `sentence-transformers`
- `rank-bm25`
- `faiss-cpu`
- `datasets`
- `peft`
- `accelerate`