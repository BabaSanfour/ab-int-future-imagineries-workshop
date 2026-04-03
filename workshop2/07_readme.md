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

- Setup and tokenization notebooks: around 10 to 20 minutes each
- Embeddings and retriever benchmarking: around 15 to 30 minutes each depending on model downloads
- LLM comparison and evaluation: around 20 to 40 minutes each depending on hardware
- Optional LoRA notebook: around 25 to 60 minutes depending on whether a GPU is available

In a fresh Colab runtime, the first model download often takes longer than the actual computation.

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

Some notebooks include commented `pip install` cells so participants can bootstrap a fresh Colab runtime quickly.

## Suggested pacing for a live workshop

If you have around 3 hours:

1. `01_tokenization_playground.ipynb` for observing how the same text becomes different model inputs
2. `02_embeddings_and_similarity.ipynb` for semantic similarity and bilingual retrieval behavior
3. `03_retriever_benchmarking_for_rag.ipynb` for the main retrieval lesson
4. `04_llm_comparison_in_same_rag_pipeline.ipynb` for generator comparison with fixed evidence
5. `05_evaluating_rag_systems.ipynb` for a structured evaluation discussion

If you have more time or a more technical audience:

1. Add a retrieval error analysis segment after notebook 3
2. Use notebook 4 to compare more model/prompt variants
3. End with notebook 6 as an optional extension on lightweight adaptation

## Suggested discussion prompts

- What makes a tokenizer a poor fit for a language or archive task?
- Which retrieval failures matter most in community-governed archive assistants?
- When is a better retriever more valuable than a better LLM?
- When should the system abstain?
- What does good citation look like in an archive assistant?
- When might LoRA help?
- When is retrieval already enough?
- Which evaluation dimensions require human review rather than automatic scoring?
- How should governance metadata affect retrieval eligibility?

## Teaching notes

- Encourage participants to paste in their own examples, but remind them to respect data handling and community governance rules.
- Keep emphasizing that retrieval quality often controls answer quality.
- Treat citation, abstention, and provenance as first-class features rather than extras.
- Use the optional LoRA notebook as a complement to retrieval-first design, not a replacement for it.
