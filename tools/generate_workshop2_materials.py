from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSHOP2 = ROOT / "workshop2"


def src(text: str) -> list[str]:
    cleaned = textwrap.dedent(text).strip("\n")
    if not cleaned:
        return []
    return [line + "\n" for line in cleaned.splitlines()]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "colab": {
                "provenance": [],
                "include_colab_link": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    path = WORKSHOP2 / filename
    path.write_text(json.dumps(notebook(cells), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(filename: str, body: str) -> None:
    path = WORKSHOP2 / filename
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")


WORKSHOP2_SEQUENCE = [
    "01_tokenization_playground.ipynb",
    "02_embeddings_and_similarity.ipynb",
    "03_retriever_benchmarking_for_rag.ipynb",
    "04_llm_comparison_in_same_rag_pipeline.ipynb",
    "05_evaluating_rag_systems.ipynb",
    "06_optional_lora_or_domain_adaptation.ipynb",
]


def shared_setup_cells(notebook_filename: str, notebook_focus: str, optional: bool = False) -> list[dict]:
    sequence_lines = "\n".join(
        f"{index}. `{name}`" + (" (optional)" if name == "06_optional_lora_or_domain_adaptation.ipynb" else "")
        for index, name in enumerate(WORKSHOP2_SEQUENCE, start=1)
    )
    optional_text = "This notebook is optional and should usually be attempted after retrieval baselines are clear." if optional else "This is a core workshop notebook."
    intro_markdown = "\n".join(
        [
            "The workshop treats the system as retrieval-first:",
            "- tokenization defines what the model sees",
            "- retrieval quality often controls answer quality",
            "- generators cannot reliably compensate for weak retrieval",
            "- evaluation should include groundedness, citation, abstention, bilingual behavior, and community-review placeholders",
            "",
            f"Current notebook: `{notebook_filename}`",
            "",
            notebook_focus,
            "",
            optional_text,
            "",
            "Workshop sequence:",
            sequence_lines,
        ]
    )

    return [
        md_cell(intro_markdown),
        md_cell(
            """
            ## Quick concept refresher

            - Tokenization turns text into units that models can process.
            - Retrieval chooses which passages are available as evidence.
            - The retriever selects context; the generator turns context into an answer.
            - Evaluation in archive assistants is multi-layered because the system must be useful, grounded, reviewable, and appropriate.
            """
        ),
        code_cell(
            """
            # Self-contained runtime setup for this notebook.
            # Each notebook includes its own seed and device checks so it can run independently in Colab.

            import random
            import sys
            from pathlib import Path

            import numpy as np

            try:
                import torch
            except ImportError:
                torch = None

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)

            if torch is not None:
                torch.manual_seed(SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED)

            DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

            print(f"Python version: {sys.version.split()[0]}")
            print(f"Working directory: {Path.cwd()}")
            print(f"Detected device: {DEVICE}")
            print(f"Seed set to: {SEED}")

            NOTEBOOK_CONTEXT = {
                "seed": SEED,
                "device": DEVICE,
                "notebook": __name__ if "__name__" in globals() else "notebook",
                "framing": "retrieval-first archive assistant workshop",
            }

            NOTEBOOK_CONTEXT
            """
        ),
    ]


tokenization_cells = [
    md_cell(
        """
        # 01. Tokenization Playground

        This notebook compares several tokenization strategies and shows how they change what a model "sees".

        We will compare:
        - a simple word-level baseline
        - a simple character-level baseline
        - a raw byte-level baseline
        - pretrained subword tokenizers:
          - WordPiece via multilingual BERT
          - byte-level BPE via GPT-2
          - SentencePiece via mT5

        The goal is not to declare one tokenizer universally best. The goal is to see where different tokenizers behave well or badly, especially for rare words, apostrophes, diacritics, and morphologically complex forms.
        """
    ),
    *shared_setup_cells(
        notebook_filename="01_tokenization_playground.ipynb",
        notebook_focus="This notebook shows how segmentation choices affect token counts, fragmentation, vocabulary coverage, and downstream behavior in retrieval or generation.",
    ),
    code_cell(
        """
        # Install tokenizer libraries if needed.
        # In Colab, uncomment the next line if transformers is not already available.

        # !pip -q install transformers pandas matplotlib seaborn

        print("Tokenizer dependencies are ready once transformers, pandas, and matplotlib are installed.")
        """
    ),
    code_cell(
        """
        # Imports for the notebook.
        # We keep them grouped so participants can see the core dependencies clearly.

        import re
        from collections import Counter

        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from transformers import AutoTokenizer

        sns.set_theme(style="whitegrid")
        """
    ),
    code_cell(
        """
        # Editable examples.
        # Participants should feel free to change these strings or add their own.
        # The morphologically rich example uses Turkish because it is easy to inspect
        # and reliably available in public tokenizer vocabularies.

        EXAMPLES = {
            "english_plain": "The archive assistant should return the source and say when it is unsure.",
            "english_with_apostrophes": "Community members' place-names shouldn't be normalized without review.",
            "morphologically_rich": "Müzelerdeki kayıtlarımızdanmışsınız gibi konuşmayalım.",
            "diacritics_and_variation": "cafe café co-operate cooperate place-name place name",
            "paste_your_own_here": "Replace this line with your own sentence, orthography, or mixed-language example.",
        }

        EXAMPLES
        """
    ),
    code_cell(
        """
        # Load three pretrained tokenizers with different segmentation styles.
        # - BERT gives us WordPiece behavior.
        # - GPT-2 gives us byte-level BPE behavior.
        # - mT5 gives us SentencePiece behavior.

        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

        print("Loaded tokenizers:")
        print("-", bert_tokenizer.name_or_path)
        print("-", gpt2_tokenizer.name_or_path)
        print("-", mt5_tokenizer.name_or_path)
        """
    ),
    code_cell(
        """
        # Baseline tokenizers.
        # These are intentionally simple so participants can inspect the assumptions directly.

        def word_level_tokenize(text: str):
            # Split into alphabetic word-like spans plus punctuation.
            return re.findall(r"\\w+|[^\\w\\s]", text, flags=re.UNICODE)


        def character_level_tokenize(text: str):
            # Characters include spaces only if you keep them.
            # We drop spaces here so token counts are easier to compare.
            return [char for char in text if not char.isspace()]


        def byte_level_tokenize(text: str):
            # Represent each UTF-8 byte as a compact hex string.
            # This is a transparent baseline for understanding byte segmentation.
            return [f"0x{byte:02x}" for byte in text.encode("utf-8")]


        def hf_tokenize(text: str, tokenizer):
            # Convert model ids back into string tokens for display.
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            return tokenizer.convert_ids_to_tokens(ids)


        TOKENIZER_REGISTRY = {
            "word_baseline": lambda text: word_level_tokenize(text),
            "character_baseline": lambda text: character_level_tokenize(text),
            "byte_baseline": lambda text: byte_level_tokenize(text),
            "wordpiece_mbert": lambda text: hf_tokenize(text, bert_tokenizer),
            "byte_bpe_gpt2": lambda text: hf_tokenize(text, gpt2_tokenizer),
            "sentencepiece_mt5": lambda text: hf_tokenize(text, mt5_tokenizer),
        }
        """
    ),
    code_cell(
        """
        # A compact comparison function.
        # This is useful for live teaching because it prints the same sentence under every tokenizer.

        def compare_tokenizations(text: str, tokenizer_registry=TOKENIZER_REGISTRY, max_preview_tokens: int = 60):
            rows = []
            print("=" * 100)
            print(f"TEXT: {text}")
            print("=" * 100)

            for name, tokenize_fn in tokenizer_registry.items():
                tokens = tokenize_fn(text)
                preview = tokens[:max_preview_tokens]
                print(f"\\n{name} | count={len(tokens)}")
                print(" | ".join(preview))
                if len(tokens) > max_preview_tokens:
                    print("... [truncated preview]")

                rows.append(
                    {
                        "tokenizer": name,
                        "token_count": len(tokens),
                        "avg_token_length_chars": sum(len(token) for token in tokens) / max(len(tokens), 1),
                        "unique_tokens": len(set(tokens)),
                    }
                )

            return pd.DataFrame(rows).sort_values("token_count")
        """
    ),
    code_cell(
        """
        # Run the comparison on every example sentence.
        # You can replace EXAMPLES with your own dictionary if you want to expand the notebook.

        all_summaries = {}

        for label, text in EXAMPLES.items():
            print(f"\\n\\n### Example: {label}")
            summary_df = compare_tokenizations(text)
            display(summary_df)
            all_summaries[label] = summary_df
        """
    ),
    code_cell(
        """
        # Convert the collected summaries into a single table for plotting.
        # Fertility here is a simple educational metric:
        # token_count divided by whitespace word count.

        plot_rows = []

        for label, text in EXAMPLES.items():
            whitespace_word_count = max(len(text.split()), 1)
            for name, tokenize_fn in TOKENIZER_REGISTRY.items():
                tokens = tokenize_fn(text)
                plot_rows.append(
                    {
                        "example": label,
                        "tokenizer": name,
                        "token_count": len(tokens),
                        "fertility_vs_whitespace_words": len(tokens) / whitespace_word_count,
                    }
                )

        plot_df = pd.DataFrame(plot_rows)
        plot_df.head()
        """
    ),
    code_cell(
        """
        # Plot token counts across tokenizers for each example.
        # This is often the fastest way to make the tokenization differences visible in a workshop.

        plt.figure(figsize=(14, 6))
        sns.barplot(data=plot_df, x="example", y="token_count", hue="tokenizer")
        plt.xticks(rotation=25, ha="right")
        plt.title("Token count changes depending on tokenizer choice")
        plt.ylabel("Token count")
        plt.xlabel("Example")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 6))
        sns.barplot(data=plot_df, x="example", y="fertility_vs_whitespace_words", hue="tokenizer")
        plt.xticks(rotation=25, ha="right")
        plt.title("Fertility relative to simple whitespace word count")
        plt.ylabel("Tokens per whitespace word")
        plt.xlabel("Example")
        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        # Inspect how a specific rare or complex word fragments.
        # This is especially helpful when discussing morphology and OOV behavior.

        candidate_words = [
            "place-names",
            "shouldn't",
            "Müzelerdeki",
            "kayıtlarımızdanmışsınız",
            "café",
        ]

        fragmentation_rows = []

        for word in candidate_words:
            for name, tokenize_fn in TOKENIZER_REGISTRY.items():
                tokens = tokenize_fn(word)
                fragmentation_rows.append(
                    {
                        "word": word,
                        "tokenizer": name,
                        "token_count": len(tokens),
                        "tokens": " | ".join(tokens),
                    }
                )

        fragmentation_df = pd.DataFrame(fragmentation_rows)
        fragmentation_df
        """
    ),
    code_cell(
        """
        # Vocabulary coverage and OOV behavior matter mostly for closed-vocabulary tokenizers.
        # The simple word baseline can demonstrate OOV counts against a tiny toy vocabulary.

        toy_vocab = {
            "the",
            "archive",
            "assistant",
            "should",
            "return",
            "source",
            "and",
            "say",
            "when",
            "it",
            "is",
            "unsure",
        }

        def toy_word_oov_report(text: str, vocab: set[str]):
            words = [token.lower() for token in word_level_tokenize(text) if re.match(r"\\w+", token)]
            in_vocab = [word for word in words if word in vocab]
            oov = [word for word in words if word not in vocab]
            return {
                "words": words,
                "in_vocab": in_vocab,
                "oov": oov,
                "oov_rate": len(oov) / max(len(words), 1),
            }


        for label, text in EXAMPLES.items():
            report = toy_word_oov_report(text, toy_vocab)
            print(f"\\nExample: {label}")
            print("Words:", report["words"])
            print("OOV:", report["oov"])
            print("OOV rate:", round(report["oov_rate"], 3))
        """
    ),
    md_cell(
        """
        ## Discussion prompts

        - Which tokenizer looked most language-agnostic?
        - Which tokenizer fragmented apostrophes or diacritics in surprising ways?
        - Which tokenizer produced the longest sequence for the morphologically rich example?
        - How might token inflation hurt retrieval chunking or generation cost?
        - When might a byte-level strategy help, and when might it become inefficient?
        """
    ),
    code_cell(
        """
        # Exercise cell: paste your own sentence here and rerun.
        # Try one example with punctuation variation and one with a long inflected form.

        custom_text = "Paste your own example here."
        custom_summary = compare_tokenizations(custom_text)
        display(custom_summary)
        """
    ),
]


embedding_cells = [
    md_cell(
        """
        # 02. Embeddings and Similarity

        This notebook shows how short archive-like passages become vectors and why embedding quality matters for retrieval.

        We will:
        - encode a small corpus
        - compare similarity scores for multiple queries
        - inspect nearest neighbors
        - visualize rough embedding neighborhoods

        The main lesson is that semantic similarity can help retrieval, but it can also fail when the model does not understand the domain, the community terminology, or mixed-language inputs.
        """
    ),
    *shared_setup_cells(
        notebook_filename="02_embeddings_and_similarity.ipynb",
        notebook_focus="This notebook turns archive-like passages into embeddings so participants can inspect nearest neighbors, semantic drift, and bilingual retrieval behavior.",
    ),
    code_cell(
        """
        # In Colab, uncomment the line below if sentence-transformers is missing.

        # !pip -q install sentence-transformers scikit-learn pandas matplotlib seaborn

        print("Embedding notebook dependencies are ready once the required packages are installed.")
        """
    ),
    code_cell(
        """
        # Imports.

        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.decomposition import PCA
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer

        sns.set_theme(style="whitegrid")
        """
    ),
    code_cell(
        """
        # A small archive-like corpus.
        # The passages are deliberately short so the similarity behavior is easy to inspect.

        corpus = [
            {
                "doc_id": "P01",
                "language": "en",
                "title": "Public metadata policy",
                "text": "Public descriptions may be shared openly, but restricted ceremonial material requires a community access review before release.",
            },
            {
                "doc_id": "P02",
                "language": "en",
                "title": "Place-name interview",
                "text": "The recording explains how elders prefer historical place-names to appear with their locally used spelling variants.",
            },
            {
                "doc_id": "P03",
                "language": "fr",
                "title": "Résumé bilingue",
                "text": "Le résumé bilingue décrit une entrevue sur les noms de lieux, les variantes orthographiques et les règles de citation.",
            },
            {
                "doc_id": "P04",
                "language": "en",
                "title": "Winter storytelling protocol",
                "text": "Several stories are marked for winter-only teaching sessions and should not be used outside the approved seasonal context.",
            },
            {
                "doc_id": "P05",
                "language": "en",
                "title": "Audio transcription note",
                "text": "This note compares manual transcription with OCR-derived transcript text and warns that speaker turns were merged incorrectly.",
            },
            {
                "doc_id": "P06",
                "language": "mixed",
                "title": "Catalog note with community term",
                "text": "The catalog entry links an English description with a community-language keyword placeholder for a kinship-based access practice.",
            },
        ]

        corpus_df = pd.DataFrame(corpus)
        corpus_df
        """
    ),
    code_cell(
        """
        # Editable queries.
        # Add your own queries to see how the embedding model behaves.

        queries = [
            "Which material requires access review before public release?",
            "How should place-names and spelling variants be cited?",
            "Quels documents parlent des variantes orthographiques?",
            "Which records are limited to winter teaching sessions?",
            "Find mixed-language metadata about kinship-based access.",
        ]

        queries
        """
    ),
    code_cell(
        """
        # Load a compact multilingual embedding model.
        # Multilingual models are useful for bilingual or mixed-language retrieval experiments.

        embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embedding_model = SentenceTransformer(embedding_model_name)
        print(f"Loaded embedding model: {embedding_model_name}")
        """
    ),
    code_cell(
        """
        # Encode the corpus and the queries.
        # normalize_embeddings=True is convenient because cosine similarity then becomes a dot product.

        corpus_texts = corpus_df["text"].tolist()
        corpus_embeddings = embedding_model.encode(
            corpus_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embeddings = embedding_model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        print("Corpus embedding shape:", corpus_embeddings.shape)
        print("Query embedding shape:", query_embeddings.shape)
        """
    ),
    code_cell(
        """
        # Rank passages for each query by cosine similarity.
        # This is the simplest dense retrieval baseline.

        def rank_passages(query: str, query_embedding, corpus_df: pd.DataFrame, corpus_embeddings, top_k: int = 3):
            scores = cosine_similarity([query_embedding], corpus_embeddings)[0]
            ranked = corpus_df.copy()
            ranked["score"] = scores
            ranked = ranked.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
            ranked.insert(0, "query", query)
            return ranked


        ranked_results = []
        for query, query_embedding in zip(queries, query_embeddings):
            ranked_results.append(rank_passages(query, query_embedding, corpus_df, corpus_embeddings, top_k=3))

        ranked_df = pd.concat(ranked_results, ignore_index=True)
        ranked_df
        """
    ),
    code_cell(
        """
        # Inspect one query at a time in a compact display.
        # This makes it easier to talk through good matches and surprising false positives.

        focus_query = queries[0]
        focus_embedding = query_embeddings[0]
        focus_results = rank_passages(focus_query, focus_embedding, corpus_df, corpus_embeddings, top_k=5)

        print("Focus query:", focus_query)
        display(focus_results[["doc_id", "language", "title", "score", "text"]])
        """
    ),
    code_cell(
        """
        # Visualize the embedding neighborhood with PCA.
        # This is only a rough projection, but it helps participants see clustering behavior.

        labels = [f"{row.doc_id}: {row.title}" for row in corpus_df.itertuples()]
        projection = PCA(n_components=2, random_state=42).fit_transform(corpus_embeddings)

        plt.figure(figsize=(10, 7))
        plt.scatter(projection[:, 0], projection[:, 1], s=100)

        for (x, y), label in zip(projection, labels):
            plt.text(x + 0.01, y + 0.01, label, fontsize=9)

        plt.title("Rough 2D view of archive-passage embedding neighborhoods")
        plt.xlabel("PCA component 1")
        plt.ylabel("PCA component 2")
        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        # Compare semantically similar versus lexically similar items.
        # This exercise is useful when discussing why dense retrieval can help beyond keyword overlap.

        comparison_queries = [
            "seasonal teaching restrictions",
            "winter-only story access",
            "orthographic variants in place names",
            "speaker turns merged in transcript",
        ]

        comparison_embeddings = embedding_model.encode(
            comparison_queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        comparison_tables = []
        for query, query_embedding in zip(comparison_queries, comparison_embeddings):
            table = rank_passages(query, query_embedding, corpus_df, corpus_embeddings, top_k=3)
            comparison_tables.append(table)

        display(pd.concat(comparison_tables, ignore_index=True))
        """
    ),
    md_cell(
        """
        ## Discussion prompts

        - Which queries worked well even when lexical overlap was weak?
        - Where did the model confuse topical similarity with the actual information need?
        - Did the bilingual query behave as expected?
        - How might a general embedding model fail on community-specific terminology or underrepresented orthographies?
        """
    ),
    code_cell(
        """
        # Exercise cell: edit the corpus or queries and rerun the encoding section.
        # This is intentionally lightweight so participants can experiment in live sessions.

        custom_query = "Find records about citation rules for place-names."
        custom_query_embedding = embedding_model.encode([custom_query], convert_to_numpy=True, normalize_embeddings=True)[0]
        custom_results = rank_passages(custom_query, custom_query_embedding, corpus_df, corpus_embeddings, top_k=3)
        display(custom_results)
        """
    ),
]


retriever_cells = [
    md_cell(
        """
        # 03. Retriever Benchmarking for RAG

        This notebook benchmarks retrieval methods in a tiny RAG-style setup.

        The most important point in this notebook is explicit:

        > A weak retriever limits downstream answer quality, no matter how fluent the generator is.

        We will compare:
        - TF-IDF lexical retrieval
        - BM25 lexical retrieval
        - dense retrieval with sentence embeddings
        - FAISS vector search over the same dense embeddings
        - a simple hybrid score
        """
    ),
    *shared_setup_cells(
        notebook_filename="03_retriever_benchmarking_for_rag.ipynb",
        notebook_focus="This notebook benchmarks retrievers directly, because weak retrieval is often the main reason a RAG assistant gives a poor or unsupported answer.",
    ),
    code_cell(
        """
        # In Colab, uncomment if needed.

        # !pip -q install sentence-transformers scikit-learn pandas matplotlib seaborn rank-bm25 faiss-cpu

        print("Retriever benchmarking dependencies are ready once the packages are installed.")
        """
    ),
    code_cell(
        """
        # Imports.

        import numpy as np
        import pandas as pd
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        import faiss
        """
    ),
    code_cell(
        """
        # A tiny curated corpus with metadata.
        # The passages are short enough to inspect manually during a live workshop.

        archive_passages = [
            {
                "doc_id": "A01",
                "source": "oral_history_box_1",
                "language": "en",
                "type": "oral_history",
                "access_level": "public",
                "text": "An elder describes the river crossing route and explains why some place-names keep older spellings in the catalog.",
            },
            {
                "doc_id": "A02",
                "source": "protocol_memo_2021",
                "language": "en",
                "type": "policy_note",
                "access_level": "restricted_placeholder",
                "text": "Ceremonial recordings require a community review before access is granted, even when short public summaries are available.",
            },
            {
                "doc_id": "A03",
                "source": "winter_story_index",
                "language": "en",
                "type": "story_index",
                "access_level": "seasonal_placeholder",
                "text": "Several teaching stories are marked as winter-only and should not be surfaced as general-purpose examples outside that season.",
            },
            {
                "doc_id": "A04",
                "source": "bilingual_catalog_sheet",
                "language": "fr",
                "type": "catalog_note",
                "access_level": "public",
                "text": "Le catalogue bilingue note des variantes orthographiques, des noms de lieux et les règles de citation pour les résumés publics.",
            },
            {
                "doc_id": "A05",
                "source": "transcription_qc_log",
                "language": "en",
                "type": "quality_log",
                "access_level": "internal_placeholder",
                "text": "The transcript quality log reports OCR errors, merged speaker turns, and missing diacritics in several archived interviews.",
            },
            {
                "doc_id": "A06",
                "source": "access_training_notes",
                "language": "mixed",
                "type": "training_note",
                "access_level": "public",
                "text": "Training notes explain a kinship-based access practice and warn that retrieval should respect governance metadata before showing results.",
            },
            {
                "doc_id": "A07",
                "source": "language_lessons_audio",
                "language": "en",
                "type": "lesson",
                "access_level": "public",
                "text": "A beginner language lesson pairs short audio clips with English glosses and a note about respectful pronunciation practice.",
            },
            {
                "doc_id": "A08",
                "source": "donor_agreement_summary",
                "language": "en",
                "type": "agreement",
                "access_level": "restricted_placeholder",
                "text": "The donor agreement summary states that certain recordings can be described at a high level but not quoted directly in generated answers.",
            },
        ]

        corpus_df = pd.DataFrame(archive_passages)
        corpus_df
        """
    ),
    code_cell(
        """
        # A tiny hand-built retrieval benchmark.
        # Each query includes the relevant document ids we hope retrieval will surface.

        benchmark_queries = [
            {
                "query_id": "Q01",
                "query": "Which records need community review before access?",
                "relevant_doc_ids": ["A02"],
            },
            {
                "query_id": "Q02",
                "query": "Quels documents parlent des variantes orthographiques et des noms de lieux?",
                "relevant_doc_ids": ["A04", "A01"],
            },
            {
                "query_id": "Q03",
                "query": "Find material that should only be used in winter teaching contexts.",
                "relevant_doc_ids": ["A03"],
            },
            {
                "query_id": "Q04",
                "query": "Which notes mention OCR errors or merged speaker turns?",
                "relevant_doc_ids": ["A05"],
            },
            {
                "query_id": "Q05",
                "query": "Find records about kinship-based access practices and governance metadata.",
                "relevant_doc_ids": ["A06"],
            },
        ]

        benchmark_df = pd.DataFrame(benchmark_queries)
        benchmark_df
        """
    ),
    code_cell(
        """
        # Chunking matters in real systems.
        # Here we keep one passage per row, but participants can experiment with chunk size below.

        CHUNK_SIZE_WORDS = 80

        print("Current educational chunk size setting:", CHUNK_SIZE_WORDS)
        print("In real archives, chunk size and metadata filtering can change retrieval quality a lot.")
        """
    ),
    code_cell(
        """
        # Build lexical retrievers.
        # TF-IDF and BM25 often behave differently when lexical overlap is sparse.

        corpus_texts = corpus_df["text"].tolist()

        tfidf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)

        bm25_corpus = [text.lower().split() for text in corpus_texts]
        bm25 = BM25Okapi(bm25_corpus)
        """
    ),
    code_cell(
        """
        # Build dense retrieval resources.
        # We compare brute-force cosine retrieval against FAISS for the same embeddings.

        embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embedding_model = SentenceTransformer(embedding_model_name)

        dense_embeddings = embedding_model.encode(
            corpus_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        faiss_index = faiss.IndexFlatIP(dense_embeddings.shape[1])
        faiss_index.add(dense_embeddings)

        print("Dense embedding matrix shape:", dense_embeddings.shape)
        """
    ),
    code_cell(
        """
        # Helper functions for each retriever.
        # Returning a DataFrame keeps the outputs easy to read and compare.

        def retrieve_tfidf(query: str, top_k: int = 3):
            query_vector = tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vector, tfidf_matrix)[0]
            result = corpus_df.copy()
            result["score"] = scores
            return result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


        def retrieve_bm25(query: str, top_k: int = 3):
            scores = bm25.get_scores(query.lower().split())
            result = corpus_df.copy()
            result["score"] = scores
            return result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


        def retrieve_dense(query: str, top_k: int = 3):
            query_vector = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores = cosine_similarity(query_vector, dense_embeddings)[0]
            result = corpus_df.copy()
            result["score"] = scores
            return result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


        def retrieve_faiss(query: str, top_k: int = 3):
            query_vector = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            scores, indices = faiss_index.search(query_vector, top_k)
            result = corpus_df.iloc[indices[0]].copy()
            result["score"] = scores[0]
            return result.reset_index(drop=True)


        def retrieve_hybrid(query: str, top_k: int = 3):
            tfidf_scores = cosine_similarity(tfidf_vectorizer.transform([query]), tfidf_matrix)[0]
            dense_scores = cosine_similarity(
                embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True),
                dense_embeddings,
            )[0]

            combined = 0.5 * tfidf_scores + 0.5 * dense_scores
            result = corpus_df.copy()
            result["score"] = combined
            return result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
        """
    ),
    code_cell(
        """
        # Quick qualitative comparison for a single query.
        # This is a good checkpoint before computing aggregate metrics.

        sample_query = benchmark_df.iloc[1]["query"]
        print("Sample query:", sample_query)

        for name, fn in {
            "tfidf": retrieve_tfidf,
            "bm25": retrieve_bm25,
            "dense": retrieve_dense,
            "faiss": retrieve_faiss,
            "hybrid": retrieve_hybrid,
        }.items():
            print(f"\\nRetriever: {name}")
            display(fn(sample_query, top_k=3)[["doc_id", "language", "type", "access_level", "score", "text"]])
        """
    ),
    code_cell(
        """
        # Metric helpers.
        # precision_at_k answers: how many of the top-k were relevant?
        # hit_rate_at_k answers: did we retrieve at least one relevant item?
        # recall_at_k answers: how much of the known relevant set did we recover?

        def precision_at_k(retrieved_ids, relevant_ids, k):
            return sum(doc_id in relevant_ids for doc_id in retrieved_ids[:k]) / max(k, 1)


        def hit_rate_at_k(retrieved_ids, relevant_ids, k):
            return float(any(doc_id in relevant_ids for doc_id in retrieved_ids[:k]))


        def recall_at_k(retrieved_ids, relevant_ids, k):
            if not relevant_ids:
                return 0.0
            return sum(doc_id in relevant_ids for doc_id in retrieved_ids[:k]) / len(relevant_ids)
        """
    ),
    code_cell(
        """
        # Benchmark every retriever over the small hand-built evaluation set.
        # These results are tiny but still useful for discussion.

        retrievers = {
            "tfidf": retrieve_tfidf,
            "bm25": retrieve_bm25,
            "dense": retrieve_dense,
            "faiss": retrieve_faiss,
            "hybrid": retrieve_hybrid,
        }

        evaluation_rows = []
        ranking_rows = []

        TOP_K = 3

        for record in benchmark_queries:
            query = record["query"]
            relevant_ids = record["relevant_doc_ids"]

            for retriever_name, retriever_fn in retrievers.items():
                retrieved = retriever_fn(query, top_k=TOP_K)
                retrieved_ids = retrieved["doc_id"].tolist()

                evaluation_rows.append(
                    {
                        "query_id": record["query_id"],
                        "retriever": retriever_name,
                        "precision_at_3": precision_at_k(retrieved_ids, relevant_ids, TOP_K),
                        "hit_rate_at_3": hit_rate_at_k(retrieved_ids, relevant_ids, TOP_K),
                        "recall_at_3": recall_at_k(retrieved_ids, relevant_ids, TOP_K),
                    }
                )

                for rank, doc_id in enumerate(retrieved_ids, start=1):
                    ranking_rows.append(
                        {
                            "query_id": record["query_id"],
                            "query": query,
                            "retriever": retriever_name,
                            "rank": rank,
                            "doc_id": doc_id,
                            "is_relevant": doc_id in relevant_ids,
                        }
                    )

        metrics_df = pd.DataFrame(evaluation_rows)
        rankings_df = pd.DataFrame(ranking_rows)

        display(metrics_df)
        """
    ),
    code_cell(
        """
        # Aggregate results by retriever.
        # This gives a compact summary for discussion.

        summary_df = (
            metrics_df.groupby("retriever")[["precision_at_3", "hit_rate_at_3", "recall_at_3"]]
            .mean()
            .sort_values("precision_at_3", ascending=False)
            .reset_index()
        )

        display(summary_df)
        """
    ),
    code_cell(
        """
        # Inspect ranking failures.
        # These are often more informative than the aggregate numbers.

        failure_cases = rankings_df[(rankings_df["rank"] == 1) & (~rankings_df["is_relevant"])]
        failure_cases
        """
    ),
    md_cell(
        """
        ## Exercises

        - Change the query wording so lexical overlap becomes weaker. Which retriever degrades first?
        - Add a more culturally specific query and note whether dense retrieval helps.
        - Change the chunk size policy in a real dataset and discuss how it could affect precision and recall.
        - Identify cases where a fluent generator might hide the fact that the retriever failed.
        """
    ),
]


llm_cells = [
    md_cell(
        """
        # 04. LLM Comparison in the Same RAG Pipeline

        This notebook compares generators while holding retrieval fixed.

        That is important because otherwise it is easy to confuse:
        - a better retriever
        - a better prompt
        - a better generator

        Here we will retrieve the same context first, then pass the same evidence into multiple open models.
        """
    ),
    *shared_setup_cells(
        notebook_filename="04_llm_comparison_in_same_rag_pipeline.ipynb",
        notebook_focus="This notebook isolates the generator by holding retrieval fixed, making it easier to compare groundedness, style, hallucination tendencies, and citation behavior across models.",
    ),
    code_cell(
        """
        # In Colab, uncomment if needed.

        # !pip -q install sentence-transformers transformers accelerate pandas

        print("LLM comparison dependencies are ready once the required packages are installed.")
        """
    ),
    code_cell(
        """
        # Imports.

        import re
        import textwrap

        import pandas as pd
        try:
            import torch
        except ImportError:
            torch = None
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        from transformers import pipeline
        """
    ),
    code_cell(
        """
        # A small retrieval corpus for the RAG experiment.
        # We keep the evidence set transparent so it is easy to inspect citations manually.

        rag_corpus = [
            {
                "chunk_id": "C01",
                "text": "Public summaries can be shown openly, but restricted ceremonial recordings require a community access review before release.",
            },
            {
                "chunk_id": "C02",
                "text": "The bilingual catalog sheet documents place-name spelling variants and recommends citing the original source note.",
            },
            {
                "chunk_id": "C03",
                "text": "Some teaching stories are marked winter-only and should not be surfaced outside the approved seasonal context.",
            },
            {
                "chunk_id": "C04",
                "text": "Transcript quality notes mention merged speaker turns, missing diacritics, and OCR mistakes in interview records.",
            },
            {
                "chunk_id": "C05",
                "text": "Governance notes explain that kinship-based access rules must be checked before showing search results to the public.",
            },
        ]

        rag_df = pd.DataFrame(rag_corpus)
        rag_df
        """
    ),
    code_cell(
        """
        # Build a fixed retriever.
        # We use a single dense retriever here so the generator comparison is the only moving part.

        retriever_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        retriever_model = SentenceTransformer(retriever_model_name)

        rag_embeddings = retriever_model.encode(
            rag_df["text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


        def retrieve_context(query: str, top_k: int = 3):
            query_embedding = retriever_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores = cosine_similarity(query_embedding, rag_embeddings)[0]
            ranked = rag_df.copy()
            ranked["score"] = scores
            return ranked.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
        """
    ),
    code_cell(
        """
        # Default models.
        # These are relatively accessible in Colab.
        # Larger placeholders are included as optional comments for participants with more compute.

        MODEL_SPECS = {
            "flan_t5_small": {
                "model_name": "google/flan-t5-small",
                "task": "text2text-generation",
            },
            "flan_t5_base": {
                "model_name": "google/flan-t5-base",
                "task": "text2text-generation",
            },
            # Optional heavier model ideas for a GPU-backed Colab session:
            # "mistral_7b_instruct": {
            #     "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
            #     "task": "text-generation",
            # },
        }

        list(MODEL_SPECS.keys())
        """
    ),
    code_cell(
        """
        # Cache pipelines so models are only loaded once.
        # This keeps repeated experimentation manageable.

        model_cache = {}
        PIPELINE_DEVICE = 0 if (torch is not None and torch.cuda.is_available()) else -1


        def get_generation_pipeline(model_key: str):
            spec = MODEL_SPECS[model_key]
            if model_key not in model_cache:
                model_cache[model_key] = pipeline(
                    spec["task"],
                    model=spec["model_name"],
                    tokenizer=spec["model_name"],
                    device=PIPELINE_DEVICE,
                )
            return model_cache[model_key]
        """
    ),
    code_cell(
        """
        # Prompt builder.
        # The prompt explicitly asks for grounded answering, citations, and abstention when evidence is weak.

        def build_prompt(query: str, retrieved_df: pd.DataFrame) -> str:
            evidence_blocks = []
            for row in retrieved_df.itertuples():
                evidence_blocks.append(f"[{row.chunk_id}] {row.text}")

            evidence_text = "\\n".join(evidence_blocks)

            return textwrap.dedent(
                f\"\"\"
                You are assisting with a community-governed archive.
                Answer the question using only the evidence below.
                If the evidence is insufficient, say ABSTAIN and explain why briefly.
                Cite chunk ids in square brackets.

                Question: {query}

                Evidence:
                {evidence_text}

                Response format:
                Answer: <grounded answer or ABSTAIN>
                Citations: [chunk ids]
                \"\"\"
            ).strip()
        """
    ),
    code_cell(
        """
        # Small robustness fix:
        # text2text-generation pipelines often return `generated_text`,
        # but keeping a parser helper makes the behavior explicit and easier to modify.

        def extract_generated_text(pipeline_output):
            first = pipeline_output[0]
            if "generated_text" in first:
                return first["generated_text"]
            if "summary_text" in first:
                return first["summary_text"]
            return str(first)


        def generate_rag_answer(model_key: str, query: str, retrieved_df: pd.DataFrame, max_new_tokens: int = 128):
            generator = get_generation_pipeline(model_key)
            prompt = build_prompt(query, retrieved_df)
            raw_output = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            return {
                "model_key": model_key,
                "prompt": prompt,
                "response": extract_generated_text(raw_output),
            }
        """
    ),
    code_cell(
        """
        # Compare models on the same retrieval result.
        # Feel free to change the query or top_k.

        query = "Which materials require review before they can be released publicly?"
        retrieved_context = retrieve_context(query, top_k=3)

        print("Retrieved context:")
        display(retrieved_context[["chunk_id", "score", "text"]])

        model_outputs = []
        for model_key in MODEL_SPECS:
            result = generate_rag_answer(model_key, query, retrieved_context)
            model_outputs.append(result)

        outputs_df = pd.DataFrame(model_outputs)
        outputs_df[["model_key", "response"]]
        """
    ),
    code_cell(
        """
        # Basic citation inspection.
        # This is not a full evaluation pipeline, but it helps surface whether a model is grounding its answer.

        def parse_citations(text: str):
            return re.findall(r"\\[(C\\d+)\\]", text)


        citation_rows = []
        retrieved_ids = set(retrieved_context["chunk_id"].tolist())

        for row in outputs_df.itertuples():
            citations = parse_citations(row.response)
            citation_rows.append(
                {
                    "model_key": row.model_key,
                    "citations_found": citations,
                    "all_citations_in_retrieved_context": set(citations).issubset(retrieved_ids),
                    "abstained": "ABSTAIN" in row.response.upper(),
                }
            )

        citation_df = pd.DataFrame(citation_rows)
        citation_df
        """
    ),
    md_cell(
        """
        ## Exercises

        - Compare a smaller and larger model while keeping retrieval fixed.
        - Ask a question that the retrieved evidence does not support. Does the model abstain?
        - Inspect whether the model cites the chunks it actually used.
        - Identify cases where a more capable generator still cannot rescue weak retrieval.
        """
    ),
    code_cell(
        """
        # Optional exercise cell:
        # try a query that should trigger abstention or expose hallucination tendencies.

        exercise_query = "Which donor agreement allows direct quotation of restricted recordings?"
        exercise_context = retrieve_context(exercise_query, top_k=3)
        display(exercise_context[["chunk_id", "score", "text"]])

        exercise_outputs = []
        for model_key in MODEL_SPECS:
            exercise_outputs.append(generate_rag_answer(model_key, exercise_query, exercise_context))

        pd.DataFrame(exercise_outputs)[["model_key", "response"]]
        """
    ),
]


evaluation_cells = [
    md_cell(
        """
        # 05. Evaluating RAG Systems

        This notebook turns evaluation into code and structured review.

        We will score a small archive-style benchmark using both automatic heuristics and qualitative review fields.

        The main lesson is that evaluation for archive assistants should not collapse everything into one generic score.
        """
    ),
    *shared_setup_cells(
        notebook_filename="05_evaluating_rag_systems.ipynb",
        notebook_focus="This notebook treats evaluation as a multi-part process that includes retrieval precision, abstention, citation quality, hallucination checks, bilingual behavior, and review placeholders.",
    ),
    code_cell(
        """
        # In Colab, uncomment if needed.

        # !pip -q install sentence-transformers transformers pandas scikit-learn

        print("Evaluation notebook dependencies are ready once the required packages are installed.")
        """
    ),
    code_cell(
        """
        # Imports.

        import re
        import textwrap

        import pandas as pd
        try:
            import torch
        except ImportError:
            torch = None
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        from transformers import pipeline
        """
    ),
    code_cell(
        """
        # Reuse a small corpus with explicit ids so citations stay inspectable.

        eval_corpus = [
            {"chunk_id": "E01", "text": "Restricted ceremonial recordings require community review before any access decision."},
            {"chunk_id": "E02", "text": "Public catalog summaries should cite the original source note when listing place-name spelling variants."},
            {"chunk_id": "E03", "text": "Winter-only stories should not be surfaced outside the approved seasonal context."},
            {"chunk_id": "E04", "text": "Transcript quality notes document OCR mistakes, missing diacritics, and merged speaker turns."},
            {"chunk_id": "E05", "text": "Governance metadata must be checked before showing kinship-based access materials to general users."},
        ]

        eval_corpus_df = pd.DataFrame(eval_corpus)
        eval_corpus_df
        """
    ),
    code_cell(
        """
        # A hand-built evaluation set.
        # We include expected support, abstention expectations, and a bilingual placeholder field.

        evaluation_set = [
            {
                "example_id": "R01",
                "query": "Which records require community review before access?",
                "query_language": "en",
                "relevant_chunk_ids": ["E01"],
                "should_abstain": False,
                "expected_keywords": ["community review", "access"],
            },
            {
                "example_id": "R02",
                "query": "Quels résumés publics doivent citer la note source originale?",
                "query_language": "fr",
                "relevant_chunk_ids": ["E02"],
                "should_abstain": False,
                "expected_keywords": ["citer", "source"],
            },
            {
                "example_id": "R03",
                "query": "Can I show winter-only stories as general examples in summer?",
                "query_language": "en",
                "relevant_chunk_ids": ["E03"],
                "should_abstain": False,
                "expected_keywords": ["winter-only", "not"],
            },
            {
                "example_id": "R04",
                "query": "Which donor agreement permits public quotation of restricted ceremonies?",
                "query_language": "en",
                "relevant_chunk_ids": [],
                "should_abstain": True,
                "expected_keywords": [],
            },
            {
                "example_id": "R05",
                "query": "Find the note about OCR mistakes and missing diacritics.",
                "query_language": "en",
                "relevant_chunk_ids": ["E04"],
                "should_abstain": False,
                "expected_keywords": ["OCR", "diacritics"],
            },
        ]

        eval_df = pd.DataFrame(evaluation_set)
        eval_df
        """
    ),
    code_cell(
        """
        # Build a small retriever and a small generator.
        # The point here is to evaluate the whole retrieval-grounded loop, not to maximize performance.

        retriever_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        eval_embeddings = retriever_model.encode(
            eval_corpus_df["text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            tokenizer="google/flan-t5-small",
            device=0 if (torch is not None and torch.cuda.is_available()) else -1,
        )
        """
    ),
    code_cell(
        """
        # Retrieval and generation helpers.
        # These remain deliberately simple so participants can modify them live.

        def retrieve_eval_context(query: str, top_k: int = 3):
            query_embedding = retriever_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores = cosine_similarity(query_embedding, eval_embeddings)[0]
            ranked = eval_corpus_df.copy()
            ranked["score"] = scores
            return ranked.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


        def build_eval_prompt(query: str, context_df: pd.DataFrame) -> str:
            evidence = "\\n".join([f"[{row.chunk_id}] {row.text}" for row in context_df.itertuples()])
            return textwrap.dedent(
                f\"\"\"
                Answer using only the evidence below.
                If the evidence is insufficient, say ABSTAIN.
                Cite chunk ids in square brackets.

                Question: {query}

                Evidence:
                {evidence}

                Response format:
                Answer: ...
                Citations: [chunk ids]
                \"\"\"
            ).strip()


        def generate_eval_answer(query: str, context_df: pd.DataFrame):
            prompt = build_eval_prompt(query, context_df)
            return generator(prompt, max_new_tokens=96, do_sample=False)[0]["generated_text"]
        """
    ),
    code_cell(
        """
        # Automatic metrics.
        # These are intentionally lightweight heuristics.
        # Real projects should combine them with human review and task-specific checks.

        def parse_chunk_citations(text: str):
            return re.findall(r"\\[(E\\d+)\\]", text)


        def retrieval_precision(retrieved_ids, relevant_ids, k):
            return sum(doc_id in relevant_ids for doc_id in retrieved_ids[:k]) / max(k, 1)


        def abstention_quality(answer: str, should_abstain: bool):
            abstained = "ABSTAIN" in answer.upper()
            return float(abstained == should_abstain)


        def citation_correctness(answer: str, relevant_ids):
            if not relevant_ids:
                return float("ABSTAIN" in answer.upper())
            citations = parse_chunk_citations(answer)
            if not citations:
                return 0.0
            return float(set(citations).issubset(set(relevant_ids)))


        def bilingual_quality_heuristic(answer: str, query_language: str):
            if query_language == "fr":
                french_markers = [" le ", " la ", " des ", " et ", " est ", " réponse", "source"]
                score = any(marker in f" {answer.lower()} " for marker in french_markers)
                return float(score)
            return 1.0


        def hallucination_flag(answer: str, relevant_ids):
            citations = parse_chunk_citations(answer)
            if "ABSTAIN" in answer.upper():
                return 0
            if relevant_ids and not citations:
                return 1
            if citations and not set(citations).intersection(set(relevant_ids)):
                return 1
            return 0
        """
    ),
    code_cell(
        """
        # Run the small benchmark.
        # We log both scores and qualitative placeholders for later manual review.

        rows = []

        for example in evaluation_set:
            context_df = retrieve_eval_context(example["query"], top_k=3)
            answer = generate_eval_answer(example["query"], context_df)
            retrieved_ids = context_df["chunk_id"].tolist()

            rows.append(
                {
                    "example_id": example["example_id"],
                    "query": example["query"],
                    "query_language": example["query_language"],
                    "retrieved_ids": retrieved_ids,
                    "answer": answer,
                    "retrieval_precision_at_3": retrieval_precision(retrieved_ids, example["relevant_chunk_ids"], 3),
                    "abstention_quality": abstention_quality(answer, example["should_abstain"]),
                    "citation_correctness": citation_correctness(answer, example["relevant_chunk_ids"]),
                    "bilingual_quality": bilingual_quality_heuristic(answer, example["query_language"]),
                    "hallucination_flag": hallucination_flag(answer, example["relevant_chunk_ids"]),
                    "community_review_status": "TO_REVIEW",
                    "community_review_notes": "",
                }
            )

        results_df = pd.DataFrame(rows)
        results_df
        """
    ),
    code_cell(
        """
        # Aggregate numeric metrics.
        # Keep in mind that these are only the quantitative layer of evaluation.

        numeric_columns = [
            "retrieval_precision_at_3",
            "abstention_quality",
            "citation_correctness",
            "bilingual_quality",
            "hallucination_flag",
        ]

        summary = results_df[numeric_columns].mean().to_frame(name="mean_score")
        summary
        """
    ),
    code_cell(
        """
        # Qualitative review template.
        # Archive assistants often need explicit human review fields, not just automatic metrics.

        qualitative_review = results_df[
            [
                "example_id",
                "query",
                "answer",
                "retrieved_ids",
                "community_review_status",
                "community_review_notes",
            ]
        ].copy()

        qualitative_review["citation_notes"] = ""
        qualitative_review["support_notes"] = ""
        qualitative_review["cultural_appropriateness_notes"] = ""
        qualitative_review
        """
    ),
    md_cell(
        """
        ## Exercises

        - Mark which examples should abstain and inspect whether the model did so.
        - Look for fluent but weakly supported answers.
        - Design one more evaluation dimension tied to governance, permissions, or community review.
        - Compare a citation failure with a retrieval failure. Which is easier to fix?
        """
    ),
]


lora_cells = [
    md_cell(
        """
        # 06. Optional LoRA or Domain Adaptation

        This optional notebook demonstrates a minimal PEFT/LoRA workflow after a retrieval baseline is already in place.

        The framing here is intentionally conservative:
        - do not assume full fine-tuning
        - do not assume LoRA replaces retrieval
        - evaluate style changes separately from actual task improvement

        In small archive assistants, LoRA can be useful, but it can also create a false sense of progress if retrieval is still weak.
        """
    ),
    *shared_setup_cells(
        notebook_filename="06_optional_lora_or_domain_adaptation.ipynb",
        notebook_focus="This notebook explores lightweight adaptation after a retrieval baseline exists, with LoRA positioned as a complement to retrieval rather than a substitute for it.",
        optional=True,
    ),
    code_cell(
        """
        # In Colab, uncomment if needed.

        # !pip -q install transformers datasets peft accelerate pandas

        print("LoRA notebook dependencies are ready once the required packages are installed.")
        """
    ),
    md_cell(
        """
        ## Practical note

        This notebook is optional because even lightweight adaptation adds complexity:
        - training loops take time
        - GPUs help a lot
        - evaluation becomes more important, not less

        If you only have CPU access, you can still read through the workflow and reduce the training set and number of steps.
        """
    ),
    code_cell(
        """
        # Imports.

        import pandas as pd
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
        """
    ),
    code_cell(
        """
        # A tiny toy dataset.
        # The examples are deliberately small because the goal is workflow literacy, not strong performance.

        toy_examples = [
            {
                "context": "[D01] Restricted ceremonial recordings require community review before access.",
                "question": "Can this ceremonial recording be released immediately?",
                "target": "ABSTAIN. The evidence says community review is required before access. Citations: [D01]",
            },
            {
                "context": "[D02] Public catalog summaries should cite the original source note when describing place-name variants.",
                "question": "How should a public summary cite place-name variants?",
                "target": "Use the original source note when citing place-name variants. Citations: [D02]",
            },
            {
                "context": "[D03] Winter-only stories should not be surfaced outside the approved seasonal context.",
                "question": "Can the assistant quote this winter-only story in summer?",
                "target": "No. The story should not be surfaced outside the approved seasonal context. Citations: [D03]",
            },
            {
                "context": "[D04] Governance metadata must be checked before showing kinship-based access materials.",
                "question": "What should the system check before showing kinship-based materials?",
                "target": "It should check governance metadata first. Citations: [D04]",
            },
        ]

        toy_df = pd.DataFrame(toy_examples)
        toy_df
        """
    ),
    code_cell(
        """
        # Choose a small seq2seq model that is realistic for demonstration.
        # flan-t5-small is light enough for a workshop demo and works well with PEFT.

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        base_model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(DEVICE)

        print("Loaded base model:", base_model_name)
        print("Using device:", DEVICE)
        """
    ),
    code_cell(
        """
        # Turn the toy examples into an instruction-style dataset.
        # We keep the prompt format simple and explicit.

        def format_prompt(example):
            return {
                "input_text": (
                    "Use only the provided context. "
                    "If the evidence is insufficient, answer ABSTAIN.\\n\\n"
                    f"Context: {example['context']}\\n"
                    f"Question: {example['question']}"
                ),
                "target_text": example["target"],
            }


        formatted_examples = [format_prompt(example) for example in toy_examples]
        dataset = Dataset.from_list(formatted_examples)
        dataset
        """
    ),
    code_cell(
        """
        # Tokenize the dataset.
        # This function is intentionally verbose so participants can modify lengths and formatting easily.

        MAX_INPUT_LENGTH = 192
        MAX_TARGET_LENGTH = 96


        def tokenize_example(batch):
            model_inputs = tokenizer(
                batch["input_text"],
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
            )
            labels = tokenizer(
                text_target=batch["target_text"],
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs


        tokenized_dataset = dataset.map(tokenize_example, batched=True, remove_columns=dataset.column_names)
        tokenized_dataset
        """
    ),
    code_cell(
        """
        # Configure LoRA.
        # For T5-style attention modules, q and v are common lightweight targets.

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q", "v"],
        )

        lora_model = get_peft_model(base_model, lora_config)
        lora_model.print_trainable_parameters()
        """
    ),
    code_cell(
        """
        # A very small training configuration.
        # This is intentionally modest so the notebook remains demonstrative rather than expensive.

        training_args = Seq2SeqTrainingArguments(
            output_dir="./lora_demo_outputs",
            learning_rate=5e-4,
            per_device_train_batch_size=2,
            num_train_epochs=8,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=torch.cuda.is_available(),
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=lora_model)

        trainer = Seq2SeqTrainer(
            model=lora_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        """
    ),
    code_cell(
        """
        # Before training, compare base and LoRA-wrapped outputs on a held-out prompt.
        # The LoRA model starts from the same base behavior before optimization.

        test_prompt = (
            "Use only the provided context. If the evidence is insufficient, answer ABSTAIN.\\n\\n"
            "Context: [D05] Public summaries can describe restricted materials at a high level but should not quote them directly.\\n"
            "Question: Can the assistant quote the restricted material directly?"
        )


        def generate_text(model, prompt, max_new_tokens=64):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {name: tensor.to(next(model.parameters()).device) for name, tensor in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)


        print("Base model output before LoRA training:")
        print(generate_text(base_model, test_prompt))

        print("\\nLoRA model output before training:")
        print(generate_text(lora_model, test_prompt))
        """
    ),
    code_cell(
        """
        # Train the LoRA adapters.
        # On CPU this may be slow; on Colab GPU it should be manageable because the dataset is tiny.

        trainer.train()
        """
    ),
    code_cell(
        """
        # Compare behavior after adaptation.
        # We are mostly looking for stylistic changes or slightly better task formatting.
        # Strong performance gains would be surprising with such a tiny dataset.

        print("Base model output after LoRA training step (base weights unchanged):")
        print(generate_text(base_model, test_prompt))

        print("\\nAdapted LoRA model output:")
        print(generate_text(lora_model, test_prompt))
        """
    ),
    md_cell(
        """
        ## Risks to discuss

        - Overfitting: the model may memorize the tiny examples rather than learn a robust behavior
        - Memorization: sensitive phrasing or archival conventions may be reproduced too literally
        - Catastrophic forgetting: less likely with LoRA than full fine-tuning, but still worth monitoring
        - False progress: style can improve while retrieval remains the real bottleneck

        A good rule of thumb is to compare any adaptation against a strong retrieval-first baseline before claiming success.
        """
    ),
    code_cell(
        """
        # Exercise ideas:
        # 1. Add more examples that require abstention.
        # 2. Add bilingual prompts and see whether the adaptation helps or just changes style.
        # 3. Lower num_train_epochs and compare whether the behavior change is stable.

        print("Suggested exercise: add one more governance-focused example and rerun tokenization + training.")
        """
    ),
]


readme_body = """
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
"""


def main() -> None:
    WORKSHOP2.mkdir(parents=True, exist_ok=True)

    write_notebook("01_tokenization_playground.ipynb", tokenization_cells)
    write_notebook("02_embeddings_and_similarity.ipynb", embedding_cells)
    write_notebook("03_retriever_benchmarking_for_rag.ipynb", retriever_cells)
    write_notebook("04_llm_comparison_in_same_rag_pipeline.ipynb", llm_cells)
    write_notebook("05_evaluating_rag_systems.ipynb", evaluation_cells)
    write_notebook("06_optional_lora_or_domain_adaptation.ipynb", lora_cells)
    write_text("07_readme.md", readme_body)

    legacy_overview = WORKSHOP2 / "00_overview_and_setup.ipynb"
    if legacy_overview.exists():
        legacy_overview.unlink()


if __name__ == "__main__":
    main()
