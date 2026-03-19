# Hands-On AI Workshop

This repository contains a beginner-friendly, Colab-ready workshop made of Jupyter notebooks that introduce core AI / ML / DL / foundation-model ideas through practice.


## Open directly in Colab

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/01_tabular_ml_basics.ipynb) `01_tabular_ml_basics.ipynb`
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/02_simple_cnn_image_classification.ipynb) `02_simple_cnn_image_classification.ipynb`
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/03_pretrained_models_with_huggingface.ipynb) `03_pretrained_models_with_huggingface.ipynb`
4. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/04_optional_rag_or_lora_extension.ipynb) `04_optional_rag_or_lora_extension.ipynb`

## Workshop setup and orientation

You can now treat this README as the setup page:
- Open the notebook you want directly in Colab using the links above.
- Each notebook installs the specific Python packages it needs.
- Run each notebook from top to bottom.
- For the CNN notebook, GPU is recommended but not required.

If you want to enable GPU in Colab:
1. Open the notebook in Colab.
2. Go to `Runtime` -> `Change runtime type`.
3. Set `Hardware accelerator` to `GPU`.

## Workshop flow

- `01_tabular_ml_basics.ipynb`
  Supervised learning on tabular data with preprocessing, model comparison, metrics, and passenger-level experiments.
- `02_simple_cnn_image_classification.ipynb`
  A small PyTorch CNN trained from scratch on FashionMNIST with training curves, confusion matrix, and prediction playground cells.
- `03_pretrained_models_with_huggingface.ipynb`
  Beginner-friendly Hugging Face pipelines for sentiment analysis and question answering, with editable examples.
- `04_optional_rag_or_lora_extension.ipynb`
  Lightweight local RAG extension with tiny notebook-defined documents, embeddings, retrieval, and grounded answers.