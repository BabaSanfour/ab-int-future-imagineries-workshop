# Workshop 3 — Hugging Face & Transformers in the Wild

    A hands-on, beginner-to-medium tour of the **Hugging Face** ecosystem across three
    different applications, keeping the workshop's critical lens on community needs,
    Indigenous and local knowledge, power, governance, and representation.

    Each notebook is self-contained and runs top-to-bottom in **Google Colab**
    (enable a GPU via `Runtime -> Change runtime type`). Every notebook is structured as
    **beginner -> medium -> optional/advanced**, with `# TODO` "your turn" cells (followed by
    solutions), "play with it" cells for experimentation, and "critical reflection" prompts.

    ## Open in Colab

    1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop3/01_computer_vision_with_transformers.ipynb) `01_computer_vision_with_transformers.ipynb`
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop3/02_finetuning_transformers_for_text.ipynb) `02_finetuning_transformers_for_text.ipynb`
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop3/03_transformers_for_sensor_time_series.ipynb) `03_transformers_for_sensor_time_series.ipynb`

    ### Solutions (fully-solved, runnable reference)

    Every blank filled in, comments trimmed, and `pip install` lines activated — use these to
    confirm everything runs, or as an answer key.

    1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop3/solutions/01_computer_vision_with_transformers.ipynb) `solutions/01_computer_vision_with_transformers.ipynb`
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop3/solutions/02_finetuning_transformers_for_text.ipynb) `solutions/02_finetuning_transformers_for_text.ipynb`
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabaSanfour/ab-int-future-imagineries-workshop/blob/main/workshop3/solutions/03_transformers_for_sensor_time_series.ipynb) `solutions/03_transformers_for_sensor_time_series.ipynb`

    ## What each notebook covers

    - **01 · Computer Vision with Transformers** — run pretrained image classifiers, do
      zero-shot classification with CLIP (you choose the categories), fine-tune a Vision
      Transformer, and **build a CNN from scratch** to see why pretraining wins. Experiments:
      perturbation robustness, backbone comparison, overfitting. Theme: biodiversity &
      environmental monitoring, and *who a model is built to see*.
    - **02 · Fine-tuning a Transformer for Text** — fine-tune DistilBERT for emotion
      classification with the Hugging Face `Trainer`, then run a reusable experiment "knob-box":
      a data-scaling curve, freeze-vs-full fine-tuning, a learning-rate sweep, a zero-shot
      baseline, plus a bias probe and optional **LoRA**. Theme: language, representation, and
      *whose values the labels encode*.
    - **03 · Transformers for Sensor Time Series** — build a Transformer encoder to detect
      pollution events in synthetic air-quality sensor data, then compare against a **1D-CNN**,
      ablate positional encoding, stress-test **sensor-failure robustness**, and probe the model
      with **occlusion**; optional forecasting with a Hugging Face **PatchTST** model. Theme:
      community environmental monitoring, data sovereignty, and *who defines "normal"*.

    ## Regenerating these notebooks

    The notebooks are generated from a single script so they stay consistent:

    ```bash
    python tools/generate_workshop3_materials.py
    ```

    Edit `tools/generate_workshop3_materials.py` and rerun to update the `.ipynb` files.
