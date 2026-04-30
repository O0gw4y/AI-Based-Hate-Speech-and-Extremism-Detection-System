# Hate Speech Classification with Twitter-RoBERTa

Fine-tuned **cardiffnlp/twitter-roberta-base-hate** on the [HateXplain](https://github.com/hate-explainer/hatexplain) dataset for 3-class hate speech detection across social media posts from Twitter and Gab.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model & Training](#model--training)
- [Evaluation Results](#evaluation-results)
- [Inference Example](#inference-example)

---

## Overview

This project fine-tunes a RoBERTa-based transformer model pre-trained on hate speech data to classify social media posts into three categories: **hatespeech**, **normal**, and **offensive**. The pipeline handles class imbalance via weighted cross-entropy loss and includes a full evaluation script with confusion matrix visualisation and single-text inference.

---

## Dataset

**HateXplain** — a benchmark dataset for hate speech detection with human rationale annotations.

| Property | Details |
|---|---|
| Source | Twitter & Gab |
| Total posts | 20,148 |
| Annotators per post | 3 |
| Label assignment | Majority vote |
| Classes (3-class) | `hatespeech`, `normal`, `offensive` |
| Classes (2-class) | `toxic`, `non-toxic` |

**Label distribution (majority vote):**

| Label | Count | Share |
|---|---|---|
| normal | 7,814 | 38.8% |
| offensive | 6,399 | 31.8% |
| hatespeech | 5,935 | 29.5% |

**Official train / val / test split:**

| Split | Samples |
|---|---|
| Train | 15,383 |
| Validation | 1,922 |
| Test | 1,924 |

> **Citation:** Mathew et al. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.* AAAI 2021. [[Paper]](https://arxiv.org/abs/2012.10289)

---

## Repository Structure

```
├── hate_speech_classifier.ipynb  # Full training pipeline
├── hate_speech_evaluation.ipynb  # Evaluation & inference script
│
└── dataset/
    ├── dataset.json              # HateXplain posts & annotations
    ├── post_id_divisions.json    # Official train/val/test splits
    ├── classes.npy               # 3-class label array
    └── classes_two.npy           # 2-class label array
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install transformers torch scikit-learn pandas numpy seaborn matplotlib sentencepiece
```

### 3. Google Colab (recommended)

Both scripts are written for Google Colab with Drive mounting. On first run, mount your Drive and set `PROJECT_DIR` to your folder path:

```python
from google.colab import drive
drive.mount("/content/drive")

PROJECT_DIR = "/content/drive/MyDrive/YOUR_FOLDER_NAME"
```

Upload the `dataset/` files to `PROJECT_DIR` before running the training script.

---

## Usage

### Training

Run `hate_speech_classifier.ipynb` end-to-end. It will:

1. Parse `dataset.json` and assign majority-vote labels
2. Split data using `post_id_divisions.json`
3. Fine-tune `cardiffnlp/twitter-roberta-base-hate`
4. Save the model and tokenizer to `PROJECT_DIR/bert_3class_final/`

```bash
jupyter nbconvert --to notebook --execute hate_speech_classifier.ipynb
```

### Evaluation

Run `hate_speech_evaluation.ipynb` after training. It will:

1. Load the saved model from `bert_3class_final/`
2. Run batch inference on the test set
3. Print accuracy, macro-F1, and a full classification report
4. Save a normalised confusion matrix to `PROJECT_DIR/evaluation/`

```bash
jupyter nbconvert --to notebook --execute hate_speech_evaluation.ipynb
```

---

## Model & Training

| Parameter | Value |
|---|---|
| Base model | `cardiffnlp/twitter-roberta-base-hate` |
| Task | 3-class sequence classification |
| Epochs | 7 |
| Batch size | 16 |
| Max sequence length | 128 tokens |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Scheduler | Linear with 10% warmup |
| Loss function | Class-weighted cross-entropy |

Class weighting is applied via `sklearn.utils.class_weight.compute_class_weight` to mitigate the label imbalance between classes.

**Training progress:**

| Epoch | Train Loss | Val Macro-F1 |
|---|---|---|
| 1 | 0.8374 | 0.6647 |
| 2 | 0.6528 | 0.6946 |
| 3 | 0.5166 | 0.6992 |
| 4 | 0.3650 | 0.6854 |
| 5 | 0.2337 | 0.7001 |
| 6 | 0.1557 | 0.6897 |
| 7 | 0.1026 | 0.6913 |

---

## Evaluation Results

Evaluated on the held-out test set of **1,924 samples**.

| Metric | Score |
|---|---|
| Accuracy | 0.69 |
| Macro F1 | 0.68 |

**Per-class breakdown:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| hatespeech | 0.74 | 0.79 | 0.77 | 594 |
| normal | 0.76 | 0.72 | 0.74 | 782 |
| offensive | 0.52 | 0.52 | 0.52 | 548 |
| **macro avg** | **0.67** | **0.68** | **0.68** | **1,924** |
| **weighted avg** | **0.69** | **0.69** | **0.69** | **1,924** |

The `offensive` class is the hardest to classify (F1: 0.52), which is consistent with findings in the HateXplain paper — offensive content often overlaps linguistically with both hate speech and normal posts, making it the most ambiguous of the three categories.

---

## Inference Example

Run the inference cells at the bottom of `hate_speech_evaluation.ipynb`. The `predict()` function can also be copied into any notebook after loading the model:

```python
samples = [
    "I love everyone in my community!",
    "I hate people from that group.",
    "You are so stupid!",
]

for text in samples:
    result = predict(text)
    print(f"Text       : {result['text']}")
    print(f"Prediction : {result['prediction']}  ({result['confidence']:.2%})")
    print(f"Probs      : {result['probabilities']}\n")
```

**Output:**

```
Text       : I love everyone in my community!
Prediction : normal  (99.76%)
Probs      : {'hatespeech': 0.13%, 'normal': 99.76%, 'offensive': 0.10%}

Text       : I hate people from that group.
Prediction : hatespeech  (67.70%)
Probs      : {'hatespeech': 67.70%, 'normal': 7.93%, 'offensive': 24.37%}

Text       : You are so stupid!
Prediction : offensive  (98.23%)
Probs      : {'hatespeech': 0.66%, 'normal': 1.11%, 'offensive': 98.23%}
```

---

## Acknowledgements

- **HateXplain dataset** — Mathew et al., AAAI 2021
- **Base model** — [cardiffnlp/twitter-roberta-base-hate](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate) by Cardiff NLP
