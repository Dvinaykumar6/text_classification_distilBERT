# Text Classification with DistilBERT

This repository implements a sentiment classification pipeline using Hugging Face's `distilbert-base-uncased` model fine-tuned on the IMDB movie review dataset. The project follows a modular, scalable structure with logging, exception handling, and artifact saving.

# Objective

Classify IMDB movie reviews into **positive** or **negative** sentiment using a fine-tuned transformer model.

---

# Project Structure

```bash

text_classification_distilBERT/
│
├── artifacts/ # Contains saved models, tokenizers, and processed data
├── logs/ # Logging files
├── notebooks/ # Jupyter notebooks for EDA and experiments
├── src/ # Source code
│ ├── components/ # Core components like data ingestion, transformation, training
| ├── pipelines/ # predit pipeline
│ ├── exception.py # Custom exception handler
│ ├── logger.py # Logging setup
│ └── utils.py # Utility functions (metrics, saving models, etc.)
├── requirements.txt # Python dependencies
├── setup.py # Installable package setup
├── README.md # Project description
└── IMDB Dataset.csv # Input dataset (ensure this exists)

```

# Model

- Pretrained: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- Fine-tuned using Hugging Face `Trainer`
- Tokenizer: `AutoTokenizer` from Transformers
- Architecture: `AutoModelForSequenceClassification` with 2 output classes

---

# Pipeline Overview

1. **Data Ingestion**
    - Loads and splits IMDB dataset into train/test
    - Stores data in `artifacts/train.csv` and `artifacts/test.csv`

2. **Data Transformation**
    - Tokenizes review texts using DistilBERT tokenizer
    - Converts labels from `"positive"`/`"negative"` → `1`/`0`
    - Returns tokenized Hugging Face `DatasetDict`

3. **Model Training**
    - Fine-tunes DistilBERT on the tokenized dataset
    - Uses `Trainer` API with metrics and early stopping
    - Saves model in `artifacts/model.pkl` or a custom directory

4. **Inference**
    - `predict_pipeline.py` allows single-text or batch prediction from `.csv`
    - Outputs saved to file (e.g., `predictions.csv`)

---

# Quick Start

### 1. Clone the repository

<!-- ```bash -->
1. git clone https://github.com/Dvinaykumar6/text_classification_distilBERT.git
cd text_classification_distilBERT
2. Install dependencies
pip install -r requirements.txt
3. Run the training pipeline
python -m src.components.data_ingestion
This triggers:

Data ingestion

Tokenization

Model training

Saving artifacts

🧪 For inference



python predict_pipeline.py --input_file sample.csv --output_file predictions.csv
