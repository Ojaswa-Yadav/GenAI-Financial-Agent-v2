# Data for GenAI Financial Agent

## Overview
This project uses two datasets to fine-tune and test the GenAI Financial Agent. These datasets provide financial question-answer pairs to train the model for financial advisory tasks.

---

## Datasets Used

### 1. **Wealth Alpaca Dataset**
- **Source**: Hugging Face ([View Dataset](https://huggingface.co/datasets/gbharti/wealth-alpaca_lora)).
- **Description**:
  - A curated dataset containing financial Q&A pairs designed for financial advisory and language tasks.
  - The dataset includes instructions (`instruction`), optional inputs (`input`), and desired outputs (`output`).

- **Usage**:
  - Load the dataset using the Hugging Face `datasets` library:
    ```python
    from datasets import load_dataset
    wealth_alpaca = load_dataset("gbharti/wealth-alpaca_lora")
    ```

---

### 2. **Financial QA 10k Dataset**
- **Source**: Kaggle ([View Dataset](https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k/data)).
- **Description**:
  - A dataset containing 10,000 financial question-answer pairs.
  - Covers diverse financial topics, such as loans, investments, and savings, making it ideal for training financial conversational agents.

- **Usage**:
  - Download the dataset from Kaggle and place the CSV file (`Financial-QA-10k.csv`) in this directory (`data/`).





