# GenAI Financial Agent

## Overview
The GenAI Financial Agent is a conversational AI system designed to assist users with financial queries by leveraging **Qwen 2.5**, fine-tuned on financial datasets using **LoRA** and **4-bit quantization**. It integrates **Retrieval-Augmented Generation (RAG)** for enhanced context-aware answers and supports both naive and agentic RAG implementations.

---

## Features
- **Fine-Tuned Qwen 2.5 Model**:
  - Optimized using **QLoRA** for parameter-efficient training.
  - Trained on a c dataset of **Wealth Alpaca** 

- **Retrieval-Augmented Generation (RAG)**:
  - **Naive RAG**: Simple retrieval-based answer generation.
  - **Agentic RAG**: Enhanced decision-making for complex financial tasks.
  -  **Fine-Tuned RAG**: Retrieval-based answer generation using the Qwen 2.5 fine tuned model with database as **Wealth Alpaca** and **Financial QA 10k**. .

- **Interactive Inference**:
  - Real-time question-answering using the fine-tuned model.


## Installation

You can install the GenAI Financial Agent by running these commands:

```bash
git clone https://github.com/Ojaswa-Yadav/GenAI-Financial-Agent-v2.git
cd GenAI-Financial-Agent-v2
pip install -r requirements.txt
