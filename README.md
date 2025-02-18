# **GenAI Financial Agent**

## **Overview**

The **GenAI Financial Agent** is a conversational AI system designed to assist users with financial queries by leveraging the **Qwen 2.5** model, fine-tuned on financial datasets using **LoRA** and 4-bit quantization. It integrates **Retrieval-Augmented Generation (RAG)** for enhanced context-aware answers and supports both naive and agentic RAG implementations.

This project provides a seamless and intelligent solution for financial decision-making, combining the latest advancements in generative AI and financial data analysis.


---

## **Key Highlights**

### **QLoRA Fine-Tuning**
- **Memory Efficiency:**
  - Used **4-bit quantization** for training on consumer GPUs, significantly reducing memory requirements.
- **Task-Specific Optimization:**
  - Fine-tuned specific attention layers for financial Q&A tasks.
- **Domain Adaptation:**
  - Trained on the **Wealth Alpaca Dataset** to enhance the model's understanding of the financial domain.


- **Naive RAG:**
  - Integrated **ChromaDB** with **MiniLM embeddings** for document retrieval, ensuring fast and accurate responses.
- **Agentic RAG:**
  - Enhanced with agentic reasoning capabilities to handle more complex financial decision-making tasks.

### **Performance Results**
| Metric               | Score |
|-----------------------|-------|
| **Context Precision** | 100%  |
| **Answer Relevance**  | 84%   |
| **Faithfulness**      | 80%   |

---

## **Features**

### **1. Fine-Tuned Qwen 2.5 Model**
- Optimized using **QLoRA** for parameter-efficient training.
- Domain-specific fine-tuning using the **Wealth Alpaca Dataset**.

### **2. Retrieval-Augmented Generation (RAG)**
- **Naive RAG:**
  - Simple and efficient retrieval-based answer generation.
- **Agentic RAG:**
  - Enhanced reasoning for complex financial tasks.
- **Fine-Tuned RAG:**
  - Retrieval-based response generation using the QLoRA-tuned Qwen 2.5 model.

### **3. Interactive Inference**
- Real-time question-answering powered by the fine-tuned model and RAG.

---

---

## **Features**

### **1. QLoRA Fine-Tuning**
- **Memory Efficiency:**
  - Used **4-bit quantization** for training on consumer GPUs, significantly reducing memory requirements.
- **Task-Specific Optimization:**
  - Fine-tuned specific attention layers for financial Q&A tasks.
- **Domain Adaptation:**
  - Trained on the **Wealth Alpaca Dataset** to enhance the model's understanding of the financial domain.

### **2. Retrieval-Augmented Generation (RAG)**
- **Naive RAG:** 
  - Simple retrieval-based answer generation.
- **Agentic RAG:** 
  - Enhanced decision-making for complex financial tasks.
- **Fine-Tuned RAG:** 
  - Retrieval-based answer generation using the fine-tuned Qwen 2.5 model with databases such as **Wealth Alpaca** and **Financial QA 10k**.

### **3. Interactive Inference**
- Real-time question-answering using the fine-tuned model, delivering precise and actionable insights for financial queries.

---

## **Installation**

You can set up the GenAI Financial Agent by running these commands:

```bash
git clone https://github.com/Ojaswa-Yadav/GenAI-Financial-Agent-v2.git
cd GenAI-Financial-Agent-v2
pip install -r requirements.txt
```

```bash

GenAI-Financial-Agent-v2/
├── data/                                    # Financial datasets for training and inference
│   ├── Combining_data                       # Raw datasets combined together
├── models/                                  # Pre-trained and fine-tuned models
│   ├── agentic_rag                          # RAG with gpt agents
│   ├── fine_tuning                          # Fine tuned Qwen 2.5 3B model 
│   └── rag                                  # Naive Rag
│   └── create-embedding                     # embedding for Rag
│   └── inference                            # Script for real-time inference
├── .gitignore                               # Ignored files for Git
├── .Licesne                                 # MIT Licesne
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies
