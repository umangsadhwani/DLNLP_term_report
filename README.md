# NLP Applications Using Various Large Language Models (LLMs)

## Overview

This repository provides an analysis of Large Language Models (LLMs) for key NLP tasks: **Document Classification**, **Text Summarization**, and **Question Answering**. It compares encoder-only models (e.g., BERT, DistilBERT) and encoder-decoder models (e.g., T5, BART), highlighting their performance, strengths, and trade-offs.

---

## Key Highlights

### Document Classification
- **Models**: `BERT`, `DistilBERT`
- **Datasets**: 
  - **AG News**: News articles classified into four categories.
  - **20 Newsgroups**: 20,000 documents spanning 20 topics.
- **Metrics**: F1 Score
- **Findings**:
  - DistilBERT is efficient and competitive for simpler tasks like AG News.
  - BERT outperforms DistilBERT on complex datasets like 20 Newsgroups.

### Text Summarization
- **Models**: `BART`, `T5`
- **Datasets**:
  - **CNN/Daily Mail**: News summarization.
  - **XSUM**: Abstractive summaries.
- **Metrics**: ROUGE and BERT-based metrics.
- **Findings**:
  - T5 excels in concise, precise summarization.
  - BART performs better on abstractive tasks with high semantic fidelity.

### Question Answering
- **Models**: `BERT`, `T5`
- **Datasets**:
  - **SQuAD v1**: Dataset with only answerable questions.
  - **SQuAD v2**: Dataset mixing answerable and unanswerable questions.
- **Metrics**: Exact Match (EM) and F1 Score.
- **Findings**:
  - BERT is highly accurate for answerable questions.
  - T5 handles ambiguous, unanswerable questions more effectively.

---

## Experimental Setup

- **Fine-Tuning**: Models fine-tuned using AdamW optimizer for 3 to 5 epochs.
- **Evaluation Metrics**: 
  - F1 Score (Classification)
  - ROUGE and BERT-based metrics (Summarization)
  - Exact Match (EM) and F1 Score (QA)

---

## Analysis

1. **Task Complexity**:
   - Simpler tasks benefit from models like DistilBERT (efficiency focus).
   - Complex tasks require richer representations (e.g., BERT).

2. **Abstractive vs. Extractive Tasks**:
   - T5's encoder-decoder structure is ideal for abstractive summarization.
   - BERT's token-level focus aligns with extractive tasks.

3. **Model Trade-Offs**:
   - DistilBERT offers speed and efficiency but struggles with nuance.
   - BART emphasizes semantic relevance, while T5 ensures concise outputs.

---

## Conclusion

This study highlights the importance of selecting LLM architectures based on task requirements:
- Encoder-only models (e.g., BERT) are robust for classification and extractive tasks.
- Encoder-decoder models (e.g., T5, BART) are versatile for generative and abstractive tasks.

As LLMs evolve, they promise to address complex challenges and redefine human-machine interactions in language processing.

---

## Usage

### Prerequisites
- Python 3.8+
- PyTorch or TensorFlow
- Hugging Face Transformers

   
