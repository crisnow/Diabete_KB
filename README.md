# Diabetes Knowledge Bot (RAG Prototype)

## Overview

This project demonstrates a **Retrieval-Augmented Generation (RAG)** system for answering questions about diabetes management. It leverages authoritative sources, automatic text extraction, and embeddings to provide accurate, context-aware responses.

The goal is to showcase **end-to-end machine learning engineering** skills including:

- Data collection & preprocessing
- Text chunking & embedding
- Vector retrieval
- LLM-based question answering

This project is designed as a **minimal viable prototype** that can be expanded into a full medical advisory tool.

---

## Features

- **Web & PDF data ingestion**: Automatically extracts text from authoritative sources (e.g., ADA, WHO, CDC).
- **Automatic chunking**: Breaks long documents into contextually relevant chunks with overlap for better retrieval.
- **Embeddings & vector search**: Uses OpenAI embeddings and FAISS for fast similarity search.
- **Simple chat interface**: Ask questions about diabetes and get context-aware answers from the RAG system.
- **Extensible & reproducible**: Easily add more documents or datasets to expand knowledge coverage.

---

## Tech Stack

| Layer                         | Tools / Libraries               |
| ----------------------------- | ------------------------------- |
| Data Extraction               | Python, Requests, BeautifulSoup |
| Text Preprocessing & Chunking | NLTK, Python                    |
| Embeddings & Retrieval        | OpenAI Embeddings, FAISS        |
| LLM / Question Answering      | OpenAI GPT-4                    |
| Optional UI                   | Streamlit / Command Line        |

---

## Usage

1. **Extract text from web pages / PDFs**

```bash
python extract_text.py
```
