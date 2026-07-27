# 🏦 CrediTrust Financial: Intelligent Complaint Analysis RAG Agent

[![CI Test Suite](https://github.com/lydiab04/rag-complaint-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/lydiab04/rag-complaint-chatbot/actions)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit%20%7C%20ChromaDB-orange)

An enterprise-grade Retrieval-Augmented Generation (RAG) agent designed to automate regulatory compliance audits and extract actionable consumer insights from unstructured Consumer Financial Protection Bureau (CFPB) complaints.

---

## 📐 System Architecture

```text
                            +-------------------+
                            |   CFPB Dataset    |
                            | (Unstructured)    |
                            +---------+---------+
                                      |
                                      v
+------------------+             +-------------------+
| User Query       |             | Preprocessing &   |
| (Streamlit UI)   |             | Dense Embeddings  |
+--------+---------+             | (all-MiniLM-L6-v2)|
         |                       +---------+---------+
         |                                 |
         |                                 v
         |                 +---------------------------+
         |                 | Persistent ChromaDB Store |
         |                 +-------------+-------------+
         |                               ^
         |                               |
         +-------------------------------+
                         Vector Search
                       (Top-K Chunks)
                                 |
                                 v
                     +-------------------+
                     |   RAG Engine      |
                     |   (distilgpt2)    |
                     +---------+---------+
                               |
                               v
                     +-------------------+
                     | Grounded Answer   |
                     | + Source Context  |
                     +-------------------+
```

---

## 🛠️ Tech Stack & Key Components

- **Vector Search & Storage:** ChromaDB with `all-MiniLM-L6-v2` dense embeddings.
- **Generative Language Model:** Hugging Face Transformers (`distilbert/distilgpt2`).
- **Frontend Dashboard:** Interactive Streamlit web interface with product filtering and execution metrics.
- **Testing & Quality Assurance:** `pytest` suite featuring mocked vector/LLM fixtures and automated GitHub Actions CI/CD workflows.

---

## ⚡ Quickstart Guide

### 1. Prerequisites

Ensure you have **Python 3.10 or higher** installed.

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/lydiab04/rag-complaint-chatbot.git
cd rag-complaint-chatbot

python -m venv .venv

# Activate on Windows
.venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Unit Tests

Verify your local setup using `pytest`:

```bash
python -m pytest
```

### 4. Launch the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

---

## 🧪 Test Coverage & CI/CD Pipeline

Continuous Integration is powered by **GitHub Actions** (`.github/workflows/ci.yml`).

Every commit and pull request automatically triggers unit tests verifying:

- ✅ Default `RAGConfig` parameters
- ✅ Input query validation and sanitization
- ✅ RAG pipeline retrieval execution
- ✅ Grounded response generation
- ✅ End-to-end pipeline stability

---

## 📁 Project Structure

```text
rag-complaint-chatbot/
├── app.py                  # Streamlit application
├── src/                    # Core RAG implementation
├── tests/                  # Unit tests
├── chroma_db/              # Persistent vector database
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```