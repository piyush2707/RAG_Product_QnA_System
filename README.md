# 🚀 RAG Product Question-Answering System (Q&A)

## 💡 Project Goal
This project demonstrates the implementation of a **Retrieval-Augmented Generation (RAG)** pipeline to build a factual and source-verified Q&A system for proprietary documents, such as **Product Manuals**. This addresses the common business problem of providing accurate, real-time customer support beyond static FAQs.

## ⚙️ Architecture: How It Works
1.  **Ingestion (`1_ingest_data.py`):** PDF/Text manuals are loaded, split into chunks, converted into numerical embeddings using `all-MiniLM-L6-v2`, and stored in a persistent **ChromaDB** vector store.
2.  **Retrieval & Generation (`2_run_query.py`):** A user query is received, its embedding is generated, and the vector store is searched for the top 3 most semantically similar chunks (**Retrieval**). These chunks are then passed as context to a **Mistral LLM** alongside the original question.
3.  **Deployment (Phase 3):** The entire system is exposed as a scalable REST API using **FastAPI** and containerized with **Docker** for production readiness.



## 🛠️ Tech Stack
* **Orchestration:** LangChain
* **LLM Provider:** Mistral AI (using `mistral-large-latest`)
* **Vector Database:** ChromaDB
* **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace)
* **Deployment:** FastAPI, Uvicorn, Docker
* **Language:** Python

## 💻 Setup and Run Guide

### Prerequisites
1.  Python 3.8+
2.  Set Environment Variable: `export MISTRAL_API_KEY="YOUR_KEY"`

### Steps
1.  **Clone the Repository:**
    ```bash
    git clone [YOUR-REPO-URL]
    cd RAG_Product_QnA_System
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Add Data:** Place your sample product manuals (PDFs) in the `./data/` directory.
4.  **1. Ingest Data (Build Index):**
    ```bash
    python scripts/1_ingest_data.py
    ```
5.  **2. Run Query (Test RAG):**
    ```bash
    python scripts/2_run_query.py
    ```
    *This will print the generated answer and the source document snippets.*

## 📈 Next Steps (MLOps Focus)
* Containerize the application using a `Dockerfile`.
* Implement a `/query` endpoint using FastAPI.
* Add Unit Tests for ingestion and retrieval accuracy.
  
