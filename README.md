# Hybrid RAG Technical Documentation Assistant

A **Hybrid Retrieval-Augmented Generation (RAG)** system for answering questions from technical documentation using both **semantic search** and **keyword-based search**.

The project combines **ChromaDB** for dense vector retrieval, **BM25** for sparse keyword retrieval, and **Reciprocal Rank Fusion (RRF)** to produce a stronger set of relevant document chunks. A locally running **Gemma model through Ollama** is then used to generate grounded answers with document citations.

> **Project Status:** Setup completed. RAG pipeline implementation in progress.

---

## Overview

Large technical documentation can be difficult to search using only traditional keyword matching or only semantic search.

This project aims to solve that problem using a **hybrid retrieval pipeline**:

```text
Technical Documentation
        ↓
Document Loading & Cleaning
        ↓
Chunking + Metadata
        ↓
 ┌──────────────────┐
 │ Dense Retrieval  │ → ChromaDB
 └──────────────────┘
          +
 ┌──────────────────┐
 │ Sparse Retrieval │ → BM25
 └──────────────────┘
          ↓
    RRF Rank Fusion
          ↓
  Top Relevant Chunks
          ↓
    Local Gemma LLM
          ↓
 Grounded Answer + Citations
```

The current documentation corpus is based on the **FastAPI documentation**.

---

## Key Features

### 1. Dense Semantic Search

Documents are converted into embedding vectors and stored in ChromaDB.

Dense retrieval helps find documents based on **meaning**, even when the exact keywords are not present.

Example:

```text
Query: How do I protect an API endpoint?

Possible relevant text:
"FastAPI provides OAuth2 utilities for securing API operations."
```

The exact word "protect" may not appear, but the semantic meaning is related.

---

### 2. Sparse Keyword Search

The project also uses **BM25** for traditional keyword-based retrieval.

This is especially useful for technical documentation containing:

- API names
- Function names
- Parameter names
- Error messages
- Configuration options
- Technical identifiers

For example:

```text
Query: APIRouter prefix
```

BM25 can directly identify chunks containing these exact technical terms.

---

### 3. Hybrid Retrieval

Dense retrieval and BM25 retrieval complement each other.

```text
Dense Search
    ↓
Semantic understanding

BM25
    ↓
Exact keyword matching

Both
    ↓
RRF
    ↓
Combined ranking
```

This makes retrieval more robust than relying on only one search method.

---

### 4. Reciprocal Rank Fusion (RRF)

RRF combines the ranked results from the dense and sparse retrievers.

Instead of directly comparing their scores, RRF uses the **rank position** of each document.

Conceptually:

```text
Dense Results       BM25 Results
     ↓                   ↓
  Ranking             Ranking
     └─────────┬─────────┘
               ↓
              RRF
               ↓
       Final Ranking
```

---

### 5. Local LLM

The project uses **Gemma through Ollama** for generation.

The LLM runs locally rather than relying on a cloud-based API.

This provides:

- Local inference
- No OpenAI/Anthropic API dependency
- Better control over the development environment
- Potentially improved privacy for local documentation

---

### 6. Citations

The system will attach metadata to document chunks during ingestion.

Example metadata:

```text
Source
Section
Document
Chunk ID
```

The retrieved metadata will be used to show where the answer came from.

The goal is:

```text
Question
   ↓
Retrieved documentation
   ↓
Gemma
   ↓
Answer
   +
Source citation
```

This makes the generated answer easier to verify.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| RAG Framework | LangChain |
| Dense Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Sparse Retrieval | BM25 |
| Rank Fusion | Reciprocal Rank Fusion |
| LLM | Gemma |
| Local LLM Runtime | Ollama |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Documentation | FastAPI Official Documentation |
| Version Control | Git + GitHub |

---

## Project Structure

```text
hybrid-rag-docs-assistant/
│
├── docs/
│   └── fastapi-docs/
│
├── src/
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── evaluation.py
│   └── api.py
│
├── data/
│   ├── chroma_db/
│   └── golden_qa.json
│
├── frontend/
│   └── app.py
│
├── requirements.txt
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aarushi1607/hybrid-rag-docs-assistant.git
cd hybrid-rag-docs-assistant
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install and run Ollama

Make sure Ollama is installed and running.

Pull the Gemma model:

```powershell
ollama pull gemma3:4b
```

Test it:

```powershell
ollama run gemma3:4b
```

---

## Current Development Progress

- [x] Git repository initialized
- [x] Project structure created
- [x] Python virtual environment created
- [x] FastAPI documentation added as corpus
- [x] Ollama configured
- [x] Gemma model downloaded
- [x] Python dependencies configured
- [ ] Document ingestion
- [ ] Document cleaning
- [ ] Document chunking
- [ ] Citation metadata
- [ ] ChromaDB dense retrieval
- [ ] BM25 sparse retrieval
- [ ] RRF hybrid retrieval
- [ ] RAG prompt
- [ ] Gemma generation
- [ ] Citation-based answers
- [ ] Evaluation dataset
- [ ] Retrieval evaluation
- [ ] End-to-end evaluation
- [ ] FastAPI backend
- [ ] Streamlit interface
- [ ] Dockerization

---

## Future Improvements

Possible future extensions include:

- Reranking retrieved documents
- Better citation formatting
- Query rewriting
- Retrieval evaluation metrics
- Answer faithfulness evaluation
- Streaming responses
- Conversation history
- Docker deployment
- Improved UI
- Support for additional documentation sources

---

## Learning Goals

This project is designed to provide practical experience with:

- Retrieval-Augmented Generation
- Vector databases
- Embeddings
- Semantic search
- Sparse retrieval
- BM25
- Hybrid search
- Rank fusion
- LLM application development
- Local LLM inference
- Document processing
- Evaluation of RAG systems
- FastAPI
- Streamlit
- Docker
- Git/GitHub

---

## License

This project is intended for educational and portfolio purposes.