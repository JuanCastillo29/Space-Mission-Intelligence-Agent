# Space Mission Intelligence Agent

**RAG System for ESA/NASA Documentation**

A retrieval-augmented generation (RAG) system that enables engineers, researchers, and analysts to query and reason across heterogeneous space-domain documentation. Given a natural language question, the system retrieves relevant information from ESA/NASA mission reports, space weather bulletins, orbital data catalogues, and technical standards, then generates a cited, contextual answer grounded in the retrieved sources.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Chunking Strategy](#chunking-strategy)
- [Embedding & Vector Store](#embedding--vector-store)
- [Retrieval & Reranking](#retrieval--reranking)
- [Generation Layer](#generation-layer)
- [Evaluation Framework](#evaluation-framework)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Overview

The Space Mission Intelligence Agent is a production-grade RAG pipeline that demonstrates end-to-end GenAI engineering. It goes from data ingestion and hybrid search to LLM-grounded answer generation with inline citations and rigorous evaluation.

### Key Features

- **Hybrid search** combining semantic similarity (HNSW) with PostgreSQL full-text search, fused via Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** with diversity filtering (MMR) to maximise context quality
- **Inline citation enforcement** — every factual claim traces back to a source document
- **Query routing** — automatically classifies queries as retrieval, structured-data, or hybrid
- **Quantified evaluation** via RAGAS metrics, custom citation accuracy, and ablation studies
- **Containerised architecture** with Docker Compose orchestration

### Strategic Objectives

This project serves as both a functional tool for space-domain knowledge retrieval and a portfolio project demonstrating production-grade GenAI engineering skills. The architecture is designed to be domain-portable — swap the data layer and prompts to target defence/cybersecurity or finance with minimal pipeline changes.

---

## Architecture

The system is organised into four independently containerised services:

| Service | Responsibility | Technology | Port |
|---------|---------------|------------|------|
| **API Server** | Query processing, retrieval orchestration, LLM calls | FastAPI, Python 3.11+ | 8000 |
| **Database** | Vector storage, relational data, full-text index | PostgreSQL 16 + pgvector | 5432 |
| **Ingestion Worker** | Document parsing, chunking, embedding, indexing | Python, Celery (optional) | — |
| **Frontend** | User interface, query input, answer display | Streamlit | 8501 |

### Pipeline Flow

```
User Query
    │
    ▼
┌──────────────┐
│ Query Router  │──── Structured query? ──▶ PostgreSQL (SQL)
└──────┬───────┘
       │ Retrieval query
       ▼
┌──────────────┐     ┌──────────────┐
│ Semantic Search│    │ Keyword Search│
│  (pgvector)   │    │ (tsvector)    │
└──────┬───────┘     └──────┬───────┘
       │                     │
       └──────┬──────────────┘
              ▼
     ┌────────────────┐
     │ RRF Fusion     │  (top 10 candidates)
     └───────┬────────┘
             ▼
     ┌────────────────┐
     │ Cross-Encoder  │  bge-reranker-v2-m3
     │ Reranking      │
     └───────┬────────┘
             ▼
     ┌────────────────┐
     │ MMR Diversity  │  (top 5 chunks)
     │ Filtering      │
     └───────┬────────┘
             ▼
     ┌────────────────┐
     │ LLM Generation │  Llama 3.1 70B (Groq)
     │ + Citations    │
     └───────┬────────┘
             ▼
     Cited Answer + Sources
```

---

## Data Sources

All sources are publicly available with no licensing restrictions. The system ingests three categories of data:

| Source | Format | Content | Volume Est. |
|--------|--------|---------|-------------|
| ESA Mission Reports | PDF, HTML | Mission descriptions, results, technical summaries | 200–500 docs |
| NASA NTRS | PDF | Technical reports, research papers | 500–1,000 docs |
| ECSS Standards | PDF | European space engineering standards | 50–100 docs |
| NOAA SWPC | JSON, Text | Space weather alerts, solar activity | Daily feeds |
| ESA Space Debris Office | CSV, JSON | Orbital debris statistics, conjunction data | Periodic reports |
| CelesTrak | TLE, JSON | Active satellite catalogue, orbital parameters | ~10,000 objects |

### Ingestion Pipeline

The ingestion pipeline is **idempotent** (re-running does not create duplicates) and preserves full provenance metadata for every chunk.

- **PDFs** — Parsed with PyMuPDF (fitz); scanned PDFs fall back to Tesseract OCR
- **HTML** — BeautifulSoup with content-focused extraction (navigation and boilerplate stripped)
- **JSON/XML** — Space weather bulletins converted to natural language summaries with metadata tags
- **TLE/CSV** — Structured data stored in PostgreSQL as relational tables, not in the vector store

---

## Chunking Strategy

| Parameter | Default | Experiment Range |
|-----------|---------|-----------------|
| Chunk size | 1,024 tokens | 512 / 1,024 / 2,048 |
| Chunk overlap | 128 tokens | 64 / 128 / 256 |
| Split boundaries | Section headers → paragraphs → sentences | — |

### Metadata Enrichment

Every chunk carries: `source_id`, `source_type`, `mission_name`, `section_path`, `date`, and `chunk_index` (for adjacent-chunk expansion).

### Special Handling

- **Tables** — extracted separately with column headers preserved; a text summary is prepended
- **Equations** — LaTeX preserved inline with a plain-language description appended via a lightweight LLM pass
- **Cross-references** — internal references resolved during chunking; referenced section titles appended as metadata

---

## Embedding & Vector Store

### Embedding Model

| Model | Dimensions | Cost | Decision |
|-------|-----------|------|----------|
| **bge-large-en-v1.5** | 1,024 | Free (local) | ✅ Primary |
| text-embedding-3-small | 1,536 | $0.02/1M tokens | Benchmark comparison |
| nomic-embed-text-v1.5 | 768 | Free (local) | Fallback |

The primary model is benchmarked against `text-embedding-3-small` on a 50-query retrieval evaluation set before committing to full indexing.

### pgvector

PostgreSQL with the pgvector extension serves as the unified vector and relational store.

**Schema:** three core tables — `documents` (source metadata), `chunks` (text + embeddings), `satellites` (structured orbital data).

**Indexing:** HNSW with `m=16`, `ef_construction=200` for sub-linear search with high recall across the 10K–50K chunk corpus.

### Hybrid Search

Semantic search alone misses exact technical terms and acronyms. The system fuses two retrieval signals:

1. **Semantic** — cosine similarity over HNSW index → top 20
2. **Keyword** — PostgreSQL full-text search (tsvector/tsquery) with a custom space-terminology dictionary → top 20
3. **Fusion** — Reciprocal Rank Fusion (RRF) → top 10 candidates passed to the reranker

---

## Retrieval & Reranking

### Two-Stage Pipeline

1. **Reranker** — `bge-reranker-v2-m3` (BAAI) scores each (query, chunk) pair via cross-encoder. Input: 10 candidates → Output: top 5.
2. **MMR Diversity Filtering** — penalises chunks too similar to already-selected ones, ensuring coverage across multiple documents and perspectives.

### Context Window Assembly

Each chunk in the final context includes source attribution (title, section, date), preserved formatting, and a numeric reference tag (`[1]`, `[2]`, …) for inline citations. Adjacent chunks can be expanded when the reranker confidence threshold indicates broader context is needed.

---

## Generation Layer

### LLM

| Model | Provider | Rationale |
|-------|----------|-----------|
| **Llama 3.1 70B** | Groq | Fast inference, strong instruction following, free tier (500K tokens/day) |
| Mistral Large | Mistral API | Fallback for European/multilingual content |

### Citation Mechanism

The system prompt enforces strict citation discipline — every factual claim must reference a numbered source. Post-processing validates that all citation tags map to real sources and strips hallucinated references. The final response includes a **Sources** section with full document title, URL, and section path.

### Query Routing

Queries are classified before pipeline selection:

- **Retrieval** — specific information from the document corpus → full RAG pipeline
- **Structured data** — answerable from PostgreSQL (orbital params, catalogues) → SQL query
- **Hybrid** — needs both document retrieval and structured data

---

## Evaluation Framework

### Test Set

50 manually curated question-answer pairs stratified across query types:

| Category | Count | Difficulty |
|----------|-------|-----------|
| Single-document factual | 15 | Easy |
| Cross-document comparison | 10 | Medium |
| Multi-source reasoning | 10 | Hard |
| Structured data | 10 | Easy–Medium |
| Unanswerable (control) | 5 | — |

### Metrics

**Retrieval:** Precision@k (k=3, 5), Recall@k (k=5, 10), MRR

**Generation (RAGAS):**

| Metric | Target |
|--------|--------|
| Faithfulness | > 0.85 |
| Answer Relevancy | > 0.80 |
| Context Precision | > 0.75 |
| Context Recall | > 0.70 |

**Custom:** Citation accuracy, hallucination rate, unanswerable detection rate

### Ablation Studies

- Semantic-only vs. hybrid search
- With vs. without cross-encoder reranking
- Chunk size: 512 vs. 1,024 vs. 2,048 tokens
- With vs. without metadata filtering

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Submit a question; returns answer with citations |
| `POST` | `/api/v1/ingest` | Ingest a new document into the system |
| `GET` | `/api/v1/documents` | List all ingested documents with metadata |
| `GET` | `/api/v1/health` | Health check (status + DB connectivity) |
| `POST` | `/api/v1/evaluate` | Run evaluation suite; returns metrics JSON |

---

## Project Structure

```
space-intelligence-agent/
├── app/                  # FastAPI application (routes, middleware, config)
├── ingestion/            # Document parsers, chunking, embedding pipeline
├── retrieval/            # Hybrid search, reranking, context assembly
├── generation/           # LLM integration, prompt templates, citation post-processing
├── evaluation/           # Test set, metrics computation, ablation scripts
├── frontend/             # Streamlit application
├── db/                   # Database migrations, schema definitions
├── scripts/              # Data download, preprocessing utilities
├── tests/                # Unit and integration tests
├── docker/               # Dockerfiles for each service
├── docker-compose.yml    # Full stack orchestration
├── Makefile              # Shortcuts: make ingest, make query, make evaluate, make test
├── .env.example          # Environment variable template
└── README.md
```

---

## Roadmap

| Phase | Focus |
|-------|-------|
| **Phase 1** ✅ | RAG pipeline, hybrid search, evaluation, deployment |
| **Phase 2** | Fine-tuning embedding and generation models on space-domain data |
| **Phase 3** | Agentic capabilities (multi-step reasoning, tool use) |
| **Phase 4** | Multimodal processing (diagrams, charts, imagery) |

---

## Author

**Juan Castillo** — ML Engineer | Deep Learning | MLOps

*Physics graduate → ML Engineer (anomaly detection, cybersecurity) → Intelligent systems for the space/defence sector.*
