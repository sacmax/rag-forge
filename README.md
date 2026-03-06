# RAG Forge

A production-grade RAG pipeline for business and legal documents, built incrementally across 6 modules — each adding a real production concern.

**Pipeline:** document → chunk → embed → store → (query time) guard → cache check → query transform → dense/hybrid retrieval → rerank → relevance gate → generate → scope check → response with citations.

---

## Modules

| | What it adds |
|--|--|
| M1 | PDF/DOCX/TXT ingestion, recursive chunking, dense retrieval, inline citations, Hit Rate + MRR eval |
| M2 | Semantic chunker (embedding-based splits), hybrid chunker (parent-child) |
| M3 | HyDE, MultiQuery, BM25+RRF hybrid search, FlashRank/Cohere/CrossEncoder rerankers, MMR |
| M4 | Input/relevance/output guardrails, semantic + embedding cache, circuit breaker + fallback LLM, OpenTelemetry tracing |
| M5 | FastAPI with SSE streaming, background ingestion, Qdrant cloud vector store, Docker Compose |
| M6 | Agentic layer: query routing, CRAG-style retrieval evaluation, multi-hop refinement |

---

## Benchmark

Evaluated on 12 golden QA pairs from a college admission brochure (k=5, top_n=3):

| Strategy | Hit Rate | MRR |
|----------|:---:|:---:|
| Dense (baseline) | 0.833 | 0.700 |
| Dense + FlashRank | 0.750 | 0.708 |
| Hybrid BM25+RRF | 0.833 | 0.700 |
| HyDE | 0.750 | 0.646 |

Reranking improves ordering (MRR) but reduces hit rate when `top_n < k` cuts borderline chunks. HyDE and hybrid retrieval are expected to outperform dense on larger, multi-document corpora.

---

## Stack

| Library | Used for |
|---------|----------|
| `litellm` | LLM calls with retry, fallback, and provider abstraction |
| `chromadb` | Local vector store with cosine similarity search |
| `qdrant-client` | Cloud vector store (M5) |
| `sentence-transformers` | Local embedding models and cross-encoder reranker |
| `flashrank` | Fast local reranker |
| `cohere` | API reranker |
| `rank-bm25` | BM25 keyword search for hybrid retrieval |
| `fastapi` + `uvicorn` | REST API with SSE streaming |
| `opentelemetry` + `arize-phoenix` | Distributed tracing and observability |
| `structlog` | Structured JSON logging |
| `tenacity` | Retry with exponential backoff on LLM calls |
