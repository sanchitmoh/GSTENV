"""
RAG knowledge base for GST domain expertise.

Modules:
- gst_knowledge: Curated knowledge corpus (30 documents covering ITC, e-invoicing,
  registration, returns, penalties, RCM, composition, blocked credits, TDS/TCS,
  place of supply, exports, anti-profiteering, ISD, e-way bill, audit, and more)
- vector_store: TF-IDF + BM25 hybrid vector store with sublinear TF, re-ranking,
  and hierarchical parent-child retrieval
- rag_engine: Full retrieval-augmented generation pipeline with contextual chunking,
  sentence-window retrieval, and re-ranking
- query_processor: GST domain synonym expansion (55+ mappings) and query rewriting
- chunker: Sentence-aware splitting with contextual headers, sliding windows,
  configurable strategy, and sentence-window mode
- faithfulness: Post-generation grounding assertions (legal + numeric claims)
- eval_rag: Retrieval quality evaluation harness (20 ground-truth queries)
"""
