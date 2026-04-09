"""
GST Knowledge Module v4 — production-grade RAG with 14 improvements.

Components:
- gst_knowledge: curated GST domain knowledge corpus
- chunker: sentence-aware chunking with contextual headers
- vector_store: TF-IDF/BM25 hybrid with inverted indices + caching
- query_processor: GST synonym expansion and stop word removal
- query_router: auto-routing, semantic cache, RAG-fusion, HyDE, GraphRAG, Self-RAG
- rag_engine: orchestrates all components into a retrieval pipeline
- eval_rag: evaluation harness with MRR/NDCG metrics
- faithfulness: grounding checker for LLM responses
"""
