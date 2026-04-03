"""
RAG knowledge base for GST domain expertise.

Modules:
- gst_knowledge: Curated knowledge corpus (12 documents)
- vector_store: TF-IDF + BM25 hybrid vector store with score thresholds
- rag_engine: Full retrieval-augmented generation pipeline
- query_processor: GST domain synonym expansion and query rewriting
- chunker: Document splitting with overlap for scalable indexing
- faithfulness: Post-generation grounding assertions
- eval_rag: Retrieval quality evaluation harness
"""
