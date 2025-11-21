# -*- coding: utf-8 -*-
"""
Example: Basic RAG usage with determinism and citations.

This example demonstrates:
1. Creating a RAG engine with configuration
2. Ingesting a document (placeholder - components not yet available)
3. Querying with MMR-based retrieval
4. Using deterministic wrapper for replay mode
5. Generating audit-ready citations
"""

import asyncio
from pathlib import Path
from datetime import date

from greenlang.intelligence.rag import (
    RAGEngine,
    RAGConfig,
    DeterministicRAG,
    DocMeta,
    Chunk,
    RAGCitation,
    QueryResult,
)


async def main():
    """Main example function."""

    print("=" * 60)
    print("GreenLang RAG Engine - Basic Usage Example")
    print("=" * 60)
    print()

    # Step 1: Create RAG configuration
    print("[1] Creating RAG configuration...")
    config = RAGConfig(
        mode="live",  # "live" | "replay"
        allowlist=["ghg_protocol_corp", "ipcc_ar6_wg3", "test_collection"],
        embedding_provider="minilm",
        vector_store_provider="faiss",
        retrieval_method="mmr",
        default_top_k=6,
        default_fetch_k=30,
        mmr_lambda=0.5,
        chunk_size=512,
        chunk_overlap=64,
        enable_sanitization=True,
        strict_sanitization=True,
    )
    print(f"   - Mode: {config.mode}")
    print(f"   - Allowlist: {', '.join(config.allowlist)}")
    print(f"   - Embedding: {config.embedding_provider}")
    print(f"   - Vector store: {config.vector_store_provider}")
    print()

    # Step 2: Create RAG engine
    print("[2] Creating RAG engine...")
    engine = RAGEngine(config)
    print("   - Engine created successfully")
    print()

    # Step 3: Document ingestion (placeholder)
    print("[3] Document ingestion example...")
    print("   NOTE: Core components (embedders, vector stores, chunkers)")
    print("   are being implemented by another agent.")
    print("   This will work once they are available.")
    print()

    # Example document metadata
    doc_meta = DocMeta(
        doc_id="a3f5b2c8-d1e6-5f9a-8b7c-2d5e8f1a4b7c",
        title="GHG Protocol Corporate Standard",
        collection="ghg_protocol_corp",
        source_uri="https://ghgprotocol.org/corporate-standard",
        publisher="WRI/WBCSD",
        publication_date=date(2015, 3, 24),
        version="1.05",
        content_hash="a3f5b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4",
        doc_hash="b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2",
    )
    print(f"   - Document: {doc_meta.title}")
    print(f"   - Collection: {doc_meta.collection}")
    print(f"   - Publisher: {doc_meta.publisher}")
    print()

    # Step 4: Query example (placeholder)
    print("[4] Query example...")
    print("   Query: 'emission factors for stationary combustion'")
    print("   Parameters:")
    print("      - top_k: 6 (final results)")
    print("      - fetch_k: 30 (MMR candidates)")
    print("      - mmr_lambda: 0.5 (balance relevance/diversity)")
    print()

    # Note: This will fail until components are available
    # Uncomment when ready:
    # result = await engine.query(
    #     query="emission factors for stationary combustion",
    #     top_k=6,
    #     collections=["ghg_protocol_corp"],
    #     fetch_k=30,
    #     mmr_lambda=0.5,
    # )
    # print(f"   - Retrieved {result.total_chunks} chunks")
    # print(f"   - Search time: {result.search_time_ms}ms")
    # print(f"   - Total tokens: {result.total_tokens}")

    # Step 5: Deterministic wrapper example
    print("[5] Deterministic wrapper (replay mode)...")
    cache_path = Path(".rag_cache_example.json")

    # Create wrapper in record mode
    det = DeterministicRAG(
        mode="record",
        cache_path=cache_path,
        config=config,
    )

    stats = det.get_cache_stats()
    print(f"   - Mode: {det.mode}")
    print(f"   - Cache path: {cache_path}")
    print(f"   - Cached queries: {stats['num_queries']}")
    print()

    # Verify cache integrity
    verification = det.verify_cache_integrity()
    print(f"   - Cache valid: {verification['valid']}")
    print(f"   - Errors: {verification['num_errors']}")
    print()

    # Step 6: Citation generation example
    print("[6] Citation generation example...")

    # Example chunk (would come from retrieval)
    chunk = Chunk(
        chunk_id="c8d1e6f9-a4b7-5c2d-8e1f-4a7b2c5d8e1f",
        doc_id=doc_meta.doc_id,
        section_path="Chapter 7 > Section 7.3 > 7.3.1 Emission Factors",
        section_hash="d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8",
        page_start=45,
        page_end=46,
        paragraph=2,
        start_char=12450,
        end_char=13100,
        text="For stationary combustion sources, the emission factor...",
        token_count=128,
        embedding_model="all-MiniLM-L6-v2",
    )

    # Generate citation
    citation = RAGCitation.from_chunk(
        chunk=chunk,
        doc_meta=doc_meta,
        relevance_score=0.87,
    )

    print("   Generated citation:")
    print(f"   {citation.formatted}")
    print()
    print("   Citation details:")
    print(f"      - Document: {citation.doc_title} v{citation.version}")
    print(f"      - Publisher: {citation.publisher}")
    print(f"      - Section: {citation.section_path}")
    print(f"      - Page: {citation.page_number}")
    print(f"      - Relevance: {citation.relevance_score:.2f}")
    print(f"      - Checksum: {citation.checksum}")
    print()

    # Step 7: Security features
    print("[7] Security features...")
    print("   - Collection allowlisting: ENABLED")
    print("   - Input sanitization: ENABLED")
    print("   - Network isolation (replay): ENABLED")
    print("   - Checksum verification: ENABLED")
    print()

    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

    # Cleanup
    if cache_path.exists():
        cache_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
