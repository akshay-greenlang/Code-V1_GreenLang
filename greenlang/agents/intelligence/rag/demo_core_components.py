# -*- coding: utf-8 -*-
"""
Demo script for INTL-104 RAG v1 core components.

This script demonstrates the usage of:
1. TokenAwareChunker - Token-based document chunking
2. EmbeddingProvider - Embedding generation (MiniLM)
3. VectorStoreProvider - FAISS vector storage
4. MMRRetriever - Maximal Marginal Relevance retrieval

Usage:
    python -m greenlang.intelligence.rag.demo_core_components
"""

import asyncio
import numpy as np
from pathlib import Path

from greenlang.agents.intelligence.rag.config import RAGConfig
from greenlang.agents.intelligence.rag.models import DocMeta, Chunk
from greenlang.agents.intelligence.rag.chunker import get_chunker
from greenlang.agents.intelligence.rag.embeddings import get_embedding_provider
from greenlang.agents.intelligence.rag.vector_stores import get_vector_store, Document
from greenlang.agents.intelligence.rag.retrievers import get_retriever
from greenlang.agents.intelligence.rag.hashing import sha256_str, file_hash
from datetime import date


async def demo_core_components():
    """
    Demonstrate RAG core components with a simple example.
    """
    print("=" * 70)
    print("INTL-104 RAG v1 Core Components Demo")
    print("=" * 70)
    print()

    # Step 1: Configuration
    print("[1/6] Configuring RAG system...")
    config = RAGConfig(
        mode="replay",  # Deterministic mode
        embedding_provider="minilm",
        vector_store_provider="faiss",
        retrieval_method="mmr",
        chunk_size=128,  # Small chunks for demo
        chunk_overlap=16,
        default_top_k=3,
        default_fetch_k=10,
        mmr_lambda=0.5,
    )
    print(f"  - Mode: {config.mode}")
    print(f"  - Embedding: {config.embedding_provider}")
    print(f"  - Vector store: {config.vector_store_provider}")
    print(f"  - Retrieval: {config.retrieval_method}")
    print()

    # Step 2: Document chunking
    print("[2/6] Chunking sample documents...")
    chunker = get_chunker(config)

    # Sample climate-related documents
    documents = [
        {
            "doc_id": "ghg-protocol-001",
            "title": "GHG Protocol Corporate Standard",
            "collection": "ghg_protocol_corp",
            "text": (
                "The GHG Protocol Corporate Accounting and Reporting Standard "
                "provides requirements and guidance for companies preparing a "
                "corporate-level GHG emissions inventory. Scope 1 emissions are "
                "direct emissions from owned or controlled sources. Scope 2 emissions "
                "are indirect emissions from the generation of purchased energy. "
                "Scope 3 emissions are all other indirect emissions that occur in "
                "the value chain."
            ),
            "section_path": "Chapter 4 > Scope Definitions",
        },
        {
            "doc_id": "ipcc-ar6-001",
            "title": "IPCC AR6 Working Group 3",
            "collection": "ipcc_ar6_wg3",
            "text": (
                "Climate change mitigation refers to efforts to reduce or prevent "
                "emission of greenhouse gases. This can be achieved through new "
                "technologies and renewable energies, making older equipment more "
                "energy efficient, or changing management practices or consumer behavior. "
                "Carbon capture and storage technologies can capture emissions from power "
                "plants and industrial facilities."
            ),
            "section_path": "Chapter 1 > Mitigation Strategies",
        },
        {
            "doc_id": "ghg-protocol-002",
            "title": "GHG Protocol Scope 3 Standard",
            "collection": "ghg_protocol_scope3",
            "text": (
                "Scope 3 emissions often represent the majority of an organization's "
                "total GHG emissions. These include upstream emissions from purchased "
                "goods and services, business travel, employee commuting, and downstream "
                "emissions from product use and end-of-life treatment. Companies should "
                "prioritize Scope 3 categories based on size, influence, risk, and stakeholder "
                "expectations."
            ),
            "section_path": "Chapter 2 > Scope 3 Categories",
        },
    ]

    all_chunks = []
    for doc_data in documents:
        chunks = chunker.chunk_document(
            text=doc_data["text"],
            doc_id=doc_data["doc_id"],
            section_path=doc_data["section_path"],
            extra={"collection": doc_data["collection"]},
        )
        all_chunks.extend(chunks)
        print(f"  - {doc_data['title']}: {len(chunks)} chunks")

    print(f"  Total chunks: {len(all_chunks)}")
    print()

    # Step 3: Embedding generation
    print("[3/6] Generating embeddings...")
    embedder = get_embedding_provider(config)

    # Extract chunk texts
    chunk_texts = [chunk.text for chunk in all_chunks]

    # Generate embeddings
    embeddings = await embedder.embed(chunk_texts)
    print(f"  - Generated {len(embeddings)} embeddings")
    print(f"  - Embedding dimension: {embedder.dim}")
    print()

    # Step 4: Vector store indexing
    print("[4/6] Indexing documents in vector store...")
    vector_store = get_vector_store(dimension=embedder.dim, config=config)

    # Group documents by collection
    collections = {}
    for chunk, embedding in zip(all_chunks, embeddings):
        collection = chunk.extra.get("collection", "unknown")
        if collection not in collections:
            collections[collection] = []
        doc = Document(chunk=chunk, embedding=embedding)
        collections[collection].append(doc)

    # Add to vector store
    for collection, docs in collections.items():
        vector_store.add_documents(docs, collection)
        print(f"  - Added {len(docs)} docs to '{collection}'")

    print(f"  Total indexed: {len(all_chunks)} documents")
    print()

    # Step 5: Query and retrieval
    print("[5/6] Performing MMR retrieval...")
    retriever = get_retriever(
        vector_store=vector_store,
        retrieval_method="mmr",
        fetch_k=config.default_fetch_k,
        top_k=config.default_top_k,
        lambda_mult=config.mmr_lambda,
    )

    # Test query
    query = "What are Scope 3 emissions?"
    print(f"  Query: '{query}'")

    # Embed query
    query_embeddings = await embedder.embed([query])
    query_embedding = query_embeddings[0]

    # Retrieve
    results = retriever.retrieve(
        query_embedding=query_embedding,
        collections=["ghg_protocol_corp", "ghg_protocol_scope3"],
    )

    print(f"  Retrieved {len(results)} documents:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n  Result {i} (MMR score: {score:.3f}):")
        print(f"    - Doc: {doc.chunk.doc_id}")
        print(f"    - Section: {doc.chunk.section_path}")
        print(f"    - Text preview: {doc.chunk.text[:100]}...")

    print()

    # Step 6: Vector store persistence
    print("[6/6] Testing vector store persistence...")
    temp_path = Path("./temp_rag_demo")
    vector_store.save(temp_path)
    print(f"  - Saved vector store to {temp_path}")

    # Load in new instance
    new_vector_store = get_vector_store(dimension=embedder.dim, config=config)
    new_vector_store.load(temp_path)
    print(f"  - Loaded vector store from {temp_path}")

    # Verify stats match
    original_stats = vector_store.get_stats()
    loaded_stats = new_vector_store.get_stats()
    print(f"  - Original: {original_stats['total_documents']} docs")
    print(f"  - Loaded: {loaded_stats['total_documents']} docs")

    # Clean up
    import shutil
    shutil.rmtree(temp_path)
    print(f"  - Cleaned up temp directory")

    print()
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - Chunked {len(documents)} documents into {len(all_chunks)} chunks")
    print(f"  - Generated {len(embeddings)} {embedder.dim}-dimensional embeddings")
    print(f"  - Indexed across {len(collections)} collections")
    print(f"  - Retrieved top-{config.default_top_k} results using MMR")
    print(f"  - Verified vector store persistence")


if __name__ == "__main__":
    asyncio.run(demo_core_components())
