# -*- coding: utf-8 -*-
"""
Example: Using Weaviate as vector store for production RAG.

This example demonstrates:
1. Starting Weaviate with Docker Compose
2. Configuring RAG to use Weaviate
3. Adding documents to Weaviate
4. Performing similarity search
5. Collection filtering
6. Health checks and stats

Prerequisites:
- Docker and Docker Compose installed
- Weaviate running (see instructions below)

To start Weaviate:
    cd docker/weaviate
    docker-compose up -d

To stop Weaviate:
    docker-compose down
"""

import numpy as np
from pathlib import Path
from datetime import date

from greenlang.intelligence.rag import (
    RAGConfig,
    WeaviateProvider,
    WeaviateClient,
    Document,
    Chunk,
    DocMeta,
    get_vector_store,
)


def main():
    """Main example function."""

    print("=" * 70)
    print("GreenLang RAG - Weaviate Integration Example")
    print("=" * 70)
    print()

    # Step 1: Check Weaviate health
    print("[1] Checking Weaviate health...")
    try:
        client = WeaviateClient(
            endpoint="http://localhost:8080",
            timeout_config=5000,
            startup_period=10,
        )
        print("   ✓ Weaviate is running and healthy")

        # Get initial stats
        stats = client.get_stats()
        print(f"   - Endpoint: {stats['endpoint']}")
        print(f"   - Total objects: {stats.get('total_objects', 0)}")
        print()
    except Exception as e:
        print(f"   ✗ Weaviate not available: {e}")
        print()
        print("   Please start Weaviate first:")
        print("     cd docker/weaviate")
        print("     docker-compose up -d")
        print()
        return

    # Step 2: Create RAG configuration for Weaviate
    print("[2] Creating RAG configuration with Weaviate...")
    config = RAGConfig(
        mode="live",
        allowlist=["test_collection", "ghg_protocol_corp"],
        embedding_provider="minilm",
        vector_store_provider="weaviate",
        weaviate_endpoint="http://localhost:8080",
        retrieval_method="similarity",
        default_top_k=5,
        chunk_size=512,
    )
    print(f"   - Vector store: {config.vector_store_provider}")
    print(f"   - Endpoint: {config.weaviate_endpoint}")
    print(f"   - Allowlist: {', '.join(config.allowlist)}")
    print()

    # Step 3: Create Weaviate provider
    print("[3] Creating Weaviate provider...")
    provider = get_vector_store(dimension=384, config=config)
    print(f"   - Provider: {type(provider).__name__}")
    print(f"   - Dimension: 384")
    print()

    # Step 4: Create sample documents
    print("[4] Creating sample documents...")

    # Sample document metadata
    doc_meta = DocMeta(
        doc_id="test-doc-001",
        title="Sample Climate Document",
        collection="test_collection",
        publisher="GreenLang",
        publication_date=date(2025, 10, 3),
        version="1.0",
        content_hash="a" * 64,
        doc_hash="b" * 64,
    )

    # Create sample chunks with embeddings
    chunks = []
    for i in range(5):
        chunk = Chunk(
            chunk_id=f"chunk-{i:03d}",
            doc_id=doc_meta.doc_id,
            section_path=f"Section {i+1}",
            section_hash="c" * 64,
            page_start=i + 1,
            page_end=i + 1,
            paragraph=0,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            text=f"This is sample text for chunk {i}. It discusses climate-related topics.",
            token_count=15,
            embedding_model="all-MiniLM-L6-v2",
        )

        # Generate random embedding (in real use, use actual embedder)
        embedding = np.random.randn(384).astype(np.float32)
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        doc = Document(chunk=chunk, embedding=embedding)
        doc.metadata["title"] = doc_meta.title
        doc.metadata["publisher"] = doc_meta.publisher
        doc.metadata["year"] = str(doc_meta.publication_date.year)
        doc.metadata["version"] = doc_meta.version

        chunks.append(doc)

    print(f"   - Created {len(chunks)} sample documents")
    print()

    # Step 5: Add documents to Weaviate
    print("[5] Adding documents to Weaviate...")
    try:
        provider.add_documents(chunks, collection="test_collection")
        print(f"   ✓ Added {len(chunks)} documents to 'test_collection'")
        print()
    except Exception as e:
        print(f"   ✗ Failed to add documents: {e}")
        print()
        return

    # Step 6: Perform similarity search
    print("[6] Performing similarity search...")

    # Create query embedding (random for this example)
    query_embedding = np.random.randn(384).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    try:
        results = provider.similarity_search(
            query_embedding=query_embedding,
            k=3,
            collections=["test_collection"],
        )

        print(f"   ✓ Found {len(results)} results")
        print()

        for i, doc in enumerate(results, 1):
            print(f"   Result {i}:")
            print(f"      - Chunk ID: {doc.chunk.chunk_id}")
            print(f"      - Section: {doc.chunk.section_path}")
            print(f"      - Page: {doc.chunk.page_start}")
            print(f"      - Distance: {doc.metadata.get('distance', 'N/A'):.4f}")
            print(f"      - Similarity: {doc.metadata.get('similarity_score', 'N/A'):.4f}")
            print(f"      - Text: {doc.chunk.text[:60]}...")
            print()
    except Exception as e:
        print(f"   ✗ Search failed: {e}")
        print()

    # Step 7: Get provider stats
    print("[7] Getting provider statistics...")
    stats = provider.get_stats()
    print(f"   - Provider: {stats['provider']}")
    print(f"   - Endpoint: {stats['endpoint']}")
    print(f"   - Dimension: {stats['dimension']}")
    print(f"   - Documents added: {stats['total_documents_added']}")
    print(f"   - Collections: {', '.join(stats['collections_added'])}")
    print()

    # Step 8: Collection filtering test
    print("[8] Testing collection filtering...")

    # Try to query with disallowed collection (should fail)
    try:
        results = provider.similarity_search(
            query_embedding=query_embedding,
            k=3,
            collections=["not_allowed_collection"],
        )
        print("   ✗ SECURITY ISSUE: Disallowed collection was not blocked!")
    except ValueError as e:
        print(f"   ✓ Collection filtering works: {str(e)[:60]}...")
    print()

    # Step 9: Save and load configuration
    print("[9] Testing save/load...")
    save_path = Path("./weaviate_test_config")

    try:
        provider.save(save_path)
        print(f"   ✓ Saved config to {save_path}")

        # Create new provider and load
        new_provider = WeaviateProvider(dimension=384, config=config)
        new_provider.load(save_path)
        print(f"   ✓ Loaded config successfully")
        print(f"   - Documents loaded: {new_provider.total_documents_added}")
        print()

        # Cleanup
        import shutil
        if save_path.exists():
            shutil.rmtree(save_path)
    except Exception as e:
        print(f"   ✗ Save/load failed: {e}")
        print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Integrate with actual embedding provider (MiniLM/OpenAI)")
    print("2. Use with RAGEngine for full pipeline")
    print("3. Test with real climate documents")
    print("4. Monitor performance with Weaviate metrics")
    print()
    print("To clean up:")
    print("  cd docker/weaviate")
    print("  docker-compose down")
    print()


if __name__ == "__main__":
    main()
