# -*- coding: utf-8 -*-
"""
Vector Store Implementations for GreenLang RAG System

Production-ready vector database integrations with:
- ChromaDB for development/local deployments with persistent storage
- Pinecone Serverless for production with auto-scaling and multi-tenancy
- Factory pattern for easy backend switching
- Batch operations (1000+ vectors/second)
- Metadata filtering for multi-tenant isolation
- Health monitoring and performance metrics
- SHA-256 provenance tracking for audit compliance

Features:
- Deterministic retrieval (zero hallucination)
- Batch upsert/add operations for high throughput
- Metadata-based filtering and tenant isolation
- Comprehensive health checks
- Performance metrics tracking
- Cost optimization through batching

Example:
    >>> from .vector_stores import create_chroma_store
    >>> store = create_chroma_store(persist_directory="./chroma_db")
    >>> ids = store.add_documents(documents, embeddings)
    >>> results, scores = store.similarity_search(query_emb, top_k=10)

Author: GreenLang Backend Team
"""

from typing import Optional, Dict, Any

# Import vector store implementations
from .chroma_store import ChromaVectorStore, ChromaDocument, ChromaCollectionStats
from .pinecone_store import PineconeVectorStore, PineconeDocument, PineconeIndexStats
from .factory import (
    VectorStoreFactory,
    VectorStoreConfig,
    VectorStoreType,
    create_chroma_store,
    create_pinecone_store,
)

__all__ = [
    # Vector Store Classes
    "ChromaVectorStore",
    "PineconeVectorStore",
    "VectorStoreFactory",
    # Configuration
    "VectorStoreConfig",
    "VectorStoreType",
    # Data Models
    "ChromaDocument",
    "ChromaCollectionStats",
    "PineconeDocument",
    "PineconeIndexStats",
    # Convenience Functions
    "create_vector_store",
    "create_chroma_store",
    "create_pinecone_store",
]


def create_vector_store(
    store_type: str = "chroma",
    config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Convenience function to create vector store instances.

    Provides unified interface for creating vector stores with automatic
    configuration management and validation.

    Args:
        store_type: Type of vector store ('chroma', 'pinecone')
        config: Configuration dictionary (merged with defaults)

    Returns:
        Configured vector store instance

    Raises:
        ValueError: If store_type is unsupported
        ImportError: If required dependencies are missing

    Example:
        >>> # Create ChromaDB store
        >>> store = create_vector_store('chroma', {
        ...     'persist_directory': './data',
        ...     'collection_name': 'my_collection'
        ... })

        >>> # Create Pinecone store
        >>> store = create_vector_store('pinecone', {
        ...     'pinecone_api_key': 'your-key',
        ...     'pinecone_environment': 'us-east-1'
        ... })

        >>> # Add and search documents
        >>> ids = store.add_documents(documents, embeddings)
        >>> results, scores = store.similarity_search(query_emb, top_k=10)
    """
    factory = VectorStoreFactory()
    store_config = VectorStoreConfig(
        store_type=store_type.lower(),
        **(config or {})
    )
    return factory.create(store_config.store_type, store_config)
