# -*- coding: utf-8 -*-
"""
GreenLang RAG (Retrieval-Augmented Generation) System

Production-ready RAG implementation with zero-hallucination guarantee
for climate intelligence and regulatory compliance.

This package provides:
- Multi-format document processing (PDF, HTML, Office, etc.)
- Multiple embedding models (sentence-transformers, OpenAI, Cohere)
- Multiple vector databases (FAISS, Pinecone, Weaviate, Qdrant, ChromaDB)
- Hybrid search strategies (semantic, keyword, MMR)
- Cross-encoder reranking for improved relevance
- Knowledge graph integration via Neo4j
- Provenance tracking with SHA-256 hashes
- 80%+ confidence threshold enforcement

Example:
    >>> from greenlang_rag import RAGSystem, create_vector_store
    >>>
    >>> # Initialize vector store
    >>> vector_store = create_vector_store("faiss", dimension=768)
    >>>
    >>> # Create RAG system
    >>> rag = RAGSystem(
    ...     vector_store=vector_store,
    ...     confidence_threshold=0.8,
    ...     use_reranker=True
    ... )
    >>>
    >>> # Ingest documents
    >>> rag.ingest_documents(documents)
    >>>
    >>> # Retrieve with confidence scoring
    >>> results = rag.retrieve("carbon emissions calculation", top_k=5)
    >>> assert results.confidence >= 0.8  # GreenLang requirement
"""

from .document_processor import (
    DocumentProcessor,
    DocumentParser,
    ChunkingStrategy,
    Document,
    DocumentMetadata
)

from .embedding_generator import (
    EmbeddingGenerator,
    EmbeddingModel,
    MultiModelEmbedding,
    EmbeddingCache
)

from .vector_store import (
    VectorStore,
    FAISSVectorStore,
    ChromaDBVectorStore,
    PineconeVectorStore,
    WeaviateVectorStore,
    QdrantVectorStore,
    HybridVectorStore,
    create_vector_store
)

from .rag_system import (
    RAGSystem,
    RetrievalResult,
    Reranker,
    ChunkingStrategy as RAGChunkingStrategy,
    EmbeddingModel as RAGEmbeddingModel
)

from .retrieval_strategies import (
    RetrievalStrategy,
    SemanticSearch,
    KeywordSearch,
    HybridSearch,
    MMRRetrieval,
    RerankedRetrieval,
    ContextAssembler
)

from .knowledge_graph import (
    KnowledgeGraphStore,
    Neo4jConnector,
    GraphRetrieval,
    EntityExtractor,
    RelationshipExtractor
)

__version__ = "1.0.0"

__all__ = [
    # Core RAG System
    "RAGSystem",
    "RetrievalResult",
    "Reranker",

    # Document Processing
    "DocumentProcessor",
    "DocumentParser",
    "Document",
    "DocumentMetadata",
    "ChunkingStrategy",

    # Embeddings
    "EmbeddingGenerator",
    "EmbeddingModel",
    "MultiModelEmbedding",
    "EmbeddingCache",

    # Vector Stores
    "VectorStore",
    "FAISSVectorStore",
    "ChromaDBVectorStore",
    "PineconeVectorStore",
    "WeaviateVectorStore",
    "QdrantVectorStore",
    "HybridVectorStore",
    "create_vector_store",

    # Retrieval Strategies
    "RetrievalStrategy",
    "SemanticSearch",
    "KeywordSearch",
    "HybridSearch",
    "MMRRetrieval",
    "RerankedRetrieval",
    "ContextAssembler",

    # Knowledge Graph
    "KnowledgeGraphStore",
    "Neo4jConnector",
    "GraphRetrieval",
    "EntityExtractor",
    "RelationshipExtractor",
]

# Configure logging
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

def configure_logging(level=logging.INFO):
    """Configure RAG system logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger