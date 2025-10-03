"""
GreenLang RAG (Retrieval-Augmented Generation) System

INTL-104: RAG v1 - Deterministic, policy-safe retrieval layer for climate standards
and internal documentation with MMR-diversified results and audit-ready citations.

Key Components:
- Vector stores (FAISS, ChromaDB, Weaviate)
- Embedding providers (MiniLM, OpenAI)
- MMR retrieval for diversity
- Collection allowlisting for security
- Deterministic hashing for replay mode
- Audit-ready citations for regulatory compliance
- Document version management for historical compliance
- CSRB governance and approval workflow
- Table extraction from climate PDFs
- Section hierarchy extraction for proper citations
"""

from greenlang.intelligence.rag.models import (
    DocMeta,
    Chunk,
    RAGCitation,
    QueryResult,
    IngestionManifest,
)
from greenlang.intelligence.rag.config import RAGConfig, get_rag_config, get_config
from greenlang.intelligence.rag.hashing import (
    sha256_str,
    canonicalize_text,
    chunk_uuid5,
    section_hash,
)
from greenlang.intelligence.rag.sanitize import (
    sanitize_rag_input,
    sanitize_citation_uri,
)
from greenlang.intelligence.rag.embeddings import (
    EmbeddingProvider,
    MiniLMProvider,
    OpenAIProvider,
    get_embedding_provider,
)
from greenlang.intelligence.rag.vector_stores import (
    VectorStoreProvider,
    FAISSProvider,
    WeaviateProvider,
    Document,
    get_vector_store,
)
from greenlang.intelligence.rag.weaviate_client import WeaviateClient
from greenlang.intelligence.rag.retrievers import (
    MMRRetriever,
    SimilarityRetriever,
    get_retriever,
)
from greenlang.intelligence.rag.chunker import (
    TokenAwareChunker,
    CharacterChunker,
    get_chunker,
)
from greenlang.intelligence.rag.version_manager import (
    DocumentVersionManager,
    VersionConflict,
)
from greenlang.intelligence.rag.governance import (
    RAGGovernance,
    ApprovalRequest,
)
from greenlang.intelligence.rag.table_extractor import (
    ClimateTableExtractor,
)
from greenlang.intelligence.rag.section_extractor import (
    SectionPathExtractor,
    SectionMatch,
)
from greenlang.intelligence.rag.engine import RAGEngine
from greenlang.intelligence.rag.determinism import DeterministicRAG

# Import standalone modules
from greenlang.intelligence.rag import ingest
from greenlang.intelligence.rag import query

__all__ = [
    # Schemas
    "DocMeta",
    "Chunk",
    "RAGCitation",
    "QueryResult",
    "IngestionManifest",
    # Config
    "RAGConfig",
    "get_rag_config",
    "get_config",
    # Hashing
    "sha256_str",
    "canonicalize_text",
    "chunk_uuid5",
    "section_hash",
    # Sanitization
    "sanitize_rag_input",
    "sanitize_citation_uri",
    # Embeddings
    "EmbeddingProvider",
    "MiniLMProvider",
    "OpenAIProvider",
    "get_embedding_provider",
    # Vector Stores
    "VectorStoreProvider",
    "FAISSProvider",
    "WeaviateProvider",
    "WeaviateClient",
    "Document",
    "get_vector_store",
    # Retrievers
    "MMRRetriever",
    "SimilarityRetriever",
    "get_retriever",
    # Chunkers
    "TokenAwareChunker",
    "CharacterChunker",
    "get_chunker",
    # Regulatory Compliance (INTL-104)
    "DocumentVersionManager",
    "VersionConflict",
    "RAGGovernance",
    "ApprovalRequest",
    "ClimateTableExtractor",
    "SectionPathExtractor",
    "SectionMatch",
    # Engine and Determinism
    "RAGEngine",
    "DeterministicRAG",
    # Standalone modules
    "ingest",
    "query",
]

__version__ = "1.0.0"
