# -*- coding: utf-8 -*-
"""
RAG system configuration with security controls and allowlisting.

CRITICAL SECURITY: This module enforces collection allowlisting, network isolation
in replay mode, and other security policies.
"""

import os
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, validator


class RAGConfig(BaseModel):
    """
    RAG system configuration.

    Controls:
    - Collection allowlisting (security)
    - Replay vs Live mode (determinism)
    - Embedding provider selection
    - Vector store selection
    - Performance parameters
    """

    # Mode control (replay/live)
    mode: Literal["replay", "live"] = Field(
        default="replay",
        description="Execution mode: replay=deterministic/offline, live=network-enabled",
    )

    # Collection allowlist (CRITICAL SECURITY)
    allowlist: List[str] = Field(
        default_factory=lambda: [
            "ghg_protocol_corp",
            "ghg_protocol_scope3",
            "ipcc_ar6_wg1",
            "ipcc_ar6_wg2",
            "ipcc_ar6_wg3",
            "gl_docs",
            "test_collection",  # For testing only
        ],
        description="Allowed collections (enforce at ingest and query time)",
    )

    # Embedding configuration
    embedding_provider: Literal["minilm", "openai", "anthropic"] = Field(
        default="minilm", description="Embedding provider to use"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    embedding_dimension: int = Field(
        default=384, description="Embedding vector dimension"
    )
    embedding_batch_size: int = Field(
        default=64, description="Batch size for embedding generation"
    )

    # Vector store configuration
    vector_store_provider: Literal["faiss", "chromadb", "weaviate"] = Field(
        default="faiss", description="Vector store provider"
    )
    vector_store_path: Optional[str] = Field(
        default=None, description="Path to vector store (FAISS/ChromaDB)"
    )
    weaviate_endpoint: str = Field(
        default="http://localhost:8080", description="Weaviate endpoint (if used)"
    )

    # Retrieval configuration
    retrieval_method: Literal["similarity", "mmr", "hybrid"] = Field(
        default="mmr", description="Retrieval method"
    )
    default_top_k: int = Field(
        default=6, description="Default number of results to return"
    )
    default_fetch_k: int = Field(
        default=30, description="Number of candidates for MMR (fetch before re-rank)"
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR lambda (0=diversity, 1=relevance)",
    )

    # Chunking configuration
    chunk_size: int = Field(
        default=512, description="Chunk size in tokens (not characters)"
    )
    chunk_overlap: int = Field(default=64, description="Chunk overlap in tokens")
    chunking_strategy: Literal["token_aware", "character", "sentence"] = Field(
        default="token_aware", description="Chunking strategy"
    )

    # Security configuration
    enable_sanitization: bool = Field(
        default=True, description="Enable input sanitization (RECOMMENDED)"
    )
    strict_sanitization: bool = Field(
        default=True, description="Use strict sanitization (block http/https URLs)"
    )
    enable_network_isolation: bool = Field(
        default=True,
        description="Block network access in replay mode (CRITICAL for determinism)",
    )

    # Performance configuration
    max_context_tokens: int = Field(
        default=4096, description="Maximum tokens to include in LLM context"
    )
    query_timeout_seconds: int = Field(
        default=30, description="Query timeout in seconds"
    )

    # Governance configuration
    require_approval: bool = Field(
        default=True, description="Require CSRB approval for new collections"
    )
    verify_checksums: bool = Field(
        default=True, description="Verify document checksums during ingestion"
    )

    @validator("allowlist")
    def validate_allowlist(cls, v):
        """Validate collection names in allowlist."""
        import re

        for collection in v:
            if not re.match(r"^[a-zA-Z0-9_-]+$", collection):
                raise ValueError(
                    f"Invalid collection name: {collection} (must be alphanumeric + underscore/hyphen)"
                )
            if len(collection) > 64:
                raise ValueError(f"Collection name too long: {collection} (max 64 chars)")
        return v

    @validator("mode")
    def validate_mode(cls, v):
        """Validate mode."""
        if v not in ("replay", "live"):
            raise ValueError(f"Invalid mode: {v} (must be 'replay' or 'live')")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "mode": "replay",
                "allowlist": [
                    "ghg_protocol_corp",
                    "ipcc_ar6_wg3",
                    "gl_docs",
                ],
                "embedding_provider": "minilm",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "vector_store_provider": "faiss",
                "retrieval_method": "mmr",
                "default_top_k": 6,
                "default_fetch_k": 30,
                "mmr_lambda": 0.5,
                "chunk_size": 512,
                "chunk_overlap": 64,
                "enable_sanitization": True,
                "strict_sanitization": True,
            }
        }


def get_rag_config() -> RAGConfig:
    """
    Get RAG configuration from environment variables.

    Environment Variables:
        GL_MODE: Execution mode (replay/live)
        GL_RAG_ALLOWLIST: Comma-separated collection names
        GL_EMBED_PROVIDER: Embedding provider (minilm/openai/anthropic)
        GL_EMBED_MODEL: Embedding model name
        GL_EMBED_DIM: Embedding dimension
        GL_EMBED_BATCH: Embedding batch size
        GL_VECTOR_STORE: Vector store provider (faiss/chromadb/weaviate)
        GL_VECTOR_STORE_PATH: Path to vector store
        WEAVIATE_ENDPOINT: Weaviate endpoint
        GL_RAG_TOP_K: Default top_k
        GL_RAG_FETCH_K: Default fetch_k
        GL_RAG_MMR_LAMBDA: MMR lambda
        GL_RAG_CHUNK_SIZE: Chunk size in tokens
        GL_RAG_CHUNK_OVERLAP: Chunk overlap in tokens

    Returns:
        RAGConfig instance
    """
    # Parse allowlist from env
    allowlist_str = os.getenv(
        "GL_RAG_ALLOWLIST",
        "ghg_protocol_corp,ghg_protocol_scope3,ipcc_ar6_wg1,ipcc_ar6_wg2,ipcc_ar6_wg3,gl_docs,test_collection",
    )
    allowlist = [c.strip() for c in allowlist_str.split(",") if c.strip()]

    config = RAGConfig(
        # Mode
        mode=os.getenv("GL_MODE", "replay"),
        # Allowlist
        allowlist=allowlist,
        # Embedding
        embedding_provider=os.getenv("GL_EMBED_PROVIDER", "minilm"),
        embedding_model=os.getenv(
            "GL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_dimension=int(os.getenv("GL_EMBED_DIM", "384")),
        embedding_batch_size=int(os.getenv("GL_EMBED_BATCH", "64")),
        # Vector store
        vector_store_provider=os.getenv("GL_VECTOR_STORE", "faiss"),
        vector_store_path=os.getenv("GL_VECTOR_STORE_PATH"),
        weaviate_endpoint=os.getenv("WEAVIATE_ENDPOINT", "http://localhost:8080"),
        # Retrieval
        retrieval_method=os.getenv("GL_RAG_METHOD", "mmr"),
        default_top_k=int(os.getenv("GL_RAG_TOP_K", "6")),
        default_fetch_k=int(os.getenv("GL_RAG_FETCH_K", "30")),
        mmr_lambda=float(os.getenv("GL_RAG_MMR_LAMBDA", "0.5")),
        # Chunking
        chunk_size=int(os.getenv("GL_RAG_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("GL_RAG_CHUNK_OVERLAP", "64")),
        chunking_strategy=os.getenv("GL_RAG_CHUNKING", "token_aware"),
        # Security
        enable_sanitization=os.getenv("GL_RAG_SANITIZE", "true").lower() == "true",
        strict_sanitization=os.getenv("GL_RAG_SANITIZE_STRICT", "true").lower()
        == "true",
        enable_network_isolation=os.getenv("GL_RAG_NETWORK_ISOLATION", "true").lower()
        == "true",
    )

    # If in replay mode and network isolation enabled, set environment variables
    if config.mode == "replay" and config.enable_network_isolation:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["PYTHONHASHSEED"] = "42"

    return config


# Singleton instance (can be overridden for testing)
_config: Optional[RAGConfig] = None


def set_rag_config(config: RAGConfig) -> None:
    """
    Set global RAG configuration (for testing).

    Args:
        config: RAGConfig instance
    """
    global _config
    _config = config


def get_config() -> RAGConfig:
    """
    Get global RAG configuration (cached).

    Returns:
        RAGConfig instance
    """
    global _config
    if _config is None:
        _config = get_rag_config()
    return _config


# Collection allowlist validation
def is_collection_allowed(collection: str, config: Optional[RAGConfig] = None) -> bool:
    """
    Check if a collection is in the allowlist.

    Args:
        collection: Collection name
        config: RAGConfig instance (defaults to global config)

    Returns:
        True if allowed, False otherwise

    Raises:
        ValueError: If collection name is invalid
    """
    if config is None:
        config = get_config()

    # Validate collection name format
    import re

    if not re.match(r"^[a-zA-Z0-9_-]+$", collection):
        raise ValueError(
            f"Invalid collection name: {collection} (must be alphanumeric + underscore/hyphen)"
        )

    return collection in config.allowlist


def enforce_allowlist(collections: List[str], config: Optional[RAGConfig] = None) -> None:
    """
    Enforce allowlist for a list of collections.

    Args:
        collections: List of collection names
        config: RAGConfig instance (defaults to global config)

    Raises:
        ValueError: If any collection is not allowed
    """
    if config is None:
        config = get_config()

    for collection in collections:
        if not is_collection_allowed(collection, config):
            raise ValueError(
                f"Collection '{collection}' is not in allowlist. "
                f"Allowed collections: {', '.join(config.allowlist)}"
            )
