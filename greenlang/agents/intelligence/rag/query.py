# -*- coding: utf-8 -*-
"""
RAG query with MMR retrieval and citation formatting.

Implements the query flow from CTO spec Section 8.

Query Pipeline:
1. Enforce allowlist
2. Embed query
3. Vector search in Weaviate/FAISS (fetch_k)
4. MMR re-rank
5. Sanitize text (strip URLs/code; no tool triggers)
6. Format citations
7. Return QueryResult[]
"""

import time
from typing import List, Optional, Dict
import numpy as np

from greenlang.agents.intelligence.rag.models import (
    QueryResult,
    Chunk,
    RAGCitation,
    DocMeta,
)
from greenlang.agents.intelligence.rag.config import (
    RAGConfig,
    get_config,
    enforce_allowlist,
)
from greenlang.agents.intelligence.rag.embeddings import get_embedding_provider
from greenlang.agents.intelligence.rag.vector_stores import get_vector_store, Document
from greenlang.agents.intelligence.rag.retrievers import get_retriever, mmr_retrieval
from greenlang.agents.intelligence.rag.sanitize import (
    sanitize_rag_input,
    sanitize_for_prompt,
    detect_suspicious_content,
)
from greenlang.agents.intelligence.rag.hashing import query_hash


async def query(
    q: str,
    top_k: int = 6,
    collections: Optional[List[str]] = None,
    fetch_k: int = 30,
    mmr_lambda: float = 0.5,
    filters: Optional[dict] = None,
    config: Optional[RAGConfig] = None,
    doc_metadata: Optional[Dict[str, DocMeta]] = None,
) -> QueryResult:
    """
    Query RAG system with MMR retrieval.

    Per CTO spec Section 8:
    1. Enforce allowlist
    2. Embed query
    3. Vector search in Weaviate/FAISS (fetch_k)
    4. MMR re-rank
    5. Sanitize text (strip URLs/code; no tool triggers)
    6. Format citations
    7. Return QueryResult[]

    Args:
        q: Query string
        top_k: Number of results to return (default: 6)
        collections: Collections to search (default: all allowed)
        fetch_k: Number of candidates for MMR (default: 30)
        mmr_lambda: MMR lambda parameter (0=diversity, 1=relevance, default: 0.5)
        filters: Additional filters for search
        config: RAG configuration (defaults to global config)
        doc_metadata: Document metadata cache for citation generation

    Returns:
        QueryResult with chunks, citations, and metadata

    Raises:
        ValueError: If collection not in allowlist
    """
    start_time = time.time()
    config = config or get_config()
    doc_metadata = doc_metadata or {}

    # Set defaults
    collections = collections or config.allowlist

    # Step 1: Sanitize input query
    original_query = q
    if config.enable_sanitization:
        q = sanitize_for_prompt(q, max_length=512)

        # Check for suspicious content
        warning = detect_suspicious_content(q)
        if warning:
            print(f"[SECURITY WARNING] Query sanitization: {warning}")

    # Step 2: Enforce collection allowlist
    enforce_allowlist(collections, config)

    # Step 3: Initialize components
    embedder = get_embedding_provider(config=config)

    vector_store = get_vector_store(
        dimension=config.embedding_dimension,
        config=config,
    )

    retriever = get_retriever(
        vector_store=vector_store,
        retrieval_method=config.retrieval_method,
        fetch_k=fetch_k,
        top_k=top_k,
        lambda_mult=mmr_lambda,
    )

    # Step 4: Embed query
    query_embedding = await _embed_query(q, embedder, config)

    # Step 5: Fetch candidates from vector store (fetch_k results)
    candidates = await _fetch_candidates(
        query_embedding=query_embedding,
        collections=collections,
        k=fetch_k,
        vector_store=vector_store,
    )

    # Step 6: Apply MMR for diversity
    if config.retrieval_method == "mmr" and len(candidates) > top_k:
        selected_chunks, scores = await _apply_mmr(
            query_embedding=query_embedding,
            candidates=candidates,
            k=top_k,
            lambda_mult=mmr_lambda,
            retriever=retriever,
        )
    else:
        # Use top candidates without MMR
        top_candidates = candidates[:top_k]
        selected_chunks = [doc.chunk for doc in top_candidates]
        scores = [1.0] * len(selected_chunks)  # Placeholder scores

    # Step 7: Generate citations
    citations = _generate_citations(
        selected_chunks,
        scores,
        doc_metadata,
    )

    # Step 8: Sanitize retrieved text
    if config.enable_sanitization:
        for chunk in selected_chunks:
            chunk.text = sanitize_rag_input(
                chunk.text,
                strict=config.strict_sanitization,
            )

    # Step 9: Compute metadata
    total_tokens = sum(chunk.token_count for chunk in selected_chunks)
    search_time_ms = int((time.time() - start_time) * 1000)

    # Compute query hash
    qhash = query_hash(
        q,
        {
            "k": top_k,
            "collections": sorted(collections),
            "fetch_k": fetch_k,
            "mmr_lambda": mmr_lambda,
        },
    )

    # Create QueryResult
    result = QueryResult(
        query=q,
        query_hash=qhash,
        chunks=selected_chunks,
        citations=citations,
        relevance_scores=scores,
        retrieval_method=config.retrieval_method,
        search_time_ms=search_time_ms,
        total_tokens=total_tokens,
        total_chunks=len(selected_chunks),
        collections_searched=collections,
        extra={
            "fetch_k": fetch_k,
            "mmr_lambda": mmr_lambda,
            "original_query": original_query,
        },
    )

    return result


def format_citation(chunk: Chunk, doc_meta: DocMeta, score: float) -> str:
    """
    Format citation per CTO spec Section 9.

    Format: "<Title> (<Year>), <Section>, p.<Page>, <URI>#<Hash>"

    Args:
        chunk: Document chunk
        doc_meta: Document metadata
        score: Relevance score

    Returns:
        Formatted citation string
    """
    # Extract year from publication date
    year = ""
    if doc_meta.publication_date:
        year = f" ({doc_meta.publication_date.year})"
    elif doc_meta.version:
        year = f" v{doc_meta.version}"

    # Format page number
    page_str = ""
    if chunk.page_start:
        if chunk.page_end and chunk.page_end != chunk.page_start:
            page_str = f", pp.{chunk.page_start}-{chunk.page_end}"
        else:
            page_str = f", p.{chunk.page_start}"

    # Format citation
    citation = (
        f"{doc_meta.title}{year}, "
        f"{chunk.section_path}{page_str}, "
        f"{doc_meta.source_uri}#{chunk.section_hash[:8]}"
    )

    # Add checksum for audit trail
    if doc_meta.content_hash:
        citation += f" [SHA256:{doc_meta.content_hash[:8]}]"

    return citation


async def _embed_query(query: str, embedder, config: RAGConfig) -> np.ndarray:
    """
    Embed query string.

    Args:
        query: Query string
        embedder: Embedder instance
        config: RAG configuration

    Returns:
        Embedding vector as numpy array
    """
    # Call embedder.embed() to get actual embedding
    embeddings = await embedder.embed([query])
    return embeddings[0]


async def _fetch_candidates(
    query_embedding: np.ndarray,
    collections: List[str],
    k: int,
    vector_store,
) -> List[Document]:
    """
    Fetch candidate documents from vector store.

    Args:
        query_embedding: Query embedding vector
        collections: Collections to search
        k: Number of candidates to fetch
        vector_store: Vector store instance

    Returns:
        List of candidate documents (with embeddings)
    """
    # Call vector_store.similarity_search(query_embedding, collections, k)
    documents = vector_store.similarity_search(
        query_embedding=query_embedding,
        k=k,
        collections=collections,
    )

    return documents


async def _apply_mmr(
    query_embedding: np.ndarray,
    candidates: List[Document],
    k: int,
    lambda_mult: float,
    retriever,
) -> tuple[List[Chunk], List[float]]:
    """
    Apply MMR for diversity.

    Args:
        query_embedding: Query embedding
        candidates: Candidate documents (with embeddings)
        k: Number of results to select
        lambda_mult: MMR lambda (0=diversity, 1=relevance)
        retriever: Retriever instance

    Returns:
        Tuple of (selected_chunks, scores)
    """
    # Apply MMR retrieval algorithm
    results = mmr_retrieval(
        query_embedding=query_embedding,
        candidates=candidates,
        lambda_mult=lambda_mult,
        k=k,
    )

    # Extract chunks and scores from (Document, score) tuples
    selected_chunks = [doc.chunk for doc, score in results]
    scores = [score for doc, score in results]

    return selected_chunks, scores


def _generate_citations(
    chunks: List[Chunk],
    scores: List[float],
    doc_metadata: Dict[str, DocMeta],
) -> List[RAGCitation]:
    """
    Generate citations for chunks.

    Args:
        chunks: Retrieved chunks
        scores: Relevance scores
        doc_metadata: Document metadata cache

    Returns:
        List of RAGCitation objects
    """
    citations = []

    for chunk, score in zip(chunks, scores):
        # Get document metadata
        doc_meta = doc_metadata.get(chunk.doc_id)

        if doc_meta:
            citation = RAGCitation.from_chunk(
                chunk=chunk,
                doc_meta=doc_meta,
                relevance_score=score,
            )
        else:
            # Fallback: create minimal citation
            citation = RAGCitation(
                doc_title=f"Document {chunk.doc_id}",
                section_path=chunk.section_path,
                section_hash=chunk.section_hash,
                checksum="unknown",
                formatted=f"{chunk.section_path} (Document {chunk.doc_id})",
                relevance_score=score,
            )

        citations.append(citation)

    return citations
