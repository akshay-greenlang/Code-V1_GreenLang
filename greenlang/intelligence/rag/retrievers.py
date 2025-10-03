"""
MMR (Maximal Marginal Relevance) retrieval for RAG system.

Implements two-stage retrieval:
1. Similarity search: Fetch top-K candidates
2. MMR re-ranking: Diversify results while maintaining relevance

Key features:
- Deterministic tie-breaking (by chunk_id ascending)
- Cosine similarity on L2-normalized vectors
- Lambda parameter for relevance vs diversity trade-off
"""

import numpy as np
from typing import List, Optional, Tuple
import logging

from greenlang.intelligence.rag.vector_stores import Document, VectorStoreProvider

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Assumes vectors are L2-normalized (unit length).

    Args:
        a: First vector (L2-normalized)
        b: Second vector (L2-normalized)

    Returns:
        Cosine similarity in [0, 1] range

    Example:
        >>> a = np.array([1.0, 0.0, 0.0])
        >>> b = np.array([0.0, 1.0, 0.0])
        >>> cosine_similarity(a, b)
        0.0
        >>> cosine_similarity(a, a)
        1.0
    """
    # For L2-normalized vectors, cosine similarity = dot product
    similarity = np.dot(a, b)

    # Clamp to [0, 1] to handle numerical errors
    return max(0.0, min(1.0, float(similarity)))


def mmr_retrieval(
    query_embedding: np.ndarray,
    candidates: List[Document],
    lambda_mult: float = 0.5,
    k: int = 6,
) -> List[Tuple[Document, float]]:
    """
    Maximal Marginal Relevance (MMR) retrieval algorithm.

    Greedy algorithm that selects documents balancing:
    - Relevance: Similarity to query
    - Diversity: Dissimilarity to already selected documents

    Algorithm:
        For each iteration i:
            score[d] = λ·sim(query, d) - (1-λ)·max_{s in selected} sim(d, s)
            selected[i] = argmax_{d in candidates} score[d]

    Deterministic tie-breaking:
        When multiple documents have the same MMR score, ties are broken
        by chunk_id in ascending order.

    Args:
        query_embedding: Query embedding vector (L2-normalized)
        candidates: List of candidate documents (with embeddings)
        lambda_mult: Trade-off parameter (0=diversity, 1=relevance)
        k: Number of documents to select

    Returns:
        List of (document, mmr_score) tuples, sorted by selection order

    Example:
        >>> query = np.array([1.0, 0.0, 0.0])
        >>> docs = [...]  # List of Document objects
        >>> results = mmr_retrieval(query, docs, lambda_mult=0.5, k=5)
        >>> for doc, score in results:
        ...     print(f"{doc.chunk.chunk_id}: {score:.3f}")
    """
    if not candidates:
        return []

    # Limit k to number of candidates
    k = min(k, len(candidates))

    # Extract embeddings from candidates
    candidate_embeddings = np.array(
        [doc.embedding for doc in candidates], dtype=np.float32
    )

    # Compute query similarities for all candidates (relevance scores)
    query_sims = np.array(
        [cosine_similarity(query_embedding, emb) for emb in candidate_embeddings]
    )

    # Track selected documents
    selected_docs: List[Tuple[Document, float]] = []
    selected_indices: List[int] = []
    unselected_indices = list(range(len(candidates)))

    # Greedy MMR selection
    for iteration in range(k):
        if not unselected_indices:
            break

        # Compute MMR scores for unselected candidates
        mmr_scores = []

        for idx in unselected_indices:
            # Relevance component: similarity to query
            relevance = query_sims[idx]

            # Diversity component: max similarity to already selected documents
            if selected_indices:
                # Compute similarity to each selected document
                similarities = [
                    cosine_similarity(
                        candidate_embeddings[idx], candidate_embeddings[sel_idx]
                    )
                    for sel_idx in selected_indices
                ]
                max_sim_to_selected = max(similarities)
            else:
                # No documents selected yet, no diversity penalty
                max_sim_to_selected = 0.0

            # MMR score: λ·relevance - (1-λ)·max_similarity
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected

            mmr_scores.append((idx, mmr_score))

        # Sort by MMR score (descending), then by chunk_id (ascending) for determinism
        mmr_scores.sort(
            key=lambda x: (-x[1], candidates[x[0]].chunk.chunk_id)
        )

        # Select top document
        best_idx, best_score = mmr_scores[0]

        selected_docs.append((candidates[best_idx], best_score))
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

        logger.debug(
            f"MMR iteration {iteration + 1}: selected chunk_id={candidates[best_idx].chunk.chunk_id}, "
            f"score={best_score:.4f}"
        )

    return selected_docs


class MMRRetriever:
    """
    MMR-based retriever for RAG system.

    Two-stage retrieval:
    1. Fetch top-K candidates from vector store (similarity search)
    2. Re-rank using MMR for diversity
    """

    def __init__(
        self,
        vector_store: VectorStoreProvider,
        fetch_k: int = 30,
        top_k: int = 6,
        lambda_mult: float = 0.5,
    ):
        """
        Initialize MMR retriever.

        Args:
            vector_store: Vector store provider
            fetch_k: Number of candidates to fetch (stage 1)
            top_k: Number of results to return (stage 2)
            lambda_mult: MMR lambda parameter (0=diversity, 1=relevance)
        """
        self.vector_store = vector_store
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.lambda_mult = lambda_mult

    def retrieve(
        self,
        query_embedding: np.ndarray,
        collections: Optional[List[str]] = None,
        fetch_k: Optional[int] = None,
        top_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using MMR.

        Args:
            query_embedding: Query embedding vector
            collections: Filter by collections (None = all)
            fetch_k: Override default fetch_k
            top_k: Override default top_k
            lambda_mult: Override default lambda_mult

        Returns:
            List of (document, mmr_score) tuples
        """
        # Use provided or default parameters
        fetch_k = fetch_k or self.fetch_k
        top_k = top_k or self.top_k
        lambda_mult = lambda_mult or self.lambda_mult

        logger.info(
            f"MMR retrieval: fetch_k={fetch_k}, top_k={top_k}, lambda={lambda_mult:.2f}, "
            f"collections={collections or 'all'}"
        )

        # Stage 1: Fetch candidates from vector store
        candidates = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=fetch_k,
            collections=collections,
        )

        if not candidates:
            logger.warning("No candidates found in vector store")
            return []

        logger.debug(f"Fetched {len(candidates)} candidates")

        # Stage 2: MMR re-ranking
        results = mmr_retrieval(
            query_embedding=query_embedding,
            candidates=candidates,
            lambda_mult=lambda_mult,
            k=top_k,
        )

        logger.info(f"MMR retrieval returned {len(results)} documents")

        return results


class SimilarityRetriever:
    """
    Simple similarity-based retriever (no MMR).

    Directly returns top-K most similar documents from vector store.
    """

    def __init__(
        self,
        vector_store: VectorStoreProvider,
        top_k: int = 6,
    ):
        """
        Initialize similarity retriever.

        Args:
            vector_store: Vector store provider
            top_k: Number of results to return
        """
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query_embedding: np.ndarray,
        collections: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents by similarity.

        Args:
            query_embedding: Query embedding vector
            collections: Filter by collections (None = all)
            top_k: Override default top_k

        Returns:
            List of (document, similarity_score) tuples
        """
        top_k = top_k or self.top_k

        logger.info(
            f"Similarity retrieval: top_k={top_k}, collections={collections or 'all'}"
        )

        # Search vector store
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k,
            collections=collections,
        )

        # Compute similarity scores (cosine similarity)
        results_with_scores = []
        for doc in results:
            similarity = cosine_similarity(query_embedding, doc.embedding)
            results_with_scores.append((doc, similarity))

        logger.info(f"Similarity retrieval returned {len(results_with_scores)} documents")

        return results_with_scores


def get_retriever(
    vector_store: VectorStoreProvider,
    retrieval_method: str = "mmr",
    fetch_k: int = 30,
    top_k: int = 6,
    lambda_mult: float = 0.5,
):
    """
    Get retriever based on method.

    Args:
        vector_store: Vector store provider
        retrieval_method: Retrieval method ("mmr" or "similarity")
        fetch_k: Number of candidates to fetch (for MMR)
        top_k: Number of results to return
        lambda_mult: MMR lambda parameter

    Returns:
        Retriever instance (MMRRetriever or SimilarityRetriever)

    Raises:
        ValueError: If retrieval method is not supported
    """
    if retrieval_method.lower() == "mmr":
        return MMRRetriever(
            vector_store=vector_store,
            fetch_k=fetch_k,
            top_k=top_k,
            lambda_mult=lambda_mult,
        )
    elif retrieval_method.lower() == "similarity":
        return SimilarityRetriever(
            vector_store=vector_store,
            top_k=top_k,
        )
    else:
        raise ValueError(
            f"Unknown retrieval method: {retrieval_method}. "
            f"Supported: mmr, similarity"
        )
