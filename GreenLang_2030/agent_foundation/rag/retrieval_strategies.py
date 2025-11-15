"""
Retrieval Strategies for GreenLang RAG System

Implements multiple retrieval strategies:
- Semantic search (vector similarity)
- Keyword search (BM25, TF-IDF)
- Hybrid search (combination)
- MMR (Maximum Marginal Relevance)
- Cross-encoder reranking
- Context assembly with source attribution
"""

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval strategy"""
    documents: List[Any]
    scores: List[float]
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: Optional[str] = None
    confidence: float = 0.0
    retrieval_time_ms: float = 0.0

    def __post_init__(self):
        """Calculate confidence and provenance after initialization"""
        if self.scores:
            self.confidence = float(np.mean(self.scores))

        if not self.provenance_hash:
            self._calculate_provenance()

    def _calculate_provenance(self):
        """Calculate SHA-256 hash for audit trail"""
        doc_hashes = [
            hashlib.md5(str(doc).encode()).hexdigest()[:8]
            for doc in self.documents
        ]
        provenance_str = f"{self.strategy}:{':'.join(doc_hashes)}"
        self.provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

    def filter_by_confidence(self, threshold: float = 0.8) -> 'RetrievalResult':
        """Filter results by confidence threshold"""
        filtered_docs = []
        filtered_scores = []

        for doc, score in zip(self.documents, self.scores):
            if score >= threshold:
                filtered_docs.append(doc)
                filtered_scores.append(score)

        return RetrievalResult(
            documents=filtered_docs,
            scores=filtered_scores,
            strategy=self.strategy,
            metadata=self.metadata,
            confidence=np.mean(filtered_scores) if filtered_scores else 0.0
        )


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies"""

    def __init__(self, name: str = "base"):
        self.name = name
        self.metrics = {
            "total_queries": 0,
            "avg_retrieval_time": 0.0,
            "avg_confidence": 0.0
        }

    @abstractmethod
    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve documents using the strategy"""
        pass

    def update_metrics(self, result: RetrievalResult, elapsed_time: float):
        """Update retrieval metrics"""
        self.metrics["total_queries"] += 1
        self.metrics["avg_retrieval_time"] = (
            (self.metrics["avg_retrieval_time"] * (self.metrics["total_queries"] - 1) + elapsed_time)
            / self.metrics["total_queries"]
        )
        self.metrics["avg_confidence"] = (
            (self.metrics["avg_confidence"] * (self.metrics["total_queries"] - 1) + result.confidence)
            / self.metrics["total_queries"]
        )


class SemanticSearch(RetrievalStrategy):
    """
    Semantic search using vector similarity
    Core strategy for dense retrieval
    """

    def __init__(self, vector_store: Any, embedding_generator: Any):
        super().__init__("semantic")
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve using semantic similarity"""
        import time
        start_time = time.time()

        # Generate embedding if not provided
        if query_embedding is None:
            query_embedding = self.embedding_generator.embed_query(query)

        # Search vector store
        documents, scores = self.vector_store.similarity_search(
            query_embedding,
            top_k,
            filters
        )

        elapsed_time = (time.time() - start_time) * 1000

        result = RetrievalResult(
            documents=documents,
            scores=scores,
            strategy=self.name,
            metadata={
                "query_length": len(query),
                "embedding_dim": query_embedding.shape[0],
                "num_results": len(documents)
            },
            retrieval_time_ms=elapsed_time
        )

        self.update_metrics(result, elapsed_time)
        return result


class KeywordSearch(RetrievalStrategy):
    """
    Keyword search using BM25 or TF-IDF
    Sparse retrieval for exact matches
    """

    def __init__(
        self,
        documents: List[Any],
        algorithm: str = "bm25",
        k1: float = 1.2,
        b: float = 0.75
    ):
        super().__init__(f"keyword_{algorithm}")
        self.documents = documents
        self.algorithm = algorithm
        self.k1 = k1  # BM25 parameter
        self.b = b    # BM25 parameter

        # Build inverted index
        self._build_index()

    def _build_index(self):
        """Build inverted index for keyword search"""
        self.inverted_index = defaultdict(list)
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_count = len(self.documents)

        total_length = 0
        for doc_id, doc in enumerate(self.documents):
            # Extract text content
            if hasattr(doc, 'content'):
                text = doc.content
            elif isinstance(doc, dict):
                text = doc.get('content', doc.get('text', ''))
            else:
                text = str(doc)

            # Tokenize
            tokens = self._tokenize(text.lower())
            doc_length = len(tokens)
            self.doc_lengths.append(doc_length)
            total_length += doc_length

            # Update inverted index
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                self.inverted_index[token].append((doc_id, count))

        self.avg_doc_length = total_length / max(1, self.doc_count)

        # Calculate IDF scores
        self.idf_scores = {}
        for token, posting_list in self.inverted_index.items():
            df = len(posting_list)
            self.idf_scores[token] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _calculate_bm25_score(
        self,
        doc_id: int,
        query_tokens: List[str],
        token_counts: Dict[str, int]
    ) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_length = self.doc_lengths[doc_id]

        for token in query_tokens:
            if token not in token_counts:
                continue

            tf = token_counts[token]
            idf = self.idf_scores.get(token, 0)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * (numerator / denominator)

        return score

    def _calculate_tfidf_score(
        self,
        doc_id: int,
        query_tokens: List[str],
        token_counts: Dict[str, int]
    ) -> float:
        """Calculate TF-IDF score for a document"""
        score = 0.0
        doc_length = self.doc_lengths[doc_id]

        for token in query_tokens:
            if token not in token_counts:
                continue

            tf = token_counts[token] / max(1, doc_length)
            idf = self.idf_scores.get(token, 0)
            score += tf * idf

        return score

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve using keyword matching"""
        import time
        start_time = time.time()

        # Tokenize query
        query_tokens = self._tokenize(query.lower())
        query_token_set = set(query_tokens)

        # Score documents
        doc_scores = []

        for doc_id in range(self.doc_count):
            # Get token counts for this document
            token_counts = {}
            for token in query_token_set:
                if token in self.inverted_index:
                    for posting_doc_id, count in self.inverted_index[token]:
                        if posting_doc_id == doc_id:
                            token_counts[token] = count
                            break

            if not token_counts:
                continue

            # Apply filters if provided
            if filters:
                doc = self.documents[doc_id]
                if hasattr(doc, 'metadata'):
                    metadata = doc.metadata
                elif isinstance(doc, dict):
                    metadata = doc.get('metadata', {})
                else:
                    metadata = {}

                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue

            # Calculate score
            if self.algorithm == "bm25":
                score = self._calculate_bm25_score(doc_id, query_tokens, token_counts)
            else:  # tf-idf
                score = self._calculate_tfidf_score(doc_id, query_tokens, token_counts)

            if score > 0:
                doc_scores.append((doc_id, score))

        # Sort by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-k documents
        documents = []
        scores = []
        for doc_id, score in doc_scores[:top_k]:
            documents.append(self.documents[doc_id])
            # Normalize score to 0-1 range
            scores.append(min(1.0, score / max(1.0, doc_scores[0][1]) if doc_scores else 0))

        elapsed_time = (time.time() - start_time) * 1000

        result = RetrievalResult(
            documents=documents,
            scores=scores,
            strategy=self.name,
            metadata={
                "query_tokens": len(query_tokens),
                "matching_docs": len(doc_scores),
                "algorithm": self.algorithm
            },
            retrieval_time_ms=elapsed_time
        )

        self.update_metrics(result, elapsed_time)
        return result


class HybridSearch(RetrievalStrategy):
    """
    Hybrid search combining semantic and keyword strategies
    Implements Reciprocal Rank Fusion (RRF) for result combination
    """

    def __init__(
        self,
        semantic_strategy: SemanticSearch,
        keyword_strategy: KeywordSearch,
        alpha: float = 0.5,
        fusion_method: str = "rrf"
    ):
        super().__init__("hybrid")
        self.semantic_strategy = semantic_strategy
        self.keyword_strategy = keyword_strategy
        self.alpha = alpha  # Weight for semantic search (0-1)
        self.fusion_method = fusion_method

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve using hybrid approach"""
        import time
        start_time = time.time()

        # Get results from both strategies
        semantic_result = self.semantic_strategy.retrieve(
            query, query_embedding, top_k * 2, filters
        )
        keyword_result = self.keyword_strategy.retrieve(
            query, None, top_k * 2, filters
        )

        # Fuse results
        if self.fusion_method == "rrf":
            documents, scores = self._reciprocal_rank_fusion(
                semantic_result.documents, semantic_result.scores,
                keyword_result.documents, keyword_result.scores,
                top_k
            )
        else:  # weighted
            documents, scores = self._weighted_fusion(
                semantic_result.documents, semantic_result.scores,
                keyword_result.documents, keyword_result.scores,
                top_k
            )

        elapsed_time = (time.time() - start_time) * 1000

        result = RetrievalResult(
            documents=documents,
            scores=scores,
            strategy=self.name,
            metadata={
                "semantic_docs": len(semantic_result.documents),
                "keyword_docs": len(keyword_result.documents),
                "fusion_method": self.fusion_method,
                "alpha": self.alpha
            },
            retrieval_time_ms=elapsed_time
        )

        self.update_metrics(result, elapsed_time)
        return result

    def _reciprocal_rank_fusion(
        self,
        docs1: List[Any], scores1: List[float],
        docs2: List[Any], scores2: List[float],
        top_k: int,
        k: int = 60
    ) -> Tuple[List[Any], List[float]]:
        """
        Reciprocal Rank Fusion (RRF) for combining results
        k: constant for RRF formula (typically 60)
        """
        # Create document scores map
        doc_scores = {}
        doc_map = {}

        # Add semantic search results
        for rank, (doc, score) in enumerate(zip(docs1, scores1)):
            doc_id = self._get_doc_id(doc)
            rrf_score = self.alpha * (1.0 / (k + rank + 1))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = doc

        # Add keyword search results
        for rank, (doc, score) in enumerate(zip(docs2, scores2)):
            doc_id = self._get_doc_id(doc)
            rrf_score = (1 - self.alpha) * (1.0 / (k + rank + 1))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Get top-k documents
        documents = []
        scores = []
        for doc_id, score in sorted_docs[:top_k]:
            documents.append(doc_map[doc_id])
            scores.append(score)

        # Normalize scores
        if scores:
            max_score = max(scores)
            scores = [s / max_score for s in scores]

        return documents, scores

    def _weighted_fusion(
        self,
        docs1: List[Any], scores1: List[float],
        docs2: List[Any], scores2: List[float],
        top_k: int
    ) -> Tuple[List[Any], List[float]]:
        """Weighted fusion of results"""
        # Normalize scores
        if scores1:
            scores1 = np.array(scores1) / max(scores1)
        else:
            scores1 = np.array([])

        if scores2:
            scores2 = np.array(scores2) / max(scores2)
        else:
            scores2 = np.array([])

        # Combine with weights
        doc_scores = {}
        doc_map = {}

        for doc, score in zip(docs1, scores1):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] = self.alpha * score
            doc_map[doc_id] = doc

        for doc, score in zip(docs2, scores2):
            doc_id = self._get_doc_id(doc)
            if doc_id in doc_scores:
                doc_scores[doc_id] += (1 - self.alpha) * score
            else:
                doc_scores[doc_id] = (1 - self.alpha) * score
                doc_map[doc_id] = doc

        # Sort and return top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        documents = [doc_map[doc_id] for doc_id, _ in sorted_docs[:top_k]]
        scores = [score for _, score in sorted_docs[:top_k]]

        return documents, scores

    def _get_doc_id(self, doc: Any) -> str:
        """Get document ID for deduplication"""
        if hasattr(doc, 'doc_id'):
            return doc.doc_id
        elif isinstance(doc, dict) and 'doc_id' in doc:
            return doc['doc_id']
        else:
            # Use content hash as ID
            content = str(doc)
            return hashlib.md5(content.encode()).hexdigest()[:16]


class MMRRetrieval(RetrievalStrategy):
    """
    Maximum Marginal Relevance (MMR) retrieval
    Balances relevance and diversity in results
    """

    def __init__(
        self,
        base_strategy: RetrievalStrategy,
        embedding_generator: Any,
        lambda_param: float = 0.7
    ):
        super().__init__("mmr")
        self.base_strategy = base_strategy
        self.embedding_generator = embedding_generator
        self.lambda_param = lambda_param  # Balance between relevance and diversity

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve using MMR algorithm"""
        import time
        start_time = time.time()

        # Get initial candidates (2x top_k)
        initial_result = self.base_strategy.retrieve(
            query, query_embedding, top_k * 2, filters
        )

        if len(initial_result.documents) <= top_k:
            # Not enough documents for MMR
            return initial_result

        # Generate embeddings for documents if needed
        if query_embedding is None:
            query_embedding = self.embedding_generator.embed_query(query)

        doc_embeddings = []
        for doc in initial_result.documents:
            if hasattr(doc, 'embedding') and doc.embedding is not None:
                doc_embeddings.append(doc.embedding)
            else:
                # Generate embedding for document
                if hasattr(doc, 'content'):
                    content = doc.content
                elif isinstance(doc, dict):
                    content = doc.get('content', doc.get('text', ''))
                else:
                    content = str(doc)

                embedding = self.embedding_generator.embed_texts([content])[0]
                doc_embeddings.append(embedding)

        doc_embeddings = np.array(doc_embeddings)

        # MMR algorithm
        selected_indices = []
        selected_docs = []
        selected_scores = []
        unselected_indices = list(range(len(initial_result.documents)))

        # Select first document (most relevant)
        first_idx = 0
        selected_indices.append(first_idx)
        selected_docs.append(initial_result.documents[first_idx])
        selected_scores.append(initial_result.scores[first_idx])
        unselected_indices.remove(first_idx)

        # Iteratively select documents
        while len(selected_docs) < top_k and unselected_indices:
            mmr_scores = []

            for idx in unselected_indices:
                # Relevance to query
                relevance = self._cosine_similarity(
                    query_embedding,
                    doc_embeddings[idx]
                )

                # Maximum similarity to already selected documents
                max_sim = 0
                for sel_idx in selected_indices:
                    sim = self._cosine_similarity(
                        doc_embeddings[idx],
                        doc_embeddings[sel_idx]
                    )
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx = mmr_scores[0][0]

            selected_indices.append(best_idx)
            selected_docs.append(initial_result.documents[best_idx])
            selected_scores.append(mmr_scores[0][1])
            unselected_indices.remove(best_idx)

        # Normalize scores
        if selected_scores:
            max_score = max(selected_scores)
            selected_scores = [s / max_score for s in selected_scores if max_score > 0]

        elapsed_time = (time.time() - start_time) * 1000

        result = RetrievalResult(
            documents=selected_docs,
            scores=selected_scores,
            strategy=self.name,
            metadata={
                "initial_candidates": len(initial_result.documents),
                "lambda": self.lambda_param,
                "base_strategy": self.base_strategy.name
            },
            retrieval_time_ms=elapsed_time
        )

        self.update_metrics(result, elapsed_time)
        return result

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)


class RerankedRetrieval(RetrievalStrategy):
    """
    Cross-encoder reranking for improved relevance
    Uses a cross-encoder model to rerank initial results
    """

    def __init__(
        self,
        base_strategy: RetrievalStrategy,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        super().__init__("reranked")
        self.base_strategy = base_strategy
        self.reranker = None
        self._load_reranker(reranker_model)

    def _load_reranker(self, model_name: str):
        """Load cross-encoder reranking model"""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(model_name)
            logger.info(f"Loaded reranker model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
        except Exception as e:
            logger.error(f"Error loading reranker: {e}")

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve with reranking"""
        import time
        start_time = time.time()

        # Get initial candidates (2x top_k for reranking)
        initial_result = self.base_strategy.retrieve(
            query, query_embedding, min(top_k * 2, 100), filters
        )

        if not self.reranker or len(initial_result.documents) <= 1:
            # No reranking available or needed
            return initial_result

        # Prepare query-document pairs
        pairs = []
        for doc in initial_result.documents:
            if hasattr(doc, 'content'):
                content = doc.content
            elif isinstance(doc, dict):
                content = doc.get('content', doc.get('text', ''))
            else:
                content = str(doc)
            pairs.append([query, content])

        # Rerank with cross-encoder
        try:
            scores = self.reranker.predict(pairs)

            # Sort by reranked scores
            doc_scores = list(zip(initial_result.documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Get top-k
            documents = [doc for doc, _ in doc_scores[:top_k]]
            scores = [float(score) for _, score in doc_scores[:top_k]]

            # Normalize scores to 0-1 range
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                if max_score > min_score:
                    scores = [(s - min_score) / (max_score - min_score) for s in scores]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            documents = initial_result.documents[:top_k]
            scores = initial_result.scores[:top_k]

        elapsed_time = (time.time() - start_time) * 1000

        result = RetrievalResult(
            documents=documents,
            scores=scores,
            strategy=self.name,
            metadata={
                "initial_candidates": len(initial_result.documents),
                "reranked": True,
                "base_strategy": self.base_strategy.name
            },
            retrieval_time_ms=elapsed_time
        )

        self.update_metrics(result, elapsed_time)
        return result


class ContextAssembler:
    """
    Assemble context from retrieved documents
    Handles source attribution and context limits
    """

    def __init__(
        self,
        max_context_length: int = 8000,
        include_metadata: bool = True,
        include_confidence: bool = True,
        source_format: str = "numbered"  # numbered, inline, footnote
    ):
        self.max_context_length = max_context_length
        self.include_metadata = include_metadata
        self.include_confidence = include_confidence
        self.source_format = source_format

    def assemble(
        self,
        documents: List[Any],
        scores: List[float],
        query: str,
        max_docs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Assemble context from documents

        Returns:
            Dictionary with:
            - context: Assembled context string
            - sources: List of source attributions
            - metadata: Additional metadata
            - provenance_hash: SHA-256 hash for audit
        """
        context_parts = []
        sources = []
        total_length = 0
        docs_used = 0

        if max_docs is None:
            max_docs = len(documents)

        for i, (doc, score) in enumerate(zip(documents[:max_docs], scores[:max_docs])):
            # Extract content
            if hasattr(doc, 'content'):
                content = doc.content
                source = getattr(doc, 'source', f"Document {i+1}")
                metadata = getattr(doc, 'metadata', {})
            elif isinstance(doc, dict):
                content = doc.get('content', doc.get('text', ''))
                source = doc.get('source', f"Document {i+1}")
                metadata = doc.get('metadata', {})
            else:
                content = str(doc)
                source = f"Document {i+1}"
                metadata = {}

            # Calculate available space
            available_space = self.max_context_length - total_length
            if available_space <= 0:
                break

            # Truncate if needed
            if len(content) > available_space:
                content = content[:available_space] + "..."

            # Format context part
            context_part = self._format_context_part(
                content, source, score, i + 1
            )

            context_parts.append(context_part)
            total_length += len(context_part)
            docs_used += 1

            # Add source attribution
            source_info = {
                "index": i + 1,
                "source": source,
                "confidence": float(score),
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }

            if self.include_metadata and metadata:
                source_info["metadata"] = metadata

            sources.append(source_info)

        # Assemble final context
        if self.source_format == "numbered":
            context = "\n\n".join(context_parts)
        elif self.source_format == "inline":
            context = "\n".join(context_parts)
        else:  # footnote
            context = "\n\n".join([part.split('\n')[1] if '\n' in part else part
                                  for part in context_parts])
            context += "\n\n---\nSources:\n"
            context += "\n".join([f"[{i+1}] {src['source']}" for i, src in enumerate(sources)])

        # Calculate provenance hash
        provenance_str = f"{query}:{context}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        return {
            "context": context,
            "sources": sources,
            "metadata": {
                "docs_used": docs_used,
                "total_length": total_length,
                "max_length": self.max_context_length,
                "avg_confidence": float(np.mean(scores[:docs_used])) if docs_used > 0 else 0,
                "query": query
            },
            "provenance_hash": provenance_hash
        }

    def _format_context_part(
        self,
        content: str,
        source: str,
        score: float,
        index: int
    ) -> str:
        """Format a single context part"""
        if self.source_format == "numbered":
            header = f"[{index}]"
            if self.include_confidence:
                header += f" (Confidence: {score:.2%})"
            header += f" Source: {source}"
            return f"{header}\n{content}"

        elif self.source_format == "inline":
            if self.include_confidence:
                return f"{content} [{source}, {score:.2%}]"
            else:
                return f"{content} [{source}]"

        else:  # footnote
            return f"[{index}] {content}"

    def assemble_for_llm(
        self,
        documents: List[Any],
        scores: List[float],
        query: str,
        instruction_template: Optional[str] = None
    ) -> str:
        """
        Assemble context specifically for LLM consumption
        Includes safety instructions per GreenLang requirements
        """
        context_data = self.assemble(documents, scores, query)

        if instruction_template is None:
            instruction_template = """Based on the following context, please answer the question.

IMPORTANT SAFETY RULES (GreenLang Zero-Hallucination Requirements):
1. Use ONLY the provided context for factual information
2. DO NOT generate any numeric calculations - only reference numbers from context
3. Current confidence level: {avg_confidence:.2%}
4. If confidence is below 80%, indicate uncertainty in your response
5. Clearly distinguish between facts from context and any interpretations

CONTEXT:
{context}

QUESTION: {query}

Please provide your answer, ensuring all factual claims are directly supported by the context above."""

        return instruction_template.format(
            context=context_data["context"],
            query=query,
            avg_confidence=context_data["metadata"]["avg_confidence"]
        )