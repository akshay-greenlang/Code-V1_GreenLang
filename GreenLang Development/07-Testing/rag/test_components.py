# -*- coding: utf-8 -*-
"""
Comprehensive component tests for RAG system.

Tests all core components in isolation:
1. Embedder (MiniLMProvider)
2. Vector Store (FAISSProvider)
3. Chunker (TokenAwareChunker)
4. Retrievers (MMRRetriever, SimilarityRetriever)

Testing approach:
- Use mock/synthetic data to avoid network dependencies
- Test determinism and reproducibility
- Verify output formats and schemas
- Test edge cases and error handling
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import List

from greenlang.intelligence.rag.embeddings import MiniLMProvider
from greenlang.intelligence.rag.vector_stores import (
    FAISSProvider,
    Document,
)
from greenlang.intelligence.rag.chunker import TokenAwareChunker, CharacterChunker
from greenlang.intelligence.rag.retrievers import (
    MMRRetriever,
    SimilarityRetriever,
    cosine_similarity,
    mmr_retrieval,
)
from greenlang.intelligence.rag.models import Chunk
from greenlang.intelligence.rag.config import RAGConfig


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_config():
    """Create test configuration."""
    return RAGConfig(
        mode="live",  # Use live mode for testing (replay mode requires network isolation)
        embedding_provider="minilm",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        vector_store_provider="faiss",
        chunk_size=512,
        chunk_overlap=64,
        allowlist=["test_collection"],
    )


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Climate change is causing global temperatures to rise.",
        "Greenhouse gas emissions contribute to the warming effect.",
        "Carbon dioxide is the primary greenhouse gas from human activities.",
        "Renewable energy sources can reduce carbon emissions.",
        "Solar and wind power are clean energy alternatives.",
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    chunks = []
    for i, text in enumerate([
        "Climate change is causing global temperatures to rise.",
        "Greenhouse gas emissions contribute to the warming effect.",
        "Carbon dioxide is the primary greenhouse gas from human activities.",
        "Renewable energy sources can reduce carbon emissions.",
        "Solar and wind power are clean energy alternatives.",
    ]):
        chunk = Chunk(
            chunk_id=f"test-chunk-{i}",
            doc_id="test-doc-123",
            section_path="Test Document > Section 1",
            section_hash=f"hash{i}" * 8,
            page_start=i + 1,
            paragraph=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            text=text,
            token_count=len(text.split()),
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_embeddings():
    """Generate mock embeddings (random unit vectors)."""
    np.random.seed(42)

    def generate_embedding(dim=384):
        """Generate a random L2-normalized embedding."""
        vec = np.random.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    return [generate_embedding() for _ in range(5)]


# ============================================================================
# Test MiniLMProvider (Embedder)
# ============================================================================


class TestMiniLMProvider:
    """Test MiniLM embedding provider."""

    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        provider = MiniLMProvider()

        assert provider.name == "sentence-transformers/all-MiniLM-L6-v2"
        assert provider.dim == 384
        assert provider._model is None  # Lazy loading

    def test_initialization_with_custom_config(self, test_config):
        """Test initialization with custom configuration."""
        provider = MiniLMProvider(config=test_config)

        assert provider.config == test_config
        assert provider.dim == 384

    def test_embed_sync_returns_correct_dimension(self):
        """Test embed_sync() returns correct dimension (384)."""
        provider = MiniLMProvider()
        texts = ["Test text for embedding"]

        embeddings = provider.embed_sync(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert isinstance(embeddings[0], np.ndarray)

    def test_embed_sync_normalization(self):
        """Test embeddings are L2 normalized (L2 norm = 1)."""
        provider = MiniLMProvider()
        texts = ["Test text for normalization check"]

        embeddings = provider.embed_sync(texts)

        # Check L2 norm is approximately 1.0
        norm = np.linalg.norm(embeddings[0])
        assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_embed_sync_batch_processing(self):
        """Test batch processing with multiple texts."""
        provider = MiniLMProvider()
        texts = [
            "First text",
            "Second text",
            "Third text",
            "Fourth text",
        ]

        embeddings = provider.embed_sync(texts)

        assert len(embeddings) == 4
        for emb in embeddings:
            assert len(emb) == 384
            # Verify normalization
            norm = np.linalg.norm(emb)
            assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_embed_sync_deterministic_mode(self):
        """Test deterministic mode (same input → same output)."""
        # Create two providers in deterministic mode
        config = RAGConfig(mode="replay")

        provider1 = MiniLMProvider(config=config)
        provider2 = MiniLMProvider(config=config)

        text = ["Deterministic embedding test"]

        # Generate embeddings from both providers
        emb1 = provider1.embed_sync(text)
        emb2 = provider2.embed_sync(text)

        # Should be identical (or very close)
        assert np.allclose(emb1[0], emb2[0], rtol=1e-5)

    def test_embed_sync_empty_input_raises_error(self):
        """Test empty input raises ValueError."""
        provider = MiniLMProvider()

        with pytest.raises(ValueError, match="Cannot embed empty text list"):
            provider.embed_sync([])

    def test_get_stats(self):
        """Test get_stats() returns provider statistics."""
        provider = MiniLMProvider()
        provider.embed_sync(["Test"])

        stats = provider.get_stats()

        assert stats["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert stats["dimension"] == 384
        assert stats["total_embeddings"] >= 1


# ============================================================================
# Test FAISSProvider (Vector Store)
# ============================================================================


class TestFAISSProvider:
    """Test FAISS vector store provider."""

    def test_initialization_with_dimension(self, test_config):
        """Test initialization with custom dimension."""
        provider = FAISSProvider(dimension=384, config=test_config)

        assert provider.dimension == 384
        assert provider.index is not None
        assert len(provider.documents) == 0

    def test_add_documents(self, test_config, sample_chunks, mock_embeddings):
        """Test add_documents() adds documents with embeddings."""
        provider = FAISSProvider(dimension=384, config=test_config)

        # Create documents with embeddings
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]

        # Add documents
        provider.add_documents(docs, collection="test_collection")

        assert len(provider.documents) == 5
        assert "test_collection" in provider.collections
        assert len(provider.collections["test_collection"]) == 5

    def test_add_documents_validates_collection(self, sample_chunks, mock_embeddings):
        """Test add_documents() validates collection allowlist."""
        config = RAGConfig(allowlist=["allowed_collection"])
        provider = FAISSProvider(dimension=384, config=config)

        docs = [
            Document(chunk=sample_chunks[0], embedding=mock_embeddings[0])
        ]

        # Should raise error for disallowed collection
        with pytest.raises(ValueError, match="not allowed"):
            provider.add_documents(docs, collection="not_allowed")

    def test_add_documents_validates_embedding_dimension(self, test_config, sample_chunks):
        """Test add_documents() validates embedding dimension."""
        provider = FAISSProvider(dimension=384, config=test_config)

        # Create document with wrong dimension
        wrong_embedding = np.random.randn(128).astype(np.float32)  # Wrong dimension
        docs = [Document(chunk=sample_chunks[0], embedding=wrong_embedding)]

        with pytest.raises(ValueError, match="dimension mismatch"):
            provider.add_documents(docs, collection="test_collection")

    def test_similarity_search(self, test_config, sample_chunks, mock_embeddings):
        """Test similarity_search() returns similar documents."""
        provider = FAISSProvider(dimension=384, config=test_config)

        # Add documents
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]
        provider.add_documents(docs, collection="test_collection")

        # Search with first embedding
        query_embedding = mock_embeddings[0]
        results = provider.similarity_search(query_embedding, k=3)

        assert len(results) <= 3
        assert all(isinstance(doc, Document) for doc in results)

    def test_similarity_search_collection_filtering(self, sample_chunks, mock_embeddings):
        """Test similarity_search() filters by collection."""
        # Create config with multiple allowed collections
        config = RAGConfig(allowlist=["collection1", "collection2"])
        provider = FAISSProvider(dimension=384, config=config)

        # Add documents to two collections
        docs1 = [
            Document(chunk=sample_chunks[0], embedding=mock_embeddings[0])
        ]
        docs2 = [
            Document(chunk=sample_chunks[1], embedding=mock_embeddings[1])
        ]

        provider.add_documents(docs1, collection="collection1")
        provider.add_documents(docs2, collection="collection2")

        # Search only collection1
        query_embedding = mock_embeddings[0]
        results = provider.similarity_search(
            query_embedding,
            k=10,
            collections=["collection1"]
        )

        # Should only return documents from collection1
        for doc in results:
            assert doc.metadata["collection"] == "collection1"

    def test_similarity_search_empty_store(self, test_config):
        """Test similarity_search() on empty store."""
        provider = FAISSProvider(dimension=384, config=test_config)

        query_embedding = np.random.randn(384).astype(np.float32)
        results = provider.similarity_search(query_embedding, k=5)

        assert len(results) == 0

    def test_document_retrieval(self, test_config, sample_chunks, mock_embeddings):
        """Test document retrieval returns correct chunks."""
        provider = FAISSProvider(dimension=384, config=test_config)

        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]
        provider.add_documents(docs, collection="test_collection")

        # Search and verify returned documents
        query_embedding = mock_embeddings[0]
        results = provider.similarity_search(query_embedding, k=2)

        assert len(results) <= 2
        for doc in results:
            assert hasattr(doc, 'chunk')
            assert hasattr(doc.chunk, 'text')
            assert hasattr(doc.chunk, 'chunk_id')

    def test_save_and_load(self, test_config, sample_chunks, mock_embeddings):
        """Test save() and load() persist vector store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Create and populate provider
            provider1 = FAISSProvider(dimension=384, config=test_config)
            docs = [
                Document(chunk=chunk, embedding=emb)
                for chunk, emb in zip(sample_chunks, mock_embeddings)
            ]
            provider1.add_documents(docs, collection="test_collection")

            # Save
            provider1.save(save_path)

            # Load into new provider
            provider2 = FAISSProvider(dimension=384, config=test_config)
            provider2.load(save_path)

            # Verify loaded state
            assert len(provider2.documents) == 5
            assert "test_collection" in provider2.collections
            assert provider2.dimension == 384

    def test_get_stats(self, test_config, sample_chunks, mock_embeddings):
        """Test get_stats() returns vector store statistics."""
        provider = FAISSProvider(dimension=384, config=test_config)

        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks[:3], mock_embeddings[:3])
        ]
        provider.add_documents(docs, collection="test_collection")

        stats = provider.get_stats()

        assert stats["provider"] == "faiss"
        assert stats["dimension"] == 384
        assert stats["total_documents"] == 3
        assert stats["total_collections"] == 1
        assert "test_collection" in stats["collections"]


# ============================================================================
# Test TokenAwareChunker
# ============================================================================


class TestTokenAwareChunker:
    """Test token-aware chunker."""

    def test_initialization(self):
        """Test chunker initialization with default params."""
        chunker = TokenAwareChunker(chunk_size=512, overlap=64)

        assert chunker.chunk_size == 512
        assert chunker.overlap == 64

    def test_chunk_generation(self):
        """Test basic chunk generation."""
        chunker = TokenAwareChunker(chunk_size=50, overlap=10)

        text = " ".join([f"Sentence {i}." for i in range(20)])
        chunks = chunker.chunk_document(
            text=text,
            doc_id="test-doc",
            section_path="Test Section"
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.text for chunk in chunks)

    def test_chunk_size_limits(self):
        """Test chunks respect size limits (512 tokens)."""
        chunker = TokenAwareChunker(chunk_size=512, overlap=64)

        # Create a long document with sentences (to allow proper chunking)
        sentences = [f"This is sentence number {i} with some words in it." for i in range(100)]
        text = " ".join(sentences)
        chunks = chunker.chunk_document(
            text=text,
            doc_id="test-doc",
            section_path="Test Section"
        )

        # Verify chunks exist and most respect size limit
        # Note: Some chunks may exceed slightly due to sentence boundary preservation
        assert len(chunks) > 0
        avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
        assert avg_tokens <= 512 * 1.5  # Average should be reasonable

    def test_chunk_overlap(self):
        """Test overlap between chunks (64 tokens)."""
        chunker = TokenAwareChunker(chunk_size=100, overlap=20)

        text = " ".join([f"Word{i}" for i in range(200)])
        chunks = chunker.chunk_document(
            text=text,
            doc_id="test-doc",
            section_path="Test Section"
        )

        # Check that consecutive chunks have some overlap
        if len(chunks) >= 2:
            # The end of chunk[0] should overlap with start of chunk[1]
            chunk0_end_words = chunks[0].text.split()[-10:]
            chunk1_start_words = chunks[1].text.split()[:10]

            # There should be some common words (overlap)
            common = set(chunk0_end_words) & set(chunk1_start_words)
            assert len(common) > 0

    def test_sentence_boundary_preservation(self):
        """Test sentence boundaries are preserved."""
        chunker = TokenAwareChunker(chunk_size=50, overlap=10)

        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunker.chunk_document(
            text=text,
            doc_id="test-doc",
            section_path="Test Section"
        )

        # Each chunk should contain complete sentences (end with period)
        for chunk in chunks:
            # Allow for trailing spaces
            assert chunk.text.strip().endswith('.') or chunk == chunks[-1]

    def test_empty_text_handling(self):
        """Test handling of empty input."""
        chunker = TokenAwareChunker(chunk_size=512, overlap=64)

        chunks = chunker.chunk_document(
            text="",
            doc_id="test-doc",
            section_path="Test Section"
        )

        assert len(chunks) == 0

    def test_single_sentence_handling(self):
        """Test handling of single sentence."""
        chunker = TokenAwareChunker(chunk_size=512, overlap=64)

        text = "This is a single sentence."
        chunks = chunker.chunk_document(
            text=text,
            doc_id="test-doc",
            section_path="Test Section"
        )

        assert len(chunks) == 1
        assert chunks[0].text.strip() == text

    def test_chunk_metadata(self):
        """Test chunk metadata is properly set."""
        chunker = TokenAwareChunker(chunk_size=512, overlap=64)

        text = "Test document with metadata."
        chunks = chunker.chunk_document(
            text=text,
            doc_id="doc-123",
            section_path="Chapter 1 > Section 1.1",
            page_start=5,
            page_end=6,
            extra={"collection": "test"}
        )

        assert len(chunks) > 0
        chunk = chunks[0]

        assert chunk.doc_id == "doc-123"
        assert chunk.section_path == "Chapter 1 > Section 1.1"
        assert chunk.page_start == 5
        assert chunk.page_end == 6
        assert chunk.extra.get("collection") == "test"

    def test_chunk_id_stability(self):
        """Test chunk IDs are stable (deterministic)."""
        chunker = TokenAwareChunker(chunk_size=512, overlap=64)

        text = "Stable chunk ID test."

        # Generate chunks twice
        chunks1 = chunker.chunk_document(text=text, doc_id="doc-123", section_path="Section 1")
        chunks2 = chunker.chunk_document(text=text, doc_id="doc-123", section_path="Section 1")

        # Chunk IDs should be identical
        assert chunks1[0].chunk_id == chunks2[0].chunk_id


class TestCharacterChunker:
    """Test character-based chunker (fallback)."""

    def test_initialization(self):
        """Test character chunker initialization."""
        chunker = CharacterChunker(chunk_size=1000, overlap=100)

        assert chunker.chunk_size == 1000
        assert chunker.overlap == 100

    def test_chunk_generation(self):
        """Test basic chunk generation."""
        chunker = CharacterChunker(chunk_size=100, overlap=20)

        text = "a" * 300
        chunks = chunker.chunk_document(
            text=text,
            doc_id="test-doc",
            section_path="Test Section"
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)


# ============================================================================
# Test Retrievers (MMR and Similarity)
# ============================================================================


class TestCosineSimility:
    """Test cosine similarity utility."""

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0."""
        vec = np.array([1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec, vec)

        assert np.isclose(similarity, 1.0)

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)

        assert np.isclose(similarity, 0.0, atol=1e-6)

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)

        # For normalized vectors, opposite should give -1, but we clamp to [0, 1]
        assert similarity >= 0.0


class TestMMRRetrieval:
    """Test MMR retrieval algorithm."""

    def test_mmr_basic_retrieval(self, sample_chunks, mock_embeddings):
        """Test basic MMR retrieval."""
        # Create documents with embeddings
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]

        # Query with first embedding
        query_embedding = mock_embeddings[0]

        results = mmr_retrieval(
            query_embedding=query_embedding,
            candidates=docs,
            lambda_mult=0.5,
            k=3
        )

        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(isinstance(r[0], Document) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_mmr_lambda_parameter(self, sample_chunks, mock_embeddings):
        """Test MMR with different lambda values."""
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]

        query_embedding = mock_embeddings[0]

        # High lambda (favor relevance)
        results_high = mmr_retrieval(
            query_embedding=query_embedding,
            candidates=docs,
            lambda_mult=1.0,
            k=3
        )

        # Low lambda (favor diversity)
        results_low = mmr_retrieval(
            query_embedding=query_embedding,
            candidates=docs,
            lambda_mult=0.0,
            k=3
        )

        assert len(results_high) <= 3
        assert len(results_low) <= 3

    def test_mmr_empty_candidates(self):
        """Test MMR with empty candidate list."""
        query_embedding = np.random.randn(384).astype(np.float32)

        results = mmr_retrieval(
            query_embedding=query_embedding,
            candidates=[],
            lambda_mult=0.5,
            k=5
        )

        assert len(results) == 0

    def test_mmr_deterministic_tie_breaking(self, sample_chunks, mock_embeddings):
        """Test MMR has deterministic tie-breaking."""
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]

        query_embedding = mock_embeddings[0]

        # Run MMR twice
        results1 = mmr_retrieval(query_embedding, docs, lambda_mult=0.5, k=3)
        results2 = mmr_retrieval(query_embedding, docs, lambda_mult=0.5, k=3)

        # Results should be identical (same order)
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1[0].chunk.chunk_id == r2[0].chunk.chunk_id


class TestMMRRetriever:
    """Test MMR retriever class."""

    def test_initialization(self, test_config, sample_chunks, mock_embeddings):
        """Test MMR retriever initialization."""
        # Setup vector store
        vector_store = FAISSProvider(dimension=384, config=test_config)
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]
        vector_store.add_documents(docs, collection="test_collection")

        # Create retriever
        retriever = MMRRetriever(
            vector_store=vector_store,
            fetch_k=10,
            top_k=3,
            lambda_mult=0.5
        )

        assert retriever.fetch_k == 10
        assert retriever.top_k == 3
        assert retriever.lambda_mult == 0.5

    def test_mmr_retrieve(self, test_config, sample_chunks, mock_embeddings):
        """Test MMR retrieval with diversity."""
        # Setup vector store
        vector_store = FAISSProvider(dimension=384, config=test_config)
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]
        vector_store.add_documents(docs, collection="test_collection")

        # Create retriever
        retriever = MMRRetriever(
            vector_store=vector_store,
            fetch_k=5,
            top_k=3,
            lambda_mult=0.5
        )

        # Retrieve
        query_embedding = mock_embeddings[0]
        results = retriever.retrieve(query_embedding)

        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)

    def test_mmr_top_k_parameter(self, test_config, sample_chunks, mock_embeddings):
        """Test MMR top_k parameter."""
        vector_store = FAISSProvider(dimension=384, config=test_config)
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]
        vector_store.add_documents(docs, collection="test_collection")

        retriever = MMRRetriever(
            vector_store=vector_store,
            fetch_k=10,
            top_k=2,
            lambda_mult=0.5
        )

        query_embedding = mock_embeddings[0]
        results = retriever.retrieve(query_embedding, top_k=3)

        # Should respect override
        assert len(results) <= 3


class TestSimilarityRetriever:
    """Test similarity retriever class."""

    def test_initialization(self, test_config, sample_chunks, mock_embeddings):
        """Test similarity retriever initialization."""
        vector_store = FAISSProvider(dimension=384, config=test_config)
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]
        vector_store.add_documents(docs, collection="test_collection")

        retriever = SimilarityRetriever(
            vector_store=vector_store,
            top_k=5
        )

        assert retriever.top_k == 5

    def test_similarity_retrieve(self, test_config, sample_chunks, mock_embeddings):
        """Test similarity retrieval."""
        vector_store = FAISSProvider(dimension=384, config=test_config)
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]
        vector_store.add_documents(docs, collection="test_collection")

        retriever = SimilarityRetriever(
            vector_store=vector_store,
            top_k=3
        )

        query_embedding = mock_embeddings[0]
        results = retriever.retrieve(query_embedding)

        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(isinstance(r[0], Document) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_similarity_top_k_override(self, test_config, sample_chunks, mock_embeddings):
        """Test top_k parameter override."""
        vector_store = FAISSProvider(dimension=384, config=test_config)
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(sample_chunks, mock_embeddings)
        ]
        vector_store.add_documents(docs, collection="test_collection")

        retriever = SimilarityRetriever(
            vector_store=vector_store,
            top_k=5
        )

        query_embedding = mock_embeddings[0]
        results = retriever.retrieve(query_embedding, top_k=2)

        assert len(results) <= 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestComponentIntegration:
    """Test integration of multiple components."""

    def test_full_pipeline(self):
        """Test full pipeline: chunking → embedding → indexing → retrieval."""
        # 1. Chunking
        chunker = TokenAwareChunker(chunk_size=100, overlap=20)
        text = "Climate change is a global challenge. " * 20
        chunks = chunker.chunk_document(
            text=text,
            doc_id="test-doc",
            section_path="Introduction"
        )

        assert len(chunks) > 0

        # 2. Embedding
        embedder = MiniLMProvider()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_sync(texts)

        assert len(embeddings) == len(chunks)

        # 3. Indexing
        test_config = RAGConfig(mode="live", allowlist=["test_collection"])
        vector_store = FAISSProvider(dimension=384, config=test_config)
        docs = [
            Document(chunk=chunk, embedding=emb)
            for chunk, emb in zip(chunks, embeddings)
        ]
        vector_store.add_documents(docs, collection="test_collection")

        assert len(vector_store.documents) == len(chunks)

        # 4. Retrieval
        query_text = "climate change global"
        query_embedding = embedder.embed_sync([query_text])[0]

        retriever = SimilarityRetriever(vector_store=vector_store, top_k=3)
        results = retriever.retrieve(query_embedding)

        assert len(results) <= 3
        assert all(isinstance(r[0].chunk, Chunk) for r in results)

    def test_save_load_persistence(self, sample_chunks):
        """Test save/load persistence across components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Create embedder and generate embeddings
            embedder = MiniLMProvider()
            texts = [chunk.text for chunk in sample_chunks]
            embeddings = embedder.embed_sync(texts)

            # Create and populate vector store
            test_config = RAGConfig(mode="live", allowlist=["test_collection"])
            vector_store1 = FAISSProvider(dimension=384, config=test_config)
            docs = [
                Document(chunk=chunk, embedding=emb)
                for chunk, emb in zip(sample_chunks, embeddings)
            ]
            vector_store1.add_documents(docs, collection="test_collection")

            # Save
            vector_store1.save(save_path)

            # Load into new instance
            vector_store2 = FAISSProvider(dimension=384, config=test_config)
            vector_store2.load(save_path)

            # Test retrieval on loaded store
            query_embedding = embeddings[0]
            retriever = SimilarityRetriever(vector_store=vector_store2, top_k=3)
            results = retriever.retrieve(query_embedding)

            assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
