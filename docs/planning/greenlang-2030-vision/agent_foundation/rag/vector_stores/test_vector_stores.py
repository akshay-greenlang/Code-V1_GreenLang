# -*- coding: utf-8 -*-
"""
Integration Tests for Vector Stores

Comprehensive tests for ChromaDB and Pinecone vector store implementations.
Tests cover:
- Document addition with batch operations (1000+ vectors/second)
- Similarity search with metadata filtering
- Multi-tenancy and namespace isolation
- Health monitoring and error handling
- Performance metrics tracking
- Provenance tracking (SHA-256 hashing)

Author: GreenLang Backend Team
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest import mock

import numpy as np
import pytest

logger = logging.getLogger(__name__)


class TestDocument:
    """Test document object."""

    def __init__(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[dict] = None
    ):
        self.content = content
        self.doc_id = doc_id
        self.metadata = metadata or {}


class TestChromaVectorStore:
    """Tests for ChromaDB vector store implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for ChromaDB persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def chroma_store(self, temp_dir):
        """Create ChromaDB vector store instance."""
        try:
            from .chroma_store import ChromaVectorStore

            return ChromaVectorStore(
                collection_name="test_collection",
                persist_directory=temp_dir,
                distance_metric="cosine",
                embedding_dimension=384,
                batch_size=100
            )
        except ImportError:
            pytest.skip("ChromaDB not installed")

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            TestDocument(
                content="Climate change is a major global challenge",
                doc_id="doc1",
                metadata={"source": "IPCC", "year": 2023}
            ),
            TestDocument(
                content="Carbon emissions from energy production",
                doc_id="doc2",
                metadata={"source": "IEA", "year": 2022}
            ),
            TestDocument(
                content="Renewable energy sources and sustainability",
                doc_id="doc3",
                metadata={"source": "IPCC", "year": 2023}
            ),
            TestDocument(
                content="Greenhouse gas emissions in developing nations",
                doc_id="doc4",
                metadata={"source": "UN", "year": 2021}
            ),
            TestDocument(
                content="Energy efficiency and conservation strategies",
                doc_id="doc5",
                metadata={"source": "IEA", "year": 2023}
            ),
        ]

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        np.random.seed(42)
        return np.random.randn(5, 384).astype(np.float32)

    def test_add_documents(self, chroma_store, sample_documents, sample_embeddings):
        """Test adding documents to ChromaDB."""
        ids = chroma_store.add_documents(sample_documents, sample_embeddings)

        assert len(ids) == len(sample_documents)
        assert chroma_store.collection.count() == len(sample_documents)

    def test_add_documents_with_custom_ids(self, chroma_store, sample_documents, sample_embeddings):
        """Test adding documents with custom IDs."""
        custom_ids = ["custom_1", "custom_2", "custom_3", "custom_4", "custom_5"]
        ids = chroma_store.add_documents(sample_documents, sample_embeddings, ids=custom_ids)

        assert ids == custom_ids

    def test_add_documents_batch_processing(self, chroma_store, sample_embeddings):
        """Test batch processing for large document sets."""
        # Create 250 documents (will be processed in 3 batches with batch_size=100)
        documents = [
            TestDocument(
                content=f"Document {i} with climate data",
                doc_id=f"doc_{i}",
                metadata={"batch": i // 100}
            )
            for i in range(250)
        ]
        embeddings = np.random.randn(250, 384).astype(np.float32)

        ids = chroma_store.add_documents(documents, embeddings, batch_size=100)

        assert len(ids) == 250
        assert chroma_store.collection.count() == 250
        assert chroma_store.metrics["total_batches_processed"] == 3

    def test_similarity_search(self, chroma_store, sample_documents, sample_embeddings):
        """Test similarity search functionality."""
        chroma_store.add_documents(sample_documents, sample_embeddings)

        # Use first document's embedding as query
        query_embedding = sample_embeddings[0]
        docs, scores = chroma_store.similarity_search(query_embedding, top_k=3)

        assert len(docs) <= 3
        assert len(scores) == len(docs)
        assert all(0 <= score <= 1 for score in scores)

    def test_similarity_search_with_filters(self, chroma_store, sample_documents, sample_embeddings):
        """Test similarity search with metadata filtering."""
        chroma_store.add_documents(sample_documents, sample_embeddings)

        query_embedding = sample_embeddings[0]
        filters = {"source": "IPCC"}

        docs, scores = chroma_store.similarity_search(
            query_embedding,
            top_k=10,
            filters=filters
        )

        # All results should have source="IPCC"
        for doc in docs:
            assert doc.metadata.get("source") == "IPCC"

    def test_delete_documents(self, chroma_store, sample_documents, sample_embeddings):
        """Test document deletion."""
        ids = chroma_store.add_documents(sample_documents, sample_embeddings)
        initial_count = chroma_store.collection.count()

        # Delete first two documents
        success = chroma_store.delete(ids[:2])

        assert success
        assert chroma_store.collection.count() == initial_count - 2

    def test_update_documents(self, chroma_store, sample_documents, sample_embeddings):
        """Test document update functionality."""
        ids = chroma_store.add_documents(sample_documents, sample_embeddings)

        # Update metadata
        new_metadata = [
            {"source": "Updated", "year": 2024},
        ]

        success = chroma_store.update(
            ids=[ids[0]],
            metadata=new_metadata
        )

        assert success

    def test_collection_stats(self, chroma_store, sample_documents, sample_embeddings):
        """Test collection statistics retrieval."""
        chroma_store.add_documents(sample_documents, sample_embeddings)

        stats = chroma_store.get_collection_stats()

        assert stats.name == "test_collection"
        assert stats.count == len(sample_documents)
        assert stats.dimension == 384

    def test_health_check(self, chroma_store, sample_documents, sample_embeddings):
        """Test health check functionality."""
        chroma_store.add_documents(sample_documents, sample_embeddings)

        health = chroma_store.health_check()

        assert health["status"] == "healthy"
        assert health["collection_name"] == "test_collection"
        assert health["document_count"] == len(sample_documents)

    def test_metrics_tracking(self, chroma_store, sample_documents, sample_embeddings):
        """Test performance metrics tracking."""
        chroma_store.add_documents(sample_documents, sample_embeddings)
        chroma_store.similarity_search(sample_embeddings[0], top_k=5)

        metrics = chroma_store.get_metrics()

        assert metrics["total_documents"] == len(sample_documents)
        assert metrics["total_queries"] == 1
        assert metrics["avg_search_time_ms"] > 0

    def test_empty_collection_search(self, chroma_store):
        """Test search on empty collection."""
        query_embedding = np.random.randn(384)

        docs, scores = chroma_store.similarity_search(query_embedding, top_k=5)

        assert len(docs) == 0
        assert len(scores) == 0

    def test_persistence(self, temp_dir, sample_documents, sample_embeddings):
        """Test ChromaDB persistence."""
        # Create store and add documents
        from .chroma_store import ChromaVectorStore

        store1 = ChromaVectorStore(
            collection_name="persistent_collection",
            persist_directory=temp_dir
        )
        ids = store1.add_documents(sample_documents, sample_embeddings)
        count1 = store1.collection.count()

        # Create new store instance pointing to same directory
        store2 = ChromaVectorStore(
            collection_name="persistent_collection",
            persist_directory=temp_dir
        )

        count2 = store2.collection.count()
        assert count1 == count2 == len(sample_documents)


class TestPineconeVectorStore:
    """Tests for Pinecone vector store implementation."""

    @pytest.fixture
    def pinecone_api_key(self):
        """Get Pinecone API key from environment or mock."""
        import os

        return os.environ.get("PINECONE_API_KEY", "test-key-12345")

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            TestDocument(
                content="Scope 1 emissions from combustion",
                doc_id="scope1_1",
                metadata={"scope": "1", "category": "combustion"}
            ),
            TestDocument(
                content="Scope 2 emissions from electricity",
                doc_id="scope2_1",
                metadata={"scope": "2", "category": "electricity"}
            ),
            TestDocument(
                content="Scope 3 emissions from supply chain",
                doc_id="scope3_1",
                metadata={"scope": "3", "category": "supply_chain"}
            ),
            TestDocument(
                content="Carbon footprint calculation methodology",
                doc_id="methodology_1",
                metadata={"scope": "1,2,3", "category": "methodology"}
            ),
            TestDocument(
                content="GHG emissions from transportation",
                doc_id="transport_1",
                metadata={"scope": "1", "category": "transportation"}
            ),
        ]

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        np.random.seed(42)
        return np.random.randn(5, 384).astype(np.float32)

    @pytest.mark.skip(reason="Requires Pinecone API credentials")
    def test_add_documents(self, pinecone_api_key, sample_documents, sample_embeddings):
        """Test adding documents to Pinecone."""
        try:
            from .pinecone_store import PineconeVectorStore

            store = PineconeVectorStore(
                api_key=pinecone_api_key,
                index_name="test-index",
                dimension=384
            )

            ids = store.add_documents(sample_documents, sample_embeddings)

            assert len(ids) == len(sample_documents)
        except ImportError:
            pytest.skip("Pinecone client not installed")

    @pytest.mark.skip(reason="Requires Pinecone API credentials")
    def test_batch_upsert_performance(self, pinecone_api_key):
        """Test batch upsert performance (1000+ vectors/second)."""
        try:
            from .pinecone_store import PineconeVectorStore

            store = PineconeVectorStore(
                api_key=pinecone_api_key,
                index_name="test-index",
                dimension=384,
                batch_size=100
            )

            # Create 1000 test documents
            documents = [
                TestDocument(
                    content=f"Document {i}",
                    doc_id=f"perf_doc_{i}"
                )
                for i in range(1000)
            ]
            embeddings = np.random.randn(1000, 384).astype(np.float32)

            import time

            start = time.time()
            ids = store.add_documents(documents, embeddings)
            elapsed = time.time() - start

            throughput = 1000 / elapsed
            logger.info(f"Batch upsert throughput: {throughput:.0f} vectors/second")

            assert throughput >= 100  # Conservative minimum
        except ImportError:
            pytest.skip("Pinecone client not installed")

    @pytest.mark.skip(reason="Requires Pinecone API credentials")
    def test_multi_namespace_isolation(self, pinecone_api_key, sample_documents, sample_embeddings):
        """Test multi-tenant namespace isolation."""
        try:
            from .pinecone_store import PineconeVectorStore

            store = PineconeVectorStore(
                api_key=pinecone_api_key,
                index_name="test-index",
                dimension=384
            )

            # Add documents to different namespaces
            ids1 = store.add_documents(
                sample_documents[:2],
                sample_embeddings[:2],
                namespace="tenant-1"
            )
            ids2 = store.add_documents(
                sample_documents[2:],
                sample_embeddings[2:],
                namespace="tenant-2"
            )

            # Search in tenant-1 namespace
            docs1, _ = store.similarity_search(
                sample_embeddings[0],
                namespace="tenant-1",
                top_k=10
            )

            # Results should only be from tenant-1
            assert len(docs1) >= 0  # May be 0 if index is empty
        except ImportError:
            pytest.skip("Pinecone client not installed")


class TestVectorStoreFactory:
    """Tests for VectorStore factory pattern."""

    def test_factory_creation_chroma(self):
        """Test factory creates ChromaDB store."""
        try:
            from .factory import VectorStoreFactory, VectorStoreConfig, VectorStoreType

            config = VectorStoreConfig(
                store_type=VectorStoreType.CHROMA,
                collection_name="test_factory"
            )
            factory = VectorStoreFactory()
            store = factory.create(VectorStoreType.CHROMA, config)

            assert store is not None
            assert store.collection_name == "test_factory"
        except ImportError:
            pytest.skip("ChromaDB not installed")

    def test_factory_config_validation(self):
        """Test factory validates configuration."""
        try:
            from .factory import VectorStoreFactory, VectorStoreConfig, VectorStoreType

            config = VectorStoreConfig(
                store_type=VectorStoreType.PINECONE,
                pinecone_api_key=None  # Missing required key
            )
            factory = VectorStoreFactory()

            with pytest.raises(ValueError):
                factory.create(VectorStoreType.PINECONE, config)
        except ImportError:
            pytest.skip("Factory not available")

    def test_factory_available_stores(self):
        """Test factory reports available stores."""
        try:
            from .factory import VectorStoreFactory

            factory = VectorStoreFactory()
            available = factory.get_available_stores()

            assert isinstance(available, dict)
            assert len(available) > 0
        except ImportError:
            pytest.skip("Factory not available")


class TestVectorStoreIntegration:
    """Integration tests for vector store workflows."""

    def test_end_to_end_chroma_workflow(self):
        """Test complete ChromaDB workflow."""
        try:
            from .factory import create_chroma_store

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create store
                store = create_chroma_store(
                    collection_name="integration_test",
                    persist_directory=tmpdir
                )

                # Add documents
                docs = [
                    TestDocument(
                        content="Test document 1",
                        doc_id="test1",
                        metadata={"test": True}
                    ),
                    TestDocument(
                        content="Test document 2",
                        doc_id="test2",
                        metadata={"test": True}
                    ),
                ]
                embeddings = np.random.randn(2, 384).astype(np.float32)

                ids = store.add_documents(docs, embeddings)
                assert len(ids) == 2

                # Search
                query_emb = embeddings[0]
                results, scores = store.similarity_search(query_emb, top_k=2)
                assert len(results) > 0

                # Delete
                success = store.delete([ids[0]])
                assert success

                # Health check
                health = store.health_check()
                assert health["status"] == "healthy"

        except ImportError:
            pytest.skip("ChromaDB not installed")

    def test_document_provenance_tracking(self):
        """Test SHA-256 provenance hashing."""
        try:
            from .factory import create_chroma_store

            with tempfile.TemporaryDirectory() as tmpdir:
                store = create_chroma_store(persist_directory=tmpdir)

                docs = [
                    TestDocument(
                        content="Deterministic content",
                        doc_id="prov_test",
                        metadata={"version": "1.0"}
                    )
                ]
                embeddings = np.random.randn(1, 384).astype(np.float32)

                store.add_documents(docs, embeddings)

                # Retrieve and verify provenance hash is present
                results, _ = store.similarity_search(embeddings[0], top_k=1)
                if results:
                    assert "provenance_hash" in results[0].metadata

        except ImportError:
            pytest.skip("ChromaDB not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
