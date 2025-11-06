"""
RAG Determinism and Reproducibility Tests
GL Intelligence Infrastructure

Tests for deterministic behavior in RAG pipeline.
Critical for audit compliance and regulatory requirements.

Version: 1.0.0
Date: 2025-11-06
"""

import pytest
import asyncio
import hashlib
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import json


class TestRAGEmbeddingDeterminism:
    """Test embedding determinism in replay mode."""

    @pytest.mark.asyncio
    async def test_embedding_determinism(self):
        """Test that same text produces identical embeddings."""
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.embedders import MiniLMProvider

        # Use replay mode for determinism
        config = RAGConfig(
            mode="replay",
            embedding_provider="minilm",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=384,
        )

        provider = MiniLMProvider(config=config)

        test_text = "Calculate emissions for natural gas combustion"

        # Generate embeddings twice
        embeddings1 = await provider.embed([test_text])
        embeddings2 = await provider.embed([test_text])

        # Should be byte-for-byte identical
        assert np.array_equal(embeddings1[0], embeddings2[0]), \
            "Embeddings should be deterministic in replay mode"

        # Verify hashes match
        hash1 = hashlib.sha256(embeddings1[0].tobytes()).hexdigest()
        hash2 = hashlib.sha256(embeddings2[0].tobytes()).hexdigest()

        assert hash1 == hash2, f"Hashes differ: {hash1} != {hash2}"

    @pytest.mark.asyncio
    async def test_batch_embedding_determinism(self):
        """Test batch embeddings produce identical results."""
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.embedders import MiniLMProvider

        config = RAGConfig(
            mode="replay",
            embedding_provider="minilm",
            embedding_dimension=384,
        )

        provider = MiniLMProvider(config=config)

        test_texts = [
            "What are Scope 1 emissions?",
            "How do heat pumps reduce carbon?",
            "Calculate natural gas emissions"
        ]

        # Generate batch embeddings twice
        batch1 = await provider.embed(test_texts)
        batch2 = await provider.embed(test_texts)

        # All embeddings should match
        for i, (emb1, emb2) in enumerate(zip(batch1, batch2)):
            assert np.array_equal(emb1, emb2), \
                f"Embedding {i} differs between runs"

    @pytest.mark.asyncio
    async def test_embedding_normalization_stable(self):
        """Test that L2 normalization is stable."""
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.embedders import MiniLMProvider

        config = RAGConfig(
            mode="replay",
            embedding_provider="minilm",
        )

        provider = MiniLMProvider(config=config)
        text = "Test normalization stability"

        # Embed multiple times
        embeddings = [await provider.embed([text]) for _ in range(3)]

        # All should have L2 norm = 1.0
        for emb in embeddings:
            norm = np.linalg.norm(emb[0])
            assert abs(norm - 1.0) < 1e-6, f"L2 norm should be 1.0, got {norm}"

        # All should be identical
        for i in range(1, len(embeddings)):
            assert np.allclose(embeddings[0][0], embeddings[i][0], rtol=1e-9), \
                f"Embedding {i} differs from first"


class TestRAGRetrievalDeterminism:
    """Test retrieval determinism."""

    @pytest.mark.asyncio
    async def test_retrieval_determinism(self, temp_dir):
        """Test that retrieval produces identical results."""
        from greenlang.intelligence.rag.engine import RAGEngine
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.models import DocMeta

        config = RAGConfig(
            mode="replay",
            embedding_provider="minilm",
            vector_store_provider="faiss",
            vector_store_path=str(temp_dir / "vector_store"),
            chunk_size=256,
            chunk_overlap=32,
            retrieval_method="similarity",
            default_top_k=5,
        )

        engine = RAGEngine(config=config)

        # Ingest test document
        test_doc = temp_dir / "test.txt"
        test_doc.write_text(
            "Natural gas combustion produces CO2 emissions. "
            "The emission factor is 0.0531 kg CO2e/kWh. "
            "This is based on the GHG Protocol methodology. "
            "Coal has a higher emission factor. "
            "Solar thermal systems have zero emissions."
        )

        doc_meta = DocMeta(
            title="Test Doc",
            source="test",
            version="1.0",
            collection="test_collection"
        )

        await engine.ingest_document(
            file_path=test_doc,
            collection="test_collection",
            doc_meta=doc_meta
        )

        query = "What is the emission factor for natural gas?"

        # Query multiple times
        results = []
        for _ in range(3):
            result = await engine.query(query, top_k=3, collections=["test_collection"])
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert len(results[0].chunks) == len(results[i].chunks), \
                f"Result {i} has different chunk count"

            for j, (chunk0, chunki) in enumerate(zip(results[0].chunks, results[i].chunks)):
                assert chunk0.chunk_id == chunki.chunk_id, \
                    f"Result {i} chunk {j} ID differs: {chunk0.chunk_id} != {chunki.chunk_id}"
                assert chunk0.text == chunki.text, \
                    f"Result {i} chunk {j} text differs"

            # Scores should be identical
            assert np.allclose(results[0].relevance_scores, results[i].relevance_scores, rtol=1e-9), \
                f"Result {i} scores differ"

    @pytest.mark.asyncio
    async def test_similarity_search_determinism(self, temp_dir):
        """Test similarity search is deterministic."""
        from greenlang.intelligence.rag.engine import RAGEngine
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.models import DocMeta

        config = RAGConfig(
            mode="replay",
            embedding_provider="minilm",
            vector_store_provider="faiss",
            vector_store_path=str(temp_dir / "vs"),
            retrieval_method="similarity",  # Pure similarity, no MMR
            default_top_k=5,
        )

        engine = RAGEngine(config=config)

        # Create test document
        test_doc = temp_dir / "sim_test.txt"
        test_doc.write_text("Test content " * 100)  # Create multiple chunks

        doc_meta = DocMeta(
            title="Similarity Test",
            source="test",
            version="1.0",
            collection="test"
        )

        await engine.ingest_document(test_doc, "test", doc_meta)

        # Query 5 times
        query = "test content"
        results = [await engine.query(query, top_k=5, collections=["test"]) for _ in range(5)]

        # All results should be byte-identical
        for i in range(1, len(results)):
            assert results[0].chunks[0].chunk_id == results[i].chunks[0].chunk_id
            assert np.allclose(results[0].relevance_scores, results[i].relevance_scores, rtol=1e-9)


class TestRAGMMRDeterminism:
    """Test MMR retrieval is deterministic."""

    @pytest.mark.asyncio
    async def test_mmr_determinism(self):
        """Test MMR retrieval produces identical results."""
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.embedders import MiniLMProvider
        from greenlang.intelligence.rag.retrievers import mmr_retrieval
        from greenlang.intelligence.rag.models import Chunk, Document

        config = RAGConfig(mode="replay")
        provider = MiniLMProvider(config=config)

        # Create test documents
        texts = [
            "Natural gas is a fossil fuel",
            "Natural gas produces CO2 emissions",
            "Coal is more carbon intensive than gas",
            "Heat pumps reduce emissions",
            "Solar thermal systems use renewable energy"
        ]

        # Embed all texts
        embeddings = await provider.embed(texts)
        query_embedding = embeddings[0]  # Use first as query

        # Create Document objects
        candidates = [
            Document(
                chunk=Chunk(
                    chunk_id=f"chunk_{i}",
                    text=text,
                    doc_id=f"doc_{i}",
                    chunk_index=0,
                    collection="test"
                ),
                embedding=emb
            )
            for i, (text, emb) in enumerate(zip(texts, embeddings))
        ]

        # Run MMR multiple times
        results_list = []
        for _ in range(5):
            results = mmr_retrieval(
                query_embedding=query_embedding,
                candidates=candidates,
                lambda_mult=0.5,
                k=3
            )
            results_list.append(results)

        # All results should be identical
        for i in range(1, len(results_list)):
            assert len(results_list[0]) == len(results_list[i]), \
                f"MMR run {i} has different result count"

            for j, ((doc0, score0), (doci, scorei)) in enumerate(zip(results_list[0], results_list[i])):
                assert doc0.chunk.chunk_id == doci.chunk.chunk_id, \
                    f"MMR run {i} result {j} differs: {doc0.chunk.chunk_id} != {doci.chunk.chunk_id}"
                assert abs(score0 - scorei) < 1e-9, \
                    f"MMR run {i} score {j} differs: {score0} != {scorei}"

    @pytest.mark.asyncio
    async def test_mmr_lambda_sensitivity(self):
        """Test MMR is deterministic across lambda values."""
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.embedders import MiniLMProvider
        from greenlang.intelligence.rag.retrievers import mmr_retrieval
        from greenlang.intelligence.rag.models import Chunk, Document

        config = RAGConfig(mode="replay")
        provider = MiniLMProvider(config=config)

        texts = ["Test text " + str(i) for i in range(10)]
        embeddings = await provider.embed(texts)

        candidates = [
            Document(
                chunk=Chunk(
                    chunk_id=f"chunk_{i}",
                    text=texts[i],
                    doc_id=f"doc_{i}",
                    chunk_index=0,
                    collection="test"
                ),
                embedding=embeddings[i]
            )
            for i in range(len(texts))
        ]

        query_embedding = embeddings[0]

        # Test different lambda values produce consistent results
        for lambda_val in [0.0, 0.3, 0.5, 0.7, 1.0]:
            results1 = mmr_retrieval(query_embedding, candidates, lambda_val, k=5)
            results2 = mmr_retrieval(query_embedding, candidates, lambda_val, k=5)

            # Same lambda should always produce same results
            assert len(results1) == len(results2)
            for (doc1, score1), (doc2, score2) in zip(results1, results2):
                assert doc1.chunk.chunk_id == doc2.chunk.chunk_id
                assert abs(score1 - score2) < 1e-9


class TestRAGHashStability:
    """Test chunk and file hash stability."""

    @pytest.mark.asyncio
    async def test_chunk_hash_stability(self, temp_dir):
        """Test that chunk hashes remain stable."""
        from greenlang.intelligence.rag.engine import RAGEngine
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.models import DocMeta

        config = RAGConfig(
            mode="replay",
            embedding_provider="minilm",
            vector_store_provider="faiss",
            vector_store_path=str(temp_dir / "vector_store"),
            chunk_size=128,
            chunk_overlap=16,
            verify_checksums=True,
        )

        engine = RAGEngine(config=config)

        # Create test document
        test_doc = temp_dir / "test.txt"
        test_content = "This is a test document for hash stability testing. " * 10
        test_doc.write_text(test_content)

        doc_meta = DocMeta(
            title="Hash Test",
            source="test",
            version="1.0",
            collection="test"
        )

        # Ingest twice
        manifest1 = await engine.ingest_document(
            file_path=test_doc,
            collection="test",
            doc_meta=doc_meta
        )

        # Clear and re-ingest
        engine = RAGEngine(config=config)
        manifest2 = await engine.ingest_document(
            file_path=test_doc,
            collection="test",
            doc_meta=doc_meta
        )

        # Manifests should be identical
        assert manifest1.file_hash == manifest2.file_hash, \
            "File hash should be stable"
        assert manifest1.total_embeddings == manifest2.total_embeddings, \
            "Number of embeddings should be identical"

    @pytest.mark.asyncio
    async def test_file_hash_consistency(self, temp_dir):
        """Test file hashing is consistent."""
        from greenlang.intelligence.rag.hashing import file_hash

        test_file = temp_dir / "hash_test.txt"
        content = "Test content for hash consistency"
        test_file.write_text(content)

        # Hash multiple times
        hashes = [file_hash(test_file) for _ in range(5)]

        # All should be identical
        assert all(h == hashes[0] for h in hashes), \
            "File hashes should be consistent"


class TestFAISSDeterminism:
    """Test FAISS vector store determinism."""

    @pytest.mark.asyncio
    async def test_faiss_exact_search(self, temp_dir):
        """Test FAISS produces exact same search results."""
        from greenlang.intelligence.rag.vectorstores import FAISSVectorStore
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.models import Chunk
        import numpy as np

        config = RAGConfig(
            vector_store_path=str(temp_dir / "faiss"),
            embedding_dimension=384,
        )

        store = FAISSVectorStore(config=config)

        # Create test data
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                text=f"Test text {i}",
                doc_id=f"doc_{i}",
                chunk_index=0,
                collection="test"
            )
            for i in range(10)
        ]

        # Use deterministic random embeddings
        np.random.seed(42)
        embeddings = np.random.randn(10, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Store chunks
        await store.store_batch("test", chunks, embeddings)

        # Search multiple times
        query = embeddings[0]
        results_list = []
        for _ in range(5):
            results = store.similarity_search(query, k=5, collections=["test"])
            results_list.append(results)

        # All results should be identical
        for i in range(1, len(results_list)):
            assert len(results_list[0]) == len(results_list[i])
            for doc0, doci in zip(results_list[0], results_list[i]):
                assert doc0.chunk.chunk_id == doci.chunk.chunk_id

    @pytest.mark.asyncio
    async def test_faiss_index_stability(self, temp_dir):
        """Test FAISS index produces stable results after save/load."""
        from greenlang.intelligence.rag.vectorstores import FAISSVectorStore
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.models import Chunk
        import numpy as np

        config = RAGConfig(
            vector_store_path=str(temp_dir / "faiss"),
            embedding_dimension=384,
        )

        store = FAISSVectorStore(config=config)

        # Create and store data
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                text=f"Text {i}",
                doc_id=f"doc_{i}",
                chunk_index=0,
                collection="test"
            )
            for i in range(5)
        ]

        np.random.seed(42)
        embeddings = np.random.randn(5, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        await store.store_batch("test", chunks, embeddings)

        # Search before save
        query = embeddings[0]
        results1 = store.similarity_search(query, k=3, collections=["test"])

        # Create new store instance (loads from disk)
        store2 = FAISSVectorStore(config=config)
        results2 = store2.similarity_search(query, k=3, collections=["test"])

        # Results should be identical
        assert len(results1) == len(results2)
        for doc1, doc2 in zip(results1, results2):
            assert doc1.chunk.chunk_id == doc2.chunk.chunk_id


class TestFullPipelineReproducibility:
    """Test full RAG pipeline reproducibility for audit compliance."""

    @pytest.mark.asyncio
    async def test_full_pipeline_reproducibility(self, temp_dir):
        """Test entire RAG pipeline produces identical results across runs."""
        from greenlang.intelligence.rag.engine import RAGEngine
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.models import DocMeta

        async def run_pipeline(run_id: str):
            """Run full RAG pipeline."""
            config = RAGConfig(
                mode="replay",
                embedding_provider="minilm",
                vector_store_provider="faiss",
                vector_store_path=str(temp_dir / f"run{run_id}"),
                chunk_size=256,
                retrieval_method="mmr",
                default_top_k=5,
                mmr_lambda=0.5,
            )

            engine = RAGEngine(config=config)

            test_doc = temp_dir / "compliance.txt"
            if not test_doc.exists():
                test_doc.write_text(
                    "GHG Protocol Corporate Standard defines Scope 1 as direct emissions. "
                    "Scope 2 covers indirect emissions from purchased electricity. "
                    "Scope 3 includes all other indirect emissions in the value chain. "
                    "Natural gas combustion produces 0.0531 kg CO2e/kWh. "
                    "Coal combustion produces 0.341 kg CO2e/kWh."
                )

            doc_meta = DocMeta(
                title="GHG Protocol",
                source="test",
                version="1.0",
                collection="compliance"
            )

            await engine.ingest_document(test_doc, "compliance", doc_meta)
            result = await engine.query("What is Scope 1?", top_k=3, collections=["compliance"])

            return result

        # Run pipeline twice
        result1 = await run_pipeline("1")
        result2 = await run_pipeline("2")

        # Results should be byte-for-byte identical
        assert len(result1.chunks) == len(result2.chunks), \
            "Number of chunks must be identical for compliance"

        for i, (chunk1, chunk2) in enumerate(zip(result1.chunks, result2.chunks)):
            assert chunk1.chunk_id == chunk2.chunk_id, \
                f"Chunk {i} ID must be identical for audit trail"
            assert chunk1.text == chunk2.text, \
                f"Chunk {i} text must be identical for audit trail"

        # Scores must be identical (within floating point precision)
        assert np.allclose(result1.relevance_scores, result2.relevance_scores, rtol=1e-9), \
            "Relevance scores must be reproducible for compliance"

    @pytest.mark.asyncio
    async def test_audit_report_generation(self, temp_dir):
        """Test audit report generation for reproducibility verification."""
        from greenlang.intelligence.rag.engine import RAGEngine
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.models import DocMeta

        config = RAGConfig(
            mode="replay",
            embedding_provider="minilm",
            vector_store_provider="faiss",
            vector_store_path=str(temp_dir / "audit"),
            verify_checksums=True,
        )

        engine = RAGEngine(config=config)

        # Create test document
        test_doc = temp_dir / "audit_test.txt"
        test_content = "Audit test content for reproducibility verification"
        test_doc.write_text(test_content)

        doc_meta = DocMeta(
            title="Audit Test",
            source="compliance",
            version="1.0",
            collection="audit"
        )

        # Ingest and generate manifest
        manifest = await engine.ingest_document(test_doc, "audit", doc_meta)

        # Verify audit trail
        assert manifest.file_hash is not None, "File hash required for audit"
        assert manifest.total_embeddings > 0, "Embeddings count required"

        # Query and verify
        result = await engine.query("test", top_k=1, collections=["audit"])

        # Generate audit report
        audit_report = {
            "file_hash": manifest.file_hash,
            "embeddings_count": manifest.total_embeddings,
            "query": "test",
            "results_count": len(result.chunks),
            "chunk_ids": [chunk.chunk_id for chunk in result.chunks],
            "relevance_scores": [float(s) for s in result.relevance_scores],
            "search_time_ms": result.search_time_ms,
        }

        # Save audit report
        audit_file = temp_dir / "audit_report.json"
        with open(audit_file, "w") as f:
            json.dump(audit_report, f, indent=2)

        # Verify report was created
        assert audit_file.exists(), "Audit report must be generated"

        # Verify report contents
        with open(audit_file, "r") as f:
            loaded_report = json.load(f)

        assert loaded_report["file_hash"] == manifest.file_hash, \
            "Audit report hash must match manifest"

        return audit_report


# Fixtures
@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for tests."""
    return tmp_path


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
