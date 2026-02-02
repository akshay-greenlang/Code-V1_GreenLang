# -*- coding: utf-8 -*-
"""
Unit tests for RAG engine and determinism wrapper.

Tests:
1. DeterministicRAG modes (replay, record, live)
2. RAGEngine initialization and configuration
3. Query hashing and caching
4. Network isolation enforcement
5. Cache integrity verification
6. Citation generation
7. Security features (allowlist, sanitization)
"""

import pytest
import asyncio
from pathlib import Path
from datetime import date
import tempfile
import json

from greenlang.intelligence.rag import (
    RAGEngine,
    RAGConfig,
    DeterministicRAG,
    DocMeta,
    Chunk,
    RAGCitation,
    QueryResult,
)
from greenlang.intelligence.rag.hashing import query_hash, canonicalize_text


class TestDeterministicRAG:
    """Test DeterministicRAG wrapper."""

    def test_initialization_modes(self):
        """Test initialization in different modes."""
        # Replay mode (no cache file exists)
        with pytest.raises(FileNotFoundError):
            det = DeterministicRAG(
                mode="replay",
                cache_path=Path("nonexistent.json"),
            )

        # Record mode (creates empty cache)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache_path = Path(f.name)

        try:
            det = DeterministicRAG(mode="record", cache_path=cache_path)
            assert det.mode == "record"
            assert len(det.cache["queries"]) == 0

            # Live mode
            det_live = DeterministicRAG(mode="live", cache_path=cache_path)
            assert det_live.mode == "live"
        finally:
            cache_path.unlink(missing_ok=True)

    def test_cache_stats(self):
        """Test cache statistics."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache_path = Path(f.name)

        try:
            det = DeterministicRAG(mode="record", cache_path=cache_path)
            stats = det.get_cache_stats()

            assert stats["mode"] == "record"
            assert stats["num_queries"] == 0
            assert "cache_path" in stats
            assert "cache_size_bytes" in stats
        finally:
            cache_path.unlink(missing_ok=True)

    def test_cache_integrity_verification(self):
        """Test cache integrity verification."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache_path = Path(f.name)

        try:
            det = DeterministicRAG(mode="record", cache_path=cache_path)
            verification = det.verify_cache_integrity()

            assert verification["valid"] is True
            assert verification["num_errors"] == 0
            assert verification["num_queries"] == 0
        finally:
            cache_path.unlink(missing_ok=True)

    def test_cache_export_import(self):
        """Test cache export and import."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache_path1 = Path(f.name)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache_path2 = Path(f.name)

        try:
            # Create cache
            det1 = DeterministicRAG(mode="record", cache_path=cache_path1)

            # Export
            det1.export_cache(cache_path2)

            # Import into new instance
            det2 = DeterministicRAG(mode="record", cache_path=cache_path1)
            det2.import_cache(cache_path2, merge=False)

            # Verify queries are the same (ignore timestamps)
            assert det1.cache["queries"] == det2.cache["queries"]
            assert det1.cache["version"] == det2.cache["version"]
        finally:
            cache_path1.unlink(missing_ok=True)
            cache_path2.unlink(missing_ok=True)


class TestRAGEngine:
    """Test RAG engine."""

    def test_initialization(self):
        """Test engine initialization."""
        config = RAGConfig(
            mode="live",
            embedding_provider="minilm",
            vector_store_provider="faiss",
        )

        engine = RAGEngine(config)
        assert engine.config == config
        assert engine.embedder is None  # Lazy loading
        assert engine.vector_store is None

    def test_collection_stats(self):
        """Test collection statistics."""
        config = RAGConfig(mode="live")
        engine = RAGEngine(config)

        stats = engine.get_collection_stats("test_collection")
        assert "collection" in stats
        assert stats["collection"] == "test_collection"

    def test_list_collections(self):
        """Test listing collections."""
        config = RAGConfig(
            mode="live",
            allowlist=["collection1", "collection2"],
        )
        engine = RAGEngine(config)

        collections = engine.list_collections()
        assert "collection1" in collections
        assert "collection2" in collections


class TestQueryHashing:
    """Test query hashing for caching."""

    def test_deterministic_hashing(self):
        """Test query hashing is deterministic."""
        query = "emission factors"
        params = {"k": 5, "collections": ["ghg"]}

        hash1 = query_hash(query, params)
        hash2 = query_hash(query, params)

        assert hash1 == hash2

    def test_different_params_different_hash(self):
        """Test different params produce different hashes."""
        query = "emission factors"

        hash1 = query_hash(query, {"k": 5, "collections": ["ghg"]})
        hash2 = query_hash(query, {"k": 6, "collections": ["ghg"]})

        assert hash1 != hash2

    def test_canonicalization(self):
        """Test text canonicalization."""
        # Windows CRLF vs Unix LF
        text1 = "Hello\r\nWorld"
        text2 = "Hello\nWorld"

        canonical1 = canonicalize_text(text1)
        canonical2 = canonicalize_text(text2)

        assert canonical1 == canonical2


class TestCitationGeneration:
    """Test citation generation."""

    def test_citation_from_chunk(self):
        """Test citation generation from chunk."""
        doc_meta = DocMeta(
            doc_id="a3f5b2c8-d1e6-5f9a-8b7c-2d5e8f1a4b7c",
            title="GHG Protocol Corporate Standard",
            collection="ghg_protocol_corp",
            source_uri="https://ghgprotocol.org/standard.pdf",
            publisher="WRI/WBCSD",
            publication_date=date(2015, 3, 24),
            version="1.05",
            content_hash="a3f5b2c8" * 8,
            doc_hash="b2c8d1e6" * 8,
        )

        chunk = Chunk(
            chunk_id="c8d1e6f9-a4b7-5c2d-8e1f-4a7b2c5d8e1f",
            doc_id=doc_meta.doc_id,
            section_path="Chapter 7 > 7.3.1",
            section_hash="d1e6f9a4" * 8,
            page_start=45,
            paragraph=2,
            start_char=1000,
            end_char=1500,
            text="Test text",
            token_count=128,
        )

        citation = RAGCitation.from_chunk(
            chunk=chunk,
            doc_meta=doc_meta,
            relevance_score=0.87,
        )

        assert citation.doc_title == "GHG Protocol Corporate Standard"
        assert citation.publisher == "WRI/WBCSD"
        assert citation.version == "1.05"
        assert citation.section_path == "Chapter 7 > 7.3.1"
        assert citation.page_number == 45
        assert citation.paragraph == 2
        assert citation.relevance_score == 0.87
        assert citation.checksum == "a3f5b2c8"
        assert "GHG Protocol" in citation.formatted
        assert "SHA256:a3f5b2c8" in citation.formatted


class TestSecurity:
    """Test security features."""

    def test_allowlist_enforcement(self):
        """Test collection allowlist enforcement."""
        from greenlang.intelligence.rag.config import is_collection_allowed

        config = RAGConfig(
            mode="live",
            allowlist=["allowed1", "allowed2"],
        )

        assert is_collection_allowed("allowed1", config) is True
        assert is_collection_allowed("allowed2", config) is True
        assert is_collection_allowed("not_allowed", config) is False

    def test_collection_name_validation(self):
        """Test collection name validation."""
        from greenlang.intelligence.rag.sanitize import validate_collection_name

        # Valid names
        assert validate_collection_name("ghg_protocol_corp") is True
        assert validate_collection_name("test-collection") is True
        assert validate_collection_name("collection123") is True

        # Invalid names
        assert validate_collection_name("../../../etc/passwd") is False
        assert validate_collection_name("test; rm -rf /") is False
        assert validate_collection_name("") is False
        assert validate_collection_name("a" * 100) is False  # Too long

    def test_input_sanitization(self):
        """Test input sanitization."""
        from greenlang.intelligence.rag.sanitize import sanitize_rag_input

        # Test code block removal
        text = "```python\nimport os\nos.system('rm -rf /')\n```"
        sanitized = sanitize_rag_input(text)
        assert "import os" not in sanitized
        assert "[code omitted]" in sanitized

        # Test URL blocking (strict mode)
        text = "Visit https://malicious.com for info"
        sanitized = sanitize_rag_input(text, strict=True)
        assert "https://malicious.com" not in sanitized
        assert "[link omitted]" in sanitized


def test_full_integration():
    """Test full integration of engine and determinism."""
    # Create config
    config = RAGConfig(
        mode="live",
        embedding_provider="minilm",
        vector_store_provider="faiss",
        retrieval_method="mmr",
        default_top_k=6,
    )

    # Create engine
    engine = RAGEngine(config)
    assert engine is not None

    # Create determinism wrapper
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        cache_path = Path(f.name)

    try:
        det = DeterministicRAG(mode="record", cache_path=cache_path, config=config)
        stats = det.get_cache_stats()
        assert stats["num_queries"] == 0
    finally:
        cache_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
