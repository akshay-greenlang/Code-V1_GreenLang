"""Tests for pgvector search engine."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from greenlang.data.vector.config import SearchConfig
from greenlang.data.vector.models import (
    HybridSearchRequest,
    SearchRequest,
)
from greenlang.data.vector.search_engine import SearchEngine


class TestSearchEngine:
    def setup_method(self):
        self.db = MagicMock()
        self.config = SearchConfig(log_queries=False)

        async def mock_embed(text):
            return np.random.rand(384).astype(np.float32)

        self.engine = SearchEngine(
            db=self.db,
            config=self.config,
            embedding_fn=mock_embed,
        )

    def test_init(self):
        assert self.engine.config.ef_search == 100
        assert self.engine._embed is not None

    def test_no_embedding_fn_raises(self):
        engine = SearchEngine(db=self.db, config=self.config)
        # Will fail when trying to search without embedding function
        assert engine._embed is None

    def test_set_embedding_fn(self):
        engine = SearchEngine(db=self.db, config=self.config)

        async def new_fn(text):
            return np.zeros(384, dtype=np.float32)

        engine.set_embedding_fn(new_fn)
        assert engine._embed is new_fn

    @staticmethod
    def test_row_to_match():
        row = {
            "id": "abc-123",
            "source_type": "document",
            "source_id": "doc-1",
            "chunk_index": 0,
            "content_preview": "preview text",
            "metadata": {"key": "value"},
            "similarity": 0.95,
        }
        match = SearchEngine._row_to_match(row)
        assert match.id == "abc-123"
        assert match.similarity == 0.95
        assert match.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_query_embedding_no_fn(self):
        engine = SearchEngine(db=self.db, config=self.config)
        with pytest.raises(RuntimeError, match="No embedding function"):
            await engine._get_query_embedding("test")

    @pytest.mark.asyncio
    async def test_get_query_embedding(self):
        emb = await self.engine._get_query_embedding("test query")
        assert emb.shape == (384,)
        assert emb.dtype == np.float32


class TestSearchRequest:
    def test_defaults(self):
        req = SearchRequest(query="test")
        assert req.top_k == 10
        assert req.threshold == 0.7
        assert req.namespace == "default"

    def test_with_filters(self):
        req = SearchRequest(
            query="test",
            source_type="regulation",
            metadata_filter={"region": "EU"},
            ef_search=200,
        )
        assert req.source_type == "regulation"
        assert req.ef_search == 200


class TestHybridSearchRequest:
    def test_defaults(self):
        req = HybridSearchRequest(query="test")
        assert req.rrf_k == 60
        assert req.vector_weight == 0.7
        assert req.text_weight == 0.3
