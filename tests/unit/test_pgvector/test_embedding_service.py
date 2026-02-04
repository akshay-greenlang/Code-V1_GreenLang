"""Tests for pgvector embedding service."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from greenlang.data.vector.config import EmbeddingConfig
from greenlang.data.vector.embedding_service import (
    EmbeddingCache,
    EmbeddingModelLoader,
    EmbeddingService,
)
from greenlang.data.vector.models import EmbeddingRequest, EmbeddingResult


class TestEmbeddingModelLoader:
    def test_singleton_pattern(self):
        loader = EmbeddingModelLoader()
        assert loader._models == {}

    @patch("greenlang.data.vector.embedding_service.EmbeddingModelLoader.get_model")
    def test_model_caching(self, mock_get):
        loader = EmbeddingModelLoader()
        mock_model = MagicMock()
        loader._models["test:cpu"] = mock_model
        assert loader._models["test:cpu"] is mock_model


class TestEmbeddingCache:
    def test_no_redis_returns_none(self):
        cache = EmbeddingCache()
        assert cache.get("text", "model") is None

    def test_no_redis_put_noop(self):
        cache = EmbeddingCache()
        emb = np.zeros(384, dtype=np.float32)
        cache.put("text", "model", emb)  # should not raise

    def test_get_batch_no_redis(self):
        cache = EmbeddingCache()
        result = cache.get_batch(["a", "b"], "model")
        assert result == {}

    def test_cache_key_format(self):
        cache = EmbeddingCache()
        key = cache._cache_key("test text", "model-v1")
        assert key.startswith("emb:model-v1:")


class TestEmbeddingService:
    def setup_method(self):
        self.config = EmbeddingConfig(cache_enabled=False)

    @pytest.mark.asyncio
    @patch("greenlang.data.vector.embedding_service._model_loader")
    async def test_embed_basic(self, mock_loader):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_loader.get_model.return_value = mock_model

        service = EmbeddingService(self.config)
        request = EmbeddingRequest(texts=["hello", "world"])
        result = await service.embed(request)

        assert isinstance(result, EmbeddingResult)
        assert result.count == 2
        assert result.dimensions == 384
        assert result.model == "all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    @patch("greenlang.data.vector.embedding_service._model_loader")
    async def test_get_embedding_single(self, mock_loader):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_loader.get_model.return_value = mock_model

        service = EmbeddingService(self.config)
        emb = await service.get_embedding("test query")
        assert emb.shape == (384,)

    @pytest.mark.asyncio
    @patch("greenlang.data.vector.embedding_service._model_loader")
    async def test_embed_batch_processing(self, mock_loader):
        # Test with small batch size
        config = EmbeddingConfig(batch_size=2, cache_enabled=False)
        mock_model = MagicMock()

        # Return different results for each batch
        mock_model.encode.side_effect = [
            np.random.rand(2, 384).astype(np.float32),
            np.random.rand(1, 384).astype(np.float32),
        ]
        mock_loader.get_model.return_value = mock_model

        service = EmbeddingService(config)
        request = EmbeddingRequest(texts=["a", "b", "c"])
        result = await service.embed(request)

        assert result.count == 3
        assert mock_model.encode.call_count == 2  # Two batches

    @pytest.mark.asyncio
    async def test_embed_and_store_requires_db(self):
        service = EmbeddingService(self.config)
        request = EmbeddingRequest(texts=["test"])

        with pytest.raises(RuntimeError, match="Database connection not configured"):
            await service.embed_and_store(request)
