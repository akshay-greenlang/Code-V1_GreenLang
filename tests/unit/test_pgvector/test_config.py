"""Tests for pgvector configuration module."""

import os
from unittest.mock import patch

import pytest

from greenlang.data.vector.config import (
    DISTANCE_OPS,
    DISTANCE_OPERATORS,
    EMBEDDING_MODELS,
    DistanceMetric,
    EmbeddingConfig,
    Environment,
    EnvironmentConfig,
    IndexConfig,
    IndexType,
    SearchConfig,
    VectorDBConfig,
)


class TestEmbeddingConfig:
    def test_default_config(self):
        config = EmbeddingConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.dimensions == 384
        assert config.batch_size == 1000
        assert config.normalize is True

    def test_auto_dimensions_from_model(self):
        config = EmbeddingConfig(model_name="all-mpnet-base-v2")
        assert config.dimensions == 768

    def test_openai_model(self):
        config = EmbeddingConfig(model_name="text-embedding-3-small")
        assert config.dimensions == 1536

    def test_model_spec_property(self):
        config = EmbeddingConfig()
        spec = config.model_spec
        assert spec.name == "all-MiniLM-L6-v2"
        assert spec.dimensions == 384

    def test_unknown_model_keeps_dimensions(self):
        config = EmbeddingConfig(model_name="custom-model", dimensions=256)
        assert config.dimensions == 256


class TestSearchConfig:
    def test_default_config(self):
        config = SearchConfig()
        assert config.default_top_k == 10
        assert config.default_threshold == 0.7
        assert config.ef_search == 100
        assert config.rrf_k == 60

    def test_distance_metric_default(self):
        config = SearchConfig()
        assert config.distance_metric == DistanceMetric.COSINE


class TestIndexConfig:
    def test_default_config(self):
        config = IndexConfig()
        assert config.index_type == IndexType.HNSW
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construction == 200

    def test_dev_environment(self):
        config = IndexConfig.for_environment(Environment.DEVELOPMENT)
        assert config.hnsw_m == 8
        assert config.hnsw_ef_construction == 100

    def test_staging_environment(self):
        config = IndexConfig.for_environment(Environment.STAGING)
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construction == 200

    def test_production_environment(self):
        config = IndexConfig.for_environment(Environment.PRODUCTION)
        assert config.hnsw_m == 24
        assert config.hnsw_ef_construction == 400


class TestVectorDBConfig:
    def test_default_config(self):
        config = VectorDBConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "greenlang"

    def test_dsn_property(self):
        config = VectorDBConfig(
            host="db.example.com",
            port=5432,
            database="test",
            user="user",
            password="pass",
            ssl_mode="require",
        )
        dsn = config.dsn
        assert "db.example.com" in dsn
        assert "test" in dsn
        assert "user" in dsn

    def test_reader_dsn_none_when_no_reader(self):
        config = VectorDBConfig()
        assert config.reader_dsn is None

    def test_reader_dsn_when_configured(self):
        config = VectorDBConfig(reader_host="reader.example.com")
        assert config.reader_dsn is not None
        assert "reader.example.com" in config.reader_dsn

    @patch.dict(os.environ, {
        "PGVECTOR_HOST": "env-host",
        "PGVECTOR_PORT": "5433",
        "PGVECTOR_DATABASE": "env-db",
    })
    def test_from_env(self):
        config = VectorDBConfig.from_env()
        assert config.host == "env-host"
        assert config.port == 5433
        assert config.database == "env-db"


class TestEnvironmentConfig:
    def test_dev_environment(self):
        config = EnvironmentConfig.for_environment("development")
        assert config.environment == Environment.DEVELOPMENT
        assert config.search.ef_search == 40

    def test_production_environment(self):
        config = EnvironmentConfig.for_environment("production")
        assert config.environment == Environment.PRODUCTION
        assert config.search.ef_search == 200


class TestDistanceMappings:
    def test_distance_ops(self):
        assert DISTANCE_OPS[DistanceMetric.COSINE] == "vector_cosine_ops"
        assert DISTANCE_OPS[DistanceMetric.L2] == "vector_l2_ops"
        assert DISTANCE_OPS[DistanceMetric.INNER_PRODUCT] == "vector_ip_ops"

    def test_distance_operators(self):
        assert DISTANCE_OPERATORS[DistanceMetric.COSINE] == "<=>"
        assert DISTANCE_OPERATORS[DistanceMetric.L2] == "<->"
        assert DISTANCE_OPERATORS[DistanceMetric.INNER_PRODUCT] == "<#>"

    def test_embedding_models_registry(self):
        assert "all-MiniLM-L6-v2" in EMBEDDING_MODELS
        assert "all-mpnet-base-v2" in EMBEDDING_MODELS
        assert "text-embedding-3-small" in EMBEDDING_MODELS
