"""
Configuration for pgvector infrastructure.

Provides environment-aware configuration for embedding models,
search parameters, database connections, and index tuning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


class IndexType(str, Enum):
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# Operator class mapping for pgvector
DISTANCE_OPS = {
    DistanceMetric.COSINE: "vector_cosine_ops",
    DistanceMetric.L2: "vector_l2_ops",
    DistanceMetric.INNER_PRODUCT: "vector_ip_ops",
}

# Distance operator mapping for queries
DISTANCE_OPERATORS = {
    DistanceMetric.COSINE: "<=>",
    DistanceMetric.L2: "<->",
    DistanceMetric.INNER_PRODUCT: "<#>",
}


@dataclass
class EmbeddingModelSpec:
    name: str
    dimensions: int
    max_tokens: int = 512
    normalize: bool = True
    description: str = ""


# Supported embedding models
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": EmbeddingModelSpec(
        name="all-MiniLM-L6-v2",
        dimensions=384,
        max_tokens=256,
        description="Fast, lightweight semantic search (5ms latency)",
    ),
    "all-mpnet-base-v2": EmbeddingModelSpec(
        name="all-mpnet-base-v2",
        dimensions=768,
        max_tokens=384,
        description="High-quality retrieval (15ms latency)",
    ),
    "text-embedding-3-small": EmbeddingModelSpec(
        name="text-embedding-3-small",
        dimensions=1536,
        max_tokens=8191,
        description="OpenAI compatibility (50ms latency)",
    ),
}


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    batch_size: int = 1000
    normalize: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_retries: int = 3
    retry_delay: float = 1.0
    device: str = "cpu"  # "cpu" or "cuda"

    def __post_init__(self):
        if self.model_name in EMBEDDING_MODELS:
            spec = EMBEDDING_MODELS[self.model_name]
            self.dimensions = spec.dimensions
            self.normalize = spec.normalize

    @property
    def model_spec(self) -> EmbeddingModelSpec:
        return EMBEDDING_MODELS.get(
            self.model_name,
            EmbeddingModelSpec(name=self.model_name, dimensions=self.dimensions),
        )


@dataclass
class SearchConfig:
    default_top_k: int = 10
    default_threshold: float = 0.7
    max_top_k: int = 1000
    ef_search: int = 100
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    enable_hybrid: bool = True
    rrf_k: int = 60  # RRF constant for hybrid search
    log_queries: bool = True


@dataclass
class IndexConfig:
    index_type: IndexType = IndexType.HNSW
    # HNSW parameters
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    # IVFFlat parameters
    ivfflat_lists: int = 1000
    # Build settings
    build_concurrently: bool = True
    maintenance_work_mem: str = "2GB"

    @classmethod
    def for_environment(cls, env: Environment) -> IndexConfig:
        configs = {
            Environment.DEVELOPMENT: cls(
                hnsw_m=8, hnsw_ef_construction=100, ivfflat_lists=100
            ),
            Environment.STAGING: cls(
                hnsw_m=16, hnsw_ef_construction=200, ivfflat_lists=500
            ),
            Environment.PRODUCTION: cls(
                hnsw_m=24, hnsw_ef_construction=400, ivfflat_lists=1000
            ),
        }
        return configs.get(env, cls())


@dataclass
class VectorDBConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "greenlang"
    user: str = "greenlang_app"
    password: str = ""
    ssl_mode: str = "require"
    min_pool_size: int = 5
    max_pool_size: int = 50
    connection_timeout: int = 30
    query_timeout: int = 30
    statement_cache_size: int = 100

    # Read replica settings
    reader_host: Optional[str] = None
    reader_port: int = 5432
    use_reader_for_search: bool = True

    @classmethod
    def from_env(cls) -> VectorDBConfig:
        return cls(
            host=os.getenv("PGVECTOR_HOST", "localhost"),
            port=int(os.getenv("PGVECTOR_PORT", "5432")),
            database=os.getenv("PGVECTOR_DATABASE", "greenlang"),
            user=os.getenv("PGVECTOR_USER", "greenlang_app"),
            password=os.getenv("PGVECTOR_PASSWORD", ""),
            ssl_mode=os.getenv("PGVECTOR_SSL_MODE", "require"),
            min_pool_size=int(os.getenv("PGVECTOR_MIN_POOL", "5")),
            max_pool_size=int(os.getenv("PGVECTOR_MAX_POOL", "50")),
            reader_host=os.getenv("PGVECTOR_READER_HOST"),
            reader_port=int(os.getenv("PGVECTOR_READER_PORT", "5432")),
        )

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )

    @property
    def reader_dsn(self) -> Optional[str]:
        if not self.reader_host:
            return None
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.reader_host}:{self.reader_port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )


@dataclass
class EnvironmentConfig:
    environment: Environment = Environment.DEVELOPMENT
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    database: VectorDBConfig = field(default_factory=VectorDBConfig)

    @classmethod
    def for_environment(cls, env: str) -> EnvironmentConfig:
        environment = Environment(env.lower())
        search_configs = {
            Environment.DEVELOPMENT: SearchConfig(ef_search=40, log_queries=True),
            Environment.STAGING: SearchConfig(ef_search=100, log_queries=True),
            Environment.PRODUCTION: SearchConfig(ef_search=200, log_queries=True),
        }
        return cls(
            environment=environment,
            embedding=EmbeddingConfig(),
            search=search_configs.get(environment, SearchConfig()),
            index=IndexConfig.for_environment(environment),
            database=VectorDBConfig.from_env(),
        )
