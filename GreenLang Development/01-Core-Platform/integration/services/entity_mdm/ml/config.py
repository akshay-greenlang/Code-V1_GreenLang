# -*- coding: utf-8 -*-
"""
Configuration for Entity Resolution ML system.

This module defines all configuration settings for the ML pipeline including
model hyperparameters, vector store settings, and training configurations.

Author: GreenLang AI
Phase: 5 - Entity Resolution ML
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from pathlib import Path
import os


class WeaviateConfig(BaseModel):
    """Configuration for Weaviate vector database."""

    host: str = Field(
        default="localhost",
        description="Weaviate host address",
    )
    port: int = Field(
        default=8080,
        description="Weaviate port",
        ge=1,
        le=65535,
    )
    grpc_port: int = Field(
        default=50051,
        description="Weaviate gRPC port for faster queries",
        ge=1,
        le=65535,
    )
    scheme: str = Field(
        default="http",
        description="Connection scheme (http/https)",
    )
    auth_client_secret: Optional[str] = Field(
        default=None,
        description="Authentication secret for Weaviate Cloud",
    )
    timeout: int = Field(
        default=30,
        description="Connection timeout in seconds",
        ge=1,
    )
    startup_period: int = Field(
        default=10,
        description="Startup period in seconds to wait for Weaviate",
        ge=1,
    )
    use_embedded: bool = Field(
        default=False,
        description="Use embedded Weaviate instance for development",
    )
    persistence_data_path: Optional[Path] = Field(
        default=None,
        description="Path for embedded Weaviate data persistence",
    )

    @validator("scheme")
    def validate_scheme(cls, v: str) -> str:
        """Validate connection scheme."""
        if v not in ["http", "https"]:
            raise ValueError("Scheme must be 'http' or 'https'")
        return v

    def get_url(self) -> str:
        """
        Get the full Weaviate URL.

        Returns:
            Complete Weaviate connection URL
        """
        return f"{self.scheme}://{self.host}:{self.port}"

    class Config:
        """Pydantic configuration."""

        frozen = True


class ModelConfig(BaseModel):
    """Configuration for ML models."""

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    bert_model: str = Field(
        default="bert-base-uncased",
        description="BERT model for pairwise matching",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Embedding vector dimension",
        ge=1,
    )
    max_sequence_length: int = Field(
        default=128,
        description="Maximum sequence length for BERT",
        ge=1,
        le=512,
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for inference",
        ge=1,
    )
    device: str = Field(
        default="cpu",
        description="Device for model inference (cpu/cuda/mps)",
    )
    fp16: bool = Field(
        default=False,
        description="Use FP16 mixed precision (requires GPU)",
    )
    model_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "greenlang" / "models",
        description="Directory to cache downloaded models",
    )

    @validator("device")
    def validate_device(cls, v: str) -> str:
        """Validate device setting."""
        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v

    class Config:
        """Pydantic configuration."""

        frozen = True


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    epochs: int = Field(
        default=10,
        description="Number of training epochs",
        ge=1,
    )
    learning_rate: float = Field(
        default=2e-5,
        description="Learning rate for optimizer",
        gt=0.0,
    )
    weight_decay: float = Field(
        default=0.01,
        description="Weight decay for optimizer",
        ge=0.0,
    )
    warmup_steps: int = Field(
        default=500,
        description="Number of warmup steps for learning rate scheduler",
        ge=0,
    )
    validation_split: float = Field(
        default=0.2,
        description="Fraction of data to use for validation",
        gt=0.0,
        lt=1.0,
    )
    early_stopping_patience: int = Field(
        default=3,
        description="Number of epochs with no improvement before early stopping",
        ge=1,
    )
    checkpoint_dir: Path = Field(
        default=Path("./checkpoints"),
        description="Directory to save model checkpoints",
    )
    save_best_only: bool = Field(
        default=True,
        description="Only save the best model during training",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of gradient accumulation steps",
        ge=1,
    )
    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
        gt=0.0,
    )

    class Config:
        """Pydantic configuration."""

        frozen = True


class ResolutionConfig(BaseModel):
    """Configuration for entity resolution pipeline."""

    candidate_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for candidate generation",
        ge=0.0,
        le=1.0,
    )
    candidate_limit: int = Field(
        default=10,
        description="Maximum number of candidates to retrieve",
        ge=1,
    )
    rerank_threshold: float = Field(
        default=0.90,
        description="Minimum confidence score for auto-matching",
        ge=0.0,
        le=1.0,
    )
    human_review_threshold: float = Field(
        default=0.90,
        description="Scores below this require human review",
        ge=0.0,
        le=1.0,
    )
    auto_match_threshold: float = Field(
        default=0.95,
        description="Scores above this are auto-matched",
        ge=0.0,
        le=1.0,
    )
    min_candidates_required: int = Field(
        default=1,
        description="Minimum candidates required for matching",
        ge=1,
    )

    @validator("auto_match_threshold")
    def validate_thresholds(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate that auto_match_threshold >= human_review_threshold."""
        if "human_review_threshold" in values and v < values["human_review_threshold"]:
            raise ValueError(
                "auto_match_threshold must be >= human_review_threshold"
            )
        return v

    class Config:
        """Pydantic configuration."""

        frozen = True


class CacheConfig(BaseModel):
    """Configuration for Redis caching."""

    enabled: bool = Field(
        default=True,
        description="Enable Redis caching",
    )
    host: str = Field(
        default="localhost",
        description="Redis host address",
    )
    port: int = Field(
        default=6379,
        description="Redis port",
        ge=1,
        le=65535,
    )
    db: int = Field(
        default=0,
        description="Redis database number",
        ge=0,
    )
    password: Optional[str] = Field(
        default=None,
        description="Redis password",
    )
    embedding_ttl: int = Field(
        default=604800,  # 7 days
        description="TTL for cached embeddings in seconds",
        ge=1,
    )
    max_connections: int = Field(
        default=10,
        description="Maximum number of Redis connections",
        ge=1,
    )
    socket_timeout: int = Field(
        default=5,
        description="Socket timeout in seconds",
        ge=1,
    )

    class Config:
        """Pydantic configuration."""

        frozen = True


class MLConfig(BaseModel):
    """Master configuration for Entity Resolution ML system."""

    weaviate: WeaviateConfig = Field(
        default_factory=WeaviateConfig,
        description="Weaviate vector store configuration",
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="ML model configuration",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration",
    )
    resolution: ResolutionConfig = Field(
        default_factory=ResolutionConfig,
        description="Entity resolution pipeline configuration",
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Redis cache configuration",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable SOC 2 compliant audit logging",
    )

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @classmethod
    def from_env(cls) -> "MLConfig":
        """
        Create configuration from environment variables.

        Returns:
            MLConfig instance populated from environment

        Environment Variables:
            WEAVIATE_HOST: Weaviate host
            WEAVIATE_PORT: Weaviate port
            WEAVIATE_AUTH_SECRET: Weaviate authentication secret
            EMBEDDING_MODEL: Sentence transformer model name
            BERT_MODEL: BERT model name
            REDIS_HOST: Redis host
            REDIS_PORT: Redis port
            REDIS_PASSWORD: Redis password
            LOG_LEVEL: Logging level
        """
        return cls(
            weaviate=WeaviateConfig(
                host=os.getenv("WEAVIATE_HOST", "localhost"),
                port=int(os.getenv("WEAVIATE_PORT", "8080")),
                auth_client_secret=os.getenv("WEAVIATE_AUTH_SECRET"),
            ),
            model=ModelConfig(
                embedding_model=os.getenv(
                    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                bert_model=os.getenv("BERT_MODEL", "bert-base-uncased"),
            ),
            cache=CacheConfig(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD"),
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    class Config:
        """Pydantic configuration."""

        frozen = True
