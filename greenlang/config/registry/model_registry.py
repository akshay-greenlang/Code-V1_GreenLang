"""
Model Registry for ML Platform Team.

Manages LLM models (Claude, GPT-4, local models) with:
- Model registration and versioning
- Capability tracking
- Performance metrics
- Cost tracking
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ModelProvider(str, Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"
    OLLAMA = "ollama"


class ModelCapability(str, Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"
    VISION = "vision"


class ModelMetadata(BaseModel):
    """Model metadata and configuration."""

    id: str = Field(..., description="Unique model ID")
    name: str = Field(..., description="Model name (e.g., claude-sonnet-4)")
    provider: ModelProvider
    version: str = Field(..., description="Model version")
    capabilities: List[ModelCapability] = Field(default_factory=list)

    # Configuration
    max_tokens: int = Field(default=4096)
    context_window: int = Field(default=200000)
    supports_streaming: bool = Field(default=True)
    supports_tools: bool = Field(default=True)

    # Performance metrics
    avg_latency_ms: Optional[float] = None
    avg_cost_per_1k_tokens: Optional[float] = None

    # Certification status
    certified_for_zero_hallucination: bool = Field(default=False)
    certification_date: Optional[datetime] = None

    # Usage tracking
    total_requests: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ModelRegistry:
    """
    In-memory model registry (will be backed by PostgreSQL in Phase 1).

    Manages:
    - Model registration
    - Version tracking
    - Performance metrics
    - Cost optimization
    """

    def __init__(self):
        self.models: Dict[str, ModelMetadata] = {}
        self._initialize_default_models()

    def _initialize_default_models(self):
        """Initialize with default supported models."""

        # Anthropic Claude models
        self.register_model(ModelMetadata(
            id="claude-sonnet-4",
            name="Claude Sonnet 4",
            provider=ModelProvider.ANTHROPIC,
            version="4.0",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE,
                ModelCapability.VISION
            ],
            max_tokens=8192,
            context_window=200000,
            avg_cost_per_1k_tokens=0.003,
            certified_for_zero_hallucination=True
        ))

        self.register_model(ModelMetadata(
            id="claude-opus-4",
            name="Claude Opus 4",
            provider=ModelProvider.ANTHROPIC,
            version="4.0",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE,
                ModelCapability.VISION
            ],
            max_tokens=8192,
            context_window=200000,
            avg_cost_per_1k_tokens=0.015,
            certified_for_zero_hallucination=True
        ))

        # OpenAI models
        self.register_model(ModelMetadata(
            id="gpt-4",
            name="GPT-4",
            provider=ModelProvider.OPENAI,
            version="gpt-4-0125-preview",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE
            ],
            max_tokens=4096,
            context_window=128000,
            avg_cost_per_1k_tokens=0.01,
            certified_for_zero_hallucination=False
        ))

    def register_model(self, model: ModelMetadata) -> ModelMetadata:
        """Register a new model."""
        model.updated_at = datetime.utcnow()
        self.models[model.id] = model
        return model

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model by ID."""
        return self.models.get(model_id)

    def list_models(
        self,
        provider: Optional[ModelProvider] = None,
        capability: Optional[ModelCapability] = None,
        certified_only: bool = False
    ) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = list(self.models.values())

        if provider:
            models = [m for m in models if m.provider == provider]

        if capability:
            models = [m for m in models if capability in m.capabilities]

        if certified_only:
            models = [m for m in models if m.certified_for_zero_hallucination]

        return models

    def update_metrics(
        self,
        model_id: str,
        requests: int = 0,
        tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: Optional[float] = None
    ):
        """Update model usage metrics."""
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        model.total_requests += requests
        model.total_tokens += tokens
        model.total_cost_usd += cost_usd

        if latency_ms and model.avg_latency_ms:
            # Running average
            total = model.total_requests
            model.avg_latency_ms = (
                (model.avg_latency_ms * (total - 1) + latency_ms) / total
            )
        elif latency_ms:
            model.avg_latency_ms = latency_ms

        model.updated_at = datetime.utcnow()

    def get_best_model_for_task(
        self,
        capability: ModelCapability,
        max_cost_per_1k_tokens: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
        certified_only: bool = True
    ) -> Optional[ModelMetadata]:
        """Get best model for a task based on criteria."""
        candidates = self.list_models(
            capability=capability,
            certified_only=certified_only
        )

        if max_cost_per_1k_tokens:
            candidates = [
                m for m in candidates
                if m.avg_cost_per_1k_tokens and m.avg_cost_per_1k_tokens <= max_cost_per_1k_tokens
            ]

        if max_latency_ms:
            candidates = [
                m for m in candidates
                if m.avg_latency_ms and m.avg_latency_ms <= max_latency_ms
            ]

        if not candidates:
            return None

        # Sort by cost (lowest first)
        candidates.sort(key=lambda m: m.avg_cost_per_1k_tokens or float('inf'))
        return candidates[0]


# Global registry instance
model_registry = ModelRegistry()
