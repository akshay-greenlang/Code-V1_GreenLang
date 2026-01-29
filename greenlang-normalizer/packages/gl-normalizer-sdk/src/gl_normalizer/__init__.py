"""
GL-FOUND-X-003: GreenLang Normalizer SDK

Python SDK for the GreenLang Unit & Reference Normalizer API.
Provides synchronous and asynchronous clients for normalizing units,
resolving entities, and managing normalization jobs.

Example:
    >>> from gl_normalizer import NormalizerClient
    >>> client = NormalizerClient(api_key="your-api-key")
    >>> result = client.normalize(100, "kWh", target_unit="MJ")
    >>> print(result.canonical_value)  # 360.0

Async Example:
    >>> import asyncio
    >>> from gl_normalizer import AsyncNormalizerClient
    >>>
    >>> async def main():
    ...     async with AsyncNormalizerClient(api_key="your-api-key") as client:
    ...         result = await client.normalize(100, "kWh", target_unit="MJ")
    ...         print(result.canonical_value)
    >>>
    >>> asyncio.run(main())
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gl-normalizer-sdk")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

# Core clients
from gl_normalizer.client import NormalizerClient
from gl_normalizer.async_client import AsyncNormalizerClient

# Models
from gl_normalizer.models import (
    # Enums
    PolicyMode,
    BatchMode,
    MatchMethod,
    EntityType,
    JobStatus,
    # Configuration
    ClientConfig,
    # Request models
    NormalizeRequest,
    NormalizeMetadata,
    ReferenceConditions,
    EntityRequest,
    EntityHints,
    # Result models
    NormalizeResult,
    EntityResult,
    BatchResult,
    BatchSummary,
    BatchItemResult,
    Job,
    Vocabulary,
    VocabularyEntry,
    # Audit models
    ConversionStep,
    ConversionTrace,
    AuditInfo,
    Warning,
)

# Exceptions
from gl_normalizer.exceptions import (
    NormalizerError,
    ConfigurationError,
    ValidationError,
    ConversionError,
    ResolutionError,
    VocabularyError,
    AuditError,
    APIError,
    RateLimitError,
    TimeoutError,
    ServiceUnavailableError,
    ConnectionError,
    JobError,
)

__all__ = [
    # Version
    "__version__",
    # Clients
    "NormalizerClient",
    "AsyncNormalizerClient",
    # Enums
    "PolicyMode",
    "BatchMode",
    "MatchMethod",
    "EntityType",
    "JobStatus",
    # Configuration
    "ClientConfig",
    # Request models
    "NormalizeRequest",
    "NormalizeMetadata",
    "ReferenceConditions",
    "EntityRequest",
    "EntityHints",
    # Result models
    "NormalizeResult",
    "EntityResult",
    "BatchResult",
    "BatchSummary",
    "BatchItemResult",
    "Job",
    "Vocabulary",
    "VocabularyEntry",
    # Audit models
    "ConversionStep",
    "ConversionTrace",
    "AuditInfo",
    "Warning",
    # Exceptions
    "NormalizerError",
    "ConfigurationError",
    "ValidationError",
    "ConversionError",
    "ResolutionError",
    "VocabularyError",
    "AuditError",
    "APIError",
    "RateLimitError",
    "TimeoutError",
    "ServiceUnavailableError",
    "ConnectionError",
    "JobError",
]
