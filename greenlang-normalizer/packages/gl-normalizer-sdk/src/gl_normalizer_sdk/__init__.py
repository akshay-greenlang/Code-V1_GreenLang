"""
GL-FOUND-X-003: GreenLang Normalizer Python SDK.

This SDK provides a convenient Python client for interacting with
the GreenLang Normalizer Service, enabling unit conversion, reference
resolution, and vocabulary management.

Example:
    >>> from gl_normalizer_sdk import NormalizerClient
    >>> client = NormalizerClient(base_url="http://localhost:8000")
    >>> result = await client.convert_unit("100 kg", "t")
    >>> print(result.converted_value)
    0.1
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gl-normalizer-sdk")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

from gl_normalizer_sdk.client import (
    NormalizerClient,
    AsyncNormalizerClient,
    ClientConfig,
)
from gl_normalizer_sdk.vocab_provider import (
    VocabProvider,
    LocalVocabProvider,
    RemoteVocabProvider,
)
from gl_normalizer_sdk.cache import (
    CacheBackend,
    MemoryCache,
    RedisCache,
    CacheConfig,
)

__all__ = [
    "__version__",
    # Client
    "NormalizerClient",
    "AsyncNormalizerClient",
    "ClientConfig",
    # Vocabulary Provider
    "VocabProvider",
    "LocalVocabProvider",
    "RemoteVocabProvider",
    # Cache
    "CacheBackend",
    "MemoryCache",
    "RedisCache",
    "CacheConfig",
]
