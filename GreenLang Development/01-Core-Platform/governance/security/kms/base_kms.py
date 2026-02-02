"""
Base KMS Provider Abstract Class
=================================

Defines the interface for all KMS provider implementations.
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, TypedDict, Tuple
from threading import Lock
import time

logger = logging.getLogger(__name__)


class KeyAlgorithm(Enum):
    """Supported key algorithms"""
    RSA_2048 = "RSA_2048"
    RSA_3072 = "RSA_3072"
    RSA_4096 = "RSA_4096"
    ECDSA_P256 = "ECDSA_P256"
    ECDSA_P384 = "ECDSA_P384"
    ECDSA_P521 = "ECDSA_P521"
    ED25519 = "ED25519"


class SigningAlgorithm(Enum):
    """Supported signing algorithms"""
    RSASSA_PSS_SHA256 = "RSASSA_PSS_SHA256"
    RSASSA_PSS_SHA384 = "RSASSA_PSS_SHA384"
    RSASSA_PSS_SHA512 = "RSASSA_PSS_SHA512"
    RSASSA_PKCS1_V1_5_SHA256 = "RSASSA_PKCS1_V1_5_SHA256"
    RSASSA_PKCS1_V1_5_SHA384 = "RSASSA_PKCS1_V1_5_SHA384"
    RSASSA_PKCS1_V1_5_SHA512 = "RSASSA_PKCS1_V1_5_SHA512"
    ECDSA_SHA256 = "ECDSA_SHA256"
    ECDSA_SHA384 = "ECDSA_SHA384"
    ECDSA_SHA512 = "ECDSA_SHA512"
    ED25519 = "ED25519"


class KMSSignResult(TypedDict):
    """Result from KMS signing operation"""
    signature: bytes
    key_id: str
    algorithm: str
    timestamp: str
    key_version: Optional[str]
    provider: str


@dataclass
class KMSKeyInfo:
    """Information about a KMS key"""
    key_id: str
    key_arn: Optional[str]  # Full ARN/URI
    algorithm: KeyAlgorithm
    created_at: datetime
    enabled: bool
    rotation_enabled: bool
    key_version: Optional[str] = None
    public_key: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KMSConfig:
    """Configuration for KMS provider"""
    provider: str  # 'aws', 'azure', 'gcp'
    key_id: str
    region: Optional[str] = None
    endpoint_url: Optional[str] = None  # For custom endpoints
    cache_ttl_seconds: int = 300  # 5 minutes
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    timeout_seconds: int = 30
    batch_size: int = 25  # For batch operations
    async_enabled: bool = True

    # Provider-specific settings
    aws_profile: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_vault_url: Optional[str] = None
    gcp_project_id: Optional[str] = None
    gcp_location_id: Optional[str] = None
    gcp_keyring_id: Optional[str] = None


class KMSProviderError(Exception):
    """Base exception for KMS provider errors"""
    pass


class KMSKeyNotFoundError(KMSProviderError):
    """Key not found in KMS"""
    pass


class KMSSigningError(KMSProviderError):
    """Error during signing operation"""
    pass


class KMSKeyRotationError(KMSProviderError):
    """Error during key rotation"""
    pass


class KeyCache:
    """Thread-safe cache for KMS keys"""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[KMSKeyInfo, float]] = {}
        self._lock = Lock()

    def get(self, key_id: str) -> Optional[KMSKeyInfo]:
        """Get key from cache if not expired"""
        with self._lock:
            if key_id in self._cache:
                key_info, timestamp = self._cache[key_id]
                if time.time() - timestamp < self.ttl_seconds:
                    logger.debug(f"Cache hit for key {key_id}")
                    return key_info
                else:
                    del self._cache[key_id]
                    logger.debug(f"Cache expired for key {key_id}")
            return None

    def set(self, key_id: str, key_info: KMSKeyInfo):
        """Add or update key in cache"""
        with self._lock:
            self._cache[key_id] = (key_info, time.time())
            logger.debug(f"Cached key {key_id}")

    def invalidate(self, key_id: Optional[str] = None):
        """Invalidate specific key or entire cache"""
        with self._lock:
            if key_id:
                self._cache.pop(key_id, None)
                logger.debug(f"Invalidated cache for key {key_id}")
            else:
                self._cache.clear()
                logger.debug("Invalidated entire cache")


class BaseKMSProvider(ABC):
    """
    Abstract base class for KMS providers

    Provides common functionality:
    - Key caching
    - Retry logic with exponential backoff
    - Batch operations
    - Async support
    """

    def __init__(self, config: KMSConfig):
        """Initialize KMS provider"""
        self.config = config
        self.cache = KeyCache(ttl_seconds=config.cache_ttl_seconds)
        self._client = None
        self._async_client = None
        logger.info(f"Initialized {self.__class__.__name__} with key {config.key_id}")

    @abstractmethod
    def _create_client(self) -> Any:
        """Create provider-specific client"""
        pass

    @abstractmethod
    def _create_async_client(self) -> Any:
        """Create provider-specific async client"""
        pass

    @property
    def client(self):
        """Get or create synchronous client"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def async_client(self):
        """Get or create async client"""
        if self._async_client is None and self.config.async_enabled:
            self._async_client = self._create_async_client()
        return self._async_client

    @abstractmethod
    def get_key_info(self, key_id: Optional[str] = None) -> KMSKeyInfo:
        """
        Get information about a KMS key

        Args:
            key_id: Key identifier (uses config.key_id if not provided)

        Returns:
            KMSKeyInfo object

        Raises:
            KMSKeyNotFoundError: If key not found
        """
        pass

    @abstractmethod
    def sign(self, data: bytes, key_id: Optional[str] = None,
             algorithm: Optional[SigningAlgorithm] = None) -> KMSSignResult:
        """
        Sign data using KMS key

        Args:
            data: Data to sign
            key_id: Key identifier (uses config.key_id if not provided)
            algorithm: Signing algorithm (provider default if not specified)

        Returns:
            KMSSignResult with signature and metadata

        Raises:
            KMSSigningError: If signing fails
        """
        pass

    @abstractmethod
    def verify(self, data: bytes, signature: bytes,
               key_id: Optional[str] = None,
               algorithm: Optional[SigningAlgorithm] = None) -> bool:
        """
        Verify signature using KMS key

        Args:
            data: Original data
            signature: Signature to verify
            key_id: Key identifier (uses config.key_id if not provided)
            algorithm: Signing algorithm used

        Returns:
            True if signature is valid

        Raises:
            KMSSigningError: If verification fails
        """
        pass

    @abstractmethod
    def rotate_key(self, key_id: Optional[str] = None) -> str:
        """
        Rotate a KMS key

        Args:
            key_id: Key identifier (uses config.key_id if not provided)

        Returns:
            New key version identifier

        Raises:
            KMSKeyRotationError: If rotation fails
        """
        pass

    @abstractmethod
    async def sign_async(self, data: bytes, key_id: Optional[str] = None,
                        algorithm: Optional[SigningAlgorithm] = None) -> KMSSignResult:
        """Async version of sign"""
        pass

    def sign_batch(self, data_items: List[bytes],
                   key_id: Optional[str] = None,
                   algorithm: Optional[SigningAlgorithm] = None) -> List[KMSSignResult]:
        """
        Sign multiple items in batch

        Args:
            data_items: List of data to sign
            key_id: Key identifier (uses config.key_id if not provided)
            algorithm: Signing algorithm

        Returns:
            List of KMSSignResult objects
        """
        results = []
        key_id = key_id or self.config.key_id

        # Process in batches
        for i in range(0, len(data_items), self.config.batch_size):
            batch = data_items[i:i + self.config.batch_size]

            # Sign each item in batch
            for data in batch:
                try:
                    result = self.sign(data, key_id, algorithm)
                    results.append(result)
                except KMSSigningError as e:
                    logger.error(f"Failed to sign item in batch: {e}")
                    # Add error result
                    results.append(KMSSignResult(
                        signature=b"",
                        key_id=key_id,
                        algorithm=str(algorithm) if algorithm else "unknown",
                        timestamp=datetime.utcnow().isoformat(),
                        key_version=None,
                        provider=self.config.provider
                    ))

        return results

    async def sign_batch_async(self, data_items: List[bytes],
                              key_id: Optional[str] = None,
                              algorithm: Optional[SigningAlgorithm] = None) -> List[KMSSignResult]:
        """
        Async batch signing

        Args:
            data_items: List of data to sign
            key_id: Key identifier
            algorithm: Signing algorithm

        Returns:
            List of KMSSignResult objects
        """
        if not self.config.async_enabled:
            return self.sign_batch(data_items, key_id, algorithm)

        key_id = key_id or self.config.key_id

        # Create tasks for all items
        tasks = []
        for data in data_items:
            task = self.sign_async(data, key_id, algorithm)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to sign item {i} in batch: {result}")
                # Add error result
                processed_results.append(KMSSignResult(
                    signature=b"",
                    key_id=key_id,
                    algorithm=str(algorithm) if algorithm else "unknown",
                    timestamp=datetime.utcnow().isoformat(),
                    key_version=None,
                    provider=self.config.provider
                ))
            else:
                processed_results.append(result)

        return processed_results

    def retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        delay = self.config.retry_base_delay

        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(min(delay, self.config.retry_max_delay))
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed")

        raise last_exception

    def get_cached_key_info(self, key_id: Optional[str] = None) -> KMSKeyInfo:
        """
        Get key info with caching

        Args:
            key_id: Key identifier

        Returns:
            KMSKeyInfo object (from cache if available)
        """
        key_id = key_id or self.config.key_id

        # Check cache first
        cached = self.cache.get(key_id)
        if cached:
            return cached

        # Fetch from KMS
        key_info = self.get_key_info(key_id)

        # Cache the result
        self.cache.set(key_id, key_info)

        return key_info

    def invalidate_cache(self, key_id: Optional[str] = None):
        """
        Invalidate key cache

        Args:
            key_id: Specific key to invalidate (all if None)
        """
        self.cache.invalidate(key_id)

    def close(self):
        """Clean up resources"""
        self.cache.invalidate()
        if self._client:
            # Provider-specific cleanup
            pass
        if self._async_client:
            # Provider-specific cleanup
            pass