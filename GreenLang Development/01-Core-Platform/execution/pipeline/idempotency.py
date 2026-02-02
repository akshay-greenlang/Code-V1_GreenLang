"""
Idempotency guarantees for GreenLang pipelines.

This module implements idempotency patterns following Stripe/Twilio best practices
to ensure pipeline operations are safely retryable and prevent duplicate processing.

Example:
    >>> from greenlang.pipeline.idempotency import IdempotentPipeline, IdempotencyManager
    >>>
    >>> @IdempotentPipeline(ttl_seconds=3600)
    >>> def process_emissions(data: dict) -> dict:
    ...     # This will only execute once per unique input
    ...     return calculate_emissions(data)
    >>>
    >>> # First call executes
    >>> result1 = process_emissions({"activity": 100})
    >>>
    >>> # Second call returns cached result
    >>> result2 = process_emissions({"activity": 100})
    >>> assert result1 == result2
"""

import hashlib
import json
import time
import pickle
import threading
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class IdempotencyStatus(str, Enum):
    """Status of idempotent operation."""
    PENDING = "pending"      # Operation in progress
    SUCCESS = "success"      # Operation completed successfully
    FAILED = "failed"        # Operation failed
    EXPIRED = "expired"      # Result expired from cache


@dataclass
class IdempotencyResult:
    """Result of an idempotent operation."""
    key: str
    status: IdempotencyStatus
    result: Any
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_id: Optional[str] = None
    ttl_seconds: Optional[int] = None

    @property
    def is_expired(self) -> bool:
        """Check if result has expired."""
        if not self.ttl_seconds or not self.created_at:
            return False
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time

    @property
    def time_to_live(self) -> Optional[int]:
        """Get remaining TTL in seconds."""
        if not self.ttl_seconds or not self.created_at:
            return None
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        remaining = (expiry_time - datetime.utcnow()).total_seconds()
        return max(0, int(remaining))


class IdempotencyKey:
    """
    Generate and validate idempotency keys.

    Following Stripe's pattern:
    - Keys are deterministic based on input
    - Include operation context
    - Support custom key generation
    """

    @staticmethod
    def generate(
        operation: str,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        custom_key: Optional[str] = None
    ) -> str:
        """
        Generate deterministic idempotency key.

        Args:
            operation: Name of the operation
            inputs: Input parameters
            context: Additional context (user_id, org_id, etc.)
            custom_key: Override with custom key

        Returns:
            Unique idempotency key

        Example:
            >>> key = IdempotencyKey.generate(
            ...     operation="calculate_emissions",
            ...     inputs={"activity": 100, "factor": 2.5},
            ...     context={"user_id": "123", "org_id": "abc"}
            ... )
        """
        if custom_key:
            return custom_key

        # Build key components
        key_data = {
            "operation": operation,
            "inputs": IdempotencyKey._normalize_inputs(inputs),
        }

        if context:
            key_data["context"] = IdempotencyKey._normalize_inputs(context)

        # Generate SHA-256 hash
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(key_str.encode())

        # Include operation prefix for readability
        return f"{operation}:{hash_obj.hexdigest()[:32]}"

    @staticmethod
    def _normalize_inputs(data: Any) -> Any:
        """Normalize inputs for consistent hashing."""
        if isinstance(data, dict):
            return {k: IdempotencyKey._normalize_inputs(v) for k, v in sorted(data.items())}
        elif isinstance(data, (list, tuple)):
            return [IdempotencyKey._normalize_inputs(item) for item in data]
        elif isinstance(data, (datetime,)):
            return data.isoformat()
        elif hasattr(data, "dict"):  # Pydantic models
            return IdempotencyKey._normalize_inputs(data.dict())
        else:
            return data

    @staticmethod
    def validate(key: str) -> bool:
        """Validate idempotency key format."""
        if not key or not isinstance(key, str):
            return False

        # Check format: operation:hash
        parts = key.split(":")
        if len(parts) != 2:
            return False

        operation, hash_part = parts

        # Validate operation name
        if not operation or not operation.replace("_", "").replace("-", "").isalnum():
            return False

        # Validate hash part (32 hex chars)
        if len(hash_part) != 32 or not all(c in "0123456789abcdef" for c in hash_part):
            return False

        return True


class StorageBackend(ABC):
    """Abstract base class for idempotency storage backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[IdempotencyResult]:
        """Retrieve result by key."""
        pass

    @abstractmethod
    def set(self, key: str, result: IdempotencyResult, ttl: Optional[int] = None) -> None:
        """Store result with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete result by key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def lock(self, key: str, timeout: int = 30) -> bool:
        """Acquire lock for key to prevent concurrent execution."""
        pass

    @abstractmethod
    def unlock(self, key: str) -> bool:
        """Release lock for key."""
        pass


class FileStorageBackend(StorageBackend):
    """File-based storage for idempotency (development/testing)."""

    def __init__(self, base_dir: str = ".idempotency_cache"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.locks: Dict[str, threading.Lock] = {}
        self.lock_mutex = threading.Lock()

    def _get_path(self, key: str) -> Path:
        """Get file path for key."""
        # Use first 2 chars for directory sharding
        shard = key.split(":")[-1][:2] if ":" in key else key[:2]
        shard_dir = self.base_dir / shard
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[IdempotencyResult]:
        """Retrieve result from file."""
        path = self._get_path(key)
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                result = pickle.load(f)

            # Check expiration
            if result.is_expired:
                self.delete(key)
                return None

            return result
        except Exception as e:
            logger.error(f"Failed to load idempotency result: {e}")
            return None

    def set(self, key: str, result: IdempotencyResult, ttl: Optional[int] = None) -> None:
        """Store result to file."""
        if ttl:
            result.ttl_seconds = ttl

        path = self._get_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.error(f"Failed to save idempotency result: {e}")

    def delete(self, key: str) -> bool:
        """Delete result file."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if result exists and not expired."""
        result = self.get(key)
        return result is not None

    def lock(self, key: str, timeout: int = 30) -> bool:
        """Acquire thread lock for key."""
        with self.lock_mutex:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
            lock = self.locks[key]

        return lock.acquire(timeout=timeout)

    def unlock(self, key: str) -> bool:
        """Release thread lock for key."""
        with self.lock_mutex:
            if key in self.locks:
                self.locks[key].release()
                return True
        return False


class RedisStorageBackend(StorageBackend):
    """Redis-based storage for idempotency (production)."""

    def __init__(self, redis_client, prefix: str = "idempotency:"):
        """
        Initialize Redis backend.

        Args:
            redis_client: Redis client instance
            prefix: Key prefix for namespacing
        """
        self.redis = redis_client
        self.prefix = prefix

    def _get_key(self, key: str) -> str:
        """Get Redis key with prefix."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[IdempotencyResult]:
        """Retrieve result from Redis."""
        redis_key = self._get_key(key)
        try:
            data = self.redis.get(redis_key)
            if not data:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
            return None

    def set(self, key: str, result: IdempotencyResult, ttl: Optional[int] = None) -> None:
        """Store result in Redis with TTL."""
        redis_key = self._get_key(key)
        try:
            data = pickle.dumps(result)
            if ttl:
                self.redis.setex(redis_key, ttl, data)
            else:
                self.redis.set(redis_key, data)
        except Exception as e:
            logger.error(f"Redis set failed: {e}")

    def delete(self, key: str) -> bool:
        """Delete result from Redis."""
        redis_key = self._get_key(key)
        return bool(self.redis.delete(redis_key))

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        redis_key = self._get_key(key)
        return bool(self.redis.exists(redis_key))

    def lock(self, key: str, timeout: int = 30) -> bool:
        """Acquire distributed lock using Redis."""
        lock_key = f"{self._get_key(key)}:lock"
        return bool(self.redis.set(lock_key, "1", nx=True, ex=timeout))

    def unlock(self, key: str) -> bool:
        """Release distributed lock."""
        lock_key = f"{self._get_key(key)}:lock"
        return bool(self.redis.delete(lock_key))


class IdempotencyManager:
    """
    Manage idempotent operations across pipelines.

    Provides:
    - Duplicate detection
    - Result caching
    - Concurrent execution prevention
    - Multiple storage backend support
    """

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        default_ttl: int = 3600,
        enable_locking: bool = True
    ):
        """
        Initialize IdempotencyManager.

        Args:
            storage: Storage backend (defaults to FileStorage)
            default_ttl: Default TTL in seconds
            enable_locking: Enable concurrent execution prevention
        """
        self.storage = storage or FileStorageBackend()
        self.default_ttl = default_ttl
        self.enable_locking = enable_locking
        self._active_operations: Dict[str, datetime] = {}

    def check_duplicate(self, key: str) -> Optional[IdempotencyResult]:
        """
        Check if operation was already executed.

        Args:
            key: Idempotency key

        Returns:
            Previous result if exists, None otherwise
        """
        if not IdempotencyKey.validate(key):
            raise ValueError(f"Invalid idempotency key: {key}")

        result = self.storage.get(key)

        if result:
            logger.info(f"Duplicate request detected for key: {key}")

            # Check if operation is still pending
            if result.status == IdempotencyStatus.PENDING:
                # Check timeout (default 5 minutes for pending)
                pending_timeout = 300
                if (datetime.utcnow() - result.created_at).total_seconds() > pending_timeout:
                    logger.warning(f"Pending operation timeout for key: {key}")
                    self.storage.delete(key)
                    return None

                logger.info(f"Operation still pending for key: {key}")

            return result

        return None

    def begin_operation(
        self,
        key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[IdempotencyResult]]:
        """
        Begin idempotent operation.

        Args:
            key: Idempotency key
            metadata: Operation metadata

        Returns:
            Tuple of (should_execute, existing_result)
        """
        # Check for duplicate
        existing = self.check_duplicate(key)
        if existing:
            return False, existing

        # Try to acquire lock
        if self.enable_locking:
            if not self.storage.lock(key, timeout=30):
                logger.warning(f"Failed to acquire lock for key: {key}")
                # Check again after lock failure
                existing = self.check_duplicate(key)
                return False, existing

        # Mark as pending
        result = IdempotencyResult(
            key=key,
            status=IdempotencyStatus.PENDING,
            result=None,
            metadata=metadata or {},
            execution_id=hashlib.sha256(f"{key}:{time.time()}".encode()).hexdigest()[:16]
        )

        self.storage.set(key, result, ttl=300)  # 5 min TTL for pending
        self._active_operations[key] = datetime.utcnow()

        return True, None

    def complete_operation(
        self,
        key: str,
        result: Any,
        ttl: Optional[int] = None
    ) -> IdempotencyResult:
        """
        Mark operation as complete with result.

        Args:
            key: Idempotency key
            result: Operation result
            ttl: TTL for result cache

        Returns:
            IdempotencyResult object
        """
        ttl = ttl or self.default_ttl

        idempotency_result = IdempotencyResult(
            key=key,
            status=IdempotencyStatus.SUCCESS,
            result=result,
            completed_at=datetime.utcnow(),
            ttl_seconds=ttl
        )

        # Store with TTL
        self.storage.set(key, idempotency_result, ttl=ttl)

        # Release lock
        if self.enable_locking:
            self.storage.unlock(key)

        # Clean up active operations
        if key in self._active_operations:
            del self._active_operations[key]

        logger.info(f"Operation completed for key: {key}")
        return idempotency_result

    def fail_operation(
        self,
        key: str,
        error: str,
        ttl: Optional[int] = None
    ) -> IdempotencyResult:
        """
        Mark operation as failed.

        Args:
            key: Idempotency key
            error: Error message
            ttl: TTL for failed result cache

        Returns:
            IdempotencyResult object
        """
        # Use shorter TTL for failures (5 minutes default)
        ttl = ttl or min(300, self.default_ttl)

        idempotency_result = IdempotencyResult(
            key=key,
            status=IdempotencyStatus.FAILED,
            result=None,
            error=error,
            completed_at=datetime.utcnow(),
            ttl_seconds=ttl
        )

        # Store with TTL
        self.storage.set(key, idempotency_result, ttl=ttl)

        # Release lock
        if self.enable_locking:
            self.storage.unlock(key)

        # Clean up active operations
        if key in self._active_operations:
            del self._active_operations[key]

        logger.error(f"Operation failed for key: {key} - {error}")
        return idempotency_result

    def cleanup_expired(self) -> int:
        """
        Clean up expired entries.

        Returns:
            Number of entries cleaned
        """
        # This would be implemented differently per backend
        # For file backend, scan and delete expired files
        # For Redis, TTL handles this automatically
        cleaned = 0
        logger.info(f"Cleaned {cleaned} expired idempotency entries")
        return cleaned


def IdempotentPipeline(
    ttl_seconds: int = 3600,
    key_generator: Optional[Callable] = None,
    manager: Optional[IdempotencyManager] = None,
    include_context: bool = True,
    retry_on_conflict: bool = False
):
    """
    Decorator to make pipeline functions idempotent.

    Args:
        ttl_seconds: TTL for cached results
        key_generator: Custom key generation function
        manager: IdempotencyManager instance (creates default if None)
        include_context: Include context in key generation
        retry_on_conflict: Retry if concurrent execution detected

    Example:
        >>> @IdempotentPipeline(ttl_seconds=3600)
        >>> def process_data(input_data: dict) -> dict:
        ...     # This will only execute once per unique input
        ...     return expensive_processing(input_data)

        >>> # Multiple calls with same input return cached result
        >>> result1 = process_data({"id": 123})
        >>> result2 = process_data({"id": 123})  # Returns cached
        >>> assert result1 == result2
    """
    def decorator(func: Callable) -> Callable:
        # Use provided manager or create default
        idempotency_manager = manager or IdempotencyManager(default_ttl=ttl_seconds)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate idempotency key
            if key_generator:
                key = key_generator(func.__name__, args, kwargs)
            else:
                # Default key generation
                operation = func.__name__

                # Combine args and kwargs for key generation
                inputs = {
                    "args": args,
                    "kwargs": kwargs
                }

                # Add context if enabled
                context = None
                if include_context:
                    context = {
                        "module": func.__module__,
                        "timestamp": datetime.utcnow().date().isoformat()  # Daily granularity
                    }

                # Check for explicit idempotency_key in kwargs
                custom_key = kwargs.get("idempotency_key")

                key = IdempotencyKey.generate(
                    operation=operation,
                    inputs=inputs,
                    context=context,
                    custom_key=custom_key
                )

            # Remove idempotency_key from kwargs if present
            kwargs_clean = {k: v for k, v in kwargs.items() if k != "idempotency_key"}

            # Check for duplicate execution
            should_execute, existing_result = idempotency_manager.begin_operation(
                key=key,
                metadata={"function": func.__name__, "module": func.__module__}
            )

            if not should_execute:
                if existing_result:
                    if existing_result.status == IdempotencyStatus.SUCCESS:
                        logger.info(f"Returning cached result for {func.__name__}")
                        return existing_result.result
                    elif existing_result.status == IdempotencyStatus.FAILED:
                        logger.warning(f"Previous execution failed for {func.__name__}")
                        # Optionally retry failed operations
                        if not retry_on_conflict:
                            raise RuntimeError(f"Previous execution failed: {existing_result.error}")
                    elif existing_result.status == IdempotencyStatus.PENDING:
                        if retry_on_conflict:
                            # Wait and retry
                            time.sleep(1)
                            return wrapper(*args, **kwargs)
                        else:
                            raise RuntimeError(f"Operation already in progress for key: {key}")

            # Execute the function
            try:
                result = func(*args, **kwargs_clean)

                # Store successful result
                idempotency_manager.complete_operation(
                    key=key,
                    result=result,
                    ttl=ttl_seconds
                )

                return result

            except Exception as e:
                # Store failed result
                idempotency_manager.fail_operation(
                    key=key,
                    error=str(e),
                    ttl=min(300, ttl_seconds)  # Shorter TTL for failures
                )
                raise

        # Add methods to access idempotency info
        wrapper.get_idempotency_key = lambda *args, **kwargs: IdempotencyKey.generate(
            func.__name__, {"args": args, "kwargs": kwargs}, None, kwargs.get("idempotency_key")
        )
        wrapper.clear_cache = lambda key: idempotency_manager.storage.delete(key)
        wrapper.manager = idempotency_manager

        return wrapper

    return decorator


class IdempotentPipelineBase:
    """
    Base class for pipelines with built-in idempotency support.

    Inherit from this class to add automatic idempotency to pipeline execute() method.
    """

    def __init__(
        self,
        idempotency_manager: Optional[IdempotencyManager] = None,
        idempotency_ttl: int = 3600,
        enable_idempotency: bool = True
    ):
        """
        Initialize idempotent pipeline.

        Args:
            idempotency_manager: Manager instance
            idempotency_ttl: TTL for cached results
            enable_idempotency: Enable/disable idempotency
        """
        self.idempotency_manager = idempotency_manager or IdempotencyManager(
            default_ttl=idempotency_ttl
        )
        self.idempotency_ttl = idempotency_ttl
        self.enable_idempotency = enable_idempotency
        self._last_idempotency_key: Optional[str] = None
        self._last_idempotency_result: Optional[IdempotencyResult] = None

    def execute(
        self,
        input_data: Any,
        idempotency_key: Optional[str] = None,
        skip_cache: bool = False,
        **kwargs
    ) -> Any:
        """
        Execute pipeline with idempotency guarantees.

        Args:
            input_data: Pipeline input data
            idempotency_key: Custom idempotency key
            skip_cache: Skip cache lookup (force execution)
            **kwargs: Additional arguments

        Returns:
            Pipeline execution result
        """
        if not self.enable_idempotency or skip_cache:
            # Execute without idempotency
            return self._execute_pipeline(input_data, **kwargs)

        # Generate idempotency key
        if not idempotency_key:
            idempotency_key = IdempotencyKey.generate(
                operation=f"{self.__class__.__name__}.execute",
                inputs={"data": input_data, "kwargs": kwargs},
                context={"pipeline": self.__class__.__name__}
            )

        self._last_idempotency_key = idempotency_key

        # Check for duplicate execution
        should_execute, existing_result = self.idempotency_manager.begin_operation(
            key=idempotency_key,
            metadata={
                "pipeline": self.__class__.__name__,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        if not should_execute and existing_result:
            self._last_idempotency_result = existing_result

            if existing_result.status == IdempotencyStatus.SUCCESS:
                logger.info(f"Returning cached result for pipeline: {self.__class__.__name__}")

                # Update provenance with idempotency info
                if hasattr(existing_result.result, "provenance"):
                    existing_result.result.provenance["idempotency"] = {
                        "key": idempotency_key,
                        "cached": True,
                        "cached_at": existing_result.created_at.isoformat(),
                        "ttl_remaining": existing_result.time_to_live
                    }

                return existing_result.result

            elif existing_result.status == IdempotencyStatus.FAILED:
                raise RuntimeError(f"Previous execution failed: {existing_result.error}")

            elif existing_result.status == IdempotencyStatus.PENDING:
                raise RuntimeError(f"Pipeline execution already in progress: {idempotency_key}")

        # Execute pipeline
        try:
            result = self._execute_pipeline(input_data, **kwargs)

            # Add idempotency info to result
            if hasattr(result, "provenance") or isinstance(result, dict):
                idempotency_info = {
                    "key": idempotency_key,
                    "cached": False,
                    "executed_at": datetime.utcnow().isoformat(),
                    "ttl_seconds": self.idempotency_ttl
                }

                if isinstance(result, dict):
                    if "provenance" not in result:
                        result["provenance"] = {}
                    result["provenance"]["idempotency"] = idempotency_info
                else:
                    result.provenance["idempotency"] = idempotency_info

            # Store successful result
            self._last_idempotency_result = self.idempotency_manager.complete_operation(
                key=idempotency_key,
                result=result,
                ttl=self.idempotency_ttl
            )

            return result

        except Exception as e:
            # Store failed result
            self._last_idempotency_result = self.idempotency_manager.fail_operation(
                key=idempotency_key,
                error=str(e),
                ttl=min(300, self.idempotency_ttl)
            )
            raise

    def _execute_pipeline(self, input_data: Any, **kwargs) -> Any:
        """
        Actual pipeline execution logic.
        Override this in subclasses.
        """
        raise NotImplementedError("Subclasses must implement _execute_pipeline")

    def get_idempotency_status(self) -> Optional[IdempotencyResult]:
        """Get status of last idempotent operation."""
        return self._last_idempotency_result

    def clear_idempotency_cache(self, key: Optional[str] = None) -> bool:
        """
        Clear idempotency cache.

        Args:
            key: Specific key to clear (or last key if None)

        Returns:
            True if cleared successfully
        """
        key = key or self._last_idempotency_key
        if key:
            return self.idempotency_manager.storage.delete(key)
        return False


# Example usage patterns
if __name__ == "__main__":
    # Example 1: Simple function with idempotency
    @IdempotentPipeline(ttl_seconds=300)
    def calculate_emissions(activity_data: dict) -> dict:
        """Calculate emissions with idempotency."""
        print(f"Executing calculation for: {activity_data}")
        # Expensive calculation here
        return {
            "emissions": activity_data.get("amount", 0) * 2.5,
            "timestamp": datetime.utcnow().isoformat()
        }

    # Example 2: Pipeline class with idempotency
    class EmissionsPipeline(IdempotentPipelineBase):
        """Example pipeline with built-in idempotency."""

        def _execute_pipeline(self, input_data: dict, **kwargs) -> dict:
            """Execute emissions calculation pipeline."""
            print(f"Processing pipeline for: {input_data}")

            # Simulate processing
            result = {
                "input": input_data,
                "emissions": input_data.get("activity", 0) * 2.5,
                "provenance": {
                    "pipeline": self.__class__.__name__,
                    "version": "1.0.0"
                }
            }

            return result

    # Test idempotency
    print("\n=== Testing Function Idempotency ===")

    # First call - executes
    result1 = calculate_emissions({"amount": 100, "type": "fuel"})
    print(f"Result 1: {result1}")

    # Second call - returns cached
    result2 = calculate_emissions({"amount": 100, "type": "fuel"})
    print(f"Result 2 (cached): {result2}")

    assert result1 == result2

    print("\n=== Testing Pipeline Idempotency ===")

    pipeline = EmissionsPipeline(idempotency_ttl=600)

    # First execution
    output1 = pipeline.execute({"activity": 50, "category": "transport"})
    print(f"Pipeline Result 1: {output1}")

    # Second execution - returns cached
    output2 = pipeline.execute({"activity": 50, "category": "transport"})
    print(f"Pipeline Result 2 (cached): {output2}")

    assert output1 == output2

    # Check idempotency status
    status = pipeline.get_idempotency_status()
    print(f"\nIdempotency Status: {status.status}")
    print(f"TTL Remaining: {status.time_to_live} seconds")

    print("\n✅ Idempotency guarantees working correctly!")