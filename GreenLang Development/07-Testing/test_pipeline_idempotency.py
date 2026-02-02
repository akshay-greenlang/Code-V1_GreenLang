"""
Tests for pipeline idempotency guarantees.
"""

import pytest
import time
import json
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from greenlang.pipeline.idempotency import (
    IdempotencyKey,
    IdempotencyResult,
    IdempotencyStatus,
    IdempotencyManager,
    IdempotentPipeline,
    IdempotentPipelineBase,
    FileStorageBackend,
    RedisStorageBackend
)


class TestIdempotencyKey:
    """Test idempotency key generation and validation."""

    def test_generate_basic_key(self):
        """Test basic key generation."""
        key = IdempotencyKey.generate(
            operation="test_op",
            inputs={"a": 1, "b": 2}
        )

        assert key.startswith("test_op:")
        assert len(key) == len("test_op:") + 32  # operation + : + 32 char hash

    def test_generate_deterministic(self):
        """Test that same inputs generate same key."""
        inputs = {"data": [1, 2, 3], "flag": True}

        key1 = IdempotencyKey.generate("op", inputs)
        key2 = IdempotencyKey.generate("op", inputs)

        assert key1 == key2

    def test_generate_with_context(self):
        """Test key generation with context."""
        key1 = IdempotencyKey.generate(
            operation="op",
            inputs={"a": 1},
            context={"user": "123"}
        )

        key2 = IdempotencyKey.generate(
            operation="op",
            inputs={"a": 1},
            context={"user": "456"}
        )

        assert key1 != key2

    def test_generate_custom_key(self):
        """Test custom key override."""
        custom = "custom_key_123"
        key = IdempotencyKey.generate(
            operation="op",
            inputs={"a": 1},
            custom_key=custom
        )

        assert key == custom

    def test_normalize_complex_inputs(self):
        """Test normalization of complex data types."""
        inputs1 = {
            "list": [3, 1, 2],
            "dict": {"z": 1, "a": 2},
            "date": datetime(2024, 1, 1, 12, 0, 0)
        }

        inputs2 = {
            "dict": {"a": 2, "z": 1},  # Different order
            "list": [3, 1, 2],
            "date": datetime(2024, 1, 1, 12, 0, 0)
        }

        key1 = IdempotencyKey.generate("op", inputs1)
        key2 = IdempotencyKey.generate("op", inputs2)

        assert key1 == key2

    def test_validate_valid_key(self):
        """Test validation of valid keys."""
        valid_key = "operation_name:abcdef0123456789abcdef0123456789"
        assert IdempotencyKey.validate(valid_key) is True

    def test_validate_invalid_keys(self):
        """Test validation of invalid keys."""
        invalid_keys = [
            "",
            "no_hash",
            "op:",
            ":hash",
            "op:short",
            "op:invalidchars!@#$%^&*()",
            "op:toolonghash0123456789abcdef01234"
        ]

        for key in invalid_keys:
            assert IdempotencyKey.validate(key) is False


class TestIdempotencyResult:
    """Test IdempotencyResult class."""

    def test_result_creation(self):
        """Test creating idempotency result."""
        result = IdempotencyResult(
            key="test:key",
            status=IdempotencyStatus.SUCCESS,
            result={"data": "test"},
            ttl_seconds=3600
        )

        assert result.key == "test:key"
        assert result.status == IdempotencyStatus.SUCCESS
        assert result.result == {"data": "test"}
        assert result.ttl_seconds == 3600

    def test_is_expired(self):
        """Test expiration check."""
        # Not expired
        result = IdempotencyResult(
            key="test",
            status=IdempotencyStatus.SUCCESS,
            result="data",
            ttl_seconds=3600,
            created_at=datetime.utcnow()
        )
        assert result.is_expired is False

        # Expired
        result.created_at = datetime.utcnow() - timedelta(hours=2)
        assert result.is_expired is True

    def test_time_to_live(self):
        """Test TTL calculation."""
        result = IdempotencyResult(
            key="test",
            status=IdempotencyStatus.SUCCESS,
            result="data",
            ttl_seconds=3600,
            created_at=datetime.utcnow()
        )

        ttl = result.time_to_live
        assert 3595 <= ttl <= 3600


class TestFileStorageBackend:
    """Test file-based storage backend."""

    def setup_method(self):
        """Set up test directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = FileStorageBackend(self.temp_dir)

    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_and_get(self):
        """Test storing and retrieving results."""
        result = IdempotencyResult(
            key="test:key",
            status=IdempotencyStatus.SUCCESS,
            result={"value": 42}
        )

        self.storage.set("test:key", result)
        retrieved = self.storage.get("test:key")

        assert retrieved is not None
        assert retrieved.key == result.key
        assert retrieved.result == result.result

    def test_get_nonexistent(self):
        """Test getting non-existent key."""
        result = self.storage.get("nonexistent")
        assert result is None

    def test_delete(self):
        """Test deleting results."""
        result = IdempotencyResult(
            key="test:key",
            status=IdempotencyStatus.SUCCESS,
            result="data"
        )

        self.storage.set("test:key", result)
        assert self.storage.exists("test:key") is True

        deleted = self.storage.delete("test:key")
        assert deleted is True
        assert self.storage.exists("test:key") is False

    def test_expired_result(self):
        """Test automatic cleanup of expired results."""
        result = IdempotencyResult(
            key="test:key",
            status=IdempotencyStatus.SUCCESS,
            result="data",
            ttl_seconds=1,
            created_at=datetime.utcnow() - timedelta(seconds=2)
        )

        self.storage.set("test:key", result)
        retrieved = self.storage.get("test:key")

        assert retrieved is None  # Expired and deleted

    def test_lock_unlock(self):
        """Test locking mechanism."""
        acquired = self.storage.lock("test:key", timeout=1)
        assert acquired is True

        released = self.storage.unlock("test:key")
        assert released is True

    def test_concurrent_locks(self):
        """Test concurrent lock attempts."""
        lock1_acquired = False
        lock2_acquired = False

        def acquire_lock1():
            nonlocal lock1_acquired
            lock1_acquired = self.storage.lock("test:key", timeout=2)
            time.sleep(0.5)  # Hold lock
            self.storage.unlock("test:key")

        def acquire_lock2():
            time.sleep(0.1)  # Ensure lock1 goes first
            nonlocal lock2_acquired
            lock2_acquired = self.storage.lock("test:key", timeout=0.2)

        t1 = threading.Thread(target=acquire_lock1)
        t2 = threading.Thread(target=acquire_lock2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        assert lock1_acquired is True
        assert lock2_acquired is False  # Should timeout


class TestRedisStorageBackend:
    """Test Redis storage backend."""

    def setup_method(self):
        """Set up mock Redis client."""
        self.redis_mock = MagicMock()
        self.storage = RedisStorageBackend(self.redis_mock, prefix="test:")

    def test_set_with_ttl(self):
        """Test storing with TTL."""
        result = IdempotencyResult(
            key="test:key",
            status=IdempotencyStatus.SUCCESS,
            result="data"
        )

        self.storage.set("test:key", result, ttl=3600)

        self.redis_mock.setex.assert_called_once()
        call_args = self.redis_mock.setex.call_args
        assert call_args[0][0] == "test:test:key"
        assert call_args[0][1] == 3600

    def test_get_existing(self):
        """Test retrieving existing key."""
        import pickle

        result = IdempotencyResult(
            key="test:key",
            status=IdempotencyStatus.SUCCESS,
            result="data"
        )

        self.redis_mock.get.return_value = pickle.dumps(result)

        retrieved = self.storage.get("test:key")

        assert retrieved is not None
        assert retrieved.key == result.key

    def test_lock_with_redis(self):
        """Test distributed lock with Redis."""
        self.redis_mock.set.return_value = True

        acquired = self.storage.lock("test:key", timeout=30)

        assert acquired is True
        self.redis_mock.set.assert_called_with(
            "test:test:key:lock", "1", nx=True, ex=30
        )


class TestIdempotencyManager:
    """Test IdempotencyManager."""

    def setup_method(self):
        """Set up manager with file storage."""
        self.temp_dir = tempfile.mkdtemp()
        storage = FileStorageBackend(self.temp_dir)
        self.manager = IdempotencyManager(storage=storage)

    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_check_duplicate_none(self):
        """Test checking for non-existent duplicate."""
        result = self.manager.check_duplicate("test:abcdef0123456789abcdef0123456789")
        assert result is None

    def test_check_duplicate_exists(self):
        """Test detecting duplicate."""
        key = "test:abcdef0123456789abcdef0123456789"

        # First operation
        should_exec, existing = self.manager.begin_operation(key)
        assert should_exec is True
        assert existing is None

        self.manager.complete_operation(key, {"result": "data"})

        # Check duplicate
        duplicate = self.manager.check_duplicate(key)
        assert duplicate is not None
        assert duplicate.status == IdempotencyStatus.SUCCESS

    def test_begin_operation_new(self):
        """Test beginning new operation."""
        key = "test:abcdef0123456789abcdef0123456789"

        should_exec, existing = self.manager.begin_operation(key)

        assert should_exec is True
        assert existing is None

    def test_begin_operation_duplicate(self):
        """Test beginning duplicate operation."""
        key = "test:abcdef0123456789abcdef0123456789"

        # First operation
        self.manager.begin_operation(key)
        self.manager.complete_operation(key, "result")

        # Duplicate attempt
        should_exec, existing = self.manager.begin_operation(key)

        assert should_exec is False
        assert existing is not None
        assert existing.result == "result"

    def test_complete_operation(self):
        """Test completing operation."""
        key = "test:abcdef0123456789abcdef0123456789"

        self.manager.begin_operation(key)
        result = self.manager.complete_operation(key, {"data": 123}, ttl=600)

        assert result.status == IdempotencyStatus.SUCCESS
        assert result.result == {"data": 123}
        assert result.ttl_seconds == 600

    def test_fail_operation(self):
        """Test failing operation."""
        key = "test:abcdef0123456789abcdef0123456789"

        self.manager.begin_operation(key)
        result = self.manager.fail_operation(key, "Error occurred")

        assert result.status == IdempotencyStatus.FAILED
        assert result.error == "Error occurred"

    def test_pending_timeout(self):
        """Test pending operation timeout."""
        key = "test:abcdef0123456789abcdef0123456789"

        # Create pending operation
        pending = IdempotencyResult(
            key=key,
            status=IdempotencyStatus.PENDING,
            result=None,
            created_at=datetime.utcnow() - timedelta(minutes=10)
        )

        self.manager.storage.set(key, pending)

        # Should timeout and return None
        result = self.manager.check_duplicate(key)
        assert result is None


class TestIdempotentPipelineDecorator:
    """Test IdempotentPipeline decorator."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        storage = FileStorageBackend(self.temp_dir)
        self.manager = IdempotencyManager(storage=storage)

    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_decorator_basic(self):
        """Test basic decorator functionality."""
        call_count = 0

        @IdempotentPipeline(ttl_seconds=60, manager=self.manager)
        def process_data(value):
            nonlocal call_count
            call_count += 1
            return value * 2

        # First call
        result1 = process_data(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - cached
        result2 = process_data(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

    def test_decorator_different_inputs(self):
        """Test decorator with different inputs."""
        @IdempotentPipeline(ttl_seconds=60, manager=self.manager)
        def process_data(value):
            return value * 2

        result1 = process_data(5)
        result2 = process_data(10)

        assert result1 == 10
        assert result2 == 20

    def test_decorator_with_kwargs(self):
        """Test decorator with keyword arguments."""
        @IdempotentPipeline(ttl_seconds=60, manager=self.manager)
        def process_data(value, multiplier=2):
            return value * multiplier

        result1 = process_data(5, multiplier=3)
        result2 = process_data(5, multiplier=3)
        result3 = process_data(5, multiplier=4)

        assert result1 == 15
        assert result2 == 15  # Cached
        assert result3 == 20  # Different

    def test_decorator_with_exception(self):
        """Test decorator handling exceptions."""
        call_count = 0

        @IdempotentPipeline(ttl_seconds=60, manager=self.manager)
        def failing_function(value):
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")

        # First call - fails
        with pytest.raises(ValueError):
            failing_function(5)

        assert call_count == 1

        # Second call - should try again (failed results have short TTL)
        # But this depends on the retry_on_conflict setting
        with pytest.raises(RuntimeError) as exc_info:
            failing_function(5)

        assert "Previous execution failed" in str(exc_info.value)

    def test_decorator_custom_key(self):
        """Test decorator with custom idempotency key."""
        @IdempotentPipeline(ttl_seconds=60, manager=self.manager)
        def process_data(value):
            return value * 2

        # Use custom key
        result1 = process_data(5, idempotency_key="custom_key_123")
        result2 = process_data(10, idempotency_key="custom_key_123")

        # Same key returns same result despite different input
        assert result1 == result2 == 10

    def test_get_idempotency_key_method(self):
        """Test getting idempotency key from decorated function."""
        @IdempotentPipeline(ttl_seconds=60, manager=self.manager)
        def process_data(value):
            return value * 2

        key = process_data.get_idempotency_key(5)
        assert key.startswith("process_data:")

    def test_clear_cache_method(self):
        """Test clearing cache for decorated function."""
        @IdempotentPipeline(ttl_seconds=60, manager=self.manager)
        def process_data(value):
            return value * 2

        result1 = process_data(5)
        key = process_data.get_idempotency_key(5)

        # Clear cache
        cleared = process_data.clear_cache(key)
        assert cleared is True

        # Should execute again (not cached)
        result2 = process_data(5)
        assert result1 == result2


class TestIdempotentPipelineBase:
    """Test IdempotentPipelineBase class."""

    class TestPipeline(IdempotentPipelineBase):
        """Test pipeline implementation."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.execution_count = 0

        def _execute_pipeline(self, input_data, **kwargs):
            self.execution_count += 1
            return {
                "input": input_data,
                "output": input_data.get("value", 0) * 2,
                "provenance": {}
            }

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        storage = FileStorageBackend(self.temp_dir)
        manager = IdempotencyManager(storage=storage)
        self.pipeline = self.TestPipeline(idempotency_manager=manager)

    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_execute_idempotent(self):
        """Test idempotent pipeline execution."""
        input_data = {"value": 5, "type": "test"}

        # First execution
        result1 = self.pipeline.execute(input_data)
        assert result1["output"] == 10
        assert self.pipeline.execution_count == 1

        # Second execution - cached
        result2 = self.pipeline.execute(input_data)
        assert result2["output"] == 10
        assert self.pipeline.execution_count == 1  # Not incremented

        # Check idempotency info in provenance
        assert "idempotency" in result2["provenance"]
        assert result2["provenance"]["idempotency"]["cached"] is True

    def test_pipeline_execute_skip_cache(self):
        """Test skipping cache."""
        input_data = {"value": 5}

        result1 = self.pipeline.execute(input_data)
        assert self.pipeline.execution_count == 1

        # Skip cache
        result2 = self.pipeline.execute(input_data, skip_cache=True)
        assert self.pipeline.execution_count == 2

    def test_pipeline_custom_idempotency_key(self):
        """Test pipeline with custom idempotency key."""
        input1 = {"value": 5}
        input2 = {"value": 10}

        # Different inputs, same key
        result1 = self.pipeline.execute(input1, idempotency_key="custom_key")
        result2 = self.pipeline.execute(input2, idempotency_key="custom_key")

        # Should return cached result
        assert result1 == result2
        assert self.pipeline.execution_count == 1

    def test_pipeline_disable_idempotency(self):
        """Test disabling idempotency."""
        pipeline = self.TestPipeline(enable_idempotency=False)

        input_data = {"value": 5}

        result1 = pipeline.execute(input_data)
        result2 = pipeline.execute(input_data)

        assert pipeline.execution_count == 2

    def test_pipeline_get_status(self):
        """Test getting idempotency status."""
        input_data = {"value": 5}

        self.pipeline.execute(input_data)
        status = self.pipeline.get_idempotency_status()

        assert status is not None
        assert status.status == IdempotencyStatus.SUCCESS

    def test_pipeline_clear_cache(self):
        """Test clearing pipeline cache."""
        input_data = {"value": 5}

        self.pipeline.execute(input_data)
        assert self.pipeline.execution_count == 1

        # Clear cache
        cleared = self.pipeline.clear_idempotency_cache()
        assert cleared is True

        # Should execute again
        self.pipeline.execute(input_data)
        assert self.pipeline.execution_count == 2


@pytest.mark.integration
class TestIdempotencyIntegration:
    """Integration tests for idempotency system."""

    def test_concurrent_execution_prevention(self):
        """Test preventing concurrent execution of same operation."""
        temp_dir = tempfile.mkdtemp()
        storage = FileStorageBackend(temp_dir)
        manager = IdempotencyManager(storage=storage)

        results = []
        errors = []

        @IdempotentPipeline(ttl_seconds=60, manager=manager, retry_on_conflict=False)
        def slow_operation(value):
            time.sleep(0.5)  # Simulate slow operation
            return value * 2

        def execute():
            try:
                result = slow_operation(5)
                results.append(result)
            except RuntimeError as e:
                errors.append(str(e))

        # Start multiple threads
        threads = [threading.Thread(target=execute) for _ in range(3)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Only one should succeed
        assert len(results) == 1
        assert results[0] == 10

        # Others should error with "in progress"
        assert len(errors) == 2
        assert all("in progress" in e for e in errors)

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_ttl_expiration(self):
        """Test TTL expiration of cached results."""
        temp_dir = tempfile.mkdtemp()
        storage = FileStorageBackend(temp_dir)
        manager = IdempotencyManager(storage=storage)

        call_count = 0

        @IdempotentPipeline(ttl_seconds=1, manager=manager)
        def timed_operation(value):
            nonlocal call_count
            call_count += 1
            return value * 2

        # First call
        result1 = timed_operation(5)
        assert result1 == 10
        assert call_count == 1

        # Immediate second call - cached
        result2 = timed_operation(5)
        assert result2 == 10
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(1.5)

        # Should execute again
        result3 = timed_operation(5)
        assert result3 == 10
        assert call_count == 2

        shutil.rmtree(temp_dir, ignore_errors=True)