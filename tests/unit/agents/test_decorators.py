# -*- coding: utf-8 -*-
"""
Comprehensive tests for agent decorators.
Tests @deterministic, @cached, @traced decorators and utilities.
"""

import pytest
import time
import hashlib
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from greenlang.agents.decorators import (
from greenlang.determinism import deterministic_random
    deterministic,
    cached,
    traced,
    TTLCache,
    clear_all_caches,
    get_cache_stats,
    _hash_args,
    _hash_value,
    _serialize_value,
    _cache_registry
)


class TestDeterministicDecorator:
    """Test @deterministic decorator."""

    def test_deterministic_basic(self):
        """Test basic deterministic execution."""

        @deterministic(seed=42)
        def add(a, b):
            return a + b

        result1 = add(10, 20)
        result2 = add(10, 20)

        assert result1 == result2
        assert result1 == 30

    def test_deterministic_seed_setting(self):
        """Test that seed is set correctly."""

        @deterministic(seed=123)
        def random_value():
            import random
            return deterministic_random().random()

        result1 = random_value()
        result2 = random_value()

        # With same seed, should get same result
        assert result1 == result2

    def test_deterministic_numpy_seed(self):
        """Test deterministic behavior with numpy."""
        np = pytest.importorskip("numpy")

        @deterministic(seed=42)
        def numpy_random():
            return np.deterministic_random().random()

        result1 = numpy_random()
        result2 = numpy_random()

        assert result1 == result2

    def test_deterministic_no_seed(self):
        """Test deterministic decorator without seed."""

        @deterministic()
        def multiply(x, y):
            return x * y

        result = multiply(5, 6)
        assert result == 30

    def test_deterministic_metadata(self):
        """Test that deterministic metadata is attached."""

        class Result:
            def __init__(self, value):
                self.value = value

        @deterministic(seed=42)
        def create_result(value):
            return Result(value)

        result = create_result(100)

        assert hasattr(result, '_deterministic_metadata')
        assert 'input_hash' in result._deterministic_metadata
        assert 'output_hash' in result._deterministic_metadata
        assert result._deterministic_metadata['seed'] == 42

    def test_deterministic_different_inputs(self):
        """Test different inputs produce different hashes."""

        input_hashes = []

        @deterministic(seed=42)
        def capture_hash(x):
            return x * 2

        # Can't directly capture hash, but can verify behavior
        result1 = capture_hash(10)
        result2 = capture_hash(20)

        assert result1 != result2

    def test_deterministic_preserves_function_name(self):
        """Test that decorator preserves function name."""

        @deterministic(seed=42)
        def my_function():
            return "test"

        assert my_function.__name__ == "my_function"

    def test_deterministic_marker_attributes(self):
        """Test that decorator adds marker attributes."""

        @deterministic(seed=99)
        def func():
            pass

        assert hasattr(func, '_is_deterministic')
        assert func._is_deterministic is True
        assert hasattr(func, '_deterministic_seed')
        assert func._deterministic_seed == 99

    def test_deterministic_with_kwargs(self):
        """Test deterministic with keyword arguments."""

        @deterministic(seed=42)
        def calculate(a, b, operation='add'):
            if operation == 'add':
                return a + b
            return a - b

        result1 = calculate(10, 5, operation='add')
        result2 = calculate(10, 5, operation='add')

        assert result1 == result2
        assert result1 == 15

    def test_deterministic_execution_time_recorded(self):
        """Test that execution time is recorded."""

        class Result:
            pass

        @deterministic(seed=42)
        def slow_function():
            time.sleep(0.01)
            return Result()

        result = slow_function()

        assert hasattr(result, '_deterministic_metadata')
        assert 'execution_time' in result._deterministic_metadata
        assert result._deterministic_metadata['execution_time'] > 0


class TestCachedDecorator:
    """Test @cached decorator."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    def test_cached_basic(self):
        """Test basic caching functionality."""
        call_count = [0]

        @cached(ttl_seconds=60)
        def expensive_func(x):
            call_count[0] += 1
            return x * 2

        result1 = expensive_func(10)
        result2 = expensive_func(10)

        assert result1 == result2
        assert result1 == 20
        assert call_count[0] == 1  # Only called once

    def test_cached_different_args(self):
        """Test caching with different arguments."""
        call_count = [0]

        @cached(ttl_seconds=60)
        def func(x):
            call_count[0] += 1
            return x * 2

        result1 = func(10)
        result2 = func(20)

        assert result1 == 20
        assert result2 == 40
        assert call_count[0] == 2  # Called twice for different args

    def test_cached_ttl_expiration(self):
        """Test that cache expires after TTL."""
        call_count = [0]

        @cached(ttl_seconds=0.1)  # 100ms TTL
        def func(x):
            call_count[0] += 1
            return x * 2

        result1 = func(10)
        time.sleep(0.15)  # Wait for expiration
        result2 = func(10)

        assert result1 == result2
        assert call_count[0] == 2  # Called twice due to expiration

    def test_cached_max_size_lru(self):
        """Test LRU eviction when max_size reached."""
        call_count = [0]

        @cached(ttl_seconds=60, max_size=2)
        def func(x):
            call_count[0] += 1
            return x * 2

        func(1)  # Cache entry 1
        func(2)  # Cache entry 2
        func(3)  # Cache entry 3 (evicts oldest)

        call_count[0] = 0  # Reset counter

        func(1)  # Should call again (was evicted)
        func(2)  # Should use cache
        func(3)  # Should use cache

        assert call_count[0] == 1  # Only func(1) called again

    def test_cached_with_kwargs(self):
        """Test caching with keyword arguments."""
        call_count = [0]

        @cached(ttl_seconds=60)
        def func(a, b=10):
            call_count[0] += 1
            return a + b

        result1 = func(5, b=10)
        result2 = func(5, b=10)
        result3 = func(5, b=20)

        assert result1 == result2
        assert result1 == 15
        assert result3 == 25
        assert call_count[0] == 2  # Different kwargs = different cache key

    def test_cached_custom_key_func(self):
        """Test caching with custom key function."""
        call_count = [0]

        def custom_key(x, y):
            # Only use x for cache key
            return str(x)

        @cached(ttl_seconds=60, cache_key_func=custom_key)
        def func(x, y):
            call_count[0] += 1
            return x + y

        result1 = func(10, 20)
        result2 = func(10, 30)  # Different y, but same cache key

        assert result1 == result2  # Returns cached value
        assert call_count[0] == 1  # Only called once

    def test_cached_cache_clear(self):
        """Test cache clearing."""
        call_count = [0]

        @cached(ttl_seconds=60)
        def func(x):
            call_count[0] += 1
            return x * 2

        func(10)  # Cache
        func.cache_clear()  # Clear
        func(10)  # Should call again

        assert call_count[0] == 2

    def test_cached_cache_stats(self):
        """Test cache statistics."""

        @cached(ttl_seconds=60, max_size=128)
        def func(x):
            return x * 2

        func(10)
        func(20)

        stats = func.cache_stats()

        assert stats['size'] == 2
        assert stats['max_size'] == 128
        assert stats['ttl_seconds'] == 60

    def test_cached_preserves_function_name(self):
        """Test that decorator preserves function name."""

        @cached(ttl_seconds=60)
        def my_cached_func():
            return "test"

        assert my_cached_func.__name__ == "my_cached_func"

    def test_cached_multiple_functions(self):
        """Test caching multiple different functions."""

        @cached(ttl_seconds=60)
        def func1(x):
            return x * 2

        @cached(ttl_seconds=60)
        def func2(x):
            return x * 3

        func1(10)
        func2(10)

        stats = get_cache_stats()

        # Should have separate caches
        assert len(stats) >= 2


class TestTracedDecorator:
    """Test @traced decorator."""

    def test_traced_basic(self):
        """Test basic tracing functionality."""

        class MockAgent:
            pass

        agent = MockAgent()

        @traced()
        def execute(self, data):
            return {"result": data * 2}

        result = execute(agent, 10)

        assert result["result"] == 20
        assert hasattr(agent, '_provenance_context')

    def test_traced_provenance_creation(self):
        """Test that provenance context is created."""

        class TestAgent:
            pass

        agent = TestAgent()

        @traced()
        def process(self, value):
            return value + 1

        process(agent, 42)

        assert hasattr(agent, '_provenance_context')
        assert agent._provenance_context.name == "TestAgent.process"

    def test_traced_input_tracking(self):
        """Test input tracking."""

        class TestAgent:
            pass

        agent = TestAgent()

        @traced(track_inputs=True)
        def process(self, x, y):
            return x + y

        process(agent, 10, 20)

        assert hasattr(agent, '_provenance_context')
        assert 'process_inputs' in agent._provenance_context.metadata

    def test_traced_output_tracking(self):
        """Test output tracking."""

        class TestAgent:
            pass

        agent = TestAgent()

        @traced(track_outputs=True)
        def process(self, x):
            return x * 2

        result = process(agent, 5)

        assert result == 10
        assert 'process_outputs' in agent._provenance_context.metadata

    def test_traced_without_tracking(self):
        """Test tracing without input/output tracking."""

        class TestAgent:
            pass

        agent = TestAgent()

        @traced(track_inputs=False, track_outputs=False)
        def process(self, x):
            return x

        process(agent, 100)

        assert hasattr(agent, '_provenance_context')
        # Should not have input/output metadata
        assert 'process_inputs' not in agent._provenance_context.metadata
        assert 'process_outputs' not in agent._provenance_context.metadata

    def test_traced_error_handling(self):
        """Test tracing with errors."""

        class TestAgent:
            pass

        agent = TestAgent()

        @traced()
        def failing_process(self, x):
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_process(agent, 10)

        # Provenance should still be recorded
        assert hasattr(agent, '_provenance_context')

    def test_traced_execution_time(self):
        """Test that execution time is recorded."""

        class TestAgent:
            pass

        agent = TestAgent()

        @traced()
        def slow_process(self, x):
            time.sleep(0.01)
            return x

        slow_process(agent, 5)

        # Context should exist with execution record
        assert hasattr(agent, '_provenance_context')

    def test_traced_preserves_function_name(self):
        """Test that decorator preserves function name."""

        @traced()
        def my_traced_func(self, x):
            return x

        assert my_traced_func.__name__ == "my_traced_func"

    @patch('greenlang.agents.decorators.ProvenanceContext')
    def test_traced_save_path(self, mock_context_class):
        """Test tracing with save path."""
        mock_ctx = MagicMock()
        mock_context_class.return_value = mock_ctx

        class TestAgent:
            pass

        agent = TestAgent()

        @traced(save_path="test_provenance.json")
        def process(self, x):
            return x * 2

        process(agent, 10)

        # Should call finalize with output_path
        mock_ctx.finalize.assert_called_once_with(output_path="test_provenance.json")


class TestTTLCache:
    """Test TTLCache class."""

    def test_cache_creation(self):
        """Test cache creation."""
        cache = TTLCache(ttl_seconds=60, max_size=100)

        assert cache.ttl_seconds == 60
        assert cache.max_size == 100
        assert len(cache.cache) == 0

    def test_cache_set_get(self):
        """Test setting and getting values."""
        cache = TTLCache(ttl_seconds=60)

        cache.set("key1", "value1")
        value = cache.get("key1")

        assert value == "value1"

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = TTLCache(ttl_seconds=0.1)

        cache.set("key1", "value1")
        time.sleep(0.15)
        value = cache.get("key1")

        assert value is None

    def test_cache_miss(self):
        """Test cache miss."""
        cache = TTLCache(ttl_seconds=60)

        value = cache.get("nonexistent")

        assert value is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when max size reached."""
        cache = TTLCache(ttl_seconds=60, max_size=2)

        cache.set("key1", "value1")
        time.sleep(0.01)  # Ensure different timestamps
        cache.set("key2", "value2")
        time.sleep(0.01)
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = TTLCache(ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert len(cache.cache) == 0
        assert cache.get("key1") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = TTLCache(ttl_seconds=120, max_size=50)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()

        assert stats['size'] == 2
        assert stats['max_size'] == 50
        assert stats['ttl_seconds'] == 120

    def test_cache_overwrites_existing(self):
        """Test that setting existing key updates value."""
        cache = TTLCache(ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key1", "value2")

        # Size should still be 1
        assert len(cache.cache) == 1
        assert cache.get("key1") == "value2"


class TestHashFunctions:
    """Test hashing utility functions."""

    def test_hash_args_simple(self):
        """Test hashing simple arguments."""
        hash1 = _hash_args((1, 2, 3), {})
        hash2 = _hash_args((1, 2, 3), {})

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest

    def test_hash_args_different(self):
        """Test that different args produce different hashes."""
        hash1 = _hash_args((1, 2, 3), {})
        hash2 = _hash_args((1, 2, 4), {})

        assert hash1 != hash2

    def test_hash_args_with_kwargs(self):
        """Test hashing with keyword arguments."""
        hash1 = _hash_args((), {"a": 1, "b": 2})
        hash2 = _hash_args((), {"a": 1, "b": 2})
        hash3 = _hash_args((), {"b": 2, "a": 1})  # Different order

        assert hash1 == hash2
        assert hash1 == hash3  # Order shouldn't matter

    def test_hash_args_mixed(self):
        """Test hashing with both args and kwargs."""
        hash1 = _hash_args((1, 2), {"c": 3, "d": 4})
        hash2 = _hash_args((1, 2), {"c": 3, "d": 4})

        assert hash1 == hash2

    def test_hash_value_simple_types(self):
        """Test hashing simple value types."""
        hash1 = _hash_value(42)
        hash2 = _hash_value(42)
        hash3 = _hash_value(43)

        assert hash1 == hash2
        assert hash1 != hash3

    def test_hash_value_strings(self):
        """Test hashing string values."""
        hash1 = _hash_value("hello")
        hash2 = _hash_value("hello")
        hash3 = _hash_value("world")

        assert hash1 == hash2
        assert hash1 != hash3

    def test_hash_value_collections(self):
        """Test hashing collections."""
        hash1 = _hash_value([1, 2, 3])
        hash2 = _hash_value([1, 2, 3])
        hash3 = _hash_value([1, 2, 4])

        assert hash1 == hash2
        assert hash1 != hash3

    def test_hash_value_dicts(self):
        """Test hashing dictionaries."""
        hash1 = _hash_value({"a": 1, "b": 2})
        hash2 = _hash_value({"b": 2, "a": 1})  # Different order
        hash3 = _hash_value({"a": 1, "b": 3})

        assert hash1 == hash2  # Order doesn't matter
        assert hash1 != hash3


class TestSerializeValue:
    """Test _serialize_value function."""

    def test_serialize_none(self):
        """Test serializing None."""
        result = _serialize_value(None)
        assert result is None

    def test_serialize_primitives(self):
        """Test serializing primitive types."""
        assert _serialize_value(True) is True
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value("hello") == "hello"

    def test_serialize_list(self):
        """Test serializing lists."""
        result = _serialize_value([1, 2, 3])
        assert result == [1, 2, 3]

    def test_serialize_tuple(self):
        """Test serializing tuples."""
        result = _serialize_value((1, 2, 3))
        assert result == [1, 2, 3]  # Converted to list

    def test_serialize_dict(self):
        """Test serializing dictionaries."""
        result = _serialize_value({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_serialize_nested(self):
        """Test serializing nested structures."""
        data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (4, 5, 6)
        }
        result = _serialize_value(data)

        assert result["list"] == [1, 2, 3]
        assert result["dict"] == {"nested": "value"}
        assert result["tuple"] == [4, 5, 6]

    def test_serialize_object_with_to_dict(self):
        """Test serializing object with to_dict method."""

        class CustomObject:
            def to_dict(self):
                return {"key": "value"}

        obj = CustomObject()
        result = _serialize_value(obj)

        assert result == {"key": "value"}

    def test_serialize_object_with_dict(self):
        """Test serializing object with __dict__."""

        class SimpleObject:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = "value2"

        obj = SimpleObject()
        result = _serialize_value(obj)

        assert "attr1" in result
        assert result["attr1"] == "value1"

    def test_serialize_fallback_to_string(self):
        """Test fallback to string representation."""

        class UnserializableObject:
            def __repr__(self):
                return "UnserializableObject()"

        obj = UnserializableObject()
        result = _serialize_value(obj)

        assert isinstance(result, str)


class TestCacheUtilities:
    """Test cache utility functions."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()

    def test_clear_all_caches(self):
        """Test clearing all caches."""

        @cached(ttl_seconds=60)
        def func1(x):
            return x * 2

        @cached(ttl_seconds=60)
        def func2(x):
            return x * 3

        func1(10)
        func2(20)

        clear_all_caches()

        # Caches should be empty
        stats = get_cache_stats()
        for cache_stats in stats.values():
            assert cache_stats['size'] == 0

    def test_get_cache_stats_empty(self):
        """Test getting stats with no caches."""
        clear_all_caches()

        stats = get_cache_stats()

        # Should have some caches registered but empty
        assert isinstance(stats, dict)

    def test_get_cache_stats_multiple_caches(self):
        """Test getting stats for multiple caches."""

        @cached(ttl_seconds=60, max_size=100)
        def func1(x):
            return x

        @cached(ttl_seconds=120, max_size=200)
        def func2(x):
            return x

        func1(1)
        func2(2)

        stats = get_cache_stats()

        assert len(stats) >= 2
        # Verify structure
        for cache_name, cache_stats in stats.items():
            assert 'size' in cache_stats
            assert 'max_size' in cache_stats
            assert 'ttl_seconds' in cache_stats


class TestDecoratorComposition:
    """Test composing multiple decorators."""

    def test_deterministic_and_cached(self):
        """Test combining @deterministic and @cached."""
        call_count = [0]

        @cached(ttl_seconds=60)
        @deterministic(seed=42)
        def complex_func(x):
            call_count[0] += 1
            import random
            return x + deterministic_random().random()

        result1 = complex_func(10)
        result2 = complex_func(10)

        # Should only call once due to cache
        assert call_count[0] == 1
        assert result1 == result2

    def test_traced_and_cached(self):
        """Test combining @traced and @cached."""

        class TestAgent:
            pass

        agent = TestAgent()
        call_count = [0]

        @traced()
        @cached(ttl_seconds=60)
        def process(self, x):
            call_count[0] += 1
            return x * 2

        result1 = process(agent, 10)
        result2 = process(agent, 10)

        # Cache should work
        assert call_count[0] == 1
        assert result1 == result2

    def test_all_three_decorators(self):
        """Test combining all three decorators."""

        class TestAgent:
            pass

        agent = TestAgent()

        @traced()
        @cached(ttl_seconds=60)
        @deterministic(seed=42)
        def complex_process(self, x):
            import random
            return x + deterministic_random().random()

        result1 = complex_process(agent, 10)
        result2 = complex_process(agent, 10)

        # All decorators should work together
        assert result1 == result2
        assert hasattr(agent, '_provenance_context')


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_cache_with_unhashable_args(self):
        """Test caching with complex arguments."""

        @cached(ttl_seconds=60)
        def func(data):
            return sum(data.values())

        result = func({"a": 1, "b": 2})
        assert result == 3

    def test_deterministic_with_exception(self):
        """Test deterministic decorator with exceptions."""

        @deterministic(seed=42)
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()

    def test_cache_registry_isolation(self):
        """Test that cache registry maintains separate caches."""

        @cached(ttl_seconds=60)
        def func_a(x):
            return x

        @cached(ttl_seconds=60)
        def func_b(x):
            return x * 2

        func_a(10)
        func_b(10)

        # Should have separate cache entries
        stats = get_cache_stats()
        assert len(stats) >= 2

    def test_traced_without_provenance_module(self):
        """Test traced decorator behavior."""
        # This test just verifies the decorator doesn't crash
        # Actual provenance integration tested separately

        class TestAgent:
            pass

        agent = TestAgent()

        @traced()
        def process(self, x):
            return x

        result = process(agent, 42)
        assert result == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
