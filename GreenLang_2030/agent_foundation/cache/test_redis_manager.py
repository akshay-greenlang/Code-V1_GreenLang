"""
Unit tests for RedisManager

Tests cover:
- Connection management
- CRUD operations
- Retry logic with exponential backoff
- Health checks
- Sentinel failover
- Error handling
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from redis_manager import (
    RedisManager,
    RedisConfig,
    RedisClusterMode,
    RedisHealthStatus,
    RedisHealthCheck,
)
from redis.exceptions import ConnectionError, TimeoutError


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
async def redis_config():
    """Create test Redis configuration."""
    return RedisConfig(
        host="localhost",
        port=6379,
        db=0,
        max_connections=10,
        max_retries=3,
    )


@pytest.fixture
async def redis_manager(redis_config):
    """Create and initialize RedisManager."""
    manager = RedisManager(redis_config)
    # Mock the client for unit tests
    manager.client = AsyncMock()
    manager._is_initialized = True
    return manager


@pytest.fixture
async def redis_manager_with_real_connection(redis_config):
    """Create RedisManager with real connection (integration test)."""
    manager = RedisManager(redis_config)
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.close()


# ==============================================================================
# CONNECTION TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_redis_initialization(redis_config):
    """Test Redis initialization."""
    manager = RedisManager(redis_config)

    # Mock successful connection
    with patch('redis.asyncio.Redis') as mock_redis:
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_redis.return_value = mock_client

        with patch('redis.asyncio.ConnectionPool') as mock_pool:
            mock_pool.return_value = AsyncMock()

            await manager.initialize()

            assert manager._is_initialized
            assert manager.client is not None


@pytest.mark.asyncio
async def test_redis_standalone_mode(redis_config):
    """Test standalone mode initialization."""
    redis_config.mode = RedisClusterMode.STANDALONE

    manager = RedisManager(redis_config)

    with patch('redis.asyncio.Redis') as mock_redis:
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client

        with patch('redis.asyncio.ConnectionPool') as mock_pool:
            mock_pool.return_value = AsyncMock()

            await manager.initialize()

            assert manager.config.mode == RedisClusterMode.STANDALONE


@pytest.mark.asyncio
async def test_redis_connection_failure():
    """Test handling of connection failures."""
    config = RedisConfig(host="invalid_host", max_retries=1)
    manager = RedisManager(config)

    with pytest.raises(ConnectionError):
        await manager.initialize()


# ==============================================================================
# CRUD OPERATION TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_set_operation(redis_manager):
    """Test set operation."""
    redis_manager.client.set = AsyncMock(return_value=True)

    result = await redis_manager.set("test_key", "test_value", ttl=300)

    assert result is True
    redis_manager.client.set.assert_called_once()


@pytest.mark.asyncio
async def test_set_with_json_serialization(redis_manager):
    """Test set with JSON serialization."""
    redis_manager.client.set = AsyncMock(return_value=True)

    data = {"name": "John", "age": 30}
    result = await redis_manager.set("user:1", data, ttl=300)

    assert result is True
    # Verify JSON serialization happened
    call_args = redis_manager.client.set.call_args
    assert '"name"' in call_args[0][1]  # JSON string


@pytest.mark.asyncio
async def test_get_operation(redis_manager):
    """Test get operation."""
    redis_manager.client.get = AsyncMock(return_value=b"test_value")

    result = await redis_manager.get("test_key")

    assert result == "test_value"
    redis_manager.client.get.assert_called_once_with("test_key")


@pytest.mark.asyncio
async def test_get_with_json_deserialization(redis_manager):
    """Test get with JSON deserialization."""
    json_data = b'{"name": "John", "age": 30}'
    redis_manager.client.get = AsyncMock(return_value=json_data)

    result = await redis_manager.get("user:1", deserialize=True)

    assert isinstance(result, dict)
    assert result["name"] == "John"
    assert result["age"] == 30


@pytest.mark.asyncio
async def test_get_nonexistent_key(redis_manager):
    """Test get for non-existent key."""
    redis_manager.client.get = AsyncMock(return_value=None)

    result = await redis_manager.get("nonexistent_key")

    assert result is None


@pytest.mark.asyncio
async def test_delete_operation(redis_manager):
    """Test delete operation."""
    redis_manager.client.delete = AsyncMock(return_value=1)

    result = await redis_manager.delete("test_key")

    assert result == 1
    redis_manager.client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_delete_multiple_keys(redis_manager):
    """Test deleting multiple keys."""
    redis_manager.client.delete = AsyncMock(return_value=3)

    result = await redis_manager.delete("key1", "key2", "key3")

    assert result == 3


@pytest.mark.asyncio
async def test_exists_operation(redis_manager):
    """Test exists operation."""
    redis_manager.client.exists = AsyncMock(return_value=1)

    result = await redis_manager.exists("test_key")

    assert result == 1


@pytest.mark.asyncio
async def test_expire_operation(redis_manager):
    """Test expire operation."""
    redis_manager.client.expire = AsyncMock(return_value=True)

    result = await redis_manager.expire("test_key", 300)

    assert result is True
    redis_manager.client.expire.assert_called_once_with("test_key", 300)


@pytest.mark.asyncio
async def test_mget_operation(redis_manager):
    """Test mget operation."""
    redis_manager.client.mget = AsyncMock(
        return_value=[b"value1", b"value2", None]
    )

    result = await redis_manager.mget("key1", "key2", "key3")

    assert len(result) == 3
    assert result[0] == "value1"
    assert result[1] == "value2"
    assert result[2] is None


@pytest.mark.asyncio
async def test_increment_operation(redis_manager):
    """Test increment operation."""
    redis_manager.client.incrby = AsyncMock(return_value=5)

    result = await redis_manager.increment("counter", 2)

    assert result == 5
    redis_manager.client.incrby.assert_called_once_with("counter", 2)


# ==============================================================================
# RETRY LOGIC TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_retry_on_connection_error(redis_manager):
    """Test retry logic on connection errors."""
    # Mock connection error then success
    redis_manager.client.get = AsyncMock(
        side_effect=[ConnectionError("Connection lost"), b"test_value"]
    )

    result = await redis_manager.get("test_key")

    assert result == "test_value"
    assert redis_manager.client.get.call_count == 2
    assert redis_manager._retry_count == 1


@pytest.mark.asyncio
async def test_retry_exhaustion(redis_manager):
    """Test behavior when retries are exhausted."""
    # Mock persistent connection error
    redis_manager.client.get = AsyncMock(
        side_effect=ConnectionError("Connection lost")
    )

    with pytest.raises(Exception):  # RedisError
        await redis_manager.get("test_key")

    assert redis_manager.client.get.call_count == redis_manager.config.max_retries


@pytest.mark.asyncio
async def test_exponential_backoff():
    """Test exponential backoff timing."""
    config = RedisConfig(max_retries=3)
    manager = RedisManager(config)
    manager.client = AsyncMock()
    manager.client.get = AsyncMock(side_effect=TimeoutError("Timeout"))
    manager._is_initialized = True

    start_time = datetime.now()

    with pytest.raises(Exception):
        await manager.get("test_key")

    elapsed = (datetime.now() - start_time).total_seconds()

    # Expected wait times: 0.1s + 0.2s + 0.4s = 0.7s minimum
    assert elapsed >= 0.7


# ==============================================================================
# HEALTH CHECK TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_health_check_healthy(redis_manager):
    """Test health check when Redis is healthy."""
    redis_manager.client.ping = AsyncMock()
    redis_manager.client.info = AsyncMock(return_value={
        "connected_clients": 10,
        "used_memory": 1024 * 1024,  # 1MB
        "keyspace_hits": 900,
        "keyspace_misses": 100,
        "total_commands_processed": 10000,
        "rdb_last_save_time": 1000000,
        "aof_enabled": 1,
    })

    health = await redis_manager.health_check()

    assert health.status == RedisHealthStatus.HEALTHY
    assert health.connected_clients == 10
    assert health.hit_rate == 0.9  # 900/1000
    assert health.aof_enabled is True


@pytest.mark.asyncio
async def test_health_check_high_latency(redis_manager):
    """Test health check with high latency."""
    async def slow_ping():
        await asyncio.sleep(0.15)  # 150ms latency

    redis_manager.client.ping = slow_ping
    redis_manager.client.info = AsyncMock(return_value={
        "connected_clients": 10,
        "used_memory": 1024,
        "keyspace_hits": 100,
        "keyspace_misses": 100,
        "total_commands_processed": 1000,
        "rdb_last_save_time": 1000000,
        "aof_enabled": 0,
    })

    health = await redis_manager.health_check()

    assert health.status == RedisHealthStatus.DEGRADED
    assert health.latency_ms > 100


@pytest.mark.asyncio
async def test_health_check_unhealthy(redis_manager):
    """Test health check when Redis is unhealthy."""
    redis_manager.client.ping = AsyncMock(
        side_effect=ConnectionError("Connection failed")
    )

    health = await redis_manager.health_check()

    assert health.status == RedisHealthStatus.UNHEALTHY
    assert "failed" in health.message.lower()


# ==============================================================================
# STATISTICS TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_operation_statistics(redis_manager):
    """Test operation statistics tracking."""
    redis_manager.client.get = AsyncMock(return_value=b"value")
    redis_manager.client.set = AsyncMock(return_value=True)

    # Perform operations
    await redis_manager.get("key1")
    await redis_manager.get("key2")
    await redis_manager.set("key3", "value3")

    stats = await redis_manager.get_stats()

    assert stats["total_operations"] == 3
    assert stats["is_initialized"] is True


@pytest.mark.asyncio
async def test_failure_statistics(redis_manager):
    """Test failure statistics tracking."""
    redis_manager.client.get = AsyncMock(
        side_effect=Exception("Operation failed")
    )

    # Attempt operation (will fail)
    try:
        await redis_manager.get("key1")
    except:
        pass

    stats = await redis_manager.get_stats()

    assert stats["failed_operations"] >= 1


# ==============================================================================
# SENTINEL TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_sentinel_initialization():
    """Test Sentinel mode initialization."""
    config = RedisConfig(
        mode=RedisClusterMode.SENTINEL,
        sentinel_hosts=[
            ("localhost", 26379),
            ("localhost", 26380),
        ],
        sentinel_master_name="mymaster",
    )

    manager = RedisManager(config)

    # Mock Sentinel
    with patch('redis.asyncio.Sentinel') as mock_sentinel:
        mock_sentinel_instance = MagicMock()
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_sentinel_instance.master_for = MagicMock(return_value=mock_client)
        mock_sentinel.return_value = mock_sentinel_instance

        await manager.initialize()

        assert manager.config.mode == RedisClusterMode.SENTINEL
        mock_sentinel.assert_called_once()


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_set_operation_error_handling(redis_manager):
    """Test error handling in set operation."""
    redis_manager.client.set = AsyncMock(
        side_effect=Exception("Set failed")
    )

    with pytest.raises(Exception):
        await redis_manager.set("key", "value")

    assert redis_manager._failed_operations >= 1


@pytest.mark.asyncio
async def test_get_operation_error_handling(redis_manager):
    """Test error handling in get operation."""
    redis_manager.client.get = AsyncMock(
        side_effect=Exception("Get failed")
    )

    with pytest.raises(Exception):
        await redis_manager.get("key")

    assert redis_manager._failed_operations >= 1


# ==============================================================================
# CLEANUP TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_close_operation(redis_manager):
    """Test proper cleanup on close."""
    redis_manager.client.close = AsyncMock()
    redis_manager.pool = AsyncMock()
    redis_manager.pool.disconnect = AsyncMock()
    redis_manager._health_check_task = None

    await redis_manager.close()

    assert redis_manager._is_initialized is False
    redis_manager.client.close.assert_called_once()
    redis_manager.pool.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager():
    """Test async context manager usage."""
    config = RedisConfig()
    manager = RedisManager(config)

    # Mock initialization and close
    manager.initialize = AsyncMock()
    manager.close = AsyncMock()

    async with manager as mgr:
        assert mgr is manager
        manager.initialize.assert_called_once()

    manager.close.assert_called_once()


# ==============================================================================
# INTEGRATION TESTS (requires running Redis)
# ==============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_redis_connection():
    """Integration test with real Redis (requires Redis running)."""
    config = RedisConfig(host="localhost", port=6379)
    manager = RedisManager(config)

    try:
        await manager.initialize()

        # Test set and get
        await manager.set("test:integration", "test_value", ttl=60)
        value = await manager.get("test:integration")
        assert value == "test_value"

        # Test delete
        deleted = await manager.delete("test:integration")
        assert deleted == 1

    except ConnectionError:
        pytest.skip("Redis not running")
    finally:
        await manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
