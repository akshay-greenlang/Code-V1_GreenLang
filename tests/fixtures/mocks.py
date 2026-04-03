# -*- coding: utf-8 -*-
"""
Generic mock factories for GreenLang tests.

These are **not** pytest fixtures -- they are importable factory classes
and functions that ``conftest.py`` files (or individual tests) can use to
create consistent, pre-wired mock objects for infrastructure services.

Domain-specific mocks (EmissionFactorDB, ERP, Regulatory API, Provenance)
live in ``tests/mocks/mock_services.py``.  This module covers the generic
infrastructure layer: Prometheus, Redis, S3, HTTP, agent registries, and
database connection pools.

Usage::

    from tests.fixtures.mocks import MockRedisClient, create_mock_db_pool

    @pytest.fixture
    def redis(self):
        return MockRedisClient()
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock


# =============================================================================
# MockPrometheusRegistry
# =============================================================================


class _MockMetric:
    """Internal helper that records observe/inc/set calls."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._observations: List[float] = []
        self._labels: Dict[Tuple[str, ...], "_MockMetric"] = {}

    # -- Counter / Gauge ------------------------------------------------

    def inc(self, amount: float = 1.0) -> None:
        """Increment counter/gauge."""
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge."""
        self._value -= amount

    def set(self, value: float) -> None:
        """Set gauge value."""
        self._value = value

    # -- Histogram / Summary --------------------------------------------

    def observe(self, value: float) -> None:
        """Record an observation (histogram/summary)."""
        self._observations.append(value)

    # -- Labels ---------------------------------------------------------

    def labels(self, *label_values: str, **label_kwargs: str) -> "_MockMetric":
        """Return a child metric scoped to the given label values."""
        key = label_values or tuple(sorted(label_kwargs.values()))
        if key not in self._labels:
            child = _MockMetric(name=f"{self.name}{key}")
            self._labels[key] = child
        return self._labels[key]

    # -- Introspection --------------------------------------------------

    @property
    def value(self) -> float:
        """Current scalar value."""
        return self._value

    @property
    def observations(self) -> List[float]:
        """All histogram/summary observations."""
        return list(self._observations)


class MockPrometheusRegistry:
    """
    Mock Prometheus collector registry for tests that need metrics.

    Provides ``counter``, ``gauge``, ``histogram``, and ``summary``
    factory methods that return ``_MockMetric`` instances.  All created
    metrics are accessible via ``registry.metrics[name]``.

    Example::

        registry = MockPrometheusRegistry()
        req_counter = registry.counter("http_requests_total", "Total HTTP requests")
        req_counter.labels(method="GET", path="/health").inc()
        assert req_counter.labels(method="GET", path="/health").value == 1.0
    """

    def __init__(self) -> None:
        self.metrics: Dict[str, _MockMetric] = {}

    def _register(self, name: str, description: str = "") -> _MockMetric:
        """Create and register a metric."""
        if name not in self.metrics:
            self.metrics[name] = _MockMetric(name, description)
        return self.metrics[name]

    def counter(self, name: str, description: str = "", labelnames: Sequence[str] = ()) -> _MockMetric:
        """Create a counter metric."""
        return self._register(name, description)

    def gauge(self, name: str, description: str = "", labelnames: Sequence[str] = ()) -> _MockMetric:
        """Create a gauge metric."""
        return self._register(name, description)

    def histogram(self, name: str, description: str = "", labelnames: Sequence[str] = (), buckets: Sequence[float] = ()) -> _MockMetric:
        """Create a histogram metric."""
        return self._register(name, description)

    def summary(self, name: str, description: str = "", labelnames: Sequence[str] = ()) -> _MockMetric:
        """Create a summary metric."""
        return self._register(name, description)

    def reset(self) -> None:
        """Clear all registered metrics."""
        self.metrics.clear()


# =============================================================================
# MockRedisClient
# =============================================================================


class MockRedisClient:
    """
    Mock Redis client with in-memory dict backend.

    Supports the most common Redis operations used in GreenLang:
    ``get``, ``set``, ``delete``, ``exists``, ``expire``, ``ttl``,
    ``incr``, ``decr``, ``hset``, ``hget``, ``hgetall``, ``pipeline``,
    ``publish``, ``keys``, and ``flushdb``.

    All methods are synchronous (matching the ``redis.Redis`` interface).
    For ``redis.asyncio.Redis`` tests, use :class:`MockAsyncRedisClient`.

    Example::

        client = MockRedisClient()
        client.set("key", "value", ex=300)
        assert client.get("key") == "value"
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._expiries: Dict[str, datetime] = {}
        self._hash_store: Dict[str, Dict[str, str]] = {}
        self._pubsub_messages: List[Dict[str, Any]] = []

    # -- String commands ------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        """Get value by key, respecting TTL."""
        self._evict_if_expired(key)
        return self._store.get(key)

    def set(self, key: str, value: Any, ex: Optional[int] = None, px: Optional[int] = None, nx: bool = False, xx: bool = False) -> Optional[bool]:
        """Set key to value with optional expiry."""
        self._evict_if_expired(key)
        if nx and key in self._store:
            return None
        if xx and key not in self._store:
            return None
        self._store[key] = str(value) if not isinstance(value, (str, bytes)) else value
        if ex is not None:
            self._expiries[key] = datetime.now() + timedelta(seconds=ex)
        elif px is not None:
            self._expiries[key] = datetime.now() + timedelta(milliseconds=px)
        return True

    def delete(self, *keys: str) -> int:
        """Delete one or more keys. Returns count of deleted keys."""
        count = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                self._expiries.pop(key, None)
                count += 1
        return count

    def exists(self, *keys: str) -> int:
        """Return the number of keys that exist."""
        count = 0
        for key in keys:
            self._evict_if_expired(key)
            if key in self._store:
                count += 1
        return count

    def expire(self, key: str, seconds: int) -> bool:
        """Set TTL on an existing key."""
        if key in self._store:
            self._expiries[key] = datetime.now() + timedelta(seconds=seconds)
            return True
        return False

    def ttl(self, key: str) -> int:
        """Return TTL in seconds. -1 if no expiry, -2 if key missing."""
        self._evict_if_expired(key)
        if key not in self._store:
            return -2
        if key not in self._expiries:
            return -1
        remaining = (self._expiries[key] - datetime.now()).total_seconds()
        return max(int(remaining), 0)

    def incr(self, key: str, amount: int = 1) -> int:
        """Increment integer value."""
        self._evict_if_expired(key)
        current = int(self._store.get(key, 0))
        new_val = current + amount
        self._store[key] = str(new_val)
        return new_val

    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement integer value."""
        return self.incr(key, -amount)

    def keys(self, pattern: str = "*") -> List[str]:
        """Return all keys (pattern filtering is simplified to prefix match)."""
        if pattern == "*":
            return list(self._store.keys())
        prefix = pattern.rstrip("*")
        return [k for k in self._store if k.startswith(prefix)]

    # -- Hash commands --------------------------------------------------

    def hset(self, name: str, key: str = "", value: str = "", mapping: Optional[Dict[str, str]] = None) -> int:
        """Set hash field(s)."""
        if name not in self._hash_store:
            self._hash_store[name] = {}
        count = 0
        if key and value:
            self._hash_store[name][key] = value
            count += 1
        if mapping:
            self._hash_store[name].update(mapping)
            count += len(mapping)
        return count

    def hget(self, name: str, key: str) -> Optional[str]:
        """Get single hash field."""
        return self._hash_store.get(name, {}).get(key)

    def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        return dict(self._hash_store.get(name, {}))

    # -- Pub/Sub --------------------------------------------------------

    def publish(self, channel: str, message: str) -> int:
        """Publish message (records for assertion; returns 0 subscribers)."""
        self._pubsub_messages.append({"channel": channel, "message": message, "timestamp": datetime.now().isoformat()})
        return 0

    # -- Pipeline -------------------------------------------------------

    def pipeline(self, transaction: bool = True) -> "MockRedisPipeline":
        """Return a mock pipeline."""
        return MockRedisPipeline(self)

    # -- Admin ----------------------------------------------------------

    def flushdb(self) -> bool:
        """Clear all data."""
        self._store.clear()
        self._expiries.clear()
        self._hash_store.clear()
        return True

    # -- Internal -------------------------------------------------------

    def _evict_if_expired(self, key: str) -> None:
        """Remove key if its TTL has passed."""
        if key in self._expiries and self._expiries[key] <= datetime.now():
            self._store.pop(key, None)
            del self._expiries[key]


class MockRedisPipeline:
    """Mock Redis pipeline that batches commands."""

    def __init__(self, client: MockRedisClient) -> None:
        self._client = client
        self._commands: List[Tuple[str, tuple, dict]] = []

    def __enter__(self) -> "MockRedisPipeline":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def set(self, key: str, value: Any, **kwargs: Any) -> "MockRedisPipeline":
        """Queue set command."""
        self._commands.append(("set", (key, value), kwargs))
        return self

    def get(self, key: str) -> "MockRedisPipeline":
        """Queue get command."""
        self._commands.append(("get", (key,), {}))
        return self

    def delete(self, *keys: str) -> "MockRedisPipeline":
        """Queue delete command."""
        self._commands.append(("delete", keys, {}))
        return self

    def execute(self) -> List[Any]:
        """Execute all queued commands and return results."""
        results: List[Any] = []
        for cmd_name, args, kwargs in self._commands:
            method = getattr(self._client, cmd_name)
            results.append(method(*args, **kwargs))
        self._commands.clear()
        return results


class MockAsyncRedisClient:
    """
    Async wrapper around :class:`MockRedisClient`.

    Delegates to the synchronous mock so behaviour is identical,
    but all public methods are coroutines.
    """

    def __init__(self) -> None:
        self._sync = MockRedisClient()

    async def get(self, key: str) -> Optional[str]:
        """Async get."""
        return self._sync.get(key)

    async def set(self, key: str, value: Any, ex: Optional[int] = None, **kw: Any) -> Optional[bool]:
        """Async set."""
        return self._sync.set(key, value, ex=ex, **kw)

    async def delete(self, *keys: str) -> int:
        """Async delete."""
        return self._sync.delete(*keys)

    async def exists(self, *keys: str) -> int:
        """Async exists."""
        return self._sync.exists(*keys)

    async def flushdb(self) -> bool:
        """Async flushdb."""
        return self._sync.flushdb()


# =============================================================================
# MockS3Client
# =============================================================================


class MockS3Client:
    """
    Mock S3 / object-storage client.

    Stores objects in an in-memory dict keyed by ``(bucket, key)``.
    Supports ``put_object``, ``get_object``, ``delete_object``,
    ``list_objects``, ``head_object``, and ``copy_object``.

    Example::

        s3 = MockS3Client()
        s3.put_object("my-bucket", "reports/2024.json", b'{"total": 42}')
        obj = s3.get_object("my-bucket", "reports/2024.json")
        assert obj["Body"] == b'{"total": 42}'
    """

    def __init__(self) -> None:
        self._objects: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._buckets: Set[str] = set()

    def create_bucket(self, bucket: str) -> Dict[str, Any]:
        """Create a bucket."""
        self._buckets.add(bucket)
        return {"Location": f"/{bucket}"}

    def put_object(self, bucket: str, key: str, body: bytes, content_type: str = "application/octet-stream", metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Store an object."""
        etag = hashlib.md5(body).hexdigest()
        self._objects[(bucket, key)] = {
            "Body": body,
            "ContentType": content_type,
            "ContentLength": len(body),
            "ETag": f'"{etag}"',
            "Metadata": metadata or {},
            "LastModified": datetime.now().isoformat(),
        }
        self._buckets.add(bucket)
        return {"ETag": f'"{etag}"'}

    def get_object(self, bucket: str, key: str) -> Dict[str, Any]:
        """Retrieve an object."""
        obj = self._objects.get((bucket, key))
        if obj is None:
            raise KeyError(f"NoSuchKey: {bucket}/{key}")
        return copy.deepcopy(obj)

    def head_object(self, bucket: str, key: str) -> Dict[str, Any]:
        """Get object metadata without body."""
        obj = self._objects.get((bucket, key))
        if obj is None:
            raise KeyError(f"NoSuchKey: {bucket}/{key}")
        return {k: v for k, v in obj.items() if k != "Body"}

    def delete_object(self, bucket: str, key: str) -> Dict[str, Any]:
        """Delete an object."""
        self._objects.pop((bucket, key), None)
        return {"DeleteMarker": False}

    def list_objects(self, bucket: str, prefix: str = "", max_keys: int = 1000) -> Dict[str, Any]:
        """List objects in a bucket with optional prefix filter."""
        contents = []
        for (b, k), obj in self._objects.items():
            if b == bucket and k.startswith(prefix):
                contents.append({
                    "Key": k,
                    "Size": obj["ContentLength"],
                    "ETag": obj["ETag"],
                    "LastModified": obj["LastModified"],
                })
                if len(contents) >= max_keys:
                    break
        return {
            "Name": bucket,
            "Prefix": prefix,
            "Contents": contents,
            "KeyCount": len(contents),
            "IsTruncated": False,
        }

    def copy_object(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> Dict[str, Any]:
        """Copy an object between locations."""
        obj = self.get_object(src_bucket, src_key)
        return self.put_object(dst_bucket, dst_key, obj["Body"], obj["ContentType"], obj.get("Metadata"))


# =============================================================================
# MockHTTPClient
# =============================================================================


class MockHTTPResponse:
    """Mock HTTP response object."""

    def __init__(self, status_code: int = 200, json_data: Any = None, text: str = "", headers: Optional[Dict[str, str]] = None) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or (json.dumps(json_data) if json_data is not None else "")
        self.headers = headers or {"content-type": "application/json"}
        self.ok = 200 <= status_code < 300

    def json(self) -> Any:
        """Parse response body as JSON."""
        if self._json_data is not None:
            return self._json_data
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        """Raise if status >= 400."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class MockHTTPClient:
    """
    Mock HTTP client for API testing.

    Pre-program responses with :meth:`add_response`, then call
    ``get`` / ``post`` / ``put`` / ``delete``.  Unmatched requests
    return 404 by default.

    All methods are async to match ``httpx.AsyncClient``.

    Example::

        http = MockHTTPClient()
        http.add_response("GET", "/api/v1/health", json_data={"status": "ok"})
        resp = await http.get("/api/v1/health")
        assert resp.status_code == 200
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url
        self._responses: Dict[Tuple[str, str], MockHTTPResponse] = {}
        self._request_log: List[Dict[str, Any]] = []

    def add_response(self, method: str, path: str, status_code: int = 200, json_data: Any = None, text: str = "", headers: Optional[Dict[str, str]] = None) -> None:
        """Pre-program a response for a given method + path."""
        self._responses[(method.upper(), path)] = MockHTTPResponse(
            status_code=status_code,
            json_data=json_data,
            text=text,
            headers=headers,
        )

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> MockHTTPResponse:
        """Send GET request."""
        return self._dispatch("GET", path, params=params, headers=headers)

    async def post(self, path: str, json: Any = None, data: Any = None, headers: Optional[Dict[str, str]] = None) -> MockHTTPResponse:
        """Send POST request."""
        return self._dispatch("POST", path, body=json or data, headers=headers)

    async def put(self, path: str, json: Any = None, data: Any = None, headers: Optional[Dict[str, str]] = None) -> MockHTTPResponse:
        """Send PUT request."""
        return self._dispatch("PUT", path, body=json or data, headers=headers)

    async def delete(self, path: str, headers: Optional[Dict[str, str]] = None) -> MockHTTPResponse:
        """Send DELETE request."""
        return self._dispatch("DELETE", path, headers=headers)

    async def patch(self, path: str, json: Any = None, headers: Optional[Dict[str, str]] = None) -> MockHTTPResponse:
        """Send PATCH request."""
        return self._dispatch("PATCH", path, body=json, headers=headers)

    @property
    def request_log(self) -> List[Dict[str, Any]]:
        """All requests that have been made, for assertion."""
        return list(self._request_log)

    def reset(self) -> None:
        """Clear all programmed responses and request log."""
        self._responses.clear()
        self._request_log.clear()

    def _dispatch(self, method: str, path: str, body: Any = None, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> MockHTTPResponse:
        """Internal dispatch."""
        self._request_log.append({
            "method": method,
            "path": path,
            "body": body,
            "params": params,
            "headers": headers,
            "timestamp": datetime.now().isoformat(),
        })
        response = self._responses.get((method, path))
        if response is None:
            return MockHTTPResponse(status_code=404, json_data={"error": "Not Found", "path": path})
        return response


# =============================================================================
# Agent Registry Mock Factory
# =============================================================================


def create_mock_agents(agent_names: Optional[List[str]] = None) -> Dict[str, Mock]:
    """
    Factory to create mock agent registry entries.

    Each agent is a ``Mock`` with an ``execute`` attribute set to
    ``AsyncMock`` that returns a default success response.

    Args:
        agent_names: List of agent names. Defaults to a standard set
            covering foundation, data, MRV, and EUDR layers.

    Returns:
        Dict mapping agent name to its Mock object.

    Example::

        agents = create_mock_agents(["intake_agent", "calc_agent"])
        result = await agents["calc_agent"].execute({"input": "data"})
        assert result["status"] == "success"
    """
    if agent_names is None:
        agent_names = [
            "intake_agent",
            "validation_agent",
            "calculation_agent",
            "reporting_agent",
            "audit_agent",
            "orchestrator_agent",
        ]

    agents: Dict[str, Mock] = {}
    for name in agent_names:
        agent = Mock(name=name)
        agent.agent_name = name
        agent.execute = AsyncMock(return_value={
            "status": "success",
            "agent_name": name,
            "result": {},
            "provenance_hash": hashlib.sha256(name.encode()).hexdigest(),
            "processing_time_ms": 42.0,
        })
        agent.health_check = AsyncMock(return_value={"status": "healthy", "agent_name": name})
        agent.get_capabilities = Mock(return_value={
            "agent_name": name,
            "version": "1.0.0",
            "supported_operations": ["execute", "health_check"],
        })
        agents[name] = agent

    return agents


# =============================================================================
# Database Connection Pool Mock Factory
# =============================================================================


class MockDBConnection:
    """Mock database connection returned by the pool."""

    def __init__(self) -> None:
        self._query_results: List[Any] = []
        self._execute_count: int = 0
        self._in_transaction: bool = False

    async def execute(self, query: str, *params: Any) -> str:
        """Execute a query (no result rows)."""
        self._execute_count += 1
        return "OK"

    async def fetch(self, query: str, *params: Any) -> List[Dict[str, Any]]:
        """Fetch rows."""
        if self._query_results:
            return self._query_results.pop(0)
        return []

    async def fetchrow(self, query: str, *params: Any) -> Optional[Dict[str, Any]]:
        """Fetch single row."""
        rows = await self.fetch(query, *params)
        return rows[0] if rows else None

    async def fetchval(self, query: str, *params: Any) -> Optional[Any]:
        """Fetch single value."""
        row = await self.fetchrow(query, *params)
        if row:
            return next(iter(row.values()))
        return None

    def transaction(self) -> "MockDBTransaction":
        """Start a transaction context manager."""
        return MockDBTransaction(self)

    def set_query_results(self, *result_sets: List[Dict[str, Any]]) -> None:
        """Pre-load query results (consumed in FIFO order)."""
        self._query_results = list(result_sets)


class MockDBTransaction:
    """Mock transaction context manager."""

    def __init__(self, conn: MockDBConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> MockDBConnection:
        self._conn._in_transaction = True
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        self._conn._in_transaction = False


class MockDBPool:
    """
    Mock database connection pool (asyncpg / psycopg_pool style).

    Example::

        pool = create_mock_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM emissions")
    """

    def __init__(self) -> None:
        self._connection = MockDBConnection()
        self._closed: bool = False

    def acquire(self) -> "_MockPoolAcquire":
        """Acquire a connection from the pool."""
        return _MockPoolAcquire(self._connection)

    async def close(self) -> None:
        """Close the pool."""
        self._closed = True

    @property
    def connection(self) -> MockDBConnection:
        """Direct access to the underlying mock connection for setup."""
        return self._connection

    @property
    def is_closed(self) -> bool:
        """Whether the pool has been closed."""
        return self._closed


class _MockPoolAcquire:
    """Async context manager for pool.acquire()."""

    def __init__(self, conn: MockDBConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> MockDBConnection:
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        pass


def create_mock_db_pool() -> MockDBPool:
    """
    Factory to create a mock database connection pool.

    Returns:
        A :class:`MockDBPool` instance with a single shared
        :class:`MockDBConnection`.  Pre-load results via
        ``pool.connection.set_query_results(...)``.

    Example::

        pool = create_mock_db_pool()
        pool.connection.set_query_results(
            [{"id": 1, "total_emissions": 42.5}],
        )
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT ...")
            assert row["total_emissions"] == 42.5
    """
    return MockDBPool()


# =============================================================================
# Convenience -- create_mock_config
# =============================================================================


def create_mock_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a minimal agent configuration dict for tests.

    Args:
        overrides: Keys to merge on top of the defaults.

    Returns:
        Configuration dict suitable for agent constructors.
    """
    config: Dict[str, Any] = {
        "tenant_id": "tenant-test-001",
        "environment": "test",
        "log_level": "DEBUG",
        "db_pool_size": 2,
        "cache_ttl_seconds": 60,
        "enable_provenance": True,
        "enable_telemetry": False,
        "max_retries": 1,
        "timeout_seconds": 10,
    }
    if overrides:
        config.update(overrides)
    return config
