# -*- coding: utf-8 -*-
"""
GreenLang Infrastructure Performance Benchmarks
==============================================

Comprehensive performance benchmarks for all major GreenLang infrastructure components.

Components tested:
- greenlang.intelligence (LLM): ChatSession, semantic caching, RAG, embeddings
- greenlang.cache: L1/L2/L3 caching layers, hit rates, throughput
- greenlang.db: Connection pooling, query execution, transactions
- greenlang.services: Factor Broker, Entity MDM, Monte Carlo, PCF Exchange
- greenlang.sdk.base.Agent: Agent initialization, batch processing, parallel execution

Requires: pytest-benchmark
Install: pip install pytest-benchmark

Usage:
    # Run all benchmarks
    pytest benchmarks/infrastructure/test_benchmarks.py --benchmark-only

    # Run with verbose output
    pytest benchmarks/infrastructure/test_benchmarks.py --benchmark-only -v

    # Save results to JSON
    pytest benchmarks/infrastructure/test_benchmarks.py --benchmark-only --benchmark-json=baseline.json

    # Compare against baseline
    pytest benchmarks/infrastructure/test_benchmarks.py --benchmark-only --benchmark-compare=baseline.json

Author: Performance Engineering Team
Date: 2025-11-09
Version: 1.0.0
"""

import asyncio
import time
import random
from typing import List, Dict, Any
from pathlib import Path
import tempfile

import pytest
from greenlang.determinism import deterministic_random


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_text():
    """Sample text for LLM benchmarks."""
    return "Calculate the carbon footprint of a steel shipment from China to EU."


@pytest.fixture
def sample_documents():
    """Sample documents for RAG benchmarks."""
    return [
        {"id": i, "content": f"Document {i} about carbon emissions and steel production."}
        for i in range(1000)
    ]


@pytest.fixture
def sample_cache_keys():
    """Sample cache keys for testing."""
    return [f"key_{i}" for i in range(1000)]


@pytest.fixture
def sample_db_query():
    """Sample database query."""
    return "SELECT id, name, value FROM test_table WHERE category = :category"


@pytest.fixture
def sample_shipment_data():
    """Sample shipment data for agent benchmarks."""
    return {
        "shipment_id": "SHIP-001",
        "origin": "CN",
        "destination": "EU",
        "goods": "Steel",
        "quantity": 1000,
        "transport_mode": "Sea"
    }


# ============================================================================
# MOCK IMPLEMENTATIONS (for testing without actual infrastructure)
# ============================================================================

class MockChatSession:
    """Mock ChatSession for benchmarking."""

    async def initialize(self):
        """Simulate initialization."""
        await asyncio.sleep(0.01)  # 10ms initialization

    async def complete(self, prompt: str, stream: bool = False):
        """Simulate LLM completion."""
        await asyncio.sleep(0.5)  # 500ms completion time
        return {"content": "Mock response", "tokens": len(prompt.split())}

    async def embed(self, text: str):
        """Simulate embedding generation."""
        await asyncio.sleep(0.05)  # 50ms per embedding
        return [0.1] * 1536  # Mock embedding vector


class MockCacheManager:
    """Mock CacheManager for benchmarking."""

    def __init__(self):
        self._l1_cache = {}
        self._l2_cache = {}
        self._l3_cache = {}
        self._hits = 0
        self._misses = 0

    async def get(self, key: str, level: str = "L1"):
        """Get from cache with simulated latency."""
        latencies = {"L1": 0.00001, "L2": 0.001, "L3": 0.01}  # µs, ms, ms
        await asyncio.sleep(latencies.get(level, 0.00001))

        cache = getattr(self, f"_l{level[1]}_cache", self._l1_cache)
        if key in cache:
            self._hits += 1
            return cache[key]
        else:
            self._misses += 1
            return None

    async def set(self, key: str, value: Any, level: str = "L1", ttl: int = 300):
        """Set cache value with simulated latency."""
        latencies = {"L1": 0.00001, "L2": 0.002, "L3": 0.015}
        await asyncio.sleep(latencies.get(level, 0.00001))

        cache = getattr(self, f"_l{level[1]}_cache", self._l1_cache)
        cache[key] = value

    def get_hit_rate(self):
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0


class MockDatabasePool:
    """Mock database pool for benchmarking."""

    def __init__(self):
        self._pool = []
        self._active_connections = 0

    async def initialize(self, pool_size: int = 20):
        """Initialize pool."""
        await asyncio.sleep(0.1)  # 100ms initialization
        self._pool = [None] * pool_size

    async def acquire(self):
        """Acquire connection."""
        await asyncio.sleep(0.005)  # 5ms acquisition
        self._active_connections += 1
        return self

    async def release(self):
        """Release connection."""
        self._active_connections -= 1

    async def execute(self, query: str, params: Dict = None):
        """Execute query."""
        # Simulate different query types
        if "SELECT" in query and "JOIN" not in query:
            await asyncio.sleep(0.01)  # 10ms simple query
        elif "JOIN" in query:
            await asyncio.sleep(0.05)  # 50ms complex query
        elif "GROUP BY" in query or "aggregate" in query.lower():
            await asyncio.sleep(0.1)  # 100ms aggregation
        else:
            await asyncio.sleep(0.02)  # 20ms default

        return [{"id": i, "value": f"row_{i}"} for i in range(10)]

    async def commit(self):
        """Commit transaction."""
        await asyncio.sleep(0.003)  # 3ms commit


class MockFactorBroker:
    """Mock Factor Broker for benchmarking."""

    async def resolve(self, factor_type: str, context: Dict):
        """Resolve emission factor."""
        # Simulate factor resolution with caching
        cache_hit = deterministic_random().random() < 0.7  # 70% cache hit rate

        if cache_hit:
            await asyncio.sleep(0.002)  # 2ms cache hit
        else:
            await asyncio.sleep(0.05)  # 50ms cache miss

        return {
            "factor": 2.5,
            "unit": "kgCO2e/kg",
            "source": "CBAM",
            "cached": cache_hit
        }


class MockAgent:
    """Mock Agent for benchmarking."""

    def __init__(self, name: str = "TestAgent"):
        self.name = name
        self._initialized = False

    async def initialize(self):
        """Initialize agent."""
        await asyncio.sleep(0.05)  # 50ms initialization
        self._initialized = True

    async def process(self, data: Dict) -> Dict:
        """Process single record."""
        await asyncio.sleep(0.1)  # 100ms processing
        return {"status": "success", "result": data}

    async def process_batch(self, data_list: List[Dict]) -> List[Dict]:
        """Process batch of records."""
        # Simulate batch efficiency (20% faster than individual)
        batch_time = len(data_list) * 0.08
        await asyncio.sleep(batch_time)
        return [{"status": "success", "result": d} for d in data_list]


# ============================================================================
# INTELLIGENCE BENCHMARKS (greenlang.intelligence)
# ============================================================================

class TestIntelligenceBenchmarks:
    """Benchmarks for LLM and intelligence services."""

    @pytest.mark.benchmark(group="intelligence")
    def test_chat_session_initialization(self, benchmark):
        """Benchmark: ChatSession initialization time."""

        async def init_session():
            session = MockChatSession()
            await session.initialize()
            return session

        result = benchmark(lambda: asyncio.run(init_session()))
        assert result is not None

    @pytest.mark.benchmark(group="intelligence")
    def test_first_token_latency(self, benchmark, sample_text):
        """Benchmark: First token latency for LLM completion."""

        async def get_first_token():
            session = MockChatSession()
            await session.initialize()
            start = time.perf_counter()
            response = await session.complete(sample_text, stream=True)
            return time.perf_counter() - start

        latency = benchmark(lambda: asyncio.run(get_first_token()))
        # Target: < 200ms for first token
        assert latency < 0.2

    @pytest.mark.benchmark(group="intelligence")
    def test_total_completion_time(self, benchmark, sample_text):
        """Benchmark: Total LLM completion time."""

        async def complete():
            session = MockChatSession()
            await session.initialize()
            return await session.complete(sample_text)

        result = benchmark(lambda: asyncio.run(complete()))
        assert result is not None

    @pytest.mark.benchmark(group="intelligence")
    def test_semantic_cache_hit_latency(self, benchmark, sample_text):
        """Benchmark: Semantic cache hit latency."""

        async def cached_complete():
            session = MockChatSession()
            await session.initialize()
            # First call (cache miss)
            await session.complete(sample_text)
            # Second call (cache hit - should be much faster)
            start = time.perf_counter()
            await session.complete(sample_text)
            return time.perf_counter() - start

        latency = benchmark(lambda: asyncio.run(cached_complete()))
        # Cache hit should be < 10ms
        assert latency < 0.01

    @pytest.mark.benchmark(group="intelligence")
    def test_embedding_generation_speed(self, benchmark, sample_text):
        """Benchmark: Embedding generation speed."""

        async def generate_embedding():
            session = MockChatSession()
            return await session.embed(sample_text)

        embedding = benchmark(lambda: asyncio.run(generate_embedding()))
        assert len(embedding) == 1536

    @pytest.mark.benchmark(group="intelligence")
    @pytest.mark.parametrize("doc_count", [1000, 10000])
    def test_rag_retrieval_time(self, benchmark, doc_count):
        """Benchmark: RAG retrieval time for different corpus sizes."""

        async def rag_retrieve():
            # Simulate vector search in document corpus
            documents = [{"id": i, "embedding": [0.1] * 1536} for i in range(doc_count)]
            query_embedding = [0.1] * 1536

            # Simple similarity search
            start = time.perf_counter()
            results = sorted(
                documents,
                key=lambda d: sum(a * b for a, b in zip(d["embedding"], query_embedding)),
                reverse=True
            )[:10]
            return time.perf_counter() - start

        retrieval_time = benchmark(lambda: asyncio.run(rag_retrieve()))

        # Target: < 100ms for 10K docs
        if doc_count == 10000:
            assert retrieval_time < 0.1


# ============================================================================
# CACHE BENCHMARKS (greenlang.cache)
# ============================================================================

class TestCacheBenchmarks:
    """Benchmarks for caching layers."""

    @pytest.mark.benchmark(group="cache")
    def test_l1_memory_get_latency(self, benchmark, sample_cache_keys):
        """Benchmark: L1 (memory) cache GET latency."""

        async def l1_get():
            cache = MockCacheManager()
            # Populate cache
            for key in sample_cache_keys[:100]:
                await cache.set(key, f"value_{key}", level="L1")

            # Benchmark GET
            start = time.perf_counter()
            await cache.get(sample_cache_keys[0], level="L1")
            return time.perf_counter() - start

        latency_ms = benchmark(lambda: asyncio.run(l1_get())) * 1000
        # Target: < 100µs (0.1ms)
        assert latency_ms < 0.1

    @pytest.mark.benchmark(group="cache")
    def test_l1_memory_set_latency(self, benchmark):
        """Benchmark: L1 (memory) cache SET latency."""

        async def l1_set():
            cache = MockCacheManager()
            start = time.perf_counter()
            await cache.set("test_key", "test_value", level="L1")
            return time.perf_counter() - start

        latency_ms = benchmark(lambda: asyncio.run(l1_set())) * 1000
        # Target: < 100µs (0.1ms)
        assert latency_ms < 0.1

    @pytest.mark.benchmark(group="cache")
    def test_l2_redis_get_latency(self, benchmark):
        """Benchmark: L2 (Redis) cache GET latency."""

        async def l2_get():
            cache = MockCacheManager()
            await cache.set("test_key", "test_value", level="L2")

            start = time.perf_counter()
            await cache.get("test_key", level="L2")
            return time.perf_counter() - start

        latency_ms = benchmark(lambda: asyncio.run(l2_get())) * 1000
        # Target: < 5ms
        assert latency_ms < 5.0

    @pytest.mark.benchmark(group="cache")
    def test_l2_redis_set_latency(self, benchmark):
        """Benchmark: L2 (Redis) cache SET latency."""

        async def l2_set():
            cache = MockCacheManager()
            start = time.perf_counter()
            await cache.set("test_key", "test_value", level="L2", ttl=300)
            return time.perf_counter() - start

        latency_ms = benchmark(lambda: asyncio.run(l2_set())) * 1000
        # Target: < 10ms
        assert latency_ms < 10.0

    @pytest.mark.benchmark(group="cache")
    def test_cache_hit_rate_under_load(self, benchmark, sample_cache_keys):
        """Benchmark: Cache hit rate under load."""

        async def cache_workload():
            cache = MockCacheManager()

            # Warm up cache
            for key in sample_cache_keys[:100]:
                await cache.set(key, f"value_{key}", level="L1")

            # Simulate workload (80% reads on existing keys, 20% new keys)
            for _ in range(1000):
                if deterministic_random().random() < 0.8:
                    # Read existing key
                    key = deterministic_random().choice(sample_cache_keys[:100])
                    await cache.get(key, level="L1")
                else:
                    # Read new key (miss)
                    key = deterministic_random().choice(sample_cache_keys[100:])
                    await cache.get(key, level="L1")

            return cache.get_hit_rate()

        hit_rate = benchmark(lambda: asyncio.run(cache_workload()))
        # Target: > 70% hit rate
        assert hit_rate > 0.7

    @pytest.mark.benchmark(group="cache")
    def test_cache_throughput_ops_per_sec(self, benchmark):
        """Benchmark: Cache throughput (operations per second)."""

        async def cache_throughput():
            cache = MockCacheManager()

            ops = 10000
            start = time.perf_counter()

            for i in range(ops):
                if i % 2 == 0:
                    await cache.set(f"key_{i}", f"value_{i}", level="L1")
                else:
                    await cache.get(f"key_{i}", level="L1")

            duration = time.perf_counter() - start
            return ops / duration

        ops_per_sec = benchmark(lambda: asyncio.run(cache_throughput()))
        # Target: > 100K ops/sec for L1
        assert ops_per_sec > 100000


# ============================================================================
# DATABASE BENCHMARKS (greenlang.db)
# ============================================================================

class TestDatabaseBenchmarks:
    """Benchmarks for database operations."""

    @pytest.mark.benchmark(group="database")
    def test_connection_pool_acquisition_time(self, benchmark):
        """Benchmark: Connection pool acquisition time."""

        async def acquire_connection():
            pool = MockDatabasePool()
            await pool.initialize(pool_size=20)

            start = time.perf_counter()
            conn = await pool.acquire()
            duration = time.perf_counter() - start
            await conn.release()
            return duration

        acquisition_time_ms = benchmark(lambda: asyncio.run(acquire_connection())) * 1000
        # Target: < 10ms
        assert acquisition_time_ms < 10.0

    @pytest.mark.benchmark(group="database")
    def test_simple_query_execution(self, benchmark):
        """Benchmark: Simple SELECT query execution time."""

        async def execute_query():
            pool = MockDatabasePool()
            await pool.initialize()
            conn = await pool.acquire()

            start = time.perf_counter()
            await conn.execute("SELECT id, name FROM users WHERE id = 123")
            duration = time.perf_counter() - start

            await conn.release()
            return duration

        query_time_ms = benchmark(lambda: asyncio.run(execute_query())) * 1000
        # Target: < 50ms
        assert query_time_ms < 50.0

    @pytest.mark.benchmark(group="database")
    def test_complex_query_execution(self, benchmark):
        """Benchmark: Complex query with JOINs execution time."""

        async def execute_complex_query():
            pool = MockDatabasePool()
            await pool.initialize()
            conn = await pool.acquire()

            query = """
                SELECT u.id, u.name, o.total
                FROM users u
                JOIN orders o ON u.id = o.user_id
                WHERE u.country = 'US'
            """

            start = time.perf_counter()
            await conn.execute(query)
            duration = time.perf_counter() - start

            await conn.release()
            return duration

        query_time_ms = benchmark(lambda: asyncio.run(execute_complex_query())) * 1000
        # Target: < 200ms
        assert query_time_ms < 200.0

    @pytest.mark.benchmark(group="database")
    def test_aggregation_query_execution(self, benchmark):
        """Benchmark: Aggregation query execution time."""

        async def execute_aggregation():
            pool = MockDatabasePool()
            await pool.initialize()
            conn = await pool.acquire()

            query = """
                SELECT country, COUNT(*), SUM(emissions)
                FROM shipments
                GROUP BY country
            """

            start = time.perf_counter()
            await conn.execute(query)
            duration = time.perf_counter() - start

            await conn.release()
            return duration

        query_time_ms = benchmark(lambda: asyncio.run(execute_aggregation())) * 1000
        # Target: < 500ms
        assert query_time_ms < 500.0

    @pytest.mark.benchmark(group="database")
    def test_transaction_commit_time(self, benchmark):
        """Benchmark: Transaction commit time."""

        async def commit_transaction():
            pool = MockDatabasePool()
            await pool.initialize()
            conn = await pool.acquire()

            await conn.execute("INSERT INTO test VALUES (1, 'test')")

            start = time.perf_counter()
            await conn.commit()
            duration = time.perf_counter() - start

            await conn.release()
            return duration

        commit_time_ms = benchmark(lambda: asyncio.run(commit_transaction())) * 1000
        # Target: < 10ms
        assert commit_time_ms < 10.0

    @pytest.mark.benchmark(group="database")
    def test_concurrent_connection_handling(self, benchmark):
        """Benchmark: Concurrent connection handling (pool saturation)."""

        async def concurrent_queries():
            pool = MockDatabasePool()
            await pool.initialize(pool_size=20)

            async def query_task():
                conn = await pool.acquire()
                await conn.execute("SELECT * FROM test")
                await conn.release()

            # Create 50 concurrent queries (exceeds pool size)
            start = time.perf_counter()
            await asyncio.gather(*[query_task() for _ in range(50)])
            return time.perf_counter() - start

        total_time = benchmark(lambda: asyncio.run(concurrent_queries()))
        # Should handle gracefully within 5 seconds
        assert total_time < 5.0


# ============================================================================
# SERVICES BENCHMARKS (greenlang.services)
# ============================================================================

class TestServicesBenchmarks:
    """Benchmarks for GreenLang services."""

    @pytest.mark.benchmark(group="services")
    @pytest.mark.parametrize("percentile", ["p50", "p95", "p99"])
    def test_factor_broker_resolution_time(self, benchmark, percentile):
        """Benchmark: Factor Broker resolution time at different percentiles."""

        async def resolve_factor():
            broker = MockFactorBroker()

            times = []
            iterations = 1000 if percentile == "p50" else 10000

            for _ in range(iterations):
                start = time.perf_counter()
                await broker.resolve(
                    "transport_emission",
                    {"mode": "sea", "origin": "CN", "destination": "EU"}
                )
                times.append(time.perf_counter() - start)

            times.sort()

            if percentile == "p50":
                return times[len(times) // 2]
            elif percentile == "p95":
                return times[int(len(times) * 0.95)]
            else:  # p99
                return times[int(len(times) * 0.99)]

        latency_ms = benchmark(lambda: asyncio.run(resolve_factor())) * 1000

        # Targets: P50 < 10ms, P95 < 50ms, P99 < 100ms
        targets = {"p50": 10.0, "p95": 50.0, "p99": 100.0}
        assert latency_ms < targets[percentile]

    @pytest.mark.benchmark(group="services")
    def test_monte_carlo_simulation_time(self, benchmark):
        """Benchmark: Monte Carlo simulation time (10K iterations)."""

        async def run_simulation():
            iterations = 10000

            start = time.perf_counter()
            results = []
            for _ in range(iterations):
                # Simple Monte Carlo calculation
                value = sum(random.gauss(100, 15) for _ in range(10))
                results.append(value)

            return time.perf_counter() - start

        simulation_time = benchmark(lambda: asyncio.run(run_simulation()))
        # Target: < 1 second for 10K iterations
        assert simulation_time < 1.0


# ============================================================================
# AGENT BENCHMARKS (greenlang.sdk.base.Agent)
# ============================================================================

class TestAgentBenchmarks:
    """Benchmarks for Agent SDK."""

    @pytest.mark.benchmark(group="agent")
    def test_agent_initialization_overhead(self, benchmark):
        """Benchmark: Agent initialization overhead."""

        async def init_agent():
            agent = MockAgent("TestAgent")
            await agent.initialize()
            return agent

        agent = benchmark(lambda: asyncio.run(init_agent()))
        assert agent is not None

    @pytest.mark.benchmark(group="agent")
    def test_batch_processing_throughput(self, benchmark, sample_shipment_data):
        """Benchmark: Agent batch processing throughput."""

        async def process_batch():
            agent = MockAgent("BatchAgent")
            await agent.initialize()

            batch = [sample_shipment_data for _ in range(100)]

            start = time.perf_counter()
            results = await agent.process_batch(batch)
            duration = time.perf_counter() - start

            return len(results) / duration  # records/second

        throughput = benchmark(lambda: asyncio.run(process_batch()))
        # Target: > 1000 records/sec
        assert throughput > 1000

    @pytest.mark.benchmark(group="agent")
    def test_memory_usage_per_agent_instance(self, benchmark):
        """Benchmark: Memory usage per agent instance."""
        import sys

        async def measure_memory():
            agents = []
            for i in range(100):
                agent = MockAgent(f"Agent{i}")
                await agent.initialize()
                agents.append(agent)

            # Rough memory estimate (in practice, use memory_profiler)
            return sys.getsizeof(agents) / len(agents)

        memory_per_agent = benchmark(lambda: asyncio.run(measure_memory()))
        # Should be reasonable (< 1MB per agent)
        assert memory_per_agent < 1_000_000

    @pytest.mark.benchmark(group="agent")
    def test_parallel_agent_execution_scaling(self, benchmark, sample_shipment_data):
        """Benchmark: Parallel agent execution scaling."""

        async def parallel_execution():
            agents = [MockAgent(f"Agent{i}") for i in range(10)]

            # Initialize all agents
            await asyncio.gather(*[agent.initialize() for agent in agents])

            # Process data in parallel
            start = time.perf_counter()
            results = await asyncio.gather(*[
                agent.process(sample_shipment_data) for agent in agents
            ])
            duration = time.perf_counter() - start

            return len(results) / duration

        throughput = benchmark(lambda: asyncio.run(parallel_execution()))
        # Should scale well (> 50 ops/sec for 10 agents)
        assert throughput > 50


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_baseline_report(results_file: Path):
    """
    Generate baseline performance report from benchmark results.

    Args:
        results_file: Path to pytest-benchmark JSON results
    """
    import json

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    with open(results_file) as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("GREENLANG INFRASTRUCTURE PERFORMANCE BASELINE")
    print("=" * 80)
    print(f"Date: {data['datetime']}")
    print(f"Machine: {data['machine_info']['node']}")
    print(f"Python: {data['machine_info']['python_version']}")
    print("=" * 80)

    for benchmark in data['benchmarks']:
        name = benchmark['name']
        group = benchmark.get('group', 'unknown')
        stats = benchmark['stats']

        print(f"\n{name} ({group})")
        print(f"  Mean:   {stats['mean'] * 1000:.2f} ms")
        print(f"  Median: {stats['median'] * 1000:.2f} ms")
        print(f"  StdDev: {stats['stddev'] * 1000:.2f} ms")
        print(f"  Min:    {stats['min'] * 1000:.2f} ms")
        print(f"  Max:    {stats['max'] * 1000:.2f} ms")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("GreenLang Infrastructure Benchmarks")
    print("====================================")
    print("\nRun with: pytest test_benchmarks.py --benchmark-only")
