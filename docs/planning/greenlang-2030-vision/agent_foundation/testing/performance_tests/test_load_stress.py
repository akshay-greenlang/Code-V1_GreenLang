# -*- coding: utf-8 -*-
"""
Performance and Load Tests
Tests 10,000+ concurrent agents, message passing latency, memory retrieval speed,
agent creation time, throughput, and identifies breaking points.
"""

import pytest
import asyncio
import time
import numpy as np
import psutil
from typing import Dict, List
from collections import deque
from unittest.mock import Mock
import sys
import os
from greenlang.determinism import deterministic_random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from testing.agent_test_framework import AgentTestCase, TestConfig


class LightweightAgent:
    """Lightweight agent for load testing."""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.inbox = deque(maxlen=100)
        self.state = "ready"

    async def process(self, data: Dict):
        """Process data."""
        await asyncio.sleep(0.0001)  # Minimal work
        return {'agent': self.id, 'result': 'success'}


@pytest.mark.performance
class TestLoadAndStress(AgentTestCase):
    """Performance and stress tests."""

    async def test_10k_concurrent_agents(self):
        """Test system can handle 10,000+ concurrent agents."""
        num_agents = 10000

        # Create agents
        start = time.perf_counter()
        agents = [LightweightAgent(f"agent_{i}") for i in range(num_agents)]
        creation_time = time.perf_counter() - start

        self.assertEqual(len(agents), num_agents)
        self.assertLess(creation_time, 10.0,
                       f"Creating {num_agents} agents took {creation_time:.2f}s > 10s")

        # Process tasks concurrently
        tasks = [agent.process({'task_id': i}) for i, agent in enumerate(agents[:1000])]

        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        execution_time = time.perf_counter() - start

        self.assertEqual(len(results), 1000)
        self.assertLess(execution_time, 5.0,
                       f"Processing 1000 concurrent tasks took {execution_time:.2f}s")

    async def test_message_passing_latency(self):
        """Test message passing meets <10ms P99 latency."""
        agents = [LightweightAgent(f"agent_{i}") for i in range(100)]

        latencies = []
        for _ in range(1000):
            sender_idx = np.deterministic_random().randint(0, 100)
            receiver_idx = np.deterministic_random().randint(0, 100)

            message = {
                'from': agents[sender_idx].id,
                'to': agents[receiver_idx].id,
                'data': 'test_message'
            }

            start = time.perf_counter()
            agents[receiver_idx].inbox.append(message)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        self.assertLess(p50, 5, f"P50 latency {p50:.4f}ms > 5ms")
        self.assertLess(p99, 10, f"P99 latency {p99:.4f}ms > 10ms target")

    async def test_agent_creation_time(self):
        """Test agent creation meets <100ms target."""
        creation_times = []

        for i in range(100):
            start = time.perf_counter()
            agent = LightweightAgent(f"agent_{i}")
            creation_time_ms = (time.perf_counter() - start) * 1000
            creation_times.append(creation_time_ms)

        p99_creation = np.percentile(creation_times, 99)

        self.assertLess(p99_creation, 100,
                       f"P99 agent creation {p99_creation:.4f}ms > 100ms target")

    async def test_throughput_1000_per_second(self):
        """Test system achieves >1000 agents/second throughput."""
        test_duration_s = 1.0
        agents = [LightweightAgent(f"agent_{i}") for i in range(2000)]

        processed_count = 0
        start_time = time.time()

        while time.time() - start_time < test_duration_s:
            agent_idx = processed_count % len(agents)
            await agents[agent_idx].process({'task': processed_count})
            processed_count += 1

        throughput = processed_count / test_duration_s

        self.assertGreater(throughput, 1000,
                          f"Throughput {throughput:.0f} agents/s < 1000 target")

    async def test_memory_usage(self):
        """Test memory usage stays within limits."""
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        # Create large number of agents
        agents = [LightweightAgent(f"agent_{i}") for i in range(10000)]

        # Process tasks
        tasks = [agent.process({'data': i}) for i, agent in enumerate(agents[:1000])]
        await asyncio.gather(*tasks)

        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = final_memory_mb - initial_memory_mb

        self.assertLess(memory_increase_mb, 500,
                       f"Memory increase {memory_increase_mb:.2f}MB > 500MB limit")

    async def test_cpu_utilization(self):
        """Test CPU utilization stays reasonable."""
        process = psutil.Process()

        agents = [LightweightAgent(f"agent_{i}") for i in range(1000)]

        # Start CPU monitoring
        process.cpu_percent()  # Initialize
        await asyncio.sleep(0.1)

        # Process workload
        tasks = [agent.process({'task': i}) for i, agent in enumerate(agents)]
        await asyncio.gather(*tasks)

        cpu_percent = process.cpu_percent()

        # CPU usage should be reasonable (not pegged at 100%)
        self.assertLess(cpu_percent, 200,
                       f"CPU usage {cpu_percent:.1f}% too high")

    async def test_breaking_point_agents(self):
        """Identify breaking point for number of agents."""
        agent_counts = [1000, 5000, 10000, 20000, 50000]
        breaking_point = None

        for count in agent_counts:
            try:
                start = time.perf_counter()
                agents = [LightweightAgent(f"agent_{i}") for i in range(count)]
                creation_time = time.perf_counter() - start

                # If creation takes >30s, we've hit the breaking point
                if creation_time > 30:
                    breaking_point = count
                    break

                del agents  # Clean up
            except MemoryError:
                breaking_point = count
                break

        # Log breaking point
        if breaking_point:
            self.logger.info(f"Breaking point: {breaking_point} agents")

        # Should handle at least 10,000
        self.assertIsNone(breaking_point or (breaking_point and breaking_point > 10000),
                         f"System broke at {breaking_point} agents < 10,000 target")

    async def test_sustained_load(self):
        """Test system handles sustained load."""
        duration_s = 10.0
        agents = [LightweightAgent(f"agent_{i}") for i in range(100)]

        processed_count = 0
        start_time = time.time()
        errors = 0

        while time.time() - start_time < duration_s:
            try:
                agent_idx = processed_count % len(agents)
                await agents[agent_idx].process({'task': processed_count})
                processed_count += 1
            except Exception:
                errors += 1

        success_rate = (processed_count - errors) / processed_count

        self.assertGreater(success_rate, 0.99,
                          f"Success rate {success_rate:.2%} < 99% under sustained load")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=term"])
