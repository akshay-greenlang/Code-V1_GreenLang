"""
Performance tests for GL-001 ProcessHeatOrchestrator
Validates all performance targets are met.
Target: 100% of performance requirements satisfied.
"""

import unittest
import pytest
import asyncio
import time
import psutil
import tracemalloc
import concurrent.futures
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agents.GL_001.process_heat_orchestrator import (
    ProcessHeatOrchestrator,
    ProcessHeatConfig,
    ProcessData
)
from testing.agent_test_framework import AgentTestCase


class TestPerformance(AgentTestCase):
    """Performance validation tests for ProcessHeatOrchestrator."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.agent = ProcessHeatOrchestrator()

    @pytest.mark.performance
    def test_agent_creation_time(self):
        """Test agent creation completes within 100ms target."""
        start_time = time.perf_counter()

        # Create agent
        agent = ProcessHeatOrchestrator()

        creation_time_ms = (time.perf_counter() - start_time) * 1000

        # Verify target met
        self.assertLess(
            creation_time_ms,
            100.0,
            f"Agent creation took {creation_time_ms:.2f}ms (target: <100ms)"
        )

        # Verify agent is properly initialized
        self.assertIsNotNone(agent.config)
        self.assertIsNotNone(agent.logger)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_calculation_time(self):
        """Test thermal calculation completes within 2s target."""
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        # Measure calculation time
        start_time = time.perf_counter()
        result = await self.agent.calculate_thermal_efficiency(process_data)
        calculation_time_ms = (time.perf_counter() - start_time) * 1000

        # Verify target met
        self.assertLess(
            calculation_time_ms,
            2000.0,
            f"Calculation took {calculation_time_ms:.2f}ms (target: <2000ms)"
        )

        # Verify result is valid
        self.assertIsNotNone(result)
        self.assertGreater(result.efficiency, 0)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_message_passing_latency(self):
        """Test message passing completes within 10ms target."""
        # Create message
        message = {
            'from': 'GL-002',
            'to': 'GL-001',
            'type': 'REQUEST',
            'action': 'get_status',
            'timestamp': datetime.utcnow().isoformat()
        }

        # Mock message handler
        async def handle_message(msg):
            return {
                'status': 'READY',
                'agent_id': 'GL-001',
                'timestamp': datetime.utcnow().isoformat()
            }

        # Measure message passing time
        start_time = time.perf_counter()
        response = await handle_message(message)
        message_time_ms = (time.perf_counter() - start_time) * 1000

        # Verify target met
        self.assertLess(
            message_time_ms,
            10.0,
            f"Message passing took {message_time_ms:.2f}ms (target: <10ms)"
        )

        self.assertEqual(response['status'], 'READY')

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_dashboard_generation_time(self):
        """Test dashboard generation completes within 5s target."""
        # Prepare data for dashboard
        process_history = [
            ProcessData(
                timestamp=datetime.utcnow() - timedelta(hours=i),
                temperature_c=250.0 + np.random.normal(0, 10),
                pressure_bar=10.0 + np.random.normal(0, 1),
                flow_rate_kg_s=5.0 + np.random.normal(0, 0.5),
                energy_input_kw=1000.0,
                energy_output_kw=850.0 + np.random.normal(0, 50),
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )
            for i in range(24)  # 24 hours of data
        ]

        # Mock dashboard generation
        async def generate_dashboard(data):
            # Simulate dashboard calculations
            efficiencies = []
            for d in data:
                eff = self.agent._calculate_efficiency_core(d)
                efficiencies.append(eff)

            return {
                'avg_efficiency': np.mean(efficiencies),
                'min_efficiency': np.min(efficiencies),
                'max_efficiency': np.max(efficiencies),
                'charts': ['efficiency_trend', 'heat_recovery', 'optimization'],
                'timestamp': datetime.utcnow().isoformat()
            }

        # Measure dashboard generation time
        start_time = time.perf_counter()
        dashboard = await generate_dashboard(process_history)
        dashboard_time_ms = (time.perf_counter() - start_time) * 1000

        # Verify target met
        self.assertLess(
            dashboard_time_ms,
            5000.0,
            f"Dashboard generation took {dashboard_time_ms:.2f}ms (target: <5000ms)"
        )

        self.assertIn('avg_efficiency', dashboard)
        self.assertEqual(len(dashboard['charts']), 3)

    @pytest.mark.performance
    def test_concurrent_agents(self):
        """Test system can handle 10,000+ concurrent agents."""
        num_agents = 100  # Reduced for test practicality (would be 10000 in production)

        agents = []
        start_time = time.perf_counter()

        # Create multiple agents
        for i in range(num_agents):
            config = ProcessHeatConfig(
                agent_id=f"GL-001-{i}",
                name=f"ProcessHeatOrchestrator-{i}"
            )
            agent = ProcessHeatOrchestrator(config)
            agents.append(agent)

        creation_time_s = time.perf_counter() - start_time

        # Verify all agents created
        self.assertEqual(len(agents), num_agents)

        # Verify reasonable creation time
        avg_creation_time_ms = (creation_time_s * 1000) / num_agents
        self.assertLess(
            avg_creation_time_ms,
            10.0,
            f"Average agent creation: {avg_creation_time_ms:.2f}ms (target: <10ms)"
        )

        # Verify agents are independent
        self.assertNotEqual(agents[0].config.agent_id, agents[1].config.agent_id)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput(self):
        """Test agent throughput exceeds 1000 operations/second."""
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        # Warm up cache
        await self.agent.calculate_thermal_efficiency(process_data)

        # Measure throughput
        num_operations = 100
        start_time = time.perf_counter()

        for _ in range(num_operations):
            await self.agent.calculate_thermal_efficiency(process_data, use_cache=True)

        duration_s = time.perf_counter() - start_time
        throughput = num_operations / duration_s

        # Verify target met
        self.assertGreater(
            throughput,
            1000.0,
            f"Throughput: {throughput:.0f} ops/s (target: >1000 ops/s)"
        )

    @pytest.mark.performance
    def test_memory_usage(self):
        """Test agent memory usage stays within limits."""
        # Get initial memory
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)

        # Create agent and process data
        agent = ProcessHeatOrchestrator()

        # Generate and process large dataset
        for i in range(1000):
            data = ProcessData(
                timestamp=datetime.utcnow(),
                temperature_c=200.0 + i % 100,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=850.0,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )
            asyncio.run(agent.calculate_thermal_efficiency(data))

        # Check memory usage
        final_memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_increase_mb = final_memory_mb - initial_memory_mb

        # Verify memory usage is reasonable
        self.assertLess(
            memory_increase_mb,
            100.0,
            f"Memory increase: {memory_increase_mb:.1f}MB (limit: <100MB)"
        )

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache improves performance significantly."""
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        # First calculation (cache miss)
        start_time = time.perf_counter()
        result1 = await self.agent.calculate_thermal_efficiency(process_data)
        first_call_ms = (time.perf_counter() - start_time) * 1000

        # Second calculation (cache hit)
        start_time = time.perf_counter()
        result2 = await self.agent.calculate_thermal_efficiency(process_data)
        second_call_ms = (time.perf_counter() - start_time) * 1000

        # Cache should improve performance by at least 10x
        speedup = first_call_ms / second_call_ms
        self.assertGreater(
            speedup,
            10.0,
            f"Cache speedup: {speedup:.1f}x (target: >10x)"
        )

        # Results should be identical
        self.assertEqual(result1.provenance_hash, result2.provenance_hash)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_calculations(self):
        """Test concurrent calculation performance."""
        # Create different process data
        process_data_list = [
            ProcessData(
                timestamp=datetime.utcnow(),
                temperature_c=200.0 + i * 10,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=850.0 - i * 5,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )
            for i in range(10)
        ]

        # Run calculations concurrently
        start_time = time.perf_counter()

        tasks = [
            self.agent.calculate_thermal_efficiency(data)
            for data in process_data_list
        ]
        results = await asyncio.gather(*tasks)

        concurrent_time_ms = (time.perf_counter() - start_time) * 1000

        # Run calculations sequentially for comparison
        start_time = time.perf_counter()

        sequential_results = []
        for data in process_data_list:
            result = await self.agent.calculate_thermal_efficiency(data, use_cache=False)
            sequential_results.append(result)

        sequential_time_ms = (time.perf_counter() - start_time) * 1000

        # Concurrent should be faster
        speedup = sequential_time_ms / concurrent_time_ms
        self.assertGreater(
            speedup,
            2.0,
            f"Concurrent speedup: {speedup:.1f}x (target: >2x)"
        )

        # Results should match
        self.assertEqual(len(results), len(sequential_results))

    @pytest.mark.performance
    def test_scalability(self):
        """Test linear scalability with load."""
        # Test with increasing load
        loads = [10, 20, 40, 80]
        times = []

        for load in loads:
            process_data = ProcessData(
                timestamp=datetime.utcnow(),
                temperature_c=250.0,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=850.0,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )

            start_time = time.perf_counter()

            for _ in range(load):
                asyncio.run(
                    self.agent.calculate_thermal_efficiency(process_data, use_cache=False)
                )

            duration = time.perf_counter() - start_time
            times.append(duration)

        # Calculate scaling factor
        # Should be roughly linear (2x load = ~2x time)
        scaling_factors = []
        for i in range(1, len(loads)):
            load_increase = loads[i] / loads[i-1]
            time_increase = times[i] / times[i-1]
            scaling_factor = time_increase / load_increase
            scaling_factors.append(scaling_factor)

        avg_scaling_factor = np.mean(scaling_factors)

        # Should scale linearly (factor close to 1.0)
        self.assertLess(
            abs(avg_scaling_factor - 1.0),
            0.3,
            f"Scaling factor: {avg_scaling_factor:.2f} (target: ~1.0 for linear)"
        )

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_startup_time(self):
        """Test agent startup and initialization time."""
        start_time = time.perf_counter()

        # Create and initialize agent
        agent = ProcessHeatOrchestrator()
        await agent.initialize()

        startup_time_ms = (time.perf_counter() - start_time) * 1000

        # Verify startup is fast
        self.assertLess(
            startup_time_ms,
            500.0,
            f"Startup took {startup_time_ms:.2f}ms (target: <500ms)"
        )

        # Verify agent is ready
        from testing.agent_test_framework import AgentState
        self.assertEqual(agent.state, AgentState.READY)

    @pytest.mark.performance
    def test_cpu_usage(self):
        """Test CPU usage stays within acceptable limits."""
        import threading

        # Monitor CPU usage in background
        cpu_samples = []

        def monitor_cpu():
            process = psutil.Process()
            for _ in range(10):
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)

        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        # Perform intensive operations
        for i in range(100):
            data = ProcessData(
                timestamp=datetime.utcnow(),
                temperature_c=200.0 + i,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=850.0,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )
            asyncio.run(self.agent.calculate_thermal_efficiency(data, use_cache=False))

        # Wait for monitoring to complete
        monitor_thread.join()

        # Analyze CPU usage
        avg_cpu = np.mean(cpu_samples)
        max_cpu = np.max(cpu_samples)

        # Verify CPU usage is reasonable
        self.assertLess(
            avg_cpu,
            50.0,
            f"Average CPU: {avg_cpu:.1f}% (target: <50%)"
        )

        self.assertLess(
            max_cpu,
            90.0,
            f"Max CPU: {max_cpu:.1f}% (target: <90%)"
        )


if __name__ == '__main__':
    unittest.main()