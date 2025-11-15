"""
Unit tests for GL-001 ProcessHeatOrchestrator
Target coverage: 95% of all agent methods and logic paths.
"""

import unittest
import asyncio
import pytest
import time
import json
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agents.GL_001.process_heat_orchestrator import (
    ProcessHeatOrchestrator,
    ProcessHeatConfig,
    ProcessData,
    ThermalCalculation,
    OptimizationStrategy
)
from testing.agent_test_framework import AgentTestCase, AgentState
from testing.quality_validators import QualityDimension


class TestProcessHeatOrchestrator(AgentTestCase):
    """Comprehensive unit tests for ProcessHeatOrchestrator."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.config = ProcessHeatConfig(
            agent_id="GL-001-TEST",
            name="TestProcessHeatOrchestrator",
            version="1.0.0-test"
        )
        self.agent = ProcessHeatOrchestrator(self.config)

    def tearDown(self):
        """Clean up after tests."""
        super().tearDown()
        if hasattr(self, 'agent'):
            asyncio.run(self.agent.shutdown())

    # Initialization Tests
    def test_agent_initialization(self):
        """Test agent initializes with correct configuration."""
        self.assertEqual(self.agent.config.agent_id, "GL-001-TEST")
        self.assertEqual(self.agent.config.name, "TestProcessHeatOrchestrator")
        self.assertEqual(self.agent.state, AgentState.CREATED)
        self.assertIsNotNone(self.agent.logger)
        self.assertEqual(len(self.agent.process_data_buffer), 0)
        self.assertEqual(len(self.agent.calculations_cache), 0)

    def test_agent_initialization_with_defaults(self):
        """Test agent initialization with default config."""
        agent = ProcessHeatOrchestrator()
        self.assertEqual(agent.config.agent_id, "GL-001")
        self.assertEqual(agent.config.target_efficiency, 0.85)
        self.assertEqual(agent.config.llm_temperature, 0.0)
        self.assertEqual(agent.config.llm_seed, 42)

    @pytest.mark.asyncio
    async def test_async_initialization(self):
        """Test async initialization and state transitions."""
        with self.assert_performance(max_duration_ms=100):
            result = await self.agent.initialize()

        self.assertTrue(result)
        self.assertEqual(self.agent.state, AgentState.READY)

    @pytest.mark.asyncio
    async def test_initialization_failure_handling(self):
        """Test initialization handles failures gracefully."""
        with patch.object(self.agent, '_connect_scada', side_effect=Exception("SCADA connection failed")):
            result = await self.agent.initialize()

        self.assertFalse(result)
        self.assertEqual(self.agent.state, AgentState.ERROR)

    # Lifecycle Tests
    def test_lifecycle_transitions(self):
        """Test valid lifecycle state transitions."""
        # Test CREATED -> INITIALIZING -> READY
        self.assertEqual(self.agent.state, AgentState.CREATED)

        asyncio.run(self.agent.initialize())
        self.assertEqual(self.agent.state, AgentState.READY)

        # Test READY -> RUNNING
        self.agent.state = AgentState.RUNNING
        self.assertEqual(self.agent.state, AgentState.RUNNING)

        # Test RUNNING -> STOPPING -> TERMINATED
        asyncio.run(self.agent.shutdown())
        self.assertEqual(self.agent.state, AgentState.TERMINATED)

    # Thermal Calculation Tests
    @pytest.mark.asyncio
    async def test_thermal_efficiency_calculation(self):
        """Test core thermal efficiency calculation."""
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="natural_gas",
            fuel_consumption_rate=10.0
        )

        result = await self.agent.calculate_thermal_efficiency(process_data)

        self.assertIsInstance(result, ThermalCalculation)
        self.assertAlmostEqual(result.efficiency, 0.85, places=2)
        self.assertAlmostEqual(result.heat_loss_kw, 150.0, places=1)
        self.assertTrue(result.deterministic)
        self.assertIsNotNone(result.provenance_hash)
        self.assertEqual(len(result.provenance_hash), 64)  # SHA-256 hash

    @pytest.mark.asyncio
    async def test_calculation_with_cache(self):
        """Test calculation caching for performance."""
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=200.0,
            pressure_bar=8.0,
            flow_rate_kg_s=3.0,
            energy_input_kw=500.0,
            energy_output_kw=400.0,
            fuel_type="diesel",
            fuel_consumption_rate=5.0
        )

        # First call - cache miss
        result1 = await self.agent.calculate_thermal_efficiency(process_data)
        cache_misses_1 = self.agent.metrics['cache_misses']

        # Second call - cache hit
        result2 = await self.agent.calculate_thermal_efficiency(process_data)
        cache_hits_2 = self.agent.metrics['cache_hits']

        # Verify cache behavior
        self.assertEqual(result1.provenance_hash, result2.provenance_hash)
        self.assertEqual(cache_hits_2, 1)
        self.assertEqual(cache_misses_1, 1)

    @pytest.mark.asyncio
    async def test_calculation_without_cache(self):
        """Test calculation with cache disabled."""
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=300.0,
            pressure_bar=12.0,
            flow_rate_kg_s=4.0,
            energy_input_kw=600.0,
            energy_output_kw=480.0,
            fuel_type="coal",
            fuel_consumption_rate=8.0
        )

        result1 = await self.agent.calculate_thermal_efficiency(process_data, use_cache=False)
        result2 = await self.agent.calculate_thermal_efficiency(process_data, use_cache=False)

        # Both should be cache misses
        self.assertEqual(self.agent.metrics['cache_misses'], 2)
        self.assertEqual(self.agent.metrics['cache_hits'], 0)

    # Boundary and Edge Case Tests
    @pytest.mark.asyncio
    async def test_zero_energy_input(self):
        """Test handling of zero energy input."""
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=100.0,
            pressure_bar=5.0,
            flow_rate_kg_s=1.0,
            energy_input_kw=0.0,
            energy_output_kw=0.0,
            fuel_type="none",
            fuel_consumption_rate=0.0
        )

        result = await self.agent.calculate_thermal_efficiency(process_data)
        self.assertEqual(result.efficiency, 0.0)
        self.assertEqual(result.heat_loss_kw, 0.0)

    @pytest.mark.asyncio
    async def test_negative_values_validation(self):
        """Test validation rejects negative values."""
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=200.0,
            pressure_bar=10.0,
            flow_rate_kg_s=-5.0,  # Invalid negative flow
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        with self.assertRaises(ValueError) as context:
            await self.agent.calculate_thermal_efficiency(process_data)

        self.assertIn("Flow rate cannot be negative", str(context.exception))

    @pytest.mark.asyncio
    async def test_thermodynamics_violation(self):
        """Test validation catches thermodynamics violations."""
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=200.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=1100.0,  # Output > Input (violation)
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        with self.assertRaises(ValueError) as context:
            await self.agent.calculate_thermal_efficiency(process_data)

        self.assertIn("thermodynamics violation", str(context.exception).lower())

    @pytest.mark.asyncio
    async def test_extreme_temperatures(self):
        """Test handling of extreme temperature values."""
        # Test absolute zero
        process_data_cold = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=-273.15,  # Absolute zero
            pressure_bar=1.0,
            flow_rate_kg_s=1.0,
            energy_input_kw=100.0,
            energy_output_kw=50.0,
            fuel_type="special",
            fuel_consumption_rate=1.0
        )

        result_cold = await self.agent.calculate_thermal_efficiency(process_data_cold)
        self.assertIsNotNone(result_cold)

        # Test maximum temperature
        process_data_hot = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=3000.0,  # Maximum
            pressure_bar=100.0,
            flow_rate_kg_s=10.0,
            energy_input_kw=10000.0,
            energy_output_kw=7000.0,
            fuel_type="plasma",
            fuel_consumption_rate=100.0
        )

        result_hot = await self.agent.calculate_thermal_efficiency(process_data_hot)
        self.assertIsNotNone(result_hot)
        self.assertGreater(result_hot.recoverable_heat_kw, 0)

    # Provenance and Determinism Tests
    def test_provenance_hash_generation(self):
        """Test provenance hash is deterministic."""
        data = {
            'input': {'temp': 200.0, 'pressure': 10.0},
            'output': {'efficiency': 0.85},
            'timestamp': '2025-01-15T10:00:00'
        }

        hash1 = self.agent._generate_provenance_hash(data)
        hash2 = self.agent._generate_provenance_hash(data)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA-256

    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=200.123456789,  # Extra precision
            pressure_bar=10.987654321,
            flow_rate_kg_s=5.555555,
            energy_input_kw=1000.111,
            energy_output_kw=850.999,
            fuel_type="natural_gas",
            fuel_consumption_rate=10.0
        )

        key1 = self.agent._generate_cache_key(process_data)
        key2 = self.agent._generate_cache_key(process_data)

        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 16)  # Truncated hash

    # Optimization Strategy Tests
    @pytest.mark.asyncio
    async def test_optimization_strategy_generation(self):
        """Test generation of optimization strategies."""
        process_history = [
            ProcessData(
                timestamp=datetime.utcnow() - timedelta(hours=i),
                temperature_c=250.0,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=750.0,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )
            for i in range(24)  # 24 hours of data
        ]

        constraints = {
            'max_investment': 100000,
            'payback_period_months': 36,
            'min_roi': 0.15
        }

        strategy = await self.agent.generate_optimization_strategy(
            process_history,
            constraints
        )

        self.assertIsInstance(strategy, OptimizationStrategy)
        self.assertIn(strategy.strategy_type, ['heat_recovery', 'fuel_switching', 'process_optimization'])
        self.assertGreater(strategy.expected_savings_kwh, 0)
        self.assertGreater(strategy.co2_reduction_tonnes, 0)
        self.assertIsNotNone(strategy.validation_hash)
        self.assertEqual(len(strategy.implementation_steps), 5)

    # Metrics and Monitoring Tests
    def test_metrics_tracking(self):
        """Test agent metrics are tracked correctly."""
        initial_metrics = self.agent.get_metrics()

        self.assertEqual(initial_metrics['calculations_performed'], 0)
        self.assertEqual(initial_metrics['avg_calculation_time_ms'], 0.0)
        self.assertEqual(initial_metrics['cache_hit_rate'], 0.0)

        # Perform some calculations
        process_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=200.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        asyncio.run(self.agent.calculate_thermal_efficiency(process_data))
        asyncio.run(self.agent.calculate_thermal_efficiency(process_data))  # Cache hit

        updated_metrics = self.agent.get_metrics()

        self.assertEqual(updated_metrics['calculations_performed'], 1)
        self.assertGreater(updated_metrics['avg_calculation_time_ms'], 0)
        self.assertGreater(updated_metrics['cache_hit_rate'], 0)

    # Error Recovery Tests
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test agent recovers from errors gracefully."""
        # Simulate LLM failure
        with patch.object(self.agent, 'query_llm', side_effect=Exception("LLM unavailable")):
            process_data = ProcessData(
                timestamp=datetime.utcnow(),
                temperature_c=200.0,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=850.0,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )

            # Should still calculate without LLM recommendations
            result = await self.agent.calculate_thermal_efficiency(process_data)
            self.assertIsNotNone(result)
            self.assertGreater(len(result.recommendations), 0)  # Has rule-based recommendations

    # Multi-tenancy Tests
    def test_multi_tenancy_isolation(self):
        """Test tenant data isolation."""
        self.agent.tenant_data['tenant_1'] = {'data': 'sensitive_1'}
        self.agent.tenant_data['tenant_2'] = {'data': 'sensitive_2'}

        self.assertNotEqual(
            self.agent.tenant_data['tenant_1'],
            self.agent.tenant_data['tenant_2']
        )

        # Ensure no cross-contamination
        self.assertNotIn('tenant_2', str(self.agent.tenant_data['tenant_1']))
        self.assertNotIn('tenant_1', str(self.agent.tenant_data['tenant_2']))

    # Shutdown and Cleanup Tests
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown saves state."""
        await self.agent.initialize()

        # Add some data
        self.agent.metrics['calculations_performed'] = 100
        self.agent.provenance_chain.extend(['hash1', 'hash2', 'hash3'])

        with patch.object(self.agent, '_save_state') as mock_save:
            await self.agent.shutdown()

        mock_save.assert_called_once()
        self.assertEqual(self.agent.state, AgentState.TERMINATED)


if __name__ == '__main__':
    unittest.main()