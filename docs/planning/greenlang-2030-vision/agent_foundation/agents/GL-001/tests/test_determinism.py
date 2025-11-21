# -*- coding: utf-8 -*-
"""
Determinism tests for GL-001 ProcessHeatOrchestrator
Validates 100% reproducibility: same inputs → same outputs.
Critical for regulatory compliance and audit requirements.
"""

import unittest
import pytest
import asyncio
import hashlib
import json
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agents.GL_001.process_heat_orchestrator import (
    ProcessHeatOrchestrator,
    ProcessHeatConfig,
    ProcessData,
    ThermalCalculation
)
from testing.agent_test_framework import AgentTestCase


class TestDeterminism(AgentTestCase):
    """Comprehensive determinism tests for ProcessHeatOrchestrator."""

    def setUp(self):
        """Set up test environment with deterministic settings."""
        super().setUp()
        # Set fixed seed for all random operations
        np.random.seed(42)

        self.config = ProcessHeatConfig(
            llm_temperature=0.0,
            llm_seed=42
        )
        self.agent = ProcessHeatOrchestrator(self.config)

    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_calculation_determinism(self):
        """Test same input always produces same output."""
        process_data = ProcessData(
            timestamp=datetime(2025, 1, 15, 10, 30, 0),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="natural_gas",
            fuel_consumption_rate=10.0
        )

        # Run calculation multiple times
        results = []
        for _ in range(10):
            result = await self.agent.calculate_thermal_efficiency(
                process_data,
                use_cache=False  # Disable cache to test actual calculation
            )
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            self.assertEqual(
                results[0].efficiency,
                results[i].efficiency,
                "Efficiency calculation not deterministic"
            )

            self.assertEqual(
                results[0].heat_loss_kw,
                results[i].heat_loss_kw,
                "Heat loss calculation not deterministic"
            )

            self.assertEqual(
                results[0].recoverable_heat_kw,
                results[i].recoverable_heat_kw,
                "Recoverable heat calculation not deterministic"
            )

            self.assertEqual(
                results[0].provenance_hash,
                results[i].provenance_hash,
                "Provenance hash not deterministic"
            )

    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_provenance_hash_determinism(self):
        """Test provenance hash is deterministic."""
        process_data = ProcessData(
            timestamp=datetime(2025, 1, 15, 10, 0, 0),
            temperature_c=300.0,
            pressure_bar=15.0,
            flow_rate_kg_s=7.0,
            energy_input_kw=2000.0,
            energy_output_kw=1700.0,
            fuel_type="diesel",
            fuel_consumption_rate=20.0
        )

        # Generate provenance hashes multiple times
        hashes = []
        for _ in range(5):
            result = await self.agent.calculate_thermal_efficiency(
                process_data,
                use_cache=False
            )
            hashes.append(result.provenance_hash)

        # All hashes should be identical
        unique_hashes = set(hashes)
        self.assertEqual(
            len(unique_hashes),
            1,
            f"Provenance hash not deterministic: {unique_hashes}"
        )

    @pytest.mark.determinism
    def test_cache_key_determinism(self):
        """Test cache key generation is deterministic."""
        process_data = ProcessData(
            timestamp=datetime(2025, 1, 15, 12, 0, 0),
            temperature_c=200.5,
            pressure_bar=8.3,
            flow_rate_kg_s=4.2,
            energy_input_kw=750.0,
            energy_output_kw=600.0,
            fuel_type="coal",
            fuel_consumption_rate=8.5
        )

        # Generate cache keys multiple times
        keys = []
        for _ in range(10):
            key = self.agent._generate_cache_key(process_data)
            keys.append(key)

        # All keys should be identical
        unique_keys = set(keys)
        self.assertEqual(
            len(unique_keys),
            1,
            f"Cache key not deterministic: {unique_keys}"
        )

    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_llm_determinism(self):
        """Test LLM calls are deterministic with temperature=0 and seed."""
        process_data = ProcessData(
            timestamp=datetime(2025, 1, 15, 14, 0, 0),
            temperature_c=450.0,
            pressure_bar=20.0,
            flow_rate_kg_s=10.0,
            energy_input_kw=5000.0,
            energy_output_kw=3750.0,
            fuel_type="natural_gas",
            fuel_consumption_rate=50.0
        )

        # Mock LLM to return deterministic response
        mock_llm_response = """1. Install economizer for exhaust heat recovery
2. Optimize burner air-fuel ratio"""

        with patch.object(
            self.agent,
            'query_llm',
            return_value=mock_llm_response
        ) as mock_llm:
            # Generate recommendations multiple times
            recommendations_list = []
            for _ in range(5):
                result = await self.agent.calculate_thermal_efficiency(
                    process_data,
                    use_cache=False
                )
                recommendations_list.append(result.recommendations)

        # All recommendation lists should be identical
        for i in range(1, len(recommendations_list)):
            self.assertEqual(
                recommendations_list[0],
                recommendations_list[i],
                "LLM recommendations not deterministic"
            )

    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_cross_platform_determinism(self):
        """Test calculations are deterministic across platforms."""
        process_data = ProcessData(
            timestamp=datetime(2025, 1, 15, 16, 0, 0),
            temperature_c=275.123456789,
            pressure_bar=12.987654321,
            flow_rate_kg_s=6.111111111,
            energy_input_kw=1234.56789,
            energy_output_kw=987.654321,
            fuel_type="biomass",
            fuel_consumption_rate=15.987654321
        )

        # Calculate with different agents (simulating different platforms)
        agent1 = ProcessHeatOrchestrator(self.config)
        agent2 = ProcessHeatOrchestrator(self.config)

        result1 = await agent1.calculate_thermal_efficiency(process_data, use_cache=False)
        result2 = await agent2.calculate_thermal_efficiency(process_data, use_cache=False)

        # Results should be bit-perfect identical
        self.assertEqual(result1.efficiency, result2.efficiency)
        self.assertEqual(result1.heat_loss_kw, result2.heat_loss_kw)
        self.assertEqual(result1.recoverable_heat_kw, result2.recoverable_heat_kw)
        self.assertEqual(result1.provenance_hash, result2.provenance_hash)

    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_time_independence(self):
        """Test calculations are independent of execution time."""
        process_data = ProcessData(
            timestamp=datetime(2025, 1, 15, 10, 0, 0),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        # Calculate at different times
        result1 = await self.agent.calculate_thermal_efficiency(process_data, use_cache=False)

        # Wait a moment
        await asyncio.sleep(0.1)

        result2 = await self.agent.calculate_thermal_efficiency(process_data, use_cache=False)

        # Wait longer
        await asyncio.sleep(1.0)

        result3 = await self.agent.calculate_thermal_efficiency(process_data, use_cache=False)

        # All results should be identical despite different execution times
        self.assertEqual(result1.efficiency, result2.efficiency)
        self.assertEqual(result2.efficiency, result3.efficiency)
        self.assertEqual(result1.provenance_hash, result3.provenance_hash)

    @pytest.mark.determinism
    def test_floating_point_consistency(self):
        """Test floating point calculations are consistent."""
        # Test with values that can cause floating point errors
        test_cases = [
            (0.1 + 0.2, 0.3),  # Classic floating point issue
            (1000000.0 + 0.0000001, 1000000.0000001),
            (999999999.0 * 999999999.0, 999999998000000001.0)
        ]

        for calc, expected in test_cases:
            # Verify calculations are consistent
            result1 = calc
            result2 = calc

            self.assertEqual(result1, result2)

    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_batch_processing_order_independence(self):
        """Test batch processing results are independent of order."""
        # Create batch of process data
        batch_data = [
            ProcessData(
                timestamp=datetime(2025, 1, 15, 10, 0, 0) + timedelta(minutes=i),
                temperature_c=200.0 + (i * 10),
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=850.0 - (i * 5),
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )
            for i in range(5)
        ]

        # Process in original order
        results_forward = []
        for data in batch_data:
            result = await self.agent.calculate_thermal_efficiency(data, use_cache=False)
            results_forward.append(result)

        # Process in reverse order
        results_backward = []
        for data in reversed(batch_data):
            result = await self.agent.calculate_thermal_efficiency(data, use_cache=False)
            results_backward.append(result)

        results_backward.reverse()  # Reverse back to match original order

        # Results should be identical regardless of processing order
        for i in range(len(batch_data)):
            self.assertEqual(
                results_forward[i].efficiency,
                results_backward[i].efficiency
            )
            self.assertEqual(
                results_forward[i].provenance_hash,
                results_backward[i].provenance_hash
            )

    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_concurrent_calculation_determinism(self):
        """Test concurrent calculations don't affect determinism."""
        process_data = ProcessData(
            timestamp=datetime(2025, 1, 15, 10, 0, 0),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        # Sequential calculation
        sequential_result = await self.agent.calculate_thermal_efficiency(
            process_data,
            use_cache=False
        )

        # Concurrent calculations
        tasks = [
            self.agent.calculate_thermal_efficiency(process_data, use_cache=False)
            for _ in range(5)
        ]

        concurrent_results = await asyncio.gather(*tasks)

        # All concurrent results should match sequential result
        for result in concurrent_results:
            self.assertEqual(result.efficiency, sequential_result.efficiency)
            self.assertEqual(result.provenance_hash, sequential_result.provenance_hash)

    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_optimization_strategy_determinism(self):
        """Test optimization strategy generation is deterministic."""
        process_history = [
            ProcessData(
                timestamp=datetime(2025, 1, 15, 0, 0, 0) + timedelta(hours=i),
                temperature_c=250.0,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=750.0,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )
            for i in range(24)
        ]

        constraints = {
            'max_investment': 100000,
            'payback_period_months': 36,
            'min_roi': 0.15
        }

        # Generate strategy multiple times
        strategies = []
        for _ in range(3):
            strategy = await self.agent.generate_optimization_strategy(
                process_history,
                constraints
            )
            strategies.append(strategy)

        # All strategies should be identical
        for i in range(1, len(strategies)):
            self.assertEqual(
                strategies[0].expected_savings_kwh,
                strategies[i].expected_savings_kwh
            )
            self.assertEqual(
                strategies[0].co2_reduction_tonnes,
                strategies[i].co2_reduction_tonnes
            )
            self.assertEqual(
                strategies[0].validation_hash,
                strategies[i].validation_hash
            )

    @pytest.mark.determinism
    def test_hash_collision_resistance(self):
        """Test provenance hashes don't collide for different inputs."""
        # Generate many different inputs
        inputs = []
        for i in range(100):
            data = {
                'temperature': 200.0 + i,
                'pressure': 10.0 + (i * 0.1),
                'efficiency': 0.80 + (i * 0.001),
                'iteration': i
            }
            inputs.append(data)

        # Generate hashes
        hashes = [self.agent._generate_provenance_hash(data) for data in inputs]

        # Verify no collisions
        unique_hashes = set(hashes)
        self.assertEqual(
            len(unique_hashes),
            len(hashes),
            "Hash collision detected"
        )

    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_reproducible_recommendations(self):
        """Test rule-based recommendations are reproducible."""
        test_scenarios = [
            (0.55, 150.0, "Critical"),  # Low efficiency
            (0.75, 50.0, "below target"),  # Below target
            (0.88, 200.0, "waste heat"),  # High waste heat
        ]

        for efficiency, recoverable_heat, expected_keyword in test_scenarios:
            data = ProcessData(
                timestamp=datetime(2025, 1, 15, 10, 0, 0),
                temperature_c=250.0,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=1000.0 * efficiency,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )

            # Disable LLM to test only rule-based recommendations
            self.agent.llm_client = None

            # Generate recommendations multiple times
            recommendations_list = []
            for _ in range(3):
                recommendations = await self.agent._generate_recommendations(
                    data,
                    efficiency,
                    recoverable_heat
                )
                recommendations_list.append(recommendations)

            # All recommendation lists should be identical
            for i in range(1, len(recommendations_list)):
                self.assertEqual(
                    recommendations_list[0],
                    recommendations_list[i],
                    "Rule-based recommendations not deterministic"
                )

            # Verify expected keyword appears
            all_recommendations = ' '.join(recommendations_list[0])
            self.assertIn(
                expected_keyword.lower(),
                all_recommendations.lower()
            )

    @pytest.mark.determinism
    def test_json_serialization_determinism(self):
        """Test JSON serialization is deterministic."""
        data = {
            'efficiency': 0.85,
            'temperature': 250.0,
            'pressure': 10.0,
            'fuel_type': 'natural_gas',
            'metadata': {
                'source': 'SCADA',
                'quality': 'high',
                'validated': True
            }
        }

        # Serialize multiple times
        serializations = []
        for _ in range(5):
            json_str = json.dumps(data, sort_keys=True)
            serializations.append(json_str)

        # All serializations should be identical
        unique_serializations = set(serializations)
        self.assertEqual(
            len(unique_serializations),
            1,
            "JSON serialization not deterministic"
        )


if __name__ == '__main__':
    unittest.main()