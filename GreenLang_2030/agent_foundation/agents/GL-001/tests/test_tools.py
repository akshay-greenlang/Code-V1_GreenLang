"""
Tool function tests for GL-001 ProcessHeatOrchestrator
Tests all tool functions independently with mocked dependencies.
Target coverage: 95% of tool functionality.
"""

import unittest
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agents.GL_001.process_heat_orchestrator import (
    ProcessHeatOrchestrator,
    ProcessHeatConfig,
    ProcessData
)
from testing.agent_test_framework import AgentTestCase


class TestProcessHeatTools(AgentTestCase):
    """Test suite for ProcessHeatOrchestrator tool functions."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.agent = ProcessHeatOrchestrator()

    # Data Validation Tools
    def test_validate_process_data_valid(self):
        """Test validation accepts valid process data."""
        valid_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="natural_gas",
            fuel_consumption_rate=10.0
        )

        # Should not raise exception
        self.agent._validate_process_data(valid_data)

    def test_validate_temperature_range(self):
        """Test temperature validation."""
        # Below minimum
        data_low = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=-300.0,  # Below absolute zero
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        with self.assertRaises(ValueError) as context:
            self.agent._validate_process_data(data_low)
        self.assertIn("Temperature", str(context.exception))

        # Above maximum
        data_high = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=4000.0,  # Above max
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        with self.assertRaises(ValueError) as context:
            self.agent._validate_process_data(data_high)
        self.assertIn("Temperature", str(context.exception))

    def test_validate_pressure_range(self):
        """Test pressure validation."""
        # Negative pressure
        data_negative = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=-10.0,  # Invalid negative
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        with self.assertRaises(ValueError) as context:
            self.agent._validate_process_data(data_negative)
        self.assertIn("Pressure", str(context.exception))

        # Above maximum
        data_high = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=2000.0,  # Above max
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        with self.assertRaises(ValueError) as context:
            self.agent._validate_process_data(data_high)
        self.assertIn("Pressure", str(context.exception))

    # Calculation Tools
    def test_calculate_efficiency_core(self):
        """Test core efficiency calculation."""
        data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        efficiency = self.agent._calculate_efficiency_core(data)
        self.assertAlmostEqual(efficiency, 0.85, places=2)

    def test_calculate_efficiency_zero_input(self):
        """Test efficiency with zero input."""
        data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=100.0,
            pressure_bar=5.0,
            flow_rate_kg_s=1.0,
            energy_input_kw=0.0,
            energy_output_kw=0.0,
            fuel_type="none",
            fuel_consumption_rate=0.0
        )

        efficiency = self.agent._calculate_efficiency_core(data)
        self.assertEqual(efficiency, 0.0)

    def test_calculate_efficiency_clamping(self):
        """Test efficiency clamping to [0, 1] range."""
        # Test clamping to 1.0 (shouldn't exceed 100%)
        data_high = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=1010.0,  # Slight measurement error
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        # Temporarily disable validation for this test
        with patch.object(self.agent, '_validate_process_data'):
            efficiency = self.agent._calculate_efficiency_core(data_high)
            self.assertEqual(efficiency, 1.0)  # Clamped to max

    def test_calculate_heat_loss(self):
        """Test heat loss calculation."""
        data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        heat_loss = self.agent._calculate_heat_loss(data)
        self.assertAlmostEqual(heat_loss, 150.0, places=1)

    def test_calculate_recoverable_heat(self):
        """Test recoverable heat calculation based on temperature."""
        data_low_temp = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=50.0,  # Low-grade heat
            pressure_bar=5.0,
            flow_rate_kg_s=2.0,
            energy_input_kw=100.0,
            energy_output_kw=70.0,
            fuel_type="gas",
            fuel_consumption_rate=1.0
        )

        heat_loss = self.agent._calculate_heat_loss(data_low_temp)
        recoverable = self.agent._calculate_recoverable_heat(data_low_temp, heat_loss)
        self.assertAlmostEqual(recoverable, heat_loss * 0.1, places=1)  # 10% recovery

        data_high_temp = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=600.0,  # Very high-grade heat
            pressure_bar=20.0,
            flow_rate_kg_s=10.0,
            energy_input_kw=5000.0,
            energy_output_kw=3500.0,
            fuel_type="coal",
            fuel_consumption_rate=50.0
        )

        heat_loss = self.agent._calculate_heat_loss(data_high_temp)
        recoverable = self.agent._calculate_recoverable_heat(data_high_temp, heat_loss)
        self.assertAlmostEqual(recoverable, heat_loss * 0.7, places=1)  # 70% recovery

    def test_calculate_optimization_potential(self):
        """Test optimization potential calculation."""
        # Below target
        potential = self.agent._calculate_optimization_potential(0.65, 0.85)
        self.assertAlmostEqual(potential, 23.53, places=1)  # (0.85-0.65)/0.85 * 100

        # At target
        potential = self.agent._calculate_optimization_potential(0.85, 0.85)
        self.assertEqual(potential, 0.0)

        # Above target
        potential = self.agent._calculate_optimization_potential(0.90, 0.85)
        self.assertEqual(potential, 0.0)  # No negative potential

    # Hash and Cache Tools
    def test_generate_cache_key(self):
        """Test cache key generation."""
        data1 = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=200.123,
            pressure_bar=10.456,
            flow_rate_kg_s=5.789,
            energy_input_kw=1000.123,
            energy_output_kw=850.456,
            fuel_type="diesel",
            fuel_consumption_rate=10.0
        )

        data2 = ProcessData(
            timestamp=datetime.utcnow() + timedelta(hours=1),  # Different timestamp
            temperature_c=200.123,
            pressure_bar=10.456,
            flow_rate_kg_s=5.789,
            energy_input_kw=1000.123,
            energy_output_kw=850.456,
            fuel_type="diesel",
            fuel_consumption_rate=10.0  # Different rate (not in cache key)
        )

        key1 = self.agent._generate_cache_key(data1)
        key2 = self.agent._generate_cache_key(data2)

        # Keys should be same (timestamp not included in cache key)
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 16)

        # Different fuel type should give different key
        data3 = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=200.123,
            pressure_bar=10.456,
            flow_rate_kg_s=5.789,
            energy_input_kw=1000.123,
            energy_output_kw=850.456,
            fuel_type="natural_gas",  # Different fuel
            fuel_consumption_rate=10.0
        )

        key3 = self.agent._generate_cache_key(data3)
        self.assertNotEqual(key1, key3)

    def test_generate_provenance_hash(self):
        """Test provenance hash generation."""
        data = {
            'input': {'temp': 200.0, 'pressure': 10.0},
            'calculation': {'efficiency': 0.85, 'heat_loss': 150.0},
            'timestamp': '2025-01-15T10:00:00'
        }

        hash1 = self.agent._generate_provenance_hash(data)

        # Verify hash properties
        self.assertEqual(len(hash1), 64)  # SHA-256
        self.assertTrue(all(c in '0123456789abcdef' for c in hash1))

        # Same data should give same hash
        hash2 = self.agent._generate_provenance_hash(data)
        self.assertEqual(hash1, hash2)

        # Different data should give different hash
        data['input']['temp'] = 201.0
        hash3 = self.agent._generate_provenance_hash(data)
        self.assertNotEqual(hash1, hash3)

    # Recommendation Tools
    @pytest.mark.asyncio
    async def test_generate_recommendations_rule_based(self):
        """Test rule-based recommendation generation."""
        # Low efficiency scenario
        data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=200.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=500.0,  # 50% efficiency
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        # Temporarily set LLM client to None to test rule-based only
        self.agent.llm_client = None

        recommendations = await self.agent._generate_recommendations(
            data,
            efficiency=0.5,
            recoverable_heat=250.0
        )

        # Should have critical efficiency warning
        self.assertTrue(any("Critical" in r for r in recommendations))
        self.assertTrue(any("60%" in r for r in recommendations))

        # Should have waste heat recovery recommendation
        self.assertTrue(any("waste heat" in r.lower() for r in recommendations))

    @pytest.mark.asyncio
    async def test_generate_recommendations_with_llm(self):
        """Test recommendation generation with LLM integration."""
        data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=700.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        # Mock LLM response
        mock_llm_response = """1. Install economizer to recover exhaust heat
2. Optimize combustion air-fuel ratio"""

        with patch.object(self.agent, 'query_llm', return_value=mock_llm_response):
            recommendations = await self.agent._generate_recommendations(
                data,
                efficiency=0.7,
                recoverable_heat=150.0
            )

        # Should include both rule-based and LLM recommendations
        self.assertTrue(len(recommendations) >= 3)
        self.assertTrue(any("economizer" in r.lower() for r in recommendations))

    # Integration Connection Tools
    @pytest.mark.asyncio
    async def test_connect_scada(self):
        """Test SCADA connection tool."""
        with patch('logging.Logger.info') as mock_log:
            await self.agent._connect_scada()

        # Should log connection
        mock_log.assert_called_with("SCADA connection established (mock)")

    @pytest.mark.asyncio
    async def test_connect_erp(self):
        """Test ERP connection tool."""
        with patch('logging.Logger.info') as mock_log:
            await self.agent._connect_erp()

        # Should log connection
        mock_log.assert_called_with("ERP connection established (mock)")

    @pytest.mark.asyncio
    async def test_load_historical_data(self):
        """Test historical data loading tool."""
        with patch('logging.Logger.info') as mock_log:
            await self.agent._load_historical_data()

        # Should log data loading
        mock_log.assert_called_with("Historical data loaded (mock)")

    # State Management Tools
    @pytest.mark.asyncio
    async def test_save_state(self):
        """Test state saving tool."""
        # Add some data to agent
        self.agent.metrics['calculations_performed'] = 50
        self.agent.provenance_chain = ['hash' + str(i) for i in range(150)]
        self.agent.calculations_cache = {'key1': Mock(), 'key2': Mock()}
        self.agent.optimization_strategies = [Mock(), Mock(), Mock()]

        with patch('logging.Logger.info') as mock_log:
            await self.agent._save_state()

        # Should log state saving with correct counts
        call_args = mock_log.call_args[0][0]
        self.assertIn("'calculations_performed': 50", call_args)
        self.assertIn("'cache_size': 2", call_args)
        self.assertIn("'strategies_count': 3", call_args)

        # Should only keep last 100 provenance entries
        self.assertIn("100", call_args)  # Provenance chain limited to 100

    # Metrics Tools
    def test_get_metrics(self):
        """Test metrics retrieval tool."""
        # Set up some metrics
        self.agent.metrics['calculations_performed'] = 10
        self.agent.metrics['total_calculation_time_ms'] = 500.0
        self.agent.metrics['cache_hits'] = 3
        self.agent.metrics['cache_misses'] = 7
        self.agent.provenance_chain = ['hash1', 'hash2']

        metrics = self.agent.get_metrics()

        # Check calculated metrics
        self.assertEqual(metrics['calculations_performed'], 10)
        self.assertAlmostEqual(metrics['avg_calculation_time_ms'], 50.0, places=1)
        self.assertAlmostEqual(metrics['cache_hit_rate'], 0.3, places=1)
        self.assertEqual(metrics['provenance_entries'], 2)
        self.assertEqual(metrics['state'], 'CREATED')

    def test_get_metrics_empty(self):
        """Test metrics with no data."""
        metrics = self.agent.get_metrics()

        self.assertEqual(metrics['calculations_performed'], 0)
        self.assertEqual(metrics['avg_calculation_time_ms'], 0.0)
        self.assertEqual(metrics['cache_hit_rate'], 0.0)
        self.assertEqual(metrics['provenance_entries'], 0)


if __name__ == '__main__':
    unittest.main()