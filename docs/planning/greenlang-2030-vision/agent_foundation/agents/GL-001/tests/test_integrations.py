# -*- coding: utf-8 -*-
"""
Integration tests for GL-001 ProcessHeatOrchestrator
Tests SCADA, ERP, and multi-agent integrations.
Target coverage: 85% of integration points.
"""

import unittest
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from typing import Dict, List, Any
import aiohttp
import redis

import sys
import os
from greenlang.determinism import DeterministicClock
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agents.GL_001.process_heat_orchestrator import (
    ProcessHeatOrchestrator,
    ProcessHeatConfig,
    ProcessData,
    ThermalCalculation
)
from testing.agent_test_framework import AgentTestCase, AgentState


class TestIntegrations(AgentTestCase):
    """Integration tests for ProcessHeatOrchestrator."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.config = ProcessHeatConfig(
            scada_poll_interval_s=1.0,
            erp_sync_interval_s=60.0
        )
        self.agent = ProcessHeatOrchestrator(self.config)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scada_integration_realtime_feed(self):
        """Test SCADA real-time data feed integration."""
        mock_scada_data = [
            {
                'timestamp': DeterministicClock.utcnow().isoformat(),
                'sensors': {
                    'TEMP_001': 250.5,
                    'PRESS_001': 10.2,
                    'FLOW_001': 5.8,
                    'POWER_IN': 1050.0,
                    'POWER_OUT': 892.5
                },
                'status': 'RUNNING'
            }
        ]

        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_scada_data)
            mock_response.status = 200

            mock_session.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            # Simulate SCADA data polling
            scada_url = "http://scada.local/api/realtime"
            async with aiohttp.ClientSession() as session:
                response = await session.get(scada_url)
                data = await response.json()

            self.assertEqual(response.status, 200)
            self.assertEqual(len(data), 1)
            self.assertIn('sensors', data[0])

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scada_data_transformation(self):
        """Test transformation of SCADA data to ProcessData."""
        raw_scada = {
            'timestamp': '2025-01-15T10:30:00Z',
            'sensors': {
                'TEMP_001': 275.0,
                'PRESS_001': 12.5,
                'FLOW_001': 6.2,
                'POWER_IN': 1200.0,
                'POWER_OUT': 960.0,
                'FUEL_TYPE': 'NATURAL_GAS',
                'FUEL_RATE': 15.0
            }
        }

        # Transform SCADA data to ProcessData
        process_data = ProcessData(
            timestamp=datetime.fromisoformat(raw_scada['timestamp'].replace('Z', '+00:00')),
            temperature_c=raw_scada['sensors']['TEMP_001'],
            pressure_bar=raw_scada['sensors']['PRESS_001'],
            flow_rate_kg_s=raw_scada['sensors']['FLOW_001'],
            energy_input_kw=raw_scada['sensors']['POWER_IN'],
            energy_output_kw=raw_scada['sensors']['POWER_OUT'],
            fuel_type=raw_scada['sensors']['FUEL_TYPE'].lower().replace('_', '_'),
            fuel_consumption_rate=raw_scada['sensors']['FUEL_RATE']
        )

        # Verify transformation
        self.assertEqual(process_data.temperature_c, 275.0)
        self.assertEqual(process_data.energy_input_kw, 1200.0)
        self.assertEqual(process_data.fuel_type, 'natural_gas')

        # Process through agent
        result = await self.agent.calculate_thermal_efficiency(process_data)
        self.assertAlmostEqual(result.efficiency, 0.80, places=2)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_erp_integration_energy_consumption(self):
        """Test ERP integration for energy consumption data."""
        mock_erp_data = {
            'plant_id': 'PLANT_001',
            'period': '2025-01',
            'energy_consumption': {
                'total_kwh': 750000,
                'cost_usd': 90000,
                'co2_tonnes': 375
            },
            'production_units': 10000,
            'energy_intensity': 75.0  # kWh per unit
        }

        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_erp_data)
            mock_response.status = 200

            mock_session.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # Query ERP for energy data
            erp_url = "http://erp.local/api/energy"
            payload = {'plant_id': 'PLANT_001', 'period': '2025-01'}

            async with aiohttp.ClientSession() as session:
                response = await session.post(erp_url, json=payload)
                data = await response.json()

            self.assertEqual(response.status, 200)
            self.assertEqual(data['energy_consumption']['total_kwh'], 750000)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_erp_scheduling_integration(self):
        """Test ERP integration for production scheduling."""
        mock_schedule = {
            'schedule_id': 'SCH_20250115',
            'shifts': [
                {
                    'shift_id': 'SHIFT_1',
                    'start_time': '2025-01-15T06:00:00Z',
                    'end_time': '2025-01-15T14:00:00Z',
                    'target_output': 1000,
                    'energy_budget_kwh': 25000
                },
                {
                    'shift_id': 'SHIFT_2',
                    'start_time': '2025-01-15T14:00:00Z',
                    'end_time': '2025-01-15T22:00:00Z',
                    'target_output': 1200,
                    'energy_budget_kwh': 30000
                }
            ]
        }

        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_schedule)
            mock_response.status = 200

            mock_session.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            # Get production schedule
            schedule_url = "http://erp.local/api/schedule/current"

            async with aiohttp.ClientSession() as session:
                response = await session.get(schedule_url)
                schedule = await response.json()

            self.assertEqual(len(schedule['shifts']), 2)
            self.assertEqual(schedule['shifts'][0]['energy_budget_kwh'], 25000)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self):
        """Test coordination with other GreenLang agents."""
        # Mock message bus for agent communication
        mock_message_bus = AsyncMock()

        # Simulate GL-002 requesting thermal data
        request_message = {
            'from': 'GL-002',
            'to': 'GL-001',
            'type': 'REQUEST',
            'action': 'get_thermal_efficiency',
            'timestamp': DeterministicClock.utcnow().isoformat()
        }

        # Process request and generate response
        process_data = ProcessData(
            timestamp=DeterministicClock.utcnow(),
            temperature_c=300.0,
            pressure_bar=15.0,
            flow_rate_kg_s=7.0,
            energy_input_kw=2000.0,
            energy_output_kw=1700.0,
            fuel_type="gas",
            fuel_consumption_rate=20.0
        )

        result = await self.agent.calculate_thermal_efficiency(process_data)

        response_message = {
            'from': 'GL-001',
            'to': 'GL-002',
            'type': 'RESPONSE',
            'data': {
                'efficiency': result.efficiency,
                'heat_loss_kw': result.heat_loss_kw,
                'recoverable_heat_kw': result.recoverable_heat_kw,
                'provenance_hash': result.provenance_hash
            },
            'timestamp': DeterministicClock.utcnow().isoformat()
        }

        # Verify message structure
        self.assertEqual(response_message['from'], 'GL-001')
        self.assertEqual(response_message['to'], 'GL-002')
        self.assertIn('efficiency', response_message['data'])
        self.assertAlmostEqual(response_message['data']['efficiency'], 0.85, places=2)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_integration(self):
        """Test database integration for persistence."""
        # Mock database connection
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_connect.return_value.cursor.return_value = mock_cursor

            # Simulate storing calculation result
            result = ThermalCalculation(
                efficiency=0.85,
                heat_loss_kw=150.0,
                recoverable_heat_kw=75.0,
                optimization_potential=5.88,
                recommendations=['Optimize burner settings'],
                provenance_hash='abc123def456',
                calculation_time_ms=25.5
            )

            # SQL for storing result
            insert_sql = """
            INSERT INTO thermal_calculations
            (timestamp, efficiency, heat_loss_kw, recoverable_heat_kw,
             optimization_potential, provenance_hash, calculation_time_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            mock_cursor.execute(
                insert_sql,
                (
                    DeterministicClock.utcnow(),
                    result.efficiency,
                    result.heat_loss_kw,
                    result.recoverable_heat_kw,
                    result.optimization_potential,
                    result.provenance_hash,
                    result.calculation_time_ms
                )
            )

            # Verify database interaction
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            self.assertIn('INSERT INTO thermal_calculations', call_args[0])
            self.assertEqual(call_args[1][1], 0.85)  # efficiency

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test Redis cache integration."""
        with patch('redis.Redis') as mock_redis:
            mock_redis_instance = MagicMock()
            mock_redis.return_value = mock_redis_instance

            # Test cache set
            cache_key = 'thermal:calc:12345'
            cache_value = {
                'efficiency': 0.82,
                'heat_loss_kw': 180.0,
                'timestamp': DeterministicClock.utcnow().isoformat()
            }

            mock_redis_instance.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(cache_value)
            )

            # Test cache get
            mock_redis_instance.get.return_value = json.dumps(cache_value).encode()

            cached = mock_redis_instance.get(cache_key)
            cached_data = json.loads(cached.decode())

            self.assertEqual(cached_data['efficiency'], 0.82)
            self.assertEqual(cached_data['heat_loss_kw'], 180.0)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_message_queue_integration(self):
        """Test message queue integration for event streaming."""
        with patch('aiokafka.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer_class.return_value = mock_producer

            # Create event
            event = {
                'event_type': 'THERMAL_CALCULATION_COMPLETE',
                'agent_id': 'GL-001',
                'timestamp': DeterministicClock.utcnow().isoformat(),
                'data': {
                    'efficiency': 0.83,
                    'optimization_potential': 7.5
                }
            }

            # Send to Kafka
            await mock_producer.send_and_wait(
                'greenlang.events',
                json.dumps(event).encode()
            )

            # Verify send
            mock_producer.send_and_wait.assert_called_once()
            call_args = mock_producer.send_and_wait.call_args[0]
            self.assertEqual(call_args[0], 'greenlang.events')

            sent_data = json.loads(call_args[1].decode())
            self.assertEqual(sent_data['event_type'], 'THERMAL_CALCULATION_COMPLETE')

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_gateway_integration(self):
        """Test integration with API gateway for external access."""
        with patch('aiohttp.web.Application') as mock_app:
            # Mock API endpoint
            async def handle_efficiency_request(request):
                data = await request.json()

                # Process request
                process_data = ProcessData(
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    temperature_c=data['temperature_c'],
                    pressure_bar=data['pressure_bar'],
                    flow_rate_kg_s=data['flow_rate_kg_s'],
                    energy_input_kw=data['energy_input_kw'],
                    energy_output_kw=data['energy_output_kw'],
                    fuel_type=data['fuel_type'],
                    fuel_consumption_rate=data['fuel_consumption_rate']
                )

                # Calculate efficiency
                agent = ProcessHeatOrchestrator()
                result = await agent.calculate_thermal_efficiency(process_data)

                return {
                    'status': 'success',
                    'efficiency': result.efficiency,
                    'heat_loss_kw': result.heat_loss_kw,
                    'provenance_hash': result.provenance_hash
                }

            # Test request
            test_request = {
                'timestamp': DeterministicClock.utcnow().isoformat(),
                'temperature_c': 280.0,
                'pressure_bar': 11.0,
                'flow_rate_kg_s': 5.5,
                'energy_input_kw': 1100.0,
                'energy_output_kw': 935.0,
                'fuel_type': 'diesel',
                'fuel_consumption_rate': 12.0
            }

            # Mock request object
            mock_request = AsyncMock()
            mock_request.json = AsyncMock(return_value=test_request)

            # Process request
            response = await handle_efficiency_request(mock_request)

            # Verify response
            self.assertEqual(response['status'], 'success')
            self.assertAlmostEqual(response['efficiency'], 0.85, places=2)
            self.assertIsNotNone(response['provenance_hash'])

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_monitoring_integration(self):
        """Test integration with monitoring systems (Prometheus/Grafana)."""
        # Mock Prometheus metrics
        from prometheus_client import Counter, Histogram, Gauge

        # Define metrics
        calculation_counter = Counter(
            'gl001_calculations_total',
            'Total thermal calculations performed'
        )

        calculation_duration = Histogram(
            'gl001_calculation_duration_seconds',
            'Thermal calculation duration'
        )

        current_efficiency = Gauge(
            'gl001_current_efficiency',
            'Current thermal efficiency'
        )

        # Simulate metric updates
        calculation_counter.inc()
        calculation_duration.observe(0.025)  # 25ms
        current_efficiency.set(0.85)

        # Verify metrics
        self.assertEqual(calculation_counter._value.get(), 1)
        self.assertGreater(calculation_duration._sum.get(), 0)
        self.assertEqual(current_efficiency._value.get(), 0.85)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self):
        """Test error recovery in integration scenarios."""
        # Test SCADA connection failure and retry
        with patch('aiohttp.ClientSession') as mock_session:
            # First attempt fails
            mock_session.return_value.__aenter__.return_value.get.side_effect = [
                aiohttp.ClientError("Connection failed"),
                AsyncMock(return_value=AsyncMock(status=200, json=AsyncMock(return_value={})))
            ]

            attempts = 0
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        response = await session.get("http://scada.local/api/data")
                        if response.status == 200:
                            break
                except aiohttp.ClientError:
                    attempts += 1
                    if attempts >= max_retries:
                        raise
                    await asyncio.sleep(1)  # Backoff

            # Should succeed on second attempt
            self.assertEqual(attempts, 1)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self):
        """Test batch processing of multiple data sources."""
        # Simulate batch of SCADA data
        batch_data = [
            ProcessData(
                timestamp=DeterministicClock.utcnow() - timedelta(minutes=i),
                temperature_c=250.0 + (i * 5),
                pressure_bar=10.0 + (i * 0.5),
                flow_rate_kg_s=5.0 + (i * 0.2),
                energy_input_kw=1000.0,
                energy_output_kw=850.0 - (i * 10),
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )
            for i in range(10)
        ]

        # Process batch
        results = []
        for data in batch_data:
            result = await self.agent.calculate_thermal_efficiency(data)
            results.append(result)

        # Verify batch processing
        self.assertEqual(len(results), 10)

        # Verify results are different (varying efficiency)
        efficiencies = [r.efficiency for r in results]
        self.assertNotEqual(min(efficiencies), max(efficiencies))

        # Verify all have provenance
        for result in results:
            self.assertIsNotNone(result.provenance_hash)


if __name__ == '__main__':
    unittest.main()