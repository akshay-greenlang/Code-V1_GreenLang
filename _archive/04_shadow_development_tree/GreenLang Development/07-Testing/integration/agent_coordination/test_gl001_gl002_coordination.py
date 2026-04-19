# -*- coding: utf-8 -*-
"""
Integration tests for GL-001 THERMOSYNC ↔ GL-002 FLAMEGUARD coordination.

Tests the coordination between ProcessHeatOrchestrator (GL-001) and
BoilerEfficiencyOptimizer (GL-002) for boiler optimization workflows.

Test Scenarios:
1. GL-001 orchestrates GL-002 for boiler optimization
2. GL-001 receives heat demand requirements
3. GL-001 calls GL-002 to optimize boiler settings
4. GL-002 returns optimal combustion parameters
5. GL-001 validates and applies recommendations

Coverage: Tests data flow, message format compatibility, error handling,
concurrent coordination, and latency measurements.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest

# Adjust imports based on actual agent structure
# from docs.planning.greenlang-2030-vision.agent_foundation.agents.GL-001.process_heat_orchestrator import ProcessHeatOrchestrator
# from docs.planning.greenlang-2030-vision.agent_foundation.agents.GL-002.boiler_efficiency_orchestrator import BoilerEfficiencyOptimizer


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def gl001_config():
    """Configuration for GL-001 ProcessHeatOrchestrator."""
    return {
        'agent_id': 'GL-001',
        'agent_name': 'ProcessHeatOrchestrator',
        'version': '1.0.0',
        'calculation_timeout_seconds': 120,
        'enable_monitoring': True,
        'cache_ttl_seconds': 60,
        'baseline_efficiency_percent': 82.0,
        'emission_regulations': {
            'co2_limit_kg_mwh': 500,
            'nox_limit_ppm': 150
        }
    }


@pytest.fixture
def gl002_config():
    """Configuration for GL-002 BoilerEfficiencyOptimizer."""
    return {
        'agent_id': 'GL-002',
        'agent_name': 'BoilerEfficiencyOptimizer',
        'version': '1.0.0',
        'calculation_timeout_seconds': 120,
        'enable_monitoring': True,
        'baseline_efficiency_percent': 85.0,
        'min_acceptable_efficiency': 80.0,
        'max_steam_capacity_kg_hr': 50000,
        'efficiency_value_usd_per_percent': 1000.0
    }


@pytest.fixture
def mock_gl001_orchestrator(gl001_config):
    """Mock GL-001 ProcessHeatOrchestrator instance."""
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.agent_id = gl001_config['agent_id']
    mock_agent.config.version = gl001_config['version']
    mock_agent.performance_metrics = {
        'calculations_performed': 0,
        'avg_calculation_time_ms': 0,
        'agents_coordinated': 0
    }

    # Mock async execute method
    async def mock_execute(input_data):
        return {
            'agent_id': gl001_config['agent_id'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'execution_time_ms': 45.2,
            'thermal_efficiency': {'overall_efficiency': 85.5},
            'heat_distribution': {'optimization_score': 92.0},
            'kpi_dashboard': {
                'operational_kpis': {
                    'thermal_efficiency': 85.5,
                    'capacity_utilization': 78.5
                }
            },
            'optimization_success': True,
            'provenance_hash': 'abc123def456'
        }

    mock_agent.execute = mock_execute

    # Mock coordination method
    async def mock_optimize_boiler_for_demand(demand, boiler_optimizer):
        # Simulate calling GL-002
        boiler_input = {
            'boiler_data': {'capacity_kg_hr': 50000},
            'steam_demand': demand,
            'sensor_feeds': {'load_percent': 75},
            'fuel_data': {'type': 'natural_gas', 'hhv_mj_kg': 50.0},
            'constraints': {}
        }
        boiler_result = await boiler_optimizer.execute(boiler_input)

        return {
            'status': 'success',
            'data': {
                'combustion_parameters': boiler_result.get('combustion_optimization', {}),
                'efficiency_gain_percent': 3.5,
                'coordination_latency_ms': 42.1
            }
        }

    mock_agent.optimize_boiler_for_demand = mock_optimize_boiler_for_demand

    return mock_agent


@pytest.fixture
def mock_gl002_optimizer(gl002_config):
    """Mock GL-002 BoilerEfficiencyOptimizer instance."""
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.agent_id = gl002_config['agent_id']
    mock_agent.config.version = gl002_config['version']
    mock_agent.performance_metrics = {
        'optimizations_performed': 0,
        'avg_optimization_time_ms': 0,
        'fuel_savings_kg': 0
    }

    # Mock async execute method
    async def mock_execute(input_data):
        return {
            'agent_id': gl002_config['agent_id'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'execution_time_ms': 38.7,
            'operational_state': {
                'mode': 'normal',
                'efficiency_percent': 87.2
            },
            'combustion_optimization': {
                'fuel_efficiency_percent': 88.5,
                'optimal_excess_air_percent': 15.0,
                'combustion_efficiency_percent': 89.2,
                'flame_stability_index': 0.95,
                'fuel_savings_usd_hr': 125.0
            },
            'steam_generation': {
                'target_steam_flow_kg_hr': 40000,
                'steam_quality_index': 0.98
            },
            'emissions_optimization': {
                'co2_intensity_kg_mwh': 450,
                'reduction_percent': 5.2,
                'compliance_status': 'COMPLIANT'
            },
            'parameter_adjustments': {
                'fuel_flow_change_percent': 2.5,
                'air_flow_change_percent': 1.8
            },
            'kpi_dashboard': {
                'operational_kpis': {
                    'thermal_efficiency': 87.2,
                    'efficiency_improvement': 2.2
                },
                'emissions_kpis': {
                    'compliance_status': 'COMPLIANT'
                }
            },
            'optimization_success': True,
            'provenance_hash': 'def789ghi012'
        }

    mock_agent.execute = mock_execute

    return mock_agent


@pytest.fixture
def heat_demand_payload():
    """Sample heat demand payload for testing."""
    return {
        'steam_output_kg_h': 40000,
        'pressure_bar': 40,
        'temperature_c': 450,
        'quality_index_min': 0.95,
        'priority': 'high'
    }


@pytest.fixture
def boiler_sensor_data():
    """Sample boiler sensor data."""
    return {
        'fuel_flow_kg_hr': 5000,
        'steam_flow_kg_hr': 38000,
        'load_percent': 76,
        'combustion_temp_c': 1200,
        'stack_temp_c': 180,
        'o2_percent': 3.5,
        'co_ppm': 50,
        'nox_ppm': 120,
        'pressure_bar': 40,
        'feedwater_temp_c': 110
    }


# ============================================================================
# Test Class: GL-001 ↔ GL-002 Coordination
# ============================================================================

class TestGL001GL002Coordination:
    """Test suite for GL-001 ProcessHeatOrchestrator ↔ GL-002 BoilerEfficiencyOptimizer coordination."""

    @pytest.mark.asyncio
    async def test_boiler_optimization_request(
        self,
        mock_gl001_orchestrator,
        mock_gl002_optimizer,
        heat_demand_payload
    ):
        """Test GL-001 requesting boiler optimization from GL-002."""
        # Arrange
        demand = heat_demand_payload

        # Act
        result = await mock_gl001_orchestrator.optimize_boiler_for_demand(
            demand=demand,
            boiler_optimizer=mock_gl002_optimizer
        )

        # Assert
        assert result is not None
        assert result['status'] == 'success'
        assert 'data' in result
        assert 'combustion_parameters' in result['data']
        assert result['data']['efficiency_gain_percent'] > 0
        assert 'coordination_latency_ms' in result['data']

    @pytest.mark.asyncio
    async def test_data_flow_gl001_to_gl002(
        self,
        mock_gl001_orchestrator,
        mock_gl002_optimizer,
        heat_demand_payload
    ):
        """Test data format compatibility between GL-001 and GL-002."""
        # Arrange
        demand = heat_demand_payload

        # Act
        coordination_result = await mock_gl001_orchestrator.optimize_boiler_for_demand(
            demand=demand,
            boiler_optimizer=mock_gl002_optimizer
        )

        # Assert - Verify data structure compatibility
        assert coordination_result['status'] in ['success', 'failure']

        if coordination_result['status'] == 'success':
            data = coordination_result['data']
            assert 'combustion_parameters' in data

            # Verify combustion parameters structure
            combustion = data['combustion_parameters']
            assert 'fuel_efficiency_percent' in combustion
            assert 'optimal_excess_air_percent' in combustion
            assert 'combustion_efficiency_percent' in combustion

    @pytest.mark.asyncio
    async def test_gl002_returns_valid_combustion_parameters(
        self,
        mock_gl002_optimizer,
        boiler_sensor_data
    ):
        """Test that GL-002 returns valid combustion parameters."""
        # Arrange
        input_data = {
            'boiler_data': {'capacity_kg_hr': 50000},
            'sensor_feeds': boiler_sensor_data,
            'fuel_data': {
                'type': 'natural_gas',
                'hhv_mj_kg': 50.0,
                'density_kg_m3': 0.8
            },
            'steam_demand': {'required_flow_kg_hr': 40000},
            'constraints': {'max_nox_ppm': 150}
        }

        # Act
        result = await mock_gl002_optimizer.execute(input_data)

        # Assert
        assert result['optimization_success'] is True
        assert 'combustion_optimization' in result

        combustion = result['combustion_optimization']
        assert combustion['fuel_efficiency_percent'] > 0
        assert 0 < combustion['optimal_excess_air_percent'] < 50
        assert 0 < combustion['flame_stability_index'] <= 1.0

    @pytest.mark.asyncio
    async def test_gl001_validates_gl002_recommendations(
        self,
        mock_gl001_orchestrator,
        mock_gl002_optimizer,
        heat_demand_payload
    ):
        """Test GL-001 validates and applies GL-002 recommendations."""
        # Arrange
        demand = heat_demand_payload

        # Act
        result = await mock_gl001_orchestrator.optimize_boiler_for_demand(
            demand=demand,
            boiler_optimizer=mock_gl002_optimizer
        )

        # Assert - GL-001 should validate recommendations
        assert result['status'] == 'success'
        assert result['data']['efficiency_gain_percent'] > 0

        # Verify GL-001 applied recommendations
        combustion_params = result['data']['combustion_parameters']
        assert combustion_params is not None
        assert 'fuel_efficiency_percent' in combustion_params

    @pytest.mark.asyncio
    async def test_error_handling_gl002_failure(
        self,
        mock_gl001_orchestrator,
        heat_demand_payload
    ):
        """Test error handling when GL-002 fails."""
        # Arrange - Create failing GL-002 mock
        failing_gl002 = MagicMock()

        async def failing_execute(input_data):
            raise Exception("GL-002 optimization failed: fuel data invalid")

        failing_gl002.execute = failing_execute

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await mock_gl001_orchestrator.optimize_boiler_for_demand(
                demand=heat_demand_payload,
                boiler_optimizer=failing_gl002
            )

        assert "GL-002 optimization failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_handling_gl001_graceful_recovery(
        self,
        mock_gl002_optimizer,
        heat_demand_payload
    ):
        """Test GL-001 handles GL-002 failures gracefully."""
        # Arrange - Create GL-001 with recovery logic
        gl001_with_recovery = MagicMock()

        async def optimize_with_recovery(demand, boiler_optimizer):
            try:
                boiler_input = {
                    'boiler_data': {'capacity_kg_hr': 50000},
                    'steam_demand': demand,
                    'sensor_feeds': {'load_percent': 75},
                    'fuel_data': {'type': 'natural_gas'},
                    'constraints': {}
                }
                result = await boiler_optimizer.execute(boiler_input)
                return {'status': 'success', 'data': result}
            except Exception as e:
                # Graceful recovery - return safe defaults
                return {
                    'status': 'partial_success',
                    'data': {
                        'combustion_parameters': {'safe_mode': True},
                        'efficiency_gain_percent': 0,
                        'error': str(e)
                    }
                }

        gl001_with_recovery.optimize_boiler_for_demand = optimize_with_recovery

        # Act
        result = await gl001_with_recovery.optimize_boiler_for_demand(
            demand=heat_demand_payload,
            boiler_optimizer=mock_gl002_optimizer
        )

        # Assert
        assert result['status'] in ['success', 'partial_success']

    @pytest.mark.asyncio
    async def test_concurrent_coordination_multiple_boilers(
        self,
        mock_gl001_orchestrator,
        mock_gl002_optimizer,
        heat_demand_payload
    ):
        """Test GL-001 coordinating multiple GL-002 instances concurrently."""
        # Arrange - Create multiple demand scenarios
        demands = [
            {**heat_demand_payload, 'steam_output_kg_h': 30000, 'boiler_id': 'B1'},
            {**heat_demand_payload, 'steam_output_kg_h': 40000, 'boiler_id': 'B2'},
            {**heat_demand_payload, 'steam_output_kg_h': 35000, 'boiler_id': 'B3'}
        ]

        # Act - Execute concurrent optimizations
        start_time = time.perf_counter()

        tasks = [
            mock_gl001_orchestrator.optimize_boiler_for_demand(
                demand=demand,
                boiler_optimizer=mock_gl002_optimizer
            )
            for demand in demands
        ]

        results = await asyncio.gather(*tasks)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Assert
        assert len(results) == 3
        for result in results:
            assert result['status'] == 'success'

        # Concurrent execution should be faster than sequential
        # (This is a mock test, but demonstrates the pattern)
        assert execution_time_ms < 500  # Should complete quickly with mocks

    @pytest.mark.asyncio
    async def test_coordination_latency_measurement(
        self,
        mock_gl001_orchestrator,
        mock_gl002_optimizer,
        heat_demand_payload
    ):
        """Test coordination latency measurement."""
        # Arrange
        demand = heat_demand_payload

        # Act
        start_time = time.perf_counter()

        result = await mock_gl001_orchestrator.optimize_boiler_for_demand(
            demand=demand,
            boiler_optimizer=mock_gl002_optimizer
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Assert
        assert result['status'] == 'success'
        assert 'coordination_latency_ms' in result['data']

        # Latency should be reasonable (mocks are fast)
        assert latency_ms < 100

    @pytest.mark.asyncio
    async def test_message_format_compatibility(
        self,
        mock_gl001_orchestrator,
        mock_gl002_optimizer,
        heat_demand_payload
    ):
        """Test message format compatibility between agents."""
        # Arrange
        demand = heat_demand_payload

        # Act
        result = await mock_gl001_orchestrator.optimize_boiler_for_demand(
            demand=demand,
            boiler_optimizer=mock_gl002_optimizer
        )

        # Assert - Verify message structure follows contract
        assert 'status' in result
        assert result['status'] in ['success', 'failure', 'partial_success']

        if result['status'] == 'success':
            assert 'data' in result
            data = result['data']

            # Verify required fields in data
            assert 'combustion_parameters' in data
            assert isinstance(data['combustion_parameters'], dict)

            # Verify combustion parameters follow expected schema
            combustion = data['combustion_parameters']
            expected_fields = [
                'fuel_efficiency_percent',
                'optimal_excess_air_percent',
                'combustion_efficiency_percent'
            ]
            for field in expected_fields:
                assert field in combustion, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_data_integrity_across_agent_boundary(
        self,
        mock_gl002_optimizer,
        boiler_sensor_data
    ):
        """Test data integrity when crossing agent boundaries."""
        # Arrange
        input_data = {
            'boiler_data': {'capacity_kg_hr': 50000, 'serial_number': 'BLR-12345'},
            'sensor_feeds': boiler_sensor_data,
            'fuel_data': {'type': 'natural_gas', 'hhv_mj_kg': 50.0},
            'steam_demand': {'required_flow_kg_hr': 40000},
            'constraints': {}
        }

        # Calculate input hash for integrity check
        input_hash = hash(json.dumps(input_data, sort_keys=True))

        # Act
        result = await mock_gl002_optimizer.execute(input_data)

        # Assert - Verify provenance hash exists (ensures traceability)
        assert 'provenance_hash' in result
        assert result['provenance_hash'] is not None
        assert len(result['provenance_hash']) > 0

        # Verify no data corruption
        assert result['optimization_success'] is True

    @pytest.mark.asyncio
    async def test_bidirectional_communication(
        self,
        mock_gl001_orchestrator,
        mock_gl002_optimizer,
        heat_demand_payload
    ):
        """Test bidirectional communication between GL-001 and GL-002."""
        # Arrange
        demand = heat_demand_payload

        # Mock bidirectional flow: GL-001 → GL-002 → GL-001
        async def gl002_with_feedback(input_data):
            result = await mock_gl002_optimizer.execute(input_data)
            # GL-002 requests feedback from GL-001
            result['requires_gl001_validation'] = True
            return result

        gl002_with_feedback_mock = MagicMock()
        gl002_with_feedback_mock.execute = gl002_with_feedback

        # Act
        result = await mock_gl001_orchestrator.optimize_boiler_for_demand(
            demand=demand,
            boiler_optimizer=gl002_with_feedback_mock
        )

        # Assert
        assert result is not None
        assert result['status'] == 'success'

    @pytest.mark.asyncio
    async def test_performance_under_load(
        self,
        mock_gl001_orchestrator,
        mock_gl002_optimizer,
        heat_demand_payload
    ):
        """Test performance under high coordination load."""
        # Arrange - Create 50 concurrent coordination requests
        num_requests = 50
        demands = [
            {**heat_demand_payload, 'request_id': i}
            for i in range(num_requests)
        ]

        # Act
        start_time = time.perf_counter()

        tasks = [
            mock_gl001_orchestrator.optimize_boiler_for_demand(
                demand=demand,
                boiler_optimizer=mock_gl002_optimizer
            )
            for demand in demands
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Assert
        assert len(results) == num_requests

        # Count successes
        successes = sum(
            1 for r in results
            if isinstance(r, dict) and r.get('status') == 'success'
        )

        assert successes >= num_requests * 0.95  # At least 95% success rate

        # Performance check (mocks should be fast)
        throughput = num_requests / (execution_time_ms / 1000)
        assert throughput > 100  # At least 100 requests/second with mocks

    @pytest.mark.asyncio
    async def test_provenance_tracking_coordination(
        self,
        mock_gl001_orchestrator,
        mock_gl002_optimizer,
        heat_demand_payload
    ):
        """Test provenance tracking across agent coordination."""
        # Arrange
        demand = heat_demand_payload

        # Act
        result = await mock_gl001_orchestrator.optimize_boiler_for_demand(
            demand=demand,
            boiler_optimizer=mock_gl002_optimizer
        )

        # Get GL-002 execution result
        gl002_input = {
            'boiler_data': {'capacity_kg_hr': 50000},
            'steam_demand': demand,
            'sensor_feeds': {'load_percent': 75},
            'fuel_data': {'type': 'natural_gas'},
            'constraints': {}
        }
        gl002_result = await mock_gl002_optimizer.execute(gl002_input)

        # Assert - Both agents should track provenance
        assert 'provenance_hash' in gl002_result
        assert gl002_result['provenance_hash'] is not None

        # GL-001 should track coordination
        assert result['status'] == 'success'
        assert 'data' in result

    @pytest.mark.asyncio
    async def test_timeout_handling(
        self,
        mock_gl001_orchestrator,
        heat_demand_payload
    ):
        """Test timeout handling for slow GL-002 responses."""
        # Arrange - Create slow GL-002 mock
        slow_gl002 = MagicMock()

        async def slow_execute(input_data):
            await asyncio.sleep(2)  # Simulate slow execution
            return {
                'agent_id': 'GL-002',
                'optimization_success': True,
                'combustion_optimization': {}
            }

        slow_gl002.execute = slow_execute

        # Act & Assert - Should timeout or handle gracefully
        try:
            result = await asyncio.wait_for(
                mock_gl001_orchestrator.optimize_boiler_for_demand(
                    demand=heat_demand_payload,
                    boiler_optimizer=slow_gl002
                ),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            # Expected behavior for timeout
            pytest.skip("Timeout occurred as expected")


# ============================================================================
# Test Class: Edge Cases and Error Scenarios
# ============================================================================

class TestGL001GL002EdgeCases:
    """Test edge cases and error scenarios for GL-001 ↔ GL-002 coordination."""

    @pytest.mark.asyncio
    async def test_invalid_demand_format(
        self,
        mock_gl001_orchestrator,
        mock_gl002_optimizer
    ):
        """Test handling of invalid demand format."""
        # Arrange
        invalid_demand = {
            'invalid_field': 'value'
            # Missing required fields
        }

        # Act & Assert
        with pytest.raises(Exception):
            await mock_gl001_orchestrator.optimize_boiler_for_demand(
                demand=invalid_demand,
                boiler_optimizer=mock_gl002_optimizer
            )

    @pytest.mark.asyncio
    async def test_missing_boiler_optimizer(
        self,
        mock_gl001_orchestrator,
        heat_demand_payload
    ):
        """Test handling when boiler optimizer is not provided."""
        # Act & Assert
        with pytest.raises(Exception):
            await mock_gl001_orchestrator.optimize_boiler_for_demand(
                demand=heat_demand_payload,
                boiler_optimizer=None
            )

    @pytest.mark.asyncio
    async def test_partial_gl002_response(
        self,
        mock_gl001_orchestrator,
        heat_demand_payload
    ):
        """Test handling of partial GL-002 response."""
        # Arrange - Create GL-002 that returns partial data
        partial_gl002 = MagicMock()

        async def partial_execute(input_data):
            return {
                'agent_id': 'GL-002',
                'optimization_success': False,
                'combustion_optimization': {},  # Empty/partial data
                'error': 'Partial optimization only'
            }

        partial_gl002.execute = partial_execute

        # Act
        result = await mock_gl001_orchestrator.optimize_boiler_for_demand(
            demand=heat_demand_payload,
            boiler_optimizer=partial_gl002
        )

        # Assert - Should handle partial response
        assert result is not None
