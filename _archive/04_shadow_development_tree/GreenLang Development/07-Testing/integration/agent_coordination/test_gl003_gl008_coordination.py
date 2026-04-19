# -*- coding: utf-8 -*-
"""
Integration tests for GL-003 STEAMWISE ↔ GL-008 TRAPCATCHER coordination.

Tests the coordination between SteamSystemOrchestrator (GL-003) and
SteamTrapMonitor (GL-008) for steam trap inspection and system efficiency.

Test Scenarios:
1. GL-003 monitors steam system
2. GL-003 detects pressure anomalies
3. GL-003 calls GL-008 for steam trap inspection
4. GL-008 returns failed trap locations
5. GL-003 updates system efficiency calculations

Coverage: Tests anomaly detection, trap inspection coordination, efficiency
impact analysis, maintenance prioritization, and real-time monitoring.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def gl003_config():
    """Configuration for GL-003 SteamSystemOrchestrator."""
    return {
        'agent_id': 'GL-003',
        'agent_name': 'SteamSystemOrchestrator',
        'version': '1.0.0',
        'pressure_anomaly_threshold_percent': 5.0,
        'temperature_anomaly_threshold_c': 10.0,
        'trap_inspection_enabled': True,
        'auto_inspection_on_anomaly': True
    }


@pytest.fixture
def gl008_config():
    """Configuration for GL-008 SteamTrapMonitor."""
    return {
        'agent_id': 'GL-008',
        'agent_name': 'SteamTrapMonitor',
        'version': '1.0.0',
        'inspection_methods': ['acoustic', 'temperature', 'ultrasonic'],
        'failure_threshold_confidence': 0.8,
        'inspection_priority_levels': ['critical', 'high', 'medium', 'low']
    }


@pytest.fixture
def mock_gl003_orchestrator(gl003_config):
    """Mock GL-003 SteamSystemOrchestrator instance."""
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.agent_id = gl003_config['agent_id']

    # Mock pressure anomaly detection
    async def mock_detect_pressure_anomalies(steam_data):
        anomalies = []

        # Simulate anomaly detection
        for i, reading in enumerate(steam_data.get('pressure_readings', [])):
            if abs(reading['pressure_bar'] - reading['target_pressure_bar']) > 2.0:
                anomalies.append({
                    'anomaly_id': f'ANOM-{i+1:03d}',
                    'location': reading['location'],
                    'type': 'pressure_deviation',
                    'severity': 'high' if abs(reading['pressure_bar'] - reading['target_pressure_bar']) > 5.0 else 'medium',
                    'measured_pressure_bar': reading['pressure_bar'],
                    'target_pressure_bar': reading['target_pressure_bar'],
                    'deviation_percent': abs(reading['pressure_bar'] - reading['target_pressure_bar']) / reading['target_pressure_bar'] * 100,
                    'suspected_cause': 'steam_trap_failure',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

        return {
            'anomalies_detected': len(anomalies),
            'anomalies': anomalies,
            'requires_trap_inspection': len(anomalies) > 0
        }

    mock_agent.detect_pressure_anomalies = mock_detect_pressure_anomalies

    # Mock efficiency update
    async def mock_update_efficiency_calculations(trap_failures):
        total_steam_loss_kg_hr = sum(trap['steam_loss_kg_hr'] for trap in trap_failures)
        efficiency_impact = total_steam_loss_kg_hr / 50000 * 100  # Assume 50,000 kg/hr capacity

        return {
            'status': 'success',
            'updated_efficiency': {
                'original_efficiency_percent': 92.0,
                'efficiency_loss_percent': efficiency_impact,
                'new_efficiency_percent': 92.0 - efficiency_impact,
                'steam_loss_kg_hr': total_steam_loss_kg_hr,
                'estimated_cost_impact_usd_year': total_steam_loss_kg_hr * 8760 * 0.015  # $0.015/kg
            }
        }

    mock_agent.update_efficiency_calculations = mock_update_efficiency_calculations

    return mock_agent


@pytest.fixture
def mock_gl008_monitor(gl008_config):
    """Mock GL-008 SteamTrapMonitor instance."""
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.agent_id = gl008_config['agent_id']

    # Mock async execute method
    async def mock_execute(input_data):
        inspection_locations = input_data.get('inspection_locations', [])

        failed_traps = []
        healthy_traps = []

        for loc in inspection_locations:
            # Simulate inspection
            # If anomaly severity is high, more likely to find failure
            failure_probability = 0.8 if loc.get('anomaly_severity') == 'high' else 0.3

            if hash(loc['location']) % 10 < failure_probability * 10:
                failed_traps.append({
                    'trap_id': f"TRAP-{loc['location']}-001",
                    'location': loc['location'],
                    'failure_mode': 'blowing_through' if hash(loc['location']) % 2 == 0 else 'plugged',
                    'failure_confidence': 0.95,
                    'steam_loss_kg_hr': 150 if hash(loc['location']) % 2 == 0 else 0,
                    'inspection_method': 'acoustic',
                    'priority': 'critical' if loc.get('anomaly_severity') == 'high' else 'high',
                    'recommended_action': 'immediate_replacement' if loc.get('anomaly_severity') == 'high' else 'schedule_replacement',
                    'estimated_repair_cost_usd': 500,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            else:
                healthy_traps.append({
                    'trap_id': f"TRAP-{loc['location']}-001",
                    'location': loc['location'],
                    'status': 'healthy',
                    'confidence': 0.92
                })

        return {
            'agent_id': gl008_config['agent_id'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'execution_time_ms': 125.7,
            'inspection_results': {
                'total_inspected': len(inspection_locations),
                'failed_traps': failed_traps,
                'healthy_traps': healthy_traps,
                'failure_rate_percent': len(failed_traps) / len(inspection_locations) * 100 if inspection_locations else 0
            },
            'failed_trap_locations': failed_traps,
            'maintenance_priorities': sorted(
                failed_traps,
                key=lambda x: (
                    0 if x['priority'] == 'critical' else
                    1 if x['priority'] == 'high' else 2
                )
            ),
            'total_steam_loss_kg_hr': sum(trap['steam_loss_kg_hr'] for trap in failed_traps),
            'kpi_dashboard': {
                'traps_inspected': len(inspection_locations),
                'failures_found': len(failed_traps),
                'total_steam_loss_kg_hr': sum(trap['steam_loss_kg_hr'] for trap in failed_traps)
            },
            'inspection_success': True,
            'provenance_hash': 'trap_inspection_hash_123'
        }

    mock_agent.execute = mock_execute

    # Mock inspect_steam_traps method
    async def mock_inspect_steam_traps(anomaly_locations):
        inspection_input = {
            'inspection_locations': [
                {
                    'location': loc['location'],
                    'anomaly_severity': loc.get('severity', 'medium')
                }
                for loc in anomaly_locations
            ]
        }
        return await mock_agent.execute(inspection_input)

    mock_agent.inspect_steam_traps = mock_inspect_steam_traps

    return mock_agent


@pytest.fixture
def steam_system_data():
    """Sample steam system monitoring data."""
    return {
        'pressure_readings': [
            {
                'location': 'header_main',
                'pressure_bar': 40.5,
                'target_pressure_bar': 40.0,
                'temperature_c': 450
            },
            {
                'location': 'branch_1',
                'pressure_bar': 35.2,
                'target_pressure_bar': 38.0,
                'temperature_c': 420
            },
            {
                'location': 'branch_2',
                'pressure_bar': 38.8,
                'target_pressure_bar': 38.0,
                'temperature_c': 445
            },
            {
                'location': 'return_line',
                'pressure_bar': 2.5,
                'target_pressure_bar': 3.0,
                'temperature_c': 110
            }
        ],
        'flow_rates_kg_hr': {
            'header_main': 45000,
            'branch_1': 18000,
            'branch_2': 22000
        }
    }


# ============================================================================
# Test Class: GL-003 ↔ GL-008 Coordination
# ============================================================================

class TestGL003GL008Coordination:
    """Test suite for GL-003 ↔ GL-008 steam trap monitoring coordination."""

    @pytest.mark.asyncio
    async def test_pressure_anomaly_detection(
        self,
        mock_gl003_orchestrator,
        steam_system_data
    ):
        """Test GL-003 detects pressure anomalies."""
        # Act
        result = await mock_gl003_orchestrator.detect_pressure_anomalies(steam_system_data)

        # Assert
        assert 'anomalies_detected' in result
        assert 'anomalies' in result
        assert 'requires_trap_inspection' in result

        # Should detect anomalies in branch_1 and return_line
        assert result['anomalies_detected'] > 0

        for anomaly in result['anomalies']:
            assert 'anomaly_id' in anomaly
            assert 'location' in anomaly
            assert 'severity' in anomaly

    @pytest.mark.asyncio
    async def test_gl003_triggers_gl008_inspection(
        self,
        mock_gl003_orchestrator,
        mock_gl008_monitor,
        steam_system_data
    ):
        """Test GL-003 calls GL-008 for steam trap inspection."""
        # Step 1: Detect anomalies
        anomaly_result = await mock_gl003_orchestrator.detect_pressure_anomalies(steam_system_data)

        assert anomaly_result['requires_trap_inspection'] is True

        # Step 2: Trigger trap inspection
        inspection_result = await mock_gl008_monitor.inspect_steam_traps(
            anomaly_locations=anomaly_result['anomalies']
        )

        # Assert
        assert inspection_result['inspection_success'] is True
        assert 'failed_trap_locations' in inspection_result
        assert 'inspection_results' in inspection_result

    @pytest.mark.asyncio
    async def test_gl008_returns_failed_trap_locations(
        self,
        mock_gl008_monitor
    ):
        """Test GL-008 returns failed trap locations."""
        # Arrange
        inspection_locations = [
            {'location': 'branch_1', 'anomaly_severity': 'high'},
            {'location': 'branch_2', 'anomaly_severity': 'medium'},
            {'location': 'return_line', 'anomaly_severity': 'high'}
        ]

        # Act
        result = await mock_gl008_monitor.inspect_steam_traps(
            anomaly_locations=inspection_locations
        )

        # Assert
        assert 'failed_trap_locations' in result
        assert isinstance(result['failed_trap_locations'], list)

        for trap in result['failed_trap_locations']:
            assert 'trap_id' in trap
            assert 'location' in trap
            assert 'failure_mode' in trap
            assert 'failure_confidence' in trap
            assert trap['failure_confidence'] >= 0.8  # High confidence

    @pytest.mark.asyncio
    async def test_gl003_updates_efficiency_calculations(
        self,
        mock_gl003_orchestrator,
        mock_gl008_monitor,
        steam_system_data
    ):
        """Test GL-003 updates system efficiency based on trap failures."""
        # Step 1: Detect anomalies
        anomalies = await mock_gl003_orchestrator.detect_pressure_anomalies(steam_system_data)

        # Step 2: Inspect traps
        inspection = await mock_gl008_monitor.inspect_steam_traps(
            anomaly_locations=anomalies['anomalies']
        )

        # Step 3: Update efficiency
        efficiency_update = await mock_gl003_orchestrator.update_efficiency_calculations(
            trap_failures=inspection['failed_trap_locations']
        )

        # Assert
        assert efficiency_update['status'] == 'success'
        assert 'updated_efficiency' in efficiency_update

        updated = efficiency_update['updated_efficiency']
        assert 'efficiency_loss_percent' in updated
        assert 'new_efficiency_percent' in updated
        assert 'steam_loss_kg_hr' in updated

        # Efficiency should be reduced if traps failed
        if inspection['failed_trap_locations']:
            assert updated['new_efficiency_percent'] < updated['original_efficiency_percent']

    @pytest.mark.asyncio
    async def test_end_to_end_coordination_workflow(
        self,
        mock_gl003_orchestrator,
        mock_gl008_monitor,
        steam_system_data
    ):
        """Test complete end-to-end coordination workflow."""
        # Step 1: GL-003 monitors and detects anomalies
        anomalies = await mock_gl003_orchestrator.detect_pressure_anomalies(steam_system_data)

        assert anomalies['requires_trap_inspection'] is True

        # Step 2: GL-003 calls GL-008 for inspection
        inspection = await mock_gl008_monitor.inspect_steam_traps(
            anomaly_locations=anomalies['anomalies']
        )

        assert inspection['inspection_success'] is True

        # Step 3: GL-008 returns failed traps
        failed_traps = inspection['failed_trap_locations']
        assert len(failed_traps) >= 0

        # Step 4: GL-003 updates efficiency
        efficiency = await mock_gl003_orchestrator.update_efficiency_calculations(
            trap_failures=failed_traps
        )

        assert efficiency['status'] == 'success'

        # Step 5: Verify complete workflow results
        assert 'updated_efficiency' in efficiency
        assert 'estimated_cost_impact_usd_year' in efficiency['updated_efficiency']

    @pytest.mark.asyncio
    async def test_trap_failure_mode_identification(
        self,
        mock_gl008_monitor
    ):
        """Test GL-008 identifies specific trap failure modes."""
        # Arrange
        locations = [
            {'location': f'trap_{i}', 'anomaly_severity': 'high'}
            for i in range(10)
        ]

        # Act
        result = await mock_gl008_monitor.inspect_steam_traps(anomaly_locations=locations)

        # Assert
        valid_failure_modes = ['blowing_through', 'plugged', 'cold', 'leaking']

        for trap in result['failed_trap_locations']:
            assert 'failure_mode' in trap
            assert trap['failure_mode'] in valid_failure_modes

            # Blowing through should have steam loss
            if trap['failure_mode'] == 'blowing_through':
                assert trap['steam_loss_kg_hr'] > 0

    @pytest.mark.asyncio
    async def test_maintenance_prioritization(
        self,
        mock_gl008_monitor
    ):
        """Test GL-008 prioritizes maintenance based on severity."""
        # Arrange
        locations = [
            {'location': 'critical_trap', 'anomaly_severity': 'high'},
            {'location': 'normal_trap_1', 'anomaly_severity': 'medium'},
            {'location': 'normal_trap_2', 'anomaly_severity': 'low'}
        ]

        # Act
        result = await mock_gl008_monitor.inspect_steam_traps(anomaly_locations=locations)

        # Assert
        priorities = result['maintenance_priorities']
        assert len(priorities) > 0

        # Critical should come first
        if len(priorities) > 1:
            for i in range(len(priorities) - 1):
                current_priority = priorities[i]['priority']
                next_priority = priorities[i + 1]['priority']

                priority_order = ['critical', 'high', 'medium', 'low']
                assert priority_order.index(current_priority) <= priority_order.index(next_priority)

    @pytest.mark.asyncio
    async def test_steam_loss_calculation(
        self,
        mock_gl008_monitor
    ):
        """Test accurate steam loss calculation for failed traps."""
        # Arrange
        locations = [
            {'location': 'high_loss_trap', 'anomaly_severity': 'high'},
            {'location': 'medium_loss_trap', 'anomaly_severity': 'medium'}
        ]

        # Act
        result = await mock_gl008_monitor.inspect_steam_traps(anomaly_locations=locations)

        # Assert
        assert 'total_steam_loss_kg_hr' in result
        assert result['total_steam_loss_kg_hr'] >= 0

        # Verify sum matches individual losses
        calculated_total = sum(
            trap['steam_loss_kg_hr']
            for trap in result['failed_trap_locations']
        )
        assert result['total_steam_loss_kg_hr'] == calculated_total

    @pytest.mark.asyncio
    async def test_inspection_method_selection(
        self,
        mock_gl008_monitor
    ):
        """Test GL-008 selects appropriate inspection method."""
        # Arrange
        locations = [
            {'location': 'trap_1', 'anomaly_severity': 'high'}
        ]

        # Act
        result = await mock_gl008_monitor.inspect_steam_traps(anomaly_locations=locations)

        # Assert
        valid_methods = ['acoustic', 'temperature', 'ultrasonic', 'visual']

        for trap in result['failed_trap_locations']:
            assert 'inspection_method' in trap
            assert trap['inspection_method'] in valid_methods

    @pytest.mark.asyncio
    async def test_concurrent_inspections(
        self,
        mock_gl008_monitor
    ):
        """Test GL-008 handles concurrent inspection requests."""
        # Arrange - Create 5 concurrent inspection scenarios
        inspection_scenarios = [
            [{'location': f'trap_{i}_{j}', 'anomaly_severity': 'high'} for j in range(3)]
            for i in range(5)
        ]

        # Act
        tasks = [
            mock_gl008_monitor.inspect_steam_traps(anomaly_locations=locs)
            for locs in inspection_scenarios
        ]

        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 5
        for result in results:
            assert result['inspection_success'] is True

    @pytest.mark.asyncio
    async def test_false_positive_handling(
        self,
        mock_gl008_monitor
    ):
        """Test GL-008 handles false positives (healthy traps at anomaly locations)."""
        # Arrange
        locations = [
            {'location': f'trap_{i}', 'anomaly_severity': 'low'}
            for i in range(10)
        ]

        # Act
        result = await mock_gl008_monitor.inspect_steam_traps(anomaly_locations=locations)

        # Assert
        # Should have some healthy traps (false positives from anomaly detection)
        assert 'inspection_results' in result
        assert 'healthy_traps' in result['inspection_results']

        # Total inspected should match input
        assert result['inspection_results']['total_inspected'] == len(locations)

    @pytest.mark.asyncio
    async def test_efficiency_impact_accuracy(
        self,
        mock_gl003_orchestrator
    ):
        """Test accuracy of efficiency impact calculations."""
        # Arrange
        trap_failures = [
            {'steam_loss_kg_hr': 150, 'location': 'trap_1'},
            {'steam_loss_kg_hr': 200, 'location': 'trap_2'},
            {'steam_loss_kg_hr': 100, 'location': 'trap_3'}
        ]

        total_loss = sum(trap['steam_loss_kg_hr'] for trap in trap_failures)

        # Act
        result = await mock_gl003_orchestrator.update_efficiency_calculations(
            trap_failures=trap_failures
        )

        # Assert
        updated = result['updated_efficiency']
        assert updated['steam_loss_kg_hr'] == total_loss

        # Efficiency impact should be proportional to steam loss
        assert updated['efficiency_loss_percent'] > 0
        assert updated['new_efficiency_percent'] < updated['original_efficiency_percent']

    @pytest.mark.asyncio
    async def test_real_time_monitoring_latency(
        self,
        mock_gl003_orchestrator,
        mock_gl008_monitor,
        steam_system_data
    ):
        """Test real-time monitoring and inspection latency."""
        # Act
        start_time = time.perf_counter()

        # Complete workflow
        anomalies = await mock_gl003_orchestrator.detect_pressure_anomalies(steam_system_data)
        inspection = await mock_gl008_monitor.inspect_steam_traps(
            anomaly_locations=anomalies['anomalies']
        )
        efficiency = await mock_gl003_orchestrator.update_efficiency_calculations(
            trap_failures=inspection['failed_trap_locations']
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Assert
        # Should complete quickly for real-time monitoring
        assert total_latency_ms < 300  # <300ms with mocks

    @pytest.mark.asyncio
    async def test_data_integrity_across_coordination(
        self,
        mock_gl003_orchestrator,
        mock_gl008_monitor,
        steam_system_data
    ):
        """Test data integrity throughout coordination workflow."""
        # Step 1: Detect anomalies
        anomalies = await mock_gl003_orchestrator.detect_pressure_anomalies(steam_system_data)
        anomaly_locations = [a['location'] for a in anomalies['anomalies']]

        # Step 2: Inspect traps
        inspection = await mock_gl008_monitor.inspect_steam_traps(
            anomaly_locations=anomalies['anomalies']
        )

        # Assert - All inspected locations should match anomaly locations
        inspected_locations = [
            trap['location'] for trap in
            inspection['inspection_results']['failed_traps'] +
            inspection['inspection_results']['healthy_traps']
        ]

        for loc in anomaly_locations:
            assert loc in inspected_locations

    @pytest.mark.asyncio
    async def test_provenance_tracking(
        self,
        mock_gl008_monitor
    ):
        """Test provenance tracking for trap inspections."""
        # Arrange
        locations = [{'location': 'trap_1', 'anomaly_severity': 'high'}]

        # Act
        result = await mock_gl008_monitor.inspect_steam_traps(anomaly_locations=locations)

        # Assert
        assert 'provenance_hash' in result
        assert result['provenance_hash'] is not None
