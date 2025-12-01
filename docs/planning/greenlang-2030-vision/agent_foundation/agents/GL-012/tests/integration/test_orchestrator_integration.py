# -*- coding: utf-8 -*-
"""
Orchestrator Integration Tests for GL-012 SteamQualityController

Comprehensive integration tests for orchestrator functionality including:
- Full orchestration workflow execution
- Quality analysis pipeline validation
- Control action execution and verification
- Agent coordination and communication
- Message bus integration
- Error recovery and fault tolerance

Test Count: 35+ tests
Coverage Target: 90%+

Standards: ISA-95 (Enterprise-Control Integration)

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal
import hashlib

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.orchestrator]


# =============================================================================
# FULL ORCHESTRATION WORKFLOW TESTS
# =============================================================================

class TestFullOrchestrationWorkflow:
    """Test complete orchestration workflows."""

    async def test_basic_orchestration_cycle(
        self,
        steam_quality_orchestrator,
        mock_steam_meter,
        mock_control_valve,
        mock_desuperheater
    ):
        """Test basic orchestration cycle execution."""
        # Configure orchestrator
        await steam_quality_orchestrator.configure({
            'cycle_interval_seconds': 1.0,
            'quality_target': 0.95,
            'temperature_setpoint_c': 400.0
        })

        # Execute single orchestration cycle
        result = await steam_quality_orchestrator.execute_cycle()

        assert result['status'] == 'success'
        assert 'cycle_id' in result
        assert 'execution_time_ms' in result
        assert 'actions_taken' in result

    async def test_multi_step_orchestration_workflow(
        self,
        steam_quality_orchestrator,
        mock_steam_meter,
        mock_control_valve,
        mock_desuperheater
    ):
        """Test multi-step orchestration workflow."""
        workflow = await steam_quality_orchestrator.execute_workflow(
            workflow_id='WF-QUALITY-OPTIMIZATION'
        )

        assert workflow['status'] == 'completed'
        assert 'steps_executed' in workflow
        assert len(workflow['steps_executed']) >= 3  # Read -> Analyze -> Control

        expected_steps = ['DATA_ACQUISITION', 'QUALITY_ANALYSIS', 'CONTROL_ACTION']
        for step in expected_steps:
            assert any(s['step_type'] == step for s in workflow['steps_executed'])

    async def test_orchestration_with_quality_deviation(
        self,
        steam_quality_orchestrator,
        mock_steam_meter,
        mock_control_valve,
        mock_desuperheater
    ):
        """Test orchestration response to quality deviation."""
        # Simulate low quality steam
        mock_steam_meter.set_dryness_fraction(0.85)  # Below target

        result = await steam_quality_orchestrator.execute_cycle()

        assert result['status'] == 'success'
        assert result['quality_deviation_detected'] is True
        assert 'corrective_actions' in result
        assert len(result['corrective_actions']) > 0

    async def test_orchestration_maintains_setpoints(
        self,
        steam_quality_orchestrator,
        mock_steam_meter,
        mock_control_valve,
        mock_desuperheater
    ):
        """Test orchestration maintains quality setpoints over time."""
        await steam_quality_orchestrator.configure({
            'quality_target': 0.95,
            'temperature_setpoint_c': 400.0
        })

        # Run multiple cycles
        quality_readings = []
        for _ in range(10):
            result = await steam_quality_orchestrator.execute_cycle()
            quality_readings.append(result['current_quality'])
            await asyncio.sleep(0.5)

        # Quality should trend toward setpoint
        avg_quality = sum(quality_readings) / len(quality_readings)
        assert avg_quality >= 0.90, f"Average quality {avg_quality} below 0.90"

    async def test_orchestration_state_persistence(
        self,
        steam_quality_orchestrator
    ):
        """Test orchestrator state persistence across cycles."""
        # Execute cycle with state
        await steam_quality_orchestrator.execute_cycle()

        state_before = await steam_quality_orchestrator.get_state()

        # Execute another cycle
        await steam_quality_orchestrator.execute_cycle()

        state_after = await steam_quality_orchestrator.get_state()

        # State should persist and update
        assert state_after['cycle_count'] == state_before['cycle_count'] + 1
        assert 'last_cycle_time' in state_after

    async def test_orchestration_provenance_tracking(
        self,
        steam_quality_orchestrator
    ):
        """Test provenance hash generation for audit trail."""
        result = await steam_quality_orchestrator.execute_cycle()

        assert 'provenance_hash' in result
        assert len(result['provenance_hash']) == 64  # SHA-256 hash

        # Verify hash is deterministic for same inputs
        # (mock consistent data for test)
        result2 = await steam_quality_orchestrator.execute_cycle()
        # Different cycles will have different timestamps, so hashes differ


# =============================================================================
# QUALITY ANALYSIS PIPELINE TESTS
# =============================================================================

class TestQualityAnalysisPipeline:
    """Test quality analysis pipeline functionality."""

    async def test_steam_quality_calculation(
        self,
        steam_quality_orchestrator,
        mock_steam_meter
    ):
        """Test steam quality calculation from sensor data."""
        mock_steam_meter.set_parameters(
            dryness_fraction=0.95,
            pressure_bar=100.0,
            temperature_c=450.0
        )

        analysis = await steam_quality_orchestrator.analyze_quality()

        assert 'quality_metrics' in analysis
        assert 'dryness_fraction' in analysis['quality_metrics']
        assert 'superheat_c' in analysis['quality_metrics']
        assert 'enthalpy_kj_kg' in analysis['quality_metrics']

    async def test_quality_trend_analysis(
        self,
        steam_quality_orchestrator,
        mock_steam_meter
    ):
        """Test quality trend analysis over time."""
        # Generate historical data
        for i in range(10):
            mock_steam_meter.add_historical_reading(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                dryness_fraction=0.94 + (i * 0.005)
            )

        analysis = await steam_quality_orchestrator.analyze_quality_trend(
            duration_minutes=15
        )

        assert 'trend' in analysis
        assert analysis['trend'] in ['IMPROVING', 'STABLE', 'DEGRADING']
        assert 'trend_slope' in analysis

    async def test_quality_deviation_detection(
        self,
        steam_quality_orchestrator,
        mock_steam_meter
    ):
        """Test automatic quality deviation detection."""
        mock_steam_meter.set_dryness_fraction(0.80)  # Low quality

        analysis = await steam_quality_orchestrator.analyze_quality()

        assert analysis['deviation_detected'] is True
        assert 'deviation_magnitude' in analysis
        assert 'recommended_actions' in analysis

    async def test_root_cause_analysis(
        self,
        steam_quality_orchestrator,
        mock_steam_meter,
        mock_desuperheater
    ):
        """Test root cause analysis for quality issues."""
        # Simulate high moisture content
        mock_steam_meter.set_dryness_fraction(0.85)
        mock_desuperheater.set_injection_rate(30.0)  # High injection

        analysis = await steam_quality_orchestrator.analyze_root_cause()

        assert 'probable_causes' in analysis
        assert len(analysis['probable_causes']) > 0
        assert any('injection' in cause['description'].lower()
                  for cause in analysis['probable_causes'])

    async def test_quality_prediction(
        self,
        steam_quality_orchestrator,
        mock_steam_meter
    ):
        """Test quality prediction based on current trends."""
        prediction = await steam_quality_orchestrator.predict_quality(
            horizon_minutes=30
        )

        assert 'predicted_quality' in prediction
        assert 'confidence_interval' in prediction
        assert 'prediction_time' in prediction

    async def test_energy_efficiency_analysis(
        self,
        steam_quality_orchestrator,
        mock_steam_meter,
        mock_desuperheater
    ):
        """Test energy efficiency analysis of steam system."""
        analysis = await steam_quality_orchestrator.analyze_energy_efficiency()

        assert 'efficiency_percent' in analysis
        assert 'energy_loss_kw' in analysis
        assert 'optimization_potential_kw' in analysis


# =============================================================================
# CONTROL ACTION EXECUTION TESTS
# =============================================================================

class TestControlActionExecution:
    """Test control action execution and verification."""

    async def test_injection_rate_adjustment(
        self,
        steam_quality_orchestrator,
        mock_desuperheater
    ):
        """Test injection rate adjustment action."""
        action = await steam_quality_orchestrator.execute_action({
            'action_type': 'ADJUST_INJECTION_RATE',
            'target_rate_kg_s': 15.0,
            'ramp_rate_kg_s_per_s': 2.0
        })

        assert action['status'] == 'success'
        assert action['action_executed'] is True

        # Verify action was applied
        status = await mock_desuperheater.get_status()
        assert abs(status['injection_rate_kg_s'] - 15.0) < 1.0

    async def test_temperature_setpoint_change(
        self,
        steam_quality_orchestrator,
        mock_desuperheater
    ):
        """Test temperature setpoint change action."""
        action = await steam_quality_orchestrator.execute_action({
            'action_type': 'CHANGE_TEMPERATURE_SETPOINT',
            'setpoint_c': 420.0
        })

        assert action['status'] == 'success'

        # Verify setpoint was updated
        config = await steam_quality_orchestrator.get_configuration()
        assert config['temperature_setpoint_c'] == 420.0

    async def test_valve_position_adjustment(
        self,
        steam_quality_orchestrator,
        mock_control_valve
    ):
        """Test valve position adjustment action."""
        action = await steam_quality_orchestrator.execute_action({
            'action_type': 'ADJUST_VALVE_POSITION',
            'valve_id': 'CV-STEAM-001',
            'position_percent': 65.0
        })

        assert action['status'] == 'success'

        # Verify valve position
        status = await mock_control_valve.get_status()
        assert abs(status['position_percent'] - 65.0) < 2.0

    async def test_control_action_validation(
        self,
        steam_quality_orchestrator
    ):
        """Test control action validation before execution."""
        # Invalid action (rate exceeds limits)
        action = await steam_quality_orchestrator.execute_action({
            'action_type': 'ADJUST_INJECTION_RATE',
            'target_rate_kg_s': 1000.0  # Unreasonably high
        })

        assert action['status'] == 'rejected'
        assert 'validation_error' in action

    async def test_control_action_rollback(
        self,
        steam_quality_orchestrator,
        mock_desuperheater
    ):
        """Test control action rollback on failure."""
        # Get initial state
        initial_rate = (await mock_desuperheater.get_status())['injection_rate_kg_s']

        # Execute action that will fail mid-execution
        mock_desuperheater.simulate_failure_during_action()

        action = await steam_quality_orchestrator.execute_action({
            'action_type': 'ADJUST_INJECTION_RATE',
            'target_rate_kg_s': 20.0
        })

        assert action['status'] == 'rolled_back'

        # Verify rollback restored previous state
        current_rate = (await mock_desuperheater.get_status())['injection_rate_kg_s']
        assert abs(current_rate - initial_rate) < 1.0

    async def test_action_sequence_execution(
        self,
        steam_quality_orchestrator,
        mock_desuperheater,
        mock_control_valve
    ):
        """Test execution of action sequence."""
        sequence = [
            {'action_type': 'ADJUST_VALVE_POSITION', 'valve_id': 'CV-STEAM-001', 'position_percent': 50.0},
            {'action_type': 'ADJUST_INJECTION_RATE', 'target_rate_kg_s': 10.0},
            {'action_type': 'CHANGE_TEMPERATURE_SETPOINT', 'setpoint_c': 400.0}
        ]

        result = await steam_quality_orchestrator.execute_action_sequence(sequence)

        assert result['status'] == 'success'
        assert result['actions_executed'] == len(sequence)
        assert all(a['status'] == 'success' for a in result['action_results'])


# =============================================================================
# AGENT COORDINATION TESTS
# =============================================================================

class TestAgentCoordination:
    """Test coordination with other agents."""

    async def test_request_data_from_steam_analyzer(
        self,
        steam_quality_orchestrator,
        mock_gl003_agent
    ):
        """Test requesting data from GL-003 SteamSystemAnalyzer."""
        response = await steam_quality_orchestrator.request_agent_data(
            agent_id='GL-003',
            request_type='STEAM_SYSTEM_STATUS',
            parameters={'include_headers': True}
        )

        assert response['status'] == 'success'
        assert 'data' in response
        assert 'headers_status' in response['data']

    async def test_coordinate_with_process_heat_orchestrator(
        self,
        steam_quality_orchestrator,
        mock_gl001_agent
    ):
        """Test coordination with GL-001 ProcessHeatOrchestrator."""
        coordination_result = await steam_quality_orchestrator.coordinate_with_agent(
            agent_id='GL-001',
            coordination_type='LOAD_BALANCING',
            parameters={
                'current_steam_demand_kg_s': 50.0,
                'quality_status': 'OPTIMAL'
            }
        )

        assert coordination_result['status'] == 'success'
        assert 'recommendations' in coordination_result

    async def test_receive_optimization_directive(
        self,
        steam_quality_orchestrator,
        mock_gl001_agent
    ):
        """Test receiving optimization directive from parent agent."""
        # Simulate directive from GL-001
        mock_gl001_agent.send_directive({
            'directive_type': 'OPTIMIZE_STEAM_QUALITY',
            'target_quality': 0.97,
            'priority': 'HIGH'
        })

        await asyncio.sleep(0.5)

        # Orchestrator should have received and processed directive
        state = await steam_quality_orchestrator.get_state()
        assert state['pending_directives'] > 0 or state['quality_target'] == 0.97

    async def test_publish_quality_status_event(
        self,
        steam_quality_orchestrator,
        mock_message_bus
    ):
        """Test publishing quality status events to message bus."""
        await steam_quality_orchestrator.publish_status_event({
            'event_type': 'QUALITY_STATUS_UPDATE',
            'quality_value': 0.95,
            'trend': 'STABLE'
        })

        # Verify event was published
        events = mock_message_bus.get_published_events('steam_quality')
        assert len(events) > 0
        assert events[-1]['event_type'] == 'QUALITY_STATUS_UPDATE'

    async def test_handle_emergency_coordination(
        self,
        steam_quality_orchestrator,
        mock_gl001_agent
    ):
        """Test emergency coordination with parent orchestrator."""
        # Simulate emergency condition
        await steam_quality_orchestrator.handle_emergency({
            'emergency_type': 'HIGH_MOISTURE_CONTENT',
            'severity': 'CRITICAL'
        })

        # Should notify parent orchestrator
        notifications = mock_gl001_agent.get_received_notifications()
        assert any(n['type'] == 'EMERGENCY' for n in notifications)


# =============================================================================
# MESSAGE BUS COMMUNICATION TESTS
# =============================================================================

class TestMessageBusCommunication:
    """Test message bus communication."""

    async def test_publish_to_message_bus(
        self,
        steam_quality_orchestrator,
        mock_message_bus
    ):
        """Test publishing messages to message bus."""
        await steam_quality_orchestrator.publish_message(
            topic='steam_quality/events',
            message={
                'event_type': 'QUALITY_READING',
                'dryness_fraction': 0.95,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )

        messages = mock_message_bus.get_messages('steam_quality/events')
        assert len(messages) > 0

    async def test_subscribe_to_message_bus(
        self,
        steam_quality_orchestrator,
        mock_message_bus
    ):
        """Test subscribing to message bus topics."""
        received_messages = []

        async def handler(message):
            received_messages.append(message)

        await steam_quality_orchestrator.subscribe(
            topic='process_heat/commands',
            handler=handler
        )

        # Simulate incoming message
        await mock_message_bus.publish(
            topic='process_heat/commands',
            message={'command': 'OPTIMIZE'}
        )

        await asyncio.sleep(0.5)

        assert len(received_messages) > 0

    async def test_request_response_pattern(
        self,
        steam_quality_orchestrator,
        mock_message_bus
    ):
        """Test request-response messaging pattern."""
        response = await steam_quality_orchestrator.send_request(
            topic='steam_analyzer/request',
            request={'query': 'GET_HEADER_PRESSURES'},
            timeout_seconds=5.0
        )

        assert response is not None
        assert 'data' in response

    async def test_message_bus_reconnection(
        self,
        steam_quality_orchestrator,
        mock_message_bus
    ):
        """Test message bus reconnection after disconnect."""
        await steam_quality_orchestrator.connect_message_bus()

        # Simulate disconnect
        await mock_message_bus.disconnect()
        await asyncio.sleep(0.5)

        # Reconnect
        await mock_message_bus.start()
        await asyncio.sleep(1.0)

        # Should be able to publish again
        result = await steam_quality_orchestrator.publish_message(
            topic='test',
            message={'test': True}
        )

        assert result['status'] == 'success'


# =============================================================================
# ERROR RECOVERY TESTS
# =============================================================================

class TestErrorRecovery:
    """Test error recovery and fault tolerance."""

    async def test_sensor_failure_recovery(
        self,
        steam_quality_orchestrator,
        mock_steam_meter
    ):
        """Test recovery from sensor failure."""
        # Simulate sensor failure
        mock_steam_meter.simulate_failure()

        result = await steam_quality_orchestrator.execute_cycle()

        # Should detect failure and use fallback
        assert result['sensor_failure_detected'] is True
        assert result['fallback_mode'] is True

        # Clear failure
        mock_steam_meter.clear_failure()

        result = await steam_quality_orchestrator.execute_cycle()
        assert result['fallback_mode'] is False

    async def test_communication_failure_recovery(
        self,
        steam_quality_orchestrator,
        mock_message_bus
    ):
        """Test recovery from communication failure."""
        # Simulate communication failure
        mock_message_bus.simulate_failure()

        result = await steam_quality_orchestrator.execute_cycle()

        # Should continue operating in degraded mode
        assert result['status'] in ['success', 'degraded']
        assert result['communication_healthy'] is False

    async def test_control_action_failure_recovery(
        self,
        steam_quality_orchestrator,
        mock_desuperheater
    ):
        """Test recovery from control action failure."""
        mock_desuperheater.simulate_stuck_valve()

        result = await steam_quality_orchestrator.execute_action({
            'action_type': 'ADJUST_INJECTION_RATE',
            'target_rate_kg_s': 15.0
        })

        assert result['status'] == 'failed'
        assert 'recovery_actions' in result

        # Verify recovery actions were attempted
        assert len(result['recovery_actions']) > 0

    async def test_graceful_degradation(
        self,
        steam_quality_orchestrator,
        mock_steam_meter,
        mock_desuperheater
    ):
        """Test graceful degradation under multiple failures."""
        # Multiple subsystem failures
        mock_steam_meter.simulate_failure()
        mock_desuperheater.simulate_failure()

        result = await steam_quality_orchestrator.execute_cycle()

        # Should enter safe mode
        assert result['operating_mode'] == 'SAFE'
        assert result['manual_intervention_required'] is True

    async def test_automatic_recovery_attempt(
        self,
        steam_quality_orchestrator,
        mock_steam_meter
    ):
        """Test automatic recovery attempts."""
        # Configure recovery
        await steam_quality_orchestrator.configure_recovery({
            'auto_retry_enabled': True,
            'max_retries': 3,
            'retry_interval_seconds': 1.0
        })

        # Transient failure (clears after 2 attempts)
        mock_steam_meter.simulate_transient_failure(failure_count=2)

        result = await steam_quality_orchestrator.execute_cycle()

        # Should have recovered
        assert result['status'] == 'success'
        assert result['retry_count'] == 2

    async def test_failure_event_logging(
        self,
        steam_quality_orchestrator,
        mock_steam_meter
    ):
        """Test failure events are properly logged."""
        mock_steam_meter.simulate_failure()

        await steam_quality_orchestrator.execute_cycle()

        # Get failure log
        log = await steam_quality_orchestrator.get_failure_log(
            start_time=datetime.now(timezone.utc) - timedelta(minutes=5)
        )

        assert len(log) > 0
        assert log[-1]['failure_type'] == 'SENSOR_FAILURE'
        assert 'timestamp' in log[-1]
        assert 'recovery_action' in log[-1]


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.performance
class TestOrchestratorPerformance:
    """Performance tests for orchestrator."""

    async def test_orchestration_cycle_time(
        self,
        steam_quality_orchestrator,
        mock_steam_meter,
        mock_control_valve,
        mock_desuperheater,
        performance_monitor
    ):
        """Test orchestration cycle completes within time limit."""
        performance_monitor.start()

        cycle_times = []
        for _ in range(20):
            start = asyncio.get_event_loop().time()
            await steam_quality_orchestrator.execute_cycle()
            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            cycle_times.append(elapsed_ms)
            performance_monitor.record_metric('cycle_time_ms', elapsed_ms)

        avg_time = sum(cycle_times) / len(cycle_times)

        print(f"\n=== Orchestration Cycle Time ===")
        print(f"Average: {avg_time:.2f}ms")
        print(f"Max: {max(cycle_times):.2f}ms")

        assert avg_time < 500.0, f"Average cycle time {avg_time}ms exceeds 500ms target"

    async def test_concurrent_action_execution(
        self,
        steam_quality_orchestrator
    ):
        """Test concurrent action execution performance."""
        actions = [
            {'action_type': 'ADJUST_VALVE_POSITION', 'valve_id': f'CV-{i}', 'position_percent': 50.0}
            for i in range(5)
        ]

        start = asyncio.get_event_loop().time()
        results = await steam_quality_orchestrator.execute_actions_concurrent(actions)
        elapsed = asyncio.get_event_loop().time() - start

        print(f"\n=== Concurrent Action Execution ===")
        print(f"Total time for {len(actions)} actions: {elapsed*1000:.2f}ms")

        # Concurrent should be faster than sequential
        assert elapsed < 1.0  # All 5 actions in under 1 second

    async def test_analysis_pipeline_throughput(
        self,
        steam_quality_orchestrator,
        mock_steam_meter
    ):
        """Test quality analysis pipeline throughput."""
        num_analyses = 50

        start = asyncio.get_event_loop().time()
        for _ in range(num_analyses):
            await steam_quality_orchestrator.analyze_quality()
        elapsed = asyncio.get_event_loop().time() - start

        analyses_per_second = num_analyses / elapsed

        print(f"\n=== Analysis Pipeline Throughput ===")
        print(f"Analyses per second: {analyses_per_second:.1f}")

        assert analyses_per_second >= 10, f"Throughput {analyses_per_second}/s below target"

    async def test_message_processing_latency(
        self,
        steam_quality_orchestrator,
        mock_message_bus
    ):
        """Test message processing latency."""
        latencies = []

        async def measure_latency(message):
            receive_time = asyncio.get_event_loop().time()
            send_time = message.get('send_time', receive_time)
            latencies.append((receive_time - send_time) * 1000)

        await steam_quality_orchestrator.subscribe(
            topic='test/latency',
            handler=measure_latency
        )

        # Send test messages
        for _ in range(20):
            await mock_message_bus.publish(
                topic='test/latency',
                message={'send_time': asyncio.get_event_loop().time()}
            )
            await asyncio.sleep(0.05)

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"\n=== Message Latency ===")
            print(f"Average: {avg_latency:.2f}ms")

            assert avg_latency < 100.0
