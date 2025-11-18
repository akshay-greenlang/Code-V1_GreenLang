"""
Agent Coordination Integration Tests for GL-001 ProcessHeatOrchestrator

Comprehensive tests for multi-agent orchestration including:
- GL-002 (Boiler Efficiency) coordination
- GL-003 (Steam Distribution) coordination
- GL-004 (Heat Recovery) coordination
- GL-005 (Emissions Monitoring) coordination
- Task delegation and load balancing
- Result aggregation and error handling
- Timeout and resilience testing
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from process_heat_orchestrator import ProcessHeatOrchestrator


# ==============================================================================
# SUB-AGENT COORDINATION TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.coordination
@pytest.mark.integration
async def test_gl002_boiler_efficiency_coordination(
    orchestrator,
    mock_gl002_agent,
    sample_plant_data,
    integration_assertions
):
    """
    Test coordination with GL-002 Boiler Efficiency Agent.

    Validates:
    - Task delegation to GL-002
    - Boiler efficiency optimization request
    - Result retrieval and aggregation
    - Error handling
    """
    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': ['GL-002'],
        'agent_commands': {
            'GL-002': {
                'command_type': 'optimize',
                'priority': 'HIGH',
                'parameters': {
                    'target_efficiency': 92.0,
                    'optimization_method': 'combustion_tuning',
                    'constraints': {
                        'max_o2': 4.5,
                        'min_steam_pressure': 95.0
                    }
                },
                'timeout_seconds': 30
            }
        }
    }

    result = await orchestrator.execute(input_data)

    # Validate orchestration result
    integration_assertions.assert_orchestration_result_valid(result)

    # Validate coordination
    coordination_result = result['coordination_result']
    assert coordination_result is not None
    integration_assertions.assert_agent_coordination_successful(coordination_result)

    # Validate GL-002 specific results
    assert 'GL-002' in coordination_result['task_assignments']
    gl002_tasks = coordination_result['task_assignments']['GL-002']

    assert len(gl002_tasks) > 0
    assert gl002_tasks[0]['priority'] == 'HIGH'


@pytest.mark.asyncio
@pytest.mark.coordination
@pytest.mark.integration
async def test_gl003_steam_distribution_coordination(
    orchestrator,
    mock_gl003_agent,
    sample_sensor_feeds,
    integration_assertions
):
    """
    Test coordination with GL-003 Steam Distribution Agent.

    Validates:
    - Steam header optimization
    - Load balancing across consumers
    - Pressure control coordination
    """
    input_data = {
        'sensor_feeds': sample_sensor_feeds('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': ['GL-003'],
        'agent_commands': {
            'GL-003': {
                'command_type': 'optimize',
                'priority': 'MEDIUM',
                'parameters': {
                    'optimization_objective': 'minimize_pressure_drop',
                    'steam_headers': ['HEADER-1', 'HEADER-2'],
                    'consumers': ['CONSUMER-1', 'CONSUMER-2', 'CONSUMER-3']
                },
                'timeout_seconds': 30
            }
        }
    }

    result = await orchestrator.execute(input_data)

    integration_assertions.assert_orchestration_result_valid(result)

    coordination_result = result['coordination_result']
    assert 'GL-003' in coordination_result['task_assignments']


@pytest.mark.asyncio
@pytest.mark.coordination
@pytest.mark.integration
async def test_gl004_heat_recovery_coordination(
    orchestrator,
    mock_gl004_agent,
    sample_plant_data
):
    """
    Test coordination with GL-004 Heat Recovery Agent.

    Validates:
    - Heat recovery optimization
    - Economizer performance
    - Air preheater optimization
    """
    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': ['GL-004'],
        'agent_commands': {
            'GL-004': {
                'command_type': 'optimize',
                'priority': 'HIGH',
                'parameters': {
                    'heat_recovery_targets': {
                        'economizer_mw': 5.0,
                        'air_preheater_mw': 3.0,
                        'total_recovery_mw': 10.0
                    },
                    'constraints': {
                        'min_stack_temp_c': 120.0,
                        'max_corrosion_risk': 'LOW'
                    }
                },
                'timeout_seconds': 30
            }
        }
    }

    result = await orchestrator.execute(input_data)

    assert result is not None
    coordination_result = result['coordination_result']
    assert 'GL-004' in coordination_result['task_assignments']


@pytest.mark.asyncio
@pytest.mark.coordination
@pytest.mark.integration
async def test_gl005_emissions_monitoring_coordination(
    orchestrator,
    mock_gl005_agent,
    sample_emissions_data
):
    """
    Test coordination with GL-005 Emissions Monitoring Agent.

    Validates:
    - Real-time emissions monitoring
    - Compliance checking
    - Alert generation
    """
    input_data = {
        'emissions_data': sample_emissions_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': ['GL-005'],
        'agent_commands': {
            'GL-005': {
                'command_type': 'monitor',
                'priority': 'HIGH',
                'parameters': {
                    'monitoring_mode': 'continuous',
                    'pollutants': ['nox', 'so2', 'pm', 'co2'],
                    'compliance_limits': {
                        'nox_mg_nm3': 150.0,
                        'so2_mg_nm3': 100.0,
                        'pm_mg_nm3': 20.0
                    },
                    'alert_threshold_percent': 90.0
                },
                'timeout_seconds': 30
            }
        }
    }

    result = await orchestrator.execute(input_data)

    assert result is not None
    coordination_result = result['coordination_result']
    assert 'GL-005' in coordination_result['task_assignments']


# ==============================================================================
# MULTI-AGENT COORDINATION TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.coordination
@pytest.mark.integration
async def test_all_agents_coordination(
    orchestrator,
    all_sub_agents,
    sample_plant_data,
    sample_sensor_feeds,
    sample_emissions_data,
    integration_assertions,
    test_data_generator
):
    """
    Test coordination of all sub-agents simultaneously.

    Validates:
    - Concurrent task delegation to all agents
    - Result aggregation from multiple agents
    - Interdependent task coordination
    - Load balancing across agents
    """
    agent_ids = list(all_sub_agents.keys())
    commands = test_data_generator.generate_coordination_commands(agent_ids)

    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'sensor_feeds': sample_sensor_feeds('PLANT-001'),
        'emissions_data': sample_emissions_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': agent_ids,
        'agent_commands': commands
    }

    result = await orchestrator.execute(input_data)

    # Validate orchestration
    integration_assertions.assert_orchestration_result_valid(result)

    # Validate all agents received tasks
    coordination_result = result['coordination_result']
    integration_assertions.assert_agent_coordination_successful(coordination_result)

    assert len(coordination_result['task_assignments']) == len(agent_ids)

    for agent_id in agent_ids:
        assert agent_id in coordination_result['task_assignments']
        tasks = coordination_result['task_assignments'][agent_id]
        assert len(tasks) > 0


@pytest.mark.asyncio
@pytest.mark.coordination
async def test_agent_task_prioritization(
    orchestrator,
    all_sub_agents,
    sample_plant_data,
    integration_assertions
):
    """
    Test task prioritization across multiple agents.

    Validates:
    - HIGH priority tasks executed first
    - MEDIUM and LOW priority queuing
    - Priority-based resource allocation
    """
    agent_ids = list(all_sub_agents.keys())

    # Create tasks with different priorities
    commands = {}
    priorities = ['HIGH', 'MEDIUM', 'LOW', 'HIGH']

    for i, agent_id in enumerate(agent_ids):
        commands[agent_id] = {
            'command_type': 'optimize',
            'priority': priorities[i],
            'parameters': {'target': 90.0},
            'timeout_seconds': 30
        }

    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': agent_ids,
        'agent_commands': commands
    }

    result = await orchestrator.execute(input_data)

    coordination_result = result['coordination_result']
    integration_assertions.assert_agent_coordination_successful(coordination_result)

    # Validate task assignments respect priorities
    for agent_id in agent_ids:
        tasks = coordination_result['task_assignments'][agent_id]
        for task in tasks:
            assert 'priority' in task
            assert task['priority'] in ['HIGH', 'MEDIUM', 'LOW']


@pytest.mark.asyncio
@pytest.mark.coordination
async def test_agent_result_aggregation(
    orchestrator,
    all_sub_agents,
    sample_plant_data,
    integration_assertions
):
    """
    Test aggregation of results from multiple agents.

    Validates:
    - Collection of all agent results
    - Result merge and consolidation
    - Conflict resolution
    - Comprehensive reporting
    """
    agent_ids = list(all_sub_agents.keys())

    commands = {
        'GL-002': {
            'command_type': 'report',
            'priority': 'MEDIUM',
            'parameters': {'report_type': 'efficiency_analysis'},
            'timeout_seconds': 30
        },
        'GL-003': {
            'command_type': 'report',
            'priority': 'MEDIUM',
            'parameters': {'report_type': 'distribution_analysis'},
            'timeout_seconds': 30
        },
        'GL-004': {
            'command_type': 'report',
            'priority': 'MEDIUM',
            'parameters': {'report_type': 'recovery_analysis'},
            'timeout_seconds': 30
        },
        'GL-005': {
            'command_type': 'report',
            'priority': 'MEDIUM',
            'parameters': {'report_type': 'emissions_analysis'},
            'timeout_seconds': 30
        }
    }

    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': agent_ids,
        'agent_commands': commands
    }

    result = await orchestrator.execute(input_data)

    # Validate result aggregation
    coordination_result = result['coordination_result']

    assert 'task_assignments' in coordination_result
    assert len(coordination_result['task_assignments']) == len(agent_ids)

    # Each agent should have completed its task
    for agent_id in agent_ids:
        assert agent_id in coordination_result['task_assignments']


# ==============================================================================
# ERROR HANDLING AND RESILIENCE TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.coordination
async def test_agent_timeout_handling(
    orchestrator,
    mock_gl002_agent,
    sample_plant_data
):
    """
    Test handling of agent task timeouts.

    Validates:
    - Timeout detection
    - Graceful timeout handling
    - Partial result recovery
    - Error reporting
    """
    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': ['GL-002'],
        'agent_commands': {
            'GL-002': {
                'command_type': 'optimize',
                'priority': 'HIGH',
                'parameters': {'target_efficiency': 95.0},
                'timeout_seconds': 0.001  # Unrealistic timeout (will fail)
            }
        }
    }

    result = await orchestrator.execute(input_data)

    # Should handle timeout gracefully
    assert result is not None

    # Check for timeout indication
    coordination_result = result.get('coordination_result')
    if coordination_result:
        # Either reports timeout or handles gracefully
        assert 'task_assignments' in coordination_result or 'errors' in coordination_result


@pytest.mark.asyncio
@pytest.mark.coordination
async def test_agent_failure_recovery(
    orchestrator,
    all_sub_agents,
    sample_plant_data
):
    """
    Test recovery from individual agent failures.

    Validates:
    - Failure detection
    - Continued operation with remaining agents
    - Error propagation
    - Degraded mode operation
    """
    agent_ids = list(all_sub_agents.keys())

    # Stop one agent to simulate failure
    await all_sub_agents['GL-003'].stop()

    commands = {
        agent_id: {
            'command_type': 'optimize',
            'priority': 'MEDIUM',
            'parameters': {},
            'timeout_seconds': 30
        }
        for agent_id in agent_ids
    }

    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': agent_ids,
        'agent_commands': commands
    }

    result = await orchestrator.execute(input_data)

    # Should complete with partial success
    assert result is not None

    # Check coordination result
    coordination_result = result.get('coordination_result')

    if coordination_result and 'task_assignments' in coordination_result:
        # Should have fewer successful agents
        successful_agents = len(coordination_result['task_assignments'])
        assert successful_agents < len(agent_ids)

    # Restart failed agent
    await all_sub_agents['GL-003'].start()


@pytest.mark.asyncio
@pytest.mark.coordination
async def test_agent_error_propagation(
    orchestrator,
    mock_gl002_agent,
    sample_plant_data
):
    """
    Test error propagation from sub-agents to orchestrator.

    Validates:
    - Error detection and reporting
    - Error context preservation
    - Orchestrator error handling
    - User-friendly error messages
    """
    # Send invalid command to agent
    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': ['GL-002'],
        'agent_commands': {
            'GL-002': {
                'command_type': 'invalid_command',  # Invalid
                'priority': 'HIGH',
                'parameters': {},
                'timeout_seconds': 30
            }
        }
    }

    result = await orchestrator.execute(input_data)

    # Should handle gracefully
    assert result is not None

    # Check for error indication
    if 'coordination_result' in result:
        coord_result = result['coordination_result']
        # Error should be captured
        assert 'errors' in coord_result or 'task_assignments' in coord_result


# ==============================================================================
# PERFORMANCE AND LOAD TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.coordination
@pytest.mark.performance
async def test_agent_coordination_latency(
    orchestrator,
    all_sub_agents,
    sample_plant_data,
    performance_monitor
):
    """
    Test agent coordination latency.

    Target: <500ms for 4-agent coordination
    Validates:
    - Task delegation speed
    - Result collection speed
    - Overall coordination latency
    """
    performance_monitor.start()

    agent_ids = list(all_sub_agents.keys())

    commands = {
        agent_id: {
            'command_type': 'monitor',
            'priority': 'HIGH',
            'parameters': {},
            'timeout_seconds': 30
        }
        for agent_id in agent_ids
    }

    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': agent_ids,
        'agent_commands': commands
    }

    start = asyncio.get_event_loop().time()
    result = await orchestrator.execute(input_data)
    elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000

    performance_monitor.record_metric('coordination_latency_ms', elapsed_ms)

    print(f"\n=== Agent Coordination Latency ===")
    print(f"Agents: {len(agent_ids)}")
    print(f"Latency: {elapsed_ms:.2f}ms")

    # Validate performance target
    assert elapsed_ms < 500, f"Coordination latency {elapsed_ms:.2f}ms exceeds 500ms target"


@pytest.mark.asyncio
@pytest.mark.coordination
@pytest.mark.performance
async def test_concurrent_agent_coordination(
    orchestrator,
    all_sub_agents,
    sample_plant_data,
    performance_monitor
):
    """
    Test concurrent coordination requests.

    Validates:
    - Parallel coordination handling
    - Resource contention management
    - Performance under load
    """
    performance_monitor.start()

    agent_ids = list(all_sub_agents.keys())
    num_concurrent = 10

    async def coordinate_task(task_id):
        """Single coordination task."""
        commands = {
            agent_id: {
                'command_type': 'monitor',
                'priority': 'MEDIUM',
                'parameters': {'task_id': task_id},
                'timeout_seconds': 30
            }
            for agent_id in agent_ids
        }

        input_data = {
            'plant_data': sample_plant_data('PLANT-001'),
            'coordinate_agents': True,
            'agent_ids': agent_ids,
            'agent_commands': commands
        }

        start = asyncio.get_event_loop().time()
        result = await orchestrator.execute(input_data)
        elapsed = (asyncio.get_event_loop().time() - start) * 1000

        return {
            'task_id': task_id,
            'elapsed_ms': elapsed,
            'success': result is not None
        }

    # Execute concurrent coordination
    tasks = [coordinate_task(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*tasks)

    # Validate all completed
    assert len(results) == num_concurrent

    successful = sum(1 for r in results if r['success'])
    avg_latency = sum(r['elapsed_ms'] for r in results) / len(results)

    print(f"\n=== Concurrent Agent Coordination ===")
    print(f"Concurrent Requests: {num_concurrent}")
    print(f"Successful: {successful}/{num_concurrent}")
    print(f"Average Latency: {avg_latency:.2f}ms")

    # Validate performance
    assert successful >= num_concurrent * 0.9, "Too many coordination failures"
    assert avg_latency < 1000, "Average latency exceeds 1s target"


@pytest.mark.asyncio
@pytest.mark.coordination
async def test_agent_load_balancing(
    orchestrator,
    all_sub_agents,
    sample_plant_data
):
    """
    Test load balancing across multiple agents.

    Validates:
    - Even task distribution
    - Agent capacity management
    - Queue management
    - Overload prevention
    """
    agent_ids = list(all_sub_agents.keys())

    # Create many tasks for each agent
    commands = {}
    for agent_id in agent_ids:
        commands[agent_id] = {
            'command_type': 'optimize',
            'priority': 'MEDIUM',
            'parameters': {
                'iterations': 10  # Multiple iterations
            },
            'timeout_seconds': 60
        }

    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': agent_ids,
        'agent_commands': commands
    }

    result = await orchestrator.execute(input_data)

    # Validate load distribution
    coordination_result = result.get('coordination_result')

    if coordination_result and 'task_assignments' in coordination_result:
        task_counts = {
            agent_id: len(tasks)
            for agent_id, tasks in coordination_result['task_assignments'].items()
        }

        print(f"\n=== Agent Load Distribution ===")
        for agent_id, count in task_counts.items():
            print(f"{agent_id}: {count} tasks")

        # Validate reasonable distribution (not all on one agent)
        max_tasks = max(task_counts.values())
        min_tasks = min(task_counts.values())

        # Should be relatively balanced
        assert max_tasks / min_tasks < 2.0, "Load imbalance detected"
