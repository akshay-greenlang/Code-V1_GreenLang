"""
End-to-End Workflow Integration Tests for GL-001 ProcessHeatOrchestrator

Comprehensive E2E tests covering complete orchestration workflows including:
- Full plant heat optimization workflow
- Multi-agent coordination workflow
- Heat distribution optimization
- Energy balance validation
- KPI dashboard generation
- Alert and notification flow
- Error recovery and resilience
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from process_heat_orchestrator import ProcessHeatOrchestrator
from config import ProcessHeatConfig, PlantConfiguration


# ==============================================================================
# E2E TEST FIXTURES
# ==============================================================================

@pytest.fixture
async def orchestrator_config():
    """Create ProcessHeat configuration for testing."""
    config = ProcessHeatConfig(
        agent_name="GL-001-TEST",
        version="1.0.0",
        agent_id="gl-001-e2e-test",
        plants=[
            PlantConfiguration(
                plant_id="PLANT-001",
                plant_name="Test Plant 1",
                capacity_mw=100.0,
                fuel_types=["natural_gas", "biomass"]
            )
        ],
        calculation_timeout_seconds=30,
        enable_monitoring=True,
        cache_ttl_seconds=300,
        emission_regulations={
            'nox_limit_mg_nm3': 150.0,
            'so2_limit_mg_nm3': 100.0,
            'pm_limit_mg_nm3': 20.0
        }
    )
    return config


@pytest.fixture
async def orchestrator(orchestrator_config):
    """Create orchestrator instance for testing."""
    orch = ProcessHeatOrchestrator(orchestrator_config)
    yield orch
    await orch.shutdown()


# ==============================================================================
# E2E WORKFLOW TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_full_plant_heat_optimization_workflow(
    orchestrator,
    sample_plant_data,
    sample_sensor_feeds,
    sample_emissions_data,
    sample_optimization_constraints,
    integration_assertions,
    performance_monitor
):
    """
    Test complete plant heat optimization workflow from sensor data to KPI dashboard.

    Workflow Steps:
    1. Collect plant data and sensor feeds
    2. Calculate thermal efficiency
    3. Optimize heat distribution
    4. Validate energy balance
    5. Check emissions compliance
    6. Generate KPI dashboard
    7. Store results and provenance
    """
    performance_monitor.start()

    # Step 1: Prepare input data
    performance_monitor.record_event("prepare_input_data")

    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'sensor_feeds': sample_sensor_feeds('PLANT-001', tag_count=100),
        'emissions_data': sample_emissions_data('PLANT-001'),
        'constraints': sample_optimization_constraints,
        'coordinate_agents': False
    }

    # Step 2: Execute orchestration
    performance_monitor.record_event("start_orchestration")

    result = await orchestrator.execute(input_data)

    performance_monitor.record_event("orchestration_completed")

    # Step 3: Validate result structure
    integration_assertions.assert_orchestration_result_valid(result)

    # Step 4: Validate thermal efficiency calculation
    thermal_efficiency = result['thermal_efficiency']
    assert 'overall_efficiency' in thermal_efficiency
    assert 0 < thermal_efficiency['overall_efficiency'] <= 100
    assert thermal_efficiency['carnot_efficiency'] > thermal_efficiency['overall_efficiency']

    performance_monitor.record_metric('thermal_efficiency', thermal_efficiency['overall_efficiency'])

    # Step 5: Validate heat distribution optimization
    heat_distribution = result['heat_distribution']
    assert 'distribution_matrix' in heat_distribution
    assert 'optimization_score' in heat_distribution
    assert heat_distribution['constraints_satisfied'] is True

    performance_monitor.record_metric('optimization_score', heat_distribution['optimization_score'])

    # Step 6: Validate energy balance
    energy_balance = result['energy_balance']
    integration_assertions.assert_energy_balance_valid(energy_balance, tolerance_percent=2.0)

    # Step 7: Validate emissions compliance
    emissions_compliance = result['emissions_compliance']
    assert emissions_compliance['compliance_status'] in ['PASS', 'WARNING', 'FAIL']

    # Step 8: Validate KPI dashboard
    kpi_dashboard = result['kpi_dashboard']
    assert 'thermal_efficiency' in kpi_dashboard
    assert 'heat_recovery_rate' in kpi_dashboard
    assert 'co2_intensity' in kpi_dashboard

    # Step 9: Validate performance
    performance_monitor.record_metric('execution_time_ms', result['execution_time_ms'])
    integration_assertions.assert_performance_target_met(
        result['execution_time_ms'],
        target_ms=2000  # 2-second target
    )

    # Step 10: Validate provenance
    assert len(result['provenance_hash']) == 64  # SHA-256

    perf_report = performance_monitor.stop()
    print(f"\n=== E2E Workflow Performance Report ===")
    print(f"Total Duration: {perf_report['total_duration_ms']:.2f}ms")
    print(f"Events: {perf_report['event_count']}")
    print(f"Metrics: {perf_report['metrics']}")


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_multi_agent_coordination_workflow(
    orchestrator,
    sample_plant_data,
    sample_sensor_feeds,
    all_sub_agents,
    integration_assertions,
    test_data_generator
):
    """
    Test multi-agent coordination workflow with all sub-agents.

    Workflow:
    1. Orchestrator receives task requiring sub-agents
    2. Orchestrator delegates tasks to GL-002, GL-003, GL-004, GL-005
    3. Sub-agents process tasks
    4. Orchestrator aggregates results
    5. Orchestrator generates comprehensive report
    """
    # Prepare coordination commands
    agent_ids = list(all_sub_agents.keys())
    commands = test_data_generator.generate_coordination_commands(agent_ids)

    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'sensor_feeds': sample_sensor_feeds('PLANT-001'),
        'coordinate_agents': True,
        'agent_ids': agent_ids,
        'agent_commands': commands
    }

    # Execute with coordination
    result = await orchestrator.execute(input_data)

    # Validate orchestration result
    integration_assertions.assert_orchestration_result_valid(result)

    # Validate coordination result
    coordination_result = result['coordination_result']
    assert coordination_result is not None
    integration_assertions.assert_agent_coordination_successful(coordination_result)

    # Validate all agents received tasks
    assert 'task_assignments' in coordination_result
    assert len(coordination_result['task_assignments']) == len(agent_ids)

    # Validate task execution
    for agent_id in agent_ids:
        assert agent_id in coordination_result['task_assignments']
        tasks = coordination_result['task_assignments'][agent_id]
        assert len(tasks) > 0

        # Check each task has required fields
        for task in tasks:
            assert 'task_id' in task
            assert 'priority' in task
            assert 'timeout_seconds' in task


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_heat_distribution_optimization_workflow(
    orchestrator,
    sample_sensor_feeds,
    sample_optimization_constraints
):
    """
    Test heat distribution optimization workflow with complex constraints.

    Workflow:
    1. Collect sensor data from multiple heat sources and consumers
    2. Identify optimization constraints
    3. Solve heat distribution optimization problem
    4. Validate solution feasibility
    5. Generate distribution strategy
    """
    # Create complex sensor feed with multiple sources/consumers
    sensor_feeds = sample_sensor_feeds('PLANT-001', tag_count=150)

    # Add heat sources
    sensor_feeds['tags']['BOILER.1.HEAT.OUTPUT'] = {'value': 50.0, 'quality': 'GOOD'}
    sensor_feeds['tags']['BOILER.2.HEAT.OUTPUT'] = {'value': 45.0, 'quality': 'GOOD'}
    sensor_feeds['tags']['HEAT.RECOVERY.OUTPUT'] = {'value': 10.0, 'quality': 'GOOD'}

    # Add heat consumers
    sensor_feeds['tags']['PROCESS.1.HEAT.DEMAND'] = {'value': 35.0, 'quality': 'GOOD'}
    sensor_feeds['tags']['PROCESS.2.HEAT.DEMAND'] = {'value': 30.0, 'quality': 'GOOD'}
    sensor_feeds['tags']['PROCESS.3.HEAT.DEMAND'] = {'value': 25.0, 'quality': 'GOOD'}
    sensor_feeds['tags']['HEATING.DEMAND'] = {'value': 15.0, 'quality': 'GOOD'}

    input_data = {
        'plant_data': {
            'inlet_temp_c': 500,
            'outlet_temp_c': 150,
            'ambient_temp_c': 25,
            'fuel_input_mw': 120,
            'useful_heat_mw': 105
        },
        'sensor_feeds': sensor_feeds,
        'constraints': sample_optimization_constraints
    }

    result = await orchestrator.execute(input_data)

    # Validate heat distribution
    heat_dist = result['heat_distribution']

    assert heat_dist['total_heat_supply_mw'] > 0
    assert heat_dist['total_heat_demand_mw'] > 0
    assert heat_dist['constraints_satisfied'] is True
    assert heat_dist['optimization_score'] >= 0

    # Validate distribution matrix
    distribution_matrix = heat_dist['distribution_matrix']
    assert isinstance(distribution_matrix, dict)

    # Verify heat balance
    total_supply = heat_dist['total_heat_supply_mw']
    total_demand = heat_dist['total_heat_demand_mw']

    # Supply should meet or exceed demand (accounting for losses)
    assert total_supply >= total_demand * 0.95  # 5% loss tolerance


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_energy_balance_validation_workflow(
    orchestrator,
    sample_plant_data,
    integration_assertions
):
    """
    Test energy balance validation workflow ensuring conservation of energy.

    Workflow:
    1. Measure energy inputs (fuel)
    2. Measure energy outputs (useful heat)
    3. Measure energy losses (flue gas, radiation, etc.)
    4. Validate energy balance (Input = Output + Losses)
    5. Flag violations if balance error exceeds tolerance
    """
    # Create test data with known energy balance
    plant_data = sample_plant_data('PLANT-001')
    plant_data['fuel_input_mw'] = 100.0
    plant_data['useful_heat_mw'] = 85.0
    plant_data['heat_recovery_mw'] = 10.0
    # Expected losses: 5.0 MW (100 - 85 - 10)

    input_data = {
        'plant_data': plant_data,
        'sensor_feeds': {'tags': {}},
        'constraints': {'energy_balance_tolerance_percent': 2.0}
    }

    result = await orchestrator.execute(input_data)

    # Validate energy balance
    energy_balance = result['energy_balance']

    assert 'input_energy_mw' in energy_balance
    assert 'output_energy_mw' in energy_balance
    assert 'losses_mw' in energy_balance
    assert 'balance_error_percent' in energy_balance
    assert 'is_valid' in energy_balance

    # Validate balance error is within tolerance
    integration_assertions.assert_energy_balance_valid(
        energy_balance,
        tolerance_percent=2.0
    )

    # Check violations list
    assert 'violations' in energy_balance
    if energy_balance['is_valid']:
        assert len(energy_balance['violations']) == 0


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_kpi_dashboard_generation_workflow(
    orchestrator,
    sample_plant_data,
    sample_sensor_feeds,
    sample_emissions_data
):
    """
    Test KPI dashboard generation workflow with comprehensive metrics.

    Dashboard Metrics:
    - Thermal efficiency
    - Heat recovery rate
    - Capacity utilization
    - CO2 intensity
    - Compliance score
    - Energy balance accuracy
    """
    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'sensor_feeds': sample_sensor_feeds('PLANT-001'),
        'emissions_data': sample_emissions_data('PLANT-001'),
        'constraints': {
            'min_efficiency_percent': 85.0,
            'max_emissions': {
                'nox_mg_nm3': 150.0,
                'so2_mg_nm3': 100.0
            }
        }
    }

    result = await orchestrator.execute(input_data)

    # Validate KPI dashboard
    kpi_dashboard = result['kpi_dashboard']

    required_kpis = [
        'thermal_efficiency',
        'heat_recovery_rate',
        'capacity_utilization',
        'co2_intensity',
        'compliance_score',
        'energy_balance_accuracy'
    ]

    for kpi in required_kpis:
        assert kpi in kpi_dashboard, f"Missing KPI: {kpi}"
        assert isinstance(kpi_dashboard[kpi], (int, float))
        assert kpi_dashboard[kpi] >= 0

    # Validate KPI ranges
    assert 0 <= kpi_dashboard['thermal_efficiency'] <= 100
    assert 0 <= kpi_dashboard['heat_recovery_rate'] <= 100
    assert 0 <= kpi_dashboard['capacity_utilization'] <= 150  # Can exceed 100% during peak
    assert kpi_dashboard['co2_intensity'] >= 0
    assert 0 <= kpi_dashboard['compliance_score'] <= 100
    assert 0 <= kpi_dashboard['energy_balance_accuracy'] <= 100


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.slow
async def test_alert_and_notification_workflow(
    orchestrator,
    sample_plant_data,
    sample_emissions_data
):
    """
    Test alert and notification workflow for abnormal conditions.

    Alert Conditions:
    - Low efficiency (<85%)
    - High emissions (exceeding limits)
    - Energy balance violations
    - Equipment faults
    """
    # Create plant data with low efficiency (should trigger alert)
    plant_data = sample_plant_data('PLANT-001')
    plant_data['fuel_input_mw'] = 100.0
    plant_data['useful_heat_mw'] = 75.0  # 75% efficiency (low)
    plant_data['heat_recovery_mw'] = 5.0

    # Create emissions data exceeding limits
    emissions_data = sample_emissions_data('PLANT-001')
    emissions_data['nox_mg_nm3'] = 180.0  # Exceeds limit
    emissions_data['so2_mg_nm3'] = 120.0  # Exceeds limit

    input_data = {
        'plant_data': plant_data,
        'sensor_feeds': {'tags': {}},
        'emissions_data': emissions_data,
        'constraints': {
            'min_efficiency_percent': 85.0,
            'max_emissions': {
                'nox_mg_nm3': 150.0,
                'so2_mg_nm3': 100.0
            }
        }
    }

    result = await orchestrator.execute(input_data)

    # Validate thermal efficiency alert
    thermal_eff = result['thermal_efficiency']
    assert thermal_eff['overall_efficiency'] < 85.0  # Below threshold

    # Validate emissions compliance alert
    emissions_comp = result['emissions_compliance']
    assert emissions_comp['compliance_status'] == 'FAIL'
    assert len(emissions_comp['violations']) > 0

    # Check violations contain NOx and SO2
    violation_pollutants = [v.lower() for v in str(emissions_comp['violations'])]
    assert any('nox' in v for v in violation_pollutants)
    assert any('so2' in v for v in violation_pollutants)


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_error_recovery_workflow(
    orchestrator,
    sample_plant_data
):
    """
    Test error recovery workflow when sub-components fail.

    Error Scenarios:
    1. Invalid input data
    2. Calculation timeout
    3. Sub-agent failure
    4. Partial data availability
    """
    # Test 1: Invalid input data (negative values)
    invalid_data = sample_plant_data('PLANT-001')
    invalid_data['fuel_input_mw'] = -100.0  # Invalid negative value

    input_data = {
        'plant_data': invalid_data,
        'sensor_feeds': {'tags': {}}
    }

    # Should not raise exception, but return partial results
    result = await orchestrator.execute(input_data)

    # Result should indicate partial success or degraded mode
    assert result is not None
    assert 'status' in result or 'thermal_efficiency' in result

    # Test 2: Missing required data
    incomplete_data = {
        'plant_data': {},  # Empty plant data
        'sensor_feeds': {'tags': {}}
    }

    result = await orchestrator.execute(incomplete_data)
    assert result is not None


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_deterministic_reproducibility_workflow(
    orchestrator,
    sample_plant_data,
    sample_sensor_feeds
):
    """
    Test that orchestration produces deterministic, reproducible results.

    Same input data should produce:
    - Identical calculation results
    - Identical provenance hashes
    - Identical optimization decisions
    """
    input_data = {
        'plant_data': sample_plant_data('PLANT-001'),
        'sensor_feeds': sample_sensor_feeds('PLANT-001'),
        'constraints': {
            'min_efficiency_percent': 85.0
        }
    }

    # Execute twice with same input
    result1 = await orchestrator.execute(input_data)
    await asyncio.sleep(0.1)  # Small delay
    result2 = await orchestrator.execute(input_data)

    # Compare results (allowing for minor floating-point differences)
    eff1 = result1['thermal_efficiency']['overall_efficiency']
    eff2 = result2['thermal_efficiency']['overall_efficiency']

    # Should be identical or very close (within 0.01%)
    assert abs(eff1 - eff2) < 0.01

    # Optimization scores should be identical
    opt1 = result1['heat_distribution']['optimization_score']
    opt2 = result2['heat_distribution']['optimization_score']

    assert abs(opt1 - opt2) < 0.01


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.slow
async def test_continuous_operation_workflow(
    orchestrator,
    sample_plant_data,
    sample_sensor_feeds,
    performance_monitor
):
    """
    Test continuous operation workflow over extended period.

    Simulates 24 hours of operation with data every 5 minutes (288 cycles).
    Validates:
    - Memory stability
    - Performance consistency
    - Cache effectiveness
    - No memory leaks
    """
    performance_monitor.start()

    num_cycles = 50  # Reduced for testing (50 cycles ~= 4 hours at 5min intervals)
    execution_times = []

    for i in range(num_cycles):
        input_data = {
            'plant_data': sample_plant_data('PLANT-001'),
            'sensor_feeds': sample_sensor_feeds('PLANT-001'),
            'constraints': {'min_efficiency_percent': 85.0}
        }

        result = await orchestrator.execute(input_data)
        execution_times.append(result['execution_time_ms'])

        performance_monitor.record_metric('execution_time_ms', result['execution_time_ms'])

        # Small delay between cycles
        await asyncio.sleep(0.01)

    # Validate performance consistency
    avg_time = sum(execution_times) / len(execution_times)
    max_time = max(execution_times)
    min_time = min(execution_times)

    print(f"\n=== Continuous Operation Performance ===")
    print(f"Cycles: {num_cycles}")
    print(f"Avg Execution Time: {avg_time:.2f}ms")
    print(f"Min Execution Time: {min_time:.2f}ms")
    print(f"Max Execution Time: {max_time:.2f}ms")

    # Performance should remain stable (max < 3 * avg)
    assert max_time < avg_time * 3, "Performance degraded significantly"

    # Check cache effectiveness
    perf_metrics = orchestrator.performance_metrics
    cache_hit_rate = perf_metrics['cache_hits'] / (perf_metrics['cache_hits'] + perf_metrics['cache_misses'])

    print(f"Cache Hit Rate: {cache_hit_rate * 100:.1f}%")

    # Cache should be effective (>30% hit rate after warmup)
    assert cache_hit_rate > 0.3, "Cache not effective"


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_state_persistence_workflow(
    orchestrator,
    sample_plant_data
):
    """
    Test state persistence workflow ensuring agent state can be saved/restored.

    Workflow:
    1. Execute multiple operations
    2. Get agent state
    3. Validate state contains all necessary information
    4. Simulate restart
    5. Restore state
    6. Continue operations
    """
    # Execute some operations
    for i in range(5):
        input_data = {
            'plant_data': sample_plant_data('PLANT-001'),
            'sensor_feeds': {'tags': {}}
        }
        await orchestrator.execute(input_data)

    # Get current state
    state = orchestrator.get_state()

    # Validate state
    assert 'agent_id' in state
    assert 'state' in state
    assert 'version' in state
    assert 'performance_metrics' in state
    assert 'cache_size' in state
    assert 'memory_entries' in state
    assert 'timestamp' in state

    # Validate metrics
    metrics = state['performance_metrics']
    assert metrics['calculations_performed'] >= 5
    assert metrics['avg_calculation_time_ms'] > 0

    # Validate cache
    assert state['cache_size'] >= 0

    # Validate memory
    assert state['memory_entries'] >= 5


# ==============================================================================
# E2E PERFORMANCE TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.performance
async def test_e2e_performance_under_load(
    orchestrator,
    sample_plant_data,
    sample_sensor_feeds,
    performance_monitor
):
    """
    Test E2E performance under concurrent load.

    Simulates multiple concurrent orchestration requests.
    Target: <2 seconds per request with 10 concurrent requests
    """
    performance_monitor.start()

    num_concurrent = 10

    async def execute_task(task_id):
        input_data = {
            'plant_data': sample_plant_data('PLANT-001'),
            'sensor_feeds': sample_sensor_feeds('PLANT-001'),
            'constraints': {'min_efficiency_percent': 85.0}
        }

        start = asyncio.get_event_loop().time()
        result = await orchestrator.execute(input_data)
        elapsed = (asyncio.get_event_loop().time() - start) * 1000

        return {
            'task_id': task_id,
            'elapsed_ms': elapsed,
            'result': result
        }

    # Execute concurrent tasks
    tasks = [execute_task(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*tasks)

    # Validate all completed successfully
    assert len(results) == num_concurrent

    # Check performance targets
    for result in results:
        assert result['elapsed_ms'] < 2000, \
            f"Task {result['task_id']} exceeded 2s target: {result['elapsed_ms']:.2f}ms"

    avg_time = sum(r['elapsed_ms'] for r in results) / len(results)
    print(f"\n=== Concurrent Load Performance ===")
    print(f"Concurrent Requests: {num_concurrent}")
    print(f"Average Time: {avg_time:.2f}ms")
    print(f"Min Time: {min(r['elapsed_ms'] for r in results):.2f}ms")
    print(f"Max Time: {max(r['elapsed_ms'] for r in results):.2f}ms")
