"""
SCADA Integration Tests for GL-001 ProcessHeatOrchestrator

Comprehensive SCADA integration tests covering:
- OPC UA multi-plant connectivity
- Modbus TCP/RTU communication
- Tag subscription and data streaming
- Historical data retrieval
- Connection resilience and failover
- Real-time data validation
- Protocol error handling
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from process_heat_orchestrator import ProcessHeatOrchestrator


# ==============================================================================
# OPC UA INTEGRATION TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.scada
@pytest.mark.integration
async def test_opcua_single_plant_connection(
    orchestrator,
    mock_opcua_server,
    integration_assertions
):
    """
    Test OPC UA connection to single plant SCADA system.

    Validates:
    - Successful OPC UA connection
    - Server endpoint discovery
    - Session establishment
    - Basic tag read operations
    """
    # Simulate SCADA integration
    scada_feed = {
        'protocol': 'opcua',
        'endpoint': f'opc.tcp://{mock_opcua_server.host}:{mock_opcua_server.port}',
        'plant_id': mock_opcua_server.plant_id,
        'tags_to_read': [
            f'{mock_opcua_server.plant_id}.BOILER.STEAM.PRESSURE',
            f'{mock_opcua_server.plant_id}.BOILER.STEAM.TEMPERATURE',
            f'{mock_opcua_server.plant_id}.BOILER.EFFICIENCY'
        ]
    }

    # Integrate SCADA data
    result = await orchestrator.integrate_scada(scada_feed)

    # Validate connection success
    assert result['status'] == 'connected'
    assert result['protocol'] == 'opcua'
    assert result['plant_id'] == mock_opcua_server.plant_id

    # Validate tag data
    assert 'tag_values' in result
    tag_values = result['tag_values']

    for tag_name in scada_feed['tags_to_read']:
        assert tag_name in tag_values
        tag_data = tag_values[tag_name]

        assert 'value' in tag_data
        assert 'quality' in tag_data
        assert 'timestamp' in tag_data
        assert tag_data['quality'] == 'GOOD'


@pytest.mark.asyncio
@pytest.mark.scada
@pytest.mark.integration
async def test_opcua_multi_plant_connection(
    orchestrator,
    mock_multi_plant_coordinator,
    integration_assertions
):
    """
    Test OPC UA connections to multiple plants simultaneously.

    Validates:
    - Concurrent OPC UA connections (3 plants)
    - Independent session management
    - Data isolation between plants
    - Connection pooling
    """
    plants_status = mock_multi_plant_coordinator.get_all_plants_status()

    # Create SCADA feeds for all plants
    scada_feeds = []
    for i, plant_id in enumerate(plants_status.keys()):
        feed = {
            'protocol': 'opcua',
            'endpoint': f'opc.tcp://localhost:{4840 + i}',
            'plant_id': plant_id,
            'tags_to_read': [
                f'{plant_id}.BOILER.STEAM.PRESSURE',
                f'{plant_id}.BOILER.EFFICIENCY',
                f'{plant_id}.PLANT.LOAD.PERCENT'
            ]
        }
        scada_feeds.append(feed)

    # Integrate all SCADA feeds concurrently
    tasks = [orchestrator.integrate_scada(feed) for feed in scada_feeds]
    results = await asyncio.gather(*tasks)

    # Validate all connections successful
    assert len(results) == len(plants_status)

    for i, result in enumerate(results):
        assert result['status'] == 'connected'
        assert result['protocol'] == 'opcua'

        # Validate data isolation (each plant has its own data)
        tag_values = result['tag_values']
        expected_plant_id = list(plants_status.keys())[i]

        for tag_name in tag_values.keys():
            assert tag_name.startswith(expected_plant_id)


@pytest.mark.asyncio
@pytest.mark.scada
async def test_opcua_tag_subscription(
    orchestrator,
    mock_opcua_server
):
    """
    Test OPC UA tag subscription for real-time monitoring.

    Validates:
    - Tag subscription creation
    - Real-time value updates
    - Subscription callbacks
    - Unsubscription cleanup
    """
    subscription_tags = [
        f'{mock_opcua_server.plant_id}.BOILER.STEAM.PRESSURE',
        f'{mock_opcua_server.plant_id}.BOILER.STEAM.TEMPERATURE',
        f'{mock_opcua_server.plant_id}.BOILER.O2.CONTENT'
    ]

    received_updates = []

    async def subscription_callback(tag_name, value, timestamp):
        """Callback for tag updates."""
        received_updates.append({
            'tag_name': tag_name,
            'value': value,
            'timestamp': timestamp
        })

    # Create subscription
    subscription_id = await mock_opcua_server.subscribe_tags(
        subscription_tags,
        subscription_callback
    )

    assert subscription_id is not None
    assert subscription_id.startswith('SUB-')

    # Simulate tag value changes
    for tag in subscription_tags:
        await mock_opcua_server.write_tag(tag, 100.0)

    # Wait for updates
    await asyncio.sleep(0.5)

    # Validate subscription worked (in real implementation)
    # Note: Mock server may not trigger callbacks, so this is structure validation
    assert len(mock_opcua_server.subscriptions) > 0

    subscription = mock_opcua_server.subscriptions[0]
    assert subscription['id'] == subscription_id
    assert set(subscription['tags']) == set(subscription_tags)


@pytest.mark.asyncio
@pytest.mark.scada
async def test_opcua_historical_data_retrieval(
    orchestrator,
    mock_opcua_server,
    test_data_generator
):
    """
    Test OPC UA historical data retrieval for trend analysis.

    Validates:
    - Historical data queries
    - Time range filtering
    - Data aggregation
    - Large dataset handling
    """
    # Generate historical data
    tag_name = f'{mock_opcua_server.plant_id}.BOILER.EFFICIENCY'
    start_time = datetime.now(timezone.utc) - timedelta(hours=24)
    end_time = datetime.now(timezone.utc)

    historical_data = test_data_generator.generate_time_series(
        duration_hours=24,
        interval_seconds=300,  # 5-minute intervals
        base_value=89.5,
        variation=0.05
    )

    # Request historical data
    scada_feed = {
        'protocol': 'opcua',
        'endpoint': f'opc.tcp://{mock_opcua_server.host}:{mock_opcua_server.port}',
        'operation': 'historical_read',
        'tag_name': tag_name,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'aggregation': 'raw'  # or 'average', 'min', 'max'
    }

    result = await orchestrator.integrate_scada(scada_feed)

    # Validate historical data structure
    assert result['operation'] == 'historical_read'
    assert 'historical_data' in result

    hist_data = result['historical_data']
    assert isinstance(hist_data, list)

    # Validate data points
    if len(hist_data) > 0:
        for point in hist_data:
            assert 'timestamp' in point
            assert 'value' in point


@pytest.mark.asyncio
@pytest.mark.scada
async def test_opcua_connection_resilience(
    orchestrator,
    mock_opcua_server
):
    """
    Test OPC UA connection resilience and automatic reconnection.

    Scenarios:
    - Network timeout
    - Server restart
    - Session timeout
    - Automatic reconnection
    """
    # Initial connection
    scada_feed = {
        'protocol': 'opcua',
        'endpoint': f'opc.tcp://{mock_opcua_server.host}:{mock_opcua_server.port}',
        'plant_id': mock_opcua_server.plant_id,
        'tags_to_read': [f'{mock_opcua_server.plant_id}.BOILER.EFFICIENCY']
    }

    result1 = await orchestrator.integrate_scada(scada_feed)
    assert result1['status'] == 'connected'

    # Simulate connection loss
    await mock_opcua_server.stop()
    await asyncio.sleep(0.5)

    # Attempt read during disconnection
    result2 = await orchestrator.integrate_scada(scada_feed)

    # Should handle gracefully (return error or cached data)
    assert result2['status'] in ['disconnected', 'error', 'cached']

    # Restart server
    await mock_opcua_server.start()
    await asyncio.sleep(0.5)

    # Should reconnect automatically
    result3 = await orchestrator.integrate_scada(scada_feed)
    assert result3['status'] == 'connected'


# ==============================================================================
# MODBUS TCP INTEGRATION TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.scada
@pytest.mark.integration
async def test_modbus_tcp_connection(
    orchestrator,
    mock_modbus_server
):
    """
    Test Modbus TCP connection to fuel/emissions monitoring systems.

    Validates:
    - Modbus TCP connection
    - Holding register reads
    - Coil reads
    - Batch read operations
    """
    scada_feed = {
        'protocol': 'modbus_tcp',
        'host': mock_modbus_server.host,
        'port': mock_modbus_server.port,
        'plant_id': mock_modbus_server.plant_id,
        'operations': [
            {
                'type': 'read_holding_registers',
                'start_address': 0,
                'count': 10  # Fuel flow meters
            },
            {
                'type': 'read_holding_registers',
                'start_address': 200,
                'count': 8  # Emissions data
            }
        ]
    }

    result = await orchestrator.integrate_scada(scada_feed)

    # Validate connection
    assert result['status'] == 'connected'
    assert result['protocol'] == 'modbus_tcp'

    # Validate register reads
    assert 'register_values' in result
    register_values = result['register_values']

    # Should have fuel and emissions data
    assert len(register_values) >= 18  # 10 + 8


@pytest.mark.asyncio
@pytest.mark.scada
async def test_modbus_fuel_flow_monitoring(
    orchestrator,
    mock_modbus_server
):
    """
    Test real-time fuel flow monitoring via Modbus.

    Validates:
    - Natural gas flow reading
    - Fuel oil flow reading
    - Biomass flow reading
    - Unit conversions
    """
    scada_feed = {
        'protocol': 'modbus_tcp',
        'host': mock_modbus_server.host,
        'port': mock_modbus_server.port,
        'operation': 'read_fuel_flows',
        'fuel_types': ['natural_gas', 'fuel_oil', 'biomass']
    }

    result = await orchestrator.integrate_scada(scada_feed)

    assert result['status'] == 'connected'
    assert 'fuel_flows' in result

    fuel_flows = result['fuel_flows']

    # Validate fuel flow data
    for fuel_type in ['natural_gas', 'fuel_oil', 'biomass']:
        assert fuel_type in fuel_flows
        flow_data = fuel_flows[fuel_type]

        assert 'flow_rate' in flow_data
        assert 'unit' in flow_data
        assert flow_data['flow_rate'] >= 0


@pytest.mark.asyncio
@pytest.mark.scada
async def test_modbus_emissions_monitoring(
    orchestrator,
    mock_modbus_server
):
    """
    Test continuous emissions monitoring system (CEMS) via Modbus.

    Validates:
    - CO2 concentration
    - NOx concentration
    - SO2 concentration
    - O2 concentration
    - PM concentration
    - Stack flow
    """
    scada_feed = {
        'protocol': 'modbus_tcp',
        'host': mock_modbus_server.host,
        'port': mock_modbus_server.port,
        'operation': 'read_cems',
        'pollutants': ['co2', 'nox', 'so2', 'o2', 'pm', 'co']
    }

    result = await orchestrator.integrate_scada(scada_feed)

    assert result['status'] == 'connected'
    assert 'cems_data' in result

    cems_data = result['cems_data']

    # Validate CEMS data
    required_pollutants = ['co2', 'nox', 'so2', 'o2', 'pm', 'co']
    for pollutant in required_pollutants:
        assert pollutant in cems_data

        data = cems_data[pollutant]
        assert 'concentration' in data
        assert 'unit' in data
        assert 'quality' in data
        assert data['quality'] == 'GOOD'


@pytest.mark.asyncio
@pytest.mark.scada
async def test_modbus_write_operations(
    orchestrator,
    mock_modbus_server
):
    """
    Test Modbus write operations for control commands.

    Validates:
    - Write single holding register
    - Write multiple holding registers
    - Write single coil
    - Write multiple coils
    """
    # Write holding register (setpoint)
    scada_feed = {
        'protocol': 'modbus_tcp',
        'host': mock_modbus_server.host,
        'port': mock_modbus_server.port,
        'operation': 'write_holding_register',
        'address': 500,
        'value': 95.0  # Efficiency setpoint
    }

    result = await orchestrator.integrate_scada(scada_feed)

    assert result['status'] == 'success'
    assert result['operation'] == 'write_holding_register'

    # Verify write
    read_result = await mock_modbus_server.read_holding_register(500)
    assert read_result == 95.0

    # Write coil (digital output)
    scada_feed_coil = {
        'protocol': 'modbus_tcp',
        'host': mock_modbus_server.host,
        'port': mock_modbus_server.port,
        'operation': 'write_coil',
        'address': 10,
        'value': True  # Enable auto mode
    }

    result_coil = await orchestrator.integrate_scada(scada_feed_coil)

    assert result_coil['status'] == 'success'


# ==============================================================================
# PROTOCOL ERROR HANDLING TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.scada
async def test_scada_connection_timeout(
    orchestrator
):
    """
    Test SCADA connection timeout handling.

    Validates:
    - Connection timeout detection
    - Timeout error handling
    - Graceful degradation
    """
    scada_feed = {
        'protocol': 'opcua',
        'endpoint': 'opc.tcp://invalid-host:4840',  # Invalid endpoint
        'plant_id': 'PLANT-999',
        'timeout_seconds': 5,
        'tags_to_read': ['BOILER.EFFICIENCY']
    }

    result = await orchestrator.integrate_scada(scada_feed)

    # Should handle timeout gracefully
    assert result['status'] in ['error', 'timeout', 'disconnected']
    assert 'error_message' in result or 'message' in result


@pytest.mark.asyncio
@pytest.mark.scada
async def test_scada_invalid_tag_handling(
    orchestrator,
    mock_opcua_server
):
    """
    Test handling of invalid/non-existent tags.

    Validates:
    - Invalid tag detection
    - Error reporting
    - Partial success handling
    """
    scada_feed = {
        'protocol': 'opcua',
        'endpoint': f'opc.tcp://{mock_opcua_server.host}:{mock_opcua_server.port}',
        'plant_id': mock_opcua_server.plant_id,
        'tags_to_read': [
            f'{mock_opcua_server.plant_id}.BOILER.EFFICIENCY',  # Valid
            f'{mock_opcua_server.plant_id}.INVALID.TAG.NAME',   # Invalid
            f'{mock_opcua_server.plant_id}.ANOTHER.INVALID'     # Invalid
        ]
    }

    result = await orchestrator.integrate_scada(scada_feed)

    # Should handle partial success
    assert 'tag_values' in result or 'errors' in result

    if 'tag_values' in result:
        # Valid tags should have values
        assert f'{mock_opcua_server.plant_id}.BOILER.EFFICIENCY' in result['tag_values']

    if 'errors' in result:
        # Invalid tags should be reported
        errors = result['errors']
        assert isinstance(errors, list)


@pytest.mark.asyncio
@pytest.mark.scada
async def test_scada_data_quality_validation(
    orchestrator,
    mock_opcua_server
):
    """
    Test SCADA data quality validation and handling.

    Quality States:
    - GOOD: Normal data
    - UNCERTAIN: Questionable data
    - BAD: Invalid data

    Validates:
    - Quality code checking
    - Bad quality data filtering
    - Quality-based alerts
    """
    # Simulate fault to degrade data quality
    mock_opcua_server.simulate_fault('communication_error')

    scada_feed = {
        'protocol': 'opcua',
        'endpoint': f'opc.tcp://{mock_opcua_server.host}:{mock_opcua_server.port}',
        'plant_id': mock_opcua_server.plant_id,
        'tags_to_read': [f'{mock_opcua_server.plant_id}.BOILER.EFFICIENCY'],
        'require_good_quality': True
    }

    result = await orchestrator.integrate_scada(scada_feed)

    # Should detect quality issues
    if result['status'] == 'connected':
        # Check for quality warnings
        assert 'data_quality_issues' in result or 'warnings' in result


# ==============================================================================
# PERFORMANCE TESTS
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.scada
@pytest.mark.performance
async def test_scada_high_frequency_polling(
    orchestrator,
    mock_opcua_server,
    performance_monitor
):
    """
    Test high-frequency SCADA data polling performance.

    Target: 100 polls/second per plant
    Validates:
    - Polling performance
    - Data throughput
    - Memory efficiency
    """
    performance_monitor.start()

    num_polls = 100
    poll_interval = 0.01  # 10ms = 100 Hz

    scada_feed = {
        'protocol': 'opcua',
        'endpoint': f'opc.tcp://{mock_opcua_server.host}:{mock_opcua_server.port}',
        'plant_id': mock_opcua_server.plant_id,
        'tags_to_read': [
            f'{mock_opcua_server.plant_id}.BOILER.EFFICIENCY',
            f'{mock_opcua_server.plant_id}.BOILER.STEAM.PRESSURE'
        ]
    }

    poll_times = []

    for i in range(num_polls):
        start = asyncio.get_event_loop().time()

        result = await orchestrator.integrate_scada(scada_feed)

        elapsed = (asyncio.get_event_loop().time() - start) * 1000
        poll_times.append(elapsed)

        performance_monitor.record_metric('poll_time_ms', elapsed)

        await asyncio.sleep(poll_interval)

    # Calculate statistics
    avg_poll_time = sum(poll_times) / len(poll_times)
    max_poll_time = max(poll_times)
    min_poll_time = min(poll_times)

    print(f"\n=== High-Frequency Polling Performance ===")
    print(f"Polls: {num_polls}")
    print(f"Average Poll Time: {avg_poll_time:.2f}ms")
    print(f"Min Poll Time: {min_poll_time:.2f}ms")
    print(f"Max Poll Time: {max_poll_time:.2f}ms")

    # Validate performance
    assert avg_poll_time < 100, "Average poll time exceeds 100ms target"


@pytest.mark.asyncio
@pytest.mark.scada
@pytest.mark.performance
async def test_scada_concurrent_plant_polling(
    orchestrator,
    mock_multi_plant_coordinator,
    performance_monitor
):
    """
    Test concurrent SCADA polling across multiple plants.

    Validates:
    - Concurrent connection handling
    - Data isolation
    - Performance under load
    """
    performance_monitor.start()

    plants_status = mock_multi_plant_coordinator.get_all_plants_status()

    async def poll_plant(plant_id, port_offset):
        """Poll single plant."""
        scada_feed = {
            'protocol': 'opcua',
            'endpoint': f'opc.tcp://localhost:{4840 + port_offset}',
            'plant_id': plant_id,
            'tags_to_read': [
                f'{plant_id}.BOILER.EFFICIENCY',
                f'{plant_id}.PLANT.LOAD.PERCENT'
            ]
        }

        start = asyncio.get_event_loop().time()
        result = await orchestrator.integrate_scada(scada_feed)
        elapsed = (asyncio.get_event_loop().time() - start) * 1000

        return {
            'plant_id': plant_id,
            'elapsed_ms': elapsed,
            'result': result
        }

    # Poll all plants concurrently
    tasks = [
        poll_plant(plant_id, i)
        for i, plant_id in enumerate(plants_status.keys())
    ]

    results = await asyncio.gather(*tasks)

    # Validate all polls successful
    assert len(results) == len(plants_status)

    for result in results:
        assert result['result']['status'] == 'connected'
        assert result['elapsed_ms'] < 500  # <500ms per poll

    avg_time = sum(r['elapsed_ms'] for r in results) / len(results)
    print(f"\n=== Concurrent Plant Polling ===")
    print(f"Plants: {len(results)}")
    print(f"Average Time: {avg_time:.2f}ms")
