# GL-003 SteamSystemAnalyzer Integration Modules

Production-ready integration connectors for industrial steam systems, sensors, and SCADA systems.

## Overview

The integration modules provide comprehensive connectivity to:
- **Steam Meters**: Modbus RTU/TCP, HART, 4-20mA analog
- **Pressure Sensors**: Multi-point monitoring with drift detection
- **Temperature Sensors**: RTD (PT100/PT1000), Thermocouples (K, J, T)
- **SCADA/DCS Systems**: OPC UA, Modbus TCP/RTU
- **Condensate Meters**: Return flow monitoring and quality analysis
- **Agent Coordination**: Multi-agent communication and task scheduling
- **Data Transformation**: Normalization, validation, quality scoring

## Features

All connectors include:
- **Connection Management**: Auto-reconnect with exponential backoff
- **Retry Logic**: Configurable retry attempts and delays
- **Circuit Breaker**: Fail-fast pattern to prevent cascading failures
- **Health Checks**: Periodic health monitoring with status reporting
- **Metrics Collection**: Performance metrics (latency, success rate, errors)
- **Thread Safety**: Async-safe operations with proper locking
- **Data Quality**: Validation, outlier detection, quality scoring
- **Mock Implementation**: Simulation mode for testing without hardware

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Dependencies include:
# - asyncio (built-in)
# - numpy (for data processing)
# - scipy (for signal processing)
```

## Quick Start

### Steam Meter Integration

```python
import asyncio
from integrations import SteamMeterConnector, SteamMeterConfig, MeterProtocol

async def main():
    # Configure steam meter
    config = SteamMeterConfig(
        host="192.168.1.100",
        port=502,
        protocol=MeterProtocol.MODBUS_TCP,
        meter_id="main_steam_header",
        sampling_rate_hz=1.0,
        units="t/hr",
        enable_totalizer=True
    )

    # Initialize connector
    connector = SteamMeterConnector(config)

    # Connect
    if await connector.connect():
        print("Connected to steam meter")

        # Read current flow
        reading = await connector.read_flow()
        print(f"Flow: {reading.flow_rate:.2f} {reading.unit}")
        print(f"Totalizer: {reading.totalizer:.0f} {reading.unit}")

        # Get averaged flow
        avg_flow = connector.get_averaged_flow(window_seconds=60)
        print(f"1-min average: {avg_flow:.2f} {config.units}")

        # Disconnect
        await connector.disconnect()

asyncio.run(main())
```

### Multi-Point Pressure Monitoring

```python
from integrations import PressureSensorConnector, PressureSensorConfig, PressureType

# Configure multiple pressure sensors
configs = [
    PressureSensorConfig(
        host="192.168.1.101",
        port=502,
        sensor_id="header_pressure",
        pressure_type=PressureType.GAUGE,
        min_pressure=0.0,
        max_pressure=20.0,
        sampling_rate_hz=2.0
    ),
    PressureSensorConfig(
        host="192.168.1.101",
        port=502,
        sensor_id="distribution_pressure",
        pressure_type=PressureType.GAUGE,
        min_pressure=0.0,
        max_pressure=15.0,
        sampling_rate_hz=2.0
    )
]

async def monitor_pressure():
    connector = PressureSensorConnector(configs)

    if await connector.connect():
        # Get all pressures
        pressures = connector.get_all_pressures()
        for sensor_id, pressure in pressures.items():
            print(f"{sensor_id}: {pressure:.2f} bar")

        # Get pressure profile
        profile = connector.get_pressure_profile()
        for sensor in profile:
            print(f"{sensor['sensor_id']}: {sensor['pressure']:.2f} {sensor['unit']}")

        # Calculate differential
        diff = connector.get_differential_pressure("header_pressure", "distribution_pressure")
        print(f"Pressure drop: {diff:.2f} bar")

        await connector.disconnect()

asyncio.run(monitor_pressure())
```

### Temperature Monitoring

```python
from integrations import (
    TemperatureSensorConnector,
    TemperatureSensorConfig,
    TemperatureSensorType
)

configs = [
    TemperatureSensorConfig(
        host="192.168.1.102",
        port=502,
        sensor_id="steam_temp",
        sensor_type=TemperatureSensorType.RTD_PT100,
        min_temp=0.0,
        max_temp=400.0,
        smoothing_window=5
    ),
    TemperatureSensorConfig(
        host="192.168.1.102",
        port=502,
        sensor_id="condensate_temp",
        sensor_type=TemperatureSensorType.THERMOCOUPLE_K,
        min_temp=0.0,
        max_temp=200.0
    )
]

async def monitor_temperature():
    connector = TemperatureSensorConnector(configs)

    if await connector.connect():
        temps = connector.get_all_temperatures()
        for sensor_id, temp in temps.items():
            print(f"{sensor_id}: {temp:.1f}°C")

        await connector.disconnect()

asyncio.run(monitor_temperature())
```

### SCADA Integration

```python
from integrations import SCADAConnector, SCADAConnectionConfig, SCADAProtocol

async def scada_integration():
    config = SCADAConnectionConfig(
        protocol=SCADAProtocol.OPC_UA,
        host="192.168.1.200",
        port=4840,
        enable_subscriptions=True
    )

    connector = SCADAConnector(config)

    if await connector.connect():
        # Read tags
        pressure = await connector.read_tag("STEAM.HEADER.PRESSURE")
        print(f"Steam pressure: {pressure} bar")

        # Write tag
        await connector.write_tag("STEAM.SETPOINT", 12.0)

        # Subscribe to tag changes
        async def on_pressure_change(tag_name, value, timestamp):
            print(f"{tag_name} changed to {value} at {timestamp}")

        await connector.subscribe("STEAM.HEADER.PRESSURE", on_pressure_change)

        # Get all current values
        values = connector.get_current_values()
        for tag, data in values.items():
            print(f"{tag}: {data['value']} {data['units']}")

        await connector.disconnect()

asyncio.run(scada_integration())
```

### Condensate Monitoring

```python
from integrations import CondensateMeterConnector, CondensateMeterConfig

async def condensate_monitoring():
    config = CondensateMeterConfig(
        host="192.168.1.103",
        port=502,
        meter_id="main_condensate_return",
        min_flow=0.0,
        max_flow=50.0,
        enable_quality_analysis=True
    )

    connector = CondensateMeterConnector(config)

    if await connector.connect():
        # Get condensate flow
        flow = connector.get_condensate_flow()
        print(f"Condensate flow: {flow:.2f} t/hr")

        # Get return percentage
        return_pct = connector.get_return_percentage()
        print(f"Return rate: {return_pct:.1f}%")

        # Calculate flash steam loss
        flash_loss = connector.calculate_flash_steam(pressure_drop_bar=3.0)
        print(f"Flash steam loss: {flash_loss:.2f} t/hr")

        await connector.disconnect()

asyncio.run(condensate_monitoring())
```

### Data Transformation Pipeline

```python
from integrations import DataTransformationPipeline
from datetime import datetime, timedelta

# Raw sensor data
raw_data = [
    {'timestamp': datetime.utcnow() - timedelta(minutes=10), 'value': 100, 'unit': 'psi', 'source': 'sensor1'},
    {'timestamp': datetime.utcnow() - timedelta(minutes=9), 'value': 102, 'unit': 'psi', 'source': 'sensor1'},
    {'timestamp': datetime.utcnow() - timedelta(minutes=8), 'value': 500, 'unit': 'psi', 'source': 'sensor1'},  # Outlier
    {'timestamp': datetime.utcnow() - timedelta(minutes=7), 'value': None, 'unit': 'psi', 'source': 'sensor1'},  # Missing
    {'timestamp': datetime.utcnow() - timedelta(minutes=6), 'value': 105, 'unit': 'psi', 'source': 'sensor1'},
]

# Processing configuration
config = {
    'target_unit': 'bar',
    'remove_outliers': True,
    'outlier_method': 'zscore',
    'outlier_threshold': 2.5,
    'impute_missing': True,
    'impute_method': 'linear'
}

# Process data
pipeline = DataTransformationPipeline()
result = pipeline.process_data(raw_data, config)

print(f"Quality score: {result['statistics']['quality_score']:.1f}")
print(f"Mean: {result['statistics']['mean']:.2f} bar")
print(f"Valid data: {result['statistics']['valid_count']}/{result['statistics']['count']}")
```

### Agent Coordination

```python
from integrations import AgentCoordinator, AgentRole, MessageType, MessagePriority, AgentMessage
import uuid

async def agent_coordination():
    # Initialize coordinator for GL-003
    coordinator = AgentCoordinator(
        agent_id="GL-003",
        role=AgentRole.HEAT_RECOVERY
    )

    await coordinator.start()

    # Send message to parent orchestrator (GL-001)
    message = AgentMessage(
        message_id=str(uuid.uuid4()),
        sender_id="GL-003",
        recipient_id="GL-001",
        message_type=MessageType.REQUEST,
        priority=MessagePriority.NORMAL,
        timestamp=datetime.utcnow(),
        payload={
            'action': 'report_steam_efficiency',
            'efficiency': 92.5,
            'steam_flow': 105.2,
            'condensate_return': 87.3
        },
        requires_response=True
    )

    await coordinator.send_message(message)

    # Request collaboration
    collaboration_id = await coordinator.request_collaboration(
        task_type="steam_system_optimization",
        required_agents=["GL-001", "GL-002", "GL-004"],
        parameters={'target_efficiency': 95.0}
    )

    print(f"Started collaboration: {collaboration_id}")

    await asyncio.sleep(5)
    await coordinator.stop()

asyncio.run(agent_coordination())
```

## Architecture Patterns

### Base Connector Pattern

All connectors inherit from `BaseConnector`:

```python
from integrations import BaseConnector, ConnectionConfig

class CustomConnector(BaseConnector):
    async def _connect_impl(self) -> bool:
        # Implement connection logic
        pass

    async def _disconnect_impl(self):
        # Implement disconnection logic
        pass

    async def _health_check_impl(self) -> bool:
        # Implement health check
        pass
```

### Circuit Breaker

Automatic circuit breaking prevents cascading failures:

```python
config = ConnectionConfig(
    host="192.168.1.100",
    port=502,
    enable_circuit_breaker=True,
    circuit_failure_threshold=5,  # Open after 5 failures
    circuit_timeout_seconds=60    # Try recovery after 60s
)
```

### Retry Logic

Exponential backoff retry:

```python
config = ConnectionConfig(
    max_retries=3,
    retry_delay_seconds=5,
    retry_backoff_multiplier=2.0,  # 5s, 10s, 20s
    max_retry_delay_seconds=300
)
```

## Configuration

### Environment Variables

```bash
# SCADA configuration
SCADA_HOST=192.168.1.200
SCADA_PORT=4840
SCADA_PROTOCOL=opc_ua

# Steam meter configuration
STEAM_METER_HOST=192.168.1.100
STEAM_METER_PORT=502
STEAM_METER_PROTOCOL=modbus_tcp

# Connection settings
CONNECTION_TIMEOUT=30
MAX_RETRIES=3
ENABLE_CIRCUIT_BREAKER=true
```

### Configuration Files

See `.env.template` for full configuration options.

## Testing

### Unit Tests

```bash
pytest tests/test_integrations.py -v
```

### Integration Tests

```bash
# With mock servers
pytest tests/integration/test_steam_meter.py -v

# With real hardware (requires hardware setup)
pytest tests/integration/test_steam_meter.py -v --real-hardware
```

### Mock Mode

All connectors support mock mode for testing:

```python
# Connectors automatically simulate when hardware unavailable
connector = SteamMeterConnector(config)
await connector.connect()  # Will succeed in mock mode
```

## Monitoring and Metrics

### Get Connector Metrics

```python
metrics = connector.get_metrics()
print(f"Total calls: {metrics['total_calls']}")
print(f"Success rate: {metrics['successful_calls'] / metrics['total_calls'] * 100:.1f}%")
print(f"Avg response time: {metrics['avg_response_time_ms']:.1f}ms")
print(f"Circuit breaker state: {metrics['circuit_breaker']['state']}")
```

### Health Status

```python
health = await connector.health_check()
print(f"Healthy: {health.is_healthy}")
print(f"State: {health.state.value}")
print(f"Response time: {health.response_time_ms}ms")
```

## Error Handling

```python
from integrations import CircuitBreakerOpenError

try:
    reading = await connector.read_flow()
except CircuitBreakerOpenError as e:
    logger.error(f"Circuit breaker open: {e}")
    # Fallback logic
except asyncio.TimeoutError:
    logger.error("Read timeout")
except Exception as e:
    logger.error(f"Read failed: {e}")
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY integrations/ ./integrations/
COPY steam_system_orchestrator.py .

CMD ["python", "steam_system_orchestrator.py"]
```

### Kubernetes

See `deployment/deployment.yaml` for Kubernetes manifests.

## Best Practices

1. **Always use async/await**: All connectors are async
2. **Handle connection loss**: Automatic reconnection is provided
3. **Validate data quality**: Check quality scores before using data
4. **Monitor health**: Regular health checks are essential
5. **Use circuit breaker**: Prevents cascade failures
6. **Log appropriately**: Use structured logging
7. **Configure timeouts**: Set realistic timeout values
8. **Test with mocks**: Use mock mode for unit tests

## Troubleshooting

### Connection Failures

```python
# Check connection state
print(f"State: {connector.state}")

# Check health
health = await connector.health_check()
print(f"Healthy: {health.is_healthy}")

# Check metrics
metrics = connector.get_metrics()
print(f"Failed calls: {metrics['failed_calls']}")
```

### Circuit Breaker Open

```python
# Manually reset circuit breaker
if connector.circuit_breaker:
    connector.circuit_breaker.reset()
```

### Poor Data Quality

```python
# Check quality score
if reading.quality_score < 50:
    logger.warning("Poor data quality")

# Use data transformation pipeline
pipeline = DataTransformationPipeline()
cleaned = pipeline.process_data(raw_data, config)
```

## Support

For issues and questions:
- **Documentation**: See `ARCHITECTURE.md` and `README.md`
- **Examples**: Check integration examples in this file
- **Tests**: Review `tests/integration/` for usage patterns

## License

Copyright (c) 2024 GreenLang Team
