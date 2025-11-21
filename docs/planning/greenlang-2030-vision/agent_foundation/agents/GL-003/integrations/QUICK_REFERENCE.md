# GL-003 Integration Modules - Quick Reference

## Module Overview

| Module | Purpose | Protocols | Key Features |
|--------|---------|-----------|--------------|
| `base_connector` | Base framework | N/A | Retry, circuit breaker, health checks |
| `steam_meter_connector` | Steam flow measurement | Modbus, HART, 4-20mA | Totalizer, quality scoring |
| `pressure_sensor_connector` | Multi-point pressure | Modbus, Analog | Drift detection, multi-zone |
| `temperature_sensor_connector` | Temperature monitoring | RTD, Thermocouple | Smoothing, CJC |
| `scada_connector` | SCADA/DCS integration | OPC UA, Modbus | Tag subscription, write-back |
| `condensate_meter_connector` | Condensate monitoring | Modbus, Analog | Flash steam, quality analysis |
| `agent_coordinator` | Multi-agent messaging | Internal | Task scheduling, state sync |
| `data_transformers` | Data processing | N/A | Unit conversion, validation |

## Quick Start Examples

### Steam Meter
```python
from integrations import SteamMeterConnector, SteamMeterConfig, MeterProtocol

config = SteamMeterConfig(
    host="192.168.1.100", port=502,
    protocol=MeterProtocol.MODBUS_TCP,
    meter_id="main_steam", sampling_rate_hz=1.0
)
connector = SteamMeterConnector(config)
await connector.connect()
reading = await connector.read_flow()
print(f"Flow: {reading.flow_rate} {reading.unit}")
```

### Pressure Sensors
```python
from integrations import PressureSensorConnector, PressureSensorConfig

configs = [
    PressureSensorConfig(
        host="192.168.1.101", port=502,
        sensor_id="header", min_pressure=0, max_pressure=20
    )
]
connector = PressureSensorConnector(configs)
await connector.connect()
pressures = connector.get_all_pressures()
```

### Temperature Sensors
```python
from integrations import TemperatureSensorConnector, TemperatureSensorConfig, TemperatureSensorType

configs = [
    TemperatureSensorConfig(
        host="192.168.1.102", port=502,
        sensor_id="steam_temp",
        sensor_type=TemperatureSensorType.RTD_PT100
    )
]
connector = TemperatureSensorConnector(configs)
await connector.connect()
temps = connector.get_all_temperatures()
```

### SCADA
```python
from integrations import SCADAConnector, SCADAConnectionConfig, SCADAProtocol

config = SCADAConnectionConfig(
    protocol=SCADAProtocol.OPC_UA,
    host="192.168.1.200", port=4840
)
connector = SCADAConnector(config)
await connector.connect()
value = await connector.read_tag("STEAM.PRESSURE")
await connector.write_tag("STEAM.SETPOINT", 12.0)
```

### Data Pipeline
```python
from integrations import DataTransformationPipeline

pipeline = DataTransformationPipeline()
result = pipeline.process_data(raw_data, {
    'target_unit': 'bar',
    'remove_outliers': True,
    'impute_missing': True
})
```

## Common Configuration Parameters

### Connection
```python
host: str = "192.168.1.100"
port: int = 502
timeout_seconds: int = 30
max_retries: int = 3
retry_delay_seconds: int = 5
```

### Circuit Breaker
```python
enable_circuit_breaker: bool = True
circuit_failure_threshold: int = 5
circuit_timeout_seconds: int = 60
```

### Health Checks
```python
health_check_interval_seconds: int = 30
```

## Key Methods

### All Connectors
```python
await connector.connect()           # Connect
await connector.disconnect()        # Disconnect
health = await connector.health_check()  # Check health
metrics = connector.get_metrics()   # Get metrics
```

### Steam Meter
```python
reading = await connector.read_flow()
avg = connector.get_averaged_flow(60)
stats = connector.get_flow_statistics(60)
totalizer = connector.get_totalizer()
```

### Pressure Sensors
```python
pressure = connector.get_pressure("sensor_id")
all_pressures = connector.get_all_pressures()
profile = connector.get_pressure_profile()
diff = connector.get_differential_pressure("high", "low")
```

### Temperature Sensors
```python
temp = connector.get_temperature("sensor_id")
all_temps = connector.get_all_temperatures()
```

### SCADA
```python
value = await connector.read_tag("tag_name")
await connector.write_tag("tag_name", value)
await connector.subscribe("tag_name", callback)
values = connector.get_current_values()
```

## Error Handling

```python
from integrations import CircuitBreakerOpenError

try:
    reading = await connector.read_flow()
except CircuitBreakerOpenError:
    # Circuit breaker is open
    pass
except asyncio.TimeoutError:
    # Operation timed out
    pass
except Exception as e:
    # Other errors
    pass
```

## Metrics

```python
metrics = connector.get_metrics()
# Returns:
{
    'total_calls': int,
    'successful_calls': int,
    'failed_calls': int,
    'total_retries': int,
    'avg_response_time_ms': float,
    'state': str,
    'health': {...},
    'circuit_breaker': {...}
}
```

## Health Status

```python
health = await connector.health_check()
# Returns:
{
    'is_healthy': bool,
    'state': ConnectionState,
    'consecutive_failures': int,
    'response_time_ms': float
}
```

## File Locations

```
GL-003/integrations/
├── __init__.py                       # Module exports
├── base_connector.py                 # Base framework
├── steam_meter_connector.py          # Steam meters
├── pressure_sensor_connector.py      # Pressure sensors
├── temperature_sensor_connector.py   # Temperature sensors
├── scada_connector.py                # SCADA/DCS
├── condensate_meter_connector.py     # Condensate meters
├── agent_coordinator.py              # Multi-agent
├── data_transformers.py              # Data processing
├── README.md                         # Full documentation
├── QUICK_REFERENCE.md                # This file
└── integration_example.py            # Complete example
```

## Common Issues

### Connection Timeout
```python
# Increase timeout
config.timeout_seconds = 60
```

### Circuit Breaker Open
```python
# Reset manually
connector.circuit_breaker.reset()
```

### Poor Data Quality
```python
# Check quality score
if reading.quality_score < 70:
    # Handle poor quality
```

## Environment Variables

```bash
# SCADA
SCADA_HOST=192.168.1.200
SCADA_PORT=4840
SCADA_PROTOCOL=opc_ua

# Steam Meter
STEAM_METER_HOST=192.168.1.100
STEAM_METER_PORT=502
STEAM_METER_PROTOCOL=modbus_tcp

# Connection
CONNECTION_TIMEOUT=30
MAX_RETRIES=3
ENABLE_CIRCUIT_BREAKER=true
```

## Performance Tuning

### High-Frequency Sampling
```python
config.sampling_rate_hz = 10.0  # 10Hz
```

### Low Latency
```python
config.timeout_seconds = 5
config.retry_delay_seconds = 1
```

### High Reliability
```python
config.max_retries = 5
config.enable_circuit_breaker = True
```

## Testing

### Mock Mode
All connectors work in mock mode without hardware:
```python
# Will automatically use mock if hardware unavailable
connector = SteamMeterConnector(config)
await connector.connect()  # Succeeds in mock mode
```

### Unit Tests
```bash
pytest tests/test_integrations.py -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

## Support

- **Full Documentation**: `README.md`
- **Complete Example**: `integration_example.py`
- **API Reference**: Inline docstrings
- **Architecture**: `../ARCHITECTURE.md`
