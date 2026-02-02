# GL-020 ECONOPULSE - Enterprise Data Integration Module

## Overview

The GL-020 ECONOPULSE integration module provides enterprise-grade connectors for economizer instrumentation and control systems. This module enables seamless data acquisition from industrial sensors, SCADA/DCS systems, process historians, and soot blower control systems.

**Agent ID:** GL-020
**Codename:** ECONOPULSE
**Version:** 1.0.0

## Architecture

```
+------------------+     +------------------+     +------------------+
|   Field Sensors  |     |   SCADA/DCS     |     |  Process Historian|
|  (RTD, TC, Flow) | --> |  (OPC UA/Modbus) | --> |  (PI, IP.21, WW)  |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        v
+------------------------------------------------------------------------+
|                     GL-020 Integration Layer                            |
|  +----------------+  +------------------+  +------------------------+   |
|  | Sensor         |  | SCADA            |  | Historian              |   |
|  | Connectors     |  | Integration      |  | Connectors             |   |
|  +----------------+  +------------------+  +------------------------+   |
|                              |                                          |
|                     +------------------+                                 |
|                     | Data Quality     |                                 |
|                     | Validation       |                                 |
|                     +------------------+                                 |
+------------------------------------------------------------------------+
         |
         v
+------------------+
| Economizer Model |
| & Analytics      |
+------------------+
```

## Module Components

### 1. Sensor Connectors (`sensor_connector.py`)

Enterprise connectors for direct sensor integration:

| Connector | Sensor Types | Protocols |
|-----------|-------------|-----------|
| `RTDTemperatureSensor` | PT100, PT1000 | Modbus RTU/TCP, 4-20mA |
| `ThermocoupleConnector` | Type J, K, T | Modbus RTU, HART |
| `FlowMeterConnector` | Orifice, Vortex, Ultrasonic | Modbus TCP/RTU |
| `PressureTransducerConnector` | Differential, Gauge, Absolute | Modbus TCP/RTU, 4-20mA |

**Key Features:**
- Automatic calibration offset application
- Bad sensor detection (stuck, out-of-range, noisy)
- Circuit breaker pattern for fault tolerance
- Thread-safe concurrent access

### 2. SCADA Integration (`scada_integration.py`)

OPC UA and Modbus TCP client for SCADA/DCS connectivity:

**Supported Protocols:**
- OPC UA (with security modes: None, Sign, SignAndEncrypt)
- Modbus TCP
- Modbus RTU (via serial gateway)

**Pre-configured Tag Groups:**
- Feedwater temperatures (inlet, outlet, intermediate)
- Flue gas temperatures (inlet, outlet, zone-based)
- Flow rates (water, gas)
- Differential pressures
- Soot blower status

**Capabilities:**
- Tag subscription with real-time callbacks
- Batch tag reading
- Historical data retrieval
- Setpoint write-back with confirmation

### 3. Soot Blower Integration (`soot_blower_integration.py`)

Complete soot blower control system interface:

**Functions:**
- Real-time blower status monitoring
- Cleaning cycle triggering with safety interlocks
- Zone-based cleaning control
- Media consumption tracking (steam/air)
- Cleaning effectiveness analysis

**Safety Interlocks:**
- Steam pressure (low/high limits)
- Steam temperature (high limit)
- Boiler load (minimum threshold)
- Blower overtravel/stuck detection
- Emergency stop
- Sequence timeout

### 4. Historian Connectors (`historian_connector.py`)

Process historian integration for time-series data:

| Connector | Historian | Protocol |
|-----------|-----------|----------|
| `OSIsoftPIConnector` | OSIsoft PI | PI Web API |
| `AspenInfoPlusConnector` | AspenTech IP.21 | SQLplus/ODBC |
| `WonderwareHistorianConnector` | Wonderware/AVEVA | InSQL/ODBC |

**Data Retrieval Methods:**
- Raw archived data
- Interpolated data
- Aggregated data (average, min, max, total, count)
- Snapshot (current) values

**Aggregation Types:**
- Average, Minimum, Maximum
- Total, Count
- Standard Deviation, Variance
- Time-weighted Average
- Range

### 5. Data Quality (`data_quality.py`)

Comprehensive data quality validation:

**Validation Checks:**
- Range checking (normal and critical limits)
- Rate-of-change limits (spike detection)
- Statistical outlier detection
- Historical comparison

**Redundancy Management:**
- 2-out-of-3 (2oo3) voting
- Average of good values
- Median selection
- High/Low select
- Priority-based selection

**Bad Data Substitution:**
- Last good value
- Linear interpolation
- Average substitution
- Manual fallback
- Redundant sensor substitution

## Installation

```bash
# Clone repository
git clone <repository-url>
cd GL-020

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
asyncio
httpx>=0.24.0
pymodbus>=3.5.0
asyncua>=1.0.0
pyodbc>=4.0.0
pandas>=2.0.0
```

## Configuration

### Environment Variables

```bash
# SCADA Configuration
SCADA_HOST=192.168.1.100
SCADA_PORT=4840
SCADA_PROTOCOL=OPC_UA

# Historian Configuration
PI_HOST=pi-server.domain.com
PI_PORT=443
PI_USE_WEB_API=true

# Credentials (use vault in production)
SCADA_USERNAME=scada_user
PI_USERNAME=pi_user
```

### Configuration Files

```yaml
# config/scada.yaml
scada:
  protocol: OPC_UA
  host: 192.168.1.100
  port: 4840
  security_mode: SignAndEncrypt
  security_policy: Basic256Sha256
  connection_timeout: 30
  session_timeout: 3600
```

## Usage Examples

### Reading Sensor Data

```python
from integrations import (
    RTDTemperatureSensor,
    SensorConfig,
    SensorType,
    SensorProtocol,
)

# Configure RTD sensor
config = SensorConfig(
    sensor_id="FW_INLET_TEMP_1",
    sensor_type=SensorType.RTD_PT100,
    protocol=SensorProtocol.MODBUS_TCP,
    address="192.168.1.50:502",
    unit_id=1,
    register_address=40001,
    engineering_unit="C",
    range_low=0.0,
    range_high=300.0,
)

# Create and connect sensor
sensor = RTDTemperatureSensor(config)
await sensor.connect()

# Read temperature
reading = await sensor.read()
print(f"Temperature: {reading.value} {reading.unit}")
print(f"Quality: {reading.quality.value}")
print(f"Confidence: {reading.confidence:.2f}")
```

### SCADA Tag Subscription

```python
from integrations import (
    SCADAClient,
    SCADAConfig,
    SCADAProtocol,
    EconomizerTagGroups,
)

# Configure SCADA client
config = SCADAConfig(
    protocol=SCADAProtocol.OPC_UA,
    host="192.168.1.100",
    port=4840,
    application_name="GL-020-ECONOPULSE",
    security_mode="SignAndEncrypt",
)

# Create client and connect
client = SCADAClient(config)
await client.connect()

# Subscribe to economizer tag groups
client.subscribe_group(EconomizerTagGroups.feedwater_temperatures())
client.subscribe_group(EconomizerTagGroups.flue_gas_temperatures())
client.subscribe_group(EconomizerTagGroups.differential_pressures())

# Define callback for value updates
def on_value_update(tag_value):
    print(f"{tag_value.tag_name}: {tag_value.value} ({tag_value.quality.value})")

# Add callback and start polling
for tag_group in EconomizerTagGroups.all_groups():
    for tag in tag_group.tags:
        client.subscribe(tag, callback=on_value_update)

await client.start_polling()
```

### Soot Blower Control

```python
from integrations import (
    SootBlowerController,
    SootBlowerConfig,
    CleaningZone,
    CleaningMediaType,
)

# Configure soot blower controller
config = SootBlowerConfig(
    controller_id="ECON_SB_01",
    controller_name="Economizer Soot Blower",
    protocol="opc_ua",
    host="192.168.1.100",
    port=4840,
    media_type=CleaningMediaType.STEAM,
    steam_pressure_setpoint_kpa=1200.0,
)

# Create controller
controller = SootBlowerController(config, scada_client=scada_client)
await controller.connect()

# Register cleaning zones
controller.register_zone(CleaningZone(
    zone_id="ECON_ZONE_A",
    name="Economizer Zone A",
    description="Upper tube bank",
    blower_ids=["SB_01", "SB_02"],
    priority=1,
    min_cleaning_interval_hours=4.0,
))

# Trigger cleaning cycle
success, message, cycle = await controller.trigger_cleaning_cycle(
    zone_id="ECON_ZONE_A",
    operator_id="operator_01",
    trigger_type="manual",
)

if success:
    print(f"Started cleaning cycle: {cycle.cycle_id}")
else:
    print(f"Failed to start: {message}")
```

### Historian Data Retrieval

```python
from integrations import (
    OSIsoftPIConnector,
    HistorianConfig,
    AggregateType,
)
from datetime import datetime, timedelta

# Configure PI historian
config = HistorianConfig(
    name="PI_SERVER",
    host="pi-server.domain.com",
    port=443,
    use_ssl=True,
)

# Create connector
pi = OSIsoftPIConnector(config, use_web_api=True)
await pi.connect()

# Read historical data
end_time = datetime.now()
start_time = end_time - timedelta(hours=24)

data = await pi.read_aggregated(
    tag_names=["ECON.FW.TEMP.INLET", "ECON.FW.TEMP.OUTLET"],
    start_time=start_time,
    end_time=end_time,
    interval_seconds=3600,  # Hourly averages
    aggregate_type=AggregateType.AVERAGE,
)

for tag_name, ts_data in data.items():
    print(f"\n{tag_name}:")
    for point in ts_data.points[-5:]:  # Last 5 points
        print(f"  {point.timestamp}: {point.value:.2f}")
```

### Data Quality Validation

```python
from integrations import (
    DataQualityValidator,
    RangeCheck,
    RateOfChangeLimit,
    create_economizer_validator,
)

# Create pre-configured validator
validator = create_economizer_validator()

# Or configure manually
validator = DataQualityValidator()
validator.configure_range_check(RangeCheck(
    tag_name="ECON.FW.TEMP.INLET",
    low_limit=40.0,
    high_limit=180.0,
    low_low_limit=20.0,
    high_high_limit=200.0,
    engineering_unit="C",
))

# Validate a reading
result = validator.validate(
    tag_name="ECON.FW.TEMP.INLET",
    value=165.5,
    timestamp=datetime.now(),
)

print(f"Validated value: {result.validated_value}")
print(f"Quality: {result.quality_flag.value}")
print(f"Confidence: {result.confidence_score.total_score:.1f}%")
```

### Redundant Sensor Voting

```python
from integrations import (
    RedundancyManager,
    RedundantSensorGroup,
    VotingMethod,
    QualityFlag,
    create_economizer_redundancy_manager,
)

# Create pre-configured manager
manager = create_economizer_redundancy_manager()

# Perform voting
sensor_values = {
    "ECON.FW.TEMP.INLET.A": (165.2, QualityFlag.GOOD),
    "ECON.FW.TEMP.INLET.B": (165.5, QualityFlag.GOOD),
    "ECON.FW.TEMP.INLET.C": (172.0, QualityFlag.GOOD),  # Outlier
}

value, quality, reason = manager.vote("FW_INLET_TEMP", sensor_values)
print(f"Selected value: {value:.2f}")
print(f"Quality: {quality.value}")
print(f"Reason: {reason}")
```

## Error Handling

All connectors implement comprehensive error handling:

```python
try:
    reading = await sensor.read()
except ConnectionError as e:
    logger.error(f"Connection lost: {e}")
    # Attempt reconnection
except TimeoutError as e:
    logger.error(f"Read timeout: {e}")
    # Use last good value
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Fail safe
```

## Circuit Breaker Pattern

All connectors implement the circuit breaker pattern:

```python
# Circuit breaker states:
# CLOSED: Normal operation
# OPEN: Failing fast (after failure_threshold failures)
# HALF_OPEN: Testing recovery (after recovery_timeout)

# Circuit breaker is automatic - no manual intervention required
# Check state programmatically:
if connector._circuit_breaker.is_available():
    # Proceed with operation
else:
    # Service unavailable, use fallback
```

## Logging

Configure logging for troubleshooting:

```python
import logging

# Enable debug logging for integrations
logging.getLogger("integrations").setLevel(logging.DEBUG)

# Or per-module:
logging.getLogger("integrations.sensor_connector").setLevel(logging.DEBUG)
logging.getLogger("integrations.scada_integration").setLevel(logging.INFO)
```

## Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests (requires test infrastructure)
pytest tests/integration/

# Run with coverage
pytest --cov=integrations tests/
```

## Security Considerations

1. **Credential Management**: Never hardcode credentials. Use vault integration:
   ```python
   connector = SCADAClient(config, vault_client=vault_client)
   ```

2. **OPC UA Security**: Use SignAndEncrypt mode in production:
   ```python
   config.security_mode = "SignAndEncrypt"
   config.security_policy = "Basic256Sha256"
   ```

3. **Network Segmentation**: Deploy integration layer in OT DMZ

4. **Audit Logging**: All write operations are logged with operator ID

## Performance Optimization

1. **Connection Pooling**: Configure appropriate pool sizes:
   ```python
   config.max_connections = 5
   ```

2. **Batch Operations**: Use batch reads for multiple tags:
   ```python
   values = await client.read_multiple_tags(tag_names)
   ```

3. **Caching**: Use tag value caching for frequently accessed data:
   ```python
   cached_value = client.get_cached_value(tag_name)
   ```

## Support

For issues or questions:
- Create an issue in the GL-020 repository
- Contact: greenlang-support@example.com

## License

Proprietary - GreenLang Project
