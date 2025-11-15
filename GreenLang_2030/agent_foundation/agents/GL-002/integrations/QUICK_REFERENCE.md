# GL-002 Integration Modules - Quick Reference Guide

**Version:** 1.0.0 | **Status:** Production Ready | **Date:** 2025-11-15

---

## Module Overview

| Module | Purpose | Key Classes | Lines |
|--------|---------|------------|-------|
| `boiler_control_connector.py` | DCS/PLC control | `BoilerControlManager`, `ModbusBoilerConnector`, `OPCUABoilerConnector`, `SafetyInterlock` | 783 |
| `fuel_management_connector.py` | Fuel systems | `FuelManagementConnector`, `FuelTank`, `FuelFlowMeter`, `FuelQualityAnalyzer`, `FuelCostOptimizer` | 900 |
| `scada_connector.py` | Real-time data | `SCADAConnector`, `AlarmManager`, `SCADADataBuffer`, `TagProcessor` | 959 |
| `emissions_monitoring_connector.py` | CEMS/compliance | `EmissionsMonitoringConnector`, `ComplianceMonitor`, `EmissionsCalculator`, `PredictiveEmissionsModel` | 1043 |
| `data_transformers.py` | Data processing | `DataTransformationPipeline`, `UnitConverter`, `DataValidator`, `OutlierDetector`, `SensorFusion` | 1301 |
| `agent_coordinator.py` | Multi-agent | `AgentCoordinator`, `MessageBus`, `TaskScheduler`, `StateManager`, `CollaborativeOptimizer` | 1105 |
| `__init__.py` | Module init | Public API exports | 167 |
| **TOTAL** | **Complete Suite** | **60+ classes** | **6,258** |

---

## Quick Start Examples

### 1. Boiler Control

```python
from gl002.integrations import (
    BoilerControlManager, BoilerControlConfig, BoilerProtocol
)

# Initialize
manager = BoilerControlManager()

# Configure and connect
config = BoilerControlConfig(
    protocol=BoilerProtocol.MODBUS_TCP,
    host="192.168.1.100",
    port=502,
    unit_id=1
)
await manager.add_connector("boiler_1", config)

# Read parameters
readings = await manager.read_all_parameters()
print(f"Steam Pressure: {readings['boiler_1_steam_pressure']}")

# Apply setpoints (with safety checks)
results = await manager.optimize_setpoints({
    'steam_pressure': 105.0,    # bar
    'o2_content': 3.5            # %
})

# Start continuous monitoring
await manager.start_monitoring(scan_interval=1)

# Cleanup
await manager.disconnect_all()
```

### 2. Fuel Management

```python
from gl002.integrations import (
    FuelManagementConnector, FuelTank, FuelType,
    FuelFlowMeter, FuelQualityAnalyzer
)

# Initialize
fuel = FuelManagementConnector()

# Add fuel tank
tank = FuelTank(
    tank_id="ng_tank_1",
    fuel_type=FuelType.NATURAL_GAS,
    capacity=1000.0,           # m3
    current_level=750.0,       # m3
    min_operating_level=100.0,
    max_fill_level=950.0
)
await fuel.add_tank(tank)

# Add flow meter
meter = FuelFlowMeter(
    meter_id="ng_meter_1",
    fuel_type=FuelType.NATURAL_GAS,
    meter_type="coriolis",
    min_flow=10,               # m3/hr
    max_flow=500,              # m3/hr
    accuracy_percent=0.5
)
await fuel.add_flow_meter(meter)

# Get status
status = await fuel.get_fuel_status()
print(f"Tank Level: {status['tank_level']}%")
print(f"Flow Rate: {status['current_flow']} m3/hr")

# Optimize fuel usage
optimization = await fuel.optimize_fuel_usage()
print(f"Recommended: {optimization['recommended_fuel']}")
print(f"Cost Savings: ${optimization['cost_savings']}/day")
```

### 3. SCADA Integration

```python
from gl002.integrations import (
    SCADAConnector, SCADAConnectionConfig, SCADAProtocol,
    SCADATag, DataQuality
)

# Initialize
scada = SCADAConnector()

# Configure connection
config = SCADAConnectionConfig(
    protocol=SCADAProtocol.OPC_UA,
    host="scada.plant.com",
    port=4840,
    tls_enabled=True
)

# Initialize connection
await scada.initialize([("plant1", config)])

# Register tags
tag = SCADATag(
    tag_name='BOILER_01.STEAM_PRESSURE',
    description='Main Boiler Steam Pressure',
    data_type='float',
    engineering_units='bar',
    scan_rate=1000,            # milliseconds
    deadband=0.5,              # Only update if change > 0.5
    min_value=0,
    max_value=150,
    alarm_limits={'H': 130, 'L': 20}
)
await scada.register_tag(tag)

# Start monitoring
await scada.start_monitoring()

# Get recent data (24-hour buffer)
data = await scada.get_recent_data(minutes=5)
for point in data:
    if point.quality == DataQuality.GOOD:
        print(f"{point.tag_name}: {point.value} {point.unit}")

# Get alarms
alarms = await scada.get_active_alarms()
for alarm in alarms:
    print(f"ALARM [{alarm.priority}]: {alarm.message}")
```

### 4. Emissions Monitoring

```python
from gl002.integrations import (
    EmissionsMonitoringConnector, CEMSConfig,
    ComplianceStandard, EmissionType
)

# Initialize
emissions = EmissionsMonitoringConnector()

# Configure CEMS
config = CEMSConfig(
    protocol='REST_API',
    host='cems.plant.com',
    port=443,
    api_key='your_api_key',
    tls_enabled=True,
    compliance_standards=[
        ComplianceStandard.EPA_PART_75,
        ComplianceStandard.EPA_MATS
    ]
)

await emissions.connect(config)

# Check compliance
compliance = await emissions.check_compliance(
    standard=ComplianceStandard.EPA_PART_75,
    averaging_period='hourly'
)

if compliance['compliant']:
    print("System in compliance")
else:
    for violation in compliance['violations']:
        print(f"VIOLATION: {violation['pollutant']} "
              f"({violation['value']} > {violation['limit']})")

# Get recent readings
readings = await emissions.get_recent_readings(minutes=60)
for reading in readings:
    print(f"{reading.pollutant}: {reading.value} {reading.unit} "
          f"(Quality: {reading.validation_status})")

# Generate regulatory report
report = await emissions.generate_regulatory_report(
    start_date='2025-01-01',
    end_date='2025-12-31',
    standard=ComplianceStandard.EPA_PART_75
)
await emissions.submit_report(report)
```

### 5. Data Transformation

```python
from gl002.integrations import (
    DataTransformationPipeline, UnitConverter,
    DataValidator, UnitSystem
)

# Unit conversion
converter = UnitConverter()

# Temperature
celsius = 100.0
fahrenheit = converter.convert_temperature(celsius, 'celsius', 'fahrenheit')
print(f"{celsius}°C = {fahrenheit}°F")

# Pressure
bar = 10.0
psi = converter.convert_pressure(bar, 'bar', 'psi')
print(f"{bar} bar = {psi} psi")

# Data validation and transformation
pipeline = DataTransformationPipeline()

raw_data = [
    {'timestamp': ..., 'value': 98.5, 'unit': 'celsius'},
    {'timestamp': ..., 'value': 'FAULT', 'unit': 'celsius'},  # Invalid
    {'timestamp': ..., 'value': 150.0, 'unit': 'celsius'},    # Spike
]

# Define schema
schema = {
    'value': {'type': float, 'min': 0, 'max': 120},
    'unit': {'type': str}
}

# Transform
result = pipeline.transform_data(
    raw_data,
    target_unit='celsius',
    target_unit_system=UnitSystem.SI,
    schema=schema,
    outlier_detection='z_score',
    imputation_method='interpolation'
)

print(f"Valid records: {result['valid_records']}")
print(f"Quality score: {result['quality_score']}/100")

# Advanced: Sensor fusion
measurements = [
    {'source': 'sensor_1', 'value': 99.5, 'confidence': 0.95},
    {'source': 'sensor_2', 'value': 100.2, 'confidence': 0.92},
]

fused = pipeline.fuse_sensors(measurements)
print(f"Fused value: {fused['value']} (confidence: {fused['confidence']})")
```

### 6. Agent Coordination

```python
from gl002.integrations import (
    AgentCoordinator, AgentMessage, AgentTask,
    MessageType, MessagePriority, AgentRole,
    TaskStatus
)

# Initialize coordinator
coordinator = AgentCoordinator()
await coordinator.initialize()

# Register this agent (GL-002)
profile = AgentProfile(
    agent_id='GL-002',
    role=AgentRole.BOILER_OPTIMIZER,
    capabilities=[...]
)
await coordinator.register_agent(profile)

# Send message to GL-001
message = AgentMessage(
    message_id='msg_001',
    sender_id='GL-002',
    recipient_id='GL-001',
    message_type=MessageType.REQUEST,
    priority=MessagePriority.HIGH,
    timestamp=datetime.utcnow(),
    payload={'request': 'optimization_targets'},
    requires_response=True
)
await coordinator.send_message(message)

# Wait for response
response = await asyncio.wait_for(
    coordinator.wait_for_response(message.message_id),
    timeout=30.0
)

# Submit task for distribution
task = AgentTask(
    task_id='task_001',
    task_type='optimize_parameters',
    requester_id='GL-001',
    priority=MessagePriority.HIGH,
    status=TaskStatus.PENDING,
    created_at=datetime.utcnow(),
    deadline=datetime.utcnow() + timedelta(minutes=5),
    parameters={'efficiency_target': 0.92}
)
await coordinator.submit_task(task)

# Monitor task
while True:
    status = await coordinator.get_task_status('task_001')
    if status.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        break
    await asyncio.sleep(1)
```

---

## Configuration Templates

### Boiler Control (Modbus)
```python
config = BoilerControlConfig(
    protocol=BoilerProtocol.MODBUS_TCP,
    host="192.168.1.100",
    port=502,
    unit_id=1,
    timeout=30,
    retry_count=3,
    retry_delay=5,
    tls_enabled=False,
    scan_rate=1,               # seconds
    max_write_rate=10,         # per minute
    safety_interlocks_enabled=True
)
```

### Boiler Control (OPC UA)
```python
config = BoilerControlConfig(
    protocol=BoilerProtocol.OPC_UA,
    host="192.168.1.101",
    port=4840,
    tls_enabled=True,
    cert_path="/path/to/client.crt",
    key_path="/path/to/client.key",
    username="boiler_user",
    password=os.getenv("OPCUA_PASSWORD")
)
```

### SCADA (OPC UA)
```python
config = SCADAConnectionConfig(
    protocol=SCADAProtocol.OPC_UA,
    host="scada.plant.com",
    port=4840,
    tls_enabled=True,
    cert_path="/path/to/cert.pem",
    ca_path="/path/to/ca.pem",
    timeout=30,
    retry_count=3,
    buffer_size=10000,
    health_check_interval=30
)
```

### SCADA (MQTT)
```python
config = SCADAConnectionConfig(
    protocol=SCADAProtocol.MQTT,
    host="mqtt.plant.com",
    port=8883,
    tls_enabled=True,
    username="scada_user",
    password=os.getenv("MQTT_PASSWORD")
)
```

### CEMS
```python
config = CEMSConfig(
    protocol='REST_API',
    host='cems.epa.gov',
    port=443,
    api_key=os.getenv('CEMS_API_KEY'),
    tls_enabled=True,
    compliance_standards=[
        ComplianceStandard.EPA_PART_75,
        ComplianceStandard.EPA_MATS
    ]
)
```

---

## Data Quality Scoring

### Formula
```
Quality Score = 40*Validity + 30*Completeness + 20*Consistency + 10*Uniqueness

Where:
  Validity (0-100):     % of values passing schema/range validation
  Completeness (0-100): % of non-null values
  Consistency (0-100):  % of consistent data types
  Uniqueness (0-100):   % of unique values (no duplicates)

Result: 0-100 overall quality score
```

### Interpretation
- **90-100:** Excellent - High quality data, safe for critical decisions
- **80-89:** Good - Suitable for most operational decisions
- **70-79:** Fair - May need additional validation
- **50-69:** Poor - Significant quality issues
- **<50:** Rejected - Data not suitable for use

---

## Security Quick Reference

### Authentication Methods
| System | Method | Credential Source |
|--------|--------|------------------|
| Modbus TCP | Unit ID | Config |
| OPC UA | Certificate + Password | Certs + Environment |
| MQTT | Username/Password | Environment |
| REST API | API Key | Environment |
| CEMS | API Key | Environment |

### All Credentials from Environment Variables
```bash
export BOILER_PASSWORD="..."
export SCADA_PASSWORD="..."
export MQTT_PASSWORD="..."
export CEMS_API_KEY="..."
export OPCUA_PASSWORD="..."
```

### TLS Configuration
```python
# All modules support TLS 1.3
config = SomeConfig(
    tls_enabled=True,
    cert_path="/path/to/client.crt",
    key_path="/path/to/client.key",
    ca_path="/path/to/ca.pem"
)
```

---

## Common Patterns

### Pattern 1: Read-Check-Write
```python
# Read current value
current = await connector.read_parameter(param)

# Validate change
is_valid, error = connector.validate_and_write(param, new_value)

# Write if valid
if is_valid:
    success = await connector.write_setpoint(param, new_value)
else:
    print(f"Validation failed: {error}")
```

### Pattern 2: Multi-Source Aggregation
```python
# Collect from multiple sources
scada_data = await scada.get_recent_data()
boiler_data = await boiler.read_all_parameters()
fuel_data = await fuel.get_status()

# Merge with quality scoring
merged = {
    **scada_data,
    **boiler_data,
    **fuel_data
}

# Calculate overall quality
quality_score = transformer.calculate_quality_score(merged)
```

### Pattern 3: Error Handling
```python
try:
    data = await connector.read_data()
except ConnectionError:
    logger.error("Connection failed, using cached data")
    data = await cache.get_latest()
except ValueError as e:
    logger.error(f"Data validation failed: {e}")
    data = None
finally:
    # Cleanup resources
    pass
```

### Pattern 4: Async Loops
```python
async def monitor_continuously():
    while True:
        try:
            data = await connector.read_data()
            await process(data)
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        finally:
            await asyncio.sleep(1)  # Scan interval

# Run in background
task = asyncio.create_task(monitor_continuously())

# Later: stop monitoring
task.cancel()
```

---

## Performance Tips

### 1. Connection Pooling
```python
# Reuse connections - don't create new ones each time
manager = BoilerControlManager()
await manager.add_connector("boiler_1", config)
# Reuse manager instance
```

### 2. Reduce Scan Rates
```python
# Lower scan rate = less network traffic
# 1000ms = 1 Hz (reasonable for most systems)
tag.scan_rate = 1000  # milliseconds

# Use deadband to prevent noisy updates
tag.deadband = 1.0    # Only update if change > 1.0
```

### 3. Buffer Management
```python
# Circular buffers have fixed memory
# 10,000 records ~ 5-10 MB
scada_buffer = SCADADataBuffer(
    max_size=10000,
    retention_hours=24
)
# Automatically removes oldest data
```

### 4. Data Transformation
```python
# Batch operations are faster
# Transform multiple points together
results = transformer.transform_batch(data_list)

# vs processing one at a time
for point in data_list:
    transformer.transform(point)  # Slower
```

---

## Troubleshooting

### Connection Failed
```python
# Check configuration
print(f"Host: {config.host}")
print(f"Port: {config.port}")

# Check credentials
print(f"Username: {config.username}")
# Password should be in environment, not printed

# Check connectivity
# Verify firewall allows connection
# Verify service is running on target
```

### Data Quality Low
```python
# Check for outliers
issues = transformer.detect_outliers(data)

# Check for missing data
missing_pct = (sum(1 for p in data if p.value is None) / len(data)) * 100

# Check for inconsistent types
for point in data:
    if not isinstance(point.value, (int, float)):
        print(f"Type error: {point}")

# Increase sample size
data = await scada.get_recent_data(minutes=60)  # More data
```

### Slow Performance
```python
# Check scan rates
# Reduce if possible
config.scan_rate = 5000  # Increase interval

# Check buffer size
# If too large, reduce it
buffer = SCADADataBuffer(max_size=1000)  # Smaller

# Check for errors
# Errors trigger retries which slow things down
logger.setLevel(logging.ERROR)  # Focus on errors
```

---

## File Reference

**Location:** `C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation/agents/GL-002/integrations/`

| File | Purpose | Key Classes |
|------|---------|------------|
| `__init__.py` | Module init | All exports |
| `boiler_control_connector.py` | Boiler control | BoilerControlManager, etc. |
| `fuel_management_connector.py` | Fuel systems | FuelManagementConnector, etc. |
| `scada_connector.py` | SCADA data | SCADAConnector, AlarmManager, etc. |
| `emissions_monitoring_connector.py` | CEMS/compliance | EmissionsMonitoringConnector, etc. |
| `data_transformers.py` | Data processing | DataTransformationPipeline, etc. |
| `agent_coordinator.py` | Agent comms | AgentCoordinator, MessageBus, etc. |
| `README.md` | Integration guide | Setup instructions |
| `INTEGRATION_VALIDATION_REPORT.md` | Detailed spec | Module specs |
| `MODULES_ARCHITECTURE.md` | Architecture | Design details |
| `QUICK_REFERENCE.md` | This file | Quick lookups |

---

## Support Resources

### Documentation
- `INTEGRATION_VALIDATION_REPORT.md` - Complete specifications
- `MODULES_ARCHITECTURE.md` - Architecture details
- `README.md` - Integration guide
- Source code - Type hints and docstrings

### Key Concepts
- **Async Operations:** All I/O is non-blocking
- **Data Quality:** Every point has 0-100 quality score
- **Safety:** Boiler writes have safety interlocks
- **Compliance:** Emissions track regulatory standards
- **Scalability:** Supports 100+ connections

### Common Tasks
1. **Connect to boiler:** `BoilerControlManager` + `BoilerControlConfig`
2. **Monitor SCADA:** `SCADAConnector` + tag registration
3. **Check emissions:** `EmissionsMonitoringConnector` + compliance check
4. **Transform data:** `DataTransformationPipeline` + unit conversion
5. **Coordinate agents:** `AgentCoordinator` + message bus

---

**Last Updated:** 2025-11-15
**Status:** APPROVED FOR PRODUCTION
**Version:** 1.0.0

For detailed information, see `INTEGRATION_VALIDATION_REPORT.md` and `MODULES_ARCHITECTURE.md`.
