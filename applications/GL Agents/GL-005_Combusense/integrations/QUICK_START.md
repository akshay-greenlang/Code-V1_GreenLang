# GL-005 Integration Connectors - Quick Start Guide

## 5-Minute Setup Guide

### Step 1: Install Dependencies
```bash
pip install asyncua pymodbus paho-mqtt prometheus-client pydantic numpy
```

### Step 2: Import Connectors
```python
from integrations import (
    DCSConnector, DCSConfig,
    PLCConnector, PLCConfig,
    CombustionAnalyzerConnector, AnalyzerConfig,
    FlameScannerConnector, FlameScannerConfig,
    TemperatureSensorArrayConnector, SensorArrayConfig,
    SCADAIntegration, SCADAConfig
)
```

### Step 3: Configure & Connect

#### DCS (Distributed Control System)
```python
async def connect_dcs():
    config = DCSConfig(
        opcua_endpoint="opc.tcp://dcs.plant.com:4840",
        modbus_host="10.0.1.100"
    )

    async with DCSConnector(config) as dcs:
        # Read process variables
        values = await dcs.read_process_variables(["FurnaceTemp"])
        print(f"Furnace: {values['FurnaceTemp']['value']}°C")
```

#### PLC (Programmable Logic Controller)
```python
async def connect_plc():
    config = PLCConfig(
        tcp_host="10.0.1.50",
        tcp_port=502
    )

    async with PLCConnector(config) as plc:
        # Read digital inputs
        inputs = await plc.read_coils(["BurnerOn"])
        print(f"Burner: {'ON' if inputs['BurnerOn'] else 'OFF'}")
```

#### Gas Analyzer
```python
async def connect_analyzer():
    config = AnalyzerConfig(
        analyzer_id="O2_01",
        mqtt_broker="mqtt.plant.com",
        gases_measured=[GasType.O2]
    )

    async with CombustionAnalyzerConnector(config) as analyzer:
        # Read O2 level
        o2 = await analyzer.read_o2_level()
        print(f"O2: {o2}%")
```

#### Flame Scanner
```python
async def connect_scanner():
    config = FlameScannerConfig(
        scanner_id="SCANNER_01",
        scanner_type=ScannerType.UV_DETECTOR,
        modbus_host="10.0.1.60"
    )

    async with FlameScannerConnector(config) as scanner:
        # Detect flame
        flame = await scanner.detect_flame_presence()
        print(f"Flame: {'PRESENT' if flame else 'ABSENT'}")
```

#### Temperature Sensors
```python
async def connect_temp_sensors():
    config = SensorArrayConfig(
        array_id="TEMP_ARRAY",
        serial_port="/dev/ttyUSB0"
    )

    async with TemperatureSensorArrayConnector(config) as array:
        # Read furnace temperature
        temp = await array.read_furnace_temperature()
        print(f"Furnace: {temp}°C")
```

#### SCADA Integration
```python
async def connect_scada():
    config = SCADAConfig(
        opcua_endpoint="opc.tcp://0.0.0.0:4840",
        mqtt_broker="mqtt.plant.com"
    )

    async with SCADAIntegration(config) as scada:
        # Publish data
        await scada.publish_real_time_data({
            "FurnaceTemp": 850.5,
            "O2Content": 3.5
        })
```

---

## Common Patterns

### Pattern 1: Read-Process-Write Loop
```python
async def control_loop():
    async with DCSConnector(dcs_config) as dcs, \
               PLCConnector(plc_config) as plc:

        while True:
            # Read sensors
            temp = await dcs.read_process_variables(["FurnaceTemp"])

            # Control logic
            if temp["FurnaceTemp"]["value"] > 900:
                # Turn off burner
                await plc.write_coils({"BurnerOn": False})

            await asyncio.sleep(1)  # 1Hz control loop
```

### Pattern 2: Multi-Sensor Data Collection
```python
async def collect_all_data():
    async with (
        DCSConnector(dcs_config) as dcs,
        CombustionAnalyzerConnector(analyzer_config) as analyzer,
        TemperatureSensorArrayConnector(temp_config) as temp_array
    ):
        # Collect all data in parallel
        dcs_data, gas_data, temp_data = await asyncio.gather(
            dcs.read_process_variables(["FurnaceTemp", "SteamPressure"]),
            analyzer.read_all_gases(),
            temp_array.read_all_zones()
        )

        return {**dcs_data, **gas_data, **temp_data}
```

### Pattern 3: Alarm Monitoring
```python
async def monitor_alarms():
    async with DCSConnector(dcs_config) as dcs:
        async def alarm_handler(alarm):
            print(f"ALARM: {alarm.message} (priority={alarm.priority})")
            # Send notification, log to database, etc.

        await dcs.subscribe_to_alarms(alarm_handler)

        # Keep running
        await asyncio.Event().wait()
```

---

## Troubleshooting

### Issue: Connection Timeout
```python
# Increase timeout
config = DCSConfig(
    opcua_endpoint="opc.tcp://dcs.plant.com:4840",
    connection_timeout=60  # Increase from 30 to 60 seconds
)
```

### Issue: Modbus CRC Errors
```python
# Reduce baud rate for noisy environments
config = PLCConfig(
    rtu_port="/dev/ttyUSB0",
    baudrate=4800  # Reduce from 9600
)
```

### Issue: MQTT Connection Refused
```python
# Check credentials and TLS settings
config = AnalyzerConfig(
    mqtt_broker="mqtt.plant.com",
    mqtt_username="user",
    mqtt_password="pass",
    mqtt_use_tls=True
)
```

---

## Performance Tips

1. **Use Connection Pooling**
   - Reuse connector instances
   - Don't create/destroy frequently

2. **Batch Operations**
   - Read multiple tags in one call
   - Use batch writes when possible

3. **Adjust Scan Rates**
   - Match to process requirements
   - Slower rates = less CPU/network

4. **Enable Compression**
   - For SCADA/MQTT with large payloads

---

## Next Steps

1. Read full documentation: `README.md`
2. Review examples: `example_usage.py`
3. Check implementation details: `IMPLEMENTATION_SUMMARY.md`
4. Deploy to development environment
5. Run integration tests

---

**Support:** support@greenlang.com
