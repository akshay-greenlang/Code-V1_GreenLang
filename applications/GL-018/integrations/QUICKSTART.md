# SCADA Integration Quick Start Guide

Get up and running with flue gas analyzer integration in 5 minutes.

## 1. Install Dependencies

```bash
pip install asyncua>=1.0.0 pymodbus>=3.0.0 pydantic>=2.0.0
```

## 2. Choose Your Analyzer

### OPC-UA Analyzers (Most Common)

```python
from integrations.scada_integration import create_scada_client, AnalyzerType, ConnectionProtocol

# ABB AO2000
client = create_scada_client(
    analyzer_type=AnalyzerType.ABB_AO2000,
    protocol=ConnectionProtocol.OPC_UA,
    host="192.168.1.100",
    port=4840,
)

# SICK MARSIC
client = create_scada_client(
    analyzer_type=AnalyzerType.SICK_MARSIC,
    protocol=ConnectionProtocol.OPC_UA,
    host="192.168.1.101",
    port=4840,
)

# Emerson Rosemount X-STREAM
client = create_scada_client(
    analyzer_type=AnalyzerType.EMERSON_XSTREAM,
    protocol=ConnectionProtocol.OPC_UA,
    host="192.168.1.102",
    port=4840,
)
```

### Modbus TCP Analyzers

```python
# Horiba PG series
client = create_scada_client(
    analyzer_type=AnalyzerType.HORIBA_PG,
    protocol=ConnectionProtocol.MODBUS_TCP,
    host="192.168.1.104",
    port=502,
    modbus_unit_id=1,
)

# Fuji Electric ZRJ
client = create_scada_client(
    analyzer_type=AnalyzerType.FUJI_ZRJ,
    protocol=ConnectionProtocol.MODBUS_RTU,
    host="/dev/ttyUSB0",  # Serial port
    modbus_unit_id=1,
)
```

## 3. Register Tags

```python
from integrations.scada_integration import create_standard_flue_gas_tags

# Use standard tags (works with most analyzers)
tags = create_standard_flue_gas_tags()
client.register_tags(tags)

# Or use analyzer-specific tags
from integrations.scada_integration import create_abb_ao2000_tags
abb_tags = create_abb_ao2000_tags()
client.register_tags(abb_tags)
```

## 4. Connect and Read

```python
import asyncio

async def main():
    # Connect
    await client.connect()

    # Read O2
    o2 = await client.read_tag("FG_O2_STACK")
    print(f"O2: {o2.value:.2f} %")

    # Read CO
    co = await client.read_tag("FG_CO_STACK")
    print(f"CO: {co.value:.1f} ppm")

    # Read multiple tags
    tags = ["FG_O2_STACK", "FG_CO_STACK", "FG_NOX_STACK"]
    results = await client.read_tags(tags)

    for tag_name, data in results.items():
        print(f"{tag_name}: {data.value:.2f} {data.engineering_unit}")

    # Disconnect
    await client.disconnect()

asyncio.run(main())
```

## 5. Real-time Monitoring (Optional)

```python
async def monitor():
    await client.connect()

    # Define callback
    def on_value_change(data_point):
        print(f"{data_point.tag_name}: {data_point.value:.2f} {data_point.engineering_unit}")

    # Subscribe to changes
    await client.subscribe_tag("FG_O2_STACK", on_value_change)
    await client.subscribe_tag("FG_CO_STACK", on_value_change)

    # Monitor for 60 seconds
    await asyncio.sleep(60)

    await client.disconnect()
```

## 6. Combustion Control (Optional)

```python
async def control_air_fuel():
    await client.connect()

    # Read current O2
    o2 = await client.read_tag("FG_O2_STACK")

    # Adjust air damper to reach target O2
    target_o2 = 3.5  # % (optimal for natural gas)

    if o2.value > target_o2 + 0.5:
        # Too much air - close damper
        await client.write_tag("AIR_DAMPER_POS", 45.0)  # 45%
        print("Reducing air flow")

    elif o2.value < target_o2 - 0.5:
        # Too little air - open damper
        await client.write_tag("AIR_DAMPER_POS", 55.0)  # 55%
        print("Increasing air flow")

    await client.disconnect()
```

## Complete Example

```python
"""
Complete flue gas analyzer integration example.
Monitors O2, CO, and NOx with alarm handling.
"""

import asyncio
import logging
from integrations.scada_integration import (
    create_scada_client,
    AnalyzerType,
    ConnectionProtocol,
    create_standard_flue_gas_tags,
    AlarmSeverity,
)

logging.basicConfig(level=logging.INFO)

async def main():
    # Create client
    client = create_scada_client(
        analyzer_type=AnalyzerType.ABB_AO2000,
        protocol=ConnectionProtocol.OPC_UA,
        host="192.168.1.100",
        port=4840,
        username="operator",
        password="password",  # Use env var in production!
    )

    # Register tags
    tags = create_standard_flue_gas_tags()
    client.register_tags(tags)

    try:
        # Connect
        print("Connecting to analyzer...")
        await client.connect()
        print("Connected!")

        # Read all gas concentrations
        print("\nCurrent readings:")
        o2 = await client.read_tag("FG_O2_STACK")
        co = await client.read_tag("FG_CO_STACK")
        nox = await client.read_tag("FG_NOX_STACK")

        print(f"  O2:  {o2.value:.2f} %")
        print(f"  CO:  {co.value:.1f} ppm")
        print(f"  NOx: {nox.value:.1f} ppm")

        # Check for alarms
        alarms = client.get_active_alarms()
        if alarms:
            print(f"\nActive alarms: {len(alarms)}")
            for alarm in alarms:
                print(f"  [{alarm.severity.value}] {alarm.message}")

                # Acknowledge critical alarms
                if alarm.severity == AlarmSeverity.CRITICAL:
                    await client.acknowledge_alarm(
                        alarm.alarm_id,
                        acknowledged_by="operator@example.com",
                        notes="Acknowledged via quickstart script"
                    )
                    print(f"    ✓ Acknowledged")

        # Get statistics
        stats = client.get_statistics()
        print(f"\nStatistics:")
        print(f"  Reads: {stats['reads']}")
        print(f"  Writes: {stats['writes']}")
        print(f"  Errors: {stats['errors']}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await client.disconnect()
        print("\nDisconnected")

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

1. **Configure your analyzer** - Update IP address and credentials
2. **Customize tags** - Add analyzer-specific tag names
3. **Add alarm handling** - Implement your alarm response logic
4. **Enable control** - Implement air/fuel ratio optimization
5. **Log data** - Send data to time-series database (InfluxDB, TimescaleDB)

## Troubleshooting

### Connection Failed

```python
# Check connection with health check
health = await client.health_check()
print(health)
```

### Wrong Tag Names

```python
# List all registered tags
all_tags = client.get_all_tags()
for tag in all_tags:
    print(f"{tag.tag_name}: {tag.parameter_type.value}")
```

### Read Timeout

```python
# Increase timeout
from integrations.scada_integration import SCADAConfig

config = SCADAConfig(
    protocol=ConnectionProtocol.OPC_UA,
    host="192.168.1.100",
    connection_timeout=30.0,  # Increase to 30 seconds
)
```

## Support

Need help? Check:
- Full documentation: `integrations/README_SCADA.md`
- Examples: `examples/scada_integration_examples.py`
- Tests: `tests/test_scada_integration.py`
- Configuration: `config/analyzer_configs.yaml`
