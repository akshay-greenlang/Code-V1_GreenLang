# GL-020 ECONOPULSE - Quick Start Guide

Get up and running with economizer data integration in 5 minutes.

## Prerequisites

- Python 3.10+
- Network access to SCADA/DCS system
- Historian credentials (if using historian connectors)

## Installation

```bash
pip install httpx pymodbus asyncua pandas
```

## Quick Start Examples

### 1. Connect to SCADA and Read Economizer Data

```python
import asyncio
from datetime import datetime

# Import integration modules
from integrations import (
    SCADAClient,
    SCADAConfig,
    SCADAProtocol,
    EconomizerTagGroups,
)


async def main():
    # Configure SCADA connection
    config = SCADAConfig(
        protocol=SCADAProtocol.OPC_UA,
        host="192.168.1.100",  # Your SCADA server IP
        port=4840,
        application_name="GL-020-ECONOPULSE",
    )

    # Create and connect
    client = SCADAClient(config)
    connected = await client.connect()

    if not connected:
        print("Failed to connect to SCADA")
        return

    print("Connected to SCADA!")

    # Subscribe to economizer tags
    client.subscribe_group(EconomizerTagGroups.feedwater_temperatures())
    client.subscribe_group(EconomizerTagGroups.flue_gas_temperatures())
    client.subscribe_group(EconomizerTagGroups.differential_pressures())

    # Read current values
    tag_names = [
        "ECON.FW.TEMP.INLET",
        "ECON.FW.TEMP.OUTLET",
        "ECON.FG.TEMP.INLET",
        "ECON.FG.TEMP.OUTLET",
        "ECON.DP.GASSIDE",
    ]

    values = await client.read_multiple_tags(tag_names)

    print("\n--- Economizer Status ---")
    for tag_name, tag_value in values.items():
        print(f"{tag_name}: {tag_value.value:.2f} ({tag_value.quality.value})")

    # Clean up
    await client.close()


# Run
asyncio.run(main())
```

### 2. Monitor Soot Blower Status

```python
import asyncio
from integrations import (
    SootBlowerController,
    SootBlowerConfig,
    CleaningZone,
    CleaningMediaType,
)


async def main():
    # Configure soot blower controller
    config = SootBlowerConfig(
        controller_id="ECON_SB_01",
        controller_name="Economizer Soot Blower",
        protocol="opc_ua",
        host="192.168.1.100",
        port=4840,
        media_type=CleaningMediaType.STEAM,
    )

    # Create controller
    controller = SootBlowerController(config)
    await controller.connect()

    # Register zones
    controller.register_zone(CleaningZone(
        zone_id="ZONE_A",
        name="Economizer Zone A",
        description="Upper tube bank",
        blower_ids=["SB_01", "SB_02"],
    ))

    controller.register_zone(CleaningZone(
        zone_id="ZONE_B",
        name="Economizer Zone B",
        description="Lower tube bank",
        blower_ids=["SB_03", "SB_04"],
    ))

    # Check blower statuses
    print("\n--- Soot Blower Status ---")
    statuses = await controller.get_all_blower_statuses()
    for blower_id, status in statuses.items():
        print(f"{blower_id}: {status.value}")

    # Check interlocks
    all_clear, active_interlocks = await controller.check_interlocks()
    print(f"\nInterlocks Clear: {all_clear}")
    if not all_clear:
        for interlock in active_interlocks:
            print(f"  Active: {interlock.interlock_type.value}")

    # Get system status
    status = controller.get_system_status()
    print(f"\nSystem Status: {status}")

    await controller.close()


asyncio.run(main())
```

### 3. Validate Sensor Data Quality

```python
from datetime import datetime
from integrations import (
    DataQualityValidator,
    RangeCheck,
    RateOfChangeLimit,
    QualityFlag,
)


def main():
    # Create validator
    validator = DataQualityValidator()

    # Configure range checks for feedwater temperature
    validator.configure_range_check(RangeCheck(
        tag_name="FW_TEMP_INLET",
        low_limit=40.0,
        high_limit=180.0,
        low_low_limit=20.0,
        high_high_limit=200.0,
        engineering_unit="C",
    ))

    # Configure rate-of-change limit
    validator.configure_roc_limit(RateOfChangeLimit(
        tag_name="FW_TEMP_INLET",
        max_rate_per_second=2.0,
        max_rate_per_minute=50.0,
    ))

    # Simulate sensor readings
    readings = [
        (165.0, "2024-01-01 10:00:00"),
        (165.5, "2024-01-01 10:00:01"),
        (166.2, "2024-01-01 10:00:02"),
        (185.0, "2024-01-01 10:00:03"),  # Out of range
        (166.8, "2024-01-01 10:00:04"),
    ]

    print("--- Data Quality Validation ---\n")

    for value, timestamp_str in readings:
        timestamp = datetime.fromisoformat(timestamp_str)
        result = validator.validate(
            tag_name="FW_TEMP_INLET",
            value=value,
            timestamp=timestamp,
        )

        print(f"Value: {value:.1f} C")
        print(f"  Quality: {result.quality_flag.value}")
        print(f"  Confidence: {result.confidence_score.total_score:.1f}%")
        print(f"  Usable: {result.is_usable}")

        for check_name, check_result, message in result.validation_results:
            if message:
                print(f"  {check_name}: {check_result.value} - {message}")
        print()


main()
```

### 4. Use Redundant Sensor Voting

```python
from integrations import (
    RedundancyManager,
    RedundantSensorGroup,
    VotingMethod,
    QualityFlag,
)


def main():
    # Create redundancy manager
    manager = RedundancyManager()

    # Register redundant temperature sensors
    manager.register_group(RedundantSensorGroup(
        group_name="FW_INLET_TEMP",
        sensor_tags=["TEMP_A", "TEMP_B", "TEMP_C"],
        voting_method=VotingMethod.TWO_OUT_OF_THREE,
        max_deviation_percent=2.0,
    ))

    # Scenario 1: All sensors agree
    print("--- Scenario 1: All Agree ---")
    sensor_values = {
        "TEMP_A": (165.2, QualityFlag.GOOD),
        "TEMP_B": (165.5, QualityFlag.GOOD),
        "TEMP_C": (165.3, QualityFlag.GOOD),
    }

    value, quality, reason = manager.vote("FW_INLET_TEMP", sensor_values)
    print(f"Result: {value:.2f} C")
    print(f"Quality: {quality.value}")
    print(f"Reason: {reason}\n")

    # Scenario 2: One sensor is outlier
    print("--- Scenario 2: One Outlier ---")
    sensor_values = {
        "TEMP_A": (165.2, QualityFlag.GOOD),
        "TEMP_B": (165.5, QualityFlag.GOOD),
        "TEMP_C": (180.0, QualityFlag.GOOD),  # Outlier
    }

    value, quality, reason = manager.vote("FW_INLET_TEMP", sensor_values)
    print(f"Result: {value:.2f} C")
    print(f"Quality: {quality.value}")
    print(f"Reason: {reason}\n")

    # Scenario 3: One sensor failed
    print("--- Scenario 3: One Failed ---")
    sensor_values = {
        "TEMP_A": (165.2, QualityFlag.GOOD),
        "TEMP_B": (165.5, QualityFlag.GOOD),
        "TEMP_C": (0.0, QualityFlag.BAD_SENSOR_FAILURE),
    }

    value, quality, reason = manager.vote("FW_INLET_TEMP", sensor_values)
    print(f"Result: {value:.2f} C")
    print(f"Quality: {quality.value}")
    print(f"Reason: {reason}")


main()
```

### 5. Read Historical Data from PI

```python
import asyncio
from datetime import datetime, timedelta
from integrations import (
    OSIsoftPIConnector,
    HistorianConfig,
    AggregateType,
)


async def main():
    # Configure PI connection
    config = HistorianConfig(
        name="PI_SERVER",
        host="pi-server.domain.com",
        port=443,
        use_ssl=True,
        username="pi_user",
        password="pi_password",  # Use vault in production!
    )

    # Create and connect
    pi = OSIsoftPIConnector(config, use_web_api=True)
    await pi.connect()

    # Define time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)

    # Read hourly averages
    print("--- Hourly Temperature Averages (Last 24h) ---\n")

    data = await pi.read_aggregated(
        tag_names=["ECON.FW.TEMP.INLET", "ECON.FW.TEMP.OUTLET"],
        start_time=start_time,
        end_time=end_time,
        interval_seconds=3600,
        aggregate_type=AggregateType.AVERAGE,
    )

    for tag_name, ts_data in data.items():
        print(f"{tag_name}:")
        for point in ts_data.points[-6:]:  # Last 6 hours
            print(f"  {point.timestamp.strftime('%H:%M')}: {point.value:.1f} C")
        print()

    # Calculate heat recovery
    if "ECON.FW.TEMP.INLET" in data and "ECON.FW.TEMP.OUTLET" in data:
        inlet_data = data["ECON.FW.TEMP.INLET"]
        outlet_data = data["ECON.FW.TEMP.OUTLET"]

        print("--- Heat Recovery ---")
        for inlet, outlet in zip(inlet_data.points[-6:], outlet_data.points[-6:]):
            delta_t = outlet.value - inlet.value
            print(f"  {inlet.timestamp.strftime('%H:%M')}: Delta-T = {delta_t:.1f} C")

    await pi.close()


asyncio.run(main())
```

## Common Patterns

### Error Handling Pattern

```python
async def safe_read(client, tag_name):
    """Safely read a tag with fallback."""
    try:
        value = await client.read_tag(tag_name)
        if value and value.is_good():
            return value.value
        else:
            # Use cached value
            cached = client.get_cached_value(tag_name)
            return cached.value if cached else None
    except Exception as e:
        print(f"Read error: {e}")
        return None
```

### Callback Pattern for Real-Time Updates

```python
from collections import deque

# Store recent values
recent_values = deque(maxlen=100)

def on_value_change(tag_value):
    """Callback for value updates."""
    recent_values.append({
        "tag": tag_value.tag_name,
        "value": tag_value.value,
        "time": tag_value.timestamp,
    })

    # Check for alarm conditions
    if tag_value.tag_name == "ECON.FW.TEMP.OUTLET":
        if tag_value.value > 240.0:
            print(f"ALARM: High outlet temperature: {tag_value.value} C")

# Subscribe with callback
client.subscribe(tag_subscription, callback=on_value_change)
```

### Batch Processing Pattern

```python
async def process_batch(client, tag_names, batch_size=50):
    """Process tags in batches."""
    results = {}

    for i in range(0, len(tag_names), batch_size):
        batch = tag_names[i:i + batch_size]
        batch_results = await client.read_multiple_tags(batch)
        results.update(batch_results)

        # Small delay between batches
        await asyncio.sleep(0.1)

    return results
```

## Troubleshooting

### Connection Issues

```python
# Check connection status
print(f"Connected: {client.is_connected}")
print(f"Circuit breaker: {client._circuit_breaker.state.name}")

# Get detailed status
status = client.get_connection_status()
print(status)
```

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("integrations").setLevel(logging.DEBUG)
```

### Test Connectivity

```python
async def test_connectivity():
    """Test SCADA connectivity."""
    config = SCADAConfig(
        protocol=SCADAProtocol.OPC_UA,
        host="192.168.1.100",
        port=4840,
    )

    client = SCADAClient(config)

    try:
        connected = await client.connect()
        print(f"Connection: {'SUCCESS' if connected else 'FAILED'}")

        if connected:
            # Test read
            value = await client.read_tag("ECON.FW.TEMP.INLET")
            print(f"Test read: {value}")

    finally:
        await client.close()
```

## Next Steps

1. Review [README_INTEGRATIONS.md](README_INTEGRATIONS.md) for detailed documentation
2. Configure range checks for your specific sensors
3. Set up redundancy groups for critical measurements
4. Configure historian integration for long-term data storage
5. Implement soot blower optimization logic

## Support

- Documentation: [README_INTEGRATIONS.md](README_INTEGRATIONS.md)
- Issues: Create an issue in the GL-020 repository
- Email: greenlang-support@example.com
