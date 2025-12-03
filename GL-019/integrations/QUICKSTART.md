# GL-019 HEATSCHEDULER Integration Quick Start Guide

Get up and running with enterprise system integrations in 10 minutes.

## Prerequisites

```bash
# Core dependencies
pip install pydantic>=2.0.0 httpx>=0.24.0

# Optional: SAP RFC (requires SAP NW RFC SDK)
pip install pyrfc

# Optional: OPC UA
pip install asyncua>=1.0.0

# Optional: Modbus
pip install pymodbus>=3.0.0
```

## 1. ERP Integration

### SAP S/4HANA

```python
import asyncio
from datetime import datetime, timedelta
from integrations.erp_connector import (
    create_erp_connector,
    ERPSystem,
    WorkOrderStatus,
)

async def main():
    # Create connector
    connector = create_erp_connector(
        erp_system=ERPSystem.SAP_S4HANA,
        host="sap.company.com",
        username="INTEGRATION_USER",
        password="secure_password",  # Use environment variable!
        system_id="PRD",
        client_id="100",
    )

    try:
        # Connect
        print("Connecting to SAP...")
        await connector.connect()
        print("Connected!")

        # Get production schedules
        schedules = await connector.get_production_schedules(
            plant_code="1000",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            include_heating_only=True,
        )

        print(f"\nFound {len(schedules)} schedules")
        for schedule in schedules:
            print(f"\nSchedule: {schedule.schedule_id}")
            print(f"  Items: {len(schedule.items)}")
            print(f"  Heating operations: {schedule.heating_operations}")

            for item in schedule.items[:5]:  # First 5 items
                print(f"    - {item.item_id}: {item.description}")
                if item.heating_required:
                    print(f"      Temperature: {item.temperature_setpoint_c}C")
                    print(f"      Power: {item.estimated_power_kw} kW")

        # Get work orders
        work_orders = await connector.get_work_orders(
            plant_code="1000",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=3),
            status_filter=[WorkOrderStatus.RELEASED, WorkOrderStatus.IN_PROGRESS],
        )

        print(f"\nActive work orders: {len(work_orders)}")

        # Get maintenance schedules
        maintenance = await connector.get_maintenance_schedules(
            plant_code="1000",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=14),
            heating_equipment_only=True,
        )

        print(f"Maintenance scheduled: {len(maintenance)}")

        # Check health
        health = await connector.health_check()
        print(f"\nHealth: {'OK' if health['healthy'] else 'ERROR'}")

    finally:
        await connector.disconnect()
        print("\nDisconnected")

asyncio.run(main())
```

### Oracle ERP Cloud

```python
from integrations.erp_connector import create_erp_connector, ERPSystem

connector = create_erp_connector(
    erp_system=ERPSystem.ORACLE_CLOUD,
    host="your-tenant.oraclecloud.com",
    username="integration_user",
    password="secure_password",
    base_url="https://your-tenant.oraclecloud.com/fscmRestApi",
)

await connector.connect()
# Same API as SAP connector
```

## 2. Energy Management Integration

### Real-Time Pricing

```python
import asyncio
from integrations.energy_management_connector import (
    create_pricing_connector,
    ISOMarket,
    PriceType,
)

async def main():
    # Create pricing connector
    pricing = create_pricing_connector(
        iso_market=ISOMarket.PJM,
        pricing_node="PJMISO",
        api_key="your_api_key",  # Get from PJM Data Miner
    )

    await pricing.connect()

    # Get current price
    price = await pricing.get_current_price(PriceType.LMP)
    print(f"Current LMP: ${price.price:.2f}/MWh")

    # Get 24-hour forecast
    forecast = await pricing.get_price_forecast(hours_ahead=24)
    print("\n24-Hour Price Forecast:")
    for timestamp, rate, period in forecast:
        print(f"  {timestamp.strftime('%H:%M')}: ${rate:.2f}/MWh ({period.value})")

    # Subscribe to updates
    def on_price_update(price_point):
        print(f"Price update: ${price_point.price:.2f}/MWh")

    await pricing.subscribe_to_prices(on_price_update)

    # Get statistics
    stats = pricing.get_price_statistics(hours=24)
    print(f"\n24h Stats: Min=${stats['min']:.2f}, Max=${stats['max']:.2f}, Avg=${stats['avg']:.2f}")

    await pricing.disconnect()

asyncio.run(main())
```

### Demand Response

```python
from integrations.energy_management_connector import (
    create_demand_response_connector,
    DemandResponseLevel,
)

async def main():
    dr = create_demand_response_connector(
        ven_id="VEN_001",
        vtn_url="https://dr.utility.com/OpenADR",
        program_id="COMMERCIAL_DR_1",
        auto_opt_in=False,
        max_curtailment_kw=500,
    )

    await dr.connect()

    # Get current signal
    signal = await dr.get_current_signal()
    if signal:
        print(f"Current DR Signal: {signal.level.value}")
        if signal.level >= DemandResponseLevel.HIGH:
            print("  HIGH demand - consider reducing load!")

    # Get active events
    events = await dr.get_active_events()
    for event in events:
        print(f"\nDR Event: {event.event_id}")
        print(f"  Type: {event.event_type.value}")
        print(f"  Start: {event.start_time}")
        print(f"  End: {event.end_time}")
        print(f"  Target reduction: {event.target_reduction_kw} kW")

        # Opt in to event
        if event.response_required:
            await dr.opt_in_event(event.event_id)
            print("  Opted in!")

    # Subscribe to events
    async def on_dr_event(event):
        print(f"New DR event: {event.event_id}")

    await dr.subscribe_to_events(on_dr_event)

    await dr.disconnect()

asyncio.run(main())
```

### Energy Meters

```python
from integrations.energy_management_connector import (
    create_energy_meter_connector,
    MeterProtocol,
)

async def main():
    # Modbus TCP meter
    meter = create_energy_meter_connector(
        protocol=MeterProtocol.MODBUS_TCP,
        host="192.168.1.50",
        meter_id="MAIN_METER_01",
        port=502,
        modbus_unit_id=1,
        poll_interval_seconds=5.0,
    )

    await meter.connect()

    # Read current values
    reading = await meter.read_meter()
    print(f"Active Power: {reading.active_power_kw:.1f} kW")
    print(f"Power Factor: {reading.power_factor:.2f}")
    print(f"Total Energy: {reading.active_energy_kwh:.0f} kWh")

    # Start continuous polling
    def on_reading(reading):
        print(f"Power: {reading.active_power_kw:.1f} kW @ {reading.timestamp}")

    await meter.start_polling(callback=on_reading)

    # Let it run for a minute
    await asyncio.sleep(60)

    await meter.stop_polling()
    await meter.disconnect()

asyncio.run(main())
```

## 3. SCADA Integration

### OPC UA Connection

```python
import asyncio
from integrations.scada_integration import (
    create_scada_client,
    ConnectionProtocol,
    HeatingEquipment,
    EquipmentType,
    ControlSetpoint,
    OperatingMode,
    create_heating_equipment_tags,
)

async def main():
    # Create SCADA client
    client = create_scada_client(
        protocol=ConnectionProtocol.OPC_UA,
        host="192.168.1.100",
        port=4840,
        username="operator",
        password="password",
    )

    # Register equipment
    furnace = HeatingEquipment(
        equipment_id="FURNACE_01",
        equipment_name="Heat Treatment Furnace 1",
        equipment_type=EquipmentType.HEAT_TREATMENT_FURNACE,
        plant_code="1000",
        rated_power_kw=500,
        max_temperature_c=1200,
        min_temperature_c=100,
        startup_time_minutes=60,
        cooldown_time_minutes=120,
    )
    client.register_equipment(furnace)

    # Register tags
    tags = create_heating_equipment_tags(
        "FURNACE_01",
        EquipmentType.HEAT_TREATMENT_FURNACE
    )
    client.register_tags(tags)

    try:
        # Connect
        print("Connecting to SCADA...")
        await client.connect()
        print("Connected!")

        # Read status
        status = await client.read_equipment_status("FURNACE_01")
        print(f"\nFurnace Status: {status.status.value}")
        print(f"  Running: {status.running}")
        print(f"  Available: {status.available}")
        print(f"  Fault: {status.fault_active}")

        # Read temperature
        temp = await client.read_temperature("FURNACE_01")
        print(f"\nTemperature:")
        print(f"  Process: {temp.process_temperature_c}C")
        print(f"  Setpoint: {temp.setpoint_temperature_c}C")
        if temp.zone_temperatures:
            for zone, t in temp.zone_temperatures.items():
                print(f"  Zone {zone}: {t}C")

        # Read power
        power = await client.read_power("FURNACE_01")
        print(f"\nPower:")
        print(f"  Active: {power.active_power_kw} kW")
        print(f"  Energy today: {power.energy_today_kwh} kWh")

        # Write setpoint (if authorized)
        setpoint = ControlSetpoint(
            equipment_id="FURNACE_01",
            temperature_setpoint_c=850,
            power_limit_kw=400,
            operating_mode=OperatingMode.SCHEDULED,
        )
        success = await client.write_setpoint("FURNACE_01", setpoint)
        print(f"\nSetpoint applied: {success}")

        # Subscribe to updates
        def on_temp_change(reading):
            print(f"Temp update: {reading.process_temperature_c}C")

        await client.subscribe_temperature("FURNACE_01", on_temp_change)

        # Monitor for 60 seconds
        await asyncio.sleep(60)

        # Check alarms
        alarms = client.get_active_alarms()
        if alarms:
            print(f"\nActive Alarms: {len(alarms)}")
            for alarm in alarms:
                print(f"  [{alarm.severity.value}] {alarm.message}")

        # Get statistics
        stats = client.get_statistics()
        print(f"\nStatistics:")
        print(f"  Reads: {stats['reads']}")
        print(f"  Writes: {stats['writes']}")
        print(f"  Errors: {stats['errors']}")

    finally:
        await client.disconnect()
        print("\nDisconnected")

asyncio.run(main())
```

### Modbus TCP Connection

```python
client = create_scada_client(
    protocol=ConnectionProtocol.MODBUS_TCP,
    host="192.168.1.101",
    port=502,
    modbus_unit_id=1,
)
# Same API as OPC UA
```

## 4. Tariff Integration

### Utility Tariffs

```python
import asyncio
from datetime import datetime
from integrations.tariff_provider import (
    create_tariff_connector,
    RatePeriod,
)

async def main():
    # Create tariff connector
    tariff = create_tariff_connector(
        utility_id="14328",  # Example utility ID
        zip_code="94102",
        api_key="your_openei_key",
    )

    await tariff.connect()

    # Get tariff schedule
    schedule = await tariff.fetch_tariff_schedule()
    print(f"Tariff: {schedule.schedule_name}")
    print(f"Utility: {schedule.utility_name}")
    print(f"Type: {schedule.tariff_type.value}")

    # Get current rate
    rate = await tariff.get_current_rate()
    print(f"\nCurrent Rate: ${rate.rate:.4f}/kWh")
    print(f"Period: {rate.rate_period.value if rate.rate_period else 'N/A'}")

    # Get rate forecast
    print("\n24-Hour Rate Forecast:")
    forecast = await tariff.get_rate_forecast(hours_ahead=24)
    for timestamp, rate_value, period in forecast:
        print(f"  {timestamp.strftime('%H:%M')}: ${rate_value:.4f}/kWh ({period.value})")

    # Calculate cost
    cost = await tariff.calculate_cost(
        energy_kwh=1000,
        peak_demand_kw=200,
    )
    print(f"\nEstimated Cost for 1000 kWh, 200 kW peak:")
    print(f"  Energy: ${cost['energy_cost']:.2f}")
    print(f"  Demand: ${cost['demand_cost']:.2f}")
    print(f"  Fixed: ${cost['fixed_cost']:.2f}")
    print(f"  Taxes: ${cost['taxes']:.2f}")
    print(f"  TOTAL: ${cost['total']:.2f}")

    # Subscribe to rate changes
    def on_rate_change(change):
        print(f"\nRATE CHANGE ALERT!")
        print(f"  Old rate: ${change.old_rate:.4f}/kWh")
        print(f"  New rate: ${change.new_rate:.4f}/kWh")
        print(f"  Change: {change.rate_change_percent:+.1f}%")

    await tariff.subscribe_to_rate_changes(on_rate_change)

    # Keep running to receive rate change alerts
    await asyncio.sleep(3600)

    await tariff.disconnect()

asyncio.run(main())
```

### LMP Pricing

```python
from integrations.tariff_provider import create_lmp_connector

lmp = create_lmp_connector(
    base_url="https://api.pjm.com",
    api_key="your_api_key",
)

await lmp.connect()

# Get current LMP
current = await lmp.get_current_lmp()
print(f"Current LMP: ${current}/MWh")

# Get historical data
history = await lmp.get_historical_lmp(
    start_time=datetime.now() - timedelta(hours=24),
    end_time=datetime.now(),
)
for timestamp, price in history:
    print(f"{timestamp}: ${price}/MWh")

# Subscribe to updates
await lmp.subscribe_to_lmp(
    callback=lambda price, ts: print(f"LMP: ${price}/MWh"),
    interval_seconds=60,
)
```

## 5. Complete Integration Example

```python
"""
Complete HEATSCHEDULER integration example.
Demonstrates all integration points working together.
"""

import asyncio
from datetime import datetime, timedelta
from integrations import (
    # ERP
    create_erp_connector,
    ERPSystem,
    # Energy Management
    create_pricing_connector,
    create_demand_response_connector,
    ISOMarket,
    # SCADA
    create_scada_client,
    ConnectionProtocol,
    HeatingEquipment,
    EquipmentType,
    create_heating_equipment_tags,
    # Tariff
    create_tariff_connector,
)


async def main():
    print("=" * 60)
    print("GL-019 HEATSCHEDULER - Integration Demo")
    print("=" * 60)

    # 1. Connect to ERP
    print("\n[1] Connecting to ERP...")
    erp = create_erp_connector(
        erp_system=ERPSystem.SAP_S4HANA,
        host="sap.company.com",
        username="integration_user",
        password="password",
        system_id="PRD",
    )
    # Note: await erp.connect() in production

    # 2. Connect to energy pricing
    print("[2] Connecting to energy pricing...")
    pricing = create_pricing_connector(
        iso_market=ISOMarket.PJM,
        pricing_node="PJMISO",
        api_key="api_key",
    )
    # Note: await pricing.connect() in production

    # 3. Connect to demand response
    print("[3] Connecting to demand response...")
    dr = create_demand_response_connector(
        ven_id="VEN001",
        program_id="DR_PROGRAM",
    )
    # Note: await dr.connect() in production

    # 4. Connect to SCADA
    print("[4] Connecting to SCADA...")
    scada = create_scada_client(
        protocol=ConnectionProtocol.OPC_UA,
        host="192.168.1.100",
        port=4840,
    )

    # Register equipment
    furnaces = [
        HeatingEquipment(
            equipment_id=f"FURNACE_0{i}",
            equipment_name=f"Heat Treatment Furnace {i}",
            equipment_type=EquipmentType.HEAT_TREATMENT_FURNACE,
            plant_code="1000",
            rated_power_kw=500,
            max_temperature_c=1200,
        )
        for i in range(1, 4)
    ]

    for furnace in furnaces:
        scada.register_equipment(furnace)
        tags = create_heating_equipment_tags(
            furnace.equipment_id,
            furnace.equipment_type
        )
        scada.register_tags(tags)

    # Note: await scada.connect() in production

    # 5. Connect to tariff provider
    print("[5] Connecting to tariff provider...")
    tariff = create_tariff_connector(
        utility_id="14328",
        zip_code="94102",
        api_key="openei_key",
    )
    # Note: await tariff.connect() in production

    print("\nAll integrations configured!")
    print("\nRegistered Equipment:")
    for eq in scada.get_all_equipment():
        print(f"  - {eq.equipment_id}: {eq.equipment_name}")

    print("\n" + "=" * 60)
    print("Integration setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Connection Failed

```python
# Check connectivity first
import socket

def check_connection(host, port):
    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

check_connection("192.168.1.100", 4840)
```

### Authentication Error

```python
# Use environment variables for credentials
import os

connector = create_erp_connector(
    erp_system=ERPSystem.SAP_S4HANA,
    host=os.getenv("SAP_HOST"),
    username=os.getenv("SAP_USER"),
    password=os.getenv("SAP_PASSWORD"),
)
```

### Timeout Issues

```python
# Increase timeout
config = SCADAConfig(
    protocol=ConnectionProtocol.OPC_UA,
    host="192.168.1.100",
    connection_timeout=30.0,  # Increase timeout
    modbus_timeout=10.0,      # For Modbus
)
```

### Rate Limiting

```python
# Adjust rate limits
config = PricingConfig(
    host="api.example.com",
    rate_limit_requests_per_minute=30,  # Reduce rate
)
```

## Next Steps

1. **Configure production credentials** - Use vault or environment variables
2. **Set up monitoring** - Implement health checks and alerting
3. **Customize tag mappings** - Match your equipment's SCADA configuration
4. **Enable rate change alerts** - Subscribe to tariff notifications
5. **Integrate with scheduler** - Connect to HEATSCHEDULER optimization engine

## Support

- **Full documentation:** `README_INTEGRATIONS.md`
- **API reference:** Module docstrings
- **Examples:** `examples/` directory
- **Email:** support@greenlang.com
