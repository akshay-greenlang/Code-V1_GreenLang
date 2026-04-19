# GL-019 HEATSCHEDULER - Integration Module

**Enterprise-grade data integration for production planning systems and energy management**

## Overview

The GL-019 HEATSCHEDULER integration module provides comprehensive connectivity to enterprise systems for optimal heat scheduling. It enables seamless data flow between production planning, energy management, SCADA/DCS systems, and tariff data providers.

## Architecture

```
                    +-------------------+
                    |   HEATSCHEDULER   |
                    |   Core Engine     |
                    +--------+----------+
                             |
         +-------------------+-------------------+
         |                   |                   |
+--------v--------+ +--------v--------+ +--------v--------+
|  ERP Connector  | |   EMS Connector | | SCADA Connector |
|   (SAP/Oracle)  | | (Pricing/DR/Grid)| | (OPC-UA/Modbus) |
+-----------------+ +-----------------+ +-----------------+
         |                   |                   |
         v                   v                   v
+--------+--------+ +--------+--------+ +--------+--------+
|  Production     | |  Energy Markets | |    Heating      |
|  Schedules      | |  Demand Response| |    Equipment    |
|  Work Orders    | |  Grid Operators | |    Control      |
|  Maintenance    | |  Energy Meters  | |    Monitoring   |
+-----------------+ +-----------------+ +-----------------+
```

## Module Structure

```
GL-019/integrations/
    __init__.py                    # Module initialization and exports
    erp_connector.py               # ERP/MES integration (SAP, Oracle)
    energy_management_connector.py # Energy management systems
    scada_integration.py           # SCADA/DCS for heating equipment
    tariff_provider.py             # Energy tariff data providers
    README_INTEGRATIONS.md         # This documentation
    QUICKSTART.md                  # Quick start guide
```

## Integration Components

### 1. ERP Connector (`erp_connector.py`)

Enterprise Resource Planning and Manufacturing Execution System integration.

**Supported Systems:**
- SAP S/4HANA (RFC/BAPI, OData REST)
- SAP ECC (RFC/BAPI)
- Oracle ERP Cloud (REST API)
- Oracle E-Business Suite (REST API)

**Features:**
- Production schedule extraction
- Work order integration
- Maintenance schedule synchronization
- Equipment master data retrieval
- Real-time schedule updates

**Key Classes:**
- `ERPConnectorBase` - Abstract base class
- `SAPERPConnector` - SAP implementation
- `OracleERPConnector` - Oracle implementation
- `ProductionSchedule` - Schedule data model
- `WorkOrder` - Work order data model
- `MaintenanceSchedule` - Maintenance data model

**Example:**
```python
from integrations.erp_connector import create_erp_connector, ERPSystem

# Create SAP connector
connector = create_erp_connector(
    erp_system=ERPSystem.SAP_S4HANA,
    host="sap.company.com",
    username="integration_user",
    password="secure_password",
    system_id="PRD",
    client_id="100",
)

# Connect and fetch schedules
await connector.connect()

schedules = await connector.get_production_schedules(
    plant_code="1000",
    start_date=datetime.now(),
    end_date=datetime.now() + timedelta(days=7),
    include_heating_only=True,
)

for schedule in schedules:
    print(f"Schedule: {schedule.schedule_id}")
    for item in schedule.items:
        if item.heating_required:
            print(f"  - {item.item_id}: {item.temperature_setpoint_c}C")
```

### 2. Energy Management Connector (`energy_management_connector.py`)

Real-time energy pricing, demand response, and grid integration.

**Integration Points:**
- Real-time pricing feeds (LMP, wholesale markets)
- Demand response signals (OpenADR 2.0b)
- Grid operator integration (ISO/RTO)
- Energy meter data collection

**Supported Markets:**
- PJM Interconnection
- ERCOT
- CAISO
- NYISO
- ISO New England
- MISO
- SPP

**Key Classes:**
- `RealTimePricingConnector` - LMP and market prices
- `DemandResponseConnector` - DR events and signals
- `GridOperatorConnector` - Grid frequency and alerts
- `EnergyMeterConnector` - Power/energy monitoring

**Example:**
```python
from integrations.energy_management_connector import (
    create_pricing_connector,
    create_demand_response_connector,
    ISOMarket,
    PriceType,
)

# Real-time pricing
pricing = create_pricing_connector(
    iso_market=ISOMarket.PJM,
    pricing_node="PJMISO",
    api_key="your_api_key",
)

await pricing.connect()
current_price = await pricing.get_current_price(PriceType.LMP)
print(f"Current LMP: ${current_price.price}/MWh")

# Price forecast
forecast = await pricing.get_price_forecast(hours_ahead=24)
for timestamp, price, period in forecast:
    print(f"{timestamp}: ${price}/MWh ({period.value})")

# Demand response
dr_connector = create_demand_response_connector(
    ven_id="VEN001",
    vtn_url="https://dr.utility.com/OpenADR",
    program_id="DR_PROGRAM_1",
)

await dr_connector.connect()
events = await dr_connector.get_active_events()

for event in events:
    print(f"DR Event: {event.event_id}")
    print(f"  Start: {event.start_time}")
    print(f"  Target reduction: {event.target_reduction_kw} kW")
```

### 3. SCADA Integration (`scada_integration.py`)

Heating equipment monitoring and control via industrial protocols.

**Supported Protocols:**
- OPC UA (Unified Architecture)
- Modbus TCP
- Modbus RTU

**Equipment Types:**
- Industrial furnaces (batch, continuous)
- Steam and hot water boilers
- Heat exchangers
- Industrial ovens and dryers
- Kilns and autoclaves

**Features:**
- Equipment availability monitoring
- Temperature and power readings
- Control setpoint writing
- Alarm management

**Key Classes:**
- `SCADAClient` - Main SCADA client
- `HeatingEquipment` - Equipment definition
- `EquipmentTag` - SCADA tag definition
- `EquipmentReading` - Status reading
- `TemperatureReading` - Temperature data
- `PowerReading` - Power consumption data
- `ControlSetpoint` - Control commands

**Example:**
```python
from integrations.scada_integration import (
    create_scada_client,
    ConnectionProtocol,
    HeatingEquipment,
    EquipmentType,
    ControlSetpoint,
    create_heating_equipment_tags,
)

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
)
client.register_equipment(furnace)

# Register tags
tags = create_heating_equipment_tags("FURNACE_01", EquipmentType.HEAT_TREATMENT_FURNACE)
client.register_tags(tags)

# Connect and read
await client.connect()

status = await client.read_equipment_status("FURNACE_01")
print(f"Status: {status.status.value}")

temp = await client.read_temperature("FURNACE_01")
print(f"Temperature: {temp.process_temperature_c}C")

power = await client.read_power("FURNACE_01")
print(f"Power: {power.active_power_kw} kW")

# Write setpoint
setpoint = ControlSetpoint(
    equipment_id="FURNACE_01",
    temperature_setpoint_c=850,
    power_limit_kw=400,
)
await client.write_setpoint("FURNACE_01", setpoint)
```

### 4. Tariff Provider (`tariff_provider.py`)

Energy tariff data retrieval and rate management.

**Data Sources:**
- OpenEI Utility Rate Database
- Direct utility APIs
- Wholesale market prices
- LMP real-time pricing

**Features:**
- Tariff rate retrieval
- Time-of-use schedule management
- Demand charge calculations
- Rate change notifications
- Cost forecasting

**Key Classes:**
- `UtilityTariffConnector` - Utility tariffs
- `WholesaleMarketConnector` - Wholesale prices
- `LMPPricingConnector` - LMP data
- `TariffSchedule` - Complete tariff
- `TimeOfUseRate` - TOU rate definition
- `DemandCharge` - Demand charge definition

**Example:**
```python
from integrations.tariff_provider import (
    create_tariff_connector,
    create_lmp_connector,
    TariffType,
)

# Utility tariff
tariff_connector = create_tariff_connector(
    utility_id="14328",  # Example utility
    zip_code="94102",
    api_key="your_openei_key",
)

await tariff_connector.connect()
tariff = await tariff_connector.fetch_tariff_schedule()

print(f"Tariff: {tariff.schedule_name}")
print(f"Type: {tariff.tariff_type.value}")

# Current rate
rate = await tariff_connector.get_current_rate()
print(f"Current rate: ${rate.rate}/kWh ({rate.rate_period.value})")

# Rate forecast
forecast = await tariff_connector.get_rate_forecast(hours_ahead=24)
for timestamp, rate_value, period in forecast:
    print(f"{timestamp.hour}:00 - ${rate_value:.4f}/kWh ({period.value})")

# Cost calculation
cost = await tariff_connector.calculate_cost(
    energy_kwh=1000,
    peak_demand_kw=200,
)
print(f"Estimated cost: ${cost['total']:.2f}")

# LMP connector
lmp = create_lmp_connector(
    base_url="https://api.iso.example.com",
    api_key="your_api_key",
)

await lmp.connect()
current_lmp = await lmp.get_current_lmp()
print(f"Current LMP: ${current_lmp}/MWh")
```

## Protocol Specifications

### OPC UA

**Endpoint Format:**
```
opc.tcp://<host>:<port>/[path]
```

**Default Port:** 4840

**Security Modes:**
- None
- Sign
- SignAndEncrypt

**Security Policies:**
- None
- Basic256Sha256

**Authentication:**
- Anonymous
- Username/Password
- X.509 Certificate

### Modbus TCP

**Default Port:** 502

**Function Codes:**
- 03: Read Holding Registers
- 04: Read Input Registers
- 06: Write Single Register
- 16: Write Multiple Registers

**Register Mapping:**
- 30001-39999: Input Registers (read-only)
- 40001-49999: Holding Registers (read/write)

### REST APIs

**Authentication Methods:**
- API Key (header or query parameter)
- OAuth 2.0 Client Credentials
- Basic Authentication
- JWT Bearer Token

**Rate Limiting:**
- Default: 100 requests/minute
- Configurable per connector

## Authentication Requirements

### SAP ERP

**RFC Connection:**
```python
{
    "ashost": "sap-server.company.com",
    "sysnr": "00",
    "client": "100",
    "user": "RFC_USER",
    "passwd": "<from vault>",
    "lang": "EN",
}
```

**OAuth 2.0 (S/4HANA Cloud):**
- Token URL: `/oauth/token`
- Grant Type: `client_credentials`
- Scope: `API_PRODUCTION_ORDER_2_SRV`

### Oracle ERP Cloud

**Basic Auth:**
- Username/password with Base64 encoding

**OAuth 2.0:**
- Token URL: `https://<host>/oauth2/v1/token`
- Grant Type: `client_credentials`

### Energy Market APIs

**PJM Data Miner:**
- API Key in query parameter
- Rate limit: 5 requests/second

**ERCOT:**
- API Key in header
- Rate limit: 100 requests/minute

**CAISO OASIS:**
- Public access (some endpoints)
- Certificate for secure endpoints

## Data Mapping Tables

### Equipment Status Mapping

| SCADA Value | EquipmentStatus |
|-------------|-----------------|
| 0 | OFF |
| 1 | STANDBY |
| 2 | HEATING_UP |
| 3 | RUNNING |
| 4 | COOLING_DOWN |
| 5 | MAINTENANCE |
| 6 | FAULT |
| 7 | EMERGENCY_STOP |

### SAP Order Status Mapping

| SAP Status | WorkOrderStatus |
|------------|-----------------|
| CRTD | CREATED |
| REL | RELEASED |
| PCNF | PARTIALLY_COMPLETE |
| CNF | COMPLETED |
| TECO | COMPLETED |
| DLT | CANCELLED |

### Rate Period Mapping

| Period | Typical Hours (Summer) | Typical Hours (Winter) |
|--------|------------------------|-------------------------|
| OFF_PEAK | 21:00 - 12:00 | 21:00 - 07:00 |
| MID_PEAK | 12:00 - 14:00, 19:00 - 21:00 | 07:00 - 17:00 |
| ON_PEAK | 14:00 - 19:00 | 17:00 - 21:00 |

## Error Handling

All connectors implement:

1. **Retry Logic** - Exponential backoff with configurable attempts
2. **Circuit Breaker** - Automatic failover when service unavailable
3. **Rate Limiting** - Token bucket algorithm
4. **Connection Pooling** - Efficient resource management
5. **Data Buffering** - Queue writes during disconnections

**Example Error Handling:**
```python
try:
    await connector.connect()
    data = await connector.fetch_data()
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    # Connector will auto-reconnect
except TimeoutError as e:
    logger.error(f"Request timeout: {e}")
    # Retry with exponential backoff
except RateLimitError as e:
    logger.warning(f"Rate limited: {e}")
    # Wait and retry
```

## Performance Considerations

### Caching

- ERP data: 5-minute cache (configurable)
- Tariff data: 1-hour cache
- SCADA readings: 1-second cache
- Price data: 1-minute cache

### Batch Operations

- ERP: Batch size 100 records
- SCADA: Batch read 50 tags
- Meter readings: 10-second intervals

### Connection Pooling

- SAP RFC: 5 connections (configurable)
- HTTP clients: 10 connections per host
- Modbus: Single connection per device

## Security Best Practices

1. **Never hardcode credentials** - Use environment variables or vault
2. **Use TLS/SSL** - Encrypt all network traffic
3. **Implement least privilege** - Limit API permissions
4. **Rotate credentials** - Regular password/key rotation
5. **Audit logging** - Log all access attempts

**Secure Configuration Example:**
```python
import os

config = SAPConfig(
    host=os.getenv("SAP_HOST"),
    username=os.getenv("SAP_USER"),
    password=vault_client.get_secret("sap_password"),
    use_ssl=True,
    certificate_path="/etc/ssl/certs/client.pem",
)
```

## Monitoring and Observability

### Health Checks

```python
# ERP health check
health = await erp_connector.health_check()
print(f"ERP healthy: {health['healthy']}")
print(f"Status: {health['status']}")

# SCADA health check
health = await scada_client.health_check()
print(f"SCADA connected: {health['connected']}")
print(f"Active alarms: {health['statistics']['active_alarms']}")
```

### Statistics

```python
# Get connector statistics
stats = connector.get_statistics()
print(f"Requests: {stats['requests']}")
print(f"Successes: {stats['successes']}")
print(f"Failures: {stats['failures']}")
print(f"Cache hits: {stats['cache_hits']}")
```

## Version History

### v1.0.0 (2024-12-03)
- Initial release
- SAP S/4HANA and ECC support
- Oracle ERP Cloud support
- OPC UA and Modbus protocols
- Real-time pricing integration
- Demand response support
- Comprehensive tariff management

## Support

- **Documentation:** https://docs.greenlang.com/gl-019/integrations
- **GitHub Issues:** https://github.com/greenlang/gl-019/issues
- **Email:** support@greenlang.com

## License

Proprietary - GreenLang Industrial Systems
(c) 2024 All Rights Reserved
