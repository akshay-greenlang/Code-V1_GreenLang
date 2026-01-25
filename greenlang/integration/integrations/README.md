# GreenLang Integrations Framework

Production-grade integration framework for external system integrations with enterprise reliability patterns.

## Overview

The GreenLang Integrations framework provides a robust, type-safe foundation for connecting to external systems including SCADA, ERP, CEMS, Historians, and CMMS platforms.

**Key Features:**
- ✅ Retry logic with exponential backoff (tenacity)
- ✅ Circuit breaker pattern (pybreaker)
- ✅ Health monitoring and status tracking
- ✅ Connection pooling and lifecycle management
- ✅ Prometheus metrics export
- ✅ Type-safe generic interfaces
- ✅ Mock implementations for testing
- ✅ Zero-hallucination data retrieval
- ✅ Provenance tracking (SHA-256 hashing)

## Architecture

```
greenlang/integrations/
├── base_connector.py       # BaseConnector abstract class
├── registry.py             # Connector discovery and factory
├── health_monitor.py       # Health monitoring and metrics
├── scada_connector.py      # SCADA integration (OPC UA, Modbus, DNP3, BACnet)
├── erp_connector.py        # ERP integration (SAP, Oracle, Dynamics, NetSuite)
├── cems_connector.py       # Emissions monitoring
├── historian_connector.py  # Time-series historians (PI, PHD, GE)
└── cmms_connector.py       # Maintenance management (Maximo, SAP PM)
```

## Quick Start

### Basic Usage

```python
from greenlang.integrations import SCADAConnector, SCADAConfig, SCADAQuery

# Create configuration
config = SCADAConfig(
    connector_id="scada-plant1",
    connector_type="scada",
    protocol="opcua",
    endpoint="opc.tcp://192.168.1.100:4840",
    max_retries=3,
    circuit_breaker_enabled=True
)

# Create connector
connector = SCADAConnector(config)

# Use async context manager
async with connector:
    # Create query
    query = SCADAQuery(
        tag_ids=["ns=2;s=Temperature", "ns=2;s=Pressure"]
    )

    # Fetch data (with retry and circuit breaker)
    payload, provenance = await connector.fetch_data(query)

    # Process data
    for tag in payload.tags:
        print(f"{tag.tag_name}: {tag.value} {tag.unit}")
```

### Using the Registry

```python
from greenlang.integrations import IntegrationRegistry

# Get global registry (auto-registers built-in connectors)
registry = IntegrationRegistry()

# List available connectors
for connector_info in registry.list_connectors():
    print(f"{connector_info.connector_type} v{connector_info.version}")

# Create connector from registry
config = SCADAConfig(...)
connector = registry.create_connector("scada-connector", config)
```

### Health Monitoring

```python
from greenlang.integrations import HealthMonitor

# Create health monitor
monitor = HealthMonitor(
    check_interval=60,  # Check every 60 seconds
    enable_prometheus=True
)

# Register connectors
monitor.register_connector(scada_connector)
monitor.register_connector(erp_connector)

# Start monitoring
await monitor.start_monitoring()

# Get health status
health = monitor.get_aggregated_health()
print(f"Healthy: {health.healthy_count}/{health.total_connectors}")

# Get metrics
metrics = monitor.get_all_metrics()
for connector_id, metric in metrics.items():
    print(f"{connector_id}: {metric.total_requests} requests")
```

## Connector Types

### SCADA Connector

Supports multiple SCADA protocols:

```python
from greenlang.integrations import SCADAConnector, SCADAConfig, SCADAQuery

config = SCADAConfig(
    connector_id="scada-thermal",
    connector_type="scada",
    protocol="opcua",  # opcua | modbus | dnp3 | bacnet
    endpoint="opc.tcp://localhost:4840",
    namespace_index=2,
    polling_interval_ms=1000
)

connector = SCADAConnector(config)
```

**Protocols:**
- OPC UA (Unified Architecture)
- Modbus TCP/RTU
- DNP3 (Distributed Network Protocol)
- BACnet (Building Automation)

**Use Cases:**
- GL-001: ThermoSync HVAC optimization
- GL-002: Industrial equipment monitoring
- GL-003: Building energy management

### ERP Connector

Integrates with enterprise systems:

```python
from greenlang.integrations import ERPConnector, ERPConfig, ERPQuery

config = ERPConfig(
    connector_id="erp-sap-prod",
    connector_type="erp",
    erp_system="sap",  # sap | oracle | dynamics | netsuite
    base_url="https://sap.company.com/api",
    client_id="...",
    client_secret="..."
)

connector = ERPConnector(config)

query = ERPQuery(
    entity_type="material",
    filters={"plant_code": "1000"},
    limit=1000
)

payload, prov = await connector.fetch_data(query)
```

**Systems:**
- SAP S/4HANA, ECC
- Oracle ERP Cloud
- Microsoft Dynamics 365
- NetSuite

**Use Cases:**
- Activity data extraction
- Master data synchronization
- Procurement data for Scope 3

### CEMS Connector

Continuous Emissions Monitoring:

```python
from greenlang.integrations import CEMSConnector, CEMSConfig, CEMSQuery

config = CEMSConfig(
    connector_id="cems-stack1",
    connector_type="cems",
    protocol="modbus",
    endpoint="192.168.1.50:502",
    stack_id="STACK-001"
)

connector = CEMSConnector(config)

query = CEMSQuery(
    pollutants=["CO2", "NOx", "SO2"],
    start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
    end_time=datetime(2025, 1, 2, tzinfo=timezone.utc)
)

payload, prov = await connector.fetch_data(query)
```

**Use Cases:**
- GL-010: Direct emissions measurement
- Regulatory compliance reporting
- Real-time emissions tracking

### Historian Connector

Time-series data historians:

```python
from greenlang.integrations import HistorianConnector, HistorianConfig, HistorianQuery

config = HistorianConfig(
    connector_id="historian-pi",
    connector_type="historian",
    historian_type="pi",  # pi | phd | ge
    server_url="https://pi.company.com"
)

connector = HistorianConnector(config)

query = HistorianQuery(
    tag_names=["ProcessTemp", "FlowRate"],
    start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
    end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
    interval="1h",
    aggregation="avg"
)

payload, prov = await connector.fetch_data(query)
```

**Systems:**
- OSIsoft PI System
- Honeywell PHD
- GE Historian

**Use Cases:**
- Historical process data
- Energy consumption trends
- Equipment performance analysis

### CMMS Connector

Maintenance management systems:

```python
from greenlang.integrations import CMMSConnector, CMMSConfig, CMMSQuery

config = CMMSConfig(
    connector_id="cmms-maximo",
    connector_type="cmms",
    cmms_system="maximo",  # maximo | sap | infor
    api_url="https://maximo.company.com/api"
)

connector = CMMSConnector(config)

query = CMMSQuery(
    query_type="work_order",
    asset_ids=["ASSET-001", "ASSET-002"],
    status="completed"
)

payload, prov = await connector.fetch_data(query)
```

**Systems:**
- IBM Maximo
- SAP PM
- Infor EAM

**Use Cases:**
- Equipment maintenance history
- Asset lifecycle data
- Downtime analysis

## Configuration

### Retry Configuration

```python
config = ConnectorConfig(
    connector_id="my-connector",
    connector_type="scada",
    max_retries=3,              # Max retry attempts
    retry_multiplier=1,          # Exponential backoff multiplier
    retry_min_wait=2,            # Min wait between retries (seconds)
    retry_max_wait=10            # Max wait between retries (seconds)
)
```

**Retry Strategy:**
- Exponential backoff with jitter
- Retries only on transient errors (ConnectionError, TimeoutError)
- Configurable max attempts and wait times

### Circuit Breaker Configuration

```python
config = ConnectorConfig(
    connector_id="my-connector",
    connector_type="scada",
    circuit_breaker_enabled=True,
    circuit_breaker_threshold=5,  # Failures before opening circuit
    circuit_breaker_timeout=60     # Recovery timeout (seconds)
)
```

**Circuit Breaker States:**
- **Closed**: Normal operation
- **Open**: Circuit tripped, requests fail fast
- **Half-Open**: Testing if service recovered

### Health Monitoring Configuration

```python
config = ConnectorConfig(
    connector_id="my-connector",
    connector_type="scada",
    health_check_interval=60,   # Health check every 60 seconds
    timeout_seconds=30           # Request timeout
)
```

## Creating Custom Connectors

### Step 1: Define Models

```python
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal

class MyQuery(BaseModel):
    """Custom query model."""

    device_id: str = Field(..., description="Device identifier")
    start_time: datetime = Field(..., description="Start time")
    end_time: datetime = Field(..., description="End time")

class MyPayload(BaseModel):
    """Custom payload model."""

    data_points: List[dict] = Field(..., description="Data points")
    total_count: int = Field(..., description="Total count")

class MyConfig(ConnectorConfig):
    """Custom configuration."""

    api_key: str = Field(..., description="API key")
    base_url: str = Field(..., description="API base URL")
```

### Step 2: Implement Connector

```python
from greenlang.integrations import BaseConnector

class MyConnector(BaseConnector[MyQuery, MyPayload, MyConfig]):
    """Custom connector implementation."""

    connector_id = "my-connector"
    connector_version = "1.0.0"

    async def connect(self) -> bool:
        """Establish connection."""
        # Initialize API client
        self._client = HTTPClient(self.config.base_url)
        return True

    async def disconnect(self) -> bool:
        """Close connection."""
        if self._client:
            await self._client.close()
        return True

    async def _health_check_impl(self) -> bool:
        """Check service health."""
        try:
            response = await self._client.get("/health")
            return response.status == 200
        except Exception:
            return False

    async def _fetch_data_impl(self, query: MyQuery) -> MyPayload:
        """
        Fetch data - ZERO HALLUCINATION.

        No LLM calls allowed in this method.
        """
        # Build API request
        params = {
            "device_id": query.device_id,
            "start": query.start_time.isoformat(),
            "end": query.end_time.isoformat()
        }

        # Fetch data from API
        response = await self._client.get("/data", params=params)
        data = response.json()

        # Parse into typed payload
        return MyPayload(
            data_points=data["points"],
            total_count=len(data["points"])
        )
```

### Step 3: Register Connector

```python
from greenlang.integrations import get_registry

# Get global registry
registry = get_registry()

# Register custom connector
registry.register(
    MyConnector,
    description="My custom connector",
    supported_protocols=["https"]
)

# Create instance from registry
config = MyConfig(...)
connector = registry.create_connector("my-connector", config)
```

## Testing

### Mock Mode

All connectors support mock mode for testing:

```python
config = SCADAConfig(
    connector_id="test-scada",
    connector_type="scada",
    protocol="opcua",
    endpoint="opc.tcp://localhost:4840",
    mock_mode=True  # Enable mock mode
)

connector = SCADAConnector(config)

# Connector will return deterministic mock data
async with connector:
    payload, prov = await connector.fetch_data(query)
    # Returns mock data, no actual network calls
```

### Test Fixtures

```python
import pytest
from greenlang.integrations import SCADAConnector, SCADAConfig

@pytest.fixture
def scada_connector():
    """Create mock SCADA connector for testing."""
    config = SCADAConfig(
        connector_id="test-scada",
        connector_type="scada",
        protocol="opcua",
        endpoint="opc.tcp://localhost:4840",
        mock_mode=True
    )
    return SCADAConnector(config)

@pytest.mark.asyncio
async def test_scada_data_fetch(scada_connector):
    """Test SCADA data fetching."""
    async with scada_connector:
        query = SCADAQuery(tag_ids=["tag1", "tag2"])
        payload, prov = await scada_connector.fetch_data(query)

        assert len(payload.tags) == 2
        assert prov.connector_id == "test-scada"
```

## Metrics and Monitoring

### Connector Metrics

Each connector tracks:

```python
# Get metrics
metrics = connector.get_metrics()

print(f"Total Requests: {metrics.total_requests}")
print(f"Successful: {metrics.successful_requests}")
print(f"Failed: {metrics.failed_requests}")
print(f"Avg Response Time: {metrics.avg_response_time_ms}ms")
print(f"Health Status: {metrics.health_status}")
```

### Prometheus Integration

When `prometheus_client` is available:

```python
from greenlang.integrations import HealthMonitor

monitor = HealthMonitor(enable_prometheus=True)
monitor.register_connector(connector)
await monitor.start_monitoring()

# Metrics exported:
# - connector_health_status
# - connector_requests_total
# - connector_response_time_seconds
# - connector_circuit_breaker_opens_total
```

## Error Handling

### Retry on Transient Errors

```python
try:
    payload, prov = await connector.fetch_data(query)
except ConnectionError as e:
    # After max_retries exhausted
    logger.error(f"Connection failed: {e}")
except TimeoutError as e:
    # Request timed out
    logger.error(f"Request timeout: {e}")
```

### Circuit Breaker

```python
from pybreaker import CircuitBreakerError

try:
    payload, prov = await connector.fetch_data(query)
except CircuitBreakerError as e:
    # Circuit is open, fail fast
    logger.error(f"Circuit breaker open: {e}")
    # Wait for recovery timeout before retrying
```

## Best Practices

### 1. Use Context Managers

```python
# ✅ Good - Ensures proper cleanup
async with connector:
    data = await connector.fetch_data(query)

# ❌ Bad - Manual lifecycle management
await connector.connect()
data = await connector.fetch_data(query)
await connector.disconnect()  # May not be called on error
```

### 2. Enable Health Monitoring

```python
# ✅ Good - Proactive health monitoring
monitor = HealthMonitor()
monitor.register_connector(connector)
await monitor.start_monitoring()

# Check health before critical operations
if connector.metrics.health_status == HealthStatus.HEALTHY:
    await connector.fetch_data(query)
```

### 3. Configure Timeouts

```python
# ✅ Good - Set appropriate timeouts
config = ConnectorConfig(
    timeout_seconds=30,  # Based on expected response time
    max_retries=3,
    retry_max_wait=10
)

# Per-request timeout override
payload, prov = await connector.fetch_data(query, timeout=60)
```

### 4. Monitor Metrics

```python
# ✅ Good - Regular metrics monitoring
metrics = connector.get_metrics()

if metrics.failed_requests > 100:
    logger.warning(f"High failure rate: {metrics.failed_requests}")

if metrics.avg_response_time_ms > 5000:
    logger.warning(f"Slow response time: {metrics.avg_response_time_ms}ms")
```

### 5. Use Mock Mode for Testing

```python
# ✅ Good - Test without external dependencies
config = ConnectorConfig(
    connector_id="test",
    connector_type="scada",
    mock_mode=True  # Deterministic mock data
)

# ❌ Bad - Tests depend on external systems
config = ConnectorConfig(
    connector_id="test",
    connector_type="scada",
    endpoint="production.example.com"  # Don't test against prod!
)
```

## Dependencies

Required:
- `tenacity>=8.2.3` - Retry logic
- `pybreaker>=1.0.1` - Circuit breaker pattern
- `pydantic>=2.5.3` - Data validation

Optional:
- `aiohttp>=3.9.3` - Async HTTP client (server extra)
- `prometheus-client>=0.19.0` - Metrics export (server extra)

Install with extras:
```bash
pip install greenlang-cli[server]  # Includes aiohttp, prometheus-client
```

## Performance

### Connection Pooling

BaseConnector supports connection pooling (implement in subclass):

```python
class MyConnector(BaseConnector):
    def __init__(self, config):
        super().__init__(config)
        self._connection_pool = ConnectionPool(
            max_size=config.connection_pool_size
        )
```

### Batch Processing

For bulk data retrieval:

```python
# Fetch multiple queries concurrently
queries = [query1, query2, query3]

results = await asyncio.gather(*[
    connector.fetch_data(q) for q in queries
])
```

### Caching

Implement caching for frequently accessed data:

```python
from functools import lru_cache

class CachedConnector(BaseConnector):
    @lru_cache(maxsize=1000)
    async def fetch_data(self, query):
        return await super().fetch_data(query)
```

## Troubleshooting

### Connection Failures

```python
# Check connectivity
is_healthy = await connector.health_check()
if not is_healthy:
    logger.error("Connector unhealthy")

# Review metrics
metrics = connector.get_metrics()
logger.info(f"Failures: {metrics.failed_requests}")
```

### Circuit Breaker Open

```python
# Wait for recovery timeout
await asyncio.sleep(config.circuit_breaker_timeout)

# Try again
try:
    payload, prov = await connector.fetch_data(query)
except CircuitBreakerError:
    # Still open, wait longer
    pass
```

### Slow Response Times

```python
# Check average response time
metrics = connector.get_metrics()
if metrics.avg_response_time_ms > 5000:
    # Increase timeout
    connector.config.timeout_seconds = 60

    # Or reduce batch size
    # Or enable caching
```

## License

MIT License - See LICENSE file for details.

## Support

- Documentation: https://greenlang.io/docs/integrations
- Issues: https://github.com/greenlang/greenlang/issues
- Discord: https://discord.gg/greenlang

---

**Version:** 1.0.0
**Author:** GreenLang Backend Team
**Date:** 2025-12-01
