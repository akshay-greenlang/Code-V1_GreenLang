# GreenLang Connectors - DATA-301 Implementation

## Overview

The GreenLang Connector SDK provides deterministic, replay-capable access to external data sources with built-in security, provenance tracking, and byte-exact reproducibility for compliance and auditing.

## Architecture

```
greenlang/connectors/
├── base.py              # Enhanced Connector base class (async-first, generic types)
├── context.py           # ConnectorContext with mode enforcement
├── snapshot.py          # Deterministic snapshot manager (reuses intelligence infrastructure)
├── errors.py            # Structured error taxonomy
├── models.py            # Common data models (TimeWindow, TSPoint, etc.)
└── grid/
    ├── __init__.py
    └── mock.py          # GridIntensityMockConnector (DATA-301 deliverable)
```

## Key Features

### ✅ **Deterministic & Reproducible**
- Byte-exact snapshots (canonical JSON, sorted keys, Decimal precision)
- SHA-256 content addressing
- Deterministic seeding for mock data
- Round-trip snapshot validation

### ✅ **Security-First**
- Default-deny egress (network disabled by default)
- Three execution modes: `record`, `replay`, `golden`
- Domain allowlisting with policy integration
- TLS enforcement
- Metadata endpoint blocking

### ✅ **Production-Ready Infrastructure**
- Reuses existing `greenlang/intelligence/determinism.py` (1200+ lines, production-tested)
- Reuses existing `greenlang/intelligence/rag/hashing.py` (canonical hashing utilities)
- Integrates with UnifiedContextManager
- Pack system support (connectors as packs)

### ✅ **Type-Safe & Modern**
- Async-first interface (`async def fetch`)
- Generic type parameters (`Connector[TQuery, TPayload, TConfig]`)
- Pydantic v2 models throughout
- Decimal for precision-critical values
- UTC-only timestamps (timezone-aware)

## Quick Start

### Basic Usage

```python
import asyncio
from datetime import datetime, timezone
from greenlang.connectors.grid.mock import GridIntensityMockConnector
from greenlang.connectors.models import GridIntensityQuery, TimeWindow
from greenlang.connectors.context import ConnectorContext

# Create connector
connector = GridIntensityMockConnector()

# Create query
query = GridIntensityQuery(
    region="CA-ON",  # ISO 3166-2 region code
    window=TimeWindow(
        start=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc),
        resolution="hour"
    )
)

# RECORD mode: Generate fresh data and cache snapshot
ctx = ConnectorContext.for_record("grid/intensity/mock")
payload, provenance = asyncio.run(connector.fetch(query, ctx))

print(f"Data points: {len(payload.series)}")
print(f"First value: {payload.series[0].value} gCO2/kWh")
print(f"Provenance: {provenance.connector_id} v{provenance.connector_version}")
```

### Replay Mode (Deterministic)

```python
# REPLAY mode: Use cached snapshot only (no network)
ctx = ConnectorContext.for_replay(
    "grid/intensity/mock",
    snapshot_path=Path("snapshots/my-snapshot.snap.json")
)
payload, provenance = asyncio.run(connector.fetch(query, ctx))
# Returns cached data, byte-exact reproducibility
```

### Golden Testing

```python
# GOLDEN mode: Use pre-recorded golden snapshot
ctx = ConnectorContext.for_golden(
    "grid/intensity/mock",
    snapshot_path=Path("tests/goldens/connectors/grid/mock_CA-ON_2025-01-01_24h.snap.json")
)
payload, provenance = asyncio.run(connector.fetch(query, ctx))
# Deterministic testing with version-controlled golden files
```

## Data Models

### TimeWindow
```python
class TimeWindow(BaseModel):
    start: datetime  # UTC, timezone-aware
    end: datetime    # UTC, timezone-aware
    resolution: Literal["hour", "day", "month"] = "hour"
```

### TSPoint (Time Series Point)
```python
class TSPoint(BaseModel):
    ts: datetime  # UTC timestamp
    value: Decimal  # Exact precision
    unit: str
    quality: Literal["estimated", "measured", "simulated", "forecast"]
```

### GridIntensityQuery
```python
class GridIntensityQuery(BaseModel):
    region: str  # ISO 3166-2 (e.g., CA-ON, US-CAISO, EU-DE)
    window: TimeWindow
```

### GridIntensityPayload
```python
class GridIntensityPayload(BaseModel):
    series: List[TSPoint]  # Time-ordered data points
    region: str
    unit: Literal["gCO2/kWh", "kgCO2/MWh", "gCO2e/kWh"]
    resolution: Literal["hour", "day", "month"]
    metadata: Optional[dict]
```

## Region Codes

Connectors use ISO 3166-2 region codes with extensions for grid operators:

| Code | Region | Grid Operator |
|------|--------|---------------|
| `CA-ON` | Ontario, Canada | IESO |
| `US-CAISO` | California ISO, USA | CAISO |
| `US-PJM` | PJM Interconnection | PJM |
| `EU-DE` | Germany | Multiple |
| `IN-NO` | Northern Grid, India | NRLDC |
| `UK-GB` | Great Britain | National Grid ESO |

## Mock Grid Intensity Connector

### Algorithm

The `GridIntensityMockConnector` generates deterministic synthetic carbon intensity data:

1. **Seed**: `SHA-256(region + start + end + resolution)`
2. **Base curve**: Seasonal sinusoid with daily pattern
   ```
   value = 300 + 200*sin(2π*hour/24) + region_offset
   ```
3. **Region offset**: `int(seed, 16) % 200` (deterministic variation by region)
4. **Clamping**: `[50, 900]` gCO2/kWh
5. **Quantization**: `0.1 g` precision using Decimal
6. **Quality**: All points marked as `"simulated"`

### Properties

- **Deterministic**: Same inputs → same outputs (always)
- **No network**: Even in record mode, no external calls
- **Byte-exact**: Snapshots are reproducible across platforms
- **Time-series**: Hourly resolution with UTC timestamps

## Execution Modes

### Record Mode
- Generates fresh data
- Optionally caches snapshot for replay
- Network allowed (if configured in context)

```python
ctx = ConnectorContext.for_record(
    "grid/intensity/mock",
    allow_egress=True,  # Allow network
    egress_allowlist=["api.example.com"]  # Allowed domains
)
```

### Replay Mode
- Uses cached snapshot only
- Network completely blocked (security enforcement)
- Fails if snapshot missing

```python
ctx = ConnectorContext.for_replay(
    "grid/intensity/mock",
    snapshot_path=Path("snapshot.snap.json")
)
```

### Golden Mode
- Uses pre-recorded golden snapshot
- For CI/CD and reference testing
- Snapshots version-controlled

```python
ctx = ConnectorContext.for_golden(
    "grid/intensity/mock",
    snapshot_path=Path("tests/goldens/...")
)
```

## Snapshot Format

Snapshots use canonical JSON for byte-exact reproducibility:

```json
{
  "connector_id": "grid/intensity/mock",
  "connector_version": "0.1.0",
  "payload": {
    "series": [
      {"ts": "2025-01-01T00:00:00Z", "value": "421", "unit": "gCO2/kWh", ...},
      ...
    ],
    "region": "CA-ON",
    "resolution": "hour",
    "metadata": {...}
  },
  "provenance": {
    "connector_id": "grid/intensity/mock",
    "mode": "record",
    "query_hash": "4cb569b738...",
    "schema_hash": "2cf663a8a0...",
    "seed": "51c3a28834...",
    "snapshot_id": "f1eb6d51f6...",
    "timestamp": "2025-10-09T04:45:51.533310Z"
  },
  "schema": {
    "payload_model": "GridIntensityPayload",
    "provenance_model": "ConnectorProvenance"
  }
}
```

### Canonical Features
- **Sorted keys**: All dict keys alphabetically sorted
- **UTF-8 encoding**: No ASCII escaping
- **No whitespace**: Compact JSON (`,` separator, `:` key-value)
- **Decimal → string**: `"421"` not `421.0` (precision stability)
- **UTC timestamps**: Always `"Z"` suffix, never `"+00:00"`
- **Snapshot ID**: SHA-256 of canonical bytes

## Creating Custom Connectors

### Step 1: Define Models

```python
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal

class MyQuery(BaseModel):
    region: str
    start: datetime
    end: datetime

class MyPayload(BaseModel):
    data: List[dict]
    metadata: dict
```

### Step 2: Implement Connector

```python
from greenlang.connectors.base import Connector, ConnectorCapabilities, ConnectorProvenance
from greenlang.connectors.context import ConnectorContext
from typing import Tuple

class MyConnector(Connector[MyQuery, MyPayload, None]):
    connector_id = "my-connector"
    connector_version = "1.0.0"

    @property
    def capabilities(self) -> ConnectorCapabilities:
        return ConnectorCapabilities(
            supports_time_series=True,
            requires_auth=True,
            min_resolution="hour"
        )

    async def fetch(
        self,
        query: MyQuery,
        ctx: ConnectorContext
    ) -> Tuple[MyPayload, ConnectorProvenance]:
        # Check egress if needed
        if ctx.mode != CacheMode.REPLAY:
            ctx.check_egress("https://api.example.com")

        # Fetch data...
        payload = MyPayload(...)

        # Build provenance
        prov = ConnectorProvenance(
            connector_id=self.connector_id,
            connector_version=self.connector_version,
            mode=ctx.mode.value,
            query_hash=compute_query_hash(query),
            schema_hash=compute_schema_hash(MyPayload),
            seed=None
        )

        return payload, prov
```

### Step 3: Package as Pack

```yaml
# pack.yaml
name: my-connector-pack
kind: connector
version: 1.0.0
license: MIT

contents:
  connectors:
    - MyConnector

capabilities:
  net:
    allow: true
    egress_allowlist:
      - "api.example.com:443"

policy:
  network:
    - "api.example.com"
```

## Testing

### Unit Tests

```python
import pytest
import asyncio
from greenlang.connectors.grid.mock import GridIntensityMockConnector
from greenlang.connectors.context import ConnectorContext

def test_deterministic():
    connector = GridIntensityMockConnector()
    ctx = ConnectorContext.for_record("grid/intensity/mock")

    payload1, _ = asyncio.run(connector.fetch(query, ctx))
    payload2, _ = asyncio.run(connector.fetch(query, ctx))

    assert payload1.series == payload2.series
```

### Golden Tests

```python
def test_golden_snapshot():
    connector = GridIntensityMockConnector()
    ctx = ConnectorContext.for_golden(
        "grid/intensity/mock",
        snapshot_path=Path("tests/goldens/connectors/grid/mock_CA-ON_2025-01-01_24h.snap.json")
    )

    payload, _ = asyncio.run(connector.fetch(query, ctx))
    assert len(payload.series) == 24
```

### Property Tests (Hypothesis)

```python
from hypothesis import given, strategies as st

@given(
    hours=st.integers(min_value=1, max_value=8760),
    region=st.sampled_from(["CA-ON", "US-CAISO", "EU-DE"])
)
def test_series_length_matches_hours(hours, region):
    connector = GridIntensityMockConnector()
    # ... test that len(payload.series) == hours
```

## Security Model

### Default Deny
- Network egress **disabled by default**
- Must explicitly allow in context
- Replay mode **always blocks** network

### Egress Control
```python
ctx = ConnectorContext.for_record(
    "my-connector",
    allow_egress=True,
    egress_allowlist=["api.example.com", "*.trusted-domain.com"],
    require_tls=True
)
```

### Policy Integration
Connectors integrate with OPA/Rego policies:

```rego
# policies/connectors.rego
package connectors

default allow = false

allow {
    input.mode == "record"
    input.connector_id == "grid/intensity/mock"
    match_domain(input.url, input.allowlist[_])
}

deny_reason["Network egress in replay mode"] {
    input.mode == "replay"
    input.request_type == "network"
}
```

## Error Handling

### Structured Errors

```python
from greenlang.connectors.errors import (
    ConnectorReplayRequired,
    ConnectorNetworkError,
    ConnectorAuthError,
    classify_connector_error
)

try:
    payload, prov = await connector.fetch(query, ctx)
except ConnectorReplayRequired as e:
    print(f"Hint: {e.context['hint']}")
except ConnectorNetworkError as e:
    print(f"Network failed: {e.message}")
```

### Error Classification

```python
try:
    response = requests.get(url)
except Exception as e:
    raise classify_connector_error(e, "my-connector", url)
```

## Pack Integration

### Loading Connectors from Packs

```python
from greenlang.packs.loader import PackLoader

loader = PackLoader()
pack = loader.load("grid-intensity-pack")

# Get connector class
ConnectorClass = pack.get_connector("GridIntensityMockConnector")

# Instantiate and use
connector = ConnectorClass()
```

## Provenance Tracking

Every connector fetch returns provenance metadata:

```python
class ConnectorProvenance(BaseModel):
    connector_id: str             # "grid/intensity/mock"
    connector_version: str        # "0.1.0"
    mode: str                     # "record" | "replay" | "golden"
    query_hash: str               # SHA-256 of query
    schema_hash: str              # SHA-256 of payload schema
    seed: Optional[str]           # Deterministic seed (if applicable)
    snapshot_id: Optional[str]    # SHA-256 of snapshot
    timestamp: datetime           # UTC timestamp
    metadata: Dict[str, Any]      # Additional metadata
```

## Next Steps

### Week 3 (Future)
- Real connectors: ElectricityMaps, WattTime, NREL
- Streaming support: `async def stream()`
- Pagination helpers
- Rate limiting with budgets
- Circuit breakers
- Retry policies with exponential backoff

### Week 6 (Future)
- ConnectorSpec v1 (following AgentSpec v2)
- Policy extensions (connectors.rego)
- Metrics and telemetry
- Performance benchmarks

## References

- **Architecture Review**: See project documentation for detailed design rationale
- **LLM Provider Pattern**: `greenlang/intelligence/providers/base.py`
- **Determinism Infrastructure**: `greenlang/intelligence/determinism.py`
- **Hashing Utilities**: `greenlang/intelligence/rag/hashing.py`
- **Pack System**: `greenlang/packs/loader.py`

---

**Implementation**: DATA-301 (Week 1 Deliverable)
**Status**: ✅ Complete
**Test Coverage**: Basic tests + golden fixtures
**Documentation**: Complete
**Integration**: Pack system, context manager, determinism infrastructure
