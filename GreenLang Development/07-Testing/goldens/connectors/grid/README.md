# Golden Fixtures for Grid Intensity Connectors

This directory contains golden fixture files for testing grid intensity connectors.

## Overview

Golden fixtures are deterministic, byte-exact snapshots of connector responses used for:
- Regression testing
- Replay mode validation
- Integration testing without network calls
- Ensuring connector output stability

## Structure

Each fixture consists of two files:
1. **`.snap.json`** - Canonical JSON snapshot containing payload and provenance
2. **`.sha256`** - SHA-256 hash of the snapshot for integrity verification

## Naming Convention

```
mock_{REGION}_{START_DATE}_{DURATION}.snap.json
mock_{REGION}_{START_DATE}_{DURATION}.snap.json.sha256
```

Example: `mock_US-CAISO_2025-01-01_24h.snap.json`

## Regions Covered

| Region | Name | Grid Operator | Fixtures |
|--------|------|---------------|----------|
| CA-ON | Ontario, Canada | IESO | 3 (24h, 48h, 168h) |
| US-CAISO | California ISO, USA | CAISO | 2 (24h, 48h) |
| US-PJM | PJM Interconnection, USA | PJM | 2 (24h, 48h) |
| EU-DE | Germany | Multiple | 2 (24h, 48h) |
| IN-NO | Northern Grid, India | NRLDC | 2 (24h, 48h) |
| UK-GB | Great Britain | National Grid ESO | 2 (24h, 48h) |

## Scenarios

| Duration | Description | Number of Regions |
|----------|-------------|-------------------|
| 24h | Single day (baseline) | 6 |
| 48h | Two days (multi-day) | 6 |
| 168h | One week (weekly pattern) | 1 (CA-ON only) |

## Total Fixtures

- **13 fixtures** (13 .snap.json files)
- **13 SHA-256 files** (13 .sha256 files)
- **6 regions** covered
- **3 scenario types** (24h, 48h, 168h)

## Disk Space

- Snapshot files: **63.1 KB**
- SHA-256 files: **832 bytes**
- **Total: 64.0 KB**

## Generation

Fixtures are generated using the deterministic mock connector algorithm:

1. Seed = SHA-256(region + start + end + resolution)
2. Generate hourly series with seasonal daily pattern
3. Base curve: sinusoid(hour) + region-hash offset
4. Clamp to [50, 900] gCO2/kWh
5. Quantize to 0.1 g (Decimal precision)

### Regeneration

To regenerate all fixtures:

```bash
python scripts/generate_golden_fixtures.py
```

To validate fixture integrity:

```bash
python scripts/validate_golden_fixtures.py
```

To generate a comprehensive report:

```bash
python scripts/fixture_report.py
```

## Usage in Tests

### Load a Golden Fixture

```python
from greenlang.connectors.context import ConnectorContext
from greenlang.connectors.grid.mock import GridIntensityMockConnector
from pathlib import Path

# Create context pointing to golden fixture
ctx = ConnectorContext.for_golden(
    connector_id="grid/intensity/mock",
    snapshot_path=Path("tests/goldens/connectors/grid/mock_US-CAISO_2025-01-01_24h.snap.json")
)

# Load connector
connector = GridIntensityMockConnector()

# Fetch will load from snapshot (no network call)
payload, prov = await connector.fetch(query, ctx)
```

### Validate Against Golden

```python
import hashlib
from pathlib import Path

# Read snapshot
snapshot_path = Path("tests/goldens/connectors/grid/mock_US-CAISO_2025-01-01_24h.snap.json")
snapshot_bytes = snapshot_path.read_bytes()

# Compute hash
actual_hash = hashlib.sha256(snapshot_bytes).hexdigest()

# Read expected hash
expected_hash = Path(f"{snapshot_path}.sha256").read_text().strip()

# Validate
assert actual_hash == expected_hash, f"Hash mismatch: {actual_hash} != {expected_hash}"
```

## Snapshot Format

Each snapshot is a canonical JSON file with this structure:

```json
{
  "connector_id": "grid/intensity/mock",
  "connector_version": "0.1.0",
  "payload": {
    "series": [
      {
        "ts": "2025-01-01T00:00:00Z",
        "value": "340",
        "unit": "gCO2/kWh",
        "quality": "simulated"
      }
    ],
    "region": "US-CAISO",
    "unit": "gCO2/kWh",
    "resolution": "hour",
    "metadata": {
      "connector": "grid/intensity/mock",
      "version": "0.1.0",
      "algorithm": "seasonal_sinusoid",
      "seed": "...",
      "region_offset": 121,
      "data_points": 24
    }
  },
  "provenance": {
    "connector_id": "grid/intensity/mock",
    "connector_version": "0.1.0",
    "mode": "record",
    "query_hash": "...",
    "schema_hash": "...",
    "seed": "...",
    "snapshot_id": "...",
    "timestamp": "2025-10-09T05:13:40.722224Z",
    "metadata": {
      "region": "US-CAISO",
      "start": "2025-01-01T00:00:00+00:00",
      "end": "2025-01-02T00:00:00+00:00",
      "resolution": "hour",
      "data_points": 24
    }
  },
  "schema": {
    "payload_model": "GridIntensityPayload",
    "provenance_model": "ConnectorProvenance"
  }
}
```

## Determinism

The mock connector is fully deterministic:
- Same query parameters → same payload data
- Seed is computed from query: SHA-256(region + start + end + resolution)
- Payload data is **byte-exact reproducible**
- Provenance timestamp varies (records generation time)

## Integrity Verification

All fixtures have been validated:
- SHA-256 hashes match stored values
- JSON is valid canonical format
- All regions have expected number of fixtures
- All scenarios are covered

## Maintenance

When updating fixtures:
1. Update connector algorithm if needed
2. Run `python scripts/generate_golden_fixtures.py`
3. Run `python scripts/validate_golden_fixtures.py` to verify
4. Commit both `.snap.json` and `.sha256` files
5. Update this README if adding new regions or scenarios

## Version History

- **2025-10-09**: Initial creation with 13 fixtures across 6 regions
  - Regions: CA-ON, US-CAISO, US-PJM, EU-DE, IN-NO, UK-GB
  - Scenarios: 24h, 48h, 168h
  - Total size: 64.0 KB

## See Also

- `scripts/generate_golden_fixtures.py` - Generation script
- `scripts/validate_golden_fixtures.py` - Validation script
- `scripts/fixture_report.py` - Reporting script
- `greenlang/connectors/grid/mock.py` - Mock connector implementation
- `greenlang/connectors/snapshot.py` - Snapshot utilities
