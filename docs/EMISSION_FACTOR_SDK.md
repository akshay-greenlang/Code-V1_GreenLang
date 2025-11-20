# Emission Factor SDK Documentation

**Version:** 1.0.0
**Last Updated:** 2025-01-19
**Status:** Production Ready

## Overview

The GreenLang Emission Factor SDK provides a production-grade infrastructure for managing and calculating with emission factors. It includes:

- **SQLite Database**: 327+ emission factors with full provenance tracking
- **Python SDK**: Type-safe client with <10ms lookups and <100ms calculations
- **CLI Tool**: Command-line interface for queries and calculations
- **Zero-Hallucination**: Deterministic calculations with complete audit trails

## Quick Start

### Installation

```bash
# Install GreenLang (includes emission factor SDK)
pip install greenlang

# Or install from source
cd Code-V1_GreenLang
pip install -e .
```

### Import Emission Factors

```bash
# Import all YAML files into database
python scripts/import_emission_factors.py --overwrite

# Output:
# Total factors processed: 327
# Successfully imported: 327
# Failed imports: 0
```

### Basic Usage (Python SDK)

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

# Initialize client
client = EmissionFactorClient()

# Get emission factor
factor = client.get_factor("fuels_diesel")
print(f"Diesel: {factor.emission_factor_kg_co2e} kg CO2e/liter")

# Calculate emissions
result = client.calculate_emissions(
    factor_id="fuels_diesel",
    activity_amount=100.0,
    activity_unit="gallon"
)

print(f"Emissions: {result.emissions_kg_co2e:.2f} kg CO2e")
print(f"Audit Hash: {result.audit_trail}")

# Close connection
client.close()
```

### Basic Usage (CLI)

```bash
# Search for factors
greenlang factors search "diesel"

# Get factor details
greenlang factors get fuels_diesel

# Calculate emissions
greenlang factors calculate --factor=fuels_diesel --amount=100 --unit=gallon

# Output:
# Activity:     100.00 gallons
# Emissions:    1,021.00 kg CO2e
# Audit Hash:   abc123...
```

## Architecture

### Components

```
greenlang/
├── data/
│   └── emission_factors.db          # SQLite database (327+ factors)
├── db/
│   └── emission_factors_schema.py   # Database schema & management
├── models/
│   └── emission_factor.py           # Pydantic data models
├── sdk/
│   └── emission_factor_client.py    # Python SDK client
└── cli/
    └── factor_query.py              # CLI tool

scripts/
└── import_emission_factors.py       # YAML import script

tests/
└── test_emission_factors.py         # Test suite (85%+ coverage)
```

### Database Schema

```sql
-- Main emission factors table
CREATE TABLE emission_factors (
    factor_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT,
    emission_factor_value REAL NOT NULL,
    unit TEXT NOT NULL,
    scope TEXT,
    source_org TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    last_updated DATE NOT NULL,
    geographic_scope TEXT,
    data_quality_tier TEXT,
    ...
);

-- Additional units (e.g., diesel in liters AND gallons)
CREATE TABLE factor_units (
    factor_id TEXT,
    unit_name TEXT,
    emission_factor_value REAL NOT NULL,
    FOREIGN KEY (factor_id) REFERENCES emission_factors(factor_id)
);

-- Gas breakdown (e.g., CO2, CH4, N2O)
CREATE TABLE factor_gas_vectors (
    factor_id TEXT,
    gas_type TEXT,
    kg_per_unit REAL NOT NULL,
    FOREIGN KEY (factor_id) REFERENCES emission_factors(factor_id)
);

-- Calculation audit log
CREATE TABLE calculation_audit_log (
    calculation_id TEXT NOT NULL UNIQUE,
    factor_id TEXT NOT NULL,
    activity_amount REAL NOT NULL,
    emissions_kg_co2e REAL NOT NULL,
    audit_hash TEXT NOT NULL,
    ...
);
```

### Data Models

```python
@dataclass
class EmissionFactor:
    """Core emission factor model."""
    factor_id: str
    name: str
    emission_factor_kg_co2e: float
    unit: str
    scope: str
    source: SourceProvenance
    geography: Geography
    data_quality: DataQualityScore
    last_updated: date
    additional_units: List[EmissionFactorUnit]
    gas_vectors: List[GasVector]

@dataclass
class EmissionResult:
    """Calculation result with audit trail."""
    activity_amount: float
    activity_unit: str
    emissions_kg_co2e: float
    factor_used: EmissionFactor
    calculation_timestamp: datetime
    audit_trail: str  # SHA-256 hash
    warnings: List[str]
```

## Python SDK Usage

### Initialize Client

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

# Default database location
client = EmissionFactorClient()

# Custom database path
client = EmissionFactorClient(db_path="/path/to/emission_factors.db")

# Context manager (auto-close)
with EmissionFactorClient() as client:
    factor = client.get_factor("fuels_diesel")
```

### Query Factors

#### Get Factor by ID

```python
# Get factor by ID
factor = client.get_factor("fuels_diesel")

print(f"Name: {factor.name}")
print(f"Value: {factor.emission_factor_kg_co2e} kg CO2e/{factor.unit}")
print(f"Source: {factor.source.source_org}")
print(f"URI: {factor.source.source_uri}")
print(f"Updated: {factor.last_updated}")

# Check for multiple units
for unit in factor.additional_units:
    print(f"  {unit.emission_factor_value} kg CO2e/{unit.unit_name}")
```

#### Search by Name

```python
# Search by name (case-insensitive, partial match)
factors = client.get_factor_by_name("diesel")

for factor in factors:
    print(f"{factor.factor_id}: {factor.name}")
```

#### Search by Category

```python
# Get all fuels
fuels = client.get_by_category("fuels")

# Get all grid factors
grids = client.get_by_category("grids")
```

#### Search by Scope

```python
# Get all Scope 1 factors
scope1 = client.get_by_scope("Scope 1")

# Get all Scope 2 factors
scope2 = client.get_by_scope("Scope 2 - Location-Based")
```

#### Advanced Search

```python
from greenlang.models.emission_factor import FactorSearchCriteria, DataQualityTier

criteria = FactorSearchCriteria(
    category="fuels",
    scope="Scope 1",
    geographic_scope="United States",
    min_quality_tier=DataQualityTier.TIER_2,
    max_age_years=3
)

factors = client.search_factors(criteria)
```

### Calculate Emissions

#### Basic Calculation

```python
result = client.calculate_emissions(
    factor_id="fuels_diesel",
    activity_amount=100.0,
    activity_unit="gallon"
)

print(f"Activity: {result.activity_amount} {result.activity_unit}")
print(f"Emissions: {result.emissions_kg_co2e:.2f} kg CO2e")
print(f"Emissions: {result.emissions_metric_tons_co2e:.4f} metric tons CO2e")
print(f"Factor Used: {result.factor_used.name}")
print(f"Factor Value: {result.factor_value_applied} kg CO2e/{result.activity_unit}")
print(f"Audit Hash: {result.audit_trail}")
```

#### With Geographic Fallback

```python
# Try to find California-specific grid factor
# Falls back to US average if California not found
result = client.calculate_emissions(
    factor_id="grids_us_caiso",
    activity_amount=1000.0,
    activity_unit="kwh",
    geography="California"
)
```

#### With Temporal Matching

```python
# Use 2023 emission factor (if available)
result = client.calculate_emissions(
    factor_id="grids_us_caiso",
    activity_amount=1000.0,
    activity_unit="kwh",
    year=2023
)
```

#### Handle Warnings

```python
result = client.calculate_emissions(
    factor_id="fuels_diesel",
    activity_amount=100.0,
    activity_unit="gallon"
)

if result.warnings:
    print("Warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")
```

### Specialized Queries

#### Get Grid Emission Factor

```python
# Get grid factor by region
grid_factor = client.get_grid_factor("CAISO")
print(f"CAISO Grid: {grid_factor.emission_factor_kg_co2e} kg CO2e/kWh")

# With year
grid_factor = client.get_grid_factor("CAISO", year=2023)
```

#### Get Fuel Emission Factor

```python
# Get fuel factor
fuel_factor = client.get_fuel_factor("diesel", unit="gallon")
print(f"Diesel: {fuel_factor.emission_factor_kg_co2e} kg CO2e/gallon")
```

### Database Statistics

```python
stats = client.get_statistics()

print(f"Total Factors: {stats['total_factors']}")
print(f"Total Calculations: {stats['total_calculations']}")
print(f"Stale Factors: {stats['stale_factors']}")

print("\nBy Category:")
for category, count in stats['by_category'].items():
    print(f"  {category}: {count}")

print("\nBy Scope:")
for scope, count in stats['by_scope'].items():
    print(f"  {scope}: {count}")
```

## CLI Tool Usage

### Installation

The CLI tool is included with GreenLang:

```bash
pip install greenlang
```

### Commands

#### List Factors

```bash
# List all factors
greenlang factors list

# Filter by category
greenlang factors list --category=fuels

# Filter by scope
greenlang factors list --scope="Scope 1"
```

#### Search Factors

```bash
# Search by name
greenlang factors search "diesel"

# Output:
# ┌──────────────────────┬─────────────────┬──────────┬────────┬───────┬──────────┐
# │ Factor ID            │ Name            │ Category │ Value  │ Unit  │ Geography│
# ├──────────────────────┼─────────────────┼──────────┼────────┼───────┼──────────┤
# │ fuels_diesel         │ Diesel Fuel     │ fuels    │ 2.6800 │ liter │ US       │
# └──────────────────────┴─────────────────┴──────────┴────────┴───────┴──────────┘
```

#### Get Factor Details

```bash
# Get complete factor information
greenlang factors get fuels_diesel

# Output:
# ======================================================================
# EMISSION FACTOR: fuels_diesel
# ======================================================================
# Name:           Diesel Fuel
# Category:       fuels
#
# EMISSION FACTOR:
#   Value:        2.680000 kg CO2e/liter
#   Unit:         liter
#
# ADDITIONAL UNITS:
#   10.210000 kg CO2e/gallon
#
# SOURCE PROVENANCE:
#   Organization: EPA
#   URI:          https://www.epa.gov/...
#   Standard:     GHG Protocol
# ...
```

#### Calculate Emissions

```bash
# Basic calculation
greenlang factors calculate \
    --factor=fuels_diesel \
    --amount=100 \
    --unit=gallon

# Output:
# ======================================================================
# EMISSION CALCULATION RESULT
# ======================================================================
#
# ACTIVITY:
#   Amount:       100.00 gallons
#
# EMISSION FACTOR USED:
#   Factor ID:    fuels_diesel
#   Name:         Diesel Fuel
#   Value:        10.210000 kg CO2e/gallon
#   Source:       EPA
#
# CALCULATION:
#   100.00 gallons × 10.210000 kg CO2e/gallon
#   = 1,021.00 kg CO2e
#   = 1.0210 metric tons CO2e
#
# AUDIT TRAIL:
#   Timestamp:    2025-01-19T10:30:45
#   Hash:         abc123def456...
```

#### Database Statistics

```bash
# Show database statistics
greenlang factors stats

# Output:
# ======================================================================
# EMISSION FACTOR DATABASE STATISTICS
# ======================================================================
#
# Total Factors:        327
# Total Calculations:   1,234
# Stale Factors:        5
#
# BY CATEGORY:
#   fuels                    78 factors
#   grids                    50 factors
#   materials                45 factors
#   transportation           40 factors
# ...
```

#### Validate Database

```bash
# Validate database integrity
greenlang factors validate-db

# Output:
# ✓ Database validation PASSED
#
# STATISTICS:
#   total_factors        327
#   categories           15
#   sources              8
#
# WARNINGS:
#   ⚠ 5 factors older than 3 years
```

#### Database Info

```bash
# Show database information
greenlang factors info

# Output:
# ======================================================================
# DATABASE INFORMATION
# ======================================================================
#
# File:      C:/Users/.../emission_factors.db
# Size:      2.45 MB
#
# TABLES:
#   emission_factors                 327 rows
#   factor_units                     156 rows
#   factor_gas_vectors                45 rows
#   calculation_audit_log          1,234 rows
#
# INDEXES:   15
# VIEWS:     4
```

## Zero-Hallucination Calculation

### Allowed (Deterministic)

```python
# ✅ Database lookups
factor = client.get_factor("fuels_diesel")

# ✅ Python arithmetic
emissions = activity_amount * factor.emission_factor_kg_co2e

# ✅ Unit conversion
gallon_factor = factor.get_factor_for_unit("gallon")
emissions = gallons * gallon_factor

# ✅ Gas vector summation
total_co2e = sum(gas.kg_per_unit * gas.gwp for gas in factor.gas_vectors)
```

### Not Allowed (Hallucination Risk)

```python
# ❌ LLM for numeric calculations
emissions = llm.calculate_emissions(activity_data)  # NEVER DO THIS

# ❌ ML model predictions for regulatory values
value = ml_model.predict(features)  # NOT FOR COMPLIANCE

# ❌ Unvalidated external API calls without provenance
result = external_api.get_value()  # NO AUDIT TRAIL
```

## Fallback Logic

### Geographic Fallback

The SDK implements intelligent geographic fallback:

1. **Exact Match**: Try exact geographic scope
2. **State Match**: Try state/province
3. **Country Match**: Try country
4. **Regional Match**: Try region (e.g., North America)
5. **Global**: Fall back to global average

```python
# Example: California electricity
# 1. Try "CAISO" (California ISO)
# 2. Fall back to "California" state average
# 3. Fall back to "United States" country average
# 4. Fall back to "North America" regional average
# 5. Fall back to "Global" average

factor = client.get_grid_factor("California")
```

### Temporal Fallback

The SDK implements temporal matching:

1. **Exact Year**: Try exact year match
2. **Most Recent**: Use most recently updated factor
3. **Warning**: Warn if factor is >3 years old

```python
# Example: 2023 grid factor
# 1. Try 2023 factor
# 2. Use most recent factor (e.g., 2024)
# 3. Warn if factor is from 2020 (>3 years old)

factor = client.get_factor("grids_us_caiso", year=2023)
```

## Error Handling

### Exception Types

```python
from greenlang.sdk.emission_factor_client import (
    EmissionFactorNotFoundError,
    UnitNotAvailableError,
    DatabaseConnectionError
)

try:
    factor = client.get_factor("nonexistent_factor")
except EmissionFactorNotFoundError as e:
    print(f"Factor not found: {e}")
    # Suggestion: Use client.get_factor_by_name() to search

try:
    result = client.calculate_emissions(
        "fuels_diesel", 100.0, "invalid_unit"
    )
except UnitNotAvailableError as e:
    print(f"Unit error: {e}")
    # Error message includes available units

try:
    client = EmissionFactorClient(db_path="/invalid/path.db")
except DatabaseConnectionError as e:
    print(f"Database error: {e}")
```

### Validation

```python
# Negative activity amounts raise ValueError
try:
    result = client.calculate_emissions("fuels_diesel", -100.0, "gallon")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Performance

### Requirements

- **Factor Lookup**: <10ms
- **Calculation**: <100ms (including audit logging)
- **Database Size**: ~2-5 MB for 327 factors

### Optimization

```python
# Enable caching (default)
client = EmissionFactorClient(enable_cache=True, cache_size=10000)

# Caching uses LRU (Least Recently Used) strategy
# Repeated lookups are nearly instant (<1ms)

# Example: Batch calculations
results = []
for activity in activities:
    # First call: ~5ms (database query)
    # Subsequent calls: <1ms (cached)
    result = client.calculate_emissions(
        "fuels_diesel", activity.amount, activity.unit
    )
    results.append(result)
```

## Integration Examples

### Flask API

```python
from flask import Flask, request, jsonify
from greenlang.sdk.emission_factor_client import EmissionFactorClient

app = Flask(__name__)
client = EmissionFactorClient()

@app.route('/api/calculate', methods=['POST'])
def calculate_emissions():
    data = request.json

    try:
        result = client.calculate_emissions(
            factor_id=data['factor_id'],
            activity_amount=float(data['amount']),
            activity_unit=data['unit']
        )

        return jsonify({
            'emissions_kg_co2e': result.emissions_kg_co2e,
            'emissions_metric_tons_co2e': result.emissions_metric_tons_co2e,
            'audit_trail': result.audit_trail,
            'warnings': result.warnings
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()
```

### Pandas Integration

```python
import pandas as pd
from greenlang.sdk.emission_factor_client import EmissionFactorClient

client = EmissionFactorClient()

# Load activity data
df = pd.read_csv('activity_data.csv')
# Columns: fuel_type, amount, unit

# Calculate emissions for each row
def calculate_row(row):
    try:
        result = client.calculate_emissions(
            factor_id=f"fuels_{row['fuel_type']}",
            activity_amount=row['amount'],
            activity_unit=row['unit']
        )
        return result.emissions_kg_co2e
    except Exception as e:
        return None

df['emissions_kg_co2e'] = df.apply(calculate_row, axis=1)

print(df[['fuel_type', 'amount', 'unit', 'emissions_kg_co2e']])
```

### Agent Integration

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient
from greenlang.agents import BaseAgent

class EmissionCalculatorAgent(BaseAgent):
    """Agent that calculates emissions."""

    def __init__(self, config):
        super().__init__(config)
        self.ef_client = EmissionFactorClient()

    def process(self, input_data):
        """Calculate emissions for activity data."""
        results = []

        for activity in input_data.activities:
            result = self.ef_client.calculate_emissions(
                factor_id=activity.factor_id,
                activity_amount=activity.amount,
                activity_unit=activity.unit
            )

            results.append({
                'activity_id': activity.id,
                'emissions_kg_co2e': result.emissions_kg_co2e,
                'audit_trail': result.audit_trail
            })

        return results
```

## Database Management

### Import YAML Files

```bash
# Import emission factors from YAML
python scripts/import_emission_factors.py --overwrite

# Custom database path
python scripts/import_emission_factors.py \
    --db-path /path/to/custom.db \
    --data-dir /path/to/yaml/files
```

### Backup Database

```bash
# Backup database
cp greenlang/data/emission_factors.db \
   greenlang/data/emission_factors_backup_$(date +%Y%m%d).db
```

### Update Factors

```python
import sqlite3

conn = sqlite3.connect("greenlang/data/emission_factors.db")
cursor = conn.cursor()

# Update factor value
cursor.execute("""
    UPDATE emission_factors
    SET emission_factor_value = 2.70,
        last_updated = DATE('now')
    WHERE factor_id = 'fuels_diesel'
""")

conn.commit()
conn.close()
```

## Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/test_emission_factors.py -v

# Run specific test class
pytest tests/test_emission_factors.py::TestEmissionFactorClient -v

# Run with coverage
pytest tests/test_emission_factors.py --cov=greenlang.sdk --cov-report=html
```

### Test Results

```
Target: 85%+ test coverage

test_emission_factors.py::TestDatabaseSchema::test_create_database PASSED
test_emission_factors.py::TestDatabaseSchema::test_database_indexes PASSED
test_emission_factors.py::TestEmissionFactorClient::test_get_factor PASSED
test_emission_factors.py::TestEmissionFactorClient::test_calculate_emissions PASSED
test_emission_factors.py::TestPerformance::test_factor_lookup_performance PASSED
test_emission_factors.py::TestPerformance::test_calculation_performance PASSED

========================= 35 passed in 2.45s =========================
Coverage: 87%
```

## Troubleshooting

### Database Not Found

```python
# Error: Database not found
DatabaseConnectionError: Database not found: /path/to/emission_factors.db

# Solution: Import YAML files first
python scripts/import_emission_factors.py
```

### Factor Not Found

```python
# Error: Emission factor not found
EmissionFactorNotFoundError: Emission factor not found: invalid_factor

# Solution: Search for available factors
factors = client.get_factor_by_name("diesel")
print([f.factor_id for f in factors])
```

### Unit Not Available

```python
# Error: Unit 'pounds' not available
UnitNotAvailableError: Unit 'pounds' not available for fuels_diesel.
Available units: liter, gallon

# Solution: Use available units
result = client.calculate_emissions("fuels_diesel", 100.0, "gallon")
```

## Roadmap

### Phase 1 (Current): 327 Factors
- ✅ Core fuels (diesel, gasoline, natural gas, coal)
- ✅ US electricity grids (26 eGRID subregions)
- ✅ Basic transportation
- ✅ Common materials

### Phase 2 (Q2 2025): 500 Factors
- ⏳ Extended transportation (60+ factors)
- ⏳ Global electricity grids (100+ countries)
- ⏳ Detailed materials (steel, cement, plastics)
- ⏳ Water and waste

### Phase 3 (Q3 2025): 1000+ Factors
- ⏳ Supply chain categories (Scope 3)
- ⏳ Industry-specific factors
- ⏳ Regional variations (EU, Asia-Pacific)
- ⏳ Temporal trends (historical factors)

## Support

### Documentation
- GitHub: https://github.com/greenlang/greenlang
- Docs: https://docs.greenlang.io
- API Reference: https://api-docs.greenlang.io

### Community
- Discord: https://discord.gg/greenlang
- Forum: https://forum.greenlang.io

### Commercial Support
- Email: support@greenlang.io
- Enterprise: enterprise@greenlang.io

## License

Copyright 2025 GreenLang. All rights reserved.

Licensed under the Apache License 2.0.
