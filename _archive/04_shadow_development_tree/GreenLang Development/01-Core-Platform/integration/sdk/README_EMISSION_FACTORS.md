# Emission Factor SDK - Quick Reference

## Installation & Setup

```bash
# 1. Import emission factors
python scripts/import_emission_factors.py --overwrite

# 2. Verify database
python -c "from greenlang.sdk.emission_factor_client import EmissionFactorClient; print(EmissionFactorClient().get_statistics())"
```

## Python Quick Start

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

# Initialize
client = EmissionFactorClient()

# Calculate emissions
result = client.calculate_emissions(
    factor_id="fuels_diesel",
    activity_amount=100.0,
    activity_unit="gallon"
)

print(f"Emissions: {result.emissions_kg_co2e:.2f} kg CO2e")
# Output: Emissions: 1021.00 kg CO2e

client.close()
```

## CLI Quick Start

```bash
# Search for factors
greenlang factors search "diesel"

# Calculate emissions
greenlang factors calculate --factor=fuels_diesel --amount=100 --unit=gallon

# Database stats
greenlang factors stats
```

## Key Features

- **Zero Hallucination**: Deterministic calculations only (no LLM in calculation path)
- **Fast Performance**: <10ms factor lookups, <100ms calculations
- **Complete Audit Trail**: SHA-256 provenance hash for every calculation
- **327+ Factors**: Fuels, electricity grids, materials, transportation
- **Type Safe**: Pydantic models with full validation
- **Geographic Fallback**: State → Country → Region → Global
- **Test Coverage**: 85%+ with comprehensive test suite

## File Structure

```
C:\Users\aksha\Code-V1_GreenLang\
├── greenlang/
│   ├── data/
│   │   └── emission_factors.db          # SQLite database (created by import script)
│   ├── db/
│   │   └── emission_factors_schema.py   # Schema creation & validation
│   ├── models/
│   │   └── emission_factor.py           # Data models (EmissionFactor, EmissionResult)
│   ├── sdk/
│   │   └── emission_factor_client.py    # Main SDK client
│   └── cli/
│       └── factor_query.py              # CLI tool
├── scripts/
│   └── import_emission_factors.py       # YAML → SQLite import script
├── tests/
│   └── test_emission_factors.py         # Test suite
├── docs/
│   └── EMISSION_FACTOR_SDK.md           # Full documentation
└── data/
    ├── emission_factors_registry.yaml            # 78 base factors
    ├── emission_factors_expansion_phase1.yaml    # +172 factors
    └── emission_factors_expansion_phase2.yaml    # +77 factors (327 total)
```

## Common Operations

### Query Operations

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

client = EmissionFactorClient()

# Get specific factor
factor = client.get_factor("fuels_diesel")

# Search by name
factors = client.get_factor_by_name("diesel")

# Get by category
fuels = client.get_by_category("fuels")
grids = client.get_by_category("grids")

# Get by scope
scope1 = client.get_by_scope("Scope 1")
scope2 = client.get_by_scope("Scope 2 - Location-Based")

# Specialized queries
grid_factor = client.get_grid_factor("CAISO")
fuel_factor = client.get_fuel_factor("diesel", unit="gallon")

client.close()
```

### Calculation Operations

```python
# Basic calculation
result = client.calculate_emissions(
    factor_id="fuels_diesel",
    activity_amount=100.0,
    activity_unit="gallon"
)

# Access results
print(f"Emissions: {result.emissions_kg_co2e} kg CO2e")
print(f"Emissions: {result.emissions_metric_tons_co2e} metric tons CO2e")
print(f"Factor Used: {result.factor_used.name}")
print(f"Audit Hash: {result.audit_trail}")

# Check warnings
if result.warnings:
    for warning in result.warnings:
        print(f"Warning: {warning}")
```

## Database Statistics

```python
stats = client.get_statistics()

print(f"Total Factors: {stats['total_factors']}")
print(f"Categories: {len(stats['by_category'])}")
print(f"Sources: {len(stats['by_source'])}")
print(f"Calculations: {stats['total_calculations']}")
```

## Error Handling

```python
from greenlang.sdk.emission_factor_client import (
    EmissionFactorNotFoundError,
    UnitNotAvailableError,
    DatabaseConnectionError
)

try:
    result = client.calculate_emissions("fuels_diesel", 100.0, "gallon")
except EmissionFactorNotFoundError:
    print("Factor not found")
except UnitNotAvailableError as e:
    print(f"Unit error: {e}")  # Includes available units
except ValueError:
    print("Invalid input (e.g., negative amount)")
```

## Performance Benchmarks

| Operation | Target | Typical |
|-----------|--------|---------|
| Factor Lookup | <10ms | ~5ms |
| Calculation | <100ms | ~20ms |
| Database Query | <50ms | ~15ms |
| Import 327 Factors | <60s | ~30s |

## Testing

```bash
# Run all tests
pytest tests/test_emission_factors.py -v

# Run with coverage
pytest tests/test_emission_factors.py --cov=greenlang.sdk --cov-report=html

# Expected: 85%+ coverage, all tests pass
```

## Troubleshooting

### Issue: Database not found
```bash
# Solution: Run import script
python scripts/import_emission_factors.py --overwrite
```

### Issue: Factor not found
```python
# Solution: Search for available factors
client = EmissionFactorClient()
factors = client.get_factor_by_name("your_search_term")
print([f.factor_id for f in factors])
```

### Issue: Unit not available
```python
# Solution: Check available units
factor = client.get_factor("fuels_diesel")
print(f"Primary unit: {factor.unit}")
print(f"Additional units: {[u.unit_name for u in factor.additional_units]}")
```

## Full Documentation

See `C:\Users\aksha\Code-V1_GreenLang\docs\EMISSION_FACTOR_SDK.md` for:
- Complete API reference
- Integration examples
- Advanced usage patterns
- Database management
- Architecture details

## Support

- GitHub Issues: https://github.com/greenlang/greenlang/issues
- Documentation: https://docs.greenlang.io
- Community: https://discord.gg/greenlang
