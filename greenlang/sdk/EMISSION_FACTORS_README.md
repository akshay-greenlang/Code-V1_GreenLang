# Emission Factor Client SDK

**Production-grade emission factor database with 500+ verified factors and zero-hallucination guarantee.**

## Quick Reference

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

# Calculate emissions
with EmissionFactorClient() as client:
    result = client.calculate_emissions(
        factor_id="fuels_diesel",
        activity_amount=100.0,
        activity_unit="gallon"
    )
    print(f"Emissions: {result.emissions_kg_co2e} kg CO2e")
    print(f"Source: {result.factor_used.source.source_uri}")
```

## Key Features

- ✅ 500+ verified emission factors
- ✅ Zero-hallucination calculations
- ✅ Complete provenance tracking
- ✅ <10ms performance
- ✅ Type-safe Pydantic models
- ✅ Unit conversions
- ✅ Geographic fallback
- ✅ SHA-256 audit trails

## Documentation

- **Quick Start:** `QUICK_START_EMISSION_FACTORS.md`
- **Migration Guide:** `docs/EMISSION_FACTOR_MIGRATION_GUIDE.md`
- **Examples:** `examples/emission_factor_integration_examples.py`
- **Tests:** `tests/integration/test_emission_factor_integration.py`

## Support

For detailed API reference, see docstrings in `greenlang/sdk/emission_factor_client.py`
