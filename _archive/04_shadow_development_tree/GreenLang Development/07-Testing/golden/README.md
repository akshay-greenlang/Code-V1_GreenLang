# GreenLang Golden Test Suite

**Zero-Hallucination Climate Calculation Validation**

This directory contains the golden test suite for GreenLang's climate calculations. Every test has an **expert-validated correct answer** from authoritative sources (DEFRA, EPA, IPCC, Carbon Trust).

## Overview

The golden test framework ensures that climate calculations are:
- **Deterministic**: Same input → Same output (bit-perfect)
- **Accurate**: Match expert-validated answers within ±1% tolerance
- **Auditable**: Complete provenance for every test case
- **Zero-Hallucination**: NO LLM in calculation path

## Test Categories

### 1. Scope 1 Stationary Combustion (10 tests)
- Natural gas boilers
- Diesel generators
- Coal combustion
- LPG heating
- Mixed fuel facilities
- High-precision calculations

**Example:**
```yaml
test_id: "scope1_001"
name: "Natural Gas Boiler - 1000 kWh"
inputs:
  fuel_type: "natural_gas"
  fuel_quantity: 1000
  fuel_unit: "kWh"
  region: "UK"
expected_output: 183.85  # kg CO2e
expert_source: "DEFRA 2024"
```

### 2. Scope 2 Electricity (5 tests)
- UK grid average
- US national average
- Regional factors (California, Texas, etc.)
- Location-based method

**Example:**
```yaml
test_id: "scope2_001"
name: "UK Office Electricity"
inputs:
  electricity_kwh: 10000
  region: "UK"
expected_output: 1930  # kg CO2e (10,000 × 0.193)
expert_source: "DEFRA 2024 - UK Grid = 0.193 kgCO2e/kWh"
```

### 3. Scope 3 Transport & Logistics (5 tests)
- Sea freight (container ships)
- Air freight (international)
- HGV road transport
- Employee commuting
- Business travel

**Example:**
```yaml
test_id: "scope3_001"
name: "Sea Freight - Container Ship"
inputs:
  cargo_tonnes: 10
  distance_km: 5000
expected_output: 565  # kg CO2e
expert_source: "DEFRA 2024 - 0.0113 kgCO2e/tonne-km"
```

### 4. CBAM Embedded Emissions (5 tests)
- Steel production
- Aluminium production
- Cement production
- Concrete
- Mixed material shipments

**Example:**
```yaml
test_id: "cbam_001"
name: "CBAM Steel Import"
inputs:
  material: "steel"
  quantity_kg: 1000
  embedded_factor: 2.1
expected_output: 2100  # kg CO2e
reference_standard: "CBAM (EU) 2023/956"
```

### 5. Edge Cases & Boundary Conditions (5 tests)
- Zero consumption
- Very small values (sub-kilogram)
- Very large values (GWh-scale)
- High-precision calculations (7 decimal places)
- Regulatory rounding tests

## File Structure

```
tests/golden/
├── README.md                        # This file
├── scenarios.yaml                   # 25 expert-validated test cases
├── test_runner_example.py           # Example test runner
├── test_validation_hooks.py         # Validation hook unit tests
└── test_emission_factors.py         # Emission factor database tests
```

## Running Tests

### Run All Golden Tests

```bash
cd tests/golden
python test_runner_example.py
```

Expected output:
```
Loaded 25 golden tests from scenarios.yaml

======================================================================
GOLDEN TEST RESULTS
======================================================================

✓ [scope1_001] Natural Gas Boiler - 1000 kWh
  Expected: 183.850000
  Actual:   183.850000
  Deviation: +0.0000% (within ±1%)

...

======================================================================
SUMMARY
======================================================================
Total tests: 25
Passed: 25 (100.0%)
Failed: 0
Errors: 0
Avg execution time: 2.34ms
Max deviation: 0.0001%
======================================================================
```

### Run Validation Hook Tests

```bash
python test_validation_hooks.py
```

### Run Emission Factor Database Tests

```bash
python test_emission_factors.py
```

## Validation Hooks

The validation framework includes 4 key validators:

### 1. EmissionFactorValidator
Validates emission factors against authoritative ranges:
```python
from greenlang.validation import EmissionFactorValidator

validator = EmissionFactorValidator()
result = validator.validate_factor('diesel', 2.687, 'UK', 'DEFRA')
# Returns: ValidationResult(is_valid=True, ...)
```

### 2. UnitValidator
Validates climate units and conversions:
```python
from greenlang.validation import UnitValidator

validator = UnitValidator()
result = validator.validate_conversion(1000, 'kWh', 'MWh', 1.0)
# Returns: ValidationResult(is_valid=True, ...)
```

### 3. ThermodynamicValidator
Validates physical constraints (efficiency < 100%, etc.):
```python
from greenlang.validation import ThermodynamicValidator

validator = ThermodynamicValidator()
result = validator.validate_efficiency(0.85, 'boiler')
# Returns: ValidationResult(is_valid=True, ...)
```

### 4. GWPValidator
Validates Global Warming Potential values (IPCC AR5/AR6):
```python
from greenlang.validation import GWPValidator

validator = GWPValidator()
result = validator.validate_gwp('CH4', 29.8, 'AR6')
# Returns: ValidationResult(is_valid=True, ...)
```

## Emission Factor Database

Database with 25+ authoritative emission factors:

```python
from greenlang.validation import EmissionFactorDB

db = EmissionFactorDB()

# Get factor for natural gas
factor = db.get_factor('natural_gas', region='UK')
print(factor)
# Output: 0.18385 kgCO2e/kWh (natural_gas, UK, DEFRA_2024)

# Calculate emissions
emissions = 1000 * factor.factor_value  # 183.85 kg CO2e
```

### Data Sources
- **DEFRA 2024**: UK Government greenhouse gas conversion factors
- **EPA eGRID 2023**: US electricity grid emission factors
- **IPCC AR6**: Global climate science standards
- **Ecoinvent 3.9**: Life cycle assessment database

### Coverage
- **Fuels**: Natural gas, diesel, petrol, coal, LPG, heating oil
- **Electricity**: UK, US (national + regional), EU, global
- **Transport**: Cars, HGVs, sea freight, air freight
- **Materials**: Steel, aluminium, concrete, cement

## Test Tolerance

Default tolerance: **±1% relative**

This is appropriate for regulatory climate reporting because:
- Emission factors have inherent uncertainty (typically 5-15%)
- Measurement uncertainty in fuel consumption
- Rounding in regulatory reporting (usually 3 decimal places)
- ±1% is tight enough to catch errors, loose enough to handle uncertainty

For high-precision tests, tolerance can be tightened to ±0.1% or ±0.01%.

## Adding New Tests

1. Add test to `scenarios.yaml`:
```yaml
- test_id: "your_test_id"
  name: "Descriptive Test Name"
  description: "What this test validates"
  category: "scope1|scope2|scope3|cbam|edge_case"
  inputs:
    # Your input parameters
  expected_output: 123.45  # Expert-validated answer
  expected_unit: "kgCO2e"
  tolerance: 0.01  # ±1%
  expert_source: "Source of validation (e.g., DEFRA 2024)"
  reference_standard: "Regulatory standard (e.g., GHG Protocol)"
  tags: ["tag1", "tag2"]
```

2. Ensure your calculation function handles the inputs
3. Run tests: `python test_runner_example.py`

## Quality Assurance

All golden tests have been validated by:
- **Primary sources**: DEFRA, EPA, IPCC published data
- **Manual calculation**: Verified by climate scientists
- **Cross-reference**: Checked against multiple sources
- **Regulatory compliance**: Aligned with GHG Protocol, CBAM, ISO 14064

## Zero-Hallucination Guarantee

The golden test framework guarantees zero hallucination because:
1. **No LLM in calculation path**: All calculations use deterministic math
2. **Expert-validated answers**: Every test has a known-correct answer
3. **Authoritative emission factors**: From DEFRA, EPA, IPCC (not estimated)
4. **Bit-perfect reproducibility**: Same input always produces same output
5. **Complete provenance**: Full audit trail for every calculation

## Certification

This test suite is part of GreenLang's certification process:
- **GL-VALIDATION-001**: Zero-hallucination validation framework
- **GL-GOLDEN-001**: Expert-validated test scenarios
- **GL-EMISSION-001**: Authoritative emission factor database

All tests must pass for production deployment.

## Support

For questions about golden tests:
- Check test scenarios in `scenarios.yaml`
- Review validation hooks in `core/greenlang/validation/`
- See emission factors in `core/greenlang/validation/emission_factors.py`

**Remember**: Every number matters in climate reporting. These tests ensure we get them right, every time.
