# Golden Test Suite for GL-FOUND-X-003

## Overview

This is the comprehensive Golden Test Suite for the GreenLang Unit & Reference Normalizer (GL-FOUND-X-003). The test suite validates:

- **Unit Conversions**: All GL Canonical Unit conversions across 6 dimensions
- **Entity Resolution**: Fuel, material, and process matching
- **Full Pipeline**: End-to-end GHG Protocol and EU CSRD/CBAM scenarios

## Test Statistics

| Category | Test Files | Test Cases | Dimensions/Types |
|----------|------------|------------|------------------|
| Unit Conversions | 6 YAML files | 130+ cases | Energy, Mass, Volume, Emissions, Pressure, Temperature |
| Entity Resolution | 3 YAML files | 63+ cases | Fuels, Materials, Processes |
| Full Pipeline | 2 YAML files | 27+ cases | GHG Protocol, EU CSRD/CBAM |

**Total: 220+ golden test cases**

## Directory Structure

```
tests/golden/
|-- __init__.py                 # Package initialization
|-- conftest.py                 # Pytest fixtures and configuration
|-- pytest.ini                  # Pytest configuration
|-- README.md                   # This file
|-- test_unit_conversion.py     # Unit conversion tests
|-- test_entity_resolution.py   # Entity resolution tests
|-- test_full_pipeline.py       # Full pipeline tests
|
|-- golden_files/
    |-- unit_conversions/
    |   |-- energy.yaml         # Energy unit conversions (25 cases)
    |   |-- mass.yaml           # Mass unit conversions (22 cases)
    |   |-- volume.yaml         # Volume unit conversions (24 cases)
    |   |-- emissions.yaml      # Emissions conversions (22 cases)
    |   |-- pressure.yaml       # Pressure conversions (20 cases)
    |   |-- temperature.yaml    # Temperature conversions (20 cases)
    |
    |-- entity_resolution/
    |   |-- fuels.yaml          # Fuel entity resolution (25 cases)
    |   |-- materials.yaml      # Material entity resolution (20 cases)
    |   |-- processes.yaml      # Process entity resolution (18 cases)
    |
    |-- full_pipeline/
        |-- ghg_protocol_scenarios.yaml   # GHG Protocol scenarios (15 cases)
        |-- eu_csrd_scenarios.yaml        # EU CSRD/CBAM scenarios (12 cases)
```

## Running Tests

### Run All Tests
```bash
cd greenlang-normalizer/tests/golden
pytest
```

### Run Specific Test Categories
```bash
# Unit conversion tests only
pytest test_unit_conversion.py -v

# Entity resolution tests only
pytest test_entity_resolution.py -v

# Full pipeline tests only
pytest test_full_pipeline.py -v
```

### Run Tests by Marker
```bash
# Compliance tests only
pytest -m compliance

# Skip slow tests
pytest -m "not slow"

# Pint cross-validation tests
pytest -m pint_cross_validation
```

### Run Tests by Pattern
```bash
# Run all energy tests
pytest -k "energy"

# Run all CBAM tests
pytest -k "cbam"

# Run exact match tests
pytest -k "exact"
```

## Test Features

### Tolerance-based Comparison
All numeric comparisons use tolerance-based matching:
- **Exact conversions**: Tolerance of 1e-12
- **Non-exact conversions**: Relative tolerance specified in test case
- **Default**: 1e-9 relative tolerance

### Cross-validation with Pint
Tests marked with `@pytest.mark.pint_cross_validation` compare results against the Pint library.

### Clear Failure Messages
Failed tests provide detailed diff information:
```
Expected: 360.0
Actual:   360.00001
Relative difference: 2.78e-08 (max: 1e-09)
```

### Test Tags
Each test case has tags for filtering:
- `exact`, `alias`, `rule`, `fuzzy` - Match method
- `si`, `imperial` - Unit system
- `common`, `edge_case` - Use case
- `cbam`, `ghg_protocol`, `csrd` - Regulatory framework

## Golden File Format

### Unit Conversion Test Case
```yaml
- name: "kWh to MJ conversion"
  description: "Convert kilowatt-hours to megajoules (1 kWh = 3.6 MJ exactly)"
  input:
    value: 100
    unit: "kWh"
    target_unit: "MJ"
  expected:
    canonical_value: 360.0
    canonical_unit: "MJ"
    confidence: 1.0
    exact: true
  tags: ["electrical", "common"]
```

### Entity Resolution Test Case
```yaml
- name: "Alias match - Nat Gas"
  description: "Common abbreviation for natural gas"
  input:
    raw_name: "Nat Gas"
    entity_type: "fuel"
  expected:
    reference_id: "GL-FUEL-NATGAS"
    canonical_name: "Natural gas"
    match_method: "alias"
    confidence: 1.0
    needs_review: false
  tags: ["alias", "common"]
```

### Full Pipeline Test Case
```yaml
- name: "Scope 1 - Stationary combustion (natural gas)"
  description: "Calculate emissions from natural gas combustion in a boiler"
  input:
    source_record_id: "ghg-sc1-001"
    policy_mode: "STRICT"
    measurements:
      - field: "fuel_consumption"
        value: 1500
        unit: "MMBtu"
        expected_dimension: "energy"
    entities:
      - field: "fuel_type"
        entity_type: "fuel"
        raw_name: "Natural Gas"
  expected:
    status: "success"
    canonical_measurements:
      - field: "fuel_consumption"
        canonical_value: 1582584.0
        canonical_unit: "MJ"
        tolerance: 10.0
    normalized_entities:
      - field: "fuel_type"
        reference_id: "GL-FUEL-NATGAS"
        confidence: 1.0
```

## Adding New Tests

### 1. Add to Existing Golden File
Edit the appropriate YAML file in `golden_files/` and add a new test case.

### 2. Create New Golden File
1. Create a new YAML file in the appropriate directory
2. Follow the schema format shown above
3. Tests will be automatically discovered

### 3. Add Programmatic Tests
Add new test methods to the existing test classes or create new test classes.

## Compliance Testing

### GHG Protocol Coverage
- Scope 1: Direct emissions (stationary, mobile, fugitive, process)
- Scope 2: Indirect emissions (location-based, market-based)
- Scope 3: Value chain emissions (Categories 1, 4, 6)

### EU CSRD Coverage
- E1: Climate change metrics
- E2: Pollution metrics
- E5: Circular economy metrics

### EU CBAM Coverage
- Iron and Steel
- Aluminum
- Cement
- Fertilizers
- Hydrogen
- Electricity

## Quality Gates

Before release, all tests must pass:
- Zero critical conversion errors
- No regression in parse success rate
- All entity resolution tests passing
- All compliance tests passing

## Maintenance

### Updating Conversion Factors
1. Update `config/canonical_units.yaml`
2. Update golden file expected values
3. Run tests to verify

### Adding New Vocabulary Entries
1. Update vocabulary configuration
2. Add golden tests for new entries
3. Run entity resolution tests

### Version Compatibility
Golden tests are versioned with the normalizer. When vocabulary or unit registry versions change, update:
- `vocabulary_version` in entity resolution files
- Expected values based on new factors
