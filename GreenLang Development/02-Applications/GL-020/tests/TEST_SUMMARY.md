# GL-020 ECONOPULSE Test Summary

## Agent Information

| Property | Value |
|----------|-------|
| Agent ID | GL-020 |
| Codename | ECONOPULSE |
| Name | EconomizerPerformanceAgent |
| Description | Monitors economizer performance and fouling |
| Test Suite Version | 1.0.0 |
| Target Coverage | 90%+ |

## Test Suite Overview

This comprehensive test suite validates all aspects of the EconomizerPerformanceAgent, including heat transfer calculations, fouling analysis, efficiency monitoring, thermal property lookups, and alert management.

## Test Structure

```
GL-020/
├── tests/
│   ├── __init__.py                           # Test package initialization
│   ├── conftest.py                           # Shared fixtures and utilities
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_heat_transfer_calculator.py  # LMTD, U-value, heat duty tests
│   │   ├── test_fouling_calculator.py        # Fouling factor, cleaning prediction
│   │   ├── test_economizer_efficiency.py     # Effectiveness, heat recovery tests
│   │   ├── test_thermal_properties.py        # Water/gas Cp, IAPWS-IF97 validation
│   │   └── test_alert_manager.py             # Threshold, rate-of-change alerts
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_end_to_end.py                # Full workflow tests
│   ├── test_data/
│   │   ├── sample_readings.json              # Sample sensor readings
│   │   └── sample_economizer_config.json     # Economizer configurations
│   └── requirements-test.txt                 # Test dependencies
└── pytest.ini                                # Pytest configuration
```

## Test Categories

### Unit Tests (Target: 95%+ Coverage)

#### 1. Heat Transfer Calculator (`test_heat_transfer_calculator.py`)

| Test Category | Tests | Coverage Focus |
|---------------|-------|----------------|
| LMTD Calculation | 15+ | Counter-flow, parallel-flow, edge cases |
| Heat Duty Calculation | 10+ | Standard, zero flow, negative values |
| U-Value Calculation | 8+ | Design conditions, degraded |
| Approach Temperature | 5+ | Counter-flow, parallel-flow |
| Effectiveness | 10+ | With/without capacity rates |
| NTU Calculation | 5+ | Various operating conditions |
| ASME PTC 4.3 Validation | 5+ | Reference case validation |

Key validations:
- LMTD accuracy within 2% of ASME PTC 4.3 examples
- Heat duty calculation: Q = m * Cp * dT
- U-value calculation: U = Q / (A * LMTD)

#### 2. Fouling Calculator (`test_fouling_calculator.py`)

| Test Category | Tests | Coverage Focus |
|---------------|-------|----------------|
| Fouling Factor | 12+ | Rf = 1/U_fouled - 1/U_clean |
| Cleanliness Factor | 8+ | CF = U_current / U_clean |
| Fouling Level | 10+ | Threshold-based classification |
| Efficiency Loss | 6+ | Loss vs U-value reduction |
| Fuel Penalty | 6+ | Cost impact calculation |
| Fouling Rate | 5+ | Trend analysis |
| Cleaning Prediction | 8+ | Days to threshold |

Fouling thresholds (m2.K/W):
- Clean: < 0.0002
- Light: 0.0002 - 0.0004
- Moderate: 0.0004 - 0.0007
- Heavy: 0.0007 - 0.001
- Severe: > 0.001

#### 3. Economizer Efficiency (`test_economizer_efficiency.py`)

| Test Category | Tests | Coverage Focus |
|---------------|-------|----------------|
| Effectiveness | 10+ | epsilon = Q_actual / Q_max |
| Heat Recovery Ratio | 6+ | HRR = Q_recovered / Q_available |
| Gas-Side Efficiency | 5+ | Temperature drop ratio |
| Water-Side Efficiency | 5+ | Temperature rise vs design |
| Design Deviation | 5+ | Actual vs design comparison |
| Performance Index | 8+ | Weighted composite metric |

#### 4. Thermal Properties (`test_thermal_properties.py`)

| Test Category | Tests | Coverage Focus |
|---------------|-------|----------------|
| Water Cp | 12+ | Various temperatures/pressures |
| IAPWS-IF97 Validation | 8+ | Reference data comparison |
| Flue Gas Cp | 10+ | Temperature, composition effects |
| Water Density | 5+ | Temperature dependence |
| Flue Gas Density | 5+ | Ideal gas calculations |
| Water Viscosity | 4+ | Temperature dependence |
| Thermal Conductivity | 4+ | Temperature dependence |

IAPWS-IF97 validation points:
- 25C, 101.325 kPa: Cp = 4.1813 kJ/kg.K
- 100C, 200 kPa: Cp = 4.2157 kJ/kg.K
- 150C, 500 kPa: Cp = 4.3100 kJ/kg.K

#### 5. Alert Manager (`test_alert_manager.py`)

| Test Category | Tests | Coverage Focus |
|---------------|-------|----------------|
| Threshold Alerts | 10+ | High/low threshold violation |
| Rate of Change Alerts | 8+ | Rapid change detection |
| Cooldown | 5+ | Prevent alert flooding |
| Deduplication | 5+ | Within-window filtering |
| Prioritization | 5+ | Severity-based sorting |
| Alert Management | 8+ | Acknowledge, resolve, clear |
| Statistics | 4+ | Active counts, history |

### Integration Tests (Target: 85%+ Coverage)

#### End-to-End Tests (`test_end_to_end.py`)

| Test Scenario | Tests | Coverage Focus |
|---------------|-------|----------------|
| Complete Pipeline | 5+ | Sensors to alerts workflow |
| Fouling Detection | 3+ | Clean vs fouled comparison |
| Alert Generation | 4+ | Multi-parameter alerts |
| Cleaning Recommendations | 4+ | Urgency, savings calculation |
| Performance Trending | 3+ | Historical analysis |
| Provenance/Audit | 3+ | Chain integrity |
| Multi-Economizer | 4+ | Parallel monitoring |
| Data Quality | 3+ | Low quality handling |
| Performance | 3+ | Throughput, scaling |

## Test Fixtures

### Economizer Configurations
- `bare_tube_economizer`: Standard bare tube design
- `finned_tube_economizer`: Extended surface with fins
- `extended_surface_economizer`: High-efficiency design
- `condensing_economizer`: Low-temperature heat recovery

### Temperature Readings
- `clean_operation_temperatures`: Normal operating conditions
- `fouled_operation_temperatures`: Degraded heat transfer
- `varying_load_temperatures`: Multiple load levels
- `edge_case_temperatures`: Boundary conditions

### Flow Readings
- `design_flow_readings`: Design flow rates
- `reduced_flow_readings`: Part-load operation
- `zero_flow_readings`: Edge case testing

### Fouling Data
- `clean_fouling_data`: Minimal fouling
- `moderate_fouling_data`: Typical mid-life
- `severe_fouling_data`: Cleaning required
- `fouling_trend_data`: Historical progression

### Alert Configurations
- `fouling_alert_config`: Threshold-based
- `effectiveness_alert_config`: Low efficiency warning
- `rate_of_change_alert_config`: Rapid degradation

## Performance Benchmarks

| Operation | Target | Notes |
|-----------|--------|-------|
| LMTD calculation | < 0.1 ms | Single calculation |
| Heat transfer (full) | < 1 ms | All parameters |
| Fouling calculation | < 0.5 ms | Factor + level |
| Efficiency calculation | < 0.5 ms | All metrics |
| Thermal property lookup | < 0.1 ms | Per property |
| Alert processing | < 0.5 ms | Per event |
| End-to-end processing | < 10 ms | Full snapshot |
| Batch throughput | > 100/s | Full snapshots |

## Coverage Requirements

| Component | Minimum Coverage |
|-----------|-----------------|
| Heat Transfer Calculator | 95% |
| Fouling Calculator | 95% |
| Efficiency Calculator | 95% |
| Thermal Properties | 95% |
| Alert Manager | 95% |
| Integration (End-to-End) | 85% |
| **Overall Target** | **90%+** |

## Running Tests

### Full Test Suite
```bash
pytest
```

### With Coverage
```bash
pytest --cov=. --cov-report=html --cov-report=term-missing
```

### Specific Markers
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Performance tests
pytest -m performance

# ASME validation tests
pytest -m asme

# IAPWS validation tests
pytest -m iapws

# Critical tests only
pytest -m critical
```

### Parallel Execution
```bash
pytest -n auto
```

### Exclude Slow Tests
```bash
pytest -m "not slow"
```

## Validation Standards

### ASME PTC 4.3 Compliance
- LMTD calculations validated against standard examples
- Heat duty calculations within 2% tolerance
- Approach temperature definitions per standard

### IAPWS-IF97 Compliance
- Water Cp validated at reference conditions
- Density calculations within 0.2% of reference
- Property correlations for 0-374C range

## Test Output

### JUnit XML (CI/CD)
```bash
pytest --junitxml=test-results.xml
```

### HTML Report
```bash
pytest --html=report.html --self-contained-html
```

### Coverage Badge
```bash
coverage-badge -o coverage.svg
```

## Author

**GL-TestEngineer**

Version: 1.0.0
Last Updated: 2025-01-01
