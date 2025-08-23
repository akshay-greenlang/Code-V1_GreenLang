# GreenLang Examples

This directory contains canonical example tests that demonstrate key GreenLang concepts. Each example is designed to teach one specific idea clearly, with stable outputs and comprehensive comments.

## Overview

We provide 18 core examples (≤60 lines each) that are:
- **Didactic**: Each teaches a specific concept
- **Dataset-driven**: Use real emission factors, no magic numbers
- **Gracefully skippable**: Skip if optional surfaces aren't available
- **Deterministic**: Produce stable, reproducible results

## Running the Examples

### Run all example tests
```bash
pytest -m example
```

### Run specific example
```bash
pytest examples/tests/ex_01_fuel_agent_basic.py -v
```

### Run with minimal output
```bash
pytest -m example -q
```

## Example Categories

### Core Agent Examples (1-6)
- **ex_01**: FuelAgent basic electricity calculation (dataset-driven)
- **ex_02**: FuelAgent error path (negative input validation)
- **ex_03**: GridFactorAgent factor retrieval with provenance
- **ex_04**: InputValidatorAgent unit normalization (sqm→sqft, MWh→kWh)
- **ex_05**: CarbonAgent aggregation invariants (additivity & percentages)
- **ex_06**: IntensityAgent formulas with zero guard

### Advanced Agent Examples (7-9)
- **ex_07**: BenchmarkAgent threshold boundaries
- **ex_08**: RecommendationAgent deterministic mapping
- **ex_09**: ReportAgent JSON and Markdown generation

### Workflow Examples (10)
- **ex_10**: Minimal YAML workflow with orchestrator

### Cross-System Examples (11-14)
- **ex_11**: Cross-country emission factor comparison
- **ex_12**: CLI run to JSON export
- **ex_13**: CLI calc to Markdown snapshot
- **ex_14**: Small portfolio processing via SDK

### Advanced Features (15-18)
- **ex_15**: Plugin discovery contract
- **ex_16**: Assistant path mocked for determinism
- **ex_17**: Dataset provenance propagation
- **ex_18**: Environment variable overrides

## Directory Structure

```
examples/
├── README.md                    # This file
├── tests/                       # Example test files
│   ├── conftest.py             # Pytest configuration
│   ├── ex_01_fuel_agent_basic.py
│   ├── ex_02_fuel_agent_error.py
│   └── ... (18 total examples)
├── fixtures/                    # Test data files
│   ├── building_india_office.json
│   ├── building_us_office.json
│   ├── portfolio_small.json
│   └── workflow_minimal.yaml
├── utils/                       # Helper utilities
│   ├── dataset.py              # Dataset loading utilities
│   └── normalizers.py          # Text normalization for snapshots
└── snapshots/                   # Snapshot test outputs
    └── reports/
        └── ex_13_cli_calc_markdown.out

```

## Key Concepts Demonstrated

### Data-Driven Testing
All examples use real emission factors from the global dataset rather than hard-coded values:
```python
factor = load_emission_factor(country="IN", fuel="electricity", unit="kWh")
assert math.isclose(result, kwh * factor, rel_tol=1e-9)
```

### Graceful Skipping
Examples skip when dependencies aren't available:
```python
try:
    from greenlang.agents.fuel_agent import FuelAgent
except Exception:
    FuelAgent = None

if FuelAgent is None:
    pytest.skip("FuelAgent not importable")
```

### Deterministic Testing
Network access is blocked and timestamps are normalized:
```python
@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Deterministic examples: no network."""
    def guard(*args, **kwargs):
        raise RuntimeError("Network access disabled")
    monkeypatch.setattr(socket, "create_connection", guard)
```

### Invariant Testing
Mathematical properties that must hold:
```python
# Sum of parts equals total
assert math.isclose(total, sum(by_fuel.values()), rel_tol=1e-9)

# Percentages sum to 100
assert math.isclose(sum(percentages.values()), 100.0, rel_tol=1e-6)
```

## Writing New Examples

When adding new examples, follow these guidelines:

1. **One concept per test**: Each example should teach exactly one thing
2. **Use fixtures**: Load test data from `fixtures/` directory
3. **Dataset-driven**: Use `load_emission_factor()` for factors
4. **Normalize outputs**: Use `normalize_text()` for stable snapshots
5. **Document clearly**: Include docstring explaining what's being tested
6. **Keep it short**: Target ≤60 lines per example
7. **Handle missing deps**: Skip gracefully if imports fail

## Advanced Examples (19-30)

### Testing Properties & Patterns (19-27)
- **ex_19**: Property-based testing (additivity, scaling, unit round-trip)
- **ex_20**: Parallel workflow determinism
- **ex_21**: Caching performance (second run faster, same bytes)
- **ex_22**: Backward compatibility for old workflow YAML
- **ex_23**: JSON Schema validation for outputs
- **ex_24**: Doctest/xdoctest examples in documentation
- **ex_25**: Concurrency testing (N parallel runs → identical results)
- **ex_26**: Long-run smoke test (100 executions, no memory blow-up)
- **ex_27**: Windows line-ending and path normalization

### Tutorials (28-30)
- **ex_28**: Custom agent tutorial - Write a new agent in 30 lines
- **ex_29**: Country factors tutorial - Add a new country factor set
- **ex_30**: Workflow XLSX tutorial - Build custom workflow and export to Excel

## Troubleshooting

### Import Errors
If examples can't import GreenLang modules:
```bash
# Install in development mode
pip install -e .
```

### Dataset Not Found
Update `CANDIDATE_PATHS` in `utils/dataset.py` if your dataset location differs.

### Network Access Blocked
This is intentional - examples should be deterministic. Use mocks for external services.

## Contributing

When contributing examples:
1. Follow the existing pattern
2. Add to this README
3. Ensure deterministic output
4. Test on multiple platforms
5. Document the teaching objective