# 🧪 GreenLang Testing Guide

## Test Suite Summary

**Status: ✅ PRODUCTION READY**
- **200+ tests** with ≥85% coverage enforced
- **Data-driven testing** using actual emission factors
- **No hardcoded values** - single source of truth
- **Fast execution** - complete suite runs in <90 seconds
- **Deterministic** - no network calls, reproducible results

## Overview

GreenLang includes a production-quality test suite with **200+ tests** ensuring reliability, accuracy, and maintainability. The test suite follows industry best practices with data-driven testing, property-based testing, comprehensive boundary testing, and enforced quality gates.

## Key Principles

### 1. Data-Driven Testing
- **No hardcoded values**: All expected emission factors are loaded from actual data files
- **Single source of truth**: Tests use the same `greenlang/data/global_emission_factors.json` that production code uses
- **Factor drift prevention**: Tests automatically stay in sync with data updates
- **Consistent factors**: All documentation and tests use the same grid factors:
  - US: 0.385 kgCO2e/kWh
  - EU: 0.23 kgCO2e/kWh  
  - IN: 0.71 kgCO2e/kWh
  - CN: 0.65 kgCO2e/kWh
  - JP: 0.45 kgCO2e/kWh
  - BR: 0.12 kgCO2e/kWh
  - (See global_emission_factors.json for complete list)

### 2. Deterministic Testing
- **No network calls**: All external APIs are mocked
- **No LLM dependencies**: AI responses are stubbed
- **Reproducible results**: Tests produce same results every run
- **Fast execution**: Complete suite runs in <90 seconds

### 3. Comprehensive Coverage
- **≥85% overall coverage** (enforced)
- **≥90% agent coverage** (enforced)
- **All code paths tested**: Including error cases and edge conditions
- **Contract validation**: All agents follow consistent input/output format

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests (150+ tests)
│   ├── agents/                 # Agent-specific tests
│   │   ├── test_fuel_agent.py
│   │   ├── test_grid_factor_agent.py
│   │   ├── test_input_validator_agent.py
│   │   ├── test_carbon_agent.py
│   │   ├── test_intensity_agent.py
│   │   ├── test_benchmark_agent.py
│   │   ├── test_benchmark_agent_boundaries.py
│   │   ├── test_building_profile_agent.py
│   │   ├── test_recommendation_agent.py
│   │   └── test_report_agent.py
│   ├── core/                   # Framework tests
│   ├── cli/                    # CLI tests (main.py is the single entrypoint)
│   ├── sdk/                    # SDK tests
│   └── data/                   # Data validation tests
├── property/                   # Property-based tests (15+ tests)
│   └── test_input_validator_properties.py
├── integration/                # Integration tests
├── fixtures/                   # Test data files
└── test_end_to_end.py        # End-to-end workflow tests
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=greenlang --cov-report=html

# Run specific test file
pytest tests/unit/agents/test_fuel_agent.py

# Run specific test
pytest tests/unit/agents/test_fuel_agent.py::TestFuelAgent::test_electricity_india_kwh

# Run by marker
pytest -m unit          # Unit tests only
pytest -m property      # Property tests only
pytest -m integration   # Integration tests only

# Run in parallel (faster)
pytest -n auto

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Quality Checks

```bash
# Linting
ruff check greenlang/ tests/

# Type checking
mypy greenlang/ --strict

# Code formatting
black --check greenlang/ tests/

# All quality checks
make quality  # If Makefile is available
```

## Test Categories

### 1. Unit Tests (tests/unit/)

#### Agent Tests
Each agent has comprehensive tests covering:
- **Happy path**: Normal operation with valid inputs
- **Error cases**: Invalid inputs, missing data
- **Edge cases**: Boundary values, empty inputs
- **Contract validation**: Consistent I/O format

Example from `test_fuel_agent.py`:
```python
def test_electricity_india_kwh(self, electricity_factors, agent_contract_validator):
    # Data-driven: reads factor from fixture loaded from global_emission_factors.json
    # This ensures tests always use the same factors as production (0.71 for India)
    expected_factor = electricity_factors.get("IN", {}).get("factor")
    consumption = 1_500_000
    expected_emissions = consumption * expected_factor
    
    result = self.agent.execute({
        "fuel_type": "electricity",
        "consumption": consumption,
        "unit": "kWh",
        "region": "IN"
    })
    
    # Contract validation
    agent_contract_validator.validate(result)
    
    # Accuracy validation
    assert abs(result["data"]["co2e_tons"] - expected_emissions / 1000) < 0.01
```

### 2. Boundary Tests (test_benchmark_agent_boundaries.py)

Comprehensive testing of all rating thresholds:
```python
@pytest.mark.parametrize("building_type,country,test_cases", [
    ("office", "IN", [
        (9.99, "A"),   # Just below A/B boundary
        (10.0, "B"),   # Exactly at boundary (inclusive)
        (10.01, "B"),  # Just above boundary
        (14.99, "B"),  # Just below B/C boundary
        (15.0, "C"),   # Exactly at boundary
        # ... all boundaries tested
    ])
])
def test_rating_boundaries(self, building_type, country, test_cases):
    # Table-driven boundary testing
```

### 3. Property-Based Tests (tests/property/)

Using Hypothesis to verify mathematical invariants:
```python
@given(
    value=st.floats(min_value=0, max_value=1e6),
    from_unit=st.sampled_from(["kWh", "MWh", "GWh"]),
    to_unit=st.sampled_from(["kWh", "MWh", "GWh"])
)
def test_unit_conversion_round_trip(value, from_unit, to_unit):
    # Property: converting there and back gives original value
    converted = convert(value, from_unit, to_unit)
    back = convert(converted, to_unit, from_unit)
    assert abs(back - value) < 1e-9
```

### 4. End-to-End Tests (test_end_to_end.py)

Complete workflow validation:
```python
def test_single_building_minimal_workflow(self):
    building = {
        "metadata": {...},
        "energy_consumption": {...}
    }
    
    # Execute full workflow
    workflow = Workflow("commercial_building_emissions")
    result = workflow.execute(building)
    
    # Validate complete chain
    assert "emissions" in result
    assert "intensity" in result
    assert "benchmark" in result
    assert "recommendations" in result
```

## Test Fixtures (conftest.py)

### Hypothesis Configuration

The test suite includes Hypothesis profiles to ensure tests complete within 90 seconds:

```python
# Configure Hypothesis for fast, deterministic tests
settings.register_profile(
    "fast",
    max_examples=5,  # Minimal for quick checks
    deadline=1000,  # 1 second deadline per test
    derandomize=True,  # Deterministic test order
)

# CI profile for automated testing
settings.register_profile(
    "ci",
    max_examples=10,  # Reduced for CI speed
    deadline=5000,  # 5 seconds deadline
    derandomize=True,
)

# Load profile from environment or default to 'fast'
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "fast"))
```

## Test Fixtures (conftest.py)

### Key Fixtures

```python
@pytest.fixture(scope="session")
def emission_factors(data_dir):
    """Load actual emission factors from data file."""
    with open(data_dir / "global_emission_factors.json") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def electricity_factors(emission_factors):
    """Extract electricity-specific factors."""
    return emission_factors.get("electricity", {})

@pytest.fixture(scope="session")
def benchmarks_data(data_dir):
    """Load benchmark thresholds from data file."""
    with open(data_dir / "global_benchmarks.json") as f:
        return json.load(f)

@pytest.fixture
def agent_contract_validator():
    """Validator for agent output contract."""
    return AgentContractValidator()
```

### Automatic Network Disabling

```python
@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch):
    """Automatically disable all network calls."""
    def mock_network_call(*args, **kwargs):
        raise RuntimeError("Network calls are disabled in tests")
    
    # Disable socket connections entirely
    def guard(*args, **kwargs):
        raise RuntimeError("Socket connections are disabled in tests")
    monkeypatch.setattr(socket, "socket", guard)
    
    # Disable httpx, requests, OpenAI, LangChain
    monkeypatch.setattr("httpx.Client.request", mock_network_call)
    monkeypatch.setattr("requests.Session.request", mock_network_call)
    monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: None)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda *args, **kwargs: None)
```

## CI/CD Integration

### GitHub Actions Workflow (.github/workflows/test.yml)

The CI pipeline enforces all quality gates:

```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - name: Run linting (ruff)
      run: ruff check greenlang/ tests/
      continue-on-error: false  # Enforced
    
    - name: Run type checking (mypy)
      run: mypy greenlang/ --strict
      continue-on-error: false  # Enforced
    
    - name: Run tests with coverage
      run: pytest --cov=greenlang --cov-fail-under=85
    
    - name: Check coverage thresholds
      # Custom script to verify ≥85% overall, ≥90% agents
```

### Quality Gates

| Gate | Threshold | Enforcement |
|------|-----------|-------------|
| Overall Coverage | ≥85% | CI fails if below |
| Agent Coverage | ≥90% | CI fails if below |
| Linting (ruff) | No errors | CI fails on errors |
| Type Checking (mypy) | Strict mode | CI fails on errors |
| Code Format (black) | Consistent | CI fails if not formatted |
| Test Performance | <90 seconds | CI fails if slower |
| Python Versions | 3.9-3.12 | All must pass |
| Operating Systems | Linux/macOS/Windows | All must pass |

## Writing New Tests

### Guidelines

1. **Use fixtures for data**: Never hardcode emission factors or benchmarks
2. **Test the contract**: Use `agent_contract_validator` for all agent tests
3. **Cover boundaries**: Test at, above, and below threshold values
4. **Test errors**: Verify proper error messages for invalid inputs
5. **Keep tests fast**: Mock expensive operations
6. **Use descriptive names**: `test_electricity_india_kwh` not `test_1`

### Example Test Template

```python
class TestNewAgent:
    @pytest.fixture
    def agent(self):
        return NewAgent()
    
    def test_happy_path(self, agent, emission_factors, agent_contract_validator):
        """Test normal operation with valid inputs."""
        # Arrange
        input_data = {"key": "value"}
        expected_factor = emission_factors.get("type", {}).get("factor")
        
        # Act
        result = agent.execute(input_data)
        
        # Assert
        agent_contract_validator.validate(result)
        assert result["success"] is True
        assert abs(result["data"]["value"] - expected_factor) < 0.01
    
    def test_invalid_input(self, agent):
        """Test error handling for invalid inputs."""
        result = agent.execute({"invalid": "data"})
        
        assert result["success"] is False
        assert "error" in result
        assert "Invalid input" in result["error"]
    
    def test_boundary_case(self, agent):
        """Test behavior at boundary values."""
        # Test minimum value
        result = agent.execute({"value": 0})
        assert result["success"] is True
        
        # Test maximum value
        result = agent.execute({"value": 1e9})
        assert result["success"] is True
```

## Debugging Failed Tests

### Common Issues and Solutions

1. **Factor Drift**
   - Problem: Test expects 0.71 but gets 0.708
   - Solution: Use fixtures that load from data files
   ```python
   # Bad: Hardcoded
   expected = 0.71
   
   # Good: Data-driven
   expected = electricity_factors.get("IN", {}).get("factor")
   ```

2. **Network Calls**
   - Problem: Test makes actual API calls
   - Solution: Mock or use disable_network_calls fixture
   ```python
   def test_api_call(self, monkeypatch):
       monkeypatch.setattr("httpx.get", lambda *args: {"data": "mocked"})
   ```

3. **Non-deterministic Results**
   - Problem: Test passes sometimes, fails others
   - Solution: Control randomness, mock timestamps
   ```python
   @pytest.fixture
   def fixed_timestamp(monkeypatch):
       monkeypatch.setattr("time.time", lambda: 1234567890)
   ```

4. **Coverage Gaps**
   - Problem: Coverage below threshold
   - Solution: Add tests for uncovered lines
   ```bash
   # Find uncovered lines
   pytest --cov=greenlang --cov-report=html
   # Open htmlcov/index.html in browser
   ```

## Performance Testing

### Benchmarking

```python
@pytest.mark.benchmark
def test_performance(benchmark):
    result = benchmark(expensive_function, arg1, arg2)
    assert result == expected
```

### Load Testing

```python
@pytest.mark.slow
def test_handles_large_portfolio():
    buildings = [create_building() for _ in range(1000)]
    start = time.time()
    result = analyze_portfolio(buildings)
    elapsed = time.time() - start
    
    assert elapsed < 10  # Should complete in 10 seconds
    assert len(result) == 1000
```

## Test Maintenance

### Regular Tasks

1. **Update test data**: When emission factors change
2. **Add new tests**: For new features or bug fixes
3. **Refactor tests**: Keep them clean and maintainable
4. **Review coverage**: Identify and fill gaps
5. **Update documentation**: Keep this guide current

### Best Practices

1. **One assertion per test**: Makes failures clear
2. **Arrange-Act-Assert**: Standard test structure
3. **Use fixtures**: Share setup between tests
4. **Test behavior, not implementation**: Focus on outputs
5. **Keep tests independent**: No shared state
6. **Fast tests first**: Run quick tests before slow ones

## Troubleshooting

### pytest Won't Run

```bash
# Install test dependencies
pip install -r requirements.txt

# Verify pytest is installed
pytest --version
```

### Tests Pass Locally but Fail in CI

1. Check Python version differences
2. Check OS-specific behavior
3. Verify all dependencies are in requirements.txt
4. Check for hardcoded paths

### Coverage Report Issues

```bash
# Generate detailed report
pytest --cov=greenlang --cov-report=term-missing

# Generate HTML report for investigation
pytest --cov=greenlang --cov-report=html
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [GitHub Actions testing](https://docs.github.com/en/actions/automating-builds-and-tests)

---

**Remember**: Good tests are the foundation of reliable software. Invest time in writing comprehensive, maintainable tests, and they will save you countless hours of debugging in the future.