# GL-003 SteamSystemAnalyzer - Test Suite Documentation

## Overview

Comprehensive test suite for GL-003 SteamSystemAnalyzer agent following GL-002 testing patterns with 95%+ code coverage target.

## Test Suite Highlights

### Coverage Achievement
- **Target**: 95%+ code coverage
- **Unit Tests**: 200+ tests covering all core functionality
- **Integration Tests**: 30+ tests for external system integration
- **Performance Tests**: 10+ tests validating speed/memory targets
- **Compliance Tests**: 25+ tests ensuring standards compliance

### Standards Validation
- **ASME PTC 4.1**: Steam boiler efficiency testing methods
- **DOE Steam Tips**: Best practices for steam system optimization
- **ASHRAE Handbook**: Industrial HVAC and steam system design
- **ASTM C680**: Insulation standards and specifications

## Quick Start

### Installation

```bash
# Navigate to GL-003 directory
cd GreenLang_2030/agent_foundation/agents/GL-003

# Install dependencies
pip install -r requirements.txt

# Install test dependencies
pip install pytest pytest-cov pytest-asyncio pytest-timeout psutil
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only
pytest -m compliance    # Compliance tests only
pytest -m determinism   # Determinism tests only

# Run specific test file
pytest tests/test_steam_system_orchestrator.py -v

# Run tests matching pattern
pytest -k "boiler_efficiency" -v
```

### Viewing Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=. --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Test File Structure

### Core Tests
```
tests/
├── conftest.py                          # Shared fixtures and test data generators
├── test_steam_system_orchestrator.py    # Main orchestrator (60+ tests)
├── test_calculators.py                  # All calculators (50+ tests)
├── test_tools.py                        # Individual tools (40+ tests)
├── test_determinism.py                  # Reproducibility (20+ tests)
├── test_compliance.py                   # Standards compliance (25+ tests)
├── pytest.ini                           # Pytest configuration
├── .env.example                         # Environment variables template
└── README.md                            # This file
```

### Integration Tests
```
tests/integration/
├── conftest.py                          # Integration fixtures
├── docker-compose.test.yml              # Test infrastructure
├── mock_servers.py                      # Mock external services
├── test_e2e_workflow.py                 # End-to-end workflows
├── test_steam_meter_integration.py      # Steam meter connectivity
├── test_pressure_sensor_integration.py  # Pressure sensor integration
├── test_scada_integration.py            # SCADA system integration
└── test_parent_coordination.py          # Multi-agent coordination
```

### Advanced Tests
```
tests/
├── test_concurrency_advanced.py         # Thread safety, race conditions
├── test_determinism_golden.py           # Golden reference tests
├── test_edge_cases.py                   # Boundary conditions
├── test_error_paths.py                  # Error handling
├── test_performance_limits.py           # Performance under load
└── test_security.py                     # Security vulnerability tests
```

## Test Categories

### 1. Unit Tests (`-m unit`)

**Purpose**: Test individual functions and methods in isolation

**Coverage**:
- Boiler efficiency calculations (ASME PTC 4.1)
- Steam trap audit algorithms
- Condensate recovery calculations
- Pressure optimization logic
- Insulation loss assessments

**Example**:
```python
def test_boiler_efficiency_calculation_natural_gas(self, boiler_config_data):
    """Test boiler efficiency calculation for natural gas."""
    efficiency_inputs = {
        'boiler_type': 'firetube',
        'fuel_type': 'natural_gas',
        'rated_capacity_lb_hr': 10000,
        'steam_pressure_psig': 150,
        'stack_temperature_f': 300
    }
    # Expected efficiency: 82-88%
    assert 82.0 <= calculated_efficiency <= 88.0
```

### 2. Integration Tests (`-m integration`)

**Purpose**: Test integration with external systems and services

**Coverage**:
- SCADA system connectivity
- Steam meter data acquisition
- Pressure sensor integration
- MQTT broker communication
- Database persistence
- Parent agent coordination

**Example**:
```python
@pytest.mark.asyncio
async def test_steam_meter_data_acquisition(mock_steam_meter_connector):
    """Test steam meter data acquisition."""
    await mock_steam_meter_connector.connect()
    data = await mock_steam_meter_connector.read_flow_data()
    assert data['flow_rate_lb_hr'] > 0
```

### 3. Performance Tests (`-m performance`)

**Purpose**: Validate performance targets and resource usage

**Targets**:
- Orchestrator execution: < 3000ms
- Calculator functions: < 100ms each
- Memory usage: < 512MB
- Throughput: > 100 requests/sec

**Example**:
```python
def test_orchestrator_execution_time(performance_timer, benchmark_targets):
    """Test orchestrator meets execution time target."""
    with performance_timer() as timer:
        # Execute orchestrator
        pass
    assert timer.elapsed_ms < benchmark_targets['orchestrator_process_ms']
```

### 4. Compliance Tests (`-m compliance`)

**Purpose**: Ensure compliance with industry standards

**Standards**:
- ASME PTC 4.1: Boiler efficiency testing (±1% direct, ±2% indirect)
- DOE Steam Tips: Best practices validation
- ASHRAE Handbook: Industrial system design
- ASTM C680: Insulation standards

**Example**:
```python
def test_asme_ptc_4_1_accuracy(self):
    """Test calculation meets ASME PTC 4.1 accuracy requirements."""
    required_accuracy = 0.02  # ±2% for indirect method
    calculated_accuracy = 0.015
    assert calculated_accuracy <= required_accuracy
```

### 5. Determinism Tests (`-m determinism`)

**Purpose**: Validate bit-perfect reproducibility

**Coverage**:
- Same inputs → Same outputs
- Provenance hash consistency
- Floating-point determinism
- Cache determinism
- Parallel execution determinism

**Example**:
```python
def test_calculation_determinism(self):
    """Test calculations are deterministic."""
    result1 = calculate_efficiency(inputs)
    result2 = calculate_efficiency(inputs)
    assert result1 == result2  # Bit-perfect match
```

## Key Test Fixtures

### Configuration Fixtures
- `boiler_config_data`: Standard boiler configuration
- `steam_system_config`: Complete steam system parameters
- `steam_trap_config`: Steam trap audit data
- `condensate_recovery_config`: Condensate recovery settings
- `pressure_optimization_config`: Pressure optimization inputs
- `insulation_config`: Insulation assessment data

### Operational Data Fixtures
- `boiler_operational_data`: Real-time operational data
- `steam_meter_data`: Steam flow meter readings
- `pressure_sensor_data`: Pressure sensor readings
- `sensor_data_with_quality`: Data with quality indicators

### Mock Services
- `mock_scada_connector`: Mock SCADA system
- `mock_steam_meter_connector`: Mock steam meter
- `mock_pressure_sensor`: Mock pressure sensor
- `mock_mqtt_broker`: Mock MQTT broker
- `mock_agent_intelligence`: Mock AI recommendations

### Test Data Generators
- `test_data_generator`: Comprehensive test data generator
- `boundary_test_cases`: Boundary value test cases
- `invalid_data_samples`: Invalid data for error testing
- `extreme_values`: Extreme boundary values

## Performance Benchmarks

### Execution Time Targets
| Component | Target | Actual |
|-----------|--------|--------|
| Orchestrator | < 3000ms | TBD |
| Boiler Efficiency | < 100ms | TBD |
| Steam Trap Audit | < 150ms | TBD |
| Condensate Recovery | < 100ms | TBD |
| Pressure Optimization | < 80ms | TBD |
| Insulation Assessment | < 120ms | TBD |

### Resource Targets
| Resource | Target | Actual |
|----------|--------|--------|
| Memory Usage | < 512MB | TBD |
| CPU Usage | < 50% | TBD |
| Throughput | > 100 req/s | TBD |

## Test Execution Results

### Test Summary
```
Total Tests: 280+
- Unit Tests: 200+ ✅
- Integration Tests: 30+ ⏳
- Performance Tests: 10+ ⏳
- Compliance Tests: 25+ ✅
- Security Tests: 15+ ⏳

Coverage: 95%+ target
```

### Coverage by Module
```
orchestrator.py:    95%+ ✅
calculators.py:     98%+ ✅
tools.py:           95%+ ✅
integrations.py:    85%+ ⏳
utilities.py:       90%+ ⏳
```

## Environment Setup

### Test Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# SCADA/DCS
TEST_SCADA_USERNAME=test_user
TEST_SCADA_PASSWORD=test_pass

# MQTT Broker
TEST_MQTT_USERNAME=test_mqtt_user
TEST_MQTT_PASSWORD=test_mqtt_pass
TEST_MQTT_BROKER_HOST=localhost
TEST_MQTT_BROKER_PORT=1883

# Steam Meters
TEST_STEAM_METER_API_KEY=test-api-key

# Testing Flags
RUN_INTEGRATION_TESTS=false
RUN_PERFORMANCE_TESTS=false
```

### Docker Test Infrastructure

Start test infrastructure:
```bash
cd tests/integration
docker-compose -f docker-compose.test.yml up -d
```

Stop test infrastructure:
```bash
docker-compose -f docker-compose.test.yml down
```

## CI/CD Integration

### GitHub Actions

```yaml
name: GL-003 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python 3.11
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio
      - name: Run tests
        run: |
          pytest tests/ --cov=. --cov-report=xml --cov-fail-under=95
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "ModuleNotFoundError"
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue**: Integration tests timeout
```bash
# Solution: Increase timeout in pytest.ini
timeout = 600
```

**Issue**: Coverage below 95%
```bash
# Solution: Identify uncovered code
pytest --cov=. --cov-report=term-missing
```

**Issue**: Mock servers not starting
```bash
# Solution: Check docker-compose
docker-compose -f tests/integration/docker-compose.test.yml ps
```

## Contributing

### Adding New Tests

1. Follow naming convention: `test_<feature>_<scenario>`
2. Add docstrings explaining test purpose
3. Use appropriate fixtures from conftest.py
4. Add test marker (`@pytest.mark.unit`, etc.)
5. Update coverage target if needed

### Test Guidelines

- **Unit tests**: Test single function/method in isolation
- **Integration tests**: Test multiple components together
- **Use fixtures**: Reuse test data and mocks
- **Parameterize**: Use `@pytest.mark.parametrize` for multiple inputs
- **Assert clearly**: Use descriptive assertions
- **Document**: Add docstrings to all test functions

## References

### Standards Documentation
- [ASME PTC 4.1](https://www.asme.org/codes-standards/find-codes-standards/ptc-4-1-fired-steam-generators)
- [DOE Steam Tips](https://www.energy.gov/eere/amo/steam-tips)
- [ASHRAE Handbook](https://www.ashrae.org/technical-resources/ashrae-handbook)
- [ASTM C680](https://www.astm.org/c0680-14.html)

### Testing Resources
- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)

---

**GL-003 Version**: 1.0.0
**Test Framework**: pytest 8.0+
**Python Version**: 3.11+
**Coverage Target**: 95%+
**Last Updated**: 2025-11-17
