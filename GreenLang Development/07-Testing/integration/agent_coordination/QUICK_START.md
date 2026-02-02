# Quick Start Guide: Agent Coordination Tests

## Installation

```bash
# Install test dependencies
pip install -r requirements_test.txt

# Or install specific packages
pip install pytest pytest-asyncio pytest-cov pytest-timeout
```

## Running Tests

### Quick Smoke Test (Fast)

```bash
# Run all coordination tests
pytest tests/integration/agent_coordination/ -v

# Expected output:
# ✓ 75+ tests passed in < 2 minutes
```

### Run Individual Test Suites

```bash
# Test GL-001 ↔ GL-002 (THERMOSYNC ↔ FLAMEGUARD)
pytest tests/integration/agent_coordination/test_gl001_gl002_coordination.py -v

# Test GL-001 ↔ GL-006 (THERMOSYNC ↔ HEATRECLAIM)
pytest tests/integration/agent_coordination/test_gl001_gl006_coordination.py -v

# Test GL-003 ↔ GL-008 (STEAMWISE ↔ TRAPCATCHER)
pytest tests/integration/agent_coordination/test_gl003_gl008_coordination.py -v

# Test GL-002 ↔ GL-010 (FLAMEGUARD ↔ EMISSIONWATCH)
pytest tests/integration/agent_coordination/test_gl002_gl010_coordination.py -v

# Test GL-001 ↔ GL-009 (THERMOSYNC ↔ THERMALIQ)
pytest tests/integration/agent_coordination/test_gl001_gl009_coordination.py -v
```

### Run with Coverage

```bash
# Generate coverage report
pytest tests/integration/agent_coordination/ \
  --cov=greenlang.agents \
  --cov-report=html \
  --cov-report=term

# View HTML report
# Open: htmlcov/index.html
```

### Run Specific Tests

```bash
# Run single test
pytest tests/integration/agent_coordination/test_gl001_gl002_coordination.py::TestGL001GL002Coordination::test_boiler_optimization_request -v

# Run test class
pytest tests/integration/agent_coordination/test_gl001_gl002_coordination.py::TestGL001GL002Coordination -v

# Run tests matching pattern
pytest tests/integration/agent_coordination/ -k "efficiency" -v
```

### Performance Testing

```bash
# Run performance tests only
pytest tests/integration/agent_coordination/ -m performance -v

# Run with performance profiling
pytest tests/integration/agent_coordination/ --durations=10 -v
```

## Test Structure

```
tests/integration/agent_coordination/
├── __init__.py                          # Package init
├── conftest.py                          # Shared fixtures
├── pytest.ini                           # Pytest configuration
├── README.md                            # Full documentation
├── QUICK_START.md                       # This file
├── test_gl001_gl002_coordination.py     # GL-001 ↔ GL-002 tests (15+ tests)
├── test_gl001_gl006_coordination.py     # GL-001 ↔ GL-006 tests (15+ tests)
├── test_gl003_gl008_coordination.py     # GL-003 ↔ GL-008 tests (15+ tests)
├── test_gl002_gl010_coordination.py     # GL-002 ↔ GL-010 tests (15+ tests)
└── test_gl001_gl009_coordination.py     # GL-001 ↔ GL-009 tests (15+ tests)
```

## Common Test Scenarios

### 1. Boiler Optimization (GL-001 ↔ GL-002)

```python
# GL-001 requests boiler optimization
result = await gl001.optimize_boiler_for_demand(
    demand={'steam_output_kg_h': 40000, 'pressure_bar': 40},
    boiler_optimizer=gl002
)

# Assert success
assert result['status'] == 'success'
assert result['data']['efficiency_gain_percent'] > 0
```

### 2. Waste Heat Recovery (GL-001 ↔ GL-006)

```python
# GL-001 identifies waste heat streams
streams = await gl001.identify_waste_heat_streams(sensor_data)

# GL-006 analyzes recovery opportunities
opportunities = await gl006.analyze_recovery_opportunities(
    waste_streams=streams['waste_heat_streams'],
    constraints={'max_payback_years': 3.0}
)

# GL-001 updates heat distribution
update = await gl001.update_heat_distribution(
    recovery_opportunities=opportunities['recommended_opportunities']
)

assert update['status'] == 'success'
```

### 3. Steam Trap Monitoring (GL-003 ↔ GL-008)

```python
# GL-003 detects pressure anomalies
anomalies = await gl003.detect_pressure_anomalies(steam_data)

# GL-003 calls GL-008 for inspection
inspection = await gl008.inspect_steam_traps(
    anomaly_locations=anomalies['anomalies']
)

# GL-003 updates efficiency
efficiency = await gl003.update_efficiency_calculations(
    trap_failures=inspection['failed_trap_locations']
)

assert efficiency['status'] == 'success'
```

### 4. Emissions Compliance (GL-002 ↔ GL-010)

```python
# GL-002 requests emission constraints
constraints = await gl010.get_emission_constraints(operation_params)

# GL-002 optimizes within constraints
optimization = await gl002.optimize_with_constraints(
    boiler_data=boiler_data,
    emission_constraints=constraints
)

# GL-010 validates compliance
validation = await gl010.validate_compliance(
    emissions_data=optimization['optimization_result']
)

assert validation['compliance_status'] == 'COMPLIANT'
```

### 5. Thermal Efficiency Analysis (GL-001 ↔ GL-009)

```python
# GL-001 requests efficiency analysis
analysis = await gl001.request_efficiency_analysis(
    thermal_data=thermal_data,
    analyzer=gl009
)

# GL-001 uses data for optimization
optimization = await gl001.use_efficiency_data_for_optimization(
    efficiency_data=analysis['efficiency_analysis']
)

assert optimization['status'] == 'success'
```

## Key Assertions

### Success Assertions

```python
# Coordination success
assert result['status'] == 'success'

# Data integrity
assert 'provenance_hash' in result

# Performance
assert result['execution_time_ms'] < 200

# Compliance
assert result['compliance_status'] == 'COMPLIANT'
```

### Data Format Assertions

```python
# Message format
assert 'agent_id' in result
assert 'timestamp' in result

# Required fields
assert 'data' in result
assert isinstance(result['data'], dict)
```

## Debugging Tests

### Run with Debug Output

```bash
# Verbose output
pytest tests/integration/agent_coordination/ -vv

# Show print statements
pytest tests/integration/agent_coordination/ -s

# Stop on first failure
pytest tests/integration/agent_coordination/ -x

# Drop into debugger on failure
pytest tests/integration/agent_coordination/ --pdb
```

### Check Test Discovery

```bash
# List all tests
pytest tests/integration/agent_coordination/ --collect-only

# List tests with markers
pytest tests/integration/agent_coordination/ --markers
```

## Expected Results

### Test Metrics

- **Total Tests**: 75+ tests
- **Pass Rate**: 100%
- **Coverage**: 85%+
- **Execution Time**: < 2 minutes
- **Success Rate**: 95%+ for performance tests

### Test Distribution

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| GL-001 ↔ GL-002 | 15+ | Boiler optimization, error handling, concurrency |
| GL-001 ↔ GL-006 | 15+ | Waste heat recovery, economic analysis |
| GL-003 ↔ GL-008 | 15+ | Steam trap monitoring, maintenance prioritization |
| GL-002 ↔ GL-010 | 15+ | Emissions compliance, multi-objective optimization |
| GL-001 ↔ GL-009 | 15+ | Thermal efficiency, exergy analysis |

## Troubleshooting

### Issue: Async tests failing

```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Use auto mode
pytest tests/integration/agent_coordination/ --asyncio-mode=auto
```

### Issue: Import errors

```bash
# Install package in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/GreenLang"
```

### Issue: Slow tests

```bash
# Run only fast tests
pytest tests/integration/agent_coordination/ -m "not slow"

# Parallelize with pytest-xdist
pip install pytest-xdist
pytest tests/integration/agent_coordination/ -n auto
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run Agent Coordination Tests
  run: |
    pytest tests/integration/agent_coordination/ -v \
      --cov=greenlang.agents \
      --cov-report=xml \
      --junitxml=junit.xml
```

### GitLab CI

```yaml
test:coordination:
  script:
    - pytest tests/integration/agent_coordination/ -v
  coverage: '/TOTAL.*\s+(\d+%)$/'
```

## Next Steps

1. **Run Tests**: Execute the test suite
2. **Review Coverage**: Check coverage report
3. **Fix Failures**: Address any failing tests
4. **Add Tests**: Extend coverage for new scenarios
5. **Document**: Update README with new test cases

## Support

- **Documentation**: See [README.md](README.md) for full details
- **Issues**: Report bugs in GitHub Issues
- **Questions**: Ask in team Slack channel

## Quick Reference

```bash
# Full test suite
pytest tests/integration/agent_coordination/ -v

# With coverage
pytest tests/integration/agent_coordination/ --cov --cov-report=html

# Specific suite
pytest tests/integration/agent_coordination/test_gl001_gl002_coordination.py -v

# Performance tests
pytest tests/integration/agent_coordination/ -m performance

# Stop on first failure
pytest tests/integration/agent_coordination/ -x
```

---

**Ready to test!** 🚀

Run `pytest tests/integration/agent_coordination/ -v` to get started.
