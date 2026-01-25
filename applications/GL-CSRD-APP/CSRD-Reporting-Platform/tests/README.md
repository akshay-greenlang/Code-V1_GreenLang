# GL-CSRD Test Suite Organization Guide

**Total Tests**: 975 tests across 14 test files
**Target Coverage**: 90%+
**Test Infrastructure**: Complete and ready for execution

---

## Overview

This test suite represents the most comprehensive validation of the GL-CSRD platform, with **975 tests** covering all aspects of CSRD/ESRS digital reporting functionality. This is **4.6× larger** than the CBAM test suite.

**Critical Gap Addressed**: These tests have NEVER been executed. This guide enables comprehensive test execution to prove the application works.

---

## Test Suite Structure

### Directory Layout

```
tests/
├── README.md                                    # This file
├── conftest.py                                  # Shared fixtures (ESRS standards, etc.)
│
├── test_calculator_agent.py                     # 109 tests - CRITICAL (Zero Hallucination)
├── test_reporting_agent.py                      # 133 tests - XBRL/ESEF generation
├── test_audit_agent.py                          # 115 tests - Compliance validation
├── test_intake_agent.py                         # 107 tests - Data ingestion
├── test_provenance.py                           # 101 tests - Audit trail
├── test_aggregator_agent.py                     # 75 tests - Framework mapping
├── test_cli.py                                  # 69 tests - CLI interface
├── test_sdk.py                                  # 61 tests - Python SDK
├── test_pipeline_integration.py                 # 59 tests - Pipeline orchestration
├── test_validation.py                           # 55 tests - Data validation
├── test_materiality_agent.py                    # 45 tests - Double materiality
├── test_encryption.py                           # 24 tests - Data encryption
├── test_automated_filing_agent_security.py      # 16 tests - Security
└── test_e2e_workflows.py                        # 6 tests - End-to-end
```

---

## Test Categories

### 1. Critical Agent Tests (109 tests)

**File**: `test_calculator_agent.py`
**Priority**: CRITICAL - Zero hallucination requirement

The Calculator Agent is the most critical component because:
- Financial calculations affect regulatory compliance
- Must be 100% deterministic (zero hallucination)
- 520+ ESRS formulas must all work correctly
- GHG Protocol emission factor lookups must be accurate

**Key Test Areas**:
- Formula engine validation (520+ formulas)
- GHG emissions calculations (Scope 1, 2, 3)
- Zero hallucination guarantee
- Calculation provenance tracking
- Emission factor lookups
- Unit conversions
- Error handling

**Execution**:
```bash
# Run all calculator tests
pytest tests/test_calculator_agent.py -v

# Run only critical tests
pytest tests/test_calculator_agent.py -m critical

# With coverage
pytest tests/test_calculator_agent.py --cov=agents.calculator_agent
```

### 2. Reporting Agent Tests (133 tests)

**File**: `test_reporting_agent.py`
**Priority**: HIGH - Regulatory filing

XBRL/ESEF report generation for EU regulatory compliance.

**Key Test Areas**:
- XBRL iXBRL generation
- ESEF package creation
- Taxonomy mapping (ESRS taxonomy)
- Digital signatures
- Report validation
- Multi-language support
- Report packaging

**Execution**:
```bash
pytest tests/test_reporting_agent.py -v
pytest tests/test_reporting_agent.py -m xbrl
```

### 3. Audit Agent Tests (115 tests)

**File**: `test_audit_agent.py`
**Priority**: HIGH - Compliance

Audit trail and compliance validation.

**Key Test Areas**:
- Audit trail generation
- Compliance checks (EU CSRD)
- Data lineage tracking
- Regulatory validation
- Audit-ready reports
- Historical tracking

**Execution**:
```bash
pytest tests/test_audit_agent.py -v
pytest tests/test_audit_agent.py -m compliance
```

### 4. Intake Agent Tests (107 tests)

**File**: `test_intake_agent.py`
**Priority**: HIGH - Data quality

Multi-format data ingestion and validation.

**Key Test Areas**:
- CSV/Excel/JSON/XML parsing
- 1,082 ESRS data points mapping
- Schema validation
- Error handling
- Template processing
- Data quality checks

**Execution**:
```bash
pytest tests/test_intake_agent.py -v
pytest tests/test_intake_agent.py -m data_ingestion
```

### 5. Provenance System Tests (101 tests)

**File**: `test_provenance.py`
**Priority**: MEDIUM - Audit requirements

Complete data lineage and audit trail.

**Key Test Areas**:
- Data lineage tracking
- Cryptographic hashing
- Tamper detection
- Historical versioning
- Audit trail completeness

**Execution**:
```bash
pytest tests/test_provenance.py -v
```

### 6. Aggregator Agent Tests (75 tests)

**File**: `test_aggregator_agent.py`
**Priority**: MEDIUM - Multi-framework

Framework integration (TCFD, GRI, SASB → ESRS).

**Key Test Areas**:
- TCFD → ESRS mapping (350+ mappings)
- GRI Standards integration
- SASB Standards integration
- Framework harmonization
- Cross-framework validation

**Execution**:
```bash
pytest tests/test_aggregator_agent.py -v
pytest tests/test_aggregator_agent.py -m tcfd
```

### 7. CLI Tests (69 tests)

**File**: `test_cli.py`
**Priority**: MEDIUM - User interface

Command-line interface validation.

**Key Test Areas**:
- All CLI commands
- Interactive mode
- Batch processing
- Error handling
- Output formatting

**Execution**:
```bash
pytest tests/test_cli.py -v
pytest tests/test_cli.py -m cli
```

### 8-14. Additional Test Files

See individual files for specific test coverage.

---

## Test Markers

### Priority Markers
- `@pytest.mark.critical` - Must pass (109 tests)
- `@pytest.mark.high` - High priority (500+ tests)
- `@pytest.mark.medium` - Standard priority (300+ tests)

### ESRS Standard Markers
- `@pytest.mark.esrs1` - ESRS 1 General Requirements
- `@pytest.mark.esrs2` - ESRS 2 General Disclosures
- `@pytest.mark.esrs_e1` - ESRS E1 Climate Change
- `@pytest.mark.esrs_e2` - ESRS E2 Pollution
- `@pytest.mark.esrs_e3` - ESRS E3 Water & Marine
- `@pytest.mark.esrs_e4` - ESRS E4 Biodiversity
- `@pytest.mark.esrs_e5` - ESRS E5 Circular Economy
- `@pytest.mark.esrs_s1` - ESRS S1 Own Workforce
- `@pytest.mark.esrs_s2` - ESRS S2 Value Chain Workers
- `@pytest.mark.esrs_s3` - ESRS S3 Communities
- `@pytest.mark.esrs_s4` - ESRS S4 Consumers
- `@pytest.mark.esrs_g1` - ESRS G1 Business Conduct

### Agent Markers
- `@pytest.mark.calculator` - Calculator Agent
- `@pytest.mark.reporting` - Reporting Agent
- `@pytest.mark.audit` - Audit Agent
- `@pytest.mark.intake` - Intake Agent
- And more...

### Type Markers
- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance tests

---

## Shared Fixtures (conftest.py)

### Path Fixtures
- `base_path` - Project root path
- `data_path` - Data directory
- `temp_dir` - Temporary directory for tests

### ESRS Data Fixtures
- `esrs_formulas` - 520+ formulas
- `emission_factors` - GHG emission factors
- `esrs_data_points` - 1,082 data points catalog

### ESRS Standard Fixtures (12 fixtures)
- `esrs1_data` - ESRS 1 test data
- `esrs2_data` - ESRS 2 test data
- `esrs_e1_data` - ESRS E1 test data (Climate)
- `esrs_e2_data` - ESRS E2 test data (Pollution)
- `esrs_e3_data` - ESRS E3 test data (Water)
- `esrs_e4_data` - ESRS E4 test data (Biodiversity)
- `esrs_e5_data` - ESRS E5 test data (Circular Economy)
- `esrs_s1_data` - ESRS S1 test data (Workforce)
- `esrs_s2_data` - ESRS S2 test data (Value Chain)
- `esrs_s3_data` - ESRS S3 test data (Communities)
- `esrs_s4_data` - ESRS S4 test data (Consumers)
- `esrs_g1_data` - ESRS G1 test data (Governance)

### Sample Data Fixtures
- `sample_esg_data` - 100-row ESG dataset
- `sample_ghg_data` - GHG emissions data
- `sample_company_info` - Company metadata

### Framework Integration Fixtures
- `tcfd_metrics` - TCFD framework data
- `gri_metrics` - GRI Standards data
- `sasb_metrics` - SASB Standards data

### Mock Agent Fixtures
- `mock_calculator_agent` - Mock Calculator
- `mock_intake_agent` - Mock Intake
- `mock_reporting_agent` - Mock Reporting

---

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov --cov-report=html

# Run parallel (8 workers)
pytest tests/ -n 8
```

### Using Test Scripts

```bash
# Sequential execution with coverage and HTML report
./scripts/run_all_tests.sh --coverage --html

# Parallel execution (auto-detect cores)
./scripts/run_tests_parallel.sh

# Parallel with 8 workers
./scripts/run_tests_parallel.sh --workers=8

# Fast tests only (skip slow)
./scripts/run_all_tests.sh --fast

# Critical tests only
./scripts/run_all_tests.sh --critical

# By agent (sequential)
./scripts/run_all_tests.sh --by-agent
```

### Selective Execution

```bash
# By marker
pytest -m calculator              # Calculator tests only
pytest -m "critical"              # Critical tests only
pytest -m "esrs_e1"               # Climate change tests
pytest -m "not slow"              # Skip slow tests

# By file
pytest tests/test_calculator_agent.py
pytest tests/test_reporting_agent.py

# By test name pattern
pytest -k "test_ghg"              # All GHG tests
pytest -k "test_calculation"      # All calculation tests

# Multiple markers
pytest -m "calculator and critical"
pytest -m "esrs_e1 or esrs_e2"
```

### With Coverage

```bash
# Basic coverage
pytest --cov=agents tests/

# HTML coverage report
pytest --cov=agents --cov-report=html tests/

# Coverage with missing lines
pytest --cov=agents --cov-report=term-missing tests/

# Coverage by component
pytest --cov=agents.calculator_agent tests/test_calculator_agent.py
```

---

## Performance Benchmarking

```bash
# Run performance benchmarks
python scripts/benchmark_csrd.py

# Benchmarks include:
# - Calculator throughput (calculations/sec)
# - Intake throughput (records/sec)
# - Reporting speed (XBRL generation)
# - Aggregator mapping speed
# - End-to-end pipeline timing
# - Test suite execution estimates
```

---

## Test Reports

### HTML Test Report

```bash
# Generate comprehensive HTML report
python scripts/generate_test_report.py

# With custom output
python scripts/generate_test_report.py --output report.html

# Report includes:
# - Test summary cards
# - Agent-by-agent results
# - ESRS coverage matrix
# - Coverage progress bars
# - Executive summary
```

### Coverage Report

```bash
# After running tests with --cov
# Open: htmlcov/index.html
```

### JUnit XML (CI/CD)

```bash
# Automatically generated in test-reports/junit/
# Used by CI/CD systems (GitHub Actions, Jenkins, etc.)
```

---

## Test Organization by ESRS Standard

### Climate (ESRS E1) - Highest Coverage
```bash
pytest -m esrs_e1 -v
# Tests: GHG emissions, climate targets, transition plans
```

### All Environmental (E1-E5)
```bash
pytest -m "esrs_e1 or esrs_e2 or esrs_e3 or esrs_e4 or esrs_e5" -v
```

### All Social (S1-S4)
```bash
pytest -m "esrs_s1 or esrs_s2 or esrs_s3 or esrs_s4" -v
```

### Governance (G1)
```bash
pytest -m esrs_g1 -v
```

---

## Test Execution Estimates

### Sequential Execution
- **Total Time**: ~8 minutes (975 tests)
- **Average per test**: ~0.5 seconds
- **Use case**: Debug mode, single-threaded

### Parallel Execution (4 workers)
- **Total Time**: ~2 minutes
- **Speedup**: 4×
- **Use case**: Standard development

### Parallel Execution (8 workers)
- **Total Time**: ~1 minute
- **Speedup**: 8×
- **Use case**: CI/CD, fast feedback

---

## Coverage Targets

| Component | Target | Priority |
|-----------|--------|----------|
| Calculator Agent | 100% | CRITICAL |
| Reporting Agent | 95% | HIGH |
| Audit Agent | 95% | HIGH |
| Intake Agent | 90% | HIGH |
| Provenance System | 90% | MEDIUM |
| Other Components | 85% | MEDIUM |
| **Overall** | **90%** | **TARGET** |

---

## Best Practices

### 1. Always Run Critical Tests First
```bash
pytest -m critical --maxfail=1
```

### 2. Use Parallel Execution for Speed
```bash
pytest -n auto tests/
```

### 3. Generate Coverage Reports
```bash
pytest --cov --cov-report=html tests/
```

### 4. Run Performance Benchmarks Regularly
```bash
python scripts/benchmark_csrd.py
```

### 5. Use Markers for Targeted Testing
```bash
pytest -m "calculator and not slow"
```

---

## Troubleshooting

### Issue: Tests not discovered
**Solution**: Ensure pytest can find tests
```bash
pytest --collect-only tests/
```

### Issue: Import errors
**Solution**: Install dependencies and check PYTHONPATH
```bash
pip install -r requirements-test.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Slow test execution
**Solution**: Use parallel execution
```bash
pytest -n auto tests/
```

### Issue: Coverage not working
**Solution**: Install pytest-cov
```bash
pip install pytest-cov
pytest --cov=agents tests/
```

### Issue: Fixture not found
**Solution**: Check conftest.py is present
```bash
ls -la tests/conftest.py
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run tests with coverage
  run: |
    pytest tests/ -n auto --cov --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

### Jenkins Example
```groovy
stage('Test') {
    steps {
        sh 'pytest tests/ -n auto --junitxml=junit.xml'
    }
}
```

---

## Additional Resources

### Documentation
- [TEST_VALIDATION_CHECKLIST.md](../TEST_VALIDATION_CHECKLIST.md) - Complete validation checklist
- [pytest.ini](../pytest.ini) - Pytest configuration
- [requirements-test.txt](../requirements-test.txt) - Test dependencies

### Scripts
- [run_all_tests.sh](../scripts/run_all_tests.sh) - Main test runner
- [run_tests_parallel.sh](../scripts/run_tests_parallel.sh) - Parallel runner
- [benchmark_csrd.py](../scripts/benchmark_csrd.py) - Performance benchmarks
- [generate_test_report.py](../scripts/generate_test_report.py) - Report generator

### External Links
- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [pytest-xdist Documentation](https://pytest-xdist.readthedocs.io/)

---

## Summary

**Test Suite Ready**: 975 tests across 14 files
**Infrastructure Complete**: Scripts, fixtures, configuration
**Coverage Target**: 90%+
**Execution Ready**: Sequential and parallel modes available

**Next Steps**:
1. Install dependencies: `pip install -r requirements-test.txt`
2. Run test discovery: `pytest --collect-only tests/`
3. Execute critical tests: `./scripts/run_all_tests.sh --critical`
4. Run full suite: `./scripts/run_all_tests.sh --coverage --html`
5. Review reports: `htmlcov/index.html` and `test-reports/html/`

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-08
**Team**: B2 - GL-CSRD Test Execution Preparation
