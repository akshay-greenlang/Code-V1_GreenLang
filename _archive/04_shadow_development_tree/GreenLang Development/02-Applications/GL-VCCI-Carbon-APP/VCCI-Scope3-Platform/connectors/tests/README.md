# ERP Connector Integration and Performance Testing Framework

**GL-VCCI Scope 3 Platform**
**Phase 4 (Weeks 24-26)**
**Version: 1.0.0**

Comprehensive integration testing framework and throughput validation system for SAP, Oracle, and Workday ERP connectors.

---

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Running Tests](#running-tests)
- [Performance Targets](#performance-targets)
- [Test Categories](#test-categories)
- [CI/CD Integration](#cicd-integration)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting](#troubleshooting)

---

## Overview

This testing framework provides:

- **Integration Tests**: Real sandbox environment testing for SAP, Oracle, and Workday connectors
- **Performance Tests**: Throughput validation (100K records/hour target)
- **Mock Servers**: CI/CD-compatible mock servers for automated testing
- **Benchmarking**: Performance metrics tracking and reporting
- **Sandbox Management**: Utilities for sandbox setup and test data management

### Test Statistics

- **Total Test Files**: 12
- **Total Lines of Code**: ~2,250
- **Integration Tests**: 45+
- **Performance Tests**: 18+
- **Connectors Covered**: SAP, Oracle, Workday

---

## Test Structure

```
connectors/tests/
├── __init__.py                      # Test package initialization
├── conftest.py                      # Shared fixtures and utilities (~150 lines)
├── README.md                        # This file (~100 lines)
│
├── integration/                     # Integration tests
│   ├── __init__.py
│   ├── test_sap_integration.py      # SAP integration tests (~300 lines, 15 tests)
│   ├── test_oracle_integration.py   # Oracle integration tests (~250 lines, 12 tests)
│   ├── test_workday_integration.py  # Workday integration tests (~150 lines, 8 tests)
│   └── test_end_to_end.py          # Multi-connector E2E tests (~200 lines, 10 tests)
│
├── performance/                     # Performance tests
│   ├── __init__.py
│   ├── test_sap_throughput.py      # SAP throughput tests (~250 lines, 10 tests)
│   ├── test_oracle_throughput.py   # Oracle throughput tests (~200 lines, 8 tests)
│   └── benchmark_report.py         # Performance reporting (~150 lines)
│
└── sandbox/                         # Sandbox setup utilities
    ├── __init__.py
    ├── sap_sandbox_setup.py        # SAP sandbox utilities (~200 lines)
    ├── oracle_sandbox_setup.py     # Oracle sandbox utilities (~180 lines)
    └── workday_sandbox_setup.py    # Workday sandbox utilities (~120 lines)
```

---

## Prerequisites

### Required Software

- Python 3.10+
- pytest 7.0+
- pytest-timeout
- pytest-benchmark
- psutil
- redis-py
- responses (for mocking)

### Install Dependencies

```bash
pip install pytest pytest-timeout pytest-benchmark psutil redis responses requests
```

### Optional Dependencies

```bash
# For memory profiling
pip install memory_profiler

# For async testing
pip install pytest-asyncio
```

---

## Environment Setup

### 1. SAP Sandbox Configuration

Set these environment variables for SAP integration tests:

```bash
# SAP Sandbox Connection
export SAP_SANDBOX_URL="https://your-sap-sandbox.com"
export SAP_SANDBOX_CLIENT_ID="your_client_id"
export SAP_SANDBOX_CLIENT_SECRET="your_client_secret"
export SAP_SANDBOX_TOKEN_URL="https://your-sap-sandbox.com/oauth/token"
export SAP_SANDBOX_OAUTH_SCOPE="API_BUSINESS_PARTNER"

# Enable integration tests
export RUN_INTEGRATION_TESTS="true"
```

### 2. Oracle Sandbox Configuration

```bash
# Oracle Sandbox Connection
export ORACLE_SANDBOX_URL="https://your-oracle-sandbox.com"
export ORACLE_SANDBOX_CLIENT_ID="your_client_id"
export ORACLE_SANDBOX_CLIENT_SECRET="your_client_secret"
export ORACLE_SANDBOX_TOKEN_URL="https://your-oracle-sandbox.com/oauth/token"

# Enable integration tests
export RUN_INTEGRATION_TESTS="true"
```

### 3. Workday Sandbox Configuration

```bash
# Workday Sandbox Connection
export WORKDAY_SANDBOX_URL="https://your-tenant.workday.com"
export WORKDAY_SANDBOX_CLIENT_ID="your_client_id"
export WORKDAY_SANDBOX_CLIENT_SECRET="your_client_secret"
export WORKDAY_SANDBOX_TENANT="your_tenant_name"

# Enable integration tests
export RUN_INTEGRATION_TESTS="true"
```

### 4. Verify Sandbox Setup

Run sandbox verification scripts:

```bash
# Verify SAP sandbox
python connectors/tests/sandbox/sap_sandbox_setup.py

# Verify Oracle sandbox
python connectors/tests/sandbox/oracle_sandbox_setup.py

# Verify Workday sandbox
python connectors/tests/sandbox/workday_sandbox_setup.py
```

---

## Running Tests

### Run All Tests

```bash
# From project root
pytest connectors/tests/ -v
```

### Run Integration Tests Only

```bash
# All integration tests
pytest connectors/tests/integration/ -v -m integration

# SAP integration tests only
pytest connectors/tests/integration/test_sap_integration.py -v

# Oracle integration tests only
pytest connectors/tests/integration/test_oracle_integration.py -v

# Workday integration tests only
pytest connectors/tests/integration/test_workday_integration.py -v

# End-to-end tests
pytest connectors/tests/integration/test_end_to_end.py -v
```

### Run Performance Tests

```bash
# All performance tests (WARNING: These can take 1+ hour)
pytest connectors/tests/performance/ -v -m performance

# SAP throughput test (100K records)
pytest connectors/tests/performance/test_sap_throughput.py::TestSAP100KThroughput -v

# Oracle throughput test (100K records)
pytest connectors/tests/performance/test_oracle_throughput.py::TestOracle100KThroughput -v

# Specific performance test classes
pytest connectors/tests/performance/test_sap_throughput.py::TestSAPBatchPerformance -v
```

### Run Tests by Marker

```bash
# Run only SAP sandbox tests
pytest -v -m sap_sandbox

# Run only Oracle sandbox tests
pytest -v -m oracle_sandbox

# Run only Workday sandbox tests
pytest -v -m workday_sandbox

# Run slow tests
pytest -v -m slow

# Exclude slow tests
pytest -v -m "not slow"
```

### Run with Mock Servers (CI/CD)

When sandbox environment variables are not set, tests automatically use mock servers:

```bash
# Run integration tests with mocks (no sandbox needed)
unset RUN_INTEGRATION_TESTS
pytest connectors/tests/integration/ -v
```

---

## Performance Targets

### Throughput Targets

| Connector | Target Throughput | Target Time | Status |
|-----------|------------------|-------------|---------|
| SAP       | 100K records/hour | <3600 seconds | ✓ Validated |
| Oracle    | 100K records/hour | <3600 seconds | ✓ Validated |
| Workday   | 50K records/hour  | N/A | ✓ Sufficient |

### Performance Metrics Tracked

- **Throughput**: Records per second/hour
- **API Latency**: Average, P95, P99 (milliseconds)
- **Memory Usage**: Peak memory consumption (MB)
- **Cache Hit Rate**: Percentage of cache hits
- **Error Rate**: Percentage of failed requests
- **Success Rate**: Percentage of successful operations

### Example Performance Output

```
============================================================
SAP 100K Throughput Test Results
============================================================
Records extracted: 100,000
Time elapsed: 3,200.00 seconds (53.33 minutes)
Throughput: 112,500 records/hour
Average rate: 31.25 records/second
API calls: 100
Average latency: 450.50 ms
P95 latency: 850.20 ms
P99 latency: 1200.50 ms
Max memory: 256.80 MB
Errors: 2
============================================================
```

---

## Test Categories

### 1. Integration Tests

**SAP Integration Tests** (15 tests)
- Connection and authentication
- Purchase order extraction
- Delta synchronization
- Pagination
- Rate limiting
- Retry logic
- Data mapping
- Deduplication
- Multi-module extraction (MM, SD, FI)
- Audit logging
- End-to-end flows

**Oracle Integration Tests** (12 tests)
- Connection and authentication
- Purchase order extraction
- Delta synchronization
- Pagination with links array
- Multi-module extraction (Procurement, SCM, Financials)
- Data mapping
- End-to-end flows

**Workday Integration Tests** (8 tests)
- Connection and authentication
- Expense report extraction
- Commute survey extraction
- Business travel extraction
- Date range filtering
- Data mapping
- End-to-end flows

**End-to-End Tests** (10 tests)
- Multi-connector integration
- Concurrent extraction
- Data quality consistency
- Schema validation
- Error handling
- Graceful degradation

### 2. Performance Tests

**SAP Performance Tests** (10 tests)
- 100K throughput validation
- Batch processing performance
- Pagination performance
- Mapping performance
- Memory usage monitoring
- Concurrent extraction
- Latency distribution
- Reliability testing

**Oracle Performance Tests** (8 tests)
- 100K throughput validation
- Batch processing performance
- Pagination performance
- Mapping performance
- Memory usage monitoring
- API response time
- Reliability testing
- Scalability testing

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: ERP Connector Integration Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  integration-tests:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-timeout pytest-benchmark psutil responses

    - name: Run integration tests (with mocks)
      run: |
        pytest connectors/tests/integration/ -v -m integration

    - name: Run unit performance tests
      run: |
        pytest connectors/tests/performance/ -v -m "performance and not slow"

  performance-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-timeout pytest-benchmark

    - name: Run performance benchmarks
      env:
        SAP_SANDBOX_URL: ${{ secrets.SAP_SANDBOX_URL }}
        SAP_SANDBOX_CLIENT_ID: ${{ secrets.SAP_SANDBOX_CLIENT_ID }}
        SAP_SANDBOX_CLIENT_SECRET: ${{ secrets.SAP_SANDBOX_CLIENT_SECRET }}
        RUN_INTEGRATION_TESTS: "true"
      run: |
        pytest connectors/tests/performance/ -v --timeout=7200

    - name: Generate performance report
      run: |
        python connectors/tests/performance/benchmark_report.py

    - name: Upload performance report
      uses: actions/upload-artifact@v3
      with:
        name: performance-report
        path: benchmark_reports/
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any

    environment {
        SAP_SANDBOX_URL = credentials('sap-sandbox-url')
        SAP_SANDBOX_CLIENT_ID = credentials('sap-client-id')
        SAP_SANDBOX_CLIENT_SECRET = credentials('sap-client-secret')
        RUN_INTEGRATION_TESTS = 'true'
    }

    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest pytest-timeout pytest-benchmark'
            }
        }

        stage('Integration Tests') {
            steps {
                sh 'pytest connectors/tests/integration/ -v -m integration --junitxml=integration-results.xml'
            }
        }

        stage('Performance Tests') {
            steps {
                sh 'pytest connectors/tests/performance/ -v -m performance --timeout=7200 --junitxml=performance-results.xml'
            }
        }

        stage('Generate Reports') {
            steps {
                sh 'python connectors/tests/performance/benchmark_report.py'
                publishHTML([
                    reportDir: 'benchmark_reports',
                    reportFiles: 'benchmark_report_*.html',
                    reportName: 'Performance Report'
                ])
            }
        }
    }

    post {
        always {
            junit '**/*-results.xml'
        }
    }
}
```

---

## Interpreting Results

### Test Pass/Fail Criteria

**Integration Tests**
- ✅ PASS: All assertions pass, no exceptions
- ❌ FAIL: Assertion fails or exception raised
- ⏭️ SKIP: Sandbox not available (expected in CI/CD)

**Performance Tests**
- ✅ PASS: Meets throughput target (≥100K records/hour)
- ⚠️ WARNING: Close to target (90-99% of target)
- ❌ FAIL: Below target (<90% of target)

### Performance Report

HTML and JSON reports are generated in `benchmark_reports/`:

- **HTML Report**: Visual dashboard with charts and tables
- **JSON Report**: Machine-readable metrics for automation

View reports:
```bash
# Generate reports
python connectors/tests/performance/benchmark_report.py

# Open HTML report
open benchmark_reports/benchmark_report_*.html
```

### Key Metrics to Monitor

1. **Throughput**: Should meet or exceed 100K records/hour for SAP and Oracle
2. **P95 Latency**: Should be <2000ms for good user experience
3. **Memory Usage**: Should not exceed 500MB growth during extraction
4. **Success Rate**: Should be >95% for production readiness
5. **Error Rate**: Should be <5% for acceptable reliability

---

## Troubleshooting

### Common Issues

#### 1. Sandbox Connection Failures

**Symptom**: Tests skip with "Sandbox not available"

**Solution**:
```bash
# Verify environment variables
python connectors/tests/sandbox/sap_sandbox_setup.py

# Check network connectivity
curl -I $SAP_SANDBOX_URL

# Verify credentials
echo $SAP_SANDBOX_CLIENT_ID
```

#### 2. Performance Tests Timeout

**Symptom**: Tests timeout after 1 hour

**Solution**:
- Increase timeout: `pytest --timeout=7200`
- Check network latency to sandbox
- Verify sandbox has sufficient test data
- Run with smaller batch sizes for testing

#### 3. Memory Issues

**Symptom**: Tests crash with MemoryError

**Solution**:
- Reduce batch size in tests
- Enable garbage collection more frequently
- Monitor system resources
- Run tests sequentially instead of parallel

#### 4. Mock Server Issues

**Symptom**: Tests fail with "Connection refused"

**Solution**:
```python
# Verify mock server is activated
from connectors.tests.sandbox.sap_sandbox_setup import SAPSandboxSetup
mock_server = SAPSandboxSetup.create_mock_server()
# Use @responses.activate decorator
```

#### 5. Rate Limiting

**Symptom**: Tests fail with rate limit errors

**Solution**:
- Adjust rate limit config in sandbox setup
- Implement exponential backoff
- Run tests with smaller batches
- Space out test execution

---

## Maintenance

### Adding New Tests

1. Create test file in appropriate directory
2. Use existing fixtures from `conftest.py`
3. Add appropriate pytest markers
4. Update this README with new test info

### Updating Sandbox Data

```bash
# Regenerate test data
python connectors/tests/sandbox/sap_sandbox_setup.py

# Cleanup old test data
python -c "from connectors.tests.sandbox.sap_sandbox_setup import SAPSandboxSetup; SAPSandboxSetup().cleanup_test_data()"
```

### Performance Baseline

Track performance over time:

```bash
# Run baseline tests
pytest connectors/tests/performance/ -v > baseline_$(date +%Y%m%d).log

# Compare with previous baseline
diff baseline_20250101.log baseline_20250106.log
```

---

## Support and Contact

For issues or questions:

- **Documentation**: See main project README
- **Bug Reports**: Create GitHub issue
- **Performance Issues**: Contact platform team
- **Sandbox Access**: Contact IT Operations

---

## Version History

- **v1.0.0** (2025-11-06): Initial release
  - 45+ integration tests
  - 18+ performance tests
  - Sandbox utilities for all three connectors
  - Comprehensive documentation

---

**GL-VCCI Scope 3 Platform - Phase 4 (Weeks 24-26)**
**Integration Testing Framework**
**Version 1.0.0**
