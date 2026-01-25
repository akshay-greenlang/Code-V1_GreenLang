# E2E Integration Tests - Quick Start Guide

**Team 8: Quality Assurance Lead**
**Created:** 2025-11-09

---

## Quick Test Execution

### Run All 5 Priority Tests

```bash
# From project root
cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform

# Run all priority tests
pytest tests/integration/test_full_5_agent_pipeline.py \
       tests/integration/test_sap_integration_e2e.py \
       tests/integration/test_batch_processing_10k.py \
       tests/integration/test_monte_carlo_uncertainty.py \
       tests/integration/test_error_scenarios.py \
       -v --tb=short

# Estimated time: ~15 minutes
```

---

## Individual Test Execution

### 1. Full 5-Agent Pipeline Test (~2 min)

```bash
pytest tests/integration/test_full_5_agent_pipeline.py -v

# Specific test
pytest tests/integration/test_full_5_agent_pipeline.py::TestFull5AgentPipeline::test_full_pipeline_100_suppliers -v
```

**What it tests:**
- Complete pipeline: Intake → Calculator → Hotspot → Engagement → Reporting
- 100 suppliers end-to-end
- Data flow validation
- Performance <30 seconds

---

### 2. SAP Integration Test (~1 min)

```bash
pytest tests/integration/test_sap_integration_e2e.py -v

# Specific test
pytest tests/integration/test_sap_integration_e2e.py::TestSAPIntegrationE2E::test_sap_extraction_to_report -v
```

**What it tests:**
- SAP S/4HANA data extraction
- Schema mapping
- Emissions calculation from SAP data
- Report generation

---

### 3. Batch Processing 10K Test (~5 min)

```bash
pytest tests/integration/test_batch_processing_10k.py -v

# Run intake only (faster)
pytest tests/integration/test_batch_processing_10k.py::TestBatchProcessing10K::test_batch_process_10k_suppliers_intake -v
```

**What it tests:**
- 10,000 suppliers in batches
- Memory management
- Throughput >33 rec/s
- No memory leaks

---

### 4. Monte Carlo Uncertainty Test (~3 min)

```bash
pytest tests/integration/test_monte_carlo_uncertainty.py -v

# Run performance test
pytest tests/integration/test_monte_carlo_uncertainty.py::TestMonteCarloUncertainty::test_monte_carlo_performance -v
```

**What it tests:**
- Monte Carlo simulation (1000 samples)
- Uncertainty propagation
- 95% confidence intervals
- Portfolio aggregation

---

### 5. Error Scenarios Test (~2 min)

```bash
pytest tests/integration/test_error_scenarios.py -v

# Run circuit breaker test
pytest tests/integration/test_error_scenarios.py::TestErrorScenarios::test_circuit_breaker_state_transitions -v
```

**What it tests:**
- Circuit breaker pattern
- Retry with exponential backoff
- Fallback mechanisms
- Graceful degradation

---

## Test with Coverage

```bash
# Generate coverage report
pytest tests/integration/ --cov=services/agents --cov-report=html --cov-report=term

# View HTML report
# Windows
start htmlcov/index.html

# Mac/Linux
open htmlcov/index.html
```

---

## Fast Test Execution (Critical Only)

```bash
# Run only critical path tests (~5 min)
pytest tests/integration/ -m critical -v

# Run with parallel execution (requires pytest-xdist)
pip install pytest-xdist
pytest tests/integration/ -m critical -n auto -v
```

---

## Debugging Failed Tests

### Show detailed output

```bash
pytest tests/integration/test_full_5_agent_pipeline.py -vv -s
```

### Show last failed tests

```bash
pytest --lf -v
```

### Run specific test with debugging

```bash
pytest tests/integration/test_full_5_agent_pipeline.py::TestFull5AgentPipeline::test_full_pipeline_100_suppliers -vv -s --pdb
```

---

## Test Markers

### Run by category

```bash
# Critical tests only
pytest tests/integration/ -m critical -v

# Performance tests only
pytest tests/integration/ -m performance -v

# Monte Carlo tests only
pytest tests/integration/ -m monte_carlo -v

# Resilience tests only
pytest tests/integration/ -m resilience -v
```

---

## Environment Setup

### Install test dependencies

```bash
pip install pytest pytest-asyncio pytest-cov pytest-timeout psutil scipy numpy pandas
```

### Check test discovery

```bash
pytest --collect-only tests/integration/
```

---

## Expected Output

### Successful Test Run

```
========================= test session starts =========================
collected 25 items

test_full_5_agent_pipeline.py::TestFull5AgentPipeline::test_full_pipeline_100_suppliers PASSED [4%]
test_sap_integration_e2e.py::TestSAPIntegrationE2E::test_sap_extraction_to_report PASSED [8%]
test_batch_processing_10k.py::TestBatchProcessing10K::test_batch_process_10k_suppliers_intake PASSED [12%]
...

========================= 25 passed in 900.23s =========================
```

---

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the correct directory
cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Memory Errors (Batch Tests)

```bash
# Reduce batch size in test
# Edit test_batch_processing_10k.py
# Change: supplier_count = 10000 → supplier_count = 1000
```

### Timeout Errors

```bash
# Increase timeout
pytest tests/integration/ --timeout=600 -v
```

---

## Performance Benchmarks

| Test | Suppliers | Expected Time | Success Criteria |
|------|-----------|---------------|------------------|
| Full Pipeline | 100 | <30s | All agents complete |
| SAP Integration | 100 | <30s | Data extracted & calculated |
| Batch Intake | 10,000 | <300s | >33 rec/s throughput |
| Monte Carlo | 1,000 x 1,000 | <30s | >10K calc/s |
| Error Scenarios | N/A | <2min | All patterns work |

---

## Test File Overview

```
tests/integration/
├── conftest.py                        # Test fixtures and factories
├── test_full_5_agent_pipeline.py     # Priority 1: Full pipeline
├── test_sap_integration_e2e.py       # Priority 2: SAP integration
├── test_batch_processing_10k.py      # Priority 3: Batch 10K
├── test_monte_carlo_uncertainty.py   # Priority 4: Uncertainty
├── test_error_scenarios.py           # Priority 5: Resilience
├── TEST_SUITE_SUMMARY.md             # Detailed documentation
└── QUICK_START.md                    # This file
```

**Total Lines of Code:** 2,917 lines across 5 test files

---

## Next Steps

1. **Run tests**
   ```bash
   pytest tests/integration/ -m critical -v
   ```

2. **Review results**
   - Check for failures
   - Review performance metrics
   - Verify coverage

3. **Fix issues**
   - Address any test failures
   - Optimize slow tests
   - Add missing test cases

4. **Integrate into CI/CD**
   - Add to GitHub Actions
   - Set up automated runs
   - Configure alerts

---

## Support

For questions or issues:
- Check TEST_SUITE_SUMMARY.md for detailed documentation
- Review test output logs
- Check agent implementation for changes

---

**Happy Testing! 🧪**
