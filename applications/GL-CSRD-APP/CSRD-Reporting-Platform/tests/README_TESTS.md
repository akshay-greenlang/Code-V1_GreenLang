# CSRD Platform - Testing Guide

## Quick Start

### Run All CalculatorAgent Tests
```bash
cd C:\Users\aksha\Code-V1_GreenLang\GL-CSRD-APP\CSRD-Reporting-Platform
python -m pytest tests/test_calculator_agent.py -v
```

### Run Specific Test Category
```bash
# Initialization tests
python -m pytest tests/test_calculator_agent.py::TestCalculatorAgentInitialization -v

# CRITICAL: Reproducibility tests (zero hallucination)
python -m pytest tests/test_calculator_agent.py::TestReproducibility -v

# Formula engine tests
python -m pytest tests/test_calculator_agent.py::TestFormulaEngine -v

# ESRS metric calculations
python -m pytest tests/test_calculator_agent.py::TestESRSMetricCalculations -v

# Integration tests
python -m pytest tests/test_calculator_agent.py::TestIntegration -v
```

### Run with Coverage Report
```bash
python -m pytest tests/test_calculator_agent.py --cov=agents.calculator_agent --cov-report=html
```

Open `htmlcov/index.html` to view coverage report.

### Run Performance Tests
```bash
python -m pytest tests/test_calculator_agent.py -k "performance" -v
```

### Run Critical Tests Only
```bash
python -m pytest tests/test_calculator_agent.py -m critical -v
```

## Test Categories

| Category | Description | Test Count | Critical |
|----------|-------------|------------|----------|
| Initialization | Agent setup and database loading | 4 | ⚠️ Medium |
| Formula Engine | Core calculation engine | 10 | 🔴 HIGH |
| Emission Factors | Database lookups | 7 | 🔴 HIGH |
| ESRS Metrics | Actual metric calculations | 10 | 🔴 HIGH |
| Reproducibility | Zero hallucination verification | 4 | 🔴 CRITICAL |
| Integration | End-to-end scenarios | 5 | ⚠️ Medium |
| Provenance | Audit trail tracking | 4 | ⚠️ Medium |
| Error Handling | Edge cases and errors | 8 | 🔴 HIGH |
| Dependencies | Topological sort | 2 | ⚠️ Medium |
| Formula Retrieval | Database queries | 3 | ⚠️ Medium |
| Pydantic Models | Model validation | 3 | 🟢 Low |

## Success Criteria

### All Tests Must Pass
```
✅ 60+ test cases passing
✅ 0 failures
✅ 0 errors
✅ Coverage >= 95%
```

### Performance Targets
```
✅ < 5ms per metric calculation
✅ Total test suite runs in < 30 seconds
```

### Zero Hallucination Verification
```
✅ Reproducibility tests pass (10 runs, bit-perfect results)
✅ All calculations deterministic
✅ No LLM in calculation path
```

## Installation

### Install Dependencies
```bash
pip install pytest pytest-cov pandas pyyaml pydantic
```

### Verify Installation
```bash
python -m pytest --version
```

## Test Files

1. **test_calculator_agent.py** (850+ lines)
   - Comprehensive test suite for CalculatorAgent
   - 11 test categories
   - 60+ test cases
   - 100% coverage target

2. **TEST_CALCULATOR_AGENT_SUMMARY.md**
   - Detailed summary of test coverage
   - Test case descriptions
   - Recommendations for future enhancements

3. **README_TESTS.md** (this file)
   - Quick reference guide
   - How to run tests
   - Success criteria

## Troubleshooting

### Tests Fail Due to Missing Files
```bash
# Verify all required files exist
ls data/esrs_formulas.yaml
ls data/emission_factors.json
ls examples/demo_esg_data.csv
```

### Import Errors
```bash
# Verify you're in the correct directory
cd C:\Users\aksha\Code-V1_GreenLang\GL-CSRD-APP\CSRD-Reporting-Platform

# Verify Python path includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Performance Tests Fail
```bash
# Run with verbose timing output
python -m pytest tests/test_calculator_agent.py::TestIntegration::test_calculate_performance_target -v -s
```

## Next Steps

After CalculatorAgent tests pass:

1. **IntakeAgent Tests** - Data ingestion and validation
2. **MaterialityAgent Tests** - AI-powered materiality assessment
3. **AggregatorAgent Tests** - Framework mapping and benchmarking
4. **ReportingAgent Tests** - XBRL/ESEF generation
5. **AuditAgent Tests** - Compliance validation

## Contact

For questions or issues:
- **Project**: GL-CSRD-APP
- **Component**: CalculatorAgent Tests
- **Author**: GreenLang CSRD Team
- **Date**: 2025-10-18
