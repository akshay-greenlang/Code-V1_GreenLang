# GreenLang v0.0.1 - Enhanced Production Readiness QA Checklist

## Overview
This enhanced checklist incorporates engineering best practices and automated testing strategies to ensure GreenLang v0.0.1 meets the highest quality standards.

## Pre-Release Sign-off Requirements
- [ ] All automated tests pass (pytest, tox, security scans)
- [ ] Test coverage ≥ 85% with no critical paths uncovered
- [ ] Zero high/critical security vulnerabilities
- [ ] Performance benchmarks met across all platforms
- [ ] Documentation validated and schemas documented
- [ ] Cross-platform compatibility verified (including ARM64)

---

## 1. Documentation Verification

### Core Documentation
- [ ] Verify GREENLANG_DOCUMENTATION.md covers all public features and APIs
- [ ] Validate all code examples execute without errors
- [ ] Confirm README.md quick start works on fresh installation
- [ ] Check all 12 CLI commands documented with working examples
- [ ] Verify API reference matches implementation (use docstring validation)

### Data Schema Documentation
- [ ] Document schema for `global_emission_factors.json`
- [ ] Document schema for `global_benchmarks.json`
- [ ] Document building input data schema
- [ ] Document workflow YAML schema
- [ ] Validate all JSON files against their schemas

### Platform-Specific Documentation
- [ ] Windows installation guide (including batch file usage)
- [ ] Linux installation guide (Ubuntu/RHEL/Debian)
- [ ] macOS installation guide (Intel and Apple Silicon)
- [ ] Docker deployment documentation
- [ ] Cloud deployment guides (AWS/Azure/GCP)

---

## 2. CLI Commands Testing

### Core Commands (All 12 Countries)
Countries to test: US, IN, EU, CN, JP, BR, KR, UK, DE, CA, AU, SG

- [ ] `greenlang --version` returns correct version
- [ ] `greenlang --help` displays all commands
- [ ] `greenlang calc` works in simple mode
- [ ] `greenlang calc --building` launches building mode
- [ ] `greenlang calc --building --country [COUNTRY]` for all 12 countries

### Input Validation Testing
- [ ] `greenlang analyze valid_building.json` processes correctly
- [ ] `greenlang analyze malformed.json` returns user-friendly error
- [ ] `greenlang analyze empty.json` handles gracefully
- [ ] `greenlang analyze nonexistent.json` provides helpful message
- [ ] `greenlang analyze ../../../etc/passwd` prevents path traversal

### Advanced Commands
- [ ] `greenlang benchmark --type [all 7 types] --country [all 12 countries]`
- [ ] `greenlang recommend` with all performance ratings
- [ ] `greenlang agents` lists exactly 9 agents
- [ ] `greenlang agent [each_agent]` shows correct details
- [ ] `greenlang ask "[query]"` handles various question types
- [ ] `greenlang run [workflow.yaml]` with valid/invalid workflows
- [ ] `greenlang init` creates correct template files

### Developer Interface Testing
- [ ] `greenlang dev` launches successfully
- [ ] Web interface responsive (if applicable - test with Playwright)
- [ ] API endpoints accessible
- [ ] Interactive features work correctly
- [ ] Error handling in dev mode

---

## 3. Agent Functionality Testing

### Core Agents with Edge Cases
- [ ] **FuelAgent**: All 15+ fuel types + invalid fuel types
- [ ] **BoilerAgent**: All boiler types + efficiency edge cases (0%, 100%, >100%)
- [ ] **CarbonAgent**: Aggregation with duplicates, empty sets, single item
- [ ] **InputValidatorAgent**: All valid formats + malformed inputs
- [ ] **ReportAgent**: All export formats + large datasets (>10MB)

### Enhanced Agents with Boundary Testing
- [ ] **BenchmarkAgent**: All country/building combinations + missing benchmarks
- [ ] **GridFactorAgent**: All countries + unsupported countries
- [ ] **BuildingProfileAgent**: All 7 types + unknown types
- [ ] **IntensityAgent**: Zero values, negative values, extreme values
- [ ] **RecommendationAgent**: All rating levels + edge cases

---

## 4. Automated Test Suite Execution

### Unit Tests
```bash
pytest tests/unit/ -v --cov=greenlang --cov-report=term-missing
```
- [ ] 100+ unit tests pass
- [ ] Coverage ≥ 90% for critical components
- [ ] No flaky tests (run 3x to verify)

### Integration Tests
```bash
pytest tests/integration/ -v --timeout=30
```
- [ ] 70+ integration tests pass
- [ ] Network isolation verified (no external calls)
- [ ] Deterministic results across runs

### Property-Based Tests
```bash
pytest tests/property/ -v --hypothesis-show-statistics
```
- [ ] Mathematical properties hold (additivity, scaling)
- [ ] Unit conversion round-trips exact
- [ ] Input validation properties verified

### Performance Tests
```bash
pytest tests/performance/ --benchmark-only --benchmark-autosave
```
- [ ] Single building < 2 seconds
- [ ] 50-building portfolio < 5 seconds
- [ ] Memory usage < 100MB
- [ ] Cache hit rate > 50%

### Snapshot Tests
```bash
pytest tests/snapshots/ --snapshot-update
```
- [ ] Report format consistency
- [ ] CLI output stability
- [ ] API response format unchanged

---

## 5. Cross-Platform Testing with Tox

### Python Version Matrix
```bash
tox -e py38,py39,py310,py311,py312
```
- [ ] Python 3.8 (all tests pass)
- [ ] Python 3.9 (all tests pass)
- [ ] Python 3.10 (all tests pass)
- [ ] Python 3.11 (all tests pass)
- [ ] Python 3.12 (all tests pass)

### Operating System Matrix
- [ ] Windows 10 (x64)
- [ ] Windows 11 (x64, ARM64)
- [ ] Ubuntu 20.04 LTS (x64, ARM64)
- [ ] Ubuntu 22.04 LTS (x64, ARM64)
- [ ] macOS 13 Ventura (Intel)
- [ ] macOS 14 Sonoma (Apple Silicon M1/M2/M3)
- [ ] RHEL 8/9 (x64)
- [ ] Docker containers (Alpine, Debian)

### Architecture Testing
- [ ] x86_64 (Intel/AMD)
- [ ] ARM64 (Apple Silicon, AWS Graviton)
- [ ] 32-bit systems (if supported)

---

## 6. Security Testing

### Dependency Scanning
```bash
pip-audit --desc
safety check --json
bandit -r greenlang/ -ll -f json
```
- [ ] No critical vulnerabilities in dependencies
- [ ] No high-severity issues in code
- [ ] All medium issues documented with justification

### Input Security Testing
- [ ] Path traversal prevention (../../../etc/passwd)
- [ ] Command injection prevention (; rm -rf /)
- [ ] XXE injection prevention (malformed XML/JSON)
- [ ] File size limits enforced (<100MB)
- [ ] Rate limiting implemented (if API exposed)

### Data Security
- [ ] No hardcoded credentials
- [ ] No API keys in code
- [ ] Sensitive data not logged
- [ ] Secure defaults configured
- [ ] HTTPS enforced for external calls

---

## 7. Cache Testing

### Cache Functionality
- [ ] Cache hit improves performance > 50%
- [ ] Cache invalidation on data change
- [ ] Cache size limits enforced
- [ ] Cache corruption handling
- [ ] Cache persistence across restarts

### Cache Invalidation Tests
```python
# tests/test_cache_invalidation.py
def test_cache_invalidates_on_factor_change():
    # Calculate with original factor
    result1 = client.calculate_emissions(...)
    # Modify emission factor
    update_emission_factor(...)
    # Calculate again - should use new factor
    result2 = client.calculate_emissions(...)
    assert result1 != result2
```

---

## 8. Configuration Testing

### Configuration Precedence
Order: CLI args > Environment vars > Config file > Defaults

- [ ] CLI arguments override all
- [ ] Environment variables override config file
- [ ] Config file overrides defaults
- [ ] Missing config uses sensible defaults
- [ ] Invalid config provides clear errors

### Configuration Validation
```bash
# Test various configuration scenarios
export GREENLANG_REGION=INVALID
greenlang calc  # Should error gracefully

export GREENLANG_REGION=IN
greenlang calc  # Should use India factors
```

---

## 9. Data Validation

### Emission Factors
- [ ] All 12 countries have complete factors
- [ ] Factors traceable to official sources
- [ ] Units consistent across all factors
- [ ] Renewable percentages sum correctly
- [ ] Historical factors for trend analysis

### Benchmarks
- [ ] All building type/country combinations present
- [ ] Thresholds mathematically consistent
- [ ] Rating boundaries non-overlapping
- [ ] Source citations included
- [ ] Update timestamps present

---

## 10. SDK Testing

### Basic SDK
```python
from greenlang.sdk import GreenLangClient
client = GreenLangClient()
```
- [ ] Import works without errors
- [ ] Basic methods available
- [ ] Backward compatibility maintained

### Enhanced SDK
```python
from greenlang.sdk.enhanced_client import GreenLangClient
client = GreenLangClient(region="IN")
```
- [ ] All regions supported
- [ ] Type hints work in IDE
- [ ] Auto-completion functional
- [ ] Async methods work correctly

### SDK Methods
- [ ] `calculate_emissions()` - all fuel types, all units
- [ ] `analyze_building()` - complete validation
- [ ] `analyze_portfolio()` - 1, 10, 100 buildings
- [ ] `get_emission_factor()` - all countries
- [ ] `get_recommendations()` - all scenarios
- [ ] `export_analysis()` - JSON/CSV/Excel with large datasets

---

## 11. Workflow Testing

### Core Workflows
- [ ] Commercial building workflow - all countries
- [ ] India BEE compliance workflow
- [ ] Portfolio analysis - various sizes
- [ ] Parallel execution - verify speedup
- [ ] Error handling - missing agents, bad data

### Workflow Features
- [ ] DAG validation
- [ ] Dependency resolution
- [ ] Parallel step execution
- [ ] Error propagation
- [ ] Result aggregation

---

## 12. Performance Benchmarks

### Response Times
- [ ] CLI startup < 1 second
- [ ] Single calculation < 100ms
- [ ] Building analysis < 2 seconds
- [ ] Portfolio (50 buildings) < 5 seconds
- [ ] Report generation < 3 seconds

### Resource Usage
- [ ] Memory < 100MB typical operation
- [ ] Memory < 500MB large portfolio
- [ ] CPU usage appropriate for operation
- [ ] No memory leaks (run for 1 hour)
- [ ] File handles properly closed

---

## 13. Type Safety and Code Quality

### Static Analysis
```bash
mypy greenlang/ --strict --show-error-codes
ruff check greenlang/ tests/ --statistics
black --check greenlang/ tests/
isort --check-only greenlang/ tests/
```
- [ ] mypy strict mode passes
- [ ] No ruff violations
- [ ] Code formatted with black
- [ ] Imports sorted with isort

### Code Coverage
```bash
pytest --cov=greenlang --cov-report=html --cov-fail-under=85
```
- [ ] Overall coverage ≥ 85%
- [ ] Critical paths 100% covered
- [ ] No untested public APIs
- [ ] Branch coverage ≥ 80%

---

## 14. User Experience Testing

### Interactive Features
- [ ] Prompts clear and helpful
- [ ] Input validation provides guidance
- [ ] Progress bars for long operations
- [ ] Cancellation works cleanly (Ctrl+C)
- [ ] Color output works on all terminals

### Error Messages
- [ ] Actionable error messages
- [ ] No stack traces in production
- [ ] Suggestions for common mistakes
- [ ] Links to documentation where relevant
- [ ] Contact information for support

---

## 15. Deployment and Installation

### Package Installation
```bash
pip install -e .
pip install greenlang  # From PyPI when published
```
- [ ] Clean installation works
- [ ] Dependencies resolve correctly
- [ ] No version conflicts
- [ ] Uninstall removes all files

### Docker Deployment
```bash
docker build -t greenlang:v0.0.1 .
docker run greenlang:v0.0.1 --version
```
- [ ] Docker image builds
- [ ] Container runs correctly
- [ ] Environment variables work
- [ ] Volume mounts functional

---

## Automated Test Execution Script

```bash
#!/bin/bash
# run_qa_tests.sh

echo "=== GreenLang QA Test Suite ==="

# 1. Security scanning
echo "Running security scans..."
pip-audit --desc
safety check
bandit -r greenlang/

# 2. Type checking and linting
echo "Running static analysis..."
mypy greenlang/ --strict
ruff check greenlang/ tests/
black --check greenlang/ tests/

# 3. Run tests with tox
echo "Running tests across Python versions..."
tox

# 4. Performance benchmarks
echo "Running performance benchmarks..."
pytest tests/performance/ --benchmark-only

# 5. Generate coverage report
echo "Generating coverage report..."
pytest --cov=greenlang --cov-report=html --cov-report=term

echo "=== QA Test Suite Complete ==="
```

---

## Sign-off Criteria

### Must Pass
- [ ] All automated tests pass
- [ ] Security scan shows no critical issues
- [ ] Performance benchmarks met
- [ ] Documentation complete and accurate
- [ ] Cross-platform testing successful

### Quality Gates
- [ ] Test coverage ≥ 85%
- [ ] Code quality A rating
- [ ] Zero P0/P1 bugs
- [ ] All P2 bugs documented
- [ ] Performance regression < 5%

### Final Approval
- [ ] QA Lead Sign-off: _______________
- [ ] Engineering Lead Sign-off: _______________
- [ ] Product Owner Sign-off: _______________
- [ ] Release Date: _______________

---

**Estimated Timeline**: 3-4 days with automated testing
**Recommended Team**: 2 QA engineers + 1 DevOps engineer
**Automation Coverage**: 80% of test cases automated

This enhanced checklist incorporates all developer recommendations and provides a robust, automated approach to ensuring GreenLang v0.0.1 is production-ready.