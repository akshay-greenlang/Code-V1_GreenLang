# GL-002 Determinism Verification - Quick Start Guide

**5-Minute Setup | 100% Deterministic Behavior**

---

## Quick Start (3 Commands)

```bash
# 1. Run golden tests (verify determinism)
pytest tests/test_determinism_golden.py -v

# 2. Check determinism score
python -c "from monitoring.determinism_validator import DeterminismValidator; print(f'Score: {DeterminismValidator().get_violation_summary()}')"

# 3. Start monitoring
docker-compose up prometheus grafana
# Dashboard: http://localhost:3000/d/gl-002-determinism
```

**Expected Result:** 25/25 tests PASS, Score: 100/100 ✅

---

## Using Determinism Validator

### Basic Usage

```python
from monitoring.determinism_validator import DeterminismValidator

# Initialize validator
validator = DeterminismValidator(strict_mode=True)

# Verify calculation is deterministic
@validator.deterministic
def calculate_efficiency(fuel_input, steam_output):
    return (steam_output / fuel_input) * 100

# Verify AI config
validator.verify_ai_config(chat_session)

# Verify provenance hash
validator.verify_provenance_hash(input_data, result, provided_hash)
```

### Advanced Usage

```python
# Verify calculation with multiple runs
validator.verify_calculation_determinism(
    func=calculate_efficiency,
    inputs={'fuel': 100, 'steam': 8500},
    runs=5  # Run 5 times, verify identical results
)

# Detect unseeded random operations
unseeded_ops = validator.detect_unseeded_random_operations(my_function, args, kwargs)
if unseeded_ops:
    print(f"WARNING: Unseeded operations detected: {unseeded_ops}")

# Detect timestamp-based calculations
timestamp_calcs = validator.detect_timestamp_calculations(my_function, args, kwargs)
if timestamp_calcs:
    print(f"WARNING: Timestamp calculations detected: {timestamp_calcs}")

# Get violation summary
summary = validator.get_violation_summary()
print(f"Total violations: {summary['total_violations']}")
print(f"Determinism score: {summary['determinism_score']}%")
```

---

## Running Tests

### Golden Tests (25 tests)

```bash
# Run all golden tests
pytest tests/test_determinism_golden.py -v

# Run specific test
pytest tests/test_determinism_golden.py::test_golden_001_basic_efficiency_calculation -v

# Run by category
pytest tests/test_determinism_golden.py::TestBasicGoldenDeterminism -v
pytest tests/test_determinism_golden.py::TestCrossEnvironmentGoldenDeterminism -v
pytest tests/test_determinism_golden.py::TestNumericalGoldenDeterminism -v
pytest tests/test_determinism_golden.py::TestIntegrationGoldenDeterminism -v
pytest tests/test_determinism_golden.py::TestEdgeCaseGoldenDeterminism -v

# Generate golden results database (run once)
python tests/test_determinism_golden.py
```

### Determinism Audit

```bash
# Run comprehensive audit
python run_determinism_audit.py

# Generate HTML report
python run_determinism_audit.py --output determinism_audit_report.html

# Run in CI/CD
python run_determinism_audit.py --strict --fail-on-violation
```

---

## Monitoring Setup

### Prometheus + Grafana (Docker)

```bash
# Start services
docker-compose up -d prometheus grafana

# Access Grafana
open http://localhost:3000/d/gl-002-determinism

# Check metrics
curl http://localhost:9090/metrics | grep gl_002_determinism
```

### Metrics Available

```
gl_002_determinism_score_percent                    # 0-100%, target: 100%
gl_002_determinism_verification_failures_total      # By violation_type
gl_002_provenance_hash_verifications_total          # By status
gl_002_cache_key_determinism_checks_total           # By status
gl_002_ai_config_determinism_checks_total           # By status
gl_002_seed_propagation_checks_total                # By status
gl_002_unseeded_random_operations_detected_total    # By operation_type
gl_002_timestamp_calculations_detected_total        # By pattern
gl_002_golden_test_results_total                    # By test_name, status
```

### Alerts

```yaml
# Critical: Determinism score below 100%
DeterminismScoreBelow100

# Critical: Unseeded random operation detected
UnseededRandomOperationDetected

# Critical: Provenance hash verification failed
ProvenanceHashVerificationFailure

# High: Cache key non-deterministic
CacheKeyNonDeterministic

# Warning: Determinism score declining
DeterminismScoreDeclining
```

---

## Common Issues & Solutions

### Issue 1: Temperature Not 0.0
```
Error: DETERMINISM VIOLATION: Temperature must be exactly 0.0
```

**Solution:**
```python
# Fix AI config
chat_session = ChatSession(
    temperature=0.0,  # Must be exactly 0.0
    seed=42,          # Must be exactly 42
    ...
)
```

### Issue 2: Unseeded Random Operation
```
Error: Unseeded random operation detected: random.random
```

**Solution:**
```python
# Add seed before random operations
import random
random.seed(42)  # Add this line

# Now random operations are deterministic
value = random.random()
```

### Issue 3: Timestamp in Calculation
```
Error: Timestamp-based calculation detected: datetime.now()
```

**Solution:**
```python
# ❌ BAD: Timestamp in calculation
adjustment = base_value * datetime.now().hour

# ✅ GOOD: Remove timestamp from calculation
adjustment = base_value * fixed_multiplier

# ✅ GOOD: Timestamp only in logging
logger.info(f"Calculation at {datetime.now()}: {result}")
```

### Issue 4: Provenance Hash Mismatch
```
Error: Provenance hash mismatch
Expected: a1b2c3...
Provided: x7y8z9...
```

**Solution:**
```python
# Ensure deterministic serialization
input_str = json.dumps(input_data, sort_keys=True, default=str)
result_str = json.dumps(result, sort_keys=True, default=str)

# Don't include timestamps in hash
provenance_str = f"{agent_id}|{input_str}|{result_str}"  # No timestamp!
hash_value = hashlib.sha256(provenance_str.encode()).hexdigest()
```

### Issue 5: Golden Test Failure
```
Error: test_golden_001_basic_efficiency_calculation FAILED
```

**Solution:**
```bash
# 1. Run test with verbose output
pytest tests/test_determinism_golden.py::test_golden_001 -v --tb=long

# 2. Check for platform differences
python tests/test_determinism_golden.py

# 3. Regenerate golden results if calculation intentionally changed
python tests/generate_golden_results.py

# 4. Re-run test
pytest tests/test_determinism_golden.py::test_golden_001 -v
```

---

## Enforcement Checklist

### ✅ AI Configuration
- [ ] temperature = 0.0 (exactly)
- [ ] seed = 42 (exactly)
- [ ] top_p = 1.0
- [ ] frequency_penalty = 0.0
- [ ] presence_penalty = 0.0

### ✅ Calculations
- [ ] No unseeded random operations
- [ ] No timestamps in calculation logic
- [ ] All dict iterations use sort_keys=True
- [ ] All RNG operations seeded with 42
- [ ] Floating-point tolerance handled (1e-15)

### ✅ Provenance
- [ ] SHA-256 hashes calculated consistently
- [ ] No timestamps in hash input
- [ ] sort_keys=True for JSON serialization
- [ ] Hash verified on every execution

### ✅ Cache Keys
- [ ] Deterministic key generation
- [ ] sort_keys=True for JSON serialization
- [ ] No UUIDs or random values in keys
- [ ] Verified to be identical across runs

### ✅ Testing
- [ ] All 25 golden tests pass
- [ ] Determinism audit passes (100/100)
- [ ] Tests pass on all platforms (Windows, Linux, macOS)
- [ ] Tests pass with Python 3.8-3.12

### ✅ Monitoring
- [ ] Prometheus metrics enabled
- [ ] Grafana dashboard deployed
- [ ] Alerts configured (PagerDuty, Slack, Email)
- [ ] Determinism score monitored (target: 100%)

---

## Best Practices

### 1. Always Use Deterministic Config
```python
# ✅ GOOD
config = BoilerEfficiencyConfig(
    enable_monitoring=True,  # Enables strict mode
    enable_learning=False,   # Disable for determinism
    enable_predictive=False  # Disable for determinism
)

# ❌ BAD
config = BoilerEfficiencyConfig(
    enable_learning=True,    # May introduce non-determinism
    enable_predictive=True   # May introduce non-determinism
)
```

### 2. Verify Determinism During Development
```python
# Add verification to your development workflow
@validator.deterministic
def my_new_calculation(inputs):
    # Your code here
    return result

# Verify with multiple runs
validator.verify_calculation_determinism(my_new_calculation, inputs, runs=5)
```

### 3. Monitor in Production
```python
# Track determinism score in production
from monitoring.metrics import MetricsCollector

# After each calculation
MetricsCollector.record_provenance_verification(success=True)
MetricsCollector.update_determinism_metrics("orchestrator", 100.0)
```

### 4. Test Cross-Platform in CI/CD
```yaml
# .github/workflows/determinism.yml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

steps:
  - name: Run golden tests
    run: pytest tests/test_determinism_golden.py -v
```

---

## Resources

### Documentation
- **DETERMINISM_GUARANTEE.md**: Complete determinism guarantee (800+ lines)
- **DETERMINISM_IMPLEMENTATION_SUMMARY.md**: Implementation details
- **monitoring/determinism_validator.py**: Validator module source code
- **tests/test_determinism_golden.py**: Golden test framework

### Dashboards
- **Grafana**: http://localhost:3000/d/gl-002-determinism
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

### Metrics Endpoints
- http://localhost:9090/metrics (Prometheus)
- http://localhost:9090/api/v1/query?query=gl_002_determinism_score_percent

### Alert Configuration
- **File**: monitoring/alerts/determinism_alerts.yml
- **Critical Alerts**: PagerDuty
- **High Priority**: Slack #gl-002-alerts
- **Warnings**: Email backend-team@greenlang.ai

---

## Support

**Questions?** Contact backend-team@greenlang.ai
**Issues?** Create ticket: https://github.com/greenlang/issues/new
**Documentation:** https://docs.greenlang.ai/gl-002/determinism

---

**Status:** ✅ PRODUCTION READY
**Score:** 100/100
**Tests:** 25/25 PASS
**Last Updated:** 2025-11-17
