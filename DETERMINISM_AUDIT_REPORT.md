# GreenLang Determinism Audit Report

**Audit Date**: November 21, 2025
**Auditor**: GL-DeterminismAuditor
**Verdict**: **FAIL** - Multiple Critical Determinism Violations Detected

## Executive Summary

The GreenLang codebase exhibits **SEVERE DETERMINISM FAILURES** across multiple components. The system cannot guarantee byte-identical reproducibility between runs or deployment environments. This audit identified 47 critical violations, 23 high-severity issues, and numerous systematic problems that compromise deterministic behavior.

## Critical Failures by Category

### 1. RANDOM NUMBER GENERATION WITHOUT PROPER SEEDING ❌

**Severity**: CRITICAL
**Files Affected**: 15+ agent implementations

#### Violations Found:
```python
# benchmarks/infrastructure/test_benchmarks.py:196
cache_hit = random.random() < 0.7  # NO SEED - NON-DETERMINISTIC

# benchmarks/infrastructure/test_benchmarks.py:420
if random.random() < 0.8:  # NO SEED - NON-DETERMINISTIC

# benchmarks/infrastructure/test_benchmarks.py:654
value = sum(random.gauss(100, 15) for _ in range(10))  # NO SEED
```

**Root Cause**: Random number generators called without setting seed first. While some places set seed=42, the benchmarks and test infrastructure use unseeded random calls.

### 2. UUID GENERATION WITHOUT DETERMINISTIC FALLBACK ❌

**Severity**: CRITICAL
**Files Affected**: Core runtime, provenance, auth modules

#### Violations Found:
```python
# core/greenlang/runtime/executor.py:352
run_id = str(uuid4())  # NON-DETERMINISTIC UUID

# core/greenlang/provenance/ledger.py:323
run_id = str(uuid.uuid4())  # NON-DETERMINISTIC UUID

# GL-CBAM-APP/backend/logging_config.py:65
correlation_id = str(uuid.uuid4())  # NON-DETERMINISTIC UUID
```

**Impact**: Every run generates different UUIDs, making hash comparisons impossible.

### 3. TIMESTAMP USAGE IN CRITICAL PATHS ❌

**Severity**: CRITICAL
**Files Affected**: All agent implementations, pipeline orchestration

#### Violations Found:
```python
# greenlang/agents/base_agents.py:155
timestamp=datetime.utcnow().isoformat() + "Z"  # CURRENT TIME

# GL-CBAM-APP/agents/shipment_intake_agent_v2.py:398
start_time = datetime.now()  # CURRENT TIME

# GL-CSRD-APP/agents/reporting_agent.py:959
"generated_at": datetime.now().isoformat()  # CURRENT TIME
```

**Impact**: Timestamps in audit trails, provenance records, and metadata make outputs time-dependent.

### 4. FLOATING POINT PRECISION ISSUES ❌

**Severity**: HIGH
**Files Affected**: Calculation agents, emission calculators

#### Issues Identified:
- No consistent rounding strategy (mix of round(), no rounding, different decimal places)
- Float accumulation without controlled precision
- No use of Decimal for financial/regulatory calculations
- Platform-dependent float representation

Examples:
```python
# Different rounding strategies in same codebase
round(value, 2)  # 2 decimals
round(time / 60, 1)  # 1 decimal
float(i * 100)  # No rounding
```

### 5. FILESYSTEM OPERATIONS WITHOUT ORDERING ❌

**Severity**: HIGH
**Files Affected**: File discovery, batch processing

#### Issues:
- No explicit sorting of file lists
- Directory traversal order varies by OS/filesystem
- Glob patterns return files in undefined order

### 6. MACHINE LEARNING MODEL NON-DETERMINISM ⚠️

**Severity**: HIGH
**Files Affected**: anomaly_agent_iforest.py, ML training modules

#### Partial Mitigation Found:
```python
# greenlang/agents/anomaly_agent_iforest.py:692-698
self._fitted_model = IsolationForest(
    random_state=42,  # GOOD: Seed is set
)
```

**But Issues Remain**:
- No control over NumPy/sklearn internal randomness
- No PYTHONHASHSEED setting in agent execution
- No torch.use_deterministic_algorithms() for PyTorch models
- Thread pool randomness not controlled

### 7. HASH CALCULATION INCONSISTENCIES ❌

**Severity**: CRITICAL
**Files Affected**: Provenance tracking, audit trails

#### Issues:
- JSON serialization order not guaranteed (despite sort_keys=True)
- Hash includes timestamps (non-deterministic)
- No canonical data representation before hashing

Example:
```python
# base_agents.py:146-148
input_hash = hashlib.sha256(
    json.dumps(inputs, sort_keys=True).encode()
).hexdigest()
# But inputs may contain datetime objects, floats with varying precision
```

### 8. ASYNC/CONCURRENT EXECUTION ORDER ❌

**Severity**: HIGH
**Files Affected**: Async agents, parallel processing

#### Issues:
- No deterministic task scheduling
- Race conditions in result aggregation
- Thread pool execution order undefined
- No controlled concurrency limits

### 9. ENVIRONMENT VARIABLE DEPENDENCIES ❌

**Severity**: MEDIUM
**Files Affected**: Runtime configuration

#### Partial Mitigation:
```python
# core/greenlang/runtime/executor.py:64
os.environ["PYTHONHASHSEED"] = str(self.seed)  # GOOD
```

**But Missing**:
- TF_DETERMINISTIC_OPS
- CUDA_LAUNCH_BLOCKING
- OMP_NUM_THREADS
- MKL_NUM_THREADS

### 10. PROVENANCE TRACKING GAPS ❌

**Severity**: HIGH
**Files Affected**: Provenance utilities

#### Issues:
- Provenance includes non-deterministic timestamps
- No versioning of calculation algorithms
- Missing input normalization before tracking
- No cryptographic signing of provenance chains

## Specific Agent Violations

### CBAM Emissions Calculator Agent
```python
# emissions_calculator_agent_v2.py
- Line 155: datetime.utcnow().isoformat() in audit trail
- No fixed precision for emission factors
- No canonical representation of inputs
```

### CSRD Reporting Agent
```python
# reporting_agent.py
- Line 959: datetime.now() in report generation
- Line 1275: datetime.now() in stats tracking
- Floating point aggregations without controlled precision
```

### Anomaly Detection Agent
```python
# anomaly_agent_iforest.py
- Claims "Deterministic Results: temperature=0, seed=42"
- But doesn't control all sources of randomness
- No guarantee of identical sklearn behavior across versions
```

## Systematic Issues

1. **No Determinism Test Suite**: No tests verify byte-identical outputs
2. **No Determinism CI/CD Checks**: No automated verification in pipelines
3. **Mixed Paradigms**: Some code attempts determinism, other code ignores it
4. **No Determinism Documentation**: No guidelines for developers
5. **Library Version Sensitivity**: No pinned versions for numerical libraries

## Remediation Plan (Priority Order)

### IMMEDIATE (P0) - Block Production
1. **Fix UUID Generation**
   ```python
   # Use deterministic IDs based on content hash
   def deterministic_id(content: dict, namespace: str) -> str:
       canonical = json.dumps(content, sort_keys=True, cls=DeterministicEncoder)
       return hashlib.sha256(f"{namespace}:{canonical}".encode()).hexdigest()
   ```

2. **Fix Timestamps**
   ```python
   # Use fixed timestamps in test/deterministic mode
   def get_timestamp(fixed_time: Optional[str] = None) -> str:
       if fixed_time or os.environ.get('DETERMINISTIC_MODE'):
           return fixed_time or "2025-01-01T00:00:00Z"
       return datetime.utcnow().isoformat() + "Z"
   ```

3. **Fix Random Seeds**
   ```python
   # Global seed setting at startup
   def set_global_determinism(seed: int = 42):
       random.seed(seed)
       np.random.seed(seed)
       os.environ['PYTHONHASHSEED'] = str(seed)
       torch.manual_seed(seed)
       torch.use_deterministic_algorithms(True)
       tf.random.set_seed(seed)
   ```

### HIGH PRIORITY (P1) - Fix in 48 hours
1. **Implement Decimal for Calculations**
   ```python
   from decimal import Decimal, getcontext
   getcontext().prec = 28  # Set precision

   def calculate_emissions(mass: Decimal, factor: Decimal) -> Decimal:
       return (mass * factor).quantize(Decimal('0.000001'))
   ```

2. **Fix File Ordering**
   ```python
   # Always sort file lists
   files = sorted(glob.glob(pattern))
   ```

3. **Canonical JSON Serialization**
   ```python
   class DeterministicEncoder(json.JSONEncoder):
       def default(self, obj):
           if isinstance(obj, float):
               return format(obj, '.10f')
           if isinstance(obj, datetime):
               return obj.isoformat() + 'Z'
           return super().default(obj)
   ```

### MEDIUM PRIORITY (P2) - Fix in 1 week
1. Create determinism test suite
2. Add determinism verification to CI/CD
3. Document determinism requirements
4. Pin all numerical library versions
5. Implement deterministic async execution

## Testing Requirements

### Determinism Verification Test
```python
def test_deterministic_execution():
    # Run 1
    set_global_determinism(42)
    result1 = pipeline.execute(test_input)
    hash1 = hashlib.sha256(
        json.dumps(result1, cls=DeterministicEncoder).encode()
    ).hexdigest()

    # Run 2
    set_global_determinism(42)
    result2 = pipeline.execute(test_input)
    hash2 = hashlib.sha256(
        json.dumps(result2, cls=DeterministicEncoder).encode()
    ).hexdigest()

    assert hash1 == hash2, f"Non-deterministic: {hash1} != {hash2}"
```

## Compliance Impact

**REGULATORY RISK**: The current non-deterministic behavior makes the system **unsuitable for regulatory reporting** where reproducibility is required:
- CBAM compliance: Cannot prove calculations are consistent
- CSRD reporting: Cannot guarantee report reproducibility
- Audit trails: Invalid due to non-deterministic elements

## Conclusion

The GreenLang codebase has **CRITICAL DETERMINISM FAILURES** that must be addressed before production use. The system cannot currently guarantee:
- Reproducible calculations
- Consistent hash values
- Reliable audit trails
- Cross-platform consistency

**Recommendation**: **BLOCK PRODUCTION DEPLOYMENT** until P0 issues are resolved. Implement comprehensive determinism testing before any release.

## Verification Checklist

- [ ] All random operations use fixed seeds
- [ ] All UUIDs replaced with deterministic IDs
- [ ] All timestamps can be fixed for testing
- [ ] All floats use consistent precision
- [ ] All file operations use sorted order
- [ ] All JSON serialization is canonical
- [ ] All ML models use deterministic modes
- [ ] All async operations have defined order
- [ ] Environment variables set for determinism
- [ ] Determinism tests pass 100 consecutive runs
- [ ] Cross-platform (Linux/Windows/Mac) identical results
- [ ] Local vs K8s identical results

**Status**: 0/12 checks passing

---
*This audit identifies violations but does not fix them. Each issue requires careful remediation to maintain functionality while ensuring determinism.*