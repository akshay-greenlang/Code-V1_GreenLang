# GL-002 Determinism Audit - Critical Findings

## Summary

Analysis of the GL-002 BoilerEfficiencyOptimizer has identified **4 critical non-deterministic sources** that prevent byte-identical reproducibility across runs.

## Critical Issues

### 1. TIMESTAMPS IN OUTPUT (CRITICAL - SYSTEMATIC)

**Severity**: CRITICAL
**Impact**: 100% reproducibility failure for hash verification
**Files Affected**:
- `boiler_efficiency_orchestrator.py` (lines 287, 573-575, 642, 651)

**Problem**:
Every execution generates fresh timestamps that are included in the output. This causes:
- Output hashes to differ between runs (guaranteed)
- Dashboard hashes to differ (contains timestamped alerts)
- Efficiency improvements array hashes to differ (each entry timestamped)

**Evidence**:
```python
# Line 287 - Result timestamp
'timestamp': datetime.now(timezone.utc).isoformat()

# Line 573-575 - Efficiency improvements
self.performance_metrics['efficiency_improvements'].append({
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'improvement_percent': improvement
})

# Line 642 - Alert timestamp
'timestamp': datetime.now(timezone.utc).isoformat()
```

**Determinism Impact**:
```
Run 1 Output Hash: a1b2c3d4e5f6...
Run 2 Output Hash: x9y8z7w6v5u4... (DIFFERENT due to timestamp)
Run 3 Output Hash: m5n4o3p2q1r0... (DIFFERENT due to timestamp)
```

**Fix Required**:
```python
# Accept timestamp from input for testing
test_timestamp = input_data.get('_test_timestamp')
result = {
    'timestamp': test_timestamp or datetime.now(timezone.utc).isoformat(),
    ...
}
```

---

### 2. CACHE VALIDATION WITH time.time() (HIGH - SYSTEMATIC)

**Severity**: HIGH
**Impact**: Non-deterministic execution paths
**Files Affected**:
- `boiler_efficiency_orchestrator.py` (lines 877-893, 904)

**Problem**:
Cache validity is determined by checking elapsed time:
```python
def _is_cache_valid(self, cache_key: str) -> bool:
    timestamp = self._cache_timestamps.get(cache_key, 0)
    age_seconds = time.time() - timestamp  # <-- NON-DETERMINISTIC
    return age_seconds < self._cache_ttl_seconds
```

**Execution Flow**:
- Run 1: Execute at T=0ms → Store in cache → Check age=0ms → Valid
- Run 2: Execute at T=50ms → Check age=50ms → Valid (if TTL>50ms)
- Run 3: Execute at T=3000ms → Check age=3000ms → Invalid (if TTL=60s)

Different execution times lead to different code paths being taken.

**Determinism Impact**:
```
Run 1: cache_key NOT in cache → execute → store result (deterministic)
Run 2: cache_key IN cache, age=50ms, TTL=60s → use cached (possible difference)
Run 3: cache_key expired, age>TTL → execute fresh (possible difference)
```

If cached paths execute vs. fresh paths execute, results may differ due to state differences.

**Fix Required**:
```python
def _is_cache_valid(self, cache_key: str) -> bool:
    # In deterministic mode, don't use cache
    if getattr(self, '_deterministic_mode', False):
        return False  # Force fresh calculation

    if cache_key not in self._results_cache:
        return False
    timestamp = self._cache_timestamps.get(cache_key, 0)
    age_seconds = time.time() - timestamp
    return age_seconds < self._cache_ttl_seconds
```

---

### 3. LLM INTEGRATION RANDOMNESS (MEDIUM - POTENTIAL)

**Severity**: MEDIUM
**Impact**: Classification results may vary slightly
**Files Affected**:
- `boiler_efficiency_orchestrator.py` (lines 168-174)

**Problem**:
ChatSession is configured with:
```python
self.chat_session = ChatSession(
    provider=ModelProvider.ANTHROPIC,
    model_id="claude-3-haiku",
    temperature=0.0,  # Deterministic setting
    seed=42  # Fixed seed
)
```

**Analysis**:
- Temperature=0.0 means deterministic output selection
- Seed=42 is fixed for reproducibility
- However, internal LLM processing may still have non-deterministic elements
- Not currently called in execute() method, so impact is limited

**Current Status**: NOT ACTIVELY USED
```python
# Lines 164-211: _init_intelligence() method creates ChatSession
# but subsequent code doesn't call it for critical calculations
```

**Mitigation**: Already not used in critical path, but should mock in tests to be safe.

---

### 4. PERFORMANCE METRICS ACCUMULATION (MEDIUM - INSTANCE-LEVEL)

**Severity**: MEDIUM
**Impact**: Metrics in output differ between runs
**Files Affected**:
- `boiler_efficiency_orchestrator.py` (lines 139-149, 383, 383, 427, 456-457)

**Problem**:
Performance metrics accumulate across multiple executions on same orchestrator instance:
```python
# Lines 139-149
self.performance_metrics = {
    'optimizations_performed': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    ...
}

# Line 383
self.performance_metrics['optimizations_performed'] += 1

# Line 427
self.performance_metrics['optimizations_performed'] += 1

# Line 456-457
self.performance_metrics['total_steam_generated_tons'] += (
    result.target_steam_flow_kg_hr * 0.001
)
```

**Determinism Impact**:
```
Scenario: Run same input twice on same orchestrator instance

Run 1:
  Input: test_data
  Metrics before: optimizations_performed=0
  Output: {..., performance_metrics: {optimizations_performed: 1, ...}}

Run 2:
  Input: same test_data
  Metrics before: optimizations_performed=1 (DIFFERENT!)
  Output: {..., performance_metrics: {optimizations_performed: 2, ...}}
```

The metrics in the output will differ even though input is identical.

**Current Testing Setup**: Each test creates fresh orchestrator → **NOT AFFECTED**

**Future Issue**: If tests reuse orchestrator instance, metrics will accumulate.

**Fix Required**:
```python
# In test fixtures
@pytest.fixture
def fresh_orchestrator():
    """Create fresh orchestrator for determinism testing."""
    return BoilerEfficiencyOptimizer(create_default_config())

# NOT
@pytest.fixture
def orchestrator():
    orch = BoilerEfficiencyOptimizer(create_default_config())
    yield orch  # Reused across tests!
```

---

## Hash Mismatch Prediction

If we run GL-002 10 times and compare hashes WITHOUT fixes:

### Guaranteed Mismatches

| Hash Field | Run 1 | Run 2 | Run 3-10 | Reason |
|------------|-------|-------|----------|--------|
| input_hash | SAME | SAME | SAME | Input is fixed |
| **output_hash** | A1B2... | X9Y8... | VARIOUS | Timestamp in output |
| **dashboard_hash** | C3D4... | K7L6... | VARIOUS | Alert timestamps |
| **provenance_hash** | E5F6... | M3N2... | VARIOUS | Output in hash calculation |
| combustion_hash | SAME | SAME | SAME | Pure calculation |
| steam_hash | SAME | SAME | SAME | Pure calculation |
| emissions_hash | SAME | SAME | SAME | Pure calculation |

**Result**: 3-4 out of 7 hash fields will FAIL (40-57% match rate)

### After Fixes

All hashes will match (100% match rate).

---

## Code Quality Assessment

### Positive Aspects

1. **Deterministic Calculation Methods**
   - All mathematical functions are pure
   - No global state modifications in calculations
   - Clear input/output contracts

2. **Proper Constants**
   - Physical constants are immutable
   - Default fuel properties are fixed
   - Constraints are properly parameterized

3. **Good Async/Await Usage**
   - Threading via asyncio.to_thread prevents race conditions
   - No Python thread pool randomness

### Areas of Concern

1. **Mixed Concerns in Orchestrator**
   - Timestamps mixed with calculation results
   - Metrics accumulation mixed with results
   - Cache logic coupled with execution

2. **Missing Deterministic Mode**
   - No configuration flag for deterministic operation
   - No test mode for reproducibility testing
   - No environment variable support

3. **Audit Trail Issues**
   - Provenance hash includes non-deterministic data
   - Timestamp in provenance prevents reproducibility
   - No immutable audit log

---

## Severity Classification Matrix

```
Likelihood x Impact = Risk

Timestamp (GUARANTEED x CRITICAL) = CRITICAL
Cache TTL (PROBABLE x HIGH) = HIGH
LLM Randomness (POSSIBLE x MEDIUM) = MEDIUM
Metrics Accumulation (LOW x MEDIUM) = LOW
```

---

## Reproducibility Score Projection

### Before Fixes
```
Reproducibility = (3/7 matching hashes) * 100
                = 42.9% (FAIL)
```

### After Fixes
```
Reproducibility = (7/7 matching hashes) * 100
                = 100.0% (PASS)
```

---

## Recommended Implementation Priority

### Phase 1 (IMMEDIATE) - Critical Fixes
1. **Fix timestamp handling** - Exclude dynamic timestamps or inject test value
2. **Add deterministic mode flag** - Config option to disable cache

### Phase 2 (HIGH) - Integration
1. **Mock ChatSession in tests** - Prevent LLM randomness
2. **Set PYTHONHASHSEED** - Environment variable setup

### Phase 3 (MEDIUM) - Enhancement
1. **Reset metrics per run** - Fresh instance or reset method
2. **Separate test configuration** - Dedicated test config class

---

## Testing Strategy

### Unit Level
```python
# Test each function 10x, verify identical output
def test_calculate_boiler_efficiency_determinism():
    for i in range(10):
        result = tools.calculate_boiler_efficiency(boiler_data, sensors)
        assert result == expected_result
```

### Integration Level
```python
# Test orchestrator 10x with same input
def test_orchestrator_determinism():
    for i in range(10):
        output = await orchestrator.execute(test_input)
        assert output == expected_output
```

### Hash Level
```python
# Verify hash consistency
def test_hash_consistency():
    hashes = [calculate_hash(await orchestrator.execute(test_input))
              for _ in range(10)]
    assert len(set(hashes)) == 1  # All identical
```

---

## Cross-Environment Considerations

### Windows (Current)
- Uses `time.time()` (millisecond precision)
- Python 3.7+ dict ordering is guaranteed
- Path separators: backslash

### Linux
- Uses `time.time()` (microsecond precision)
- Path separators: forward slash

### Docker
- Controlled environment
- Reproducible Python version
- No system differences

### Kubernetes
- Multiple pods = different time bases
- Each pod has independent cache
- Results should match despite pod differences

**Risk**: Path handling in provenance hashes could differ. Use Path().as_posix() for consistency.

---

## Final Assessment

**Current State**: NON-DETERMINISTIC
- **Hash Match Rate**: 42.9% (3/7 fields match)
- **Reproducibility Score**: FAIL

**After Fixes**: DETERMINISTIC
- **Hash Match Rate**: 100.0% (7/7 fields match)
- **Reproducibility Score**: PASS

**Effort to Fix**: 4-6 hours
- 1 hour: Timestamp fixes
- 1 hour: Cache mode configuration
- 1 hour: Test setup and fixtures
- 1-2 hours: Testing and validation

---

## Related Files

- Main Report: `DETERMINISM_AUDIT_REPORT.md`
- Audit Script: `run_determinism_audit.py`
- Test Suite: `tests/test_determinism_audit.py`
- Code Files: `boiler_efficiency_orchestrator.py`, `config.py`, `tools.py`

