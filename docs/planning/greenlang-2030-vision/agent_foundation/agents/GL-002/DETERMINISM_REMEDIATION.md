# GL-002 Determinism Audit - Remediation Checklist

**Audit Date**: November 15, 2025
**Status**: Pre-Execution Analysis - Remediation Plan
**Target**: 100% Reproducibility (PASS)

---

## ISSUE SUMMARY

| Issue | Severity | Impact | Est. Effort | Priority |
|-------|----------|--------|-------------|----------|
| Timestamps in output | CRITICAL | Hash failure | 1 hour | 1 |
| Cache TTL timing | HIGH | Non-deterministic paths | 1 hour | 2 |
| LLM randomness | MEDIUM | Test variation | 0.5 hour | 3 |
| Metrics accumulation | MEDIUM | Output variation | 0.5 hour | 4 |
| **TOTAL** | | | **3 hours** | |

---

## PHASE 1: TIMESTAMP FIXES (CRITICAL)

### 1.1 Add Test Timestamp Support

**File**: `config.py`
**Change Type**: Add field to BoilerEfficiencyConfig

```python
# BEFORE: No test timestamp support
class BoilerEfficiencyConfig(BaseModel):
    agent_id: str = Field("GL-002", description="Agent identifier")
    # ... other fields ...

# AFTER: Add deterministic testing fields
class BoilerEfficiencyConfig(BaseModel):
    agent_id: str = Field("GL-002", description="Agent identifier")

    # NEW: Deterministic testing support
    deterministic_mode: bool = Field(
        False,
        description="Enable deterministic mode for testing"
    )
    test_timestamp: Optional[str] = Field(
        None,
        description="Fixed timestamp for deterministic testing (ISO format)"
    )

    # ... other fields ...
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Add `deterministic_mode: bool` field
- [ ] Add `test_timestamp: Optional[str]` field
- [ ] Update create_default_config() to set defaults
- [ ] Add field validation (if test_timestamp provided, must be ISO format)
- [ ] Document fields in class docstring

**Expected Time**: 15 minutes

---

### 1.2 Fix Result Timestamp in Execute Method

**File**: `boiler_efficiency_orchestrator.py`
**Location**: Line 287 in execute() method

```python
# BEFORE (line 287):
result = {
    'agent_id': self.config.agent_id,
    'timestamp': datetime.now(timezone.utc).isoformat(),  # NON-DETERMINISTIC
    'execution_time_ms': round(execution_time_ms, 2),
    # ... other fields ...
}

# AFTER:
result = {
    'agent_id': self.config.agent_id,
    'timestamp': self.boiler_config.test_timestamp or
                 datetime.now(timezone.utc).isoformat(),
    'execution_time_ms': round(execution_time_ms, 2),
    # ... other fields ...
}
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Replace datetime.now() with test_timestamp check
- [ ] Test with fixed timestamp
- [ ] Verify output hash consistency

**Expected Time**: 10 minutes

---

### 1.3 Fix Dashboard Alert Timestamps

**File**: `boiler_efficiency_orchestrator.py`
**Location**: Lines 642, 651

```python
# BEFORE (lines 642, 651):
alerts.append({
    'level': 'warning',
    'category': 'efficiency',
    'message': f'Efficiency {state.efficiency_percent:.1f}% below minimum...',
    'timestamp': datetime.now(timezone.utc).isoformat()  # NON-DETERMINISTIC
})

# AFTER:
alerts.append({
    'level': 'warning',
    'category': 'efficiency',
    'message': f'Efficiency {state.efficiency_percent:.1f}% below minimum...',
    'timestamp': self.boiler_config.test_timestamp or
                 datetime.now(timezone.utc).isoformat()
})
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Replace both alert timestamp instances
- [ ] Verify all alert types use fixed timestamp
- [ ] Test dashboard hash consistency

**Expected Time**: 10 minutes

---

### 1.4 Fix Efficiency Improvements Timestamps

**File**: `boiler_efficiency_orchestrator.py`
**Location**: Lines 573-575

```python
# BEFORE (lines 573-575):
self.performance_metrics['efficiency_improvements'].append({
    'timestamp': datetime.now(timezone.utc).isoformat(),  # NON-DETERMINISTIC
    'improvement_percent': improvement
})

# AFTER:
self.performance_metrics['efficiency_improvements'].append({
    'timestamp': self.boiler_config.test_timestamp or
                 datetime.now(timezone.utc).isoformat(),
    'improvement_percent': improvement
})
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Fix efficiency improvements timestamp
- [ ] Verify performance metrics hash consistency

**Expected Time**: 10 minutes

---

## PHASE 2: CACHE CONFIGURATION (HIGH)

### 2.1 Add Deterministic Mode to Orchestrator

**File**: `boiler_efficiency_orchestrator.py`
**Location**: Constructor (__init__)

```python
# BEFORE (in __init__):
def __init__(self, config: BoilerEfficiencyConfig):
    base_config = AgentConfig(...)
    super().__init__(base_config)
    self.boiler_config = config
    self.tools = BoilerEfficiencyTools()
    # ... rest of init ...

# AFTER:
def __init__(self, config: BoilerEfficiencyConfig):
    base_config = AgentConfig(...)
    super().__init__(base_config)
    self.boiler_config = config
    self.tools = BoilerEfficiencyTools()

    # NEW: Store deterministic mode flag
    self._deterministic_mode = config.deterministic_mode

    # ... rest of init ...
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Add `_deterministic_mode` instance variable
- [ ] Store from config parameter
- [ ] Document in class docstring

**Expected Time**: 5 minutes

---

### 2.2 Fix Cache Validity Check

**File**: `boiler_efficiency_orchestrator.py`
**Location**: _is_cache_valid() method (lines 877-893)

```python
# BEFORE (lines 877-893):
def _is_cache_valid(self, cache_key: str) -> bool:
    if cache_key not in self._results_cache:
        return False
    timestamp = self._cache_timestamps.get(cache_key, 0)
    age_seconds = time.time() - timestamp
    return age_seconds < self._cache_ttl_seconds

# AFTER:
def _is_cache_valid(self, cache_key: str) -> bool:
    # Disable cache in deterministic mode for reproducibility
    if self._deterministic_mode:
        return False

    if cache_key not in self._results_cache:
        return False
    timestamp = self._cache_timestamps.get(cache_key, 0)
    age_seconds = time.time() - timestamp
    return age_seconds < self._cache_ttl_seconds
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Add deterministic mode check at start
- [ ] Return False to skip cache
- [ ] Test with deterministic_mode=True
- [ ] Verify execution paths are consistent

**Expected Time**: 10 minutes

---

### 2.3 Store Cache TTL Configuration

**File**: `config.py`
**Location**: BoilerEfficiencyConfig class

**Already Exists**: Line 204
```python
cache_ttl_seconds: int = Field(60, description="Cache time-to-live")
```

**Action Required**: Add note in documentation

```python
# Update docstring:
cache_ttl_seconds: int = Field(
    60,
    description="Cache time-to-live in seconds. Set to 0 for deterministic testing."
)
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Document that cache_ttl_seconds=0 disables cache
- [ ] Update README with deterministic testing instructions

**Expected Time**: 5 minutes

---

## PHASE 3: MOCK EXTERNAL SERVICES (MEDIUM)

### 3.1 Create Mock ChatSession Fixture

**File**: `tests/conftest.py`
**Action**: Add new fixture for mocking ChatSession

```python
# ADD to tests/conftest.py:

import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_chat_session():
    """
    Mock ChatSession for deterministic testing.
    Prevents LLM randomness from affecting test results.
    """
    mock_session = Mock()
    mock_session.chat = Mock(return_value={
        'content': 'normal',  # Fixed classification
        'tokens': 10,
        'model': 'claude-3-haiku'
    })

    # Create patch context
    with patch(
        'boiler_efficiency_orchestrator.ChatSession',
        return_value=mock_session
    ):
        yield mock_session
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Create conftest.py fixture
- [ ] Mock ChatSession.chat() method
- [ ] Return fixed classification results
- [ ] Document fixture usage
- [ ] Add to test docstrings

**Expected Time**: 20 minutes

---

### 3.2 Update Determinism Tests to Use Mock

**File**: `tests/test_determinism_audit.py`
**Location**: Test fixtures and test methods

```python
# UPDATE all determinism tests to use mock:

@pytest.mark.asyncio
async def test_orchestrator_determinism(mock_chat_session):
    """Test orchestrator produces deterministic results."""
    # Now uses mocked ChatSession instead of real LLM
    # ... test code ...
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Add mock_chat_session parameter to test methods
- [ ] Verify mock is applied
- [ ] Test runs successfully with mock

**Expected Time**: 15 minutes

---

## PHASE 4: ENVIRONMENT SETUP (MEDIUM)

### 4.1 Add Environment Variable Support

**File**: `run_determinism_audit.py` and CI/CD scripts

```bash
# BEFORE: No environment setup
python run_determinism_audit.py

# AFTER: Set environment variables
export PYTHONHASHSEED=42
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

python run_determinism_audit.py
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Add environment setup to run_determinism_audit.py
- [ ] Document in README
- [ ] Add to GitHub Actions workflow
- [ ] Add to local testing instructions

**Expected Time**: 15 minutes

---

### 4.2 Create Docker Determinism Image

**File**: `deployment/Dockerfile.test`

```dockerfile
# NEW FILE: deployment/Dockerfile.test

FROM python:3.11-slim

WORKDIR /app

# Set determinism environment
ENV PYTHONHASHSEED=42
ENV PYTHONDONTWRITEBYTECODE=1

# Copy code
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run determinism audit
CMD ["python", "run_determinism_audit.py"]
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Create Dockerfile for testing
- [ ] Set PYTHONHASHSEED
- [ ] Build and test locally
- [ ] Test consistency across container runs

**Expected Time**: 20 minutes

---

## PHASE 5: METRICS RESET (MEDIUM)

### 5.1 Add Metrics Reset Method

**File**: `boiler_efficiency_orchestrator.py`
**Location**: Add new method to class

```python
# ADD new method:

def reset_performance_metrics(self):
    """
    Reset performance metrics to initial state.
    Used for deterministic testing to ensure clean state.
    """
    self.performance_metrics = {
        'optimizations_performed': 0,
        'avg_optimization_time_ms': 0,
        'fuel_savings_kg': 0,
        'emissions_reduced_kg': 0,
        'efficiency_improvements': [],
        'cache_hits': 0,
        'cache_misses': 0,
        'agents_coordinated': 0,
        'errors_recovered': 0,
        'total_steam_generated_tons': 0
    }
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Add reset_performance_metrics() method
- [ ] Call in test setup
- [ ] Document usage
- [ ] Test metrics are fresh after reset

**Expected Time**: 10 minutes

---

### 5.2 Use Fresh Orchestrator in Tests

**File**: `tests/test_determinism_audit.py`

```python
# UPDATE fixture:

@pytest.fixture
def orchestrator(deterministic_config):
    """Create fresh orchestrator for each test."""
    # Remove any @pytest.fixture(scope="session")
    # Force function-level scope (new instance per test)
    orchestrator = BoilerEfficiencyOptimizer(deterministic_config)
    orchestrator.reset_performance_metrics()
    yield orchestrator
    # Cleanup after test
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Update fixture to create fresh instance
- [ ] Set function scope (not session)
- [ ] Call reset_performance_metrics()
- [ ] Verify metrics are fresh each test

**Expected Time**: 10 minutes

---

## VERIFICATION TESTS

### Test 1: Timestamp Determinism

```python
def test_fixed_timestamp_determinism():
    """Verify fixed timestamp produces deterministic output."""
    config = create_default_config()
    config.deterministic_mode = True
    config.test_timestamp = "2025-11-15T12:00:00Z"

    orchestrator = BoilerEfficiencyOptimizer(config)

    hashes = []
    for i in range(5):
        result = asyncio.run(orchestrator.execute(test_input))
        output_hash = calculate_hash(result)
        hashes.append(output_hash)

    # All hashes should be identical
    assert len(set(hashes)) == 1, "Hashes differ with fixed timestamp"
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Implement test
- [ ] Run and verify passes
- [ ] Test both with and without deterministic_mode

**Expected Time**: 15 minutes

---

### Test 2: Cache Disabling Determinism

```python
def test_cache_disabled_in_deterministic_mode():
    """Verify cache is disabled in deterministic mode."""
    config = create_default_config()
    config.deterministic_mode = True
    config.cache_ttl_seconds = 60  # TTL is set but ignored

    orchestrator = BoilerEfficiencyOptimizer(config)

    # First call - should execute
    start_time = time.time()
    result1 = asyncio.run(orchestrator.execute(test_input))
    time1 = time.time() - start_time

    # Second call immediately - should also execute (cache disabled)
    start_time = time.time()
    result2 = asyncio.run(orchestrator.execute(test_input))
    time2 = time.time() - start_time

    # Both should be similar execution time (no cached result reuse)
    # Cache would return instantly, execution takes time
    assert time2 > 10  # Execution is expensive
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Implement test
- [ ] Run and verify cache is disabled
- [ ] Check execution paths are consistent

**Expected Time**: 15 minutes

---

### Test 3: Full Determinism Audit

```bash
# Run the comprehensive audit
python run_determinism_audit.py

# Expected output:
# Status: PASS
# All Hashes Match: YES
# Numerical Stability: 100.0%
```

**Status**: ⬜ NOT STARTED
**Checklist**:
- [ ] Run audit script
- [ ] Verify all 10 runs succeed
- [ ] Verify all hashes match
- [ ] Verify stability is 100%

**Expected Time**: 5 minutes (after fixes complete)

---

## IMPLEMENTATION SCHEDULE

### Day 1 (Phase 1 & 2)
- [ ] Add deterministic_mode and test_timestamp to config.py (15 min)
- [ ] Fix all timestamp instances in boiler_efficiency_orchestrator.py (30 min)
- [ ] Add _deterministic_mode to orchestrator init (5 min)
- [ ] Fix _is_cache_valid() method (10 min)
- [ ] Test changes locally (20 min)
- **Total: ~1.5 hours**

### Day 2 (Phase 3 & 4)
- [ ] Create mock ChatSession fixture (20 min)
- [ ] Update tests to use mock (15 min)
- [ ] Add environment variable support (15 min)
- [ ] Create Docker test image (20 min)
- [ ] Test in Docker container (15 min)
- **Total: ~1.5 hours**

### Day 3 (Phase 5 & Verification)
- [ ] Add metrics reset method (10 min)
- [ ] Update test fixtures (10 min)
- [ ] Create verification tests (30 min)
- [ ] Run full audit (5 min)
- [ ] Document results (20 min)
- **Total: ~1.5 hours**

**Total Effort: ~4.5 hours**

---

## SUCCESS CRITERIA

All items must be completed for **PASS** status:

- [ ] Phase 1: All timestamp fixes implemented
- [ ] Phase 2: Cache disabled in deterministic mode
- [ ] Phase 3: Mock ChatSession in place
- [ ] Phase 4: Environment variables configured
- [ ] Phase 5: Metrics reset working
- [ ] All verification tests pass
- [ ] Run determinism audit: RESULT = PASS
- [ ] All hashes match 100%
- [ ] Numerical stability 100%

---

## ROLLBACK PLAN

If issues occur during implementation:

1. **Git Rollback**
   ```bash
   git reset --hard HEAD
   ```

2. **Quick Test**
   ```bash
   python run_determinism_audit.py --quick
   ```

3. **Incremental Approach**
   - Complete Phase 1 only if issues
   - Test thoroughly before Phase 2
   - Use feature branches

---

## DOCUMENTATION UPDATES

### 1. Update README.md

Add section:
```markdown
## Deterministic Testing

To run determinism tests:

```bash
export PYTHONHASHSEED=42
python run_determinism_audit.py
```

Expected output:
- Status: PASS
- All Hashes Match: YES
- Numerical Stability: 100.0%
```

### 2. Update API Documentation

Add to class docstrings:
```python
"""
Supports deterministic mode for reproducible testing.
When deterministic_mode=True:
- Cache is disabled
- Fixed timestamps used if test_timestamp provided
- All outputs are bit-identical for same input
"""
```

### 3. Update Contributing Guide

Add section on determinism testing.

---

## SIGN-OFF

**Remediation Plan Prepared**: November 15, 2025
**Target Completion Date**: November 17, 2025
**Estimated Effort**: 4-6 hours
**Difficulty**: Medium
**Risk Level**: Low (backward compatible changes)

Once all phases complete, GL-002 will achieve **100% reproducibility** and support deterministic computation verification.

