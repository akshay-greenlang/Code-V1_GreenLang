# GL-002 Critical Fixes Required

**Priority Level:** CRITICAL - Must fix before production deployment
**Estimated Effort:** 3-4 weeks
**Last Updated:** 2025-11-15

---

## CRITICAL ISSUE #1: Broken Relative Imports (8 Files)

### Status: BLOCKING - Will cause runtime errors

### Problem:
Calculator modules use absolute imports instead of relative imports, causing `ModuleNotFoundError`.

### Affected Files:
```
calculators/blowdown_optimizer.py (Line 15)
calculators/combustion_efficiency.py (Line 15)
calculators/control_optimization.py (Line 15)
calculators/economizer_performance.py (Line 15)
calculators/emissions_calculator.py (Line 16)
calculators/fuel_optimization.py (Line 16)
calculators/heat_transfer.py (Line 15)
calculators/steam_generation.py (Line 16)
```

### Current Code:
```python
from provenance import ProvenanceTracker, ProvenanceRecord
from provenance import ProvenanceTracker
```

### Fixed Code:
```python
from .provenance import ProvenanceTracker, ProvenanceRecord
from .provenance import ProvenanceTracker
```

### Test:
```bash
cd /c/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation/agents/GL-002
python -c "from calculators import combustion_efficiency"
# Should not raise ModuleNotFoundError
```

### Effort: 15 minutes (automated fix possible)

---

## CRITICAL ISSUE #2: Missing Type Hints (629 Functions)

### Status: BLOCKING - Cannot use type checkers

### Problem:
Only ~1.3% of functions have complete type hints. This prevents:
- mypy/pyright validation
- IDE autocomplete
- Runtime type checking libraries
- Clear API contracts

### Top Offenders:
1. `tools.py` - 926 lines, <5% type hints
2. `calculators/emissions_calculator.py` - 760 lines
3. `calculators/steam_generation.py` - 782 lines
4. `integrations/data_transformers.py` - 1,301 lines

### Example Fixes:

#### In `tools.py`:

**Before:**
```python
def _calculate_theoretical_air(self, fuel_properties):
    """Calculate theoretical air requirement for complete combustion."""
    C = fuel_properties['carbon_percent'] / 100
    return 11.51 * C
```

**After:**
```python
def _calculate_theoretical_air(
    self,
    fuel_properties: Dict[str, Any]
) -> float:
    """
    Calculate theoretical air requirement for complete combustion.

    Args:
        fuel_properties: Dictionary containing carbon_percent, hydrogen_percent, etc.

    Returns:
        Theoretical air required in kg air per kg fuel.
    """
    C = fuel_properties['carbon_percent'] / 100
    return 11.51 * C
```

#### In `boiler_efficiency_orchestrator.py`:

**Before:**
```python
async def _analyze_operational_state_async(
    self,
    boiler_data: Dict[str, Any],
    sensor_feeds: Dict[str, Any]
) -> BoilerOperationalState:  # ✅ Good
    """Analyze current boiler operational state asynchronously."""
    cache_key = self._get_cache_key('state_analysis', {  # ❌ Missing return type
        'boiler': boiler_data,
        'sensors': sensor_feeds
    })
```

**After:**
```python
async def _analyze_operational_state_async(
    self,
    boiler_data: Dict[str, Any],
    sensor_feeds: Dict[str, Any]
) -> BoilerOperationalState:
    """Analyze current boiler operational state asynchronously."""
    cache_key: str = self._get_cache_key('state_analysis', {
        'boiler': boiler_data,
        'sensors': sensor_feeds
    })
```

### Automated Tools:
```bash
# Install pyright
pip install pyright

# Check for type errors
pyright --outputjson > type_errors.json

# Or use mypy
pip install mypy
mypy --strict . --ignore-missing-imports
```

### Effort: 8-10 hours

---

## CRITICAL ISSUE #3: Hardcoded Credentials (3 Instances)

### Status: BLOCKING - Security vulnerability

### Problem:
Credentials appear in test files, which could be committed to git.

### Affected Files:
1. `tests/test_integrations.py` - Lines with hardcoded tokens
2. `tests/test_security.py` - Hardcoded password and API key

### Current Code:
```python
# tests/test_integrations.py
assert erp_connector.auth_token == "auth-token-123"
assert cloud_connector.access_token == "token-123"

# tests/test_security.py
password = "SecurePassword123!"
api_key = "sk_live_abcd1234efgh5678ijkl9012mnop3456"
```

### Fixed Code:
```python
# tests/test_integrations.py
import os

TEST_AUTH_TOKEN = os.getenv("TEST_AUTH_TOKEN", "test-token-placeholder")
TEST_CLOUD_TOKEN = os.getenv("TEST_CLOUD_TOKEN", "test-token-placeholder")

def test_auth():
    assert erp_connector.auth_token == TEST_AUTH_TOKEN
    assert cloud_connector.access_token == TEST_CLOUD_TOKEN

# tests/test_security.py
import os

TEST_PASSWORD = os.getenv("TEST_PASSWORD", "test-password-placeholder")
TEST_API_KEY = os.getenv("TEST_API_KEY", "test-api-key-placeholder")

def test_password_security():
    password = TEST_PASSWORD
    # ... test logic
```

### Add to `.gitignore`:
```
.env
.env.local
.env.*.local
test_credentials.json
```

### Add Pre-commit Hook:
```bash
# .git/hooks/pre-commit
#!/bin/bash
grep -r "password\|api_key\|auth_token\|secret" --include="*.py" . | \
grep -E "=\s*['\"]" && {
    echo "ERROR: Hardcoded credentials detected!"
    exit 1
}
```

### Effort: 30 minutes

---

## CRITICAL ISSUE #4: Race Condition in Cache (Thread-Safety)

### Status: BLOCKING - Concurrency issue

### Problem:
The results cache and performance metrics use non-thread-safe dictionary operations with multiple async tasks.

### Affected Code:
File: `boiler_efficiency_orchestrator.py`

```python
# Lines 152-155
self._results_cache = {}
self._cache_timestamps = {}

# Lines 340-341 - UNSAFE
if self._is_cache_valid(cache_key):  # Check
    return self._results_cache[cache_key]  # Get (race condition possible)

# Lines 903-915 - UNSAFE concurrent modification
if len(self._results_cache) > 200:
    oldest_keys = sorted(...)[:50]
    for key in oldest_keys:
        del self._results_cache[key]  # Multiple tasks could delete same key
```

### Fixed Code:

```python
from threading import RLock
from collections import OrderedDict

class BoilerEfficiencyOptimizer(BaseAgent):
    def __init__(self, config: BoilerEfficiencyConfig):
        # ... existing code ...

        # Add thread-safe cache with lock
        self._cache_lock = RLock()  # Reentrant lock for nested calls
        self._results_cache = OrderedDict()  # FIFO eviction
        self._cache_timestamps = OrderedDict()

        # Add lock for performance metrics
        self._metrics_lock = RLock()

    async def _analyze_operational_state_async(
        self,
        boiler_data: Dict[str, Any],
        sensor_feeds: Dict[str, Any]
    ) -> BoilerOperationalState:
        """Analyze current boiler operational state asynchronously."""
        cache_key = self._get_cache_key('state_analysis', {
            'boiler': boiler_data,
            'sensors': sensor_feeds
        })

        # SAFE: Use lock for cache check and retrieve
        with self._cache_lock:
            if self._is_cache_valid(cache_key):
                self.performance_metrics['cache_hits'] += 1
                return self._results_cache[cache_key]

        # Analyze state...
        self.performance_metrics['cache_misses'] += 1

        # Store with lock
        with self._cache_lock:
            self._store_in_cache(cache_key, operational_state)

        return operational_state

    def _store_in_cache(self, cache_key: str, result: Any) -> None:
        """Store result in cache with thread-safety."""
        with self._cache_lock:
            self._results_cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()

            # Limit cache size - FIFO eviction
            if len(self._results_cache) > 200:
                # Remove oldest key(s)
                for _ in range(50):
                    if self._results_cache:
                        oldest_key = next(iter(self._results_cache))
                        del self._results_cache[oldest_key]
                        del self._cache_timestamps[oldest_key]

    def _update_performance_metrics(
        self,
        execution_time_ms: float,
        combustion_result: CombustionOptimizationResult,
        emissions_result: Any
    ) -> None:
        """Update performance metrics with thread-safety."""
        with self._metrics_lock:
            # Update average optimization time
            n = self.performance_metrics['optimizations_performed']
            if n > 0:
                current_avg = self.performance_metrics['avg_optimization_time_ms']
                self.performance_metrics['avg_optimization_time_ms'] = (
                    (current_avg * (n - 1) + execution_time_ms) / n
                )

            # Update fuel savings
            if hasattr(combustion_result, 'fuel_saved_kg'):
                self.performance_metrics['fuel_savings_kg'] += combustion_result.fuel_saved_kg
```

### Test:
```python
# tests/test_concurrency.py
import asyncio

async def test_cache_thread_safety():
    config = create_test_config()
    optimizer = BoilerEfficiencyOptimizer(config)

    # Simulate 100 concurrent requests
    tasks = [
        optimizer._analyze_operational_state_async(
            {'test': i},
            {'load': i % 100}
        )
        for i in range(100)
    ]

    results = await asyncio.gather(*tasks)

    # Verify cache integrity
    assert len(optimizer._results_cache) <= 200
    assert optimizer.performance_metrics['cache_hits'] > 0
    assert optimizer.performance_metrics['cache_misses'] > 0
```

### Effort: 2-3 hours

---

## HIGH PRIORITY: Input Validation

### Status: BLOCKING - No validation of constraint relationships

### Problem:
Constraint validation doesn't verify that max > min.

### Affected File:
`config.py` - OperationalConstraints class

### Current Code:
```python
class OperationalConstraints(BaseModel):
    max_pressure_bar: float = Field(...)
    min_pressure_bar: float = Field(...)
    # No validation that max > min!
```

### Fixed Code:
```python
from pydantic import validator

class OperationalConstraints(BaseModel):
    max_pressure_bar: float = Field(..., description="Maximum operating pressure")
    min_pressure_bar: float = Field(..., description="Minimum operating pressure")
    max_temperature_c: float = Field(..., description="Maximum operating temperature")
    min_temperature_c: float = Field(..., description="Minimum operating temperature")
    max_excess_air_percent: float = Field(25.0, le=50, description="Maximum excess air")
    min_excess_air_percent: float = Field(5.0, ge=0, description="Minimum excess air")

    @validator('max_pressure_bar', pre=False, always=False)
    def validate_pressure_range(cls, v: float, values: Dict) -> float:
        """Ensure max_pressure >= min_pressure."""
        if 'min_pressure_bar' in values:
            if v < values['min_pressure_bar']:
                raise ValueError(
                    f"max_pressure_bar ({v}) must be >= min_pressure_bar ({values['min_pressure_bar']})"
                )
        return v

    @validator('max_temperature_c')
    def validate_temperature_range(cls, v: float, values: Dict) -> float:
        """Ensure max_temperature >= min_temperature."""
        if 'min_temperature_c' in values:
            if v < values['min_temperature_c']:
                raise ValueError(
                    f"max_temperature_c ({v}) must be >= min_temperature_c ({values['min_temperature_c']})"
                )
        return v

    @validator('max_excess_air_percent')
    def validate_excess_air_range(cls, v: float, values: Dict) -> float:
        """Ensure max_excess_air >= min_excess_air."""
        if 'min_excess_air_percent' in values:
            if v < values['min_excess_air_percent']:
                raise ValueError(
                    f"max_excess_air ({v}) must be >= min_excess_air ({values['min_excess_air_percent']})"
                )
        return v
```

### Test:
```python
# tests/test_validation.py
import pytest

def test_constraint_validation():
    # Should raise ValueError
    with pytest.raises(ValueError, match="max_pressure must be >="):
        OperationalConstraints(
            max_pressure_bar=5.0,
            min_pressure_bar=10.0,  # Reversed!
            # ... other fields
        )

    # Should succeed
    constraints = OperationalConstraints(
        max_pressure_bar=42,
        min_pressure_bar=5,
        # ... other fields
    )
    assert constraints.max_pressure_bar >= constraints.min_pressure_bar
```

### Effort: 2 hours

---

## HIGH PRIORITY: Add Timeout Enforcement

### Status: HIGH - Config specifies timeout but not used

### Problem:
Async operations use `asyncio.to_thread` without timeout enforcement.

### Affected File:
`boiler_efficiency_orchestrator.py` - lines 235-256, 356-360, 418-423, etc.

### Current Code:
```python
async def _analyze_operational_state_async(self, boiler_data, sensor_feeds):
    result = await asyncio.to_thread(
        self.tools.calculate_boiler_efficiency,
        boiler_data,
        sensor_feeds
    )  # Could hang forever!
```

### Fixed Code:
```python
async def _analyze_operational_state_async(
    self,
    boiler_data: Dict[str, Any],
    sensor_feeds: Dict[str, Any]
) -> BoilerOperationalState:
    """Analyze current boiler operational state asynchronously."""
    cache_key = self._get_cache_key('state_analysis', {
        'boiler': boiler_data,
        'sensors': sensor_feeds
    })

    if self._is_cache_valid(cache_key):
        self.performance_metrics['cache_hits'] += 1
        return self._results_cache[cache_key]

    self.performance_metrics['cache_misses'] += 1

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                self.tools.calculate_boiler_efficiency,
                boiler_data,
                sensor_feeds
            ),
            timeout=self.boiler_config.calculation_timeout_seconds
        )
    except asyncio.TimeoutError as e:
        logger.error(
            f"Boiler efficiency calculation timed out after "
            f"{self.boiler_config.calculation_timeout_seconds}s"
        )
        raise TimeoutError(
            f"Boiler efficiency calculation exceeded "
            f"{self.boiler_config.calculation_timeout_seconds}s timeout"
        ) from e

    # ... rest of method
```

### Apply to All Async Calls:
```python
# Apply to lines:
# - 356-360: _analyze_operational_state_async
# - 418-423: _optimize_combustion_async
# - 448-453: _optimize_steam_generation_async
# - 476-480: _minimize_emissions_async
# - 505-510: _calculate_parameter_adjustments_async
# - 731-736: _coordinate_agents_async
```

### Effort: 2 hours

---

## HIGH PRIORITY: Input Validation - Null Checks

### Status: HIGH - No null/None validation

### Problem:
Data transformation code assumes valid data without checking for None.

### Affected File:
`integrations/data_transformers.py`

### Current Code:
```python
def process_scada_data(self, scada_feed: Dict[str, Any]) -> Dict[str, Any]:
    """Process SCADA data for boiler optimization."""
    processed = {
        'timestamp': datetime.now().isoformat(),
        'tags_processed': len(scada_feed.get('tags', {})),  # scada_feed could be None!
        'data_quality': 'good' if scada_feed.get('quality', 100) > 90 else 'poor',
        'values': {}
    }

    for tag, value in scada_feed.get('tags', {}).items():
        if 'boiler' in tag.lower():  # tag could be None!
            processed['values'][tag] = value
```

### Fixed Code:
```python
def process_scada_data(self, scada_feed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process SCADA data for boiler optimization.

    Args:
        scada_feed: Raw SCADA data dictionary

    Returns:
        Processed SCADA data with validation and quality metrics

    Raises:
        ValueError: If scada_feed is None
        TypeError: If scada_feed is not a dictionary
    """
    if scada_feed is None:
        raise ValueError("SCADA feed cannot be None")
    if not isinstance(scada_feed, dict):
        raise TypeError(f"SCADA feed must be a dictionary, got {type(scada_feed)}")

    processed = {
        'timestamp': datetime.now().isoformat(),
        'tags_processed': len(scada_feed.get('tags', {})),
        'data_quality': 'good' if scada_feed.get('quality', 100) > 90 else 'poor',
        'values': {}
    }

    tags = scada_feed.get('tags', {})
    if not isinstance(tags, dict):
        logger.warning(f"Invalid tags format: {type(tags)}, expected dict")
        return processed

    for tag, value in tags.items():
        if tag is None:
            logger.warning(f"Skipping None tag with value {value}")
            continue
        if value is None:
            logger.debug(f"Skipping None value for tag {tag}")
            continue

        if isinstance(tag, str) and 'boiler' in tag.lower():
            processed['values'][tag] = value

    return processed
```

### Effort: 3-4 hours

---

## Summary Table

| Issue | Files | Effort | Blocker |
|-------|-------|--------|---------|
| Fix relative imports | 8 calc files | 15 min | YES |
| Add type hints | 31 files | 8-10 hrs | YES |
| Remove hardcoded credentials | 2 test files | 30 min | YES |
| Add cache locking | 1 main file | 2-3 hrs | YES |
| Add constraint validation | 1 config file | 2 hrs | YES |
| Add timeout enforcement | 1 main file | 2 hrs | YES |
| Add null checks | Multiple | 3-4 hrs | YES |
| **TOTAL** | | **18-22 hrs** | |

---

## Implementation Plan

### Phase 1: Critical Fixes (4 hours)
1. Fix relative imports (15 min)
2. Remove hardcoded credentials (30 min)
3. Add cache locking (2-3 hrs)

### Phase 2: Type Hints (10 hours)
1. Start with public API methods
2. Add return types for all functions
3. Add parameter types for all functions
4. Use mypy --strict to verify

### Phase 3: Validation & Error Handling (6-8 hours)
1. Add constraint validation
2. Add timeout enforcement
3. Add null/None checks throughout

### Phase 4: Testing & Verification (4 hours)
1. Update tests for new validation
2. Add concurrency tests
3. Run full test suite with type checking

---

## Success Criteria

- [ ] All 8 relative imports fixed (calculators can import from parent)
- [ ] Type checker (mypy/pyright) reports 0 errors
- [ ] No hardcoded credentials in any file
- [ ] Cache operations are thread-safe under concurrent load
- [ ] All constraints validated at instantiation time
- [ ] All async operations have timeout enforcement
- [ ] All input parameters validated (None checks)
- [ ] Full test suite passes
- [ ] Code quality report shows PASS for all critical items

---

**Next Step:** Create tracking issue for each fix and assign to development team.
