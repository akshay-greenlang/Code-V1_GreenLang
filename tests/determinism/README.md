# Determinism Testing Framework - GreenLang Phase 3

Comprehensive testing infrastructure for ensuring reproducible, deterministic results across all GreenLang agents.

## Overview

This framework provides three complementary testing approaches:

1. **Hash-Based Determinism Testing** - Verify identical outputs across multiple runs
2. **Snapshot Testing** - Golden file comparison for regression detection
3. **Property-Based Testing** - Invariant verification with Hypothesis

## Quick Start

### Running Determinism Tests

```bash
# Run all determinism tests
pytest tests/determinism/ -v

# Run only FuelAgent determinism tests
pytest tests/determinism/test_fuel_agent_determinism.py -v

# Run with determinism marker
pytest -m determinism -v

# Run property-based tests only
pytest -m property -v

# Verbose output with test durations
pytest tests/determinism/ -v --durations=10
```

### Environment Variables

```bash
# Update snapshots automatically (use with caution)
UPDATE_SNAPSHOTS=1 pytest tests/determinism/ -v

# Set Hypothesis profile (fast, dev, ci)
HYPOTHESIS_PROFILE=dev pytest tests/determinism/ -v
```

---

## Framework Components

### 1. DeterminismTester (`test_framework.py`)

Hash-based reproducibility verification for agents.

#### Features

- SHA256 hash comparison of outputs
- Automatic normalization of timestamps, platform-specific values
- Byte-level and field-level difference detection
- Support for both sync and async agents
- Cross-platform compatibility

#### Usage

```python
from tests.determinism.test_framework import DeterminismTester, assert_deterministic

# Create tester
tester = DeterminismTester(
    normalize_platform=True,
    normalize_timestamps=True,
    normalize_floats=True,
    float_precision=6,
)

# Test async agent
result = await tester.test_agent_async(
    agent=my_agent,
    payload={"fuel_type": "natural_gas", "amount": 1000},
    runs=5,
    store_outputs=True,
)

# Assert determinism
assert_deterministic(result)

# Examine results
print(f"Deterministic: {result.is_deterministic}")
print(f"Unique hashes: {len(set(result.hashes))}")
print(f"Differences: {result.differences}")
```

#### Test Sync Agent

```python
# Test sync agent
result = tester.test_agent_sync(
    agent=my_agent,
    payload=payload,
    runs=5,
)

assert_deterministic(result)
```

#### Test Any Function

```python
# Test any function (auto-detects sync/async)
def my_calculation(x, y):
    return x * y + 10

result = tester.test_function(my_calculation, 5, 3, runs=10)
assert_deterministic(result)
```

#### Normalization Options

```python
tester = DeterminismTester(
    normalize_platform=True,      # Remove platform-specific values
    normalize_timestamps=True,    # Remove timestamps
    normalize_floats=True,        # Round floats
    float_precision=6,            # Decimal places
    sort_keys=True,              # Sort JSON keys
)
```

### 2. SnapshotManager (`snapshot_manager.py`)

Golden file testing for regression detection.

#### Features

- JSON-based snapshot storage with pretty formatting
- Detailed diff reporting (added/removed/changed fields)
- Auto-update mode for blessing new snapshots
- Cross-platform path normalization
- Snapshot metadata tracking

#### Usage

```python
from tests.determinism.snapshot_manager import SnapshotManager, assert_snapshot_matches

# Create manager
manager = SnapshotManager(
    snapshot_dir=None,  # Default: tests/determinism/snapshots
    auto_update=False,  # Set to True to update snapshots
    normalize_output=True,
)

# Save a snapshot
agent_output = agent.run(payload)
manager.save_snapshot("test_fuel_natural_gas_1000", agent_output)

# Compare to snapshot
diff = manager.compare_snapshot("test_fuel_natural_gas_1000", new_output)
assert_snapshot_matches(diff)

# Examine diff
print(diff)
# Output:
# Added fields: data.new_field
# Changed fields: 2
# Details:
# [CHANGED] data.emissions: 5310.0 -> 5320.0
```

#### Snapshot Management

```python
# List all snapshots
snapshots = manager.list_snapshots()
print(snapshots)  # ['test_fuel_natural_gas_1000', ...]

# Get snapshot info
info = manager.get_snapshot_info("test_fuel_natural_gas_1000")
print(info)
# {'test_name': '...', 'path': '...', 'size_bytes': 1024, 'exists': True}

# Delete snapshot
manager.delete_snapshot("old_test")
```

#### Auto-Update Mode

```bash
# Update all snapshots (use with caution!)
UPDATE_SNAPSHOTS=1 pytest tests/determinism/ -v
```

Or in code:

```python
manager = SnapshotManager(auto_update=True)
diff = manager.compare_snapshot("test_name", output)
# Snapshot will be automatically updated if different
```

### 3. Property-Based Testing (`test_properties.py`)

Invariant verification with Hypothesis for robust testing.

#### Features

- Automated test case generation
- Edge case discovery
- Invariant verification across wide input ranges
- Configurable test profiles (fast, dev, ci)

#### Usage

```python
from hypothesis import given, strategies as st, settings
import pytest

@pytest.mark.property
@given(
    amount=st.floats(min_value=0.1, max_value=100000.0),
    renewable_percentage=st.floats(min_value=0.0, max_value=100.0),
)
@settings(max_examples=20, deadline=5000)
def test_emissions_non_negative(agent, amount, renewable_percentage):
    """Property: Emissions are always >= 0."""
    result = agent.calculate_emissions(amount, renewable_percentage)
    assert result["emissions_kg_co2e"] >= 0
```

#### Common Property Patterns

**Output Structure Validation**
```python
@given(fuel_type=st.sampled_from(["natural_gas", "diesel", "coal"]))
def test_output_structure_valid(agent, fuel_type):
    result = agent.run({"fuel_type": fuel_type, ...})
    assert "emissions_kg_co2e" in result
    assert isinstance(result["emissions_kg_co2e"], float)
```

**Determinism Property**
```python
@given(amount=st.floats(min_value=1.0, max_value=10000.0))
def test_calculation_deterministic(agent, amount):
    results = [agent.calculate(amount) for _ in range(3)]
    assert all(r == results[0] for r in results)
```

**Physical Constraints**
```python
@given(
    base_amount=st.floats(min_value=1.0, max_value=1000.0),
    scale_factor=st.floats(min_value=2.0, max_value=10.0),
)
def test_emissions_scale_linearly(agent, base_amount, scale_factor):
    emissions1 = agent.calculate(base_amount)
    emissions2 = agent.calculate(base_amount * scale_factor)
    assert abs(emissions2 - emissions1 * scale_factor) < 0.01
```

#### Hypothesis Profiles

```bash
# Fast profile (5 examples, 1s deadline) - default
pytest tests/determinism/test_properties.py -v

# Dev profile (100 examples, 10s deadline)
HYPOTHESIS_PROFILE=dev pytest tests/determinism/test_properties.py -v

# CI profile (10 examples, 5s deadline, deterministic)
HYPOTHESIS_PROFILE=ci pytest tests/determinism/test_properties.py -v
```

---

## Adding Determinism Tests for New Agents

### Step 1: Create Test File

Create `tests/determinism/test_{agent_name}_determinism.py`:

```python
"""Determinism tests for {AgentName}."""

import pytest
from greenlang.agents.{agent_module} import {AgentClass}
from tests.determinism.test_framework import DeterminismTester, assert_deterministic
from tests.determinism.snapshot_manager import SnapshotManager, assert_snapshot_matches

@pytest.mark.determinism
class Test{AgentName}Determinism:

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return {AgentClass}()

    @pytest.fixture
    def tester(self):
        """Create DeterminismTester."""
        return DeterminismTester()

    @pytest.fixture
    def snapshot_manager(self):
        """Create SnapshotManager."""
        return SnapshotManager()

    # Test 1: Hash-based determinism
    @pytest.mark.asyncio
    async def test_hash_determinism(self, agent, tester):
        payload = {"input": "test"}
        result = await tester.test_agent_async(agent, payload, runs=5)
        assert_deterministic(result)

    # Test 2: Snapshot testing
    def test_output_snapshot(self, agent, snapshot_manager):
        payload = {"input": "test"}
        output = agent.run(payload)
        diff = snapshot_manager.compare_snapshot("agent_test_case", output)
        assert_snapshot_matches(diff)

    # Test 3: Property-based testing
    @pytest.mark.property
    @given(amount=st.floats(min_value=0.1, max_value=1000.0))
    def test_output_valid(self, agent, amount):
        result = agent.run({"amount": amount})
        assert result["success"] is True
        assert "data" in result
```

### Step 2: Add Test Fixtures

Add agent-specific fixtures in conftest or test file:

```python
@pytest.fixture
def sample_payload():
    """Standard test payload for agent."""
    return {
        "field1": "value1",
        "field2": 100.0,
    }

@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock external dependencies."""
    # Mock API calls, file I/O, etc.
    pass
```

### Step 3: Run Tests

```bash
# Run new agent determinism tests
pytest tests/determinism/test_{agent_name}_determinism.py -v

# Create initial snapshots
UPDATE_SNAPSHOTS=1 pytest tests/determinism/test_{agent_name}_determinism.py -v
```

---

## Test Markers

The framework uses pytest markers for organizing tests:

```python
@pytest.mark.determinism      # All determinism tests
@pytest.mark.property         # Property-based tests
@pytest.mark.snapshot         # Snapshot tests
@pytest.mark.asyncio          # Async tests
```

### Running by Marker

```bash
# Run only determinism tests
pytest -m determinism -v

# Run property tests
pytest -m property -v

# Run snapshot tests
pytest -m snapshot -v

# Combine markers
pytest -m "determinism and not property" -v
```

---

## Best Practices

### 1. Deterministic Agent Design

**DO:**
- Use `seed=42` and `temperature=0` for AI agents
- Use deterministic algorithms (avoid random())
- Normalize timestamps and platform-specific values
- Use fixed precision for floating-point numbers

**DON'T:**
- Use `random.random()` without seeding
- Include timestamps in outputs
- Depend on system-specific values
- Use non-deterministic external APIs

### 2. Test Coverage

Each agent should have:
- ✅ Hash-based determinism test (5+ runs)
- ✅ Snapshot test for key scenarios
- ✅ Property tests for invariants
- ✅ Edge case tests (zero, negative, large values)
- ✅ Cross-platform compatibility test

### 3. Snapshot Management

**Guidelines:**
- Keep snapshots small and focused
- One snapshot per test scenario
- Use descriptive snapshot names
- Review diffs before updating
- Commit snapshots to git

**Naming Convention:**
```
{agent_name}_{scenario}_{params}.snapshot.json

Examples:
- fuel_agent_natural_gas_1000_therms.snapshot.json
- carbon_agent_total_emissions_office.snapshot.json
- grid_factor_us_ca_peak_hour.snapshot.json
```

### 4. Property Test Design

**Good Properties:**
- Non-negativity: `result >= 0`
- Monotonicity: `f(x) <= f(x + delta)`
- Idempotence: `f(f(x)) == f(x)`
- Commutativity: `f(a, b) == f(b, a)`
- Linear scaling: `f(n*x) == n*f(x)`

**Example:**
```python
@given(x=st.floats(min_value=0, max_value=1000))
def test_monotonicity(agent, x):
    """Emissions should increase with fuel amount."""
    result1 = agent.calculate(x)
    result2 = agent.calculate(x + 10)
    assert result2 >= result1
```

### 5. Handling Non-Determinism

If your agent has legitimate non-determinism:

1. **Isolate non-deterministic parts:**
   ```python
   def get_timestamp():
       return datetime.now()  # Non-deterministic

   def calculate_emissions(amount, timestamp):
       # Deterministic given inputs
       return amount * factor
   ```

2. **Use normalization:**
   ```python
   tester = DeterminismTester(
       normalize_timestamps=True,  # Strip timestamps
       normalize_platform=True,    # Strip platform info
   )
   ```

3. **Mock non-deterministic dependencies:**
   ```python
   @patch('module.get_timestamp')
   def test_determinism(mock_timestamp, agent):
       mock_timestamp.return_value = "2024-01-01T00:00:00Z"
       # Test is now deterministic
   ```

---

## Troubleshooting

### Tests Fail Intermittently

**Cause:** Non-deterministic behavior in agent or test.

**Solutions:**
1. Check for `random()` calls without seeding
2. Verify AI agents use `temperature=0, seed=42`
3. Mock external APIs and file I/O
4. Enable normalization in tester

### Snapshots Don't Match

**Cause:** Agent output changed (regression or intentional).

**Solutions:**
1. Review diff: `pytest tests/determinism/ -v`
2. If intentional: `UPDATE_SNAPSHOTS=1 pytest`
3. If regression: Fix agent code

### Property Tests Fail

**Cause:** Property violation found by Hypothesis.

**Solutions:**
1. Review failing example in pytest output
2. Add `.example()` decorator to reproduce
3. Fix agent to satisfy property
4. Or adjust property if too strict

### Async Tests Timeout

**Cause:** Slow agent operations or deadlocks.

**Solutions:**
1. Increase deadline: `@settings(deadline=10000)`
2. Mock slow operations
3. Use smaller test inputs
4. Check for async/await issues

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Determinism Tests

on: [push, pull_request]

jobs:
  determinism:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio hypothesis

      - name: Run determinism tests
        run: |
          pytest tests/determinism/ -v --tb=short
        env:
          HYPOTHESIS_PROFILE: ci

      - name: Upload snapshots on failure
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: snapshots
          path: tests/determinism/snapshots/
```

---

## API Reference

### DeterminismTester

```python
class DeterminismTester:
    def __init__(
        self,
        normalize_platform: bool = True,
        normalize_timestamps: bool = True,
        normalize_floats: bool = True,
        float_precision: int = 6,
        sort_keys: bool = True,
    )

    def compute_hash(self, data: Any) -> str
    def compare_outputs(self, output1: Any, output2: Any) -> List[Dict]

    async def test_agent_async(
        self, agent: Any, payload: Dict, runs: int = 5
    ) -> DeterminismResult

    def test_agent_sync(
        self, agent: Any, payload: Dict, runs: int = 5
    ) -> DeterminismResult

    def test_function(
        self, func: Callable, *args, runs: int = 5, **kwargs
    ) -> DeterminismResult
```

### SnapshotManager

```python
class SnapshotManager:
    def __init__(
        self,
        snapshot_dir: Optional[Path] = None,
        auto_update: Optional[bool] = None,
        normalize_output: bool = True,
    )

    def save_snapshot(self, test_name: str, output: Any) -> Path
    def load_snapshot(self, test_name: str) -> Optional[Dict]
    def compare_snapshot(self, test_name: str, actual_output: Any) -> SnapshotDiff
    def delete_snapshot(self, test_name: str) -> bool
    def list_snapshots(self) -> List[str]
    def get_snapshot_info(self, test_name: str) -> Optional[Dict]
```

### Helper Functions

```python
def assert_deterministic(result: DeterminismResult, message: Optional[str] = None)
def assert_snapshot_matches(diff: SnapshotDiff, message: Optional[str] = None)
def test_agent_determinism(agent: Any, payload: Dict, runs: int = 5) -> DeterminismResult
```

---

## Examples

See `tests/determinism/test_fuel_agent_determinism.py` for comprehensive examples.

---

## Support

For questions or issues:
1. Check this README
2. Review existing test files
3. Open an issue in the repository
4. Contact the GreenLang team

---

**Phase:** Phase 3 - Production Hardening
**Date:** November 2024
**Version:** 1.0.0
**Author:** GreenLang Framework Team
