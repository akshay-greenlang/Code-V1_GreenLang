# GreenLang ScenarioSpec v1 - Comprehensive Guide

**Version:** 1.0.0
**Date:** October 2025
**Spec:** SIM-401 (Scenario Spec & Seeded RNG)
**Status:** Production Ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Deterministic RNG Matters](#why-deterministic-rng-matters)
3. [Root Seed vs Derived Substreams](#root-seed-vs-derived-substreams)
4. [Live vs Replay Modes](#live-vs-replay-modes)
5. [Scenario Specification Format](#scenario-specification-format)
6. [Provenance Fields](#provenance-fields)
7. [Usage Examples](#usage-examples)
8. [Reproducibility Guide](#reproducibility-guide)
9. [Advanced Topics](#advanced-topics)

---

## Introduction

GreenLang ScenarioSpec v1 provides a **declarative, version-controlled specification format** for defining deterministic and stochastic climate modeling scenarios. It enables:

- **Parameter sweeps** (grid search over discrete values)
- **Monte Carlo simulations** (stochastic sampling from distributions)
- **Deterministic reproducibility** (byte-exact results across OS/Python versions)
- **Provenance tracking** (complete audit trail for regulatory compliance)
- **Hierarchical RNG substreams** (adding/removing parameters doesn't affect existing streams)

**Key Features:**
- Pure Python (no NumPy dependency for core RNG)
- Cross-platform deterministic (Linux, macOS, Windows)
- YAML/JSON specification format
- JSON Schema validation
- Pydantic-based type safety
- Integration with GreenLang provenance system

---

## Why Deterministic RNG Matters

### The Problem: Non-Deterministic Simulations

Traditional Monte Carlo simulations often produce **different results** when re-run due to:
1. **Platform differences** (Linux vs Windows vs macOS)
2. **Python version changes** (NumPy RNG changed in 1.17+)
3. **Library updates** (different random number algorithms)
4. **Concurrency** (thread scheduling affects RNG state)

This is **catastrophic** for:
- **Regulatory compliance** (auditors can't reproduce your numbers)
- **Debugging** (can't reproduce bugs)
- **Peer review** (scientists can't validate your work)
- **CI/CD** (tests flake randomly)

### The Solution: GreenLang GLRNG

GLRNG (GreenLang RNG) provides:

1. **Pure Python Implementation**
   - No dependency on NumPy's RNG (which has changed algorithms)
   - SplitMix64 algorithm (stable, well-tested)
   - Explicit floating-point rounding for cross-platform consistency

2. **HMAC-SHA256 Substream Derivation**
   - Cryptographically secure seed derivation
   - Hierarchical path-based streams
   - Adding/removing parameters doesn't affect other streams

3. **Cross-Platform Determinism**
   - Byte-exact results on Linux, macOS, Windows
   - Python 3.10, 3.11, 3.12 identical outputs
   - ARM and x86 architectures produce same results

### Real-World Example

```python
from greenlang.simulation.rng import GLRNG

# Same seed = same results, always
rng1 = GLRNG(seed=42)
rng2 = GLRNG(seed=42)

# These will be identical across all platforms
assert rng1.normal(100, 15) == rng2.normal(100, 15)
# Both produce: 107.341892 (exactly, on all systems)
```

---

## Root Seed vs Derived Substreams

### The Problem: Sequential RNG State

Traditional approach:
```python
# BAD: Sequential state sharing
rng = Random(42)

# Trial 0
for param in params:
    value = rng.uniform()  # Consumes RNG state

# Trial 1 - RNG state has moved!
# Adding/removing a parameter in Trial 0 changes ALL subsequent values
```

**Problem:** Adding a parameter to Trial 0 changes **all values** in Trial 1+.

### The Solution: Hierarchical Substreams

GreenLang approach:
```python
# GOOD: Independent substreams
root_rng = GLRNG(seed=42)

# Each trial gets independent stream
for trial in range(100):
    trial_rng = root_rng.spawn(f"trial:{trial}")

    # Each parameter gets independent stream within trial
    for param in params:
        param_rng = trial_rng.spawn(f"param:{param.name}")
        value = param_rng.uniform()
```

**Benefits:**
- Adding/removing parameters doesn't affect other parameters
- Trials are independent (can run in parallel)
- Reproducible even when scenario structure changes

### HMAC-SHA256 Derivation

Substream seeds are derived via:
```python
seed_child = HMAC-SHA256(
    key=seed_root_bytes,
    msg=f"scenario:{name}|param:{param_id}|trial:{trial_idx}"
).digest()[:8]  # First 8 bytes → 64-bit seed
```

This ensures:
1. **Collision resistance** (different paths = different seeds)
2. **Determinism** (same path = same seed, always)
3. **Independence** (substreams are statistically independent)

---

## Live vs Replay Modes

GreenLang scenarios support two execution modes:

### Replay Mode (Default - Deterministic)

```yaml
mode: "replay"
```

**Behavior:**
- All data comes from **cached snapshots**
- No network calls to external APIs
- 100% reproducible
- Default in CI/CD
- Required for regulatory audits

**Use Cases:**
- Automated testing
- Peer review
- Regulatory submissions
- Debugging

**Example:**
```python
from greenlang.simulation.runner import ScenarioRunner

runner = ScenarioRunner("scenarios/baseline.yaml")
# mode="replay" → uses cached grid intensity data
results = [model(**params) for params in runner.generate_samples()]
```

### Live Mode (Non-Deterministic)

```yaml
mode: "live"
```

**Behavior:**
- Fetches **fresh data** from external connectors
- Network calls to APIs (ElectricityMaps, WattTime, NREL, etc.)
- Non-reproducible (data changes over time)
- Requires authentication and rate limit management

**Use Cases:**
- Production forecasting
- Real-time optimization
- What-if analysis with current data

**Example:**
```python
runner = ScenarioRunner("scenarios/realtime_forecast.yaml")
# mode="live" → fetches current grid intensity from API
results = [model(**params) for params in runner.generate_samples()]
```

### Mode Enforcement

GreenLang enforces mode constraints:
- **Replay mode in CI**: Attempting Live mode in CI fails with `GLSecurityError.EGRESS_BLOCKED`
- **Snapshot validation**: Replay mode verifies snapshot hash for data integrity
- **Provenance recording**: Mode is recorded in provenance for audit trails

---

## Scenario Specification Format

### YAML Structure

```yaml
schema_version: "1.0.0"
name: "building_decarb_baseline"
description: "Baseline + retrofit sweep with MC on price sensitivity"
seed: 123456789
mode: "replay"

parameters:
  # Deterministic sweep parameters
  - id: "retrofit_level"
    type: "sweep"
    values: ["none", "light", "deep"]

  - id: "chiller_cop"
    type: "sweep"
    values: [3.2, 3.6, 4.0]

  # Stochastic parameters (sampled per trial)
  - id: "electricity_price_usd_per_kwh"
    type: "distribution"
    distribution:
      kind: "triangular"
      low: 0.08
      mode: 0.12
      high: 0.22

monte_carlo:
  trials: 2000
  seed_strategy: "derive-by-path"

metadata:
  owner: "squad-sim-ml"
  tags: ["buildings", "retrofit", "pricing", "monte-carlo"]
```

### Parameter Types

#### 1. Sweep Parameters (Deterministic Grid)

```yaml
- id: "temperature"
  type: "sweep"
  values: [15, 20, 25, 30]  # 4 discrete values
```

**Behavior:** Cartesian product over all sweep parameters.

**Example:** 3 retrofit levels × 4 temperatures = 12 combinations

#### 2. Distribution Parameters (Stochastic Sampling)

```yaml
- id: "price"
  type: "distribution"
  distribution:
    kind: "triangular"
    low: 0.08
    mode: 0.12
    high: 0.22
```

**Supported Distributions:**

**Uniform:**
```yaml
distribution:
  kind: "uniform"
  low: 0.0
  high: 1.0
```

**Normal (Gaussian):**
```yaml
distribution:
  kind: "normal"
  mean: 100.0
  std: 15.0
```

**Lognormal:**
```yaml
distribution:
  kind: "lognormal"
  mean: 0.0     # Log-space mean
  sigma: 1.0    # Log-space std
```

**Triangular:**
```yaml
distribution:
  kind: "triangular"
  low: 0.08
  mode: 0.12
  high: 0.22
```

### Validation Rules

The spec enforces:
- `seed`: 0 ≤ seed ≤ 2^64-1
- `parameters`: Non-empty array
- `parameter.id`: Valid Python identifier
- `parameter.id`: Unique across all parameters
- **Sweep constraints:**
  - `values`: Non-empty, ≤ 100,000 elements
- **Distribution constraints:**
  - Uniform: `low < high`
  - Normal: `std > 0`
  - Lognormal: `sigma > 0`
  - Triangular: `low ≤ mode ≤ high`
- **Monte Carlo:**
  - Required if any distribution parameters
  - `trials`: 1 ≤ trials ≤ 10,000,000

---

## Provenance Fields

### Scenario Provenance Structure

When a scenario runs, GreenLang records:

```json
{
  "scenario": {
    "name": "building_decarb_baseline",
    "spec_hash": "sha256:a1b2c3...",
    "seed_root": 123456789,
    "seed_strategy": "derive-by-path",
    "mode": "replay",
    "spec_version": "1.0.0"
  },
  "randomness": {
    "seed_path": "scenario:building_decarb_baseline|param:electricity_price_usd_per_kwh|trial:42",
    "seed_child": 9876543210123456789,
    "rng_algo": "splitmix64",
    "float_precision": 6
  },
  "execution": {
    "started_at": "2025-10-21T10:30:00Z",
    "finished_at": "2025-10-21T10:45:00Z",
    "status": "success"
  },
  "artifacts": [
    {
      "name": "results",
      "path": "output/results.parquet",
      "sha256": "d4e5f6..."
    }
  ]
}
```

### Reproducibility Contract

Given provenance record, you can **exactly reproduce** the run:

```bash
gl run --scenario scenarios/baseline.yaml \
       --seed-root 123456789 \
       --spec-hash a1b2c3... \
       --mode replay
```

**Guarantees:**
- Byte-exact outputs
- Identical random samples
- Same execution order
- Matching provenance hash

---

## Usage Examples

### Example 1: Simple Parameter Sweep

```yaml
# scenarios/simple_sweep.yaml
schema_version: "1.0.0"
name: "simple_temperature_sweep"
seed: 42
parameters:
  - id: "temperature"
    type: "sweep"
    values: [15, 20, 25, 30]
```

```python
from greenlang.simulation.spec import from_yaml
from greenlang.simulation.runner import ScenarioRunner

runner = ScenarioRunner("scenarios/simple_sweep.yaml")
for params in runner.generate_samples():
    print(f"Temperature: {params['temperature']}")
# Output: 15, 20, 25, 30
```

### Example 2: Monte Carlo Sampling

```yaml
# scenarios/monte_carlo.yaml
schema_version: "1.0.0"
name: "price_sensitivity"
seed: 123
parameters:
  - id: "price"
    type: "distribution"
    distribution:
      kind: "triangular"
      low: 0.08
      mode: 0.12
      high: 0.22
monte_carlo:
  trials: 1000
```

```python
runner = ScenarioRunner("scenarios/monte_carlo.yaml")
prices = [params['price'] for params in runner.generate_samples()]
# 1000 samples from triangular distribution
```

### Example 3: Combined Sweep + Monte Carlo

```yaml
schema_version: "1.0.0"
name: "combined_scenario"
seed: 42
parameters:
  - id: "retrofit"
    type: "sweep"
    values: ["none", "light", "deep"]
  - id: "price"
    type: "distribution"
    distribution:
      kind: "uniform"
      low: 0.10
      high: 0.20
monte_carlo:
  trials: 500
```

**Result:** 3 retrofit levels × 500 trials = 1,500 total samples

### Example 4: Custom RNG Usage

```python
from greenlang.simulation.runner import ScenarioRunner

runner = ScenarioRunner("scenarios/my_scenario.yaml")

# Get custom RNG for additional sampling
custom_rng = runner.get_rng("custom:sensitivity_analysis")

# Generate custom samples
samples = [custom_rng.normal(0, 1) for _ in range(100)]
```

---

## Reproducibility Guide

### How to Reproduce a Run

**Step 1:** Save provenance ledger
```python
runner = ScenarioRunner("scenarios/baseline.yaml")
results = process_scenario(runner)
ledger_path = runner.finalize()
# Saves to: .greenlang/provenance/run_YYYYMMDD_HHMMSS.json
```

**Step 2:** Extract seed and hash
```json
{
  "scenario": {
    "seed_root": 123456789,
    "spec_hash": "sha256:a1b2c3..."
  }
}
```

**Step 3:** Reproduce exactly
```python
# Same seed + same spec = same results
runner = ScenarioRunner("scenarios/baseline.yaml")
assert runner.spec.seed == 123456789
results_v2 = process_scenario(runner)
assert results == results_v2  # Byte-exact match
```

### Validation Checklist

✅ **Same seed:** Verify `seed` field matches
✅ **Same spec:** Verify `spec_hash` matches (use `gl scenario hash`)
✅ **Replay mode:** Ensure `mode="replay"` (no live data)
✅ **Same Python version:** Recommended but not required
✅ **Same OS:** Not required (cross-platform deterministic)

### Debugging Non-Reproducibility

If results don't match:

1. **Check spec hash:**
   ```bash
   gl scenario hash scenarios/baseline.yaml
   ```

2. **Verify seed:**
   ```python
   assert runner.spec.seed == expected_seed
   ```

3. **Check mode:**
   ```python
   assert runner.spec.mode == "replay"
   ```

4. **Compare provenance:**
   ```bash
   diff run1_provenance.json run2_provenance.json
   ```

---

## Advanced Topics

### Parallel Execution

```python
from concurrent.futures import ProcessPoolExecutor
from greenlang.simulation.runner import ScenarioRunner

def run_trial(params):
    return model(**params)

runner = ScenarioRunner("scenarios/large_mc.yaml")
with ProcessPoolExecutor() as executor:
    results = list(executor.map(run_trial, runner.generate_samples()))
```

**Note:** Trials are independent (different substreams), so parallel execution is safe.

### Custom Seed Strategies

```yaml
monte_carlo:
  trials: 1000
  seed_strategy: "derive-by-path"  # Default (recommended)
  # Options:
  # - "derive-by-path": HMAC derivation (stable under changes)
  # - "sequence": Sequential seeds (seed+1, seed+2, ...)
  # - "fixed": Same seed for all trials (not recommended)
```

### Sensitivity Analysis

```yaml
name: "sensitivity_analysis"
seed: 42
parameters:
  - id: "base_case"
    type: "sweep"
    values: [1.0]
  - id: "sensitivity_param"
    type: "distribution"
    distribution:
      kind: "normal"
      mean: 1.0
      std: 0.1
monte_carlo:
  trials: 10000
```

### Integration with Agents

```python
from greenlang.simulation.runner import ScenarioRunner
from greenlang.agents.boiler_agent import BoilerAgent

runner = ScenarioRunner("scenarios/boiler_sizing.yaml")
agent = BoilerAgent()

results = []
for params in runner.generate_samples():
    result = agent.run(params)
    results.append(result)
```

---

## JSON Schema Validation

Validate YAML before loading:

```bash
# Using JSON Schema (in CI)
ajv validate -s greenlang/simulation/_schema/scenario_v1.json \
             -d scenarios/baseline.yaml
```

```python
# Using Python
from greenlang.simulation.spec import validate_spec
import yaml

with open("scenarios/baseline.yaml") as f:
    data = yaml.safe_load(f)

try:
    spec = validate_spec(data)
    print("✓ Valid scenario spec")
except GLValidationError as e:
    print(f"✗ Validation failed: {e.message}")
    print(f"  Code: {e.code}")
    print(f"  Path: {e.path}")
```

---

## CLI Commands

```bash
# Validate scenario
gl scenario validate scenarios/baseline.yaml

# Run scenario
gl scenario run scenarios/baseline.yaml --output results.json

# Compute hash
gl scenario hash scenarios/baseline.yaml
# Output: sha256:a1b2c3...

# Replay from provenance
gl scenario replay .greenlang/provenance/run_20251021_103000.json
```

---

## Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| `GLValidationError.SCENARIO_SCHEMA` | YAML doesn't match schema | Check YAML syntax and required fields |
| `GLValidationError.SCENARIO_SEED_RANGE` | Seed outside 0 to 2^64-1 | Use valid seed value |
| `GLValidationError.SCENARIO_DIST_PARAM` | Invalid distribution params | Check low < high, std > 0, etc. |
| `GLValidationError.CONSTRAINT` | General constraint violation | Read error message for specific constraint |
| `GLValidationError.DUPLICATE_NAME` | Duplicate parameter IDs | Ensure unique parameter IDs |

---

## References

- **SIM-401 Specification:** Scenario Spec & Seeded RNG
- **JSON Schema:** `greenlang/simulation/_schema/scenario_v1.json`
- **Examples:** `docs/scenarios/examples/`
- **API Reference:** `greenlang.simulation.spec`, `greenlang.simulation.rng`

---

## Support

For questions or issues:
- GitHub: https://github.com/greenlang/greenlang/issues
- Docs: https://docs.greenlang.io/scenarios
- Email: support@greenlang.io

---

**Last Updated:** October 2025
**Authors:** GreenLang Simulation & ML Squad
