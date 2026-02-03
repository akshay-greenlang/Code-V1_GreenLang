# GreenLang Determinism Contract v1.0

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** 2026-02-03

## Overview

The Determinism Contract specifies what aspects of GreenLang execution are guaranteed to produce identical results given the same inputs, and what aspects may vary. This is critical for audit-grade climate compliance where reproducibility builds trust.

**Core Principle:**
> For climate compliance, auditability beats cleverness. Deterministic outputs are worth more than clever optimizations.

## Determinism Tiers

### Tier 1: Guaranteed Stable (MUST be identical)

These outputs MUST be byte-identical given the same inputs and configuration:

| Component | Guarantee | Verification Method |
|-----------|-----------|---------------------|
| Calculation results | Identical to 15 decimal places | Numeric comparison |
| SHA-256 hashes | Byte-identical | Hash comparison |
| run.json structure | Field-for-field identical | JSON diff |
| Factor applications | Same factor -> same result | Golden test |
| Unit conversions | Deterministic to full precision | Golden test |
| Step ordering | Topological order preserved | Graph comparison |

### Tier 2: Functionally Stable (Semantically equivalent)

These outputs are semantically equivalent but may have non-significant variations:

| Component | Guarantee | Acceptable Variation |
|-----------|-----------|---------------------|
| Timestamps | Reproducible with DeterministicClock | Real timestamps vary |
| UUIDs | Reproducible with seeded generator | Random UUIDs vary |
| Floating point | 1e-10 tolerance | Platform-specific precision |
| JSON key ordering | Sorted keys for hashing | Display order may vary |

### Tier 3: Non-Deterministic (May vary)

These are explicitly NOT guaranteed to be deterministic:

| Component | Reason | Mitigation |
|-----------|--------|------------|
| LLM outputs | Model non-determinism | Audit trail + manual review |
| External APIs | Network/service variability | Caching + mocking |
| Wall-clock time | System-dependent | DeterministicClock for tests |
| Random operations | Inherently random | Seeding + documentation |

## What IS Guaranteed Stable

### 1. Calculation Outputs

**Guarantee:** Given identical inputs and emission factor versions, all calculations produce identical results.

```python
# Example: Emissions calculation
input_1 = {"fuel_type": "diesel", "amount": 100, "unit": "liters"}
input_2 = {"fuel_type": "diesel", "amount": 100, "unit": "liters"}

result_1 = calculate_emissions(input_1, ef_version="DEFRA-2024")
result_2 = calculate_emissions(input_2, ef_version="DEFRA-2024")

assert result_1 == result_2  # MUST be true
assert result_1["co2e_kg"] == Decimal("265.72")  # Exact value
```

**Implementation Requirements:**
- Use `FinancialDecimal` for all monetary/emission values
- Specify precision explicitly (e.g., 15 decimal places)
- Document rounding rules

### 2. Provenance Hashes

**Guarantee:** SHA-256 hashes for artifacts are computed deterministically.

```python
def compute_provenance_hash(data: dict) -> str:
    """Compute deterministic hash of data."""
    # Sort keys for deterministic ordering
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
```

**Hash Stability Rules:**
- JSON keys MUST be sorted alphabetically
- Numbers MUST use consistent string representation
- No whitespace variations
- UTF-8 encoding only

### 3. run.json Artifact

**Guarantee:** run.json is byte-stable for identical runs.

```python
# Required field ordering for determinism
RUN_JSON_FIELD_ORDER = [
    "schema_version",
    "run_id",
    "pipeline",
    "status",
    "success",
    "started_at",
    "completed_at",
    "duration_ms",
    "inputs",
    "outputs",
    "steps",
    "errors",
    "provenance"
]
```

**Serialization Rules:**
- Use `sort_keys=True` for JSON serialization
- Use `separators=(',', ':')` for compact format
- Timestamps in ISO 8601 format with UTC timezone
- Numbers as strings for exact precision

### 4. Factor Applications

**Guarantee:** Emission factor lookups and applications are deterministic.

```python
# Factor lookup is deterministic
factor = get_emission_factor(
    source="DEFRA",
    year=2024,
    fuel_type="diesel",
    unit="kg_CO2e_per_liter"
)
assert factor == Decimal("2.6572")  # Always this exact value

# Application is deterministic
emissions = apply_factor(amount=100, factor=factor)
assert emissions == Decimal("265.72")  # Always this exact value
```

### 5. Unit Conversions

**Guarantee:** Unit conversions are exact and reversible.

```python
# Conversion is deterministic
liters = convert_units(100, "gallons", "liters")
assert liters == Decimal("378.5411784")  # Exact conversion factor

# Conversion is reversible (within precision)
gallons_back = convert_units(liters, "liters", "gallons")
assert abs(gallons_back - Decimal("100")) < Decimal("1e-10")
```

## What is NOT Guaranteed Stable

### 1. LLM Outputs

**Reality:** Even with temperature=0 and seed, LLM outputs can vary across:
- Model versions
- API endpoint regions
- Time of invocation
- Batching/scheduling differences

**Mitigation Strategy:**

```python
class LLMOutputHandler:
    """Handle non-deterministic LLM outputs safely."""

    def invoke_llm(self, prompt: str, context: dict) -> LLMResult:
        result = self.llm.generate(
            prompt=prompt,
            temperature=0,
            seed=42,
            model_version=self.pinned_model
        )

        # Record everything for audit
        return LLMResult(
            output=result.text,
            audit_record={
                "prompt_hash": hash(prompt),
                "model_id": result.model_id,
                "model_version": result.model_version,
                "timestamp": datetime.utcnow().isoformat(),
                "temperature": 0,
                "seed": 42,
                "token_count": result.usage.total_tokens
            }
        )
```

**Rules for LLM Usage:**
1. NEVER use LLM-generated numbers directly in calculations
2. ALWAYS record prompt, response, and model version
3. ALWAYS mark LLM-derived outputs as non-deterministic
4. USE LLM for summarization, not computation

### 2. External API Responses

**Reality:** External APIs (weather, grid factors, prices) return different data over time.

**Mitigation Strategy:**

```python
class ExternalDataHandler:
    """Cache and version external data for reproducibility."""

    def fetch_with_cache(self, url: str, cache_key: str) -> dict:
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Fetch fresh data
        response = requests.get(url)
        data = response.json()

        # Cache with timestamp and hash
        self.cache.set(
            cache_key,
            data,
            metadata={
                "fetched_at": datetime.utcnow().isoformat(),
                "url": url,
                "hash": hash(json.dumps(data, sort_keys=True))
            }
        )
        return data
```

### 3. Timestamps

**For Production:** Use real timestamps
**For Tests:** Use DeterministicClock

```python
class DeterministicClock:
    """Deterministic clock for reproducible tests."""

    def __init__(self, start_time: datetime = None):
        self._time = start_time or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self._tick = timedelta(seconds=1)

    def now(self) -> datetime:
        """Return current deterministic time."""
        return self._time

    def advance(self, delta: timedelta = None):
        """Advance time by specified amount."""
        self._time += delta or self._tick

# Usage in tests
@pytest.fixture
def deterministic_clock():
    return DeterministicClock(datetime(2026, 2, 3, 12, 0, 0, tzinfo=timezone.utc))

def test_pipeline_execution(deterministic_clock):
    orchestrator = Orchestrator(clock=deterministic_clock)
    result = orchestrator.execute(pipeline, input_data)

    # Timestamps are now deterministic
    assert result["started_at"] == "2026-02-03T12:00:00+00:00"
```

## Determinism Verification

### Golden Test Pattern

```python
def test_emissions_calculation_determinism():
    """Verify emissions calculation is deterministic."""
    input_data = load_golden_input("test_data/emissions_input.json")
    expected_output = load_golden_output("test_data/emissions_expected.json")

    result = calculate_emissions(input_data)

    # Exact match required
    assert result == expected_output

    # Hash must match
    assert compute_hash(result) == compute_hash(expected_output)
```

### Hash Comparison

```python
def verify_run_determinism(run1_path: str, run2_path: str) -> bool:
    """Verify two runs produced identical deterministic outputs."""
    run1 = load_run(run1_path)
    run2 = load_run(run2_path)

    # Compare deterministic fields only
    deterministic_fields = ["outputs", "provenance.hash", "steps.*.outputs"]

    for field in deterministic_fields:
        value1 = get_field(run1, field)
        value2 = get_field(run2, field)
        if value1 != value2:
            return False

    return True
```

### Byte-Stable Verification

```python
def verify_byte_stability(artifact_path: str, expected_hash: str) -> bool:
    """Verify artifact is byte-identical to expected."""
    with open(artifact_path, 'rb') as f:
        content = f.read()

    actual_hash = hashlib.sha256(content).hexdigest()
    return actual_hash == expected_hash
```

## Implementation Requirements

### For Agent Developers

1. **Use Deterministic Types**
   ```python
   from decimal import Decimal
   from greenlang.types import FinancialDecimal

   # Good: Exact decimal arithmetic
   emissions = FinancialDecimal("265.72")

   # Bad: Floating point
   emissions = 265.72  # May vary by platform
   ```

2. **Handle Randomness Explicitly**
   ```python
   import random

   class DeterministicAgent(BaseAgent):
       def __init__(self, seed: int = 42):
           self.rng = random.Random(seed)

       def run(self, input_data):
           # Reproducible random if needed
           value = self.rng.random()
   ```

3. **Document Non-Determinism**
   ```python
   class LLMSummaryAgent(BaseAgent):
       """
       Generate text summaries using LLM.

       DETERMINISM: NON-DETERMINISTIC
       - LLM outputs may vary between runs
       - Use for display only, not calculations
       - All outputs recorded for audit
       """
   ```

### For Pipeline Authors

1. **Mark Non-Deterministic Steps**
   ```yaml
   steps:
     - name: calculate-emissions
       agent: emissions-calculator
       deterministic: true  # Required

     - name: generate-summary
       agent: llm-summarizer
       deterministic: false  # Explicit marking
       audit_required: true
   ```

2. **Separate Deterministic and Non-Deterministic**
   ```yaml
   # Deterministic section (auditable)
   steps:
     - name: normalize
     - name: calculate
     - name: validate

   # Non-deterministic section (informational)
   post_steps:
     - name: summarize
       deterministic: false
     - name: suggest-improvements
       deterministic: false
   ```

### For Platform Operators

1. **Pin Dependencies**
   ```
   # requirements.txt - exact versions
   pandas==2.0.0
   numpy==1.24.0
   greenlang==0.3.0
   ```

2. **Control Environment**
   ```dockerfile
   FROM python:3.10-slim

   # Pin system locale for consistent string sorting
   ENV LANG=en_US.UTF-8
   ENV LC_ALL=en_US.UTF-8

   # Pin timezone for consistent time handling
   ENV TZ=UTC
   ```

## LLM Determinism Contract

### Best Effort Settings

```python
llm_config = {
    "temperature": 0,          # Minimum randomness
    "seed": 42,                # Fixed seed
    "top_p": 1.0,              # No nucleus sampling
    "model": "gpt-4-0613",     # Pinned version
}
```

### Audit Trail Requirements

For any LLM invocation, record:

```json
{
  "invocation_id": "uuid",
  "timestamp": "2026-02-03T12:00:00Z",
  "model": {
    "provider": "openai",
    "model_id": "gpt-4-0613",
    "api_version": "2024-01-01"
  },
  "request": {
    "prompt_hash": "sha256:abc...",
    "temperature": 0,
    "seed": 42,
    "max_tokens": 1000
  },
  "response": {
    "output_hash": "sha256:def...",
    "token_count": 150,
    "finish_reason": "stop"
  }
}
```

### When LLM Output Differs

If an LLM produces different output on retry:

1. **Log the discrepancy**
2. **Use the original output** (first invocation wins)
3. **Flag for human review** if difference is significant
4. **Never use LLM output for calculations**

## Verification Commands

```bash
# Verify a run is deterministic
gl verify --determinism run.json

# Compare two runs
gl verify --compare run1.json run2.json

# Run golden tests
gl test --golden tests/golden/

# Generate determinism report
gl verify --report run.json > determinism_report.json
```

## Appendix A: Determinism Checklist

### Before Release

- [ ] All calculations use FinancialDecimal
- [ ] All JSON serialization uses sort_keys=True
- [ ] All timestamps use UTC with ISO 8601 format
- [ ] All random operations are seeded
- [ ] All LLM outputs are marked non-deterministic
- [ ] Golden tests pass
- [ ] Hash comparison tests pass

### For Each Pipeline

- [ ] Deterministic steps marked `deterministic: true`
- [ ] Non-deterministic steps marked `deterministic: false`
- [ ] External data cached with version
- [ ] LLM usage has audit trail
- [ ] Unit tests include determinism verification

## Appendix B: Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Dict ordering | Hash changes | Use sort_keys=True |
| Float precision | Values differ slightly | Use Decimal |
| Timestamp variation | Times differ | Use DeterministicClock |
| Random without seed | Different results | Seed all RNGs |
| LLM in calculations | Audit fails | Use tools only |
| Locale differences | String sorting varies | Pin LC_ALL |
