# Citation Implementation Guide for AI Agents

**Status:** Infrastructure Created, Implementation Pending
**Date:** October 22, 2025
**Completion:** 60% (Structure + Types complete, Agent integration pending)

---

## What's Been Completed

### 1. Citation Data Structures ✅ COMPLETE

**File:** `greenlang/agents/citations.py` (398 lines)

**Structures Created:**

#### EmissionFactorCitation
Complete citation tracking for emission factors with:
- `source`: Data source name (e.g., "EPA eGRID 2025")
- `factor_name`: Descriptive name
- `value` + `unit`: Numeric value with unit
- **`ef_cid`**: Content identifier for verification
- `version`: Data version
- `last_updated`: Timestamp
- `confidence`: high/medium/low
- `region`: Geographic region
- `gwp_set`: GWP set used (AR6GWP100, etc.)
- `formatted()` method: Human-readable output

#### CalculationCitation
Tracks calculation steps:
- `step_name`: Calculation name
- `formula`: Formula used
- `inputs` + `output`: Values tracked
- `tool_call_id`: Links to runtime

#### DataSourceCitation
External data source tracking:
- `source_name` + `source_type`
- `query`: Parameters used
- `checksum`: SHA-256 of data
- `url`: Source URL

#### CitationBundle
Aggregates all citations:
- `emission_factors`: List of EF citations
- `calculations`: List of calc citations
- `data_sources`: List of data citations
- `to_dict()`: JSON serialization
- `formatted_summary()`: Multi-line display

**Helper Functions:**
- `generate_ef_cid()`: Creates deterministic CID
- `create_emission_factor_citation()`: Convenience constructor

### 2. Type Definitions Updated ✅ COMPLETE

**File:** `greenlang/agents/types.py`

**Added Citations Field:**
```python
class FuelOutput(TypedDict):
    # ... existing fields ...
    citations: NotRequired[list]  # List of EmissionFactorCitation objects
```

### 3. Provenance Metadata Added ✅ COMPLETE

All 3 agents now export provenance in metadata:

**FuelAgent+AI** (`fuel_agent_ai.py:420-422`):
```python
"seed": 42,  # Reproducibility seed
"temperature": 0.0,  # Deterministic temperature
"deterministic": True,  # Deterministic execution flag
```

**CarbonAgent+AI** (`carbon_agent_ai.py:564-566`):
```python
"seed": 42,  # Reproducibility seed
"temperature": 0.0,  # Deterministic temperature
"deterministic": True,
```

**GridFactorAgent+AI** (`grid_factor_agent_ai.py:648-650`):
```python
"seed": 42,  # Reproducibility seed
"temperature": 0.0,  # Deterministic temperature
"deterministic": True,
```

---

## What Still Needs Implementation

### Task 1: Integrate Citations into FuelAgent+AI

**Location:** `greenlang/agents/fuel_agent_ai.py`

**Required Changes:**

#### Step 1: Import Citation Classes
```python
# Add to imports (around line 41)
from .citations import (
    EmissionFactorCitation,
    CitationBundle,
    create_emission_factor_citation,
)
```

#### Step 2: Track Citations in Tools

**In `_calculate_emissions_impl()` (around line 278):**
```python
# After calculating emissions
emission_factor_value = 5.31  # Example for natural gas
citation = create_emission_factor_citation(
    source="EPA Emission Factors",
    factor_name=f"{fuel_type} Combustion",
    value=emission_factor_value,
    unit="kgCO2e/therm",
    version="2025.1",
    last_updated=datetime(2025, 1, 15),
    confidence="high",
    gwp_set="AR6GWP100"
)

# Store in instance variable
self._current_citations = [citation]

# Return in tool result
return {
    "emissions_kg_co2e": emissions,
    "emission_factor": emission_factor_value,
    "emission_factor_unit": "kgCO2e/therm",
    "citations": [citation.to_dict()],  # Add this
}
```

#### Step 3: Add Citations to Output

**In `_build_output()` (around line 615):**
```python
output: FuelOutput = {
    # ... existing fields ...
    "citations": [c.to_dict() for c in self._current_citations],  # Add this
}
```

**Estimated Effort:** 2-3 hours

---

### Task 2: Integrate Citations into CarbonAgent+AI

**Location:** `greenlang/agents/carbon_agent_ai.py`

**Required Changes:**

#### Step 1: Import Citation Classes
```python
from .citations import (
    EmissionFactorCitation,
    CitationBundle,
    CalculationCitation,
    create_emission_factor_citation,
)
```

#### Step 2: Track Citations in Aggregation

**In `_aggregate_emissions_impl()` (around line 245):**
```python
# For each emission source in the aggregation
citations = []
for emission in emissions_list:
    citation = create_emission_factor_citation(
        source=emission.get("source", "Unknown"),
        factor_name=f"{emission['scope']} Emissions",
        value=emission["emissions_kg"],
        unit="kgCO2e",
        version=emission.get("version", "unknown"),
        confidence="high",
    )
    citations.append(citation)

# Add calculation citation
calc_citation = CalculationCitation(
    step_name="aggregate_emissions",
    formula="sum(scope1, scope2, scope3)",
    inputs={"num_sources": len(emissions_list)},
    output={"value": total_emissions, "unit": "kgCO2e"},
)

# Return in tool result
return {
    "total_emissions_kg": total_emissions,
    "citations": [c.to_dict() for c in citations],
    "calculation": calc_citation.dict(),
}
```

#### Step 3: Update Output Types

**In `types.py`:**
```python
class CarbonOutput(TypedDict):
    # ... existing fields ...
    citations: NotRequired[list]
```

**Estimated Effort:** 2-3 hours

---

### Task 3: Integrate Citations into GridFactorAgent+AI

**Location:** `greenlang/agents/grid_factor_agent_ai.py`

**Required Changes:**

#### Step 1: Import Citation Classes
```python
from .citations import (
    EmissionFactorCitation,
    DataSourceCitation,
    CitationBundle,
    create_emission_factor_citation,
)
```

#### Step 2: Track Citations in Lookup

**In `_lookup_grid_intensity_impl()` (around line 190):**
```python
# After looking up grid intensity
citation = create_emission_factor_citation(
    source="EPA eGRID 2025",
    factor_name=f"Grid Intensity - {country}",
    value=grid_intensity,
    unit="kgCO2e/kWh",
    version="2025.1",
    last_updated=datetime(2025, 1, 15),
    confidence="high",
    region=country,
)

# Return in tool result
return {
    "grid_intensity": grid_intensity,
    "unit": "kgCO2e/kWh",
    "country": country,
    "citation": citation.to_dict(),  # Add this
}
```

#### Step 3: Add to Output

**In `_build_output()` (around line 730):**
```python
# Collect all citations from tool results
citations = []
if "lookup" in tool_results:
    citations.append(tool_results["lookup"].get("citation"))

output: GridFactorOutput = {
    # ... existing fields ...
    "citations": [c for c in citations if c],  # Filter None values
}
```

**Estimated Effort:** 2-3 hours

---

### Task 4: Create Tests for Citation Tracking

**New File:** `tests/agents/test_citations.py`

```python
"""
Tests for citation tracking in AI agents.

Verifies that:
1. Citations are present in all outputs
2. EF CIDs are generated correctly
3. Citation bundles aggregate properly
4. Formatted output is human-readable
"""

import pytest
from datetime import datetime
from greenlang.agents.citations import (
    EmissionFactorCitation,
    generate_ef_cid,
    create_emission_factor_citation,
    CitationBundle,
)


def test_generate_ef_cid_deterministic():
    """EF CID generation is deterministic."""
    cid1 = generate_ef_cid("EPA", "Natural Gas", 5.31, "kgCO2e/therm", "2025.1")
    cid2 = generate_ef_cid("EPA", "Natural Gas", 5.31, "kgCO2e/therm", "2025.1")
    assert cid1 == cid2
    assert cid1.startswith("ef_")
    assert len(cid1) == 19  # "ef_" + 16 hex chars


def test_create_emission_factor_citation():
    """Citation creation includes all required fields."""
    citation = create_emission_factor_citation(
        source="EPA eGRID 2025",
        factor_name="US Grid Average",
        value=0.385,
        unit="kgCO2e/kWh",
        version="2025.1",
        confidence="high",
    )

    assert citation.source == "EPA eGRID 2025"
    assert citation.value == 0.385
    assert citation.unit == "kgCO2e/kWh"
    assert citation.ef_cid.startswith("ef_")
    assert citation.confidence == "high"


def test_citation_formatted_output():
    """Formatted citation is human-readable."""
    citation = create_emission_factor_citation(
        source="EPA eGRID",
        factor_name="US Average",
        value=0.385,
        unit="kgCO2e/kWh",
        version="2025.1",
        region="US",
        confidence="high",
        last_updated=datetime(2025, 1, 15),
    )

    formatted = citation.formatted()
    assert "EPA eGRID" in formatted
    assert "v2025.1" in formatted
    assert "(US)" in formatted
    assert "0.385 kgCO2e/kWh" in formatted
    assert "Updated: 2025-01-15" in formatted
    assert "Confidence: high" in formatted


def test_citation_bundle_aggregation():
    """Citation bundles aggregate multiple citations."""
    citation1 = create_emission_factor_citation(
        source="EPA", factor_name="Natural Gas", value=5.31, unit="kgCO2e/therm"
    )
    citation2 = create_emission_factor_citation(
        source="EPA", factor_name="Electricity", value=0.385, unit="kgCO2e/kWh"
    )

    bundle = CitationBundle(
        agent_id="test_agent",
        emission_factors=[citation1, citation2],
    )

    assert len(bundle.emission_factors) == 2
    assert bundle.agent_id == "test_agent"

    # Test serialization
    bundle_dict = bundle.to_dict()
    assert bundle_dict["total_citations"] == 2
    assert len(bundle_dict["emission_factors"]) == 2


def test_fuel_agent_output_has_citations():
    """FuelAgent output includes citations (integration test)."""
    # TODO: Implement after citation integration
    pass


def test_carbon_agent_output_has_citations():
    """CarbonAgent output includes citations (integration test)."""
    # TODO: Implement after citation integration
    pass


def test_grid_factor_agent_output_has_citations():
    """GridFactorAgent output includes citations (integration test)."""
    # TODO: Implement after citation integration
    pass
```

**Estimated Effort:** 1-2 hours

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)

1. ✅ **DONE:** Create citation data structures
2. ✅ **DONE:** Add citations field to type definitions
3. ✅ **DONE:** Add seed/temperature to metadata
4. **TODO:** Integrate citations into FuelAgent+AI (2-3 hours)
5. **TODO:** Integrate citations into CarbonAgent+AI (2-3 hours)
6. **TODO:** Integrate citations into GridFactorAgent+AI (2-3 hours)

### Phase 2: Testing & Validation (0.5-1 day)

7. **TODO:** Create citation tests (1-2 hours)
8. **TODO:** Update existing agent tests to verify citations (1-2 hours)
9. **TODO:** Run full test suite to verify no regressions

### Phase 3: Documentation (0.5 day)

10. **TODO:** Add citation examples to agent READMEs
11. **TODO:** Update API documentation
12. **TODO:** Create citation usage guide for developers

---

## Total Estimated Effort

**Completed:** ~8 hours (structure + types + metadata)
**Remaining:** ~12-16 hours (integration + tests + docs)
**Total:** ~20-24 hours (2.5-3 days)

---

## Example: Complete Citation Flow

### Input
```python
payload = {
    "fuel_type": "natural_gas",
    "amount": 1000,
    "unit": "therms",
}
```

### Processing
```python
# Tool calculates emissions
emission_factor = 5.31  # kgCO2e/therm

# Citation created
citation = create_emission_factor_citation(
    source="EPA Emission Factors",
    factor_name="Natural Gas Combustion",
    value=5.31,
    unit="kgCO2e/therm",
    version="2025.1",
    confidence="high",
)

# EF CID generated: ef_a1b2c3d4e5f6g7h8
```

### Output
```json
{
  "success": true,
  "data": {
    "co2e_emissions_kg": 5310.0,
    "fuel_type": "natural_gas",
    "consumption_amount": 1000,
    "consumption_unit": "therms",
    "emission_factor": 5.31,
    "emission_factor_unit": "kgCO2e/therm",
    "citations": [
      {
        "source": "EPA Emission Factors",
        "factor_name": "Natural Gas Combustion",
        "value": 5.31,
        "unit": "kgCO2e/therm",
        "ef_cid": "ef_a1b2c3d4e5f6g7h8",
        "version": "2025.1",
        "confidence": "high",
        "formatted": "EPA Emission Factors v2025.1: Natural Gas Combustion = 5.31 kgCO2e/therm (Confidence: high)"
      }
    ]
  },
  "metadata": {
    "seed": 42,
    "temperature": 0.0,
    "deterministic": true,
    "ai_calls": 1,
    "tool_calls": 2,
    "total_cost_usd": 0.00465
  }
}
```

---

## Benefits of This Implementation

### 1. Complete Audit Trail
- Every number traces back to its source
- EF CIDs enable verification
- Calculation steps documented

### 2. Reproducibility
- Seed exported in metadata
- Citations include version info
- Deterministic execution guaranteed

### 3. Compliance
- Matches RAG citation quality
- Follows "Using Tools, Not Guessing" principle
- Meets DOC-601 requirements

### 4. User Trust
- Transparent data sources
- Confidence levels explicit
- Human-readable formatted output

---

## Next Steps

**Immediate (This Week):**
1. Implement citation tracking in FuelAgent+AI (Task 1)
2. Implement citation tracking in CarbonAgent+AI (Task 2)
3. Implement citation tracking in GridFactorAgent+AI (Task 3)

**Follow-Up (Next Week):**
4. Create comprehensive tests (Task 4)
5. Update documentation with citation examples
6. Verify all tests pass

**Optional (When Time Permits):**
7. Add EF CID database for lookup/verification
8. Create citation visualization tools
9. Implement citation export formats (BibTeX, etc.)

---

**Document Status:** Implementation Guide
**Created:** October 22, 2025
**Completion:** 60% (Infrastructure complete, integration pending)
**Estimated Time to 100%:** 2.5-3 days
