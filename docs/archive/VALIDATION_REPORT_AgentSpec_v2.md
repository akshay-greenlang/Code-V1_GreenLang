# AgentSpec v2 Schema Compliance Validation Report

**Date:** 2025-11-06
**Session:** 10 (Continuation)
**Task:** Phase 2 STANDARDIZATION - Pack.yaml Schema Refinement

---

## Executive Summary

Completed **Phase 1 of AgentSpec v2 schema compliance refinement**, fixing explicit schema violations across 11 pack.yaml files. Validation testing identified that 1/12 packs now pass full schema validation, with remaining issues related to complex type representation.

**Status:** ✅ Explicit violations fixed, ⚠️ Complex type schema design issues identified for Phase 2

---

## Work Completed

### 1. Schema Analysis
- ✅ Read and analyzed AgentSpec v2 Pydantic schema (greenlang/specs/agentspec_v2.py)
- ✅ Identified 5 categories of explicit schema violations
- ✅ Created automated fix script (fix_pack_schemas.py)

### 2. Automated Fixes Applied (11/12 packs)

**Files modified:**
1. packs/carbon_ai/pack.yaml
2. packs/grid_factor_ai/pack.yaml
3. packs/boiler_replacement_ai/pack.yaml
4. packs/industrial_process_heat_ai/pack.yaml
5. packs/decarbonization_roadmap_ai/pack.yaml
6. packs/industrial_heat_pump_ai/pack.yaml
7. packs/recommendation_ai/pack.yaml
8. packs/report_ai/pack.yaml
9. packs/anomaly_iforest_ai/pack.yaml
10. packs/forecast_sarima_ai/pack.yaml
11. packs/waste_heat_recovery_ai/pack.yaml

**Note:** packs/fuel_ai/pack.yaml already compliant - no changes needed

**Fixes applied:**

#### a. AI Budget Section
- ❌ Removed: `warn_at_usd` (not in schema)
- ✅ Retained: `max_cost_usd` only

#### b. AI Tools Section
- ✅ Added required fields to all tools:
  - `schema_in: {type: object, properties: {}}`
  - `schema_out: {type: object, properties: {}}`
  - `impl: python://greenlang.agents.tools:{tool_name}`
  - `safe: true`

#### c. Provenance Section
- ✅ Renamed: `ef_pinning` → `pin_ef: true`
- ✅ Updated: `gwp_set: "AR6"` → `"AR6GWP100"`
- ✅ Added: `record: [inputs, outputs, factors, code_sha, seed]`
- ❌ Removed: `audit_fields`, `citation_required`, `determinism_required`, `standards`

#### d. Realtime Section
- ✅ Flattened structure:
  ```yaml
  # Before (nested):
  realtime:
    replay_mode:
      enabled: true
      seed: 42
      temperature: 0.0
      deterministic: true

  # After (flat):
  realtime:
    default_mode: replay
    snapshot_path: null
    connectors: []
  ```

#### e. Metadata Section
- ❌ Completely removed (not in AgentSpec v2 schema)

---

## Validation Results

**Script:** `validate_agentspec_v2_packs.py`
**Schema:** greenlang.specs.agentspec_v2.AgentSpecV2 (Pydantic)

### Summary
- ✅ **Passed:** 1/12 (8.3%)
  - fuel_ai
- ❌ **Failed:** 11/12 (91.7%)
  - carbon_ai (18 errors)
  - grid_factor_ai (22 errors)
  - boiler_replacement_ai (24 errors)
  - industrial_process_heat_ai (17 errors)
  - decarbonization_roadmap_ai (14 errors)
  - industrial_heat_pump_ai (14 errors)
  - recommendation_ai (24 errors)
  - report_ai (16 errors)
  - anomaly_iforest_ai (35 errors)
  - forecast_sarima_ai (27 errors)
  - waste_heat_recovery_ai (26 errors)

### Remaining Issues Categories

#### 1. Complex Type Schema Mismatch (CRITICAL)
**Issue:** AgentSpec v2 only supports primitive dtypes in IOField:
- Allowed: 'float32', 'float64', 'int32', 'int64', 'string', 'bool'
- Not allowed: 'list', 'object'

**Impact:** Many inputs/outputs use `dtype: "list"` or `dtype: "object"` for complex data structures

**Example violations:**
```yaml
# Current (FAILS):
compute:
  inputs:
    processes:
      dtype: "list"  # NOT ALLOWED

  outputs:
    portfolio_results:
      dtype: "object"  # NOT ALLOWED
```

**Recommendation:** Schema needs extension to support complex types, or pack.yaml files need redesign to represent complex data as JSON strings

#### 2. Missing Unit Fields (CRITICAL)
**Issue:** All IOField instances require `unit` field, but complex types don't have one

**Example:**
```yaml
outputs:
  waste_heat_sources:
    dtype: "list"
    # Missing: unit field
```

**Recommendation:** Define unit semantics for complex types or use `unit: "1"` for dimensionless

#### 3. Unit Whitelist Violations (HIGH)
**Issue:** Many domain-specific units not in approved climate units whitelist

**Violations found:**
- Temperature: `°C`, `°F` (encoding issue with degree symbol)
- Time: `years`, `hours`
- Area: `sqft`
- Count: `people`
- Energy: `MMBtu/year`, `Btu/hr`
- Cost: `USD/kWh`, `USD/MMBtu`, `USD/ton`
- Dimensionless: `fraction`, `%`

**Recommendation:** Expand climate units whitelist or use unit: "1" for dimensionless ratios

#### 4. Extra Field Violations (MEDIUM)
**Issue:** Output fields have extra fields not permitted by schema

**Examples:**
```yaml
outputs:
  emissions_breakdown:
    dtype: "object"
    items: {...}  # EXTRA - not allowed

  grid_mix:
    properties: {...}  # EXTRA - not allowed

  citations:
    required: false  # EXTRA - not allowed
```

**Recommendation:** Remove extra fields or extend OutputField schema

#### 5. Constraint Violations (LOW)
**Issue:** Fields have `default` value with `required: true`

**Schema rule:** `default` only allowed when `required: false`

**Recommendation:** Fix field definitions to comply with constraint

#### 6. Factors Field Type (LOW)
**Issue:** `factors` field should be dict, currently empty list `[]`

**Fix:** Change to `factors: {}` or define actual factor references

---

## Impact Assessment

### Production Readiness
- ✅ **Explicit schema violations:** Fixed (budget, provenance, realtime, tools, metadata)
- ⚠️ **Complex type support:** Needs schema extension or redesign
- ⚠️ **Unit whitelist:** Needs expansion for industrial/HVAC domains
- ✅ **Validation infrastructure:** Created and tested

### Next Steps (Phase 2 Continuation)

#### Option A: Schema Extension (Recommended)
1. Extend AgentSpec v2 Pydantic schema to support:
   - Complex dtypes ('list', 'object')
   - Nested field definitions for complex types
   - Domain-specific units (temperature, time, area, cost)
2. Re-validate all packs after schema update

#### Option B: Pack.yaml Redesign
1. Redesign complex fields to use JSON string representation
2. Use unit: "1" for all complex/dimensionless fields
3. Remove extra fields not in schema
4. Re-validate all packs

#### Option C: Hybrid Approach
1. Extend schema for critical complex types
2. Expand unit whitelist for common domains
3. Simplify pack.yaml where possible

---

## Files Created/Modified

### Created:
1. `fix_pack_schemas.py` - Automated schema fixer (188 lines)
2. `validate_agentspec_v2_packs.py` - Pydantic-based validator (117 lines)
3. `VALIDATION_REPORT_AgentSpec_v2.md` - This report

### Modified:
- 11 pack.yaml files (explicit violations fixed)

---

## Conclusion

Successfully completed **Phase 1 of AgentSpec v2 schema refinement**, eliminating explicit violations in budget, provenance, realtime, tools, and metadata sections. Validation infrastructure created and tested.

**Identified fundamental schema design mismatch** between pack.yaml complex type usage and AgentSpec v2 Pydantic schema's primitive-only type system. This requires architectural decision before Phase 2 can continue.

**Recommendation:** Convene schema working group to decide on Option A (extend schema), Option B (redesign packs), or Option C (hybrid). Current work provides solid foundation for whichever path is chosen.

---

**Report Author:** Claude Code
**Validation Date:** 2025-11-06
**AgentSpec Version:** 2.0.0
**Packs Validated:** 12 (fuel_ai, carbon_ai, grid_factor_ai, boiler_replacement_ai, industrial_process_heat_ai, decarbonization_roadmap_ai, industrial_heat_pump_ai, recommendation_ai, report_ai, anomaly_iforest_ai, forecast_sarima_ai, waste_heat_recovery_ai)
