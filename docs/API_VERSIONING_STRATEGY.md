# FuelAgentAI API Versioning Strategy

**Version:** 1.0
**Date:** 2025-10-24
**Owner:** GreenLang Framework Team
**Status:** Design Phase

---

## Executive Summary

This document defines the API versioning strategy for FuelAgentAI v2, ensuring **zero breaking changes** for existing customers while introducing enhanced capabilities (multi-gas reporting, provenance tracking, data quality scoring).

**Key Decisions:**
- ✅ **Dual API support:** v1 (legacy) and v2 (enhanced) run in parallel
- ✅ **Backward compatibility:** v1 API unchanged for 12+ months
- ✅ **Progressive enhancement:** Clients opt-in to v2 features
- ✅ **Graceful deprecation:** 6-month notice before v1 sunset
- ✅ **Feature flags:** Per-request control of output format

**Timeline:**
- **2025-Q4:** Launch v2 API (both v1 and v2 supported)
- **2026-Q2:** Deprecation notice for v1 API
- **2026-Q3:** Sunset v1 API (12 months after v2 launch)

---

## 1. Current State (v1 API)

### v1 Input Schema
```json
{
  "fuel_type": "diesel",
  "amount": 1000,
  "unit": "gallons",
  "country": "US",
  "year": 2024,
  "renewable_percentage": 0,
  "efficiency": 1.0,
  "scope": "1",
  "location": "California",
  "metadata": {}
}
```

**Required fields:** `fuel_type`, `amount`, `unit`
**Optional fields:** `country` (default: "US"), `year` (default: current year), etc.

### v1 Output Schema
```json
{
  "co2e_emissions_kg": 10210.0,
  "fuel_type": "diesel",
  "consumption_amount": 1000.0,
  "consumption_unit": "gallons",
  "emission_factor": 10.21,
  "emission_factor_unit": "kgCO2e/gallons",
  "country": "US",
  "scope": "1",
  "energy_content_mmbtu": 138.7,
  "renewable_offset_applied": false,
  "efficiency_adjusted": false,
  "recommendations": [],
  "explanation": "Calculated 10,210 kg CO2e emissions...",
  "metadata": {
    "agent_id": "fuel_ai",
    "calculation_time_ms": 125.5,
    "ai_calls": 1,
    "tool_calls": 1,
    "total_cost_usd": 0.0025
  }
}
```

### v1 Limitations
- ❌ Single CO2e value (no CH4/N2O breakdown)
- ❌ No emission factor source attribution
- ❌ No data quality indicators
- ❌ No uncertainty ranges
- ❌ No compliance markers (CSRD, CDP)
- ❌ No factor versioning
- ❌ No licensing information

---

## 2. Enhanced API (v2)

### v2 Input Schema (Superset of v1)
```json
{
  // ========== v1 FIELDS (UNCHANGED) ==========
  "fuel_type": "diesel",
  "amount": 1000,
  "unit": "gallons",
  "country": "US",
  "year": 2024,
  "renewable_percentage": 0,
  "efficiency": 1.0,
  "scope": "1",
  "location": "California",
  "metadata": {},

  // ========== v2 ENHANCEMENTS (OPTIONAL) ==========
  "region_hint": "US-CA",           // Sub-national region
  "scope2_mode": "location",        // "location" | "market" (for electricity)
  "boundary": "combustion",         // "combustion" | "WTT" | "WTW"
  "gwp_set": "IPCC_AR6_100",        // "IPCC_AR6_100" | "IPCC_AR6_20"
  "heating_value_basis": "HHV",     // "HHV" | "LHV"
  "temp_C": 15.0,                   // Reference temperature
  "biogenic_share_pct": 0,          // Biomass blend percentage
  "recs_pct": 0,                    // Renewable energy certificates
  "vintage": "latest",              // Factor version selection
  "calculation_id": "calc_123",     // Client tracking ID

  // ========== OUTPUT CONTROL ==========
  "response_format": "enhanced"     // "legacy" | "enhanced" | "compact"
}
```

**Key Design Principles:**
1. **All v1 fields remain valid** - No breaking changes
2. **All v2 fields are optional** - Defaults maintain v1 behavior
3. **`response_format` controls output** - Clients choose v1 or v2 response

### v2 Output Schema (Enhanced)
```json
{
  // ========== v1 FIELDS (UNCHANGED) ==========
  "co2e_emissions_kg": 10210.0,
  "fuel_type": "diesel",
  "consumption_amount": 1000.0,
  "consumption_unit": "gallons",
  "emission_factor": 10.21,
  "emission_factor_unit": "kgCO2e/gallons",
  "country": "US",
  "scope": "1",
  "energy_content_mmbtu": 138.7,
  "renewable_offset_applied": false,
  "efficiency_adjusted": false,
  "recommendations": [],
  "explanation": "...",
  "metadata": {},

  // ========== v2 ENHANCEMENTS ==========
  "vectors_kg": {
    "CO2": 10180.0,
    "CH4": 0.82,
    "N2O": 0.164,
    "biogenic_CO2": 0.0
  },

  "co2e_by_gwp": {
    "co2e_100yr": 10210.0,
    "co2e_20yr": 10256.0
  },

  "factor_record": {
    "factor_id": "EF:US:diesel:2024:v1",
    "source_org": "EPA",
    "source_publication": "Emission Factors for GHG Inventories 2024",
    "source_year": 2024,
    "methodology": "IPCC_Tier_1",
    "valid_from": "2024-01-01",
    "valid_to": "2024-12-31",
    "gwp_set": "IPCC_AR6_100",
    "boundary": "combustion",
    "citation": "EPA (2024). Emission Factors for Greenhouse Gas Inventories. URL: https://..."
  },

  "quality": {
    "uncertainty_95ci_pct": 5.0,
    "dqs": {
      "temporal": 5,
      "geographical": 4,
      "technological": 4,
      "representativeness": 4,
      "methodological": 5,
      "overall_score": 4.4,
      "rating": "high_quality"
    }
  },

  "compliance": {
    "frameworks": ["GHG_Protocol", "IPCC_2006", "EPA_MRR"],
    "csrd_compliant": true,
    "cdp_reportable": true
  },

  "provenance_hash": "sha256:abc123..."
}
```

**Response Format Options:**

1. **`response_format: "legacy"`** (default for v1 clients)
   - Returns ONLY v1 fields
   - Exactly matches current v1 output
   - No breaking changes

2. **`response_format: "enhanced"`** (full v2 features)
   - Returns v1 fields + v2 enhancements
   - Multi-gas breakdown
   - Full provenance and quality metadata

3. **`response_format: "compact"`** (minimal for mobile/IoT)
   - Returns only essential fields
   - Reduces payload size ~60%

---

## 3. Versioning Approaches (Evaluated)

### Approach A: URL Versioning ❌ (Rejected)
```
POST /api/v1/fuel/calculate  # Legacy
POST /api/v2/fuel/calculate  # Enhanced
```

**Pros:**
- Clear separation
- Easy to deprecate old version

**Cons:**
- **Code duplication** (maintain two endpoints)
- **Data duplication** (two sets of tests)
- URL proliferation (v3, v4, ...)
- Doesn't support progressive enhancement

**Decision:** REJECTED (too rigid)

---

### Approach B: Header Versioning ❌ (Rejected)
```
POST /api/fuel/calculate
Headers:
  API-Version: 1
  API-Version: 2
```

**Pros:**
- Single endpoint
- Clean URL

**Cons:**
- **Hidden versioning** (not discoverable)
- Caching issues (Vary: API-Version)
- Client libraries often don't expose headers easily

**Decision:** REJECTED (poor discoverability)

---

### Approach C: Content Negotiation ❌ (Rejected)
```
POST /api/fuel/calculate
Headers:
  Accept: application/vnd.greenlang.fuel.v1+json
  Accept: application/vnd.greenlang.fuel.v2+json
```

**Pros:**
- RESTful standard
- Versioning in media type

**Cons:**
- **Overly complex** for simple use case
- Poor tooling support (Postman, curl)
- Confusing for developers

**Decision:** REJECTED (overkill)

---

### Approach D: Request Parameter Versioning ✅ (SELECTED)
```
POST /api/fuel/calculate
Body:
{
  "fuel_type": "diesel",
  "amount": 1000,
  "unit": "gallons",
  "response_format": "enhanced"  // "legacy" | "enhanced" | "compact"
}
```

**Pros:**
- ✅ **Single endpoint** (no code duplication)
- ✅ **Progressive enhancement** (clients opt-in)
- ✅ **Backward compatible** (default = "legacy")
- ✅ **Self-documenting** (version in request body)
- ✅ **Feature flags** (granular control)
- ✅ **Easy testing** (same endpoint, different params)

**Cons:**
- ⚠️ Requires input validation (ensure valid format)
- ⚠️ Output schema varies (clients must handle)

**Decision:** SELECTED (best balance)

---

## 4. Implementation Strategy

### Phase 1: Dual Support (Weeks 1-6)
Implement v2 features while maintaining v1 compatibility.

```python
# greenlang/agents/fuel_agent_ai.py

def run(self, payload: Union[FuelInput, FuelInputV2]) -> Union[FuelOutput, FuelOutputV2]:
    """
    Run fuel emissions calculation.

    Automatically detects v1 vs v2 input and returns appropriate output.
    """

    # Detect response format (default: legacy for v1 compatibility)
    response_format = payload.get("response_format", "legacy")

    # Parse input (v2 is superset of v1)
    input_data = self._parse_input_v2(payload)

    # Run calculation (same engine for both)
    calculation_result = self._calculate(input_data)

    # Format output based on response_format
    if response_format == "legacy":
        return self._format_output_v1(calculation_result)
    elif response_format == "enhanced":
        return self._format_output_v2(calculation_result)
    elif response_format == "compact":
        return self._format_output_compact(calculation_result)
    else:
        raise ValueError(f"Invalid response_format: {response_format}")


def _format_output_v1(self, result: CalculationResult) -> FuelOutput:
    """Format output as v1 (legacy) - backward compatible"""
    return {
        "co2e_emissions_kg": result.co2e_100yr,
        "fuel_type": result.fuel_type,
        "consumption_amount": result.amount,
        "consumption_unit": result.unit,
        "emission_factor": result.factor_co2e,  # Aggregated CO2e only
        "emission_factor_unit": f"kgCO2e/{result.unit}",
        "country": result.country,
        "scope": result.scope,
        "energy_content_mmbtu": result.energy_content,
        "renewable_offset_applied": result.renewable_offset > 0,
        "efficiency_adjusted": result.efficiency != 1.0,
        "recommendations": result.recommendations,
        "explanation": result.explanation,
        "metadata": result.metadata
    }


def _format_output_v2(self, result: CalculationResult) -> FuelOutputV2:
    """Format output as v2 (enhanced) - full features"""
    output_v1 = self._format_output_v1(result)  # Include all v1 fields

    # Add v2 enhancements
    output_v1.update({
        "vectors_kg": {
            "CO2": result.co2_kg,
            "CH4": result.ch4_kg,
            "N2O": result.n2o_kg,
            "biogenic_CO2": result.biogenic_co2_kg
        },
        "co2e_by_gwp": {
            "co2e_100yr": result.co2e_100yr,
            "co2e_20yr": result.co2e_20yr if result.co2e_20yr else None
        },
        "factor_record": {
            "factor_id": result.factor.factor_id,
            "source_org": result.factor.provenance.source_org,
            "source_publication": result.factor.provenance.source_publication,
            "source_year": result.factor.provenance.source_year,
            "methodology": result.factor.provenance.methodology.value,
            "valid_from": result.factor.valid_from.isoformat(),
            "valid_to": result.factor.valid_to.isoformat() if result.factor.valid_to else None,
            "gwp_set": result.factor.gwp_100yr.gwp_set.value,
            "boundary": result.factor.boundary.value,
            "citation": result.factor.provenance.citation
        },
        "quality": {
            "uncertainty_95ci_pct": result.uncertainty_pct,
            "dqs": {
                "temporal": result.factor.dqs.temporal,
                "geographical": result.factor.dqs.geographical,
                "technological": result.factor.dqs.technological,
                "representativeness": result.factor.dqs.representativeness,
                "methodological": result.factor.dqs.methodological,
                "overall_score": result.factor.dqs.overall_score,
                "rating": result.factor.dqs.rating.value
            }
        },
        "compliance": {
            "frameworks": result.factor.compliance_frameworks,
            "csrd_compliant": "CSRD" in result.factor.compliance_frameworks,
            "cdp_reportable": True  # All factors are CDP reportable
        },
        "provenance_hash": result.factor.content_hash
    })

    return output_v1


def _format_output_compact(self, result: CalculationResult) -> Dict:
    """Format output as compact (minimal for mobile/IoT)"""
    return {
        "co2e_kg": result.co2e_100yr,
        "fuel": result.fuel_type,
        "amount": result.amount,
        "unit": result.unit,
        "factor": result.factor_co2e,
        "scope": result.scope,
        "quality_score": result.factor.dqs.overall_score,
        "uncertainty_pct": result.uncertainty_pct
    }
```

### Phase 2: Deprecation Notice (Week 24 - 6 months after launch)

#### Deprecation Headers
```python
if response_format == "legacy":
    # Add deprecation headers
    response_headers = {
        "X-API-Deprecation": "true",
        "X-API-Sunset-Date": "2026-09-30",
        "X-API-Migration-Guide": "https://docs.greenlang.ai/fuel-agent/migration-v2",
        "Warning": '299 - "Legacy response format deprecated. Migrate to response_format=enhanced by 2026-Q3"'
    }
```

#### Email Notifications
Send to all active v1 API users:
```
Subject: [Action Required] FuelAgentAI v1 API Deprecation - Migrate by 2026-Q3

Dear Customer,

We're writing to inform you that the legacy (v1) response format for FuelAgentAI will be sunset on September 30, 2026 (6 months from today).

**What's Changing:**
- The v1 response format (single CO2e value) is being replaced by v2 (multi-gas breakdown)
- This enables CSRD compliance, CDP reporting, and enhanced data quality

**Action Required:**
Update your API calls to include: "response_format": "enhanced"

**Timeline:**
- Today: Deprecation notice
- 2026-Q3: v1 format sunset (September 30)
- After 2026-Q3: All responses return v2 format by default

**Migration Guide:**
https://docs.greenlang.ai/fuel-agent/migration-v2

**Need Help?**
Reply to this email or contact support@greenlang.ai

Best regards,
GreenLang Team
```

### Phase 3: Sunset (Week 52 - 12 months after launch)

```python
def run(self, payload: Dict) -> Dict:
    """
    Run fuel emissions calculation.

    As of 2026-Q3, legacy format is no longer supported.
    """

    response_format = payload.get("response_format", "enhanced")  # Default changed to v2

    if response_format == "legacy":
        # Soft sunset: log warning, return v2 anyway
        logger.warning(
            f"Client requested legacy format, but it's no longer supported. "
            f"Returning enhanced format. Client IP: {request.client_ip}"
        )
        response_format = "enhanced"

    # Continue with v2 processing...
```

---

## 5. Feature Flags (Granular Control)

In addition to `response_format`, allow clients to enable/disable specific v2 features:

```json
{
  "fuel_type": "diesel",
  "amount": 1000,
  "unit": "gallons",

  "response_format": "enhanced",
  "features": {
    "multi_gas_breakdown": true,      // Include CO2/CH4/N2O vectors
    "provenance_tracking": true,      // Include factor source info
    "data_quality_scoring": true,     // Include DQS
    "uncertainty_ranges": true,       // Include ±X% confidence intervals
    "compliance_markers": true,       // Include CSRD/CDP flags
    "alternative_gwp": false,         // Include 20-year GWP (expensive)
    "explanations": true              // Include AI explanations
  }
}
```

**Use Cases:**
1. **Mobile apps:** Disable explanations to reduce payload
2. **Batch processing:** Disable provenance (already cached locally)
3. **Compliance reporting:** Enable all features
4. **Cost optimization:** Disable expensive features (alternative GWP)

**Default (if `features` not specified):**
```json
{
  "multi_gas_breakdown": true,
  "provenance_tracking": true,
  "data_quality_scoring": true,
  "uncertainty_ranges": true,
  "compliance_markers": true,
  "alternative_gwp": false,  // Disabled by default (reduces cost)
  "explanations": true
}
```

---

## 6. Testing Strategy

### Backward Compatibility Tests
```python
class TestBackwardCompatibility:
    """Ensure v2 implementation doesn't break v1 clients"""

    def test_v1_input_v1_output(self):
        """v1 input without response_format returns v1 output"""
        v1_input = {
            "fuel_type": "diesel",
            "amount": 1000,
            "unit": "gallons"
        }

        result = agent.run(v1_input)

        # Verify v1 output schema
        assert "co2e_emissions_kg" in result
        assert "emission_factor" in result
        assert "metadata" in result

        # Verify v2 fields NOT present
        assert "vectors_kg" not in result
        assert "factor_record" not in result
        assert "quality" not in result

    def test_v1_input_exact_match(self):
        """v1 input returns EXACT same output as before v2 implementation"""
        v1_input = {"fuel_type": "diesel", "amount": 1000, "unit": "gallons"}

        # Golden master from v1 implementation
        expected_v1_output = {
            "co2e_emissions_kg": 10210.0,
            "emission_factor": 10.21,
            # ... (full v1 output)
        }

        result = agent.run(v1_input)

        # Exact match (excluding metadata timestamps)
        assert result["co2e_emissions_kg"] == expected_v1_output["co2e_emissions_kg"]
        assert result["emission_factor"] == expected_v1_output["emission_factor"]

    def test_v2_input_v1_format_requested(self):
        """v2 input with response_format=legacy returns v1 output"""
        v2_input = {
            "fuel_type": "diesel",
            "amount": 1000,
            "unit": "gallons",
            "boundary": "WTT",  # v2 field
            "response_format": "legacy"
        }

        result = agent.run(v2_input)

        # Should return v1 format (no v2 fields)
        assert "vectors_kg" not in result


class TestV2Features:
    """Test v2 enhancements"""

    def test_multi_gas_breakdown(self):
        """v2 output includes CO2/CH4/N2O vectors"""
        input_data = {
            "fuel_type": "diesel",
            "amount": 1000,
            "unit": "gallons",
            "response_format": "enhanced"
        }

        result = agent.run(input_data)

        assert "vectors_kg" in result
        assert "CO2" in result["vectors_kg"]
        assert "CH4" in result["vectors_kg"]
        assert "N2O" in result["vectors_kg"]

        # Validate GWP calculation
        co2 = result["vectors_kg"]["CO2"]
        ch4 = result["vectors_kg"]["CH4"]
        n2o = result["vectors_kg"]["N2O"]

        co2e_calculated = co2 + (ch4 * 28) + (n2o * 273)
        assert result["co2e_by_gwp"]["co2e_100yr"] == pytest.approx(co2e_calculated, rel=0.001)

    def test_provenance_tracking(self):
        """v2 output includes factor source info"""
        input_data = {
            "fuel_type": "diesel",
            "amount": 1000,
            "unit": "gallons",
            "response_format": "enhanced"
        }

        result = agent.run(input_data)

        assert "factor_record" in result
        assert "source_org" in result["factor_record"]
        assert "citation" in result["factor_record"]
        assert result["factor_record"]["source_org"] == "EPA"

    def test_data_quality_scoring(self):
        """v2 output includes DQS"""
        input_data = {
            "fuel_type": "diesel",
            "amount": 1000,
            "unit": "gallons",
            "response_format": "enhanced"
        }

        result = agent.run(input_data)

        assert "quality" in result
        assert "dqs" in result["quality"]
        assert 1 <= result["quality"]["dqs"]["overall_score"] <= 5


class TestFeatureFlags:
    """Test granular feature control"""

    def test_disable_provenance(self):
        """Disabling provenance excludes factor_record"""
        input_data = {
            "fuel_type": "diesel",
            "amount": 1000,
            "unit": "gallons",
            "response_format": "enhanced",
            "features": {
                "provenance_tracking": False
            }
        }

        result = agent.run(input_data)

        assert "vectors_kg" in result  # Still present
        assert "factor_record" not in result  # Excluded
```

---

## 7. Client Migration Examples

### Example 1: Python Client (requests)
```python
import requests

# ========== BEFORE (v1) ==========
v1_request = {
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons"
}

response = requests.post(
    "https://api.greenlang.ai/fuel/calculate",
    json=v1_request
)

v1_result = response.json()
co2e = v1_result["co2e_emissions_kg"]  # 10210.0


# ========== AFTER (v2) ==========
v2_request = {
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "response_format": "enhanced"  # ← ONLY CHANGE NEEDED
}

response = requests.post(
    "https://api.greenlang.ai/fuel/calculate",
    json=v2_request
)

v2_result = response.json()

# v1 fields still work
co2e = v2_result["co2e_emissions_kg"]  # 10210.0 (same as v1)

# NEW: Multi-gas breakdown
co2 = v2_result["vectors_kg"]["CO2"]   # 10180.0
ch4 = v2_result["vectors_kg"]["CH4"]   # 0.82
n2o = v2_result["vectors_kg"]["N2O"]   # 0.164

# NEW: Provenance
source = v2_result["factor_record"]["source_org"]  # "EPA"
citation = v2_result["factor_record"]["citation"]  # "EPA (2024)..."

# NEW: Quality
dqs_score = v2_result["quality"]["dqs"]["overall_score"]  # 4.4
uncertainty = v2_result["quality"]["uncertainty_95ci_pct"]  # 5.0
```

### Example 2: JavaScript Client (fetch)
```javascript
// ========== BEFORE (v1) ==========
const v1_request = {
  fuel_type: "diesel",
  amount: 1000,
  unit: "gallons"
};

const response = await fetch("https://api.greenlang.ai/fuel/calculate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(v1_request)
});

const v1_result = await response.json();
const co2e = v1_result.co2e_emissions_kg;  // 10210.0


// ========== AFTER (v2) ==========
const v2_request = {
  fuel_type: "diesel",
  amount: 1000,
  unit: "gallons",
  response_format: "enhanced"  // ← ONLY CHANGE
};

const response = await fetch("https://api.greenlang.ai/fuel/calculate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(v2_request)
});

const v2_result = await response.json();

// v1 fields still work
const co2e = v2_result.co2e_emissions_kg;  // 10210.0

// NEW: Multi-gas
const { CO2, CH4, N2O } = v2_result.vectors_kg;

// NEW: Provenance
const source = v2_result.factor_record.source_org;  // "EPA"

// NEW: Quality
const dqs = v2_result.quality.dqs.overall_score;  // 4.4
```

### Example 3: Gradual Migration (Feature Flags)
```python
# Step 1: Enable v2, but only multi-gas (lowest risk)
request = {
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "response_format": "enhanced",
    "features": {
        "multi_gas_breakdown": True,
        "provenance_tracking": False,
        "data_quality_scoring": False
    }
}

# Step 2: Once stable, add provenance
request["features"]["provenance_tracking"] = True

# Step 3: Add quality scoring
request["features"]["data_quality_scoring"] = True

# Step 4: Enable all (full v2)
request["features"] = {}  # Use defaults (all enabled)
```

---

## 8. Documentation & Communication

### 8.1 API Documentation Updates
```markdown
# FuelAgentAI API Reference

## POST /fuel/calculate

Calculate emissions from fuel consumption.

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| fuel_type | string | Yes | - | Fuel type (diesel, natural_gas, etc.) |
| amount | number | Yes | - | Consumption amount |
| unit | string | Yes | - | Unit (gallons, kWh, therms) |
| country | string | No | "US" | ISO country code |
| **response_format** | **string** | **No** | **"legacy"** | **"legacy" \| "enhanced" \| "compact"** |
| boundary | string | No | "combustion" | "combustion" \| "WTT" \| "WTW" |
| gwp_set | string | No | "IPCC_AR6_100" | GWP reference set |

### Response (response_format: "legacy")
Single CO2e value (backward compatible).

```json
{
  "co2e_emissions_kg": 10210.0,
  "emission_factor": 10.21,
  ...
}
```

### Response (response_format: "enhanced")
Multi-gas breakdown + provenance + quality.

```json
{
  "co2e_emissions_kg": 10210.0,
  "vectors_kg": {"CO2": 10180, "CH4": 0.82, "N2O": 0.164},
  "factor_record": {...},
  "quality": {...},
  ...
}
```

### Migration Guide
See [Migration to v2](./migration-v2.md) for details.
```

### 8.2 Changelog
```markdown
# Changelog

## [2.0.0] - 2025-10-24

### Added
- **Multi-gas breakdown:** CO2, CH4, N2O reported separately
- **Provenance tracking:** Emission factor source attribution
- **Data quality scoring:** 5-dimension DQS per GHG Protocol
- **Uncertainty quantification:** ±X% confidence intervals
- **Compliance markers:** CSRD, CDP, GHG Protocol alignment
- **Feature flags:** Granular control over output fields
- **Alternative GWP:** 20-year horizon support (optional)

### Changed
- **Default response format:** Remains "legacy" for backward compatibility
- **Emission factor storage:** Upgraded to EmissionFactorRecord v2 schema
- **API endpoint:** Single endpoint supports both v1 and v2

### Deprecated
- **response_format: "legacy"** will be sunset in 2026-Q3
- Clients should migrate to **response_format: "enhanced"**

### Backward Compatibility
- ✅ All v1 inputs remain valid
- ✅ Default output format unchanged (legacy)
- ✅ v1 clients experience zero breaking changes
```

### 8.3 Migration Checklist for Customers
```markdown
# FuelAgentAI v2 Migration Checklist

## ✅ Step 1: Review Changes (Week 1)
- [ ] Read [What's New in v2](./whats-new-v2.md)
- [ ] Review [API Reference](./api-reference.md)
- [ ] Watch [Migration Video Tutorial](https://youtu.be/...)

## ✅ Step 2: Update Development (Week 2-3)
- [ ] Add `"response_format": "enhanced"` to API calls
- [ ] Update response parsing to handle new fields
- [ ] Test in development environment
- [ ] Update unit tests

## ✅ Step 3: Test in Staging (Week 4)
- [ ] Deploy to staging environment
- [ ] Run integration tests
- [ ] Compare v1 vs v2 results (should match for co2e_emissions_kg)
- [ ] Verify new fields (vectors_kg, factor_record, quality)

## ✅ Step 4: Production Rollout (Week 5-6)
- [ ] Deploy to production with feature flag OFF
- [ ] Gradually enable v2 for subset of traffic (10% → 50% → 100%)
- [ ] Monitor error rates and latency
- [ ] Verify correctness in production

## ✅ Step 5: Cleanup (Week 7)
- [ ] Remove v1 response parsing code (if applicable)
- [ ] Update documentation
- [ ] Train team on new fields
- [ ] Celebrate! 🎉

## Need Help?
- Email: support@greenlang.ai
- Slack: #fuel-agent-migration
- Office Hours: Thursdays 2-3pm PT
```

---

## 9. Rollout Plan

### Week 0-6: Development
- [ ] Implement EmissionFactorRecord v2
- [ ] Build response formatters (v1, v2, compact)
- [ ] Write backward compatibility tests
- [ ] Internal testing

### Week 6: Soft Launch (Beta)
- [ ] Deploy to production (behind feature flag)
- [ ] Invite 5 beta customers
- [ ] Collect feedback
- [ ] Fix bugs

### Week 8: Public Launch
- [ ] Announce v2 availability
- [ ] Publish migration guide
- [ ] Update documentation
- [ ] Email all customers

### Week 12: Adoption Push
- [ ] Migration office hours (weekly)
- [ ] Customer success check-ins
- [ ] Usage analytics dashboard
- [ ] Case studies (early adopters)

### Week 24: Deprecation Notice (6 months)
- [ ] Email deprecation notice
- [ ] Add deprecation headers
- [ ] Log v1 usage stats
- [ ] Contact holdouts

### Week 48: Sunset Preparation (12 months)
- [ ] Final migration reminders
- [ ] Offer migration assistance
- [ ] Identify stragglers
- [ ] Plan sunset date

### Week 52: Sunset (12 months)
- [ ] Change default response_format to "enhanced"
- [ ] Log v1 requests as warnings
- [ ] Monitor for issues
- [ ] Deprecate v1 code (keep for 3 months emergency rollback)

---

## 10. Success Metrics

### Adoption Rate
- **Target:** 80% of active clients migrate to v2 within 6 months
- **Measurement:** % of API calls with `response_format: "enhanced"`

### Backward Compatibility
- **Target:** Zero customer-reported breaking changes
- **Measurement:** Support tickets related to v2 upgrade

### Performance
- **Target:** v2 response time ≤ v1 response time + 50ms
- **Measurement:** p50, p95, p99 latency

### Cost
- **Target:** v2 cost per calculation ≤ $0.01 (4× v1 max)
- **Measurement:** Average AI + infrastructure cost

### Customer Satisfaction
- **Target:** NPS ≥ 50 for v2 migration experience
- **Measurement:** Post-migration survey

---

## 11. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **v2 breaks v1 clients** | Low | Critical | Comprehensive backward compat tests, gradual rollout |
| **Low adoption rate** | Medium | High | Migration incentives, office hours, case studies |
| **Performance regression** | Medium | Medium | Performance benchmarks, caching, optimization |
| **Cost overrun** | Medium | High | Budget monitoring, cost caps, feature flags |
| **Data quality issues** | Low | High | Validation tests, golden set parity, peer review |
| **Customer confusion** | High | Medium | Clear documentation, examples, video tutorials |
| **Deprecation resistance** | Medium | Medium | Early communication, extended timeline, support |

---

## 12. Conclusion

This API versioning strategy ensures a **smooth, zero-downtime transition** from FuelAgentAI v1 to v2:

✅ **Backward Compatible:** v1 clients see zero changes
✅ **Progressive Enhancement:** Clients opt-in to v2 features
✅ **Graceful Deprecation:** 12-month timeline with 6-month notice
✅ **Feature Flags:** Granular control over capabilities
✅ **Clear Migration Path:** Documentation, examples, support

**Timeline:**
- 2025-Q4: Launch v2 (both supported)
- 2026-Q2: Deprecation notice
- 2026-Q3: v1 sunset

**Next Steps:**
1. Implement response formatters
2. Write backward compatibility tests
3. Update documentation
4. Launch beta program
5. Monitor adoption

---

**Document Owner:** API Team Lead
**Approvers:** CTO, Product Manager, Customer Success
**Next Review:** 2025-11-01 (after v2 implementation)
