# PACK-029 Interim Targets Pack -- API Reference

**Pack ID:** PACK-029-interim-targets
**Version:** 1.0.0
**API Version:** v1
**Base URL:** `https://api.greenlang.io/v1/packs/029`
**Authentication:** JWT Bearer Token (SEC-001)
**Content Type:** `application/json`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Common Types](#common-types)
3. [Error Handling](#error-handling)
4. [Rate Limits](#rate-limits)
5. [Engines API](#engines-api)
   - [Interim Target Engine](#interim-target-engine)
   - [Quarterly Monitoring Engine](#quarterly-monitoring-engine)
   - [Annual Review Engine](#annual-review-engine)
   - [Variance Analysis Engine](#variance-analysis-engine)
   - [Trend Extrapolation Engine](#trend-extrapolation-engine)
   - [Corrective Action Engine](#corrective-action-engine)
   - [Target Recalibration Engine](#target-recalibration-engine)
   - [SBTi Validation Engine](#sbti-validation-engine)
   - [Carbon Budget Tracker Engine](#carbon-budget-tracker-engine)
   - [Alert Generation Engine](#alert-generation-engine)
6. [Workflows API](#workflows-api)
   - [Interim Target Setting Workflow](#interim-target-setting-workflow)
   - [Quarterly Monitoring Workflow](#quarterly-monitoring-workflow)
   - [Annual Progress Review Workflow](#annual-progress-review-workflow)
   - [Variance Investigation Workflow](#variance-investigation-workflow)
   - [Corrective Action Planning Workflow](#corrective-action-planning-workflow)
   - [Annual Reporting Workflow](#annual-reporting-workflow)
   - [Target Recalibration Workflow](#target-recalibration-workflow)
7. [Templates API](#templates-api)
8. [Integrations API](#integrations-api)
9. [Webhooks](#webhooks)
10. [SDK Reference](#sdk-reference)

---

## Authentication

All API requests require a valid JWT token in the `Authorization` header.

```http
Authorization: Bearer <jwt_token>
```

### Obtaining a Token

```http
POST /v1/auth/token
Content-Type: application/json

{
  "username": "target_manager@company.com",
  "password": "********"
}
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2g..."
}
```

### Required Permissions

| Endpoint Category | Required Role | Permission |
|-------------------|--------------|------------|
| Engine execution | `target_manager` or higher | `pack029:engine:execute` |
| Workflow execution | `target_manager` or higher | `pack029:workflow:execute` |
| Report generation | `progress_analyst` or higher | `pack029:template:render` |
| Configuration | `interim_targets_admin` | `pack029:config:write` |
| Read-only access | `progress_analyst` or higher | `pack029:read` |
| Health check | Any authenticated user | `pack029:health:read` |
| Alert management | `target_manager` or higher | `pack029:alerts:manage` |

---

## Common Types

### Enumerations

#### ClimateAmbition

```python
class ClimateAmbition(str, Enum):
    CELSIUS_1_5 = "1.5c"          # SBTi 1.5C aligned (4.2%/yr)
    WELL_BELOW_2C = "wb2c"        # SBTi Well-Below 2C (2.5%/yr)
    TWO_C = "2c"                  # 2C aligned (1.5%/yr)
    RACE_TO_ZERO = "race_to_zero" # Race to Zero (7.0%/yr, 50% by 2030)
```

#### ScopeType

```python
class ScopeType(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"
    ALL_SCOPES = "all_scopes"
```

#### PathwayShape

```python
class PathwayShape(str, Enum):
    LINEAR = "linear"                   # Equal annual reduction
    FRONT_LOADED = "front_loaded"       # Faster early reductions (sqrt)
    BACK_LOADED = "back_loaded"         # Slower early, accelerating later (x^2)
    MILESTONE_BASED = "milestone_based" # Custom milestones with linear interp
    CONSTANT_RATE = "constant_rate"     # Compound annual reduction (exponential)
```

#### TargetType

```python
class TargetType(str, Enum):
    NEAR_TERM = "near_term"   # 0-10 years from baseline
    MID_TERM = "mid_term"     # 10-20 years from baseline
    LONG_TERM = "long_term"   # 20+ years from baseline
    NET_ZERO = "net_zero"     # Final target year
```

#### ValidationStatus

```python
class ValidationStatus(str, Enum):
    ALIGNED = "aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    MISALIGNED = "misaligned"
    EXCEEDS_MINIMUM = "exceeds_minimum"
    REQUIRES_REVIEW = "requires_review"
```

#### RAGStatus

```python
class RAGStatus(str, Enum):
    GREEN = "green"   # Within 5% of target
    AMBER = "amber"   # 5-15% above target
    RED = "red"       # More than 15% above target
```

### Common Models

#### BaselineData

```python
class BaselineData(BaseModel):
    base_year: int              # 2015-2025
    scope_1_tco2e: Decimal      # Scope 1 baseline (tCO2e), >= 0
    scope_2_tco2e: Decimal      # Scope 2 baseline (tCO2e), >= 0
    scope_3_tco2e: Decimal      # Scope 3 baseline (tCO2e), >= 0
    total_tco2e: Decimal        # Auto-calculated if zero
    scope_2_method: str         # "market_based" or "location_based"
    is_flag_sector: bool        # Has FLAG sector emissions
    flag_emissions_tco2e: Decimal  # FLAG emissions subset
```

#### LongTermTarget

```python
class LongTermTarget(BaseModel):
    target_year: int            # 2030-2070 (default 2050)
    reduction_pct: Decimal      # 0-100 (default 90)
    residual_emissions_pct: Decimal  # 0-100 (default 10)
    net_zero_year: int          # 2030-2070 (default 2050)
    includes_scope_3: bool      # Default True
```

#### ProvenanceMetadata

```python
class ProvenanceMetadata(BaseModel):
    result_id: str              # UUID4
    engine_version: str         # Semantic version
    calculated_at: datetime     # UTC timestamp
    processing_time_ms: float   # Execution duration
    provenance_hash: str        # SHA-256 hash
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "PACK029_VALIDATION_ERROR",
    "message": "Baseline year must be between 2015 and 2025",
    "field": "baseline.base_year",
    "details": {
      "provided_value": 2010,
      "min_value": 2015,
      "max_value": 2025
    }
  },
  "request_id": "req_abc123",
  "timestamp": "2026-03-19T10:30:00Z"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `PACK029_VALIDATION_ERROR` | 400 | Input validation failed |
| `PACK029_BASELINE_MISSING` | 400 | Required baseline data not provided |
| `PACK029_TARGET_CONFLICT` | 400 | Conflicting target parameters |
| `PACK029_ENTITY_NOT_FOUND` | 404 | Entity ID not found |
| `PACK029_TARGET_NOT_FOUND` | 404 | Interim targets not set for entity |
| `PACK029_UNAUTHORIZED` | 401 | Authentication required |
| `PACK029_FORBIDDEN` | 403 | Insufficient permissions |
| `PACK029_RATE_LIMITED` | 429 | Rate limit exceeded |
| `PACK029_ENGINE_ERROR` | 500 | Internal engine calculation error |
| `PACK029_INTEGRATION_ERROR` | 502 | External integration failure |

---

## Rate Limits

| Tier | Requests/min | Concurrent workflows | Batch size |
|------|-------------|---------------------|------------|
| Standard | 60 | 5 | 50 |
| Professional | 120 | 10 | 100 |
| Enterprise | 300 | 25 | 500 |

---

## Engines API

### Interim Target Engine

**Endpoint:** `POST /v1/packs/029/engines/interim-targets/calculate`

Calculates 5-year and 10-year interim targets from baseline emissions and long-term net-zero targets. Supports SBTi validation, scope-specific timelines, and 5 pathway shapes.

#### Request

```json
{
  "entity_name": "Acme Corp",
  "entity_id": "acme-001",
  "baseline": {
    "base_year": 2021,
    "scope_1_tco2e": 50000,
    "scope_2_tco2e": 30000,
    "scope_3_tco2e": 120000,
    "scope_2_method": "market_based",
    "is_flag_sector": false
  },
  "long_term_target": {
    "target_year": 2050,
    "reduction_pct": 90,
    "net_zero_year": 2050,
    "includes_scope_3": true
  },
  "ambition_level": "1.5c",
  "pathway_shape": "linear",
  "scope_3_lag_years": 0,
  "generate_5_year_targets": true,
  "generate_10_year_targets": true,
  "include_sbti_validation": true,
  "reporting_year": 2024
}
```

#### Response

```json
{
  "result_id": "550e8400-e29b-41d4-a716-446655440000",
  "engine_version": "1.0.0",
  "calculated_at": "2026-03-19T10:30:00Z",
  "entity_name": "Acme Corp",
  "ambition_level": "1.5c",
  "pathway_shape": "linear",
  "baseline_year": 2021,
  "baseline_total_tco2e": 200000,
  "net_zero_year": 2050,
  "scope_timelines": [
    {
      "scope": "scope_1_2",
      "baseline_tco2e": 80000,
      "near_term_year": 2030,
      "near_term_target_tco2e": 55172.41,
      "near_term_reduction_pct": 31.03,
      "long_term_year": 2050,
      "long_term_target_tco2e": 8000,
      "long_term_reduction_pct": 90.00,
      "annual_rate_pct": 7.766,
      "milestones": [...]
    },
    {
      "scope": "scope_3",
      "baseline_tco2e": 120000,
      "near_term_year": 2030,
      "near_term_target_tco2e": 82758.62,
      "near_term_reduction_pct": 31.03,
      "milestones": [...]
    }
  ],
  "five_year_targets": [...],
  "ten_year_targets": [...],
  "sbti_validation": {
    "is_compliant": true,
    "ambition_level": "1.5C Aligned",
    "total_checks": 8,
    "passed_checks": 8,
    "failed_checks": 0,
    "validation_notes": [...]
  },
  "implied_temperature_score": 1.50,
  "annual_reduction_rate_scope12_pct": 7.766,
  "annual_reduction_rate_scope3_pct": 7.766,
  "total_abatement_required_tco2e": 180000,
  "processing_time_ms": 45.2,
  "provenance_hash": "a1b2c3d4e5f6..."
}
```

#### Calculation Formulas

**Linear Interim Target:**
```
target(t) = baseline * (1 - reduction_pct * (t - base_year) / (target_year - base_year))
```

**Annual Rate (compound):**
```
remaining = 1 - reduction_pct / 100
annual_rate_pct = (1 - remaining^(1/years)) * 100
```

**Temperature Score (simplified SBTi):**
```
temp = 1.5 + max(0, (4.2 - annual_rate) / 4.2) * 2.0
capped at 4.0C
```

**Cumulative Budget (trapezoidal):**
```
budget(t) = SUM over intervals: (emissions_prev + emissions_current) / 2 * years_elapsed
```

#### Python SDK

```python
from engines.interim_target_engine import InterimTargetEngine, InterimTargetInput

engine = InterimTargetEngine()
result = await engine.calculate(input_data)
result_batch = await engine.calculate_batch(inputs_list)
```

---

### Quarterly Monitoring Engine

**Endpoint:** `POST /v1/packs/029/engines/quarterly-monitoring/monitor`

Compares quarterly actual emissions against interim targets, generates RAG status, and triggers alerts.

#### Request

```json
{
  "entity_id": "acme-001",
  "reporting_quarter": "2025-Q2",
  "actual_emissions": {
    "scope_1_tco2e": 11500,
    "scope_2_tco2e": 7000,
    "scope_3_tco2e": 28000
  },
  "rag_thresholds": {
    "green_pct": 5,
    "amber_pct": 15
  }
}
```

#### Response

```json
{
  "quarter": "2025-Q2",
  "overall_rag": "green",
  "scope_results": [
    {
      "scope": "scope_1_2",
      "actual_tco2e": 18500,
      "target_tco2e": 19200,
      "variance_tco2e": -700,
      "variance_pct": -3.6,
      "rag_status": "green",
      "trend_direction": "improving",
      "trend_velocity_pct_per_quarter": -2.1
    }
  ],
  "alerts": [],
  "cumulative_ytd_actual_tco2e": 37800,
  "cumulative_ytd_target_tco2e": 38400,
  "annualized_projection_tco2e": 75600,
  "processing_time_ms": 12.4,
  "provenance_hash": "b2c3d4e5f6a7..."
}
```

---

### Annual Review Engine

**Endpoint:** `POST /v1/packs/029/engines/annual-review/review`

Performs comprehensive annual progress assessment with year-over-year comparison, cumulative budget tracking, and pathway adherence scoring.

#### Request

```json
{
  "entity_id": "acme-001",
  "reporting_year": 2025,
  "annual_actual": {
    "scope_1_tco2e": 44000,
    "scope_2_tco2e": 26000,
    "scope_3_tco2e": 105000
  },
  "include_budget_analysis": true,
  "include_trend_forecast": true
}
```

#### Response

```json
{
  "reporting_year": 2025,
  "overall_assessment": "on_track",
  "yoy_comparison": {
    "previous_year_tco2e": 185000,
    "current_year_tco2e": 175000,
    "absolute_change_tco2e": -10000,
    "percentage_change": -5.4,
    "required_change_pct": -4.2
  },
  "cumulative_budget": {
    "total_budget_tco2e": 3500000,
    "consumed_tco2e": 755000,
    "remaining_tco2e": 2745000,
    "burn_rate_pct": 21.6,
    "years_until_exhaustion": 18.2
  },
  "pathway_adherence_score": 92,
  "forward_projection": {
    "projected_2030_tco2e": 138000,
    "target_2030_tco2e": 116000,
    "gap_tco2e": 22000,
    "on_track_probability": 0.72
  },
  "processing_time_ms": 85.6,
  "provenance_hash": "c3d4e5f6a7b8..."
}
```

---

### Variance Analysis Engine

**Endpoint:** `POST /v1/packs/029/engines/variance-analysis/decompose`

Performs LMDI (Logarithmic Mean Divisia Index) decomposition of emissions variance into activity, intensity, and structural effects. Guarantees perfect decomposition (sum of effects = total variance).

#### Request

```json
{
  "entity_id": "acme-001",
  "period_start": "2023",
  "period_end": "2024",
  "emissions_start": 200000,
  "emissions_end": 185000,
  "activity_data_start": {
    "revenue_musd": 500,
    "production_tonnes": 120000,
    "employees": 5000
  },
  "activity_data_end": {
    "revenue_musd": 550,
    "production_tonnes": 125000,
    "employees": 5200
  },
  "decomposition_method": "lmdi_additive"
}
```

#### Response

```json
{
  "period": "2023 -> 2024",
  "total_change_tco2e": -15000,
  "decomposition": {
    "activity_effect_tco2e": 15000,
    "intensity_effect_tco2e": -28000,
    "structural_effect_tco2e": -2000,
    "sum_of_effects_tco2e": -15000,
    "is_perfect_decomposition": true,
    "residual_tco2e": 0
  },
  "percentage_contributions": {
    "activity_effect_pct": -100.0,
    "intensity_effect_pct": 186.7,
    "structural_effect_pct": 13.3
  },
  "interpretation": {
    "activity_narrative": "Business growth (revenue +10%, production +4.2%) added 15,000 tCO2e",
    "intensity_narrative": "Emission intensity improvements removed 28,000 tCO2e",
    "structural_narrative": "Structural shifts removed 2,000 tCO2e",
    "overall_narrative": "Net reduction of 15,000 tCO2e despite business growth, driven by intensity improvements"
  },
  "processing_time_ms": 23.1,
  "provenance_hash": "d4e5f6a7b8c9..."
}
```

#### LMDI Formulas

**Additive LMDI (default):**
```
L(E_0, E_t) = (E_t - E_0) / (ln(E_t) - ln(E_0))      [logarithmic mean]

Activity Effect    = SUM_i [ L(E_i0, E_it) * ln(A_t / A_0) ]
Intensity Effect   = SUM_i [ L(E_i0, E_it) * ln(I_it / I_i0) ]
Structural Effect  = SUM_i [ L(E_i0, E_it) * ln(S_it / S_i0) ]

Perfect decomposition: Activity + Intensity + Structural = Total Change (always)
```

**Multiplicative LMDI:**
```
D_act = exp( SUM_i [ w_i * ln(A_t / A_0) ] )
D_int = exp( SUM_i [ w_i * ln(I_it / I_i0) ] )
D_str = exp( SUM_i [ w_i * ln(S_it / S_i0) ] )

Perfect decomposition: D_act * D_int * D_str = E_t / E_0 (always)
```

---

### Trend Extrapolation Engine

**Endpoint:** `POST /v1/packs/029/engines/trend-extrapolation/forecast`

Generates emissions forecasts using three models: linear regression, exponential smoothing, and ARIMA.

#### Request

```json
{
  "entity_id": "acme-001",
  "historical_emissions": [
    {"year": 2021, "tco2e": 200000},
    {"year": 2022, "tco2e": 192000},
    {"year": 2023, "tco2e": 185000},
    {"year": 2024, "tco2e": 178000}
  ],
  "forecast_periods": 6,
  "models": ["linear", "exponential_smoothing", "arima"],
  "confidence_levels": [0.80, 0.95]
}
```

#### Response

```json
{
  "forecasts": {
    "linear": {
      "predictions": [
        {"year": 2025, "tco2e": 171000, "ci_80": [168000, 174000], "ci_95": [165000, 177000]},
        {"year": 2026, "tco2e": 164000, "ci_80": [159000, 169000], "ci_95": [155000, 173000]}
      ],
      "r_squared": 0.997,
      "mae": 500,
      "rmse": 612,
      "mape": 0.3
    },
    "exponential_smoothing": {
      "predictions": [...],
      "alpha": 0.45,
      "mae": 800,
      "rmse": 950
    },
    "arima": {
      "predictions": [...],
      "order": [1, 1, 0],
      "aic": 42.3,
      "mae": 600,
      "rmse": 720
    }
  },
  "recommended_model": "linear",
  "recommendation_reason": "Highest R-squared (0.997) and lowest MAPE (0.3%)"
}
```

---

### Corrective Action Engine

**Endpoint:** `POST /v1/packs/029/engines/corrective-actions/plan`

Optimizes a portfolio of emission reduction initiatives to close the gap to target within budget and timeline constraints.

#### Request

```json
{
  "entity_id": "acme-001",
  "gap_tco2e": 7000,
  "available_initiatives": [
    {
      "id": "init-001",
      "name": "LED lighting upgrade",
      "reduction_tco2e": 2000,
      "cost_usd": 500000,
      "implementation_years": 1,
      "certainty": "high"
    },
    {
      "id": "init-002",
      "name": "Heat pump installation",
      "reduction_tco2e": 3500,
      "cost_usd": 1200000,
      "implementation_years": 2,
      "certainty": "medium"
    },
    {
      "id": "init-003",
      "name": "Solar PV installation",
      "reduction_tco2e": 4000,
      "cost_usd": 2000000,
      "implementation_years": 1,
      "certainty": "high"
    }
  ],
  "max_budget_usd": 3500000,
  "max_years": 3,
  "optimization_strategy": "cost_effective"
}
```

#### Response

```json
{
  "gap_tco2e": 7000,
  "selected_initiatives": [
    {
      "id": "init-001",
      "name": "LED lighting upgrade",
      "reduction_tco2e": 2000,
      "cost_usd": 500000,
      "cost_per_tco2e": 250,
      "start_year": 2025,
      "completion_year": 2025
    },
    {
      "id": "init-002",
      "name": "Heat pump installation",
      "reduction_tco2e": 3500,
      "cost_usd": 1200000,
      "cost_per_tco2e": 343,
      "start_year": 2025,
      "completion_year": 2026
    },
    {
      "id": "init-003",
      "name": "Solar PV installation",
      "reduction_tco2e": 4000,
      "cost_usd": 2000000,
      "cost_per_tco2e": 500,
      "start_year": 2026,
      "completion_year": 2026
    }
  ],
  "total_reduction_tco2e": 9500,
  "total_cost_usd": 3700000,
  "gap_closure_pct": 135.7,
  "years_to_close": 2,
  "cost_per_tco2e_portfolio": 389,
  "scenarios": {
    "optimistic": {"closure_pct": 135.7, "years": 1.5},
    "baseline": {"closure_pct": 100.0, "years": 2.0},
    "pessimistic": {"closure_pct": 78.6, "years": 3.0}
  }
}
```

---

### Target Recalibration Engine

**Endpoint:** `POST /v1/packs/029/engines/recalibration/recalibrate`

Recalculates interim targets after trigger events (acquisitions, divestments, methodology changes).

#### Supported Triggers

| Trigger | Threshold | Description |
|---------|-----------|-------------|
| `acquisition` | >5% baseline change | Acquired entity emissions added to baseline |
| `divestment` | >5% baseline change | Divested entity emissions removed from baseline |
| `methodology_change` | Any change | GHG calculation methodology update |
| `base_year_update` | N/A | Base year restated with new data |
| `scope_boundary_change` | Any change | Scope boundary expanded or contracted |
| `organic_growth` | >10% baseline change | Significant organic growth beyond projections |

---

### SBTi Validation Engine

**Endpoint:** `POST /v1/packs/029/engines/sbti-validation/validate`

Validates interim targets against all 21 SBTi Corporate Net-Zero Standard v1.2 criteria.

#### 21 Validation Criteria

```json
{
  "criteria": [
    {"id": 1, "name": "scope_12_coverage", "threshold": "95%", "type": "minimum"},
    {"id": 2, "name": "scope_3_coverage", "threshold": "67%", "type": "minimum"},
    {"id": 3, "name": "near_term_ambition", "threshold": "4.2%/yr (1.5C)", "type": "minimum"},
    {"id": 4, "name": "near_term_timeframe", "threshold": "5-10 years", "type": "range"},
    {"id": 5, "name": "near_term_latest_year", "threshold": "2030", "type": "maximum"},
    {"id": 6, "name": "long_term_reduction", "threshold": "90%", "type": "minimum"},
    {"id": 7, "name": "long_term_timeframe", "threshold": "by 2050", "type": "maximum"},
    {"id": 8, "name": "no_backsliding", "threshold": "monotonic", "type": "constraint"},
    {"id": 9, "name": "base_year_recency", "threshold": "2 years", "type": "maximum"},
    {"id": 10, "name": "scope_3_lag", "threshold": "5 years", "type": "maximum"},
    {"id": 11, "name": "flag_separate_target", "threshold": "required if FLAG", "type": "conditional"},
    {"id": 12, "name": "absolute_target_type", "threshold": "absolute or intensity", "type": "check"},
    {"id": 13, "name": "no_double_counting", "threshold": "no overlap", "type": "constraint"},
    {"id": 14, "name": "recalculation_policy", "threshold": "documented", "type": "check"},
    {"id": 15, "name": "base_year_consistency", "threshold": "same across scopes", "type": "check"},
    {"id": 16, "name": "method_consistency", "threshold": "GHG Protocol", "type": "check"},
    {"id": 17, "name": "target_boundary", "threshold": "defined", "type": "check"},
    {"id": 18, "name": "exclusions_justified", "threshold": "documented", "type": "check"},
    {"id": 19, "name": "renewable_energy", "threshold": "RE100 compatible", "type": "check"},
    {"id": 20, "name": "carbon_credits", "threshold": "excluded from near-term", "type": "constraint"},
    {"id": 21, "name": "neutralization_plan", "threshold": "for residuals", "type": "check"}
  ]
}
```

---

### Carbon Budget Tracker Engine

**Endpoint:** `POST /v1/packs/029/engines/carbon-budget/track`

Tracks cumulative carbon budget consumption and remaining allowance.

#### Budget Calculation

```
Total Budget = SUM over [base_year, target_year]: annual_target_tco2e
              (using trapezoidal integration for continuous pathway)

Consumed = SUM over [base_year, current_year]: actual_annual_tco2e

Remaining = Total Budget - Consumed

Burn Rate = Consumed / Total Budget * 100

Years Until Exhaustion = Remaining / current_annual_actual
```

---

### Alert Generation Engine

**Endpoint:** `POST /v1/packs/029/engines/alerts/generate`

Generates threshold-based alerts for off-track performance with configurable severity and escalation.

#### Alert Types

| Alert Type | Trigger | Default Severity | Escalation |
|------------|---------|-----------------|------------|
| `off_track_amber` | Actual 5-15% above target | WARNING | Email to target manager |
| `off_track_red` | Actual >15% above target | CRITICAL | Email + Slack to leadership |
| `trend_deteriorating` | 3 consecutive quarters worsening | WARNING | Email to target manager |
| `budget_exhaustion` | Budget burn rate >110% of plan | CRITICAL | Email to CFO + sustainability |
| `milestone_missed` | 5-year milestone not met | CRITICAL | Board notification |
| `recalibration_needed` | Trigger event detected | INFO | Email to target manager |

---

## Workflows API

### Interim Target Setting Workflow

**Endpoint:** `POST /v1/packs/029/workflows/interim-target-setting/execute`

5-phase workflow: BaselineImport -> InterimCalc -> SBTiValidation -> PathwayGen -> TargetReport

```python
from workflows.interim_target_setting_workflow import InterimTargetSettingWorkflow

workflow = InterimTargetSettingWorkflow(preset="sbti_15c")
result = await workflow.execute(
    entity_name="Acme Corp",
    entity_id="acme-001",
    baseline={...},
    long_term_target={...},
)
```

### Quarterly Monitoring Workflow

**Endpoint:** `POST /v1/packs/029/workflows/quarterly-monitoring/execute`

4-phase workflow: DataCollection -> ProgressCheck -> TrendAnalysis -> QuarterlyReport

### Annual Progress Review Workflow

**Endpoint:** `POST /v1/packs/029/workflows/annual-progress-review/execute`

5-phase workflow: AnnualDataCollect -> YoYComparison -> BudgetCheck -> TrendForecast -> AnnualReport

### Variance Investigation Workflow

**Endpoint:** `POST /v1/packs/029/workflows/variance-investigation/execute`

4-phase workflow: DataPrep -> LMDIDecomposition -> RootCauseAttribution -> VarianceReport

### Corrective Action Planning Workflow

**Endpoint:** `POST /v1/packs/029/workflows/corrective-action-planning/execute`

5-phase workflow: GapQuantification -> InitiativeScanning -> MACCOptimization -> ScheduleGen -> ActionPlanReport

### Annual Reporting Workflow

**Endpoint:** `POST /v1/packs/029/workflows/annual-reporting/execute`

4-phase workflow: DataConsolidation -> CDPExport -> TCFDExport -> SBTiDisclosure

### Target Recalibration Workflow

**Endpoint:** `POST /v1/packs/029/workflows/target-recalibration/execute`

4-phase workflow: TriggerDetection -> BaselineAdjustment -> TargetRecalc -> RecalibrationReport

---

## Templates API

### Render Template

**Endpoint:** `POST /v1/packs/029/templates/{template_id}/render`

#### Supported Templates

| Template ID | Formats | Description |
|------------|---------|-------------|
| `interim-targets-summary` | MD, HTML, JSON, PDF | All interim targets overview |
| `quarterly-progress` | MD, HTML, JSON, PDF | Quarterly RAG dashboard |
| `annual-progress` | MD, HTML, JSON, PDF | Annual progress report |
| `variance-waterfall` | MD, HTML, JSON, PDF | LMDI decomposition waterfall |
| `corrective-action-plan` | MD, HTML, JSON, PDF | Gap closure action plan |
| `sbti-validation` | MD, HTML, JSON, PDF | 21-criteria SBTi assessment |
| `cdp-export` | JSON, XLSX | CDP C4.1/C4.2 export |
| `tcfd-disclosure` | MD, HTML, JSON, PDF | TCFD Metrics and Targets |
| `carbon-budget` | MD, HTML, JSON, PDF | Cumulative budget status |
| `executive-dashboard` | HTML, PDF | Board-level KPI dashboard |

---

## Integrations API

### PACK-021 Bridge

**Endpoint:** `GET /v1/packs/029/integrations/pack021/baseline/{entity_id}`

Imports baseline emissions and long-term targets from PACK-021 Net Zero Starter Pack.

### PACK-028 Bridge

**Endpoint:** `GET /v1/packs/029/integrations/pack028/pathway/{entity_id}`

Imports sector pathways and abatement levers from PACK-028 Sector Pathway Pack.

### MRV Bridge

**Endpoint:** `POST /v1/packs/029/integrations/mrv/emissions`

Routes to appropriate MRV agents (30 total) for actual emissions calculation.

### SBTi Portal Bridge

**Endpoint:** `POST /v1/packs/029/integrations/sbti/submit`

Formats and prepares SBTi target submission package.

### CDP Bridge

**Endpoint:** `POST /v1/packs/029/integrations/cdp/export`

Generates CDP Climate Change C4.1 and C4.2 formatted output.

### TCFD Bridge

**Endpoint:** `POST /v1/packs/029/integrations/tcfd/export`

Generates TCFD Metrics and Targets disclosure content.

### Alerting Bridge

**Endpoint:** `POST /v1/packs/029/integrations/alerts/send`

Sends alerts via configured channels (email, Slack, Teams).

### Health Check

**Endpoint:** `GET /v1/packs/029/integrations/health`

Returns 20-category health check status.

---

## Webhooks

### Supported Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `target.created` | New interim targets set | Full target result |
| `monitoring.completed` | Quarterly monitoring done | RAG status + variance |
| `alert.generated` | New alert triggered | Alert details |
| `recalibration.triggered` | Recalibration event | Trigger details |
| `milestone.reached` | 5-year milestone achieved | Milestone details |
| `milestone.missed` | 5-year milestone missed | Milestone + gap |

### Webhook Configuration

```json
{
  "url": "https://your-app.com/webhooks/pack029",
  "events": ["alert.generated", "milestone.missed"],
  "secret": "whsec_...",
  "active": true
}
```

---

## SDK Reference

### Python SDK

```python
from pack029 import InterimTargetsClient

client = InterimTargetsClient(
    base_url="https://api.greenlang.io/v1",
    api_key="your-api-key",
)

# Set interim targets
targets = await client.set_targets(entity_id="acme-001", baseline={...})

# Monitor quarterly
status = await client.monitor_quarterly(entity_id="acme-001", quarter="2025-Q2")

# Decompose variance
variance = await client.decompose_variance(entity_id="acme-001", period="2024")

# Plan corrective actions
plan = await client.plan_corrective_actions(entity_id="acme-001", gap_tco2e=7000)
```

---

**End of API Reference**
