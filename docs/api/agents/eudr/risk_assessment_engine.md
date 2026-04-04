# AGENT-EUDR-028: Risk Assessment Engine API

**Agent ID:** `GL-EUDR-RAE-028`
**Prefix:** `/api/v1/eudr/risk-assessment-engine`
**Version:** 1.0.0
**PRD:** PRD-AGENT-EUDR-028
**Regulation:** EU 2023/1115 (EUDR) -- Risk assessment per Articles 4, 9, 10, 12, 13, 29, 31

## Purpose

The Risk Assessment Engine agent executes the full EUDR risk assessment
pipeline: aggregating risk factors from upstream agents, calculating composite
scores across environmental/governance/supply-chain/social dimensions,
evaluating Article 10(2) criteria, applying EU country benchmarks (Article 29),
classifying risk levels, checking simplified due diligence eligibility
(Article 13), recording risk trends, supporting manual overrides with audit
trails, and performing batch assessments.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/assess` | Execute full risk assessment pipeline | JWT |
| GET | `/assess/{operation_id}` | Get assessment operation status | JWT |
| POST | `/composite-score` | Calculate composite risk score | JWT |
| POST | `/evaluate-criteria` | Evaluate Article 10(2) criteria | JWT |
| GET | `/benchmarks/{country_code}` | Get country benchmark | JWT |
| POST | `/benchmarks/batch` | Batch country benchmarks | JWT |
| POST | `/classify` | Classify risk level | JWT |
| POST | `/simplified-dd/check` | Check simplified DD eligibility | JWT |
| POST | `/override` | Apply risk override | JWT |
| GET | `/trend/{operator_id}/{commodity}` | Get risk trend analysis | JWT |
| POST | `/assess/batch` | Batch risk assessment | JWT |
| GET | `/health` | Health check | None |

**Total: 12 endpoints**

---

## Endpoints

### POST /api/v1/eudr/risk-assessment-engine/assess

Execute the complete 10-step risk assessment pipeline for an operator,
commodity, and set of sourcing countries.

**Pipeline Steps:**
1. Aggregate risk factors from upstream agents
2. Calculate composite risk score
3. Evaluate Article 10(2) criteria
4. Retrieve EU country benchmarks
5. Classify overall risk level
6. Check simplified DD eligibility
7. Record trend data point
8. Generate risk assessment report
9. Compute provenance hash
10. Return operation result

**Request:**

```json
{
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "country_codes": ["GH", "CI"],
  "supplier_ids": ["sup-001", "sup-002"]
}
```

**Response (200 OK):**

```json
{
  "operation_id": "rao_001",
  "status": "completed",
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "composite_score": {
    "overall_score": "0.52",
    "dimensions": {
      "environmental": "0.58",
      "governance": "0.45",
      "supply_chain": "0.51",
      "social": "0.42"
    }
  },
  "risk_level": "standard",
  "article10_criteria": {
    "deforestation_prevalence": {"met": true, "score": 0.55},
    "conflict_sanctions": {"met": false, "score": 0.12},
    "corruption_index": {"met": true, "score": 0.51},
    "legal_complexity": {"met": true, "score": 0.48},
    "supply_chain_complexity": {"met": true, "score": 0.52},
    "mixing_substitution": {"met": false, "score": 0.15},
    "indigenous_rights": {"met": false, "score": 0.20}
  },
  "country_benchmarks": {
    "GH": "standard",
    "CI": "high"
  },
  "simplified_dd_eligible": false,
  "report_id": "rpt_rae_001",
  "provenance_hash": "sha256:a1b2c3d4...",
  "completed_at": "2026-04-04T10:05:00Z"
}
```

---

### POST /api/v1/eudr/risk-assessment-engine/composite-score

Calculate a composite risk score from individual risk factor inputs using
multi-dimensional weighted scoring. This endpoint can be called independently
outside the full pipeline.

**Request:**

```json
{
  "factor_inputs": [
    {
      "factor_name": "deforestation_rate",
      "dimension": "environmental",
      "value": 0.65,
      "weight": 0.3,
      "source": "GL-EUDR-DAS-020"
    },
    {
      "factor_name": "governance_index",
      "dimension": "governance",
      "value": 0.45,
      "weight": 0.25,
      "source": "GL-EUDR-CRE-016"
    },
    {
      "factor_name": "supplier_score",
      "dimension": "supply_chain",
      "value": 0.35,
      "weight": 0.25,
      "source": "GL-EUDR-SRS-017"
    },
    {
      "factor_name": "indigenous_risk",
      "dimension": "social",
      "value": 0.20,
      "weight": 0.2,
      "source": "GL-EUDR-IRC-022"
    }
  ],
  "country_codes": ["GH"]
}
```

**Response (200 OK):**

```json
{
  "overall_score": "0.44",
  "dimensions": {
    "environmental": {"score": "0.65", "weight": 0.3, "weighted": "0.195"},
    "governance": {"score": "0.45", "weight": 0.25, "weighted": "0.1125"},
    "supply_chain": {"score": "0.35", "weight": 0.25, "weighted": "0.0875"},
    "social": {"score": "0.20", "weight": 0.2, "weighted": "0.04"}
  },
  "factors_used": 4,
  "country_adjustments_applied": true
}
```

---

### POST /api/v1/eudr/risk-assessment-engine/evaluate-criteria

Evaluate the seven criteria specified in EUDR Article 10(2) for risk
assessment:
- (a) Prevalence of deforestation/degradation
- (b) Conflict, sanctions, or instability
- (c) Corruption perception index
- (d) Legal framework complexity
- (e) Supply chain complexity
- (f) Commodity mixing/substitution risk
- (g) Indigenous peoples and local community rights concerns

**Request:**

```json
{
  "factor_inputs": [
    {"factor_name": "deforestation_rate", "dimension": "environmental", "value": 0.65}
  ],
  "composite_score": {
    "overall_score": "0.52",
    "dimensions": {"environmental": "0.58", "governance": "0.45"}
  },
  "country_codes": ["GH", "CI"]
}
```

**Response (200 OK):**

```json
{
  "criteria_met_count": 4,
  "criteria_total": 7,
  "criteria": [
    {"criterion": "a_deforestation", "met": true, "score": 0.55, "threshold": 0.3},
    {"criterion": "b_conflict_sanctions", "met": false, "score": 0.12, "threshold": 0.3},
    {"criterion": "c_corruption", "met": true, "score": 0.51, "threshold": 0.5},
    {"criterion": "d_legal_complexity", "met": true, "score": 0.48, "threshold": 0.4},
    {"criterion": "e_supply_chain", "met": true, "score": 0.52, "threshold": 0.4},
    {"criterion": "f_mixing_substitution", "met": false, "score": 0.15, "threshold": 0.3},
    {"criterion": "g_indigenous_rights", "met": false, "score": 0.20, "threshold": 0.3}
  ],
  "enhanced_dd_required": false
}
```

---

### POST /api/v1/eudr/risk-assessment-engine/override

Apply a manual risk score override to a completed assessment. Overrides
require justification and are recorded in the provenance audit trail per
EUDR Article 31.

**Request:**

```json
{
  "assessment_id": "rao_001",
  "overridden_score": 0.35,
  "reason": "expert_judgment",
  "justification": "On-site inspection confirmed supplier compliance; satellite anomaly was cloud shadow artifact",
  "overridden_by": "senior-analyst@company.com"
}
```

**Response (200 OK):**

```json
{
  "override_id": "ovr_001",
  "assessment_id": "rao_001",
  "original_score": "0.52",
  "overridden_score": "0.35",
  "reason": "expert_judgment",
  "overridden_by": "senior-analyst@company.com",
  "provenance_hash": "sha256:x1y2z3...",
  "audit_trail_id": "at_001",
  "created_at": "2026-04-04T10:15:00Z"
}
```

---

### GET /api/v1/eudr/risk-assessment-engine/trend/{operator_id}/{commodity}

Retrieve temporal risk trend analysis for a specific operator and commodity,
including trend direction, drift detection, and historical data points.

**Response (200 OK):**

```json
{
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "trend_direction": "improving",
  "current_score": "0.52",
  "score_30d_ago": "0.58",
  "score_90d_ago": "0.65",
  "drift_detected": false,
  "data_points": [
    {"date": "2025-07-01", "score": "0.65"},
    {"date": "2025-10-01", "score": "0.60"},
    {"date": "2026-01-01", "score": "0.58"},
    {"date": "2026-04-01", "score": "0.52"}
  ],
  "trend_confidence": 0.85,
  "next_assessment_recommended": "2026-07-01"
}
```

---

### GET /api/v1/eudr/risk-assessment-engine/benchmarks/{country_code}

Retrieve the EU-published benchmark classification for a specific country
(low/standard/high risk) per EUDR Article 29.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `country_code` | string | ISO 3166-1 alpha-2 country code |

**Response (200 OK):**

```json
{
  "country_code": "GH",
  "country_name": "Ghana",
  "benchmark_level": "standard",
  "effective_date": "2025-06-30",
  "indicators": {
    "deforestation_rate": 0.55,
    "governance_score": 0.48,
    "enforcement_capacity": 0.42
  },
  "eu_commission_reference": "C(2025)4567",
  "review_date": "2027-06-30"
}
```

---

### POST /api/v1/eudr/risk-assessment-engine/assess/batch

Execute multiple risk assessments in batch. Each assessment is processed
sequentially to maintain deterministic ordering and provenance chain integrity.

**Request:**

```json
{
  "assessments": [
    {
      "operator_id": "OP-2024-001",
      "commodity": "cocoa",
      "country_codes": ["GH"],
      "supplier_ids": ["sup-001"]
    },
    {
      "operator_id": "OP-2024-001",
      "commodity": "palm_oil",
      "country_codes": ["MY", "ID"],
      "supplier_ids": ["sup-005", "sup-006"]
    }
  ]
}
```

**Response (200 OK):**

```json
[
  {
    "operation_id": "rao_002",
    "status": "completed",
    "operator_id": "OP-2024-001",
    "commodity": "cocoa",
    "risk_level": "standard",
    "composite_score": {"overall_score": "0.48"}
  },
  {
    "operation_id": "rao_003",
    "status": "completed",
    "operator_id": "OP-2024-001",
    "commodity": "palm_oil",
    "risk_level": "high",
    "composite_score": {"overall_score": "0.71"}
  }
]
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 404 | `benchmark_not_found` | No EU benchmark for the specified country |
| 422 | `invalid_override` | Override parameters fail validation (score out of range, justification too short) |
| 500 | `assessment_failed` | Internal error during risk assessment pipeline |
| 503 | `service_unavailable` | Risk Assessment Engine service is initializing |
