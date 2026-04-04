# AGENT-EUDR-018: Commodity Risk Analyzer API

**Agent ID:** `GL-EUDR-CRA-018`
**Prefix:** `/v1/eudr-cra`
**Version:** 1.0.0
**PRD:** AGENT-EUDR-018
**Regulation:** EU 2023/1115 (EUDR) -- Commodity-specific risk per Article 10(2)

## Purpose

The Commodity Risk Analyzer agent assesses risk at the commodity level for
the seven EUDR-regulated commodities (cattle, cocoa, coffee, oil palm,
rubber, soya, wood) and their derived products. It builds commodity risk
profiles, traces derived product chains, monitors price volatility and
supply disruptions, detects substitution fraud, tracks regulatory
requirements, manages due diligence workflows, and analyzes portfolio
concentration.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/commodities/profile` | Create commodity risk profile | JWT |
| POST | `/commodities/batch` | Batch create profiles | JWT |
| GET | `/commodities/{commodity}/risk` | Get commodity risk | JWT |
| GET | `/commodities/{commodity}/history` | Get risk history | JWT |
| GET | `/commodities/compare` | Compare commodities | JWT |
| GET | `/commodities/summary` | Get summary of all | JWT |
| POST | `/derived-products/analyze` | Analyze derived product | JWT |
| GET | `/derived-products/{product_id}/chain` | Get derivation chain | JWT |
| GET | `/derived-products/{product_id}/risk` | Get derived product risk | JWT |
| GET | `/derived-products/mapping` | Get HS code mapping | JWT |
| POST | `/derived-products/trace` | Trace product origin | JWT |
| GET | `/prices/current` | Get current prices | JWT |
| GET | `/prices/history` | Get price history | JWT |
| GET | `/prices/volatility` | Get price volatility | JWT |
| GET | `/prices/disruptions` | List supply disruptions | JWT |
| POST | `/prices/forecast` | Forecast price trend | JWT |
| POST | `/production/forecast` | Forecast production | JWT |
| GET | `/production/yield` | Get yield data | JWT |
| GET | `/production/climate-impact` | Get climate impact | JWT |
| GET | `/production/seasonal` | Get seasonal patterns | JWT |
| GET | `/production/summary` | Get production summary | JWT |
| POST | `/substitution/detect` | Detect substitution fraud | JWT |
| GET | `/substitution/history` | Get substitution history | JWT |
| GET | `/substitution/alerts` | List substitution alerts | JWT |
| POST | `/substitution/verify` | Verify commodity identity | JWT |
| GET | `/substitution/patterns` | Get known patterns | JWT |
| GET | `/regulatory/requirements` | Get regulatory requirements | JWT |
| POST | `/regulatory/check` | Check regulatory compliance | JWT |
| GET | `/regulatory/penalty` | Get penalty schedule | JWT |
| GET | `/regulatory/updates` | Get regulatory updates | JWT |
| GET | `/regulatory/docs` | Get regulatory documents | JWT |
| POST | `/due-diligence/initiate` | Initiate commodity DD | JWT |
| GET | `/due-diligence/{dd_id}/status` | Get DD status | JWT |
| POST | `/due-diligence/{dd_id}/evidence` | Add DD evidence | JWT |
| GET | `/due-diligence/pending` | List pending DD | JWT |
| POST | `/due-diligence/{dd_id}/complete` | Complete DD | JWT |
| POST | `/portfolio/analyze` | Analyze portfolio | JWT |
| GET | `/portfolio/concentration` | Get concentration risk | JWT |
| GET | `/portfolio/diversification` | Get diversification score | JWT |
| GET | `/portfolio/summary` | Get portfolio summary | JWT |
| GET | `/health` | Health check | None |

**Total: 42 endpoints**

---

## Endpoints

### POST /v1/eudr-cra/commodities/profile

Create a comprehensive risk profile for an EUDR-regulated commodity,
analyzing sourcing countries, price trends, supply chain complexity,
regulatory landscape, and substitution risk.

**Request:**

```json
{
  "commodity": "cocoa",
  "operator_id": "OP-2024-001",
  "sourcing_countries": ["GH", "CI", "CM"],
  "volume_tonnes": 5000.0,
  "period": "2026-Q1"
}
```

**Response (200 OK):**

```json
{
  "commodity": "cocoa",
  "overall_risk_score": 0.48,
  "risk_level": "standard",
  "country_risk_breakdown": {
    "GH": {"score": 0.52, "level": "standard"},
    "CI": {"score": 0.61, "level": "high"},
    "CM": {"score": 0.45, "level": "standard"}
  },
  "price_volatility": 0.18,
  "supply_disruption_risk": "low",
  "substitution_risk": 0.12,
  "derived_products_count": 47,
  "regulatory_requirements": {
    "due_diligence_level": "standard",
    "simplified_eligible": false,
    "enhanced_required_countries": ["CI"]
  },
  "profiled_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-cra/substitution/detect

Detect potential commodity substitution fraud by analyzing product
characteristics, pricing anomalies, and supply chain patterns.

**Request:**

```json
{
  "batch_id": "batch-001",
  "commodity": "cocoa",
  "claimed_origin": "GH",
  "product_characteristics": {
    "fat_content_pct": 54.2,
    "moisture_pct": 3.1,
    "shell_content_pct": 1.8
  },
  "price_per_tonne": 3200.0
}
```

**Response (200 OK):**

```json
{
  "detection_id": "sub_001",
  "substitution_detected": false,
  "confidence": 0.92,
  "risk_score": 0.08,
  "checks_performed": [
    {"check": "characteristic_analysis", "passed": true, "note": "Within expected range for Ghana cocoa"},
    {"check": "price_anomaly", "passed": true, "note": "Price consistent with current market"},
    {"check": "origin_pattern", "passed": true, "note": "Volume consistent with Ghana exports"}
  ],
  "detected_at": "2026-04-04T10:10:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_commodity` | Commodity not in EUDR scope |
| 404 | `profile_not_found` | Commodity profile not found |
| 422 | `invalid_hs_code` | HS code does not map to EUDR commodity |
| 503 | `market_data_unavailable` | Price feed provider is down |
