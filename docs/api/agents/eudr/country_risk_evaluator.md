# AGENT-EUDR-016: Country Risk Evaluator API

**Agent ID:** `GL-EUDR-CRE-016`
**Prefix:** `/v1/eudr-cre`
**Version:** 1.0.0
**PRD:** AGENT-EUDR-016
**Regulation:** EU 2023/1115 (EUDR) -- Country risk benchmarking per Article 29

## Purpose

The Country Risk Evaluator agent assesses country-level risk for EUDR
compliance by evaluating deforestation rates, governance quality, corruption
indices, rule of law, trade flow patterns, and regulatory enforcement
capacity. It maps to the EU Commission's three-tier country benchmarking
system (low / standard / high risk) defined in Article 29 and generates
due diligence requirements based on country classification.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/countries/assess` | Assess country risk | JWT |
| GET | `/countries` | List assessed countries | JWT |
| GET | `/countries/{country_code}` | Get country risk profile | JWT |
| GET | `/countries/{country_code}/history` | Get country risk history | JWT |
| POST | `/countries/compare` | Compare countries | JWT |
| POST | `/commodities/risk` | Get commodity-country risk | JWT |
| GET | `/commodities/{commodity}/countries` | List countries for commodity | JWT |
| POST | `/commodities/matrix` | Get risk matrix | JWT |
| GET | `/commodities/summary` | Get commodity risk summary | JWT |
| POST | `/hotspots/detect` | Detect deforestation hotspots | JWT |
| GET | `/hotspots` | List hotspots | JWT |
| GET | `/hotspots/{hotspot_id}` | Get hotspot details | JWT |
| POST | `/hotspots/monitor` | Start hotspot monitoring | JWT |
| GET | `/hotspots/alerts` | Get hotspot alerts | JWT |
| POST | `/governance/evaluate` | Evaluate governance score | JWT |
| GET | `/governance/{country_code}` | Get governance profile | JWT |
| GET | `/governance/indicators` | List governance indicators | JWT |
| POST | `/governance/trend` | Get governance trend | JWT |
| POST | `/due-diligence/classify` | Classify DD requirements | JWT |
| GET | `/due-diligence/{country_code}` | Get DD classification | JWT |
| POST | `/due-diligence/batch` | Batch DD classification | JWT |
| GET | `/due-diligence/simplified` | List simplified DD eligible | JWT |
| POST | `/due-diligence/enhanced` | Get enhanced DD requirements | JWT |
| POST | `/trade-flows/analyze` | Analyze trade flows | JWT |
| GET | `/trade-flows/{country_code}` | Get trade flow data | JWT |
| POST | `/trade-flows/disruptions` | Detect supply disruptions | JWT |
| GET | `/trade-flows/routes` | Get trade routes | JWT |
| GET | `/trade-flows/sanctions` | Check sanctions | JWT |
| POST | `/reports/generate` | Generate country risk report | JWT |
| GET | `/reports` | List reports | JWT |
| GET | `/reports/{report_id}` | Get report details | JWT |
| GET | `/reports/{report_id}/download` | Download report | JWT |
| POST | `/reports/comparative` | Generate comparative report | JWT |
| GET | `/regulatory/updates` | Get regulatory updates | JWT |
| POST | `/regulatory/impact` | Assess regulatory impact | JWT |
| GET | `/regulatory/eu-benchmarks` | Get EU benchmark list | JWT |
| POST | `/regulatory/alert` | Set regulatory alert | JWT |
| GET | `/health` | Health check | None |

**Total: 38 endpoints**

---

## Endpoints

### POST /v1/eudr-cre/countries/assess

Perform a comprehensive country-level risk assessment for EUDR compliance,
aggregating deforestation data, governance indicators, corruption indices,
trade flow patterns, and regulatory enforcement metrics.

**Request:**

```json
{
  "country_code": "GH",
  "commodities": ["cocoa", "palm_oil"],
  "include_subnational": true,
  "data_sources": ["fao", "wri_gfw", "transparency_intl", "world_bank"]
}
```

**Response (200 OK):**

```json
{
  "country_code": "GH",
  "country_name": "Ghana",
  "overall_risk_level": "standard",
  "overall_risk_score": 0.52,
  "eu_benchmark": "standard",
  "dimension_scores": {
    "deforestation": 0.58,
    "governance": 0.45,
    "corruption": 0.51,
    "rule_of_law": 0.48,
    "regulatory_enforcement": 0.55
  },
  "commodity_risks": {
    "cocoa": {"risk_level": "standard", "score": 0.55},
    "palm_oil": {"risk_level": "high", "score": 0.68}
  },
  "due_diligence_requirement": "standard",
  "assessed_at": "2026-04-04T10:00:00Z",
  "data_freshness": "2026-03-28"
}
```

---

### POST /v1/eudr-cre/hotspots/detect

Detect active deforestation hotspots within a country using near-real-time
satellite data and historical trend analysis.

**Request:**

```json
{
  "country_code": "GH",
  "commodity": "cocoa",
  "min_area_hectares": 10.0,
  "date_range_start": "2025-06-01",
  "date_range_end": "2026-01-31"
}
```

**Response (200 OK):**

```json
{
  "country_code": "GH",
  "total_hotspots": 7,
  "hotspots": [
    {
      "hotspot_id": "hs_gh_001",
      "centroid": {"latitude": 6.35, "longitude": -2.10},
      "area_hectares": 45.2,
      "severity": "high",
      "first_detected": "2025-09-15",
      "last_detected": "2026-01-20",
      "commodity_overlap": "cocoa",
      "region": "Western Region"
    }
  ],
  "detected_at": "2026-04-04T10:10:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_country_code` | Country code not recognized (ISO 3166-1) |
| 404 | `country_not_assessed` | No assessment data for this country |
| 422 | `invalid_commodity` | Commodity not in EUDR scope |
| 503 | `data_source_unavailable` | External data source is down |
