# AGENT-EUDR-017: Supplier Risk Scorer API

**Agent ID:** `GL-EUDR-SRS-017`
**Prefix:** `/v1/eudr-srs`
**Version:** 1.0.0
**PRD:** AGENT-EUDR-017
**Regulation:** EU 2023/1115 (EUDR) -- Supplier risk assessment per Article 10

## Purpose

The Supplier Risk Scorer agent evaluates individual suppliers in the EUDR
supply chain for compliance risk. It computes risk scores based on supplier
documentation completeness, certification validity, geographic sourcing
patterns, network analysis (sub-supplier risk propagation), due diligence
history, and continuous monitoring signals. Scores drive due diligence
intensity decisions under EUDR Article 10.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/suppliers/score` | Score a supplier | JWT |
| GET | `/suppliers` | List scored suppliers | JWT |
| GET | `/suppliers/{supplier_id}` | Get supplier risk profile | JWT |
| GET | `/suppliers/{supplier_id}/history` | Get score history | JWT |
| POST | `/suppliers/batch` | Batch score suppliers | JWT |
| POST | `/suppliers/compare` | Compare supplier scores | JWT |
| POST | `/due-diligence/check` | Check DD compliance | JWT |
| GET | `/due-diligence/{supplier_id}` | Get DD status | JWT |
| POST | `/due-diligence/batch` | Batch DD check | JWT |
| GET | `/due-diligence/pending` | List pending DD | JWT |
| POST | `/due-diligence/complete` | Mark DD complete | JWT |
| POST | `/documentation/analyze` | Analyze documentation | JWT |
| GET | `/documentation/{supplier_id}` | Get doc status | JWT |
| POST | `/documentation/gaps` | Identify doc gaps | JWT |
| GET | `/documentation/expiring` | List expiring documents | JWT |
| POST | `/documentation/request` | Request missing documents | JWT |
| POST | `/certifications/validate` | Validate certification | JWT |
| GET | `/certifications/{supplier_id}` | Get certifications | JWT |
| POST | `/certifications/verify` | Verify with issuer | JWT |
| GET | `/certifications/expiring` | List expiring certs | JWT |
| POST | `/certifications/equivalence` | Check EUDR equivalence | JWT |
| POST | `/geographic/analyze` | Analyze sourcing geography | JWT |
| GET | `/geographic/{supplier_id}` | Get geographic profile | JWT |
| POST | `/geographic/overlap` | Check area overlap | JWT |
| GET | `/geographic/countries` | Get country breakdown | JWT |
| POST | `/geographic/risk-zones` | Identify risk zones | JWT |
| POST | `/network/analyze` | Analyze supplier network | JWT |
| GET | `/network/{supplier_id}` | Get network position | JWT |
| POST | `/network/propagate` | Propagate network risk | JWT |
| GET | `/network/clusters` | Get supplier clusters | JWT |
| GET | `/network/critical-paths` | Get critical supply paths | JWT |
| POST | `/monitoring/configure` | Configure monitoring | JWT |
| GET | `/monitoring/{supplier_id}` | Get monitoring status | JWT |
| POST | `/monitoring/alerts` | Get monitoring alerts | JWT |
| GET | `/monitoring/dashboard` | Get monitoring dashboard | JWT |
| POST | `/monitoring/schedule` | Schedule monitoring scan | JWT |
| POST | `/reports/generate` | Generate risk report | JWT |
| GET | `/reports` | List reports | JWT |
| GET | `/reports/{report_id}` | Get report details | JWT |
| GET | `/reports/{report_id}/download` | Download report | JWT |
| POST | `/reports/comparative` | Generate comparative report | JWT |
| POST | `/reports/portfolio` | Generate portfolio report | JWT |
| GET | `/health` | Health check | None |

**Total: 43 endpoints**

---

## Endpoints

### POST /v1/eudr-srs/suppliers/score

Calculate a comprehensive risk score for a supplier based on documentation,
certifications, geographic sourcing, network position, and historical
compliance.

**Request:**

```json
{
  "supplier_id": "sup-001",
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "include_network": true,
  "include_geographic": true,
  "scoring_model": "eudr_v2"
}
```

**Response (200 OK):**

```json
{
  "supplier_id": "sup-001",
  "supplier_name": "Adansi Farm Cooperative",
  "overall_risk_score": 0.35,
  "risk_level": "low",
  "dimension_scores": {
    "documentation": 0.25,
    "certification": 0.20,
    "geographic": 0.45,
    "network": 0.40,
    "historical": 0.30
  },
  "certifications": [
    {"type": "rainforest_alliance", "valid": true, "expiry": "2027-03-15"}
  ],
  "country_risk": "standard",
  "due_diligence_requirement": "standard",
  "recommendations": [
    "Request updated geolocation data for 3 plots"
  ],
  "scored_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-srs/network/analyze

Analyze the supplier network to identify risk propagation paths, critical
dependencies, and concentration risks.

**Request:**

```json
{
  "supplier_id": "sup-001",
  "depth": 3,
  "include_indirect": true,
  "risk_threshold": 0.6
}
```

**Response (200 OK):**

```json
{
  "supplier_id": "sup-001",
  "network_size": 24,
  "tiers_analyzed": 3,
  "high_risk_suppliers": 2,
  "concentration_risk": 0.42,
  "critical_paths": [
    {
      "path": ["sup-001", "sub-003", "farm-012"],
      "path_risk": 0.72,
      "bottleneck": "sub-003"
    }
  ],
  "clusters": [
    {"cluster_id": "cl_01", "suppliers": 8, "risk_level": "low"},
    {"cluster_id": "cl_02", "suppliers": 3, "risk_level": "high"}
  ],
  "analyzed_at": "2026-04-04T10:10:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_supplier` | Supplier data fails validation |
| 404 | `supplier_not_found` | Supplier ID not found |
| 422 | `invalid_commodity` | Commodity not in EUDR scope |
| 429 | `rate_limit_exceeded` | Too many requests |
