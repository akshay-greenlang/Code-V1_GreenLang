# AGENT-EUDR-004: Forest Cover Analysis API

**Agent ID:** `GL-EUDR-FCA-004`
**Prefix:** `/v1/eudr-fca`
**Version:** 1.0.0
**PRD:** GL-EUDR-FCA-004
**Regulation:** EU 2023/1115 (EUDR) -- Deforestation-free verification per Article 2(1)

## Purpose

The Forest Cover Analysis agent determines whether production plots meet the
EUDR "deforestation-free" requirement by analyzing canopy density, classifying
forest types against FAO definitions, reconstructing historical forest cover
to the December 31, 2020 cutoff date, verifying deforestation-free status,
computing biomass and fragmentation metrics, and generating compliance reports.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/density/analyze` | Analyze canopy density | JWT |
| POST | `/density/batch` | Batch canopy density analysis | JWT |
| GET | `/density/{analysis_id}` | Get density analysis result | JWT |
| GET | `/density/{analysis_id}/history` | Get density change history | JWT |
| POST | `/density/compare` | Compare density between periods | JWT |
| POST | `/classify` | Classify forest type | JWT |
| POST | `/classify/batch` | Batch forest classification | JWT |
| GET | `/classify/{classification_id}` | Get classification result | JWT |
| GET | `/classify/types` | List supported forest types | JWT |
| POST | `/historical/reconstruct` | Reconstruct historical cover | JWT |
| POST | `/historical/batch` | Batch historical reconstruction | JWT |
| GET | `/historical/{reconstruction_id}` | Get reconstruction result | JWT |
| POST | `/historical/compare` | Compare historical periods | JWT |
| GET | `/historical/{reconstruction_id}/sources` | Get data sources used | JWT |
| POST | `/verify` | Verify deforestation-free status | JWT |
| POST | `/verify/batch` | Batch deforestation-free check | JWT |
| GET | `/verify/{verification_id}` | Get verification result | JWT |
| GET | `/verify/{verification_id}/evidence` | Get verification evidence | JWT |
| POST | `/verify/complete` | Mark verification complete | JWT |
| POST | `/analysis/height` | Analyze canopy height | JWT |
| POST | `/analysis/fragmentation` | Analyze forest fragmentation | JWT |
| POST | `/analysis/biomass` | Estimate above-ground biomass | JWT |
| GET | `/analysis/{analysis_id}/profile` | Get analysis profile | JWT |
| POST | `/analysis/compare` | Compare analysis results | JWT |
| POST | `/reports/generate` | Generate compliance report | JWT |
| GET | `/reports/{report_id}` | Get report details | JWT |
| GET | `/reports/{report_id}/download` | Download report file | JWT |
| POST | `/reports/batch` | Batch report generation | JWT |
| GET | `/health` | Health check | None |
| GET | `/version` | API version info | None |

**Total: 30 endpoints + health + version**

---

## Endpoints

### POST /v1/eudr-fca/density/analyze

Analyze canopy density for a production plot using satellite-derived canopy
cover metrics. The FAO threshold of 10% canopy cover at 5m height is used
as the baseline forest definition.

**Request:**

```json
{
  "plot_id": "plot-GH-001",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[-1.5, 6.0], [-1.5, 6.01], [-1.49, 6.01], [-1.49, 6.0], [-1.5, 6.0]]]
  },
  "analysis_date": "2026-01-15",
  "data_source": "sentinel2",
  "resolution_m": 10
}
```

**Response (200 OK):**

```json
{
  "analysis_id": "den_001",
  "plot_id": "plot-GH-001",
  "canopy_cover_pct": 72.3,
  "is_forest": true,
  "fao_threshold_met": true,
  "mean_height_m": 18.5,
  "analysis_date": "2026-01-15",
  "confidence": 0.89,
  "data_source": "sentinel2",
  "completed_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-fca/verify

Verify whether a production plot qualifies as "deforestation-free" under
EUDR Article 2(1) by comparing current forest cover against the December 31,
2020 baseline.

**Request:**

```json
{
  "plot_id": "plot-GH-001",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[-1.5, 6.0], [-1.5, 6.01], [-1.49, 6.01], [-1.49, 6.0], [-1.5, 6.0]]]
  },
  "cutoff_date": "2020-12-31",
  "commodity": "cocoa"
}
```

**Response (200 OK):**

```json
{
  "verification_id": "vfy_001",
  "plot_id": "plot-GH-001",
  "deforestation_free": true,
  "baseline_canopy_pct": 74.1,
  "current_canopy_pct": 72.3,
  "change_pct": -1.8,
  "cutoff_date": "2020-12-31",
  "verification_date": "2026-01-15",
  "status": "verified",
  "risk_level": "low",
  "evidence_count": 3,
  "provenance_hash": "sha256:f1g2h3i4...",
  "completed_at": "2026-04-04T10:10:00Z"
}
```

---

### POST /v1/eudr-fca/historical/reconstruct

Reconstruct historical forest cover for a plot back to the EUDR cutoff date
using multi-temporal satellite imagery and land cover classification models.

**Request:**

```json
{
  "plot_id": "plot-GH-001",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[-1.5, 6.0], [-1.5, 6.01], [-1.49, 6.01], [-1.49, 6.0], [-1.5, 6.0]]]
  },
  "start_date": "2018-01-01",
  "end_date": "2026-01-15",
  "interval_months": 6
}
```

**Response (200 OK):**

```json
{
  "reconstruction_id": "recon_001",
  "plot_id": "plot-GH-001",
  "time_series": [
    {"date": "2018-01-01", "canopy_pct": 78.2, "classification": "tropical_moist"},
    {"date": "2018-07-01", "canopy_pct": 77.8, "classification": "tropical_moist"},
    {"date": "2020-12-31", "canopy_pct": 74.1, "classification": "tropical_moist"},
    {"date": "2026-01-15", "canopy_pct": 72.3, "classification": "tropical_moist"}
  ],
  "trend": "gradual_decline",
  "deforestation_event_detected": false,
  "data_sources": ["sentinel2", "landsat8", "hansen_gfc"],
  "confidence": 0.87
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_geometry` | GeoJSON geometry is invalid |
| 404 | `analysis_not_found` | Analysis or verification ID not found |
| 422 | `unsupported_commodity` | Commodity not in EUDR scope |
| 503 | `data_source_unavailable` | Satellite data source is down |
