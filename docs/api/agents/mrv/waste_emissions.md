# Waste Treatment Emissions Agent API Reference (AGENT-MRV-007)

## Overview

The Waste Treatment Emissions Agent (GL-MRV-X-007) calculates GHG Protocol Scope 1 emissions from biological and thermal waste treatment processes following IPCC 2006 Guidelines Volume 5. Covers composting, incineration, anaerobic digestion, open burning, landfill, mechanical-biological treatment, autoclaving, pyrolysis, gasification, and wastewater treatment.

**API Prefix:** `/api/v1/waste-treatment-emissions`
**Agent ID:** GL-MRV-X-007
**Status:** Production Ready

**Treatment Methods:** composting, incineration, anaerobic_digestion, open_burning, landfill, mechanical_biological, autoclaving, pyrolysis, gasification, wastewater_treatment

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/calculations` | Create waste treatment calculation | 201, 400, 500 |
| 2 | POST | `/calculations/batch` | Batch calculations | 201, 400, 500 |
| 3 | GET | `/calculations` | List calculations | 200, 500 |
| 4 | GET | `/calculations/{calc_id}` | Get calculation | 200, 404, 500 |
| 5 | DELETE | `/calculations/{calc_id}` | Delete calculation | 200, 404, 500 |
| 6 | POST | `/facilities` | Register treatment facility | 201, 400, 500 |
| 7 | GET | `/facilities` | List facilities | 200, 500 |
| 8 | GET | `/facilities/{facility_id}` | Get facility | 200, 404, 500 |
| 9 | PUT | `/facilities/{facility_id}` | Update facility | 200, 400, 404, 500 |
| 10 | POST | `/waste-streams` | Register waste stream | 201, 400, 500 |
| 11 | GET | `/waste-streams` | List waste streams | 200, 500 |
| 12 | GET | `/waste-streams/{stream_id}` | Get waste stream | 200, 404, 500 |
| 13 | PUT | `/waste-streams/{stream_id}` | Update waste stream | 200, 400, 404, 500 |
| 14 | POST | `/treatment-events` | Create treatment event | 201, 400, 500 |
| 15 | GET | `/treatment-events` | List treatment events | 200, 500 |
| 16 | POST | `/methane-recovery` | Create methane recovery record | 201, 400, 500 |
| 17 | GET | `/methane-recovery/{facility_id}/history` | Get recovery history | 200, 404, 500 |
| 18 | POST | `/compliance/check` | Check compliance | 200, 400, 500 |
| 19 | GET | `/compliance/{compliance_id}` | Get compliance result | 200, 404, 500 |
| 20 | POST | `/uncertainty` | Run uncertainty analysis | 200, 400, 500 |

---

## Endpoints

### 1. POST /calculations

Create a waste treatment emission calculation.

**Request Body:**

```json
{
  "facility_id": "wt-facility-001",
  "treatment_method": "incineration",
  "waste_type": "municipal_solid_waste",
  "waste_quantity_tonnes": 5000.0,
  "waste_composition": {
    "food_waste_pct": 25.0,
    "paper_pct": 20.0,
    "plastic_pct": 15.0,
    "textile_pct": 5.0,
    "other_pct": 35.0
  },
  "dry_matter_fraction": 0.65,
  "fossil_carbon_fraction": 0.40,
  "oxidation_factor": 0.95,
  "gwp_source": "AR6",
  "tenant_id": "tenant-001"
}
```

**Response (201 Created):**

```json
{
  "calc_id": "wt_abc123",
  "facility_id": "wt-facility-001",
  "treatment_method": "incineration",
  "total_co2e_kg": 1250000.0,
  "co2_fossil_kg": 1200000.0,
  "co2_biogenic_kg": 850000.0,
  "ch4_kg": 50.0,
  "n2o_kg": 25.0,
  "provenance_hash": "sha256:i9j0k1l2...",
  "created_at": "2026-04-01T10:30:00Z"
}
```

---

### 6. POST /facilities

Register a waste treatment facility with treatment capability metadata.

**Request Body:**

```json
{
  "name": "Municipal Incinerator Plant A",
  "treatment_methods": ["incineration"],
  "capacity_tonnes_per_year": 100000,
  "country": "DE",
  "region": "Bavaria",
  "has_energy_recovery": true,
  "has_methane_capture": false,
  "tenant_id": "tenant-001"
}
```

---

### 10. POST /waste-streams

Register a waste stream with composition and classification data.

---

### 14. POST /treatment-events

Record an individual waste treatment event linking a waste stream to a facility and treatment method.

---

### 16. POST /methane-recovery

Record methane recovery/capture data for landfill or anaerobic digestion facilities. Captured methane reduces net emissions.

---

### 18. POST /compliance/check

Check results against GHG Protocol, IPCC 2006, CSRD ESRS E1, ISO 14064, and national waste reporting frameworks.

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid treatment method, waste type, or composition |
| 404 | Not Found -- Facility, stream, or calculation not found |
| 500 | Internal Server Error |
