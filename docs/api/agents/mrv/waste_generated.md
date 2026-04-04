# Waste Generated in Operations Agent API Reference (AGENT-MRV-017)

## Overview

The Waste Generated in Operations Agent (GL-MRV-S3-005) calculates GHG Protocol Scope 3 Category 5 emissions from the disposal and treatment of waste generated in the reporting organization's operations. Provides treatment-specific calculation endpoints for landfill, incineration, recycling, composting, anaerobic digestion, and wastewater.

**API Prefix:** `/api/v1/waste-generated`
**Agent ID:** GL-MRV-S3-005
**Status:** Production Ready

**Treatment Methods:**
- Landfill (with/without gas capture)
- Incineration (with/without energy recovery)
- Recycling
- Composting
- Anaerobic digestion
- Wastewater treatment

**Calculation Methods:** waste_type_specific, average_data, supplier_specific

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/calculate` | General waste emission calculation | 201, 400, 500 |
| 2 | POST | `/calculate/batch` | Batch calculation | 201, 400, 500 |
| 3 | POST | `/calculate/landfill` | Landfill-specific calculation | 201, 400, 500 |
| 4 | POST | `/calculate/incineration` | Incineration-specific calculation | 201, 400, 500 |
| 5 | POST | `/calculate/recycling` | Recycling-specific calculation | 201, 400, 500 |
| 6 | POST | `/calculate/composting` | Composting-specific calculation | 201, 400, 500 |
| 7 | POST | `/calculate/anaerobic-digestion` | AD-specific calculation | 201, 400, 500 |
| 8 | POST | `/calculate/wastewater` | Wastewater-specific calculation | 201, 400, 500 |
| 9 | GET | `/calculations/{id}` | Get calculation | 200, 404, 500 |
| 10 | GET | `/calculations` | List calculations | 200, 500 |
| 11 | DELETE | `/calculations/{id}` | Delete calculation | 200, 404, 500 |
| 12 | GET | `/emission-factors` | List emission factors | 200, 500 |
| 13 | GET | `/emission-factors/{waste_type}` | Get factors by waste type | 200, 404, 500 |
| 14 | GET | `/waste-types` | List waste types | 200 |
| 15 | GET | `/treatment-methods` | List treatment methods | 200 |
| 16 | POST | `/compliance/check` | Check compliance | 201, 400, 500 |
| 17 | POST | `/uncertainty/analyze` | Run uncertainty analysis | 201, 400, 500 |
| 18 | GET | `/aggregations/{period}` | Get aggregated results | 200, 400, 500 |
| 19 | POST | `/diversion/analyze` | Analyze waste diversion rates | 201, 400, 500 |
| 20 | GET | `/provenance/{calculation_id}` | Get provenance chain | 200, 404, 500 |

---

## Endpoints

### 1. POST /calculate

General waste emission calculation that auto-selects the treatment method based on input data.

**Request Body:**

```json
{
  "waste_type": "municipal_solid_waste",
  "treatment_method": "landfill",
  "quantity_tonnes": 500.0,
  "waste_composition": {
    "food_pct": 30.0,
    "paper_pct": 25.0,
    "plastic_pct": 15.0,
    "garden_pct": 10.0,
    "other_pct": 20.0
  },
  "facility_id": "facility-001",
  "reporting_period": "2025",
  "tenant_id": "tenant-001"
}
```

**Response (201 Created):**

```json
{
  "calculation_id": "wg_abc123",
  "waste_type": "municipal_solid_waste",
  "treatment_method": "landfill",
  "total_co2e_kg": 125000.0,
  "ch4_kg": 3200.0,
  "co2_kg": 45000.0,
  "n2o_kg": 15.0,
  "biogenic_co2_kg": 80000.0,
  "quantity_tonnes": 500.0,
  "ef_used_kgco2e_per_tonne": 250.0,
  "provenance_hash": "sha256:k7l8m9n0...",
  "created_at": "2026-04-01T10:30:00Z"
}
```

---

### 3. POST /calculate/landfill

Landfill-specific calculation with gas capture, methane oxidation factor, and first-order decay parameters.

**Request Body:**

```json
{
  "waste_type": "municipal_solid_waste",
  "quantity_tonnes": 500.0,
  "methane_generation_potential": 0.06,
  "methane_correction_factor": 1.0,
  "oxidation_factor": 0.10,
  "gas_capture_efficiency": 0.75,
  "doc_fraction": 0.15,
  "facility_id": "facility-001",
  "tenant_id": "tenant-001"
}
```

---

### 4. POST /calculate/incineration

Incineration-specific calculation with fossil carbon fraction, oxidation factor, and energy recovery credits.

---

### 5. POST /calculate/recycling

Recycling-specific calculation accounting for avoided primary production emissions.

---

### 6. POST /calculate/composting

Composting-specific calculation with CH4 and N2O emission factors based on composting technology.

---

### 7. POST /calculate/anaerobic-digestion

Anaerobic digestion calculation with biogas capture rates and digestate handling emissions.

---

### 8. POST /calculate/wastewater

Wastewater treatment calculation with BOD/COD loading, treatment type, and sludge handling.

---

### 14. GET /waste-types

List all supported waste types with descriptions and typical emission factor ranges.

---

### 15. GET /treatment-methods

List all supported treatment methods with descriptions and applicable waste types.

---

### 19. POST /diversion/analyze

Analyze waste diversion rates (landfill diversion percentage, recycling rate, recovery rate) and their emission reduction impact.

---

### 20. GET /provenance/{calculation_id}

Retrieve complete SHA-256 provenance tracking for a calculation, including input data hash, emission factor hash, and audit trail.

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid waste type, treatment method, or parameters |
| 404 | Not Found -- Calculation not found |
| 500 | Internal Server Error |
