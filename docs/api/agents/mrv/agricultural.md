# Agricultural Emissions Agent API Reference (AGENT-MRV-008)

## Overview

The Agricultural Emissions Agent (GL-MRV-X-008) calculates GHG Protocol Scope 1 agricultural emissions following IPCC 2006 Guidelines Volume 4 Chapter 10-11. Covers enteric fermentation, manure management, agricultural soils (direct and indirect N2O), rice cultivation, field burning, liming, and urea application.

**API Prefix:** `/api/v1/agricultural-emissions`
**Agent ID:** GL-MRV-X-008
**Status:** Production Ready

**Emission Sources:**
- `enteric_fermentation` -- CH4 from ruminant digestion
- `manure_management` -- CH4 and N2O from manure storage/treatment
- `agricultural_soils` -- Direct and indirect N2O from fertilizer/residues
- `rice_cultivation` -- CH4 from flooded rice paddies
- `field_burning` -- CH4, N2O, CO, NOx from crop residue burning
- `liming` -- CO2 from limestone and dolomite application
- `urea_application` -- CO2 from urea hydrolysis

**Animal Types:** dairy_cattle, non_dairy_cattle, buffalo, sheep, goats, camels, horses, swine, poultry, and others

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/calculations` | Create agricultural calculation | 201, 400, 500 |
| 2 | POST | `/calculations/batch` | Batch calculations | 201, 400, 500 |
| 3 | GET | `/calculations` | List calculations | 200, 500 |
| 4 | GET | `/calculations/{calc_id}` | Get calculation | 200, 404, 500 |
| 5 | DELETE | `/calculations/{calc_id}` | Delete calculation | 200, 404, 500 |
| 6 | POST | `/farms` | Register farm | 201, 400, 500 |
| 7 | GET | `/farms` | List farms | 200, 500 |
| 8 | GET | `/farms/{farm_id}` | Get farm | 200, 404, 500 |
| 9 | PUT | `/farms/{farm_id}` | Update farm | 200, 400, 404, 500 |
| 10 | POST | `/livestock` | Register livestock herd | 201, 400, 500 |
| 11 | GET | `/livestock` | List livestock | 200, 500 |
| 12 | GET | `/livestock/{herd_id}` | Get livestock herd | 200, 404, 500 |
| 13 | PUT | `/livestock/{herd_id}` | Update livestock herd | 200, 400, 404, 500 |
| 14 | POST | `/cropland-inputs` | Create cropland input record | 201, 400, 500 |
| 15 | GET | `/cropland-inputs` | List cropland inputs | 200, 500 |
| 16 | POST | `/rice-fields` | Register rice field | 201, 400, 500 |
| 17 | GET | `/rice-fields` | List rice fields | 200, 500 |
| 18 | POST | `/compliance/check` | Check compliance | 200, 400, 500 |
| 19 | GET | `/compliance/{compliance_id}` | Get compliance result | 200, 404, 500 |
| 20 | POST | `/uncertainty` | Run uncertainty analysis | 200, 400, 500 |

---

## Endpoints

### 1. POST /calculations

Create an agricultural emission calculation.

**Request Body:**

```json
{
  "farm_id": "farm-001",
  "emission_source": "enteric_fermentation",
  "animal_type": "dairy_cattle",
  "head_count": 500,
  "weight_kg": 600,
  "milk_yield_kg_per_year": 8000,
  "feed_digestibility_pct": 65.0,
  "tier": "tier_2",
  "climate_zone": "temperate",
  "gwp_source": "AR6",
  "tenant_id": "tenant-001"
}
```

**Response (201 Created):**

```json
{
  "calc_id": "ag_abc123",
  "farm_id": "farm-001",
  "emission_source": "enteric_fermentation",
  "total_co2e_kg": 1050000.0,
  "ch4_kg": 37500.0,
  "n2o_kg": 0.0,
  "tier": "tier_2",
  "ef_used_kg_ch4_per_head": 75.0,
  "provenance_hash": "sha256:m3n4o5p6...",
  "created_at": "2026-04-01T10:30:00Z"
}
```

---

### 6. POST /farms

Register a farm with geographic and operational metadata.

**Request Body:**

```json
{
  "name": "Green Valley Dairy Farm",
  "country": "US",
  "region": "Wisconsin",
  "climate_zone": "temperate",
  "farm_type": "dairy",
  "area_ha": 250.0,
  "latitude": 43.0731,
  "longitude": -89.4012,
  "tenant_id": "tenant-001"
}
```

---

### 10. POST /livestock

Register a livestock herd for a farm.

**Request Body:**

```json
{
  "farm_id": "farm-001",
  "animal_type": "dairy_cattle",
  "head_count": 500,
  "average_weight_kg": 600,
  "milk_yield_kg_per_year": 8000,
  "manure_system": "lagoon",
  "grazing_fraction": 0.3,
  "tenant_id": "tenant-001"
}
```

---

### 14. POST /cropland-inputs

Record cropland fertilizer, residue, and amendment inputs for soil N2O calculations.

**Request Body:**

```json
{
  "farm_id": "farm-001",
  "input_type": "synthetic_fertilizer",
  "nitrogen_kg": 15000.0,
  "area_ha": 100.0,
  "crop_type": "corn",
  "application_method": "broadcast",
  "tenant_id": "tenant-001"
}
```

---

### 16. POST /rice-fields

Register a rice field for CH4 emission calculations from flooded paddies.

---

### 18. POST /compliance/check

Check results against GHG Protocol, IPCC 2006 Vol 4, CSRD ESRS E1, ISO 14064, national agriculture reporting.

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid emission source, animal type, or parameters |
| 404 | Not Found -- Farm, herd, or calculation not found |
| 500 | Internal Server Error |
