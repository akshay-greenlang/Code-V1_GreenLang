# Land Use Emissions Agent API Reference (AGENT-MRV-006)

## Overview

The Land Use Emissions Agent (GL-MRV-X-006) calculates GHG Protocol Scope 1 emissions from land use, land-use change, and forestry (LULUCF) following IPCC 2006 Guidelines Volume 4. Covers carbon stock changes across 6 IPCC land categories and 5 carbon pools using stock-difference and gain-loss methods.

**API Prefix:** `/api/v1/land-use-emissions`
**Agent ID:** GL-MRV-X-006
**Status:** Production Ready

**IPCC Land Categories:** forest_land, cropland, grassland, wetland, settlement, other_land

**Carbon Pools:** above_ground_biomass, below_ground_biomass, dead_wood, litter, soil_organic_carbon

**Calculation Methods:** stock_difference, gain_loss

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/calculations` | Create LULUCF calculation | 201, 400, 500 |
| 2 | POST | `/calculations/batch` | Batch LULUCF calculations | 201, 400, 500 |
| 3 | GET | `/calculations` | List calculations | 200, 500 |
| 4 | GET | `/calculations/{calc_id}` | Get calculation | 200, 404, 500 |
| 5 | DELETE | `/calculations/{calc_id}` | Delete calculation | 200, 404, 500 |
| 6 | POST | `/carbon-stocks` | Create carbon stock record | 201, 400, 500 |
| 7 | GET | `/carbon-stocks/{stock_id}` | Get carbon stock | 200, 404, 500 |
| 8 | GET | `/carbon-stocks/summary` | Carbon stock summary | 200, 500 |
| 9 | POST | `/land-parcels` | Register land parcel | 201, 400, 500 |
| 10 | GET | `/land-parcels` | List land parcels | 200, 500 |
| 11 | GET | `/land-parcels/{parcel_id}` | Get land parcel | 200, 404, 500 |
| 12 | PUT | `/land-parcels/{parcel_id}` | Update land parcel | 200, 400, 404, 500 |
| 13 | POST | `/transitions` | Create land-use transition | 201, 400, 500 |
| 14 | GET | `/transitions` | List transitions | 200, 500 |
| 15 | GET | `/transitions/matrix` | Transition matrix | 200, 500 |
| 16 | POST | `/soc-assessments` | Create SOC assessment | 201, 400, 500 |
| 17 | GET | `/soc-assessments/{assessment_id}` | Get SOC assessment | 200, 404, 500 |
| 18 | POST | `/compliance/check` | Check compliance | 200, 400, 500 |
| 19 | GET | `/compliance/{compliance_id}` | Get compliance result | 200, 404, 500 |
| 20 | POST | `/uncertainty` | Run uncertainty analysis | 200, 400, 500 |

---

## Endpoints

### 1. POST /calculations

Create a LULUCF carbon stock change calculation.

**Request Body:**

```json
{
  "facility_id": "site-001",
  "land_category": "forest_land",
  "method": "stock_difference",
  "area_ha": 150.0,
  "carbon_stock_initial_tC": 12500.0,
  "carbon_stock_final_tC": 12350.0,
  "period_years": 5,
  "carbon_pools": ["above_ground_biomass", "below_ground_biomass", "soil_organic_carbon"],
  "climate_zone": "tropical_moist",
  "gwp_source": "AR6",
  "tenant_id": "tenant-001"
}
```

**Response (201 Created):**

```json
{
  "calc_id": "lu_abc123",
  "facility_id": "site-001",
  "land_category": "forest_land",
  "method": "stock_difference",
  "total_co2e_tonnes": -550.0,
  "annual_co2e_tonnes": -110.0,
  "carbon_stock_change_tC": -150.0,
  "by_pool": {
    "above_ground_biomass": -90.0,
    "below_ground_biomass": -25.0,
    "soil_organic_carbon": -35.0
  },
  "provenance_hash": "sha256:e5f6g7h8...",
  "created_at": "2026-04-01T10:30:00Z"
}
```

---

### 2. POST /calculations/batch

Batch multiple LULUCF calculations. Accepts a list of calculation requests and returns individual and aggregate results.

---

### 5. DELETE /calculations/{calc_id}

Soft-delete a calculation. Data is retained for audit compliance.

---

### 6. POST /carbon-stocks

Create a carbon stock inventory record for a land parcel.

**Request Body:**

```json
{
  "parcel_id": "parcel-001",
  "measurement_date": "2026-01-15",
  "carbon_pools": {
    "above_ground_biomass": 85.0,
    "below_ground_biomass": 22.0,
    "dead_wood": 5.0,
    "litter": 3.0,
    "soil_organic_carbon": 120.0
  },
  "measurement_method": "field_inventory",
  "unit": "tC_per_ha",
  "tenant_id": "tenant-001"
}
```

---

### 9. POST /land-parcels

Register a land parcel with geographic and classification data.

**Request Body:**

```json
{
  "name": "North Forest Block",
  "land_category": "forest_land",
  "area_ha": 150.0,
  "country": "DE",
  "climate_zone": "temperate_continental",
  "soil_type": "mineral",
  "latitude": 51.1657,
  "longitude": 10.4515,
  "tenant_id": "tenant-001"
}
```

---

### 13. POST /transitions

Record a land-use change transition between IPCC categories.

**Request Body:**

```json
{
  "parcel_id": "parcel-001",
  "from_category": "grassland",
  "to_category": "cropland",
  "transition_date": "2025-06-01",
  "area_ha": 25.0,
  "driver": "agricultural_expansion",
  "tenant_id": "tenant-001"
}
```

---

### 15. GET /transitions/matrix

Retrieve the land-use transition matrix showing area changes between all category pairs for a given period and tenant.

---

### 16. POST /soc-assessments

Create a Soil Organic Carbon (SOC) assessment using IPCC stock change factors (land use, management, input factors).

---

### 18. POST /compliance/check

Check LULUCF calculation results against regulatory frameworks (GHG Protocol, IPCC 2006, CSRD ESRS E1, ISO 14064).

---

### 20. POST /uncertainty

Run Monte Carlo or analytical uncertainty analysis on LULUCF calculations.

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid land category, method, or input parameters |
| 404 | Not Found -- Calculation, parcel, or stock record not found |
| 500 | Internal Server Error |
