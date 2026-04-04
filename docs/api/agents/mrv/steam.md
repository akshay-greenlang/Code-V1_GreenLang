# Steam/Heat Purchase Agent API Reference (AGENT-MRV-011)

## Overview

The Steam/Heat Purchase Agent (GL-MRV-X-022) calculates GHG Protocol Scope 2 emissions from purchased steam, district heating, district cooling, and CHP/cogeneration thermal output. Implements the GHG Protocol Scope 2 Guidance (2015) with four calculation methodologies.

**API Prefix:** `/api/v1/steam-heat-purchase`
**Agent ID:** GL-MRV-X-022
**Status:** Production Ready

**Calculation Methodologies:**
1. **Direct Emission Factor** -- Composite kgCO2e/GJ factor applied to metered consumption
2. **Fuel-Based** -- Per-gas emissions (CO2, CH4, N2O) from fuel type, quantity, and boiler efficiency. 14 fuel types including biomass with separate biogenic CO2 reporting
3. **COP-Based** -- Cooling output converted to energy input via coefficient of performance (9 cooling technologies)
4. **CHP Allocation** -- Efficiency, energy, or exergy methods per GHG Protocol guidance

**Supported Regulatory Frameworks (7):**
GHG Protocol Scope 2 Guidance, IPCC 2006, ISO 14064-1:2018, CSRD/ESRS E1, US EPA GHGRP, UK DEFRA SECR, CDP Climate Change

**Emission Factor Data:**
- 14 fuel types with per-gas emission factors (IPCC 2006 Vol 2)
- 13 regional district heating factors (kgCO2e/GJ delivered heat)
- 9 cooling system technologies with COP ranges
- 5 CHP fuel types with electrical, thermal, and overall efficiencies

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/calculate/steam` | Calculate steam emissions | 201, 400, 500 |
| 2 | POST | `/calculate/heating` | Calculate district heating emissions | 201, 400, 500 |
| 3 | POST | `/calculate/cooling` | Calculate district cooling emissions | 201, 400, 500 |
| 4 | POST | `/calculate/chp` | Calculate CHP-allocated emissions | 201, 400, 500 |
| 5 | POST | `/calculate/batch` | Batch calculation | 201, 400, 500 |
| 6 | GET | `/factors/fuels` | List all fuel emission factors | 200 |
| 7 | GET | `/factors/fuels/{fuel_type}` | Get specific fuel emission factor | 200, 404 |
| 8 | GET | `/factors/heating/{region}` | Get district heating network factor | 200, 404 |
| 9 | GET | `/factors/cooling/{technology}` | Get cooling system COP | 200, 404 |
| 10 | GET | `/factors/chp-defaults` | Get CHP default parameters | 200 |
| 11 | POST | `/facilities` | Register a facility | 201, 400, 409 |
| 12 | GET | `/facilities/{facility_id}` | Get facility | 200, 404 |
| 13 | POST | `/suppliers` | Register steam/heat supplier | 201, 400, 409 |
| 14 | GET | `/suppliers/{supplier_id}` | Get supplier | 200, 404 |
| 15 | POST | `/uncertainty` | Run uncertainty analysis | 200, 400, 404, 500 |
| 16 | POST | `/compliance/check` | Run compliance check | 200, 400, 500 |
| 17 | GET | `/compliance/frameworks` | List available frameworks | 200 |
| 18 | POST | `/aggregate` | Aggregate calculation results | 200, 400, 500 |
| 19 | GET | `/calculations/{calc_id}` | Get calculation result | 200, 404 |
| 20 | GET | `/health` | Service health check | 200 |

---

## Endpoints

### 1. POST /calculate/steam

Calculate Scope 2 emissions from purchased steam consumption using the fuel-based method.

**Formula (fuel-based):**
```
effective_consumption = consumption_gj * (1 - condensate_return_pct/100)
fuel_input_gj = effective_consumption / boiler_efficiency
emissions_per_gas = fuel_input_gj * gas_ef_per_gj
co2e = sum(emissions_per_gas * gwp)
```

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "consumption_gj": 5000.0,
  "fuel_type": "natural_gas",
  "boiler_efficiency": 0.85,
  "supplier_id": "supplier-001",
  "steam_pressure": "medium",
  "steam_quality": "saturated",
  "condensate_return_pct": 15.0,
  "gwp_source": "AR6",
  "data_quality_tier": "tier_2",
  "reporting_period": "annual",
  "tenant_id": "tenant-001"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `facility_id` | string | Yes | Reference to consuming facility |
| `consumption_gj` | float | Yes | Steam consumption in GJ at building meter (>0) |
| `fuel_type` | string | No | Fuel type (14 supported, default from supplier) |
| `boiler_efficiency` | float | No | Boiler efficiency override 0-1 |
| `supplier_id` | string | No | Steam supplier reference |
| `steam_pressure` | string | No | low, medium, high, very_high |
| `steam_quality` | string | No | saturated, superheated, wet |
| `condensate_return_pct` | float | No | Condensate return 0-100 (default 0) |
| `gwp_source` | string | No | AR4, AR5, AR6, AR6_20YR (default AR6) |
| `data_quality_tier` | string | No | tier_1, tier_2, tier_3 (default tier_1) |
| `tenant_id` | string | No | Multi-tenancy identifier |

**Response (201 Created):**

```json
{
  "calc_id": "sh_abc123",
  "energy_type": "steam",
  "facility_id": "facility-001",
  "total_co2e_kg": 294.125000,
  "co2_kg": 292.500000,
  "ch4_kg": 0.005875,
  "n2o_kg": 0.000588,
  "biogenic_co2_kg": 0.0,
  "fuel_input_gj": 5000.0,
  "boiler_efficiency": 0.85,
  "gwp_source": "AR6",
  "provenance_hash": "sha256:q7r8s9t0...",
  "created_at": "2026-04-01T10:30:00Z"
}
```

---

### 2. POST /calculate/heating

Calculate Scope 2 emissions from district heating consumption.

**Formula:**
```
adjusted_consumption = consumption_gj * (1 + distribution_loss_pct)
co2e_kg = adjusted_consumption * ef_kgco2e_per_gj
```

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "consumption_gj": 2000.0,
  "region": "germany",
  "network_type": "municipal",
  "supplier_ef_kgco2e_per_gj": null,
  "distribution_loss_pct": null,
  "gwp_source": "AR6",
  "tenant_id": "tenant-001"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `facility_id` | string | Yes | Consuming facility reference |
| `consumption_gj` | float | Yes | District heating consumption in GJ (>0) |
| `region` | string | Yes | Region for factor lookup (13 regions) |
| `network_type` | string | No | municipal, industrial, campus, mixed |
| `supplier_ef_kgco2e_per_gj` | float | No | Supplier-specific EF override |
| `distribution_loss_pct` | float | No | Distribution loss fraction override 0-1 |

**Supported Regions:** denmark, sweden, finland, germany, poland, netherlands, france, uk, us, china, japan, south_korea, global_default

---

### 3. POST /calculate/cooling

Calculate Scope 2 emissions from district cooling consumption using COP-based method.

**Formula:**
```
energy_input_gj = cooling_output_gj / cop
co2e_kg = energy_input_gj * energy_source_ef_per_gj
```

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "cooling_output_gj": 1500.0,
  "technology": "centrifugal_chiller",
  "cop": null,
  "grid_ef_kgco2e_per_kwh": 0.45,
  "gwp_source": "AR6",
  "tenant_id": "tenant-001"
}
```

**Supported Technologies:** centrifugal_chiller, screw_chiller, reciprocating_chiller, absorption_single, absorption_double, absorption_triple, free_cooling, ice_storage, thermal_storage

---

### 4. POST /calculate/chp

Calculate CHP/cogeneration emission allocation between thermal and electrical outputs.

**Allocation Methods:**
- **Efficiency** (default): `heat_share = (Q_heat/eta_thermal) / ((Q_heat/eta_thermal) + (Q_elec/eta_elec))`
- **Energy**: `heat_share = Q_heat / (Q_heat + Q_elec)`
- **Exergy**: `carnot = 1 - (T_ambient_K/T_steam_K); heat_share = (Q_heat*carnot) / ((Q_heat*carnot) + Q_elec)`

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "total_fuel_gj": 10000.0,
  "fuel_type": "natural_gas",
  "heat_output_gj": 4000.0,
  "power_output_gj": 3500.0,
  "cooling_output_gj": 0.0,
  "method": "efficiency",
  "electrical_efficiency": 0.35,
  "thermal_efficiency": 0.45,
  "steam_temperature_c": 180.0,
  "ambient_temperature_c": 25.0,
  "gwp_source": "AR6",
  "tenant_id": "tenant-001"
}
```

---

### 5. POST /calculate/batch

Batch multiple calculations (steam, heating, cooling, or CHP) in a single request.

**Request Body:**

```json
{
  "batch_id": "batch-001",
  "tenant_id": "tenant-001",
  "requests": [
    { "energy_type": "steam", "facility_id": "f-1", "consumption_gj": 3000 },
    { "energy_type": "district_heating", "facility_id": "f-2", "consumption_gj": 1500, "region": "sweden" }
  ]
}
```

---

### 6. GET /factors/fuels

List all 14 fuel emission factors (CO2, CH4, N2O per GJ, default efficiency, biogenic flag). No authentication required.

---

### 7. GET /factors/fuels/{fuel_type}

Get emission factor for a specific fuel type. Returns per-gas factors and default boiler efficiency.

**Supported Fuel Types:** natural_gas, fuel_oil_2, fuel_oil_6, coal_bituminous, coal_subbituminous, coal_lignite, lpg, biomass_wood, biomass_biogas, municipal_waste, waste_heat, geothermal, solar_thermal, electric

---

### 8. GET /factors/heating/{region}

Get district heating emission factor (kgCO2e/GJ) and distribution loss percentage for a region.

---

### 9. GET /factors/cooling/{technology}

Get cooling system COP range (min, max, default) and primary energy source for a technology.

---

### 10. GET /factors/chp-defaults

Get default CHP efficiency parameters (electrical, thermal, overall) for all 5 supported CHP fuel types.

---

### 11. POST /facilities

Register a facility for thermal energy tracking.

**Request Body:**

```json
{
  "name": "Manufacturing Plant Alpha",
  "facility_type": "industrial",
  "country": "DE",
  "region": "germany",
  "latitude": 51.1657,
  "longitude": 10.4515,
  "steam_suppliers": ["supplier-001"],
  "heating_network": "dh-network-A",
  "cooling_system": "dc-system-B",
  "tenant_id": "tenant-001"
}
```

---

### 13. POST /suppliers

Register a steam/heat supplier with fuel mix, boiler efficiency, and composite emission factor.

**Request Body:**

```json
{
  "name": "City Steam Works",
  "fuel_mix": { "natural_gas": 0.8, "biomass_wood": 0.2 },
  "boiler_efficiency": 0.88,
  "supplier_ef_kgco2e_per_gj": 62.5,
  "country": "DE",
  "region": "germany",
  "verified": true,
  "data_quality_tier": "tier_2",
  "tenant_id": "tenant-001"
}
```

---

### 15. POST /uncertainty

Run Monte Carlo simulation or analytical error propagation on a calculation result.

**Request Body:**

```json
{
  "calc_id": "sh_abc123",
  "method": "monte_carlo",
  "iterations": 10000,
  "confidence_level": 95.0,
  "activity_data_uncertainty_pct": 5.0,
  "emission_factor_uncertainty_pct": 10.0,
  "efficiency_uncertainty_pct": 5.0,
  "seed": 42
}
```

---

### 16. POST /compliance/check

Evaluate calculation against regulatory frameworks.

**Request Body:**

```json
{
  "calc_id": "sh_abc123",
  "frameworks": ["ghg_protocol_scope2", "iso_14064", "csrd_e1"]
}
```

---

### 20. GET /health

Service health check. Returns status, agent ID, version, and uptime.

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid energy type, fuel type, region, or technology |
| 404 | Not Found -- Calculation, facility, supplier, fuel type, or region not found |
| 409 | Conflict -- Duplicate facility or supplier ID |
| 500 | Internal Server Error |
