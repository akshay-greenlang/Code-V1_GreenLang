# Employee Commuting Agent API Reference (AGENT-MRV-020)

## Overview

The Employee Commuting Agent (GL-MRV-S3-007) calculates GHG Protocol Scope 3 Category 7 emissions from employee commuting. Supports 14 commute modes, 3 calculation methods (employee-specific survey, average-data census, spend-based EEIO), telework/WFH emissions, multi-modal trip calculations, survey processing with statistical extrapolation, and mode share analysis.

**API Prefix:** `/api/v1/employee-commuting`
**Agent ID:** GL-MRV-S3-007
**Status:** Production Ready

**Commute Modes (14):**
SOV, carpool, vanpool, bus, metro, light_rail, commuter_rail, ferry, motorcycle, e_bike, e_scooter, cycling, walking, telework

**Calculation Methods:**
- **Employee-Specific (Survey)** -- Individual employee commute data from surveys
- **Average-Data (Census)** -- National/regional average distances and mode shares (10 countries + global)
- **Spend-Based (EEIO)** -- NAICS-based EEIO factors with CPI deflation and currency conversion

**Regulatory Frameworks (7):**
GHG Protocol, ISO 14064, CSRD ESRS E1, CDP, SBTi, SB 253, GRI 305

**Pipeline Stages (10):**
validate, classify, normalize, resolve_efs, calculate_commute, calculate_telework, extrapolate, compliance, aggregate, seal

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/calculate` | Full pipeline calculation | 201, 400, 500 |
| 2 | POST | `/calculate/batch` | Batch calculation (up to 50K employees) | 201, 400, 500 |
| 3 | POST | `/calculate/commute` | Single commute mode calculation | 201, 400, 500 |
| 4 | POST | `/calculate/telework` | Telework/WFH emissions | 201, 400, 500 |
| 5 | POST | `/calculate/survey` | Process employee survey | 201, 400, 500 |
| 6 | POST | `/calculate/average-data` | Average-data method | 201, 400, 500 |
| 7 | POST | `/calculate/spend` | Spend-based method | 201, 400, 500 |
| 8 | POST | `/calculate/multi-modal` | Multi-modal trip calculation | 201, 400, 500 |
| 9 | GET | `/calculations/{calculation_id}` | Get calculation detail | 200, 404, 500 |
| 10 | GET | `/calculations` | List calculations | 200, 500 |
| 11 | DELETE | `/calculations/{calculation_id}` | Delete calculation | 200, 404, 500 |
| 12 | GET | `/emission-factors` | List emission factors | 200, 500 |
| 13 | GET | `/emission-factors/{mode}` | Get factors by commute mode | 200, 400, 500 |
| 14 | GET | `/commute-modes` | List all 14 commute modes | 200, 500 |
| 15 | GET | `/working-days/{region}` | Get working days by region | 200, 400, 500 |
| 16 | GET | `/commute-averages` | Get average commute distances | 200, 500 |
| 17 | GET | `/grid-factors/{country}` | Get grid EF for telework | 200, 400, 500 |
| 18 | POST | `/compliance/check` | Multi-framework compliance check | 201, 400, 500 |
| 19 | POST | `/uncertainty/analyze` | Uncertainty analysis | 201, 400, 500 |
| 20 | GET | `/aggregations/{period}` | Aggregated emissions | 200, 400, 500 |
| 21 | POST | `/mode-share/analyze` | Mode share analysis | 201, 400, 500 |
| 22 | GET | `/provenance/{calculation_id}` | Provenance chain | 200, 404, 500 |
| -- | GET | `/health` | Service health check | 200 |

---

## Endpoints

### 1. POST /calculate

Calculate employee commuting emissions through the full 10-stage pipeline.

**Request Body:**

```json
{
  "mode": "sov",
  "commute_data": {
    "distance_km": 25.0,
    "vehicle_type": "car_medium_petrol",
    "frequency": 5
  },
  "employee_id": "emp-001",
  "tenant_id": "tenant-001",
  "telework_data": {
    "days_per_week": 2,
    "daily_kwh": 4.0,
    "region": "US"
  },
  "reporting_period": "2025"
}
```

**Response (201 Created):**

```json
{
  "success": true,
  "calculation_id": "ec_abc123",
  "mode": "sov",
  "method": "employee_specific",
  "total_co2e_kg": 2850.5,
  "commute_co2e_kg": 2650.0,
  "telework_co2e_kg": 120.5,
  "wtt_co2e_kg": 80.0,
  "dqi_score": 4.2,
  "provenance_hash": "sha256:o1p2q3r4...",
  "detail": { "...": "..." },
  "calculated_at": "2026-04-01T10:30:00Z"
}
```

---

### 2. POST /calculate/batch

Batch calculate for up to 50,000 employees with parallel execution and workforce extrapolation.

**Request Body:**

```json
{
  "employees": [
    { "mode": "sov", "distance_km": 25.0, "vehicle_type": "car_medium_petrol" },
    { "mode": "bus", "distance_km": 15.0 },
    { "mode": "cycling", "distance_km": 5.0 }
  ],
  "reporting_period": "2025",
  "total_employees": 5000,
  "extrapolate": true
}
```

**Response (201 Created):**

```json
{
  "success": true,
  "batch_id": "batch_abc123",
  "total_employees": 5000,
  "successful": 3,
  "failed": 0,
  "total_co2e_kg": 5200.0,
  "extrapolated_co2e_kg": 8666666.0,
  "results": [ { "...": "..." } ],
  "errors": [],
  "reporting_period": "2025"
}
```

---

### 3. POST /calculate/commute

Calculate emissions for a single commute mode without the full pipeline.

**Request Body:**

```json
{
  "mode": "bus",
  "distance_km": 15.0,
  "frequency": 5,
  "working_days": 230,
  "round_trip": false
}
```

**Response (201 Created):**

```json
{
  "success": true,
  "calculation_id": "ec_def456",
  "mode": "bus",
  "distance_km": 15.0,
  "annual_distance_km": 34500.0,
  "co2e_kg": 2760.0,
  "wtt_co2e_kg": 276.0,
  "total_co2e_kg": 3036.0,
  "ef_used": 0.08,
  "ef_source": "DEFRA",
  "provenance_hash": "sha256:s5t6u7v8...",
  "calculated_at": "2026-04-01T10:30:00Z"
}
```

---

### 4. POST /calculate/telework

Calculate home office energy emissions for telework/remote work.

**Request Body:**

```json
{
  "frequency": "hybrid_3",
  "region": "US",
  "daily_kwh": 4.0,
  "seasonal_adjustment": "full_seasonal",
  "egrid_subregion": "CAMX",
  "equipment_lifecycle": true
}
```

**Response (201 Created):**

```json
{
  "success": true,
  "calculation_id": "ec_ghi789",
  "frequency": "hybrid_3",
  "telework_days_per_year": 138,
  "daily_kwh": 4.0,
  "annual_kwh": 552.0,
  "grid_ef_kgco2e_per_kwh": 0.21,
  "telework_co2e_kg": 115.92,
  "seasonal_adjustment_applied": true,
  "avoided_commute_co2e_kg": 1800.0,
  "equipment_co2e_kg": 25.0,
  "region": "US",
  "provenance_hash": "sha256:w9x0y1z2...",
  "calculated_at": "2026-04-01T10:30:00Z"
}
```

**Telework Frequencies:** full_remote, hybrid_4, hybrid_3, hybrid_2, hybrid_1, office_full

---

### 5. POST /calculate/survey

Process employee commute survey data with statistical extrapolation.

**Request Body:**

```json
{
  "survey_method": "random_sample",
  "responses": [
    { "mode": "sov", "distance_km": 30.0, "vehicle_type": "car_medium_petrol", "frequency": 5 },
    { "mode": "metro", "distance_km": 12.0, "frequency": 5 },
    { "mode": "cycling", "distance_km": 5.0, "frequency": 4 }
  ],
  "total_employees": 500,
  "response_rate": 0.65,
  "confidence_level": 0.95
}
```

**Response (201 Created):**

```json
{
  "success": true,
  "survey_id": "survey_abc123",
  "survey_method": "random_sample",
  "total_employees": 500,
  "respondents": 3,
  "response_rate": 0.65,
  "sample_co2e_kg": 5200.0,
  "extrapolated_co2e_kg": 866666.0,
  "extrapolation_factor": 166.67,
  "per_employee_avg_co2e_kg": 1733.33,
  "ci_lower_kg": 750000.0,
  "ci_upper_kg": 983332.0,
  "dqi_score": 3.5,
  "mode_share": { "sov": 0.33, "metro": 0.33, "cycling": 0.33 },
  "provenance_hash": "sha256:a3b4c5d6..."
}
```

---

### 6. POST /calculate/average-data

Calculate using national average commute distances and mode shares scaled by headcount.

**Request Body:**

```json
{
  "total_employees": 1000,
  "country_code": "US",
  "mode_share": null,
  "average_distance_km": null,
  "working_days": null
}
```

---

### 7. POST /calculate/spend

Calculate using EEIO factors with CPI deflation and currency conversion.

**Request Body:**

```json
{
  "naics_code": "485210",
  "amount": 50000.0,
  "currency": "USD",
  "reporting_year": 2025,
  "spend_category": "transit_subsidy"
}
```

**NAICS Codes:** 485000, 485110, 485210, 487110, 488490, 532100, 811100, 447000

---

### 8. POST /calculate/multi-modal

Calculate emissions for a commute with up to 5 transport segments.

**Request Body:**

```json
{
  "legs": [
    { "mode": "sov", "distance_km": 5.0, "vehicle_type": "car_medium_petrol" },
    { "mode": "commuter_rail", "distance_km": 25.0 },
    { "mode": "walking", "distance_km": 0.5 }
  ],
  "frequency": 5,
  "working_days": 230
}
```

---

### 14. GET /commute-modes

List all 14 supported commute modes with metadata (category, vehicle subtypes, EF sources, default occupancy).

---

### 15. GET /working-days/{region}

Get working days for a region (calendar weekdays, public holidays, PTO, sick days, net working days).

**Supported Regions:** US, GB, DE, FR, JP, CA, AU, IN, CN, BR, KR, GLOBAL

---

### 16. GET /commute-averages

Get average commute distances by country from census data (10 countries + global).

---

### 17. GET /grid-factors/{country}

Get grid emission factors for telework calculations. Returns kgCO2e/kWh from IEA 2024 (19 countries). US includes 26 eGRID subregional factors.

---

### 18. POST /compliance/check

Check results against 7 regulatory frameworks. Validates telework disclosure, mode share reporting, double-counting prevention (10 rules), survey methodology documentation, and materiality thresholds.

---

### 19. POST /uncertainty/analyze

Monte Carlo, analytical, or IPCC Tier 2 uncertainty analysis. Employee commuting uncertainty typically +/-10-60% depending on method.

---

### 21. POST /mode-share/analyze

Analyze workforce commute mode distribution with benchmark comparison and ranked intervention opportunities (transit subsidies, EV incentives, cycle-to-work schemes).

---

### GET /health

Service health check with per-engine status (7 engines: database, personal vehicle calculator, public transit calculator, active transport calculator, telework calculator, compliance checker, pipeline).

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid mode, method, region, or parameters |
| 404 | Not Found -- Calculation or provenance not found |
| 500 | Internal Server Error |
