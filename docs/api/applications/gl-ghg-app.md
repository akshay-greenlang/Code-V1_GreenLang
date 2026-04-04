# GL-GHG-APP -- GHG Corporate Platform API Reference

**Source:** `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/*.py`
**Version:** 1.0

---

## Overview

The GL-GHG-APP provides a REST API for corporate GHG (Greenhouse Gas) inventory management following the GHG Protocol Corporate Standard. It covers Scope 1, Scope 2, and Scope 3 emissions data submission, inventory lifecycle management, reduction target tracking, third-party verification, and reporting.

---

## Route Modules (8)

| Prefix | Tag | Module | Endpoints | Description |
|--------|-----|--------|-----------|-------------|
| `/api/v1/scope1` | Scope 1 Emissions | `scope1_routes.py` | 6 | Direct emissions (8 source categories, Kyoto basket gases) |
| `/api/v1/scope2` | Scope 2 Emissions | `scope2_routes.py` | 7 | Indirect energy emissions (location-based, market-based, dual reporting) |
| `/api/v1/scope3` | Scope 3 Emissions | `scope3_routes.py` | 8 | Value chain emissions (15 categories) |
| `/api/v1/inventory` | Inventory | `inventory_routes.py` | 10 | GHG inventory lifecycle management |
| `/api/v1/targets` | Targets | `target_routes.py` | 7 | Emission reduction targets |
| `/api/v1/verification` | Verification | `verification_routes.py` | 7 | Third-party verification workflow |
| `/api/v1/reporting` | Reporting | `reporting_routes.py` | 7 | Report generation and export |
| `/api/v1/ghg/dashboard` | Dashboard | `dashboard_routes.py` | 5 | Dashboard aggregations and trends |
| `/api/v1/ghg/settings` | Settings | `settings_routes.py` | 3 | Application configuration |

---

## Scope 1 Endpoints

**Prefix:** `/api/v1/scope1`

Manages direct GHG emissions per GHG Protocol Chapter 4. Covers 8 source categories and 7 Kyoto basket gases (CO2, CH4, N2O, HFC, PFC, SF6, NF3).

| Method | Path | Summary |
|--------|------|---------|
| POST | `/data` | Submit Scope 1 activity data |
| GET | `/data` | List Scope 1 data submissions |
| GET | `/data/{submission_id}` | Get specific submission |
| GET | `/summary` | Scope 1 emission summary |
| GET | `/by-category` | Breakdown by source category |
| POST | `/calculate` | Calculate Scope 1 emissions |

### POST /api/v1/scope1/data

**Request Body:**

```json
{
  "category": "stationary_combustion",
  "facility_id": "ent_abc123",
  "facility_name": "East Coast Plant",
  "quantity": 150000,
  "unit": "therms",
  "fuel_type": "natural_gas",
  "emission_factor": 5.302,
  "emission_factor_source": "EPA",
  "calculation_tier": "tier_1",
  "period_start": "2025-01-01",
  "period_end": "2025-12-31",
  "notes": "Annual natural gas consumption from utility bills"
}
```

**Source categories:** `stationary_combustion`, `mobile_combustion`, `process_emissions`, `fugitive_emissions`, `refrigerants`, `land_use`, `waste_treatment`, `agricultural`

**Calculation tiers (IPCC):** `tier_1`, `tier_2`, `tier_3`

---

## Scope 2 Endpoints

**Prefix:** `/api/v1/scope2`

| Method | Path | Summary |
|--------|------|---------|
| POST | `/data` | Submit Scope 2 energy data |
| GET | `/data` | List Scope 2 data submissions |
| GET | `/data/{submission_id}` | Get specific submission |
| GET | `/summary` | Scope 2 emission summary |
| GET | `/location-based` | Location-based calculation results |
| GET | `/market-based` | Market-based calculation results |
| POST | `/dual-report` | Generate dual-reporting comparison |

---

## Scope 3 Endpoints

**Prefix:** `/api/v1/scope3`

| Method | Path | Summary |
|--------|------|---------|
| POST | `/data` | Submit Scope 3 activity data |
| GET | `/data` | List Scope 3 submissions |
| GET | `/data/{submission_id}` | Get specific submission |
| GET | `/summary` | Scope 3 emission summary |
| GET | `/by-category` | Breakdown by category (1-15) |
| GET | `/materiality` | Category materiality ranking |
| GET | `/hotspots` | Emission hotspot analysis |
| POST | `/calculate` | Calculate Scope 3 emissions |

---

## Inventory Endpoints

**Prefix:** `/api/v1/inventory`

| Method | Path | Summary |
|--------|------|---------|
| POST | `/` | Create new inventory |
| GET | `/` | List inventories |
| POST | `/import` | Import inventory data |
| PUT | `/{inventory_id}` | Update inventory |
| POST | `/{inventory_id}/consolidate` | Run consolidation |
| GET | `/{inventory_id}/summary` | Inventory summary |
| POST | `/{inventory_id}/lock` | Lock inventory for verification |
| GET | `/{inventory_id}/audit-trail` | Audit trail |
| POST | `/{inventory_id}/base-year-recalculate` | Recalculate base year |

---

## Target Endpoints

**Prefix:** `/api/v1/targets`

| Method | Path | Summary |
|--------|------|---------|
| POST | `/` | Create reduction target |
| GET | `/` | List targets |
| GET | `/{target_id}` | Get target details |
| GET | `/{target_id}/progress` | Progress tracking |
| GET | `/{target_id}/trajectory` | Emission trajectory |
| DELETE | `/{target_id}` | Delete target |

---

## Verification Endpoints

**Prefix:** `/api/v1/verification`

| Method | Path | Summary |
|--------|------|---------|
| POST | `/requests` | Create verification request |
| GET | `/requests` | List verification requests |
| POST | `/requests/{id}/submit` | Submit for verification |
| POST | `/requests/{id}/findings` | Add verifier findings |
| POST | `/requests/{id}/complete` | Complete verification |
| POST | `/requests/{id}/evidence` | Upload evidence document |
| GET | `/requests/{id}/status` | Verification status |

---

## Reporting Endpoints

**Prefix:** `/api/v1/reporting`

| Method | Path | Summary |
|--------|------|---------|
| POST | `/generate` | Generate GHG report |
| GET | `/reports` | List generated reports |
| GET | `/reports/{id}` | Get report details |
| GET | `/reports/{id}/download` | Download report file |
| GET | `/reports/{id}/metrics` | Report metrics summary |
| GET | `/templates` | Available report templates |
| POST | `/export` | Export data in custom format |

---

## Source Files

- `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/scope1_routes.py`
- `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/scope2_routes.py`
- `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/scope3_routes.py`
- `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/inventory_routes.py`
- `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/target_routes.py`
- `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/verification_routes.py`
- `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/reporting_routes.py`
- `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/dashboard_routes.py`
- `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/settings_routes.py`
