# GL-SBTi-APP -- SBTi Target Platform API Reference

**Source:** `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/*.py`
**Version:** 1.0 Beta

---

## Overview

The GL-SBTi-APP provides a REST API for Science Based Targets initiative (SBTi) target management. It supports the full SBTi v2.1 criteria for near-term, long-term, and net-zero targets across all emission scopes. The API covers target-setting pathways, validation, sector-specific guidance, progress tracking, temperature alignment scoring, base year recalculation, financial institution targets, and submission review workflows.

---

## Route Modules (16)

| Prefix | Tag | Module | Endpoints | Description |
|--------|-----|--------|-----------|-------------|
| `/api/v1/sbti/targets` | Targets | `target_routes.py` | 11 | Target CRUD, status lifecycle, coverage validation |
| `/api/v1/sbti/pathways` | Pathways | `pathway_routes.py` | 9 | Decarbonization pathway modeling |
| `/api/v1/sbti/validation` | Validation | `validation_routes.py` | 7 | Target validation against SBTi criteria |
| `/api/v1/sbti/sectors` | Sectors | `sector_routes.py` | 6 | Sector-specific guidance and benchmarks |
| `/api/v1/sbti/progress` | Progress | `progress_routes.py` | 8 | Target progress tracking |
| `/api/v1/sbti/temperature` | Temperature | `temperature_routes.py` | 5 | Temperature alignment scoring |
| `/api/v1/sbti/recalculation` | Recalculation | `recalculation_routes.py` | 6 | Base year recalculation triggers |
| `/api/v1/sbti/framework` | Framework | `framework_routes.py` | 4 | SBTi framework criteria reference |
| `/api/v1/sbti/review` | Review | `review_routes.py` | 7 | Submission review workflow |
| `/api/v1/sbti/scope3` | Scope 3 | `scope3_routes.py` | 7 | Scope 3 coverage requirements (40% threshold) |
| `/api/v1/sbti/fi` | Financial Institutions | `fi_routes.py` | 13 | Financial institution sector targets (PCAF) |
| `/api/v1/sbti/flags` | Flags | `flag_routes.py` | 8 | Target compliance flags and alerts |
| `/api/v1/sbti/gap-analysis` | Gap Analysis | `gap_routes.py` | 7 | Readiness gap identification |
| `/api/v1/sbti/reporting` | Reporting | `reporting_routes.py` | 7 | Report generation |
| `/api/v1/sbti/dashboard` | Dashboard | `dashboard_routes.py` | 6 | Dashboard aggregations |
| `/api/v1/sbti/settings` | Settings | `settings_routes.py` | 13 | SBTi configuration |

---

## Target Endpoints

**Prefix:** `/api/v1/sbti/targets`

| Method | Path | Summary |
|--------|------|---------|
| GET | `/` | List all targets |
| GET | `/{target_id}` | Get target details |
| POST | `/` | Create new target |
| PUT | `/{target_id}` | Update target |
| DELETE | `/{target_id}` | Delete target |
| PUT | `/{target_id}/status` | Update target status |
| GET | `/{target_id}/coverage` | Get scope coverage analysis |
| POST | `/submission-form` | Generate SBTi submission form |
| GET | `/summary` | Target portfolio summary |
| POST | `/{target_id}/validate` | Validate target against SBTi criteria |

### Target Types

| Type | Description | Timeframe |
|------|-------------|-----------|
| `near_term` | 5-10 year absolute or intensity reduction | 2030 |
| `long_term` | Aligned with net-zero by 2050 | 2050 |
| `net_zero` | Commitment to net-zero across all material scopes | 2050 |

### Target Scopes

`scope_1`, `scope_2`, `scope_1_2`, `scope_3`, `all_scopes`

### Target Methods

`absolute`, `intensity_physical`, `intensity_economic`, `supplier_engagement`

### Ambition Levels

`1.5C`, `well_below_2C`

### Target Lifecycle Status

```
draft -> pending_validation -> submitted -> validated -> approved -> active -> expired/withdrawn
```

### SBTi v2.1 Criteria Referenced

- C1-C5: Target boundary and timeframe
- C6-C8: Level of ambition (1.5C, well-below 2C)
- C13-C15: Scope 3 requirements (40% threshold)
- C20-C23: Net-zero target requirements

---

## Pathway Endpoints (9)

**Prefix:** `/api/v1/sbti/pathways`

8 POST endpoints for different pathway modeling methods plus 1 GET for pathway comparison.

---

## Financial Institution Endpoints (13)

**Prefix:** `/api/v1/sbti/fi`

Manages financial institution sector targets using PCAF methodology for portfolio emissions.

---

## Source Files

- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/target_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/pathway_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/validation_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/sector_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/progress_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/temperature_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/recalculation_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/framework_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/review_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/scope3_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/fi_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/flag_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/gap_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/reporting_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/dashboard_routes.py`
- `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/settings_routes.py`
