# GL-Taxonomy-APP -- EU Taxonomy Platform API Reference

**Source:** `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/*.py`
**Version:** 1.0 Alpha

---

## Overview

The GL-Taxonomy-APP provides a REST API for EU Taxonomy Regulation compliance. It implements the full 4-step alignment test: eligibility screening, substantial contribution (SC), do no significant harm (DNSH), and minimum safeguards. The API also covers Green Asset Ratio (GAR) reporting, KPI calculations, activity cataloging, portfolio analysis, regulatory tracking, data quality assessment, gap analysis, and multi-format reporting.

---

## Route Modules (16)

| Prefix | Tag | Module | Endpoints | Description |
|--------|-----|--------|-----------|-------------|
| `/api/v1/taxonomy/screening` | Eligibility Screening | `screening_routes.py` | 7 | Step 1: Eligibility screening by NACE code |
| `/api/v1/taxonomy/alignment` | Alignment | `alignment_routes.py` | 8 | Full 4-step alignment assessment |
| `/api/v1/taxonomy/sc` | Substantial Contribution | `sc_routes.py` | 8 | Step 2: SC criteria testing |
| `/api/v1/taxonomy/dnsh` | DNSH | `dnsh_routes.py` | 9 | Step 3: Do No Significant Harm criteria |
| `/api/v1/taxonomy/safeguards` | Minimum Safeguards | `safeguards_routes.py` | 7 | Step 4: OECD Guidelines, UN Guiding Principles |
| `/api/v1/taxonomy/gar` | Green Asset Ratio | `gar_routes.py` | 7 | GAR calculation and reporting |
| `/api/v1/taxonomy/kpi` | KPI | `kpi_routes.py` | 7 | Turnover, CapEx, OpEx KPI calculations |
| `/api/v1/taxonomy/activities` | Activities | `activity_routes.py` | 8 | EU Taxonomy activity catalog |
| `/api/v1/taxonomy/portfolios` | Portfolios | `portfolio_routes.py` | 8 | Portfolio-level analysis |
| `/api/v1/taxonomy/regulatory` | Regulatory | `regulatory_routes.py` | 5 | Delegated Act tracking |
| `/api/v1/taxonomy/data-quality` | Data Quality | `data_quality_routes.py` | 6 | Data quality scoring and improvement |
| `/api/v1/taxonomy/gap-analysis` | Gap Analysis | `gap_routes.py` | 7 | Alignment gap identification |
| `/api/v1/taxonomy/reporting` | Reporting | `reporting_routes.py` | 8 | Report generation and export |
| `/api/v1/taxonomy/dashboard` | Dashboard | `dashboard_routes.py` | 6 | Dashboard aggregations |
| `/api/v1/taxonomy/settings` | Settings | `settings_routes.py` | 8 | Application configuration |

---

## 4-Step Alignment Test

The EU Taxonomy alignment test evaluates economic activities in four sequential steps:

```
Step 1: Eligibility Screening
    |
    v
Step 2: Substantial Contribution (SC)
    |
    v
Step 3: Do No Significant Harm (DNSH)
    |
    v
Step 4: Minimum Safeguards
    |
    v
ALIGNED (if all four steps pass)
```

---

## Screening Endpoints (Step 1)

**Prefix:** `/api/v1/taxonomy/screening`

| Method | Path | Summary |
|--------|------|---------|
| POST | `/screen` | Screen single activity for eligibility |
| POST | `/batch` | Batch screen multiple NACE codes |
| GET | `/results` | List screening results |
| GET | `/results/{result_id}` | Get specific screening result |
| POST | `/de-minimis` | Apply de minimis threshold |
| GET | `/sector-breakdown` | Eligible vs. non-eligible by sector |
| DELETE | `/results/{result_id}` | Delete screening result |

### POST /api/v1/taxonomy/screening/screen

**Request Body:**

```json
{
  "org_id": "org_001",
  "activity_name": "Solar PV Installation",
  "nace_code": "D35.11",
  "turnover_eur": 5000000,
  "capex_eur": 1200000,
  "opex_eur": 300000
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `org_id` | string | Yes | Organization ID |
| `activity_name` | string | Yes | Economic activity name (1-500 chars) |
| `nace_code` | string | Yes | NACE code (e.g., D35.11) |
| `turnover_eur` | float | Yes | Annual turnover in EUR |
| `capex_eur` | float | No | Annual CapEx in EUR |
| `opex_eur` | float | No | Annual OpEx in EUR |
| `description` | string | No | Activity description (max 2000 chars) |

**Eligibility statuses:** `eligible`, `not_eligible`, `partially_eligible`, `pending`

---

## Alignment Endpoints (Full Test)

**Prefix:** `/api/v1/taxonomy/alignment`

| Method | Path | Summary |
|--------|------|---------|
| POST | `/assess` | Run full 4-step alignment assessment |
| POST | `/batch-assess` | Batch alignment assessment |
| POST | `/quick-check` | Quick alignment pre-check |
| GET | `/results` | List alignment results |
| GET | `/results/{result_id}` | Get specific alignment result |
| GET | `/results/{result_id}/details` | Detailed step-by-step breakdown |
| GET | `/summary` | Organization-level alignment summary |
| GET | `/trends` | Alignment trends over time |

---

## DNSH Endpoints (Step 3)

**Prefix:** `/api/v1/taxonomy/dnsh`

Tests against 6 environmental objectives:

| # | Environmental Objective |
|---|------------------------|
| 1 | Climate change mitigation |
| 2 | Climate change adaptation |
| 3 | Sustainable use and protection of water and marine resources |
| 4 | Transition to a circular economy |
| 5 | Pollution prevention and control |
| 6 | Protection and restoration of biodiversity and ecosystems |

| Method | Path | Summary |
|--------|------|---------|
| POST | `/assess` | Full DNSH assessment |
| POST | `/climate-mitigation` | Test against climate mitigation |
| POST | `/climate-adaptation` | Test against climate adaptation |
| POST | `/water` | Test against water objective |
| POST | `/circular-economy` | Test against circular economy |
| POST | `/pollution` | Test against pollution prevention |
| POST | `/biodiversity` | Test against biodiversity |
| GET | `/results` | List DNSH results |
| GET | `/matrix` | DNSH criteria matrix for activities |

---

## GAR Endpoints

**Prefix:** `/api/v1/taxonomy/gar`

Green Asset Ratio (GAR) calculation for financial institutions.

| Method | Path | Summary |
|--------|------|---------|
| POST | `/calculate` | Calculate GAR |
| POST | `/banking-book` | Banking book GAR calculation |
| POST | `/trading-book` | Trading book calculation |
| POST | `/fee-income` | Fee and commission income |
| GET | `/results` | List GAR results |
| GET | `/results/{id}` | Get GAR result details |
| GET | `/templates` | GAR reporting templates |
| POST | `/export` | Export GAR report |
| GET | `/trends` | GAR trends over reporting periods |
| POST | `/validate` | Validate GAR data |

---

## Source Files

- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/screening_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/alignment_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/sc_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/dnsh_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/safeguards_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/gar_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/kpi_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/activity_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/portfolio_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/regulatory_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/data_quality_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/gap_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/reporting_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/dashboard_routes.py`
- `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/settings_routes.py`
