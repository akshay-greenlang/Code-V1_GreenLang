# GL-CDP-APP -- CDP Disclosure Platform API Reference

**Source:** `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/*.py`
**Version:** 1.0 Beta

---

## Overview

The GL-CDP-APP provides a REST API for CDP (formerly Carbon Disclosure Project) Climate Change questionnaire management. It covers the full CDP disclosure lifecycle: questionnaire structure (13 modules, M0-M13, 200+ questions), response drafting and review, automated scoring prediction, gap analysis, benchmarking, supply chain engagement, transition planning, and multi-format reporting.

---

## Route Modules (10)

| Prefix | Tag | Module | Endpoints | Description |
|--------|-----|--------|-----------|-------------|
| `/api/v1/cdp/questionnaires` | Questionnaires | `questionnaire_routes.py` | 8 | Questionnaire instances, modules, question structure |
| `/api/v1/cdp/responses` | Responses | `response_routes.py` | 12 | Response drafting, review, approval, evidence |
| `/api/v1/cdp/scoring` | Scoring | `scoring_routes.py` | 8 | Score prediction, band analysis, improvement recs |
| `/api/v1/cdp/gap-analysis` | Gap Analysis | `gap_analysis_routes.py` | 7 | Gap identification, readiness assessment |
| `/api/v1/cdp/benchmarking` | Benchmarking | `benchmarking_routes.py` | 6 | Peer comparison, sector benchmarks |
| `/api/v1/cdp/supply-chain` | Supply Chain | `supply_chain_routes.py` | 8 | Supplier engagement, Scope 3 collaboration |
| `/api/v1/cdp/transition-plans` | Transition Plans | `transition_plan_routes.py` | 8 | 1.5C pathway planning, decarbonization roadmap |
| `/api/v1/cdp/reporting` | Reporting | `reporting_routes.py` | 7 | Report generation and export |
| `/api/v1/cdp/dashboard` | Dashboard | `dashboard_routes.py` | 6 | Disclosure progress, score trends |
| `/api/v1/cdp/settings` | Settings | `settings_routes.py` | 5 | Configuration and data management |

---

## CDP Modules (M0-M13)

The questionnaire structure follows the CDP Climate Change questionnaire format:

| Module | Name | Description |
|--------|------|-------------|
| M0 | Introduction | Organization profile, reporting boundary, base year |
| M1 | Governance | Board oversight, management responsibility |
| M2 | Policies and Commitments | Climate policies, deforestation-free commitments |
| M3 | Risks and Opportunities | Climate risk assessment |
| M4 | Strategy | Business strategy alignment, scenario analysis |
| M5 | Transition Plans | 1.5C pathway, decarbonization roadmap |
| M6 | Implementation | Emissions reduction initiatives |
| M7 | Environmental Performance - Climate | Scope 1/2/3 emissions |
| M8 | Environmental Performance - Forests | Deforestation (if applicable) |
| M9 | Environmental Performance - Water | Water security (if applicable) |
| M10 | Supply Chain | Supplier engagement, Scope 3 collaboration |
| M11 | Additional Metrics | Sector-specific, energy mix |
| M12 | Financial Services | Portfolio emissions (if FS sector) |
| M13 | Sign Off | Authorization, verification statement |

---

## Questionnaire Endpoints

| Method | Path | Summary |
|--------|------|---------|
| GET | `/` | List questionnaire instances |
| GET | `/{questionnaire_id}` | Get questionnaire details |
| POST | `/` | Create questionnaire instance |
| GET | `/{id}/modules` | List modules for questionnaire |
| GET | `/{id}/modules/{module_id}` | Get specific module |
| GET | `/{id}/modules/{module_id}/questions` | List questions in module |
| GET | `/{id}/progress` | Module completion progress |
| PUT | `/{id}/status` | Update questionnaire status |

### Questionnaire Statuses

`not_started` -> `in_progress` -> `in_review` -> `approved` -> `submitted`

### Question Types

`text`, `numeric`, `percentage`, `table`, `multi_select`, `single_select`, `yes_no`

### Scoring Levels

`disclosure`, `awareness`, `management`, `leadership`

---

## Response Endpoints

| Method | Path | Summary |
|--------|------|---------|
| GET | `/` | List responses |
| GET | `/{response_id}` | Get specific response |
| POST | `/` | Create draft response |
| PUT | `/{response_id}` | Update response |
| PATCH | `/{response_id}` | Partial update |
| GET | `/{response_id}/history` | Response edit history |
| POST | `/{response_id}/submit-review` | Submit for review |
| DELETE | `/{response_id}` | Delete draft response |
| POST | `/{response_id}/approve` | Approve response |
| POST | `/{response_id}/evidence` | Attach evidence document |
| POST | `/{response_id}/ai-draft` | AI-assisted response drafting |
| POST | `/{response_id}/validate` | Validate against CDP rules |

---

## Scoring Endpoints

| Method | Path | Summary |
|--------|------|---------|
| GET | `/predict` | Predict overall CDP score |
| GET | `/predict/{module_id}` | Module-level score prediction |
| GET | `/bands` | Score band definitions (D- to A) |
| POST | `/simulate` | Simulate score with hypothetical improvements |
| GET | `/improvement-roadmap` | Prioritized improvement recommendations |
| GET | `/peer-comparison` | Score comparison with peers |
| GET | `/historical-trend` | Year-over-year score trend |
| GET | `/category-breakdown` | Score by scoring category |

---

## Source Files

- `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/questionnaire_routes.py`
- `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/response_routes.py`
- `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/scoring_routes.py`
- `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/gap_analysis_routes.py`
- `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/benchmarking_routes.py`
- `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/supply_chain_routes.py`
- `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/transition_plan_routes.py`
- `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/reporting_routes.py`
- `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/dashboard_routes.py`
- `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/settings_routes.py`
