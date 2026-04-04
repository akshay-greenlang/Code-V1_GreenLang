# GL-TCFD-APP -- TCFD Disclosure Platform API Reference

**Source:** `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/*.py`
**Version:** 1.0 Beta

---

## Overview

The GL-TCFD-APP provides a REST API for TCFD (Task Force on Climate-related Financial Disclosures) and ISSB/IFRS S2 disclosure management. It covers all four TCFD pillars -- Governance, Strategy, Risk Management, and Metrics and Targets -- plus scenario analysis, physical and transition risk assessment, opportunity identification, and financial impact quantification.

---

## Route Modules (15)

| Prefix | Tag | Module | Description |
|--------|-----|--------|-------------|
| `/api/v1/tcfd/governance` | Governance | `governance_routes.py` | Board oversight, management roles, maturity scoring |
| `/api/v1/tcfd/strategy` | Strategy | `strategy_routes.py` | Strategic planning, business impact analysis |
| `/api/v1/tcfd/risk-management` | Risk Management | `risk_management_routes.py` | Risk identification, assessment, mitigation |
| `/api/v1/tcfd/metrics` | Metrics | `metrics_routes.py` | Climate metrics and targets tracking |
| `/api/v1/tcfd/scenarios` | Scenarios | `scenario_routes.py` | Scenario analysis (1.5C, 2C, 4C pathways) |
| `/api/v1/tcfd/physical-risk` | Physical Risk | `physical_risk_routes.py` | Physical risk assessment (acute and chronic) |
| `/api/v1/tcfd/transition-risk` | Transition Risk | `transition_risk_routes.py` | Transition risk assessment (policy, technology, market, reputation) |
| `/api/v1/tcfd/opportunities` | Opportunities | `opportunity_routes.py` | Climate-related opportunity identification |
| `/api/v1/tcfd/financial` | Financial Impact | `financial_routes.py` | Financial impact quantification |
| `/api/v1/tcfd/disclosures` | Disclosures | `disclosure_routes.py` | Disclosure document management |
| `/api/v1/tcfd/issb` | ISSB/IFRS S2 | `issb_routes.py` | IFRS S2 alignment mapping |
| `/api/v1/tcfd/gap-analysis` | Gap Analysis | `gap_routes.py` | TCFD disclosure gap identification |
| `/api/v1/tcfd/dashboard` | Dashboard | `dashboard_routes.py` | Disclosure progress dashboards |
| `/api/v1/tcfd/settings` | Settings | `settings_routes.py` | Application configuration |

---

## Governance Endpoints

**Prefix:** `/api/v1/tcfd/governance`

TCFD Pillar 1 -- Governance. Manages board and management-level governance assessments, role assignments, maturity scoring, board climate competency evaluation, and governance disclosure text generation.

TCFD Recommended Disclosures:
- (a) Board oversight of climate-related risks and opportunities
- (b) Management's role in assessing and managing climate-related risks and opportunities

ISSB/IFRS S2 references: paragraphs 26-27.

**Governance Role Types:** `board_chair`, `board_member`, `committee_chair`, `ceo`, `cfo`, `cso`, `cro`, `sustainability_director`, `risk_manager`, `other`

**Oversight Frequency:** `quarterly`, `semi_annually`, `annually`, `ad_hoc`

**Maturity Levels:** `level_1_initial` through `level_5_leading`

### Example: Create Governance Assessment

**Request Body:**

```json
{
  "reporting_year": 2025,
  "board_oversight": true,
  "oversight_frequency": "quarterly",
  "dedicated_committee": true,
  "committee_name": "Sustainability Committee",
  "management_accountability": true
}
```

---

## Dashboard Endpoints (8)

**Prefix:** `/api/v1/tcfd/dashboard`

| Method | Path | Summary |
|--------|------|---------|
| GET | `/summary` | Overall TCFD disclosure summary |
| GET | `/pillar-progress` | Progress by TCFD pillar |
| GET | `/risk-overview` | Risk landscape overview |
| GET | `/metrics-snapshot` | Key climate metrics snapshot |
| GET | `/scenario-results` | Scenario analysis results summary |
| GET | `/disclosure-completeness` | Disclosure completeness by recommendation |
| GET | `/timeline` | Disclosure timeline |
| GET | `/issb-alignment` | ISSB alignment status |

---

## Source Files

- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/governance_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/strategy_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/risk_management_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/metrics_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/scenario_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/physical_risk_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/transition_risk_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/opportunity_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/financial_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/disclosure_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/issb_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/gap_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/dashboard_routes.py`
- `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/settings_routes.py`
