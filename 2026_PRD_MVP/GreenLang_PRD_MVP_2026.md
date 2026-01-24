# GreenLang MVP Product Requirements Document (2026)

## CBAM Compliance Essentials Pack (2026)

**Document Version:** 2.0.0
**Status:** Updated per CTO Review
**Last Updated:** January 2026
**Document Owner:** GreenLang Product Team

---

## Table of Contents

1. [Document Control](#1-document-control)
2. [Executive Summary](#2-executive-summary)
3. [Background & Strategic Context](#3-background--strategic-context)
4. [Problem Statement](#4-problem-statement)
5. [Goals, Non-Goals & Success Criteria](#5-goals-non-goals--success-criteria)
6. [Target Users & Personas](#6-target-users--personas)
7. [MVP Scope Definition](#7-mvp-scope-definition)
8. [Functional Requirements](#8-functional-requirements)
9. [Non-Functional Requirements](#9-non-functional-requirements)
10. [Data Model & Artifacts](#10-data-model--artifacts)
11. [Agent Architecture & Pipeline](#11-agent-architecture--pipeline)
12. [Policy Engine & Compliance Rules](#12-policy-engine--compliance-rules)
13. [User Journeys](#13-user-journeys)
14. [Input/Output Specifications](#14-inputoutput-specifications)
15. [Error Handling & Validation](#15-error-handling--validation)
16. [Security & Privacy](#16-security--privacy)
17. [Testing Strategy](#17-testing-strategy)
18. [Deployment & Packaging](#18-deployment--packaging)
19. [Observability & Debugging](#19-observability--debugging)
20. [Regulatory Accuracy & Compliance](#20-regulatory-accuracy--compliance)
21. [Success Metrics & KPIs](#21-success-metrics--kpis)
22. [Risk Analysis & Mitigations](#22-risk-analysis--mitigations)
23. [Delivery Milestones](#23-delivery-milestones)
24. [Future Considerations (Post-MVP)](#24-future-considerations-post-mvp)
25. [Appendices](#appendices)

---

## 1. Document Control

| Field | Value |
|-------|-------|
| **Product** | GreenLang (Climate Operating System / SDK) |
| **MVP Name** | GreenLang CBAM Compliance Essentials Pack (2026) |
| **Primary Module** | GL-CBAM-APP (productized as a Pack) |
| **Repository** | Code-V1_GreenLang |
| **Target Release** | Q1 2026 |
| **Business Model** | Open-source + Paid Support |

### Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | January 2026 | GreenLang Team | Initial PRD based on stakeholder interviews |
| 2.0.0 | January 2026 | GreenLang Team | Updated per CTO review - 2026 operational compliance, policy guardrails |

### Key Changes in v2.0.0

| Change Area | v1.0 | v2.0 | Rationale |
|-------------|------|------|-----------|
| Regulatory scope | Transitional only | Transitional closeout + 2026 operational readiness | CBAM fully operational from 1 Jan 2026 |
| Default factors | Defaults-first (blanket) | Policy-guarded defaults (period-aware + 20% cap) | Default values constrained after Q2 2024 |
| Product categories | Steel & Aluminum together | Iron & Steel first, Aluminum fast-follow | Iron & Steel = 98% of CBAM volumes |
| Error handling | Fail-fast only | Fail-fast + collect-all mode | Better UX for Excel-first users |
| Interface | CLI only | CLI + Setup Wizard + Web UI | Reduce friction for SMB users |
| Authorization | Not addressed | Authorization readiness checks | Critical for 2026 compliance |

---

## 2. Executive Summary

### 2.1 One-Paragraph Summary

The GreenLang CBAM Compliance Essentials Pack (2026) is a **deterministic, audit-ready CBAM compliance tool** that supports EU importers through both the **transitional closeout** (Q4 2025 amendments) and **2026 operational readiness**. It produces:

1. **CBAM Quarterly Report XML** - EU Transitional Registry format with XSD validation
2. **2026 Emissions Ledger** - Structured emissions data for certificate planning
3. **Authorization Readiness Dossier** - Structured package for registry workflow
4. **Complete Audit Bundle** - Evidence, lineage, assumptions, run manifest

This demonstrates GreenLang's core thesis: **deterministic workflows + audit-ready evidence packaging + reusable agents**.

### 2.2 Why This MVP

- **Timely:** CBAM operational from January 1, 2026 - importers need compliance tools NOW
- **Clear Definition of Done:** Valid XML + audit bundle + authorization readiness
- **Demonstrates Platform Primitives:** Determinism, pinned versions, lineage, evidence packaging
- **Generalizable:** Same primitives apply to MRV, CSRD, Scope 3, facility/fleet packs
- **Market Gap:** SMB importers underserved by expensive consulting/enterprise solutions

### 2.3 Key Decisions Summary

| Decision Area | Choice | Rationale |
|---------------|--------|-----------|
| **Product Name** | CBAM Compliance Essentials Pack (2026) | Positions for current regulatory reality |
| **Target Customer** | Small/Mid EU Importers (10-100 lines/quarter) | Largest underserved segment |
| **Business Model** | Open-source + paid support | Builds trust, enables enterprise upsell |
| **Regulatory Scope** | Transitional closeout + 2026 operational readiness | Aligned with current CBAM phase |
| **Product Categories** | Iron & Steel first (72xx, 73xx), Aluminum fast-follow (76xx) | 98% of CBAM volumes |
| **Default Values** | Policy-guarded (period-aware, 20% cap enforced) | EU method required from 2025 |
| **Interface** | CLI + Web UI + Setup Wizard | Reduce friction for Excel-first users |
| **Authorization** | Readiness checks + structured dossier | Critical for 2026 compliance |
| **Error Handling** | Fail-fast (default) + collect-all mode | Better UX options |
| **Deployment** | CLI + Docker + Web interface | Flexible deployment options |
| **Security** | Local-first, no network | Data never leaves user's machine |

---

## 3. Background & Strategic Context

### 3.1 Regulatory Context: CBAM Timeline

The **Carbon Border Adjustment Mechanism (CBAM)** has evolved through distinct phases:

| Phase | Period | Requirements |
|-------|--------|--------------|
| **Transitional Period** | Oct 2023 - Dec 2025 | Reporting only, quarterly submissions |
| **Operational Phase** | Jan 2026+ | Authorizations required, certificate purchases |

**Critical 2026 Changes:**
- **Authorization Requirement:** Importers above 50 tonnes/year must have authorization or application reference by March 31, 2026
- **EU Method Only:** From January 1, 2025, only EU calculation method accepted
- **Default Value Restrictions:** 20% cap on estimations for complex goods (from Q3 2024)

### 3.2 Default Value Rules (Policy-Critical)

| Reporting Period | Default Value Rules |
|-----------------|---------------------|
| Q4 2023 - Q2 2024 | Defaults allowed without quantitative limit |
| Q3 2024 - Q4 2025 | Estimations capped at 20% of total embedded emissions for complex goods |
| 2025+ | Only EU method accepted |
| 2026+ | Operational regime - stricter verification |

**MVP Must Enforce These Rules** - generating "valid XML" that is methodologically non-compliant is a regulatory risk.

### 3.3 2026 Volumetric Reality

Based on EU Commission operational statistics (early 2026):
- **Iron & Steel:** 98% of CBAM-covered import volumes
- **Aluminum:** ~2% of volumes

**Decision:** Start with Iron & Steel (72xx, 73xx), add Aluminum as fast-follow.

### 3.4 GreenLang Platform Context

| Component | Status | Relevance to MVP |
|-----------|--------|------------------|
| Calculation Engine | Stable | Core emissions calculations |
| Emission Factor Library | 1,000+ factors | CBAM default values + sources |
| Agent Framework | Stable | Pipeline orchestration |
| GL-CBAM-APP | 95% mature | Existing CBAM logic + XML export |
| Packs Concept | Documented | Reusable agent configurations |

---

## 4. Problem Statement

### 4.1 User Problems (2026 Context)

| Problem | 2026 Impact | Urgency |
|---------|-------------|---------|
| **Authorization Confusion** | Missing authorization = customs delays/penalties | CRITICAL |
| **Methodology Compliance** | Using wrong method = report rejection | HIGH |
| **Default Value Misuse** | Exceeding 20% cap = non-compliance | HIGH |
| **Data Fragmentation** | Manual spreadsheet consolidation | MEDIUM |
| **Audit Defensibility** | No evidence trail | MEDIUM |

### 4.2 Why Now

- CBAM is **fully operational** from January 1, 2026
- Importers must have **authorization/application by March 31, 2026**
- Q4 2025 reports due by **January 31, 2026**
- Market is **scrambling** for compliance tools

---

## 5. Goals, Non-Goals & Success Criteria

### 5.1 Goals (MVP Must Accomplish)

| ID | Goal | Description | Acceptance Criteria |
|----|------|-------------|-------------------|
| **G1** | Two operational modes | Support transitional closeout AND 2026 readiness | Both modes functional |
| **G2** | Policy-compliant outputs | Enforce default value rules by period | Policy engine validates all outputs |
| **G3** | Authorization readiness | Generate authorization dossier | Dossier includes all required elements |
| **G4** | Registry-ready XML | XSD-valid XML for EU Transitional Registry | 100% XSD validation pass rate |
| **G5** | Full audit trail | Every value has evidence, lineage, provenance | 100% traceability |
| **G6** | Deterministic runs | Same inputs + versions = identical outputs | Hash verification passes |
| **G7** | Actionable errors | Clear errors with collect-all option | User can fix all issues in one pass |

### 5.2 Non-Goals (Explicitly Out of Scope)

| Non-Goal | Rationale | Future |
|----------|-----------|--------|
| Certificate purchase tracking | Definitive phase complexity | v2.0 |
| Supplier portal | UI complexity | v1.1+ |
| PDF extraction | Nondeterminism risk | v1.1+ |
| Cement/Fertilizers/Electricity/Hydrogen | Lower volume | v1.1+ |

### 5.3 Success Criteria

| Criterion | Validation Method |
|-----------|------------------|
| Policy engine enforces all period rules | Automated policy tests |
| Authorization readiness dossier generated | Expert review |
| XSD validation passes 100% | Schema validation tests |
| Default value usage tracked and capped | Run statistics verification |
| Collect-all mode shows all errors | User testing |
| Web UI functional | User testing |

---

## 6. Target Users & Personas

### 6.1 Primary Persona: Trade Compliance Manager (SMB Importer)

| Attribute | Description |
|-----------|-------------|
| **Company Size** | 50-500 employees, €10M-€200M revenue |
| **Import Volume** | 10-100 CBAM import lines per quarter |
| **Technical Skill** | Comfortable with Excel, prefers visual interfaces |
| **2026 Pain Points** | Authorization deadlines, methodology confusion, penalty fear |
| **Primary Tool** | Excel/spreadsheets |

**Quote:** *"I need to know if I'm compliant for 2026. Show me what I need to do and help me file correctly."*

### 6.2 Interface Requirements (Based on Persona)

| Requirement | Implementation |
|-------------|----------------|
| Excel-first workflow | Upload Excel, get results |
| Visual feedback | Web UI with progress, results display |
| Error summary | Show all errors at once (collect-all mode) |
| Guidance | Setup wizard for first-time users |
| Downloads | One-click download of all artifacts |

---

## 7. MVP Scope Definition

### 7.1 Two Operational Modes

#### Mode 1: Transitional Closeout

| Aspect | Details |
|--------|---------|
| **Purpose** | Q4 2025 reporting, corrections, amendments |
| **Output** | Quarterly XML + audit bundle |
| **Timeline** | Q4 2025 deadline: January 31, 2026 |
| **Policy** | Enforce 20% default cap for complex goods |

#### Mode 2: 2026 Compliance Onboarding

| Aspect | Details |
|--------|---------|
| **Purpose** | Prepare for 2026 operational compliance |
| **Output** | Emissions ledger + authorization readiness dossier |
| **Timeline** | Authorization deadline: March 31, 2026 |
| **Policy** | Check tonnage thresholds, generate auth requirements |

### 7.2 Product Categories (Phased)

| Phase | Categories | CN Codes | Volume Share |
|-------|-----------|----------|--------------|
| **MVP (P0)** | Iron & Steel | 72xx, 73xx | ~98% |
| **Fast-Follow (P1)** | Aluminum | 76xx | ~2% |
| **Future (P2)** | Cement, Fertilizers, etc. | Various | <1% |

### 7.3 Interface Deliverables

| Interface | Description | Priority |
|-----------|-------------|----------|
| **CLI** | `gl-cbam run` command | P0 |
| **Web UI** | Browser-based interface at localhost:8000 | P0 |
| **Setup Wizard** | First-time configuration helper | P1 |

---

## 8. Functional Requirements

### 8.1 P0 - Must Have

#### FR-1: Dual Mode Operation

| Requirement | Details |
|-------------|---------|
| **Mode Selection** | `--mode transitional` or `--mode operational` |
| **Default** | Transitional mode |
| **Behavior** | Different policy rules and outputs per mode |

#### FR-2: Policy Engine

| Requirement | Details |
|-------------|---------|
| **Period Detection** | Automatically detect reporting period from config |
| **Rule Enforcement** | Apply correct default value rules per period |
| **Cap Enforcement** | Reject if defaults exceed 20% for post-Q2-2024 periods |
| **Compliance Check** | Validate against EU method requirements |

#### FR-3: Authorization Readiness (Mode 2)

| Requirement | Details |
|-------------|---------|
| **Tonnage Check** | Calculate annual tonnage, flag if >50t |
| **Authorization Status** | Track if authorization/application exists |
| **Dossier Generation** | Create structured package for registry workflow |
| **Deadline Alerts** | Flag approaching authorization deadline |

#### FR-4: Web Interface

| Requirement | Details |
|-------------|---------|
| **Upload** | Drag-and-drop config YAML and imports file |
| **Progress** | Real-time progress display |
| **Results** | Display emissions summary, compliance status |
| **Downloads** | One-click download of all artifacts |
| **Errors** | Display all validation errors with fixes |

#### FR-5: Collect-All Error Mode

| Requirement | Details |
|-------------|---------|
| **Flag** | `--collect-errors` or checkbox in Web UI |
| **Behavior** | Collect all validation errors before stopping |
| **Output** | Single error report with all issues |
| **UX** | User fixes all issues in one iteration |

#### FR-6: XML Generation + XSD Validation

(Same as v1.0 - unchanged)

#### FR-7: Audit Bundle

(Same as v1.0 - unchanged)

### 8.2 P1 - Should Have

#### FR-8: Setup Wizard

| Requirement | Details |
|-------------|---------|
| **First Run** | Detect first run, offer wizard |
| **Steps** | Company info → Reporting period → File selection |
| **Output** | Generated config.yaml |

#### FR-9: Aluminum Support (Fast-Follow)

| Requirement | Details |
|-------------|---------|
| **CN Codes** | 76xx |
| **Factors** | Add aluminum emission factors |
| **Timeline** | 2-4 weeks after initial release |

---

## 9. Non-Functional Requirements

### 9.1 Policy Compliance

| Requirement | Specification |
|-------------|---------------|
| **Period-Aware Rules** | Different validation per reporting period |
| **Default Cap** | Enforce 20% estimation cap where applicable |
| **Method Validation** | Ensure EU method compliance from 2025 |
| **Authorization Check** | Verify requirements for >50t importers |

### 9.2 User Experience

| Requirement | Specification |
|-------------|---------------|
| **Time to First Report** | <15 minutes via Web UI |
| **Error Resolution** | Single iteration via collect-all mode |
| **Visual Feedback** | Real-time progress in Web UI |
| **Mobile Responsive** | Web UI works on tablet |

### 9.3 Performance

| Requirement | Specification |
|-------------|---------------|
| **Import Lines** | Handle 1,000+ lines |
| **Web UI Response** | <3 second initial load |
| **Processing** | <60 seconds for 100 lines |

---

## 10. Data Model & Artifacts

### 10.1 New: Authorization Readiness Dossier

```json
{
  "dossier_type": "authorization_readiness",
  "generated_at": "2026-01-15T10:30:00Z",
  "company": {
    "name": "Example Importer GmbH",
    "eori_number": "DE123456789012345"
  },
  "authorization_status": {
    "required": true,
    "reason": "Annual imports exceed 50 tonnes",
    "annual_tonnage_estimate": 250.5,
    "application_deadline": "2026-03-31",
    "current_status": "not_applied"
  },
  "2025_summary": {
    "q1_emissions_tco2e": 450.2,
    "q2_emissions_tco2e": 380.5,
    "q3_emissions_tco2e": 520.1,
    "q4_emissions_tco2e": null,
    "total_tonnage": 185.3
  },
  "checklist": [
    {"item": "EORI number verified", "status": "complete"},
    {"item": "2025 Q1-Q3 reports submitted", "status": "complete"},
    {"item": "Q4 2025 report prepared", "status": "pending"},
    {"item": "Authorization application submitted", "status": "pending"}
  ],
  "recommended_actions": [
    "Submit Q4 2025 report by January 31, 2026",
    "Apply for CBAM authorization before March 31, 2026"
  ]
}
```

### 10.2 New: Policy Validation Result

```json
{
  "policy_version": "2026.1",
  "reporting_period": "Q4 2025",
  "checks": [
    {
      "rule": "default_value_cap",
      "status": "pass",
      "details": "Default values: 18% (under 20% cap)"
    },
    {
      "rule": "eu_method_compliance",
      "status": "pass",
      "details": "All calculations use EU method"
    },
    {
      "rule": "complex_goods_handling",
      "status": "pass",
      "details": "No complex goods detected"
    }
  ],
  "overall_status": "compliant"
}
```

### 10.3 Updated Output Artifacts

| Artifact | Mode 1 | Mode 2 | Description |
|----------|--------|--------|-------------|
| `cbam_report.xml` | ✓ | - | EU Registry XML |
| `emissions_ledger.json` | ✓ | ✓ | Structured emissions data |
| `authorization_dossier.json` | - | ✓ | Authorization readiness package |
| `policy_validation.json` | ✓ | ✓ | Policy compliance checks |
| `report_summary.xlsx` | ✓ | ✓ | Human-readable summary |
| `claims.json` | ✓ | ✓ | Claim graph |
| `lineage.json` | ✓ | ✓ | Provenance graph |
| `assumptions.json` | ✓ | ✓ | Assumptions registry |
| `gap_report.json` | ✓ | ✓ | Missing data opportunities |
| `run_manifest.json` | ✓ | ✓ | Version manifest |

---

## 11. Agent Architecture & Pipeline

### 11.1 Updated 8-Agent Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               CBAM Compliance Essentials Pipeline (2026)                     │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │   Inputs     │
     │  (CSV/YAML)  │
     └──────┬───────┘
            │
            ▼
┌───────────────────────┐
│  Agent 1: Orchestrator │  Plans execution, manages modes
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Agent 2: Validator   │  Schema + business rules
│  (Fail-fast or        │  + collect-all mode
│   Collect-all)        │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Agent 3: Policy      │  *** NEW ***
│  Engine               │  Period-aware rules,
│                       │  default caps, EU method
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Agent 4: Normalizer  │  Units, CN codes, countries
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Agent 5: Factor      │  Emission factors
│  Library              │  with policy constraints
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Agent 6: Calculator  │  Emissions computation
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Agent 7: Exporter    │  XML + Ledger + Dossier
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Agent 8: Evidence    │  Audit bundle
│  Packager             │
└───────────┬───────────┘
            │
            ▼
     ┌──────────────┐
     │   Outputs    │
     │ (XML/Bundle) │
     └──────────────┘
```

### 11.2 New Agent: Policy Engine (Agent 3)

| Attribute | Specification |
|-----------|---------------|
| **Agent ID** | GL-CBAM-POLICY-001 |
| **Layer** | Foundation & Governance |
| **Inputs** | Reporting period, calculation results, method selections |
| **Outputs** | Policy validation result, compliance flags, warnings |
| **Rules Enforced** | Default cap (20%), EU method, authorization requirements |

---

## 12. Policy Engine & Compliance Rules

### 12.1 Reporting Period Rules

```python
POLICY_RULES = {
    # Q4 2023 - Q2 2024: Defaults allowed without limit
    "2023-Q4": {"default_cap": None, "method": "any"},
    "2024-Q1": {"default_cap": None, "method": "any"},
    "2024-Q2": {"default_cap": None, "method": "any"},

    # Q3 2024 - Q4 2025: 20% cap on estimations
    "2024-Q3": {"default_cap": 0.20, "method": "eu_preferred"},
    "2024-Q4": {"default_cap": 0.20, "method": "eu_preferred"},
    "2025-Q1": {"default_cap": 0.20, "method": "eu_only"},
    "2025-Q2": {"default_cap": 0.20, "method": "eu_only"},
    "2025-Q3": {"default_cap": 0.20, "method": "eu_only"},
    "2025-Q4": {"default_cap": 0.20, "method": "eu_only"},

    # 2026+: Operational phase
    "2026-Q1": {"default_cap": 0.20, "method": "eu_only", "auth_required": True},
}
```

### 12.2 Default Value Cap Enforcement

```python
def validate_default_usage(results, period_rules):
    """
    Validate that default value usage doesn't exceed cap.
    """
    if period_rules["default_cap"] is None:
        return {"status": "pass", "message": "No cap for this period"}

    total_emissions = sum(r.total_emissions for r in results)
    default_emissions = sum(r.total_emissions for r in results
                           if r.method == "default")

    default_percentage = default_emissions / total_emissions

    if default_percentage > period_rules["default_cap"]:
        return {
            "status": "fail",
            "message": f"Default values: {default_percentage:.1%} exceeds {period_rules['default_cap']:.0%} cap",
            "recommendation": "Obtain supplier-specific data for more imports"
        }

    return {"status": "pass", "message": f"Default values: {default_percentage:.1%} (within cap)"}
```

### 12.3 Authorization Threshold Check

```python
def check_authorization_requirement(annual_tonnage):
    """
    Check if authorization is required (>50 tonnes/year).
    """
    THRESHOLD = 50  # tonnes

    if annual_tonnage > THRESHOLD:
        return {
            "required": True,
            "reason": f"Annual imports ({annual_tonnage:.1f}t) exceed {THRESHOLD}t threshold",
            "deadline": "2026-03-31",
            "action": "Submit authorization application"
        }

    return {
        "required": False,
        "reason": f"Annual imports ({annual_tonnage:.1f}t) below {THRESHOLD}t threshold"
    }
```

---

## 13. User Journeys

### 13.1 Journey 1: Web UI First Report (Primary)

**User:** Trade Compliance Manager
**Goal:** Generate Q4 2025 report using Web UI

| Step | User Action | System Response |
|------|-------------|-----------------|
| 1 | Open browser to localhost:8000 | Web UI loads with upload form |
| 2 | Drag config.yaml to upload zone | Config parsed, company info displayed |
| 3 | Drag imports.xlsx to upload zone | File validated, 25 lines detected |
| 4 | Click "Generate Report" | Progress bar shows stages |
| 5 | Review results | Emissions summary, compliance status shown |
| 6 | View errors (if any) | All errors listed with fixes |
| 7 | Click "Download All" | ZIP with all artifacts downloaded |

**Time:** ~10 minutes

### 13.2 Journey 2: Authorization Readiness Check

**User:** Importer preparing for 2026
**Goal:** Determine authorization requirements

| Step | User Action | System Response |
|------|-------------|-----------------|
| 1 | Select "2026 Compliance Mode" | Mode switched |
| 2 | Upload 2025 import data | Annual tonnage calculated |
| 3 | View authorization status | "Required: Yes - 250t exceeds 50t threshold" |
| 4 | Download dossier | authorization_dossier.json with checklist |
| 5 | Review recommended actions | Clear next steps displayed |

---

## 14. Input/Output Specifications

### 14.1 Updated Config Schema

```yaml
# cbam_config.yaml (v2.0)
version: "2.0"

# Mode selection
mode: "transitional"  # "transitional" or "operational"

# Declarant Information
declarant:
  name: "Example Importer GmbH"
  eori_number: "DE123456789012345"
  address:
    street: "Industriestrasse 1"
    city: "Berlin"
    postal_code: "10115"
    country: "DE"
  contact:
    name: "Hans Mueller"
    email: "hans.mueller@example.com"

# Reporting Period
reporting_period:
  quarter: "Q4"
  year: 2025

# Authorization Status (for 2026 mode)
authorization:
  status: "not_applied"  # "not_applied", "pending", "approved"
  application_reference: null
  approval_reference: null

# Settings
settings:
  # Error handling mode
  error_mode: "fail_fast"  # "fail_fast" or "collect_all"

  # Policy enforcement
  enforce_default_cap: true

  # Aggregation
  aggregation: "by_product_origin"
```

---

## 15. Error Handling & Validation

### 15.1 Updated Error Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Fail-Fast** | Stop at first error | CI/CD, automated pipelines |
| **Collect-All** | Gather all errors, then stop | Interactive use, Excel-first users |

### 15.2 New Policy Errors

| Error Code | Category | Description |
|------------|----------|-------------|
| `POL-001` | Policy | Default value cap exceeded |
| `POL-002` | Policy | Non-EU method used after 2025 |
| `POL-003` | Policy | Authorization required but not provided |
| `POL-004` | Policy | Complex goods require actual data |

### 15.3 Error Report Format (Collect-All)

```json
{
  "total_errors": 5,
  "by_category": {
    "validation": 3,
    "policy": 2
  },
  "errors": [
    {
      "code": "VAL-004",
      "location": "imports.xlsx:12:cn_code",
      "message": "CN code '7201000' has only 7 digits",
      "fix": "Add leading zero: '07201000'"
    },
    {
      "code": "POL-001",
      "location": "global",
      "message": "Default values: 25% exceeds 20% cap",
      "fix": "Obtain supplier data for at least 5% more emissions"
    }
  ]
}
```

---

## 16. Security & Privacy

(Same as v1.0 - Local-first, no network)

---

## 17. Testing Strategy

### 17.1 New Policy Tests

| Test | Description |
|------|-------------|
| `test_q2_2024_no_cap` | Verify no cap enforced for Q2 2024 |
| `test_q3_2024_20_percent_cap` | Verify 20% cap enforced |
| `test_2025_eu_method_only` | Verify EU method required |
| `test_authorization_threshold` | Verify 50t threshold detection |

### 17.2 Web UI Tests

| Test | Description |
|------|-------------|
| `test_file_upload` | Verify drag-and-drop upload |
| `test_progress_display` | Verify real-time progress |
| `test_error_display` | Verify all errors shown |
| `test_download_all` | Verify ZIP generation |

---

## 18. Deployment & Packaging

### 18.1 Installation Options

```bash
# CLI only
pip install greenlang-cbam-pack

# With Web UI
pip install greenlang-cbam-pack[web]

# Docker (includes Web UI)
docker pull greenlang/cbam-pack:2.0.0
```

### 18.2 Web UI Startup

```bash
# Start Web UI
gl-cbam web --port 8000

# Or with Docker
docker run -p 8000:8000 greenlang/cbam-pack:2.0.0 web
```

---

## 19-24. (Remaining Sections)

(Observability, Regulatory Accuracy, Metrics, Risks, Milestones, Future - updated with 2026 context)

---

## Appendices

### Appendix A: CTO Review Feedback Implementation

| CTO Feedback | Implementation |
|--------------|----------------|
| "Transitional-only is time-misaligned" | Added 2026 operational mode |
| "Defaults-first is not compliant" | Added policy engine with period-aware rules |
| "Authorization is central to 2026" | Added authorization readiness checks |
| "Iron & Steel dominates volumes" | Made Aluminum fast-follow |
| "Add collect-all errors for Excel users" | Added collect-all error mode |
| "Add wizard to reduce friction" | Added setup wizard + Web UI |

### Appendix B: Updated CLI Reference

```bash
# Transitional mode (Q4 2025)
gl-cbam run --mode transitional --config cbam.yaml --imports imports.xlsx --out ./output/

# Operational mode (2026)
gl-cbam run --mode operational --config cbam.yaml --imports imports.xlsx --out ./output/

# Collect all errors
gl-cbam run --config cbam.yaml --imports imports.xlsx --out ./output/ --collect-errors

# Start Web UI
gl-cbam web --port 8000

# Validate only
gl-cbam validate --config cbam.yaml --imports imports.xlsx
```

---

**End of Document**

*GreenLang CBAM Compliance Essentials Pack PRD v2.0.0 - Updated per CTO Review*
