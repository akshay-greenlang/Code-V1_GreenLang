# AGENT-EUDR-024: Third-Party Audit Manager -- Technical Architecture Specification

## Document Info

| Field | Value |
|-------|-------|
| **Document ID** | ARCH-AGENT-EUDR-024 |
| **Agent ID** | GL-EUDR-TAM-024 |
| **Component** | Third-Party Audit Manager Agent |
| **Category** | EUDR Regulatory Agent -- Audit & Verification Management |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Architecture Specification |
| **Author** | GL-AppArchitect |
| **Date** | 2026-03-10 |
| **Regulation** | Regulation (EU) 2023/1115 -- EUDR, Articles 4, 9, 10, 11, 14-16, 18-23, 29, 31; ISO 19011:2018; ISO/IEC 17065:2012; ISO/IEC 17021-1:2015 |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **DB Migration** | V112 |
| **Metric Prefix** | `gl_eudr_tam_` |
| **Config Prefix** | `GL_EUDR_TAM_` |
| **API Prefix** | `/v1/eudr-tam` |
| **RBAC Prefix** | `eudr-tam:` |

---

## 1. Executive Summary

### 1.1 Purpose

AGENT-EUDR-024 Third-Party Audit Manager is a specialized compliance agent providing end-to-end management of the third-party audit lifecycle for EUDR compliance verification. It manages audit planning and risk-based scheduling, auditor qualification tracking per ISO/IEC 17065 and ISO/IEC 17021-1, audit execution monitoring with EUDR-specific checklists, non-conformance detection and classification (critical/major/minor), corrective action request (CAR) lifecycle management with SLA enforcement, certification scheme audit coordination across 5 major schemes (FSC, PEFC, RSPO, Rainforest Alliance, ISCC), ISO 19011:2018 compliant report generation, competent authority liaison workflows for 27 EU Member States, and audit analytics with trend detection.

The agent is the 24th in the EUDR agent family and establishes a new Audit and Verification Management sub-category. It integrates with 7 existing EUDR agents: EUDR-001 (Supply Chain Mapping Master), EUDR-016 (Country Risk Evaluator), EUDR-017 (Supplier Risk Scorer), EUDR-020 (Deforestation Alert System), EUDR-021 (Indigenous Rights Checker), EUDR-022 (Protected Area Validator), and EUDR-023 (Legal Compliance Verifier).

### 1.2 Regulatory Driver

EUDR Articles 10-11 mandate risk-based due diligence including independent verification audits. Articles 14-16 empower competent authorities to conduct checks on operators. Recital 43 references certification schemes (FSC, PEFC, RSPO) as complementary tools. Non-compliance exposes operators to penalties of up to 4% of annual EU turnover under Articles 22-23.

### 1.3 Key Differentiators

- **Risk-based audit scheduling** driven by EUDR-016/017/020 risk signals, not calendar-based recertification
- **Five-scheme integration** with FSC, PEFC, RSPO, Rainforest Alliance, ISCC for cross-scheme redundancy reduction
- **Competent authority liaison** for 27 EU Member State inspection response management
- **ISO 19011 compliant reporting** with automated evidence package assembly
- **Zero-hallucination guarantee** -- all scheduling, classification, and scoring via deterministic rule engines
- **Full CAR lifecycle management** with SLA enforcement, 4-stage escalation, and closure verification

### 1.4 Performance Targets

| Metric | Target |
|--------|--------|
| Audit schedule generation (500 suppliers) | < 2s |
| NC classification (per finding) | < 500ms |
| CAR SLA countdown update latency | < 1 minute |
| Audit report generation (50-finding PDF) | < 30s |
| Certification status sync (per scheme) | < 5 minutes |
| Auditor matching query (1,000 auditors) | < 500ms |
| Analytics dashboard load (12-month window) | < 3s |
| API p95 latency (standard queries) | < 200ms |
| Evidence file upload (100 MB) | < 60s |
| Authority evidence package assembly | < 2 minutes |
| Redis cache hit rate target | > 65% |

### 1.5 Development Estimates

| Phase | Scope | Duration | Engineers |
|-------|-------|----------|-----------|
| Phase 1 | Core engines (1-5), models, config, provenance | 3 weeks | 2 |
| Phase 2 | Engines 6-7, scheme integration, report generation | 2 weeks | 2 |
| Phase 3 | Authority liaison, analytics, API routes, auth | 2 weeks | 2 |
| Phase 4 | DB migration, Grafana dashboard, integration testing | 1 week | 2 |
| Phase 5 | Testing (unit + integration + performance), security audit | 2 weeks | 2 |
| **Total** | **Complete agent** | **10 weeks** | **2 engineers** |

### 1.6 Estimated Output

- ~45 files (agent code + API + reference data)
- ~42K lines of code
- ~800+ tests
- V112 database migration (13 tables + 4 hypertables)
- 1 Grafana dashboard (20 panels)

---

## 2. System Architecture Overview

### 2.1 Component Diagram

```
+-----------------------------------------------------------------------------------+
|                              GL-EUDR-APP v1.0 (Frontend)                          |
+--------------------------------------+--------------------------------------------+
                                       |
+--------------------------------------v--------------------------------------------+
|                           Unified API Layer (FastAPI)                              |
|                         /v1/eudr-tam  (~35 endpoints)                             |
+---+------+------+------+------+------+------+------+------+----------------------+
    |      |      |      |      |      |      |      |      |
    v      v      v      v      v      v      v      v      v
+------+ +------+ +------+ +------+ +------+ +------+ +------+
|Plan  | |Audit | |Exec  | |NC    | |CAR   | |Scheme| |Report|
|Engine| |Reg   | |Engine| |Engine| |Engine| |Integ | |Engine|
|(E1)  | |(E2)  | |(E3)  | |(E4)  | |(E5)  | |(E6)  | |(E7)  |
+--+---+ +--+---+ +--+---+ +--+---+ +--+---+ +--+---+ +--+---+
   |        |        |        |        |        |        |
   +--------+--------+--------+--------+--------+--------+
                              |
              +---------------+----------------+
              |               |                |
    +---------v--+   +--------v---+   +--------v-------+
    | PostgreSQL |   |   Redis    |   |  S3 / MinIO    |
    | TimescaleDB|   |  Cache     |   | Evidence Store |
    | (17 tables)|   | SLA/Status |   | Report Store   |
    +------------+   +------------+   +----------------+

+-----------------------------------------------------------------------------------+
|                        Upstream EUDR Agent Integrations                            |
|                                                                                   |
|  EUDR-001    EUDR-016    EUDR-017    EUDR-020    EUDR-021   EUDR-022   EUDR-023  |
|  Supply      Country     Supplier    Deforestation Indigenous Protected  Legal     |
|  Chain Map   Risk Eval   Risk Score  Alert System  Rights    Area Valid  Compliance|
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|                     External Certification Scheme APIs                             |
|  FSC DB  |  PEFC Search  |  RSPO PalmTrace  |  RA Portal  |  ISCC DB             |
+-----------------------------------------------------------------------------------+
```

### 2.2 Engine Interaction Flow

```
                     Risk Signals
                  (EUDR-016/017/020)
                         |
                         v
               +-------------------+
               | E1: Audit Planning|-------> Audit Schedule
               |    & Scheduling   |         (audit_schedule hypertable)
               +--------+----------+
                        |
                        v
               +-------------------+
               | E2: Auditor       |-------> Auditor Assignment
               |    Registry       |         (competence matching)
               +--------+----------+
                        |
                        v
               +-------------------+
               | E3: Audit         |-------> Checklists, Evidence
               |    Execution      |         (S3 evidence store)
               +--------+----------+
                        |
                        v
               +-------------------+
               | E4: Non-Conformance|------> NC Records
               |    Detection      |         (nc_trend_log hypertable)
               +--------+----------+
                        |
                        v
               +-------------------+
               | E5: CAR           |-------> CAR Lifecycle
               |    Management     |         (car_sla_log hypertable)
               +--------+----------+
                        |
                        |    +-------------------+
                        +--->| E6: Certification  |<--- Scheme APIs
                        |    |    Integration     |
                        |    +-------------------+
                        v
               +-------------------+
               | E7: Audit         |-------> PDF/JSON/HTML/XLSX/XML
               |    Reporting      |         (S3 report store)
               +-------------------+
```

---

## 3. Module Structure

### 3.1 Directory Layout

```
greenlang/agents/eudr/third_party_audit/
    __init__.py                          # Public API exports
    config.py                            # GL_EUDR_TAM_ env prefix configuration
    models.py                            # Pydantic v2 models (~35 models)
    metrics.py                           # 20 Prometheus metrics (gl_eudr_tam_ prefix)
    provenance.py                        # SHA-256 provenance tracking
    setup.py                             # ThirdPartyAuditService facade
    #
    # === 7 Processing Engines ===
    #
    audit_planner.py                     # Engine 1: Risk-based scheduling
    auditor_registry.py                  # Engine 2: Qualification tracking
    audit_execution.py                   # Engine 3: Checklist and evidence management
    nc_engine.py                         # Engine 4: Detection, classification, RCA
    car_manager.py                       # Engine 5: Lifecycle, SLA, escalation
    scheme_integration.py                # Engine 6: 5-scheme connector
    report_generator.py                  # Engine 7: ISO 19011 compliant reports
    #
    # === API Layer ===
    #
    api/
        __init__.py
        router.py                        # Main router (/v1/eudr-tam), sub-router aggregation
        dependencies.py                  # FastAPI dependencies (service injection, auth)
        schemas.py                       # API-specific request/response schemas
        planning_routes.py               # Audit planning and scheduling (4 endpoints)
        auditor_routes.py                # Auditor registry and matching (5 endpoints)
        execution_routes.py              # Audit execution and checklist (5 endpoints)
        nc_routes.py                     # Non-conformance management (5 endpoints)
        car_routes.py                    # CAR lifecycle (5 endpoints)
        scheme_routes.py                 # Certification scheme (3 endpoints)
        report_routes.py                 # Report generation (2 endpoints)
        authority_routes.py              # Competent authority liaison (3 endpoints)
        analytics_routes.py              # Analytics and dashboard (5 endpoints)
        admin_routes.py                  # Admin and health (2 endpoints)
    #
    # === Reference Data ===
    #
    reference_data/
        __init__.py
        audit_criteria.py                # EUDR audit checklist criteria (versioned)
        scheme_criteria.py               # FSC/PEFC/RSPO/RA/ISCC audit criteria mappings
        nc_classification_rules.py       # Deterministic NC severity rules
        car_sla_definitions.py           # SLA deadlines and escalation rules
        authority_profiles.py            # 27 EU Member State competent authority profiles
        scheme_eudr_coverage.py          # Scheme-to-EUDR coverage matrix
```

### 3.2 Test Directory Layout

```
tests/agents/eudr/third_party_audit/
    __init__.py
    conftest.py                          # Shared fixtures, mock services, test data
    test_audit_planner.py                # Engine 1: 80+ tests
    test_auditor_registry.py             # Engine 2: 50+ tests
    test_audit_execution.py              # Engine 3: 70+ tests
    test_nc_engine.py                    # Engine 4: 80+ tests
    test_car_manager.py                  # Engine 5: 80+ tests
    test_scheme_integration.py           # Engine 6: 60+ tests
    test_report_generator.py             # Engine 7: 50+ tests
    test_authority_liaison.py            # Authority workflows: 40+ tests
    test_analytics.py                    # Analytics engine: 50+ tests
    test_api_routes.py                   # API layer: 50+ tests
    test_models.py                       # Pydantic models: 40+ tests
    test_config.py                       # Configuration: 15+ tests
    test_provenance.py                   # Provenance chain: 20+ tests
    test_golden_scenarios.py             # 50 golden test scenarios
    test_determinism.py                  # 15+ bit-perfect reproducibility tests
    test_performance.py                  # 25+ performance benchmark tests
    test_security.py                     # 10+ security and RBAC tests
```

### 3.3 Deployment Artifacts

```
deployment/
    database/migrations/sql/
        V112__agent_eudr_third_party_audit_manager.sql
    monitoring/dashboards/
        eudr-third-party-audit-manager.json
```

---

## 4. Engine Specifications

### 4.1 Engine 1: Audit Planning and Scheduling Engine

**File:** `audit_planner.py`
**Purpose:** Generates risk-based audit schedules by dynamically calculating audit frequency, scope, and depth from upstream risk signals.

**Responsibilities:**
- Calculate composite audit priority score per supplier using 5 weighted risk factors
- Assign audit frequency tiers: HIGH (quarterly), STANDARD (semi-annual), LOW (annual)
- Determine audit scope per supplier: FULL, TARGETED, SURVEILLANCE, UNSCHEDULED
- Determine audit depth per risk level: on-site field verification, on-site document review, remote desktop review
- Generate annual audit calendar with quarterly review cycles
- Detect scheduling conflicts (auditor unavailability, overlapping audits, holiday restrictions)
- Trigger unscheduled audits from events (deforestation alerts, certification suspensions, authority requests)
- Integrate certification scheme recertification timelines (FSC 5y, PEFC 5y, RSPO 5y, RA 3y, ISCC 1y)
- Track audit resource budget (auditor-days) with utilization forecasts
- Support multi-site audit planning for suppliers with multiple production sites

**Risk-Based Audit Priority Formula (Deterministic):**

```python
from decimal import Decimal, ROUND_HALF_UP

def calculate_audit_priority_score(
    country_risk: Decimal,          # 0-100 from EUDR-016
    supplier_risk: Decimal,         # 0-100 from EUDR-017
    nc_history_score: Decimal,      # weighted sum: critical=30, major=15, minor=5 / audit_count
    certification_gap_score: Decimal,  # (1 - certification_coverage) * 100
    deforestation_alert_score: Decimal,  # max alert severity within 25km, 0-100
    days_since_last_audit: int,
    scheduled_interval_days: int,
) -> Decimal:
    """All inputs are Decimal; output is Decimal(5,2). Deterministic."""
    WEIGHT_COUNTRY = Decimal("0.25")
    WEIGHT_SUPPLIER = Decimal("0.25")
    WEIGHT_NC = Decimal("0.20")
    WEIGHT_CERT = Decimal("0.15")
    WEIGHT_DEFOR = Decimal("0.15")

    base_score = (
        country_risk * WEIGHT_COUNTRY
        + supplier_risk * WEIGHT_SUPPLIER
        + nc_history_score * WEIGHT_NC
        + certification_gap_score * WEIGHT_CERT
        + deforestation_alert_score * WEIGHT_DEFOR
    )

    recency_multiplier = min(
        Decimal(str(days_since_last_audit)) / Decimal(str(scheduled_interval_days)),
        Decimal("2.0")
    )

    return (base_score * recency_multiplier).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

# Frequency assignment (deterministic thresholds):
#   score >= 70.00  -> HIGH   (quarterly, 90-day interval)
#   score >= 40.00  -> STANDARD (semi-annual, 180-day interval)
#   score <  40.00  -> LOW   (annual, 365-day interval)
```

**Upstream Dependencies:**
- EUDR-016 Country Risk Evaluator: `get_country_risk_for_audit(country_code)` returns risk_score (0-100)
- EUDR-017 Supplier Risk Scorer: `get_supplier_risk(supplier_id)` returns composite_risk (0-100)
- EUDR-020 Deforestation Alert System: `get_deforestation_alerts_for_supplier(supplier_id)` returns alert_risk_score (0-100)
- EUDR-001 Supply Chain Mapping Master: `get_supplier_sites(operator_id)` returns supplier inventory with site locations

**Event-Triggered Unscheduled Audit Rules (Deterministic):**

| Trigger Event | Source Agent | Trigger Window | Audit Scope |
|--------------|-------------|----------------|-------------|
| Deforestation alert (HIGH/CRITICAL severity) | EUDR-020 | Within 14 days | UNSCHEDULED + FULL |
| Critical NC from related supplier | Self (E4) | Within 30 days | UNSCHEDULED + TARGETED |
| Certification suspension | Self (E6) | Within 7 days | UNSCHEDULED + FULL |
| Competent authority request | Self (authority liaison) | Per authority deadline | UNSCHEDULED + scope per request |
| Country risk reclassification (Standard->High) | EUDR-016 | Within 48 hours (recalc) | Recalculate all affected |

**Edge Cases:**
- New supplier with no audit history: assign HIGH frequency for first audit cycle
- Supplier with expired certification: trigger immediate unscheduled audit
- Multiple deforestation alerts for same supplier: use highest severity, no duplicate triggers
- Country risk reclassification: recalculate all affected supplier schedules within 48 hours

**Zero-Hallucination Guarantee:** All scheduling uses deterministic Decimal arithmetic with fixed weights and thresholds. No LLM in the scheduling path. Same risk inputs always produce same schedule output (bit-perfect).

**Estimated Lines of Code:** 3,500-4,000

---

### 4.2 Engine 2: Auditor Registry and Qualification Engine

**File:** `auditor_registry.py`
**Purpose:** Maintains a centralized registry of auditor profiles compliant with ISO/IEC 17065 and ISO/IEC 17021-1 competence requirements.

**Responsibilities:**
- Maintain auditor profiles: name, ID, organization, accreditation status, expiry date, scope
- Track commodity-sector competence per auditor (7 EUDR commodities)
- Track scheme-specific qualifications (FSC Lead Auditor, RSPO Lead Assessor, etc.)
- Track regional competence: country expertise, language proficiency
- Manage conflict-of-interest declarations with rotation requirements per ISO/IEC 17021-1 Clause 7.2.8
- Record audit performance history: audit count, findings per audit ratio, CAR closure rate
- Validate certification body (CB) accreditation against IAF MLA signatories
- Implement auditor-to-audit matching algorithm (competence match scoring)
- Track CPD hours and compliance per accreditation body requirements
- Alert on qualification expiry (60-day advance warning for accreditation, CPD shortfall, rotation requirement)
- Support bulk import from certification body registries

**Auditor Matching Algorithm (Deterministic):**

```python
def calculate_auditor_match_score(
    auditor: AuditorProfile,
    audit_requirements: AuditRequirements,
) -> Decimal:
    """
    Returns 0-100 match score. Deterministic.
    Disqualifying conditions return score = 0.
    """
    # Disqualifiers (hard constraints)
    if auditor.accreditation_status != "active":
        return Decimal("0")
    if auditor.accreditation_expiry < audit_requirements.audit_date:
        return Decimal("0")
    if not auditor.cpd_compliant:
        return Decimal("0")
    if _has_conflict_of_interest(auditor, audit_requirements.supplier_id):
        return Decimal("0")

    score = Decimal("0")

    # Commodity competence (weight: 30)
    if audit_requirements.commodity in auditor.commodity_competencies:
        score += Decimal("30")

    # Scheme qualification (weight: 25)
    if audit_requirements.scheme in auditor.scheme_qualifications:
        score += Decimal("25")

    # Country expertise (weight: 20)
    if audit_requirements.country_code in auditor.country_expertise:
        score += Decimal("20")

    # Language match (weight: 15)
    if audit_requirements.required_language in auditor.languages:
        score += Decimal("15")

    # Performance rating (weight: 10)
    score += (auditor.performance_rating / Decimal("100")) * Decimal("10")

    return score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
```

**Data Sources:**
- FSC accredited auditor list (info.fsc.org)
- RSPO accredited assessor list (rspo.org)
- PEFC certified body registry (pefc.org)
- Rainforest Alliance auditor database
- IAF MLA signatory list for CB accreditation verification

**Zero-Hallucination Guarantee:** Auditor matching uses deterministic weighted scoring with fixed weights. No LLM used for competence evaluation. Conflict-of-interest checks use rule-based evaluation against declared relationships.

**Estimated Lines of Code:** 2,800-3,200

---

### 4.3 Engine 3: Audit Execution Engine

**File:** `audit_execution.py`
**Purpose:** Manages real-time audit execution including checklist management, evidence collection, sampling plans, and progress tracking.

**Responsibilities:**
- Provide EUDR-specific audit checklists covering Articles 3, 4, 9, 10, 11 and Article 2(40) categories
- Provide scheme-specific checklists for FSC (P1-P10), PEFC (C1-C7), RSPO (P1-P7), RA (Ch 1-6), ISCC (SR 1-6)
- Auto-map scheme criteria to EUDR articles for unified checklist view
- Track checklist completion in real-time (pass/fail/NA per criterion with evidence attachment)
- Manage evidence collection with type classification, metadata tagging, and SHA-256 integrity hashing
- Implement sampling plan management per ISO 19011:2018 Annex A (statistical/judgmental)
- Track audit fieldwork schedule: site visits, interviews, document review sessions, opening/closing meetings
- Support audit modality: on-site, remote, hybrid, and unannounced
- Generate real-time audit progress dashboard data
- Manage audit team composition and role-based access
- Track stakeholder interviews with structured templates
- Support audit hold/suspension for critical mid-audit escalation
- Record audit completion with closing meeting notes and expected report delivery date

**Checklist Structure (Versioned):**

```python
class AuditCriterion(BaseModel):
    """Single audit checklist criterion."""
    criterion_id: str              # e.g., "EUDR-ART9-GEO-001"
    category: str                  # "eudr" | "fsc" | "pefc" | "rspo" | "ra" | "iscc"
    reference: str                 # "EUDR Art. 9(1)" or "FSC P6 C6.1"
    description: str
    eudr_article_mapping: Optional[str]  # Maps scheme criteria to EUDR article
    result: Optional[str]          # "pass" | "fail" | "na" | None
    evidence_ids: List[str]        # Linked evidence items
    auditor_notes: Optional[str]
    assessed_at: Optional[datetime]
    assessed_by: Optional[str]
```

**Evidence Management:**
- Evidence types: permit, certificate, photo, GPS record, interview transcript, lab result, document scan
- Metadata: date, location (lat/lon), source, capture device
- SHA-256 hash computed on upload for integrity verification
- Storage: S3 with path pattern `s3://gl-eudr-tam-evidence/{operator_id}/{audit_id}/{evidence_id}`
- Size limits: 100 MB per file, 5 GB per audit evidence package
- Encryption: AES-256-GCM at rest via SEC-003

**Sampling Plan (ISO 19011 Annex A):**

```python
def calculate_sample_size(
    population_size: int,
    risk_level: str,           # "high" | "standard" | "low"
    confidence_level: Decimal,  # default 95%
) -> int:
    """Deterministic sample size calculation."""
    RISK_MULTIPLIERS = {
        "high": Decimal("1.5"),
        "standard": Decimal("1.0"),
        "low": Decimal("0.7"),
    }
    base_sample = int(
        (Decimal(str(population_size)) ** Decimal("0.5")).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    adjusted = int(
        (Decimal(str(base_sample)) * RISK_MULTIPLIERS[risk_level]).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    return max(adjusted, 1)
```

**Zero-Hallucination Guarantee:** Checklist criteria are versioned reference data, not LLM-generated. Sample size calculation is deterministic. Evidence hashing uses SHA-256. Progress percentages use integer arithmetic (passed_count / total_count * 100).

**Estimated Lines of Code:** 4,000-4,500

---

### 4.4 Engine 4: Non-Conformance Detection Engine

**File:** `nc_engine.py`
**Purpose:** Classifies audit findings into structured non-conformances with deterministic severity levels and root cause analysis frameworks.

**Responsibilities:**
- Classify NCs into three severity levels: CRITICAL (30-day deadline), MAJOR (90-day), MINOR (365-day)
- Map each NC to EUDR articles, certification scheme clauses, and Article 2(40) categories
- Implement root cause analysis frameworks: "5 Whys" structured questioning, Ishikawa (fishbone) with 6 categories
- Link NCs to evidence collected during audit execution (Engine 3)
- Detect NC patterns across audits: recurring findings, regional systemic issues, seasonal patterns
- Assign NC risk impact score combining severity, article criticality, supply chain impact
- Manage NC status lifecycle: OPEN -> ACKNOWLEDGED -> CAR_ISSUED -> CAP_SUBMITTED -> IN_PROGRESS -> VERIFICATION_PENDING -> CLOSED | ESCALATED
- Generate NC summary reports per audit, supplier, region, and commodity
- Support NC dispute process with auditor review workflow
- Track observations and OFIs separately from NCs
- Integrate with EUDR-023 for Article 2(40) legal compliance NC mapping

**NC Severity Classification Rules (Deterministic, Rule-Based):**

```python
NC_CRITICAL_RULES = [
    # Rule ID, Condition, EUDR Article Reference
    ("CR-001", "deforestation_after_cutoff == True", "Art. 3"),
    ("CR-002", "fraudulent_documentation == True", "Art. 10(2)(e)"),
    ("CR-003", "production_in_protected_area_unauthorized == True", "Art. 9/10"),
    ("CR-004", "forced_labour_or_child_labour == True", "Art. 2(40) Cat 5"),
    ("CR-005", "intentional_mixing_undisclosed == True", "Art. 10(2)(b)"),
    ("CR-006", "missing_geolocation_pct > 50", "Art. 9"),
]

NC_MAJOR_RULES = [
    ("MJ-001", "10 <= missing_geolocation_pct <= 50", "Art. 9"),
    ("MJ-002", "incomplete_coc_pct > 20", "Art. 10(2)(f)"),
    ("MJ-003", "certification_expired_no_renewal == True", "Scheme"),
    ("MJ-004", "non_compliant_article_2_40_categories > 2", "Art. 2(40)"),
    ("MJ-005", "mass_balance_discrepancy_pct > 5", "Art. 10(2)(f)"),
    ("MJ-006", "no_risk_assessment_high_risk_country == True", "Art. 10"),
]

NC_MINOR_RULES = [
    ("MN-001", "admin_documentation_gap_pct < 10", "Art. 31"),
    ("MN-002", "1 <= mass_balance_discrepancy_pct <= 5", "Art. 10(2)(f)"),
    ("MN-003", "scheme_procedural_nc == True", "Scheme"),
    ("MN-004", "training_records_not_current == True", "Art. 2(40) Cat 5"),
    ("MN-005", "single_article_2_40_partial_compliance == True", "Art. 2(40)"),
]

def classify_nc_severity(finding_data: Dict) -> str:
    """Deterministic rule-based classification. Checks CRITICAL first, then MAJOR, then MINOR."""
    for rule_id, condition_key, _ in NC_CRITICAL_RULES:
        if _evaluate_rule(finding_data, condition_key):
            return "critical"
    for rule_id, condition_key, _ in NC_MAJOR_RULES:
        if _evaluate_rule(finding_data, condition_key):
            return "major"
    for rule_id, condition_key, _ in NC_MINOR_RULES:
        if _evaluate_rule(finding_data, condition_key):
            return "minor"
    return "observation"  # Does not match any NC rule
```

**NC Risk Impact Score (Deterministic):**

```python
def calculate_nc_risk_impact(
    severity: str,
    eudr_article_criticality: Decimal,  # 0-100, pre-defined per article
    supply_chain_volume_pct: Decimal,   # % of operator's volume affected
    supplier_risk_level: Decimal,       # 0-100 from EUDR-017
) -> Decimal:
    SEVERITY_WEIGHTS = {
        "critical": Decimal("40"),
        "major": Decimal("25"),
        "minor": Decimal("10"),
        "observation": Decimal("0"),
    }
    score = (
        SEVERITY_WEIGHTS[severity]
        + eudr_article_criticality * Decimal("0.25")
        + supply_chain_volume_pct * Decimal("0.20")
        + supplier_risk_level * Decimal("0.15")
    )
    return min(score, Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
```

**Root Cause Analysis Frameworks:**
- **5 Whys:** Structured 5-level questioning template stored as JSONB
- **Ishikawa (Fishbone):** 6 categories: People, Process, Equipment, Materials, Environment, Management

**Zero-Hallucination Guarantee:** NC classification uses deterministic rule evaluation against pre-coded conditions. No LLM used for severity determination. Risk impact score uses fixed Decimal weights. Rule IDs are recorded in the NC record for full traceability.

**Estimated Lines of Code:** 3,800-4,200

---

### 4.5 Engine 5: CAR Management Engine

**File:** `car_manager.py`
**Purpose:** Manages the full corrective action request lifecycle with SLA enforcement and 4-stage escalation.

**Responsibilities:**
- Issue CARs automatically upon NC classification (severity >= MINOR)
- Manage CAR lifecycle: ISSUED -> ACKNOWLEDGED -> RCA_SUBMITTED -> CAP_SUBMITTED -> CAP_APPROVED -> IN_PROGRESS -> EVIDENCE_SUBMITTED -> VERIFICATION_PENDING -> CLOSED | REJECTED
- Enforce SLA deadlines with real-time countdown: Critical 30 days, Major 90 days, Minor 365 days
- Implement 4-stage escalation for overdue CARs
- Manage CAP review workflow (lead auditor reviews for adequacy)
- Track implementation evidence uploads linked to specific CAR and NC
- Manage effectiveness verification (follow-up audit or desktop review)
- Support CAR grouping (multiple related NCs under single CAR)
- Track CAR metrics per supplier (total, open, average closure time, SLA compliance rate)
- Integrate with EUDR-017 (open CARs feed supplier risk score)
- Support competent authority-issued CARs under Article 18

**SLA Definition Table (Deterministic):**

| NC Severity | CAR Deadline | Acknowledge By | RCA Due | CAP Due | Esc. Stage 1 (75%) | Esc. Stage 2 (90%) | Esc. Stage 3 (100%) |
|-------------|-------------|----------------|---------|---------|--------------------|--------------------|---------------------|
| Critical | 30 days | Day 3 | Day 7 | Day 14 | Day 22 | Day 27 | Day 31 |
| Major | 90 days | Day 7 | Day 14 | Day 30 | Day 67 | Day 81 | Day 91 |
| Minor | 365 days | Day 14 | Day 30 | Day 60 | Day 274 | Day 328 | Day 366 |

**SLA Calculation (Deterministic):**

```python
def calculate_sla_deadline(
    severity: str,
    issued_at: datetime,
    timezone: str = "UTC",
) -> datetime:
    """Deterministic SLA deadline from severity and issuance timestamp."""
    SLA_DAYS = {
        "critical": 30,
        "major": 90,
        "minor": 365,
    }
    return issued_at + timedelta(days=SLA_DAYS[severity])

def calculate_sla_status(
    sla_deadline: datetime,
    current_time: datetime,
) -> str:
    """Deterministic SLA status from deadline and current time."""
    total_duration = (sla_deadline - current_time).total_seconds()
    if total_duration <= 0:
        return "overdue"
    remaining_pct = total_duration / (sla_deadline - issued_at).total_seconds() * 100
    if remaining_pct <= 10:
        return "critical"    # 90%+ elapsed
    if remaining_pct <= 25:
        return "warning"     # 75%+ elapsed
    return "on_track"
```

**Escalation Rules (4 Stages):**

| Stage | Trigger | Action |
|-------|---------|--------|
| 1 | SLA at 75% elapsed | Email to auditee + operator compliance officer |
| 2 | SLA at 90% elapsed | Escalate to audit programme manager + supplier relationship manager |
| 3 | SLA exceeded (100%) | Escalate to Head of Compliance; status -> OVERDUE; supplier risk score increase via EUDR-017 |
| 4 | SLA exceeded by 30+ days (Critical/Major) | Certification suspension recommendation; competent authority notification recommendation |

**Downstream Integration:**
- EUDR-017 Supplier Risk Scorer: open CARs (especially Critical/Major) increase supplier risk score; CAR closure decreases risk score
- Event emission: `car.status_changed`, `car.sla_warning`, `car.escalated` events for cross-agent consumption

**Zero-Hallucination Guarantee:** SLA deadlines are deterministic integer day calculations. Escalation triggers are deterministic percentage thresholds. No LLM in the CAR management path.

**Estimated Lines of Code:** 3,500-4,000

---

### 4.6 Engine 6: Certification Integration Engine

**File:** `scheme_integration.py`
**Purpose:** Bidirectional integration with 5 major certification schemes for audit coordination and redundancy reduction.

**Responsibilities:**
- Integrate with 5 certification scheme databases: FSC, PEFC, RSPO, Rainforest Alliance, ISCC
- Import certificate status: number, status (active/suspended/terminated), scope, expiry, certified products/sites
- Map certification scheme audit requirements to EUDR articles (coverage matrix)
- Identify audit scope overlap across schemes for combined audit recommendations
- Coordinate audit scheduling across schemes to minimize separate site visits
- Import certification audit results with NC classification mapping to unified taxonomy
- Export EUDR compliance evidence to certification bodies
- Monitor certification status changes (daily sync) with alerts on suspensions/withdrawals
- Track mutual recognition agreements between schemes
- Generate certification coverage reports per supplier (EUDR gap identification)
- Map scheme NC taxonomies to unified GreenLang NC taxonomy

**Scheme-to-EUDR Coverage Matrix (Pre-Coded Reference Data):**

| EUDR Requirement | FSC | PEFC | RSPO | RA | ISCC |
|------------------|-----|------|------|-----|------|
| Art. 3 Deforestation-free | FULL | FULL | FULL | FULL | FULL |
| Art. 9 Geolocation | PARTIAL | PARTIAL | FULL | PARTIAL | PARTIAL |
| Art. 10 Risk assessment | FULL | FULL | FULL | FULL | PARTIAL |
| Art. 2(40) Cat 1 Land use | FULL | FULL | FULL | FULL | PARTIAL |
| Art. 2(40) Cat 2 Environment | FULL | FULL | FULL | FULL | FULL |
| Art. 2(40) Cat 3 Forest rules | FULL | FULL | PARTIAL | PARTIAL | NONE |
| Art. 2(40) Cat 4 Third party | FULL | FULL | FULL | FULL | PARTIAL |
| Art. 2(40) Cat 5 Labour | FULL | FULL | FULL | FULL | PARTIAL |
| Art. 2(40) Cat 6 Human rights | FULL | PARTIAL | FULL | FULL | PARTIAL |
| Art. 2(40) Cat 7 FPIC | FULL | PARTIAL | FULL | FULL | NONE |
| Art. 2(40) Cat 8 Tax/customs | FULL | FULL | PARTIAL | PARTIAL | PARTIAL |

Coverage values: FULL = certification audit covers this EUDR requirement; PARTIAL = partially covers; NONE = not covered.

**NC Taxonomy Mapping (Deterministic):**

| Source Scheme | Scheme NC Level | GreenLang Unified Level |
|---------------|-----------------|------------------------|
| FSC | Major | major |
| FSC | Minor | minor |
| FSC | Observation (OFI) | observation |
| PEFC | Major NC | major |
| PEFC | Minor NC | minor |
| PEFC | Observation | observation |
| RSPO | Major NC | major |
| RSPO | Minor NC | minor |
| RSPO | Observation | observation |
| Rainforest Alliance | Critical | critical |
| Rainforest Alliance | Major | major |
| Rainforest Alliance | Minor | minor |
| Rainforest Alliance | Improvement Need | observation |
| ISCC | Major NC | major |
| ISCC | Minor NC | minor |
| ISCC | Observation | observation |

**External API Integration (Adapter Pattern with Circuit Breaker):**

| Scheme | API Endpoint | Rate Limit | Sync Frequency | Fallback |
|--------|-------------|------------|----------------|----------|
| FSC | info.fsc.org/api/v1 | 120 req/min | Daily | Manual CSV import |
| PEFC | pefc.org/search | 60 req/min | Daily | Web scrape adapter |
| RSPO | rspo.org/palmtrace/api | 60 req/min | Daily | Manual CSV import |
| Rainforest Alliance | cert.ra.org/api | 60 req/min | Daily | Manual CSV import |
| ISCC | iscc-system.org/api | 60 req/min | Daily | Manual CSV import |

Each scheme adapter implements the `CertificationSchemeAdapter` interface:
```python
class CertificationSchemeAdapter(Protocol):
    async def fetch_certificate(self, cert_number: str) -> CertificateRecord: ...
    async def sync_certificates(self, supplier_id: str) -> List[CertificateRecord]: ...
    async def check_status(self, cert_number: str) -> str: ...
```

**Zero-Hallucination Guarantee:** Coverage matrix is pre-coded reference data, not LLM-generated. NC taxonomy mapping is deterministic lookup table. Certificate status is imported from authoritative scheme databases.

**Estimated Lines of Code:** 3,500-4,000

---

### 4.7 Engine 7: Audit Reporting Engine

**File:** `report_generator.py`
**Purpose:** Generates ISO 19011:2018 Clause 6.6 compliant audit reports in multiple formats and languages.

**Responsibilities:**
- Generate reports structured per ISO 19011:2018 Clause 6.6 (objectives, scope, criteria, client, team, dates, findings, conclusions)
- Include evidence package with cross-referenced documents, SHA-256 integrity hashes
- Document sampling rationale per ISO 19011 Annex A
- Include auditor credential summary from Engine 2
- Generate finding detail sections with NC severity, EUDR article reference, scheme clause, RCA, corrective action timeline
- Include scheme-specific report sections (FSC-PRO-20-003 format, RSPO format, etc.)
- Support 5 output formats: PDF (primary), JSON (machine-readable), HTML (web), XLSX (data analysis), XML (regulatory)
- Support 5 languages: EN, FR, DE, ES, PT with template-based rendering and locale formatting
- Include report metadata: ID, timestamp, version, generator version, SHA-256 hash, provenance chain
- Support report amendment workflow with change tracking and original preservation

**ISO 19011 Report Structure:**

| Section | Content Source |
|---------|--------------|
| Audit objectives | From audit record (Engine 1 scope) |
| Audit scope | From checklist scope (Engine 3) |
| Audit criteria | EUDR articles + scheme clauses (reference data) |
| Audit client | Operator profile from database |
| Audit team members | From auditor registry (Engine 2) |
| Dates and locations | From fieldwork schedule (Engine 3) |
| Audit findings | From NC engine (Engine 4), categorized by severity |
| Audit conclusions | Auto-generated from finding distribution (deterministic) |
| Statement of confidentiality | Template section |
| Distribution list | From audit management data |

**Report Generation Pipeline:**

```
Audit Data Aggregation -> Template Selection (format + language)
    -> Data Injection (Jinja2) -> Compliance Score Calculation
    -> Visualization Generation (charts) -> Format Rendering
    -> SHA-256 Hash Computation -> S3 Upload -> Provenance Record
```

**Technology:**
- PDF rendering: WeasyPrint + Jinja2 HTML templates
- JSON: Pydantic model serialization
- HTML: Jinja2 templates with CSS styling
- XLSX: openpyxl with template workbooks
- XML: lxml with schema validation

**Multi-Language Support:**
- Template files per language: `templates/{lang}/audit_report.html`
- Locale-specific date formatting (e.g., "10 mars 2026" for FR)
- Locale-specific number formatting (e.g., "1.234,56" for DE)
- Static text translated in template; dynamic data (finding statements) remain in original language

**Zero-Hallucination Guarantee:** All report data sourced from deterministic engine outputs. LLM may be used ONLY for executive summary narrative generation (clearly marked as AI-generated). All numeric values, scores, and compliance determinations are engine-calculated. Report SHA-256 hash enables tamper detection.

**Estimated Lines of Code:** 3,200-3,800

---

## 5. Data Flow Architecture

### 5.1 P0 Feature Data Flows

**F1: Audit Planning Flow**
```
EUDR-016 (country_risk) ----+
EUDR-017 (supplier_risk) ---+---> E1: calculate_priority_score()
EUDR-020 (alert_score) -----+         |
EUDR-001 (supplier_list) ---+         v
                                 audit_schedule hypertable
                                       |
                                       v
                              API: GET /v1/eudr-tam/audits/schedule
```

**F2: Auditor Assignment Flow**
```
E1: audit_requirements ----> E2: calculate_auditor_match_score()
                                       |
                                       v
                              Ranked auditor list
                                       |
                                       v
                              API: POST /v1/eudr-tam/auditors/match
```

**F3: Audit Execution Flow**
```
API: POST /v1/eudr-tam/audits -----> E3: create_audit()
                                            |
API: PUT criterion_result -------> E3: update_checklist()
API: POST evidence upload -------> E3: store_evidence() -> S3
API: GET progress ---------------> E3: calculate_progress()
```

**F4: NC Detection Flow**
```
E3: failed_criteria -----> E4: classify_nc_severity()
EUDR-023 (legal_check) ---+         |
                                    v
                           NC record + rule_ids applied
                                    |
                                    v
                           nc_trend_log hypertable
```

**F5: CAR Lifecycle Flow**
```
E4: nc_classification -----> E5: issue_car()
                                    |
                                    v
                              CAR record (sla_deadline)
                                    |
               +--------------------+--------------------+
               v                    v                    v
         Auditee: ACKNOWLEDGE  E5: monitor_sla()   E5: escalate()
                                    |                    |
                                    v                    v
                           car_sla_log hypertable   EUDR-017 risk update
```

**F6: Scheme Integration Flow**
```
Scheme APIs (5) ----> E6: sync_certificates()
                              |
                              v
                     certification_certificates table
                              |
                              v
                     E6: calculate_coverage_matrix()
                              |
                              v
                     E1: adjust_audit_scope() (reduced for certified criteria)
```

**F7: Report Generation Flow**
```
E3 (checklists) --+
E4 (NCs) ---------+---> E7: generate_report()
E2 (auditors) ----+           |
E5 (CARs) --------+           v
                       Template rendering (Jinja2)
                               |
                               v
                       SHA-256 hash + S3 upload
                               |
                               v
                       audit_reports table
```

**F8: Authority Liaison Flow**
```
Authority request ----> API: POST /v1/eudr-tam/authority/interactions
                              |
                              v
                        authority_interactions table
                              |
                              v
                        SLA countdown (Redis)
                              |
                              v
                        E7: assemble_evidence_package()
                              |
                              v
                        S3: evidence package
```

**F9: Analytics Flow**
```
nc_trend_log hypertable -------+
car_sla_log hypertable --------+---> Continuous Aggregates
audit_schedule hypertable -----+           |
audits table -------------------+          v
                                  API: GET /v1/eudr-tam/analytics/*
```

---

## 6. Database Schema (V112)

### 6.1 Overview

- **Migration:** `V112__agent_eudr_third_party_audit_manager.sql`
- **Schema:** `eudr_third_party_audit`
- **Tables:** 17 (13 regular + 4 TimescaleDB hypertables)
- **Estimated indexes:** ~50
- **Retention:** 5 years (EUDR Article 31)

### 6.2 Regular Tables (13)

**Table 1: `eudr_third_party_audit.audits`** -- Core audit records

| Column | Type | Constraints |
|--------|------|-------------|
| audit_id | UUID | PK DEFAULT gen_random_uuid() |
| operator_id | UUID | NOT NULL |
| supplier_id | UUID | NOT NULL |
| audit_type | VARCHAR(30) | NOT NULL DEFAULT 'full' CHECK IN ('full','targeted','surveillance','unscheduled') |
| modality | VARCHAR(30) | NOT NULL DEFAULT 'on_site' CHECK IN ('on_site','remote','hybrid','unannounced') |
| certification_scheme | VARCHAR(50) | |
| eudr_articles | JSONB | DEFAULT '[]' |
| planned_date | DATE | NOT NULL |
| actual_start_date | DATE | |
| actual_end_date | DATE | |
| lead_auditor_id | UUID | |
| audit_team | JSONB | DEFAULT '[]' |
| status | VARCHAR(30) | NOT NULL DEFAULT 'planned' CHECK IN ('planned','auditor_assigned','in_preparation','in_progress','fieldwork_complete','report_drafting','report_issued','car_follow_up','closed','cancelled') |
| priority_score | NUMERIC(5,2) | DEFAULT 0.0 |
| country_code | CHAR(2) | NOT NULL |
| commodity | VARCHAR(50) | NOT NULL |
| site_ids | JSONB | DEFAULT '[]' |
| checklist_completion | NUMERIC(5,2) | DEFAULT 0.0 |
| findings_count | JSONB | DEFAULT '{"critical":0,"major":0,"minor":0,"observation":0}' |
| evidence_count | INTEGER | DEFAULT 0 |
| report_id | UUID | |
| trigger_reason | VARCHAR(200) | |
| provenance_hash | VARCHAR(64) | NOT NULL |
| metadata | JSONB | DEFAULT '{}' |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** operator_id, supplier_id, status, planned_date, country_code, certification_scheme, (operator_id, status), (supplier_id, planned_date)

---

**Table 2: `eudr_third_party_audit.auditors`** -- Auditor registry

| Column | Type | Constraints |
|--------|------|-------------|
| auditor_id | UUID | PK DEFAULT gen_random_uuid() |
| full_name | VARCHAR(500) | NOT NULL |
| organization | VARCHAR(500) | NOT NULL |
| accreditation_body | VARCHAR(200) | |
| accreditation_status | VARCHAR(30) | DEFAULT 'active' CHECK IN ('active','suspended','withdrawn','expired') |
| accreditation_expiry | DATE | |
| accreditation_scope | JSONB | DEFAULT '[]' |
| commodity_competencies | JSONB | DEFAULT '[]' |
| scheme_qualifications | JSONB | DEFAULT '[]' |
| country_expertise | JSONB | DEFAULT '[]' |
| languages | JSONB | DEFAULT '[]' |
| conflict_of_interest | JSONB | DEFAULT '[]' |
| audit_count | INTEGER | DEFAULT 0 |
| performance_rating | NUMERIC(5,2) | DEFAULT 0.0 |
| cpd_hours | INTEGER | DEFAULT 0 |
| cpd_compliant | BOOLEAN | DEFAULT TRUE |
| contact_email | VARCHAR(500) | |
| metadata | JSONB | DEFAULT '{}' |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** accreditation_status, organization, commodity_competencies (GIN), scheme_qualifications (GIN), country_expertise (GIN)

---

**Table 3: `eudr_third_party_audit.audit_checklists`** -- Audit checklists

| Column | Type | Constraints |
|--------|------|-------------|
| checklist_id | UUID | PK DEFAULT gen_random_uuid() |
| audit_id | UUID | NOT NULL FK -> audits |
| checklist_type | VARCHAR(50) | NOT NULL |
| checklist_version | VARCHAR(20) | NOT NULL |
| criteria | JSONB | NOT NULL DEFAULT '[]' |
| completion_percentage | NUMERIC(5,2) | DEFAULT 0.0 |
| total_criteria | INTEGER | DEFAULT 0 |
| passed_criteria | INTEGER | DEFAULT 0 |
| failed_criteria | INTEGER | DEFAULT 0 |
| na_criteria | INTEGER | DEFAULT 0 |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** audit_id, (audit_id, checklist_type)

---

**Table 4: `eudr_third_party_audit.audit_evidence`** -- Evidence items

| Column | Type | Constraints |
|--------|------|-------------|
| evidence_id | UUID | PK DEFAULT gen_random_uuid() |
| audit_id | UUID | NOT NULL FK -> audits |
| evidence_type | VARCHAR(50) | NOT NULL CHECK IN ('permit','certificate','photo','gps_record','interview_transcript','lab_result','document_scan','other') |
| file_name | VARCHAR(500) | |
| file_path | VARCHAR(1000) | |
| file_size_bytes | BIGINT | |
| mime_type | VARCHAR(100) | |
| description | TEXT | |
| tags | JSONB | DEFAULT '[]' |
| location_latitude | DOUBLE PRECISION | |
| location_longitude | DOUBLE PRECISION | |
| captured_date | TIMESTAMPTZ | |
| sha256_hash | VARCHAR(64) | NOT NULL |
| uploaded_by | VARCHAR(100) | |
| metadata | JSONB | DEFAULT '{}' |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** audit_id, evidence_type, sha256_hash

---

**Table 5: `eudr_third_party_audit.non_conformances`** -- NC records

| Column | Type | Constraints |
|--------|------|-------------|
| nc_id | UUID | PK DEFAULT gen_random_uuid() |
| audit_id | UUID | NOT NULL FK -> audits |
| finding_statement | TEXT | NOT NULL |
| objective_evidence | TEXT | NOT NULL |
| severity | VARCHAR(20) | NOT NULL CHECK IN ('critical','major','minor','observation') |
| eudr_article | VARCHAR(20) | |
| scheme_clause | VARCHAR(100) | |
| article_2_40_category | VARCHAR(50) | |
| root_cause_analysis | JSONB | |
| root_cause_method | VARCHAR(30) | CHECK IN ('five_whys','ishikawa','direct') |
| risk_impact_score | NUMERIC(5,2) | DEFAULT 0.0 |
| status | VARCHAR(30) | NOT NULL DEFAULT 'open' CHECK IN ('open','acknowledged','car_issued','cap_submitted','in_progress','verification_pending','closed','escalated','disputed') |
| car_id | UUID | |
| evidence_ids | JSONB | DEFAULT '[]' |
| classification_rules_applied | JSONB | DEFAULT '[]' |
| disputed | BOOLEAN | DEFAULT FALSE |
| dispute_rationale | TEXT | |
| provenance_hash | VARCHAR(64) | NOT NULL |
| detected_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| resolved_at | TIMESTAMPTZ | |

**Indexes:** audit_id, severity, status, (audit_id, severity), (status, severity), car_id

---

**Table 6: `eudr_third_party_audit.corrective_action_requests`** -- CARs

| Column | Type | Constraints |
|--------|------|-------------|
| car_id | UUID | PK DEFAULT gen_random_uuid() |
| nc_ids | JSONB | NOT NULL DEFAULT '[]' |
| audit_id | UUID | NOT NULL FK -> audits |
| supplier_id | UUID | NOT NULL |
| severity | VARCHAR(20) | NOT NULL CHECK IN ('critical','major','minor') |
| sla_deadline | TIMESTAMPTZ | NOT NULL |
| sla_status | VARCHAR(20) | DEFAULT 'on_track' CHECK IN ('on_track','warning','critical','overdue') |
| status | VARCHAR(30) | NOT NULL DEFAULT 'issued' CHECK IN ('issued','acknowledged','rca_submitted','cap_submitted','cap_approved','in_progress','evidence_submitted','verification_pending','closed','rejected','overdue','escalated') |
| issued_by | VARCHAR(100) | NOT NULL |
| issued_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| acknowledged_at | TIMESTAMPTZ | |
| rca_submitted_at | TIMESTAMPTZ | |
| cap_submitted_at | TIMESTAMPTZ | |
| cap_approved_at | TIMESTAMPTZ | |
| evidence_submitted_at | TIMESTAMPTZ | |
| verified_at | TIMESTAMPTZ | |
| closed_at | TIMESTAMPTZ | |
| corrective_action_plan | JSONB | |
| verification_outcome | VARCHAR(30) | CHECK IN ('effective','not_effective') |
| verification_evidence_ids | JSONB | DEFAULT '[]' |
| escalation_level | INTEGER | DEFAULT 0 CHECK (escalation_level BETWEEN 0 AND 4) |
| escalation_history | JSONB | DEFAULT '[]' |
| provenance_hash | VARCHAR(64) | NOT NULL |
| metadata | JSONB | DEFAULT '{}' |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** audit_id, supplier_id, status, sla_status, severity, (supplier_id, status), (sla_status, severity), (status, sla_deadline)

---

**Table 7: `eudr_third_party_audit.certification_certificates`** -- Scheme certificates

| Column | Type | Constraints |
|--------|------|-------------|
| certificate_id | UUID | PK DEFAULT gen_random_uuid() |
| supplier_id | UUID | NOT NULL |
| scheme | VARCHAR(50) | NOT NULL CHECK IN ('fsc','pefc','rspo','rainforest_alliance','iscc') |
| certificate_number | VARCHAR(200) | NOT NULL |
| status | VARCHAR(30) | NOT NULL DEFAULT 'active' CHECK IN ('active','suspended','terminated','expired') |
| scope | VARCHAR(200) | |
| supply_chain_model | VARCHAR(30) | |
| issue_date | DATE | |
| expiry_date | DATE | |
| certified_products | JSONB | DEFAULT '[]' |
| certified_sites | JSONB | DEFAULT '[]' |
| certification_body | VARCHAR(500) | |
| last_audit_date | DATE | |
| next_audit_date | DATE | |
| eudr_coverage_matrix | JSONB | DEFAULT '{}' |
| metadata | JSONB | DEFAULT '{}' |
| synced_at | TIMESTAMPTZ | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| UNIQUE(scheme, certificate_number) | | |

**Indexes:** supplier_id, scheme, status, expiry_date, (supplier_id, scheme), (scheme, status)

---

**Table 8: `eudr_third_party_audit.audit_reports`** -- Generated reports

| Column | Type | Constraints |
|--------|------|-------------|
| report_id | UUID | PK DEFAULT gen_random_uuid() |
| audit_id | UUID | NOT NULL FK -> audits |
| report_type | VARCHAR(50) | NOT NULL DEFAULT 'iso_19011' |
| report_version | INTEGER | DEFAULT 1 |
| language | VARCHAR(5) | DEFAULT 'en' CHECK IN ('en','fr','de','es','pt') |
| format | VARCHAR(10) | NOT NULL DEFAULT 'pdf' CHECK IN ('pdf','json','html','xlsx','xml') |
| file_path | VARCHAR(1000) | |
| file_size_bytes | BIGINT | |
| sha256_hash | VARCHAR(64) | NOT NULL |
| sections | JSONB | DEFAULT '{}' |
| finding_count | JSONB | DEFAULT '{}' |
| evidence_package_path | VARCHAR(1000) | |
| is_amended | BOOLEAN | DEFAULT FALSE |
| amendment_rationale | TEXT | |
| previous_version_id | UUID | |
| generated_by | VARCHAR(100) | |
| generated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** audit_id, (audit_id, report_type), sha256_hash

---

**Table 9: `eudr_third_party_audit.authority_interactions`** -- Competent authority interactions

| Column | Type | Constraints |
|--------|------|-------------|
| interaction_id | UUID | PK DEFAULT gen_random_uuid() |
| operator_id | UUID | NOT NULL |
| authority_name | VARCHAR(500) | NOT NULL |
| member_state | CHAR(2) | NOT NULL |
| interaction_type | VARCHAR(50) | NOT NULL CHECK IN ('document_request','inspection_notification','unannounced_inspection','corrective_action_order','interim_measure','definitive_measure','information_request') |
| received_date | TIMESTAMPTZ | NOT NULL |
| response_deadline | TIMESTAMPTZ | NOT NULL |
| response_sla_status | VARCHAR(20) | DEFAULT 'on_track' CHECK IN ('on_track','warning','critical','overdue') |
| internal_tasks | JSONB | DEFAULT '[]' |
| evidence_package_id | UUID | |
| response_submitted_at | TIMESTAMPTZ | |
| authority_decision | TEXT | |
| enforcement_measures | JSONB | DEFAULT '[]' |
| status | VARCHAR(30) | NOT NULL DEFAULT 'open' CHECK IN ('open','in_progress','responded','closed') |
| provenance_hash | VARCHAR(64) | NOT NULL |
| metadata | JSONB | DEFAULT '{}' |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** operator_id, status, member_state, (operator_id, status), (status, response_deadline)

---

**Table 10: `eudr_third_party_audit.competent_authorities`** -- Authority reference data

| Column | Type | Constraints |
|--------|------|-------------|
| authority_id | UUID | PK DEFAULT gen_random_uuid() |
| member_state | CHAR(2) | NOT NULL UNIQUE |
| authority_name | VARCHAR(500) | NOT NULL |
| legal_basis | TEXT | |
| inspection_focus | JSONB | DEFAULT '[]' |
| contact_details | JSONB | DEFAULT '{}' |
| default_response_days | INTEGER | DEFAULT 30 |
| active | BOOLEAN | DEFAULT TRUE |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** member_state, active

---

**Table 11: `eudr_third_party_audit.scheme_nc_mappings`** -- NC taxonomy mapping reference

| Column | Type | Constraints |
|--------|------|-------------|
| mapping_id | UUID | PK DEFAULT gen_random_uuid() |
| scheme | VARCHAR(50) | NOT NULL |
| scheme_nc_level | VARCHAR(50) | NOT NULL |
| unified_nc_level | VARCHAR(20) | NOT NULL |
| sla_days | INTEGER | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| UNIQUE(scheme, scheme_nc_level) | | |

**Indexes:** scheme, (scheme, scheme_nc_level)

---

**Table 12: `eudr_third_party_audit.audit_team_assignments`** -- Audit team member assignments

| Column | Type | Constraints |
|--------|------|-------------|
| assignment_id | UUID | PK DEFAULT gen_random_uuid() |
| audit_id | UUID | NOT NULL FK -> audits |
| auditor_id | UUID | NOT NULL FK -> auditors |
| role | VARCHAR(50) | NOT NULL CHECK IN ('lead_auditor','co_auditor','technical_expert','observer','trainee') |
| match_score | NUMERIC(5,2) | |
| assigned_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** audit_id, auditor_id, (audit_id, role)

---

**Table 13: `eudr_third_party_audit.stakeholder_interviews`** -- Interview records

| Column | Type | Constraints |
|--------|------|-------------|
| interview_id | UUID | PK DEFAULT gen_random_uuid() |
| audit_id | UUID | NOT NULL FK -> audits |
| interview_type | VARCHAR(50) | NOT NULL CHECK IN ('community','worker','management','government','other') |
| interviewee_role | VARCHAR(200) | |
| scheduled_date | TIMESTAMPTZ | |
| conducted_date | TIMESTAMPTZ | |
| template_id | VARCHAR(50) | |
| outcome_summary | TEXT | |
| evidence_ids | JSONB | DEFAULT '[]' |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** audit_id, interview_type

---

### 6.3 TimescaleDB Hypertables (4)

**Hypertable 14: `eudr_third_party_audit.audit_schedule`** -- Time-series audit planning

| Column | Type | Constraints |
|--------|------|-------------|
| schedule_id | UUID | DEFAULT gen_random_uuid() |
| operator_id | UUID | NOT NULL |
| supplier_id | UUID | NOT NULL |
| planned_quarter | VARCHAR(7) | NOT NULL |
| audit_type | VARCHAR(30) | NOT NULL |
| modality | VARCHAR(30) | NOT NULL |
| priority_score | NUMERIC(5,2) | |
| risk_factors | JSONB | DEFAULT '{}' |
| assigned_auditor_id | UUID | |
| certification_scheme | VARCHAR(50) | |
| status | VARCHAR(30) | DEFAULT 'planned' |
| linked_audit_id | UUID | |
| scheduled_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (schedule_id, scheduled_at) | | |

- Chunk interval: 90 days
- Retention policy: 5 years (EUDR Article 31)

---

**Hypertable 15: `eudr_third_party_audit.nc_trend_log`** -- NC trend tracking

| Column | Type | Constraints |
|--------|------|-------------|
| log_id | UUID | DEFAULT gen_random_uuid() |
| audit_id | UUID | NOT NULL |
| supplier_id | UUID | NOT NULL |
| nc_id | UUID | NOT NULL |
| severity | VARCHAR(20) | NOT NULL |
| eudr_article | VARCHAR(20) | |
| scheme_clause | VARCHAR(100) | |
| country_code | CHAR(2) | |
| commodity | VARCHAR(50) | |
| root_cause_category | VARCHAR(100) | |
| recorded_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (log_id, recorded_at) | | |

- Chunk interval: 30 days
- Retention policy: 5 years
- Continuous aggregate: `gl_eudr_tam_monthly_nc_summary` (monthly counts by severity, country, commodity)

---

**Hypertable 16: `eudr_third_party_audit.car_sla_log`** -- CAR SLA tracking

| Column | Type | Constraints |
|--------|------|-------------|
| log_id | UUID | DEFAULT gen_random_uuid() |
| car_id | UUID | NOT NULL |
| previous_status | VARCHAR(30) | |
| new_status | VARCHAR(30) | |
| sla_remaining_days | INTEGER | |
| escalation_level | INTEGER | |
| changed_by | VARCHAR(100) | |
| changed_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (log_id, changed_at) | | |

- Chunk interval: 30 days
- Retention policy: 5 years

---

**Hypertable 17: `eudr_third_party_audit.audit_trail`** -- Immutable audit log

| Column | Type | Constraints |
|--------|------|-------------|
| trail_id | UUID | DEFAULT gen_random_uuid() |
| entity_type | VARCHAR(50) | NOT NULL |
| entity_id | UUID | NOT NULL |
| action | VARCHAR(50) | NOT NULL |
| before_value | JSONB | |
| after_value | JSONB | |
| actor | VARCHAR(100) | NOT NULL |
| ip_address | VARCHAR(45) | |
| recorded_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (trail_id, recorded_at) | | |

- Chunk interval: 30 days
- Retention policy: 5 years (EUDR Article 31)

### 6.4 Continuous Aggregates (2)

**1. `gl_eudr_tam_monthly_nc_summary`**
```sql
SELECT
    time_bucket('1 month', recorded_at) AS month,
    country_code,
    commodity,
    severity,
    COUNT(*) AS nc_count
FROM eudr_third_party_audit.nc_trend_log
GROUP BY 1, 2, 3, 4;
```

**2. `gl_eudr_tam_monthly_car_performance`**
```sql
SELECT
    time_bucket('1 month', changed_at) AS month,
    new_status,
    COUNT(*) AS transition_count,
    AVG(sla_remaining_days) AS avg_sla_remaining
FROM eudr_third_party_audit.car_sla_log
GROUP BY 1, 2;
```

---

## 7. API Architecture (~35 Endpoints)

**API Prefix:** `/v1/eudr-tam`

### 7.1 Audit Planning Routes (`planning_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/audits/schedule/generate` | `eudr-tam:audits:schedule` | Generate risk-based audit schedule for operator |
| GET | `/audits/schedule` | `eudr-tam:audits:read` | Get audit calendar (filters: date range, supplier, scheme) |
| PUT | `/audits/schedule/{schedule_id}` | `eudr-tam:audits:schedule` | Update scheduled audit (reschedule, reassign) |
| POST | `/audits/schedule/trigger` | `eudr-tam:audits:schedule` | Trigger unscheduled audit (event-based) |

### 7.2 Audit Management Routes (in `planning_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/audits` | `eudr-tam:audits:write` | Create a new audit |
| GET | `/audits` | `eudr-tam:audits:read` | List audits (filters: status, supplier, scheme, date) |
| GET | `/audits/{audit_id}` | `eudr-tam:audits:read` | Get audit details with full status |
| PUT | `/audits/{audit_id}` | `eudr-tam:audits:write` | Update audit (status, team, dates) |
| DELETE | `/audits/{audit_id}` | `eudr-tam:audits:write` | Cancel audit |

### 7.3 Auditor Registry Routes (`auditor_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/auditors` | `eudr-tam:auditors:write` | Register a new auditor |
| GET | `/auditors` | `eudr-tam:auditors:read` | List auditors (filters: scheme, commodity, country, status) |
| GET | `/auditors/{auditor_id}` | `eudr-tam:auditors:read` | Get auditor profile with performance |
| PUT | `/auditors/{auditor_id}` | `eudr-tam:auditors:write` | Update auditor profile |
| POST | `/auditors/match` | `eudr-tam:auditors:read` | Match auditors to audit requirements |

### 7.4 Audit Execution Routes (`execution_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET | `/audits/{audit_id}/checklist` | `eudr-tam:execution:read` | Get audit checklist with completion |
| PUT | `/audits/{audit_id}/checklist/{criterion_id}` | `eudr-tam:execution:write` | Update checklist criterion result |
| POST | `/audits/{audit_id}/evidence` | `eudr-tam:execution:write` | Upload audit evidence |
| GET | `/audits/{audit_id}/evidence` | `eudr-tam:execution:read` | List audit evidence items |
| GET | `/audits/{audit_id}/progress` | `eudr-tam:execution:read` | Get real-time audit progress |

### 7.5 Non-Conformance Routes (`nc_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/audits/{audit_id}/ncs` | `eudr-tam:ncs:write` | Create non-conformance finding |
| GET | `/audits/{audit_id}/ncs` | `eudr-tam:ncs:read` | List NCs for an audit |
| GET | `/ncs/{nc_id}` | `eudr-tam:ncs:read` | Get NC details with evidence and RCA |
| PUT | `/ncs/{nc_id}` | `eudr-tam:ncs:write` | Update NC (status, RCA, dispute) |
| POST | `/ncs/{nc_id}/rca` | `eudr-tam:ncs:write` | Submit root cause analysis |

### 7.6 CAR Lifecycle Routes (`car_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/cars` | `eudr-tam:cars:write` | Issue a new CAR |
| GET | `/cars` | `eudr-tam:cars:read` | List CARs (filters: status, severity, SLA, supplier) |
| GET | `/cars/{car_id}` | `eudr-tam:cars:read` | Get CAR details with full lifecycle |
| PUT | `/cars/{car_id}` | `eudr-tam:cars:respond` | Update CAR status (acknowledge, submit CAP, evidence) |
| POST | `/cars/{car_id}/verify` | `eudr-tam:cars:verify` | Submit verification outcome |

### 7.7 Certification Scheme Routes (`scheme_routes.py`) -- 3 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET | `/schemes/certificates` | `eudr-tam:schemes:read` | List certification certificates |
| POST | `/schemes/certificates/sync` | `eudr-tam:schemes:sync` | Trigger certification status sync |
| GET | `/schemes/coverage/{supplier_id}` | `eudr-tam:schemes:read` | Get EUDR coverage matrix for supplier |

### 7.8 Report Routes (`report_routes.py`) -- 2 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/audits/{audit_id}/report` | `eudr-tam:reports:generate` | Generate audit report |
| GET | `/reports/{report_id}` | `eudr-tam:reports:read` | Download audit report |

### 7.9 Authority Liaison Routes (`authority_routes.py`) -- 3 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/authority/interactions` | `eudr-tam:authority:write` | Log new authority interaction |
| GET | `/authority/interactions` | `eudr-tam:authority:read` | List authority interactions |
| PUT | `/authority/interactions/{id}` | `eudr-tam:authority:write` | Update interaction (response, status) |

### 7.10 Analytics Routes (`analytics_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET | `/analytics/findings` | `eudr-tam:analytics:read` | Finding trend analytics |
| GET | `/analytics/auditor-performance` | `eudr-tam:analytics:read` | Auditor benchmarking data |
| GET | `/analytics/compliance-rates` | `eudr-tam:analytics:read` | Compliance rate trends |
| GET | `/analytics/car-performance` | `eudr-tam:analytics:read` | CAR lifecycle analytics |
| GET | `/analytics/dashboard` | `eudr-tam:analytics:read` | Executive dashboard data |

### 7.11 Admin Routes (`admin_routes.py`) -- 2 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET | `/health` | (public) | Health check |
| GET | `/stats` | `eudr-tam:analytics:read` | Service statistics |

**Total: 39 endpoints**

---

## 8. Integration Architecture

### 8.1 Upstream Dependencies (7 EUDR Agents)

| Agent | Integration | Data Consumed |
|-------|-------------|---------------|
| EUDR-001 Supply Chain Mapping Master | REST API | Supplier inventory, site locations, supply chain complexity metrics |
| EUDR-016 Country Risk Evaluator | REST API | Country risk level (Low/Standard/High), risk_score (0-100), audit_frequency_multiplier |
| EUDR-017 Supplier Risk Scorer | REST API (bidirectional) | Composite supplier risk (0-100); TAM sends open CAR counts back |
| EUDR-020 Deforestation Alert System | REST API + Events | Alert proximity to supplier plots, severity, audit trigger recommendation |
| EUDR-021 Indigenous Rights Checker | REST API | FPIC verification status for audit checklist FPIC criteria |
| EUDR-022 Protected Area Validator | REST API | Protected area overlap status for audit checklist PA criteria |
| EUDR-023 Legal Compliance Verifier | REST API | Legal compliance assessment for Article 2(40) audit checklist criteria |

### 8.2 Downstream Consumers

| Consumer | Data Provided |
|----------|---------------|
| GL-EUDR-APP v1.0 | Audit data, NC status, CAR tracking, analytics for frontend dashboards |
| EUDR-017 Supplier Risk Scorer | Open CARs, NC history, audit outcomes for risk score adjustment |
| EUDR-001 Supply Chain Mapping Master | Audit status, compliance rating for graph node enrichment |
| EUDR-023 Legal Compliance Verifier | Legal compliance NCs and document verification findings |

### 8.3 API Contracts

**Provided to EUDR-017 (Supplier Risk Scorer):**
```python
async def get_supplier_audit_risk(supplier_id: str) -> Dict:
    """Returns audit compliance risk data for supplier risk scoring."""
    return {
        "supplier_id": str,
        "total_audits": int,
        "last_audit_date": str,              # ISO 8601
        "days_since_last_audit": int,
        "open_critical_ncs": int,
        "open_major_ncs": int,
        "open_minor_ncs": int,
        "open_cars": int,
        "overdue_cars": int,
        "car_sla_compliance_rate": Decimal,  # 0-100
        "repeat_nc_rate": Decimal,           # 0-100
        "certification_status": Dict,        # {scheme: status}
        "audit_compliance_risk_score": Decimal,  # 0-100
        "provenance_hash": str,
    }
```

**Consumed from EUDR-016 (Country Risk Evaluator):**
```python
async def get_country_risk_for_audit(country_code: str) -> Dict:
    """Returns country risk data for audit scheduling."""
    return {
        "country_code": str,
        "risk_level": str,                   # "low" | "standard" | "high"
        "risk_score": Decimal,               # 0-100
        "audit_frequency_multiplier": Decimal,  # 1.0 / 1.5 / 2.0
        "provenance_hash": str,
    }
```

**Consumed from EUDR-020 (Deforestation Alert System):**
```python
async def get_deforestation_alerts_for_supplier(supplier_id: str) -> Dict:
    """Returns deforestation alerts for unscheduled audit triggers."""
    return {
        "supplier_id": str,
        "active_alerts": List[Dict],         # severity, proximity_km, detected_date
        "highest_severity": str,
        "alert_risk_score": Decimal,         # 0-100
        "audit_trigger_recommended": bool,
        "provenance_hash": str,
    }
```

### 8.4 External Integrations (5 Certification Scheme APIs)

| System | Protocol | Rate Limit | Auth | Fallback |
|--------|----------|------------|------|----------|
| FSC Certificate Database | REST/JSON | 120 req/min | API Key | CSV import |
| PEFC Certificate Search | REST/JSON | 60 req/min | API Key | Web adapter |
| RSPO PalmTrace | REST/JSON | 60 req/min | OAuth2 | CSV import |
| Rainforest Alliance Portal | REST/JSON | 60 req/min | API Key | CSV import |
| ISCC Certification DB | REST/JSON | 60 req/min | API Key | CSV import |

### 8.5 Infrastructure Integrations

| Component | Usage |
|-----------|-------|
| PostgreSQL 14+ (TimescaleDB) | Primary data store (17 tables, 4 hypertables) |
| Redis 7+ | SLA countdown caching, audit status caching, analytics query caching |
| S3 (AWS/MinIO) | Evidence files, audit reports, evidence packages |
| SEC-001 JWT Auth | JWT RS256 token validation on all protected endpoints |
| SEC-002 RBAC | 22 permissions with `eudr-tam:` prefix |
| SEC-003 Encryption | AES-256-GCM for auditor PII, evidence, authority communications |
| OBS-001 Prometheus | 20 metrics (10 counters, 5 histograms, 5 gauges) |
| OBS-003 OpenTelemetry | Distributed tracing across engine and agent calls |

---

## 9. Security Architecture

### 9.1 Authentication

- All endpoints (except `/health`) require JWT authentication via SEC-001
- JWT RS256 token validation with configurable JWKS endpoint
- API key support for machine-to-machine integrations (scheme sync, inter-agent calls)
- Token expiry: 1 hour (configurable via `GL_EUDR_TAM_TOKEN_EXPIRY_S`)

### 9.2 Authorization (RBAC -- 22 Permissions)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-tam:audits:read` | View audit records | Viewer, Analyst, Auditor, Compliance Officer, Admin |
| `eudr-tam:audits:write` | Create, update, cancel audits | Compliance Officer, Audit Programme Manager, Admin |
| `eudr-tam:audits:schedule` | Generate and modify audit schedules | Audit Programme Manager, Compliance Officer, Admin |
| `eudr-tam:auditors:read` | View auditor registry | Viewer, Analyst, Compliance Officer, Audit Programme Manager, Admin |
| `eudr-tam:auditors:write` | Register, update auditor profiles | Audit Programme Manager, Admin |
| `eudr-tam:execution:read` | View checklists and evidence | Viewer, Auditor, Compliance Officer, Admin |
| `eudr-tam:execution:write` | Update checklists, upload evidence | Auditor, Compliance Officer, Admin |
| `eudr-tam:ncs:read` | View non-conformances | Viewer, Analyst, Auditor, Compliance Officer, Admin |
| `eudr-tam:ncs:write` | Create, classify, update NCs | Auditor, Compliance Officer, Admin |
| `eudr-tam:ncs:dispute` | Dispute NC classification | Supplier, Compliance Officer, Admin |
| `eudr-tam:cars:read` | View CARs | Viewer, Analyst, Auditor, Supplier, Compliance Officer, Admin |
| `eudr-tam:cars:write` | Issue, update CARs | Auditor, Compliance Officer, Admin |
| `eudr-tam:cars:respond` | Acknowledge, submit CAP, evidence | Supplier, Compliance Officer |
| `eudr-tam:cars:verify` | Verify corrective action effectiveness | Auditor, Compliance Officer, Admin |
| `eudr-tam:cars:close` | Close verified CARs | Compliance Officer, Admin |
| `eudr-tam:schemes:read` | View certification scheme data | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-tam:schemes:sync` | Trigger certification status sync | Compliance Officer, Admin |
| `eudr-tam:reports:read` | View and download audit reports | Viewer, Auditor, Compliance Officer, Admin |
| `eudr-tam:reports:generate` | Generate audit reports | Auditor, Compliance Officer, Admin |
| `eudr-tam:authority:read` | View authority interactions | Compliance Officer, Legal Officer, Admin |
| `eudr-tam:authority:write` | Log and manage authority interactions | Compliance Officer, Legal Officer, Admin |
| `eudr-tam:analytics:read` | View analytics and dashboards | Analyst, Compliance Officer, Audit Programme Manager, Admin |

### 9.3 Data Privacy and GDPR

| Data Type | Classification | Protection |
|-----------|---------------|------------|
| Auditor personal data | PII | AES-256 encryption; GDPR consent; right to erasure |
| Supplier contact info | PII | AES-256 encryption; data minimization; 5-year retention |
| Audit evidence | Confidential | AES-256 at rest; RBAC access control |
| Authority communications | Highly Confidential | AES-256; restricted to Legal Officer + Admin |
| Audit findings and NCs | Confidential | RBAC access control; 5-year retention |
| Analytics (aggregated) | Internal | Role-based access; anonymized where possible |

### 9.4 Rate Limiting

- Default: 100 requests/minute per tenant
- Evidence upload: 20 requests/minute per tenant
- Report generation: 10 requests/minute per tenant
- Admin endpoints: 20 requests/minute per tenant
- Configurable via `GL_EUDR_TAM_RATE_LIMIT_*` environment variables

---

## 10. Observability Architecture

### 10.1 Prometheus Metrics (20)

**Prefix:** `gl_eudr_tam_`

#### Counters (10)

| # | Metric | Labels | Description |
|---|--------|--------|-------------|
| 1 | `gl_eudr_tam_audits_created_total` | type, scheme | Audits created by type and scheme |
| 2 | `gl_eudr_tam_audits_completed_total` | outcome | Audits completed by outcome |
| 3 | `gl_eudr_tam_audits_cancelled_total` | reason | Audits cancelled |
| 4 | `gl_eudr_tam_ncs_detected_total` | severity, country | NCs detected by severity and country |
| 5 | `gl_eudr_tam_ncs_resolved_total` | severity | NCs resolved by severity |
| 6 | `gl_eudr_tam_cars_issued_total` | severity | CARs issued by severity |
| 7 | `gl_eudr_tam_cars_closed_total` | severity | CARs closed by severity |
| 8 | `gl_eudr_tam_cars_overdue_total` | severity | CARs that exceeded SLA |
| 9 | `gl_eudr_tam_cars_escalated_total` | level | CARs escalated by escalation level |
| 10 | `gl_eudr_tam_reports_generated_total` | format, language | Audit reports generated |

#### Histograms (5)

| # | Metric | Buckets | Description |
|---|--------|---------|-------------|
| 11 | `gl_eudr_tam_scheduling_duration_seconds` | 0.1, 0.5, 1, 2, 5, 10 | Audit schedule generation latency |
| 12 | `gl_eudr_tam_nc_classification_duration_seconds` | 0.05, 0.1, 0.25, 0.5, 1 | NC classification latency |
| 13 | `gl_eudr_tam_report_generation_duration_seconds` | 1, 5, 10, 20, 30, 60 | Report generation latency |
| 14 | `gl_eudr_tam_api_request_duration_seconds` | 0.01, 0.05, 0.1, 0.2, 0.5, 1 | API request latency by endpoint |
| 15 | `gl_eudr_tam_authority_interactions_total` | type, member_state | Authority interactions by type and state |

#### Gauges (5)

| # | Metric | Description |
|---|--------|-------------|
| 16 | `gl_eudr_tam_active_audits` | Currently active audits |
| 17 | `gl_eudr_tam_open_cars` | Currently open CARs by severity |
| 18 | `gl_eudr_tam_car_sla_compliance_rate` | CAR SLA compliance percentage (0-100) |
| 19 | `gl_eudr_tam_scheme_sync_freshness_seconds` | Time since last scheme sync |
| 20 | `gl_eudr_tam_cache_hit_ratio` | Redis cache hit ratio (0.0-1.0) |

### 10.2 Grafana Dashboard

**File:** `deployment/monitoring/dashboards/eudr-third-party-audit-manager.json`

**Dashboard Panels (20):**

| Row | Panel | Metric Source |
|-----|-------|---------------|
| 1 | Active Audits (stat) | `gl_eudr_tam_active_audits` |
| 1 | Open CARs by Severity (stat) | `gl_eudr_tam_open_cars` |
| 1 | CAR SLA Compliance Rate (gauge) | `gl_eudr_tam_car_sla_compliance_rate` |
| 1 | Cache Hit Ratio (gauge) | `gl_eudr_tam_cache_hit_ratio` |
| 2 | Audits Created Over Time (time series) | `rate(gl_eudr_tam_audits_created_total)` |
| 2 | Audits Completed Over Time (time series) | `rate(gl_eudr_tam_audits_completed_total)` |
| 3 | NCs Detected by Severity (stacked bar) | `rate(gl_eudr_tam_ncs_detected_total)` |
| 3 | NCs Resolved Over Time (time series) | `rate(gl_eudr_tam_ncs_resolved_total)` |
| 4 | CARs Issued vs Closed (bar) | `rate(gl_eudr_tam_cars_issued_total)` / `rate(gl_eudr_tam_cars_closed_total)` |
| 4 | CARs Overdue (time series) | `rate(gl_eudr_tam_cars_overdue_total)` |
| 4 | CAR Escalations by Level (stacked bar) | `rate(gl_eudr_tam_cars_escalated_total)` |
| 5 | Schedule Generation Latency (heatmap) | `gl_eudr_tam_scheduling_duration_seconds` |
| 5 | NC Classification Latency (heatmap) | `gl_eudr_tam_nc_classification_duration_seconds` |
| 5 | Report Generation Latency (heatmap) | `gl_eudr_tam_report_generation_duration_seconds` |
| 6 | API Request Latency p50/p95/p99 (time series) | `gl_eudr_tam_api_request_duration_seconds` |
| 6 | API Errors (time series) | Derived from 4xx/5xx in API histogram |
| 7 | Reports Generated by Format (pie) | `gl_eudr_tam_reports_generated_total` |
| 7 | Authority Interactions by Type (bar) | `gl_eudr_tam_authority_interactions_total` |
| 8 | Scheme Sync Freshness (time series) | `gl_eudr_tam_scheme_sync_freshness_seconds` |
| 8 | Audits Cancelled (time series) | `rate(gl_eudr_tam_audits_cancelled_total)` |

### 10.3 OpenTelemetry Tracing

- Span creation for each engine call (7 engines)
- Cross-agent trace propagation for EUDR-016/017/020 API calls
- Trace attributes: `audit_id`, `supplier_id`, `operator_id`, `engine_name`
- Sampling rate: 10% in production, 100% in staging

---

## 11. Technology Stack

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| Language | Python | 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | 0.104.0+ | Async, OpenAPI, Pydantic v2 native |
| ASGI Server | Uvicorn | 0.24.0+ | High-performance async server |
| Data Validation | Pydantic | 2.5.0+ | Type-safe models with JSON serialization |
| Database | PostgreSQL | 14+ | Primary persistent store |
| Time-Series | TimescaleDB | 2.13+ | Hypertables for audit trail, NC trends, CAR SLA logs |
| Cache | Redis | 7+ | SLA countdown, status caching, analytics caching |
| Object Storage | S3 / MinIO | - | Evidence files, audit reports |
| PDF Generation | WeasyPrint | 60+ | ISO 19011 report rendering from HTML/CSS |
| Templating | Jinja2 | 3.1+ | Report templates (5 languages) |
| Excel Output | openpyxl | 3.1+ | XLSX report format |
| XML Processing | lxml | 4.9+ | XML report format with schema validation |
| Numeric | Decimal (stdlib) | - | Zero-hallucination Decimal arithmetic |
| Data Processing | Pandas | 2.1+ | Analytics aggregation and trend analysis |
| Authentication | JWT (RS256) | via SEC-001 | python-jose 3.3+ |
| Authorization | RBAC | via SEC-002 | 22 permissions |
| Encryption | AES-256-GCM | via SEC-003 | cryptography 41+ |
| Monitoring | Prometheus | via OBS-001 | prometheus_client 0.19+ |
| Tracing | OpenTelemetry | via OBS-003 | opentelemetry-sdk 1.21+ |
| Hashing | hashlib (stdlib) | - | SHA-256 provenance chains |
| HTTP Client | httpx | 0.25+ | Async HTTP for scheme API integrations |
| CI/CD | GitHub Actions | via INFRA-007 | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | via INFRA-001 | Standard GreenLang deployment |
| Container | Docker | - | python:3.11-slim base image |

---

## 12. Zero-Hallucination Architecture

### 12.1 Principles

| Principle | Implementation |
|-----------|---------------|
| Deterministic calculations | Decimal arithmetic for all scores; fixed weights; quantized rounding |
| No LLM in critical path | Scheduling, NC classification, SLA calculation, compliance scoring all rule-based |
| Authoritative data sources | Certification data from scheme registries; authority profiles from EU designations |
| Full provenance tracking | SHA-256 hash on every audit finding, NC classification, CAR status, report |
| Immutable audit trail | `audit_trail` hypertable records all changes with before/after values |
| Version-controlled criteria | EUDR checklists and scheme criteria versioned with effective dates |
| Bit-perfect reproducibility | Same inputs produce same outputs across Python 3.11/3.12 |

### 12.2 LLM Usage Boundaries

| Permitted (Non-Critical Path) | Prohibited (Critical Path) |
|-------------------------------|---------------------------|
| Executive summary narrative in reports | NC severity classification |
| Audit finding description enrichment | Risk-based audit scheduling |
| Root cause suggestion prompts (UI hints) | CAR SLA deadline calculation |
| Analytics narrative commentary | Compliance score computation |
| | Auditor match scoring |
| | Coverage matrix determination |

### 12.3 Provenance Chain

```
Audit Provenance Chain:
    hash_0 = SHA-256(risk_inputs || scheduling_parameters)
    hash_1 = SHA-256(hash_0 || auditor_assignment)
    hash_2 = SHA-256(hash_1 || checklist_results || evidence_hashes)
    hash_3 = SHA-256(hash_2 || nc_classifications || rules_applied)
    hash_4 = SHA-256(hash_3 || car_lifecycle_events)
    hash_5 = SHA-256(hash_4 || report_content)
    chain_hash = SHA-256(hash_5 || final_audit_status)

    The chain_hash stored on the audit record enables bit-perfect
    verification that all inputs produced the stated audit outcome.
```

---

## 13. Configuration

### 13.1 Environment Variables (`GL_EUDR_TAM_` prefix)

```yaml
# Database
GL_EUDR_TAM_DATABASE_URL: postgresql+asyncpg://...
GL_EUDR_TAM_POOL_SIZE: 10
GL_EUDR_TAM_POOL_TIMEOUT_S: 30
GL_EUDR_TAM_POOL_RECYCLE_S: 3600

# Redis
GL_EUDR_TAM_REDIS_URL: redis://...
GL_EUDR_TAM_REDIS_TTL_S: 86400
GL_EUDR_TAM_REDIS_KEY_PREFIX: tam

# S3
GL_EUDR_TAM_S3_EVIDENCE_BUCKET: gl-eudr-tam-evidence
GL_EUDR_TAM_S3_REPORTS_BUCKET: gl-eudr-tam-reports
GL_EUDR_TAM_S3_REGION: eu-west-1

# Risk weights (configurable per operator)
GL_EUDR_TAM_WEIGHT_COUNTRY_RISK: 0.25
GL_EUDR_TAM_WEIGHT_SUPPLIER_RISK: 0.25
GL_EUDR_TAM_WEIGHT_NC_HISTORY: 0.20
GL_EUDR_TAM_WEIGHT_CERT_GAP: 0.15
GL_EUDR_TAM_WEIGHT_DEFORESTATION: 0.15

# Frequency thresholds
GL_EUDR_TAM_HIGH_RISK_THRESHOLD: 70
GL_EUDR_TAM_STANDARD_RISK_THRESHOLD: 40

# SLA deadlines (days)
GL_EUDR_TAM_SLA_CRITICAL_DAYS: 30
GL_EUDR_TAM_SLA_MAJOR_DAYS: 90
GL_EUDR_TAM_SLA_MINOR_DAYS: 365

# Escalation thresholds (percentage of SLA elapsed)
GL_EUDR_TAM_ESCALATION_STAGE_1_PCT: 75
GL_EUDR_TAM_ESCALATION_STAGE_2_PCT: 90

# Scheme sync
GL_EUDR_TAM_SCHEME_SYNC_INTERVAL_HOURS: 24
GL_EUDR_TAM_FSC_API_URL: https://info.fsc.org/api/v1
GL_EUDR_TAM_RSPO_API_URL: https://rspo.org/api/v1
GL_EUDR_TAM_PEFC_API_URL: https://pefc.org/api/v1
GL_EUDR_TAM_RA_API_URL: https://cert.ra.org/api
GL_EUDR_TAM_ISCC_API_URL: https://iscc-system.org/api

# Rate limiting
GL_EUDR_TAM_RATE_LIMIT_DEFAULT: 100
GL_EUDR_TAM_RATE_LIMIT_UPLOAD: 20
GL_EUDR_TAM_RATE_LIMIT_REPORT: 10

# Provenance
GL_EUDR_TAM_PROVENANCE_ENABLED: true
GL_EUDR_TAM_RETENTION_YEARS: 5

# Logging
GL_EUDR_TAM_LOG_LEVEL: INFO
```

---

## 14. Deployment Architecture

### 14.1 Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY greenlang/ greenlang/
EXPOSE 8024
CMD ["uvicorn", "greenlang.agents.eudr.third_party_audit.api.router:app",
     "--host", "0.0.0.0", "--port", "8024"]
```

### 14.2 Kubernetes

- Deployment: 2-8 replicas (HPA on CPU at 70%)
- Service: ClusterIP on port 8024
- Health probes: `/v1/eudr-tam/health` (liveness + readiness)
- Resource limits: 512Mi memory, 500m CPU per pod
- Resource requests: 256Mi memory, 250m CPU per pod
- ConfigMap: non-secret `GL_EUDR_TAM_*` configuration
- Secret: database URLs, API keys (sourced from Vault via SEC-006)

### 14.3 Caching Strategy (Redis)

| Cache Layer | Key Pattern | TTL | Purpose |
|-------------|-------------|-----|---------|
| Audit status | `tam:audit:{audit_id}:status` | 5 minutes | Frequent status polling |
| CAR SLA status | `tam:car:{car_id}:sla` | 1 minute | Real-time SLA countdown |
| Auditor pool | `tam:auditors:{scheme}:{commodity}` | 1 hour | Auditor matching |
| Certification status | `tam:cert:{scheme}:{cert_number}` | 1 hour | Scheme cert caching |
| Analytics results | `tam:analytics:{query_hash}` | 15 minutes | Dashboard query caching |
| Authority profiles | `tam:authority:{member_state}` | 24 hours | Static reference data |
| Coverage matrix | `tam:coverage:{supplier_id}` | 1 hour | Scheme-EUDR mapping |

**Expected cache hit rate:** >65%

---

## 15. Testing Strategy

### 15.1 Coverage Targets

| Category | Tests | Description |
|----------|-------|-------------|
| Audit Planning | 80+ | Risk scoring, frequency, triggers, multi-scheme, calendar |
| Auditor Registry | 50+ | Profile CRUD, matching, CoI, accreditation, rotation |
| Audit Execution | 70+ | Checklists, evidence, progress, sampling, modality |
| NC Classification | 80+ | All rules, EUDR mapping, RCA, patterns, disputes |
| CAR Lifecycle | 80+ | All status transitions, SLA, escalation, closure, grouping |
| Scheme Integration | 60+ | 5 schemes, coverage matrix, NC mapping, sync |
| Report Generation | 50+ | ISO 19011, 5 formats, 5 languages, evidence, amendments |
| Authority Liaison | 40+ | 27 states, interaction types, SLA, response, evidence |
| Analytics | 50+ | Trends, benchmarking, rates, CAR perf, cost, dashboard |
| API Routes | 50+ | All 39 endpoints, auth, errors, pagination |
| Golden Scenarios | 50 | 7 commodities x 7 scenarios + 1 multi-scheme |
| Integration | 40+ | Cross-agent flows (7 agents), event bus, contracts |
| Performance | 25+ | Schedule gen, NC classify, report gen, analytics |
| Determinism | 15+ | Bit-perfect reproducibility (100x runs, SHA-256 match) |
| Security | 10+ | PII encryption, RBAC enforcement, evidence access |
| **Total** | **800+** | |

### 15.2 Golden Test Scenarios (50)

Each of the 7 EUDR commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood) is tested against 7 audit scenarios:

1. **Clean audit** -- Full scope, zero NCs -> CLOSED with clean report, no CARs
2. **Critical NC** -- Post-2020 deforestation evidence -> CRITICAL classification, 30-day CAR, immediate escalation
3. **Major NC with closure** -- Incomplete CoC documentation -> MAJOR classification, 90-day CAR, verified closure
4. **Multiple minor NCs** -- 5 admin gaps grouped under 1 CAR -> 5 MINOR NCs, grouped CAR, 365-day SLA
5. **Multi-scheme audit** -- FSC + RA supplier -> overlap detection, reduced criteria, dual taxonomy mapping
6. **Authority inspection** -- NVWA document request -> interaction logged, 30-day SLA, evidence package generated
7. **Overdue CAR escalation** -- Major CAR exceeds 90 days -> 4-stage escalation, risk score increase

Total: 7 x 7 = 49 + 1 multi-scheme cross-commodity = 50 golden tests.

### 15.3 Determinism Tests

1. Run audit priority calculation 100 times with identical inputs -- verify SHA-256 match
2. Run NC classification 100 times with identical finding data -- verify identical severity
3. Run SLA calculation 100 times with identical timestamps -- verify identical deadlines
4. Run auditor matching 100 times with identical pool -- verify identical ranking
5. Verify across Python 3.11 and 3.12 for platform independence
6. Verify Decimal arithmetic produces identical results to reference calculations

---

## 16. Risks and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | Certification scheme APIs change or become unavailable | Medium | Medium | Adapter pattern; local cache; manual CSV import fallback; circuit breaker |
| 2 | NC classification inconsistent with expert judgment | Medium | High | 500-case calibration corpus; 95% accuracy target; operator feedback loop |
| 3 | CAR SLA enforcement perceived as too rigid | Medium | Medium | Configurable SLA deadlines; grace period option; escalation warnings |
| 4 | Authority workflows vary across 27 Member States | High | Medium | Start with top 6 states (DE, FR, NL, BE, IT, ES); generic template for others |
| 5 | Cross-scheme coordination resisted by certification bodies | Medium | Medium | Focus on operator-side coordination; maintain scheme-specific reports |
| 6 | Audit evidence volumes overwhelm storage | Low | Medium | S3 lifecycle policies; compression; 5 TB capacity plan |
| 7 | Integration complexity with 7 upstream agents | Medium | Medium | Well-defined interfaces; mock adapters; circuit breaker; health monitoring |
| 8 | Auditor PII creates GDPR compliance risk | Medium | High | AES-256 encryption; GDPR consent; right to erasure; DPO review |
| 9 | ISO 19011 report format changes | Low | Low | Template-based generation; version-controlled templates |
| 10 | Low auditor adoption of digital checklists | Medium | Medium | Intuitive UX; offline capability; mobile-responsive; pilot programme |

---

## 17. Appendices

### A. EUDR Articles Covered

Articles 4(2), 9(1), 10(1), 10(2)(a-f), 11(1), 14(1), 14(4), 15, 16, 18, 19-20, 21, 22, 23, 29, 31.

### B. ISO Standards Addressed

- ISO 19011:2018 -- Auditing management systems (report structure, auditor competence, audit programme management)
- ISO/IEC 17065:2012 -- Certification body requirements (CB accreditation validation)
- ISO/IEC 17021-1:2015 -- Management system certification body requirements (auditor competence, impartiality)
- ISO/IEC 17011:2017 -- Accreditation body requirements (accreditation status tracking)

### C. Certification Scheme Recertification Timelines

| Scheme | Surveillance | Recertification | NC Major Deadline | NC Minor Deadline |
|--------|-------------|-----------------|-------------------|-------------------|
| FSC | Annual | 5 years | 3 months | 12 months |
| PEFC | Annual | 5 years | 3 months | 12 months |
| RSPO | Annual | 5 years | 3 months | 12 months |
| Rainforest Alliance | Annual | 3 years | 3 months | 12 months |
| ISCC | Annual | Annual | Per audit | Per audit |

### D. Provenance Chain Design

```
Audit Provenance Chain:
    hash_0 = SHA-256(risk_inputs || scheduling_parameters)
    hash_1 = SHA-256(hash_0 || auditor_assignment_data)
    hash_2 = SHA-256(hash_1 || checklist_results || evidence_hashes)
    hash_3 = SHA-256(hash_2 || nc_classifications || classification_rules)
    hash_4 = SHA-256(hash_3 || car_lifecycle_events)
    hash_5 = SHA-256(hash_4 || report_content_hash)
    chain_hash = SHA-256(hash_5 || final_audit_status)
```

### E. References

1. Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR)
2. ISO 19011:2018 -- Guidelines for auditing management systems
3. ISO/IEC 17065:2012 -- Requirements for bodies certifying products, processes and services
4. ISO/IEC 17021-1:2015 -- Requirements for bodies providing audit and certification
5. FSC-STD-20-007 -- Forest Management Evaluations
6. FSC-STD-20-011 -- Chain of Custody Evaluations
7. RSPO Principles and Criteria (2018)
8. PEFC ST 2003:2020 -- Chain of Custody
9. Rainforest Alliance Sustainable Agriculture Standard (2020)
10. ISCC EU/PLUS System Requirements
11. Directive (EU) 2024/1760 -- CSDDD
12. IAF Multilateral Recognition Arrangement (MLA)

---

*End of Architecture Specification*
