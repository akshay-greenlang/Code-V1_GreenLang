# PRD: AGENT-EUDR-017 -- Supplier Risk Scorer Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-017 |
| **Agent ID** | GL-EUDR-SRS-017 |
| **Component** | Supplier Risk Scorer Agent |
| **Category** | EUDR Regulatory Agent -- Risk Assessment |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-09 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-09 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 4, 9, 10, 11, 12, 13, 14 (Due Diligence & Supplier Assessment) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Articles 10-13 require EU operators and traders to perform risk assessment and due diligence not only at the country level but at the individual supplier level. While AGENT-EUDR-016 (Country Risk Evaluator) provides country-level and commodity-level risk scoring, EUDR compliance fundamentally depends on understanding the risk profile of each specific supplier in the operator's supply chain. A country may be classified as "standard risk," but an individual supplier within that country may exhibit critical risk due to poor documentation, expired certifications, sourcing from deforestation hotspots, non-responsive due diligence behavior, or opaque sub-supplier networks.

Today, EU operators and compliance teams face the following problems when assessing supplier-level risk for EUDR compliance:

- **No composite supplier risk scoring engine**: Operators assess suppliers ad hoc using inconsistent criteria. There is no automated system that synthesizes geographic sourcing patterns, compliance history, documentation quality, certification status, traceability completeness, financial stability, environmental performance, and social compliance into a single, weighted, auditable composite risk score per supplier.
- **No due diligence history tracking**: Operators lack a systematic record of all due diligence activities, non-conformances, corrective actions, and interactions per supplier. When auditors request DD history, compliance teams scramble to assemble records from email threads, spreadsheets, and disconnected systems.
- **No documentation completeness analysis**: EUDR requires specific documents per supplier (geolocation data, DDS references, product descriptions, quantities, harvesting dates, compliance declarations). Operators have no automated way to score document completeness, track expiry, identify gaps, or verify consistency across document sets.
- **No certification validation engine**: Suppliers present certifications (FSC, PEFC, RSPO, Rainforest Alliance, UTZ, ISCC) as evidence of compliance, but operators cannot systematically validate scope coverage, expiry dates, chain of custody alignment, or certification body accreditation. Fraudulent or misrepresented certifications go undetected.
- **No geographic sourcing pattern analysis**: A supplier's declared sourcing locations may overlap with deforestation hotspots, protected areas, or indigenous territories. Operators have no automated cross-referencing between supplier sourcing patterns and the sub-national risk data produced by AGENT-EUDR-016.
- **No supplier network and relationship analysis**: EUDR supply chains involve sub-suppliers, intermediaries, and complex routing. Operators cannot map supplier-to-supplier relationships, detect circular dependencies, or assess how sub-supplier risk propagates to direct suppliers.
- **No continuous supplier monitoring**: Supplier risk is dynamic -- certifications expire, sourcing patterns shift, sanctions lists update, and environmental incidents occur. Operators perform point-in-time assessments with no continuous monitoring or alerting.
- **No supplier risk reporting for auditors**: Compliance officers need standardized, auditable supplier risk reports for regulatory submission and third-party audits. Currently, these are manually assembled and lack provenance tracking.

Without solving these problems, EU operators face penalties of up to 4% of annual EU turnover, confiscation of goods, temporary exclusion from public procurement, and public naming. Supplier-level due diligence failures are the most common cause of EUDR non-compliance findings.

### 1.2 Solution Overview

Agent-EUDR-017: Supplier Risk Scorer is a specialized risk assessment agent that provides comprehensive, multi-dimensional, supplier-level risk evaluation for EUDR compliance. It is the SECOND agent in the Risk Assessment sub-category, following AGENT-EUDR-016 (Country Risk Evaluator). While EUDR-016 answers "How risky is this country/commodity?", EUDR-017 answers "How risky is this specific supplier?" The agent drills down from country-level risk to individual supplier risk scoring based on 8 weighted risk factors, providing the granularity required by EUDR Articles 10-13 for operator due diligence.

The agent integrates deeply with AGENT-EUDR-016 for country-level risk baselines, AGENT-EUDR-001 for supply chain graph context, AGENT-EUDR-008 for multi-tier supplier data, and the GL-EUDR-APP platform for end-user delivery. It operates under strict zero-hallucination guarantees with deterministic scoring, full provenance tracking, and Decimal arithmetic.

Core capabilities:

1. **Composite supplier risk scoring** -- Weighted multi-factor risk scores (0-100) for every supplier, combining 8 risk factors: geographic sourcing (20%), compliance history (15%), documentation quality (15%), certification status (15%), traceability completeness (10%), financial stability (10%), environmental performance (10%), social compliance (5%). All calculations are deterministic, reproducible, and fully auditable.
2. **Due diligence history tracking** -- Complete lifecycle tracking of all DD activities, non-conformances, corrective action plans, escalations, and regulatory submission readiness per supplier with cost tracking and gap identification.
3. **Documentation completeness analysis** -- Automated scoring of document quality (completeness, accuracy, consistency, timeliness) against EUDR-mandated document requirements per supplier, with gap identification, expiry tracking, and automated request generation.
4. **Certification and standards validation** -- Systematic validation of supplier certifications (FSC, PEFC, RSPO, Rainforest Alliance, UTZ, ISCC, organic, Fair Trade) including scope verification, chain of custody validation, expiry monitoring, accreditation checks, and fraudulent certification detection.
5. **Geographic sourcing pattern analysis** -- Mapping of supplier sourcing locations to deforestation risk zones, integration with EUDR-016 country/sub-national risk scores, sourcing concentration analysis, protected area proximity, and satellite alert cross-referencing.
6. **Supplier network and relationship analysis** -- Graph-based mapping of supplier-to-supplier relationships (sub-suppliers, intermediaries), risk propagation through supply chains, shared supplier detection, circular relationship identification, and ultimate source tracing capability scoring.
7. **Continuous monitoring and alert engine** -- Real-time supplier risk monitoring with configurable frequencies, alert generation for threshold breaches, sanction list screening, watchlist management, portfolio risk aggregation, and compliance calendar tracking.
8. **Supplier risk reporting and analytics** -- Individual and portfolio-level risk reports, comparative analysis, trend dashboards, regulatory submission formatting (Article 4 DDS), benchmarking, executive summaries, and audit-ready documentation packages.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Supplier coverage | Score 100% of active suppliers in operator's supply chain | % of suppliers with active risk scores |
| Risk factor completeness | 8/8 factors scored for 80%+ of suppliers | Average factor completeness across supplier base |
| Risk score accuracy | 100% deterministic, reproducible | Bit-perfect reproducibility tests |
| DD history completeness | 100% of DD activities tracked with audit trail | % of DD activities with complete records |
| Document gap detection | Identify 95%+ of missing EUDR-required documents | Precision/recall against manual audit |
| Certification validation accuracy | 100% of expired/invalid certifications detected | Zero false negatives on certification validation |
| Assessment throughput | < 200ms per single supplier assessment | p99 latency under load |
| Portfolio assessment | < 30 seconds for 1,000-supplier portfolio | Batch assessment performance |
| Alert latency | < 1 hour from trigger event to alert delivery | Time from data change to notification |
| EUDR compliance coverage | 100% of Articles 10-13 supplier DD requirements | Regulatory compliance matrix |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated supplier risk management and due diligence tools market of 2-4 billion EUR.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers requiring automated supplier risk assessment for EUDR-regulated commodity suppliers, estimated at 600M-1B EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 40-60M EUR in supplier risk scoring module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) managing hundreds to thousands of suppliers of EUDR-regulated commodities
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya) with complex, multi-tier supplier networks
- Timber and paper industry operators sourcing from tropical and temperate forests
- Compliance officers responsible for EUDR supplier due diligence decisions

**Secondary:**
- Customs brokers and freight forwarders managing supplier documentation for EUDR goods
- Commodity traders and intermediaries operating across multi-country supplier portfolios
- Certification bodies (FSC, RSPO, Rainforest Alliance) validating supplier compliance claims
- Compliance consultants and auditors performing third-party EUDR supplier verification
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Spreadsheet | No cost; familiar | Cannot synthesize 8 risk factors; no continuous monitoring; no audit trail; error-prone | Automated, deterministic, 8-factor composite, continuous monitoring |
| Generic supplier management (SAP Ariba, Jaggaer) | Enterprise procurement integration; broad supplier data | Not EUDR-specific; no deforestation risk; no certification validation; no geographic sourcing analysis | Purpose-built for EUDR; deforestation-aware; certification validation engine |
| ESG risk platforms (EcoVadis, Sedex) | Broad sustainability scoring; established databases | Not EUDR-specific; no Article 10-13 alignment; no EUDR document requirements; no sub-national geo risk | EUDR regulatory fidelity; EUDR document tracking; sub-national risk integration |
| Niche EUDR tools (Preferred by Nature) | Commodity expertise; audit experience | Single-commodity focus; manual scoring; no continuous monitoring; no supplier network analysis | All 7 commodities; automated scoring; continuous monitoring; network graph analysis |
| Consulting firms (Deloitte, EY, PwC) | Expert judgment; credibility with regulators | Expensive (20K-100K EUR per supplier audit); slow (weeks); point-in-time; not scalable | Automated; < 200ms per assessment; continuous; scalable to thousands of suppliers |

### 2.4 Differentiation Strategy

1. **EUDR regulatory fidelity** -- Every risk factor, weight, and scoring criterion maps to specific EUDR Article 10-13 requirements for supplier due diligence.
2. **8-factor composite scoring** -- The most comprehensive supplier risk model in the EUDR compliance market, covering geographic, compliance, documentation, certification, traceability, financial, environmental, and social dimensions.
3. **Country-to-supplier risk integration** -- Seamless integration with EUDR-016 country risk scores, enabling supplier risk to inherit and refine country-level baselines.
4. **Continuous monitoring** -- Not a point-in-time assessment; continuous risk monitoring with configurable alert thresholds, sanction screening, and certification expiry tracking.
5. **Zero-hallucination scoring** -- Deterministic, auditable, reproducible risk calculations with no LLM in the critical path. Every score traceable to source data.
6. **Supply chain network intelligence** -- Graph-based analysis of supplier relationships, sub-supplier risk propagation, and ultimate source tracing capability -- unique in the EUDR compliance market.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to perform Article 10-13 compliant supplier risk assessment | 100% of customers pass supplier DD audits | Q2 2026 |
| BG-2 | Reduce time-to-assess supplier risk from days to seconds | 99%+ reduction in assessment time (days to < 200ms) | Q2 2026 |
| BG-3 | Become the reference supplier risk scoring platform for EUDR compliance | 500+ enterprise customers using supplier risk module | Q4 2026 |
| BG-4 | Prevent compliance penalties for customers through proactive supplier monitoring | Zero EUDR penalties attributable to undetected supplier risk for active customers | Ongoing |
| BG-5 | Enable portfolio-level risk management across supplier bases of 1,000+ suppliers | Support batch assessment and portfolio analytics | Q3 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Comprehensive supplier scoring | Score every supplier with 8-factor weighted composite risk (0-100) |
| PG-2 | DD lifecycle tracking | Track complete due diligence history per supplier with audit trail |
| PG-3 | Document completeness | Automated EUDR document gap analysis and quality scoring per supplier |
| PG-4 | Certification validation | Validate scope, expiry, chain of custody, and accreditation for all certifications |
| PG-5 | Geographic risk integration | Cross-reference supplier sourcing with EUDR-016 country and sub-national risk |
| PG-6 | Network risk analysis | Map and score supplier relationship networks with risk propagation |
| PG-7 | Continuous monitoring | Real-time monitoring with alerts for risk threshold breaches and data changes |
| PG-8 | Audit-ready reporting | Generate compliant supplier risk reports for regulatory submission and audits |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Single supplier assessment | < 200ms p99 per supplier risk assessment |
| TG-2 | Batch portfolio assessment | < 30 seconds for 1,000 suppliers |
| TG-3 | Report generation | < 5 seconds per individual supplier risk report PDF |
| TG-4 | API response time | < 150ms p95 for standard queries |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |
| TG-7 | Alert latency | < 1 hour from trigger event to notification delivery |

---

## 4. User Personas

### Persona 1: Compliance Officer -- Sofia (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of EUDR Compliance at a large EU cocoa importer |
| **Company** | 6,000 employees, managing 400+ cocoa suppliers across 15 countries |
| **EUDR Pressure** | Board mandate to have all suppliers risk-scored before SME enforcement deadline; auditor requesting supplier DD evidence |
| **Pain Points** | Cannot systematically score suppliers; relies on ad hoc questionnaires and gut feel; no visibility into supplier certification validity; spends 3 days per supplier on manual DD; cannot prioritize which suppliers need enhanced DD |
| **Goals** | Automated supplier risk scores for entire portfolio; prioritized action list for high-risk suppliers; audit-ready DD documentation per supplier; continuous monitoring with alerts |
| **Technical Skill** | Moderate -- comfortable with web applications and dashboards |

### Persona 2: Supply Chain Risk Analyst -- Markus (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Risk Analyst at an EU timber trading company |
| **Company** | 1,200 employees, managing 250+ wood suppliers across 30+ countries |
| **EUDR Pressure** | Must assess risk for every supplier relationship including sub-suppliers; must detect high-risk supplier networks |
| **Pain Points** | Cannot map supplier-to-sub-supplier relationships; no way to propagate sub-supplier risk upward; certification scope verification is manual and error-prone; no visibility into supplier sourcing pattern changes |
| **Goals** | Supplier network risk map; sub-supplier risk propagation scoring; automated certification validation; geographic sourcing pattern monitoring with deforestation alert cross-reference |
| **Technical Skill** | High -- comfortable with data tools, APIs, and analytical platforms |

### Persona 3: Procurement Manager -- Isabelle (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Procurement Director at a palm oil refinery |
| **Company** | 3,500 employees, sourcing from 180+ palm oil suppliers in Indonesia and Malaysia |
| **EUDR Pressure** | Must ensure all supplier documentation is complete before purchasing; must manage certification renewals across supplier base |
| **Pain Points** | Document collection from suppliers takes weeks; no automated gap analysis; certification expiry surprises disrupt procurement; no comparative view of supplier risk across portfolio |
| **Goals** | Automated document completeness scoring per supplier; certification expiry dashboard; comparative supplier risk ranking for sourcing decisions; automated document request generation |
| **Technical Skill** | Low-moderate -- uses ERP and web applications |

### Persona 4: External Auditor -- Dr. Weber (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must verify operator's supplier risk assessment methodology, data quality, and DD completeness |
| **Pain Points** | Operators provide inconsistent supplier risk evidence; no standardized scoring methodology; no provenance tracking for risk data; DD history is fragmented and incomplete |
| **Goals** | Access read-only supplier risk data with full provenance; verify scoring methodology and factor weights; audit DD activity history; validate certification status claims; review supplier risk score change history |
| **Technical Skill** | Moderate -- comfortable with audit software and report review |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(2)** | Due diligence -- collect information, assess risk, mitigate risk | Composite supplier risk scoring (F1) with 8 weighted factors; DD history tracking (F2) |
| **Art. 4(2)(a-e)** | Information collection requirements: product description, quantity, supplier info, country of production, geolocation | Documentation completeness analyzer (F3) tracks all required data elements per supplier |
| **Art. 4(2)(f)** | Supply chain information for relevant commodity or product | Supplier network analyzer (F6) maps full supplier chain context |
| **Art. 9(1)(a-d)** | Geolocation of all plots of land where commodity was produced | Geographic sourcing analyzer (F5) validates supplier-declared sourcing locations |
| **Art. 10(1)** | Risk assessment -- assess and identify risk of non-compliance | Composite risk scoring (F1) with per-supplier risk level classification |
| **Art. 10(2)(a)** | Risk criterion: complexity of relevant supply chain | Network analyzer (F6) scores supplier chain complexity and depth |
| **Art. 10(2)(b)** | Risk criterion: risk of circumvention or mixing with unknown origin | Network analyzer (F6) detects circular relationships and intermediary risk amplification |
| **Art. 10(2)(c)** | Risk criterion: risk of deforestation in country of production | Geographic sourcing analyzer (F5) integrates EUDR-016 country and sub-national risk |
| **Art. 10(2)(d)** | Risk criterion: risk of non-compliance with country legislation | Certification validator (F4) assesses supplier's legal compliance evidence |
| **Art. 10(2)(e)** | Risk criterion: concerns about country of production or origin | Geographic sourcing analyzer (F5) flags suppliers sourcing from high-risk regions |
| **Art. 10(2)(f)** | Risk criterion: risk of circumvention through mixing | Network analyzer (F6) assesses mixing risk through intermediaries |
| **Art. 11** | Risk mitigation measures -- enhanced measures for high-risk | Monitoring engine (F7) triggers enhanced monitoring for high/critical risk suppliers |
| **Art. 12** | Submission of DDS to EU Information System | Risk reporting engine (F8) formats supplier risk data for DDS inclusion |
| **Art. 13** | Simplified due diligence for low-risk | DD history tracker (F2) identifies suppliers eligible for simplified DD |
| **Art. 14** | Obligations of traders | Supplier scoring (F1) differentiates operator vs. trader obligations |
| **Art. 31** | Record keeping for 5 years | Immutable audit log and risk score change history with 5-year retention |

### 5.2 Supplier Due Diligence Requirements (Articles 10-13 Mapping)

| DD Requirement | EUDR Article | Agent Feature | Scoring Impact |
|----------------|-------------|---------------|----------------|
| Supplier identification and verification | Art. 4(2) | F1 (identity in composite score) | 5 points in compliance_history factor |
| Geolocation data collection from supplier | Art. 9 | F5 (geographic sourcing) | 20% weight in composite score |
| Supplier compliance declaration | Art. 4(2) | F3 (documentation) | 15% weight in composite score |
| Certification evidence collection | Art. 10(2)(d) | F4 (certification) | 15% weight in composite score |
| Supply chain complexity assessment | Art. 10(2)(a) | F6 (network analysis) | Sub-factor in traceability_completeness |
| Mixing and circumvention risk | Art. 10(2)(b,f) | F6 (network analysis) | Risk amplification in composite score |
| Country/region risk assessment | Art. 10(2)(c,e) | F5 (geographic sourcing) | 20% weight in composite score |
| Enhanced DD for high-risk suppliers | Art. 11 | F7 (monitoring) | Triggers enhanced monitoring frequency |
| Simplified DD for low-risk suppliers | Art. 13 | F2 (DD tracker) | Reduced DD requirements and audit frequency |
| Record keeping (5 years) | Art. 31 | F2 (DD tracker) | Immutable audit trail |
| DDS submission with supplier data | Art. 12 | F8 (reporting) | DDS-formatted supplier risk export |

### 5.3 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for supplier geographic sourcing verification |
| June 29, 2023 | Regulation entered into force | Legal basis for all supplier risk assessment |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | All supplier risk assessments must be operational for large operators |
| June 30, 2026 | Enforcement for SMEs | SME supplier onboarding wave; agent must handle 5-10x supplier volume |
| Ongoing (quarterly) | Operator DDS submission | Supplier risk reports must be current for quarterly DDS filing |
| Ongoing (periodic) | EC benchmarking list updates | Supplier geographic risk must be recalculated when country classifications change |

---

## 6. Features and Requirements

### 6.1 Scope -- In and Out

**In Scope (v1.0):**
- Composite supplier risk scoring with 8 weighted factors for unlimited suppliers
- Due diligence activity history tracking with non-conformance and corrective action management
- EUDR document completeness scoring and gap analysis per supplier
- Certification validation (FSC, PEFC, RSPO, Rainforest Alliance, UTZ, ISCC, organic, Fair Trade)
- Geographic sourcing pattern analysis with EUDR-016 country/sub-national risk integration
- Supplier network and relationship mapping with risk propagation
- Continuous monitoring with configurable alerts, sanction screening, and watchlists
- Supplier risk reporting for regulatory submission, audits, and portfolio analytics
- Integration with EUDR-016, EUDR-001, EUDR-008, and GL-EUDR-APP

**Out of Scope (v1.0):**
- Direct supplier communication portal (supplier self-service data entry deferred to Phase 2)
- Financial credit scoring integration (Dun & Bradstreet, Moody's -- defer to Phase 2)
- Automated supplier deselection recommendations (legal liability concern)
- Predictive ML models for supplier risk forecasting (defer to Phase 2)
- Mobile native application (web responsive design only)
- Direct EU Information System submission (handled by GL-EUDR-APP DDS module)
- Blockchain-based immutable supplier records (SHA-256 provenance hashes sufficient)

### 6.2 Zero-Hallucination Principles

All 8 engines in this agent operate under strict zero-hallucination guarantees:

| Principle | Implementation |
|-----------|---------------|
| **Deterministic calculations** | Same inputs always produce the same risk scores (bit-perfect reproducibility) |
| **No LLM in critical path** | All risk scoring, classification, and analysis use deterministic algorithms only |
| **Authoritative data sources only** | All risk factors sourced from operator data, certifying bodies, EUDR-016 country risk, satellite data; no synthetic data |
| **Full provenance tracking** | Every risk score includes SHA-256 hash, data source citations, and calculation timestamps |
| **Immutable audit trail** | All risk score changes recorded in `gl_eudr_srs_audit_log` with before/after values |
| **Decimal arithmetic** | Risk scores use Decimal type to prevent floating-point drift |
| **Version-controlled data** | Supplier risk profiles are versioned; any change creates a new version with timestamp |

### 6.3 Must-Have Features (P0 -- Launch Blockers)

All 8 features below are P0 launch blockers. The agent cannot ship without all 8 features operational. Features 1-5 form the core supplier risk assessment engine; Features 6-8 form the network analysis, monitoring, and reporting layer.

**P0 Features 1-5: Core Supplier Risk Assessment Engine**

---

#### Feature 1: Supplier Composite Risk Scoring Engine

**User Story:**
```
As a compliance officer,
I want a composite risk score for every supplier based on multiple weighted risk factors,
So that I can quickly determine which suppliers pose the highest EUDR compliance risk and prioritize my due diligence efforts.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F1.1: Calculates composite supplier risk score (0-100) from 8 weighted factors: geographic_sourcing (20%), compliance_history (15%), documentation_quality (15%), certification_status (15%), traceability_completeness (10%), financial_stability (10%), environmental_performance (10%), social_compliance (5%); all weights operator-configurable within bounds (each factor 2-40%, total must equal 100%)
- [ ] F1.2: Classifies suppliers into 4 risk levels based on composite score: LOW (0-25), MEDIUM (25-50), HIGH (50-75), CRITICAL (75-100); thresholds configurable per operator
- [ ] F1.3: Computes confidence score (0-100) based on data completeness for each risk factor, with minimum data thresholds: confidence >= 80 requires at least 6/8 factors with primary data; confidence 50-79 requires at least 4/8 factors; confidence < 50 flags supplier for manual review
- [ ] F1.4: Performs trend analysis classifying each supplier's risk trajectory as improving, stable, or deteriorating based on rolling 12-month window of historical risk scores with minimum 3 data points required
- [ ] F1.5: Calculates peer group benchmarking within commodity and region, providing percentile rank (0-100) against comparable suppliers (same commodity, same country or region), with minimum peer group size of 5 for statistical validity
- [ ] F1.6: Computes sub-scores for each of the 8 risk factor dimensions (0-100 each), displayed alongside composite score for factor-level drill-down and root cause analysis
- [ ] F1.7: Triggers automatic re-scoring when any underlying data changes (new DD activity, document upload, certification update, geographic sourcing change, EUDR-016 country reclassification), with re-score queued and processed within 5 minutes
- [ ] F1.8: Supports batch scoring for portfolio assessment: score up to 5,000 suppliers in a single batch request with progress tracking and partial result delivery
- [ ] F1.9: Maintains version-controlled risk scores with immutable change history, recording previous score, new score, change reason, changed_by, affected factors, and timestamp for every score update
- [ ] F1.10: Validates all risk scores are deterministic: same supplier data inputs always produce the same composite score (bit-perfect reproducibility verified by SHA-256 hash matching across 100 repeated calculations)

**Risk Calculation Formula:**
```
Composite_Supplier_Risk = (
    geographic_sourcing_score * W_geographic         # default 0.20
    + compliance_history_score * W_compliance        # default 0.15
    + documentation_quality_score * W_documentation  # default 0.15
    + certification_status_score * W_certification   # default 0.15
    + traceability_completeness_score * W_trace      # default 0.10
    + financial_stability_score * W_financial        # default 0.10
    + environmental_performance_score * W_environ    # default 0.10
    + social_compliance_score * W_social             # default 0.05
)

Classification:
  LOW:      0 <= Composite <= 25
  MEDIUM:   25 < Composite <= 50
  HIGH:     50 < Composite <= 75
  CRITICAL: 75 < Composite <= 100
```

**Non-Functional Requirements:**
- Performance: < 200ms per single supplier assessment; < 30 seconds for 1,000-supplier batch
- Determinism: Bit-perfect reproducibility across runs
- Precision: Decimal arithmetic for all score calculations
- Auditability: SHA-256 provenance hash on every assessment

**Dependencies:**
- AGENT-EUDR-016 Country Risk Evaluator for geographic risk baselines
- AGENT-EUDR-008 Multi-Tier Supplier Tracker for supplier master data
- AGENT-FOUND-005 Citations & Evidence Agent for source attribution
- AGENT-FOUND-008 Reproducibility Agent for determinism verification

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- New supplier with no history -> Score with available data only; set confidence = low; flag for expedited data collection
- Supplier with only 1-2 factors available -> Calculate partial score with proportional re-weighting; reduce confidence; mandate data enrichment
- Supplier country reclassified by EC -> Trigger immediate re-score for all suppliers in that country; generate impact notification

---

#### Feature 2: Supplier Due Diligence History Tracker

**User Story:**
```
As a compliance officer,
I want a complete, auditable history of all due diligence activities performed on each supplier,
So that I can demonstrate to auditors and regulators that I have fulfilled my EUDR Article 10-13 obligations.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F2.1: Tracks all due diligence activity types per supplier: initial assessment, periodic review, enhanced DD, site visit, questionnaire, document collection, satellite verification request, corrective action plan, escalation, closure; each activity recorded with date, actor, outcome, duration, and cost
- [ ] F2.2: Computes DD completion rate per supplier (0-100%) measuring the percentage of required DD activities completed against the EUDR-mandated checklist for the supplier's risk level (simplified/standard/enhanced DD requirements)
- [ ] F2.3: Computes DD quality score per supplier (0-100) based on activity thoroughness, evidence quality, timeliness of completion, and findings resolution rate
- [ ] F2.4: Tracks non-conformances per supplier with severity classification (critical/major/minor), root cause categorization, affected EUDR article, and resolution status (open/in_progress/closed/recurring)
- [ ] F2.5: Monitors corrective action plans (CAPs) per supplier with milestone tracking, deadline management, evidence of completion, effectiveness verification, and automatic escalation for overdue CAPs (configurable escalation at 7, 14, 30 days overdue)
- [ ] F2.6: Tracks due diligence cost per supplier (in EUR) broken down by activity type, enabling ROI analysis and cost benchmarking across the supplier portfolio
- [ ] F2.7: Maintains historical score progression tracking with timestamped score snapshots, enabling visualization of how a supplier's risk score has evolved over months and years
- [ ] F2.8: Performs gap identification in DD coverage, flagging suppliers where required DD activities are missing, overdue, or stale (older than configured freshness threshold: default 6 months for standard DD, 3 months for enhanced DD)
- [ ] F2.9: Implements escalation workflow for non-responsive suppliers with configurable escalation levels: Level 1 (reminder at 14 days), Level 2 (manager notification at 30 days), Level 3 (procurement hold recommendation at 60 days), Level 4 (relationship review at 90 days)
- [ ] F2.10: Computes regulatory submission readiness score per supplier (0-100) indicating completeness of DD evidence required for inclusion in the operator's DDS per Article 12, with itemized checklist of missing elements

**Non-Functional Requirements:**
- Completeness: 100% of DD activities recorded with actor, timestamp, and outcome
- Retention: All DD history retained for minimum 5 years per EUDR Article 31
- Auditability: Every DD record includes provenance hash and is immutable once finalized

**Dependencies:**
- Feature 1 (Supplier Composite Risk Scoring) for risk level driving DD requirements
- AGENT-EUDR-008 Multi-Tier Supplier Tracker for supplier master data and activity history
- GL-EUDR-APP notification service for escalation alerts

**Estimated Effort:** 3 weeks (1 backend engineer)

---

#### Feature 3: Documentation Completeness Analyzer

**User Story:**
```
As a procurement manager,
I want automated scoring of document quality and completeness for every supplier,
So that I can identify which suppliers have EUDR documentation gaps and generate targeted document requests.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F3.1: Scores document quality per supplier across 4 dimensions: completeness (% of required documents present), accuracy (cross-reference consistency), consistency (no conflicting data across documents), and timeliness (% of documents within validity period); each dimension scored 0-100, combined into composite document quality score
- [ ] F3.2: Tracks all EUDR-required documents per supplier: geolocation data (GPS/polygon), DDS references, product descriptions, quantities with units, harvesting/production dates, compliance declarations, operator registration numbers, CN/HS codes, country of production evidence, and supplier chain documentation
- [ ] F3.3: Implements document version control with automatic expiry tracking: each document has upload date, effective date, expiry date, and version number; system generates alerts at configurable intervals before expiry (default: 90, 60, 30, 7 days)
- [ ] F3.4: Performs missing document identification and gap analysis against EUDR-mandated document matrix, generating prioritized gap list per supplier sorted by regulatory criticality (critical gaps block DDS submission; high gaps reduce compliance score; medium gaps flagged for collection)
- [ ] F3.5: Implements document authenticity indicators through cross-reference validation: geolocation data cross-referenced with EUDR-016 country data, quantities cross-referenced with trade flow data, harvesting dates cross-referenced with seasonal patterns, and CN codes validated against EUDR Annex I mapping
- [ ] F3.6: Accepts documents in multiple formats: PDF, XML, JSON, GeoJSON, CSV, Excel (.xlsx), with automatic format detection and content extraction using AGENT-DATA-001 (PDF) and AGENT-DATA-002 (Excel/CSV)
- [ ] F3.7: Performs language detection on submitted documents and flags documents requiring translation, with support for EU official languages and key supplier languages (English, French, German, Spanish, Portuguese, Indonesian, Malay, Vietnamese)
- [ ] F3.8: Tracks document submission deadlines per supplier aligned with quarterly DDS filing calendar, generating countdown notifications and overdue alerts
- [ ] F3.9: Generates automated document request communications per supplier listing specific missing documents, required format, submission deadline, and upload instructions; request templates available in 8 languages
- [ ] F3.10: Computes document quality trend analysis per supplier showing improvement or deterioration over time, with quarterly trend reports and flag for suppliers with declining documentation quality

**Non-Functional Requirements:**
- Coverage: Track all 12+ EUDR-required document types per supplier
- Performance: Document quality score calculated in < 500ms per supplier
- Storage: Documents stored in S3 with AES-256 encryption at rest

**Dependencies:**
- AGENT-DATA-001 PDF & Invoice Extractor for document content extraction
- AGENT-DATA-002 Excel/CSV Normalizer for structured data extraction
- AGENT-FOUND-005 Citations & Evidence Agent for cross-reference validation
- S3 object storage for document files

**Estimated Effort:** 3 weeks (1 backend engineer, 1 integration engineer)

---

#### Feature 4: Certification & Standards Validator

**User Story:**
```
As a supply chain risk analyst,
I want automated validation of all supplier certifications including scope, expiry, and chain of custody,
So that I can ensure certification claims are legitimate and provide meaningful risk mitigation for EUDR compliance.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F4.1: Validates supplier certifications across 8 major schemes: FSC (Forest Stewardship Council), PEFC (Programme for the Endorsement of Forest Certification), RSPO (Roundtable on Sustainable Palm Oil), Rainforest Alliance, UTZ (now merged with Rainforest Alliance), ISCC (International Sustainability and Carbon Certification), organic (EU/USDA), and Fair Trade; each certification stored with scheme name, certificate number, scope, valid_from, valid_to, and certification body
- [ ] F4.2: Monitors certification expiry with advance alerts at configurable intervals (default: 180, 90, 60, 30, 14, 7 days before expiry); expired certifications immediately flagged and supplier risk score recalculated with certification_status factor set to maximum risk
- [ ] F4.3: Performs scope verification validating that the certification covers the specific products and geographic regions claimed by the supplier; e.g., an FSC certificate for plantation wood does not cover natural forest timber; an RSPO certificate for Malaysia does not cover Indonesian operations
- [ ] F4.4: Validates chain of custody models per certification scheme: FSC-COC (Chain of Custody), FSC-CW (Controlled Wood), RSPO-SCC (Supply Chain Certification), RSPO-MB (Mass Balance), RSPO-SG (Segregated), RSPO-IP (Identity Preserved); verifies supplier's claimed custody model matches certificate scope
- [ ] F4.5: Verifies certification body accreditation by checking the certifying organization against scheme-specific accreditation registries (FSC: ASI-accredited; RSPO: approved certification bodies; PEFC: national governing bodies); flags certifications from non-accredited bodies
- [ ] F4.6: Computes multi-scheme aggregation score (0-100) when a supplier holds multiple certifications, applying diminishing returns (first scheme: full credit; second scheme: 50% additional credit; third+: 25% each), capped at 100; reflects that multiple certifications provide incremental but not unlimited risk reduction
- [ ] F4.7: Performs certification coverage vs. supplier volume alignment analysis: compares certified product volume against total supplier volume to calculate % certified; flags suppliers where certification covers < 50% of supplied volume as partially certified with reduced risk credit
- [ ] F4.8: Implements fraudulent certification detection heuristics: flags certifications with invalid certificate numbers (format check per scheme), expired certifying bodies, certificates covering impossibly large geographic areas, duplicate certificate numbers across suppliers, and certificates issued by entities not in accreditation registries
- [ ] F4.9: Maintains scheme equivalence mapping for cross-scheme comparison: maps FSC to PEFC equivalence levels, RSPO to ISCC commodity overlap, Rainforest Alliance to UTZ merger status, enabling operators to compare certification quality across different schemes for the same commodity
- [ ] F4.10: Computes certified material percentage credit system: for mass balance suppliers, tracks percentage of input material that is certified vs. uncertified, and applies proportional risk credit (e.g., 70% certified material = 70% of maximum certification risk reduction credit)

**Non-Functional Requirements:**
- Coverage: Validation rules for all 8 certification schemes
- Performance: Certification validation < 100ms per supplier
- Accuracy: Zero false negatives on expired certification detection; < 1% false positive rate on fraud detection heuristics

**Dependencies:**
- Certification scheme registry APIs (FSC Certificate Database, RSPO public registry)
- Feature 1 (Supplier Composite Risk Scoring) for certification factor integration
- AGENT-EUDR-008 Multi-Tier Supplier Tracker for supplier certification records

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data engineer)

---

**P0 Features 5-8: Geographic Analysis, Network Intelligence, Monitoring, and Reporting Layer**

> Features 5 through 8 are P0 launch blockers. Without geographic sourcing analysis, network intelligence, continuous monitoring, and risk reporting, the core supplier assessment engine cannot deliver end-user value. These features are the delivery mechanism through which compliance officers, analysts, and auditors interact with the supplier risk engine.

---

#### Feature 5: Geographic Sourcing Pattern Analyzer

**User Story:**
```
As a compliance officer,
I want to see how each supplier's sourcing locations map to deforestation risk zones,
So that I can identify suppliers sourcing from high-risk areas and apply enhanced due diligence per EUDR Article 10(2)(c,e).
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F5.1: Maps all declared supplier sourcing locations (GPS coordinates, polygon boundaries, administrative regions) to EUDR-016 country risk levels and sub-national hotspot scores, providing a combined geographic risk score (0-100) per supplier
- [ ] F5.2: Integrates with AGENT-EUDR-016 Country Risk Evaluator to consume country-level composite risk scores, commodity-specific risk scores, and sub-national hotspot data for every supplier sourcing location
- [ ] F5.3: Performs sub-national risk mapping at province/district level, identifying suppliers whose declared sourcing regions overlap with EUDR-016 deforestation hotspots (risk score > 65) and automatically applying risk amplification factor (1.2x-1.5x configurable)
- [ ] F5.4: Computes sourcing concentration analysis per supplier: single-source suppliers (100% from one region) receive higher geographic risk than diversified suppliers (multiple regions); Herfindahl-Hirschman Index (HHI) calculated on sourcing distribution
- [ ] F5.5: Tracks historical sourcing pattern changes per supplier, flagging significant shifts: new sourcing region added, existing region dropped, volume redistribution > 20% between regions; each change triggers geographic risk re-assessment
- [ ] F5.6: Calculates proximity to protected areas and indigenous territories for each supplier sourcing location using AGENT-DATA-006 GIS/Mapping Connector spatial operations; flags locations within 10km of protected areas or indigenous territories
- [ ] F5.7: Performs seasonal sourcing pattern analysis correlating supplier delivery volumes with known deforestation seasons (dry season burning, harvest periods), flagging suppliers with delivery spikes aligned with peak deforestation periods
- [ ] F5.8: Assesses new sourcing region risk for suppliers that declare new production origins, running immediate EUDR-016 risk assessment on the new region before it is approved for sourcing
- [ ] F5.9: Computes supply chain depth analysis per supplier distinguishing direct sourcing (supplier owns/controls production) from indirect sourcing (supplier aggregates from sub-suppliers), with indirect sourcing receiving higher risk due to reduced traceability
- [ ] F5.10: Cross-references supplier sourcing locations against satellite deforestation alerts from AGENT-DATA-007, flagging suppliers whose declared sourcing locations intersect with active deforestation alerts (GLAD, RADD) within the past 12 months

**Non-Functional Requirements:**
- Spatial precision: Sourcing location matching accurate to 1km resolution
- Performance: Geographic risk calculation < 500ms per supplier
- Integration: Real-time data feed from EUDR-016 with < 1-hour propagation latency

**Dependencies:**
- AGENT-EUDR-016 Country Risk Evaluator for country/sub-national risk scores and hotspot data
- AGENT-DATA-006 GIS/Mapping Connector for spatial operations and protected area data
- AGENT-DATA-007 Deforestation Satellite Connector for deforestation alert cross-referencing
- PostGIS extension for spatial queries

**Estimated Effort:** 3 weeks (1 backend engineer, 1 GIS specialist)

---

#### Feature 6: Supplier Network & Relationship Analyzer

**User Story:**
```
As a supply chain risk analyst,
I want to map and analyze supplier-to-supplier relationships including sub-suppliers and intermediaries,
So that I can understand how risk propagates through the supply chain and identify hidden risk concentrations per EUDR Article 10(2)(a,b,f).
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F6.1: Maps supplier-to-supplier relationships (direct supplier -> sub-supplier -> sub-sub-supplier) as a directed graph, storing relationship type (sources_from, processes_for, trades_through, certifies), volume, and duration for each edge
- [ ] F6.2: Identifies supply chain depth and complexity per supplier: calculates tier depth (1-N), total sub-supplier count, unique country count, and intermediary count; computes complexity score (1-10) used as input to traceability_completeness risk factor
- [ ] F6.3: Detects circular and recursive supplier relationships (A -> B -> C -> A) that may indicate commodity laundering or data errors, flagging all suppliers involved in detected cycles for manual review
- [ ] F6.4: Implements risk propagation through supply chains: a supplier's risk score is influenced by its sub-suppliers' risk scores using configurable propagation formula: `propagated_risk = own_risk * 0.7 + max(sub_supplier_risks) * 0.3`; high-risk sub-suppliers infect their direct suppliers
- [ ] F6.5: Detects shared suppliers across importers (where configured for multi-tenant visibility): identifies suppliers that serve multiple operators, enabling industry-level risk intelligence and collaborative due diligence
- [ ] F6.6: Generates supplier consolidation recommendations identifying opportunities to reduce supply chain complexity: flag redundant intermediaries, suggest direct sourcing alternatives, and calculate estimated risk reduction from consolidation
- [ ] F6.7: Performs network graph analysis computing centrality metrics (degree centrality, betweenness centrality) to identify critical suppliers whose failure or non-compliance would have disproportionate impact on the operator's supply chain
- [ ] F6.8: Analyzes country-of-origin routing per supplier to detect potential transshipment: flags suppliers in low-risk countries whose declared origins do not match expected production patterns (e.g., Singapore-based palm oil supplier with no local production)
- [ ] F6.9: Computes intermediary risk amplification scoring: each intermediary in the supply chain between producer and importer adds a configurable risk increment (default: +5 points per intermediary layer, capped at +25), reflecting reduced traceability through longer chains
- [ ] F6.10: Scores ultimate source tracing capability per supplier (0-100): measures the percentage of supplied volume that can be traced back to specific production plots with verified geolocation, considering supply chain depth and data availability at each tier

**Non-Functional Requirements:**
- Scale: Support supplier networks with up to 50,000 nodes (suppliers + sub-suppliers)
- Performance: Network analysis < 5 seconds for 10,000-node supplier graph
- Memory: < 1 GB for 50,000-node network graph in memory

**Dependencies:**
- AGENT-EUDR-001 Supply Chain Mapping Master for graph infrastructure and traversal
- AGENT-EUDR-008 Multi-Tier Supplier Tracker for sub-supplier data
- Feature 1 (Supplier Composite Risk Scoring) for individual supplier risk scores
- NetworkX or equivalent graph library for network analysis

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 data engineer)

---

#### Feature 7: Supplier Monitoring & Alert Engine

**User Story:**
```
As a compliance officer,
I want continuous monitoring of all suppliers with automatic alerts when risk thresholds are breached,
So that I can respond proactively to emerging supplier risks rather than discovering them during periodic reviews.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F7.1: Implements continuous supplier risk monitoring at configurable frequencies: LOW risk suppliers (quarterly), MEDIUM risk (monthly), HIGH risk (weekly), CRITICAL risk (daily); frequencies adjustable per operator
- [ ] F7.2: Generates alerts when supplier risk score crosses configured threshold boundaries (e.g., supplier moves from MEDIUM to HIGH, or any factor score increases by > 15 points), with alert severity (info/warning/critical) and recommended actions
- [ ] F7.3: Implements change detection for supplier behavior and characteristics: monitors sourcing location changes, certification status changes, document expiry events, DD activity gaps, non-conformance trends, and volume pattern anomalies
- [ ] F7.4: Integrates news monitoring for supplier-related events: deforestation incidents, legal proceedings, sanctions, environmental violations, labor disputes, and bankruptcy filings; events sourced from configurable news feeds and matched to supplier entities using name and location matching
- [ ] F7.5: Performs sanction list screening against EU Consolidated List, OFAC SDN List, UN Security Council Sanctions, and World Bank Debarment List; screens all suppliers on initial scoring and at configurable intervals (default: weekly); any match generates a CRITICAL alert
- [ ] F7.6: Manages supplier watchlist: operators can manually flag suppliers for enhanced monitoring (increased frequency, additional alerts) with reason and review date; watchlisted suppliers receive CRITICAL-equivalent monitoring frequency regardless of current risk score
- [ ] F7.7: Implements automated re-assessment scheduling: when a supplier's risk score changes, the system automatically schedules the next DD activity based on the new risk level and creates a task in the DD tracker (Feature 2)
- [ ] F7.8: Generates portfolio-level risk heat map showing all suppliers positioned by risk score (x-axis) and volume/value (y-axis), with visual highlighting of concentration risk (too many high-risk suppliers in same country/commodity)
- [ ] F7.9: Computes portfolio risk aggregation metrics: weighted average portfolio risk score, portfolio Herfindahl-Hirschman Index (concentration risk by country, commodity, certification status), and portfolio Value-at-Risk (estimated financial exposure from non-compliant suppliers)
- [ ] F7.10: Maintains compliance calendar with supplier-specific milestones: DD review due dates, certification renewal deadlines, document submission deadlines, CAP milestones, and DDS filing dates; exportable as iCal feed with configurable reminder periods

**Non-Functional Requirements:**
- Latency: Alert delivery < 1 hour from trigger event detection
- Reliability: 99.9% uptime for monitoring engine
- Scale: Monitor up to 10,000 suppliers concurrently per operator
- Delivery: Alert channels: email, webhook, in-app notification

**Dependencies:**
- Feature 1 (Supplier Composite Risk Scoring) for threshold evaluation
- Feature 2 (DD History Tracker) for activity scheduling
- Feature 4 (Certification Validator) for expiry events
- GL-EUDR-APP notification service for alert delivery
- Sanction list data feeds (EU, OFAC, UN, World Bank)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 DevOps engineer)

---

#### Feature 8: Supplier Risk Reporting & Analytics

**User Story:**
```
As a compliance officer,
I want automated, audit-ready supplier risk reports and portfolio analytics,
So that I can include supplier risk data in my Due Diligence Statement, present to auditors, and make data-driven procurement decisions.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F8.1: Generates comprehensive individual supplier risk reports in PDF, JSON, and HTML formats, including composite score, all 8 factor scores, DD history summary, document completeness, certification status, geographic sourcing map, network context, trend chart, and recommendations
- [ ] F8.2: Generates portfolio-level risk analytics reports covering: total suppliers assessed, risk distribution (count per level), average portfolio risk score, top 10 highest-risk suppliers, most improved suppliers, most deteriorated suppliers, and concentration risk metrics
- [ ] F8.3: Produces comparative supplier analysis reports enabling side-by-side comparison of 2-10 suppliers across all 8 risk factors, with spider/radar charts for visual factor comparison and ranking table
- [ ] F8.4: Generates risk trend dashboards showing supplier risk score evolution over time (12-month, 24-month views), with annotated events (certification changes, DD activities, non-conformances, corrective actions) overlaid on the trend line
- [ ] F8.5: Produces regulatory submission formatting for Article 4 DDS: formats supplier risk assessment data into the structured format required for the Due Diligence Statement, including risk assessment methodology description, factor weights, data sources, and conclusions per supplier
- [ ] F8.6: Generates benchmarking reports comparing individual supplier risk against peer group (same commodity, same country, same size category), showing percentile rank, factor-by-factor comparison, and improvement recommendations
- [ ] F8.7: Produces executive summary reports with key portfolio risk indicators: compliance readiness score, portfolio risk trend, top risks requiring attention, estimated DD cost for next quarter, and progress against risk reduction targets
- [ ] F8.8: Exports risk matrices in multiple formats (CSV, JSON, XLSX, PDF) for integration with external BI tools (Grafana, Tableau, Power BI), with standardized schema for supplier risk metrics
- [ ] F8.9: Generates audit-ready documentation packages per supplier containing: risk assessment report, DD activity log, document inventory, certification records, geographic sourcing evidence, non-conformance history, and CAP status; all with SHA-256 provenance hashes
- [ ] F8.10: Tracks KPIs across the supplier risk program: number of suppliers assessed, percentage of portfolio scored, average assessment frequency, risk reduction trend (quarter-over-quarter), DD completion rates, and certification coverage improvement; KPIs displayed on dedicated analytics dashboard

**Non-Functional Requirements:**
- Performance: PDF generation < 5 seconds per individual supplier report; portfolio report < 30 seconds for 500 suppliers
- Quality: Reports pass WCAG 2.1 AA accessibility standards
- Size: PDF reports optimized to < 5 MB each
- Languages: Report generation in English (EN), French (FR), German (DE), Spanish (ES), Portuguese (PT)

**Dependencies:**
- Features 1-7 for all supplier risk assessment data
- Report generation library (WeasyPrint or ReportLab for PDF)
- Jinja2 templates for multi-format output
- i18n framework for multi-language support
- S3 object storage for generated reports

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend/template engineer)

---

### 6.4 Could-Have Features (P2 -- Nice to Have)

#### Feature 9: Supplier Self-Service Portal
- Supplier-facing web portal for self-service data entry
- Document upload workflow with guided requirements
- Certification upload and renewal management
- Sub-supplier declaration interface
- Real-time compliance status visibility for suppliers

#### Feature 10: Predictive Supplier Risk Forecasting
- Machine learning models to forecast supplier risk trajectory
- Early warning system for suppliers likely to deteriorate
- Scenario modeling for certification expiry impact
- Seasonal risk prediction based on historical patterns

#### Feature 11: Financial Risk Integration
- Integration with Dun & Bradstreet, Moody's, and Creditsafe
- Financial health scoring as enhanced input to financial_stability factor
- Bankruptcy prediction models
- Payment behavior correlation with compliance behavior

---

### 6.5 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Automated supplier deselection or contract termination (legal liability concern)
- Real-time satellite imagery processing per supplier (EUDR-003 provides this)
- Carbon footprint calculation per supplier (GL-GHG-APP scope)
- Social audit integration beyond basic compliance (CSDDD scope)
- Blockchain-based immutable supplier records (SHA-256 provenance hashes sufficient)
- Mobile native application (web responsive only)
- Direct EU Information System submission (GL-EUDR-APP DDS module)
- Supplier payment management or invoice processing

---

## 7. Technical Requirements

### 7.1 Architecture Overview

```
                                    +---------------------------+
                                    |     GL-EUDR-APP v1.0      |
                                    |   Frontend (React/TS)     |
                                    +-------------+-------------+
                                                  |
                                    +-------------v-------------+
                                    |     Unified API Layer      |
                                    |       (FastAPI)            |
                                    +-------------+-------------+
                                                  |
            +-------------------------------------+-------------------------------------+
            |                                     |                                     |
+-----------v-----------+           +-------------v-------------+           +-----------v-----------+
| AGENT-EUDR-017        |           | AGENT-EUDR-016            |           | AGENT-EUDR-001        |
| Supplier Risk         |<--------->| Country Risk              |<--------->| Supply Chain Mapping  |
| Scorer                |           | Evaluator                 |           | Master                |
|                       |           |                           |           |                       |
| - Risk Scoring Engine |           | - Country Risk Scoring    |           | - Graph Engine        |
| - DD History Tracker  |           | - Commodity Analyzer      |           | - Risk Propagation    |
| - Doc Analyzer        |           | - Hotspot Detector        |           | - Gap Analysis        |
| - Cert Validator      |           | - Governance Engine       |           | - Visualization       |
| - Geo Sourcing        |           | - DD Classifier           |           |                       |
| - Network Analyzer    |           | - Trade Flow Analyzer     |           +-----------------------+
| - Monitoring Engine   |           | - Report Generator        |
| - Report Engine       |           | - Regulatory Tracker      |           +---------------------------+
+-----------+-----------+           +---------------------------+           | AGENT-EUDR-008            |
            |                                                               | Multi-Tier Supplier       |
            |                                                               | Tracker                   |
+-----------v-----------+           +---------------------------+           | - Supplier Master Data    |
| Sanction Lists        |           | AGENT-DATA-006            |           | - Sub-tier Discovery      |
|                       |           | GIS/Mapping Connector     |           +---------------------------+
| - EU Consolidated     |           | - Spatial Operations      |
| - OFAC SDN            |           | - Protected Area DB       |           +---------------------------+
| - UN Sanctions        |           +---------------------------+           | AGENT-DATA-007            |
| - World Bank          |                                                   | Deforestation Satellite   |
+-----------------------+           +---------------------------+           | - GFW / GLAD / RADD      |
                                    | AGENT-DATA-001/002        |           +---------------------------+
                                    | PDF + Excel Extractors    |
                                    | - Document Processing     |
                                    +---------------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/supplier_risk_scorer/
    __init__.py                              # Public API exports
    config.py                                # SupplierRiskScorerConfig with GL_EUDR_SRS_ env prefix
    models.py                                # Pydantic v2 models for supplier risk data
    supplier_risk_scorer.py                  # SupplierRiskScorer: composite scoring engine (F1)
    due_diligence_tracker.py                 # DueDiligenceTracker: DD history tracking (F2)
    documentation_analyzer.py                # DocumentationAnalyzer: document completeness (F3)
    certification_validator.py               # CertificationValidator: certification validation (F4)
    geographic_sourcing_analyzer.py          # GeographicSourcingAnalyzer: geographic risk (F5)
    network_analyzer.py                      # NetworkAnalyzer: supplier relationships (F6)
    monitoring_alert_engine.py               # MonitoringAlertEngine: continuous monitoring (F7)
    risk_reporting_engine.py                 # RiskReportingEngine: reporting & analytics (F8)
    provenance.py                            # ProvenanceTracker: SHA-256 hash chains
    metrics.py                               # 18 Prometheus self-monitoring metrics
    setup.py                                 # SupplierRiskScorerService facade
    reference_data/
        __init__.py
        supplier_risk_database.py            # Default risk factor weights, thresholds, scoring rules
        certification_schemes.py             # Certification scheme validation rules and registries
        document_requirements.py             # EUDR document requirements matrix per supplier type
    api/
        __init__.py
        router.py                            # FastAPI router (~40 endpoints)
        schemas.py                           # API request/response Pydantic schemas
        dependencies.py                      # FastAPI dependency injection
        scoring_routes.py                    # Supplier risk scoring endpoints
        dd_routes.py                         # Due diligence history endpoints
        document_routes.py                   # Documentation analysis endpoints
        certification_routes.py              # Certification validation endpoints
        geographic_routes.py                 # Geographic sourcing endpoints
        network_routes.py                    # Network analysis endpoints
        monitoring_routes.py                 # Monitoring and alert endpoints
        reporting_routes.py                  # Reporting and analytics endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Supplier Risk Levels (4-tier)
class SupplierRiskLevel(str, Enum):
    LOW = "low"                 # 0-25
    MEDIUM = "medium"           # 25-50
    HIGH = "high"               # 50-75
    CRITICAL = "critical"       # 75-100

class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"

class ConfidenceLevel(str, Enum):
    HIGH = "high"               # 6+ of 8 factors with primary data
    MEDIUM = "medium"           # 4-5 factors with primary data
    LOW = "low"                 # < 4 factors with primary data

class DDActivityType(str, Enum):
    INITIAL_ASSESSMENT = "initial_assessment"
    PERIODIC_REVIEW = "periodic_review"
    ENHANCED_DD = "enhanced_dd"
    SITE_VISIT = "site_visit"
    QUESTIONNAIRE = "questionnaire"
    DOCUMENT_COLLECTION = "document_collection"
    SATELLITE_VERIFICATION = "satellite_verification"
    CORRECTIVE_ACTION = "corrective_action"
    ESCALATION = "escalation"
    CLOSURE = "closure"

class NonConformanceSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

class CertificationScheme(str, Enum):
    FSC = "fsc"
    PEFC = "pefc"
    RSPO = "rspo"
    RAINFOREST_ALLIANCE = "rainforest_alliance"
    UTZ = "utz"
    ISCC = "iscc"
    ORGANIC = "organic"
    FAIR_TRADE = "fair_trade"

# Supplier Risk Assessment
class SupplierRiskAssessment(BaseModel):
    assessment_id: str                          # UUID
    supplier_id: str                            # Reference to supplier master
    supplier_name: str
    composite_score: Decimal                    # 0-100 (Decimal for precision)
    risk_level: SupplierRiskLevel               # low/medium/high/critical
    factor_scores: Dict[str, Decimal]           # 8 factor scores (0-100 each)
    factor_weights: Dict[str, Decimal]          # 8 factor weights (sum = 1.0)
    confidence_score: Decimal                   # 0-100
    confidence_level: ConfidenceLevel           # high/medium/low
    trend: TrendDirection                       # improving/stable/deteriorating
    peer_percentile: Optional[int]              # 0-100 (vs peer group)
    peer_group_size: Optional[int]
    commodities: List[str]                      # EUDR commodities supplied
    countries: List[str]                        # ISO 3166-1 alpha-2 sourcing countries
    data_sources: List[DataSourceCitation]
    provenance_hash: str                        # SHA-256
    assessed_at: datetime
    version: int

# Due Diligence Activity Record
class DDActivityRecord(BaseModel):
    activity_id: str
    supplier_id: str
    activity_type: DDActivityType
    description: str
    actor: str                                  # Who performed the activity
    outcome: str                                # pass/fail/partial/pending
    duration_hours: Optional[Decimal]
    cost_eur: Optional[Decimal]
    evidence_refs: List[str]                    # Document/evidence references
    findings: Optional[str]
    non_conformances_found: int
    next_review_date: Optional[date]
    provenance_hash: str
    performed_at: datetime

# Non-Conformance Record
class NonConformanceRecord(BaseModel):
    nc_id: str
    supplier_id: str
    severity: NonConformanceSeverity
    eudr_article: str                           # Affected EUDR article
    description: str
    root_cause: str
    status: str                                 # open/in_progress/closed/recurring
    corrective_action_plan: Optional[str]
    cap_deadline: Optional[date]
    cap_milestones: List[Dict[str, Any]]
    resolved_at: Optional[datetime]
    detected_at: datetime

# Supplier Certification Record
class SupplierCertification(BaseModel):
    certification_id: str
    supplier_id: str
    scheme: CertificationScheme
    certificate_number: str
    scope_products: List[str]
    scope_regions: List[str]
    custody_model: Optional[str]                # FSC-COC, RSPO-SG, etc.
    certifying_body: str
    accreditation_verified: bool
    valid_from: date
    valid_to: date
    is_expired: bool
    coverage_volume_pct: Optional[Decimal]      # % of supplier volume covered
    effectiveness_score: Optional[Decimal]      # 0-100
    fraud_flags: List[str]                      # Any detected anomalies
    provenance_hash: str
    validated_at: datetime

# Geographic Sourcing Profile
class GeographicSourcingProfile(BaseModel):
    profile_id: str
    supplier_id: str
    sourcing_locations: List[Dict[str, Any]]    # lat, lon, region, country, volume_pct
    geographic_risk_score: Decimal              # 0-100
    concentration_hhi: Decimal                  # 0-10000 HHI
    hotspot_overlap_count: int
    protected_area_proximity_count: int
    indigenous_territory_overlap: bool
    deforestation_alert_count_12m: int
    direct_sourcing_pct: Decimal                # % direct vs indirect
    country_risk_scores: Dict[str, Decimal]     # country_code -> EUDR-016 risk score
    provenance_hash: str
    analyzed_at: datetime

# Supplier Network Node (for graph analysis)
class SupplierNetworkNode(BaseModel):
    node_id: str
    supplier_id: str
    supplier_name: str
    node_type: str                              # direct_supplier, sub_supplier, intermediary
    country_code: str
    risk_score: Optional[Decimal]
    tier_depth: int
    degree_centrality: Optional[float]
    betweenness_centrality: Optional[float]

# Alert Record
class SupplierAlert(BaseModel):
    alert_id: str
    supplier_id: str
    alert_type: str                             # threshold_breach, expiry, sanction, watchlist, etc.
    severity: str                               # info/warning/critical
    title: str
    description: str
    trigger_data: Dict[str, Any]
    recommended_actions: List[str]
    acknowledged: bool
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    created_at: datetime
```

### 7.4 Database Schema (New Migration: V105)

```sql
-- =========================================================================
-- V105: AGENT-EUDR-017 Supplier Risk Scorer Schema
-- Agent: GL-EUDR-SRS-017
-- Tables: 12 (3 hypertables, 2 continuous aggregates)
-- Prefix: gl_eudr_srs_
-- =========================================================================

CREATE SCHEMA IF NOT EXISTS eudr_supplier_risk_scorer;

-- 1. Supplier Risk Assessments (hypertable on assessed_at)
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_supplier_risks (
    assessment_id UUID DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    supplier_name VARCHAR(500) NOT NULL,
    composite_score NUMERIC(5,2) NOT NULL CHECK (composite_score >= 0 AND composite_score <= 100),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    geographic_sourcing_score NUMERIC(5,2) NOT NULL DEFAULT 50,
    compliance_history_score NUMERIC(5,2) NOT NULL DEFAULT 50,
    documentation_quality_score NUMERIC(5,2) NOT NULL DEFAULT 50,
    certification_status_score NUMERIC(5,2) NOT NULL DEFAULT 50,
    traceability_completeness_score NUMERIC(5,2) NOT NULL DEFAULT 50,
    financial_stability_score NUMERIC(5,2) NOT NULL DEFAULT 50,
    environmental_performance_score NUMERIC(5,2) NOT NULL DEFAULT 50,
    social_compliance_score NUMERIC(5,2) NOT NULL DEFAULT 50,
    factor_weights JSONB NOT NULL DEFAULT '{"geographic_sourcing":0.20,"compliance_history":0.15,"documentation_quality":0.15,"certification_status":0.15,"traceability_completeness":0.10,"financial_stability":0.10,"environmental_performance":0.10,"social_compliance":0.05}',
    confidence_score NUMERIC(5,2) NOT NULL DEFAULT 0,
    confidence_level VARCHAR(10) NOT NULL CHECK (confidence_level IN ('high', 'medium', 'low')),
    trend VARCHAR(20) NOT NULL CHECK (trend IN ('improving', 'stable', 'deteriorating')),
    peer_percentile INTEGER CHECK (peer_percentile >= 0 AND peer_percentile <= 100),
    peer_group_size INTEGER,
    commodities JSONB DEFAULT '[]',
    countries JSONB DEFAULT '[]',
    data_sources JSONB NOT NULL DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    version INTEGER DEFAULT 1,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (assessment_id, assessed_at)
);

SELECT create_hypertable('eudr_supplier_risk_scorer.gl_eudr_srs_supplier_risks', 'assessed_at');

-- 2. Due Diligence Activity Records (hypertable on performed_at)
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_dd_activities (
    activity_id UUID DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    activity_type VARCHAR(50) NOT NULL,
    description TEXT,
    actor VARCHAR(200) NOT NULL,
    outcome VARCHAR(20) NOT NULL CHECK (outcome IN ('pass', 'fail', 'partial', 'pending')),
    duration_hours NUMERIC(6,1),
    cost_eur NUMERIC(12,2),
    evidence_refs JSONB DEFAULT '[]',
    findings TEXT,
    non_conformances_found INTEGER DEFAULT 0,
    next_review_date DATE,
    provenance_hash VARCHAR(64) NOT NULL,
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (activity_id, performed_at)
);

SELECT create_hypertable('eudr_supplier_risk_scorer.gl_eudr_srs_dd_activities', 'performed_at');

-- 3. Non-Conformance Records
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_non_conformances (
    nc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'major', 'minor')),
    eudr_article VARCHAR(20),
    description TEXT NOT NULL,
    root_cause TEXT,
    status VARCHAR(20) NOT NULL CHECK (status IN ('open', 'in_progress', 'closed', 'recurring')),
    corrective_action_plan TEXT,
    cap_deadline DATE,
    cap_milestones JSONB DEFAULT '[]',
    resolved_at TIMESTAMPTZ,
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Supplier Certifications
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_certifications (
    certification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    scheme VARCHAR(50) NOT NULL,
    certificate_number VARCHAR(200) NOT NULL,
    scope_products JSONB DEFAULT '[]',
    scope_regions JSONB DEFAULT '[]',
    custody_model VARCHAR(50),
    certifying_body VARCHAR(300) NOT NULL,
    accreditation_verified BOOLEAN DEFAULT FALSE,
    valid_from DATE NOT NULL,
    valid_to DATE NOT NULL,
    is_expired BOOLEAN DEFAULT FALSE,
    coverage_volume_pct NUMERIC(5,2),
    effectiveness_score NUMERIC(5,2),
    fraud_flags JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (supplier_id, scheme, certificate_number)
);

-- 5. Supplier Documents
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    document_type VARCHAR(100) NOT NULL,
    document_name VARCHAR(500) NOT NULL,
    format VARCHAR(20) NOT NULL,
    file_path VARCHAR(1000),
    file_size_bytes BIGINT,
    language VARCHAR(10),
    effective_date DATE,
    expiry_date DATE,
    is_expired BOOLEAN DEFAULT FALSE,
    completeness_score NUMERIC(5,2),
    accuracy_score NUMERIC(5,2),
    consistency_score NUMERIC(5,2),
    version INTEGER DEFAULT 1,
    provenance_hash VARCHAR(64),
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. Geographic Sourcing Profiles
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_geographic_profiles (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    sourcing_locations JSONB NOT NULL DEFAULT '[]',
    geographic_risk_score NUMERIC(5,2) NOT NULL,
    concentration_hhi NUMERIC(8,2),
    hotspot_overlap_count INTEGER DEFAULT 0,
    protected_area_proximity_count INTEGER DEFAULT 0,
    indigenous_territory_overlap BOOLEAN DEFAULT FALSE,
    deforestation_alert_count_12m INTEGER DEFAULT 0,
    direct_sourcing_pct NUMERIC(5,2),
    country_risk_scores JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (supplier_id)
);

-- 7. Supplier Network Relationships
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_network_relationships (
    relationship_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_supplier_id UUID NOT NULL,
    target_supplier_id UUID NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,
    commodity VARCHAR(20),
    volume_pct NUMERIC(5,2),
    tier_depth INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_supplier_id, target_supplier_id, relationship_type)
);

-- 8. Supplier Alerts (hypertable on created_at)
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_alerts (
    alert_id UUID DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    trigger_data JSONB DEFAULT '{}',
    recommended_actions JSONB DEFAULT '[]',
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(200),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (alert_id, created_at)
);

SELECT create_hypertable('eudr_supplier_risk_scorer.gl_eudr_srs_alerts', 'created_at');

-- 9. Supplier Risk Score History
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_risk_history (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    previous_score NUMERIC(5,2),
    new_score NUMERIC(5,2) NOT NULL,
    previous_risk_level VARCHAR(20),
    new_risk_level VARCHAR(20) NOT NULL,
    affected_factors JSONB DEFAULT '[]',
    change_reason VARCHAR(500) NOT NULL,
    change_source VARCHAR(200),
    changed_by VARCHAR(200),
    provenance_hash VARCHAR(64),
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 10. Watchlist Entries
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_watchlist (
    watchlist_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    reason TEXT NOT NULL,
    added_by VARCHAR(200) NOT NULL,
    review_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (supplier_id)
);

-- 11. Generated Risk Reports
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    format VARCHAR(10) NOT NULL CHECK (format IN ('pdf', 'json', 'html', 'csv', 'xlsx')),
    language VARCHAR(5) NOT NULL DEFAULT 'en',
    supplier_ids JSONB DEFAULT '[]',
    parameters JSONB DEFAULT '{}',
    file_path VARCHAR(1000),
    file_size_bytes BIGINT,
    provenance_hash VARCHAR(64),
    generated_by VARCHAR(200),
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 12. Immutable Audit Log
CREATE TABLE eudr_supplier_risk_scorer.gl_eudr_srs_audit_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    actor VARCHAR(200) NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    previous_state JSONB,
    new_state JSONB,
    ip_address VARCHAR(45),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================================================================
-- Continuous Aggregates
-- =========================================================================

-- Daily supplier risk score averages
CREATE MATERIALIZED VIEW eudr_supplier_risk_scorer.gl_eudr_srs_daily_risk_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', assessed_at) AS bucket,
    risk_level,
    COUNT(*) AS assessment_count,
    AVG(composite_score) AS avg_composite_score,
    MIN(composite_score) AS min_composite_score,
    MAX(composite_score) AS max_composite_score,
    AVG(confidence_score) AS avg_confidence_score
FROM eudr_supplier_risk_scorer.gl_eudr_srs_supplier_risks
GROUP BY bucket, risk_level;

-- Daily alert aggregates
CREATE MATERIALIZED VIEW eudr_supplier_risk_scorer.gl_eudr_srs_daily_alert_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', created_at) AS bucket,
    severity,
    alert_type,
    COUNT(*) AS alert_count,
    COUNT(*) FILTER (WHERE acknowledged = TRUE) AS acknowledged_count
FROM eudr_supplier_risk_scorer.gl_eudr_srs_alerts
GROUP BY bucket, severity, alert_type;

-- =========================================================================
-- Indexes
-- =========================================================================

-- Supplier risks
CREATE INDEX idx_srs_risks_supplier ON eudr_supplier_risk_scorer.gl_eudr_srs_supplier_risks(supplier_id);
CREATE INDEX idx_srs_risks_level ON eudr_supplier_risk_scorer.gl_eudr_srs_supplier_risks(risk_level);
CREATE INDEX idx_srs_risks_score ON eudr_supplier_risk_scorer.gl_eudr_srs_supplier_risks(composite_score);
CREATE INDEX idx_srs_risks_confidence ON eudr_supplier_risk_scorer.gl_eudr_srs_supplier_risks(confidence_level);

-- DD activities
CREATE INDEX idx_srs_dd_supplier ON eudr_supplier_risk_scorer.gl_eudr_srs_dd_activities(supplier_id);
CREATE INDEX idx_srs_dd_type ON eudr_supplier_risk_scorer.gl_eudr_srs_dd_activities(activity_type);
CREATE INDEX idx_srs_dd_outcome ON eudr_supplier_risk_scorer.gl_eudr_srs_dd_activities(outcome);

-- Non-conformances
CREATE INDEX idx_srs_nc_supplier ON eudr_supplier_risk_scorer.gl_eudr_srs_non_conformances(supplier_id);
CREATE INDEX idx_srs_nc_severity ON eudr_supplier_risk_scorer.gl_eudr_srs_non_conformances(severity);
CREATE INDEX idx_srs_nc_status ON eudr_supplier_risk_scorer.gl_eudr_srs_non_conformances(status);

-- Certifications
CREATE INDEX idx_srs_cert_supplier ON eudr_supplier_risk_scorer.gl_eudr_srs_certifications(supplier_id);
CREATE INDEX idx_srs_cert_scheme ON eudr_supplier_risk_scorer.gl_eudr_srs_certifications(scheme);
CREATE INDEX idx_srs_cert_expiry ON eudr_supplier_risk_scorer.gl_eudr_srs_certifications(valid_to);
CREATE INDEX idx_srs_cert_expired ON eudr_supplier_risk_scorer.gl_eudr_srs_certifications(is_expired) WHERE is_expired = TRUE;

-- Documents
CREATE INDEX idx_srs_doc_supplier ON eudr_supplier_risk_scorer.gl_eudr_srs_documents(supplier_id);
CREATE INDEX idx_srs_doc_type ON eudr_supplier_risk_scorer.gl_eudr_srs_documents(document_type);
CREATE INDEX idx_srs_doc_expiry ON eudr_supplier_risk_scorer.gl_eudr_srs_documents(expiry_date);

-- Geographic profiles
CREATE INDEX idx_srs_geo_supplier ON eudr_supplier_risk_scorer.gl_eudr_srs_geographic_profiles(supplier_id);
CREATE INDEX idx_srs_geo_risk ON eudr_supplier_risk_scorer.gl_eudr_srs_geographic_profiles(geographic_risk_score);

-- Network relationships
CREATE INDEX idx_srs_net_source ON eudr_supplier_risk_scorer.gl_eudr_srs_network_relationships(source_supplier_id);
CREATE INDEX idx_srs_net_target ON eudr_supplier_risk_scorer.gl_eudr_srs_network_relationships(target_supplier_id);
CREATE INDEX idx_srs_net_type ON eudr_supplier_risk_scorer.gl_eudr_srs_network_relationships(relationship_type);

-- Alerts
CREATE INDEX idx_srs_alert_supplier ON eudr_supplier_risk_scorer.gl_eudr_srs_alerts(supplier_id);
CREATE INDEX idx_srs_alert_severity ON eudr_supplier_risk_scorer.gl_eudr_srs_alerts(severity);
CREATE INDEX idx_srs_alert_ack ON eudr_supplier_risk_scorer.gl_eudr_srs_alerts(acknowledged) WHERE acknowledged = FALSE;

-- Risk history
CREATE INDEX idx_srs_history_supplier ON eudr_supplier_risk_scorer.gl_eudr_srs_risk_history(supplier_id);
CREATE INDEX idx_srs_history_changed ON eudr_supplier_risk_scorer.gl_eudr_srs_risk_history(changed_at);

-- Watchlist
CREATE INDEX idx_srs_watchlist_active ON eudr_supplier_risk_scorer.gl_eudr_srs_watchlist(is_active) WHERE is_active = TRUE;

-- Reports
CREATE INDEX idx_srs_reports_type ON eudr_supplier_risk_scorer.gl_eudr_srs_reports(report_type);

-- Audit log
CREATE INDEX idx_srs_audit_entity ON eudr_supplier_risk_scorer.gl_eudr_srs_audit_log(entity_type, entity_id);
CREATE INDEX idx_srs_audit_actor ON eudr_supplier_risk_scorer.gl_eudr_srs_audit_log(actor);
CREATE INDEX idx_srs_audit_created ON eudr_supplier_risk_scorer.gl_eudr_srs_audit_log(created_at);
```

### 7.5 API Endpoints (~40)

| Method | Path | Description |
|--------|------|-------------|
| **Supplier Risk Scoring** | | |
| POST | `/v1/eudr-srs/suppliers/assess` | Assess risk for a single supplier (full composite scoring) |
| POST | `/v1/eudr-srs/suppliers/assess-batch` | Batch assess risk for multiple suppliers |
| GET | `/v1/eudr-srs/suppliers/{supplier_id}/risk` | Get latest risk assessment for a supplier |
| GET | `/v1/eudr-srs/suppliers` | List all supplier risk assessments (with filters: risk_level, commodity, country) |
| GET | `/v1/eudr-srs/suppliers/{id1}/compare/{id2}` | Compare two suppliers side-by-side |
| GET | `/v1/eudr-srs/suppliers/{supplier_id}/trends` | Get risk score trend over time |
| GET | `/v1/eudr-srs/suppliers/{supplier_id}/factors` | Get detailed factor score breakdown |
| **Due Diligence History** | | |
| POST | `/v1/eudr-srs/dd-activities` | Record a new DD activity |
| GET | `/v1/eudr-srs/dd-activities/{supplier_id}` | Get DD activity history for a supplier |
| GET | `/v1/eudr-srs/dd-activities/{supplier_id}/completeness` | Get DD completion rate and quality score |
| GET | `/v1/eudr-srs/dd-activities/{supplier_id}/gaps` | Get DD coverage gaps |
| **Non-Conformances** | | |
| POST | `/v1/eudr-srs/non-conformances` | Record a new non-conformance |
| GET | `/v1/eudr-srs/non-conformances/{supplier_id}` | Get non-conformances for a supplier |
| PATCH | `/v1/eudr-srs/non-conformances/{nc_id}` | Update non-conformance status/CAP |
| **Documentation** | | |
| POST | `/v1/eudr-srs/documents/analyze` | Analyze document completeness for a supplier |
| GET | `/v1/eudr-srs/documents/{supplier_id}` | Get document inventory for a supplier |
| GET | `/v1/eudr-srs/documents/{supplier_id}/gaps` | Get missing document analysis |
| POST | `/v1/eudr-srs/documents/{supplier_id}/request` | Generate document request for supplier |
| **Certifications** | | |
| POST | `/v1/eudr-srs/certifications/validate` | Validate a supplier certification |
| GET | `/v1/eudr-srs/certifications/{supplier_id}` | Get certifications for a supplier |
| GET | `/v1/eudr-srs/certifications/expiring` | List certifications expiring within period |
| GET | `/v1/eudr-srs/certifications/{supplier_id}/score` | Get aggregated certification score |
| **Geographic Sourcing** | | |
| POST | `/v1/eudr-srs/geographic/analyze` | Analyze geographic sourcing for a supplier |
| GET | `/v1/eudr-srs/geographic/{supplier_id}` | Get geographic sourcing profile |
| GET | `/v1/eudr-srs/geographic/{supplier_id}/hotspots` | Get hotspot overlaps for supplier sourcing |
| GET | `/v1/eudr-srs/geographic/{supplier_id}/changes` | Get sourcing pattern changes |
| **Network Analysis** | | |
| POST | `/v1/eudr-srs/network/analyze` | Analyze supplier network relationships |
| GET | `/v1/eudr-srs/network/{supplier_id}` | Get supplier network graph |
| GET | `/v1/eudr-srs/network/{supplier_id}/propagation` | Get risk propagation analysis |
| GET | `/v1/eudr-srs/network/centrality` | Get critical supplier identification |
| **Monitoring & Alerts** | | |
| GET | `/v1/eudr-srs/alerts` | List all active alerts (with filters) |
| GET | `/v1/eudr-srs/alerts/{alert_id}` | Get alert details |
| PATCH | `/v1/eudr-srs/alerts/{alert_id}/acknowledge` | Acknowledge an alert |
| POST | `/v1/eudr-srs/watchlist` | Add supplier to watchlist |
| GET | `/v1/eudr-srs/watchlist` | List watchlisted suppliers |
| DELETE | `/v1/eudr-srs/watchlist/{supplier_id}` | Remove supplier from watchlist |
| GET | `/v1/eudr-srs/portfolio/heatmap` | Get portfolio risk heat map data |
| **Reports** | | |
| POST | `/v1/eudr-srs/reports/generate` | Generate a supplier risk report |
| GET | `/v1/eudr-srs/reports/{report_id}` | Get report metadata |
| GET | `/v1/eudr-srs/reports/{report_id}/download` | Download report file |
| GET | `/v1/eudr-srs/reports/kpis` | Get supplier risk program KPIs |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_srs_assessments_total` | Counter | Supplier risk assessments performed |
| 2 | `gl_eudr_srs_batch_assessments_total` | Counter | Batch portfolio assessments completed |
| 3 | `gl_eudr_srs_dd_activities_total` | Counter | DD activities recorded by type |
| 4 | `gl_eudr_srs_non_conformances_total` | Counter | Non-conformances recorded by severity |
| 5 | `gl_eudr_srs_certifications_validated_total` | Counter | Certifications validated by scheme |
| 6 | `gl_eudr_srs_alerts_generated_total` | Counter | Alerts generated by severity and type |
| 7 | `gl_eudr_srs_reports_generated_total` | Counter | Reports generated by type and format |
| 8 | `gl_eudr_srs_api_errors_total` | Counter | API errors by endpoint and status code |
| 9 | `gl_eudr_srs_assessment_duration_seconds` | Histogram | Single supplier assessment latency |
| 10 | `gl_eudr_srs_batch_assessment_duration_seconds` | Histogram | Batch assessment latency |
| 11 | `gl_eudr_srs_certification_validation_duration_seconds` | Histogram | Certification validation latency |
| 12 | `gl_eudr_srs_network_analysis_duration_seconds` | Histogram | Network analysis latency |
| 13 | `gl_eudr_srs_report_generation_duration_seconds` | Histogram | Report generation latency |
| 14 | `gl_eudr_srs_active_suppliers` | Gauge | Total suppliers with active risk assessments |
| 15 | `gl_eudr_srs_critical_risk_suppliers` | Gauge | Suppliers classified as CRITICAL risk |
| 16 | `gl_eudr_srs_high_risk_suppliers` | Gauge | Suppliers classified as HIGH risk |
| 17 | `gl_eudr_srs_unacknowledged_alerts` | Gauge | Alerts pending acknowledgment |
| 18 | `gl_eudr_srs_expired_certifications` | Gauge | Supplier certifications currently expired |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for risk history |
| Spatial | PostGIS + Shapely | Geographic sourcing analysis, hotspot intersection |
| Cache | Redis | Supplier risk score caching, assessment result caching |
| Object Storage | S3 | Generated reports, uploaded documents |
| Graph | NetworkX | Supplier network analysis, centrality computation |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Arithmetic | Python Decimal | Precision for risk score calculations (no floating-point drift) |
| PDF Generation | WeasyPrint | HTML-to-PDF for report generation |
| Templates | Jinja2 | Multi-format report templates (HTML/PDF) |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based supplier risk data access control |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-srs:suppliers:read` | View supplier risk assessments | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-srs:suppliers:assess` | Trigger supplier risk assessments | Analyst, Compliance Officer, Admin |
| `eudr-srs:suppliers:compare` | Compare supplier risk profiles | Analyst, Compliance Officer, Admin |
| `eudr-srs:dd-activities:read` | View DD activity history | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-srs:dd-activities:write` | Record DD activities | Analyst, Compliance Officer, Admin |
| `eudr-srs:non-conformances:read` | View non-conformance records | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-srs:non-conformances:write` | Record and update non-conformances | Analyst, Compliance Officer, Admin |
| `eudr-srs:documents:read` | View document inventory and analysis | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-srs:documents:analyze` | Trigger document completeness analysis | Analyst, Compliance Officer, Admin |
| `eudr-srs:documents:request` | Generate document requests to suppliers | Compliance Officer, Admin |
| `eudr-srs:certifications:read` | View certification records | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-srs:certifications:validate` | Trigger certification validation | Analyst, Compliance Officer, Admin |
| `eudr-srs:geographic:read` | View geographic sourcing profiles | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-srs:geographic:analyze` | Trigger geographic sourcing analysis | Analyst, Compliance Officer, Admin |
| `eudr-srs:network:read` | View supplier network analysis | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-srs:network:analyze` | Trigger network analysis | Analyst, Compliance Officer, Admin |
| `eudr-srs:alerts:read` | View supplier alerts | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-srs:alerts:manage` | Acknowledge and manage alerts | Analyst, Compliance Officer, Admin |
| `eudr-srs:watchlist:read` | View supplier watchlist | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-srs:watchlist:manage` | Add/remove suppliers from watchlist | Compliance Officer, Admin |
| `eudr-srs:reports:read` | View generated reports | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-srs:reports:generate` | Generate supplier risk reports | Analyst, Compliance Officer, Admin |
| `eudr-srs:config:manage` | Manage risk factor weights, thresholds, and monitoring config | Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent/Source | Integration | Data Flow |
|-------------|-------------|-----------|
| AGENT-EUDR-016 Country Risk Evaluator | Risk data feed | Country risk scores, commodity risk, sub-national hotspots -> geographic_sourcing factor (F5) |
| AGENT-EUDR-008 Multi-Tier Supplier Tracker | Supplier master data | Supplier records, sub-tier relationships, activity history -> F1, F2, F6 |
| AGENT-EUDR-001 Supply Chain Mapping Master | Graph context | Supply chain graph, node risk, custody chain -> F6 network analysis |
| AGENT-DATA-006 GIS/Mapping Connector | Spatial operations | Protected area proximity, indigenous territory overlap -> F5 |
| AGENT-DATA-007 Deforestation Satellite Connector | Deforestation alerts | GLAD/RADD alerts for sourcing location cross-reference -> F5 |
| AGENT-DATA-001 PDF Extractor | Document processing | Extract content from uploaded supplier PDF documents -> F3 |
| AGENT-DATA-002 Excel/CSV Normalizer | Data extraction | Extract structured data from supplier spreadsheets -> F3 |
| AGENT-FOUND-005 Citations & Evidence | Source tracking | Citation generation and evidence linking -> all features |
| AGENT-FOUND-008 Reproducibility Agent | Determinism verification | Bit-perfect verification of risk calculations -> F1 |
| Sanction list feeds | External data | EU/OFAC/UN/World Bank sanction screening -> F7 |
| Certification registries | External APIs | FSC/RSPO/PEFC public certificate databases -> F4 |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-001 Supply Chain Mapping Master | Risk data feed | Supplier risk scores -> supply chain graph node risk attributes |
| GL-EUDR-APP v1.0 Platform | API integration | Supplier risk dashboards, portfolio analytics, alert display |
| GL-EUDR-APP DDS Reporting Engine | Risk assessment section | Supplier risk data formatted for DDS Article 4(2) submission |
| External Auditors | Read-only API + reports | Supplier risk exports for third-party verification |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Supplier Risk Assessment (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Supplier Risk" module -> "Risk Scoring" tab
3. Selects supplier (e.g., "PT Sawit Lestari, Indonesia") from supplier list
4. System displays comprehensive supplier risk profile:
   - Composite score: 68 (HIGH)
   - Factor breakdown:
     - Geographic sourcing: 82 (sources from West Kalimantan hotspot)
     - Compliance history: 55 (2 open non-conformances)
     - Documentation quality: 45 (missing harvesting dates, incomplete geolocation)
     - Certification status: 35 (RSPO-MB valid, covers 70% volume)
     - Traceability completeness: 72 (3-tier depth, 2 intermediaries)
     - Financial stability: 60 (stable, medium-size operation)
     - Environmental performance: 78 (deforestation alert proximity)
     - Social compliance: 50 (no reported issues, limited data)
   - Confidence: 75 (MEDIUM -- missing financial and social data)
   - Trend: DETERIORATING (score increased 12 points in 6 months)
   - Peer percentile: 78th (higher risk than 78% of Indonesian palm oil suppliers)
5. Officer reviews recommendations:
   - "Collect missing geolocation data for 3 sourcing locations"
   - "Schedule satellite verification for West Kalimantan sourcing area"
   - "Request updated compliance declaration (expired 45 days ago)"
6. Officer clicks "Generate Report" -> selects PDF format
7. System generates supplier risk report in < 5 seconds
8. Officer includes report in quarterly DDS submission
```

#### Flow 2: Portfolio Risk Review (Risk Analyst)

```
1. Risk analyst logs in to GL-EUDR-APP
2. Navigates to "Supplier Risk" module -> "Portfolio Analytics" tab
3. System displays portfolio dashboard:
   - Total suppliers: 420
   - Risk distribution: 85 LOW, 180 MEDIUM, 120 HIGH, 35 CRITICAL
   - Average portfolio risk: 47 (MEDIUM)
   - Concentration risk: HHI = 2,400 (moderate concentration in Indonesia)
4. Analyst clicks "CRITICAL Risk Suppliers" filter
5. System shows 35 critical-risk suppliers sorted by composite score
6. Analyst selects "View Heat Map" -> sees suppliers plotted by risk vs. volume
7. Identifies 5 high-volume, critical-risk suppliers requiring immediate attention
8. Exports comparative analysis report for management review
```

#### Flow 3: Certification Expiry Management (Procurement Manager)

```
1. Procurement manager logs in to GL-EUDR-APP
2. Navigates to "Supplier Risk" module -> "Certifications" tab
3. System displays certification dashboard:
   - Certifications expiring in 30 days: 12
   - Certifications expiring in 90 days: 38
   - Currently expired: 5 (RED alert)
4. Manager clicks on expired certifications
5. System shows 5 suppliers with expired certifications:
   - Risk scores already recalculated (certification factor = max risk)
   - Automated renewal requests already sent (14 days ago)
6. Manager escalates 2 non-responsive suppliers to compliance officer
7. Reviews suppliers with certifications expiring in 30 days
8. Triggers bulk document request for renewal evidence
```

### 8.2 Key Screen Descriptions

**Supplier Risk Dashboard:**
- Left panel: supplier search and filter (by risk level, commodity, country, certification status)
- Main panel: selected supplier risk profile with radar chart (8 factors)
- Right panel: quick actions (assess, generate report, add to watchlist)
- Bottom panel: risk trend chart (12-month history) with annotated events

**Portfolio Analytics View:**
- Top bar: KPI summary cards (total suppliers, average risk, critical count, alert count)
- Main panel: risk distribution chart (bar chart by risk level) + heat map (risk vs. volume)
- Filter panel: commodity, country, risk level, certification scheme
- Table: sortable supplier list with key metrics

**Certification Management Dashboard:**
- Expiry timeline: visual timeline showing certification expiry dates
- Scheme distribution: pie chart of certifications by scheme
- Alert panel: expired and expiring certifications with action buttons
- Coverage analysis: certified vs. uncertified volume per supplier

**Alert Center:**
- Alert feed: chronological list of alerts with severity indicators
- Filter: by severity, type, supplier, date range
- Bulk acknowledge: select multiple alerts for acknowledgment
- Watchlist panel: currently watchlisted suppliers with monitoring status

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 8 P0 features (Features 1-8) implemented and tested
  - [ ] Feature 1: Supplier Composite Risk Scoring Engine -- 8-factor weighted composite, 4-tier classification, peer benchmarking
  - [ ] Feature 2: Due Diligence History Tracker -- activity recording, non-conformance tracking, escalation workflows
  - [ ] Feature 3: Documentation Completeness Analyzer -- EUDR document matrix, gap analysis, expiry tracking
  - [ ] Feature 4: Certification & Standards Validator -- 8 schemes, scope verification, chain of custody, fraud detection
  - [ ] Feature 5: Geographic Sourcing Pattern Analyzer -- EUDR-016 integration, hotspot cross-reference, protected areas
  - [ ] Feature 6: Supplier Network & Relationship Analyzer -- graph analysis, risk propagation, centrality metrics
  - [ ] Feature 7: Supplier Monitoring & Alert Engine -- continuous monitoring, sanction screening, watchlist
  - [ ] Feature 8: Supplier Risk Reporting & Analytics -- PDF/JSON/HTML, portfolio analytics, audit packages
- [ ] >= 85% test coverage achieved (600+ tests)
- [ ] Security audit passed (JWT + RBAC integrated, 23 permissions)
- [ ] Performance targets met (< 200ms per assessment, < 30s for 1,000 batch, < 5s per report)
- [ ] Risk scores validated deterministic (bit-perfect reproducibility)
- [ ] All 8 risk factors tested across 50+ supplier profiles with known baselines
- [ ] Certification validation tested for all 8 schemes with valid and invalid certificates
- [ ] Geographic sourcing integration validated with EUDR-016 for 20+ countries
- [ ] Supplier network analysis validated with graphs of 1,000+ nodes
- [ ] API documentation complete (OpenAPI spec, ~40 endpoints)
- [ ] Database migration V105 tested and validated
- [ ] Integration with EUDR-016, EUDR-001, EUDR-008 verified
- [ ] 5 beta customers successfully using supplier risk scoring
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 500+ supplier risk assessments consumed by customers
- 100+ DD activities tracked
- 200+ supplier risk reports generated
- Average assessment latency < 200ms (p99)
- Deterministic reproducibility maintained at 100%
- < 5 support tickets per customer

**60 Days:**
- 2,000+ unique suppliers assessed across customer base
- 500+ certification validations performed
- 100+ alerts generated and actioned
- Geographic sourcing analysis active for 15+ countries
- 3+ portfolio-level analytics reports per customer
- NPS > 45 from compliance officer persona

**90 Days:**
- 5,000+ active supplier risk profiles maintained
- 1,000+ DD activities tracked with complete audit trail
- 300+ non-conformances tracked with CAP management
- Continuous monitoring operational for all customer supplier portfolios
- Zero EUDR penalties attributable to undetected supplier risk for active customers
- Full integration with GL-EUDR-APP DDS workflow operational
- NPS > 55

---

## 10. Timeline and Milestones

### Phase 1: Core Risk Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Supplier Composite Risk Scoring Engine (Feature 1): 8-factor composite, 4-tier classification, batch scoring | Senior Backend Engineer |
| 2-3 | Due Diligence History Tracker (Feature 2): activity recording, non-conformance tracking, escalation | Backend Engineer |
| 3-4 | Documentation Completeness Analyzer (Feature 3): document matrix, gap analysis, expiry tracking | Backend Engineer |
| 4-5 | Certification & Standards Validator (Feature 4): 8 schemes, scope verification, fraud detection | Data Engineer |
| 5-6 | Geographic Sourcing Pattern Analyzer (Feature 5): EUDR-016 integration, hotspot cross-reference | GIS Specialist |

**Milestone: Core supplier risk assessment engine operational with 5 core features (Week 6)**

### Phase 2: Network, Monitoring, API, and Reporting (Weeks 7-11)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Supplier Network & Relationship Analyzer (Feature 6): graph analysis, risk propagation | Senior Backend Engineer |
| 8-9 | Supplier Monitoring & Alert Engine (Feature 7): continuous monitoring, sanction screening | Backend + DevOps |
| 9-10 | REST API Layer: ~40 endpoints, authentication, rate limiting | Backend Engineer |
| 10-11 | Supplier Risk Reporting & Analytics (Feature 8): PDF/JSON/HTML, multi-language, portfolio | Backend + Template Engineer |

**Milestone: Full API operational with network analysis, monitoring, and reporting (Week 11)**

### Phase 3: Integration, Testing, and Hardening (Weeks 12-15)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 12 | Integration with EUDR-016, EUDR-001, EUDR-008 | Backend Engineer |
| 12-13 | RBAC integration (23 permissions), Prometheus metrics (18), Grafana dashboard | DevOps + Backend |
| 13-14 | Complete test suite: 600+ tests, golden tests, integration tests, determinism tests | Test Engineer |
| 14-15 | Performance testing, load testing (1,000-supplier batch), security audit | DevOps + Security |

**Milestone: All 8 P0 features implemented with full integration and test coverage (Week 15)**

### Phase 4: Validation and Launch (Weeks 16-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 16 | Certification validation against real scheme registries (FSC, RSPO) | Data Engineer |
| 16-17 | Database migration V105 finalized and tested | DevOps |
| 17 | Beta customer onboarding (5 customers) with real supplier data | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 8 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Supplier self-service portal (Feature 9)
- Predictive supplier risk forecasting (Feature 10)
- Financial risk integration with D&B/Moody's (Feature 11)
- Expanded certification scheme coverage (additional 5+ schemes)
- Multi-language supplier communication templates (10+ languages)

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Provides country/commodity/hotspot risk data; well-defined API |
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Provides supply chain graph context for network analysis |
| AGENT-EUDR-008 Multi-Tier Supplier Tracker | BUILT | Low | Provides supplier master data and sub-tier relationships |
| AGENT-DATA-006 GIS/Mapping Connector | BUILT (100%) | Low | Spatial operations for geographic sourcing analysis |
| AGENT-DATA-007 Deforestation Satellite Connector | BUILT (100%) | Low | Deforestation alerts for sourcing location cross-reference |
| AGENT-DATA-001 PDF & Invoice Extractor | BUILT (100%) | Low | Document content extraction for completeness analysis |
| AGENT-DATA-002 Excel/CSV Normalizer | BUILT (100%) | Low | Structured data extraction from supplier documents |
| AGENT-FOUND-005 Citations & Evidence Agent | BUILT (100%) | Low | Source attribution for all risk assessments |
| AGENT-FOUND-008 Reproducibility Agent | BUILT (100%) | Low | Determinism verification for risk calculations |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Integration target for UI and DDS workflow |
| PostgreSQL + TimescaleDB + PostGIS | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| FSC Certificate Database (info.fsc.org) | Available | Medium | Public API; cached data; fallback to manual verification |
| RSPO Public Registry | Available | Medium | Web scraping with cache; fallback to manual verification |
| PEFC Chain of Custody Registry | Available | Low | Cached data; periodic refresh |
| EU Consolidated Sanctions List | Published | Low | Daily update feed; cached data |
| OFAC SDN List | Published | Low | Daily update feed; cached data |
| UN Security Council Sanctions | Published | Low | Periodic update; cached data |
| EC EUDR implementing regulations | Evolving | Medium | Configuration-driven compliance rules |
| Certification body accreditation registries | Varies | Medium | Periodic bulk download; cached data; manual override |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Supplier data quality is poor or incomplete, reducing confidence of risk scores | High | High | Confidence scoring flags low-data suppliers; gap analysis generates targeted data collection requests; partial scoring with re-weighting |
| R2 | Certification scheme registries become unavailable or change access terms | Medium | Medium | Offline cached data; multi-source fallback; adapter pattern isolates integration; manual override for urgent validations |
| R3 | Supplier volume exceeds performance targets (> 5,000 suppliers per operator) | Medium | Medium | Horizontal scaling with K8s; Redis caching for hot scores; batch processing with async queue; progressive loading |
| R4 | Geographic sourcing data from suppliers is inaccurate or deliberately misleading | Medium | High | Cross-reference with satellite data (EUDR-003); flag inconsistencies between declared locations and trade flow patterns; reduce confidence for unverified locations |
| R5 | Sanction list screening generates excessive false positives | Medium | Medium | Tunable matching thresholds; fuzzy match with configurable confidence floor; human-in-the-loop review for borderline matches |
| R6 | Supplier network analysis at scale causes memory or performance issues | Low | Medium | Lazy graph loading; limit graph depth for real-time queries; pre-computed centrality metrics; async background analysis for large networks |
| R7 | Operators customize risk factor weights to game the system (minimize reported risk) | Low | Medium | Enforce minimum weight bounds (2% per factor); audit log all weight changes; flag deviations from default; regulatory compliance mode locks weights to recommended values |
| R8 | Integration complexity with 5+ upstream EUDR/DATA agents | Medium | Medium | Well-defined interfaces; mock adapters for testing; circuit breaker pattern; graceful degradation when upstream unavailable |
| R9 | Multi-language report generation quality varies across languages | Low | Low | Professional template review for all 5 languages; user feedback loop; fallback to English when translation quality insufficient |
| R10 | Regulatory guidance on supplier DD methodology changes after launch | Medium | Medium | Configuration-driven scoring rules; modular factor weights; rapid reconfiguration without code deployment; regulatory update monitoring via EUDR-016 F8 |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Supplier Risk Scoring Tests | 100+ | Composite calculation, 8-factor weighting, classification thresholds, confidence scoring, trend analysis, peer benchmarking, batch scoring, edge cases |
| DD History Tracker Tests | 60+ | Activity recording, completion rates, non-conformance tracking, CAP monitoring, escalation workflows, readiness scoring |
| Documentation Analyzer Tests | 60+ | Document matrix, gap analysis, expiry tracking, cross-reference validation, multi-format acceptance, quality scoring |
| Certification Validator Tests | 70+ | All 8 schemes, scope verification, chain of custody, expiry detection, fraud heuristics, multi-scheme aggregation, coverage analysis |
| Geographic Sourcing Tests | 50+ | EUDR-016 integration, hotspot cross-reference, concentration HHI, protected area proximity, deforestation alert matching |
| Network Analyzer Tests | 50+ | Graph construction, risk propagation, cycle detection, centrality metrics, intermediary amplification, shared supplier detection |
| Monitoring & Alert Tests | 40+ | Threshold alerts, sanction screening, watchlist management, monitoring frequencies, portfolio metrics, compliance calendar |
| Reporting & Analytics Tests | 40+ | All formats (PDF/JSON/HTML), multi-language, portfolio analytics, KPI calculation, audit packages, DDS formatting |
| API Tests | 50+ | All ~40 endpoints, auth, error handling, pagination, rate limiting, batch operations |
| Integration Tests | 30+ | Cross-agent integration with EUDR-016/001/008, DATA-006/007/001/002 |
| Performance Tests | 20+ | Single assessment, batch 1,000, report generation, concurrent queries, network analysis at scale |
| Golden Tests | 50+ | Known supplier risk scenarios (25 suppliers x 2 scenarios each) |
| Determinism Tests | 30+ | Bit-perfect reproducibility across runs, Python versions, Decimal arithmetic verification |
| **Total** | **650+** | |

### 13.2 Golden Test Scenarios

Each of the following 25 representative supplier profiles will have 2 golden test scenarios:

**Critical-Risk Suppliers (5):** Uncertified palm oil supplier in West Kalimantan; cocoa trader with 4 open non-conformances in Cote d'Ivoire; timber supplier in DRC with no geolocation data; soya exporter in Cerrado with deforestation alerts; rubber collector in Cambodia with expired certifications

**High-Risk Suppliers (7):** RSPO-MB palm oil supplier with 60% coverage; FSC-CW wood supplier with scope mismatch; cocoa cooperative in Ghana with incomplete documentation; coffee exporter in Vietnam with 3 intermediaries; cattle supplier in Bolivia with protected area proximity; soya trader in Paraguay with sourcing concentration; palm oil mill in Sumatra with seasonal deforestation correlation

**Medium-Risk Suppliers (7):** RSPO-SG certified palm oil supplier in Malaysia; FSC-COC certified timber supplier in Brazil (Atlantic Forest); Rainforest Alliance cocoa cooperative in Ecuador; UTZ coffee farm in Colombia; rubber plantation in Thailand with complete documentation; soya farm in Argentina with full geolocation; cattle ranch in Australia with low country risk

**Low-Risk Suppliers (6):** FSC-certified timber supplier in Finland; organic coffee cooperative in Ethiopia with Rainforest Alliance; RSPO-IP palm oil supplier in Papua New Guinea; certified soya from United States; wood supplier in Canada with full traceability; rubber from sustainable plantation in Malaysia with all certifications valid

**Scenarios per supplier:**
1. **Full assessment** -- Complete 8-factor composite scoring with all factor sub-scores -> expect exact score match with golden baseline
2. **Score change trigger** -- Simulate data change (certification expiry, new non-conformance, sourcing location change) -> expect correct re-scored value, correct risk level transition, correct alert generation

Total: 25 suppliers x 2 scenarios = 50 golden test scenarios

### 13.3 Determinism Tests

Every risk calculation engine will include determinism tests that:
1. Run the same calculation 100 times with identical inputs
2. Verify bit-perfect identical outputs (SHA-256 hash match)
3. Test across Python versions (3.11, 3.12) to ensure no platform-dependent behavior
4. Verify Decimal arithmetic produces identical results to reference calculations
5. Verify that factor weight normalization produces consistent results for custom weight profiles

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **DD** | Due Diligence -- the process of risk assessment and mitigation required by EUDR |
| **FSC** | Forest Stewardship Council -- forest product certification scheme |
| **PEFC** | Programme for the Endorsement of Forest Certification -- forest certification umbrella scheme |
| **RSPO** | Roundtable on Sustainable Palm Oil -- palm oil certification scheme |
| **ISCC** | International Sustainability and Carbon Certification -- multi-commodity sustainability scheme |
| **UTZ** | UTZ Certified (now merged with Rainforest Alliance) -- sustainable farming certification |
| **FSC-COC** | FSC Chain of Custody -- certification for product tracking through supply chain |
| **FSC-CW** | FSC Controlled Wood -- risk-based system for non-FSC-certified material |
| **RSPO-SCC** | RSPO Supply Chain Certification -- certification for palm oil supply chain actors |
| **RSPO-MB** | RSPO Mass Balance -- custody model allowing mixing of certified and uncertified palm oil |
| **RSPO-SG** | RSPO Segregated -- custody model keeping certified and uncertified palm oil separate |
| **RSPO-IP** | RSPO Identity Preserved -- custody model maintaining physical separation from single source |
| **CAP** | Corrective Action Plan -- structured plan to address non-conformances |
| **NC** | Non-Conformance -- finding where a supplier fails to meet a requirement |
| **HHI** | Herfindahl-Hirschman Index -- measure of market/sourcing concentration (0-10,000) |
| **GLAD** | Global Land Analysis and Discovery -- deforestation alert system from University of Maryland |
| **RADD** | Radar Alerts for Detecting Deforestation -- radar-based deforestation alert system |
| **OFAC** | Office of Foreign Assets Control -- US sanctions enforcement agency |
| **SDN** | Specially Designated Nationals -- OFAC sanctions list |
| **ASI** | Accreditation Services International -- FSC's accreditation body |
| **FPIC** | Free, Prior and Informed Consent -- principle for indigenous peoples' rights |

### Appendix B: Supplier Risk Factor Definitions

| Factor | Weight (Default) | Score Range | Description | Key Data Sources |
|--------|-----------------|-------------|-------------|-----------------|
| Geographic Sourcing | 20% | 0-100 | Risk based on where the supplier sources commodities; integrates EUDR-016 country risk, sub-national hotspot scores, protected area proximity, and deforestation alert overlap | EUDR-016, AGENT-DATA-006, AGENT-DATA-007 |
| Compliance History | 15% | 0-100 | Risk based on supplier's DD track record: non-conformance count/severity, CAP completion rate, DD activity timeliness, responsiveness to requests | DD activity records, non-conformance database |
| Documentation Quality | 15% | 0-100 | Risk based on completeness, accuracy, consistency, and timeliness of EUDR-required documents submitted by the supplier | Document inventory, EUDR document matrix |
| Certification Status | 15% | 0-100 | Risk based on certification coverage, validity, scope alignment, chain of custody model, and certifying body accreditation | Certification records, scheme registries |
| Traceability Completeness | 10% | 0-100 | Risk based on ability to trace supplied products back to production plots; considers supply chain depth, intermediary count, geolocation availability, and custody chain completeness | EUDR-001 graph data, EUDR-008 supplier data |
| Financial Stability | 10% | 0-100 | Risk based on supplier's financial health indicators: company age, employee count, revenue stability, payment behavior | Supplier master data, financial indicators |
| Environmental Performance | 10% | 0-100 | Risk based on supplier's environmental track record: environmental incidents, regulatory violations, satellite-detected anomalies near sourcing locations | Environmental databases, satellite data |
| Social Compliance | 5% | 0-100 | Risk based on supplier's social compliance: labor practices, community relations, indigenous rights respect, social audit results | Social audit data, incident databases |

### Appendix C: EUDR Document Requirements Matrix

| Document Type | EUDR Article | Required For | Criticality |
|---------------|-------------|-------------|-------------|
| Geolocation data (GPS coordinates) | Art. 9(1)(a-c) | All suppliers with plots <= 4 ha | Critical |
| Geolocation data (polygon boundaries) | Art. 9(1)(d) | All suppliers with plots > 4 ha | Critical |
| Product description | Art. 4(2)(a) | All suppliers | Critical |
| Quantity and unit | Art. 4(2)(b) | All suppliers | Critical |
| Country of production | Art. 4(2)(c) | All suppliers | Critical |
| Supplier identification | Art. 4(2)(d) | All suppliers | Critical |
| Compliance declaration | Art. 4(2)(e) | All suppliers | Critical |
| Supply chain documentation | Art. 4(2)(f) | All suppliers | High |
| Harvesting/production date | Art. 9(1) | All producer suppliers | High |
| DDS reference number | Art. 4 | All suppliers linked to DDS | High |
| CN/HS code classification | EUDR Annex I | All suppliers | High |
| Certification evidence | Art. 10(2)(d) | Certified suppliers | Medium |
| Sub-supplier declarations | Art. 4(2)(f) | Intermediary suppliers | Medium |
| Operator registration number | Art. 6 | Operators/traders on EU market | Medium |

### Appendix D: Certification Scheme Validation Rules

| Scheme | Certificate Format | Scope Fields | Custody Models | Accreditation Body |
|--------|-------------------|-------------|----------------|-------------------|
| FSC | FSC-C######, FSC-F###### | Product groups, species, geographic scope | COC, CW, FM | ASI (Accreditation Services International) |
| PEFC | PEFC/##-##-## | Product categories, geographic scope | COC, Project | National PEFC governing bodies |
| RSPO | RSPO-#######-## | Palm oil products, geographic scope | IP, SG, MB | RSPO-approved CBs |
| Rainforest Alliance | RA-CERT-###### | Crop types, farm locations | Chain of Custody | Rainforest Alliance |
| UTZ | UTZ-CERT-###### | Coffee, cocoa, tea, hazelnut | Chain of Custody | UTZ (now RA) |
| ISCC | ISCC-###-###-## | Biofuels, food, feed, chemicals | Mass Balance | ISCC Association |
| Organic (EU) | EU-###-######-# | Organic products, conversion status | Chain of Custody | EU-accredited CBs |
| Fair Trade | FLO-ID-###### | Crop types, producer groups | Chain of Custody | FLOCERT |

### Appendix E: Sanction List Sources

| List | Authority | Update Frequency | Scope |
|------|-----------|-----------------|-------|
| EU Consolidated List | European Union | Daily | Entities and individuals subject to EU sanctions |
| OFAC SDN List | US Treasury | Daily | Specially Designated Nationals and blocked persons |
| UN Security Council Consolidated List | United Nations | As updated | Entities subject to UN sanctions |
| World Bank Debarment List | World Bank Group | As updated | Firms debarred from World Bank projects |

### Appendix F: Integration API Contracts

**Consumed from EUDR-016 (Country Risk Evaluator):**
```python
# Country risk data for supplier geographic sourcing analysis
def get_country_risk(country_code: str) -> Dict:
    """Returns: {composite_score, risk_level, commodity_risks, hotspots, provenance_hash}"""

def get_commodity_risk(country_code: str, commodity: str) -> Dict:
    """Returns: {risk_score, risk_level, deforestation_correlation, certification_effectiveness}"""

def get_hotspots(country_code: str) -> List[Dict]:
    """Returns: [{hotspot_id, region_name, risk_score, center_lat, center_lon, area_km2}]"""
```

**Provided to EUDR-001 (Supply Chain Mapping Master):**
```python
# Supplier risk data for supply chain graph node enrichment
{
    "supplier_id": "uuid-...",
    "supplier_name": "PT Sawit Lestari",
    "composite_score": 68,
    "risk_level": "high",
    "confidence_level": "medium",
    "trend": "deteriorating",
    "factor_scores": {
        "geographic_sourcing": 82,
        "compliance_history": 55,
        "documentation_quality": 45,
        "certification_status": 35,
        "traceability_completeness": 72,
        "financial_stability": 60,
        "environmental_performance": 78,
        "social_compliance": 50
    },
    "certifications": ["RSPO-MB"],
    "open_non_conformances": 2,
    "provenance_hash": "sha256:a1b2c3..."
}
```

**Consumed from EUDR-008 (Multi-Tier Supplier Tracker):**
```python
# Supplier master data
def get_supplier(supplier_id: str) -> Dict:
    """Returns: {supplier_id, name, country, commodities, tier, sub_suppliers, certifications}"""

def get_sub_suppliers(supplier_id: str) -> List[Dict]:
    """Returns list of sub-supplier records with relationship details"""
```

### Appendix G: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EUDR)
2. European Commission EUDR Guidance Document -- Due Diligence and Supplier Assessment
3. European Commission EUDR FAQ -- Operator Obligations and Supplier Due Diligence
4. FSC Chain of Custody Standard (FSC-STD-40-004 V3-1)
5. RSPO Supply Chain Certification Standard (2020)
6. PEFC Chain of Custody Standard (PEFC ST 2002:2020)
7. Rainforest Alliance Sustainable Agriculture Standard (2020)
8. ISCC System Document 203: Traceability and Chain of Custody
9. EU Council Regulation (EC) No 765/2008 -- Accreditation requirements
10. OFAC SDN List Technical Documentation
11. EU Consolidated Sanctions List -- Technical Format Specification
12. Global Forest Watch Technical Documentation -- GLAD Alerts, RADD Alerts
13. ISO 3166-1 -- Country Codes Standard
14. EUDR Annex I -- Products Covered by the Regulation

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-09 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-09 | GL-ProductManager | Initial draft created |
| 1.0.0 | 2026-03-09 | GL-ProductManager | Finalized: all 8 P0 features confirmed (80 sub-requirements: F1.1-F1.10 through F8.1-F8.10), regulatory coverage verified (Articles 4/9/10/11/12/13/14/31), 8-factor composite scoring model defined, integration with EUDR-016/EUDR-001/EUDR-008 documented, 12-table DB schema V105 designed with 3 hypertables and 2 continuous aggregates, ~40 API endpoints specified, 18 Prometheus metrics defined, 23 RBAC permissions mapped, 650+ test target set, approval granted |
