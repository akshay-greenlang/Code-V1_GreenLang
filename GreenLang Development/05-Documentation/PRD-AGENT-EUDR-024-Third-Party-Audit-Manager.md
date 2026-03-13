# PRD: AGENT-EUDR-024 -- Third-Party Audit Manager Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-024 |
| **Agent ID** | GL-EUDR-TAM-024 |
| **Component** | Third-Party Audit Manager Agent |
| **Category** | EUDR Regulatory Agent -- Audit & Verification Management |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-10 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 4, 9, 10, 11, 14, 15, 16, 18, 19, 20, 21, 22, 23, 29, 31; ISO 19011:2018; ISO/IEC 17065:2012; ISO/IEC 17021-1:2015 |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) establishes a comprehensive due diligence framework that requires operators and traders to verify that products placed on the EU market are deforestation-free and legally produced. Articles 14-16 empower competent authorities to conduct checks on operators, including document-based checks and physical inspections. Article 10 mandates that operators assess the risk of non-compliance through robust verification activities. Article 11 requires risk mitigation measures including independent auditing and verification. Recital 43 explicitly references the role of certification schemes (FSC, PEFC, RSPO) as complementary tools in the due diligence process, while cautioning that they cannot substitute for the operator's own due diligence obligations.

In practice, third-party auditing is the backbone of EUDR compliance verification. When operators source from standard-risk or high-risk countries (Article 29 benchmarking), enhanced due diligence is triggered, frequently requiring independent third-party verification audits. Certification schemes such as FSC (Forest Stewardship Council), PEFC (Programme for the Endorsement of Forest Certification), RSPO (Roundtable on Sustainable Palm Oil), Rainforest Alliance, and ISCC (International Sustainability and Carbon Certification) each conduct their own audit cycles with accredited certification bodies. Competent authorities in EU Member States conduct enforcement inspections and may request audit documentation at any time. The audit lifecycle -- from planning through execution, finding resolution, and competent authority response -- is a multi-stakeholder, multi-timeline, document-intensive process that spans months and involves dozens of participants.

Today, EU operators and compliance teams face the following critical gaps when managing third-party audit activities:

- **No centralized audit lifecycle management**: Audit activities are tracked in disconnected systems -- certification body portals, email threads, shared drives, and spreadsheets. There is no unified platform that manages the complete audit lifecycle from planning through closure across all certification schemes and regulatory inspections. A typical large importer with 200+ suppliers may be managing 50+ audits per year across 5 different certification schemes with no central visibility.
- **No risk-based audit scheduling**: EUDR Article 10 mandates risk-based due diligence, yet audit scheduling is typically calendar-driven (annual recertification) rather than risk-driven. There is no system that dynamically adjusts audit frequency, scope, and depth based on country risk (Article 29), supplier risk, deforestation alerts, non-conformance history, and certification expiry timelines. High-risk suppliers receive the same audit frequency as low-risk ones.
- **No auditor qualification tracking**: ISO/IEC 17065 and ISO/IEC 17021-1 require that auditors possess specific competencies for the sectors and regions they audit. There is no centralized registry to track auditor qualifications, accreditation status, competence domains (commodity expertise, language, regional knowledge), conflict-of-interest declarations, and performance history. Operators cannot verify that the auditor assigned to a palm oil audit in Indonesia has the required competencies.
- **No structured non-conformance management**: When audits identify non-conformances (NCs), the findings are documented in PDF reports with no structured data model. There is no system to classify NCs by severity (critical/major/minor per ISO 19011 and ISO/IEC 17021-1), perform root cause analysis, track corrective action requests (CARs) against SLA deadlines, verify corrective action effectiveness, and analyze NC trends across suppliers and regions.
- **No corrective action request (CAR) lifecycle management**: CARs issued following audit findings require structured follow-up: acknowledgment by the audited party, root cause analysis, corrective action plan submission, implementation evidence collection, verification of effectiveness, and formal closure. This multi-step lifecycle is managed through email exchanges with no SLA enforcement, escalation rules, or closure verification workflow.
- **No cross-scheme audit coordination**: Suppliers certified under multiple schemes (e.g., FSC + Rainforest Alliance, or RSPO + ISCC) undergo redundant audits that assess overlapping requirements. There is no system to coordinate audit schedules across schemes, identify shared audit scope, reduce audit burden through mutual recognition, or consolidate findings from multiple scheme audits into a unified compliance view.
- **No ISO 19011 compliant report generation**: Audit reports must meet ISO 19011:2018 requirements for structure, content, and evidence documentation. There is no automated system that generates standardized audit reports with findings, evidence packages, sampling rationale, auditor credentials, and corrective action timelines in a format accepted by certification bodies and competent authorities.
- **No competent authority liaison workflow**: When EU Member State competent authorities (Articles 14-16) request documentation, initiate inspections, or issue enforcement actions, operators must respond within specified timelines. There is no structured system to receive authority requests, coordinate response preparation across internal teams, track response deadlines, manage inspection logistics, and maintain a complete record of regulatory interactions.
- **No audit analytics or trend detection**: Without structured audit data, operators cannot identify systemic compliance patterns -- which regions generate the most critical findings, which suppliers have recurring issues, whether corrective actions are effective, or how audit performance trends over time. This prevents proactive compliance improvement and risk-based resource allocation.

Without solving these problems, EU operators face regulatory enforcement action under Articles 23-25 (penalties up to 4% of annual EU turnover), suspension of market access, confiscation of goods, public naming, and loss of certification scheme credentials that are increasingly required by buyers and financial institutions.

### 1.2 Solution Overview

Agent-EUDR-024: Third-Party Audit Manager is a specialized compliance agent that provides end-to-end management of the third-party audit lifecycle for EUDR compliance verification. It is the 24th agent in the EUDR agent family and establishes a new Audit and Verification Management sub-category. The agent manages audit planning and risk-based scheduling, auditor qualification tracking, audit execution monitoring, non-conformance detection and classification, corrective action request (CAR) lifecycle management, certification scheme audit coordination, ISO 19011 compliant report generation, competent authority liaison workflows, and audit analytics with trend detection.

The agent builds on and integrates with the existing EUDR agent ecosystem: EUDR-001 (Supply Chain Mapping Master) for audit scope determination from supply chain graphs, EUDR-016 (Country Risk Evaluator) for risk-based audit prioritization by country, EUDR-017 (Supplier Risk Scorer) for supplier-level audit scheduling triggers, EUDR-020 (Deforestation Alert System) for deforestation-alert-triggered audits, EUDR-021 (Indigenous Rights Checker) for FPIC verification audit requirements, EUDR-022 (Protected Area Validator) for protected area audit checks, and EUDR-023 (Legal Compliance Verifier) for legal audit verification and document compliance integration.

Core capabilities:

1. **Audit planning and scheduling engine** -- Risk-based audit scheduling that dynamically calculates audit frequency, scope, and depth based on country risk (Article 29), supplier risk score, non-conformance history, certification expiry dates, deforestation alerts, and regulatory inspection triggers. Maintains a centralized audit calendar with multi-scheme coordination, resource allocation, and conflict detection. Supports annual audit planning cycles with quarterly reviews.
2. **Auditor registry and qualification tracker** -- Centralized registry of auditor profiles compliant with ISO/IEC 17065 and ISO/IEC 17021-1 competence requirements. Tracks accreditation status, commodity-sector expertise, regional and language qualifications, conflict-of-interest declarations, audit history, and performance ratings. Matches auditors to audit assignments based on competence requirements and availability.
3. **Audit execution tracker** -- Real-time monitoring of audit execution including checklist management with EUDR-specific audit criteria, evidence collection tracking, sampling plan management (ISO 19011 Annex A guidance), fieldwork scheduling, stakeholder interview coordination, and audit progress dashboards. Supports both on-site and remote/hybrid audit modalities.
4. **Non-conformance detection engine** -- Structured classification of audit findings into critical, major, and minor non-conformances per ISO 19011 and certification scheme severity definitions. Automated root cause analysis using the "5 Whys" and Ishikawa (fishbone) frameworks. Links non-conformances to specific EUDR articles, certification scheme clauses, and supplier risk factors. Trend analysis across audits, suppliers, and regions.
5. **Corrective action request (CAR) manager** -- Full CAR lifecycle management: issuance with severity-based SLA deadlines, auditee acknowledgment tracking, root cause analysis documentation, corrective action plan (CAP) submission and review, implementation evidence collection, effectiveness verification audit, and formal closure with sign-off. Escalation rules for overdue CARs with configurable notification chains.
6. **Certification scheme integration** -- Bidirectional integration with 5 major certification schemes (FSC, PEFC, RSPO, Rainforest Alliance, ISCC). Maps scheme-specific audit requirements to EUDR articles. Coordinates audit schedules across schemes to reduce redundancy. Imports certification audit results and exports EUDR compliance evidence. Tracks certificate status (active/suspended/withdrawn) and recertification timelines.
7. **Audit report generator** -- Automated generation of ISO 19011:2018 compliant audit reports including executive summary, audit scope and objectives, audit criteria, findings with evidence references, non-conformance detail, sampling rationale, auditor credentials, corrective action timelines, and conclusion. Generates evidence packages with document cross-references. Supports PDF, JSON, HTML, and XLSX formats in 5 languages (EN, FR, DE, ES, PT).
8. **Competent authority liaison** -- Structured workflow for managing interactions with EU Member State competent authorities under Articles 14-16. Receives and logs authority requests (document requests, inspection notifications, enforcement actions), coordinates internal response preparation, tracks response deadlines with SLA countdowns, manages on-site inspection logistics, and maintains a complete audit trail of all regulatory interactions. Supports 27 EU Member State competent authority profiles.
9. **Audit analytics and trends** -- Comprehensive analytics engine providing audit finding trends across time, regions, commodities, and suppliers. Auditor performance benchmarking (findings per audit, CAR closure rates, audit duration). Compliance rate tracking (% of audits with zero critical/major findings). Predictive risk indicators from NC patterns. Cost analysis per audit type. Dashboard with configurable KPIs and drill-down capability.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Audit lifecycle coverage | 100% of audit stages managed (plan through closure) | Feature completeness against ISO 19011 lifecycle |
| Risk-based scheduling accuracy | 90%+ of high-risk suppliers audited within risk-adjusted frequency | Compliance with risk-based audit calendar |
| Auditor qualification compliance | 100% of audit assignments matched to qualified auditors | Competence matrix validation per assignment |
| Non-conformance classification accuracy | >= 95% agreement with expert auditor classification | Cross-validation with senior auditor review panel |
| CAR closure within SLA | >= 85% of CARs closed within severity-based SLA deadlines | SLA compliance tracking |
| Certification scheme coverage | 5 schemes integrated (FSC, PEFC, RSPO, RA, ISCC) | Scheme count with active integration |
| Audit report generation time | < 30 seconds per ISO 19011 compliant report | Time from request to PDF delivery |
| Competent authority response SLA | 100% of responses within regulatory deadline | SLA compliance tracking |
| Audit analytics dashboard load | < 3 seconds for full analytics dashboard | Frontend render benchmark |
| Cross-scheme audit reduction | >= 20% reduction in redundant audit activities | Before/after audit count comparison |
| Agent integration health | 99.9% message delivery for cross-agent events | Integration health monitoring |
| Zero-hallucination guarantee | 100% deterministic calculations, no LLM in critical path | Bit-perfect reproducibility tests |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, combined with the global sustainability audit management market estimated at 4-6 billion EUR. Third-party verification is a mandatory component of EUDR due diligence, and CSDDD (effective 2027) will further expand audit requirements.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers requiring third-party audit management for EUDR compliance, plus 500+ certification bodies conducting EUDR-relevant audits, estimated at 600M-900M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 35-55M EUR in audit management module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) managing 50+ third-party audits annually across multiple certification schemes and producing countries
- Multinational food and beverage companies with cocoa, coffee, palm oil, and soya supply chains requiring certification scheme audit coordination (FSC, RSPO, Rainforest Alliance)
- Timber and paper industry operators with FSC/PEFC certification audit management requirements
- Compliance officers responsible for EUDR due diligence with competent authority inspection response obligations

**Secondary:**
- Certification bodies (FSC, RSPO, PEFC, Rainforest Alliance, ISCC) requiring audit management platforms for their accredited auditors
- Third-party audit firms conducting EUDR verification audits for multiple operators
- Commodity traders managing supplier audit programs across multiple origins
- Financial institutions requiring audit evidence for EUDR-linked portfolio risk assessment
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Spreadsheet / Email | No cost; familiar | Cannot manage multi-scheme lifecycle; no SLA enforcement; no analytics; error-prone; no audit trail | Full lifecycle management, automated SLAs, cross-scheme coordination |
| Generic Audit Management (SAP GRC Audit, MetricStream, AuditBoard) | Enterprise integration; broad audit scope | Not EUDR-specific; no certification scheme integration; no Article 29 risk-based scheduling; no EUDR article mapping | Purpose-built for EUDR; 5-scheme integration; risk-based scheduling linked to EUDR-016/017 |
| Certification Body Platforms (FSC Info, RSPO ACOP, PEFC Online) | Authoritative for their scheme; free | Single-scheme only; no cross-scheme coordination; no operator-side workflow; no competent authority liaison | Multi-scheme coordination; operator-centric workflow; competent authority response management |
| Niche EUDR Compliance Tools (Preferred by Nature, Ecosphere+) | EUDR domain expertise | Limited audit lifecycle management; no CAR tracking; no auditor registry; no ISO 19011 reporting | Full audit lifecycle; ISO 19011 compliance; auditor qualification tracking; CAR management |
| In-house Custom Builds | Tailored to organization | 12-18 month build; no regulatory updates; no scheme integration; no scale | Ready now; continuous updates; 5-scheme integration; production-grade |

### 2.4 Differentiation Strategy

1. **EUDR-native audit management** -- Not a generic audit tool adapted for EUDR; purpose-built with every audit criterion, NC classification, and report field mapped to specific EUDR articles.
2. **Risk-based scheduling from EUDR risk agents** -- Audit frequency dynamically driven by EUDR-016 (country risk), EUDR-017 (supplier risk), and EUDR-020 (deforestation alerts), not just calendar-based recertification cycles.
3. **Five-scheme integration** -- Simultaneous coordination of FSC, PEFC, RSPO, Rainforest Alliance, and ISCC audits with cross-scheme redundancy reduction.
4. **Competent authority liaison** -- Unique capability to manage EU regulatory inspection responses with 27 Member State profiles, deadline tracking, and evidence assembly.
5. **Integration depth** -- Pre-built connectors to 7 EUDR agents (001, 016, 017, 020, 021, 022, 023) and GL-EUDR-APP.
6. **Zero-hallucination audit analytics** -- Deterministic finding classification, trend analysis, and risk scoring with SHA-256 provenance hashes; no LLM in the critical path.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to manage EUDR audit lifecycle end-to-end in a single platform | 100% of audit lifecycle stages supported (plan to closure) | Q2 2026 |
| BG-2 | Reduce audit management overhead from weeks to hours per audit cycle | 80% reduction in administrative audit management time | Q2 2026 |
| BG-3 | Achieve >= 85% CAR closure rate within SLA deadlines across customer base | Aggregate CAR SLA compliance rate | Q3 2026 |
| BG-4 | Become the reference audit management platform for EUDR compliance | 500+ enterprise customers using audit management module | Q4 2026 |
| BG-5 | Reduce redundant audit activities by >= 20% through cross-scheme coordination | Before/after audit count per supplier | Q4 2026 |
| BG-6 | Support CSDDD readiness by building audit infrastructure reusable for CSDDD Article 11 | Audit module extensible for CSDDD human rights and environmental audit requirements | Q1 2027 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Risk-based audit planning | Dynamically schedule audits based on country risk, supplier risk, NC history, and deforestation alerts |
| PG-2 | Auditor competence management | Track and validate auditor qualifications per ISO/IEC 17065 and ISO/IEC 17021-1 |
| PG-3 | Structured NC management | Classify, track, and analyze non-conformances with root cause analysis and trend detection |
| PG-4 | CAR lifecycle management | Manage corrective action requests from issuance through verified closure with SLA enforcement |
| PG-5 | Multi-scheme coordination | Coordinate audits across FSC, PEFC, RSPO, Rainforest Alliance, and ISCC to reduce redundancy |
| PG-6 | ISO 19011 reporting | Generate audit reports compliant with ISO 19011:2018 including evidence packages |
| PG-7 | Regulatory response management | Manage competent authority interactions with deadline tracking and evidence assembly |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Audit scheduling performance | < 2 seconds to generate risk-based audit schedule for 500 suppliers |
| TG-2 | NC classification performance | < 500ms per non-conformance classification and severity assessment |
| TG-3 | CAR SLA tracking | Real-time SLA countdown with < 1 minute update latency |
| TG-4 | Report generation | < 30 seconds per ISO 19011 compliant PDF report |
| TG-5 | API response time | < 200ms p95 for standard queries |
| TG-6 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-7 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |
| TG-8 | Audit data freshness | Certification status sync within 24 hours of scheme database update |

---

## 4. Regulatory Requirements

### 4.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(2)** | Due diligence obligation -- collect information, assess risk, mitigate risk through verification activities | Audit planning engine (F1) schedules verification audits; audit execution tracker (F3) manages evidence collection; NC engine (F4) assesses compliance gaps |
| **Art. 9(1)** | Geolocation of all plots of land; product information for DDS | Audit checklists (F3) include plot geolocation verification criteria; audit reports (F7) document geolocation verification evidence |
| **Art. 10(1)** | Risk assessment -- operators shall assess and identify risk of non-compliance | Risk-based audit scheduling (F1) prioritizes audits based on assessed risk; NC trends (F9) inform ongoing risk assessment |
| **Art. 10(2)(a)** | Complexity of the relevant supply chain as risk factor | Audit scope determination integrates EUDR-001 supply chain complexity metrics; complex chains trigger expanded audit scope |
| **Art. 10(2)(b)** | Risk of circumvention or mixing with unknown-origin products | Audit checklists (F3) include chain-of-custody verification, mass balance checks, and segregation verification criteria |
| **Art. 10(2)(d)** | Prevalence of deforestation or forest degradation in the country of production | Country risk from EUDR-016 drives audit frequency; deforestation alerts from EUDR-020 trigger unscheduled audits |
| **Art. 10(2)(e)** | Concerns about the country of production including corruption and document fraud | Country risk-adjusted audit depth; audit checklists include document authenticity verification; red flag-triggered audits |
| **Art. 10(2)(f)** | Risk of circumvention or mixing with products of unknown origin | Mass balance verification in audit checklists; chain-of-custody audit criteria per certification scheme requirements |
| **Art. 11(1)** | Risk mitigation -- additional information, independent surveys, audits, or other measures | Third-party audit management is a primary Article 11 risk mitigation tool; CAR management (F5) ensures mitigation action completion |
| **Art. 14(1)** | Competent authorities shall carry out checks on operators and traders | Competent authority liaison (F8) manages inspection request receipt, response preparation, and evidence assembly |
| **Art. 14(4)** | Checks shall be based on a risk-based approach including country benchmarking | Risk-based audit scheduling (F1) aligns with competent authority risk-based approach |
| **Art. 15** | Checks on operators and traders -- document-based and physical | Audit execution tracker (F3) supports both document review and on-site inspection modalities; evidence packages (F7) prepared for authority review |
| **Art. 16** | Checks to include examination of due diligence system, document checks, field checks | Comprehensive audit checklist covering DDS verification, document examination, and field inspection criteria |
| **Art. 18** | Requests for corrective action by competent authorities | CAR manager (F5) tracks authority-issued corrective actions alongside scheme-issued CARs; competent authority liaison (F8) manages response workflow |
| **Art. 19-20** | Interim and definitive measures by competent authorities | Competent authority liaison (F8) tracks enforcement measures, suspension orders, and compliance restoration timelines |
| **Art. 21** | Obligation for operators to inform competent authorities of non-compliance | Audit findings and NC detection (F4) generate structured reports suitable for self-disclosure to competent authorities |
| **Art. 22** | Penalties -- effective, proportionate, dissuasive | Audit analytics (F9) tracks penalty risk exposure based on open NCs and overdue CARs; risk dashboard for management reporting |
| **Art. 23** | Specific penalties including fines up to 4% of turnover | Penalty risk assessment integrated with audit finding severity and CAR closure status |
| **Art. 29** | Country benchmarking (Low/Standard/High risk) | Country risk classification from EUDR-016 drives audit frequency multipliers and scope adjustments |
| **Art. 31** | Record keeping for 5 years | All audit records, NC findings, CARs, reports, and authority interactions retained for minimum 5 years with immutable audit trail |

### 4.2 ISO Standards Addressed

| Standard | Scope | Agent Implementation |
|----------|-------|---------------------|
| **ISO 19011:2018** | Guidelines for auditing management systems -- audit principles, audit programme management, conducting audits, auditor competence | Audit report generator (F7) produces ISO 19011 compliant reports; auditor registry (F2) tracks competence per Clause 7; audit planning (F1) follows Clause 5 programme management; audit execution (F3) implements Clause 6 conducting guidelines |
| **ISO/IEC 17065:2012** | Conformity assessment -- requirements for bodies certifying products, processes, and services | Auditor registry (F2) validates certification body accreditation against ISO/IEC 17065; certification scheme integration (F6) verifies CB accreditation status |
| **ISO/IEC 17021-1:2015** | Conformity assessment -- requirements for bodies providing audit and certification of management systems | Auditor competence requirements (F2) aligned with Clause 7 (competence and evaluation of auditors); impartiality requirements tracked in conflict-of-interest declarations |
| **ISO/IEC 17011:2017** | Conformity assessment -- requirements for accreditation bodies | Accreditation body status tracked in auditor registry (F2); accreditation validity verified for each certification body |

### 4.3 Certification Scheme Audit Requirements

| Scheme | Audit Standard | Audit Frequency | NC Classification | EUDR Relevance |
|--------|---------------|----------------|-------------------|----------------|
| **FSC** | FSC-STD-20-007 (Forest Management Evaluations); FSC-STD-20-011 (Chain of Custody Evaluations) | Annual surveillance; 5-year recertification | Major (3-month deadline); Minor (12-month deadline); Observation | Timber and wood products; covers deforestation-free, legal harvest, indigenous rights (FPIC), worker rights |
| **PEFC** | PEFC ST 2003:2020 (Chain of Custody); PEFC ST 1003:2018 (SFM) | Annual surveillance; 5-year recertification | Major NC; Minor NC; Observation | Timber and wood products; covers legal compliance, environmental protection, stakeholder rights |
| **RSPO** | RSPO Principles & Criteria (2018); RSPO SCCS | Annual surveillance; 5-year recertification | Major NC (3-month deadline); Minor NC (12-month deadline); Observation | Palm oil; covers deforestation (HCV/HCS), legal compliance, FPIC, labour rights |
| **Rainforest Alliance** | RA Sustainable Agriculture Standard (2020); RA CoC Standard | Annual audit; 3-year certification cycle | Critical (immediate); Major (3-month); Minor (12-month); Improvement Need | Cocoa, coffee, tea; covers deforestation, legal compliance, human rights, livelihoods |
| **ISCC** | ISCC EU/PLUS System Requirements; ISCC 202 Sustainability Requirements | Annual audit; annual recertification | Major NC; Minor NC; Observation | Biomass and biofuels supply chains; covers deforestation, land use, GHG, traceability |

### 4.4 Competent Authority Framework (Articles 14-16)

| EU Member State | Competent Authority | Inspection Focus | Response Timeline |
|----------------|--------------------|--------------------|-------------------|
| Germany | BMEL (Bundesministerium fur Ernahrung und Landwirtschaft) | Document checks, DDS verification, supply chain traceability | 30 days for document requests |
| France | DGCCRF (Direction Generale de la Concurrence) + DGAL | Product checks, import controls, DDS verification | 30 days for document requests |
| Netherlands | NVWA (Nederlandse Voedsel- en Warenautoriteit) | Import checks, timber traceability, palm oil supply chains | 30 days for document requests |
| Belgium | FOD Economie / SPF Economie | Product checks, supply chain verification | 30 days for document requests |
| Italy | Ministero dell'Ambiente e della Sicurezza Energetica | Environmental compliance, deforestation-free verification | 30 days for document requests |
| Spain | MITECO (Ministerio para la Transicion Ecologica) | Timber and wood product checks, supply chain traceability | 30 days for document requests |
| Other EU-27 | Member State designated authority per Article 14 | As specified by national implementing legislation | As specified by national regulation |

### 4.5 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for all deforestation-free verification in audit criteria |
| June 29, 2023 | EUDR entered into force | Legal basis for audit compliance verification |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Full audit management must be operational; competent authorities conducting checks |
| June 30, 2026 | Enforcement for SMEs | SME audit wave; agent must handle 10x audit volume |
| 2027 | CSDDD enforcement begins | Audit management module extended for CSDDD Article 11 audit requirements |
| Ongoing | Certification scheme audit cycles | Annual surveillance and 3-5 year recertification tracked per scheme |
| Ongoing | Competent authority inspections | Authority requests received and processed per Articles 14-16 |

---

## 5. Scope and Zero-Hallucination Principles

### 5.1 Scope -- In and Out

**In Scope (v1.0):**
- Risk-based audit planning and scheduling for EUDR compliance across 7 commodities and 20+ producing countries
- Auditor registry with ISO/IEC 17065 and ISO/IEC 17021-1 competence tracking for auditors and certification bodies
- Audit execution monitoring with EUDR-specific checklists, evidence collection, and sampling plan management
- Non-conformance detection and classification (critical/major/minor) with root cause analysis frameworks
- Corrective action request (CAR) full lifecycle management with SLA enforcement and escalation
- Certification scheme integration for FSC, PEFC, RSPO, Rainforest Alliance, and ISCC
- ISO 19011:2018 compliant audit report generation in 5 formats and 5 languages
- Competent authority liaison workflow for 27 EU Member State authorities
- Audit analytics with finding trends, auditor performance, compliance rates, and cost analysis
- Integration with EUDR-001, EUDR-016, EUDR-017, EUDR-020, EUDR-021, EUDR-022, EUDR-023

**Out of Scope (v1.0):**
- Conducting actual audits (agent manages the audit lifecycle; human auditors conduct the audits)
- Accreditation body management (agent verifies accreditation status; does not manage accreditation processes)
- Financial management of audit fees and invoicing (agent tracks audit costs for analytics; does not process payments)
- Direct submission of audit reports to certification scheme registries (agent generates reports; submission is manual or via scheme portals)
- Real-time video audit monitoring or remote audit streaming capability (defer to Phase 2)
- Predictive ML models for audit outcome forecasting (defer to Phase 2)
- Mobile native application for field auditors (web responsive only for v1.0)
- Blockchain-based audit evidence immutability (SHA-256 provenance hashes provide sufficient integrity)

### 5.2 Zero-Hallucination Principles

All 9 features in this agent operate under strict zero-hallucination guarantees:

| Principle | Implementation |
|-----------|---------------|
| **Deterministic calculations** | Same inputs always produce the same risk-based audit schedules, NC classifications, CAR SLA calculations, and compliance scores (bit-perfect reproducibility) |
| **No LLM in critical path** | All audit scheduling, NC severity classification, CAR SLA management, compliance scoring, and trend analysis use deterministic rule-based algorithms only |
| **Authoritative data sources only** | All certification scheme data sourced from official scheme registries; competent authority profiles from official EU Member State designations; ISO standards from published normative documents |
| **Full provenance tracking** | Every audit finding, NC classification, CAR status change, and compliance score includes SHA-256 hash, data source citations, and calculation timestamps |
| **Immutable audit trail** | All audit data changes recorded in `gl_eudr_tam_audit_log` with before/after values, actor identification, and timestamp |
| **Decimal arithmetic** | Compliance scores, risk calculations, and SLA durations use Decimal type to prevent floating-point drift |
| **Version-controlled audit criteria** | EUDR audit checklists and certification scheme criteria are versioned; any update creates a new version with effective date and source attribution |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-5 form the core audit lifecycle engine; Features 6-9 form the integration, reporting, and analytics layer.

**P0 Features 1-5: Core Audit Lifecycle Engine**

---

#### Feature 1: Audit Planning and Scheduling Engine

**User Story:**
```
As a compliance officer,
I want the system to automatically generate a risk-based audit schedule for all my suppliers,
So that high-risk suppliers receive more frequent and deeper audits aligned with EUDR Article 10 requirements.
```

**Acceptance Criteria (12 sub-requirements):**
- [ ] F1.1: Calculates audit frequency per supplier based on composite risk score integrating country risk (EUDR-016, weight 0.25), supplier risk (EUDR-017, weight 0.25), NC history (weight 0.20), certification status (weight 0.15), and deforestation alert proximity (EUDR-020, weight 0.15)
- [ ] F1.2: Assigns audit frequency tiers: HIGH risk = quarterly audit; STANDARD risk = semi-annual audit; LOW risk = annual audit; with configurable frequency multipliers
- [ ] F1.3: Determines audit scope per supplier: FULL scope (all EUDR criteria + certification scheme criteria), TARGETED scope (specific risk areas only), SURVEILLANCE scope (maintenance/follow-up)
- [ ] F1.4: Determines audit depth per risk level: HIGH risk = on-site with field verification; STANDARD risk = on-site document review; LOW risk = remote/desktop review
- [ ] F1.5: Generates annual audit calendar with quarterly review cycles, showing planned audits per supplier with assigned month, scope, depth, and estimated duration
- [ ] F1.6: Detects scheduling conflicts: auditor unavailability, overlapping audits for same supplier across schemes, holiday and travel restrictions
- [ ] F1.7: Triggers unscheduled audits based on events: deforestation alert from EUDR-020 (trigger within 14 days), critical NC from related supplier (trigger within 30 days), certification suspension (trigger within 7 days), competent authority request (trigger within specified deadline)
- [ ] F1.8: Integrates certification scheme recertification timelines: FSC 5-year cycle, PEFC 5-year cycle, RSPO 5-year cycle, RA 3-year cycle, ISCC annual cycle
- [ ] F1.9: Tracks audit resource budget (auditor-days) and provides utilization forecasts per quarter
- [ ] F1.10: Supports multi-site audit planning for suppliers with multiple production sites and processing facilities
- [ ] F1.11: Generates audit planning summary report with risk distribution, resource allocation, and schedule overview
- [ ] F1.12: All scheduling calculations are deterministic: same risk inputs produce same audit schedule (bit-perfect)

**Risk-Based Audit Frequency Formula:**
```
Audit_Priority_Score = (
    Country_Risk * 0.25 +
    Supplier_Risk * 0.25 +
    NC_History_Score * 0.20 +
    Certification_Gap_Score * 0.15 +
    Deforestation_Alert_Score * 0.15
) * Recency_Multiplier

Where:
- Country_Risk = EUDR-016 country risk score (0-100, normalized)
- Supplier_Risk = EUDR-017 supplier risk score (0-100, normalized)
- NC_History_Score = weighted sum of open NCs (critical=30, major=15, minor=5) / audit count
- Certification_Gap_Score = (1 - certification_coverage) * 100
- Deforestation_Alert_Score = max alert severity within 25km of supplier plots (0-100)
- Recency_Multiplier = days_since_last_audit / scheduled_interval (capped at 2.0)

Frequency Assignment:
- Score >= 70: HIGH (quarterly)
- Score 40-69: STANDARD (semi-annual)
- Score < 40: LOW (annual)
```

**Non-Functional Requirements:**
- Performance: Generate audit schedule for 500 suppliers in < 2 seconds
- Determinism: Bit-perfect reproducibility of schedule from same risk inputs
- Configurability: Risk weights and frequency thresholds adjustable per operator without code changes
- Auditability: Schedule generation recorded with full calculation provenance

**Dependencies:**
- EUDR-016 Country Risk Evaluator (country risk scores)
- EUDR-017 Supplier Risk Scorer (supplier risk scores)
- EUDR-020 Deforestation Alert System (alert proximity data)
- EUDR-001 Supply Chain Mapping Master (supplier and site inventory)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- New supplier with no audit history -- assign HIGH frequency for first audit cycle
- Supplier with expired certification -- trigger immediate unscheduled audit
- Country risk reclassification (Standard to High) -- recalculate all affected supplier schedules within 48 hours
- Multiple deforestation alerts for same supplier -- use highest severity, do not duplicate audit triggers

---

#### Feature 2: Auditor Registry and Qualification Tracker

**User Story:**
```
As an audit programme manager,
I want a centralized registry of all auditors and certification bodies with their qualifications and competencies,
So that I can assign qualified auditors to each audit per ISO/IEC 17065 and ISO/IEC 17021-1 requirements.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F2.1: Maintains auditor profiles with: name, auditor ID, employing organization, accreditation body, accreditation status (active/suspended/withdrawn), accreditation expiry date, and accreditation scope
- [ ] F2.2: Tracks commodity-sector competence per auditor: which of the 7 EUDR commodities the auditor is qualified to audit, with certification scheme-specific qualifications (FSC Lead Auditor, RSPO Lead Assessor, etc.)
- [ ] F2.3: Tracks regional competence per auditor: countries/regions of expertise, language capabilities (production country language proficiency), and cultural competence assessment
- [ ] F2.4: Manages conflict-of-interest declarations: auditor self-declarations, employer relationship checks, previous audit history with the auditee (rotation requirements per ISO/IEC 17021-1 Clause 7.2.8), and financial interest screening
- [ ] F2.5: Records audit performance history per auditor: number of audits conducted, findings per audit ratio, CAR closure rate for auditor-identified findings, average audit duration, and client feedback scores
- [ ] F2.6: Validates certification body accreditation: verifies CB is accredited by an IAF MLA signatory accreditation body, checks accreditation scope covers the relevant certification scheme, and monitors accreditation status changes
- [ ] F2.7: Implements auditor-to-audit matching algorithm: given audit requirements (commodity, country, scheme, scope), returns ranked list of qualified available auditors with competence match score
- [ ] F2.8: Tracks continuing professional development (CPD): training records, scheme-specific update courses, and CPD hour compliance per accreditation body requirements
- [ ] F2.9: Alerts on qualification expiry: accreditation expiry (60-day advance warning), CPD shortfall, scheme registration lapse, and conflict-of-interest rotation requirement
- [ ] F2.10: Supports bulk import of auditor data from certification body registries and manual entry for independent auditors

**Non-Functional Requirements:**
- Data Quality: Auditor accreditation status verified against CB registry at least quarterly
- Performance: Auditor matching query < 500ms for pool of 1,000 auditors
- Privacy: Auditor personal data stored with AES-256 encryption; GDPR-compliant data handling
- Auditability: All profile changes logged with timestamp and actor

**Dependencies:**
- SEC-003 Encryption at Rest (AES-256 for auditor personal data)
- SEC-002 RBAC (role-based access to auditor registry)
- Certification body registries (FSC, RSPO, PEFC accredited auditor lists)

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Auditor with suspended accreditation assigned to upcoming audit -- flag and require reassignment
- Auditor rotation requirement triggered mid-cycle -- suggest replacement with competence match
- Certification body loses accreditation -- flag all active audits by that CB's auditors

---

#### Feature 3: Audit Execution Tracker

**User Story:**
```
As a compliance officer,
I want to monitor the progress of each audit in real-time with structured checklists and evidence tracking,
So that I can ensure audits are conducted thoroughly and all EUDR-relevant criteria are assessed.
```

**Acceptance Criteria (12 sub-requirements):**
- [ ] F3.1: Provides EUDR-specific audit checklists covering: deforestation-free verification (Article 3), geolocation verification (Article 9), supply chain traceability (Article 4), risk assessment completeness (Article 10), risk mitigation measures (Article 11), legal compliance (Article 2(40) via EUDR-023), indigenous rights (via EUDR-021), and protected area status (via EUDR-022)
- [ ] F3.2: Provides scheme-specific audit checklists for FSC (Principles 1-10), PEFC (Criteria 1-7), RSPO (Principles 1-7), Rainforest Alliance (Chapter 1-6), and ISCC (Sustainability Requirements 1-6) with auto-mapping to EUDR articles
- [ ] F3.3: Tracks checklist completion in real-time: percentage of criteria assessed, pass/fail/not-applicable per criterion, evidence attachment per criterion, and auditor notes
- [ ] F3.4: Manages evidence collection: document upload with type classification (permit, certificate, photo, GPS record, interview transcript, lab result), metadata tagging (date, location, source), and SHA-256 hash for integrity verification
- [ ] F3.5: Implements sampling plan management per ISO 19011:2018 Annex A: defines sampling methodology (statistical/judgmental), calculates sample size based on population and risk level, tracks sampled items, and records sampling rationale for audit report
- [ ] F3.6: Tracks audit fieldwork schedule: site visit itinerary, stakeholder interview schedule, document review sessions, opening/closing meeting coordination, and travel logistics
- [ ] F3.7: Supports audit modality management: on-site audit, remote/desktop audit, hybrid audit (remote preparation + on-site verification), and unannounced audit
- [ ] F3.8: Generates real-time audit progress dashboard: overall completion percentage, findings-to-date summary, outstanding evidence requests, and timeline adherence
- [ ] F3.9: Manages audit team composition: lead auditor, co-auditors, technical experts, observers, and trainee auditors with role-based access to audit data
- [ ] F3.10: Tracks stakeholder interviews: community consultations, worker interviews, management interviews, government official meetings with structured interview templates and outcome records
- [ ] F3.11: Supports audit hold/suspension: when critical issues are discovered mid-audit that require immediate escalation before audit can continue
- [ ] F3.12: Records audit completion with formal closing meeting notes, preliminary findings summary, and expected report delivery date

**Non-Functional Requirements:**
- Real-time: Checklist updates visible to all authorized users within 30 seconds
- Storage: Evidence files up to 100 MB per document; total evidence package up to 5 GB per audit
- Offline: Audit checklists downloadable for offline completion during field visits (sync on reconnect)
- Security: Evidence files encrypted at rest (AES-256); access restricted to audit team and authorized stakeholders

**Dependencies:**
- EUDR-021 Indigenous Rights Checker (FPIC audit criteria)
- EUDR-022 Protected Area Validator (protected area audit criteria)
- EUDR-023 Legal Compliance Verifier (legal compliance audit criteria)
- SEC-003 Encryption at Rest (evidence file encryption)
- S3 Object Storage (evidence file storage)

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 frontend engineer)

---

#### Feature 4: Non-Conformance Detection Engine

**User Story:**
```
As a lead auditor,
I want to classify audit findings into structured non-conformances with severity levels and root cause analysis,
So that I can provide clear, actionable findings that map to specific EUDR requirements and certification scheme clauses.
```

**Acceptance Criteria (12 sub-requirements):**
- [ ] F4.1: Classifies non-conformances into three severity levels per ISO 19011 and certification scheme conventions:
  - **Critical NC**: Immediate threat to product integrity, human safety, or environmental protection; systemic failure; evidence of fraud or intentional non-compliance. Requires immediate corrective action (0-30 days). May trigger certification suspension.
  - **Major NC**: Significant failure to meet a requirement that undermines the effectiveness of the management system; single major deviation from audit criteria. Requires corrective action within 90 days (3 months).
  - **Minor NC**: Isolated or minor deviation that does not undermine system effectiveness; single minor non-fulfillment of a requirement. Requires corrective action within 365 days (12 months).
- [ ] F4.2: Maps each NC to specific EUDR articles (Art. 3, 4, 9, 10, 11, 29, 31), certification scheme clauses (FSC Principle/Criterion, RSPO P&C number, etc.), and applicable legislation category (Article 2(40) categories 1-8)
- [ ] F4.3: Implements root cause analysis (RCA) frameworks: "5 Whys" structured questioning, Ishikawa (fishbone) diagram with 6 categories (People, Process, Equipment, Materials, Environment, Management), and direct/contributing/root cause classification
- [ ] F4.4: Links NCs to evidence collected during audit execution (F3): evidence attachments, checklist references, interview records, and photographic evidence
- [ ] F4.5: Detects NC patterns across audits: recurring findings for same supplier (repeat NCs), common findings across suppliers in same region (systemic issues), and seasonal patterns (harvest season non-compliance)
- [ ] F4.6: Assigns NC risk impact score: combination of severity, affected EUDR article criticality, supply chain impact (volume and value of affected products), and supplier risk level
- [ ] F4.7: Supports NC status lifecycle: OPEN (newly identified) -> ACKNOWLEDGED (auditee confirms receipt) -> CAR_ISSUED (corrective action requested) -> CAP_SUBMITTED (action plan received) -> IN_PROGRESS (corrective action underway) -> VERIFICATION_PENDING (evidence submitted) -> CLOSED (verified resolved) or ESCALATED (unresolved within SLA)
- [ ] F4.8: Generates NC summary reports per audit, per supplier, per region, and per commodity with distribution charts and trend analysis
- [ ] F4.9: Supports NC dispute process: auditee can dispute NC classification with supporting evidence; dispute reviewed by lead auditor; outcome recorded with rationale
- [ ] F4.10: Flags "observations" and "opportunities for improvement" (OFI) as separate from NCs per ISO 19011 convention; tracked but not subject to CAR requirements
- [ ] F4.11: Integrates with EUDR-023 Legal Compliance Verifier for NCs related to Article 2(40) legal compliance categories
- [ ] F4.12: All NC classifications are deterministic: same finding description and evidence inputs produce same severity classification (rule-based, no LLM)

**NC Severity Classification Rules:**
```
CRITICAL if any of:
  - Evidence of deforestation after Dec 31, 2020 cutoff (Art. 3 violation)
  - Fraudulent documentation detected (forged permits, fabricated GPS coordinates)
  - Production in protected area without authorization (Art. 9/10)
  - Evidence of forced labour or child labour (Art. 2(40) Cat 5)
  - Intentional mixing of compliant and non-compliant product without disclosure
  - Missing geolocation data for > 50% of production plots

MAJOR if any of:
  - Missing geolocation for 10-50% of production plots (Art. 9 partial)
  - Incomplete chain of custody documentation for > 20% of batches
  - Expired certification with no renewal application
  - Non-compliance with > 2 Article 2(40) categories (EUDR-023)
  - Mass balance discrepancy > 5% (Art. 10(2)(f))
  - No risk assessment documented for high-risk country supply chain

MINOR if any of:
  - Administrative documentation gaps (< 10% of records)
  - Minor mass balance discrepancy (1-5%)
  - Certification scheme procedural non-compliance (e.g., late annual report)
  - Training records not up to date
  - Single Article 2(40) category with partial compliance
```

**Non-Functional Requirements:**
- Accuracy: >= 95% classification agreement with expert auditor panel (validated against 500 test cases)
- Performance: NC classification < 500ms per finding
- Determinism: Same inputs produce same classification (bit-perfect)
- Auditability: Classification rationale recorded with rule reference and evidence links

**Dependencies:**
- Feature 3 (Audit Execution Tracker) for evidence and checklist data
- EUDR-023 Legal Compliance Verifier for legal compliance NC mapping
- EUDR-021 Indigenous Rights Checker for FPIC-related NCs
- EUDR-022 Protected Area Validator for protected area NCs

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

#### Feature 5: Corrective Action Request (CAR) Manager

**User Story:**
```
As a compliance officer,
I want to issue, track, and verify corrective action requests with enforced SLA deadlines and escalation rules,
So that audit findings are resolved effectively and within required timelines per certification scheme and regulatory requirements.
```

**Acceptance Criteria (12 sub-requirements):**
- [ ] F5.1: Issues CARs automatically upon NC classification: each NC with severity >= Minor generates a CAR with severity-based SLA deadline (Critical: 30 days, Major: 90 days, Minor: 365 days), assigned to the auditee's designated compliance contact
- [ ] F5.2: Manages CAR lifecycle stages: ISSUED -> ACKNOWLEDGED (auditee confirms within 7 days) -> RCA_SUBMITTED (root cause analysis within 14 days) -> CAP_SUBMITTED (corrective action plan within 30 days for Critical/Major) -> IN_PROGRESS (implementation underway) -> EVIDENCE_SUBMITTED (auditee submits closure evidence) -> VERIFICATION (auditor reviews evidence) -> CLOSED or REJECTED (back to IN_PROGRESS)
- [ ] F5.3: Enforces SLA deadlines with countdown tracking: real-time SLA countdown displayed on CAR dashboard; amber warning at 75% of SLA elapsed; red warning at 90% of SLA elapsed
- [ ] F5.4: Implements escalation rules for overdue CARs:
  - Stage 1 (SLA at 75%): Email notification to auditee and operator compliance officer
  - Stage 2 (SLA at 90%): Escalation to audit programme manager and supplier relationship manager
  - Stage 3 (SLA exceeded): Escalation to Head of Compliance; CAR status set to OVERDUE; supplier risk score increased via EUDR-017
  - Stage 4 (SLA exceeded by 30+ days for Critical/Major): Certification suspension recommendation; competent authority notification recommendation
- [ ] F5.5: Manages corrective action plan (CAP) review: lead auditor reviews submitted CAP for adequacy (addresses root cause, defines specific actions, includes timeline, assigns responsibility); approves or returns for revision with feedback
- [ ] F5.6: Tracks implementation evidence: auditee uploads evidence of corrective action implementation (photos, documents, updated procedures, training records, system screenshots); evidence linked to specific CAR and NC
- [ ] F5.7: Manages effectiveness verification: follow-up audit or desktop review to verify corrective action has been effective; verification can be performed by original auditor or different qualified auditor; verification outcome recorded with evidence
- [ ] F5.8: Supports CAR grouping: multiple NCs from same audit with related root cause can be grouped under a single CAR with consolidated corrective action plan
- [ ] F5.9: Tracks CAR metrics per supplier: total CARs issued, open CARs by severity, average closure time, SLA compliance rate, repeat CARs (same finding type), and closure trend
- [ ] F5.10: Generates CAR status reports: per supplier, per audit, per region, per certification scheme, with overdue highlight and escalation status
- [ ] F5.11: Integrates with EUDR-017 Supplier Risk Scorer: open CARs (especially Critical and Major) feed supplier risk score; CAR closure positively adjusts risk score
- [ ] F5.12: Supports competent authority-issued CARs: when competent authorities issue corrective action requirements under Article 18, these are tracked alongside scheme-issued CARs with authority-specific SLA timelines

**CAR SLA Timeline Summary:**
| NC Severity | CAR Deadline | Acknowledge | RCA Due | CAP Due | Escalation Stage 1 | Escalation Stage 2 | Escalation Stage 3 |
|-------------|-------------|-------------|---------|---------|--------------------|--------------------|---------------------|
| Critical | 30 days | 3 days | 7 days | 14 days | Day 22 (75%) | Day 27 (90%) | Day 31 |
| Major | 90 days | 7 days | 14 days | 30 days | Day 67 (75%) | Day 81 (90%) | Day 91 |
| Minor | 365 days | 14 days | 30 days | 60 days | Day 274 (75%) | Day 328 (90%) | Day 366 |

**Non-Functional Requirements:**
- SLA Accuracy: Real-time countdown with < 1 minute update latency; timezone-aware deadlines
- Notification: Escalation emails sent within 5 minutes of trigger condition
- Persistence: CAR records retained for minimum 5 years per Article 31
- Performance: CAR dashboard loads in < 2 seconds for 1,000 active CARs

**Dependencies:**
- Feature 4 (Non-Conformance Detection Engine) for NC data
- EUDR-017 Supplier Risk Scorer (CAR status feeds supplier risk)
- Email notification service
- SEC-001 JWT Authentication (auditee access to CAR portal)

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 frontend engineer)

---

**P0 Features 6-9: Integration, Reporting, and Analytics Layer**

> Features 6, 7, 8, and 9 are P0 launch blockers. Without certification scheme integration, standardized reporting, competent authority management, and analytics, the core audit lifecycle engine cannot deliver end-user value in the multi-scheme, multi-authority EUDR compliance environment.

---

#### Feature 6: Certification Scheme Integration

**User Story:**
```
As a compliance officer,
I want the system to integrate with all major certification schemes so I can coordinate audits across schemes and reduce redundant verification activities,
So that my suppliers are not burdened with overlapping audits and I have a unified view of certification compliance.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F6.1: Integrates with 5 certification scheme databases to import certificate status:
  - FSC: Certificate number, status (active/suspended/terminated), scope (FM/CoC), expiry date, certified products, certified sites
  - PEFC: Certificate number, status, scope (SFM/CoC), expiry date, certified products
  - RSPO: Membership number, certification status, supply chain model (IP/SG/MB), expiry date, certified mills/estates
  - Rainforest Alliance: Certificate holder ID, status, scope, expiry date, certified commodities, certified origins
  - ISCC: Certificate number, status, scope, expiry date, valid raw materials, valid sites
- [ ] F6.2: Maps certification scheme audit requirements to EUDR articles: generates a coverage matrix showing which EUDR articles and Article 2(40) categories are covered by each scheme's audit criteria (FULL/PARTIAL/NONE)
- [ ] F6.3: Identifies audit scope overlap across schemes: when a supplier holds multiple certifications, calculates shared audit criteria (e.g., FSC P1 + RSPO P2 both cover legal compliance) and recommends combined audit scope to reduce redundancy
- [ ] F6.4: Coordinates audit scheduling across schemes: aligns audit windows to minimize separate site visits; proposes combined audit dates when scheme rules allow; tracks scheme-specific scheduling constraints (e.g., RSPO requires audit within 12 months of previous)
- [ ] F6.5: Imports certification audit results: scheme audit findings, NC classifications (mapped to GreenLang NC taxonomy), certification decision (certified/suspended/withdrawn), and next audit date
- [ ] F6.6: Exports EUDR compliance evidence to certification bodies: provides structured evidence packages for scheme auditors assessing EUDR-relevant criteria during certification audits
- [ ] F6.7: Monitors certification status changes: daily sync with scheme databases; alerts when certificate status changes (new suspension, withdrawal, reinstatement, scope change); triggers re-evaluation of affected supplier audit schedule
- [ ] F6.8: Tracks mutual recognition agreements: identifies which scheme pairs have mutual recognition (e.g., PEFC recognizes certain national schemes); applies recognition to reduce audit scope where applicable
- [ ] F6.9: Generates certification coverage reports per supplier: which EUDR requirements are covered by certification vs. requiring direct verification; gap identification with recommended actions
- [ ] F6.10: Supports scheme-specific NC classification mapping: translates each scheme's NC taxonomy (FSC: Major/Minor/OFI; RSPO: Major/Minor/Observation; RA: Critical/Major/Minor/IN) to the GreenLang unified NC taxonomy

**Scheme-to-EUDR Coverage Matrix:**
| EUDR Requirement | FSC | PEFC | RSPO | RA | ISCC |
|------------------|-----|------|------|-----|------|
| Art. 3 Deforestation-free | FULL (P6, P9) | FULL (C4, C5) | FULL (P7 HCV/HCS) | FULL (Ch 2) | FULL (SR 1) |
| Art. 9 Geolocation | PARTIAL (FM scope only) | PARTIAL (FM scope) | FULL (estate/plot mapping) | PARTIAL (farm mapping) | PARTIAL (site mapping) |
| Art. 10 Risk assessment | FULL (P1-P10) | FULL (C1-C7) | FULL (P1-P7) | FULL (Ch 1-6) | PARTIAL (SR 1-6) |
| Art. 2(40) Cat 1 Land use | FULL (P2, P3) | FULL (C1) | FULL (P2) | FULL (Ch 4) | PARTIAL (SR 2) |
| Art. 2(40) Cat 2 Environment | FULL (P6, P9) | FULL (C4, C5) | FULL (P5) | FULL (Ch 2) | FULL (SR 1) |
| Art. 2(40) Cat 3 Forest rules | FULL (P1, P5, P6) | FULL (C1, C3) | PARTIAL (P7) | PARTIAL (Ch 2) | NONE |
| Art. 2(40) Cat 4 Third party | FULL (P3, P4) | FULL (C2) | FULL (P2, P6) | FULL (Ch 5) | PARTIAL (SR 5) |
| Art. 2(40) Cat 5 Labour | FULL (P2, P4) | FULL (C2) | FULL (P6) | FULL (Ch 5) | PARTIAL (SR 5) |
| Art. 2(40) Cat 6 Human rights | FULL (P2, P4) | PARTIAL (C2) | FULL (P1, P6) | FULL (Ch 5) | PARTIAL (SR 5) |
| Art. 2(40) Cat 7 FPIC | FULL (P3, P4) | PARTIAL (C2) | FULL (P2) | FULL (Ch 5.4) | NONE |
| Art. 2(40) Cat 8 Tax/customs | FULL (P1) | FULL (C1) | PARTIAL (P1) | PARTIAL (Ch 4) | PARTIAL (SR 3) |

**Non-Functional Requirements:**
- Data Freshness: Certification status synced within 24 hours of scheme database update
- Performance: Coverage matrix generation < 500ms per supplier
- Reliability: Adapter pattern for each scheme with circuit breaker; manual import fallback
- Privacy: Certification data stored per scheme data sharing agreements

**Dependencies:**
- FSC Certificate Database API
- RSPO Certificate Database / ACOP
- PEFC Certificate Search
- Rainforest Alliance Certificate Database
- ISCC Certificate Database
- EUDR-023 Legal Compliance Verifier (Article 2(40) coverage data)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 integration engineer)

---

#### Feature 7: Audit Report Generator

**User Story:**
```
As a lead auditor,
I want to generate standardized audit reports that comply with ISO 19011:2018 requirements,
So that audit documentation is consistent, complete, and accepted by certification bodies and competent authorities.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F7.1: Generates audit reports structured per ISO 19011:2018 Clause 6.6 with the following sections: audit objectives, audit scope, audit criteria, audit client identification, audit team members, dates and locations, audit findings (categorized by severity), audit conclusions, audit recommendations, and distribution list
- [ ] F7.2: Includes evidence package: cross-referenced evidence documents with finding ID linkage, evidence type classification, and SHA-256 integrity hashes; evidence package can be exported as a separate zip archive
- [ ] F7.3: Documents sampling rationale per ISO 19011 Annex A: sample size calculation, sampling methodology (statistical/judgmental), sampled items list, and representativeness justification
- [ ] F7.4: Includes auditor credential summary: lead auditor qualifications, team member credentials, conflict-of-interest declarations, and competence match to audit scope
- [ ] F7.5: Generates finding detail sections: each NC includes finding statement, objective evidence, severity classification with rationale, EUDR article reference, certification scheme clause reference, root cause analysis, and corrective action timeline
- [ ] F7.6: Includes certification scheme-specific report sections: FSC audit report format (FSC-PRO-20-003), RSPO assessment report format, RA audit report requirements, PEFC report requirements, ISCC audit report format
- [ ] F7.7: Supports 5 output formats: PDF (primary), JSON (machine-readable), HTML (web display), XLSX (data analysis), and structured XML (regulatory submission)
- [ ] F7.8: Supports 5 languages: English, French, German, Spanish, and Portuguese with template-based rendering and locale-appropriate date/number formatting
- [ ] F7.9: Includes report metadata: report ID, generation timestamp, report version, generator agent version, SHA-256 hash of complete report for integrity verification, and provenance chain from audit data to report
- [ ] F7.10: Supports report amendment workflow: when post-report corrections are needed, generates amended report with change tracking (delta), original report preserved, and amendment rationale recorded

**Non-Functional Requirements:**
- Performance: Report generation < 30 seconds for a 50-finding audit report with 200 evidence attachments
- Compliance: Report structure validated against ISO 19011 Clause 6.6 checklist
- Integrity: SHA-256 hash on every generated report; tamper detection on subsequent access
- Storage: Reports stored in S3 with versioning; retained for 5 years per Article 31

**Dependencies:**
- Feature 3 (Audit Execution Tracker) for checklist, evidence, and fieldwork data
- Feature 4 (Non-Conformance Detection Engine) for NC classification and RCA data
- Feature 2 (Auditor Registry) for auditor credential data
- S3 Object Storage (report storage)
- PDF generation library (WeasyPrint or ReportLab)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 template engineer)

---

#### Feature 8: Competent Authority Liaison

**User Story:**
```
As a compliance officer,
I want a structured workflow for managing interactions with EU competent authorities,
So that I can respond to inspection requests, document demands, and enforcement actions within regulatory deadlines.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F8.1: Maintains profiles for 27 EU Member State competent authorities: authority name, contact details, legal basis for enforcement (national implementing legislation), inspection focus areas, typical document request formats, and response timeline requirements
- [ ] F8.2: Receives and logs authority interactions by type: Document Request (Article 15 -- written request for DDS, supply chain evidence, audit reports), Inspection Notification (Article 15 -- advance notice of on-site inspection), Unannounced Inspection (Article 15(2) -- no advance notice), Corrective Action Order (Article 18), Interim Measure (Article 19), Definitive Measure (Article 20), and Information Request (Article 21)
- [ ] F8.3: Tracks response deadlines with SLA countdown: configurable per Member State and interaction type (default 30 days for document requests; 5 days for urgent requests; immediate for on-site inspections)
- [ ] F8.4: Coordinates internal response preparation: assigns response tasks to relevant internal teams (compliance, legal, supply chain, quality), tracks task completion, assembles evidence package from audit records, and manages internal review/approval workflow before submission
- [ ] F8.5: Generates authority-ready evidence packages: compiles relevant DDS data, supply chain graphs (from EUDR-001), audit reports (from F7), NC status, CAR closure evidence, and certification status into a structured package matching authority request format
- [ ] F8.6: Manages on-site inspection logistics: inspector access scheduling, document preparation, site access coordination, escort assignment, interview room booking, and post-inspection debrief tracking
- [ ] F8.7: Tracks enforcement measures and compliance restoration: if authority issues suspension (Article 19) or definitive measure (Article 20), tracks compliance restoration requirements, monitors restoration progress, and documents compliance restoration evidence for authority review
- [ ] F8.8: Records all authority interactions in immutable audit trail: incoming communications, internal actions, outgoing responses, evidence submitted, and authority decisions
- [ ] F8.9: Generates authority interaction reports: per Member State, per interaction type, per time period, with compliance rate (responses within SLA), enforcement action tracking, and trend analysis
- [ ] F8.10: Supports multi-authority coordination: when same product/supplier is subject to checks by multiple Member State authorities, coordinates responses to ensure consistency and avoid conflicting submissions

**Non-Functional Requirements:**
- Response Time: SLA countdown visible in real-time with < 1 minute update latency
- Notification: Escalation emails sent within 5 minutes of deadline warning threshold
- Security: Authority communications encrypted in transit and at rest; access restricted to authorized personnel
- Compliance: All interactions archived for 5 years per Article 31

**Dependencies:**
- EUDR-001 Supply Chain Mapping Master (supply chain evidence for authority responses)
- Feature 7 (Audit Report Generator) for audit report evidence
- GL-EUDR-APP DDS module (DDS data for authority requests)
- Email notification service

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend engineer)

---

#### Feature 9: Audit Analytics and Trends

**User Story:**
```
As a head of compliance,
I want comprehensive analytics on audit findings, auditor performance, and compliance trends,
So that I can identify systemic issues, allocate resources effectively, and demonstrate continuous improvement to stakeholders.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F9.1: Provides finding trend analysis: NC count by severity over time (monthly/quarterly/annual), by region, by commodity, by certification scheme, and by EUDR article; identifies increasing/decreasing trends with statistical significance
- [ ] F9.2: Provides auditor performance benchmarking: findings per audit day, NC severity distribution per auditor, average audit duration, CAR closure rate for auditor-identified findings, and client satisfaction scores; anonymized comparison against auditor pool average
- [ ] F9.3: Provides compliance rate tracking: percentage of audits with zero critical NCs, percentage of audits with zero major NCs, percentage of suppliers achieving full compliance, and compliance rate trend over time
- [ ] F9.4: Provides CAR performance analytics: average closure time by NC severity, SLA compliance rate, repeat CAR rate (same finding type for same supplier), escalation rate, and root cause category distribution
- [ ] F9.5: Provides cost analysis: average audit cost per supplier (auditor-days x day rate), cost per finding, cost per commodity, cost per region, and audit programme budget vs. actual spend tracking
- [ ] F9.6: Provides risk-audit correlation: analyzes relationship between supplier risk scores and audit finding severity; identifies suppliers where risk prediction is accurate vs. where audits reveal unexpected findings
- [ ] F9.7: Provides certification scheme analysis: scheme-specific finding rates, coverage gap analysis across schemes, mutual recognition impact on audit volume, and scheme audit quality comparison
- [ ] F9.8: Provides competent authority analytics: interaction frequency per Member State, response SLA compliance rate, enforcement action trends, and inspection outcome distribution
- [ ] F9.9: Generates executive dashboards with configurable KPIs: selectable time periods, drill-down by region/commodity/scheme/supplier, export to PDF/XLSX, and scheduled email distribution
- [ ] F9.10: All analytics calculations are deterministic: same audit data produces same analytics outputs (bit-perfect); all calculations use Decimal arithmetic for financial metrics

**Non-Functional Requirements:**
- Performance: Dashboard loads in < 3 seconds for 12-month analytics window with 5,000+ audit records
- Refresh: Analytics data refreshed hourly; real-time counters for active CARs and SLA status
- Export: All charts and tables exportable to PDF and XLSX
- Configurability: KPIs, thresholds, and time periods configurable per operator

**Dependencies:**
- Features 1-8 (all audit lifecycle data feeds analytics)
- PostgreSQL + TimescaleDB (time-series aggregation)
- Grafana (optional dashboard integration)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Predictive Audit Risk Model
- ML model trained on historical NC patterns to predict audit outcomes for scheduled audits
- Risk-based resource allocation recommendations (more auditor-days for predicted high-finding audits)
- Anomaly detection for unusual audit patterns (auditor always finding zero NCs, seasonal spikes)

#### Feature 11: Remote Audit Video Integration
- Secure video conferencing integration for remote audit sessions
- Screen sharing for document review sessions
- Video evidence capture and timestamped recording linked to audit checklist items
- Remote GPS verification through mobile camera geolocation

#### Feature 12: Supplier Self-Assessment Portal
- Supplier self-assessment questionnaires aligned with EUDR criteria
- Pre-audit readiness scoring based on self-assessment responses
- Document pre-upload for auditor review before on-site visit
- Self-assessment-to-audit-finding correlation analysis

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Accreditation body management and accreditation process workflow (agent verifies accreditation; does not manage it)
- Audit fee invoicing and financial management (agent tracks costs for analytics; does not process payments)
- Direct API submission to certification scheme registries (agent generates reports; submission is manual)
- Real-time IoT sensor integration for continuous monitoring between audits
- Mobile native application for field auditors (web responsive only)
- Blockchain-based audit evidence immutability (SHA-256 hashes provide sufficient integrity)
- Automated translation of audit findings across languages (template-based multi-language reports; not real-time translation)

---

## 7. Technical Architecture Overview

### 7.1 Architecture Diagram

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
        +------------------+----------------------+----------------------+------------------+
        |                  |                      |                      |                  |
+-------v-------+  +-------v-------+  +-----------v-----------+  +-------v-------+  +-------v-------+
| AGENT-EUDR-024|  | AGENT-EUDR-001|  | AGENT-EUDR-016        |  | AGENT-EUDR-017|  | AGENT-EUDR-020|
| Third-Party   |  | Supply Chain  |  | Country Risk           |  | Supplier Risk |  | Deforestation |
| Audit Manager |  | Mapping Master|  | Evaluator              |  | Scorer        |  | Alert System  |
|               |  |               |  |                        |  |               |  |               |
| - Audit Planner|  | - Graph Engine|  | - Risk Scoring         |  | - Composite   |  | - Alert Feed  |
| - Auditor Reg |  | - Multi-Tier  |  | - Country Benchmarking |  |   Risk Score  |  | - Proximity   |
| - Exec Tracker|  | - Gap Analysis|  | - Governance Index     |  | - NC History  |  |   Detection   |
| - NC Engine   |  |               |  |                        |  |               |  |               |
| - CAR Manager |  +---------------+  +------------------------+  +---------------+  +---------------+
| - Scheme Intg |
| - Report Gen  |         +---------------------------+         +---------------------------+
| - Auth Liaison|         | AGENT-EUDR-021            |         | AGENT-EUDR-022            |
| - Analytics   |         | Indigenous Rights Checker  |         | Protected Area Validator  |
+-------+-------+         | - FPIC Verification       |         | - Protected Area Checks   |
        |                  | - Audit FPIC Criteria     |         | - Audit PA Criteria       |
        |                  +---------------------------+         +---------------------------+
        |
        |                  +---------------------------+
        +----------------->| AGENT-EUDR-023            |
                           | Legal Compliance Verifier |
                           | - Legal Audit Verification|
                           | - Document Compliance     |
                           +---------------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/third_party_audit/
    __init__.py                          # Public API exports
    config.py                            # ThirdPartyAuditConfig with GL_EUDR_TAM_ env prefix
    models.py                            # Pydantic v2 models for audits, NCs, CARs, auditors
    audit_planner.py                     # AuditPlanningEngine: risk-based scheduling
    auditor_registry.py                  # AuditorRegistryEngine: qualification tracking
    audit_execution.py                   # AuditExecutionTracker: checklist and evidence management
    nc_engine.py                         # NonConformanceEngine: detection, classification, RCA
    car_manager.py                       # CARManager: lifecycle, SLA, escalation
    scheme_integration.py                # CertificationSchemeIntegration: 5-scheme connector
    report_generator.py                  # AuditReportGenerator: ISO 19011 compliant reports
    authority_liaison.py                 # CompetentAuthorityLiaison: regulatory interactions
    audit_analytics.py                   # AuditAnalyticsEngine: trends, benchmarking, KPIs
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains
    metrics.py                           # 20 Prometheus self-monitoring metrics
    setup.py                             # ThirdPartyAuditService facade
    api/
        __init__.py
        router.py                        # FastAPI router (~35 endpoints)
        planning_routes.py               # Audit planning and scheduling endpoints
        auditor_routes.py                # Auditor registry and matching endpoints
        execution_routes.py              # Audit execution and checklist endpoints
        nc_routes.py                     # Non-conformance management endpoints
        car_routes.py                    # CAR lifecycle endpoints
        scheme_routes.py                 # Certification scheme endpoints
        report_routes.py                 # Report generation endpoints
        authority_routes.py              # Competent authority liaison endpoints
        analytics_routes.py              # Analytics and dashboard endpoints
```

---

## 8. Data Model and Schemas

### 8.1 Key Data Models (Pydantic v2)

```python
# Audit Status Lifecycle
class AuditStatus(str, Enum):
    PLANNED = "planned"                  # Scheduled in audit calendar
    AUDITOR_ASSIGNED = "auditor_assigned"  # Auditor confirmed
    IN_PREPARATION = "in_preparation"    # Pre-audit documentation underway
    IN_PROGRESS = "in_progress"          # Audit execution active
    FIELDWORK_COMPLETE = "fieldwork_complete"  # On-site work done
    REPORT_DRAFTING = "report_drafting"  # Report being generated
    REPORT_ISSUED = "report_issued"      # Final report delivered
    CAR_FOLLOW_UP = "car_follow_up"      # CARs being tracked
    CLOSED = "closed"                    # All CARs resolved, audit complete
    CANCELLED = "cancelled"              # Audit cancelled

# Audit Scope
class AuditScope(str, Enum):
    FULL = "full"                        # All EUDR criteria + scheme criteria
    TARGETED = "targeted"                # Specific risk areas only
    SURVEILLANCE = "surveillance"        # Maintenance/follow-up
    UNSCHEDULED = "unscheduled"          # Event-triggered audit

# Audit Modality
class AuditModality(str, Enum):
    ON_SITE = "on_site"                  # Physical on-site audit
    REMOTE = "remote"                    # Desktop/remote audit
    HYBRID = "hybrid"                    # Remote prep + on-site verification
    UNANNOUNCED = "unannounced"          # Unannounced on-site audit

# NC Severity
class NCSeverity(str, Enum):
    CRITICAL = "critical"                # Immediate action required (30 days)
    MAJOR = "major"                      # Significant failure (90 days)
    MINOR = "minor"                      # Isolated deviation (365 days)
    OBSERVATION = "observation"          # Not an NC; improvement opportunity

# CAR Status
class CARStatus(str, Enum):
    ISSUED = "issued"
    ACKNOWLEDGED = "acknowledged"
    RCA_SUBMITTED = "rca_submitted"
    CAP_SUBMITTED = "cap_submitted"
    CAP_APPROVED = "cap_approved"
    IN_PROGRESS = "in_progress"
    EVIDENCE_SUBMITTED = "evidence_submitted"
    VERIFICATION_PENDING = "verification_pending"
    CLOSED = "closed"
    REJECTED = "rejected"
    OVERDUE = "overdue"
    ESCALATED = "escalated"

# Certification Scheme
class CertificationScheme(str, Enum):
    FSC = "fsc"
    PEFC = "pefc"
    RSPO = "rspo"
    RAINFOREST_ALLIANCE = "rainforest_alliance"
    ISCC = "iscc"

# Authority Interaction Type
class AuthorityInteractionType(str, Enum):
    DOCUMENT_REQUEST = "document_request"
    INSPECTION_NOTIFICATION = "inspection_notification"
    UNANNOUNCED_INSPECTION = "unannounced_inspection"
    CORRECTIVE_ACTION_ORDER = "corrective_action_order"
    INTERIM_MEASURE = "interim_measure"
    DEFINITIVE_MEASURE = "definitive_measure"
    INFORMATION_REQUEST = "information_request"

# Core Models
class Audit(BaseModel):
    audit_id: str                        # Unique audit identifier
    operator_id: str                     # Operator owning this audit
    supplier_id: str                     # Supplier being audited
    audit_type: AuditScope               # Full/Targeted/Surveillance/Unscheduled
    modality: AuditModality              # On-site/Remote/Hybrid/Unannounced
    certification_scheme: Optional[CertificationScheme]  # If scheme audit
    eudr_articles: List[str]             # EUDR articles in audit scope
    planned_date: date                   # Scheduled audit date
    actual_start_date: Optional[date]    # Actual start
    actual_end_date: Optional[date]      # Actual completion
    lead_auditor_id: str                 # Assigned lead auditor
    audit_team: List[str]                # Audit team member IDs
    status: AuditStatus
    priority_score: Decimal              # Risk-based priority (0-100)
    country_code: str                    # Country of audited site
    commodity: str                       # Primary commodity
    site_ids: List[str]                  # Audited site identifiers
    checklist_completion: Decimal        # 0-100 percentage
    findings_count: Dict[str, int]       # {critical: N, major: N, minor: N, observation: N}
    evidence_count: int                  # Number of evidence items collected
    report_id: Optional[str]            # Generated report ID
    provenance_hash: str                 # SHA-256
    created_at: datetime
    updated_at: datetime

class Auditor(BaseModel):
    auditor_id: str
    full_name: str
    organization: str                    # Employing CB or audit firm
    accreditation_body: str              # IAF MLA signatory body
    accreditation_status: str            # active/suspended/withdrawn
    accreditation_expiry: date
    commodity_competencies: List[str]    # EUDR commodities qualified for
    scheme_qualifications: List[str]     # FSC Lead Auditor, RSPO Assessor, etc.
    country_expertise: List[str]         # ISO 3166-1 alpha-2 codes
    languages: List[str]                 # ISO 639-1 codes
    conflict_of_interest: List[Dict]     # CoI declarations
    audit_count: int                     # Total audits conducted
    performance_rating: Decimal          # 0-100
    cpd_hours: int                       # Continuing professional development
    cpd_compliant: bool
    created_at: datetime
    updated_at: datetime

class NonConformance(BaseModel):
    nc_id: str
    audit_id: str
    finding_statement: str               # What was found
    objective_evidence: str              # Evidence supporting the finding
    severity: NCSeverity
    eudr_article: Optional[str]          # Mapped EUDR article
    scheme_clause: Optional[str]         # Mapped scheme clause
    article_2_40_category: Optional[str] # Mapped legislation category
    root_cause_analysis: Optional[Dict]  # 5-Whys or Ishikawa output
    risk_impact_score: Decimal           # 0-100
    status: str                          # NC lifecycle status
    car_id: Optional[str]               # Linked CAR
    evidence_ids: List[str]             # Linked evidence items
    disputed: bool
    dispute_rationale: Optional[str]
    provenance_hash: str
    detected_at: datetime
    resolved_at: Optional[datetime]

class CorrectiveActionRequest(BaseModel):
    car_id: str
    nc_ids: List[str]                    # Linked NCs (can be grouped)
    audit_id: str
    supplier_id: str
    severity: NCSeverity                 # Highest severity of linked NCs
    sla_deadline: datetime               # Calculated from severity
    sla_status: str                      # on_track/warning/critical/overdue
    status: CARStatus
    issued_by: str                       # Auditor or authority ID
    issued_at: datetime
    acknowledged_at: Optional[datetime]
    rca_submitted_at: Optional[datetime]
    cap_submitted_at: Optional[datetime]
    cap_approved_at: Optional[datetime]
    evidence_submitted_at: Optional[datetime]
    verified_at: Optional[datetime]
    closed_at: Optional[datetime]
    corrective_action_plan: Optional[Dict]  # Structured CAP
    verification_outcome: Optional[str]  # effective/not_effective
    escalation_level: int                # 0-4
    provenance_hash: str

class CompetentAuthorityInteraction(BaseModel):
    interaction_id: str
    operator_id: str
    authority_name: str
    member_state: str                    # ISO 3166-1 alpha-2
    interaction_type: AuthorityInteractionType
    received_date: datetime
    response_deadline: datetime
    response_sla_status: str             # on_track/warning/overdue
    internal_tasks: List[Dict]           # Response preparation tasks
    evidence_package_id: Optional[str]
    response_submitted_at: Optional[datetime]
    authority_decision: Optional[str]
    enforcement_measures: Optional[List[Dict]]
    status: str                          # open/in_progress/responded/closed
    provenance_hash: str
```

### 8.2 Database Schema (New Migration: V112)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_third_party_audit;

-- ============================================================
-- Table 1: Audits (core audit records)
-- ============================================================
CREATE TABLE eudr_third_party_audit.audits (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    supplier_id UUID NOT NULL,
    audit_type VARCHAR(30) NOT NULL DEFAULT 'full',
    modality VARCHAR(30) NOT NULL DEFAULT 'on_site',
    certification_scheme VARCHAR(50),
    eudr_articles JSONB DEFAULT '[]',
    planned_date DATE NOT NULL,
    actual_start_date DATE,
    actual_end_date DATE,
    lead_auditor_id UUID,
    audit_team JSONB DEFAULT '[]',
    status VARCHAR(30) NOT NULL DEFAULT 'planned',
    priority_score NUMERIC(5,2) DEFAULT 0.0,
    country_code CHAR(2) NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    site_ids JSONB DEFAULT '[]',
    checklist_completion NUMERIC(5,2) DEFAULT 0.0,
    findings_count JSONB DEFAULT '{"critical":0,"major":0,"minor":0,"observation":0}',
    evidence_count INTEGER DEFAULT 0,
    report_id UUID,
    trigger_reason VARCHAR(200),
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Table 2: Auditors (auditor registry)
-- ============================================================
CREATE TABLE eudr_third_party_audit.auditors (
    auditor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(500) NOT NULL,
    organization VARCHAR(500) NOT NULL,
    accreditation_body VARCHAR(200),
    accreditation_status VARCHAR(30) DEFAULT 'active',
    accreditation_expiry DATE,
    accreditation_scope JSONB DEFAULT '[]',
    commodity_competencies JSONB DEFAULT '[]',
    scheme_qualifications JSONB DEFAULT '[]',
    country_expertise JSONB DEFAULT '[]',
    languages JSONB DEFAULT '[]',
    conflict_of_interest JSONB DEFAULT '[]',
    audit_count INTEGER DEFAULT 0,
    performance_rating NUMERIC(5,2) DEFAULT 0.0,
    cpd_hours INTEGER DEFAULT 0,
    cpd_compliant BOOLEAN DEFAULT TRUE,
    contact_email VARCHAR(500),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Table 3: Audit checklists
-- ============================================================
CREATE TABLE eudr_third_party_audit.audit_checklists (
    checklist_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id UUID NOT NULL REFERENCES eudr_third_party_audit.audits(audit_id),
    checklist_type VARCHAR(50) NOT NULL,
    checklist_version VARCHAR(20) NOT NULL,
    criteria JSONB NOT NULL DEFAULT '[]',
    completion_percentage NUMERIC(5,2) DEFAULT 0.0,
    total_criteria INTEGER DEFAULT 0,
    passed_criteria INTEGER DEFAULT 0,
    failed_criteria INTEGER DEFAULT 0,
    na_criteria INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Table 4: Audit evidence
-- ============================================================
CREATE TABLE eudr_third_party_audit.audit_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id UUID NOT NULL REFERENCES eudr_third_party_audit.audits(audit_id),
    evidence_type VARCHAR(50) NOT NULL,
    file_name VARCHAR(500),
    file_path VARCHAR(1000),
    file_size_bytes BIGINT,
    mime_type VARCHAR(100),
    description TEXT,
    tags JSONB DEFAULT '[]',
    location_latitude DOUBLE PRECISION,
    location_longitude DOUBLE PRECISION,
    captured_date TIMESTAMPTZ,
    sha256_hash VARCHAR(64) NOT NULL,
    uploaded_by VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Table 5: Non-conformances
-- ============================================================
CREATE TABLE eudr_third_party_audit.non_conformances (
    nc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id UUID NOT NULL REFERENCES eudr_third_party_audit.audits(audit_id),
    finding_statement TEXT NOT NULL,
    objective_evidence TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    eudr_article VARCHAR(20),
    scheme_clause VARCHAR(100),
    article_2_40_category VARCHAR(50),
    root_cause_analysis JSONB,
    root_cause_method VARCHAR(30),
    risk_impact_score NUMERIC(5,2) DEFAULT 0.0,
    status VARCHAR(30) NOT NULL DEFAULT 'open',
    car_id UUID,
    evidence_ids JSONB DEFAULT '[]',
    disputed BOOLEAN DEFAULT FALSE,
    dispute_rationale TEXT,
    classification_rules_applied JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

-- ============================================================
-- Table 6: Corrective Action Requests (CARs)
-- ============================================================
CREATE TABLE eudr_third_party_audit.corrective_action_requests (
    car_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nc_ids JSONB NOT NULL DEFAULT '[]',
    audit_id UUID NOT NULL REFERENCES eudr_third_party_audit.audits(audit_id),
    supplier_id UUID NOT NULL,
    severity VARCHAR(20) NOT NULL,
    sla_deadline TIMESTAMPTZ NOT NULL,
    sla_status VARCHAR(20) DEFAULT 'on_track',
    status VARCHAR(30) NOT NULL DEFAULT 'issued',
    issued_by VARCHAR(100) NOT NULL,
    issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    rca_submitted_at TIMESTAMPTZ,
    cap_submitted_at TIMESTAMPTZ,
    cap_approved_at TIMESTAMPTZ,
    evidence_submitted_at TIMESTAMPTZ,
    verified_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    corrective_action_plan JSONB,
    verification_outcome VARCHAR(30),
    verification_evidence_ids JSONB DEFAULT '[]',
    escalation_level INTEGER DEFAULT 0,
    escalation_history JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Table 7: Certification scheme certificates
-- ============================================================
CREATE TABLE eudr_third_party_audit.certification_certificates (
    certificate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    scheme VARCHAR(50) NOT NULL,
    certificate_number VARCHAR(200) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'active',
    scope VARCHAR(200),
    supply_chain_model VARCHAR(30),
    issue_date DATE,
    expiry_date DATE,
    certified_products JSONB DEFAULT '[]',
    certified_sites JSONB DEFAULT '[]',
    certification_body VARCHAR(500),
    last_audit_date DATE,
    next_audit_date DATE,
    eudr_coverage_matrix JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    synced_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(scheme, certificate_number)
);

-- ============================================================
-- Table 8: Audit reports
-- ============================================================
CREATE TABLE eudr_third_party_audit.audit_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id UUID NOT NULL REFERENCES eudr_third_party_audit.audits(audit_id),
    report_type VARCHAR(50) NOT NULL DEFAULT 'iso_19011',
    report_version INTEGER DEFAULT 1,
    language VARCHAR(5) DEFAULT 'en',
    format VARCHAR(10) NOT NULL DEFAULT 'pdf',
    file_path VARCHAR(1000),
    file_size_bytes BIGINT,
    sha256_hash VARCHAR(64) NOT NULL,
    sections JSONB DEFAULT '{}',
    finding_count JSONB DEFAULT '{}',
    evidence_package_path VARCHAR(1000),
    is_amended BOOLEAN DEFAULT FALSE,
    amendment_rationale TEXT,
    previous_version_id UUID,
    generated_by VARCHAR(100),
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Table 9: Competent authority interactions
-- ============================================================
CREATE TABLE eudr_third_party_audit.authority_interactions (
    interaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    authority_name VARCHAR(500) NOT NULL,
    member_state CHAR(2) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    received_date TIMESTAMPTZ NOT NULL,
    response_deadline TIMESTAMPTZ NOT NULL,
    response_sla_status VARCHAR(20) DEFAULT 'on_track',
    internal_tasks JSONB DEFAULT '[]',
    evidence_package_id UUID,
    response_submitted_at TIMESTAMPTZ,
    authority_decision TEXT,
    enforcement_measures JSONB DEFAULT '[]',
    status VARCHAR(30) NOT NULL DEFAULT 'open',
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Table 10: Audit schedule (hypertable for time-series planning)
-- ============================================================
CREATE TABLE eudr_third_party_audit.audit_schedule (
    schedule_id UUID DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    supplier_id UUID NOT NULL,
    planned_quarter VARCHAR(7) NOT NULL,
    audit_type VARCHAR(30) NOT NULL,
    modality VARCHAR(30) NOT NULL,
    priority_score NUMERIC(5,2),
    risk_factors JSONB DEFAULT '{}',
    assigned_auditor_id UUID,
    certification_scheme VARCHAR(50),
    status VARCHAR(30) DEFAULT 'planned',
    linked_audit_id UUID,
    scheduled_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_third_party_audit.audit_schedule', 'scheduled_at');

-- ============================================================
-- Table 11: NC trend log (hypertable)
-- ============================================================
CREATE TABLE eudr_third_party_audit.nc_trend_log (
    log_id UUID DEFAULT gen_random_uuid(),
    audit_id UUID NOT NULL,
    supplier_id UUID NOT NULL,
    nc_id UUID NOT NULL,
    severity VARCHAR(20) NOT NULL,
    eudr_article VARCHAR(20),
    scheme_clause VARCHAR(100),
    country_code CHAR(2),
    commodity VARCHAR(50),
    root_cause_category VARCHAR(100),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_third_party_audit.nc_trend_log', 'recorded_at');

-- ============================================================
-- Table 12: CAR SLA tracking log (hypertable)
-- ============================================================
CREATE TABLE eudr_third_party_audit.car_sla_log (
    log_id UUID DEFAULT gen_random_uuid(),
    car_id UUID NOT NULL,
    previous_status VARCHAR(30),
    new_status VARCHAR(30),
    sla_remaining_days INTEGER,
    escalation_level INTEGER,
    changed_by VARCHAR(100),
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_third_party_audit.car_sla_log', 'changed_at');

-- ============================================================
-- Table 13: Audit trail (immutable log)
-- ============================================================
CREATE TABLE eudr_third_party_audit.audit_trail (
    trail_id UUID DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    before_value JSONB,
    after_value JSONB,
    actor VARCHAR(100) NOT NULL,
    ip_address VARCHAR(45),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_third_party_audit.audit_trail', 'recorded_at');

-- ============================================================
-- Indexes
-- ============================================================
CREATE INDEX idx_audits_operator ON eudr_third_party_audit.audits(operator_id);
CREATE INDEX idx_audits_supplier ON eudr_third_party_audit.audits(supplier_id);
CREATE INDEX idx_audits_status ON eudr_third_party_audit.audits(status);
CREATE INDEX idx_audits_planned_date ON eudr_third_party_audit.audits(planned_date);
CREATE INDEX idx_audits_country ON eudr_third_party_audit.audits(country_code);
CREATE INDEX idx_audits_scheme ON eudr_third_party_audit.audits(certification_scheme);
CREATE INDEX idx_auditors_accreditation ON eudr_third_party_audit.auditors(accreditation_status);
CREATE INDEX idx_auditors_org ON eudr_third_party_audit.auditors(organization);
CREATE INDEX idx_nc_audit ON eudr_third_party_audit.non_conformances(audit_id);
CREATE INDEX idx_nc_severity ON eudr_third_party_audit.non_conformances(severity);
CREATE INDEX idx_nc_status ON eudr_third_party_audit.non_conformances(status);
CREATE INDEX idx_car_audit ON eudr_third_party_audit.corrective_action_requests(audit_id);
CREATE INDEX idx_car_supplier ON eudr_third_party_audit.corrective_action_requests(supplier_id);
CREATE INDEX idx_car_status ON eudr_third_party_audit.corrective_action_requests(status);
CREATE INDEX idx_car_sla_status ON eudr_third_party_audit.corrective_action_requests(sla_status);
CREATE INDEX idx_certs_supplier ON eudr_third_party_audit.certification_certificates(supplier_id);
CREATE INDEX idx_certs_scheme ON eudr_third_party_audit.certification_certificates(scheme);
CREATE INDEX idx_certs_status ON eudr_third_party_audit.certification_certificates(status);
CREATE INDEX idx_reports_audit ON eudr_third_party_audit.audit_reports(audit_id);
CREATE INDEX idx_authority_operator ON eudr_third_party_audit.authority_interactions(operator_id);
CREATE INDEX idx_authority_status ON eudr_third_party_audit.authority_interactions(status);
CREATE INDEX idx_evidence_audit ON eudr_third_party_audit.audit_evidence(audit_id);
CREATE INDEX idx_checklists_audit ON eudr_third_party_audit.audit_checklists(audit_id);
```

---

## 9. API and Integration Points

### 9.1 API Endpoints (~35)

| Method | Path | Description |
|--------|------|-------------|
| **Audit Planning** | | |
| POST | `/v1/audits/schedule/generate` | Generate risk-based audit schedule for operator |
| GET | `/v1/audits/schedule` | Get audit calendar (with date range, supplier, scheme filters) |
| PUT | `/v1/audits/schedule/{schedule_id}` | Update scheduled audit (reschedule, reassign) |
| POST | `/v1/audits/schedule/trigger` | Trigger unscheduled audit (event-based) |
| **Audit Management** | | |
| POST | `/v1/audits` | Create a new audit |
| GET | `/v1/audits` | List audits (with filters: status, supplier, scheme, date range) |
| GET | `/v1/audits/{audit_id}` | Get audit details with full status |
| PUT | `/v1/audits/{audit_id}` | Update audit (status, team, dates) |
| DELETE | `/v1/audits/{audit_id}` | Cancel audit |
| **Auditor Registry** | | |
| POST | `/v1/auditors` | Register a new auditor |
| GET | `/v1/auditors` | List auditors (with filters: scheme, commodity, country, status) |
| GET | `/v1/auditors/{auditor_id}` | Get auditor profile with performance |
| PUT | `/v1/auditors/{auditor_id}` | Update auditor profile |
| POST | `/v1/auditors/match` | Match auditors to audit requirements |
| **Audit Execution** | | |
| GET | `/v1/audits/{audit_id}/checklist` | Get audit checklist with completion status |
| PUT | `/v1/audits/{audit_id}/checklist/{criterion_id}` | Update checklist criterion result |
| POST | `/v1/audits/{audit_id}/evidence` | Upload audit evidence |
| GET | `/v1/audits/{audit_id}/evidence` | List audit evidence items |
| GET | `/v1/audits/{audit_id}/progress` | Get real-time audit progress |
| **Non-Conformances** | | |
| POST | `/v1/audits/{audit_id}/ncs` | Create non-conformance finding |
| GET | `/v1/audits/{audit_id}/ncs` | List NCs for an audit |
| GET | `/v1/ncs/{nc_id}` | Get NC details with evidence and RCA |
| PUT | `/v1/ncs/{nc_id}` | Update NC (status, RCA, dispute) |
| POST | `/v1/ncs/{nc_id}/rca` | Submit root cause analysis |
| **Corrective Action Requests** | | |
| POST | `/v1/cars` | Issue a new CAR |
| GET | `/v1/cars` | List CARs (with filters: status, severity, SLA, supplier) |
| GET | `/v1/cars/{car_id}` | Get CAR details with full lifecycle |
| PUT | `/v1/cars/{car_id}` | Update CAR status (acknowledge, submit CAP, submit evidence) |
| POST | `/v1/cars/{car_id}/verify` | Submit verification outcome |
| **Certification Schemes** | | |
| GET | `/v1/schemes/certificates` | List certification certificates (with filters) |
| POST | `/v1/schemes/certificates/sync` | Trigger certification status sync |
| GET | `/v1/schemes/coverage/{supplier_id}` | Get EUDR coverage matrix for supplier |
| **Reports** | | |
| POST | `/v1/audits/{audit_id}/report` | Generate audit report |
| GET | `/v1/reports/{report_id}` | Download audit report |
| **Competent Authority** | | |
| POST | `/v1/authority/interactions` | Log new authority interaction |
| GET | `/v1/authority/interactions` | List authority interactions |
| PUT | `/v1/authority/interactions/{id}` | Update interaction (response, status) |
| **Analytics** | | |
| GET | `/v1/analytics/findings` | Get finding trend analytics |
| GET | `/v1/analytics/auditor-performance` | Get auditor benchmarking data |
| GET | `/v1/analytics/compliance-rates` | Get compliance rate trends |
| GET | `/v1/analytics/car-performance` | Get CAR lifecycle analytics |
| GET | `/v1/analytics/dashboard` | Get executive dashboard data |
| **Health** | | |
| GET | `/health` | Service health check |

### 9.2 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| EUDR-001 Supply Chain Mapping Master | Supply chain graph data | Supplier inventory, site locations, supply chain complexity metrics -> audit scope determination |
| EUDR-016 Country Risk Evaluator | Country risk scores | Country risk level (Low/Standard/High) -> audit frequency calculation |
| EUDR-017 Supplier Risk Scorer | Supplier risk scores | Composite supplier risk -> audit priority scoring; CAR status -> supplier risk adjustment |
| EUDR-020 Deforestation Alert System | Deforestation alerts | Alert proximity to supplier plots -> unscheduled audit triggers |
| EUDR-021 Indigenous Rights Checker | FPIC compliance data | FPIC verification status -> audit checklist criteria for indigenous rights |
| EUDR-022 Protected Area Validator | Protected area data | Protected area overlap status -> audit checklist criteria for PA verification |
| EUDR-023 Legal Compliance Verifier | Legal compliance data | Legal compliance assessment -> audit checklist criteria for Article 2(40) categories |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| GL-EUDR-APP v1.0 | API integration | Audit data, NC status, CAR tracking, analytics -> frontend dashboards |
| EUDR-017 Supplier Risk Scorer | Risk factor feed | Open CARs, NC history, audit outcomes -> supplier risk score adjustment |
| EUDR-001 Supply Chain Mapping Master | Graph enrichment | Audit status, compliance rating -> supply chain node attributes |
| EUDR-023 Legal Compliance Verifier | Audit findings feed | Legal compliance NCs, document verification findings -> legal compliance record |
| External Auditors | CAR portal | CAR assignments, evidence upload, verification workflow -> auditor interface |
| Competent Authorities | Evidence packages | Audit reports, NC status, compliance documentation -> regulatory response |

### 9.3 Prometheus Self-Monitoring Metrics (20)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_tam_audits_created_total` | Counter | Audits created by type and scheme |
| 2 | `gl_eudr_tam_audits_completed_total` | Counter | Audits completed by outcome |
| 3 | `gl_eudr_tam_audits_cancelled_total` | Counter | Audits cancelled |
| 4 | `gl_eudr_tam_ncs_detected_total` | Counter | NCs detected by severity |
| 5 | `gl_eudr_tam_ncs_resolved_total` | Counter | NCs resolved by severity |
| 6 | `gl_eudr_tam_cars_issued_total` | Counter | CARs issued by severity |
| 7 | `gl_eudr_tam_cars_closed_total` | Counter | CARs closed by severity |
| 8 | `gl_eudr_tam_cars_overdue_total` | Counter | CARs that exceeded SLA |
| 9 | `gl_eudr_tam_cars_escalated_total` | Counter | CARs escalated by level |
| 10 | `gl_eudr_tam_reports_generated_total` | Counter | Audit reports generated by format |
| 11 | `gl_eudr_tam_authority_interactions_total` | Counter | Authority interactions by type |
| 12 | `gl_eudr_tam_scheme_syncs_total` | Counter | Certification scheme syncs by scheme |
| 13 | `gl_eudr_tam_scheduling_duration_seconds` | Histogram | Audit schedule generation latency |
| 14 | `gl_eudr_tam_nc_classification_duration_seconds` | Histogram | NC classification latency |
| 15 | `gl_eudr_tam_report_generation_duration_seconds` | Histogram | Report generation latency |
| 16 | `gl_eudr_tam_api_request_duration_seconds` | Histogram | API request latency by endpoint |
| 17 | `gl_eudr_tam_errors_total` | Counter | Errors by operation type |
| 18 | `gl_eudr_tam_active_audits` | Gauge | Currently active audits |
| 19 | `gl_eudr_tam_open_cars` | Gauge | Currently open CARs by severity |
| 20 | `gl_eudr_tam_car_sla_compliance_rate` | Gauge | CAR SLA compliance percentage |

---

## 10. Security and Compliance

### 10.1 Authentication and Authorization

| Security Layer | Implementation |
|---------------|----------------|
| **Authentication** | JWT (RS256) via SEC-001; all API endpoints require valid JWT |
| **Authorization** | RBAC via SEC-002 with 22 permissions (see 10.2) |
| **Encryption at Rest** | AES-256-GCM via SEC-003 for auditor PII, evidence files, and authority communications |
| **Encryption in Transit** | TLS 1.3 via SEC-004 for all API traffic |
| **Audit Logging** | All data mutations logged via SEC-005 centralized audit logging |
| **Secrets Management** | Certification scheme API keys stored in Vault via SEC-006 |

### 10.2 RBAC Permissions (22)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-tam:audits:read` | View audit records and status | Viewer, Analyst, Auditor, Compliance Officer, Admin |
| `eudr-tam:audits:write` | Create, update, cancel audits | Compliance Officer, Audit Programme Manager, Admin |
| `eudr-tam:audits:schedule` | Generate and modify audit schedules | Audit Programme Manager, Compliance Officer, Admin |
| `eudr-tam:auditors:read` | View auditor registry | Viewer, Analyst, Compliance Officer, Audit Programme Manager, Admin |
| `eudr-tam:auditors:write` | Register, update auditor profiles | Audit Programme Manager, Admin |
| `eudr-tam:execution:read` | View audit checklists and evidence | Viewer, Auditor, Compliance Officer, Admin |
| `eudr-tam:execution:write` | Update checklists, upload evidence | Auditor, Compliance Officer, Admin |
| `eudr-tam:ncs:read` | View non-conformances | Viewer, Analyst, Auditor, Compliance Officer, Admin |
| `eudr-tam:ncs:write` | Create, classify, update NCs | Auditor, Compliance Officer, Admin |
| `eudr-tam:ncs:dispute` | Dispute NC classification | Supplier (auditee), Compliance Officer, Admin |
| `eudr-tam:cars:read` | View corrective action requests | Viewer, Analyst, Auditor, Supplier, Compliance Officer, Admin |
| `eudr-tam:cars:write` | Issue, update CARs | Auditor, Compliance Officer, Admin |
| `eudr-tam:cars:respond` | Acknowledge, submit CAP, submit evidence | Supplier (auditee), Compliance Officer |
| `eudr-tam:cars:verify` | Verify corrective action effectiveness | Auditor, Compliance Officer, Admin |
| `eudr-tam:cars:close` | Close verified CARs | Compliance Officer, Admin |
| `eudr-tam:schemes:read` | View certification scheme data | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-tam:schemes:sync` | Trigger certification status sync | Compliance Officer, Admin |
| `eudr-tam:reports:read` | View and download audit reports | Viewer, Auditor, Compliance Officer, Admin |
| `eudr-tam:reports:generate` | Generate audit reports | Auditor, Compliance Officer, Admin |
| `eudr-tam:authority:read` | View authority interactions | Compliance Officer, Legal Officer, Admin |
| `eudr-tam:authority:write` | Log and manage authority interactions | Compliance Officer, Legal Officer, Admin |
| `eudr-tam:analytics:read` | View audit analytics and dashboards | Analyst, Compliance Officer, Audit Programme Manager, Admin |

### 10.3 Data Privacy and GDPR

| Data Type | Classification | Protection |
|-----------|---------------|------------|
| Auditor personal data (name, email, qualifications) | PII | AES-256 encryption; GDPR consent required; right to erasure supported |
| Supplier contact information | PII | AES-256 encryption; data minimization; retention per Article 31 (5 years) |
| Audit evidence (photos, documents) | Confidential | AES-256 encryption at rest; access-controlled per RBAC |
| Authority communications | Highly Confidential | AES-256 encryption; restricted to Legal Officer + Admin roles |
| Audit findings and NCs | Confidential | Access-controlled per RBAC; retained 5 years |
| Analytics and aggregated data | Internal | Role-based access; anonymized where possible |

---

## 11. Performance and Scalability

### 11.1 Performance Targets

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Audit schedule generation (500 suppliers) | < 2 seconds | p99 latency |
| NC classification | < 500ms per finding | p99 latency |
| CAR SLA countdown update | < 1 minute latency | Update propagation time |
| Audit report generation (PDF, 50 findings) | < 30 seconds | p99 latency |
| Certification status sync (per scheme) | < 5 minutes | Full sync duration |
| Auditor matching query (1,000 auditors) | < 500ms | p99 latency |
| Analytics dashboard load (12-month window) | < 3 seconds | Frontend render time |
| API response (standard queries) | < 200ms | p95 latency |
| Evidence file upload (100 MB) | < 60 seconds | Upload completion time |
| Authority evidence package assembly | < 2 minutes | Package generation time |

### 11.2 Scalability Requirements

| Dimension | Current Target | Future Scale |
|-----------|---------------|--------------|
| Concurrent audits | 500 active audits | 5,000 active audits |
| Total audit records | 10,000 per year | 100,000 per year |
| Active CARs | 2,000 simultaneously | 20,000 simultaneously |
| Auditor registry | 1,000 auditors | 10,000 auditors |
| Certification certificates | 5,000 | 50,000 |
| Authority interactions | 500 per year | 5,000 per year |
| Evidence storage | 500 GB | 5 TB |
| Concurrent API users | 200 | 2,000 |

### 11.3 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for audit trail, NC trends, CAR SLA logs |
| Cache | Redis | Audit status caching, SLA countdown caching, analytics query caching |
| Object Storage | S3 | Evidence files, audit reports, evidence packages |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| PDF Generation | WeasyPrint + Jinja2 | ISO 19011 report templates |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access control with 22 permissions |
| Encryption | AES-256-GCM via SEC-003 | Auditor PII, evidence, authority communications |
| Monitoring | Prometheus + Grafana | 20 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

---

## 12. Testing and Quality Assurance

### 12.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Audit Planning Tests | 80+ | Risk-based scheduling, frequency calculation, event triggers, multi-scheme coordination, calendar generation |
| Auditor Registry Tests | 50+ | Profile CRUD, competence matching, conflict-of-interest, accreditation validation, rotation rules |
| Audit Execution Tests | 70+ | Checklist management, evidence upload, progress tracking, sampling plans, modality management |
| NC Classification Tests | 80+ | Severity classification (all rules), EUDR article mapping, root cause analysis, pattern detection, dispute process |
| CAR Lifecycle Tests | 80+ | Full lifecycle (all status transitions), SLA calculation, escalation rules, closure verification, grouping |
| Certification Scheme Tests | 60+ | 5 scheme integrations, coverage matrix, NC taxonomy mapping, status sync, mutual recognition |
| Report Generation Tests | 50+ | ISO 19011 structure, 5 formats, 5 languages, evidence packages, amendment workflow, hash verification |
| Authority Liaison Tests | 40+ | 27 Member State profiles, interaction types, SLA tracking, response coordination, evidence assembly |
| Analytics Tests | 50+ | Finding trends, auditor benchmarking, compliance rates, CAR performance, cost analysis, dashboard |
| API Tests | 50+ | All ~35 endpoints, auth, error handling, pagination, rate limiting |
| Golden Tests | 50 | 7 commodities x 7 audit scenarios (see 12.2) + 1 multi-scheme scenario |
| Integration Tests | 40+ | Cross-agent data flow with EUDR-001/016/017/020/021/022/023, event bus, API contracts |
| Performance Tests | 25+ | Schedule generation, NC classification, report generation, analytics load, concurrent queries |
| Determinism Tests | 15+ | Bit-perfect reproducibility for scheduling, NC classification, SLA calculation, analytics |
| Security Tests | 10+ | Auditor PII encryption, RBAC enforcement, evidence access control, authority data protection |
| **Total** | **800+** | |

### 12.2 Golden Test Scenarios

Each of the 7 EUDR commodities will have 7 golden test audit scenarios:

1. **Clean audit** -- Full scope audit with zero NCs; audit lifecycle from planned to closed with clean report -> expect CLOSED status, report generated, no CARs
2. **Critical NC audit** -- Audit discovers evidence of post-2020 deforestation; critical NC raised -> expect CRITICAL classification, 30-day CAR issued, immediate escalation, certification suspension recommendation
3. **Major NC with CAR closure** -- Audit finds incomplete chain of custody documentation; major NC with 90-day CAR -> expect MAJOR classification, CAR lifecycle through to verified closure, supplier risk adjustment
4. **Multiple minor NCs** -- Audit finds 5 minor administrative gaps; all grouped under single CAR -> expect 5 MINOR NCs, 1 grouped CAR, 365-day SLA, consolidated corrective action plan
5. **Multi-scheme audit** -- Supplier with FSC + Rainforest Alliance; combined audit scope -> expect overlap detection, reduced audit criteria, findings mapped to both scheme taxonomies
6. **Competent authority inspection** -- Authority issues document request for timber supply chain -> expect interaction logged, 30-day SLA countdown, evidence package generated from audit records, response tracked
7. **Overdue CAR escalation** -- Major NC CAR exceeds 90-day SLA -> expect escalation through 4 stages, supplier risk score increased, competent authority notification triggered

Total: 7 commodities x 7 scenarios = 49 golden test scenarios (+ 1 multi-scheme cross-commodity scenario = 50)

### 12.3 Determinism Tests

Every scoring and calculation engine will include determinism tests that:
1. Run the same calculation 100 times with identical inputs
2. Verify bit-perfect identical outputs (SHA-256 hash match)
3. Test across Python versions (3.11, 3.12) to ensure no platform-dependent behavior
4. Verify Decimal arithmetic produces identical results to reference calculations
5. Verify risk-based audit scheduling is deterministic with identical risk inputs
6. Verify NC severity classification is deterministic with identical finding inputs
7. Verify CAR SLA deadlines are deterministic with identical NC severity and issuance timestamps

---

## 13. Documentation Requirements

### 13.1 Technical Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| API Reference (OpenAPI 3.1) | Complete endpoint documentation with request/response schemas, examples, and error codes | Developers, Integration Partners |
| Data Model Reference | Entity relationships, field definitions, validation rules, and migration scripts | Backend Engineers, DBAs |
| Integration Guide | Cross-agent integration patterns, event contracts, and API contracts for EUDR-001/016/017/020/021/022/023 | Platform Engineers |
| Configuration Reference | All `GL_EUDR_TAM_` environment variables, risk weight configuration, SLA thresholds | DevOps, Platform Engineers |
| Deployment Guide | Kubernetes manifests, Helm chart values, migration steps, health check configuration | DevOps |

### 13.2 User Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| Audit Planning Guide | How to generate risk-based audit schedules, interpret priority scores, manage audit calendar | Compliance Officers, Audit Programme Managers |
| Auditor Management Guide | How to register auditors, track qualifications, manage conflicts of interest, use matching algorithm | Audit Programme Managers |
| NC and CAR Management Guide | How to classify NCs, issue CARs, track lifecycle, manage escalations, verify closures | Auditors, Compliance Officers |
| Certification Scheme Guide | How to manage multi-scheme certifications, interpret coverage matrices, coordinate audits | Compliance Officers |
| Competent Authority Response Guide | How to log authority interactions, prepare responses, assemble evidence packages, meet deadlines | Compliance Officers, Legal Officers |
| Analytics Dashboard Guide | How to interpret audit analytics, configure KPIs, generate reports, use drill-down features | Heads of Compliance, Management |

---

## 14. Implementation Roadmap

### Phase 1: Core Audit Lifecycle Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Audit Planning and Scheduling Engine (Feature 1): risk-based frequency calculation, audit calendar, event triggers, multi-site planning | Senior Backend Engineer |
| 2-3 | Auditor Registry and Qualification Tracker (Feature 2): profile management, competence matching, CoI tracking, accreditation validation | Backend Engineer |
| 3-4 | Audit Execution Tracker (Feature 3): checklist management, evidence collection, sampling plans, progress dashboard | Senior Backend Engineer + Frontend Engineer |
| 4-5 | Non-Conformance Detection Engine (Feature 4): severity classification rules, EUDR article mapping, root cause analysis, pattern detection | Senior Backend Engineer |
| 5-6 | Corrective Action Request Manager (Feature 5): CAR lifecycle, SLA engine, escalation rules, closure verification | Senior Backend Engineer + Frontend Engineer |

**Milestone: Core audit lifecycle engine operational with 5 core features (Week 6)**

### Phase 2: Integration, Reporting, and Authority Management (Weeks 7-11)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Certification Scheme Integration (Feature 6): 5-scheme connectors (FSC, PEFC, RSPO, RA, ISCC), coverage matrix, NC taxonomy mapping | Backend Engineer + Integration Engineer |
| 8-9 | Audit Report Generator (Feature 7): ISO 19011 templates, 5 formats, 5 languages, evidence packages, amendment workflow | Backend Engineer + Template Engineer |
| 9-10 | Competent Authority Liaison (Feature 8): 27 Member State profiles, interaction workflow, SLA tracking, evidence package assembly | Backend Engineer + Frontend Engineer |
| 10-11 | REST API Layer: ~35 endpoints, authentication, rate limiting, OpenAPI documentation | Backend Engineer |

**Milestone: Full API operational with scheme integration, reporting, and authority management (Week 11)**

### Phase 3: Analytics, RBAC, and Observability (Weeks 12-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 12 | Audit Analytics and Trends (Feature 9): finding trends, auditor benchmarking, compliance rates, CAR performance, cost analysis | Backend Engineer + Data Engineer |
| 12-13 | RBAC integration (22 permissions), AES-256 encryption for auditor PII and authority data, Prometheus metrics (20) | Backend + DevOps |
| 13-14 | Grafana dashboard, OpenTelemetry tracing, event bus integration, end-to-end integration testing with EUDR agents | DevOps + Backend |

**Milestone: All 9 P0 features implemented with full integration, analytics, and observability (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 800+ tests, golden tests for all 7 commodities, NC classification validation | Test Engineer |
| 16-17 | Performance testing, security audit (PII encryption, RBAC enforcement), load testing (concurrent audits) | DevOps + Security |
| 17 | Database migration V112 finalized and tested (13 tables, 4 hypertables) | DevOps |
| 17-18 | Beta customer onboarding (5 customers with active audit programmes) | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 9 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Predictive audit risk model (Feature 10)
- Remote audit video integration (Feature 11)
- Supplier self-assessment portal (Feature 12)
- Additional certification scheme integrations (SAN, organic, fair trade)
- CSDDD Article 11 audit management extension

---

## 15. User Experience

### 15.1 User Personas

#### Persona 1: Audit Programme Manager -- Heinrich (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Audit Programme at a European chocolate and confectionery manufacturer |
| **Company** | 8,000 employees; sourcing cocoa from Ghana, CDI, Ecuador; coffee from Colombia, Vietnam; palm oil from Indonesia, Malaysia |
| **EUDR Pressure** | Managing 80+ third-party audits per year across FSC, Rainforest Alliance, and RSPO; competent authority inspections increasing |
| **Pain Points** | No centralized view of audit lifecycle across schemes; CARs tracked in spreadsheets with missed SLAs; cannot demonstrate audit trend improvement to board; redundant audits for multi-certified suppliers |
| **Goals** | Single platform for all audit management; risk-based scheduling; automated SLA tracking; cross-scheme coordination; executive analytics |
| **Technical Skill** | Moderate -- comfortable with web applications and audit management tools |

#### Persona 2: Lead Auditor -- Isabelle (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | FSC Lead Auditor at an accredited certification body |
| **EUDR Pressure** | Increasing demand for EUDR-aligned audits; need to demonstrate competence for EUDR verification |
| **Pain Points** | Paper-based checklists; evidence scattered across photos, notes, and emails; report writing takes 2-3 days per audit; no standardized NC classification framework |
| **Goals** | Digital checklists with auto-mapping to EUDR articles; structured evidence collection; automated report generation; consistent NC classification |
| **Technical Skill** | Moderate -- comfortable with audit software; uses mobile for field evidence capture |

#### Persona 3: Compliance Officer -- Jan (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | EUDR Compliance Manager at a European timber importer |
| **Company** | 1,200 employees; importing tropical timber from Indonesia, Brazil, DRC, Cameroon |
| **EUDR Pressure** | Competent authority (NVWA) conducting increased checks on timber importers; must respond to document requests within 30 days |
| **Pain Points** | No system for tracking authority interactions; evidence assembly takes weeks; cannot demonstrate EUDR compliance improvements to authority; overdue CARs from last FSC audit creating risk |
| **Goals** | Structured authority interaction management; rapid evidence assembly; CAR closure tracking with SLA enforcement; compliance trend demonstration |
| **Technical Skill** | Moderate -- comfortable with compliance management systems |

#### Persona 4: Supplier Quality Manager -- Arun (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Quality Manager at a palm oil mill in Indonesia (RSPO certified) |
| **EUDR Pressure** | EU customer requires EUDR-aligned audit evidence; multiple audits per year from different customers |
| **Pain Points** | Receives CARs via email with unclear deadlines; evidence submission format varies by customer; no visibility into own CAR closure status; redundant audit requests |
| **Goals** | Clear CAR requirements and deadlines; structured evidence submission portal; visibility into own compliance status; reduced audit burden through coordination |
| **Technical Skill** | Low-moderate -- comfortable with basic web applications |

### 15.2 User Flows

#### Flow 1: Annual Audit Planning (Audit Programme Manager)

```
1. Heinrich logs in to GL-EUDR-APP, navigates to "Audit Management"
2. Clicks "Generate Annual Audit Schedule" for 2026
3. System retrieves supplier inventory from EUDR-001 (250 suppliers)
4. System retrieves risk scores from EUDR-016 (country) and EUDR-017 (supplier)
5. System calculates priority scores and assigns frequency:
   - 45 suppliers at HIGH (quarterly audit) -- mostly in Indonesia, DRC
   - 120 suppliers at STANDARD (semi-annual) -- Ghana, CDI, Colombia
   - 85 suppliers at LOW (annual) -- certified, low-risk countries
6. System generates audit calendar: 310 audits planned for 2026
7. System identifies 15 cross-scheme overlaps (FSC + RA)
   -> Recommends combined audits reducing count to 295
8. Heinrich reviews and approves schedule
9. System assigns auditors via matching algorithm
10. Audit invitations sent to auditors and suppliers
```

#### Flow 2: Audit Execution and NC Management (Lead Auditor)

```
1. Isabelle receives audit assignment for cocoa farm cooperative in Ghana
2. Opens digital checklist in GL-EUDR-APP: 85 EUDR criteria + 42 RA criteria
3. Conducts 3-day on-site audit:
   - Day 1: Document review (permits, certifications, CoC records)
   - Day 2: Field visits (farm plots, GPS verification, worker interviews)
   - Day 3: Stakeholder interviews, closing meeting
4. During audit, marks criteria as PASS/FAIL/NA with evidence uploads:
   - Uploads 45 evidence items (photos, scanned permits, GPS records)
   - Each evidence item tagged with type, location, date
5. Identifies 3 findings:
   - MAJOR: Missing environmental impact assessment for 2 farm groups (Art. 2(40) Cat 2)
   - MINOR: Training records not current for 5 workers (Art. 2(40) Cat 5)
   - OBSERVATION: Recommended improvement to record-keeping system
6. System auto-classifies: Major NC -> 90-day CAR; Minor NC -> 365-day CAR
7. Isabelle completes checklist (100%) and initiates report generation
8. System generates ISO 19011 compliant PDF report in < 30 seconds
9. Report delivered to Heinrich (operator) and cooperative (auditee)
```

#### Flow 3: Competent Authority Response (Compliance Officer)

```
1. Jan receives document request from NVWA (Netherlands) for timber supply chain
2. Logs interaction in GL-EUDR-APP: type = Document Request; deadline = 30 days
3. System starts 30-day SLA countdown
4. Jan assigns response tasks to internal teams:
   - Supply chain team: Provide supply chain graph for Indonesian timber
   - Legal team: Compile SVLK certificates and export permits
   - Audit team: Compile latest FSC audit reports and CAR status
5. Teams upload evidence to centralized response package
6. At Day 20: System sends amber warning (75% SLA elapsed)
7. Jan reviews assembled evidence package:
   - Supply chain graph (from EUDR-001)
   - 12 audit reports (from Feature 7)
   - 8 SVLK certificates (from EUDR-023)
   - CAR status summary (all CARs closed within SLA)
8. Jan submits response to NVWA before deadline
9. System records submission and tracks authority decision
```

### 15.3 Key Screen Descriptions

**Audit Calendar Dashboard:**
- Calendar view: monthly/quarterly/annual with audits color-coded by status (planned=blue, in-progress=yellow, complete=green, overdue=red)
- Left sidebar: filter panel (supplier, scheme, country, commodity, audit type, status)
- Top bar: summary statistics (total planned, in-progress, completed, overdue)
- Right sidebar: selected audit detail panel with quick-action buttons (assign auditor, reschedule, view report)

**CAR Tracking Dashboard:**
- Kanban board view: columns = CAR status stages (Issued -> Acknowledged -> CAP Submitted -> In Progress -> Verification -> Closed)
- SLA indicators: green/amber/red countdown badges on each CAR card
- Filter bar: severity, supplier, scheme, SLA status, date range
- Alert panel: overdue CARs requiring escalation with one-click escalation action
- Trend chart: CAR closure rate over time with SLA compliance percentage

**Analytics Dashboard:**
- Four quadrant layout: (1) Finding Trends, (2) Compliance Rates, (3) Auditor Performance, (4) CAR Performance
- Configurable time period selector (month/quarter/year)
- Drill-down capability: click any chart element to see underlying data
- Export buttons: PDF summary, XLSX detail
- KPI cards at top: total audits, critical NCs, open CARs, SLA compliance rate, authority response rate

---

## 16. Dependencies

### 16.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable; supply chain graph API available |
| AGENT-EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Country risk scores API available |
| AGENT-EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Supplier risk scores API available; bidirectional integration defined |
| AGENT-EUDR-020 Deforestation Alert System | BUILT (100%) | Low | Alert proximity API available |
| AGENT-EUDR-021 Indigenous Rights Checker | BUILT (100%) | Low | FPIC audit criteria data available |
| AGENT-EUDR-022 Protected Area Validator | BUILT (100%) | Low | Protected area audit criteria data available |
| AGENT-EUDR-023 Legal Compliance Verifier | BUILT (100%) | Low | Legal compliance audit criteria data available |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Frontend integration target available |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC |
| SEC-003 Encryption at Rest | BUILT (100%) | Low | AES-256 for auditor PII and evidence |
| S3 Object Storage | Production Ready | Low | Evidence file and report storage |
| Redis Cache | Production Ready | Low | SLA countdown and analytics caching |

### 16.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| FSC Certificate Database API | Available (API) | Medium | API rate limits; local cache with daily sync; manual import fallback |
| RSPO Certificate Database | Available (web/API) | Medium | Data format may change; adapter pattern with fallback |
| PEFC Certificate Search | Available (web) | Medium | No formal API; web extraction with adapter pattern |
| Rainforest Alliance Database | Available (partnership) | Medium | Establish data partnership; manual import as fallback |
| ISCC Certificate Database | Available (API) | Low | Stable API; local cache |
| EU Member State competent authority designations | Published | Low | Static data; updated when Member States notify changes |
| ISO 19011:2018 standard | Published | Low | Stable normative document; report templates based on published clauses |
| ISO/IEC 17065:2012 standard | Published | Low | Stable; accreditation requirement definitions |

---

## 17. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Certification scheme APIs change format or become unavailable, breaking integration | Medium | Medium | Adapter pattern isolates scheme integration; local certificate cache; manual import fallback; multi-scheme redundancy |
| R2 | NC classification produces inconsistent results compared to auditor expert judgment | Medium | High | Conservative initial classification rules; calibration against 500-case test corpus validated by expert panel; operator feedback loop for rule refinement; 95% accuracy target |
| R3 | CAR SLA enforcement perceived as too rigid by suppliers, causing resistance | Medium | Medium | Configurable SLA deadlines per operator; grace period option; escalation stages provide warning before penalty; supplier portal with clear deadline visibility |
| R4 | Competent authority response workflows vary significantly across 27 Member States | High | Medium | Start with top 6 Member States (DE, FR, NL, BE, IT, ES); generic workflow template for others; configurable per-authority response templates; iterative expansion |
| R5 | Cross-scheme audit coordination faces resistance from certification bodies protecting their audit revenue | Medium | Medium | Focus on operator-side coordination; demonstrate audit quality improvement; support scheme-accepted combined audit formats; maintain scheme-specific report generation |
| R6 | Audit evidence volumes overwhelm storage and processing capacity | Low | Medium | S3-based storage with lifecycle policies; evidence compression; thumbnail generation for photos; lazy loading for large files; 5 TB capacity planning |
| R7 | Integration complexity with 7 upstream EUDR agents creates fragility | Medium | Medium | Well-defined interfaces; mock adapters for testing; circuit breaker pattern; integration health monitoring (20 metrics); retry logic with exponential backoff |
| R8 | Auditor PII handling creates GDPR compliance risk | Medium | High | AES-256 encryption; GDPR consent tracking; right to erasure implementation; privacy impact assessment; DPO review before launch |
| R9 | ISO 19011 report format requirements change with standard revision | Low | Low | Template-based report generation; report structure configurable; version-controlled templates; quick update turnaround |
| R10 | Low adoption by auditors accustomed to paper-based or proprietary audit tools | Medium | Medium | Intuitive UX for auditors; offline checklist capability; mobile-responsive evidence capture; training materials; pilot programme with 3 certification bodies |

---

## 18. Success Criteria

### 18.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Audit Planning -- risk-based scheduling, event triggers, multi-scheme coordination
  - [ ] Feature 2: Auditor Registry -- qualification tracking, competence matching, CoI management
  - [ ] Feature 3: Audit Execution -- checklists, evidence collection, sampling plans, progress tracking
  - [ ] Feature 4: NC Detection -- severity classification, EUDR mapping, root cause analysis, pattern detection
  - [ ] Feature 5: CAR Manager -- lifecycle stages, SLA enforcement, escalation rules, closure verification
  - [ ] Feature 6: Scheme Integration -- 5 schemes integrated, coverage matrix, NC taxonomy mapping
  - [ ] Feature 7: Report Generator -- ISO 19011 compliant, 5 formats, 5 languages, evidence packages
  - [ ] Feature 8: Authority Liaison -- 27 Member State profiles, interaction workflow, SLA tracking
  - [ ] Feature 9: Analytics -- finding trends, auditor benchmarking, compliance rates, dashboards
- [ ] >= 85% test coverage achieved (800+ tests)
- [ ] Security audit passed (JWT + RBAC integrated, 22 permissions, PII encryption)
- [ ] Performance targets met (< 2s scheduling, < 500ms NC classification, < 30s report generation)
- [ ] NC classification validated against expert panel (>= 95% agreement on 500-case corpus)
- [ ] All calculations verified deterministic (bit-perfect reproducibility)
- [ ] API documentation complete (OpenAPI spec, ~35 endpoints)
- [ ] Database migration V112 tested and validated (13 tables, 4 hypertables)
- [ ] Integration with all 7 dependent EUDR agents verified
- [ ] 5 beta customers with active audit programmes successfully using the platform
- [ ] No critical or high-severity bugs in backlog

### 18.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ audits managed through the platform
- 100+ NCs classified with severity and EUDR article mapping
- 50+ CARs issued with active SLA tracking
- Certification status synced for 3+ schemes
- < 5 support tickets per customer
- p99 API latency < 200ms

**60 Days:**
- 200+ audits managed
- 500+ NCs classified
- 200+ CARs tracked; >= 80% SLA compliance rate
- 5 certification schemes fully integrated
- 10+ competent authority interactions managed
- NPS > 45 from audit programme manager persona

**90 Days:**
- 500+ audits managed across multiple operators
- 1,000+ NCs classified with trend analysis active
- >= 85% CAR SLA compliance rate
- >= 20% reduction in redundant audit activities for multi-certified suppliers
- Full analytics dashboard operational with executive reporting
- Zero missed competent authority response deadlines
- NPS > 55

---

## 19. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **NC** | Non-Conformance -- audit finding of non-compliance with a requirement |
| **CAR** | Corrective Action Request -- formal request to address an audit finding |
| **CAP** | Corrective Action Plan -- documented plan to address audit findings |
| **RCA** | Root Cause Analysis -- systematic process to identify underlying cause of NC |
| **SLA** | Service Level Agreement -- deadline for completing required action |
| **ISO 19011** | International standard providing guidelines for auditing management systems |
| **ISO/IEC 17065** | International standard for conformity assessment bodies certifying products, processes, and services |
| **ISO/IEC 17021-1** | International standard for conformity assessment bodies providing audit and certification of management systems |
| **IAF** | International Accreditation Forum -- global association of accreditation bodies |
| **MLA** | Multilateral Recognition Arrangement -- IAF agreement on mutual acceptance of accreditation |
| **CB** | Certification Body -- organization that conducts audits and issues certifications |
| **FSC** | Forest Stewardship Council -- certification scheme for responsible forest management |
| **PEFC** | Programme for the Endorsement of Forest Certification |
| **RSPO** | Roundtable on Sustainable Palm Oil -- certification scheme for sustainable palm oil |
| **ISCC** | International Sustainability and Carbon Certification |
| **RA** | Rainforest Alliance -- certification for sustainable agriculture |
| **FPIC** | Free, Prior and Informed Consent |
| **HCV** | High Conservation Value -- areas with significant biological, ecological, social, or cultural values |
| **HCS** | High Carbon Stock -- methodology to distinguish forest areas from degraded lands |
| **CoC** | Chain of Custody -- tracking certified material through the supply chain |
| **SVLK** | Sistem Verifikasi Legalitas Kayu -- Indonesian Timber Legality Verification System |
| **FLEGT** | Forest Law Enforcement, Governance and Trade |
| **CPD** | Continuing Professional Development -- ongoing training for auditors |
| **CoI** | Conflict of Interest -- relationship that may compromise auditor impartiality |
| **OFI** | Opportunity for Improvement -- audit observation that is not an NC |
| **CSDDD** | Corporate Sustainability Due Diligence Directive (EU, 2024) |
| **NVWA** | Nederlandse Voedsel- en Warenautoriteit -- Netherlands Food and Consumer Product Safety Authority |
| **BMEL** | Bundesministerium fur Ernahrung und Landwirtschaft -- German Federal Ministry of Food and Agriculture |
| **DGCCRF** | Direction Generale de la Concurrence, de la Consommation et de la Repression des Fraudes -- French Directorate-General for Competition Policy, Consumer Affairs and Fraud Control |

### Appendix B: ISO 19011:2018 Audit Report Structure (Clause 6.6)

| Report Section | Content | Agent Feature |
|---------------|---------|---------------|
| **Audit objectives** | Statement of what the audit aimed to achieve | Feature 7: auto-populated from audit scope |
| **Audit scope** | Boundaries and extent of the audit (sites, processes, time period) | Feature 7: from Feature 3 checklist scope |
| **Audit criteria** | Set of requirements against which audit was conducted | Feature 7: EUDR articles + scheme clauses |
| **Audit client** | Organization that requested the audit | Feature 7: operator profile data |
| **Audit team members** | Lead auditor and team with qualifications | Feature 7: from Feature 2 auditor registry |
| **Dates and locations** | When and where audit activities took place | Feature 7: from Feature 3 fieldwork schedule |
| **Audit findings** | Results of evaluating audit evidence against criteria | Feature 7: from Feature 4 NC engine |
| **Audit conclusions** | Overall assessment including degree of conformity | Feature 7: auto-generated from findings |
| **Statement of confidentiality** | Confidentiality obligations | Feature 7: template section |
| **Distribution list** | Intended recipients of the report | Feature 7: from audit management data |

### Appendix C: Integration API Contracts

**Provided to EUDR-017 (Supplier Risk Scorer):**
```python
# Audit compliance data for supplier risk scoring
def get_supplier_audit_risk(supplier_id: str) -> Dict:
    """Returns: {
        supplier_id: str,
        total_audits: int,
        last_audit_date: str,                # ISO 8601
        days_since_last_audit: int,
        open_critical_ncs: int,
        open_major_ncs: int,
        open_minor_ncs: int,
        open_cars: int,
        overdue_cars: int,
        car_sla_compliance_rate: Decimal,    # 0-100
        repeat_nc_rate: Decimal,             # 0-100
        certification_status: Dict[str, str], # {scheme: status}
        audit_compliance_risk_score: Decimal, # 0-100
        provenance_hash: str
    }"""
```

**Provided to EUDR-001 (Supply Chain Mapping Master):**
```python
# Audit status for supply chain graph node enrichment
def get_supplier_audit_status(supplier_id: str) -> Dict:
    """Returns: {
        supplier_id: str,
        audit_status: str,                   # compliant/at_risk/non_compliant/unaudited
        last_audit_date: str,
        next_audit_date: str,
        open_critical_ncs: int,
        open_major_ncs: int,
        certification_schemes: List[str],
        audit_compliance_rating: Decimal,    # 0-100
        provenance_hash: str
    }"""
```

**Consumed from EUDR-016 (Country Risk Evaluator):**
```python
# Country risk data for audit scheduling
def get_country_risk_for_audit(country_code: str) -> Dict:
    """Returns: {
        country_code: str,
        risk_level: str,                     # low/standard/high
        risk_score: Decimal,                 # 0-100
        governance_score: Decimal,           # 0-100
        enforcement_score: Decimal,          # 0-100
        audit_frequency_multiplier: Decimal, # 1.0/1.5/2.0
        provenance_hash: str
    }"""
```

**Consumed from EUDR-020 (Deforestation Alert System):**
```python
# Deforestation alert data for unscheduled audit triggers
def get_deforestation_alerts_for_supplier(supplier_id: str) -> Dict:
    """Returns: {
        supplier_id: str,
        active_alerts: List[{
            alert_id: str,
            severity: str,                   # low/medium/high/critical
            proximity_km: Decimal,
            detected_date: str,
            coordinates: Tuple[float, float],
            confidence: Decimal
        }],
        highest_severity: str,
        alert_risk_score: Decimal,           # 0-100
        audit_trigger_recommended: bool,
        provenance_hash: str
    }"""
```

### Appendix D: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EUDR)
2. European Commission EUDR Guidance Document -- Risk Assessment and Due Diligence
3. ISO 19011:2018 -- Guidelines for auditing management systems
4. ISO/IEC 17065:2012 -- Conformity assessment -- Requirements for bodies certifying products, processes and services
5. ISO/IEC 17021-1:2015 -- Conformity assessment -- Requirements for bodies providing audit and certification of management systems
6. ISO/IEC 17011:2017 -- Conformity assessment -- Requirements for accreditation bodies
7. FSC-STD-20-007 -- Forest Management Evaluations
8. FSC-STD-20-011 -- Chain of Custody Evaluations
9. FSC-PRO-20-003 -- Processing Certification Reports
10. RSPO Principles and Criteria (2018)
11. RSPO Supply Chain Certification Standard (SCCS)
12. PEFC ST 2003:2020 -- Chain of Custody of Forest and Tree Based Products
13. PEFC ST 1003:2018 -- Sustainable Forest Management
14. Rainforest Alliance Sustainable Agriculture Standard (2020)
15. Rainforest Alliance Chain of Custody Standard
16. ISCC EU/PLUS System Requirements
17. ISCC 202 Sustainability Requirements
18. Directive (EU) 2024/1760 -- Corporate Sustainability Due Diligence Directive (CSDDD)
19. EU FLEGT Regulation 2173/2005
20. IAF Multilateral Recognition Arrangement (MLA) -- Scope and Rules
21. ILO Core Conventions (C029, C087, C098, C100, C105, C111, C138, C182)
22. UN Guiding Principles on Business and Human Rights (2011)
23. OECD Due Diligence Guidance for Responsible Supply Chains (2018)

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-10 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| Audit Standards Specialist | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-10 | GL-ProductManager | Initial draft created: all 9 P0 features specified, regulatory coverage verified (EUDR Articles 4/9/10/11/14-16/18-23/29/31 + ISO 19011/17065/17021-1), 5 certification scheme integration mapped, 13-table database schema defined, ~35 API endpoints, 22 RBAC permissions, 20 Prometheus metrics, 800+ test targets |
