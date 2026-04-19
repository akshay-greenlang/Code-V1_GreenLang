# PRD: AGENT-EUDR-032 -- Grievance Mechanism Manager

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-032 |
| **Agent ID** | GL-EUDR-GMM-032 |
| **Component** | Grievance Mechanism Manager Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence (Category 5) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-12 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 10, 11, 14-16, 29, 31; EU Corporate Sustainability Due Diligence Directive (CSDDD) Articles 7, 8, 9; UN Guiding Principles on Business and Human Rights (UNGP) Principle 31; ILO Convention 169; OECD Guidelines for Multinational Enterprises Chapter IV |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **Prerequisite Agent** | AGENT-EUDR-031 Stakeholder Engagement Tool (provides base grievance intake, triage, investigation, resolution) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) and the EU Corporate Sustainability Due Diligence Directive (CSDDD) together mandate that operators establish, maintain, and continuously improve grievance mechanisms that are accessible, transparent, predictable, equitable, and rights-compatible. Agent EUDR-031 (Stakeholder Engagement Tool) provides the foundational grievance mechanism: multi-channel intake, triage classification, investigation workflow, resolution management, and appeal processing. This foundational layer handles the operational lifecycle of individual grievances -- receiving, investigating, and resolving complaints one at a time.

However, operating a grievance mechanism at scale across multi-country, multi-commodity EUDR supply chains exposes critical gaps that a single-case operational tool cannot address. When an operator sources cocoa from 200+ cooperatives across 12 countries, rubber from 150+ smallholders in Southeast Asia, and timber from 80+ concessions in Central Africa, the grievance mechanism generates thousands of cases over time. Extracting intelligence from this volume, managing complex multi-party disputes, verifying that remediation actions actually work, predicting which grievances will escalate, handling collective complaints from entire communities, and generating the specific regulatory reports demanded by competent authorities under EUDR Articles 14-16 and CSDDD Article 8 -- these are advanced management capabilities that sit above the operational layer.

Today, EU operators with active EUDR-031 grievance mechanisms face the following advanced management gaps:

- **No grievance analytics or pattern detection**: Individual grievances are processed and resolved, but no system analyzes grievance data across time, geography, commodity, and supply chain segment to identify patterns, trends, clusters, and hotspots. When 40 separate land rights complaints arise from different communities in the same Indonesian province over 6 months, EUDR-031 processes each individually but cannot identify the systemic pattern. EUDR Article 10(2)(e) requires operators to consider "information from consultations with indigenous peoples, local communities, and other stakeholders" as risk factors -- grievance patterns are a primary source of such information, but without analytics, this intelligence is lost.

- **No systematic root cause analysis**: EUDR-031's investigation workflow documents findings per case, but there is no systematic methodology for identifying root causes that span multiple grievances. When 15 environmental complaints across 3 countries all trace back to a single supplier's inadequate waste management practices, operators need structured 5-Whys analysis, fishbone diagramming, and cross-case correlation to surface that root cause. Without systematic root cause analysis, operators remediate symptoms case by case while the underlying problem persists, generating more grievances and increasing regulatory risk.

- **No multi-party mediation management**: EUDR-031 handles bilateral grievances (complainant vs. operator). Real-world EUDR supply chain disputes frequently involve multiple parties: an indigenous community, a cooperative, a local government authority, an NGO monitor, a certification body, and the EU operator, all with different interests, legal standings, and desired outcomes. Managing multi-party mediation requires neutral mediator assignment, structured mediation sessions, position documentation from each party, settlement negotiation tracking, agreement drafting, and compliance monitoring. EUDR-031 explicitly lists "Legal mediation or arbitration services for disputes" as out of scope (Section 3.4, item 10).

- **No remediation effectiveness tracking**: EUDR-031 tracks remediation actions to completion (status: pending/in-progress/completed). But completion is not effectiveness. When an operator commits to replanting 500 hectares of degraded forest as remediation for a deforestation grievance, EUDR-031 marks the action complete when replanting begins. There is no system to verify that the replanting succeeded after 12, 24, and 36 months; that the planted species are appropriate; that the community considers the remediation adequate; or that the underlying deforestation driver was actually addressed. CSDDD Article 9 requires companies to "remediate adverse impacts" -- this implies verified effectiveness, not mere activity completion.

- **No grievance risk scoring or predictive analytics**: Some grievances are routine inquiries that resolve quickly; others are early warning signals of systemic problems that, if unaddressed, will escalate to regulatory investigations, certification suspensions, or litigation. EUDR-031 classifies severity at triage time (critical/high/medium/low) based on the complaint content. There is no predictive system that uses historical grievance data, supply chain risk signals, country context, and stakeholder relationship status to score the probability that a grievance will escalate, recur, or trigger regulatory action. Without predictive risk scoring, operators allocate resources equally across grievances rather than prioritizing the ones most likely to cause material harm.

- **No collective grievance handling**: Individual stakeholders submit individual complaints to EUDR-031. But EUDR supply chain impacts frequently affect entire communities, worker groups, or stakeholder coalitions. When deforestation destroys the watershed serving 12 villages, a single grievance case in EUDR-031 cannot represent the collective harm. Collective grievances require class-action style management: representative complainant identification, affected population mapping, collective harm assessment, community-wide remediation, and distribution of remedy across all affected parties. ILO Convention 169 and UNDRIP recognize the collective rights of indigenous peoples, which demands collective grievance handling capability.

- **No regulatory-specific reporting**: EUDR-031 generates a "Grievance Mechanism Annual Report" for general compliance purposes. But competent authorities under EUDR Articles 14-16 may request specific inspection documentation: detailed case files for selected grievances, statistical analysis of grievance patterns by commodity and country, evidence of mechanism effectiveness per UNGP Principle 31 criteria, cross-referencing of grievance data with risk assessment outcomes, and demonstration that grievance intelligence feeds back into the due diligence cycle. CSDDD Article 8 requires companies to report on grievance mechanism effectiveness with specific metrics. The OECD Guidelines for Multinational Enterprises (Chapter IV) recommend that companies publish grievance mechanism performance data. These are specialized regulatory reports that go far beyond what EUDR-031's general compliance reporter can produce.

Without these advanced capabilities, operators face the following risks: failure to detect systemic supply chain problems until they escalate to enforcement actions (up to 4% of EU annual turnover under EUDR); inability to satisfy CSDDD Article 8 effectiveness requirements when enforcement begins in 2027; inability to respond to competent authority inspection requests for grievance analytics under Articles 14-16; persistent remediation failures that generate repeat grievances and erode community trust; and litigation risk from affected communities and NGOs who can demonstrate that the operator possessed grievance data showing systemic problems but failed to act on the patterns.

### 1.2 Solution Overview

Agent-EUDR-032: Grievance Mechanism Manager is a specialized advanced management layer that sits on top of EUDR-031's operational grievance mechanism. It consumes grievance data from EUDR-031's grievance tables, enriches it with supply chain context from EUDR-001, risk signals from EUDR-016/017/018/028, indigenous rights data from EUDR-021, and mitigation intelligence from EUDR-025/029, and provides seven advanced engines that transform raw grievance operations into strategic grievance intelligence.

The relationship between EUDR-031 and EUDR-032 is analogous to the relationship between a transactional database and a business intelligence layer: EUDR-031 is the system of record for grievance operations (intake, triage, investigate, resolve); EUDR-032 is the system of intelligence for grievance management (analyze, predict, mediate, verify, report).

Core capabilities:

1. **Grievance Analytics Engine** -- Pattern detection, trend analysis, clustering, hotspot identification, and statistical analysis across all grievance data. Uses time-series analysis on grievance volumes by geography, commodity, category, and severity. Applies DBSCAN clustering to identify geographic hotspots. Detects seasonal patterns, escalation trends, and recurrence patterns. Generates executive dashboards with drill-down capability. All analytics are deterministic -- no LLM in the analytics path.

2. **Root Cause Analysis Engine** -- Systematic root cause identification using structured methodologies: 5-Whys decomposition with evidence linkage at each level, Ishikawa (fishbone) diagram generation across 6 cause categories (People, Process, Policy, Place, Product, Partner), cross-case correlation analysis that identifies common causal factors across seemingly unrelated grievances, and causal chain visualization. Generates root cause reports with confidence scoring and recommended systemic interventions.

3. **Multi-Party Mediation Manager** -- Complex mediation workflow management for disputes involving 3+ parties. Neutral mediator assignment from approved mediator registry. Structured mediation session management with agenda, position documentation from each party, negotiation tracking, and minutes. Settlement negotiation with term tracking, draft agreement management, and party sign-off workflow. Post-settlement compliance monitoring with milestone tracking and verification. Integrates with EUDR-031 for grievance case context and with EUDR-025 for remediation plan design.

4. **Remediation Effectiveness Tracker** -- Longitudinal measurement and verification of remediation action effectiveness. Defines effectiveness metrics per remediation type (environmental restoration, community compensation, process improvement, policy change). Tracks leading indicators (activity milestones) and lagging indicators (outcome metrics) over configurable monitoring periods (6, 12, 24, 36 months). Stakeholder satisfaction re-assessment at defined intervals. Integration with EUDR-003/004 satellite monitoring for environmental remediation verification. Generates effectiveness scorecards with trend visualization.

5. **Grievance Risk Scoring Engine** -- Predictive analytics using historical grievance data and contextual signals to score individual grievances and grievance portfolios for escalation probability, recurrence risk, regulatory exposure, and reputational impact. Risk model uses deterministic weighted scoring across 8 dimensions: severity, category, geographic risk, supplier risk, historical recurrence, stakeholder influence, media/NGO attention, and regulatory sensitivity. Generates risk-ranked grievance portfolios for resource prioritization. All scoring deterministic and reproducible.

6. **Collective Grievance Handler** -- Management of class-action style collective grievances affecting multiple stakeholders. Representative complainant identification and verification. Affected population mapping with geographic and demographic analysis. Collective harm assessment methodology with standardized impact metrics. Community-wide remediation planning with equitable distribution of remedy. Integration with EUDR-021 for indigenous collective rights recognition. Collective settlement management with multi-party agreement and benefit distribution tracking.

7. **Regulatory Reporter** -- Specialized regulatory report generation for competent authority inspections, CSDDD compliance, and international standards. Report types: (a) EUDR Article 16 Inspection Report -- detailed grievance case files, pattern analysis, and mechanism effectiveness evidence for competent authority inspections; (b) CSDDD Article 8 Compliance Report -- grievance mechanism establishment, accessibility, effectiveness metrics, and remediation outcomes per CSDDD requirements; (c) UNGP Principle 31 Effectiveness Assessment -- evaluation against all 8 effectiveness criteria with evidence mapping; (d) OECD Guidelines Performance Report -- grievance mechanism performance data per OECD recommendations; (e) ILO Convention 169 Indigenous Grievance Report -- indigenous-specific grievance handling, collective rights recognition, and FPIC-related grievance outcomes; (f) Cross-Regulatory Consolidated Report -- unified report covering all regulatory frameworks. All reports include SHA-256 provenance hashes, regulatory article mapping, and evidence cross-references.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Pattern detection accuracy | >= 95% of systemic patterns identified within 30 days of emergence | Precision/recall against expert-identified patterns |
| Root cause identification rate | >= 80% of recurring grievances traced to root cause within 60 days | % of recurrent cases with documented root cause |
| Multi-party mediation resolution rate | >= 70% of multi-party disputes resolved through mediation | % of mediation cases reaching settlement |
| Mediation time-to-resolution | < 90 days median for standard multi-party disputes | Median days from mediation initiation to settlement |
| Remediation effectiveness verification | 100% of high-severity remediations tracked for minimum 12 months | % of remediations with longitudinal effectiveness data |
| Remediation success rate | >= 75% of remediations rated "effective" or "highly effective" | Effectiveness scorecards at 12-month mark |
| Grievance risk prediction accuracy | >= 80% accuracy for escalation predictions (AUC >= 0.80) | Backtested against historical escalation outcomes |
| Collective grievance coverage | 100% of community-wide impacts handled as collective grievances | % of multi-stakeholder impacts with collective case |
| Regulatory report generation time | < 30 seconds per report (JSON); < 60 seconds (PDF) | Report generation benchmarks |
| Regulatory report completeness | 100% of required fields populated per regulatory framework | Schema validation per framework specification |
| Zero-hallucination guarantee | 100% deterministic analytics and scoring, no LLM in critical path | Bit-perfect reproducibility tests |
| EUDR Article 14-16 inspection readiness | 100% of inspection data producible within 24 hours of request | Mock inspection response time |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: The global grievance mechanism and case management technology market is estimated at 6-8 billion EUR, driven by the convergence of EUDR (400,000+ operators), CSDDD (13,000+ companies by 2029), the UK Modern Slavery Act, the German Supply Chain Due Diligence Act (LkSG), the French Duty of Vigilance Law, and the Norwegian Transparency Act. Each regulation mandates or recommends operational grievance mechanisms with effectiveness monitoring and regulatory reporting.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of EUDR-regulated commodities that already operate or are building grievance mechanisms under EUDR-031, plus the 13,000+ CSDDD-scope companies requiring advanced grievance analytics and regulatory reporting, estimated at 500-900M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 300+ enterprise customers in Year 1 deploying EUDR-032 on top of EUDR-031 grievance infrastructure, representing 20-35M EUR in grievance management module ARR. Additional revenue from CSDDD and LkSG cross-sell as enforcement timelines converge.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) with operational EUDR-031 grievance mechanisms generating 100+ grievances annually
- Multinational food and beverage companies with multi-country supply chains generating complex, multi-party grievances
- Timber and paper industry operators facing indigenous rights disputes requiring mediation
- Companies subject to both EUDR and CSDDD requiring dual-framework grievance reporting
- Operators in high-risk commodity supply chains (palm oil, cocoa, rubber) with elevated grievance volumes

**Secondary:**
- Compliance consulting firms managing grievance mechanisms on behalf of multiple operators
- Certification bodies (FSC, RSPO, PEFC, Rainforest Alliance) requiring evidence of grievance mechanism effectiveness for certification audits
- Financial institutions requiring advanced grievance analytics for ESG due diligence
- NGOs and indigenous rights organizations monitoring operator grievance mechanism performance
- Industry associations coordinating sector-wide grievance mechanisms (e.g., cocoa industry collective mechanisms)

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| EUDR-031 alone (baseline) | Operational grievance lifecycle; multi-channel intake; UNGP Principle 31 compliant | No analytics; no mediation; no effectiveness tracking; no predictive scoring; no collective handling; no regulatory-specific reporting | Complete advanced management layer on top of EUDR-031 operational base |
| Generic case management (ServiceNow, Zendesk, Freshdesk) | Mature case management; analytics dashboards; SLA tracking | Not EUDR/CSDDD-aware; no mediation workflows; no remediation effectiveness tracking; no indigenous rights; no regulatory reporting | Purpose-built for EUDR/CSDDD regulatory compliance; indigenous rights-aware; regulatory report templates |
| GRC platforms (OneTrust, Navex Global, NICE Actimize) | Enterprise risk management; whistleblower channels; compliance reporting | Generic compliance; no EUDR-specific analytics; no root cause methodology; no multi-party mediation; no collective grievance handling | EUDR-native analytics; structured root cause methodology; multi-party mediation; collective grievance handling |
| Human rights due diligence platforms (Ulula, Diginex, Fair Wage Network) | Worker voice; community feedback; human rights focus | Narrow scope (labor/worker focus); no EUDR integration; no supply chain risk linkage; no regulatory reporting for EUDR/CSDDD | Full EUDR/CSDDD/UNGP/ILO/OECD coverage; supply chain-integrated; multi-regulatory reporting |
| Sustainability consultancies (ERM, BSR, Shift) | Deep mediation expertise; UNGP knowledge; community relationships | Project-based (EUR 100-300K per engagement); no technology platform; no real-time analytics; not scalable | Always-on platform; 10x more cost-effective; real-time analytics; scales to thousands of grievances |
| In-house custom analytics | Tailored to organization; full data control | 12-18 month build; no regulatory templates; no mediation workflows; no benchmarking | Ready now; pre-built regulatory templates; structured mediation; cross-customer benchmarking |

### 2.4 Differentiation Strategy

1. **EUDR-031 native extension** -- Not a standalone grievance tool. Designed specifically to extend EUDR-031's operational base with advanced management capabilities, sharing the same database schema, authentication, and supply chain context.
2. **Multi-regulatory reporting** -- Single platform generates reports for EUDR Articles 14-16, CSDDD Article 8, UNGP Principle 31, OECD Guidelines, and ILO Convention 169 from the same grievance data.
3. **Structured root cause methodology** -- The only platform with built-in 5-Whys and Ishikawa analysis engines that link root causes to grievance evidence and generate systemic intervention recommendations.
4. **Multi-party mediation** -- Purpose-built mediation workflow for complex supply chain disputes involving indigenous communities, cooperatives, local governments, NGOs, certification bodies, and operators.
5. **Remediation verification** -- Longitudinal effectiveness tracking with satellite integration for environmental remediation verification, closing the loop between promise and outcome.
6. **Zero-hallucination analytics** -- All pattern detection, clustering, scoring, and reporting is deterministic with no LLM in the critical path. Every analytical output is reproducible and provenance-tracked.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable operators to demonstrate grievance mechanism effectiveness to competent authorities under EUDR Articles 14-16 | 100% of customers pass competent authority inspection for grievance mechanism component | Q3 2026 |
| BG-2 | Reduce recurring grievances by identifying and addressing root causes | 50% reduction in grievance recurrence rate within 12 months of deployment | Q1 2027 |
| BG-3 | Resolve multi-party disputes through structured mediation instead of litigation | 70%+ mediation success rate; 60% reduction in dispute-related legal costs | Q4 2026 |
| BG-4 | Build CSDDD-ready grievance effectiveness reporting ahead of 2027 enforcement | Grievance management module satisfies CSDDD Article 8 effectiveness requirements | Q4 2026 |
| BG-5 | Prevent regulatory penalties by detecting systemic grievance patterns early | Zero EUDR penalties for active customers attributable to undetected grievance patterns | Ongoing |
| BG-6 | Become the reference advanced grievance management platform for EUDR and CSDDD compliance | 300+ enterprise customers using grievance management module | Q4 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Grievance intelligence | Transform raw grievance data into actionable intelligence through pattern detection, trend analysis, and hotspot identification |
| PG-2 | Root cause resolution | Identify systemic root causes behind recurring grievances using structured methodologies (5-Whys, Ishikawa) |
| PG-3 | Multi-party mediation | Manage complex multi-stakeholder disputes through structured mediation workflows with settlement tracking |
| PG-4 | Remediation verification | Track and verify the long-term effectiveness of remediation actions with longitudinal outcome measurement |
| PG-5 | Predictive risk scoring | Score grievances for escalation probability, recurrence risk, and regulatory exposure using deterministic predictive models |
| PG-6 | Collective grievance management | Handle class-action style collective grievances with representative complainant identification, population mapping, and community-wide remedy |
| PG-7 | Regulatory reporting | Generate inspection-ready reports for EUDR, CSDDD, UNGP, OECD, and ILO frameworks from a single grievance dataset |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Analytics query performance | < 2 seconds p99 for grievance analytics queries across 100,000+ grievance records |
| TG-2 | Pattern detection latency | < 30 seconds for full pattern analysis on 10,000 grievance records |
| TG-3 | Root cause analysis throughput | < 10 seconds for 5-Whys decomposition with cross-case correlation |
| TG-4 | Risk scoring performance | < 500ms per grievance risk score calculation |
| TG-5 | Report generation performance | < 30 seconds per regulatory report (JSON); < 60 seconds (PDF) |
| TG-6 | API response time | < 200ms p95 for standard queries |
| TG-7 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-8 | Zero-hallucination | 100% deterministic analytics and scoring, bit-perfect reproducibility |
| TG-9 | Data freshness | Analytics updated within 15 minutes of new grievance data in EUDR-031 |
| TG-10 | Mediation workflow performance | < 500ms per workflow state transition |

### 3.4 Non-Goals

1. Grievance intake, triage, investigation, and resolution processing (EUDR-031 handles operational grievance lifecycle)
2. Stakeholder registry management (EUDR-031 Feature 1 handles stakeholder mapping)
3. FPIC workflow management (EUDR-031 Feature 2 handles FPIC processes)
4. Multi-channel communication dispatch (EUDR-031 Feature 5 handles communications)
5. Supply chain graph topology management (EUDR-001 handles graph operations)
6. Risk assessment calculation (EUDR-028 Risk Assessment Engine handles overall risk)
7. Mitigation measure design (EUDR-029 Mitigation Measure Designer handles measure design)
8. DDS document generation (EUDR-030 Documentation Generator handles DDS assembly)
9. Direct legal representation or binding arbitration services
10. Payment processing for settlement or compensation disbursement

---

## 4. User Personas

### Persona 1: Grievance Manager -- Helena (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Grievance Management at a large EU cocoa and coffee importer |
| **Company** | 8,000 employees, sourcing from 15 countries, 500+ cooperatives, 400+ active grievances |
| **EUDR Pressure** | Competent authority inspection scheduled for Q3 2026; board mandate to demonstrate grievance mechanism effectiveness; CSDDD preparation required |
| **Pain Points** | Drowning in individual case processing; cannot see patterns across 400+ grievances; same types of complaints recur from different regions without anyone connecting the dots; multi-party disputes with indigenous communities, cooperatives, and local government stall without structured mediation; remediation actions are marked "complete" but community satisfaction remains low; cannot produce the analytics reports competent authorities are requesting |
| **Goals** | Pattern detection across grievance portfolio; systematic root cause identification; structured mediation for complex disputes; verified remediation effectiveness; inspection-ready regulatory reports |
| **Technical Skill** | High -- comfortable with analytics dashboards, case management systems, and compliance platforms |

### Persona 2: Mediator -- Dr. Laurent (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | External mediator specializing in land rights and indigenous community disputes |
| **Company** | Independent mediation practice serving EU commodity importers |
| **EUDR Pressure** | Engaged by operators to mediate complex supply chain disputes involving indigenous communities, cooperatives, and government authorities |
| **Pain Points** | No structured platform for multi-party mediation; position documents exchanged via email; settlement terms tracked in Word documents; no compliance monitoring after agreement; difficulty demonstrating mediation effectiveness to operators |
| **Goals** | Structured mediation workspace with party management, session scheduling, position documentation, settlement tracking, and post-agreement monitoring; evidence trail for mediation quality |
| **Technical Skill** | Moderate -- comfortable with web platforms, document management, and video conferencing |

### Persona 3: Compliance Officer -- Erik (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | VP Regulatory Compliance at a European rubber and tire manufacturer |
| **Company** | 20,000 employees, rubber supply chain spanning Thailand, Indonesia, Vietnam, and Cameroon |
| **EUDR Pressure** | Must demonstrate grievance mechanism effectiveness to competent authorities under EUDR Articles 14-16; preparing for CSDDD Article 8 compliance; FSC and FSC-controlled wood certification requires grievance mechanism effectiveness evidence |
| **Pain Points** | Cannot demonstrate that the grievance mechanism is a "source of continuous learning" per UNGP Principle 31(g); no data connecting grievance patterns to risk assessment updates; no regulatory-specific report templates; auditors request analytics that the current system cannot produce |
| **Goals** | Regulatory reports mapped to specific EUDR Articles, CSDDD Articles, UNGP Principles, and OECD Guidelines; grievance-to-risk feedback loop; annual effectiveness assessment; inspection-ready documentation package |
| **Technical Skill** | Moderate -- comfortable with compliance platforms, reporting tools, and regulatory documentation |

### Persona 4: Remediation Coordinator -- Isabel (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Remediation Program Manager at an EU palm oil refinery |
| **Company** | 4,000 employees, sourcing from Indonesia and Malaysia, 50+ active remediation programs |
| **EUDR Pressure** | CSDDD Article 9 requires verified remediation of adverse impacts; certification bodies requesting evidence of remediation effectiveness; communities reporting that promised remediations have not materialized |
| **Pain Points** | Remediation marked "complete" when activity starts, not when outcome is verified; no longitudinal tracking of environmental restoration success; no stakeholder re-assessment of satisfaction; cannot demonstrate to auditors that remediations actually achieved their intended outcomes |
| **Goals** | Longitudinal remediation effectiveness tracking; satellite-verified environmental restoration; stakeholder satisfaction re-assessment; effectiveness scorecards for each remediation program |
| **Technical Skill** | Moderate-high -- comfortable with project management tools, GIS platforms, and sustainability reporting |

### Persona 5: External Auditor -- Dr. Hofmann (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must evaluate grievance mechanism effectiveness as part of EUDR due diligence audits |
| **Pain Points** | Operators provide basic case counts but no pattern analysis; cannot verify that grievance intelligence feeds back into risk assessment; no standardized way to assess UNGP Principle 31 compliance across all 8 criteria; remediation effectiveness data incomplete or absent |
| **Goals** | Read-only access to grievance analytics dashboards; UNGP Principle 31 effectiveness scorecard; root cause analysis reports; remediation effectiveness data; regulatory report verification with provenance hashes |
| **Technical Skill** | Moderate -- comfortable with audit software, analytics dashboards, and compliance assessment tools |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 10(1)** | Operators shall assess and identify risk of non-compliance | Grievance Analytics Engine (F1) identifies systemic risk patterns from grievance data; Grievance Risk Scoring Engine (F5) quantifies escalation and recurrence risk; analytics output feeds EUDR-028 risk assessment |
| **Art. 10(2)(e)** | Risk factor: information from consultations with indigenous peoples, local communities, and other stakeholders | Grievance pattern data, root cause findings, and collective grievance outcomes feed into risk assessment as structured stakeholder intelligence per Article 10(2)(e) |
| **Art. 11(1)** | Adopt adequate and proportionate risk mitigation measures | Root Cause Analysis Engine (F2) identifies systemic interventions; Remediation Effectiveness Tracker (F4) verifies that mitigation measures work; feedback loop to EUDR-029 for mitigation refinement |
| **Art. 11(2)(c)** | Mitigation: other measures to manage and mitigate non-negligible risk | Multi-party mediation (F3), collective grievance resolution (F6), and verified remediation (F4) are recognized risk mitigation measures |
| **Art. 14(1)** | Competent authorities shall carry out checks on operators' due diligence systems | Regulatory Reporter (F7) generates EUDR Article 16 Inspection Reports with grievance analytics, pattern data, root cause findings, and mechanism effectiveness evidence on demand |
| **Art. 14(2)** | Checks shall include examination of the due diligence system, including risk assessment and risk mitigation | Inspection reports include evidence that grievance intelligence is integrated into risk assessment (EUDR-028) and risk mitigation (EUDR-029) cycles |
| **Art. 15(1)** | Competent authorities may require operators to take remedial action | Remediation Effectiveness Tracker (F4) provides evidence of remedial action implementation and outcome verification |
| **Art. 16(1)** | Powers of competent authorities including requesting information and documents | Regulatory Reporter (F7) produces inspection-ready documentation packages within 24 hours of request; all data provenance-tracked |
| **Art. 29(3)(c)** | Country benchmarking based on respect for indigenous peoples' rights | Collective Grievance Handler (F6) and indigenous-specific grievance analytics contribute to country-level indigenous rights assessment via EUDR-016 |
| **Art. 31(1)** | Record keeping for 5 years | All analytics results, root cause analyses, mediation records, remediation effectiveness data, risk scores, and regulatory reports retained for minimum 5 years with immutable audit trail |

### 5.2 CSDDD Articles Addressed

| CSDDD Article | Requirement | Agent Implementation |
|---------------|-------------|---------------------|
| **Art. 7** | Meaningful engagement with affected stakeholders in due diligence | Multi-Party Mediation Manager (F3) provides structured engagement for dispute resolution; Collective Grievance Handler (F6) ensures collective stakeholder voice is heard |
| **Art. 8(1)** | Companies shall establish and maintain a complaints procedure | EUDR-031 provides the complaints procedure; EUDR-032 provides the effectiveness monitoring and management layer required for "maintain" obligation |
| **Art. 8(2)** | Complaints procedure must be accessible to affected persons | Regulatory Reporter (F7) generates CSDDD Article 8 accessibility evidence including channel coverage, language coverage, and geographic reach |
| **Art. 8(3)** | Companies shall respond to complainants and seek resolution | Analytics Engine (F1) measures response times, resolution rates, and satisfaction scores; Regulatory Reporter (F7) generates Article 8(3) compliance evidence |
| **Art. 8(4)** | Companies shall publish information on how the complaints procedure can be used | Regulatory Reporter (F7) generates public-facing mechanism performance reports |
| **Art. 9(1)** | Companies shall take appropriate measures to remediate adverse impacts | Remediation Effectiveness Tracker (F4) provides verified remediation evidence; Root Cause Analysis Engine (F2) ensures systemic remediation |
| **Art. 9(3)** | Where adverse impact cannot be immediately remediated, companies shall develop corrective action plans | Root Cause Analysis Engine (F2) generates systemic corrective action plans; Remediation Effectiveness Tracker (F4) monitors plan implementation |

### 5.3 UN Guiding Principles (UNGP) Principle 31 Effectiveness Assessment

| Criterion | Assessment Method in EUDR-032 |
|-----------|-------------------------------|
| **(a) Legitimate** | Regulatory Reporter (F7) documents governance structure, independence of mediation, and oversight mechanisms; Multi-Party Mediation Manager (F3) provides neutral mediator assignment |
| **(b) Accessible** | Analytics Engine (F1) measures channel utilization, language coverage, geographic reach, and time-to-access metrics; identifies accessibility gaps |
| **(c) Predictable** | Analytics Engine (F1) measures SLA compliance rates, process stage durations, and outcome consistency; Regulatory Reporter (F7) publishes process guidelines and timelines |
| **(d) Equitable** | Analytics Engine (F1) measures outcome fairness across stakeholder types, geographic regions, and severity levels; identifies systemic bias in resolution outcomes |
| **(e) Transparent** | Regulatory Reporter (F7) generates public-facing anonymized reports; Analytics Engine (F1) provides dashboard data for stakeholder-facing transparency portal |
| **(f) Rights-compatible** | Root Cause Analysis Engine (F2) ensures remediation aligns with international human rights standards; Collective Grievance Handler (F6) recognizes collective rights; outcomes do not preclude legal remedy |
| **(g) Source of continuous learning** | **Primary differentiator of EUDR-032**. Analytics Engine (F1) detects patterns that feed EUDR-028 risk assessment; Root Cause Analysis Engine (F2) identifies systemic interventions; Remediation Effectiveness Tracker (F4) measures learning effectiveness; feedback loop documented in regulatory reports |
| **(h) Based on engagement and dialogue** | Multi-Party Mediation Manager (F3) provides structured dialogue; Collective Grievance Handler (F6) enables community-wide engagement; stakeholder satisfaction tracked longitudinally |

### 5.4 ILO Convention 169 Requirements

| ILO 169 Article | Requirement | Agent Implementation |
|-----------------|-------------|---------------------|
| **Art. 6(1)(a)** | Consult indigenous peoples through appropriate procedures | Collective Grievance Handler (F6) manages indigenous collective complaints through culturally appropriate processes; mediation respects community governance |
| **Art. 12** | Indigenous peoples shall be safeguarded against violations of their rights and shall be able to take legal proceedings | Grievance mechanism does not preclude legal remedy; Regulatory Reporter (F7) documents accessibility for indigenous complainants |
| **Art. 15(2)** | Fair compensation for damages sustained as a result of resource exploitation activities | Remediation Effectiveness Tracker (F4) verifies compensation delivery and adequacy; Collective Grievance Handler (F6) manages equitable distribution |

### 5.5 OECD Guidelines for Multinational Enterprises

| OECD Chapter | Requirement | Agent Implementation |
|--------------|-------------|---------------------|
| **Ch. IV, Para 46** | Enterprises should provide or participate in operational-level grievance mechanisms | EUDR-031 provides the mechanism; EUDR-032 ensures it meets OECD effectiveness expectations |
| **Ch. IV, Para 47** | Grievance mechanisms should be a source of learning | Analytics Engine (F1) and Root Cause Analysis Engine (F2) extract systematic learning; feedback loop to risk assessment documented |
| **Ch. IV, Para 48** | Enterprises should report on grievance mechanism performance | Regulatory Reporter (F7) generates OECD Guidelines Performance Reports with mechanism metrics, pattern analysis, and effectiveness assessment |

### 5.6 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 30, 2025 | EUDR enforcement for large operators (ACTIVE) | Operators must have grievance mechanism operational; EUDR-032 provides the management layer |
| June 30, 2026 | EUDR enforcement for SMEs | SME onboarding wave; simplified analytics templates required |
| 2027 (phased) | CSDDD enforcement begins | CSDDD Article 8 grievance mechanism effectiveness reporting mandatory; EUDR-032 Regulatory Reporter ready |
| Ongoing (annually) | CSDDD Article 8(4) reporting | Annual grievance mechanism performance publication |
| Ongoing | EUDR Articles 14-16 competent authority inspections | Inspection-ready reports available on demand |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 7 features below are P0 launch blockers. The agent cannot ship without all 7 features operational. Features 1-2 form the analytics and intelligence core; Features 3-4 form the advanced dispute and remediation management layer; Features 5-6 form the predictive and collective handling capability; Feature 7 is the regulatory delivery mechanism.

**P0 Features 1-2: Analytics and Intelligence Core**

---

#### Feature 1: Grievance Analytics Engine

**User Story:**
```
As a grievance manager,
I want to analyze patterns, trends, clusters, and hotspots across my entire grievance portfolio,
So that I can identify systemic problems, allocate resources effectively,
and demonstrate to competent authorities that my grievance mechanism is a source of continuous learning.
```

**Acceptance Criteria:**
- [ ] Time-series trend analysis on grievance volumes by: geography (country, region), commodity, category, severity, supply chain segment, and stakeholder type
- [ ] DBSCAN geographic clustering to identify grievance hotspots with configurable distance threshold (default 50km radius) and minimum cluster size (default 3 grievances)
- [ ] Category-based pattern detection: identifies when a specific grievance category (e.g., land rights) exceeds statistical threshold (configurable, default 2 standard deviations above rolling 90-day mean) in a specific geography or supply chain segment
- [ ] Recurrence analysis: identifies grievances that share characteristics with previously resolved cases (same location, same supplier, same category within configurable time window)
- [ ] Escalation trend detection: identifies increasing severity or frequency patterns that predict future escalation
- [ ] Seasonal pattern detection: identifies cyclical grievance patterns linked to production seasons, certification cycles, or regulatory timelines
- [ ] Cross-supply-chain correlation: identifies when grievance patterns in one commodity supply chain correlate with patterns in another (e.g., land rights issues affecting both palm oil and rubber in same region)
- [ ] Executive dashboard with drill-down: high-level KPIs (total active, resolution rate, average resolution time, satisfaction score, pattern count) with ability to drill into specific patterns, geographies, and commodities
- [ ] Comparative benchmarking: compare grievance metrics across supply chain segments, commodities, and time periods
- [ ] Export analytics results as structured JSON, CSV, and PDF for integration with EUDR-028 risk assessment and EUDR-029 mitigation design
- [ ] All analytics deterministic: same input data produces identical analytical outputs

**Non-Functional Requirements:**
- Performance: Full pattern analysis < 30 seconds for 10,000 grievance records; dashboard refresh < 2 seconds
- Freshness: Analytics updated within 15 minutes of new grievance data in EUDR-031
- Determinism: Bit-perfect reproducibility across runs with same input data
- Scalability: Handles 100,000+ historical grievance records without degradation

**Dependencies:**
- EUDR-031 grievance tables (eudr_stakeholder_engagement.grievances, eudr_stakeholder_engagement.grievance_status_transitions)
- EUDR-001 supply chain graph for supply chain segment context
- EUDR-016 Country Risk Evaluator for geographic risk context
- TimescaleDB time-series functions for temporal analytics

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 data engineer)

**Edge Cases:**
- Very few grievances (< 10 total) -- analytics returns "insufficient data" with minimum sample size warning
- All grievances from single location -- clustering returns single cluster with geographic concentration alert
- Grievance data quality issues (missing geography, missing category) -- analytics handles partial data with completeness scoring

---

#### Feature 2: Root Cause Analysis Engine

**User Story:**
```
As a grievance manager,
I want to systematically identify the root causes behind recurring grievances
using structured methodologies,
So that I can design systemic interventions that address underlying problems
rather than treating symptoms case by case.
```

**Acceptance Criteria:**
- [ ] **5-Whys Decomposition**: Guided workflow for iterative "why" questioning with evidence linkage at each level; supports branching (multiple contributing causes at any level); generates 5-Whys tree visualization; stores completed analyses with provenance hash
- [ ] **Ishikawa (Fishbone) Diagram**: Categorizes causes across 6 standard dimensions -- People (training, awareness, capacity), Process (procedures, workflows, controls), Policy (corporate policy, supplier policy, regulatory gap), Place (geographic, environmental, infrastructure), Product (commodity characteristics, quality issues), Partner (supplier practices, certification gaps, sub-tier issues); generates fishbone visualization; supports custom cause categories
- [ ] **Cross-Case Correlation**: Analyzes multiple grievances to identify common causal factors; correlation based on: shared supplier, shared geography, shared commodity, shared category, shared timeframe, shared supply chain segment; generates correlation matrix with strength scoring (0-1.0)
- [ ] **Causal Chain Visualization**: Renders interactive causal chain from root cause to grievance manifestation with intermediate factors; supports zoom, drill-down, and evidence inspection at each node
- [ ] **Root Cause Classification**: Classifies identified root causes into taxonomy: Systemic (organizational/policy), Operational (process/procedure), Environmental (geographic/climatic), Supplier-Specific (individual supplier behavior), Regulatory (legal/regulatory gap), External (market/political)
- [ ] **Intervention Recommendation**: For each identified root cause, generates recommended systemic interventions with: description, scope (which supply chains affected), estimated effort, expected impact on grievance recurrence, timeline, and responsible party suggestion
- [ ] **Confidence Scoring**: Each root cause finding includes confidence score (0-100) based on: number of supporting grievances, evidence quality, correlation strength, and consistency of causal pattern
- [ ] **Root Cause Report**: Generates structured report with executive summary, methodology, findings, causal visualizations, intervention recommendations, and evidence appendix; SHA-256 provenance hash
- [ ] Integration with EUDR-029 Mitigation Measure Designer for feeding root cause findings into mitigation planning
- [ ] All analysis deterministic and reproducible

**Non-Functional Requirements:**
- Performance: 5-Whys session response < 2 seconds per step; cross-case correlation < 10 seconds for 1,000 cases
- Auditability: Complete audit trail of all root cause analyses with analyst identity, methodology used, evidence cited, and conclusions drawn
- Determinism: Cross-case correlation produces identical results on same input data

**Dependencies:**
- Feature 1 (Grievance Analytics Engine) for pattern and cluster data
- EUDR-031 grievance case data for individual case details
- EUDR-029 Mitigation Measure Designer for intervention planning integration

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

**P0 Features 3-4: Advanced Dispute and Remediation Management**

---

#### Feature 3: Multi-Party Mediation Manager

**User Story:**
```
As a mediator,
I want a structured workspace to manage complex supply chain disputes involving
multiple stakeholders with different interests and legal standings,
So that I can facilitate fair resolution through mediation rather than litigation,
with documented evidence of the mediation process for regulatory compliance.
```

**Acceptance Criteria:**
- [ ] **Mediation Case Creation**: Create mediation case from one or more EUDR-031 grievances; link to supply chain nodes (EUDR-001); define dispute scope, parties, and desired outcomes
- [ ] **Party Management**: Register 3+ parties per mediation case with: party identity, legal standing, representation (individual, community representative, legal counsel, NGO advocate), contact details, language preferences, position summary, desired outcomes
- [ ] **Neutral Mediator Assignment**: Maintain approved mediator registry with: qualifications, specializations (land rights, environmental, labor, indigenous rights), language capabilities, availability, conflict-of-interest declarations; assign mediator to case with party acceptance workflow
- [ ] **Mediation Session Management**: Schedule mediation sessions with: agenda, invited parties, location (physical or virtual), pre-session materials distribution, session minutes capture, action item tracking, next session scheduling
- [ ] **Position Documentation**: Each party submits structured position documents with: factual claims, legal basis, evidence references, desired outcomes, acceptable compromise ranges; position documents visible to mediator and (optionally) to other parties
- [ ] **Negotiation Tracking**: Track negotiation progress across mediation sessions with: issues list, status per issue (open/progressing/agreed/impasse), party positions per issue, mediator notes, proposed compromises
- [ ] **Settlement Management**: Draft settlement agreement with: agreed terms per issue, party obligations, timeline for implementation, verification mechanisms, consequences of non-compliance; multi-party sign-off workflow with electronic signatures
- [ ] **Post-Settlement Compliance Monitoring**: Track settlement implementation with: milestone checklist, evidence of completion per obligation, party confirmation, dispute resolution if party alleges non-compliance
- [ ] **Mediation Confidentiality**: Mediation communications and position documents confidential by default; disclosure only with all-party consent or legal requirement; separation of mediation data from general grievance data
- [ ] **Mediation Outcome Classification**: Classify outcomes as: Full Settlement (all issues resolved), Partial Settlement (some issues resolved, others referred), Impasse (mediation failed, parties may pursue other remedies), Withdrawn (party withdrew from mediation)
- [ ] Complete audit trail of all mediation activities with timestamps, actors, and provenance hashes

**Mediation Lifecycle State Machine:**
```
CASE_CREATED --> PARTIES_REGISTERED --> MEDIATOR_ASSIGNED --> MEDIATOR_ACCEPTED
    --> PRE_MEDIATION_PREPARATION --> MEDIATION_IN_PROGRESS --> [SESSION_N]
    --> SETTLEMENT_DRAFTED --> SETTLEMENT_SIGNED --> IMPLEMENTATION_MONITORING
    --> SETTLEMENT_VERIFIED --> CLOSED
    --> IMPASSE_DECLARED --> REFERRED_TO_ALTERNATIVE_PROCESS
```

**Non-Functional Requirements:**
- Performance: Session creation < 500ms; party lookup < 100ms; settlement draft generation < 5 seconds
- Security: Mediation data encrypted at rest (AES-256); access restricted to assigned mediator and registered parties; no cross-case data leakage
- Confidentiality: Mediation communications privileged; system enforces confidentiality boundary

**Dependencies:**
- EUDR-031 grievance data for originating complaint context
- EUDR-031 Stakeholder Mapper (Feature 1) for party identification
- EUDR-025 Risk Mitigation Advisor for remediation plan design within settlements
- SEC-002 RBAC for mediation-specific access control
- SEC-003 Encryption for mediation data protection

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 frontend engineer)

**Edge Cases:**
- Party withdraws during mediation -- record withdrawal; remaining parties may continue or terminate
- Mediator conflict of interest discovered mid-process -- reassignment workflow; all sessions remain valid
- Settlement terms violated -- trigger compliance dispute sub-process within same mediation case
- Cross-jurisdictional dispute (parties in different countries) -- mediation supports multi-timezone scheduling and multi-language documentation

---

#### Feature 4: Remediation Effectiveness Tracker

**User Story:**
```
As a remediation coordinator,
I want to measure and verify the long-term effectiveness of remediation actions
over 6, 12, 24, and 36 month monitoring periods,
So that I can demonstrate to regulators and stakeholders that our remediation
actually achieves its intended outcomes, not just that activities were performed.
```

**Acceptance Criteria:**
- [ ] **Remediation Program Creation**: Create monitoring program for each remediation action from EUDR-031; define: remediation type, intended outcome, success criteria, monitoring duration (6/12/24/36 months), monitoring frequency (monthly/quarterly/semi-annual), responsible party
- [ ] **Effectiveness Metrics Definition**: Pre-built metric templates per remediation type:
  - Environmental Restoration: hectares restored, species survival rate, canopy cover %, NDVI change, water quality metrics
  - Community Compensation: payment disbursement rate, beneficiary satisfaction, livelihood impact
  - Process Improvement: recurrence rate reduction, compliance metric improvement, audit finding closure
  - Policy Change: policy adoption rate, stakeholder awareness, behavioral compliance
  - Infrastructure: completion status, utilization rate, maintenance compliance, community benefit
- [ ] **Leading Indicator Tracking**: Activity milestones (e.g., "replanting commenced", "training delivered", "policy published") with evidence and completion dates
- [ ] **Lagging Indicator Tracking**: Outcome metrics measured at defined intervals (e.g., "tree survival rate at 12 months = 78%", "community satisfaction at 24 months = 4.2/5") with evidence
- [ ] **Stakeholder Satisfaction Re-Assessment**: Periodic re-assessment of complainant and affected community satisfaction at configurable intervals; satisfaction trend tracking; comparison with initial resolution satisfaction
- [ ] **Satellite-Verified Environmental Remediation**: Integration with EUDR-003 (Satellite Monitoring) and EUDR-004 (Forest Cover Analysis) for remote verification of environmental restoration claims; NDVI comparison between pre-remediation and post-remediation satellite imagery
- [ ] **Effectiveness Scoring**: Composite effectiveness score (0-100) calculated from: activity completion rate (leading), outcome achievement rate (lagging), stakeholder satisfaction, and verification confidence; classification: Highly Effective (85-100), Effective (70-84), Partially Effective (50-69), Ineffective (25-49), Failed (0-24)
- [ ] **Effectiveness Scorecard**: Per-remediation visual scorecard with: timeline, milestones achieved, outcome metrics with trend, stakeholder satisfaction trend, effectiveness score with classification, satellite verification results (where applicable)
- [ ] **Portfolio Dashboard**: Aggregate effectiveness metrics across all remediation programs with drill-down by remediation type, geography, commodity, and time period
- [ ] **Escalation on Underperformance**: Automatic alerts when remediation effectiveness drops below configurable threshold (default: Partially Effective); escalation to EUDR-029 for mitigation refinement
- [ ] All effectiveness calculations deterministic; provenance hash on every scorecard

**Effectiveness Scoring Formula:**
```
Effectiveness_Score = (
    Activity_Completion_Rate * W_activity +
    Outcome_Achievement_Rate * W_outcome +
    Stakeholder_Satisfaction_Normalized * W_satisfaction +
    Verification_Confidence * W_verification
)

Where:
- W_activity = 0.20 (configurable)
- W_outcome = 0.40 (configurable)
- W_satisfaction = 0.25 (configurable)
- W_verification = 0.15 (configurable)
- All scores normalized to 0-100 scale
- Decimal arithmetic (no floating-point drift)
```

**Non-Functional Requirements:**
- Performance: Scorecard calculation < 2 seconds; portfolio dashboard < 5 seconds for 500 programs
- Longitudinal: System retains and queries effectiveness data spanning 36+ months
- Determinism: Effectiveness scores bit-perfect reproducible

**Dependencies:**
- EUDR-031 remediation action data (grievance.remediation_actions)
- EUDR-003 Satellite Monitoring Agent for environmental verification imagery
- EUDR-004 Forest Cover Analysis Agent for NDVI calculations
- EUDR-029 Mitigation Measure Designer for underperformance escalation
- EUDR-031 Communication Hub (Feature 5) for stakeholder re-assessment surveys

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 data engineer)

---

**P0 Features 5-6: Predictive and Collective Handling**

---

#### Feature 5: Grievance Risk Scoring Engine

**User Story:**
```
As a grievance manager,
I want each grievance and my overall grievance portfolio scored for escalation probability,
recurrence risk, regulatory exposure, and reputational impact,
So that I can prioritize resources toward the grievances most likely to cause material harm
and proactively intervene before escalation occurs.
```

**Acceptance Criteria:**
- [ ] **Individual Grievance Risk Score**: Deterministic weighted score (0-100) across 8 dimensions:
  - (1) Severity Score (0-100): Based on EUDR-031 triage severity classification
  - (2) Category Score (0-100): Based on category risk weight (human rights = 95, land rights = 90, environmental = 85, labor rights = 80, community impact = 70, process complaint = 40, information request = 10)
  - (3) Geographic Risk Score (0-100): Country/region risk from EUDR-016 Country Risk Evaluator
  - (4) Supplier Risk Score (0-100): Supplier compliance history from EUDR-017 Supplier Risk Scorer
  - (5) Historical Recurrence Score (0-100): Based on count and frequency of similar grievances from same location/supplier/category within lookback window (default 24 months)
  - (6) Stakeholder Influence Score (0-100): Based on complainant type, organizational affiliation, media connections, and legal representation
  - (7) Media/NGO Attention Score (0-100): Based on whether grievance topic, location, or supplier has active media or NGO scrutiny (fed from external monitoring or manual flag)
  - (8) Regulatory Sensitivity Score (0-100): Based on whether grievance relates to active regulatory proceedings, certification audits, or competent authority focus areas
- [ ] **Composite Risk Score**: Weighted aggregate of 8 dimensions with configurable weights (default equal weighting at 12.5% each)
- [ ] **Risk Classification**: Critical (85-100), High (70-84), Elevated (50-69), Moderate (25-49), Low (0-24)
- [ ] **Escalation Probability**: Deterministic probability estimate (0-100%) based on historical escalation rates for grievances with similar risk profiles
- [ ] **Portfolio Risk Score**: Aggregate risk score across all active grievances; identifies portfolio-level risk concentration by geography, commodity, supplier, and category
- [ ] **Risk-Ranked Grievance List**: All active grievances ranked by composite risk score for resource prioritization
- [ ] **Risk Trend Monitoring**: Track individual grievance risk scores over time; alert on increasing risk trajectory
- [ ] **Threshold Alerts**: Configurable alerts when individual grievance risk exceeds threshold (default: Elevated) or portfolio risk concentration exceeds threshold
- [ ] All scoring deterministic and bit-perfect reproducible; no LLM involvement

**Risk Scoring Formula:**
```
Grievance_Risk_Score = sum(
    Dimension_Score[i] * Dimension_Weight[i]
    for i in [severity, category, geographic, supplier, recurrence,
              stakeholder_influence, media_ngo, regulatory_sensitivity]
)

Default Weights: all 0.125 (equal weighting, configurable per operator)
```

**Non-Functional Requirements:**
- Performance: Individual risk score calculation < 500ms; portfolio scoring < 5 seconds for 1,000 active grievances
- Determinism: Bit-perfect reproducibility across runs
- Configurability: Risk dimension weights adjustable per operator without code changes

**Dependencies:**
- Feature 1 (Grievance Analytics Engine) for recurrence and trend data
- EUDR-031 grievance data for severity and category
- EUDR-016 Country Risk Evaluator for geographic risk
- EUDR-017 Supplier Risk Scorer for supplier risk
- EUDR-031 Stakeholder Mapper for stakeholder influence context

**Estimated Effort:** 2 weeks (1 senior backend engineer)

---

#### Feature 6: Collective Grievance Handler

**User Story:**
```
As a grievance manager,
I want to manage class-action style collective grievances where an adverse impact
affects an entire community, worker group, or stakeholder coalition,
So that I can address community-wide harms through collective processes
that recognize group rights per ILO Convention 169 and UNDRIP.
```

**Acceptance Criteria:**
- [ ] **Collective Case Creation**: Create collective grievance from: (a) multiple individual EUDR-031 grievances identified as related by Analytics Engine (Feature 1), (b) direct collective submission by community representative, or (c) referral from Multi-Party Mediation Manager (Feature 3)
- [ ] **Representative Complainant Management**: Identify and verify representative complainants for the collective; verify authority to represent affected group through: community meeting minutes, organizational mandate, legal authorization, or customary leadership recognition
- [ ] **Affected Population Mapping**: Define and map the affected population with: geographic boundary (polygon), estimated population count, demographic breakdown (where available), stakeholder types affected, supply chain nodes linked to the affected area
- [ ] **Collective Harm Assessment**: Structured harm assessment methodology with standardized impact metrics across 5 domains:
  - Environmental: area affected (ha), resource degradation level, ecosystem service loss, restoration timeline
  - Economic: livelihood impact (number of affected households), income loss estimate, asset damage
  - Social: community displacement, cultural heritage impact, health impact, education disruption
  - Rights: FPIC violation, land rights violation, labor rights violation, access restrictions
  - Temporal: duration of adverse impact, ongoing vs. historical, reversibility assessment
- [ ] **Collective Impact Score**: Deterministic composite score (0-100) from 5 harm domains with configurable weights
- [ ] **Community-Wide Remediation Planning**: Design remediation that addresses collective harm with: remediation type per harm domain, beneficiary identification, equitable distribution methodology, timeline, monitoring plan; integration with EUDR-029 for remediation measure design
- [ ] **Collective Settlement Management**: Multi-party settlement agreement covering: community-wide remedy terms, individual compensation where applicable, benefit-sharing arrangements, environmental restoration commitments, monitoring and verification obligations, dispute resolution clause
- [ ] **Benefit Distribution Tracking**: Track distribution of remedy/compensation across affected population with: beneficiary registry, distribution schedule, delivery confirmation, equity analysis
- [ ] **Indigenous Collective Rights**: Special handling for indigenous collective grievances per ILO Convention 169 and UNDRIP: recognition of collective land rights, community decision-making processes for remedy acceptance, culturally appropriate remediation, integration with EUDR-021 territory data
- [ ] **Collective Case Reporting**: Generate reports on collective grievance outcomes for regulatory compliance and stakeholder transparency
- [ ] All assessments deterministic; provenance hash on every collective case record

**Non-Functional Requirements:**
- Performance: Collective case creation < 3 seconds; population mapping query < 5 seconds; harm assessment calculation < 2 seconds
- Data: Handles collective cases with up to 10,000 affected individuals
- Auditability: Complete audit trail of all collective case decisions with evidence linkage

**Dependencies:**
- Feature 1 (Grievance Analytics Engine) for pattern-based collective case identification
- Feature 3 (Multi-Party Mediation Manager) for multi-party dispute resolution within collective cases
- EUDR-031 grievance data for linked individual complaints
- EUDR-021 Indigenous Rights Checker for indigenous territory data and collective rights context
- EUDR-001 Supply Chain Mapping Master for affected supply chain node identification
- EUDR-029 Mitigation Measure Designer for community-wide remediation design

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 domain expert)

---

**P0 Feature 7: Regulatory Delivery Mechanism**

---

#### Feature 7: Regulatory Reporter

**User Story:**
```
As a compliance officer,
I want to generate regulatory-specific grievance mechanism reports
that map precisely to the requirements of EUDR Articles 14-16, CSDDD Article 8,
UNGP Principle 31, OECD Guidelines, and ILO Convention 169,
So that I can respond to competent authority inspections, satisfy CSDDD reporting obligations,
and demonstrate grievance mechanism effectiveness to auditors and certification bodies.
```

**Acceptance Criteria:**
- [ ] Generates 6 specialized report types:
  - (a) **EUDR Article 16 Inspection Report**: Competent authority inspection package including: grievance case inventory (filterable by commodity, country, severity, date range), pattern analysis summary from Feature 1, root cause findings from Feature 2, mediation outcomes from Feature 3, remediation effectiveness data from Feature 4, risk scores from Feature 5, collective grievance outcomes from Feature 6, mechanism performance KPIs, evidence that grievance intelligence feeds into EUDR-028 risk assessment and EUDR-029 mitigation planning
  - (b) **CSDDD Article 8 Compliance Report**: Grievance mechanism establishment evidence, accessibility metrics (channels, languages, geographic reach), complaint processing statistics (volume, resolution rate, time-to-resolution), stakeholder satisfaction metrics, remediation outcomes, mechanism governance description, annual effectiveness assessment, public disclosure data
  - (c) **UNGP Principle 31 Effectiveness Assessment**: Structured evaluation against all 8 effectiveness criteria (legitimate, accessible, predictable, equitable, transparent, rights-compatible, source of continuous learning, based on engagement and dialogue) with evidence mapping, dimension scoring (0-100 per criterion), aggregate effectiveness score, improvement recommendations
  - (d) **OECD Guidelines Performance Report**: Grievance mechanism performance data per OECD Chapter IV recommendations: case volumes and trends, resolution effectiveness, stakeholder engagement quality, learning outcomes, cross-supply-chain comparison
  - (e) **ILO Convention 169 Indigenous Grievance Report**: Indigenous-specific grievance handling documentation: indigenous grievance volumes and categories, collective rights recognition evidence, culturally appropriate process evidence, FPIC-related grievance outcomes, indigenous community satisfaction, collective grievance resolution, benefit distribution equity
  - (f) **Cross-Regulatory Consolidated Report**: Unified report covering all regulatory frameworks from a single grievance dataset; matrix showing how each grievance mechanism component satisfies each regulatory requirement; gap analysis identifying uncovered requirements
- [ ] Each report includes: executive summary, regulatory article-by-article compliance mapping, detailed findings with evidence cross-references, statistical analysis, trend visualizations, recommendations, provenance chain, generation metadata
- [ ] SHA-256 provenance hash on every generated report
- [ ] Configurable report parameters: date range, commodity filter, country filter, severity filter, supply chain segment filter
- [ ] Formats: PDF (human-readable with charts and visualizations), JSON (machine-readable for EUDR-030 integration), HTML (web display), XLSX (tabular data export)
- [ ] Multi-language report generation: EN, FR, DE, ES, PT, ID, SW
- [ ] Batch report generation for portfolio-level reporting
- [ ] Report scheduling: configure automatic report generation at defined intervals (monthly, quarterly, annually) with email distribution
- [ ] Integration with EUDR-030 Documentation Generator for DDS package inclusion

**Non-Functional Requirements:**
- Performance: < 30 seconds per report (JSON); < 60 seconds (PDF); < 120 seconds for cross-regulatory consolidated report
- Completeness: All mandatory fields populated when upstream data available; missing fields flagged with data source reference
- Integrity: Reports immutable after generation; provenance hash verification on download
- On-Demand: Any report producible within 24 hours of competent authority request (most within minutes)

**Dependencies:**
- Features 1-6 (all core engines provide data for regulatory reports)
- EUDR-030 Documentation Generator for DDS package integration
- EUDR-031 grievance operational data for baseline statistics
- S3/Object Storage (INFRA-004) for report storage and retrieval

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 report engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 8: Grievance Benchmarking Service
- Cross-customer anonymized benchmarking of grievance mechanism performance
- Industry-level benchmarks by commodity (cocoa, palm oil, rubber, etc.)
- Regional benchmarks by producing country
- Operators can compare their mechanism performance against industry averages
- Opt-in data sharing with full anonymization

#### Feature 9: Stakeholder Trust Index
- Longitudinal trust measurement combining: grievance recurrence rate, satisfaction trends, engagement quality scores, remediation effectiveness, and stakeholder feedback
- Trust index per stakeholder group, region, and commodity supply chain
- Early warning when trust index declines below threshold
- Trust recovery tracking after successful remediation

#### Feature 10: Grievance-Informed Supply Chain Optimization
- Recommendation engine that uses grievance pattern data to suggest supply chain modifications
- Identify suppliers with persistent grievance patterns for enhanced monitoring or exit
- Suggest alternative sourcing routes that avoid grievance hotspots
- Cost-benefit analysis of supply chain changes informed by grievance risk

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Binding arbitration or legal adjudication (platform facilitates mediation only)
- Payment processing for compensation or settlement disbursement (integrate with external financial systems)
- NLP/AI-based grievance text analysis (all analytics use structured data for determinism)
- Real-time social media monitoring for grievance-related content (defer to v2.0)
- Mobile native application (responsive web for v1.0)
- Automated mediator matching using AI (manual assignment with registry search for v1.0)
- Predictive modeling using machine learning (v1.0 uses deterministic weighted scoring)

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
| AGENT-EUDR-032        |           | AGENT-EUDR-031            |           | AGENT-EUDR-030        |
| Grievance Mechanism   |<--------->| Stakeholder Engagement    |<--------->| Documentation         |
| Manager               |           | Tool                      |           | Generator             |
|                       |           |                           |           |                       |
| - Analytics Engine    |           | - Stakeholder Mapper      |           | - DDS Generation      |
| - Root Cause Engine   |           | - FPIC Workflow Engine    |           | - Package Builder     |
| - Mediation Manager   |           | - Grievance Mechanism     |           | - Submission Engine    |
| - Effectiveness Track |           | - Consultation Manager    |           |                       |
| - Risk Scoring Engine |           | - Communication Hub       |           |                       |
| - Collective Handler  |           | - Engagement Verifier     |           |                       |
| - Regulatory Reporter |           | - Compliance Reporter     |           |                       |
+-----------+-----------+           +---------------------------+           +-----------------------+
            |
+-----------v-----------+           +---------------------------+           +---------------------------+
| AGENT-EUDR-001        |           | AGENT-EUDR-016/017/018    |           | AGENT-EUDR-021            |
| Supply Chain Mapping  |           | Country/Supplier/Commodity|           | Indigenous Rights         |
| Master                |           | Risk Evaluators           |           | Checker                   |
|                       |           |                           |           |                           |
| - Graph Engine        |           | - Risk Scores             |           | - Territory Data          |
| - Node/Edge data      |           | - Country Benchmarking    |           | - Collective Rights       |
+-----------------------+           +---------------------------+           +---------------------------+
            |
+-----------v-----------+           +---------------------------+
| AGENT-EUDR-025/029    |           | AGENT-EUDR-003/004        |
| Risk Mitigation /     |           | Satellite Monitoring /    |
| Mitigation Designer   |           | Forest Cover Analysis     |
|                       |           |                           |
| - Remediation Plans   |           | - NDVI Verification       |
| - Measure Effectiveness|          | - Restoration Monitoring  |
+-----------------------+           +---------------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/grievance_mechanism_manager/
    __init__.py                          # Public API exports
    config.py                            # GrievanceMechanismManagerConfig with GL_EUDR_GMM_ env prefix
    models.py                            # Pydantic v2 models for analytics, mediation, effectiveness, risk, collective
    analytics_engine.py                  # GrievanceAnalyticsEngine: pattern detection, trends, clustering, hotspots
    root_cause_engine.py                 # RootCauseAnalysisEngine: 5-Whys, Ishikawa, cross-case correlation
    mediation_manager.py                 # MultiPartyMediationManager: mediation workflow, sessions, settlements
    effectiveness_tracker.py             # RemediationEffectivenessTracker: longitudinal monitoring, scoring
    risk_scoring_engine.py               # GrievanceRiskScoringEngine: 8-dimension risk scoring, portfolio risk
    collective_handler.py                # CollectiveGrievanceHandler: population mapping, harm assessment, remedy
    regulatory_reporter.py               # RegulatoryReporter: 6 report types, multi-framework compliance
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains
    metrics.py                           # 15 Prometheus self-monitoring metrics
    setup.py                             # GrievanceMechanismManagerService facade
    api/
        __init__.py
        router.py                        # FastAPI router (28+ endpoints)
        analytics_routes.py              # Analytics query and dashboard endpoints
        root_cause_routes.py             # Root cause analysis endpoints
        mediation_routes.py              # Mediation workflow endpoints
        effectiveness_routes.py          # Remediation effectiveness endpoints
        risk_routes.py                   # Grievance risk scoring endpoints
        collective_routes.py             # Collective grievance endpoints
        report_routes.py                 # Regulatory report generation endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Grievance Pattern Types
class PatternType(str, Enum):
    GEOGRAPHIC_CLUSTER = "geographic_cluster"
    CATEGORY_SPIKE = "category_spike"
    RECURRENCE = "recurrence"
    ESCALATION_TREND = "escalation_trend"
    SEASONAL = "seasonal"
    CROSS_SUPPLY_CHAIN = "cross_supply_chain"

# Grievance Pattern
class GrievancePattern(BaseModel):
    pattern_id: str                      # UUID
    pattern_type: PatternType
    description: str
    affected_grievance_ids: List[str]
    geographic_scope: Optional[Dict[str, Any]]  # country, region, coordinates
    commodity_scope: List[str]
    category_scope: List[str]
    time_range_start: datetime
    time_range_end: datetime
    statistical_significance: float       # p-value or confidence
    severity_assessment: str              # critical/high/medium/low
    recommended_actions: List[str]
    provenance_hash: str
    detected_at: datetime

# Root Cause Analysis
class RootCauseClassification(str, Enum):
    SYSTEMIC = "systemic"
    OPERATIONAL = "operational"
    ENVIRONMENTAL = "environmental"
    SUPPLIER_SPECIFIC = "supplier_specific"
    REGULATORY = "regulatory"
    EXTERNAL = "external"

class RootCauseAnalysis(BaseModel):
    analysis_id: str                      # UUID
    methodology: str                      # five_whys, ishikawa, cross_case_correlation
    linked_grievance_ids: List[str]
    linked_pattern_ids: List[str]
    root_cause_description: str
    root_cause_classification: RootCauseClassification
    causal_chain: List[Dict[str, Any]]    # Ordered cause-effect links
    ishikawa_categories: Optional[Dict[str, List[str]]]  # 6P categories
    confidence_score: float               # 0-100
    recommended_interventions: List[Dict[str, Any]]
    analyst_id: str
    evidence_ids: List[str]
    provenance_hash: str
    created_at: datetime

# Mediation Status
class MediationStatus(str, Enum):
    CASE_CREATED = "case_created"
    PARTIES_REGISTERED = "parties_registered"
    MEDIATOR_ASSIGNED = "mediator_assigned"
    MEDIATOR_ACCEPTED = "mediator_accepted"
    PRE_MEDIATION = "pre_mediation"
    IN_PROGRESS = "in_progress"
    SETTLEMENT_DRAFTED = "settlement_drafted"
    SETTLEMENT_SIGNED = "settlement_signed"
    IMPLEMENTATION_MONITORING = "implementation_monitoring"
    SETTLEMENT_VERIFIED = "settlement_verified"
    IMPASSE = "impasse"
    CLOSED = "closed"

class MediationCase(BaseModel):
    mediation_id: str                     # UUID
    linked_grievance_ids: List[str]
    linked_collective_case_id: Optional[str]
    dispute_scope: str
    parties: List[Dict[str, Any]]         # party_id, name, type, role, legal_standing
    mediator_id: Optional[str]
    current_status: MediationStatus
    status_history: List[Dict[str, Any]]
    sessions: List[Dict[str, Any]]
    issues: List[Dict[str, Any]]          # issue, status, party_positions
    settlement_terms: Optional[Dict[str, Any]]
    settlement_signed_at: Optional[datetime]
    implementation_milestones: List[Dict[str, Any]]
    outcome_classification: Optional[str]
    provenance_hash: str
    created_at: datetime
    updated_at: datetime

# Remediation Effectiveness
class EffectivenessClassification(str, Enum):
    HIGHLY_EFFECTIVE = "highly_effective"  # 85-100
    EFFECTIVE = "effective"               # 70-84
    PARTIALLY_EFFECTIVE = "partially_effective"  # 50-69
    INEFFECTIVE = "ineffective"           # 25-49
    FAILED = "failed"                     # 0-24

class RemediationProgram(BaseModel):
    program_id: str                       # UUID
    linked_grievance_id: str
    remediation_type: str                 # environmental, compensation, process, policy, infrastructure
    intended_outcome: str
    success_criteria: List[Dict[str, Any]]
    monitoring_duration_months: int        # 6, 12, 24, or 36
    monitoring_frequency: str             # monthly, quarterly, semi_annual
    leading_indicators: List[Dict[str, Any]]
    lagging_indicators: List[Dict[str, Any]]
    satisfaction_assessments: List[Dict[str, Any]]
    satellite_verification: Optional[Dict[str, Any]]
    effectiveness_score: Optional[float]   # 0-100
    effectiveness_classification: Optional[EffectivenessClassification]
    responsible_party: str
    provenance_hash: str
    created_at: datetime
    updated_at: datetime

# Grievance Risk Score
class GrievanceRiskLevel(str, Enum):
    CRITICAL = "critical"                 # 85-100
    HIGH = "high"                         # 70-84
    ELEVATED = "elevated"                 # 50-69
    MODERATE = "moderate"                 # 25-49
    LOW = "low"                           # 0-24

class GrievanceRiskScore(BaseModel):
    score_id: str                         # UUID
    grievance_id: str
    severity_score: float
    category_score: float
    geographic_score: float
    supplier_score: float
    recurrence_score: float
    stakeholder_influence_score: float
    media_ngo_score: float
    regulatory_sensitivity_score: float
    composite_score: float                # 0-100
    risk_level: GrievanceRiskLevel
    escalation_probability: float         # 0-100%
    dimension_weights: Dict[str, float]
    provenance_hash: str
    scored_at: datetime

# Collective Grievance
class CollectiveGrievance(BaseModel):
    collective_id: str                    # UUID
    linked_grievance_ids: List[str]
    representative_complainants: List[Dict[str, Any]]
    affected_population: Dict[str, Any]   # boundary, count, demographics
    harm_assessment: Dict[str, Any]       # 5-domain assessment
    collective_impact_score: float        # 0-100
    remediation_plan: Optional[Dict[str, Any]]
    settlement: Optional[Dict[str, Any]]
    benefit_distribution: List[Dict[str, Any]]
    is_indigenous_collective: bool
    territory_id: Optional[str]
    supply_chain_node_ids: List[str]
    current_status: str
    provenance_hash: str
    created_at: datetime
    updated_at: datetime
```

### 7.4 Database Schema (New Migration: V120)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_grievance_manager;

-- Grievance patterns detected by analytics engine
CREATE TABLE eudr_grievance_manager.grievance_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_type VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    affected_grievance_ids JSONB NOT NULL DEFAULT '[]',
    geographic_scope JSONB,
    commodity_scope JSONB DEFAULT '[]',
    category_scope JSONB DEFAULT '[]',
    time_range_start TIMESTAMPTZ NOT NULL,
    time_range_end TIMESTAMPTZ NOT NULL,
    statistical_significance NUMERIC(8,6),
    severity_assessment VARCHAR(20) NOT NULL,
    recommended_actions JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Root cause analyses
CREATE TABLE eudr_grievance_manager.root_cause_analyses (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    methodology VARCHAR(50) NOT NULL,
    linked_grievance_ids JSONB NOT NULL DEFAULT '[]',
    linked_pattern_ids JSONB DEFAULT '[]',
    root_cause_description TEXT NOT NULL,
    root_cause_classification VARCHAR(50) NOT NULL,
    causal_chain JSONB NOT NULL DEFAULT '[]',
    ishikawa_categories JSONB,
    five_whys_tree JSONB,
    confidence_score NUMERIC(5,2) DEFAULT 0.0,
    recommended_interventions JSONB DEFAULT '[]',
    analyst_id VARCHAR(100) NOT NULL,
    evidence_ids JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Mediation cases
CREATE TABLE eudr_grievance_manager.mediation_cases (
    mediation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    linked_grievance_ids JSONB NOT NULL DEFAULT '[]',
    linked_collective_case_id UUID,
    dispute_scope TEXT NOT NULL,
    parties JSONB NOT NULL DEFAULT '[]',
    mediator_id UUID,
    current_status VARCHAR(50) NOT NULL DEFAULT 'case_created',
    issues JSONB DEFAULT '[]',
    settlement_terms JSONB,
    settlement_signed_at TIMESTAMPTZ,
    implementation_milestones JSONB DEFAULT '[]',
    outcome_classification VARCHAR(50),
    is_confidential BOOLEAN DEFAULT TRUE,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Mediation sessions
CREATE TABLE eudr_grievance_manager.mediation_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mediation_id UUID NOT NULL REFERENCES eudr_grievance_manager.mediation_cases(mediation_id),
    session_number INTEGER NOT NULL,
    scheduled_date TIMESTAMPTZ NOT NULL,
    location VARCHAR(500),
    is_virtual BOOLEAN DEFAULT FALSE,
    agenda JSONB DEFAULT '[]',
    attending_parties JSONB DEFAULT '[]',
    minutes TEXT,
    action_items JSONB DEFAULT '[]',
    session_status VARCHAR(30) DEFAULT 'scheduled',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Mediation status transitions (hypertable)
CREATE TABLE eudr_grievance_manager.mediation_status_transitions (
    transition_id UUID DEFAULT gen_random_uuid(),
    mediation_id UUID NOT NULL,
    from_status VARCHAR(50),
    to_status VARCHAR(50) NOT NULL,
    transitioned_by VARCHAR(100) NOT NULL,
    notes TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    transitioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_grievance_manager.mediation_status_transitions', 'transitioned_at');

-- Mediator registry
CREATE TABLE eudr_grievance_manager.mediator_registry (
    mediator_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    organization VARCHAR(500),
    qualifications JSONB DEFAULT '[]',
    specializations JSONB DEFAULT '[]',
    languages JSONB DEFAULT '[]',
    country_code CHAR(2),
    is_active BOOLEAN DEFAULT TRUE,
    conflict_of_interest_declarations JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Remediation effectiveness programs
CREATE TABLE eudr_grievance_manager.remediation_programs (
    program_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    linked_grievance_id UUID NOT NULL,
    remediation_type VARCHAR(50) NOT NULL,
    intended_outcome TEXT NOT NULL,
    success_criteria JSONB NOT NULL DEFAULT '[]',
    monitoring_duration_months INTEGER NOT NULL DEFAULT 12,
    monitoring_frequency VARCHAR(30) DEFAULT 'quarterly',
    responsible_party VARCHAR(200) NOT NULL,
    effectiveness_score NUMERIC(5,2),
    effectiveness_classification VARCHAR(30),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Remediation effectiveness measurements (hypertable)
CREATE TABLE eudr_grievance_manager.effectiveness_measurements (
    measurement_id UUID DEFAULT gen_random_uuid(),
    program_id UUID NOT NULL,
    measurement_type VARCHAR(30) NOT NULL,  -- leading, lagging, satisfaction, satellite
    metric_name VARCHAR(200) NOT NULL,
    metric_value NUMERIC(18,4),
    metric_unit VARCHAR(50),
    target_value NUMERIC(18,4),
    evidence_ids JSONB DEFAULT '[]',
    notes TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_grievance_manager.effectiveness_measurements', 'measured_at');

-- Grievance risk scores (hypertable)
CREATE TABLE eudr_grievance_manager.grievance_risk_scores (
    score_id UUID DEFAULT gen_random_uuid(),
    grievance_id UUID NOT NULL,
    severity_score NUMERIC(5,2) DEFAULT 0.0,
    category_score NUMERIC(5,2) DEFAULT 0.0,
    geographic_score NUMERIC(5,2) DEFAULT 0.0,
    supplier_score NUMERIC(5,2) DEFAULT 0.0,
    recurrence_score NUMERIC(5,2) DEFAULT 0.0,
    stakeholder_influence_score NUMERIC(5,2) DEFAULT 0.0,
    media_ngo_score NUMERIC(5,2) DEFAULT 0.0,
    regulatory_sensitivity_score NUMERIC(5,2) DEFAULT 0.0,
    composite_score NUMERIC(5,2) DEFAULT 0.0,
    risk_level VARCHAR(20) NOT NULL,
    escalation_probability NUMERIC(5,2) DEFAULT 0.0,
    dimension_weights JSONB NOT NULL DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_grievance_manager.grievance_risk_scores', 'scored_at');

-- Collective grievances
CREATE TABLE eudr_grievance_manager.collective_grievances (
    collective_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    linked_grievance_ids JSONB NOT NULL DEFAULT '[]',
    representative_complainants JSONB NOT NULL DEFAULT '[]',
    affected_population JSONB NOT NULL DEFAULT '{}',
    harm_assessment JSONB DEFAULT '{}',
    collective_impact_score NUMERIC(5,2),
    remediation_plan JSONB,
    settlement JSONB,
    benefit_distribution JSONB DEFAULT '[]',
    is_indigenous_collective BOOLEAN DEFAULT FALSE,
    territory_id UUID,
    supply_chain_node_ids JSONB DEFAULT '[]',
    current_status VARCHAR(50) NOT NULL DEFAULT 'created',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Regulatory reports generated (hypertable)
CREATE TABLE eudr_grievance_manager.regulatory_reports (
    report_id UUID DEFAULT gen_random_uuid(),
    report_type VARCHAR(100) NOT NULL,
    regulatory_framework VARCHAR(50) NOT NULL,
    operator_id UUID NOT NULL,
    commodity VARCHAR(50),
    country_code CHAR(2),
    date_range_start TIMESTAMPTZ,
    date_range_end TIMESTAMPTZ,
    format VARCHAR(10) NOT NULL DEFAULT 'json',
    file_path VARCHAR(1000),
    file_size_bytes BIGINT,
    provenance_hash VARCHAR(64) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_grievance_manager.regulatory_reports', 'generated_at');

-- Indexes
CREATE INDEX idx_patterns_type ON eudr_grievance_manager.grievance_patterns(pattern_type);
CREATE INDEX idx_patterns_severity ON eudr_grievance_manager.grievance_patterns(severity_assessment);
CREATE INDEX idx_patterns_active ON eudr_grievance_manager.grievance_patterns(is_active);
CREATE INDEX idx_rca_classification ON eudr_grievance_manager.root_cause_analyses(root_cause_classification);
CREATE INDEX idx_rca_methodology ON eudr_grievance_manager.root_cause_analyses(methodology);
CREATE INDEX idx_mediation_status ON eudr_grievance_manager.mediation_cases(current_status);
CREATE INDEX idx_mediation_mediator ON eudr_grievance_manager.mediation_cases(mediator_id);
CREATE INDEX idx_sessions_mediation ON eudr_grievance_manager.mediation_sessions(mediation_id);
CREATE INDEX idx_sessions_date ON eudr_grievance_manager.mediation_sessions(scheduled_date);
CREATE INDEX idx_mediators_active ON eudr_grievance_manager.mediator_registry(is_active);
CREATE INDEX idx_programs_grievance ON eudr_grievance_manager.remediation_programs(linked_grievance_id);
CREATE INDEX idx_programs_type ON eudr_grievance_manager.remediation_programs(remediation_type);
CREATE INDEX idx_programs_classification ON eudr_grievance_manager.remediation_programs(effectiveness_classification);
CREATE INDEX idx_collective_status ON eudr_grievance_manager.collective_grievances(current_status);
CREATE INDEX idx_collective_indigenous ON eudr_grievance_manager.collective_grievances(is_indigenous_collective);
CREATE INDEX idx_reports_type ON eudr_grievance_manager.regulatory_reports(report_type);
CREATE INDEX idx_reports_framework ON eudr_grievance_manager.regulatory_reports(regulatory_framework);
CREATE INDEX idx_reports_operator ON eudr_grievance_manager.regulatory_reports(operator_id);
```

### 7.5 API Endpoints (28+)

| Method | Path | Description |
|--------|------|-------------|
| **Grievance Analytics** | | |
| GET | `/v1/analytics/patterns` | List detected grievance patterns (with filters: type, severity, commodity, date range) |
| GET | `/v1/analytics/patterns/{pattern_id}` | Get pattern details with linked grievances |
| POST | `/v1/analytics/analyze` | Trigger full pattern analysis on grievance dataset |
| GET | `/v1/analytics/trends` | Get time-series trend data (with filters: category, geography, commodity) |
| GET | `/v1/analytics/clusters` | Get geographic cluster data for hotspot visualization |
| GET | `/v1/analytics/dashboard` | Get executive dashboard KPIs |
| **Root Cause Analysis** | | |
| POST | `/v1/root-cause/five-whys` | Start a 5-Whys analysis session for a set of grievances |
| POST | `/v1/root-cause/ishikawa` | Generate Ishikawa diagram for a set of grievances |
| POST | `/v1/root-cause/correlate` | Run cross-case correlation analysis |
| GET | `/v1/root-cause/analyses` | List root cause analyses (with filters: classification, methodology) |
| GET | `/v1/root-cause/analyses/{analysis_id}` | Get root cause analysis details with causal chain |
| **Multi-Party Mediation** | | |
| POST | `/v1/mediation/cases` | Create a new mediation case |
| GET | `/v1/mediation/cases` | List mediation cases (with filters: status, mediator) |
| GET | `/v1/mediation/cases/{mediation_id}` | Get mediation case details |
| POST | `/v1/mediation/cases/{mediation_id}/parties` | Add party to mediation case |
| POST | `/v1/mediation/cases/{mediation_id}/assign-mediator` | Assign mediator |
| POST | `/v1/mediation/cases/{mediation_id}/sessions` | Create mediation session |
| PUT | `/v1/mediation/cases/{mediation_id}/settlement` | Record settlement terms |
| PUT | `/v1/mediation/cases/{mediation_id}/status` | Update mediation status |
| GET | `/v1/mediation/mediators` | List available mediators (with filters: specialization, language) |
| **Remediation Effectiveness** | | |
| POST | `/v1/effectiveness/programs` | Create remediation monitoring program |
| GET | `/v1/effectiveness/programs` | List programs (with filters: type, classification) |
| GET | `/v1/effectiveness/programs/{program_id}` | Get program details with scorecard |
| POST | `/v1/effectiveness/programs/{program_id}/measurements` | Record effectiveness measurement |
| GET | `/v1/effectiveness/dashboard` | Get portfolio effectiveness dashboard |
| **Grievance Risk Scoring** | | |
| POST | `/v1/risk/score/{grievance_id}` | Calculate risk score for a grievance |
| POST | `/v1/risk/portfolio` | Calculate portfolio risk scores |
| GET | `/v1/risk/rankings` | Get risk-ranked grievance list |
| **Collective Grievances** | | |
| POST | `/v1/collective/cases` | Create collective grievance case |
| GET | `/v1/collective/cases` | List collective cases (with filters: status, indigenous) |
| GET | `/v1/collective/cases/{collective_id}` | Get collective case details |
| PUT | `/v1/collective/cases/{collective_id}/harm-assessment` | Record harm assessment |
| PUT | `/v1/collective/cases/{collective_id}/settlement` | Record collective settlement |
| **Regulatory Reports** | | |
| POST | `/v1/reports/generate` | Generate regulatory report (specify type, framework, filters) |
| GET | `/v1/reports` | List generated reports |
| GET | `/v1/reports/{report_id}` | Download a generated report |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (15)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_gmm_patterns_detected_total` | Counter | Patterns detected by type |
| 2 | `gl_eudr_gmm_root_cause_analyses_total` | Counter | Root cause analyses completed by methodology |
| 3 | `gl_eudr_gmm_mediation_cases_created_total` | Counter | Mediation cases created |
| 4 | `gl_eudr_gmm_mediation_settlements_total` | Counter | Mediation settlements reached |
| 5 | `gl_eudr_gmm_remediation_programs_total` | Counter | Remediation programs created by type |
| 6 | `gl_eudr_gmm_effectiveness_measurements_total` | Counter | Effectiveness measurements recorded |
| 7 | `gl_eudr_gmm_risk_scores_calculated_total` | Counter | Grievance risk scores calculated |
| 8 | `gl_eudr_gmm_collective_cases_total` | Counter | Collective grievance cases created |
| 9 | `gl_eudr_gmm_regulatory_reports_total` | Counter | Regulatory reports generated by type and framework |
| 10 | `gl_eudr_gmm_processing_duration_seconds` | Histogram | Processing latency by operation type |
| 11 | `gl_eudr_gmm_analytics_query_duration_seconds` | Histogram | Analytics query latency |
| 12 | `gl_eudr_gmm_errors_total` | Counter | Errors by operation type |
| 13 | `gl_eudr_gmm_active_mediation_cases` | Gauge | Currently active mediation cases |
| 14 | `gl_eudr_gmm_active_remediation_programs` | Gauge | Currently active remediation monitoring programs |
| 15 | `gl_eudr_gmm_portfolio_risk_score` | Gauge | Current portfolio-level grievance risk score |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for analytics and audit trail |
| Analytics | NumPy, SciPy, scikit-learn (DBSCAN only) | Deterministic statistical analysis and geographic clustering |
| Cache | Redis | Analytics result caching, dashboard data caching |
| Object Storage | S3 | Generated reports, mediation documents, evidence files |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access to grievance management data |
| Encryption | AES-256-GCM via SEC-003 | Mediation data and sensitive grievance data encryption at rest |
| Monitoring | Prometheus + Grafana | 15 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-gmm:analytics:read` | View grievance analytics, patterns, and dashboards | Viewer, Analyst, Grievance Manager, Compliance Officer, Admin |
| `eudr-gmm:analytics:execute` | Trigger pattern analysis runs | Analyst, Grievance Manager, Compliance Officer, Admin |
| `eudr-gmm:root-cause:read` | View root cause analyses | Viewer, Analyst, Grievance Manager, Compliance Officer, Admin |
| `eudr-gmm:root-cause:write` | Create and manage root cause analyses | Analyst, Grievance Manager, Admin |
| `eudr-gmm:mediation:read` | View mediation cases (respecting confidentiality) | Grievance Manager, Mediator, Compliance Officer, Admin |
| `eudr-gmm:mediation:write` | Create and manage mediation cases | Grievance Manager, Admin |
| `eudr-gmm:mediation:mediate` | Conduct mediation sessions and propose settlements | Mediator, Admin |
| `eudr-gmm:effectiveness:read` | View remediation effectiveness data | Viewer, Analyst, Remediation Coordinator, Grievance Manager, Compliance Officer, Admin |
| `eudr-gmm:effectiveness:write` | Create programs and record measurements | Remediation Coordinator, Grievance Manager, Admin |
| `eudr-gmm:risk:read` | View grievance risk scores and rankings | Viewer, Analyst, Grievance Manager, Compliance Officer, Admin |
| `eudr-gmm:risk:execute` | Trigger risk score calculations | Analyst, Grievance Manager, Compliance Officer, Admin |
| `eudr-gmm:collective:read` | View collective grievance cases | Viewer, Analyst, Grievance Manager, Compliance Officer, Admin |
| `eudr-gmm:collective:write` | Create and manage collective grievance cases | Grievance Manager, Admin |
| `eudr-gmm:reports:read` | View and download generated regulatory reports | Viewer, Analyst, Grievance Manager, Compliance Officer, Auditor, Admin |
| `eudr-gmm:reports:generate` | Generate regulatory reports | Grievance Manager, Compliance Officer, Admin |
| `eudr-gmm:mediators:manage` | Manage mediator registry | Grievance Manager, Admin |
| `eudr-gmm:audit:read` | View audit trail and provenance data | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| AGENT-EUDR-031 Stakeholder Engagement Tool | Grievance tables, stakeholder registry | Grievance records, status transitions, stakeholder data -> analytics input |
| AGENT-EUDR-001 Supply Chain Mapping Master | Graph engine, node data | Supply chain context for grievance segmentation and pattern analysis |
| AGENT-EUDR-016 Country Risk Evaluator | Country risk scores | Geographic risk dimension for grievance risk scoring |
| AGENT-EUDR-017 Supplier Risk Scorer | Supplier risk scores | Supplier risk dimension for grievance risk scoring |
| AGENT-EUDR-018 Commodity Risk Analyzer | Commodity risk scores | Commodity context for pattern analysis |
| AGENT-EUDR-021 Indigenous Rights Checker | Territory data, collective rights | Indigenous collective grievance context |
| AGENT-EUDR-003 Satellite Monitoring Agent | Satellite imagery | Environmental remediation verification |
| AGENT-EUDR-004 Forest Cover Analysis Agent | NDVI calculations | Forest restoration effectiveness measurement |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-028 Risk Assessment Engine | Grievance patterns and risk scores | Pattern intelligence -> risk assessment input per Article 10(2)(e) |
| AGENT-EUDR-029 Mitigation Measure Designer | Root cause findings, effectiveness data | Root causes -> systemic interventions; effectiveness data -> mitigation refinement |
| AGENT-EUDR-030 Documentation Generator | Regulatory reports | Grievance mechanism reports -> DDS package inclusion |
| GL-EUDR-APP v1.0 | API integration | Analytics dashboards, mediation workspace, effectiveness scorecards -> frontend |
| External Auditors | Read-only API + report downloads | Regulatory reports, effectiveness data, UNGP assessment -> third-party verification |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Grievance Pattern Detection and Root Cause Analysis (Grievance Manager)

```
1. Helena logs in to GL-EUDR-APP -> Grievance Management module
2. Dashboard shows alert: "New pattern detected: 12 land rights grievances
   clustered in West Kalimantan, Indonesia (3-month window)"
3. Helena clicks pattern -> sees geographic cluster map with 12 grievance pins
4. All 12 linked to different cooperatives but same palm oil supply chain segment
5. Helena initiates Root Cause Analysis -> selects "Cross-Case Correlation"
6. System identifies common factor: all 12 cooperatives share same sub-tier
   supplier who expanded plantations into community forest land
7. Helena initiates "5-Whys" on the root cause:
   Why 1: Plantation expanded into community land -> No boundary verification
   Why 2: No boundary verification -> Supplier lacked geolocation validation
   Why 3: Lacked validation -> Not required by operator purchasing terms
   Why 4: Not required -> Procurement policy gap on land rights
   Root Cause: Systemic procurement policy gap on land rights verification
8. System generates intervention recommendation: "Update procurement policy
   to require plot boundary verification against community land registries"
9. Helena exports root cause report -> feeds into EUDR-029 mitigation planning
10. Root cause finding feeds into EUDR-028 risk assessment for supplier portfolio
```

#### Flow 2: Multi-Party Mediation (Mediator)

```
1. Mediation case created from EUDR-031 grievance: indigenous community
   alleges rubber plantation expanded into ancestral territory
2. 4 parties registered: Adat community, rubber cooperative, local government
   land agency, EU tire manufacturer (operator)
3. Dr. Laurent assigned as neutral mediator; all parties accept
4. Pre-mediation: each party submits position document via platform
5. Session 1: parties present positions; mediator documents issues list:
   (a) boundary dispute, (b) compensation for affected area,
   (c) future access rights, (d) environmental restoration
6. Session 2: boundary evidence reviewed; partial agreement on (a)
7. Session 3: compensation terms negotiated; agreement on (b) and (c)
8. Session 4: environmental restoration plan agreed; full settlement reached
9. Settlement document generated with all terms; all parties sign electronically
10. Post-settlement monitoring: milestones tracked over 24 months
11. System verifies environmental restoration via satellite imagery integration
```

#### Flow 3: Regulatory Inspection Response (Compliance Officer)

```
1. Erik receives competent authority inspection notice under EUDR Article 16
2. Authority requests: grievance mechanism documentation, pattern analysis,
   evidence of mechanism effectiveness, remediation outcomes
3. Erik navigates to Regulatory Reports -> selects "EUDR Article 16 Inspection Report"
4. Configures: date range (last 12 months), commodity (rubber), country (Thailand)
5. System generates report in < 60 seconds with:
   - Grievance case inventory: 47 cases, 38 resolved, 6 in progress, 3 in mediation
   - Pattern analysis: 2 patterns detected and addressed (root cause reports attached)
   - Mechanism effectiveness: UNGP Principle 31 score 82/100 (Good)
   - Remediation outcomes: 12 programs, 8 rated Effective or Highly Effective
   - Feedback loop evidence: 3 risk assessment updates informed by grievance data
6. Report includes SHA-256 provenance hash and evidence cross-references
7. Erik downloads PDF and JSON versions; provides to competent authority
8. Also generates CSDDD Article 8 report for upcoming CSDDD preparation
```

### 8.2 Key Screen Descriptions

**Grievance Analytics Dashboard:**
- Top row: KPI cards (active grievances, patterns detected, resolution rate, avg resolution time, portfolio risk score, remediation effectiveness avg)
- Map panel: geographic heatmap of grievance clusters with zoom and filter
- Trend chart: time-series of grievance volumes by category with anomaly highlighting
- Pattern list: active patterns sorted by severity with drill-down to linked grievances
- Risk distribution: chart showing grievance count by risk level

**Root Cause Analysis Workspace:**
- Split view: left panel shows linked grievances; right panel shows analysis canvas
- 5-Whys view: interactive tree with expandable why-levels and evidence attachments
- Ishikawa view: fishbone diagram with 6P categories and draggable cause entries
- Correlation matrix: heatmap showing cross-case factor correlation strength
- Intervention panel: recommended systemic actions with effort/impact scoring

**Mediation Case View:**
- Case header: parties, mediator, status, timeline
- Issues tracker: list of dispute issues with status badges (open/progressing/agreed/impasse)
- Session timeline: chronological session list with minutes and action items
- Position comparison: side-by-side party position documents per issue
- Settlement builder: term-by-term settlement drafting with party approval workflow

**Remediation Effectiveness Dashboard:**
- Portfolio cards: programs by classification (Highly Effective / Effective / Partial / Ineffective / Failed)
- Scorecard view: per-program visual with leading/lagging indicators, satisfaction trend, satellite verification
- Timeline: milestone tracking with evidence markers and status colors
- Comparison: multi-program effectiveness comparison chart

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 7 P0 features (Features 1-7) implemented and tested
  - [ ] Feature 1: Grievance Analytics Engine -- pattern detection, trends, clusters, dashboards
  - [ ] Feature 2: Root Cause Analysis Engine -- 5-Whys, Ishikawa, cross-case correlation
  - [ ] Feature 3: Multi-Party Mediation Manager -- party management, sessions, settlement, monitoring
  - [ ] Feature 4: Remediation Effectiveness Tracker -- longitudinal monitoring, scoring, satellite verification
  - [ ] Feature 5: Grievance Risk Scoring Engine -- 8-dimension scoring, portfolio risk, rankings
  - [ ] Feature 6: Collective Grievance Handler -- population mapping, harm assessment, collective remedy
  - [ ] Feature 7: Regulatory Reporter -- 6 report types across EUDR/CSDDD/UNGP/OECD/ILO frameworks
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC + AES-256 for mediation data integrated)
- [ ] Performance targets met (analytics < 30s for 10K records; risk scoring < 500ms; reports < 60s)
- [ ] All analytics verified deterministic (bit-perfect reproducibility tests passing)
- [ ] Integration with EUDR-031 grievance data verified
- [ ] Integration with EUDR-028/029 feedback loop verified
- [ ] Integration with EUDR-003/004 satellite verification verified
- [ ] API documentation complete (OpenAPI spec)
- [ ] Database migration V120 tested and validated
- [ ] 3 beta customers successfully deployed and validated analytics outputs
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 30+ operators activated grievance analytics
- 50+ patterns detected across customer portfolio
- 10+ root cause analyses completed
- 5+ mediation cases initiated
- < 5 support tickets per customer
- p99 analytics query latency < 2 seconds in production

**60 Days:**
- 100+ operators using analytics dashboards
- 100+ root cause analyses completed
- 20+ mediation settlements reached
- 50+ remediation programs under effectiveness monitoring
- Regulatory reports generated for 3+ frameworks
- Grievance pattern intelligence feeding EUDR-028 risk assessment for 50+ operators

**90 Days:**
- 200+ operators active on EUDR-032
- 500+ grievance patterns detected and addressed
- 70%+ mediation resolution rate achieved
- First 12-month remediation effectiveness reviews completed
- Zero competent authority inspection failures attributable to grievance mechanism gaps
- Customer NPS > 50 from grievance manager persona

---

## 10. Timeline and Milestones

### Phase 1: Analytics and Intelligence Core (Weeks 1-7)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Analytics Engine (Feature 1): time-series trends, DBSCAN clustering, pattern detection | Senior Backend + Data Engineer |
| 3-4 | Analytics Engine: dashboard API, recurrence analysis, cross-supply-chain correlation | Senior Backend + Data Engineer |
| 5-6 | Root Cause Analysis Engine (Feature 2): 5-Whys workflow, Ishikawa generation | Senior Backend Engineer |
| 6-7 | Root Cause Analysis Engine: cross-case correlation, intervention recommendations, reporting | Senior Backend Engineer |

**Milestone: Analytics and root cause engines operational with EUDR-031 integration (Week 7)**

### Phase 2: Advanced Dispute and Remediation Management (Weeks 8-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 8-9 | Multi-Party Mediation Manager (Feature 3): case management, party registration, mediator assignment | Senior Backend + Frontend Engineer |
| 10-11 | Mediation Manager: session management, settlement tracking, compliance monitoring | Senior Backend + Frontend Engineer |
| 12-13 | Remediation Effectiveness Tracker (Feature 4): program creation, metrics definition, measurement recording | Senior Backend + Data Engineer |
| 13-14 | Effectiveness Tracker: scoring, satellite integration, portfolio dashboard | Senior Backend + Data Engineer |

**Milestone: Mediation and effectiveness tracking fully operational (Week 14)**

### Phase 3: Predictive, Collective, and Regulatory (Weeks 15-20)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Grievance Risk Scoring Engine (Feature 5): 8-dimension scoring, portfolio risk, alerts | Senior Backend Engineer |
| 17-18 | Collective Grievance Handler (Feature 6): population mapping, harm assessment, collective settlement | Senior Backend Engineer + Domain Expert |
| 19-20 | Regulatory Reporter (Feature 7): 6 report types, multi-language, scheduling, EUDR-030 integration | Senior Backend + Report Engineer |

**Milestone: All 7 P0 features implemented (Week 20)**

### Phase 4: Testing and Launch (Weeks 21-24)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 21-22 | Complete test suite: 800+ tests, determinism verification, integration tests | Test Engineer |
| 22-23 | Performance testing, security audit, load testing | DevOps + Security |
| 23 | Database migration V120 finalized and tested | DevOps |
| 23-24 | Beta customer onboarding (3 customers), launch readiness review | Product + Engineering |
| 24 | Production launch | All |

**Milestone: Production launch with all 7 P0 features (Week 24)**

### Phase 5: Enhancements (Weeks 25-30)

- Grievance benchmarking service (Feature 8)
- Stakeholder trust index (Feature 9)
- Grievance-informed supply chain optimization (Feature 10)
- Advanced predictive modeling with historical backtesting
- Additional regulatory framework templates (LkSG, French Duty of Vigilance)

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-031 Stakeholder Engagement Tool | BUILT (100%) | Low | Stable; provides grievance operational data |
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable; provides supply chain graph context |
| AGENT-EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Stable; provides geographic risk scores |
| AGENT-EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Stable; provides supplier risk scores |
| AGENT-EUDR-018 Commodity Risk Analyzer | BUILT (100%) | Low | Stable; provides commodity risk context |
| AGENT-EUDR-021 Indigenous Rights Checker | BUILT (100%) | Low | Stable; provides territory and collective rights data |
| AGENT-EUDR-025 Risk Mitigation Advisor | BUILT (100%) | Low | Stable; provides remediation plan context |
| AGENT-EUDR-028 Risk Assessment Engine | BUILT (100%) | Low | Stable; consumes grievance intelligence output |
| AGENT-EUDR-029 Mitigation Measure Designer | BUILT (100%) | Low | Stable; consumes root cause findings |
| AGENT-EUDR-030 Documentation Generator | BUILT (100%) | Low | Stable; consumes regulatory reports for DDS |
| AGENT-EUDR-003 Satellite Monitoring Agent | BUILT (100%) | Low | Stable; provides satellite imagery for remediation verification |
| AGENT-EUDR-004 Forest Cover Analysis Agent | BUILT (100%) | Low | Stable; provides NDVI calculations |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| SEC-003 Encryption at Rest | BUILT (100%) | Low | Standard encryption for mediation data |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EUDR Articles 14-16 inspection format specifications | Evolving | Medium | Flexible report templates; adapter pattern for format changes |
| CSDDD Article 8 reporting specifications | Draft (enforcement 2027) | Medium | Early implementation based on directive text; update when implementing acts published |
| UNGP Principle 31 effectiveness criteria | Stable | Low | Well-established criteria; implemented per published guidance |
| OECD Guidelines Chapter IV | Stable | Low | Well-established guidelines |
| ILO Convention 169 indigenous rights requirements | Stable | Low | Long-standing convention |
| scikit-learn DBSCAN implementation | Stable (v1.x) | Low | Widely used; deterministic with fixed random state |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EUDR competent authority inspection format not yet standardized | Medium | High | Flexible report templates with configurable sections; adapter pattern; update templates as standards emerge |
| R2 | Insufficient grievance data volume for meaningful pattern detection in early deployments | High | Medium | Minimum sample size thresholds with clear warnings; analytics degrade gracefully; start with simpler trend analysis before requiring cluster detection |
| R3 | Mediation data confidentiality breach | Low | Critical | AES-256 encryption at rest; mediation data in separate schema with restricted RBAC; no cross-case data leakage; confidentiality boundary enforced in code |
| R4 | CSDDD Article 8 implementing acts change reporting requirements | Medium | Medium | Modular report templates; regulatory framework abstraction layer; hot-reloadable template configuration |
| R5 | Remediation effectiveness tracking requires long monitoring periods (12-36 months) | High | Medium | Launch with 6-month initial assessments; long-term tracking as ongoing capability; stakeholder satisfaction as early effectiveness proxy |
| R6 | Satellite-based remediation verification accuracy limitations | Medium | Medium | Satellite verification as supporting evidence, not sole determinant; combine with ground-truth stakeholder assessments; confidence scoring on satellite results |
| R7 | Multi-party mediation complexity varies significantly by dispute type | Medium | Medium | Flexible mediation workflow; configurable stages; mediator discretion on process; edge case documentation |
| R8 | Cross-case correlation produces false positive root cause associations | Medium | Medium | Confidence scoring on all root cause findings; analyst review required before intervention recommendations; minimum correlation threshold configurable |
| R9 | Collective grievance population mapping accuracy for informal settlements | Medium | Medium | Support approximate population estimates with confidence ranges; field verification workflow; integration with EUDR-021 territory data as reference |
| R10 | Integration complexity with 12+ upstream EUDR agents | Medium | Medium | Well-defined interfaces; mock adapters for testing; circuit breaker pattern; graceful degradation when upstream data unavailable |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Analytics Engine Unit Tests | 120+ | Pattern detection, trend analysis, clustering, recurrence, dashboards |
| Root Cause Analysis Tests | 80+ | 5-Whys workflow, Ishikawa generation, cross-case correlation, confidence scoring |
| Mediation Manager Tests | 100+ | Party management, session lifecycle, settlement workflow, compliance monitoring, confidentiality |
| Effectiveness Tracker Tests | 80+ | Program lifecycle, measurement recording, scoring, satellite integration, portfolio dashboard |
| Risk Scoring Tests | 70+ | 8-dimension scoring, portfolio risk, determinism verification, threshold alerts |
| Collective Grievance Tests | 60+ | Population mapping, harm assessment, collective settlement, indigenous handling |
| Regulatory Reporter Tests | 80+ | All 6 report types, multi-format, multi-language, provenance verification |
| API Tests | 80+ | All 28+ endpoints, auth, error handling, pagination, rate limiting |
| Integration Tests | 40+ | Cross-agent integration with EUDR-031/028/029/003/004/016/017/021 |
| Determinism Tests | 30+ | Bit-perfect reproducibility for analytics, scoring, and reporting |
| Performance Tests | 20+ | 10K/50K/100K grievance datasets, concurrent queries, report generation timing |
| Golden Tests | 40+ | Predefined grievance datasets with known patterns, root causes, and expected analytics outputs |
| **Total** | **800+** | |

### 13.2 Golden Test Scenarios

Each golden test scenario uses a predefined grievance dataset with known patterns:

1. **Geographic Cluster**: 15 land rights grievances in 3 adjacent Indonesian districts -> expect DBSCAN cluster detection
2. **Category Spike**: Sudden increase in environmental grievances in Brazilian soy region -> expect trend anomaly detection
3. **Recurrence Pattern**: Same supplier generates 5 labor rights grievances over 18 months -> expect recurrence detection
4. **Cross-Supply-Chain**: Land rights issues in same region affecting both palm oil and rubber -> expect cross-commodity correlation
5. **Root Cause -- 5-Whys**: 8 environmental grievances tracing to single supplier's waste management -> expect correct 5-Whys decomposition
6. **Root Cause -- Ishikawa**: Multi-factor root cause with People + Process + Policy contributors -> expect correct 6P categorization
7. **Mediation Lifecycle**: 4-party dispute from case creation through settlement to compliance verification -> expect correct state transitions
8. **Remediation Effectiveness**: Environmental restoration program with 12-month monitoring data -> expect correct effectiveness scoring
9. **Risk Scoring**: Grievance with high severity, high-risk country, recurring supplier -> expect Critical risk classification
10. **Collective Grievance**: Community-wide deforestation impact affecting 500+ households -> expect correct population mapping and collective harm assessment

Total: 10 scenarios x 4 variations = 40 golden test cases

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **CSDDD** | EU Corporate Sustainability Due Diligence Directive |
| **UNGP** | UN Guiding Principles on Business and Human Rights |
| **ILO 169** | ILO Convention 169 on Indigenous and Tribal Peoples |
| **OECD Guidelines** | OECD Guidelines for Multinational Enterprises on Responsible Business Conduct |
| **FPIC** | Free, Prior and Informed Consent |
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise -- geographic clustering algorithm |
| **Ishikawa** | Also known as fishbone diagram; root cause analysis technique categorizing causes into branches |
| **5-Whys** | Iterative interrogative technique for exploring cause-and-effect relationships |
| **Collective Grievance** | A grievance representing harm to a group of stakeholders rather than an individual |
| **Remediation Effectiveness** | The degree to which a remediation action achieves its intended outcome over time |
| **Leading Indicator** | A metric that predicts future remediation success (e.g., activity completion) |
| **Lagging Indicator** | A metric that measures actual remediation outcome (e.g., forest canopy recovery) |
| **Grievance Risk Score** | Composite score predicting likelihood of grievance escalation, recurrence, or regulatory impact |
| **Mediation** | Structured dispute resolution process involving a neutral third party facilitating negotiation |

### Appendix B: EUDR-031 vs EUDR-032 Boundary Definition

| Capability | EUDR-031 (Operational) | EUDR-032 (Management) |
|------------|------------------------|------------------------|
| Grievance intake | Multi-channel intake, submission processing | Not handled (uses EUDR-031 data) |
| Triage | Severity/category/urgency classification | Risk scoring adds escalation probability |
| Investigation | Assign investigator, collect evidence, document findings | Root cause analysis adds systematic methodology |
| Resolution | Propose resolution, track implementation | Remediation effectiveness verifies long-term outcomes |
| Appeal | Appeal submission and review | Not handled (uses EUDR-031 appeal data) |
| Pattern detection | Not available | Full analytics engine with clustering and trends |
| Root cause analysis | Per-case findings only | Cross-case 5-Whys, Ishikawa, correlation |
| Multi-party mediation | Not available (out of scope) | Full mediation workflow with sessions and settlements |
| Collective grievances | Not available | Full collective case management with population mapping |
| Predictive risk scoring | Static severity classification | Dynamic 8-dimension predictive scoring |
| Regulatory reporting | General annual grievance report | 6 specialized regulatory framework reports |
| Remediation effectiveness | Tracks activity completion | Tracks long-term outcome effectiveness |

### Appendix C: Regulatory Article Cross-Reference Matrix

| Report Type | EUDR Art. 10 | EUDR Art. 11 | EUDR Art. 14-16 | EUDR Art. 29 | EUDR Art. 31 | CSDDD Art. 7 | CSDDD Art. 8 | CSDDD Art. 9 | UNGP P.31 | OECD Ch.IV | ILO 169 |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| EUDR Art. 16 Inspection | X | X | X | X | X | | | | X | | |
| CSDDD Art. 8 Compliance | | | | | X | X | X | X | X | | |
| UNGP P.31 Effectiveness | X | | | | | X | X | | X | | |
| OECD Performance | | X | | | | | | | X | X | |
| ILO 169 Indigenous | X | | X | X | X | | | | | | X |
| Cross-Regulatory | X | X | X | X | X | X | X | X | X | X | X |

### Appendix D: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EUDR)
2. Directive (EU) 2024/1760 of the European Parliament and of the Council (CSDDD)
3. UN Guiding Principles on Business and Human Rights (2011)
4. ILO Convention 169 on Indigenous and Tribal Peoples (1989)
5. OECD Guidelines for Multinational Enterprises on Responsible Business Conduct (2023 update)
6. UN Declaration on the Rights of Indigenous Peoples (UNDRIP, 2007)
7. ISO 37002:2021 -- Whistleblowing Management Systems
8. OHCHR Accountability and Remedy Project: Improving effectiveness of non-State-based grievance mechanisms
9. Shift Project: "Remediation, Grievance Mechanisms, and the Corporate Responsibility to Respect Human Rights" (2014)
10. Business & Human Rights Resource Centre: Corporate Human Rights Benchmark -- Grievance Mechanism Indicators

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-12 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| Human Rights Advisor | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-12 | GL-ProductManager | Initial draft created: 7 P0 features (Analytics Engine, Root Cause Analysis, Multi-Party Mediation, Remediation Effectiveness, Grievance Risk Scoring, Collective Grievance Handler, Regulatory Reporter), regulatory coverage verified (EUDR Art. 10/11/14-16/29/31, CSDDD Art. 7/8/9, UNGP P.31, OECD Ch.IV, ILO 169), module path aligned with GreenLang conventions, integration with EUDR-031 boundary clearly defined |
