# PRD: AGENT-EUDR-034 -- Annual Review Scheduler Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-034 |
| **Agent ID** | GL-EUDR-ARS-034 |
| **Component** | Annual Review Scheduler Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence (Category 5) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-12 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 8(1) (annual renewal of DDS), 13 (simplified due diligence monitoring and review), 14-16 (competent authority inspections and checks), 29 (country benchmarking updates), 31 (5-year record keeping) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **Prerequisite Agents** | AGENT-EUDR-001 (Supply Chain Mapping Master), AGENT-EUDR-006 (Plot Boundary Manager), AGENT-EUDR-026 (Due Diligence Orchestrator), AGENT-EUDR-028 (Risk Assessment Engine), AGENT-EUDR-030 (Documentation Generator), AGENT-EUDR-033 (Continuous Monitoring Agent) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) is not a single filing obligation. Article 8(1) mandates that operators and traders who have submitted a Due Diligence Statement (DDS) must renew that statement at least once every 12 months. This creates a perpetual compliance cycle: every DDS has a 12-month validity window, after which the operator must re-execute the full due diligence process -- re-gather Article 9 information, re-assess risk under Article 10, re-evaluate and update mitigation measures under Article 11, and re-submit a renewed DDS under Article 12. For a large EU importer with 500 active DDS submissions covering multiple commodities, countries, and supply chains, this means managing 500 overlapping annual renewal cycles, each with its own expiry date, its own set of dependent data elements, and its own coordination requirements across upstream agents.

Article 13 adds further complexity for operators benefiting from simplified due diligence for products originating from low-risk countries. While simplified due diligence reduces the depth of risk assessment required, it does not eliminate the obligation to review the due diligence at regular intervals and to ensure that the conditions for simplified treatment (low-risk country classification) still hold. If the European Commission reclassifies a country from Low to Standard or High risk under Article 29, operators must immediately exit simplified due diligence and conduct full risk assessment -- a transition that requires coordinated action across multiple agents within a tight timeframe.

Articles 14 through 16 grant competent authorities the power to conduct checks on operators at any time. Article 14(1) specifies that competent authorities shall carry out checks to verify that operators comply with their due diligence obligations, including the obligation to renew DDS submissions. An operator that allows DDS submissions to lapse without renewal faces enforcement action regardless of whether the underlying supply chain has changed. The penalty regime under Articles 23-25 includes fines of up to 4% of annual EU turnover, confiscation of goods, temporary exclusion from public procurement, and public naming.

The GreenLang platform has built a comprehensive suite of 33 EUDR agents spanning supply chain traceability (EUDR-001 through EUDR-015), risk assessment (EUDR-016 through EUDR-020), environmental and social due diligence (EUDR-021 through EUDR-022), and due diligence workflow agents (EUDR-023 through EUDR-033). These agents collectively perform every analytical step of the due diligence lifecycle: mapping supply chains, verifying geolocation, analyzing satellite imagery, assessing risk, designing mitigation measures, generating documentation, managing stakeholder engagement, processing grievances, and providing continuous monitoring. However, none of these agents manages the temporal dimension of compliance -- the annual review cycle itself. Today, EU operators face the following critical gaps:

- **No annual review cycle management**: The 33 upstream agents execute due diligence tasks when invoked. But there is no scheduling engine that tracks when each DDS was submitted, calculates when its renewal is due, determines the review preparation window (e.g., 90 days before expiry), assigns review tasks to responsible personnel, and ensures that all renewal activities are completed before the 12-month deadline. Operators currently track DDS expiry dates in spreadsheets or calendar entries, a method that fails catastrophically at scale. A large operator with 500+ active DDS submissions cannot manually track 500 overlapping 12-month cycles, each requiring coordination across 6-10 upstream agents, without missing deadlines.

- **No regulatory deadline tracking with escalation**: EUDR Article 8(1) creates a hard 12-month deadline for each DDS. Missing this deadline does not merely create a compliance gap -- it renders the associated products non-compliant for EU market placement until a new DDS is submitted. There is no system that tracks these deadlines with tiered escalation: early warning (90 days), standard alert (60 days), urgent alert (30 days), critical escalation (7 days), and overdue (0 days). Without automated escalation, deadline awareness depends on individual memory and manual follow-up, which breaks down when compliance officers are managing hundreds of concurrent deadlines across multiple commodities and countries.

- **No review checklist generation per commodity and context**: An annual review of a cocoa supply chain from Ghana requires different verification steps than a review of a timber supply chain from Brazil or a palm oil supply chain from Indonesia. The Article 9 information requirements are the same, but the practical verification steps differ by commodity (e.g., rubber latex aggregation challenges vs. cattle pasture rotation tracking), by country (high-risk vs. low-risk country determines simplified vs. full due diligence), and by risk profile (supply chains with prior deforestation alerts require enhanced re-verification). There is no engine that auto-generates context-aware, commodity-specific, risk-adapted annual review checklists that guide compliance officers through every step of the renewal process.

- **No cross-entity review coordination**: An annual DDS review does not happen in isolation. Renewing a DDS requires re-verification of supply chain data (EUDR-001), re-validation of plot boundaries (EUDR-006), refreshed risk assessments (EUDR-028), updated mitigation measures where needed (EUDR-029), and regenerated documentation (EUDR-030). These upstream dependencies must be executed in the correct order, and the results must cascade through the due diligence pipeline. Today, compliance officers must manually trigger each upstream agent, wait for results, feed outputs to downstream agents, and assemble the final renewal package. This manual orchestration takes 20-40 hours per DDS renewal and is error-prone, with missed dependencies resulting in incomplete renewals.

- **No multi-year comparison capability**: EUDR compliance is not just about annual renewal -- it is about demonstrating continuous improvement and tracking year-over-year changes. Competent authorities conducting checks under Article 14 may ask operators to demonstrate how their supply chain knowledge, risk assessment, and mitigation measures have evolved since the previous DDS submission. Without a multi-year comparison engine, operators cannot systematically identify what changed between Year 1 and Year 2: new suppliers added, suppliers removed, risk scores shifted, deforestation alerts triggered, plot boundaries modified, certifications obtained or lost, mitigation measures implemented or discontinued. This gap makes it impossible to provide competent authorities with the longitudinal compliance narrative they expect.

- **No unified compliance calendar**: An operator's EUDR compliance obligations span multiple temporal dimensions: DDS renewal deadlines (12-month cycles per product), competent authority inspection windows, third-party audit schedules (EUDR-024), certification renewal dates, country benchmarking update cycles (Article 29), data freshness windows (Article 8), and regulatory publication cycles. These temporal obligations are currently scattered across different systems: DDS dates in the documentation generator, audit dates in the audit manager, certification dates in document authentication, and inspection dates in email correspondence. There is no unified calendar that consolidates all compliance temporal obligations into a single view, identifies conflicts and overlaps, and enables proactive scheduling.

- **No automated notification and assignment system**: Annual reviews require coordination across multiple stakeholders: compliance officers initiate reviews, supply chain analysts update supply chain data, procurement managers contact suppliers for updated information, risk analysts refresh risk assessments, and external auditors may conduct verification. There is no automated system that notifies the right stakeholders at the right time, assigns specific review tasks based on role and expertise, tracks task completion, sends reminders for overdue tasks, escalates to management when deadlines are at risk, and generates completion confirmations when all review steps are finished.

- **No review history and audit trail**: EUDR Article 31 requires operators to retain due diligence records for at least 5 years. This includes records of when annual reviews were conducted, what was reviewed, what changes were identified, what actions were taken, and what the outcome was. Without a structured review history, operators cannot demonstrate to competent authorities that they have complied with the annual review obligation, that reviews were timely and thorough, and that identified issues were addressed. The absence of an auditable review history is itself a compliance risk.

Without an annual review scheduling agent, the entire EUDR compliance lifecycle lacks temporal governance. The 33 upstream agents provide the analytical capability to perform due diligence. EUDR-033 (Continuous Monitoring Agent) provides real-time surveillance between cycles. But without a scheduling agent that manages the annual renewal cycle itself -- tracking deadlines, generating checklists, coordinating entities, managing notifications, comparing years, and maintaining review history -- operators cannot sustain compliance over the multi-year horizon that EUDR demands. Annual review management is the temporal backbone of EUDR compliance.

### 1.2 Solution Overview

Agent-EUDR-034: Annual Review Scheduler is a specialized temporal governance agent that manages the complete annual due diligence review lifecycle mandated by EUDR Article 8(1). It operates as the "calendar and conductor" of the EUDR agent ecosystem -- the agent that ensures every DDS is renewed on time, every review is thorough, every stakeholder is notified, every dependency is coordinated, and every review cycle is recorded with full audit trail. It bridges the gap between the analytical agents that perform due diligence and the temporal obligations that require due diligence to be renewed annually.

The agent consumes DDS submission data from EUDR-030 (Documentation Generator), supply chain data from EUDR-001 (Supply Chain Mapping Master), plot data from EUDR-006 (Plot Boundary Manager), risk assessment data from EUDR-028 (Risk Assessment Engine), and monitoring events from EUDR-033 (Continuous Monitoring Agent). It produces review schedules, compliance calendars, review checklists, stakeholder notifications, multi-year comparison reports, and review audit trails that feed into the due diligence orchestration layer (EUDR-026) and documentation generator (EUDR-030).

Core capabilities:

1. **Review Cycle Manager** -- Manages the complete annual review lifecycle for every active DDS. Tracks DDS submission dates, calculates renewal deadlines (12 months from submission), defines configurable review windows (default: 90 days before expiry for review preparation start, 30 days for review completion target, 7 days for critical escalation). Supports multiple review cycle statuses: Not Started, Scheduled, In Progress, Under Review, Completed, Overdue. Handles multi-commodity operators with hundreds of overlapping review cycles through intelligent scheduling that prevents bottleneck periods. Supports both calendar-year-aligned and rolling 12-month review strategies. Manages review cycle dependencies: if a DDS renewal depends on an updated risk assessment from EUDR-028, the review cycle cannot advance to "Under Review" until the risk assessment is complete. Provides review cycle dashboards showing pipeline status across all active DDS submissions.

2. **Regulatory Deadline Tracker** -- Precision tracking of EUDR Article 8(1) 12-month renewal requirements with multi-tiered escalation. Calculates exact renewal deadlines for every active DDS based on submission date recorded in EUDR-030. Implements five escalation tiers: Planning (90 days before expiry, triggers review initiation), Standard Alert (60 days, triggers stakeholder notification), Urgent Alert (30 days, triggers management escalation), Critical Escalation (7 days, triggers executive notification and emergency review protocol), and Overdue (0 days, triggers compliance incident and remediation workflow). Tracks submission deadlines to competent authorities when inspection-triggered reviews have fixed response windows. Integrates with EUDR-033 (Continuous Monitoring Agent) regulatory update feed to detect changes to the 12-month renewal period if the regulation is amended. Provides deadline analytics: average time to complete review, review completion rate by deadline tier, overdue rate, and trending.

3. **Review Checklist Generator** -- Auto-generates comprehensive, context-aware annual review checklists tailored to the specific commodity, country of origin, risk profile, supply chain complexity, and prior review history of each DDS. Implements commodity-specific checklist templates for all seven EUDR-regulated commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood), each reflecting the unique supply chain characteristics and verification requirements of that commodity (e.g., animal movement records for cattle, cooperative aggregation verification for cocoa, concession boundary re-verification for wood). Adapts checklists based on country risk classification under Article 29: full due diligence checklists for Standard and High-risk countries, simplified due diligence checklists for Low-risk countries. Adapts checklists based on prior risk assessment results: supply chains with prior deforestation alerts, high risk scores, or unresolved gaps receive enhanced verification checklists. Ensures 100% coverage of Article 9 information requirements, Article 10 risk assessment criteria, and Article 11 mitigation measure evaluation in every checklist. Tracks checklist completion status with percentage-complete tracking and blocking item identification.

4. **Entity Review Coordinator** -- Orchestrates the cross-entity dependencies required for a complete annual DDS review. Defines and manages review dependency chains: supply chain re-mapping (EUDR-001) must complete before risk re-assessment (EUDR-028), which must complete before mitigation re-evaluation (EUDR-029), which must complete before DDS regeneration (EUDR-030). Triggers upstream agent re-execution through the Due Diligence Orchestrator (EUDR-026) with review context parameters. Tracks completion status of each upstream dependency per review cycle. Handles parallel execution where dependencies allow (e.g., plot re-verification via EUDR-006 can proceed in parallel with supplier risk re-scoring via EUDR-017). Detects and resolves dependency deadlocks (e.g., circular dependencies between risk assessment and mitigation evaluation). Provides dependency graph visualization showing the status of all entities involved in each review cycle.

5. **Multi-Year Comparison Engine** -- Generates structured year-over-year comparison reports that document how an operator's EUDR compliance posture has evolved across annual review cycles. Compares current-year data against previous-year data across all dimensions: supply chain topology (nodes added/removed, tier depth changes, new commodities), geographic footprint (plots added/removed, boundary changes, new countries of origin), risk profile (risk score changes per supplier, per country, per commodity, new risk factors identified), deforestation status (new alerts, resolved alerts, alert trends), documentation completeness (Article 9 field coverage changes, gap closure progress), mitigation effectiveness (measures implemented, measures discontinued, effectiveness scores), and certification status (new certifications obtained, certifications lost, certifications upgraded/downgraded). Flags statistically significant changes using configurable materiality thresholds. Generates executive summary highlighting the top 10 most material changes between review cycles. Produces audit-ready comparison reports with full provenance hashes for every data point compared. Supports multi-year trending across 3+ years for operators with sufficient review history.

6. **Compliance Calendar** -- Provides a unified, interactive calendar consolidating all EUDR temporal obligations across the operator's entire compliance portfolio. Aggregates DDS renewal deadlines from the Review Cycle Manager (F1), competent authority inspection schedules from EUDR-024 (Third-Party Audit Manager), certification renewal dates from EUDR-012 (Document Authentication), country benchmarking update cycles from EUDR-033 (Continuous Monitoring Agent), data freshness expiry dates from EUDR-033, stakeholder engagement milestones from EUDR-031 (Stakeholder Engagement Tool), and grievance resolution deadlines from EUDR-032 (Grievance Mechanism Manager). Displays calendar in day/week/month/quarter/year views with color-coded event categories. Detects scheduling conflicts (e.g., 50 DDS renewals due in the same week) and recommends load-balanced scheduling. Supports calendar export in iCalendar (.ics), CSV, and JSON formats. Provides calendar analytics: upcoming obligations by type, overdue count, workload forecasting, and capacity planning.

7. **Automated Notification System** -- Delivers timely, role-appropriate, actionable notifications to all stakeholders involved in the annual review process. Implements multi-channel notification delivery: email, webhook (for integration with Slack, Teams, and custom systems), in-app dashboard notifications, and SMS for critical escalations. Defines notification templates for each review lifecycle event: review initiation, task assignment, task reminder (3-day and 1-day before due), task overdue, dependency completion, review milestone, review completion, deadline escalation (per tier), and review cancellation. Implements role-based notification routing: compliance officers receive all review notifications, supply chain analysts receive supply chain re-mapping tasks, risk analysts receive risk re-assessment tasks, procurement managers receive supplier outreach tasks, management receives escalation notifications, and auditors receive review completion confirmations. Supports notification preferences: per-user opt-in/opt-out by notification type and channel. Tracks notification delivery status (sent, delivered, read, actioned) with full audit trail. Implements notification throttling to prevent alert fatigue (maximum configurable notifications per user per day with digest option).

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| DDS renewal on-time rate | >= 99% of DDS renewed before 12-month expiry | % of DDS renewals completed before deadline |
| Review cycle completion time | < 30 days from initiation to DDS resubmission (vs. 20-40 hours manual effort spread over 60+ days) | Average elapsed time from review initiation to DDS regeneration |
| Review checklist completeness | 100% of Article 8/9/10/11 requirements covered in every checklist | Automated checklist-to-regulation mapping validation |
| Deadline escalation accuracy | Zero missed deadlines without prior escalation notification | % of overdue DDS that had all 5 escalation tiers fired before expiry |
| Entity coordination efficiency | < 5 manual interventions per review cycle (vs. 30+ manual steps currently) | Count of manual actions required per review cycle |
| Multi-year comparison coverage | 100% of comparable dimensions included in year-over-year reports | Dimension coverage audit per comparison report |
| Notification delivery rate | >= 99.5% of notifications delivered within 5 minutes of trigger | Notification delivery latency tracking |
| Calendar accuracy | 100% of EUDR temporal obligations reflected in compliance calendar | Completeness audit against all upstream agent deadlines |
| Review history audit readiness | 100% of completed reviews retrievable with full audit trail within 5 years | Retention compliance validation and retrieval tests |
| Zero-hallucination guarantee | 100% deterministic scheduling, deadline calculation, and checklist generation, no LLM in critical path | Bit-perfect reproducibility tests |
| Stakeholder satisfaction | >= 85% satisfaction rating from review participants | Post-review survey scores |
| Overdue review rate | < 1% of annual reviews overdue beyond 7-day grace period | % of reviews entering Overdue status |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, each requiring annual DDS renewal management, representing a regulatory compliance scheduling and workflow market of 2-4 billion EUR. The annual renewal obligation ensures recurring demand year after year, making this a perpetual market rather than a one-time compliance event.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities requiring automated annual review management to sustain compliance across multiple DDS submissions, estimated at 400M-800M EUR. Operators with 50+ active DDS submissions represent the highest-value segment due to the complexity of managing overlapping review cycles.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 25-50M EUR in annual review management module ARR. Pricing model based on number of active DDS submissions managed (per-DDS-per-year), creating natural expansion revenue as customers onboard more commodities and supply chains.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) with 100+ active DDS submissions requiring automated annual review cycle management at scale
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya) with supply chains spanning 10+ countries and 200+ direct suppliers, facing annual renewal obligations across all commodity-country combinations
- Timber and paper industry operators with complex processing chains where annual re-verification of plot boundaries, concession rights, and chain of custody is critical
- Automotive and tire manufacturers (rubber) with geographically concentrated supply chains in Southeast Asia requiring intensive annual re-verification due to deforestation risk
- Meat and leather importers (cattle) requiring annual re-verification of animal movement records and pasture geolocation across South American supply chains

**Secondary:**
- Compliance consulting firms managing annual review cycles for multiple operator clients, requiring multi-tenant scheduling and portfolio management
- Certification bodies (FSC, RSPO, PEFC, Rainforest Alliance) that need to coordinate their own audit schedules with operator annual review timelines
- Financial institutions requiring evidence that portfolio companies are maintaining annual EUDR compliance renewals as part of ESG risk management
- Trade associations coordinating annual review schedules across member companies to optimize shared supplier outreach and data collection
- SME importers (1,000-10,000 shipments/year) preparing for the June 30, 2026 enforcement deadline who need structured annual review management from day one

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual spreadsheet/calendar tracking | No cost; familiar; flexible | Does not scale beyond 20-30 DDS; no escalation; no checklist generation; no cross-entity coordination; no audit trail; error-prone | Automated scheduling for 1,000+ DDS; 5-tier escalation; commodity-specific checklists; full audit trail |
| Generic project management (Jira, Asana, Monday.com) | Task tracking; notifications; collaboration | Not EUDR-specific; no regulatory deadline calculation; no Article 8/9/10/11 checklist generation; no multi-year comparison; no agent integration | Purpose-built for EUDR Article 8(1); auto-generates regulatory checklists; native integration with 33 upstream agents |
| ERP compliance modules (SAP GRC, Oracle GRC) | Enterprise integration; workflow engine; approval chains | Heavyweight; expensive; 12-18 month implementation; not EUDR-specific; no commodity-specific logic; no deforestation context | EUDR-native; commodity-aware; deploys in weeks; lighter weight; regulatory updates included |
| Niche EUDR compliance tools (Preferred by Nature, Ecosphere+) | Commodity expertise; certification knowledge | No annual review scheduling; point-in-time assessment only; no multi-year comparison; no cross-agent orchestration | Full annual review lifecycle; multi-year trending; orchestrates across 33 EUDR agents |
| Custom internal compliance calendars | Tailored to organization; low incremental cost | Fragile; single-user; no scalable notification; no dependency management; no checklist generation; maintenance burden | Production-grade; multi-user; scalable notifications; dependency management; self-maintaining |
| Compliance calendar SaaS (Diligent, Ideagen) | Multi-regulation calendar; board reporting; broad compliance coverage | Generic across all regulations; no EUDR commodity logic; no deforestation context; no Article 8 deadline calculation; no agent integration | EUDR-specialized; commodity-specific; calculates Article 8 deadlines; integrates with full EUDR agent ecosystem |

### 2.4 Differentiation Strategy

1. **EUDR Article 8(1) native** -- The only annual review management system built specifically for EUDR's 12-month DDS renewal obligation, with deadline calculations derived directly from DDS submission dates recorded in EUDR-030.
2. **Commodity-specific intelligence** -- Review checklists are not generic compliance templates. They are tailored to the specific verification requirements of each of the 7 EUDR-regulated commodities, reflecting real-world supply chain complexities (cooperative aggregation for cocoa, latex aggregation for rubber, animal movement for cattle, concession management for wood).
3. **33-agent orchestration** -- Annual reviews are not standalone events. They trigger coordinated re-execution of upstream agents (EUDR-001 supply chain re-mapping, EUDR-006 plot re-verification, EUDR-028 risk re-assessment, EUDR-029 mitigation re-evaluation, EUDR-030 DDS regeneration) through the Due Diligence Orchestrator (EUDR-026).
4. **Multi-year compliance intelligence** -- The only platform that provides structured year-over-year comparison across all compliance dimensions, enabling operators to demonstrate continuous improvement to competent authorities and provide the longitudinal compliance narrative that auditors expect.
5. **Zero-hallucination temporal governance** -- All deadline calculations, checklist generation, and scheduling decisions are deterministic with no LLM in the critical path. Every scheduled event, deadline, and notification is reproducible and provenance-tracked.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Ensure zero DDS lapses for active customers due to missed annual review deadlines | >= 99% on-time DDS renewal rate | Q3 2026 |
| BG-2 | Reduce the elapsed time for annual DDS review from 60+ days to under 30 days | Average review cycle time < 30 days | Q3 2026 |
| BG-3 | Reduce manual effort per annual review from 20-40 hours to under 4 hours | Average manual intervention count < 5 per review cycle | Q3 2026 |
| BG-4 | Enable operators to demonstrate continuous year-over-year compliance improvement to competent authorities | 100% of reviews include multi-year comparison reports | Q4 2026 |
| BG-5 | Maintain continuous competent authority inspection readiness with respect to annual review obligations | < 30 minutes from inspection request to review history report | Q3 2026 |
| BG-6 | Become the reference annual EUDR review management platform | 500+ enterprise customers using annual review scheduling module | Q4 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Complete review cycle management | Track every active DDS through its 12-month lifecycle from submission to renewal with configurable review windows and multi-status tracking |
| PG-2 | Precise regulatory deadline tracking | Calculate and track exact Article 8(1) renewal deadlines with 5-tier escalation and zero missed deadlines |
| PG-3 | Commodity-specific checklist generation | Auto-generate annual review checklists tailored to commodity, country, risk profile, and prior review history |
| PG-4 | Cross-entity coordination | Orchestrate review dependencies across supply chain mapping, plot verification, risk assessment, mitigation evaluation, and DDS regeneration agents |
| PG-5 | Multi-year comparison | Generate structured year-over-year comparison reports documenting compliance evolution across all dimensions |
| PG-6 | Unified compliance calendar | Consolidate all EUDR temporal obligations into a single interactive calendar with conflict detection and workload forecasting |
| PG-7 | Automated stakeholder notification | Deliver timely, role-appropriate, multi-channel notifications throughout the review lifecycle |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Review scheduling performance | < 5 seconds to generate review schedule for 1,000 active DDS submissions |
| TG-2 | Checklist generation performance | < 3 seconds to generate a commodity-specific, context-aware review checklist |
| TG-3 | Multi-year comparison performance | < 30 seconds for full year-over-year comparison across 10,000 supply chain entities |
| TG-4 | Calendar rendering performance | < 2 seconds to render compliance calendar with 5,000+ events |
| TG-5 | Notification dispatch latency | < 5 seconds from trigger event to notification dispatch |
| TG-6 | API response time | < 200ms p95 for standard scheduling queries |
| TG-7 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-8 | Zero-hallucination | 100% deterministic scheduling, deadline calculation, and checklist generation, bit-perfect reproducibility |
| TG-9 | Resource efficiency | < 256 MB memory baseline for scheduling engine; < 2 CPU cores sustained |
| TG-10 | Concurrent review management | Support 10,000+ concurrent active review cycles without performance degradation |

### 3.4 Non-Goals

1. Performing due diligence analysis (EUDR-001 through EUDR-029 handle analytical work; this agent schedules and coordinates reviews but does not perform the underlying analysis)
2. Generating or submitting DDS documents (EUDR-030 handles documentation generation; this agent triggers DDS regeneration through the orchestration layer)
3. Continuous real-time monitoring (EUDR-033 handles continuous monitoring; this agent manages the scheduled annual review cycle)
4. Risk assessment calculation (EUDR-028 handles risk assessment; this agent triggers risk re-assessment as a review dependency)
5. Mitigation measure design (EUDR-029 handles mitigation design; this agent tracks mitigation re-evaluation as a review step)
6. Direct communication with suppliers (EUDR-031 Stakeholder Engagement Tool handles supplier communications; this agent triggers outreach through that tool)
7. Grievance management (EUDR-032 handles grievances; this agent does not process complaints)
8. Satellite imagery analysis or deforestation detection (EUDR-003/004/005/020 handle satellite analytics)
9. Legal interpretation of regulatory changes (this agent tracks regulatory deadlines but does not provide legal counsel)
10. Financial or commercial aspects of supply chain management (this agent focuses exclusively on compliance review scheduling)

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, sourcing cocoa from 12 countries, 350+ active DDS submissions across cocoa, palm oil, and soya supply chains |
| **EUDR Pressure** | Must renew 350+ DDS submissions on rolling 12-month cycles; each renewal requires re-verification of supply chain data, risk assessments, and mitigation measures across 200+ suppliers and 1,000+ production plots; competent authorities expect evidence of systematic annual review process |
| **Pain Points** | Tracks DDS expiry dates in an Excel spreadsheet that is out of date within a week; cannot prioritize which reviews to start first when 40 DDS renewals fall in the same month; no automated notification when review deadlines approach; manual coordination with 6 different teams (supply chain, procurement, risk, legal, sustainability, IT) takes 25+ hours per review; no structured way to compare this year's review findings with last year's |
| **Goals** | Automated review scheduling that initiates 90 days before expiry; role-based task assignment with deadline tracking; priority ranking of reviews based on risk level and complexity; multi-year comparison reports for board and auditor presentation; unified calendar showing all compliance deadlines in one view; email alerts when any review is at risk of missing its deadline |
| **Technical Skill** | Moderate -- comfortable with web applications, dashboards, and collaboration tools but not a developer |

### Persona 2: Supply Chain Analyst -- Lukas (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Supply Chain Analyst at an EU timber importer |
| **Company** | 800 employees, importing tropical and temperate wood from 20+ countries, 180 active DDS submissions |
| **EUDR Pressure** | Timber supply chains are the deepest (5-8 tiers) and most complex of all EUDR commodities; annual reviews require re-verification of forest concession boundaries, chain of custody through multiple processing steps (sawmill, veneer, furniture), and species composition verification; plot boundary re-verification is particularly critical as concession boundaries change due to government reclassification, illegal encroachment, and natural boundary shifts |
| **Pain Points** | Annual review of one timber DDS requires coordinating with EUDR-001 (re-map supply chain), EUDR-006 (re-verify plot boundaries), EUDR-012 (re-validate FSC certificates), EUDR-028 (re-assess risk), and EUDR-030 (regenerate DDS); manual orchestration of this dependency chain takes 3-4 weeks per DDS; does not know the correct order of operations and sometimes triggers risk assessment before supply chain re-mapping is complete, producing stale results; no tool to visualize what changed since last year's review |
| **Goals** | Automated dependency chain management that triggers upstream agents in the correct order; clear visualization of which review steps are complete, in progress, and blocked; year-over-year comparison showing exactly what changed in the timber supply chain; checklist specific to timber supply chains (concession re-verification, species validation, processing chain re-mapping) |
| **Technical Skill** | High -- comfortable with data tools, APIs, graph databases, and analytical platforms |

### Persona 3: Procurement Manager -- Ana (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Procurement Director at a palm oil refinery |
| **Company** | 3,000 employees, sourcing from 200+ plantations across Indonesia and Malaysia, 120 active DDS submissions |
| **EUDR Pressure** | Annual reviews require re-engagement with 200+ plantation suppliers for updated geolocation data, certification renewals, and sub-tier disclosure; many suppliers are small-scale and slow to respond; review bottleneck is almost always supplier data collection |
| **Pain Points** | Gets no advance warning about upcoming reviews; receives ad hoc requests from compliance team 2 weeks before deadline; cannot plan supplier outreach in advance; does not know which suppliers need re-engagement until the review starts; spends 60% of review time chasing suppliers for updated data; no way to batch supplier outreach for multiple DDS reviews that share the same suppliers |
| **Goals** | 90-day advance notification of supplier data collection needs; consolidated supplier outreach list across all upcoming reviews (avoid contacting the same supplier multiple times for different DDS reviews); progress tracking for supplier response rates; escalation to compliance officer when suppliers are non-responsive; historical supplier response metrics to identify chronically slow responders |
| **Technical Skill** | Low-moderate -- uses ERP, web applications, and email |

### Persona 4: Risk Analyst -- Stefan (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Risk Analyst at a European rubber manufacturer |
| **Company** | 2,500 employees, sourcing rubber from Southeast Asia and West Africa, 90 active DDS submissions |
| **EUDR Pressure** | Annual risk re-assessment is the most analytically intensive part of the review cycle; must determine whether risk profiles have changed since the previous assessment; must evaluate whether mitigation measures remain effective; must determine whether new risk factors have emerged |
| **Pain Points** | Does not receive structured review assignments; compliance officer sends ad hoc emails requesting risk re-assessment without specifying scope or providing previous year's assessment results; must manually retrieve last year's risk data from EUDR-028 to perform comparison; no structured format for documenting what changed in the risk profile year-over-year; sometimes performs risk re-assessment before supply chain re-mapping is complete, resulting in assessment based on stale supply chain data |
| **Goals** | Structured risk re-assessment task assignment with clear scope definition; automatic provision of previous year's risk assessment data as baseline for comparison; dependency enforcement ensuring supply chain re-mapping completes before risk re-assessment begins; structured year-over-year risk comparison report; clear documentation of risk score changes with root cause analysis |
| **Technical Skill** | High -- comfortable with risk modeling tools, analytics platforms, and statistical analysis |

### Persona 5: External Auditor -- Dr. Hofmann (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm conducting EUDR compliance audits for 50+ operators |
| **EUDR Pressure** | Must verify that operators conduct timely and thorough annual reviews per Article 8(1); must assess whether review processes are systematic and comprehensive; must verify that year-over-year changes are identified and addressed |
| **Pain Points** | Operators provide inconsistent documentation of annual review processes; some operators cannot demonstrate when reviews were conducted; review checklists are ad hoc and incomplete; no standardized way to verify that all Article 9/10/11 requirements were re-evaluated; year-over-year comparison data is unavailable or manually compiled and unreliable |
| **Goals** | Read-only access to review history with full audit trail; evidence that reviews were initiated on time (before 12-month expiry); verification that review checklists covered all regulatory requirements; access to multi-year comparison reports with provenance hashes; verification that review dependencies were executed in correct order; evidence of stakeholder notification and task completion tracking |
| **Technical Skill** | Moderate -- comfortable with audit software, compliance dashboards, and evidence review platforms |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(2)** | Operators shall exercise due diligence -- collect information, assess risk, mitigate risk -- for each relevant product prior to placement on the EU market | Annual Review Scheduler ensures that the due diligence exercised at initial DDS submission is systematically renewed before the 12-month expiry, maintaining the continuous compliance posture required for lawful market placement |
| **Art. 8(1)** | The due diligence statement shall be renewed at least once every 12 months; operators shall update their due diligence statement whenever they become aware of relevant new information, including substantiated concerns | Review Cycle Manager (F1) calculates 12-month renewal deadlines for every active DDS, tracks review cycle status, and ensures renewal is completed before expiry. Regulatory Deadline Tracker (F2) implements 5-tier escalation to prevent any DDS from expiring without renewal. Entity Review Coordinator (F4) ensures all upstream agents are re-executed as part of the renewal process |
| **Art. 8(2)** | Operators shall review the information collected, the risk assessment carried out, and the risk mitigation measures taken as part of due diligence at least annually | Review Checklist Generator (F3) ensures that every annual review systematically covers re-validation of Article 9 information, re-assessment of Article 10 risk criteria, and re-evaluation of Article 11 mitigation measures |
| **Art. 9(1)(a-d)** | Information to be collected: product description, quantity, country of production, geolocation of all plots of land, date/period of production, supplier and buyer identification | Review Checklist Generator (F3) includes Article 9 information re-verification as mandatory checklist items in every annual review; Entity Review Coordinator (F4) triggers EUDR-001 supply chain re-mapping and EUDR-006 plot re-verification to refresh Article 9 data |
| **Art. 10(1)** | Operators shall assess and identify the risk of non-compliance in the supply chain | Entity Review Coordinator (F4) triggers EUDR-028 Risk Assessment Engine re-execution as a mandatory dependency in every annual review cycle; Multi-Year Comparison Engine (F5) documents year-over-year risk profile changes |
| **Art. 10(2)(a-g)** | Risk assessment criteria: deforestation risk in country of production, forest area change, supply chain complexity, circumvention risk, country concerns, consultation with civil society and indigenous peoples, third-party information | Review Checklist Generator (F3) maps each criterion to specific checklist items, ensuring all 7 risk assessment criteria are re-evaluated in every annual review |
| **Art. 11(1)** | Where the risk assessment identifies a non-negligible risk, operators shall adopt adequate and proportionate risk mitigation measures | Entity Review Coordinator (F4) triggers EUDR-029 Mitigation Measure Designer re-evaluation when annual risk re-assessment identifies non-negligible risk; Multi-Year Comparison Engine (F5) tracks mitigation measure effectiveness year-over-year |
| **Art. 11(2)** | Risk mitigation measures shall include: (a) requesting additional information or documentation, (b) carrying out independent surveys or audits, (c) any other adequate measure | Review Checklist Generator (F3) includes mitigation measure verification steps mapped to Article 11(2)(a-c) categories; checklist tracks whether each mitigation measure from the previous cycle remains effective or requires update |
| **Art. 12(1)** | Operators shall submit a due diligence statement to the EU Information System | Entity Review Coordinator (F4) triggers EUDR-030 Documentation Generator to regenerate the DDS as the final step of the annual review cycle; Review Cycle Manager (F1) tracks DDS resubmission as the review completion milestone |
| **Art. 13** | Simplified due diligence for products from countries classified as low-risk by the Commission | Review Checklist Generator (F3) generates simplified review checklists for low-risk country DDS submissions; Regulatory Deadline Tracker (F2) monitors Article 29 country reclassification events from EUDR-033 that would require transition from simplified to full review |
| **Art. 14(1)** | Competent authorities shall carry out checks on operators to verify compliance with due diligence obligations | All review cycle data (schedules, checklists, completion records, multi-year comparisons) maintained as audit-ready evidence for competent authority inspection; review history retrievable within 30 minutes of inspection request |
| **Art. 14(2)** | Checks shall include examination of the due diligence system, risk assessment, and risk mitigation measures | Multi-Year Comparison Engine (F5) provides competent authorities with structured evidence of year-over-year due diligence evolution; review completion records demonstrate systematic annual compliance renewal |
| **Art. 15(1)** | Competent authorities shall perform risk-based checks, including documentary checks | Review history and completion records serve as documentary evidence for risk-based checks; automated compliance calendar demonstrates proactive deadline management |
| **Art. 16(1)** | Substantive checks -- competent authorities shall verify the accuracy and completeness of the due diligence statement | Multi-year comparison reports with provenance hashes enable competent authorities to verify the accuracy of DDS renewal claims; review checklists document which Article 9/10/11 requirements were re-verified |
| **Art. 29(1-3)** | Country benchmarking: Low, Standard, High risk classification by the European Commission, reviewed periodically | Compliance Calendar (F6) tracks EC country benchmarking publication cycles; Regulatory Deadline Tracker (F2) triggers emergency review protocol when a country is reclassified from Low to Standard/High risk, requiring affected DDS submissions to transition from simplified to full due diligence |
| **Art. 31(1)** | Operators shall retain documentation of the due diligence exercised for at least 5 years from the date of the statement | All review cycle records, checklists, completion evidence, multi-year comparisons, notification logs, and dependency execution records retained for minimum 5 years with immutable audit trail; retention tracking and expiry alerting integrated into review history |

### 5.2 Annual Review Requirements by Commodity

| Commodity | Key Annual Re-Verification Steps | Typical Review Complexity | Commodity-Specific Checklist Focus |
|-----------|----------------------------------|---------------------------|-----------------------------------|
| **Cattle** | Re-verify pasture GPS coordinates; re-check animal movement records; re-validate slaughterhouse and packer certifications; re-assess grazing land deforestation status against Dec 31, 2020 cutoff | High (animal movement tracking, pasture rotation, multi-establishment records) | Animal traceability re-verification, pasture expansion monitoring, veterinary certification currency |
| **Cocoa** | Re-verify smallholder farm GPS/polygons; re-assess cooperative aggregation records; re-validate Rainforest Alliance/UTZ certifications; re-check deforestation alerts on cocoa-growing plots; re-map sub-tier cooperative membership | Very High (thousands of smallholders per cooperative; high cooperative membership turnover) | Smallholder plot re-registration, cooperative membership changes, yield-to-area ratio validation |
| **Coffee** | Re-verify farm GPS coordinates with altitude; re-assess wet mill and dry mill processing chain; re-validate certifications (Rainforest Alliance, Fair Trade); re-check origin segregation through processing | High (altitude-specific origin segregation; wet/dry processing chain complexity) | Processing chain re-verification, altitude/origin segregation validation, cherry-to-green conversion tracking |
| **Palm Oil** | Re-verify plantation GPS/polygons; re-assess mill supply base; re-validate RSPO/ISPO certifications; re-check deforestation alerts on plantation concessions; re-assess peatland drainage status | High (high deforestation risk; RSPO mass balance challenges; peatland complexity) | Concession boundary re-verification, RSPO certification chain currency, peatland status re-assessment |
| **Rubber** | Re-verify smallholder plot GPS coordinates; re-assess latex aggregation chain; re-validate processing plant certifications; re-check deforestation alerts in rubber-growing regions | High (latex aggregation destroys traceability; smallholder fragmentation) | Aggregation point re-mapping, smallholder registration currency, latex volume reconciliation |
| **Soya** | Re-verify farm GPS/polygons; re-assess silo co-mingling practices; re-validate crusher certifications; re-check deforestation in Cerrado and Amazon biomes; re-assess land conversion status | Medium-High (large volumes; co-mingling at silos; biome-specific deforestation pressure) | Silo co-mingling re-assessment, biome boundary re-verification, land conversion monitoring |
| **Wood** | Re-verify forest concession boundaries; re-assess harvesting permits and quotas; re-validate FSC/PEFC chain of custody; re-check species composition; re-map processing chain (sawmill -> veneer -> furniture); re-verify legal logging authorization | Very High (concession boundary changes; multi-step processing; species mixing risk; legal authorization complexity) | Concession boundary re-survey, harvesting permit currency, species composition re-verification, processing chain re-mapping |

### 5.3 Key Regulatory Dates and Review Triggers

| Date/Trigger | Milestone | Agent Impact |
|--------------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | All annual reviews must re-verify deforestation-free status against this cutoff; no change to cutoff date expected |
| 12 months from each DDS submission | Article 8(1) renewal deadline per DDS | Review Cycle Manager (F1) calculates and tracks individual 12-month deadlines for every active DDS |
| EC country benchmarking updates (Art. 29) | Country risk reclassification | Regulatory Deadline Tracker (F2) triggers emergency reviews for all DDS submissions sourcing from reclassified countries |
| June 30, 2026 | SME enforcement begins | First annual review cycle for SME operators begins June 30, 2027; agent must handle SME onboarding wave |
| Competent authority inspection request | Ad hoc review demand | Agent must produce review history and compliance evidence within 30 minutes |
| Substantiated concern received | Article 8(1) triggered update | Review Cycle Manager (F1) initiates an out-of-cycle review when a substantiated concern triggers DDS update obligation |
| Certification expiry | Supplier-level re-verification needed | Compliance Calendar (F6) tracks certification expiry dates; Review Checklist Generator (F3) adds certification re-validation to upcoming review checklists |
| Deforestation alert on monitored plot | Risk re-assessment needed | Entity Review Coordinator (F4) triggers expedited risk re-assessment through EUDR-028 when EUDR-033 reports a deforestation alert correlated with an active DDS |

### 5.4 Simplified vs. Full Due Diligence Review Requirements

| Review Aspect | Full Due Diligence Review (Standard/High-Risk Countries) | Simplified Due Diligence Review (Low-Risk Countries, Art. 13) |
|---------------|----------------------------------------------------------|--------------------------------------------------------------|
| Article 9 information re-verification | Full re-verification of all Article 9 data elements | Re-verification of product description, quantity, and supplier identification; geolocation verification may use existing data if unchanged |
| Article 10 risk assessment | Full risk re-assessment across all 7 criteria (Art. 10(2)(a-g)) | Simplified risk review confirming low-risk country classification still applies; abbreviated assessment of supply chain complexity and circumvention risk |
| Article 11 mitigation evaluation | Full re-evaluation of all active mitigation measures; assessment of new risks requiring new measures | Abbreviated mitigation review; typically no active mitigation measures for low-risk countries |
| Supply chain re-mapping depth | Full multi-tier re-mapping to production plot level | Tier 1 re-confirmation with spot-check verification of deeper tiers |
| Satellite/deforestation re-verification | Full satellite re-verification of all production plots | Abbreviated verification based on country-level monitoring data |
| Review checklist length | 40-60 checklist items | 15-25 checklist items |
| Typical review duration | 15-30 days | 5-10 days |
| Estimated effort | 15-30 hours per DDS | 3-8 hours per DDS |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 7 features below are P0 launch blockers. The agent cannot ship without all 7 features operational. Features 1-4 form the core scheduling and orchestration engine; Features 5-7 form the intelligence, visualization, and communication layer.

**P0 Features 1-4: Core Scheduling and Orchestration Engine**

---

#### Feature 1: Review Cycle Manager

**User Story:**
```
As a compliance officer,
I want every active DDS to be automatically tracked through a 12-month review lifecycle with configurable review windows,
So that I can ensure no DDS expires without renewal and every review is completed on time.
```

**Acceptance Criteria:**
- [ ] Automatically imports all active DDS submissions from EUDR-030 Documentation Generator, including DDS reference number, submission date, commodity, country of production, operator ID, and associated supply chain graph ID
- [ ] Calculates the 12-month renewal deadline for each active DDS based on its original submission date
- [ ] Defines configurable review windows per operator with defaults: review preparation start at 90 days before expiry, review completion target at 30 days before expiry, critical escalation at 7 days before expiry
- [ ] Supports six review cycle statuses with defined transitions: Not Started -> Scheduled -> In Progress -> Under Review -> Completed -> Archived; plus Overdue as an exceptional status reachable from any non-terminal state
- [ ] Prevents backward status transitions (e.g., cannot move from Under Review back to Scheduled) except through explicit administrative override with audit log entry
- [ ] Creates review cycle records automatically when a DDS enters its review preparation window (90 days before expiry by default)
- [ ] Supports intelligent scheduling for operators with 100+ concurrent review cycles: distributes review initiation across the review window to prevent bottleneck weeks where all reviews start simultaneously
- [ ] Calculates review workload per week/month based on active review cycles, enabling capacity planning
- [ ] Supports both calendar-year-aligned review strategy (all reviews for a commodity aligned to the same quarter) and rolling 12-month review strategy (each DDS on its own 12-month cycle from submission)
- [ ] Handles out-of-cycle reviews triggered by substantiated concerns (Article 8(1) triggered update) or competent authority requests, with separate tracking and priority escalation
- [ ] Tracks review completion percentage based on checklist items completed and dependency milestones reached
- [ ] Marks a review as Completed only when all mandatory checklist items are marked done, all review dependencies have completed successfully, and a renewed DDS has been generated by EUDR-030
- [ ] Archives completed review cycles with full audit trail and links to all artifacts (checklists, dependency results, comparison reports, notifications)
- [ ] Provides review pipeline dashboard showing all active reviews grouped by status, with filters by commodity, country, operator, and deadline urgency

**Non-Functional Requirements:**
- Performance: Schedule generation for 1,000 active DDS < 5 seconds
- Scalability: Support 10,000+ concurrent active review cycles
- Reliability: Zero missed deadline calculations (validated against reference implementation)
- Auditability: Every status transition recorded with actor, timestamp, and reason

**Dependencies:**
- EUDR-030 Documentation Generator (DDS submission records, DDS reference numbers)
- EUDR-001 Supply Chain Mapping Master (supply chain graph IDs associated with DDS)
- PostgreSQL + TimescaleDB (review cycle persistence)
- Redis (review schedule caching and status lookups)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- DDS submitted on February 29 (leap year) -- renewal deadline calculated as February 28 in non-leap years
- DDS submission date retroactively corrected in EUDR-030 -- review cycle deadline automatically recalculated
- Operator submits a new DDS for the same commodity/country before the existing DDS expires -- new review cycle created, old cycle archived
- Multiple DDS submissions for the same operator sharing the same suppliers -- review coordinator identifies shared dependencies to avoid duplicate upstream agent executions
- DDS rejected by EU Information System after submission -- review cycle returns to "In Progress" with remediation task added to checklist

---

#### Feature 2: Regulatory Deadline Tracker

**User Story:**
```
As a compliance officer,
I want precision tracking of every Article 8(1) renewal deadline with automated multi-tier escalation,
So that I never miss a DDS renewal deadline and management is alerted well before any deadline becomes critical.
```

**Acceptance Criteria:**
- [ ] Calculates exact renewal deadlines for every active DDS: submission_date + 12 months (accounting for month-end edge cases and leap years)
- [ ] Implements five escalation tiers with configurable thresholds and default values:
  - Tier 1 -- Planning: 90 days before expiry (triggers review initiation and checklist generation)
  - Tier 2 -- Standard Alert: 60 days before expiry (triggers stakeholder notification and task assignment)
  - Tier 3 -- Urgent Alert: 30 days before expiry (triggers management escalation if review not yet In Progress)
  - Tier 4 -- Critical Escalation: 7 days before expiry (triggers executive notification and emergency review protocol)
  - Tier 5 -- Overdue: 0 days (triggers compliance incident record, blocks associated products from new market placement until DDS is renewed)
- [ ] Fires escalation events exactly once per tier per review cycle (no duplicate escalation for the same DDS at the same tier)
- [ ] Supports per-operator escalation threshold customization (e.g., some operators want 120-day planning window instead of 90 days)
- [ ] Tracks additional regulatory deadlines beyond DDS renewal: competent authority inspection response windows (typically 10-30 business days), certification renewal deadlines (per certificate expiry from EUDR-012), and country benchmarking review triggers (from EUDR-033)
- [ ] Integrates with EUDR-033 Continuous Monitoring Agent to detect regulatory changes that affect the 12-month renewal period (in case the regulation is amended)
- [ ] Triggers emergency review protocol when EUDR-033 reports a country reclassification (Article 29) affecting DDS submissions using simplified due diligence: all affected DDS must be re-reviewed under full due diligence requirements within a configurable emergency window (default: 30 days)
- [ ] Provides deadline analytics dashboard: on-time completion rate, average time-to-complete, overdue rate, escalation frequency by tier, and trending
- [ ] Supports deadline simulation: "What if we receive 200 new DDS submissions next month? What does the deadline calendar look like?"
- [ ] Exports deadline data in CSV, JSON, and iCalendar (.ics) formats

**Non-Functional Requirements:**
- Accuracy: Deadline calculations validated against reference calendar library; zero tolerance for incorrect dates
- Timeliness: Escalation events fired within 60 seconds of reaching escalation threshold
- Durability: Deadline data persisted with point-in-time recovery; no deadline lost due to system restart

**Dependencies:**
- EUDR-030 Documentation Generator (DDS submission dates)
- EUDR-033 Continuous Monitoring Agent (regulatory update feed, country benchmarking changes)
- EUDR-012 Document Authentication (certification expiry dates)
- Notification System (Feature 7) for escalation dispatch

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- DDS submitted at 23:59 UTC on January 31 -- renewal deadline is January 31 of the following year at 23:59 UTC
- EC publishes country reclassification effective immediately -- all affected DDS escalated to emergency review within 1 hour
- Multiple DDS for same operator reach same escalation tier on same day -- consolidated escalation notification (not 200 separate emails)
- Operator requests deadline extension -- system records request but does not modify regulatory deadline (12-month period is legally fixed); marks review as Overdue if extension pushes past deadline

---

#### Feature 3: Review Checklist Generator

**User Story:**
```
As a compliance officer,
I want an automatically generated, commodity-specific, context-aware annual review checklist for every DDS renewal,
So that I can ensure every required verification step is completed and no regulatory requirement is missed.
```

**Acceptance Criteria:**
- [ ] Generates checklists tailored to the specific commodity (7 commodity-specific templates: cattle, cocoa, coffee, palm oil, rubber, soya, wood) with commodity-specific verification steps reflecting real-world supply chain characteristics
- [ ] Adapts checklists based on country risk classification (Article 29): full due diligence checklists for Standard/High-risk countries (40-60 items), simplified checklists for Low-risk countries (15-25 items)
- [ ] Adapts checklists based on the previous review cycle's findings: adds enhanced verification items for supply chains with prior deforestation alerts, high risk scores, unresolved gaps, or competent authority observations
- [ ] Organizes checklist items into regulatory sections mapped to EUDR Articles:
  - Section A: Article 9 Information Re-Verification (product description, quantities, country of production, geolocation, supplier/buyer identification)
  - Section B: Article 10 Risk Re-Assessment (all 7 criteria: (a) deforestation risk, (b) forest area change, (c) supply chain complexity, (d) circumvention risk, (e) country concerns, (f) mixing risk, (g) third-party information)
  - Section C: Article 11 Mitigation Measure Re-Evaluation (existing measures effectiveness, new risk factors requiring new measures, mitigation measure implementation evidence)
  - Section D: Article 12 DDS Regeneration (data aggregation, schema validation, submission preparation)
  - Section E: Cross-Cutting Verification (certification currency, supply chain graph completeness, plot boundary accuracy, document authentication status)
- [ ] Classifies each checklist item as Mandatory (must complete to mark review as done), Recommended (should complete but not blocking), or Optional (complete if time permits)
- [ ] Assigns estimated duration per checklist item based on complexity and historical completion data
- [ ] Assigns recommended role per checklist item (compliance officer, supply chain analyst, risk analyst, procurement manager, external auditor)
- [ ] Tracks completion status per item: Not Started, In Progress, Completed, Skipped (with mandatory justification for Mandatory items), Blocked (with dependency reference)
- [ ] Calculates overall checklist completion percentage with weighted scoring (Mandatory items carry higher weight)
- [ ] Supports checklist item dependencies (e.g., "Re-assess risk" is blocked until "Re-map supply chain" is completed)
- [ ] Links each checklist item to the upstream EUDR agent responsible for that verification (e.g., "Re-verify plot boundaries" links to EUDR-006)
- [ ] Generates checklist completion report with evidence links for each completed item
- [ ] Supports checklist versioning: if the EUDR regulation is amended and new requirements are added, new checklist items are appended to in-progress reviews

**Non-Functional Requirements:**
- Generation Time: < 3 seconds to generate a full context-aware checklist
- Completeness: 100% coverage of Article 8/9/10/11 requirements in every checklist (validated by regulatory mapping matrix)
- Reproducibility: Same DDS context produces same checklist (deterministic generation)

**Dependencies:**
- EUDR-028 Risk Assessment Engine (previous risk assessment results for context adaptation)
- EUDR-001 Supply Chain Mapping Master (supply chain complexity data for checklist adaptation)
- EUDR-033 Continuous Monitoring Agent (deforestation alert history, certification status)
- EUDR-030 Documentation Generator (previous DDS content for comparison baseline)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 regulatory subject matter expert)

**Edge Cases:**
- DDS covers multiple commodities (e.g., chocolate containing cocoa + soya + palm oil) -- generate merged checklist covering all commodity-specific requirements with deduplication of shared items
- Country reclassified during an in-progress review -- checklist dynamically expanded from simplified to full due diligence template; already-completed items retained
- New EUDR implementing regulation adds a new Article 9 data requirement -- new mandatory checklist item added to all in-progress and future reviews
- First-ever annual review (no previous cycle history) -- generate baseline checklist without year-over-year comparison items; flag as "initial review" for auditor context

---

#### Feature 4: Entity Review Coordinator

**User Story:**
```
As a supply chain analyst,
I want the annual review to automatically trigger and coordinate all upstream agent re-executions in the correct dependency order,
So that I do not have to manually orchestrate 6-10 agent re-runs for every DDS renewal.
```

**Acceptance Criteria:**
- [ ] Defines review dependency graph specifying the execution order for upstream agent re-execution per annual review cycle:
  ```
  1. EUDR-001 Supply Chain Re-Mapping (can run in parallel with Step 2)
  2. EUDR-006 Plot Boundary Re-Verification (can run in parallel with Step 1)
  3. EUDR-012 Document/Certification Re-Authentication (depends on Steps 1 & 2)
  4. EUDR-017 Supplier Risk Re-Scoring (depends on Steps 1, 2, & 3)
  5. EUDR-028 Risk Re-Assessment Engine (depends on Steps 1, 2, 3, & 4)
  6. EUDR-029 Mitigation Measure Re-Evaluation (depends on Step 5; skipped if risk is negligible)
  7. EUDR-030 DDS Regeneration (depends on all previous steps)
  ```
- [ ] Triggers upstream agent re-execution through the Due Diligence Orchestrator (EUDR-026) with review context parameters: review_cycle_id, previous_assessment_id (for comparison), review_type (annual, emergency, out-of-cycle)
- [ ] Tracks completion status of each dependency: Pending, Triggered, In Progress, Completed, Failed, Skipped
- [ ] Executes parallel dependencies concurrently (e.g., EUDR-001 and EUDR-006 in parallel) to minimize total review elapsed time
- [ ] Detects and handles dependency failures: if a dependency fails (e.g., EUDR-028 risk assessment fails), marks downstream dependencies as Blocked, notifies responsible stakeholder, and provides failure details for remediation
- [ ] Supports dependency retry: failed dependencies can be retried without restarting the entire review cycle
- [ ] Identifies shared dependencies across concurrent review cycles: if 10 DDS renewals all require EUDR-001 re-mapping of the same supply chain, triggers a single re-mapping operation and shares the result across all 10 reviews
- [ ] Handles conditional dependencies: EUDR-029 mitigation re-evaluation is only triggered if EUDR-028 risk re-assessment identifies non-negligible risk; if risk is negligible, EUDR-029 is skipped and the review proceeds directly to EUDR-030 DDS regeneration
- [ ] Provides dependency graph visualization: interactive Gantt-style view showing each dependency, its status, elapsed time, and blocking relationships
- [ ] Calculates critical path: identifies which dependencies are on the critical path (any delay in these dependencies directly extends total review time) and highlights them for priority attention
- [ ] Records dependency execution timestamps, durations, and result summaries for audit trail

**Non-Functional Requirements:**
- Coordination Overhead: < 100ms to evaluate dependency graph and determine next executable step
- Parallelism: Support concurrent execution of up to 10 parallel dependencies per review cycle
- Reliability: Dependency state survives system restarts (persisted in PostgreSQL)
- Deduplication: Shared dependency detection across review cycles eliminates 30%+ of redundant upstream agent executions

**Dependencies:**
- EUDR-026 Due Diligence Orchestrator (agent execution trigger interface)
- EUDR-001, EUDR-006, EUDR-012, EUDR-017, EUDR-028, EUDR-029, EUDR-030 (upstream agents to coordinate)
- PostgreSQL (dependency state persistence)

**Estimated Effort:** 4 weeks (1 senior backend engineer)

**Edge Cases:**
- Upstream agent EUDR-028 returns a different risk level than the previous year -- conditional dependency for EUDR-029 evaluated dynamically based on new risk result
- Two concurrent review cycles attempt to trigger the same EUDR-001 re-mapping simultaneously -- deduplication engine merges into single execution
- EUDR-026 orchestrator is temporarily unavailable -- dependency trigger retried with exponential backoff; review cycle paused but not cancelled
- Manual override needed -- administrator can mark a dependency as "manually completed" with evidence attachment and justification, bypassing automated agent execution

---

**P0 Features 5-7: Intelligence, Visualization, and Communication Layer**

> Features 5, 6, and 7 are P0 launch blockers. Without multi-year comparison, compliance calendar, and automated notifications, the scheduling engine cannot deliver operational value. These features transform raw scheduling data into actionable intelligence, visual oversight, and stakeholder coordination.

---

#### Feature 5: Multi-Year Comparison Engine

**User Story:**
```
As a compliance officer,
I want structured year-over-year comparison reports showing exactly what changed between annual review cycles,
So that I can demonstrate continuous compliance improvement to competent authorities and identify emerging risks.
```

**Acceptance Criteria:**
- [ ] Compares current-year review data against the most recent previous-year review data across all compliance dimensions:
  - **Supply Chain Topology**: Nodes added (new suppliers, new intermediaries), nodes removed (suppliers dropped), tier depth changes, new commodities sourced, supply chain complexity score change
  - **Geographic Footprint**: Plots added (new production plots), plots removed, plot boundary changes (area increase/decrease > 5%), new countries of origin, countries removed from sourcing
  - **Risk Profile**: Risk score changes per supplier (significant change defined as >= 10-point shift on 0-100 scale), risk level transitions (Low -> Standard, Standard -> High, etc.), new risk factors identified, risk factors resolved
  - **Deforestation Status**: New deforestation alerts on monitored plots, resolved deforestation alerts, alert density trends (increasing/decreasing/stable), deforestation-free verification status changes
  - **Documentation Completeness**: Article 9 field coverage changes (new fields populated, fields that became incomplete), gap closure progress (gaps resolved since last review), new gaps identified
  - **Mitigation Effectiveness**: Mitigation measures implemented since last review, measures discontinued, effectiveness score changes, new measures added, residual risk changes
  - **Certification Status**: New certifications obtained (FSC, RSPO, etc.), certifications lost/expired, certifications upgraded (e.g., RSPO Credits -> RSPO Mass Balance -> RSPO Segregated)
- [ ] Flags statistically significant changes using configurable materiality thresholds:
  - Supply chain: > 10% change in node count is material
  - Risk: >= 10-point change in composite risk score is material
  - Deforestation: Any new deforestation alert on a monitored plot is material
  - Geography: Any new country of origin or removed country is material
  - Certification: Any certification loss is material
- [ ] Generates executive summary highlighting the top 10 most material changes between cycles, ranked by compliance impact
- [ ] Produces audit-ready comparison reports with:
  - Side-by-side data tables for each dimension
  - Change magnitude and direction indicators (arrows: up/down/stable)
  - Root cause annotations (where available from upstream agents)
  - Provenance hashes (SHA-256) for every compared data point
  - Report generation timestamp and operator identity
- [ ] Supports multi-year trending across 3+ consecutive review cycles: line charts showing risk score trajectory, supply chain growth/contraction, deforestation alert frequency, and compliance readiness score over time
- [ ] Generates change-specific drill-down reports: click on "15 new suppliers added" to see the list of 15 suppliers with full details
- [ ] Exports comparison reports in PDF, CSV, JSON, and HTML formats
- [ ] Integrates with EUDR-030 Documentation Generator to include comparison summary in the renewed DDS as supporting evidence

**Non-Functional Requirements:**
- Performance: Full year-over-year comparison < 30 seconds for 10,000 supply chain entities
- Accuracy: 100% of comparable dimensions covered; zero omissions (validated by dimension coverage matrix)
- Reproducibility: Same review cycle pair produces identical comparison report (deterministic)
- Provenance: SHA-256 hash on every data point ensures tamper-evidence

**Dependencies:**
- EUDR-001 Supply Chain Mapping Master (current and previous supply chain graph snapshots)
- EUDR-006 Plot Boundary Manager (current and previous plot data)
- EUDR-028 Risk Assessment Engine (current and previous risk assessment results)
- EUDR-029 Mitigation Measure Designer (current and previous mitigation records)
- EUDR-030 Documentation Generator (previous DDS content for comparison)
- EUDR-033 Continuous Monitoring Agent (deforestation alert history)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data engineer)

**Edge Cases:**
- First annual review (no previous cycle) -- generate baseline report with "Initial Review" notation; all metrics reported as absolute values without change indicators
- Operator changes commodity scope between years (e.g., adds rubber to existing cocoa portfolio) -- report new commodity as "New in this cycle" rather than showing 100% increase
- Supply chain graph restructured (operator switched from rolling mapping to calendar-aligned) -- comparison engine aligns on closest temporal match rather than exact date match
- Data from previous year is incomplete (operator was onboarding and not all agents were active) -- report missing dimensions with "Previous data unavailable" notation

---

#### Feature 6: Compliance Calendar

**User Story:**
```
As a compliance officer,
I want a unified calendar showing all EUDR temporal obligations across my entire compliance portfolio,
So that I can see everything in one place, identify scheduling conflicts, and plan workload across the year.
```

**Acceptance Criteria:**
- [ ] Aggregates temporal obligations from all relevant sources:
  - DDS renewal deadlines from Review Cycle Manager (F1) -- color: blue
  - Review escalation milestones (90/60/30/7/0 day tiers) -- color: blue gradient (darker = more urgent)
  - Competent authority inspection schedules from EUDR-024 Third-Party Audit Manager -- color: red
  - Certification renewal dates from EUDR-012 Document Authentication -- color: orange
  - Country benchmarking update cycles from EUDR-033 Continuous Monitoring Agent -- color: purple
  - Data freshness expiry dates from EUDR-033 -- color: yellow
  - Stakeholder engagement milestones from EUDR-031 Stakeholder Engagement Tool -- color: green
  - Grievance resolution deadlines from EUDR-032 Grievance Mechanism Manager -- color: pink
  - Review dependency milestones from Entity Review Coordinator (F4) -- color: gray
- [ ] Displays calendar in five views: Day, Week, Month, Quarter, and Year
- [ ] Color-codes events by category (as defined above) with consistent legend
- [ ] Supports filtering by: commodity, country of origin, event type, urgency level, assigned stakeholder, review cycle status
- [ ] Detects scheduling conflicts: weeks with > N reviews due (configurable threshold, default N=10) flagged as overloaded
- [ ] Recommends load-balanced rescheduling: when conflicts detected, suggests redistributing review starts across the review window to smooth workload
- [ ] Calculates workload forecast: projects upcoming review workload per week/month for the next 12 months based on current DDS portfolio and renewal dates
- [ ] Supports capacity planning: compares projected workload against team capacity (configurable FTE count and hours-per-review estimate)
- [ ] Displays event detail panel on click: shows full event details, associated DDS, responsible stakeholder, and linked review cycle
- [ ] Exports calendar in iCalendar (.ics) format for integration with Outlook, Google Calendar, and Apple Calendar
- [ ] Exports calendar data in CSV and JSON formats for reporting and analytics
- [ ] Provides calendar summary statistics: total upcoming events by type, overdue count, this week's critical items, and next month's workload projection

**Non-Functional Requirements:**
- Rendering: < 2 seconds to render calendar with 5,000+ events
- Interactivity: Filter and view changes respond in < 200ms
- Accuracy: 100% of EUDR temporal obligations reflected (zero missing events)
- Accessibility: WCAG 2.1 AA compliance for calendar navigation and color-blind users

**Dependencies:**
- Review Cycle Manager (F1) for DDS renewal dates
- Regulatory Deadline Tracker (F2) for escalation milestones
- EUDR-024 Third-Party Audit Manager (inspection schedules)
- EUDR-012 Document Authentication (certification dates)
- EUDR-033 Continuous Monitoring Agent (data freshness dates, regulatory update cycles)
- EUDR-031 Stakeholder Engagement Tool (engagement milestones)
- EUDR-032 Grievance Mechanism Manager (grievance deadlines)
- Frontend framework: React (existing GL-EUDR-APP frontend)

**Estimated Effort:** 3 weeks (1 frontend engineer, 1 backend engineer)

**Edge Cases:**
- Calendar displays events across year boundary (December/January view) -- seamless rendering without duplication
- Operator has 0 active DDS submissions -- calendar shows non-DDS obligations only (certifications, inspections, etc.)
- Time zone handling: all deadlines stored and displayed in UTC with operator-local time zone overlay option
- Very large operator with 10,000+ calendar events per year -- virtualized calendar rendering with pagination and lazy loading

---

#### Feature 7: Automated Notification System

**User Story:**
```
As a compliance officer,
I want timely, role-appropriate notifications delivered through my preferred channel for every important review event,
So that I and my team always know what needs attention and nothing falls through the cracks.
```

**Acceptance Criteria:**
- [ ] Implements multi-channel notification delivery:
  - Email (SMTP integration): formatted HTML email with event details, action links, and deadline countdown
  - Webhook (HTTP POST): JSON payload for integration with Slack, Microsoft Teams, PagerDuty, and custom systems
  - In-app dashboard notifications: real-time notification feed in GL-EUDR-APP with read/unread tracking
  - SMS (via configurable SMS gateway): for critical escalations only (Tier 4 and Tier 5)
- [ ] Defines notification templates for 12 review lifecycle events:
  1. Review Initiation: "Annual review for DDS {ref} has been scheduled. Review window opens in {days} days."
  2. Task Assignment: "{role}, you have been assigned: {task_description} for DDS {ref}. Due: {date}."
  3. Task Reminder (3-day warning): "Reminder: {task_description} for DDS {ref} is due in 3 days."
  4. Task Reminder (1-day warning): "URGENT: {task_description} for DDS {ref} is due tomorrow."
  5. Task Overdue: "OVERDUE: {task_description} for DDS {ref} was due on {date} and is not complete."
  6. Dependency Completion: "Dependency completed: {agent_name} has finished for review cycle {ref}. Next step: {next_dependency}."
  7. Review Milestone: "Review milestone: DDS {ref} review is {percentage}% complete. {remaining} items remaining."
  8. Review Completion: "Annual review COMPLETE for DDS {ref}. Renewed DDS submitted to EU Information System."
  9. Deadline Escalation -- Tier 2 (60 days): "Standard alert: DDS {ref} renewal due in 60 days. Review status: {status}."
  10. Deadline Escalation -- Tier 3 (30 days): "URGENT: DDS {ref} renewal due in 30 days. Review status: {status}. Management escalation triggered."
  11. Deadline Escalation -- Tier 4 (7 days): "CRITICAL: DDS {ref} renewal due in 7 DAYS. Executive escalation. Immediate action required."
  12. Deadline Escalation -- Tier 5 (Overdue): "DDS {ref} has EXPIRED. Compliance incident recorded. Remediation required immediately."
- [ ] Implements role-based notification routing:
  - Compliance Officer: receives all 12 notification types
  - Supply Chain Analyst: receives task assignment, task reminder, task overdue, dependency completion (for supply chain tasks only)
  - Risk Analyst: receives task assignment, task reminder, task overdue, dependency completion (for risk assessment tasks only)
  - Procurement Manager: receives task assignment, task reminder, task overdue (for supplier outreach tasks only)
  - Management: receives deadline escalation Tier 3+ and review completion only
  - Executive: receives deadline escalation Tier 4+ only
  - External Auditor: receives review completion confirmation only
- [ ] Supports per-user notification preferences: opt-in/opt-out by notification type and channel
- [ ] Implements notification throttling: maximum configurable notifications per user per day (default: 20) with digest option that consolidates notifications into a single daily summary email
- [ ] Tracks notification delivery status: Sent, Delivered, Read (for email with tracking pixel; for in-app), Actioned (user clicked action link)
- [ ] Records full notification audit trail: every notification sent is logged with recipient, channel, timestamp, delivery status, and content hash
- [ ] Supports notification templates in multiple languages: English, German, French, Spanish, Portuguese, and Indonesian (matching EUDR commodity-producing country languages)
- [ ] Consolidates notifications across concurrent review cycles: if an analyst has 5 tasks due this week across 5 different DDS reviews, sends one consolidated "weekly task summary" instead of 5 separate notifications

**Non-Functional Requirements:**
- Dispatch Latency: < 5 seconds from trigger event to notification dispatch
- Delivery Rate: >= 99.5% of notifications delivered (email + webhook + in-app combined)
- Throttling: Prevent alert fatigue by enforcing per-user daily limits
- Internationalization: Notification content rendered in recipient's preferred language

**Dependencies:**
- Email service (SMTP or SES integration)
- Webhook dispatcher (HTTP client with retry logic)
- GL-EUDR-APP in-app notification system
- SMS gateway (Twilio or equivalent, for critical escalations)
- User profile service (notification preferences, language preferences, role assignments)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend engineer)

**Edge Cases:**
- Email delivery fails (SMTP error) -- retry with exponential backoff (3 retries over 1 hour); fall back to in-app notification; log delivery failure
- User has opted out of all notification channels -- record that notification was suppressed (for audit trail) but do not deliver; escalate Tier 4+ to management regardless of user preferences (compliance-critical notifications cannot be suppressed)
- Webhook endpoint returns 5xx error -- retry 3 times with 30-second interval; mark as failed if all retries fail; fall back to email
- User is assigned to multiple roles -- deduplicate notifications (do not send the same notification twice even if user qualifies through multiple role-based routes)
- Notification template references a field that is null in the review context -- render with placeholder text "[data pending]" rather than failing to send

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 8: Review Template Library
- Pre-built review templates for specific industry verticals (chocolate manufacturers, timber importers, automotive/tire companies)
- Templates encode industry-specific best practices and common supply chain patterns
- Operators can customize templates and save as organizational templates
- Templates shared across the GreenLang customer community with anonymized benchmark data

#### Feature 9: AI-Assisted Review Prioritization
- Machine learning model trained on historical review outcomes to predict which reviews are most likely to uncover compliance issues
- Prioritizes reviews based on predicted compliance risk, enabling operators to focus limited review capacity on highest-risk DDS submissions
- Identifies "fast-track" reviews for DDS with minimal year-over-year change (low-risk, stable supply chains)
- Generates review effort estimates based on supply chain complexity and historical completion data

#### Feature 10: Stakeholder Performance Analytics
- Tracks individual and team performance across review cycles: tasks completed on time, average task duration, responsiveness to assignments
- Identifies bottleneck stakeholders who consistently delay review cycles
- Generates review cycle post-mortem reports: what went well, what delayed the review, lessons learned
- Benchmarks operator review performance against anonymized customer cohort

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Performing actual due diligence analysis (EUDR-001 through EUDR-029 handle analysis; this agent schedules and coordinates, not analyzes)
- Automated DDS submission to the EU Information System (EUDR-030 handles submission; this agent triggers regeneration)
- Predictive analytics for future regulatory changes (this agent tracks current deadlines, not predicts future ones)
- Mobile native application (web responsive design only for v1.0; mobile-optimized calendar view)
- Integration with non-EUDR regulatory calendars (e.g., CSRD, CBAM, CSDDD deadlines; defer to Phase 2 cross-regulation calendar)
- Financial impact analysis of missed reviews (defer to GL-EUDR-APP business intelligence module)
- Automated reviewer assignment optimization using constraint satisfaction (manual assignment with recommendation in v1.0)

---

## 7. Technical Requirements

### 7.1 Architecture Overview

```
                                    +---------------------------+
                                    |     GL-EUDR-APP v1.0      |
                                    |   Frontend (React/TS)     |
                                    |   - Compliance Calendar   |
                                    |   - Review Dashboard      |
                                    |   - Notification Center   |
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
| AGENT-EUDR-034        |           | AGENT-EUDR-026            |           | AGENT-EUDR-033        |
| Annual Review         |---------->| Due Diligence             |<--------->| Continuous            |
| Scheduler             |           | Orchestrator              |           | Monitoring Agent      |
|                       |           |                           |           |                       |
| - Review Cycle Mgr    |           | - Agent Execution         |           | - Regulatory Updates  |
| - Deadline Tracker    |           | - Workflow Management     |           | - Data Freshness      |
| - Checklist Generator |           | - Dependency Resolution   |           | - Country Benchmarks  |
| - Entity Coordinator  |           |                           |           | - Deforestation Alerts|
| - Comparison Engine   |           +---------------------------+           +-----------------------+
| - Compliance Calendar |                       |
| - Notification System |                       |
+-----------+-----------+                       |
            |                                   |
            |           +-----+-----+-----+-----+-----+-----+-----+
            |           |     |     |     |     |     |     |     |
            |         001   006   012   017   028   029   030   024
            |        (SCM) (Plot)(Doc) (Risk)(RAE) (MMD)(DGen)(Audit)
            |
+-----------v-----------+           +---------------------------+           +---------------------------+
| Notification          |           | Calendar Service          |           | Comparison Service        |
| Dispatch Engine       |           |                           |           |                           |
|                       |           | - Event Aggregation       |           | - Snapshot Retrieval      |
| - Email (SMTP/SES)   |           | - Conflict Detection      |           | - Dimension Comparison    |
| - Webhook (HTTP)      |           | - Workload Forecasting    |           | - Materiality Analysis    |
| - In-App (WebSocket)  |           | - iCal Export             |           | - Report Generation       |
| - SMS (Twilio)        |           +---------------------------+           +---------------------------+
+-----------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/annual_review_scheduler/
    __init__.py                          # Public API exports
    config.py                            # AnnualReviewSchedulerConfig with GL_EUDR_ARS_ env prefix
    models.py                            # Pydantic v2 models for review cycles, checklists, calendars
    review_cycle_manager.py              # ReviewCycleManager: lifecycle tracking, status transitions
    deadline_tracker.py                  # DeadlineTracker: deadline calculation, 5-tier escalation
    checklist_generator.py               # ChecklistGenerator: commodity-specific, context-aware checklists
    entity_coordinator.py                # EntityReviewCoordinator: dependency graph, parallel execution
    comparison_engine.py                 # MultiYearComparisonEngine: year-over-year analysis
    compliance_calendar.py               # ComplianceCalendar: event aggregation, conflict detection
    notification_engine.py               # NotificationEngine: multi-channel dispatch, throttling
    notification_templates.py            # NotificationTemplates: 12 lifecycle event templates, i18n
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains for review artifacts
    metrics.py                           # 15 Prometheus self-monitoring metrics
    setup.py                             # AnnualReviewSchedulerService facade
    api/
        __init__.py
        router.py                        # FastAPI router (30+ endpoints)
        review_routes.py                 # Review cycle CRUD and status management
        deadline_routes.py               # Deadline tracking and escalation endpoints
        checklist_routes.py              # Checklist generation and completion tracking
        coordinator_routes.py            # Dependency coordination and status endpoints
        comparison_routes.py             # Multi-year comparison report endpoints
        calendar_routes.py               # Compliance calendar event and export endpoints
        notification_routes.py           # Notification management and preference endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Review Cycle Status
class ReviewCycleStatus(str, Enum):
    NOT_STARTED = "not_started"      # DDS exists but review window not yet open
    SCHEDULED = "scheduled"          # Review window opened, review planned
    IN_PROGRESS = "in_progress"      # Review actively underway
    UNDER_REVIEW = "under_review"    # All tasks complete, final review/approval pending
    COMPLETED = "completed"          # Review done, renewed DDS submitted
    ARCHIVED = "archived"            # Historical review cycle
    OVERDUE = "overdue"              # 12-month deadline passed without completion

# Review Cycle
class ReviewCycle(BaseModel):
    cycle_id: str                    # UUID, unique identifier
    dds_reference: str               # Reference to the DDS being renewed
    operator_id: str                 # Operator who owns this DDS
    commodity: EUDRCommodity         # Primary commodity (cattle/cocoa/coffee/palm_oil/rubber/soya/wood)
    country_of_origin: str           # ISO 3166-1 alpha-2 country code
    supply_chain_graph_id: str       # Reference to EUDR-001 supply chain graph
    original_dds_submission_date: datetime  # When the DDS was originally submitted
    renewal_deadline: datetime       # original_dds_submission_date + 12 months
    review_window_start: datetime    # renewal_deadline - review_preparation_days (default 90)
    review_completion_target: datetime  # renewal_deadline - completion_target_days (default 30)
    critical_escalation_date: datetime  # renewal_deadline - critical_days (default 7)
    status: ReviewCycleStatus
    review_type: ReviewType          # ANNUAL, EMERGENCY, OUT_OF_CYCLE
    due_diligence_type: DueDiligenceType  # FULL, SIMPLIFIED (based on Art. 29 country risk)
    checklist_id: Optional[str]      # Reference to generated checklist
    comparison_report_id: Optional[str]  # Reference to multi-year comparison report
    renewed_dds_reference: Optional[str]  # Reference to the renewed DDS once generated
    completion_percentage: float     # 0.0 to 100.0
    assigned_stakeholders: List[StakeholderAssignment]
    dependency_statuses: List[DependencyStatus]
    escalation_history: List[EscalationEvent]
    previous_cycle_id: Optional[str]  # Link to the previous annual review cycle
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

# Checklist Item
class ChecklistItem(BaseModel):
    item_id: str                     # UUID
    checklist_id: str                # Parent checklist
    section: ChecklistSection        # ARTICLE_9, ARTICLE_10, ARTICLE_11, ARTICLE_12, CROSS_CUTTING
    article_reference: str           # e.g., "Art. 9(1)(a)", "Art. 10(2)(c)"
    description: str                 # Human-readable description of the verification step
    commodity_specific: bool         # True if this item is commodity-specific
    classification: ItemClassification  # MANDATORY, RECOMMENDED, OPTIONAL
    assigned_role: str               # compliance_officer, supply_chain_analyst, risk_analyst, etc.
    upstream_agent: Optional[str]    # EUDR agent responsible (e.g., "EUDR-001", "EUDR-028")
    estimated_duration_hours: float  # Estimated time to complete
    depends_on: List[str]            # IDs of checklist items this item depends on
    status: ChecklistItemStatus      # NOT_STARTED, IN_PROGRESS, COMPLETED, SKIPPED, BLOCKED
    skip_justification: Optional[str]  # Required if status = SKIPPED and classification = MANDATORY
    evidence_links: List[str]        # Links to evidence artifacts for completed items
    completed_by: Optional[str]      # User who completed the item
    completed_at: Optional[datetime]
    notes: Optional[str]

# Escalation Event
class EscalationEvent(BaseModel):
    event_id: str
    cycle_id: str
    escalation_tier: int             # 1 (Planning), 2 (Standard), 3 (Urgent), 4 (Critical), 5 (Overdue)
    triggered_at: datetime
    notification_ids: List[str]      # References to notifications dispatched
    recipients: List[str]            # User IDs notified
    review_status_at_trigger: ReviewCycleStatus
    days_until_deadline: int

# Calendar Event
class CalendarEvent(BaseModel):
    event_id: str
    event_type: CalendarEventType    # DDS_RENEWAL, ESCALATION, INSPECTION, CERTIFICATION,
                                     # BENCHMARKING, FRESHNESS, ENGAGEMENT, GRIEVANCE, DEPENDENCY
    source_agent: str                # Agent that generated this event (e.g., "EUDR-034", "EUDR-024")
    title: str                       # Human-readable event title
    description: str                 # Event details
    event_date: datetime             # When the event occurs/is due
    urgency: EventUrgency            # LOW, MEDIUM, HIGH, CRITICAL
    associated_dds: Optional[str]    # DDS reference if applicable
    commodity: Optional[EUDRCommodity]
    country: Optional[str]
    assigned_to: Optional[str]       # User ID if event has an assignee
    color_category: str              # Color code for calendar rendering
    metadata: Dict[str, Any]

# Multi-Year Comparison Report
class ComparisonReport(BaseModel):
    report_id: str
    operator_id: str
    dds_reference: str
    commodity: EUDRCommodity
    current_cycle_id: str
    previous_cycle_id: str
    current_cycle_date: datetime
    previous_cycle_date: datetime
    dimensions: Dict[str, DimensionComparison]  # supply_chain, geographic, risk, etc.
    material_changes: List[MaterialChange]  # Changes exceeding materiality thresholds
    executive_summary: str           # Top 10 material changes narrative
    trend_data: Optional[Dict[str, List[TrendPoint]]]  # Multi-year trend if 3+ cycles
    provenance_hash: str             # SHA-256 of the complete report
    generated_at: datetime
```

### 7.4 Database Schema (New Migration: V119)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_annual_review_scheduler;

-- Review cycles (one per DDS per annual review period)
CREATE TABLE eudr_annual_review_scheduler.review_cycles (
    cycle_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_reference VARCHAR(100) NOT NULL,
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    country_of_origin CHAR(2) NOT NULL,
    supply_chain_graph_id UUID,
    original_dds_submission_date TIMESTAMPTZ NOT NULL,
    renewal_deadline TIMESTAMPTZ NOT NULL,
    review_window_start TIMESTAMPTZ NOT NULL,
    review_completion_target TIMESTAMPTZ NOT NULL,
    critical_escalation_date TIMESTAMPTZ NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'not_started',
    review_type VARCHAR(30) NOT NULL DEFAULT 'annual',
    due_diligence_type VARCHAR(30) NOT NULL DEFAULT 'full',
    checklist_id UUID,
    comparison_report_id UUID,
    renewed_dds_reference VARCHAR(100),
    completion_percentage NUMERIC(5,2) DEFAULT 0.0,
    previous_cycle_id UUID,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

-- Review checklists
CREATE TABLE eudr_annual_review_scheduler.review_checklists (
    checklist_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_id UUID NOT NULL REFERENCES eudr_annual_review_scheduler.review_cycles(cycle_id),
    commodity VARCHAR(50) NOT NULL,
    country_risk_level VARCHAR(20) NOT NULL,
    due_diligence_type VARCHAR(30) NOT NULL,
    total_items INTEGER DEFAULT 0,
    mandatory_items INTEGER DEFAULT 0,
    completed_items INTEGER DEFAULT 0,
    completion_percentage NUMERIC(5,2) DEFAULT 0.0,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Checklist items
CREATE TABLE eudr_annual_review_scheduler.checklist_items (
    item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checklist_id UUID NOT NULL REFERENCES eudr_annual_review_scheduler.review_checklists(checklist_id),
    section VARCHAR(50) NOT NULL,
    article_reference VARCHAR(30),
    description TEXT NOT NULL,
    commodity_specific BOOLEAN DEFAULT FALSE,
    classification VARCHAR(20) NOT NULL DEFAULT 'mandatory',
    assigned_role VARCHAR(50),
    upstream_agent VARCHAR(30),
    estimated_duration_hours NUMERIC(5,2),
    status VARCHAR(20) NOT NULL DEFAULT 'not_started',
    skip_justification TEXT,
    evidence_links JSONB DEFAULT '[]',
    completed_by VARCHAR(100),
    completed_at TIMESTAMPTZ,
    notes TEXT,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Checklist item dependencies
CREATE TABLE eudr_annual_review_scheduler.checklist_dependencies (
    dependency_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id UUID NOT NULL REFERENCES eudr_annual_review_scheduler.checklist_items(item_id),
    depends_on_item_id UUID NOT NULL REFERENCES eudr_annual_review_scheduler.checklist_items(item_id),
    CONSTRAINT unique_dependency UNIQUE (item_id, depends_on_item_id)
);

-- Review dependencies (upstream agent coordination)
CREATE TABLE eudr_annual_review_scheduler.review_dependencies (
    dependency_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_id UUID NOT NULL REFERENCES eudr_annual_review_scheduler.review_cycles(cycle_id),
    agent_id VARCHAR(30) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    execution_order INTEGER NOT NULL,
    can_parallel_with JSONB DEFAULT '[]',
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    triggered_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,
    failure_reason TEXT,
    result_reference VARCHAR(200),
    retry_count INTEGER DEFAULT 0,
    is_conditional BOOLEAN DEFAULT FALSE,
    condition_met BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Escalation events (hypertable)
CREATE TABLE eudr_annual_review_scheduler.escalation_events (
    event_id UUID DEFAULT gen_random_uuid(),
    cycle_id UUID NOT NULL,
    escalation_tier INTEGER NOT NULL,
    notification_ids JSONB DEFAULT '[]',
    recipients JSONB DEFAULT '[]',
    review_status_at_trigger VARCHAR(30),
    days_until_deadline INTEGER,
    triggered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_annual_review_scheduler.escalation_events', 'triggered_at');

-- Calendar events (hypertable)
CREATE TABLE eudr_annual_review_scheduler.calendar_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    source_agent VARCHAR(30) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    event_date TIMESTAMPTZ NOT NULL,
    urgency VARCHAR(20) NOT NULL DEFAULT 'medium',
    associated_dds VARCHAR(100),
    commodity VARCHAR(50),
    country CHAR(2),
    assigned_to VARCHAR(100),
    color_category VARCHAR(30),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_annual_review_scheduler.calendar_events', 'event_date');

-- Multi-year comparison reports
CREATE TABLE eudr_annual_review_scheduler.comparison_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    dds_reference VARCHAR(100) NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    current_cycle_id UUID NOT NULL REFERENCES eudr_annual_review_scheduler.review_cycles(cycle_id),
    previous_cycle_id UUID NOT NULL,
    current_cycle_date TIMESTAMPTZ NOT NULL,
    previous_cycle_date TIMESTAMPTZ NOT NULL,
    dimensions JSONB NOT NULL DEFAULT '{}',
    material_changes JSONB DEFAULT '[]',
    executive_summary TEXT,
    trend_data JSONB,
    provenance_hash VARCHAR(64) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Notification log (hypertable)
CREATE TABLE eudr_annual_review_scheduler.notification_log (
    log_id UUID DEFAULT gen_random_uuid(),
    cycle_id UUID,
    notification_type VARCHAR(50) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    recipient VARCHAR(100) NOT NULL,
    template_name VARCHAR(50),
    subject VARCHAR(500),
    content_hash VARCHAR(64),
    delivery_status VARCHAR(20) NOT NULL DEFAULT 'sent',
    delivered_at TIMESTAMPTZ,
    read_at TIMESTAMPTZ,
    actioned_at TIMESTAMPTZ,
    language VARCHAR(10) DEFAULT 'en',
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_annual_review_scheduler.notification_log', 'sent_at');

-- Review cycle status history (audit trail, hypertable)
CREATE TABLE eudr_annual_review_scheduler.review_status_history (
    history_id UUID DEFAULT gen_random_uuid(),
    cycle_id UUID NOT NULL,
    previous_status VARCHAR(30),
    new_status VARCHAR(30) NOT NULL,
    changed_by VARCHAR(100),
    change_reason TEXT,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_annual_review_scheduler.review_status_history', 'changed_at');

-- Indexes
CREATE INDEX idx_cycles_operator ON eudr_annual_review_scheduler.review_cycles(operator_id);
CREATE INDEX idx_cycles_dds ON eudr_annual_review_scheduler.review_cycles(dds_reference);
CREATE INDEX idx_cycles_status ON eudr_annual_review_scheduler.review_cycles(status);
CREATE INDEX idx_cycles_deadline ON eudr_annual_review_scheduler.review_cycles(renewal_deadline);
CREATE INDEX idx_cycles_commodity ON eudr_annual_review_scheduler.review_cycles(commodity);
CREATE INDEX idx_cycles_country ON eudr_annual_review_scheduler.review_cycles(country_of_origin);
CREATE INDEX idx_checklists_cycle ON eudr_annual_review_scheduler.review_checklists(cycle_id);
CREATE INDEX idx_items_checklist ON eudr_annual_review_scheduler.checklist_items(checklist_id);
CREATE INDEX idx_items_status ON eudr_annual_review_scheduler.checklist_items(status);
CREATE INDEX idx_deps_cycle ON eudr_annual_review_scheduler.review_dependencies(cycle_id);
CREATE INDEX idx_deps_status ON eudr_annual_review_scheduler.review_dependencies(status);
CREATE INDEX idx_calendar_type ON eudr_annual_review_scheduler.calendar_events(event_type);
CREATE INDEX idx_calendar_dds ON eudr_annual_review_scheduler.calendar_events(associated_dds);
CREATE INDEX idx_comparison_operator ON eudr_annual_review_scheduler.comparison_reports(operator_id);
CREATE INDEX idx_comparison_dds ON eudr_annual_review_scheduler.comparison_reports(dds_reference);
CREATE INDEX idx_notification_cycle ON eudr_annual_review_scheduler.notification_log(cycle_id);
CREATE INDEX idx_notification_recipient ON eudr_annual_review_scheduler.notification_log(recipient);
```

### 7.5 API Endpoints (30+)

| Method | Path | Description |
|--------|------|-------------|
| **Review Cycle Management** | | |
| POST | `/v1/reviews` | Create a new review cycle (manual or system-triggered) |
| GET | `/v1/reviews` | List review cycles with filters (status, commodity, country, operator, deadline range) |
| GET | `/v1/reviews/{cycle_id}` | Get review cycle details with checklist, dependencies, and comparison status |
| PUT | `/v1/reviews/{cycle_id}/status` | Update review cycle status (with transition validation) |
| GET | `/v1/reviews/pipeline` | Get review pipeline dashboard data (counts by status, urgency distribution) |
| GET | `/v1/reviews/workload` | Get workload forecast per week/month for next 12 months |
| **Deadline Tracking** | | |
| GET | `/v1/deadlines` | List all upcoming deadlines with escalation tier information |
| GET | `/v1/deadlines/overdue` | List overdue review cycles requiring immediate attention |
| GET | `/v1/deadlines/analytics` | Get deadline analytics (on-time rate, average completion time, trending) |
| POST | `/v1/deadlines/simulate` | Simulate deadline impact of adding N new DDS submissions |
| **Checklist Management** | | |
| POST | `/v1/reviews/{cycle_id}/checklist/generate` | Generate commodity-specific review checklist |
| GET | `/v1/reviews/{cycle_id}/checklist` | Get review checklist with item statuses |
| PUT | `/v1/reviews/{cycle_id}/checklist/items/{item_id}` | Update checklist item status (complete, skip, block) |
| GET | `/v1/reviews/{cycle_id}/checklist/completion` | Get checklist completion report with evidence links |
| **Dependency Coordination** | | |
| GET | `/v1/reviews/{cycle_id}/dependencies` | Get dependency graph with statuses |
| POST | `/v1/reviews/{cycle_id}/dependencies/{dep_id}/trigger` | Manually trigger a dependency execution |
| POST | `/v1/reviews/{cycle_id}/dependencies/{dep_id}/retry` | Retry a failed dependency |
| PUT | `/v1/reviews/{cycle_id}/dependencies/{dep_id}/override` | Mark dependency as manually completed (admin) |
| GET | `/v1/reviews/{cycle_id}/dependencies/critical-path` | Get critical path analysis |
| **Multi-Year Comparison** | | |
| POST | `/v1/reviews/{cycle_id}/comparison/generate` | Generate year-over-year comparison report |
| GET | `/v1/reviews/{cycle_id}/comparison` | Get comparison report |
| GET | `/v1/reviews/{cycle_id}/comparison/trends` | Get multi-year trend data (3+ cycles) |
| GET | `/v1/reviews/{cycle_id}/comparison/export` | Export comparison report (PDF, CSV, JSON, HTML) |
| **Compliance Calendar** | | |
| GET | `/v1/calendar` | Get compliance calendar events with filters (date range, type, commodity) |
| GET | `/v1/calendar/conflicts` | Get scheduling conflict analysis |
| GET | `/v1/calendar/forecast` | Get workload forecast and capacity planning data |
| GET | `/v1/calendar/export` | Export calendar (iCal, CSV, JSON) |
| **Notifications** | | |
| GET | `/v1/notifications` | Get notification history with filters |
| GET | `/v1/notifications/preferences` | Get current user's notification preferences |
| PUT | `/v1/notifications/preferences` | Update notification preferences |
| GET | `/v1/notifications/stats` | Get notification analytics (delivery rate, read rate, action rate) |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (15)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_ars_review_cycles_created_total` | Counter | Review cycles created by type (annual, emergency, out_of_cycle) |
| 2 | `gl_eudr_ars_review_cycles_completed_total` | Counter | Review cycles completed successfully |
| 3 | `gl_eudr_ars_review_cycles_overdue_total` | Counter | Review cycles that entered overdue status |
| 4 | `gl_eudr_ars_escalation_events_total` | Counter | Escalation events fired by tier (1-5) |
| 5 | `gl_eudr_ars_checklists_generated_total` | Counter | Review checklists generated by commodity |
| 6 | `gl_eudr_ars_checklist_items_completed_total` | Counter | Checklist items completed by section |
| 7 | `gl_eudr_ars_dependencies_triggered_total` | Counter | Upstream agent dependencies triggered by agent |
| 8 | `gl_eudr_ars_dependencies_failed_total` | Counter | Upstream agent dependencies that failed by agent |
| 9 | `gl_eudr_ars_comparison_reports_generated_total` | Counter | Multi-year comparison reports generated |
| 10 | `gl_eudr_ars_notifications_sent_total` | Counter | Notifications dispatched by channel and type |
| 11 | `gl_eudr_ars_review_cycle_duration_seconds` | Histogram | Time from review initiation to completion |
| 12 | `gl_eudr_ars_checklist_generation_duration_seconds` | Histogram | Checklist generation latency |
| 13 | `gl_eudr_ars_comparison_generation_duration_seconds` | Histogram | Comparison report generation latency |
| 14 | `gl_eudr_ars_active_review_cycles` | Gauge | Currently active review cycles by status |
| 15 | `gl_eudr_ars_days_until_nearest_deadline` | Gauge | Days until the nearest DDS renewal deadline across all operators |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Scheduling | APScheduler + custom deadline engine | APScheduler for cron-style periodic scans; custom engine for precise deadline calculations |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for events and logs |
| Cache | Redis | Review cycle status caching, calendar event caching, notification deduplication |
| Object Storage | S3 | Comparison reports, checklist exports, notification archives |
| Email | SMTP / Amazon SES | Notification dispatch |
| Webhook | HTTP client (httpx) | Integration with Slack, Teams, custom endpoints |
| SMS | Twilio API | Critical escalation notifications |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Calendar Export | icalendar library | iCalendar (.ics) format generation |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based review access control |
| Monitoring | Prometheus + Grafana | 15 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

The following permissions will be registered in the GreenLang PERMISSION_MAP for RBAC enforcement:

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-ars:reviews:read` | View review cycles and pipeline dashboards | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ars:reviews:write` | Create, update, and manage review cycles | Compliance Officer, Admin |
| `eudr-ars:reviews:status` | Update review cycle status transitions | Compliance Officer, Admin |
| `eudr-ars:deadlines:read` | View deadlines and escalation status | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ars:deadlines:simulate` | Run deadline simulation scenarios | Analyst, Compliance Officer, Admin |
| `eudr-ars:checklists:read` | View review checklists and completion status | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ars:checklists:write` | Generate checklists, update item statuses | Analyst, Compliance Officer, Admin |
| `eudr-ars:checklists:skip` | Skip mandatory checklist items (with justification) | Compliance Officer, Admin |
| `eudr-ars:dependencies:read` | View dependency graphs and status | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ars:dependencies:trigger` | Manually trigger upstream agent execution | Analyst, Compliance Officer, Admin |
| `eudr-ars:dependencies:override` | Mark dependencies as manually completed | Admin |
| `eudr-ars:comparison:read` | View multi-year comparison reports | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ars:comparison:generate` | Generate comparison reports | Analyst, Compliance Officer, Admin |
| `eudr-ars:comparison:export` | Export comparison reports (PDF, CSV, JSON) | Analyst, Compliance Officer, Admin |
| `eudr-ars:calendar:read` | View compliance calendar | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ars:calendar:export` | Export calendar data (iCal, CSV, JSON) | Analyst, Compliance Officer, Admin |
| `eudr-ars:notifications:read` | View notification history and analytics | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ars:notifications:preferences` | Manage own notification preferences | All authenticated users |
| `eudr-ars:audit:read` | View review status history and audit trails | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| AGENT-EUDR-030 Documentation Generator | DDS submission records | DDS reference numbers, submission dates, commodity, country -> review cycle creation and deadline calculation |
| AGENT-EUDR-001 Supply Chain Mapping Master | Supply chain graph data | Current and historical supply chain snapshots -> multi-year comparison, checklist context |
| AGENT-EUDR-006 Plot Boundary Manager | Plot boundary data | Current and historical plot data -> multi-year geographic comparison, checklist verification items |
| AGENT-EUDR-028 Risk Assessment Engine | Risk assessment results | Current and previous risk scores -> multi-year risk comparison, checklist context adaptation |
| AGENT-EUDR-029 Mitigation Measure Designer | Mitigation measure records | Current and previous mitigation data -> comparison, checklist mitigation section |
| AGENT-EUDR-033 Continuous Monitoring Agent | Regulatory updates, data freshness, deforestation alerts | Country benchmarking changes -> emergency review triggers; data freshness dates -> calendar; deforestation alert history -> checklist adaptation |
| AGENT-EUDR-024 Third-Party Audit Manager | Audit schedules | Inspection dates, audit windows -> compliance calendar events |
| AGENT-EUDR-012 Document Authentication | Certification records | Certification expiry dates -> compliance calendar, checklist certification verification |
| AGENT-EUDR-031 Stakeholder Engagement Tool | Engagement milestones | Engagement dates -> compliance calendar events |
| AGENT-EUDR-032 Grievance Mechanism Manager | Grievance deadlines | Resolution deadlines -> compliance calendar events |
| AGENT-EUDR-017 Supplier Risk Scorer | Supplier risk scores | Current and historical scores -> multi-year comparison, entity coordination trigger |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-026 Due Diligence Orchestrator | Review coordination | Review initiation commands, dependency trigger requests -> orchestrated agent re-execution |
| AGENT-EUDR-030 Documentation Generator | DDS regeneration trigger | Review completion -> DDS regeneration request with updated data |
| GL-EUDR-APP v1.0 | API integration | Calendar data, review dashboards, notification feed -> frontend visualization |
| External Auditors | Read-only API + exports | Review history, completion records, comparison reports -> audit evidence |
| Management Dashboards | API integration | Pipeline status, overdue counts, workload forecasting -> executive reporting |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Annual Review Initiation and Completion (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Annual Review Scheduler" module
3. Dashboard shows review pipeline: 12 Scheduled, 8 In Progress, 3 Under Review, 2 Overdue
4. System has automatically created review cycles for all DDS approaching 90-day window
5. Officer clicks on a Scheduled review for cocoa DDS (DDS-2025-0142, Ghana)
6. System displays auto-generated commodity-specific checklist (48 items for cocoa/full DD)
7. Officer reviews checklist sections: Art. 9 (12 items), Art. 10 (14 items), Art. 11 (8 items),
   Art. 12 (6 items), Cross-cutting (8 items)
8. Officer clicks "Initiate Review" -> status moves to In Progress
9. System triggers dependency chain through EUDR-026:
   - EUDR-001 supply chain re-mapping (parallel)
   - EUDR-006 plot boundary re-verification (parallel)
   -> EUDR-012 document re-authentication
   -> EUDR-017 supplier risk re-scoring
   -> EUDR-028 risk re-assessment
   -> EUDR-029 mitigation re-evaluation (if non-negligible risk)
   -> EUDR-030 DDS regeneration
10. Officer monitors dependency progress on Gantt-style visualization
11. As dependencies complete, associated checklist items auto-complete with evidence links
12. Officer completes remaining manual checklist items (supplier outreach confirmation, etc.)
13. All mandatory items complete -> Officer clicks "Submit for Review" -> status moves to Under Review
14. System generates multi-year comparison report (vs. previous year)
15. Officer reviews comparison: 3 new suppliers, 2 risk score increases, 1 new certification
16. Officer approves comparison findings -> EUDR-030 generates renewed DDS
17. Review cycle marked Completed -> notifications sent to all stakeholders
18. Review cycle archived with full audit trail
```

#### Flow 2: Deadline Escalation Path (Multi-Stakeholder)

```
Day 0: DDS-2025-0287 (palm oil, Indonesia) submitted
Day 275 (90 days before expiry):
  - Tier 1 Planning escalation fires
  - Compliance Officer receives: "Annual review for DDS-2025-0287 has been scheduled"
  - Review cycle auto-created with status Scheduled
  - Checklist generated (52 items for palm oil/full DD/high-risk country)
Day 300:
  - Compliance Officer initiates review -> status In Progress
  - Supply chain analyst assigned: re-map palm oil supply chain
  - Procurement manager assigned: contact 45 plantation suppliers for updated GPS data
Day 305 (60 days before expiry):
  - Tier 2 Standard Alert fires
  - All assigned stakeholders receive: "DDS-2025-0287 renewal due in 60 days"
  - Progress: 35% complete (supply chain re-mapping done, 20/45 suppliers responded)
Day 330:
  - Risk analyst completes risk re-assessment via EUDR-028
  - 2 suppliers show increased deforestation risk -> EUDR-029 mitigation triggered
Day 335 (30 days before expiry):
  - Tier 3 Urgent Alert fires
  - Progress: 72% complete (mitigation measures in progress, 38/45 suppliers responded)
  - Management receives escalation: "DDS-2025-0287 renewal at 72%, 30 days remaining"
Day 350:
  - All checklist items complete, comparison report generated
  - DDS regenerated by EUDR-030 and submitted to EU Information System
  - Review cycle marked Completed (15 days before expiry)
  - No Tier 4 or Tier 5 escalation needed
```

#### Flow 3: Emergency Review Triggered by Country Reclassification

```
1. EUDR-033 Continuous Monitoring Agent detects EC country benchmarking update:
   Country X reclassified from Low to Standard risk
2. EUDR-034 receives country reclassification event
3. System queries all active DDS submissions with country_of_origin = Country X
4. Finds 15 DDS submissions currently using simplified due diligence
5. System creates 15 emergency review cycles with:
   - review_type = EMERGENCY
   - due_diligence_type changed from SIMPLIFIED to FULL
   - Accelerated deadline: 30 days from reclassification date
6. Checklists expanded from simplified (20 items) to full (45+ items)
7. Compliance Officer receives CRITICAL notification:
   "EMERGENCY: Country X reclassified to Standard risk. 15 DDS require full DD review within 30 days."
8. Officer prioritizes emergency reviews; system identifies shared suppliers across 15 DDS
9. Entity Coordinator deduplicates shared dependencies:
   instead of 15 separate EUDR-001 re-mappings, triggers 3 (one per shared supply chain graph)
10. Emergency reviews processed with accelerated dependency execution
11. All 15 DDS renewed under full due diligence before 30-day deadline
```

#### Flow 4: Multi-Year Comparison Review (External Auditor)

```
1. External auditor Dr. Hofmann receives read-only access to operator's review history
2. Navigates to "Review History" in GL-EUDR-APP
3. Selects DDS-2024-0056 (cocoa, Cote d'Ivoire) -> sees 3 annual review cycles (2024, 2025, 2026)
4. Opens 2026 multi-year comparison report
5. Report shows:
   - Supply chain: +12 new cooperative members, -3 cooperatives removed, tier depth unchanged at 5
   - Geographic: +8 new plots registered, 2 plot boundaries expanded (total +15 hectares)
   - Risk: composite risk score decreased from 62 to 54 (improvement); 2 suppliers moved Low->Standard
   - Deforestation: 1 deforestation alert resolved; 0 new alerts
   - Certifications: 3 cooperatives obtained Rainforest Alliance certification
   - Mitigation: enhanced monitoring implemented for 2 suppliers with increased risk
6. Auditor clicks on "2 suppliers moved Low->Standard" -> drill-down shows supplier details
7. Auditor verifies that enhanced due diligence was applied to these suppliers
8. Auditor clicks on "3-year trend" -> line chart shows declining composite risk score over 3 years
9. Auditor exports comparison report as PDF with provenance hashes for audit file
10. Auditor notes in audit report: "Operator demonstrates systematic annual review process with
    documented year-over-year improvement in risk profile. Evidence supports continuous compliance."
```

### 8.2 Key Screen Descriptions

**Review Pipeline Dashboard:**
- Summary cards: Total Active Reviews, Scheduled, In Progress, Under Review, Completed (this quarter), Overdue
- Pipeline funnel visualization: Not Started -> Scheduled -> In Progress -> Under Review -> Completed
- Priority queue: reviews ranked by urgency (days to deadline), risk level, and commodity
- Quick filters: commodity, country, status, assigned stakeholder, deadline range
- Workload histogram: reviews per week for next 12 weeks

**Review Detail View:**
- Header: DDS reference, commodity, country, deadline countdown, current status badge
- Tabs: Checklist | Dependencies | Comparison | Notifications | History
- Checklist tab: regulatory-section-organized checklist with progress bar, item status icons, evidence links
- Dependencies tab: Gantt-style dependency graph with parallel/sequential lanes, status colors, critical path highlighting
- Comparison tab: year-over-year comparison dashboard with dimension cards, change indicators, and drill-down links
- Notifications tab: timeline of all notifications sent for this review cycle
- History tab: complete status transition audit trail with timestamps, actors, and reasons

**Compliance Calendar View:**
- Full-screen calendar with month view as default
- Left sidebar: filter panel (commodity, country, event type, urgency, assigned stakeholder)
- Color-coded events by category with legend at bottom
- Conflict indicators: weeks with > 10 reviews show warning badge
- Right panel: event detail on click with full context and action links
- Top bar: view switcher (Day/Week/Month/Quarter/Year), export button, workload forecast toggle

**Multi-Year Comparison Report View:**
- Executive summary at top: top 10 material changes with severity badges
- Dimension tabs: Supply Chain | Geography | Risk | Deforestation | Documentation | Mitigation | Certification
- Each dimension: side-by-side tables, change magnitude indicators, trend sparklines
- Provenance footer: SHA-256 hash, generation timestamp, data sources referenced
- Export bar: PDF, CSV, JSON, HTML export buttons

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 7 P0 features (Features 1-7) implemented and tested
  - [ ] Feature 1: Review Cycle Manager -- lifecycle tracking, status transitions, intelligent scheduling, pipeline dashboard
  - [ ] Feature 2: Regulatory Deadline Tracker -- 12-month deadline calculation, 5-tier escalation, emergency review protocol
  - [ ] Feature 3: Review Checklist Generator -- 7 commodity templates, simplified/full adaptation, Article 8/9/10/11 coverage
  - [ ] Feature 4: Entity Review Coordinator -- dependency graph, parallel execution, shared dependency deduplication, critical path
  - [ ] Feature 5: Multi-Year Comparison Engine -- 7 dimensions, materiality thresholds, executive summary, trend analysis
  - [ ] Feature 6: Compliance Calendar -- event aggregation from 8 sources, conflict detection, workload forecasting, export
  - [ ] Feature 7: Automated Notification System -- 4 channels, 12 templates, role-based routing, throttling, i18n
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC integrated with 19 permissions)
- [ ] Performance targets met (< 5s review scheduling for 1,000 DDS, < 3s checklist generation, < 30s comparison for 10,000 entities)
- [ ] All 7 commodity-specific checklist templates validated by regulatory subject matter expert
- [ ] Deadline calculations validated against reference calendar for all edge cases (leap years, month-end, year boundary)
- [ ] 5-tier escalation system tested end-to-end with simulated deadlines
- [ ] Dependency coordination tested with all upstream agents (EUDR-001, 006, 012, 017, 028, 029, 030)
- [ ] Multi-year comparison validated against manually compiled comparison for 3 golden test operators
- [ ] Notification delivery tested across all 4 channels (email, webhook, in-app, SMS)
- [ ] API documentation complete (OpenAPI spec for 30+ endpoints)
- [ ] Database migration V119 tested and validated
- [ ] Integration with EUDR-026 orchestrator verified for review-triggered agent re-execution
- [ ] 5 beta customers successfully completed at least one annual review cycle using the system
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 100+ review cycles created by customers across all 7 commodities
- Average review cycle initiation accuracy: 100% (all cycles initiated before 90-day window closes)
- Zero missed Tier 1 (Planning) escalations
- Checklist generation satisfaction: >= 80% of users rate checklists as "useful" or "very useful"
- < 5 support tickets per customer related to annual review scheduling
- p95 API response time < 200ms in production

**60 Days:**
- 300+ review cycles active across customer base
- Average review cycle duration < 35 days (target: 30 days within 90 days of launch)
- >= 95% on-time DDS renewal rate (target: 99% within 90 days of launch)
- Multi-year comparison reports generated for 50%+ of completed reviews
- Compliance calendar adoption: 80%+ of active users access calendar weekly
- Notification delivery rate >= 99.5% across all channels

**90 Days:**
- 500+ review cycles completed
- Average review cycle duration < 30 days
- >= 99% on-time DDS renewal rate
- Zero EUDR penalties for active customers attributable to missed annual review deadlines
- Multi-year comparison reports generated for 90%+ of completed reviews
- NPS > 50 from compliance officer persona
- Average manual intervention < 5 per review cycle (vs. 30+ baseline)

---

## 10. Timeline and Milestones

### Phase 1: Core Scheduling Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Review Cycle Manager (Feature 1): lifecycle tracking, status transitions, intelligent scheduling, pipeline queries | Senior Backend Engineer |
| 2-3 | Regulatory Deadline Tracker (Feature 2): deadline calculation, 5-tier escalation, emergency review protocol | Backend Engineer |
| 3-5 | Review Checklist Generator (Feature 3): 7 commodity templates, simplified/full adaptation, Article 8/9/10/11 coverage, item dependency management | Backend Engineer + Regulatory SME |
| 5-6 | Entity Review Coordinator (Feature 4): dependency graph definition, parallel execution, shared dependency detection, EUDR-026 integration | Senior Backend Engineer |

**Milestone: Core scheduling and coordination engine operational (Week 6)**

### Phase 2: Intelligence and Visualization (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Multi-Year Comparison Engine (Feature 5): dimension comparison, materiality analysis, executive summary, trend analysis, export | Backend Engineer + Data Engineer |
| 8-10 | Compliance Calendar (Feature 6): event aggregation from 8 sources, conflict detection, workload forecasting, calendar views, iCal export | Frontend Engineer + Backend Engineer |

**Milestone: Full intelligence and calendar capabilities operational (Week 10)**

### Phase 3: Notifications, API, and Integration (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | Automated Notification System (Feature 7): 4 channels, 12 templates, role-based routing, throttling, i18n, preference management | Backend Engineer + Frontend Engineer |
| 12-13 | REST API Layer: 30+ endpoints, authentication (JWT), authorization (RBAC), rate limiting, OpenAPI documentation | Backend Engineer |
| 13-14 | Integration testing: EUDR-026 orchestrator integration, EUDR-030 DDS regeneration trigger, EUDR-033 monitoring feed integration, end-to-end review cycle testing | Backend Engineer + QA Engineer |

**Milestone: All 7 P0 features implemented with full API and integrations (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 500+ tests, golden tests for all 7 commodities, deadline edge cases, escalation scenarios, dependency coordination | QA Engineer + Backend Engineer |
| 16-17 | Performance testing, security audit, load testing (10,000 concurrent review cycles) | DevOps + Security |
| 17 | Database migration V119 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers) with real annual review cycles | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 7 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Review Template Library (Feature 8)
- AI-Assisted Review Prioritization (Feature 9)
- Stakeholder Performance Analytics (Feature 10)
- Cross-regulation calendar integration (CSRD, CBAM, CSDDD)
- Mobile-optimized calendar and notification views

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-030 Documentation Generator | BUILT (100%) | Low | Stable; DDS submission records API well-defined |
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable; graph snapshot API for comparison |
| AGENT-EUDR-006 Plot Boundary Manager | BUILT (100%) | Low | Stable; plot data API for comparison |
| AGENT-EUDR-026 Due Diligence Orchestrator | BUILT (100%) | Low | Stable; agent execution trigger API defined |
| AGENT-EUDR-028 Risk Assessment Engine | BUILT (100%) | Low | Stable; risk assessment API for comparison |
| AGENT-EUDR-029 Mitigation Measure Designer | BUILT (100%) | Low | Stable; mitigation records API for comparison |
| AGENT-EUDR-033 Continuous Monitoring Agent | BUILT (100%) | Low | Stable; regulatory update feed, freshness data |
| AGENT-EUDR-024 Third-Party Audit Manager | BUILT (100%) | Low | Stable; audit schedule API for calendar |
| AGENT-EUDR-012 Document Authentication | BUILT (100%) | Low | Stable; certification expiry API for calendar |
| AGENT-EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Stable; risk score API for coordination trigger |
| AGENT-EUDR-031 Stakeholder Engagement Tool | BUILT (100%) | Low | Stable; engagement milestone API for calendar |
| AGENT-EUDR-032 Grievance Mechanism Manager | BUILT (100%) | Low | Stable; grievance deadline API for calendar |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| Redis | Production Ready | Low | Standard cache infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU EUDR Article 8(1) 12-month renewal requirement | Active regulation | Low | Core regulatory basis; unlikely to change in short term |
| EC country benchmarking updates (Article 29) | Published periodically | Medium | EUDR-033 monitors; emergency review protocol handles reclassification |
| EU Information System DDS schema | Published (v1.x) | Medium | EUDR-030 handles schema adaptation; this agent triggers regeneration |
| Email delivery infrastructure (SMTP/SES) | Available | Low | Multiple provider fallback; retry logic |
| SMS gateway (Twilio) | Available | Low | SMS used only for critical escalations; fallback to email |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EC changes the 12-month renewal period (e.g., to 6 months or 24 months) through implementing regulation | Low | High | Renewal period is configurable per operator; EUDR-033 regulatory update feed triggers configuration update; all deadline calculations recalculated automatically |
| R2 | Large operator with 1,000+ DDS submissions overwhelms the review pipeline (all submitted in the same month, all expire in the same month) | Medium | High | Intelligent scheduling distributes reviews across the review window; workload forecasting identifies bottleneck periods 12 months in advance; capacity planning tool enables proactive team scaling |
| R3 | Upstream agent (EUDR-028 risk assessment) fails during a review dependency chain, blocking review completion | Medium | Medium | Dependency retry mechanism with exponential backoff; manual override capability for failed dependencies; review cycle does not auto-expire on dependency failure (remains In Progress with blocking notification) |
| R4 | Notification fatigue causes stakeholders to ignore escalation alerts | Medium | High | Notification throttling (configurable daily limit); digest mode consolidates notifications; escalation tiers ensure critical alerts are differentiated from routine notifications; SMS reserved for Tier 4+ ensures channel credibility |
| R5 | Multi-year comparison data incomplete for operators who recently onboarded | High (Year 1) | Medium | Graceful degradation: first-year reviews generate baseline reports without year-over-year comparison; comparison becomes available from second year onward; baseline reports clearly labeled |
| R6 | Country reclassification triggers emergency reviews for many DDS simultaneously | Low | High | Emergency review protocol includes shared dependency deduplication (single EUDR-001 re-mapping shared across DDS in same supply chain); prioritization engine sequences reviews by risk level |
| R7 | Checklist templates become outdated when EUDR implementing regulations add new requirements | Medium | Medium | Checklist templates are configuration-driven (not hardcoded); regulatory updates from EUDR-033 trigger template review; new items can be appended to in-progress reviews without restarting |
| R8 | External auditors cannot access review history due to permission configuration issues | Low | Medium | Dedicated Auditor RBAC role with read-only access to review history, checklists, comparison reports, and notification logs; tested in beta with 2 audit firms |
| R9 | Calendar event aggregation from 8+ sources creates inconsistent or duplicate events | Medium | Medium | Event deduplication engine based on source_agent + event_type + event_date + associated_dds composite key; conflict detection identifies overlapping events from different sources |
| R10 | Regulatory deadline calculation errors due to time zone handling | Low | High | All deadlines calculated and stored in UTC; operator-local display is presentation-only; comprehensive leap year, month-end, and year-boundary edge case test suite with 50+ test cases |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Review Cycle Manager Unit Tests | 80+ | Lifecycle tracking, status transitions, intelligent scheduling, pipeline queries, edge cases (leap year, month-end) |
| Deadline Tracker Unit Tests | 60+ | Deadline calculation, 5-tier escalation, emergency review triggers, threshold customization, consolidation |
| Checklist Generator Unit Tests | 70+ | 7 commodity templates, simplified/full adaptation, Article mapping coverage, item dependencies, dynamic expansion |
| Entity Coordinator Unit Tests | 60+ | Dependency graph, parallel execution, shared dependency deduplication, failure handling, retry, override |
| Comparison Engine Unit Tests | 60+ | 7 dimension comparisons, materiality thresholds, executive summary, trend analysis, provenance hashing |
| Calendar Unit Tests | 50+ | Event aggregation from 8 sources, conflict detection, workload forecasting, export (iCal/CSV/JSON) |
| Notification System Unit Tests | 50+ | 4 channels, 12 templates, role-based routing, throttling, i18n, deduplication, delivery tracking |
| API Tests | 60+ | All 30+ endpoints, authentication, authorization, error handling, pagination, rate limiting |
| Golden Tests (Commodity-Specific) | 35+ | 7 commodities x 5 scenarios per commodity (annual review, emergency, simplified, multi-year comparison, overdue escalation) |
| Integration Tests | 30+ | Cross-agent integration: EUDR-026 orchestration, EUDR-030 DDS regeneration, EUDR-033 monitoring feed, EUDR-028 risk re-assessment |
| Performance Tests | 20+ | 1,000/5,000/10,000 concurrent review cycles, 5,000+ calendar events, 10,000-entity comparison, notification dispatch at scale |
| **Total** | **575+** | |

### 13.2 Golden Test Scenarios per Commodity

Each of the 7 commodities will have 5 dedicated golden test scenarios:

1. **Standard Annual Review (Full DD)**: Complete annual review cycle for a Standard/High-risk country DDS, including full checklist generation, dependency coordination, comparison report, and DDS renewal. Expected: review completes within 30 days with 100% checklist coverage.

2. **Emergency Review (Country Reclassification)**: Country reclassified from Low to Standard risk. DDS previously using simplified DD must transition to full DD within 30-day emergency window. Expected: emergency cycle created, checklist expanded, accelerated dependencies executed.

3. **Simplified Due Diligence Review**: Annual review for a Low-risk country DDS. Simplified checklist generated with abbreviated verification steps. Expected: review completes within 10 days with 100% simplified checklist coverage.

4. **Multi-Year Comparison (3 Cycles)**: Third annual review cycle for a DDS with 2 prior review cycles. Full multi-year comparison generated with trend analysis across all 7 dimensions. Expected: comparison report shows year-over-year trends, material changes flagged, executive summary generated.

5. **Overdue Escalation Path**: Review cycle that progresses through all 5 escalation tiers (90/60/30/7/0 days) without completion. Expected: all 5 escalation tiers fire on schedule, correct recipients notified at each tier, compliance incident created at Tier 5.

Total: 7 commodities x 5 scenarios = 35 golden test scenarios

### 13.3 Edge Case Test Suite (Deadline Calculations)

| Edge Case | Input | Expected Output |
|-----------|-------|----------------|
| Standard 12-month calculation | DDS submitted 2026-03-15 | Renewal deadline 2027-03-15 |
| Leap year submission | DDS submitted 2028-02-29 | Renewal deadline 2029-02-28 |
| Month-end (31-day month to 30-day month) | DDS submitted 2026-01-31 | Renewal deadline 2027-01-31 |
| Year boundary | DDS submitted 2026-12-15 | Renewal deadline 2027-12-15 |
| February submission | DDS submitted 2026-02-28 | Renewal deadline 2027-02-28 |
| Feb 29 leap to non-leap | DDS submitted 2024-02-29 | Renewal deadline 2025-02-28 |
| January 1 submission | DDS submitted 2026-01-01 | Renewal deadline 2027-01-01 |
| December 31 submission | DDS submitted 2026-12-31 | Renewal deadline 2027-12-31 |
| Escalation Tier 1 (90 days before) | Deadline 2027-03-15 | Tier 1 fires 2026-12-15 |
| Escalation Tier 5 (overdue) | Deadline 2027-03-15, not completed | Tier 5 fires 2027-03-15 |

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4, submitted to the EU Information System |
| **Annual Review** | The process of renewing a DDS within the 12-month period mandated by Article 8(1) |
| **Review Cycle** | A single instance of an annual review, tracked from initiation to completion |
| **Review Window** | The configurable period before a DDS expiry during which the review should be conducted (default: 90 days) |
| **Escalation Tier** | One of 5 urgency levels triggered as a deadline approaches: Planning (90d), Standard (60d), Urgent (30d), Critical (7d), Overdue (0d) |
| **Simplified Due Diligence** | Abbreviated due diligence process available for products from low-risk countries per Article 13 |
| **Full Due Diligence** | Complete due diligence process required for products from standard or high-risk countries |
| **Emergency Review** | An out-of-cycle review triggered by a material compliance event (e.g., country reclassification) |
| **Dependency Chain** | The ordered sequence of upstream agent re-executions required to complete an annual review |
| **Multi-Year Comparison** | A structured report comparing compliance data across annual review cycles |
| **Compliance Calendar** | A unified temporal view of all EUDR obligations across an operator's compliance portfolio |
| **Materiality Threshold** | A configurable threshold above which a year-over-year change is flagged as significant |
| **Critical Path** | The sequence of review dependencies whose combined duration determines the minimum total review time |
| **Review Cycle Status** | One of: Not Started, Scheduled, In Progress, Under Review, Completed, Archived, Overdue |
| **CN Code** | Combined Nomenclature -- EU product classification code |
| **HS Code** | Harmonized System -- international product classification code |

### Appendix B: EUDR Article 8 -- Due Diligence Statement Requirements

Per Article 8(1), the Due Diligence Statement must contain:
- (a) The name, postal address, email address, and (where applicable) EORI number of the operator or trader
- (b) The HS code(s) and a description of the relevant commodity or product, including the trade name where applicable
- (c) The quantity of the relevant commodity or product
- (d) The country of production, and the geolocation of all plots of land where the relevant commodities were produced
- (e) The date or time range of production
- (f) The names, postal addresses, and email addresses of persons in the supply chain
- (g) Sufficiently conclusive and verifiable information that the relevant commodities and products are deforestation-free
- (h) Sufficiently conclusive and verifiable information that the production of the relevant commodities was in accordance with the relevant legislation of the country of production

**Key for Annual Review**: All of these elements must be re-verified and updated during the annual review cycle. The Review Checklist Generator (Feature 3) maps each of these elements to specific checklist items.

### Appendix C: Annual Review Dependency Chain (Detailed)

```
                    +-------------------+
                    |  Review Initiated |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+       +-----------v-----------+
    | EUDR-001           |       | EUDR-006               |
    | Supply Chain       |       | Plot Boundary           |
    | Re-Mapping         |       | Re-Verification         |
    +--------+-----------+       +----------+--------------+
              |                             |
              +--------------+--------------+
                             |
                   +---------v---------+
                   | EUDR-012           |
                   | Document/Cert      |
                   | Re-Authentication  |
                   +---------+----------+
                             |
                   +---------v---------+
                   | EUDR-017           |
                   | Supplier Risk      |
                   | Re-Scoring         |
                   +---------+----------+
                             |
                   +---------v---------+
                   | EUDR-028           |
                   | Risk               |
                   | Re-Assessment      |
                   +---------+----------+
                             |
                    +--------v--------+
                    | Risk Negligible? |
                    +--------+--------+
                         |       |
                    YES  |       |  NO
                         |       |
              +----------+  +----v--------------+
              |             | EUDR-029           |
              |             | Mitigation         |
              |             | Re-Evaluation      |
              |             +----+---------------+
              |                  |
              +-------+----------+
                      |
            +---------v---------+
            | EUDR-030           |
            | DDS                |
            | Regeneration       |
            +---------+----------+
                      |
            +---------v---------+
            | Review Complete    |
            +-------------------+
```

### Appendix D: Notification Template Examples

**Tier 3 Urgent Escalation (Email Template):**

```
Subject: URGENT -- DDS {dds_reference} Annual Review Due in 30 Days

Dear {recipient_name},

This is an urgent notification regarding the annual review for:

  DDS Reference: {dds_reference}
  Commodity: {commodity}
  Country of Origin: {country}
  Renewal Deadline: {deadline_date}
  Days Remaining: 30

Current Review Status: {review_status}
Checklist Completion: {completion_percentage}%
Blocking Items: {blocking_items_count}

This notification has been escalated to management per the operator's
escalation policy. Immediate action is required to ensure the DDS is
renewed before the Article 8(1) deadline.

Action Required:
  - Review blocking items and assign resources
  - Ensure all upstream dependencies are in progress
  - Contact procurement for outstanding supplier data

Review Dashboard: {review_dashboard_url}

This is an automated notification from GreenLang EUDR Annual Review Scheduler.
```

### Appendix E: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 -- EU Deforestation Regulation
2. EU Deforestation Regulation Guidance Document (European Commission)
3. EUDR Technical Specifications for the Information System
4. Article 8(1) -- Due Diligence Statement Renewal Requirements
5. Article 13 -- Simplified Due Diligence for Low-Risk Countries
6. Articles 14-16 -- Competent Authority Checks and Substantive Verification
7. Article 29 -- Country Benchmarking Classification
8. Article 31 -- Record Keeping Requirements (5-year retention)
9. ISO 22095:2020 -- Chain of Custody -- General Terminology and Models
10. GreenLang EUDR Agent Ecosystem Technical Documentation (EUDR-001 through EUDR-033)

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-12 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-12 | GL-ProductManager | Initial draft created: comprehensive PRD for AGENT-EUDR-034 Annual Review Scheduler Agent with 7 P0 features covering review cycle management, regulatory deadline tracking with 5-tier escalation, commodity-specific checklist generation for all 7 EUDR commodities, entity review coordination with dependency graph management, multi-year comparison engine across 7 compliance dimensions, unified compliance calendar aggregating events from 8 upstream agents, and automated multi-channel notification system with 12 lifecycle event templates. Regulatory coverage verified against Articles 4, 8(1), 9, 10, 11, 12, 13, 14-16, 29, 31. Database migration V119 schema defined with 9 tables and 4 hypertables. 30+ API endpoints specified. 19 RBAC permissions defined. 15 Prometheus metrics defined. Integration points specified for 12 upstream agents and 5 downstream consumers. Test strategy covers 575+ tests including 35 golden test scenarios (7 commodities x 5 scenarios). |
