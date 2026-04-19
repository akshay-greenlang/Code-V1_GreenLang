# PRD: AGENT-EUDR-033 -- Continuous Monitoring Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-033 |
| **Agent ID** | GL-EUDR-CM-033 |
| **Component** | Continuous Monitoring Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence (Category 5) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-12 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 8 (data freshness), 10 (due diligence/risk assessment), 11 (risk mitigation), 13 (simplified due diligence monitoring), 14-16 (competent authorities), 29 (country benchmarking), 31 (record keeping) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **Prerequisite Agents** | AGENT-EUDR-001 (Supply Chain Mapping Master), AGENT-EUDR-017 (Supplier Risk Scorer), AGENT-EUDR-020 (Deforestation Alert System), AGENT-EUDR-027 (Information Gathering Agent), AGENT-DATA-016 (Data Freshness Monitor) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) is not a one-time compliance exercise. Article 10(1) requires operators to "assess and identify the risk" of non-compliance on an ongoing basis. Article 8 mandates that information supporting the Due Diligence Statement (DDS) must be current and not older than 12 months. Article 13 establishes that even operators benefiting from simplified due diligence for low-risk country products must "review" their due diligence at regular intervals. Articles 14 through 16 grant competent authorities the power to conduct checks at any time, meaning operators must maintain continuous compliance readiness -- not merely comply at the moment of DDS submission and then allow their data, risk assessments, and supply chain knowledge to decay.

The GreenLang platform has built a comprehensive suite of 32 EUDR agents spanning supply chain traceability (EUDR-001 through EUDR-015), risk assessment (EUDR-016 through EUDR-020), environmental and social due diligence (EUDR-021 through EUDR-022), and due diligence workflow agents (EUDR-023 through EUDR-032). These agents perform deep analytical work at specific points in time: mapping supply chains, scoring risks, verifying deforestation-free status, authenticating documents, designing mitigation measures, generating DDS documentation, managing stakeholder engagement, and processing grievances. However, all of these agents operate on a request-response or batch-processing model. They execute when invoked, produce results, and then go silent until the next invocation.

This creates a critical gap: **nothing watches the supply chain between invocations**. The world does not pause between due diligence cycles. Between one DDS submission and the next, suppliers change ownership, certifications expire, countries get reclassified to higher risk categories, deforestation alerts fire on production plots, plot boundaries shift, new regulatory guidance is published, and commodity sourcing patterns evolve. Without continuous monitoring, operators discover these changes only when they run their next due diligence cycle -- weeks or months after the change occurred -- by which time the compliance impact has compounded and remediation options have narrowed.

Today, EU operators face the following critical continuous monitoring gaps:

- **No real-time supply chain surveillance**: The 32 upstream agents produce a snapshot of supply chain compliance at the time of execution. Between executions, there is no surveillance system watching for changes in supplier status, ownership transfers, new sub-tier relationships, facility closures, or operational disruptions. When a Tier 2 cocoa cooperative in Ghana loses its Rainforest Alliance certification, operators sourcing through that cooperative are not notified until the next manual review cycle. By then, non-compliant cocoa may already have entered the EU market, exposing the operator to penalties of up to 4% of annual EU turnover.

- **No deforestation alert correlation with supply chain**: EUDR-020 (Deforestation Alert System) monitors satellite imagery and fires alerts when deforestation is detected in monitored areas. However, these alerts exist in isolation -- they identify geographic deforestation events but do not automatically correlate those events with specific supply chain nodes, products in transit, or DDS submissions. When satellite imagery detects forest clearing on a palm oil plantation in Kalimantan, the operator must manually cross-reference the alert coordinates against their supply chain graph (EUDR-001), identify affected shipments, assess which DDS documents are impacted, and determine whether products already placed on the EU market are now potentially non-compliant. This manual correlation process takes 2-8 hours per alert and is error-prone.

- **No automated compliance scanning**: EUDR compliance involves dozens of interrelated requirements across Articles 4, 8, 9, 10, 11, 12, and 29. Each requirement has specific data freshness thresholds, documentation requirements, and validation criteria. There is no automated scanner that continuously validates all supply chain entities against the full EUDR requirements matrix and flags entities that have fallen out of compliance since their last verification. Operators rely on periodic manual reviews, which cannot keep pace with the volume of entities in complex multi-tier supply chains.

- **No change detection across supply chain dimensions**: Supply chains are dynamic across multiple dimensions: geographic (plot boundaries change, new plots are added, existing plots are subdivided or consolidated), organizational (suppliers merge, are acquired, change ownership, open or close facilities), documentary (certifications expire, are renewed, suspended, or revoked; trade licenses lapse), and commodity-related (sourcing patterns shift, new commodities are introduced, volumes change significantly). Without automated change detection across all these dimensions, operators cannot maintain the "current and accurate" supply chain knowledge that EUDR demands.

- **No risk score degradation monitoring**: EUDR-017 (Supplier Risk Scorer) calculates risk scores based on a multi-dimensional assessment at the time of execution. These scores can degrade between assessments due to external events: a country's Transparency International CPI score drops, a supplier's certification is suspended, deforestation alerts fire near a supplier's production plots, or a competent authority issues an enforcement action against a supplier. Without continuous risk monitoring, operators do not know that a supplier rated "Low Risk" three months ago has accumulated enough negative signals to warrant reclassification to "High Risk" and enhanced due diligence under Article 10.

- **No data freshness enforcement**: EUDR Article 8 requires that information supporting the DDS is current. Article 31 requires 5-year record retention, but does not exempt operators from maintaining current data. When information underlying a DDS becomes stale -- geolocation data not re-verified within 12 months, supplier certifications not re-checked after expiry date, country risk classifications not updated after EC benchmark revision -- the DDS may no longer be valid. There is no automated system enforcing data freshness policies, alerting operators when specific data elements are approaching or have exceeded their validity window, and triggering data refresh workflows before staleness causes non-compliance.

- **No regulatory update tracking**: The EUDR regulatory landscape is evolving. The European Commission issues implementing regulations, publishes country benchmarking updates (Article 29), revises the list of derived products in Annex I, issues guidance documents, and amends technical specifications for the EU Information System. Competent authorities in member states publish enforcement priorities, inspection criteria, and penalty guidelines. Without automated tracking of these regulatory changes, operators may be applying outdated rules, missing new requirements, or failing to adapt their due diligence processes to current regulatory expectations.

Without a continuous monitoring agent, the entire EUDR compliance posture degrades between due diligence cycles. The 32 upstream agents build compliance at a point in time; without continuous monitoring, that compliance erodes daily until the next cycle. Operators face penalties of up to 4% of annual EU turnover, confiscation of goods, temporary exclusion from public procurement, and public naming under Articles 23-25 -- penalties that can be triggered not just by initial non-compliance but by failure to maintain ongoing compliance.

### 1.2 Solution Overview

Agent-EUDR-033: Continuous Monitoring Agent is a specialized always-on surveillance agent that provides real-time, automated, continuous monitoring of EUDR supply chain compliance across all dimensions. It operates as the "watchtower" of the EUDR agent ecosystem -- the agent that never sleeps, continuously scanning supply chains, correlating external signals, validating compliance status, detecting changes, monitoring risk trajectories, enforcing data freshness policies, and tracking regulatory evolution. It bridges the gap between point-in-time due diligence executions by maintaining continuous compliance awareness and triggering corrective actions the moment a compliance-relevant change is detected.

The agent consumes data from across the GreenLang EUDR agent ecosystem: supply chain graphs from EUDR-001, risk scores from EUDR-017, deforestation alerts from EUDR-020, information packages from EUDR-027, and data freshness metrics from AGENT-DATA-016. It produces a continuous stream of monitoring events, compliance alerts, change notifications, risk degradation warnings, data freshness violations, and regulatory update advisories that feed into the due diligence orchestration layer (EUDR-026), risk assessment engine (EUDR-028), mitigation measure designer (EUDR-029), and documentation generator (EUDR-030).

Core capabilities:

1. **Real-Time Supply Chain Monitoring Engine** -- Continuous surveillance of all supply chain entities (suppliers, plots, facilities, certifications, documents) for status changes, ownership transfers, certification events, facility updates, and operational disruptions. Implements event-driven monitoring with configurable watch rules per entity type, severity classification for detected changes, and automated notification dispatch. Monitors supplier databases, certification body feeds, corporate registry changes, and trade compliance databases in near-real-time with configurable polling intervals (minimum 15 minutes, default 1 hour).

2. **Deforestation Alert Correlation Engine** -- Automated correlation of deforestation alerts from EUDR-020 with the supply chain graph from EUDR-001. When a deforestation alert fires for a geographic area, this engine identifies all production plots within or overlapping the alert polygon, traces those plots forward through the supply chain graph to identify affected suppliers, shipments, batches, and products, determines which DDS submissions reference affected plots, and generates correlated impact assessments with recommended actions. Eliminates the 2-8 hour manual correlation process.

3. **Automated Compliance Scanner** -- Scheduled and on-demand scanning of all supply chain entities against the complete EUDR requirements matrix. Validates Article 9 information completeness (geolocation, product description, quantities, supplier identification), Article 10 risk assessment currency, Article 11 mitigation measure effectiveness, Article 12 DDS validity, and Article 29 country classification alignment. Generates compliance scorecards per entity, per commodity, and per supply chain with trend tracking over time.

4. **Change Detection Engine** -- Multi-dimensional change detection across geographic (plot boundary modifications, new plot registrations, plot subdivisions), organizational (supplier ownership transfers, facility openings/closures, corporate restructuring), documentary (certification expiry, renewal, suspension, revocation; trade license changes), and commodity (sourcing pattern shifts, volume anomalies, new commodity introduction) dimensions. Uses configurable change detection rules with materiality thresholds to distinguish routine updates from compliance-significant changes.

5. **Risk Score Degradation Monitor** -- Continuous tracking of risk score trajectories for all supply chain entities. Monitors external risk signals (country CPI changes, certification status changes, deforestation alert proximity, enforcement actions, media reports, NGO alerts) and calculates real-time risk score adjustments without waiting for a full EUDR-017 re-assessment cycle. Implements configurable degradation thresholds that trigger alerts when a supplier's effective risk score crosses from one risk level to another (e.g., Low to Standard, Standard to High). Generates risk trajectory visualizations showing score evolution over time.

6. **Data Freshness Enforcement Engine** -- Automated enforcement of data freshness policies aligned with EUDR Article 8 requirements. Defines freshness windows for each data category (geolocation verification: 12 months; certification validity: per certificate expiry date; country risk classification: per EC publication cycle; supplier risk assessment: 6 months; satellite verification: 3 months). Tracks the age of every data element across all supply chain entities. Generates tiered alerts (approaching expiry, expired, critically overdue) and triggers automated data refresh workflows through EUDR-027 (Information Gathering Agent) when data elements approach their freshness limits.

7. **Regulatory Update Tracker** -- Automated monitoring of EU regulatory publications, competent authority announcements, EC implementing regulations, country benchmarking updates (Article 29), Annex I derived product list amendments, EU Information System technical specification changes, and member state enforcement guidance. Parses regulatory feeds, classifies updates by impact severity and affected scope (commodity, country, operator type), maps updates to affected supply chain entities, and generates regulatory change impact assessments with recommended operator actions. Maintains a regulatory change log with full provenance for audit trail.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Supply chain entity monitoring coverage | 100% of active entities continuously monitored | % of entities with active watch rules |
| Deforestation alert correlation time | < 60 seconds from alert receipt to impact assessment | p99 latency from EUDR-020 alert to correlated impact report |
| Deforestation alert correlation accuracy | >= 98% correct supply chain linkage | Precision/recall against manually verified correlations |
| Compliance scan coverage | 100% of EUDR requirements validated per entity | Requirements matrix coverage per scan cycle |
| Compliance scan cycle time | < 15 minutes for 10,000 entities full scan | p99 scan duration under production load |
| Change detection latency | < 5 minutes from change occurrence to detection | Time delta between external change timestamp and detection event |
| Change detection accuracy | >= 95% true positive rate, < 5% false positive rate | Precision/recall against manually verified change events |
| Risk score degradation detection time | < 30 minutes from signal receipt to risk adjustment | Latency from external signal to updated risk trajectory |
| Data freshness violation detection | 100% of stale data elements detected before regulatory deadline | % of freshness violations caught proactively |
| Data freshness alert lead time | >= 30 days before data element expiry | Average alert-to-expiry interval |
| Regulatory update detection | 100% of published EUDR regulatory changes tracked within 24 hours | Coverage audit against official publication feeds |
| Monitoring uptime | >= 99.9% availability (< 8.76 hours downtime per year) | Uptime monitoring via OBS-001 Prometheus |
| Alert actionability | >= 90% of alerts result in operator action within 48 hours | Alert-to-action tracking |
| Zero-hallucination guarantee | 100% deterministic monitoring and alerting, no LLM in critical path | Bit-perfect reproducibility tests |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated continuous compliance monitoring technology market of 4-7 billion EUR driven by the shift from point-in-time auditing to continuous assurance models across all ESG regulations (EUDR, CSDDD, CSRD, CBAM).
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities requiring automated continuous monitoring to maintain compliance between due diligence cycles, estimated at 700M-1.2B EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 40-70M EUR in continuous monitoring module ARR. Premium pricing justified by the insurance value: continuous monitoring prevents penalties that can reach 4% of annual EU turnover.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) of EUDR-regulated commodities with complex, multi-tier supply chains requiring real-time surveillance
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya) with supply chains spanning 10+ countries and 100+ direct suppliers
- Timber and paper industry operators with dynamic supply chains where plot boundaries, concession rights, and certification status change frequently
- Companies with active EUDR compliance programs using GreenLang agents EUDR-001 through EUDR-032 that need continuous monitoring between due diligence cycles

**Secondary:**
- Compliance consulting firms managing ongoing EUDR monitoring for multiple operator clients
- Certification bodies (FSC, RSPO, PEFC, Rainforest Alliance) requiring real-time supply chain surveillance to maintain chain-of-custody integrity
- Financial institutions requiring continuous ESG risk monitoring for portfolio companies with EUDR exposure
- Insurance underwriters offering EUDR compliance insurance products requiring continuous risk monitoring
- SME importers (1,000-10,000 shipments/year) preparing for June 30, 2026 enforcement deadline

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual periodic review | No technology cost; deep institutional knowledge | Infrequent (quarterly at best); cannot scale; misses inter-cycle changes; 40+ hours per review cycle | Always-on, sub-minute detection; scales to 100K+ entities |
| Generic supply chain monitoring (Resilinc, Everstream, Interos) | Multi-risk monitoring; supply chain disruption alerts; news monitoring | Not EUDR-specific; no deforestation alert correlation; no Article 8 freshness enforcement; no regulatory compliance scanning | Purpose-built for EUDR; deforestation alert correlation; Article 8 freshness enforcement; full EUDR requirements scanning |
| Satellite monitoring platforms (Global Forest Watch, Planet, Satelligence) | High-resolution satellite imagery; deforestation detection; geographic coverage | Satellite-only; no supply chain correlation; no compliance scanning; no certification monitoring; no risk score tracking | Full supply chain correlation; multi-dimensional monitoring; certification + compliance + risk + freshness + regulatory |
| ESG data providers (MSCI, Sustainalytics, S&P Global ESG) | Broad ESG data coverage; company-level risk ratings; portfolio screening | Company-level only (not supply chain level); quarterly updates; no real-time; no EUDR-specific compliance scanning | Plot-level granularity; real-time updates; EUDR-specific compliance validation; supply chain graph integration |
| Custom alerting (manual rules in ERP/procurement) | Tailored to organization; low incremental cost | Fragile; single-dimension; no regulatory intelligence; no deforestation correlation; no risk trajectory | Multi-dimensional; regulatory-aware; satellite-correlated; risk trajectory analysis; production-grade |
| Niche EUDR compliance tools (Preferred by Nature, Ecosphere+) | Commodity expertise; certification knowledge | Point-in-time assessment; no continuous monitoring; limited automation; manual updates | Always-on monitoring; automated correlation; full EUDR requirements coverage; ecosystem integration |

### 2.4 Differentiation Strategy

1. **Always-on surveillance** -- Not a periodic assessment tool. Monitors continuously with sub-minute detection latency for critical changes, filling the gap between point-in-time due diligence cycles.
2. **Deforestation-to-supply-chain correlation** -- The only platform that automatically correlates satellite deforestation alerts (EUDR-020) with supply chain graph nodes (EUDR-001) to produce actionable impact assessments in under 60 seconds.
3. **Multi-dimensional change detection** -- Monitors geographic, organizational, documentary, and commodity dimensions simultaneously, detecting compound changes that single-dimension monitors miss.
4. **Regulatory intelligence** -- Automated tracking of EU regulatory changes with impact assessment mapping to specific supply chain entities and compliance requirements.
5. **Ecosystem integration** -- Native integration with all 32 upstream EUDR agents, reusing their data, models, and analytical capabilities rather than rebuilding from scratch.
6. **Zero-hallucination monitoring** -- All monitoring, detection, correlation, and alerting is deterministic with no LLM in the critical path. Every alert is reproducible and provenance-tracked.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Eliminate compliance decay between due diligence cycles | Zero instances of undetected compliance-relevant changes persisting > 48 hours for active customers | Q3 2026 |
| BG-2 | Reduce EUDR penalty exposure from stale data and missed changes | Zero EUDR penalties for active customers attributable to monitoring gaps | Ongoing |
| BG-3 | Reduce time from deforestation detection to supply chain impact assessment | < 60 seconds (from current 2-8 hours manual process) | Q2 2026 |
| BG-4 | Ensure Article 8 data freshness compliance for all active DDS submissions | 100% of DDS-supporting data within freshness windows at all times | Q3 2026 |
| BG-5 | Maintain continuous competent authority inspection readiness | < 2 hours from inspection request to full compliance status report | Q3 2026 |
| BG-6 | Become the reference continuous EUDR monitoring platform | 500+ enterprise customers using continuous monitoring module | Q4 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Real-time supply chain surveillance | Monitor all supply chain entities continuously for status changes, ownership transfers, certification events, and operational disruptions |
| PG-2 | Deforestation-supply chain correlation | Automatically correlate deforestation alerts with supply chain graph nodes, shipments, and DDS submissions |
| PG-3 | Continuous compliance validation | Scan all supply chain entities against the complete EUDR requirements matrix on a continuous basis |
| PG-4 | Multi-dimensional change detection | Detect and classify changes across geographic, organizational, documentary, and commodity dimensions |
| PG-5 | Risk trajectory monitoring | Track risk score evolution over time and detect degradation before risk level thresholds are crossed |
| PG-6 | Data freshness enforcement | Enforce Article 8 data freshness policies with proactive alerting and automated data refresh triggering |
| PG-7 | Regulatory change intelligence | Track EUDR regulatory changes and assess their impact on specific supply chain entities and compliance requirements |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Monitoring engine uptime | >= 99.9% availability (< 8.76 hours downtime per year) |
| TG-2 | Event processing throughput | 10,000+ monitoring events per minute |
| TG-3 | Alert generation latency | < 5 seconds from event detection to alert dispatch |
| TG-4 | Compliance scan performance | < 15 minutes for full scan of 10,000 supply chain entities |
| TG-5 | Deforestation correlation performance | < 60 seconds for full supply chain impact assessment per alert |
| TG-6 | Data freshness check performance | < 1 minute for full freshness validation of 100,000 data elements |
| TG-7 | API response time | < 200ms p95 for standard monitoring queries |
| TG-8 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-9 | Zero-hallucination | 100% deterministic monitoring, alerting, and scoring, bit-perfect reproducibility |
| TG-10 | Resource efficiency | < 512 MB memory baseline for monitoring engine; < 4 CPU cores sustained |

### 3.4 Non-Goals

1. Supply chain graph construction or modification (EUDR-001 handles graph operations; this agent reads the graph but does not modify it)
2. Risk score calculation from scratch (EUDR-017/028 handle risk assessment; this agent monitors risk trajectories and detects degradation)
3. Deforestation detection from satellite imagery (EUDR-020 handles satellite analysis; this agent correlates alerts with supply chain)
4. Information gathering from external databases (EUDR-027 handles information gathering; this agent triggers refresh workflows)
5. Due diligence orchestration (EUDR-026 handles orchestration; this agent feeds monitoring events into the orchestration pipeline)
6. DDS document generation or submission (EUDR-030 handles documentation; this agent monitors DDS validity)
7. Mitigation measure design or implementation (EUDR-029 handles mitigation; this agent monitors mitigation effectiveness)
8. Grievance processing or management (EUDR-031/032 handle grievances; this agent does not process complaints)
9. Direct communication with suppliers (EUDR-031 Stakeholder Engagement Tool handles communications)
10. Legal interpretation of regulatory changes (this agent tracks and classifies regulatory updates but does not provide legal advice)

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, sourcing cocoa from 12 countries, 800+ supply chain entities monitored |
| **EUDR Pressure** | Must maintain continuous compliance across 300+ active DDS submissions; competent authority inspections can occur at any time; board expects real-time compliance dashboards |
| **Pain Points** | Discovers certification expirations weeks after they occur; deforestation alerts require 4+ hours of manual supply chain cross-referencing; no visibility into risk score changes between quarterly assessments; data freshness violations discovered only during DDS renewal; cannot tell if a DDS submitted 6 months ago is still valid today |
| **Goals** | Real-time compliance dashboard showing status of all supply chain entities; instant notification when any entity changes status; automated deforestation-to-supply-chain impact assessment; proactive data freshness alerts 30+ days before expiry; continuous compliance readiness for competent authority inspections |
| **Technical Skill** | Moderate -- comfortable with web applications and dashboards but not a developer |

### Persona 2: Supply Chain Risk Analyst -- Stefan (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Risk Analyst at a European palm oil refinery |
| **Company** | 3,000 employees, sourcing from 200+ plantations across Indonesia, Malaysia, and Papua New Guinea |
| **EUDR Pressure** | Palm oil supply chains face highest deforestation risk; must monitor 200+ plantations continuously for land use changes; country risk classifications may change with EC updates |
| **Pain Points** | Risk scores become stale within weeks of assessment; cannot track multiple external risk signals simultaneously; deforestation alerts arrive without supply chain context; has no early warning system for risk threshold crossings; spends 20+ hours per month manually updating risk assessments |
| **Goals** | Continuous risk trajectory visualization for all suppliers; automated risk degradation alerts before thresholds are crossed; deforestation alerts correlated with specific suppliers, batches, and products; external risk signal integration (CPI, enforcement actions, certification changes); risk trend reports for board reporting |
| **Technical Skill** | High -- comfortable with analytics tools, APIs, and risk modeling platforms |

### Persona 3: Data Steward -- Katrina (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | EUDR Data Quality Manager at a timber importing consortium |
| **Company** | Consortium of 15 EU timber importers sharing data infrastructure, 5,000+ supply chain entities |
| **EUDR Pressure** | Article 8 requires current data; 5,000+ entities with dozens of data elements each means 100,000+ data freshness deadlines to track; manual tracking is impossible |
| **Pain Points** | Cannot track freshness of 100,000+ data elements across 5,000 entities; discovers stale data only when DDS generation fails validation; no proactive alerting for approaching expiry dates; data refresh requests sent manually to suppliers with no tracking; geolocation re-verification backlog grows silently |
| **Goals** | Automated freshness tracking for every data element with tiered alerts (30-day warning, 7-day warning, expired, critically overdue); bulk freshness dashboard with drill-down by entity, data type, and commodity; automated refresh workflow triggering through EUDR-027; freshness compliance reporting for consortium governance |
| **Technical Skill** | High -- comfortable with data management tools, databases, and quality frameworks |

### Persona 4: Regulatory Affairs Specialist -- Dr. Schneider (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Affairs at a European rubber manufacturer |
| **Company** | 12,000 employees, operating across 8 EU member states, rubber supply chain spanning 6 countries |
| **EUDR Pressure** | Must track EUDR implementing regulations, EC guidance, country benchmarking updates, and member state enforcement priorities across 8 jurisdictions; board expects advance warning of regulatory changes |
| **Pain Points** | Monitors 15+ regulatory sources manually; EC country benchmarking updates published without advance notice; implementing regulations change technical requirements for DDS submission; member state competent authorities publish different enforcement priorities; no systematic way to assess the impact of a regulatory change on the company's 400 active supply chain entities |
| **Goals** | Automated tracking of all EUDR-relevant regulatory publications; impact assessment mapping regulatory changes to specific supply chain entities and DDS submissions; advance warning of upcoming regulatory changes; cross-jurisdiction regulatory change calendar; audit trail of regulatory awareness for competent authority inspections |
| **Technical Skill** | High -- comfortable with legal databases, regulatory tracking tools, and compliance platforms |

### Persona 5: External Auditor -- Dr. Hofmann (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must verify that operators maintain continuous compliance, not just point-in-time compliance at DDS submission |
| **Pain Points** | Operators present compliance snapshots that may not reflect current status; no way to verify continuous monitoring between audits; data freshness compliance difficult to assess retroactively; cannot verify that deforestation alerts were properly correlated and acted upon |
| **Goals** | Read-only access to continuous monitoring dashboards; historical monitoring event logs for audit period; evidence of deforestation alert correlation and response; data freshness compliance history; regulatory change tracking and response audit trail with provenance hashes |
| **Technical Skill** | Moderate -- comfortable with audit software, analytics dashboards, and compliance assessment tools |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(2)** | Operators shall exercise due diligence -- collect information, assess risk, mitigate risk | Continuous Monitoring Agent ensures that due diligence information remains current between formal assessment cycles; triggers re-assessment when monitored conditions change |
| **Art. 8(1)** | The due diligence statement shall contain information that is accurate and verifiable | Data Freshness Enforcement Engine (F6) ensures all DDS-supporting data remains within validity windows; Automated Compliance Scanner (F3) continuously validates accuracy and completeness |
| **Art. 8(2)** | Operators shall not place products on the EU market unless they have exercised due diligence and the risk is negligible | Continuous monitoring ensures that the negligible risk determination made at DDS submission time has not been invalidated by subsequent events (deforestation alerts, certification expirations, risk score degradation) |
| **Art. 9(1)(a-d)** | Geolocation of all plots of land | Change Detection Engine (F4) monitors plot boundary changes, new plot registrations, and subdivisions; Data Freshness Enforcement Engine (F6) enforces 12-month geolocation re-verification |
| **Art. 10(1)** | Operators shall assess and identify the risk of non-compliance | Risk Score Degradation Monitor (F5) ensures risk assessments remain current by tracking external risk signals and detecting degradation between formal assessment cycles |
| **Art. 10(2)(a)** | Risk factor: information relating to the risk of deforestation or forest degradation associated with the country of production | Deforestation Alert Correlation Engine (F2) links deforestation events to supply chain entities; Regulatory Update Tracker (F7) monitors EC country benchmarking publications |
| **Art. 10(2)(b)** | Risk factor: presence of forests and forest area change in the country of production | Deforestation Alert Correlation Engine (F2) monitors forest cover changes near production plots via integration with EUDR-020 satellite alerts |
| **Art. 10(2)(e)** | Risk factor: concerns about the country of production | Regulatory Update Tracker (F7) monitors country risk reclassification by EC; Risk Score Degradation Monitor (F5) incorporates country-level signal changes |
| **Art. 10(2)(f)** | Risk factor: risk of circumvention or mixing with products of unknown origin | Change Detection Engine (F4) detects supplier ownership changes, new supply chain intermediaries, and volume anomalies that may indicate circumvention |
| **Art. 10(2)(g)** | Risk factor: the complexity of the relevant supply chain | Change Detection Engine (F4) monitors supply chain topology changes that affect complexity assessments; Real-Time Supply Chain Monitoring Engine (F1) detects new intermediaries and route changes |
| **Art. 11(1)** | Adopt adequate and proportionate risk mitigation measures | Continuous monitoring detects when mitigation measures become ineffective due to changed circumstances, triggering re-assessment through EUDR-029 |
| **Art. 13** | Simplified due diligence for low-risk country products, with monitoring obligation | Regulatory Update Tracker (F7) monitors EC country benchmarking changes that could move a country from Low to Standard/High risk, invalidating simplified due diligence eligibility |
| **Art. 14(1)** | Competent authorities shall carry out checks on operators | Automated Compliance Scanner (F3) maintains continuous inspection readiness; all monitoring data is audit-ready with provenance hashes |
| **Art. 14(2)** | Checks shall include examination of due diligence system, risk assessment, and risk mitigation | All seven monitoring engines maintain auditable logs of monitoring events, detected changes, and triggered actions for competent authority review |
| **Art. 15(1)** | Competent authorities may require operators to take remedial action | Continuous monitoring provides evidence of ongoing remedial actions and their effectiveness; Data Freshness Enforcement Engine (F6) tracks remediation data currency |
| **Art. 16(1)** | Powers of competent authorities -- request information and documents | Monitoring event logs, change detection history, compliance scan results, and risk trajectory data are all retrievable within minutes for competent authority requests |
| **Art. 29(1-3)** | Country benchmarking: Low, Standard, High risk classification by EC | Regulatory Update Tracker (F7) monitors EC country benchmarking publications; Change Detection Engine (F4) detects country reclassification; Risk Score Degradation Monitor (F5) propagates country risk changes to affected entities |
| **Art. 31(1)** | Record keeping for 5 years | All monitoring events, alerts, compliance scan results, change detection records, risk trajectory data, data freshness logs, and regulatory update tracking retained for minimum 5 years with immutable audit trail |

### 5.2 Monitoring Requirements by EUDR Article 8 Data Categories

| Data Category | Freshness Requirement | Monitoring Frequency | Alert Threshold |
|---------------|----------------------|---------------------|-----------------|
| Product description (HS/CN codes) | Per EU Annex I amendments | On regulatory publication | Immediate on Annex I change |
| Quantity/weight | Per shipment | Real-time per custody transfer | Volume anomaly > 20% from baseline |
| Country of production | Per EC benchmarking update | On EC publication | Immediate on country reclassification |
| Geolocation (GPS/polygon) | 12 months maximum | Monthly re-verification check | 30-day warning before 12-month expiry |
| Date of production | Per batch | Real-time per custody transfer | Production date > 12 months from DDS |
| Supplier identification | Per corporate registry change | Weekly corporate registry scan | Immediate on ownership change |
| Deforestation-free evidence | Per satellite monitoring cycle | Per EUDR-020 alert cycle (daily) | Immediate on deforestation alert |
| Legal production evidence | Per certification cycle | Daily certification status check | 60-day warning before certification expiry |
| Risk assessment | 6 months recommended | Continuous risk signal monitoring | Immediate on risk level threshold crossing |
| Mitigation measures | Per mitigation plan cycle | Monthly effectiveness review | Immediate on mitigation failure detection |

### 5.3 Key Regulatory Monitoring Sources

| Source | Type | Update Frequency | Monitoring Method |
|--------|------|-----------------|-------------------|
| Official Journal of the EU (EUR-Lex) | EUDR regulations, implementing acts, delegated acts | As published | RSS/API feed monitoring |
| European Commission EUDR portal | Guidance documents, FAQ updates, country benchmarking | Quarterly (benchmarking); ad hoc (guidance) | Web scraping with change detection |
| EU EUDR Information System | Technical specifications, API changes, schema updates | As published | Technical bulletin monitoring |
| Member state competent authority websites (27 EU member states) | Enforcement priorities, inspection criteria, penalty guidelines | As published | Web monitoring per jurisdiction |
| EUDR Committee proceedings | Draft implementing acts, stakeholder consultations | As scheduled | Publication monitoring |
| FAO / UN COMTRADE | Production and trade statistics | Monthly/quarterly | API polling |
| Global Forest Watch | Deforestation alerts, forest cover data | Weekly | API integration (via EUDR-020) |
| Transparency International CPI | Country corruption perception index | Annual | Annual data refresh trigger |
| Certification body databases (FSC, RSPO, PEFC, RA) | Certificate status changes | Daily | API polling per certification body |

### 5.4 Key Regulatory Dates and Monitoring Implications

| Date | Milestone | Monitoring Implication |
|------|-----------|----------------------|
| December 31, 2020 | EUDR deforestation cutoff date | All deforestation alerts monitored against this baseline date |
| December 30, 2025 | Large operator enforcement (ACTIVE) | All large operator supply chains under continuous monitoring |
| June 30, 2026 | SME enforcement | SME onboarding wave; monitoring engine must scale to handle increased entity volume |
| Quarterly (ongoing) | EC country benchmarking updates | Regulatory Update Tracker must detect and propagate within 24 hours of publication |
| Annual (ongoing) | Certification renewal cycles | Data Freshness Engine tracks all certification expiry dates with 60-day advance warning |
| Ongoing | Satellite monitoring cycles | Deforestation alerts correlated within 60 seconds of receipt from EUDR-020 |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 7 features below are P0 launch blockers. The agent cannot ship without all 7 features operational. Features 1-2 form the real-time surveillance layer; Features 3-5 form the analytical monitoring layer; Features 6-7 form the governance and regulatory intelligence layer.

**P0 Features 1-2: Real-Time Surveillance Layer**

---

#### Feature 1: Real-Time Supply Chain Monitoring Engine

**User Story:**
```
As a compliance officer,
I want all my supply chain entities (suppliers, plots, facilities, certifications)
to be continuously monitored for status changes, ownership transfers, and certification events,
So that I am immediately aware of any change that could affect my EUDR compliance status
and can take corrective action before non-compliance materializes.
```

**Acceptance Criteria:**
- [ ] Monitors all active supply chain entity types: suppliers, producers, processors, traders, plots, facilities, certifications, documents, and batches
- [ ] Implements configurable watch rules per entity type with customizable monitoring frequency (minimum 15 minutes, default 1 hour, maximum 24 hours)
- [ ] Detects supplier status changes: active, suspended, terminated, merged, acquired, bankrupt
- [ ] Detects certification events: issued, renewed, suspended, revoked, expired, scope-changed
- [ ] Detects facility events: opened, closed, relocated, expanded, contracted, permit-changed
- [ ] Detects ownership transfers: corporate acquisition, management buyout, joint venture formation, divestiture
- [ ] Classifies detected changes by severity: Critical (immediate compliance impact), High (compliance impact within 30 days), Medium (potential compliance impact), Low (informational)
- [ ] Generates structured monitoring events with: entity_id, event_type, severity, timestamp, previous_state, new_state, affected_dds_ids, recommended_actions, provenance_hash
- [ ] Dispatches alerts through configurable channels: in-app notification, email, webhook, SMS (for critical alerts), Slack/Teams integration
- [ ] Maintains monitoring event log with full audit trail and 5-year retention per Article 31
- [ ] Supports bulk entity onboarding: import 10,000+ entities with watch rules from CSV/JSON
- [ ] Provides monitoring status dashboard: entities monitored, events detected (24h/7d/30d), alert distribution by severity, entity coverage percentage

**Non-Functional Requirements:**
- Availability: >= 99.9% uptime for monitoring engine
- Latency: < 5 minutes from external change occurrence to detection event generation
- Throughput: Process 10,000+ monitoring events per minute
- Scalability: Support 500,000+ monitored entities across all operator tenants
- Storage: < 100 bytes per monitoring event for efficient time-series storage
- Reliability: Zero lost monitoring events (at-least-once delivery guarantee)

**Dependencies:**
- EUDR-001 Supply Chain Mapping Master (source of supply chain entity graph)
- EUDR-012 Document Authentication Agent (certification status feeds)
- EUDR-008 Multi-Tier Supplier Tracker (supplier hierarchy data)
- AGENT-DATA-003 ERP/Finance Connector (supplier master data changes)
- AGENT-DATA-004 API Gateway Agent (external API integration)
- OBS-004 Alerting and Notification Platform (alert dispatch infrastructure)

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 infrastructure engineer)

**Edge Cases:**
- Entity deleted from supply chain graph while monitoring active -- gracefully deactivate watch rules and archive monitoring history
- Multiple simultaneous changes to same entity within one monitoring cycle -- deduplicate and report as single compound event
- External data source temporarily unavailable -- implement exponential backoff retry with circuit breaker; report data source health in monitoring dashboard
- Watch rule frequency exceeds external API rate limits -- implement adaptive rate limiting per data source with fair queuing across operators
- Entity appears in multiple operator supply chains -- deduplicate monitoring effort but generate operator-specific alerts

---

#### Feature 2: Deforestation Alert Correlation Engine

**User Story:**
```
As a supply chain risk analyst,
I want deforestation alerts from satellite monitoring to be automatically correlated
with my supply chain graph, identifying affected plots, suppliers, shipments, and DDS submissions,
So that I can assess the compliance impact of a deforestation event within seconds
instead of the 2-8 hours required for manual cross-referencing.
```

**Acceptance Criteria:**
- [ ] Receives deforestation alert events from EUDR-020 Deforestation Alert System in real-time via event bus or webhook
- [ ] Extracts alert polygon/coordinates and performs spatial intersection against all registered production plots in EUDR-001 supply chain graph
- [ ] Identifies all production plots that overlap with or are within configurable buffer distance (default 1km) of the deforestation alert polygon
- [ ] Traces affected plots forward through the supply chain graph to identify: affected suppliers (all tiers), affected batches (in transit and delivered), affected products (placed on EU market), affected DDS submissions (active and historical)
- [ ] Calculates deforestation overlap metrics: percentage of plot area affected, distance from plot boundary to deforestation event, temporal correlation (deforestation date vs. production date)
- [ ] Generates correlated impact assessment report containing: alert details (source, coordinates, area, confidence, date), affected entity list (plots, suppliers, batches, products, DDS), impact severity classification (Direct Overlap, Buffer Zone, Indirect via Supply Chain), recommended actions per affected entity
- [ ] Supports three correlation modes: Strict (direct polygon overlap only), Standard (overlap + 1km buffer), Extended (overlap + 5km buffer for high-risk commodities)
- [ ] Caches recent correlation results with configurable TTL to avoid redundant computation for overlapping alerts
- [ ] Generates alert-to-DDS linkage: for each active DDS, lists all deforestation alerts that affect any of its referenced plots
- [ ] Provides interactive correlation visualization: map overlay showing deforestation alert polygon, affected plots, and supply chain path highlighting

**Non-Functional Requirements:**
- Latency: < 60 seconds from alert receipt to complete correlated impact assessment
- Accuracy: >= 98% correct supply chain linkage (precision and recall)
- Spatial precision: Plot-to-alert intersection computed with 1-meter accuracy using PostGIS spatial operations
- Throughput: Process 100+ deforestation alerts per hour during peak satellite pass windows
- Reproducibility: Identical alert + identical supply chain graph = identical correlation result (deterministic)

**Dependencies:**
- EUDR-020 Deforestation Alert System (source of deforestation alert events)
- EUDR-001 Supply Chain Mapping Master (supply chain graph for forward tracing)
- EUDR-002 Geolocation Verification Agent (plot coordinate validation)
- EUDR-006 Plot Boundary Manager Agent (plot polygon data)
- AGENT-DATA-006 GIS/Mapping Connector (PostGIS spatial operations)
- PostGIS extension for spatial intersection queries

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 GIS specialist)

**Edge Cases:**
- Deforestation alert overlaps plots belonging to multiple operators -- generate operator-specific impact assessments with appropriate data isolation
- Production plot has no polygon data (point coordinates only) -- use configurable radius buffer (default 500m) for point-to-polygon intersection
- Deforestation alert area is very large (> 10,000 hectares) -- implement pagination and progressive correlation with intermediate results
- Supply chain graph is incomplete (missing intermediary tiers) -- correlate to known entities and flag incomplete tracing path
- Historical deforestation alert received (date before EUDR cutoff 2020-12-31) -- correlate but classify as "Pre-Cutoff Alert" with different severity

---

**P0 Features 3-5: Analytical Monitoring Layer**

---

#### Feature 3: Automated Compliance Scanner

**User Story:**
```
As a compliance officer,
I want all my supply chain entities to be continuously scanned against the complete
EUDR requirements matrix, generating compliance scorecards with trend tracking,
So that I can maintain inspection readiness at all times and identify
compliance degradation before it becomes a regulatory violation.
```

**Acceptance Criteria:**
- [ ] Defines a complete EUDR requirements matrix covering: Article 9 information completeness (10 mandatory data elements), Article 10 risk assessment currency, Article 11 mitigation measure status, Article 12 DDS validity, Article 29 country classification alignment
- [ ] Scans each supply chain entity against all applicable requirements from the matrix
- [ ] Supports three scan modes: Full Scan (all entities, all requirements -- scheduled daily), Incremental Scan (entities changed since last scan -- triggered by change events), Targeted Scan (specific entity or entity group -- on-demand)
- [ ] Generates per-entity compliance scorecard: overall score (0-100), requirement-by-requirement pass/fail/warning status, non-compliance detail with remediation guidance, trend over last 30/60/90 days
- [ ] Generates per-commodity compliance scorecard: aggregated across all entities in the commodity supply chain
- [ ] Generates per-operator compliance scorecard: aggregated across all commodities and entities
- [ ] Classifies non-compliance findings by severity: Critical (immediate regulatory violation), High (violation within 30 days without action), Medium (compliance risk), Low (best practice recommendation)
- [ ] Tracks compliance score trends over time with time-series storage for historical analysis
- [ ] Triggers automated remediation workflows for specific non-compliance types: missing geolocation triggers EUDR-027 information gathering; expired certification triggers supplier notification; stale risk assessment triggers EUDR-028 re-assessment
- [ ] Provides compliance scan history with full audit trail for competent authority inspection

**Compliance Requirements Matrix:**

| # | Requirement | Article | Validation Rule | Severity if Failed |
|---|------------|---------|-----------------|-------------------|
| CR-01 | Product HS/CN code present and valid | Art. 9(1) | HS/CN code exists and matches EU Combined Nomenclature | Critical |
| CR-02 | Quantity specified with valid units | Art. 9(1) | Quantity > 0 with recognized unit (kg, tonnes, m3, head) | Critical |
| CR-03 | Country of production identified | Art. 9(1) | ISO 3166-1 alpha-2 country code present | Critical |
| CR-04 | Geolocation present for all plots | Art. 9(1)(a-d) | GPS coordinates within valid range; polygon for plots > 4 ha | Critical |
| CR-05 | Geolocation verified within 12 months | Art. 8 | Last verification date < 12 months from current date | High |
| CR-06 | Production date or date range specified | Art. 9(1) | Valid date/date range present | Critical |
| CR-07 | Supplier identification complete | Art. 9(1) | Supplier name, address, registration number present | Critical |
| CR-08 | Deforestation-free evidence available | Art. 3, 9 | Satellite verification result available; no active deforestation alerts on plot | Critical |
| CR-09 | Legal production evidence available | Art. 3, 9 | Valid certification or legal compliance verification present | High |
| CR-10 | Risk assessment conducted and current | Art. 10 | Risk assessment date < 6 months; risk score assigned | High |
| CR-11 | Risk assessment covers all Art. 10(2) criteria | Art. 10(2)(a-g) | All 7 risk criteria evaluated | High |
| CR-12 | Country risk aligned with EC benchmarking | Art. 29 | Entity country risk matches current EC classification | High |
| CR-13 | Mitigation measures in place for high-risk entities | Art. 11 | If risk = High, mitigation plan exists and is active | Critical |
| CR-14 | Mitigation measures effective (residual risk negligible) | Art. 11 | Mitigation effectiveness verified; residual risk <= threshold | High |
| CR-15 | DDS submitted and acknowledged | Art. 12 | DDS submission receipt present; status = acknowledged | Critical |
| CR-16 | DDS supporting data within freshness window | Art. 8, 31 | All DDS-referenced data elements within freshness limits | High |
| CR-17 | Certification valid and current | Art. 10 | Certificate not expired, suspended, or revoked | High |
| CR-18 | Supply chain fully traceable to plot level | Art. 4(2)(f), 9 | Complete traceability path exists from product to origin plot | Critical |
| CR-19 | Record retention compliance (5-year) | Art. 31 | All due diligence records retained and retrievable | Medium |
| CR-20 | Simplified DD eligibility current | Art. 13 | If using simplified DD, country classification still = Low | High |

**Non-Functional Requirements:**
- Performance: Full scan of 10,000 entities < 15 minutes; incremental scan < 2 minutes for 500 changed entities
- Determinism: Same entity data + same requirements matrix = same compliance score (bit-perfect)
- Auditability: Every scan result stored with timestamp, entity snapshot hash, and requirements version
- Reporting: Export compliance scorecards as PDF, CSV, or JSON for audit documentation

**Dependencies:**
- EUDR-001 Supply Chain Mapping Master (entity graph data)
- EUDR-027 Information Gathering Agent (triggered for data refresh on non-compliance)
- EUDR-028 Risk Assessment Engine (triggered for re-assessment on stale risk data)
- EUDR-030 Documentation Generator (DDS validity data)
- AGENT-DATA-016 Data Freshness Monitor (freshness metadata)

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 backend engineer)

**Edge Cases:**
- Entity is newly onboarded with incomplete data -- score as "Pending Onboarding" rather than "Non-Compliant" for 30-day grace period
- Requirements matrix is updated between scan cycles -- re-scan affected entities against new requirements with version tracking
- Entity data is being updated during scan -- implement snapshot isolation to ensure consistent scan results
- Compliance score calculation depends on external data that is temporarily unavailable -- use last known value with "stale data" flag

---

#### Feature 4: Change Detection Engine

**User Story:**
```
As a supply chain risk analyst,
I want the system to automatically detect and classify changes across all dimensions
of my supply chain -- geographic, organizational, documentary, and commodity --
So that I can distinguish routine updates from compliance-significant changes
and focus my attention on changes that require immediate action.
```

**Acceptance Criteria:**
- [ ] Monitors geographic changes: plot boundary modifications (area change > 5%), new plot registrations, plot subdivisions, plot consolidations, coordinate corrections > 100 meters
- [ ] Monitors organizational changes: supplier ownership transfers, corporate name changes, facility openings/closures, management changes, corporate restructuring, bankruptcy filings, merger/acquisition activity
- [ ] Monitors documentary changes: certification issued/renewed/suspended/revoked/expired, trade license changes, export permit changes, phytosanitary certificate updates, organic certification changes
- [ ] Monitors commodity changes: new commodity added to supplier portfolio, commodity removed, volume changes > 20% from 12-month baseline, new sourcing country added, sourcing country removed
- [ ] Classifies changes by materiality: Material (affects EUDR compliance determination), Significant (may affect compliance if combined with other changes), Routine (administrative update, no compliance impact)
- [ ] Classifies changes by urgency: Immediate (requires action within 24 hours), Near-Term (action within 7 days), Standard (action within 30 days), Informational (no action required)
- [ ] Implements configurable change detection rules with operator-adjustable materiality thresholds per entity type and change dimension
- [ ] Supports compound change detection: identifies when multiple individually routine changes occurring close in time on the same entity constitute a material change pattern (e.g., ownership change + volume increase + new sourcing country = potential circumvention)
- [ ] Generates change events with: entity_id, change_dimension, change_type, materiality, urgency, previous_value, new_value, detection_timestamp, evidence_references, affected_compliance_requirements, recommended_actions
- [ ] Maintains change history timeline per entity with visualization support (sparkline of changes over time)
- [ ] Supports change event correlation across entities: detects when the same type of change occurs across multiple entities simultaneously (e.g., 10 suppliers in the same country all lose certification within a week)

**Change Detection Rule Examples:**

| Rule ID | Dimension | Change Type | Threshold | Materiality | Urgency |
|---------|-----------|-------------|-----------|-------------|---------|
| CDR-001 | Geographic | Plot area change | > 5% area delta | Material | Near-Term |
| CDR-002 | Geographic | Plot coordinate shift | > 100m distance | Material | Immediate |
| CDR-003 | Geographic | New plot registration | Any new plot | Significant | Standard |
| CDR-004 | Organizational | Ownership transfer | Any transfer | Material | Immediate |
| CDR-005 | Organizational | Facility closure | Any closure | Material | Immediate |
| CDR-006 | Organizational | Name change only | Any name change | Routine | Informational |
| CDR-007 | Documentary | Certification expiry | Cert expired | Material | Immediate |
| CDR-008 | Documentary | Certification suspension | Cert suspended | Material | Immediate |
| CDR-009 | Documentary | Certification renewal | Cert renewed | Routine | Informational |
| CDR-010 | Commodity | Volume anomaly | > 20% from baseline | Significant | Near-Term |
| CDR-011 | Commodity | New sourcing country | New country added | Material | Near-Term |
| CDR-012 | Compound | Ownership + volume + country | Concurrent within 30d | Material | Immediate |

**Non-Functional Requirements:**
- Detection latency: < 5 minutes from data change to change event generation
- Accuracy: >= 95% true positive rate; < 5% false positive rate
- Throughput: Process 50,000+ entity data updates per hour
- Storage: Change history retained for 5 years per Article 31
- Configurability: Operators can customize materiality thresholds without code changes

**Dependencies:**
- EUDR-001 Supply Chain Mapping Master (entity baseline data and graph topology)
- EUDR-006 Plot Boundary Manager Agent (plot boundary change data)
- EUDR-007 GPS Coordinate Validator Agent (coordinate change validation)
- EUDR-012 Document Authentication Agent (certification change data)
- EUDR-008 Multi-Tier Supplier Tracker (supplier hierarchy changes)
- AGENT-DATA-016 Data Freshness Monitor (data update timestamps)

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 data engineer)

**Edge Cases:**
- Bulk data import triggers thousands of change events simultaneously -- implement batching with deduplication and consolidation reporting
- Change detection rule conflict (same change matches multiple rules with different severity) -- apply highest severity rule
- Entity data corrected (legitimate correction vs. actual change) -- support "correction" event type that does not trigger compliance alerts
- Historical data backfill creates false change events -- implement backfill mode that suppresses change detection for bulk historical imports
- Circular change detection (A changes -> triggers B check -> B changes -> triggers A check) -- implement change propagation depth limit (default 3 levels)

---

#### Feature 5: Risk Score Degradation Monitor

**User Story:**
```
As a supply chain risk analyst,
I want to continuously track risk score trajectories for all supply chain entities
and receive alerts when a supplier's effective risk score is degrading toward
a higher risk level threshold,
So that I can initiate enhanced due diligence proactively before the entity crosses
a risk level boundary that triggers regulatory obligations.
```

**Acceptance Criteria:**
- [ ] Maintains a real-time risk trajectory for every monitored supply chain entity, updated as external risk signals are received
- [ ] Monitors external risk signals from multiple sources: country CPI changes (Transparency International), certification status changes (FSC/RSPO/PEFC/RA databases), deforestation alert proximity (EUDR-020), competent authority enforcement actions (regulatory feeds), media/NGO reports (news monitoring), supplier compliance history (internal data)
- [ ] Calculates real-time risk score adjustments using deterministic weighted signal aggregation without requiring a full EUDR-017 re-assessment cycle
- [ ] Implements configurable risk degradation thresholds per risk level boundary: Low-to-Standard threshold (default: score reaches 40), Standard-to-High threshold (default: score reaches 70)
- [ ] Generates tiered degradation alerts: Warning (score within 10 points of threshold), Alert (score within 5 points of threshold), Threshold Crossed (score exceeds threshold)
- [ ] Provides risk trajectory visualization per entity: line chart showing risk score over time with threshold lines, signal annotations, and trend projection
- [ ] Calculates risk velocity: the rate of risk score change over time (points per day/week), enabling prediction of when a threshold will be crossed at current trajectory
- [ ] Identifies risk concentration: groups of entities whose risk scores are degrading simultaneously, potentially indicating a systemic issue (e.g., country-wide CPI drop affecting all suppliers in that country)
- [ ] Triggers automated re-assessment workflow through EUDR-028 Risk Assessment Engine when a threshold crossing is confirmed
- [ ] Triggers automated enhanced due diligence workflow through EUDR-026 Due Diligence Orchestrator when an entity crosses from Standard to High risk
- [ ] Maintains risk trajectory history for 5 years with full provenance for audit trail

**Risk Signal Weights:**

| Signal Source | Signal Type | Weight | Update Frequency |
|---------------|------------|--------|-----------------|
| Transparency International | Country CPI change | 0.15 | Annual |
| EC Country Benchmarking | Risk classification change | 0.25 | Quarterly |
| EUDR-020 | Deforestation alert within 5km of entity plot | 0.20 | Real-time |
| Certification Bodies | Certificate suspension/revocation | 0.15 | Daily |
| Competent Authorities | Enforcement action against entity | 0.15 | As published |
| Internal Compliance | Previous non-compliance finding | 0.05 | Per scan cycle |
| Media/NGO | Negative report mentioning entity | 0.05 | Daily |

**Risk Score Adjustment Formula:**
```
Adjusted_Risk_Score = Base_Risk_Score + SUM(Signal_Impact_i * Signal_Weight_i)

Where:
- Base_Risk_Score = Last EUDR-017 calculated score
- Signal_Impact_i = Normalized impact value (0-100) for signal i
- Signal_Weight_i = Configurable weight for signal source i
- All weights sum to 1.0
- Final score capped at [0, 100]
- All calculations deterministic (same signals + same weights = same score)
```

**Non-Functional Requirements:**
- Latency: < 30 minutes from external signal receipt to risk trajectory update
- Determinism: Bit-perfect reproducibility of risk adjustments across runs
- Performance: Risk trajectory update < 100ms per entity; bulk update of 10,000 entities < 5 minutes
- Visualization: Risk trajectory chart renders < 2 seconds for 12-month history
- Storage: Risk trajectory data points stored at 1-hour resolution (8,760 points per entity per year)

**Dependencies:**
- EUDR-017 Supplier Risk Scorer (base risk scores)
- EUDR-020 Deforestation Alert System (deforestation proximity signals)
- EUDR-028 Risk Assessment Engine (triggered for full re-assessment)
- EUDR-026 Due Diligence Orchestrator (triggered for enhanced due diligence)
- EUDR-016 Country Risk Evaluator (country-level risk signal changes)
- EUDR-019 Corruption Index Monitor (CPI data changes)
- OBS-004 Alerting and Notification Platform (degradation alert dispatch)

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 data engineer)

**Edge Cases:**
- Multiple conflicting risk signals received simultaneously (one positive, one negative) -- apply both signals with net effect; do not suppress positive signals
- Base risk score not yet calculated by EUDR-017 (new entity) -- use default risk score based on country + commodity classification until formal assessment
- Risk signal source temporarily unavailable -- retain last known signal value with staleness indicator; do not reset to default
- Entity risk score oscillates around threshold (crossing back and forth) -- implement hysteresis band (5 points) to prevent alert fatigue from threshold oscillation
- Risk velocity calculation during periods of no signal changes -- report velocity as zero with "stable" classification

---

**P0 Features 6-7: Governance and Regulatory Intelligence Layer**

---

#### Feature 6: Data Freshness Enforcement Engine

**User Story:**
```
As a data steward,
I want automated enforcement of EUDR Article 8 data freshness policies
across all 100,000+ data elements in my supply chain,
with proactive alerts before data expires and automated refresh workflow triggering,
So that no DDS submission is ever invalidated by stale supporting data
and I can demonstrate continuous data currency to competent authorities.
```

**Acceptance Criteria:**
- [ ] Defines freshness policies for each data category per EUDR requirements: geolocation verification (12 months), certification validity (per certificate expiry), country risk classification (per EC publication cycle), supplier risk assessment (6 months), satellite deforestation verification (3 months), supplier identification (12 months), production evidence (per production cycle)
- [ ] Tracks the age of every individual data element across all supply chain entities with element-level granularity
- [ ] Generates tiered freshness alerts: Green (> 60 days to expiry), Yellow (30-60 days to expiry), Orange (7-30 days to expiry), Red (< 7 days to expiry), Expired (past freshness window), Critically Overdue (> 30 days past freshness window)
- [ ] Dispatches alert notifications through configurable channels with escalation: Yellow = in-app only; Orange = in-app + email; Red = in-app + email + manager escalation; Expired = in-app + email + SMS + dashboard highlight; Critically Overdue = all channels + compliance officer notification
- [ ] Triggers automated data refresh workflows through EUDR-027 Information Gathering Agent when data elements enter the Orange zone (30 days before expiry)
- [ ] Tracks refresh workflow status: triggered, in-progress, completed, failed, overdue
- [ ] Provides data freshness dashboard: aggregate freshness status across all entities, drill-down by entity/commodity/data-type/country, trend charts showing freshness compliance over time
- [ ] Calculates aggregate freshness score per entity (0-100) based on the freshness status of all its data elements, weighted by criticality
- [ ] Links freshness violations to affected DDS submissions: identifies which active DDS documents are at risk due to stale supporting data
- [ ] Supports operator-configurable freshness policies that can be stricter than EUDR minimums (e.g., company policy requires geolocation re-verification every 6 months instead of 12)
- [ ] Provides freshness compliance report for competent authority inspection with complete audit trail

**Freshness Policy Default Configuration:**

| Data Category | EUDR Minimum | GreenLang Default | Warning Trigger | Refresh Trigger |
|---------------|-------------|-------------------|-----------------|-----------------|
| Geolocation (GPS/polygon) | 12 months | 12 months | 60 days before expiry | 30 days before expiry |
| Certification validity | Per cert expiry | Per cert expiry | 60 days before expiry | 30 days before expiry |
| Country risk classification | Per EC update | Per EC update | Immediate on EC publication | Immediate on EC publication |
| Supplier risk assessment | Not specified | 6 months | 45 days before expiry | 30 days before expiry |
| Satellite verification | Not specified | 3 months | 30 days before expiry | 14 days before expiry |
| Supplier identification | Not specified | 12 months | 60 days before expiry | 30 days before expiry |
| Production evidence | Per production cycle | 12 months | 60 days before expiry | 30 days before expiry |
| DDS submission | Per product placement | Per product placement | 30 days before renewal | 14 days before renewal |

**Non-Functional Requirements:**
- Performance: Full freshness scan of 100,000 data elements < 1 minute
- Accuracy: 100% of stale data elements detected (zero false negatives for expired data)
- Alert timeliness: Freshness alerts generated within 1 hour of data element entering a new tier
- Storage: Freshness status history retained for 5 years per Article 31
- Configurability: Freshness policies adjustable per operator, per commodity, per entity type without code changes

**Dependencies:**
- AGENT-DATA-016 Data Freshness Monitor (core freshness calculation engine)
- EUDR-027 Information Gathering Agent (automated refresh workflow target)
- EUDR-001 Supply Chain Mapping Master (entity data element inventory)
- EUDR-030 Documentation Generator (DDS validity and supporting data linkage)
- OBS-004 Alerting and Notification Platform (alert dispatch infrastructure)

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 backend engineer)

**Edge Cases:**
- Data element has no defined expiry date (e.g., historical production date) -- classify as "No Expiry" with periodic validity review (annual)
- Multiple data elements for the same attribute (e.g., two certifications for same supplier) -- track freshness independently; entity is compliant if at least one is fresh
- Freshness policy changed retroactively -- re-evaluate all existing data elements against new policy; generate retroactive violation report
- Automated refresh workflow fails repeatedly -- escalate to manual refresh with increasing urgency tier after 3 failed attempts
- Data element refreshed with unchanged value -- reset freshness timer; do not require value change for freshness compliance

---

#### Feature 7: Regulatory Update Tracker

**User Story:**
```
As a regulatory affairs specialist,
I want the system to automatically monitor all EUDR-relevant regulatory publications,
classify their impact, and map them to my specific supply chain entities and compliance requirements,
So that I have advance warning of regulatory changes that affect my operations
and can adapt my due diligence processes before new requirements take effect.
```

**Acceptance Criteria:**
- [ ] Monitors regulatory publication feeds: Official Journal of the EU (EUR-Lex RSS/API), European Commission EUDR portal, EU EUDR Information System technical bulletins, 27 EU member state competent authority websites, EUDR Committee proceedings
- [ ] Classifies regulatory updates by type: Implementing regulation, Delegated regulation, Guidance document, Country benchmarking update (Article 29), Annex I amendment, Technical specification change, Enforcement guidance, FAQ update, Penalty guidance
- [ ] Classifies regulatory updates by impact severity: Critical (changes legal obligations within 30 days), High (changes compliance requirements within 90 days), Medium (changes best practices or interpretation), Low (informational, no compliance impact)
- [ ] Maps regulatory updates to affected scope: commodity (which of the 7 commodities affected), country (which countries affected), operator type (large operators, SMEs, traders), supply chain dimension (geographic, organizational, documentary, commodity), EUDR articles (which articles amended or reinterpreted)
- [ ] Performs automated impact assessment: identifies specific supply chain entities affected by the regulatory change, quantifies the number of entities requiring action, estimates compliance effort (hours) to adapt, generates recommended action plan with timeline
- [ ] Tracks Article 29 country benchmarking updates with special handling: detects country reclassification (Low to Standard, Standard to High, or reverse), identifies all supply chain entities sourcing from reclassified countries, triggers automated risk score re-propagation through EUDR-017, alerts operators using simplified due diligence (Article 13) for reclassified countries
- [ ] Provides regulatory change calendar: upcoming effective dates, consultation deadlines, implementation timelines
- [ ] Generates regulatory change impact reports with: change description, affected scope, impact assessment, recommended actions, compliance deadline, evidence of awareness (provenance hash + timestamp for audit trail)
- [ ] Supports manual regulatory update entry for regulatory changes not captured by automated feeds (e.g., informal guidance from competent authority meetings)
- [ ] Maintains regulatory change log with 5-year retention and immutable audit trail for demonstrating regulatory awareness to competent authorities

**Non-Functional Requirements:**
- Detection: 100% of published EUDR regulatory changes detected within 24 hours of publication
- Classification accuracy: >= 95% correct severity and scope classification
- Impact assessment: < 5 minutes from regulatory change detection to impact assessment report
- Availability: >= 99.5% uptime for regulatory feed monitoring
- Coverage: All 27 EU member state competent authority publications monitored
- Auditability: Complete regulatory awareness audit trail with timestamps and provenance hashes

**Dependencies:**
- EUDR-016 Country Risk Evaluator (country risk reclassification impact)
- EUDR-017 Supplier Risk Scorer (risk re-propagation triggers)
- EUDR-001 Supply Chain Mapping Master (entity mapping for impact assessment)
- EUDR-028 Risk Assessment Engine (triggered for re-assessment when regulations change)
- AGENT-DATA-004 API Gateway Agent (regulatory feed API integration)
- OBS-004 Alerting and Notification Platform (regulatory alert dispatch)

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 data engineer)

**Edge Cases:**
- Regulatory publication in language other than English -- support EU official languages (24 languages) with automated translation detection; classify as "Pending Translation Review" until verified
- Contradictory guidance from different member state competent authorities -- flag as "Regulatory Conflict" with both versions referenced; alert regulatory affairs team
- Draft regulation published for consultation vs. final regulation -- track lifecycle stage (Proposal, Consultation, Adopted, In Force) and alert only on final adoption with "advance notice" alerts for consultations
- Regulatory feed temporarily unavailable -- retry with exponential backoff; report feed health in monitoring dashboard; alert if feed unavailable > 24 hours
- Annex I amendment adds new derived products to regulation scope -- identify current supply chain entities that match new product definitions and generate compliance gap report

---

### 6.2 Should-Have Features (P1 -- High Priority)

#### Feature 8: Monitoring Intelligence Dashboard

- Executive-level dashboard with real-time compliance status across all supply chain entities
- Drill-down from aggregate metrics to individual entity monitoring details
- Customizable alert prioritization and filtering
- Scheduled monitoring summary reports (daily, weekly, monthly)
- Integration with GL-EUDR-APP frontend for unified user experience

#### Feature 9: Predictive Compliance Forecasting

- Project compliance score trajectories based on current trends and known upcoming events
- Forecast data freshness expiry clusters to plan refresh campaigns
- Predict risk score degradation based on external signal trends
- Estimate compliance effort required per quarter based on monitoring intelligence

---

### 6.3 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Cross-Operator Anonymized Benchmarking

- Anonymized comparison of monitoring metrics across GreenLang customer base
- Industry-average compliance scores by commodity, country, and company size
- Identification of monitoring best practices from high-performing operators

#### Feature 11: Machine Learning Anomaly Detection

- Unsupervised anomaly detection on monitoring event streams
- Identification of unusual patterns that rule-based detection may miss
- Anomaly explanation with contributing factors for human review

---

### 6.4 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Direct satellite imagery processing (EUDR-020 handles satellite analysis; this agent consumes alerts)
- Supply chain graph modification (EUDR-001 handles graph operations; this agent monitors and reads)
- Risk score calculation from scratch (EUDR-017/028 handle risk assessment; this agent monitors trajectories)
- Automated regulatory compliance remediation (this agent detects and alerts; human decision required for remediation)
- Mobile native application (web responsive monitoring dashboard only for v1.0)
- Supplier communication dispatch (EUDR-031 handles stakeholder communications)
- Integration with non-EUDR regulatory frameworks (CSRD, CBAM, CSDDD monitoring deferred to Phase 2)

---

## 7. Technical Requirements

### 7.1 Architecture Overview

```
                              +------------------------------------------+
                              |           GL-EUDR-APP v1.0               |
                              |   Frontend (React/TS) -- Monitoring UI    |
                              +--------------------+---------------------+
                                                   |
                              +--------------------v---------------------+
                              |          Unified API Layer                |
                              |            (FastAPI)                      |
                              +--------------------+---------------------+
                                                   |
         +-------------------------+---------------+---------------+-------------------------+
         |                         |                               |                         |
+--------v---------+    +---------v---------+          +----------v----------+    +----------v----------+
| AGENT-EUDR-033   |    | AGENT-EUDR-001    |          | AGENT-EUDR-020      |    | AGENT-EUDR-017      |
| Continuous       |<-->| Supply Chain      |          | Deforestation       |    | Supplier Risk       |
| Monitoring Agent |    | Mapping Master    |          | Alert System        |    | Scorer              |
|                  |    |                   |          |                     |    |                     |
| Engines:         |    | - Graph Engine    |          | - Sentinel-2 Client |    | - Risk Calculator   |
| F1: SC Monitor   |    | - Risk Propagation|          | - Landsat Client    |    | - Signal Aggregator |
| F2: Defo Correl. |    | - Gap Analysis    |          | - GFW Client        |    | - Score History     |
| F3: Compliance   |    | - Visualization   |          | - NDVI Calculator   |    |                     |
| F4: Change Det.  |    |                   |          | - Alert Dispatcher  |    |                     |
| F5: Risk Degrad. |    |                   |          |                     |    |                     |
| F6: Freshness    |    |                   |          |                     |    |                     |
| F7: Reg. Tracker |    |                   |          |                     |    |                     |
+--------+---------+    +-------------------+          +---------------------+    +---------------------+
         |
         |         +-------------------+    +-------------------+    +-------------------+
         +-------->| AGENT-EUDR-027    |    | AGENT-EUDR-028    |    | AGENT-DATA-016    |
                   | Information       |    | Risk Assessment   |    | Data Freshness    |
                   | Gathering Agent   |    | Engine            |    | Monitor           |
                   |                   |    |                   |    |                   |
                   | - External DB     |    | - Risk Engine     |    | - Freshness Calc  |
                   | - Cert Verify     |    | - Criteria Eval   |    | - Staleness Det.  |
                   | - Data Normalize  |    | - Score Compose   |    | - Alert Engine    |
                   +-------------------+    +-------------------+    +-------------------+

         +-------------------+    +-------------------+    +-------------------+
         | AGENT-EUDR-026    |    | AGENT-EUDR-029    |    | AGENT-EUDR-030    |
         | Due Diligence     |    | Mitigation        |    | Documentation     |
         | Orchestrator      |    | Measure Designer  |    | Generator         |
         |                   |    |                   |    |                   |
         | - Workflow Engine |    | - Strategy Engine |    | - DDS Generator   |
         | - Stage Manager   |    | - Measure Builder |    | - Package Builder |
         +-------------------+    +-------------------+    +-------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/continuous_monitoring/
    __init__.py                              # Public API exports
    config.py                                # ContinuousMonitoringConfig with GL_EUDR_CM_ env prefix
    models.py                                # Pydantic v2 models for monitoring events, alerts, scan results
    provenance.py                            # ProvenanceTracker: SHA-256 hash chains for all monitoring data
    metrics.py                               # 18 Prometheus self-monitoring metrics (gl_eudr_cm_ prefix)
    supply_chain_monitor.py                  # SupplyChainMonitorEngine: real-time entity surveillance (F1)
    deforestation_correlator.py              # DeforestationCorrelationEngine: alert-to-supply-chain mapping (F2)
    compliance_scanner.py                    # ComplianceScannerEngine: EUDR requirements matrix scanning (F3)
    change_detector.py                       # ChangeDetectionEngine: multi-dimensional change detection (F4)
    risk_degradation_monitor.py              # RiskDegradationMonitor: risk trajectory tracking (F5)
    data_freshness_enforcer.py               # DataFreshnessEnforcementEngine: Article 8 freshness policies (F6)
    regulatory_update_tracker.py             # RegulatoryUpdateTracker: regulatory change monitoring (F7)
    alert_dispatcher.py                      # AlertDispatcher: multi-channel alert routing and escalation
    monitoring_scheduler.py                  # MonitoringScheduler: cron-based scheduling for scan cycles
    event_bus.py                             # EventBus: internal event routing between engines
    setup.py                                 # ContinuousMonitoringService facade (singleton, thread-safe)
    api/
        __init__.py
        router.py                            # FastAPI router (30+ endpoints)
        monitoring_routes.py                 # Supply chain monitoring endpoints
        correlation_routes.py                # Deforestation correlation endpoints
        compliance_routes.py                 # Compliance scanning endpoints
        change_routes.py                     # Change detection endpoints
        risk_routes.py                       # Risk degradation monitoring endpoints
        freshness_routes.py                  # Data freshness enforcement endpoints
        regulatory_routes.py                 # Regulatory update tracking endpoints
        dashboard_routes.py                  # Monitoring dashboard endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Monitoring Event Types
class MonitoringEventType(str, Enum):
    SUPPLIER_STATUS_CHANGE = "supplier_status_change"
    CERTIFICATION_EVENT = "certification_event"
    FACILITY_EVENT = "facility_event"
    OWNERSHIP_TRANSFER = "ownership_transfer"
    DEFORESTATION_ALERT_CORRELATED = "deforestation_alert_correlated"
    COMPLIANCE_VIOLATION_DETECTED = "compliance_violation_detected"
    CHANGE_DETECTED = "change_detected"
    RISK_DEGRADATION = "risk_degradation"
    RISK_THRESHOLD_CROSSED = "risk_threshold_crossed"
    DATA_FRESHNESS_WARNING = "data_freshness_warning"
    DATA_FRESHNESS_EXPIRED = "data_freshness_expired"
    REGULATORY_UPDATE = "regulatory_update"
    COMPOUND_CHANGE = "compound_change"

# Alert Severity
class AlertSeverity(str, Enum):
    CRITICAL = "critical"      # Immediate compliance impact
    HIGH = "high"              # Compliance impact within 30 days
    MEDIUM = "medium"          # Potential compliance impact
    LOW = "low"                # Informational
    INFO = "info"              # No compliance impact

# Monitoring Event
class MonitoringEvent(BaseModel):
    event_id: str                          # UUID
    event_type: MonitoringEventType
    severity: AlertSeverity
    entity_id: str                         # Affected supply chain entity
    entity_type: str                       # supplier, plot, certification, etc.
    operator_id: str                       # Operator tenant
    previous_state: Optional[Dict[str, Any]]
    new_state: Optional[Dict[str, Any]]
    change_summary: str                    # Human-readable change description
    affected_dds_ids: List[str]            # DDS submissions affected
    affected_commodities: List[str]
    affected_countries: List[str]
    recommended_actions: List[str]
    evidence_references: List[str]
    provenance_hash: str                   # SHA-256
    detected_at: datetime
    source: str                            # Monitoring engine that generated event
    metadata: Dict[str, Any]

# Compliance Scan Result
class ComplianceScanResult(BaseModel):
    scan_id: str
    scan_type: str                         # full, incremental, targeted
    entity_id: str
    operator_id: str
    overall_score: float                   # 0-100
    requirement_results: List[RequirementResult]
    non_compliance_count: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    previous_score: Optional[float]
    score_delta: Optional[float]
    remediation_actions: List[str]
    scanned_at: datetime
    requirements_version: str
    entity_snapshot_hash: str              # SHA-256 of entity data at scan time
    provenance_hash: str

# Deforestation Correlation Result
class DeforestationCorrelationResult(BaseModel):
    correlation_id: str
    alert_id: str                          # EUDR-020 alert reference
    alert_polygon: Dict[str, Any]          # GeoJSON
    alert_date: datetime
    alert_confidence: float
    alert_area_hectares: float
    affected_plots: List[AffectedPlot]
    affected_suppliers: List[AffectedSupplier]
    affected_batches: List[str]
    affected_products: List[str]
    affected_dds_ids: List[str]
    correlation_mode: str                  # strict, standard, extended
    impact_severity: str                   # direct_overlap, buffer_zone, indirect
    recommended_actions: List[str]
    correlated_at: datetime
    correlation_duration_ms: int
    provenance_hash: str

# Risk Trajectory Data Point
class RiskTrajectoryPoint(BaseModel):
    entity_id: str
    timestamp: datetime
    base_risk_score: float                 # Last EUDR-017 score
    adjusted_risk_score: float             # After external signal adjustments
    risk_level: str                        # LOW, STANDARD, HIGH
    active_signals: List[RiskSignal]
    risk_velocity: float                   # Points per day
    threshold_distance: float              # Points to next threshold
    provenance_hash: str

# Data Freshness Status
class DataFreshnessStatus(BaseModel):
    element_id: str
    entity_id: str
    data_category: str
    last_verified_at: datetime
    freshness_window_days: int
    expires_at: datetime
    days_until_expiry: int
    freshness_tier: str                    # green, yellow, orange, red, expired, critically_overdue
    refresh_workflow_id: Optional[str]
    refresh_workflow_status: Optional[str]
    linked_dds_ids: List[str]
    provenance_hash: str

# Regulatory Update Record
class RegulatoryUpdate(BaseModel):
    update_id: str
    publication_date: datetime
    source: str                            # EUR-Lex, EC portal, member state, etc.
    update_type: str                       # implementing_regulation, guidance, benchmarking, etc.
    title: str
    summary: str
    impact_severity: AlertSeverity
    affected_commodities: List[str]
    affected_countries: List[str]
    affected_operator_types: List[str]
    affected_articles: List[str]
    affected_entity_count: int
    compliance_deadline: Optional[datetime]
    recommended_actions: List[str]
    source_url: str
    document_hash: str                     # SHA-256 of source document
    detected_at: datetime
    provenance_hash: str
```

### 7.4 Database Schema (New Migration: V119)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_continuous_monitoring;

-- Monitoring events (hypertable for time-series)
CREATE TABLE eudr_continuous_monitoring.monitoring_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    entity_id UUID NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    operator_id UUID NOT NULL,
    previous_state JSONB,
    new_state JSONB,
    change_summary TEXT NOT NULL,
    affected_dds_ids JSONB DEFAULT '[]',
    affected_commodities JSONB DEFAULT '[]',
    affected_countries JSONB DEFAULT '[]',
    recommended_actions JSONB DEFAULT '[]',
    evidence_references JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    source VARCHAR(100) NOT NULL,
    metadata JSONB DEFAULT '{}',
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_continuous_monitoring.monitoring_events', 'detected_at');

-- Compliance scan results (hypertable)
CREATE TABLE eudr_continuous_monitoring.compliance_scan_results (
    scan_id UUID DEFAULT gen_random_uuid(),
    scan_type VARCHAR(20) NOT NULL,
    entity_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    overall_score NUMERIC(5,2) NOT NULL,
    requirement_results JSONB NOT NULL,
    non_compliance_count INTEGER DEFAULT 0,
    critical_count INTEGER DEFAULT 0,
    high_count INTEGER DEFAULT 0,
    medium_count INTEGER DEFAULT 0,
    low_count INTEGER DEFAULT 0,
    previous_score NUMERIC(5,2),
    score_delta NUMERIC(5,2),
    remediation_actions JSONB DEFAULT '[]',
    requirements_version VARCHAR(20) NOT NULL,
    entity_snapshot_hash VARCHAR(64) NOT NULL,
    provenance_hash VARCHAR(64) NOT NULL,
    scanned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_continuous_monitoring.compliance_scan_results', 'scanned_at');

-- Deforestation correlation results (hypertable)
CREATE TABLE eudr_continuous_monitoring.deforestation_correlations (
    correlation_id UUID DEFAULT gen_random_uuid(),
    alert_id UUID NOT NULL,
    alert_polygon GEOMETRY(POLYGON, 4326) NOT NULL,
    alert_date TIMESTAMPTZ NOT NULL,
    alert_confidence NUMERIC(5,4),
    alert_area_hectares NUMERIC(12,2),
    affected_plots JSONB NOT NULL DEFAULT '[]',
    affected_suppliers JSONB NOT NULL DEFAULT '[]',
    affected_batches JSONB DEFAULT '[]',
    affected_products JSONB DEFAULT '[]',
    affected_dds_ids JSONB DEFAULT '[]',
    correlation_mode VARCHAR(20) NOT NULL,
    impact_severity VARCHAR(50) NOT NULL,
    recommended_actions JSONB DEFAULT '[]',
    correlation_duration_ms INTEGER,
    provenance_hash VARCHAR(64) NOT NULL,
    correlated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_continuous_monitoring.deforestation_correlations', 'correlated_at');

-- Risk trajectory data points (hypertable)
CREATE TABLE eudr_continuous_monitoring.risk_trajectories (
    trajectory_id UUID DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    base_risk_score NUMERIC(5,2) NOT NULL,
    adjusted_risk_score NUMERIC(5,2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    active_signals JSONB DEFAULT '[]',
    risk_velocity NUMERIC(8,4) DEFAULT 0.0,
    threshold_distance NUMERIC(5,2),
    provenance_hash VARCHAR(64) NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_continuous_monitoring.risk_trajectories', 'recorded_at');

-- Change detection events (hypertable)
CREATE TABLE eudr_continuous_monitoring.change_events (
    change_id UUID DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    change_dimension VARCHAR(50) NOT NULL,
    change_type VARCHAR(100) NOT NULL,
    materiality VARCHAR(20) NOT NULL,
    urgency VARCHAR(20) NOT NULL,
    previous_value JSONB,
    new_value JSONB,
    rule_id VARCHAR(50),
    affected_requirements JSONB DEFAULT '[]',
    recommended_actions JSONB DEFAULT '[]',
    is_compound BOOLEAN DEFAULT FALSE,
    compound_components JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_continuous_monitoring.change_events', 'detected_at');

-- Data freshness tracking
CREATE TABLE eudr_continuous_monitoring.data_freshness_status (
    element_id UUID DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    data_category VARCHAR(50) NOT NULL,
    last_verified_at TIMESTAMPTZ NOT NULL,
    freshness_window_days INTEGER NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    freshness_tier VARCHAR(30) NOT NULL,
    refresh_workflow_id UUID,
    refresh_workflow_status VARCHAR(30),
    linked_dds_ids JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(entity_id, data_category)
);

-- Data freshness history (hypertable)
CREATE TABLE eudr_continuous_monitoring.data_freshness_history (
    history_id UUID DEFAULT gen_random_uuid(),
    element_id UUID NOT NULL,
    entity_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    data_category VARCHAR(50) NOT NULL,
    freshness_tier VARCHAR(30) NOT NULL,
    days_until_expiry INTEGER,
    action_taken VARCHAR(100),
    provenance_hash VARCHAR(64) NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_continuous_monitoring.data_freshness_history', 'recorded_at');

-- Regulatory updates
CREATE TABLE eudr_continuous_monitoring.regulatory_updates (
    update_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    publication_date TIMESTAMPTZ NOT NULL,
    source VARCHAR(200) NOT NULL,
    update_type VARCHAR(50) NOT NULL,
    title VARCHAR(1000) NOT NULL,
    summary TEXT,
    impact_severity VARCHAR(20) NOT NULL,
    affected_commodities JSONB DEFAULT '[]',
    affected_countries JSONB DEFAULT '[]',
    affected_operator_types JSONB DEFAULT '[]',
    affected_articles JSONB DEFAULT '[]',
    affected_entity_count INTEGER DEFAULT 0,
    compliance_deadline TIMESTAMPTZ,
    recommended_actions JSONB DEFAULT '[]',
    source_url VARCHAR(2000),
    document_hash VARCHAR(64),
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Monitoring watch rules
CREATE TABLE eudr_continuous_monitoring.watch_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    entity_id UUID,
    entity_type VARCHAR(50),
    rule_type VARCHAR(50) NOT NULL,
    monitoring_frequency_minutes INTEGER NOT NULL DEFAULT 60,
    is_active BOOLEAN DEFAULT TRUE,
    configuration JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Freshness policies (operator-configurable)
CREATE TABLE eudr_continuous_monitoring.freshness_policies (
    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    data_category VARCHAR(50) NOT NULL,
    commodity VARCHAR(50),
    freshness_window_days INTEGER NOT NULL,
    warning_days_before INTEGER NOT NULL DEFAULT 60,
    refresh_trigger_days_before INTEGER NOT NULL DEFAULT 30,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(operator_id, data_category, commodity)
);

-- Indexes
CREATE INDEX idx_events_operator ON eudr_continuous_monitoring.monitoring_events(operator_id, detected_at DESC);
CREATE INDEX idx_events_entity ON eudr_continuous_monitoring.monitoring_events(entity_id, detected_at DESC);
CREATE INDEX idx_events_severity ON eudr_continuous_monitoring.monitoring_events(severity, detected_at DESC);
CREATE INDEX idx_events_type ON eudr_continuous_monitoring.monitoring_events(event_type);
CREATE INDEX idx_scans_operator ON eudr_continuous_monitoring.compliance_scan_results(operator_id, scanned_at DESC);
CREATE INDEX idx_scans_entity ON eudr_continuous_monitoring.compliance_scan_results(entity_id, scanned_at DESC);
CREATE INDEX idx_correlations_alert ON eudr_continuous_monitoring.deforestation_correlations(alert_id);
CREATE INDEX idx_trajectories_entity ON eudr_continuous_monitoring.risk_trajectories(entity_id, recorded_at DESC);
CREATE INDEX idx_changes_operator ON eudr_continuous_monitoring.change_events(operator_id, detected_at DESC);
CREATE INDEX idx_changes_entity ON eudr_continuous_monitoring.change_events(entity_id, detected_at DESC);
CREATE INDEX idx_changes_dimension ON eudr_continuous_monitoring.change_events(change_dimension);
CREATE INDEX idx_freshness_operator ON eudr_continuous_monitoring.data_freshness_status(operator_id);
CREATE INDEX idx_freshness_tier ON eudr_continuous_monitoring.data_freshness_status(freshness_tier);
CREATE INDEX idx_freshness_expires ON eudr_continuous_monitoring.data_freshness_status(expires_at);
CREATE INDEX idx_regulatory_severity ON eudr_continuous_monitoring.regulatory_updates(impact_severity);
CREATE INDEX idx_regulatory_date ON eudr_continuous_monitoring.regulatory_updates(publication_date DESC);
CREATE INDEX idx_watch_operator ON eudr_continuous_monitoring.watch_rules(operator_id, is_active);
CREATE INDEX idx_correlations_polygon ON eudr_continuous_monitoring.deforestation_correlations USING GIST(alert_polygon);
```

### 7.5 API Endpoints (30+)

| Method | Path | Description |
|--------|------|-------------|
| **Supply Chain Monitoring** | | |
| POST | `/v1/monitoring/watch-rules` | Create watch rules for entities |
| GET | `/v1/monitoring/watch-rules` | List watch rules (with filters) |
| PUT | `/v1/monitoring/watch-rules/{rule_id}` | Update watch rule configuration |
| DELETE | `/v1/monitoring/watch-rules/{rule_id}` | Deactivate a watch rule |
| GET | `/v1/monitoring/events` | List monitoring events (with filters: severity, type, entity, date range) |
| GET | `/v1/monitoring/events/{event_id}` | Get monitoring event details |
| GET | `/v1/monitoring/status` | Get monitoring status dashboard data |
| **Deforestation Correlation** | | |
| POST | `/v1/monitoring/correlations/run` | Trigger manual correlation for a specific alert |
| GET | `/v1/monitoring/correlations` | List correlation results (with filters) |
| GET | `/v1/monitoring/correlations/{correlation_id}` | Get correlation result details |
| GET | `/v1/monitoring/correlations/by-dds/{dds_id}` | Get all correlations affecting a specific DDS |
| **Compliance Scanning** | | |
| POST | `/v1/monitoring/compliance/scan` | Trigger compliance scan (full, incremental, or targeted) |
| GET | `/v1/monitoring/compliance/results` | List scan results (with filters) |
| GET | `/v1/monitoring/compliance/results/{scan_id}` | Get scan result details |
| GET | `/v1/monitoring/compliance/scorecard/{entity_id}` | Get entity compliance scorecard |
| GET | `/v1/monitoring/compliance/scorecard/commodity/{commodity}` | Get commodity compliance scorecard |
| GET | `/v1/monitoring/compliance/trends` | Get compliance score trends |
| **Change Detection** | | |
| GET | `/v1/monitoring/changes` | List change events (with filters: dimension, materiality, urgency, date range) |
| GET | `/v1/monitoring/changes/{change_id}` | Get change event details |
| GET | `/v1/monitoring/changes/entity/{entity_id}/timeline` | Get change timeline for an entity |
| GET | `/v1/monitoring/changes/correlated` | Get correlated change events (compound changes) |
| **Risk Degradation** | | |
| GET | `/v1/monitoring/risk/trajectory/{entity_id}` | Get risk trajectory for an entity |
| GET | `/v1/monitoring/risk/degradation-alerts` | List risk degradation alerts |
| GET | `/v1/monitoring/risk/concentration` | Get risk concentration analysis |
| GET | `/v1/monitoring/risk/velocity` | Get risk velocity report (fastest degrading entities) |
| **Data Freshness** | | |
| GET | `/v1/monitoring/freshness/status` | Get aggregate freshness status dashboard |
| GET | `/v1/monitoring/freshness/entity/{entity_id}` | Get freshness status for entity data elements |
| GET | `/v1/monitoring/freshness/violations` | List freshness violations (with filters: tier, category) |
| POST | `/v1/monitoring/freshness/policies` | Create or update freshness policy |
| GET | `/v1/monitoring/freshness/policies` | List freshness policies |
| GET | `/v1/monitoring/freshness/dds-impact` | Get DDS documents at risk from freshness violations |
| **Regulatory Updates** | | |
| GET | `/v1/monitoring/regulatory/updates` | List regulatory updates (with filters: type, severity, commodity, country) |
| GET | `/v1/monitoring/regulatory/updates/{update_id}` | Get regulatory update details with impact assessment |
| POST | `/v1/monitoring/regulatory/updates` | Manually register a regulatory update |
| GET | `/v1/monitoring/regulatory/calendar` | Get regulatory change calendar |
| GET | `/v1/monitoring/regulatory/impact/{update_id}` | Get impact assessment for a regulatory update |
| **Health and Admin** | | |
| GET | `/health` | Service health check |
| GET | `/v1/monitoring/admin/data-source-health` | Get health status of all monitored external data sources |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_cm_events_detected_total` | Counter | Monitoring events detected by type and severity |
| 2 | `gl_eudr_cm_events_dispatched_total` | Counter | Alerts dispatched by channel (email, webhook, SMS, in-app) |
| 3 | `gl_eudr_cm_correlations_total` | Counter | Deforestation alert correlations executed |
| 4 | `gl_eudr_cm_correlation_duration_seconds` | Histogram | Deforestation correlation latency |
| 5 | `gl_eudr_cm_compliance_scans_total` | Counter | Compliance scans executed by type (full, incremental, targeted) |
| 6 | `gl_eudr_cm_compliance_scan_duration_seconds` | Histogram | Compliance scan latency |
| 7 | `gl_eudr_cm_changes_detected_total` | Counter | Change events detected by dimension and materiality |
| 8 | `gl_eudr_cm_risk_degradations_total` | Counter | Risk degradation events by threshold type |
| 9 | `gl_eudr_cm_freshness_violations_total` | Counter | Data freshness violations by tier and category |
| 10 | `gl_eudr_cm_freshness_refreshes_triggered_total` | Counter | Automated data refresh workflows triggered |
| 11 | `gl_eudr_cm_regulatory_updates_total` | Counter | Regulatory updates detected by type and severity |
| 12 | `gl_eudr_cm_processing_duration_seconds` | Histogram | Processing operation latency by engine |
| 13 | `gl_eudr_cm_errors_total` | Counter | Errors by engine and error type |
| 14 | `gl_eudr_cm_active_watch_rules` | Gauge | Currently active watch rules |
| 15 | `gl_eudr_cm_monitored_entities` | Gauge | Total entities under monitoring |
| 16 | `gl_eudr_cm_avg_compliance_score` | Gauge | Average compliance score across all entities |
| 17 | `gl_eudr_cm_data_source_health` | Gauge | External data source health status (1 = healthy, 0 = unhealthy) |
| 18 | `gl_eudr_cm_event_queue_depth` | Gauge | Depth of monitoring event processing queue |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Scheduling | APScheduler + Celery Beat | Cron-based monitoring cycle scheduling with distributed execution |
| Event Processing | Redis Streams + Celery | Event-driven monitoring with at-least-once delivery guarantee |
| Spatial | PostGIS + Shapely + GeoJSON | Deforestation alert polygon intersection, buffer zone calculations |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for monitoring events |
| Cache | Redis | Alert deduplication, correlation result caching, entity state caching |
| Object Storage | S3 | Regulatory document archives, correlation report exports |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based monitoring access control |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated monitoring dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment; monitoring engine runs as always-on pod |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-cm:monitoring:read` | View monitoring events and status dashboard | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cm:monitoring:configure` | Create, update, delete watch rules | Analyst, Compliance Officer, Admin |
| `eudr-cm:correlations:read` | View deforestation correlation results | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cm:correlations:execute` | Trigger manual correlation runs | Analyst, Compliance Officer, Admin |
| `eudr-cm:compliance:read` | View compliance scan results and scorecards | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cm:compliance:execute` | Trigger compliance scans (full, incremental, targeted) | Analyst, Compliance Officer, Admin |
| `eudr-cm:changes:read` | View change detection events and timelines | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cm:risk:read` | View risk trajectories and degradation alerts | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cm:freshness:read` | View data freshness status and violations | Viewer, Analyst, Data Steward, Compliance Officer, Admin |
| `eudr-cm:freshness:configure` | Create and update freshness policies | Data Steward, Compliance Officer, Admin |
| `eudr-cm:regulatory:read` | View regulatory updates and impact assessments | Viewer, Analyst, Regulatory Affairs, Compliance Officer, Admin |
| `eudr-cm:regulatory:manage` | Manually register regulatory updates | Regulatory Affairs, Compliance Officer, Admin |
| `eudr-cm:admin:read` | View system health, data source status, queue depth | Admin |
| `eudr-cm:admin:configure` | Configure system-level monitoring parameters | Admin |
| `eudr-cm:export:data` | Export monitoring data (CSV, JSON, PDF) | Analyst, Compliance Officer, Admin |
| `eudr-cm:audit:read` | View monitoring audit trail with provenance hashes | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| EUDR-001 Supply Chain Mapping Master | Supply chain graph, entity registry | Entity data, graph topology, batch records -> monitoring targets |
| EUDR-002 Geolocation Verification Agent | Plot coordinate data | Plot coordinates for deforestation correlation spatial queries |
| EUDR-006 Plot Boundary Manager Agent | Plot polygon data | Plot polygons for spatial intersection with deforestation alerts |
| EUDR-008 Multi-Tier Supplier Tracker | Supplier hierarchy | Supplier relationship changes -> change detection events |
| EUDR-012 Document Authentication Agent | Certification status | Certificate lifecycle events -> monitoring events |
| EUDR-016 Country Risk Evaluator | Country risk data | Country risk changes -> risk degradation signals |
| EUDR-017 Supplier Risk Scorer | Base risk scores | Risk score baselines for trajectory tracking |
| EUDR-019 Corruption Index Monitor | CPI data | CPI changes -> risk degradation signals |
| EUDR-020 Deforestation Alert System | Deforestation alerts | Alert events -> deforestation correlation engine |
| EUDR-027 Information Gathering Agent | Data refresh workflows | Freshness violations -> trigger data refresh |
| EUDR-030 Documentation Generator | DDS data | DDS validity and supporting data linkage |
| AGENT-DATA-003 ERP/Finance Connector | Supplier master data | Supplier data changes -> change detection |
| AGENT-DATA-004 API Gateway Agent | External API integration | Regulatory feeds, certification databases |
| AGENT-DATA-006 GIS/Mapping Connector | PostGIS operations | Spatial queries for deforestation correlation |
| AGENT-DATA-016 Data Freshness Monitor | Freshness metadata | Core freshness calculation support |
| OBS-004 Alerting and Notification Platform | Alert dispatch | Multi-channel alert delivery infrastructure |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| EUDR-026 Due Diligence Orchestrator | Event-driven triggers | Monitoring events trigger due diligence workflows |
| EUDR-028 Risk Assessment Engine | Re-assessment triggers | Risk degradation events trigger full re-assessment |
| EUDR-029 Mitigation Measure Designer | Mitigation triggers | Compliance violations trigger mitigation measure review |
| EUDR-030 Documentation Generator | DDS validity data | Freshness violations and change events feed DDS validity tracking |
| GL-EUDR-APP v1.0 | API integration | Monitoring dashboard data, alerts, compliance scorecards |
| External Auditors | Read-only API + exports | Monitoring audit trail for third-party verification |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Deforestation Alert Impact Assessment (Risk Analyst)

```
1. EUDR-020 detects deforestation event in Riau Province, Indonesia
2. Deforestation Correlation Engine receives alert within 30 seconds
3. Engine performs spatial intersection: 3 palm oil plots overlap alert polygon
4. Engine traces forward through supply chain graph:
   Plot-A -> Mill-1 -> Refinery-X -> Trader-Y -> Importer-Z (this operator)
   Plot-B -> Mill-2 -> Refinery-X -> Trader-Y -> Importer-Z (this operator)
   Plot-C -> Mill-1 -> Refinery-X -> Trader-Y -> Importer-Z (this operator)
5. Engine identifies: 2 active batches in transit, 1 DDS submission affected
6. Engine generates correlated impact assessment (severity: CRITICAL)
7. Risk analyst receives CRITICAL alert via SMS, email, and in-app notification
8. Analyst opens impact assessment in GL-EUDR-APP monitoring dashboard
9. Interactive map shows deforestation polygon overlaid on affected plots
10. Supply chain path highlighted from plots through all intermediaries to importer
11. Analyst clicks "Initiate Enhanced Due Diligence" -> triggers EUDR-026 workflow
12. All actions logged with provenance hashes for competent authority audit trail
```

#### Flow 2: Data Freshness Expiry Prevention (Data Steward)

```
1. Data Freshness Enforcement Engine runs hourly freshness scan
2. Scan identifies: 47 geolocation records entering Orange zone (30 days to expiry)
3. Dashboard updates: 47 entities move from Green to Orange freshness tier
4. Data steward receives daily freshness summary email
5. Steward opens freshness dashboard in GL-EUDR-APP
6. Dashboard shows: 47 Orange, 12 Yellow, 3 Red, 0 Expired
7. For the 47 Orange items, automated refresh workflows already triggered via EUDR-027
8. Steward reviews 3 Red items (< 7 days to expiry) requiring manual intervention
9. Steward contacts suppliers directly for urgent re-verification
10. As refreshed data arrives, freshness timers reset automatically
11. Weekly freshness compliance report shows improving trend (99.2% -> 99.8%)
```

#### Flow 3: Regulatory Change Response (Regulatory Affairs Specialist)

```
1. EC publishes country benchmarking update: Vietnam reclassified from Standard to High risk
2. Regulatory Update Tracker detects publication within 4 hours via EUR-Lex feed
3. Tracker classifies: CRITICAL severity, affects rubber and coffee commodities
4. Impact assessment identifies: 45 supply chain entities sourcing from Vietnam
5. Tracker generates recommended actions:
   a. All 45 entities require enhanced due diligence (Article 10)
   b. 12 DDS submissions using simplified DD (Article 13) are now invalid
   c. Risk scores for all Vietnam-sourcing entities must be re-propagated
6. Regulatory affairs specialist receives CRITICAL alert
7. Specialist reviews impact assessment in monitoring dashboard
8. Clicks "Trigger Risk Re-Propagation" -> EUDR-017 re-calculates Vietnam entity scores
9. Clicks "Invalidate Simplified DD" -> EUDR-030 marks affected DDS for renewal
10. Compliance scanner detects 12 new Article 13 violations in next incremental scan
11. All regulatory awareness actions logged with provenance for competent authority audit
```

#### Flow 4: Compound Change Detection (Compliance Officer)

```
1. Change Detection Engine detects 3 changes on Supplier-47 within 14 days:
   a. Day 1: Ownership transfer (new parent company registered in Malaysia)
   b. Day 7: Volume increase of 35% (above 20% threshold)
   c. Day 14: New sourcing country added (Myanmar)
2. Engine applies compound change rule CDR-012: Ownership + Volume + Country = MATERIAL + IMMEDIATE
3. Engine generates compound change event (severity: CRITICAL)
4. Compliance officer receives immediate alert: "Potential circumvention pattern detected"
5. Alert includes: all 3 individual changes, compound rule match, risk assessment
6. Officer opens entity timeline view: visual timeline of all changes
7. Officer initiates investigation workflow
8. Investigation finds legitimate expansion, not circumvention
9. Officer marks compound event as "Investigated - Cleared" with explanation
10. Audit trail preserved for competent authority inspection
```

### 8.2 Key Screen Descriptions

**Continuous Monitoring Dashboard (Main View):**
- Top bar: aggregate monitoring statistics: entities monitored, events (24h), alerts by severity, overall compliance score, data freshness percentage
- Left column: real-time event feed with severity color coding and entity type icons
- Center: interactive world map showing supply chain entities with color-coded compliance status (green = compliant, yellow = warning, red = non-compliant) and deforestation alert overlay
- Right column: risk trajectory sparklines for top 10 highest-risk entities
- Bottom: data freshness gauge showing percentage of data within freshness windows

**Deforestation Correlation View:**
- Full-screen map with deforestation alert polygon overlay in red
- Affected plot polygons highlighted in orange with overlap percentage labels
- Supply chain path trace lines from affected plots through intermediaries to importer
- Right panel: impact assessment details with affected entities, batches, DDS, and recommended actions
- Action buttons: "Initiate Enhanced Due Diligence", "Suspend Affected Batches", "Generate Audit Report"

**Compliance Scorecard View:**
- Entity header with overall compliance score gauge (0-100)
- 20-requirement compliance matrix with pass/fail/warning status per requirement
- Non-compliance detail cards with remediation guidance and action buttons
- Score trend chart: 30/60/90-day compliance score history
- Comparison view: entity score vs. commodity average vs. portfolio average

**Data Freshness Dashboard:**
- Summary donut chart: % of data elements by freshness tier (Green/Yellow/Orange/Red/Expired)
- Freshness calendar: heat map showing expiry dates over next 90 days
- Violation list: grouped by tier with entity name, data category, days to expiry, and action status
- Refresh workflow tracker: triggered workflows with status (pending/in-progress/completed/failed)
- DDS impact panel: active DDS documents at risk from freshness violations

**Regulatory Change Calendar:**
- Calendar view with regulatory publication dates and compliance deadlines
- Color-coded by impact severity (red = critical, orange = high, yellow = medium, blue = low)
- Click-through to full regulatory update detail with impact assessment
- Filters: by commodity, country, update type, severity
- Right sidebar: upcoming compliance deadlines with countdown timers

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 7 P0 features (Features 1-7) implemented and tested
  - [ ] Feature 1: Real-Time Supply Chain Monitoring Engine -- entity surveillance, watch rules, event generation, alert dispatch
  - [ ] Feature 2: Deforestation Alert Correlation Engine -- spatial intersection, forward tracing, impact assessment, < 60s latency
  - [ ] Feature 3: Automated Compliance Scanner -- full EUDR requirements matrix (20 requirements), entity/commodity/operator scorecards, trend tracking
  - [ ] Feature 4: Change Detection Engine -- geographic, organizational, documentary, commodity dimensions, compound change detection
  - [ ] Feature 5: Risk Score Degradation Monitor -- risk trajectory tracking, external signal integration, threshold crossing alerts, risk velocity
  - [ ] Feature 6: Data Freshness Enforcement Engine -- Article 8 freshness policies, tiered alerts, automated refresh triggering, DDS impact linkage
  - [ ] Feature 7: Regulatory Update Tracker -- regulatory feed monitoring, impact classification, entity mapping, country benchmarking tracking
- [ ] >= 85% test coverage achieved (line coverage); >= 90% branch coverage
- [ ] Security audit passed (JWT + RBAC integrated, 16 permissions registered)
- [ ] Performance targets met:
  - [ ] Monitoring event processing: 10,000+ events per minute
  - [ ] Deforestation correlation: < 60 seconds per alert
  - [ ] Compliance scan: < 15 minutes for 10,000 entities
  - [ ] Data freshness scan: < 1 minute for 100,000 elements
  - [ ] Alert dispatch: < 5 seconds from detection to delivery
- [ ] Monitoring engine uptime verified at >= 99.9% in staging environment (7-day soak test)
- [ ] All 7 EUDR commodity supply chains tested with golden monitoring scenarios
- [ ] Deforestation correlation accuracy validated >= 98% against manually verified test cases
- [ ] Change detection accuracy validated >= 95% true positive, < 5% false positive
- [ ] Data freshness enforcement validated: 100% detection of expired data elements (zero false negatives)
- [ ] API documentation complete (OpenAPI spec for all 30+ endpoints)
- [ ] Database migration V119 tested and validated
- [ ] Integration with EUDR-001, EUDR-017, EUDR-020, EUDR-027, AGENT-DATA-016 verified end-to-end
- [ ] 5 beta customers with continuous monitoring active for >= 14 days
- [ ] No critical or high-severity bugs in backlog
- [ ] Grafana monitoring dashboard operational with all 18 Prometheus metrics

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ operators with continuous monitoring active
- 10,000+ entities under continuous watch
- Average compliance score trending upward (week-over-week improvement)
- < 5 support tickets per customer related to monitoring
- Monitoring engine uptime >= 99.9% in production
- Average deforestation correlation time < 45 seconds
- Data freshness compliance >= 95% across all monitored entities

**60 Days:**
- 200+ operators with continuous monitoring active
- 50,000+ entities under continuous watch
- Average compliance score >= 80% across all monitored entities
- Deforestation alert correlation operational for all 7 commodities
- Data freshness compliance >= 97%
- Regulatory update tracker covering all 27 EU member states
- < 3 support tickets per customer related to monitoring
- Zero undetected compliance-relevant changes persisting > 48 hours

**90 Days:**
- 500+ operators with continuous monitoring active
- 200,000+ entities under continuous watch
- Average compliance score >= 85%
- Data freshness compliance >= 99%
- Zero EUDR penalties for active customers attributable to monitoring gaps
- Regulatory update detection within 24 hours of publication: 100%
- NPS > 50 from compliance officer persona
- Revenue contribution: 15-20% of total GL-EUDR-APP ARR

---

## 10. Timeline and Milestones

### Phase 1: Core Monitoring Infrastructure (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Real-Time Supply Chain Monitoring Engine (Feature 1): entity surveillance framework, watch rules, event bus, alert dispatcher | Senior Backend Engineer + Infrastructure Engineer |
| 2-3 | Monitoring scheduler, event persistence (TimescaleDB hypertables), monitoring dashboard API | Senior Backend Engineer |
| 3-4 | Deforestation Alert Correlation Engine (Feature 2): spatial intersection, forward tracing, impact assessment | Senior Backend Engineer + GIS Specialist |
| 4-5 | Change Detection Engine (Feature 4): multi-dimensional detection rules, compound change analysis | Senior Backend Engineer + Data Engineer |
| 5-6 | Risk Score Degradation Monitor (Feature 5): risk trajectory tracking, external signal integration, threshold alerts | Senior Backend Engineer |

**Milestone: Core monitoring infrastructure operational with Features 1, 2, 4, 5 (Week 6)**

### Phase 2: Compliance and Freshness Engines (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Automated Compliance Scanner (Feature 3): EUDR requirements matrix (20 requirements), scan modes, scorecard generation | Senior Backend Engineer + Backend Engineer |
| 8-9 | Data Freshness Enforcement Engine (Feature 6): freshness policies, tiered alerts, refresh workflow triggering | Senior Backend Engineer + Backend Engineer |
| 9-10 | REST API Layer: 30+ endpoints, authentication, rate limiting, pagination | Backend Engineer |

**Milestone: All analytical engines operational with full API (Week 10)**

### Phase 3: Regulatory Intelligence and Integration (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | Regulatory Update Tracker (Feature 7): regulatory feed monitoring, impact classification, entity mapping, country benchmarking | Senior Backend Engineer + Data Engineer |
| 12-13 | Cross-agent integration: EUDR-001, EUDR-017, EUDR-020, EUDR-027, EUDR-028, EUDR-030, AGENT-DATA-016 | Senior Backend Engineer |
| 13-14 | RBAC integration (16 permissions), Grafana dashboard (18 metrics), end-to-end integration testing | Backend Engineer + DevOps |

**Milestone: All 7 P0 features implemented with full integration (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 800+ tests, golden monitoring scenarios for all 7 commodities, performance tests | Test Engineer |
| 16-17 | Performance testing (10K+ events/min, 60s correlation, 15-min scan), security audit, 7-day soak test for uptime | DevOps + Security |
| 17 | Database migration V119 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers, 14-day monitoring period) | Product + Engineering |
| 18 | Launch readiness review (all 7 P0 features verified) and go-live | All |

**Milestone: Production launch with all 7 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Monitoring Intelligence Dashboard (Feature 8, P1)
- Predictive Compliance Forecasting (Feature 9, P1)
- Cross-operator anonymized benchmarking (Feature 10, P2)
- ML anomaly detection (Feature 11, P2)
- Multi-regulatory framework monitoring (CSRD, CSDDD, CBAM)
- Performance optimization for 500K+ entity monitoring

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable, production-ready; provides entity graph for monitoring |
| EUDR-002 Geolocation Verification Agent | BUILT (100%) | Low | Stable; provides plot coordinate data for correlation |
| EUDR-006 Plot Boundary Manager Agent | BUILT (100%) | Low | Stable; provides plot polygon data for spatial intersection |
| EUDR-008 Multi-Tier Supplier Tracker | BUILT (100%) | Low | Stable; provides supplier hierarchy for change detection |
| EUDR-012 Document Authentication Agent | BUILT (100%) | Low | Stable; provides certification status feeds |
| EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Stable; provides country risk data for regulatory tracking |
| EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Stable; provides base risk scores for trajectory tracking |
| EUDR-019 Corruption Index Monitor | BUILT (100%) | Low | Stable; provides CPI data for risk degradation signals |
| EUDR-020 Deforestation Alert System | BUILT (100%) | Low | Stable; primary input for deforestation correlation engine |
| EUDR-026 Due Diligence Orchestrator | BUILT (100%) | Low | Stable; downstream consumer of monitoring events |
| EUDR-027 Information Gathering Agent | BUILT (100%) | Low | Stable; target for automated data refresh workflows |
| EUDR-028 Risk Assessment Engine | BUILT (100%) | Low | Stable; triggered for full re-assessment on risk degradation |
| EUDR-029 Mitigation Measure Designer | BUILT (100%) | Low | Stable; triggered for mitigation review on compliance violations |
| EUDR-030 Documentation Generator | BUILT (100%) | Low | Stable; provides DDS validity data for freshness tracking |
| AGENT-DATA-003 ERP/Finance Connector | BUILT (100%) | Low | Stable; supplier master data changes |
| AGENT-DATA-004 API Gateway Agent | BUILT (100%) | Low | Stable; external API integration for regulatory feeds |
| AGENT-DATA-006 GIS/Mapping Connector | BUILT (100%) | Low | Stable; PostGIS operations for spatial queries |
| AGENT-DATA-016 Data Freshness Monitor | BUILT (100%) | Low | Stable; core freshness calculation engine |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Integration points defined for monitoring dashboard |
| PostgreSQL + TimescaleDB + PostGIS | Production Ready | Low | Standard infrastructure; hypertables for time-series |
| Redis | Production Ready | Low | Event streaming and caching infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration; 16 new permissions |
| OBS-004 Alerting and Notification Platform | BUILT (100%) | Low | Multi-channel alert dispatch infrastructure |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EUR-Lex API / RSS feed | Available | Low | Well-established EU publication service; cached locally |
| EC EUDR portal | Available | Medium | Web scraping with change detection; fallback to manual entry |
| EU EUDR Information System technical specs | Published | Medium | Adapter pattern for specification version changes |
| EC country benchmarking list (Article 29) | Published; updated periodically | Medium | Database-driven; hot-reloadable on detection |
| Certification body databases (FSC, RSPO, PEFC, RA) | Available via API | Medium | Multi-provider with per-body adapter; circuit breaker on API failure |
| Transparency International CPI | Annual publication | Low | Annual data refresh with automatic detection |
| Satellite data providers (via EUDR-020) | Available | Low | Consumed via EUDR-020, not directly; EUDR-020 handles provider failover |
| Global Forest Watch API | Available | Low | Consumed via EUDR-020; fallback to cached data |
| 27 EU member state competent authority websites | Variable availability | High | Phased rollout starting with top 10 jurisdictions; manual entry fallback for unreliable sources |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | External data source instability (certification body APIs, regulatory feeds) | Medium | High | Circuit breaker pattern per data source; exponential backoff retry; fallback to last known state; data source health dashboard with alerting |
| R2 | Alert fatigue from excessive monitoring events | High | Medium | Configurable severity thresholds; materiality filters; compound event consolidation; daily digest mode for low-severity events; operator-tunable watch rules |
| R3 | Deforestation alert volume spikes (e.g., fire season in Southeast Asia) | Medium | Medium | Event queue with backpressure; priority processing for operators with affected entities; correlation result caching for overlapping alerts |
| R4 | Monitoring engine downtime during critical compliance period | Low | Critical | Kubernetes pod with liveness/readiness probes; automatic restart; Redis-backed event queue for durability during restarts; 99.9% uptime SLO with burn rate alerts |
| R5 | False positive change detection causing unnecessary operator actions | Medium | Medium | Configurable materiality thresholds; hysteresis bands for threshold oscillation; "correction" event type for legitimate data updates; operator feedback loop to tune rules |
| R6 | EC country benchmarking update missed or delayed detection | Low | High | Multiple monitoring channels (EUR-Lex API, EC portal, news monitoring); manual entry fallback; daily coverage audit |
| R7 | Performance degradation at scale (500K+ entities) | Medium | Medium | TimescaleDB hypertable partitioning; Redis caching for hot paths; lazy loading for historical data; horizontal pod scaling |
| R8 | Data freshness refresh workflows fail silently | Medium | High | Workflow status tracking with retry counts; escalation after 3 failures; fallback to manual refresh notification; freshness violation alerts continue until data is actually refreshed |
| R9 | Regulatory feed in non-English language delays impact assessment | Medium | Low | Automated language detection; classification as "Pending Translation Review"; prioritize English-language publications for automated processing |
| R10 | Integration complexity with 16+ upstream agent dependencies | Medium | Medium | Well-defined interfaces with versioned contracts; mock adapters for testing; circuit breaker on each integration point; degraded mode when specific agents unavailable |
| R11 | EUDR regulation delayed or amended, reducing monitoring urgency | Low | Medium | Agent design is regulation-agnostic at the engine level; compliance requirements matrix is configuration-driven; can adapt to regulation changes without engine rewrites |
| R12 | Competitor launches continuous monitoring before GreenLang | Medium | Medium | Deep ecosystem integration with 32 EUDR agents provides moat; deforestation correlation engine is uniquely differentiated; faster iteration cycle |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Supply Chain Monitor Unit Tests | 100+ | Watch rule CRUD, event generation, alert dispatch, entity surveillance, severity classification |
| Deforestation Correlation Tests | 80+ | Spatial intersection, forward tracing, impact assessment, buffer zones, correlation modes |
| Compliance Scanner Tests | 120+ | All 20 requirements validation, scan modes, scorecard generation, trend calculation, determinism |
| Change Detection Tests | 100+ | All 4 dimensions, materiality classification, compound detection, rule engine, threshold tuning |
| Risk Degradation Tests | 80+ | Trajectory tracking, signal aggregation, threshold crossing, velocity calculation, hysteresis |
| Data Freshness Tests | 90+ | Freshness calculation, tiered alerts, policy enforcement, refresh triggering, DDS linkage |
| Regulatory Tracker Tests | 70+ | Feed parsing, classification, impact mapping, country benchmarking, calendar generation |
| API Tests | 80+ | All 30+ endpoints, auth, error handling, pagination, filtering, rate limiting |
| Event Bus Tests | 40+ | Event routing, at-least-once delivery, backpressure, queue depth management |
| Golden Monitoring Tests | 49+ | All 7 commodities x 7 monitoring scenarios (see Section 13.2) |
| Integration Tests | 40+ | Cross-agent integration with EUDR-001/017/020/027/028/030, DATA-016 |
| Performance Tests | 25+ | 10K events/min, 60s correlation, 15-min scan, 1-min freshness, concurrent queries |
| Soak Tests | 5+ | 7-day continuous operation, memory stability, event processing consistency |
| **Total** | **880+** | |

### 13.2 Golden Monitoring Test Scenarios

Each of the 7 commodities will have a dedicated golden monitoring test supply chain with:

1. **Deforestation alert correlation** -- Alert fires near a production plot; verify correct supply chain correlation, affected entity identification, and impact assessment generation
2. **Certification expiry detection** -- Supplier certification expires; verify detection within monitoring cycle, severity classification, and alert dispatch
3. **Country risk reclassification** -- Production country reclassified by EC; verify regulatory update detection, entity impact mapping, and risk re-propagation trigger
4. **Compound change pattern** -- Ownership transfer + volume anomaly + new country; verify compound change detection and circumvention pattern alert
5. **Data freshness violation** -- Geolocation data exceeds 12-month window; verify freshness violation detection, DDS impact linkage, and refresh workflow trigger
6. **Risk threshold crossing** -- Supplier risk score degrades from Standard to High; verify trajectory detection, threshold alert, and enhanced due diligence trigger
7. **Compliance scan degradation** -- Entity compliance score drops below 70%; verify scorecard update, trend detection, and remediation action generation

Total: 7 commodities x 7 scenarios = 49 golden monitoring test scenarios

### 13.3 Performance Test Scenarios

| Scenario | Target | Test Method |
|----------|--------|-------------|
| Event throughput | 10,000+ events/minute sustained | Generate synthetic monitoring events at target rate; measure processing latency and queue depth |
| Deforestation correlation | < 60 seconds per alert | Fire deforestation alerts against supply chains of varying sizes (100, 1K, 10K, 100K entities) |
| Full compliance scan | < 15 minutes for 10,000 entities | Generate 10,000 entity dataset; run full scan; measure wall-clock time |
| Data freshness scan | < 1 minute for 100,000 elements | Generate 100,000 freshness records; run full scan; measure wall-clock time |
| Concurrent API queries | < 200ms p95 under 100 concurrent users | Load test all API endpoints with 100 concurrent connections |
| Monitoring engine memory | < 512 MB baseline, < 2 GB under load | Monitor RSS over 24-hour period with production-like event stream |
| 7-day soak test | Zero memory leaks, zero event loss, >= 99.9% uptime | Run monitoring engine continuously for 7 days with sustained event stream |

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4/12 |
| **Continuous Monitoring** | Always-on surveillance of supply chain compliance status between due diligence cycles |
| **Deforestation Correlation** | Process of linking satellite deforestation alerts to specific supply chain entities, batches, and DDS submissions |
| **Compliance Scorecard** | Quantitative assessment of an entity's compliance with the EUDR requirements matrix (0-100 score) |
| **Change Detection** | Automated identification of changes in supply chain data across geographic, organizational, documentary, and commodity dimensions |
| **Risk Trajectory** | The evolution of an entity's risk score over time, tracked as a time series |
| **Risk Velocity** | The rate of risk score change over time (points per day), used to predict threshold crossings |
| **Risk Degradation** | Increase in an entity's risk score due to external signals, without a formal re-assessment |
| **Data Freshness** | The currency of a data element relative to its defined validity window per EUDR Article 8 |
| **Freshness Tier** | Classification of data element age: Green (> 60 days to expiry), Yellow (30-60), Orange (7-30), Red (< 7), Expired, Critically Overdue |
| **Watch Rule** | A configurable monitoring rule that defines what to monitor, how frequently, and at what severity threshold |
| **Compound Change** | A pattern where multiple individually routine changes on the same entity constitute a material change when analyzed together |
| **Materiality** | Classification of whether a detected change affects EUDR compliance determination |
| **Regulatory Feed** | An automated data source providing regulatory publication updates (e.g., EUR-Lex RSS, EC portal) |
| **Country Benchmarking** | EC classification of countries as Low, Standard, or High risk under EUDR Article 29 |
| **Simplified Due Diligence** | Reduced due diligence requirements for products sourced from Low-risk countries under EUDR Article 13 |
| **Competent Authority** | National authority designated by each EU member state to enforce EUDR |
| **CN Code** | Combined Nomenclature -- EU product classification code |
| **HS Code** | Harmonized System -- international product classification code |
| **Hysteresis Band** | A buffer zone around a threshold that prevents alert oscillation when a value fluctuates near the boundary |

### Appendix B: EUDR Article 8 -- Information in Due Diligence Statements

Per Article 8, the DDS shall contain:
- (a) Name, postal address, and email address of the operator or trader
- (b) HS heading or CN code (6 or 8 digits)
- (c) Description of the relevant commodity or product, including trade name
- (d) Quantity
- (e) Country of production
- (f) Geolocation of all plots of land where the relevant commodity was produced
- (g) Date or time range of production
- (h) Name and postal address of all suppliers in the supply chain
- (i) Name and postal address of the buyer(s)
- (j) Adequately conclusive and verifiable information that the products are deforestation-free
- (k) Adequately conclusive and verifiable information that the products have been produced in accordance with relevant legislation of the country of production

All of these data elements are subject to freshness validation by the Data Freshness Enforcement Engine (Feature 6).

### Appendix C: Article 29 Country Benchmarking Framework

The European Commission classifies countries as Low, Standard, or High risk based on:
1. Rate of deforestation and forest degradation
2. Rate of expansion of agriculture for relevant commodities
3. Production trends of relevant commodities
4. Information from indigenous peoples, local communities, and civil society
5. The country's engagement with the EU on deforestation
6. The existence and enforcement of national laws on deforestation
7. The country's ratification and implementation of international conventions

The Regulatory Update Tracker (Feature 7) monitors EC publications for country benchmarking changes and automatically maps reclassifications to affected supply chain entities.

### Appendix D: Monitoring Event Lifecycle

```
External Signal (certification expiry, deforestation alert, regulatory publication, etc.)
    |
    v
[Monitoring Engine] -- detects change based on watch rules
    |
    v
[Event Bus] -- routes event to appropriate engine
    |
    v
[Engine Processing] -- classifies severity, maps to entities, generates recommendations
    |
    v
[Event Persistence] -- stores in TimescaleDB hypertable with provenance hash
    |
    v
[Alert Dispatcher] -- routes alert to configured channels based on severity
    |
    v
[Operator Dashboard] -- displays in real-time monitoring UI
    |
    v
[Downstream Trigger] -- triggers EUDR-026/027/028/029/030 as appropriate
    |
    v
[Audit Trail] -- complete provenance chain for competent authority inspection
```

### Appendix E: Integration Event Contracts

**Inbound: Deforestation Alert (from EUDR-020)**
```json
{
    "alert_id": "uuid",
    "alert_type": "deforestation",
    "source": "global_forest_watch | sentinel2 | landsat",
    "polygon": { "type": "Polygon", "coordinates": [...] },
    "detected_date": "2026-03-12T00:00:00Z",
    "confidence": 0.95,
    "area_hectares": 12.5,
    "country_code": "ID",
    "metadata": {}
}
```

**Outbound: Monitoring Event (to EUDR-026 Due Diligence Orchestrator)**
```json
{
    "event_id": "uuid",
    "event_type": "deforestation_alert_correlated",
    "severity": "critical",
    "entity_id": "uuid",
    "operator_id": "uuid",
    "affected_dds_ids": ["uuid1", "uuid2"],
    "recommended_actions": [
        "initiate_enhanced_due_diligence",
        "suspend_affected_batches",
        "notify_competent_authority"
    ],
    "correlation_details": {
        "alert_id": "uuid",
        "affected_plots": 3,
        "affected_suppliers": 2,
        "impact_severity": "direct_overlap"
    },
    "provenance_hash": "sha256:...",
    "detected_at": "2026-03-12T14:30:00Z"
}
```

**Outbound: Data Refresh Request (to EUDR-027 Information Gathering Agent)**
```json
{
    "request_id": "uuid",
    "trigger": "data_freshness_violation",
    "entity_id": "uuid",
    "operator_id": "uuid",
    "data_category": "geolocation",
    "current_age_days": 340,
    "freshness_window_days": 365,
    "days_until_expiry": 25,
    "urgency": "orange",
    "requested_at": "2026-03-12T10:00:00Z"
}
```

### Appendix F: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023
2. EU Deforestation Regulation Guidance Document (European Commission)
3. EUDR Technical Specifications for the Information System
4. Commission Implementing Regulation on the EU Information System for EUDR
5. EUDR Article 29 Country Benchmarking Methodology
6. ISO 22095:2020 -- Chain of Custody -- General Terminology and Models
7. FSC Chain of Custody Standard (FSC-STD-40-004)
8. RSPO Supply Chain Certification Standard
9. Global Forest Watch Technical Documentation
10. Transparency International Corruption Perceptions Index Methodology
11. EUR-Lex Publication Feed Technical Documentation
12. PostGIS Spatial Operations Reference
13. TimescaleDB Hypertable Best Practices for Time-Series Data

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
| 1.0.0-draft | 2026-03-12 | GL-ProductManager | Initial draft created: 7 P0 features (Real-Time Supply Chain Monitoring, Deforestation Alert Correlation, Automated Compliance Scanner, Change Detection Engine, Risk Score Degradation Monitor, Data Freshness Enforcement, Regulatory Update Tracker), full regulatory mapping (Articles 4/8/9/10/11/13/14-16/29/31), 30+ API endpoints, 18 Prometheus metrics, 16 RBAC permissions, V119 migration schema, 880+ test target, 49 golden scenarios |
