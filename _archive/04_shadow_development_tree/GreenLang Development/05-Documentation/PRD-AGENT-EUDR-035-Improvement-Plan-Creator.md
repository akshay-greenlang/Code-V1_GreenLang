# PRD: AGENT-EUDR-035 -- Improvement Plan Creator Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-035 |
| **Agent ID** | GL-EUDR-IPC-035 |
| **Component** | Improvement Plan Creator Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence (Category 5) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-12 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 8 (due diligence system), 10 (risk assessment/mitigation), 11 (risk mitigation measures), 13 (continuous improvement and simplified DD review), 14-16 (competent authority inspections), 29 (country benchmarking), 31 (record keeping); EU Corporate Sustainability Due Diligence Directive (CSDDD) Article 9 (remediation), Article 10 (prevention); ISO 31000:2018 Risk Management; ISO 14001:2015 (PDCA continuous improvement); OECD Guidelines for Multinational Enterprises Chapter IV |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **Prerequisite Agents** | AGENT-EUDR-028 (Risk Assessment Engine), AGENT-EUDR-029 (Mitigation Measure Designer), AGENT-EUDR-032 (Grievance Mechanism Manager), AGENT-EUDR-033 (Continuous Monitoring Agent), AGENT-EUDR-034 (Annual Review Scheduler) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Article 13 mandates that operators and traders "review and, where necessary, update their due diligence" to ensure continuous improvement of their compliance posture. Article 10 requires ongoing risk assessment that is "adequate and proportionate" to the complexity of the supply chain. Article 11 requires that risk mitigation measures be maintained and adapted when their effectiveness deteriorates or new risks emerge. The CSDDD (Corporate Sustainability Due Diligence Directive) Article 9 mandates remediation of adverse impacts, and Article 10 requires companies to adopt prevention plans that address root causes of identified harms. ISO 14001:2015 establishes the Plan-Do-Check-Act (PDCA) continuous improvement cycle as the gold standard for environmental management systems. Together, these regulatory and standards frameworks demand not just point-in-time compliance, but a structured, documented, measurable, and continuously improving compliance system.

The GreenLang platform has built a comprehensive suite of 34 EUDR agents spanning the full due diligence lifecycle. Supply Chain Traceability agents (EUDR-001 through EUDR-015) handle information gathering -- supply chain mapping, geolocation verification, satellite monitoring, forest cover analysis, land use change detection, plot boundary management, GPS coordinate validation, multi-tier supplier tracking, chain of custody, segregation verification, mass balance calculation, document authentication, blockchain integration, QR code generation, and mobile data collection. Risk Assessment agents (EUDR-016 through EUDR-020) handle multi-dimensional risk scoring -- country risk evaluation, supplier risk scoring, commodity risk analysis, corruption index monitoring, and deforestation alerting. Due Diligence agents (EUDR-021 through EUDR-034) handle indigenous rights checking, protected area validation, legal compliance verification, third-party audit management, risk mitigation advisory, due diligence orchestration, information gathering coordination, risk assessment engine operation, mitigation measure design, documentation generation, stakeholder engagement, grievance mechanism management, continuous monitoring, and annual review scheduling.

These 34 agents collectively generate an enormous volume of compliance intelligence: risk scores across five dimensions, compliance gaps against every EUDR article, grievance patterns across geographies and commodities, deforestation alerts correlated with supply chains, audit findings from third-party verifications, annual review comparisons showing year-over-year changes, mitigation effectiveness measurements, continuous monitoring events, and data freshness violations. However, this intelligence currently exists as isolated outputs from independent agents. There is no engine that consolidates these findings, identifies the root causes behind recurring issues, prioritizes improvements by impact and urgency, generates actionable improvement plans with SMART objectives, assigns responsibilities using RACI matrices, tracks implementation progress, and measures the effectiveness of improvements over time.

Today, EU operators face the following critical gaps in continuous improvement:

- **No cross-agent finding aggregation**: The 34 upstream agents produce findings in their own formats -- EUDR-028 produces risk scores with five-dimensional decomposition, EUDR-032 produces grievance analytics with pattern detection, EUDR-033 produces continuous monitoring events with compliance scorecards, EUDR-034 produces annual review comparisons with year-over-year delta analysis, and EUDR-029 produces mitigation effectiveness measurements. There is no aggregation engine that normalizes, deduplicates, correlates, and prioritizes findings from all these sources into a unified findings register. Compliance officers must manually review outputs from each agent, cross-reference findings across agents, identify which findings from different agents represent the same underlying issue, and assemble a holistic picture of their compliance posture. For a large operator with 500+ active DDS submissions across seven commodities, this manual aggregation takes 80-120 hours per quarter and produces incomplete, inconsistent results.

- **No systematic gap analysis against EUDR articles**: EUDR compliance requires adherence to a complex web of interrelated articles. Article 4 defines the due diligence obligation. Article 9 specifies information gathering requirements. Article 10 enumerates risk assessment criteria (a) through (k). Article 11 mandates mitigation adequacy. Article 12 defines DDS content requirements. Article 13 mandates review and continuous improvement. Article 29 establishes country benchmarking. Article 31 requires five-year record retention. While individual agents address specific articles within their scope, there is no systematic engine that maps an operator's entire compliance posture against every relevant EUDR article, identifies gaps and partial compliance at the article level, prioritizes gaps by enforcement severity and penalty exposure, and generates a structured gap register with remediation pathways. Operators cannot answer the fundamental question: "For each EUDR article, where exactly are we falling short, how severe is the gap, and what must we do to close it?"

- **No root cause analysis across findings**: When EUDR-028 identifies elevated supplier risk for 12 suppliers in Indonesia, EUDR-033 detects certification expiries for 8 of those same suppliers, EUDR-032 receives grievances from communities near 5 of those suppliers, and EUDR-034's annual review shows declining compliance scores for the Indonesian supply chain segment over two consecutive years, these are symptoms of a systemic root cause -- not four independent problems. Without structured root cause analysis using methodologies such as 5-Whys decomposition and Ishikawa (fishbone) diagramming, operators treat each symptom independently. They renew certifications, address individual grievances, implement targeted mitigation for elevated risk scores, and adjust annual review checklists -- but the underlying root cause (for example, a regional deforestation driver, a supplier governance failure, or an inadequate capacity building program) persists and continues generating new symptoms. Manual root cause analysis across 34 agent outputs is practically impossible at scale.

- **No structured improvement plan generation**: Even when operators identify compliance gaps and root causes, translating those findings into actionable improvement plans requires significant effort. Each improvement action must be Specific (clearly defined scope), Measurable (quantifiable success criteria), Achievable (realistic given available resources), Relevant (directly addresses an identified gap or root cause), and Time-bound (clear deadline). For a compliance officer managing 200+ active findings, manually designing SMART improvement actions for each finding, estimating resource requirements, sequencing actions by dependency, and compiling these into a coherent improvement plan takes 40-60 hours and produces plans that vary wildly in quality depending on the individual's experience and methodology.

- **No prioritization framework**: Not all compliance gaps are equally urgent or impactful. A missing polygon for a 5-hectare plot in a high-risk country (Article 9(1)(d) violation, high enforcement exposure, immediate penalty risk) is more urgent than an incomplete supplier questionnaire for a low-risk country supplier (Article 4(2) gap, low enforcement exposure, future risk). Without a structured prioritization framework that considers enforcement severity, penalty exposure, risk impact, resource requirements, dependency chains, and deadline proximity, operators allocate improvement resources based on which finding arrived most recently or which compliance officer has the loudest voice -- not on objective, risk-weighted priority.

- **No progress tracking or effectiveness measurement**: Once improvement plans are created (manually), there is no systematic tracking of implementation progress, no measurement of whether implemented improvements actually achieved their intended effect, no comparison of planned versus actual resource consumption, and no feedback loop that captures improvement outcomes to inform future planning. Improvement plans are created in documents and spreadsheets, shared by email, and tracked through ad-hoc status meetings. Actions are forgotten, deadlines slip, effectiveness is assumed rather than verified, and the entire continuous improvement cycle lacks the rigor and traceability that EUDR Articles 13-16 demand for regulatory inspection.

- **No stakeholder assignment and accountability**: Improvement plans require coordinated action across multiple stakeholders: compliance officers design plans, supply chain analysts update data, procurement managers engage suppliers, risk analysts refresh assessments, auditors conduct verifications, and management approves resource allocation. Without structured RACI (Responsible, Accountable, Consulted, Informed) matrix assignment, role-based task routing, deadline tracking, and escalation workflows, improvement actions lack clear ownership. When no one is accountable for a specific improvement action, that action does not get done.

- **No audit trail for continuous improvement**: EUDR Articles 14-16 grant competent authorities the power to inspect an operator's due diligence system at any time. Article 13 specifically requires continuous improvement. Competent authorities may ask operators to demonstrate: what findings were identified, how they were prioritized, what improvement actions were planned, who was responsible, what progress was made, what outcomes were achieved, and how those outcomes fed back into the next improvement cycle. Without a structured audit trail that records every step of the continuous improvement lifecycle with timestamps, actor identification, and provenance hashes, operators cannot demonstrate to regulators that they have a functioning continuous improvement system -- even if they are informally improving.

Without solving these problems, EU operators cannot fulfill the continuous improvement mandate of EUDR Article 13 or the remediation and prevention requirements of CSDDD Articles 9-10. They possess the analytical intelligence from 34 upstream agents but lack the structured process to translate that intelligence into measurable improvement. This exposes them to enforcement action under Articles 23-25 (penalties up to 4% of annual EU turnover), competent authority findings of inadequate due diligence systems under Articles 14-16, and reputational damage from failure to demonstrate systematic continuous improvement to stakeholders, investors, and certification bodies.

### 1.2 Solution Overview

Agent-EUDR-035: Improvement Plan Creator is a specialized continuous improvement engine that transforms the collective intelligence output of all 34 upstream EUDR agents into structured, prioritized, actionable, tracked, and measurable improvement plans. It consolidates findings from risk assessments (EUDR-028), grievance mechanisms (EUDR-032), continuous monitoring (EUDR-033), annual reviews (EUDR-034), and mitigation effectiveness tracking (EUDR-029) into a unified findings register; maps those findings against every relevant EUDR article to identify compliance gaps; applies root cause analysis methodologies to connect recurring symptoms to systemic causes; generates SMART improvement actions with resource estimates; prioritizes actions using Eisenhower matrix and risk-weighted scoring; assigns stakeholders using RACI matrices; tracks implementation progress with milestone management; measures improvement effectiveness through before-and-after comparison; and generates audit-ready continuous improvement documentation for competent authority inspection. It operates as a purely deterministic, zero-hallucination workflow and computation engine with no LLM in the critical path.

Core capabilities:

1. **Finding Aggregator** -- Consolidates findings from all upstream EUDR agents into a unified, normalized, deduplicated findings register. Ingests risk assessment results from EUDR-028 (composite scores, five-dimensional decomposition, Article 10(2) criteria evaluations, risk trend data), grievance intelligence from EUDR-032 (pattern analytics, root cause analyses, remediation effectiveness scores, collective grievance impacts), continuous monitoring events from EUDR-033 (compliance alerts, deforestation correlations, risk degradation warnings, data freshness violations, regulatory update impacts), annual review findings from EUDR-034 (year-over-year comparison deltas, missed deadline incidents, review checklist failures), mitigation effectiveness data from EUDR-029 (pre/post risk score comparisons, measure implementation completeness, verification results), and audit findings from EUDR-024 (third-party audit non-conformances, corrective action requirements). Normalizes all findings to a common schema with severity classification, EUDR article mapping, affected supply chain scope, temporal context, and evidence references. Deduplicates findings where multiple agents report the same underlying issue. Correlates related findings across agents using configurable correlation rules.

2. **Gap Analysis Engine** -- Systematically maps the operator's compliance posture against every relevant EUDR article and generates a structured gap register. Evaluates compliance against Article 4 (due diligence obligation completeness), Article 9 (information gathering requirements -- product description, quantities, country of production, geolocation, supplier identification), Article 10(2) criteria (a) through (k) (risk assessment comprehensiveness), Article 11 (mitigation adequacy and proportionality), Article 12 (DDS content and validity), Article 13 (review and continuous improvement), Article 29 (country benchmarking alignment), and Article 31 (record retention). For each article, calculates a compliance score (0-100), identifies specific gaps, classifies gap severity (Critical, High, Medium, Low), estimates enforcement exposure (penalty probability and magnitude), and generates remediation pathways. Produces a compliance heat map showing article-by-article compliance with drill-down capability.

3. **Action Plan Generator** -- Generates SMART (Specific, Measurable, Achievable, Relevant, Time-bound) improvement actions for every identified finding and gap. Each action includes: unique action ID, title, detailed description, linked finding IDs, linked gap IDs, linked EUDR article references, SMART criteria definition (specific scope, measurable success metric with target value, achievability assessment with prerequisites, relevance justification linking to regulatory requirement, time-bound deadline with milestones), resource estimate (personnel hours, cost range in EUR, external vendor requirements), dependency chain (actions that must complete before this action can start), risk if not implemented (regulatory exposure, penalty estimate, operational impact), and evidence requirements for completion verification. Supports three plan types: Corrective Action Plans (address existing non-conformances), Preventive Action Plans (prevent anticipated non-conformances), and Continuous Improvement Plans (enhance compliance beyond minimum requirements). Generates plans at portfolio level (all commodities), commodity level, supply chain level, and individual entity level.

4. **Root Cause Mapper** -- Applies structured root cause analysis methodologies to connect recurring findings to systemic causes. Implements 5-Whys decomposition: for each finding, recursively asks "Why did this occur?" through five levels, requiring evidence linkage at each level, converging to a root cause that can be addressed systemically. Implements Ishikawa (fishbone) diagram generation across six cause categories adapted for EUDR compliance: People (capacity gaps, training deficits, accountability failures), Process (workflow gaps, quality gate failures, handoff errors), Policy (regulatory interpretation errors, threshold miscalibration, scope exclusions), Place (geographic risk factors, infrastructure limitations, connectivity gaps), Product (commodity-specific traceability challenges, transformation complexity, batch mixing), and Partner (supplier governance failures, certification body limitations, auditor capacity). Implements cross-finding correlation analysis that identifies common causal factors across seemingly unrelated findings from different agents, geographies, and commodities. Generates root cause registers with confidence scoring, evidence chains, and recommended systemic interventions. Tracks root cause resolution over time to verify that addressing the root cause eliminates the associated symptoms.

5. **Prioritization Matrix** -- Applies structured prioritization to rank improvement actions by objective, risk-weighted criteria. Implements Eisenhower matrix classification across two dimensions: Urgency (driven by deadline proximity, enforcement timeline, penalty imminence) and Importance (driven by risk severity, compliance gap magnitude, stakeholder impact). Produces four quadrants: Q1 Do First (urgent and important -- immediate regulatory deadlines, critical compliance gaps), Q2 Schedule (important but not urgent -- strategic improvements, capacity building, system enhancements), Q3 Delegate (urgent but less important -- routine compliance tasks, data refresh, minor corrections), Q4 Eliminate/Defer (neither urgent nor important -- nice-to-have improvements, deferred features). Within each quadrant, applies risk-weighted scoring using the formula: Priority_Score = (Severity_Weight * 0.30) + (Enforcement_Exposure * 0.25) + (Risk_Impact * 0.20) + (Resource_Efficiency * 0.15) + (Dependency_Criticality * 0.10). All scoring is deterministic, configurable per operator, and reproducible. Generates prioritized action backlogs, resource allocation recommendations, and critical path analysis.

6. **Progress Tracker** -- Tracks the implementation lifecycle of every improvement action through defined states: PLANNED, APPROVED, IN_PROGRESS, BLOCKED, COMPLETED, VERIFIED, CLOSED. Records milestones with timestamp and actor identification. Tracks resource consumption (actual versus planned personnel hours and cost). Measures implementation velocity (actions completed per period). Detects and alerts on blocked actions, overdue actions, and stalled progress. Generates progress dashboards showing overall plan completion percentage, action status distribution, burndown charts (planned versus actual), and trending. Implements effectiveness measurement: after an action is completed, compares the relevant compliance metric before and after implementation to quantify improvement. For example, if an action aimed to reduce supplier risk scores by 20 points, the tracker measures actual pre/post risk score change and calculates achievement percentage. Feeds effectiveness data back to the Action Plan Generator and Prioritization Matrix to improve future planning.

7. **Stakeholder Assignment Engine** -- Assigns improvement actions to responsible parties using structured RACI (Responsible, Accountable, Consulted, Informed) matrices. Defines role-based assignment rules: Compliance Officers are typically Accountable for regulatory gap closure, Supply Chain Analysts are Responsible for data quality improvements, Procurement Managers are Responsible for supplier engagement actions, Risk Analysts are Responsible for risk assessment refresh actions, External Auditors are Consulted for verification actions, and Management is Informed of all plan progress. Supports individual and team assignment. Implements workload balancing: detects when a stakeholder is overloaded (configurable maximum concurrent actions per person) and recommends rebalancing. Sends automated notifications on assignment, deadline approach, overdue, and escalation. Generates accountability reports showing assignment distribution, completion rates per stakeholder, and overdue rates per stakeholder. Maintains full audit trail of all assignments, reassignments, and escalations.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Finding aggregation coverage | 100% of findings from all 34 upstream agents consolidated | % of agent outputs ingested and normalized |
| Finding deduplication accuracy | >= 95% correct deduplication (no false merges, no missed duplicates) | Precision/recall against manually validated sample |
| Gap analysis article coverage | 100% of relevant EUDR articles evaluated per operator | Article coverage matrix per gap analysis run |
| SMART action quality | 100% of generated actions pass SMART validation (all 5 criteria defined) | Automated SMART criteria completeness check |
| Root cause identification rate | >= 80% of recurring findings traced to documented root cause within 60 days | % of recurrent findings with root cause mapping |
| Prioritization consistency | 100% deterministic, reproducible prioritization across runs | Bit-perfect reproducibility tests |
| Action plan generation time | < 10 seconds for portfolio-level plan covering 500+ findings | Plan generation benchmarks |
| Progress tracking coverage | 100% of actions have tracked lifecycle status | % of actions with defined state |
| Stakeholder assignment coverage | 100% of actions assigned with RACI matrix | % of actions with R and A roles defined |
| Improvement effectiveness rate | >= 70% of completed actions achieve >= 80% of target improvement | Actual versus target metric comparison |
| Overdue action detection | 100% of overdue actions flagged within 1 hour of deadline | Alert latency benchmarks |
| Audit trail completeness | 100% of plan lifecycle events recorded with provenance | Audit trail gap analysis |
| Regulatory acceptance | 100% of improvement documentation accepted in competent authority inspections | Inspection outcome tracking |
| Bit-perfect reproducibility | Same input produces identical output across runs | Reproducibility test suite with SHA-256 hash comparison |
| Test coverage | 300+ tests, >= 85% line coverage | Test suite metrics |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated continuous improvement and compliance management market of 2-4 billion EUR as operators require systematic tools to demonstrate the ongoing improvement mandated by Article 13 and to prepare for competent authority inspections under Articles 14-16.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities requiring automated improvement planning, gap analysis, and progress tracking to maintain continuous EUDR compliance, estimated at 400-800M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 25-45M EUR in improvement planning module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) of EUDR-regulated commodities managing complex multi-commodity, multi-country portfolios where continuous improvement across hundreds of DDS submissions requires systematic automation
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya) with supply chains spanning 10+ countries and 500+ suppliers, generating volumes of compliance findings that cannot be manually aggregated and prioritized
- Timber and paper industry operators with 5-8 tier supply chains and complex processing transformations that generate recurring traceability gaps requiring systematic root cause analysis
- Meat and leather importers (cattle) managing pasture rotation and animal movement traceability challenges that produce commodity-specific improvement requirements

**Secondary:**
- Compliance consultants and advisory firms managing EUDR improvement programs for multiple clients who require standardized improvement planning methodologies and progress reporting
- Certification bodies (FSC, RSPO, Rainforest Alliance, UTZ) that require operators to demonstrate continuous improvement as a condition of certification renewal and who need structured evidence of improvement plan implementation
- Third-party auditors who assess the adequacy of operators' due diligence systems under Articles 14-16 and who require access to structured improvement plans and progress evidence
- SME importers (1,000-10,000 shipments/year) entering EUDR compliance from June 30, 2026, who need guided improvement planning to close initial compliance gaps efficiently with limited resources

**Tertiary:**
- Competent authorities in EU member states who inspect operators' due diligence systems and require access to structured, auditable continuous improvement documentation
- Industry associations developing sector-wide improvement benchmarks and best-practice standards for EUDR compliance
- Investors and ESG rating agencies evaluating companies' regulatory compliance maturity and continuous improvement track records

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Spreadsheet | No cost; familiar format; flexible | Cannot aggregate findings from 34 agents; no automated root cause analysis; no structured prioritization; no progress tracking at scale; 80-120 hours per quarter | Automated aggregation, root cause, prioritization, and tracking in < 10 seconds |
| Generic GRC Platforms (Archer, ServiceNow GRC, MetricStream) | Enterprise integration; broad compliance scope; established workflow engines | Not EUDR-specific; no Article 10(2) criteria mapping; no EUDR gap analysis; no understanding of commodity-specific supply chain traceability requirements; no integration with deforestation monitoring | Purpose-built for EUDR Articles 4-31; integrated with 34 EUDR-specific agents; commodity-aware improvement planning |
| Consulting Firms (Big 4, boutique EUDR advisors) | Deep expertise; bespoke analysis; regulatory interpretation | EUR 100K-500K per engagement; 3-6 month delivery timelines; non-reproducible; no continuous tracking; no real-time aggregation | Automated, reproducible, continuous; EUR 5K-20K annual; instant plan generation; always-on progress tracking |
| Niche EUDR Compliance Tools (Preferred by Nature, Ecosphere+) | Commodity expertise; EUDR awareness | Single-commodity focus; no cross-agent aggregation; limited root cause analysis; no RACI workflow; no effectiveness measurement | All 7 commodities; 34-agent integration; full root cause and RACI; effectiveness measurement loop |
| In-house Custom Builds | Tailored to organization; full control | 12-18 month build cycle; no regulatory update mechanism; no cross-organization benchmarking; no scale | Ready now; continuous regulatory updates; production-grade; benchmarking-ready |

### 2.4 Differentiation Strategy

1. **34-agent intelligence fusion** -- The only improvement planning engine that aggregates and correlates findings from 34 specialized EUDR compliance agents spanning supply chain traceability, risk assessment, environmental due diligence, grievance management, continuous monitoring, and annual review scheduling.
2. **Regulatory fidelity** -- Every gap, finding, and improvement action maps to specific EUDR articles with enforcement severity and penalty exposure estimates. Not a generic compliance tool adapted for EUDR, but purpose-built for EUDR Article 13 continuous improvement.
3. **Structured root cause methodology** -- 5-Whys and Ishikawa analysis adapted specifically for EUDR supply chain compliance, with cross-agent correlation that identifies systemic causes invisible to single-agent analysis.
4. **Zero-hallucination determinism** -- All aggregation, gap analysis, prioritization, and scoring uses deterministic computation with Decimal arithmetic and no LLM in the critical path. Every output is bit-perfect reproducible and auditable.
5. **Closed-loop effectiveness measurement** -- Not just plan generation but outcome verification. Before-and-after metric comparison proves that improvements actually worked, feeding outcomes back to improve future planning.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to demonstrate EUDR Article 13 continuous improvement to competent authorities | 100% of customers pass Article 13 inspection requirements | Q2 2026 |
| BG-2 | Reduce time to create compliance improvement plans from 80-120 hours to < 1 hour | 99% reduction in improvement planning time | Q2 2026 |
| BG-3 | Improve compliance posture measurably year over year for active customers | Average 15-point improvement in composite compliance score per annual cycle | Q4 2026 |
| BG-4 | Become the reference continuous improvement solution for EUDR compliance | 500+ enterprise customers using improvement planning | Q4 2026 |
| BG-5 | Reduce EUDR enforcement penalties for customers through proactive gap closure | Zero penalty actions attributable to lack of continuous improvement for active customers | Ongoing |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Unified finding aggregation | Consolidate findings from all 34 upstream EUDR agents into a single, normalized, deduplicated findings register |
| PG-2 | Article-level gap analysis | Map operator compliance against every relevant EUDR article and generate structured gap register |
| PG-3 | SMART action planning | Generate improvement actions that pass 100% SMART validation with resource estimates and dependency chains |
| PG-4 | Systematic root cause analysis | Connect recurring findings to systemic root causes using 5-Whys and Ishikawa methodologies |
| PG-5 | Risk-weighted prioritization | Prioritize improvement actions by enforcement severity, risk impact, resource efficiency, and deadline proximity |
| PG-6 | Stakeholder accountability | Assign all actions using RACI matrices with automated notification and escalation |
| PG-7 | Progress and effectiveness tracking | Track action lifecycle with effectiveness measurement and closed-loop feedback |
| PG-8 | Audit-ready documentation | Generate competent authority-ready continuous improvement documentation with full provenance |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Finding aggregation throughput | Process 10,000+ findings per minute from all upstream agents |
| TG-2 | Gap analysis latency | < 5 seconds for full article-by-article gap analysis per operator |
| TG-3 | Action plan generation | < 10 seconds for portfolio-level plan covering 500+ findings |
| TG-4 | Root cause computation | < 3 seconds for 5-Whys decomposition per finding chain |
| TG-5 | Prioritization calculation | < 2 seconds for risk-weighted prioritization of 1,000+ actions |
| TG-6 | API response time | < 200ms p95 for standard queries |
| TG-7 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-8 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |

---

## 4. User Personas

### Persona 1: Chief Compliance Officer -- Dr. Katrin Bauer (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Chief Compliance Officer at a large EU food and beverage conglomerate |
| **Company** | 12,000 employees, sourcing cocoa, coffee, palm oil, and soya from 25+ countries through 800+ suppliers |
| **EUDR Pressure** | Board mandate to achieve and demonstrate continuous compliance improvement; competent authority inspection scheduled for Q3 2026; must present evidence of systematic improvement since enforcement date |
| **Pain Points** | Drowning in findings from 34 EUDR agents with no unified view; spends 120+ hours per quarter manually aggregating and prioritizing; cannot demonstrate to the board or regulators that improvements are systematic and measurable; recurring issues across Indonesian palm oil supply chain keep appearing in different agent outputs but nobody has connected them to a root cause |
| **Goals** | Single dashboard showing all findings prioritized by risk; automated improvement plans that she can approve and delegate; year-over-year improvement metrics she can present to the board and competent authorities; root cause elimination that stops recurring findings |
| **Technical Skill** | Moderate -- comfortable with web applications, dashboards, and PDF reports; not a developer |

### Persona 2: Supply Chain Compliance Analyst -- Jonas Eriksson (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Supply Chain Compliance Analyst at an EU timber importer |
| **Company** | 2,000 employees, importing tropical and temperate wood from 15+ countries through 200+ supply chain nodes |
| **EUDR Pressure** | Must close 47 open compliance gaps identified across annual reviews and continuous monitoring; responsible for tracking improvement actions across 5 internal teams and 30+ suppliers |
| **Pain Points** | Improvement actions assigned by email get lost; no visibility into which actions are blocked, overdue, or completed; cannot measure whether completed actions actually improved compliance; spends 20 hours per week chasing progress updates from colleagues and suppliers |
| **Goals** | Clear action assignments with RACI matrix and automated reminders; progress dashboard showing real-time status of all actions; effectiveness measurement showing which improvements are working; dependency tracking so he knows which actions to unblock first |
| **Technical Skill** | High -- comfortable with data tools, APIs, dashboards, and basic scripting |

### Persona 3: Regional Procurement Director -- Sofia Vasquez (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Regional Procurement Director for Latin America at an EU rubber and tire manufacturer |
| **Company** | 5,000 employees, sourcing rubber from 150+ smallholders across Brazil, Guatemala, and Cote d'Ivoire |
| **EUDR Pressure** | Supplier capability gaps are the root cause behind 60% of compliance findings; must design and implement supplier improvement programs that demonstrably reduce risk scores |
| **Pain Points** | Receives improvement actions from compliance team but they are too generic ("improve supplier traceability") without SMART criteria or resource estimates; cannot estimate budget requirements for supplier improvement programs; no evidence that supplier training investments are actually reducing compliance risk |
| **Goals** | SMART actions with specific supplier improvement scope, measurable targets, and realistic timelines; resource estimates she can use for budget planning; effectiveness data showing return on supplier improvement investment |
| **Technical Skill** | Moderate -- uses ERP, procurement tools, and web applications |

### Persona 4: External Auditor -- Marc Lefebvre (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm specializing in EUDR compliance verification |
| **EUDR Pressure** | Must assess whether operators have adequate continuous improvement systems per Article 13 during competent authority inspections |
| **Pain Points** | Operators provide ad-hoc improvement documentation -- some in spreadsheets, some in email threads, some in project management tools; no standardized format for improvement plan evidence; cannot verify that improvement actions were actually implemented or effective; audit evidence is scattered across systems |
| **Goals** | Access to structured improvement plans with full RACI assignment and timeline; evidence of action completion with before-and-after metrics; root cause documentation showing systemic analysis; audit trail with SHA-256 provenance hashes for data integrity |
| **Technical Skill** | Moderate -- comfortable with audit software, document review, and compliance frameworks |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(2)** | Operators shall exercise due diligence with regard to all relevant commodities and products placed on or exported from the Union market | Finding Aggregator consolidates all due diligence findings; Gap Analysis Engine maps compliance against Article 4 completeness requirements |
| **Art. 8(1)** | Due diligence system must be reviewed and updated at least annually | Progress Tracker integrates with EUDR-034 annual review findings; Action Plan Generator creates improvement actions from annual review deltas |
| **Art. 9(1)(a-d)** | Information gathering requirements: product description, geolocation, quantities, supply chain data | Gap Analysis Engine evaluates Article 9 field completeness across all supply chain entities; maps information gaps to improvement actions |
| **Art. 10(1)** | Risk assessment must be adequate and proportionate | Gap Analysis Engine evaluates risk assessment comprehensiveness against Article 10(2) criteria (a) through (k); generates improvement actions for assessment gaps |
| **Art. 10(2)(a-k)** | Specific risk assessment criteria including supply chain complexity, deforestation prevalence, mixing risk, circumvention risk | Root Cause Mapper connects risk assessment findings to systemic causes; Prioritization Matrix weights improvement actions by Article 10(2) criterion severity |
| **Art. 11(1)** | Risk mitigation measures must be adequate and proportionate to reduce risk to negligible | Gap Analysis Engine evaluates mitigation adequacy by comparing pre/post risk scores from EUDR-029; generates corrective actions when mitigation is insufficient |
| **Art. 11(2)(a-c)** | Mitigation measure categories: additional information, independent audits, other adapted measures | Action Plan Generator maps improvement actions to Article 11(2) categories; ensures all three categories are represented in improvement plans |
| **Art. 12** | Due Diligence Statement content and submission requirements | Gap Analysis Engine validates DDS completeness against Article 12 requirements; generates corrective actions for incomplete or invalid DDS elements |
| **Art. 13** | Review and continuous improvement of due diligence system | Core regulatory driver for this agent. All seven engines collectively implement the continuous improvement cycle mandated by Article 13. Progress Tracker measures improvement over time. Audit trail documents improvement lifecycle for regulatory inspection |
| **Art. 14(1)** | Competent authorities shall carry out checks to verify operator compliance | Audit trail and documentation generation ensure all improvement activities are inspection-ready with full provenance |
| **Art. 15** | Risk-based checks by competent authorities | Prioritization Matrix aligns improvement priorities with enforcement risk, ensuring high-risk gaps are addressed first |
| **Art. 16** | Substantive checks verifying accuracy of due diligence documentation | Progress Tracker provides before-and-after evidence of improvement effectiveness; Root Cause Mapper provides systemic analysis documentation |
| **Art. 23-25** | Penalties: fines up to 4% of EU turnover, confiscation, market exclusion, public naming | Prioritization Matrix estimates penalty exposure per gap and prioritizes improvement actions by enforcement severity |
| **Art. 29** | Country benchmarking (Low/Standard/High risk) | Finding Aggregator ingests country benchmarking changes from EUDR-033; Gap Analysis Engine evaluates compliance against updated benchmarking requirements |
| **Art. 31** | Record keeping for at least 5 years | All improvement plans, progress records, effectiveness measurements, and audit trails retained for 5+ years with immutable provenance hashes |

### 5.2 CSDDD Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **CSDDD Art. 9** | Remediation of adverse impacts | Action Plan Generator creates remediation-specific improvement actions linked to identified adverse impacts; Progress Tracker measures remediation effectiveness |
| **CSDDD Art. 10** | Prevention of adverse impacts through prevention plans | Preventive Action Plans address anticipated non-conformances; Root Cause Mapper identifies systemic causes to prevent recurrence |

### 5.3 International Standards Alignment

| Standard | Requirement | Agent Implementation |
|----------|-------------|---------------------|
| **ISO 14001:2015** | Plan-Do-Check-Act (PDCA) continuous improvement cycle | All seven engines map to PDCA: Plan (Finding Aggregator + Gap Analysis + Action Plan Generator), Do (Progress Tracker + Stakeholder Assignment), Check (Effectiveness Measurement), Act (Root Cause Mapper + Prioritization Matrix feedback loop) |
| **ISO 31000:2018** | Risk treatment and continuous monitoring of risk management effectiveness | Action plans are risk-weighted treatments; effectiveness measurement verifies risk reduction; feedback loop ensures treatment adequacy |
| **OECD Guidelines Ch. IV** | Environmental due diligence with continuous improvement | Improvement plans address environmental due diligence gaps; progress documentation supports OECD reporting requirements |
| **UNGP Principle 17** | Human rights due diligence should cover adverse impacts the enterprise may cause, contribute to, or be directly linked to | Finding Aggregator includes indigenous rights findings from EUDR-021 and grievance intelligence from EUDR-032; improvement plans address human rights gaps |

### 5.4 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for all deforestation-free verification; gaps related to pre-cutoff verification are highest priority |
| June 29, 2023 | Regulation entered into force | Legal basis for all compliance checks and improvement requirements |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | First annual review cycle begins; improvement plans must address findings from first compliance year |
| June 30, 2026 | Enforcement for SMEs | SME onboarding generates surge of initial compliance gaps requiring systematic improvement planning |
| December 30, 2026 | First annual DDS renewal deadline for large operators | Improvement effectiveness must be demonstrable by first renewal; progress tracking critical |
| 2027 (expected) | CSDDD enforcement begins | Improvement plans must extend to CSDDD remediation and prevention requirements |
| Ongoing (quarterly) | Country benchmarking updates by EC | Finding Aggregator ingests benchmarking changes; improvement plans updated for reclassified countries |
| Ongoing (annual) | Article 8(1) DDS renewal cycles | Annual review findings from EUDR-034 feed into improvement planning cycle |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 7 features below are P0 launch blockers. The agent cannot ship without all 7 features operational. Features 1-4 form the core intelligence and analysis engine; Features 5-7 form the execution, tracking, and accountability layer.

**P0 Features 1-4: Core Intelligence and Analysis Engine**

---

#### Feature 1: Finding Aggregator

**User Story:**
```
As a Chief Compliance Officer,
I want all compliance findings from every EUDR agent consolidated into a single, prioritized view,
So that I can understand my complete compliance posture without manually aggregating outputs from 34 agents.
```

**Acceptance Criteria:**
- [ ] Ingests risk assessment results from EUDR-028: composite scores, five-dimensional decomposition (country, supplier, commodity, corruption, deforestation), Article 10(2) criteria evaluations, risk classification levels, risk trend data, and risk override records
- [ ] Ingests grievance intelligence from EUDR-032: pattern analytics results, root cause analysis outputs, remediation effectiveness scores, collective grievance impact assessments, and regulatory reports
- [ ] Ingests continuous monitoring events from EUDR-033: compliance alerts, deforestation alert correlations, risk degradation warnings, data freshness violations, change detection events, and regulatory update impacts
- [ ] Ingests annual review findings from EUDR-034: year-over-year comparison deltas, missed deadline incidents, review checklist failures, multi-year trend data, and entity review status
- [ ] Ingests mitigation data from EUDR-029: pre/post risk score comparisons, measure implementation completeness, effectiveness verification results, and residual risk determinations
- [ ] Ingests audit findings from EUDR-024: third-party audit non-conformances, corrective action requirements, certification gap assessments, and audit schedule deviations
- [ ] Normalizes all findings to a common schema: finding_id, source_agent, finding_type, severity (CRITICAL/HIGH/MEDIUM/LOW), EUDR article reference, affected_scope (operator_id, commodity, country, supplier_id, dds_id), temporal_context (detected_at, relevant_period), evidence_references (list of source document IDs with SHA-256 hashes), and description
- [ ] Deduplicates findings where multiple agents report the same underlying issue using configurable correlation rules based on affected scope overlap, temporal proximity, and semantic similarity
- [ ] Correlates related findings across agents: groups findings that share the same affected supply chain segment, the same root cause (when known from EUDR-032), or the same EUDR article reference
- [ ] Calculates aggregate finding statistics: total findings by severity, by source agent, by EUDR article, by commodity, by country, and by time period
- [ ] Supports incremental ingestion: processes new findings without re-processing historical findings
- [ ] Maintains finding lineage: every aggregated finding links back to its source agent output with provenance hash
- [ ] Generates unified findings register exportable as JSON, CSV, and PDF

**Non-Functional Requirements:**
- Throughput: Process 10,000+ findings per minute from all upstream agents
- Latency: < 500ms to aggregate findings for a single operator query
- Deduplication Accuracy: >= 95% precision (no false merges), >= 90% recall (no missed duplicates)
- Storage: Retain all findings for 5+ years per Article 31
- Reproducibility: Deterministic aggregation (same inputs produce same output)

**Dependencies:**
- EUDR-028 Risk Assessment Engine (risk scores and decompositions)
- EUDR-032 Grievance Mechanism Manager (grievance analytics and patterns)
- EUDR-033 Continuous Monitoring Agent (monitoring events and alerts)
- EUDR-034 Annual Review Scheduler (review findings and comparisons)
- EUDR-029 Mitigation Measure Designer (mitigation effectiveness data)
- EUDR-024 Third-Party Audit Manager (audit findings)

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 data engineer)

**Edge Cases:**
- Upstream agent unavailable during aggregation -- Use last known data, flag as stale, continue aggregation
- Conflicting severity classifications from different agents for same issue -- Apply highest severity (conservative approach)
- Finding from agent that has been deprecated or replaced -- Maintain historical findings, mark source as deprecated

---

#### Feature 2: Gap Analysis Engine

**User Story:**
```
As a compliance officer,
I want a systematic article-by-article assessment of my EUDR compliance posture,
So that I can identify exactly where I am falling short and prioritize remediation by regulatory severity.
```

**Acceptance Criteria:**
- [ ] Evaluates compliance against Article 4 (due diligence obligation): checks that DDS exists for all products, that due diligence system covers all three phases (information gathering, risk assessment, risk mitigation), and that the system is documented and auditable
- [ ] Evaluates compliance against Article 9 (information gathering): validates completeness of product descriptions, quantities, country of production, geolocation (GPS coordinates for all plots, polygons for plots > 4 ha), supplier identification, and date/time ranges for every active DDS
- [ ] Evaluates compliance against Article 10(2) criteria (a) through (k): validates that risk assessment addresses supply chain complexity (a), deforestation prevalence (b), forest degradation (c), commodity risk indicators (d), country concerns (e), mixing/unknown origin risk (f), circumvention risk (g), operator non-compliance history (h), scientific studies (i), indigenous peoples information (j), and international agreements (k)
- [ ] Evaluates compliance against Article 11 (risk mitigation): validates that mitigation measures are implemented for all HIGH and CRITICAL risk assessments, that measures are adequate (risk reduced to negligible/low), and that measures map to Article 11(2) categories
- [ ] Evaluates compliance against Article 12 (DDS content): validates all mandatory DDS fields, checks DDS validity period (< 12 months), and verifies submission status
- [ ] Evaluates compliance against Article 13 (continuous improvement): validates that review cycle is active, that improvement actions exist for identified gaps, and that effectiveness is being measured
- [ ] Evaluates compliance against Article 29 (country benchmarking): validates that country risk classifications are current and that supply chains from reclassified countries have been re-assessed
- [ ] Evaluates compliance against Article 31 (record retention): validates that all due diligence records are retained and accessible for 5+ years
- [ ] Calculates per-article compliance score (0-100) based on configurable weighting of sub-requirements within each article
- [ ] Classifies gap severity: CRITICAL (enforcement imminent, penalty likely), HIGH (significant non-compliance, enforcement possible), MEDIUM (partial compliance, improvement needed), LOW (minor gap, best-practice enhancement)
- [ ] Estimates enforcement exposure per gap: probability of detection by competent authority (based on inspection targeting criteria), potential penalty range (based on Article 23-25 penalty framework), and market access impact
- [ ] Generates remediation pathways: for each gap, specifies what must be done, which upstream agent provides the capability, estimated effort, and suggested timeline
- [ ] Produces compliance heat map: matrix of articles vs. commodities/countries with color-coded compliance scores
- [ ] Supports gap analysis at portfolio level (all commodities), commodity level, supply chain level, and individual entity level
- [ ] Tracks gap closure over time with trend reporting

**Non-Functional Requirements:**
- Latency: < 5 seconds for full article-by-article gap analysis per operator
- Coverage: 100% of EUDR articles evaluated per analysis run
- Accuracy: Gap detection validated against manual expert assessment (>= 95% agreement)
- Reproducibility: Deterministic gap scoring (same input produces same output)

**Dependencies:**
- Feature 1: Finding Aggregator (provides consolidated findings as input)
- EUDR-030 Documentation Generator (DDS content and submission status)
- EUDR-028 Risk Assessment Engine (risk assessment completeness and scores)
- EUDR-029 Mitigation Measure Designer (mitigation implementation and effectiveness)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- Operator has no DDS submissions yet (pre-compliance) -- Generate "initial compliance baseline" with 100% gaps, providing a starting improvement roadmap
- Article requirement is ambiguous or subject to regulatory interpretation -- Use conservative interpretation, flag for Compliance Officer review
- Country reclassified mid-analysis -- Recalculate gaps for affected supply chains using new classification

---

#### Feature 3: Action Plan Generator

**User Story:**
```
As a compliance officer,
I want automatically generated improvement actions that are specific, measurable, achievable, relevant, and time-bound,
So that I can immediately begin closing compliance gaps without spending weeks designing action plans manually.
```

**Acceptance Criteria:**
- [ ] Generates a unique improvement action for every identified finding and gap
- [ ] Each action includes: unique action_id, title (imperative form, < 120 characters), detailed description (100-500 words), linked finding IDs (from Finding Aggregator), linked gap IDs (from Gap Analysis Engine), linked EUDR article references
- [ ] Each action passes SMART validation:
  - Specific: clearly defined scope stating what will be done, for which supply chain entities, and to what standard
  - Measurable: quantifiable success metric with target value (e.g., "Reduce supplier risk score from 72 to < 30")
  - Achievable: prerequisite assessment confirming that required capabilities, data, and access exist
  - Relevant: explicit linkage to specific EUDR article requirement or identified root cause
  - Time-bound: deadline with intermediate milestones at 25%, 50%, 75%, and 100% completion
- [ ] Each action includes resource estimate: personnel hours (range: optimistic, expected, pessimistic), cost estimate in EUR (range), required skill set, and external vendor requirements
- [ ] Each action includes dependency chain: list of action IDs that must complete before this action can start, and list of action IDs that are blocked by this action
- [ ] Each action includes risk-if-not-implemented assessment: regulatory exposure (penalty probability and magnitude), operational impact, and reputational risk
- [ ] Each action includes evidence requirements: list of artifacts that must be produced or collected to verify completion (e.g., updated geolocation data, refreshed risk assessment, supplier certification copy)
- [ ] Supports three plan types: Corrective Action Plans (address existing non-conformances -- immediate remediation), Preventive Action Plans (prevent anticipated non-conformances -- proactive measures), and Continuous Improvement Plans (enhance compliance beyond minimum requirements -- strategic enhancements)
- [ ] Generates plans at four levels: portfolio level (all commodities for operator), commodity level (single commodity), supply chain level (single supply chain graph), and entity level (single supplier or single DDS)
- [ ] Consolidates related actions into improvement themes (e.g., "Indonesian Palm Oil Traceability Improvement" grouping 12 related actions)
- [ ] Generates action plan summary with total resource estimate, critical path timeline, and expected compliance score improvement

**Non-Functional Requirements:**
- Latency: < 10 seconds for portfolio-level plan covering 500+ findings
- Quality: 100% of generated actions pass automated SMART validation
- Reproducibility: Deterministic plan generation (same input produces same output)
- Configurability: Action templates configurable per operator, per commodity, and per country

**Dependencies:**
- Feature 1: Finding Aggregator (findings as input)
- Feature 2: Gap Analysis Engine (gaps as input)
- Action template library (curated library of 100+ action templates)

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 domain specialist)

**Edge Cases:**
- Finding has no known remediation pathway -- Generate "investigation action" to research remediation options, flag for expert review
- Action dependency creates circular chain -- Detect cycle, break by splitting one action into two sequential sub-actions
- Resource estimate exceeds operator capacity -- Flag as "requires resource expansion" and suggest phased implementation with critical-path-first sequencing

---

#### Feature 4: Root Cause Mapper

**User Story:**
```
As a supply chain compliance analyst,
I want to understand why the same types of compliance findings keep recurring,
So that I can address the systemic root cause instead of treating symptoms one at a time.
```

**Acceptance Criteria:**
- [ ] Implements 5-Whys decomposition: for each recurring finding or finding cluster, recursively asks "Why did this occur?" through up to five levels, requiring evidence linkage at each level (source agent output, data reference, or expert annotation)
- [ ] Implements Ishikawa (fishbone) diagram generation across six EUDR-adapted cause categories:
  - People: staff capacity gaps, training deficits, role ambiguity, accountability failures, knowledge attrition
  - Process: workflow gaps, quality gate failures, handoff errors between agents, manual process bottlenecks, escalation failures
  - Policy: regulatory interpretation errors, threshold miscalibration, scope exclusions, insufficient coverage, outdated policies
  - Place: geographic risk factors (high-deforestation regions, infrastructure limitations, connectivity gaps, jurisdictional complexity)
  - Product: commodity-specific traceability challenges (latex aggregation, batch mixing, animal movement, species mixing), transformation complexity, unit conversion errors
  - Partner: supplier governance failures, certification body limitations, auditor capacity gaps, competent authority delays, NGO relationship gaps
- [ ] Implements cross-finding correlation analysis: identifies common causal factors across findings from different agents, different geographies, different commodities, and different time periods using configurable correlation rules
- [ ] Detects recurring findings: identifies findings that appear in consecutive analysis cycles (monthly, quarterly, annual) affecting the same or similar supply chain scope, indicating that previous remediation did not address the root cause
- [ ] Generates root cause registers: each root cause includes unique root_cause_id, description, evidence chain (linked 5-Whys or Ishikawa analysis), affected finding IDs, confidence score (0-100 based on evidence strength), and recommended systemic intervention
- [ ] Links root causes to improvement actions: each systemic intervention recommendation in the root cause register maps to one or more actions in the Action Plan Generator
- [ ] Tracks root cause resolution over time: monitors whether the symptoms (recurring findings) cease after the root cause intervention is implemented, updating confidence score based on outcome evidence
- [ ] Generates root cause analysis reports with visual 5-Whys trees and Ishikawa diagrams exportable as JSON, SVG, and PDF
- [ ] Supports collaborative root cause analysis: multiple analysts can contribute evidence and annotations to a root cause investigation

**Non-Functional Requirements:**
- Latency: < 3 seconds for 5-Whys decomposition per finding chain
- Accuracy: >= 80% of identified root causes validated by expert review within 60 days
- Reproducibility: Deterministic correlation analysis (same input produces same output)
- Storage: Root cause register retained for 5+ years per Article 31

**Dependencies:**
- Feature 1: Finding Aggregator (findings as input, including historical findings for recurrence detection)
- EUDR-032 Grievance Mechanism Manager (provides its own root cause analysis outputs that can seed this engine)
- EUDR-034 Annual Review Scheduler (provides year-over-year comparison data for trend detection)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- 5-Whys decomposition reaches level 5 without converging to an actionable root cause -- Allow extension to level 7 with analyst approval; flag as "complex root cause requiring expert investigation"
- Root cause spans multiple operators (industry-wide issue) -- Document as "systemic industry root cause" and link to landscape-level intervention recommendations
- Insufficient evidence to confirm root cause -- Assign confidence score < 50, flag as "hypothesis requiring validation", generate evidence-gathering action

---

**P0 Features 5-7: Execution, Tracking, and Accountability Layer**

> Features 5, 6, and 7 are P0 launch blockers. Without prioritization, progress tracking, and stakeholder assignment, the intelligence generated by Features 1-4 cannot be translated into executed improvements. These features are the execution mechanism through which compliance officers, analysts, and procurement managers drive measurable compliance improvement.

---

#### Feature 5: Prioritization Matrix

**User Story:**
```
As a Chief Compliance Officer,
I want improvement actions ranked by objective criteria so I can allocate limited compliance resources to the highest-impact actions first,
So that I maximize compliance improvement per euro invested and address the most dangerous gaps before competent authority inspection.
```

**Acceptance Criteria:**
- [ ] Implements Eisenhower matrix classification across two dimensions:
  - Urgency: scored 0-100 based on deadline proximity (weight 0.40), enforcement timeline (weight 0.30), and penalty imminence (weight 0.30)
  - Importance: scored 0-100 based on risk severity (weight 0.35), compliance gap magnitude (weight 0.30), and stakeholder impact (weight 0.35)
- [ ] Classifies every action into one of four quadrants:
  - Q1 Do First (Urgency >= 60 AND Importance >= 60): immediate regulatory deadlines, critical compliance gaps, active enforcement risk
  - Q2 Schedule (Urgency < 60 AND Importance >= 60): strategic improvements, capacity building, system enhancements, long-term risk reduction
  - Q3 Delegate (Urgency >= 60 AND Importance < 60): routine compliance tasks, data refresh, minor corrections, administrative actions
  - Q4 Defer (Urgency < 60 AND Importance < 60): nice-to-have improvements, deferred features, low-impact enhancements
- [ ] Within each quadrant, applies risk-weighted priority scoring:
  ```
  Priority_Score = (Severity_Weight * 0.30) + (Enforcement_Exposure * 0.25) +
                   (Risk_Impact * 0.20) + (Resource_Efficiency * 0.15) +
                   (Dependency_Criticality * 0.10)

  Where:
  - Severity_Weight = normalized severity score (0-100) from Gap Analysis Engine
  - Enforcement_Exposure = estimated penalty probability * penalty magnitude / max_penalty
  - Risk_Impact = composite risk score delta if gap is not closed (from EUDR-028)
  - Resource_Efficiency = expected improvement per resource unit (EUR or person-hour)
  - Dependency_Criticality = number of downstream actions blocked by this action
  ```
- [ ] All scoring uses Decimal arithmetic to prevent floating-point drift
- [ ] Generates prioritized action backlog: ordered list of all actions by Priority_Score within each Eisenhower quadrant
- [ ] Generates resource allocation recommendations: optimal distribution of available budget and personnel across action quadrants and themes
- [ ] Generates critical path analysis: identifies the longest dependency chain and the actions that, if delayed, would delay the entire improvement program
- [ ] Supports configurable weights: operators can adjust the five scoring factors to reflect their specific risk appetite and resource constraints
- [ ] Supports constraint-based optimization: given a budget cap (EUR) and personnel cap (FTEs), recommends the subset of actions that maximizes total risk reduction within the constraints
- [ ] Recalculates priorities in real-time as actions are completed, new findings arrive, or external conditions change (country reclassification, new enforcement guidance)
- [ ] Generates priority change log: when an action's priority changes, records the reason (new finding, completed dependency, weight adjustment) with timestamp and actor

**Non-Functional Requirements:**
- Latency: < 2 seconds for prioritization of 1,000+ actions
- Determinism: Bit-perfect reproducibility across runs with same inputs and configuration
- Configurability: All weights and thresholds adjustable per operator without code changes

**Dependencies:**
- Feature 3: Action Plan Generator (actions as input)
- Feature 2: Gap Analysis Engine (severity and enforcement exposure scores)
- EUDR-028 Risk Assessment Engine (risk impact estimates)

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- All actions in Q1 (everything urgent and important) -- Apply within-quadrant ranking to determine execution order; recommend resource expansion if Q1 actions exceed capacity
- No budget constraint provided -- Prioritize purely by risk-weighted score without cost optimization
- Two actions have identical Priority_Score -- Tiebreak by: (1) higher dependency criticality, (2) lower resource requirement, (3) earlier deadline

---

#### Feature 6: Progress Tracker

**User Story:**
```
As a supply chain compliance analyst,
I want real-time visibility into the implementation status of every improvement action,
So that I can identify blocked and overdue actions, measure progress, and demonstrate improvement to auditors.
```

**Acceptance Criteria:**
- [ ] Tracks every improvement action through defined lifecycle states: PLANNED (action generated but not yet approved), APPROVED (approved for implementation by authorized role), IN_PROGRESS (implementation underway), BLOCKED (implementation halted due to external dependency or impediment), COMPLETED (all implementation steps finished, evidence collected), VERIFIED (effectiveness verified through before/after metric comparison), CLOSED (action formally closed with outcome documented)
- [ ] Records milestones at 25%, 50%, 75%, and 100% completion with timestamp, actor identification, and progress notes
- [ ] Tracks resource consumption: actual personnel hours versus planned, actual cost versus budget, with variance calculation and burn rate projection
- [ ] Measures implementation velocity: actions completed per week, per month, per quarter, with trend analysis
- [ ] Detects and alerts on: blocked actions (action in BLOCKED state for > configurable threshold, default 7 days), overdue actions (current date past action deadline), stalled progress (action in IN_PROGRESS state with no milestone update for > configurable threshold, default 14 days)
- [ ] Generates progress dashboards: overall plan completion percentage, action status distribution (pie chart by state), burndown chart (planned vs. actual completion curve), and velocity trend
- [ ] Implements effectiveness measurement: after an action reaches COMPLETED state, compares the relevant compliance metric before implementation (baseline captured at PLANNED state) against after implementation (measured at VERIFIED state). Calculates achievement percentage: (Actual_Improvement / Target_Improvement) * 100
- [ ] Supports effectiveness metrics per action type: for risk reduction actions, measures pre/post risk score delta; for gap closure actions, measures pre/post compliance score delta; for root cause actions, measures recurrence rate change; for process improvement actions, measures throughput/error rate change
- [ ] Feeds effectiveness data back to Action Plan Generator and Prioritization Matrix: action types that consistently achieve < 50% of target improvement are flagged for methodology review; action types that consistently exceed targets are recommended more frequently
- [ ] Generates audit-ready progress reports: timeline of all actions with status transitions, evidence references, and outcome measurements, exportable as JSON and PDF with SHA-256 provenance hashes
- [ ] Supports plan versioning: when the improvement plan is updated (new actions added, actions reprioritized, actions cancelled), creates a new plan version with change log

**Non-Functional Requirements:**
- Latency: < 200ms for status query per action; < 2 seconds for full dashboard data
- Alerting: Overdue and blocked action alerts delivered within 1 hour of threshold breach
- Storage: All progress data retained for 5+ years per Article 31
- Reproducibility: Effectiveness calculations deterministic with Decimal arithmetic

**Dependencies:**
- Features 1-5 (findings, gaps, actions, root causes, priorities as inputs)
- EUDR-028 Risk Assessment Engine (for risk score before/after comparison)
- EUDR-033 Continuous Monitoring Agent (for real-time compliance score tracking)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend engineer)

**Edge Cases:**
- Action completed but metric worsened (negative effectiveness) -- Record as "action ineffective", flag for root cause investigation, recommend corrective action
- Effectiveness metric not available for measurement (upstream agent data delayed) -- Place action in "pending verification" state, retry measurement after configurable delay (default 7 days)
- Action cancelled mid-implementation -- Record final state as CANCELLED with cancellation reason, preserve all progress data for audit trail

---

#### Feature 7: Stakeholder Assignment Engine

**User Story:**
```
As a compliance officer,
I want every improvement action clearly assigned to a responsible party with defined accountability,
So that actions get done on time and I can track who is responsible for what.
```

**Acceptance Criteria:**
- [ ] Assigns every action using a RACI matrix:
  - Responsible (R): The person or team doing the work. Exactly one R per action
  - Accountable (A): The person who approves the work and is ultimately answerable. Exactly one A per action
  - Consulted (C): People whose input is needed before or during the work. Zero or more C per action
  - Informed (I): People who need to be notified of progress and outcomes. Zero or more I per action
- [ ] Implements role-based default assignment rules:
  - Compliance Officer: A (Accountable) for regulatory gap closure actions, R (Responsible) for compliance documentation actions
  - Supply Chain Analyst: R for data quality improvement actions, R for supply chain re-mapping actions
  - Procurement Manager: R for supplier engagement actions, R for supplier capacity building actions
  - Risk Analyst: R for risk assessment refresh actions, R for risk re-evaluation actions
  - IT/Data Engineer: R for system integration actions, R for data migration actions
  - External Auditor: C for verification actions, I for all compliance actions
  - Management/Director: A for resource allocation decisions, I for all plan progress
- [ ] Supports manual override of default assignments by authorized roles (Compliance Officer, Admin)
- [ ] Implements workload balancing: monitors concurrent active actions per stakeholder (configurable maximum, default 15), alerts when a stakeholder approaches or exceeds maximum, recommends rebalancing when workload distribution is skewed (Gini coefficient > 0.6)
- [ ] Sends automated notifications via configurable channels (email, webhook, in-app):
  - Assignment notification: when an action is assigned to a stakeholder
  - Deadline approaching: 7 days and 1 day before action deadline
  - Overdue notification: when action deadline passes without completion
  - Blocked notification: when an action's dependency is blocked or overdue
  - Escalation notification: when an action is overdue by > configurable threshold (default 14 days), escalate to A (Accountable) role
  - Completion notification: when a dependent action completes, unblocking this action
- [ ] Generates accountability reports:
  - Assignment distribution: actions per stakeholder by status (planned, in progress, completed, overdue)
  - Completion rates per stakeholder: % of assigned actions completed on time
  - Overdue rates per stakeholder: % of assigned actions that are or were overdue
  - Workload forecast: projected actions per stakeholder for next 30/60/90 days
- [ ] Supports team assignment: actions can be assigned to a team with team lead as R and team members as contributors
- [ ] Supports delegation: R can delegate an action to a sub-assignee while retaining tracking visibility
- [ ] Maintains full audit trail: every assignment, reassignment, delegation, and escalation recorded with timestamp, actor, and reason

**Non-Functional Requirements:**
- Notification latency: < 5 minutes from trigger event to notification delivery
- Scalability: Support 10,000+ concurrent action assignments across 500+ operators
- Multi-channel: Support email, webhook (Slack/Teams integration), and in-app notifications
- Configurability: Notification templates, escalation thresholds, and default assignments configurable per operator

**Dependencies:**
- Feature 3: Action Plan Generator (actions to assign)
- Feature 6: Progress Tracker (status data for notifications and reporting)
- SEC-001 JWT Authentication (stakeholder identity)
- SEC-002 RBAC Authorization (role-based assignment rules)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 integration engineer)

**Edge Cases:**
- Stakeholder leaves the organization -- Detect unassigned actions, alert Admin, recommend reassignment
- No stakeholder with required skill set available -- Escalate to Management with recommendation to hire or contract
- Stakeholder rejects assignment -- Record rejection with reason, escalate to A (Accountable) for reassignment

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 8: Benchmarking Engine
- Compare operator's compliance posture against anonymized industry benchmarks
- Identify areas where operator is ahead of or behind industry average
- Generate competitive positioning reports for board and investor presentations
- Support sector-specific benchmarks (food and beverage, timber, automotive)

#### Feature 9: AI-Powered Improvement Recommendations
- Machine learning model trained on historical improvement outcomes to predict which actions will be most effective for a given finding profile
- Predict improvement timeline and resource requirements based on similar past implementations
- Recommend optimal action sequencing based on historical dependency resolution patterns
- All recommendations explainable (SHAP values) and auditable with deterministic fallback

#### Feature 10: Improvement Plan Collaboration Portal
- Multi-stakeholder collaboration workspace for improvement plan review and refinement
- Discussion threads linked to specific actions and findings
- Document sharing and co-editing for improvement evidence
- External stakeholder (supplier, auditor, certifier) portal access with role-based visibility

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Financial accounting integration for improvement budget tracking against general ledger (defer to GL-Finance integration)
- Carbon footprint improvement planning (defer to GL-GHG-APP integration)
- Real-time video conferencing for improvement review meetings (use existing tools: Teams, Zoom)
- Mobile native application (web responsive design only for v1.0)
- Automated supplier payment for improvement-related invoices
- Integration with third-party project management tools (Jira, Asana, Monday.com) -- defer to Phase 2 via webhook API

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
| AGENT-EUDR-035        |           | AGENT-EUDR-028            |           | AGENT-EUDR-033        |
| Improvement Plan      |<--------->| Risk Assessment           |<--------->| Continuous Monitoring  |
| Creator               |           | Engine                    |           | Agent                 |
|                       |           |                           |           |                       |
| - Finding Aggregator  |           | - Composite Risk Calc     |           | - Supply Chain Watch   |
| - Gap Analysis Engine |           | - Risk Factor Aggregation |           | - Deforestation Corr.  |
| - Action Plan Gen.    |           | - Art 10(2) Evaluation    |           | - Compliance Scanner   |
| - Root Cause Mapper   |           | - Risk Classification    |           | - Change Detection     |
| - Prioritization      |           | - Risk Trend Analyzer     |           | - Risk Degradation     |
| - Progress Tracker    |           | - Risk Report Gen.        |           | - Data Freshness       |
| - Stakeholder Assign. |           |                           |           | - Regulatory Tracker   |
+-----------+-----------+           +---------------------------+           +-----------------------+
            |
            +---+---+---+---+
            |   |   |   |   |
+-----------v-+ | +-v-+ | +-v-----------+
| EUDR-032    | | |029| | | EUDR-034    |
| Grievance   | | |MMD| | | Annual Rev. |
| Mechanism   | | +---+ | | Scheduler   |
| Manager     | |       | |             |
+-------------+ |       | +-------------+
                |       |
          +-----v-+   +-v-----------+
          |EUDR-024|   | EUDR-029    |
          |Audit   |   | Mitigation  |
          |Manager |   | Measure Des.|
          +--------+   +-------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/improvement_plan_creator/
    __init__.py                          # Public API exports
    config.py                            # ImprovementPlanConfig with GL_EUDR_IPC_ env prefix
    models.py                            # Pydantic v2 models: findings, gaps, actions, root causes, RACI
    finding_aggregator.py                # FindingAggregator: cross-agent finding consolidation
    gap_analysis_engine.py               # GapAnalysisEngine: article-by-article compliance mapping
    action_plan_generator.py             # ActionPlanGenerator: SMART action creation
    root_cause_mapper.py                 # RootCauseMapper: 5-Whys, Ishikawa, correlation
    prioritization_matrix.py             # PrioritizationMatrix: Eisenhower + risk-weighted scoring
    progress_tracker.py                  # ProgressTracker: lifecycle, effectiveness, dashboards
    stakeholder_assignment.py            # StakeholderAssignment: RACI, workload, notifications
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains for audit trail
    metrics.py                           # 15 Prometheus self-monitoring metrics
    setup.py                             # ImprovementPlanService facade (singleton, thread-safe)
    api/
        __init__.py
        router.py                        # FastAPI router (30+ endpoints)
        finding_routes.py                # Finding aggregation and query endpoints
        gap_routes.py                    # Gap analysis and compliance heat map endpoints
        action_routes.py                 # Action plan CRUD, SMART validation, plan generation
        root_cause_routes.py             # Root cause analysis, 5-Whys, Ishikawa endpoints
        priority_routes.py               # Prioritization, constraint optimization endpoints
        progress_routes.py               # Progress tracking, effectiveness, dashboard endpoints
        stakeholder_routes.py            # RACI assignment, workload, notification endpoints
        report_routes.py                 # Audit-ready report generation endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Finding severity classification
class FindingSeverity(str, Enum):
    CRITICAL = "critical"     # Enforcement imminent, penalty likely
    HIGH = "high"             # Significant non-compliance, enforcement possible
    MEDIUM = "medium"         # Partial compliance, improvement needed
    LOW = "low"               # Minor gap, best-practice enhancement

# Finding source agents
class FindingSource(str, Enum):
    RISK_ASSESSMENT = "eudr_028"        # Risk Assessment Engine
    MITIGATION_DESIGNER = "eudr_029"    # Mitigation Measure Designer
    GRIEVANCE_MANAGER = "eudr_032"      # Grievance Mechanism Manager
    CONTINUOUS_MONITORING = "eudr_033"  # Continuous Monitoring Agent
    ANNUAL_REVIEW = "eudr_034"          # Annual Review Scheduler
    AUDIT_MANAGER = "eudr_024"          # Third-Party Audit Manager

# Gap severity classification
class GapSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Action lifecycle states
class ActionStatus(str, Enum):
    PLANNED = "planned"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    VERIFIED = "verified"
    CLOSED = "closed"
    CANCELLED = "cancelled"

# Action plan types
class PlanType(str, Enum):
    CORRECTIVE = "corrective"          # Address existing non-conformances
    PREVENTIVE = "preventive"          # Prevent anticipated non-conformances
    CONTINUOUS_IMPROVEMENT = "continuous_improvement"  # Enhance beyond minimum

# Eisenhower quadrants
class EisenhowerQuadrant(str, Enum):
    DO_FIRST = "q1_do_first"           # Urgent AND Important
    SCHEDULE = "q2_schedule"           # Important but NOT Urgent
    DELEGATE = "q3_delegate"           # Urgent but NOT Important
    DEFER = "q4_defer"                 # Neither Urgent NOR Important

# RACI role types
class RACIRole(str, Enum):
    RESPONSIBLE = "responsible"
    ACCOUNTABLE = "accountable"
    CONSULTED = "consulted"
    INFORMED = "informed"

# Root cause analysis method
class RootCauseMethod(str, Enum):
    FIVE_WHYS = "five_whys"
    ISHIKAWA = "ishikawa"
    CROSS_CORRELATION = "cross_correlation"

# Ishikawa cause categories (EUDR-adapted)
class IshikawaCauseCategory(str, Enum):
    PEOPLE = "people"
    PROCESS = "process"
    POLICY = "policy"
    PLACE = "place"
    PRODUCT = "product"
    PARTNER = "partner"

# Normalized finding from any upstream agent
class NormalizedFinding(BaseModel):
    finding_id: str                    # Unique identifier
    source_agent: FindingSource        # Which agent produced this finding
    source_finding_id: str             # Original finding ID in source agent
    finding_type: str                  # Agent-specific finding type
    severity: FindingSeverity
    eudr_articles: List[str]           # Relevant EUDR article references
    operator_id: str
    affected_scope: Dict[str, Any]     # commodity, country, supplier_id, dds_id
    temporal_context: Dict[str, Any]   # detected_at, relevant_period
    description: str
    evidence_references: List[Dict[str, str]]  # doc_id, hash pairs
    is_duplicate: bool                 # True if deduplicated
    duplicate_of: Optional[str]        # Finding ID this is a duplicate of
    correlation_group: Optional[str]   # Correlation group ID for related findings
    provenance_hash: str               # SHA-256
    ingested_at: datetime

# Compliance gap per EUDR article
class ComplianceGap(BaseModel):
    gap_id: str
    operator_id: str
    eudr_article: str                  # e.g., "Article 9(1)(d)"
    article_description: str
    compliance_score: Decimal          # 0-100
    gap_severity: GapSeverity
    gap_description: str
    affected_scope: Dict[str, Any]     # commodity, country, supply_chain_id
    enforcement_exposure: Dict[str, Any]  # probability, penalty_range, market_impact
    remediation_pathway: str
    related_finding_ids: List[str]
    detected_at: datetime
    resolved_at: Optional[datetime]
    provenance_hash: str

# SMART improvement action
class ImprovementAction(BaseModel):
    action_id: str
    plan_id: str                       # Parent improvement plan
    title: str                         # Imperative form, < 120 chars
    description: str                   # 100-500 words
    plan_type: PlanType
    linked_finding_ids: List[str]
    linked_gap_ids: List[str]
    eudr_article_refs: List[str]
    smart_criteria: Dict[str, Any]     # specific, measurable, achievable, relevant, time_bound
    resource_estimate: Dict[str, Any]  # hours_range, cost_range_eur, skill_set, vendor_reqs
    dependencies: List[str]            # Action IDs that must complete before this
    blocked_by: List[str]              # Action IDs blocking this
    risk_if_not_implemented: Dict[str, Any]  # regulatory, operational, reputational
    evidence_requirements: List[str]
    eisenhower_quadrant: EisenhowerQuadrant
    priority_score: Decimal
    status: ActionStatus
    raci: Dict[RACIRole, List[str]]    # role -> list of stakeholder IDs
    milestones: List[Dict[str, Any]]   # 25%, 50%, 75%, 100% with timestamps
    baseline_metric: Optional[Decimal] # Metric value before implementation
    target_metric: Optional[Decimal]   # Target metric value after implementation
    actual_metric: Optional[Decimal]   # Actual metric value after verification
    effectiveness_pct: Optional[Decimal]  # (actual - baseline) / (target - baseline) * 100
    created_at: datetime
    updated_at: datetime
    deadline: datetime
    completed_at: Optional[datetime]
    verified_at: Optional[datetime]
    closed_at: Optional[datetime]
    provenance_hash: str

# Root cause record
class RootCause(BaseModel):
    root_cause_id: str
    method: RootCauseMethod
    description: str
    evidence_chain: List[Dict[str, Any]]  # Level-by-level evidence for 5-Whys or category evidence for Ishikawa
    affected_finding_ids: List[str]
    confidence_score: Decimal          # 0-100
    recommended_intervention: str
    linked_action_ids: List[str]
    recurrence_count: int              # Times the root cause has been observed
    resolution_status: str             # open, in_progress, resolved, monitoring
    detected_at: datetime
    resolved_at: Optional[datetime]
    provenance_hash: str

# Improvement plan (collection of actions)
class ImprovementPlan(BaseModel):
    plan_id: str
    operator_id: str
    plan_type: PlanType
    plan_level: str                    # portfolio, commodity, supply_chain, entity
    scope_filter: Dict[str, Any]       # commodity, country, supply_chain_id
    total_actions: int
    actions_by_status: Dict[str, int]
    actions_by_quadrant: Dict[str, int]
    total_resource_hours: Decimal
    total_resource_cost_eur: Decimal
    critical_path_days: int
    expected_compliance_improvement: Decimal  # Projected score increase
    version: int
    created_at: datetime
    updated_at: datetime
    provenance_hash: str
```

### 7.4 Database Schema (New Migration: V119)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_improvement_plan;

-- Unified findings register (normalized from all upstream agents)
CREATE TABLE eudr_improvement_plan.findings (
    finding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_agent VARCHAR(20) NOT NULL,
    source_finding_id VARCHAR(100) NOT NULL,
    finding_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    eudr_articles JSONB DEFAULT '[]',
    operator_id UUID NOT NULL,
    affected_scope JSONB DEFAULT '{}',
    temporal_context JSONB DEFAULT '{}',
    description TEXT NOT NULL,
    evidence_references JSONB DEFAULT '[]',
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of UUID,
    correlation_group UUID,
    provenance_hash VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_improvement_plan.findings', 'ingested_at');

-- Compliance gaps per EUDR article
CREATE TABLE eudr_improvement_plan.compliance_gaps (
    gap_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    eudr_article VARCHAR(30) NOT NULL,
    article_description TEXT NOT NULL,
    compliance_score NUMERIC(5,2) DEFAULT 0.0,
    gap_severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    gap_description TEXT NOT NULL,
    affected_scope JSONB DEFAULT '{}',
    enforcement_exposure JSONB DEFAULT '{}',
    remediation_pathway TEXT,
    related_finding_ids JSONB DEFAULT '[]',
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_improvement_plan.compliance_gaps', 'detected_at');

-- Improvement plans (container for actions)
CREATE TABLE eudr_improvement_plan.plans (
    plan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    plan_type VARCHAR(30) NOT NULL DEFAULT 'corrective',
    plan_level VARCHAR(30) NOT NULL DEFAULT 'portfolio',
    scope_filter JSONB DEFAULT '{}',
    plan_name VARCHAR(500),
    total_actions INTEGER DEFAULT 0,
    actions_by_status JSONB DEFAULT '{}',
    actions_by_quadrant JSONB DEFAULT '{}',
    total_resource_hours NUMERIC(12,2) DEFAULT 0.0,
    total_resource_cost_eur NUMERIC(14,2) DEFAULT 0.0,
    critical_path_days INTEGER DEFAULT 0,
    expected_compliance_improvement NUMERIC(5,2) DEFAULT 0.0,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Improvement actions (SMART actions within plans)
CREATE TABLE eudr_improvement_plan.actions (
    action_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id UUID NOT NULL REFERENCES eudr_improvement_plan.plans(plan_id),
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    plan_type VARCHAR(30) NOT NULL DEFAULT 'corrective',
    linked_finding_ids JSONB DEFAULT '[]',
    linked_gap_ids JSONB DEFAULT '[]',
    eudr_article_refs JSONB DEFAULT '[]',
    smart_criteria JSONB NOT NULL DEFAULT '{}',
    resource_estimate JSONB DEFAULT '{}',
    dependencies JSONB DEFAULT '[]',
    blocked_by JSONB DEFAULT '[]',
    risk_if_not_implemented JSONB DEFAULT '{}',
    evidence_requirements JSONB DEFAULT '[]',
    eisenhower_quadrant VARCHAR(20) NOT NULL DEFAULT 'q2_schedule',
    priority_score NUMERIC(8,4) DEFAULT 0.0,
    status VARCHAR(20) NOT NULL DEFAULT 'planned',
    raci JSONB DEFAULT '{}',
    milestones JSONB DEFAULT '[]',
    baseline_metric NUMERIC(10,4),
    target_metric NUMERIC(10,4),
    actual_metric NUMERIC(10,4),
    effectiveness_pct NUMERIC(7,2),
    deadline TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    verified_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    provenance_hash VARCHAR(64) NOT NULL
);

-- Root cause analysis records
CREATE TABLE eudr_improvement_plan.root_causes (
    root_cause_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    method VARCHAR(30) NOT NULL,
    description TEXT NOT NULL,
    evidence_chain JSONB NOT NULL DEFAULT '[]',
    affected_finding_ids JSONB DEFAULT '[]',
    confidence_score NUMERIC(5,2) DEFAULT 0.0,
    recommended_intervention TEXT,
    linked_action_ids JSONB DEFAULT '[]',
    recurrence_count INTEGER DEFAULT 1,
    resolution_status VARCHAR(20) DEFAULT 'open',
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

-- RACI assignments
CREATE TABLE eudr_improvement_plan.raci_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action_id UUID NOT NULL REFERENCES eudr_improvement_plan.actions(action_id),
    stakeholder_id VARCHAR(100) NOT NULL,
    stakeholder_name VARCHAR(500),
    stakeholder_role VARCHAR(100),
    raci_role VARCHAR(20) NOT NULL,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    assigned_by VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    deactivated_at TIMESTAMPTZ,
    deactivation_reason TEXT
);

-- Action status transitions (audit log)
CREATE TABLE eudr_improvement_plan.action_transitions (
    transition_id UUID DEFAULT gen_random_uuid(),
    action_id UUID NOT NULL,
    previous_status VARCHAR(20),
    new_status VARCHAR(20) NOT NULL,
    transitioned_by VARCHAR(100) NOT NULL,
    notes TEXT,
    evidence_refs JSONB DEFAULT '[]',
    transitioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_improvement_plan.action_transitions', 'transitioned_at');

-- Notification log
CREATE TABLE eudr_improvement_plan.notifications (
    notification_id UUID DEFAULT gen_random_uuid(),
    action_id UUID,
    recipient_id VARCHAR(100) NOT NULL,
    notification_type VARCHAR(50) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    subject VARCHAR(500),
    body TEXT,
    delivery_status VARCHAR(20) DEFAULT 'pending',
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,
    read_at TIMESTAMPTZ
);

SELECT create_hypertable('eudr_improvement_plan.notifications', 'sent_at');

-- Plan version history (audit trail)
CREATE TABLE eudr_improvement_plan.plan_versions (
    version_id UUID DEFAULT gen_random_uuid(),
    plan_id UUID NOT NULL,
    version INTEGER NOT NULL,
    change_description TEXT,
    snapshot_data JSONB NOT NULL,
    provenance_hash VARCHAR(64) NOT NULL,
    created_by VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_improvement_plan.plan_versions', 'created_at');

-- Indexes
CREATE INDEX idx_findings_operator ON eudr_improvement_plan.findings(operator_id);
CREATE INDEX idx_findings_severity ON eudr_improvement_plan.findings(severity);
CREATE INDEX idx_findings_source ON eudr_improvement_plan.findings(source_agent);
CREATE INDEX idx_findings_correlation ON eudr_improvement_plan.findings(correlation_group);
CREATE INDEX idx_gaps_operator ON eudr_improvement_plan.compliance_gaps(operator_id);
CREATE INDEX idx_gaps_article ON eudr_improvement_plan.compliance_gaps(eudr_article);
CREATE INDEX idx_gaps_severity ON eudr_improvement_plan.compliance_gaps(gap_severity);
CREATE INDEX idx_actions_plan ON eudr_improvement_plan.actions(plan_id);
CREATE INDEX idx_actions_status ON eudr_improvement_plan.actions(status);
CREATE INDEX idx_actions_quadrant ON eudr_improvement_plan.actions(eisenhower_quadrant);
CREATE INDEX idx_actions_priority ON eudr_improvement_plan.actions(priority_score DESC);
CREATE INDEX idx_actions_deadline ON eudr_improvement_plan.actions(deadline);
CREATE INDEX idx_raci_action ON eudr_improvement_plan.raci_assignments(action_id);
CREATE INDEX idx_raci_stakeholder ON eudr_improvement_plan.raci_assignments(stakeholder_id);
CREATE INDEX idx_transitions_action ON eudr_improvement_plan.action_transitions(action_id);
CREATE INDEX idx_root_causes_operator ON eudr_improvement_plan.root_causes(operator_id);
CREATE INDEX idx_root_causes_status ON eudr_improvement_plan.root_causes(resolution_status);
CREATE INDEX idx_notifications_recipient ON eudr_improvement_plan.notifications(recipient_id);
CREATE INDEX idx_plan_versions_plan ON eudr_improvement_plan.plan_versions(plan_id);
```

### 7.5 API Endpoints (30+)

| Method | Path | Description |
|--------|------|-------------|
| **Finding Aggregation** | | |
| POST | `/v1/findings/aggregate` | Trigger finding aggregation from all upstream agents |
| GET | `/v1/findings` | List findings with filters (severity, source, article, commodity, country) |
| GET | `/v1/findings/{finding_id}` | Get finding details with evidence references |
| GET | `/v1/findings/statistics` | Get aggregate finding statistics (by severity, source, article) |
| GET | `/v1/findings/correlations` | List finding correlation groups |
| **Gap Analysis** | | |
| POST | `/v1/gaps/analyze` | Trigger article-by-article gap analysis for an operator |
| GET | `/v1/gaps` | List compliance gaps with filters (article, severity, commodity) |
| GET | `/v1/gaps/{gap_id}` | Get gap details with remediation pathway |
| GET | `/v1/gaps/heatmap` | Get compliance heat map data (articles x commodities/countries) |
| GET | `/v1/gaps/trends` | Get gap closure trends over time |
| PUT | `/v1/gaps/{gap_id}/resolve` | Mark gap as resolved with evidence |
| **Action Plans** | | |
| POST | `/v1/plans/generate` | Generate improvement plan (portfolio, commodity, supply chain, or entity level) |
| GET | `/v1/plans` | List improvement plans with filters (type, level, status) |
| GET | `/v1/plans/{plan_id}` | Get plan details with action summary |
| GET | `/v1/plans/{plan_id}/actions` | List actions within a plan with filters (status, quadrant, priority) |
| POST | `/v1/plans/{plan_id}/actions` | Add manual action to plan |
| GET | `/v1/plans/{plan_id}/actions/{action_id}` | Get action details with SMART criteria and RACI |
| PUT | `/v1/plans/{plan_id}/actions/{action_id}` | Update action (status, milestones, metrics) |
| DELETE | `/v1/plans/{plan_id}/actions/{action_id}` | Cancel an action with reason |
| GET | `/v1/plans/{plan_id}/critical-path` | Get critical path analysis |
| GET | `/v1/plans/{plan_id}/resource-summary` | Get resource allocation summary |
| **Root Cause Analysis** | | |
| POST | `/v1/root-causes/analyze` | Trigger root cause analysis for a finding or finding cluster |
| GET | `/v1/root-causes` | List root causes with filters (method, status, confidence) |
| GET | `/v1/root-causes/{root_cause_id}` | Get root cause with evidence chain and linked actions |
| GET | `/v1/root-causes/{root_cause_id}/ishikawa` | Get Ishikawa diagram data |
| GET | `/v1/root-causes/{root_cause_id}/five-whys` | Get 5-Whys decomposition tree |
| PUT | `/v1/root-causes/{root_cause_id}` | Update root cause (add evidence, update status) |
| **Prioritization** | | |
| POST | `/v1/prioritize` | Run prioritization matrix on plan actions |
| GET | `/v1/prioritize/matrix` | Get Eisenhower matrix distribution |
| POST | `/v1/prioritize/optimize` | Constraint-based optimization (given budget/FTE caps) |
| **Progress Tracking** | | |
| GET | `/v1/progress/dashboard` | Get progress dashboard data (completion %, burndown, velocity) |
| GET | `/v1/progress/effectiveness` | Get effectiveness measurement data (pre/post metrics) |
| GET | `/v1/progress/overdue` | List overdue and blocked actions |
| **Stakeholder Assignment** | | |
| POST | `/v1/assignments/assign` | Assign RACI roles to action |
| GET | `/v1/assignments/workload` | Get workload distribution per stakeholder |
| GET | `/v1/assignments/accountability` | Get accountability report per stakeholder |
| **Reports** | | |
| POST | `/v1/reports/improvement` | Generate audit-ready improvement report (JSON or PDF) |
| POST | `/v1/reports/article-13` | Generate Article 13 continuous improvement documentation |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (15)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_ipc_findings_ingested_total` | Counter | Findings ingested by source agent and severity |
| 2 | `gl_eudr_ipc_findings_deduplicated_total` | Counter | Findings deduplicated (duplicate detections) |
| 3 | `gl_eudr_ipc_gaps_detected_total` | Counter | Compliance gaps detected by article and severity |
| 4 | `gl_eudr_ipc_gaps_resolved_total` | Counter | Compliance gaps resolved |
| 5 | `gl_eudr_ipc_actions_created_total` | Counter | Improvement actions created by plan type and quadrant |
| 6 | `gl_eudr_ipc_actions_completed_total` | Counter | Improvement actions completed |
| 7 | `gl_eudr_ipc_actions_overdue_total` | Counter | Improvement actions that became overdue |
| 8 | `gl_eudr_ipc_root_causes_identified_total` | Counter | Root causes identified by method |
| 9 | `gl_eudr_ipc_root_causes_resolved_total` | Counter | Root causes resolved |
| 10 | `gl_eudr_ipc_processing_duration_seconds` | Histogram | Processing latency by operation type (aggregate, gap_analyze, plan_generate, prioritize, root_cause) |
| 11 | `gl_eudr_ipc_api_request_duration_seconds` | Histogram | API request latency by endpoint |
| 12 | `gl_eudr_ipc_errors_total` | Counter | Errors by operation type |
| 13 | `gl_eudr_ipc_active_plans` | Gauge | Currently active improvement plans |
| 14 | `gl_eudr_ipc_effectiveness_avg` | Gauge | Average effectiveness percentage across completed actions |
| 15 | `gl_eudr_ipc_notifications_sent_total` | Counter | Notifications sent by type and channel |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for findings, transitions, notifications |
| Cache | Redis | Finding aggregation cache, priority score cache, dashboard data cache |
| Object Storage | S3 | Report exports (PDF, JSON), plan snapshots, evidence attachments |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Arithmetic | Python Decimal | Prevents floating-point drift in priority scoring and effectiveness calculations |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based action assignment and plan access control |
| Monitoring | Prometheus + Grafana | 15 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| Notifications | Email (SMTP/SES), Webhooks, In-App | Multi-channel notification delivery |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-ipc:findings:read` | View aggregated findings | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ipc:findings:aggregate` | Trigger finding aggregation | Analyst, Compliance Officer, Admin |
| `eudr-ipc:gaps:read` | View compliance gaps and heat maps | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ipc:gaps:analyze` | Trigger gap analysis | Analyst, Compliance Officer, Admin |
| `eudr-ipc:gaps:resolve` | Mark gaps as resolved | Compliance Officer, Admin |
| `eudr-ipc:plans:read` | View improvement plans and actions | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ipc:plans:write` | Create, modify improvement plans | Compliance Officer, Admin |
| `eudr-ipc:plans:generate` | Trigger automated plan generation | Compliance Officer, Admin |
| `eudr-ipc:actions:read` | View improvement actions | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ipc:actions:write` | Create, update, cancel actions | Analyst, Compliance Officer, Admin |
| `eudr-ipc:actions:approve` | Approve actions for implementation | Compliance Officer, Admin |
| `eudr-ipc:actions:verify` | Verify action effectiveness | Compliance Officer, Admin |
| `eudr-ipc:root-causes:read` | View root cause analyses | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ipc:root-causes:write` | Create, update root cause analyses | Analyst, Compliance Officer, Admin |
| `eudr-ipc:priority:read` | View prioritization matrix | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ipc:priority:configure` | Adjust prioritization weights | Compliance Officer, Admin |
| `eudr-ipc:assignments:read` | View RACI assignments and workload | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ipc:assignments:write` | Assign and reassign stakeholders | Compliance Officer, Admin |
| `eudr-ipc:progress:read` | View progress dashboards and effectiveness | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ipc:reports:generate` | Generate audit-ready reports | Compliance Officer, Admin |
| `eudr-ipc:reports:read` | View generated reports | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-ipc:audit:read` | View full audit trail (transitions, notifications, versions) | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| EUDR-028 Risk Assessment Engine | Risk scores, decompositions, Article 10(2) evaluations, trends | Risk findings -> Finding Aggregator; risk scores -> Gap Analysis Engine and Effectiveness Measurement |
| EUDR-029 Mitigation Measure Designer | Mitigation strategies, implementation status, effectiveness verification | Mitigation findings -> Finding Aggregator; pre/post scores -> Effectiveness Measurement |
| EUDR-032 Grievance Mechanism Manager | Pattern analytics, root causes, remediation effectiveness, collective impacts | Grievance findings -> Finding Aggregator; root causes -> Root Cause Mapper seeding |
| EUDR-033 Continuous Monitoring Agent | Compliance alerts, deforestation correlations, risk degradation, data freshness, regulatory updates | Monitoring events -> Finding Aggregator; compliance scores -> Gap Analysis Engine |
| EUDR-034 Annual Review Scheduler | Year-over-year comparisons, missed deadlines, checklist failures, trends | Review findings -> Finding Aggregator; comparison data -> Root Cause trend analysis |
| EUDR-024 Third-Party Audit Manager | Audit non-conformances, corrective actions, certification gaps | Audit findings -> Finding Aggregator |
| EUDR-030 Documentation Generator | DDS content and submission status | DDS completeness data -> Gap Analysis Engine (Article 12 evaluation) |
| SEC-001 JWT Authentication | Stakeholder identity | User identity for RACI assignment and audit trail |
| SEC-002 RBAC Authorization | Role-based access control | Permission enforcement for all operations |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| GL-EUDR-APP v1.0 | API integration | Improvement plans, dashboards, reports -> frontend display |
| EUDR-026 Due Diligence Orchestrator | Improvement plan status | Action completion status -> workflow coordination |
| EUDR-030 Documentation Generator | Improvement documentation | Improvement plan data -> DDS continuous improvement section |
| EUDR-034 Annual Review Scheduler | Improvement progress | Year-over-year improvement metrics -> annual review comparison |
| External Auditors | Read-only API + reports | Audit-ready improvement documentation for competent authority inspection |
| Management Dashboards | API integration | Executive-level improvement KPIs, compliance trajectory, resource utilization |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Quarterly Improvement Cycle (Chief Compliance Officer)

```
1. CCO logs in to GL-EUDR-APP -> navigates to "Improvement Planning" module
2. Clicks "Aggregate Findings" -> system ingests findings from EUDR-028/029/032/033/034/024
   -> 847 findings consolidated from 6 agents; 123 duplicates removed; 48 correlation groups formed
3. Clicks "Run Gap Analysis" -> system evaluates compliance against all EUDR articles
   -> Compliance heat map displayed: Article 9 score 78%, Article 10 score 85%, Article 11 score 62%
4. Reviews critical gaps: 3 CRITICAL gaps in Article 11 (mitigation inadequacy for palm oil)
5. Clicks "Generate Improvement Plan" -> selects "Portfolio Level, Corrective"
   -> System generates 156 SMART actions in < 10 seconds
6. Reviews Eisenhower matrix: 12 actions in Q1 (Do First), 67 in Q2 (Schedule),
   48 in Q3 (Delegate), 29 in Q4 (Defer)
7. Approves Q1 actions -> system assigns RACI roles per default rules
   -> Notifications sent to 8 stakeholders
8. Over following weeks, monitors progress dashboard:
   -> Burndown chart shows 75% of Q1 actions completed in 3 weeks
   -> 2 blocked actions flagged -> CCO escalates to procurement director
9. After Q1 actions complete, reviews effectiveness:
   -> Average 82% effectiveness (actual vs. target improvement)
   -> Article 11 compliance score improved from 62% to 79%
10. Generates "Article 13 Continuous Improvement Report" for competent authority file
```

#### Flow 2: Root Cause Investigation (Supply Chain Analyst)

```
1. Analyst notices 14 findings from different agents all involving Indonesian rubber suppliers
2. Opens "Root Cause Analysis" module -> selects 14 related findings
3. Clicks "Run 5-Whys Analysis" -> system begins decomposition:
   Level 1: Why are Indonesian rubber suppliers non-compliant?
   -> Evidence: 14 findings from EUDR-017 (supplier risk), EUDR-033 (certification expiry),
      EUDR-032 (community grievances)
   Level 2: Why do these suppliers have elevated risk and expired certifications?
   -> Evidence: EUDR-016 shows Indonesia country risk score increased after CPI drop
   Level 3: Why did supplier certifications lapse without renewal?
   -> Evidence: EUDR-034 annual review shows suppliers lacked resources for recertification
   Level 4: Why do suppliers lack resources for certification?
   -> Evidence: EUDR-032 grievance data shows smallholder economic pressure
   Level 5: ROOT CAUSE -> Inadequate supplier capacity building program for Indonesian
      rubber smallholders; no training, no financial support, no recertification assistance
4. System generates root cause record with 85% confidence score
5. System recommends systemic intervention: "Design comprehensive Indonesian rubber
   supplier capacity building program" linked to CSDDD Article 10 prevention obligations
6. Analyst approves -> system generates 5 SMART actions for the intervention
7. Over 6 months, monitors whether the 14 recurring findings cease
```

#### Flow 3: Stakeholder Action Management (Procurement Director)

```
1. Procurement Director opens "My Actions" dashboard
2. Sees 8 actions assigned with R (Responsible) role, 3 overdue
3. Opens overdue action: "Engage 12 Ghanaian cocoa cooperatives for plot GPS data collection"
   -> SMART criteria: Specific (12 cooperatives in Ashanti region), Measurable (100% GPS coverage),
      Achievable (mobile data collector available via EUDR-015), Relevant (Article 9(1)(a) gap),
      Time-bound (45 days, deadline April 30, 2026)
   -> Resource estimate: 120 person-hours, EUR 15,000 travel + equipment
   -> Dependency: EUDR-015 mobile data collector configured (COMPLETED)
4. Updates progress: milestone 50% (6 of 12 cooperatives visited), uploads field visit evidence
5. System updates progress dashboard, sends "milestone achieved" notification to CCO
6. After all 12 cooperatives visited, marks action COMPLETED
7. System triggers effectiveness verification: checks EUDR-028 risk score for Ghanaian cocoa
   -> Pre-action: Article 9 compliance 64% -> Post-action: Article 9 compliance 91%
   -> Effectiveness: 96% of target achieved
8. Action moves to VERIFIED -> CLOSED
```

### 8.2 Key Screen Descriptions

**Unified Findings Dashboard:**
- Summary cards: Total findings (847), Critical (23), High (89), Medium (412), Low (323)
- Donut chart: findings by source agent (EUDR-028: 34%, EUDR-033: 28%, EUDR-032: 15%, EUDR-034: 12%, EUDR-029: 7%, EUDR-024: 4%)
- Trend chart: findings over time (monthly, with severity breakdown)
- Correlation groups: expandable list showing related findings grouped together
- Filters: severity, source agent, EUDR article, commodity, country, date range

**Compliance Heat Map:**
- Matrix: rows = EUDR articles (4, 8, 9, 10, 11, 12, 13, 29, 31), columns = commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood)
- Cell color: green (>= 80%), yellow (50-79%), red (< 50%)
- Click cell to drill down: specific gaps for that article-commodity combination
- Side panel: compliance score trend over time for selected article

**Eisenhower Matrix View:**
- Four-quadrant grid with action count per quadrant
- Q1 (top-right, red): Do First actions with countdown to deadline
- Q2 (top-left, blue): Schedule actions with resource allocation bars
- Q3 (bottom-right, yellow): Delegate actions with assignee names
- Q4 (bottom-left, gray): Defer actions with deferral justification
- Click any action to open detail panel with SMART criteria, RACI, and progress

**Progress Dashboard:**
- Overall completion: circular progress indicator with percentage
- Burndown chart: planned vs. actual action completion over time
- Velocity chart: actions completed per week with trend line
- Status distribution: bar chart by action status (planned through closed)
- Effectiveness histogram: distribution of effectiveness percentages across completed actions
- Overdue alert panel: list of overdue and blocked actions with escalation status

**Root Cause Analysis View:**
- 5-Whys tree: vertical tree visualization with evidence links at each level
- Ishikawa diagram: fishbone visualization with six cause categories
- Root cause register: table of identified root causes with confidence scores
- Resolution tracker: status indicators (open, in progress, resolved, monitoring)
- Recurrence chart: timeline showing finding occurrences before and after root cause intervention

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 7 P0 features (Features 1-7) implemented and tested
  - [ ] Feature 1: Finding Aggregator -- ingests from all 6 upstream agents, normalizes, deduplicates, correlates
  - [ ] Feature 2: Gap Analysis Engine -- evaluates all relevant EUDR articles, generates compliance heat map
  - [ ] Feature 3: Action Plan Generator -- creates 100% SMART-validated actions with resource estimates
  - [ ] Feature 4: Root Cause Mapper -- 5-Whys, Ishikawa, cross-correlation functional
  - [ ] Feature 5: Prioritization Matrix -- Eisenhower classification, risk-weighted scoring, constraint optimization
  - [ ] Feature 6: Progress Tracker -- lifecycle management, effectiveness measurement, dashboards
  - [ ] Feature 7: Stakeholder Assignment -- RACI assignment, workload balancing, notifications
- [ ] >= 85% test coverage achieved (300+ tests)
- [ ] Security audit passed (JWT + RBAC integrated with 22 permissions)
- [ ] Performance targets met (< 10 seconds plan generation for 500+ findings)
- [ ] All 7 commodity-specific improvement plan templates tested with golden test fixtures
- [ ] Prioritization scoring verified deterministic (bit-perfect reproducibility)
- [ ] Gap analysis validated against manual expert assessment (>= 95% agreement)
- [ ] API documentation complete (OpenAPI spec for 30+ endpoints)
- [ ] Database migration V119 tested and validated
- [ ] Integration with EUDR-028, EUDR-029, EUDR-032, EUDR-033, EUDR-034, EUDR-024 verified
- [ ] 5 beta customers successfully generated and tracked improvement plans
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ improvement plans generated by customers
- 500+ SMART actions created and tracked
- Average action SMART validation pass rate >= 99%
- < 5 support tickets per customer
- p95 API latency < 200ms in production
- Finding aggregation covering all 6 upstream agent types

**60 Days:**
- 200+ improvement plans active
- 3,000+ SMART actions tracked
- Average plan effectiveness >= 65% (completed actions achieving target)
- Root cause analysis adopted by 30%+ of active customers
- Gap analysis run at least monthly by 80% of customers
- Overdue action rate < 15%

**90 Days:**
- 500+ improvement plans across portfolio, commodity, and entity levels
- 10,000+ SMART actions tracked
- Average plan effectiveness >= 70%
- Measurable compliance score improvement (>= 10 points average) for customers with 90+ day history
- Zero competent authority findings of inadequate continuous improvement for active customers
- NPS > 50 from compliance officer persona

---

## 10. Timeline and Milestones

### Phase 1: Core Intelligence Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Finding Aggregator (Feature 1): ingestion from 6 agents, normalization schema, deduplication engine, correlation rules | Senior Backend Engineer |
| 2-3 | Gap Analysis Engine (Feature 2): article-by-article evaluation, compliance scoring, gap severity classification, heat map generation | Senior Backend Engineer |
| 3-5 | Action Plan Generator (Feature 3): SMART action creation, action template library (100+ templates), resource estimation, dependency chain | Senior Backend Engineer + Domain Specialist |
| 5-6 | Root Cause Mapper (Feature 4): 5-Whys decomposition, Ishikawa diagram generation, cross-finding correlation, root cause register | Senior Backend Engineer |

**Milestone: Core intelligence engine operational with all P0 analysis features (Week 6)**

### Phase 2: Execution and Tracking Layer (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Prioritization Matrix (Feature 5): Eisenhower classification, risk-weighted scoring, constraint optimization, critical path analysis | Backend Engineer |
| 8-9 | Progress Tracker (Feature 6): lifecycle management, milestone tracking, effectiveness measurement, dashboards | Backend Engineer + Frontend Engineer |
| 9-10 | Stakeholder Assignment (Feature 7): RACI assignment, workload balancing, notification system, accountability reporting | Backend Engineer + Integration Engineer |

**Milestone: Full execution and tracking layer operational (Week 10)**

### Phase 3: API, Integration, and Reporting (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | REST API Layer: 30+ endpoints, authentication, authorization, rate limiting, OpenAPI documentation | Backend Engineer |
| 12-13 | Upstream agent integration: EUDR-028, EUDR-029, EUDR-032, EUDR-033, EUDR-034, EUDR-024 data connectors | Integration Engineer |
| 13-14 | Report generation: audit-ready improvement reports (JSON, PDF), Article 13 continuous improvement documentation, provenance hashes | Backend Engineer |

**Milestone: Full API operational with integrations and reporting (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 300+ tests, golden tests for all 7 commodities, integration tests, performance tests | Test Engineer |
| 16-17 | Performance testing (10K+ findings, 1K+ actions), security audit, load testing | DevOps + Security |
| 17 | Database migration V119 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers), feedback incorporation | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 7 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Benchmarking Engine (Feature 8)
- AI-Powered Improvement Recommendations (Feature 9)
- Collaboration Portal (Feature 10)
- Integration with Jira/Asana/Monday.com via webhook API
- Advanced analytics: improvement program ROI calculator, compliance trajectory forecasting

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EUDR-028 Risk Assessment Engine | BUILT (100%) | Low | Stable, production-ready; well-defined risk score output schema |
| EUDR-029 Mitigation Measure Designer | BUILT (100%) | Low | Stable, production-ready; effectiveness data available via API |
| EUDR-032 Grievance Mechanism Manager | BUILT (100%) | Low | Stable; pattern analytics and root cause data available |
| EUDR-033 Continuous Monitoring Agent | BUILT (100%) | Low | Stable; compliance events and alerts available |
| EUDR-034 Annual Review Scheduler | BUILT (100%) | Low | Stable; year-over-year comparison data available |
| EUDR-024 Third-Party Audit Manager | BUILT (100%) | Low | Stable; audit findings and non-conformances available |
| EUDR-030 Documentation Generator | BUILT (100%) | Low | DDS content and submission status for Article 12 gap analysis |
| EUDR-026 Due Diligence Orchestrator | BUILT (100%) | Low | Workflow coordination for improvement action triggers |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Frontend framework and API gateway operational |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure, hypertable support verified |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration with 22 new permissions |
| OBS-001 Prometheus Metrics | BUILT (100%) | Low | 15 self-monitoring metrics for operational visibility |
| OBS-004 Alerting Platform | BUILT (100%) | Low | Alert routing for overdue and blocked action notifications |
| AGENT-FOUND-005 Citations & Evidence | BUILT (100%) | Low | EUDR article citations for gap analysis mapping |
| AGENT-FOUND-008 Reproducibility Agent | BUILT (100%) | Low | Bit-perfect verification of priority scoring and gap analysis |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU Information System DDS schema | Published (v1.x) | Medium | Adapter pattern for schema version changes; Article 12 gap analysis validated against current schema |
| EC country benchmarking list (Article 29) | Published; updated periodically | Medium | Database-driven, hot-reloadable via EUDR-033; gap analysis recalculates on benchmarking updates |
| EU EUDR implementing regulations | Evolving | Medium | Configuration-driven article evaluation rules; new articles can be added without code changes |
| CSDDD enforcement timeline | Expected 2027 | Low | CSDDD remediation and prevention already included in action templates; enforcement date monitoring via EUDR-033 |
| Competent authority inspection criteria | Varies by member state | Medium | Conservative compliance evaluation; Article 13 documentation covers broadest interpretation |
| Email delivery service (SMTP/SES) | Available | Low | Multi-channel notification fallback (webhook, in-app if email fails) |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Upstream agent output schema changes break finding ingestion | Medium | High | Versioned ingestion adapters per agent; schema validation at ingestion boundary; backward-compatible normalization |
| R2 | Finding deduplication produces false merges (different issues merged as duplicate) | Medium | Medium | Conservative deduplication with configurable similarity thresholds; manual override to split incorrectly merged findings; deduplication accuracy monitored as KPI |
| R3 | Gap analysis article evaluation becomes outdated when regulation is amended | Medium | High | Configuration-driven article evaluation rules stored in database; rules can be updated without code deployment; EUDR-033 regulatory update feed triggers rule review |
| R4 | SMART action templates too generic for commodity-specific contexts | Medium | Medium | Commodity-specific template variants for all 7 commodities; template customization per operator; continuous template refinement based on effectiveness feedback |
| R5 | Root cause analysis produces incorrect root causes (low confidence) | Medium | Medium | Confidence scoring on all root causes; causes with < 50 confidence flagged for expert review; effectiveness tracking validates root cause accuracy over time |
| R6 | Stakeholder notification fatigue leads to ignored alerts | Medium | Medium | Configurable notification throttling; daily digest option; maximum notifications per user per day; priority-based notification routing |
| R7 | Improvement plan scope overwhelms operator resources | High | Medium | Constraint-based optimization feature recommends achievable subset; Eisenhower matrix separates urgent from important; phased implementation recommendations |
| R8 | Effectiveness measurement delayed due to upstream agent data latency | Medium | Low | Configurable verification delay; retry mechanism for pending verifications; partial effectiveness reporting while awaiting full data |
| R9 | Competent authority expects different improvement documentation format | Medium | High | Flexible report templates; Article 13 documentation follows conservative interpretation; PDF and JSON formats for different inspector preferences |
| R10 | Integration complexity with 6 upstream agent data feeds | Medium | Medium | Well-defined ingestion interfaces; mock adapters for testing; circuit breaker pattern; incremental ingestion (no need for all agents simultaneously) |
| R11 | Large operators with 500+ DDS generate improvement plans exceeding UI rendering capacity | Low | Medium | Server-side pagination; plan summary views with drill-down; lazy loading of action details; export to CSV/PDF for offline review |
| R12 | Manual root cause annotations from analysts are inconsistent or incomplete | Medium | Low | Structured annotation templates with mandatory fields; validation rules for evidence linkage; collaborative review workflow for root cause confirmation |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Finding Aggregator Unit Tests | 50+ | Ingestion from each agent type, normalization, deduplication (true positives, true negatives, edge cases), correlation grouping, incremental ingestion |
| Gap Analysis Unit Tests | 40+ | Per-article evaluation (Articles 4, 8, 9, 10, 11, 12, 13, 29, 31), compliance scoring, severity classification, enforcement exposure estimation, heat map generation |
| Action Plan Generator Unit Tests | 50+ | SMART validation (all 5 criteria), resource estimation, dependency chain resolution, template application, plan level generation (portfolio, commodity, supply chain, entity) |
| Root Cause Mapper Unit Tests | 40+ | 5-Whys decomposition (1-7 levels), Ishikawa diagram generation (all 6 categories), cross-finding correlation, confidence scoring, recurrence detection |
| Prioritization Matrix Unit Tests | 30+ | Eisenhower classification (all 4 quadrants), risk-weighted scoring (deterministic), constraint optimization (budget and FTE caps), critical path analysis, tiebreaking rules |
| Progress Tracker Unit Tests | 30+ | Lifecycle state transitions (all valid and invalid transitions), milestone recording, resource tracking, effectiveness measurement (positive, negative, partial), velocity calculation |
| Stakeholder Assignment Unit Tests | 30+ | RACI assignment (default rules, manual override), workload balancing (overload detection), notification triggers (all types), delegation, accountability reporting |
| API Tests | 40+ | All 30+ endpoints, authentication, authorization (22 permissions), error handling, pagination, rate limiting |
| Integration Tests | 30+ | Cross-agent integration with EUDR-028, EUDR-029, EUDR-032, EUDR-033, EUDR-034, EUDR-024 |
| Golden Tests | 35+ | All 7 commodities x 5 scenarios (complete compliance, critical gaps, recurring findings, multi-root-cause, resource-constrained) |
| Reproducibility Tests | 15+ | Bit-perfect verification of gap scores, priority scores, effectiveness calculations, action plan generation |
| Performance Tests | 10+ | 10K+ findings aggregation, 500+ finding plan generation, 1K+ action prioritization, concurrent API load |
| **Total** | **300+** | |

### 13.2 Golden Test Scenarios

Each of the 7 EUDR commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood) will have dedicated golden test scenarios:

1. **Complete compliance scenario**: Operator with full supply chain traceability, low risk scores, no gaps, no grievances. Expected: clean gap analysis, minimal improvement actions (continuous improvement type only), high compliance scores across all articles.

2. **Critical gap scenario**: Operator with multiple CRITICAL gaps in Article 9 (missing geolocation), Article 11 (inadequate mitigation), and Article 12 (expired DDS). Expected: corrective action plan with Q1 (Do First) actions, high enforcement exposure estimates, RACI assignment to Compliance Officer as Accountable.

3. **Recurring finding scenario**: Same finding type appears in three consecutive quarterly cycles from EUDR-033 and EUDR-028. Expected: Root Cause Mapper triggers 5-Whys analysis, identifies systemic root cause, generates root cause intervention action, tracks recurrence cessation after intervention.

4. **Multi-root-cause scenario**: Findings from EUDR-032 (grievances), EUDR-017 (supplier risk), EUDR-033 (certification expiry), and EUDR-034 (declining annual review scores) all trace to two interconnected root causes. Expected: Root Cause Mapper identifies both root causes with evidence chains, generates linked intervention actions, maps Ishikawa diagram across People, Partner, and Place categories.

5. **Resource-constrained scenario**: Operator has 200+ findings requiring improvement actions but budget cap of EUR 100,000 and 3 FTE capacity. Expected: Prioritization Matrix applies constraint optimization, recommends highest-impact subset within budget, generates phased implementation plan with Q1 critical actions and Q2 scheduled actions.

Total: 7 commodities x 5 scenarios = 35 golden test scenarios

### 13.3 Integration Test Matrix

| Source Agent | Finding Types Tested | Normalization Verified | Deduplication Tested |
|-------------|---------------------|----------------------|---------------------|
| EUDR-028 | Composite risk scores, 5-dimensional decomposition, Article 10(2) criteria, risk trends | Schema mapping, severity translation, article reference extraction | Cross-agent duplicate detection with EUDR-033 risk degradation events |
| EUDR-029 | Strategy effectiveness, measure implementation, verification results, residual risk | Effectiveness metric extraction, status normalization | Cross-agent duplicate detection with EUDR-028 mitigation-triggered re-assessments |
| EUDR-032 | Pattern analytics, root causes, remediation effectiveness, collective grievances | Grievance severity mapping, root cause seed extraction | Cross-agent correlation with EUDR-033 community-related monitoring events |
| EUDR-033 | Compliance alerts, deforestation correlations, risk degradation, data freshness, regulatory updates | Event severity mapping, temporal context extraction, alert deduplication | Cross-agent duplicate detection with EUDR-028 risk score changes |
| EUDR-034 | Year-over-year deltas, missed deadlines, checklist failures, multi-year trends | Delta severity translation, article reference mapping | Cross-agent correlation with EUDR-033 data freshness violations |
| EUDR-024 | Audit non-conformances, corrective actions, certification gaps, schedule deviations | Non-conformance severity mapping, article reference extraction | Cross-agent duplicate detection with EUDR-033 certification change events |

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **CSDDD** | EU Corporate Sustainability Due Diligence Directive |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **SMART** | Specific, Measurable, Achievable, Relevant, Time-bound -- framework for actionable objectives |
| **RACI** | Responsible, Accountable, Consulted, Informed -- stakeholder assignment matrix |
| **PDCA** | Plan-Do-Check-Act -- ISO 14001 continuous improvement cycle |
| **5-Whys** | Root cause analysis technique that iteratively asks "why" to drill to systemic causes |
| **Ishikawa Diagram** | Also known as fishbone diagram; structured root cause categorization method |
| **Eisenhower Matrix** | 2x2 prioritization framework based on urgency and importance |
| **Corrective Action** | Action taken to eliminate the cause of a detected non-conformance |
| **Preventive Action** | Action taken to eliminate the cause of a potential non-conformance before it occurs |
| **Continuous Improvement** | Recurring activity to enhance performance beyond minimum compliance requirements |
| **Finding** | A compliance-relevant observation, non-conformance, risk signal, or alert from any upstream EUDR agent |
| **Gap** | A measured shortfall between current compliance posture and the requirements of a specific EUDR article |
| **Root Cause** | The fundamental reason for a non-conformance or recurring finding that, if eliminated, prevents recurrence |
| **Effectiveness** | The degree to which an implemented improvement action achieved its intended compliance improvement |
| **Enforcement Exposure** | The estimated probability and magnitude of regulatory penalty for a given compliance gap |
| **Competent Authority** | The EU member state authority responsible for EUDR enforcement and operator inspection |

### Appendix B: EUDR Article 13 -- Review and Continuous Improvement

Article 13 of EUDR (Regulation (EU) 2023/1115) mandates that operators and traders who have exercised due diligence shall review and, where necessary, update their due diligence. This continuous improvement obligation is not a one-time exercise but an ongoing requirement that persists for the entire period during which the operator places EUDR-regulated products on the EU market.

Key requirements addressed by this agent:
- **Review obligation**: Regular review of the due diligence system's adequacy, including information gathering (Article 9), risk assessment (Article 10), and risk mitigation (Article 11)
- **Update obligation**: When review identifies gaps or inadequacies, operators must update their systems and procedures
- **Documentation obligation**: Reviews and updates must be documented for competent authority inspection under Articles 14-16
- **Retention obligation**: Review documentation must be retained for at least 5 years under Article 31

The Improvement Plan Creator agent directly implements this obligation by providing the structured process for identifying what needs to be improved (Finding Aggregator + Gap Analysis Engine), determining root causes (Root Cause Mapper), defining improvement actions (Action Plan Generator), prioritizing resources (Prioritization Matrix), executing improvements (Stakeholder Assignment + Progress Tracker), and verifying effectiveness (Progress Tracker effectiveness measurement).

### Appendix C: CSDDD Articles 9 and 10 -- Remediation and Prevention

The EU Corporate Sustainability Due Diligence Directive establishes additional requirements for remediation and prevention of adverse impacts:

**Article 9 (Remediation)**: Companies must provide remediation when they have caused or contributed to actual adverse impacts. The Improvement Plan Creator supports this by generating Corrective Action Plans that address identified adverse impacts (including those surfaced through EUDR-032 Grievance Mechanism Manager) with SMART remediation actions, stakeholder assignment, and effectiveness verification.

**Article 10 (Prevention)**: Companies must take appropriate measures to prevent potential adverse impacts that have been identified through due diligence. The Improvement Plan Creator supports this by generating Preventive Action Plans based on risk assessment findings from EUDR-028, continuous monitoring alerts from EUDR-033, and root cause analysis that identifies systemic vulnerabilities before they manifest as non-conformances.

### Appendix D: Prioritization Scoring Methodology

The risk-weighted priority scoring formula used by the Prioritization Matrix (Feature 5) is designed to be fully deterministic, configurable, and auditable:

```
Priority_Score = (Severity_Weight * W_s) + (Enforcement_Exposure * W_e) +
                 (Risk_Impact * W_r) + (Resource_Efficiency * W_re) +
                 (Dependency_Criticality * W_d)

Default weights:
  W_s  = 0.30 (Severity Weight)
  W_e  = 0.25 (Enforcement Exposure)
  W_r  = 0.20 (Risk Impact)
  W_re = 0.15 (Resource Efficiency)
  W_d  = 0.10 (Dependency Criticality)

Where:
  Severity_Weight = Gap severity normalized to 0-100:
    CRITICAL = 100, HIGH = 75, MEDIUM = 50, LOW = 25

  Enforcement_Exposure = (Detection_Probability * Penalty_Magnitude) / Max_Penalty
    Detection_Probability: 0.0-1.0 based on inspection targeting criteria
    Penalty_Magnitude: estimated fine in EUR
    Max_Penalty: 4% of operator annual EU turnover

  Risk_Impact = Delta in composite risk score if gap is not closed
    Sourced from EUDR-028 Risk Assessment Engine
    Normalized to 0-100 scale

  Resource_Efficiency = Expected_Improvement / Resource_Cost
    Expected_Improvement: projected compliance score increase (0-100)
    Resource_Cost: estimated person-hours + EUR cost, normalized to 0-100

  Dependency_Criticality = Number of downstream actions blocked by this action
    Normalized to 0-100 where max = total action count in plan
```

All calculations use Python Decimal arithmetic with ROUND_HALF_UP rounding mode. Priority scores are stored with 4 decimal places. Operators can adjust all weights (W_s, W_e, W_r, W_re, W_d) through configuration without code changes. Weight adjustments are recorded in the audit trail with justification text.

### Appendix E: Root Cause Analysis Methodologies

**5-Whys Methodology (EUDR-Adapted):**
1. Level 1: What finding was detected? (Linked to normalized finding from Finding Aggregator)
2. Level 2: Why did this finding occur? (Evidence from agent-specific analysis)
3. Level 3: Why did the Level 2 cause exist? (Cross-agent correlation evidence)
4. Level 4: Why did the Level 3 cause persist? (Systemic analysis -- policy, process, capacity)
5. Level 5: What is the root cause? (Actionable systemic cause that, if addressed, prevents recurrence)

Each level requires:
- Evidence reference (source agent output ID, document ID, or expert annotation)
- Confidence score (0-100) for the causal linkage
- Analyst identification (for audit trail)

**Ishikawa Categories (EUDR-Adapted):**
- **People**: Compliance team capacity, training gaps, role clarity, accountability, knowledge retention, language barriers in supply chain communication
- **Process**: Due diligence workflow gaps, quality gate failures, agent coordination handoff errors, escalation process failures, data refresh cadence
- **Policy**: EUDR article interpretation, risk threshold calibration, scope definition gaps, materiality threshold settings, simplified DD eligibility criteria
- **Place**: Country-specific risk factors (deforestation rates, governance quality, infrastructure), regional connectivity, jurisdictional complexity, protected area proximity
- **Product**: Commodity traceability inherent challenges (latex aggregation, cattle movement, batch mixing), transformation complexity, unit conversion, species identification
- **Partner**: Supplier governance maturity, certification body capacity, auditor availability, competent authority inspection practices, NGO relationship status

### Appendix F: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EUDR)
2. EU Corporate Sustainability Due Diligence Directive (CSDDD)
3. ISO 14001:2015 -- Environmental Management Systems (PDCA continuous improvement)
4. ISO 31000:2018 -- Risk Management -- Guidelines
5. ISO 19011:2018 -- Guidelines for Auditing Management Systems
6. OECD Guidelines for Multinational Enterprises on Responsible Business Conduct (2023 update)
7. UN Guiding Principles on Business and Human Rights (UNGP)
8. ILO Convention 169 -- Indigenous and Tribal Peoples Convention
9. EUDR Technical Specifications for the EU Information System
10. EC Country Benchmarking Methodology (Article 29)

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
| 1.0.0-draft | 2026-03-12 | GL-ProductManager | Initial draft created: 14 sections, 7 P0 features (Finding Aggregator, Gap Analysis Engine, Action Plan Generator, Root Cause Mapper, Prioritization Matrix, Progress Tracker, Stakeholder Assignment), 30+ API endpoints, V119 migration schema, 22 RBAC permissions, 15 Prometheus metrics, 300+ test target, regulatory coverage for EUDR Articles 4/8/9/10/11/12/13/14-16/23-25/29/31 plus CSDDD Articles 9-10 |
