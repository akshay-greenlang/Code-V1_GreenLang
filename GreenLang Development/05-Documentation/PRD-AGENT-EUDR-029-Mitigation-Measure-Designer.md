# PRD: AGENT-EUDR-029 -- Mitigation Measure Designer

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-029 |
| **Agent ID** | GL-EUDR-MMD-029 |
| **Component** | Mitigation Measure Designer Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence (Category 5) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-11 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-11 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 10, 11, 12, 13, 14-16, 29, 31 |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Article 11 mandates that when a risk assessment (Article 10) identifies non-negligible risk that products placed on the EU market are not compliant, operators "shall not place the relevant products on the market or export them" until they have "carried out adequate and proportionate risk mitigation measures to reach the conclusion that the risk of non-compliance is negligible." This creates a mandatory risk mitigation gate between risk assessment and market placement -- operators who identify elevated risk must design, implement, verify, and document mitigation measures before their products can legally enter the EU market.

The GreenLang platform has built a comprehensive risk assessment engine (EUDR-028) that aggregates signals from five upstream risk agents (EUDR-016 Country Risk Evaluator, EUDR-017 Supplier Risk Scorer, EUDR-018 Commodity Risk Analyzer, EUDR-019 Corruption Index Monitor, EUDR-020 Deforestation Alert System) into composite risk scores classified across five tiers (NEGLIGIBLE, LOW, STANDARD, HIGH, CRITICAL). When EUDR-028 classifies risk as HIGH (61-80) or CRITICAL (81-100), Article 11 mitigation is triggered. However, the platform currently has no engine to design, manage, track, verify, and document the mitigation measures that Article 11 requires. Operators face the following critical gaps:

- **No systematic risk decomposition for mitigation targeting**: When EUDR-028 produces a composite risk score of HIGH or CRITICAL, compliance officers receive a single score and a decomposition showing which risk dimensions are elevated. However, there is no engine that analyzes this decomposition and translates it into specific, actionable mitigation measures. Compliance officers must manually interpret risk decomposition outputs, identify which risk drivers can be mitigated, research available mitigation options, and design a mitigation strategy -- a process that takes 40-80 hours per product line and produces inconsistent, non-reproducible results across teams and time periods.

- **No curated mitigation measure library**: Article 11(2) specifies three categories of risk mitigation measures: (a) requiring additional information, data, or documents from suppliers, (b) carrying out independent surveys and audits, and (c) other measures adapted to the complexity of the supply chain. Operators have no standardized library of proven mitigation measure templates organized by Article 11(2) category, risk dimension, commodity type, and expected effectiveness. Each mitigation strategy is designed from scratch, leading to inconsistent approaches, missed best practices, and no institutional learning.

- **No effectiveness estimation before commitment**: Before committing resources to mitigation measures (which can cost EUR 5,000-200,000+ for independent audits and site visits), operators need to estimate whether proposed measures will plausibly reduce risk to the negligible/low level required by Article 11(1). There is no estimation engine that predicts expected risk reduction based on measure type, applicability to the specific risk drivers, and implementation quality factors. Operators invest in mitigation measures without confidence that the measures will achieve the required risk reduction.

- **No implementation lifecycle tracking**: Mitigation measures have complex lifecycles: they must be proposed, approved by compliance leadership, assigned to responsible parties, implemented with evidence collection, verified for effectiveness, and closed. There is no structured workflow engine that tracks each measure through these stages, records milestones and evidence, monitors deadlines, and alerts on overdue measures. Compliance officers manage mitigation activities through spreadsheets and email, losing visibility into progress and creating audit trail gaps.

- **No risk re-evaluation loop**: Article 11(1) requires operators to "reach the conclusion that the risk of non-compliance is negligible" after mitigation. This means the risk assessment must be re-run with updated data reflecting the mitigation measures taken. There is no automated verification loop that, after measures are implemented, triggers EUDR-028 to re-evaluate risk with the updated inputs and compares pre-mitigation versus post-mitigation scores to determine whether the Article 11 threshold has been met.

- **No Article 11 compliance documentation**: Articles 14-16 empower competent authorities to inspect operators' due diligence systems, including risk mitigation measures. Article 31 requires records to be kept for 5 years. Operators must demonstrate to regulators exactly what mitigation measures were taken, why they were chosen, what evidence was collected, whether risk was successfully reduced, and the complete decision chain from risk trigger to resolution. There is no report generation engine that produces DDS-ready Article 11 compliance documentation with full provenance tracking.

- **No approval workflow with role-based gates**: Mitigation measures often involve significant expenditure (third-party audits, site visits, enhanced monitoring) and regulatory commitment. There is no structured approval workflow that requires Compliance Officer sign-off before measures proceed to implementation, tracks approval decisions with justification, and maintains an audit trail of who authorized what and when.

- **No institutional learning from mitigation outcomes**: When mitigation measures succeed or fail in reducing risk, that outcome information is not captured or fed back into future strategy design. There is no feedback mechanism that tracks which measures achieved their expected risk reduction for which risk dimensions and commodities, enabling continuous improvement of mitigation strategy effectiveness.

Without solving these problems, operators who identify elevated risk through EUDR-028 are left without a systematic, auditable, efficient process to satisfy Article 11. They cannot design targeted mitigation strategies, estimate effectiveness, track implementation, verify risk reduction, or generate compliant documentation. This delays market placement, increases compliance costs, creates audit trail gaps, and exposes operators to penalties of up to 4% of annual EU turnover, goods confiscation, temporary exclusion from public procurement, and public naming under Articles 23-25.

### 1.2 Solution Overview

Agent-EUDR-029: Mitigation Measure Designer is the compliance engine for EUDR Article 11 risk mitigation. It receives risk assessment results from EUDR-028 (Risk Assessment Engine), analyzes risk decomposition to identify which risk dimensions are driving elevated risk, designs targeted mitigation strategies from a curated library of 50+ measure templates mapped to Article 11(2) categories, estimates measure effectiveness using deterministic formulas, tracks implementation lifecycle with milestone tracking, triggers risk re-evaluation through EUDR-028 to verify risk reduction, orchestrates the full Article 11 workflow from risk trigger to closure, and generates DDS-ready mitigation compliance reports. It operates as a purely deterministic, zero-hallucination workflow and computation engine with no LLM in the critical path.

Core capabilities:

1. **Mitigation Strategy Designer** -- Analyzes risk assessment decomposition from EUDR-028 to identify which risk dimensions are driving elevated risk. Uses decision tree logic to select optimal mitigation measures for each driver: for country risk, request country-specific documentation and independent audits; for supplier risk, enhanced supplier verification and site visits; for deforestation risk, satellite monitoring intensification and plot-level verification; for commodity risk, traceability enhancement and certification requirements; for corruption risk, independent third-party audits and enhanced documentation. Supports multi-driver strategies where multiple risk dimensions are elevated simultaneously. Each strategy maps every proposed measure to a specific Article 11(2) category.

2. **Measure Template Library** -- Curated library of 50+ mitigation measure templates organized by Article 11(2) category: (a) additional information from suppliers, (b) independent surveys and audits, (c) other adapted measures. Each template includes: unique measure ID, title, description, applicable risk dimensions, Article 11(2) category, expected base effectiveness (risk reduction percentage), typical implementation timeline (days), required resources (cost range, personnel), evidence requirements, and regulatory references. Templates are configurable per commodity and per country risk level. Includes pre-built templates for all 7 EUDR commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood).

3. **Effectiveness Estimator** -- Estimates expected risk reduction from proposed mitigation measures using deterministic formulas. For each measure: Expected_Risk_Reduction = Base_Effectiveness * Applicability_Factor * Implementation_Quality_Factor. Provides conservative, moderate, and optimistic estimates. Validates that proposed measures can plausibly reduce risk to negligible/low level before the operator commits resources. Uses Decimal arithmetic to prevent floating-point drift. All estimates are reproducible.

4. **Measure Implementation Tracker** -- Tracks the lifecycle of each mitigation measure through defined states: PROPOSED, APPROVED, IN_PROGRESS, COMPLETED, VERIFIED, CLOSED. Records milestones, evidence uploads (document IDs, photo references, audit report links), responsible parties, deadlines, and progress notes. Generates implementation progress reports. Sends alerts for overdue measures. Calculates implementation completeness percentage per strategy and per measure.

5. **Risk Reduction Verifier** -- After measures are implemented, triggers re-evaluation of risk by invoking EUDR-028 Risk Assessment Engine with updated data reflecting the mitigation measures taken. Compares pre-mitigation versus post-mitigation risk scores at both composite and per-dimension levels. Verifies that risk has been reduced to NEGLIGIBLE or LOW level as required by Article 11(1). If reduction is insufficient, recommends additional measures and initiates a secondary mitigation cycle.

6. **Compliance Workflow Engine** -- Orchestrates the full Article 11 workflow: receive risk trigger from EUDR-028 (HIGH or CRITICAL classification) -> design mitigation strategy -> estimate effectiveness -> approval gate (Compliance Officer authorization) -> implement measures -> verify risk reduction -> generate report -> close workflow or escalate. Supports parallel measure tracks (multiple measures implemented simultaneously). Integrates with EUDR-028 for risk triggers and verification. Manages approval workflows with role-based gates.

7. **Mitigation Report Generator** -- Generates DDS-ready Article 11 compliance reports formatted for regulatory inspection. Reports include: risk trigger summary (from EUDR-028 assessment), risk decomposition showing which dimensions triggered mitigation, mitigation strategy designed with Article 11(2) category mapping, measures implemented with evidence references, effectiveness verification results (pre vs. post risk scores), final risk determination, regulatory references for every decision, and SHA-256 provenance hashes for data integrity verification. Supports JSON (machine-readable) and PDF (human-readable) output formats.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Article 11 criteria coverage | 100% of Article 11(1) and 11(2) requirements addressed | Regulatory compliance audit |
| Average risk reduction achieved | > 30 points for HIGH risk assessments; > 50 points for CRITICAL | Pre vs. post mitigation score comparison |
| Measure implementation tracking | 100% of measures have tracked status | % of measures with defined lifecycle state |
| Workflow completion time | < 30 days average from risk trigger to workflow closure | Workflow duration metrics |
| Report generation time | < 5 seconds per DDS-ready report (JSON); < 15 seconds (PDF) | Report generation benchmarks |
| Strategy design time | < 2 seconds per mitigation strategy computation | Strategy design latency |
| Effectiveness estimation accuracy | Within +/- 10 points of actual risk reduction achieved | Estimated vs. actual comparison over 100+ completed workflows |
| Template library coverage | 50+ templates covering all 7 commodities and all 3 Article 11(2) categories | Template count and coverage matrix |
| Bit-perfect reproducibility | Same input produces identical output across runs | Reproducibility test suite with hash comparison |
| Regulatory acceptance | 100% of mitigation reports accepted in DDS submissions | EU Information System validation |
| Test coverage | 200+ tests, >= 85% line coverage | Test suite metrics |
| Overdue measure detection | 100% of overdue measures flagged within 1 hour | Alert latency benchmarks |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated risk mitigation management market of 1-3 billion EUR as operators require systematic tools to design, implement, and document Article 11 mitigation measures.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities that identify non-negligible risk and must demonstrate adequate mitigation before market placement, estimated at 300-600M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 20-40M EUR in mitigation management module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) of EUDR-regulated commodities who regularly encounter elevated risk assessments requiring Article 11 mitigation
- Multinational food and beverage companies sourcing cocoa, coffee, palm oil, and soya from high-risk regions where mitigation is routinely triggered
- Timber and paper industry operators with complex, multi-country supply chains involving high-deforestation-risk countries
- Rubber and cattle product importers sourcing from countries with elevated corruption and governance risk

**Secondary:**
- Customs brokers and freight forwarders handling EUDR-regulated goods where Article 11 mitigation may be delegated by operators
- Commodity traders and intermediaries with pass-through due diligence obligations requiring mitigation documentation
- Compliance consultants advising multiple operators on EUDR Article 11 strategy and documentation
- Certification bodies (FSC, RSPO, Rainforest Alliance) offering risk mitigation as a value-added verification service
- SME importers (1,000-10,000 shipments/year) preparing for June 30, 2026 enforcement and building mitigation capability

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Spreadsheet | No cost; familiar process | No structured workflow; no template library; no effectiveness estimation; 40-80 hours per mitigation cycle; fails audit | Automated strategy design, template library, workflow tracking, < 2 seconds per strategy |
| Generic GRC platforms (ServiceNow, Archer) | Enterprise workflow management; configurable | Not EUDR-specific; no Article 11(2) category mapping; no risk agent integration; no DDS output | Purpose-built for Article 11; integrated with EUDR-028 risk engine; DDS-ready reports |
| Management consulting firms | Deep regulatory expertise; bespoke strategy | EUR 50K-200K per engagement; slow (weeks-months); not scalable; no real-time tracking | Automated, EUR 5K-20K per year; instant strategy design; 50+ reusable templates |
| Niche EUDR compliance tools | Commodity expertise; some risk tools | No mitigation lifecycle management; no effectiveness estimation; no verification loop | Full lifecycle from trigger to verification; deterministic effectiveness estimation; EUDR-028 integration |
| In-house custom builds | Tailored to organization; full control | 12-18 month build time; no template library; no cross-customer learning | Ready now; 50+ pre-built templates; continuous regulatory updates |

### 2.4 Differentiation Strategy

1. **EUDR-028 integration** -- No competitor directly integrates with a production-grade, 5-agent composite risk assessment engine. The Mitigation Measure Designer receives precise risk decomposition from EUDR-028 and feeds verification results back, creating a closed-loop mitigation cycle that no manual or standalone tool can replicate.

2. **Article 11(2) category-mapped template library** -- 50+ mitigation measure templates mapped to the three specific categories of Article 11(2), pre-calibrated per commodity and country risk level. No competitor offers this level of regulatory specificity in mitigation measure design.

3. **Deterministic effectiveness estimation** -- Before operators commit resources, the engine estimates expected risk reduction using reproducible, auditable formulas. This eliminates the guesswork inherent in manual mitigation planning.

4. **Full lifecycle workflow with verification loop** -- From risk trigger to strategy design to implementation tracking to risk re-evaluation to report generation, the entire Article 11 workflow is managed in a single, auditable system with SHA-256 provenance tracking at every step.

5. **DDS-native Article 11 reporting** -- Mitigation reports are formatted directly for EU Information System DDS submission, eliminating manual transcription and ensuring regulatory compliance of documentation.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to satisfy Article 11 mitigation requirements with defensible, documented strategies | 100% of customers pass Article 11 regulatory audits | Q2 2026 |
| BG-2 | Reduce mitigation strategy design time from weeks to seconds | > 99% reduction in strategy design time | Q2 2026 |
| BG-3 | Reduce average time from risk trigger to market placement approval | < 30 days average workflow completion (vs. 60-120 days manual) | Q3 2026 |
| BG-4 | Eliminate audit trail gaps in mitigation documentation | 100% of mitigation measures fully documented with provenance | Q2 2026 |
| BG-5 | Become the reference mitigation management system for EUDR compliance | 500+ enterprise customers | Q4 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Targeted strategy design | Analyze EUDR-028 risk decomposition and design mitigation strategies targeting specific risk drivers |
| PG-2 | Template-based measure selection | Provide 50+ curated, Article 11(2)-mapped mitigation measure templates for all 7 commodities |
| PG-3 | Effectiveness estimation | Estimate expected risk reduction before resource commitment using deterministic formulas |
| PG-4 | Implementation lifecycle tracking | Track every measure through PROPOSED -> APPROVED -> IN_PROGRESS -> COMPLETED -> VERIFIED -> CLOSED |
| PG-5 | Risk reduction verification | Re-evaluate risk through EUDR-028 after mitigation and verify Article 11(1) compliance |
| PG-6 | Compliance workflow orchestration | Manage the full Article 11 workflow with approval gates and parallel measure tracks |
| PG-7 | DDS-ready reporting | Generate Article 11 compliance reports in EU Information System format |
| PG-8 | Institutional learning | Track which measures succeed for which risk dimensions and commodities to improve future strategies |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Strategy design latency | < 2 seconds p95 for mitigation strategy computation |
| TG-2 | Effectiveness estimation latency | < 500ms p95 per measure estimation |
| TG-3 | API response time | < 200ms p95 for standard queries |
| TG-4 | Report generation latency | < 5 seconds p95 (JSON), < 15 seconds p95 (PDF) |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |
| TG-7 | Memory efficiency | < 256 MB for 1,000 concurrent active workflows in memory |
| TG-8 | Database query performance | < 50ms p95 for strategy and measure lookups |

---

## 4. User Personas

### Persona 1: Compliance Officer -- Elena (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of EUDR Compliance at a large EU chocolate manufacturer |
| **Company** | 8,000 employees, importing cocoa from 15 countries including Ghana, Cote d'Ivoire, Cameroon, and Indonesia |
| **EUDR Pressure** | Board-level mandate to achieve full Article 11 compliance; competent authority audit expected Q3 2026; 30% of supply chains currently flagged as HIGH risk by EUDR-028 |
| **Pain Points** | Manually designs mitigation strategies in PowerPoint; no standardized template library; cannot estimate if proposed measures will actually reduce risk sufficiently; tracks implementation in Excel with frequent gaps; generates mitigation reports manually in Word; 60+ hours per mitigation cycle; no systematic verification that risk has been reduced after measures are taken |
| **Goals** | Automated, targeted mitigation strategy design based on risk decomposition; pre-commitment effectiveness estimation; structured implementation tracking; automated risk re-evaluation; DDS-ready Article 11 reports; institutional learning from past mitigation outcomes |
| **Technical Skill** | Moderate -- comfortable with web applications, dashboards, and report generation |

### Persona 2: Risk Analyst -- Marco (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Risk Analyst at an EU palm oil refinery |
| **Company** | 3,000 employees, sourcing from 200+ plantations across Indonesia and Malaysia; routinely encounters HIGH and CRITICAL risk assessments due to country and deforestation risk |
| **EUDR Pressure** | Must process 50+ mitigation workflows per quarter; needs to select optimal mitigation measures from available options and estimate cost-effectiveness before recommending to leadership |
| **Pain Points** | No structured way to select measures based on specific risk drivers; cannot estimate risk reduction before committing to expensive audits (EUR 20K-50K each); no feedback on which past measures actually worked; spends 3 days per mitigation strategy design; manually tracks 100+ active measures across spreadsheets |
| **Goals** | Data-driven measure selection based on risk decomposition; cost-effectiveness estimation; automated tracking of all active measures with deadline alerts; access to historical measure effectiveness data; ability to design multi-driver strategies for complex risk profiles |
| **Technical Skill** | High -- comfortable with data analysis tools, APIs, risk models, and quantitative methods |

### Persona 3: Procurement Manager -- Isabelle (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Procurement Director at an EU timber importer |
| **Company** | 1,500 employees, importing tropical hardwood from 10+ countries; many suppliers in Central Africa and Southeast Asia |
| **EUDR Pressure** | Must implement supplier-facing mitigation measures (request additional documentation, arrange site visits, require certifications); needs to coordinate with suppliers across multiple time zones and languages |
| **Pain Points** | Receives mitigation action items from compliance team but no structured workflow to execute them; cannot track which suppliers have responded to documentation requests; no visibility into overall mitigation progress; supplier communication is ad hoc via email |
| **Goals** | Clear task list of supplier-facing measures with deadlines; evidence upload interface for supplier documentation; progress dashboard showing measure completion per supplier; notification when measures are overdue |
| **Technical Skill** | Low-moderate -- uses ERP and web applications; comfortable with task lists and dashboards |

### Persona 4: External Auditor -- Dr. Richter (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm specializing in supply chain compliance |
| **EUDR Pressure** | Must verify that operators' Article 11 mitigation measures are adequate, proportionate, and documented; needs to assess whether risk was genuinely reduced through the measures taken |
| **Pain Points** | Operators provide mitigation documentation in inconsistent formats; cannot verify whether risk re-evaluation was genuinely performed after mitigation; no standardized way to assess measure adequacy; override and approval justifications are often missing |
| **Goals** | Access to read-only mitigation workflow records with full provenance; ability to verify pre vs. post mitigation risk scores independently; evidence package review with document integrity verification; complete audit trail of approval decisions and measure outcomes |
| **Technical Skill** | Moderate -- comfortable with audit software, document review, and regulatory assessment frameworks |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 11(1)** | Where risk assessment identifies non-negligible risk, operators shall carry out adequate and proportionate risk mitigation measures to reach the conclusion of negligible risk before placing products on the market | Compliance Workflow Engine triggers mitigation when EUDR-028 classifies risk as HIGH or CRITICAL; Risk Reduction Verifier confirms risk reduced to NEGLIGIBLE/LOW before workflow closure |
| **Art. 11(2)(a)** | Risk mitigation measures may include: requiring additional information, data, or documents from the operators or traders that supplied the relevant commodities or products | Measure Template Library provides 15+ templates in category (a): enhanced supplier declarations, additional origin documentation, certificate of legality requests, production date verification, geolocation enhancement, customs documentation, phytosanitary certificates |
| **Art. 11(2)(b)** | Risk mitigation measures may include: carrying out independent surveys and audits | Measure Template Library provides 15+ templates in category (b): independent third-party audits, site visit inspections, satellite monitoring intensification, laboratory testing, sample verification, community consultation |
| **Art. 11(2)(c)** | Risk mitigation measures may include: any other measures adapted to the complexity of the supply chain and relevant risk factors | Measure Template Library provides 20+ templates in category (c): supply chain simplification, alternative sourcing, enhanced traceability systems, blockchain verification, certification enrollment, training programs, stakeholder engagement |
| **Art. 10** | Risk assessment results triggering Article 11 | Risk decomposition from EUDR-028 used by Mitigation Strategy Designer to identify specific risk drivers and select targeted measures |
| **Art. 12** | Due diligence statement content | Mitigation Report Generator produces mitigation documentation formatted for DDS submission |
| **Art. 13** | Simplified due diligence (relevant for post-mitigation eligibility) | Risk Reduction Verifier checks if post-mitigation score qualifies for simplified DD under Article 13 |
| **Art. 14-16** | Competent authority inspections of due diligence systems | All mitigation strategies, measures, approvals, evidence, and verification results stored with SHA-256 provenance hashes for regulatory inspection |
| **Art. 29** | Country benchmarking affecting mitigation strategy intensity | Mitigation Strategy Designer adjusts measure intensity based on country benchmarking level (HIGH-risk countries trigger more intensive measures) |
| **Art. 31** | Record keeping for 5 years | All mitigation records stored in TimescaleDB with 5-year retention policies; immutable audit trail |

### 5.2 Article 11(2) Mitigation Measure Categories

The following table details the three Article 11(2) categories and representative mitigation measure templates within each:

| Category | Article | Description | Template Count | Example Measures |
|----------|---------|-------------|----------------|------------------|
| **(a) Additional information from suppliers** | 11(2)(a) | Requiring additional information, data, or documents from suppliers | 15+ | Request origin country documentation, demand GPS coordinates for production plots, require supplier sustainability certification copies, request production date verification, demand chain of custody records, request legal compliance certificates |
| **(b) Independent surveys and audits** | 11(2)(b) | Carrying out independent surveys and audits | 15+ | Commission third-party field audit, conduct satellite monitoring review, arrange independent laboratory analysis, perform on-site supplier visit, engage local community consultation, conduct mass balance verification audit, request independent certification body assessment |
| **(c) Other adapted measures** | 11(2)(c) | Any other measures adapted to supply chain complexity | 20+ | Implement enhanced traceability technology (QR, blockchain), switch to certified-only sourcing, diversify supply chain away from high-risk origins, require supplier enrollment in certification scheme, implement continuous satellite monitoring of production plots, conduct supplier training on EUDR requirements, establish local monitoring partnerships |

### 5.3 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for all deforestation-free verification; mitigation measures must address post-cutoff deforestation risk |
| June 29, 2023 | Regulation entered into force | Legal basis for all Article 11 mitigation requirements |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Operators must have mitigation capability operational for Article 11 compliance |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; mitigation engine must scale to handle increased workflow volume |
| Quarterly (ongoing) | EC country benchmarking updates | Mitigation strategies may need redesign when country risk classifications change |
| Ongoing | Competent authority inspections | Mitigation documentation must be audit-ready at all times per Articles 14-16 |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-5 form the core mitigation computation and lifecycle engine; Features 6-9 form the workflow, reporting, and integration layer that delivers mitigation intelligence to users and regulators.

**P0 Features 1-5: Core Mitigation Computation and Lifecycle Engine**

---

#### Feature 1: Mitigation Strategy Designer

**User Story:**
```
As a compliance officer,
I want the system to analyze my risk assessment decomposition and design a targeted mitigation strategy,
So that I can address the specific risk drivers identified by EUDR-028 with appropriate Article 11(2) measures.
```

**Acceptance Criteria:**
- [ ] Accepts risk assessment results from EUDR-028 including: composite risk score, risk level (HIGH or CRITICAL), per-dimension scores and confidences, Article 10(2) criteria evaluation, country benchmarking classification
- [ ] Decomposes the composite risk score to identify which dimensions are driving elevated risk (dimensions scoring > 60)
- [ ] Ranks risk drivers by weighted contribution to the composite score (highest contribution first)
- [ ] For each risk driver, selects optimal mitigation measures from the template library (Feature 2) using decision tree logic:
  - Country risk > 60: request country-specific documentation + independent country audit + local community consultation
  - Supplier risk > 60: enhanced supplier verification + site visit + compliance history review
  - Deforestation risk > 60: satellite monitoring intensification + plot-level GPS verification + field audit
  - Commodity risk > 60: traceability enhancement + certification requirements + chain of custody audit
  - Corruption risk > 60: independent third-party audit + enhanced documentation + anti-fraud verification
  - Supply chain complexity > 60: supply chain simplification + multi-tier verification + enhanced traceability
  - Mixing risk > 60: identity preserved sourcing + mass balance audit + batch segregation verification
  - Circumvention risk > 60: trade flow analysis + origin verification + customs documentation audit
- [ ] Maps every proposed measure to its Article 11(2) category: (a), (b), or (c)
- [ ] Supports multi-driver strategies where 2+ risk dimensions are elevated simultaneously
- [ ] Generates a MitigationStrategy object with: strategy ID, linked assessment ID, risk trigger summary, proposed measures (ordered by expected impact), estimated total risk reduction, estimated timeline, estimated resource requirements
- [ ] All strategy design logic is deterministic (same risk decomposition produces same strategy)
- [ ] Generates SHA-256 provenance hash of the strategy for audit trail

**Non-Functional Requirements:**
- Performance: < 2 seconds for strategy design computation
- Determinism: Bit-perfect reproducibility across runs
- Auditability: Complete decision log showing why each measure was selected for each risk driver

**Dependencies:**
- EUDR-028 Risk Assessment Engine (risk assessment results, composite score decomposition)
- Measure Template Library (Feature 2) for measure selection
- Effectiveness Estimator (Feature 3) for risk reduction estimation

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- All 8 risk dimensions are elevated (> 60) -- Design comprehensive multi-driver strategy addressing all dimensions; flag as "complex mitigation required"
- Only 1 dimension is elevated -- Design focused single-driver strategy
- Composite score is CRITICAL (> 80) -- Apply enhanced measure intensity; require multiple measures per driver
- Risk decomposition shows no single dimension > 60 but composite is HIGH -- Composite effect; apply moderate measures across multiple dimensions
- EUDR-028 assessment has missing dimensions -- Design strategy for available dimensions only; flag missing dimensions as "assessment incomplete"

---

#### Feature 2: Measure Template Library

**User Story:**
```
As a risk analyst,
I want access to a curated library of 50+ proven mitigation measure templates,
So that I can select standardized, Article 11(2)-mapped measures with known effectiveness instead of designing measures from scratch.
```

**Acceptance Criteria:**
- [ ] Contains 50+ mitigation measure templates organized by Article 11(2) category: (a) 15+ templates, (b) 15+ templates, (c) 20+ templates
- [ ] Each template includes: measure_id, title, description, article_11_2_category (a/b/c), applicable_risk_dimensions (list), applicable_commodities (list of 7), applicable_country_risk_levels (LOW/STANDARD/HIGH), base_effectiveness_percent (expected risk reduction 0-100), typical_timeline_days, resource_requirements (cost range EUR, personnel), evidence_requirements (list of required evidence types), regulatory_references (list of EUDR article references), implementation_instructions (step-by-step guide)
- [ ] Supports filtering by: Article 11(2) category, risk dimension, commodity, country risk level
- [ ] Supports full-text search across template titles and descriptions
- [ ] Templates are configurable: operators can adjust base_effectiveness and timeline for their specific context
- [ ] Templates include pre-built configurations for all 7 EUDR commodities:
  - Cattle: 7+ commodity-specific templates (animal movement records, pasture verification, slaughterhouse audit)
  - Cocoa: 7+ templates (cooperative audit, farm-level GPS, child labor verification, Rainforest Alliance enrollment)
  - Coffee: 7+ templates (altitude/origin verification, wet mill inspection, export license validation)
  - Palm oil: 7+ templates (RSPO certification, mill audit, No Deforestation/No Peat/No Exploitation verification)
  - Rubber: 7+ templates (smallholder mapping, latex aggregation verification, GPSNR enrollment)
  - Soya: 7+ templates (farm boundary verification, silo segregation audit, Cerrado monitoring)
  - Wood: 7+ templates (FSC/PEFC certification, sawmill chain of custody, species verification, FLEGT license)
- [ ] Supports template versioning: updated templates do not retroactively change active strategies
- [ ] Provides template usage statistics: how many times each template has been used, average actual effectiveness achieved
- [ ] Supports custom template creation by operators (Admin role only)

**Non-Functional Requirements:**
- Lookup Performance: < 10ms for template filtering and retrieval
- Coverage: All 3 Article 11(2) categories represented with minimum 15 templates each
- Extensibility: New templates can be added without code deployment (database-driven)

**Dependencies:**
- PostgreSQL for template storage
- EUDR commodity classification data

**Estimated Effort:** 2 weeks (1 backend engineer, 1 regulatory domain expert)

**Edge Cases:**
- No template applicable to a specific risk dimension + commodity combination -- Return "no standard template; custom measure required" with guidance
- Template effectiveness is 0% for a given context -- Exclude template from strategy; log exclusion reason
- Operator creates custom template with unrealistic effectiveness (> 90%) -- Flag as "requires validation" in audit trail
- Template version is deprecated while active strategy uses it -- Active strategy retains original version; new strategies use updated version

---

#### Feature 3: Effectiveness Estimator

**User Story:**
```
As a compliance officer,
I want to estimate the expected risk reduction of proposed mitigation measures before committing resources,
So that I can confirm the measures will plausibly reduce risk to the negligible/low level required by Article 11(1).
```

**Acceptance Criteria:**
- [ ] Computes expected risk reduction for each proposed measure using the formula: Expected_Risk_Reduction = Base_Effectiveness * Applicability_Factor * Implementation_Quality_Factor
- [ ] Base_Effectiveness: from measure template (0-100 scale representing expected percentage risk reduction for the applicable dimension)
- [ ] Applicability_Factor: computed based on how well the measure matches the specific risk driver:
  - Exact match (measure targets the elevated dimension): 1.0
  - Related match (measure addresses a correlated dimension): 0.5-0.8
  - Partial match (measure has indirect impact): 0.2-0.4
- [ ] Implementation_Quality_Factor: configurable per operator based on their implementation capability:
  - HIGH: 1.0 (dedicated compliance team, experienced with EUDR)
  - STANDARD: 0.8 (moderate compliance capability)
  - LOW: 0.6 (limited compliance resources)
- [ ] Provides three estimate tiers: CONSERVATIVE (factors * 0.7), MODERATE (factors * 1.0), OPTIMISTIC (factors * 1.3, capped at base effectiveness)
- [ ] Computes aggregate risk reduction for the full strategy (all measures combined, accounting for diminishing returns): Aggregate_Reduction = 1 - PRODUCT(1 - R_i) where R_i is each measure's expected reduction as a fraction
- [ ] Validates feasibility: projects post-mitigation composite score and determines if it will fall below the NEGLIGIBLE/LOW threshold (< 30)
- [ ] Uses Decimal arithmetic throughout to prevent floating-point drift
- [ ] Generates a feasibility assessment: FEASIBLE (projected post-mitigation < 30), LIKELY_FEASIBLE (< 45), UNCERTAIN (< 60), UNLIKELY (>= 60)
- [ ] All calculations are deterministic and reproducible with SHA-256 provenance hash

**Non-Functional Requirements:**
- Performance: < 500ms per strategy effectiveness estimation
- Accuracy: Within +/- 10 points of actual risk reduction (validated over 100+ completed workflows)
- Reproducibility: Bit-perfect across runs

**Dependencies:**
- Measure Template Library (Feature 2) for base effectiveness values
- EUDR-028 Risk Assessment Engine (current composite score for projection)

**Estimated Effort:** 2 weeks (1 senior backend engineer)

**Edge Cases:**
- All proposed measures have 0 effectiveness for the elevated dimension -- Return UNLIKELY feasibility with recommendation to add more measures
- Strategy contains only 1 measure targeting a CRITICAL (> 80) score -- Likely insufficient; estimate shows score remains HIGH; recommend additional measures
- Implementation quality is LOW with moderate measures -- Honest estimation showing reduced expected reduction; flag risk of insufficient mitigation
- Aggregate reduction mathematically sufficient but depends on all measures succeeding -- Flag dependency risk; recommend contingency measures

---

#### Feature 4: Measure Implementation Tracker

**User Story:**
```
As a procurement manager,
I want to track the implementation status of each mitigation measure with milestones, evidence, and deadlines,
So that I can monitor progress, identify overdue measures, and ensure complete evidence collection for regulatory inspection.
```

**Acceptance Criteria:**
- [ ] Tracks each measure through 6 lifecycle states: PROPOSED -> APPROVED -> IN_PROGRESS -> COMPLETED -> VERIFIED -> CLOSED
- [ ] Records transitions between states with: timestamp, actor (user ID), justification/notes, evidence references
- [ ] Supports milestone tracking within each measure: define sub-tasks with due dates and completion status
- [ ] Records evidence uploads per measure: document IDs, photo references, audit report links, certificate copies, with file integrity verification (SHA-256 hash)
- [ ] Assigns responsible parties to each measure with role and contact information
- [ ] Sets deadlines per measure with configurable escalation rules: warning at 7 days before deadline, alert at deadline, escalation at 3 days overdue, critical alert at 7 days overdue
- [ ] Calculates implementation completeness percentage per measure (% of milestones completed) and per strategy (% of measures in COMPLETED or later state)
- [ ] Generates implementation progress reports: summary dashboard showing all active measures by state, completion percentages, overdue items, evidence gaps
- [ ] Supports bulk status updates for multiple measures
- [ ] Maintains immutable audit trail of all state transitions, evidence uploads, and deadline changes

**Non-Functional Requirements:**
- Real-time: State transitions reflected in queries within 1 second
- Alerting: Overdue alerts delivered within 1 hour of deadline breach (via OBS-004 Alerting Platform)
- Auditability: Complete, immutable lifecycle history for every measure

**Dependencies:**
- PostgreSQL for measure state storage
- S3 for evidence document storage
- OBS-004 Alerting Platform for deadline alerts
- SEC-001 JWT Authentication for actor identification

**Estimated Effort:** 3 weeks (1 backend engineer)

**Edge Cases:**
- Measure stuck in APPROVED for > 30 days without starting -- Generate escalation alert
- Evidence upload fails integrity check (hash mismatch) -- Reject upload with error; log incident
- Measure is COMPLETED but no evidence uploaded -- Block transition to VERIFIED; require minimum evidence
- Responsible party changed mid-implementation -- Record party change in audit trail; transfer all active milestones
- Measure cancelled after approval -- Transition to CLOSED with cancellation reason; record in audit trail

---

#### Feature 5: Risk Reduction Verifier

**User Story:**
```
As a compliance officer,
I want the system to automatically re-evaluate risk after mitigation measures are implemented,
So that I can verify that risk has been reduced to the negligible/low level required by Article 11(1) before placing products on the market.
```

**Acceptance Criteria:**
- [ ] Triggered when all measures in a strategy reach COMPLETED state (or when manually triggered by authorized user)
- [ ] Invokes EUDR-028 Risk Assessment Engine to compute a new composite risk assessment using updated data reflecting the mitigation measures taken
- [ ] Passes updated risk factor inputs to EUDR-028: updated supplier compliance data (post-audit), updated deforestation monitoring results, updated certification status, enhanced geolocation data, updated chain of custody records
- [ ] Compares pre-mitigation risk assessment with post-mitigation risk assessment at both composite and per-dimension levels
- [ ] Computes actual risk reduction: Pre_Score - Post_Score for composite and each dimension
- [ ] Compares actual risk reduction against estimated risk reduction (from Feature 3) and records the variance
- [ ] Determines Article 11(1) compliance: POST_RISK_LEVEL must be NEGLIGIBLE or LOW (composite score < 30) for market placement approval
- [ ] If risk remains HIGH or CRITICAL after mitigation:
  - Generates "INSUFFICIENT MITIGATION" finding
  - Identifies which dimensions remain elevated
  - Recommends additional measures (secondary mitigation cycle)
  - Updates workflow status to IN_PROGRESS with secondary mitigation phase
- [ ] If risk is successfully reduced to NEGLIGIBLE/LOW:
  - Generates "MITIGATION SUCCESSFUL" finding
  - Records final risk determination with Article 11(1) compliance confirmation
  - Transitions workflow to VERIFIED state
- [ ] All verification results stored with SHA-256 provenance hash linking pre and post assessments
- [ ] Records effectiveness feedback: feeds actual reduction data back to Effectiveness Estimator for future calibration

**Non-Functional Requirements:**
- Latency: Verification completes within 30 seconds (including EUDR-028 re-assessment)
- Accuracy: 100% correct Article 11(1) compliance determination
- Traceability: Complete link between pre-mitigation assessment, measures taken, and post-mitigation assessment

**Dependencies:**
- EUDR-028 Risk Assessment Engine (re-assessment invocation)
- Measure Implementation Tracker (Feature 4) for completion status
- Effectiveness Estimator (Feature 3) for estimated vs. actual comparison

**Estimated Effort:** 2 weeks (1 senior backend engineer)

**Edge Cases:**
- EUDR-028 is unavailable during verification -- Retry with exponential backoff; alert if unavailable > 1 hour
- Post-mitigation score is higher than pre-mitigation score (risk increased) -- Flag as anomaly; require investigation; do not approve market placement
- Partial mitigation: some dimensions reduced but others unchanged -- Report per-dimension results; overall compliance based on composite score
- Multiple verification attempts for same strategy -- Record each attempt; use most recent for compliance determination
- Country benchmarking changed between pre and post assessment -- Record benchmarking change; note impact on score comparison

---

**P0 Features 6-9: Workflow, Reporting, and Integration Layer**

> Features 6, 7, 8, and 9 are P0 launch blockers. Without workflow orchestration, reporting, approval gates, and audit logging, the core computation and lifecycle engine cannot deliver complete regulatory value. These features are the delivery mechanism through which compliance officers, procurement managers, auditors, and regulators interact with the mitigation engine.

---

#### Feature 6: Compliance Workflow Engine

**User Story:**
```
As a compliance officer,
I want the full Article 11 mitigation workflow orchestrated from risk trigger to closure,
So that I can manage the entire mitigation lifecycle without missing steps or losing audit trail integrity.
```

**Acceptance Criteria:**
- [ ] Automatically triggered when EUDR-028 classifies a risk assessment as HIGH or CRITICAL
- [ ] Manages the following workflow states: TRIGGERED -> STRATEGY_DESIGNED -> PENDING_APPROVAL -> APPROVED -> IMPLEMENTING -> VERIFICATION_PENDING -> VERIFIED -> CLOSED (or ESCALATED)
- [ ] TRIGGERED: receives risk assessment result from EUDR-028; invokes Mitigation Strategy Designer (Feature 1)
- [ ] STRATEGY_DESIGNED: strategy computed; transitions to PENDING_APPROVAL
- [ ] PENDING_APPROVAL: requires Compliance Officer role approval; records approval decision, justification, and timestamp
- [ ] APPROVED: measures released for implementation; transitions to IMPLEMENTING
- [ ] IMPLEMENTING: monitors measure implementation status (Feature 4); transitions to VERIFICATION_PENDING when all measures COMPLETED
- [ ] VERIFICATION_PENDING: invokes Risk Reduction Verifier (Feature 5); transitions to VERIFIED on success or back to IMPLEMENTING if additional measures needed
- [ ] VERIFIED: risk confirmed reduced to NEGLIGIBLE/LOW; generates Article 11 compliance report (Feature 7)
- [ ] CLOSED: workflow successfully completed; market placement approved; records closure timestamp
- [ ] ESCALATED: used when mitigation is insufficient after secondary cycle; escalates to senior management
- [ ] Supports parallel measure tracks: multiple measures can be in different lifecycle states simultaneously
- [ ] Records complete workflow state history with timestamps, actors, and justifications
- [ ] Supports manual workflow state overrides by Admin role with mandatory justification
- [ ] Generates workflow duration metrics for operational monitoring
- [ ] Supports batch workflows for multiple product-supplier combinations

**Non-Functional Requirements:**
- State Consistency: Workflow state transitions are atomic and consistent
- Concurrency: Supports 500+ concurrent active workflows
- Alerting: State transition notifications sent to relevant parties within 5 minutes

**Dependencies:**
- EUDR-028 Risk Assessment Engine (risk triggers, verification)
- Features 1-5 (strategy design, templates, estimation, tracking, verification)
- SEC-002 RBAC Authorization (approval gates)
- OBS-004 Alerting Platform (notifications)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- Workflow triggered for assessment that is already in an active workflow -- Detect duplicate; link to existing workflow; do not create duplicate
- Approval rejected -- Record rejection reason; allow strategy redesign or workflow cancellation
- Measure added to strategy after approval -- Require re-approval for modified strategy
- Workflow inactive for > 90 days -- Generate stale workflow alert; recommend review or closure
- Operator wants to proceed to market without completing mitigation -- Block: Article 11(1) prohibits market placement; record attempted bypass in audit trail

---

#### Feature 7: Mitigation Report Generator

**User Story:**
```
As a compliance officer,
I want to generate DDS-ready Article 11 compliance reports documenting all mitigation measures taken,
So that I can include complete mitigation documentation in my Due Diligence Statement and satisfy regulatory inspection requirements.
```

**Acceptance Criteria:**
- [ ] Generates mitigation reports containing: risk trigger summary (EUDR-028 assessment ID, composite score, risk level, risk decomposition), mitigation strategy designed (strategy ID, proposed measures with Article 11(2) category, selection rationale), effectiveness estimation results (expected risk reduction per measure and aggregate), implementation summary (measure status, timelines, evidence references), verification results (pre vs. post risk scores, per-dimension comparison, Article 11(1) determination), final risk determination (post-mitigation risk level, market placement approval status), regulatory references (Article 11(1), 11(2)(a-c) citations), provenance hashes (SHA-256 covering all inputs and outputs)
- [ ] Produces JSON output formatted for EU Information System DDS submission (Article 12 compliance)
- [ ] Produces PDF output with executive summary, detailed findings, evidence appendix, and methodology documentation
- [ ] Includes visual charts in PDF reports: risk score waterfall (showing reduction contribution of each measure), timeline chart (measure implementation over time), dimension comparison (pre vs. post by dimension)
- [ ] Validates report completeness: all required fields populated or flagged as missing
- [ ] Supports multi-language output: EN (default), FR, DE, ES, PT
- [ ] Supports batch report generation for multiple workflows
- [ ] Includes metadata: report ID, workflow ID, operator ID, commodity, generation timestamp
- [ ] Embeds provenance hash chain linking: EUDR-028 assessment -> strategy -> measures -> evidence -> verification -> report

**Non-Functional Requirements:**
- Generation Speed: < 5 seconds per report (JSON), < 15 seconds per report (PDF)
- Compliance: 100% of required DDS mitigation fields populated
- Schema Validation: 100% pass rate against EU Information System DDS specification

**Dependencies:**
- Features 1-6 (all computation and workflow features for report content)
- EU Information System DDS schema specification
- PDF generation library (WeasyPrint or ReportLab)
- GL-EUDR-APP DDS Reporting Engine for integration

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend engineer)

**Edge Cases:**
- Workflow closed without verification (manual override) -- Include override notation and justification in report
- Secondary mitigation cycle performed -- Include both primary and secondary mitigation documentation
- Report requested for in-progress workflow -- Generate interim report with "IN PROGRESS" watermark; note incomplete measures
- Multi-commodity strategy -- Generate separate reports per commodity with cross-references
- Report requested in unsupported language -- Fall back to English with warning

---

#### Feature 8: Approval Workflow with Role-Based Gates

**User Story:**
```
As a senior compliance officer,
I want to review and approve mitigation strategies before implementation begins,
So that I can ensure measures are adequate, proportionate, and cost-effective before committing resources.
```

**Acceptance Criteria:**
- [ ] Requires Compliance Officer or Admin role approval before any mitigation strategy transitions from PENDING_APPROVAL to APPROVED
- [ ] Approval request includes: strategy summary, proposed measures (with Article 11(2) category), effectiveness estimation, estimated cost, estimated timeline
- [ ] Approval decision records: approved/rejected, decision justification (minimum 30 characters), approver user ID, timestamp
- [ ] Rejected strategies can be redesigned: compliance officer can modify measure selection and resubmit
- [ ] Supports delegated approval: Compliance Officer can delegate approval authority to a named Risk Analyst for specific commodities
- [ ] Sends approval request notification to designated approvers via email and in-app alert
- [ ] Tracks approval response time: time from request to decision
- [ ] Supports multi-level approval for high-cost strategies (> EUR 100,000 estimated cost): requires both Compliance Officer and Director-level approval
- [ ] All approval decisions are immutable and included in the audit trail
- [ ] Generates approval summary report for management review

**Non-Functional Requirements:**
- Authorization: Only Compliance Officer, Director, and Admin roles can approve
- Notification: Approval requests delivered within 5 minutes
- Audit: Every approval/rejection recorded with immutable trail

**Dependencies:**
- SEC-001 JWT Authentication for approver identity
- SEC-002 RBAC Authorization for role verification
- SEC-005 Centralized Audit Logging for approval audit trail
- OBS-004 Alerting Platform for notifications

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Approval not received within 7 days -- Send reminder to approver and escalate to backup approver
- Approver is also the strategy designer -- Require secondary approver (separation of duties)
- Strategy modified after approval -- Invalidate previous approval; require re-approval
- Approval delegation expires -- Revert to primary approver; notify parties
- Emergency mitigation needed (product at port) -- Support "expedited approval" with post-hoc documentation requirement

---

#### Feature 9: Audit Trail and Institutional Learning

**User Story:**
```
As an external auditor,
I want a complete, immutable audit trail of all mitigation decisions, and
As a risk analyst,
I want historical data on which mitigation measures actually worked for which risk dimensions,
So that auditors can verify Article 11 compliance and analysts can improve future mitigation strategies.
```

**Acceptance Criteria:**
- [ ] Records every event in the mitigation lifecycle as an immutable audit log entry: workflow creation, strategy design, measure selection (with rationale), approval decisions, state transitions, evidence uploads, verification results, report generation, workflow closure/escalation
- [ ] Each audit entry includes: event_id, event_type, workflow_id, strategy_id, measure_id (if applicable), actor (user ID), timestamp, event_data (JSON), provenance_hash
- [ ] Supports audit trail query by: workflow_id, strategy_id, measure_id, actor, event_type, date range
- [ ] Tracks institutional learning data: for each completed workflow, records estimated_reduction vs. actual_reduction per measure, enabling effectiveness calibration
- [ ] Computes measure effectiveness statistics: average actual reduction per template, success rate (% of measures achieving > 80% of estimated reduction), failure patterns (which measures consistently underperform)
- [ ] Provides effectiveness leaderboard: templates ranked by actual average effectiveness per commodity and risk dimension
- [ ] Supports effectiveness data export for analysis
- [ ] Audit trail entries stored in TimescaleDB hypertable with 5-year retention per Article 31
- [ ] Audit trail is append-only; no deletion or modification of historical entries
- [ ] Supports external audit queries: read-only access for Auditor role with configurable data scope

**Non-Functional Requirements:**
- Immutability: Audit entries cannot be modified or deleted after creation
- Performance: Audit trail query < 200ms for 12-month date range
- Storage: Efficient time-series storage using TimescaleDB with compression

**Dependencies:**
- TimescaleDB for audit trail storage
- SEC-005 Centralized Audit Logging for integration
- Features 1-7 (event sources for audit trail)

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Audit trail storage approaches capacity -- TimescaleDB compression and retention policies handle automatically; alert when 80% capacity reached
- Audit query for workflow that spans multiple years -- Cross-chunk query handled by TimescaleDB
- Institutional learning data has < 10 completed workflows -- Return "insufficient data for statistical analysis"; use template defaults
- Effectiveness statistics show template consistently underperforms -- Flag template for review; recommend effectiveness reduction in template metadata

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: AI-Assisted Mitigation Recommendation
- Use historical effectiveness data to suggest optimal measure combinations for specific risk profiles
- Predict most cost-effective mitigation strategy based on past outcomes
- Recommend measures based on similar operator-commodity-country combinations
- Note: Recommendations only; all final decisions remain deterministic and human-approved

#### Feature 11: Supplier Self-Service Mitigation Portal
- Dedicated portal for suppliers to respond to Article 11(2)(a) information requests
- Self-service evidence upload with guided data collection forms
- Supplier-side progress tracking of their mitigation obligations
- Multi-language support for global supplier base

#### Feature 12: Cost-Benefit Analysis Engine
- Estimate total cost of mitigation strategy (audit fees, travel, monitoring, certification)
- Compare cost against potential penalty for non-compliance (4% of EU turnover)
- Generate ROI analysis for mitigation investment
- Track actual costs versus estimated costs for budget optimization

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Automated supplier payment for audit services (defer to procurement integration)
- Real-time chatbot for mitigation guidance (defer to v2.0)
- Mobile native mitigation tracking application (web-only for v1.0)
- Integration with non-EUDR regulations (CSDDD, CSRD) for mitigation synergies (defer to cross-regulation platform)
- Machine learning-based effectiveness prediction (deterministic formulas only for v1.0)
- Automated competent authority notification of mitigation completion (manual DDS submission for v1.0)

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
                  +---------------------------------+----------------------------------+
                  |                                                                    |
    +-------------v-------------+                                        +-------------v--------------+
    | AGENT-EUDR-029            |                                        | AGENT-EUDR-026             |
    | Mitigation Measure        |<-------------------------------------->| Due Diligence Orchestrator |
    | Designer                  |   (orchestrator invokes MMD for        |                            |
    |                           |    Phase 3: Risk Mitigation)           | - Workflow Engine          |
    | - Strategy Designer       |                                        | - Quality Gates            |
    | - Template Library        |                                        | - State Manager            |
    | - Effectiveness Estimator |                                        +----------------------------+
    | - Implementation Tracker  |
    | - Risk Reduction Verifier |<--------+
    | - Compliance Workflow     |         |
    | - Report Generator        |         |  (verification loop:
    +----+------+------+-------+         |   re-evaluate risk
         |      |      |                 |   after mitigation)
         |      |      |                 |
    +----v------v------v---------+  +----v---------------------+
    | AGENT-EUDR-028             |  | AGENT-EUDR-027           |
    | Risk Assessment Engine     |  | Information Gathering    |
    |                            |  | Agent                    |
    | - Composite Calculator     |  | - External DB Connector  |
    | - Factor Aggregator        |  | - Certification Validator|
    | - Art.10(2) Evaluator      |  | - Evidence Package Asm.  |
    | - Risk Classification      |  +-------------------------+
    +----+------+------+-------+
         |      |      |
    +----v-+ +--v--+ +-v----+------------------------------------------+
    |      | |     | |      |      |              |                    |
+---v--+ +-v--+ +-v--+ +---v--+ +-v-----------+ +v-----------------+  |
|EUDR  | |EUDR| |EUDR| |EUDR  | |EUDR         | |EUDR              |  |
| 016  | | 017| | 018| | 019  | | 020         | | 021              |  |
|Country| |Supp| |Comm| |Corr  | |Deforestation| |Indigenous Rights |  |
|Risk   | |Risk| |Risk| |Index | |Alert System | |Checker           |  |
|Eval.  | |Scor| |Anal| |Mon.  | |             | |(Art 10(2)(j))    |  |
+------+ +----+ +----+ +------+ +-------------+ +------------------+  |
                                                                       |
                                                           +-----------v-----------+
                                                           | PostgreSQL+TimescaleDB |
                                                           | - Mitigation strategies|
                                                           | - Measure templates    |
                                                           | - Implementation state |
                                                           | - Verification results |
                                                           | - Audit trail          |
                                                           +-----------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/mitigation_measure_designer/
    __init__.py                          # Public API exports
    config.py                            # MitigationMeasureDesignerConfig with GL_EUDR_MMD_ env prefix
    models.py                            # 12+ enums, 20+ Pydantic v2 models
    mitigation_strategy_designer.py      # Risk-driven mitigation strategy design (Feature 1)
    measure_template_library.py          # 50+ curated measure templates (Feature 2)
    effectiveness_estimator.py           # Deterministic risk reduction estimation (Feature 3)
    measure_implementation_tracker.py    # Measure lifecycle tracking (Feature 4)
    risk_reduction_verifier.py           # Post-mitigation risk re-evaluation (Feature 5)
    compliance_workflow_engine.py        # Full Article 11 workflow orchestration (Feature 6)
    mitigation_report_generator.py       # DDS-ready Article 11 reports (Feature 7)
    provenance.py                        # SHA-256 hash chains for all calculations and decisions
    metrics.py                           # 18 Prometheus metrics with gl_eudr_mmd_ prefix
    setup.py                             # MitigationMeasureDesignerService facade
    api.py                               # FastAPI routes (12+ endpoints)
```

### 7.3 Data Models (Key Entities)

```python
from enum import Enum
from decimal import Decimal
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


# Article 11(2) mitigation measure categories
class Article11Category(str, Enum):
    ADDITIONAL_INFO = "a"       # Art. 11(2)(a): additional information from suppliers
    INDEPENDENT_AUDIT = "b"     # Art. 11(2)(b): independent surveys and audits
    OTHER_ADAPTED = "c"         # Art. 11(2)(c): other measures adapted to complexity


# Risk dimensions that can drive mitigation (aligned with EUDR-028 RiskDimension)
class MitigationRiskDimension(str, Enum):
    COUNTRY = "country"
    COMMODITY = "commodity"
    SUPPLIER = "supplier"
    DEFORESTATION = "deforestation"
    CORRUPTION = "corruption"
    SUPPLY_CHAIN_COMPLEXITY = "supply_chain_complexity"
    MIXING = "mixing"
    CIRCUMVENTION = "circumvention"


# EUDR commodities
class EUDRCommodity(str, Enum):
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


# Measure lifecycle states
class MeasureStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    CLOSED = "closed"
    CANCELLED = "cancelled"


# Workflow lifecycle states
class WorkflowStatus(str, Enum):
    TRIGGERED = "triggered"
    STRATEGY_DESIGNED = "strategy_designed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    IMPLEMENTING = "implementing"
    VERIFICATION_PENDING = "verification_pending"
    VERIFIED = "verified"
    CLOSED = "closed"
    ESCALATED = "escalated"


# Feasibility assessment levels
class FeasibilityLevel(str, Enum):
    FEASIBLE = "feasible"                # Projected post-mitigation < 30
    LIKELY_FEASIBLE = "likely_feasible"  # Projected post-mitigation < 45
    UNCERTAIN = "uncertain"              # Projected post-mitigation < 60
    UNLIKELY = "unlikely"                # Projected post-mitigation >= 60


# Effectiveness estimate tier
class EstimateTier(str, Enum):
    CONSERVATIVE = "conservative"  # factors * 0.7
    MODERATE = "moderate"          # factors * 1.0
    OPTIMISTIC = "optimistic"      # factors * 1.3 (capped)


# Implementation quality level
class ImplementationQuality(str, Enum):
    HIGH = "high"          # Factor = 1.0
    STANDARD = "standard"  # Factor = 0.8
    LOW = "low"            # Factor = 0.6


# Approval decision
class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


# Mitigation measure template
class MeasureTemplate(BaseModel):
    template_id: str
    title: str
    description: str
    article_11_2_category: Article11Category
    applicable_risk_dimensions: List[MitigationRiskDimension]
    applicable_commodities: List[EUDRCommodity]
    applicable_country_risk_levels: List[str]       # ["low", "standard", "high"]
    base_effectiveness_percent: Decimal = Field(ge=0, le=100)
    typical_timeline_days: int
    resource_requirements: Dict[str, str]            # {"cost_range_eur": "5000-20000", "personnel": "1-2"}
    evidence_requirements: List[str]
    regulatory_references: List[str]
    implementation_instructions: str
    version: int = 1
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


# Proposed mitigation measure (instantiated from template for a specific strategy)
class MitigationMeasure(BaseModel):
    measure_id: str
    strategy_id: str
    template_id: str
    title: str
    article_11_2_category: Article11Category
    target_risk_dimension: MitigationRiskDimension
    status: MeasureStatus = MeasureStatus.PROPOSED
    responsible_party: Optional[str] = None
    deadline: Optional[datetime] = None
    milestones: List[Dict] = []                      # [{"title": str, "due_date": str, "completed": bool}]
    evidence_references: List[str] = []              # Document IDs, URLs
    estimated_risk_reduction: Optional[Decimal] = None
    actual_risk_reduction: Optional[Decimal] = None
    implementation_notes: str = ""
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


# Effectiveness estimation result
class EffectivenessEstimate(BaseModel):
    estimate_id: str
    strategy_id: str
    measure_id: str
    base_effectiveness: Decimal
    applicability_factor: Decimal
    implementation_quality_factor: Decimal
    conservative_reduction: Decimal
    moderate_reduction: Decimal
    optimistic_reduction: Decimal
    provenance_hash: str


# Strategy-level effectiveness projection
class StrategyEffectivenessProjection(BaseModel):
    strategy_id: str
    pre_mitigation_composite: Decimal
    projected_post_mitigation_composite: Decimal
    aggregate_risk_reduction: Decimal
    feasibility: FeasibilityLevel
    per_measure_estimates: List[EffectivenessEstimate]
    estimate_tier: EstimateTier
    provenance_hash: str


# Mitigation strategy (collection of measures for a risk assessment)
class MitigationStrategy(BaseModel):
    strategy_id: str
    assessment_id: str                               # EUDR-028 risk assessment that triggered mitigation
    operator_id: str
    commodity: EUDRCommodity
    supplier_id: Optional[str] = None
    country_codes: List[str]
    pre_mitigation_score: Decimal
    pre_mitigation_level: str                        # "high" or "critical"
    risk_drivers: List[Dict]                          # [{"dimension": str, "score": Decimal, "contribution": Decimal}]
    measures: List[MitigationMeasure]
    effectiveness_projection: Optional[StrategyEffectivenessProjection] = None
    article_11_2_coverage: Dict[str, int]            # {"a": 3, "b": 2, "c": 1} -- count per category
    estimated_total_timeline_days: int
    estimated_total_cost_eur: Optional[str] = None   # Range string, e.g., "15000-50000"
    created_at: datetime
    updated_at: datetime
    provenance_hash: str


# Verification result (pre vs. post mitigation comparison)
class VerificationResult(BaseModel):
    verification_id: str
    strategy_id: str
    workflow_id: str
    pre_assessment_id: str
    post_assessment_id: str
    pre_composite_score: Decimal
    post_composite_score: Decimal
    actual_composite_reduction: Decimal
    pre_dimension_scores: Dict[str, Decimal]
    post_dimension_scores: Dict[str, Decimal]
    per_dimension_reduction: Dict[str, Decimal]
    estimated_vs_actual_variance: Decimal
    article_11_1_compliant: bool                     # True if post_composite < 30
    post_risk_level: str
    recommendation: str                               # "market_placement_approved" or "additional_mitigation_required"
    verified_at: datetime
    provenance_hash: str


# Approval record
class ApprovalRecord(BaseModel):
    approval_id: str
    strategy_id: str
    workflow_id: str
    decision: ApprovalDecision
    justification: str = Field(min_length=30)
    approver_user_id: str
    approver_role: str
    approved_at: datetime
    cost_threshold_exceeded: bool = False             # True if multi-level approval needed
    secondary_approver_id: Optional[str] = None
    secondary_approved_at: Optional[datetime] = None


# Workflow state record
class MitigationWorkflow(BaseModel):
    workflow_id: str
    assessment_id: str
    strategy_id: Optional[str] = None
    operator_id: str
    commodity: EUDRCommodity
    status: WorkflowStatus
    current_mitigation_cycle: int = 1                 # 1 = primary, 2 = secondary, etc.
    trigger_risk_level: str
    trigger_composite_score: Decimal
    approval_record: Optional[ApprovalRecord] = None
    verification_result: Optional[VerificationResult] = None
    report_id: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_duration_days: Optional[int] = None
    provenance_hash: Optional[str] = None


# Audit trail event
class AuditEvent(BaseModel):
    event_id: str
    event_type: str                                   # "workflow_created", "strategy_designed", "measure_approved", etc.
    workflow_id: str
    strategy_id: Optional[str] = None
    measure_id: Optional[str] = None
    actor: str                                        # User ID
    timestamp: datetime
    event_data: Dict                                  # Event-specific payload
    provenance_hash: str


# Mitigation report
class MitigationReport(BaseModel):
    report_id: str
    workflow_id: str
    strategy_id: str
    assessment_id: str
    operator_id: str
    operator_name: str
    commodity: EUDRCommodity
    risk_trigger_summary: Dict                        # Pre-mitigation assessment summary
    strategy_summary: Dict                            # Measures designed with Article 11(2) mapping
    effectiveness_estimation: Dict                    # Expected reductions
    implementation_summary: Dict                      # Measure statuses and evidence
    verification_result: Optional[Dict] = None        # Pre vs. post comparison
    final_risk_determination: Dict                    # Post-mitigation risk level
    regulatory_references: List[str]
    report_language: str = "en"
    generated_at: datetime
    provenance_hash: str
    dds_format_version: str = "1.0"
```

### 7.4 Database Schema (New Migration: V117)

```sql
-- Migration: V117__agent_eudr_mitigation_measure_designer.sql
-- Agent: EUDR-029 Mitigation Measure Designer
-- Category: Due Diligence (Category 5)

CREATE SCHEMA IF NOT EXISTS eudr_mitigation_measure_designer;

-- Mitigation strategies linked to risk assessments
CREATE TABLE eudr_mitigation_measure_designer.mitigation_strategies (
    strategy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    supplier_id UUID,
    country_codes JSONB NOT NULL DEFAULT '[]',
    pre_mitigation_score NUMERIC(5,2) NOT NULL,
    pre_mitigation_level VARCHAR(20) NOT NULL,
    risk_drivers JSONB NOT NULL DEFAULT '[]',
    article_11_2_coverage JSONB DEFAULT '{}',
    estimated_total_timeline_days INTEGER,
    estimated_total_cost_eur VARCHAR(50),
    current_cycle INTEGER DEFAULT 1,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

-- Individual mitigation measures with lifecycle tracking
CREATE TABLE eudr_mitigation_measure_designer.mitigation_measures (
    measure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID NOT NULL REFERENCES eudr_mitigation_measure_designer.mitigation_strategies(strategy_id),
    template_id VARCHAR(100) NOT NULL,
    title VARCHAR(500) NOT NULL,
    article_11_2_category CHAR(1) NOT NULL,
    target_risk_dimension VARCHAR(40) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'proposed',
    responsible_party VARCHAR(200),
    deadline TIMESTAMPTZ,
    milestones JSONB DEFAULT '[]',
    evidence_references JSONB DEFAULT '[]',
    estimated_risk_reduction NUMERIC(5,2),
    actual_risk_reduction NUMERIC(5,2),
    implementation_notes TEXT DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Measure template library (50+ templates)
CREATE TABLE eudr_mitigation_measure_designer.measure_templates (
    template_id VARCHAR(100) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    article_11_2_category CHAR(1) NOT NULL,
    applicable_risk_dimensions JSONB NOT NULL DEFAULT '[]',
    applicable_commodities JSONB NOT NULL DEFAULT '[]',
    applicable_country_risk_levels JSONB DEFAULT '["low","standard","high"]',
    base_effectiveness_percent NUMERIC(5,2) NOT NULL,
    typical_timeline_days INTEGER NOT NULL,
    resource_requirements JSONB DEFAULT '{}',
    evidence_requirements JSONB DEFAULT '[]',
    regulatory_references JSONB DEFAULT '[]',
    implementation_instructions TEXT,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    usage_count INTEGER DEFAULT 0,
    avg_actual_effectiveness NUMERIC(5,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Evidence and document attachments for measures
CREATE TABLE eudr_mitigation_measure_designer.measure_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measure_id UUID NOT NULL REFERENCES eudr_mitigation_measure_designer.mitigation_measures(measure_id),
    evidence_type VARCHAR(50) NOT NULL,
    document_reference VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64),
    description TEXT,
    uploaded_by VARCHAR(100) NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Effectiveness estimation records
CREATE TABLE eudr_mitigation_measure_designer.effectiveness_estimates (
    estimate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID NOT NULL REFERENCES eudr_mitigation_measure_designer.mitigation_strategies(strategy_id),
    measure_id UUID REFERENCES eudr_mitigation_measure_designer.mitigation_measures(measure_id),
    base_effectiveness NUMERIC(5,2) NOT NULL,
    applicability_factor NUMERIC(4,3) NOT NULL,
    implementation_quality_factor NUMERIC(4,3) NOT NULL,
    conservative_reduction NUMERIC(5,2) NOT NULL,
    moderate_reduction NUMERIC(5,2) NOT NULL,
    optimistic_reduction NUMERIC(5,2) NOT NULL,
    aggregate_strategy_reduction NUMERIC(5,2),
    projected_post_score NUMERIC(5,2),
    feasibility VARCHAR(30),
    provenance_hash VARCHAR(64) NOT NULL,
    estimated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Post-mitigation verification results
CREATE TABLE eudr_mitigation_measure_designer.verification_results (
    verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID NOT NULL REFERENCES eudr_mitigation_measure_designer.mitigation_strategies(strategy_id),
    workflow_id UUID NOT NULL,
    pre_assessment_id UUID NOT NULL,
    post_assessment_id UUID NOT NULL,
    pre_composite_score NUMERIC(5,2) NOT NULL,
    post_composite_score NUMERIC(5,2) NOT NULL,
    actual_composite_reduction NUMERIC(5,2) NOT NULL,
    pre_dimension_scores JSONB NOT NULL DEFAULT '{}',
    post_dimension_scores JSONB NOT NULL DEFAULT '{}',
    per_dimension_reduction JSONB NOT NULL DEFAULT '{}',
    estimated_vs_actual_variance NUMERIC(5,2),
    article_11_1_compliant BOOLEAN NOT NULL,
    post_risk_level VARCHAR(20) NOT NULL,
    recommendation VARCHAR(50) NOT NULL,
    provenance_hash VARCHAR(64) NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Workflow state machine records
CREATE TABLE eudr_mitigation_measure_designer.workflow_states (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL,
    strategy_id UUID REFERENCES eudr_mitigation_measure_designer.mitigation_strategies(strategy_id),
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'triggered',
    current_mitigation_cycle INTEGER DEFAULT 1,
    trigger_risk_level VARCHAR(20) NOT NULL,
    trigger_composite_score NUMERIC(5,2) NOT NULL,
    approval_record JSONB,
    verification_id UUID REFERENCES eudr_mitigation_measure_designer.verification_results(verification_id),
    report_id UUID,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    total_duration_days INTEGER,
    provenance_hash VARCHAR(64),
    CONSTRAINT fk_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

-- Implementation milestone tracking
CREATE TABLE eudr_mitigation_measure_designer.implementation_milestones (
    milestone_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measure_id UUID NOT NULL REFERENCES eudr_mitigation_measure_designer.mitigation_measures(measure_id),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    due_date TIMESTAMPTZ,
    completed BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMPTZ,
    completed_by VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Immutable audit trail (hypertable for time-series)
CREATE TABLE eudr_mitigation_measure_designer.audit_log (
    event_id UUID DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    workflow_id UUID,
    strategy_id UUID,
    measure_id UUID,
    actor VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_mitigation_measure_designer.audit_log', 'recorded_at');

-- Indexes for efficient queries
CREATE INDEX idx_strategies_assessment ON eudr_mitigation_measure_designer.mitigation_strategies(assessment_id);
CREATE INDEX idx_strategies_operator ON eudr_mitigation_measure_designer.mitigation_strategies(operator_id);
CREATE INDEX idx_strategies_commodity ON eudr_mitigation_measure_designer.mitigation_strategies(commodity);
CREATE INDEX idx_strategies_level ON eudr_mitigation_measure_designer.mitigation_strategies(pre_mitigation_level);
CREATE INDEX idx_strategies_created ON eudr_mitigation_measure_designer.mitigation_strategies(created_at DESC);

CREATE INDEX idx_measures_strategy ON eudr_mitigation_measure_designer.mitigation_measures(strategy_id);
CREATE INDEX idx_measures_status ON eudr_mitigation_measure_designer.mitigation_measures(status);
CREATE INDEX idx_measures_template ON eudr_mitigation_measure_designer.mitigation_measures(template_id);
CREATE INDEX idx_measures_dimension ON eudr_mitigation_measure_designer.mitigation_measures(target_risk_dimension);
CREATE INDEX idx_measures_deadline ON eudr_mitigation_measure_designer.mitigation_measures(deadline);

CREATE INDEX idx_templates_category ON eudr_mitigation_measure_designer.measure_templates(article_11_2_category);
CREATE INDEX idx_templates_active ON eudr_mitigation_measure_designer.measure_templates(is_active);

CREATE INDEX idx_evidence_measure ON eudr_mitigation_measure_designer.measure_evidence(measure_id);
CREATE INDEX idx_estimates_strategy ON eudr_mitigation_measure_designer.effectiveness_estimates(strategy_id);
CREATE INDEX idx_verification_strategy ON eudr_mitigation_measure_designer.verification_results(strategy_id);
CREATE INDEX idx_verification_workflow ON eudr_mitigation_measure_designer.verification_results(workflow_id);

CREATE INDEX idx_workflows_assessment ON eudr_mitigation_measure_designer.workflow_states(assessment_id);
CREATE INDEX idx_workflows_operator ON eudr_mitigation_measure_designer.workflow_states(operator_id);
CREATE INDEX idx_workflows_status ON eudr_mitigation_measure_designer.workflow_states(status);
CREATE INDEX idx_workflows_started ON eudr_mitigation_measure_designer.workflow_states(started_at DESC);

CREATE INDEX idx_milestones_measure ON eudr_mitigation_measure_designer.implementation_milestones(measure_id);
CREATE INDEX idx_milestones_due ON eudr_mitigation_measure_designer.implementation_milestones(due_date);

CREATE INDEX idx_audit_workflow ON eudr_mitigation_measure_designer.audit_log(workflow_id);
CREATE INDEX idx_audit_strategy ON eudr_mitigation_measure_designer.audit_log(strategy_id);
CREATE INDEX idx_audit_type ON eudr_mitigation_measure_designer.audit_log(event_type);
CREATE INDEX idx_audit_actor ON eudr_mitigation_measure_designer.audit_log(actor);

-- Retention policy for audit log hypertable
SELECT add_retention_policy('eudr_mitigation_measure_designer.audit_log', INTERVAL '5 years');
```

### 7.5 API Endpoints (12+)

| Method | Path | Description |
|--------|------|-------------|
| **Strategy Design** | | |
| POST | `/v1/eudr/mitigation-measure-designer/design-strategy` | Design mitigation strategy from risk assessment result |
| GET | `/v1/eudr/mitigation-measure-designer/strategies/{strategy_id}` | Get strategy details with measures and effectiveness projection |
| GET | `/v1/eudr/mitigation-measure-designer/strategies` | List strategies with filters (operator, commodity, status, date range) |
| **Measure Lifecycle** | | |
| POST | `/v1/eudr/mitigation-measure-designer/measures/{measure_id}/approve` | Approve a proposed measure (Compliance Officer role) |
| POST | `/v1/eudr/mitigation-measure-designer/measures/{measure_id}/start` | Start measure implementation |
| POST | `/v1/eudr/mitigation-measure-designer/measures/{measure_id}/complete` | Mark measure as completed with evidence |
| **Verification** | | |
| POST | `/v1/eudr/mitigation-measure-designer/verify/{strategy_id}` | Verify risk reduction after mitigation (invoke EUDR-028 re-assessment) |
| **Templates** | | |
| GET | `/v1/eudr/mitigation-measure-designer/templates` | List available measure templates (filter by category, dimension, commodity) |
| GET | `/v1/eudr/mitigation-measure-designer/templates/{template_id}` | Get template details |
| **Reports** | | |
| POST | `/v1/eudr/mitigation-measure-designer/generate-report/{strategy_id}` | Generate DDS-ready Article 11 mitigation report |
| **Workflows** | | |
| GET | `/v1/eudr/mitigation-measure-designer/workflows/{workflow_id}/status` | Get workflow status with full state history |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_mmd_strategies_designed_total` | Counter | Mitigation strategies created, labeled by commodity and pre_mitigation_level |
| 2 | `gl_eudr_mmd_measures_proposed_total` | Counter | Measures proposed, labeled by article_11_2_category and risk_dimension |
| 3 | `gl_eudr_mmd_measures_approved_total` | Counter | Measures approved, labeled by article_11_2_category |
| 4 | `gl_eudr_mmd_measures_completed_total` | Counter | Measures completed, labeled by article_11_2_category and risk_dimension |
| 5 | `gl_eudr_mmd_verifications_total` | Counter | Risk reduction verifications, labeled by result (compliant/non_compliant) |
| 6 | `gl_eudr_mmd_reports_generated_total` | Counter | Mitigation reports generated, labeled by format (json/pdf) |
| 7 | `gl_eudr_mmd_workflows_closed_total` | Counter | Workflows successfully closed, labeled by commodity |
| 8 | `gl_eudr_mmd_api_errors_total` | Counter | API errors, labeled by endpoint and error_type |
| 9 | `gl_eudr_mmd_strategy_design_duration_seconds` | Histogram | Strategy design computation latency |
| 10 | `gl_eudr_mmd_effectiveness_estimation_duration_seconds` | Histogram | Effectiveness estimation latency |
| 11 | `gl_eudr_mmd_verification_duration_seconds` | Histogram | Risk reduction verification latency (including EUDR-028 call) |
| 12 | `gl_eudr_mmd_report_generation_duration_seconds` | Histogram | Report generation latency, labeled by format |
| 13 | `gl_eudr_mmd_workflow_duration_seconds` | Histogram | End-to-end workflow duration (trigger to closure) |
| 14 | `gl_eudr_mmd_active_workflows` | Gauge | Currently active mitigation workflows |
| 15 | `gl_eudr_mmd_overdue_measures` | Gauge | Measures past their deadline |
| 16 | `gl_eudr_mmd_average_risk_reduction` | Gauge | Rolling average actual risk reduction across completed workflows |
| 17 | `gl_eudr_mmd_pending_approvals` | Gauge | Measures awaiting Compliance Officer approval |
| 18 | `gl_eudr_mmd_template_library_size` | Gauge | Number of active templates in the library |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL 16 + TimescaleDB | Persistent storage + time-series hypertables for audit trail |
| Arithmetic | Python `decimal.Decimal` | Exact decimal arithmetic for effectiveness estimation; no floating-point drift |
| Data Models | Pydantic v2 | Type-safe, validated, JSON-compatible models |
| Hashing | hashlib SHA-256 | Provenance hash chains for audit trail and report integrity |
| Cache | Redis | Template caching, active workflow state caching |
| Object Storage | S3 | Evidence document storage, PDF report storage |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access control for approvals and overrides |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across EUDR-028 verification calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |
| PDF Generation | WeasyPrint | HTML-to-PDF for DDS-ready Article 11 reports |

### 7.8 RBAC Permissions (SEC-002 Integration)

The following permissions will be registered in the GreenLang PERMISSION_MAP for RBAC enforcement:

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-mmd:strategies:read` | View mitigation strategies and measures | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-mmd:strategies:write` | Create and modify mitigation strategies | Analyst, Compliance Officer, Admin |
| `eudr-mmd:measures:read` | View individual measure details and status | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-mmd:measures:write` | Update measure status, add evidence | Analyst, Compliance Officer, Procurement Manager, Admin |
| `eudr-mmd:measures:approve` | Approve proposed measures for implementation | Compliance Officer, Admin |
| `eudr-mmd:templates:read` | View measure template library | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-mmd:templates:write` | Create and modify measure templates | Compliance Officer, Admin |
| `eudr-mmd:verification:read` | View verification results | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-mmd:verification:execute` | Trigger risk reduction verification | Compliance Officer, Admin |
| `eudr-mmd:reports:read` | Download generated mitigation reports | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-mmd:reports:generate` | Generate DDS-ready mitigation reports | Compliance Officer, Admin |
| `eudr-mmd:workflows:read` | View workflow status and history | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-mmd:workflows:manage` | Override workflow states, escalate/close | Compliance Officer, Admin |
| `eudr-mmd:overrides:write` | Override workflow state with justification | Admin |
| `eudr-mmd:audit:read` | View full audit trail and provenance hashes | Auditor (read-only), Compliance Officer, Admin |
| `eudr-mmd:effectiveness:read` | View institutional learning effectiveness data | Analyst, Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| EUDR-028 Risk Assessment Engine | API / database | Risk assessment results (composite score, decomposition, risk level, Article 10(2) criteria) -> Mitigation Strategy Designer trigger input |
| EUDR-028 Risk Assessment Engine | API invocation | Risk re-assessment after mitigation -> Risk Reduction Verifier consumes post-mitigation assessment |
| EUDR-016 Country Risk Evaluator | Database | Country risk factors -> Strategy Designer uses for country-specific measure selection |
| EUDR-017 Supplier Risk Scorer | Database | Supplier risk factors -> Strategy Designer uses for supplier-specific measure selection |
| EUDR-018 Commodity Risk Analyzer | Database | Commodity risk factors -> Strategy Designer uses for commodity-specific measure selection |
| EUDR-019 Corruption Index Monitor | Database | Corruption risk factors -> Strategy Designer uses for corruption-specific measure selection |
| EUDR-020 Deforestation Alert System | Database | Deforestation alerts -> Strategy Designer uses for deforestation-specific measure selection |
| EUDR-027 Information Gathering Agent | API / database | Additional data collection for Article 11(2)(a) measures -> Evidence package integration |
| AGENT-FOUND-008 Reproducibility Agent | Verification | Bit-perfect verification of effectiveness estimation calculations |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| EUDR-026 Due Diligence Orchestrator | API invocation | Orchestrator calls MMD for Phase 3 risk mitigation; receives workflow status and completion confirmation |
| GL-EUDR-APP v1.0 Platform | API integration | Strategy data, measure status, workflow progress, reports -> frontend mitigation dashboard |
| DDS Reporting (Article 12) | Report integration | Mitigation report formatted for DDS submission -> Article 11 section of DDS |
| External Auditors | Read-only API + PDF reports | Mitigation reports, audit trails, and evidence packages for third-party verification |
| EUDR-030+ Future Agents | API / database | Mitigation outcomes and institutional learning data for future agent enhancements |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: End-to-End Mitigation Workflow (Compliance Officer)

```
1. EUDR-028 classifies risk assessment as HIGH (composite: 72.5) for cocoa from Ghana
   -> System automatically triggers EUDR-029 mitigation workflow
   -> Workflow status: TRIGGERED
2. Mitigation Strategy Designer analyzes risk decomposition:
   - Country risk: 65 (driver)
   - Deforestation risk: 78 (driver)
   - Supplier risk: 55 (moderate)
   - Corruption risk: 60 (borderline driver)
3. System designs targeted strategy:
   - Measure 1: Request country-specific origin documentation [Art 11(2)(a)]
   - Measure 2: Commission independent field audit of production plots [Art 11(2)(b)]
   - Measure 3: Intensify satellite monitoring of supplier plots [Art 11(2)(c)]
   - Measure 4: Request updated anti-corruption compliance certificates [Art 11(2)(a)]
   -> Workflow status: STRATEGY_DESIGNED
4. Effectiveness Estimator projects:
   - Aggregate expected reduction: 35 points (moderate estimate)
   - Projected post-mitigation score: 37.5 -> STANDARD risk
   - Feasibility: LIKELY_FEASIBLE
   -> Compliance officer reviews and requests additional measure
5. Compliance officer adds Measure 5: Require supplier FSC certification enrollment [Art 11(2)(c)]
   - Updated projection: 42 points reduction -> 30.5 -> STANDARD (borderline)
   - Feasibility updated: LIKELY_FEASIBLE
6. Compliance officer submits strategy for approval
   -> Workflow status: PENDING_APPROVAL
7. Director reviews and approves strategy
   -> Workflow status: APPROVED -> IMPLEMENTING
8. Procurement manager implements measures over 25 days:
   - Measure 1: Completed (documentation received, uploaded)
   - Measure 2: Completed (audit report received, uploaded)
   - Measure 3: Completed (satellite subscription activated)
   - Measure 4: Completed (certificates received)
   - Measure 5: Completed (supplier enrolled in FSC)
9. All measures reach COMPLETED
   -> Workflow status: VERIFICATION_PENDING
10. Risk Reduction Verifier invokes EUDR-028 with updated data:
    - Post-mitigation composite: 28.3 -> LOW risk
    - Actual reduction: 44.2 points (better than estimated 42)
    - Article 11(1) compliant: YES
    -> Workflow status: VERIFIED
11. System generates DDS-ready Article 11 compliance report
12. Compliance officer reviews report and closes workflow
    -> Workflow status: CLOSED
    -> Market placement approved
```

#### Flow 2: Insufficient Mitigation (Secondary Cycle)

```
1. EUDR-028 classifies rubber import from Indonesia as CRITICAL (composite: 85.2)
2. Strategy designed with 6 measures targeting country, deforestation, and corruption risk
3. Effectiveness Estimator projects: reduction 48 points -> projected 37.2 -> STANDARD
   Feasibility: UNCERTAIN
4. Compliance officer acknowledges risk and approves strategy
5. All 6 measures implemented over 40 days
6. Risk Reduction Verifier invokes EUDR-028:
   - Post-mitigation composite: 42.8 -> STANDARD (reduced but still above threshold)
   - Article 11(1) compliant: NO (score > 30)
   - Recommendation: ADDITIONAL_MITIGATION_REQUIRED
7. System identifies remaining elevated dimensions:
   - Deforestation risk still 58 (reduced from 90 but still elevated)
8. System designs secondary mitigation strategy:
   - Measure 7: Commission independent satellite analysis of all plots [Art 11(2)(b)]
   - Measure 8: Require supplier switch to identity-preserved chain of custody [Art 11(2)(c)]
9. Secondary cycle: approval -> implementation -> verification
   - Post-secondary mitigation composite: 26.1 -> LOW
   - Article 11(1) compliant: YES
10. Workflow closed after secondary cycle; report documents both cycles
```

#### Flow 3: Procurement Manager Implementation (Supplier-Facing)

```
1. Procurement manager receives notification: "3 measures assigned for implementation"
2. Opens mitigation dashboard -> sees measures with deadlines:
   - Measure 1: "Request origin documentation from Supplier X" -- Due in 14 days
   - Measure 2: "Request GPS coordinates for plots 12-15" -- Due in 21 days
   - Measure 3: "Request FSC certificate copy" -- Due in 28 days
3. Sends requests to suppliers via email (using system-generated templates)
4. As responses arrive:
   - Uploads origin documentation -> evidence attached to Measure 1 -> marks COMPLETED
   - Uploads GPS data -> evidence attached to Measure 2 -> marks COMPLETED
5. Measure 3: supplier has not responded at 25 days -> system sends overdue alert
6. Procurement manager follows up; certificate arrives at day 30 (2 days late)
   - Uploads certificate -> marks COMPLETED -> system records late completion
7. All supplier-facing measures completed; system notifies compliance officer
```

### 8.2 Key Screen Descriptions

**Mitigation Workflow Dashboard:**
- Summary cards: Active Workflows, Pending Approvals, Overdue Measures, Completed This Month
- Workflow list: table showing all active workflows with status, commodity, risk level, days active, progress %
- Status distribution chart: pie chart showing workflows by status
- Risk reduction trend: line chart showing average risk reduction over time

**Strategy Design View:**
- Risk decomposition panel: horizontal bar chart showing per-dimension scores with threshold line at 60
- Driver identification: highlighted dimensions above threshold with contribution percentage
- Proposed measures list: table with template title, Article 11(2) category, target dimension, expected reduction
- Effectiveness projection panel: waterfall chart showing expected reduction from each measure
- Feasibility indicator: color-coded badge (green/yellow/orange/red)
- Action buttons: "Add Measure", "Remove Measure", "Submit for Approval"

**Implementation Tracking View:**
- Timeline Gantt chart: measures on vertical axis, time on horizontal axis, color-coded by status
- Evidence panel: per-measure list of uploaded evidence with integrity status
- Milestone tracker: checklist per measure showing sub-task completion
- Overdue alerts: highlighted measures past deadline with escalation status
- Progress summary: strategy-level and per-measure completion percentages

**Verification Results View:**
- Score comparison panel: side-by-side pre vs. post risk assessment
- Dimension waterfall: chart showing score reduction per dimension
- Article 11(1) compliance badge: large green (COMPLIANT) or red (NON-COMPLIANT) indicator
- Estimated vs. actual comparison: table showing each measure's expected and actual effectiveness
- Recommendation panel: next steps (market placement approved or additional mitigation required)

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Mitigation Strategy Designer -- risk decomposition analysis, decision tree measure selection, multi-driver strategies
  - [ ] Feature 2: Measure Template Library -- 50+ templates across all 3 Article 11(2) categories and all 7 commodities
  - [ ] Feature 3: Effectiveness Estimator -- deterministic formula, three-tier estimation, feasibility assessment
  - [ ] Feature 4: Measure Implementation Tracker -- 6-state lifecycle, milestones, evidence, deadline alerts
  - [ ] Feature 5: Risk Reduction Verifier -- EUDR-028 re-assessment integration, Article 11(1) compliance determination
  - [ ] Feature 6: Compliance Workflow Engine -- full state machine, approval gates, parallel tracks
  - [ ] Feature 7: Mitigation Report Generator -- JSON and PDF DDS-ready output with provenance hashes
  - [ ] Feature 8: Approval Workflow -- role-based gates, multi-level approval, separation of duties
  - [ ] Feature 9: Audit Trail and Institutional Learning -- immutable trail, effectiveness statistics
- [ ] >= 85% test coverage achieved (line coverage >= 85%, branch coverage >= 90%)
- [ ] Security audit passed (JWT + RBAC integrated)
- [ ] Performance targets met (< 2 seconds strategy design, < 500ms estimation, < 5 seconds report)
- [ ] Effectiveness estimation validated against 100+ golden test scenarios
- [ ] Bit-perfect reproducibility verified across environments
- [ ] All 7 commodity types tested with representative mitigation scenarios
- [ ] Integration with EUDR-028 verified end-to-end (trigger -> mitigation -> verification)
- [ ] Integration with EUDR-026 orchestrator verified (Phase 3 invocation)
- [ ] API documentation complete (OpenAPI spec)
- [ ] Database migration V117 tested and validated
- [ ] Grafana dashboard operational with all 18 metrics
- [ ] RBAC permissions registered and tested (16 permissions)
- [ ] 50+ measure templates loaded and validated
- [ ] 5 beta customers successfully completed mitigation workflows
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ mitigation workflows triggered by customers
- Average strategy design latency < 2 seconds
- 100% of measures have tracked lifecycle status
- Overdue measure detection within 1 hour of deadline breach
- < 5 support tickets per customer related to mitigation management
- Zero failed verification loops due to EUDR-028 integration issues

**60 Days:**
- 200+ mitigation workflows completed (CLOSED status)
- Average workflow duration < 30 days
- Average risk reduction > 30 points for HIGH risk assessments
- Effectiveness estimation accuracy within +/- 10 points of actual reduction (over 50+ completed workflows)
- 50+ unique measure templates used by customers
- 100+ DDS-ready mitigation reports generated
- NPS > 40 from compliance officer persona

**90 Days:**
- 500+ mitigation workflows processed
- < 10% of workflows require secondary mitigation cycle (indicating strong initial strategy design)
- Institutional learning data covering all 7 commodities
- Zero EUDR penalties attributed to inadequate Article 11 mitigation for active customers
- Effectiveness estimation continuously calibrated with actual outcome data
- Average workflow duration trending below 25 days
- NPS > 50

---

## 10. Implementation Plan

### 10.1 Milestones

#### Phase 1: Core Computation Engine (Weeks 1-4)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1 | Config, models, and provenance module; Measure Template Library (Feature 2) with 50+ templates loaded | Backend Engineer |
| 2 | Mitigation Strategy Designer (Feature 1) with decision tree logic and multi-driver support | Senior Backend Engineer |
| 3 | Effectiveness Estimator (Feature 3) with Decimal arithmetic and three-tier estimation | Senior Backend Engineer |
| 4 | Measure Implementation Tracker (Feature 4) with 6-state lifecycle and milestone tracking | Backend Engineer |

**Milestone: Core computation and lifecycle engine operational (Week 4)**

#### Phase 2: Verification, Workflow, and Reporting (Weeks 5-8)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 5 | Risk Reduction Verifier (Feature 5) with EUDR-028 integration | Senior Backend Engineer |
| 6 | Compliance Workflow Engine (Feature 6) with full state machine and approval gates | Senior Backend Engineer |
| 7 | Approval Workflow (Feature 8) with role-based gates; Audit Trail (Feature 9) with institutional learning | Backend Engineer |
| 8 | Mitigation Report Generator (Feature 7) -- JSON and PDF output; REST API Layer: 12+ endpoints | Backend Engineer + Frontend Engineer |

**Milestone: Full API operational with workflow, verification, reporting, and approvals (Week 8)**

#### Phase 3: Integration and Testing (Weeks 9-12)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 9 | Integration with EUDR-028 (trigger and verification loop) and EUDR-026 (orchestrator Phase 3) | Integration Engineer |
| 10 | Integration with upstream risk agents (EUDR-016 through EUDR-020) for strategy design context | Integration Engineer |
| 11 | Complete test suite: 200+ tests, golden tests for all 7 commodities, performance tests | Test Engineer |
| 12 | Database migration V117 finalized; Grafana dashboard; RBAC integration; auth integration | DevOps + Backend |

**Milestone: All 9 P0 features integrated, tested, and production-ready (Week 12)**

#### Phase 4: Launch (Weeks 13-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 13 | Performance testing, security audit, load testing (500+ concurrent workflows) | DevOps + Security |
| 14 | Beta customer onboarding (5 customers), launch readiness review, go-live | Product + Engineering |

**Milestone: Production launch with all 9 P0 features (Week 14)**

### 10.2 Dependencies

#### Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EUDR-028 Risk Assessment Engine | BUILT (100%) | Low | Stable, production-ready; provides risk triggers and re-assessment |
| EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Stable; provides country risk factors for strategy design |
| EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Stable; provides supplier risk factors for strategy design |
| EUDR-018 Commodity Risk Analyzer | BUILT (100%) | Low | Stable; provides commodity risk factors for strategy design |
| EUDR-019 Corruption Index Monitor | BUILT (100%) | Low | Stable; provides corruption risk factors for strategy design |
| EUDR-020 Deforestation Alert System | BUILT (100%) | Low | Stable; provides deforestation risk factors for strategy design |
| EUDR-026 Due Diligence Orchestrator | BUILT (100%) | Low | Invokes MMD for Phase 3 orchestration |
| EUDR-027 Information Gathering Agent | BUILT (100%) | Low | Provides additional data for Article 11(2)(a) measures |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration for approval gates |
| SEC-005 Centralized Audit Logging | BUILT (100%) | Low | Audit trail integration |
| OBS-004 Alerting Platform | BUILT (100%) | Low | Overdue measure alerts and approval notifications |

#### External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU Information System DDS schema (Article 11 section) | Published (v1.x) | Medium | Adapter pattern for schema version changes |
| EC Article 29 country benchmarking list | Published; updated quarterly | Medium | Consumed via EUDR-028; strategy auto-adjusts to benchmarking changes |
| Third-party audit provider APIs | Variable | Medium | Manual evidence upload as fallback; no hard dependency on provider APIs |
| Certification body databases (FSC, RSPO) | Available | Low | Evidence validated manually if API unavailable |

### 10.3 Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EC publishes specific Article 11 implementing guidance that changes measure requirements | Medium | High | Modular template library with database-driven templates; new templates added without code deployment; configurable measure categories |
| R2 | Operators find effectiveness estimation insufficiently accurate | Medium | Medium | Conservative estimation by default; continuous calibration from institutional learning data; clearly communicated as estimation (not guarantee) |
| R3 | EUDR-028 re-assessment after mitigation takes too long for time-critical product flows | Low | High | Async verification with notification; fast-path for straightforward assessments; cached recent assessments |
| R4 | Approval workflows create bottlenecks (approvers not responsive) | Medium | Medium | Escalation rules (7-day reminder, 14-day auto-escalation); delegated approval; expedited approval for emergencies |
| R5 | Template library does not cover specific niche mitigation scenarios | Medium | Medium | Custom template creation capability; regular library updates based on customer feedback; "other adapted measures" category (c) provides flexibility |
| R6 | Suppliers do not respond to Article 11(2)(a) information requests | High | High | Overdue tracking with escalation; alternative measure recommendation (switch to audit-based approach); documented non-response for regulatory record |
| R7 | Secondary mitigation cycle required too frequently (indicating poor initial strategy design) | Medium | Medium | Institutional learning feedback loop; strategy design improvement based on outcome data; conservative initial estimation to set expectations |
| R8 | Performance degradation with many concurrent active workflows | Low | Medium | Horizontal scaling on K8s; Redis caching for active workflows; database query optimization with indexes |
| R9 | Competent authority challenges mitigation measure adequacy | Medium | High | Every measure mapped to Article 11(2) category; full provenance trail; deterministic methodology documentation; regulatory references in reports |
| R10 | Evidence document storage grows unbounded | Medium | Low | S3 with lifecycle policies; TimescaleDB compression for audit trail; retention policies enforced per Article 31 (5 years) |

---

## 11. Test Strategy

### 11.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Strategy Designer Tests | 60+ | Risk decomposition analysis, decision tree logic, multi-driver strategies, edge cases |
| Template Library Tests | 30+ | Template CRUD, filtering, search, versioning, commodity-specific templates |
| Effectiveness Estimator Tests | 50+ | Formula accuracy, Decimal arithmetic, three-tier estimation, feasibility assessment, golden values |
| Implementation Tracker Tests | 40+ | 6-state lifecycle transitions, milestones, evidence, deadlines, overdue detection |
| Risk Reduction Verifier Tests | 30+ | EUDR-028 integration, pre vs. post comparison, Article 11(1) compliance, secondary cycle trigger |
| Workflow Engine Tests | 40+ | State machine transitions, approval gates, parallel tracks, escalation, batch workflows |
| Report Generator Tests | 30+ | JSON/PDF output, DDS schema validation, multi-language, provenance hashes |
| Approval Workflow Tests | 20+ | Role-based gates, multi-level approval, delegation, separation of duties |
| Audit Trail Tests | 20+ | Immutability, institutional learning statistics, time-series queries |
| API Tests | 40+ | All 12+ endpoints, auth, error handling, pagination |
| Integration Tests | 20+ | Cross-agent integration with EUDR-028, EUDR-026, upstream risk agents |
| Performance Tests | 15+ | Strategy design latency, estimation latency, concurrent workflows, report generation |
| Golden Tests | 35+ | All 7 commodities with complete mitigation workflows (LOW/HIGH/CRITICAL scenarios) |
| **Total** | **430+** | |

### 11.2 Golden Test Design

Each of the 7 EUDR commodities will have dedicated golden test scenarios:

1. **HIGH risk single-driver scenario** -- One dimension elevated; targeted single-driver strategy -> expect focused mitigation, successful reduction in 1 cycle
2. **HIGH risk multi-driver scenario** -- 3+ dimensions elevated; multi-driver strategy -> expect comprehensive mitigation, successful reduction in 1 cycle
3. **CRITICAL risk scenario** -- Composite > 80; intensive measures required -> expect multi-measure strategy, may require secondary cycle
4. **Insufficient mitigation scenario** -- Measures implemented but risk remains HIGH -> expect secondary cycle trigger, additional measures recommended
5. **Missing data scenario** -- Risk assessment has incomplete dimensions -> expect strategy designed for available dimensions with "incomplete" flags

Total: 7 commodities x 5 scenarios = 35 core golden test scenarios

### 11.3 Reproducibility Testing

Every golden test will be executed 100 times across:
- Different Python interpreter sessions
- Different operating systems (Linux, macOS, Windows)
- Different hardware architectures (x86_64, ARM64)

All executions must produce bit-identical SHA-256 provenance hashes for strategy design and effectiveness estimation, verifying deterministic behavior.

---

## 12. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **Article 11** | EUDR article requiring risk mitigation measures when non-negligible risk is identified |
| **Article 11(2)(a)** | Mitigation category: requiring additional information, data, or documents from suppliers |
| **Article 11(2)(b)** | Mitigation category: carrying out independent surveys and audits |
| **Article 11(2)(c)** | Mitigation category: other measures adapted to supply chain complexity |
| **Risk Decomposition** | Breakdown of composite risk score into per-dimension contributions |
| **Risk Driver** | A risk dimension scoring above the elevated threshold (> 60) that requires targeted mitigation |
| **Mitigation Strategy** | Coordinated set of measures designed to reduce risk for a specific risk assessment |
| **Measure Template** | Standardized, reusable mitigation measure definition with known effectiveness parameters |
| **Effectiveness Estimation** | Deterministic prediction of expected risk reduction from proposed measures |
| **Risk Reduction Verification** | Post-mitigation re-assessment confirming risk has been reduced to acceptable level |
| **Mitigation Workflow** | End-to-end lifecycle from risk trigger to market placement approval |
| **Institutional Learning** | Feedback loop tracking actual vs. estimated effectiveness to improve future strategies |
| **Provenance Hash** | SHA-256 hash covering all inputs and outputs for audit verification |

### Appendix B: Effectiveness Estimation Formula

```
Mitigation Effectiveness Estimation
=====================================

Per-Measure Estimation:
    Expected_Risk_Reduction = Base_Effectiveness * Applicability_Factor * Implementation_Quality_Factor

Where:
    Base_Effectiveness: from measure template (0-100 scale, e.g., 25 = expected 25% dimension score reduction)
    Applicability_Factor: measure relevance to the specific risk driver
        Exact match:   1.0 (measure directly targets the elevated dimension)
        Related match:  0.5-0.8 (measure addresses a correlated dimension)
        Partial match:  0.2-0.4 (measure has indirect impact)
    Implementation_Quality_Factor: operator capability
        HIGH:     1.0 (dedicated compliance team, EUDR-experienced)
        STANDARD: 0.8 (moderate compliance capability)
        LOW:      0.6 (limited compliance resources)

Three-Tier Estimation:
    CONSERVATIVE: Expected_Risk_Reduction * 0.7
    MODERATE:     Expected_Risk_Reduction * 1.0
    OPTIMISTIC:   Expected_Risk_Reduction * 1.3 (capped at Base_Effectiveness)

Aggregate Strategy Reduction (diminishing returns):
    Aggregate_Reduction = 1 - PRODUCT(1 - R_i / 100)
    Where R_i is each measure's moderate expected reduction

    This models the fact that multiple measures targeting the same dimension
    have diminishing marginal returns (the second measure reduces a smaller
    remaining risk pool).

Post-Mitigation Score Projection:
    Projected_Post_Score = Pre_Score * (1 - Aggregate_Reduction)

Feasibility Assessment:
    FEASIBLE:         Projected_Post_Score < 30 (NEGLIGIBLE/LOW tier)
    LIKELY_FEASIBLE:  Projected_Post_Score < 45
    UNCERTAIN:        Projected_Post_Score < 60
    UNLIKELY:         Projected_Post_Score >= 60

Example Calculation:
    Pre-mitigation composite: 72.5 (HIGH)
    Risk drivers: Country (65), Deforestation (78)

    Measure 1: Request origin documentation
        Base: 15, Applicability: 1.0 (country), Quality: 0.8 (STANDARD)
        Expected: 15 * 1.0 * 0.8 = 12.0 points
        Conservative: 8.4, Moderate: 12.0, Optimistic: 15.0 (capped at 15)

    Measure 2: Independent field audit
        Base: 25, Applicability: 1.0 (deforestation), Quality: 0.8
        Expected: 25 * 1.0 * 0.8 = 20.0 points
        Conservative: 14.0, Moderate: 20.0, Optimistic: 25.0

    Measure 3: Satellite monitoring intensification
        Base: 20, Applicability: 0.8 (related to deforestation), Quality: 0.8
        Expected: 20 * 0.8 * 0.8 = 12.8 points
        Conservative: 8.96, Moderate: 12.8, Optimistic: 16.64 (capped at 16.0)

    Aggregate Reduction (moderate):
        = 1 - (1 - 12.0/100) * (1 - 20.0/100) * (1 - 12.8/100)
        = 1 - 0.88 * 0.80 * 0.872
        = 1 - 0.6139
        = 0.3861 (38.61%)

    Projected Post-Score: 72.5 * (1 - 0.3861) = 44.5 -> STANDARD
    Feasibility: LIKELY_FEASIBLE (< 45)
```

### Appendix C: Article 11 Workflow State Machine

```
TRIGGERED
    |
    v
STRATEGY_DESIGNED
    |
    v
PENDING_APPROVAL ---(rejected)---> STRATEGY_DESIGNED (redesign)
    |                                    |
    (approved)                           |
    |                                    |
    v                                    |
APPROVED                                |
    |                                    |
    v                                    |
IMPLEMENTING                             |
    |                                    |
    (all measures COMPLETED)             |
    |                                    |
    v                                    |
VERIFICATION_PENDING                     |
    |                  |                 |
    (risk reduced      (risk still       |
     to LOW/NEG)       HIGH/CRITICAL)    |
    |                  |                 |
    v                  v                 |
VERIFIED          IMPLEMENTING (cycle 2) |
    |                  |                 |
    v                  +---- (if cycle 2 fails) ---> ESCALATED
CLOSED
```

### Appendix D: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023
2. EU Deforestation Regulation Guidance Document (European Commission)
3. EUDR Technical Specifications for the EU Information System
4. Article 11 Risk Mitigation Measures -- Implementation Guidance (European Commission)
5. Article 10(2) Risk Assessment Criteria -- Detailed Implementation Notes (EC)
6. Article 29 Country Benchmarking Methodology (European Commission)
7. FSC Chain of Custody Standard (FSC-STD-40-004) -- Mitigation measure reference
8. RSPO Supply Chain Certification Standard -- Mitigation measure reference
9. ISO 31000:2018 -- Risk Management Guidelines
10. ISO 31010:2019 -- Risk Assessment Techniques

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-11 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-11 | GL-ProductManager | Initial draft created |
| 1.0.0 | 2026-03-11 | GL-ProductManager | Finalized: all 9 P0 features confirmed, regulatory coverage verified (Articles 10, 11, 12, 13, 14-16, 29, 31), effectiveness estimation formula defined with example calculation, Article 11(2)(a-c) category mapping complete, 50+ measure templates specified, 18 Prometheus metrics, V117 migration schema with 9 tables, 16 RBAC permissions, 12+ API endpoints, full workflow state machine, approval gates, institutional learning, approval granted |
