# PRD: AGENT-EUDR-025 -- Risk Mitigation Advisor Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-025 |
| **Agent ID** | GL-EUDR-RMA-025 |
| **Component** | Risk Mitigation Advisor Agent |
| **Category** | EUDR Regulatory Agent -- Risk Mitigation & Remediation Intelligence |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-11 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 8, 10, 11, 29, 31; ISO 31000:2018 Risk Management; ISO 14001:2015 Environmental Management |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Article 11 mandates that when a risk assessment identifies a non-negligible risk of non-compliance, operators and traders must adopt risk mitigation measures that are "adequate and proportionate" to reduce the identified risk to a negligible level before placing products on the EU market. Article 10(2) specifies the risk factors that trigger mitigation obligations: country risk (Article 29 benchmarking), supplier compliance history, commodity deforestation association, complexity of the supply chain, and risk of circumvention or mixing with products of unknown origin. Article 8 establishes the due diligence system requirement, under which risk mitigation is the third and final mandatory step (after information gathering and risk assessment). The enforcement date for large operators was December 30, 2025, and SME enforcement follows on June 30, 2026.

The GreenLang platform has built a comprehensive risk assessment capability through EUDR agents 016 through 024: Country Risk Evaluator (EUDR-016), Supplier Risk Scorer (EUDR-017), Commodity Risk Analyzer (EUDR-018), Corruption Index Monitor (EUDR-019), Deforestation Alert System (EUDR-020), Indigenous Rights Checker (EUDR-021), Protected Area Validator (EUDR-022), Legal Compliance Verifier (EUDR-023), and Third-Party Audit Manager (EUDR-024). These nine agents collectively generate detailed, multi-dimensional risk signals across country, supplier, commodity, corruption, deforestation, indigenous rights, protected area, legal, and audit dimensions. However, risk identification without risk mitigation is operationally useless. Today, operators face the following critical gaps after risk assessment:

- **No intelligent mitigation strategy recommendation**: When a risk assessment identifies high-risk suppliers, countries, or commodities, operators receive risk scores but no actionable guidance on which specific mitigation measures to deploy. Compliance officers must manually research, design, and implement mitigation plans from scratch for each risk finding, drawing on fragmented knowledge from consultants, certification body guidance, and regulatory advisories. There is no system that recommends proven, context-appropriate mitigation strategies based on the specific risk profile.
- **No structured remediation plan management**: Risk mitigation requires multi-step, multi-stakeholder remediation plans with defined timelines, milestones, responsible parties, and success criteria. Operators manage these plans through email, spreadsheets, and project management tools disconnected from the risk assessment data. There is no integrated system that generates structured remediation plans linked to specific risk findings, tracks implementation progress, and verifies completion.
- **No supplier capacity building framework**: Many EUDR compliance gaps stem not from supplier unwillingness but from supplier inability -- smallholder farmers in developing countries lack the knowledge, technology, and resources to implement deforestation-free practices. There is no structured framework for designing supplier training programs, allocating technical assistance resources, tracking capacity building progress, and measuring the resulting risk reduction.
- **No mitigation measure knowledge base**: Effective risk mitigation draws on hundreds of proven measures across categories: enhanced monitoring, supplier engagement, certification support, alternative sourcing, landscape-level interventions, community partnerships, legal compliance assistance, and technology deployment. This institutional knowledge resides in the heads of experienced consultants and in scattered best-practice publications. There is no searchable, categorized, evidence-based library of mitigation measures with effectiveness data and applicability criteria.
- **No mitigation effectiveness tracking**: After deploying mitigation measures, operators have no systematic way to measure whether the measures actually reduced risk. Without before-and-after risk scoring, ROI analysis, and impact measurement, operators cannot determine which investments are working, which need adjustment, and which should be abandoned in favor of alternatives.
- **No adaptive management capability**: Risk profiles change continuously as new deforestation alerts emerge, country risk classifications are updated, supplier compliance evolves, and regulations are amended. Static mitigation plans become obsolete. There is no continuous monitoring system that detects when existing mitigation measures become inadequate and recommends adaptive adjustments in real time.
- **No cost-benefit optimization for mitigation budgets**: Compliance budgets are finite. Operators must allocate limited resources across dozens of suppliers, multiple commodities, and various risk categories. There is no optimization engine that maximizes risk reduction per euro spent, identifies the most cost-effective mitigation strategies, and generates budget allocation recommendations based on risk priority.
- **No multi-stakeholder collaboration platform**: Effective EUDR risk mitigation requires coordination across internal compliance teams, procurement departments, supplier quality teams, NGO partners, certification bodies, and competent authorities. There is no integrated collaboration hub that connects these stakeholders around shared mitigation objectives with role-appropriate access, communication channels, and progress tracking.
- **No audit-ready mitigation documentation**: EUDR Articles 10 and 11 require operators to document risk mitigation measures as part of their due diligence statements. Competent authorities may request evidence of mitigation adequacy at any time (Articles 14-16). There is no system that automatically generates audit-ready mitigation documentation with evidence trails, effectiveness metrics, and regulatory compliance mapping.

Without solving these problems, EU operators face enforcement action under Articles 23-25 (penalties up to 4% of annual EU turnover), confiscation of goods at EU borders, mandatory market withdrawal, temporary exclusion from public procurement, public naming, and reputational damage. More critically, without effective risk mitigation, the EUDR's core objective -- halting deforestation driven by EU consumption -- cannot be achieved even if risks are accurately identified.

### 1.2 Solution Overview

Agent-EUDR-025: Risk Mitigation Advisor is a specialized compliance agent that provides intelligent, data-driven risk mitigation strategy recommendation, remediation plan management, supplier capacity building, effectiveness tracking, adaptive management, cost-benefit optimization, stakeholder collaboration, and audit-ready documentation for EUDR compliance. It is the 25th agent in the EUDR agent family and establishes a new Risk Mitigation & Remediation Intelligence sub-category that transforms risk assessment outputs into actionable, measurable, cost-effective mitigation outcomes.

The agent is the operational bridge between the risk assessment layer (EUDR-016 through EUDR-024) and the compliance reporting layer (GL-EUDR-APP DDS generation). It consumes risk signals from all nine upstream risk assessment agents, applies ML-powered recommendation algorithms and deterministic rule engines to select appropriate mitigation strategies, designs structured remediation plans, tracks implementation and effectiveness, and generates the mitigation evidence required for Due Diligence Statements.

Core capabilities:

1. **Risk Mitigation Strategy Selector** -- ML-powered recommendation engine that consumes risk inputs from EUDR-016 through EUDR-024 and recommends context-appropriate mitigation strategies. Uses gradient-boosted decision trees trained on 10,000+ historical mitigation outcomes to rank strategies by predicted effectiveness, cost, and implementation complexity. All recommendations are explainable (SHAP values) and auditable. Deterministic fallback mode ensures zero-hallucination operation.

2. **Remediation Plan Designer** -- Generates structured, multi-phase remediation plans with SMART milestones (Specific, Measurable, Achievable, Relevant, Time-bound), assigned responsible parties, resource requirements, KPI definitions, dependency tracking, and escalation triggers. Plans are linked to specific risk findings from upstream agents and validated against ISO 31000 risk treatment requirements.

3. **Supplier Capacity Building Manager** -- Designs and manages supplier development programs including training curricula, technical assistance packages, resource allocation, progress tracking, and competency assessments. Supports 4 capacity building tiers (awareness, basic compliance, advanced practices, leadership) with commodity-specific and region-specific content modules.

4. **Mitigation Measure Library** -- Searchable knowledge base of 500+ proven mitigation measures organized across 8 risk categories (country, supplier, commodity, corruption, deforestation, indigenous rights, protected areas, legal compliance). Each measure includes effectiveness evidence, cost estimates, implementation complexity, applicability criteria, prerequisite conditions, and expected risk reduction range.

5. **Effectiveness Tracking Engine** -- Measures mitigation impact through before-and-after risk scoring, ROI analysis, trend detection, and statistical significance testing. Compares predicted versus actual risk reduction. Generates effectiveness reports with confidence intervals and attribution analysis. Feeds outcomes back to the Strategy Selector to improve future recommendations.

6. **Continuous Monitoring & Adaptive Management** -- Monitors active mitigation plans against real-time risk signals from EUDR-016 through EUDR-024. Detects plan drift (mitigation not keeping pace with evolving risk), trigger events (new deforestation alert, country reclassification, audit non-conformance), and effectiveness degradation. Recommends adaptive adjustments including plan acceleration, scope expansion, strategy replacement, and emergency response activation.

7. **Cost-Benefit Optimizer** -- Budget allocation engine that maximizes aggregate risk reduction subject to budget constraints using linear programming and portfolio optimization techniques. Calculates cost-effectiveness ratios for each mitigation measure, generates Pareto-optimal budget allocation scenarios, and provides sensitivity analysis for budget trade-offs.

8. **Stakeholder Collaboration Hub** -- Multi-party coordination platform connecting internal compliance teams, procurement, supplier quality, NGO partners, certification bodies, and competent authorities around shared mitigation objectives. Role-based access control, threaded communication, document sharing, task assignment, and progress dashboards. Supplier portal for self-service mitigation progress reporting.

9. **Mitigation Reporting & Documentation** -- Generates audit-ready mitigation evidence for EUDR Due Diligence Statements (Articles 10, 11), competent authority inspections (Articles 14-16), certification scheme reviews, and third-party audits. Produces structured reports mapping each identified risk to deployed mitigation measures, implementation evidence, effectiveness metrics, and residual risk assessment. Formats: PDF, JSON, HTML, XLSX. Multi-language support (EN, FR, DE, ES, PT).

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Mitigation strategy recommendation accuracy | >= 85% of recommended strategies achieve predicted risk reduction (within +/- 15%) | Predicted vs. actual risk reduction comparison |
| Remediation plan completion rate | >= 80% of plans completed within planned timeline | Plan milestone tracking |
| Supplier capacity building coverage | 100% of high-risk suppliers enrolled in capacity building programs | Enrollment tracking vs. high-risk supplier list |
| Mitigation measure library coverage | 500+ measures across 8 risk categories | Library content count and category coverage |
| Risk reduction achieved | >= 40% average risk score reduction for suppliers under active mitigation | Before-and-after risk score comparison |
| Cost-effectiveness ratio | >= 3:1 risk-reduction-value to mitigation-cost ratio | Cost-benefit analysis engine output |
| Adaptive management response time | < 48 hours from trigger event to plan adjustment recommendation | Time from event detection to recommendation |
| Stakeholder collaboration adoption | >= 70% of mitigation plans with multi-party participation | Collaboration hub usage analytics |
| Documentation completeness | 100% of active mitigations with audit-ready evidence packages | Documentation completeness scoring |
| Processing performance | < 2 seconds per mitigation recommendation | p99 latency under load |
| Determinism | 100% reproducible for deterministic mode (zero-hallucination) | Bit-perfect reproducibility tests |
| EUDR Article compliance | Full coverage of Articles 8, 10, 11, 29, 31 mitigation requirements | Regulatory compliance matrix |
| ISO 31000 alignment | 100% of risk treatment processes aligned with ISO 31000:2018 | Framework compliance audit |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated risk management and compliance remediation technology market of 2-4 billion EUR.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers requiring structured risk mitigation programs for 7 regulated commodities, estimated at 600M-1B EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1 leveraging risk mitigation advisory capabilities, representing 40-70M EUR in risk mitigation module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) with active EUDR compliance programs
- Multinational food and beverage companies sourcing cocoa, coffee, palm oil, soya
- Timber and paper industry operators with complex multi-tier supply chains
- Automotive and tire manufacturers (rubber supply chain mitigation)
- Meat and leather importers (cattle deforestation risk mitigation)

**Secondary:**
- Compliance consulting firms advising EU operators on EUDR mitigation
- Certification bodies (FSC, RSPO, PEFC, Rainforest Alliance) supporting member remediation
- NGOs and landscape-level initiative coordinators (Tropical Forest Alliance, CDP Forests)
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026
- Financial institutions requiring portfolio-level deforestation risk mitigation evidence

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Consultant-driven | Deep domain expertise; custom strategies | EUR 50-200K per engagement; slow (months); not scalable; no effectiveness tracking | ML-powered instant recommendations; 500+ measure library; automated effectiveness tracking |
| Generic GRC platforms (SAP GRC, MetricStream) | Enterprise risk management; audit workflow | Not EUDR-specific; no deforestation domain knowledge; no supplier capacity building | Purpose-built for EUDR Article 11; 9-agent risk integration; commodity-specific measures |
| Certification scheme platforms (FSC/RSPO portals) | Scheme-specific mitigation guidance | Siloed to single scheme; no cross-scheme optimization; no cost-benefit analysis | Cross-scheme, multi-risk-category optimization with budget allocation |
| Sustainability consultancies (ERM, WSP, Quantis) | Expert knowledge; stakeholder relationships | Project-based; no continuous monitoring; no adaptive management; expensive | Always-on monitoring; real-time adaptive management; 10x more cost-effective |
| In-house custom builds | Tailored to organization | 12-18 month build; no ML model training data; no mitigation library | Ready now; pre-trained on 10,000+ outcomes; 500+ measure library |

### 2.4 Differentiation Strategy

1. **Nine-agent risk integration** -- The only platform that consumes risk signals from 9 specialized EUDR risk assessment agents (EUDR-016 through 024) for holistic mitigation strategy selection.
2. **ML-powered with deterministic fallback** -- Gradient-boosted recommendation engine for optimal strategy selection with guaranteed deterministic mode for audit-critical scenarios.
3. **500+ proven mitigation measures** -- The largest structured, evidence-based mitigation measure library in the EUDR compliance market.
4. **Closed-loop effectiveness tracking** -- Before-and-after risk scoring with ROI analysis feeds back into recommendation engine for continuous improvement.
5. **ISO 31000 alignment** -- Full alignment with the international standard for risk management, providing enterprise-grade risk treatment process governance.
6. **Cost-benefit optimization** -- Mathematical optimization of mitigation budgets using linear programming, unique in the EUDR compliance market.

---

## 3. Regulatory and Legal Requirements

### 3.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 8(1)** | Operators shall establish and implement a due diligence system that includes risk mitigation measures | Complete due diligence system integration: information (EUDR-001), assessment (EUDR-016-024), mitigation (EUDR-025) |
| **Art. 8(3)** | Due diligence system shall be reviewed at least once a year and updated when relevant | Continuous Monitoring & Adaptive Management engine with configurable review cycles and auto-update triggers |
| **Art. 10(1)** | Operators shall assess the risk that relevant commodities are non-compliant | Consumes risk assessment outputs from all 9 upstream risk agents (EUDR-016 through 024) |
| **Art. 10(2)(a)** | Risk factor: complexity of the relevant supply chain | Supply chain complexity mitigation strategies: simplification, tier reduction, alternative sourcing |
| **Art. 10(2)(b)** | Risk factor: risk of circumvention or mixing with unknown origin products | Anti-circumvention mitigation measures: segregation enforcement, mass balance verification, traceability enhancement |
| **Art. 10(2)(c)** | Risk factor: risk of non-compliance of the country or parts thereof | Country-specific mitigation strategies from EUDR-016 integration: enhanced monitoring, supplier diversification, landscape-level interventions |
| **Art. 10(2)(d)** | Risk factor: risk linked to the country of production | Country risk mitigation: governance support, capacity building, certification scheme enrollment |
| **Art. 10(2)(e)** | Risk factor: concerns about the country of production or origin, including corruption | Anti-corruption mitigation measures from EUDR-019 integration: due diligence enhancement, transparency requirements, third-party verification |
| **Art. 10(2)(f)** | Risk factor: concerns about the supplier | Supplier-specific remediation plans: capacity building, monitoring enhancement, corrective action management |
| **Art. 11(1)** | Where risk assessment identifies non-negligible risk, operators shall adopt risk mitigation measures adequate and proportionate to reduce risk to negligible level | Core agent function: strategy selection, plan design, implementation tracking, effectiveness verification |
| **Art. 11(2)(a)** | Mitigation measure: additional information, data, or documents from suppliers | Supplier information request workflows; capacity building for data provision; technology deployment |
| **Art. 11(2)(b)** | Mitigation measure: independent surveys and audits | Third-party audit integration (EUDR-024); independent verification scheduling; audit finding remediation |
| **Art. 11(2)(c)** | Mitigation measure: other measures to manage and mitigate non-negligible risk | Full 500+ measure library covering all risk categories; ML-powered strategy selection for novel situations |
| **Art. 12(2)(d)** | DDS to include risk mitigation measures adopted | Mitigation Reporting & Documentation engine generates DDS-ready mitigation evidence |
| **Art. 29(1)** | Country benchmarking -- low, standard, high risk | Country risk-level-specific mitigation strategy templates: simplified (low), standard (standard), enhanced (high) |
| **Art. 29(3)** | Benchmarking criteria include governance, enforcement, indigenous rights | Multi-dimensional mitigation addressing all benchmarking criteria through specialized measures |
| **Art. 31(1)** | Record keeping for 5 years | All mitigation records, plans, evidence, and effectiveness data retained for minimum 5 years with immutable audit trail |

### 3.2 ISO 31000:2018 Risk Management Alignment

The agent's architecture is designed to comply fully with ISO 31000:2018 "Risk management -- Guidelines," the international standard for risk management frameworks and processes.

| ISO 31000 Clause | Requirement | Agent Implementation |
|-------------------|-------------|---------------------|
| **5.5 Risk Treatment** | Select and implement risk treatment options | Strategy Selector recommends treatment options; Plan Designer implements selected treatments |
| **5.5.1** | Risk treatment options include avoiding, taking/increasing, removing the source, changing the likelihood, changing the consequences, sharing the risk, retaining the risk | All 7 treatment types represented in Mitigation Measure Library with category mapping |
| **5.5.2** | Risk treatment may introduce new risks or modify existing risks | Effectiveness Tracking Engine monitors for secondary risk emergence; Adaptive Management detects unintended consequences |
| **5.5.3** | Risk treatment selection considers cost-benefit, stakeholder views, regulatory obligations, and social responsibility | Cost-Benefit Optimizer, Stakeholder Collaboration Hub, regulatory mapping, and community engagement measures |
| **5.6 Monitoring and Review** | Continual monitoring of the risk management process | Continuous Monitoring & Adaptive Management provides real-time oversight of all active mitigation plans |
| **5.7 Recording and Reporting** | Document risk management activities and outcomes | Mitigation Reporting & Documentation generates comprehensive records per ISO 31000 reporting requirements |
| **6.1 Continual Improvement** | Improve the suitability, adequacy, and effectiveness of risk management | Closed-loop feedback from Effectiveness Tracking to Strategy Selector; adaptive plan adjustment; annual review automation |

### 3.3 Additional Regulatory Frameworks

| Framework | Relevance | Agent Support |
|-----------|-----------|---------------|
| EU Corporate Sustainability Due Diligence Directive (CSDDD) | Supply chain human rights and environmental due diligence mitigation | Mitigation measures for human rights risks; FPIC remediation (via EUDR-021); environmental impact mitigation |
| ISO 14001:2015 Environmental Management | Environmental risk management and continual improvement | Environmental risk mitigation measures; monitoring and measurement; corrective action tracking |
| ILO Convention 169 | Indigenous peoples' rights mitigation | FPIC remediation workflows; community engagement measures; benefit-sharing agreement support (via EUDR-021) |
| FSC Controlled Wood Standard | Risk assessment and mitigation for forestry | Forestry-specific mitigation measures; FSC mitigation requirement mapping |
| RSPO Principles & Criteria | Palm oil risk mitigation | Palm oil-specific mitigation measures; RSPO remediation and grievance support |
| UN Guiding Principles on Business and Human Rights | Human rights due diligence mitigation | Access to remedy; grievance mechanisms; human rights impact mitigation measures |

### 3.4 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Mitigation strategies for post-cutoff deforestation findings |
| June 29, 2023 | Regulation entered into force | Legal basis for all mitigation obligations |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Operators must have mitigation systems operational |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; capacity building programs must scale |
| Ongoing (quarterly) | Country benchmarking updates by EC | Adaptive management triggered by country reclassification |
| Ongoing (annually) | Article 8(3) due diligence system review | Annual mitigation effectiveness review automation |

---

## 4. User Stories and Personas

### 4.1 Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, sourcing cocoa from 12 countries via 300+ suppliers |
| **EUDR Pressure** | Board-level mandate to achieve EUDR compliance; risk assessment complete but mitigation plan gaps |
| **Pain Points** | Has risk scores from 9 dimensions but no actionable mitigation guidance; manually designing remediation plans for 45 high-risk suppliers; cannot demonstrate mitigation adequacy to auditors; no way to track whether mitigation investments are working |
| **Goals** | Automated mitigation strategy recommendations for each risk finding; structured remediation plans with progress tracking; audit-ready documentation for DDS submission; ROI visibility on mitigation spend |
| **Technical Skill** | Moderate -- comfortable with web applications and dashboards but not a developer |

### 4.2 Persona 2: Sustainability Director -- Henrik (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Director of Sustainability at an EU palm oil refinery |
| **Company** | 3,000 employees, sourcing from 200+ plantations across Indonesia and Malaysia |
| **EUDR Pressure** | Must demonstrate adequate risk mitigation for high-risk country sourcing; investor pressure for measurable deforestation risk reduction |
| **Pain Points** | Spending EUR 2M/year on compliance activities with no clear ROI; cannot optimize budget allocation across 200+ suppliers; supplier capacity building programs lack structure and measurable outcomes |
| **Goals** | Optimize mitigation budget for maximum risk reduction; structured capacity building programs with measurable outcomes; cost-benefit analysis for board reporting; continuous monitoring of mitigation effectiveness |
| **Technical Skill** | Moderate-high -- comfortable with analytics dashboards and data-driven decision making |

### 4.3 Persona 3: Procurement Manager -- Ana (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Procurement Director responsible for supplier management |
| **Company** | Large EU timber importer, 800 employees, importing from 20+ countries |
| **EUDR Pressure** | Must implement mitigation measures at supplier level; manage supplier remediation timelines |
| **Pain Points** | Suppliers overwhelmed by compliance requirements; no structured engagement framework; tracking remediation progress across 150+ suppliers is manual and error-prone |
| **Goals** | Structured supplier engagement workflows; capacity building programs that suppliers can follow; progress dashboards showing supplier improvement over time; collaboration tools for supplier communication |
| **Technical Skill** | Low-moderate -- uses ERP and web applications |

### 4.4 Persona 4: External Auditor -- Dr. Hofmann (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must verify that operator mitigation measures are adequate and proportionate per Article 11 |
| **Pain Points** | Operators provide unstructured mitigation evidence; cannot verify mitigation effectiveness claims; no standardized format for mitigation documentation |
| **Goals** | Access structured mitigation evidence packages; verify before-and-after risk scores; validate mitigation measure appropriateness against risk findings; audit trail for all mitigation decisions |
| **Technical Skill** | Moderate -- comfortable with audit software and structured documentation |

### 4.5 Persona 5: NGO Partnership Coordinator -- Sophie (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Landscape Program Manager at a conservation NGO (e.g., Rainforest Alliance, WWF) |
| **Company** | International conservation NGO with landscape-level programs |
| **EUDR Pressure** | Partners with operators on deforestation-free sourcing; needs visibility into operator mitigation commitments |
| **Pain Points** | Cannot track operator mitigation implementation; no shared platform for joint capacity building; reporting to donors requires evidence of impact |
| **Goals** | Collaborative mitigation planning with operator partners; shared progress tracking; impact measurement for landscape-level outcomes; communication channel for coordination |
| **Technical Skill** | Moderate -- uses project management and GIS tools |

### 4.6 User Stories

**US-001: Risk-Based Mitigation Recommendation**
```
As a compliance officer,
I want to receive intelligent, context-specific mitigation strategy recommendations for each identified risk,
So that I can quickly design adequate and proportionate mitigation measures per EUDR Article 11.
```

**US-002: Structured Remediation Plan**
```
As a compliance officer,
I want to generate structured remediation plans with timelines, milestones, and KPIs for each high-risk supplier,
So that I can track mitigation implementation systematically and demonstrate progress to auditors.
```

**US-003: Supplier Capacity Building**
```
As a procurement manager,
I want to enroll high-risk suppliers in structured capacity building programs with progress tracking,
So that suppliers can improve their practices and reduce deforestation risk in their operations.
```

**US-004: Mitigation Measure Search**
```
As a sustainability director,
I want to search a library of proven mitigation measures filtered by risk category, commodity, and cost range,
So that I can identify the most appropriate interventions for my specific supply chain context.
```

**US-005: Effectiveness Measurement**
```
As a sustainability director,
I want to see before-and-after risk scores for every deployed mitigation measure with ROI analysis,
So that I can demonstrate mitigation effectiveness to investors and optimize future spending.
```

**US-006: Adaptive Plan Adjustment**
```
As a compliance officer,
I want to be automatically alerted when a mitigation plan needs adjustment due to new risk signals,
So that I can respond quickly to evolving risks and maintain adequate mitigation per Article 8(3).
```

**US-007: Budget Optimization**
```
As a sustainability director,
I want the system to recommend optimal budget allocation across my mitigation portfolio,
So that I can maximize risk reduction within my available compliance budget.
```

**US-008: Stakeholder Collaboration**
```
As a procurement manager,
I want a shared platform where I can coordinate mitigation activities with suppliers, NGOs, and certification bodies,
So that all parties work toward aligned mitigation objectives with visibility into progress.
```

**US-009: Audit-Ready Documentation**
```
As an external auditor,
I want to access structured mitigation evidence packages that map each risk finding to deployed measures and outcomes,
So that I can efficiently verify mitigation adequacy for EUDR compliance certification.
```

**US-010: Emergency Response**
```
As a compliance officer,
I want immediate mitigation response protocols when a critical deforestation alert or rights violation is detected,
So that I can take swift action to protect my supply chain compliance and limit exposure.
```

**US-011: Cross-Risk Mitigation Coordination**
```
As a compliance officer,
I want mitigation plans that address multiple risk dimensions simultaneously (country + supplier + commodity),
So that I avoid duplicative efforts and achieve holistic risk reduction.
```

**US-012: Supplier Self-Service Reporting**
```
As a supplier,
I want a portal where I can report my mitigation progress, upload evidence, and view my risk improvement over time,
So that I can demonstrate my commitment to deforestation-free production to my EU buyer.
```

---

## 5. Feature Requirements

### 5.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-4 form the core mitigation intelligence engine; Features 5-7 form the optimization and monitoring layer; Features 8-9 form the collaboration and documentation layer.

**P0 Features 1-4: Core Mitigation Intelligence Engine**

---

#### Feature 1: Risk Mitigation Strategy Selector

**User Story:**
```
As a compliance officer,
I want the system to analyze risk assessment outputs from 9 upstream agents and recommend the most effective mitigation strategies,
So that I can quickly identify adequate and proportionate measures per EUDR Article 11.
```

**Acceptance Criteria:**
- [ ] Consumes risk inputs from all 9 upstream agents: EUDR-016 (country risk score, due diligence level), EUDR-017 (supplier risk score, risk factors), EUDR-018 (commodity risk profile, deforestation correlation), EUDR-019 (corruption perception index, governance score), EUDR-020 (deforestation alerts, severity, proximity), EUDR-021 (indigenous territory overlap, FPIC status), EUDR-022 (protected area proximity, IUCN category), EUDR-023 (legal compliance gaps, permit status), EUDR-024 (audit findings, non-conformances, CAR status)
- [ ] Implements ML-powered recommendation engine using gradient-boosted decision trees (XGBoost/LightGBM) trained on 10,000+ historical mitigation outcomes
- [ ] Generates ranked list of recommended strategies (top 5) with predicted effectiveness score (0-100), predicted cost range (EUR), implementation complexity (Low/Medium/High), and time-to-effect estimate
- [ ] Provides explainable recommendations using SHAP (SHapley Additive exPlanations) values showing which risk factors drove each recommendation
- [ ] Implements deterministic fallback mode using rule-based decision trees when ML model confidence is below 0.7 threshold
- [ ] Supports 8 risk categories for strategy recommendation: country, supplier, commodity, corruption, deforestation, indigenous rights, protected areas, legal compliance
- [ ] Generates composite mitigation strategies that address multiple risk dimensions simultaneously
- [ ] Validates recommended strategies against ISO 31000 risk treatment taxonomy (avoid, reduce, share, retain)
- [ ] Provides strategy comparison view showing trade-offs between recommended approaches
- [ ] Records all recommendations with provenance trail (input data hash, model version, parameters, timestamp) for audit purposes

**Risk Category Strategy Mapping:**

| Risk Category | Source Agent | Example Mitigation Strategies |
|---------------|-------------|-------------------------------|
| Country Risk | EUDR-016 | Enhanced monitoring, supplier diversification, landscape-level intervention |
| Supplier Risk | EUDR-017 | Capacity building, corrective action plan, supplier replacement timeline |
| Commodity Risk | EUDR-018 | Certification enrollment, traceability enhancement, commodity substitution |
| Corruption Risk | EUDR-019 | Enhanced due diligence, third-party verification, transparency requirements |
| Deforestation Risk | EUDR-020 | Immediate sourcing suspension, satellite monitoring enhancement, restoration program |
| Indigenous Rights | EUDR-021 | FPIC remediation, community engagement program, benefit-sharing agreement |
| Protected Areas | EUDR-022 | Buffer zone restoration, encroachment prevention, relocation support |
| Legal Compliance | EUDR-023 | Legal gap remediation, permit acquisition support, certification alignment |

**Non-Functional Requirements:**
- Performance: < 2 seconds per recommendation for single supplier context
- Throughput: 1,000 recommendations per minute for batch processing
- Determinism: Deterministic mode produces bit-perfect results across runs
- Auditability: Complete recommendation provenance with SHA-256 hashes

**Dependencies:**
- EUDR-016 through EUDR-024 risk assessment agent APIs
- XGBoost/LightGBM for ML recommendation engine
- SHAP library for explainability
- Mitigation Measure Library (Feature 4) for strategy catalog

**Estimated Effort:** 4 weeks (1 ML engineer, 1 backend engineer)

**Edge Cases:**
- All upstream agents report low risk -> Recommend monitoring-only measures; no active mitigation required
- Single agent unavailable -> Use cached last-known risk score with staleness indicator; flag reduced confidence
- Conflicting risk signals (e.g., low country risk but high deforestation alert) -> Weight real-time signals (deforestation) over static assessments (country)
- Novel risk pattern not in training data -> Fall back to deterministic rule engine; flag for model retraining

---

#### Feature 2: Remediation Plan Designer

**User Story:**
```
As a compliance officer,
I want to generate structured remediation plans with timelines, milestones, and KPIs for each risk finding,
So that I can manage mitigation implementation systematically per ISO 31000 risk treatment planning.
```

**Acceptance Criteria:**
- [ ] Generates multi-phase remediation plans based on Strategy Selector recommendations
- [ ] Each plan includes: plan ID, linked risk finding(s), selected mitigation strategies, SMART milestones, timeline (start/end dates), responsible parties, resource requirements (budget, personnel, equipment), KPI definitions, dependency graph, escalation triggers
- [ ] Supports 4 plan phases: Preparation (weeks 1-2), Implementation (weeks 3-8), Verification (weeks 9-10), Monitoring (ongoing)
- [ ] Auto-generates SMART milestones: Specific (linked to risk factor), Measurable (quantified KPI), Achievable (validated against resource availability), Relevant (mapped to EUDR article), Time-bound (deadline date)
- [ ] Supports plan templates for common remediation scenarios (supplier capacity building, certification enrollment, enhanced monitoring, alternative sourcing)
- [ ] Tracks milestone completion with evidence upload requirements (documents, photos, audit reports, satellite imagery)
- [ ] Implements Gantt chart view showing plan timeline with dependencies and critical path
- [ ] Supports plan versioning with change history and approval workflows
- [ ] Generates plan status dashboard: On Track, At Risk, Delayed, Completed, Abandoned
- [ ] Links each plan element to specific EUDR article requirements for regulatory traceability
- [ ] Supports plan cloning for applying successful remediation patterns to similar suppliers

**Plan Template Examples:**

| Template | Phases | Typical Duration | Risk Categories |
|----------|--------|------------------|-----------------|
| Supplier Capacity Building | Assessment -> Training -> Practice Change -> Verification | 12-24 weeks | Supplier, Commodity |
| Emergency Deforestation Response | Suspend -> Investigate -> Remediate -> Resume | 2-8 weeks | Deforestation |
| Certification Enrollment | Gap Assessment -> Preparation -> Audit -> Certification | 24-52 weeks | Legal, Commodity |
| Enhanced Monitoring Deployment | Baseline -> Deploy -> Calibrate -> Operate | 4-8 weeks | Country, Deforestation |
| FPIC Remediation | Identify -> Engage -> Consult -> Agree -> Monitor | 16-36 weeks | Indigenous Rights |
| Legal Gap Closure | Assessment -> Legal Support -> Permit Acquisition -> Verification | 8-24 weeks | Legal Compliance |
| Anti-Corruption Measures | Assessment -> Controls -> Training -> Monitoring | 8-16 weeks | Corruption |
| Protected Area Buffer Restoration | Assessment -> Planning -> Restoration -> Monitoring | 24-52 weeks | Protected Areas |

**Non-Functional Requirements:**
- Plan generation: < 5 seconds for single supplier remediation plan
- Storage: Support 10,000+ concurrent active plans
- Versioning: Complete change history retained for 5 years per Article 31

**Dependencies:**
- Strategy Selector (Feature 1) for recommended strategies
- Project management engine for Gantt chart and dependency tracking
- Document storage (S3) for evidence uploads
- Notification service for milestone reminders and escalation alerts

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend engineer)

---

#### Feature 3: Supplier Capacity Building Manager

**User Story:**
```
As a procurement manager,
I want to enroll suppliers in structured capacity building programs with training, technical assistance, and progress tracking,
So that I can help suppliers improve their practices and achieve EUDR compliance.
```

**Acceptance Criteria:**
- [ ] Supports 4 capacity building tiers: Tier 1 (Awareness -- basic EUDR requirements education), Tier 2 (Basic Compliance -- data collection, GPS capture, documentation), Tier 3 (Advanced Practices -- sustainable agriculture, deforestation-free production), Tier 4 (Leadership -- peer mentoring, community engagement, certification readiness)
- [ ] Provides commodity-specific training modules for all 7 EUDR commodities: cattle, cocoa, coffee, palm oil, rubber, soya, wood
- [ ] Provides region-specific content for 20+ commodity-producing countries with local language support
- [ ] Includes training content types: video tutorials, written guides, interactive checklists, self-assessment quizzes, field manuals
- [ ] Tracks individual supplier progress through capacity building tiers with competency assessments at each gate
- [ ] Allocates technical assistance resources: field trainers, agronomists, GIS specialists, legal advisors
- [ ] Manages resource scheduling and availability across programs
- [ ] Generates capacity building scorecards per supplier showing: current tier, modules completed, competency scores, risk reduction achieved
- [ ] Supports group training sessions with attendance tracking and post-training assessment
- [ ] Integrates with EUDR-017 Supplier Risk Scorer to correlate capacity building progress with risk score improvement
- [ ] Provides mobile-friendly content delivery for field-based suppliers in low-connectivity environments

**Capacity Building Content Matrix:**

| Tier | Cattle | Cocoa | Coffee | Palm Oil | Rubber | Soya | Wood |
|------|--------|-------|--------|----------|--------|------|------|
| T1 Awareness | 4 modules | 4 modules | 4 modules | 4 modules | 4 modules | 4 modules | 4 modules |
| T2 Basic | 8 modules | 8 modules | 8 modules | 8 modules | 8 modules | 8 modules | 8 modules |
| T3 Advanced | 6 modules | 6 modules | 6 modules | 6 modules | 6 modules | 6 modules | 6 modules |
| T4 Leadership | 4 modules | 4 modules | 4 modules | 4 modules | 4 modules | 4 modules | 4 modules |
| **Total** | **22** | **22** | **22** | **22** | **22** | **22** | **22** |

**Non-Functional Requirements:**
- Content delivery: < 3 seconds page load for training modules
- Offline support: Training content downloadable for offline access
- Scale: Support 10,000+ concurrent supplier enrollments
- Languages: Training content in EN, FR, DE, ES, PT, ID, MS (7 languages)

**Dependencies:**
- EUDR-017 Supplier Risk Scorer for supplier risk profiles
- Content management system for training material storage
- Mobile-responsive frontend for supplier portal
- Notification service for progress reminders

**Estimated Effort:** 4 weeks (1 backend engineer, 1 frontend engineer, 1 content specialist)

---

#### Feature 4: Mitigation Measure Library

**User Story:**
```
As a compliance officer,
I want to search a comprehensive library of proven mitigation measures with effectiveness data,
So that I can identify the most appropriate interventions for my specific risk context.
```

**Acceptance Criteria:**
- [ ] Contains 500+ proven mitigation measures organized across 8 risk categories: country (65+), supplier (80+), commodity (75+), corruption (55+), deforestation (70+), indigenous rights (50+), protected areas (55+), legal compliance (60+)
- [ ] Each measure includes: measure ID, name, description, risk category, sub-category, target risk factors, applicability criteria (commodity, country, supply chain tier), effectiveness evidence (sources, confidence level), cost estimate range (EUR), implementation complexity (Low/Medium/High/Very High), time-to-effect (weeks), prerequisite conditions, expected risk reduction range (%), ISO 31000 treatment type, EUDR article alignment
- [ ] Supports full-text search with relevance ranking across measure names, descriptions, and tags
- [ ] Supports faceted filtering by: risk category, commodity, country, cost range, complexity, effectiveness rating, ISO 31000 type
- [ ] Provides measure detail view with effectiveness evidence from case studies, academic research, and certification body guidance
- [ ] Supports measure comparison view (side-by-side) for decision support
- [ ] Includes version-controlled measure data with update history tracking
- [ ] Supports community-contributed measures with review and approval workflow
- [ ] Tags each measure with applicable EUDR articles and certification scheme requirements (FSC, RSPO, PEFC, RA, ISCC)
- [ ] Generates measure recommendation packages grouped by risk scenario (e.g., "High-risk palm oil from Indonesia -- recommended package")

**Measure Category Breakdown:**

| Risk Category | Count | Example Measures |
|---------------|-------|-----------------|
| Country Risk | 65+ | Landscape-level program participation, multi-stakeholder initiative enrollment, government partnership, bilateral trade agreement leverage |
| Supplier Risk | 80+ | Training program enrollment, corrective action plan, supplier scorecard implementation, site visit scheduling, alternative supplier identification |
| Commodity Risk | 75+ | Commodity-specific certification enrollment (FSC/RSPO/RA), sustainable production training, yield improvement technical assistance, intercropping promotion |
| Corruption Risk | 55+ | Third-party transaction verification, enhanced due diligence protocol, anti-bribery training, whistleblower mechanism deployment, payment transparency |
| Deforestation Risk | 70+ | Satellite monitoring enhancement, zero-deforestation commitment, forest restoration program, fire prevention training, agroforestry transition support |
| Indigenous Rights | 50+ | FPIC process implementation, community benefit-sharing agreement, grievance mechanism establishment, cultural heritage protection, land rights documentation support |
| Protected Areas | 55+ | Buffer zone restoration, encroachment monitoring, community-based conservation support, alternative livelihood program, reforestation initiative |
| Legal Compliance | 60+ | Legal permit acquisition support, environmental impact assessment, labour law compliance training, tax compliance assistance, export documentation support |

**Non-Functional Requirements:**
- Search: < 500ms for full-text search across 500+ measures
- Content: Version-controlled with quarterly update cycle
- Scalability: Support growth to 2,000+ measures without performance degradation

**Dependencies:**
- PostgreSQL full-text search for measure catalog
- Content management system for measure data maintenance
- Strategy Selector (Feature 1) consumes library for recommendations

**Estimated Effort:** 3 weeks (1 domain specialist, 1 backend engineer)

---

**P0 Features 5-7: Optimization and Monitoring Layer**

---

#### Feature 5: Effectiveness Tracking Engine

**User Story:**
```
As a sustainability director,
I want to measure the actual impact of deployed mitigation measures through before-and-after risk scoring and ROI analysis,
So that I can demonstrate mitigation effectiveness and optimize future investments.
```

**Acceptance Criteria:**
- [ ] Captures baseline risk scores from all 9 upstream agents at mitigation plan activation (T0 snapshot)
- [ ] Captures periodic risk score updates at configurable intervals (default: monthly) from all 9 upstream agents
- [ ] Calculates risk reduction delta per risk dimension: delta = (T0 score - Tn score) / T0 score * 100
- [ ] Calculates composite risk reduction across all dimensions using the same weighting as the upstream risk assessment
- [ ] Performs ROI analysis: ROI = (Risk Reduction Value - Mitigation Cost) / Mitigation Cost * 100, where Risk Reduction Value = Risk Score Reduction * Penalty Exposure * Probability Factor
- [ ] Tracks predicted versus actual risk reduction (from Strategy Selector predictions) with deviation analysis
- [ ] Implements statistical significance testing (paired t-test) for risk reduction claims with configurable confidence levels (default: 95%)
- [ ] Generates effectiveness trend charts showing risk score trajectory over time per supplier, commodity, and country
- [ ] Identifies underperforming mitigation measures (actual reduction < 50% of predicted) and recommends strategy adjustment
- [ ] Generates executive effectiveness dashboard: total risk reduction achieved, total mitigation spend, overall ROI, top/bottom performing measures
- [ ] Feeds effectiveness outcomes back to Strategy Selector ML model for continuous improvement of future recommendations
- [ ] Produces audit-ready effectiveness reports with statistical methodology documentation

**Effectiveness Metrics Framework:**

| Metric | Formula | Frequency |
|--------|---------|-----------|
| Risk Reduction Rate | (Baseline Risk - Current Risk) / Baseline Risk | Monthly |
| Time to First Improvement | Days from plan activation to first measurable risk reduction | Per plan |
| Cost per Risk Point Reduced | Total Mitigation Cost / Total Risk Points Reduced | Quarterly |
| Strategy Accuracy | Count(Actual within +/-15% of Predicted) / Total Recommendations | Quarterly |
| ROI | (Risk Reduction Value - Mitigation Cost) / Mitigation Cost | Semi-annually |
| Supplier Improvement Rate | % of suppliers showing >= 20% risk reduction within 6 months | Semi-annually |

**Non-Functional Requirements:**
- Calculation: < 3 seconds for full effectiveness assessment per supplier
- Storage: 5-year historical effectiveness data retention
- Determinism: All calculations reproducible with documented methodology
- Accuracy: Risk reduction calculations use Decimal arithmetic (no floating-point drift)

**Dependencies:**
- All 9 upstream risk agents (EUDR-016 through 024) for periodic risk score snapshots
- Strategy Selector (Feature 1) for predicted effectiveness values
- Remediation Plan Designer (Feature 2) for plan timeline and cost data
- Statistical libraries (SciPy) for significance testing

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data engineer)

---

#### Feature 6: Continuous Monitoring & Adaptive Management

**User Story:**
```
As a compliance officer,
I want the system to continuously monitor active mitigation plans against real-time risk signals and recommend adjustments when conditions change,
So that I can maintain adequate mitigation in response to evolving risks per EUDR Article 8(3).
```

**Acceptance Criteria:**
- [ ] Subscribes to real-time event streams from all 9 upstream risk agents via message queue
- [ ] Detects 6 trigger event types that require mitigation plan adjustment:
  - Country reclassification (e.g., standard -> high risk from EUDR-016)
  - Supplier risk score spike (> 20% increase from EUDR-017)
  - New deforestation alert on monitored plot (from EUDR-020)
  - Indigenous rights violation report in sourcing region (from EUDR-021)
  - Protected area encroachment detection (from EUDR-022)
  - Audit non-conformance finding (from EUDR-024)
- [ ] Generates adaptive adjustment recommendations within 48 hours of trigger event detection
- [ ] Supports 5 adjustment types: Plan Acceleration (shorten timelines), Scope Expansion (add measures), Strategy Replacement (swap underperforming measures), Emergency Response Activation (immediate action protocol), Plan De-escalation (reduce measures when risk decreases)
- [ ] Implements alert fatigue prevention: consolidates multiple related triggers into single adjustment recommendation; respects configurable quiet periods
- [ ] Tracks plan drift metric: deviation between planned mitigation trajectory and actual risk trajectory
- [ ] Implements annual due diligence system review automation per Article 8(3): generates review report, identifies systemic gaps, recommends structural improvements
- [ ] Provides real-time mitigation dashboard showing all active plans, current status, risk trajectories, and pending adjustments
- [ ] Supports configurable escalation chains for unacknowledged adjustment recommendations (24h -> team lead, 48h -> director, 72h -> executive)
- [ ] Records all trigger events, recommendations, and decisions with provenance trail for audit purposes

**Trigger Event Response Matrix:**

| Trigger Event | Severity | Response Time SLA | Default Adjustment |
|---------------|----------|-------------------|--------------------|
| Critical deforestation alert | Critical | 4 hours | Emergency sourcing suspension + investigation |
| Country reclassification to high risk | High | 48 hours | Enhanced due diligence activation + strategy review |
| Supplier risk spike > 50% | High | 24 hours | Plan acceleration + additional measures |
| Supplier risk spike 20-50% | Medium | 48 hours | Plan scope expansion |
| Audit critical non-conformance | High | 24 hours | CAR-linked remediation plan activation |
| Indigenous rights violation | High | 24 hours | FPIC remediation + community engagement |
| Protected area encroachment | High | 24 hours | Buffer zone intervention + encroachment prevention |
| Risk score improvement > 30% | Low | 1 week | Plan de-escalation assessment |

**Non-Functional Requirements:**
- Event processing: < 5 seconds from event receipt to trigger detection
- Recommendation generation: < 48 hours from trigger to recommendation
- Availability: 99.9% uptime for monitoring service
- Message delivery: At-least-once delivery guarantee for risk events

**Dependencies:**
- All 9 upstream risk agents (EUDR-016 through 024) for event streams
- Message queue (Redis Streams or Kafka) for event processing
- Strategy Selector (Feature 1) for adjustment recommendations
- Notification service for escalation alerts

**Estimated Effort:** 3 weeks (1 backend engineer, 1 DevOps engineer)

---

#### Feature 7: Cost-Benefit Optimizer

**User Story:**
```
As a sustainability director,
I want the system to recommend optimal budget allocation across my mitigation portfolio,
So that I can maximize aggregate risk reduction within my available compliance budget.
```

**Acceptance Criteria:**
- [ ] Accepts budget constraints (total annual budget, per-supplier budget caps, per-category allocation limits)
- [ ] Calculates cost-effectiveness ratio for each candidate mitigation measure: CE = Expected Risk Reduction / Expected Cost
- [ ] Implements linear programming optimization (PuLP/OR-Tools) to maximize aggregate risk reduction subject to budget constraints
- [ ] Generates Pareto-optimal frontier showing trade-offs between budget levels and risk reduction outcomes
- [ ] Supports multi-scenario analysis: "What if budget increases by 20%?", "What if we drop the lowest-performing supplier?"
- [ ] Provides sensitivity analysis: which budget allocation decisions have the largest impact on aggregate risk
- [ ] Prioritizes mitigation investments using RICE framework: Reach (suppliers affected), Impact (risk reduction), Confidence (prediction reliability), Effort (cost and complexity)
- [ ] Generates quarterly budget allocation recommendations with rationale documentation
- [ ] Tracks actual spend versus planned allocation with variance reporting
- [ ] Supports multi-year budget planning with year-over-year risk reduction projections
- [ ] Generates board-ready financial reports: total compliance spend, ROI achieved, projected penalty avoidance, cost per compliant supplier

**Optimization Model:**
```
Maximize: SUM(risk_reduction_i * weight_i) for all suppliers i
Subject to:
  SUM(cost_i) <= total_budget
  cost_i <= per_supplier_cap for all suppliers i
  SUM(cost_j) <= category_budget_k for all measures j in category k
  risk_reduction_i >= 0 for all suppliers i

Where:
  risk_reduction_i = f(selected_measures, baseline_risk, supplier_context)
  cost_i = SUM(measure_cost) for selected measures for supplier i
  weight_i = risk_level_weight * commodity_weight * volume_weight
```

**Non-Functional Requirements:**
- Optimization: < 30 seconds for portfolio of 500 suppliers with 20 candidate measures each
- Scenario analysis: < 10 seconds per scenario evaluation
- Determinism: Same inputs produce same optimization results

**Dependencies:**
- PuLP or OR-Tools for linear programming
- Mitigation Measure Library (Feature 4) for cost estimates
- Strategy Selector (Feature 1) for predicted risk reduction values
- Effectiveness Tracking Engine (Feature 5) for historical cost-effectiveness data

**Estimated Effort:** 3 weeks (1 optimization engineer, 1 backend engineer)

---

**P0 Features 8-9: Collaboration and Documentation Layer**

---

#### Feature 8: Stakeholder Collaboration Hub

**User Story:**
```
As a procurement manager,
I want a shared platform to coordinate mitigation activities with suppliers, NGOs, and certification bodies,
So that all stakeholders work toward aligned mitigation objectives with visibility into progress.
```

**Acceptance Criteria:**
- [ ] Supports 6 stakeholder roles: Internal Compliance Team, Procurement Team, Supplier, NGO Partner, Certification Body, Competent Authority (read-only)
- [ ] Provides role-based access control: suppliers see only their own plans and data; NGOs see landscape-level aggregates; authorities see compliance documentation
- [ ] Implements threaded communication channels per mitigation plan with @mention, file attachment, and read receipts
- [ ] Supports task assignment to any stakeholder with due dates, priority, and completion tracking
- [ ] Provides supplier self-service portal: view mitigation plan, report progress, upload evidence documents, view risk improvement trajectory
- [ ] Supports NGO partnership workspace: shared landscape-level mitigation goals, joint progress tracking, impact reporting
- [ ] Implements document sharing with version control and access logging
- [ ] Generates stakeholder-specific progress dashboards: supplier view (my plan status), NGO view (landscape progress), internal view (portfolio overview)
- [ ] Supports bulk communication: send mitigation updates to all suppliers in a category, country, or risk level
- [ ] Provides notification preferences per stakeholder: email, in-app, SMS (for critical escalations)
- [ ] Implements activity audit trail: all collaboration actions logged with timestamp and actor for regulatory evidence

**Stakeholder Access Matrix:**

| Capability | Internal Team | Procurement | Supplier | NGO Partner | Cert Body | Authority |
|-----------|---------------|-------------|----------|-------------|-----------|-----------|
| View all plans | Yes | Yes | Own only | Landscape | Scheme-related | Requested |
| Create/edit plans | Yes | Limited | No | No | No | No |
| Report progress | Yes | Yes | Own plan | Joint plans | Audit results | No |
| Upload evidence | Yes | Yes | Own plan | Joint plans | Audit reports | No |
| View risk scores | Full | Full | Own risk | Aggregate | Scheme-related | Full |
| Communication | Full | Full | Plan-scoped | Landscape | Scheme | Request-response |
| Analytics | Full | Full | Own trends | Landscape | Scheme | Compliance |
| Export reports | Full | Full | Own | Landscape | Scheme | Compliance |

**Non-Functional Requirements:**
- Availability: 99.9% uptime for collaboration platform
- Notification: < 5 seconds for real-time notification delivery
- Storage: Unlimited document storage (S3-backed)
- Security: End-to-end encryption for sensitive communications; GDPR-compliant data handling
- Languages: Platform UI in 7 languages (EN, FR, DE, ES, PT, ID, MS)

**Dependencies:**
- SEC-001 JWT Authentication for multi-party access
- SEC-002 RBAC Authorization for role-based access control
- S3 for document storage
- WebSocket or SSE for real-time notifications
- Email service for external notifications

**Estimated Effort:** 4 weeks (1 backend engineer, 1 frontend engineer)

---

#### Feature 9: Mitigation Reporting & Documentation

**User Story:**
```
As an external auditor,
I want to access comprehensive, structured mitigation evidence packages for each operator,
So that I can verify mitigation adequacy and proportionality per EUDR Article 11.
```

**Acceptance Criteria:**
- [ ] Generates DDS Mitigation Section (Article 12(2)(d)): for each identified risk, documents the specific mitigation measures adopted, implementation status, and effectiveness evidence
- [ ] Generates Competent Authority Response Package (Articles 14-16): comprehensive mitigation documentation package for regulatory inspection response, including risk-to-mitigation mapping, implementation evidence, and residual risk assessment
- [ ] Generates Third-Party Audit Evidence Package (EUDR-024 integration): structured mitigation evidence for audit finding verification, CAR closure evidence, and continuous improvement documentation
- [ ] Generates Annual Due Diligence Review Report (Article 8(3)): annual assessment of mitigation system effectiveness with improvement recommendations
- [ ] Produces Risk-to-Mitigation Mapping Report: for every risk finding from EUDR-016 through 024, maps the specific mitigation measure(s) deployed, implementation status, and measured effectiveness
- [ ] Generates Supplier Mitigation Scorecard: per-supplier report showing risk profile, deployed measures, progress status, effectiveness metrics, and residual risk
- [ ] Generates Portfolio Mitigation Summary: aggregate report across all suppliers showing total risk reduction achieved, budget utilization, ROI, and compliance readiness
- [ ] Includes provenance hashes (SHA-256) on all report data for integrity verification
- [ ] Supports 5 output formats: PDF (human-readable), JSON (machine-readable), HTML (web-embeddable), XLSX (analyst-friendly), XML (regulatory submission)
- [ ] Supports multi-language report generation: EN, FR, DE, ES, PT
- [ ] Provides report scheduling: automated generation on configurable intervals (weekly, monthly, quarterly)
- [ ] Maintains 5-year report archive per Article 31 with version history

**Report Types:**

| Report | Audience | Frequency | Format | EUDR Article |
|--------|----------|-----------|--------|-------------|
| DDS Mitigation Section | EU Information System | Per DDS submission | JSON/XML | Art. 12(2)(d) |
| Competent Authority Package | EU Member State authorities | On request | PDF + JSON | Art. 14-16 |
| Annual DDS Review | Internal + Auditor | Annual | PDF | Art. 8(3) |
| Supplier Mitigation Scorecard | Supplier + Internal | Monthly | PDF + HTML | Art. 10-11 |
| Portfolio Summary | Board + Investors | Quarterly | PDF + XLSX | Art. 8 |
| Risk-Mitigation Mapping | Auditor | Per audit | PDF + JSON | Art. 10-11 |
| Effectiveness Analysis | Internal + Investors | Quarterly | PDF + XLSX | Art. 11 |

**Non-Functional Requirements:**
- Generation: < 10 seconds per report for single supplier scope
- Batch generation: 100 supplier scorecards in < 5 minutes
- Storage: 5-year archive with immutable version history
- Compliance: 100% coverage of Article 11 documentation requirements

**Dependencies:**
- All upstream features (1-8) for data inputs
- PDF generation library (ReportLab or WeasyPrint)
- S3 for report archive storage
- GL-EUDR-APP DDS generation workflow for DDS integration

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend engineer)

---

### 5.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Predictive Risk Trajectory Modeling
- Forecast risk score evolution 3/6/12 months ahead based on current mitigation trajectory
- Predict which suppliers will achieve compliance targets and which will miss them
- Generate early warning alerts for projected mitigation failure
- Support scenario modeling: "What if we accelerate capacity building by 50%?"

#### Feature 11: Landscape-Level Mitigation Coordination
- Aggregate mitigation efforts across multiple operators in shared sourcing regions
- Coordinate landscape-level interventions (jurisdictional approaches, sectoral commitments)
- Track collective impact on regional deforestation rates
- Support Tropical Forest Alliance and other multi-stakeholder initiative reporting

#### Feature 12: AI-Powered Root Cause Analysis
- Analyze patterns across mitigation failures to identify systemic root causes
- Recommend structural interventions beyond individual supplier remediation
- Detect correlation between mitigation effectiveness and contextual factors (country, season, commodity price)
- Generate root cause analysis reports for compliance improvement planning

---

### 5.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Carbon credit generation from mitigation activities (defer to GL-GHG-APP integration)
- Direct financial payments to suppliers for mitigation compliance (defer to payment platform)
- Legal advisory services integration (defer to legal technology partnership)
- Real-time video monitoring of supplier implementation (defer to IoT integration phase)
- Mobile native application (web responsive design only for v1.0)
- Blockchain-based mitigation evidence immutability (SHA-256 provenance hashes provide sufficient integrity)

---

## 6. Technical Architecture Overview

### 6.1 Architecture Diagram

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
        +-----------------------------+-----------------------------+
        |                             |                             |
+-------v--------+        +----------v----------+        +---------v--------+
| AGENT-EUDR-025 |        | Risk Assessment     |        | Reporting Layer  |
| Risk Mitigation|<------>| Layer (EUDR-016-024)|        | (GL-EUDR-APP DDS)|
| Advisor        |        |                     |        |                  |
|                |        | EUDR-016 Country    |        | DDS Generator    |
| - Strategy     |        | EUDR-017 Supplier   |        | Compliance Rpts  |
|   Selector     |        | EUDR-018 Commodity  |        |                  |
| - Plan Designer|        | EUDR-019 Corruption |        +------------------+
| - Capacity Bld |        | EUDR-020 Deforest.  |
| - Measure Lib  |        | EUDR-021 Indigenous |
| - Effectiveness|        | EUDR-022 Protected  |
| - Adaptive Mgmt|        | EUDR-023 Legal      |
| - Cost-Benefit |        | EUDR-024 Audit      |
| - Collaboration|        +---------------------+
| - Reporting    |
+-------+--------+
        |
+-------v--------------------------+
| Data & Infrastructure Layer      |
| PostgreSQL + TimescaleDB         |
| Redis Cache                      |
| S3 Object Storage                |
| Message Queue (Redis Streams)    |
+----------------------------------+
```

### 6.2 Module Structure

```
greenlang/agents/eudr/risk_mitigation_advisor/
    __init__.py                              # Public API exports
    config.py                                # RiskMitigationAdvisorConfig with GL_EUDR_RMA_ env prefix
    models.py                                # Pydantic v2 models for all domain entities
    strategy_selector.py                     # StrategySelector: ML-powered recommendation engine
    remediation_plan_designer.py             # RemediationPlanDesigner: structured plan generation
    supplier_capacity_builder.py             # SupplierCapacityBuilder: capacity building management
    mitigation_measure_library.py            # MitigationMeasureLibrary: 500+ measure catalog
    effectiveness_tracker.py                 # EffectivenessTracker: before/after scoring, ROI
    adaptive_management.py                   # AdaptiveManagement: continuous monitoring, triggers
    cost_benefit_optimizer.py                # CostBenefitOptimizer: budget allocation optimization
    stakeholder_collaboration.py             # StakeholderCollaboration: multi-party coordination
    mitigation_reporter.py                   # MitigationReporter: audit-ready documentation
    provenance.py                            # ProvenanceTracker: SHA-256 hash chains
    metrics.py                               # 18 Prometheus self-monitoring metrics
    setup.py                                 # RiskMitigationAdvisorService facade
    reference_data/
        __init__.py
        mitigation_measures.py               # 500+ mitigation measure definitions
        plan_templates.py                    # Remediation plan templates
        capacity_building_curricula.py       # Training content metadata
        risk_category_strategies.py          # Category-specific strategy rules
    ml/
        __init__.py
        model_trainer.py                     # Model training pipeline
        feature_engineering.py               # Feature extraction from risk inputs
        model_registry.py                    # Model versioning and deployment
        shap_explainer.py                    # SHAP-based explainability
    api/
        __init__.py
        router.py                            # FastAPI router (35+ endpoints)
        strategy_routes.py                   # Strategy recommendation endpoints
        plan_routes.py                       # Remediation plan CRUD endpoints
        capacity_routes.py                   # Capacity building endpoints
        library_routes.py                    # Mitigation measure library endpoints
        effectiveness_routes.py              # Effectiveness tracking endpoints
        monitoring_routes.py                 # Adaptive management endpoints
        optimization_routes.py               # Cost-benefit optimization endpoints
        collaboration_routes.py              # Stakeholder collaboration endpoints
        reporting_routes.py                  # Report generation endpoints
```

### 6.3 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| ML Framework | XGBoost + LightGBM | Gradient-boosted trees for strategy recommendation |
| Explainability | SHAP | Model-agnostic explainability for audit compliance |
| Optimization | PuLP + OR-Tools | Linear programming for cost-benefit optimization |
| Statistics | SciPy + NumPy | Statistical significance testing, risk calculations |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables |
| Cache | Redis | Strategy recommendation caching, session management |
| Message Queue | Redis Streams | Event-driven adaptive management triggers |
| Object Storage | S3 | Documents, reports, evidence packages, ML models |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access control for multi-party collaboration |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| PDF Generation | ReportLab + WeasyPrint | Audit-ready PDF report generation |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

---

## 7. Data Model and Schemas

### 7.1 Core Domain Models

```python
# Risk Mitigation Strategy
class MitigationStrategy(BaseModel):
    strategy_id: str                     # Unique identifier
    name: str                            # Human-readable strategy name
    description: str                     # Detailed description
    risk_categories: List[RiskCategory]  # Applicable risk categories
    iso_31000_type: ISO31000TreatmentType  # Avoid/Reduce/Share/Retain
    target_risk_factors: List[str]       # Specific risk factors addressed
    predicted_effectiveness: float       # 0-100 predicted risk reduction
    confidence_score: float              # 0-1 ML model confidence
    cost_estimate: CostEstimate          # Low/Medium/High with EUR range
    implementation_complexity: Complexity # Low/Medium/High/VeryHigh
    time_to_effect_weeks: int            # Expected weeks to measurable impact
    prerequisite_conditions: List[str]   # Conditions that must be met first
    eudr_articles: List[str]             # Linked EUDR articles
    shap_explanation: Dict[str, float]   # SHAP values for explainability
    measure_ids: List[str]              # Linked mitigation measures from library
    provenance_hash: str                 # SHA-256

class RiskCategory(str, Enum):
    COUNTRY = "country"
    SUPPLIER = "supplier"
    COMMODITY = "commodity"
    CORRUPTION = "corruption"
    DEFORESTATION = "deforestation"
    INDIGENOUS_RIGHTS = "indigenous_rights"
    PROTECTED_AREAS = "protected_areas"
    LEGAL_COMPLIANCE = "legal_compliance"

class ISO31000TreatmentType(str, Enum):
    AVOID = "avoid"            # Decide not to start/continue the activity
    REDUCE = "reduce"          # Remove the source or change likelihood/consequences
    SHARE = "share"            # Transfer risk through insurance or outsourcing
    RETAIN = "retain"          # Accept the risk by informed decision

# Remediation Plan
class RemediationPlan(BaseModel):
    plan_id: str                         # Unique identifier
    operator_id: str                     # Owner operator
    supplier_id: Optional[str]           # Target supplier (if supplier-specific)
    risk_finding_ids: List[str]          # Linked risk findings
    strategies: List[MitigationStrategy] # Selected strategies
    status: PlanStatus                   # Draft/Active/Completed/Suspended/Abandoned
    phases: List[PlanPhase]              # Multi-phase plan structure
    milestones: List[Milestone]          # SMART milestones
    kpis: List[KPI]                      # Key performance indicators
    budget_allocated: Decimal            # EUR
    budget_spent: Decimal                # EUR
    start_date: date
    target_end_date: date
    actual_end_date: Optional[date]
    responsible_parties: List[ResponsibleParty]
    escalation_triggers: List[EscalationTrigger]
    version: int
    created_at: datetime
    updated_at: datetime

class PlanStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    DELAYED = "delayed"
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    ABANDONED = "abandoned"

class Milestone(BaseModel):
    milestone_id: str
    plan_id: str
    name: str
    description: str
    phase: str
    due_date: date
    completed_date: Optional[date]
    status: MilestoneStatus              # Pending/InProgress/Completed/Overdue
    kpi_target: Optional[str]
    evidence_required: List[str]         # Required evidence types
    evidence_uploaded: List[EvidenceDocument]
    eudr_article: Optional[str]

# Mitigation Measure (Library Entry)
class MitigationMeasure(BaseModel):
    measure_id: str
    name: str
    description: str
    risk_category: RiskCategory
    sub_category: str
    target_risk_factors: List[str]
    applicability: MeasureApplicability  # Commodities, countries, tiers
    effectiveness_evidence: List[EvidenceSource]
    effectiveness_rating: float          # 0-100
    cost_estimate_eur: CostRange         # Min/Max EUR
    implementation_complexity: Complexity
    time_to_effect_weeks: int
    prerequisite_conditions: List[str]
    expected_risk_reduction_pct: CostRange  # Min/Max %
    iso_31000_type: ISO31000TreatmentType
    eudr_articles: List[str]
    certification_schemes: List[str]     # FSC, RSPO, etc.
    tags: List[str]
    version: str
    last_updated: date

# Effectiveness Record
class EffectivenessRecord(BaseModel):
    record_id: str
    plan_id: str
    supplier_id: str
    measurement_date: datetime
    baseline_risk_scores: Dict[str, float]   # Risk category -> baseline score
    current_risk_scores: Dict[str, float]    # Risk category -> current score
    risk_reduction_pct: Dict[str, float]     # Risk category -> reduction %
    composite_reduction_pct: float           # Weighted composite reduction
    predicted_reduction_pct: float           # Strategy Selector prediction
    deviation_pct: float                     # Actual vs predicted deviation
    roi: Optional[float]                     # Return on investment
    cost_to_date: Decimal                    # EUR spent to date
    statistical_significance: bool           # p-value < 0.05
    p_value: Optional[float]
    provenance_hash: str

# Capacity Building Enrollment
class CapacityBuildingEnrollment(BaseModel):
    enrollment_id: str
    supplier_id: str
    program_id: str
    commodity: str
    current_tier: int                        # 1-4
    modules_completed: int
    modules_total: int
    competency_scores: Dict[str, float]      # Module -> score
    enrolled_date: date
    target_completion_date: date
    status: EnrollmentStatus                 # Active/Paused/Completed/Withdrawn
    risk_score_at_enrollment: float
    current_risk_score: float
```

### 7.2 Database Schema (New Migration: V113)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_risk_mitigation_advisor;

-- Mitigation strategies (recommended and selected)
CREATE TABLE eudr_risk_mitigation_advisor.mitigation_strategies (
    strategy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    supplier_id UUID,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    risk_categories JSONB NOT NULL DEFAULT '[]',
    iso_31000_type VARCHAR(50) NOT NULL,
    target_risk_factors JSONB DEFAULT '[]',
    predicted_effectiveness NUMERIC(5,2) DEFAULT 0.0,
    confidence_score NUMERIC(4,3) DEFAULT 0.0,
    cost_estimate JSONB DEFAULT '{}',
    implementation_complexity VARCHAR(20) DEFAULT 'medium',
    time_to_effect_weeks INTEGER DEFAULT 8,
    prerequisite_conditions JSONB DEFAULT '[]',
    eudr_articles JSONB DEFAULT '[]',
    shap_explanation JSONB DEFAULT '{}',
    measure_ids JSONB DEFAULT '[]',
    model_version VARCHAR(50),
    provenance_hash VARCHAR(64) NOT NULL,
    status VARCHAR(30) DEFAULT 'recommended',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Remediation plans
CREATE TABLE eudr_risk_mitigation_advisor.remediation_plans (
    plan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    supplier_id UUID,
    plan_name VARCHAR(500) NOT NULL,
    risk_finding_ids JSONB DEFAULT '[]',
    strategy_ids JSONB DEFAULT '[]',
    status VARCHAR(30) DEFAULT 'draft',
    phases JSONB DEFAULT '[]',
    budget_allocated NUMERIC(18,2) DEFAULT 0.0,
    budget_spent NUMERIC(18,2) DEFAULT 0.0,
    start_date DATE,
    target_end_date DATE,
    actual_end_date DATE,
    responsible_parties JSONB DEFAULT '[]',
    escalation_triggers JSONB DEFAULT '[]',
    plan_template VARCHAR(100),
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Plan milestones
CREATE TABLE eudr_risk_mitigation_advisor.plan_milestones (
    milestone_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id UUID NOT NULL REFERENCES eudr_risk_mitigation_advisor.remediation_plans(plan_id),
    name VARCHAR(500) NOT NULL,
    description TEXT,
    phase VARCHAR(100),
    due_date DATE NOT NULL,
    completed_date DATE,
    status VARCHAR(30) DEFAULT 'pending',
    kpi_target VARCHAR(200),
    evidence_required JSONB DEFAULT '[]',
    evidence_uploaded JSONB DEFAULT '[]',
    eudr_article VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Mitigation measure library
CREATE TABLE eudr_risk_mitigation_advisor.mitigation_measures (
    measure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    risk_category VARCHAR(50) NOT NULL,
    sub_category VARCHAR(100),
    target_risk_factors JSONB DEFAULT '[]',
    applicability JSONB DEFAULT '{}',
    effectiveness_evidence JSONB DEFAULT '[]',
    effectiveness_rating NUMERIC(5,2) DEFAULT 0.0,
    cost_estimate_min NUMERIC(18,2),
    cost_estimate_max NUMERIC(18,2),
    implementation_complexity VARCHAR(20) DEFAULT 'medium',
    time_to_effect_weeks INTEGER DEFAULT 8,
    prerequisite_conditions JSONB DEFAULT '[]',
    expected_risk_reduction_min NUMERIC(5,2),
    expected_risk_reduction_max NUMERIC(5,2),
    iso_31000_type VARCHAR(50),
    eudr_articles JSONB DEFAULT '[]',
    certification_schemes JSONB DEFAULT '[]',
    tags JSONB DEFAULT '[]',
    version VARCHAR(20) DEFAULT '1.0.0',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Effectiveness tracking records (hypertable)
CREATE TABLE eudr_risk_mitigation_advisor.effectiveness_records (
    record_id UUID DEFAULT gen_random_uuid(),
    plan_id UUID NOT NULL,
    supplier_id UUID NOT NULL,
    baseline_risk_scores JSONB NOT NULL,
    current_risk_scores JSONB NOT NULL,
    risk_reduction_pct JSONB NOT NULL,
    composite_reduction_pct NUMERIC(5,2),
    predicted_reduction_pct NUMERIC(5,2),
    deviation_pct NUMERIC(5,2),
    roi NUMERIC(10,2),
    cost_to_date NUMERIC(18,2),
    statistical_significance BOOLEAN DEFAULT FALSE,
    p_value NUMERIC(6,4),
    provenance_hash VARCHAR(64) NOT NULL,
    measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_risk_mitigation_advisor.effectiveness_records', 'measured_at');

-- Capacity building enrollments
CREATE TABLE eudr_risk_mitigation_advisor.capacity_building_enrollments (
    enrollment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    program_id VARCHAR(100) NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    current_tier INTEGER DEFAULT 1,
    modules_completed INTEGER DEFAULT 0,
    modules_total INTEGER DEFAULT 22,
    competency_scores JSONB DEFAULT '{}',
    enrolled_date DATE NOT NULL,
    target_completion_date DATE,
    status VARCHAR(30) DEFAULT 'active',
    risk_score_at_enrollment NUMERIC(5,2),
    current_risk_score NUMERIC(5,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Adaptive management trigger events (hypertable)
CREATE TABLE eudr_risk_mitigation_advisor.trigger_events (
    event_id UUID DEFAULT gen_random_uuid(),
    plan_id UUID,
    trigger_type VARCHAR(50) NOT NULL,
    source_agent VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT NOT NULL,
    risk_data JSONB DEFAULT '{}',
    recommended_adjustment JSONB DEFAULT '{}',
    adjustment_type VARCHAR(50),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_risk_mitigation_advisor.trigger_events', 'detected_at');

-- Cost-benefit optimization results
CREATE TABLE eudr_risk_mitigation_advisor.optimization_results (
    optimization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    budget_total NUMERIC(18,2) NOT NULL,
    budget_constraints JSONB DEFAULT '{}',
    optimization_result JSONB NOT NULL,
    pareto_frontier JSONB DEFAULT '[]',
    sensitivity_analysis JSONB DEFAULT '{}',
    total_predicted_risk_reduction NUMERIC(5,2),
    solver_status VARCHAR(30),
    computation_time_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Stakeholder collaboration messages (hypertable)
CREATE TABLE eudr_risk_mitigation_advisor.collaboration_messages (
    message_id UUID DEFAULT gen_random_uuid(),
    plan_id UUID NOT NULL,
    sender_id VARCHAR(100) NOT NULL,
    sender_role VARCHAR(50) NOT NULL,
    message_type VARCHAR(30) DEFAULT 'text',
    content TEXT NOT NULL,
    attachments JSONB DEFAULT '[]',
    mentions JSONB DEFAULT '[]',
    read_by JSONB DEFAULT '[]',
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_risk_mitigation_advisor.collaboration_messages', 'sent_at');

-- Mitigation reports (hypertable)
CREATE TABLE eudr_risk_mitigation_advisor.mitigation_reports (
    report_id UUID DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    report_scope JSONB DEFAULT '{}',
    report_data JSONB NOT NULL,
    format VARCHAR(10) DEFAULT 'pdf',
    language VARCHAR(5) DEFAULT 'en',
    s3_key VARCHAR(500),
    provenance_hash VARCHAR(64) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_risk_mitigation_advisor.mitigation_reports', 'generated_at');

-- Indexes
CREATE INDEX idx_strategies_operator ON eudr_risk_mitigation_advisor.mitigation_strategies(operator_id);
CREATE INDEX idx_strategies_supplier ON eudr_risk_mitigation_advisor.mitigation_strategies(supplier_id);
CREATE INDEX idx_strategies_status ON eudr_risk_mitigation_advisor.mitigation_strategies(status);
CREATE INDEX idx_plans_operator ON eudr_risk_mitigation_advisor.remediation_plans(operator_id);
CREATE INDEX idx_plans_supplier ON eudr_risk_mitigation_advisor.remediation_plans(supplier_id);
CREATE INDEX idx_plans_status ON eudr_risk_mitigation_advisor.remediation_plans(status);
CREATE INDEX idx_milestones_plan ON eudr_risk_mitigation_advisor.plan_milestones(plan_id);
CREATE INDEX idx_milestones_status ON eudr_risk_mitigation_advisor.plan_milestones(status);
CREATE INDEX idx_milestones_due ON eudr_risk_mitigation_advisor.plan_milestones(due_date);
CREATE INDEX idx_measures_category ON eudr_risk_mitigation_advisor.mitigation_measures(risk_category);
CREATE INDEX idx_measures_complexity ON eudr_risk_mitigation_advisor.mitigation_measures(implementation_complexity);
CREATE INDEX idx_measures_active ON eudr_risk_mitigation_advisor.mitigation_measures(is_active);
CREATE INDEX idx_measures_fulltext ON eudr_risk_mitigation_advisor.mitigation_measures USING gin(to_tsvector('english', name || ' ' || description));
CREATE INDEX idx_effectiveness_plan ON eudr_risk_mitigation_advisor.effectiveness_records(plan_id);
CREATE INDEX idx_effectiveness_supplier ON eudr_risk_mitigation_advisor.effectiveness_records(supplier_id);
CREATE INDEX idx_enrollments_supplier ON eudr_risk_mitigation_advisor.capacity_building_enrollments(supplier_id);
CREATE INDEX idx_enrollments_status ON eudr_risk_mitigation_advisor.capacity_building_enrollments(status);
CREATE INDEX idx_triggers_plan ON eudr_risk_mitigation_advisor.trigger_events(plan_id);
CREATE INDEX idx_triggers_severity ON eudr_risk_mitigation_advisor.trigger_events(severity);
CREATE INDEX idx_triggers_resolved ON eudr_risk_mitigation_advisor.trigger_events(resolved);
CREATE INDEX idx_collaboration_plan ON eudr_risk_mitigation_advisor.collaboration_messages(plan_id);
CREATE INDEX idx_reports_operator ON eudr_risk_mitigation_advisor.mitigation_reports(operator_id);
CREATE INDEX idx_reports_type ON eudr_risk_mitigation_advisor.mitigation_reports(report_type);
```

---

## 8. API and Integration Points

### 8.1 API Endpoints (35+)

| Method | Path | Description |
|--------|------|-------------|
| **Strategy Recommendation** | | |
| POST | `/v1/mitigation/strategies/recommend` | Generate mitigation strategy recommendations for a risk context |
| GET | `/v1/mitigation/strategies` | List recommended strategies (with filters) |
| GET | `/v1/mitigation/strategies/{strategy_id}` | Get strategy details with SHAP explanation |
| POST | `/v1/mitigation/strategies/{strategy_id}/select` | Select a strategy for implementation |
| GET | `/v1/mitigation/strategies/{strategy_id}/explain` | Get detailed SHAP explainability report |
| **Remediation Plans** | | |
| POST | `/v1/mitigation/plans` | Create a new remediation plan |
| GET | `/v1/mitigation/plans` | List plans (with filters: status, supplier, commodity) |
| GET | `/v1/mitigation/plans/{plan_id}` | Get plan details with milestones and progress |
| PUT | `/v1/mitigation/plans/{plan_id}` | Update plan details |
| PUT | `/v1/mitigation/plans/{plan_id}/status` | Update plan status |
| POST | `/v1/mitigation/plans/{plan_id}/clone` | Clone plan as template for another supplier |
| GET | `/v1/mitigation/plans/{plan_id}/gantt` | Get Gantt chart data for plan timeline |
| **Plan Milestones** | | |
| POST | `/v1/mitigation/plans/{plan_id}/milestones` | Add milestone to plan |
| PUT | `/v1/mitigation/plans/{plan_id}/milestones/{milestone_id}` | Update milestone status/evidence |
| POST | `/v1/mitigation/plans/{plan_id}/milestones/{milestone_id}/evidence` | Upload evidence for milestone |
| **Capacity Building** | | |
| POST | `/v1/mitigation/capacity-building/enroll` | Enroll supplier in capacity building program |
| GET | `/v1/mitigation/capacity-building/enrollments` | List enrollments (with filters) |
| GET | `/v1/mitigation/capacity-building/enrollments/{enrollment_id}` | Get enrollment progress |
| PUT | `/v1/mitigation/capacity-building/enrollments/{enrollment_id}/progress` | Update module completion |
| GET | `/v1/mitigation/capacity-building/scorecard/{supplier_id}` | Get supplier capacity scorecard |
| **Mitigation Measure Library** | | |
| GET | `/v1/mitigation/measures` | Search/list mitigation measures (with faceted filters) |
| GET | `/v1/mitigation/measures/{measure_id}` | Get measure details |
| GET | `/v1/mitigation/measures/compare` | Compare measures side-by-side |
| GET | `/v1/mitigation/measures/packages/{risk_scenario}` | Get recommended measure package for scenario |
| **Effectiveness Tracking** | | |
| GET | `/v1/mitigation/effectiveness/{plan_id}` | Get effectiveness metrics for plan |
| GET | `/v1/mitigation/effectiveness/supplier/{supplier_id}` | Get supplier effectiveness history |
| GET | `/v1/mitigation/effectiveness/portfolio` | Get portfolio-level effectiveness summary |
| GET | `/v1/mitigation/effectiveness/roi` | Get ROI analysis across mitigation portfolio |
| **Adaptive Management** | | |
| GET | `/v1/mitigation/monitoring/triggers` | List active trigger events |
| PUT | `/v1/mitigation/monitoring/triggers/{event_id}/acknowledge` | Acknowledge a trigger event |
| GET | `/v1/mitigation/monitoring/dashboard` | Get real-time monitoring dashboard data |
| GET | `/v1/mitigation/monitoring/drift/{plan_id}` | Get plan drift analysis |
| **Cost-Benefit Optimization** | | |
| POST | `/v1/mitigation/optimization/run` | Run budget optimization |
| GET | `/v1/mitigation/optimization/{optimization_id}` | Get optimization results |
| GET | `/v1/mitigation/optimization/{optimization_id}/pareto` | Get Pareto frontier data |
| GET | `/v1/mitigation/optimization/{optimization_id}/sensitivity` | Get sensitivity analysis |
| **Collaboration** | | |
| POST | `/v1/mitigation/collaboration/{plan_id}/messages` | Post message to plan thread |
| GET | `/v1/mitigation/collaboration/{plan_id}/messages` | Get plan conversation thread |
| POST | `/v1/mitigation/collaboration/{plan_id}/tasks` | Assign task to stakeholder |
| GET | `/v1/mitigation/collaboration/supplier-portal/{supplier_id}` | Supplier self-service portal data |
| **Reporting** | | |
| POST | `/v1/mitigation/reports/generate` | Generate mitigation report |
| GET | `/v1/mitigation/reports` | List generated reports |
| GET | `/v1/mitigation/reports/{report_id}/download` | Download report file |
| GET | `/v1/mitigation/reports/dds-section/{operator_id}` | Get DDS mitigation section data |
| **Health** | | |
| GET | `/health` | Service health check |

### 8.2 Upstream Dependencies (Data Sources)

| Agent | Integration Method | Data Consumed |
|-------|-------------------|---------------|
| EUDR-016 Country Risk Evaluator | REST API + Event Stream | Country risk scores, due diligence levels, hotspot data, governance indices |
| EUDR-017 Supplier Risk Scorer | REST API + Event Stream | Supplier risk scores, risk factors, compliance history, risk level changes |
| EUDR-018 Commodity Risk Analyzer | REST API | Commodity risk profiles, deforestation correlation, seasonal patterns |
| EUDR-019 Corruption Index Monitor | REST API + Event Stream | CPI scores, governance scores, corruption alerts, risk level changes |
| EUDR-020 Deforestation Alert System | Event Stream (priority) | Deforestation alerts with severity, proximity, affected plots, cutoff verification |
| EUDR-021 Indigenous Rights Checker | REST API + Event Stream | Territory overlaps, FPIC status, rights violation alerts |
| EUDR-022 Protected Area Validator | REST API + Event Stream | Protected area overlaps, buffer zone violations, IUCN category impacts |
| EUDR-023 Legal Compliance Verifier | REST API | Legal compliance gaps, permit status, certification validity |
| EUDR-024 Third-Party Audit Manager | REST API + Event Stream | Audit findings, non-conformances, CAR status, audit results |
| AGENT-DATA-005 EUDR Traceability | REST API | Supply chain data for context enrichment |
| AGENT-FOUND-005 Citations | REST API | Regulatory references for compliance mapping |

### 8.3 Downstream Consumers

| Consumer | Integration Method | Data Provided |
|----------|-------------------|---------------|
| GL-EUDR-APP v1.0 | API integration | Mitigation dashboard data, plan status, effectiveness metrics |
| GL-EUDR-APP DDS Generator | API | DDS mitigation section (Article 12(2)(d)) |
| EUDR-017 Supplier Risk Scorer | Event Stream | Mitigation status updates that influence supplier risk scoring |
| EUDR-024 Third-Party Audit Manager | API | CAR-linked remediation plans, mitigation evidence for audit verification |
| External Auditors | Read-only API + Report Downloads | Audit-ready mitigation evidence packages |
| Competent Authorities | Report Downloads | Regulatory compliance documentation packages |

---

## 9. Security and Compliance

### 9.1 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-rma:strategies:read` | View mitigation strategy recommendations | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rma:strategies:execute` | Generate new strategy recommendations | Analyst, Compliance Officer, Admin |
| `eudr-rma:plans:read` | View remediation plans and milestones | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rma:plans:write` | Create, update, clone remediation plans | Analyst, Compliance Officer, Admin |
| `eudr-rma:plans:approve` | Approve plan activation and completion | Compliance Officer, Admin |
| `eudr-rma:capacity:read` | View capacity building enrollments | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rma:capacity:manage` | Manage capacity building programs | Procurement Manager, Compliance Officer, Admin |
| `eudr-rma:library:read` | Browse mitigation measure library | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rma:library:manage` | Add/update mitigation measures | Compliance Officer, Admin |
| `eudr-rma:effectiveness:read` | View effectiveness tracking metrics | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rma:monitoring:read` | View adaptive management dashboard | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rma:monitoring:acknowledge` | Acknowledge trigger events | Analyst, Compliance Officer, Admin |
| `eudr-rma:optimization:execute` | Run cost-benefit optimization | Sustainability Director, Compliance Officer, Admin |
| `eudr-rma:optimization:read` | View optimization results | Analyst, Compliance Officer, Admin |
| `eudr-rma:collaboration:participate` | Post messages, upload documents | All authenticated roles |
| `eudr-rma:collaboration:manage` | Manage collaboration settings, assign tasks | Compliance Officer, Admin |
| `eudr-rma:reports:read` | View and download reports | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-rma:reports:generate` | Generate new reports | Analyst, Compliance Officer, Admin |
| `eudr-rma:reports:dds` | Export DDS mitigation section | Compliance Officer, Admin |
| `eudr-rma:audit:read` | View audit trail and provenance data | Auditor (read-only), Compliance Officer, Admin |
| `eudr-rma:supplier-portal:access` | Supplier self-service portal access | Supplier (external role) |

### 9.2 Data Security Requirements

| Requirement | Implementation |
|-------------|---------------|
| Data at rest encryption | AES-256-GCM via SEC-003 for all database columns containing sensitive data |
| Data in transit encryption | TLS 1.3 via SEC-004 for all API endpoints and inter-agent communication |
| Authentication | JWT (RS256) via SEC-001 for all endpoints |
| Authorization | RBAC with 21 granular permissions via SEC-002 |
| Audit logging | All CRUD operations logged via SEC-005 Centralized Audit Logging |
| Data provenance | SHA-256 hashes on all strategies, plans, effectiveness records, and reports |
| PII protection | Supplier contact information protected via SEC-011 PII Detection/Redaction |
| Multi-tenant isolation | Operator-level data isolation enforced at query layer |
| GDPR compliance | Data retention policies, right to erasure support, consent tracking |
| Secrets management | ML model credentials, API keys stored in Vault via SEC-006 |

---

## 10. Performance and Scalability

### 10.1 Performance Targets

| Operation | Target Latency | Measurement |
|-----------|---------------|-------------|
| Strategy recommendation (single supplier) | < 2 seconds | p99 under load |
| Remediation plan generation | < 5 seconds | p99 under load |
| Mitigation measure search | < 500ms | p99 under load |
| Effectiveness calculation (single plan) | < 3 seconds | p99 under load |
| Adaptive management event processing | < 5 seconds | p99 event-to-detection |
| Cost-benefit optimization (500 suppliers) | < 30 seconds | p99 under load |
| Report generation (single supplier) | < 10 seconds | p99 under load |
| Batch report generation (100 suppliers) | < 5 minutes | Total processing time |
| API response (standard queries) | < 200ms | p95 under load |
| Collaboration message delivery | < 5 seconds | p99 real-time |

### 10.2 Scalability Targets

| Dimension | Target | Architecture Support |
|-----------|--------|---------------------|
| Concurrent active plans | 10,000+ | PostgreSQL with appropriate indexing |
| Concurrent supplier enrollments | 10,000+ | Horizontal scaling with K8s |
| Mitigation measures in library | 2,000+ | Full-text search with GIN index |
| Strategy recommendations per minute | 1,000+ | Batch processing with model caching |
| Historical effectiveness records | 10M+ | TimescaleDB hypertable with retention |
| Collaboration messages | 100M+ | TimescaleDB hypertable with archival |
| Concurrent API users | 500+ | FastAPI async with connection pooling |
| ML model inference | 100 inferences/second | Model caching, batch prediction |

### 10.3 Resource Requirements

| Resource | Development | Staging | Production |
|----------|-------------|---------|------------|
| CPU | 2 vCPU | 4 vCPU | 8 vCPU |
| Memory | 4 GB | 8 GB | 16 GB |
| ML Model Memory | 512 MB | 1 GB | 2 GB |
| Database Storage | 10 GB | 50 GB | 500 GB |
| S3 Storage | 5 GB | 25 GB | 250 GB |

---

## 11. Testing and Quality Assurance

### 11.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Strategy Selector Unit Tests | 120+ | ML recommendation, deterministic fallback, SHAP explainability, edge cases |
| Remediation Plan Tests | 80+ | Plan generation, milestone tracking, Gantt chart, template cloning |
| Capacity Building Tests | 70+ | Enrollment, tier progression, competency assessment, scorecard generation |
| Mitigation Measure Library Tests | 60+ | Search, filtering, comparison, measure packaging |
| Effectiveness Tracking Tests | 90+ | Baseline capture, delta calculation, ROI, statistical significance, audit trail |
| Adaptive Management Tests | 80+ | Trigger detection, adjustment recommendation, escalation, alert fatigue prevention |
| Cost-Benefit Optimizer Tests | 70+ | LP optimization, Pareto frontier, sensitivity analysis, constraint handling |
| Stakeholder Collaboration Tests | 60+ | Role-based access, messaging, task assignment, supplier portal |
| Reporting Tests | 70+ | All 7 report types, 5 formats, multi-language, provenance hashes |
| API Tests | 90+ | All 35+ endpoints, auth, error handling, pagination, rate limiting |
| Integration Tests | 40+ | Cross-agent integration with EUDR-016 through 024 |
| Performance Tests | 25+ | Latency targets, batch processing, optimization solver timing |
| ML Model Tests | 30+ | Model accuracy, training pipeline, feature engineering, model versioning |
| Golden Tests | 35+ | End-to-end mitigation scenarios for all 7 commodities and 8 risk categories |
| **Total** | **920+** | |

### 11.2 Golden Test Scenarios

Each golden test validates a complete mitigation lifecycle from risk input to effectiveness measurement:

| # | Scenario | Risk Source | Expected Outcome |
|---|----------|-------------|-----------------|
| 1 | High-risk country cocoa sourcing | EUDR-016 (high country risk) | Enhanced monitoring + landscape intervention recommended |
| 2 | Supplier with critical audit findings | EUDR-024 (critical NC) | CAR-linked remediation plan with 2-week SLA |
| 3 | Deforestation alert on palm oil plot | EUDR-020 (critical alert) | Emergency response: suspend + investigate + remediate |
| 4 | Indigenous territory overlap | EUDR-021 (direct overlap) | FPIC remediation workflow with community engagement |
| 5 | Protected area buffer zone violation | EUDR-022 (buffer encroachment) | Buffer restoration plan + encroachment prevention |
| 6 | Legal compliance gap (missing permits) | EUDR-023 (permit gap) | Legal gap closure plan with permit acquisition support |
| 7 | High corruption risk origin | EUDR-019 (high CPI risk) | Anti-corruption measures + third-party verification |
| 8 | Multi-dimensional risk (country + supplier + commodity) | EUDR-016/017/018 | Composite mitigation strategy addressing all dimensions |
| 9 | Budget-constrained optimization | All agents | Optimal budget allocation across 50 suppliers |
| 10 | Adaptive management trigger cascade | EUDR-020 event | Plan acceleration + scope expansion recommendation |

### 11.3 Test Coverage Targets

| Coverage Type | Target |
|---------------|--------|
| Line coverage | >= 85% |
| Branch coverage | >= 80% |
| Integration test coverage | >= 70% |
| API endpoint coverage | 100% |
| Report format coverage | 100% (all 5 formats) |

---

## 12. Documentation Requirements

### 12.1 Technical Documentation

| Document | Audience | Format |
|----------|----------|--------|
| API Reference (OpenAPI 3.1 spec) | Developers, integrators | Swagger UI + JSON |
| Data Model Documentation | Developers, database administrators | Markdown + ER diagrams |
| ML Model Documentation | Data scientists, auditors | Markdown + model card |
| Integration Guide | Platform engineers | Markdown with code examples |
| Database Migration Guide (V113) | DevOps engineers | SQL + migration notes |

### 12.2 User Documentation

| Document | Audience | Format |
|----------|----------|--------|
| Mitigation Strategy Guide | Compliance officers | HTML/PDF user guide |
| Remediation Plan Management Guide | Compliance + procurement teams | HTML/PDF user guide |
| Supplier Capacity Building Handbook | Procurement managers + suppliers | HTML/PDF + mobile guide |
| Cost-Benefit Analysis User Guide | Sustainability directors | HTML/PDF with worked examples |
| Collaboration Hub User Guide | All stakeholders | HTML/PDF per role |

### 12.3 Compliance Documentation

| Document | Audience | Format |
|----------|----------|--------|
| EUDR Article 11 Mitigation Mapping | Regulators, auditors | PDF + JSON |
| ISO 31000 Alignment Assessment | Internal compliance, certification bodies | PDF |
| SHAP Explainability Methodology | Auditors, regulators | PDF technical note |
| Data Provenance and Integrity Documentation | Auditors | PDF with SHA-256 verification guide |

---

## 13. Implementation Roadmap

### Phase 1: Core Mitigation Intelligence (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Mitigation Measure Library (Feature 4): 500+ measures, search, filtering | Domain Specialist + Backend |
| 2-3 | Strategy Selector -- Deterministic Mode (Feature 1): rule-based recommendation engine | Backend Engineer |
| 3-4 | Strategy Selector -- ML Mode (Feature 1): model training, SHAP integration | ML Engineer |
| 4-5 | Remediation Plan Designer (Feature 2): plan generation, milestones, templates | Backend Engineer |
| 5-6 | Supplier Capacity Building Manager (Feature 3): enrollment, tiers, content framework | Backend + Frontend |

**Milestone: Core mitigation intelligence engine operational (Week 6)**

### Phase 2: Optimization and Monitoring (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Effectiveness Tracking Engine (Feature 5): baseline capture, delta calculation, ROI | Backend + Data Engineer |
| 8-9 | Continuous Monitoring & Adaptive Management (Feature 6): event processing, triggers, adjustments | Backend + DevOps |
| 9-10 | Cost-Benefit Optimizer (Feature 7): LP optimization, Pareto frontier, sensitivity analysis | Optimization + Backend |

**Milestone: Full optimization and monitoring layer operational (Week 10)**

### Phase 3: Collaboration and Documentation (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | Stakeholder Collaboration Hub (Feature 8): messaging, tasks, supplier portal | Backend + Frontend |
| 12-13 | Mitigation Reporting & Documentation (Feature 9): all 7 report types, 5 formats | Backend + Frontend |
| 13-14 | REST API Layer: 35+ endpoints, authentication, RBAC integration, rate limiting | Backend Engineer |

**Milestone: All 9 P0 features implemented with full API (Week 14)**

### Phase 4: Testing, Integration, and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 920+ tests, golden tests for all scenarios | Test Engineer |
| 16-17 | Cross-agent integration testing with EUDR-016 through 024 | Integration Engineer |
| 17 | Database migration V113 finalized; security audit; performance testing | DevOps + Security |
| 17-18 | Beta customer onboarding (5 customers); documentation completion | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 9 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Predictive risk trajectory modeling (Feature 10)
- Landscape-level mitigation coordination (Feature 11)
- AI-powered root cause analysis (Feature 12)
- ML model v2 with expanded training data from production usage
- Additional training content for 10+ additional languages

---

## 14. User Experience

### 14.1 User Flows

#### Flow 1: Initial Mitigation Setup (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Risk Mitigation" module
3. System displays risk summary from EUDR-016 through 024:
   - 12 high-risk suppliers, 23 standard-risk, 85 low-risk
   - 3 critical deforestation alerts
   - 2 indigenous territory overlaps
   - 5 legal compliance gaps
4. Officer clicks "Generate Mitigation Strategies" for high-risk suppliers
5. Strategy Selector analyzes all risk dimensions per supplier
   -> Returns ranked mitigation strategies with SHAP explanations
6. Officer reviews recommendations, selects preferred strategies
7. System generates structured remediation plans per supplier
   -> Plans include milestones, KPIs, timelines, responsible parties
8. Officer reviews and approves plans (status: Draft -> Active)
9. System sends notifications to responsible parties and suppliers
10. Dashboard shows all active plans with progress tracking
```

#### Flow 2: Supplier Capacity Building Enrollment (Procurement Manager)

```
1. Procurement manager opens "Capacity Building" tab
2. System shows list of suppliers eligible for capacity building
   -> Filtered by risk level (high-risk suppliers prioritized)
3. Manager selects 10 palm oil suppliers from Indonesia
4. System recommends Tier 1 (Awareness) program for palm oil
   -> 4 modules: EUDR basics, GPS capture training, documentation, self-assessment
5. Manager confirms enrollment and sets target completion date
6. System sends enrollment notification to suppliers with portal link
7. Suppliers access mobile-friendly training portal
   -> Complete video tutorials, checklists, and quizzes
8. Manager monitors progress dashboard:
   - 7/10 suppliers completed Module 1
   - 3/10 suppliers started Module 2
   - Average competency score: 72%
9. As suppliers complete Tier 1, risk scores improve
10. Manager advances qualifying suppliers to Tier 2 (Basic Compliance)
```

#### Flow 3: Emergency Deforestation Response (Compliance Officer)

```
1. EUDR-020 detects critical deforestation alert on monitored plot
   -> Alert: 15 ha clearing detected on Plantation-47 (palm oil, Indonesia)
2. Adaptive Management engine processes trigger event
   -> Severity: CRITICAL, Response SLA: 4 hours
3. System generates emergency response recommendation:
   - Immediate: Suspend sourcing from Plantation-47
   - 24 hours: Notify supplier and request investigation report
   - 1 week: Deploy satellite monitoring enhancement
   - 2 weeks: Commission independent site investigation
4. Compliance officer receives urgent notification (email + SMS)
5. Officer reviews recommendation and activates emergency protocol
6. System creates emergency remediation plan with accelerated timeline
7. Supplier receives notification via Collaboration Hub
8. Officer tracks investigation progress through milestone updates
9. Investigation determines: clearing was on adjacent plot (false positive)
10. Officer resolves alert, documents investigation, plan marked completed
```

#### Flow 4: Budget Optimization (Sustainability Director)

```
1. Sustainability director opens "Cost-Benefit" module
2. Enters budget parameters:
   - Total annual mitigation budget: EUR 2,000,000
   - Per-supplier cap: EUR 50,000
   - Priority: maximize deforestation risk reduction
3. System runs LP optimization across 120 suppliers
   -> Evaluates 20 candidate measures per supplier
   -> Solves optimization in ~25 seconds
4. System displays Pareto frontier:
   - Scenario A: EUR 1.5M spend -> 35% average risk reduction
   - Scenario B: EUR 2.0M spend -> 48% average risk reduction
   - Scenario C: EUR 2.5M spend -> 52% average risk reduction
5. Director selects Scenario B and reviews allocation details
6. System shows per-supplier budget allocation with expected outcomes
7. Sensitivity analysis reveals: 60% of risk reduction comes from top 20 suppliers
8. Director approves allocation -> plans auto-generated for each supplier
9. Quarterly review: actual spend vs. planned, actual risk reduction vs. predicted
10. Director adjusts next quarter allocation based on effectiveness data
```

#### Flow 5: Audit Evidence Preparation (External Auditor)

```
1. Competent authority requests EUDR compliance documentation (Art. 14)
2. Compliance officer opens "Reporting" module
3. Selects "Competent Authority Response Package"
4. System generates comprehensive documentation:
   - Risk assessment summary (from EUDR-016-024)
   - Risk-to-mitigation mapping for every finding
   - Implementation evidence for each measure
   - Effectiveness metrics with before/after scores
   - Residual risk assessment
   - All provenance hashes for data integrity
5. Report generated in PDF + JSON (< 10 seconds)
6. Officer reviews and downloads report package
7. Package includes SHA-256 integrity hashes on all data
8. Auditor verifies hashes and reviews structured evidence
9. Audit finds mitigation measures adequate and proportionate
10. Documentation archived for 5-year retention (Art. 31)
```

### 14.2 Key Screen Descriptions

**Mitigation Dashboard (Main View):**
- Top cards: Total active plans, Plans on track, Plans at risk, Plans delayed
- Risk reduction summary: aggregate % reduction achieved across portfolio
- Budget utilization: allocated vs. spent with forecast
- Right panel: Recent trigger events requiring attention
- Bottom: Supplier risk heatmap with mitigation status overlay

**Strategy Recommendation View:**
- Left panel: Risk summary from 9 upstream agents (radar chart)
- Center: Ranked strategy recommendations with effectiveness scores
- SHAP waterfall chart showing risk factor contributions
- Right panel: Strategy detail with cost, timeline, complexity, prerequisites
- Action buttons: Select Strategy, Compare Strategies, View Alternatives

**Remediation Plan View:**
- Gantt chart showing plan timeline with phases and milestones
- Milestone list with status indicators (green/yellow/red)
- Evidence upload area for completed milestones
- KPI tracking panel with target vs. actual
- Collaboration thread for plan-specific communication

**Effectiveness Dashboard:**
- Time-series chart: risk score trajectory per supplier/commodity
- Before/after comparison: baseline vs. current risk scores
- ROI analysis: spend vs. risk reduction value
- Statistical significance indicators for risk reduction claims
- Underperforming measures flagged for review

**Cost-Benefit Optimizer View:**
- Budget input panel: total, constraints, priorities
- Pareto frontier visualization (interactive scatter plot)
- Sensitivity analysis heatmap
- Per-supplier allocation table with expected outcomes
- Scenario comparison tool (side-by-side)

---

## 15. Success Criteria

### 15.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Risk Mitigation Strategy Selector -- ML + deterministic recommendation
  - [ ] Feature 2: Remediation Plan Designer -- plan generation, milestones, templates
  - [ ] Feature 3: Supplier Capacity Building Manager -- enrollment, tiers, progress tracking
  - [ ] Feature 4: Mitigation Measure Library -- 500+ measures, search, filtering
  - [ ] Feature 5: Effectiveness Tracking Engine -- before/after scoring, ROI, significance testing
  - [ ] Feature 6: Continuous Monitoring & Adaptive Management -- triggers, adjustments, escalation
  - [ ] Feature 7: Cost-Benefit Optimizer -- LP optimization, Pareto frontier, sensitivity
  - [ ] Feature 8: Stakeholder Collaboration Hub -- messaging, tasks, supplier portal
  - [ ] Feature 9: Mitigation Reporting & Documentation -- 7 report types, 5 formats
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC integrated with 21 permissions)
- [ ] Performance targets met (< 2 seconds strategy recommendation p99)
- [ ] ML model validated: >= 85% of predictions within +/-15% of actual outcomes (backtested)
- [ ] Deterministic fallback mode verified bit-perfect reproducible
- [ ] All 35+ API endpoints documented (OpenAPI spec)
- [ ] Database migration V113 tested and validated
- [ ] Integration with all 9 upstream risk agents (EUDR-016 through 024) verified
- [ ] 500+ mitigation measures loaded and validated in library
- [ ] 5 beta customers successfully using mitigation features
- [ ] No critical or high-severity bugs in backlog

### 15.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ active remediation plans created by customers
- 100+ strategy recommendations generated
- 20+ suppliers enrolled in capacity building programs
- Average recommendation acceptance rate >= 60%
- < 5 support tickets per customer
- p99 recommendation latency < 2 seconds in production

**60 Days:**
- 200+ active remediation plans
- 500+ strategy recommendations generated
- 100+ supplier capacity building enrollments
- Average risk reduction >= 15% for plans active > 30 days
- Effectiveness tracking operational for 80%+ of active plans
- First budget optimization runs completed by 10+ customers

**90 Days:**
- 500+ active remediation plans
- 1,000+ strategy recommendations generated
- 300+ capacity building enrollments
- Average risk reduction >= 25% for plans active > 60 days
- Average ROI >= 2:1 across completed plans
- Zero EUDR penalties attributed to inadequate mitigation for active customers
- NPS > 50 from compliance officer persona

---

## 16. Dependencies

### 16.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Stable, production-ready |
| EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Stable, production-ready |
| EUDR-018 Commodity Risk Analyzer | BUILT (100%) | Low | Stable, production-ready |
| EUDR-019 Corruption Index Monitor | BUILT (100%) | Low | Stable, production-ready |
| EUDR-020 Deforestation Alert System | BUILT (100%) | Low | Stable, production-ready |
| EUDR-021 Indigenous Rights Checker | BUILT (100%) | Low | Integration points defined |
| EUDR-022 Protected Area Validator | BUILT (100%) | Low | Integration points defined |
| EUDR-023 Legal Compliance Verifier | BUILT (100%) | Low | Integration points defined |
| EUDR-024 Third-Party Audit Manager | BUILT (100%) | Low | Integration points defined |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Frontend integration defined |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| Redis Cache + Streams | Production Ready | Low | Standard infrastructure |
| S3 Object Storage | Production Ready | Low | Standard infrastructure |

### 16.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| XGBoost/LightGBM Python libraries | Stable (open-source) | Low | Pinned versions in requirements |
| SHAP library | Stable (open-source) | Low | Pinned version; fallback to feature importance |
| PuLP/OR-Tools optimization libraries | Stable (open-source) | Low | Pinned versions; fallback to heuristic optimization |
| SciPy statistics library | Stable (open-source) | Low | Pinned version |
| EC country benchmarking list | Published; updated periodically | Medium | Database-driven, hot-reloadable country risk |
| EU EUDR implementing regulations | Evolving | Medium | Configuration-driven compliance rules |
| Mitigation best practice publications | Published by certification bodies, NGOs | Low | Versioned in library with update cycle |
| Training content for capacity building | To be developed | Medium | Phase content creation by commodity; start with top 3 commodities |

---

## 17. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | ML model accuracy below target due to limited training data at launch | Medium | High | Deterministic fallback mode ensures operational capability without ML; expand training data from production usage; start with conservative predictions |
| R2 | Suppliers resistant to capacity building enrollment or participation | High | High | Mobile-optimized content delivery; local language support; demonstrate risk score improvement as incentive; partner with certification bodies for trusted delivery |
| R3 | Mitigation measure library content gaps for specific commodity-country combinations | Medium | Medium | Prioritize top 10 commodity-country pairs (80% of traffic); community contribution mechanism for edge cases; quarterly library expansion cycle |
| R4 | Integration complexity with 9 upstream risk agents creating reliability issues | Medium | High | Circuit breaker pattern on all agent integrations; cached last-known values; graceful degradation with reduced confidence indicators |
| R5 | Cost-benefit optimizer produces counter-intuitive results that users distrust | Medium | Medium | Transparent optimization model with sensitivity analysis; human-in-the-loop approval; override capability for manual adjustments |
| R6 | EUDR regulation amended with new mitigation requirements | Low | Medium | Modular architecture allows rapid addition of new measures; configurable rule engine; regulatory update tracking |
| R7 | Stakeholder collaboration adoption lower than expected | Medium | Medium | Progressive rollout starting with internal teams; supplier portal launched after internal workflows validated; training and onboarding support |
| R8 | Effectiveness tracking shows low mitigation impact, undermining product value | Medium | High | Set realistic expectations (Article 11 requires "adequate and proportionate" not perfect); benchmark against industry averages; focus on risk trend improvement |
| R9 | Performance degradation with large optimization portfolios (>1000 suppliers) | Low | Medium | Portfolio partitioning by commodity/country; parallel solver execution; caching of intermediate results |
| R10 | Competitive tools launch with similar mitigation advisory capabilities | Medium | Medium | Deeper 9-agent integration as unique differentiator; faster iteration cycle; superior ML model training from production data |
| R11 | Audit fatigue from excessive trigger events overwhelming compliance teams | Medium | Medium | Alert consolidation and deduplication; configurable quiet periods; severity-based filtering; digest mode for low-severity events |
| R12 | Training content becomes outdated as regulations and best practices evolve | Medium | Medium | Quarterly content review cycle; version-controlled content library; community-contributed updates with editorial review |

---

## 18. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **ISO 31000** | International standard for risk management -- Guidelines (2018 edition) |
| **SHAP** | SHapley Additive exPlanations -- model-agnostic explainability technique |
| **SMART** | Specific, Measurable, Achievable, Relevant, Time-bound -- milestone quality criteria |
| **RICE** | Reach, Impact, Confidence, Effort -- prioritization framework |
| **CAR** | Corrective Action Request -- formal request to address audit non-conformance |
| **FPIC** | Free, Prior and Informed Consent -- indigenous peoples' right to consent to activities on their territories |
| **ROI** | Return on Investment -- (Value Gained - Cost) / Cost |
| **LP** | Linear Programming -- mathematical optimization technique for resource allocation |
| **Pareto Frontier** | Set of solutions where no objective can be improved without worsening another |
| **XGBoost** | eXtreme Gradient Boosting -- ensemble ML algorithm for classification/regression |
| **LightGBM** | Light Gradient Boosting Machine -- Microsoft's gradient boosting framework |
| **GIN Index** | Generalized Inverted Index -- PostgreSQL index type for full-text search |
| **Competent Authority** | EU Member State authority responsible for EUDR enforcement (Articles 14-16) |

### Appendix B: ISO 31000:2018 Risk Treatment Options Mapping

| ISO 31000 Treatment | EUDR Context | Example Mitigation Measures |
|---------------------|-------------|---------------------------|
| **Avoid** | Decide not to source from high-risk origin | Supplier replacement, commodity substitution, market exit |
| **Reduce (Likelihood)** | Decrease probability of non-compliance | Training, certification enrollment, monitoring enhancement |
| **Reduce (Consequence)** | Decrease impact of non-compliance discovery | Early detection, rapid response protocols, contingency plans |
| **Share** | Transfer risk to third party | Insurance, contractual risk allocation, consortium participation |
| **Retain** | Accept residual risk with monitoring | Documented risk acceptance, enhanced monitoring, contingency budget |

### Appendix C: EUDR Article 11 Mitigation Measure Categories

Per Article 11(2), operators must adopt measures that may include but are not limited to:

| Article 11(2) Clause | Mitigation Category | Agent Feature |
|----------------------|--------------------|----|
| (a) Additional information, data, or documents from suppliers | Supplier information enhancement | Strategy Selector, Capacity Building Manager |
| (b) Independent surveys and audits | Third-party verification | EUDR-024 integration, Strategy Selector |
| (c) Other measures to manage and mitigate risk | Comprehensive mitigation library | Mitigation Measure Library (500+ measures) |

### Appendix D: Mitigation Measure Library Category Summary

| Category | Sub-Categories | Measure Count | Key Sources |
|----------|---------------|---------------|-------------|
| Country Risk | Landscape programs, government partnerships, multi-stakeholder initiatives, trade agreements | 65+ | Tropical Forest Alliance, FCPF, EU FLEGT |
| Supplier Risk | Training, corrective actions, scorecards, site visits, alternative sourcing | 80+ | ISO 9001, FSC guidance, RSPO guidance |
| Commodity Risk | Certification, sustainable production, yield improvement, intercropping, agroforestry | 75+ | FSC, RSPO, Rainforest Alliance, ISCC |
| Corruption Risk | Due diligence, anti-bribery, transparency, whistleblower, payment verification | 55+ | OECD Anti-Bribery Convention, UN Convention against Corruption |
| Deforestation Risk | Satellite monitoring, zero-deforestation commitments, restoration, fire prevention | 70+ | GFW, Hansen GFC, GLAD, Copernicus |
| Indigenous Rights | FPIC implementation, community engagement, benefit-sharing, grievance mechanisms | 50+ | ILO 169, UNDRIP, FSC FPIC guidance |
| Protected Areas | Buffer restoration, encroachment prevention, conservation support, alternative livelihoods | 55+ | WDPA, IUCN Green List, CBD |
| Legal Compliance | Permit support, legal assessment, certification, labour compliance, tax compliance | 60+ | National legislation databases, ILO Conventions |

### Appendix E: Risk Assessment Agent Integration Matrix

| Risk Agent | Data Points Consumed | Event Types | Response Time |
|-----------|---------------------|-------------|---------------|
| EUDR-016 Country Risk | risk_score, risk_level, due_diligence_level, hotspots, governance_index | country_reclassification | < 2s API |
| EUDR-017 Supplier Risk | risk_score, risk_level, risk_factors, compliance_history | risk_spike | < 2s API |
| EUDR-018 Commodity Risk | commodity_profile, deforestation_correlation, seasonal_pattern | profile_update | < 2s API |
| EUDR-019 Corruption | cpi_score, governance_score, corruption_alerts | corruption_alert | < 2s API |
| EUDR-020 Deforestation | alert_severity, proximity, affected_plots, cutoff_status | deforestation_alert | < 5s event |
| EUDR-021 Indigenous Rights | territory_overlap, fpic_status, violation_alerts | rights_violation | < 5s event |
| EUDR-022 Protected Areas | overlap_result, iucn_category, buffer_violations | encroachment_alert | < 5s event |
| EUDR-023 Legal Compliance | compliance_gaps, permit_status, certification_validity | compliance_change | < 2s API |
| EUDR-024 Audit Manager | audit_findings, non_conformances, car_status | nc_detected | < 5s event |

### Appendix F: Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_rma_strategies_recommended_total` | Counter | Strategy recommendations generated |
| 2 | `gl_eudr_rma_strategies_selected_total` | Counter | Strategies selected for implementation |
| 3 | `gl_eudr_rma_plans_created_total` | Counter | Remediation plans created |
| 4 | `gl_eudr_rma_plans_completed_total` | Counter | Remediation plans completed |
| 5 | `gl_eudr_rma_milestones_completed_total` | Counter | Plan milestones completed |
| 6 | `gl_eudr_rma_milestones_overdue_total` | Counter | Plan milestones past due date |
| 7 | `gl_eudr_rma_capacity_enrollments_total` | Counter | Capacity building enrollments by commodity |
| 8 | `gl_eudr_rma_effectiveness_measurements_total` | Counter | Effectiveness tracking measurements |
| 9 | `gl_eudr_rma_trigger_events_total` | Counter | Adaptive management trigger events by type |
| 10 | `gl_eudr_rma_optimization_runs_total` | Counter | Cost-benefit optimization runs |
| 11 | `gl_eudr_rma_reports_generated_total` | Counter | Reports generated by type |
| 12 | `gl_eudr_rma_processing_duration_seconds` | Histogram | Processing latency by operation type |
| 13 | `gl_eudr_rma_ml_inference_duration_seconds` | Histogram | ML model inference latency |
| 14 | `gl_eudr_rma_optimization_duration_seconds` | Histogram | Optimization solver duration |
| 15 | `gl_eudr_rma_errors_total` | Counter | Errors by operation type |
| 16 | `gl_eudr_rma_active_plans` | Gauge | Currently active remediation plans |
| 17 | `gl_eudr_rma_avg_risk_reduction_pct` | Gauge | Average risk reduction across active plans |
| 18 | `gl_eudr_rma_avg_roi` | Gauge | Average ROI across completed plans |

### Appendix G: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EU Deforestation Regulation)
2. ISO 31000:2018 -- Risk management -- Guidelines
3. ISO 14001:2015 -- Environmental management systems -- Requirements with guidance for use
4. ISO 19011:2018 -- Guidelines for auditing management systems
5. EU Deforestation Regulation Guidance Document (European Commission)
6. EUDR Technical Specifications for the Information System
7. Tropical Forest Alliance Operational Guidance for Deforestation-Free Sourcing
8. FSC Controlled Wood Standard (FSC-STD-40-005)
9. RSPO Principles and Criteria for the Production of Sustainable Palm Oil
10. Rainforest Alliance Sustainable Agriculture Standard 2020
11. ISCC EU Certification System Document
12. ILO Convention 169 -- Indigenous and Tribal Peoples Convention
13. United Nations Declaration on the Rights of Indigenous Peoples (UNDRIP)
14. UN Guiding Principles on Business and Human Rights
15. OECD Convention on Combating Bribery of Foreign Public Officials
16. XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)
17. SHAP: A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017)

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
| 1.0.0-draft | 2026-03-11 | GL-ProductManager | Initial draft created: 9 P0 features, 9-agent risk integration (EUDR-016-024), ISO 31000 alignment, 35+ API endpoints, V113 migration schema, 920+ test targets, 18 Prometheus metrics |
