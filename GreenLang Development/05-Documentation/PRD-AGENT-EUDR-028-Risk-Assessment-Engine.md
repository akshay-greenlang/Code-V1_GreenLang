# PRD: AGENT-EUDR-028 -- Risk Assessment Engine

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-028 |
| **Agent ID** | GL-EUDR-RAE-028 |
| **Component** | Risk Assessment Engine Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence (Category 5) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-11 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-11 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 10, 11, 12, 13, 29 |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Article 10 mandates that operators and traders perform a comprehensive risk assessment as the second phase of their due diligence system. This risk assessment must determine whether products placed on the EU market are deforestation-free and legally produced, considering a broad set of criteria enumerated in Article 10(2)(a) through (k). The risk assessment must be documented as part of the Due Diligence Statement (DDS) submitted per Article 12 and must be repeatable, transparent, and auditable under regulatory inspection per Articles 14-16.

The GreenLang platform has built five specialized risk assessment agents -- EUDR-016 Country Risk Evaluator (country-level governance and deforestation rates), EUDR-017 Supplier Risk Scorer (supplier compliance history and certifications), EUDR-018 Commodity Risk Analyzer (commodity-specific deforestation association), EUDR-019 Corruption Index Monitor (corruption perception and rule of law indicators), and EUDR-020 Deforestation Alert System (real-time satellite-based deforestation alerts). These agents each produce high-quality, domain-specific risk scores on a 0-100 scale with confidence intervals. However, Article 10 requires a holistic, composite risk assessment that integrates all these dimensions into a single, defensible risk determination. Today, operators face the following critical gaps:

- **No composite risk score computation**: The five upstream risk agents (EUDR-016 through EUDR-020) each produce independent, single-dimension risk scores. There is no engine that aggregates these scores into a weighted composite risk score that reflects the relative importance of each risk dimension as specified by Article 10(2). Compliance officers must manually combine scores using spreadsheets, introducing inconsistency, subjectivity, and non-reproducibility that would not withstand regulatory scrutiny.

- **No Article 10(2) criteria evaluation**: Article 10(2) enumerates 10+ specific risk criteria that must be evaluated, including supply chain complexity (a), prevalence of deforestation in the country of production (b), risk of mixing with products of unknown origin (f), and risk of circumvention (g). There is no systematic engine that maps each criterion to its data sources, evaluates it against defined thresholds, and produces a criterion-by-criterion assessment record. Operators cannot demonstrate to competent authorities that they have considered every required criterion.

- **No country benchmarking integration**: Article 29 establishes the European Commission's country benchmarking system, classifying countries as Low, Standard, or High risk based on deforestation rates, agricultural expansion trends, and governance indicators. This benchmarking directly affects due diligence obligations: Article 13 provides simplified due diligence for products exclusively from Low-risk countries. There is no engine that integrates the EC's country benchmarking classifications, applies the correct due diligence pathway, and adjusts risk calculations accordingly.

- **No simplified due diligence determination**: Article 13 permits operators to apply simplified due diligence procedures when products originate exclusively from countries classified as Low risk under Article 29, provided additional conditions are met. There is no automated engine that evaluates eligibility for simplified due diligence, determines which Article 10(2) criteria can be reduced, and generates the appropriate documentation for the simplified pathway.

- **No risk level classification with regulatory thresholds**: The regulation implies a risk determination that leads to one of several outcomes: negligible risk (no further action needed), low risk (standard monitoring), standard risk (full due diligence), high risk (enhanced due diligence per Article 11), or critical risk (product cannot be placed on the market). There is no standardized classification engine with configurable thresholds that maps composite scores to risk levels and triggers the appropriate regulatory response.

- **No risk trend analysis**: Article 10(1) requires operators to assess risk "taking into account" the information gathered under Article 9. Risk is not static -- it evolves as new satellite data arrives, supplier compliance histories change, country benchmarkings are updated, and deforestation alerts are issued. There is no temporal analysis engine that tracks how risk scores change over time for the same operator-commodity-supplier combination, identifies trends, and alerts operators to deteriorating risk profiles.

- **No risk decomposition for transparency**: When competent authorities inspect an operator's risk assessment under Articles 14-16, they require full transparency into how the composite risk score was derived. There is no decomposition engine that breaks down a composite score into its constituent factors, shows the weight and contribution of each dimension, and presents the calculation in a format that regulators can independently verify.

- **No risk override with audit trail**: Expert compliance officers may have information not captured by automated risk agents -- for example, a recent site visit, a confidential supplier disclosure, or knowledge of an impending regulatory change. There is no mechanism for authorized users to override the computed risk score with a manual adjustment, record the justification, and maintain a complete audit trail of overrides for regulatory inspection.

- **No DDS-ready risk assessment reports**: Article 12 requires that the DDS contain the results of the risk assessment. There is no report generation engine that formats risk assessment results in the structure required by the EU Information System, including composite scores, criterion-by-criterion evaluations, country benchmarking determinations, and risk level classifications.

Without solving these problems, operators cannot produce defensible, comprehensive, transparent risk assessments that satisfy Article 10 requirements. Manual risk aggregation across five independent risk agents is subjective, non-reproducible, and would not withstand regulatory audit. Operators face penalties of up to 4% of annual EU turnover, goods confiscation, temporary exclusion from public procurement, and public naming under Articles 23-25.

### 1.2 Solution Overview

Agent-EUDR-028: Risk Assessment Engine is the core computation engine for EUDR Article 10 risk assessment. It aggregates risk signals from the five upstream risk assessment agents (EUDR-016 through EUDR-020), computes deterministic weighted composite risk scores, evaluates all Article 10(2) criteria, integrates Article 29 country benchmarking, determines simplified due diligence eligibility per Article 13, classifies risk levels against configurable thresholds, tracks risk trends over time, produces DDS-ready risk assessment reports, and supports expert risk overrides with full audit trail. It operates as a purely deterministic, zero-hallucination computation engine with no LLM in the critical path.

Core capabilities:

1. **Composite Risk Score Calculator** -- Computes weighted composite risk scores using the formula: Composite_Risk = SUM(W_i x S_i x C_i) / SUM(W_i x C_i), where W_i is the configurable weight for risk dimension i, S_i is the score (0-100) from the upstream agent, and C_i is the confidence (0-1) from the upstream agent. Default weights are calibrated to EUDR Article 10(2) criteria importance. All calculations use Decimal arithmetic to prevent floating-point drift. Results are bit-perfect reproducible.

2. **Risk Factor Aggregation Engine** -- Collects, validates, normalizes, and timestamps risk signals from EUDR-016 (country risk), EUDR-017 (supplier risk), EUDR-018 (commodity risk), EUDR-019 (corruption risk), and EUDR-020 (deforestation alerts). Handles missing signals with configurable fallback strategies (use last known value, use dimension average, or flag as incomplete). Produces a unified risk factor input vector for composite scoring.

3. **Article 10(2) Criteria Evaluation Engine** -- Systematically evaluates all Article 10(2) criteria: (a) supply chain complexity, (b) prevalence of deforestation in the country of production, (c) prevalence of forest degradation, (d) risk indicators related to relevant commodities, (e) concerns about the country of production or parts thereof, (f) risk of mixing with products of unknown origin, (g) risk of circumvention of the regulation, (h) history of non-compliance by operators, (i) complementary information from scientific studies, (j) information from indigenous peoples, (k) relevant international agreements. Each criterion is evaluated against its specific data sources, thresholds, and scoring methodology.

4. **Country Benchmarking Integration Engine** -- Integrates the European Commission's Article 29 country benchmarking classifications (Low/Standard/High). Maintains a hot-reloadable country benchmarking database that can be updated within minutes of EC publication. Applies benchmarking multipliers to composite risk scores. Adjusts risk assessment rigor based on country classification.

5. **Simplified Due Diligence Engine** -- Evaluates eligibility for Article 13 simplified due diligence based on: (i) all products originate from countries classified as Low risk, (ii) composite risk score is below the simplified threshold (default < 30), and (iii) no deforestation alerts active for the relevant geolocation. When eligible, reduces the required Article 10(2) criteria evaluation and generates simplified documentation.

6. **Risk Classification and Threshold Engine** -- Classifies composite risk scores into five regulatory tiers: NEGLIGIBLE (0-15), LOW (16-30), STANDARD (31-60), HIGH (61-80), CRITICAL (81-100). Thresholds are configurable per operator and per commodity. Classification triggers regulatory workflows: NEGLIGIBLE/LOW allow market placement, STANDARD requires documentation, HIGH triggers Article 11 mitigation, CRITICAL blocks market placement pending resolution.

7. **Risk Trend Analyzer** -- Tracks risk scores over time for each operator-commodity-supplier-country combination. Computes moving averages, trend direction (improving/stable/deteriorating), rate of change, and projected risk trajectory. Generates trend alerts when risk is deteriorating beyond configurable thresholds. Provides historical risk timeline visualization data.

8. **Risk Report Generator** -- Produces DDS-ready risk assessment reports formatted for the EU Information System per Article 12. Reports include: composite risk score with decomposition, Article 10(2) criterion-by-criterion evaluation, country benchmarking determination, simplified due diligence eligibility assessment, risk level classification, risk trend summary, and provenance hashes for every input and calculation. Supports JSON (machine-readable) and PDF (human-readable) output formats.

9. **Risk Override and Manual Adjustment Engine** -- Allows authorized users (Compliance Officer, Admin) to override computed risk scores with manual adjustments. Requires mandatory justification text, supporting evidence references, and expiration date. Maintains complete audit trail of all overrides including original score, override score, justification, actor, and timestamp. Overrides are flagged in DDS reports for regulatory transparency.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Composite risk calculation accuracy | 100% match with manual reference calculations | Validation against 500+ known-value golden tests |
| Bit-perfect reproducibility | Same input produces identical output across runs | Reproducibility test suite with hash comparison |
| Article 10(2) criteria coverage | All 10+ criteria evaluated per assessment | Criteria completeness audit |
| Risk classification consistency | 100% consistent classification for same input | Idempotency tests across 1,000 assessments |
| Country benchmarking integration | < 5 minutes to apply new EC benchmarking list | Benchmarking update latency test |
| Simplified DD eligibility accuracy | 100% correct eligibility determination | Cross-validation against manual determination |
| Assessment throughput | 500+ assessments per minute | Load test under sustained throughput |
| Risk report generation time | < 5 seconds per DDS-ready report | Report generation benchmarks |
| Risk trend data coverage | 12+ months of historical tracking per combination | Data retention validation |
| Override audit trail completeness | 100% of overrides fully documented | Audit trail gap analysis |
| Factor aggregation latency | < 500ms to aggregate 5 upstream agent scores | p99 latency benchmarks |
| EUDR regulatory acceptance | 100% of risk reports accepted in DDS submissions | EU Information System validation |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated risk assessment automation market of 2-4 billion EUR as operators seek defensible, reproducible risk scoring systems to satisfy Article 10 audit requirements.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities requiring automated, multi-dimensional risk assessment engines, estimated at 500M-900M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 30-50M EUR in risk assessment module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) of EUDR-regulated commodities requiring defensible risk assessments
- Multinational food and beverage companies sourcing cocoa, coffee, palm oil, and soya from high-risk regions
- Timber and paper industry operators with complex, multi-country supply chains
- Rubber and cattle product importers facing high regulatory scrutiny

**Secondary:**
- Customs brokers and freight forwarders handling EUDR-regulated goods on behalf of clients
- Commodity traders and intermediaries with pass-through due diligence obligations
- Compliance consultants advising multiple operators on EUDR risk assessment methodology
- Certification bodies (FSC, RSPO, Rainforest Alliance) offering risk assessment as a value-added service
- SME importers (1,000-10,000 shipments/year) preparing for June 30, 2026 enforcement

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Spreadsheet | No cost; familiar process | Subjective, non-reproducible, fails audit; 20-40 hours per assessment | Deterministic, auditable, < 5 seconds per assessment |
| Generic ESG risk platforms (EcoVadis, Sustainalytics) | Broad ESG coverage; market recognition | Not EUDR-specific; no Article 10(2) criteria mapping; no country benchmarking; no DDS integration | Purpose-built for EUDR Article 10; full criteria coverage; DDS-ready output |
| Niche EUDR compliance tools (Preferred by Nature) | Commodity expertise; regulatory knowledge | Single-commodity; manual risk scoring; no upstream agent integration | All 7 commodities; automated 5-agent aggregation; deterministic scoring |
| Management consulting firms | Deep regulatory expertise; bespoke analysis | EUR 50K-200K per assessment; slow (weeks); not scalable; no reproducibility | Automated, EUR 5K-20K per year; instant; fully reproducible |
| In-house custom builds | Tailored to organization; full control | 12-18 month build time; no regulatory updates; no upstream agent ecosystem | Ready now; continuous regulatory updates; integrated with 27 upstream agents |

### 2.4 Differentiation Strategy

1. **Five-agent upstream integration** -- No competitor integrates purpose-built country, supplier, commodity, corruption, and deforestation risk agents into a single composite scoring engine. GreenLang's Risk Assessment Engine consumes validated, confidence-scored signals from 5 specialized agents, each with 500+ tests and production-grade reliability.

2. **Full Article 10(2) criteria mapping** -- Every criterion in Article 10(2)(a) through (k) is systematically mapped to data sources, evaluated against thresholds, and documented in the risk report. No competitor provides this level of regulatory fidelity.

3. **Deterministic, zero-hallucination computation** -- All risk calculations use deterministic weighted formulas with Decimal arithmetic. No LLM is used in the critical path. Results are bit-perfect reproducible across runs, a critical requirement for regulatory audit.

4. **Country benchmarking and simplified DD automation** -- Automatic integration of EC Article 29 country benchmarking with hot-reload capability, plus automated Article 13 simplified due diligence eligibility determination. This reduces compliance burden for low-risk commodity flows.

5. **DDS-native output** -- Risk assessment reports are formatted directly for EU Information System DDS submission, eliminating manual transcription and format conversion.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to produce defensible Article 10 risk assessments | 100% of customers pass Article 10 regulatory audits | Q2 2026 |
| BG-2 | Reduce risk assessment time from days to seconds | > 99% reduction in assessment time | Q2 2026 |
| BG-3 | Eliminate subjective, non-reproducible risk scoring | 100% bit-perfect reproducibility across all assessments | Q2 2026 |
| BG-4 | Automate simplified due diligence for low-risk commodity flows | 100% of eligible flows receive automated simplified DD | Q3 2026 |
| BG-5 | Become the reference risk assessment engine for EUDR compliance | 500+ enterprise customers | Q4 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Composite risk scoring | Aggregate 5 upstream risk dimensions into a single, weighted, confidence-adjusted composite score |
| PG-2 | Full Article 10(2) evaluation | Systematically evaluate all 10+ criteria with documented methodology and data sources |
| PG-3 | Country benchmarking integration | Apply EC Article 29 Low/Standard/High classifications with hot-reload capability |
| PG-4 | Simplified DD automation | Determine Article 13 eligibility and generate simplified documentation |
| PG-5 | Risk classification | Classify composite scores into 5 regulatory tiers with configurable thresholds |
| PG-6 | Risk trend analysis | Track risk evolution over time with trend detection and alerting |
| PG-7 | DDS-ready reporting | Generate risk assessment reports in EU Information System format |
| PG-8 | Expert override support | Allow manual risk adjustments with mandatory justification and audit trail |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Assessment throughput | 500+ composite risk assessments per minute |
| TG-2 | Single assessment latency | < 200ms p95 for composite score computation |
| TG-3 | Report generation latency | < 5 seconds p95 per DDS-ready report |
| TG-4 | API response time | < 200ms p95 for standard queries |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |
| TG-7 | Memory efficiency | < 512 MB for 10,000 concurrent risk assessments in memory |
| TG-8 | Database query performance | < 50ms p95 for risk history lookups |

---

## 4. User Personas

### Persona 1: Compliance Officer -- Sofia (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of EUDR Compliance at a large EU food manufacturer |
| **Company** | 8,000 employees, importing cocoa, coffee, palm oil, and soya from 25+ countries |
| **EUDR Pressure** | Board-level mandate to achieve full Article 10 compliance; competent authority audit expected in Q3 2026 |
| **Pain Points** | Manually aggregates risk scores from 5 different systems into a spreadsheet; cannot demonstrate reproducibility to auditors; spends 3 days per commodity per quarter on risk assessment; no visibility into Article 10(2) criteria coverage; no trend data showing risk improvement |
| **Goals** | Automated, defensible composite risk scoring; full Article 10(2) criteria documentation; DDS-ready reports; trend analysis showing compliance improvement over time |
| **Technical Skill** | Moderate -- comfortable with web applications, dashboards, and report generation |

### Persona 2: Risk Analyst -- Henrik (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Risk Analyst at an EU timber importer |
| **Company** | 1,200 employees, importing tropical and temperate wood from 15+ countries including several High-risk regions |
| **EUDR Pressure** | Must evaluate risk for 500+ supplier-country combinations; needs to identify which supply chains qualify for simplified DD under Article 13 |
| **Pain Points** | No systematic way to evaluate all Article 10(2) criteria; cannot efficiently identify which suppliers/countries drive the most risk; no mechanism to track how risk changes after mitigation measures are implemented; manually maintains risk override spreadsheet |
| **Goals** | Granular risk decomposition showing contribution of each dimension; automated simplified DD eligibility check; risk trend tracking to measure mitigation effectiveness; documented override capability with audit trail |
| **Technical Skill** | High -- comfortable with data analysis tools, APIs, risk models, and statistical concepts |

### Persona 3: Regulatory Affairs Director -- Dr. Martens (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Director of Regulatory Affairs at a multinational palm oil processor |
| **Company** | 5,000 employees, sourcing from 300+ plantations across Indonesia, Malaysia, and Colombia |
| **EUDR Pressure** | Must prepare for competent authority inspections; needs to demonstrate that the company's risk assessment methodology is systematic, comprehensive, and aligned with EUDR requirements |
| **Pain Points** | Cannot demonstrate to auditors how composite risk scores are computed; no criterion-by-criterion Article 10(2) evaluation record; no integration with EC country benchmarking updates; risk reports are manually formatted and may not match DDS requirements |
| **Goals** | Transparent, decomposable risk scoring that auditors can verify independently; automated country benchmarking integration; DDS-compliant risk reports; complete audit trail of every risk assessment decision |
| **Technical Skill** | Moderate -- legal and regulatory background, comfortable with compliance systems and audit documentation |

### Persona 4: External Auditor -- Ms. Lindqvist (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm specializing in supply chain compliance |
| **EUDR Pressure** | Must verify that operators' risk assessments satisfy Article 10 requirements during regulatory audits; needs to independently reproduce risk calculations |
| **Pain Points** | Operators provide risk assessment documentation in inconsistent formats; cannot independently verify composite score calculations; no standardized way to check Article 10(2) criteria coverage; override justifications are often missing or inadequate |
| **Goals** | Access to read-only risk assessment details with full calculation provenance; ability to verify composite scores independently using the formula and inputs; criterion-by-criterion evaluation records; complete override audit trail with justifications |
| **Technical Skill** | Moderate -- comfortable with audit software, document review, and mathematical verification |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 10(1)** | Operators shall assess risk of non-compliance of relevant commodities and products using information gathered under Article 9 | Composite Risk Score Calculator aggregates all available risk signals into a defensible risk determination |
| **Art. 10(2)(a)** | Risk of deforestation, forest degradation, or non-compliance with relevant legislation in the country of production, including the prevalence of deforestation and forest degradation | Country Risk Evaluator (EUDR-016) input aggregated via Risk Factor Aggregation Engine |
| **Art. 10(2)(b)** | Prevalence of deforestation and forest degradation in the country of production or parts thereof | Deforestation Alert System (EUDR-020) input + country-level deforestation rate data |
| **Art. 10(2)(c)** | Prevalence of deforestation and forest degradation in the country of origin of the relevant commodity or product | Commodity Risk Analyzer (EUDR-018) input mapped to country-commodity deforestation rates |
| **Art. 10(2)(d)** | Risk indicators related to relevant commodities or products | Commodity Risk Analyzer (EUDR-018) commodity-specific risk indicators |
| **Art. 10(2)(e)** | Concerns about the country of production or parts thereof, including the level of corruption, prevalence of document and data falsification | Corruption Index Monitor (EUDR-019) + Country Risk Evaluator (EUDR-016) governance scores |
| **Art. 10(2)(f)** | Risk of mixing with products of unknown origin | Article 10(2) Criteria Evaluator assesses mixing risk based on chain of custody model and supply chain topology |
| **Art. 10(2)(g)** | Risk of circumvention of the regulation or its implementing or delegated acts | Article 10(2) Criteria Evaluator assesses circumvention risk based on routing patterns and country risk |
| **Art. 10(2)(h)** | History of compliance of the supplier | Supplier Risk Scorer (EUDR-017) compliance history and certification tracking |
| **Art. 10(2)(i)** | Complementary information on compliance with the regulation, including information made available under certification or other third-party verified schemes | Supplier Risk Scorer (EUDR-017) certification status + Article 10(2) Criteria Evaluator |
| **Art. 10(2)(j)** | Information from subnational assessment, including from indigenous peoples and local communities | Article 10(2) Criteria Evaluator integrates subnational assessment data from EUDR-021 Indigenous Rights Checker |
| **Art. 10(2)(k)** | Information from relevant international agreements to which the country of production is a party | Article 10(2) Criteria Evaluator evaluates international agreement compliance (Paris Agreement, CBD) |
| **Art. 11** | Risk mitigation measures proportionate to risk level | Risk Classification Engine determines risk level that triggers Article 11 mitigation requirements |
| **Art. 12** | Due diligence statement content requirements | Risk Report Generator produces DDS-formatted risk assessment results |
| **Art. 13** | Simplified due diligence for low-risk country products | Simplified Due Diligence Engine evaluates eligibility and generates simplified documentation |
| **Art. 29** | Country benchmarking by the European Commission | Country Benchmarking Integration Engine maintains and applies EC Low/Standard/High classifications |
| **Art. 31** | Record keeping for 5 years | All risk assessments, overrides, and reports stored with immutable provenance hashes for 5+ years |

### 5.2 Article 10(2) Risk Assessment Criteria

The following table details all Article 10(2) criteria and their evaluation methodology within the Risk Assessment Engine:

| Criterion | Article | Description | Data Source(s) | Scoring Method | Weight (Default) |
|-----------|---------|-------------|----------------|----------------|------------------|
| Country deforestation prevalence | 10(2)(a-b) | Rate and prevalence of deforestation and forest degradation in the country of production | EUDR-016 Country Risk Evaluator, FAO Global Forest Resources Assessment, Global Forest Watch | 0-100 scale based on annual deforestation rate percentile ranking | 0.20 |
| Commodity risk indicators | 10(2)(c-d) | Commodity-specific deforestation association and risk indicators | EUDR-018 Commodity Risk Analyzer, peer-reviewed deforestation driver studies | 0-100 scale based on commodity-deforestation correlation strength | 0.15 |
| Supplier compliance history | 10(2)(h-i) | Historical compliance record, certification status, third-party verification | EUDR-017 Supplier Risk Scorer, FSC/RSPO/RA certification databases | 0-100 scale inverse of compliance score (higher = more risk) | 0.20 |
| Corruption and governance concerns | 10(2)(e) | Corruption perception, rule of law, regulatory quality, document falsification risk | EUDR-019 Corruption Index Monitor, TI CPI, World Bank WGI | 0-100 scale based on governance indicator percentile | 0.10 |
| Deforestation alerts | 10(2)(a-c) | Active deforestation alerts within or near production geolocations | EUDR-020 Deforestation Alert System, Sentinel-2, Landsat, GFW | 0-100 scale based on alert proximity, severity, and recency | 0.20 |
| Supply chain complexity | 10(2)(a) | Number of tiers, intermediaries, processing steps, and custody model changes | Supply chain graph from EUDR-001, tier depth, node count | 0-100 scale based on complexity index formula | 0.05 |
| Mixing risk | 10(2)(f) | Risk that compliant products are mixed with products of unknown origin | Chain of custody model, mass balance analysis, processing facility audit | 0-100 scale based on custody model and mixing indicators | 0.05 |
| Circumvention risk | 10(2)(g) | Risk that products are routed through low-risk countries to circumvent regulation | Trade flow anomaly detection, re-export patterns, country of dispatch vs. origin | 0-100 scale based on circumvention indicator analysis | 0.05 |
| Indigenous peoples information | 10(2)(j) | Information from subnational assessment, indigenous peoples, local communities | EUDR-021 Indigenous Rights Checker, FPIC documentation | Qualitative flag (elevated risk if concerns identified) | Qualitative modifier |
| International agreements | 10(2)(k) | Compliance with relevant international agreements (Paris Agreement, CBD, CITES) | Country treaty ratification status, compliance monitoring reports | Qualitative flag (elevated risk if non-compliance identified) | Qualitative modifier |

### 5.3 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for all deforestation-free verification; risk scores reference this cutoff |
| June 29, 2023 | Regulation entered into force | Legal basis for all risk assessment criteria |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Risk assessment engine must be operational for all large operator assessments |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; engine must scale to handle increased assessment volume |
| Quarterly (ongoing) | EC country benchmarking updates | Country Benchmarking Integration Engine must hot-reload updated classifications |
| Annual (ongoing) | TI Corruption Perception Index update | Corruption dimension risk scores must be recalculated |
| Biannual (ongoing) | World Bank WGI update | Governance dimension risk scores must be recalculated |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-6 form the core risk computation engine; Features 7-9 form the reporting and adjustment layer that delivers risk intelligence to users and regulators.

**P0 Features 1-6: Core Risk Computation Engine**

---

#### Feature 1: Composite Risk Score Calculator

**User Story:**
```
As a compliance officer,
I want a single, weighted composite risk score computed from all available risk dimensions,
So that I can make a defensible risk determination for Article 10 and include it in my DDS.
```

**Acceptance Criteria:**
- [ ] Computes composite risk using the formula: Composite_Risk = SUM(W_i x S_i x C_i) / SUM(W_i x C_i)
- [ ] Supports 8 quantitative risk dimensions with configurable weights (country, commodity, supplier, deforestation, corruption, supply_chain_complexity, mixing, circumvention)
- [ ] Supports 2 qualitative risk modifiers (indigenous_peoples, international_agreements) that elevate risk when flagged
- [ ] Uses Decimal arithmetic throughout to prevent floating-point drift
- [ ] Accepts configurable weight profiles per operator and per commodity
- [ ] Default weights: W_country=0.20, W_commodity=0.15, W_supplier=0.20, W_deforestation=0.20, W_corruption=0.10, W_complexity=0.05, W_mixing=0.05, W_circumvention=0.05
- [ ] Validates that weights sum to 1.0 (within tolerance of 0.001)
- [ ] Produces composite score on 0-100 scale with 2 decimal places
- [ ] Includes confidence-weighted aggregation: dimensions with higher confidence contribute more to the composite
- [ ] Generates SHA-256 provenance hash of all inputs and computed output for audit trail
- [ ] Achieves bit-perfect reproducibility: same inputs always produce identical output

**Non-Functional Requirements:**
- Performance: < 10ms per composite score computation
- Accuracy: 100% match with reference manual calculations (tested against 500+ golden values)
- Reproducibility: Bit-perfect across runs, environments, and Python versions
- Auditability: Complete calculation provenance with SHA-256 hash chain

**Dependencies:**
- EUDR-016 Country Risk Evaluator (country risk scores)
- EUDR-017 Supplier Risk Scorer (supplier risk scores)
- EUDR-018 Commodity Risk Analyzer (commodity risk scores)
- EUDR-019 Corruption Index Monitor (corruption risk scores)
- EUDR-020 Deforestation Alert System (deforestation risk scores)

**Estimated Effort:** 2 weeks (1 senior backend engineer)

**Edge Cases:**
- All input scores are 0 -- Composite score is 0 (NEGLIGIBLE)
- All input scores are 100 -- Composite score is 100 (CRITICAL)
- One dimension has 0 confidence -- That dimension is excluded from composite (zero weight contribution)
- All dimensions have 0 confidence -- Return error: "insufficient data for risk assessment"
- Weights do not sum to 1.0 -- Reject configuration with validation error
- Missing upstream agent score -- Use fallback strategy (configurable: last known, dimension average, or fail)
- Qualitative modifier flags active -- Apply configurable penalty (default: +10 points to composite)

---

#### Feature 2: Risk Factor Aggregation Engine

**User Story:**
```
As a risk analyst,
I want all risk signals from the 5 upstream agents collected, validated, and normalized into a unified input vector,
So that the composite scoring engine has clean, consistent, timestamped inputs for every risk dimension.
```

**Acceptance Criteria:**
- [ ] Collects risk scores from EUDR-016 (country), EUDR-017 (supplier), EUDR-018 (commodity), EUDR-019 (corruption), EUDR-020 (deforestation)
- [ ] Validates each input score: must be numeric, 0-100 range, with confidence 0-1
- [ ] Normalizes scores to consistent 0-100 scale if upstream agents use different ranges
- [ ] Timestamps each input with collection time for freshness tracking
- [ ] Handles missing inputs with configurable fallback strategies: USE_LAST_KNOWN (use most recent historical value), USE_DEFAULT (use dimension average from reference data), FAIL (reject assessment as incomplete)
- [ ] Detects stale inputs (score older than configurable threshold, default 30 days) and flags as degraded confidence
- [ ] Resolves conflicting signals (e.g., country risk LOW but deforestation alert CRITICAL) by preserving both and letting the composite formula weight them
- [ ] Produces a structured RiskFactorVector with all dimensions, scores, confidences, timestamps, and freshness indicators
- [ ] Logs all aggregation decisions for audit trail
- [ ] Supports batch aggregation for multiple operator-commodity-supplier combinations

**Non-Functional Requirements:**
- Latency: < 500ms to aggregate all 5 upstream agent scores (including database lookups)
- Reliability: Graceful degradation when 1-2 upstream agents are unavailable
- Consistency: Deterministic aggregation output for same inputs

**Dependencies:**
- EUDR-016 through EUDR-020 risk assessment agents (API or database integration)
- PostgreSQL for historical score storage and fallback lookups

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- All 5 upstream agents unavailable -- Return error: "no risk data available"
- Only 1 of 5 agents provides data -- Compute composite with available data, flag as "partial assessment" with reduced confidence
- Upstream agent returns score outside 0-100 range -- Clamp to 0-100 and log warning
- Upstream agent returns confidence > 1.0 or < 0 -- Clamp to 0-1 and log warning
- Two consecutive scores from same agent differ by > 50 points -- Flag as anomaly for review

---

#### Feature 3: Article 10(2) Criteria Evaluation Engine

**User Story:**
```
As a compliance officer,
I want every Article 10(2) criterion systematically evaluated and documented,
So that I can demonstrate to competent authorities that my risk assessment is comprehensive and compliant.
```

**Acceptance Criteria:**
- [ ] Evaluates all 10 Article 10(2) criteria listed in Section 5.2
- [ ] Maps each criterion to its specific data source(s) and scoring methodology
- [ ] Produces a criterion-by-criterion evaluation record with: criterion ID, criterion text, evaluation result (score 0-100 or qualitative flag), data sources used, evaluation methodology, evidence references
- [ ] Flags criteria with insufficient data as "not fully evaluated" with remediation guidance
- [ ] Computes a criteria coverage score (% of criteria fully evaluated)
- [ ] Supports commodity-specific criterion weights (e.g., mixing risk is higher for palm oil than for timber)
- [ ] Integrates with EUDR-021 Indigenous Rights Checker for criterion (j) evaluation
- [ ] Integrates with country treaty databases for criterion (k) evaluation
- [ ] Generates Article 10(2) compliance summary showing pass/fail per criterion
- [ ] All evaluations are deterministic and reproducible

**Non-Functional Requirements:**
- Completeness: All 10 criteria evaluated for every standard assessment
- Latency: < 2 seconds for full 10-criterion evaluation
- Auditability: Every criterion evaluation includes methodology documentation

**Dependencies:**
- Risk Factor Aggregation Engine (Feature 2) for upstream risk scores
- EUDR-001 Supply Chain Mapping Master for supply chain complexity data
- EUDR-009 Chain of Custody Agent for mixing risk assessment
- EUDR-021 Indigenous Rights Checker for criterion (j) data

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- Indigenous peoples criterion (j) -- no indigenous communities in production area: mark as "not applicable, low risk"
- International agreements criterion (k) -- country not party to relevant agreement: elevate risk for that criterion
- Supply chain complexity criterion (a) -- supply chain not yet mapped: flag as "insufficient data, elevated risk"
- Criterion evaluation produces conflicting signals -- document both signals and apply conservative (higher risk) interpretation

---

#### Feature 4: Country Benchmarking Integration Engine (Article 29)

**User Story:**
```
As a compliance officer,
I want the EC's Article 29 country benchmarking classifications automatically integrated into my risk assessment,
So that my risk scoring reflects the official EU country risk determinations and I can identify simplified DD eligibility.
```

**Acceptance Criteria:**
- [ ] Maintains a database of country benchmarking classifications: LOW, STANDARD, HIGH
- [ ] Supports hot-reload: new EC benchmarking lists applied within 5 minutes of publication without service restart
- [ ] Applies benchmarking multipliers to composite risk: LOW = 0.7x, STANDARD = 1.0x, HIGH = 1.5x (configurable)
- [ ] Handles multi-country supply chains: if any origin country is HIGH risk, the overall benchmarking is HIGH
- [ ] Handles subnational benchmarking: Article 29(4) allows EC to benchmark parts of countries differently
- [ ] Tracks benchmarking history: records when a country's classification changed and recalculates affected assessments
- [ ] Generates country benchmarking report showing all origin countries with their classifications
- [ ] Flags products from newly reclassified HIGH-risk countries for immediate re-assessment
- [ ] Supports "pending" classification for countries not yet benchmarked by EC (treated as STANDARD)
- [ ] Integrates with EUDR-016 Country Risk Evaluator for additional country-level data

**Non-Functional Requirements:**
- Data Freshness: Country benchmarking database updated within 24 hours of EC publication
- Lookup Performance: < 5ms per country classification lookup
- Coverage: All 195+ countries with valid ISO 3166-1 alpha-2 codes

**Dependencies:**
- EC country benchmarking publication feed (official EU sources)
- EUDR-016 Country Risk Evaluator for supplementary country data
- PostgreSQL for benchmarking database

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Country not yet benchmarked by EC -- Treat as STANDARD, flag as "pending classification"
- Country reclassified from LOW to HIGH -- Trigger re-assessment of all active assessments for that country
- Product sourced from 3 countries (LOW, STANDARD, HIGH) -- Overall benchmarking is HIGH (highest-risk-wins)
- Subnational benchmarking applied -- Use subnational classification for plots within that region
- EC publishes partial update (some countries only) -- Merge with existing classifications

---

#### Feature 5: Simplified Due Diligence Engine (Article 13)

**User Story:**
```
As a risk analyst,
I want the system to automatically determine whether a product qualifies for simplified due diligence under Article 13,
So that I can reduce compliance burden for low-risk commodity flows while maintaining regulatory compliance.
```

**Acceptance Criteria:**
- [ ] Evaluates simplified DD eligibility based on three conditions: (i) all origin countries classified as LOW risk per Article 29, (ii) composite risk score < 30 (configurable threshold), (iii) no active deforestation alerts for relevant geolocations
- [ ] When eligible, generates simplified DD assessment that reduces Article 10(2) criteria evaluation to the minimum required set
- [ ] Documents simplified DD eligibility determination with evidence for each condition
- [ ] Automatically revokes simplified DD eligibility if conditions change (country reclassified, new deforestation alert)
- [ ] Generates simplified DDS documentation per Article 13 requirements
- [ ] Tracks simplified DD usage per operator for regulatory reporting
- [ ] Prevents simplified DD for products with any HIGH or CRITICAL risk dimension score (> 60)
- [ ] Provides clear audit trail showing why simplified DD was granted or denied
- [ ] Supports operator-level configuration to opt out of simplified DD (always use full assessment)

**Non-Functional Requirements:**
- Accuracy: 100% correct eligibility determination (tested against 200+ scenarios)
- Latency: < 100ms for eligibility check
- Compliance: Full alignment with Article 13 requirements as currently published

**Dependencies:**
- Country Benchmarking Integration Engine (Feature 4) for country classifications
- Composite Risk Score Calculator (Feature 1) for overall risk score
- EUDR-020 Deforestation Alert System for active alert status

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Product from LOW-risk country but with active deforestation alert -- Deny simplified DD
- Multi-country product where all countries are LOW risk -- Eligible for simplified DD
- Country reclassified from LOW to STANDARD mid-assessment -- Revoke simplified DD, switch to full assessment
- Operator opts out of simplified DD -- Always perform full assessment regardless of eligibility
- Composite risk is 29.99 (just below threshold) -- Eligible; document margin
- Composite risk is 30.01 (just above threshold) -- Not eligible; document margin

---

#### Feature 6: Risk Classification and Threshold Engine

**User Story:**
```
As a compliance officer,
I want composite risk scores classified into clear regulatory tiers with defined thresholds,
So that I know exactly what due diligence actions are required for each risk level.
```

**Acceptance Criteria:**
- [ ] Classifies composite risk scores into 5 tiers: NEGLIGIBLE (0-15), LOW (16-30), STANDARD (31-60), HIGH (61-80), CRITICAL (81-100)
- [ ] Supports configurable tier thresholds per operator and per commodity
- [ ] Maps each tier to a regulatory action: NEGLIGIBLE/LOW = market placement allowed, STANDARD = full documentation required, HIGH = Article 11 mitigation required, CRITICAL = market placement blocked pending resolution
- [ ] Applies hysteresis to prevent classification oscillation: a score must cross the threshold by > 2 points to trigger reclassification
- [ ] Generates classification change events when an operator-commodity-supplier combination changes tier
- [ ] Tracks classification distribution: count and percentage of assessments in each tier
- [ ] Supports bulk classification for portfolio-level risk overview
- [ ] Provides classification confidence based on input data quality
- [ ] Documents classification methodology for regulatory transparency

**Non-Functional Requirements:**
- Consistency: 100% consistent classification for identical inputs
- Performance: < 1ms per classification decision
- Configurability: Thresholds adjustable without code deployment

**Dependencies:**
- Composite Risk Score Calculator (Feature 1) for input scores

**Estimated Effort:** 1 week (1 backend engineer)

**Edge Cases:**
- Score exactly on threshold boundary (e.g., 30.00) -- Classify in the lower tier (LOW, not STANDARD)
- Score oscillates between 29 and 31 across consecutive assessments -- Hysteresis prevents flip-flopping
- Custom thresholds overlap (operator configures LOW: 0-40, STANDARD: 30-60) -- Reject configuration with validation error
- Score is NaN or null -- Return error: "cannot classify undefined score"

---

**P0 Features 7-9: Reporting and Adjustment Layer**

> Features 7, 8, and 9 are P0 launch blockers. Without trend analysis, DDS-ready reports, and expert override capability, the core computation engine cannot deliver complete regulatory value. These features are the delivery mechanism through which compliance officers, regulators, and auditors interact with the risk engine.

---

#### Feature 7: Risk Trend Analysis

**User Story:**
```
As a risk analyst,
I want to track how risk scores for a specific operator-commodity-supplier combination change over time,
So that I can identify improving or deteriorating risk profiles and measure the effectiveness of mitigation actions.
```

**Acceptance Criteria:**
- [ ] Stores every risk assessment result with timestamp for historical tracking
- [ ] Computes 30-day, 90-day, and 12-month moving averages for composite and per-dimension scores
- [ ] Determines trend direction: IMPROVING (score decreasing by > 5 points over window), STABLE (change < 5 points), DETERIORATING (score increasing by > 5 points over window)
- [ ] Computes rate of change: points per month for composite score
- [ ] Generates trend alerts when: (a) risk is deteriorating for 3+ consecutive assessments, (b) risk crosses a tier threshold, (c) any single dimension spikes by > 20 points
- [ ] Provides historical timeline data for visualization (timestamp, composite score, per-dimension scores, risk level)
- [ ] Supports filtering by operator, commodity, supplier, country, and date range
- [ ] Retains trend data for minimum 5 years per EUDR Article 31 record-keeping requirement
- [ ] Supports batch trend analysis for portfolio-level risk monitoring

**Non-Functional Requirements:**
- Storage: Efficient time-series storage using TimescaleDB hypertables
- Query Performance: < 200ms for 12-month trend query
- Alerting: Trend alerts delivered within 5 minutes of detection

**Dependencies:**
- Composite Risk Score Calculator (Feature 1) for assessment results
- TimescaleDB for time-series storage
- OBS-004 Alerting Platform for alert delivery

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- First assessment for a combination (no history) -- Return "insufficient data for trend analysis"
- Gap in assessment history (no assessment for 6+ months) -- Flag gap and compute trend only over available data
- Assessment frequency changes (monthly to weekly) -- Normalize trend calculations to fixed time windows
- Operator changes risk weight configuration -- Note configuration change in trend timeline; do not retroactively recalculate

---

#### Feature 8: Risk Report Generator (DDS-Ready)

**User Story:**
```
As a compliance officer,
I want to generate DDS-ready risk assessment reports that contain all the information required by Article 12,
So that I can include the risk assessment results directly in my Due Diligence Statement submission.
```

**Acceptance Criteria:**
- [ ] Generates risk assessment reports containing: composite risk score with full decomposition, Article 10(2) criterion-by-criterion evaluation, country benchmarking determination, simplified DD eligibility assessment, risk level classification, risk trend summary (if history available), override records (if any)
- [ ] Produces JSON output formatted for EU Information System DDS submission
- [ ] Produces PDF output with executive summary, detailed findings, methodology documentation, and evidence references
- [ ] Includes SHA-256 provenance hash covering all inputs and calculations
- [ ] Includes metadata: assessment ID, operator ID, commodity, assessment date, report generation timestamp
- [ ] Validates report against DDS schema before finalizing
- [ ] Supports multi-language output: EN (default), FR, DE, ES, PT
- [ ] Supports batch report generation for multiple assessments
- [ ] Embeds risk decomposition charts (score breakdown by dimension) in PDF reports
- [ ] Includes regulatory reference citations for every criterion evaluation

**Non-Functional Requirements:**
- Generation Speed: < 5 seconds per report (JSON), < 15 seconds per report (PDF)
- Compliance: 100% of required DDS fields populated
- Schema Validation: 100% pass rate against EU Information System DDS specification

**Dependencies:**
- Features 1-6 (all computation features for report content)
- EU Information System DDS schema specification
- PDF generation library (WeasyPrint or ReportLab)
- GL-EUDR-APP DDS Reporting Engine for integration

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend engineer)

**Edge Cases:**
- Assessment with overrides -- Include both original and overridden scores in report with justification
- Simplified DD assessment -- Generate abbreviated report with simplified DD justification
- Multi-commodity assessment -- Generate separate report per commodity with cross-reference
- Missing trend data (first assessment) -- Omit trend section with note "insufficient history"
- Report requested in unsupported language -- Fall back to English with warning

---

#### Feature 9: Risk Override and Manual Adjustment

**User Story:**
```
As a senior compliance officer,
I want to override the computed risk score with a manual adjustment when I have information not captured by automated agents,
So that I can incorporate expert judgment while maintaining a complete audit trail for regulatory inspection.
```

**Acceptance Criteria:**
- [ ] Allows authorized users (roles: Compliance Officer, Admin) to override composite risk score and/or individual dimension scores
- [ ] Requires mandatory fields for every override: justification text (min 50 characters), supporting evidence references (document IDs or URLs), expiration date (max 12 months from override date)
- [ ] Override can increase or decrease the risk score
- [ ] Maintains complete audit trail: original computed score, override score, override reason, supporting evidence, actor (user ID), timestamp, expiration date
- [ ] Flags overridden assessments in DDS reports with explicit "MANUAL OVERRIDE APPLIED" notation
- [ ] Expired overrides automatically revert to the most recent computed score
- [ ] Supports override of individual dimension scores (not just composite)
- [ ] Provides override history view showing all overrides for a given operator-commodity-supplier combination
- [ ] Sends notification to Admin when overrides are applied or expire
- [ ] Limits override score change to +/- 40 points from computed score (configurable) to prevent abuse

**Non-Functional Requirements:**
- Authorization: Only Compliance Officer and Admin roles can apply overrides
- Audit: Every override recorded with immutable audit trail
- Transparency: Overrides visible to auditors in read-only mode

**Dependencies:**
- SEC-001 JWT Authentication for user identity
- SEC-002 RBAC Authorization for permission enforcement
- SEC-005 Centralized Audit Logging for override audit trail

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Override applied to an assessment that is subsequently recalculated -- Override persists until expiration; recalculated score stored alongside
- Multiple overrides on same assessment -- Most recent override wins; all overrides in audit trail
- Override score exceeds 100 or drops below 0 -- Clamp to valid range
- Override justification is too short (< 50 characters) -- Reject with validation error
- Override expires while assessment is being used in active DDS -- Trigger re-assessment notification
- Admin revokes an override before expiration -- Record revocation in audit trail with reason

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Predictive Risk Scoring
- Use historical risk trends to project future risk scores for a 3-6 month horizon
- Identify operators at risk of crossing tier thresholds before it happens
- Generate proactive mitigation recommendations based on projected risk trajectory
- Use time-series forecasting models (ARIMA, Prophet) on historical risk data

#### Feature 11: Peer Benchmarking
- Compare an operator's risk profile against anonymized industry peers
- Show percentile ranking for composite and per-dimension risk scores
- Identify dimensions where operator is significantly worse than peers
- Provide improvement recommendations based on peer best practices

#### Feature 12: Automated Risk Mitigation Recommendation Engine
- Based on risk decomposition, automatically suggest specific mitigation actions
- Rank recommendations by expected risk reduction impact
- Integrate with EUDR-025 Risk Mitigation Advisor for detailed action plans
- Track mitigation implementation and measure actual risk reduction

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Machine learning-based risk scoring models (deterministic formulas only for v1.0; ML deferred to v2.0)
- Real-time streaming risk updates (batch/on-demand assessment only for v1.0)
- Carbon footprint integration with risk assessment (defer to GL-GHG-APP integration)
- Mobile native risk assessment application (web-only for v1.0)
- Automated competent authority submission (manual DDS submission for v1.0)
- Multi-regulation risk aggregation (EUDR-only for v1.0; CSDDD/CSRD integration deferred)

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
    | AGENT-EUDR-028            |                                        | AGENT-EUDR-026             |
    | Risk Assessment Engine    |<-------------------------------------->| Due Diligence Orchestrator |
    |                           |   (orchestrator invokes RAE for        |                            |
    | - Composite Calculator    |    Phase 2: Risk Assessment)           | - Workflow Engine          |
    | - Factor Aggregator       |                                        | - Quality Gates            |
    | - Art.10(2) Evaluator     |                                        | - State Manager            |
    | - Country Benchmarking    |                                        +----------------------------+
    | - Simplified DD Engine    |
    | - Risk Classification     |
    | - Trend Analyzer          |
    | - Report Generator        |
    | - Override Engine         |
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
                                                           | - Risk assessments     |
                                                           | - Trend time-series    |
                                                           | - Country benchmarks   |
                                                           | - Override audit trail |
                                                           +-----------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/risk_assessment_engine/
    __init__.py                          # Public API exports
    config.py                            # RiskAssessmentEngineConfig with GL_EUDR_RAE_ env prefix
    models.py                            # 10+ enums, 15+ Pydantic v2 models
    composite_risk_calculator.py         # Weighted composite score computation (Feature 1)
    risk_factor_aggregator.py            # Multi-source risk signal aggregation (Feature 2)
    article10_criteria_evaluator.py      # Article 10(2)(a-k) criteria evaluation (Feature 3)
    country_benchmark_engine.py          # Article 29 country benchmarking (Feature 4)
    simplified_dd_engine.py              # Article 13 simplified due diligence (Feature 5)
    risk_classification_engine.py        # 5-tier risk classification (Feature 6)
    risk_trend_analyzer.py               # Historical risk trend tracking (Feature 7)
    risk_report_generator.py             # DDS-ready risk assessment reports (Feature 8)
    risk_override_engine.py              # Manual override with audit trail (Feature 9)
    provenance.py                        # SHA-256 hash chains for all calculations
    metrics.py                           # 18 Prometheus metrics with gl_eudr_rae_ prefix
    setup.py                             # RiskAssessmentEngineService facade
    api.py                               # FastAPI routes (15+ endpoints)
```

### 7.3 Data Models (Key Entities)

```python
from enum import Enum
from decimal import Decimal
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


# Risk dimension enumeration mapping to Article 10(2) criteria
class RiskDimension(str, Enum):
    COUNTRY = "country"                              # Art. 10(2)(a-b)
    COMMODITY = "commodity"                          # Art. 10(2)(c-d)
    SUPPLIER = "supplier"                            # Art. 10(2)(h-i)
    DEFORESTATION = "deforestation"                  # Art. 10(2)(a-c)
    CORRUPTION = "corruption"                        # Art. 10(2)(e)
    SUPPLY_CHAIN_COMPLEXITY = "supply_chain_complexity"  # Art. 10(2)(a)
    MIXING = "mixing"                                # Art. 10(2)(f)
    CIRCUMVENTION = "circumvention"                  # Art. 10(2)(g)
    INDIGENOUS_PEOPLES = "indigenous_peoples"         # Art. 10(2)(j) -- qualitative
    INTERNATIONAL_AGREEMENTS = "international_agreements"  # Art. 10(2)(k) -- qualitative


# 5-tier risk classification
class RiskLevel(str, Enum):
    NEGLIGIBLE = "negligible"    # 0-15: No further action required
    LOW = "low"                  # 16-30: Standard monitoring
    STANDARD = "standard"        # 31-60: Full documentation required
    HIGH = "high"                # 61-80: Article 11 mitigation required
    CRITICAL = "critical"        # 81-100: Market placement blocked


# Risk assessment lifecycle status
class RiskAssessmentStatus(str, Enum):
    PENDING = "pending"                        # Assessment created, not yet computed
    AGGREGATING = "aggregating"                # Collecting upstream risk signals
    COMPUTING = "computing"                    # Computing composite score
    EVALUATING_CRITERIA = "evaluating_criteria" # Evaluating Art. 10(2) criteria
    CLASSIFYING = "classifying"                # Applying risk classification
    COMPLETED = "completed"                    # Assessment complete
    OVERRIDDEN = "overridden"                  # Manual override applied
    EXPIRED = "expired"                        # Assessment older than validity period
    FAILED = "failed"                          # Assessment failed due to error


# Article 29 country benchmarking classification
class CountryBenchmarkLevel(str, Enum):
    LOW = "low"            # Simplified DD eligible
    STANDARD = "standard"  # Default classification
    HIGH = "high"          # Enhanced DD required
    PENDING = "pending"    # Not yet classified by EC


# Risk trend direction
class TrendDirection(str, Enum):
    IMPROVING = "improving"        # Score decreasing (risk reducing)
    STABLE = "stable"              # Score within +/- 5 points
    DETERIORATING = "deteriorating" # Score increasing (risk growing)
    INSUFFICIENT_DATA = "insufficient_data"  # Not enough history


# Fallback strategy for missing upstream data
class FallbackStrategy(str, Enum):
    USE_LAST_KNOWN = "use_last_known"  # Use most recent historical value
    USE_DEFAULT = "use_default"        # Use reference data average
    FAIL = "fail"                      # Reject assessment as incomplete


# Individual risk factor input from an upstream agent
class RiskFactorInput(BaseModel):
    dimension: RiskDimension
    source_agent: str                    # e.g., "EUDR-016", "EUDR-017"
    raw_score: Decimal = Field(ge=0, le=100)
    confidence: Decimal = Field(ge=0, le=1)
    timestamp: datetime
    data_freshness_days: int = 0
    is_stale: bool = False               # True if older than freshness threshold
    metadata: Dict[str, str] = {}


# Composite risk score output
class CompositeRiskScore(BaseModel):
    assessment_id: str
    operator_id: str
    commodity: str
    supplier_id: Optional[str] = None
    country_code: str
    overall_score: Decimal = Field(ge=0, le=100)
    risk_level: RiskLevel
    factor_scores: Dict[RiskDimension, Decimal]
    factor_confidences: Dict[RiskDimension, Decimal]
    weights: Dict[RiskDimension, Decimal]
    weighted_contributions: Dict[RiskDimension, Decimal]  # W_i * S_i * C_i per dimension
    qualitative_flags: Dict[RiskDimension, bool]           # True if qualitative modifier active
    country_benchmark: CountryBenchmarkLevel
    benchmark_multiplier: Decimal
    simplified_dd_eligible: bool
    provenance_hash: str                                   # SHA-256 of all inputs + output
    computed_at: datetime
    status: RiskAssessmentStatus


# Article 10(2) single criterion evaluation result
class CriterionEvaluation(BaseModel):
    criterion_id: str                    # e.g., "art10_2_a", "art10_2_f"
    criterion_article: str               # e.g., "Article 10(2)(a)"
    criterion_text: str                  # Full text of the criterion
    evaluation_score: Optional[Decimal]  # 0-100 for quantitative; None for qualitative
    qualitative_result: Optional[str]    # For qualitative criteria: "no_concern" / "concern_identified"
    data_sources: List[str]              # List of data sources used
    methodology: str                     # Description of evaluation method
    evidence_references: List[str]       # Document/record IDs supporting evaluation
    is_fully_evaluated: bool             # False if insufficient data
    remediation_guidance: Optional[str]  # If not fully evaluated, what data is needed


# Complete Article 10(2) criteria evaluation
class Article10CriteriaResult(BaseModel):
    assessment_id: str
    criteria_evaluations: List[CriterionEvaluation]
    total_criteria: int
    fully_evaluated_count: int
    coverage_score: Decimal              # % of criteria fully evaluated
    overall_criteria_risk: RiskLevel
    evaluation_timestamp: datetime
    provenance_hash: str


# Country benchmarking record
class CountryBenchmark(BaseModel):
    country_code: str                    # ISO 3166-1 alpha-2
    country_name: str
    benchmark_level: CountryBenchmarkLevel
    effective_date: datetime
    source_publication: str              # EC publication reference
    previous_level: Optional[CountryBenchmarkLevel] = None
    reclassification_date: Optional[datetime] = None
    subnational_overrides: Dict[str, CountryBenchmarkLevel] = {}  # region -> level
    metadata: Dict[str, str] = {}


# Complete DDS-ready risk assessment report
class RiskAssessmentReport(BaseModel):
    report_id: str
    assessment_id: str
    operator_id: str
    operator_name: str
    commodity: str
    supplier_id: Optional[str] = None
    supplier_name: Optional[str] = None
    country_codes: List[str]
    composite_score: CompositeRiskScore
    criteria_result: Article10CriteriaResult
    country_benchmarks: List[CountryBenchmark]
    simplified_dd_eligible: bool
    simplified_dd_justification: Optional[str] = None
    risk_level: RiskLevel
    regulatory_action: str               # "market_placement_allowed", "documentation_required", etc.
    trend_summary: Optional[Dict] = None
    overrides: List[Dict] = []
    report_language: str = "en"
    generated_at: datetime
    provenance_hash: str                 # SHA-256 covering entire report
    dds_format_version: str = "1.0"


# Risk trend data point
class RiskTrendPoint(BaseModel):
    timestamp: datetime
    composite_score: Decimal
    risk_level: RiskLevel
    dimension_scores: Dict[RiskDimension, Decimal]
    assessment_id: str
    is_override: bool = False


# Risk trend analysis result
class RiskTrendAnalysis(BaseModel):
    operator_id: str
    commodity: str
    supplier_id: Optional[str] = None
    country_code: str
    trend_direction: TrendDirection
    moving_average_30d: Optional[Decimal] = None
    moving_average_90d: Optional[Decimal] = None
    moving_average_12m: Optional[Decimal] = None
    rate_of_change_per_month: Optional[Decimal] = None
    data_points: List[RiskTrendPoint]
    data_point_count: int
    earliest_assessment: Optional[datetime] = None
    latest_assessment: Optional[datetime] = None
    alerts: List[str] = []


# Risk override record
class RiskOverride(BaseModel):
    override_id: str
    assessment_id: str
    original_composite_score: Decimal
    override_composite_score: Decimal
    original_risk_level: RiskLevel
    override_risk_level: RiskLevel
    dimension_overrides: Dict[RiskDimension, Decimal] = {}  # Per-dimension overrides
    justification: str = Field(min_length=50)
    supporting_evidence: List[str]       # Document IDs or URLs
    applied_by: str                      # User ID
    applied_at: datetime
    expires_at: datetime                 # Max 12 months from applied_at
    is_active: bool = True
    revoked_by: Optional[str] = None
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[str] = None


# Top-level risk assessment operation
class RiskAssessmentOperation(BaseModel):
    operation_id: str
    assessment_id: str
    operator_id: str
    commodity: str
    supplier_id: Optional[str] = None
    country_codes: List[str]
    status: RiskAssessmentStatus
    factor_inputs: List[RiskFactorInput]
    composite_score: Optional[CompositeRiskScore] = None
    criteria_result: Optional[Article10CriteriaResult] = None
    country_benchmarks: List[CountryBenchmark] = []
    simplified_dd_eligible: Optional[bool] = None
    risk_level: Optional[RiskLevel] = None
    overrides: List[RiskOverride] = []
    report: Optional[RiskAssessmentReport] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    provenance_hash: Optional[str] = None
```

### 7.4 Database Schema (New Migration: V116)

```sql
-- Migration: V116__agent_eudr_risk_assessment_engine.sql
-- Agent: EUDR-028 Risk Assessment Engine
-- Category: Due Diligence (Category 5)

CREATE SCHEMA IF NOT EXISTS eudr_risk_assessment_engine;

-- Risk assessments (primary table)
CREATE TABLE eudr_risk_assessment_engine.risk_assessments (
    assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    supplier_id UUID,
    country_codes JSONB NOT NULL DEFAULT '[]',
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    composite_score NUMERIC(5,2),
    risk_level VARCHAR(20),
    simplified_dd_eligible BOOLEAN DEFAULT FALSE,
    country_benchmark_level VARCHAR(20),
    benchmark_multiplier NUMERIC(4,2) DEFAULT 1.0,
    criteria_coverage_score NUMERIC(5,2),
    is_overridden BOOLEAN DEFAULT FALSE,
    weight_profile JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    CONSTRAINT fk_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

-- Risk factor inputs from upstream agents (hypertable)
CREATE TABLE eudr_risk_assessment_engine.risk_factor_inputs (
    input_id UUID DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL,
    dimension VARCHAR(40) NOT NULL,
    source_agent VARCHAR(20) NOT NULL,
    raw_score NUMERIC(5,2) NOT NULL,
    confidence NUMERIC(4,3) NOT NULL,
    is_stale BOOLEAN DEFAULT FALSE,
    data_freshness_days INTEGER DEFAULT 0,
    fallback_used VARCHAR(20),
    metadata JSONB DEFAULT '{}',
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_risk_assessment_engine.risk_factor_inputs', 'collected_at');

-- Composite score decomposition (weighted contributions per dimension)
CREATE TABLE eudr_risk_assessment_engine.score_decomposition (
    decomposition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL REFERENCES eudr_risk_assessment_engine.risk_assessments(assessment_id),
    dimension VARCHAR(40) NOT NULL,
    weight NUMERIC(4,3) NOT NULL,
    raw_score NUMERIC(5,2) NOT NULL,
    confidence NUMERIC(4,3) NOT NULL,
    weighted_contribution NUMERIC(8,4) NOT NULL,
    is_qualitative BOOLEAN DEFAULT FALSE,
    qualitative_flag BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Article 10(2) criteria evaluations (hypertable)
CREATE TABLE eudr_risk_assessment_engine.criteria_evaluations (
    evaluation_id UUID DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL,
    criterion_id VARCHAR(20) NOT NULL,
    criterion_article VARCHAR(30) NOT NULL,
    criterion_text TEXT NOT NULL,
    evaluation_score NUMERIC(5,2),
    qualitative_result VARCHAR(30),
    data_sources JSONB DEFAULT '[]',
    methodology TEXT,
    evidence_references JSONB DEFAULT '[]',
    is_fully_evaluated BOOLEAN DEFAULT TRUE,
    remediation_guidance TEXT,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_risk_assessment_engine.criteria_evaluations', 'evaluated_at');

-- Country benchmarking classifications
CREATE TABLE eudr_risk_assessment_engine.country_benchmarks (
    benchmark_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    benchmark_level VARCHAR(20) NOT NULL DEFAULT 'standard',
    effective_date TIMESTAMPTZ NOT NULL,
    source_publication VARCHAR(500),
    previous_level VARCHAR(20),
    reclassification_date TIMESTAMPTZ,
    subnational_overrides JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_country_benchmark UNIQUE (country_code, effective_date)
);

-- Risk trend time-series (hypertable for efficient temporal queries)
CREATE TABLE eudr_risk_assessment_engine.risk_trends (
    trend_id UUID DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    supplier_id UUID,
    country_code CHAR(2) NOT NULL,
    composite_score NUMERIC(5,2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    dimension_scores JSONB DEFAULT '{}',
    assessment_id UUID NOT NULL,
    is_override BOOLEAN DEFAULT FALSE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_risk_assessment_engine.risk_trends', 'recorded_at');

-- Risk overrides with audit trail
CREATE TABLE eudr_risk_assessment_engine.risk_overrides (
    override_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL REFERENCES eudr_risk_assessment_engine.risk_assessments(assessment_id),
    original_composite_score NUMERIC(5,2) NOT NULL,
    override_composite_score NUMERIC(5,2) NOT NULL,
    original_risk_level VARCHAR(20) NOT NULL,
    override_risk_level VARCHAR(20) NOT NULL,
    dimension_overrides JSONB DEFAULT '{}',
    justification TEXT NOT NULL,
    supporting_evidence JSONB DEFAULT '[]',
    applied_by VARCHAR(100) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    revoked_by VARCHAR(100),
    revoked_at TIMESTAMPTZ,
    revocation_reason TEXT
);

-- Risk assessment reports
CREATE TABLE eudr_risk_assessment_engine.risk_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL REFERENCES eudr_risk_assessment_engine.risk_assessments(assessment_id),
    report_format VARCHAR(10) NOT NULL DEFAULT 'json',
    report_language VARCHAR(5) NOT NULL DEFAULT 'en',
    report_content JSONB,
    report_pdf_path VARCHAR(500),
    dds_format_version VARCHAR(10) DEFAULT '1.0',
    provenance_hash VARCHAR(64) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Simplified due diligence eligibility records
CREATE TABLE eudr_risk_assessment_engine.simplified_dd_records (
    record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL REFERENCES eudr_risk_assessment_engine.risk_assessments(assessment_id),
    is_eligible BOOLEAN NOT NULL,
    all_countries_low_risk BOOLEAN NOT NULL,
    composite_below_threshold BOOLEAN NOT NULL,
    no_active_deforestation_alerts BOOLEAN NOT NULL,
    threshold_used NUMERIC(5,2) NOT NULL DEFAULT 30.0,
    denial_reasons JSONB DEFAULT '[]',
    evidence JSONB DEFAULT '{}',
    determined_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX idx_assessments_operator ON eudr_risk_assessment_engine.risk_assessments(operator_id);
CREATE INDEX idx_assessments_commodity ON eudr_risk_assessment_engine.risk_assessments(commodity);
CREATE INDEX idx_assessments_status ON eudr_risk_assessment_engine.risk_assessments(status);
CREATE INDEX idx_assessments_risk_level ON eudr_risk_assessment_engine.risk_assessments(risk_level);
CREATE INDEX idx_assessments_supplier ON eudr_risk_assessment_engine.risk_assessments(supplier_id);
CREATE INDEX idx_assessments_created ON eudr_risk_assessment_engine.risk_assessments(created_at DESC);
CREATE INDEX idx_factor_inputs_assessment ON eudr_risk_assessment_engine.risk_factor_inputs(assessment_id);
CREATE INDEX idx_factor_inputs_dimension ON eudr_risk_assessment_engine.risk_factor_inputs(dimension);
CREATE INDEX idx_decomposition_assessment ON eudr_risk_assessment_engine.score_decomposition(assessment_id);
CREATE INDEX idx_criteria_assessment ON eudr_risk_assessment_engine.criteria_evaluations(assessment_id);
CREATE INDEX idx_benchmarks_country ON eudr_risk_assessment_engine.country_benchmarks(country_code);
CREATE INDEX idx_benchmarks_level ON eudr_risk_assessment_engine.country_benchmarks(benchmark_level);
CREATE INDEX idx_trends_operator_commodity ON eudr_risk_assessment_engine.risk_trends(operator_id, commodity);
CREATE INDEX idx_trends_country ON eudr_risk_assessment_engine.risk_trends(country_code);
CREATE INDEX idx_trends_risk_level ON eudr_risk_assessment_engine.risk_trends(risk_level);
CREATE INDEX idx_overrides_assessment ON eudr_risk_assessment_engine.risk_overrides(assessment_id);
CREATE INDEX idx_overrides_active ON eudr_risk_assessment_engine.risk_overrides(is_active);
CREATE INDEX idx_overrides_expires ON eudr_risk_assessment_engine.risk_overrides(expires_at);
CREATE INDEX idx_reports_assessment ON eudr_risk_assessment_engine.risk_reports(assessment_id);
CREATE INDEX idx_simplified_dd_assessment ON eudr_risk_assessment_engine.simplified_dd_records(assessment_id);

-- Retention policies for hypertables
SELECT add_retention_policy('eudr_risk_assessment_engine.risk_factor_inputs', INTERVAL '5 years');
SELECT add_retention_policy('eudr_risk_assessment_engine.criteria_evaluations', INTERVAL '5 years');
SELECT add_retention_policy('eudr_risk_assessment_engine.risk_trends', INTERVAL '5 years');

-- Continuous aggregate for monthly risk trend summary
CREATE MATERIALIZED VIEW eudr_risk_assessment_engine.monthly_risk_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', recorded_at) AS month,
    operator_id,
    commodity,
    country_code,
    AVG(composite_score) AS avg_composite_score,
    MIN(composite_score) AS min_composite_score,
    MAX(composite_score) AS max_composite_score,
    COUNT(*) AS assessment_count
FROM eudr_risk_assessment_engine.risk_trends
GROUP BY time_bucket('1 month', recorded_at), operator_id, commodity, country_code;

SELECT add_continuous_aggregate_policy('eudr_risk_assessment_engine.monthly_risk_summary',
    start_offset => INTERVAL '3 months',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 day');
```

### 7.5 API Endpoints (15+)

| Method | Path | Description |
|--------|------|-------------|
| **Risk Assessment CRUD** | | |
| POST | `/v1/eudr/risk-assessment/assessments` | Create and compute a new risk assessment |
| GET | `/v1/eudr/risk-assessment/assessments` | List risk assessments (with filters: operator, commodity, risk_level, status) |
| GET | `/v1/eudr/risk-assessment/assessments/{assessment_id}` | Get full assessment details with decomposition |
| DELETE | `/v1/eudr/risk-assessment/assessments/{assessment_id}` | Archive a risk assessment |
| POST | `/v1/eudr/risk-assessment/assessments/batch` | Create multiple assessments in batch |
| **Composite Score** | | |
| POST | `/v1/eudr/risk-assessment/compute` | Compute composite risk score from provided factor inputs |
| GET | `/v1/eudr/risk-assessment/assessments/{assessment_id}/decomposition` | Get score decomposition by dimension |
| **Article 10(2) Criteria** | | |
| GET | `/v1/eudr/risk-assessment/assessments/{assessment_id}/criteria` | Get Article 10(2) criteria evaluation results |
| POST | `/v1/eudr/risk-assessment/assessments/{assessment_id}/criteria/evaluate` | Trigger criteria re-evaluation |
| **Country Benchmarking** | | |
| GET | `/v1/eudr/risk-assessment/benchmarks` | List all country benchmarking classifications |
| GET | `/v1/eudr/risk-assessment/benchmarks/{country_code}` | Get benchmarking for a specific country |
| PUT | `/v1/eudr/risk-assessment/benchmarks` | Bulk update country benchmarking classifications |
| **Simplified Due Diligence** | | |
| GET | `/v1/eudr/risk-assessment/assessments/{assessment_id}/simplified-dd` | Get simplified DD eligibility determination |
| **Risk Trends** | | |
| GET | `/v1/eudr/risk-assessment/trends` | Get risk trends (filter by operator, commodity, supplier, country, date range) |
| GET | `/v1/eudr/risk-assessment/trends/alerts` | Get active trend alerts |
| **Reports** | | |
| POST | `/v1/eudr/risk-assessment/assessments/{assessment_id}/report` | Generate DDS-ready risk assessment report (JSON or PDF) |
| GET | `/v1/eudr/risk-assessment/assessments/{assessment_id}/report` | Download generated report |
| **Overrides** | | |
| POST | `/v1/eudr/risk-assessment/assessments/{assessment_id}/override` | Apply a risk override |
| GET | `/v1/eudr/risk-assessment/assessments/{assessment_id}/overrides` | List override history for assessment |
| DELETE | `/v1/eudr/risk-assessment/overrides/{override_id}` | Revoke an active override |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_rae_assessments_total` | Counter | Risk assessments computed, labeled by commodity and risk_level |
| 2 | `gl_eudr_rae_factor_aggregations_total` | Counter | Risk factor aggregation operations, labeled by dimension |
| 3 | `gl_eudr_rae_benchmarks_applied_total` | Counter | Country benchmarking lookups applied, labeled by benchmark_level |
| 4 | `gl_eudr_rae_criteria_evaluations_total` | Counter | Article 10(2) criteria evaluations, labeled by criterion_id |
| 5 | `gl_eudr_rae_classifications_total` | Counter | Risk classifications computed, labeled by risk_level |
| 6 | `gl_eudr_rae_reports_generated_total` | Counter | Risk reports generated, labeled by format (json/pdf) |
| 7 | `gl_eudr_rae_overrides_applied_total` | Counter | Manual risk overrides applied |
| 8 | `gl_eudr_rae_api_errors_total` | Counter | API errors, labeled by endpoint and error_type |
| 9 | `gl_eudr_rae_assessment_duration_seconds` | Histogram | End-to-end risk assessment computation latency |
| 10 | `gl_eudr_rae_aggregation_duration_seconds` | Histogram | Risk factor aggregation latency |
| 11 | `gl_eudr_rae_classification_duration_seconds` | Histogram | Risk classification computation latency |
| 12 | `gl_eudr_rae_report_generation_duration_seconds` | Histogram | Report generation latency, labeled by format |
| 13 | `gl_eudr_rae_criteria_evaluation_duration_seconds` | Histogram | Article 10(2) full criteria evaluation latency |
| 14 | `gl_eudr_rae_active_assessments` | Gauge | Currently in-progress risk assessments |
| 15 | `gl_eudr_rae_high_risk_operators` | Gauge | Count of operators with HIGH or CRITICAL risk level |
| 16 | `gl_eudr_rae_average_risk_score` | Gauge | Rolling average composite risk score across all assessments |
| 17 | `gl_eudr_rae_override_count` | Gauge | Currently active risk overrides |
| 18 | `gl_eudr_rae_trend_data_points` | Gauge | Total risk trend data points stored |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL 16 + TimescaleDB | Persistent storage + time-series hypertables for trends |
| Arithmetic | Python `decimal.Decimal` | Exact decimal arithmetic; no floating-point drift |
| Data Models | Pydantic v2 | Type-safe, validated, JSON-compatible models |
| Hashing | hashlib SHA-256 | Provenance hash chains for audit trail |
| Cache | Redis | Country benchmark caching, recent assessment caching |
| Object Storage | S3 | PDF report storage, large evidence packages |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access control for assessments and overrides |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across upstream agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |
| PDF Generation | WeasyPrint | HTML-to-PDF for DDS-ready reports |

### 7.8 RBAC Permissions (SEC-002 Integration)

The following permissions will be registered in the GreenLang PERMISSION_MAP for RBAC enforcement:

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-rae:assessments:read` | View risk assessments and scores | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rae:assessments:write` | Create and trigger risk assessments | Analyst, Compliance Officer, Admin |
| `eudr-rae:assessments:delete` | Archive risk assessments | Admin |
| `eudr-rae:decomposition:read` | View score decomposition details | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rae:criteria:read` | View Article 10(2) criteria evaluations | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rae:criteria:evaluate` | Trigger criteria re-evaluation | Analyst, Compliance Officer, Admin |
| `eudr-rae:benchmarks:read` | View country benchmarking classifications | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rae:benchmarks:write` | Update country benchmarking data | Compliance Officer, Admin |
| `eudr-rae:simplified-dd:read` | View simplified DD eligibility | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rae:trends:read` | View risk trends and alerts | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rae:reports:generate` | Generate DDS-ready risk reports | Compliance Officer, Admin |
| `eudr-rae:reports:read` | Download generated reports | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rae:overrides:apply` | Apply manual risk overrides | Compliance Officer, Admin |
| `eudr-rae:overrides:read` | View override history and audit trail | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-rae:overrides:revoke` | Revoke active overrides | Compliance Officer, Admin |
| `eudr-rae:audit:read` | View full audit trail and provenance hashes | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| EUDR-016 Country Risk Evaluator | API / database | Country risk scores (0-100) + confidence -> RiskFactorInput (COUNTRY dimension) |
| EUDR-017 Supplier Risk Scorer | API / database | Supplier risk scores (0-100) + confidence -> RiskFactorInput (SUPPLIER dimension) |
| EUDR-018 Commodity Risk Analyzer | API / database | Commodity risk scores (0-100) + confidence -> RiskFactorInput (COMMODITY dimension) |
| EUDR-019 Corruption Index Monitor | API / database | Corruption risk scores (0-100) + confidence -> RiskFactorInput (CORRUPTION dimension) |
| EUDR-020 Deforestation Alert System | API / database | Deforestation risk scores (0-100) + confidence -> RiskFactorInput (DEFORESTATION dimension) |
| EUDR-001 Supply Chain Mapping Master | API / database | Supply chain graph complexity metrics -> SUPPLY_CHAIN_COMPLEXITY scoring |
| EUDR-009 Chain of Custody Agent | API / database | Custody model and mixing indicators -> MIXING risk scoring |
| EUDR-021 Indigenous Rights Checker | API / database | Indigenous rights concerns -> INDIGENOUS_PEOPLES qualitative flag |
| AGENT-FOUND-008 Reproducibility Agent | Verification | Bit-perfect verification of composite score calculations |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| EUDR-026 Due Diligence Orchestrator | API invocation | Orchestrator calls RAE for Phase 2 risk assessment; receives CompositeRiskScore + Article10CriteriaResult |
| EUDR-025 Risk Mitigation Advisor | API / database | Risk level and decomposition -> targeted mitigation recommendations |
| GL-EUDR-APP v1.0 Platform | API integration | Risk scores, trends, reports -> frontend dashboard and DDS generation |
| EUDR-027 Information Gathering Agent | API / database | Validates information completeness before risk assessment begins |
| External Auditors | Read-only API + PDF reports | Risk reports and audit trails for third-party verification |

---

## 8. Implementation Plan

### 8.1 Milestones

#### Phase 1: Core Computation Engine (Weeks 1-4)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1 | Config, models, and provenance module; Risk Factor Aggregation Engine (Feature 2) | Backend Engineer |
| 2 | Composite Risk Score Calculator (Feature 1) with Decimal arithmetic and golden tests | Senior Backend Engineer |
| 3 | Risk Classification and Threshold Engine (Feature 6); Country Benchmarking Integration (Feature 4) | Backend Engineer |
| 4 | Simplified Due Diligence Engine (Feature 5); Article 10(2) Criteria Evaluator (Feature 3) | Senior Backend Engineer |

**Milestone: Core computation engine operational with all scoring, classification, and criteria evaluation (Week 4)**

#### Phase 2: Reporting and Adjustment Layer (Weeks 5-8)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 5 | Risk Trend Analyzer (Feature 7) with TimescaleDB hypertables | Backend Engineer |
| 6 | Risk Report Generator (Feature 8) -- JSON and PDF output | Backend Engineer + Frontend Engineer |
| 7 | Risk Override and Manual Adjustment Engine (Feature 9) with audit trail | Backend Engineer |
| 8 | REST API Layer: 15+ endpoints, authentication, rate limiting, OpenAPI docs | Backend Engineer |

**Milestone: Full API operational with reporting, overrides, and trend analysis (Week 8)**

#### Phase 3: Integration and Testing (Weeks 9-12)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 9 | Integration with EUDR-016 through EUDR-020 upstream agents | Integration Engineer |
| 10 | Integration with EUDR-026 orchestrator, EUDR-025 mitigation advisor | Integration Engineer |
| 11 | Complete test suite: 500+ tests, golden tests, performance tests | Test Engineer |
| 12 | Database migration V116 finalized; Grafana dashboard; RBAC integration | DevOps + Backend |

**Milestone: All 9 P0 features integrated, tested, and production-ready (Week 12)**

#### Phase 4: Launch (Weeks 13-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 13 | Performance testing, security audit, load testing (500+ assessments/min) | DevOps + Security |
| 14 | Beta customer onboarding (5 customers), launch readiness review, go-live | Product + Engineering |

**Milestone: Production launch with all 9 P0 features (Week 14)**

### 8.2 Dependencies

#### Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Stable, production-ready; provides country risk scores |
| EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Stable, production-ready; provides supplier risk scores |
| EUDR-018 Commodity Risk Analyzer | BUILT (100%) | Low | Stable, production-ready; provides commodity risk scores |
| EUDR-019 Corruption Index Monitor | BUILT (100%) | Low | Stable, production-ready; provides corruption risk scores |
| EUDR-020 Deforestation Alert System | BUILT (100%) | Low | Stable, production-ready; provides deforestation risk scores |
| EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Provides supply chain complexity data |
| EUDR-009 Chain of Custody Agent | BUILT (100%) | Low | Provides custody model and mixing indicators |
| EUDR-021 Indigenous Rights Checker | BUILT (100%) | Low | Provides indigenous rights concern flags |
| EUDR-026 Due Diligence Orchestrator | BUILT (100%) | Low | Invokes RAE for Phase 2 orchestration |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| SEC-005 Centralized Audit Logging | BUILT (100%) | Low | Override audit trail integration |

#### External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EC Article 29 country benchmarking list | Published; updated quarterly | Medium | Hot-reloadable database; adapter pattern for format changes |
| EU Information System DDS schema | Published (v1.x) | Medium | Adapter pattern for schema version changes |
| Transparency International CPI (annual) | Available | Low | Cached data with annual refresh |
| World Bank WGI (biannual) | Available | Low | Cached data with biannual refresh |

### 8.3 Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EC changes Article 10(2) criteria or adds new criteria in implementing regulations | Low | High | Modular criteria evaluator with configurable criterion list; new criteria can be added without code redesign |
| R2 | EC country benchmarking format or publication mechanism changes | Medium | Medium | Adapter pattern isolates ingestion layer; multiple format parsers |
| R3 | Upstream risk agents (EUDR-016 to 020) return inconsistent score ranges or formats | Low | Medium | Risk Factor Aggregator validates and normalizes all inputs; clamps out-of-range values |
| R4 | Composite risk formula weights are disputed by competent authorities | Medium | High | Weights are fully configurable per operator; formula is transparent and documented; regulators can inspect methodology |
| R5 | Performance degradation under high assessment volume (500+ per minute) | Low | Medium | Horizontal scaling on K8s; Redis caching for benchmark lookups; TimescaleDB optimized queries |
| R6 | Simplified DD eligibility determination challenged by competent authority | Medium | High | Full evidence trail for every eligibility condition; conservative thresholds; operator opt-out option |
| R7 | Manual overrides abused to artificially lower risk scores | Low | High | Override limits (+/- 40 points max); mandatory justification (50+ chars); expiration (12 months max); admin notification; audit trail |
| R8 | Risk trend data storage grows unbounded over 5-year retention period | Medium | Medium | TimescaleDB compression; continuous aggregates for older data; retention policies enforced |
| R9 | Integration complexity with 5 upstream agents and 3 downstream consumers | Medium | Medium | Well-defined interfaces; mock adapters for testing; circuit breaker pattern for upstream calls |
| R10 | PDF report generation performance under batch workload | Medium | Low | Async report generation; queue-based processing; S3 storage for completed reports |

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Composite Risk Score Calculator -- weighted formula, Decimal arithmetic, bit-perfect reproducibility
  - [ ] Feature 2: Risk Factor Aggregation Engine -- 5-agent input collection, validation, fallback strategies
  - [ ] Feature 3: Article 10(2) Criteria Evaluation -- all 10 criteria evaluated with documentation
  - [ ] Feature 4: Country Benchmarking Integration -- hot-reloadable EC classifications, benchmarking multipliers
  - [ ] Feature 5: Simplified Due Diligence Engine -- 3-condition eligibility determination, Article 13 documentation
  - [ ] Feature 6: Risk Classification and Threshold Engine -- 5-tier classification with hysteresis
  - [ ] Feature 7: Risk Trend Analysis -- moving averages, trend detection, alerting
  - [ ] Feature 8: Risk Report Generator -- JSON and PDF DDS-ready output
  - [ ] Feature 9: Risk Override and Manual Adjustment -- audit trail, expiration, revocation
- [ ] >= 85% test coverage achieved (line coverage >= 85%, branch coverage >= 90%)
- [ ] Security audit passed (JWT + RBAC integrated)
- [ ] Performance targets met (< 200ms p95 single assessment, 500+ assessments/min)
- [ ] Composite risk formula validated against 500+ golden test values (100% match)
- [ ] Bit-perfect reproducibility verified across environments
- [ ] All 7 commodity types tested with representative risk assessment scenarios
- [ ] Integration with EUDR-016 through EUDR-020 verified end-to-end
- [ ] Integration with EUDR-026 orchestrator verified (Phase 2 invocation)
- [ ] API documentation complete (OpenAPI spec)
- [ ] Database migration V116 tested and validated
- [ ] Grafana dashboard operational with all 18 metrics
- [ ] RBAC permissions registered and tested (16 permissions)
- [ ] 5 beta customers successfully completed risk assessments
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 200+ risk assessments completed by customers
- Average assessment latency < 200ms p95
- 100% composite score reproducibility verified
- Country benchmarking database current with latest EC publication
- < 5 support tickets per customer related to risk assessment
- Zero failed assessments due to upstream agent integration issues

**60 Days:**
- 1,000+ risk assessments completed
- Average Article 10(2) criteria coverage >= 90%
- 50+ simplified DD eligibility determinations processed
- Risk trend analysis active for 100+ operator-commodity combinations
- 100+ DDS-ready risk reports generated
- NPS > 40 from compliance officer persona

**90 Days:**
- 5,000+ risk assessments completed
- < 20 manual overrides applied (indicating strong automated scoring accuracy)
- Zero EUDR penalties attributed to inadequate risk assessment for active customers
- Risk trend data covering full quarter for early adopters
- Regulatory audit readiness confirmed by 3+ beta customers
- NPS > 50

---

## 10. Test Strategy

### 10.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Composite Score Calculator Tests | 120+ | Weighted formula, Decimal arithmetic, edge cases, golden values |
| Risk Factor Aggregation Tests | 80+ | Input validation, normalization, fallback strategies, stale detection |
| Article 10(2) Criteria Tests | 100+ | All 10 criteria, data source mapping, coverage scoring, remediation |
| Country Benchmarking Tests | 60+ | Classification lookups, hot-reload, multi-country, subnational |
| Simplified DD Engine Tests | 50+ | Eligibility conditions, boundary cases, revocation, documentation |
| Risk Classification Tests | 40+ | 5-tier classification, hysteresis, configurable thresholds |
| Risk Trend Analyzer Tests | 70+ | Moving averages, trend detection, alerts, time-series queries |
| Risk Report Generator Tests | 60+ | JSON/PDF output, DDS schema validation, multi-language |
| Risk Override Tests | 50+ | Apply, revoke, expire, audit trail, authorization checks |
| API Tests | 80+ | All 15+ endpoints, auth, error handling, pagination |
| Integration Tests | 40+ | Cross-agent integration with EUDR-016 through EUDR-020 |
| Performance Tests | 20+ | Throughput, latency, concurrent assessments, trend query performance |
| Golden Tests | 30+ | Known-value composite scores across all 7 commodities |
| **Total** | **800+** | |

### 10.2 Golden Test Design

Each of the 7 EUDR commodities will have dedicated golden test scenarios:

1. **Low-risk scenario** -- All dimensions score LOW, LOW-risk country benchmark -> expect NEGLIGIBLE/LOW composite, simplified DD eligible
2. **Standard-risk scenario** -- Mixed dimension scores, STANDARD country benchmark -> expect STANDARD composite, full documentation required
3. **High-risk scenario** -- Multiple HIGH dimension scores, HIGH country benchmark -> expect HIGH composite, Article 11 mitigation triggered
4. **Critical-risk scenario** -- CRITICAL deforestation alert + HIGH country + HIGH corruption -> expect CRITICAL composite, market placement blocked
5. **Missing data scenario** -- 2 of 5 upstream agents unavailable -> expect partial assessment with degraded confidence

Total: 7 commodities x 5 scenarios = 35 core golden test scenarios + 15 edge case scenarios = 50 golden tests

### 10.3 Reproducibility Testing

Every golden test will be executed 100 times across:
- Different Python interpreter sessions
- Different operating systems (Linux, macOS, Windows)
- Different hardware architectures (x86_64, ARM64)

All executions must produce bit-identical SHA-256 provenance hashes, verifying deterministic behavior.

---

## 11. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **Article 10** | EUDR article requiring operators to perform risk assessment |
| **Article 10(2)** | Sub-article listing 10+ specific risk assessment criteria |
| **Article 13** | EUDR article providing simplified due diligence for low-risk country products |
| **Article 29** | EUDR article establishing EC country benchmarking system |
| **Composite Risk Score** | Weighted aggregation of multiple risk dimension scores into a single 0-100 value |
| **Risk Dimension** | A single aspect of risk (country, commodity, supplier, deforestation, corruption, etc.) |
| **Risk Level** | Classification tier (NEGLIGIBLE, LOW, STANDARD, HIGH, CRITICAL) |
| **Country Benchmarking** | EC classification of countries as Low, Standard, or High risk for EUDR |
| **Simplified Due Diligence** | Reduced due diligence obligations for products from Low-risk countries |
| **Hysteresis** | Band around threshold to prevent classification oscillation between tiers |
| **Provenance Hash** | SHA-256 hash covering all inputs and outputs for audit verification |
| **Risk Override** | Manual adjustment of computed risk score by authorized user with justification |
| **Risk Trend** | Historical trajectory of risk scores over time for a given combination |

### Appendix B: Composite Risk Calculation Formula

```
Composite Risk Score Calculation
================================

Formula:
    Composite_Risk = SUM(W_i * S_i * C_i) / SUM(W_i * C_i)

Where:
    W_i = Weight for risk dimension i (configurable, must sum to 1.0)
    S_i = Score for risk dimension i (0-100 scale, from upstream agent)
    C_i = Confidence for risk dimension i (0-1 scale, from upstream agent)

Default Weight Configuration:
    W_country           = 0.20  (EUDR-016 Country Risk Evaluator)
    W_commodity          = 0.15  (EUDR-018 Commodity Risk Analyzer)
    W_supplier           = 0.20  (EUDR-017 Supplier Risk Scorer)
    W_deforestation      = 0.20  (EUDR-020 Deforestation Alert System)
    W_corruption         = 0.10  (EUDR-019 Corruption Index Monitor)
    W_complexity         = 0.05  (Supply chain complexity from EUDR-001)
    W_mixing             = 0.05  (Mixing risk from EUDR-009)
    W_circumvention      = 0.05  (Circumvention risk from trade flow analysis)
    ---------
    TOTAL                = 1.00

Country Benchmarking Multiplier:
    If country benchmark = LOW:      multiplier = 0.70
    If country benchmark = STANDARD: multiplier = 1.00
    If country benchmark = HIGH:     multiplier = 1.50

    Adjusted_Composite = min(100, Composite_Risk * multiplier)

Qualitative Modifiers:
    If INDIGENOUS_PEOPLES flag = True:        Adjusted_Composite += 10
    If INTERNATIONAL_AGREEMENTS flag = True:  Adjusted_Composite += 10
    Final score clamped to [0, 100]

Risk Level Classification:
    NEGLIGIBLE:  0.00 - 15.00
    LOW:        15.01 - 30.00
    STANDARD:   30.01 - 60.00
    HIGH:       60.01 - 80.00
    CRITICAL:   80.01 - 100.00

    Threshold boundary: score exactly on boundary classified in lower tier
    Hysteresis band: +/- 2 points around thresholds to prevent oscillation

Simplified Due Diligence Eligibility (Article 13):
    Eligible when ALL of the following are true:
    1. All origin countries classified as LOW by EC Article 29 benchmarking
    2. Adjusted composite risk score < 30.00 (configurable threshold)
    3. No active deforestation alerts for relevant geolocations (EUDR-020)
    4. No individual dimension score > 60 (no HIGH or CRITICAL dimensions)

Example Calculation:
    Inputs:
        Country:       S=45, C=0.95, W=0.20
        Commodity:     S=35, C=0.90, W=0.15
        Supplier:      S=25, C=0.85, W=0.20
        Deforestation: S=60, C=0.98, W=0.20
        Corruption:    S=55, C=0.92, W=0.10
        Complexity:    S=40, C=0.80, W=0.05
        Mixing:        S=30, C=0.75, W=0.05
        Circumvention: S=20, C=0.70, W=0.05

    Numerator = (0.20*45*0.95) + (0.15*35*0.90) + (0.20*25*0.85) +
                (0.20*60*0.98) + (0.10*55*0.92) + (0.05*40*0.80) +
                (0.05*30*0.75) + (0.05*20*0.70)
              = 8.55 + 4.725 + 4.25 + 11.76 + 5.06 + 1.60 + 1.125 + 0.70
              = 37.77

    Denominator = (0.20*0.95) + (0.15*0.90) + (0.20*0.85) + (0.20*0.98) +
                  (0.10*0.92) + (0.05*0.80) + (0.05*0.75) + (0.05*0.70)
                = 0.19 + 0.135 + 0.17 + 0.196 + 0.092 + 0.04 + 0.0375 + 0.035
                = 0.8955

    Composite_Risk = 37.77 / 0.8955 = 42.18

    Country benchmark = STANDARD -> multiplier = 1.0
    Adjusted_Composite = 42.18 * 1.0 = 42.18
    No qualitative flags active.

    Risk Level = STANDARD (30.01 - 60.00)
    Regulatory Action: Full documentation required
    Simplified DD: NOT eligible (score >= 30)
```

### Appendix C: Article 29 Country Benchmarking Reference

The European Commission classifies countries as Low, Standard, or High risk based on:
- Rate of deforestation and forest degradation
- Rate of expansion of agriculture for relevant commodities and relevant products
- Production trends of relevant commodities and relevant products
- Information provided by and in consultation with countries of production, including indigenous peoples, local communities, and other relevant stakeholders

The agent implements a configurable country benchmarking database that:
1. Stores all 195+ country classifications with effective dates
2. Supports subnational benchmarking per Article 29(4)
3. Hot-reloads from EC publication within 5 minutes
4. Tracks classification history for trend analysis
5. Triggers re-assessment of affected assessments on reclassification

### Appendix D: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023
2. EU Deforestation Regulation Guidance Document (European Commission)
3. EUDR Technical Specifications for the EU Information System
4. Article 10(2) Risk Assessment Criteria -- Detailed Implementation Notes (EC)
5. Article 29 Country Benchmarking Methodology (European Commission)
6. Article 13 Simplified Due Diligence Guidance (European Commission)
7. Transparency International Corruption Perceptions Index Methodology
8. World Bank Worldwide Governance Indicators Methodology
9. Global Forest Watch Technical Documentation
10. ISO 31000:2018 -- Risk Management Guidelines

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
| 1.0.0 | 2026-03-11 | GL-ProductManager | Finalized: all 9 P0 features confirmed, regulatory coverage verified (Articles 10, 11, 12, 13, 29, 31), composite risk formula defined with example calculation, 5-tier classification with hysteresis, 18 Prometheus metrics, V116 migration schema, 16 RBAC permissions, 15+ API endpoints, full Article 10(2) criteria mapping, approval granted |
