# PRD: AGENT-EUDR-016 -- Country Risk Evaluator Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-016 |
| **Agent ID** | GL-EUDR-CRE-016 |
| **Component** | Country Risk Evaluator Agent |
| **Category** | EUDR Regulatory Agent -- Risk Assessment |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-09 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-09 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Article 29 (Benchmarking), Articles 10-11 (Due Diligence) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Article 29 mandates that the European Commission benchmark all third countries and territories into three risk categories -- low, standard, and high -- based on deforestation and forest degradation rates, agricultural expansion for EUDR-regulated commodities, governance quality, and enforcement capacity. This benchmarking directly determines the level of due diligence that EU operators and traders must perform: simplified due diligence for low-risk countries (Article 13), standard due diligence for standard-risk countries (Articles 10-11), and enhanced due diligence with mandatory satellite verification for high-risk countries (Article 11). The enforcement date for large operators was December 30, 2025, and SME enforcement follows on June 30, 2026.

Today, EU operators and compliance teams face the following problems when performing country-level risk assessment for EUDR compliance:

- **No composite risk scoring engine**: Operators rely on fragmented, manually compiled country risk data. There is no automated system that synthesizes multiple authoritative data sources (FAO, Global Forest Watch, World Bank WGI, Transparency International CPI) into a single, weighted, auditable composite risk score for 200+ countries.
- **No commodity-specific risk differentiation**: A country may be high-risk for palm oil but low-risk for coffee. Current tools treat country risk as monolithic, failing to differentiate across the 7 EUDR-regulated commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood).
- **No sub-national hotspot detection**: Country-level risk assessments mask critical sub-national variation. Brazil overall may be "high risk," but the actual deforestation is concentrated in the Legal Amazon and Cerrado, not in the Atlantic Forest region. Operators cannot identify and monitor deforestation hotspots at the province or state level.
- **No governance index integration**: Forest governance quality -- rule of law, corruption, environmental law enforcement, indigenous rights recognition -- varies enormously across countries and is a key determinant of actual deforestation risk. No existing tool systematically integrates World Bank WGI, CPI, and forest governance frameworks.
- **No automated due diligence classification**: Operators manually determine whether simplified, standard, or enhanced due diligence is required for each country-commodity-origin combination. This is error-prone, costly, and slow.
- **No trade flow analysis for re-export risk**: Commodities are frequently transshipped or re-exported through low-risk countries to obscure their true high-risk origin. Operators have no automated way to detect these commodity laundering patterns.
- **No regulatory update tracking**: The EC benchmarking list is updated periodically. When a country is reclassified (e.g., from standard to high risk), all existing risk assessments and due diligence levels must be recalculated. There is no automated system to track these changes and propagate their impact.
- **No risk reporting for auditors**: Compliance officers need standardized, auditable risk assessment reports for regulatory submission and third-party audits. Currently, these are manually assembled from disparate sources.

Without solving these problems, EU operators face penalties of up to 4% of annual EU turnover, confiscation of goods, temporary exclusion from public procurement, and public naming. Enhanced due diligence failures for high-risk country imports carry the steepest penalties.

### 1.2 Solution Overview

Agent-EUDR-016: Country Risk Evaluator is a specialized risk assessment agent that provides comprehensive, multi-dimensional country and commodity risk evaluation for EUDR compliance. It is the FIRST agent in the Risk Assessment sub-category, following the completion of EUDR-001 through EUDR-015 (Supply Chain Traceability). The agent synthesizes data from authoritative global sources into deterministic, auditable, version-controlled risk scores that drive due diligence level classification for every country-commodity-origin combination.

The agent consumes and extends the existing country risk databases already built in the GreenLang platform (`greenlang/data/eudr_country_risk.py` with 25+ countries, 1900+ lines; `greenlang/agents/eudr/multi_tier_supplier/reference_data/country_risk_scores.py` with 85+ countries, 1470+ lines) and transforms them into a full-featured, production-grade risk evaluation engine with 8 specialized processing engines.

Core capabilities:

1. **Composite country risk scoring** -- Weighted multi-factor risk scores (0-100) for 200+ countries, combining deforestation rate (30%), governance index (20%), enforcement score (15%), corruption perception index (15%), forest law compliance (10%), and historical trend (10%). All calculations are deterministic, reproducible, and fully auditable.
2. **Commodity-specific risk analysis** -- Per-commodity risk profiles for all 7 EUDR-regulated commodities per country, incorporating production volume, deforestation correlation, seasonal variation, and certification scheme effectiveness.
3. **Deforestation hotspot detection** -- Sub-national hotspot identification using Global Forest Watch tree cover loss data, fire alert correlation, protected area proximity, and indigenous territory overlap. Spatial clustering for automated hotspot boundary generation.
4. **Governance index engine** -- Systematic integration of World Bank Worldwide Governance Indicators, Transparency International CPI, FAO/ITTO forest governance frameworks, legal framework strength scoring, and environmental law enforcement effectiveness.
5. **Due diligence level classification** -- Automated 3-tier classification (simplified/standard/enhanced) per Article 10-13, with dynamic reclassification, cost estimation, audit frequency recommendation, and regulatory submission requirement mapping.
6. **Trade flow analysis** -- Bilateral trade flow mapping, re-export risk detection (commodity laundering through low-risk countries), HS code mapping, trade route risk scoring, and sanction overlay.
7. **Risk report generation** -- Automated generation of country risk profiles, commodity-country matrices, executive summaries, comparative analyses, and regulatory submission-ready risk documentation in PDF/JSON/HTML with multi-language support.
8. **Regulatory update tracking** -- Monitoring of EC benchmarking list updates, country reclassification impact assessment, enforcement action tracking, and stakeholder notification on material changes.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Country coverage | 200+ countries scored | Count of countries in risk database |
| Commodity coverage | All 7 EUDR commodities per country | Commodity-country matrix completeness |
| Risk score accuracy | 100% deterministic, reproducible | Bit-perfect reproducibility tests |
| EC benchmark alignment | 100% match with published EC classifications | Validation against EC benchmarking list |
| Hotspot detection precision | >= 90% match with Global Forest Watch alerts | Cross-validation with GFW 2024 data |
| Due diligence classification accuracy | 100% correct per EUDR Articles 10-13 | Validation against regulatory test cases |
| Assessment throughput | < 500ms per country assessment | p99 latency under load |
| Report generation | < 5 seconds per country risk profile | Time from request to PDF/JSON delivery |
| Regulatory update latency | < 24 hours from EC publication to system update | Time from publication to risk score recalculation |
| EUDR compliance coverage | 100% of Article 29 requirements | Regulatory compliance matrix |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated country risk assessment and due diligence tools market of 1-2 billion EUR.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers requiring automated country risk assessment for 7 regulated commodities, estimated at 400-600M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 30-50M EUR in risk assessment module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) of EUDR-regulated commodities requiring automated risk assessment
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya)
- Timber and paper industry operators sourcing from tropical regions
- Compliance officers responsible for EUDR due diligence decisions

**Secondary:**
- Customs brokers and freight forwarders handling EUDR-regulated goods
- Commodity traders and intermediaries operating in multi-country supply chains
- Certification bodies (FSC, RSPO, Rainforest Alliance) validating deforestation-free claims
- Compliance consultants and auditors performing third-party EUDR verification
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Spreadsheet | No cost; familiar | Cannot synthesize multi-factor data; no commodity granularity; error-prone; no updates | Automated, deterministic, 200+ countries, 7 commodities, continuous updates |
| Generic ESG risk platforms (RepRisk, MSCI) | Broad ESG coverage; brand recognition | Not EUDR-specific; no Article 29 alignment; no commodity-specific scoring | Purpose-built for EUDR; Article 29 fidelity; commodity-specific risk |
| Global Forest Watch | Authoritative deforestation data | Data platform, not compliance tool; no risk scoring; no due diligence classification | Integrates GFW data into compliance workflow; automated DD classification |
| Niche EUDR tools (Preferred by Nature) | Commodity expertise | Single-commodity focus; manual processes; no sub-national hotspot detection | All 7 commodities; automated hotspot detection; trade flow analysis |
| Consulting firms (Deloitte, EY) | Expert judgment; credibility | Expensive (50K-200K EUR); slow (weeks); not scalable; point-in-time assessment | Automated; < 500ms per assessment; continuous monitoring; auditable |
| In-house custom builds | Tailored to org | 6-12 month build; no regulatory updates; no scale; data acquisition burden | Ready now; authoritative data sources integrated; production-grade |

### 2.4 Differentiation Strategy

1. **Regulatory fidelity** -- Every risk factor, weight, and classification maps to a specific EUDR Article and EC benchmarking methodology.
2. **Commodity granularity** -- Per-commodity risk profiles, not monolithic country scores. Brazil is HIGH for soya/cattle but STANDARD for coffee.
3. **Sub-national precision** -- Hotspot detection at province/state level, not just country level. Amazon vs. Cerrado vs. Atlantic Forest.
4. **Zero-hallucination risk calculations** -- Deterministic, auditable, reproducible scoring with no LLM in the critical path. Every score traceable to authoritative data.
5. **Integration depth** -- Pre-built integration with EUDR-001 (supply chain), EUDR-003 (satellite), EUDR-005 (land use), EUDR-008 (supplier tracker), and GL-EUDR-APP platform.
6. **Existing data foundation** -- Builds on 25+ country risk profiles and 85+ country risk scores already in production within the GreenLang platform.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to perform Article 29-compliant country risk assessment | 100% of customers pass Article 10/11 due diligence audits | Q2 2026 |
| BG-2 | Reduce time-to-assess country risk from days to seconds | 99.5% reduction in assessment time (days to < 500ms) | Q2 2026 |
| BG-3 | Become the reference country risk evaluation platform for EUDR | 500+ enterprise customers using risk module | Q4 2026 |
| BG-4 | Prevent compliance penalties for customers | Zero EUDR penalties attributable to incorrect risk assessment for active customers | Ongoing |
| BG-5 | Support trade flow risk detection for commodity laundering | Detect 90%+ of re-export risk patterns | Q3 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Comprehensive country scoring | Score 200+ countries with multi-factor weighted composite risk |
| PG-2 | Commodity-specific risk | Differentiate risk across all 7 EUDR commodities per country |
| PG-3 | Sub-national hotspot detection | Identify deforestation hotspots at province/state level |
| PG-4 | Governance integration | Systematically integrate WGI, CPI, and forest governance indicators |
| PG-5 | Automated DD classification | Classify due diligence level for every country-commodity combination |
| PG-6 | Trade flow risk analysis | Detect re-export risk and commodity laundering patterns |
| PG-7 | Audit-ready reporting | Generate compliant risk reports for regulatory submission |
| PG-8 | Regulatory update tracking | Monitor and propagate EC benchmarking changes within 24 hours |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Assessment throughput | < 500ms p99 per country risk assessment |
| TG-2 | Batch assessment | 200 countries in < 30 seconds |
| TG-3 | Report generation | < 5 seconds per country risk profile PDF |
| TG-4 | API response time | < 200ms p95 for standard queries |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |
| TG-7 | Data freshness | Risk scores updated within 24 hours of source data publication |

---

## 4. User Personas

### Persona 1: Compliance Officer -- Elena (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of EUDR Compliance at a large EU palm oil refinery |
| **Company** | 4,000 employees, importing from Indonesia, Malaysia, Colombia, Guatemala |
| **EUDR Pressure** | Board mandate to automate risk assessment; current manual process takes 2 weeks per quarterly review |
| **Pain Points** | Cannot differentiate sub-national risk (Sumatra vs. Java); no commodity-specific scoring; manual EC benchmark tracking; no audit trail for due diligence decisions |
| **Goals** | Automated country risk assessment for every shipment; sub-national hotspot alerting; audit-ready reports for quarterly DDS submission |
| **Technical Skill** | Moderate -- comfortable with web applications and dashboards |

### Persona 2: Risk Analyst -- Thomas (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Risk Analyst at a commodity trading firm |
| **Company** | 2,000 employees, trading cocoa, coffee, and soya across 40+ origin countries |
| **EUDR Pressure** | Must assess risk for every new sourcing origin; must detect commodity laundering via transshipment countries |
| **Pain Points** | No systematic trade flow risk analysis; country risk data scattered across multiple sources; no governance index integration; risk reports are ad hoc |
| **Goals** | Real-time risk scoring for new origins; trade flow analysis for re-export detection; comparative country risk reports; regulatory change notifications |
| **Technical Skill** | High -- comfortable with data tools, APIs, and analytical platforms |

### Persona 3: Sustainability Director -- Marie (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | VP Sustainability at a European chocolate manufacturer |
| **Company** | 8,000 employees, sourcing cocoa from Cote d'Ivoire, Ghana, Ecuador, Peru |
| **EUDR Pressure** | Must demonstrate deforestation-free sourcing to board and investors; West African sourcing under intense scrutiny |
| **Pain Points** | Cannot show improving risk trends over time; no visibility into certification scheme effectiveness per country; no comparative analysis across origins |
| **Goals** | Country risk trend reports for investor presentations; certification effectiveness scoring; peer comparison across cocoa origins |
| **Technical Skill** | Low-moderate -- uses dashboards and reports; does not write queries |

### Persona 4: External Auditor -- Dr. Fischer (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must verify operator risk assessment methodology and data quality |
| **Pain Points** | Operators provide inconsistent risk assessments; no standardized methodology; no provenance tracking for risk data sources |
| **Goals** | Access read-only risk assessment data with full provenance; verify data source citations; validate due diligence classification logic; audit risk score change history |
| **Technical Skill** | Moderate -- comfortable with audit software and report review |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 2(27-29)** | Definitions of "low-risk", "standard-risk", and "high-risk" country classifications | Three-tier risk classification with configurable thresholds aligned to EC benchmarking |
| **Art. 10(1)** | Risk assessment -- operators shall assess and identify risk of non-compliance | Composite country risk scoring engine (F1) with 6 weighted risk factors |
| **Art. 10(2)(a)** | Risk assessment criterion: complexity of relevant supply chain | Trade flow analysis engine (F6) scores supply chain route complexity |
| **Art. 10(2)(b)** | Risk assessment criterion: risk of circumvention or mixing | Re-export risk detection in trade flow analyzer (F6) identifies commodity laundering |
| **Art. 10(2)(c)** | Risk assessment criterion: risk of deforestation in the country of production | Deforestation hotspot detection (F3) with sub-national granularity |
| **Art. 10(2)(d)** | Risk assessment criterion: risk of non-compliance with country of production legislation | Governance index engine (F4) assesses legal framework and enforcement quality |
| **Art. 10(2)(e)** | Risk assessment criterion: concerns about the country of production or origin | Country risk scoring (F1) with EC benchmark alignment and trend analysis |
| **Art. 10(2)(f)** | Risk assessment criterion: risk of circumvention through mixing with unknown origin | Trade flow analyzer (F6) identifies transshipment and re-export patterns |
| **Art. 11** | Risk mitigation -- enhanced measures for high-risk scenarios | Due diligence classifier (F5) recommends mitigation actions per risk level |
| **Art. 13** | Simplified due diligence for low-risk countries | DD classifier (F5) identifies low-risk country-commodity combinations eligible for simplified DD |
| **Art. 29(1)** | EC shall classify countries into low, standard, or high risk | Country risk scoring engine (F1) aligns with EC benchmarking methodology |
| **Art. 29(2)(a)** | Benchmark criterion: rate of deforestation and forest degradation | Deforestation rate factor (30% weight) in composite score |
| **Art. 29(2)(b)** | Benchmark criterion: rate of expansion of agriculture for relevant commodities | Commodity-specific risk analyzer (F2) tracks production expansion vs. forest loss |
| **Art. 29(2)(c)** | Benchmark criterion: production trends of relevant commodities | Commodity risk analyzer (F2) analyzes production volume trends |
| **Art. 29(3)** | Criteria for country assessment: enforcement, respect of indigenous peoples, anti-corruption, UNFCCC participation | Governance index engine (F4) scores all four criteria separately |
| **Art. 31** | Record keeping for 5 years | Immutable audit log and risk score change history with 5-year retention |

### 5.2 EC Benchmarking Methodology (Article 29)

The European Commission assesses countries based on the following criteria, which this agent maps directly to risk factors:

| EC Criterion (Art. 29(2-3)) | Agent Risk Factor | Weight | Data Source |
|------------------------------|-------------------|--------|-------------|
| Rate of deforestation and forest degradation | `deforestation_rate` | 30% | FAO FRA 2025, Global Forest Watch 2024 |
| Enforcement of environmental legislation | `governance_index` | 20% | World Bank WGI 2024, FAO forest governance |
| Environmental law enforcement effectiveness | `enforcement_score` | 15% | Custom scoring from WGI, FLEGT status, enforcement data |
| Corruption perception | `corruption_index` | 15% | Transparency International CPI 2024 |
| Respect for indigenous peoples' rights | `forest_law_compliance` | 10% | ILO Convention 169 ratification, land tenure security |
| Historical deforestation trend | `historical_trend` | 10% | 5-year trend from GFW Hansen data (2019-2024) |

**Composite Score Formula:**
```
Composite_Risk_Score = (
    deforestation_rate * 0.30
    + (100 - governance_index) * 0.20
    + (100 - enforcement_score) * 0.15
    + corruption_index * 0.15
    + (100 - forest_law_compliance) * 0.10
    + historical_trend_risk * 0.10
)

Where:
- All inputs normalized to 0-100 scale
- Higher score = higher risk
- Classification thresholds:
  - LOW:      0-30
  - STANDARD: 31-65
  - HIGH:     66-100
```

### 5.3 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for all forest cover assessments |
| June 29, 2023 | Regulation entered into force | Legal basis for all risk classification |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | All country risk assessments must be operational |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; agent must handle 10x scale |
| Ongoing (periodic) | EC benchmarking list updates | Agent must consume and apply updated classifications within 24 hours |
| Ongoing (quarterly) | Operator due diligence statement submission | Risk reports must be current for quarterly DDS filing |

---

## 6. Features and Requirements

### 6.1 Scope -- In and Out

**In Scope (v1.0):**
- Composite country risk scoring for 200+ countries
- Commodity-specific risk analysis for all 7 EUDR commodities
- Sub-national deforestation hotspot detection
- Governance and institutional capacity assessment
- Automated due diligence level classification
- Trade flow analysis and re-export risk detection
- Risk report generation (PDF/JSON/HTML, multi-language)
- Regulatory update tracking and impact assessment
- Integration with EUDR-001, EUDR-003, EUDR-005, EUDR-008
- Consumption and extension of existing `eudr_country_risk.py` and `country_risk_scores.py` data

**Out of Scope (v1.0):**
- Real-time satellite imagery processing (defer to EUDR-003 integration)
- Carbon footprint calculation per country (defer to GL-GHG-APP integration)
- Predictive machine learning for future risk forecasting (defer to Phase 2)
- Mobile native application (web responsive design only)
- Blockchain-based immutable risk records (SHA-256 provenance hashes sufficient)
- Direct submission to EU Information System (handled by GL-EUDR-APP DDS module)

### 6.2 Zero-Hallucination Principles

All 8 engines in this agent operate under strict zero-hallucination guarantees:

| Principle | Implementation |
|-----------|---------------|
| **Deterministic calculations** | Same inputs always produce the same risk scores (bit-perfect reproducibility) |
| **No LLM in critical path** | All risk scoring, classification, and hotspot detection use deterministic algorithms only |
| **Authoritative data sources only** | All risk factors sourced from FAO, GFW, World Bank, TI, EC; no synthetic data |
| **Full provenance tracking** | Every risk score includes SHA-256 hash, data source citations, and calculation timestamps |
| **Immutable audit trail** | All risk score changes recorded in `gl_eudr_cre_audit_log` with before/after values |
| **Decimal arithmetic** | Risk scores use Decimal type to prevent floating-point drift |
| **Version-controlled data** | Risk databases are versioned; any change creates a new version with timestamp |

### 6.3 Must-Have Features (P0 -- Launch Blockers)

All 8 features below are P0 launch blockers. The agent cannot ship without all 8 features operational. Features 1-5 form the core risk assessment engine; Features 6-8 form the analysis, reporting, and monitoring layer.

**P0 Features 1-5: Core Risk Assessment Engine**

---

#### Feature 1: Country Risk Scoring Engine

**User Story:**
```
As a compliance officer,
I want a composite risk score for every country based on multiple authoritative data sources,
So that I can quickly determine the risk level for any import origin and satisfy EUDR Article 29 requirements.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F1.1: Implements EC benchmarking risk classification (low/standard/high) for 200+ countries, aligned with Article 29 criteria and configurable classification thresholds (default: low 0-30, standard 31-65, high 66-100)
- [ ] F1.2: Calculates composite risk score 0-100 from 6 weighted factors: deforestation rate (30%), governance index (20%), enforcement score (15%), corruption perception index (15%), forest law compliance (10%), historical trend (10%); all weights operator-configurable
- [ ] F1.3: Performs time-series risk trend analysis classifying each country as improving, stable, or deteriorating based on 5-year rolling window of annual risk scores
- [ ] F1.4: Assigns confidence level (high/medium/low) to each risk score based on data freshness (< 6 months = high, 6-12 months = medium, > 12 months = low) and source quality (primary/secondary/estimated)
- [ ] F1.5: Calculates comparison metrics against neighboring countries for regional context, including regional average, regional percentile rank, and highest/lowest risk neighbor identification
- [ ] F1.6: Supports custom weight profiles per operator/organization, allowing operators to adjust factor weights within permitted bounds (each factor 5-50%, total must equal 100%)
- [ ] F1.7: Normalizes risk scores to percentile ranking across all assessed countries, providing both absolute score (0-100) and relative percentile (0-100th percentile)
- [ ] F1.8: Tracks complete data source attribution and citation for every risk factor, including source name, publication date, access URL, and SHA-256 hash of source data file
- [ ] F1.9: Maintains version-controlled risk scores with immutable change history, recording previous score, new score, change reason, changed_by, and timestamp for every score update
- [ ] F1.10: Validates all risk scores against EC published benchmarks; any divergence between agent classification and EC classification is flagged as a critical alert requiring manual review

**Risk Calculation Formula:**
```
Composite_Risk_Score = (
    deforestation_rate_normalized * W_deforestation     # default 0.30
    + (100 - governance_index) * W_governance           # default 0.20
    + (100 - enforcement_score) * W_enforcement         # default 0.15
    + corruption_index * W_corruption                   # default 0.15
    + (100 - forest_law_compliance) * W_law_compliance  # default 0.10
    + historical_trend_risk * W_trend                   # default 0.10
)

Classification:
  LOW:      Composite_Risk_Score <= 30
  STANDARD: 30 < Composite_Risk_Score <= 65
  HIGH:     Composite_Risk_Score > 65
```

**Non-Functional Requirements:**
- Performance: < 50ms per single country assessment; < 30 seconds for full 200-country batch
- Determinism: Bit-perfect reproducibility across runs
- Precision: Decimal arithmetic for all score calculations
- Auditability: SHA-256 provenance hash on every assessment

**Dependencies:**
- Existing `greenlang/data/eudr_country_risk.py` (25+ countries, forest data, commodity risks)
- Existing `greenlang/agents/eudr/multi_tier_supplier/reference_data/country_risk_scores.py` (85+ countries, risk factor breakdowns)
- AGENT-FOUND-005 Citations & Evidence Agent for source attribution

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- Unknown country (not in database) -> Classify as STANDARD with confidence = low, flag for manual review
- Missing risk factor data -> Use regional average as fallback; document the imputation; reduce confidence level
- EC benchmark mismatch -> Agent classification differs from EC published list -> Critical alert, EC classification takes precedence

---

#### Feature 2: Commodity-Specific Risk Analyzer

**User Story:**
```
As a risk analyst,
I want per-commodity risk profiles for each country,
So that I can make sourcing decisions that account for the specific deforestation risk of each commodity in each origin.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F2.1: Provides commodity-specific risk profiles per country for all 7 EUDR-regulated commodities (cattle, cocoa, coffee, oil_palm, rubber, soya, wood), with individual risk scores (0-100) and risk levels (low/standard/high) per commodity-country pair
- [ ] F2.2: Incorporates production volume analysis and market share per commodity per country, sourcing data from FAO FAOSTAT, USDA FAS, and industry associations (ICO, ICCO, World Steel Association)
- [ ] F2.3: Calculates commodity-specific deforestation correlation factors quantifying the statistical relationship between commodity production expansion and forest loss per country (0.0-1.0 correlation coefficient)
- [ ] F2.4: Models seasonal risk variation capturing dry season burning patterns (fire risk index), harvest period activity spikes, and seasonal deforestation peaks per commodity-country combination
- [ ] F2.5: Computes commodity supply chain complexity score (1-10) based on typical tier depth, intermediary count, aggregation frequency, and traceability difficulty per commodity-country pair
- [ ] F2.6: Identifies cross-commodity risk correlations for linked commodities (e.g., soya/cattle in Brazil share deforestation frontiers; cocoa/rubber in Cote d'Ivoire share forest conversion), surfacing combined risk when both commodities are sourced from the same country
- [ ] F2.7: Rates certification scheme effectiveness per commodity-country combination (0-100), evaluating schemes (FSC, RSPO, Rainforest Alliance, RTRS, UTZ, Fairtrade) based on audit rigor, coverage, and historical performance
- [ ] F2.8: Provides origin-specific quality and legality indicators, including percentage of commodity legally produced, percentage traceable to origin, and percentage covered by recognized certification schemes
- [ ] F2.9: Tracks commodity price-deforestation correlation using 5-year price data overlaid with deforestation rate, identifying periods where price spikes drove increased clearing
- [ ] F2.10: Assesses commodity-specific regulatory framework per country, including existenceirtd enforcement of commodity-specific regulations (e.g., Brazil Forest Code, Indonesia peatland moratorium, Malaysia MSPO mandate)

**Non-Functional Requirements:**
- Coverage: Risk profiles for all 7 commodities x all countries where production occurs
- Performance: < 100ms per commodity-country assessment
- Data Quality: Confidence flag on every data point; fallback to regional average when country data unavailable

**Dependencies:**
- Feature 1 (Country Risk Scoring Engine) for base country scores
- Existing `commodity_risks` dictionaries in `eudr_country_risk.py`
- AGENT-DATA-009 Spend Data Categorizer for commodity classification

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data engineer)

---

#### Feature 3: Deforestation Hotspot Detector

**User Story:**
```
As a compliance officer,
I want to identify specific sub-national regions where deforestation is concentrated,
So that I can apply enhanced due diligence to imports originating from those specific hotspots.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F3.1: Integrates Global Forest Watch data feeds for near-real-time deforestation alert processing, consuming GFW GLAD alerts, RADD alerts, and integrated deforestation alerts with < 48-hour latency from GFW publication
- [ ] F3.2: Processes Hansen tree cover loss data (2001-present) to establish historical baselines, annual loss rates, and cumulative loss since the EUDR cutoff date (December 31, 2020) at 30-meter resolution aggregated to sub-national administrative boundaries
- [ ] F3.3: Classifies sub-national regions (province/state level) into low/standard/high risk using the same 3-tier framework as country-level classification, with region-specific composite scores that may differ from the parent country score
- [ ] F3.4: Performs spatial clustering of deforestation alerts using DBSCAN-like algorithm to identify hotspot boundaries, with configurable parameters for minimum cluster size (default: 10 alerts), maximum distance between points (default: 5km), and temporal window (default: 12 months)
- [ ] F3.5: Computes historical deforestation trend per hotspot (improving/stable/deteriorating) using linear regression on annual tree cover loss data, with statistical significance testing (p < 0.05)
- [ ] F3.6: Correlates fire alerts from FIRMS/VIIRS data sources with deforestation patterns, computing fire-deforestation correlation coefficient and flagging hotspots where fire activity precedes forest clearing (indicative of slash-and-burn conversion)
- [ ] F3.7: Calculates protected area proximity scores for each hotspot, measuring minimum distance to nearest protected area boundary, percentage of hotspot overlapping protected areas, and listing affected protected area names and IUCN categories
- [ ] F3.8: Performs indigenous territory overlap detection using RAISG, LandMark, and national indigenous territory databases, flagging hotspots that intersect or are within 10km of recognized indigenous territories
- [ ] F3.9: Attributes primary deforestation drivers per hotspot from a controlled vocabulary (agriculture_expansion, cattle_ranching, palm_oil_plantation, logging_legal, logging_illegal, mining, infrastructure, fire, smallholder_agriculture), using a deterministic rule-based classifier based on commodity production data and land use patterns
- [ ] F3.10: Supports configurable alert thresholds per region/commodity, allowing operators to set custom notification triggers (e.g., "alert me if > 100 hectares lost in West Kalimantan in any 30-day period for palm oil sourcing")

**Non-Functional Requirements:**
- Spatial precision: Hotspot boundaries accurate to 1km resolution
- Latency: Hotspot detection completes in < 10 seconds for any country
- Data: Process 5+ years of historical data for trend analysis
- Auditability: Every hotspot detection includes data source, algorithm version, and timestamp

**Dependencies:**
- AGENT-DATA-007 Deforestation Satellite Connector for GFW/FIRMS data
- AGENT-DATA-006 GIS/Mapping Connector for spatial operations
- Existing `REGION_RISK_DATABASE` in `eudr_country_risk.py` (Brazil, Indonesia, Cote d'Ivoire regions)

**Estimated Effort:** 4 weeks (1 backend engineer, 1 GIS specialist)

---

#### Feature 4: Governance Index Engine

**User Story:**
```
As a risk analyst,
I want systematic governance and institutional capacity scores for each country,
So that I can assess whether a country's governance framework provides adequate protection against deforestation.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F4.1: Integrates World Bank Worldwide Governance Indicators (WGI) for all 6 dimensions: Voice and Accountability, Political Stability, Government Effectiveness, Regulatory Quality, Rule of Law, and Control of Corruption; normalized to 0-100 scale
- [ ] F4.2: Integrates Transparency International Corruption Perceptions Index (CPI), normalized to 0-100 corruption risk scale (inverted: CPI score 100 = 0 corruption risk, CPI score 0 = 100 corruption risk)
- [ ] F4.3: Performs forest governance assessment using FAO/ITTO framework criteria: forest policy and legislation quality, institutional arrangements, financial mechanisms, enforcement capacity, and transparency/accountability; scored 0-100
- [ ] F4.4: Scores legal framework strength (0-100) based on: existence of forest protection law (20%), land tenure clarity (20%), environmental impact assessment requirements (20%), indigenous rights recognition (20%), and penalty framework adequacy (20%)
- [ ] F4.5: Evaluates environmental law enforcement effectiveness (0-100) based on: prosecution rates for forest crimes, fines collected vs. imposed, number of enforcement officers per forest area, satellite monitoring utilization, and inter-agency coordination
- [ ] F4.6: Assesses cross-border enforcement cooperation capability (0-100) based on: INTERPOL participation, bilateral enforcement agreements, FLEGT VPA status (negotiating/implementing/in-force), and Mutual Legal Assistance Treaty coverage
- [ ] F4.7: Scores indigenous peoples' rights recognition (0-100) based on: ILO Convention 169 ratification, FPIC (Free Prior Informed Consent) legal requirement, indigenous territory demarcation percentage, and land dispute resolution mechanisms
- [ ] F4.8: Evaluates environmental impact assessment (EIA) requirement compliance (0-100) based on: EIA legal requirement for forest conversion, EIA enforcement rate, public consultation requirements, and cumulative impact assessment capability
- [ ] F4.9: Scores judicial independence and rule of law (0-100) using WGI Rule of Law indicator, judiciary corruption perception, environmental court existence, and average time-to-resolution for forest crime cases
- [ ] F4.10: Scores government transparency and data availability (0-100) based on: forest inventory publication frequency, deforestation data transparency, open data portal availability, REDD+ reporting compliance, and UNFCCC National Communication timeliness

**Non-Functional Requirements:**
- Coverage: Governance scores for all 200+ countries in risk database
- Update Frequency: Annual refresh aligned with WGI and CPI publication cycles
- Data Quality: Source attribution for every indicator score

**Dependencies:**
- World Bank Open Data API for WGI indicators
- Transparency International CPI dataset
- FAO Global Forest Resources Assessment
- ITTO status reports

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data analyst)

---

#### Feature 5: Due Diligence Level Classifier

**User Story:**
```
As a compliance officer,
I want the system to automatically determine the required due diligence level for each import,
So that I can ensure my company performs the correct level of due diligence as required by EUDR Articles 10-13.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F5.1: Implements 3-tier due diligence classification per EUDR Articles 10-13: simplified due diligence (Art. 13, low-risk countries), standard due diligence (Arts. 10-11, standard-risk), and enhanced due diligence (Art. 11, high-risk with mandatory satellite verification)
- [ ] F5.2: Dynamically reclassifies due diligence level when underlying risk scores change, triggering automatic recalculation when country scores, commodity scores, or EC benchmarks are updated
- [ ] F5.3: Classifies at the granular level of country + commodity + sub-national origin combination, recognizing that the same country may require different DD levels for different commodities or regions (e.g., Brazil: enhanced DD for soya from Cerrado, standard DD for coffee from Minas Gerais)
- [ ] F5.4: Implements override rules for specific high-risk areas regardless of country classification, allowing administrators to force enhanced DD for specific regions (e.g., any import from a deforestation hotspot flagged by F3 automatically requires enhanced DD)
- [ ] F5.5: Calculates certification-based risk mitigation credit (0-30 points reduction), recognizing that credible certification (FSC, RSPO, Rainforest Alliance with score >= 70 in F2.7) can reduce the effective risk score, potentially lowering the required DD level
- [ ] F5.6: Provides due diligence cost estimation per classification level, using configurable cost models: simplified DD (1-2 hours, EUR 200-500), standard DD (4-8 hours, EUR 1,000-3,000), enhanced DD (16-40 hours, EUR 5,000-15,000) per shipment
- [ ] F5.7: Maps regulatory submission requirements per DD level, specifying which documentation, declarations, and evidence must be collected and included in the DDS for each level
- [ ] F5.8: Estimates time-to-compliance per due diligence level, accounting for data collection lead times, satellite verification processing, supplier questionnaire response times, and report generation
- [ ] F5.9: Recommends audit frequency per classification: simplified (annual audit), standard (semi-annual audit), enhanced (quarterly audit with satellite re-verification), adjustable per operator risk appetite
- [ ] F5.10: Generates classification change notifications and impact analysis when a country or commodity is reclassified, computing the number of affected active imports, estimated cost impact, and compliance action timeline

**Due Diligence Classification Matrix:**

| Risk Level | DD Level | Key Requirements | Satellite Required | Audit Frequency |
|------------|----------|-----------------|-------------------|-----------------|
| Low (0-30) | Simplified | Basic documentation, supplier declarations | No | Annual |
| Standard (31-65) | Standard | Full documentation, risk assessment, supplier verification | Recommended | Semi-annual |
| High (66-100) | Enhanced | Full documentation, satellite verification, independent audit, supplier site visits | Mandatory | Quarterly |

**Non-Functional Requirements:**
- Accuracy: 100% correct classification validated against EUDR regulatory test cases
- Speed: < 10ms per classification decision
- Auditability: Every classification includes decision rationale with risk score breakdown

**Dependencies:**
- Feature 1 (Country Risk Scoring)
- Feature 2 (Commodity-Specific Risk)
- Feature 3 (Hotspot Detection) for region-level overrides
- Existing `DueDiligenceLevel` enum and `get_due_diligence_level()` in `eudr_country_risk.py`

**Estimated Effort:** 2 weeks (1 backend engineer)

---

**P0 Features 6-8: Analysis, Reporting, and Monitoring Layer**

> Features 6, 7, and 8 are P0 launch blockers. Without trade flow analysis, risk reporting, and regulatory update tracking, the core risk assessment engine cannot deliver end-user value. These features are the delivery mechanism through which compliance officers, analysts, and auditors interact with the risk engine.

---

#### Feature 6: Trade Flow Analyzer

**User Story:**
```
As a risk analyst,
I want to analyze commodity trade flows between countries,
So that I can detect re-export risk, commodity laundering patterns, and high-risk trade routes.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F6.1: Maps bilateral trade flows (country-to-country) for all 7 EUDR commodities using UN Comtrade, Eurostat, and national customs data, covering 200+ origin countries to 27 EU member states
- [ ] F6.2: Tracks EU import volume per commodity per origin country with quarterly granularity, computing year-over-year change, market share, and concentration risk (Herfindahl-Hirschman Index)
- [ ] F6.3: Scores trade route risk based on transshipment countries along the route, applying the "highest-risk-in-chain" principle where a route through a high-risk transshipment country inherits that risk
- [ ] F6.4: Detects re-export risk (commodity laundering) by identifying patterns where a low-risk country exports significantly more of a commodity than it produces, flagging import-for-re-export patterns with statistical anomaly detection
- [ ] F6.5: Maps HS/CN codes to EUDR commodities and derived products using the EUDR Annex I mapping (1,200+ product codes), supporting both 6-digit HS and 8-digit CN code lookups
- [ ] F6.6: Performs trade flow trend analysis identifying volume changes (surge detection), new trade routes (route emergence), and diversification patterns (concentration decrease) that may indicate circumvention or market shifts
- [ ] F6.7: Overlays EU sanction and embargo data on trade flows, flagging commodities transiting through or originating from sanctioned entities/countries (e.g., Russia wood sanctions)
- [ ] F6.8: Assesses free trade agreement (FTA) impact on EUDR commodity flows, identifying preferential trade routes and potential regulatory arbitrage
- [ ] F6.9: Profiles EU ports of entry by risk, scoring each port based on volume of high-risk commodity imports, inspection rate, and historical compliance violations
- [ ] F6.10: Generates trade documentation requirement matrix per route, specifying required certificates, declarations, and supporting evidence based on origin country risk, commodity type, and EU member state requirements

**Non-Functional Requirements:**
- Coverage: Trade flow data for all 7 commodities, 200+ origins, 27 EU destinations
- Performance: Trade flow query < 2 seconds for any bilateral route
- Data Freshness: Quarterly trade data updates aligned with Comtrade releases

**Dependencies:**
- Feature 1 (Country Risk Scoring) for origin/transshipment country risk
- UN Comtrade API for bilateral trade data
- Eurostat for EU import statistics
- EUDR Annex I for HS/CN code mapping

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data engineer)

---

#### Feature 7: Risk Report Generator

**User Story:**
```
As a compliance officer,
I want automated, audit-ready risk assessment reports,
So that I can include them in my Due Diligence Statement and present them to auditors and regulators.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F7.1: Generates comprehensive country risk profile reports in PDF, JSON, and HTML formats, including composite score, all 6 risk factor scores, commodity risk matrix, hotspot map, governance indicators, and trend analysis
- [ ] F7.2: Generates commodity-country matrix reports showing risk levels for all 7 commodities across selected countries or all countries, with color-coded risk heatmap and sortable columns
- [ ] F7.3: Produces executive summary reports with key risk indicators: top 10 highest-risk origin countries for operator's commodities, most significant risk changes in past quarter, due diligence cost projection, and compliance readiness score
- [ ] F7.4: Generates comparative analysis reports comparing a country against regional peers, global average, and operator-specified benchmark countries, with spider/radar charts for visual risk factor comparison
- [ ] F7.5: Produces temporal trend reports showing risk score evolution over time per country, with annotated events (regulatory changes, deforestation spikes, governance improvements), presented as time-series charts
- [ ] F7.6: Generates due diligence requirement reports per import scenario, detailing the specific documentation, evidence, and verification steps required based on the classified DD level, formatted as actionable checklists
- [ ] F7.7: Creates regulatory submission-ready risk documentation formatted to meet DDS requirements under EUDR Article 4(2), including structured risk assessment narrative, data source citations, and methodology description
- [ ] F7.8: Exports dashboard data packages for BI integration (Grafana, Tableau, Power BI) in CSV, JSON, and XLSX formats with standardized schema for risk metrics
- [ ] F7.9: Maintains complete audit trail of risk assessment decisions within reports, linking every score to source data, calculation method, and assessor (system or manual override), with SHA-256 provenance hashes
- [ ] F7.10: Supports multi-language report generation in English (EN), French (FR), German (DE), Spanish (ES), and Portuguese (PT) using translated templates with language-appropriate regulatory terminology

**Non-Functional Requirements:**
- Performance: PDF generation < 5 seconds per country profile; matrix report < 10 seconds for 50 countries
- Quality: Reports pass WCAG 2.1 AA accessibility standards
- Size: PDF reports optimized to < 5 MB each

**Dependencies:**
- Features 1-6 for all risk assessment data
- Report generation library (WeasyPrint or ReportLab for PDF)
- Jinja2 templates for multi-format output
- i18n framework for multi-language support

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend/template engineer)

---

#### Feature 8: Regulatory Update Tracker

**User Story:**
```
As a compliance officer,
I want to be automatically notified when the EC reclassifies a country or updates EUDR guidance,
So that I can immediately reassess my risk exposure and adjust due diligence procedures.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F8.1: Monitors EC benchmarking list publications for country risk classification updates, polling the EC EUDR regulatory portal at configurable intervals (default: daily) and parsing updates from official EC communications
- [ ] F8.2: Tracks and records every country reclassification (risk level change) with previous classification, new classification, effective date, EC reference document, and calculated impact on operator's active imports
- [ ] F8.3: Detects new EUDR implementing regulations, delegated acts, and amendments published in the Official Journal of the European Union, indexing them by topic, affected articles, and effective date
- [ ] F8.4: Performs impact assessment of regulatory changes on existing risk scores, computing: number of affected countries, number of affected active imports for the operator, estimated cost impact of DD level changes, and required compliance action timeline
- [ ] F8.5: Tracks grace period and transition timelines for regulatory changes, maintaining a calendar of compliance deadlines per change, with configurable reminder periods (default: 90, 60, 30, 7 days before deadline)
- [ ] F8.6: Indexes regulatory FAQ documents, EC guidance documents, and implementing technical specifications, making them searchable by keyword, article reference, commodity, and country
- [ ] F8.7: Tracks enforcement actions per country, recording penalties imposed, products seized, companies sanctioned, and enforcement trends, as indicators of actual regulatory enforcement intensity
- [ ] F8.8: Cross-references EUDR country classifications with other forest regulations: UK Environment Act 2021, US Lacey Act, Australian Illegal Logging Prohibition Act, identifying countries that face enhanced scrutiny across multiple jurisdictions
- [ ] F8.9: Maintains regulatory calendar with all upcoming EUDR deadlines, EC review dates, benchmarking update cycles, and member state implementation milestones, exportable as iCal feed
- [ ] F8.10: Sends stakeholder notifications on material regulatory changes via configurable channels (email, webhook, in-app notification), with notification severity levels (critical/important/informational) and per-operator subscription preferences

**Non-Functional Requirements:**
- Latency: Regulatory updates detected within 24 hours of publication
- Completeness: 100% of EC-published EUDR updates captured
- Retention: All regulatory records retained for minimum 5 years per Article 31

**Dependencies:**
- EC EUDR regulatory portal
- Official Journal of the European Union (EUR-Lex)
- GL-EUDR-APP notification service
- AGENT-FOUND-005 Citations & Evidence Agent for document indexing

**Estimated Effort:** 2 weeks (1 backend engineer)

---

### 6.4 Could-Have Features (P2 -- Nice to Have)

#### Feature 9: Predictive Risk Forecasting
- Machine learning models to forecast country risk trajectory
- Scenario modeling for policy changes (e.g., "What if Indonesia strengthens enforcement?")
- Early warning system for emerging deforestation frontiers
- Climate change impact on future deforestation risk

#### Feature 10: Supplier-Country Risk Correlation
- Link individual supplier risk scores to country risk baselines
- Identify suppliers that outperform or underperform their country's baseline
- Supplier portfolio risk concentration analysis by country
- Supplier switching recommendations based on country risk changes

#### Feature 11: Interactive Risk Map Dashboard
- Global risk heatmap with drill-down to country and sub-national level
- Commodity flow visualization (Sankey diagrams of trade routes)
- Real-time deforestation alert overlay on world map
- Customizable dashboard with operator-specific KPIs

---

### 6.5 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Real-time satellite imagery processing (EUDR-003 provides this)
- Carbon emissions calculation per country (GL-GHG-APP scope)
- Social risk assessment beyond indigenous rights (CSDDD scope)
- Predictive ML models for deforestation (defer to Phase 2)
- Mobile native application (web responsive only)
- Direct EU Information System submission (GL-EUDR-APP DDS module)
- Automated supplier deselection recommendations (legal liability concern)

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
| AGENT-EUDR-016        |           | AGENT-EUDR-001            |           | AGENT-DATA-007        |
| Country Risk          |<--------->| Supply Chain Mapping       |<--------->| Deforestation         |
| Evaluator             |           | Master                    |           | Satellite Connector   |
|                       |           |                           |           |                       |
| - Risk Scoring Engine |           | - Graph Engine             |           | - GFW Client          |
| - Commodity Analyzer  |           | - Risk Propagation         |           | - FIRMS/VIIRS Client  |
| - Hotspot Detector    |           | - Gap Analysis             |           | - Hansen Data Client  |
| - Governance Engine   |           | - Visualization API        |           | - Forest Change Det.  |
| - DD Classifier       |           |                           |           |                       |
| - Trade Flow Analyzer |           +---------------------------+           +-----------------------+
| - Report Generator    |
| - Regulatory Tracker  |           +---------------------------+           +---------------------------+
+-----------+-----------+           | AGENT-EUDR-005            |           | AGENT-EUDR-008            |
            |                       | Land Use Change Detector  |           | Multi-Tier Supplier       |
            |                       |                           |           | Tracker                   |
+-----------v-----------+           | - Land Use Classification |           | - Supplier Risk Scoring   |
| Existing Risk Data    |           | - Change Detection        |           | - Sub-tier Discovery      |
|                       |           +---------------------------+           +---------------------------+
| - eudr_country_risk.py|
| - country_risk_scores |           +---------------------------+
|   .py                 |           | AGENT-DATA-006            |
+-----------------------+           | GIS/Mapping Connector     |
                                    | - Spatial Operations      |
                                    | - Protected Area DB       |
                                    +---------------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/country_risk_evaluator/
    __init__.py                              # Public API exports
    config.py                                # CountryRiskEvaluatorConfig with GL_EUDR_CRE_ env prefix
    models.py                                # Pydantic v2 models for risk data
    country_risk_scorer.py                   # CountryRiskScorer: composite scoring engine (F1)
    commodity_risk_analyzer.py               # CommodityRiskAnalyzer: per-commodity risk (F2)
    deforestation_hotspot_detector.py        # DeforestationHotspotDetector: sub-national hotspots (F3)
    governance_index_engine.py               # GovernanceIndexEngine: governance assessment (F4)
    due_diligence_classifier.py              # DueDiligenceClassifier: DD level determination (F5)
    trade_flow_analyzer.py                   # TradeFlowAnalyzer: trade flow risk analysis (F6)
    risk_report_generator.py                 # RiskReportGenerator: report generation (F7)
    regulatory_update_tracker.py             # RegulatoryUpdateTracker: regulatory monitoring (F8)
    provenance.py                            # ProvenanceTracker: SHA-256 hash chains
    metrics.py                               # 18 Prometheus self-monitoring metrics
    setup.py                                 # CountryRiskEvaluatorService facade
    reference_data/
        __init__.py
        country_database.py                  # Extended country risk database (200+ countries)
        governance_indices.py                # WGI, CPI, forest governance data
        commodity_production.py              # Production volumes per country per commodity
        trade_flow_data.py                   # Bilateral trade flow baselines
        hs_code_mapping.py                   # HS/CN code to EUDR commodity mapping
        certification_schemes.py             # Certification scheme effectiveness data
        protected_areas.py                   # Protected area boundaries and metadata
        indigenous_territories.py            # Indigenous territory data
    api/
        __init__.py
        router.py                            # FastAPI router (~37 endpoints)
        schemas.py                           # API request/response Pydantic schemas
        dependencies.py                      # FastAPI dependency injection
        country_routes.py                    # Country risk assessment endpoints
        commodity_routes.py                  # Commodity risk analysis endpoints
        hotspot_routes.py                    # Deforestation hotspot endpoints
        governance_routes.py                 # Governance index endpoints
        due_diligence_routes.py              # Due diligence classification endpoints
        trade_flow_routes.py                 # Trade flow analysis endpoints
        report_routes.py                     # Report generation endpoints
        regulatory_routes.py                 # Regulatory update endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Risk Levels (aligned with existing enums in eudr_country_risk.py)
class RiskLevel(str, Enum):
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"

class DueDiligenceLevel(str, Enum):
    SIMPLIFIED = "simplified"       # Art. 13 -- low risk
    STANDARD = "standard"           # Arts. 10-11 -- standard risk
    ENHANCED = "enhanced"           # Art. 11 -- high risk

class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"

class ConfidenceLevel(str, Enum):
    HIGH = "high"           # Data < 6 months old, primary source
    MEDIUM = "medium"       # Data 6-12 months old, or secondary source
    LOW = "low"             # Data > 12 months old, or estimated

# Country Risk Assessment
class CountryRiskAssessment(BaseModel):
    assessment_id: str                          # UUID
    country_code: str                           # ISO 3166-1 alpha-2
    country_name: str                           # Full name
    composite_score: Decimal                    # 0-100 (Decimal for precision)
    risk_level: RiskLevel                       # low/standard/high
    risk_factors: Dict[str, Decimal]            # 6 factor scores
    factor_weights: Dict[str, Decimal]          # 6 factor weights (sum = 1.0)
    trend: TrendDirection                       # improving/stable/deteriorating
    confidence: ConfidenceLevel                 # high/medium/low
    percentile_rank: int                        # 0-100 (vs. all countries)
    regional_context: Dict[str, Any]            # regional avg, rank, neighbors
    data_sources: List[DataSourceCitation]       # provenance
    ec_benchmark_alignment: bool                # matches EC published list
    provenance_hash: str                        # SHA-256
    assessed_at: datetime
    version: int

# Commodity Risk Profile
class CommodityRiskProfile(BaseModel):
    profile_id: str
    country_code: str
    commodity: EUDRCommodity                    # cattle/cocoa/coffee/oil_palm/rubber/soya/wood
    risk_score: Decimal                         # 0-100
    risk_level: RiskLevel
    production_volume_mt: Optional[Decimal]     # metric tonnes
    market_share_pct: Optional[Decimal]         # % of global production
    deforestation_correlation: Decimal          # 0.0-1.0
    seasonal_risk_index: Dict[str, Decimal]     # month -> risk
    supply_chain_complexity: int                # 1-10
    certification_effectiveness: Decimal        # 0-100
    regulatory_framework_score: Decimal         # 0-100
    provenance_hash: str
    assessed_at: datetime

# Deforestation Hotspot
class DeforestationHotspot(BaseModel):
    hotspot_id: str
    country_code: str
    region_name: str                            # Province/state name
    risk_score: Decimal                         # 0-100
    risk_level: RiskLevel
    center_lat: float
    center_lon: float
    area_km2: float
    tree_cover_loss_since_cutoff_km2: Decimal   # Since Dec 31, 2020
    annual_loss_rate_pct: Decimal
    trend: TrendDirection
    primary_drivers: List[str]                  # Controlled vocabulary
    protected_area_overlap_pct: Decimal
    protected_areas_affected: List[str]
    indigenous_territory_overlap: bool
    indigenous_territories_affected: List[str]
    fire_correlation: Decimal                   # 0.0-1.0
    primary_commodities: List[EUDRCommodity]
    alert_count_12m: int
    detected_at: datetime

# Due Diligence Classification
class DueDiligenceClassification(BaseModel):
    classification_id: str
    country_code: str
    commodity: EUDRCommodity
    region: Optional[str]                       # Sub-national region
    dd_level: DueDiligenceLevel
    country_risk_score: Decimal
    commodity_risk_score: Decimal
    region_override: bool                       # Hotspot override applied
    certification_credit: Decimal               # 0-30 points reduction
    effective_risk_score: Decimal               # After credits/overrides
    cost_estimate_eur: Decimal                  # Estimated DD cost
    time_estimate_hours: Decimal                # Estimated DD hours
    audit_frequency: str                        # annual/semi-annual/quarterly
    required_documentation: List[str]
    satellite_verification_required: bool
    decision_rationale: str
    provenance_hash: str
    classified_at: datetime

# Trade Flow Record
class TradeFlowRecord(BaseModel):
    flow_id: str
    origin_country: str
    destination_country: str
    commodity: EUDRCommodity
    volume_mt: Decimal
    value_eur: Optional[Decimal]
    hs_code: str
    period: str                                 # YYYY-QN format
    route_risk_score: Decimal
    transshipment_countries: List[str]
    re_export_risk: bool
    re_export_confidence: Decimal               # 0.0-1.0
    data_source: str
```

### 7.4 Database Schema (New Migration: V104)

```sql
-- =========================================================================
-- V104: AGENT-EUDR-016 Country Risk Evaluator Schema
-- Agent: GL-EUDR-CRE-016
-- Tables: 12 (3 hypertables)
-- Prefix: gl_eudr_cre_
-- =========================================================================

CREATE SCHEMA IF NOT EXISTS eudr_country_risk_evaluator;

-- 1. Country Risk Assessments (hypertable on assessed_at)
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_country_risks (
    assessment_id UUID DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    composite_score NUMERIC(5,2) NOT NULL CHECK (composite_score >= 0 AND composite_score <= 100),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'standard', 'high')),
    deforestation_rate_score NUMERIC(5,2) NOT NULL,
    governance_index_score NUMERIC(5,2) NOT NULL,
    enforcement_score NUMERIC(5,2) NOT NULL,
    corruption_index_score NUMERIC(5,2) NOT NULL,
    forest_law_compliance_score NUMERIC(5,2) NOT NULL,
    historical_trend_score NUMERIC(5,2) NOT NULL,
    factor_weights JSONB NOT NULL DEFAULT '{"deforestation_rate":0.30,"governance_index":0.20,"enforcement_score":0.15,"corruption_index":0.15,"forest_law_compliance":0.10,"historical_trend":0.10}',
    trend VARCHAR(20) NOT NULL CHECK (trend IN ('improving', 'stable', 'deteriorating')),
    confidence VARCHAR(10) NOT NULL CHECK (confidence IN ('high', 'medium', 'low')),
    percentile_rank INTEGER CHECK (percentile_rank >= 0 AND percentile_rank <= 100),
    regional_context JSONB DEFAULT '{}',
    data_sources JSONB NOT NULL DEFAULT '[]',
    ec_benchmark_alignment BOOLEAN DEFAULT TRUE,
    provenance_hash VARCHAR(64) NOT NULL,
    version INTEGER DEFAULT 1,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (assessment_id, assessed_at)
);

SELECT create_hypertable('eudr_country_risk_evaluator.gl_eudr_cre_country_risks', 'assessed_at');

-- 2. Commodity-Specific Risks
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_commodity_risks (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    commodity VARCHAR(20) NOT NULL CHECK (commodity IN ('cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood')),
    risk_score NUMERIC(5,2) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'standard', 'high')),
    production_volume_mt NUMERIC(18,2),
    market_share_pct NUMERIC(5,2),
    deforestation_correlation NUMERIC(4,3) CHECK (deforestation_correlation >= 0 AND deforestation_correlation <= 1),
    seasonal_risk_index JSONB DEFAULT '{}',
    supply_chain_complexity INTEGER CHECK (supply_chain_complexity >= 1 AND supply_chain_complexity <= 10),
    certification_effectiveness NUMERIC(5,2),
    regulatory_framework_score NUMERIC(5,2),
    provenance_hash VARCHAR(64) NOT NULL,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (country_code, commodity)
);

-- 3. Deforestation Hotspot Records (hypertable on detected_at)
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_hotspots (
    hotspot_id UUID DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    region_name VARCHAR(200) NOT NULL,
    risk_score NUMERIC(5,2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    center_lat DOUBLE PRECISION NOT NULL,
    center_lon DOUBLE PRECISION NOT NULL,
    area_km2 NUMERIC(12,2),
    tree_cover_loss_since_cutoff_km2 NUMERIC(12,2),
    annual_loss_rate_pct NUMERIC(5,3),
    trend VARCHAR(20) NOT NULL CHECK (trend IN ('improving', 'stable', 'deteriorating')),
    primary_drivers JSONB DEFAULT '[]',
    protected_area_overlap_pct NUMERIC(5,2) DEFAULT 0,
    protected_areas_affected JSONB DEFAULT '[]',
    indigenous_territory_overlap BOOLEAN DEFAULT FALSE,
    indigenous_territories_affected JSONB DEFAULT '[]',
    fire_correlation NUMERIC(4,3) DEFAULT 0,
    primary_commodities JSONB DEFAULT '[]',
    alert_count_12m INTEGER DEFAULT 0,
    cluster_config JSONB DEFAULT '{}',
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (hotspot_id, detected_at)
);

SELECT create_hypertable('eudr_country_risk_evaluator.gl_eudr_cre_hotspots', 'detected_at');

-- 4. Governance Indicator Scores
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_governance_indices (
    index_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    wgi_voice_accountability NUMERIC(5,2),
    wgi_political_stability NUMERIC(5,2),
    wgi_government_effectiveness NUMERIC(5,2),
    wgi_regulatory_quality NUMERIC(5,2),
    wgi_rule_of_law NUMERIC(5,2),
    wgi_control_of_corruption NUMERIC(5,2),
    cpi_score NUMERIC(5,2),
    forest_governance_score NUMERIC(5,2),
    legal_framework_score NUMERIC(5,2),
    enforcement_effectiveness NUMERIC(5,2),
    cross_border_cooperation NUMERIC(5,2),
    indigenous_rights_score NUMERIC(5,2),
    eia_compliance NUMERIC(5,2),
    judicial_independence NUMERIC(5,2),
    transparency_score NUMERIC(5,2),
    composite_governance_score NUMERIC(5,2) NOT NULL,
    data_sources JSONB NOT NULL DEFAULT '[]',
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (country_code)
);

-- 5. Due Diligence Level Classifications
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_due_diligence_levels (
    classification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    commodity VARCHAR(20) NOT NULL,
    region VARCHAR(200),
    dd_level VARCHAR(20) NOT NULL CHECK (dd_level IN ('simplified', 'standard', 'enhanced')),
    country_risk_score NUMERIC(5,2) NOT NULL,
    commodity_risk_score NUMERIC(5,2) NOT NULL,
    region_override BOOLEAN DEFAULT FALSE,
    certification_credit NUMERIC(5,2) DEFAULT 0,
    effective_risk_score NUMERIC(5,2) NOT NULL,
    cost_estimate_eur NUMERIC(12,2),
    time_estimate_hours NUMERIC(6,1),
    audit_frequency VARCHAR(30),
    required_documentation JSONB DEFAULT '[]',
    satellite_verification_required BOOLEAN DEFAULT FALSE,
    decision_rationale TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    classified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (country_code, commodity, region)
);

-- 6. Bilateral Trade Flow Data
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_trade_flows (
    flow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    origin_country CHAR(2) NOT NULL,
    destination_country CHAR(2) NOT NULL,
    commodity VARCHAR(20) NOT NULL,
    volume_mt NUMERIC(18,2),
    value_eur NUMERIC(18,2),
    hs_code VARCHAR(20),
    period VARCHAR(10) NOT NULL,
    route_risk_score NUMERIC(5,2),
    transshipment_countries JSONB DEFAULT '[]',
    re_export_risk BOOLEAN DEFAULT FALSE,
    re_export_confidence NUMERIC(4,3) DEFAULT 0,
    data_source VARCHAR(200),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 7. Generated Risk Reports
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_risk_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    format VARCHAR(10) NOT NULL CHECK (format IN ('pdf', 'json', 'html', 'csv', 'xlsx')),
    language VARCHAR(5) NOT NULL DEFAULT 'en',
    country_codes JSONB DEFAULT '[]',
    commodities JSONB DEFAULT '[]',
    parameters JSONB DEFAULT '{}',
    file_path VARCHAR(1000),
    file_size_bytes BIGINT,
    provenance_hash VARCHAR(64),
    generated_by VARCHAR(200),
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 8. Regulatory Update Records (hypertable on published_at)
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_regulatory_updates (
    update_id UUID DEFAULT gen_random_uuid(),
    update_type VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    source_url VARCHAR(2000),
    source_document VARCHAR(500),
    affected_countries JSONB DEFAULT '[]',
    affected_commodities JSONB DEFAULT '[]',
    affected_articles JSONB DEFAULT '[]',
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'important', 'informational')),
    effective_date DATE,
    grace_period_end DATE,
    impact_assessment JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed', 'applied', 'archived')),
    reviewed_by VARCHAR(200),
    reviewed_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (update_id, published_at)
);

SELECT create_hypertable('eudr_country_risk_evaluator.gl_eudr_cre_regulatory_updates', 'published_at');

-- 9. Individual Risk Factor Scores
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_risk_factors (
    factor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    factor_name VARCHAR(100) NOT NULL,
    factor_value NUMERIC(5,2) NOT NULL,
    normalized_value NUMERIC(5,2) NOT NULL,
    data_source VARCHAR(500),
    source_date DATE,
    confidence VARCHAR(10) NOT NULL DEFAULT 'medium',
    provenance_hash VARCHAR(64),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (country_code, factor_name)
);

-- 10. Risk Score Change History
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_risk_history (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    previous_score NUMERIC(5,2),
    new_score NUMERIC(5,2) NOT NULL,
    previous_risk_level VARCHAR(20),
    new_risk_level VARCHAR(20) NOT NULL,
    change_reason VARCHAR(500) NOT NULL,
    change_source VARCHAR(200),
    changed_by VARCHAR(200),
    provenance_hash VARCHAR(64),
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 11. Certification Scheme Effectiveness
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_certifications (
    certification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scheme_name VARCHAR(200) NOT NULL,
    country_code CHAR(2) NOT NULL,
    commodity VARCHAR(20) NOT NULL,
    effectiveness_score NUMERIC(5,2) NOT NULL CHECK (effectiveness_score >= 0 AND effectiveness_score <= 100),
    coverage_pct NUMERIC(5,2),
    audit_rigor_score NUMERIC(5,2),
    historical_performance NUMERIC(5,2),
    recognized_by_ec BOOLEAN DEFAULT FALSE,
    data_source VARCHAR(500),
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (scheme_name, country_code, commodity)
);

-- 12. Immutable Audit Log
CREATE TABLE eudr_country_risk_evaluator.gl_eudr_cre_audit_log (
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
-- Indexes
-- =========================================================================

-- Country risks
CREATE INDEX idx_cre_country_risks_country ON eudr_country_risk_evaluator.gl_eudr_cre_country_risks(country_code);
CREATE INDEX idx_cre_country_risks_level ON eudr_country_risk_evaluator.gl_eudr_cre_country_risks(risk_level);
CREATE INDEX idx_cre_country_risks_score ON eudr_country_risk_evaluator.gl_eudr_cre_country_risks(composite_score);

-- Commodity risks
CREATE INDEX idx_cre_commodity_risks_country ON eudr_country_risk_evaluator.gl_eudr_cre_commodity_risks(country_code);
CREATE INDEX idx_cre_commodity_risks_commodity ON eudr_country_risk_evaluator.gl_eudr_cre_commodity_risks(commodity);
CREATE INDEX idx_cre_commodity_risks_level ON eudr_country_risk_evaluator.gl_eudr_cre_commodity_risks(risk_level);

-- Hotspots
CREATE INDEX idx_cre_hotspots_country ON eudr_country_risk_evaluator.gl_eudr_cre_hotspots(country_code);
CREATE INDEX idx_cre_hotspots_level ON eudr_country_risk_evaluator.gl_eudr_cre_hotspots(risk_level);
CREATE INDEX idx_cre_hotspots_coords ON eudr_country_risk_evaluator.gl_eudr_cre_hotspots(center_lat, center_lon);

-- Governance
CREATE INDEX idx_cre_governance_country ON eudr_country_risk_evaluator.gl_eudr_cre_governance_indices(country_code);

-- Due diligence
CREATE INDEX idx_cre_dd_country_commodity ON eudr_country_risk_evaluator.gl_eudr_cre_due_diligence_levels(country_code, commodity);
CREATE INDEX idx_cre_dd_level ON eudr_country_risk_evaluator.gl_eudr_cre_due_diligence_levels(dd_level);

-- Trade flows
CREATE INDEX idx_cre_trade_origin ON eudr_country_risk_evaluator.gl_eudr_cre_trade_flows(origin_country);
CREATE INDEX idx_cre_trade_destination ON eudr_country_risk_evaluator.gl_eudr_cre_trade_flows(destination_country);
CREATE INDEX idx_cre_trade_commodity ON eudr_country_risk_evaluator.gl_eudr_cre_trade_flows(commodity);
CREATE INDEX idx_cre_trade_period ON eudr_country_risk_evaluator.gl_eudr_cre_trade_flows(period);
CREATE INDEX idx_cre_trade_reexport ON eudr_country_risk_evaluator.gl_eudr_cre_trade_flows(re_export_risk) WHERE re_export_risk = TRUE;

-- Reports
CREATE INDEX idx_cre_reports_type ON eudr_country_risk_evaluator.gl_eudr_cre_risk_reports(report_type);

-- Regulatory updates
CREATE INDEX idx_cre_regulatory_severity ON eudr_country_risk_evaluator.gl_eudr_cre_regulatory_updates(severity);
CREATE INDEX idx_cre_regulatory_status ON eudr_country_risk_evaluator.gl_eudr_cre_regulatory_updates(status);

-- Risk factors
CREATE INDEX idx_cre_risk_factors_country ON eudr_country_risk_evaluator.gl_eudr_cre_risk_factors(country_code);

-- Risk history
CREATE INDEX idx_cre_risk_history_country ON eudr_country_risk_evaluator.gl_eudr_cre_risk_history(country_code);
CREATE INDEX idx_cre_risk_history_changed ON eudr_country_risk_evaluator.gl_eudr_cre_risk_history(changed_at);

-- Certifications
CREATE INDEX idx_cre_certifications_scheme ON eudr_country_risk_evaluator.gl_eudr_cre_certifications(scheme_name);
CREATE INDEX idx_cre_certifications_country ON eudr_country_risk_evaluator.gl_eudr_cre_certifications(country_code);

-- Audit log
CREATE INDEX idx_cre_audit_entity ON eudr_country_risk_evaluator.gl_eudr_cre_audit_log(entity_type, entity_id);
CREATE INDEX idx_cre_audit_actor ON eudr_country_risk_evaluator.gl_eudr_cre_audit_log(actor);
CREATE INDEX idx_cre_audit_created ON eudr_country_risk_evaluator.gl_eudr_cre_audit_log(created_at);
```

### 7.5 API Endpoints (~37)

| Method | Path | Description |
|--------|------|-------------|
| **Country Risk Assessment** | | |
| POST | `/v1/eudr-cre/countries/assess` | Assess risk for a single country (full composite scoring) |
| GET | `/v1/eudr-cre/countries/{country_code}` | Get latest risk assessment for a country |
| GET | `/v1/eudr-cre/countries` | List all country risk assessments (with filters: risk_level, region) |
| GET | `/v1/eudr-cre/countries/{code1}/compare/{code2}` | Compare two countries side-by-side |
| GET | `/v1/eudr-cre/countries/{country_code}/trends` | Get risk score trend over time |
| **Commodity Risk Analysis** | | |
| POST | `/v1/eudr-cre/commodities/analyze` | Analyze commodity risk for a country |
| GET | `/v1/eudr-cre/commodities/{country_code}/{commodity}` | Get commodity risk profile |
| GET | `/v1/eudr-cre/commodities/matrix` | Get commodity-country risk matrix |
| GET | `/v1/eudr-cre/commodities/correlations` | Get cross-commodity risk correlations |
| **Deforestation Hotspots** | | |
| POST | `/v1/eudr-cre/hotspots/detect` | Trigger hotspot detection for a country |
| GET | `/v1/eudr-cre/hotspots/{hotspot_id}` | Get hotspot details |
| GET | `/v1/eudr-cre/hotspots` | List hotspots (with filters: country, risk_level, commodity) |
| GET | `/v1/eudr-cre/hotspots/alerts` | Get recent deforestation alerts |
| GET | `/v1/eudr-cre/hotspots/clustering` | Get hotspot cluster analysis results |
| **Governance Indices** | | |
| POST | `/v1/eudr-cre/governance/evaluate` | Evaluate governance for a country |
| GET | `/v1/eudr-cre/governance/{country_code}` | Get governance index for a country |
| GET | `/v1/eudr-cre/governance` | List governance indices for all countries |
| GET | `/v1/eudr-cre/governance/compare` | Compare governance across countries |
| **Due Diligence Classification** | | |
| POST | `/v1/eudr-cre/due-diligence/classify` | Classify DD level for country-commodity-region |
| GET | `/v1/eudr-cre/due-diligence/{country_code}/{commodity}` | Get DD classification |
| GET | `/v1/eudr-cre/due-diligence` | List all DD classifications |
| GET | `/v1/eudr-cre/due-diligence/cost-estimate` | Get DD cost estimate for a scenario |
| GET | `/v1/eudr-cre/due-diligence/audit-frequency` | Get audit frequency recommendation |
| **Trade Flow Analysis** | | |
| POST | `/v1/eudr-cre/trade-flows/analyze` | Analyze trade flows for a commodity route |
| GET | `/v1/eudr-cre/trade-flows/{flow_id}` | Get trade flow details |
| GET | `/v1/eudr-cre/trade-flows` | List trade flows (with filters) |
| GET | `/v1/eudr-cre/trade-flows/routes` | Get risk-scored trade routes |
| GET | `/v1/eudr-cre/trade-flows/re-export-risk` | List re-export risk detections |
| **Risk Reports** | | |
| POST | `/v1/eudr-cre/reports/generate` | Generate a risk report |
| GET | `/v1/eudr-cre/reports/{report_id}` | Get report metadata |
| GET | `/v1/eudr-cre/reports` | List generated reports |
| GET | `/v1/eudr-cre/reports/{report_id}/download` | Download report file |
| GET | `/v1/eudr-cre/reports/executive-summary` | Get executive summary data |
| **Regulatory Updates** | | |
| POST | `/v1/eudr-cre/regulatory/track` | Trigger regulatory update check |
| GET | `/v1/eudr-cre/regulatory/{update_id}` | Get regulatory update details |
| GET | `/v1/eudr-cre/regulatory` | List regulatory updates |
| GET | `/v1/eudr-cre/regulatory/impact-assessment` | Get impact assessment for a regulatory change |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_cre_assessments_total` | Counter | Country risk assessments performed |
| 2 | `gl_eudr_cre_commodity_analyses_total` | Counter | Commodity risk analyses completed |
| 3 | `gl_eudr_cre_hotspots_detected_total` | Counter | Deforestation hotspots detected |
| 4 | `gl_eudr_cre_classifications_total` | Counter | Due diligence classifications performed |
| 5 | `gl_eudr_cre_reports_generated_total` | Counter | Risk reports generated by type |
| 6 | `gl_eudr_cre_trade_analyses_total` | Counter | Trade flow analyses completed |
| 7 | `gl_eudr_cre_regulatory_updates_total` | Counter | Regulatory updates processed |
| 8 | `gl_eudr_cre_api_errors_total` | Counter | API errors by endpoint and status code |
| 9 | `gl_eudr_cre_assessment_duration_seconds` | Histogram | Country risk assessment latency |
| 10 | `gl_eudr_cre_commodity_analysis_duration_seconds` | Histogram | Commodity analysis latency |
| 11 | `gl_eudr_cre_hotspot_detection_duration_seconds` | Histogram | Hotspot detection latency |
| 12 | `gl_eudr_cre_classification_duration_seconds` | Histogram | DD classification latency |
| 13 | `gl_eudr_cre_report_generation_duration_seconds` | Histogram | Report generation latency |
| 14 | `gl_eudr_cre_active_hotspots` | Gauge | Currently active deforestation hotspots |
| 15 | `gl_eudr_cre_countries_assessed` | Gauge | Total countries with active risk assessments |
| 16 | `gl_eudr_cre_high_risk_countries` | Gauge | Number of countries classified as high risk |
| 17 | `gl_eudr_cre_pending_reclassifications` | Gauge | Countries pending reclassification review |
| 18 | `gl_eudr_cre_stale_assessments` | Gauge | Assessments older than configured freshness threshold |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for risk history |
| Spatial | PostGIS + Shapely | Hotspot spatial analysis, protected area intersection |
| Cache | Redis | Risk score caching, assessment result caching |
| Object Storage | S3 | Generated reports, trade flow data snapshots |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Arithmetic | Python Decimal | Precision for risk score calculations (no floating-point drift) |
| PDF Generation | WeasyPrint | HTML-to-PDF for report generation |
| Templates | Jinja2 | Multi-format report templates (HTML/PDF) |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based risk data access control |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-cre:countries:read` | View country risk assessments | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cre:countries:assess` | Trigger country risk assessments | Analyst, Compliance Officer, Admin |
| `eudr-cre:commodities:read` | View commodity risk profiles | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cre:commodities:analyze` | Trigger commodity risk analysis | Analyst, Compliance Officer, Admin |
| `eudr-cre:hotspots:read` | View deforestation hotspots | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cre:hotspots:detect` | Trigger hotspot detection | Analyst, Compliance Officer, Admin |
| `eudr-cre:governance:read` | View governance indices | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cre:governance:evaluate` | Trigger governance evaluation | Analyst, Compliance Officer, Admin |
| `eudr-cre:due-diligence:read` | View DD classifications | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cre:due-diligence:classify` | Trigger DD classification | Compliance Officer, Admin |
| `eudr-cre:trade-flows:read` | View trade flow data | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cre:trade-flows:analyze` | Trigger trade flow analysis | Analyst, Compliance Officer, Admin |
| `eudr-cre:reports:read` | View generated reports | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cre:reports:generate` | Generate risk reports | Analyst, Compliance Officer, Admin |
| `eudr-cre:reports:download` | Download report files | Analyst, Compliance Officer, Admin |
| `eudr-cre:regulatory:read` | View regulatory updates | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cre:regulatory:manage` | Manage regulatory update tracking | Compliance Officer, Admin |
| `eudr-cre:config:manage` | Manage custom weight profiles and thresholds | Admin |
| `eudr-cre:audit:read` | View audit trail | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent/Source | Integration | Data Flow |
|-------------|-------------|-----------|
| Existing `eudr_country_risk.py` | Direct import | 25+ country risk profiles, forest data, commodity risks -> seed data for F1, F2 |
| Existing `country_risk_scores.py` | Direct import | 85+ country risk scores with factor breakdowns -> seed data for F1, F4 |
| AGENT-DATA-007 Deforestation Satellite | GFW/FIRMS data | Tree cover loss, fire alerts -> hotspot detection (F3) |
| AGENT-DATA-006 GIS/Mapping Connector | Spatial operations | Protected area intersections, distance calculations -> F3 |
| AGENT-EUDR-005 Land Use Change Detector | Land use data | Land use classification changes -> F3 driver attribution |
| AGENT-FOUND-005 Citations & Evidence | Source tracking | Citation generation and evidence linking -> all features |
| AGENT-FOUND-008 Reproducibility | Determinism verification | Bit-perfect verification of risk calculations -> F1, F2, F5 |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-001 Supply Chain Mapping | Risk data feed | Country + commodity risk scores -> graph node risk attributes |
| AGENT-EUDR-008 Multi-Tier Supplier Tracker | Supplier country risk | Country risk scores -> supplier risk baseline |
| GL-EUDR-APP v1.0 Platform | API integration | Risk dashboards, DD classification display, report downloads |
| GL-EUDR-APP DDS Reporting Engine | Risk assessment section | Country risk data formatted for DDS Article 4(2) submission |
| External Auditors | Read-only API + reports | Risk exports for third-party verification |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Country Risk Assessment (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Risk Assessment" module -> "Country Risk" tab
3. Selects country (e.g., Indonesia) from dropdown or world map
4. System displays comprehensive risk profile:
   - Composite score: 74 (HIGH)
   - Factor breakdown: deforestation 82, governance 32, enforcement 28, corruption 62, law 28, trend 65
   - Commodity risk matrix: palm_oil HIGH, rubber HIGH, wood HIGH, cocoa STANDARD, coffee STANDARD
   - Deforestation hotspots: Sumatra (85), Kalimantan (88), Papua (82)
   - Trend: DETERIORATING over 5 years
   - Governance details: WGI scores, CPI rank, FLEGT status
5. Officer clicks "Generate Report" -> selects PDF format
6. System generates country risk profile PDF in < 5 seconds
7. Officer downloads report for DDS documentation
```

#### Flow 2: Due Diligence Classification (Risk Analyst)

```
1. Analyst receives import notification: palm oil from West Kalimantan, Indonesia
2. Opens "Due Diligence" module -> enters country, commodity, region
3. System classifies:
   - Country risk: 74 (HIGH)
   - Commodity risk: 82 (HIGH, palm oil in Indonesia)
   - Region: West Kalimantan identified as deforestation hotspot (score 88)
   - Region override: ACTIVE (hotspot -> forced enhanced DD)
   - Certification credit: RSPO certified, effectiveness 65 -> 15 point credit
   - Effective risk score: 67 (still HIGH after credit)
   - DD Level: ENHANCED
   - Required: satellite verification, independent audit, supplier site visit
   - Estimated cost: EUR 8,500
   - Estimated time: 24 hours
4. System generates DD requirement checklist
5. Analyst triggers satellite verification workflow (-> EUDR-003)
6. System records classification in audit log
```

#### Flow 3: Trade Flow Investigation (Risk Analyst)

```
1. Analyst notices unusual pattern: increased palm oil imports from Singapore
2. Opens "Trade Flows" module -> selects palm_oil, origin: Singapore
3. System displays:
   - Singapore palm oil production: 0 MT (zero domestic production)
   - Singapore palm oil re-exports: 2.5M MT/year
   - Primary origins of Singapore palm oil: Indonesia (65%), Malaysia (30%), Papua New Guinea (5%)
   - Re-export risk: CRITICAL (confidence: 0.95)
   - Route risk score: 85 (inherits Indonesia HIGH risk)
4. System flags all Singapore-origin palm oil imports for enhanced DD
5. Analyst clicks "View Trade Route Map" -> sees bilateral flows
6. System recommends: "Apply origin-country DD level (Indonesia: ENHANCED) for Singapore re-exports"
```

### 8.2 Key Screen Descriptions

**Country Risk Dashboard:**
- World map heatmap: countries colored by risk level (green/yellow/red)
- Left sidebar: country search and filter panel
- Main panel: selected country risk profile with radar chart (6 factors)
- Right panel: commodity risk matrix for selected country
- Bottom panel: risk trend chart (5-year history)

**Hotspot Map View:**
- Satellite imagery base layer with hotspot overlay
- Hotspot clusters colored by severity
- Protected area boundaries shown as dashed outlines
- Indigenous territory boundaries shown as hatched areas
- Alert timeline slider: filter hotspots by detection date range

**Due Diligence Classification Panel:**
- Input: country, commodity, region (dropdown selectors)
- Output: classified DD level with visual indicator (green/yellow/red badge)
- Factor breakdown: which factors drove the classification
- Checklist: required documentation per DD level
- Cost and time estimate

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 8 P0 features (Features 1-8) implemented and tested
  - [ ] Feature 1: Country Risk Scoring Engine -- 200+ countries, 6-factor composite, EC alignment
  - [ ] Feature 2: Commodity-Specific Risk Analyzer -- all 7 commodities per country
  - [ ] Feature 3: Deforestation Hotspot Detector -- sub-national detection, GFW integration
  - [ ] Feature 4: Governance Index Engine -- WGI, CPI, forest governance integration
  - [ ] Feature 5: Due Diligence Level Classifier -- 3-tier classification, dynamic reclassification
  - [ ] Feature 6: Trade Flow Analyzer -- bilateral mapping, re-export detection
  - [ ] Feature 7: Risk Report Generator -- PDF/JSON/HTML, multi-language
  - [ ] Feature 8: Regulatory Update Tracker -- EC monitoring, impact assessment
- [ ] >= 85% test coverage achieved (500+ tests)
- [ ] Security audit passed (JWT + RBAC integrated, 19 permissions)
- [ ] Performance targets met (< 500ms per assessment, < 5s per report)
- [ ] Risk scores validated against EC published benchmarks (100% alignment)
- [ ] Risk calculations verified deterministic (bit-perfect reproducibility)
- [ ] All 7 commodity risk profiles tested for 20+ countries
- [ ] Hotspot detection validated against Global Forest Watch 2024 data
- [ ] Due diligence classifications validated against 100+ regulatory test cases
- [ ] API documentation complete (OpenAPI spec, 37 endpoints)
- [ ] Database migration V104 tested and validated
- [ ] Integration with EUDR-001, EUDR-003, EUDR-005, EUDR-008 verified
- [ ] 5 beta customers successfully using country risk assessments
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 100+ country risk assessments consumed by customers
- 50+ risk reports generated
- 200+ due diligence classifications performed
- Average assessment latency < 500ms (p99)
- EC benchmark alignment maintained at 100%
- < 5 support tickets per customer

**60 Days:**
- 300+ unique country-commodity combinations assessed
- 200+ risk reports generated across all formats
- Trade flow analysis active for 10+ bilateral routes
- Hotspot detection covering 20+ high-risk countries
- 3+ regulatory updates processed and applied
- NPS > 45 from compliance officer persona

**90 Days:**
- 500+ active country risk profiles maintained
- 1,000+ due diligence classifications in production
- 50+ re-export risk detections flagged
- Zero EUDR penalties attributable to incorrect risk assessment for active customers
- Full integration with GL-EUDR-APP DDS workflow operational
- NPS > 55

---

## 10. Timeline and Milestones

### Phase 1: Core Risk Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Country Risk Scoring Engine (Feature 1): 6-factor composite, 200+ countries, EC alignment | Senior Backend Engineer |
| 2-3 | Governance Index Engine (Feature 4): WGI, CPI, forest governance integration | Backend Engineer |
| 3-4 | Commodity-Specific Risk Analyzer (Feature 2): 7 commodities, per-country profiles | Data Engineer |
| 4-5 | Due Diligence Level Classifier (Feature 5): 3-tier classification, dynamic rules | Backend Engineer |
| 5-6 | Deforestation Hotspot Detector (Feature 3): GFW integration, spatial clustering | GIS Specialist |

**Milestone: Core risk assessment engine operational with all 5 core features (Week 6)**

### Phase 2: Analysis, API, and Reporting (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Trade Flow Analyzer (Feature 6): bilateral mapping, re-export detection | Backend Engineer |
| 8-9 | REST API Layer: 37 endpoints, authentication, rate limiting | Backend Engineer |
| 9-10 | Risk Report Generator (Feature 7): PDF/JSON/HTML, multi-language templates | Backend + Template Engineer |

**Milestone: Full API operational with trade flow analysis and report generation (Week 10)**

### Phase 3: Monitoring, Integration, and Testing (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11 | Regulatory Update Tracker (Feature 8): EC monitoring, impact assessment | Backend Engineer |
| 11-12 | Integration with EUDR-001, EUDR-003, EUDR-005, EUDR-008 | Backend Engineer |
| 12-13 | RBAC integration, Prometheus metrics, Grafana dashboard | DevOps + Backend |
| 13-14 | Complete test suite: 500+ tests, golden tests, integration tests | Test Engineer |

**Milestone: All 8 P0 features implemented with full integration (Week 14)**

### Phase 4: Validation and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | EC benchmark validation, regulatory test case verification | Compliance + Test Engineer |
| 16-17 | Performance testing, security audit, load testing | DevOps + Security |
| 17 | Database migration V104 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers) | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 8 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Predictive risk forecasting (Feature 9)
- Supplier-country risk correlation (Feature 10)
- Interactive risk map dashboard (Feature 11)
- Expanded country coverage to 250+ countries
- Additional certification scheme effectiveness data

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| `greenlang/data/eudr_country_risk.py` | BUILT (25+ countries, 1900+ lines) | Low | Stable, production-ready; consumed as seed data |
| `greenlang/agents/eudr/multi_tier_supplier/reference_data/country_risk_scores.py` | BUILT (85+ countries, 1470+ lines) | Low | Stable, production-ready; consumed as seed data |
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Integration points defined |
| AGENT-EUDR-003 Satellite Monitoring | BUILT | Low | GFW data provider |
| AGENT-EUDR-005 Land Use Change Detector | BUILT | Low | Land use data provider |
| AGENT-EUDR-008 Multi-Tier Supplier Tracker | BUILT | Low | Supplier risk consumer |
| AGENT-DATA-006 GIS/Mapping Connector | BUILT (100%) | Low | Spatial operations |
| AGENT-DATA-007 Deforestation Satellite Connector | BUILT (100%) | Low | GFW/FIRMS data |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Integration target |
| PostgreSQL + TimescaleDB + PostGIS | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EC EUDR country benchmarking list | Published (evolving) | Medium | Database-driven; hot-reloadable; adapter pattern |
| FAO Global Forest Resources Assessment | Published (2025 edition) | Low | Annual update cycle; offline data cache |
| Global Forest Watch API | Available | Low | Multi-endpoint fallback; cached data |
| World Bank WGI | Published (2024) | Low | Annual update; offline data cache |
| Transparency International CPI | Published (2024) | Low | Annual update; offline data cache |
| UN Comtrade trade data | Available | Medium | Quarterly updates; cached snapshots |
| EU EUDR implementing regulations | Evolving | Medium | Configuration-driven compliance rules; regulatory monitor |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EC publishes country benchmarking list that differs from agent's scoring methodology | High | High | Agent treats EC list as authoritative; any divergence flagged; EC classification takes precedence over agent scoring |
| R2 | External data source (GFW, WGI, CPI) becomes unavailable or changes format | Medium | Medium | Offline data cache; multi-source fallback; adapter pattern isolates data ingestion |
| R3 | Country reclassification causes cascade of DD level changes, overwhelming operators | Medium | High | Impact assessment with transition timeline; grace period tracking; prioritized action lists |
| R4 | Commodity laundering patterns evolve to evade re-export detection | Medium | Medium | Continuous refinement of statistical anomaly detection; operator feedback loop |
| R5 | Sub-national hotspot data gaps for less-monitored countries | High | Medium | Use country-level fallback when sub-national data unavailable; flag with reduced confidence |
| R6 | Governance indicator data lag (WGI/CPI published annually) | Medium | Low | Use latest available data with explicit staleness flag; supplement with more frequent FLEGT/enforcement data |
| R7 | Integration complexity with 4 upstream EUDR agents | Medium | Medium | Well-defined interfaces; mock adapters for testing; circuit breaker pattern |
| R8 | Operators customize risk weights to game the system (lower their risk) | Low | Medium | Enforce minimum weight bounds (5% per factor); audit log all weight changes; flag deviations from default |
| R9 | Report generation performance degrades for multi-country batch reports | Low | Medium | Async report generation with webhook notification; S3 storage for large reports |
| R10 | Regulatory FAQ and guidance document volume overwhelms indexing | Low | Low | Focus on material changes; severity-based prioritization; automated relevance scoring |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Country Risk Scoring Tests | 80+ | Composite calculation, factor weighting, normalization, EC alignment, edge cases |
| Commodity Risk Analyzer Tests | 70+ | All 7 commodities, production data, correlation, seasonal variation, certification |
| Hotspot Detection Tests | 60+ | Spatial clustering, GFW data processing, protected area overlap, trend analysis |
| Governance Index Tests | 50+ | WGI integration, CPI scoring, forest governance, legal framework, enforcement |
| DD Classification Tests | 60+ | 3-tier classification, dynamic reclassification, overrides, certification credits |
| Trade Flow Analysis Tests | 50+ | Bilateral mapping, re-export detection, route scoring, HS code mapping |
| Report Generation Tests | 40+ | All formats (PDF/JSON/HTML), multi-language, template rendering, provenance |
| Regulatory Tracker Tests | 30+ | EC monitoring, reclassification tracking, impact assessment, notifications |
| API Tests | 50+ | All 37 endpoints, auth, error handling, pagination, rate limiting |
| Integration Tests | 30+ | Cross-agent integration with EUDR-001/003/005/008, existing risk data import |
| Performance Tests | 20+ | Single assessment, batch assessment, report generation, concurrent queries |
| Golden Tests | 60+ | Known country risk scenarios (20 countries x 3 scenarios each) |
| **Total** | **600+** | |

### 13.2 Golden Test Scenarios

Each of the following 20 representative countries will have 3 golden test scenarios:

**High-Risk Countries (8):** Brazil (BR), Indonesia (ID), Democratic Republic of the Congo (CD), Cote d'Ivoire (CI), Malaysia (MY), Paraguay (PY), Bolivia (BO), Nigeria (NG)

**Standard-Risk Countries (6):** Colombia (CO), Ghana (GH), Peru (PE), Vietnam (VN), Argentina (AR), Ethiopia (ET)

**Low-Risk Countries (6):** Germany (DE), Sweden (SE), Finland (FI), Canada (CA), United States (US), New Zealand (NZ)

**Scenarios per country:**
1. **Full assessment** -- Complete composite scoring with all 6 factors, commodity risk matrix, governance indices -> expect exact score match with golden baseline
2. **DD classification** -- Country + commodity + region combination -> expect correct DD level, cost estimate, and documentation requirements
3. **Trend analysis** -- 5-year risk evolution -> expect correct trend direction (improving/stable/deteriorating)

Total: 20 countries x 3 scenarios = 60 golden test scenarios

### 13.3 Determinism Tests

Every risk calculation engine will include determinism tests that:
1. Run the same calculation 100 times with identical inputs
2. Verify bit-perfect identical outputs (SHA-256 hash match)
3. Test across Python versions (3.11, 3.12) to ensure no platform-dependent behavior
4. Verify Decimal arithmetic produces identical results to reference calculations

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **EC** | European Commission -- the institution responsible for EUDR implementation and country benchmarking |
| **WGI** | World Bank Worldwide Governance Indicators -- 6 dimensions of governance quality |
| **CPI** | Corruption Perceptions Index -- Transparency International's annual corruption ranking |
| **GFW** | Global Forest Watch -- WRI platform providing near-real-time deforestation monitoring data |
| **FAO FRA** | Food and Agriculture Organization Global Forest Resources Assessment |
| **FLEGT** | Forest Law Enforcement, Governance and Trade -- EU bilateral agreements with timber-producing countries |
| **FIRMS** | Fire Information for Resource Management System -- NASA satellite fire detection |
| **VIIRS** | Visible Infrared Imaging Radiometer Suite -- satellite instrument for fire and land surface monitoring |
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise -- clustering algorithm for hotspot detection |
| **DD** | Due Diligence -- the process of risk assessment and mitigation required by EUDR |
| **HS Code** | Harmonized System -- international product classification code (6-digit) |
| **CN Code** | Combined Nomenclature -- EU product classification code (8-digit extension of HS) |
| **FPIC** | Free, Prior and Informed Consent -- principle for indigenous peoples' rights |
| **RSPO** | Roundtable on Sustainable Palm Oil -- palm oil certification scheme |
| **FSC** | Forest Stewardship Council -- forest product certification scheme |
| **RTRS** | Round Table on Responsible Soy -- soya certification scheme |

### Appendix B: Existing Country Risk Data Summary

The agent consumes and extends two existing country risk databases in the GreenLang platform:

**1. `greenlang/data/eudr_country_risk.py` (1900+ lines, 25+ countries):**
- Data classes: `ForestData`, `CountryRisk`, `RegionRisk`
- Enums: `RiskLevel`, `DueDiligenceLevel`, `ForestType`
- `COUNTRY_RISK_DATABASE`: 25+ countries with full forest data, commodity risks, governance/enforcement scores, regions of concern
- `REGION_RISK_DATABASE`: Sub-national risk data for Brazil (3 regions), Indonesia (3 regions), Cote d'Ivoire (1 region)
- Functions: `get_country_risk()`, `get_risk_level()`, `get_commodity_risk()`, `get_due_diligence_level()`, `assess_country_risk()`, `is_deforestation_hotspot()`, etc.

**2. `greenlang/agents/eudr/multi_tier_supplier/reference_data/country_risk_scores.py` (1470+ lines, 85+ countries):**
- `COUNTRY_RISK_SCORES`: 85+ countries with 4-factor risk breakdown (deforestation_rate, governance_index, enforcement_score, corruption_index)
- `REGIONAL_RISK_AGGREGATES`: Regional averages for 15 regions
- Composite score formula: `deforestation_rate * 0.35 + (100 - governance_index) * 0.25 + (100 - enforcement_score) * 0.25 + corruption_index * 0.15`
- Functions: `get_country_risk()`, `get_risk_factors()`, `get_high_risk_countries()`, `get_countries_by_commodity()`, `get_risk_by_region()`

**Migration strategy:**
- Agent-EUDR-016 imports and unifies data from both sources into a single 200+ country database
- Existing 4-factor composite formula is extended to 6 factors (adding forest_law_compliance and historical_trend)
- Existing commodity_risks dictionaries are enriched with production volume, correlation, and certification data
- All existing region data is preserved and extended with spatial clustering from GFW data
- Backward compatibility maintained: existing lookup functions continue to work

### Appendix C: EC Benchmarking Criteria (Article 29 Reference)

Per Article 29(2), the EC assesses countries based on:
- (a) Rate of deforestation and forest degradation
- (b) Rate of expansion of agricultural land for relevant commodities
- (c) Production trends of relevant commodities and of relevant products

Per Article 29(3), the EC also considers:
- (a) The country's contribution to deforestation and forest degradation
- (b) National legislation, ratification of relevant international conventions, enforcement
- (c) Respect for rights of indigenous peoples and local communities
- (d) Cooperation with the EU
- (e) Anti-corruption measures
- (f) National REDD+ contribution

The agent maps all criteria to specific risk factors with assigned weights as documented in Section 5.2.

### Appendix D: EUDR Commodity Classification (Annex I Reference)

| Commodity | EUDR Annex I Coverage | Key HS Codes | Derived Products |
|-----------|----------------------|-------------|-----------------|
| Cattle | Live bovine animals, beef, leather | 0102, 0201-0202, 4101-4115 | Leather goods, canned beef, gelatin |
| Cocoa | Cocoa beans, paste, butter, powder | 1801-1806 | Chocolate, cocoa products |
| Coffee | Green, roasted, extracts | 0901 | Instant coffee, coffee extracts |
| Oil Palm | Palm oil, palm kernel oil | 1511, 1513 | Oleochemicals, biodiesel, food products |
| Rubber | Natural rubber, latex | 4001, 4005-4017 | Tires, gloves, footwear |
| Soya | Soybeans, soybean oil, meal | 1201, 1507, 2304 | Animal feed, tofu, lecithin |
| Wood | Timber, wood products, paper | 4401-4421, 4700-4813, 9401-9403 | Furniture, paper, packaging, charcoal |

### Appendix E: Risk Factor Data Sources

| Risk Factor | Primary Source | Secondary Source | Update Frequency |
|-------------|---------------|-----------------|-----------------|
| Deforestation Rate | FAO FRA 2025 | Global Forest Watch 2024 | Annual (FAO), Monthly (GFW) |
| Governance Index | World Bank WGI 2024 | Freedom House | Annual |
| Enforcement Score | Custom (WGI + FLEGT + enforcement data) | EIA reports | Annual |
| Corruption Index | Transparency International CPI 2024 | World Bank CPIA | Annual |
| Forest Law Compliance | FAO LEX, ILO Convention 169 | National legislation review | Biennial |
| Historical Trend | GFW Hansen tree cover loss (2019-2024) | FAO FRA trend data | Annual |

### Appendix F: Integration API Contracts

**Consumed from EUDR-001 (Supply Chain Mapping Master):**
```python
# Country risk data feed for graph node risk attributes
def get_country_risk_for_node(country_code: str, commodity: str) -> Dict:
    """Returns: {risk_level, risk_score, dd_level, hotspot_flag, provenance_hash}"""
```

**Provided to EUDR-001:**
```python
# Risk assessment result
{
    "country_code": "ID",
    "commodity": "palm_oil",
    "risk_level": "high",
    "risk_score": 82,
    "dd_level": "enhanced",
    "hotspot_regions": ["Sumatra", "Kalimantan", "Papua"],
    "satellite_verification_required": true,
    "provenance_hash": "sha256:a1b2c3..."
}
```

**Consumed from EUDR-008 (Multi-Tier Supplier Tracker):**
```python
# Supplier country for risk baseline
def get_supplier_country_risk(supplier_id: str) -> Dict:
    """Returns country risk baseline for supplier's registered country"""
```

### Appendix G: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EUDR)
2. European Commission EUDR Guidance Document -- Risk Assessment and Due Diligence
3. European Commission Country Benchmarking Methodology (Article 29 Implementation)
4. FAO Global Forest Resources Assessment 2025
5. Global Forest Watch Technical Documentation -- Tree Cover Loss, GLAD Alerts, RADD Alerts
6. World Bank Worldwide Governance Indicators Methodology (Kaufmann et al.)
7. Transparency International Corruption Perceptions Index 2024 Methodology
8. ITTO/FAO Framework for Assessing and Monitoring Forest Governance
9. Hansen, M.C. et al. "High-Resolution Global Maps of 21st-Century Forest Cover Change" -- Science 342 (2013)
10. UN Comtrade International Trade Statistics Database
11. Eurostat -- EU Trade in Goods Statistics
12. EUDR Annex I -- Products Covered by the Regulation
13. ISO 3166-1 -- Country Codes Standard
14. ILO Convention 169 -- Indigenous and Tribal Peoples Convention

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
| 1.0.0 | 2026-03-09 | GL-ProductManager | Finalized: all 8 P0 features confirmed (80 sub-requirements), regulatory coverage verified (Articles 2/10/11/13/29/31), integration with existing eudr_country_risk.py and country_risk_scores.py documented, 12-table DB schema V104 designed, 37 API endpoints specified, 18 Prometheus metrics defined, 600+ test target set, approval granted |
