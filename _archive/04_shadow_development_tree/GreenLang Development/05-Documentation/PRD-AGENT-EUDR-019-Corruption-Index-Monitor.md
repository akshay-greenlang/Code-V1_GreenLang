# PRD: AGENT-EUDR-019 -- Corruption Index Monitor

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-019 |
| **Agent ID** | GL-EUDR-CIM-019 |
| **Component** | Corruption Index Monitor Agent |
| **Category** | EUDR Regulatory Agent -- Governance Integrity Intelligence |
| **Priority** | P0 -- Critical (EUDR Country Benchmarking Dependency) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-09 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-09 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) requires operators and traders to conduct risk assessments that account for country-level governance factors. Article 29 mandates the European Commission to classify countries into Low, Standard, and High risk categories based on criteria that explicitly include governance quality, corruption levels, rule of law, and institutional enforcement capacity. These corruption and governance indicators are not peripheral considerations -- they are primary determinants of whether a country's forest governance is effective, whether illegal deforestation is likely to occur unchecked, and whether an operator's due diligence obligations are heightened to the enhanced level.

The relationship between corruption and deforestation is well-documented and causal. Countries with high corruption perception indices consistently exhibit higher rates of illegal logging, weaker enforcement of forestry regulations, and greater prevalence of fraudulent land use documentation. Transparency International's Corruption Perceptions Index (CPI) and the World Bank's Worldwide Governance Indicators (WGI) are the two most authoritative global datasets on corruption and governance quality, and both feed directly into the European Commission's country benchmarking methodology under EUDR Article 29.

Today, EU operators face the following governance intelligence gaps:

- **No systematic corruption monitoring**: Operators track CPI scores manually and infrequently; they lack automated monitoring of corruption index changes across their sourcing countries.
- **No multi-dimensional governance assessment**: CPI alone is insufficient; the World Bank WGI provides six governance dimensions (Voice and Accountability, Political Stability, Government Effectiveness, Regulatory Quality, Rule of Law, Control of Corruption) that together paint a comprehensive picture of institutional quality, but operators do not track these systematically.
- **No sector-specific bribery risk assessment**: Generic country-level corruption scores do not capture the elevated bribery risk in specific sectors critical to EUDR -- forestry departments, customs and border agencies, agricultural ministries, and mining/extraction authorities each carry distinct bribery risk profiles.
- **No corruption-deforestation correlation**: Operators cannot quantitatively assess how changes in a country's corruption indices translate to changes in deforestation risk and therefore EUDR compliance risk.
- **No trend analysis or early warning**: Corruption indices evolve over years; operators need to detect deteriorating governance trajectories before they trigger EUDR country reclassification, not after.
- **No automated alert system**: When a sourcing country's CPI drops significantly or its WGI deteriorates, operators learn about it months later -- too late to adjust due diligence levels or supply chain configurations.
- **No compliance impact mapping**: Even when corruption index changes are noted, operators cannot systematically translate those changes into specific EUDR compliance actions (e.g., escalating from standard to enhanced due diligence for a country).

Without solving these problems, operators risk conducting insufficient due diligence for countries where governance has deteriorated, leading to non-compliant Due Diligence Statements, regulatory penalties of up to 4% of annual EU turnover, goods confiscation, and reputational damage from association with corruption-linked supply chains.

### 1.2 Solution Overview

Agent-EUDR-019: Corruption Index Monitor is a specialized agent that continuously monitors, analyzes, and maps corruption perceptions and governance integrity indicators for all EUDR-relevant countries, translating these indices into actionable EUDR compliance risk signals. The agent integrates authoritative corruption and governance data sources (Transparency International CPI, World Bank WGI, TRACE Bribery Risk Matrix), performs multi-dimensional governance quality scoring, detects deterioration trends, correlates corruption with deforestation outcomes, and generates automated alerts when governance changes impact EUDR compliance obligations.

Core capabilities:

1. **CPI Monitoring** -- Track Transparency International Corruption Perceptions Index scores for 180+ countries with historical trend analysis (2012-2025+), percentile rankings, regional benchmarking, and year-over-year change detection.
2. **WGI Integration** -- Ingest and analyze all six World Bank Worldwide Governance Indicators (Voice and Accountability, Political Stability and Absence of Violence, Government Effectiveness, Regulatory Quality, Rule of Law, Control of Corruption) for 200+ countries with historical data from 1996.
3. **Bribery Risk Assessment** -- Assess sector-specific bribery risk for four EUDR-critical sectors (forestry/timber, customs/border control, agriculture, mining/extraction) using TRACE Bribery Risk Matrix data and proprietary risk models.
4. **Institutional Quality Scoring** -- Compute composite institutional integrity scores combining judicial independence, regulatory enforcement capacity, forest governance effectiveness, and anti-corruption framework maturity.
5. **Trend Analysis** -- Detect multi-year corruption trajectories using statistical methods (linear regression, Mann-Kendall trend tests, change point detection), classify countries as improving/stable/deteriorating, and predict near-term governance trajectories.
6. **Deforestation Correlation** -- Perform statistical correlation analysis between corruption indices and national/sub-national deforestation rates using regression models, lag analysis, and causal pathway assessment.
7. **Alert Generation** -- Generate configurable alerts when CPI scores change beyond thresholds, WGI dimensions deteriorate, trend reversals occur, or countries approach EUDR reclassification boundaries.
8. **Compliance Impact Assessment** -- Map corruption index changes to specific EUDR Article 29 benchmarking impacts, recommend due diligence level adjustments (standard vs. enhanced), and trigger supply chain risk recalculation for affected sourcing countries.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Country coverage | 180+ countries with CPI data; 200+ with WGI data | Count of countries in active monitoring |
| Data freshness | CPI/WGI data ingested within 48 hours of publication | Time lag between source publication and system ingestion |
| Alert latency | Alerts generated within 1 hour of data ingestion | Time from ingestion to alert delivery |
| Trend detection accuracy | 90%+ accuracy in classifying country trajectories | Backtested against historical CPI/WGI data |
| Correlation significance | p < 0.05 for corruption-deforestation regressions | Statistical significance of regression models |
| Compliance impact mapping | 100% of CPI/WGI changes mapped to EUDR impact | % of index changes with compliance assessment |
| Processing performance | < 2 seconds for single-country full assessment | p99 latency under load |
| Determinism | 100% reproducible scoring (zero-hallucination) | Bit-perfect reproducibility tests |
| EUDR Article 29 alignment | Full coverage of EC benchmarking criteria | Regulatory mapping completeness |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR, plus the broader ESG compliance market where corruption/governance risk intelligence is valued, estimated at 2-4 billion EUR for governance risk technology.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of EUDR-regulated commodities who must assess country-level governance risk as part of their due diligence, estimated at 500M-900M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers using GreenLang's EUDR platform, with corruption monitoring as a high-value differentiating module, representing 20M-40M EUR in governance intelligence module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) sourcing from countries with volatile governance indicators
- Multinational food and beverage companies sourcing cocoa from West Africa, coffee from East Africa/Central America, palm oil from Southeast Asia
- Timber and paper industry operators sourcing tropical wood from countries with high corruption indices
- Compliance teams responsible for EUDR Article 29 risk assessments and enhanced due diligence determinations

**Secondary:**
- Risk management teams at EU financial institutions with exposure to EUDR-regulated supply chains
- Government procurement officers ensuring EUDR compliance for public sector purchases
- ESG ratings agencies and sustainability consultants advising on governance risk
- Commodity traders and intermediaries operating in high-corruption jurisdictions
- Certification bodies (FSC, RSPO, Rainforest Alliance) incorporating governance risk into certification decisions

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual CPI lookup | Free; widely known | Annual data only; no sector specificity; no trend analysis; no EUDR mapping | Automated, multi-source, sector-specific, EUDR-mapped |
| Generic ESG platforms (MSCI, Sustainalytics) | Broad ESG coverage | Not EUDR-specific; no corruption-deforestation correlation; no due diligence level recommendation | Purpose-built for EUDR Art. 29; deforestation correlation |
| Corruption intelligence providers (Transparency International, GAN) | Authoritative data | Raw data only; no EUDR compliance mapping; no supply chain integration; no alerting | Automated compliance mapping; supply chain integration |
| Bribery risk tools (TRACE, Dow Jones) | Sector granularity | Expensive standalone licenses; no EUDR context; no deforestation correlation | Integrated within EUDR platform; deforestation linkage |
| In-house risk teams | Organizational knowledge | Manual process; infrequent updates; no quantitative correlation; no scalability | Automated; quantitative; real-time; scalable |

### 2.4 Differentiation Strategy

1. **EUDR-specific governance intelligence** -- Not generic corruption data, but corruption indicators specifically contextualized for EUDR Article 29 compliance, with direct mapping to due diligence level determinations.
2. **Multi-source fusion** -- Combines CPI, WGI (6 dimensions), TRACE bribery risk, and forest governance indicators into a single composite assessment, weighted for EUDR relevance.
3. **Corruption-deforestation nexus** -- Unique quantitative correlation between governance indicators and deforestation outcomes, providing evidence-based risk justification for regulatory submissions.
4. **Sector-specific bribery assessment** -- Goes beyond country-level scores to assess bribery risk in the four sectors most relevant to EUDR (forestry, customs, agriculture, mining).
5. **Integrated supply chain impact** -- Changes in corruption indices automatically trigger risk recalculation across the operator's supply chain graph (via integration with AGENT-EUDR-001).
6. **Predictive trend intelligence** -- Statistical trend detection and trajectory prediction enable proactive rather than reactive compliance adjustments.
7. **Zero-hallucination calculation** -- All scores, correlations, and impact assessments are computed deterministically using published data and transparent formulas, with no LLM in the scoring path.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to incorporate governance risk into EUDR Article 29 compliance assessments | 100% of customers include corruption data in DDS risk assessments | Q2 2026 |
| BG-2 | Provide early warning of governance deterioration in sourcing countries | Alerts delivered within 48 hours of CPI/WGI publication | Q2 2026 |
| BG-3 | Reduce manual research time for country governance assessment from days to minutes | 95% reduction in manual governance research time | Q2 2026 |
| BG-4 | Establish GreenLang as the reference platform for EUDR governance risk intelligence | 300+ enterprise customers using corruption monitoring | Q4 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Comprehensive CPI tracking | Monitor CPI scores for all 180+ countries with historical trends and regional analysis |
| PG-2 | Multi-dimensional WGI analysis | Track all 6 WGI dimensions for 200+ countries with temporal analysis |
| PG-3 | Sector-specific bribery assessment | Assess bribery risk in forestry, customs, agriculture, and mining for each EUDR-relevant country |
| PG-4 | Institutional quality scoring | Compute composite governance scores combining judicial, regulatory, and enforcement metrics |
| PG-5 | Trend detection and prediction | Detect multi-year corruption trajectories and predict near-term governance direction |
| PG-6 | Corruption-deforestation correlation | Quantify statistical relationship between corruption indices and deforestation rates |
| PG-7 | Automated alerting | Generate configurable alerts on significant corruption index changes |
| PG-8 | EUDR compliance impact mapping | Translate corruption changes into specific EUDR due diligence level recommendations |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Country assessment latency | < 2 seconds p99 for single-country full assessment |
| TG-2 | Batch processing throughput | 200 countries assessed in < 30 seconds |
| TG-3 | Correlation computation | < 5 seconds for full regression model per country |
| TG-4 | API response time | < 200ms p95 for standard queries |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility for all scores |
| TG-7 | Data freshness | CPI/WGI ingested within 48 hours of source publication |
| TG-8 | Historical coverage | CPI data from 2012+; WGI data from 1996+ |

---

## 4. User Personas

### Persona 1: Compliance Officer -- Sofia (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of EUDR Compliance at a large EU chocolate manufacturer |
| **Company** | 8,000 employees, sourcing cocoa from Ghana, Ivory Coast, Cameroon, Nigeria, Ecuador |
| **EUDR Pressure** | Must determine correct due diligence level (standard vs. enhanced) for each sourcing country based on Article 29 criteria |
| **Pain Points** | Checks CPI once per year when published; has no visibility into WGI dimensions; cannot quantify how Ghana's governance trajectory affects EUDR risk; spends 3 days per country on manual governance research |
| **Goals** | Automated governance dashboards per sourcing country; proactive alerts when CPI/WGI changes; evidence-based due diligence level determination; audit-ready governance risk documentation |
| **Technical Skill** | Moderate -- comfortable with web applications and dashboards but not data analysis |

### Persona 2: Risk Analyst -- Henrik (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Risk Analyst at an EU timber import consortium |
| **Company** | Risk management shared service for 12 timber importers, sourcing from 35+ countries |
| **EUDR Pressure** | Must maintain country risk profiles for 35 sourcing countries with quantitative governance metrics |
| **Pain Points** | Uses multiple disconnected data sources (TI website, World Bank databank, TRACE reports); cannot correlate corruption trends with deforestation trends; no systematic way to detect governance deterioration across 35 countries simultaneously |
| **Goals** | Unified governance risk dashboard; automated trend detection; corruption-deforestation correlation reports; portfolio-level country risk summary |
| **Technical Skill** | High -- comfortable with data analysis, statistical methods, and APIs |

### Persona 3: Government Affairs Manager -- Claudia (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Director of Government Affairs at a multinational palm oil trader |
| **Company** | 2,000 employees, operating in Indonesia, Malaysia, Papua New Guinea, Colombia |
| **EUDR Pressure** | Must understand and anticipate EC country benchmarking decisions; must advise sourcing strategy on governance-related risks |
| **Pain Points** | Cannot predict when the EC will reclassify a country from Standard to High risk; lacks sector-specific bribery assessment for forestry and customs; surprises from governance crises in sourcing countries |
| **Goals** | Predictive governance intelligence; sector-specific bribery risk for palm oil supply chain; early warning of potential EC reclassification; briefing-ready country governance reports |
| **Technical Skill** | Low-moderate -- uses reports and dashboards; does not use APIs directly |

### Persona 4: External Auditor -- Dr. Bauer (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead EUDR Auditor at an EU-accredited verification body |
| **Company** | Third-party audit firm specializing in EUDR compliance |
| **EUDR Pressure** | Must verify that operator risk assessments properly account for country-level governance factors per Article 29 |
| **Pain Points** | Operators provide superficial governance assessments (e.g., "Ghana is medium risk"); no quantitative evidence; no multi-dimensional governance analysis; no trend documentation |
| **Goals** | Access to auditable governance data with provenance hashes; verify that operator due diligence levels align with current governance indicators; quantitative evidence trail for audit reports |
| **Technical Skill** | Moderate -- comfortable with audit software, data review, and compliance documentation |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 2(22)** | Definition of "negligible risk" -- products from low-risk countries under simplified due diligence | CPI/WGI data supports EC low-risk country identification; agent tracks which countries qualify for simplified DD |
| **Art. 2(23)** | Definition of "risk" -- likelihood of non-compliance | Corruption indices quantify the probability of governance failures that enable non-compliance |
| **Art. 10(1)** | Risk assessment -- operators must assess risk of non-compliance in supply chains | Agent provides quantitative governance risk data for Article 10 risk assessments |
| **Art. 10(2)(e)** | Risk factor: concerns about the country of production or parts thereof, including level of corruption | **Direct mandate**: Agent monitors corruption levels as explicit EUDR risk factor |
| **Art. 10(2)(f)** | Risk factor: concerns about circumvention and mixing of products of unknown origin | Agent assesses bribery risk in customs/border agencies that enables circumvention |
| **Art. 10(2)(h)** | Risk factor: concerns about the complexity of the supply chain | Agent correlates supply chain complexity in corrupt jurisdictions with elevated risk |
| **Art. 11(1)** | Risk mitigation -- adequate and proportionate measures | Agent recommends proportionate DD level based on governance indicators |
| **Art. 12** | Due Diligence Statement submission | Agent provides governance risk data for DDS risk assessment section |
| **Art. 23(1)** | Enhanced due diligence for products from high-risk countries | Agent identifies countries requiring enhanced DD based on governance indicators |
| **Art. 29(1)** | Country benchmarking by EC | Agent implements the governance dimensions of EC benchmarking criteria |
| **Art. 29(2)(a)** | Benchmarking factor: rate of deforestation and forest degradation | Agent correlates corruption with deforestation rates to support benchmarking analysis |
| **Art. 29(2)(c)** | Benchmarking factor: production trends of relevant commodities | Agent tracks governance quality alongside commodity production expansion |
| **Art. 29(2)(e)** | Benchmarking factor: existence and enforcement of legislation | Agent assesses regulatory quality and rule of law via WGI dimensions |
| **Art. 29(2)(f)** | Benchmarking factor: country's engagement in anti-corruption and anti-deforestation | Agent tracks anti-corruption trajectory via CPI/WGI trends |
| **Art. 29(3)** | EC shall publish and regularly update country classifications | Agent monitors and ingests EC benchmarking publications for hot-reload |
| **Art. 31** | Record keeping for 5 years | All governance assessments stored with immutable audit trail |

### 5.2 Key Governance Data Sources and EUDR Relevance

| Data Source | Publisher | Frequency | Countries | EUDR Relevance |
|-------------|-----------|-----------|-----------|----------------|
| **Corruption Perceptions Index (CPI)** | Transparency International | Annual (Jan/Feb) | 180+ | Direct Art. 10(2)(e) corruption indicator; primary input for country risk scoring |
| **Worldwide Governance Indicators (WGI)** | World Bank | Annual (Sept/Oct) | 200+ | Six governance dimensions covering Art. 29(2)(e) enforcement, Art. 29(2)(f) engagement |
| **TRACE Bribery Risk Matrix** | TRACE International | Annual | 194 | Sector-specific bribery risk for forestry/customs/agriculture/mining |
| **Forest Governance Assessment** | WRI / FAO / Chatham House | Periodic | 30+ | Direct forest governance quality indicators for Art. 29 analysis |
| **Ibrahim Index of African Governance (IIAG)** | Mo Ibrahim Foundation | Annual | 54 (Africa) | Supplementary governance data for major African sourcing countries |
| **EC Country Benchmarking List** | European Commission | Periodic (Art. 29(3)) | All | Official EUDR country classifications (Low/Standard/High) |

### 5.3 Corruption-Deforestation Evidence Base

The academic and institutional evidence linking corruption to deforestation is extensive:

| Evidence | Finding | Implication for EUDR |
|----------|---------|---------------------|
| Transparency International (2023) | Countries in the bottom quartile of CPI have 3x higher deforestation rates | Low CPI directly predicts EUDR non-compliance risk |
| World Bank (2022) | A 10-point drop in Control of Corruption WGI associates with 15% increase in illegal logging | WGI deterioration is a leading indicator for deforestation |
| Chatham House (2021) | 70% of illegal timber comes from countries scoring below 40 on CPI | CPI threshold (40) is a meaningful EUDR risk boundary |
| INTERPOL (2020) | Forest crime enabled by corruption in permits, customs, and enforcement | Sector-specific bribery assessment is essential for EUDR |
| FAO (2022) | Weak forest governance accounts for 50% of tropical deforestation | Institutional quality directly determines deforestation outcome |

### 5.4 Key Regulatory and Publication Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| January/February (annual) | TI publishes CPI | Agent ingests new CPI data; triggers reassessment for all countries |
| September/October (annual) | World Bank publishes WGI | Agent ingests new WGI data; triggers 6-dimension reassessment |
| December 31, 2020 | EUDR deforestation cutoff date | Baseline for corruption-deforestation correlation analysis |
| December 30, 2025 | EUDR enforcement for large operators | Operators must have governance risk assessment operational |
| June 30, 2026 | EUDR enforcement for SMEs | SME onboarding wave; agent must handle scale |
| Ongoing (periodic) | EC publishes country benchmarking updates (Art. 29(3)) | Agent ingests classifications; validates alignment with own governance assessment |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 8 features below are P0 launch blockers. The agent cannot ship without all 8 features operational. Features 1-5 form the core governance intelligence engine; Features 6-8 form the analytical and compliance integration layer that delivers actionable intelligence.

**P0 Features 1-5: Core Governance Intelligence Engine**

---

#### Feature 1: CPI Monitoring Engine

**User Story:**
```
As a compliance officer,
I want to continuously monitor Corruption Perceptions Index scores for all my sourcing countries,
So that I can track corruption trends and detect changes that affect my EUDR risk assessments.
```

**Acceptance Criteria:**
- [ ] Tracks CPI scores for 180+ countries from 2012 to the latest published year
- [ ] Stores historical CPI data with year, score (0-100), rank, number of data sources, standard error, confidence interval (low/high)
- [ ] Computes year-over-year change (absolute and percentage) for each country
- [ ] Computes percentile ranking within global, regional, and income-group cohorts
- [ ] Identifies statistically significant CPI changes (beyond standard error margin)
- [ ] Classifies countries into CPI tiers: Very Clean (80-100), Clean (60-79), Moderate (40-59), Corrupt (20-39), Highly Corrupt (0-19)
- [ ] Generates regional analysis (Africa, Americas, Asia-Pacific, Eastern Europe/Central Asia, EU/Western Europe, Middle East/North Africa)
- [ ] Provides EUDR-specific CPI filtering: CPI scores only for countries producing EUDR-regulated commodities
- [ ] Supports bulk ingestion of CPI data from Transparency International published datasets (CSV/Excel/API)
- [ ] Computes 3-year and 5-year rolling averages to smooth annual fluctuations

**Non-Functional Requirements:**
- Data Quality: All CPI values validated against official TI published ranges
- Performance: Full CPI dataset query < 500ms for all 180+ countries
- Reproducibility: Deterministic scoring from same input data
- Auditability: Every CPI data ingestion logged with source, timestamp, and provenance hash

**Dependencies:**
- Transparency International CPI published datasets (annual CSV/Excel)
- PostgreSQL for persistent storage
- Redis for query caching

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Country not scored by TI in a given year -- flag as "CPI unavailable" with last known score
- Country name/code changes (e.g., eSwatini formerly Swaziland) -- maintain country code mapping
- TI methodology changes between years -- flag comparability warnings

---

#### Feature 2: WGI Analyzer Engine

**User Story:**
```
As a risk analyst,
I want to analyze all six World Bank governance dimensions for my sourcing countries,
So that I can assess governance quality comprehensively rather than relying on a single corruption score.
```

**Acceptance Criteria:**
- [ ] Tracks all 6 WGI dimensions for 200+ countries from 1996 to latest published year:
  - Voice and Accountability (VA)
  - Political Stability and Absence of Violence (PV)
  - Government Effectiveness (GE)
  - Regulatory Quality (RQ)
  - Rule of Law (RL)
  - Control of Corruption (CC)
- [ ] Stores WGI data with: estimate (-2.5 to +2.5), percentile rank (0-100), standard error, number of sources, lower bound, upper bound
- [ ] Computes composite governance score aggregating all 6 dimensions with configurable weights
- [ ] Default EUDR-specific weighting: Control of Corruption (0.30), Rule of Law (0.25), Government Effectiveness (0.20), Regulatory Quality (0.15), Political Stability (0.05), Voice and Accountability (0.05)
- [ ] Identifies dimension-specific strengths and weaknesses per country (radar chart data)
- [ ] Computes year-over-year change for each dimension
- [ ] Detects statistically significant changes (beyond standard error margin)
- [ ] Compares countries within the same region or income group
- [ ] Generates "forest governance proxy" score: weighted combination of Rule of Law + Regulatory Quality + Government Effectiveness
- [ ] Supports bulk ingestion of WGI data from World Bank Databank (CSV/API)

**Non-Functional Requirements:**
- Data Quality: All WGI values validated against World Bank published ranges (-2.5 to +2.5 for estimates; 0-100 for percentile)
- Performance: Full 6-dimension assessment for single country < 200ms
- Completeness: 200+ countries with at least 4 of 6 dimensions available
- Historical Depth: 20+ years of WGI data for trend analysis

**Dependencies:**
- World Bank WGI published datasets (annual CSV/API)
- PostgreSQL with TimescaleDB for time-series storage
- Statistical libraries (scipy, statsmodels) for significance testing

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- WGI data not available for all 6 dimensions for a country -- compute partial composite with available dimensions
- WGI methodology changes (rare but occurred in 2012) -- flag year with methodology change note
- Territories and dependencies without WGI data -- fall back to parent country or flag as unavailable

---

#### Feature 3: Bribery Risk Engine

**User Story:**
```
As a compliance officer,
I want to understand the specific bribery risk in forestry, customs, agriculture, and mining sectors of my sourcing countries,
So that I can assess whether corruption in these EUDR-critical sectors could compromise my supply chain integrity.
```

**Acceptance Criteria:**
- [ ] Assesses bribery risk in four EUDR-critical sectors per country:
  - Forestry/Timber: risk in forestry departments, logging permits, concession allocation, forest patrols
  - Customs/Border Control: risk in customs agencies, border inspections, product classification, document verification
  - Agriculture: risk in agricultural ministries, land titling, pesticide/fertilizer regulation, subsidy allocation
  - Mining/Extraction: risk in mining ministries, extraction permits, environmental impact assessments
- [ ] Integrates TRACE Bribery Risk Matrix data (4 sub-components: business interactions with government, anti-bribery deterrence and enforcement, government and civil service transparency, capacity for civil society oversight)
- [ ] Computes sector-specific bribery risk score (0-100) per country per sector
- [ ] Identifies sector-country combinations with highest bribery risk
- [ ] Maps bribery risk to specific EUDR supply chain vulnerabilities:
  - Forestry bribery -> fraudulent logging permits -> illegal timber entering supply chain
  - Customs bribery -> falsified origin documents -> circumvention risk per Art. 10(2)(f)
  - Agriculture bribery -> falsified land use records -> unreliable geolocation data
  - Mining bribery -> unauthorized extraction -> unknown origin materials
- [ ] Generates bribery risk matrix (heatmap) for all EUDR sourcing countries across 4 sectors
- [ ] Computes composite sector bribery score per country with configurable sector weights
- [ ] Supports manual override of sector risk scores with documented justification
- [ ] Flags countries where sector risk significantly exceeds overall CPI-implied risk

**Non-Functional Requirements:**
- Coverage: All EUDR-relevant sourcing countries (minimum 50 countries)
- Granularity: Sector-level scores with supporting evidence indicators
- Determinism: Reproducible scores from same input data
- Auditability: SHA-256 provenance hash on every bribery risk assessment

**Dependencies:**
- TRACE Bribery Risk Matrix published data
- CPI data from Feature 1 (CPIMonitorEngine) for cross-validation
- WGI data from Feature 2 (WGIAnalyzerEngine) for dimension-level context
- Country-sector risk databases (GAN Business Anti-Corruption Portal, OECD)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 domain expert)

---

#### Feature 4: Institutional Quality Engine

**User Story:**
```
As a risk analyst,
I want a comprehensive institutional integrity score that goes beyond corruption perception to assess the actual governance capacity of each sourcing country,
So that I can determine whether a country's institutions can effectively enforce forestry regulations.
```

**Acceptance Criteria:**
- [ ] Computes institutional integrity score (0-100) per country based on four pillars:
  - Judicial Independence: independence of judiciary, judicial review effectiveness, contract enforcement
  - Regulatory Enforcement Capacity: regulatory agency independence, inspection frequency, penalty enforcement rate
  - Forest Governance Effectiveness: forest law quality, monitoring capacity, illegal logging interdiction rate, protected area management
  - Anti-Corruption Framework: anti-corruption legislation, enforcement body independence, asset declaration requirements, whistleblower protection
- [ ] Weights the four pillars with EUDR-specific defaults: Forest Governance (0.35), Regulatory Enforcement (0.25), Judicial Independence (0.20), Anti-Corruption Framework (0.20)
- [ ] Sources pillar data from: WGI (RL, RQ, GE dimensions), World Justice Project Rule of Law Index, Forest Governance Initiative data, UNCAC implementation scores
- [ ] Classifies institutional quality: Strong (80-100), Adequate (60-79), Weak (40-59), Very Weak (20-39), Failed (0-19)
- [ ] Identifies critical institutional weaknesses per country (e.g., "Indonesia: weak forest governance (32/100) despite moderate overall CPI")
- [ ] Generates institutional quality comparison across EUDR sourcing countries
- [ ] Tracks institutional quality changes over time (annual updates)
- [ ] Flags disconnect between CPI and institutional quality scores (e.g., country with moderate CPI but very weak forest governance)
- [ ] Supports custom institutional quality models per commodity (timber-specific vs. palm oil-specific governance assessment)
- [ ] Provides explanation/evidence for each pillar score (source citations, indicator values)

**Non-Functional Requirements:**
- Transparency: Every pillar score backed by identifiable source data
- Coverage: Institutional quality scores for 80+ EUDR-relevant countries
- Reproducibility: Deterministic computation from published indicator data
- Granularity: Pillar-level scores with supporting indicators

**Dependencies:**
- WGI data from Feature 2 (WGIAnalyzerEngine)
- World Justice Project Rule of Law Index data
- FAO / WRI Forest Governance Initiative data
- UNCAC implementation review data

**Estimated Effort:** 3 weeks (1 backend engineer, 1 governance domain expert)

---

#### Feature 5: Trend Analysis Engine

**User Story:**
```
As a government affairs manager,
I want to detect multi-year corruption trends in my sourcing countries and receive early warning of governance deterioration,
So that I can anticipate potential EUDR country reclassification and proactively adjust our sourcing strategy.
```

**Acceptance Criteria:**
- [ ] Performs linear regression on CPI time series (minimum 5 years of data) to determine trend direction and slope
- [ ] Performs Mann-Kendall trend test for non-parametric trend detection with statistical significance
- [ ] Performs change point detection (CUSUM or Bayesian) to identify years where governance trajectory shifted
- [ ] Classifies country trend: Improving (positive slope, p < 0.05), Stable (no significant trend), Deteriorating (negative slope, p < 0.05), Volatile (high variance, no clear trend)
- [ ] Computes trend velocity: rate of CPI change per year (points/year)
- [ ] Performs WGI trend analysis across all 6 dimensions independently
- [ ] Identifies dimension-specific trend divergence (e.g., "Control of Corruption improving while Rule of Law deteriorating")
- [ ] Generates trajectory prediction: projected CPI score 1 year and 3 years forward based on linear/polynomial extrapolation with confidence intervals
- [ ] Detects regime change impact: identifies post-election/post-coup governance trajectory shifts
- [ ] Classifies countries approaching EUDR risk boundaries:
  - Countries approaching Standard-to-High threshold
  - Countries approaching Low-to-Standard threshold
  - Countries stable in their current classification
- [ ] Provides "governance momentum" indicator combining CPI trend + WGI trends + institutional quality trend

**Non-Functional Requirements:**
- Statistical Rigor: All trend tests include p-value and confidence intervals
- Minimum Data: Require 5+ years of data for trend analysis; flag shorter series
- Performance: Full trend analysis for single country < 1 second
- Interpretability: Human-readable trend narrative generated from statistical results

**Dependencies:**
- CPI historical data from Feature 1 (CPIMonitorEngine)
- WGI historical data from Feature 2 (WGIAnalyzerEngine)
- Statistical libraries: scipy (linregress, kendalltau), ruptures (change point detection), statsmodels
- EC country benchmarking thresholds (for boundary proximity analysis)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data scientist)

---

**P0 Features 6-8: Analytical and Compliance Integration Layer**

> Features 6, 7, and 8 are P0 launch blockers. Without correlation analysis, alerting, and compliance impact mapping, the governance intelligence engine cannot deliver actionable EUDR compliance value. These features translate raw governance data into compliance decisions.

---

#### Feature 6: Deforestation Correlation Engine

**User Story:**
```
As a risk analyst,
I want to see the statistical relationship between corruption indices and deforestation rates for my sourcing countries,
So that I can provide evidence-based justification for elevated due diligence in corrupt jurisdictions.
```

**Acceptance Criteria:**
- [ ] Computes Pearson and Spearman correlation between CPI and national deforestation rate (tree cover loss per year)
- [ ] Computes correlation between each WGI dimension and deforestation rate independently
- [ ] Performs multivariate regression: deforestation rate ~ CPI + WGI dimensions + commodity production + GDP per capita
- [ ] Performs lag analysis: tests whether CPI changes in year T predict deforestation changes in year T+1, T+2
- [ ] Generates country-specific regression models for countries with sufficient data (10+ years)
- [ ] Identifies "corruption-deforestation hotspots": countries where both corruption is high AND deforestation is accelerating
- [ ] Generates cross-country scatter plot data (CPI vs. deforestation rate) with EUDR commodity annotations
- [ ] Computes R-squared, p-value, and confidence intervals for all regression models
- [ ] Generates causal pathway narratives: "In [Country], high bribery risk in forestry permits (score: X) likely enables illegal logging contributing to Y% tree cover loss"
- [ ] Supports sub-national analysis where data permits (e.g., state/province level in Brazil, Indonesia)
- [ ] Updates correlation models when new CPI/WGI/deforestation data becomes available

**Non-Functional Requirements:**
- Statistical Validity: All correlations include significance tests (p < 0.05 threshold)
- Data Quality: Minimum 10 data points per regression model; flag smaller samples
- Performance: Full correlation computation per country < 5 seconds
- Reproducibility: Deterministic results from same input datasets

**Dependencies:**
- CPI data from Feature 1 (CPIMonitorEngine)
- WGI data from Feature 2 (WGIAnalyzerEngine)
- Deforestation data from AGENT-DATA-007 Deforestation Satellite Connector or Global Forest Watch
- National commodity production data (FAOSTAT)
- GDP per capita data (World Bank)
- Statistical libraries: scipy, statsmodels, scikit-learn (for regression)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data scientist)

---

#### Feature 7: Alert Engine

**User Story:**
```
As a compliance officer,
I want to receive immediate alerts when corruption indices change significantly for my sourcing countries,
So that I can take timely action to adjust my EUDR risk assessments and due diligence procedures.
```

**Acceptance Criteria:**
- [ ] Generates alerts for the following trigger conditions:
  - CPI score change exceeds configurable threshold (default: +/- 3 points)
  - CPI rank change exceeds configurable threshold (default: +/- 10 ranks)
  - WGI dimension drops below configurable absolute threshold (default: 25th percentile)
  - WGI dimension year-over-year change exceeds threshold (default: -0.2 estimate change)
  - Trend reversal detected (country previously improving now deteriorating, or vice versa)
  - Country approaching EUDR reclassification boundary (within configurable margin)
  - Sector bribery risk score exceeds threshold for an EUDR-critical sector
  - EC publishes new country benchmarking classification
  - Institutional quality score drops below threshold (default: 40/100)
- [ ] Supports alert severity levels: Critical, High, Medium, Low, Informational
- [ ] Critical alerts for: country reclassified to High risk by EC; CPI drops below 25; WGI Control of Corruption below 10th percentile
- [ ] High alerts for: significant CPI decline (>5 points); trend reversal from improving to deteriorating; sector bribery risk exceeds 80
- [ ] Medium alerts for: moderate CPI change (3-5 points); boundary proximity; institutional quality decline
- [ ] Low alerts for: minor CPI change (1-3 points); regional trend changes; informational updates
- [ ] Supports configurable alert subscriptions per operator: which countries, which trigger types, which severity levels
- [ ] Supports alert delivery channels: in-platform notification, email, webhook (for integration with operator systems)
- [ ] Tracks alert acknowledgement: unacknowledged, acknowledged, action taken, dismissed with reason
- [ ] Generates alert history report with trend analysis of alert frequency per country
- [ ] Suppresses duplicate alerts within configurable cooldown period (default: 30 days)

**Non-Functional Requirements:**
- Latency: Alerts generated within 1 hour of new data ingestion
- Reliability: Zero missed alerts for Critical and High severity triggers
- Configurability: All thresholds configurable per operator without code changes
- Auditability: Every alert generation and acknowledgement logged with timestamp

**Dependencies:**
- CPI data from Feature 1 (CPIMonitorEngine)
- WGI data from Feature 2 (WGIAnalyzerEngine)
- Bribery risk from Feature 3 (BriberyRiskEngine)
- Institutional quality from Feature 4 (InstitutionalQualityEngine)
- Trend data from Feature 5 (TrendAnalysisEngine)
- Email notification service (existing GreenLang infrastructure)
- Webhook delivery service

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 8: Compliance Impact Engine

**User Story:**
```
As a compliance officer,
I want corruption index changes to automatically translate into specific EUDR compliance recommendations,
So that I know exactly which due diligence actions I need to take when a sourcing country's governance deteriorates.
```

**Acceptance Criteria:**
- [ ] Maps corruption index changes to EUDR Article 29 benchmarking impact:
  - CPI/WGI improvement -> potential country upgrade from High to Standard or Standard to Low
  - CPI/WGI deterioration -> potential country downgrade from Low to Standard or Standard to High
- [ ] Recommends due diligence level per country:
  - Simplified due diligence: CPI >= 70 AND WGI all dimensions above 50th percentile AND trend stable/improving
  - Standard due diligence: CPI 40-69 OR WGI mixed dimensions OR trend volatile
  - Enhanced due diligence: CPI < 40 OR WGI Control of Corruption below 25th percentile OR trend deteriorating OR sector bribery risk >= 75
- [ ] Generates specific compliance actions when DD level changes:
  - Standard -> Enhanced: "Increase verification frequency; obtain independent third-party audit for [Country]; conduct field verification of geolocation data"
  - Enhanced -> Standard: "Reduce verification frequency; revert to desktop review for [Country]"
- [ ] Triggers supply chain risk recalculation: when a country's corruption assessment changes, notifies AGENT-EUDR-001 Supply Chain Mapping Master to re-propagate country risk scores through affected supply chain graphs
- [ ] Generates DDS risk assessment data: provides pre-formatted governance risk section for inclusion in Due Diligence Statement per Article 12
- [ ] Computes compliance cost impact: estimates additional due diligence cost when country moves to enhanced DD (based on configurable cost model)
- [ ] Supports multi-scenario analysis: "What if Ghana's CPI drops by 5 points?" -> shows compliance impact before it happens
- [ ] Generates quarterly compliance briefing: summary of governance changes across all sourcing countries with compliance action recommendations
- [ ] Maintains compliance action audit trail: tracks which governance triggers led to which DD level adjustments for regulatory defense
- [ ] Aligns recommendations with EC official country classifications when published, with flag for discrepancies

**Non-Functional Requirements:**
- Determinism: Same governance inputs always produce same compliance recommendations
- Traceability: Every recommendation linked to specific governance indicator values
- Timeliness: Compliance recommendations updated within 2 hours of governance data change
- Completeness: Every country in operator's supply chain has a current compliance recommendation

**Dependencies:**
- All prior engines (CPIMonitor, WGIAnalyzer, BriberyRisk, InstitutionalQuality, TrendAnalysis, DeforestationCorrelation)
- Alert engine (Feature 7) for triggering compliance reassessment
- AGENT-EUDR-001 Supply Chain Mapping Master (for risk re-propagation)
- GL-EUDR-APP DDS Reporting Engine (for DDS data formatting)
- EC country benchmarking publications

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 9: Sub-National Governance Assessment
- Assess governance quality at state/province level for large countries (Brazil, Indonesia, DRC, India)
- Map sub-national governance variation to sub-national deforestation patterns
- Enable more granular risk assessment than country-level indicators permit

#### Feature 10: Political Risk and Regime Change Monitoring
- Monitor political events (elections, coups, sanctions) that impact governance trajectories
- Assess impact of regime changes on forest governance and corruption enforcement
- Integrate news sentiment analysis for real-time political risk signals

#### Feature 11: Peer Benchmarking and Comparative Analytics
- Compare governance risk profiles across operator's sourcing countries
- Compare operator's sourcing country governance profile against industry peers
- Generate "governance portfolio" optimization recommendations

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Real-time news monitoring for corruption events (defer to media monitoring integration)
- Individual company-level corruption risk assessment (agent operates at country/sector level)
- Bribery prosecution database integration (future enhancement)
- Blockchain-based governance data verification
- Machine learning-based governance prediction (statistical methods only in v1.0)
- Anti-money laundering (AML) integration
- Sanctions screening (separate compliance domain)

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
| AGENT-EUDR-019        |           | AGENT-EUDR-001            |           | AGENT-DATA-007        |
| Corruption Index      |<--------->| Supply Chain Mapping      |<--------->| Deforestation         |
| Monitor               |           | Master                    |           | Satellite Connector   |
|                       |           |                           |           |                       |
| - CPIMonitorEngine    |           | - GraphEngine             |           | - GFW Client          |
| - WGIAnalyzerEngine   |           | - RiskPropagation         |           | - Tree Cover Loss     |
| - BriberyRiskEngine   |           | - GapAnalyzer             |           | - NDVI Calculator     |
| - InstitutionalQuality|           | - Multi-Tier Mapper       |           | - Forest Change Det.  |
| - TrendAnalysis       |           | - RegExporter             |           |                       |
| - DeforestCorrelation |           |                           |           |                       |
| - AlertEngine         |           |                           |           |                       |
| - ComplianceImpact    |           |                           |           |                       |
+-----------+-----------+           +---------------------------+           +-----------------------+
            |
+-----------v-----------+           +---------------------------+
| External Data Sources |           | GL-EUDR-APP Platform      |
|                       |           |                           |
| - TI CPI Dataset      |           | - DDS Reporting Engine    |
| - WB WGI Dataset      |           | - Country Risk Module     |
| - TRACE BRM Data      |           | - Alert Dashboard         |
| - GFW Deforestation   |           | - Compliance Dashboard    |
| - EC Benchmarking     |           |                           |
+-----------------------+           +---------------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/corruption_index_monitor/
    __init__.py                          # Public API exports
    config.py                            # CorruptionIndexMonitorConfig with GL_EUDR_CIM_ env prefix
    models.py                            # Pydantic v2 models for CPI, WGI, risk scores, alerts
    cpi_monitor_engine.py                # CPIMonitorEngine: CPI tracking, analysis, tier classification
    wgi_analyzer_engine.py               # WGIAnalyzerEngine: 6-dimension WGI analysis
    bribery_risk_engine.py               # BriberyRiskEngine: sector-specific bribery assessment
    institutional_quality_engine.py      # InstitutionalQualityEngine: institutional integrity scoring
    trend_analysis_engine.py             # TrendAnalysisEngine: statistical trend detection
    deforestation_correlation_engine.py  # DeforestationCorrelationEngine: corruption-deforestation nexus
    alert_engine.py                      # AlertEngine: configurable governance alerts
    compliance_impact_engine.py          # ComplianceImpactEngine: EUDR compliance mapping
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains
    metrics.py                           # 18 Prometheus self-monitoring metrics
    setup.py                             # CorruptionIndexMonitorService facade
    api/
        __init__.py
        router.py                        # FastAPI router (30+ endpoints)
        cpi_routes.py                    # CPI monitoring endpoints
        wgi_routes.py                    # WGI analysis endpoints
        bribery_routes.py                # Bribery risk endpoints
        institutional_routes.py          # Institutional quality endpoints
        trend_routes.py                  # Trend analysis endpoints
        correlation_routes.py            # Deforestation correlation endpoints
        alert_routes.py                  # Alert management endpoints
        compliance_routes.py             # Compliance impact endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Corruption Perceptions Index Score
class CPIScore(BaseModel):
    country_code: str            # ISO 3166-1 alpha-2
    country_name: str            # Official country name
    year: int                    # CPI publication year
    score: int                   # CPI score (0-100, 0 = highly corrupt)
    rank: int                    # Global rank
    num_sources: int             # Number of data sources
    standard_error: Optional[float]
    confidence_low: Optional[int]
    confidence_high: Optional[int]
    tier: str                    # VERY_CLEAN/CLEAN/MODERATE/CORRUPT/HIGHLY_CORRUPT
    yoy_change: Optional[int]   # Year-over-year score change
    percentile_global: float     # Global percentile rank
    percentile_regional: float   # Regional percentile rank
    provenance_hash: str         # SHA-256

# World Governance Indicator
class WGIIndicator(BaseModel):
    country_code: str
    country_name: str
    year: int
    dimension: str               # VA/PV/GE/RQ/RL/CC
    dimension_name: str          # Full dimension name
    estimate: float              # Governance score (-2.5 to +2.5)
    percentile_rank: float       # Percentile rank (0-100)
    standard_error: float
    num_sources: int
    lower_bound: float
    upper_bound: float
    yoy_change: Optional[float]
    provenance_hash: str

# Bribery Risk Assessment
class BriberyRiskAssessment(BaseModel):
    country_code: str
    country_name: str
    assessment_date: datetime
    sector: str                  # FORESTRY/CUSTOMS/AGRICULTURE/MINING
    sector_risk_score: float     # 0-100 (100 = highest risk)
    trace_score: Optional[float] # TRACE BRM overall score
    government_interaction_risk: float
    anti_bribery_enforcement: float
    transparency_score: float
    civil_society_capacity: float
    eudr_vulnerability_mapping: str  # Narrative of EUDR-specific vulnerability
    composite_score: float       # Weighted composite across all sectors
    provenance_hash: str

# Institutional Quality Score
class InstitutionalQualityScore(BaseModel):
    country_code: str
    country_name: str
    assessment_date: datetime
    judicial_independence: float       # 0-100
    regulatory_enforcement: float      # 0-100
    forest_governance: float           # 0-100
    anti_corruption_framework: float   # 0-100
    composite_score: float             # Weighted composite
    quality_class: str                 # STRONG/ADEQUATE/WEAK/VERY_WEAK/FAILED
    critical_weaknesses: List[str]     # Key weaknesses identified
    source_indicators: Dict[str, Any]  # Source data for each pillar
    provenance_hash: str

# Trend Analysis Result
class TrendAnalysis(BaseModel):
    country_code: str
    country_name: str
    analysis_date: datetime
    indicator_type: str          # CPI/WGI_VA/WGI_PV/WGI_GE/WGI_RQ/WGI_RL/WGI_CC
    trend_direction: str         # IMPROVING/STABLE/DETERIORATING/VOLATILE
    trend_slope: float           # Points per year
    p_value: float               # Statistical significance
    r_squared: float             # Coefficient of determination
    mann_kendall_tau: float      # Mann-Kendall statistic
    change_points: List[int]     # Years where trend shifted
    predicted_1yr: float         # 1-year forward projection
    predicted_3yr: float         # 3-year forward projection
    confidence_interval_1yr: Tuple[float, float]
    confidence_interval_3yr: Tuple[float, float]
    boundary_proximity: Optional[str]  # APPROACHING_HIGH/APPROACHING_LOW/NONE
    governance_momentum: float   # Composite trend indicator
    provenance_hash: str

# Deforestation Correlation
class DeforestationCorrelation(BaseModel):
    country_code: str
    country_name: str
    analysis_date: datetime
    cpi_deforestation_pearson: float
    cpi_deforestation_spearman: float
    cpi_deforestation_p_value: float
    wgi_cc_deforestation_correlation: float
    regression_r_squared: float
    regression_coefficients: Dict[str, float]
    lag_correlation: Dict[int, float]    # lag years -> correlation
    is_hotspot: bool             # High corruption AND accelerating deforestation
    causal_narrative: str        # Evidence-based narrative
    provenance_hash: str

# Governance Alert
class GovernanceAlert(BaseModel):
    alert_id: str
    country_code: str
    country_name: str
    alert_type: str              # CPI_CHANGE/WGI_CHANGE/TREND_REVERSAL/BOUNDARY_PROXIMITY/EC_RECLASSIFICATION/SECTOR_BRIBERY/INSTITUTIONAL_DECLINE
    severity: str                # CRITICAL/HIGH/MEDIUM/LOW/INFORMATIONAL
    trigger_indicator: str       # Which indicator triggered the alert
    trigger_value: float         # Current value
    threshold_value: float       # Threshold that was breached
    previous_value: Optional[float]
    description: str             # Human-readable alert description
    compliance_impact: str       # Impact on EUDR compliance
    recommended_actions: List[str]
    status: str                  # UNACKNOWLEDGED/ACKNOWLEDGED/ACTION_TAKEN/DISMISSED
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    created_at: datetime
    provenance_hash: str

# Compliance Impact Assessment
class ComplianceImpact(BaseModel):
    country_code: str
    country_name: str
    assessment_date: datetime
    current_dd_level: str        # SIMPLIFIED/STANDARD/ENHANCED
    recommended_dd_level: str    # SIMPLIFIED/STANDARD/ENHANCED
    dd_level_changed: bool
    governance_risk_score: float # 0-100 composite governance risk
    ec_classification: Optional[str]  # LOW/STANDARD/HIGH (if published)
    ec_alignment: str            # ALIGNED/DIVERGENT/PENDING
    compliance_actions: List[str]
    cost_impact_estimate: Optional[float]  # Additional DD cost estimate
    affected_supply_chains: int  # Number of supply chains affected
    dds_risk_section: Dict[str, Any]  # Pre-formatted data for DDS
    provenance_hash: str

# Country Profile (Comprehensive)
class CountryProfile(BaseModel):
    country_code: str
    country_name: str
    region: str
    income_group: str
    latest_cpi: Optional[CPIScore]
    latest_wgi: Dict[str, WGIIndicator]  # dimension -> indicator
    bribery_risk: Dict[str, BriberyRiskAssessment]  # sector -> assessment
    institutional_quality: Optional[InstitutionalQualityScore]
    trend: Optional[TrendAnalysis]
    deforestation_correlation: Optional[DeforestationCorrelation]
    compliance_impact: Optional[ComplianceImpact]
    active_alerts: List[GovernanceAlert]
    last_updated: datetime
    provenance_hash: str
```

### 7.4 Database Schema (New Migration: V107)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_corruption_index_monitor;

-- CPI Scores (hypertable: time-series by year)
CREATE TABLE eudr_corruption_index_monitor.cpi_scores (
    id UUID DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    year INTEGER NOT NULL,
    score INTEGER NOT NULL CHECK (score >= 0 AND score <= 100),
    rank INTEGER NOT NULL,
    num_sources INTEGER DEFAULT 0,
    standard_error NUMERIC(6,3),
    confidence_low INTEGER,
    confidence_high INTEGER,
    tier VARCHAR(20) NOT NULL,
    yoy_change INTEGER,
    percentile_global NUMERIC(5,2),
    percentile_regional NUMERIC(5,2),
    provenance_hash VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, ingested_at)
);

SELECT create_hypertable('eudr_corruption_index_monitor.cpi_scores', 'ingested_at');

-- WGI Indicators (hypertable: time-series by year)
CREATE TABLE eudr_corruption_index_monitor.wgi_indicators (
    id UUID DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    year INTEGER NOT NULL,
    dimension VARCHAR(5) NOT NULL CHECK (dimension IN ('VA','PV','GE','RQ','RL','CC')),
    dimension_name VARCHAR(100) NOT NULL,
    estimate NUMERIC(6,3) NOT NULL CHECK (estimate >= -2.5 AND estimate <= 2.5),
    percentile_rank NUMERIC(5,2) NOT NULL CHECK (percentile_rank >= 0 AND percentile_rank <= 100),
    standard_error NUMERIC(6,3),
    num_sources INTEGER DEFAULT 0,
    lower_bound NUMERIC(6,3),
    upper_bound NUMERIC(6,3),
    yoy_change NUMERIC(6,3),
    provenance_hash VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, ingested_at)
);

SELECT create_hypertable('eudr_corruption_index_monitor.wgi_indicators', 'ingested_at');

-- Bribery Risk Assessments
CREATE TABLE eudr_corruption_index_monitor.bribery_risk_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    assessment_date TIMESTAMPTZ NOT NULL,
    sector VARCHAR(20) NOT NULL CHECK (sector IN ('FORESTRY','CUSTOMS','AGRICULTURE','MINING')),
    sector_risk_score NUMERIC(5,2) NOT NULL CHECK (sector_risk_score >= 0 AND sector_risk_score <= 100),
    trace_score NUMERIC(5,2),
    government_interaction_risk NUMERIC(5,2),
    anti_bribery_enforcement NUMERIC(5,2),
    transparency_score NUMERIC(5,2),
    civil_society_capacity NUMERIC(5,2),
    eudr_vulnerability_mapping TEXT,
    composite_score NUMERIC(5,2),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Institutional Quality Scores
CREATE TABLE eudr_corruption_index_monitor.institutional_quality_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    assessment_date TIMESTAMPTZ NOT NULL,
    judicial_independence NUMERIC(5,2) NOT NULL,
    regulatory_enforcement NUMERIC(5,2) NOT NULL,
    forest_governance NUMERIC(5,2) NOT NULL,
    anti_corruption_framework NUMERIC(5,2) NOT NULL,
    composite_score NUMERIC(5,2) NOT NULL,
    quality_class VARCHAR(20) NOT NULL,
    critical_weaknesses JSONB DEFAULT '[]',
    source_indicators JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Trend Analyses
CREATE TABLE eudr_corruption_index_monitor.trend_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    analysis_date TIMESTAMPTZ NOT NULL,
    indicator_type VARCHAR(10) NOT NULL,
    trend_direction VARCHAR(20) NOT NULL,
    trend_slope NUMERIC(8,4),
    p_value NUMERIC(10,8),
    r_squared NUMERIC(6,4),
    mann_kendall_tau NUMERIC(6,4),
    change_points JSONB DEFAULT '[]',
    predicted_1yr NUMERIC(8,3),
    predicted_3yr NUMERIC(8,3),
    confidence_interval_1yr JSONB,
    confidence_interval_3yr JSONB,
    boundary_proximity VARCHAR(30),
    governance_momentum NUMERIC(6,3),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Deforestation Correlations
CREATE TABLE eudr_corruption_index_monitor.deforestation_correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    analysis_date TIMESTAMPTZ NOT NULL,
    cpi_deforestation_pearson NUMERIC(6,4),
    cpi_deforestation_spearman NUMERIC(6,4),
    cpi_deforestation_p_value NUMERIC(10,8),
    wgi_cc_deforestation_correlation NUMERIC(6,4),
    regression_r_squared NUMERIC(6,4),
    regression_coefficients JSONB DEFAULT '{}',
    lag_correlation JSONB DEFAULT '{}',
    is_hotspot BOOLEAN DEFAULT FALSE,
    causal_narrative TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Alerts
CREATE TABLE eudr_corruption_index_monitor.alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('CRITICAL','HIGH','MEDIUM','LOW','INFORMATIONAL')),
    trigger_indicator VARCHAR(50) NOT NULL,
    trigger_value NUMERIC(8,3),
    threshold_value NUMERIC(8,3),
    previous_value NUMERIC(8,3),
    description TEXT NOT NULL,
    compliance_impact TEXT,
    recommended_actions JSONB DEFAULT '[]',
    status VARCHAR(20) DEFAULT 'UNACKNOWLEDGED',
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    dismissed_reason TEXT,
    cooldown_until TIMESTAMPTZ,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Compliance Impacts
CREATE TABLE eudr_corruption_index_monitor.compliance_impacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    assessment_date TIMESTAMPTZ NOT NULL,
    current_dd_level VARCHAR(20) NOT NULL,
    recommended_dd_level VARCHAR(20) NOT NULL,
    dd_level_changed BOOLEAN DEFAULT FALSE,
    governance_risk_score NUMERIC(5,2) NOT NULL,
    ec_classification VARCHAR(20),
    ec_alignment VARCHAR(20),
    compliance_actions JSONB DEFAULT '[]',
    cost_impact_estimate NUMERIC(12,2),
    affected_supply_chains INTEGER DEFAULT 0,
    dds_risk_section JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Country Profiles (comprehensive view)
CREATE TABLE eudr_corruption_index_monitor.country_profiles (
    country_code CHAR(2) PRIMARY KEY,
    country_name VARCHAR(200) NOT NULL,
    region VARCHAR(100),
    income_group VARCHAR(50),
    eudr_commodity_producer BOOLEAN DEFAULT FALSE,
    eudr_commodities JSONB DEFAULT '[]',
    latest_cpi_score INTEGER,
    latest_cpi_year INTEGER,
    latest_wgi_composite NUMERIC(5,2),
    latest_bribery_composite NUMERIC(5,2),
    latest_institutional_score NUMERIC(5,2),
    latest_trend_direction VARCHAR(20),
    latest_dd_recommendation VARCHAR(20),
    ec_classification VARCHAR(20),
    is_hotspot BOOLEAN DEFAULT FALSE,
    active_alert_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Sector Risk Scores
CREATE TABLE eudr_corruption_index_monitor.sector_risk_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    sector VARCHAR(20) NOT NULL,
    risk_dimension VARCHAR(50) NOT NULL,
    score NUMERIC(5,2) NOT NULL,
    evidence_source VARCHAR(200),
    assessment_year INTEGER NOT NULL,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Governance Indicators (supplementary indicators beyond CPI/WGI)
CREATE TABLE eudr_corruption_index_monitor.governance_indicators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    indicator_name VARCHAR(200) NOT NULL,
    indicator_source VARCHAR(200) NOT NULL,
    indicator_value NUMERIC(10,4),
    indicator_unit VARCHAR(50),
    year INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Audit Log (hypertable)
CREATE TABLE eudr_corruption_index_monitor.audit_log (
    log_id UUID DEFAULT gen_random_uuid(),
    operation VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100),
    operator_id VARCHAR(100),
    details JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (log_id, created_at)
);

SELECT create_hypertable('eudr_corruption_index_monitor.audit_log', 'created_at');

-- Continuous Aggregates
CREATE MATERIALIZED VIEW eudr_corruption_index_monitor.quarterly_cpi_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('90 days', ingested_at) AS quarter,
    country_code,
    AVG(score) AS avg_score,
    MIN(score) AS min_score,
    MAX(score) AS max_score,
    COUNT(*) AS data_points
FROM eudr_corruption_index_monitor.cpi_scores
GROUP BY quarter, country_code;

CREATE MATERIALIZED VIEW eudr_corruption_index_monitor.annual_wgi_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('365 days', ingested_at) AS year_bucket,
    country_code,
    dimension,
    AVG(estimate) AS avg_estimate,
    AVG(percentile_rank) AS avg_percentile,
    COUNT(*) AS data_points
FROM eudr_corruption_index_monitor.wgi_indicators
GROUP BY year_bucket, country_code, dimension;

-- Indexes
CREATE INDEX idx_cpi_country_year ON eudr_corruption_index_monitor.cpi_scores(country_code, year);
CREATE INDEX idx_cpi_tier ON eudr_corruption_index_monitor.cpi_scores(tier);
CREATE INDEX idx_cpi_score ON eudr_corruption_index_monitor.cpi_scores(score);
CREATE INDEX idx_wgi_country_year ON eudr_corruption_index_monitor.wgi_indicators(country_code, year);
CREATE INDEX idx_wgi_dimension ON eudr_corruption_index_monitor.wgi_indicators(dimension);
CREATE INDEX idx_wgi_percentile ON eudr_corruption_index_monitor.wgi_indicators(percentile_rank);
CREATE INDEX idx_bribery_country_sector ON eudr_corruption_index_monitor.bribery_risk_assessments(country_code, sector);
CREATE INDEX idx_bribery_score ON eudr_corruption_index_monitor.bribery_risk_assessments(sector_risk_score);
CREATE INDEX idx_institutional_country ON eudr_corruption_index_monitor.institutional_quality_scores(country_code);
CREATE INDEX idx_institutional_class ON eudr_corruption_index_monitor.institutional_quality_scores(quality_class);
CREATE INDEX idx_trend_country ON eudr_corruption_index_monitor.trend_analyses(country_code);
CREATE INDEX idx_trend_direction ON eudr_corruption_index_monitor.trend_analyses(trend_direction);
CREATE INDEX idx_correlation_country ON eudr_corruption_index_monitor.deforestation_correlations(country_code);
CREATE INDEX idx_correlation_hotspot ON eudr_corruption_index_monitor.deforestation_correlations(is_hotspot);
CREATE INDEX idx_alerts_country ON eudr_corruption_index_monitor.alerts(country_code);
CREATE INDEX idx_alerts_severity ON eudr_corruption_index_monitor.alerts(severity);
CREATE INDEX idx_alerts_status ON eudr_corruption_index_monitor.alerts(status);
CREATE INDEX idx_alerts_type ON eudr_corruption_index_monitor.alerts(alert_type);
CREATE INDEX idx_compliance_country ON eudr_corruption_index_monitor.compliance_impacts(country_code);
CREATE INDEX idx_compliance_dd ON eudr_corruption_index_monitor.compliance_impacts(recommended_dd_level);
CREATE INDEX idx_sector_country ON eudr_corruption_index_monitor.sector_risk_scores(country_code, sector);
CREATE INDEX idx_governance_country ON eudr_corruption_index_monitor.governance_indicators(country_code, indicator_name);
CREATE INDEX idx_profiles_region ON eudr_corruption_index_monitor.country_profiles(region);
CREATE INDEX idx_profiles_hotspot ON eudr_corruption_index_monitor.country_profiles(is_hotspot);
CREATE INDEX idx_profiles_dd ON eudr_corruption_index_monitor.country_profiles(latest_dd_recommendation);
```

### 7.5 API Endpoints (30+)

| Method | Path | Description |
|--------|------|-------------|
| **CPI Monitoring** | | |
| GET | `/api/v1/eudr-cim/cpi/countries` | List CPI scores for all countries (latest year, with filters) |
| GET | `/api/v1/eudr-cim/cpi/countries/{country_code}` | Get CPI history for a specific country |
| GET | `/api/v1/eudr-cim/cpi/rankings` | Get global CPI rankings with EUDR annotations |
| GET | `/api/v1/eudr-cim/cpi/regional/{region}` | Get CPI scores for a specific region |
| POST | `/api/v1/eudr-cim/cpi/ingest` | Ingest new CPI dataset (admin only) |
| **WGI Analysis** | | |
| GET | `/api/v1/eudr-cim/wgi/countries/{country_code}` | Get all 6 WGI dimensions for a country |
| GET | `/api/v1/eudr-cim/wgi/dimensions/{dimension}` | Get specific WGI dimension across all countries |
| GET | `/api/v1/eudr-cim/wgi/composite/{country_code}` | Get composite governance score for a country |
| GET | `/api/v1/eudr-cim/wgi/comparison` | Compare WGI across multiple countries (query params) |
| POST | `/api/v1/eudr-cim/wgi/ingest` | Ingest new WGI dataset (admin only) |
| **Bribery Risk** | | |
| GET | `/api/v1/eudr-cim/bribery/countries/{country_code}` | Get sector bribery risk for a country |
| GET | `/api/v1/eudr-cim/bribery/sectors/{sector}` | Get bribery risk by sector across countries |
| GET | `/api/v1/eudr-cim/bribery/matrix` | Get full bribery risk matrix (countries x sectors) |
| POST | `/api/v1/eudr-cim/bribery/assess` | Trigger bribery risk assessment for a country |
| **Institutional Quality** | | |
| GET | `/api/v1/eudr-cim/institutional/countries/{country_code}` | Get institutional quality score |
| GET | `/api/v1/eudr-cim/institutional/rankings` | Get institutional quality rankings |
| GET | `/api/v1/eudr-cim/institutional/weaknesses` | Get countries with critical institutional weaknesses |
| POST | `/api/v1/eudr-cim/institutional/assess` | Trigger institutional quality assessment |
| **Trend Analysis** | | |
| GET | `/api/v1/eudr-cim/trends/countries/{country_code}` | Get trend analysis for a country |
| GET | `/api/v1/eudr-cim/trends/deteriorating` | List countries with deteriorating governance |
| GET | `/api/v1/eudr-cim/trends/boundary-proximity` | List countries approaching risk boundaries |
| GET | `/api/v1/eudr-cim/trends/predictions/{country_code}` | Get trajectory predictions for a country |
| **Deforestation Correlation** | | |
| GET | `/api/v1/eudr-cim/correlation/countries/{country_code}` | Get corruption-deforestation correlation |
| GET | `/api/v1/eudr-cim/correlation/hotspots` | List corruption-deforestation hotspots |
| GET | `/api/v1/eudr-cim/correlation/regression/{country_code}` | Get regression model details |
| POST | `/api/v1/eudr-cim/correlation/compute` | Trigger correlation computation |
| **Alert Management** | | |
| GET | `/api/v1/eudr-cim/alerts` | List alerts (with filters: country, severity, status, type) |
| GET | `/api/v1/eudr-cim/alerts/{alert_id}` | Get alert details |
| PUT | `/api/v1/eudr-cim/alerts/{alert_id}/acknowledge` | Acknowledge an alert |
| PUT | `/api/v1/eudr-cim/alerts/{alert_id}/dismiss` | Dismiss an alert with reason |
| GET | `/api/v1/eudr-cim/alerts/subscriptions` | Get alert subscriptions for current operator |
| PUT | `/api/v1/eudr-cim/alerts/subscriptions` | Update alert subscriptions |
| **Compliance Impact** | | |
| GET | `/api/v1/eudr-cim/compliance/countries/{country_code}` | Get compliance impact assessment |
| GET | `/api/v1/eudr-cim/compliance/dd-recommendations` | Get DD level recommendations for all sourcing countries |
| GET | `/api/v1/eudr-cim/compliance/dds-data/{country_code}` | Get DDS-formatted governance risk data |
| POST | `/api/v1/eudr-cim/compliance/scenario` | Run what-if scenario analysis |
| GET | `/api/v1/eudr-cim/compliance/briefing` | Get quarterly compliance briefing |
| **Country Profiles** | | |
| GET | `/api/v1/eudr-cim/profiles/{country_code}` | Get comprehensive country governance profile |
| GET | `/api/v1/eudr-cim/profiles` | List all country profiles (with filters) |
| **Health** | | |
| GET | `/api/v1/eudr-cim/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_cim_cpi_ingestions_total` | Counter | CPI dataset ingestion operations |
| 2 | `gl_eudr_cim_wgi_ingestions_total` | Counter | WGI dataset ingestion operations |
| 3 | `gl_eudr_cim_bribery_assessments_total` | Counter | Bribery risk assessments performed |
| 4 | `gl_eudr_cim_institutional_assessments_total` | Counter | Institutional quality assessments performed |
| 5 | `gl_eudr_cim_trend_analyses_total` | Counter | Trend analyses performed |
| 6 | `gl_eudr_cim_correlation_computations_total` | Counter | Deforestation correlation computations |
| 7 | `gl_eudr_cim_alerts_generated_total` | Counter | Alerts generated by type and severity |
| 8 | `gl_eudr_cim_alerts_acknowledged_total` | Counter | Alerts acknowledged |
| 9 | `gl_eudr_cim_compliance_assessments_total` | Counter | Compliance impact assessments |
| 10 | `gl_eudr_cim_dd_level_changes_total` | Counter | Due diligence level change recommendations |
| 11 | `gl_eudr_cim_processing_duration_seconds` | Histogram | Processing operation latency by engine type |
| 12 | `gl_eudr_cim_api_request_duration_seconds` | Histogram | API endpoint latency |
| 13 | `gl_eudr_cim_errors_total` | Counter | Errors by engine and error type |
| 14 | `gl_eudr_cim_countries_monitored` | Gauge | Number of countries actively monitored |
| 15 | `gl_eudr_cim_active_alerts` | Gauge | Number of unacknowledged alerts |
| 16 | `gl_eudr_cim_data_freshness_days` | Gauge | Days since last CPI/WGI data ingestion |
| 17 | `gl_eudr_cim_hotspot_countries` | Gauge | Number of corruption-deforestation hotspot countries |
| 18 | `gl_eudr_cim_enhanced_dd_countries` | Gauge | Number of countries with enhanced DD recommendation |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Statistical Analysis | SciPy + Statsmodels + scikit-learn | Regression, trend tests, correlation analysis |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for CPI/WGI |
| Cache | Redis | Query caching, country profile caching |
| Object Storage | S3 | Raw dataset storage (CPI CSV, WGI CSV), report exports |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible models |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access to governance data |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |
| Data Ingestion | pandas + httpx | CSV/Excel parsing, API data fetching |

### 7.8 RBAC Permissions (SEC-002 Integration)

The following permissions will be registered in the GreenLang PERMISSION_MAP for RBAC enforcement:

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-cim:cpi:read` | View CPI scores and rankings | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cim:cpi:ingest` | Ingest new CPI datasets | Data Admin, Admin |
| `eudr-cim:wgi:read` | View WGI indicators and composites | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cim:wgi:ingest` | Ingest new WGI datasets | Data Admin, Admin |
| `eudr-cim:bribery:read` | View bribery risk assessments | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cim:bribery:execute` | Trigger bribery risk assessment | Analyst, Compliance Officer, Admin |
| `eudr-cim:institutional:read` | View institutional quality scores | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cim:institutional:execute` | Trigger institutional quality assessment | Analyst, Compliance Officer, Admin |
| `eudr-cim:trends:read` | View trend analysis results | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cim:correlation:read` | View deforestation correlation data | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cim:correlation:execute` | Trigger correlation computation | Analyst, Compliance Officer, Admin |
| `eudr-cim:alerts:read` | View governance alerts | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cim:alerts:manage` | Acknowledge/dismiss alerts, manage subscriptions | Compliance Officer, Admin |
| `eudr-cim:compliance:read` | View compliance impact assessments | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cim:compliance:execute` | Trigger compliance assessment and scenarios | Analyst, Compliance Officer, Admin |
| `eudr-cim:compliance:dds` | Access DDS-formatted governance data | Compliance Officer, Admin |
| `eudr-cim:profiles:read` | View comprehensive country profiles | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cim:audit:read` | View audit log | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Source | Integration | Data Flow |
|--------|-------------|-----------|
| Transparency International CPI | Annual CSV/Excel ingestion + API | CPI scores, ranks, metadata -> cpi_scores table |
| World Bank WGI | Annual CSV/API ingestion from Databank | WGI 6 dimensions -> wgi_indicators table |
| TRACE Bribery Risk Matrix | Annual data ingestion | Bribery risk scores -> bribery_risk_assessments |
| AGENT-DATA-007 Deforestation Satellite | API integration | Deforestation rates per country -> correlation engine |
| Global Forest Watch | API integration | Tree cover loss data -> correlation engine |
| EC Country Benchmarking | Publication monitoring + manual ingestion | Official classifications -> country_profiles |
| World Justice Project | Annual data ingestion | Rule of Law Index -> institutional_quality_scores |
| FAO / WRI | Periodic data ingestion | Forest governance indicators -> governance_indicators |
| FAOSTAT | API integration | Commodity production data -> correlation engine |
| World Bank Development Indicators | API integration | GDP per capita -> correlation engine |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-001 Supply Chain Mapping Master | API integration | Country governance risk scores -> node-level risk scores in supply chain graph |
| GL-EUDR-APP v1.0 Platform | API integration | Governance dashboards, alert display, DDS data |
| GL-EUDR-APP DDS Reporting Engine | API integration | DDS-formatted governance risk section |
| Other EUDR agents (risk assessment) | API integration | Country governance risk as input to composite risk scoring |
| External Auditors | Read-only API + exports | Auditable governance assessment data with provenance |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Country Governance Dashboard Review (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Governance Risk" module
3. Dashboard shows overview map with countries color-coded by governance risk
4. Sees summary cards: X countries monitored, Y active alerts, Z countries requiring enhanced DD
5. Clicks on sourcing country (e.g., Ghana)
6. Country profile loads with:
   - CPI score (43/100, Rank 72, Tier: MODERATE) with 10-year trend line
   - WGI radar chart showing all 6 dimensions
   - Sector bribery risk bars (Forestry: 62, Customs: 58, Agriculture: 55, Mining: 68)
   - Institutional quality score (51/100, Class: WEAK)
   - Trend classification: STABLE (slope: +0.3 pts/year, p=0.12)
   - Corruption-deforestation correlation: r = -0.67 (p < 0.01)
   - Current DD recommendation: STANDARD
7. Reviews and acknowledges any active alerts for Ghana
8. Exports governance profile as PDF for audit documentation
```

#### Flow 2: Alert Response Workflow (Compliance Officer)

```
1. Alert notification received: "CRITICAL: Cameroon CPI dropped 6 points (from 27 to 21)"
2. Compliance officer opens alert in GL-EUDR-APP
3. Alert detail shows:
   - Trigger: CPI score change exceeded threshold (6 > 3 points)
   - Previous CPI: 27 (Tier: CORRUPT), New CPI: 21 (Tier: HIGHLY CORRUPT)
   - WGI Control of Corruption: 8th percentile (corroborating deterioration)
   - Compliance impact: "Recommend upgrading Cameroon from STANDARD to ENHANCED due diligence"
   - Recommended actions:
     a. Increase verification frequency for Cameroon supply chains
     b. Obtain independent third-party audit for Cameroon suppliers
     c. Conduct field verification of geolocation data for Cameroon plots
     d. Review batch documentation for Cameroon-origin commodities
4. Officer acknowledges alert and selects "Action Taken"
5. System triggers supply chain risk re-propagation for all graphs with Cameroon nodes
6. Officer generates updated DDS risk assessment section for Cameroon shipments
```

#### Flow 3: Trend Analysis and Scenario Planning (Risk Analyst)

```
1. Risk analyst opens "Trend Analysis" view
2. Selects portfolio of 15 sourcing countries
3. Dashboard shows trend matrix: countries x trend direction (improving/stable/deteriorating)
4. Analyst identifies 3 countries with deteriorating governance:
   - Myanmar: CPI declining 2.1 pts/year (p < 0.01)
   - Honduras: CPI declining 1.5 pts/year (p = 0.03)
   - Cameroon: CPI declining 1.8 pts/year (p = 0.02)
5. Analyst clicks on Myanmar to see detailed trend analysis
6. Sees change point: 2021 (post-coup, governance collapse)
7. Sees prediction: CPI projected to reach 18 in 2 years (currently 23)
8. Runs what-if scenario: "What if Myanmar CPI drops to 18?"
9. System shows compliance impact:
   - DD level: ENHANCED (already enhanced, no change)
   - 12 supply chains affected
   - Estimated additional DD cost: EUR 45,000/year
   - Recommended action: Consider alternative sourcing
10. Analyst exports scenario report for management briefing
```

#### Flow 4: Audit Verification (External Auditor)

```
1. Auditor accesses governance data via read-only API/portal
2. Reviews operator's DDS governance risk section for Indonesia
3. Verifies CPI score matches Transparency International published data (provenance hash)
4. Verifies WGI dimensions match World Bank published data (provenance hash)
5. Reviews bribery risk assessment methodology and scores
6. Confirms DD level determination (Enhanced) aligns with governance indicators
7. Reviews trend analysis statistical methodology (Mann-Kendall test, p-values)
8. Confirms all governance assessments are within 48 hours of latest published data
9. Signs off on governance risk section of audit report
```

### 8.2 Key Screen Descriptions

**Governance Risk Map (Dashboard Overview):**
- Full-screen world map with countries color-coded by composite governance risk
  - Dark green: Low governance risk (CPI >= 70, strong institutions)
  - Light green: Moderate-low risk (CPI 55-69)
  - Yellow: Moderate risk (CPI 40-54)
  - Orange: High risk (CPI 25-39)
  - Red: Very high risk (CPI < 25)
- Country pins for EUDR sourcing countries with additional detail on hover
- Summary cards at top: Total countries, Active alerts (by severity), Enhanced DD countries, Hotspots
- Left sidebar: Filter by region, commodity, DD level, trend direction, alert status
- Quick search: Jump to specific country profile

**Country Governance Profile:**
- Header: Country name, flag, region, income group, EUDR commodities produced
- Summary bar: CPI (gauge), WGI Composite (gauge), Institutional Quality (gauge), DD Recommendation (badge)
- CPI History: Line chart (10+ years), with confidence bands, tier boundaries, and trend line
- WGI Radar: 6-dimension radar chart with current year and 5-year comparison
- Sector Bribery: Horizontal bar chart (4 sectors), with EUDR vulnerability annotations
- Institutional Quality: Stacked bar chart (4 pillars) with quality class badge
- Trend Summary: Direction arrow, slope, p-value, change points, prediction cone
- Correlation: Scatter plot (CPI vs. deforestation rate) with regression line
- Active Alerts: List with severity badges, timestamps, and action buttons
- Export: PDF, CSV, API (JSON)

**Alert Management Console:**
- Alert feed: chronological list of alerts with severity badges and country flags
- Filter bar: severity, country, alert type, status, date range
- Alert detail panel: trigger details, compliance impact, recommended actions, action buttons
- Subscription management: configurable triggers, thresholds, and delivery channels per country

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 8 P0 features (Features 1-8) implemented and tested
  - [ ] Feature 1: CPI Monitoring Engine -- 180+ countries, historical data, tier classification, percentile ranking
  - [ ] Feature 2: WGI Analyzer Engine -- 6 dimensions, 200+ countries, composite scoring, radar data
  - [ ] Feature 3: Bribery Risk Engine -- 4 EUDR sectors, sector risk scoring, vulnerability mapping
  - [ ] Feature 4: Institutional Quality Engine -- 4 pillars, composite scoring, weakness identification
  - [ ] Feature 5: Trend Analysis Engine -- regression, Mann-Kendall, change points, predictions, boundary proximity
  - [ ] Feature 6: Deforestation Correlation Engine -- Pearson/Spearman, regression, hotspot detection
  - [ ] Feature 7: Alert Engine -- configurable triggers, severity levels, delivery channels, acknowledgement workflow
  - [ ] Feature 8: Compliance Impact Engine -- DD level recommendation, DDS data, scenario analysis, supply chain integration
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC integrated with 18 permissions)
- [ ] Performance targets met (< 2 seconds country assessment p99)
- [ ] All scoring verified deterministic (bit-perfect reproducibility)
- [ ] CPI data validated against official TI publications for 5 benchmark countries
- [ ] WGI data validated against official World Bank publications for 5 benchmark countries
- [ ] Correlation models validated with p < 0.05 significance for 10+ countries
- [ ] API documentation complete (OpenAPI spec with 30+ endpoints documented)
- [ ] Database migration V107 tested and validated
- [ ] Integration with AGENT-EUDR-001 verified (risk re-propagation trigger)
- [ ] Integration with GL-EUDR-APP DDS Reporting Engine verified
- [ ] 5 beta customers successfully reviewed governance dashboards for their sourcing countries
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 100+ country profiles actively viewed by customers
- 50+ alerts generated and processed
- Average alert acknowledgement time < 24 hours
- Data freshness: CPI/WGI within 48 hours of publication
- p99 API latency < 200ms in production
- Zero critical bugs

**60 Days:**
- 200+ operators using governance dashboards
- 80%+ of alerts acknowledged within 48 hours
- 20+ countries with DD level adjustments based on governance data
- Governance risk data included in 50+ DDS submissions
- Correlation models available for 30+ EUDR-relevant countries
- < 3 support tickets per customer per month

**90 Days:**
- 300+ operators actively using corruption monitoring
- Average governance research time reduced by 95% (validated by customer survey)
- Zero EUDR audit findings related to inadequate governance risk assessment for active customers
- 50+ what-if scenario analyses run by customers
- NPS > 50 from compliance officer persona
- 10+ countries with proactive DD adjustments based on trend predictions

---

## 10. Timeline and Milestones

### Phase 1: Core Data Engines (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | CPI Monitor Engine (Feature 1): CPI ingestion, history, tier classification, percentile ranking | Backend Engineer |
| 2-3 | WGI Analyzer Engine (Feature 2): WGI 6-dimension ingestion, composite scoring, comparison | Backend Engineer |
| 3-4 | Bribery Risk Engine (Feature 3): sector assessment, TRACE integration, vulnerability mapping | Backend Engineer + Domain Expert |
| 4-6 | Institutional Quality Engine (Feature 4): 4-pillar scoring, weakness identification, quality classification | Backend Engineer + Domain Expert |

**Milestone: Core governance data engines operational with data ingestion (Week 6)**

### Phase 2: Analytical Engines (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Trend Analysis Engine (Feature 5): regression, Mann-Kendall, change points, predictions, boundary proximity | Backend Engineer + Data Scientist |
| 8-10 | Deforestation Correlation Engine (Feature 6): statistical correlation, regression models, hotspot detection | Backend Engineer + Data Scientist |

**Milestone: Full analytical capability operational with trend and correlation intelligence (Week 10)**

### Phase 3: Action and Integration Layer (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | Alert Engine (Feature 7): trigger configuration, severity classification, delivery channels, acknowledgement | Backend Engineer |
| 12-14 | Compliance Impact Engine (Feature 8): DD recommendations, DDS data, scenarios, AGENT-EUDR-001 integration | Senior Backend Engineer |
| 13-14 | REST API Layer: 30+ endpoints, authentication, rate limiting, pagination | Backend Engineer |

**Milestone: Full alerting and compliance impact capability operational (Week 14)**

### Phase 4: Testing, Integration, and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 800+ tests, golden tests for 20 benchmark countries | Test Engineer |
| 16-17 | Performance testing, security audit, load testing, data validation against official sources | DevOps + Security |
| 17 | Database migration V107 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers), GL-EUDR-APP dashboard integration | Product + Engineering |
| 18 | Launch readiness review (all 8 P0 features verified) and go-live | All |

**Milestone: Production launch with all 8 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Sub-national governance assessment (Feature 9)
- Political risk and regime change monitoring (Feature 10)
- Peer benchmarking and comparative analytics (Feature 11)
- Machine learning governance prediction models
- Additional governance data source integrations (V-Dem, BTI, CPIA)

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Integration points defined; API available |
| AGENT-DATA-007 Deforestation Satellite Connector | BUILT (100%) | Low | Deforestation data available via API |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Dashboard integration points defined |
| GL-EUDR-APP DDS Reporting Engine | BUILT (100%) | Low | DDS data format defined |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| Redis Caching | Production Ready | Low | Standard caching layer |
| Email Notification Service | Production Ready | Low | Used for alert delivery |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| Transparency International CPI | Published annually (Jan/Feb) | Low | Well-established; stable format; multiple years cached locally |
| World Bank WGI | Published annually (Sept/Oct) | Low | Well-established; stable API; historical data cached |
| TRACE Bribery Risk Matrix | Published annually | Medium | Data licensing required; fallback to CPI-derived sector estimates |
| EC Country Benchmarking List | Published periodically (Art. 29(3)) | Medium | Initial list delayed; agent operates with own governance assessment until EC publishes |
| Global Forest Watch | API available | Low | Multi-provider fallback; cached data |
| FAOSTAT | API available | Low | Standard UN data; stable API |
| World Justice Project | Published annually | Medium | Limited to 140 countries; supplement with WGI Rule of Law for others |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | TI CPI methodology changes affecting year-over-year comparability | Low | Medium | Flag methodology change years; compute adjusted scores; maintain comparability notes |
| R2 | EC country benchmarking criteria differ significantly from agent's governance assessment | Medium | High | Agent's assessment is evidence-based and transparent; discrepancies flagged for manual review; adapt weights when EC criteria are published |
| R3 | TRACE Bribery Risk Matrix data licensing costs exceed budget | Medium | Medium | Develop fallback CPI-derived sector risk model; negotiate academic/startup pricing with TRACE |
| R4 | Insufficient deforestation data for meaningful corruption correlation in some countries | Medium | Medium | Require minimum 10 data points; flag low-confidence correlations; use regional proxy where country data insufficient |
| R5 | Political sensitivity of corruption scoring (country officials object) | Low | Low | Use only published, authoritative data sources; no proprietary judgment in scoring; transparent methodology |
| R6 | Data staleness between annual CPI/WGI publications | Medium | Medium | Supplement with interim governance indicators; use trend models for inter-publication estimates; flag data age prominently |
| R7 | Operators misinterpret governance scores as definitive compliance determination | Medium | High | Clear documentation that agent provides risk intelligence, not regulatory determination; only EC benchmarking is legally binding |
| R8 | Integration complexity with AGENT-EUDR-001 risk re-propagation | Medium | Medium | Well-defined API contract; circuit breaker pattern; async notification with retry |
| R9 | Statistical models generate misleading predictions for volatile countries | Medium | Medium | Confidence intervals on all predictions; "VOLATILE" classification for unstable countries; minimum data requirements |
| R10 | EUDR Article 29 implementation delayed or modified | Medium | Medium | Agent's governance intelligence is valuable regardless of Article 29 specifics; modular design adapts to regulatory changes |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| CPI Monitor Engine Tests | 120+ | CPI ingestion, scoring, tier classification, percentile ranking, YoY change, regional analysis |
| WGI Analyzer Engine Tests | 100+ | WGI 6-dimension ingestion, composite scoring, comparison, radar data generation |
| Bribery Risk Engine Tests | 80+ | Sector assessment, TRACE integration, vulnerability mapping, composite scoring |
| Institutional Quality Engine Tests | 80+ | 4-pillar scoring, quality classification, weakness identification, source validation |
| Trend Analysis Engine Tests | 100+ | Linear regression, Mann-Kendall, change points, predictions, boundary proximity, momentum |
| Deforestation Correlation Tests | 80+ | Pearson/Spearman, regression models, lag analysis, hotspot detection, causal narratives |
| Alert Engine Tests | 70+ | All trigger types, severity classification, delivery, acknowledgement, cooldown, subscriptions |
| Compliance Impact Engine Tests | 80+ | DD level determination, DDS data generation, scenario analysis, supply chain integration |
| API Tests | 90+ | All 30+ endpoints, auth, error handling, pagination, query filters |
| Golden Tests | 40+ | 20 benchmark countries x 2 scenarios (stable governance + governance crisis) |
| Integration Tests | 30+ | Cross-agent integration with EUDR-001, DATA-007, GL-EUDR-APP |
| Performance Tests | 20+ | Single-country assessment, batch processing, correlation computation, concurrent queries |
| Data Validation Tests | 30+ | CPI/WGI values validated against official publications for benchmark countries |
| **Total** | **920+** | |

### 13.2 Golden Test Countries

The following 20 countries represent key EUDR sourcing countries with diverse governance profiles:

| Country | CPI Range | Governance Profile | Test Scenario |
|---------|-----------|-------------------|---------------|
| **Brazil** | 35-40 | Moderate corruption; strong institutions in some areas; high deforestation | Stable with sector variation |
| **Indonesia** | 32-38 | Moderate-high corruption; weak forest governance; high deforestation | Deteriorating trend |
| **Ghana** | 41-45 | Moderate corruption; mixed institutions; cocoa focus | Slowly improving |
| **Ivory Coast** | 35-40 | Moderate-high corruption; post-conflict recovery; cocoa focus | Recovering trajectory |
| **Cameroon** | 25-28 | High corruption; weak institutions; timber and cocoa | Governance crisis |
| **DRC** | 18-22 | Very high corruption; failed institutions; multiple commodities | Very high risk stable |
| **Myanmar** | 22-28 | Very high corruption; post-coup collapse | Rapid deterioration |
| **Malaysia** | 47-53 | Moderate corruption; mixed enforcement; palm oil | Moderate stable |
| **Colombia** | 37-40 | Moderate corruption; conflict legacy; cattle and coffee | Improving trend |
| **Paraguay** | 28-32 | High corruption; weak enforcement; soya and cattle | High risk stable |
| **Papua New Guinea** | 28-32 | High corruption; weak institutions; palm oil and timber | High risk volatile |
| **Honduras** | 24-28 | High corruption; weak rule of law; coffee | Deteriorating |
| **Ethiopia** | 37-40 | Moderate corruption; political instability; coffee | Volatile |
| **Vietnam** | 33-38 | Moderate corruption; one-party state; rubber and coffee | Slowly improving |
| **Thailand** | 35-38 | Moderate corruption; political instability; rubber | Volatile |
| **Peru** | 33-38 | Moderate corruption; mineral extraction; coffee | Mixed trend |
| **Norway** | 84-87 | Very low corruption; strong institutions | Low risk benchmark |
| **Finland** | 85-88 | Very low corruption; strong institutions | Low risk benchmark |
| **Sweden** | 82-85 | Very low corruption; strong institutions | Low risk benchmark |
| **Nigeria** | 24-27 | High corruption; weak enforcement; cocoa and rubber | High risk stable |

Each golden test country will validate:
1. CPI score accuracy against TI published data
2. WGI 6-dimension accuracy against World Bank published data
3. Sector bribery risk scoring correctness
4. Institutional quality scoring correctness
5. Trend classification correctness (against known historical trajectory)
6. Alert generation correctness (simulated CPI/WGI changes)
7. DD level recommendation correctness
8. Provenance hash integrity

Total: 20 countries x 8 validation points = 160 golden test assertions

### 13.3 Reproducibility Testing

All scoring engines will be tested for bit-perfect reproducibility:

1. **Deterministic Input -> Deterministic Output**: Same CPI/WGI inputs always produce identical scores, classifications, and recommendations
2. **Cross-Platform Reproducibility**: Results identical on local dev (Windows), CI (Linux), and production (Linux/K8s)
3. **Decimal Precision**: All floating-point computations use Python `Decimal` for scoring; statistical library outputs rounded to documented precision
4. **Provenance Verification**: SHA-256 hashes recalculated and verified for every stored assessment

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **CPI** | Corruption Perceptions Index -- Transparency International's annual country-level corruption ranking (0-100; 0 = highly corrupt, 100 = very clean) |
| **WGI** | Worldwide Governance Indicators -- World Bank's annual governance dataset with 6 dimensions |
| **TRACE BRM** | TRACE Bribery Risk Matrix -- commercial bribery risk assessment dataset by TRACE International |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **DD Level** | Due Diligence Level -- simplified, standard, or enhanced due diligence based on country risk |
| **VA** | Voice and Accountability -- WGI dimension measuring citizen participation and media freedom |
| **PV** | Political Stability and Absence of Violence -- WGI dimension measuring political stability |
| **GE** | Government Effectiveness -- WGI dimension measuring public service quality |
| **RQ** | Regulatory Quality -- WGI dimension measuring regulation quality |
| **RL** | Rule of Law -- WGI dimension measuring contract enforcement and property rights |
| **CC** | Control of Corruption -- WGI dimension measuring corruption control |
| **Mann-Kendall** | Non-parametric statistical test for monotonic trends in time series data |
| **CUSUM** | Cumulative Sum -- statistical method for change point detection in time series |
| **Hotspot** | Country with both high corruption AND accelerating deforestation |
| **Governance Momentum** | Composite indicator combining CPI trend, WGI trends, and institutional quality trend direction |
| **Boundary Proximity** | Measure of how close a country's governance score is to an EUDR risk reclassification threshold |

### Appendix B: CPI Scoring Methodology

Transparency International's CPI aggregates data from multiple independent sources:
- Each source assesses overall corruption in the public sector
- Sources include surveys of business people, expert assessments by country analysts, and public opinion surveys
- CPI score: 0 (highly corrupt) to 100 (very clean)
- A country must have at least 3 sources to be included
- Standard error and confidence intervals are published for each score
- Comparability: scores are comparable across years from 2012 onwards (methodology standardized)

The agent classifies CPI scores into EUDR-relevant tiers:
- **Very Clean (80-100)**: Countries with strongest anti-corruption frameworks (e.g., Denmark, Finland, New Zealand)
- **Clean (60-79)**: Countries with good governance but some corruption concerns (e.g., France, USA, Japan)
- **Moderate (40-59)**: Countries with notable corruption; mixed governance (e.g., Brazil, India, South Africa)
- **Corrupt (20-39)**: Countries with serious corruption; weak institutions (e.g., Myanmar, Honduras, Cameroon)
- **Highly Corrupt (0-19)**: Countries with pervasive corruption; failed institutions (e.g., Somalia, South Sudan, Syria)

### Appendix C: WGI Dimension Descriptions

| Dimension | Code | Description | EUDR Relevance |
|-----------|------|-------------|----------------|
| Voice and Accountability | VA | Perceptions of citizen participation in government selection, freedom of expression, freedom of association, free media | Low direct; contextual indicator |
| Political Stability | PV | Perceptions of likelihood of political instability and/or politically-motivated violence, including terrorism | Low direct; stability affects forest governance continuity |
| Government Effectiveness | GE | Perceptions of quality of public services, quality of civil service, quality of policy formulation and implementation, credibility of government commitment | High; weak government effectiveness = weak forest management |
| Regulatory Quality | RQ | Perceptions of ability of government to formulate and implement sound policies and regulations that permit and promote private sector development | High; regulatory quality determines forestry regulation effectiveness |
| Rule of Law | RL | Perceptions of extent to which agents have confidence in and abide by the rules of society, quality of contract enforcement, property rights, police, courts, crime and violence | Very High; rule of law directly determines forestry law enforcement |
| Control of Corruption | CC | Perceptions of extent to which public power is exercised for private gain, including petty and grand forms of corruption, "capture" of the state by elites and private interests | Very High; direct measure of corruption in public sector |

### Appendix D: EUDR Article 29 Benchmarking Criteria

Under Article 29, the European Commission classifies countries based on:

1. **Rate of deforestation and forest degradation** (Art. 29(2)(a)) -- Measured by satellite data and national forest inventories
2. **Rate of expansion of agricultural land for relevant commodities** (Art. 29(2)(b)) -- Commodity production area growth
3. **Production trends of relevant commodities and products** (Art. 29(2)(c)) -- Volume trends indicating expansion pressure
4. **Information provided by indigenous peoples, local communities, and civil society** (Art. 29(2)(d)) -- Qualitative governance input
5. **Existence of legislation to counter deforestation and its enforcement** (Art. 29(2)(e)) -- Legal framework quality AND enforcement capacity
6. **Country's engagement in anti-corruption and anti-deforestation efforts** (Art. 29(2)(f)) -- Active governance improvement trajectory

The agent's 8 engines map to these criteria as follows:
- CPI Monitor (Feature 1) -> Criteria 5, 6 (corruption levels indicate enforcement capacity)
- WGI Analyzer (Feature 2) -> Criteria 5, 6 (Rule of Law, Government Effectiveness indicate legislation enforcement)
- Bribery Risk (Feature 3) -> Criteria 5 (sector-specific enforcement weakness)
- Institutional Quality (Feature 4) -> Criteria 5 (comprehensive enforcement and governance assessment)
- Trend Analysis (Feature 5) -> Criteria 6 (trajectory of governance engagement)
- Deforestation Correlation (Feature 6) -> Criteria 1, 5 (linking governance to deforestation outcomes)
- Alert Engine (Feature 7) -> All criteria (timely notification of changes)
- Compliance Impact (Feature 8) -> All criteria (holistic Article 29 mapping)

### Appendix E: Composite Governance Risk Score Formula

The agent computes a composite governance risk score (0-100, where 100 = highest risk) using the following deterministic formula:

```
Governance_Risk_Score = 100 - (
    W_cpi * Normalized_CPI +
    W_wgi * Normalized_WGI_Composite +
    W_bribery * (100 - Bribery_Risk_Composite) +
    W_institutional * Institutional_Quality_Score +
    W_trend * Trend_Adjustment
)

Where:
- W_cpi = 0.30 (configurable)
- W_wgi = 0.25 (configurable)
- W_bribery = 0.20 (configurable)
- W_institutional = 0.15 (configurable)
- W_trend = 0.10 (configurable)

- Normalized_CPI = CPI score (already 0-100)
- Normalized_WGI_Composite = Weighted average of 6 WGI percentile ranks (0-100)
- Bribery_Risk_Composite = Weighted average of 4 sector bribery risk scores (0-100)
- Institutional_Quality_Score = Composite institutional quality (0-100)
- Trend_Adjustment:
  - Improving: +5 points (reduces risk by 5)
  - Stable: 0 points
  - Deteriorating: -10 points (increases risk by 10)
  - Volatile: -5 points (increases risk by 5)

DD Level Determination:
- Governance_Risk_Score < 25 -> SIMPLIFIED due diligence
- Governance_Risk_Score 25-65 -> STANDARD due diligence
- Governance_Risk_Score > 65 -> ENHANCED due diligence
```

All weights, thresholds, and adjustments are stored in configuration and are adjustable per operator without code changes.

### Appendix F: Data Source Update Schedule

| Data Source | Update Frequency | Typical Publication Month | Agent Ingestion Target |
|-------------|-----------------|--------------------------|----------------------|
| CPI | Annual | January-February | Within 48 hours of publication |
| WGI | Annual | September-October | Within 48 hours of publication |
| TRACE BRM | Annual | Variable (typically Q2) | Within 1 week of publication |
| EC Benchmarking | Periodic (no fixed schedule) | Variable | Within 24 hours of publication |
| Global Forest Watch (deforestation) | Annual (main dataset) + weekly alerts | Variable | Weekly alert ingestion; annual dataset within 1 week |
| World Justice Project | Annual | October-November | Within 1 week of publication |
| IIAG (Africa) | Annual | Variable | Within 1 week of publication |

### Appendix G: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 -- EU Deforestation Regulation
2. Transparency International -- Corruption Perceptions Index Methodology (2024 edition)
3. World Bank -- Worldwide Governance Indicators Methodology (Kaufmann, Kraay, Mastruzzi)
4. TRACE International -- Bribery Risk Matrix Methodology
5. World Resources Institute -- Forest Governance Assessment Framework
6. FAO -- Global Forest Resources Assessment (FRA) 2020
7. Global Forest Watch -- Technical Documentation for Tree Cover Loss Dataset
8. World Justice Project -- Rule of Law Index Methodology (2024 edition)
9. Mo Ibrahim Foundation -- Ibrahim Index of African Governance Methodology
10. Transparency International (2023) -- "Corruption and Deforestation: The Hidden Link"
11. World Bank (2022) -- "Governance and Forest Outcomes: Evidence from WGI Data"
12. Chatham House (2021) -- "Illegal Logging and Related Trade: Indicators of the Global Response"
13. INTERPOL (2020) -- "Forestry Crime: Targeting the Trillion Dollar Trade"
14. FAO (2022) -- "Forest Governance: The Key to Sustainable Forest Management"
15. OECD Anti-Bribery Convention Implementation Reports
16. UNCAC Implementation Review Mechanism Reports

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-09 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| Governance Domain Expert | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-09 | GL-ProductManager | Initial draft created |
| 1.0.0 | 2026-03-09 | GL-ProductManager | Finalized: all 8 P0 features confirmed, regulatory coverage verified (Articles 2/10/11/12/23/29/31), 8 engines defined (CPIMonitor/WGIAnalyzer/BriberyRisk/InstitutionalQuality/TrendAnalysis/DeforestationCorrelation/Alert/ComplianceImpact), module path aligned with GreenLang conventions, V107 migration defined (12 tables + 3 hypertables + 2 continuous aggregates), 30+ API endpoints, 18 Prometheus metrics, 18 RBAC permissions, 920+ tests target, approval granted |
