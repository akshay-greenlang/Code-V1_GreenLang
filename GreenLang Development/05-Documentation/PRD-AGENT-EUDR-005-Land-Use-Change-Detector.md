# PRD: AGENT-EUDR-005 -- Land Use Change Detector

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-005 |
| **Agent ID** | GL-EUDR-LUC-005 |
| **Component** | Land Use Change Detector Agent |
| **Category** | EUDR Regulatory Agent -- Land Use Intelligence |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-07 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-07 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) defines "deforestation" as the conversion of forest to agricultural use (Article 2(1)) and "forest degradation" as structural changes to forest canopy cover from primary/naturally regenerating forest to plantation forest or other wooded land (Article 2(5)). Central to EUDR compliance is the ability to prove that the land where a commodity was produced has not undergone deforestation or forest degradation since the cutoff date of December 31, 2020 (Article 2(3)).

While AGENT-EUDR-003 (Satellite Monitoring) detects spectral changes and AGENT-EUDR-004 (Forest Cover Analysis) measures forest canopy density, neither agent performs **land use classification** or **land use transition analysis** -- the definitive determination of *what the land was used for before and after the cutoff date*. This is a critical gap because:

- **Deforestation is defined as land use change, not just canopy loss**: A forest with temporary canopy loss from natural disturbance (fire, storm) that regenerates is NOT deforestation. A forest converted to cropland IS deforestation, even if some trees remain.
- **No temporal land use trajectory analysis**: Existing agents detect point-in-time forest status but cannot determine whether a plot underwent gradual conversion (progressive clearing over 3 years) or abrupt change (overnight clearing). Regulatory auditors require evidence of the change trajectory.
- **No land use classification at the cutoff date**: To prove "deforestation-free", one must demonstrate that the land was classified as agricultural (not forest) BEFORE December 31, 2020, or that it was forest BOTH before and after the cutoff. Neither scenario can be verified without land use classification.
- **No agricultural conversion detection**: EUDR specifically targets conversion of forest to agricultural use. Detecting *what the forest was converted to* (cropland, pasture, plantation) requires a land use classifier, not just a forest/non-forest binary.
- **No urban encroachment monitoring**: Forests near expanding urban areas face conversion pressure. Monitoring urban encroachment helps identify at-risk plots before conversion occurs.
- **No transition matrix analysis**: Understanding the full pattern of land use changes across a supply chain's source region (how much forest became cropland vs. how much cropland became urban) requires transition matrix analysis that no current agent provides.

Without a dedicated Land Use Change Detector, EU operators cannot definitively prove that their sourcing plots are "deforestation-free" as defined by Article 2(3), exposing them to penalties of up to 4% of annual EU turnover, goods confiscation, and exclusion from public procurement.

### 1.2 Solution Overview

Agent-EUDR-005: Land Use Change Detector is a specialized agent that classifies land use types, detects transitions between them over time, verifies compliance with the EUDR cutoff date, and provides definitive evidence-based verdicts on whether land use change constitutes deforestation or forest degradation under the regulation. It operates as a multi-temporal remote sensing analysis engine that combines spectral classification with temporal trajectory analysis.

The agent bridges the gap between AGENT-EUDR-003 (which detects spectral anomalies) and AGENT-EUDR-004 (which measures forest canopy) by providing the authoritative **land use classification** and **transition determination** layer that is required for EUDR Article 2(1) deforestation determination.

Core capabilities:

1. **Multi-class land use classification** -- Classifies land into 10 IPCC-aligned land use categories (forest, cropland, grassland, wetland, settlement, other land, and sub-categories) using multi-spectral satellite imagery with ensemble classification methods.
2. **Temporal transition detection** -- Detects land use transitions between any two time periods, identifying exactly when and how land use changed (e.g., "tropical moist forest -> oil palm plantation, detected March 2022").
3. **Change trajectory analysis** -- Analyzes the temporal progression of land use change: abrupt (clearcut), gradual (progressive thinning), oscillating (seasonal agriculture), or stable (no change). Distinguishes permanent conversion from temporary disturbance.
4. **EUDR cutoff date verification** -- Compares land use classification at the cutoff date (December 31, 2020) against current classification to determine if deforestation or forest degradation occurred per Article 2(1) and 2(5).
5. **Agricultural conversion detection** -- Specifically identifies conversion from forest to agricultural use (the regulatory definition of deforestation), distinguishing cropland expansion, pasture establishment, and plantation development.
6. **Conversion risk assessment** -- Scores each plot's risk of future land use conversion based on proximity to deforestation frontiers, agricultural expansion trends, infrastructure development, and demographic pressure.
7. **Urban encroachment analysis** -- Monitors urban and infrastructure expansion near forested production areas, identifying plots at risk of indirect land use change.
8. **Compliance reporting** -- Generates regulatory-grade evidence packages with before/after land use maps, transition matrices, temporal trajectories, and definitive EUDR compliance verdicts.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Land use classification accuracy | >= 85% overall accuracy (kappa >= 0.80) | Confusion matrix against ground truth validation set |
| Transition detection precision | >= 90% for forest-to-agriculture transitions | Precision/recall against manually verified transitions |
| Cutoff date compliance accuracy | 100% deterministic, reproducible verdicts | Bit-perfect reproducibility tests |
| Classification categories | 10 IPCC land use classes + sub-categories | Category coverage matrix |
| Temporal trajectory analysis | Distinguish 4+ trajectory types | Trajectory type classification accuracy |
| Processing throughput | 1,000+ plots classified per minute | Benchmark under load |
| API response time | < 200ms p95 for single-plot classification | API latency monitoring |
| EUDR commodity coverage | All 7 regulated commodities | Commodity-specific conversion detection verified |
| Evidence package completeness | 100% of reports include before/after maps + trajectory | Automated report validation |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with land use change verification being a mandatory component of every Due Diligence Statement. Estimated market for land use verification technology: 1-2 billion EUR.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers requiring definitive land use classification and transition evidence for EUDR compliance, estimated at 400M-600M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1 as part of the integrated EUDR compliance platform, representing 25M-40M EUR in land use analysis module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) requiring definitive proof of deforestation-free status for all sourcing plots
- Multinational food and beverage companies needing land use transition evidence for cocoa, coffee, palm oil, and soya supply chains
- Timber and paper industry operators requiring forest-to-forest continuity verification
- Commodity certification bodies (FSC, RSPO, Rainforest Alliance) needing independent land use verification

**Secondary:**
- Government regulators and customs authorities verifying operator DDS claims
- Environmental NGOs monitoring deforestation commitments
- Insurance companies assessing land use change risk for agricultural portfolios
- Financial institutions evaluating EUDR compliance risk in lending portfolios
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual verification / field visits | Ground truth accuracy | Extremely expensive; not scalable; slow (weeks per plot) | Automated, scalable, instant classification with evidence |
| Generic remote sensing platforms (Google Earth Engine, Descartes Labs) | Powerful tools; global coverage | Require GIS expertise; not EUDR-specific; no compliance verdicts | Purpose-built for EUDR Art. 2; automatic compliance verdicts |
| Niche deforestation tools (Global Forest Watch, Planet NICFI) | Free/low cost; established datasets | Forest/non-forest binary only; no land use classification; no transition matrix | 10-class classification; full transition analysis; trajectory detection |
| Certification body tools (RSPO satellite monitoring, FSC Satellite) | Commodity-specific; established workflows | Single-commodity; limited to certified areas; no regulatory export | All 7 commodities; full supply chain integration; DDS-ready output |
| In-house GIS teams | Customized to organization | Expensive; slow; no standardization; no regulatory formatting | Standardized; faster; regulatory-grade output; integrated with EUDR platform |

### 2.4 Differentiation Strategy

1. **EUDR-native land use classification** -- Not generic remote sensing; every classification maps directly to EUDR Article 2 definitions.
2. **Temporal trajectory intelligence** -- Goes beyond point-in-time comparison to analyze the full temporal trajectory of change, distinguishing permanent conversion from temporary disturbance.
3. **Integration depth** -- Pre-built integration with AGENT-EUDR-003 (satellite imagery), AGENT-EUDR-004 (forest cover), AGENT-EUDR-002 (geolocation), and the GL-EUDR-APP platform.
4. **Zero-hallucination verdicts** -- All classification and transition detection is deterministic, auditable, and reproducible with no LLM in the critical path.
5. **Evidence-grade reporting** -- Output is structured for direct inclusion in DDS submissions and regulatory audits.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to definitively prove "deforestation-free" status per EUDR Art. 2(3) | 100% of customer plots with land use classification evidence | Q2 2026 |
| BG-2 | Reduce land use verification cost from field visits ($500+/plot) to automated analysis ($0.50/plot) | 99% cost reduction | Q2 2026 |
| BG-3 | Provide regulatory-grade evidence accepted by EU competent authorities | Zero DDS rejections due to insufficient land use evidence | Q3 2026 |
| BG-4 | Identify at-risk plots before conversion occurs (proactive compliance) | 80% of conversion events predicted 6+ months in advance | Q4 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Multi-class land use classification | Classify land into 10 IPCC categories using multi-temporal spectral analysis |
| PG-2 | Transition detection | Detect and characterize land use transitions between any two dates |
| PG-3 | Trajectory analysis | Determine whether change is abrupt, gradual, oscillating, or stable |
| PG-4 | Cutoff compliance | Verify land use status at Dec 31, 2020 vs. current for EUDR compliance |
| PG-5 | Agricultural conversion | Specifically detect forest-to-agriculture conversion (EUDR deforestation definition) |
| PG-6 | Conversion risk scoring | Score each plot's risk of future land use conversion |
| PG-7 | Evidence reporting | Generate regulatory-grade evidence packages for DDS submission |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Classification accuracy | >= 85% overall accuracy, kappa >= 0.80 |
| TG-2 | Processing throughput | 1,000+ plots per minute for single-date classification |
| TG-3 | Temporal depth | Analyze land use from 2018 to present (3 years pre-cutoff) |
| TG-4 | API response time | < 200ms p95 for single-plot classification |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, sourcing cocoa from 12 countries |
| **EUDR Pressure** | Must demonstrate that ALL cocoa sourcing plots are deforestation-free since Dec 31, 2020 |
| **Pain Points** | Has satellite monitoring data showing "no change" but cannot prove the land was forest before 2020; lacks formal land use classification evidence; field verification too expensive at scale |
| **Goals** | Definitive land use classification at cutoff date and present; evidence packages for DDS; automated monitoring for any future conversion |
| **Technical Skill** | Moderate -- comfortable with web applications but not remote sensing |

### Persona 2: Supply Chain Analyst -- Lukas (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Supply Chain Analyst at an EU palm oil importer |
| **Company** | 800 employees, sourcing palm oil from Indonesia and Malaysia |
| **EUDR Pressure** | Must verify that plantation plots were NOT forest before Dec 31, 2020 |
| **Pain Points** | Many plantations were established on former forest land; needs to determine exact conversion date; cannot distinguish oil palm plantation from natural forest using simple NDVI |
| **Goals** | Temporal trajectory showing when each plot was converted; transition matrix for entire sourcing region; risk assessment for plots near deforestation frontiers |
| **Technical Skill** | High -- comfortable with data tools, APIs, and GIS basics |

### Persona 3: External Auditor -- Dr. Hofmann (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must verify operator claims about land use status with independent evidence |
| **Pain Points** | Operators provide inconsistent land use evidence; no standardized format for land use change proof; difficult to verify temporal claims without multi-date analysis |
| **Goals** | Access independent land use classification with provenance; verify transition evidence against satellite record; validate temporal trajectory claims |
| **Technical Skill** | Moderate -- comfortable with audit software and GIS reports |

### Persona 4: Procurement Manager -- Ana (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Procurement Director at a rubber importer |
| **Company** | 3,000 employees, sourcing from Southeast Asia |
| **EUDR Pressure** | Must screen new suppliers for land use conversion risk |
| **Pain Points** | Cannot assess whether potential supplier plots have deforestation history; no way to predict which plots are at risk of future conversion |
| **Goals** | Pre-screening tool for new supplier plots; conversion risk scores for portfolio management; early warning for at-risk plots |
| **Technical Skill** | Low-moderate -- uses procurement platforms |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 2(1)** | "Deforestation" = conversion of forest to agricultural use, whether human-induced or not | LandUseClassifier + TransitionDetector: classify pre/post land use; detect forest-to-agriculture transitions |
| **Art. 2(2)** | "Forest" = land spanning > 0.5ha with trees > 5m and canopy > 10%, or trees able to reach those thresholds in situ (FAO definition) | Integration with AGENT-EUDR-004 CanopyDensityMapper + CanopyHeightModeler for forest definition checks |
| **Art. 2(3)** | "Deforestation-free" = produced on land not subject to deforestation after December 31, 2020 | CutoffDateVerifier: compare land use at cutoff vs. current; issue definitive verdict |
| **Art. 2(4)** | Agricultural use excludes "tree plantations used for the production of wood" and forestry use | LandUseClassifier: distinguish timber plantations (NOT deforestation) from agricultural plantations (IS deforestation for EUDR) |
| **Art. 2(5)** | "Forest degradation" = structural changes from primary/naturally regenerating forest to plantation forest or other wooded land | TransitionDetector: detect forest type degradation (natural forest -> plantation) |
| **Art. 3** | Prohibition on non-compliant products | CutoffDateVerifier: flag non-compliant plots; ComplianceReporter: generate evidence for prohibition enforcement |
| **Art. 4(2)** | Due diligence -- collect information on production land | LandUseClassifier: provide land use classification as part of due diligence data |
| **Art. 9(1)** | Geolocation of all plots of land | Integration with AGENT-EUDR-002 for plot geolocation; spatial analysis of land use at plot coordinates |
| **Art. 10(1-2)** | Risk assessment for non-compliance | ConversionRiskAssessor: score each plot's risk of having undergone or undergoing deforestation |
| **Art. 10(2)(a)** | Complexity of the relevant supply chain | Regional transition matrix analysis: understand land use change patterns across sourcing regions |
| **Art. 10(2)(e)** | Concerns about the country of production | Country-level land use change rates integrated into risk assessment |
| **Art. 29** | Country benchmarking (Low/Standard/High risk) | Country risk factors from land use change rates inform ConversionRiskAssessor |
| **Art. 31** | Record keeping for 5 years | All classifications, transitions, and verdicts stored with 5-year retention in TimescaleDB |

### 5.2 IPCC Land Use Categories

The agent uses the IPCC Guidelines for National Greenhouse Gas Inventories (2006, refined 2019) land use classification system, aligned with EUDR regulatory definitions:

| IPCC Category | Sub-categories | EUDR Relevance |
|---------------|---------------|----------------|
| **Forest Land** | Tropical moist, tropical dry, temperate, boreal, mangrove | Primary category -- conversion FROM forest = deforestation |
| **Cropland** | Annual crops, perennial crops, agroforestry, fallow | Primary target -- conversion TO cropland from forest = deforestation |
| **Grassland** | Managed pasture, natural grassland, savanna | Conversion target -- forest to pasture = deforestation for cattle |
| **Wetland** | Peatland, swamp, floodplain | Protected in many jurisdictions; conversion is high-risk |
| **Settlement** | Urban, peri-urban, infrastructure, mining | Urban encroachment near forests signals conversion pressure |
| **Other Land** | Bare soil, rock, ice, sand | Transitional state during land clearing |
| **Plantation Forest** | Timber plantation, pulpwood, managed forest | Art. 2(4): NOT agricultural use; conversion FROM natural forest = degradation |
| **Oil Palm Plantation** | Established, young, mature | Art. 2(4): IS agricultural use; conversion FROM forest = deforestation |
| **Rubber Plantation** | Established, young, mature | Art. 2(4): IS agricultural use; conversion FROM forest = deforestation |
| **Water Body** | River, lake, reservoir, ocean | Reference class for validation |

### 5.3 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | CutoffDateVerifier baseline date for all transition analysis |
| June 29, 2023 | Regulation entered into force | Legal basis for all compliance checks |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Land use classification evidence required in DDS submissions |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; agent must handle increased classification volume |
| Ongoing | Land use monitoring | ContinuousMonitor: ongoing classification updates for active plots |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 8 features below are P0 launch blockers. Features 1-5 form the core land use intelligence engine; Features 6-8 form the risk assessment and reporting layer.

**P0 Features 1-5: Core Land Use Intelligence Engine**

---

#### Feature 1: Multi-Class Land Use Classifier (LandUseClassifier)

**User Story:**
```
As a compliance officer,
I want to know the precise land use classification of every sourcing plot,
So that I can determine if the land was forest or agricultural use at the EUDR cutoff date.
```

**Acceptance Criteria:**
- [ ] Classifies land into 10 IPCC-aligned categories with sub-categories (see Section 5.2)
- [ ] Uses 5 classification methods: spectral signature, vegetation indices (NDVI/EVI/NDMI/SAVI), temporal phenology, texture analysis, and ensemble voting
- [ ] Supports multi-temporal classification: classify land use at any date from 2018 to present
- [ ] Handles 7 EUDR commodity-specific plantations (oil palm, rubber, cocoa, coffee, soya, cattle pasture, timber) as distinct classes
- [ ] Achieves >= 85% overall accuracy and kappa >= 0.80 against validation dataset
- [ ] Provides per-class confidence scores (0-100) for each classification result
- [ ] Distinguishes Article 2(4) exclusions: timber plantation (NOT agricultural) vs. commodity plantation (IS agricultural)
- [ ] Handles cloud-contaminated imagery using temporal compositing and gap-filling
- [ ] Supports batch classification of 10,000+ plots in a single job
- [ ] All classification is deterministic (same input -> same output)

**Non-Functional Requirements:**
- Performance: Single-plot classification < 500ms; batch of 1,000 < 60 seconds
- Accuracy: Per-class precision and recall reported; confusion matrix generated
- Reproducibility: Deterministic classification with configurable random seed

**Dependencies:**
- AGENT-EUDR-003 SpectralIndexCalculator for NDVI/EVI/NDMI/SAVI values
- AGENT-EUDR-004 CanopyDensityMapper for canopy density inputs
- AGENT-EUDR-004 ForestTypeClassifier for forest sub-type information

**Estimated Effort:** 3 weeks (1 senior ML/remote sensing engineer)

---

#### Feature 2: Temporal Transition Detector (TransitionDetector)

**User Story:**
```
As a supply chain analyst,
I want to detect all land use transitions that occurred on a plot between any two dates,
So that I can determine if and when deforestation or forest degradation occurred.
```

**Acceptance Criteria:**
- [ ] Detects transitions between any pair of the 10 land use classes
- [ ] Generates transition matrix showing from-class to to-class with area and percentage
- [ ] Identifies transition date range with monthly granularity (e.g., "between March 2021 and May 2021")
- [ ] Classifies transition type: deforestation (forest -> agriculture), degradation (natural forest -> plantation), reforestation (non-forest -> forest), urbanization (any -> settlement), and 6 other types
- [ ] Detects "no significant transition" when classification remains stable
- [ ] Handles gradual transitions spanning multiple observation periods
- [ ] Produces transition evidence with before/after spectral comparison
- [ ] Achieves >= 90% precision for forest-to-agriculture transition detection
- [ ] Supports multi-period analysis: compare land use at N time points (not just two)
- [ ] All transition detection is deterministic

**Non-Functional Requirements:**
- Performance: Single-plot transition analysis < 2 seconds (including both classifications)
- Accuracy: False positive rate < 10% for deforestation transitions
- Auditability: SHA-256 provenance hash on every transition result

**Dependencies:**
- Feature 1 (LandUseClassifier) for pre/post classification
- AGENT-EUDR-003 BaselineManager for Dec 2020 baseline data
- AGENT-EUDR-003 ForestChangeDetector for complementary change detection

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

#### Feature 3: Temporal Trajectory Analyzer (TemporalTrajectoryAnalyzer)

**User Story:**
```
As a compliance officer,
I want to understand the temporal trajectory of land use change on each plot,
So that I can distinguish permanent conversion (deforestation) from temporary disturbance (fire recovery).
```

**Acceptance Criteria:**
- [ ] Analyzes time series of land use classifications from 2018 to present
- [ ] Identifies 5 trajectory types: stable, abrupt change, gradual change, oscillating, recovery
- [ ] For "abrupt change": pinpoints the change date with monthly accuracy
- [ ] For "gradual change": estimates the start and end dates of the transition period
- [ ] For "oscillating": identifies the cycle period (e.g., annual crop rotation)
- [ ] For "recovery": detects post-disturbance regeneration and estimates recovery completeness
- [ ] Provides trajectory confidence score (0-100)
- [ ] Generates trajectory visualization data (time series chart coordinates)
- [ ] Distinguishes natural disturbance (fire, storm, drought) from anthropogenic conversion using spatial patterns
- [ ] All trajectory analysis is deterministic

**Non-Functional Requirements:**
- Performance: Single-plot trajectory analysis < 3 seconds
- Temporal depth: Minimum 3 years of data (2018-2020) before cutoff for baseline
- Auditability: Every trajectory determination includes evidence chain

**Dependencies:**
- Feature 1 (LandUseClassifier) for multi-date classification
- AGENT-EUDR-003 CloudGapFiller for temporal gap filling
- AGENT-EUDR-004 HistoricalReconstructor for historical baseline

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 4: EUDR Cutoff Date Verifier (CutoffDateVerifier)

**User Story:**
```
As a compliance officer,
I want a definitive verdict on whether each sourcing plot experienced deforestation after December 31, 2020,
So that I can include this determination in my Due Diligence Statement with full confidence.
```

**Acceptance Criteria:**
- [ ] Issues one of 5 verdicts: COMPLIANT (no deforestation), NON_COMPLIANT (deforestation detected), DEGRADED (forest degradation detected), INCONCLUSIVE (insufficient data), PRE_EXISTING_AGRICULTURE (was already agriculture before cutoff)
- [ ] Classifies land use at cutoff date (December 31, 2020) using best available imagery
- [ ] Classifies current land use (most recent clear imagery)
- [ ] Compares cutoff vs. current classification to determine if transition occurred
- [ ] Applies EUDR deforestation definition: forest -> agricultural use (Article 2(1))
- [ ] Applies EUDR degradation definition: primary/regenerating forest -> plantation/wooded land (Article 2(5))
- [ ] Correctly handles Article 2(4) exclusions: timber plantation -> timber plantation is NOT deforestation
- [ ] Provides evidence package: cutoff classification, current classification, transition analysis, trajectory, confidence
- [ ] Conservative approach: when evidence is ambiguous, issues INCONCLUSIVE rather than false COMPLIANT
- [ ] 100% deterministic and reproducible
- [ ] Integrates with AGENT-EUDR-004 DeforestationFreeVerifier for cross-validation

**Non-Functional Requirements:**
- Accuracy: Zero false COMPLIANT verdicts (conservative bias acceptable)
- Performance: Single-plot verification < 5 seconds
- Auditability: SHA-256 provenance hash on every verdict

**Dependencies:**
- Feature 1 (LandUseClassifier) for cutoff and current classification
- Feature 2 (TransitionDetector) for transition analysis
- Feature 3 (TemporalTrajectoryAnalyzer) for trajectory context
- AGENT-EUDR-004 DeforestationFreeVerifier for cross-validation

**Estimated Effort:** 2 weeks (1 senior backend engineer)

---

#### Feature 5: Agricultural Conversion Detector (CroplandExpansionDetector)

**User Story:**
```
As a supply chain analyst,
I want to detect specifically where forest has been converted to agricultural use,
So that I can identify the exact EUDR-relevant deforestation events in my sourcing regions.
```

**Acceptance Criteria:**
- [ ] Detects 7 commodity-specific conversion types: forest-to-palm-oil, forest-to-rubber, forest-to-cocoa, forest-to-coffee, forest-to-soya, forest-to-pasture, forest-to-timber-plantation
- [ ] Distinguishes smallholder clearings (< 5 ha) from industrial-scale clearings (> 50 ha)
- [ ] Detects progressive expansion patterns (clearing front advancing over time)
- [ ] Provides expansion rate estimation (hectares/year for a given region)
- [ ] Maps conversion hotspots: spatial clustering of recent conversion events
- [ ] Identifies "leapfrog" conversion patterns (isolated clearings ahead of main front)
- [ ] Generates commodity-specific conversion probability maps
- [ ] All detection is deterministic

**Non-Functional Requirements:**
- Precision: >= 85% for commodity-specific conversion type identification
- Spatial resolution: Detect clearings as small as 0.5 hectares (FAO minimum forest area)
- Temporal resolution: Monthly detection of new conversions

**Dependencies:**
- Feature 1 (LandUseClassifier) for pre/post classification
- Feature 2 (TransitionDetector) for transition detection
- AGENT-EUDR-004 ForestTypeClassifier for forest sub-type information
- AGENT-EUDR-003 AlertGenerator for alert integration

**Estimated Effort:** 2 weeks (1 backend engineer)

---

**P0 Features 6-8: Risk Assessment and Reporting Layer**

---

#### Feature 6: Conversion Risk Assessor (ConversionRiskAssessor)

**User Story:**
```
As a procurement manager,
I want to know the risk that each sourcing plot will undergo deforestation in the future,
So that I can proactively manage my supply chain and avoid sourcing from at-risk areas.
```

**Acceptance Criteria:**
- [ ] Calculates composite conversion risk score (0-100) for each plot
- [ ] Uses 8 risk factors: proximity to deforestation frontier (20%), historical conversion rate in region (15%), road/infrastructure proximity (15%), population density trend (10%), agricultural commodity price trend (10%), protected area proximity (10%), governance index (10%), slope/accessibility (10%)
- [ ] Classifies risk into 4 tiers: LOW (0-25), MODERATE (26-50), HIGH (51-75), CRITICAL (76-100)
- [ ] Generates risk heatmap for spatial visualization
- [ ] Identifies "deforestation frontier": the leading edge of active conversion in a region
- [ ] Provides 6-month, 12-month, and 24-month conversion probability estimates
- [ ] All risk calculations are deterministic (configurable weights, no LLM)
- [ ] Updates risk scores when new satellite data or transition events are detected

**Non-Functional Requirements:**
- Performance: Risk scoring for 10,000 plots < 30 seconds
- Accuracy: Risk predictions validated against subsequent conversion events (target >= 70% precision for high-risk)
- Configurability: All risk weights adjustable per operator

**Dependencies:**
- Feature 1 (LandUseClassifier) for current land use context
- Feature 5 (CroplandExpansionDetector) for regional expansion patterns
- AGENT-EUDR-002 CoordinateValidator for plot geolocation
- AGENT-DATA-020 Climate Hazard Connector for environmental risk factors

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 7: Urban Encroachment Analyzer (UrbanEncroachmentAnalyzer)

**User Story:**
```
As a compliance officer,
I want to monitor urban and infrastructure expansion near my sourcing plots,
So that I can identify indirect land use change pressure and anticipate future conversion risk.
```

**Acceptance Criteria:**
- [ ] Detects urban/settlement expansion in the vicinity of production plots (configurable buffer: 1-50 km)
- [ ] Classifies infrastructure types: roads, buildings, mining, industrial, residential
- [ ] Calculates urban expansion rate (hectares/year) for surrounding area
- [ ] Identifies "pressure corridors": infrastructure development that typically precedes forest conversion
- [ ] Maps accessibility changes: new roads that open previously inaccessible forest areas
- [ ] Estimates time-to-conversion based on historical urban expansion patterns in similar regions
- [ ] Generates urban proximity risk factor for integration with ConversionRiskAssessor
- [ ] All analysis is deterministic

**Non-Functional Requirements:**
- Performance: Single-plot analysis < 2 seconds; batch of 1,000 < 120 seconds
- Spatial coverage: Analyze buffer zone up to 50 km radius around each plot
- Accuracy: >= 80% for urban expansion detection

**Dependencies:**
- Feature 1 (LandUseClassifier) for settlement classification
- AGENT-EUDR-002 CoordinateValidator for plot geolocation
- AGENT-EUDR-003 ImageryAcquisitionEngine for temporal satellite data

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 8: Compliance Reporter (ComplianceReporter)

**User Story:**
```
As a compliance officer,
I want to generate regulatory-grade evidence packages showing land use classification and change history,
So that I can include definitive evidence in my Due Diligence Statement and respond to auditor requests.
```

**Acceptance Criteria:**
- [ ] Generates 5 report types: single-plot land use report, regional transition report, commodity conversion report, cutoff compliance report, risk assessment report
- [ ] Exports in 4 formats: JSON (API), PDF (human-readable), CSV (data export), EUDR XML (regulatory submission)
- [ ] Includes before/after land use classification maps with color-coded legend
- [ ] Includes transition matrix visualization
- [ ] Includes temporal trajectory chart
- [ ] Includes confidence scores and data quality indicators
- [ ] Includes provenance chain (SHA-256 hashes) for all input data and intermediate results
- [ ] Auto-flags NON_COMPLIANT and DEGRADED verdicts for immediate attention
- [ ] Integrates with GL-EUDR-APP DDS Reporting Engine for DDS submission
- [ ] Generates batch reports for entire supply chain sourcing regions

**Non-Functional Requirements:**
- Performance: Single-plot report < 5 seconds; batch of 100 < 120 seconds
- Completeness: All required EUDR evidence fields populated
- Format compliance: PDF reports follow GreenLang brand template

**Dependencies:**
- Features 1-7 for all analysis results
- AGENT-EUDR-001 DDSReportingEngine for DDS integration
- GL-EUDR-APP platform for report storage and retrieval

**Estimated Effort:** 2 weeks (1 backend engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 9: Land Use Simulation and Scenario Modeling
- Model future land use under different scenarios (business-as-usual, conservation, expansion)
- Predict deforestation risk under climate change scenarios
- Simulate impact of new road construction on surrounding forest conversion

#### Feature 10: Community-Based Monitoring Integration
- Accept ground truth observations from local communities
- Integrate crowdsourced land use data for validation
- Support indigenous land rights mapping

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Real-time drone-based land use monitoring
- Carbon stock change calculation (defer to AGENT-MRV-006 Land Use Emissions Agent)
- Biodiversity impact assessment from land use change
- Predictive machine learning models trained on labeled deforestation data (defer to v2.0)
- Native mobile app for field-level land use verification

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
| AGENT-EUDR-005        |           | AGENT-EUDR-003            |           | AGENT-EUDR-004        |
| Land Use Change       |<--------->| Satellite Monitoring      |<--------->| Forest Cover          |
| Detector              |           |                           |           | Analysis              |
|                       |           | - ImageryAcquisition      |           |                       |
| - LandUseClassifier   |           | - SpectralIndexCalc       |           | - CanopyDensityMapper |
| - TransitionDetector  |           | - BaselineManager         |           | - ForestTypeClassifier|
| - TrajectoryAnalyzer  |           | - ForestChangeDetector    |           | - HistoricalRecon     |
| - CutoffDateVerifier  |           | - DataFusionEngine        |           | - DeforestFreeVerif   |
| - CroplandExpansion   |           | - CloudGapFiller          |           | - CanopyHeightModeler |
| - ConversionRiskAssess|           | - ContinuousMonitor       |           | - FragmentAnalyzer    |
| - UrbanEncroachment   |           | - AlertGenerator          |           | - BiomassEstimator    |
| - ComplianceReporter  |           |                           |           | - ComplianceReporter  |
+-----------+-----------+           +---------------------------+           +-----------------------+
            |
+-----------v-----------+           +---------------------------+
| AGENT-EUDR-002        |           | AGENT-EUDR-001            |
| Geolocation           |           | Supply Chain Mapping      |
| Verification          |           | Master                    |
|                       |           |                           |
| - CoordinateValidator |           | - DDSReportingEngine      |
| - PolygonTopology     |           | - RiskPropagation         |
| - ProtectedAreaCheck  |           | - GapAnalysis             |
+-----------------------+           +---------------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/land_use_change/
    __init__.py                          # Public API exports (~240 lines)
    config.py                            # LandUseChangeConfig with GL_EUDR_LUC_ env prefix (~700 lines)
    models.py                            # Pydantic v2 models for land use, transitions, trajectories (~1,800 lines)
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains (~700 lines)
    metrics.py                           # 18 Prometheus metrics (~600 lines)
    setup.py                             # LandUseChangeService facade (~2,700 lines)
    land_use_classifier.py               # LandUseClassifier: multi-class classification (~1,500 lines)
    transition_detector.py               # TransitionDetector: temporal transition detection (~1,400 lines)
    temporal_trajectory_analyzer.py      # TemporalTrajectoryAnalyzer: trajectory analysis (~1,300 lines)
    cutoff_date_verifier.py              # CutoffDateVerifier: EUDR cutoff compliance (~1,200 lines)
    cropland_expansion_detector.py       # CroplandExpansionDetector: agricultural conversion (~1,400 lines)
    conversion_risk_assessor.py          # ConversionRiskAssessor: risk scoring (~1,300 lines)
    urban_encroachment_analyzer.py       # UrbanEncroachmentAnalyzer: urban pressure (~1,200 lines)
    compliance_reporter.py               # ComplianceReporter: evidence reporting (~1,800 lines)
    reference_data/
        __init__.py
        land_use_parameters.py           # IPCC land use class definitions (~600 lines)
        spectral_signatures.py           # Spectral signatures for land use types (~550 lines)
        transition_rules.py              # Regulatory transition classification rules (~500 lines)
    api/
        __init__.py
        router.py                        # FastAPI router aggregating sub-routers (~200 lines)
        schemas.py                       # Request/response Pydantic schemas (~2,200 lines)
        dependencies.py                  # Auth, rate limiting, DI (~700 lines)
        classification_routes.py         # Land use classification endpoints (~600 lines)
        transition_routes.py             # Transition detection endpoints (~600 lines)
        trajectory_routes.py             # Trajectory analysis endpoints (~500 lines)
        verification_routes.py           # Cutoff verification endpoints (~650 lines)
        risk_routes.py                   # Risk assessment + urban encroachment (~550 lines)
        report_routes.py                 # Report generation endpoints (~500 lines)
```

### 7.3 Data Models (Key Entities)

```python
# Land Use Categories (IPCC-aligned)
class LandUseCategory(str, Enum):
    FOREST_LAND = "forest_land"
    CROPLAND = "cropland"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    SETTLEMENT = "settlement"
    OTHER_LAND = "other_land"
    PLANTATION_FOREST = "plantation_forest"
    OIL_PALM_PLANTATION = "oil_palm_plantation"
    RUBBER_PLANTATION = "rubber_plantation"
    WATER_BODY = "water_body"

# Transition Types
class TransitionType(str, Enum):
    DEFORESTATION = "deforestation"          # forest -> agriculture
    DEGRADATION = "degradation"              # natural forest -> plantation
    REFORESTATION = "reforestation"          # non-forest -> forest
    AGRICULTURAL_INTENSIFICATION = "agricultural_intensification"
    URBANIZATION = "urbanization"            # any -> settlement
    ABANDONMENT = "abandonment"              # agriculture -> natural regrowth
    STABLE = "stable"                        # no change
    WATER_CHANGE = "water_change"
    WETLAND_CONVERSION = "wetland_conversion"
    OTHER = "other"

# Trajectory Types
class TrajectoryType(str, Enum):
    STABLE = "stable"
    ABRUPT_CHANGE = "abrupt_change"
    GRADUAL_CHANGE = "gradual_change"
    OSCILLATING = "oscillating"
    RECOVERY = "recovery"

# EUDR Compliance Verdicts
class ComplianceVerdict(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    DEGRADED = "degraded"
    INCONCLUSIVE = "inconclusive"
    PRE_EXISTING_AGRICULTURE = "pre_existing_agriculture"
```

### 7.4 Database Schema (New Migration: V093)

10 tables, 5 hypertables, 2 continuous aggregates, 28 indexes, 5 retention policies.

### 7.5 API Endpoints (30+)

| Method | Path | Description |
|--------|------|-------------|
| **Classification** | | |
| POST | `/v1/eudr-luc/classify` | Classify land use for a single plot |
| POST | `/v1/eudr-luc/classify/batch` | Batch classify multiple plots |
| GET | `/v1/eudr-luc/classify/{plot_id}` | Get classification results |
| GET | `/v1/eudr-luc/classify/{plot_id}/history` | Classification history |
| POST | `/v1/eudr-luc/classify/compare` | Compare classifications at two dates |
| **Transitions** | | |
| POST | `/v1/eudr-luc/transitions/detect` | Detect land use transitions |
| POST | `/v1/eudr-luc/transitions/batch` | Batch transition detection |
| GET | `/v1/eudr-luc/transitions/{plot_id}` | Get transition results |
| POST | `/v1/eudr-luc/transitions/matrix` | Generate transition matrix |
| GET | `/v1/eudr-luc/transitions/types` | List transition types |
| **Trajectories** | | |
| POST | `/v1/eudr-luc/trajectory/analyze` | Analyze temporal trajectory |
| POST | `/v1/eudr-luc/trajectory/batch` | Batch trajectory analysis |
| GET | `/v1/eudr-luc/trajectory/{plot_id}` | Get trajectory results |
| **Verification** | | |
| POST | `/v1/eudr-luc/verify/cutoff` | Verify EUDR cutoff compliance |
| POST | `/v1/eudr-luc/verify/batch` | Batch cutoff verification |
| GET | `/v1/eudr-luc/verify/{plot_id}` | Get verification results |
| GET | `/v1/eudr-luc/verify/{plot_id}/evidence` | Get verification evidence |
| POST | `/v1/eudr-luc/verify/complete` | Complete verification pipeline |
| **Risk & Urban** | | |
| POST | `/v1/eudr-luc/risk/assess` | Assess conversion risk |
| POST | `/v1/eudr-luc/risk/batch` | Batch risk assessment |
| GET | `/v1/eudr-luc/risk/{plot_id}` | Get risk scores |
| POST | `/v1/eudr-luc/urban/analyze` | Analyze urban encroachment |
| POST | `/v1/eudr-luc/urban/batch` | Batch urban analysis |
| GET | `/v1/eudr-luc/urban/{plot_id}` | Get urban encroachment results |
| **Reports** | | |
| POST | `/v1/eudr-luc/reports/generate` | Generate compliance report |
| GET | `/v1/eudr-luc/reports/{report_id}` | Get report |
| GET | `/v1/eudr-luc/reports/{report_id}/download` | Download report |
| POST | `/v1/eudr-luc/reports/batch` | Batch report generation |
| **Batch Jobs** | | |
| POST | `/v1/eudr-luc/batch` | Submit batch job |
| DELETE | `/v1/eudr-luc/batch/{batch_id}` | Cancel batch job |
| **Health** | | |
| GET | `/v1/eudr-luc/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_luc_classifications_total` | Counter | Land use classifications by method and category |
| 2 | `gl_eudr_luc_transitions_detected_total` | Counter | Transitions detected by type |
| 3 | `gl_eudr_luc_trajectories_analyzed_total` | Counter | Trajectory analyses completed |
| 4 | `gl_eudr_luc_cutoff_verifications_total` | Counter | Cutoff date verifications by verdict |
| 5 | `gl_eudr_luc_conversions_detected_total` | Counter | Agricultural conversions by commodity |
| 6 | `gl_eudr_luc_risk_assessments_total` | Counter | Risk assessments by tier |
| 7 | `gl_eudr_luc_urban_analyses_total` | Counter | Urban encroachment analyses |
| 8 | `gl_eudr_luc_reports_generated_total` | Counter | Reports generated by format |
| 9 | `gl_eudr_luc_analysis_duration_seconds` | Histogram | Analysis operation latency by operation type |
| 10 | `gl_eudr_luc_classification_accuracy` | Gauge | Current classification accuracy estimate |
| 11 | `gl_eudr_luc_api_errors_total` | Counter | API errors by operation type |
| 12 | `gl_eudr_luc_batch_jobs_total` | Counter | Batch jobs submitted |
| 13 | `gl_eudr_luc_batch_jobs_active` | Gauge | Currently active batch jobs |
| 14 | `gl_eudr_luc_plots_processed_total` | Counter | Total plots processed |
| 15 | `gl_eudr_luc_deforestation_events_total` | Counter | Deforestation events detected |
| 16 | `gl_eudr_luc_data_quality_score` | Gauge | Data quality score by source |
| 17 | `gl_eudr_luc_avg_confidence_score` | Gauge | Average classification confidence |
| 18 | `gl_eudr_luc_non_compliant_plots_total` | Counter | Non-compliant plots detected |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Classification | Scikit-learn / custom spectral classifiers | Deterministic, reproducible classification |
| Spatial | Shapely + GeoJSON | Polygon operations, spatial analysis |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables |
| Cache | Redis | Classification result caching |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access control |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

18 permissions with `eudr-luc` resource prefix, mapped to 4 EUDR roles (auditor, compliance_officer, supply_chain_analyst, procurement_manager).

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| AGENT-EUDR-003 Satellite Monitoring | SpectralIndexCalculator, BaselineManager, CloudGapFiller | Spectral indices, baseline data, gap-filled imagery -> classification inputs |
| AGENT-EUDR-004 Forest Cover Analysis | CanopyDensityMapper, ForestTypeClassifier, HistoricalReconstructor | Canopy metrics, forest types, historical reconstructions -> classification context |
| AGENT-EUDR-002 Geolocation Verification | CoordinateValidator, ProtectedAreaChecker | Plot coordinates, protected area overlap -> spatial context |
| AGENT-DATA-020 Climate Hazard Connector | Environmental risk data | Climate hazards -> conversion risk factors |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-001 Supply Chain Mapping | Risk propagation | Land use verdicts -> supply chain node risk scores |
| GL-EUDR-APP v1.0 | API integration | Classification, transition, and compliance data -> frontend display and DDS generation |
| AGENT-EUDR-004 Forest Cover Analysis | Cross-validation | Land use classification -> independent verification of forest cover verdicts |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Cutoff Compliance Verification (Compliance Officer)

```
1. Compliance officer navigates to "Land Use Analysis" module in GL-EUDR-APP
2. Selects "Cutoff Compliance Verification" workflow
3. Uploads/selects list of sourcing plot IDs
4. System classifies land use at December 31, 2020 (cutoff)
5. System classifies current land use
6. System detects transitions between cutoff and present
7. System analyzes temporal trajectory for each transition
8. System issues verdicts: COMPLIANT / NON_COMPLIANT / DEGRADED / INCONCLUSIVE / PRE_EXISTING_AGRICULTURE
9. Dashboard shows: 85% COMPLIANT, 5% NON_COMPLIANT, 3% DEGRADED, 2% INCONCLUSIVE, 5% PRE_EXISTING_AGRICULTURE
10. Officer clicks NON_COMPLIANT plots -> sees before/after maps and transition evidence
11. Officer downloads evidence packages for DDS submission
```

#### Flow 2: Regional Transition Analysis (Supply Chain Analyst)

```
1. Analyst navigates to "Transition Analysis" module
2. Draws a region of interest (ROI) on the map covering sourcing area
3. System generates transition matrix for the ROI (2020 vs. 2025)
4. Matrix shows: 2,500 ha forest -> cropland, 800 ha forest -> palm oil plantation, 200 ha grassland -> settlement
5. Analyst clicks "forest -> palm oil" cell -> sees spatial map of conversion locations
6. System highlights conversion hotspots and frontier zones
7. Analyst generates regional transition report for commodity risk assessment
```

#### Flow 3: Conversion Risk Screening (Procurement Manager)

```
1. Procurement manager opens "Risk Screening" module
2. Enters 50 potential new supplier plot coordinates
3. System runs risk assessment on all 50 plots
4. Results: 35 LOW risk, 10 MODERATE risk, 3 HIGH risk, 2 CRITICAL risk
5. Manager clicks CRITICAL risk plots -> sees proximity to deforestation frontier, recent nearby conversions, road construction
6. Manager decides to exclude CRITICAL risk plots from sourcing consideration
7. Manager exports risk report for procurement decision documentation
```

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 8 P0 features (Features 1-8) implemented and tested
- [ ] >= 85% overall classification accuracy verified against validation dataset
- [ ] >= 90% precision for deforestation transition detection
- [ ] Zero false COMPLIANT verdicts (conservative bias verified)
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC integrated)
- [ ] Performance targets met (< 200ms p95 API response)
- [ ] All 7 commodity-specific conversions tested with golden test fixtures
- [ ] Cutoff compliance verification deterministic (bit-perfect reproducibility)
- [ ] API documentation complete (OpenAPI spec)
- [ ] Database migration V093 tested and validated
- [ ] Integration with AGENT-EUDR-002/003/004 verified

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:** 500+ plots classified; average confidence >= 80%; < 5 support tickets/customer
**60 Days:** 5,000+ plots classified; transition detection precision >= 90% validated; 3+ commodities analyzed
**90 Days:** 50,000+ plots classified; zero false compliant verdicts validated; NPS > 50

---

## 10. Timeline and Milestones

### Phase 1: Core Classification (Weeks 1-4)
- LandUseClassifier, TransitionDetector

### Phase 2: Temporal Analysis and Verification (Weeks 5-8)
- TemporalTrajectoryAnalyzer, CutoffDateVerifier, CroplandExpansionDetector

### Phase 3: Risk Assessment and Reporting (Weeks 9-12)
- ConversionRiskAssessor, UrbanEncroachmentAnalyzer, ComplianceReporter

### Phase 4: API, Testing, and Launch (Weeks 13-16)
- Full API layer, 500+ tests, performance testing, integration testing, launch

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk |
|------------|--------|------|
| AGENT-EUDR-003 Satellite Monitoring | BUILT (100%) | Low |
| AGENT-EUDR-004 Forest Cover Analysis | BUILT (100%) | Low |
| AGENT-EUDR-002 Geolocation Verification | BUILT (100%) | Low |
| AGENT-EUDR-001 Supply Chain Mapping | BUILT (100%) | Low |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low |
| SEC-001 JWT Authentication | BUILT (100%) | Low |
| SEC-002 RBAC Authorization | BUILT (100%) | Low |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Classification accuracy below 85% for some commodity types | Medium | High | Ensemble methods; multiple spectral indices; temporal phenology; fallback to INCONCLUSIVE |
| R2 | Cloud cover prevents cutoff date classification | Medium | Medium | Temporal compositing; gap-filling from AGENT-EUDR-003; extend search window +/- 60 days |
| R3 | Commodity-specific plantation discrimination poor for young plantations | High | Medium | Temporal phenology analysis; integration with AGENT-EUDR-004 forest type classifier |
| R4 | Gradual conversion escapes point-in-time detection | Medium | High | Trajectory analysis with monthly time steps; trend detection algorithms |
| R5 | EUDR Article 2 definitions evolve through implementing regulations | Low | Medium | Configuration-driven transition classification rules; hot-reloadable parameters |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Target Count | Description |
|----------|-------------|-------------|
| LandUseClassifier Tests | 70+ | All 10 classes, 5 methods, accuracy, batch, confidence |
| TransitionDetector Tests | 65+ | All transition types, matrix, temporal, evidence |
| TemporalTrajectoryAnalyzer Tests | 55+ | 5 trajectory types, temporal depth, confidence |
| CutoffDateVerifier Tests | 70+ | 5 verdicts, conservative bias, cross-validation |
| CroplandExpansionDetector Tests | 55+ | 7 commodity conversions, spatial patterns |
| ConversionRiskAssessor Tests | 50+ | 4 risk tiers, 8 factors, determinism |
| UrbanEncroachmentAnalyzer Tests | 45+ | Infrastructure types, expansion rates, buffers |
| Models Tests | 100+ | All enums, models, validation, serialization |
| **Total** | **500+** | |

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **IPCC** | Intergovernmental Panel on Climate Change |
| **NDVI** | Normalized Difference Vegetation Index |
| **EVI** | Enhanced Vegetation Index |
| **Transition Matrix** | Matrix showing land use change from class i to class j over a period |
| **Cutoff Date** | December 31, 2020 -- the baseline date for EUDR deforestation-free determination |
| **Deforestation** | Conversion of forest to agricultural use (EUDR Article 2(1)) |
| **Forest Degradation** | Structural changes from primary/regenerating forest to plantation/wooded land (EUDR Article 2(5)) |

### Appendix B: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council
2. IPCC Guidelines for National Greenhouse Gas Inventories (2006, Refined 2019) -- Volume 4: AFOLU
3. FAO Forest Resources Assessment -- Forest Definition
4. IPCC Good Practice Guidance for Land Use, Land-Use Change and Forestry
5. Copernicus Global Land Service -- Land Cover Classification
6. Hansen et al. (2013) -- High-Resolution Global Maps of 21st-Century Forest Cover Change

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-07 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-03-07 | GL-ProductManager | Initial PRD created following EUDR-001 format standard |
