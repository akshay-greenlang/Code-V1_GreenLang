# PRD-PACK-046: Intensity Metrics Pack

**Pack ID:** PACK-046-intensity-metrics
**Category:** GHG Accounting Packs
**Tier:** Enterprise
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-26
**Prerequisite:** PACK-041 (Scope 1-2 Complete) recommended; enhanced with PACK-042/043 (Scope 3 Starter/Complete), PACK-044 (Inventory Management), PACK-045 (Base Year Management)

---

## 1. Executive Summary

### 1.1 Problem Statement

Intensity metrics -- emissions normalised by a business activity denominator such as revenue, headcount, floor area, or production output -- are the primary lens through which investors, regulators, and management evaluate carbon efficiency. The EU Corporate Sustainability Reporting Directive (CSRD) via ESRS E1-6, the US SEC Climate Disclosure Rules, CDP Climate Change, SBTi, and ISO 14064-1:2018 all require or strongly encourage intensity metric reporting. Despite their centrality, organisations face persistent challenges:

1. **Denominator selection complexity**: Organisations must choose from dozens of potential denominators (revenue, FTE, m2 GLA, tonnes produced, MWh generated, bed-days, tonne-km, etc.) that vary by sector, scope, framework, and audience. Selecting the wrong denominator can misrepresent carbon efficiency -- for example, revenue-based intensity falls when prices rise even if physical efficiency is unchanged. No systematic methodology exists to guide denominator selection across multiple frameworks simultaneously.

2. **Multi-framework intensity alignment**: ESRS E1-6 requires "GHG intensity per net revenue" as a mandatory metric, CDP asks for "Scope 1+2 per unit revenue" and sector-specific physical intensities, SBTi mandates sector-specific physical intensity pathways (SDA), SEC requires "GHG intensity metrics if used internally", and ISO 14064-1 requires "GHG intensity ratios". Each framework defines intensity differently -- some include Scope 3, some exclude market-based Scope 2, some require physical denominators only. Maintaining consistent yet framework-specific intensity calculations manually is error-prone and audit-risky.

3. **Decomposition analysis gaps**: When total emissions change, management needs to understand how much is due to growth (activity effect), structural shifts in portfolio mix (structure effect), and genuine efficiency improvement (intensity effect). Kaya identity and LMDI (Logarithmic Mean Divisia Index) decomposition are the standard methodologies, but most organisations report only headline intensity ratios without decomposition, losing the crucial distinction between efficiency gains and growth effects.

4. **Benchmarking inconsistency**: Comparing intensity metrics against sector peers requires standardised denominators, consistent Scope boundaries, comparable reporting periods, and appropriate peer group selection. Published benchmarks from CDP, MSCI, Sustainalytics, ISS, and TPI use different methodologies, scopes, and data vintage, making direct comparison misleading without normalisation.

5. **Target-setting methodology deficiencies**: SBTi Sectoral Decarbonisation Approach (SDA) requires sector-specific intensity pathways with convergence to 2050 targets. Organisations must calculate their current intensity position relative to the sector pathway, determine the required annual intensity reduction rate, and project forward. Most organisations use simple linear interpolation rather than the correct SDA convergence methodology, leading to incorrect target-setting and potential SBTi rejection.

6. **Temporal inconsistency**: Intensity metrics calculated across multiple years must use consistent denominators, scope boundaries, and emission factor sets. When denominator data is restated (e.g., revenue restatement, FTE methodology change), all historical intensity metrics must be recalculated. Without systematic time-series management, historical intensity trends become unreliable and non-comparable.

7. **Data quality opaqueness**: Intensity metrics compound errors from both numerator (emissions) and denominator (activity data), yet uncertainty is rarely quantified. A tCO2e/MEUR metric with +/-15% uncertainty on emissions and +/-5% uncertainty on revenue has a combined uncertainty of +/-16% -- material for investment decisions and regulatory compliance but almost never disclosed.

8. **Sector-specific methodology gaps**: Different sectors require fundamentally different intensity approaches: real estate uses CRREM pathways and GRESB benchmarks, transport uses GLEC framework and tCO2e/tonne-km, power generation uses gCO2/kWh and PCAF Scope 3 intensity, financial institutions use financed emissions intensity (tCO2e/MEUR invested per PCAF), agriculture uses tCO2e/tonne product and tCO2e/hectare. Generic intensity tools fail to capture these sector nuances.

### 1.2 Solution Overview

PACK-046 is the **Intensity Metrics Pack** -- the sixth pack in the "GHG Accounting Packs" category, complementing PACK-041 (Scope 1-2 Complete), PACK-042 (Scope 3 Starter), PACK-043 (Scope 3 Complete), PACK-044 (Inventory Management), and PACK-045 (Base Year Management). While existing packs calculate raw emissions and manage inventory governance, PACK-046 focuses exclusively on the intensity dimension: normalising emissions by activity, decomposing drivers, benchmarking performance, setting science-based intensity targets, projecting scenarios, and generating framework-compliant intensity disclosures.

The pack provides systematic denominator selection guidance for 25+ standard denominators across 12 sectors, automated intensity calculation with configurable scope inclusion and emission factor methodology, LMDI decomposition of emissions changes into activity, structure, and intensity effects, peer benchmarking against published sector intensity data from CDP, TPI, GRESB, and CRREM, SBTi Sectoral Decarbonisation Approach (SDA) intensity target calculation and tracking, scenario analysis modelling efficiency improvements and growth projections, data quality scoring and uncertainty propagation for intensity metrics, and multi-framework disclosure mapping covering ESRS E1-6, CDP, SEC, SBTi, ISO 14064-1, and TCFD.

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the complete intensity metrics lifecycle from denominator selection through verified disclosure.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet Approach | PACK-046 Intensity Metrics Pack |
|-----------|-------------------------------|----------------------------------|
| Denominator selection | Ad hoc choice, often revenue-only | Systematic 25+ denominator library with framework alignment scoring |
| Multi-framework compliance | Manual mapping per framework | Automated ESRS E1-6 + CDP + SEC + SBTi + ISO + TCFD mapping |
| Decomposition analysis | Rarely performed | Full LMDI decomposition (activity, structure, intensity effects) |
| Peer benchmarking | Manual lookup, inconsistent scopes | Automated normalised benchmarking with peer group management |
| Target calculation | Simple linear interpolation | SBTi SDA convergence methodology with sector pathways |
| Temporal consistency | Manual recalculation on restatement | Automated cascade recalculation with base year alignment |
| Uncertainty quantification | Never disclosed | Combined numerator/denominator uncertainty propagation |
| Sector specificity | Generic revenue intensity only | 12 sector-specific methodologies (CRREM, GLEC, PCAF, etc.) |
| Audit trail | Spreadsheet with manual notes | SHA-256 provenance, full calculation lineage, digital audit trail |
| Processing time | 2-5 days per reporting period | <30 minutes for full intensity analytics pipeline |

### 1.4 Intensity Metric Definition

An intensity metric normalises absolute emissions by a relevant activity denominator:

```
Intensity = Total_Emissions_tCO2e / Activity_Denominator
```

| Component | Description | Examples |
|-----------|-------------|----------|
| Numerator | GHG emissions in tCO2e | Scope 1, Scope 2 (location/market), Scope 1+2, Scope 1+2+3, specific scope/category |
| Denominator | Business activity metric | Revenue (MEUR), FTE, m2 GLA, MWh generated, tonnes produced, tonne-km, beds, customers |
| Result | Intensity ratio | tCO2e/MEUR, tCO2e/FTE, kgCO2e/m2, gCO2/kWh, tCO2e/tonne-km |

### 1.5 Standard Denominator Library (25+)

| # | Denominator | Unit | Applicable Sectors | Primary Framework |
|---|-------------|------|-------------------|-------------------|
| 1 | Revenue | MEUR | All sectors | ESRS E1-6 (mandatory), CDP, SEC |
| 2 | Full-Time Equivalents | FTE | Services, office-based | CDP, GRI |
| 3 | Gross Leasable Area | m2 GLA | Real estate, commercial buildings | GRESB, CRREM |
| 4 | Production output | tonnes | Manufacturing, mining, cement, steel | SBTi SDA, CDP |
| 5 | Electricity generated | MWh | Power generation | SBTi SDA (gCO2/kWh), CDP, TPI |
| 6 | Vehicle-kilometres | vkm | Transport | GLEC, CDP |
| 7 | Tonne-kilometres | tkm | Freight transport | GLEC, SBTi SDA |
| 8 | Passenger-kilometres | pkm | Passenger transport | SBTi SDA, CDP |
| 9 | Bed-days | bed-day | Healthcare, hospitality | Sector-specific |
| 10 | Floor area | m2 | All buildings | ISO 50001, ASHRAE |
| 11 | Units produced | units | Discrete manufacturing | ISO 14064-1 |
| 12 | Assets under management | MEUR | Financial services | PCAF |
| 13 | Lending portfolio | MEUR | Banking | PCAF |
| 14 | Insurance premiums | MEUR | Insurance | PCAF |
| 15 | Hectares | ha | Agriculture, forestry | CDP, SBTi FLAG |
| 16 | Tonnes of product | t | Food, beverage, chemicals | SBTi SDA |
| 17 | Cement produced | t clinker | Cement | SBTi SDA |
| 18 | Steel produced | t crude steel | Steel | SBTi SDA |
| 19 | Aluminium produced | t aluminium | Aluminium | SBTi SDA |
| 20 | Paper produced | t paper | Pulp & paper | SBTi SDA |
| 21 | Number of customers | customers | Utilities, telecom | Sector-specific |
| 22 | Square metres of space cooled | m2 | Data centres, retail | PUE-related |
| 23 | Data throughput | TB | Data centres, ICT | Sector-specific |
| 24 | Meals served | meals | Hospitality, food service | Sector-specific |
| 25 | Nights sold | room-nights | Hospitality | GRESB, HCMI |

### 1.6 Target Users

**Primary:**
- Corporate sustainability teams needing intensity metrics for CSRD/ESRS E1, CDP, and SBTi reporting
- Portfolio managers (real estate, financial services) requiring asset-level intensity benchmarking
- Manufacturing companies tracking physical intensity against SBTi SDA sector pathways
- Chief Sustainability Officers needing decomposition analysis for board reporting

**Secondary:**
- ESG analysts and investors evaluating carbon efficiency across portfolios
- Third-party verifiers performing ISAE 3410 limited/reasonable assurance on intensity disclosures
- Consultants preparing clients for multi-framework intensity reporting
- Sector bodies and industry associations aggregating peer intensity data
- SBTi validators assessing target ambition and progress

### 1.7 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Intensity calculation accuracy | 100% match with reference values | Cross-validated against manual calculation test cases |
| LMDI decomposition accuracy | <0.1% residual (activity + structure + intensity = total change) | Mathematical closure validation |
| Multi-framework mapping coverage | 100% of ESRS E1-6, CDP, SEC, SBTi required intensity fields | Framework checklist completeness |
| SBTi SDA pathway calculation | 100% match with SBTi assessment tool | Cross-validated against SBTi reference calculations |
| Denominator data validation | >95% of input data passes quality checks | Data quality profiling pass rate |
| Processing time (full pipeline) | <30 minutes per reporting period | End-to-end pipeline execution time |
| Benchmark coverage | 12 sectors with published peer data | Sector coverage count |
| Uncertainty quantification | Combined uncertainty reported for 100% of intensity metrics | Uncertainty propagation completeness |
| Customer satisfaction | NPS >55 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| EU CSRD / ESRS E1 | Delegated Regulation 2023/2772, Appendix E | E1-6: "GHG intensity per net revenue" mandatory; additional physical intensity metrics per sector |
| SBTi Corporate Net-Zero Standard | v1.1 (2023) | Sectoral Decarbonisation Approach (SDA) requires sector-specific physical intensity pathways |
| CDP Climate Change | 2026 Questionnaire | C6.10: Scope 1+2 intensity (revenue), C-XX sector-specific physical intensities |
| SEC Climate Disclosure Rules | Final Rule 33-11275 (2024) | Item 1504: GHG intensity metrics "if the registrant uses GHG intensity" |
| GHG Protocol Corporate Standard | WRI/WBCSD (2004, revised 2015) | Chapter 9: Setting GHG targets, intensity metrics as complementary to absolute |
| ISO 14064-1:2018 | ISO (2018) | Clause 5.3.4: GHG intensity ratios; Clause 9.3: Reporting intensity |

### 2.2 Sector-Specific Frameworks

| Framework | Sector | Intensity Methodology |
|-----------|--------|----------------------|
| CRREM | Real estate | kgCO2e/m2/yr decarbonisation pathways per building type and climate zone |
| GRESB | Real estate | Like-for-like intensity (kgCO2e/m2) and absolute performance |
| GLEC Framework | Transport | gCO2e/tkm, gCO2e/pkm, gCO2e/TEU-km for multimodal transport |
| PCAF Standard | Financial services | Financed emissions intensity (tCO2e/MEUR) per asset class |
| TPI (Transition Pathway Initiative) | All | Management quality and carbon performance benchmarks |
| SBTi SDA | Manufacturing, power, transport, buildings | Sector convergence pathways to 2050 with annual intensity targets |
| SBTi FLAG | Agriculture, forestry | tCO2e/tonne product, tCO2e/hectare for FLAG sectors |
| HCMI (Hotel Carbon Measurement Initiative) | Hospitality | kgCO2e/occupied-room-night |

### 2.3 Supporting Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| GHG Protocol Scope 2 Guidance | WRI/WBCSD (2015) | Location-based vs market-based intensity implications |
| TCFD Recommendations | FSB/TCFD (2017) | Metrics and targets: "Organisations should provide GHG metrics normalised by industry-specific measures" |
| GRI 305 | GRI Standards 2016 (updated 2023) | 305-4: GHG emissions intensity |
| IFRS S2 | ISSB (2023) | Climate-related disclosures: intensity metrics where material |
| ISO 50001:2018 | Energy management | Energy Performance Indicators (EnPI) as energy intensity |
| IPCC AR6 | WG3 Chapter 11 | Sector-specific intensity pathways and benchmarks |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | Intensity calculation, decomposition, benchmarking, and target engines |
| Workflows | 8 | Multi-phase orchestration workflows |
| Templates | 10 | Report, dashboard, and analysis templates |
| Integrations | 12 | Agent, app, data, and system bridges |
| Presets | 8 | Sector-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `denominator_registry_engine.py` | Manages the library of 25+ standard denominators with metadata (unit, applicable sectors, framework alignment, validation rules). Provides denominator selection guidance based on sector, reporting frameworks, and organisational characteristics. Validates denominator data quality, completeness, and temporal consistency. Supports custom denominator registration with unit definitions. Handles unit conversion between denominator variants (e.g., MEUR to MUSD, m2 to sq ft). |
| 2 | `intensity_calculation_engine.py` | Core intensity metric calculation engine. Computes emissions/denominator ratios for any combination of scope inclusion (Scope 1, 2-location, 2-market, 1+2, 1+2+3, individual Scope 3 categories), denominator type, and time period. Handles edge cases (zero denominator, negative values, partial data). Supports weighted-average intensity for multi-entity consolidation per GHG Protocol Chapter 8. All arithmetic uses Python Decimal for precision. |
| 3 | `decomposition_engine.py` | LMDI (Logarithmic Mean Divisia Index) decomposition of emissions changes between two periods into activity effect, structure effect, and intensity effect. Implements both additive and multiplicative LMDI-I and LMDI-II methods per Ang (2004, 2015). Handles multi-sector decomposition (corporate with multiple business units), multi-level decomposition (group -> division -> site), and zero-value handling with L'Hopital limits. Validates decomposition closure (sum of effects equals total change within numerical precision). |
| 4 | `benchmarking_engine.py` | Peer intensity benchmarking against published sector data from CDP, TPI, GRESB, CRREM, and custom peer groups. Normalises comparison by adjusting for scope boundary differences, emission factor methodology, denominator definitions, and reporting period alignment. Calculates percentile ranking, distance-to-best-in-class, and sector average gap. Supports custom peer group definition and weighting. |
| 5 | `target_pathway_engine.py` | Science-based intensity target calculation per SBTi Sectoral Decarbonisation Approach (SDA). Implements sector convergence pathways for all SBTi-covered sectors with correct convergence formulae (not simple linear interpolation). Calculates required annual intensity reduction rate, current position vs pathway, gap analysis, and target year metrics. Supports both well-below-2C and 1.5C pathways. Includes SBTi FLAG (Forest, Land and Agriculture) methodology for agricultural intensity targets. |
| 6 | `trend_analysis_engine.py` | Multi-year intensity trend analysis with statistical tests. Calculates year-on-year and cumulative intensity changes (absolute and percentage), compound annual reduction rate (CARR), rolling averages, and trend significance (Mann-Kendall test). Projects forward intensity trajectory using linear, exponential, and logarithmic regression. Compares actual trajectory against target pathway. Separates organic intensity improvement from M&A and restatement effects using PACK-045 base year adjustments. |
| 7 | `scenario_engine.py` | What-if scenario modelling for intensity metrics. Models impact of: (a) efficiency improvements reducing numerator (emission reduction projects), (b) activity changes affecting denominator (revenue growth, production increase, portfolio expansion), (c) structural changes (acquisition/divestiture, sector mix shift), (d) methodology changes (emission factor updates, scope expansion). Runs Monte Carlo simulation with configurable iterations for probabilistic intensity projections. Calculates probability of achieving intensity targets under each scenario. |
| 8 | `uncertainty_engine.py` | Uncertainty quantification for intensity metrics propagating errors from both numerator (emissions) and denominator (activity data). Implements error propagation per IPCC Guidelines for National GHG Inventories (Tier 1 and Tier 2). Calculates combined uncertainty using root-sum-of-squares for independent errors. Generates uncertainty bands (90% and 95% confidence intervals) for all intensity metrics. Assesses data quality using a 5-point scale per GHG Protocol Corporate Standard Appendix A. |
| 9 | `disclosure_mapping_engine.py` | Maps calculated intensity metrics to specific disclosure requirements across ESRS E1-6 (mandatory revenue intensity + sector physical), CDP C6.10 (revenue intensity), CDP sector modules (physical intensity), SEC Item 1504 (used internally), SBTi (SDA pathway metrics), ISO 14064-1 Clause 9.3 (intensity ratios), TCFD metrics, GRI 305-4 (intensity), and IFRS S2. Validates completeness of required disclosures per framework. Generates framework-specific data tables. |
| 10 | `intensity_reporting_engine.py` | Aggregates all intensity analysis results into configurable dashboard and report formats. Provides executive summary with key intensity metrics, trend sparklines, benchmark positioning, and target progress. Generates detailed analytical reports with decomposition waterfall charts, benchmark spider diagrams, scenario fan charts, and uncertainty error bars. Supports export in MD, HTML, PDF, JSON, CSV, and XBRL formats for direct regulatory filing. |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `denominator_setup_workflow.py` | 4: SectorIdentification -> DenominatorSelection -> DataCollection -> Validation | End-to-end denominator configuration from sector identification through data validation |
| 2 | `intensity_calculation_workflow.py` | 4: DataIngestion -> ScopeConfiguration -> IntensityCalculation -> QualityAssurance | Core intensity calculation with scope selection and quality checks |
| 3 | `decomposition_analysis_workflow.py` | 3: PeriodSelection -> LMDIDecomposition -> EffectInterpretation | Period-over-period decomposition analysis |
| 4 | `benchmarking_workflow.py` | 4: PeerGroupDefinition -> DataNormalisation -> BenchmarkComparison -> RankingReport | Peer benchmarking with normalisation and ranking |
| 5 | `target_setting_workflow.py` | 4: BaselineCalculation -> PathwaySelection -> TargetCalculation -> ValidationReport | SBTi SDA intensity target calculation and validation |
| 6 | `scenario_analysis_workflow.py` | 3: ScenarioDefinition -> Simulation -> ProbabilityAssessment | What-if scenario modelling with Monte Carlo simulation |
| 7 | `disclosure_preparation_workflow.py` | 4: MetricAggregation -> FrameworkMapping -> CompletenessCheck -> DisclosurePackage | Multi-framework disclosure preparation and completeness validation |
| 8 | `full_intensity_pipeline_workflow.py` | 8: DenominatorSetup -> IntensityCalculation -> Decomposition -> Benchmarking -> TargetTracking -> ScenarioAnalysis -> DisclosureMapping -> ReportGeneration | Complete end-to-end intensity analytics pipeline |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `intensity_executive_dashboard.py` | MD, HTML, PDF, JSON | Executive summary: key intensity metrics, YoY change, benchmark position, target progress |
| 2 | `intensity_detailed_report.py` | MD, HTML, PDF, JSON | Comprehensive intensity analysis: all metrics, all scopes, all denominators, time series |
| 3 | `decomposition_waterfall.py` | MD, HTML, PDF, JSON | LMDI decomposition results with waterfall chart data: activity, structure, intensity effects |
| 4 | `benchmark_comparison.py` | MD, HTML, PDF, JSON | Peer benchmarking report: percentile ranking, sector average gap, best-in-class distance |
| 5 | `target_pathway_report.py` | MD, HTML, PDF, JSON | SBTi SDA target tracking: current vs pathway, annual progress, gap analysis, trajectory projection |
| 6 | `scenario_analysis_report.py` | MD, HTML, PDF, JSON | Scenario modelling results: base/optimistic/pessimistic, Monte Carlo distributions, probability of target achievement |
| 7 | `uncertainty_report.py` | MD, HTML, PDF, JSON | Data quality and uncertainty analysis: combined uncertainty bands, data quality scores, improvement recommendations |
| 8 | `esrs_e1_intensity_disclosure.py` | MD, HTML, PDF, JSON, XBRL | ESRS E1-6 compliant intensity disclosure with mandatory revenue intensity and sector physical intensity |
| 9 | `cdp_intensity_disclosure.py` | MD, HTML, PDF, JSON | CDP C6.10 and sector module compliant intensity data tables |
| 10 | `intensity_kpi_scorecard.py` | MD, HTML, PDF, JSON | KPI scorecard: traffic-light status for each intensity metric vs target, trend indicators, action items |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 10-phase DAG pipeline: DenominatorSetup -> DataIngestion -> EmissionsRetrieval -> IntensityCalculation -> Decomposition -> Benchmarking -> TargetTracking -> ScenarioAnalysis -> DisclosureMapping -> ReportGeneration. Conditional phases for decomposition (requires multi-year data), benchmarking (requires peer data), and scenario analysis (optional). Retry with exponential backoff, SHA-256 provenance chain, phase-level caching. |
| 2 | `mrv_bridge.py` | Routes to all 30 AGENT-MRV agents for emissions numerator data: MRV-001 through MRV-008 (Scope 1), MRV-009 through MRV-013 (Scope 2), MRV-014 through MRV-028 (Scope 3 categories), MRV-029 (Category Mapper), MRV-030 (Audit Trail). Bi-directional: MRV provides emissions; PACK-046 provides intensity context for MRV reporting. |
| 3 | `data_bridge.py` | Routes to AGENT-DATA agents for denominator data: DATA-002 (Excel/CSV for activity data imports), DATA-003 (ERP/Finance for revenue, FTE, production data), DATA-001 (PDF extraction for annual reports and financial statements), DATA-010 (Data Quality Profiler for denominator data assessment), DATA-015 (Cross-Source Reconciliation for denominator consistency). |
| 4 | `pack041_bridge.py` | PACK-041 Scope 1-2 Complete integration: retrieves consolidated Scope 1 and Scope 2 emissions totals (location-based and market-based) as numerator inputs. Shares organisational boundary definition and consolidation approach. |
| 5 | `pack042_043_bridge.py` | PACK-042/043 Scope 3 integration: retrieves Scope 3 category totals for Scope 1+2+3 intensity metrics. Handles both Starter (PACK-042) and Complete (PACK-043) Scope 3 data with coverage flags. |
| 6 | `pack044_bridge.py` | PACK-044 Inventory Management integration: retrieves inventory period definitions, data collection status, review/approval workflow status, and inventory versioning data for temporal consistency. |
| 7 | `pack045_bridge.py` | PACK-045 Base Year Management integration: retrieves base year emissions and denominator data, recalculation flags, and adjusted historical time series for consistent intensity trending. Critical for target tracking against base year intensity. |
| 8 | `benchmark_data_bridge.py` | External benchmark data integration: connects to published sector intensity databases (CDP public data, TPI benchmarks, GRESB data, CRREM pathways), refreshes benchmark data annually, normalises peer data to comparable scopes and denominators. |
| 9 | `sbti_pathway_bridge.py` | SBTi sector pathway data: maintains the latest SBTi SDA intensity pathways for all covered sectors, updates when SBTi publishes new pathways, provides convergence targets for target-setting engine. |
| 10 | `health_check.py` | 20-category system verification covering all 10 engines, 8 workflows, database connectivity, cache status, MRV bridge, DATA bridge, PACK-041/042/043/044/045 bridges, benchmark data freshness, SBTi pathway currency, and authentication/authorisation. |
| 11 | `setup_wizard.py` | 8-step guided configuration: organisation profile (sector, sub-sector, size), denominator selection (guided by sector/framework), scope inclusion rules, benchmark peer group definition, target parameters (SBTi pathway, custom targets), reporting frameworks (ESRS, CDP, SEC, SBTi, ISO), time series configuration (base year, reporting periods), and output preferences. |
| 12 | `alert_bridge.py` | Alert and notification integration: intensity threshold breach alerts, target off-track warnings, benchmark ranking change notifications, denominator data collection reminders, disclosure deadline alerts, and benchmark data update notifications. |

### 3.6 Presets

| # | Preset | Sector | Key Characteristics |
|---|--------|--------|---------------------|
| 1 | `corporate_multi_sector.yaml` | Multi-sector conglomerate | Revenue denominator primary (ESRS E1-6); multiple physical intensity metrics by business unit; decomposition by division; SBTi absolute + intensity combination targets. |
| 2 | `manufacturing.yaml` | Manufacturing (general) | Physical output denominator (t, units); tCO2e/tonne-product primary; SBTi SDA pathway; process intensity benchmarking; production-normalised trending. |
| 3 | `real_estate.yaml` | Real estate / Property | kgCO2e/m2 GLA primary; CRREM pathway alignment; GRESB benchmarking; like-for-like intensity; building-type segmentation; NABERS/EPC alignment. |
| 4 | `power_generation.yaml` | Electricity generation | gCO2/kWh primary; SBTi power sector pathway; grid intensity benchmarking; fuel-mix decomposition; capacity factor adjustment. |
| 5 | `transport_logistics.yaml` | Transport & logistics | gCO2e/tkm and gCO2e/pkm primary; GLEC framework; multimodal benchmarking; load factor normalisation; SBTi transport pathway. |
| 6 | `financial_services.yaml` | Banking / Insurance / Asset management | PCAF financed emissions intensity (tCO2e/MEUR invested/lent/insured); asset-class decomposition; portfolio alignment benchmarking; SBTi financial sector targets. |
| 7 | `food_agriculture.yaml` | Food, beverage, agriculture | tCO2e/tonne-product primary; SBTi FLAG methodology; land-use intensity (tCO2e/ha); supply chain intensity; water-energy-carbon nexus. |
| 8 | `sme_simplified.yaml` | SME (any sector, <250 employees) | Simplified 5-engine flow (denominator, intensity, trend, disclosure, reporting); revenue and FTE denominators only; simplified benchmarking; reduced data requirements; guided walkthrough. |

---

## 4. Engine Specifications

### 4.1 Engine 1: Denominator Registry Engine

**Purpose:** Manage and validate business activity denominators for intensity metric calculations.

**Denominator Data Model:**

| Field | Type | Description |
|-------|------|-------------|
| denominator_id | str | Unique identifier (e.g., `DEN-REV-EUR`) |
| name | str | Display name (e.g., "Net Revenue") |
| unit | str | Measurement unit (e.g., "MEUR") |
| category | Enum | FINANCIAL, PHYSICAL, HEADCOUNT, AREA, ENERGY, CUSTOM |
| applicable_sectors | List[str] | Sectors where this denominator is relevant |
| framework_alignment | Dict[str, str] | Framework -> requirement status (MANDATORY, RECOMMENDED, OPTIONAL) |
| validation_rules | List[Rule] | Data validation rules (positive, non-zero, max change YoY, etc.) |
| conversion_factors | Dict[str, Decimal] | Unit conversion factors (e.g., EUR->USD, m2->sqft) |

**Denominator Selection Algorithm:**

```
For each candidate denominator D:
    sector_score = 100 if D.applicable_to(org.sector) else 0
    framework_score = sum(weight[f] * alignment[D][f] for f in org.frameworks) / sum(weight[f])
    data_availability_score = 100 if D.data_available(org) else 50 if D.data_estimable(org) else 0
    relevance_score = sector_score * 0.3 + framework_score * 0.5 + data_availability_score * 0.2

Recommended denominators = sorted by relevance_score, top N
```

**Key Models:**
- `DenominatorDefinition` - ID, name, unit, category, sectors, frameworks, validation rules
- `DenominatorValue` - Denominator ID, period, value (Decimal), unit, data quality score, source
- `DenominatorRecommendation` - Ranked list of recommended denominators with relevance scores
- `DenominatorValidationResult` - Validation pass/fail, issues found, data quality assessment

**Non-Functional Requirements:**
- Denominator lookup and filtering: <100ms
- Validation of 100 denominator values: <1 second
- Unit conversion: exact Decimal arithmetic, no floating-point

### 4.2 Engine 2: Intensity Calculation Engine

**Purpose:** Core intensity metric calculation for any scope/denominator combination.

**Intensity Calculation:**

```
intensity = emissions_tco2e / denominator_value

Where:
    emissions_tco2e = total emissions for selected scopes (Decimal)
    denominator_value = activity metric value for the period (Decimal, > 0)

For multi-entity weighted average:
    intensity_consolidated = SUM(entity_emissions) / SUM(entity_denominators)
    (NOT the average of entity intensities -- per GHG Protocol)
```

**Scope Configuration Options:**

| Config | Numerator Includes | Use Case |
|--------|-------------------|----------|
| SCOPE_1_ONLY | Scope 1 | Direct emission efficiency |
| SCOPE_2_LOCATION | Scope 2 (location-based) | Grid-average efficiency |
| SCOPE_2_MARKET | Scope 2 (market-based) | Procurement efficiency |
| SCOPE_1_2_LOCATION | Scope 1 + 2 (location) | Standard operational efficiency |
| SCOPE_1_2_MARKET | Scope 1 + 2 (market) | Standard with RE procurement |
| SCOPE_1_2_3 | Scope 1 + 2 + 3 | Full value chain efficiency |
| SCOPE_3_SPECIFIC | Individual Scope 3 category | Category-specific intensity |
| CUSTOM | Configurable combination | Framework-specific requirements |

**Edge Case Handling:**

| Edge Case | Handling | Rationale |
|-----------|----------|-----------|
| Zero denominator | Return None with warning flag | Division by zero; denominator data issue |
| Negative emissions | Allow (carbon removals) | GHG Protocol allows negative for removals |
| Negative denominator | Reject with error | Business metrics should be non-negative |
| Partial scope data | Calculate with coverage flag | Indicate incomplete scope coverage |
| Mixed currencies | Convert to base currency at period-average rate | Currency consistency |

**Key Models:**
- `IntensityInput` - Emissions data (by scope), denominator data, scope configuration, period
- `IntensityResult` - Intensity value (Decimal), unit string, scope coverage, data quality, provenance hash
- `IntensityTimeSeries` - List of IntensityResult by period for trending
- `ConsolidatedIntensity` - Multi-entity weighted average with entity-level breakdown

**Non-Functional Requirements:**
- Single intensity calculation: <50ms
- Full portfolio (1000 entities, 10 denominators, 5 years): <30 seconds
- Decimal precision: 6 decimal places for intensity values

### 4.3 Engine 3: Decomposition Engine

**Purpose:** LMDI decomposition of emissions changes into activity, structure, and intensity effects.

**LMDI-I Additive Decomposition (Ang, 2004):**

```
Total change: dE = E_t - E_0

Activity effect:    dE_act = SUM_i [ L(E_i_t, E_i_0) * ln(D_t / D_0) ]
Structure effect:   dE_str = SUM_i [ L(E_i_t, E_i_0) * ln(S_i_t / S_i_0) ]
Intensity effect:   dE_int = SUM_i [ L(E_i_t, E_i_0) * ln(I_i_t / I_i_0) ]

Where:
    E_i = emissions from sector/entity i
    D = total activity (denominator aggregate)
    S_i = activity share of sector i (D_i / D)
    I_i = intensity of sector i (E_i / D_i)
    L(a,b) = (a - b) / (ln(a) - ln(b))  [logarithmic mean]

Closure: dE = dE_act + dE_str + dE_int  (exact, no residual)
```

**Zero-Value Handling:**

```
If E_i_0 = 0 and E_i_t > 0:  New sector entry -- attributed to activity effect
If E_i_0 > 0 and E_i_t = 0:  Sector exit -- attributed to activity effect
If E_i_0 = 0 and E_i_t = 0:  No contribution
If S_i_0 = 0 or S_i_t = 0:   Use limit of L(a,b) as a->0 or b->0
```

**Key Models:**
- `DecompositionInput` - Emissions by entity/sector for two periods, denominator by entity/sector
- `DecompositionResult` - Activity effect (tCO2e), structure effect (tCO2e), intensity effect (tCO2e), total change, closure residual, entity-level contributions
- `DecompositionTimeSeries` - Multi-period decomposition results for waterfall visualisation

**Non-Functional Requirements:**
- Decomposition (50 entities, 2 periods): <2 seconds
- Closure residual: <0.001 tCO2e (numerical precision limit)
- Support both additive and multiplicative LMDI variants

### 4.4 Engine 4: Benchmarking Engine

**Purpose:** Peer intensity benchmarking with normalisation and ranking.

**Benchmark Methodology:**

```
For each peer P in peer_group:
    intensity_P_normalised = normalise(intensity_P_raw, scope_adjustment, denominator_adjustment, period_adjustment)

percentile_rank = count(peers where intensity_normalised < org_intensity_normalised) / total_peers * 100
gap_to_average = org_intensity - mean(peer_intensities)
gap_to_best = org_intensity - min(peer_intensities)
gap_to_target = org_intensity - target_intensity
```

**Normalisation Adjustments:**

| Adjustment | Method | Purpose |
|-----------|--------|---------|
| Scope alignment | Add/subtract estimated scope components | Ensure comparable scope boundaries |
| Denominator standardisation | Convert to common unit and definition | Ensure comparable denominators |
| Period alignment | Pro-rata to calendar year or fiscal year | Ensure comparable time periods |
| Currency conversion | Period-average exchange rate | Ensure comparable financial denominators |
| Climate adjustment | HDD/CDD normalisation | For building intensity comparisons |
| Size adjustment | Log-linear regression residual | Account for economies of scale |

**Benchmark Data Sources:**

| Source | Coverage | Update Frequency | Data Elements |
|--------|----------|-----------------|---------------|
| CDP Open Data | 18,000+ companies | Annual | Scope 1+2 intensity by revenue, sector |
| TPI | 600+ companies, high-impact sectors | Annual | Carbon performance score, intensity metrics |
| GRESB | 1,800+ real estate entities | Annual | Like-for-like intensity, GRESB score |
| CRREM | All building types, 40+ countries | Annual | 1.5C and 2C pathway targets (kgCO2e/m2/yr) |
| Custom peer group | User-defined | User-defined | Any intensity metric |

**Key Models:**
- `BenchmarkInput` - Organisation intensity, peer group definition, normalisation settings
- `BenchmarkResult` - Percentile rank, gap-to-average, gap-to-best, gap-to-target, peer distribution statistics
- `PeerGroup` - List of peer organisations with intensity data and metadata
- `NormalisedComparison` - Peer-by-peer normalised intensity values and ranking

### 4.5 Engine 5: Target Pathway Engine

**Purpose:** SBTi Sectoral Decarbonisation Approach (SDA) intensity target calculation.

**SDA Convergence Formula (SBTi):**

```
For convergence pathways:
    I_target(y) = I_sector_2050 + (I_company_base - I_sector_2050) * (2050 - y) / (2050 - base_year)

Where:
    I_target(y) = company intensity target for year y
    I_sector_2050 = sector pathway endpoint (from SBTi)
    I_company_base = company intensity in base year
    base_year = SBTi-approved base year (typically within 2 years of submission)

Annual intensity reduction rate:
    r = 1 - (I_target(y+1) / I_target(y))
```

**SBTi Sector Pathways (Key Sectors):**

| Sector | Denominator | 2030 Target (1.5C) | 2050 Target (1.5C) | Source |
|--------|-------------|-------------------|-------------------|--------|
| Power generation | gCO2/kWh | ~138 | ~0 | IEA NZE |
| Steel | tCO2/t crude steel | ~1.2 | ~0.05 | IEA NZE |
| Cement | tCO2/t cite | ~0.43 | ~0.06 | IEA NZE |
| Aluminium | tCO2/t Al | ~2.7 | ~0 | IEA NZE |
| Commercial buildings | kgCO2/m2 | ~45 | ~0 | CRREM |
| Road transport (freight) | gCO2/tkm | ~70 | ~0 | IEA NZE |
| Road transport (passenger) | gCO2/pkm | ~75 | ~0 | IEA NZE |

**Key Models:**
- `TargetInput` - Base year intensity, sector, pathway choice (1.5C/WB2C), target years
- `TargetResult` - Annual intensity targets, required reduction rate, gap analysis, alignment score
- `PathwayComparison` - Actual trajectory vs target pathway with variance analysis
- `TargetProgress` - Current year position, % of target achieved, on-track/off-track assessment

### 4.6 Engine 6: Trend Analysis Engine

**Purpose:** Multi-year intensity trend analysis with statistical rigour.

**Trend Metrics:**

| Metric | Formula | Description |
|--------|---------|-------------|
| YoY Change | (I_t - I_{t-1}) / I_{t-1} * 100 | Year-on-year percentage change |
| CARR | (I_t / I_0)^(1/n) - 1 | Compound Annual Reduction Rate |
| Rolling Average | mean(I_{t-k+1}..I_t) | k-year rolling average (smoothing) |
| Mann-Kendall tau | Non-parametric trend test statistic | Statistical significance of monotonic trend |
| Sen's slope | Median of all pairwise slopes | Robust trend magnitude estimator |

**Regression Models for Projection:**

| Model | Formula | Use Case |
|-------|---------|----------|
| Linear | I(t) = a + bt | Constant absolute reduction |
| Exponential | I(t) = a * exp(bt) | Constant percentage reduction |
| Logarithmic | I(t) = a + b*ln(t) | Decelerating reduction |
| Power | I(t) = a * t^b | Flexible curvature |

**Key Models:**
- `TrendInput` - Intensity time series, trend configuration (smoothing, projection years)
- `TrendResult` - YoY changes, CARR, rolling averages, trend significance, projections, confidence bands

### 4.7 Engine 7: Scenario Engine

**Purpose:** What-if scenario modelling for intensity metrics.

**Scenario Types:**

| Scenario | Numerator Effect | Denominator Effect | Net Intensity Effect |
|----------|-----------------|-------------------|---------------------|
| Efficiency improvement | Decrease | Unchanged | Decrease (good) |
| Revenue growth (same efficiency) | Unchanged | Increase | Decrease (cosmetic) |
| Production increase (same efficiency) | Proportional increase | Increase | Unchanged |
| Acquisition (higher intensity target) | Increase | Increase | Depends on acquired intensity |
| Fuel switching | Decrease | Unchanged | Decrease (good) |
| Grid decarbonisation | Decrease (Scope 2) | Unchanged | Decrease (external) |

**Monte Carlo Simulation:**

```
For each iteration i in 1..N:
    emissions_i = sample(emissions_distribution)
    denominator_i = sample(denominator_distribution)
    intensity_i = emissions_i / denominator_i

probability_of_target = count(intensity_i <= target) / N
mean_intensity = mean(intensity_i for i in 1..N)
p10_intensity = percentile(10, intensity_i)
p90_intensity = percentile(90, intensity_i)
```

**Key Models:**
- `ScenarioInput` - Base case, scenario parameters, Monte Carlo configuration
- `ScenarioResult` - Scenario intensity values, probability distribution, target achievement probability
- `MonteCarloDistribution` - Mean, median, p10, p25, p75, p90, p95, p99

### 4.8 Engine 8: Uncertainty Engine

**Purpose:** Combined uncertainty quantification for intensity metrics.

**Error Propagation (IPCC Tier 1):**

```
For intensity I = E / D:
    U_I = sqrt(U_E^2 + U_D^2)

Where:
    U_E = relative uncertainty of emissions (%)
    U_D = relative uncertainty of denominator (%)
    U_I = relative uncertainty of intensity (%)

90% confidence interval: I * (1 +/- 1.645 * U_I / 100)
95% confidence interval: I * (1 +/- 1.960 * U_I / 100)
```

**Data Quality Scoring (5-point scale per GHG Protocol):**

| Score | Description | Typical Uncertainty |
|-------|-------------|-------------------|
| 1 | Audited, measured data | +/- 2-5% |
| 2 | Calculated from reliable data | +/- 5-10% |
| 3 | Calculated from assumptions | +/- 10-20% |
| 4 | Estimated, low confidence | +/- 20-50% |
| 5 | Proxy, very low confidence | +/- 50-100% |

**Key Models:**
- `UncertaintyInput` - Emissions with uncertainty, denominator with uncertainty
- `UncertaintyResult` - Combined uncertainty, confidence intervals, data quality scores, recommendations

### 4.9 Engine 9: Disclosure Mapping Engine

**Purpose:** Map intensity metrics to multi-framework disclosure requirements.

**Framework Mapping Matrix:**

| Framework | Required Intensity Metrics | Scope | Denominator | Mandatory/Optional |
|-----------|--------------------------|-------|-------------|-------------------|
| ESRS E1-6 | GHG intensity per net revenue | 1+2+3 | Net revenue (MEUR) | Mandatory |
| ESRS E1 | Sector physical intensity | Varies | Sector-specific | Mandatory (where applicable) |
| CDP C6.10 | Revenue intensity | 1+2 | Revenue | Mandatory |
| CDP Sector | Physical intensity | Varies | Sector-specific | Mandatory (sector questionnaires) |
| SEC | Intensity metrics if used internally | 1+2 (min) | As used internally | Conditional |
| SBTi | Sector pathway intensity | Varies | SDA denominator | Mandatory (for SDA targets) |
| ISO 14064-1 | GHG intensity ratios | 1+2 (min) | Organisation-chosen | Recommended |
| TCFD | Normalised metrics | 1+2 (min) | Industry-specific | Recommended |
| GRI 305-4 | GHG emissions intensity | 1 (min) | Organisation-chosen | If material |
| IFRS S2 | Intensity metrics | 1+2 (min) | As used internally | If material |

**Key Models:**
- `DisclosureInput` - Calculated intensity metrics, framework selection
- `DisclosureResult` - Framework-specific formatted outputs, completeness checklist, gap analysis
- `FrameworkMapping` - Framework, metric ID, description, scope, denominator, mandatory flag

### 4.10 Engine 10: Intensity Reporting Engine

**Purpose:** Aggregate all intensity analysis into configurable reports and dashboards.

**Report Sections:**

| Section | Content | Data Source |
|---------|---------|------------|
| Executive Summary | Key intensity metrics, headline changes, benchmark position | Engines 2, 4, 6 |
| Intensity Metrics Table | All metrics by scope, denominator, period | Engine 2 |
| Decomposition Analysis | Activity/structure/intensity waterfall | Engine 3 |
| Benchmark Positioning | Percentile rank, peer comparison | Engine 4 |
| Target Progress | SDA pathway tracking, gap analysis | Engine 5 |
| Trend Analysis | Multi-year trends, projections, statistical tests | Engine 6 |
| Scenario Results | What-if outcomes, Monte Carlo distributions | Engine 7 |
| Data Quality | Uncertainty bands, DQ scores, improvement actions | Engine 8 |
| Disclosure Readiness | Framework compliance checklist, data gaps | Engine 9 |

**Export Formats:**

| Format | Use Case | Features |
|--------|----------|----------|
| MD | Internal documentation | Markdown tables, structured text |
| HTML | Web dashboard | Interactive charts placeholder data |
| PDF | Board/investor presentation | Formatted layout, charts |
| JSON | API consumption, system integration | Machine-readable structured data |
| CSV | Data analysis, spreadsheet | Tabular data export |
| XBRL | Regulatory filing (ESRS) | Tagged taxonomy data |

**Key Models:**
- `ReportInput` - Sections to include, format, filters, branding
- `ReportResult` - Rendered report content, metadata, provenance hash
- `DashboardData` - JSON-structured data for frontend dashboard rendering

---

## 5. Database Schema

### 5.1 Migration Overview

| Migration | Table(s) | Purpose |
|-----------|----------|---------|
| V376 | Core schema: `intensity_denominators`, `intensity_metrics`, `intensity_config` | Core intensity data model |
| V377 | `intensity_denominator_values`, `intensity_denominator_registry` | Denominator value storage and registry |
| V378 | `intensity_calculations`, `intensity_time_series` | Calculated intensity metrics and time series |
| V379 | `intensity_decompositions`, `intensity_decomposition_effects` | LMDI decomposition results |
| V380 | `intensity_benchmarks`, `intensity_peer_groups`, `intensity_peer_data` | Benchmark data and peer groups |
| V381 | `intensity_targets`, `intensity_target_pathways`, `intensity_target_progress` | SBTi SDA targets and tracking |
| V382 | `intensity_scenarios`, `intensity_scenario_results`, `intensity_monte_carlo` | Scenario analysis and Monte Carlo |
| V383 | `intensity_uncertainty`, `intensity_data_quality` | Uncertainty and data quality |
| V384 | `intensity_disclosures`, `intensity_disclosure_mappings` | Framework disclosure mappings |
| V385 | Views, indexes, seed data | Performance views, composite indexes, reference data |

---

## 6. Testing Strategy

### 6.1 Test Categories

| Category | Count | Focus |
|----------|-------|-------|
| Unit tests (engines) | ~400 | Individual engine calculations, edge cases, precision |
| Unit tests (workflows) | ~100 | Workflow phase transitions, error handling |
| Unit tests (templates) | ~80 | Report generation, format correctness |
| Unit tests (integrations) | ~100 | Bridge routing, health checks |
| Integration tests | ~120 | Multi-engine pipelines, database operations |
| E2E tests | ~50 | Full pipeline end-to-end with realistic data |
| **Total** | **~850+** | **Comprehensive coverage** |

### 6.2 Key Test Scenarios

| Scenario | Validation |
|----------|-----------|
| Intensity calculation precision | Decimal arithmetic match against reference values |
| LMDI decomposition closure | Activity + structure + intensity = total (within numerical precision) |
| SBTi SDA pathway match | Match against SBTi assessment tool reference values |
| Zero denominator handling | Graceful handling, no division by zero |
| Multi-entity consolidation | Weighted average (not average of averages) |
| Historical restatement cascade | Denominator change propagates to all historical intensity metrics |
| Uncertainty propagation | Combined uncertainty matches IPCC Tier 1 formula |
| Framework disclosure completeness | 100% of mandatory fields populated per framework |

---

## 7. Performance Requirements

| Operation | Target | Notes |
|-----------|--------|-------|
| Single intensity calculation | <50ms | One scope/denominator/period |
| Full portfolio calculation | <30s | 1000 entities, 10 denominators, 5 years |
| LMDI decomposition | <2s | 50 entities, 2 periods |
| Monte Carlo simulation | <60s | 10,000 iterations, 5 scenarios |
| Full pipeline (all 10 phases) | <30min | Complete intensity analytics |
| Report generation | <30s | All formats including XBRL |

---

## 8. Security & Access Control

| Resource | Permission | Roles |
|----------|-----------|-------|
| Denominator data (read) | `pack046:denominator:read` | analyst, manager, admin |
| Denominator data (write) | `pack046:denominator:write` | manager, admin |
| Intensity calculation (execute) | `pack046:intensity:calculate` | analyst, manager, admin |
| Target setting (configure) | `pack046:target:configure` | manager, admin |
| Benchmark data (read) | `pack046:benchmark:read` | analyst, manager, admin |
| Disclosure package (export) | `pack046:disclosure:export` | manager, admin |
| Configuration (admin) | `pack046:config:admin` | admin |

---

## 9. Dependencies

### 9.1 Internal Dependencies

| Dependency | Type | Required/Optional |
|-----------|------|-------------------|
| PACK-041 (Scope 1-2 Complete) | Emissions numerator | Required |
| PACK-042 (Scope 3 Starter) | Scope 3 numerator | Optional (for Scope 1+2+3 intensity) |
| PACK-043 (Scope 3 Complete) | Full Scope 3 numerator | Optional (enterprise) |
| PACK-044 (Inventory Management) | Inventory governance | Optional (recommended) |
| PACK-045 (Base Year Management) | Historical consistency | Optional (recommended for trending) |
| AGENT-MRV (30 agents) | Emissions data | Required (via PACK-041+) |
| AGENT-DATA (20 agents) | Denominator data intake | Required |
| AGENT-FOUND (10 agents) | Orchestration, schema, audit | Required |

### 9.2 External Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | >=3.11 | Runtime |
| pydantic | >=2.0 | Configuration and data models |
| PyYAML | >=6.0 | Preset loading |
| numpy | >=1.24 | Statistical calculations (Mann-Kendall, Monte Carlo) |
| scipy | >=1.11 | Statistical tests, regression |

---

## 10. Roadmap

| Version | Features |
|---------|----------|
| 1.0.0 | Full 10-engine pack with all workflows, templates, integrations, presets |
| 1.1.0 | Real-time benchmark data feeds, automated peer group refresh |
| 1.2.0 | AI-assisted anomaly detection on intensity outliers |
| 2.0.0 | Predictive intensity modelling with ML-based projections |
