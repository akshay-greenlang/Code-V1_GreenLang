# PRD-PACK-035: Energy Benchmark Pack

**Pack ID:** PACK-035-energy-benchmark
**Category:** Energy Efficiency Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Draft
**Author:** GreenLang Product Team
**Date:** 2026-03-21
**Prerequisite:** None (standalone; enhanced with PACK-031 Industrial Energy Audit Pack, PACK-032 Building Energy Assessment Pack, and PACK-033 Quick Wins Identifier Pack if present; complemented by PACK-021/022/023 Net Zero Packs)

---

## 1. Executive Summary

### 1.1 Problem Statement

Energy benchmarking is the systematic process of comparing a facility's energy performance against peers, standards, and best practices. The EU Energy Efficiency Directive (EED) 2023/1791 and building performance regulations increasingly mandate energy benchmarking as the foundation for energy management. Yet organizations face persistent challenges:

1. **No standardised EUI framework**: Energy Use Intensity (EUI) in kWh/m2/yr is the universal metric, but calculating comparable EUI requires weather normalisation (HDD/CDD regression), occupancy adjustment, operating-hours correction, and activity-level normalisation. Most organisations use raw EUI without adjustments, producing misleading peer comparisons. A warehouse in Helsinki and one in Madrid cannot be meaningfully compared without degree-day normalisation.

2. **Fragmented benchmark databases**: Benchmark data is scattered across ENERGY STAR Portfolio Manager (US), CIBSE TM46 (UK), DIN V 18599 (Germany), RT 2020 (France), BPIE building stock data (EU), and national energy agencies. No single system aggregates these sources into a unified, queryable benchmark database with consistent methodology. Organisations spend weeks manually researching applicable benchmarks.

3. **Static point-in-time comparisons**: Most benchmarking exercises produce a single snapshot comparison. True energy performance management requires continuous benchmarking with trend analysis, cumulative sum (CUSUM) tracking, statistical process control (SPC), and automated alerting when performance deviates from expected trajectories. Without continuous monitoring, regression to poor performance goes undetected for months.

4. **Portfolio-level blindspots**: Organisations with 10-1000+ facilities need portfolio-level benchmarking to identify worst performers, best practices for replication, and overall portfolio energy intensity trends. Current approaches rely on spreadsheets that cannot handle weather normalisation, mixed building types, or rolling 12-month calculations across hundreds of sites.

5. **Missing regulatory alignment**: Energy Performance Certificates (EPCs), Display Energy Certificates (DECs), and building performance standards (EU EPBD recast 2024, NYC LL97, UK MEES) all require benchmarking data in specific formats. Organisations maintain separate data flows for each regulation rather than a single benchmarking platform that feeds all compliance requirements.

6. **Poor target-setting foundation**: Without robust benchmarking, organisations cannot set meaningful energy reduction targets. Science-based targets, carbon budgets, and energy intensity reduction commitments require a validated baseline and peer context. Top-quartile, median, and best-in-class benchmarks should inform target ambition levels.

7. **Lack of gap analysis**: Knowing that a facility is in the bottom quartile is useful; knowing exactly which end-use categories (lighting, HVAC, process, plug loads) drive the performance gap is actionable. Disaggregated benchmarking by end-use category, operating period, and energy carrier enables targeted improvement programmes.

8. **Weather normalisation complexity**: Proper weather normalisation requires degree-day regression (simple or change-point models), Typical Meteorological Year (TMY) data, and statistical validation (R-squared, CV(RMSE), t-ratio). Most organisations either skip normalisation entirely or use simplistic HDD-only methods that fail for cooling-dominated or mixed-mode buildings.

### 1.2 Solution Overview

PACK-035 is the **Energy Benchmark Pack** -- the fifth pack in the "Energy Efficiency Packs" category. It provides a comprehensive energy benchmarking platform covering EUI calculation, weather normalisation, peer comparison, sector-specific benchmarking, portfolio management, performance gap analysis, regulatory compliance, target setting, trend analysis, and continuous monitoring.

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the complete benchmarking lifecycle from data intake through continuous performance monitoring.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet Approach | PACK-035 Energy Benchmark Pack |
|-----------|-------------------------------|--------------------------------|
| EUI calculation | Raw consumption / floor area | Weather-normalised, occupancy-adjusted, activity-corrected EUI |
| Benchmark databases | Manual lookup from 5+ sources | Unified database: ENERGY STAR, CIBSE TM46, DIN V 18599, BPIE, national agencies |
| Weather normalisation | HDD-only or none | Multi-variable regression (HDD+CDD), change-point models, TMY data, statistical validation |
| Peer comparison | Single benchmark value | Percentile distribution (P10-P90), quartile bands, best-in-class targets |
| Portfolio benchmarking | Spreadsheet per site | Automated rolling 12-month across 1000+ sites with mixed building types |
| Gap analysis | Overall EUI gap only | Disaggregated by end-use (lighting, HVAC, process, plug loads, DHW) |
| Performance tracking | Annual snapshot | Continuous CUSUM, SPC charts, automated regression alerts |
| Regulatory compliance | Separate data flows per regulation | Single platform feeding EPC, DEC, EPBD, LL97, MEES, NABERS |
| Target setting | Ad hoc top-quartile target | Data-driven targets with peer context, SBTi alignment, trajectory modelling |
| Audit trail | Spreadsheet-based | SHA-256 provenance, full calculation lineage, digital audit trail |

### 1.4 Benchmark Categories

| # | Category | Benchmark Sources | Key Metrics |
|---|----------|------------------|-------------|
| 1 | Overall EUI | ENERGY STAR, CIBSE TM46, DIN V 18599, BPIE | kWh/m2/yr (site energy, source energy, primary energy) |
| 2 | Heating intensity | Degree-day normalised | kWh/m2/HDD, heating energy per unit floor area |
| 3 | Cooling intensity | Degree-day normalised | kWh/m2/CDD, cooling energy per unit floor area |
| 4 | Lighting power density | EN 15193, ASHRAE 90.1 | W/m2, LENI (kWh/m2/yr) |
| 5 | Plug load density | CIBSE Guide F, ASHRAE benchmarks | W/m2, kWh/m2/yr |
| 6 | Process energy intensity | Sector-specific (BREF documents, EPA) | kWh/unit output, GJ/tonne product |
| 7 | Water heating intensity | Building type benchmarks | kWh/m2/yr, kWh/occupant/yr |
| 8 | Carbon intensity | GHG Protocol, national grid factors | kgCO2e/m2/yr, kgCO2e/unit output |
| 9 | Energy cost intensity | Tariff-normalised | EUR/m2/yr, EUR/unit output |
| 10 | Peak demand | Load factor analysis | kW/m2, load factor (%), demand diversity |

### 1.5 Target Users

**Primary:**
- Energy managers responsible for multi-site energy performance reporting and target setting
- Facilities managers seeking to understand how their buildings compare to peers
- Sustainability officers preparing EPC/DEC submissions and EPBD compliance
- Portfolio managers ranking facilities for investment prioritisation

**Secondary:**
- Corporate real estate teams evaluating acquisition/disposition decisions based on energy performance
- Energy consultants performing benchmarking studies for clients
- Utility programme administrators segmenting customers for demand-side management
- Regulators and building performance standard administrators
- Investors screening building portfolios for climate risk (CRREM alignment)

### 1.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| EUI calculation accuracy | Within 2% of manual calculation | Cross-validated against reference calculations |
| Weather normalisation R-squared | >0.75 for heating/cooling-dominated buildings | Statistical fit of regression model |
| Benchmark database coverage | >90% of EU commercial building types | Mapped against CIBSE TM46 + DIN V 18599 categories |
| Portfolio processing speed | <5 min for 500 sites rolling 12-month | End-to-end processing time |
| Peer comparison accuracy | Within 5 percentile points of ENERGY STAR | Cross-validated against Portfolio Manager scores |
| Continuous monitoring latency | <1 hour from data receipt to alert | Time from meter data to deviation alert |
| Regulatory report generation | <10 min per building | Time to generate EPC/DEC-format output |
| Customer NPS | >55 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| EU Energy Efficiency Directive (EED) | Directive 2023/1791 (recast) | Article 11 energy audit benchmarking requirements; Article 8 energy savings quantification requires baseline benchmarks |
| EU Energy Performance of Buildings Directive (EPBD) | Directive 2024/1275 (recast) | EPC calculation methodology, building performance standards, MEPS thresholds, zero-emission building (ZEB) definitions |
| ISO 50001:2018 | Energy management systems | Clause 6.3 energy review requires energy performance benchmarking; Clause 6.6 EnPI requires peer comparison |
| ISO 50006:2014 | Energy baselines and EnPIs | Methodology for energy baseline establishment, EnPI selection, and normalisation |
| ASHRAE Standard 100-2018 | Energy efficiency in existing buildings | Benchmarking methodology, EUI targets by building type and climate zone |
| ENERGY STAR Portfolio Manager | US EPA | Building energy performance scoring methodology (1-100 scale) |

### 2.2 Supporting Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| CIBSE TM46:2008 | Energy benchmarks for buildings | UK building type energy benchmarks (electricity and fossil fuel) |
| DIN V 18599 | Energy assessment of buildings (Germany) | German building energy calculation methodology and reference values |
| EN 15603:2008 | Energy performance of buildings | Overall energy use and definition of energy ratings |
| EN 16798-1:2019 | Indoor environmental criteria | Occupancy and activity profiles for normalisation |
| ASHRAE Guideline 14-2014 | Measurement of energy savings | Statistical requirements for regression-based normalisation |
| CRREM 2024 | Carbon Risk Real Estate Monitor | Stranding risk pathways for building energy performance |

### 2.3 Carbon and Financial Standards

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2015) | Scope 1+2 carbon intensity benchmarking |
| SBTi Corporate Framework | SBTi (2024) | Target setting informed by benchmark peer performance |
| ESRS E1 Climate Change | EU CSRD (2023) | E1-5 energy consumption disclosure; benchmarked energy intensity |
| GRESB | Global Real Estate Sustainability Benchmark | Portfolio-level energy benchmarking for real estate funds |
| NABERS | National Australian Built Environment Rating System | Energy performance rating methodology (reference model) |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | Benchmarking calculation, comparison, and analysis engines |
| Workflows | 8 | Multi-phase orchestration workflows |
| Templates | 10 | Report, dashboard, and certificate templates |
| Integrations | 12 | Agent, data, benchmark database, and system bridges |
| Presets | 8 | Building/facility-type-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Class | File | Purpose |
|---|--------|-------|------|---------|
| 1 | EUI Calculator Engine | `EUICalculatorEngine` | `engines/eui_calculator_engine.py` | Calculates Energy Use Intensity (EUI) in kWh/m2/yr with multiple accounting boundaries: site energy (delivered at meter), source/primary energy (upstream losses applied via source energy factors), and cost-normalised energy. Supports floor area definitions: Gross Internal Area (GIA), Net Internal Area (NIA), Gross Lettable Area (GLA), and Treated Floor Area (TFA). Handles multi-fuel buildings with separate electricity, gas, oil, LPG, district heating, and district cooling metering. Calculates rolling 12-month, calendar year, and custom period EUI. Applies occupancy normalisation (kWh/m2/occupant-hour) and activity normalisation (kWh/m2/degree-day) for fair comparison. All arithmetic uses Python Decimal. Zero-hallucination. |
| 2 | Peer Comparison Engine | `PeerComparisonEngine` | `engines/peer_comparison_engine.py` | Compares facility EUI against peer groups from benchmark databases. Determines percentile ranking (P1-P99) within peer group using interpolated cumulative distribution. Identifies quartile band (Q1 best, Q4 worst) and distance to next quartile boundary. Supports multiple peer group definitions: building type, climate zone, floor area range, vintage, occupancy pattern. Generates ENERGY STAR-style 1-100 performance scores using lookup tables. Calculates peer group statistics: mean, median, P10, P25, P50, P75, P90, standard deviation, coefficient of variation. Zero-hallucination. |
| 3 | Sector Benchmark Engine | `SectorBenchmarkEngine` | `engines/sector_benchmark_engine.py` | Maintains comprehensive benchmark database covering 50+ building types across 6 benchmark sources: ENERGY STAR (US), CIBSE TM46 (UK), DIN V 18599 (Germany), RT 2020 (France), BPIE stock data (EU-27), and national energy agencies. Maps between classification systems (CIBSE categories, ENERGY STAR property types, DIN building use profiles). Provides typical, good practice, and best practice benchmarks for each building type. Includes process energy benchmarks for 20+ industrial sectors from EU BREF documents. Updates annually with latest published benchmark values. Zero-hallucination: all benchmarks from published authoritative sources. |
| 4 | Weather Normalisation Engine | `WeatherNormalisationEngine` | `engines/weather_normalisation_engine.py` | Performs statistical weather normalisation using degree-day regression models. Supports simple linear regression (energy vs HDD or CDD), 3-parameter change-point models (heating-only, cooling-only), 4-parameter change-point models (heating+cooling with baseload), and 5-parameter change-point models (heating+cooling with separate slopes and baseload). Calculates Heating Degree Days (HDD) and Cooling Degree Days (CDD) from hourly weather data with configurable base temperatures. Uses Typical Meteorological Year (TMY) data for normalised year calculation. Validates regression quality: R-squared >0.75, CV(RMSE) <20%, t-ratio >2.0 per ASHRAE Guideline 14. Handles non-weather-dependent facilities (data centers, process industry) with production-normalised models. Zero-hallucination. |
| 5 | Energy Performance Gap Engine | `EnergyPerformanceGapEngine` | `engines/energy_performance_gap_engine.py` | Analyses the gap between actual energy performance and benchmark targets. Disaggregates overall EUI gap by end-use category: lighting, space heating, space cooling, ventilation, domestic hot water, plug loads/equipment, process energy, and other. Uses end-use disaggregation methods: sub-metering data (preferred), CIBSE TM22 analysis, ASHRAE energy balance, or statistical estimation from building type profiles. Calculates gap in kWh/m2/yr, percentage, and annual cost (EUR/yr) for each end-use. Ranks end-uses by gap magnitude to identify priority improvement areas. Maps gaps to applicable PACK-033 quick wins and PACK-031/032 deep measures. Zero-hallucination. |
| 6 | Portfolio Benchmark Engine | `PortfolioBenchmarkEngine` | `engines/portfolio_benchmark_engine.py` | Benchmarks portfolios of 1-1000+ facilities with automated aggregation and ranking. Calculates portfolio-level metrics: area-weighted average EUI, total energy consumption, energy cost, and carbon emissions. Ranks facilities by EUI percentile within their building type peer group. Identifies best performers (top quartile) for best-practice replication and worst performers (bottom quartile) for priority intervention. Generates portfolio distribution charts (histogram, box plot, scatter). Tracks year-over-year portfolio improvement with weather-normalised trend. Supports multi-entity hierarchies (region, country, business unit, site). Zero-hallucination. |
| 7 | Regression Analysis Engine | `RegressionAnalysisEngine` | `engines/regression_analysis_engine.py` | Advanced statistical regression engine for energy modelling. Supports Ordinary Least Squares (OLS), change-point regression (2P, 3P, 4P, 5P models per ASHRAE Inverse Modelling Toolkit), and multivariate regression with production drivers. Calculates model statistics: R-squared, adjusted R-squared, CV(RMSE), NMBE, F-statistic, t-ratios, residual autocorrelation (Durbin-Watson). Performs model selection using Bayesian Information Criterion (BIC) to choose optimal model complexity. Detects change points in energy data (operational changes, equipment upgrades) using CUSUM analysis. Provides prediction intervals for forecasted consumption. All calculations use scipy.stats and numpy. Zero-hallucination. |
| 8 | Performance Rating Engine | `PerformanceRatingEngine` | `engines/performance_rating_engine.py` | Generates energy performance ratings aligned with multiple rating systems. Calculates ENERGY STAR score (1-100) using source EUI lookup tables by building type and climate region. Generates EPC-style A-G ratings using primary energy calculations per EN 15603. Calculates Display Energy Certificate (DEC) Operational Rating using actual energy consumption vs CIBSE TM46 benchmarks. Generates NABERS-style star ratings (1-6 stars). Calculates CRREM stranding year based on building energy trajectory vs decarbonisation pathway. Supports custom rating scales with configurable thresholds. Zero-hallucination: ratings from published methodology and lookup tables. |
| 9 | Trend Analysis Engine | `TrendAnalysisEngine` | `engines/trend_analysis_engine.py` | Analyses energy performance trends over time with statistical rigour. Calculates rolling 12-month EUI with weather normalisation to remove seasonal effects. Performs CUSUM (Cumulative Sum) analysis to detect persistent performance shifts. Applies Statistical Process Control (SPC) with control limits (UCL/LCL at 3-sigma) to identify statistically significant deviations. Calculates year-over-year change with confidence intervals. Performs Mann-Kendall trend test for long-term trend significance. Generates forecasts using exponential smoothing (Holt-Winters) for budget planning. Detects step changes from equipment upgrades or operational changes. Zero-hallucination. |
| 10 | Benchmark Report Engine | `BenchmarkReportEngine` | `engines/benchmark_report_engine.py` | Aggregates all benchmark analysis into configurable report and dashboard formats. Generates facility-level benchmark reports with EUI, peer comparison, performance rating, gap analysis, and trend data. Generates portfolio-level reports with rankings, distributions, and improvement trajectories. Creates regulatory-format outputs (EPC data, DEC operational ratings, GRESB data submission). Produces executive summaries with key performance indicators and action items. Supports export in MD, HTML, PDF, JSON, and CSV formats. Tracks report generation provenance with SHA-256 hashing. Zero-hallucination. |

### 3.3 Workflows

| # | Workflow | Phases | File | Purpose |
|---|----------|--------|------|---------|
| 1 | Initial Benchmark Workflow | 4: DataCollection -> EUICalculation -> PeerComparison -> BenchmarkReport | `workflows/initial_benchmark_workflow.py` | First-time benchmarking of a facility from data intake through peer comparison report |
| 2 | Continuous Monitoring Workflow | 3: DataIngestion -> PerformanceTracking -> AlertGeneration | `workflows/continuous_monitoring_workflow.py` | Ongoing monthly/quarterly performance monitoring with CUSUM and SPC analysis |
| 3 | Peer Comparison Workflow | 4: PeerGroupSelection -> WeatherNormalisation -> PercentileRanking -> ComparisonReport | `workflows/peer_comparison_workflow.py` | Detailed peer comparison with weather-normalised EUI and percentile ranking |
| 4 | Portfolio Benchmark Workflow | 4: PortfolioDataCollection -> FacilityBenchmarking -> PortfolioAggregation -> PortfolioReport | `workflows/portfolio_benchmark_workflow.py` | Multi-site portfolio benchmarking with ranking and distribution analysis |
| 5 | Performance Gap Workflow | 3: BenchmarkEstablishment -> EndUseDisaggregation -> GapAnalysisReport | `workflows/performance_gap_workflow.py` | End-use disaggregated gap analysis identifying priority improvement areas |
| 6 | Regulatory Compliance Workflow | 4: DataValidation -> RatingCalculation -> CertificateGeneration -> SubmissionPackage | `workflows/regulatory_compliance_workflow.py` | EPC, DEC, and building performance standard compliance reporting |
| 7 | Target Setting Workflow | 3: BaselineEstablishment -> PeerContextAnalysis -> TargetDefinition | `workflows/target_setting_workflow.py` | Data-driven target setting using benchmark context and peer performance |
| 8 | Full Assessment Workflow | 6: FacilitySetup -> DataIngestion -> BenchmarkCalculation -> PeerComparison -> GapAnalysis -> ReportGeneration | `workflows/full_assessment_workflow.py` | Complete end-to-end benchmark assessment from onboarding to final report |

### 3.4 Templates

| # | Template | Class | File | Purpose |
|---|----------|-------|------|---------|
| 1 | EUI Benchmark Report | `EUIBenchmarkReportTemplate` | `templates/eui_benchmark_report.py` | Facility EUI report with weather-normalised values, trend charts, and methodology notes |
| 2 | Peer Comparison Report | `PeerComparisonReportTemplate` | `templates/peer_comparison_report.py` | Percentile ranking against peer group with distribution charts and quartile analysis |
| 3 | Sector Benchmark Report | `SectorBenchmarkReportTemplate` | `templates/sector_benchmark_report.py` | Building type benchmark comparison across multiple benchmark databases |
| 4 | Energy Performance Certificate | `EnergyPerformanceCertificateTemplate` | `templates/energy_performance_certificate.py` | EPC/DEC-format energy performance certificate with A-G rating |
| 5 | Portfolio Dashboard | `PortfolioDashboardTemplate` | `templates/portfolio_dashboard.py` | Multi-site portfolio dashboard with rankings, distributions, and heat maps |
| 6 | Gap Analysis Report | `GapAnalysisReportTemplate` | `templates/gap_analysis_report.py` | End-use disaggregated gap analysis with waterfall charts and improvement priorities |
| 7 | Target Tracking Report | `TargetTrackingReportTemplate` | `templates/target_tracking_report.py` | Performance vs. target tracking with trajectory forecasts and variance analysis |
| 8 | Regulatory Compliance Report | `RegulatoryComplianceReportTemplate` | `templates/regulatory_compliance_report.py` | Multi-regulation compliance report (EPBD, MEES, LL97, NABERS) |
| 9 | Trend Analysis Report | `TrendAnalysisReportTemplate` | `templates/trend_analysis_report.py` | Multi-year trend analysis with CUSUM, SPC charts, and forecast |
| 10 | Executive Summary Report | `ExecutiveSummaryReportTemplate` | `templates/executive_summary_report.py` | C-suite executive summary with KPIs, peer position, and action items |

### 3.5 Integrations

| # | Integration | Class | File | Purpose |
|---|-------------|-------|------|---------|
| 1 | Pack Orchestrator | `EnergyBenchmarkOrchestrator` | `integrations/pack_orchestrator.py` | 12-phase pipeline orchestrator with DAG dependency resolution and provenance tracking |
| 2 | MRV Benchmark Bridge | `MRVBenchmarkBridge` | `integrations/mrv_benchmark_bridge.py` | Routes emission factor data from MRV agents for carbon intensity benchmarking |
| 3 | Data Benchmark Bridge | `DataBenchmarkBridge` | `integrations/data_benchmark_bridge.py` | Routes data intake through DATA agents for meter data and utility bills |
| 4 | PACK-031 Bridge | `Pack031Bridge` | `integrations/pack031_bridge.py` | Imports energy audit baselines and EnPI data from Industrial Energy Audit Pack |
| 5 | PACK-032 Bridge | `Pack032Bridge` | `integrations/pack032_bridge.py` | Imports building assessment data and zone-level energy breakdown |
| 6 | PACK-033 Bridge | `Pack033Bridge` | `integrations/pack033_bridge.py` | Imports quick win identification results and links gap analysis to improvement measures |
| 7 | Energy Star Bridge | `EnergyStarBridge` | `integrations/energy_star_bridge.py` | Connects to ENERGY STAR Portfolio Manager for score calculation and benchmark data |
| 8 | Weather Service Bridge | `WeatherServiceBridge` | `integrations/weather_service_bridge.py` | Retrieves weather station data, HDD/CDD series, and TMY data for normalisation |
| 9 | EPC Registry Bridge | `EPCRegistryBridge` | `integrations/epc_registry_bridge.py` | Connects to EPC registries for certificate data and benchmark comparison |
| 10 | Benchmark Database Bridge | `BenchmarkDatabaseBridge` | `integrations/benchmark_database_bridge.py` | Unified interface to CIBSE TM46, DIN V 18599, BPIE, and national benchmark databases |
| 11 | Health Check | `HealthCheck` | `integrations/health_check.py` | 15-category system health verification covering engines, databases, and connectivity |
| 12 | Setup Wizard | `SetupWizard` | `integrations/setup_wizard.py` | 8-step guided facility configuration with building type selection and data import |

### 3.6 Presets

| # | Preset | File | Description |
|---|--------|------|-------------|
| 1 | Commercial Office | `config/presets/commercial_office.yaml` | Offices: EUI 150-300 kWh/m2/yr, HVAC+lighting dominant, ENERGY STAR scoring |
| 2 | Industrial Manufacturing | `config/presets/industrial_manufacturing.yaml` | Manufacturing: EUI 200-1000 kWh/m2/yr, process-energy dominant, kWh/unit output |
| 3 | Retail Store | `config/presets/retail_store.yaml` | Retail: EUI 200-500 kWh/m2/yr, lighting+refrigeration dominant |
| 4 | Warehouse Logistics | `config/presets/warehouse_logistics.yaml` | Warehouse: EUI 50-200 kWh/m2/yr, heating+lighting dominant |
| 5 | Healthcare Facility | `config/presets/healthcare_facility.yaml` | Hospitals: EUI 300-700 kWh/m2/yr, 24/7 operation, special ventilation |
| 6 | Educational Campus | `config/presets/educational_campus.yaml` | Education: EUI 100-350 kWh/m2/yr, seasonal occupancy patterns |
| 7 | Data Center | `config/presets/data_center.yaml` | Data centres: PUE-based benchmarking, cooling-dominant, kWh/kW IT load |
| 8 | Multi-Site Portfolio | `config/presets/multi_site_portfolio.yaml` | Portfolio: mixed building types, weighted aggregation, facility ranking |

---

## 4. Database Schema

### 4.1 Migrations (V266-V275)

| Migration | Description |
|-----------|-------------|
| V266__pack035_energy_benchmark_001 | Core benchmark schema: facility profiles, building types, floor areas, metering points |
| V267__pack035_energy_benchmark_002 | Benchmark database tables: ENERGY STAR, CIBSE TM46, DIN V 18599, BPIE, sector benchmarks |
| V268__pack035_energy_benchmark_003 | EUI calculation tables: energy data, consumption records, EUI results, normalised values |
| V269__pack035_energy_benchmark_004 | Weather data tables: stations, HDD/CDD series, TMY data, regression models |
| V270__pack035_energy_benchmark_005 | Peer comparison tables: peer groups, percentile rankings, comparison results |
| V271__pack035_energy_benchmark_006 | Portfolio tables: portfolio definitions, facility memberships, aggregation results |
| V272__pack035_energy_benchmark_007 | Performance rating tables: EPC ratings, ENERGY STAR scores, CRREM pathways |
| V273__pack035_energy_benchmark_008 | Trend analysis tables: CUSUM data, SPC charts, alerts, forecasts |
| V274__pack035_energy_benchmark_009 | Gap analysis tables: end-use disaggregation, gap results, improvement mapping |
| V275__pack035_energy_benchmark_010 | Views, indexes, RLS policies, seed data (benchmarks, weather stations, emission factors) |

---

## 5. Agent Dependencies

### 5.1 MRV Agents

| Agent | Required | Purpose |
|-------|----------|---------|
| AGENT-MRV-001 (Stationary Combustion) | Yes | Scope 1 emission factors for heating fuel carbon intensity benchmarking |
| AGENT-MRV-009 (Scope 2 Location-Based) | Yes | Location-based grid emission factors for carbon intensity |
| AGENT-MRV-010 (Scope 2 Market-Based) | Yes | Market-based emission factors for carbon intensity |
| AGENT-MRV-013 (Dual Reporting) | Yes | Reconcile location-based and market-based carbon intensity |

### 5.2 Data Agents

| Agent | Required | Purpose |
|-------|----------|---------|
| AGENT-DATA-001 (PDF Extractor) | Yes | Extract energy data from utility bills |
| AGENT-DATA-002 (Excel/CSV Normalizer) | Yes | Normalise meter data exports |
| AGENT-DATA-004 (API Gateway) | Yes | Route external API connections for benchmarks and weather |
| AGENT-DATA-010 (Data Quality Profiler) | Yes | Profile data quality of energy consumption data |
| AGENT-DATA-013 (Outlier Detection) | Yes | Detect anomalous consumption readings |
| AGENT-DATA-014 (Time Series Gap Filler) | Yes | Fill gaps in meter data time series |
| AGENT-DATA-018 (Data Lineage Tracker) | Yes | Track data lineage from meters through to benchmark reports |

### 5.3 Foundation Agents

| Agent | Required | Purpose |
|-------|----------|---------|
| AGENT-FOUND-001 (Orchestrator) | Yes | DAG-based pipeline orchestration |
| AGENT-FOUND-002 (Schema Compiler) | Yes | Validate benchmark data schemas |
| AGENT-FOUND-003 (Unit Normalizer) | Yes | Normalise energy units (kWh, MJ, GJ, therms) |
| AGENT-FOUND-004 (Assumptions Registry) | Yes | Track assumptions in normalisation and benchmarking |
| AGENT-FOUND-005 (Citations) | Yes | Link benchmarks to authoritative sources |
| AGENT-FOUND-006 (Access Policy) | Yes | Multi-site access control for portfolio benchmarking |
| AGENT-FOUND-008 (Reproducibility) | Yes | SHA-256 provenance for all calculations |
| AGENT-FOUND-010 (Observability) | Yes | Pipeline monitoring and telemetry |

### 5.4 Pack Dependencies

| Pack | Required | Purpose |
|------|----------|---------|
| PACK-031 (Industrial Energy Audit) | No | Import energy baselines for industrial facilities |
| PACK-032 (Building Energy Assessment) | No | Import building assessment zone-level data |
| PACK-033 (Quick Wins Identifier) | No | Link gap analysis to applicable quick win measures |

---

## 6. Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Single facility EUI calculation | <500ms | Including weather normalisation |
| Peer comparison ranking | <200ms | Against pre-computed peer distribution |
| Portfolio benchmark (500 sites) | <5 minutes | Rolling 12-month with normalisation |
| Weather normalisation regression | <2 seconds | Per meter point, 3-year data |
| Gap analysis (end-use disaggregation) | <3 seconds | Per facility |
| Report generation (single facility) | <10 seconds | Full benchmark report with charts |
| Portfolio report generation | <60 seconds | 500-site portfolio |
| Cache hit ratio | >80% | Benchmark lookups and weather data |
| Memory ceiling | 4096 MB | Per worker process |

---

## 7. Security & Access Control

| Aspect | Specification |
|--------|---------------|
| Authentication | JWT (RS256) |
| Authorization | RBAC with facility-level and portfolio-level access control |
| Encryption at rest | AES-256-GCM |
| Encryption in transit | TLS 1.3 |
| Audit logging | All benchmark operations logged |
| PII redaction | Facility addresses and contact details redacted in shared reports |
| Data classification | INTERNAL, CONFIDENTIAL |
| Required roles | energy_manager, facility_manager, sustainability_officer, portfolio_manager, analyst, viewer, admin |

---

## 8. Testing

| Category | Coverage Target | Description |
|----------|----------------|-------------|
| Unit tests | 85%+ | All engines, models, calculations |
| Integration tests | 80%+ | Workflow execution, agent bridges |
| End-to-end tests | 75%+ | Full benchmark pipelines |
| Performance tests | All targets met | Benchmark processing speed validation |
| Regression tests | 100% known benchmarks | Cross-validation against published benchmark values |
| Total test target | 850+ tests | Comprehensive coverage across all components |

---

## 9. Compliance Frameworks

| Framework | Coverage |
|-----------|----------|
| EU EED | Article 8/11 energy audit benchmarking |
| EU EPBD | EPC calculation, DEC operational rating, MEPS thresholds |
| ISO 50001:2018 | Energy review, EnPI, benchmarking requirements |
| ISO 50006:2014 | Energy baseline and performance indicator methodology |
| ASHRAE Standard 100 | Building energy benchmarking methodology |
| GHG Protocol | Carbon intensity benchmarking (Scope 1+2) |
| SBTi | Target setting informed by benchmark context |
| GRESB | Portfolio energy performance reporting |
| CRREM | Building stranding risk assessment |
| NABERS | Energy performance rating methodology |
