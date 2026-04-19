# PRD-PACK-040: Measurement & Verification (M&V) Pack

**Pack ID:** PACK-040-mv
**Category:** Energy Efficiency Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-23
**Prerequisite:** None (standalone; enhanced with PACK-031 Industrial Energy Audit, PACK-032 Building Energy Assessment, PACK-033 Quick Wins Identifier, PACK-039 Energy Monitoring if present)

---

## 1. Executive Summary

### 1.1 Problem Statement

Energy efficiency projects generate billions of dollars in claimed savings annually, yet organizations consistently fail to verify whether those savings actually materialized. The International Performance Measurement and Verification Protocol (IPMVP) estimates that 30-40% of claimed energy savings are inaccurate -- either overestimated due to inadequate baseline modeling, or underestimated due to failure to account for non-routine adjustments. This verification gap undermines confidence in energy efficiency investments and creates six critical challenges:

1. **Inadequate baseline development**: Energy baselines require multivariate regression modeling that accounts for weather (HDD/CDD with facility-specific balance points), production volume, occupancy, operating hours, and other independent variables. Most organizations use simple average consumption as a baseline, which fails to account for 40-60% of consumption variance. Change-point regression models (3P cooling, 4P, 5P) are required for weather-dependent facilities but rarely implemented due to mathematical complexity. Without proper baselines, savings calculations are fundamentally unreliable.

2. **Missing routine and non-routine adjustments**: Post-retrofit consumption must be adjusted for changes in conditions between baseline and reporting periods. Routine adjustments (weather normalization, production normalization) are sometimes performed; non-routine adjustments (floor area changes, occupancy shifts, equipment additions, schedule changes) are almost never quantified. IPMVP requires both, and failure to apply them can create apparent "savings" or "losses" of 20-50% that have nothing to do with the efficiency measure.

3. **Uncertainty quantification gaps**: ASHRAE Guideline 14-2014 requires that verified savings exceed twice the standard error of the estimate at 68% confidence (or the fractional savings uncertainty be less than 50% at 68% confidence). Most M&V practitioners cannot calculate fractional savings uncertainty, t-statistics for regression coefficients, or propagate measurement uncertainty through savings calculations. Without uncertainty quantification, organizations cannot determine whether observed savings are statistically significant or merely noise.

4. **IPMVP option misalignment**: The four IPMVP options (A: Retrofit Isolation - Key Parameter Measurement, B: Retrofit Isolation - All Parameter Measurement, C: Whole Facility, D: Calibrated Simulation) have different applicability, accuracy, cost, and complexity trade-offs. Most practitioners default to Option C (whole facility) even when retrofit isolation (A/B) would provide better accuracy at lower cost. Conversely, Option C is sometimes used when the ECM affects less than 10% of whole-facility energy, making savings invisible in meter noise.

5. **Savings persistence degradation**: Energy savings from efficiency measures degrade over time due to equipment aging, maintenance neglect, operational drift, and behavioral reversion. IPMVP recommends ongoing M&V throughout the useful life of measures, yet most M&V stops after the first year. Studies show that 15-25% of Year 1 savings are lost by Year 3 without persistence monitoring. Performance contracts (ESPCs/ESCOs) require multi-year savings verification that current tools cannot support efficiently.

6. **Compliance and reporting deficiencies**: ISO 50015:2014, ISO 50001:2018, FEMP M&V Guidelines 4.0, and EU EED Article 7 all require documented M&V. Energy Performance Contracts (EPCs) under EU Directive 2012/27/EU require independent savings verification. California Title 24 and ASHRAE 90.1 require commissioning verification. Most organizations lack standardized M&V reporting that satisfies multiple regulatory frameworks simultaneously.

### 1.2 Solution Overview

PACK-040 is the **Measurement & Verification (M&V) Pack** -- the tenth pack in the "Energy Efficiency Packs" category. It provides a comprehensive, standards-compliant M&V platform covering the complete savings verification lifecycle from baseline development through long-term persistence tracking.

The pack implements all four IPMVP options with full statistical rigor per ASHRAE Guideline 14-2014, including multivariate regression with change-point models, routine and non-routine adjustments, fractional savings uncertainty quantification, and automated compliance checking against ISO 50015:2014, FEMP M&V Guidelines 4.0, and EU EED Article 7.

Key capabilities:
- **Baseline development**: OLS regression, 3P/4P/5P change-point models, TOWT (Time-of-Week & Temperature), automated model selection with CVRMSE/NMBE validation
- **Savings calculation**: Avoided energy use, normalized savings, cost savings with uncertainty bounds
- **IPMVP Options A-D**: Complete implementation with option selection guidance, boundary definition, and verification protocols
- **Routine adjustments**: Weather normalization (HDD/CDD with optimized balance points), production normalization, occupancy adjustment
- **Non-routine adjustments**: Floor area changes, equipment additions/removals, schedule changes, static factor corrections
- **Uncertainty analysis**: Measurement uncertainty, sampling uncertainty, model uncertainty, combined uncertainty propagation per ASHRAE 14
- **Metering plan**: Meter selection, calibration requirements, sampling protocols, data quality checks
- **Persistence tracking**: Multi-year savings tracking, degradation analysis, re-commissioning triggers
- **Compliance reporting**: ISO 50015, FEMP, IPMVP, EU EED, ASHRAE 14 compliant M&V reports

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the complete M&V lifecycle.

Every calculation is **zero-hallucination** (deterministic formulas and lookups only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet M&V | PACK-040 M&V Pack |
|-----------|--------------------------|-------------------|
| Baseline modeling | Simple average or single-variable | Multivariate regression, 3P/4P/5P change-point, TOWT, automated model selection |
| Model validation | Manual R-squared check | CVRMSE, NMBE, t-statistics, F-test, DW, autocorrelation, all per ASHRAE 14 |
| Adjustments | Weather only (if any) | Routine (weather, production, occupancy) + non-routine (floor area, equipment, schedule) |
| Uncertainty | Not calculated | Full ASHRAE 14 fractional savings uncertainty with measurement + model + sampling |
| IPMVP compliance | Partial (usually Option C only) | All 4 options with automated option selection and compliance checking |
| Time to complete | 2-4 weeks per project | 2-4 hours per project (10-20x faster) |
| Multi-year tracking | Manual annual updates | Automated persistence tracking with degradation alerts |
| Reporting | Custom spreadsheets | Standardized reports compliant with ISO 50015, FEMP, IPMVP, EU EED |
| Audit trail | File versions | SHA-256 provenance, full calculation lineage, digital audit trail |

### 1.4 Target Users

**Primary:**
- Energy managers performing post-retrofit savings verification under ISO 50001
- ESCO/EPC project managers verifying guaranteed savings in performance contracts
- M&V practitioners conducting independent savings verification
- Facility managers tracking energy savings from implemented efficiency measures

**Secondary:**
- FEMP compliance officers verifying federal energy savings
- Utility program evaluators verifying demand-side management program savings
- Corporate sustainability teams reporting verified emissions reductions
- Building commissioning agents verifying commissioning savings
- Finance teams validating energy savings for green bond reporting

### 1.5 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Baseline model accuracy | CVRMSE <25% monthly, <30% daily (ASHRAE 14) | Validated against independent regression tools |
| Savings calculation accuracy | Within 10% of independent M&V practitioner results | Cross-validated against manual M&V calculations |
| IPMVP compliance score | >95% on automated compliance checklist | Checked against IPMVP Core Concepts 2022 requirements |
| Time to complete M&V analysis | <4 hours per project (vs. 2-4 weeks manual) | Time from data intake to verified savings report |
| Uncertainty quantification | 100% match with ASHRAE 14 reference calculations | Cross-validated against ASHRAE 14 example problems |
| Persistence detection | Detect >10% savings degradation within 30 days | Alert response time for performance degradation |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| IPMVP Core Concepts 2022 | Efficiency Valuation Organization (EVO) | Primary M&V protocol: Options A/B/C/D, baseline development, adjustments, savings calculation |
| ASHRAE Guideline 14-2014 | Measurement of Energy, Demand, and Water Savings | Statistical requirements: CVRMSE <25% monthly, NMBE <5%, regression model validation, uncertainty quantification |
| ISO 50015:2014 | Measurement and verification of energy performance | M&V framework aligned with IPMVP; defines M&V plan, baseline, reporting period, adjustments |
| ISO 50001:2018 | Energy management systems | Clause 6.6 planning for collection of energy data; Clause 9.1 monitoring, measurement, analysis |
| ISO 50006:2014 | Energy baselines and energy performance indicators | Baseline development methodology, EnPI calculation, baseline adjustment criteria |
| FEMP M&V Guidelines 4.0 | Federal Energy Management Program | US federal M&V requirements for ESPC and UESC contracts |

### 2.2 Supporting Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| ASHRAE 90.1-2022 | Energy standard for buildings | Commissioning verification baseline requirements |
| EU EED Article 7 | Directive 2023/1791 | Energy savings calculation and verification methodology |
| EU EPC Directive | Directive 2012/27/EU Article 18 | Energy Performance Contract savings verification requirements |
| GHG Protocol | WRI/WBCSD Corporate Standard | Emissions reduction verification from energy savings |
| California Title 24 | Building Energy Efficiency Standards | Commissioning and M&V verification requirements |
| BPA M&V Protocol | Bonneville Power Administration | Regional M&V guidelines for Pacific Northwest |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | Baseline development, savings calculation, uncertainty, IPMVP options, regression, weather, metering, adjustment, persistence, reporting |
| Workflows | 8 | Multi-phase orchestration workflows for complete M&V lifecycle |
| Templates | 10 | M&V plan, baseline, savings, uncertainty, annual, option comparison, metering, persistence, executive, compliance reports |
| Integrations | 12 | Platform agent bridges, cross-pack bridges, weather service, utility data |
| Presets | 8 | Facility-type-specific M&V configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |

### 3.2 Engines

| # | Engine | File | Class | Purpose |
|---|--------|------|-------|---------|
| 1 | Baseline Engine | `baseline_engine.py` | `BaselineEngine` | Energy baseline development using multivariate regression with change-point models (3P cooling, 3P heating, 4P, 5P). Automated model selection comparing OLS, change-point, and TOWT models. Calculates R-squared, adjusted R-squared, CVRMSE, NMBE, t-statistics, F-statistic, Durbin-Watson. Supports daily, weekly, and monthly baselines. Balance point optimization for HDD/CDD. Handles multiple independent variables (temperature, production, occupancy, daylight hours). |
| 2 | Adjustment Engine | `adjustment_engine.py` | `AdjustmentEngine` | Routine and non-routine adjustment calculation per IPMVP. Routine adjustments: weather normalization (HDD/CDD to reporting period conditions), production normalization, occupancy adjustment, operating hours correction. Non-routine adjustments: floor area changes (proportional scaling), equipment additions/removals (engineering estimates), schedule changes (delta calculation), static factor corrections. Adjustment documentation with justification and uncertainty. |
| 3 | Savings Engine | `savings_engine.py` | `SavingsEngine` | Energy savings calculation from adjusted baseline vs. reporting period consumption. Avoided energy use = adjusted baseline - reporting period actual. Normalized savings = baseline prediction at standard conditions - reporting period at standard conditions. Cost savings using blended energy rates with demand charge allocation. Cumulative savings tracking over multi-year periods. Annualization for partial-year reporting periods. |
| 4 | Uncertainty Engine | `uncertainty_engine.py` | `UncertaintyEngine` | Comprehensive uncertainty quantification per ASHRAE Guideline 14-2014. Measurement uncertainty (meter accuracy, CT/PT errors, calibration drift). Model uncertainty (regression standard error, prediction interval). Sampling uncertainty (for Option A key parameter measurement). Combined uncertainty propagation using root-sum-square. Fractional savings uncertainty (FSU) at 68% and 90% confidence. Minimum detectable savings threshold. |
| 5 | IPMVP Option Engine | `ipmvp_option_engine.py` | `IPMVPOptionEngine` | Implementation of all four IPMVP options. Option A: Retrofit isolation with key parameter measurement (stipulated values for non-measured parameters). Option B: Retrofit isolation with all parameter measurement (short-term or continuous). Option C: Whole facility comparison using utility meters. Option D: Calibrated simulation using energy models (DOE-2, EnergyPlus calibration criteria). Automated option selection based on ECM type, measurement boundary, cost-effectiveness, and accuracy requirements. |
| 6 | Regression Engine | `regression_engine.py` | `RegressionEngine` | Statistical regression modeling for M&V baselines. OLS (Ordinary Least Squares) via normal equation. 3-parameter change-point models (3P cooling: flat + slope above change-point; 3P heating: flat + slope below change-point). 4-parameter model (heating slope + flat + cooling slope). 5-parameter model (heating slope + flat + cooling slope with two change-points). TOWT (Time-of-Week and Temperature) model for hourly/daily data. Model diagnostics: residual analysis, autocorrelation, heteroscedasticity, influential observations. |
| 7 | Weather Engine | `weather_engine.py` | `WeatherEngine` | Weather normalization for M&V calculations. HDD/CDD calculation with variable balance points (optimized via regression). TMY (Typical Meteorological Year) data for standard conditions. Degree-day regression for weather-dependent savings normalization. Balance point optimization using iterative regression (ASHRAE method). Weather data quality checks (completeness, range, consistency). Multi-source weather data reconciliation. |
| 8 | Metering Engine | `metering_engine.py` | `MeteringEngine` | M&V metering plan development and data management. Meter selection based on IPMVP option and measurement boundary. Calibration requirements and scheduling per ANSI C12.20. Sampling protocol design for Option A (sample size calculation using t-distribution). Data quality assessment (completeness, accuracy, consistency). Gap detection and acceptable gap-filling methods. Meter uncertainty specification and propagation. |
| 9 | Persistence Engine | `persistence_engine.py` | `PersistenceEngine` | Long-term savings persistence tracking and analysis. Year-over-year savings comparison with trend analysis. Degradation rate calculation (linear, exponential, step-change). Re-commissioning triggers when savings fall below thresholds. Performance guarantee tracking for ESCO/EPC contracts. Seasonal savings pattern analysis. Savings persistence factor calculation (actual/expected ratio). |
| 10 | MV Reporting Engine | `mv_reporting_engine.py` | `MVReportingEngine` | Comprehensive M&V report generation. M&V plan document per IPMVP template. Baseline report with regression results and model validation. Savings verification report with uncertainty bounds. Annual M&V summary with cumulative savings. Compliance checking against IPMVP, ISO 50015, FEMP, ASHRAE 14. Multi-format export (MD, HTML, JSON). Automated report scheduling. |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `baseline_development_workflow.py` | 4: DataCollection -> ModelSelection -> RegressionFitting -> Validation | Build and validate energy baseline with automated model selection |
| 2 | `mv_plan_workflow.py` | 4: ECMReview -> OptionSelection -> BoundaryDefinition -> MeteringPlan | Develop complete M&V plan per IPMVP |
| 3 | `option_selection_workflow.py` | 3: ECMCharacterization -> OptionEvaluation -> RecommendationReport | Select optimal IPMVP option for each ECM |
| 4 | `post_installation_workflow.py` | 4: Installation Verification -> MeterCommissioning -> ShortTermTest -> BaselineUpdate | Post-installation verification and meter commissioning |
| 5 | `savings_verification_workflow.py` | 4: DataCollection -> AdjustmentCalc -> SavingsCalc -> UncertaintyAnalysis | Complete savings verification for reporting period |
| 6 | `annual_reporting_workflow.py` | 3: DataAggregation -> ComplianceCheck -> ReportGeneration | Annual M&V report generation with compliance verification |
| 7 | `persistence_tracking_workflow.py` | 3: PerformanceMonitoring -> DegradationAnalysis -> AlertGeneration | Ongoing savings persistence monitoring |
| 8 | `full_mv_workflow.py` | 8: MVPlan -> BaselineDev -> PostInstall -> SavingsVerify -> Uncertainty -> Persistence -> Compliance -> Reporting | Master workflow orchestrating complete M&V lifecycle |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `mv_plan_report.py` | MD, HTML, JSON | M&V plan document per IPMVP: ECM description, IPMVP option, measurement boundary, baseline period, reporting period, metering plan, adjustment methodology |
| 2 | `baseline_report.py` | MD, HTML, JSON | Baseline analysis report: regression model results, model validation statistics, independent variable analysis, residual diagnostics, model selection rationale |
| 3 | `savings_report.py` | MD, HTML, JSON | Savings verification report: adjusted baseline vs. actual, avoided energy, cost savings, uncertainty bounds, cumulative savings |
| 4 | `uncertainty_report.py` | MD, HTML, JSON | Uncertainty analysis report: measurement uncertainty, model uncertainty, sampling uncertainty, combined FSU, minimum detectable savings |
| 5 | `annual_mv_report.py` | MD, HTML, JSON | Annual M&V summary: year-to-date savings, cumulative savings, trend analysis, compliance status, next steps |
| 6 | `option_comparison_report.py` | MD, HTML, JSON | IPMVP option comparison: suitability scores, cost-effectiveness analysis, accuracy trade-offs, recommendation with justification |
| 7 | `metering_plan_report.py` | MD, HTML, JSON | Metering plan: meter inventory, calibration schedule, sampling protocol, data management procedures |
| 8 | `persistence_report.py` | MD, HTML, JSON | Persistence tracking report: year-over-year savings trends, degradation analysis, persistence factors, re-commissioning recommendations |
| 9 | `executive_summary_report.py` | MD, HTML, JSON | 2-4 page executive summary: verified savings, financial impact, compliance status, risk assessment |
| 10 | `compliance_report.py` | MD, HTML, JSON | Standards compliance report: IPMVP checklist, ISO 50015 conformity, FEMP requirements, ASHRAE 14 statistical criteria |

### 3.5 Integrations

| # | Integration | File | Class | Purpose |
|---|-------------|------|-------|---------|
| 1 | Pack Orchestrator | `pack_orchestrator.py` | `MVOrchestrator` | 12-phase DAG pipeline with dependency resolution, parallel execution, retry with exponential backoff, SHA-256 provenance |
| 2 | MRV Bridge | `mrv_bridge.py` | `MRVBridge` | Routes verified energy savings to MRV agents for emissions reduction verification |
| 3 | Data Bridge | `data_bridge.py` | `DataBridge` | Routes to DATA agents for meter data, utility bills, quality profiling |
| 4 | PACK-031 Bridge | `pack031_bridge.py` | `Pack031Bridge` | Imports industrial energy audit baselines, ECM specifications, equipment data |
| 5 | PACK-032 Bridge | `pack032_bridge.py` | `Pack032Bridge` | Imports building energy assessment data, retrofit specifications |
| 6 | PACK-033 Bridge | `pack033_bridge.py` | `Pack033Bridge` | Imports quick win measures, estimated savings for verification |
| 7 | PACK-039 Bridge | `pack039_bridge.py` | `Pack039Bridge` | Imports real-time monitoring data, EnPI baselines, meter registry |
| 8 | Weather Service Bridge | `weather_service_bridge.py` | `WeatherServiceBridge` | Weather data for HDD/CDD calculation, TMY data, balance point optimization |
| 9 | Utility Data Bridge | `utility_data_bridge.py` | `UtilityDataBridge` | Utility billing data import, rate schedules, demand data |
| 10 | Health Check | `health_check.py` | `HealthCheck` | 20-category system health verification |
| 11 | Setup Wizard | `setup_wizard.py` | `SetupWizard` | 9-step guided M&V project configuration |
| 12 | Alert Bridge | `alert_bridge.py` | `AlertBridge` | Multi-channel alerting for savings degradation, compliance deadlines, report scheduling |

### 3.6 Presets

| # | Preset | Facility Type | Key Characteristics |
|---|--------|--------------|---------------------|
| 1 | `commercial_office.yaml` | Commercial Office | Weather-dependent (HVAC 40-55%), Option C whole-facility common, 3P/4P change-point models, HDD/CDD normalization |
| 2 | `manufacturing.yaml` | Manufacturing | Production-dependent, Option B retrofit isolation common, production normalization, multi-shift scheduling |
| 3 | `retail_portfolio.yaml` | Retail Chain | Multi-site portfolio M&V, Option C with weather + sales normalization, portfolio-level sampling |
| 4 | `hospital.yaml` | Healthcare | 24/7 operation, Option B for specific systems, high baseline stability requirement, steam/chilled water metering |
| 5 | `university_campus.yaml` | University Campus | Multi-building, Option C per building, academic calendar adjustments, central plant M&V |
| 6 | `government_femp.yaml` | Government (FEMP) | FEMP M&V Guidelines 4.0 compliance, ESPC contract verification, federal reporting requirements |
| 7 | `esco_performance_contract.yaml` | ESCO/EPC | Performance guarantee verification, multi-year tracking, shared savings calculations, dispute resolution |
| 8 | `portfolio_mv.yaml` | Multi-Site Portfolio | Statistical sampling across sites, portfolio-level uncertainty, aggregated savings reporting |

### 3.7 Database Migrations (V316-V325)

| Version | File | Description |
|---------|------|-------------|
| V316 | `V316__pack040_mv_001.sql` | M&V projects, ECM registry, measurement boundaries |
| V317 | `V317__pack040_mv_002.sql` | Baseline models, regression parameters, model validation |
| V318 | `V318__pack040_mv_003.sql` | Adjustment records, routine and non-routine |
| V319 | `V319__pack040_mv_004.sql` | Savings calculations, verified savings, cost savings |
| V320 | `V320__pack040_mv_005.sql` | Uncertainty analysis, measurement and model errors |
| V321 | `V321__pack040_mv_006.sql` | IPMVP option records, option selection rationale |
| V322 | `V322__pack040_mv_007.sql` | Metering plans, calibration records, sampling protocols |
| V323 | `V323__pack040_mv_008.sql` | Persistence tracking, degradation records, alerts |
| V324 | `V324__pack040_mv_009.sql` | M&V reports, compliance records, report schedules |
| V325 | `V325__pack040_mv_010.sql` | Views, indexes, audit trail, seed data |

---

## 4. Engine Specifications

### 4.1 Engine 1: Baseline Engine

**Purpose:** Develop statistically valid energy baselines using multivariate regression with change-point models per IPMVP and ASHRAE Guideline 14.

**Baseline Model Types:**
| Model | Equation | Use Case |
|-------|----------|----------|
| Simple linear | E = a + b*X | Single variable (e.g., production only) |
| Multivariate linear | E = a + b1*X1 + b2*X2 + ... | Multiple independent variables |
| 3P Cooling | E = a + b*max(0, T-Tcp) | Cooling-dominant buildings above change-point |
| 3P Heating | E = a + b*max(0, Thp-T) | Heating-dominant buildings below change-point |
| 4P | E = a + bh*max(0,Thp-T) + bc*max(0,T-Tcp) | Both heating and cooling with same base |
| 5P | E = a + bh*max(0,Thp-T) + bc*max(0,T-Tcp) | Separate heating and cooling change-points |
| TOWT | E = f(TOW, T) | Time-of-week and temperature (hourly/daily) |

**Model Validation Criteria (ASHRAE 14):**
| Criterion | Monthly | Daily | Hourly |
|-----------|---------|-------|--------|
| CVRMSE | <25% | <30% | <30% |
| NMBE | +/-5% | +/-10% | +/-10% |
| R-squared | >0.70 | >0.50 | N/A |

**Key Models:**
- `BaselineConfig` - Project ID, baseline period, independent variables, model type preference, validation criteria
- `BaselineModel` - Regression coefficients, change-points, statistics, residuals, validation results
- `ModelComparison` - Side-by-side model results for automated selection
- `BaselineResult` - Selected model, validation status, provenance hash

### 4.2 Engine 2: Adjustment Engine

**Purpose:** Calculate routine and non-routine adjustments per IPMVP methodology.

**Routine Adjustments:**
| Type | Method | Example |
|------|--------|---------|
| Weather | HDD/CDD normalization using regression model | Adjust for warmer/cooler reporting period |
| Production | Production-normalized EnPI | Adjust for different production volume |
| Occupancy | Occupancy-weighted adjustment | Adjust for changed occupancy patterns |
| Operating hours | Hours-proportional scaling | Adjust for extended/reduced operating hours |

**Non-Routine Adjustments:**
| Type | Method | Example |
|------|--------|---------|
| Floor area change | Proportional EUI scaling | Building expansion/contraction |
| Equipment addition | Engineering estimate of new load | New server room, new production line |
| Equipment removal | Engineering estimate of removed load | Decommissioned equipment |
| Schedule change | Delta between old/new schedules | Changed operating hours |
| Static factor | Fixed value applied to baseline | Known one-time change |

### 4.3 Engine 3: Savings Engine

**Savings Calculation Methods:**
- Avoided energy = Adjusted baseline prediction - Reporting period actual
- Normalized savings = Baseline at standard conditions - Reporting period at standard conditions
- Cost savings = Energy savings * blended rate + demand savings * demand rate
- Cumulative savings = Sum of periodic savings over contract/tracking period
- Annualized savings = Periodic savings scaled to full year

### 4.4 Engine 4: Uncertainty Engine

**Uncertainty Components (ASHRAE 14):**
- Measurement uncertainty: Meter accuracy class, CT/PT errors, calibration drift
- Model uncertainty: Regression standard error, prediction interval width
- Sampling uncertainty: Sample size, coefficient of variation, t-distribution
- Combined uncertainty: Root-sum-square of independent components
- Fractional savings uncertainty: FSU = (combined uncertainty / savings) * 100%

**Key Formulas:**
- FSU = (t * sqrt(n+2/n) * CVRMSE * Ebaseline) / (Ebaseline - Eactual) * 100%
- Minimum detectable savings = 2 * standard error of estimate
- Required sample size (Option A) = (t * CV / precision)^2

### 4.5 Engine 5: IPMVP Option Engine

**Option Comparison Matrix:**
| Aspect | Option A | Option B | Option C | Option D |
|--------|----------|----------|----------|----------|
| Measurement boundary | Retrofit isolation | Retrofit isolation | Whole facility | Whole facility |
| Parameter measurement | Key parameter(s) | All parameters | Utility meters | Simulation |
| Stipulated values | Yes (non-measured) | No | No | N/A |
| Best for | Single systems | Complex retrofits | Multiple ECMs | New construction |
| Typical cost | Low | Medium-High | Low-Medium | High |
| Typical accuracy | Medium | High | Medium | High |

### 4.6 Engine 6: Regression Engine

**Regression Methods:**
- OLS via normal equation: beta = (X'X)^(-1) X'y
- 3P change-point: Iterative optimization of change-point temperature
- 4P/5P models: Grid search or Nelder-Mead for optimal change-points
- TOWT: Combined time-of-week indicator variables with temperature regression
- Diagnostics: Residual plots, QQ plots, Durbin-Watson, Cook's distance, VIF

### 4.7 Engine 7: Weather Engine

**Weather Normalization:**
- HDD = max(0, Tbase - Tavg) for each day
- CDD = max(0, Tavg - Tbase) for each day
- Balance point optimization: Iterate Tbase to maximize R-squared
- TMY normalization: Scale savings to typical year weather conditions
- Degree-day regression: E = a + b_h*HDD + b_c*CDD

### 4.8 Engine 8: Metering Engine

**Metering Plan Components:**
- Meter selection matrix (IPMVP option vs. measurement point)
- Calibration schedule per ANSI C12.20 (revenue meters: +/-0.2%, utility: +/-0.5%)
- Sampling protocol for Option A (90% confidence, +/-10% precision)
- Data management: collection frequency, storage, quality checks, gap handling

### 4.9 Engine 9: Persistence Engine

**Persistence Metrics:**
- Persistence factor = Actual savings Year N / Expected savings Year N
- Degradation rate = (Year 1 savings - Year N savings) / (N-1) per year
- Exponential decay: S(t) = S0 * exp(-lambda * t)
- Step-change detection for sudden performance drops
- Re-commissioning trigger: Persistence factor < 0.80

### 4.10 Engine 10: MV Reporting Engine

**Report Types:**
- M&V Plan (pre-retrofit documentation)
- Baseline Report (model development and validation)
- Post-Installation Report (installation verification)
- Savings Verification Report (periodic savings with uncertainty)
- Annual M&V Summary (year-end comprehensive)
- Persistence Report (multi-year tracking)
- Compliance Report (standards conformity)

---

## 5. Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| Baseline regression | <15 seconds | 3 years daily data, 5P model |
| Savings calculation | <5 seconds | Per ECM per reporting period |
| Uncertainty analysis | <10 seconds | Full ASHRAE 14 propagation |
| Report generation | <10 seconds | Per report |
| Full M&V workflow | <5 minutes | Single project, all phases |

---

## 6. Security

| Requirement | Implementation |
|-------------|---------------|
| Multi-tenant isolation | PostgreSQL RLS policies |
| Encryption at rest | AES-256-GCM |
| Provenance | SHA-256 hash on all outputs |
| RBAC | Role-based access per platform policy |
| Audit trail | Full calculation lineage |

---

## 7. Testing

| Category | Target |
|----------|--------|
| Framework | pytest |
| Minimum coverage | 85% |
| Estimated tests | 850+ |
| Test types | Unit, integration, regression, statistical validation |
| ASHRAE 14 validation | Cross-validated against published example problems |
| IPMVP compliance | Automated compliance checklist verification |
