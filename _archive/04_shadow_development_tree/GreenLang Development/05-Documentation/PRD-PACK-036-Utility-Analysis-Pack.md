# PRD-PACK-036: Utility Analysis Pack

**Pack ID:** PACK-036-utility-analysis
**Category:** Energy Efficiency Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-22
**Prerequisite:** None (standalone; enhanced with PACK-031 Industrial Energy Audit Pack, PACK-032 Building Energy Assessment Pack, PACK-033 Quick Wins Identifier Pack, and PACK-035 Energy Benchmark Pack if present)

---

## 1. Executive Summary

### 1.1 Problem Statement

Utility costs represent one of the largest controllable operating expenses for commercial and industrial organisations, typically accounting for EUR 20-500/m2/yr depending on facility type. Without systematic analysis of utility bills, rate structures, demand profiles, and procurement options, organisations routinely overpay by 5-15% through billing errors, suboptimal rate structures, poor demand management, and missed procurement opportunities. The EU Energy Efficiency Directive (EED) 2023/1791 mandates transparent billing and cost information, yet most organisations struggle to gain complete visibility into their utility spend. Key challenges include:

1. **Billing errors go undetected**: Studies consistently show that 1-3% of utility bills contain errors -- estimated reads mistakenly applied, wrong rate schedules, duplicate charges, incorrect meter multipliers, and miscalculated taxes. For a portfolio of 50 buildings with EUR 10M annual utility spend, undetected billing errors represent EUR 100,000-300,000 in annual overcharges. Manual bill review catches less than 20% of errors because reviewers lack the tools to compare line items against published tariff schedules, validate meter read sequences, or detect statistical anomalies across billing history.

2. **Suboptimal rate structures**: Utility tariffs have become increasingly complex, with time-of-use (TOU), critical peak pricing (CPP), real-time pricing (RTP), demand ratchet, declining block, interruptible, and green/renewable tariff options available in most jurisdictions. Selecting the wrong rate structure can cost 5-15% more than the optimal tariff. Yet most organisations remain on their initial rate schedule for years without performing comparative analysis. Rate switching requires modelling historical consumption under alternative tariffs -- a calculation that involves disaggregating consumption by TOU period, modelling demand charges under different ratchet clauses, and projecting costs across seasonal and annual cycles.

3. **Demand charges are poorly managed**: For commercial and industrial facilities, demand charges (based on peak kW or kVA) can represent 30-70% of the electricity bill. A single 15-minute demand spike during the year can establish a demand ratchet that inflates billed demand for 11 subsequent months. Most organisations do not monitor interval data (15-minute or 30-minute readings) and therefore cannot identify the specific events driving demand charges. Without load profiling, load factor analysis, and coincident peak identification, demand reduction strategies (load shifting, peak shaving, demand response) cannot be properly evaluated.

4. **Cost allocation is inaccurate**: Multi-tenant buildings, campus facilities, and manufacturing sites need to allocate utility costs to departments, tenants, processes, or cost centres. Without sub-metering (or where sub-metering is incomplete), organisations resort to floor area proration that ignores actual consumption patterns. A data centre tenant consuming 300 W/m2 may be charged the same per-m2 rate as an administrative office consuming 30 W/m2. Inaccurate cost allocation fails to incentivise energy efficiency and creates disputes between tenants and landlords.

5. **Budget forecasts are unreliable**: Utility budgets are typically based on simple extrapolation of prior-year costs, ignoring weather variability, rate escalation, planned efficiency measures, and occupancy changes. Budget variance of 10-20% is common, leading to either over-budgeting (tying up capital) or under-budgeting (requiring emergency approvals). Weather normalisation using degree-day regression can reduce forecast error to 3-5%, but requires sophisticated statistical modelling per ASHRAE Guideline 14-2014 and ISO 50006:2014.

6. **Procurement decisions lack data**: Energy procurement is increasingly complex, with deregulated markets offering fixed-price, index-linked, block-and-index, and shaped supply contracts. Organisations making procurement decisions without market intelligence, load shape analysis, and risk quantification (VaR/CVaR) routinely overpay by 3-8%. Green procurement (PPAs, RECs, GOs) adds further complexity as organisations pursue 100% renewable targets while managing cost premiums.

7. **Benchmarking is superficial**: Organisations want to compare utility costs across facilities and against peers, but raw cost comparison is misleading without normalisation for building type, climate zone, operating hours, and utility rate differences. Weather-normalised EUI (Energy Use Intensity) and cost-per-m2 benchmarks from ENERGY STAR, CIBSE TM46, and BOMA provide meaningful comparison, but calculating these metrics requires integration of consumption, weather, area, and rate data from multiple sources.

8. **Regulatory charges are opaque**: Non-commodity charges (transmission, distribution, capacity, renewable obligation, carbon levies) can represent 40-60% of the electricity bill in jurisdictions like Germany (EEG-Umlage, StromNEV, KWK-Umlage), the UK (TNUoS, DUoS, BSUoS, RO, CfD, CM), and the US (RPS, SBC). Energy-intensive industries may be eligible for exemptions and reductions that save 10-30% on these charges, but eligibility assessment requires detailed analysis of consumption patterns, sector classification, and regulatory thresholds.

### 1.2 Solution Overview

PACK-036 is the **Utility Analysis Pack** -- the sixth pack in the "Energy Efficiency Packs" category, complementing PACK-031 (Industrial Energy Audit), PACK-032 (Building Energy Assessment), PACK-033 (Quick Wins Identifier), PACK-034 (ISO 50001 EnMS), and PACK-035 (Energy Benchmark). While other packs focus on physical energy auditing and management systems, PACK-036 focuses specifically on the financial and commercial aspects of utility management.

The pack provides 10 calculation engines covering the complete utility management lifecycle:

1. **Utility Bill Parser** -- Automated extraction from PDF, CSV, Excel, and EDI formats with line-item disaggregation and billing error detection
2. **Rate Structure Analyzer** -- Tariff modelling for TOU, CPP, RTP, demand ratchet, and declining block rates with optimal rate identification
3. **Demand Analysis** -- Peak demand profiling, load factor calculation, load duration curves, and demand response opportunity assessment
4. **Cost Allocation** -- Multi-method allocation (sub-metered, floor area, activity-based, hybrid) with tenant chargeback generation
5. **Budget Forecasting** -- Weather-normalised forecasts with Monte Carlo uncertainty quantification and multi-scenario analysis
6. **Procurement Intelligence** -- Market price tracking, contract analysis, VaR/CVaR risk assessment, and PPA evaluation
7. **Utility Benchmark** -- Cost-per-m2, EUI comparison, ENERGY STAR and CIBSE TM46 benchmarks with portfolio ranking
8. **Regulatory Charge Optimizer** -- Non-commodity charge analysis, exemption eligibility assessment, and charge reduction strategies
9. **Weather Normalization** -- 2P-5P degree-day regression models with ASHRAE 14 validation criteria
10. **Utility Reporting** -- Multi-format report generation (MD/HTML/JSON/CSV/PDF) with executive dashboards

The pack also includes 8 workflows, 10 report templates, 12 integrations, 8 facility-type presets, and 10 database migrations (V276-V285).

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Consultant Approach | PACK-036 Utility Analysis Pack |
|-----------|------------------------------|-------------------------------|
| Bill auditing | Manual review (catches <20% of errors) | Automated parsing with 17 error detection rules (catches 80-95% of errors) |
| Rate optimisation | Annual consultant study (EUR 5-15K per facility) | Continuous automated analysis across all available tariffs |
| Demand management | Reactive peak response | Proactive profiling with 10 demand reduction strategies modelled |
| Cost allocation | Floor area proration (30-50% error) | Multi-method allocation with reconciliation check (5-10% error) |
| Budget accuracy | +/- 15-20% variance | +/- 3-5% with weather normalisation and Monte Carlo P10/P50/P90 |
| Procurement | Relationship-based, price-taker | Data-driven with VaR risk assessment and market benchmarking |
| Benchmarking | Raw cost comparison (misleading) | Weather-normalised, building-type-adjusted peer benchmarking |
| Regulatory charges | Accepted as given | Analysed for exemption eligibility and optimisation (10-30% reduction potential) |
| Report production | Manual (40-80 hours per portfolio) | Automated (4-8 hours per portfolio, 10x faster) |
| Audit trail | Spreadsheet-based | SHA-256 provenance, full calculation lineage, digital audit trail |

### 1.4 Target Users

**Primary:**
- Energy managers responsible for utility cost management and reporting
- Facilities managers seeking to reduce operating costs through utility optimisation
- Finance/procurement teams managing utility budgets and supply contracts
- Property/portfolio managers benchmarking utility costs across building estates

**Secondary:**
- Sustainability officers linking utility costs to carbon reduction and EU Taxonomy reporting
- Building owners/landlords performing tenant cost allocation and chargeback
- Consultants providing utility analysis services to multiple clients
- CFOs requiring accurate utility budget forecasts and variance analysis

### 1.5 Target Sectors

| # | Sector | Typical Utility Cost (EUR/m2/yr) | Priority Analysis Areas |
|---|--------|----------------------------------|------------------------|
| 1 | Office Buildings | 20-60 | TOU rate optimisation, tenant allocation, demand management |
| 2 | Manufacturing | 30-150 | Demand charge optimisation, process allocation, EII exemptions |
| 3 | Retail Stores | 25-80 | Multi-site benchmarking, portfolio procurement, seasonal demand |
| 4 | Warehouses | 10-40 | Lighting cost analysis, cold storage demand, flat rate comparison |
| 5 | Healthcare | 40-100 | 24/7 cost analysis, steam billing, departmental allocation |
| 6 | Education | 15-50 | Term-time analysis, campus allocation, holiday shutdown savings |
| 7 | Data Centers | 100-500 | Demand optimisation, PUE-based allocation, renewable PPA |
| 8 | Multi-Site Portfolios | Varies | Cross-site benchmarking, consolidated procurement, portfolio ranking |

---

## 2. Regulatory Basis

### 2.1 EU Energy Efficiency Directive (EED) -- Directive (EU) 2023/1791

The recast EED mandates transparent and accurate utility billing (Article 10a) and cost information (Article 11) that enables consumers to make informed energy efficiency decisions. PACK-036 supports EED compliance by providing:
- Granular bill parsing with line-item disaggregation for billing transparency
- Rate comparison analysis enabling informed tariff switching decisions
- Weather-normalised benchmarking for fair performance comparison
- Budget forecasting with savings quantification to justify efficiency investments

### 2.2 ISO 50001:2018 -- Energy Management Systems

ISO 50001 requires organisations to monitor, measure, analyse, and evaluate energy performance including energy procurement (Clause 8.3). PACK-036 supports:
- Energy review (Clause 6.3) through detailed consumption and cost analysis
- Energy baseline establishment (Clause 6.5) through weather-normalised baselines
- Energy performance indicators (Clause 6.4) through cost-based EnPIs
- Energy procurement specifications (Clause 8.3) through procurement intelligence

### 2.3 ISO 50006:2014 -- Energy Performance Indicators

PACK-036 implements ISO 50006 regression models for weather normalisation and production normalisation, enabling accurate period-over-period performance comparison independent of weather and operational variations.

### 2.4 ASHRAE Guideline 14-2014

PACK-036 implements ASHRAE Guideline 14 utility bill analysis methodology (Option C -- Whole Building) including:
- Weather normalisation using degree-day regression (2P through 5P models)
- Model validation using CV(RMSE) < 25% monthly, |NMBE| < 0.5% criteria
- Savings quantification with uncertainty bounds per ASHRAE 14 Section 4

### 2.5 ENERGY STAR Portfolio Manager

PACK-036 integrates utility cost benchmarking compatible with ENERGY STAR Portfolio Manager property types, enabling EUI and cost-per-area comparison against US commercial building stock.

### 2.6 CIBSE TM46:2008

PACK-036 incorporates CIBSE TM46 energy benchmarks by building type for UK-context peer comparison, converting energy benchmarks to cost benchmarks using applicable tariff rates.

### 2.7 EU Taxonomy Regulation 2020/852

Energy efficiency investments identified through utility analysis may qualify as substantial contributions to climate change mitigation under the EU Taxonomy. PACK-036 provides the financial evidence for Taxonomy-aligned reporting.

---

## 3. Technical Architecture

### 3.1 Engine Architecture

All 10 engines follow the standard GreenLang engine pattern:

```
Engine (Pydantic v2 Models + Pure Functions)
    |
    +-- Input Models (Pydantic BaseModel with Field validators)
    +-- Output Models (Pydantic BaseModel with provenance_hash)
    +-- Enum Definitions (str, Enum for type safety)
    +-- Reference Data (deterministic lookup tables)
    +-- Calculation Methods (pure functions, Decimal arithmetic)
    +-- Provenance Hashing (SHA-256 on all result objects)
```

**Common Helpers** (present in every engine):
- `_utcnow()` -- UTC timestamp generation
- `_new_uuid()` -- UUID4 generation
- `_compute_hash()` -- SHA-256 provenance hashing
- `_decimal()` -- Decimal conversion with ROUND_HALF_UP
- `_safe_divide()` -- Division with zero-denominator protection
- `_round_val()` -- Configurable decimal place rounding

### 3.2 Zero-Hallucination Guarantee

Every calculation path is deterministic:
- **No LLM** in any calculation path
- **Decimal arithmetic** for all financial calculations (no floating-point)
- **Published reference data** only (tariff schedules, emission factors, benchmarks)
- **SHA-256 provenance** on every result for audit trail integrity
- **Bit-perfect reproducibility** across runs with identical inputs

### 3.3 Data Flow

```
Utility Bills (PDF/CSV/Excel/EDI)
    |
    v
[Engine 1: Bill Parser] --> Parsed bills with line items
    |
    +---> [Engine 2: Rate Analyzer] --> Rate comparison & optimal tariff
    +---> [Engine 3: Demand Analysis] --> Load profiles & peak events
    +---> [Engine 8: Reg Charge Optimizer] --> Charge analysis & exemptions
    |
    v
[Engine 4: Cost Allocation] --> Departmental/tenant allocations
    |
    v
[Engine 9: Weather Normalization] --> Weather-adjusted baselines
    |
    v
[Engine 5: Budget Forecasting] --> P10/P50/P90 budget projections
    |
    v
[Engine 6: Procurement Intelligence] --> Market data & contract analysis
    |
    v
[Engine 7: Utility Benchmark] --> Peer comparison & portfolio ranking
    |
    v
[Engine 10: Utility Reporting] --> MD/HTML/JSON/CSV/PDF reports
```

---

## 4. Engine Specifications

### 4.1 Engine 1: Utility Bill Parser Engine

**Class:** `UtilityBillParserEngine`
**File:** `engines/utility_bill_parser_engine.py`
**Lines:** ~2,504

**Purpose:** Parses and normalises utility bills from multiple formats and utility types with line-item disaggregation and billing error detection.

**Enums (7):**
- `BillFormat` -- PDF, CSV, EXCEL, EDI_810, EDI_867, XML, JSON
- `UtilityType` -- ELECTRICITY, NATURAL_GAS, WATER, STEAM, CHILLED_WATER, DISTRICT_HEATING, PROPANE, FUEL_OIL
- `ChargeCategory` -- ENERGY, DEMAND, FIXED, TAX, SURCHARGE, REGULATORY, CREDIT, ADJUSTMENT
- `BillStatus` -- DRAFT, PARSED, VALIDATED, ERROR_FLAGGED, AUDITED, CORRECTED, APPROVED, DISPUTED
- `ReadType` -- ACTUAL, ESTIMATED, CUSTOMER_READ, SMART_METER, CHECK_READ
- `AnomalyType` -- HIGH_CONSUMPTION, LOW_CONSUMPTION, ESTIMATED_READ, DUPLICATE_CHARGE, RATE_MISMATCH, PERIOD_OVERLAP, NEGATIVE_CONSUMPTION, UNUSUAL_DEMAND
- `ValidationSeverity` -- INFO, WARNING, ERROR, CRITICAL

**Models (8):** `BillInput`, `BillLineItem`, `MeterReading`, `BillAnomaly`, `BillValidation`, `ParsedBill`, `BillAuditResult`, `BatchParseResult`

**Key Methods:**
| Method | Description |
|--------|-------------|
| `parse_bill()` | Parse single bill from raw input with format auto-detection |
| `parse_batch()` | Parse multiple bills with parallel processing |
| `validate_bill()` | Validate parsed bill against physical constraints |
| `detect_anomalies()` | Statistical anomaly detection across billing history |
| `recalculate_bill()` | Recalculate bill charges from consumption and rate schedule |
| `verify_taxes()` | Verify tax calculations against applicable tax rates |
| `detect_duplicates()` | Identify duplicate charges across bills |
| `check_meter_sequence()` | Validate meter read sequence continuity |

### 4.2 Engine 2: Rate Structure Analyzer Engine

**Class:** `RateStructureAnalyzerEngine`
**File:** `engines/rate_structure_analyzer_engine.py`
**Lines:** ~1,853

**Purpose:** Models and analyses utility rate structures to identify optimal tariff selection and switching opportunities across 500+ tariff types.

**Enums (6):** `RateType`, `TierType`, `TOUPeriod`, `SeasonType`, `DemandRatchetType`, `RateStatus`

**Models (12):** `RateSchedule`, `RateTier`, `TOUSchedule`, `DemandCharge`, `RateComponent`, `RateComparisonInput`, `RateComparisonResult`, `BillRecalculation`, `SwitchRecommendation`, `RatchetAnalysis`, `BlendedRate`, `RateOptimizationResult`

**Key Methods:**
| Method | Description |
|--------|-------------|
| `model_rate()` | Model a complete rate schedule with all components |
| `recalculate_under_rate()` | Recalculate historical bills under alternative rate |
| `compare_rates()` | Side-by-side comparison of multiple rate structures |
| `find_optimal_rate()` | Identify lowest-cost rate from available options |
| `calculate_blended_rate()` | Calculate effective blended unit rate |
| `analyse_demand_ratchet()` | Quantify demand ratchet exposure and cost impact |
| `model_tou_savings()` | Model savings from TOU period alignment |

### 4.3 Engine 3: Demand Analysis Engine

**Class:** `DemandAnalysisEngine`
**File:** `engines/demand_analysis_engine.py`
**Lines:** ~2,305

**Purpose:** Analyses electrical demand profiles from 15-minute interval data to identify peak demand reduction and demand management opportunities.

**Enums (6):** `IntervalResolution`, `PeakType`, `LoadShape`, `DemandStrategy`, `PowerFactorStatus`, `SeasonalPeak`

**Models (14):** `IntervalRecord`, `DemandProfile`, `PeakEvent`, `LoadDurationCurve`, `LoadFactorResult`, `CoincidentPeak`, `DemandRatchetExposure`, `PowerFactorAnalysis`, `DemandReductionOpportunity`, `DRProgramEligibility`, `DemandChargeSavings`, `LoadShiftScenario`, `PeakShavingResult`, `DemandAnalysisResult`

**Key Methods:**
| Method | Description |
|--------|-------------|
| `build_profile()` | Build complete demand profile from interval data |
| `calculate_load_factor()` | Average demand / peak demand ratio |
| `generate_load_duration_curve()` | Percentage of time at each demand level |
| `identify_peak_events()` | Find top-N demand events with timing and context |
| `analyse_coincident_peak()` | Compare facility peak vs. utility system peak |
| `assess_power_factor()` | Calculate PF and correction recommendations |
| `model_load_shifting()` | Model demand charge savings from load shifting |
| `model_peak_shaving()` | Model battery/thermal storage peak shaving |
| `evaluate_demand_response()` | Assess DR programme eligibility and revenue |

### 4.4 Engine 4: Cost Allocation Engine

**Class:** `CostAllocationEngine`
**File:** `engines/cost_allocation_engine.py`
**Lines:** ~2,139

**Purpose:** Allocates utility costs to departments, tenants, processes, and cost centres using multiple allocation methods with reconciliation.

**Enums (6):** `AllocationMethod`, `EntityType`, `CostPool`, `ReconciliationStatus`, `ChargebackFormat`, `DistributionBasis`

**Models (9):** `AllocationEntity`, `AllocationRule`, `MeterMapping`, `CostComponent`, `AllocationResult`, `ChargebackInvoice`, `ReconciliationReport`, `GiniCoefficient`, `CostAllocationResult`

**Key Methods:**
| Method | Description |
|--------|-------------|
| `allocate_by_meter()` | Direct allocation from sub-meter readings |
| `allocate_by_area()` | Proportional allocation by floor area |
| `allocate_by_activity()` | Activity-based costing allocation |
| `allocate_hybrid()` | Combined metered + estimated allocation |
| `distribute_common_area()` | Allocate shared services costs |
| `generate_chargeback()` | Produce tenant chargeback invoices |
| `reconcile()` | Compare sum of allocations vs. main meter |
| `calculate_gini()` | Measure allocation fairness with Gini coefficient |

### 4.5 Engine 5: Budget Forecasting Engine

**Class:** `BudgetForecastingEngine`
**File:** `engines/budget_forecasting_engine.py`
**Lines:** ~2,141

**Purpose:** Generates 12-36 month utility budget forecasts with weather normalisation, Monte Carlo uncertainty quantification, and multi-scenario analysis.

**Enums (7):** `ForecastMethod`, `ScenarioType`, `ConfidenceLevel`, `EscalationMethod`, `BudgetPeriod`, `VarianceCategory`, `MonteCarloSeed`

**Models (10):** `HistoricalPeriod`, `ForecastInput`, `MonthlyForecast`, `ScenarioResult`, `ConfidenceInterval`, `VarianceDecomposition`, `BudgetLine`, `RateEscalation`, `MonteCarloBatch`, `BudgetForecastResult`

**Key Methods:**
| Method | Description |
|--------|-------------|
| `forecast_linear_trend()` | Linear trend extrapolation |
| `forecast_seasonal()` | Seasonal decomposition forecast |
| `forecast_weather_adjusted()` | Degree-day regression forecast |
| `forecast_monte_carlo()` | 1000+ iteration Monte Carlo simulation |
| `forecast_ensemble()` | Weighted ensemble of multiple methods |
| `apply_rate_escalation()` | Apply annual rate escalation factors |
| `apply_scenario_adjustments()` | Apply planned changes (efficiency, occupancy) |
| `decompose_variance()` | Weather / rate / volume / efficiency variance split |
| `generate_budget()` | Produce complete budget package with P10/P50/P90 |

### 4.6 Engine 6: Procurement Intelligence Engine

**Class:** `ProcurementIntelligenceEngine`
**File:** `engines/procurement_intelligence_engine.py`
**Lines:** ~2,354

**Purpose:** Market intelligence and decision support for utility procurement including VaR/CVaR risk assessment and hedging strategy evaluation.

**Enums (7):** `ContractType`, `MarketRegion`, `PricingStructure`, `HedgingStrategy`, `RECType`, `RiskMetric`, `ProcurementStatus`

**Models (15):** `MarketPrice`, `ForwardCurve`, `ContractTerms`, `SupplierOffer`, `HedgingResult`, `VaRResult`, `CVaRResult`, `PPAAssessment`, `RECPricing`, `RFPSpecification`, `SupplierComparison`, `ProcurementScenario`, `RiskProfile`, `GreenProcurement`, `ProcurementResult`

**Key Methods:**
| Method | Description |
|--------|-------------|
| `track_market_prices()` | Monitor wholesale electricity and gas prices |
| `analyse_contract()` | Evaluate existing supply contract terms |
| `compare_suppliers()` | Side-by-side normalised supplier comparison |
| `calculate_var()` | Value at Risk for procurement cost exposure |
| `calculate_cvar()` | Conditional VaR (expected shortfall) |
| `evaluate_hedging()` | Compare fixed vs. floating vs. hybrid strategies |
| `assess_ppa()` | Evaluate physical and virtual PPA opportunities |
| `generate_rfp()` | Create standardised RFP with usage profiles |
| `analyse_green_tariffs()` | Compare green/renewable tariff options |

### 4.7 Engine 7: Utility Benchmark Engine

**Class:** `UtilityBenchmarkEngine`
**File:** `engines/utility_benchmark_engine.py`
**Lines:** ~2,069

**Purpose:** Benchmarks utility costs and consumption against peers, published standards, and historical performance using ENERGY STAR, CIBSE TM46, and ASHRAE 100 methodologies.

**Enums (7):** `BenchmarkStandard`, `MetricType`, `NormalizationMethod`, `PeerGroupType`, `QuartileRank`, `TrendDirection`, `OutlierMethod`

**Models (11):** `FacilityMetrics`, `PeerGroup`, `BenchmarkTarget`, `BenchmarkResult`, `PortfolioRanking`, `CostOutlier`, `TrendAnalysis`, `PerformanceGap`, `ImprovementTarget`, `CrossSiteComparison`, `BenchmarkAnalysisResult`

**Key Methods:**
| Method | Description |
|--------|-------------|
| `calculate_eui()` | Site and source EUI with weather normalisation |
| `calculate_cost_intensity()` | Cost per m2 by utility type |
| `benchmark_against_standard()` | Compare against ENERGY STAR, CIBSE TM46 |
| `rank_portfolio()` | Rank facilities by cost efficiency metrics |
| `identify_outliers()` | IQR and Z-score outlier detection |
| `analyse_trends()` | Year-over-year and month-over-month trending |
| `quantify_gap()` | Savings potential from gap to benchmark target |
| `compare_cross_site()` | Normalised cross-site comparison |

### 4.8 Engine 8: Regulatory Charge Optimizer Engine

**Class:** `RegulatoryChargeOptimizerEngine`
**File:** `engines/regulatory_charge_optimizer_engine.py`
**Lines:** ~2,225

**Purpose:** Analyses non-commodity charges on utility bills and identifies exemption eligibility and charge reduction strategies for Germany, UK, and US regulatory frameworks.

**Enums (6):** `ChargeType`, `JurisdictionType`, `ExemptionType`, `OptimizationAction`, `ComplianceStatus`, `ChargeCategory`

**Models (10):** `RegulatoryCharge`, `ChargeBreakdown`, `ExemptionAssessment`, `ExemptionCriteria`, `OptimizationOpportunity`, `TRIADManagement`, `CapacityCharge`, `NetworkCharge`, `RegulatoryForecast`, `RegulatoryOptimizationResult`

**Reference Data:**
- German charges: EEG-Umlage, StromNEV, KWK-Umlage, Offshore-Umlage, Stromsteuer, Konzessionsabgabe
- UK charges: TNUoS, DUoS, BSUoS, RO, CfD, CM, FiT, CCL, CPS
- US charges: RPS, SBC, nuclear decommissioning, RGGI, demand response surcharge

### 4.9 Engine 9: Weather Normalization Engine

**Class:** `WeatherNormalizationEngine`
**File:** `engines/weather_normalization_engine.py`
**Lines:** ~2,257

**Purpose:** Normalises utility consumption for weather effects using degree-day regression models validated against ASHRAE Guideline 14 criteria.

**Regression Models (7):**
| Model | Parameters | Use Case |
|-------|-----------|----------|
| SIMPLE_HDD | 2 (slope, intercept) | Heating-only buildings |
| SIMPLE_CDD | 2 (slope, intercept) | Cooling-only buildings |
| HDD_CDD | 3 (HDD slope, CDD slope, intercept) | Dual-fuel buildings |
| 3P_HEATING | 3 (base load, slope, change point) | Heating with base load |
| 3P_COOLING | 3 (base load, slope, change point) | Cooling with base load |
| 4P | 4 (base, heating slope, cooling slope, change point) | Combined with single change point |
| 5P | 5 (base, heating slope, cooling slope, 2 change points) | Full model |

**ASHRAE 14 Validation Criteria:**
- R-squared >= 0.70
- CV(RMSE) <= 25% (monthly data)
- |NMBE| <= 0.5%
- Durbin-Watson statistic for autocorrelation

**Key Methods:**
| Method | Description |
|--------|-------------|
| `calculate_degree_days()` | Monthly HDD/CDD from daily weather data |
| `fit_regression()` | Simple linear HDD/CDD regression |
| `fit_change_point_model()` | 3P/4P/5P models via grid search + golden-section |
| `validate_model()` | ASHRAE 14 fit criteria validation |
| `select_best_model()` | Auto-select best model from all 7 types |
| `normalize_consumption()` | Weather-normalise to TMY or long-term average |
| `quantify_weather_impact()` | Decompose changes into weather vs. operational |
| `project_climate_impact()` | RCP/SSP climate scenario projections |

### 4.10 Engine 10: Utility Reporting Engine

**Class:** `UtilityReportingEngine`
**File:** `engines/utility_reporting_engine.py`
**Lines:** ~2,422

**Purpose:** Generates comprehensive utility analysis reports combining outputs from all engines into formatted deliverables in multiple formats.

**Output Formats:** Markdown, HTML (with CSS), JSON, CSV, PDF

**Report Types (14):** Executive Dashboard, Bill Audit Report, Rate Comparison, Demand Profile, Cost Allocation, Budget Forecast, Procurement Brief, Benchmark Report, Regulatory Charge Analysis, Savings Tracker, Monthly Summary, Quarterly Review, Annual Report, Portfolio Overview

**Key Methods:**
| Method | Description |
|--------|-------------|
| `generate_monthly_summary()` | Monthly KPI dashboard with trend indicators |
| `generate_bill_audit_report()` | Complete bill audit findings |
| `generate_executive_dashboard()` | C-suite KPIs and action items |
| `render_report()` | Multi-format report rendering |
| `generate_kpi_set()` | Standard KPI calculations |
| `build_chart_data()` | Structured chart data for visualisation |

---

## 5. Workflow Specifications

### 5.1 Overview

| # | Workflow | Phases | Schedule | Duration |
|---|----------|--------|----------|----------|
| 1 | Bill Audit | 4 | Monthly | 60 min |
| 2 | Rate Optimisation | 4 | Annual | 45 min |
| 3 | Demand Management | 4 | Quarterly | 90 min |
| 4 | Cost Allocation | 4 | Monthly | 30 min |
| 5 | Budget Planning | 4 | Annual | 120 min |
| 6 | Procurement Analysis | 4 | Annual | 90 min |
| 7 | Benchmark Analysis | 3 | Quarterly | 45 min |
| 8 | Full Utility Analysis | 8 | Annual | 240 min |

### 5.2 Workflow Details

**WF-1: Bill Audit Workflow** (4 phases)
- Phase 1: BillIngestion -- Import bills from PDF/CSV/Excel/EDI
- Phase 2: BillParsing -- Extract line items, validate format
- Phase 3: ErrorDetection -- 17 error types, duplicate detection, rate verification
- Phase 4: AuditReporting -- Summary, financial impact, refund documentation

**WF-2: Rate Optimisation Workflow** (4 phases)
- Phase 1: LoadProfileAnalysis -- 12+ months consumption and demand profiling
- Phase 2: RateModeling -- Recalculate under all available rate structures
- Phase 3: RateComparison -- Rank alternatives by total cost
- Phase 4: OptimizationReport -- Switching recommendation with implementation steps

**WF-3: Demand Management Workflow** (4 phases)
- Phase 1: DemandProfiling -- Interval data analysis, peak patterns
- Phase 2: PeakIdentification -- Top-N demand events, coincident peak analysis
- Phase 3: StrategyDevelopment -- 10 demand reduction strategies modelled
- Phase 4: DemandReport -- Prioritised action plan with savings quantification

**WF-4: Cost Allocation Workflow** (4 phases)
- Phase 1: MeterMapping -- Configure meter hierarchy and entity mapping
- Phase 2: CostPooling -- Disaggregate costs into allocatable pools
- Phase 3: AllocationCalculation -- Apply allocation methods per entity
- Phase 4: ReconciliationReport -- Verify sum vs. main meter, generate chargebacks

**WF-5: Budget Planning Workflow** (4 phases)
- Phase 1: HistoricalAnalysis -- 24-36 months trend and anomaly detection
- Phase 2: ForecastModeling -- Weather-normalised baseline with 5 methods
- Phase 3: ScenarioAnalysis -- Base/optimistic/pessimistic/custom scenarios
- Phase 4: BudgetReport -- Monthly projections with P10/P50/P90 intervals

**WF-6: Procurement Analysis Workflow** (4 phases)
- Phase 1: MarketAssessment -- Wholesale prices, forward curves, volatility
- Phase 2: LoadProfiling -- Load shape classification, volume profiling
- Phase 3: StrategyDevelopment -- 7 contract structures evaluated
- Phase 4: ProcurementReport -- RFP specifications, supplier comparison, recommendation

**WF-7: Benchmark Analysis Workflow** (3 phases)
- Phase 1: MetricsCalculation -- EUI, cost intensity, demand intensity
- Phase 2: PeerComparison -- Z-score ranking against CIBSE TM46 peers
- Phase 3: BenchmarkReport -- Gap analysis and improvement targets

**WF-8: Full Utility Analysis Workflow** (8 phases)
- Orchestrates all 7 individual workflows into a single end-to-end pipeline
- Selectively enable/disable phases via configuration flags
- Produces consolidated executive summary with prioritised action plan

---

## 6. Template Specifications

10 report templates, each supporting Markdown, HTML, JSON, and CSV output:

| # | Template | Category | Key Content |
|---|----------|----------|-------------|
| 1 | Bill Audit Report | Audit | Billing errors, financial impact, refund amounts |
| 2 | Rate Comparison Report | Rates | Alternative tariff comparison, switching savings |
| 3 | Demand Profile Report | Demand | Peak summary, load duration, DR opportunities |
| 4 | Cost Allocation Report | Allocation | Entity-level breakdown, reconciliation, chargebacks |
| 5 | Budget Forecast Report | Budget | Monthly projections, confidence intervals, scenarios |
| 6 | Procurement Strategy Report | Procurement | Market overview, contract comparison, hedging |
| 7 | Benchmark Report | Benchmark | EUI comparison, Energy Star score, peer ranking |
| 8 | Regulatory Charge Report | Regulatory | Charge decomposition, exemptions, optimisation |
| 9 | Executive Dashboard | Executive | KPI cards, trends, RAG status, action items |
| 10 | Utility Savings Report | Savings | Implemented savings, IPMVP verification, ROI |

All templates include:
- SHA-256 provenance hash appended as HTML comment (MD/HTML) or field (JSON)
- Currency formatting respecting `config.currency_symbol`
- Module version `36.0.0` for traceability

---

## 7. Integration Specifications

### 7.1 Integration Summary

| # | Integration | Lines | Purpose |
|---|-------------|-------|---------|
| 1 | PackOrchestrator | 971 | 10-phase DAG pipeline with parallel execution |
| 2 | MRVBridge | 665 | Routes to MRV-001, 009-013 for emission-cost correlation |
| 3 | DataBridge | 540 | Routes to DATA-001, 002, 003, 010, 014, 018 |
| 4 | Pack031Bridge | 352 | Equipment and baseline data from Industrial Energy Audit |
| 5 | Pack032Bridge | 407 | Building envelope and zone data from Building Assessment |
| 6 | Pack033Bridge | 443 | Quick win savings and M&V data from Quick Wins Identifier |
| 7 | UtilityProviderBridge | 529 | Green Button, EDI, OAuth2, API, SFTP utility connections |
| 8 | WeatherBridge | 514 | NOAA ISD, Meteostat, Open-Meteo, TMY data |
| 9 | MarketDataBridge | 508 | EPEX SPOT, ICE, CME, EEX market prices |
| 10 | HealthCheck | 666 | 20-category system health verification |
| 11 | SetupWizard | 643 | 8-step guided configuration with 8 presets |
| 12 | AlertBridge | 665 | 8 alert types, 4 channels, rule-based evaluation |

### 7.2 Agent Dependencies

**MRV Agents:**
- AGENT-MRV-001: Stationary Combustion (gas utility emissions)
- AGENT-MRV-009: Scope 2 Location-Based (electricity emissions)
- AGENT-MRV-010: Scope 2 Market-Based (PPA/GO emissions)
- AGENT-MRV-013: Dual Reporting Reconciliation

**DATA Agents:**
- AGENT-DATA-001: PDF & Invoice Extractor
- AGENT-DATA-002: Excel/CSV Normalizer
- AGENT-DATA-003: ERP/Finance Connector
- AGENT-DATA-010: Data Quality Profiler
- AGENT-DATA-013: Outlier Detection Agent
- AGENT-DATA-014: Time Series Gap Filler
- AGENT-DATA-018: Data Lineage Tracker
- AGENT-DATA-019: Validation Rule Engine

**FOUNDATION Agents:**
- AGENT-FOUND-001: Orchestrator (DAG)
- AGENT-FOUND-002: Schema Compiler
- AGENT-FOUND-003: Unit Normalizer
- AGENT-FOUND-004: Assumptions Registry
- AGENT-FOUND-005: Citations & Evidence
- AGENT-FOUND-006: Access & Policy Guard
- AGENT-FOUND-008: Reproducibility Agent
- AGENT-FOUND-010: Observability Agent

---

## 8. Configuration

### 8.1 Pack Configuration

**File:** `config/pack_config.py` (1,507 lines)

Top-level `PackConfig` with nested configuration sections:
- `FacilityConfig` -- Facility metadata, floor area, operating hours, climate zone
- `CommodityConfig` -- Per-utility settings (electricity, gas, water, steam)
- `BillAuditConfig` -- Anomaly thresholds, lookback period, tolerance levels
- `RateOptimizationConfig` -- TOU, demand ratchet, green tariff analysis flags
- `DemandConfig` -- Interval resolution, peak thresholds, power factor targets
- `BudgetConfig` -- Forecast horizon, method selection, Monte Carlo iterations
- `BenchmarkConfig` -- Standards selection, peer group, weather normalisation
- `ReportingConfig` -- Output formats, frequency, recipients
- `CarbonConfig` -- Emission factors, carbon price, scope tracking
- `SecurityConfig` -- Roles, data classification, audit logging
- `PerformanceConfig` -- Batch sizes, cache TTL, parallel engines
- `AuditTrailConfig` -- Provenance, lineage, retention

### 8.2 Presets (8)

| # | Preset | Facility Type | Key Focus |
|---|--------|---------------|-----------|
| 1 | office_building | OFFICE | TOU optimisation, demand management, tenant allocation |
| 2 | manufacturing | MANUFACTURING | Demand charges, power factor, process allocation |
| 3 | retail_store | RETAIL | Multi-site portfolio, portfolio procurement |
| 4 | warehouse | WAREHOUSE | Simple rates, seasonal demand, cold storage |
| 5 | healthcare | HEALTHCARE | 24/7 operations, ENSEMBLE forecasting, steam |
| 6 | education | EDUCATION | Seasonal occupancy, weather normalisation |
| 7 | data_center | DATA_CENTER | Flat demand, PUE tracking, market-based carbon |
| 8 | multi_site_portfolio | MULTI_SITE | Cross-facility benchmarking, consolidated procurement |

---

## 9. Database Schema

### 9.1 Migration Summary

| Migration | Tables | Key Features |
|-----------|--------|--------------|
| V276 | gl_utility_bills, gl_bill_line_items, gl_meter_readings | 11 commodity types, 8 bill statuses, 7 bill formats |
| V277 | gl_bill_errors, gl_bill_audit_results | 17 error types, 4 severity levels, auto-correction |
| V278 | gl_rate_structures, gl_rate_tiers, gl_tou_schedules, gl_demand_charges, gl_rate_comparisons | 9 rate types, TOU periods, demand ratchets |
| V279 | gl_interval_data, gl_demand_profiles, gl_peak_events | BRIN indexes, TimescaleDB hypertable, load factor |
| V280 | gl_allocation_entities, gl_allocation_rules, gl_allocation_results, gl_allocation_line_items | 9 entity types, 10 allocation methods |
| V281 | gl_budget_forecasts, gl_monthly_forecasts, gl_budget_variances, gl_forecast_scenarios | 11 forecasting methods, Monte Carlo confidence |
| V282 | gl_procurement_contracts, gl_market_prices, gl_procurement_analyses, gl_green_procurements | 10 contract types, VaR risk, RECs/PPAs |
| V283 | gl_facility_benchmarks, gl_benchmark_targets, gl_portfolio_rankings | ENERGY STAR/CIBSE TM46, 14 seed rows |
| V284 | gl_regulatory_charges, gl_charge_optimizations, gl_exemption_assessments | 18 charge types, 13 exemption types |
| V285 | gl_weather_normalizations, gl_degree_days, gl_utility_reports, gl_utility_kpis, pack036_audit_trail + 2 materialized views | ASHRAE 14 validation, 10 regression models |

### 9.2 Schema Totals

- **Tables:** 30 (including 1 audit trail)
- **Materialized Views:** 2 (mv_facility_cost_summary, mv_portfolio_utility_overview)
- **Indexes:** ~180 (B-tree, BRIN, GIN, composite, partial)
- **RLS Policies:** 46 (tenant isolation + service bypass)
- **CHECK Constraints:** ~170
- **Triggers:** 22 (fn_set_updated_at on all tables)
- **Seed Data:** 14 benchmark target rows (CIBSE TM46 + ENERGY STAR)

### 9.3 Schema Design Patterns

- `pack036_utility_analysis` schema isolation
- UUID primary keys (`gen_random_uuid()`)
- `tenant_id` for multi-tenant isolation
- `provenance_hash VARCHAR(64)` for SHA-256 audit trails
- `created_at` / `updated_at` TIMESTAMPTZ columns
- `metadata JSONB DEFAULT '{}'` for extensibility
- Row-Level Security with tenant isolation + `greenlang_service` bypass
- GIN indexes on JSONB columns
- BRIN indexes for time-series data (`gl_interval_data`)
- TimescaleDB hypertable conversion (conditional)

---

## 10. Testing Strategy

### 10.1 Test Suite Overview

| # | Test File | Coverage Area | ~Tests |
|---|-----------|---------------|--------|
| 1 | conftest.py | Shared fixtures (14 fixtures) | -- |
| 2 | test_init.py | All __init__.py files | ~10 |
| 3 | test_manifest.py | pack.yaml validation | ~15 |
| 4 | test_config.py | Config module, presets, validation | ~25 |
| 5 | test_utility_bill_parser.py | Engine 1 | ~50 |
| 6 | test_rate_structure_analyzer.py | Engine 2 | ~50 |
| 7 | test_demand_analysis.py | Engine 3 | ~50 |
| 8 | test_cost_allocation.py | Engine 4 | ~50 |
| 9 | test_budget_forecasting.py | Engine 5 | ~50 |
| 10 | test_procurement_intelligence.py | Engine 6 | ~45 |
| 11 | test_utility_benchmark.py | Engine 7 | ~45 |
| 12 | test_regulatory_charge_optimizer.py | Engine 8 | ~45 |
| 13 | test_weather_normalization.py | Engine 9 | ~55 |
| 14 | test_utility_reporting.py | Engine 10 | ~50 |
| 15 | test_workflows.py | All 8 workflows | ~55 |
| 16 | test_templates.py | All 10 templates | ~50 |
| 17 | test_integrations.py | All 12 integrations | ~50 |
| 18 | test_e2e.py | End-to-end pipelines | ~30 |
| 19 | test_compliance.py | Regulatory compliance | ~30 |
| 20 | test_performance.py | Performance benchmarks | ~30 |
| **Total** | **20 files** | **Full coverage** | **~850+** |

### 10.2 Test Categories

- **Unit tests**: Individual engine methods with known inputs/outputs
- **Integration tests**: Cross-engine data flow and bridge connectivity
- **End-to-end tests**: Complete workflow execution from bill input to report output
- **Compliance tests**: Regulatory standard adherence (EED, ISO 50001, ASHRAE 14)
- **Performance tests**: Instantiation (<100ms), parsing (<500ms), Monte Carlo (<15s)

### 10.3 Coverage Target

- **Overall target**: 85%+
- **Engine coverage**: 90%+ (calculation paths fully tested)
- **Provenance validation**: Every result object verified for 64-char hex SHA-256 hash

---

## 11. Deployment Guide

### 11.1 Prerequisites

- Python >= 3.11
- PostgreSQL >= 16 (with pgvector extension)
- Redis >= 7
- GreenLang Platform >= 2.0.0

### 11.2 Installation

1. Deploy database migrations V276-V285
2. Load preset configurations
3. Seed rate schedule and benchmark datasets
4. Configure utility provider connections
5. Run health check to verify all components
6. Execute setup wizard for facility configuration

### 11.3 Infrastructure Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU cores | 4 | 8 |
| Memory (GB) | 8 | 16 |
| Storage (GB) | 50 | 100 |
| DB connections | 10 | 30 |

---

## 12. Component Summary

| Component | Count | Total Lines |
|-----------|-------|-------------|
| Engines | 10 | ~22,825 |
| engines/__init__.py | 1 | 571 |
| Workflows | 8 | ~7,975 |
| Templates | 10 | ~6,500 |
| Integrations | 12 | ~6,903 |
| Config (pack_config.py) | 1 | 1,507 |
| Presets (YAML) | 8 | ~1,000 |
| Tests | 20 | ~12,000 |
| Migrations (SQL) | 10 | ~8,000 |
| pack.yaml | 1 | 1,717 |
| __init__.py files | 8 | ~800 |
| **TOTAL** | **89+ files** | **~70,000+ lines** |

---

## 13. Approval

| Role | Name | Date | Status |
|------|------|------|--------|
| Product Owner | GreenLang Product Team | 2026-03-22 | APPROVED |
| Technical Lead | GreenLang Engineering | 2026-03-22 | APPROVED |
| Regulatory SME | GreenLang Compliance | 2026-03-22 | APPROVED |
| Security Review | GreenLang Security | 2026-03-22 | APPROVED |

---

*Document version: 1.0.0 | Last updated: 2026-03-22 | Author: GreenLang Product Team*
