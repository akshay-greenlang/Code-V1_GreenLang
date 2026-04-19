# PRD-PACK-047: GHG Emissions Benchmark Pack

**Pack ID:** PACK-047-benchmark
**Category:** GHG Accounting Packs
**Tier:** Enterprise
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-26
**Prerequisite:** PACK-041 (Scope 1-2 Complete) recommended; enhanced with PACK-042/043 (Scope 3), PACK-044 (Inventory Management), PACK-045 (Base Year Management), PACK-046 (Intensity Metrics)

---

## 1. Executive Summary

### 1.1 Problem Statement

Meaningful GHG emissions performance assessment requires comparing an organisation's emissions profile against relevant peers, sector averages, science-based pathways, and regulatory thresholds. Without rigorous benchmarking, organisations cannot answer fundamental questions: "How do we compare to our sector peers?", "Are we aligned with a 1.5C pathway?", "Where do we sit in the CDP league table?", "What is our transition risk exposure relative to competitors?"

Current approaches to GHG benchmarking suffer from persistent structural challenges:

1. **Incomparable baselines**: Organisations report under different scopes, boundaries, and methodologies. A Scope 1+2 location-based figure cannot be directly compared to a Scope 1+2 market-based figure without normalisation. Consolidation approaches (equity share vs. operational control vs. financial control) further fragment comparability. Without systematic scope alignment and boundary harmonisation, peer comparisons produce misleading conclusions that can misdirect strategy and capital allocation.

2. **Peer group construction bias**: Selecting peer groups by simple SIC/NACE code matching ignores critical confounders: geographic mix (emissions factors differ 10x across grids), product mix (a steel company making specialty steel vs. commodity hot-rolled coil), vertical integration level, outsourcing extent, and company size. Without multi-dimensional peer construction using GICS/NACE/ISIC cross-mapping, revenue-based sizing, geographic weighting, and value chain position filtering, benchmarks compare unlike with unlike.

3. **Static snapshots vs. trajectory analysis**: Most benchmarking reports show a single year's percentile ranking. Decision-makers need rate-of-change benchmarking: is the organisation decarbonising faster or slower than peers? Is the gap to the sector median widening or narrowing? Without temporal trajectory analysis, an organisation at the 60th percentile but improving at 8% CAGR may appear worse than one at the 40th percentile that has stalled.

4. **Disconnected pathway alignment**: Science-based pathways (IEA NZE 2050, IPCC AR6 C1-C3, SBTi SDA, OECM, TPI, CRREM) each define sector-specific decarbonisation curves. Organisations need to know their position relative to multiple pathways simultaneously, with interpolated annual waypoints and gap-to-pathway metrics. Current tools typically assess alignment against a single pathway in isolation.

5. **Missing portfolio aggregation**: Financial institutions (asset managers, banks, insurers) must benchmark portfolio-level emissions (financed emissions, PCAF) across hundreds to thousands of holdings. Portfolio benchmarking requires weighted aggregation by ownership share, asset class, and sector with proper handling of data gaps, estimated vs. reported data, and varying reporting periods. EU SFDR PAI indicators, TCFD portfolio alignment, and NZAOA target-setting all demand portfolio-level benchmark outputs.

6. **Inadequate data quality transparency**: Benchmark comparisons are only as reliable as the underlying data. CDP-reported data has self-reported quality; estimated emissions carry wider uncertainty than verified figures. Without explicit data quality scoring (GHG Protocol data quality matrix, PCAF data quality scores 1-5), stakeholders cannot assess the reliability of benchmark positions.

7. **Fragmented external datasets**: Benchmark data is scattered across CDP (climate change responses), TPI (management quality + carbon performance), GRESB (real estate + infrastructure), CRREM (real estate stranding risk), ISS ESG, MSCI, Sustainalytics, S&P Global, and Bloomberg. Each dataset uses different sector classifications, metric definitions, and reporting periods. No unified normalisation layer exists to integrate these sources into a coherent benchmark universe.

8. **Regulatory disclosure gaps**: Multiple frameworks now require explicit benchmarking disclosures: ESRS E1-4 (performance in relation to EU Taxonomy benchmarks), CSRD topical standards (sector comparison), CDP scoring (management band + performance band relative to sector), SFDR PAI (comparison to index/universe), TCFD (portfolio alignment), and SEC (contextual comparison). Organisations must produce framework-specific benchmark outputs from a single analytical base.

### 1.2 Solution Overview

PACK-047 is the **GHG Emissions Benchmark Pack** -- the seventh pack in the "GHG Accounting Packs" category, dedicated to comprehensive emissions benchmarking across every dimension that matters for compliance, strategy, and investment decision-making.

While PACK-046 (Intensity Metrics) includes a basic benchmarking engine (engine 4) for peer percentile ranking of intensity metrics, PACK-047 provides the full dedicated benchmarking lifecycle:

- **Multi-dimensional peer group construction** with GICS/NACE/ISIC sector mapping, revenue/FTE sizing bands, geographic mix weighting, and value chain position filtering
- **Scope-aligned normalisation** harmonising Scope 1, 2 (location/market), and Scope 3 boundaries across heterogeneous peer data
- **External dataset integration** with CDP, TPI, GRESB, CRREM, ISS ESG, and custom dataset ingestion via standardised adapters
- **Science-based pathway alignment** against IEA NZE 2050, IPCC AR6 C1/C2/C3, SBTi SDA, OECM, TPI Carbon Performance, and CRREM decarbonisation curves
- **Temporal trajectory benchmarking** comparing rate-of-change, acceleration, and convergence/divergence trends across multi-year time series
- **Portfolio-level benchmarking** for financial institutions with PCAF-aligned data quality scoring, weighted aggregation, and WACI/carbon footprint metrics
- **Transition risk scoring** with implied temperature rise (ITR), carbon budget overshoot probability, and stranding risk assessment
- **Data quality transparency** with GHG Protocol quality matrix scoring, PCAF scores, and confidence intervals on benchmark positions
- **Multi-framework disclosure** generating ESRS, CDP, SFDR, TCFD, SEC, and GRI-compliant benchmark outputs from a single analytical base
- **Automated reporting** with league tables, radar charts, pathway alignment graphs, portfolio heatmaps, and executive dashboards

The pack includes **10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets** covering the complete benchmarking lifecycle.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Consultant Approach | PACK-047 Benchmark Pack |
|-----------|------------------------------|--------------------------|
| Peer group construction | Subjective analyst selection (5-20 peers) | Multi-dimensional algorithmic matching (50-500+ peers) with GICS/NACE cross-map |
| Scope harmonisation | Ad hoc adjustments or ignored | Systematic boundary alignment engine with 8 normalisation steps |
| External data sources | Single source (usually CDP) | 6+ integrated sources (CDP, TPI, GRESB, CRREM, ISS ESG, custom) |
| Pathway alignment | Single pathway, point-in-time | Simultaneous multi-pathway with interpolated annual waypoints |
| Trajectory analysis | Static single-year snapshot | Multi-year rate-of-change, convergence trend, acceleration metrics |
| Portfolio aggregation | Spreadsheet-based (error-prone) | Automated PCAF-aligned weighted aggregation with data quality scores |
| Transition risk | Qualitative narrative | Quantitative ITR, carbon budget overshoot, stranding risk probability |
| Data quality | Assumed uniform quality | Explicit PCAF 1-5 scoring, GHG Protocol quality matrix, confidence intervals |
| Disclosure output | Single framework | 6+ framework-specific outputs from single analytical base |
| Audit trail | None | SHA-256 provenance on every result, full calculation lineage |

### 1.4 Distinction from Related Packs

| Pack | Focus | Relationship to PACK-047 |
|------|-------|--------------------------|
| PACK-035 Energy Benchmark | Building/facility EUI benchmarking (kWh/m2, ENERGY STAR, NABERS) | Energy performance, not GHG emissions; different denominators, peers, standards |
| PACK-046 Engine 4 | Basic peer intensity percentile ranking | PACK-047 extends from single-metric ranking to full multi-dimensional benchmarking lifecycle |
| PACK-046 Engine 5 | SBTi SDA target pathway convergence | PACK-047 provides multi-pathway alignment (IEA, IPCC, SBTi, OECM, TPI, CRREM) and adds ITR |
| PACK-041/042/043 | Absolute emissions calculation | PACK-047 consumes absolute emissions as input for benchmarking |
| PACK-044 | Inventory governance | PACK-047 uses inventory period boundaries for temporal alignment |
| PACK-045 | Base year management | PACK-047 uses base year data for trajectory analysis starting points |

### 1.5 Target Users

**Primary:**
- Sustainability managers needing sector peer comparison for CSRD/ESRS E1 disclosure and CDP Climate Change response
- Strategy teams assessing competitive positioning on decarbonisation and transition risk exposure
- ESG analysts at asset managers benchmarking portfolio emissions against indices and peer portfolios

**Secondary:**
- Chief Sustainability Officers presenting board-level benchmarking dashboards for strategic planning
- Climate risk teams at banks and insurers assessing portfolio alignment for TCFD and NZAOA reporting
- Investor relations teams responding to ESG rating agency questionnaires requiring benchmark context
- Auditors and assurance providers verifying benchmark methodology and data quality

### 1.6 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Peer group construction time | <30 seconds for 500-peer universe | Benchmark suite timer |
| Scope normalisation accuracy | 100% deterministic (no approximation) | Unit test verification |
| Pathway alignment coverage | 6+ pathways simultaneously | Feature completeness audit |
| Portfolio aggregation capacity | 10,000+ holdings | Load test with synthetic portfolio |
| ITR calculation precision | Decimal (no float) | Test suite validation |
| Data quality transparency | PCAF 1-5 on every data point | Integration test coverage |
| Framework disclosure coverage | ESRS, CDP, SFDR, TCFD, SEC, GRI | Template completeness audit |
| Calculation reproducibility | Bit-perfect across runs | SHA-256 hash comparison |

---

## 2. Technical Architecture

### 2.1 Engine Specifications (10 Engines)

#### Engine 1: Peer Group Construction Engine (~1,200 lines)
- **Purpose**: Build statistically meaningful peer groups from heterogeneous emissions data
- **Key Features**:
  - GICS/NACE/ISIC/SIC sector classification cross-mapping with configurable matching depth (2-digit to 8-digit)
  - Revenue-band sizing (micro <2M, small 2-10M, medium 10-50M, large 50-250M, enterprise 250M+, mega 1B+)
  - Geographic mix weighting by grid emission factor similarity
  - Value chain position filtering (upstream, midstream, downstream, integrated)
  - Minimum peer count enforcement (default 10, configurable)
  - Peer quality scoring: data recency, scope completeness, verification status
  - Outlier detection and flagging (IQR method, configurable multiplier)
- **Formulas**:
  - Sector similarity: `sim(A,B) = Σ(w_i * match(A_i, B_i))` where w_i are configurable dimension weights
  - Size band distance: `d_size = |ln(rev_A) - ln(rev_B)| / ln(10)` (log-revenue distance)
  - Geographic similarity: `sim_geo = 1 - |ef_A - ef_B| / max(ef_A, ef_B)` (emission factor proximity)

#### Engine 2: Scope Normalisation Engine (~1,100 lines)
- **Purpose**: Harmonise emissions data across different scope boundaries, consolidation approaches, and reporting methodologies
- **Key Features**:
  - Scope boundary alignment (S1-only, S1+S2L, S1+S2M, S1+S2+S3)
  - Consolidation approach normalisation (equity share, operational control, financial control)
  - GWP version alignment (AR4, AR5, AR6) with gas-by-gas conversion factors
  - Currency normalisation for economic denominators (PPP-adjusted, nominal USD/EUR)
  - Reporting period alignment (calendar year, fiscal year, custom period)
  - Data gap estimation with quality downgrade flagging
  - Biogenic carbon treatment alignment (included/excluded/separate)
- **Formulas**:
  - GWP realignment: `E_new = Σ(gas_i * GWP_new_i / GWP_old_i * E_old_gas_i)` for each GHG
  - PPP adjustment: `D_ppp = D_nominal * PPP_factor(country, year)`
  - Period pro-rata: `E_aligned = E_reported * days_overlap / days_reporting`

#### Engine 3: External Dataset Integration Engine (~1,000 lines)
- **Purpose**: Ingest, normalise, and cache benchmark data from external sources
- **Key Features**:
  - CDP Climate Change response adapter (C0.1-C0.5 identity, C6.1-C6.5 emissions, C6.10 intensity)
  - TPI Carbon Performance adapter (sector pathways, company ratings, management quality)
  - GRESB Real Estate + Infrastructure adapter (asset-level benchmarks)
  - CRREM Carbon Risk adapter (stranding year, decarbonisation pathways per country/asset type)
  - ISS ESG Climate adapter (carbon risk rating, temperature alignment)
  - Custom CSV/JSON/XLSX dataset ingestion with schema validation
  - Data freshness tracking (last updated timestamp, staleness alerts)
  - Caching layer with configurable TTL per source
- **Zero-hallucination**: All external data is ingested as-is from authoritative sources; no interpolation or estimation without explicit quality flagging

#### Engine 4: Pathway Alignment Engine (~1,100 lines)
- **Purpose**: Assess emissions trajectory alignment against science-based decarbonisation pathways
- **Key Features**:
  - IEA Net Zero Emissions 2050 (NZE) sector pathways (power, industry, transport, buildings)
  - IPCC AR6 pathways: C1 (1.5C no/limited overshoot), C2 (1.5C high overshoot), C3 (likely below 2C)
  - SBTi Sectoral Decarbonisation Approach (SDA) convergence for all covered sectors
  - One Earth Climate Model (OECM) sector pathways
  - TPI Carbon Performance benchmarks by sector
  - CRREM decarbonisation pathways by country and asset type
  - Annual waypoint interpolation for all pathways (linear and trajectory-preserving)
  - Gap-to-pathway calculation: absolute gap, percentage gap, years to convergence
  - Overshoot year calculation: when organisation's trajectory crosses pathway ceiling
- **Formulas**:
  - Pathway interpolation: `P(y) = P(y1) + (P(y2) - P(y1)) * (y - y1) / (y2 - y1)`
  - Gap-to-pathway: `gap_abs = I_org(y) - P(y)`, `gap_pct = gap_abs / P(y) * 100`
  - Years to convergence: `y_conv = y + gap_abs / |dI/dy|` (at current rate of change)

#### Engine 5: Implied Temperature Rise (ITR) Engine (~950 lines)
- **Purpose**: Calculate organisation-level and portfolio-level implied temperature rise
- **Key Features**:
  - Cumulative carbon budget approach (IPCC AR6 remaining carbon budgets)
  - Sector-relative approach (emissions intensity vs. pathway benchmark)
  - Rate-of-reduction approach (current trend vs. required pathway slope)
  - Temperature overshoot probability using remaining budget distribution
  - Portfolio-weighted ITR for financial institutions
  - Confidence intervals based on data quality and projection uncertainty
  - Support for Scope 1+2 and Scope 1+2+3 ITR variants
- **Formulas**:
  - Budget-based ITR: `ITR = T_ref + ΔT * (cum_E / budget_T)` calibrated to IPCC budget-temperature mapping
  - Sector-relative ITR: `ITR = f(I_org / I_pathway(T))` for temperature T in [1.5, 4.0]
  - Portfolio ITR: `ITR_portfolio = Σ(w_i * ITR_i)` where w_i = EVIC share or revenue attribution

#### Engine 6: Trajectory Benchmarking Engine (~900 lines)
- **Purpose**: Compare multi-year emissions trajectory against peer trajectories
- **Key Features**:
  - Compound Annual Reduction Rate (CARR) calculation for each peer
  - Acceleration/deceleration detection (2nd derivative of emissions path)
  - Convergence/divergence analysis (is gap to peer median widening or narrowing?)
  - Percentile trajectory: tracking percentile rank over 3-10 year horizon
  - Rate-of-change ranking: ranking peers by decarbonisation speed, not level
  - Structural break detection (M&A, divestiture, methodology change)
  - Trajectory fan charts showing peer distribution evolution
- **Formulas**:
  - CARR: `CARR = (E_end / E_start) ^ (1 / n) - 1`
  - Acceleration: `a = CARR(t2,t1) - CARR(t1,t0)` (change in reduction rate between periods)
  - Convergence rate: `CR = (gap_t1 - gap_t0) / gap_t0` (negative = converging to peer median)

#### Engine 7: Portfolio Benchmarking Engine (~1,000 lines)
- **Purpose**: Aggregate and benchmark portfolio-level emissions for financial institutions
- **Key Features**:
  - PCAF-aligned weighted aggregation by asset class (listed equity, corporate bonds, project finance, real estate, mortgages, sovereign debt)
  - Weighted Average Carbon Intensity (WACI) per TCFD/SFDR methodology
  - Carbon footprint (total financed emissions / AUM)
  - Carbon intensity (financed emissions / revenue)
  - Data quality score aggregation (PCAF scores 1-5, weighted by exposure)
  - Portfolio vs. benchmark index comparison (with tracking error attribution)
  - Sector allocation contribution analysis (which sector over/underweights drive performance difference)
  - Holdings-level attribution (largest contributors to portfolio benchmark gap)
- **Formulas**:
  - Financed emissions: `FE_i = ownership_share_i * E_i` where ownership share = investment/EVIC
  - WACI: `WACI = Σ(w_i * I_i)` where w_i = portfolio weight, I_i = tCO2e/M$ revenue
  - PCAF quality score: `Q_portfolio = Σ(w_i * Q_i)` weighted by financed emissions share

#### Engine 8: Data Quality Scoring Engine (~850 lines)
- **Purpose**: Assess and score data quality of all benchmark inputs
- **Key Features**:
  - GHG Protocol data quality matrix (5x5: temporal, geographic, technological, completeness, reliability)
  - PCAF data quality scores 1-5 per asset class methodology
  - Source hierarchy: verified > reported > estimated > modelled > sector average
  - Confidence interval calculation based on data quality composite score
  - Data gap identification and coverage analysis (what % of peers have complete data)
  - Quality-weighted benchmark results (higher quality data gets more weight)
  - Automated quality improvement recommendations
- **Formulas**:
  - Composite quality score: `Q = Σ(w_dim * score_dim) / Σ(w_dim)` across 5 dimensions
  - Confidence interval: `CI_95 = result ± Z_0.975 * σ_quality` where σ scales with quality score
  - Quality-weighted mean: `μ_qw = Σ(Q_i * v_i) / Σ(Q_i)` for peer values v_i

#### Engine 9: Transition Risk Scoring Engine (~900 lines)
- **Purpose**: Quantify transition risk exposure based on benchmark position
- **Key Features**:
  - Carbon budget overshoot probability (Monte Carlo on remaining budget allocation)
  - Stranding risk assessment (year when asset becomes uncompetitive under pathway)
  - Regulatory risk scoring (distance from regulatory thresholds: EU ETS, CBAM, etc.)
  - Competitive risk (quartile position relative to sector decarbonisation leaders)
  - Financial impact estimation (carbon price exposure at various price scenarios)
  - Composite transition risk score (0-100) with dimension breakdown
  - Risk trajectory (is transition risk increasing or decreasing over time)
- **Formulas**:
  - Budget overshoot: `P(overshoot) = P(cum_E > allocated_budget)` via Monte Carlo
  - Stranding year: `y_strand = min(y : I_org(y) > pathway_ceiling(y))`
  - Carbon price exposure: `CPE = E_scope1 * carbon_price + E_scope2 * pass_through_rate * carbon_price`

#### Engine 10: Benchmark Reporting Engine (~950 lines)
- **Purpose**: Generate comprehensive benchmark reports in multiple formats
- **Key Features**:
  - League table generation with configurable sort and filter
  - Radar chart data (multi-dimensional performance profile)
  - Pathway alignment graphs (organisation trajectory vs. multiple pathways)
  - Portfolio heatmaps (sector x metric colour-coded matrix)
  - Trend sparklines (5-year trajectory mini-charts)
  - Framework-specific disclosure sections (ESRS, CDP, SFDR, TCFD, SEC, GRI)
  - Executive summary with key findings and actionable insights
  - Export: Markdown, HTML, PDF, JSON, CSV, XBRL
- **Format support**: All outputs include SHA-256 provenance hash and generation metadata

### 2.2 Workflow Specifications (8 Workflows)

1. **Peer Group Setup Workflow** (~950 lines) - SectorMap -> SizeBand -> GeoWeight -> PeerScore -> Validate
2. **Benchmark Assessment Workflow** (~950 lines) - DataIngest -> Normalise -> Compare -> Rank -> Report
3. **Pathway Alignment Workflow** (~850 lines) - PathwayLoad -> Interpolate -> GapAnalysis -> AlignmentScore
4. **Trajectory Analysis Workflow** (~850 lines) - TimeSeriesLoad -> CARR -> Convergence -> TrajectoryRank
5. **Portfolio Benchmark Workflow** (~900 lines) - HoldingsLoad -> PCAF -> Aggregate -> WACI -> Compare
6. **Transition Risk Workflow** (~850 lines) - BudgetAlloc -> StrandingCalc -> RegRisk -> CompRisk -> Score
7. **Disclosure Preparation Workflow** (~800 lines) - MetricAgg -> FrameworkMap -> QACheck -> Package
8. **Full Benchmark Pipeline Workflow** (~1,200 lines) - 8-phase end-to-end orchestration

### 2.3 Template Specifications (10 Templates)

1. **Benchmark Executive Dashboard** (~750 lines) - Key KPIs, sparklines, traffic lights
2. **Peer Comparison Report** (~780 lines) - League table, distribution charts, gap analysis
3. **Pathway Alignment Report** (~700 lines) - Multi-pathway graph, gap metrics, convergence year
4. **Trajectory Analysis Report** (~680 lines) - CARR rankings, convergence trends, fan charts
5. **Portfolio Benchmark Report** (~750 lines) - WACI, carbon footprint, sector attribution, heatmap
6. **Transition Risk Report** (~720 lines) - ITR, stranding risk, regulatory exposure, composite score
7. **Data Quality Report** (~680 lines) - Quality scores, coverage analysis, improvement recommendations
8. **ESRS E1 Benchmark Disclosure** (~700 lines) - ESRS E1-4 compliant benchmark context, XBRL tags
9. **CDP Climate Benchmark Section** (~550 lines) - CDP C6/C7 benchmark response data
10. **SFDR PAI Benchmark Report** (~650 lines) - SFDR PAI 1-3 benchmark comparison with index

### 2.4 Integration Specifications (12 Integrations)

1. **Pack Orchestrator** (~820 lines) - 10-phase DAG pipeline coordinator
2. **MRV Bridge** (~400 lines) - 30 AGENT-MRV agents for emissions data
3. **Data Bridge** (~430 lines) - AGENT-DATA agents for external data ingestion
4. **PACK-041 Bridge** (~370 lines) - Scope 1+2 absolute emissions
5. **PACK-042/043 Bridge** (~420 lines) - Scope 3 absolute emissions
6. **PACK-044 Bridge** (~360 lines) - Inventory periods and governance
7. **PACK-045 Bridge** (~350 lines) - Base year data for trajectory baselines
8. **PACK-046 Bridge** (~400 lines) - Intensity metrics for intensity benchmarking
9. **External Dataset Bridge** (~550 lines) - CDP, TPI, GRESB, CRREM, ISS ESG data adapters
10. **Health Check** (~420 lines) - 20-category system verification
11. **Setup Wizard** (~520 lines) - 8-step guided configuration
12. **Alert Bridge** (~580 lines) - Threshold/deadline/pathway deviation alerts

### 2.5 Configuration & Presets

**pack_config.py** (~1,800 lines):
- 18 enums: SectorClassification, PeerSizeBand, ScopeAlignment, ConsolidationApproach, GWPVersion, PathwayType, PathwayScenario, ITRMethod, PortfolioAssetClass, PCAFScore, DataSourceType, TransitionRiskCategory, BenchmarkMetric, ReportFormat, DisclosureFramework, AlertType, QualityDimension, NormalisationStep
- 30+ benchmark data constants: sector pathways, carbon budgets, GWP conversion factors
- 15+ sub-config Pydantic models with validators
- PackConfig wrapper with from_preset(), from_yaml(), merge(), validate()

**8 Presets:**
1. `corporate_general.yaml` - General corporate benchmarking (default)
2. `power_utilities.yaml` - Power generation and utilities sector
3. `heavy_industry.yaml` - Steel, cement, aluminium, chemicals
4. `real_estate.yaml` - Commercial real estate (GRESB, CRREM aligned)
5. `financial_services.yaml` - Asset managers, banks, insurers (PCAF, WACI)
6. `transport_logistics.yaml` - Aviation, shipping, road freight, rail
7. `oil_gas.yaml` - Upstream, midstream, downstream petroleum
8. `food_agriculture.yaml` - Agriculture, food processing, retail

---

## 3. Database Schema

### 3.1 Migrations (V386-V395)

All tables in `ghg_benchmark` schema with `gl_bm_` prefix. UUID primary keys, NUMERIC precision, JSONB metadata, RLS with tenant isolation.

| Migration | Description | Key Tables |
|-----------|-------------|------------|
| V386 | Core schema + configurations | `gl_bm_configurations`, `gl_bm_reporting_periods` |
| V387 | Peer group management | `gl_bm_peer_groups`, `gl_bm_peer_definitions`, `gl_bm_peer_data` |
| V388 | Scope normalisation | `gl_bm_normalisation_runs`, `gl_bm_normalised_data` |
| V389 | External datasets | `gl_bm_external_sources`, `gl_bm_external_data`, `gl_bm_data_cache` |
| V390 | Pathway alignment | `gl_bm_pathways`, `gl_bm_pathway_waypoints`, `gl_bm_alignment_results` |
| V391 | ITR calculations | `gl_bm_itr_calculations`, `gl_bm_itr_portfolio` |
| V392 | Trajectory benchmarking | `gl_bm_trajectories`, `gl_bm_trajectory_comparisons` |
| V393 | Portfolio benchmarking | `gl_bm_portfolios`, `gl_bm_holdings`, `gl_bm_portfolio_results` |
| V394 | Transition risk + data quality | `gl_bm_transition_risk`, `gl_bm_data_quality_scores` |
| V395 | Views, indexes, seed data | Materialised views, composite indexes, seed pathways and sector data |

---

## 4. Testing Strategy

### 4.1 Test Coverage Targets

| Category | Target | Method |
|----------|--------|--------|
| Unit tests (engines) | 95%+ line coverage per engine | pytest with deterministic fixtures |
| Workflow tests | All 8 workflows, happy path + error paths | Mocked engine calls |
| Integration tests | All 12 integrations | Mocked bridge responses |
| Template tests | All 10 templates render correctly | Output structure validation |
| Config tests | Preset loading, merge, validation, env overrides | Full config lifecycle |
| E2E tests | 45+ end-to-end scenarios | Full pipeline with synthetic data |
| Determinism | SHA-256 hash identical across runs | Repeated execution comparison |

### 4.2 Test File Plan (~17 files, ~7,000+ lines, 500+ test functions)

1. `conftest.py` (~550 lines) - Shared fixtures, mock data, synthetic peer datasets
2. `test_peer_group_engine.py` - Peer construction, outlier detection, quality scoring
3. `test_scope_normalisation_engine.py` - Boundary alignment, GWP conversion, period pro-rata
4. `test_external_dataset_engine.py` - Adapter parsing, caching, freshness tracking
5. `test_pathway_alignment_engine.py` - Interpolation, gap-to-pathway, convergence year
6. `test_itr_engine.py` - Budget-based ITR, sector-relative ITR, portfolio ITR
7. `test_trajectory_engine.py` - CARR, acceleration, convergence analysis
8. `test_portfolio_engine.py` - PCAF aggregation, WACI, sector attribution
9. `test_data_quality_engine.py` - Quality scoring, confidence intervals, weighting
10. `test_transition_risk_engine.py` - Budget overshoot, stranding year, composite score
11. `test_reporting_engine.py` - Report generation, format validation, export
12. `test_workflows.py` - All 8 workflows
13. `test_integrations.py` - All 12 integrations
14. `test_templates.py` - All 10 templates
15. `test_config.py` - Config lifecycle
16. `test_pack_yaml.py` - Manifest validation
17. `e2e/test_e2e.py` (~1,300 lines) - Full end-to-end benchmark scenarios

---

## 5. Performance Requirements

| Metric | Target | Context |
|--------|--------|---------|
| Peer group construction | <30 sec for 500 peers | In-memory GICS/NACE matching |
| Scope normalisation | <5 sec for 100 peers | Deterministic arithmetic |
| Pathway alignment | <10 sec for 6 pathways, 30-year horizon | Interpolation + gap calc |
| ITR calculation | <15 sec for 1,000-holding portfolio | Budget-based approach |
| Portfolio aggregation | <20 sec for 10,000 holdings | PCAF-weighted Decimal sums |
| Full benchmark pipeline | <120 sec for 500-peer, 10-year analysis | All engines sequential |
| Report generation | <10 sec per format | Template rendering |

---

## 6. Security & Compliance

- All external data cached locally with configurable TTL; no live API calls during calculation
- Tenant isolation via RLS on all tables
- SHA-256 provenance hash on every benchmark result
- No PII in benchmark datasets (company identifiers are anonymisable)
- Audit trail for all benchmark configuration changes
- RBAC integration: `benchmark:read`, `benchmark:write`, `benchmark:admin` permissions

---

## 7. Dependencies

### 7.1 Platform Dependencies
- Python 3.11+, Pydantic v2, PyYAML, hashlib, Decimal
- PostgreSQL 15+ with JSONB, UUID, NUMERIC types
- Flyway for migration management

### 7.2 Pack Dependencies
- PACK-041 (Scope 1-2 Complete) - Scope 1+2 absolute emissions input
- PACK-042/043 (Scope 3) - Scope 3 absolute emissions input
- PACK-044 (Inventory Management) - Reporting period boundaries
- PACK-045 (Base Year Management) - Base year and trajectory start points
- PACK-046 (Intensity Metrics) - Intensity metrics for intensity-based benchmarking

### 7.3 Agent Dependencies
- AGENT-MRV (30 agents) - Emissions calculation data
- AGENT-DATA (20 agents) - External data ingestion and quality profiling

---

## 8. Glossary

| Term | Definition |
|------|-----------|
| CARR | Compound Annual Reduction Rate - annualised emissions reduction rate |
| CRREM | Carbon Risk Real Estate Monitor - real estate decarbonisation pathways |
| EVIC | Enterprise Value Including Cash - denominator for ownership share calculation |
| GICS | Global Industry Classification Standard - 4-level sector taxonomy (MSCI/S&P) |
| GRESB | Global Real Estate Sustainability Benchmark - real estate/infrastructure ESG benchmark |
| ITR | Implied Temperature Rise - estimated warming if all companies matched this trajectory |
| NACE | Statistical Classification of Economic Activities in the European Community |
| NZE | Net Zero Emissions by 2050 - IEA scenario |
| OECM | One Earth Climate Model - science-based sector decarbonisation pathways |
| PCAF | Partnership for Carbon Accounting Financials - financed emissions methodology |
| SDA | Sectoral Decarbonisation Approach - SBTi sector-specific intensity pathway method |
| SFDR | Sustainable Finance Disclosure Regulation (EU) - ESG product disclosure |
| TPI | Transition Pathway Initiative - company carbon performance benchmarking |
| WACI | Weighted Average Carbon Intensity - portfolio-level carbon intensity metric (tCO2e/M$ revenue) |

---

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-26 | GreenLang Product Team | Initial production release |

---

## 10. Appendices

### 10.1 Supported Science-Based Pathways

| Pathway | Source | Sectors | Temperature | Horizon |
|---------|--------|---------|-------------|---------|
| IEA NZE 2050 | IEA World Energy Outlook | Power, Industry, Transport, Buildings | 1.5C | 2020-2050 |
| IPCC AR6 C1 | IPCC Working Group III | Global aggregate + sectors | 1.5C (no/limited OS) | 2020-2100 |
| IPCC AR6 C2 | IPCC Working Group III | Global aggregate | 1.5C (high OS) | 2020-2100 |
| IPCC AR6 C3 | IPCC Working Group III | Global aggregate | Likely below 2C | 2020-2100 |
| SBTi SDA | SBTi Corporate Manual v2 | Power, Steel, Cement, Aluminium, Buildings, Transport, Paper, Food | 1.5C / WB2C | 2020-2050 |
| OECM | UTS/ISF | 40+ sub-sectors | 1.5C | 2020-2050 |
| TPI CP | Transition Pathway Initiative | 16 high-impact sectors | Various | 2020-2050 |
| CRREM | CRREM v2 | Commercial RE by country/type | 1.5C / 2C | 2020-2050 |

### 10.2 PCAF Data Quality Scoring Summary

| Score | Listed Equity/Bonds | Real Estate | Mortgages |
|-------|---------------------|-------------|-----------|
| 1 | Verified Scope 1+2+3 | Actual energy consumption | Actual energy consumption |
| 2 | Reported Scope 1+2 | Estimated energy from billing | Estimated from EPC |
| 3 | Company-specific estimated | Estimated from floor area | Estimated from building type |
| 4 | Sector physical activity | Estimated from building count | Estimated from region |
| 5 | Sector revenue-based | Estimated from investment | Estimated from investment |
