# PRD-PACK-028: Sector Pathway Pack

**Pack ID:** PACK-028-sector-pathway
**Category:** Net Zero Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Approved
**Author:** GreenLang Product Team
**Date:** 2026-03-19
**Prerequisite:** PACK-021 Net Zero Starter Pack (recommended)

---

## 1. Executive Summary

### 1.1 Problem Statement

Achieving net-zero emissions requires sector-specific decarbonization pathways that align with global climate scenarios. The SBTi Sectoral Decarbonization Approach (SDA) and IEA Net Zero by 2050 roadmap provide science-based intensity convergence pathways for 15+ high-emitting sectors. However, organizations face critical challenges:

1. **Sector-specific pathway complexity**: Each sector has unique decarbonization challenges - power requires grid transformation, steel needs green hydrogen, cement faces process emissions, aviation depends on sustainable fuels. Generic absolute contraction approaches (ACA) fail to capture sector dynamics.

2. **Intensity metric alignment**: SDA requires sector-specific intensity metrics (tCO2e/MWh for power, tCO2e/tonne for steel/cement, gCO2/pkm for aviation). Companies struggle to map operational data to these metrics and track convergence to sector benchmarks.

3. **Multi-scenario pathway modeling**: IEA NZE 2050 provides pathways for 1.5°C, but companies need 2°C and well-below-2°C scenarios for risk analysis. Modeling technology transitions (coal-to-renewable, gas-to-hydrogen, diesel-to-electric) across scenarios requires sector-specific abatement curves.

4. **Technology transition roadmaps**: Each sector has distinct technology pathways - renewable capacity expansion for power, electrification + hydrogen for steel, CCS for cement, SAF for aviation. Translating sector pathways into technology adoption schedules and CapEx requirements is manual and error-prone.

5. **Gap analysis vs. sector benchmarks**: Companies need to compare their current intensity and trajectory against sector leaders, sector average, and science-based pathways. Identifying the gap to sector convergence and required acceleration is critical for board-level strategy.

6. **Abatement waterfall by lever**: Sector pathways decompose into specific levers (renewable procurement, fuel switching, electrification, efficiency, CCS, hydrogen). Quantifying the contribution of each lever to pathway achievement and sequencing implementation requires detailed sector knowledge.

### 1.2 Solution Overview

PACK-028 is the **Sector Pathway Pack** - the fourth pack in the "Net Zero Packs" category. It provides deep sector-specific decarbonization pathway analysis aligned with SBTi SDA methodology (12 sectors) and IEA Net Zero roadmap (15+ sectors), enabling organizations to design science-based transition strategies tailored to their sector's unique challenges.

The pack delivers:
- **15+ sector-specific pathways**: Power, steel, cement, aluminum, chemicals, pulp/paper, aviation, shipping, road transport, rail, buildings (residential/commercial), agriculture, food/beverage, oil & gas, and cross-sector coverage
- **SBTi SDA compliance**: Automatic sector classification (NACE/GICS/ISIC), intensity metric calculation, convergence pathway generation, and SBTi target validation
- **IEA scenario integration**: 1.5°C (NZE 2050), 2°C (APS), and well-below-2°C pathway modeling with technology-specific trajectories
- **Technology transition roadmaps**: Sector-specific technology adoption schedules (renewable capacity, hydrogen production, CCS deployment, fleet electrification, etc.)
- **Abatement waterfall analysis**: Lever-by-lever emission reduction contribution with cost curves, implementation timelines, and dependency mapping
- **Sector benchmarking**: Comparison against sector leaders, SBTi-validated peers, and IEA pathway milestones with gap-to-benchmark analysis

Every calculation is **zero-hallucination** (deterministic lookups from SBTi/IEA published data, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | PACK-021 (Starter) | PACK-022 (Acceleration) | PACK-028 (Sector Pathway) |
|-----------|-------------------|------------------------|---------------------------|
| Sector coverage | Generic (ACA only) | 12 SDA sectors | 15+ sectors (SDA + IEA integration) |
| Intensity metrics | Basic (tCO2e/revenue) | Sector-specific | 20+ sector intensity metrics |
| Pathway scenarios | Single (1.5°C) | 3 scenarios | 5 scenarios (1.5°C, WB2C, 2°C, APS, STEPS) |
| Technology roadmaps | Generic actions | Technology categories | Technology-specific adoption schedules |
| Abatement detail | MACC (generic) | MACC by action | Waterfall by sector lever |
| Sector benchmarks | Basic peer comparison | SBTi peer comparison | IEA pathway + peer + leader benchmarking |
| Gap analysis | Target vs. current | Target vs. trajectory | Sector pathway convergence analysis |
| Data sources | GHG Protocol | GHG Protocol + SBTi | GHG Protocol + SBTi + IEA NZE + IPCC AR6 |

### 1.4 Target Users

**Primary:**
- Sustainability directors in carbon-intensive sectors (energy, industry, transport)
- Climate strategy teams requiring SBTi SDA pathway validation
- Board members evaluating sector-specific transition strategies
- Companies in power, steel, cement, chemicals, aviation, shipping sectors

**Secondary:**
- Financial institutions assessing portfolio alignment to sector pathways
- ESG analysts benchmarking companies against sector decarbonization trajectories
- Policy makers evaluating corporate climate plans against sectoral targets
- External auditors verifying sector pathway compliance

### 1.5 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Sector pathway accuracy | 100% match with SBTi SDA tool | Validated against SBTi sector pathway calculators |
| IEA scenario alignment | ±5% from IEA NZE milestones | Cross-validated with IEA NZE 2050 data tables |
| Intensity metric coverage | 15+ sectors, 20+ metrics | Number of sector-intensity metric pairs supported |
| Technology roadmap completeness | 100% of IEA key technologies | Technologies mapped per IEA NZE technology chapters |
| Gap analysis accuracy | ±2% from manual calculation | Tested against 100 sector pathway scenarios |
| Customer sector coverage | >80% of SBTi-committed companies | Sectors represented in customer base |

---

## 2. Sector Coverage

### 2.1 Primary Sectors (SBTi SDA Coverage)

| Sector | SDA Methodology | Intensity Metric | IEA Chapter |
|--------|----------------|------------------|-------------|
| Power Generation | SDA-Power | gCO2/kWh | Chapter 3: Electricity |
| Steel | SDA-Steel | tCO2e/tonne crude steel | Chapter 5: Industry (Steel) |
| Cement | SDA-Cement | tCO2e/tonne cement | Chapter 5: Industry (Cement) |
| Aluminum | SDA-Aluminum | tCO2e/tonne aluminum | Chapter 5: Industry (Aluminum) |
| Pulp & Paper | SDA-Pulp | tCO2e/tonne pulp | Chapter 5: Industry (Pulp) |
| Chemicals | SDA-Chemicals | tCO2e/tonne product | Chapter 5: Industry (Chemicals) |
| Aviation | SDA-Aviation | gCO2/pkm (passenger-km) | Chapter 4: Transport (Aviation) |
| Shipping | SDA-Shipping | gCO2/tkm (tonne-km) | Chapter 4: Transport (Shipping) |
| Road Transport | SDA-Transport | gCO2/vkm (vehicle-km) | Chapter 4: Transport (Road) |
| Rail | SDA-Rail | gCO2/pkm | Chapter 4: Transport (Rail) |
| Buildings (Residential) | SDA-Buildings | kgCO2/m²/year | Chapter 2: Buildings (Residential) |
| Buildings (Commercial) | SDA-Buildings | kgCO2/m²/year | Chapter 2: Buildings (Commercial) |

### 2.2 Extended Sectors (IEA NZE Coverage)

| Sector | Intensity Metric | IEA Chapter |
|--------|------------------|-------------|
| Agriculture | tCO2e/tonne food | Chapter 6: Agriculture |
| Food & Beverage | tCO2e/tonne product | Chapter 5: Industry (Food) |
| Oil & Gas (Upstream) | gCO2/MJ energy produced | Chapter 1: Energy Supply |
| Cross-Sector | Generic ACA fallback | Multiple chapters |

---

## 3. Core Components

### 3.1 Engines (8 Sector Pathway Engines)

| # | Engine | Purpose | Key Inputs | Key Outputs |
|---|--------|---------|------------|-------------|
| 1 | `sector_classification_engine.py` | Automatic sector classification using NACE, GICS, ISIC codes with SDA sector mapping | Company profile, revenue breakdown, activity data | SectorClass: primary sector, sub-sectors, SDA eligibility |
| 2 | `intensity_calculator_engine.py` | Sector-specific intensity metric calculation (20+ metrics) with data normalization | Activity data, emissions, sector type | IntensityMetrics: base year intensity, current intensity, trend |
| 3 | `pathway_generator_engine.py` | SBTi SDA + IEA NZE pathway generation for 15+ sectors with 5 scenario support | Sector, base year intensity, target year, scenario | PathwayResult: year-by-year intensity targets, convergence curve |
| 4 | `convergence_analyzer_engine.py` | Sector intensity convergence analysis vs. SBTi/IEA benchmarks with gap quantification | Current trajectory, sector pathway | ConvergenceAnalysis: gap to pathway, required acceleration, risk level |
| 5 | `technology_roadmap_engine.py` | Technology transition roadmaps with IEA milestone mapping (400+ milestones) | Sector pathway, technology availability, CapEx budget | TechRoadmap: technology adoption schedule, CapEx phasing, dependencies |
| 6 | `abatement_waterfall_engine.py` | Sector-specific abatement waterfall with lever-by-lever contribution analysis | Sector pathway, reduction actions, costs | AbatementWaterfall: lever contributions, cost curves, sequencing |
| 7 | `sector_benchmark_engine.py` | Multi-dimensional sector benchmarking (peer, leader, SBTi-validated, IEA pathway) | Company metrics, sector, region | BenchmarkResult: percentile vs. peers, gap to leader, pathway alignment |
| 8 | `scenario_comparison_engine.py` | Multi-scenario pathway comparison (1.5°C, WB2C, 2°C, APS, STEPS) with risk analysis | Sector, scenarios, uncertainty ranges | ScenarioComparison: scenario comparison matrix, risk assessment, optimal path |

### 3.2 Workflows (6 Sector Pathway Workflows)

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `sector_pathway_design_workflow.py` | 5: SectorClassify -> IntensityCalc -> PathwayGen -> GapAnalysis -> ValidationReport | Design SBTi SDA-aligned sector pathway |
| 2 | `pathway_validation_workflow.py` | 4: DataValidation -> PathwayValidation -> SBTiCheck -> ComplianceReport | Validate sector pathway against SBTi criteria |
| 3 | `technology_planning_workflow.py` | 5: TechInventory -> RoadmapGen -> CapExMapping -> DependencyAnalysis -> ImplementationPlan | Build sector-specific technology transition roadmap |
| 4 | `progress_monitoring_workflow.py` | 4: IntensityUpdate -> ConvergenceCheck -> BenchmarkUpdate -> ProgressReport | Monitor sector intensity progress vs. pathway |
| 5 | `multi_scenario_analysis_workflow.py` | 5: ScenarioSetup -> PathwayModeling -> RiskAnalysis -> ScenarioCompare -> StrategyRecommend | Compare pathways across climate scenarios |
| 6 | `full_sector_assessment_workflow.py` | 7: Classify -> Pathway -> Technology -> Abatement -> Benchmark -> Scenarios -> Strategy | End-to-end sector pathway assessment |

### 3.3 Templates (8 Sector Pathway Reports)

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `sector_pathway_report.py` | MD, HTML, JSON, PDF | Sector-specific decarbonization pathway with SDA/IEA alignment |
| 2 | `intensity_convergence_report.py` | MD, HTML, JSON, PDF | Intensity metric tracking and convergence analysis |
| 3 | `technology_roadmap_report.py` | MD, HTML, JSON, PDF | Technology transition roadmap with IEA milestones |
| 4 | `abatement_waterfall_report.py` | MD, HTML, JSON, PDF | Sector abatement waterfall with lever contributions |
| 5 | `sector_benchmark_report.py` | MD, HTML, JSON, PDF | Multi-dimensional sector benchmarking dashboard |
| 6 | `scenario_comparison_report.py` | MD, HTML, JSON, PDF | Multi-scenario pathway comparison and risk analysis |
| 7 | `sbti_validation_report.py` | MD, HTML, JSON, PDF | SBTi SDA pathway validation and compliance report |
| 8 | `sector_strategy_report.py` | MD, HTML, JSON, PDF | Executive sector transition strategy document |

### 3.4 Integrations (10 System Bridges)

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 10-phase DAG pipeline with sector-specific conditional routing |
| 2 | `sbti_sda_bridge.py` | SBTi SDA sector pathway data and validation tools integration |
| 3 | `iea_nze_bridge.py` | IEA Net Zero by 2050 sector pathway and milestone data integration |
| 4 | `ipcc_ar6_bridge.py` | IPCC AR6 sector-specific emission factors and pathways integration |
| 5 | `pack021_bridge.py` | PACK-021 baseline and target engines integration |
| 6 | `mrv_bridge.py` | All 30 MRV agents for sector-specific emissions calculation |
| 7 | `decarb_bridge.py` | Decarbonization agents for sector-specific reduction actions |
| 8 | `data_bridge.py` | 20 DATA agents for sector activity data intake |
| 9 | `health_check.py` | 20-category system verification including sector data freshness |
| 10 | `setup_wizard.py` | 7-step guided sector pathway configuration wizard |

### 3.5 Presets (6 Sector Configurations)

| # | Preset | Sectors | Key Characteristics |
|---|--------|---------|---------------------|
| 1 | `heavy_industry.yaml` | Steel, Cement, Aluminum, Chemicals | SDA mandatory, very high process emissions, CCS/hydrogen pathways |
| 2 | `power_utilities.yaml` | Power Generation, District Heating | SDA mandatory, grid decarbonization, renewable capacity expansion |
| 3 | `transport.yaml` | Aviation, Shipping, Road, Rail | SDA for each mode, fuel switching (SAF, hydrogen, electric) |
| 4 | `buildings.yaml` | Residential, Commercial Real Estate | SDA buildings pathway, energy efficiency, heat pumps, district heating |
| 5 | `light_industry.yaml` | Pulp & Paper, Food & Beverage | SDA available, energy efficiency focus, biomass/bioenergy |
| 6 | `agriculture.yaml` | Agriculture, Land Use | FLAG pathway integration, N2O/CH4 reduction, soil carbon |

---

## 4. Technical Requirements

### 4.1 SBTi SDA Methodology Compliance

**SDA Sector Pathway Requirements:**
- Sector classification per SBTi sector taxonomy (NACE Rev.2, GICS, ISIC Rev.4)
- Intensity convergence calculation using SBTi published convergence factors
- Base year intensity normalization and validation
- Target year intensity alignment with sector pathway
- Linear or non-linear convergence pathway generation
- Coverage requirement validation (95% Scope 1+2 for SDA sectors)

**SBTi Target Validation:**
- Near-term target ambition check (1.5°C alignment)
- Long-term target alignment (net-zero by 2050)
- Sector-specific target formulation verification
- Boundary consistency checks (operational vs. financial control)
- Double-counting prevention across scopes

### 4.2 IEA Net Zero Scenario Integration

**IEA NZE 2050 Data Integration:**
- Sector pathway data from IEA NZE 2050 report (2023 update)
- 400+ technology milestones mapped to sector pathways
- Regional pathway variants (OECD, emerging markets, global)
- Technology adoption curves (S-curves for renewable capacity, hydrogen, CCS)
- Cost assumptions from IEA Technology Collaboration Programmes

**Supported IEA Scenarios:**
- NZE (Net Zero Emissions by 2050) - 1.5°C, 50% probability
- APS (Announced Pledges Scenario) - 1.7°C
- STEPS (Stated Policies Scenario) - 2.4°C
- WB2C (Well-Below 2°C) - <2°C with 66% probability
- 2°C (2 Degrees Celsius Scenario) - 2°C with 50% probability

### 4.3 Sector-Specific Intensity Metrics

**Power Sector:**
- gCO2/kWh (grid average emission factor)
- tCO2e/MWh by generation source (coal, gas, nuclear, renewable)
- Capacity-weighted intensity (MW installed capacity)

**Steel Sector:**
- tCO2e/tonne crude steel (integrated blast furnace-basic oxygen furnace)
- tCO2e/tonne crude steel (electric arc furnace with scrap)
- tCO2e/tonne direct reduced iron (DRI)

**Cement Sector:**
- tCO2e/tonne clinker (process + combustion emissions)
- tCO2e/tonne cement (with clinker-to-cement ratio)
- tCO2e/m³ concrete (downstream)

**Aviation Sector:**
- gCO2/pkm (passenger-kilometer)
- gCO2/RTK (revenue tonne-kilometer)
- L fuel / 100 pkm (efficiency metric)

**Buildings Sector:**
- kgCO2/m²/year (operational emissions)
- kgCO2/m² (embodied emissions, lifetime)
- kWh/m²/year (energy use intensity)

### 4.4 Multi-Year Pathway Projections

**Pathway Time Horizons:**
- Near-term: Base year → 2030 (annual granularity)
- Medium-term: 2030 → 2040 (annual granularity)
- Long-term: 2040 → 2050 (annual granularity)
- Extended: 2050 → 2070 (5-year granularity for residual emissions planning)

**Convergence Modeling:**
- Linear convergence (straight-line reduction)
- Exponential convergence (accelerating reduction)
- S-curve convergence (technology adoption dynamics)
- Stepped convergence (policy-driven milestones)

### 4.5 Science-Based Target Validation

**Validation Criteria:**
- Sector pathway alignment within ±10% of SBTi benchmark
- Coverage requirements (95% Scope 1+2 for SDA sectors)
- Ambition level (1.5°C alignment for near-term targets)
- Boundary consistency (no exclusions without justification)
- Base year recalculation for structural changes (M&A, divestitures)
- Timeframe compliance (5-10 years for near-term, by 2050 for long-term)

**Validation Outputs:**
- SBTi submission package generator
- Validation report with pass/fail criteria
- Gap analysis for failed criteria
- Improvement recommendations

---

## 5. Key Features

### 5.1 15+ Sector Pathway Coverage

**Power Sector:**
- Coal-to-renewable transition pathways
- Gas peaking plant phase-out schedules
- Renewable capacity expansion curves (solar PV, wind, hydro, geothermal)
- Grid energy storage deployment (batteries, pumped hydro, hydrogen)
- Nuclear capacity modeling (baseload, SMR deployment)

**Steel Sector:**
- Blast furnace efficiency improvements
- Electric arc furnace (EAF) transition with scrap availability
- Direct reduced iron (DRI) with green hydrogen
- Carbon capture and storage (CCS) for integrated steel plants
- Scrap recycling rate optimization

**Cement Sector:**
- Clinker-to-cement ratio reduction
- Alternative fuels (biomass, waste-derived)
- Carbon capture utilization and storage (CCUS)
- Low-carbon cement alternatives (blended cements, geopolymers)
- Process efficiency improvements (high-efficiency kilns)

**Aluminum Sector:**
- Primary aluminum efficiency (Hall-Héroult process optimization)
- Secondary aluminum (recycling) expansion
- Inert anode technology deployment
- Renewable electricity procurement for smelting
- Low-carbon alumina production

**Aviation Sector:**
- Fleet renewal with fuel-efficient aircraft
- Sustainable aviation fuel (SAF) adoption curve
- Operational efficiency (load factor, routing optimization)
- Hydrogen aircraft deployment timeline (short-haul)
- Electric aircraft for ultra-short-haul (<500 km)

**Shipping Sector:**
- Fleet efficiency improvements (hull design, propulsion)
- Alternative fuels (LNG, methanol, ammonia, hydrogen)
- Wind-assisted propulsion
- Slow steaming and route optimization
- Port electrification (shore power)

**Buildings Sector:**
- Building envelope efficiency (insulation, windows)
- Heating system transition (gas boilers → heat pumps)
- District heating/cooling integration
- On-site renewable generation (rooftop solar)
- Smart building energy management

### 5.2 SBTi Corporate Standard & FLAG Guidance Alignment

**Corporate Standard Alignment:**
- Scope 1+2 coverage requirements (95% for SDA sectors)
- Scope 3 coverage thresholds (67% of total Scope 3)
- Near-term target setting (5-10 years, 4.2% annual linear reduction for 1.5°C)
- Long-term target setting (by 2050, 90%+ reduction)
- Neutralization of residual emissions (permanent removals only)

**FLAG (Forest, Land, and Agriculture) Guidance:**
- Land-use change emissions accounting
- Agricultural emissions (N2O from fertilizer, CH4 from livestock, rice)
- Soil carbon sequestration potential
- Avoided deforestation credits
- Afforestation/reforestation pathways
- FLAG-specific target formulation (separate from fossil targets)

### 5.3 IEA NZE 2050 Scenario Integration

**Technology Milestone Mapping:**
- 400+ IEA technology milestones mapped to sector pathways
- Technology readiness level (TRL) assessment
- Cost decline curves (learning rates from IEA cost databases)
- Regional technology availability (OECD vs. emerging markets)
- Technology interdependencies (hydrogen production → steel DRI)

**IEA Sector Pathway Integration:**
- Direct import of IEA sector pathway CSVs
- Automatic pathway adjustment for regional context
- Scenario switching (NZE, APS, STEPS, WB2C, 2°C)
- Milestone validation (company progress vs. IEA milestones)

### 5.4 Sector Intensity Convergence Calculations

**Convergence Methodology:**
- Global convergence: All companies in sector converge to same absolute intensity by 2050
- Regional convergence: Regional pathways with differentiated intensity targets
- Peer convergence: Best-in-class peer intensity as convergence target
- Hybrid convergence: Combination of global pathway + peer benchmarking

**Mathematical Models:**
- Linear convergence: `Intensity(t) = Intensity(base) - (t - base_year) × reduction_rate`
- Exponential convergence: `Intensity(t) = Intensity(base) × exp(-k × (t - base_year))`
- S-curve convergence: `Intensity(t) = Intensity(2050) + (Intensity(base) - Intensity(2050)) / (1 + exp(k × (t - t_inflection)))`

### 5.5 Multi-Scenario Modeling

**Scenario Framework:**
| Scenario | Temperature | Probability | IEA Reference | SBTi Alignment |
|----------|-------------|-------------|---------------|----------------|
| 1.5°C (NZE) | +1.5°C | 50% | IEA NZE 2050 | SBTi 1.5°C pathway |
| WB2C | <2°C | 66% | IEA WB2C | SBTi well-below 2°C |
| 2°C | +2°C | 50% | IEA 2DS | SBTi 2°C pathway |
| APS | +1.7°C | N/A | IEA APS | Announced pledges |
| STEPS | +2.4°C | N/A | IEA STEPS | Current policies |

**Scenario Comparison:**
- Side-by-side pathway comparison
- Investment requirement deltas
- Technology adoption timeline differences
- Risk-return analysis per scenario
- Optimal pathway recommendation with sensitivity analysis

### 5.6 Gap Analysis vs. Sector Benchmarks

**Benchmark Sources:**
- SBTi-validated companies in same sector
- IEA sector pathway milestones
- Peer group intensity averages (by revenue band, region)
- Sector leaders (top decile intensity performers)
- Regulatory requirements (EU ETS benchmarks, EPA standards)

**Gap Metrics:**
- Intensity gap (current vs. pathway, % difference)
- Time-to-convergence (years until pathway alignment)
- Required acceleration rate (additional % reduction/year needed)
- Investment gap (CapEx delta to close gap)
- Technology gap (technology adoption lag vs. leaders)

### 5.7 Abatement Waterfall Analysis

**Sector-Specific Levers:**

**Power Sector:**
1. Renewable capacity expansion (solar, wind, hydro)
2. Coal plant phase-out / retirement
3. Gas peaking plant efficiency
4. Grid energy storage deployment
5. Demand response and smart grid
6. Nuclear capacity (baseload or SMR)
7. Carbon capture and storage (fossil generation)

**Steel Sector:**
1. Blast furnace efficiency improvements
2. Electric arc furnace transition
3. Green hydrogen DRI deployment
4. CCS for integrated plants
5. Scrap recycling rate increase
6. Energy efficiency (waste heat recovery)

**Cement Sector:**
1. Clinker substitution (fly ash, slag, calcined clay)
2. Alternative fuels (biomass, waste)
3. Energy efficiency (high-efficiency kilns)
4. Carbon capture and storage
5. Low-carbon cement products
6. Circular economy (concrete reuse)

**Waterfall Calculation:**
- Lever contribution to total abatement (tCO2e, %)
- Cost per tCO2e abated (EUR/tCO2e)
- Implementation timeline (year-by-year deployment)
- Lever interdependencies (e.g., EAF steel requires renewable electricity)
- Cumulative abatement curve (waterfall chart)

### 5.8 Technology Transition Roadmaps

**Roadmap Components:**
- Technology inventory (current state, TRL levels)
- Adoption curves (S-curve modeling with market penetration %)
- CapEx phasing (multi-year investment schedule)
- OpEx impact (change in operating costs post-adoption)
- Dependency mapping (technology prerequisites)
- Risk assessment (technology maturity, supply chain, cost uncertainty)

**IEA Milestone Integration:**
- Automatic milestone mapping from IEA NZE 2050 tables
- Milestone compliance tracking (on-track vs. off-track)
- Gap alerts for missed milestones
- Catch-up scenario modeling

---

## 6. Regulatory Alignment

### 6.1 SBTi Corporate Standard v2.0

**Alignment Points:**
- Near-term target ambition (1.5°C alignment, 4.2% annual reduction)
- Long-term target ambition (90%+ reduction by 2050)
- Sector-specific pathway validation (SDA for 12 sectors)
- Coverage requirements (95% Scope 1+2, 67% Scope 3)
- Target recalculation triggers (structural changes >5%)
- Neutralization vs. compensation distinction

### 6.2 SBTi FLAG Guidance

**Alignment Points:**
- Separate FLAG target formulation
- Land-use change accounting methodologies
- Soil carbon sequestration quantification
- FLAG Scope 3 coverage (supply chain land use)
- FLAG-specific emission factors (IPCC Agriculture sector)

### 6.3 IEA Net Zero by 2050 Roadmap

**Alignment Points:**
- Sector pathway data (15+ sectors)
- Technology milestone tracking (400+ milestones)
- Regional pathway variants (global, OECD, emerging markets)
- Cost assumptions (IEA technology cost databases)
- Energy demand projections (sector activity forecasts)

### 6.4 IPCC AR6 Pathways

**Alignment Points:**
- Global warming potential (GWP-100) values from AR6
- Sector-specific emission factors (IPCC 2006 Guidelines with 2019 refinements)
- Carbon budget alignment (global and sector-specific)
- Mitigation pathway scenarios (SSP1-1.9, SSP1-2.6)

### 6.5 National Climate Plans (NDCs)

**Alignment Points:**
- Sector targets from Nationally Determined Contributions
- Technology deployment targets (renewable capacity, EV adoption)
- Policy milestone tracking (carbon pricing, efficiency standards)
- Regional ambition levels (EU vs. US vs. China vs. India)

---

## 7. Agent Dependencies

### 7.1 MRV Agents (30)
All 30 AGENT-MRV agents for sector-specific emission calculations via `mrv_bridge.py`.

### 7.2 Data Agents (20)
All 20 AGENT-DATA agents for sector activity data intake via `data_bridge.py`.

### 7.3 Foundation Agents (10)
All 10 AGENT-FOUND agents for orchestration, schema, units, audit, etc.

### 7.4 Application Dependencies
- GL-GHG-APP: GHG inventory management
- GL-SBTi-APP: SBTi pathway validation and temperature scoring
- GL-CDP-APP: CDP sector-specific questionnaire responses
- GL-TCFD-APP: TCFD scenario analysis and metrics
- GL-Taxonomy-APP: EU Taxonomy sector criteria (energy, industry, transport)

### 7.5 Pack Dependencies
- PACK-021: Net Zero Starter Pack (baseline, targets, gap analysis)
- PACK-022: Net Zero Acceleration Pack (optional, for advanced scenario modeling)

---

## 8. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Sector pathway accuracy | 100% match with SBTi SDA tool | Cross-validation with SBTi official calculators |
| IEA scenario alignment | ±5% from IEA NZE milestones | Validation against IEA NZE 2050 data tables |
| Sector coverage | 15+ sectors | Number of sectors with full pathway support |
| Intensity metric coverage | 20+ sector-specific metrics | Count of sector-intensity metric pairs |
| Technology milestone coverage | 400+ IEA milestones | Number of IEA milestones mapped to sectors |
| Convergence calculation accuracy | ±2% from manual calculation | Tested against 100+ sector pathway scenarios |
| Customer sector representation | >70% of SBTi-committed sectors | Sectors represented in customer base |
| Pathway generation speed | <5 min per sector | Time from input to validated pathway output |

---

**Document Status:** Approved
**Next Steps:** Create detailed to-do list, launch parallel development agents
**Dependencies:** PACK-021, PACK-022, all MRV/DATA/FOUND agents
**Estimated Development Time:** 40-60 hours across 8 parallel agents

---

**End of PRD**
