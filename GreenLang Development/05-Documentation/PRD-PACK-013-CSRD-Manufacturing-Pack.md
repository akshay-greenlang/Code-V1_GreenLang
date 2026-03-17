# PRD-PACK-013: CSRD Manufacturing Pack

**Version**: 1.0
**Status**: APPROVED
**Author**: GreenLang Platform Team
**Date**: 2026-03-16
**Category**: EU Compliance Solution Pack

---

## 1. Executive Summary

PACK-013 is the **CSRD Manufacturing Pack**, a sector-specific Solution Pack targeting manufacturing companies (NACE Division C: C10-C33) required to comply with the EU Corporate Sustainability Reporting Directive (CSRD). Since the Omnibus I Directive (2026/470) permanently eliminated sector-specific ESRS standards, this pack fills the gap by providing manufacturing-specific intelligence, calculation engines, and reporting templates that the regulation no longer prescribes.

The pack covers: process emissions (cement, steel, aluminum, chemicals, glass), product carbon footprints (ISO 14067/ESPR/DPP), energy intensity metrics (MJ/unit), circular economy (waste diversion, recycled content, EPR), water & pollution (ESRS E2/E3, IED), BAT compliance (BREF/BAT-AEL), and manufacturing supply chain traceability (Scope 3 Cat 1/4/5/11/12).

**Target Users**: Manufacturing companies with >1,000 employees AND >EUR 450M turnover (Omnibus I threshold), subject to CSRD reporting under ESRS Set 1.

---

## 2. Regulatory Scope

### 2.1 Primary Regulations
- **CSRD** (Directive 2022/2464): Corporate Sustainability Reporting Directive
- **ESRS Set 1** (Delegated Regulation 2023/2772): Cross-sector standards E1-E5, S1-S4, G1
- **Omnibus I** (Directive 2026/470): Revised thresholds, 61% datapoint reduction, materiality-only
- **EU ETS Phase IV**: 4.3% annual linear reduction factor, free allocation phase-out 2026-2034
- **CBAM** (Regulation 2023/956): Carbon Border Adjustment for cement, steel, aluminum, fertilizers, electricity, hydrogen
- **IED 2.0** (Directive 2024/XX): Industrial Emissions Directive recast, BAT-AEL compliance
- **EED** (Directive 2023/1791): Energy Efficiency Directive, energy audits/ISO 50001
- **ESPR** (Regulation 2024/XX): Ecodesign for Sustainable Products, Digital Product Passport
- **EUDR** (Regulation 2023/1115): Deforestation-free supply chains

### 2.2 Standards & Frameworks
- **GHG Protocol**: Corporate Standard, Scope 3 Standard, Product Life Cycle Standard
- **ISO 14064-1**: Organization-level GHG quantification
- **ISO 14067**: Product carbon footprint
- **ISO 50001**: Energy management systems
- **TNFD**: Taskforce on Nature-related Financial Disclosures (LEAP approach)
- **SBTi**: Science-Based Targets initiative (sector pathways for manufacturing)
- **BREFs**: Best Available Techniques Reference Documents (18 sector-specific)

---

## 3. Architecture

### 3.1 Pack Structure
```
PACK-013-csrd-manufacturing/
├── __init__.py
├── pack.yaml
├── config/
│   ├── __init__.py
│   ├── pack_config.py
│   ├── presets/
│   │   ├── heavy_industry.yaml      # Steel, cement, aluminum, glass
│   │   ├── discrete_manufacturing.yaml  # Automotive, electronics, machinery
│   │   ├── process_manufacturing.yaml   # Chemicals, food & beverage, textiles
│   │   ├── light_manufacturing.yaml     # Consumer goods, packaging
│   │   ├── multi_site.yaml              # Multi-facility groups
│   │   └── sme_manufacturer.yaml        # SME manufacturers
│   └── demo/
│       └── demo_config.yaml
├── engines/
│   ├── __init__.py
│   ├── process_emissions_engine.py       # Engine 1: Industrial process emissions
│   ├── energy_intensity_engine.py        # Engine 2: Manufacturing energy metrics
│   ├── product_carbon_footprint_engine.py # Engine 3: Product-level PCF
│   ├── circular_economy_engine.py        # Engine 4: Circular economy metrics
│   ├── water_pollution_engine.py         # Engine 5: Water & pollution tracking
│   ├── bat_compliance_engine.py          # Engine 6: BAT/BREF compliance
│   ├── supply_chain_emissions_engine.py  # Engine 7: Manufacturing Scope 3
│   └── manufacturing_benchmark_engine.py # Engine 8: Sector benchmarking
├── workflows/
│   ├── __init__.py
│   ├── manufacturing_emissions_workflow.py    # Workflow 1: Full emissions assessment
│   ├── product_pcf_workflow.py               # Workflow 2: Product carbon footprint
│   ├── circular_economy_workflow.py          # Workflow 3: Circular economy readiness
│   ├── bat_compliance_workflow.py            # Workflow 4: BAT compliance assessment
│   ├── supply_chain_assessment_workflow.py   # Workflow 5: Supply chain emissions
│   ├── esrs_manufacturing_workflow.py        # Workflow 6: ESRS disclosure for manufacturing
│   ├── decarbonization_roadmap_workflow.py   # Workflow 7: Decarbonization planning
│   └── regulatory_compliance_workflow.py     # Workflow 8: Multi-regulation compliance
├── templates/
│   ├── __init__.py
│   ├── process_emissions_report.py           # Template 1: Process emissions breakdown
│   ├── product_pcf_label.py                  # Template 2: Product carbon footprint label
│   ├── energy_performance_report.py          # Template 3: Energy intensity dashboard
│   ├── circular_economy_report.py            # Template 4: Circular economy metrics
│   ├── bat_compliance_report.py              # Template 5: BAT compliance assessment
│   ├── water_pollution_report.py             # Template 6: Water & pollution disclosure
│   ├── manufacturing_scorecard.py            # Template 7: Manufacturing sustainability scorecard
│   └── decarbonization_roadmap.py            # Template 8: Decarbonization pathway
├── integrations/
│   ├── __init__.py
│   ├── pack_orchestrator.py                  # Master orchestrator (11-phase)
│   ├── csrd_pack_bridge.py                   # Bridge to PACK-001/002/003
│   ├── cbam_pack_bridge.py                   # Bridge to PACK-004/005
│   ├── mrv_industrial_bridge.py              # Bridge to MRV industrial agents
│   ├── data_manufacturing_bridge.py          # Bridge to DATA agents + ERP/MES
│   ├── eu_ets_bridge.py                      # EU ETS registry integration
│   ├── taxonomy_bridge.py                    # EU Taxonomy alignment for manufacturing
│   ├── supply_chain_bridge.py                # Supply chain traceability integration
│   ├── health_check.py                       # 22-category health verification
│   └── setup_wizard.py                       # Manufacturing-specific guided setup
└── tests/
    ├── conftest.py
    ├── test_manifest.py
    ├── test_config.py
    ├── test_demo.py
    ├── test_process_emissions.py
    ├── test_energy_intensity.py
    ├── test_product_pcf.py
    ├── test_circular_economy.py
    ├── test_water_pollution.py
    ├── test_bat_compliance.py
    ├── test_supply_chain_emissions.py
    ├── test_manufacturing_benchmark.py
    ├── test_workflows.py
    ├── test_templates.py
    ├── test_integrations.py
    ├── test_e2e.py
    └── test_agent_integration.py
```

### 3.2 Components Summary
- **8 Engines**: Process emissions, energy intensity, product PCF, circular economy, water/pollution, BAT compliance, supply chain emissions, manufacturing benchmark
- **8 Workflows**: Manufacturing emissions, product PCF, circular economy, BAT compliance, supply chain, ESRS manufacturing, decarbonization roadmap, regulatory compliance
- **8 Templates**: Process emissions, PCF label, energy performance, circular economy, BAT compliance, water/pollution, manufacturing scorecard, decarbonization roadmap
- **10 Integrations**: Pack orchestrator, CSRD bridge, CBAM bridge, MRV industrial bridge, data manufacturing bridge, EU ETS bridge, taxonomy bridge, supply chain bridge, health check, setup wizard
- **6 Presets**: Heavy industry, discrete manufacturing, process manufacturing, light manufacturing, multi-site, SME manufacturer

---

## 4. Engine Specifications

### 4.1 Engine 1: Process Emissions Engine
**File**: `engines/process_emissions_engine.py`
**Purpose**: Calculate industrial process emissions for manufacturing sub-sectors

**Key Features**:
- **Sub-sector models**: Cement (clinker calcination, kiln fuel, SCM), Steel (BF-BOF, EAF, DRI-H2), Aluminum (electrolysis PFC), Chemicals (ammonia, nitric acid, adipic acid N2O), Glass (batch decomposition, cullet ratio), Ceramics, Pulp & Paper
- **Process chemistry**: Stoichiometric CO2 from raw material decomposition (e.g., CaCO3 → CaO + CO2)
- **Emission factors**: EU ETS product benchmarks (clinker 0.766 tCO2/t, hot metal 1.328 tCO2/t, aluminum 1.514 tCO2/t)
- **CBAM embedded emissions**: Simple and complex goods calculation per CBAM methodology
- **Abatement tracking**: CCS/CCUS captured amounts, alternative feedstock CO2 reduction

**Models**: ProcessEmissionsConfig, FacilityData, ProcessLine, ProcessEmissionsResult, SubSectorBreakdown, CBAMEmbeddedEmissions

### 4.2 Engine 2: Energy Intensity Engine
**File**: `engines/energy_intensity_engine.py`
**Purpose**: Calculate manufacturing-specific energy performance metrics

**Key Features**:
- **Specific Energy Consumption (SEC)**: MJ/tonne product, MJ/unit, MJ/EUR revenue
- **Energy source breakdown**: Electricity, natural gas, coal, biomass, hydrogen, steam, heat
- **Benchmark comparison**: BAT-AEL energy benchmarks per BREF document
- **ISO 50001 compliance**: Energy baseline, EnPIs, targets, improvement tracking
- **EED obligations**: Energy audit compliance (10-85 TJ/year audit, >85 TJ/year ISO 50001)
- **Decarbonization potential**: Electrification, green hydrogen, heat pump, waste heat recovery

**Models**: EnergyIntensityConfig, EnergyConsumptionData, ProductionVolumeData, EnergyIntensityResult, BenchmarkComparison, DecarbonizationOpportunity

### 4.3 Engine 3: Product Carbon Footprint Engine
**File**: `engines/product_carbon_footprint_engine.py`
**Purpose**: Calculate product-level carbon footprints per ISO 14067

**Key Features**:
- **Lifecycle stages**: Cradle-to-gate, gate-to-gate, cradle-to-grave
- **Allocation methods**: Mass-based, economic, physical causality
- **BOM integration**: Bill of Materials → emission factor lookup → allocated emissions
- **ESPR/DPP readiness**: Digital Product Passport data attributes (100+ fields)
- **Battery Regulation**: Carbon footprint declaration for battery products
- **Category rules**: Product Category Rules (PCR) for sector-specific methodologies
- **Data quality scoring**: PCAF-inspired 5-level DQ for product data

**Models**: PCFConfig, ProductData, BOMComponent, LifecycleStage, PCFResult, DPPData, AllocationResult

### 4.4 Engine 4: Circular Economy Engine
**File**: `engines/circular_economy_engine.py`
**Purpose**: Calculate circular economy metrics per ESRS E5

**Key Features**:
- **Material Circularity Index (MCI)**: Ellen MacArthur Foundation methodology
- **Recycled content tracking**: Pre-consumer and post-consumer recycled content %
- **Waste streams**: Hazardous, non-hazardous, by type (metal, plastic, organic, e-waste, packaging)
- **Waste hierarchy compliance**: Prevention > reuse > recycling > recovery > disposal
- **Extended Producer Responsibility**: EPR scheme fees, eco-modulation, recycling targets
- **Critical Raw Materials**: CRM Regulation compliance (25% recycling capacity by 2030)
- **Product recyclability**: Design for disassembly, recyclability score

**Models**: CircularEconomyConfig, MaterialFlowData, WasteStreamData, CircularEconomyResult, MCIResult, EPRCompliance, WasteHierarchyBreakdown

### 4.5 Engine 5: Water & Pollution Engine
**File**: `engines/water_pollution_engine.py`
**Purpose**: Track water usage and pollution per ESRS E2/E3

**Key Features**:
- **Water balance**: Intake (surface, ground, third-party, rainwater) → consumption → discharge
- **Water stress assessment**: WRI Aqueduct methodology, site-level water stress scoring
- **Pollutant tracking**: Nitrogen, phosphorus, heavy metals, VOCs, particulates, SOx, NOx
- **REACH/CLP compliance**: Substances of Very High Concern (SVHC) > 0.1% w/w tracking
- **IED wastewater**: Industrial wastewater treatment compliance, BAT-AEL for water
- **ESRS E2 metrics**: Pollutant emissions to air/water/soil, microplastics
- **ESRS E3 metrics**: Water withdrawal/discharge/consumption by source, water-stressed breakdown

**Models**: WaterPollutionConfig, WaterIntakeData, WaterDischargeData, PollutantData, WaterPollutionResult, WaterStressAssessment, PollutantInventory

### 4.6 Engine 6: BAT Compliance Engine
**File**: `engines/bat_compliance_engine.py`
**Purpose**: Check compliance against Best Available Techniques per IED

**Key Features**:
- **BREF database**: 18 sector-specific + cross-cutting BREF reference documents
- **BAT-AEL comparison**: Compare facility emissions intensity vs BAT-Associated Emission Levels
- **Compliance status**: Compliant, within range, non-compliant, derogation
- **Transformation plan**: IED 2.0 mandatory transformation plan readiness
- **Technology assessment**: TRL scoring, marginal abatement cost curves
- **Improvement roadmap**: BAT upgrade pathway, investment requirements, timeline
- **Penalty risk**: IED penalties (min EUR 3M or 3% annual turnover)

**Models**: BATConfig, FacilityBATData, BREFReference, BATAELRange, BATComplianceResult, TransformationPlan, TechnologyAssessment, AbatementOption

### 4.7 Engine 7: Supply Chain Emissions Engine
**File**: `engines/supply_chain_emissions_engine.py`
**Purpose**: Calculate manufacturing-specific Scope 3 emissions

**Key Features**:
- **BOM-based calculation**: Map bill of materials to emission factors per component
- **4 calculation methods**: Supplier-specific, hybrid, average-data, spend-based
- **Data quality scoring**: 5-level manufacturing DQ (verified > supplier-reported > industry average > spend-based > estimated)
- **Supplier engagement**: CDP Supply Chain, Catena-X, TfS, WBCSD PACT integration
- **Priority categories**: Cat 1 (purchased goods), Cat 4 (upstream transport), Cat 5 (waste), Cat 9 (downstream transport), Cat 11 (use of sold products), Cat 12 (end-of-life)
- **Hotspot analysis**: Identify top emitting suppliers, materials, components
- **Improvement tracking**: YoY supplier data quality improvement targets

**Models**: SupplyChainConfig, SupplierData, BOMEmissions, SupplyChainResult, HotspotAnalysis, SupplierScorecard, DataQualityAssessment

### 4.8 Engine 8: Manufacturing Benchmark Engine
**File**: `engines/manufacturing_benchmark_engine.py`
**Purpose**: Benchmark manufacturing sustainability performance against sector peers

**Key Features**:
- **Sector benchmarks**: Automotive, chemicals, food & beverage, textiles, electronics, steel, cement, aluminum
- **KPI comparison**: Emission intensity, energy intensity, water intensity, waste intensity, circularity rate
- **EU ETS benchmarks**: Product benchmark comparison, free allocation calculation
- **SBTi sector pathways**: Manufacturing-specific decarbonization targets (1.5C/WB2C)
- **Peer comparison**: Industry percentile ranking (top quartile, median, bottom quartile)
- **Improvement trajectory**: Historic trend analysis, target gap assessment
- **OEE sustainability overlay**: Overall Equipment Effectiveness × sustainability factor

**Models**: BenchmarkConfig, FacilityKPIs, SectorBenchmark, BenchmarkResult, PeerComparison, SBTiAlignment, TrajectoryAnalysis

---

## 5. Workflow Specifications

### 5.1 Manufacturing Emissions Workflow
4-phase: Data Collection → Process Emissions Calculation → Energy & Scope Analysis → Consolidation & Reporting

### 5.2 Product PCF Workflow
5-phase: Product Selection → BOM Mapping → Lifecycle Assessment → Allocation → PCF Label Generation

### 5.3 Circular Economy Workflow
4-phase: Material Flow Mapping → Waste Stream Analysis → Circularity Metrics → EPR Compliance

### 5.4 BAT Compliance Workflow
4-phase: BREF Identification → Performance Assessment → Gap Analysis → Transformation Planning

### 5.5 Supply Chain Assessment Workflow
5-phase: Supplier Inventory → Data Collection → Calculation → Hotspot Analysis → Engagement Plan

### 5.6 ESRS Manufacturing Workflow
4-phase: Materiality Assessment → Data Point Collection → Disclosure Generation → Audit Preparation

### 5.7 Decarbonization Roadmap Workflow
5-phase: Baseline → Technology Assessment → Target Setting → Investment Planning → Monitoring

### 5.8 Regulatory Compliance Workflow
3-phase: Regulation Mapping → Compliance Assessment → Action Plan

---

## 6. Integration Specifications

### 6.1 Pack Orchestrator
11-phase manufacturing pipeline: initialization → data_intake → quality_assurance → process_emissions → energy_analysis → product_pcf → circular_economy → water_pollution → bat_compliance → supply_chain → reporting

### 6.2 CSRD Pack Bridge
Bridge to PACK-001/002/003 for base CSRD reporting (ESRS E1 cross-sector metrics, materiality, governance)

### 6.3 CBAM Pack Bridge
Bridge to PACK-004/005 for CBAM-affected manufacturers (steel, cement, aluminum, fertilizers, hydrogen importers/producers)

### 6.4 MRV Industrial Bridge
Bridge to AGENT-MRV industrial agents (steel, cement, aluminum, chemicals, glass, pulp/paper, food processing) + all 30 standard MRV agents

### 6.5 Data Manufacturing Bridge
Bridge to AGENT-DATA agents with manufacturing-specific routing (ERP for BOM/production, supplier questionnaire for Scope 3, validation for process data)

### 6.6 EU ETS Bridge
Integration with EU ETS registry for compliance obligations, free allocation tracking, benchmark comparison

### 6.7 Taxonomy Bridge
EU Taxonomy alignment assessment for manufacturing activities (climate mitigation/adaptation substantial contribution, DNSH)

### 6.8 Supply Chain Bridge
Integration for multi-tier supply chain emissions data exchange (Catena-X, TfS, WBCSD PACT)

### 6.9 Health Check
22-category health verification covering all engines, workflows, templates, integrations, agent connections

### 6.10 Setup Wizard
8-step manufacturing-specific guided setup: FI type selection → sub-sector configuration → regulation mapping → data source connection → baseline calculation → target setting → workflow activation → go-live

---

## 7. Preset Specifications

### 7.1 Heavy Industry Preset
Steel, cement, aluminum, glass, ceramics. EU ETS + CBAM enabled. BAT compliance critical. High process emissions. Product benchmarks active.

### 7.2 Discrete Manufacturing Preset
Automotive, electronics, machinery, equipment. Product PCF critical. Supply chain dominant (Cat 1 + Cat 11). ESPR/DPP relevant. OEE tracking.

### 7.3 Process Manufacturing Preset
Chemicals, food & beverage, textiles, pharmaceuticals. Process emissions + water/pollution critical. REACH/CLP compliance. Batch process tracking.

### 7.4 Light Manufacturing Preset
Consumer goods, packaging, furniture. Circular economy focus. EPR compliance. Lower emissions intensity. Supply chain dominant.

### 7.5 Multi-Site Preset
Manufacturing groups with multiple facilities. Consolidation across sites. Transfer pricing for emissions. Group-level reporting.

### 7.6 SME Manufacturer Preset
Simplified reporting. Scope 1+2 focus. Optional Scope 3. Guided data entry. Industry-average emission factors.

---

## 8. Agent Dependencies (75 agents)

### 8.1 AGENT-MRV (30 agents)
All 30 MRV agents: Scope 1 (001-008), Scope 2 (009-013), Scope 3 (014-030)

### 8.2 AGENT-DATA (20 agents)
All 20 DATA agents: Intake (001-007), Quality (008-019), Geo (020)

### 8.3 AGENT-FOUND (10 agents)
All 10 Foundation agents: Orchestrator, Schema, Units, Assumptions, Citations, Access, Registry, Reproducibility, QA, Observability

### 8.4 Industrial MRV Agents (7)
Steel, Cement, Aluminum, Chemicals, Glass, Pulp/Paper, Food Processing

### 8.5 Other Agents (8)
IoT Streaming, Supply Chain ESG, ERP Connector (manufacturing modules), Validation Rule Engine, Climate Hazard Connector, Data Quality Profiler, Supplier Questionnaire, Data Lineage

---

## 9. Manufacturing Sub-Sectors Covered

| NACE Code | Sub-Sector | Key Engines | Dominant Scope 3 |
|-----------|-----------|-------------|-------------------|
| C10-C12 | Food, beverages, tobacco | Process, Water, Circular | Cat 1 (50-75%) |
| C13-C15 | Textiles, wearing apparel, leather | Circular, Supply Chain, Water | Cat 1 (40-60%) |
| C16-C18 | Wood, paper, printing | Process, BAT, Circular | Cat 1 (30-50%) |
| C19-C20 | Coke, petroleum, chemicals | Process, BAT, Water, ETS | Cat 1 (30-50%) |
| C21 | Pharmaceuticals | Water, Supply Chain | Cat 1 (40-60%) |
| C22-C23 | Rubber, plastics, minerals | Process, Circular, BAT | Cat 1 (30-50%) |
| C24 | Basic metals (steel, aluminum) | Process, ETS, CBAM, BAT | Cat 1 (20-40%) |
| C25 | Fabricated metal products | Energy, Supply Chain | Cat 1 (40-60%) |
| C26-C27 | Electronics, electrical equipment | PCF, Supply Chain, Circular | Cat 1+11 (60-80%) |
| C28 | Machinery & equipment | PCF, Supply Chain | Cat 1+11 (50-70%) |
| C29-C30 | Motor vehicles, transport equipment | PCF, Supply Chain, Circular | Cat 1+11 (70-90%) |
| C31-C33 | Furniture, other manufacturing, repair | Circular, Energy | Cat 1 (30-50%) |

---

## 10. Testing Strategy

### 10.1 Unit Tests (18 test files, 600+ tests target)
- conftest.py: Shared fixtures (manufacturing company, facility data, BOM, process data)
- test_manifest.py: Pack YAML validation (60+ tests)
- test_config.py: Config system tests (45+ tests)
- test_demo.py: Demo smoke tests (60+ tests)
- 8 engine test files (30-50 tests each)
- test_workflows.py: All 8 workflows (30+ tests)
- test_templates.py: All 8 templates + registry (28+ tests)
- test_integrations.py: All 10 integrations (20+ tests)
- test_e2e.py: End-to-end flows (12+ tests)
- test_agent_integration.py: Agent wiring (12+ tests)

### 10.2 Key Test Scenarios
- Cement plant: clinker calcination → process emissions → CBAM embedded → BAT check → ETS compliance
- Automotive OEM: BOM → product PCF → Scope 3 Cat 1/11 → SBTi alignment → ESPR/DPP
- Chemical plant: process chemistry → water/pollution → REACH → IED → BAT compliance
- Food manufacturer: agricultural inputs → water → waste → circular economy → EPR
- Multi-site group: facility consolidation → group reporting → ESRS disclosure

---

## 11. Regulatory Timeline

| Date | Regulation | Impact |
|------|-----------|--------|
| 2026-01-01 | CBAM 2.5% certificate surrender | Steel/cement/aluminum importers |
| 2026-07-01 | IED 2.0 transposition deadline | BAT-AEL compliance |
| 2026-10-01 | EED energy audit deadline (10-85 TJ/yr) | Manufacturing facilities |
| 2027-02-01 | Battery passport required | Battery manufacturers |
| 2027-10-01 | EED ISO 50001 deadline (>85 TJ/yr) | Large manufacturing |
| 2027-H2 | ESPR first delegated acts | Product categories TBD |
| 2028-01-01 | CBAM 10% certificate surrender | Increasing CBAM costs |
| 2029-07-01 | CSDDD single application date | Supply chain due diligence |
| 2030-01-01 | EU ETS -62% vs 2005 | Major ETS tightening |
| 2034-01-01 | CBAM 100% / ETS free allocation 0% | Full carbon border adjustment |

---

## 12. Acceptance Criteria

1. All 8 engines implement deterministic calculations with SHA-256 provenance hashing
2. All Pydantic v2 models with field_validator/model_validator (NO `from __future__ import annotations`)
3. All 6 presets load and validate without errors
4. All 8 workflows complete end-to-end with demo data
5. All 8 templates generate valid output
6. All 10 integrations pass health check
7. 600+ unit tests, 100% pass rate
8. Cross-pack bridges verify connectivity to PACK-001/002/003 (CSRD), PACK-004/005 (CBAM), PACK-008 (Taxonomy)
9. Manufacturing-specific KPIs: emission intensity (tCO2e/unit), energy intensity (MJ/unit), water intensity (m3/unit), waste intensity (kg/unit), circularity rate (%)
10. Demo mode: GreenManufacturing GmbH synthetic facility with cement + steel + chemicals processes

---

## 13. Non-Functional Requirements

- **Performance**: All engines complete in <5s for single facility, <30s for 100-facility group
- **Determinism**: Identical inputs produce identical outputs and provenance hashes
- **Extensibility**: New manufacturing sub-sectors addable via configuration without code changes
- **Auditability**: Complete calculation chain traceable from raw data to reported metric
- **Compatibility**: Python 3.11+, Pydantic v2, no external API calls required for core calculations

---

## 14. Dependencies

| Dependency | Component | Version | Required |
|-----------|-----------|---------|----------|
| pydantic | All | >=2.0 | Yes |
| pyyaml | Config | >=6.0 | Yes |
| PACK-001 | CSRD Starter | 1.0 | Bridge |
| PACK-004 | CBAM Readiness | 1.0 | Bridge |
| AGENT-MRV-001..030 | MRV Agents | 1.0 | Integration |
| AGENT-DATA-001..020 | Data Agents | 1.0 | Integration |
| AGENT-FOUND-001..010 | Foundation | 1.0 | Integration |
