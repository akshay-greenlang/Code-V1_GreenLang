# PACK-028: Sector Pathway Pack

**Pack ID:** PACK-028-sector-pathway
**Category:** Net Zero Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Production Ready
**Date:** 2026-03-19
**Author:** GreenLang Platform Engineering
**Prerequisite:** PACK-021 Net Zero Starter Pack (recommended)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Quick Start Guide](#quick-start-guide)
3. [Sector Coverage](#sector-coverage)
4. [Architecture Overview](#architecture-overview)
5. [Core Components](#core-components)
6. [Installation Guide](#installation-guide)
7. [Configuration Guide](#configuration-guide)
8. [Usage Examples](#usage-examples)
9. [SBTi SDA Compliance](#sbti-sda-compliance)
10. [IEA NZE Integration](#iea-nze-integration)
11. [Pathway Modeling](#pathway-modeling)
12. [Technology Roadmaps](#technology-roadmaps)
13. [Abatement Waterfall Analysis](#abatement-waterfall-analysis)
14. [Sector Benchmarking](#sector-benchmarking)
15. [Multi-Scenario Analysis](#multi-scenario-analysis)
16. [Security Model](#security-model)
17. [Performance Specifications](#performance-specifications)
18. [Troubleshooting](#troubleshooting)
19. [Frequently Asked Questions](#frequently-asked-questions)
20. [Related Documentation](#related-documentation)

---

## Executive Summary

### What is PACK-028?

PACK-028 is the **Sector Pathway Pack** -- the fourth pack in the GreenLang "Net Zero Packs" category. It provides deep, sector-specific decarbonization pathway analysis aligned with the SBTi Sectoral Decarbonization Approach (SDA) methodology and the IEA Net Zero by 2050 roadmap. The pack enables organizations to design science-based transition strategies tailored to the unique challenges of their specific sector.

Unlike generic absolute contraction approaches, PACK-028 models intensity convergence pathways for 15+ high-emitting sectors, mapping technology transitions, abatement levers, and investment requirements to sector-specific decarbonization trajectories.

### Why Sector-Specific Pathways Matter

Generic emission reduction targets fail to account for the fundamental differences in how sectors decarbonize:

- **Power generation** requires grid-scale transformation from fossil fuels to renewables, storage, and nuclear.
- **Steel production** depends on the transition from blast furnace to electric arc furnace and direct reduced iron with green hydrogen.
- **Cement manufacturing** faces irreducible process emissions from clinite calcination, requiring carbon capture.
- **Aviation** relies on sustainable aviation fuels and next-generation aircraft technology with decades-long fleet replacement cycles.
- **Buildings** need deep retrofit programs, heat pump deployment, and district heating/cooling integration.

The SBTi Sectoral Decarbonization Approach recognizes these differences by defining sector-specific intensity convergence pathways. PACK-028 implements the full SDA methodology for 12 SBTi-covered sectors and extends coverage to 15+ sectors using IEA NZE data.

### Key Capabilities

| Capability | Description |
|-----------|-------------|
| **15+ Sector Pathways** | Power, steel, cement, aluminum, chemicals, pulp/paper, aviation, shipping, road transport, rail, buildings (residential/commercial), agriculture, food/beverage, oil & gas, cross-sector |
| **SBTi SDA Compliance** | Automatic sector classification (NACE/GICS/ISIC), intensity metric calculation, convergence pathway generation, SBTi target validation |
| **IEA Scenario Integration** | 5 scenarios (NZE 1.5C, WB2C, 2C, APS, STEPS) with 400+ technology milestones |
| **Technology Transition Roadmaps** | Sector-specific technology adoption schedules with S-curve modeling, CapEx phasing, and dependency mapping |
| **Abatement Waterfall Analysis** | Lever-by-lever emission reduction contribution with cost curves, implementation timelines, and sequencing |
| **Sector Benchmarking** | Comparison against SBTi-validated peers, sector leaders, IEA pathway milestones, and regulatory benchmarks |
| **Convergence Analysis** | Gap-to-pathway quantification with required acceleration rate and investment delta |
| **Multi-Scenario Modeling** | Side-by-side pathway comparison across 5 climate scenarios with risk-return analysis |
| **20+ Intensity Metrics** | Sector-specific physical intensity metrics (gCO2/kWh, tCO2e/tonne, gCO2/pkm, kgCO2/m2/year, etc.) |
| **Zero-Hallucination** | All calculations use deterministic lookups from SBTi/IEA published data; no LLM in any calculation path |

### How PACK-028 Differs from Other Net Zero Packs

| Dimension | PACK-021 (Starter) | PACK-022 (Acceleration) | PACK-027 (Enterprise) | **PACK-028 (Sector Pathway)** |
|-----------|-------------------|------------------------|----------------------|------------------------------|
| **Focus** | Getting started | Accelerating reduction | Enterprise operations | Sector-specific pathways |
| **Sector coverage** | Generic (ACA only) | 12 SDA sectors | Multi-sector enterprise | **15+ sectors (SDA + IEA)** |
| **Intensity metrics** | tCO2e/revenue | Sector-specific basics | Revenue-based | **20+ physical intensity metrics** |
| **Pathway scenarios** | Single (1.5C) | 3 scenarios | 3 scenarios + Monte Carlo | **5 scenarios (NZE, WB2C, 2C, APS, STEPS)** |
| **Technology roadmaps** | Generic actions | Technology categories | Technology aware | **400+ IEA milestone mapping** |
| **Abatement analysis** | Basic MACC | MACC by action | MACC + carbon pricing | **Sector-specific waterfall by lever** |
| **Benchmarking** | Basic peers | SBTi peers | Multi-dimensional | **IEA pathway + peer + leader + regulatory** |
| **Gap analysis** | Target vs. current | Target vs. trajectory | Financial gap | **Sector convergence gap analysis** |
| **Data sources** | GHG Protocol | GHG Protocol + SBTi | GHG Protocol + SBTi + ERP | **GHG Protocol + SBTi + IEA NZE + IPCC AR6** |
| **Convergence model** | Linear only | Linear + exponential | Monte Carlo | **Linear + exponential + S-curve + stepped** |

### Target Users

| Persona | Role | Key PACK-028 Value |
|---------|------|-------------------|
| Sustainability Director | Climate strategy lead in carbon-intensive sectors | Sector-specific pathway design, SBTi SDA validation, technology roadmaps |
| Climate Strategy Manager | Transition planning specialist | Multi-scenario analysis, abatement waterfall, gap analysis vs. sector pathway |
| Board Member | Climate governance | Sector benchmark comparison, pathway convergence dashboards, investment requirements |
| SBTi Submission Lead | Target setting specialist | SDA pathway generation, intensity convergence calculation, SBTi validation report |
| Technology Planning Lead | Capital investment planning | Technology transition roadmaps, CapEx phasing, IEA milestone tracking |
| ESG Analyst | Sector benchmarking | Peer comparison, sector leader analysis, IEA pathway alignment scoring |
| Financial Institution | Portfolio alignment | Sector pathway alignment scoring, temperature rating by sector, transition risk |
| External Auditor | Pathway verification | Deterministic calculations, SHA-256 provenance, audit trail |

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Sector pathway accuracy | 100% match with SBTi SDA tool | Cross-validation against SBTi sector calculators |
| IEA scenario alignment | +/-5% from IEA NZE milestones | Validated against IEA NZE 2050 data tables |
| Sector coverage | 15+ sectors | Number of sectors with full pathway support |
| Intensity metric coverage | 20+ metrics | Number of sector-intensity metric pairs |
| Technology milestone coverage | 400+ IEA milestones | IEA milestones mapped to sector pathways |
| Convergence accuracy | +/-2% from manual calculation | 100+ sector pathway test scenarios |
| Pathway generation speed | Less than 5 minutes per sector | Time from input to validated pathway output |
| Test pass rate | 100% | All tests passing |

---

## Quick Start Guide

### Prerequisites

- GreenLang platform v1.0+ deployed
- PostgreSQL 16 with TimescaleDB extension
- Redis 7+ cache cluster
- Python 3.11+ runtime
- Platform migrations V001-V128 applied
- PACK-021 (recommended) or standalone configuration
- PACK-028 migrations V181-V186 applied

### Step 1: Install the Pack

```bash
# From the GreenLang root directory
cd packs/net-zero/PACK-028-sector-pathway

# Install dependencies
pip install -r requirements.txt

# Verify pack structure
python -c "from config.pack_config import PackConfig; print('PACK-028 loaded successfully')"
```

### Step 2: Run the Health Check

```python
from integrations.health_check import HealthCheck

hc = HealthCheck()
result = hc.run()
print(f"Health Score: {result.overall_score}/100")
print(f"Status: {result.status}")

# Verify all 20 categories pass
for category in result.categories:
    print(f"  [{category.status}] {category.name}: {category.score}/100")

# Expected output:
# Health Score: 100/100
# Status: HEALTHY
```

### Step 3: Classify Your Sector

```python
from engines.sector_classification_engine import SectorClassificationEngine

engine = SectorClassificationEngine()

result = engine.classify(
    company_profile={
        "name": "SteelCorp International",
        "nace_codes": ["C24.10"],          # Manufacture of basic iron and steel
        "gics_code": "15104020",            # Steel
        "isic_code": "2410",               # Manufacture of basic iron and steel
        "revenue_breakdown": {
            "integrated_steel": 0.75,       # 75% BF-BOF integrated steel
            "eaf_steel": 0.20,              # 20% EAF steel
            "other": 0.05,                  # 5% downstream processing
        },
        "primary_products": ["hot_rolled_coil", "cold_rolled_coil", "rebar"],
    }
)

print(f"Primary Sector: {result.primary_sector}")
# Output: Primary Sector: Steel

print(f"SDA Eligible: {result.sda_eligible}")
# Output: SDA Eligible: True

print(f"SDA Methodology: {result.sda_methodology}")
# Output: SDA Methodology: SDA-Steel

print(f"Intensity Metric: {result.intensity_metric}")
# Output: Intensity Metric: tCO2e/tonne crude steel

print(f"IEA Chapter: {result.iea_chapter}")
# Output: IEA Chapter: Chapter 5: Industry (Steel)
```

### Step 4: Generate a Sector Pathway

```python
from engines.pathway_generator_engine import PathwayGeneratorEngine

engine = PathwayGeneratorEngine()

result = engine.generate(
    sector="steel",
    base_year=2023,
    base_year_intensity=1.85,              # tCO2e/tonne crude steel (world average ~1.85)
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    production_forecast={
        2023: 50_000_000,                  # 50M tonnes
        2030: 55_000_000,
        2040: 58_000_000,
        2050: 60_000_000,
    },
    region="global",
)

print(f"Pathway: {result.pathway_name}")
# Output: Pathway: SDA-Steel-NZE-1.5C

print(f"Base Year Intensity: {result.base_intensity:.2f} tCO2e/t")
print(f"2030 Target Intensity: {result.target_2030:.2f} tCO2e/t")
print(f"2050 Target Intensity: {result.target_2050:.2f} tCO2e/t")

# Year-by-year pathway
for year_data in result.annual_pathway:
    print(f"  {year_data.year}: {year_data.intensity:.3f} tCO2e/t "
          f"(absolute: {year_data.absolute_emissions:,.0f} tCO2e)")
```

### Step 5: Analyze Convergence

```python
from engines.convergence_analyzer_engine import ConvergenceAnalyzerEngine

engine = ConvergenceAnalyzerEngine()

result = engine.analyze(
    current_intensity=1.65,                # Current intensity
    current_year=2025,
    sector_pathway=pathway_result,
    company_trajectory={
        2023: 1.85,
        2024: 1.75,
        2025: 1.65,
    },
)

print(f"Gap to Pathway: {result.gap_to_pathway:.1%}")
print(f"Gap to 2030 Target: {result.gap_to_2030:.1%}")
print(f"Required Annual Reduction: {result.required_annual_reduction:.1%}")
print(f"Risk Level: {result.risk_level}")
print(f"Time to Convergence: {result.time_to_convergence_years:.1f} years")
```

### Step 6: Build Technology Roadmap

```python
from engines.technology_roadmap_engine import TechnologyRoadmapEngine

engine = TechnologyRoadmapEngine()

result = engine.build(
    sector="steel",
    pathway=pathway_result,
    current_technology_mix={
        "bf_bof": 0.75,                   # 75% blast furnace-basic oxygen furnace
        "eaf_scrap": 0.20,                # 20% electric arc furnace (scrap)
        "dri_natural_gas": 0.05,          # 5% DRI with natural gas
    },
    capex_budget_annual_usd=500_000_000,
    region="eu",
)

print(f"Technology Transitions Required:")
for transition in result.transitions:
    print(f"  {transition.from_tech} -> {transition.to_tech}")
    print(f"    Start: {transition.start_year}, Complete: {transition.end_year}")
    print(f"    CapEx: ${transition.capex_total:,.0f}")
    print(f"    Reduction: {transition.emission_reduction_tco2e:,.0f} tCO2e")
```

### Step 7: Generate Reports

```python
from templates.sector_pathway_report import SectorPathwayReport
from templates.technology_roadmap_report import TechnologyRoadmapReport

# Sector pathway report
pathway_report = SectorPathwayReport()
output = pathway_report.render(
    pathway=pathway_result,
    convergence=convergence_result,
    format="html",
)

# Technology roadmap report
tech_report = TechnologyRoadmapReport()
output = tech_report.render(
    roadmap=roadmap_result,
    format="html",
)
```

---

## Sector Coverage

### Primary Sectors (SBTi SDA Coverage)

PACK-028 provides full SBTi Sectoral Decarbonization Approach (SDA) pathway support for 12 sectors recognized in the SBTi Corporate Standard.

| # | Sector | SDA Methodology | Intensity Metric | IEA Chapter | Key Technologies |
|---|--------|----------------|------------------|-------------|-----------------|
| 1 | Power Generation | SDA-Power | gCO2/kWh | Ch. 3: Electricity | Solar PV, wind, nuclear, grid storage, CCS |
| 2 | Steel | SDA-Steel | tCO2e/tonne crude steel | Ch. 5: Industry (Steel) | EAF, green hydrogen DRI, CCS, scrap recycling |
| 3 | Cement | SDA-Cement | tCO2e/tonne cement | Ch. 5: Industry (Cement) | Clinker substitution, alt fuels, CCUS, novel cements |
| 4 | Aluminum | SDA-Aluminum | tCO2e/tonne aluminum | Ch. 5: Industry (Aluminum) | Inert anode, renewable smelting, recycling |
| 5 | Pulp & Paper | SDA-Pulp | tCO2e/tonne pulp | Ch. 5: Industry (Pulp) | Biomass, black liquor, energy efficiency |
| 6 | Chemicals | SDA-Chemicals | tCO2e/tonne product | Ch. 5: Industry (Chemicals) | Electrification, hydrogen, bio-feedstocks, CCS |
| 7 | Aviation | SDA-Aviation | gCO2/pkm | Ch. 4: Transport (Aviation) | SAF, fleet renewal, hydrogen aircraft, operational efficiency |
| 8 | Shipping | SDA-Shipping | gCO2/tkm | Ch. 4: Transport (Shipping) | Ammonia, methanol, wind-assist, slow steaming |
| 9 | Road Transport | SDA-Transport | gCO2/vkm | Ch. 4: Transport (Road) | BEV, FCEV, biofuels, autonomous driving |
| 10 | Rail | SDA-Rail | gCO2/pkm | Ch. 4: Transport (Rail) | Electrification, hydrogen trains, regenerative braking |
| 11 | Buildings (Residential) | SDA-Buildings | kgCO2/m2/year | Ch. 2: Buildings (Residential) | Heat pumps, insulation, solar, district heating |
| 12 | Buildings (Commercial) | SDA-Buildings | kgCO2/m2/year | Ch. 2: Buildings (Commercial) | HVAC upgrades, BMS, LED, on-site renewables |

### Extended Sectors (IEA NZE Coverage)

Beyond SBTi SDA sectors, PACK-028 provides IEA-based pathway support for 3 additional sectors.

| # | Sector | Intensity Metric | IEA Chapter | Key Technologies |
|---|--------|------------------|-------------|-----------------|
| 13 | Agriculture | tCO2e/tonne food | Ch. 6: Agriculture | Precision farming, methane capture, soil carbon |
| 14 | Food & Beverage | tCO2e/tonne product | Ch. 5: Industry (Food) | Energy efficiency, refrigeration, packaging |
| 15 | Oil & Gas (Upstream) | gCO2/MJ energy | Ch. 1: Energy Supply | Methane reduction, electrification, CCS, managed decline |
| 16 | Cross-Sector | Generic ACA fallback | Multiple chapters | Energy efficiency, renewable procurement, fleet electrification |

### Sector Classification Mapping

PACK-028 automatically classifies companies into sectors using three international classification systems.

#### NACE Rev.2 Mapping

| NACE Code | NACE Description | PACK-028 Sector |
|-----------|-----------------|----------------|
| D35.11 | Production of electricity | Power Generation |
| C24.10 | Manufacture of basic iron and steel | Steel |
| C23.51 | Manufacture of cement | Cement |
| C24.42 | Aluminium production | Aluminum |
| C17.11 | Manufacture of pulp | Pulp & Paper |
| C20.11-C20.60 | Manufacture of chemicals | Chemicals |
| H51.10 | Passenger air transport | Aviation |
| H50.10 | Sea and coastal freight water transport | Shipping |
| H49.10-H49.39 | Land transport | Road Transport |
| H49.10 | Passenger rail transport | Rail |
| F41-F43 | Construction of buildings | Buildings |
| A01.10-A01.64 | Crop and animal production | Agriculture |
| C10.10-C11.07 | Manufacture of food and beverages | Food & Beverage |
| B06.10-B06.20 | Extraction of crude petroleum and gas | Oil & Gas |

#### GICS Mapping

| GICS Code | GICS Sub-Industry | PACK-028 Sector |
|-----------|-------------------|----------------|
| 55101010 | Electric Utilities | Power Generation |
| 15104020 | Steel | Steel |
| 15102010 | Construction Materials | Cement |
| 15104010 | Aluminum | Aluminum |
| 15105020 | Paper Products | Pulp & Paper |
| 15101020 | Commodity Chemicals | Chemicals |
| 20302010 | Airlines | Aviation |
| 20305010 | Marine | Shipping |
| 20304020 | Trucking | Road Transport |
| 20304010 | Railroads | Rail |
| 60101010-60102040 | Real Estate | Buildings |
| 30202010 | Agricultural Products | Agriculture |
| 30201010-30201030 | Food Products | Food & Beverage |
| 10102020 | Oil & Gas E&P | Oil & Gas |

#### ISIC Rev.4 Mapping

| ISIC Code | ISIC Description | PACK-028 Sector |
|-----------|-----------------|----------------|
| 3510 | Electric power generation | Power Generation |
| 2410 | Manufacture of basic iron and steel | Steel |
| 2394 | Manufacture of cement | Cement |
| 2420 | Manufacture of basic precious metals | Aluminum |
| 1701 | Manufacture of pulp | Pulp & Paper |
| 2011-2029 | Manufacture of chemicals | Chemicals |
| 5110 | Passenger air transport | Aviation |
| 5012 | Sea freight water transport | Shipping |
| 4921-4922 | Road transport | Road Transport |
| 4911 | Passenger rail transport | Rail |
| 4100 | Construction of buildings | Buildings |
| 0111-0150 | Crop and animal production | Agriculture |
| 1010-1104 | Manufacture of food and beverages | Food & Beverage |
| 0610-0620 | Extraction of crude petroleum and gas | Oil & Gas |

### Sector Intensity Metrics

Each sector uses specific physical intensity metrics defined by the SBTi SDA methodology and IEA NZE reporting framework.

#### Power Generation Metrics

| Metric ID | Name | Unit | SDA/IEA Source |
|-----------|------|------|---------------|
| `PWR-01` | Grid average emission intensity | gCO2/kWh | SBTi SDA-Power |
| `PWR-02` | Generation source intensity | tCO2e/MWh by source | IEA NZE Ch.3 |
| `PWR-03` | Capacity-weighted intensity | gCO2/kWh (installed MW) | IEA NZE Ch.3 |
| `PWR-04` | Lifecycle emission intensity | gCO2e/kWh (full lifecycle) | IPCC AR6 |

#### Steel Metrics

| Metric ID | Name | Unit | SDA/IEA Source |
|-----------|------|------|---------------|
| `STL-01` | Crude steel intensity (BF-BOF) | tCO2e/tonne crude steel | SBTi SDA-Steel |
| `STL-02` | Crude steel intensity (EAF) | tCO2e/tonne crude steel | SBTi SDA-Steel |
| `STL-03` | DRI intensity | tCO2e/tonne DRI | IEA NZE Ch.5 |
| `STL-04` | Hot metal intensity | tCO2e/tonne hot metal | IEA NZE Ch.5 |

#### Cement Metrics

| Metric ID | Name | Unit | SDA/IEA Source |
|-----------|------|------|---------------|
| `CMT-01` | Clinker intensity | tCO2e/tonne clinker | SBTi SDA-Cement |
| `CMT-02` | Cement intensity | tCO2e/tonne cement | SBTi SDA-Cement |
| `CMT-03` | Concrete intensity | tCO2e/m3 concrete | IEA NZE Ch.5 |
| `CMT-04` | Clinker-to-cement ratio | dimensionless ratio | SBTi SDA-Cement |

#### Aluminum Metrics

| Metric ID | Name | Unit | SDA/IEA Source |
|-----------|------|------|---------------|
| `ALU-01` | Primary aluminum intensity | tCO2e/tonne aluminum | SBTi SDA-Aluminum |
| `ALU-02` | Secondary aluminum intensity | tCO2e/tonne aluminum | IEA NZE Ch.5 |
| `ALU-03` | Alumina production intensity | tCO2e/tonne alumina | IEA NZE Ch.5 |

#### Aviation Metrics

| Metric ID | Name | Unit | SDA/IEA Source |
|-----------|------|------|---------------|
| `AVN-01` | Passenger intensity | gCO2/pkm | SBTi SDA-Aviation |
| `AVN-02` | Freight intensity | gCO2/RTK | SBTi SDA-Aviation |
| `AVN-03` | Fuel efficiency | L fuel/100 pkm | IEA NZE Ch.4 |
| `AVN-04` | SAF blend ratio | % of total fuel | IEA NZE Ch.4 |

#### Shipping Metrics

| Metric ID | Name | Unit | SDA/IEA Source |
|-----------|------|------|---------------|
| `SHP-01` | Freight intensity | gCO2/tkm | SBTi SDA-Shipping |
| `SHP-02` | Energy efficiency | gCO2/DWT-nm | IMO DCS |
| `SHP-03` | Carbon intensity indicator | gCO2/cargo capacity-nm | IMO CII |

#### Buildings Metrics

| Metric ID | Name | Unit | SDA/IEA Source |
|-----------|------|------|---------------|
| `BLD-01` | Operational emission intensity (res) | kgCO2/m2/year | SBTi SDA-Buildings |
| `BLD-02` | Operational emission intensity (com) | kgCO2/m2/year | SBTi SDA-Buildings |
| `BLD-03` | Energy use intensity | kWh/m2/year | IEA NZE Ch.2 |
| `BLD-04` | Embodied carbon | kgCO2/m2 (lifecycle) | IPCC AR6 |

#### Agriculture Metrics

| Metric ID | Name | Unit | SDA/IEA Source |
|-----------|------|------|---------------|
| `AGR-01` | Food production intensity | tCO2e/tonne food | IEA NZE Ch.6 |
| `AGR-02` | Land use intensity | tCO2e/hectare | SBTi FLAG |
| `AGR-03` | Livestock intensity | kgCO2e/kg protein | IPCC AR6 |

---

## Architecture Overview

### System Architecture

```
+==============================================================================+
|                           PRESENTATION TIER                                    |
|  +--------------------+  +--------------------+  +--------------------+       |
|  | Sector Pathway     |  | Technology Roadmap |  | Scenario Compare  |       |
|  | Dashboard (HTML)   |  | Viewer (HTML)      |  | Dashboard (HTML)  |       |
|  +--------------------+  +--------------------+  +--------------------+       |
|  +--------------------+  +--------------------+  +--------------------+       |
|  | Benchmark Report   |  | Abatement Waterfall|  | API Endpoints     |       |
|  | (PDF/HTML)         |  | Chart (HTML)       |  | (REST JSON)       |       |
|  +--------------------+  +--------------------+  +--------------------+       |
+==============================================================================+
|                           APPLICATION TIER                                     |
|  +------------------------------------------------------------------------+  |
|  |                   Pack Orchestrator (10-Phase DAG Pipeline)              |  |
|  |  +-----------+ +-----------+ +-----------+ +-----------+ +-----------+ |  |
|  |  | Sector    | | Pathway   | | Tech      | | Progress  | | Multi-    | |  |
|  |  | Design    | | Validate  | | Planning  | | Monitor   | | Scenario  | |  |
|  |  | Workflow  | | Workflow  | | Workflow  | | Workflow  | | Analysis  | |  |
|  |  +-----------+ +-----------+ +-----------+ +-----------+ +-----------+ |  |
|  |                    +-------------------------------------------+        |  |
|  |                    | Full Sector Assessment Workflow            |        |  |
|  |                    +-------------------------------------------+        |  |
|  +------------------------------------------------------------------------+  |
|  +------------------------------------------------------------------------+  |
|  |                         8 Engines                                       |  |
|  |  +---------------+ +---------------+ +---------------+ +-------------+ |  |
|  |  | Sector        | | Intensity     | | Pathway       | | Convergence | |  |
|  |  | Classification| | Calculator    | | Generator     | | Analyzer    | |  |
|  |  +---------------+ +---------------+ +---------------+ +-------------+ |  |
|  |  +---------------+ +---------------+ +---------------+ +-------------+ |  |
|  |  | Technology    | | Abatement     | | Sector        | | Scenario    | |  |
|  |  | Roadmap       | | Waterfall     | | Benchmark     | | Comparison  | |  |
|  |  +---------------+ +---------------+ +---------------+ +-------------+ |  |
|  +------------------------------------------------------------------------+  |
+==============================================================================+
|                            DATA TIER                                          |
|  +--------------------+  +--------------------+  +--------------------+      |
|  | PostgreSQL 16      |  | Redis 7            |  | SBTi SDA Pathway  |      |
|  | TimescaleDB        |  | Cache + Sessions   |  | Reference Data    |      |
|  | Pack tables        |  |                    |  |                    |      |
|  +--------------------+  +--------------------+  +--------------------+      |
|  +--------------------+  +--------------------+  +--------------------+      |
|  | IEA NZE 2050       |  | IPCC AR6           |  | PACK-021          |      |
|  | Sector Pathways    |  | Emission Factors   |  | Baseline/Targets  |      |
|  +--------------------+  +--------------------+  +--------------------+      |
+==============================================================================+
```

### Data Flow

```
Company Profile + Activity Data
    |
    v
Sector Classification Engine
    |  (NACE/GICS/ISIC -> sector mapping)
    v
Intensity Calculator Engine
    |  (Activity data + emissions -> sector intensity)
    v
Pathway Generator Engine
    |  (SBTi SDA + IEA NZE -> convergence pathway)
    |
    +---> Convergence Analyzer Engine
    |         |  (Current trajectory vs. pathway -> gap analysis)
    |         v
    |     Gap Analysis Report
    |
    +---> Technology Roadmap Engine
    |         |  (Pathway + IEA milestones -> tech adoption schedule)
    |         v
    |     Technology Roadmap Report
    |
    +---> Abatement Waterfall Engine
    |         |  (Pathway + levers -> contribution analysis)
    |         v
    |     Abatement Waterfall Report
    |
    +---> Sector Benchmark Engine
    |         |  (Company vs. peers, leaders, pathway -> percentile)
    |         v
    |     Sector Benchmark Report
    |
    +---> Scenario Comparison Engine
              |  (5 scenarios -> side-by-side comparison + risk)
              v
          Scenario Comparison Report
```

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **Deterministic pathway calculations** | SBTi SDA factors and IEA pathway data are constants from published sources; no LLM or probabilistic model in calculation path |
| **SHA-256 provenance hashing** | Every pathway calculation output is cryptographically hashed for audit trail integrity |
| **Sector-specific convergence models** | Different sectors follow different convergence curves (linear for some, S-curve for technology-driven sectors) |
| **Reference data versioning** | SBTi and IEA update pathway data periodically; the pack versions all reference data with effective dates |
| **PACK-021 integration** | Baseline emissions from PACK-021 feed directly into sector intensity calculations, avoiding data duplication |
| **Regional pathway support** | IEA provides differentiated pathways for OECD, emerging markets, and global; the pack supports all three |
| **Modular engine design** | Each engine operates independently; users can run sector classification without pathway generation |
| **Cache-friendly calculation** | Pathway lookups and emission factor lookups are heavily cached in Redis for sub-second response |

### Component Interactions

```
                          +---------------------------+
                          |    PACK-021 (Baseline)    |
                          |    pack021_bridge.py      |
                          +---------------------------+
                                      |
                                      | Baseline emissions + targets
                                      v
+------------------+    +---------------------------+    +--------------------+
| SBTi SDA Data    |--->|  Sector Classification    |--->| Intensity          |
| sbti_sda_bridge  |    |  Engine                   |    | Calculator Engine  |
+------------------+    +---------------------------+    +--------------------+
                                                              |
                                                              | Sector intensity
                                                              v
+------------------+    +---------------------------+    +--------------------+
| IEA NZE Data     |--->|  Pathway Generator        |--->| Convergence        |
| iea_nze_bridge   |    |  Engine                   |    | Analyzer Engine    |
+------------------+    +---------------------------+    +--------------------+
                                |
                                | Pathway targets
                                v
+------------------+    +---------------------------+    +--------------------+
| IPCC AR6 Data    |--->|  Technology Roadmap        |--->| Abatement          |
| ipcc_ar6_bridge  |    |  Engine                   |    | Waterfall Engine   |
+------------------+    +---------------------------+    +--------------------+
                                                              |
                                                              v
                        +---------------------------+    +--------------------+
                        |  Sector Benchmark         |    | Scenario           |
                        |  Engine                   |    | Comparison Engine  |
                        +---------------------------+    +--------------------+
```

---

## Core Components

### Engines (8)

| # | Engine | File | Purpose |
|---|--------|------|---------|
| 1 | Sector Classification Engine | `sector_classification_engine.py` | Automatic sector classification using NACE, GICS, ISIC codes with SDA sector mapping |
| 2 | Intensity Calculator Engine | `intensity_calculator_engine.py` | Sector-specific intensity metric calculation (20+ metrics) with data normalization |
| 3 | Pathway Generator Engine | `pathway_generator_engine.py` | SBTi SDA + IEA NZE pathway generation for 15+ sectors with 5 scenario support |
| 4 | Convergence Analyzer Engine | `convergence_analyzer_engine.py` | Sector intensity convergence analysis vs. SBTi/IEA benchmarks with gap quantification |
| 5 | Technology Roadmap Engine | `technology_roadmap_engine.py` | Technology transition roadmaps with IEA milestone mapping (400+ milestones) |
| 6 | Abatement Waterfall Engine | `abatement_waterfall_engine.py` | Sector-specific abatement waterfall with lever-by-lever contribution analysis |
| 7 | Sector Benchmark Engine | `sector_benchmark_engine.py` | Multi-dimensional sector benchmarking (peer, leader, SBTi-validated, IEA pathway) |
| 8 | Scenario Comparison Engine | `scenario_comparison_engine.py` | Multi-scenario pathway comparison (NZE, WB2C, 2C, APS, STEPS) with risk analysis |

### Workflows (6)

| # | Workflow | File | Phases | Purpose |
|---|----------|------|--------|---------|
| 1 | Sector Pathway Design | `sector_pathway_design_workflow.py` | 5 | SectorClassify -> IntensityCalc -> PathwayGen -> GapAnalysis -> ValidationReport |
| 2 | Pathway Validation | `pathway_validation_workflow.py` | 4 | DataValidation -> PathwayValidation -> SBTiCheck -> ComplianceReport |
| 3 | Technology Planning | `technology_planning_workflow.py` | 5 | TechInventory -> RoadmapGen -> CapExMapping -> DependencyAnalysis -> ImplementationPlan |
| 4 | Progress Monitoring | `progress_monitoring_workflow.py` | 4 | IntensityUpdate -> ConvergenceCheck -> BenchmarkUpdate -> ProgressReport |
| 5 | Multi-Scenario Analysis | `multi_scenario_analysis_workflow.py` | 5 | ScenarioSetup -> PathwayModeling -> RiskAnalysis -> ScenarioCompare -> StrategyRecommend |
| 6 | Full Sector Assessment | `full_sector_assessment_workflow.py` | 7 | Classify -> Pathway -> Technology -> Abatement -> Benchmark -> Scenarios -> Strategy |

### Templates (8)

| # | Template | File | Formats | Purpose |
|---|----------|------|---------|---------|
| 1 | Sector Pathway Report | `sector_pathway_report.py` | MD, HTML, JSON, PDF | Sector pathway with SDA/IEA alignment |
| 2 | Intensity Convergence Report | `intensity_convergence_report.py` | MD, HTML, JSON, PDF | Intensity tracking and convergence analysis |
| 3 | Technology Roadmap Report | `technology_roadmap_report.py` | MD, HTML, JSON, PDF | Technology transition roadmap with IEA milestones |
| 4 | Abatement Waterfall Report | `abatement_waterfall_report.py` | MD, HTML, JSON, PDF | Sector abatement waterfall with lever contributions |
| 5 | Sector Benchmark Report | `sector_benchmark_report.py` | MD, HTML, JSON, PDF | Multi-dimensional sector benchmarking dashboard |
| 6 | Scenario Comparison Report | `scenario_comparison_report.py` | MD, HTML, JSON, PDF | Multi-scenario comparison and risk analysis |
| 7 | SBTi Validation Report | `sbti_validation_report.py` | MD, HTML, JSON, PDF | SBTi SDA pathway validation and compliance |
| 8 | Sector Strategy Report | `sector_strategy_report.py` | MD, HTML, JSON, PDF | Executive sector transition strategy document |

### Integrations (10)

| # | Integration | File | Purpose |
|---|-------------|------|---------|
| 1 | Pack Orchestrator | `pack_orchestrator.py` | 10-phase DAG pipeline with sector-specific conditional routing |
| 2 | SBTi SDA Bridge | `sbti_sda_bridge.py` | SBTi SDA sector pathway data and validation tools integration |
| 3 | IEA NZE Bridge | `iea_nze_bridge.py` | IEA Net Zero by 2050 sector pathway and milestone data |
| 4 | IPCC AR6 Bridge | `ipcc_ar6_bridge.py` | IPCC AR6 sector-specific emission factors and pathways |
| 5 | PACK-021 Bridge | `pack021_bridge.py` | PACK-021 baseline and target engines integration |
| 6 | MRV Bridge | `mrv_bridge.py` | All 30 MRV agents for sector-specific emissions calculation |
| 7 | Decarbonization Bridge | `decarb_bridge.py` | Decarbonization agents for sector-specific reduction actions |
| 8 | Data Bridge | `data_bridge.py` | 20 DATA agents for sector activity data intake |
| 9 | Health Check | `health_check.py` | 20-category system verification including sector data freshness |
| 10 | Setup Wizard | `setup_wizard.py` | 7-step guided sector pathway configuration wizard |

### Presets (6)

| # | Preset | File | Sectors | Key Characteristics |
|---|--------|------|---------|---------------------|
| 1 | Heavy Industry | `heavy_industry.yaml` | Steel, Cement, Aluminum, Chemicals | SDA mandatory, high process emissions, CCS/hydrogen |
| 2 | Power Utilities | `power_utilities.yaml` | Power Generation, District Heating | SDA mandatory, grid decarbonization, renewable expansion |
| 3 | Transport | `transport.yaml` | Aviation, Shipping, Road, Rail | SDA per mode, fuel switching (SAF, hydrogen, electric) |
| 4 | Buildings | `buildings.yaml` | Residential, Commercial | SDA buildings, energy efficiency, heat pumps |
| 5 | Light Industry | `light_industry.yaml` | Pulp & Paper, Food & Beverage | SDA available, efficiency focus, biomass/bioenergy |
| 6 | Agriculture | `agriculture.yaml` | Agriculture, Land Use | FLAG pathway, N2O/CH4 reduction, soil carbon |

---

## Installation Guide

### System Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| CPU | 2 vCPU | 4+ vCPU | Multi-scenario analysis benefits from more cores |
| RAM | 4 GB | 8 GB | Reference data caching + engine processing |
| Storage | 1 GB | 5 GB | SBTi/IEA/IPCC reference datasets |
| Database | PostgreSQL 16 + TimescaleDB | Same | 6 pack-specific tables |
| Cache | Redis 7+ | Same | Pathway data caching, intermediate results |
| Network | Outbound HTTPS | Same | Reference data updates |
| Python | 3.11+ | 3.12 | Pydantic v2 required |

### Installation Steps

#### 1. Verify Platform Prerequisites

```bash
# Verify Python version
python --version
# Expected: Python 3.11.x or higher

# Verify PostgreSQL connection
psql -h $DB_HOST -U $DB_USER -d greenlang -c "SELECT version();"

# Verify Redis connection
redis-cli -h $REDIS_HOST ping
# Expected: PONG

# Verify platform migrations
psql -h $DB_HOST -U $DB_USER -d greenlang -c \
  "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"
# Expected: V128 or higher
```

#### 2. Apply Pack Migrations

```bash
# Apply PACK-028 specific migrations (6 migrations)
for i in $(seq -w 1 6); do
  psql -h $DB_HOST -U $DB_USER -d greenlang -f \
    migrations/V181-PACK028-${i}.sql
done

# Verify migration application
psql -h $DB_HOST -U $DB_USER -d greenlang -c \
  "SELECT version, description FROM schema_migrations WHERE version LIKE 'V181%' ORDER BY version;"
```

#### 3. Configure Environment Variables

```bash
# Required environment variables
export SECTOR_PATHWAY_DB_HOST="localhost"
export SECTOR_PATHWAY_DB_PORT="5432"
export SECTOR_PATHWAY_DB_NAME="greenlang"
export SECTOR_PATHWAY_REDIS_HOST="localhost"
export SECTOR_PATHWAY_REDIS_PORT="6379"
export SECTOR_PATHWAY_LOG_LEVEL="INFO"
export SECTOR_PATHWAY_PROVENANCE="true"

# Optional: PACK-021 integration
export SECTOR_PATHWAY_PACK021_ENABLED="true"
export SECTOR_PATHWAY_PACK021_BASE_URL="http://localhost:8021"

# Optional: IEA data path
export SECTOR_PATHWAY_IEA_DATA_DIR="/data/iea_nze_2050"
export SECTOR_PATHWAY_SBTI_DATA_DIR="/data/sbti_sda"
export SECTOR_PATHWAY_IPCC_DATA_DIR="/data/ipcc_ar6"
```

#### 4. Run Health Check

```python
from integrations.health_check import HealthCheck

hc = HealthCheck()
result = hc.run()

# Verify all 20 categories pass
for category in result.categories:
    print(f"  [{category.status}] {category.name}: {category.score}/100")

assert result.overall_score >= 90, f"Health check score too low: {result.overall_score}"
```

#### 5. Load a Sector Preset

```python
from config.pack_config import PackConfig

# Load the heavy industry preset
config = PackConfig.from_preset("heavy_industry")

# Or load transport preset
config = PackConfig.from_preset("transport")

# Or configure manually
config = PackConfig(
    sector="steel",
    sda_methodology="SDA-Steel",
    scenarios=["nze_15c", "wb2c", "2c", "aps", "steps"],
    base_year=2023,
    target_year_near=2030,
    target_year_long=2050,
    region="global",
)
```

---

## Configuration Guide

### Configuration Hierarchy

Configuration is resolved in the following order (later overrides earlier):

1. **Base `pack.yaml` manifest** -- default values for all settings
2. **Sector preset YAML** -- sector-specific overrides (e.g., `heavy_industry.yaml`)
3. **Environment variables** -- `SECTOR_PATHWAY_*` prefix overrides
4. **Runtime overrides** -- explicit parameters passed at execution time

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECTOR_PATHWAY_DB_HOST` | PostgreSQL host | `localhost` |
| `SECTOR_PATHWAY_DB_PORT` | PostgreSQL port | `5432` |
| `SECTOR_PATHWAY_DB_NAME` | Database name | `greenlang` |
| `SECTOR_PATHWAY_REDIS_HOST` | Redis host | `localhost` |
| `SECTOR_PATHWAY_REDIS_PORT` | Redis port | `6379` |
| `SECTOR_PATHWAY_LOG_LEVEL` | Log level | `INFO` |
| `SECTOR_PATHWAY_PROVENANCE` | Enable SHA-256 provenance | `true` |
| `SECTOR_PATHWAY_PACK021_ENABLED` | Integrate with PACK-021 | `true` |
| `SECTOR_PATHWAY_PACK021_BASE_URL` | PACK-021 API URL | `http://localhost:8021` |
| `SECTOR_PATHWAY_IEA_DATA_DIR` | IEA NZE data directory | `/data/iea_nze_2050` |
| `SECTOR_PATHWAY_SBTI_DATA_DIR` | SBTi SDA data directory | `/data/sbti_sda` |
| `SECTOR_PATHWAY_IPCC_DATA_DIR` | IPCC AR6 data directory | `/data/ipcc_ar6` |
| `SECTOR_PATHWAY_CACHE_TTL` | Cache TTL (seconds) | `3600` |
| `SECTOR_PATHWAY_MAX_SCENARIOS` | Max concurrent scenarios | `5` |

### Preset Configuration Examples

#### Heavy Industry Preset

```yaml
# heavy_industry.yaml
sectors:
  - steel
  - cement
  - aluminum
  - chemicals
sda_mandatory: true
process_emissions_focus: true
convergence_model: s_curve
technology_focus:
  - green_hydrogen
  - ccs_ccus
  - electrification
  - energy_efficiency
  - circular_economy
iea_milestones:
  chapters: [5]
  technology_categories:
    - hydrogen_production
    - carbon_capture
    - industrial_electrification
    - material_efficiency
```

#### Transport Preset

```yaml
# transport.yaml
sectors:
  - aviation
  - shipping
  - road_transport
  - rail
sda_mandatory: true
convergence_model: exponential
technology_focus:
  - sustainable_aviation_fuel
  - fleet_electrification
  - hydrogen_fuel_cell
  - alternative_marine_fuels
  - autonomous_efficiency
iea_milestones:
  chapters: [4]
  technology_categories:
    - electric_vehicles
    - hydrogen_fuel_cells
    - biofuels
    - fleet_efficiency
```

---

## Usage Examples

### Example 1: Full Sector Pathway Assessment

```python
from workflows.full_sector_assessment_workflow import FullSectorAssessmentWorkflow

workflow = FullSectorAssessmentWorkflow(config=config)
result = workflow.execute(
    company_profile={
        "name": "EuroSteel AG",
        "nace_codes": ["C24.10"],
        "gics_code": "15104020",
        "base_year": 2023,
        "base_year_production_tonnes": 5_000_000,
        "base_year_emissions_tco2e": 9_250_000,
        "current_technology_mix": {
            "bf_bof": 0.80,
            "eaf_scrap": 0.15,
            "dri_natural_gas": 0.05,
        },
    },
    scenarios=["nze_15c", "wb2c", "2c"],
)

# Access comprehensive results
print(f"Sector: {result.sector_classification.primary_sector}")
print(f"Base Intensity: {result.intensity.base_year_intensity:.2f} tCO2e/t")
print(f"NZE 2030 Target: {result.pathway.target_2030:.2f} tCO2e/t")
print(f"Gap to Pathway: {result.convergence.gap_to_pathway:.1%}")
print(f"Key Technologies: {result.technology_roadmap.key_transitions}")
print(f"Top Abatement Lever: {result.abatement_waterfall.top_lever}")
print(f"Peer Percentile: {result.benchmark.percentile_vs_peers}")
```

### Example 2: Multi-Sector Portfolio Analysis

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine

classifier = SectorClassificationEngine()
pathway_gen = PathwayGeneratorEngine()

portfolio_sectors = [
    {"name": "PowerGen Division", "nace": "D35.11", "intensity": 0.45, "unit": "tCO2e/MWh"},
    {"name": "Steel Division", "nace": "C24.10", "intensity": 1.85, "unit": "tCO2e/t"},
    {"name": "Cement Division", "nace": "C23.51", "intensity": 0.62, "unit": "tCO2e/t"},
]

for division in portfolio_sectors:
    sector = classifier.classify({"nace_codes": [division["nace"]]})
    pathway = pathway_gen.generate(
        sector=sector.primary_sector,
        base_year=2023,
        base_year_intensity=division["intensity"],
        target_year_near=2030,
        target_year_long=2050,
        scenario="nze_15c",
    )
    print(f"\n{division['name']}:")
    print(f"  Sector: {sector.primary_sector}")
    print(f"  2030 Target: {pathway.target_2030:.3f} {division['unit']}")
    print(f"  2050 Target: {pathway.target_2050:.3f} {division['unit']}")
```

### Example 3: Technology Transition Roadmap

```python
from engines.technology_roadmap_engine import TechnologyRoadmapEngine

engine = TechnologyRoadmapEngine()

result = engine.build(
    sector="power",
    pathway=pathway_result,
    current_technology_mix={
        "coal": 0.40,
        "natural_gas": 0.25,
        "nuclear": 0.10,
        "hydro": 0.08,
        "solar_pv": 0.10,
        "wind_onshore": 0.05,
        "wind_offshore": 0.02,
    },
    installed_capacity_mw=50_000,
    capex_budget_annual_usd=2_000_000_000,
    region="eu",
)

# Technology adoption schedule
for year in range(2025, 2051, 5):
    mix = result.get_technology_mix(year)
    print(f"\n{year} Technology Mix:")
    for tech, share in sorted(mix.items(), key=lambda x: -x[1]):
        if share > 0.01:
            print(f"  {tech}: {share:.1%}")

# IEA milestone tracking
for milestone in result.iea_milestones:
    status = "ON TRACK" if milestone.on_track else "OFF TRACK"
    print(f"  [{status}] {milestone.year}: {milestone.description}")

# CapEx phasing
for year_capex in result.capex_schedule:
    print(f"  {year_capex.year}: ${year_capex.amount:,.0f} "
          f"({year_capex.technology}: {year_capex.description})")
```

### Example 4: Abatement Waterfall by Lever

```python
from engines.abatement_waterfall_engine import AbatementWaterfallEngine

engine = AbatementWaterfallEngine()

result = engine.analyze(
    sector="cement",
    pathway=pathway_result,
    current_emissions_tco2e=3_100_000,
    current_clinker_ratio=0.75,
    current_alternative_fuel_share=0.15,
    target_year=2030,
)

print("Abatement Waterfall (2023 -> 2030):")
print(f"  Starting emissions: {result.start_emissions:,.0f} tCO2e")

cumulative = 0
for lever in result.levers:
    cumulative += lever.reduction_tco2e
    print(f"  {lever.name}:")
    print(f"    Reduction: {lever.reduction_tco2e:,.0f} tCO2e ({lever.reduction_pct:.1%})")
    print(f"    Cost: EUR {lever.cost_per_tco2e:,.0f}/tCO2e")
    print(f"    Timeline: {lever.start_year}-{lever.end_year}")
    print(f"    Cumulative: {cumulative:,.0f} tCO2e")

print(f"  Ending emissions: {result.end_emissions:,.0f} tCO2e")
print(f"  Total reduction: {result.total_reduction:,.0f} tCO2e ({result.total_reduction_pct:.1%})")
```

### Example 5: Sector Benchmarking

```python
from engines.sector_benchmark_engine import SectorBenchmarkEngine

engine = SectorBenchmarkEngine()

result = engine.benchmark(
    sector="steel",
    company_intensity=1.65,
    company_year=2025,
    region="eu",
    production_tonnes=5_000_000,
)

print(f"Sector: Steel")
print(f"Company Intensity: {result.company_intensity:.2f} tCO2e/t")
print(f"Sector Average: {result.sector_average:.2f} tCO2e/t")
print(f"Sector Leader (P10): {result.sector_leader:.2f} tCO2e/t")
print(f"SBTi Peer Average: {result.sbti_peer_average:.2f} tCO2e/t")
print(f"IEA 2025 Benchmark: {result.iea_benchmark:.2f} tCO2e/t")
print(f"Company Percentile: {result.percentile}th")
print(f"Gap to Leader: {result.gap_to_leader:.1%}")
print(f"Gap to Pathway: {result.gap_to_pathway:.1%}")
```

### Example 6: Multi-Scenario Comparison

```python
from engines.scenario_comparison_engine import ScenarioComparisonEngine

engine = ScenarioComparisonEngine()

result = engine.compare(
    sector="power",
    base_year_intensity=0.45,
    scenarios=["nze_15c", "wb2c", "2c", "aps", "steps"],
    milestones=[2025, 2030, 2035, 2040, 2045, 2050],
)

print("Scenario Comparison (Power Sector):")
print(f"{'Year':<8}", end="")
for scenario in result.scenarios:
    print(f"{scenario.name:<12}", end="")
print()

for year in result.milestones:
    print(f"{year:<8}", end="")
    for scenario in result.scenarios:
        intensity = scenario.get_intensity(year)
        print(f"{intensity:<12.3f}", end="")
    print()

# Risk analysis
print(f"\nOptimal Pathway: {result.optimal_pathway}")
print(f"Highest Risk Scenario: {result.highest_risk_scenario}")
print(f"Investment Range: ${result.min_investment:,.0f} - ${result.max_investment:,.0f}")
```

---

## SBTi SDA Compliance

### Sectoral Decarbonization Approach Overview

The SBTi SDA allocates the global carbon budget to individual sectors based on their contribution to global emissions and their sector-specific decarbonization potential. Companies within each sector converge to a common intensity level by 2050.

### SDA Sector Classification Rules

PACK-028 applies the following rules for sector classification:

1. **Primary sector determination**: Based on the largest revenue-contributing activity that matches an SDA sector.
2. **Multi-sector companies**: Companies with significant activity in multiple SDA sectors may set separate SDA targets for each division.
3. **SDA eligibility**: If a company's primary sector matches one of the 12 SDA sectors, SDA pathway is mandatory per SBTi Corporate Standard.
4. **Fallback to ACA**: Sectors not covered by SDA use the Absolute Contraction Approach (4.2%/year for 1.5C).

### SDA Coverage Requirements

| Requirement | Value | PACK-028 Validation |
|-------------|-------|-------------------|
| Scope 1+2 boundary | 95% of total Scope 1+2 | Auto-checked against PACK-021 baseline |
| Scope 3 boundary | 67% of total Scope 3 | Auto-checked for relevant categories |
| Base year intensity | Verified against sector benchmarks | Range validation + outlier detection |
| Target year (near-term) | 5-10 years from submission | Date validation |
| Target year (long-term) | Net-zero by 2050 | Fixed endpoint validation |
| Annual reduction rate | Sector-specific convergence | Rate calculation + validation |

### SDA Convergence Calculation

For each sector, the SDA convergence pathway is calculated as:

```
Company_Target(t) = Company_Intensity(base) + (Sector_Pathway(t) - Sector_Pathway(base))
                    * (Company_Intensity(base) / Sector_Average(base))
```

Where:
- `Company_Intensity(base)` is the company's base year intensity
- `Sector_Pathway(t)` is the SBTi sector pathway intensity at year t
- `Sector_Pathway(base)` is the SBTi sector pathway intensity at base year
- `Sector_Average(base)` is the global sector average at base year

### SBTi Validation Criteria Automated in PACK-028

| Criterion | Description | Auto-Check |
|-----------|-------------|-----------|
| Sector match | Company classified in correct SDA sector | NACE/GICS/ISIC verification |
| Base year recency | Base year within most recent 2 years | Date validation |
| Intensity calculation | Correct intensity metric for sector | Metric-sector pair validation |
| Convergence accuracy | Target aligns with sector pathway | +/-10% tolerance check |
| Coverage 95% S1+2 | 95% of Scope 1+2 emissions covered | Boundary completeness check |
| Coverage 67% S3 | 67% of Scope 3 emissions covered | Category coverage check |
| Ambition level | 1.5C alignment for near-term | Rate validation (>=4.2%/yr ACA equivalent) |
| Timeframe | 5-10 year near-term window | Date range validation |
| Double-counting | No scope overlap | Cross-scope reconciliation |
| Recalculation policy | Defined and consistent | Policy document validation |

---

## IEA NZE Integration

### IEA Net Zero by 2050 Scenario Data

PACK-028 integrates directly with IEA NZE 2050 sector pathway data, providing year-by-year intensity trajectories and technology milestones for all 15+ sectors.

### Supported IEA Scenarios

| Scenario | Temperature Outcome | Probability | IEA Reference | Use Case |
|----------|-------------------|-------------|---------------|----------|
| NZE (Net Zero Emissions) | +1.5C | 50% | IEA NZE 2050 (2023 update) | Most ambitious, SBTi 1.5C aligned |
| WB2C (Well-Below 2C) | Less than 2C | 66% | IEA WB2C | Strong ambition, SBTi WB2C aligned |
| 2C | +2C | 50% | IEA 2DS | Moderate ambition |
| APS (Announced Pledges) | +1.7C | N/A | IEA APS | Based on government commitments |
| STEPS (Stated Policies) | +2.4C | N/A | IEA STEPS | Business-as-usual trajectory |

### IEA Technology Milestones

PACK-028 maps 400+ IEA technology milestones to sector pathways. Examples:

#### Power Sector Milestones

| Year | Milestone | Source |
|------|-----------|--------|
| 2025 | No new unabated coal plants approved | IEA NZE Ch.3 |
| 2030 | Renewable capacity reaches 11,000 GW globally | IEA NZE Ch.3 |
| 2030 | 60% of global power from renewables | IEA NZE Ch.3 |
| 2035 | All unabated coal plants in advanced economies retired | IEA NZE Ch.3 |
| 2040 | 80% of global power from clean sources | IEA NZE Ch.3 |
| 2050 | Power sector reaches net-zero emissions | IEA NZE Ch.3 |

#### Steel Sector Milestones

| Year | Milestone | Source |
|------|-----------|--------|
| 2025 | First commercial green hydrogen DRI plant | IEA NZE Ch.5 |
| 2030 | 10% of steel production via green hydrogen DRI | IEA NZE Ch.5 |
| 2030 | EAF share reaches 40% globally | IEA NZE Ch.5 |
| 2040 | 30% of steel production near-zero emission | IEA NZE Ch.5 |
| 2050 | Steel sector reaches near-zero emissions intensity | IEA NZE Ch.5 |

#### Aviation Sector Milestones

| Year | Milestone | Source |
|------|-----------|--------|
| 2025 | SAF production reaches 10 billion litres | IEA NZE Ch.4 |
| 2030 | SAF represents 10% of aviation fuel | IEA NZE Ch.4 |
| 2030 | New aircraft 30% more fuel efficient than 2019 | IEA NZE Ch.4 |
| 2035 | Hydrogen aircraft enter service for short-haul | IEA NZE Ch.4 |
| 2050 | SAF represents 70% of aviation fuel | IEA NZE Ch.4 |

### Regional Pathway Variants

IEA NZE provides differentiated pathways for three regions:

| Region | Characteristics | Pathway Adjustment |
|--------|---------------|-------------------|
| OECD (Advanced Economies) | Earlier milestones, faster phase-out | -5 to -10 years vs. global |
| Emerging Markets (China, India, etc.) | Later milestones, support-dependent | +5 to +10 years vs. global |
| Global Average | Weighted average across all regions | Baseline pathway |

---

## Pathway Modeling

### Convergence Models

PACK-028 supports four convergence models for sector pathway generation:

#### 1. Linear Convergence

```
Intensity(t) = Intensity(base) - (t - base_year) * reduction_rate
```

Best for: Sectors with steady, policy-driven reductions (buildings, some transport).

#### 2. Exponential Convergence

```
Intensity(t) = Intensity(base) * exp(-k * (t - base_year))
```

Best for: Sectors with accelerating reduction potential (power, road transport).

#### 3. S-Curve Convergence

```
Intensity(t) = Intensity(2050) + (Intensity(base) - Intensity(2050))
               / (1 + exp(k * (t - t_inflection)))
```

Best for: Sectors dependent on technology adoption cycles (steel, cement, aviation).

#### 4. Stepped Convergence

```
Intensity(t) = Intensity(step_n)  for t in [step_n_start, step_n_end]
```

Best for: Sectors with discrete policy milestones (shipping IMO regulations, EU ETS phases).

### Pathway Time Horizons

| Horizon | Period | Granularity | Purpose |
|---------|--------|-------------|---------|
| Near-term | Base year to 2030 | Annual | SBTi near-term target setting |
| Medium-term | 2030 to 2040 | Annual | Technology transition planning |
| Long-term | 2040 to 2050 | Annual | Net-zero target achievement |
| Extended | 2050 to 2070 | 5-year | Residual emissions management |

### Production Forecast Integration

Sector pathways are expressed as intensity metrics. To derive absolute emissions, PACK-028 combines intensity pathways with production forecasts:

```
Absolute_Emissions(t) = Intensity(t) * Production(t)
```

Where `Production(t)` accounts for:
- Company growth projections
- Market demand forecasts (IEA sector demand data)
- Circular economy effects (recycling rates, material efficiency)
- Demand reduction (energy efficiency, modal shift)

---

## Technology Roadmaps

### Technology Transition Framework

Each sector's technology roadmap follows a structured framework:

1. **Current state assessment**: Inventory of existing technology base and performance
2. **Target state definition**: Technology mix required to achieve pathway targets
3. **Gap analysis**: Technologies needing adoption, scale-up, or phase-out
4. **S-curve modeling**: Technology adoption curves with market penetration forecasting
5. **CapEx phasing**: Multi-year investment schedule by technology
6. **Dependency mapping**: Technology prerequisites and interdependencies
7. **Risk assessment**: Technology maturity, supply chain, cost uncertainty

### Technology Readiness Levels

| TRL | Description | Example |
|-----|-------------|---------|
| 1-3 | Research / Proof of Concept | Direct air capture at scale |
| 4-6 | Pilot / Demonstration | Green hydrogen DRI steel |
| 7-8 | First-of-a-kind commercial | Inert anode aluminum smelting |
| 9 | Full commercial deployment | Solar PV, onshore wind, EAF steel |

### Sector Technology Libraries

Each sector includes a curated technology library with:
- Technology name and description
- Current TRL level
- Cost decline curve (learning rate)
- Maximum abatement potential (tCO2e/year per unit)
- Deployment lead time (years)
- Dependencies (e.g., green hydrogen requires renewable electricity)
- Regional availability (OECD vs. emerging markets)

See individual sector guides in `docs/SECTOR_GUIDES/` for detailed technology libraries per sector.

---

## Abatement Waterfall Analysis

### Waterfall Methodology

The abatement waterfall decomposes the total emission reduction required to meet the sector pathway into individual levers. Each lever represents a specific technology, practice, or policy change.

### Waterfall Calculation Steps

1. **Start**: Current year emissions (tCO2e)
2. **Lever 1**: First abatement lever applied (reduction in tCO2e)
3. **Lever 2**: Second abatement lever applied (incremental reduction)
4. ... (additional levers)
5. **End**: Target year emissions (tCO2e)
6. **Residual**: Remaining emissions requiring offsets or CDR

### Lever Attributes

Each abatement lever includes:
- **Name**: Descriptive lever name (e.g., "Clinker substitution")
- **Reduction**: Absolute reduction (tCO2e) and percentage of total
- **Cost**: Marginal abatement cost (EUR/tCO2e)
- **Timeline**: Implementation start and end years
- **Dependencies**: Other levers that must be implemented first
- **Certainty**: Confidence level (high/medium/low)
- **Technology**: Underlying technology or practice

### Lever Interdependencies

Some levers depend on or enable others:
- EAF steel transition requires renewable electricity availability
- Green hydrogen DRI requires green hydrogen production at scale
- CCS for cement requires CO2 transport and storage infrastructure
- SAF for aviation requires sustainable feedstock supply chains

PACK-028 models these dependencies using a directed acyclic graph (DAG) to ensure realistic sequencing.

---

## Sector Benchmarking

### Benchmark Dimensions

PACK-028 provides multi-dimensional benchmarking across five reference points:

| Dimension | Description | Data Source |
|-----------|-------------|-------------|
| Sector Average | Average intensity of all companies in sector | Industry databases, CDP data |
| Sector Leader (P10) | Top decile intensity performers | Industry databases, CDP data |
| SBTi-Validated Peers | Average of companies with validated SBTi targets | SBTi database |
| IEA Pathway Benchmark | IEA NZE 2050 sector milestone for current year | IEA NZE 2050 data tables |
| Regulatory Benchmark | EU ETS benchmark, EPA standards | Regulatory databases |

### Benchmark Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| Intensity percentile | Company rank vs. sector peers | Percentile (0-100) |
| Gap to average | Difference from sector average | % above/below |
| Gap to leader | Difference from top decile | % above/below |
| Gap to pathway | Difference from IEA pathway | % above/below |
| Time to convergence | Years until company meets pathway | Years |
| Required acceleration | Additional annual reduction needed | %/year |
| Investment delta | CapEx difference to close gap | USD |

---

## Multi-Scenario Analysis

### Scenario Comparison Framework

PACK-028 generates side-by-side pathway comparisons across 5 IEA scenarios.

### Comparison Outputs

| Output | Description |
|--------|-------------|
| Pathway matrix | Year-by-year intensity for each scenario |
| Investment matrix | Cumulative CapEx required per scenario |
| Technology matrix | Technology adoption timeline per scenario |
| Risk matrix | Transition risk level per scenario |
| Optimal pathway | Recommended pathway with justification |
| Sensitivity analysis | Key parameters driving scenario divergence |

### Risk-Return Analysis

Each scenario includes a risk-return profile:

| Scenario | Transition Risk | Physical Risk | Investment Required | Strategic Positioning |
|----------|----------------|--------------|-------------------|----------------------|
| NZE 1.5C | Low (proactive) | Lowest | Highest | First-mover advantage |
| WB2C | Low-Medium | Low | High | Strong alignment |
| 2C | Medium | Medium | Medium | Adequate alignment |
| APS | Medium-High | Medium-High | Medium-Low | Policy-dependent |
| STEPS | Highest | Highest | Lowest | Stranded asset risk |

---

## Security Model

### Role-Based Access Control

| Role | Description | Key Permissions |
|------|-------------|----------------|
| `sector_pathway_admin` | System administrator | Full configuration, reference data management |
| `pathway_designer` | Pathway design specialist | Create/edit pathways, run all engines, generate reports |
| `sector_analyst` | Read-only analyst | View pathways, benchmarks, scenarios (no edits) |
| `auditor` | External/internal auditor | View all data, provenance hashes, audit trail |

### Data Protection

| Control | Implementation |
|---------|---------------|
| Encryption at rest | AES-256-GCM for all pathway and calculation data |
| Encryption in transit | TLS 1.3 for all API communication |
| Provenance hashing | SHA-256 on all calculation outputs |
| Audit trail | Immutable append-only log |
| Reference data integrity | SHA-256 checksums on SBTi/IEA/IPCC data files |

---

## Performance Specifications

### Engine Latency Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Sector classification | Less than 2 seconds | Lookup-based |
| Intensity calculation | Less than 5 seconds | Single-sector |
| Pathway generation | Less than 30 seconds | Single sector, single scenario |
| Convergence analysis | Less than 10 seconds | Single sector |
| Technology roadmap | Less than 60 seconds | 400+ milestones |
| Abatement waterfall | Less than 30 seconds | Single sector |
| Sector benchmarking | Less than 15 seconds | Multi-dimension |
| Scenario comparison | Less than 120 seconds | 5 scenarios |
| Full sector assessment | Less than 5 minutes | All engines |
| API response (p95) | Less than 2 seconds | Standard queries |

### Cache Performance

| Metric | Target |
|--------|--------|
| SBTi pathway cache hit ratio | 95%+ |
| IEA milestone cache hit ratio | 90%+ |
| Emission factor cache hit ratio | 95%+ |
| Redis response time (p95) | Less than 5 ms |

---

## Troubleshooting

### Common Issues

#### Sector Classification Returns "Cross-Sector"

**Symptom:** Company classified as "Cross-Sector" instead of a specific SDA sector.

**Cause:** NACE/GICS/ISIC codes do not match any SDA sector, or revenue breakdown does not meet 50% threshold for any single sector.

**Resolution:**
```python
from engines.sector_classification_engine import SectorClassificationEngine

engine = SectorClassificationEngine()
result = engine.classify(
    company_profile={
        "nace_codes": ["C24.10"],
        "revenue_breakdown": {"integrated_steel": 0.60, "trading": 0.40},
    },
    debug=True,
)
print(result.classification_trace)
# Review trace to understand classification logic
```

#### Pathway Intensity Does Not Match SBTi Tool

**Symptom:** Generated pathway intensity differs from SBTi Target Setting Tool output.

**Cause:** Different base year, different sector average, or different convergence factors.

**Resolution:**
```python
# Verify base year and sector average used
print(f"Base year: {pathway_result.base_year}")
print(f"Base intensity: {pathway_result.base_intensity}")
print(f"Sector average (base year): {pathway_result.sector_average_base}")
print(f"SDA convergence factor: {pathway_result.convergence_factor}")
print(f"SDA reference version: {pathway_result.sbti_reference_version}")

# Ensure SBTi SDA data version matches
from integrations.sbti_sda_bridge import SBTiSDABridge
bridge = SBTiSDABridge()
print(f"SDA data version: {bridge.version}")
```

#### IEA Milestone Status Shows "Unknown"

**Symptom:** Technology milestones show status "UNKNOWN" instead of "ON TRACK" or "OFF TRACK".

**Cause:** IEA NZE data files not loaded or data directory misconfigured.

**Resolution:**
```bash
# Verify IEA data directory
ls $SECTOR_PATHWAY_IEA_DATA_DIR

# Verify data file integrity
python -c "
from integrations.iea_nze_bridge import IEANZEBridge
bridge = IEANZEBridge()
print(f'Milestones loaded: {bridge.milestone_count}')
print(f'Sectors covered: {bridge.sectors_covered}')
print(f'Data version: {bridge.data_version}')
"
```

#### Convergence Analysis Shows Negative Gap

**Symptom:** Gap-to-pathway shows a negative value (company already exceeds pathway target).

**Cause:** Company intensity is already below the sector pathway target for the current year.

**Resolution:** This is a valid result. A negative gap means the company is ahead of the sector pathway. The convergence analyzer will report "AHEAD OF PATHWAY" status.

#### Abatement Waterfall Total Does Not Match Pathway Reduction

**Symptom:** Sum of lever reductions does not equal the total pathway reduction.

**Cause:** Lever interactions (synergies or conflicts) create non-additive effects. The waterfall engine accounts for these via interaction factors.

**Resolution:**
```python
# View interaction effects
for interaction in result.lever_interactions:
    print(f"  {interaction.lever_a} x {interaction.lever_b}: "
          f"{interaction.effect_tco2e:+,.0f} tCO2e ({interaction.type})")
```

---

## Frequently Asked Questions

### General

**Q: Do I need PACK-021 to use PACK-028?**

A: PACK-021 (Net Zero Starter Pack) is recommended but not required. PACK-028 can operate standalone if you provide baseline emissions data directly. However, integrating with PACK-021 provides automated baseline feed and avoids data duplication.

**Q: Can PACK-028 handle multi-sector conglomerates?**

A: Yes. Multi-sector companies can set separate SDA targets for each division. The sector classification engine supports revenue-based sector allocation, and pathways are generated per division.

**Q: How often are SBTi SDA and IEA NZE reference data updated?**

A: SBTi updates SDA convergence factors approximately annually. IEA updates NZE scenario data with each World Energy Outlook edition. PACK-028 versions all reference data and supports side-by-side comparison of different data versions.

### SBTi

**Q: What if my sector is not covered by SDA?**

A: Sectors not covered by the 12 SDA methodologies default to the Absolute Contraction Approach (ACA) with 4.2%/year linear reduction for 1.5C alignment. PACK-028 handles this automatically via the "Cross-Sector" classification.

**Q: Can I use both SDA and ACA for different divisions?**

A: Yes. SBTi allows mixed approaches where SDA-eligible divisions use SDA and other divisions use ACA. PACK-028 supports this via per-division pathway configuration.

### Technology

**Q: How does PACK-028 handle emerging technologies with low TRL?**

A: Technologies below TRL 7 are included in long-term roadmaps (2035-2050) with higher uncertainty ranges. The technology roadmap engine applies conservative adoption curves for low-TRL technologies and flags them for review.

**Q: Can I add custom technologies to the roadmap?**

A: Yes. The technology roadmap engine supports custom technology entries with user-defined TRL, cost curves, and abatement potential.

### Data

**Q: Where does sector benchmark data come from?**

A: Sector averages and leader intensities are derived from CDP climate change questionnaire responses, IEA sector statistics, and SBTi validated targets database. Data is updated annually.

**Q: Can I import my own sector pathway data?**

A: Yes. PACK-028 supports custom pathway import via CSV or JSON format. Custom pathways can be used alongside SBTi SDA and IEA NZE pathways for comparison.

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API reference for all engines, workflows, templates, and integrations |
| [USER_GUIDE.md](docs/USER_GUIDE.md) | Detailed usage guide with step-by-step walkthroughs |
| [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) | Integration setup for SBTi, IEA, IPCC, PACK-021, MRV/DATA agents |
| [VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md) | Test results, accuracy validation, performance benchmarks |
| [DEPLOYMENT_CHECKLIST.md](docs/DEPLOYMENT_CHECKLIST.md) | Production deployment checklist and configuration guide |
| [SECTOR_GUIDES/](docs/SECTOR_GUIDES/) | 15+ sector-specific pathway guides |
| [CHANGELOG.md](docs/CHANGELOG.md) | Version history and release notes |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Development setup, coding standards, contribution guidelines |

### Sector-Specific Guides

| Guide | Sector |
|-------|--------|
| [SECTOR_GUIDE_POWER.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_POWER.md) | Power Generation |
| [SECTOR_GUIDE_STEEL.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_STEEL.md) | Steel |
| [SECTOR_GUIDE_CEMENT.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_CEMENT.md) | Cement |
| [SECTOR_GUIDE_ALUMINUM.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_ALUMINUM.md) | Aluminum |
| [SECTOR_GUIDE_CHEMICALS.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_CHEMICALS.md) | Chemicals |
| [SECTOR_GUIDE_PULP_PAPER.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_PULP_PAPER.md) | Pulp & Paper |
| [SECTOR_GUIDE_AVIATION.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_AVIATION.md) | Aviation |
| [SECTOR_GUIDE_SHIPPING.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_SHIPPING.md) | Shipping |
| [SECTOR_GUIDE_ROAD_TRANSPORT.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_ROAD_TRANSPORT.md) | Road Transport |
| [SECTOR_GUIDE_RAIL.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_RAIL.md) | Rail |
| [SECTOR_GUIDE_BUILDINGS.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_BUILDINGS.md) | Buildings |
| [SECTOR_GUIDE_AGRICULTURE.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_AGRICULTURE.md) | Agriculture |
| [SECTOR_GUIDE_FOOD_BEVERAGE.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_FOOD_BEVERAGE.md) | Food & Beverage |
| [SECTOR_GUIDE_OIL_GAS.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_OIL_GAS.md) | Oil & Gas |
| [SECTOR_GUIDE_CROSS_SECTOR.md](docs/SECTOR_GUIDES/SECTOR_GUIDE_CROSS_SECTOR.md) | Cross-Sector |

---

## License

Proprietary -- GreenLang Platform. All rights reserved.

## Support

- **Professional Support:** sector-pathway-support@greenlang.io
- **Documentation:** docs.greenlang.io/packs/net-zero/sector-pathway
- **Slack Channel:** #pack-028-sector-pathway
- **SLA:** 99.9% uptime, 4-hour response for P1 issues
