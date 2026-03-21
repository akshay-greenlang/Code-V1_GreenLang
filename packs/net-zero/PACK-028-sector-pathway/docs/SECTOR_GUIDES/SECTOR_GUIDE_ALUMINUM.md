# Sector Guide: Aluminum

**Sector ID:** `aluminum`
**SDA Methodology:** SDA-Aluminum
**Intensity Metric:** tCO2e/tonne aluminum
**IEA Chapter:** Chapter 5 -- Industry (Aluminum)

---

## Sector Overview

The aluminum sector is responsible for approximately 2% of global CO2 emissions, producing around 70 million tonnes of primary aluminum annually. Aluminum production is one of the most electricity-intensive industrial processes, with smelting (the Hall-Heroult electrolysis process) consuming approximately 14,000-16,000 kWh per tonne of primary aluminum. This makes the carbon intensity of aluminum production highly dependent on the electricity grid mix.

The sector has two distinct production pathways: primary production (from bauxite ore through alumina refining and electrolytic smelting) and secondary production (recycling of scrap aluminum). Secondary production requires only 5-8% of the energy needed for primary production, making it a critical decarbonization lever. However, demand growth means primary production remains essential.

Direct emissions from aluminum smelting include perfluorocarbon (PFC) emissions from anode effects and CO2 from the consumption of carbon anodes. The transition to inert anodes would eliminate these direct process emissions entirely, representing a transformational technology change.

The SBTi SDA-Aluminum methodology uses tCO2e per tonne of primary aluminum as the convergence metric, covering both direct (Scope 1) and indirect (Scope 2) emissions.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `ALU-01` | Primary aluminum intensity (overall) | tCO2e/tonne primary aluminum | Primary SDA metric. Total Scope 1+2 emissions per tonne of primary aluminum produced |
| `ALU-02` | Smelter intensity | tCO2e/tonne aluminum | Electrolysis-specific intensity including direct + indirect emissions |
| `ALU-03` | Alumina refining intensity | tCO2e/tonne alumina | Bayer process intensity for alumina refining |
| `ALU-04` | Anode-specific emissions | tCO2e/tonne aluminum | CO2 from carbon anode consumption in electrolysis |
| `ALU-05` | PFC emissions intensity | tCO2e/tonne aluminum | Perfluorocarbon emissions from anode effects |
| `ALU-06` | Secondary aluminum intensity | tCO2e/tonne aluminum | Recycled aluminum production intensity |

### Calculating ALU-01 (Primary Metric)

```python
intensity_tco2e_per_t = (scope1_tco2e + scope2_tco2e) / primary_aluminum_production_tonnes

# Where Scope 1 includes:
#   - Anode consumption CO2: ~1.5 tCO2e/t Al (Soderberg) or ~1.6 tCO2e/t Al (prebaked)
#   - PFC emissions: 0.01-0.5 tCO2e/t Al (varies by technology and anode effect frequency)
#   - Fuel combustion for alumina refining (if integrated)
#   - Fuel combustion for anode baking
#
# Where Scope 2 includes:
#   - Purchased electricity for smelting: ~14,500 kWh/t Al (industry average)
#   - Purchased electricity for alumina refining
#   - Purchased steam/heat
```

**Example:**
- Scope 1: 2,800,000 tCO2e (anode CO2 + PFC + combustion)
- Scope 2: 8,500,000 tCO2e (purchased electricity at ~550 gCO2/kWh grid)
- Primary aluminum production: 1,200,000 tonnes
- Intensity: (2,800,000 + 8,500,000) / 1,200,000 = 9.42 tCO2e/t

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway

| Year | Intensity (tCO2e/t) | Reduction from 2020 |
|------|---------------------|-------------------|
| 2020 | 10.0 (global avg) | Baseline |
| 2025 | 8.5 | -15% |
| 2030 | 6.5 | -35% |
| 2035 | 4.5 | -55% |
| 2040 | 2.8 | -72% |
| 2045 | 1.2 | -88% |
| 2050 | 0.5 | -95% |

### Regional Pathway Variants

| Year | Global | OECD | China | Middle East |
|------|--------|------|-------|-------------|
| 2020 | 10.0 | 6.5 | 14.0 | 8.5 |
| 2025 | 8.5 | 5.5 | 12.0 | 7.0 |
| 2030 | 6.5 | 3.8 | 9.0 | 5.5 |
| 2035 | 4.5 | 2.2 | 6.5 | 3.8 |
| 2040 | 2.8 | 1.0 | 4.0 | 2.5 |
| 2050 | 0.5 | 0.3 | 0.8 | 0.5 |

**Note:** Regional variation is extreme in aluminum due to electricity source differences. Iceland and Norway (hydro-powered smelters) achieve ~2.0 tCO2e/t, while China (coal-dominated grid) averages ~14.0 tCO2e/t. This 7x difference makes grid decarbonization the single most important lever.

---

## Technology Landscape

### Current Technology Mix (Global, 2023)

| Technology | Share | Intensity (tCO2e/t) | Status |
|-----------|-------|---------------------|--------|
| Prebaked anode (coal-powered grid) | 45% | 12.0-16.0 | Dominant in China |
| Prebaked anode (gas-powered grid) | 15% | 6.0-8.0 | Middle East, Russia |
| Prebaked anode (hydro-powered) | 12% | 1.5-3.0 | Norway, Iceland, Canada |
| Soderberg anode (various grids) | 3% | 10.0-18.0 | Phasing out |
| Secondary (recycled) aluminum | 25% | 0.3-0.8 | Growing significantly |

### Key Technology Transitions

#### 1. Grid Decarbonization (Renewable Electricity for Smelting)

- **Transition**: Shift smelting electricity from coal/gas to renewable sources (hydro, solar, wind)
- **Timeline**: Continuous to 2050, major shift 2025-2040
- **Reduction**: 40-80% of total intensity (grid-dependent)
- **Cost**: Varies; PPAs can be cost-neutral or slightly positive
- **Dependencies**: Renewable electricity availability at smelter locations, grid infrastructure
- **Critical importance**: Smelting accounts for ~60-70% of total aluminum emissions when grid is fossil-fuel based

#### 2. Inert Anode Technology

- **Transition**: Replace consumable carbon anodes with inert (non-consumable) anodes, eliminating direct CO2 from anode consumption
- **Timeline**: Pilot 2025-2030, commercial scale 2030-2040
- **Reduction**: Eliminates ~1.5 tCO2e/t (anode CO2) = 15-20% of total
- **By-product**: Produces oxygen instead of CO2
- **Cost**: EUR 200-400/tCO2e initially, declining with scale
- **Certainty**: Medium (Elysis JV between Rio Tinto and Alcoa targeting commercial scale by 2024-2026)
- **Dependencies**: Material science advances for inert anode durability, cell retrofit costs

#### 3. Increased Secondary (Recycled) Aluminum Production

- **Transition**: Increase scrap collection, sorting, and recycling rates
- **Current recycling rate**: ~34% globally (end-of-life recycling rate ~76%)
- **Target**: 40% of production from secondary by 2030, 50% by 2050
- **Reduction**: 90-95% per tonne shifted from primary to secondary
- **Cost**: Negative (secondary production is cheaper than primary)
- **Dependencies**: Scrap availability, alloy sorting technology, demand for recycled content

#### 4. Alumina Refining Decarbonization

- **Transition**: Replace fossil fuel (typically natural gas or coal) in Bayer process with:
  - Electric boilers powered by renewables
  - Mechanical vapor recompression (MVR)
  - Green hydrogen for calcination
- **Timeline**: Electric boilers available now; MVR scaling 2025-2030; hydrogen 2030+
- **Reduction**: 15-25% of total emissions (alumina refining = ~2.0-3.0 tCO2e/t alumina)
- **Cost**: EUR 30-80/tCO2e
- **Dependencies**: Renewable electricity availability at refinery sites

#### 5. PFC Emission Reduction

- **Transition**: Improve process control to reduce anode effect frequency and duration
- **Current**: Industry average ~0.3 tCO2e/t Al; best practice <0.05 tCO2e/t Al
- **Reduction**: 2-5% of total emissions
- **Cost**: Negative (process efficiency improvement)
- **Certainty**: High (proven technology and operational practices)

#### 6. Carbon Capture on Smelter and Refinery Emissions

- **Transition**: Post-combustion capture on alumina refinery flue gas; capture on smelter pot gas
- **Timeline**: Pilot 2028-2032, commercial 2032-2040
- **Reduction**: 30-60% of remaining direct emissions
- **Cost**: EUR 80-150/tCO2e
- **Certainty**: Low-Medium (aluminum flue gas has low CO2 concentration, making capture expensive)

---

## Abatement Levers

### Lever Waterfall (Typical Smelter with Coal-Based Grid, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Renewable electricity PPAs for smelting | 25-40% | -10 to +20 | High |
| 2 | Increased secondary aluminum share | 10-15% | -30 to -10 | High |
| 3 | PFC reduction (anode effect control) | 2-4% | -15 to -5 | High |
| 4 | Energy efficiency in smelting (retrofit) | 3-5% | -10 to 0 | High |
| 5 | Alumina refinery electrification | 5-8% | +20 to +50 | Medium |
| 6 | Inert anode pilot (partial deployment) | 3-5% | +80 to +200 | Medium |

### Lever Waterfall (Hydro-Powered Smelter, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Inert anode (eliminate anode CO2) | 40-50% | +80 to +200 | Medium |
| 2 | Alumina refinery electrification | 15-25% | +20 to +50 | Medium |
| 3 | Increased secondary aluminum share | 10-15% | -30 to -10 | High |
| 4 | PFC reduction | 5-10% | -15 to -5 | High |
| 5 | Energy efficiency improvements | 3-5% | -10 to 0 | High |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | All new smelters built with renewable electricity contracts |
| 2025 | PFC emissions reduced to below 0.1 tCO2e/t aluminum globally |
| 2030 | 50% of aluminum smelting powered by renewable electricity |
| 2030 | Inert anode technology commercially available |
| 2030 | Secondary aluminum share reaches 40% of total production |
| 2035 | All new smelters use inert anode technology |
| 2040 | 80% of smelting electricity from clean sources |
| 2040 | Alumina refining largely electrified or using green hydrogen |
| 2050 | Aluminum sector reaches near-zero emissions intensity |

---

## Benchmarks (2024)

| Benchmark | Value (tCO2e/t) | Source |
|-----------|-----------------|--------|
| Global average (primary) | 10.0 | International Aluminium Institute (IAI) |
| China average | 14.0 | IAI Regional Statistics |
| OECD average | 6.5 | IAI Regional Statistics |
| Hydro-powered smelters (best) | 1.8 | Company disclosures (Hydro, Alcoa Iceland) |
| Sector leader (P10) | 3.5 | CDP Climate 2024 |
| SBTi peer average | 7.5 | SBTi Database 2024 |
| Secondary aluminum (best) | 0.3 | IAI Recycling Statistics |
| IEA NZE 2025 target | 8.5 | IEA NZE 2050 |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.technology_roadmap_engine import TechnologyRoadmapEngine
from engines.abatement_waterfall_engine import AbatementWaterfallEngine

# Step 1: Classify sector
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["C24.42"]})
# Result: aluminum

# Step 2: Generate pathway
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="aluminum",
    base_year=2023,
    base_year_intensity=9.42,  # tCO2e/t
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="exponential",
    production_forecast={2023: 1_200_000, 2030: 1_400_000, 2050: 1_600_000},  # tonnes
    region="global",
)

print(f"2030 Target: {pathway.target_2030:.2f} tCO2e/t")
print(f"2050 Target: {pathway.target_2050:.2f} tCO2e/t")

# Step 3: Build technology roadmap
roadmap = TechnologyRoadmapEngine()
tech_result = roadmap.build(
    sector="aluminum",
    pathway=pathway,
    current_technology_mix={
        "prebaked_coal_grid": 0.55,
        "prebaked_gas_grid": 0.15,
        "prebaked_hydro": 0.10,
        "secondary_recycled": 0.20,
    },
    installed_capacity_tonnes=1_200_000,
    capex_budget_annual_usd=500_000_000,
    region="global",
)

# Step 4: Abatement waterfall
waterfall = AbatementWaterfallEngine()
abatement = waterfall.analyze(
    sector="aluminum",
    base_year=2023,
    base_year_intensity=9.42,
    target_year=2030,
    target_intensity=6.5,
    production_tonnes=1_200_000,
    available_levers=[
        "renewable_electricity_ppa",
        "secondary_aluminum_increase",
        "pfc_reduction",
        "energy_efficiency",
        "alumina_electrification",
    ],
)
```

---

## Regulatory Context

| Regulation | Relevance to Aluminum Sector |
|-----------|----------------------------|
| EU ETS Phase 4 | Covers aluminum smelting and alumina refining; free allocation declining |
| EU CBAM | Aluminum is a covered product; importers must purchase CBAM certificates |
| EU Green Deal | Industrial decarbonization targets affecting aluminum |
| China ETS | Potential expansion to cover aluminum smelting (currently not covered) |
| US IRA | Tax credits for clean energy applicable to smelter electricity sourcing |
| ASI Performance Standard | Aluminium Stewardship Initiative certification for responsible production |

---

## Special Considerations

### Grid Sensitivity

Aluminum is uniquely sensitive to electricity grid carbon intensity because smelting consumes ~14,500 kWh per tonne. The following table illustrates the impact:

| Grid Carbon Intensity | Scope 2 from Smelting | Total Intensity (approx.) |
|-----------------------|----------------------|--------------------------|
| 50 gCO2/kWh (hydro/nuclear) | 0.7 tCO2e/t | 2.2 tCO2e/t |
| 200 gCO2/kWh (gas-dominant) | 2.9 tCO2e/t | 4.4 tCO2e/t |
| 500 gCO2/kWh (mixed) | 7.3 tCO2e/t | 8.8 tCO2e/t |
| 800 gCO2/kWh (coal-dominant) | 11.6 tCO2e/t | 13.1 tCO2e/t |
| 1000 gCO2/kWh (unabated coal) | 14.5 tCO2e/t | 16.0 tCO2e/t |

This means that **relocating smelting to regions with clean electricity** is itself a major decarbonization strategy, though it raises questions about energy security, supply chain resilience, and economic development.

### CBAM Implications

Under the EU Carbon Border Adjustment Mechanism, aluminum imports into the EU will be subject to carbon costs based on their embedded emissions. This creates a financial incentive for non-EU producers to decarbonize, as high-carbon aluminum will face a cost disadvantage in the EU market.

---

## References

1. SBTi SDA-Aluminum Methodology, SBTi Target Setting Tool V3.0 (2025)
2. IEA Net Zero by 2050, Chapter 5: Industry (Aluminum)
3. International Aluminium Institute (IAI) Greenhouse Gas Emissions Data 2024
4. IPCC AR6 WGIII, Chapter 11: Industry
5. Aluminium Stewardship Initiative (ASI) Performance Standard V3
6. IEA Aluminium Technology Roadmap (2021)

---

**End of Aluminum Sector Guide**
