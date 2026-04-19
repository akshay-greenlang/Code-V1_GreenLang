# Sector Guide: Food & Beverage

**Sector ID:** `food_beverage`
**SDA Methodology:** Extended (IEA-based, SBTi FLAG guidance)
**Intensity Metric:** tCO2e/tonne product or tCO2e/million EUR revenue
**IEA Chapter:** Chapter 5 -- Industry (Food Processing) / Chapter 7 -- Agriculture (supply chain)

---

## Sector Overview

The food and beverage manufacturing sector encompasses the processing, packaging, and distribution of food products, from raw agricultural inputs to finished consumer goods. While the sector's direct manufacturing emissions (Scope 1+2) represent approximately 1-2% of global industrial CO2 emissions, the full value chain (Scope 3, including agricultural supply chain, packaging, distribution, retail, and consumer use) accounts for approximately 25-30% of global greenhouse gas emissions.

Key characteristics of the food and beverage sector:

1. **Scope 3 dominated**: 80-95% of total emissions are in Scope 3, primarily from agricultural raw materials (upstream) and consumer use/disposal (downstream)
2. **Energy-intensive processing**: Heating, cooling, drying, refrigeration, and sterilization are major energy consumers
3. **Refrigeration emissions**: HFC refrigerants contribute significant GHG emissions (high GWP)
4. **Water-intensive**: Water use is material and energy-intensive (heating, treatment)
5. **Waste and packaging**: Food waste and packaging contribute to lifecycle emissions
6. **SBTi FLAG**: Food companies with significant agricultural supply chains must set FLAG targets
7. **Consumer influence**: Product formulation and portfolio choices influence upstream emissions

The SBTi applies both the SDA approach (for Scope 1+2 intensity) and FLAG guidance (for land-related Scope 3) to food and beverage companies. PACK-028 models both dimensions.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `FB-01` | Manufacturing intensity | tCO2e/tonne product | Primary metric. Scope 1+2 per tonne of manufactured output |
| `FB-02` | Revenue-based intensity | tCO2e/million EUR revenue | Alternative for diversified portfolios |
| `FB-03` | Energy intensity | MJ/tonne product | Total energy per tonne of product |
| `FB-04` | Refrigerant emissions intensity | tCO2e/tonne product | HFC/refrigerant leakage emissions |
| `FB-05` | Supply chain intensity (Scope 3 Cat 1) | tCO2e/tonne raw material | Agricultural supply chain GHG intensity |
| `FB-06` | Packaging intensity | kgCO2e/unit product | GHG per unit of packaged product |
| `FB-07` | Food loss/waste rate | % of production | Percentage of production lost or wasted |

### Calculating FB-01 (Primary Metric)

```python
# Scope 1: On-site fuel combustion + process emissions + refrigerant leaks
scope1_tco2e = (
    natural_gas_combustion_tco2e +     # Boilers, ovens, dryers
    other_fuel_tco2e +                 # Oil, coal, biomass (fossil portion)
    refrigerant_leakage_tco2e +        # HFC/HFO emissions (GWP-weighted)
    process_co2_tco2e +                # Fermentation CO2, etc.
    transport_tco2e                     # Own fleet
)

# Scope 2: Purchased electricity + steam/heat/cooling
scope2_tco2e = (
    purchased_electricity_tco2e +
    purchased_steam_tco2e +
    purchased_cooling_tco2e
)

intensity_tco2e_per_t = (scope1_tco2e + scope2_tco2e) / total_product_tonnes
```

**Example (Dairy Products Manufacturer):**
- Scope 1: 80,000 tCO2e (natural gas for pasteurization, drying; refrigerant leaks)
- Scope 2: 50,000 tCO2e (purchased electricity for refrigeration, processing)
- Total product: 500,000 tonnes
- Manufacturing intensity: (80,000 + 50,000) / 500,000 = 0.26 tCO2e/t

---

## SBTi SDA / FLAG Pathway

### NZE 1.5C Convergence Pathway (Manufacturing -- Scope 1+2)

| Year | Intensity (tCO2e/t product) | Reduction from 2020 |
|------|---------------------------|-------------------|
| 2020 | 0.35 (global avg) | Baseline |
| 2025 | 0.29 | -17% |
| 2030 | 0.20 | -43% |
| 2035 | 0.12 | -66% |
| 2040 | 0.06 | -83% |
| 2045 | 0.02 | -94% |
| 2050 | 0.01 | -97% |

### FLAG Pathway (Agricultural Supply Chain -- Scope 3)

| Year | Reduction from 2020 | Key Actions |
|------|---------------------|-------------|
| 2025 | -10% | Sustainable sourcing, deforestation-free commitments |
| 2030 | -25% | Regenerative agriculture programs, supplier engagement |
| 2035 | -35% | Scaled regenerative practices, low-emission inputs |
| 2040 | -45% | Advanced agricultural technologies, carbon farming |
| 2050 | -55% | Residual agricultural emissions remain |

### Sub-Sector Variants (Manufacturing Intensity)

| Sub-Sector | 2020 Avg | 2030 Target | 2050 Target | Key Processes |
|-----------|----------|-------------|-------------|---------------|
| Dairy processing | 0.30 | 0.18 | 0.01 | Pasteurization, drying, refrigeration |
| Meat processing | 0.45 | 0.27 | 0.02 | Refrigeration, cooking, rendering |
| Beverages (brewing) | 0.08 | 0.05 | 0.003 | Brewing, packaging, refrigeration |
| Beverages (soft drinks) | 0.05 | 0.03 | 0.002 | Mixing, carbonation, packaging |
| Bakery and confectionery | 0.25 | 0.15 | 0.01 | Baking, packaging |
| Frozen foods | 0.50 | 0.30 | 0.02 | Freezing, cold storage, distribution |
| Sugar refining | 0.40 | 0.24 | 0.01 | Boiling, crystallization, drying |

---

## Technology Landscape

### Key Technology Transitions

#### 1. Thermal Process Electrification

- **Transition**: Replace natural gas boilers and steam systems with:
  - Industrial heat pumps (up to 150 deg C)
  - Electric boilers
  - Microwave and infrared heating
  - Ohmic heating
- **Reduction**: 30-60% of Scope 1 (thermal processes = largest direct emission source)
- **Timeline**: Heat pumps available now; high-temp applications 2025-2030
- **Cost**: EUR 20-60/tCO2e
- **Certainty**: High (heat pumps for <100 deg C); Medium (high-temperature processes)

#### 2. Refrigerant Transition (Low-GWP)

- **Transition**: Replace HFC refrigerants (GWP 1,000-4,000) with natural refrigerants (CO2/R744, ammonia/R717, hydrocarbons) or low-GWP HFOs
- **Reduction**: 80-99% of refrigerant-related emissions
- **Timeline**: Ongoing (EU F-Gas Regulation phase-down)
- **Cost**: EUR 10-40/tCO2e (system retrofit/replacement)
- **Certainty**: High (natural refrigerants proven; regulatory drivers strong)
- **Regulatory**: EU F-Gas Regulation requires HFC phase-down of 80% by 2030

#### 3. Renewable Electricity and Green Heat

- **Transition**: Procure 100% renewable electricity; source green heat (biomass, biogas, solar thermal)
- **Reduction**: Scope 2 to near-zero; partial Scope 1 reduction
- **Cost**: Neutral to +15 EUR/tCO2e
- **Timeline**: Immediate (procurement decisions)
- **Certainty**: Very High

#### 4. Energy Efficiency in Processing

- **Transition**: Waste heat recovery, process optimization, efficient motors, variable speed drives, insulation
- **Reduction**: 10-25% of total energy use
- **Cost**: Negative (energy savings)
- **Timeline**: Continuous improvement
- **Certainty**: Very High
- **Specific measures**:
  - Waste heat recovery from pasteurization: saves 15-25% of thermal energy
  - Variable speed drives on motors: saves 10-30% of electricity
  - Improved insulation on cold stores: saves 5-15% of refrigeration energy
  - Pinch analysis for heat integration: saves 10-20% of thermal energy

#### 5. Sustainable Packaging

- **Transition**: Reduce packaging weight, switch to recycled/recyclable materials, eliminate non-recyclable plastics
- **Reduction**: 10-30% of packaging-related Scope 3 emissions
- **Cost**: Variable (premium for some alternatives; savings from lightweighting)
- **Timeline**: Ongoing; regulatory drivers (EU Packaging Regulation)
- **Certainty**: Medium-High

#### 6. Supply Chain Decarbonization (FLAG)

- **Transition**: Engage agricultural suppliers on:
  - Regenerative farming practices
  - Reduced fertilizer use and methane inhibitors
  - Zero-deforestation sourcing
  - Sustainable livestock management
- **Reduction**: 15-30% of Scope 3 Category 1 emissions by 2030
- **Cost**: EUR 10-60/tCO2e (supplier support programs)
- **Certainty**: Medium (depends on supplier adoption and MRV)

#### 7. Fleet Electrification and Logistics Optimization

- **Transition**: Electric delivery vehicles, route optimization, modal shift (road to rail)
- **Reduction**: 5-15% of transport-related emissions (own fleet + upstream/downstream)
- **Cost**: Neutral to +EUR 30/tCO2e (EVs approaching TCO parity)
- **Timeline**: 2024-2035
- **Certainty**: High

---

## Abatement Levers

### Lever Waterfall (Food Manufacturer, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Renewable electricity procurement | 20-30% | 0 to +15 | Very High |
| 2 | Energy efficiency (waste heat, VSD, insulation) | 10-15% | -20 to -5 | Very High |
| 3 | Heat pump installation (low-temp processes) | 10-15% | +20 to +50 | High |
| 4 | Refrigerant transition (natural refrigerants) | 5-10% | +10 to +40 | High |
| 5 | Biogas/biomass for remaining thermal needs | 5-8% | +15 to +40 | High |
| 6 | Fleet electrification | 3-5% | 0 to +30 | High |
| 7 | Food waste reduction | 2-4% | -15 to 0 | High |

### Lever Waterfall (Beverage Company, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Renewable electricity (100% green PPA) | 30-40% | 0 to +15 | Very High |
| 2 | Refrigerant transition (CO2/NH3 systems) | 10-15% | +10 to +40 | High |
| 3 | Energy efficiency (brewing, cooling) | 8-12% | -20 to -5 | Very High |
| 4 | Thermal electrification (electric boilers) | 8-12% | +20 to +50 | High |
| 5 | Packaging lightweighting and recycled content | 5-8% | -5 to +20 | High |
| 6 | On-site solar PV | 3-5% | -10 to +5 | High |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | All major food companies committed to SBTi targets (including FLAG) |
| 2025 | HFC refrigerant phase-down reaches 50% in food cold chains (EU) |
| 2030 | Food manufacturing energy intensity reduced 25% from 2020 |
| 2030 | 50% of food companies sourcing 100% renewable electricity |
| 2030 | Zero-deforestation supply chains achieved for all major commodities |
| 2030 | Food loss and waste reduced 30% from 2020 (SDG 12.3) |
| 2035 | Heat pumps deployed for 50% of low-temperature process heating |
| 2040 | Food manufacturing largely electrified |
| 2050 | Food manufacturing reaches near-zero Scope 1+2 emissions |

---

## Benchmarks (2024)

| Benchmark | Value | Unit | Source |
|-----------|-------|------|--------|
| Global average (food manufacturing) | 0.35 | tCO2e/t product | IEA Industry |
| Dairy processing average | 0.30 | tCO2e/t product | IDF/IEA |
| Meat processing average | 0.45 | tCO2e/t product | GFLI |
| Beverages (brewing) average | 0.08 | tCO2e/t product | Beverage Industry Environmental Roundtable |
| Sector leader (P10) | 0.05 | tCO2e/t product | CDP Climate 2024 |
| SBTi peer average | 0.22 | tCO2e/t product | SBTi Database 2024 |
| Best practice (large diversified) | 0.08 | tCO2e/t product | Company disclosures |
| Revenue-based sector average | 45 | tCO2e/M EUR | CDP Climate 2024 |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.convergence_analyzer_engine import ConvergenceAnalyzerEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["C10.51"]})
# Result: food_beverage (dairy processing)

# Generate pathway
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="food_beverage",
    sub_sector="dairy_processing",
    base_year=2023,
    base_year_intensity=0.26,  # tCO2e/t product
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="linear",
    production_forecast={
        2023: 500_000,  # tonnes
        2030: 550_000,
        2050: 600_000,
    },
    region="eu",
)

print(f"2030 Target: {pathway.target_2030:.3f} tCO2e/t")
print(f"2050 Target: {pathway.target_2050:.3f} tCO2e/t")

# Convergence analysis
convergence = ConvergenceAnalyzerEngine()
analysis = convergence.analyze(
    sector="food_beverage",
    company_trajectory=[
        {"year": 2019, "intensity": 0.32},
        {"year": 2020, "intensity": 0.30},
        {"year": 2021, "intensity": 0.28},
        {"year": 2022, "intensity": 0.27},
        {"year": 2023, "intensity": 0.26},
    ],
    pathway=pathway,
)

print(f"On track: {analysis.on_track}")
print(f"Annual reduction rate: {analysis.annual_reduction_rate:.1%}")
```

---

## Special Considerations

### Scope 3 Dominance

For most food companies, Scope 3 emissions (especially Category 1: Purchased Goods and Services, representing agricultural raw materials) are 5-20x larger than Scope 1+2. Example emission profile:

| Scope | Share | Source |
|-------|-------|--------|
| Scope 1 | 5-10% | On-site fuel, refrigerants, fleet |
| Scope 2 | 5-10% | Purchased electricity, steam |
| Scope 3 Category 1 | 50-70% | Agricultural raw materials |
| Scope 3 Category 4 | 5-10% | Transportation |
| Scope 3 Categories 9-12 | 10-20% | Distribution, consumer use, end-of-life |

PACK-028 models Scope 1+2 through the SDA pathway and connects to SBTi FLAG guidance for agricultural supply chain targets via the PACK-021 Bridge and MRV Bridge.

### Cold Chain Emissions

The cold chain (refrigeration, frozen storage, cold transport) is a significant emission source for food companies. Key considerations:
- Refrigerant leakage can contribute 5-20% of total Scope 1
- Cold storage electricity can be 30-60% of total Scope 2
- Both are addressable through natural refrigerants and renewable electricity

### Food Waste and Loss

Reducing food waste addresses both direct emissions (waste treatment) and indirect emissions (avoided production of wasted food). The UN SDG 12.3 target calls for halving food waste by 2030.

---

## References

1. SBTi FLAG Guidance, Version 1.1 (2024)
2. IEA Net Zero by 2050, Chapter 5: Industry (Food Processing)
3. FAO Food Loss and Waste Database 2024
4. IPCC AR6 WGIII, Chapter 12: Cross-Sectoral Perspectives (Food Systems)
5. World Resources Institute: Creating a Sustainable Food Future (2019, updated 2024)
6. Beverage Industry Environmental Roundtable (BIER) Benchmarking 2024
7. Global Food Loss and Waste Standard (WRI/FUSIONS)

---

**End of Food & Beverage Sector Guide**
