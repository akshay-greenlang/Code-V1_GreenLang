# Sector Guide: Chemicals

**Sector ID:** `chemicals`
**SDA Methodology:** SDA-Chemicals
**Intensity Metric:** tCO2e/tonne chemical product
**IEA Chapter:** Chapter 5 -- Industry (Chemicals)

---

## Sector Overview

The chemicals sector is the third-largest industrial emitter of CO2, responsible for approximately 4-5% of global direct CO2 emissions. The sector produces a vast range of products from bulk petrochemicals (ethylene, propylene, ammonia, methanol) to specialty chemicals, polymers, fertilizers, and pharmaceuticals. Annual global chemical production exceeds 2.3 billion tonnes.

The chemicals sector presents unique decarbonization challenges due to:

1. **Dual use of fossil fuels**: Hydrocarbons serve as both energy source (fuel) and feedstock (raw material), meaning fuel switching alone cannot eliminate all emissions
2. **Process emissions**: Chemical reactions (e.g., steam methane reforming for hydrogen/ammonia) release CO2 as a by-product
3. **Product diversity**: Thousands of different products with different production routes and emission profiles
4. **Feedstock lock-in**: Many production processes are designed around fossil hydrocarbon feedstocks

The SBTi SDA-Chemicals methodology typically uses tCO2e per tonne of product for high-volume chemicals (ammonia, ethylene, methanol) and tCO2e per unit of revenue for diversified chemical companies.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `CHM-01` | Overall chemical production intensity | tCO2e/tonne product | Primary SDA metric (weighted average across product portfolio) |
| `CHM-02` | Ammonia production intensity | tCO2e/tonne NH3 | Ammonia-specific intensity (largest single chemical by emissions) |
| `CHM-03` | Ethylene production intensity | tCO2e/tonne ethylene | High-value chemical (HVC) intensity |
| `CHM-04` | Methanol production intensity | tCO2e/tonne methanol | Methanol production pathway intensity |
| `CHM-05` | Revenue-based intensity | tCO2e/million EUR revenue | Alternative metric for diversified portfolios |
| `CHM-06` | Feedstock energy intensity | GJ/tonne product | Energy input per unit of product (feedstock + process) |

### Calculating CHM-01 (Primary Metric)

```python
# For single-product chemical producers:
intensity_tco2e_per_t = (scope1_tco2e + scope2_tco2e) / product_output_tonnes

# For multi-product chemical companies (weighted average):
total_emissions = scope1_tco2e + scope2_tco2e
total_weighted_production = sum(
    product_tonnes[p] * weighting_factor[p] for p in products
)
intensity_tco2e_per_t_weighted = total_emissions / total_weighted_production

# Ammonia-specific (CHM-02):
# Natural gas SMR route: ~1.6-2.4 tCO2e/t NH3
# Coal gasification route: ~3.5-5.0 tCO2e/t NH3
# Green ammonia (electrolysis + Haber-Bosch): ~0.1-0.5 tCO2e/t NH3
```

**Example (Ammonia Producer):**
- Scope 1: 3,200,000 tCO2e (SMR process emissions + fuel combustion)
- Scope 2: 400,000 tCO2e (purchased electricity)
- Ammonia production: 1,800,000 tonnes
- Intensity: (3,200,000 + 400,000) / 1,800,000 = 2.00 tCO2e/t NH3

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway (Ammonia)

| Year | Intensity (tCO2e/t NH3) | Reduction from 2020 |
|------|------------------------|-------------------|
| 2020 | 2.40 (global avg) | Baseline |
| 2025 | 2.10 | -13% |
| 2030 | 1.60 | -33% |
| 2035 | 1.05 | -56% |
| 2040 | 0.55 | -77% |
| 2045 | 0.20 | -92% |
| 2050 | 0.05 | -98% |

### NZE 1.5C Convergence Pathway (Ethylene/HVC)

| Year | Intensity (tCO2e/t HVC) | Reduction from 2020 |
|------|------------------------|-------------------|
| 2020 | 1.50 (global avg) | Baseline |
| 2025 | 1.30 | -13% |
| 2030 | 1.00 | -33% |
| 2035 | 0.65 | -57% |
| 2040 | 0.35 | -77% |
| 2045 | 0.15 | -90% |
| 2050 | 0.05 | -97% |

### NZE 1.5C Convergence Pathway (Methanol)

| Year | Intensity (tCO2e/t MeOH) | Reduction from 2020 |
|------|--------------------------|-------------------|
| 2020 | 0.75 (global avg) | Baseline |
| 2025 | 0.65 | -13% |
| 2030 | 0.50 | -33% |
| 2035 | 0.30 | -60% |
| 2040 | 0.15 | -80% |
| 2045 | 0.05 | -93% |
| 2050 | 0.02 | -97% |

---

## Technology Landscape

### Key Chemical Products and Current Production Routes

| Product | Global Production (Mt/yr) | Dominant Route | Intensity (tCO2e/t) | Share of Sector Emissions |
|---------|--------------------------|----------------|---------------------|--------------------------|
| Ammonia (NH3) | 185 | Steam methane reforming (SMR) | 1.6-2.4 | ~30% |
| Ethylene (C2H4) | 200 | Naphtha/ethane steam cracking | 1.0-2.0 | ~20% |
| Propylene (C3H6) | 130 | Steam cracking + FCC | 1.0-1.5 | ~10% |
| Methanol (CH3OH) | 100 | Natural gas reforming | 0.5-1.0 | ~5% |
| Chlorine/NaOH | 75 | Chlor-alkali electrolysis | 0.5-1.5 | ~5% |
| Others | Various | Various | Various | ~30% |

### Key Technology Transitions

#### 1. Green Hydrogen for Ammonia Production

- **Transition**: Replace steam methane reforming with electrolysis-based hydrogen + Haber-Bosch
- **Timeline**: Pilot 2024-2028, commercial from 2028, scale-up to 2040
- **Reduction**: 85-95% vs. SMR ammonia
- **Cost**: Green ammonia currently 2-3x conventional; cost parity projected 2030-2035
- **Dependencies**: Green hydrogen cost (<$2/kg), renewable electricity, electrolyzer capacity
- **Scale impact**: Ammonia is 30% of sector emissions, so green ammonia alone addresses the largest single source

#### 2. Electrification of Steam Crackers

- **Transition**: Replace fossil fuel-fired furnaces with electric furnaces for ethylene/propylene cracking
- **Timeline**: Pilot (BASF/SABIC/Linde e-furnace) 2023-2026; commercial 2028+
- **Reduction**: 80-90% of furnace emissions (furnace = ~50% of cracker emissions)
- **Cost**: EUR 40-80/tCO2e (depends on electricity price)
- **Dependencies**: High-temperature electric heating technology, renewable electricity at scale
- **Note**: Does not address feedstock emissions (hydrocarbon feedstock remains)

#### 3. Bio-Based and Recycled Feedstocks

- **Transition**: Replace fossil feedstocks with biomass-derived or recycled plastic feedstocks
- **Timeline**: Growing from 2023; significant share by 2035+
- **Reduction**: 50-90% vs. fossil feedstock (lifecycle)
- **Cost**: Currently 2-5x conventional feedstock; declining with scale
- **Dependencies**: Sustainable biomass supply, chemical recycling infrastructure, quality standards

#### 4. Carbon Capture on Chemical Plants

- **Transition**: CCS on ammonia plants (high-purity CO2 stream), steam crackers, other high-emission units
- **Timeline**: Ammonia CCS commercial now (pure CO2 already separated); cracker CCS 2028-2035
- **Reduction**: 85-95% of captured point-source emissions
- **Cost**: EUR 20-40/tCO2e (ammonia, pure CO2); EUR 60-100/tCO2e (dilute streams)
- **Certainty**: High for ammonia (pure CO2 stream); Medium for other chemical processes
- **Note**: Ammonia plants already separate CO2 as part of the process. Currently most is vented; CCS requires only compression and transport/storage

#### 5. Process Intensification and Energy Efficiency

- **Transition**: Advanced catalysts, modular reactors, heat integration, waste heat recovery
- **Reduction**: 10-20% of total emissions
- **Cost**: Negative to neutral (energy savings offset investment)
- **Timeline**: Continuous improvement
- **Certainty**: High

#### 6. E-Methanol and E-Fuels

- **Transition**: Produce methanol from captured CO2 + green hydrogen (Power-to-Methanol)
- **Timeline**: Pilot 2025-2028, commercial 2030+
- **Reduction**: 80-95% vs. fossil methanol (if using DAC or biogenic CO2)
- **Cost**: Currently 3-5x conventional methanol; declining with green H2 costs
- **Dependencies**: Green hydrogen supply, CO2 source (DAC, biogenic, or industrial point source)

---

## Abatement Levers

### Lever Waterfall (Typical Petrochemical Complex, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Energy efficiency and heat integration | 8-12% | -20 to -5 | High |
| 2 | Renewable electricity procurement | 5-10% | 0 to +20 | High |
| 3 | CCS on ammonia plant (pure CO2 stream) | 10-15% | +20 to +40 | High |
| 4 | Green hydrogen for ammonia (partial) | 5-10% | +50 to +100 | Medium |
| 5 | Electric steam cracker (pilot) | 3-5% | +40 to +80 | Medium |
| 6 | Bio-based feedstock substitution (partial) | 2-4% | +30 to +60 | Medium |

### Lever Waterfall (Ammonia-Only Producer, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | CCS on existing SMR plant | 30-50% | +20 to +40 | High |
| 2 | Green hydrogen electrolyzer (new capacity) | 20-30% | +50 to +100 | Medium |
| 3 | Energy efficiency and process optimization | 5-10% | -20 to -5 | High |
| 4 | Renewable electricity for utilities | 5-8% | 0 to +20 | High |
| 5 | Waste heat recovery | 3-5% | -15 to -5 | High |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | First commercial-scale green ammonia plants operational |
| 2025 | Electric steam cracker pilot plants demonstrated |
| 2030 | 10% of ammonia production from low-carbon hydrogen |
| 2030 | CCS deployed on 20% of ammonia production capacity |
| 2030 | Bio-based feedstock share reaches 5% of chemical production |
| 2035 | Electric steam crackers commercially deployed at scale |
| 2035 | 30% of ammonia production from green/blue hydrogen |
| 2040 | 50% of chemical production from low-carbon processes |
| 2040 | Chemical recycling provides 15% of plastic feedstock |
| 2050 | Chemical sector achieves near-zero emissions intensity |

---

## Benchmarks (2024)

| Benchmark | Value | Unit | Source |
|-----------|-------|------|--------|
| Global average (ammonia) | 2.40 | tCO2e/t NH3 | IFA/IEA |
| Best practice SMR ammonia | 1.60 | tCO2e/t NH3 | IFA Benchmarks |
| Green ammonia | 0.10-0.50 | tCO2e/t NH3 | Company disclosures |
| Blue ammonia (SMR + CCS) | 0.30-0.60 | tCO2e/t NH3 | Company disclosures |
| Global average (ethylene) | 1.50 | tCO2e/t HVC | APPE/IEA |
| Best practice ethylene (naphtha) | 0.80 | tCO2e/t HVC | IEA Chemicals Roadmap |
| Best practice ethylene (ethane) | 0.50 | tCO2e/t HVC | IEA Chemicals Roadmap |
| Sector leader (P10, diversified) | 0.45 | tCO2e/t product | CDP Climate 2024 |
| SBTi peer average | 1.10 | tCO2e/t product | SBTi Database 2024 |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.scenario_comparison_engine import ScenarioComparisonEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["C20.15"]})
# Result: chemicals

# Generate pathway for ammonia producer
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="chemicals",
    sub_sector="ammonia",
    base_year=2023,
    base_year_intensity=2.00,  # tCO2e/t NH3
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="s_curve",
    production_forecast={2023: 1_800_000, 2030: 2_000_000, 2050: 2_200_000},  # tonnes NH3
    region="global",
)

print(f"2030 Target: {pathway.target_2030:.2f} tCO2e/t NH3")
print(f"2050 Target: {pathway.target_2050:.2f} tCO2e/t NH3")

# Multi-scenario comparison
scenario_engine = ScenarioComparisonEngine()
comparison = scenario_engine.compare(
    sector="chemicals",
    sub_sector="ammonia",
    base_year=2023,
    base_year_intensity=2.00,
    scenarios=["nze_15c", "wb2c", "2c", "aps", "steps"],
    production_forecast={2023: 1_800_000, 2030: 2_000_000, 2050: 2_200_000},
)

for scenario in comparison.scenarios:
    print(f"{scenario.name}: 2030={scenario.target_2030:.2f}, 2050={scenario.target_2050:.2f}")
```

---

## Sub-Sector Considerations

### Ammonia

Ammonia is the single largest contributor to chemical sector emissions (~30%) and the most straightforward to decarbonize because:
- SMR produces a pure CO2 stream that is cheap to capture
- Green hydrogen + Haber-Bosch is a proven alternative pathway
- Ammonia is also being developed as a hydrogen carrier and clean fuel

### Petrochemicals (Ethylene, Propylene)

Petrochemicals face the "dual use" challenge: fossil hydrocarbons are both fuel and feedstock. Decarbonization requires:
1. Electrifying the heat (steam cracker furnaces)
2. Replacing fossil feedstock with bio-based or recycled alternatives
3. Using CCS for remaining process emissions

### Chlor-Alkali

Already electricity-intensive; decarbonization primarily through renewable electricity procurement.

### Specialty and Fine Chemicals

Lower absolute emissions but often higher intensity per tonne. Revenue-based metrics (CHM-05) are more appropriate for diversified specialty chemical companies.

---

## References

1. SBTi SDA-Chemicals Methodology, SBTi Target Setting Tool V3.0 (2025)
2. IEA Net Zero by 2050, Chapter 5: Industry (Chemicals)
3. IEA The Future of Petrochemicals (2018)
4. IPCC AR6 WGIII, Chapter 11: Industry
5. International Fertilizer Association (IFA) Ammonia Production Benchmarks 2024
6. Cefic Chemical Industry Facts and Figures 2024
7. DECHEMA Roadmap Chemie 2050

---

**End of Chemicals Sector Guide**
