# Sector Guide: Pulp & Paper

**Sector ID:** `pulp_paper`
**SDA Methodology:** SDA-Pulp & Paper
**Intensity Metric:** tCO2e/tonne product
**IEA Chapter:** Chapter 5 -- Industry (Pulp & Paper)

---

## Sector Overview

The pulp and paper sector accounts for approximately 1-2% of global industrial CO2 emissions, producing around 410 million tonnes of paper and paperboard and 190 million tonnes of market pulp annually. While smaller in absolute emissions than steel or cement, the sector is significant because it is both an energy-intensive industry and a major user of biomass energy, giving it unique decarbonization characteristics.

Key features of the pulp and paper sector:

1. **High biogenic energy share**: The sector already derives ~55-60% of its energy from biomass (black liquor, bark, wood waste), making it one of the most bio-energy-intensive industries
2. **Combined heat and power (CHP)**: Most pulp mills operate CHP systems, generating both steam and electricity for mill operations
3. **Diverse product range**: Products range from commodity grades (containerboard, newsprint) to specialty grades (tissue, specialty papers, dissolving pulp)
4. **Recycling integration**: Paper recycling rates exceed 70% in many markets, with recycled fiber being a major feedstock
5. **Carbon sinks**: Forest management for fiber supply can function as a carbon sink if managed sustainably

The SBTi SDA-Pulp & Paper methodology uses tCO2e per tonne of product (pulp, paper, or paperboard) as the convergence metric, covering Scope 1 and Scope 2 emissions. Biogenic CO2 from sustainable biomass combustion is counted as carbon-neutral per GHG Protocol guidance.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `PP-01` | Overall production intensity | tCO2e/tonne product | Primary SDA metric. Total Scope 1+2 (fossil) per tonne of paper/pulp/board |
| `PP-02` | Pulp mill intensity | tCO2e/tonne pulp | Chemical (kraft) or mechanical pulp production |
| `PP-03` | Paper mill intensity | tCO2e/tonne paper | Paper machine and finishing operations |
| `PP-04` | Integrated mill intensity | tCO2e/tonne product | Combined pulp + paper operations at integrated mills |
| `PP-05` | Energy intensity | GJ/tonne product | Total energy consumption per tonne of product |
| `PP-06` | Biogenic energy share | % of total energy | Proportion of energy from biomass (sustainability indicator) |

### Calculating PP-01 (Primary Metric)

```python
# Only fossil CO2 counts toward SDA target (biogenic CO2 from sustainable biomass = 0)
fossil_scope1_tco2e = (
    fossil_fuel_combustion_tco2e +  # natural gas, coal, oil
    purchased_fossil_steam_tco2e +  # if applicable
    lime_kiln_fossil_emissions +     # fossil CO2 from lime kiln
    process_emissions                # non-energy process CO2
)
# Note: Biogenic CO2 from black liquor, bark, wood waste = 0 in GHG Protocol

scope2_tco2e = purchased_electricity_tco2e + purchased_steam_tco2e

intensity_tco2e_per_t = (fossil_scope1_tco2e + scope2_tco2e) / total_product_tonnes
```

**Example (Integrated Kraft Pulp & Paper Mill):**
- Fossil Scope 1: 180,000 tCO2e (natural gas for lime kiln + auxiliary boiler)
- Scope 2: 120,000 tCO2e (purchased electricity)
- Biogenic emissions: 1,500,000 tCO2e (black liquor + bark = NOT counted)
- Total product: 800,000 tonnes (paper + market pulp)
- Intensity: (180,000 + 120,000) / 800,000 = 0.375 tCO2e/t

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway

| Year | Intensity (tCO2e/t) | Reduction from 2020 |
|------|---------------------|-------------------|
| 2020 | 0.55 (global avg) | Baseline |
| 2025 | 0.47 | -15% |
| 2030 | 0.35 | -36% |
| 2035 | 0.22 | -60% |
| 2040 | 0.12 | -78% |
| 2045 | 0.05 | -91% |
| 2050 | 0.02 | -96% |

### Sub-Sector Pathway Variants

| Year | Integrated Kraft Mill | Recycled Paper Mill | Mechanical Pulp Mill |
|------|----------------------|--------------------|--------------------|
| 2020 | 0.40 | 0.65 | 0.80 |
| 2025 | 0.33 | 0.55 | 0.65 |
| 2030 | 0.24 | 0.40 | 0.48 |
| 2035 | 0.15 | 0.25 | 0.30 |
| 2040 | 0.08 | 0.14 | 0.16 |
| 2050 | 0.02 | 0.03 | 0.03 |

**Note:** Integrated kraft mills typically have the lowest fossil intensity because they generate most of their energy from black liquor combustion in recovery boilers. Mechanical pulp mills and recycled paper mills are more electricity-intensive and therefore more dependent on grid decarbonization.

---

## Technology Landscape

### Current Energy Mix (Global Average, 2023)

| Energy Source | Share of Total Energy | Status |
|--------------|----------------------|--------|
| Black liquor (biogenic) | 30-35% | Core to kraft process |
| Bark and wood waste (biogenic) | 15-20% | Mill residue utilization |
| Natural gas | 15-20% | Fossil fuel (lime kiln, boilers) |
| Purchased electricity | 10-15% | Grid-dependent |
| Coal | 5-10% | Phasing out |
| Oil | 3-5% | Declining |
| Other biomass | 2-5% | Growing |

### Key Technology Transitions

#### 1. Fossil Fuel Elimination in Lime Kilns

- **Transition**: Replace natural gas/oil in rotary lime kilns with:
  - Biomass gasification (syngas)
  - Electric lime kilns (emerging)
  - Green hydrogen
- **Timeline**: Biomass gasification commercial now; electric kilns 2028-2035
- **Reduction**: 15-25% of fossil Scope 1 (lime kiln = largest fossil emission source in kraft mills)
- **Cost**: EUR 30-70/tCO2e (biomass gasification); EUR 60-120/tCO2e (electric)
- **Certainty**: High (biomass gasification); Medium (electric)

#### 2. Electrification of Steam and Heat

- **Transition**: Replace fossil-fuel-fired auxiliary boilers with:
  - Electric boilers
  - Heat pumps (for low-temperature heat)
  - Electrode boilers
- **Timeline**: Electric boilers available now; heat pumps scaling 2025-2030
- **Reduction**: 10-20% of fossil Scope 1
- **Cost**: EUR 20-60/tCO2e (depends on electricity price)
- **Certainty**: High

#### 3. Renewable Electricity Procurement

- **Transition**: Shift from grid electricity to renewable PPAs, on-site renewables, or green tariffs
- **Timeline**: Continuous, market-dependent
- **Reduction**: 20-40% of Scope 2 emissions (varies by current grid carbon intensity)
- **Cost**: Neutral to slightly positive in most markets
- **Certainty**: High

#### 4. Energy Efficiency and Process Optimization

- **Transition**: Shoe press technology, high-consistency forming, waste heat recovery, optimized drying
- **Reduction**: 10-15% of total energy consumption
- **Cost**: Negative (energy savings)
- **Certainty**: High
- **Specific measures**:
  - Advanced shoe press: saves 5-10 kWh/t paper (reduced drying energy)
  - Waste heat recovery from paper machine exhaust: saves 0.5-1.0 GJ/t
  - Optimized steam and condensate systems: 5-8% thermal energy reduction

#### 5. Increased Biomass Energy Utilization

- **Transition**: Maximize use of mill residues and sustainably sourced biomass
- **Current**: ~55% of energy from biomass (global average for chemical pulp mills)
- **Target**: 70-80% by 2030 for integrated mills
- **Reduction**: Displaces remaining fossil fuel use
- **Cost**: Low (mill residues are low-cost; external biomass moderate)
- **Dependencies**: Sustainable biomass supply, avoiding competition with food/fiber

#### 6. BECCS (Bioenergy with Carbon Capture and Storage)

- **Transition**: Add CCS to biomass-fired boilers (recovery boiler, bark boiler) to achieve negative emissions
- **Timeline**: Pilot 2025-2030, commercial 2030-2040
- **Reduction**: Net-negative emissions potential (capturing biogenic CO2)
- **Cost**: EUR 60-120/tCO2e
- **Certainty**: Medium (technology proven; economics and CO2 storage infrastructure are barriers)
- **Strategic importance**: Pulp mills are ideal BECCS candidates due to large, concentrated biogenic CO2 streams

---

## Abatement Levers

### Lever Waterfall (Integrated Kraft Pulp & Paper Mill, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Energy efficiency (shoe press, heat recovery) | 10-15% | -20 to -5 | High |
| 2 | Renewable electricity PPAs | 15-25% | 0 to +15 | High |
| 3 | Biomass gasification for lime kiln | 15-20% | +30 to +70 | High |
| 4 | Electric boiler for auxiliary steam | 10-15% | +20 to +60 | High |
| 5 | Increased bark/residue utilization | 5-10% | -5 to +10 | High |
| 6 | BECCS pilot (recovery boiler) | 5-10% (net-negative) | +60 to +120 | Medium |

### Lever Waterfall (Recycled Paper Mill, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Renewable electricity procurement | 25-35% | 0 to +15 | High |
| 2 | Natural gas to electric boiler conversion | 20-30% | +20 to +60 | High |
| 3 | Energy efficiency (drying optimization) | 10-15% | -20 to -5 | High |
| 4 | Heat pump integration (low-temp heat) | 5-10% | +15 to +40 | Medium |
| 5 | On-site solar PV / wind | 3-5% | -5 to +10 | High |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | Average energy efficiency improvement of 10% vs. 2015 baseline |
| 2025 | All new pulp mills designed for >80% bioenergy self-sufficiency |
| 2030 | 50% reduction in fossil fuel use across the sector |
| 2030 | First commercial BECCS projects on pulp mills operational |
| 2030 | Coal phase-out complete in pulp and paper sector globally |
| 2035 | Lime kiln fossil fuel replacement reaches 50% of capacity |
| 2040 | 80% of sector energy from renewable/biogenic sources |
| 2050 | Pulp and paper sector achieves near-zero fossil emissions |

---

## Benchmarks (2024)

| Benchmark | Value (tCO2e/t) | Source |
|-----------|-----------------|--------|
| Global average (all grades) | 0.55 | CEPI/FAO |
| Integrated kraft mill (best practice) | 0.15 | CEPI Benchmarks |
| Recycled paper mill (average) | 0.65 | CEPI Benchmarks |
| Mechanical pulp mill (average) | 0.80 | CEPI Benchmarks |
| Sector leader (P10) | 0.12 | CDP Climate 2024 |
| SBTi peer average | 0.40 | SBTi Database 2024 |
| IEA NZE 2025 target | 0.47 | IEA NZE 2050 |
| Nordic average (kraft) | 0.18 | CEPI Country Statistics |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.sector_benchmark_engine import SectorBenchmarkEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["C17.12"]})
# Result: pulp_paper

# Generate pathway
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="pulp_paper",
    base_year=2023,
    base_year_intensity=0.375,  # tCO2e/t product
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="linear",
    production_forecast={2023: 800_000, 2030: 850_000, 2050: 900_000},  # tonnes
    region="oecd",
)

print(f"2030 Target: {pathway.target_2030:.3f} tCO2e/t")
print(f"2050 Target: {pathway.target_2050:.3f} tCO2e/t")

# Benchmark against peers
benchmark = SectorBenchmarkEngine()
bench_result = benchmark.assess(
    sector="pulp_paper",
    company_intensity=0.375,
    year=2023,
    sub_sector="integrated_kraft",
    region="oecd",
)

print(f"Percentile rank: {bench_result.percentile_rank}")
print(f"vs. Sector average: {bench_result.vs_sector_average:+.1%}")
print(f"vs. SBTi peers: {bench_result.vs_sbti_peers:+.1%}")
```

---

## Special Considerations

### Biogenic Emissions Accounting

The pulp and paper sector has unique emissions accounting requirements:

- **Biogenic CO2 from biomass combustion** (black liquor, bark, wood waste) is reported separately and counted as zero in GHG Protocol Scope 1 if the biomass is from sustainably managed forests
- **This means the SDA intensity metric only covers fossil CO2**, which is typically 20-40% of the total CO2 emitted from a kraft mill
- Companies must demonstrate sustainable forest management (e.g., FSC, PEFC certification) for biogenic emissions to be counted as zero
- BECCS (capturing biogenic CO2) can result in negative emissions, which is a unique opportunity for the sector

### Forest Carbon and Scope 3

While not part of the SDA pathway (Scope 1+2 only), forest carbon sequestration is a critical aspect of the sector's climate impact:
- Sustainable forest management can maintain or increase forest carbon stocks
- Scope 3 Category 1 (purchased wood fiber) is material for non-integrated paper mills
- SBTi FLAG guidance applies to companies with significant forest/land use activities

### Paper Recycling

Higher paper recycling rates reduce the need for virgin fiber but increase dependence on purchased electricity (recycled paper mills are more electricity-intensive per tonne than integrated kraft mills). The net emission impact depends on the carbon intensity of the electricity grid.

---

## References

1. SBTi SDA-Pulp & Paper Methodology, SBTi Target Setting Tool V3.0 (2025)
2. IEA Net Zero by 2050, Chapter 5: Industry (Pulp & Paper)
3. CEPI Key Statistics 2024 (Confederation of European Paper Industries)
4. FAO Global Forest Products Statistics 2024
5. IPCC AR6 WGIII, Chapter 11: Industry
6. CEPI Two Team Project: Energy Efficiency and Carbon Reduction in European Paper Industry

---

**End of Pulp & Paper Sector Guide**
