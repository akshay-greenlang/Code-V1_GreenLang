# Sector Guide: Power Generation

**Sector ID:** `power_generation`
**SDA Methodology:** SDA-Power
**Intensity Metric:** gCO2/kWh
**IEA Chapter:** Chapter 3 -- Electricity

---

## Sector Overview

The power generation sector is responsible for approximately 36% of global energy-related CO2 emissions, making it the single largest emitting sector. The SBTi SDA-Power methodology uses grid average emission intensity (gCO2/kWh) as the convergence metric, reflecting the overall carbon intensity of electricity generated.

Decarbonizing power generation is foundational because many other sector pathways (steel, transport, buildings) depend on clean electricity availability. The IEA NZE 2050 scenario requires the power sector to reach net-zero emissions by 2050, with advanced economies achieving this by 2035.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `PWR-01` | Grid average emission intensity | gCO2/kWh | Primary SDA metric. Total Scope 1+2 emissions divided by total electricity generated |
| `PWR-02` | Generation source intensity | tCO2e/MWh | Intensity by generation source (coal, gas, nuclear, renewable) |
| `PWR-03` | Capacity-weighted intensity | gCO2/kWh | Intensity weighted by installed capacity (MW) |
| `PWR-04` | Lifecycle emission intensity | gCO2e/kWh | Full lifecycle emissions including construction and decommissioning |

### Calculating PWR-01 (Primary Metric)

```python
intensity_gco2_kwh = (scope1_tco2e + scope2_tco2e) * 1_000_000 / electricity_generated_kwh
```

**Example:**
- Scope 1 emissions: 4,500,000 tCO2e
- Scope 2 emissions: 50,000 tCO2e (self-consumption)
- Electricity generated: 10,000,000 MWh = 10,000,000,000 kWh
- Intensity: (4,550,000 * 1,000,000) / 10,000,000,000 = 455 gCO2/kWh

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway

| Year | Intensity (gCO2/kWh) | Reduction from 2020 |
|------|---------------------|-------------------|
| 2020 | 450 (global average) | Baseline |
| 2025 | 350 | -22% |
| 2030 | 220 | -51% |
| 2035 | 100 | -78% |
| 2040 | 40 | -91% |
| 2045 | 10 | -98% |
| 2050 | 0 | -100% |

### Regional Pathway Variants

| Year | Global | OECD | Emerging Markets |
|------|--------|------|-----------------|
| 2025 | 350 | 280 | 420 |
| 2030 | 220 | 140 | 300 |
| 2035 | 100 | 0 | 200 |
| 2040 | 40 | 0 | 80 |
| 2050 | 0 | 0 | 0 |

---

## Technology Landscape

### Current Technology Mix (Global, 2023)

| Technology | Share | Intensity (gCO2/kWh) | Status |
|-----------|-------|---------------------|--------|
| Coal | 36% | 900-1,100 | Phase-out required |
| Natural Gas | 22% | 350-500 | Transition fuel |
| Nuclear | 10% | 5-12 | Low-carbon baseload |
| Hydropower | 15% | 4-24 | Established renewable |
| Solar PV | 7% | 20-50 | Rapid growth |
| Wind (onshore) | 6% | 7-15 | Established renewable |
| Wind (offshore) | 2% | 8-20 | Growing rapidly |
| Biomass/Other | 2% | 50-230 | Variable |

### Key Technology Transitions

#### 1. Coal Phase-Out
- **Transition**: Coal plants retired or retrofitted with CCS
- **Timeline**: OECD by 2035, global by 2040
- **Cost**: Stranded asset cost offset by avoided fuel + carbon costs
- **Dependencies**: Replacement capacity must be built first

#### 2. Renewable Capacity Expansion
- **Transition**: Solar PV + onshore/offshore wind to 60%+ of generation by 2030
- **Timeline**: Continuous to 2050
- **Cost**: LCOE now below coal and gas in most regions
- **Dependencies**: Grid infrastructure, permitting, supply chains

#### 3. Grid-Scale Energy Storage
- **Transition**: Battery storage (4-8 hour), pumped hydro, hydrogen storage
- **Timeline**: Rapid deployment 2025-2040
- **Cost**: Battery costs declining 10-15%/year
- **Dependencies**: Battery manufacturing capacity, mineral supply

#### 4. Nuclear Capacity
- **Transition**: Existing fleet life extensions + new build (large reactors + SMRs)
- **Timeline**: SMR deployment from 2030
- **Cost**: High upfront, low marginal
- **Dependencies**: Regulatory approval, waste management, public acceptance

#### 5. Grid Infrastructure
- **Transition**: Transmission expansion, interconnectors, smart grid, demand response
- **Timeline**: Continuous investment required
- **Cost**: $100-300B/year globally
- **Dependencies**: Permitting, right-of-way, cross-border agreements

#### 6. Carbon Capture (Fossil + BECCS)
- **Transition**: CCS on remaining gas plants, BECCS for negative emissions
- **Timeline**: CCS commercial from 2030, BECCS from 2035
- **Cost**: $60-120/tCO2e
- **Dependencies**: CO2 transport and storage infrastructure

---

## Abatement Levers

### Lever Waterfall (Typical Utility, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Renewable capacity expansion (solar + wind) | 25-35% | -20 to -40 (saves money) | High |
| 2 | Coal plant retirement / phase-out | 15-25% | -10 to +20 | High |
| 3 | Gas plant efficiency improvements | 3-5% | -10 to 0 | High |
| 4 | Grid-scale battery storage | 5-8% | +10 to +40 | High |
| 5 | Demand response / smart grid | 2-4% | -15 to -5 | Medium |
| 6 | Nuclear capacity (life extension / new build) | 5-10% | +30 to +80 | Medium |
| 7 | CCS on remaining fossil generation | 5-10% | +60 to +120 | Medium |

---

## IEA Key Milestones

| Year | Milestone | Status Tracking |
|------|-----------|----------------|
| 2025 | No new unabated coal plants approved anywhere globally | Company: No coal expansion in pipeline |
| 2025 | Annual solar PV additions reach 630 GW globally | Company: Solar PV capacity additions on track |
| 2030 | Renewable capacity reaches 11,000 GW globally | Company: Renewable share vs. target |
| 2030 | 60% of global electricity from renewables | Company: Renewable generation share |
| 2035 | All unabated coal plants in advanced economies retired | Company: Coal plant retirement schedule |
| 2035 | Advanced economies achieve net-zero electricity | Company: Grid emission factor trend |
| 2040 | 80% of global electricity from clean sources | Company: Clean energy share |
| 2050 | Global power sector reaches net-zero emissions | Company: Absolute emissions trajectory |

---

## Benchmarks

### Global Power Sector Benchmarks (2024)

| Benchmark | Value (gCO2/kWh) | Source |
|-----------|------------------|--------|
| Global average | 450 | IEA Statistics 2024 |
| OECD average | 320 | IEA Statistics 2024 |
| EU average | 230 | EEA 2024 |
| Sector leader (P10) | 45 | CDP Climate 2024 |
| SBTi peer average | 280 | SBTi Database 2024 |
| IEA NZE 2025 target | 350 | IEA NZE 2050 |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.technology_roadmap_engine import TechnologyRoadmapEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["D35.11"]})
# Result: power_generation

# Generate pathway
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="power_generation",
    base_year=2023,
    base_year_intensity=455,  # gCO2/kWh
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="exponential",
    production_forecast={2023: 10_000_000, 2030: 12_000_000, 2050: 15_000_000},  # MWh
    region="eu",
)

print(f"2030 Target: {pathway.target_2030:.0f} gCO2/kWh")
print(f"2050 Target: {pathway.target_2050:.0f} gCO2/kWh")

# Build technology roadmap
roadmap = TechnologyRoadmapEngine()
result = roadmap.build(
    sector="power_generation",
    pathway=pathway,
    current_technology_mix={
        "coal": 0.30, "natural_gas": 0.25, "nuclear": 0.15,
        "hydro": 0.10, "solar_pv": 0.10, "wind_onshore": 0.08, "wind_offshore": 0.02,
    },
    installed_capacity_mw=15_000,
    capex_budget_annual_usd=2_000_000_000,
    region="eu",
)
```

---

## Regulatory Context

| Regulation | Relevance to Power Sector |
|-----------|--------------------------|
| EU ETS Phase 4 | Carbon pricing for power generators (EUR 80-100/tCO2e) |
| EU Fit for 55 | 55% GHG reduction by 2030, renewable energy targets |
| REPowerEU | Accelerated renewable deployment, reduced Russian gas dependency |
| US IRA | Tax credits for clean energy (PTC, ITC, 45Q for CCS) |
| UK Carbon Price Floor | Minimum carbon price for power generators |
| China ETS | Covers power sector (world's largest ETS by covered emissions) |

---

## References

1. SBTi SDA-Power Methodology, SBTi Target Setting Tool V3.0 (2025)
2. IEA Net Zero by 2050: A Roadmap for the Global Energy Sector, 2023 Update
3. IEA World Energy Outlook 2024, Chapter 3: Electricity
4. IPCC AR6 WGIII, Chapter 6: Energy Systems
5. IRENA Renewable Power Generation Costs 2024
6. Global Electricity Review 2024, Ember

---

**End of Power Generation Sector Guide**
