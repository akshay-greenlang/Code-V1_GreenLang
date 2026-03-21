# Sector Guide: Rail

**Sector ID:** `rail`
**SDA Methodology:** SDA-Rail
**Intensity Metric:** gCO2/pkm (passenger) or gCO2/tkm (freight)
**IEA Chapter:** Chapter 4 -- Transport (Rail)

---

## Sector Overview

Rail transport is one of the most energy-efficient and lowest-carbon modes of transport per unit of work. The global rail sector accounts for approximately 0.4% of global energy-related CO2 emissions, despite carrying 8% of passenger transport and 7% of freight transport. This inherent efficiency makes rail a critical part of the decarbonization solution, as modal shift from road and aviation to rail can reduce overall transport sector emissions.

Key characteristics of the rail sector:

1. **Already highly efficient**: Rail has the lowest carbon intensity per passenger-km or tonne-km of any motorized transport mode
2. **High electrification potential**: Unlike aviation and shipping, rail can be directly electrified; approximately 50% of global rail traffic is already electric
3. **Dual fuel mix**: Electric traction (overhead catenary or third rail) and diesel traction
4. **Long asset lifecycles**: Rolling stock has 30-40 year lifespans; infrastructure even longer
5. **Modal shift benefit**: Growth in rail share reduces emissions from road and aviation
6. **Urban/suburban context**: Metro and commuter rail systems are essential for urban decarbonization

The SBTi SDA-Rail methodology uses gCO2 per passenger-kilometer (pkm) for passenger services and gCO2 per net-tonne-kilometer (ntkm) for freight services.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `RAL-01` | Passenger rail intensity | gCO2/pkm | Primary SDA metric for passenger rail. Total CO2 per passenger-km |
| `RAL-02` | Freight rail intensity | gCO2/ntkm | Primary SDA metric for freight rail. CO2 per net tonne-km |
| `RAL-03` | Energy intensity (passenger) | kWh/pkm or MJ/pkm | Energy consumption per passenger-km |
| `RAL-04` | Energy intensity (freight) | kWh/ntkm or MJ/ntkm | Energy consumption per net tonne-km |
| `RAL-05` | Electrification rate | % of track-km | Percentage of railway network that is electrified |
| `RAL-06` | Traction energy mix | % electric vs. diesel | Share of traction energy from electric vs. diesel sources |

### Calculating RAL-01 (Primary Metric -- Passenger)

```python
# Electric traction emissions
electric_co2 = electricity_consumed_kwh * grid_emission_factor_kgco2_per_kwh / 1000

# Diesel traction emissions
diesel_co2 = diesel_consumed_litres * diesel_ef_kgco2_per_litre / 1000
# Diesel emission factor: 2.68 kgCO2/litre

total_co2_tonnes = electric_co2 + diesel_co2
total_pkm = total_passengers * average_journey_distance_km

intensity_gco2_per_pkm = (total_co2_tonnes * 1_000_000) / total_pkm
```

**Example (National Railway Operator):**
- Electric traction: 3,000,000 MWh at 250 gCO2/kWh = 750,000 tCO2
- Diesel traction: 200,000,000 litres * 2.68 = 536,000 tCO2
- Total CO2: 1,286,000 tCO2
- Total passenger-km: 40,000,000,000 pkm (40 billion pkm)
- Intensity: (1,286,000 * 1,000,000) / 40,000,000,000 = 32.2 gCO2/pkm

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway (Passenger)

| Year | Intensity (gCO2/pkm) | Reduction from 2020 |
|------|---------------------|-------------------|
| 2020 | 30 (global avg) | Baseline |
| 2025 | 25 | -17% |
| 2030 | 18 | -40% |
| 2035 | 10 | -67% |
| 2040 | 5 | -83% |
| 2045 | 2 | -93% |
| 2050 | 0.5 | -98% |

### NZE 1.5C Convergence Pathway (Freight)

| Year | Intensity (gCO2/ntkm) | Reduction from 2020 |
|------|----------------------|-------------------|
| 2020 | 20 (global avg) | Baseline |
| 2025 | 16 | -20% |
| 2030 | 11 | -45% |
| 2035 | 6 | -70% |
| 2040 | 3 | -85% |
| 2045 | 1 | -95% |
| 2050 | 0.3 | -99% |

### Regional Pathway Variants (Passenger)

| Year | Global | EU (highly electrified) | India/China (mixed) | US/Canada (diesel-heavy) |
|------|--------|------------------------|--------------------|-----------------------|
| 2020 | 30 | 20 | 25 | 55 |
| 2025 | 25 | 15 | 20 | 45 |
| 2030 | 18 | 8 | 14 | 30 |
| 2035 | 10 | 3 | 8 | 18 |
| 2040 | 5 | 1 | 4 | 8 |
| 2050 | 0.5 | 0.2 | 0.4 | 1.0 |

---

## Technology Landscape

### Current Traction Mix (Global, 2023)

| Traction Type | Share of Traffic | Intensity (approx.) | Status |
|--------------|-----------------|---------------------|--------|
| Electric (catenary/third rail, clean grid) | 25% | 5-15 gCO2/pkm | Growing |
| Electric (catenary, average grid) | 25% | 15-40 gCO2/pkm | Growing |
| Diesel | 30% | 40-80 gCO2/pkm | Declining |
| Diesel-electric (hybrid) | 5% | 30-50 gCO2/pkm | Growing |
| Metro/Light rail (electric) | 10% | 5-25 gCO2/pkm | Growing |
| High-speed rail (electric) | 5% | 8-20 gCO2/pkm | Growing rapidly |

### Key Technology Transitions

#### 1. Network Electrification

- **Transition**: Extend electrified rail network (catenary or third rail)
- **Current**: ~50% of global rail traffic on electric traction; but only ~35% of track-km electrified
- **Target**: 70% of track-km electrified by 2030; 90% by 2050
- **Reduction**: Eliminates diesel emissions on electrified routes (100% if grid is clean)
- **Cost**: EUR 1-3 million per km of electrification (varies by terrain and corridor)
- **Timeline**: 10-15 years per major corridor; continuous investment
- **Certainty**: Very High (proven technology; primary pathway for rail decarbonization)

#### 2. Battery-Electric Multiple Units (BEMUs)

- **Transition**: Replace diesel trains with battery-electric trains on non-electrified branch lines
- **Timeline**: Commercially available now (Stadler FLIRT Akku, Siemens Mireo Plus B)
- **Range**: 80-150 km on battery (charges on electrified sections)
- **Reduction**: 100% tailpipe on battery operation
- **Cost**: 20-40% premium over diesel trains; lower operating costs
- **Best applications**: Branch lines connected to electrified mainlines; short non-electrified gaps
- **Certainty**: High (commercial products available)

#### 3. Hydrogen Fuel Cell Trains

- **Transition**: Replace diesel trains with hydrogen fuel cell trains on non-electrified lines
- **Timeline**: Commercially available (Alstom Coradia iLINT, Siemens Mireo Plus H)
- **Range**: 600-1,000 km (longer range than BEMU)
- **Reduction**: 100% tailpipe; lifecycle depends on hydrogen source
- **Cost**: 20-50% premium over diesel; lower operating cost with cheap green hydrogen
- **Best applications**: Long non-electrified routes where electrification is not economical
- **Dependencies**: Green hydrogen supply, refueling infrastructure
- **Certainty**: High (commercially proven in Germany, France, Italy)

#### 4. Renewable Electricity Procurement

- **Transition**: Source electricity for traction from renewable sources (PPAs, green tariffs)
- **Reduction**: Reduces Scope 2 to near-zero
- **Cost**: Neutral to slightly positive in most markets
- **Timeline**: Immediate (procurement decision)
- **Certainty**: Very High
- **Example**: Dutch Railways (NS) claims 100% wind-powered operations since 2017

#### 5. Energy Recovery and Efficiency

- **Transition**: Regenerative braking, lightweight rolling stock, energy-efficient driving, optimized timetabling
- **Reduction**: 10-25% of energy consumption
- **Cost**: Negative (energy savings)
- **Timeline**: Continuous improvement
- **Certainty**: High
- **Measures**:
  - Regenerative braking: recovers 15-30% of traction energy
  - Driver advisory systems (DAS): 5-10% energy savings
  - Lightweight materials: 3-8% reduction per tonne of weight saved
  - Optimized acceleration/braking profiles: 5-8% savings

#### 6. High-Speed Rail Expansion

- **Transition**: Build new high-speed rail lines to replace short-haul aviation
- **Reduction**: 80-95% per passenger-km vs. aviation for comparable journeys
- **Timeline**: Major projects 2025-2040
- **Cost**: High infrastructure cost (EUR 15-40 million per km); competitive operations
- **Certainty**: High (proven technology and demand)
- **Modal shift impact**: HSR captures 60-80% of air traffic on routes <600 km

---

## Abatement Levers

### Lever Waterfall (National Railway Operator, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Renewable electricity procurement (100% green PPAs) | 30-45% | 0 to +15 | Very High |
| 2 | Network electrification (priority corridors) | 10-15% | +50 to +150 | High |
| 3 | Battery-electric trains (branch lines) | 5-8% | +20 to +60 | High |
| 4 | Energy efficiency (regenerative braking, DAS) | 8-12% | -20 to -5 | High |
| 5 | Hydrogen trains (remote non-electrified lines) | 3-5% | +40 to +100 | Medium |
| 6 | Fleet renewal (modern efficient rolling stock) | 5-8% | +10 to +30 | High |
| 7 | Biofuel blending for remaining diesel fleet | 2-4% | +20 to +50 | High |

### Lever Waterfall (Freight Railway Operator, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Renewable electricity for electric traction | 25-35% | 0 to +15 | Very High |
| 2 | Electrification of key freight corridors | 10-15% | +40 to +120 | High |
| 3 | Operational efficiency (train length, load factor) | 8-12% | -25 to -5 | High |
| 4 | Eco-driving and DAS | 5-8% | -15 to -5 | High |
| 5 | Locomotive fleet renewal | 5-8% | +10 to +30 | High |
| 6 | Battery-electric shunting locomotives | 2-4% | +20 to +50 | High |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | No new diesel-only passenger trains ordered in OECD countries |
| 2025 | 55% of rail passenger traffic on electric traction globally |
| 2030 | 70% of rail network track-km electrified in advanced economies |
| 2030 | Battery and hydrogen trains replace all new diesel orders |
| 2030 | Rail freight modal share increases by 25% vs. 2020 |
| 2035 | All rail passenger services in EU and Japan fully zero-emission |
| 2040 | 90% of global rail traffic on electric or zero-emission traction |
| 2050 | Rail sector achieves near-zero emissions globally |

---

## Benchmarks (2024)

| Benchmark | Value | Unit | Source |
|-----------|-------|------|--------|
| Global average (passenger) | 30 | gCO2/pkm | UIC Statistics |
| EU average (passenger) | 20 | gCO2/pkm | UIC/EEA |
| Japan (Shinkansen) | 12 | gCO2/pkm | JR Company reports |
| India (average) | 25 | gCO2/pkm | Indian Railways |
| US (Amtrak, average) | 55 | gCO2/pkm | Amtrak Sustainability Report |
| Sector leader (fully electric, clean grid) | 5 | gCO2/pkm | Company disclosures |
| Global average (freight) | 20 | gCO2/ntkm | UIC Statistics |
| EU average (freight) | 15 | gCO2/ntkm | UIC/EEA |
| US Class I railroads (freight) | 15 | gCO2/ntkm | AAR/DOE |
| SBTi peer average | 22 | gCO2/pkm | SBTi Database 2024 |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.sector_benchmark_engine import SectorBenchmarkEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["H49.10"]})
# Result: rail

# Generate pathway
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="rail",
    sub_sector="passenger",
    base_year=2023,
    base_year_intensity=32.2,  # gCO2/pkm
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="linear",
    production_forecast={
        2023: 40_000_000_000,  # pkm
        2030: 50_000_000_000,
        2050: 65_000_000_000,
    },
    region="oecd",
    electrification_rate_current=0.65,
    electrification_rate_plan={2025: 0.70, 2030: 0.82, 2040: 0.95},
)

print(f"2030 Target: {pathway.target_2030:.1f} gCO2/pkm")
print(f"2050 Target: {pathway.target_2050:.1f} gCO2/pkm")

# Benchmark
benchmark = SectorBenchmarkEngine()
bench_result = benchmark.assess(
    sector="rail",
    company_intensity=32.2,
    year=2023,
    sub_sector="passenger_national",
    region="oecd",
)

print(f"Percentile rank: P{bench_result.percentile_rank}")
print(f"vs. Sector average: {bench_result.vs_sector_average:+.1%}")
```

---

## Special Considerations

### Modal Shift Benefits

Rail's greatest climate contribution may be through modal shift rather than direct emission reductions. The emission savings from shifting traffic from road and aviation to rail are substantial:

| Shift | Emission Saving per pkm Shifted | Context |
|-------|-------------------------------|---------|
| Aviation to High-Speed Rail | 80-95% | Routes <600 km |
| Private car to Intercity Rail | 60-80% | Medium-distance travel |
| Private car to Urban Metro/Light Rail | 70-90% | Urban commuting |
| Road freight to Rail freight | 60-80% | Intermodal logistics |

PACK-028 can model these modal shift scenarios using the Scenario Comparison Engine, quantifying the system-level emission benefits of rail investment.

### Scope 2 Sensitivity

For electric railways, the carbon intensity is almost entirely determined by the electricity grid carbon intensity. This means:
- A railway operator purchasing 100% renewable electricity can achieve near-zero emissions immediately
- The same physical railway on a coal-dominated grid would have 10-20x higher intensity
- This makes green electricity procurement the single fastest lever for electric rail operators

### Infrastructure vs. Operations

Rail decarbonization involves both infrastructure (electrification, which is capital-intensive and slow) and operations (procurement, efficiency, which can be fast). PACK-028 models both dimensions and helps operators prioritize the optimal mix.

---

## Regulatory Context

| Regulation | Relevance to Rail |
|-----------|------------------|
| EU Sustainable and Smart Mobility Strategy | Target: double high-speed rail by 2030, triple by 2050 |
| EU TEN-T Regulation | Trans-European Transport Network: electrification and interoperability requirements |
| European Year of Rail (legacy) | Policy focus on rail as the sustainable transport backbone |
| EU ETS (indirect) | Carbon costs on electricity affect electric rail operating costs |
| National rail investment programs | UK, France, Germany, India, China major rail investment programs |
| IMO/ICAO comparison | Rail competes with regulated aviation and shipping; favored in policy |

---

## References

1. SBTi SDA-Rail Methodology, SBTi Target Setting Tool V3.0 (2025)
2. IEA Net Zero by 2050, Chapter 4: Transport (Rail)
3. UIC (International Union of Railways) Statistics 2024
4. IPCC AR6 WGIII, Chapter 10: Transport
5. European Rail Research Advisory Council (ERRAC) Rail 2050 Vision
6. IEA The Future of Rail (2019)

---

**End of Rail Sector Guide**
