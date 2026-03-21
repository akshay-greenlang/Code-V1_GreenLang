# Sector Guide: Road Transport

**Sector ID:** `road_transport`
**SDA Methodology:** SDA-Road Transport
**Intensity Metric:** gCO2/pkm (passenger) or gCO2/tkm (freight)
**IEA Chapter:** Chapter 4 -- Transport (Road)

---

## Sector Overview

Road transport is the largest transport sub-sector by emissions, accounting for approximately 16% of global energy-related CO2 emissions (~6 Gt CO2/year). It encompasses passenger vehicles (cars, buses, two/three-wheelers), light commercial vehicles (LCVs), and heavy-duty vehicles (trucks, HDVs). The global vehicle fleet exceeds 1.4 billion units, with approximately 90 million new vehicles sold annually.

Unlike aviation and shipping, road transport has a clear and increasingly cost-competitive decarbonization pathway through direct electrification. Battery electric vehicles (BEVs) are experiencing rapid adoption, with global EV sales exceeding 14 million in 2023 (18% of new car sales). For heavy-duty long-haul trucking, both battery-electric and hydrogen fuel cell technologies are competing as zero-emission solutions.

Key characteristics:

1. **Electrification pathway**: BEVs are the primary decarbonization solution for passenger vehicles and increasingly for medium/heavy-duty vehicles
2. **Grid dependency**: EV emissions depend on electricity grid carbon intensity
3. **Rapid technology shift**: EV cost parity with ICE vehicles achieved in many markets
4. **Infrastructure requirements**: Charging infrastructure buildout is critical
5. **Diverse regulatory landscape**: Different countries have different ICE phase-out dates (Norway 2025, EU 2035, etc.)
6. **Biofuel role**: Sustainable biofuels serve as a transition fuel, especially for existing fleet

The SBTi SDA-Road Transport methodology uses gCO2 per vehicle-kilometer (vkm) or gCO2 per passenger-kilometer (pkm) for passenger vehicles and gCO2 per tonne-kilometer (tkm) for freight.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `RDT-01` | Passenger vehicle intensity | gCO2/vkm | Primary SDA metric for passenger vehicles. Tailpipe CO2 per vehicle-km |
| `RDT-02` | Passenger mobility intensity | gCO2/pkm | CO2 per passenger-kilometer (accounts for occupancy) |
| `RDT-03` | Heavy-duty vehicle intensity | gCO2/tkm | CO2 per tonne-kilometer for freight transport |
| `RDT-04` | Fleet average CO2 | gCO2/km | Average new vehicle fleet CO2 emissions |
| `RDT-05` | Well-to-wheel intensity | gCO2e/km | Full lifecycle fuel/energy pathway emissions |
| `RDT-06` | Zero-emission vehicle share | % of fleet | Percentage of fleet that is zero-emission (BEV/FCEV) |

### Calculating RDT-01 (Primary Metric - Passenger)

```python
# Tank-to-wheel (tailpipe) emissions
total_co2_tonnes = sum(
    fuel_litres[fuel] * fuel_density[fuel] * emission_factor[fuel]
    for fuel in fuels
)
# Emission factors (kgCO2/litre):
# Gasoline: 2.31, Diesel: 2.68, LPG: 1.51, CNG: 2.75 (per m3)
# BEV: 0.0 (tailpipe), FCEV: 0.0 (tailpipe)

total_vehicle_km = sum(vehicle_km_per_vehicle * number_of_vehicles)

intensity_gco2_per_vkm = (total_co2_tonnes * 1_000_000) / total_vehicle_km

# Well-to-wheel for BEV:
wtw_bev_gco2_per_km = electricity_kwh_per_km * grid_emission_factor_gco2_per_kwh
# Example: 0.18 kWh/km * 300 gCO2/kWh = 54 gCO2/km (well-to-wheel)
```

**Example (Automotive OEM Fleet):**
- Total fuel consumption: 15,000,000,000 litres gasoline equivalent
- Total CO2: 15,000,000,000 * 2.31 / 1000 = 34,650,000 tCO2 (fleet-level)
- Fleet average new car: 120 gCO2/km (WLTP)
- Average occupancy: 1.5 passengers
- Intensity: 120 / 1.5 = 80 gCO2/pkm

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway (Passenger Vehicles)

| Year | New Fleet Avg (gCO2/km) | On-Road Fleet Avg (gCO2/km) | Reduction from 2020 |
|------|------------------------|---------------------------|-------------------|
| 2020 | 140 (global avg new) | 185 (on-road) | Baseline |
| 2025 | 110 | 170 | -8% |
| 2030 | 60 | 140 | -24% |
| 2035 | 15 | 105 | -43% |
| 2040 | 0 | 70 | -62% |
| 2045 | 0 | 35 | -81% |
| 2050 | 0 | 10 | -95% |

### NZE 1.5C Convergence Pathway (Heavy-Duty Freight)

| Year | Intensity (gCO2/tkm) | Reduction from 2020 |
|------|---------------------|-------------------|
| 2020 | 80 (global avg) | Baseline |
| 2025 | 72 | -10% |
| 2030 | 55 | -31% |
| 2035 | 35 | -56% |
| 2040 | 18 | -78% |
| 2045 | 8 | -90% |
| 2050 | 3 | -96% |

### EV Sales Share Trajectory (Passenger Cars)

| Year | Global | EU | China | US |
|------|--------|-----|-------|-----|
| 2023 | 18% | 22% | 35% | 9% |
| 2025 | 25% | 35% | 45% | 15% |
| 2030 | 60% | 80% | 75% | 50% |
| 2035 | 85% | 100% | 90% | 80% |
| 2040 | 95% | 100% | 98% | 95% |
| 2050 | 100% | 100% | 100% | 100% |

---

## Technology Landscape

### Current Powertrain Mix (Global New Sales, 2023)

| Powertrain | Market Share | Tailpipe CO2 (gCO2/km) | Status |
|-----------|-------------|----------------------|--------|
| Gasoline ICE | 45% | 120-180 | Declining |
| Diesel ICE | 15% | 110-160 | Declining rapidly |
| Hybrid (HEV) | 12% | 80-120 | Growing (transition) |
| Plug-in Hybrid (PHEV) | 5% | 30-60 (WLTP; higher real-world) | Growing |
| Battery Electric (BEV) | 18% | 0 (tailpipe) | Rapid growth |
| Fuel Cell (FCEV) | <0.5% | 0 (tailpipe) | Niche |
| CNG/LPG | 4% | 100-140 | Stable/declining |
| Flex-fuel (ethanol) | <1% | 80-120 (fossil portion) | Regional (Brazil) |

### Key Technology Transitions

#### 1. Battery Electric Vehicles (BEV) -- Passenger

- **Transition**: Replace ICE passenger vehicles with BEVs
- **Timeline**: Ongoing; cost parity achieved 2024-2025 in many segments; 100% of new sales by 2035-2040
- **Reduction**: 100% tailpipe; 50-90% well-to-wheel (grid dependent)
- **Cost**: Approaching parity; total cost of ownership (TCO) already favorable in many markets
- **Range**: Current BEVs: 300-600 km; improving annually
- **Charging**: Level 2 (home/work: 7-22 kW); DC fast charge (50-350 kW)
- **Certainty**: Very High (market tipping point passed in 2023-2024)

#### 2. Battery Electric Trucks (BET) -- Medium/Heavy Duty

- **Transition**: Replace diesel trucks with battery-electric trucks
- **Timeline**: Medium-duty (2024-2028); Heavy-duty long-haul (2028-2035)
- **Range**: Medium-duty: 200-400 km today; Heavy-duty: 300-800 km by 2030 (megawatt charging)
- **Reduction**: 100% tailpipe; ~60-85% well-to-wheel
- **Cost**: TCO parity for medium-duty by 2025-2027; heavy-duty by 2028-2030
- **Dependencies**: Megawatt Charging System (MCS) infrastructure; battery energy density
- **Certainty**: High (medium-duty); Medium-High (heavy-duty long-haul)

#### 3. Hydrogen Fuel Cell Vehicles (FCEV) -- Heavy Duty

- **Transition**: Hydrogen fuel cell trucks for long-haul freight
- **Timeline**: Pilot 2024-2027; commercial from 2027-2030
- **Range**: 500-1,000 km
- **Reduction**: 100% tailpipe; depends on hydrogen source (green H2: ~90% WTW reduction)
- **Cost**: Currently 2-3x diesel TCO; parity projected 2030-2035
- **Dependencies**: Green hydrogen infrastructure, refueling stations
- **Certainty**: Medium (competing with BET for long-haul; outcome uncertain)

#### 4. Sustainable Biofuels

- **Transition**: Replace fossil fuels with advanced biofuels (HVO, FAME, bioethanol, biomethane)
- **Timeline**: Available now; scaling up
- **Reduction**: 50-90% lifecycle (depending on feedstock and pathway)
- **Role**: Bridge fuel for existing ICE fleet; important for markets with slow EV adoption
- **Cost**: 1.5-3x conventional fuel
- **Certainty**: High (technology proven; supply constraints)
- **Limitations**: Sustainable feedstock availability limits scale

#### 5. Vehicle Efficiency Improvements (ICE)

- **Transition**: Improved ICE efficiency through hybridization, downsizing, lightweight materials
- **Reduction**: 15-30% vs. current average
- **Timeline**: Ongoing (but diminishing as EV transition accelerates)
- **Cost**: Negative to neutral (fuel savings)
- **Certainty**: High
- **Note**: Increasingly irrelevant as EV transition progresses

#### 6. Modal Shift and Demand Reduction

- **Transition**: Shift from private cars to public transport, cycling, walking; shift freight from road to rail
- **Reduction**: Variable; highly dependent on urban planning and infrastructure
- **Timeline**: Ongoing; requires sustained policy support
- **Cost**: Variable (infrastructure investment vs. congestion/health benefits)
- **Certainty**: Medium (depends on political will and urban planning)

---

## Abatement Levers

### Lever Waterfall (Automotive OEM, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | BEV sales increase (18% to 60%+ of new sales) | 25-35% | -20 to +30 | Very High |
| 2 | PHEV sales increase | 5-8% | 0 to +20 | High |
| 3 | ICE efficiency improvement (remaining ICE) | 5-8% | -10 to +10 | High |
| 4 | Vehicle weight reduction (lightweighting) | 2-4% | +10 to +30 | High |
| 5 | Manufacturing decarbonization | 2-3% | +20 to +60 | Medium |
| 6 | Sustainable material sourcing | 1-2% | +15 to +40 | Medium |

### Lever Waterfall (Road Freight Operator, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Operational efficiency (routing, load optimization) | 8-12% | -30 to -10 | High |
| 2 | Fleet renewal to Euro VI+ compliant trucks | 5-8% | -10 to +10 | High |
| 3 | BET deployment (urban/regional routes) | 10-15% | -10 to +30 | High |
| 4 | Biofuel blending (HVO) for remaining diesel fleet | 5-10% | +20 to +50 | High |
| 5 | Eco-driving training and telematics | 3-5% | -20 to -5 | High |
| 6 | Hydrogen fuel cell pilot (1-2 long-haul routes) | 2-5% | +50 to +150 | Medium |
| 7 | Aerodynamic improvements (trailers) | 2-4% | -5 to +10 | High |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | Global EV sales share reaches 25% of new cars |
| 2025 | No new ICE car models without electrified variant in OECD |
| 2030 | 60% of new car sales globally are electric |
| 2030 | 30% of new truck sales are zero-emission |
| 2030 | 1 million public fast-charging points globally |
| 2035 | All new cars in advanced economies are zero-emission |
| 2035 | 50% of heavy-duty truck sales are zero-emission |
| 2040 | All new cars globally are zero-emission |
| 2040 | No new ICE heavy-duty trucks sold in advanced economies |
| 2050 | Nearly all vehicles on road are zero-emission |
| 2050 | Road transport reaches near-zero emissions |

---

## Benchmarks (2024)

| Benchmark | Value | Unit | Source |
|-----------|-------|------|--------|
| Global average new car (WLTP) | 140 | gCO2/km | ICCT Global Update |
| EU average new car (WLTP) | 107 | gCO2/km | EEA Monitoring |
| EU 2025 target (new cars) | 93.6 | gCO2/km | EU CO2 Standards |
| EU 2030 target (new cars) | 49.5 | gCO2/km | EU CO2 Standards (Fit for 55) |
| EU 2035 target (new cars) | 0 | gCO2/km | EU CO2 Standards (100% ZEV) |
| Sector leader (P10, OEM fleet) | 45 | gCO2/km | CDP Climate 2024 |
| BEV average (tailpipe) | 0 | gCO2/km | By definition |
| BEV well-to-wheel (EU grid) | 40-60 | gCO2/km | JRC Well-to-Wheels |
| BEV well-to-wheel (global avg) | 50-90 | gCO2/km | IEA GEVO |
| Heavy-duty truck average | 80 | gCO2/tkm | ICCT HDV Data |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.convergence_analyzer_engine import ConvergenceAnalyzerEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["C29.10"]})
# Result: road_transport

# Generate pathway for automotive OEM
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="road_transport",
    sub_sector="passenger_vehicles",
    base_year=2023,
    base_year_intensity=120.0,  # gCO2/km (fleet average new sales)
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="s_curve",
    production_forecast={
        2023: 4_500_000,  # vehicles
        2030: 5_000_000,
        2050: 4_800_000,
    },
    region="eu",
    ev_sales_trajectory={
        2023: 0.22, 2025: 0.35, 2030: 0.80, 2035: 1.0
    },
)

print(f"2030 Target: {pathway.target_2030:.0f} gCO2/km")
print(f"2035 Target: {pathway.target_2035:.0f} gCO2/km")

# Convergence analysis
convergence = ConvergenceAnalyzerEngine()
analysis = convergence.analyze(
    sector="road_transport",
    company_trajectory=[
        {"year": 2019, "intensity": 130.0},
        {"year": 2020, "intensity": 128.0},
        {"year": 2021, "intensity": 125.0},
        {"year": 2022, "intensity": 122.0},
        {"year": 2023, "intensity": 120.0},
    ],
    pathway=pathway,
    ev_share_current=0.22,
    ev_share_plan={2025: 0.35, 2030: 0.80},
)

print(f"On track for 2030: {analysis.on_track}")
print(f"Required annual reduction: {analysis.required_annual_reduction:.1f} gCO2/km/yr")
```

---

## Regulatory Context

| Regulation | Relevance to Road Transport |
|-----------|---------------------------|
| EU CO2 Standards for Cars | -55% by 2030 (vs. 2021), -100% by 2035 (zero-emission) |
| EU CO2 Standards for Vans | -50% by 2030, -100% by 2035 |
| EU CO2 Standards for HDVs | -45% by 2030, -65% by 2035, -90% by 2040 |
| US EPA GHG Standards | Increasingly stringent; ~50% EV by 2032 |
| China NEV Mandate | 40% NEV share of new sales by 2030 |
| Norway ICE Ban | 100% ZEV new car sales from 2025 |
| UK ZEV Mandate | 80% ZEV new car sales by 2030, 100% by 2035 |
| India CAFE Standards | 113 gCO2/km by 2023 (fleet average, MIDC) |

---

## Special Considerations

### Well-to-Wheel vs. Tank-to-Wheel

The SDA methodology primarily uses tank-to-wheel (tailpipe) emissions for road transport. However, for a complete climate assessment, well-to-wheel (WTW) analysis is critical:

| Powertrain | Tank-to-Wheel | Well-to-Wheel (EU 2023) | Well-to-Wheel (Coal Grid) |
|-----------|--------------|------------------------|--------------------------|
| Gasoline ICE | 140 gCO2/km | 170 gCO2/km | 170 gCO2/km |
| Diesel ICE | 130 gCO2/km | 165 gCO2/km | 165 gCO2/km |
| BEV (battery) | 0 gCO2/km | 45 gCO2/km | 110 gCO2/km |
| FCEV (green H2) | 0 gCO2/km | 30 gCO2/km | 120 gCO2/km |
| FCEV (grey H2) | 0 gCO2/km | 180 gCO2/km | 250 gCO2/km |

BEVs have a substantial WTW advantage even on moderately carbon-intensive grids. On coal-dominated grids, the advantage is smaller but still significant.

### Embedded Emissions (Scope 3)

Vehicle manufacturing emissions (especially battery production) represent a growing share of lifecycle emissions as tailpipe emissions decline. Battery manufacturing contributes ~40-80 kgCO2e/kWh of battery capacity. For a 60 kWh battery, this is 2.4-4.8 tCO2e per vehicle. Decarbonizing battery supply chains (mining, processing, cell manufacturing) is increasingly important.

---

## References

1. SBTi SDA-Road Transport Methodology, SBTi Target Setting Tool V3.0 (2025)
2. IEA Net Zero by 2050, Chapter 4: Transport (Road)
3. IEA Global EV Outlook 2024
4. IPCC AR6 WGIII, Chapter 10: Transport
5. ICCT Global Update: Light-Duty Vehicle GHG and Fuel Economy Standards (2024)
6. European Environment Agency (EEA) CO2 Emission Performance Standards Monitoring (2024)
7. BloombergNEF EV Outlook 2024

---

**End of Road Transport Sector Guide**
