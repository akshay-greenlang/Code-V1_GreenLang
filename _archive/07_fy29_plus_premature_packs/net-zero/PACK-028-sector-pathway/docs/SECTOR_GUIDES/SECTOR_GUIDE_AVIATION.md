# Sector Guide: Aviation

**Sector ID:** `aviation`
**SDA Methodology:** SDA-Aviation
**Intensity Metric:** gCO2/pkm (passenger) or gCO2/tkm (freight)
**IEA Chapter:** Chapter 4 -- Transport (Aviation)

---

## Sector Overview

The aviation sector accounts for approximately 2.5-3% of global CO2 emissions, producing around 900 million tonnes of CO2 annually (pre-pandemic levels, now recovered and growing). Aviation is considered one of the hardest-to-abate sectors due to the fundamental physics constraints of flight: aircraft require high energy density fuels, and battery-electric propulsion is limited to short-range applications due to weight constraints.

Key characteristics of aviation decarbonization:

1. **Energy density requirements**: Jet fuel has an energy density of ~43 MJ/kg; current batteries achieve ~0.9 MJ/kg, making battery-electric flight viable only for short distances (<500 km) with small aircraft
2. **Long asset lifecycles**: Aircraft have 25-30 year operational lifespans, meaning fleet turnover is slow
3. **International governance**: International aviation emissions are governed by ICAO (International Civil Aviation Organization) through the CORSIA scheme, not national climate policies
4. **Growth pressure**: Demand for air travel is projected to grow 3-4% annually, potentially doubling by 2050
5. **No direct electrification path**: Unlike road transport, aviation cannot be directly electrified for medium/long-haul flights in the foreseeable future

The SBTi SDA-Aviation methodology uses gCO2 per passenger-kilometer (pkm) for passenger aviation and gCO2 per tonne-kilometer (tkm) for freight aviation. The pathway accounts for fleet efficiency improvements and sustainable aviation fuel (SAF) uptake.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `AVI-01` | Passenger intensity | gCO2/pkm | Primary SDA metric for passenger aviation. CO2 per passenger-kilometer |
| `AVI-02` | Freight intensity | gCO2/tkm | Primary SDA metric for freight aviation. CO2 per tonne-kilometer |
| `AVI-03` | Fleet fuel efficiency | L/100pkm | Fuel consumption per 100 passenger-kilometers |
| `AVI-04` | SAF blend rate | % of total fuel | Percentage of total fuel that is sustainable aviation fuel |
| `AVI-05` | Carbon intensity of fuel | gCO2/MJ | Well-to-wake emission intensity of fuel mix |
| `AVI-06` | Load factor adjusted intensity | gCO2/ASK | CO2 per available seat-kilometer (operational metric) |

### Calculating AVI-01 (Primary Metric)

```python
# Passenger intensity
total_co2_kg = total_fuel_consumed_kg * fuel_emission_factor_kgco2_per_kg_fuel
# Standard Jet A-1: 3.16 kgCO2/kg fuel
# SAF lifecycle emissions vary: 0.3-1.5 kgCO2/kg (depends on feedstock)

# Blended fuel emission factor
fuel_ef_blended = (
    (1 - saf_blend_rate) * 3.16 +  # Conventional jet fuel
    saf_blend_rate * saf_lifecycle_ef   # SAF (lifecycle)
)

total_co2_tonnes = total_fuel_kg * fuel_ef_blended / 1000
passenger_km = total_passengers * average_distance_km
# Or more precisely: sum of (passengers_per_flight * great_circle_distance_km) for all flights

intensity_gco2_per_pkm = (total_co2_tonnes * 1_000_000) / passenger_km
```

**Example (Major Network Airline):**
- Total fuel consumption: 5,000,000 tonnes (all conventional jet fuel)
- Total CO2: 5,000,000 * 3.16 = 15,800,000 tCO2
- Total passenger-km: 180,000,000,000 pkm (180 billion pkm)
- Intensity: (15,800,000 * 1,000,000) / 180,000,000,000 = 87.8 gCO2/pkm

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway (Passenger)

| Year | Intensity (gCO2/pkm) | Reduction from 2020 |
|------|---------------------|-------------------|
| 2020 | 90 (global avg) | Baseline |
| 2025 | 82 | -9% |
| 2030 | 68 | -24% |
| 2035 | 50 | -44% |
| 2040 | 32 | -64% |
| 2045 | 18 | -80% |
| 2050 | 8 | -91% |

### NZE 1.5C Convergence Pathway (Freight)

| Year | Intensity (gCO2/tkm) | Reduction from 2020 |
|------|---------------------|-------------------|
| 2020 | 600 (global avg) | Baseline |
| 2025 | 540 | -10% |
| 2030 | 430 | -28% |
| 2035 | 310 | -48% |
| 2040 | 190 | -68% |
| 2045 | 100 | -83% |
| 2050 | 40 | -93% |

### Decomposition of Pathway (Passenger)

| Lever | Contribution to 2050 Target |
|-------|---------------------------|
| Fleet fuel efficiency improvement | 20-25% of total reduction |
| Sustainable aviation fuel (SAF) | 45-55% of total reduction |
| Hydrogen/electric aircraft (short-haul) | 10-15% of total reduction |
| Operational improvements (ATM, routing) | 5-10% of total reduction |
| Carbon removal offsets (residual) | 5-10% of total reduction |

---

## Technology Landscape

### Current Fleet Efficiency (2023)

| Aircraft Category | Typical Intensity | Example Aircraft |
|------------------|------------------|-----------------|
| Long-haul widebody (new gen) | 60-75 gCO2/pkm | A350-900, B787-9 |
| Long-haul widebody (older gen) | 85-110 gCO2/pkm | A340-300, B777-200 |
| Medium-haul narrowbody (new gen) | 65-80 gCO2/pkm | A320neo, B737 MAX 8 |
| Medium-haul narrowbody (older gen) | 85-100 gCO2/pkm | A320ceo, B737-800 |
| Regional jet | 100-140 gCO2/pkm | E190-E2, CRJ-900 |
| Regional turboprop | 60-90 gCO2/pkm | ATR 72-600, Dash 8-400 |
| Freighter (dedicated) | 500-800 gCO2/tkm | B747-8F, A330-200F |
| Belly cargo (passenger aircraft) | 200-400 gCO2/tkm | Various widebodies |

### Key Technology Transitions

#### 1. Sustainable Aviation Fuel (SAF)

- **Description**: Drop-in fuels produced from sustainable feedstocks that can replace conventional jet fuel without aircraft modification
- **Types**:
  - **HEFA**: Hydroprocessed esters and fatty acids (from used cooking oil, animal fats) -- commercially available now
  - **FT-SPK**: Fischer-Tropsch synthetic paraffinic kerosene (from biomass gasification) -- scaling up
  - **AtJ**: Alcohol-to-Jet (from ethanol/methanol) -- early commercial
  - **PtL/e-fuels**: Power-to-Liquid (green hydrogen + captured CO2) -- pilot phase
- **Lifecycle emission reduction**: 50-95% vs. conventional jet fuel (depending on pathway)
- **Current SAF production**: ~500,000 tonnes/year (~0.15% of total jet fuel demand)
- **Target**: 10% of fuel by 2030 (IATA/EU mandates), 65% by 2050
- **Cost**: 2-5x conventional jet fuel (HEFA at ~2x; PtL at ~4-5x)
- **Maximum blend rate**: Currently approved up to 50% blend (100% SAF flights demonstrated)
- **Critical importance**: SAF is the primary decarbonization lever for aviation (>50% of the pathway)

#### 2. Next-Generation Aircraft

- **Transition**: New aircraft designs with 20-30% fuel efficiency improvement over current generation
- **Timeline**: Entry into service 2035-2040 (next-gen narrowbody and widebody programs)
- **Technologies**:
  - Open rotor / ultra-high bypass turbofan engines
  - Advanced aerodynamics (laminar flow, blended wing body)
  - Lightweight composite structures
  - Active load alleviation
- **Reduction**: 20-30% fuel burn improvement per seat-km
- **Cost**: Aircraft price premium offset by fuel savings
- **Certainty**: Medium-High (evolutionary improvement, not revolutionary)

#### 3. Hydrogen-Powered Aircraft

- **Transition**: Hydrogen combustion or fuel cell propulsion for short/medium-haul flights
- **Timeline**: Demonstrators by 2028; commercial entry 2035-2040 (Airbus ZEROe program)
- **Range**: Initially limited to 1,000-2,000 km
- **Reduction**: Zero CO2 in flight (lifecycle depends on hydrogen source)
- **Dependencies**: Green hydrogen infrastructure at airports, aircraft certification, airport infrastructure
- **Challenges**: Hydrogen has 4x the volume of jet fuel per unit energy (requires larger tanks)
- **Certainty**: Medium (significant engineering and infrastructure challenges)

#### 4. Battery-Electric Aircraft

- **Transition**: Electric propulsion for short-range (<500 km) flights
- **Timeline**: Regional/commuter aircraft 2028-2030; larger aircraft post-2035
- **Reduction**: Zero direct emissions
- **Current limitations**: Battery energy density (~250 Wh/kg vs. ~12,000 Wh/kg for jet fuel)
- **Viable applications**: Regional flights, island routes, air taxis, training aircraft
- **Certainty**: Medium for regional; Low for medium-haul

#### 5. Operational Improvements

- **Air traffic management (ATM) optimization**: More direct routing, continuous descent approaches, formation flying
- **Reduction**: 5-10% of total emissions
- **Timeline**: Continuous (Single European Sky ATM Research -- SESAR; NextGen in US)
- **Cost**: Neutral to negative (fuel savings)
- **Dependencies**: International coordination, airspace modernization
- **Additional measures**: Optimal speed, altitude, and routing; reduced tankering; lightweight operations

#### 6. Carbon Removal and Offsetting

- **CORSIA**: Carbon Offsetting and Reduction Scheme for International Aviation
- **Coverage**: International flights between participating states
- **Phase**: Pilot (2021-2023), Phase 1 (2024-2026, voluntary), Phase 2 (2027+, mandatory for major states)
- **Role in pathway**: Addresses residual emissions that cannot be eliminated by technology

---

## Abatement Levers

### Lever Waterfall (Major Network Airline, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Fleet renewal (retire old, acquire A320neo/A350) | 10-15% | -20 to -10 (net fuel savings) | High |
| 2 | SAF procurement (target 10% blend by 2030) | 8-12% | +80 to +200 | High |
| 3 | Operational efficiency (routing, speed, weight) | 3-5% | -15 to -5 | High |
| 4 | Ground operations electrification | 1-2% | +10 to +30 | High |
| 5 | Load factor optimization | 2-3% | -10 to 0 | High |
| 6 | Contrail avoidance (flight path adjustment) | 0-2% (non-CO2) | +5 to +15 | Medium |

### Lever Waterfall (Regional Airline, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Fleet renewal (turboprop upgrade, E2 jets) | 12-18% | -20 to -5 | High |
| 2 | SAF procurement (target 10% blend) | 8-12% | +80 to +200 | High |
| 3 | Electric/hybrid aircraft pilot (2-3 routes) | 2-5% | +100 to +300 | Low-Medium |
| 4 | Operational efficiency | 3-5% | -15 to -5 | High |
| 5 | Ground operations electrification | 2-3% | +10 to +30 | High |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | SAF production reaches 2 million tonnes/year globally |
| 2025 | All major airlines have committed to SBTi targets |
| 2025 | 100% SAF certified for single-fuel-type use (no blending required) |
| 2030 | SAF accounts for 10% of aviation fuel consumption globally |
| 2030 | Electric/hydrogen aircraft demonstrators certified for passenger service |
| 2030 | CORSIA Phase 2 mandatory for all major aviation states |
| 2035 | Next-generation narrowbody aircraft enters service |
| 2035 | SAF production reaches 100 million tonnes/year |
| 2035 | First hydrogen-powered commercial aircraft in regional service |
| 2040 | SAF accounts for 30% of aviation fuel globally |
| 2040 | Electric aircraft serving routes up to 500 km |
| 2050 | Aviation achieves ~90% reduction in gCO2/pkm vs. 2020 |
| 2050 | SAF accounts for 65%+ of aviation fuel; residual offset by carbon removal |

---

## Benchmarks (2024)

| Benchmark | Value | Unit | Source |
|-----------|-------|------|--------|
| Global average (passenger) | 90 | gCO2/pkm | ICAO Environmental Report 2024 |
| European average (passenger) | 82 | gCO2/pkm | EUROCONTROL |
| US major airlines average | 88 | gCO2/pkm | DOT/Airlines for America |
| Sector leader (P10) | 62 | gCO2/pkm | CDP Climate 2024 |
| SBTi peer average | 78 | gCO2/pkm | SBTi Database 2024 |
| Best-in-class new aircraft | 55-65 | gCO2/pkm | Manufacturer specifications |
| Low-cost carrier average | 70-80 | gCO2/pkm | Airline disclosures |
| Full-service carrier average | 85-100 | gCO2/pkm | Airline disclosures |
| IEA NZE 2025 target | 82 | gCO2/pkm | IEA NZE 2050 |

**Note:** Low-cost carriers (LCCs) typically have lower gCO2/pkm than full-service carriers due to higher load factors (90%+ vs. 80-85%), higher seat density, and newer fleet average age. However, their rapid growth can lead to higher absolute emissions.

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.convergence_analyzer_engine import ConvergenceAnalyzerEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["H51.10"]})
# Result: aviation

# Generate pathway
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="aviation",
    base_year=2023,
    base_year_intensity=87.8,  # gCO2/pkm
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="s_curve",
    production_forecast={
        2023: 180_000_000_000,  # pkm
        2030: 220_000_000_000,
        2040: 280_000_000_000,
        2050: 340_000_000_000,
    },
    region="global",
)

print(f"2030 Target: {pathway.target_2030:.1f} gCO2/pkm")
print(f"2050 Target: {pathway.target_2050:.1f} gCO2/pkm")

# Convergence analysis
convergence = ConvergenceAnalyzerEngine()
analysis = convergence.analyze(
    sector="aviation",
    company_trajectory=[
        {"year": 2019, "intensity": 92.0},
        {"year": 2020, "intensity": 95.0},  # COVID disruption
        {"year": 2021, "intensity": 93.0},
        {"year": 2022, "intensity": 89.5},
        {"year": 2023, "intensity": 87.8},
    ],
    pathway=pathway,
    current_saf_blend=0.005,  # 0.5%
    fleet_average_age_years=12.5,
    fleet_renewal_plan={
        "2024-2026": {"retirements": 15, "deliveries": 25, "type": "A320neo/A350"},
        "2027-2030": {"retirements": 20, "deliveries": 30, "type": "A321XLR/next-gen"},
    },
)

print(f"On track: {analysis.on_track}")
print(f"Gap to 2030: {analysis.gap_2030:+.1f} gCO2/pkm")
print(f"Key risk: {analysis.top_risk}")
```

---

## Non-CO2 Climate Effects

Aviation has significant non-CO2 climate effects that are not captured in the SDA pathway but are increasingly recognized:

| Effect | Warming Impact (vs. CO2 only) | Certainty |
|--------|------------------------------|-----------|
| Contrails and contrail cirrus | 1.5-3.0x multiplier | Medium |
| NOx effects (net warming) | 0.3-0.5x multiplier | Medium |
| Soot/black carbon | 0.05-0.1x multiplier | Low-Medium |
| Water vapor | Small at current altitudes | Low |
| **Total effective radiative forcing** | **~2.5-3.5x CO2 alone** | **Medium** |

The EU is considering inclusion of non-CO2 effects in its regulatory framework (EU ETS for aviation). PACK-028 can model these effects using a configurable effective radiative forcing (ERF) multiplier, though the SDA pathway itself covers CO2 only.

---

## Regulatory Context

| Regulation | Relevance to Aviation |
|-----------|----------------------|
| ICAO CORSIA | Global carbon offsetting scheme for international aviation |
| EU ETS (Aviation) | Covers all intra-EEA flights + departing flights to non-CORSIA states |
| EU ReFuelEU Aviation | Mandatory SAF blending (2% 2025, 6% 2030, 20% 2035, 70% 2050) |
| UK SAF Mandate | Mandatory SAF blending (10% by 2030) |
| US Inflation Reduction Act | SAF tax credit ($1.25-$1.75/gallon) |
| ICAO LTAG | Long-Term Aspirational Goal: net-zero by 2050 |

---

## References

1. SBTi SDA-Aviation Methodology, SBTi Target Setting Tool V3.0 (2025)
2. IEA Net Zero by 2050, Chapter 4: Transport (Aviation)
3. ICAO Environmental Report 2024
4. IATA Net Zero 2050 Roadmap (updated 2024)
5. IPCC AR6 WGIII, Chapter 10: Transport
6. ATAG Waypoint 2050 (Air Transport Action Group)
7. Lee, D.S. et al. (2021) "The contribution of global aviation to anthropogenic climate forcing for 2000 to 2018," Atmospheric Environment

---

**End of Aviation Sector Guide**
