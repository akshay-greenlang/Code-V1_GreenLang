# Sector Guide: Shipping

**Sector ID:** `shipping`
**SDA Methodology:** SDA-Shipping
**Intensity Metric:** gCO2/tkm (tonne-kilometer)
**IEA Chapter:** Chapter 4 -- Transport (Shipping)

---

## Sector Overview

International shipping carries approximately 80-90% of global trade by volume, transporting around 11 billion tonnes of cargo annually. The sector accounts for approximately 2.5-3% of global CO2 emissions, producing around 900 million tonnes of CO2 per year. Shipping is one of the most carbon-efficient modes of transport on a per-tonne-kilometer basis, but its sheer scale makes it a significant emitter.

Key characteristics of the shipping sector:

1. **International governance**: International shipping is governed by the International Maritime Organization (IMO), which adopted a revised GHG strategy in 2023 targeting net-zero emissions by or around 2050
2. **Long asset lifecycles**: Ships have 25-30 year operational lifespans, making fleet turnover slow and technology lock-in a concern
3. **Fuel diversity**: Heavy fuel oil (HFO) historically dominant; transition underway to LNG, methanol, ammonia, and hydrogen
4. **Operational measures**: Speed reduction ("slow steaming") is the fastest and cheapest emission reduction lever
5. **Energy density requirements**: Long-distance ocean shipping requires high energy density fuels, limiting battery-electric applications to short-sea and harbor operations
6. **Diverse fleet**: From container ships and bulk carriers to tankers, LNG carriers, and roll-on/roll-off vessels, each with different decarbonization challenges

The SBTi SDA-Shipping methodology uses gCO2 per tonne-kilometer (tkm) as the primary convergence metric, aligned with the IMO Carbon Intensity Indicator (CII) framework.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `SHP-01` | Transport work intensity | gCO2/tkm | Primary SDA metric. CO2 per tonne-kilometer of cargo transported |
| `SHP-02` | Annual Efficiency Ratio (AER) | gCO2/dwt-nm | IMO CII metric. CO2 per deadweight tonne-nautical mile |
| `SHP-03` | Energy Efficiency Operational Indicator (EEOI) | gCO2/tkm | IMO operational efficiency metric (actual cargo carried) |
| `SHP-04` | Fuel intensity | g fuel/tkm | Fuel consumption per tonne-kilometer |
| `SHP-05` | Fleet average CII rating | A-E scale | IMO CII rating distribution across fleet |
| `SHP-06` | Zero/low-carbon fuel share | % of energy | Percentage of energy from zero/low-carbon fuels |

### Calculating SHP-01 (Primary Metric)

```python
# Total CO2 from fuel consumption
total_co2_tonnes = sum(
    fuel_consumed_tonnes[fuel] * emission_factor[fuel]
    for fuel in fuels
)
# Emission factors (kgCO2/kg fuel):
# HFO: 3.114, VLSFO: 3.151, MGO/MDO: 3.206, LNG: 2.750, Methanol: 1.375
# Green methanol (biogenic): 0.0, Green ammonia: 0.0, Green hydrogen: 0.0

# Transport work
total_tkm = sum(
    cargo_tonnes[voyage] * distance_nm[voyage] * 1.852  # convert nm to km
    for voyage in voyages
)

intensity_gco2_per_tkm = (total_co2_tonnes * 1_000_000) / total_tkm
```

**Example (Container Shipping Line):**
- Total fuel consumption: 2,000,000 tonnes (90% VLSFO, 10% LNG)
- Total CO2: (1,800,000 * 3.151) + (200,000 * 2.750) = 6,221,800 tCO2
- Total transport work: 500,000,000,000 tkm (500 billion tkm)
- Intensity: (6,221,800 * 1,000,000) / 500,000,000,000 = 12.4 gCO2/tkm

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway

| Year | Intensity (gCO2/tkm) | Reduction from 2020 |
|------|---------------------|-------------------|
| 2020 | 11.0 (global avg) | Baseline |
| 2025 | 9.5 | -14% |
| 2030 | 7.0 | -36% |
| 2035 | 4.5 | -59% |
| 2040 | 2.5 | -77% |
| 2045 | 1.0 | -91% |
| 2050 | 0.3 | -97% |

### Pathway by Ship Type

| Year | Container | Bulk Carrier | Tanker | LNG Carrier |
|------|-----------|-------------|--------|-------------|
| 2020 | 10.0 | 5.0 | 6.0 | 15.0 |
| 2025 | 8.5 | 4.3 | 5.1 | 12.5 |
| 2030 | 6.3 | 3.0 | 3.8 | 9.0 |
| 2035 | 4.0 | 1.8 | 2.3 | 5.5 |
| 2040 | 2.2 | 1.0 | 1.3 | 3.0 |
| 2050 | 0.3 | 0.1 | 0.2 | 0.3 |

---

## Technology Landscape

### Current Fleet and Fuel Mix (2023)

| Fuel Type | Share of Fleet Energy | CO2 Intensity | Status |
|-----------|----------------------|---------------|--------|
| Heavy Fuel Oil (HFO) | 45% | 3.114 kgCO2/kg | Declining, IMO 2020 limits |
| Very Low Sulphur Fuel Oil (VLSFO) | 30% | 3.151 kgCO2/kg | Growing (IMO 2020 compliance) |
| Marine Gas Oil (MGO/MDO) | 10% | 3.206 kgCO2/kg | Stable |
| LNG | 10% | 2.750 kgCO2/kg | Growing rapidly |
| Methanol | 2% | 1.375 kgCO2/kg (fossil) | Rapid growth in newbuilds |
| LPG | 2% | 3.000 kgCO2/kg | Niche |
| Other (ammonia, H2, battery, wind) | <1% | Near-zero | Emerging |

### Key Technology Transitions

#### 1. Alternative Fuels -- Green Methanol

- **Transition**: Dual-fuel engines running on green methanol (from biomass or e-methanol from green H2 + captured CO2)
- **Timeline**: Commercially available now; major newbuild orders (Maersk leading)
- **Reduction**: 65-95% lifecycle (depending on feedstock: bio-methanol vs. e-methanol)
- **Cost**: Green methanol currently 2-3x conventional fuel; declining with scale
- **Infrastructure**: Methanol bunkering available in major ports; global rollout 2025-2030
- **Certainty**: High (proven technology; commercial orders placed)
- **Key advantage**: Drop-in compatible with modified engines; liquid fuel (easy to handle)

#### 2. Alternative Fuels -- Green Ammonia

- **Transition**: Engines burning ammonia (green ammonia from electrolysis + Haber-Bosch)
- **Timeline**: Engine development 2024-2027; first commercial vessels 2027-2030
- **Reduction**: 100% CO2 reduction (zero carbon in fuel)
- **Challenges**: Toxicity, NOx emissions, engine development, bunkering infrastructure
- **Cost**: Projected cost parity with conventional fuel by 2035-2040
- **Certainty**: Medium (engine technology still maturing)
- **Key challenge**: Ammonia is toxic and requires strict safety protocols

#### 3. Wind-Assisted Propulsion

- **Transition**: Rotor sails, wing sails, kites, or suction sails to supplement engine power
- **Timeline**: Commercially available now; scaling 2025-2030
- **Reduction**: 5-30% fuel savings (route and weather dependent)
- **Cost**: EUR 20-60/tCO2e
- **Certainty**: High (proven technology; multiple commercial installations)
- **Best applications**: Bulk carriers and tankers on favorable trade routes

#### 4. Energy Efficiency and Operational Measures

- **Speed optimization**: Reducing speed by 10% saves ~27% fuel (cubic relationship)
- **Hull form optimization**: Advanced hull designs, bulbous bow optimization
- **Air lubrication**: Micro-bubble systems reduce hull friction by 5-10%
- **Waste heat recovery**: Recover heat from exhaust gas for power generation
- **Weather routing**: Optimized routing to minimize fuel consumption
- **Hull coating**: Low-friction coatings reduce drag by 3-8%
- **Combined reduction**: 20-35% through operational and technical measures
- **Cost**: Mostly negative (fuel savings exceed investment)
- **Certainty**: High

#### 5. LNG (Transition Fuel)

- **Transition**: Dual-fuel LNG engines replacing HFO/VLSFO
- **Reduction**: ~20-25% CO2 reduction vs. HFO (tank-to-wake)
- **Methane slip concern**: Upstream and onboard methane leakage can reduce or negate CO2 benefit on a GHG basis
- **Timeline**: Significant fleet share already (10%+); growing with newbuilds
- **Role**: Bridge fuel to 2030-2035; can transition to bio-LNG or synthetic LNG
- **Certainty**: High (technology); Medium (climate benefit due to methane slip)

#### 6. Battery-Electric and Hydrogen (Short-Sea)

- **Transition**: Battery-electric ferries, hydrogen fuel cell vessels for short-sea and harbor operations
- **Timeline**: Battery ferries commercial now (Norway, Denmark); hydrogen vessels 2025-2030
- **Reduction**: 100% direct emissions
- **Range limitation**: Batteries viable for <50 nautical miles; hydrogen for <200 nm
- **Cost**: Battery ferries competitive for short routes; hydrogen still premium
- **Certainty**: High (short-sea battery); Medium (hydrogen vessels)

#### 7. Carbon Capture On Board

- **Transition**: Shipboard carbon capture from exhaust gas
- **Timeline**: Pilot 2024-2027; commercial 2028-2035
- **Reduction**: 70-90% of exhaust CO2
- **Challenges**: Space requirements, energy penalty, CO2 storage and offloading
- **Cost**: EUR 80-150/tCO2e
- **Certainty**: Low-Medium (space constraints on vessels)

---

## Abatement Levers

### Lever Waterfall (Container Shipping Line, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Speed optimization (slow steaming) | 15-20% | -40 to -20 | High |
| 2 | Fleet renewal (dual-fuel methanol/LNG vessels) | 10-15% | +20 to +60 | High |
| 3 | Green methanol procurement (partial fleet) | 5-10% | +60 to +150 | High |
| 4 | Energy efficiency (hull optimization, coatings) | 5-8% | -20 to -5 | High |
| 5 | Wind-assisted propulsion (rotor sails on bulk routes) | 3-5% | +20 to +60 | Medium |
| 6 | Waste heat recovery systems | 2-3% | -10 to +10 | High |

### Lever Waterfall (Bulk Carrier Fleet, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Speed optimization | 15-25% | -40 to -20 | High |
| 2 | Wind-assisted propulsion (rotor sails) | 8-15% | +20 to +60 | Medium |
| 3 | Hull form optimization and air lubrication | 5-8% | -15 to +10 | High |
| 4 | LNG dual-fuel conversion | 5-8% | +30 to +70 | High |
| 5 | Weather routing optimization | 3-5% | -20 to -5 | High |
| 6 | Green ammonia pilot (1-2 vessels) | 2-5% | +80 to +200 | Medium |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | 5% of new ship orders for zero-emission-capable vessels |
| 2025 | IMO CII ratings enforced; D/E rated ships face operational restrictions |
| 2026 | IMO adopts mid-term GHG measures (fuel standard and/or carbon levy) |
| 2030 | 5% of shipping fuel from zero/near-zero-emission sources |
| 2030 | Green methanol and ammonia bunkering available in top 20 ports |
| 2030 | All newbuild ships "zero-emission fuel ready" |
| 2035 | 15% of shipping fuel from zero/near-zero-emission sources |
| 2035 | Wind-assisted propulsion on 10% of applicable fleet |
| 2040 | 30% of shipping fuel from zero/near-zero-emission sources |
| 2050 | Shipping sector achieves net-zero emissions |

---

## Benchmarks (2024)

| Benchmark | Value (gCO2/tkm) | Source |
|-----------|------------------|--------|
| Global average (all ship types) | 11.0 | IMO GHG Study 2024 |
| Container ships (average) | 10.0 | IMO/UNCTAD |
| Bulk carriers (average) | 5.0 | IMO/UNCTAD |
| Tankers (average) | 6.0 | IMO/UNCTAD |
| Sector leader (P10) | 3.5 | CDP Climate 2024 |
| SBTi peer average | 7.5 | SBTi Database 2024 |
| Best-in-class container ships | 5.0 | Company disclosures |
| IMO CII A-rating threshold (container) | <6.5 | IMO MEPC |
| IEA NZE 2025 target | 9.5 | IEA NZE 2050 |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.technology_roadmap_engine import TechnologyRoadmapEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["H50.20"]})
# Result: shipping

# Generate pathway
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="shipping",
    base_year=2023,
    base_year_intensity=12.4,  # gCO2/tkm
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="s_curve",
    production_forecast={
        2023: 500_000_000_000,  # tkm
        2030: 600_000_000_000,
        2050: 750_000_000_000,
    },
    region="global",
)

print(f"2030 Target: {pathway.target_2030:.1f} gCO2/tkm")
print(f"2050 Target: {pathway.target_2050:.1f} gCO2/tkm")

# Technology roadmap
roadmap = TechnologyRoadmapEngine()
tech_result = roadmap.build(
    sector="shipping",
    pathway=pathway,
    current_technology_mix={
        "hfo_vlsfo": 0.75,
        "lng_dual_fuel": 0.15,
        "methanol_dual_fuel": 0.05,
        "mgo": 0.05,
    },
    fleet_size_vessels=250,
    fleet_average_age_years=11.5,
    capex_budget_annual_usd=1_500_000_000,
    region="global",
)

for transition in tech_result.transitions:
    print(f"{transition.name}: {transition.timeline}, {transition.reduction_pct:.0%}")
```

---

## IMO Regulatory Framework

### Current and Upcoming Regulations

| Regulation | Status | Impact |
|-----------|--------|--------|
| **IMO 2020 Sulphur Cap** | In force | 0.5% sulphur limit (fuel quality) |
| **EEDI** (Energy Efficiency Design Index) | In force (Phase 3) | Minimum efficiency for newbuilds |
| **EEXI** (Existing Ship EEI) | In force (2023) | Efficiency requirements for existing ships |
| **CII** (Carbon Intensity Indicator) | In force (2023) | Annual operational rating A-E; corrective action for D/E |
| **IMO GHG Strategy (2023 Revision)** | Adopted | Net-zero by or around 2050; 20% zero/near-zero fuels by 2030, 80% by 2040 |
| **Mid-Term Measures** (GHG fuel standard / levy) | Under negotiation (2025-2026) | Economic measures to incentivize clean fuels |

### CII Rating Framework

| Rating | Description | Consequence |
|--------|-------------|-------------|
| A | Superior performance | Incentives (port fee reductions) |
| B | Good performance | No action required |
| C | Moderate performance | Required improvement plan |
| D | Poor performance | Corrective action plan mandatory |
| E | Inferior performance | Operational restrictions possible |

The CII boundaries tighten annually by 2%, meaning ships must continuously improve to maintain their rating. This creates ongoing pressure for efficiency improvements and fuel switching.

---

## References

1. SBTi SDA-Shipping Methodology, SBTi Target Setting Tool V3.0 (2025)
2. IEA Net Zero by 2050, Chapter 4: Transport (Shipping)
3. IMO Fourth GHG Study (2020) and Fifth GHG Study (2024)
4. IMO 2023 Strategy on Reduction of GHG Emissions from Ships (MEPC 80)
5. IPCC AR6 WGIII, Chapter 10: Transport
6. UNCTAD Review of Maritime Transport 2024
7. Global Maritime Forum Annual Progress Report 2024

---

**End of Shipping Sector Guide**
