# Sector Guide: Buildings (Residential & Commercial)

**Sector ID:** `buildings_residential` / `buildings_commercial`
**SDA Methodology:** SDA-Buildings
**Intensity Metric:** kgCO2/m2/year
**IEA Chapter:** Chapter 6 -- Buildings

---

## Sector Overview

The buildings sector accounts for approximately 28% of global energy-related CO2 emissions when including both direct emissions (on-site fuel combustion for space heating, water heating, cooking) and indirect emissions (purchased electricity for lighting, cooling, appliances). This makes buildings the single largest end-use sector by energy consumption, consuming approximately 30% of global final energy.

The global building stock is approximately 235 billion m2 of floor area, split roughly 75% residential and 25% commercial. This stock is expected to grow by 50% by 2050, primarily in developing economies, adding the equivalent of New York City's entire building stock every month.

Key characteristics:

1. **Long building lifespans**: Buildings last 50-100+ years, creating long-term lock-in of energy performance
2. **Dual emission sources**: Direct (on-site fuel) and indirect (purchased electricity and heat)
3. **Heating dominance**: Space and water heating account for ~50-60% of building energy use in cold climates
4. **Electrification pathway**: Heat pumps are the primary technology for eliminating direct emissions
5. **Envelope efficiency**: Building insulation and passive design can reduce energy demand by 50-80%
6. **Distributed ownership**: Millions of building owners make coordinated action challenging
7. **Regional variation**: Heating-dominant (northern climates) vs. cooling-dominant (tropical/subtropical) regions

The SBTi SDA-Buildings methodology uses kgCO2 per square meter per year (kgCO2/m2/yr) as the convergence metric, separately for residential and commercial buildings.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `BLD-01` | Residential building intensity | kgCO2/m2/year | Primary SDA metric for residential buildings |
| `BLD-02` | Commercial building intensity | kgCO2/m2/year | Primary SDA metric for commercial buildings |
| `BLD-03` | Heating energy intensity | kWh/m2/year | Energy for space heating per unit floor area |
| `BLD-04` | Cooling energy intensity | kWh/m2/year | Energy for space cooling per unit floor area |
| `BLD-05` | Total energy use intensity (EUI) | kWh/m2/year | Total energy consumption per unit floor area |
| `BLD-06` | On-site fossil fuel share | % of total energy | Share of energy from on-site fossil fuel combustion |
| `BLD-07` | Renewable energy share | % of total energy | On-site + purchased renewable energy share |

### Calculating BLD-01 / BLD-02 (Primary Metrics)

```python
# Direct emissions (Scope 1)
direct_co2_tonnes = sum(
    fuel_consumed[fuel] * emission_factor[fuel]
    for fuel in on_site_fuels
)
# Common fuels: natural gas (2.02 kgCO2/m3), heating oil (2.68 kgCO2/litre),
# LPG (1.51 kgCO2/litre), coal (varies)

# Indirect emissions (Scope 2)
indirect_co2_tonnes = purchased_electricity_kwh * grid_ef_kgco2_per_kwh / 1000
indirect_co2_tonnes += purchased_heat_kwh * district_heat_ef / 1000

total_co2_tonnes = direct_co2_tonnes + indirect_co2_tonnes
total_floor_area_m2 = sum(building_floor_area_m2 for building in portfolio)

intensity_kgco2_per_m2_yr = (total_co2_tonnes * 1000) / total_floor_area_m2
```

**Example (Commercial Real Estate Portfolio):**
- Total floor area: 2,000,000 m2 across 50 office buildings
- Natural gas (heating): 40,000,000 m3 * 2.02 = 80,800 tCO2
- Purchased electricity: 300,000 MWh at 300 gCO2/kWh = 90,000 tCO2
- Total CO2: 170,800 tCO2
- Intensity: (170,800 * 1000) / 2,000,000 = 85.4 kgCO2/m2/yr

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway (Residential)

| Year | Intensity (kgCO2/m2/yr) | Reduction from 2020 |
|------|------------------------|-------------------|
| 2020 | 30 (global avg) | Baseline |
| 2025 | 25 | -17% |
| 2030 | 18 | -40% |
| 2035 | 10 | -67% |
| 2040 | 5 | -83% |
| 2045 | 2 | -93% |
| 2050 | 0.5 | -98% |

### NZE 1.5C Convergence Pathway (Commercial)

| Year | Intensity (kgCO2/m2/yr) | Reduction from 2020 |
|------|------------------------|-------------------|
| 2020 | 55 (global avg) | Baseline |
| 2025 | 45 | -18% |
| 2030 | 30 | -45% |
| 2035 | 18 | -67% |
| 2040 | 8 | -85% |
| 2045 | 3 | -95% |
| 2050 | 0.8 | -99% |

### Regional Pathway Variants (Residential)

| Year | Global | EU | US/Canada | China | India |
|------|--------|-----|-----------|-------|-------|
| 2020 | 30 | 35 | 40 | 25 | 10 |
| 2025 | 25 | 28 | 32 | 20 | 9 |
| 2030 | 18 | 18 | 22 | 14 | 7 |
| 2035 | 10 | 8 | 12 | 8 | 5 |
| 2040 | 5 | 3 | 5 | 4 | 3 |
| 2050 | 0.5 | 0.3 | 0.5 | 0.5 | 0.3 |

**Note:** Regional variation reflects different climate zones, building stock age, heating/cooling balance, and grid carbon intensity. Northern European/North American buildings have higher heating-related emissions; South Asian buildings have lower overall intensity but rapidly growing cooling demand.

---

## Technology Landscape

### Current Building Energy Mix (Global, 2023)

| Energy Source | Share of Building Energy | Status |
|--------------|-------------------------|--------|
| Natural gas (direct) | 22% | Declining in new construction |
| Electricity (grid) | 35% | Growing (electrification) |
| District heating | 8% | Stable; decarbonizing |
| Oil/kerosene (direct) | 5% | Declining |
| Coal (direct) | 4% | Declining rapidly |
| Traditional biomass | 15% | Declining (developing economies) |
| Modern biomass | 3% | Growing |
| Solar thermal | 2% | Growing |
| Heat pumps (electricity) | 6% | Rapid growth |

### Key Technology Transitions

#### 1. Heat Pumps (Space Heating & Water Heating)

- **Transition**: Replace gas/oil boilers with electric heat pumps (air-source, ground-source, or water-source)
- **Efficiency**: COP 3.0-5.0 (delivers 3-5 kWh heat per 1 kWh electricity)
- **Timeline**: Commercially mature now; rapid deployment 2023-2040
- **Reduction**: 50-80% of heating emissions (depending on grid, replaces gas at ~200 gCO2/kWh thermal with electricity at ~100 gCO2/kWh thermal via COP 3)
- **Cost**: Equipment: EUR 8,000-15,000 per residential unit (air-source); payback 5-10 years
- **Global heat pump stock**: ~190 million units (2023); IEA target: 600 million by 2030
- **Certainty**: Very High (commercially proven, cost-effective)
- **Key policy**: EU ban on fossil fuel boilers in new buildings from 2025; phase-out in existing buildings varies by country

#### 2. Building Envelope Improvements

- **Transition**: Deep renovation of existing building envelopes (insulation, windows, airtightness)
- **Reduction**: 40-80% of heating energy demand per building
- **Current renovation rate**: ~1% per year globally (EU average ~1.2%)
- **Required renovation rate**: 2-3% per year to meet 2050 targets
- **Cost**: EUR 150-400/m2 for deep renovation
- **Timeline**: Continuous; must accelerate dramatically
- **Certainty**: Very High (proven; barrier is scale and cost)
- **Standards**: Passive House (~15 kWh/m2/yr heating); Nearly Zero-Energy Buildings (nZEB)

#### 3. Renewable Electricity for Buildings

- **Transition**: On-site solar PV, green electricity procurement, community solar
- **Reduction**: Reduces Scope 2 to near-zero
- **Cost**: Rooftop solar PV LCOE: EUR 0.04-0.08/kWh (competitive)
- **Potential**: 25-40% of building electricity demand from on-site solar PV
- **Timeline**: Immediate deployment
- **Certainty**: Very High

#### 4. Smart Building Technologies

- **Transition**: Building management systems (BMS), smart thermostats, demand response, LED lighting
- **Reduction**: 10-30% of total building energy use
- **Cost**: Often negative (energy savings exceed investment)
- **Timeline**: Rapid deployment now
- **Certainty**: Very High
- **Measures**: Occupancy sensors, daylight harvesting, HVAC optimization, predictive maintenance

#### 5. District Heating Decarbonization

- **Transition**: Convert district heating from fossil fuels to:
  - Large-scale heat pumps (waste heat, geothermal, ambient)
  - Biomass/biogas
  - Solar thermal
  - Industrial waste heat
  - Geothermal
- **Reduction**: 60-100% of district heating emissions
- **Timeline**: Major transition 2025-2040
- **Cost**: EUR 30-80/tCO2e
- **Certainty**: High (multiple proven technologies)
- **Relevance**: Significant in Nordic countries, Eastern Europe, China

#### 6. Efficient Cooling Technologies

- **Transition**: High-efficiency air conditioning, passive cooling design, cool roofs, thermal storage
- **Context**: Cooling demand growing rapidly (3x increase expected by 2050)
- **Reduction**: 40-60% of cooling energy per building
- **Cost**: Variable (passive design negative; efficient AC slight premium)
- **Timeline**: Critical in rapidly urbanizing tropical regions
- **Certainty**: High

#### 7. Zero-Carbon-Ready New Buildings

- **Transition**: All new buildings designed to zero-carbon-ready standard
- **Standard**: Highly insulated, all-electric, solar PV, no on-site fossil fuels
- **IEA target**: 100% of new buildings zero-carbon-ready by 2030
- **Cost**: 0-5% construction cost premium (declining with scale)
- **Certainty**: Very High (codes and standards driving adoption)

---

## Abatement Levers

### Lever Waterfall (Commercial Real Estate Portfolio, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Renewable electricity procurement (100% green) | 25-35% | 0 to +15 | Very High |
| 2 | LED lighting and smart controls retrofit | 8-12% | -30 to -10 | Very High |
| 3 | Heat pump installation (replace gas boilers) | 15-20% | +20 to +60 | High |
| 4 | Building envelope improvement (priority buildings) | 10-15% | +40 to +120 | High |
| 5 | Smart BMS and HVAC optimization | 5-10% | -20 to +10 | High |
| 6 | On-site solar PV (suitable rooftops) | 5-8% | -10 to +10 | High |
| 7 | Efficient cooling upgrades | 3-5% | -10 to +20 | High |

### Lever Waterfall (Residential Housing Provider, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Heat pump deployment (gas boiler replacement) | 25-35% | +20 to +60 | High |
| 2 | Building insulation (wall, roof, floor) | 15-25% | +40 to +100 | High |
| 3 | Window upgrades (double to triple glazing) | 5-10% | +50 to +120 | High |
| 4 | Green electricity procurement | 10-15% | 0 to +15 | Very High |
| 5 | Smart thermostats and controls | 5-8% | -20 to -5 | High |
| 6 | Solar PV and battery (suitable buildings) | 3-5% | -5 to +15 | High |
| 7 | Hot water heat pumps | 3-5% | +15 to +40 | High |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | All new buildings in OECD designed to zero-carbon-ready standard |
| 2025 | No new fossil fuel boiler sales in advanced economies (heat pumps only) |
| 2025 | LED share of lighting sales reaches 90% globally |
| 2030 | Building renovation rates reach 2.5% per year in EU |
| 2030 | 600 million heat pumps installed globally |
| 2030 | 100% of new buildings globally are zero-carbon-ready |
| 2035 | All buildings in advanced economies use zero-carbon heating |
| 2035 | Average building EUI reduced by 40% from 2020 |
| 2040 | 80% of existing building stock retrofitted or replaced |
| 2050 | Buildings sector reaches net-zero operational emissions |

---

## Benchmarks (2024)

### Residential Buildings

| Benchmark | Value (kgCO2/m2/yr) | Source |
|-----------|---------------------|--------|
| Global average | 30 | IEA Buildings Report |
| EU average | 35 | EU Buildings Observatory |
| US average | 40 | EIA RECS |
| Passive House standard | 2-5 | Passive House Institute |
| Sector leader (P10) | 5 | CDP/GRESB 2024 |
| SBTi peer average | 22 | SBTi Database 2024 |
| IEA NZE 2025 target | 25 | IEA NZE 2050 |

### Commercial Buildings

| Benchmark | Value (kgCO2/m2/yr) | Source |
|-----------|---------------------|--------|
| Global average | 55 | IEA Buildings Report |
| EU average (office) | 50 | EU Buildings Observatory |
| US average (office) | 65 | ENERGY STAR Portfolio Manager |
| LEED Platinum (typical) | 20-30 | USGBC |
| Net-zero energy building | 5-10 | Various |
| Sector leader (P10) | 15 | GRESB 2024 |
| SBTi peer average | 40 | SBTi Database 2024 |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.abatement_waterfall_engine import AbatementWaterfallEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["L68.20"]})
# Result: buildings_commercial

# Generate pathway
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="buildings_commercial",
    base_year=2023,
    base_year_intensity=85.4,  # kgCO2/m2/yr
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="linear",
    production_forecast={
        2023: 2_000_000,  # m2
        2030: 2_200_000,
        2050: 2_500_000,
    },
    region="eu",
    building_type="office",
    climate_zone="temperate",
)

print(f"2030 Target: {pathway.target_2030:.1f} kgCO2/m2/yr")
print(f"2050 Target: {pathway.target_2050:.1f} kgCO2/m2/yr")

# Abatement waterfall
waterfall = AbatementWaterfallEngine()
abatement = waterfall.analyze(
    sector="buildings_commercial",
    base_year=2023,
    base_year_intensity=85.4,
    target_year=2030,
    target_intensity=30.0,
    floor_area_m2=2_000_000,
    available_levers=[
        "green_electricity_ppa",
        "led_lighting_smart_controls",
        "heat_pump_retrofit",
        "envelope_improvement",
        "bms_optimization",
        "onsite_solar_pv",
    ],
    current_energy_mix={
        "natural_gas": 0.45,
        "grid_electricity": 0.50,
        "district_heating": 0.05,
    },
)

for lever in abatement.levers:
    print(f"{lever.name}: -{lever.reduction_pct:.0%}, {lever.cost_eur_per_tco2e:.0f} EUR/tCO2e")
```

---

## Building Certification and Rating Systems

| System | Scope | Relevance |
|--------|-------|-----------|
| LEED (Leadership in Energy and Environmental Design) | Global | Energy performance, materials, indoor environment |
| BREEAM (Building Research Establishment Environmental Assessment Method) | Global (UK origin) | Comprehensive sustainability assessment |
| GRESB (Global Real Estate Sustainability Benchmark) | Global (investors) | Portfolio-level ESG benchmarking for real estate |
| Energy Performance Certificates (EPCs) | EU | Mandatory energy rating A-G for buildings |
| ENERGY STAR | US | Portfolio Manager for commercial building benchmarking |
| Passive House / Passivhaus | Global | Ultra-low energy demand standard |
| DGNB | Germany/Europe | Comprehensive sustainability certification |
| NABERS | Australia | National building energy rating scheme |
| CRREM (Carbon Risk Real Estate Monitor) | Global | Stranding risk assessment for real estate portfolios |

PACK-028 integrates with CRREM pathways for stranding risk analysis, allowing real estate investors to assess which buildings in their portfolio are at risk of becoming "stranded assets" due to tightening energy performance regulations.

---

## Regulatory Context

| Regulation | Relevance to Buildings |
|-----------|----------------------|
| EU Energy Performance of Buildings Directive (EPBD) recast | Zero-emission buildings for new construction (2028 public, 2030 all); MEPS for worst-performing buildings |
| EU ETS 2 (ETS for buildings and road transport) | Carbon pricing for building heating fuels from 2027 |
| EU Renovation Wave | Target 35 million buildings renovated by 2030 |
| UK Future Homes Standard | New homes 75-80% less carbon than current standard from 2025 |
| US IRA (Inflation Reduction Act) | Tax credits for heat pumps, insulation, solar PV, energy efficiency |
| NYC Local Law 97 | Carbon emission limits for large buildings (>25,000 sq ft) |
| Japan ZEH/ZEB targets | Net-zero energy in new homes by 2030 |
| China Green Building Codes | Mandatory energy efficiency standards for new buildings |

---

## References

1. SBTi SDA-Buildings Methodology (Residential and Commercial), SBTi Target Setting Tool V3.0 (2025)
2. IEA Net Zero by 2050, Chapter 6: Buildings
3. IPCC AR6 WGIII, Chapter 9: Buildings
4. IEA Tracking Clean Energy Progress: Buildings (2024)
5. Global Alliance for Buildings and Construction (GlobalABC) Global Status Report 2024
6. CRREM Carbon Risk Real Estate Monitor: Decarbonisation Pathways (2024 update)
7. EU Buildings Observatory (2024 data)

---

**End of Buildings Sector Guide**
