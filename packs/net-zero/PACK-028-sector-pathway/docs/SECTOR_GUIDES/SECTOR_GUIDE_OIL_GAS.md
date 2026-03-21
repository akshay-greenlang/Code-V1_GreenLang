# Sector Guide: Oil & Gas

**Sector ID:** `oil_gas`
**SDA Methodology:** Extended (IEA-based)
**Intensity Metric:** kgCO2e/boe (barrel of oil equivalent) or tCO2e/TJ
**IEA Chapter:** Chapter 2 -- Energy Supply (Oil & Gas)

---

## Sector Overview

The oil and gas sector is central to the global energy transition debate. The sector's own operational emissions (Scope 1+2) account for approximately 5% of global energy-related CO2 emissions (~2.5 Gt CO2e/year), but the combustion of its products by end-users (Scope 3 Category 11: Use of Sold Products) represents approximately 40% of all global CO2 emissions (~15 Gt CO2/year).

Key characteristics:

1. **Scope 3 dominated**: Product combustion (Scope 3) is 5-10x larger than operational emissions (Scope 1+2)
2. **Methane emissions**: Fugitive methane from production, processing, and transport is a significant near-term climate issue (CH4 GWP-100 = 29.8 for fossil sources)
3. **Flaring and venting**: Routine flaring and venting of associated gas wastes energy and emits CO2/CH4
4. **Energy transition risk**: Long-term demand for oil and gas declines in all IEA scenarios, creating stranded asset risk
5. **Diversification**: Many oil and gas companies are investing in renewables, hydrogen, CCS, and low-carbon fuels
6. **SBTi position**: SBTi has developed specific guidance for oil and gas companies, requiring Scope 3 targets

The SBTi currently requires oil and gas companies to set targets covering Scope 1, 2, and 3. PACK-028 implements the IEA NZE pathway for oil and gas operational emissions and provides Scope 3 product-based pathway modeling.

**Note on SDA methodology**: Oil and gas is not a traditional SDA sector because the SDA approach applies to sectors where the same product (e.g., steel, cement) is made by many producers who converge to a common intensity. Oil and gas companies produce a commodity with relatively uniform combustion emission factors. Instead, PACK-028 uses IEA NZE-aligned absolute and intensity pathways.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `OG-01` | Upstream intensity | kgCO2e/boe produced | Scope 1+2 per barrel of oil equivalent produced |
| `OG-02` | Methane intensity | % CH4/total gas | Methane emissions as percentage of total gas produced |
| `OG-03` | Flaring intensity | m3 gas flared/boe | Gas flared per unit of oil production |
| `OG-04` | Refining intensity | kgCO2e/boe refined | Scope 1+2 per barrel refined |
| `OG-05` | LNG intensity | kgCO2e/tonne LNG | Scope 1+2 per tonne of LNG produced |
| `OG-06` | Product carbon intensity | gCO2e/MJ | Well-to-burn emission intensity of products |
| `OG-07` | Portfolio emission intensity | tCO2e/TJ | Total Scope 1+2+3 per unit of energy sold |

### Calculating OG-01 (Primary Upstream Metric)

```python
# Upstream Scope 1 emissions
upstream_scope1 = (
    production_combustion_tco2e +       # Power generation for operations
    flaring_tco2e +                      # Routine and safety flaring
    venting_ch4_tco2e +                  # Methane venting (CH4 * 29.8)
    fugitive_ch4_tco2e +                 # Fugitive methane leaks (CH4 * 29.8)
    process_co2_tco2e                    # Acid gas removal, etc.
)

# Upstream Scope 2 emissions
upstream_scope2 = purchased_electricity_tco2e + purchased_heat_tco2e

# Production
total_production_boe = oil_production_boe + gas_production_boe + ngl_production_boe

intensity_kgco2e_per_boe = (upstream_scope1 + upstream_scope2) * 1000 / total_production_boe
```

**Example (Upstream Oil & Gas Producer):**
- Upstream Scope 1: 5,000,000 tCO2e (combustion, flaring, fugitives)
- Upstream Scope 2: 500,000 tCO2e
- Total production: 200,000,000 boe
- Intensity: (5,500,000 * 1000) / 200,000,000 = 27.5 kgCO2e/boe

---

## IEA NZE Pathway

### NZE 1.5C Pathway (Upstream Intensity)

| Year | Intensity (kgCO2e/boe) | Reduction from 2020 |
|------|----------------------|-------------------|
| 2020 | 30 (global avg) | Baseline |
| 2025 | 22 | -27% |
| 2030 | 15 | -50% |
| 2035 | 10 | -67% |
| 2040 | 6 | -80% |
| 2045 | 3 | -90% |
| 2050 | 1.5 | -95% |

### NZE Production Volume Pathway

| Year | Oil (Mb/d) | Gas (bcm/yr) | Change from 2020 |
|------|------------|-------------|-------------------|
| 2020 | 90 | 3,950 | Baseline |
| 2025 | 88 | 4,050 | -2% oil, +3% gas |
| 2030 | 72 | 3,600 | -20% oil, -9% gas |
| 2035 | 55 | 3,000 | -39% oil, -24% gas |
| 2040 | 40 | 2,300 | -56% oil, -42% gas |
| 2050 | 24 | 1,200 | -73% oil, -70% gas |

**Critical implication**: In the NZE scenario, oil and gas production volumes decline dramatically. Companies must manage declining production while reducing operational intensity.

### Methane Reduction Pathway

| Year | Target Methane Intensity | Key Action |
|------|------------------------|------------|
| 2020 | 2.0% (global avg) | Baseline |
| 2025 | 0.5% | LDAR programs, equipment upgrades |
| 2030 | 0.2% | Near-zero methane by 2030 (IEA/OGMP pledge) |
| 2050 | <0.1% | Comprehensive methane elimination |

---

## Technology Landscape

### Key Decarbonization Levers (Scope 1+2)

#### 1. Methane Emission Reduction

- **Transition**: Leak Detection and Repair (LDAR), replace pneumatic devices, eliminate venting, upgrade compressors
- **Reduction**: 60-80% of methane emissions (which are 30-50% of total upstream Scope 1)
- **Cost**: Often negative (captured gas has commercial value)
- **Timeline**: Immediate; IEA estimates 75% of methane reductions cost <$0
- **Certainty**: Very High
- **Regulatory**: US EPA methane rule, EU Methane Regulation, OGMP 2.0

#### 2. Flaring Reduction and Elimination

- **Transition**: Capture and utilize associated gas instead of flaring; eliminate routine flaring
- **Reduction**: 5-15% of Scope 1
- **Cost**: Often negative (gas sales revenue)
- **Timeline**: World Bank Zero Routine Flaring by 2030 initiative
- **Certainty**: High
- **Note**: Routine flaring wastes ~150 bcm of gas annually

#### 3. Electrification of Operations

- **Transition**: Replace gas turbines and diesel generators with grid electricity (preferably renewable)
- **Examples**: Electrification of offshore platforms (Norway model), electric compressors, electric drive pumps
- **Reduction**: 15-30% of operational combustion emissions
- **Cost**: EUR 30-80/tCO2e (depends on infrastructure)
- **Timeline**: Ongoing; Norway platforms largely electrified by 2030
- **Certainty**: High

#### 4. Renewable Energy for Operations

- **Transition**: On-site solar PV and wind for operational power; green electricity PPAs
- **Reduction**: 10-25% of Scope 2 and partial Scope 1
- **Cost**: Often competitive or negative
- **Timeline**: Immediate
- **Certainty**: Very High

#### 5. Carbon Capture, Utilization and Storage (CCUS)

- **Transition**: CCS on refinery and processing plant emissions; CCS on hydrogen production (blue hydrogen)
- **Reduction**: 40-90% of point-source emissions
- **Cost**: EUR 30-80/tCO2e (processing plants, high-purity CO2); EUR 60-120/tCO2e (refinery FCC, dilute)
- **Timeline**: Several large-scale projects operational; scaling 2025-2035
- **Certainty**: Medium-High (technology proven; cost and storage availability are barriers)

#### 6. Energy Efficiency in Refining

- **Transition**: Process optimization, heat integration, waste heat recovery, efficient equipment
- **Reduction**: 10-20% of refining emissions
- **Cost**: Negative (energy savings)
- **Timeline**: Continuous improvement
- **Certainty**: High

---

## Abatement Levers

### Lever Waterfall (Integrated Oil & Gas Company, 2023-2030)

| # | Lever | Reduction (Scope 1+2) | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------------------|-----------------|-----------|
| 1 | Methane LDAR and equipment upgrades | 15-25% | -30 to 0 | Very High |
| 2 | Flaring elimination | 5-10% | -20 to 0 | High |
| 3 | Electrification of operations | 10-15% | +30 to +80 | High |
| 4 | Renewable electricity for operations | 8-12% | 0 to +15 | Very High |
| 5 | Energy efficiency in refining | 5-10% | -15 to -5 | High |
| 6 | CCS on refinery/processing (pilot) | 5-8% | +40 to +100 | Medium |

### Lever Waterfall (Upstream E&P, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Methane emission reduction (LDAR, equipment) | 25-40% | -30 to 0 | Very High |
| 2 | Flaring elimination (zero routine flaring) | 10-15% | -20 to 0 | High |
| 3 | Platform/facility electrification | 15-20% | +30 to +80 | High |
| 4 | On-site renewable power | 5-10% | -5 to +15 | High |
| 5 | Operational efficiency | 5-8% | -15 to -5 | High |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | 75% reduction in methane emissions from oil and gas operations |
| 2025 | No new oil and gas exploration or development approvals (NZE scenario) |
| 2030 | Zero routine flaring globally |
| 2030 | All oil and gas operations achieving near-zero methane (<0.2%) |
| 2030 | CCS deployed on all new gas processing facilities |
| 2035 | All offshore platforms electrified (advanced economies) |
| 2040 | Oil demand at 40 Mb/d (56% below 2020) |
| 2050 | Oil demand at 24 Mb/d (73% below 2020); gas at 1,200 bcm (70% below 2020) |
| 2050 | Remaining oil and gas production at near-zero operational intensity |

---

## Benchmarks (2024)

| Benchmark | Value | Unit | Source |
|-----------|-------|------|--------|
| Global average upstream intensity | 30 | kgCO2e/boe | IOGP/IEA |
| OECD average upstream | 22 | kgCO2e/boe | IOGP |
| Sector leader (P10, upstream) | 10 | kgCO2e/boe | CDP Climate 2024 |
| Global average methane intensity | 2.0% | CH4/gas | IEA Methane Tracker |
| Best practice methane | 0.1% | CH4/gas | OGMP Gold Standard |
| Global average refining intensity | 25 | kgCO2e/boe | Solomon Associates |
| Sector leader (P10, refining) | 15 | kgCO2e/boe | CDP Climate 2024 |
| SBTi peer average | 20 | kgCO2e/boe | SBTi Database 2024 |
| IEA NZE 2025 target | 22 | kgCO2e/boe | IEA NZE 2050 |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.scenario_comparison_engine import ScenarioComparisonEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["B06.10"]})
# Result: oil_gas

# Generate pathway
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="oil_gas",
    sub_sector="upstream",
    base_year=2023,
    base_year_intensity=27.5,  # kgCO2e/boe
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="exponential",
    production_forecast={
        2023: 200_000_000,  # boe
        2030: 180_000_000,  # declining production in NZE
        2040: 120_000_000,
        2050: 80_000_000,
    },
    region="global",
    methane_intensity_current=0.015,  # 1.5%
)

print(f"2030 Target: {pathway.target_2030:.1f} kgCO2e/boe")
print(f"2050 Target: {pathway.target_2050:.1f} kgCO2e/boe")

# Scenario comparison (critical for oil & gas strategic planning)
scenario_engine = ScenarioComparisonEngine()
comparison = scenario_engine.compare(
    sector="oil_gas",
    base_year=2023,
    base_year_intensity=27.5,
    scenarios=["nze_15c", "wb2c", "2c", "aps", "steps"],
    production_forecast_by_scenario={
        "nze_15c": {2030: 180_000_000, 2050: 80_000_000},
        "wb2c": {2030: 190_000_000, 2050: 120_000_000},
        "2c": {2030: 195_000_000, 2050: 140_000_000},
        "aps": {2030: 195_000_000, 2050: 150_000_000},
        "steps": {2030: 200_000_000, 2050: 200_000_000},
    },
)

for scenario in comparison.scenarios:
    print(f"{scenario.name}:")
    print(f"  2030 intensity: {scenario.target_2030:.1f} kgCO2e/boe")
    print(f"  2050 production: {scenario.production_2050:,.0f} boe")
    print(f"  Stranded asset risk: {scenario.stranded_asset_risk}")
```

---

## Special Considerations

### Scope 3 Category 11 (Use of Sold Products)

For oil and gas companies, Scope 3 Category 11 is by far the largest emission category, typically 5-10x Scope 1+2. Example:

| Scope | Typical Share | Source |
|-------|-------------|--------|
| Scope 1 | 8-12% | Operational combustion, flaring, fugitives |
| Scope 2 | 1-3% | Purchased electricity |
| Scope 3 Cat 11 | 80-90% | Combustion of sold products by customers |
| Scope 3 Other | 2-5% | Supply chain, employee travel, etc. |

The SBTi requires oil and gas companies to set Scope 3 targets. In the NZE scenario, this means aligning with declining hydrocarbon demand.

### Transition Strategy and Stranded Assets

Oil and gas companies face fundamental strategic choices:
1. **Managed decline**: Reduce production in line with NZE demand, maximize value from existing assets
2. **Diversification**: Invest in renewables, hydrogen, CCS, EV charging, and other low-carbon businesses
3. **Efficiency focus**: Reduce operational intensity while maintaining production volumes

PACK-028's Scenario Comparison Engine is particularly valuable for oil and gas, as it models these strategic options across different IEA scenarios.

### OGMP 2.0 (Oil & Gas Methane Partnership)

The OGMP 2.0 framework provides a standardized approach to methane measurement, reporting, and verification:
- **Level 1-3**: Source-level measurement (bottom-up)
- **Level 4**: Site-level measurement
- **Level 5**: Reconciliation with satellite and aerial measurements (top-down)

PACK-028 integrates with OGMP 2.0 reporting levels through the MRV Bridge.

---

## Regulatory Context

| Regulation | Relevance to Oil & Gas |
|-----------|----------------------|
| EU Methane Regulation | Mandatory LDAR, methane intensity limits, import requirements |
| US EPA Methane Rule | Standards for new and existing oil and gas sources |
| EU ETS Phase 4 | Covers refining; carbon pricing ~EUR 80-100/tCO2 |
| EU CBAM | Potential future coverage of refined products |
| Norway CO2 Tax | ~EUR 85/tCO2 on offshore petroleum activities |
| IEA NZE Scenario | No new oil/gas development beyond already approved projects |
| Global Methane Pledge | 30% methane reduction by 2030 (signed by 150+ countries) |
| ISSB S2 / SEC Climate | Climate risk disclosure including Scope 3 for fossil fuel companies |

---

## References

1. IEA Net Zero by 2050, Chapter 2: Energy Supply (Oil & Gas)
2. IEA World Energy Outlook 2024 (NZE, APS, STEPS scenarios)
3. IEA Methane Tracker 2024
4. IPCC AR6 WGIII, Chapter 6: Energy Systems
5. IOGP Environmental Performance Indicators 2024
6. OGMP 2.0 Framework Documentation
7. SBTi Oil and Gas Guidance (2024)
8. World Bank Zero Routine Flaring by 2030 Initiative

---

**End of Oil & Gas Sector Guide**
