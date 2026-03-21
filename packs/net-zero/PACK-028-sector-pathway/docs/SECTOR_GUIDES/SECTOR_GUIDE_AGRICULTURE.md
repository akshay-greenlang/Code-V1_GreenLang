# Sector Guide: Agriculture

**Sector ID:** `agriculture`
**SDA Methodology:** Extended (IEA-based, SBTi FLAG guidance)
**Intensity Metric:** tCO2e/tonne product or tCO2e/hectare
**IEA Chapter:** Chapter 7 -- Agriculture & Land Use

---

## Sector Overview

Agriculture is responsible for approximately 10-12% of global greenhouse gas emissions (5.8 Gt CO2e/year), rising to 21-23% when including land use change and forestry (AFOLU -- Agriculture, Forestry and Other Land Use). Unlike energy-intensive industrial sectors, agriculture's emissions profile is dominated by non-CO2 greenhouse gases: methane (CH4) from enteric fermentation and rice cultivation, and nitrous oxide (N2O) from fertilizer use and manure management.

Key characteristics of agricultural emissions:

1. **Non-CO2 dominated**: ~50% CH4, ~30% N2O, ~20% CO2 (energy use and land conversion)
2. **Biological processes**: Emissions are inherently linked to biological systems (livestock, crops, soils), making elimination impossible -- only reduction
3. **Food security nexus**: Emission reduction must not compromise food production for a growing global population
4. **Carbon sequestration potential**: Agricultural soils and agroforestry can sequester carbon, providing negative emissions
5. **Distributed sources**: Hundreds of millions of farms globally, from subsistence to industrial scale
6. **SBTi FLAG guidance**: The SBTi's Forest, Land and Agriculture (FLAG) guidance applies to companies with significant land-based emissions

The SBTi FLAG guidance uses intensity metrics (tCO2e per tonne of product) and absolute reduction targets, with sector-specific pathways for major agricultural commodities. PACK-028 implements the extended IEA-based pathway for agriculture, complementing SBTi FLAG targets.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `AGR-01` | Overall agricultural intensity | tCO2e/tonne product | Primary metric. Total GHG per tonne of agricultural output |
| `AGR-02` | Livestock intensity (beef) | kgCO2e/kg protein or kgCO2e/kg LW | GHG per kg of protein or liveweight for beef |
| `AGR-03` | Livestock intensity (dairy) | kgCO2e/kg FPCM | GHG per kg of fat-and-protein-corrected milk |
| `AGR-04` | Crop intensity | tCO2e/tonne crop | GHG per tonne of crop produced |
| `AGR-05` | Land-based intensity | tCO2e/hectare/year | Total GHG per hectare of managed land |
| `AGR-06` | Fertilizer emission intensity | kgN2O-N/kg N applied | Nitrous oxide per unit of nitrogen fertilizer applied |
| `AGR-07` | Enteric fermentation intensity | kgCH4/head/year | Methane from livestock digestion per animal |
| `AGR-08` | Land use change emissions | tCO2e/hectare/year | CO2 from deforestation and land conversion |

### Calculating AGR-01 (Primary Metric)

```python
# Total GHG emissions (using GWP-100 from IPCC AR6)
# CH4 GWP-100 = 27.9 (fossil) / 27.2 (biogenic, with feedback)
# N2O GWP-100 = 273

total_ghg_tco2e = (
    co2_emissions_tonnes +                           # Energy use, lime, urea
    ch4_emissions_tonnes * 27.2 +                    # Enteric fermentation, rice, manure
    n2o_emissions_tonnes * 273                        # Fertilizer, manure, crop residues
)

total_product_tonnes = sum(
    product_output_tonnes[product] * allocation_factor[product]
    for product in products
)

intensity_tco2e_per_t = total_ghg_tco2e / total_product_tonnes
```

**Example (Mixed Farming Operation -- Dairy):**
- Enteric fermentation CH4: 5,000 tonnes CH4 = 136,000 tCO2e
- Manure management CH4: 800 tonnes CH4 = 21,760 tCO2e
- Manure management N2O: 50 tonnes N2O = 13,650 tCO2e
- Fertilizer N2O: 120 tonnes N2O = 32,760 tCO2e
- Energy CO2 (farm machinery, heating): 15,000 tCO2
- Total: 219,170 tCO2e
- Milk production: 200,000 tonnes FPCM
- Intensity: 219,170 / 200,000 = 1.10 kgCO2e/kg FPCM

---

## SBTi FLAG / IEA Pathway

### NZE Convergence Pathway (Overall Agriculture)

| Year | Reduction from 2020 (%) | Key Actions |
|------|------------------------|-------------|
| 2020 | Baseline | Current practices |
| 2025 | -8% | Improved fertilizer management, feed additives |
| 2030 | -20% | Precision agriculture, methane inhibitors at scale |
| 2035 | -32% | Agroforestry, soil carbon, advanced breeding |
| 2040 | -42% | Technology maturation, circular agriculture |
| 2045 | -50% | Near-optimized agricultural systems |
| 2050 | -55% to -60% | Residual emissions remain (biogenic CH4/N2O) |

**Note:** Agriculture cannot achieve zero emissions because biological processes inherently produce CH4 and N2O. The IEA NZE scenario assumes agriculture reduces emissions by ~55-60% by 2050, with remaining emissions offset by carbon removal (BECCS, DAC, afforestation).

### Commodity-Specific Pathways

#### Beef

| Year | Intensity (kgCO2e/kg protein) | Reduction |
|------|------------------------------|-----------|
| 2020 | 300 (global avg) | Baseline |
| 2030 | 230 | -23% |
| 2040 | 170 | -43% |
| 2050 | 130 | -57% |

#### Dairy

| Year | Intensity (kgCO2e/kg FPCM) | Reduction |
|------|---------------------------|-----------|
| 2020 | 1.30 (global avg) | Baseline |
| 2030 | 1.00 | -23% |
| 2040 | 0.75 | -42% |
| 2050 | 0.60 | -54% |

#### Rice

| Year | Intensity (kgCO2e/kg rice) | Reduction |
|------|--------------------------|-----------|
| 2020 | 1.80 (global avg, paddy) | Baseline |
| 2030 | 1.30 | -28% |
| 2040 | 0.90 | -50% |
| 2050 | 0.65 | -64% |

#### Row Crops (Wheat, Corn, Soy)

| Year | Intensity (kgCO2e/kg) | Reduction |
|------|----------------------|-----------|
| 2020 | 0.40 (global avg) | Baseline |
| 2030 | 0.30 | -25% |
| 2040 | 0.22 | -45% |
| 2050 | 0.16 | -60% |

---

## Technology Landscape

### Emission Sources and Current Practices

| Source | Share of Agriculture GHG | Gas | Current Mitigation Status |
|--------|------------------------|-----|--------------------------|
| Enteric fermentation (ruminants) | 30% | CH4 | Feed additives emerging |
| Rice cultivation (flooded paddies) | 8% | CH4 | AWD practice spreading |
| Manure management | 12% | CH4 + N2O | Biogas digesters growing |
| Synthetic fertilizer application | 15% | N2O | Precision ag emerging |
| Crop residue management | 5% | N2O + CO2 | Conservation tillage growing |
| Energy use (machinery, heating) | 10% | CO2 | Electrification beginning |
| Land use change (deforestation for ag) | 15% | CO2 | Regulation increasing |
| Other (lime, urea, burning) | 5% | CO2 + N2O | Improved practices |

### Key Technology Transitions

#### 1. Enteric Methane Reduction (Feed Additives)

- **Transition**: Feed additives (3-NOP / Bovaer, red seaweed / Asparagopsis, essential oils) that inhibit methanogenesis in ruminant stomachs
- **Timeline**: 3-NOP commercially available 2023+; seaweed scaling 2025-2030
- **Reduction**: 20-80% of enteric CH4 per animal (varies by additive and dose)
- **Cost**: EUR 20-80/tCO2e
- **Certainty**: High (3-NOP: proven; regulatory approvals secured); Medium (seaweed: supply constraints)
- **Scale**: Potential to reduce largest single agricultural emission source by 30-50%

#### 2. Precision Agriculture and Smart Fertilization

- **Transition**: Variable rate application, nitrification inhibitors, slow-release fertilizers, soil sensors, satellite-guided application
- **Reduction**: 15-30% of fertilizer N2O emissions
- **Cost**: Negative to neutral (input savings)
- **Timeline**: Available now; scaling with digital agriculture
- **Certainty**: High
- **Additional benefits**: Reduced water pollution, improved yields

#### 3. Alternate Wetting and Drying (AWD) for Rice

- **Transition**: Replace continuous flooding of rice paddies with periodic drainage
- **Reduction**: 30-50% of rice CH4 emissions
- **Cost**: Negative (water savings)
- **Timeline**: Available now; spreading across Asia
- **Certainty**: High
- **Dependencies**: Water management infrastructure, farmer training

#### 4. Manure Management (Anaerobic Digestion)

- **Transition**: Capture CH4 from manure via anaerobic digesters, producing biogas for energy
- **Reduction**: 60-80% of manure CH4; produces renewable energy
- **Cost**: EUR 15-50/tCO2e (partially offset by biogas revenue)
- **Timeline**: Commercially available; scaling up
- **Certainty**: High
- **Best applications**: Large dairy and pig operations

#### 5. Soil Carbon Sequestration

- **Transition**: Cover crops, reduced tillage, crop rotation, biochar, compost application
- **Sequestration potential**: 0.3-1.5 tCO2e/hectare/year (varies by practice and climate)
- **Permanence**: Uncertain (soil carbon can be released if practices change)
- **Cost**: Negative to EUR 30/tCO2e
- **Timeline**: Available now; MRV methods improving
- **Certainty**: Medium (sequestration rates uncertain; permanence questions)
- **Note**: SBTi FLAG guidance requires separate accounting of removals vs. reductions

#### 6. Agroforestry and Silvopasture

- **Transition**: Integrate trees into agricultural landscapes (silvopasture, alley cropping, windbreaks)
- **Sequestration**: 2-10 tCO2e/hectare/year (depending on system and climate)
- **Co-benefits**: Biodiversity, erosion control, shade for livestock, diversified income
- **Cost**: EUR -20 to +30/tCO2e (often net positive economics)
- **Timeline**: Gradual adoption; long maturation period (10-20 years for full carbon benefit)
- **Certainty**: Medium-High (carbon sequestration well-documented; scale uncertain)

#### 7. Agricultural Energy Electrification

- **Transition**: Electric tractors, heat pumps for controlled environment agriculture, solar-powered irrigation
- **Reduction**: 80-100% of on-farm energy CO2
- **Timeline**: Electric tractors emerging (2024-2028); solar irrigation available now
- **Cost**: EUR 30-80/tCO2e (electric equipment premium declining)
- **Certainty**: Medium-High (technology available; adoption depends on cost and infrastructure)

#### 8. Zero-Deforestation Supply Chains

- **Transition**: Eliminate agricultural land expansion into forests (soy, palm oil, cattle, cocoa)
- **Reduction**: Prevents ~4-5 Gt CO2/year from land use change globally
- **Policy**: EU Deforestation Regulation (EUDR) requires verified deforestation-free supply chains
- **Cost**: Variable (depends on productivity improvements to compensate)
- **Certainty**: High (if enforced); cross-references AGENT-EUDR agents

---

## Abatement Levers

### Lever Waterfall (Dairy Farm, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Enteric methane inhibitor (3-NOP) | 15-25% | +20 to +50 | High |
| 2 | Anaerobic digestion of manure | 10-15% | +15 to +50 | High |
| 3 | Precision fertilizer management | 5-8% | -10 to +10 | High |
| 4 | Improved herd genetics and management | 5-8% | -5 to +15 | High |
| 5 | On-farm renewable energy (solar PV) | 3-5% | -10 to +10 | High |
| 6 | Soil carbon (cover crops, reduced tillage) | 3-8% | -5 to +30 | Medium |
| 7 | Agroforestry / silvopasture | 2-5% | -20 to +30 | Medium |

### Lever Waterfall (Crop Farm, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Precision fertilizer application (VRA) | 15-25% | -10 to +10 | High |
| 2 | Nitrification inhibitors | 8-15% | +10 to +30 | High |
| 3 | Cover crops and crop rotation | 5-10% | -5 to +15 | Medium |
| 4 | Reduced tillage / no-till | 5-8% | -10 to +5 | Medium |
| 5 | On-farm renewable energy | 5-8% | -10 to +10 | High |
| 6 | Electric/biofuel farm machinery | 3-5% | +30 to +80 | Medium |
| 7 | Biochar application | 2-5% | +20 to +60 | Low-Medium |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | Methane-reducing feed additives commercially available in all major markets |
| 2025 | Global pledge on 30% methane reduction (Global Methane Pledge progress) |
| 2030 | Agricultural methane emissions reduced 20% from 2020 levels |
| 2030 | Zero net deforestation for agricultural expansion |
| 2030 | Precision agriculture adopted on 50% of cropland in advanced economies |
| 2030 | AWD rice cultivation adopted on 30% of global paddy area |
| 2035 | Soil carbon MRV methods standardized and verified |
| 2040 | 50% of farm machinery electrified or running on biofuels |
| 2050 | Agriculture reaches 55-60% reduction from 2020; residual offset by removals |

---

## Benchmarks (2024)

| Benchmark | Value | Unit | Source |
|-----------|-------|------|--------|
| Global average dairy | 1.30 | kgCO2e/kg FPCM | FAO GLEAM |
| EU average dairy | 1.00 | kgCO2e/kg FPCM | JRC/FAO |
| Best practice dairy (NZ) | 0.70 | kgCO2e/kg FPCM | DairyNZ |
| Global average beef (grazing) | 300 | kgCO2e/kg protein | FAO GLEAM |
| EU average beef | 200 | kgCO2e/kg protein | JRC/FAO |
| Global average rice | 1.80 | kgCO2e/kg paddy | IRRI |
| Sector leader (diversified ag) | 0.25 | tCO2e/t product | CDP Climate 2024 |
| SBTi FLAG peer average | 0.55 | tCO2e/t product | SBTi Database 2024 |

---

## PACK-028 Usage Example

```python
from engines.sector_classification_engine import SectorClassificationEngine
from engines.pathway_generator_engine import PathwayGeneratorEngine
from engines.technology_roadmap_engine import TechnologyRoadmapEngine

# Classify
classifier = SectorClassificationEngine()
sector = classifier.classify({"nace_codes": ["A01.41"]})
# Result: agriculture (dairy farming)

# Generate pathway
pathway_gen = PathwayGeneratorEngine()
pathway = pathway_gen.generate(
    sector="agriculture",
    sub_sector="dairy",
    base_year=2023,
    base_year_intensity=1.10,  # kgCO2e/kg FPCM
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="linear",
    production_forecast={
        2023: 200_000,  # tonnes FPCM
        2030: 220_000,
        2050: 240_000,
    },
    region="oecd",
    ghg_profile={
        "enteric_ch4": 0.55,   # 55% of total GHG
        "manure_ch4": 0.10,
        "manure_n2o": 0.06,
        "fertilizer_n2o": 0.15,
        "energy_co2": 0.07,
        "other": 0.07,
    },
)

print(f"2030 Target: {pathway.target_2030:.2f} kgCO2e/kg FPCM")
print(f"2050 Target: {pathway.target_2050:.2f} kgCO2e/kg FPCM")

# Technology roadmap
roadmap = TechnologyRoadmapEngine()
tech = roadmap.build(
    sector="agriculture",
    sub_sector="dairy",
    pathway=pathway,
    farm_size_hectares=5000,
    herd_size=3000,
    current_practices={
        "feed_additive": False,
        "anaerobic_digestion": False,
        "precision_fertilizer": "basic",
        "renewable_energy": "partial",
    },
    capex_budget_annual_usd=2_000_000,
)

for lever in tech.recommended_levers:
    print(f"{lever.name}: {lever.reduction_pct:.0%}, payback {lever.payback_years:.1f} yrs")
```

---

## SBTi FLAG Guidance Integration

PACK-028 integrates with the SBTi FLAG (Forest, Land and Agriculture) guidance, which applies to companies where FLAG-related emissions exceed:
- 20% of total Scope 1+2+3 emissions, OR
- The company is in a FLAG-designated sector (agriculture, food/beverage, forestry, paper, tobacco)

**Key FLAG requirements:**
- Separate FLAG and non-FLAG targets
- Land-related removals reported separately from emission reductions
- Commodity-specific intensity metrics (beef, dairy, rice, palm oil, soy, etc.)
- No-deforestation commitment required by 2025

PACK-028 models both FLAG-compliant intensity targets and IEA NZE pathway alignment.

---

## Special Considerations

### Biogenic vs. Fossil Methane

The IPCC AR6 introduced distinct GWP values for biogenic and fossil methane. Agricultural methane is biogenic (part of the short carbon cycle), meaning its warming impact differs from fossil methane. PACK-028 uses:
- Biogenic CH4 GWP-100: 27.2 (default, IPCC AR6)
- Fossil CH4 GWP-100: 29.8 (for comparison with energy sector)
- GWP* (flow-based metric): Available as alternative metric in PACK-028 for methane-heavy sectors

### Carbon Removals

Agricultural carbon removals (soil carbon sequestration, agroforestry) are modeled separately from emission reductions in PACK-028, consistent with SBTi FLAG guidance. Removals cannot substitute for emission reductions.

---

## References

1. SBTi FLAG Guidance, Version 1.1 (2024)
2. IEA Net Zero by 2050, Chapter 7: Agriculture & Land Use
3. FAO GLEAM (Global Livestock Environmental Assessment Model) 2024
4. IPCC AR6 WGIII, Chapter 7: Agriculture, Forestry, and Other Land Use
5. Global Methane Pledge Progress Report 2024
6. CGIAR Research Programs: Climate Change, Agriculture and Food Security
7. FAO Pathways to Lower Emissions (2024)

---

**End of Agriculture Sector Guide**
