# Sector Guide: Steel

**Sector ID:** `steel`
**SDA Methodology:** SDA-Steel
**Intensity Metric:** tCO2e/tonne crude steel
**IEA Chapter:** Chapter 5 -- Industry (Steel)

---

## Sector Overview

The steel sector accounts for approximately 7-9% of global CO2 emissions, producing around 2.6 billion tonnes of crude steel annually. It is the largest industrial emitter. The primary SDA metric is tCO2e per tonne of crude steel, reflecting the carbon intensity of steel production across different production routes.

Steel production follows two primary routes: the integrated blast furnace-basic oxygen furnace (BF-BOF) route, which accounts for ~70% of global production and has high emissions (~2.0-2.3 tCO2e/t), and the electric arc furnace (EAF) route using scrap steel, which accounts for ~25% and has significantly lower emissions (~0.3-0.6 tCO2e/t depending on grid electricity source).

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `STL-01` | Crude steel intensity (overall) | tCO2e/tonne crude steel | Primary SDA metric. Weighted average across all production routes |
| `STL-01a` | BF-BOF intensity | tCO2e/tonne crude steel | Intensity for blast furnace-basic oxygen furnace route |
| `STL-02` | EAF intensity | tCO2e/tonne crude steel | Intensity for electric arc furnace route |
| `STL-03` | DRI intensity | tCO2e/tonne DRI | Intensity for direct reduced iron production |
| `STL-04` | Hot metal intensity | tCO2e/tonne hot metal | Intensity of blast furnace hot metal |

### Calculating STL-01 (Primary Metric)

```python
intensity_tco2e_per_t = (scope1_tco2e + scope2_tco2e) / crude_steel_production_tonnes
```

**Example:**
- Scope 1: 7,500,000 tCO2e (combustion + process emissions)
- Scope 2: 1,200,000 tCO2e (purchased electricity)
- Crude steel production: 5,000,000 tonnes
- Intensity: (7,500,000 + 1,200,000) / 5,000,000 = 1.74 tCO2e/t

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway

| Year | Intensity (tCO2e/t) | Reduction from 2020 |
|------|---------------------|-------------------|
| 2020 | 1.85 (global avg) | Baseline |
| 2025 | 1.60 | -14% |
| 2030 | 1.25 | -32% |
| 2035 | 0.85 | -54% |
| 2040 | 0.55 | -70% |
| 2045 | 0.28 | -85% |
| 2050 | 0.10 | -95% |

---

## Technology Landscape

### Current Technology Mix (Global, 2023)

| Technology | Share | Intensity (tCO2e/t) | Status |
|-----------|-------|---------------------|--------|
| BF-BOF (integrated) | 70% | 2.0-2.3 | Dominant, high emissions |
| EAF (scrap) | 25% | 0.3-0.6 | Growing, low emissions |
| DRI-NG (natural gas) | 4% | 1.0-1.4 | Moderate emissions |
| DRI-H2 (green hydrogen) | <1% | 0.05-0.3 | Emerging, near-zero |

### Key Technology Transitions

#### 1. BF-BOF to EAF Transition
- **Transition**: Replace integrated steelmaking with EAF using scrap
- **Timeline**: Continuous, constrained by scrap availability
- **Reduction**: -75% intensity per tonne shifted
- **Dependencies**: Scrap availability, electricity grid carbon intensity

#### 2. Green Hydrogen DRI
- **Transition**: Direct Reduced Iron using green hydrogen instead of natural gas/coal
- **Timeline**: Pilot by 2025, commercial from 2028, scale-up to 2040
- **Reduction**: -95% vs. BF-BOF
- **Dependencies**: Green hydrogen supply, electrolyzer capacity, renewable electricity

#### 3. CCS on BF-BOF
- **Transition**: Carbon capture retrofit on existing blast furnaces
- **Timeline**: Commercial from 2028-2030
- **Reduction**: -60-90% of BF-BOF emissions (depending on capture rate)
- **Dependencies**: CO2 transport and storage infrastructure

#### 4. Energy Efficiency
- **Transition**: Waste heat recovery, top-gas recycling, process optimization
- **Timeline**: Continuous improvement
- **Reduction**: -5-15% of total emissions
- **Dependencies**: Capital investment, operational expertise

#### 5. Scrap Recycling Rate Increase
- **Transition**: Improve scrap collection, sorting, and quality
- **Timeline**: Gradual increase through 2050
- **Reduction**: Enables more EAF production
- **Dependencies**: Scrap availability, collection infrastructure, quality standards

---

## Abatement Levers

### Lever Waterfall (Typical Integrated Steelmaker, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Energy efficiency + waste heat recovery | 5-8% | -20 to -10 | High |
| 2 | Scrap recycling rate increase | 5-8% | -5 to +10 | High |
| 3 | BF-BOF to EAF transition (partial) | 10-15% | +20 to +50 | High |
| 4 | Renewable electricity procurement | 3-5% | +10 to +30 | High |
| 5 | Green hydrogen DRI pilot | 5-10% | +50 to +100 | Medium |
| 6 | CCS retrofit (pilot on 1 BF) | 5-8% | +60 to +120 | Medium |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | First commercial green hydrogen DRI plant |
| 2030 | 10% of steel production via green hydrogen DRI |
| 2030 | EAF share reaches 40% globally |
| 2035 | 50% of new steelmaking capacity is near-zero |
| 2040 | 30% of steel production near-zero emission |
| 2050 | Steel sector reaches near-zero emissions intensity |

---

## Benchmarks (2024)

| Benchmark | Value (tCO2e/t) | Source |
|-----------|-----------------|--------|
| Global average | 1.85 | World Steel Association |
| OECD average | 1.65 | World Steel Association |
| EU average | 1.55 | European Steel Association |
| Sector leader (P10) | 0.95 | CDP Climate 2024 |
| SBTi peer average | 1.55 | SBTi Database 2024 |
| Best EAF operators | 0.30 | Company disclosures |

---

## PACK-028 Usage Example

```python
from workflows.full_sector_assessment_workflow import FullSectorAssessmentWorkflow

workflow = FullSectorAssessmentWorkflow()
result = workflow.execute(
    company_profile={
        "name": "EuroSteel AG",
        "nace_codes": ["C24.10"],
        "base_year": 2023,
        "base_year_production_tonnes": 5_000_000,
        "base_year_emissions_tco2e": 9_250_000,
        "current_technology_mix": {
            "bf_bof": 0.75, "eaf_scrap": 0.20, "dri_natural_gas": 0.05
        },
    },
    scenarios=["nze_15c", "wb2c", "2c"],
)
```

---

## References

1. SBTi SDA-Steel Methodology, SBTi Target Setting Tool V3.0 (2025)
2. IEA Net Zero by 2050, Chapter 5: Industry (Steel)
3. IEA Iron and Steel Technology Roadmap (2020)
4. World Steel Association Sustainability Indicators 2024
5. IPCC AR6 WGIII, Chapter 11: Industry

---

**End of Steel Sector Guide**
