# Sector Guide: Cement

**Sector ID:** `cement`
**SDA Methodology:** SDA-Cement
**Intensity Metric:** tCO2e/tonne cement
**IEA Chapter:** Chapter 5 -- Industry (Cement)

---

## Sector Overview

The cement sector is responsible for approximately 7% of global CO2 emissions, producing around 4.1 billion tonnes of cement annually. Uniquely among industrial sectors, approximately 60% of cement emissions come from the chemical process of calcination (CaCO3 -> CaO + CO2), which cannot be reduced through fuel switching alone. This makes cement one of the hardest-to-abate sectors, requiring carbon capture technology for deep decarbonization.

The SDA-Cement methodology uses tCO2e per tonne of cement as the convergence metric, accounting for both process and combustion emissions.

---

## Intensity Metrics

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `CMT-01` | Clinker intensity | tCO2e/tonne clinker | Process + combustion emissions per tonne of clinker |
| `CMT-02` | Cement intensity | tCO2e/tonne cement | Primary SDA metric. Including clinker ratio effect |
| `CMT-03` | Concrete intensity | tCO2e/m3 concrete | Downstream product intensity |
| `CMT-04` | Clinker-to-cement ratio | dimensionless | Key lever for reducing cement intensity |

### Calculating CMT-02 (Primary Metric)

```python
intensity_tco2e_per_t = (scope1_tco2e + scope2_tco2e) / cement_production_tonnes

# Where Scope 1 = process_emissions + combustion_emissions
# process_emissions = clinker_production * 0.525 (calcination factor)
# combustion_emissions = fuel_energy * emission_factor
```

---

## SBTi SDA Pathway

### NZE 1.5C Convergence Pathway

| Year | Intensity (tCO2e/t) | Reduction from 2020 |
|------|---------------------|-------------------|
| 2020 | 0.62 (global avg) | Baseline |
| 2025 | 0.56 | -10% |
| 2030 | 0.47 | -24% |
| 2035 | 0.35 | -44% |
| 2040 | 0.25 | -60% |
| 2045 | 0.12 | -81% |
| 2050 | 0.04 | -94% |

---

## Technology Landscape

### Key Decarbonization Levers

#### 1. Clinker Substitution
- **Action**: Reduce clinker-to-cement ratio by substituting with fly ash, slag, calcined clay, or limestone
- **Current ratio**: ~0.70 (global average)
- **Target ratio**: 0.55-0.60 by 2030, 0.50 by 2050
- **Reduction**: ~10-15% per 0.10 ratio reduction
- **Cost**: Negative to neutral (reduces raw material costs)
- **Certainty**: High (proven technology, LC3 gaining traction)

#### 2. Alternative Fuels
- **Action**: Replace fossil fuels with biomass, waste-derived fuels (RDF/SRF), tires
- **Current share**: ~18% globally
- **Target share**: 40% by 2030, 60% by 2050
- **Reduction**: 10-20% of combustion emissions
- **Cost**: Neutral to slightly positive
- **Certainty**: High

#### 3. Energy Efficiency
- **Action**: Upgrade to high-efficiency precalciner kilns, waste heat recovery, process optimization
- **Reduction**: 5-10%
- **Cost**: Negative (energy savings)
- **Certainty**: High

#### 4. Carbon Capture (CCUS)
- **Action**: Post-combustion or oxy-fuel capture on kiln emissions
- **Timeline**: Pilot 2025-2030, commercial scale 2030-2040
- **Reduction**: 60-95% of remaining emissions
- **Cost**: EUR 60-120/tCO2e
- **Certainty**: Medium (technology proven, cost uncertain)
- **Critical importance**: Required because ~60% of emissions are process CO2

#### 5. Novel Cements
- **Action**: Geopolymer cements, belite-ye'elimite-ferrite (BYF), calcium sulfoaluminate
- **Reduction**: 30-80% vs. Portland cement
- **Timeline**: Limited commercial availability by 2030
- **Cost**: Variable, premium products
- **Certainty**: Low-Medium (standards, acceptance)

---

## Abatement Levers (Typical Cement Company, 2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Clinker substitution (0.75 -> 0.65) | 8-10% | -15 to 0 | High |
| 2 | Alternative fuels (15% -> 40%) | 6-8% | 0 to +15 | High |
| 3 | Energy efficiency (kiln upgrade) | 4-5% | -10 to -5 | High |
| 4 | Renewable electricity | 3-4% | +10 to +25 | High |
| 5 | CCS pilot (1 kiln, 30% capture) | 3-5% | +70 to +120 | Medium |

---

## IEA Key Milestones

| Year | Milestone |
|------|-----------|
| 2025 | Average clinker-to-cement ratio below 0.70 globally |
| 2030 | Alternative fuel share reaches 30% globally |
| 2030 | 10 large-scale CCS projects on cement plants operational |
| 2030 | Blended cement share reaches 50% of production |
| 2040 | CCS deployed on 30% of cement production |
| 2050 | Cement sector achieves near-zero intensity |

---

## Benchmarks (2024)

| Benchmark | Value (tCO2e/t) | Source |
|-----------|-----------------|--------|
| Global average | 0.62 | GCCA |
| EU average | 0.58 | GCCA Europe |
| Sector leader (P10) | 0.42 | CDP Climate 2024 |
| SBTi peer average | 0.55 | SBTi Database 2024 |
| IEA NZE 2025 target | 0.56 | IEA NZE 2050 |

---

## References

1. SBTi SDA-Cement Methodology, SBTi Target Setting Tool V3.0 (2025)
2. IEA Technology Roadmap: Low-Carbon Transition in the Cement Industry (2018)
3. GCCA Getting the Numbers Right (GNR) Database 2024
4. IPCC AR6 WGIII, Chapter 11: Industry (Cement)
5. IEA Net Zero by 2050, Chapter 5: Industry (Cement)

---

**End of Cement Sector Guide**
