# Sector Guide: Cross-Sector

**Sector ID:** `cross_sector`
**SDA Methodology:** N/A (Multi-sector aggregation)
**Intensity Metric:** Various (sector-weighted)
**IEA Chapter:** Cross-cutting

---

## Sector Overview

The Cross-Sector classification in PACK-028 addresses companies and organizations that operate across multiple SDA sectors and cannot be cleanly mapped to a single sector pathway. This includes diversified industrial conglomerates, holding companies, multi-sector utilities, and organizations whose business activities span two or more distinct SBTi SDA sectors.

Key scenarios where Cross-Sector classification applies:

1. **Diversified conglomerates**: Companies with divisions in multiple sectors (e.g., steel + chemicals + power generation)
2. **Multi-sector utilities**: Utilities that operate across power generation, district heating, and gas distribution
3. **Financial institutions**: Banks, insurers, and asset managers with portfolios spanning all sectors
4. **Government entities**: National, regional, or municipal entities with cross-sector emission portfolios
5. **Real estate + services**: Companies with building operations across residential and commercial
6. **Vertically integrated supply chains**: Companies operating across agriculture, food processing, and retail

The Cross-Sector approach in PACK-028 applies a **weighted sectoral decomposition** methodology, where each business segment is mapped to its corresponding SDA sector pathway, and the overall target is the production-weighted combination of individual sector pathways.

---

## Cross-Sector Methodology

### Weighted Sectoral Decomposition

For companies that operate across multiple SDA sectors, the overall intensity pathway is calculated as:

```python
# Step 1: Classify each business segment into its SDA sector
segments = [
    {"name": "Steel Division", "sector": "steel", "production_t": 2_000_000,
     "emissions_tco2e": 3_500_000},
    {"name": "Cement Division", "sector": "cement", "production_t": 5_000_000,
     "emissions_tco2e": 3_000_000},
    {"name": "Power Division", "sector": "power_generation", "production_mwh": 10_000_000,
     "emissions_tco2e": 4_000_000},
]

# Step 2: Calculate segment-level intensities
for seg in segments:
    seg["intensity"] = seg["emissions_tco2e"] / seg["production"]

# Step 3: Generate individual sector pathways for each segment
for seg in segments:
    seg["pathway"] = PathwayGeneratorEngine().generate(
        sector=seg["sector"],
        base_year=2023,
        base_year_intensity=seg["intensity"],
        target_year_near=2030,
        target_year_long=2050,
        scenario="nze_15c",
    )

# Step 4: Aggregate using emission-weighted approach
for year in range(2023, 2051):
    total_target_emissions = sum(
        seg["pathway"].get_target(year) * seg["production_forecast"][year]
        for seg in segments
    )
    total_production = sum(seg["production_forecast"][year] for seg in segments)
    # Overall absolute target for the year
    company_target_absolute[year] = total_target_emissions
```

### Intensity Metrics for Cross-Sector

| Metric ID | Name | Unit | Description |
|-----------|------|------|-------------|
| `XSC-01` | Absolute emissions trajectory | tCO2e/year | Total Scope 1+2 emissions across all segments |
| `XSC-02` | Revenue-based intensity | tCO2e/million EUR | Alternative metric for conglomerates |
| `XSC-03` | Segment-weighted intensity | dimensionless index | Normalized index tracking convergence across segments |
| `XSC-04` | Sector alignment score | 0-100 | Weighted average of individual sector alignment scores |
| `XSC-05` | Temperature alignment | deg C | Implied temperature rise based on weighted sector pathways |

### Calculating XSC-04 (Sector Alignment Score)

```python
# Each segment gets an alignment score (0-100) based on its position
# relative to its sector-specific SDA pathway

segment_scores = []
for seg in segments:
    # Score of 100 = on or below the 1.5C pathway
    # Score of 50 = on the 2C pathway
    # Score of 0 = no reduction trajectory
    alignment = convergence_analyzer.score(
        sector=seg["sector"],
        current_intensity=seg["current_intensity"],
        year=2023,
        scenario="nze_15c",
    )
    segment_scores.append({
        "segment": seg["name"],
        "score": alignment.score,
        "weight": seg["emissions_share"],
    })

# Weighted average
overall_score = sum(s["score"] * s["weight"] for s in segment_scores)
# Example: 72.5 out of 100
```

---

## Cross-Sector Pathway Design

### Step 1: Segmentation

Decompose the company into business segments that each map to a single SDA sector:

| Segment | SDA Sector | Primary Metric | NACE Code |
|---------|-----------|---------------|-----------|
| Iron & Steel Manufacturing | `steel` | tCO2e/t crude steel | C24.10 |
| Cement Production | `cement` | tCO2e/t cement | C23.51 |
| Power Generation | `power_generation` | gCO2/kWh | D35.11 |
| Chemical Production | `chemicals` | tCO2e/t product | C20.xx |
| Real Estate Operations | `buildings_commercial` | kgCO2/m2/yr | L68.20 |
| Logistics Fleet | `road_transport` | gCO2/tkm | H49.41 |

### Step 2: Individual Pathway Generation

Generate SDA-compliant pathways for each segment individually:

```python
from engines.pathway_generator_engine import PathwayGeneratorEngine

pathway_gen = PathwayGeneratorEngine()

steel_pathway = pathway_gen.generate(
    sector="steel", base_year=2023, base_year_intensity=1.75,
    target_year_near=2030, scenario="nze_15c",
)

cement_pathway = pathway_gen.generate(
    sector="cement", base_year=2023, base_year_intensity=0.58,
    target_year_near=2030, scenario="nze_15c",
)

power_pathway = pathway_gen.generate(
    sector="power_generation", base_year=2023, base_year_intensity=420,
    target_year_near=2030, scenario="nze_15c",
)
```

### Step 3: Aggregation

Combine individual pathways into an overall company trajectory:

```python
from workflows.full_sector_assessment_workflow import FullSectorAssessmentWorkflow

workflow = FullSectorAssessmentWorkflow()
result = workflow.execute(
    company_profile={
        "name": "IndustrialCo AG",
        "segments": [
            {
                "name": "Steel Division",
                "nace_codes": ["C24.10"],
                "base_year_production": 2_000_000,
                "base_year_emissions": 3_500_000,
                "production_forecast": {2030: 2_100_000, 2050: 2_200_000},
            },
            {
                "name": "Cement Division",
                "nace_codes": ["C23.51"],
                "base_year_production": 5_000_000,
                "base_year_emissions": 2_900_000,
                "production_forecast": {2030: 5_200_000, 2050: 5_500_000},
            },
            {
                "name": "Power Division",
                "nace_codes": ["D35.11"],
                "base_year_production": 10_000_000,
                "base_year_emissions": 4_200_000,
                "production_forecast": {2030: 11_000_000, 2050: 13_000_000},
            },
        ],
    },
    scenarios=["nze_15c", "wb2c", "2c"],
)
```

### Step 4: Reporting

Generate a combined cross-sector report showing:
- Individual segment alignment scores
- Overall company alignment score (emission-weighted)
- Segment-level technology roadmaps
- Combined abatement waterfall
- Priority actions ranked by impact and cost-effectiveness

---

## Financial Institution Application

### Portfolio Alignment

Financial institutions (banks, asset managers, insurers) use cross-sector pathway analysis to assess portfolio alignment with climate scenarios:

```python
from engines.scenario_comparison_engine import ScenarioComparisonEngine

scenario_engine = ScenarioComparisonEngine()

# Assess portfolio of financed emissions by sector
portfolio = {
    "power_generation": {"financed_emissions_tco2e": 5_000_000, "intensity": 380},
    "steel": {"financed_emissions_tco2e": 2_000_000, "intensity": 1.70},
    "cement": {"financed_emissions_tco2e": 1_500_000, "intensity": 0.60},
    "oil_gas": {"financed_emissions_tco2e": 8_000_000, "intensity": 28.0},
    "buildings_commercial": {"financed_emissions_tco2e": 1_000_000, "intensity": 60},
    "road_transport": {"financed_emissions_tco2e": 500_000, "intensity": 130},
}

# Calculate weighted portfolio alignment
portfolio_alignment = scenario_engine.assess_portfolio(
    portfolio=portfolio,
    scenario="nze_15c",
    assessment_year=2023,
    target_year=2030,
)

print(f"Portfolio temperature alignment: {portfolio_alignment.implied_temperature:.1f} deg C")
print(f"Overall alignment score: {portfolio_alignment.score:.0f}/100")

for sector, result in portfolio_alignment.sector_results.items():
    print(f"  {sector}: {result.score:.0f}/100, gap: {result.gap_to_pathway:+.1%}")
```

### PCAF (Partnership for Carbon Accounting Financials)

PACK-028 integrates with PCAF methodology for calculating financed emissions across asset classes:
- Listed equity and corporate bonds
- Business loans and unlisted equity
- Project finance
- Commercial real estate
- Mortgages

---

## SBTi Approach for Cross-Sector Companies

### SBTi Target-Setting Methods

| Company Type | SBTi Method | PACK-028 Implementation |
|-------------|-------------|------------------------|
| Single SDA sector | SDA convergence | Sector-specific pathway |
| Multi-sector (all SDA) | SDA per segment + aggregation | Weighted sectoral decomposition |
| Non-SDA sector | Absolute contraction approach | IEA NZE absolute pathway |
| Financial institution | SBTi Financial Institution guidance | Portfolio alignment scoring |
| Mixed SDA + non-SDA | Hybrid approach | Segment-by-segment with weighted rollup |

### Absolute Contraction Approach (Non-SDA Segments)

For business segments that don't map to an SDA sector, PACK-028 uses the absolute contraction approach:

```python
# Absolute contraction: reduce absolute emissions by a fixed annual rate
# 1.5C-aligned: -4.2% per year (linear, 2020-2050)
# WB2C: -2.5% per year

annual_reduction_rate = 0.042  # 4.2% per year for 1.5C

for year in range(base_year + 1, 2051):
    target_emissions[year] = target_emissions[year - 1] * (1 - annual_reduction_rate)
```

---

## Abatement Levers (Cross-Sector)

### Common Cross-Sector Levers

| # | Lever | Applicability | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|--------------|-----------|-----------------|-----------|
| 1 | Renewable electricity procurement | All sectors | 15-40% of Scope 2 | 0 to +15 | Very High |
| 2 | Energy efficiency programs | All sectors | 5-20% of total | -20 to -5 | Very High |
| 3 | On-site solar PV | All sectors with roof/land | 3-10% of Scope 2 | -10 to +10 | High |
| 4 | Fleet electrification | Transport-dependent | 5-15% of fleet emissions | 0 to +30 | High |
| 5 | Building envelope and HVAC | Real estate, offices | 10-30% of building emissions | +20 to +80 | High |
| 6 | Internal carbon pricing | Strategic planning tool | Drives investment decisions | N/A | N/A |

---

## IEA Key Milestones (Cross-Sector)

| Year | Milestone |
|------|-----------|
| 2025 | All major companies have set SBTi-aligned targets across all segments |
| 2025 | Internal carbon pricing adopted by 80% of Fortune 500 companies |
| 2030 | 50% of electricity consumption from renewable sources (all sectors) |
| 2030 | Cross-sector energy efficiency improvement of 4% per year |
| 2035 | All new construction zero-carbon-ready (buildings segments) |
| 2040 | All light-duty vehicle fleets zero-emission |
| 2050 | All sectors achieve net-zero emissions per NZE pathway |

---

## Benchmarks (2024)

| Benchmark | Value | Unit | Source |
|-----------|-------|------|--------|
| Fortune 500 average intensity | 150 | tCO2e/M USD revenue | CDP Climate 2024 |
| SBTi-committed companies average | 85 | tCO2e/M USD revenue | SBTi Database 2024 |
| Climate Action 100+ focus companies | 120 | tCO2e/M USD revenue | CA100+ Benchmark |
| Financial sector portfolio alignment | 2.7 | deg C implied temp | PCAF/TPI 2024 |
| Best practice conglomerate | 35 | tCO2e/M USD revenue | Company disclosures |

---

## PACK-028 Usage Example

```python
from workflows.full_sector_assessment_workflow import FullSectorAssessmentWorkflow
from templates.sector_strategy_report import SectorStrategyReport

# Full cross-sector assessment
workflow = FullSectorAssessmentWorkflow()
result = workflow.execute(
    company_profile={
        "name": "GlobalIndustries Corp",
        "classification": "cross_sector",
        "segments": [
            {
                "name": "Steel Operations",
                "sector": "steel",
                "nace_codes": ["C24.10"],
                "base_year_production": {"tonnes": 3_000_000},
                "base_year_emissions_tco2e": 5_500_000,
                "production_forecast": {2030: 3_200_000, 2050: 3_500_000},
            },
            {
                "name": "Power Generation",
                "sector": "power_generation",
                "nace_codes": ["D35.11"],
                "base_year_production": {"mwh": 15_000_000},
                "base_year_emissions_tco2e": 6_000_000,
                "production_forecast": {2030: 18_000_000, 2050: 22_000_000},
            },
            {
                "name": "Corporate Real Estate",
                "sector": "buildings_commercial",
                "nace_codes": ["L68.20"],
                "base_year_production": {"m2": 500_000},
                "base_year_emissions_tco2e": 30_000,
                "production_forecast": {2030: 550_000, 2050: 600_000},
            },
        ],
    },
    scenarios=["nze_15c", "wb2c", "2c"],
)

# Generate executive strategy report
report = SectorStrategyReport()
output = report.render(
    data=result,
    format="html",
    include_charts=True,
    include_segment_detail=True,
    include_investment_analysis=True,
)

print(f"Overall alignment score: {result.alignment_score:.0f}/100")
print(f"Implied temperature: {result.implied_temperature:.1f} deg C")
for seg in result.segment_results:
    print(f"  {seg.name}: score={seg.score:.0f}, gap_2030={seg.gap_2030:+.1%}")
```

---

## Special Considerations

### Allocation and Double Counting

When a conglomerate reports cross-sector emissions:
- Internal transactions (e.g., power division supplying electricity to steel division) must be carefully allocated to avoid double counting
- PACK-028 provides allocation rules consistent with GHG Protocol guidance
- Scope 2 of one segment may overlap with Scope 1 of another segment within the same company

### Portfolio Transition Planning

Cross-sector companies can use PACK-028 to model portfolio transition strategies:
- **Divest high-carbon**: Model the emission impact of divesting high-intensity segments
- **Invest in low-carbon**: Model the impact of acquiring or growing low-carbon businesses
- **Transform existing**: Model the impact of decarbonizing existing high-carbon segments
- **Balanced approach**: Combine divestment, investment, and transformation

### Temperature Alignment Scoring

PACK-028 calculates an implied temperature alignment score for cross-sector companies using the IEA scenario framework:
- Each segment is scored against its sector-specific NZE pathway
- Scores are emission-weighted to produce an overall portfolio temperature
- This aligns with TCFD/ISSB disclosure requirements

---

## References

1. SBTi Corporate Standard v2.0 (2025)
2. SBTi Financial Institutions Guidance (2024)
3. IEA Net Zero by 2050 (cross-sector analysis)
4. PCAF Global GHG Accounting and Reporting Standard for the Financial Industry (2022)
5. TPI (Transition Pathway Initiative) Benchmark Methodology (2024)
6. IPCC AR6 WGIII, Chapter 12: Cross-Sectoral Perspectives
7. GHG Protocol Corporate Standard (2004, updated 2015)

---

**End of Cross-Sector Guide**
