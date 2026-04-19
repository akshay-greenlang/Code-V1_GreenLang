# PACK-028 Sector Pathway Pack -- User Guide

**Pack ID:** PACK-028-sector-pathway
**Version:** 1.0.0
**Last Updated:** 2026-03-19

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Sector Pathway Design Walkthrough](#sector-pathway-design-walkthrough)
4. [SBTi SDA Validation Guide](#sbti-sda-validation-guide)
5. [Technology Roadmap Planning](#technology-roadmap-planning)
6. [Progress Monitoring](#progress-monitoring)
7. [Multi-Scenario Analysis](#multi-scenario-analysis)
8. [Abatement Waterfall Analysis](#abatement-waterfall-analysis)
9. [Sector Benchmarking](#sector-benchmarking)
10. [Report Generation](#report-generation)
11. [Best Practices](#best-practices)
12. [Appendix: Sector Reference Data](#appendix-sector-reference-data)

---

## Introduction

### Purpose of This Guide

This guide provides step-by-step instructions for using PACK-028 Sector Pathway Pack to design, validate, and monitor sector-specific decarbonization pathways. It covers all major use cases from initial sector classification through ongoing progress monitoring.

### Who Should Read This Guide

- **Sustainability teams** designing sector-specific decarbonization strategies
- **Climate analysts** modeling SBTi SDA pathways
- **Technology planners** building sector transition roadmaps
- **Board-level users** reviewing sector pathway dashboards and scenario comparisons
- **ESG analysts** benchmarking company performance against sector pathways

### Prerequisites

Before using PACK-028, ensure:

1. The GreenLang platform is deployed and operational
2. PACK-028 migrations (V181-V186) have been applied
3. You have appropriate role permissions (`pathway_designer` or `sector_analyst`)
4. Baseline emissions data is available (directly or via PACK-021 integration)
5. Your sector-specific activity data is prepared (production volumes, technology mix, etc.)

### Key Concepts

| Concept | Definition |
|---------|-----------|
| **SDA** | Sectoral Decarbonization Approach -- SBTi methodology for setting sector-specific intensity targets |
| **Intensity Metric** | Emission rate per unit of sector-specific output (e.g., tCO2e per tonne of steel) |
| **Convergence** | The process of a company's intensity approaching the sector pathway target over time |
| **Pathway** | Year-by-year intensity trajectory from base year to target year |
| **Scenario** | Climate outcome framework (NZE 1.5C, WB2C, 2C, APS, STEPS) |
| **Abatement Lever** | Specific technology or practice change that reduces emissions |
| **IEA Milestone** | Technology deployment target from IEA NZE 2050 roadmap |
| **Convergence Model** | Mathematical function describing how intensity decreases over time |

---

## Getting Started

### Step 1: Run the Setup Wizard

The setup wizard guides you through initial configuration in 7 steps.

```python
from integrations.setup_wizard import SetupWizard

wizard = SetupWizard()

# Step 1: Select your sector
result = wizard.run(step=1, data={
    "sector": "steel",
    "sub_sectors": ["integrated_steel", "eaf_steel"],
})
print(f"Sector selected: {result.sector}")
print(f"SDA methodology: {result.sda_methodology}")
print(f"Intensity metric: {result.intensity_metric}")
```

```python
# Step 2: Company profile
result = wizard.run(step=2, data={
    "name": "SteelCorp International",
    "nace_codes": ["C24.10"],
    "gics_code": "15104020",
    "country": "DE",
    "production_volume_tonnes": 5_000_000,
    "employees": 12_000,
})
```

```python
# Step 3: Base year configuration
result = wizard.run(step=3, data={
    "base_year": 2023,
    "base_year_intensity": 1.85,
    "base_year_production_tonnes": 5_000_000,
    "base_year_emissions_tco2e": 9_250_000,
    "base_year_scope1_tco2e": 7_500_000,
    "base_year_scope2_tco2e": 1_750_000,
})
```

```python
# Step 4: Scenario selection
result = wizard.run(step=4, data={
    "primary_scenario": "nze_15c",
    "comparison_scenarios": ["wb2c", "2c"],
    "convergence_model": "s_curve",
    "target_year_near": 2030,
    "target_year_long": 2050,
})
```

```python
# Step 5: Technology assessment
result = wizard.run(step=5, data={
    "current_technology_mix": {
        "bf_bof": 0.75,
        "eaf_scrap": 0.20,
        "dri_natural_gas": 0.05,
    },
    "installed_capacity_mtpa": 5.0,
    "average_plant_age_years": 22,
})
```

```python
# Step 6: Budget and timeline
result = wizard.run(step=6, data={
    "capex_budget_annual_usd": 500_000_000,
    "planning_horizon_year": 2050,
    "investment_decision_cycle": "annual",
})
```

```python
# Step 7: Validation and go-live
result = wizard.run(step=7, data={
    "run_health_check": True,
    "generate_initial_pathway": True,
    "enable_monitoring": True,
})

print(f"Setup complete: {result.status}")
print(f"Health check: {result.health_score}/100")
print(f"Initial pathway ID: {result.pathway_id}")
```

### Step 2: Verify Health Check

After setup, verify the system is healthy:

```python
from integrations.health_check import HealthCheck

hc = HealthCheck()
result = hc.run()

print(f"Overall Score: {result.overall_score}/100")
print(f"Status: {result.status}")

for category in result.categories:
    status_icon = "OK" if category.score >= 90 else "WARN" if category.score >= 70 else "FAIL"
    print(f"  [{status_icon}] {category.name}: {category.score}/100")
```

Expected output for a healthy system:

```
Overall Score: 100/100
Status: HEALTHY
  [OK] Database Connectivity: 100/100
  [OK] Redis Cache: 100/100
  [OK] SBTi SDA Data: 100/100
  [OK] IEA NZE Data: 100/100
  [OK] IPCC AR6 Data: 100/100
  [OK] MRV Agent Connectivity: 100/100
  [OK] DATA Agent Connectivity: 100/100
  [OK] FOUND Agent Connectivity: 100/100
  [OK] Engine Availability: 100/100
  [OK] Workflow Availability: 100/100
  [OK] Template Availability: 100/100
  [OK] Integration Availability: 100/100
  [OK] Migration Status: 100/100
  [OK] PACK-021 Bridge: 100/100
  [OK] Sector Data Freshness: 100/100
  [OK] Benchmark Data Freshness: 100/100
  [OK] Emission Factor Data: 100/100
  [OK] Cache Performance: 100/100
  [OK] API Response Time: 100/100
  [OK] Provenance Integrity: 100/100
```

---

## Sector Pathway Design Walkthrough

This section walks through the complete process of designing a sector-specific decarbonization pathway, using the steel sector as an example.

### Phase 1: Sector Classification

The first step is classifying your company into the correct SBTi SDA sector.

```python
from engines.sector_classification_engine import SectorClassificationEngine

engine = SectorClassificationEngine()

result = engine.classify(
    company_profile={
        "name": "SteelCorp International",
        "nace_codes": ["C24.10"],
        "gics_code": "15104020",
        "isic_code": "2410",
        "revenue_breakdown": {
            "integrated_steel": 0.75,
            "eaf_steel": 0.20,
            "downstream_processing": 0.05,
        },
        "primary_products": [
            "hot_rolled_coil",
            "cold_rolled_coil",
            "galvanized_steel",
            "rebar",
        ],
    }
)

# Review classification results
print(f"Primary Sector: {result.primary_sector}")
print(f"Sub-Sectors: {result.sub_sectors}")
print(f"SDA Eligible: {result.sda_eligible}")
print(f"SDA Methodology: {result.sda_methodology}")
print(f"Intensity Metric: {result.intensity_metric}")
print(f"IEA Chapter: {result.iea_chapter}")
print(f"Confidence: {result.confidence_score:.0%}")
```

**Understanding Classification Results:**

- **Primary Sector**: The SDA sector used for pathway generation. Must match one of the 12 SBTi SDA sectors or 4 extended IEA sectors.
- **SDA Eligible**: Whether the company can use SDA (mandatory for matching sectors per SBTi Corporate Standard).
- **Confidence Score**: How certain the classification is (>90% = high confidence, 70-90% = review recommended).

**Multi-Sector Companies:**

If your company operates across multiple SDA sectors, you can classify each division separately:

```python
# Classify each division
divisions = [
    {"name": "Steel Division", "nace_codes": ["C24.10"], "revenue_share": 0.60},
    {"name": "Cement Division", "nace_codes": ["C23.51"], "revenue_share": 0.25},
    {"name": "Services", "nace_codes": ["M71.12"], "revenue_share": 0.15},
]

for division in divisions:
    result = engine.classify({"nace_codes": division["nace_codes"]})
    print(f"{division['name']}: {result.primary_sector} "
          f"(SDA: {result.sda_eligible}, Metric: {result.intensity_metric})")
```

### Phase 2: Intensity Calculation

Calculate your sector-specific intensity metric from activity and emissions data.

```python
from engines.intensity_calculator_engine import IntensityCalculatorEngine

engine = IntensityCalculatorEngine()

result = engine.calculate(
    sector="steel",
    activity_data={
        "crude_steel_production_tonnes": 5_000_000,
        "bf_bof_production_tonnes": 3_750_000,
        "eaf_production_tonnes": 1_000_000,
        "dri_production_tonnes": 250_000,
    },
    emissions_data={
        "scope1_tco2e": 7_500_000,
        "scope2_location_tco2e": 1_500_000,
        "scope2_market_tco2e": 1_200_000,
        "process_emissions_tco2e": 3_200_000,
        "combustion_emissions_tco2e": 4_300_000,
    },
    reporting_year=2023,
    region="eu",
)

print(f"Primary Intensity: {result.primary_intensity.value:.2f} "
      f"{result.primary_intensity.unit}")
print(f"Data Quality Score: {result.primary_intensity.data_quality_score:.1f}")

# Review sub-intensities by production route
for sub in result.sub_intensities:
    print(f"  {sub.name}: {sub.value:.2f} {sub.unit} "
          f"(share: {sub.production_share:.0%})")

# Review trend
if result.trend:
    print(f"Trend: {result.trend.direction} ({result.trend.annual_change_pct:+.1%}/year)")
```

**Key Intensity Metrics by Sector:**

| Sector | Primary Metric | How to Calculate |
|--------|---------------|-----------------|
| Power | gCO2/kWh | (Scope 1 + Scope 2) / Total electricity generated |
| Steel | tCO2e/t crude steel | (Scope 1 + Scope 2) / Total crude steel production |
| Cement | tCO2e/t cement | (Scope 1 + Scope 2) / Total cement production |
| Aviation | gCO2/pkm | Total CO2 / Total passenger-kilometers |
| Buildings | kgCO2/m2/year | Annual (Scope 1 + Scope 2) / Total floor area |

### Phase 3: Pathway Generation

Generate the SBTi SDA convergence pathway aligned with your chosen climate scenario.

```python
from engines.pathway_generator_engine import PathwayGeneratorEngine

engine = PathwayGeneratorEngine()

result = engine.generate(
    sector="steel",
    base_year=2023,
    base_year_intensity=1.85,
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
    convergence_model="s_curve",
    production_forecast={
        2023: 5_000_000,
        2025: 5_100_000,
        2030: 5_500_000,
        2035: 5_700_000,
        2040: 5_800_000,
        2045: 5_900_000,
        2050: 6_000_000,
    },
    region="global",
)

# Display key targets
print(f"Pathway: {result.pathway_name}")
print(f"Base Intensity ({result.base_year}): {result.base_intensity:.2f} tCO2e/t")
print(f"2030 Target: {result.target_2030:.2f} tCO2e/t")
print(f"2040 Target: {result.target_2040:.2f} tCO2e/t")
print(f"2050 Target: {result.target_2050:.2f} tCO2e/t")
print(f"Near-Term Reduction Rate: {result.annual_reduction_rate_near:.1%}/year")
print(f"Long-Term Reduction Rate: {result.annual_reduction_rate_long:.1%}/year")

# Display year-by-year pathway
print("\nYear  | Intensity  | Absolute Emissions  | Production  | Convergence")
print("------|------------|---------------------|-------------|------------")
for point in result.annual_pathway:
    print(f"{point.year}  | {point.intensity:>9.3f}  | "
          f"{point.absolute_emissions:>18,.0f}  | "
          f"{point.production_volume:>10,.0f}  | {point.convergence_pct:>9.1%}")
```

**Understanding Convergence Models:**

Choose the convergence model that best fits your sector's decarbonization dynamics:

| Model | Best For | Characteristics |
|-------|---------|----------------|
| **Linear** | Buildings, some transport | Steady, predictable reduction year-over-year |
| **Exponential** | Power, road transport | Slow start, accelerating reduction as technologies deploy |
| **S-curve** | Steel, cement, aviation | Slow start (R&D), rapid mid-period (deployment), plateau (optimization) |
| **Stepped** | Shipping (IMO milestones) | Discrete jumps at policy milestone dates |

### Phase 4: Gap Analysis

Analyze the gap between your current trajectory and the sector pathway.

```python
from engines.convergence_analyzer_engine import ConvergenceAnalyzerEngine

engine = ConvergenceAnalyzerEngine()

result = engine.analyze(
    current_intensity=1.65,
    current_year=2025,
    sector_pathway=pathway_result,
    company_trajectory=[
        {"year": 2021, "intensity": 1.92},
        {"year": 2022, "intensity": 1.85},
        {"year": 2023, "intensity": 1.78},
        {"year": 2024, "intensity": 1.72},
        {"year": 2025, "intensity": 1.65},
    ],
)

print(f"Convergence Status: {result.convergence_status}")
print(f"Current Gap to Pathway: {result.gap_to_pathway.current_gap_pct:.1%}")
print(f"Gap Direction: {result.gap_to_pathway.direction}")
print(f"Gap Narrowing: {result.gap_to_pathway.narrowing}")
print(f"\nGap to 2030 Target: {result.gap_to_2030_target.gap_pct:.1%}")
print(f"Achievable at Current Rate: {result.gap_to_2030_target.achievable_at_current_rate}")
print(f"\nRequired Annual Reduction: {result.required_annual_reduction.to_2030_target:.1%}/year")
print(f"Current Reduction Rate: {result.required_annual_reduction.current_rate:.1%}/year")
print(f"Acceleration Needed: {result.required_annual_reduction.acceleration_needed_pct:.1%}")
print(f"\nRisk Level: {result.risk_level}")
print(f"Time to Convergence: {result.time_to_convergence_years:.1f} years")

# Review recommendations
print("\nRecommendations:")
for rec in result.recommendations:
    print(f"  [{rec.priority}] {rec.action}")
    print(f"      Impact: {rec.impact_tco2e_per_tonne:.2f} tCO2e/t")
```

### Phase 5: Using the Workflow

For convenience, run all phases together using the Sector Pathway Design Workflow:

```python
from workflows.sector_pathway_design_workflow import SectorPathwayDesignWorkflow

workflow = SectorPathwayDesignWorkflow(config=config)

result = workflow.execute(
    company_profile={
        "name": "SteelCorp International",
        "nace_codes": ["C24.10"],
        "base_year": 2023,
        "base_year_production_tonnes": 5_000_000,
        "base_year_emissions_tco2e": 9_250_000,
    },
    target_scenario="nze_15c",
    target_year_near=2030,
    target_year_long=2050,
)

# Review workflow results
for phase in result.phases:
    print(f"Phase {phase.phase}: {phase.name} - {phase.status} ({phase.duration_ms}ms)")

print(f"\nFinal Result:")
print(f"  Sector: {result.final_result.sector}")
print(f"  Base Intensity: {result.final_result.base_intensity:.2f}")
print(f"  2030 Target: {result.final_result.target_2030_intensity:.2f}")
print(f"  Gap to Pathway: {result.final_result.gap_to_pathway:.1%}")
print(f"  SBTi Aligned: {result.final_result.sbti_aligned}")
```

---

## SBTi SDA Validation Guide

### Understanding SBTi SDA Requirements

The SBTi Sectoral Decarbonization Approach requires:

1. **Correct sector classification** -- company must be classified in one of 12 SDA sectors
2. **Physical intensity metric** -- sector-specific physical output metric (not revenue-based)
3. **Base year recency** -- base year within 2 most recent reporting years
4. **Coverage** -- 95% of Scope 1+2 emissions, 67% of Scope 3
5. **Convergence** -- intensity must converge to SBTi sector pathway by 2050
6. **Near-term ambition** -- 5-10 year target aligned with 1.5C
7. **Long-term** -- 90%+ absolute reduction by 2050

### Running SBTi Validation

```python
from workflows.pathway_validation_workflow import PathwayValidationWorkflow

workflow = PathwayValidationWorkflow(config=config)

result = workflow.execute(
    pathway_id=pathway_result.pathway_id,
    validation_scope="full",  # "full" or "near_term_only"
)

print(f"Validation Result: {'PASS' if result.sbti_aligned else 'FAIL'}")
print(f"Score: {result.validation_score}/100")

# Review individual criteria
for criterion in result.criteria:
    status = "PASS" if criterion.passed else "FAIL"
    print(f"  [{status}] {criterion.id}: {criterion.description}")
    if not criterion.passed:
        print(f"         Current: {criterion.current_value}")
        print(f"         Required: {criterion.required_value}")
        print(f"         Fix: {criterion.remediation}")
```

### SDA Validation Criteria Checklist

| # | Criterion | What PACK-028 Checks |
|---|-----------|---------------------|
| 1 | Sector classification correct | NACE/GICS/ISIC match to SDA sector |
| 2 | Intensity metric correct | Sector-specific physical metric used |
| 3 | Base year within 2 years | Base year is 2022 or later (for 2024 submission) |
| 4 | Scope 1+2 coverage >= 95% | Boundary completeness vs. total emissions |
| 5 | Scope 3 coverage >= 67% | Category materiality assessment |
| 6 | Convergence to sector pathway | Company intensity converges within +/-10% of SBTi pathway |
| 7 | Near-term target 5-10 years | Target year within valid window |
| 8 | 1.5C ambition for near-term | Reduction rate >= 4.2%/year ACA equivalent |
| 9 | Net-zero by 2050 | Long-term target achieves 90%+ reduction |
| 10 | Recalculation policy defined | Triggers and thresholds documented |

### Common Validation Failures and Fixes

**Failure: Coverage below 95%**
```
Fix: Expand organizational boundary to include all majority-owned entities.
     Review PACK-021 entity list for excluded subsidiaries.
```

**Failure: Intensity not converging**
```
Fix: Verify base year intensity against sector average.
     Ensure production forecast is realistic.
     Consider a more aggressive technology transition plan.
```

**Failure: Near-term ambition insufficient**
```
Fix: Increase near-term reduction actions.
     Add technology transitions in 2025-2030 window.
     Consider accelerating EAF transition or renewable procurement.
```

---

## Technology Roadmap Planning

### Building a Technology Roadmap

```python
from engines.technology_roadmap_engine import TechnologyRoadmapEngine

engine = TechnologyRoadmapEngine()

result = engine.build(
    sector="steel",
    pathway=pathway_result,
    current_technology_mix={
        "bf_bof": 0.75,
        "eaf_scrap": 0.20,
        "dri_natural_gas": 0.05,
    },
    installed_capacity={
        "total_mtpa": 5.0,
        "bf_bof_mtpa": 3.75,
        "eaf_mtpa": 1.0,
        "dri_mtpa": 0.25,
    },
    capex_budget_annual_usd=500_000_000,
    region="eu",
    planning_horizon=2050,
)

# Review technology transitions
print("Technology Transitions:")
for transition in result.transitions:
    print(f"\n  {transition.from_technology} -> {transition.to_technology}")
    print(f"    Period: {transition.start_year} - {transition.completion_year}")
    print(f"    CapEx: ${transition.capex_total_usd:,.0f}")
    print(f"    Emission Reduction: {transition.emission_reduction_tco2e_pa:,.0f} tCO2e/year")
    print(f"    TRL: {transition.trl}")
    print(f"    Confidence: {transition.confidence}")
    if transition.dependencies:
        print(f"    Dependencies: {', '.join(transition.dependencies)}")

# Review technology mix evolution
print("\nTechnology Mix Over Time:")
for year in [2025, 2030, 2035, 2040, 2045, 2050]:
    mix = result.get_technology_mix(year)
    print(f"\n  {year}:")
    for tech, share in sorted(mix.items(), key=lambda x: -x[1]):
        if share > 0.01:
            bar = "#" * int(share * 40)
            print(f"    {tech:<25} {share:>5.1%} {bar}")
```

### IEA Milestone Tracking

```python
# Review IEA milestone alignment
print("\nIEA Milestone Tracking:")
for milestone in result.iea_milestones:
    status_icon = "ON" if milestone.on_track else "OFF"
    print(f"  [{status_icon} TRACK] {milestone.year}: {milestone.description}")
    print(f"          Company Action: {milestone.company_action}")
```

### CapEx Schedule

```python
# Review CapEx schedule
print("\nCapEx Schedule:")
total_capex = 0
for year_capex in result.capex_schedule:
    total_capex += year_capex.amount_usd
    print(f"  {year_capex.year}: ${year_capex.amount_usd:>15,.0f}  "
          f"{year_capex.technology}: {year_capex.description}")
print(f"\n  Total CapEx: ${total_capex:,.0f}")
```

### Dependency Analysis

```python
# Review dependencies
print("\nTechnology Dependencies:")
for node in result.dependency_graph.nodes:
    print(f"  {node.id}: {node.status} "
          f"(available from {node.earliest_availability})")

for edge in result.dependency_graph.edges:
    print(f"  {edge.from_tech} --[{edge.type}]--> {edge.to_dependency}")
```

---

## Progress Monitoring

### Setting Up Monitoring

```python
from workflows.progress_monitoring_workflow import ProgressMonitoringWorkflow

workflow = ProgressMonitoringWorkflow(config=config)

# Run quarterly or annually
result = workflow.execute(
    pathway_id=pathway_result.pathway_id,
    current_data={
        "reporting_year": 2025,
        "production_tonnes": 5_100_000,
        "scope1_tco2e": 6_800_000,
        "scope2_location_tco2e": 1_400_000,
        "scope2_market_tco2e": 1_100_000,
        "technology_mix": {
            "bf_bof": 0.70,
            "eaf_scrap": 0.24,
            "dri_natural_gas": 0.06,
        },
    },
)

# Review progress
print(f"Current Intensity: {result.current_intensity:.2f} tCO2e/t")
print(f"Pathway Target for {result.current_year}: {result.pathway_target:.2f} tCO2e/t")
print(f"Gap: {result.gap_to_pathway:.1%}")
print(f"Status: {result.convergence_status}")
print(f"On Track for 2030: {result.on_track_2030}")
print(f"On Track for 2050: {result.on_track_2050}")

# Review benchmark position update
print(f"\nBenchmark Position:")
print(f"  Sector Percentile: {result.benchmark.percentile}th")
print(f"  vs. Sector Average: {result.benchmark.vs_average:+.2f} tCO2e/t")
print(f"  vs. SBTi Peers: {result.benchmark.vs_sbti_peers:+.2f} tCO2e/t")
```

### Setting Up Alerts

Configure alerts for pathway deviation:

```python
from config.pack_config import PackConfig

config.monitoring = {
    "alert_threshold_gap_pct": 0.05,    # Alert if gap exceeds 5%
    "alert_threshold_risk_level": "high", # Alert if risk reaches high
    "milestone_alerts": True,            # Alert on IEA milestone deadlines
    "benchmark_alerts": True,            # Alert on benchmark changes
    "notification_channels": ["email", "slack"],
    "notification_recipients": [
        "sustainability-team@company.com",
        "#sustainability-alerts",
    ],
}
```

---

## Multi-Scenario Analysis

### Running Multi-Scenario Comparison

```python
from workflows.multi_scenario_analysis_workflow import MultiScenarioAnalysisWorkflow

workflow = MultiScenarioAnalysisWorkflow(config=config)

result = workflow.execute(
    sector="steel",
    base_year=2023,
    base_year_intensity=1.85,
    scenarios=["nze_15c", "wb2c", "2c", "aps", "steps"],
    production_forecast={
        2023: 5_000_000,
        2030: 5_500_000,
        2040: 5_800_000,
        2050: 6_000_000,
    },
    region="global",
    include_investment_analysis=True,
    include_risk_assessment=True,
)

# Display scenario comparison matrix
print(f"{'Scenario':<20} {'2030':>8} {'2040':>8} {'2050':>8} {'Investment':>15}")
print("-" * 60)
for scenario in result.scenarios:
    print(f"{scenario.name:<20} "
          f"{scenario.target_2030:>8.2f} "
          f"{scenario.target_2040:>8.2f} "
          f"{scenario.target_2050:>8.2f} "
          f"${scenario.cumulative_investment:>13,.0f}")

# Display risk-return analysis
print(f"\n{'Scenario':<20} {'Trans. Risk':>12} {'Phys. Risk':>12} {'Position':>15}")
print("-" * 60)
for scenario in result.scenarios:
    print(f"{scenario.name:<20} "
          f"{scenario.transition_risk:>12} "
          f"{scenario.physical_risk:>12} "
          f"{scenario.positioning:>15}")

# Display optimal pathway recommendation
print(f"\nRecommended Pathway: {result.optimal_pathway.scenario}")
print(f"Rationale: {result.optimal_pathway.rationale}")
```

### Understanding Scenario Differences

| Scenario | 2030 Steel Intensity | 2050 Steel Intensity | Total Investment | Key Difference |
|----------|---------------------|---------------------|-----------------|----------------|
| NZE 1.5C | 1.25 tCO2e/t | 0.10 tCO2e/t | $5.5B | Fastest transition, green H2 DRI by 2030 |
| WB2C | 1.35 tCO2e/t | 0.20 tCO2e/t | $4.5B | Moderate pace, green H2 DRI by 2035 |
| 2C | 1.45 tCO2e/t | 0.35 tCO2e/t | $3.5B | Slower transition, CCS focus |
| APS | 1.50 tCO2e/t | 0.50 tCO2e/t | $2.8B | Policy-dependent, incremental |
| STEPS | 1.60 tCO2e/t | 0.80 tCO2e/t | $1.5B | Minimal change, stranded asset risk |

---

## Abatement Waterfall Analysis

### Building an Abatement Waterfall

```python
from engines.abatement_waterfall_engine import AbatementWaterfallEngine

engine = AbatementWaterfallEngine()

result = engine.analyze(
    sector="steel",
    pathway=pathway_result,
    current_emissions_tco2e=9_250_000,
    current_production_tonnes=5_000_000,
    sector_parameters={
        "bf_bof_share": 0.75,
        "eaf_share": 0.20,
        "dri_share": 0.05,
        "scrap_recycling_rate": 0.35,
        "energy_efficiency_gj_per_tonne": 20.5,
        "renewable_electricity_share": 0.15,
    },
    target_year=2030,
)

# Display waterfall
print(f"Abatement Waterfall: Steel Sector ({result.start_year} -> {result.target_year})")
print(f"Starting Emissions: {result.start_emissions_tco2e:,.0f} tCO2e")
print()

cumulative = 0
for lever in result.levers:
    cumulative += lever.reduction_tco2e
    cost_label = f"EUR {lever.cost_per_tco2e_eur:,.0f}/tCO2e"
    if lever.cost_per_tco2e_eur < 0:
        cost_label += " (SAVES MONEY)"

    bar_width = int(lever.reduction_pct * 100)
    bar = "=" * bar_width

    print(f"  {lever.name}")
    print(f"    Reduction: {lever.reduction_tco2e:>12,.0f} tCO2e ({lever.reduction_pct:>5.1%})")
    print(f"    Cost: {cost_label}")
    print(f"    Timeline: {lever.start_year}-{lever.end_year}")
    print(f"    Certainty: {lever.certainty}")
    print(f"    {bar}")
    print()

print(f"Ending Emissions: {result.end_emissions_tco2e:,.0f} tCO2e")
print(f"Total Reduction: {result.total_reduction_tco2e:,.0f} tCO2e ({result.total_reduction_pct:.1%})")
print(f"Residual: {result.residual_emissions_tco2e:,.0f} tCO2e ({result.residual_pct:.1%} of base)")
```

### Typical Steel Sector Levers (2023-2030)

| # | Lever | Reduction | Cost (EUR/tCO2e) | Certainty |
|---|-------|-----------|-----------------|-----------|
| 1 | Energy efficiency (waste heat recovery) | 3-5% | -10 to -20 (saves money) | High |
| 2 | Scrap recycling rate increase | 5-8% | -5 to +10 | High |
| 3 | BF-BOF to EAF transition | 10-15% | +20 to +50 | High |
| 4 | Renewable electricity procurement | 3-5% | +10 to +30 | High |
| 5 | Green hydrogen DRI (pilot) | 5-10% | +50 to +100 | Medium |
| 6 | CCS retrofit (pilot) | 5-8% | +60 to +120 | Medium |

---

## Sector Benchmarking

### Running a Benchmark Analysis

```python
from engines.sector_benchmark_engine import SectorBenchmarkEngine

engine = SectorBenchmarkEngine()

result = engine.benchmark(
    sector="steel",
    company_intensity=1.65,
    company_year=2025,
    region="eu",
    production_tonnes=5_000_000,
)

# Display benchmark summary
print(f"Sector: Steel")
print(f"Company Intensity: {result.company_intensity:.2f} tCO2e/t")
print(f"Company Percentile: {result.percentile_rank}th")
print()

print(f"{'Benchmark':<25} {'Value':>8} {'Gap':>10} {'Status':<20}")
print("-" * 65)
for name, benchmark in result.benchmarks.items():
    gap_str = f"{benchmark.company_vs_benchmark:+.2f}"
    print(f"{name:<25} {benchmark.value:>8.2f} {gap_str:>10} {benchmark.company_status:<20}")
```

### Interpreting Benchmark Results

- **Below Average**: Your intensity is lower than the sector average (positive for climate)
- **Above Average**: Your intensity is higher than the sector average (needs improvement)
- **Above Leader**: Your intensity is above the top-decile performers
- **Above Pathway**: Your intensity is above the IEA NZE pathway target for the current year
- **Below Pathway**: Your intensity is below the IEA pathway (you are ahead of the pathway)

---

## Report Generation

### Available Reports

| Report | Use Case | Typical Audience |
|--------|---------|-----------------|
| Sector Pathway Report | Full sector pathway analysis | Strategy team, SBTi submission |
| Intensity Convergence Report | Progress tracking | Quarterly review meetings |
| Technology Roadmap Report | Technology planning | CapEx committee, board |
| Abatement Waterfall Report | Lever-by-lever analysis | Operations team, strategy |
| Sector Benchmark Report | Competitive positioning | Board, investor relations |
| Scenario Comparison Report | Strategic planning | Board, risk committee |
| SBTi Validation Report | SBTi submission | SBTi submission team |
| Sector Strategy Report | Executive summary | CEO, board, investors |

### Generating Reports

```python
from templates.sector_pathway_report import SectorPathwayReport

report = SectorPathwayReport()
output = report.render(
    pathway=pathway_result,
    convergence=convergence_result,
    format="html",
    branding={
        "company_name": "SteelCorp International",
        "logo_url": "https://steelcorp.com/logo.png",
        "report_date": "2026-03-19",
        "confidentiality": "Board Confidential",
    },
)

# Save to file
with open("steel_pathway_report.html", "w") as f:
    f.write(output.content)
print(f"Report generated: {output.content_length_bytes:,} bytes")
```

### Batch Report Generation

```python
from templates.sector_strategy_report import SectorStrategyReport

# Generate executive strategy report combining all analyses
report = SectorStrategyReport()
output = report.render(
    classification=classification_result,
    pathway=pathway_result,
    convergence=convergence_result,
    technology_roadmap=roadmap_result,
    abatement_waterfall=waterfall_result,
    benchmark=benchmark_result,
    scenario_comparison=scenario_result,
    format="pdf",
)
```

---

## Best Practices

### Data Quality

1. **Use physical activity data** wherever possible (production volumes, energy consumption, fuel quantities) rather than financial proxies.
2. **Verify base year intensity** against published sector averages to catch data errors early.
3. **Document assumptions** for all production forecasts, especially post-2030 projections.
4. **Update intensity annually** with verified data to maintain convergence tracking accuracy.

### Pathway Design

1. **Start with the correct sector** -- misclassification will produce incorrect pathways. Use the debug mode on the classification engine if uncertain.
2. **Choose convergence model carefully** -- S-curve is appropriate for technology-driven sectors (steel, cement); linear for policy-driven sectors (buildings).
3. **Use realistic production forecasts** -- overly optimistic growth assumptions will overestimate absolute emissions even with intensity improvements.
4. **Validate against SBTi tool** -- cross-check PACK-028 pathway outputs with the official SBTi Target Setting Tool for submission.

### Technology Roadmaps

1. **Account for technology dependencies** -- green hydrogen DRI requires renewable electricity; CCS requires CO2 transport infrastructure.
2. **Apply conservative TRL assessments** -- technologies below TRL 7 should have longer timelines and wider cost uncertainty ranges.
3. **Phase CapEx realistically** -- align technology investments with equipment replacement cycles and plant maintenance windows.
4. **Track IEA milestones quarterly** -- milestone tracking provides early warning of pathway deviation.

### Scenario Analysis

1. **Include at least 3 scenarios** -- NZE 1.5C (ambitious), 2C (moderate), and STEPS (baseline) provide useful strategic boundaries.
2. **Review sensitivity drivers** -- understand which parameters most influence scenario divergence to focus strategic planning.
3. **Update scenario analysis annually** -- IEA updates scenario data periodically; refresh your analysis with the latest data.

### Reporting

1. **Generate board reports quarterly** -- maintain visibility on pathway progress and benchmark position.
2. **Use SBTi validation reports** for internal review before official submission.
3. **Include provenance hashes** in all external disclosures for audit trail integrity.

---

## Appendix: Sector Reference Data

### SBTi SDA Convergence Targets (NZE 1.5C)

| Sector | 2020 Global Avg | 2030 Target | 2040 Target | 2050 Target |
|--------|----------------|-------------|-------------|-------------|
| Power (gCO2/kWh) | 450 | 220 | 60 | 0 |
| Steel (tCO2e/t) | 1.85 | 1.25 | 0.55 | 0.10 |
| Cement (tCO2e/t) | 0.62 | 0.47 | 0.25 | 0.04 |
| Aluminum (tCO2e/t) | 11.5 | 7.5 | 3.5 | 0.5 |
| Chemicals (tCO2e/t) | 1.20 | 0.85 | 0.45 | 0.10 |
| Pulp & Paper (tCO2e/t) | 0.45 | 0.30 | 0.15 | 0.02 |
| Aviation (gCO2/pkm) | 90 | 72 | 45 | 15 |
| Shipping (gCO2/tkm) | 9.0 | 6.5 | 3.0 | 0.5 |
| Road Transport (gCO2/vkm) | 180 | 120 | 50 | 5 |
| Rail (gCO2/pkm) | 25 | 15 | 5 | 0 |
| Buildings Res. (kgCO2/m2/yr) | 30 | 18 | 8 | 1 |
| Buildings Com. (kgCO2/m2/yr) | 45 | 25 | 10 | 2 |

**Note:** These values are illustrative and aligned with SBTi SDA Tool V3.0 and IEA NZE 2050 (2023 update). Actual SBTi convergence factors are loaded from the SBTi SDA Bridge integration and may be updated with new SBTi tool versions.

### IEA NZE Key Milestones Summary

| Year | Power | Industry | Transport | Buildings |
|------|-------|----------|-----------|-----------|
| 2025 | No new unabated coal | First H2 DRI plant | 20M EVs/year | All new buildings NZE-ready |
| 2030 | 60% renewables | 10% low-carbon steel | 60% EV sales | Heat pump sales surpass gas boilers |
| 2035 | No unabated coal (OECD) | CCS at scale | No new ICE cars | 50% existing buildings retrofitted |
| 2040 | 80% clean power | 30% near-zero industry | 50% SAF blend | 70% existing buildings retrofitted |
| 2050 | Net-zero power | Near-zero industry | Net-zero transport | Net-zero buildings |

---

**End of User Guide**
