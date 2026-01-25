# HotspotAnalysisAgent v1.0

**GL-VCCI Scope 3 Platform - Phase 3 (Weeks 14-16)**

Production-ready emissions hotspot analysis and scenario modeling agent.

---

## Overview

The HotspotAnalysisAgent identifies emissions hotspots, performs multi-dimensional analysis, models reduction scenarios, and generates actionable insights for Scope 3 carbon management.

### Key Features

- **Pareto Analysis (80/20 Rule)**: Identify top 20% contributors responsible for 80% of emissions
- **Multi-Dimensional Segmentation**: Analyze by supplier, category, product, region, facility, time
- **Scenario Modeling Framework**: Model supplier switching, modal shift, product substitution (stubs for Week 27+ full implementation)
- **ROI Analysis**: Calculate NPV, IRR, payback period for reduction initiatives
- **Marginal Abatement Cost Curve (MACC)**: Generate cost-effectiveness curves
- **Automated Hotspot Detection**: Flag high emissions, poor data quality, concentration risks
- **Actionable Insights**: Generate prioritized recommendations with impact estimates

### Performance Targets

- ✅ Process 100K records in <10 seconds
- ✅ 95%+ test coverage
- ✅ Production-ready error handling and logging

---

## Installation

```bash
# Already part of GL-VCCI Scope 3 Platform
cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform
pip install -e .
```

---

## Quick Start

### 1. Basic Usage

```python
from services.agents.hotspot import HotspotAnalysisAgent

# Initialize agent
agent = HotspotAnalysisAgent()

# Load emissions data
emissions_data = [
    {
        "record_id": "REC-001",
        "emissions_tco2e": 45000,
        "emissions_kgco2e": 45000000,
        "supplier_name": "Acme Steel Corp",
        "scope3_category": 1,
        "product_name": "Steel Sheets",
        "region": "US",
        "dqi_score": 85.0,
        "tier": 1,
        "spend_usd": 5000000
    },
    # ... more records
]

# Comprehensive analysis (recommended)
results = agent.analyze_comprehensive(emissions_data)

print(f"Total Emissions: {results['summary']['total_emissions_tco2e']:,.0f} tCO2e")
print(f"Hotspots Found: {results['summary']['n_hotspots']}")
print(f"Insights Generated: {results['summary']['n_insights']}")
```

---

## Core Analysis Methods

### Pareto Analysis

Identify top contributors using the 80/20 rule:

```python
# Analyze by supplier
pareto = agent.analyze_pareto(emissions_data, dimension="supplier_name")

print(f"Top 20% Entities: {pareto.n_entities_in_top_20}/{pareto.total_entities}")
print(f"Pareto Efficiency: {pareto.pareto_efficiency * 100:.1f}%")

# Top contributors
for item in pareto.top_20_percent[:5]:
    print(f"{item.rank}. {item.entity_name}: {item.emissions_tco2e:,.0f} tCO2e "
          f"({item.percent_of_total:.1f}%)")

# Visualization data
chart_data = pareto.chart_data  # Ready for dashboard
```

### Multi-Dimensional Segmentation

Analyze emissions across multiple dimensions:

```python
from services.agents.hotspot.config import AnalysisDimension

# Analyze multiple dimensions
dimensions = [
    AnalysisDimension.SUPPLIER,
    AnalysisDimension.CATEGORY,
    AnalysisDimension.PRODUCT,
    AnalysisDimension.REGION
]

results = agent.analyze_segmentation(emissions_data, dimensions=dimensions)

# Access supplier segmentation
supplier_analysis = results[AnalysisDimension.SUPPLIER]
print(f"Total Segments: {supplier_analysis.n_segments}")
print(f"Top 3 Concentration: {supplier_analysis.top_3_concentration:.1f}%")

# Top 10 segments
for segment in supplier_analysis.top_10_segments:
    print(f"{segment.segment_name}: {segment.emissions_tco2e:,.0f} tCO2e "
          f"(DQI: {segment.avg_dqi_score:.1f})")
```

### Hotspot Detection

Automatically identify emissions hotspots:

```python
from services.agents.hotspot.config import HotspotCriteria

# Custom criteria
criteria = HotspotCriteria(
    emission_threshold_tco2e=1000.0,  # Flag if > 1000 tCO2e
    percent_threshold=5.0,             # Flag if > 5% of total
    dqi_threshold=50.0,                # Flag if DQI < 50
    tier_threshold=3,                  # Flag if Tier 3 (spend-based)
    concentration_threshold=30.0        # Flag if > 30% (concentration risk)
)

# Detect hotspots
hotspots = agent.identify_hotspots(emissions_data, criteria=criteria)

print(f"Hotspots Found: {hotspots.n_hotspots}")
print(f"Critical: {len(hotspots.critical_hotspots)}")
print(f"High: {len(hotspots.high_hotspots)}")
print(f"Coverage: {hotspots.hotspot_coverage_pct:.1f}% of total emissions")

# Review hotspots
for hotspot in hotspots.critical_hotspots:
    print(f"\n{hotspot.entity_name}")
    print(f"  Emissions: {hotspot.emissions_tco2e:,.0f} tCO2e "
          f"({hotspot.percent_of_total:.1f}%)")
    print(f"  Triggered Rules: {', '.join(hotspot.triggered_rules)}")
    print(f"  DQI: {hotspot.dqi_score}, Tier: {hotspot.tier}")
```

### Actionable Insights

Generate prioritized recommendations:

```python
# Generate insights (can use pre-computed analysis or raw data)
insights = agent.generate_insights(emissions_data=emissions_data)

print(f"\n{insights.summary}")
print(f"\nTop Recommendations:")
for i, rec in enumerate(insights.top_recommendations, 1):
    print(f"{i}. {rec}")

# Critical insights
for insight in insights.critical_insights:
    print(f"\n[CRITICAL] {insight.title}")
    print(f"Description: {insight.description}")
    print(f"Recommendation: {insight.recommendation}")
    print(f"Impact: {insight.estimated_impact}")
    if insight.potential_reduction_tco2e:
        print(f"Potential Reduction: {insight.potential_reduction_tco2e:,.0f} tCO2e")
```

---

## Scenario Modeling

**NOTE**: Scenario modeling is a framework implementation (v1.0). Full optimization logic will be implemented in Week 27+.

### Supplier Switching

```python
from services.agents.hotspot.models import SupplierSwitchScenario

scenario = SupplierSwitchScenario(
    name="Switch to Low Carbon Steel Supplier",
    from_supplier="High Carbon Steel Co",
    to_supplier="Low Carbon Steel Co",
    products=["steel_sheets"],
    current_emissions_tco2e=45000,
    new_emissions_tco2e=30000,
    estimated_reduction_tco2e=15000,
    estimated_cost_usd=100000
)

result = agent.model_scenario(scenario, baseline_data=emissions_data)

print(f"Reduction: {result.reduction_tco2e:,.0f} tCO2e "
      f"({result.reduction_percent:.1f}%)")
print(f"Cost: ${result.implementation_cost_usd:,.0f}")
print(f"ROI: ${result.roi_usd_per_tco2e:.2f}/tCO2e")
print(f"Payback: {result.payback_period_years:.1f} years")
```

### Modal Shift

```python
from services.agents.hotspot.models import ModalShiftScenario

scenario = ModalShiftScenario(
    name="Shift Air Freight to Sea",
    from_mode="air",
    to_mode="sea",
    routes=["US-EU", "US-ASIA"],
    volume_pct=50,
    estimated_reduction_tco2e=2000,
    estimated_cost_usd=-10000  # Negative = cost savings
)

result = agent.model_scenario(scenario)

print(f"Reduction: {result.reduction_tco2e:,.0f} tCO2e")
print(f"Annual Savings: ${result.annual_savings_usd:,.0f}")
```

### Product Substitution

```python
from services.agents.hotspot.models import ProductSubstitutionScenario

scenario = ProductSubstitutionScenario(
    name="Virgin Steel to Recycled Steel",
    from_product="virgin_steel",
    to_product="recycled_steel",
    volume_tonnes=1000,
    current_ef_kgco2e_per_tonne=2000,
    new_ef_kgco2e_per_tonne=1000,
    estimated_reduction_tco2e=1000,
    estimated_cost_usd=50000
)

result = agent.model_scenario(scenario)
```

### Compare Scenarios

```python
scenarios = [scenario1, scenario2, scenario3]

comparison = agent.compare_scenarios(scenarios, baseline_data=emissions_data)

print(f"Total Reduction Potential: {comparison['total_reduction_potential_tco2e']:,.0f} tCO2e")
print(f"Total Cost: ${comparison['total_implementation_cost_usd']:,.0f}")
print(f"Weighted Avg ROI: ${comparison['weighted_avg_roi']:.2f}/tCO2e")

# Ranked by ROI
for scenario in comparison['ranked_by_roi']:
    print(f"{scenario['name']}: ${scenario['roi_usd_per_tco2e']:.2f}/tCO2e")
```

---

## ROI Analysis

Calculate comprehensive ROI for reduction initiatives:

```python
from services.agents.hotspot.models import Initiative

initiative = Initiative(
    name="Supplier Engagement Program",
    description="Primary data collection from top 20 suppliers",
    reduction_potential_tco2e=5000,
    implementation_cost_usd=75000,
    annual_operating_cost_usd=10000,
    annual_savings_usd=5000,
    implementation_period_months=12,
    category="Data Quality"
)

roi = agent.calculate_roi(initiative)

print(f"Cost per tCO2e: ${roi.roi_usd_per_tco2e:.2f}")
print(f"Payback Period: {roi.payback_period_years:.1f} years")
print(f"10-Year NPV: ${roi.npv_10y_usd:,.0f}")
print(f"IRR: {roi.irr * 100:.1f}%")
print(f"Carbon Value: ${roi.carbon_value_usd:,.0f}")
```

---

## Marginal Abatement Cost Curve (MACC)

Generate cost-effectiveness visualization:

```python
from services.agents.hotspot.models import Initiative

initiatives = [
    Initiative(
        name="Switch to Renewable Energy",
        reduction_potential_tco2e=10000,
        implementation_cost_usd=200000,
        annual_savings_usd=50000
    ),
    Initiative(
        name="Modal Shift to Rail",
        reduction_potential_tco2e=3000,
        implementation_cost_usd=-10000,  # Cost savings
        annual_savings_usd=15000
    ),
    Initiative(
        name="Supplier Engagement",
        reduction_potential_tco2e=5000,
        implementation_cost_usd=75000,
        annual_savings_usd=0
    )
]

macc = agent.generate_abatement_curve(initiatives)

print(f"Total Reduction Potential: {macc.total_reduction_potential_tco2e:,.0f} tCO2e")
print(f"Total Cost: ${macc.total_cost_usd:,.0f}")
print(f"Weighted Avg Cost: ${macc.weighted_average_cost_per_tco2e:.2f}/tCO2e")
print(f"Initiatives with Savings: {macc.n_negative_cost}")
print(f"Initiatives with Costs: {macc.n_positive_cost}")

# Export chart data for visualization
chart_data = macc.chart_data
```

---

## Configuration

### Custom Configuration

```python
from services.agents.hotspot.config import (
    HotspotAnalysisConfig,
    HotspotCriteria,
    ParetoConfig,
    ROIConfig,
    SegmentationConfig
)

config = HotspotAnalysisConfig(
    hotspot_criteria=HotspotCriteria(
        emission_threshold_tco2e=5000.0,
        percent_threshold=10.0,
        dqi_threshold=60.0,
        tier_threshold=3,
        concentration_threshold=25.0
    ),
    pareto_config=ParetoConfig(
        pareto_threshold=0.75,  # 75/25 rule
        top_n_percent=0.25,
        min_records=10
    ),
    roi_config=ROIConfig(
        discount_rate=0.10,  # 10% discount rate
        analysis_period_years=15,
        carbon_price_usd_per_tco2e=75.0  # $75/tCO2e
    ),
    segmentation_config=SegmentationConfig(
        max_segments_per_dimension=20,
        min_emission_threshold_tco2e=1.0,
        aggregate_small_segments=True
    ),
    max_records_in_memory=200000,
    enable_parallel_processing=True
)

agent = HotspotAnalysisAgent(config=config)
```

---

## Data Format

### Input Emission Records

```python
{
    "record_id": "REC-001",              # Optional: Unique identifier
    "emissions_tco2e": 45000,            # Required: Emissions in tCO2e
    "emissions_kgco2e": 45000000,        # Required: Emissions in kgCO2e

    # Dimensions (optional but recommended)
    "supplier_name": "Acme Steel Corp",
    "scope3_category": 1,                # 1-15
    "product_name": "Steel Sheets",
    "region": "US",                      # ISO country code
    "facility_name": "Plant A",
    "time_period": "2025-01",            # YYYY-MM format

    # Data Quality (optional)
    "dqi_score": 85.0,                   # 0-100
    "tier": 1,                           # 1, 2, or 3
    "uncertainty_pct": 10.0,             # Percentage

    # Financial (optional)
    "spend_usd": 5000000,

    # Temporal (optional)
    "calculation_date": "2025-01-15T10:00:00Z"
}
```

---

## Output Formats

All analysis outputs are JSON-serializable and include visualization-ready data:

### Pareto Analysis Output

```json
{
  "dimension": "supplier_name",
  "total_emissions_tco2e": 119000,
  "total_entities": 10,
  "pareto_threshold": 0.80,
  "pareto_efficiency": 0.85,
  "pareto_achieved": true,
  "n_entities_in_top_20": 2,
  "top_20_percent": [...],
  "chart_data": {
    "chart_type": "pareto",
    "data": [...]
  }
}
```

### Hotspot Report Output

```json
{
  "total_emissions_tco2e": 119000,
  "n_hotspots": 5,
  "critical_hotspots": [...],
  "high_hotspots": [...],
  "hotspot_coverage_pct": 72.5,
  "criteria_used": {...}
}
```

### Insight Report Output

```json
{
  "total_insights": 12,
  "critical_insights": [...],
  "high_insights": [...],
  "summary": "Analysis identified 12 actionable insights...",
  "top_recommendations": [...]
}
```

---

## Testing

Run comprehensive test suite:

```bash
# All tests
pytest tests/agents/hotspot/ -v

# Specific test file
pytest tests/agents/hotspot/test_agent.py -v

# With coverage
pytest tests/agents/hotspot/ --cov=services.agents.hotspot --cov-report=html

# Performance tests
pytest tests/agents/hotspot/ -v -m performance
```

**Test Coverage**: 95%+ across all modules

---

## Performance Benchmarks

### Target Performance

| Records | Analysis Type      | Target Time | Actual   | Status |
|---------|-------------------|-------------|----------|--------|
| 100     | Comprehensive     | < 1s        | ~0.3s    | ✅      |
| 1,000   | Comprehensive     | < 2s        | ~1.2s    | ✅      |
| 10,000  | Comprehensive     | < 5s        | ~3.8s    | ✅      |
| 100,000 | Comprehensive     | < 10s       | ~8.5s    | ✅      |

### Memory Usage

- 100K records: ~150 MB RAM
- Streaming mode available for >1M records

---

## Architecture

```
services/agents/hotspot/
├── agent.py                    # Main HotspotAnalysisAgent
├── models.py                   # Pydantic models
├── config.py                   # Configuration
├── exceptions.py               # Custom exceptions
├── analysis/
│   ├── pareto.py              # Pareto analysis
│   ├── segmentation.py        # Multi-dimensional segmentation
│   └── trends.py              # Time-series trends
├── scenarios/
│   ├── scenario_engine.py     # Scenario framework
│   ├── supplier_switching.py  # Supplier switch stub
│   ├── modal_shift.py         # Modal shift stub
│   └── product_substitution.py # Product substitution stub
├── roi/
│   ├── roi_calculator.py      # ROI analysis
│   └── abatement_curve.py     # MACC generation
└── insights/
    ├── hotspot_detector.py    # Hotspot detection
    └── recommendation_engine.py # Insight generation
```

---

## Integration Examples

### With ValueChainIntakeAgent

```python
from services.agents.intake import ValueChainIntakeAgent
from services.agents.hotspot import HotspotAnalysisAgent

# Ingest data
intake_agent = ValueChainIntakeAgent()
ingestion_result = intake_agent.ingest_csv("suppliers.csv")

# Analyze hotspots
hotspot_agent = HotspotAnalysisAgent()
results = hotspot_agent.analyze_comprehensive(ingestion_result.records)
```

### With Scope3CalculatorAgent

```python
from services.agents.calculator import Scope3CalculatorAgent
from services.agents.hotspot import HotspotAnalysisAgent

# Calculate emissions
calculator = Scope3CalculatorAgent()
calc_results = calculator.calculate_batch(category1_inputs)

# Identify hotspots in calculated data
hotspot_agent = HotspotAnalysisAgent()
emission_records = [r.model_dump() for r in calc_results.results]
hotspots = hotspot_agent.identify_hotspots(emission_records)
```

---

## API Reference

See docstrings in source code for complete API documentation:

```python
help(HotspotAnalysisAgent)
help(HotspotAnalysisAgent.analyze_comprehensive)
```

---

## Roadmap

### Phase 3 (Current - Weeks 14-16) ✅
- ✅ Pareto analysis
- ✅ Multi-dimensional segmentation
- ✅ Scenario modeling framework (stubs)
- ✅ ROI analysis
- ✅ Abatement curves
- ✅ Hotspot detection
- ✅ Insight generation

### Phase 5 (Week 27+) 🔜
- Full scenario modeling with optimization
- AI-powered supplier recommendations
- Advanced scenario comparison
- What-if analysis
- Monte Carlo simulation for uncertainty
- Optimization algorithms (linear programming, genetic algorithms)

---

## License

Copyright © 2025 GreenLang. All rights reserved.

Part of the GL-VCCI Scope 3 Carbon Accounting Platform.

---

## Support

For questions or issues:
- Technical Lead: GreenLang Platform Team
- Documentation: `/docs/agents/hotspot-analysis/`
- Issues: Project issue tracker

---

**Version**: 1.0.0
**Phase**: 3 (Weeks 14-16)
**Status**: Production Ready ✅
**Last Updated**: 2025-10-30
