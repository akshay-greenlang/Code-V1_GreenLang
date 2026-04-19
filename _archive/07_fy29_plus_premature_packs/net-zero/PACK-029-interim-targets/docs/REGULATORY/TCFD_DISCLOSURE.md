# Regulatory Compliance Guide: TCFD Metrics and Targets Disclosure

**Pack:** PACK-029 Interim Targets Pack
**Version:** 1.0.0
**Standard:** TCFD Recommendations (June 2017, Updated October 2021)
**Focus:** Pillar 4 -- Metrics and Targets

---

## Table of Contents

1. [Overview](#overview)
2. [TCFD Framework Summary](#tcfd-framework-summary)
3. [Metrics and Targets Pillar Requirements](#metrics-and-targets-pillar-requirements)
4. [Recommended Disclosures](#recommended-disclosures)
5. [Scope 1, 2, and 3 Emissions Disclosure](#scope-1-2-and-3-emissions-disclosure)
6. [Target Disclosure Requirements](#target-disclosure-requirements)
7. [Performance Against Targets](#performance-against-targets)
8. [Scenario Analysis Integration](#scenario-analysis-integration)
9. [PACK-029 TCFD Export Mapping](#pack-029-tcfd-export-mapping)
10. [Cross-Industry Metrics](#cross-industry-metrics)
11. [Sector-Specific Considerations](#sector-specific-considerations)
12. [ISSB IFRS S2 Alignment](#issb-ifrs-s2-alignment)
13. [Best Practices](#best-practices)

---

## Overview

The Task Force on Climate-related Financial Disclosures (TCFD) provides a framework for companies to disclose climate-related financial information. PACK-029 focuses on the fourth TCFD pillar -- **Metrics and Targets** -- which requires organizations to disclose the metrics and targets used to assess and manage climate-related risks and opportunities.

### TCFD Adoption Status

| Jurisdiction | Status | Mandatory? |
|-------------|--------|------------|
| United Kingdom | Required for premium-listed companies | Yes (since 2022) |
| European Union | Incorporated into CSRD/ESRS | Yes (phased from 2024) |
| Japan | Required for prime market listings | Yes (since 2023) |
| Singapore | Required for SGX-listed companies | Yes (since 2024) |
| Hong Kong | Required for HKEX-listed companies | Yes (phased from 2025) |
| United States | SEC climate rule (scope varies) | Partial (litigation pending) |
| Australia | Mandatory climate disclosure | Yes (from 2025) |
| Canada | Required for federally regulated entities | Yes (from 2024) |

### PACK-029 TCFD Coverage

| TCFD Pillar | Coverage | PACK-029 Components |
|-------------|----------|---------------------|
| Governance | Partial (target governance) | SBTi Validation Engine (C20) |
| Strategy | Partial (scenario targets) | Trend Extrapolation Engine |
| Risk Management | Not primary focus | Via PACK-028 Bridge |
| **Metrics and Targets** | **100%** | **All 10 engines** |

---

## TCFD Framework Summary

### Four Pillars

```
TCFD Recommendations
    |
    +-- 1. GOVERNANCE
    |      Describe the organization's governance around
    |      climate-related risks and opportunities
    |
    +-- 2. STRATEGY
    |      Describe the actual and potential impacts of
    |      climate-related risks and opportunities
    |
    +-- 3. RISK MANAGEMENT
    |      Describe how the organization identifies,
    |      assesses, and manages climate-related risks
    |
    +-- 4. METRICS AND TARGETS   <-- PACK-029 focus
           Disclose the metrics and targets used to assess
           and manage relevant climate-related risks
```

### Pillar 4: Metrics and Targets -- Recommended Disclosures

| Disclosure | Reference | Description |
|-----------|-----------|-------------|
| **a)** | Metrics | Disclose the metrics used to assess climate-related risks and opportunities |
| **b)** | GHG Emissions | Disclose Scope 1, Scope 2, and (if appropriate) Scope 3 GHG emissions and related risks |
| **c)** | Targets | Describe the targets used and performance against targets |

---

## Metrics and Targets Pillar Requirements

### Disclosure 4(a): Climate-Related Metrics

Organizations should disclose metrics used to assess climate-related risks and opportunities in line with their strategy and risk management process:

| Metric Category | Examples | PACK-029 Source |
|-----------------|----------|-----------------|
| GHG emissions | Total Scope 1, 2, 3 (tCO2e) | MRV Bridge |
| Carbon intensity | tCO2e per revenue unit | Interim Target Engine |
| Energy use | Total energy consumption (GJ) | MRV Bridge |
| Renewable energy | % of total energy from renewables | MRV Bridge |
| Water use | Total water withdrawal | Not covered (future) |
| Internal carbon price | USD per tCO2e | Configuration |
| Climate-related investment | Capital allocated to decarbonization | Corrective Action Engine |
| Temperature score | Implied warming pathway | Interim Target Engine |

### Disclosure 4(b): GHG Emissions

PACK-029 provides GHG emissions data through the MRV Bridge:

```json
{
  "tcfd_emissions_disclosure": {
    "reporting_year": 2024,
    "scope_1": {
      "total_tco2e": 50000,
      "methodology": "GHG Protocol Corporate Standard",
      "consolidation_approach": "Operational control",
      "verification_status": "Third-party limited assurance",
      "year_over_year_change_pct": -5.2
    },
    "scope_2": {
      "location_based_tco2e": 35000,
      "market_based_tco2e": 25000,
      "methodology": "GHG Protocol Scope 2 Guidance",
      "year_over_year_change_pct": -12.5
    },
    "scope_3": {
      "total_tco2e": 115000,
      "categories_reported": [
        {"category": 1, "name": "Purchased goods and services", "tco2e": 45000},
        {"category": 4, "name": "Upstream transportation", "tco2e": 22000},
        {"category": 6, "name": "Business travel", "tco2e": 8000},
        {"category": 7, "name": "Employee commuting", "tco2e": 5000},
        {"category": 11, "name": "Use of sold products", "tco2e": 35000}
      ],
      "categories_screened": 15,
      "categories_material": 5,
      "coverage_pct": 72.3,
      "year_over_year_change_pct": -3.8
    }
  }
}
```

### Disclosure 4(c): Targets and Performance

PACK-029 is the primary source for targets and performance data:

```json
{
  "tcfd_targets_disclosure": {
    "targets": [
      {
        "description": "Reduce Scope 1+2 emissions 46.5% by 2030 from 2021 baseline",
        "type": "Absolute GHG emissions reduction",
        "scope": "Scope 1 and 2",
        "metric": "tCO2e",
        "base_year": 2021,
        "base_year_emissions": 80000,
        "target_year": 2030,
        "target_emissions": 42800,
        "target_reduction_pct": 46.5,
        "science_based": true,
        "alignment": "1.5C (SBTi validated)",
        "current_year_emissions": 68000,
        "progress_pct": 32.3,
        "on_track": true,
        "interim_milestones": [
          {"year": 2025, "target": 63200, "actual": null},
          {"year": 2027, "target": 53000, "actual": null}
        ]
      },
      {
        "description": "Achieve net-zero emissions by 2050 across all scopes",
        "type": "Net-zero target",
        "scope": "Scope 1, 2, and 3",
        "metric": "tCO2e",
        "base_year": 2021,
        "base_year_emissions": 200000,
        "target_year": 2050,
        "target_emissions": 20000,
        "target_reduction_pct": 90.0,
        "neutralization_plan": "Permanent CDR for residual 10%",
        "science_based": true,
        "alignment": "SBTi Corporate Net-Zero Standard"
      }
    ],
    "performance_narrative": "In 2024, total Scope 1+2 emissions decreased 15% from the 2021 baseline, driven primarily by renewable electricity procurement (-6,500 tCO2e) and energy efficiency improvements (-5,500 tCO2e). The company is on track to meet its near-term 2030 target of 46.5% reduction. LMDI variance analysis shows that despite 12% business growth (+9,600 tCO2e), emission intensity improvements (-21,600 tCO2e) more than offset the activity effect."
  }
}
```

---

## Recommended Disclosures

### Minimum Disclosure Elements

| Element | Required | PACK-029 Source |
|---------|----------|-----------------|
| Scope 1 emissions (tCO2e) | Yes | MRV Bridge |
| Scope 2 emissions (location-based) | Yes | MRV Bridge |
| Scope 2 emissions (market-based) | Recommended | MRV Bridge |
| Scope 3 emissions (material categories) | Recommended (many jurisdictions: required) | MRV Bridge |
| Emissions reduction target | Yes | Interim Target Engine |
| Progress against target | Yes | Annual Review Engine |
| Internal carbon price (if used) | Recommended | Configuration |
| Climate-related capital expenditure | Recommended | Corrective Action Engine |
| Temperature alignment | Recommended | Interim Target Engine |

### Enhanced Disclosure Elements (for Leadership)

| Element | Value Added | PACK-029 Source |
|---------|------------|-----------------|
| LMDI variance decomposition | Shows root causes of change | Variance Analysis Engine |
| Quarterly progress trajectory | Demonstrates active monitoring | Quarterly Monitoring Engine |
| Corrective action details | Shows management response | Corrective Action Engine |
| Carbon budget status | Shows cumulative perspective | Carbon Budget Tracker |
| Trend forecasts with CIs | Shows expected trajectory | Trend Extrapolation Engine |
| SBTi 21-criteria status | Demonstrates rigor | SBTi Validation Engine |
| Multi-scenario projections | Shows resilience | Three-Scenario Analysis |

---

## Scope 1, 2, and 3 Emissions Disclosure

### Disclosure Format

PACK-029 generates TCFD-compliant emissions disclosure:

```
Emissions Disclosure Table
==========================

| Metric                              | 2024      | 2023      | Change  |
|-------------------------------------|-----------|-----------|---------|
| Scope 1 (tCO2e)                     | 47,400    | 50,000    | -5.2%   |
| Scope 2 - Location-based (tCO2e)    | 30,600    | 35,000    | -12.6%  |
| Scope 2 - Market-based (tCO2e)      | 20,600    | 25,000    | -17.6%  |
| Scope 1+2 Market-based (tCO2e)      | 68,000    | 75,000    | -9.3%   |
| Scope 3 (tCO2e)                     | 115,000   | 119,500   | -3.8%   |
| Total (Market-based, tCO2e)         | 183,000   | 194,500   | -5.9%   |
| Revenue (M USD)                     | 560       | 500       | +12.0%  |
| Intensity (tCO2e / M USD revenue)   | 326.8     | 389.0     | -16.0%  |
```

### Methodology Notes

PACK-029 includes methodology notes with each TCFD export:

```
Methodology Notes:
1. GHG emissions calculated in accordance with the GHG Protocol Corporate Standard
2. Scope 2 reported under both location-based and market-based approaches per GHG Protocol Scope 2 Guidance
3. Scope 3 covers Categories 1, 4, 6, 7, and 11 representing 72.3% of total Scope 3
4. Global Warming Potentials from IPCC AR6
5. Organizational boundary set using operational control approach
6. All calculations performed using deterministic Decimal arithmetic with SHA-256 provenance hashing
7. Emissions data subject to limited assurance by [Assurance Provider]
```

---

## Target Disclosure Requirements

### Target Attributes

The TCFD recommends disclosing the following for each target:

| Attribute | Example | PACK-029 Field |
|-----------|---------|----------------|
| Whether absolute or intensity | Absolute | `target_type` |
| Time frame | 2030 | `target_year` |
| Base year | 2021 | `base_year` |
| Key performance indicators (KPIs) | tCO2e, tCO2e/M USD | `metric` |
| Target against base year | -46.5% | `reduction_pct` |
| Interim targets | -20% by 2025, -35% by 2028 | `milestones` |
| Whether science-based | Yes (SBTi 1.5C) | `science_based` |
| Scope coverage | Scope 1+2 | `scope_coverage` |

### PACK-029 Target Disclosure Generation

```python
class TCFDTargetDisclosure:
    """Generate TCFD-compliant target disclosure."""

    def generate(self, interim_result: InterimTargetResult) -> TCFDTargetSection:
        return TCFDTargetSection(
            targets=[
                TCFDTarget(
                    description=self._format_description(st),
                    target_type="Absolute" if st.metric == "absolute" else "Intensity",
                    scope=st.scope_type.value,
                    base_year=interim_result.base_year,
                    base_year_value=st.base_year_emissions,
                    target_year=st.target_year,
                    target_value=st.target_emissions,
                    reduction_pct=st.reduction_from_baseline_pct,
                    science_based=interim_result.sbti_validation.overall_pass,
                    alignment=interim_result.ambition.value,
                    interim_milestones=[
                        {"year": m.year, "target": m.target_emissions}
                        for m in st.milestones
                    ],
                )
                for st in interim_result.scope_timelines
            ],
            methodology_note=self._generate_methodology_note(interim_result),
        )
```

---

## Performance Against Targets

### Year-Over-Year Performance Table

PACK-029 generates a multi-year performance tracking table:

```
Target Performance Summary
==========================

Target: 46.5% reduction in Scope 1+2 emissions by 2030 (from 2021 baseline of 80,000 tCO2e)

| Year | Target  | Actual  | Gap    | % Achieved | RAG   |
|------|---------|---------|--------|------------|-------|
| 2022 | 76,280  | 75,000  | -1,280 | 13.4%      | GREEN |
| 2023 | 72,560  | 75,000  | +2,440 | 13.4%      | AMBER |
| 2024 | 68,840  | 68,000  | -840   | 32.3%      | GREEN |
| 2025 | 65,120  | --      | --     | --         | --    |
| ...  |         |         |        |            |       |
| 2030 | 42,800  | --      | --     | --         | --    |
```

### Variance Narrative for TCFD

PACK-029 generates detailed variance narratives suitable for TCFD disclosure:

```
Performance Against Target (2024):

Scope 1+2 emissions decreased 9.3% year-over-year and 15.0% from the 2021 baseline,
reaching 68,000 tCO2e against a 2024 interim target of 68,840 tCO2e.

LMDI decomposition of the 2023-2024 change reveals:
- Activity effect: +9,600 tCO2e (revenue growth of 12.0%)
- Intensity effect: -16,600 tCO2e (emission intensity improved 18.0%)
- Net change: -7,000 tCO2e

Key drivers of intensity improvement:
1. Renewable electricity PPA: -6,500 tCO2e
2. Energy efficiency program: -5,500 tCO2e
3. Heat pump installations: -2,100 tCO2e
4. Fleet electrification: -2,500 tCO2e

The company is on track to meet its 2030 near-term target. Trend extrapolation
(linear regression, MAPE 3.2%) projects 2030 emissions of 41,500 tCO2e, below
the target of 42,800 tCO2e.
```

---

## Scenario Analysis Integration

### TCFD Scenario Analysis Requirements

TCFD recommends that organizations describe the resilience of their strategy under different climate scenarios, including a 2 degrees Celsius or lower scenario.

### PACK-029 Scenario Support

The Trend Extrapolation Engine provides forward projections under three scenarios:

```json
{
  "scenario_analysis": {
    "baseline_scenario": {
      "description": "Current trajectory continues",
      "2030_emissions": 42000,
      "2050_emissions": 18000,
      "meets_near_term_target": true,
      "meets_long_term_target": true,
      "implied_temperature": "1.5C"
    },
    "delayed_action_scenario": {
      "description": "Corrective actions delayed by 2 years",
      "2030_emissions": 52000,
      "2050_emissions": 28000,
      "meets_near_term_target": false,
      "meets_long_term_target": false,
      "implied_temperature": "1.8C",
      "gap_to_target": 9200,
      "additional_investment_needed": 4500000
    },
    "accelerated_action_scenario": {
      "description": "All planned initiatives plus additional measures",
      "2030_emissions": 35000,
      "2050_emissions": 12000,
      "meets_near_term_target": true,
      "meets_long_term_target": true,
      "implied_temperature": "1.3C",
      "surplus_reduction": 7800
    }
  }
}
```

### Physical Risk Integration

While PACK-029 focuses on transition metrics (targets and performance), the TCFD also requires physical risk metrics. PACK-029 integrates with PACK-028 (Sector Pathway) for physical risk context:

```
Physical Risk Considerations for Target Planning:
- Heating/cooling demand changes due to climate (affects Scope 1+2)
- Supply chain disruptions (affects Scope 3)
- Water stress impacts on operations
- Extreme weather events affecting energy supply

PACK-029 accounts for weather normalization in variance analysis
(LMDI weather effect factor).
```

---

## PACK-029 TCFD Export Mapping

### TCFD Disclosure Template

The TCFD Disclosure Template (`tcfd_disclosure_template.py`) generates complete Metrics and Targets pillar content:

```python
class TCFDDisclosureTemplate:
    """Generate TCFD Metrics and Targets disclosure content."""

    SUPPORTED_FORMATS = ["md", "html", "json", "pdf"]

    def render(
        self,
        interim_targets: InterimTargetResult,
        annual_review: AnnualReviewResult,
        variance_analysis: Optional[VarianceAnalysisResult] = None,
        trend_forecast: Optional[TrendForecastResult] = None,
        corrective_actions: Optional[CorrectiveActionResult] = None,
        format: str = "md",
    ) -> str:
        """Render TCFD Metrics and Targets disclosure."""

        sections = []

        # 4(a) Metrics used
        sections.append(self._render_metrics_section(
            annual_review, variance_analysis
        ))

        # 4(b) GHG emissions
        sections.append(self._render_emissions_section(
            annual_review
        ))

        # 4(c) Targets and performance
        sections.append(self._render_targets_section(
            interim_targets, annual_review, trend_forecast, corrective_actions
        ))

        return self._format_output(sections, format)
```

### Output Structure

| Section | Content | Length |
|---------|---------|--------|
| 4(a) Metrics | Climate metrics table, intensity ratios, carbon price | ~500 words |
| 4(b) Emissions | Scope 1/2/3 table, methodology, YoY comparison | ~800 words |
| 4(c) Targets | Target details, performance, variance, forecast | ~1,200 words |
| Total | Complete Metrics and Targets pillar | ~2,500 words |

---

## Cross-Industry Metrics

### TCFD Cross-Industry Metrics

| Metric | Unit | PACK-029 Source | Category |
|--------|------|-----------------|----------|
| GHG emissions (Scope 1) | tCO2e | MRV Bridge | GHG |
| GHG emissions (Scope 2) | tCO2e | MRV Bridge | GHG |
| GHG emissions (Scope 3) | tCO2e | MRV Bridge | GHG |
| GHG emissions intensity | tCO2e/unit | Interim Target Engine | GHG |
| Internal carbon price | USD/tCO2e | Configuration | Transition Risk |
| Climate-related CapEx | USD | Corrective Action Engine | Transition Risk |
| % revenue from low-carbon products | % | Not primary (future) | Opportunity |
| Weighted avg carbon intensity | tCO2e/M USD | Calculation | Portfolio |

### Transition Risk Metrics from PACK-029

```json
{
  "transition_metrics": {
    "emission_reduction_achieved_pct": 15.0,
    "emission_reduction_target_pct": 46.5,
    "on_track_status": "GREEN",
    "climate_related_capex": {
      "current_year": 1200000,
      "planned_next_year": 1800000,
      "total_planned": 5000000,
      "as_pct_of_total_capex": 8.5
    },
    "carbon_price_assumptions": {
      "internal_carbon_price": 80,
      "currency": "USD",
      "per_unit": "tCO2e",
      "applied_to": "Investment decisions and MACC analysis"
    },
    "implied_temperature_rise": 1.5,
    "sbti_alignment": "1.5C (near-term validated)"
  }
}
```

---

## Sector-Specific Considerations

### Energy Sector

| Additional Metric | Description | PACK-029 Support |
|-------------------|-------------|------------------|
| Energy production mix (%) | Fossil vs. renewable | Via LMDI structural effect |
| Capacity additions (MW) | Low-carbon capacity | Corrective Action Engine |
| Methane emissions | Separate methane tracking | MRV Bridge (Agent-MRV-005) |

### Financial Sector

| Additional Metric | Description | PACK-029 Support |
|-------------------|-------------|------------------|
| Financed emissions | tCO2e per portfolio | PACK-028 Bridge |
| Portfolio alignment | Temperature score | Interim Target Engine |
| Green vs. brown lending | Lending book composition | Not primary |

### Manufacturing

| Additional Metric | Description | PACK-029 Support |
|-------------------|-------------|------------------|
| Process emissions | Industrial process GHG | MRV Bridge (Agent-MRV-004) |
| Energy intensity | GJ/unit of production | Via LMDI decomposition |
| Waste emissions | tCO2e from waste | MRV Bridge (Agent-MRV-007) |

### Real Estate

| Additional Metric | Description | PACK-029 Support |
|-------------------|-------------|------------------|
| Energy intensity per m2 | kWh/m2 | Via LMDI decomposition |
| Building certification | LEED, BREEAM coverage | Not primary |
| Retrofit investment | CapEx on energy efficiency | Corrective Action Engine |

---

## ISSB IFRS S2 Alignment

### ISSB Convergence

The International Sustainability Standards Board (ISSB) issued IFRS S2 (Climate-related Disclosures), which builds on and incorporates TCFD. PACK-029's TCFD outputs are designed to be compatible with IFRS S2:

| TCFD Disclosure | IFRS S2 Paragraph | PACK-029 Coverage |
|-----------------|-------------------|-------------------|
| 4(a) Metrics | Para 29-33 | Climate metrics table |
| 4(b) Emissions | Para 29(a) | Scope 1/2/3 emissions |
| 4(c) Targets | Para 33-36 | Target details and progress |
| Transition plans | Para 14(a) | Corrective Action Engine |
| Scenario analysis | Para 22 | Trend Extrapolation (3 scenarios) |

### IFRS S2 Additional Requirements

| Requirement | Description | PACK-029 Response |
|-------------|-------------|-------------------|
| Transition plan | How targets will be met | Corrective Action Plan |
| Carbon credits | Planned use of credits | Excluded from targets (SBTi) |
| Internal carbon price | Used for decisions | Configuration option |
| Capital deployment | Alignment of CapEx | Investment summary |
| Remuneration | Climate-linked pay | Not covered (governance) |

---

## Best Practices

### For Comprehensive TCFD Disclosure

1. **Disclose all three scopes** even if Scope 3 is estimated
2. **Use both location and market-based** Scope 2 methods
3. **Include interim milestones** not just end-year targets
4. **Explain drivers of change** using LMDI decomposition
5. **Show multi-year trends** (minimum 3 years comparison)
6. **Provide forward projections** with confidence intervals
7. **Link metrics to strategy** showing how targets drive business decisions
8. **Disclose methodology** including GWP values, consolidation approach
9. **Note data quality** and any limitations or estimates
10. **Align with SBTi** for science-based credibility

### PACK-029 TCFD Workflow

```
Annual TCFD Disclosure Process:

1. Collect emissions data (MRV Bridge)
2. Run Annual Progress Review
3. Run Variance Analysis (LMDI)
4. Run Trend Extrapolation (forward projection)
5. Generate TCFD Disclosure Template
6. Review and approve narrative
7. Integrate into annual/sustainability report
8. Publish and submit to TCFD Knowledge Hub
```

---

## References

- TCFD Recommendations: https://www.fsb-tcfd.org/recommendations/
- TCFD Implementation Guidance: https://www.fsb-tcfd.org/publications/
- TCFD 2021 Status Report: https://www.fsb-tcfd.org/publications/
- ISSB IFRS S2: https://www.ifrs.org/issued-standards/ifrs-sustainability-standards-navigator/ifrs-s2-climate-related-disclosures/
- TCFD Knowledge Hub: https://www.tcfdhub.org/

---

**End of TCFD Disclosure Guide**
