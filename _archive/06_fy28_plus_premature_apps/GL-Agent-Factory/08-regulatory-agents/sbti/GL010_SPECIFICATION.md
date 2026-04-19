# GL-010 SBTi Validation Agent Specification

**Agent ID:** gl-010-sbti-validation-v1
**Version:** 1.0.0
**Date:** 2025-12-04
**Priority:** P4-STANDARD
**Deadline:** Ongoing (Continuous validation cycles)
**Status:** SPECIFICATION COMPLETE

---

## 1. Executive Summary

### 1.1 Framework Overview

**Science Based Targets initiative (SBTi)**

The Science Based Targets initiative provides companies with a clearly defined pathway to reduce greenhouse gas emissions in line with the Paris Agreement goals. SBTi validates corporate emission reduction targets against climate science and tracks progress toward net-zero.

**Key Requirements:**
- Targets aligned with 1.5C or well-below 2C pathways
- Scope 1 and 2: 95% coverage required
- Scope 3: Required if >40% of total emissions
- Near-term (5-10 years) and long-term (by 2050) targets
- Annual progress tracking and reporting
- FLAG (Forest, Land and Agriculture) targets for land-use sectors

### 1.2 Agent Purpose

The SBTi Validation Agent automates the calculation, validation, and tracking of science-based targets. It provides:

1. **Target Setting** - Calculate 1.5C and well-below 2C aligned targets
2. **Methodology Selection** - Apply Absolute Contraction or SDA approaches
3. **Scope Coverage Validation** - Verify Scope 1, 2, 3 coverage requirements
4. **Progress Tracking** - Monitor year-over-year emission reductions
5. **FLAG Target Support** - Land-use sector specific targets
6. **Net-Zero Alignment** - Validate long-term net-zero commitments

---

## 2. SBTi Framework Specification

### 2.1 Target Types and Criteria

```yaml
sbti_target_types:
  near_term_targets:
    definition: "5-10 year targets from date of submission"
    ambition_levels:
      1_5C:
        description: "Aligned with 1.5C pathway"
        annual_reduction: 4.2%  # Linear annual reduction
        total_reduction_by_2030: "42% from 2020 base year"
        required_for: "Commitment after July 2022"

      well_below_2C:
        description: "Aligned with well-below 2C pathway"
        annual_reduction: 2.5%
        total_reduction_by_2030: "25% from 2020 base year"
        status: "No longer accepted for new commitments"

    scope_requirements:
      scope_1_2:
        coverage: "95% minimum"
        target_type: "Absolute or intensity"

      scope_3:
        threshold: ">40% of total emissions"
        coverage: "67% minimum when required"
        timeline: "Same timeline as Scope 1&2 or 5-10 years max"

  long_term_targets:
    definition: "Net-zero by 2050 or earlier"
    requirements:
      absolute_reduction: "90-95% from base year"
      residual_emissions: "5-10% maximum"
      neutralization: "Only for residual emissions"
      verification: "Third-party verification required"

  flag_targets:
    definition: "Forest, Land and Agriculture sector targets"
    applicable_sectors:
      - "Food and beverage"
      - "Agriculture"
      - "Forestry"
      - "Land use"
      - "Commodities trading"
    requirements:
      separate_target: "FLAG emissions tracked separately"
      no_deforestation: "Zero deforestation commitment required"
      land_sinks: "Removals can count toward target"
```

### 2.2 Target Setting Methods

```yaml
target_setting_methods:
  absolute_contraction_approach:
    abbreviation: "ACA"
    description: "Linear reduction in absolute emissions"
    formula: |
      Target_Emissions_Year_N =
        Base_Year_Emissions * (1 - Annual_Reduction_Rate)^N

      Where:
      - Annual_Reduction_Rate = 4.2% for 1.5C (linear)
      - N = Years from base year

    applicability:
      - "All sectors"
      - "All companies"
      - "Default method for Scope 1&2"

    example:
      base_year: 2020
      base_emissions: 100000  # tCO2e
      target_year: 2030
      annual_reduction: 4.2%
      target_emissions: 100000 * (1 - 0.042)^10 = 65,000  # tCO2e

  sectoral_decarbonization_approach:
    abbreviation: "SDA"
    description: "Sector-specific intensity pathway"
    formula: |
      Target_Intensity_Year_N =
        Sector_Pathway_Intensity(Year_N)

      Target_Emissions = Target_Intensity * Activity_Data

    applicability:
      - "Sector-specific pathways available"
      - "Power generation"
      - "Transport (aviation, shipping, road)"
      - "Buildings"
      - "Cement"
      - "Steel"
      - "Aluminum"

    sector_pathways:
      power_generation:
        metric: "gCO2/kWh"
        2020_benchmark: 450
        2030_target: 138
        2050_target: 0

      road_transport:
        metric: "gCO2/pkm or gCO2/tkm"
        pathway: "IEA Net Zero"

      aviation:
        metric: "gCO2/RTK"
        pathway: "SBTi Aviation"

      steel:
        metric: "tCO2/t crude steel"
        2020_benchmark: 1.89
        2030_target: 1.28
        2050_target: 0.10

      cement:
        metric: "kgCO2/t cite"
        2020_benchmark: 611
        2030_target: 469
        2050_target: 107

      aluminum:
        metric: "tCO2/t aluminum"
        2020_benchmark: 11.5
        2030_target: 3.3
        2050_target: 0.5

  scope_3_methods:
    supplier_engagement:
      description: "Engage suppliers to set their own SBTs"
      requirement: "67% of suppliers by emissions have SBTs within 5 years"
      measurement: "% of supplier emissions from SBT-committed suppliers"

    physical_intensity:
      description: "Reduce emissions per unit of product"
      formula: "Emissions / Physical Output (e.g., tCO2e/product)"

    economic_intensity:
      description: "Reduce emissions per revenue"
      formula: "Emissions / Revenue (tCO2e/$M revenue)"
      note: "Requires absolute emissions don't increase"

    absolute_scope_3:
      description: "Absolute reduction in Scope 3 emissions"
      alignment: "Same as Scope 1&2 targets"
```

### 2.3 SBTi Criteria (Version 5.1)

```yaml
sbti_criteria_v5_1:
  version: "5.1"
  effective_date: "2024-07"

  commitment_criteria:
    eligibility:
      - "All companies with operations in any sector"
      - "Must have emissions data for base year"
      - "Must commit to set target within 24 months"

  target_criteria:
    timeframe:
      near_term: "5-10 years from submission"
      long_term: "By 2050 or earlier"

    base_year:
      requirement: "Most recent year with verifiable data"
      maximum_age: "2 years prior to submission"
      exceptions: "COVID-affected years may be adjusted"

    coverage:
      scope_1: "95% of emissions"
      scope_2: "95% of emissions"
      scope_3:
        threshold: "40% of total emissions"
        coverage_if_required: "67% of Scope 3 emissions"
        categories_to_include: "All relevant categories"

    ambition:
      minimum: "1.5C aligned (4.2% annual reduction)"
      scope_3_minimum: "Well-below 2C aligned"

  validation_process:
    submission:
      - "Target submission form"
      - "GHG inventory (Scope 1, 2, 3)"
      - "Supporting documentation"

    review_timeline: "30 business days"

    outcomes:
      - "Approved"
      - "Approved with conditions"
      - "Rejected (requires resubmission)"

  recalculation_policy:
    triggers:
      - "Structural changes (M&A, divestiture)"
      - "Methodology changes"
      - "Errors discovered"
      - "Changes >5% of base year emissions"

    requirement: "Recalculate base year and targets"
```

---

## 3. Agent Architecture

### 3.1 Agent Specification (AgentSpec v2)

```yaml
agent_id: gl-010-sbti-validation-v1
name: "SBTi Validation Agent"
version: "1.0.0"
type: target-validation
priority: P4-STANDARD
deadline: "ongoing"

description: |
  Automated science-based target setting, validation, and progress tracking
  agent. Supports SBTi criteria v5.1, multiple target-setting methodologies,
  and annual progress monitoring for 1.5C and net-zero alignment.

framework_context:
  framework: "Science Based Targets initiative"
  version: "Criteria v5.1"
  update_frequency: "Annual criteria updates"
  validation_body: "SBTi"

inputs:
  company_profile:
    type: object
    required: true
    properties:
      company_id:
        type: string
        description: "Unique company identifier"

      company_name:
        type: string

      sector:
        type: string
        enum: ["power_generation", "transport_road", "transport_aviation", "transport_shipping", "buildings", "cement", "steel", "aluminum", "chemicals", "food_beverage", "retail", "financial_services", "technology", "other"]

      industry:
        type: string
        description: "Specific industry within sector"

      revenue_usd:
        type: number
        description: "Annual revenue (for intensity targets)"

      employees:
        type: integer

      geographic_scope:
        type: array
        items:
          type: string
          description: "ISO country codes"

      sbti_status:
        type: string
        enum: ["no_commitment", "committed", "targets_set", "net_zero_committed"]

  emissions_inventory:
    type: object
    required: true
    properties:
      base_year:
        type: integer
        minimum: 2015
        description: "GHG inventory base year"

      base_year_emissions:
        type: object
        properties:
          scope_1:
            type: object
            properties:
              total_tco2e:
                type: number
              categories:
                type: object
                properties:
                  stationary_combustion:
                    type: number
                  mobile_combustion:
                    type: number
                  fugitive_emissions:
                    type: number
                  process_emissions:
                    type: number

          scope_2:
            type: object
            properties:
              location_based_tco2e:
                type: number
              market_based_tco2e:
                type: number

          scope_3:
            type: object
            properties:
              total_tco2e:
                type: number
              categories:
                type: object
                properties:
                  cat_1_purchased_goods:
                    type: number
                  cat_2_capital_goods:
                    type: number
                  cat_3_fuel_energy:
                    type: number
                  cat_4_upstream_transport:
                    type: number
                  cat_5_waste:
                    type: number
                  cat_6_business_travel:
                    type: number
                  cat_7_commuting:
                    type: number
                  cat_8_upstream_leased:
                    type: number
                  cat_9_downstream_transport:
                    type: number
                  cat_10_processing:
                    type: number
                  cat_11_use_of_products:
                    type: number
                  cat_12_end_of_life:
                    type: number
                  cat_13_downstream_leased:
                    type: number
                  cat_14_franchises:
                    type: number
                  cat_15_investments:
                    type: number

          flag_emissions:
            type: object
            description: "Forest, Land and Agriculture emissions"
            properties:
              total_tco2e:
                type: number
              land_use_change:
                type: number
              agriculture:
                type: number
              forestry:
                type: number

      current_year:
        type: integer

      current_year_emissions:
        type: object
        description: "Same structure as base_year_emissions"

      emissions_history:
        type: array
        description: "Historical emissions for progress tracking"
        items:
          type: object
          properties:
            year:
              type: integer
            scope_1_tco2e:
              type: number
            scope_2_tco2e:
              type: number
            scope_3_tco2e:
              type: number

  activity_data:
    type: object
    description: "For intensity targets"
    properties:
      base_year:
        type: object
        properties:
          revenue_usd:
            type: number
          production_units:
            type: number
          production_unit_name:
            type: string
          employees:
            type: integer
          floor_area_m2:
            type: number
          passenger_km:
            type: number
          tonne_km:
            type: number

      current_year:
        type: object
        description: "Same structure as base_year"

  existing_targets:
    type: object
    description: "Previously set or committed targets"
    properties:
      near_term_target:
        type: object
        properties:
          target_year:
            type: integer
          target_type:
            type: string
            enum: ["absolute", "intensity"]
          scope_coverage:
            type: array
            items:
              type: string
              enum: ["scope_1", "scope_2", "scope_3"]
          reduction_percentage:
            type: number
          ambition_level:
            type: string
            enum: ["1_5C", "well_below_2C"]
          validation_date:
            type: string
            format: date
          methodology:
            type: string
            enum: ["ACA", "SDA"]

      long_term_target:
        type: object
        properties:
          net_zero_year:
            type: integer
            maximum: 2050
          residual_emissions_percent:
            type: number
            maximum: 10

      flag_target:
        type: object
        properties:
          target_year:
            type: integer
          reduction_percentage:
            type: number
          no_deforestation_commitment:
            type: boolean

  target_request:
    type: object
    description: "Parameters for new target calculation"
    properties:
      target_type:
        type: string
        enum: ["near_term", "long_term", "flag", "all"]

      ambition_level:
        type: string
        enum: ["1_5C", "well_below_2C"]
        default: "1_5C"

      target_year:
        type: integer
        description: "Target year (5-10 years from now)"

      methodology_preference:
        type: string
        enum: ["ACA", "SDA", "auto"]
        default: "auto"

      scope_3_approach:
        type: string
        enum: ["absolute", "intensity", "supplier_engagement"]

outputs:
  target_calculation:
    type: object
    properties:
      calculation_id:
        type: string
        format: uuid

      company_id:
        type: string

      calculation_date:
        type: string
        format: date-time

      recommended_targets:
        type: object
        properties:
          near_term:
            type: object
            properties:
              base_year:
                type: integer
              target_year:
                type: integer
              methodology:
                type: string

              scope_1_2_target:
                type: object
                properties:
                  base_emissions_tco2e:
                    type: number
                  target_emissions_tco2e:
                    type: number
                  reduction_percentage:
                    type: number
                  annual_reduction_rate:
                    type: number
                  ambition_level:
                    type: string
                  coverage_percentage:
                    type: number

              scope_3_target:
                type: object
                properties:
                  required:
                    type: boolean
                  reason:
                    type: string
                  base_emissions_tco2e:
                    type: number
                  target_emissions_tco2e:
                    type: number
                  reduction_percentage:
                    type: number
                  approach:
                    type: string
                  coverage_percentage:
                    type: number
                  key_categories:
                    type: array
                    items:
                      type: string

          long_term:
            type: object
            properties:
              net_zero_year:
                type: integer
              total_reduction_required:
                type: number
              residual_emissions_tco2e:
                type: number
              residual_emissions_percent:
                type: number
              neutralization_required_tco2e:
                type: number

          flag_target:
            type: object
            properties:
              applicable:
                type: boolean
              base_emissions_tco2e:
                type: number
              target_emissions_tco2e:
                type: number
              reduction_percentage:
                type: number
              no_deforestation_required:
                type: boolean

      sbti_criteria_check:
        type: object
        properties:
          compliant:
            type: boolean
          criteria_version:
            type: string
          checks:
            type: array
            items:
              type: object
              properties:
                criterion:
                  type: string
                status:
                  type: string
                  enum: ["PASS", "FAIL", "WARNING"]
                details:
                  type: string

      sector_pathway_comparison:
        type: object
        description: "For SDA methodology"
        properties:
          sector:
            type: string
          pathway_metric:
            type: string
          current_intensity:
            type: number
          target_intensity:
            type: number
          pathway_benchmark:
            type: number
          alignment_status:
            type: string

      audit_trail:
        type: object
        properties:
          input_hash:
            type: string
          output_hash:
            type: string
          calculation_parameters:
            type: object
          methodology_version:
            type: string

  progress_report:
    type: object
    properties:
      report_id:
        type: string
        format: uuid

      reporting_year:
        type: integer

      progress_vs_target:
        type: object
        properties:
          scope_1_2:
            type: object
            properties:
              base_year_emissions:
                type: number
              target_year_emissions:
                type: number
              current_year_emissions:
                type: number
              expected_emissions_current_year:
                type: number
              actual_reduction_percent:
                type: number
              required_reduction_percent:
                type: number
              on_track:
                type: boolean
              gap_tco2e:
                type: number
              trajectory_status:
                type: string
                enum: ["ahead", "on_track", "behind", "significantly_behind"]

          scope_3:
            type: object
            properties:
              # Same structure as scope_1_2

          flag:
            type: object
            properties:
              # Same structure as scope_1_2

      year_over_year_change:
        type: object
        properties:
          scope_1_change_percent:
            type: number
          scope_2_change_percent:
            type: number
          scope_3_change_percent:
            type: number
          total_change_percent:
            type: number

      recommendations:
        type: array
        items:
          type: object
          properties:
            category:
              type: string
            action:
              type: string
            impact_tco2e:
              type: number
            priority:
              type: string

  sbti_submission_package:
    type: object
    properties:
      package_id:
        type: string
        format: uuid

      submission_date:
        type: string
        format: date

      target_submission_form:
        type: object
        description: "SBTi target submission form data"

      supporting_documentation:
        type: object
        properties:
          ghg_inventory:
            type: object
          target_calculations:
            type: object
          methodology_documentation:
            type: object

      validation_readiness:
        type: object
        properties:
          ready:
            type: boolean
          missing_items:
            type: array
            items:
              type: string
          recommendations:
            type: array
            items:
              type: string

tools:
  - name: target_calculator
    type: calculator
    description: "Calculate science-based emission reduction targets"
    inputs: ["emissions_inventory", "target_request"]
    outputs: ["target_calculation"]

  - name: aca_calculator
    type: calculator
    description: "Absolute Contraction Approach calculation"
    inputs: ["base_emissions", "target_year", "ambition_level"]
    outputs: ["target_emissions", "annual_reduction"]

  - name: sda_calculator
    type: calculator
    description: "Sectoral Decarbonization Approach calculation"
    inputs: ["sector", "activity_data", "target_year"]
    outputs: ["target_intensity", "pathway_comparison"]

  - name: scope3_analyzer
    type: analyzer
    description: "Analyze Scope 3 emissions and recommend approach"
    inputs: ["scope_3_emissions", "total_emissions"]
    outputs: ["scope3_required", "key_categories", "recommended_approach"]

  - name: flag_target_calculator
    type: calculator
    description: "Calculate FLAG sector targets"
    inputs: ["flag_emissions", "sector"]
    outputs: ["flag_target"]

  - name: progress_tracker
    type: analyzer
    description: "Track progress against targets"
    inputs: ["emissions_history", "targets"]
    outputs: ["progress_report"]

  - name: criteria_validator
    type: validator
    description: "Validate targets against SBTi criteria"
    inputs: ["targets", "emissions_data"]
    outputs: ["criteria_check"]

  - name: net_zero_calculator
    type: calculator
    description: "Calculate net-zero pathway"
    inputs: ["base_emissions", "net_zero_year"]
    outputs: ["reduction_pathway", "residual_emissions"]

  - name: submission_generator
    type: generator
    description: "Generate SBTi submission package"
    inputs: ["targets", "emissions_inventory"]
    outputs: ["submission_package"]

  - name: provenance_tracker
    type: utility
    description: "Track calculation provenance"
    inputs: ["inputs", "outputs"]
    outputs: ["provenance_hash"]

evaluation:
  golden_tests:
    total_count: 50
    categories:
      target_calculation: 15
      progress_tracking: 12
      criteria_validation: 13
      sector_pathways: 10

  accuracy_thresholds:
    target_calculation: 0.01  # +/- 1%
    progress_tracking: 0.02   # +/- 2%

  benchmarks:
    latency_p95_seconds: 10
    cost_per_calculation_usd: 0.30

certification:
  required_approvals:
    - climate_science_team
    - sbti_methodology_expert

  compliance_checks:
    - sbti_criteria_v5_1
    - ghg_protocol_alignment
    - calculation_reproducibility
```

---

## 4. Calculation Formulas

### 4.1 Absolute Contraction Approach (ACA)

```python
def calculate_aca_target(
    base_year_emissions: float,
    base_year: int,
    target_year: int,
    ambition_level: str = "1_5C"
) -> ACATargetResult:
    """
    Calculate target using Absolute Contraction Approach.

    Formula:
    Target_Emissions = Base_Emissions * (1 - r)^n

    Where:
    - r = Annual linear reduction rate
    - n = Number of years from base year to target year

    Annual reduction rates:
    - 1.5C: 4.2% per year
    - Well-below 2C: 2.5% per year

    Source: SBTi Criteria v5.1, Section 4.3
    """

    # Define annual reduction rates
    reduction_rates = {
        "1_5C": 0.042,          # 4.2% annual reduction
        "well_below_2C": 0.025  # 2.5% annual reduction
    }

    r = reduction_rates.get(ambition_level, 0.042)
    n = target_year - base_year

    # Validate timeframe (5-10 years for near-term)
    if not (5 <= n <= 10):
        raise ValueError(f"Target timeframe {n} years must be 5-10 years")

    # Calculate target emissions
    target_emissions = base_year_emissions * ((1 - r) ** n)

    # Calculate total reduction percentage
    total_reduction = (1 - (target_emissions / base_year_emissions)) * 100

    return ACATargetResult(
        base_year=base_year,
        base_year_emissions=base_year_emissions,
        target_year=target_year,
        target_emissions=target_emissions,
        annual_reduction_rate=r * 100,
        total_reduction_percent=total_reduction,
        ambition_level=ambition_level,
        methodology="ACA"
    )
```

### 4.2 Sectoral Decarbonization Approach (SDA)

```python
def calculate_sda_target(
    sector: str,
    base_year: int,
    target_year: int,
    activity_data: dict
) -> SDATargetResult:
    """
    Calculate target using Sectoral Decarbonization Approach.

    Formula:
    Target_Intensity = Sector_Pathway_Intensity(target_year)
    Target_Emissions = Target_Intensity * Projected_Activity

    The SDA uses sector-specific decarbonization pathways that
    allocate the global carbon budget to individual sectors.

    Source: SBTi Sectoral Decarbonization Approach (SDA) Tool
    """

    # Sector pathway data (from IEA/SBTi)
    sector_pathways = {
        "power_generation": {
            "metric": "gCO2/kWh",
            "2020": 450,
            "2025": 294,
            "2030": 138,
            "2035": 50,
            "2040": 10,
            "2050": 0
        },
        "steel": {
            "metric": "tCO2/t crude steel",
            "2020": 1.89,
            "2025": 1.58,
            "2030": 1.28,
            "2040": 0.52,
            "2050": 0.10
        },
        "cement": {
            "metric": "kgCO2/t cementitious",
            "2020": 611,
            "2025": 545,
            "2030": 469,
            "2040": 310,
            "2050": 107
        },
        "aluminum": {
            "metric": "tCO2/t aluminum",
            "2020": 11.5,
            "2025": 7.5,
            "2030": 3.3,
            "2040": 1.5,
            "2050": 0.5
        },
        "road_transport": {
            "metric": "gCO2/pkm",
            "2020": 115,
            "2025": 95,
            "2030": 70,
            "2040": 30,
            "2050": 0
        }
    }

    if sector not in sector_pathways:
        raise ValueError(f"Sector {sector} not supported for SDA")

    pathway = sector_pathways[sector]

    # Interpolate for target year
    target_intensity = interpolate_pathway(pathway, target_year)
    base_intensity = interpolate_pathway(pathway, base_year)

    # Calculate current intensity
    current_emissions = activity_data["base_year_emissions"]
    current_activity = activity_data["base_year_activity"]
    current_intensity = current_emissions / current_activity

    # Calculate target emissions based on projected activity
    projected_activity = activity_data.get(
        "projected_activity_target_year",
        current_activity  # Assume no growth if not provided
    )

    target_emissions = target_intensity * projected_activity

    return SDATargetResult(
        sector=sector,
        base_year=base_year,
        target_year=target_year,
        metric=pathway["metric"],
        base_intensity=base_intensity,
        current_intensity=current_intensity,
        target_intensity=target_intensity,
        pathway_benchmark=target_intensity,
        target_emissions=target_emissions,
        alignment_status="aligned" if current_intensity <= base_intensity else "above_pathway",
        methodology="SDA"
    )

def interpolate_pathway(pathway: dict, year: int) -> float:
    """Linear interpolation between pathway years."""
    years = sorted([int(k) for k in pathway.keys() if k.isdigit()])

    if year <= years[0]:
        return pathway[str(years[0])]
    if year >= years[-1]:
        return pathway[str(years[-1])]

    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            y1, y2 = pathway[str(years[i])], pathway[str(years[i + 1])]
            t = (year - years[i]) / (years[i + 1] - years[i])
            return y1 + t * (y2 - y1)
```

### 4.3 Scope 3 Target Calculation

```python
def calculate_scope3_target(
    scope_3_emissions: dict,
    total_emissions: float,
    approach: str,
    target_year: int,
    base_year: int
) -> Scope3TargetResult:
    """
    Calculate Scope 3 target per SBTi requirements.

    Requirements:
    - Scope 3 target required if Scope 3 > 40% of total
    - Must cover 67% of Scope 3 emissions
    - Ambition: minimum well-below 2C (2.5% annual)
    - Approaches: Absolute, Physical Intensity, Supplier Engagement

    Source: SBTi Criteria v5.1, Section 4.4
    """

    total_scope_3 = scope_3_emissions.get("total_tco2e", 0)
    scope_3_ratio = total_scope_3 / total_emissions if total_emissions > 0 else 0

    # Check if Scope 3 target required
    scope_3_required = scope_3_ratio > 0.40

    if not scope_3_required:
        return Scope3TargetResult(
            required=False,
            reason=f"Scope 3 is {scope_3_ratio*100:.1f}% of total (threshold: 40%)"
        )

    # Identify key categories (largest contributors)
    categories = scope_3_emissions.get("categories", {})
    sorted_categories = sorted(
        categories.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Coverage: include categories until 67% coverage
    coverage_target = 0.67
    cumulative = 0
    covered_categories = []
    covered_emissions = 0

    for cat_name, cat_emissions in sorted_categories:
        coverage = cumulative / total_scope_3 if total_scope_3 > 0 else 0
        if coverage >= coverage_target:
            break
        covered_categories.append(cat_name)
        covered_emissions += cat_emissions
        cumulative += cat_emissions

    # Calculate target based on approach
    n = target_year - base_year

    if approach == "absolute":
        # Minimum 2.5% annual reduction (well-below 2C)
        r = 0.025
        target_emissions = covered_emissions * ((1 - r) ** n)

    elif approach == "supplier_engagement":
        # 67% of suppliers set SBTs within 5 years
        target_emissions = None  # Not emissions-based
        supplier_target = 0.67

    elif approach == "intensity":
        # Physical or economic intensity reduction
        r = 0.025
        target_emissions = covered_emissions * ((1 - r) ** n)

    return Scope3TargetResult(
        required=True,
        reason=f"Scope 3 is {scope_3_ratio*100:.1f}% of total (>40% threshold)",
        base_emissions=total_scope_3,
        covered_emissions=covered_emissions,
        coverage_percent=(covered_emissions / total_scope_3) * 100,
        covered_categories=covered_categories,
        target_emissions=target_emissions,
        approach=approach,
        reduction_percent=((covered_emissions - target_emissions) / covered_emissions) * 100 if target_emissions else None
    )
```

### 4.4 Net-Zero Target Calculation

```python
def calculate_net_zero_target(
    base_year_emissions: float,
    base_year: int,
    net_zero_year: int = 2050
) -> NetZeroTargetResult:
    """
    Calculate long-term net-zero target per SBTi Net-Zero Standard.

    Requirements:
    - 90-95% absolute reduction from base year
    - Only residual emissions (5-10%) can be neutralized
    - Neutralization must be through permanent carbon removal
    - Net-zero year: 2050 or earlier

    Source: SBTi Corporate Net-Zero Standard v1.0
    """

    # Validate net-zero year
    if net_zero_year > 2050:
        raise ValueError("Net-zero year must be 2050 or earlier")

    # Calculate required reduction (90-95%)
    min_reduction = 0.90
    max_reduction = 0.95
    recommended_reduction = 0.93  # Middle ground

    # Calculate target emissions
    target_emissions = base_year_emissions * (1 - recommended_reduction)

    # Residual emissions = what remains after 90-95% reduction
    residual_emissions = target_emissions
    residual_percent = (residual_emissions / base_year_emissions) * 100

    # Neutralization required = residual emissions
    neutralization_required = residual_emissions

    # Calculate annual reduction pathway
    years_to_net_zero = net_zero_year - base_year
    annual_reduction_required = 1 - ((1 - recommended_reduction) ** (1 / years_to_net_zero))

    # Generate interim milestones
    milestones = []
    for year in range(base_year, net_zero_year + 1, 5):
        years_elapsed = year - base_year
        interim_emissions = base_year_emissions * ((1 - annual_reduction_required) ** years_elapsed)
        interim_reduction = ((base_year_emissions - interim_emissions) / base_year_emissions) * 100
        milestones.append({
            "year": year,
            "emissions_tco2e": interim_emissions,
            "reduction_percent": interim_reduction
        })

    return NetZeroTargetResult(
        base_year=base_year,
        base_year_emissions=base_year_emissions,
        net_zero_year=net_zero_year,
        total_reduction_required_percent=recommended_reduction * 100,
        target_emissions_tco2e=target_emissions,
        residual_emissions_tco2e=residual_emissions,
        residual_emissions_percent=residual_percent,
        neutralization_required_tco2e=neutralization_required,
        annual_reduction_rate=annual_reduction_required * 100,
        milestones=milestones
    )
```

### 4.5 Progress Tracking

```python
def track_progress(
    emissions_history: list,
    target: dict
) -> ProgressReport:
    """
    Track progress against science-based target.

    Metrics:
    1. Actual emissions vs target pathway
    2. Year-over-year change
    3. Required annual reduction to get back on track
    4. Trajectory status (ahead/on-track/behind)

    Source: SBTi Progress Monitoring Requirements
    """

    base_year = target["base_year"]
    target_year = target["target_year"]
    base_emissions = target["base_emissions_tco2e"]
    target_emissions = target["target_emissions_tco2e"]
    annual_rate = target["annual_reduction_rate"]

    # Get latest year data
    latest = max(emissions_history, key=lambda x: x["year"])
    current_year = latest["year"]
    current_emissions = latest["total_tco2e"]

    # Calculate expected emissions for current year (linear pathway)
    years_from_base = current_year - base_year
    expected_emissions = base_emissions * ((1 - annual_rate) ** years_from_base)

    # Calculate actual reduction
    actual_reduction = ((base_emissions - current_emissions) / base_emissions) * 100
    expected_reduction = ((base_emissions - expected_emissions) / base_emissions) * 100

    # Gap analysis
    gap_tco2e = current_emissions - expected_emissions
    gap_percent = (gap_tco2e / expected_emissions) * 100 if expected_emissions > 0 else 0

    # Determine trajectory status
    if gap_percent < -5:
        trajectory_status = "ahead"
    elif gap_percent <= 5:
        trajectory_status = "on_track"
    elif gap_percent <= 15:
        trajectory_status = "behind"
    else:
        trajectory_status = "significantly_behind"

    # Calculate required annual reduction to reach target
    years_remaining = target_year - current_year
    if years_remaining > 0 and current_emissions > target_emissions:
        required_reduction = 1 - ((target_emissions / current_emissions) ** (1 / years_remaining))
    else:
        required_reduction = 0

    # Year-over-year change
    if len(emissions_history) >= 2:
        previous = sorted(emissions_history, key=lambda x: x["year"])[-2]
        yoy_change = ((current_emissions - previous["total_tco2e"]) / previous["total_tco2e"]) * 100
    else:
        yoy_change = None

    return ProgressReport(
        current_year=current_year,
        base_year=base_year,
        target_year=target_year,
        base_emissions_tco2e=base_emissions,
        current_emissions_tco2e=current_emissions,
        target_emissions_tco2e=target_emissions,
        expected_emissions_tco2e=expected_emissions,
        actual_reduction_percent=actual_reduction,
        expected_reduction_percent=expected_reduction,
        gap_tco2e=gap_tco2e,
        gap_percent=gap_percent,
        trajectory_status=trajectory_status,
        on_track=trajectory_status in ["ahead", "on_track"],
        required_annual_reduction_remaining=required_reduction * 100,
        yoy_change_percent=yoy_change
    )
```

---

## 5. FLAG Target Requirements

### 5.1 FLAG Sector Specification

```yaml
flag_targets:
  name: "Forest, Land and Agriculture Targets"
  version: "1.0"
  effective_date: "2022-09"

  applicable_sectors:
    primary:
      - "Agricultural production"
      - "Food processing"
      - "Food retail"
      - "Forestry and logging"
      - "Paper and pulp"
    secondary:
      - "Commodity trading"
      - "Restaurants and food service"
      - "Textiles (natural fibers)"

  scope:
    included_emissions:
      - "Land use change (deforestation)"
      - "Agricultural production emissions"
      - "Land management"
      - "Fertilizer use"
      - "Livestock emissions"
      - "Rice cultivation"
      - "Forestry operations"

    excluded:
      - "Processing and manufacturing (covered in standard targets)"
      - "Transportation (covered in Scope 3)"

  target_requirements:
    no_deforestation:
      requirement: "Mandatory commitment"
      deadline: "2025 (commodity sourcing)"
      scope: "All forest-risk commodities"

    reduction_target:
      minimum: "3.03% annual reduction"
      ambition: "1.5C aligned"
      methodology: "FLAG Pathway"

    land_sinks:
      eligibility: "Can count toward target"
      verification: "Third-party verified"
      permanence: "Minimum 40 years"

  flag_pathway:
    2020_benchmark: "11.9 Gt CO2e (global)"
    2030_target: "5.9 Gt CO2e (50% reduction)"
    2050_target: "Net negative (-3.2 Gt CO2e)"
```

---

## 6. Golden Test Scenarios

### 6.1 Target Calculation Tests (15 tests)

```yaml
golden_tests_target_calculation:
  - test_id: SBTI-CALC-001
    name: "1.5C ACA target for manufacturing company"
    input:
      company_profile:
        sector: "other"
      emissions_inventory:
        base_year: 2020
        base_year_emissions:
          scope_1:
            total_tco2e: 50000
          scope_2:
            market_based_tco2e: 30000
          scope_3:
            total_tco2e: 120000
      target_request:
        ambition_level: "1_5C"
        target_year: 2030
    expected:
      recommended_targets:
        near_term:
          scope_1_2_target:
            base_emissions_tco2e: 80000
            target_emissions_tco2e: 52000  # ~35% reduction
            annual_reduction_rate: 4.2
            ambition_level: "1_5C"

  - test_id: SBTI-CALC-002
    name: "Scope 3 target required (>40%)"
    input:
      emissions_inventory:
        scope_1:
          total_tco2e: 10000
        scope_2:
          market_based_tco2e: 10000
        scope_3:
          total_tco2e: 100000  # 83% of total
    expected:
      recommended_targets:
        scope_3_target:
          required: true
          reason: "Scope 3 is 83% of total emissions (>40% threshold)"

  - test_id: SBTI-CALC-003
    name: "Scope 3 not required (<40%)"
    input:
      emissions_inventory:
        scope_1:
          total_tco2e: 50000
        scope_2:
          market_based_tco2e: 30000
        scope_3:
          total_tco2e: 20000  # 20% of total
    expected:
      recommended_targets:
        scope_3_target:
          required: false

  - test_id: SBTI-CALC-004
    name: "SDA target for power generation"
    input:
      company_profile:
        sector: "power_generation"
      activity_data:
        base_year:
          production_units: 10000000  # 10 TWh
          production_unit_name: "MWh"
        base_year_emissions: 4500000  # tCO2e (450 gCO2/kWh)
      target_request:
        methodology_preference: "SDA"
        target_year: 2030
    expected:
      recommended_targets:
        near_term:
          methodology: "SDA"
          sector_pathway_comparison:
            current_intensity: 450  # gCO2/kWh
            target_intensity: 138  # gCO2/kWh by 2030

  - test_id: SBTI-CALC-005
    name: "SDA target for steel production"
    input:
      company_profile:
        sector: "steel"
      activity_data:
        base_year:
          production_units: 1000000  # 1 Mt crude steel
          production_unit_name: "t crude steel"
        base_year_emissions: 1890000  # 1.89 tCO2/t
      target_request:
        methodology_preference: "SDA"
        target_year: 2030
    expected:
      recommended_targets:
        near_term:
          sector_pathway_comparison:
            target_intensity: 1.28  # tCO2/t by 2030

  - test_id: SBTI-CALC-006
    name: "Net-zero target calculation"
    input:
      emissions_inventory:
        base_year: 2020
        base_year_emissions:
          scope_1:
            total_tco2e: 100000
          scope_2:
            total_tco2e: 50000
          scope_3:
            total_tco2e: 350000
      target_request:
        target_type: "long_term"
    expected:
      recommended_targets:
        long_term:
          net_zero_year: 2050
          total_reduction_required: 93  # 90-95%
          residual_emissions_percent: 7
          neutralization_required_tco2e: 35000

  - test_id: SBTI-CALC-007
    name: "FLAG target for food company"
    input:
      company_profile:
        sector: "food_beverage"
      emissions_inventory:
        flag_emissions:
          total_tco2e: 500000
          land_use_change: 200000
          agriculture: 300000
      target_request:
        target_type: "flag"
    expected:
      recommended_targets:
        flag_target:
          applicable: true
          no_deforestation_required: true
          reduction_percentage:
            range: [30, 50]  # By 2030

  - test_id: SBTI-CALC-008
    name: "Combined near-term and FLAG targets"
    input:
      company_profile:
        sector: "food_beverage"
      emissions_inventory:
        scope_1:
          total_tco2e: 100000
        scope_2:
          total_tco2e: 50000
        scope_3:
          total_tco2e: 350000
        flag_emissions:
          total_tco2e: 200000
    expected:
      recommended_targets:
        near_term:
          methodology: "ACA"
        flag_target:
          applicable: true
          separate_from_non_flag: true

  - test_id: SBTI-CALC-009
    name: "5-year near-term target (minimum)"
    input:
      target_request:
        target_year: 2030  # 5 years from 2025
        base_year: 2025
    expected:
      validation:
        valid: true
        timeframe_check: "PASS"

  - test_id: SBTI-CALC-010
    name: "11-year target (invalid - too long)"
    input:
      target_request:
        target_year: 2036
        base_year: 2025
    expected:
      validation:
        valid: false
        error: "Target timeframe must be 5-10 years"

  - test_id: SBTI-CALC-011
    name: "Base year too old"
    input:
      emissions_inventory:
        base_year: 2018  # >2 years before submission in 2025
    expected:
      validation:
        valid: false
        warning: "Base year should be within 2 years of submission"

  - test_id: SBTI-CALC-012
    name: "Scope coverage validation"
    input:
      emissions_inventory:
        scope_1:
          total_tco2e: 100000
          coverage_percent: 80  # Below 95% requirement
    expected:
      sbti_criteria_check:
        checks:
          - criterion: "Scope 1 coverage"
            status: "FAIL"
            details: "Coverage 80% below 95% requirement"

  - test_id: SBTI-CALC-013
    name: "Intensity target with activity growth"
    input:
      target_request:
        target_type: "intensity"
      activity_data:
        base_year:
          revenue_usd: 1000000000
          emissions: 100000
        projected:
          revenue_usd: 1500000000  # 50% growth
    expected:
      recommended_targets:
        intensity_target:
          base_intensity: 100  # tCO2e/$M
          target_intensity: 65  # ~35% reduction

  - test_id: SBTI-CALC-014
    name: "Supplier engagement approach for Scope 3"
    input:
      scope_3_approach: "supplier_engagement"
    expected:
      recommended_targets:
        scope_3_target:
          approach: "supplier_engagement"
          target: "67% of suppliers by emissions have SBTs by 2030"

  - test_id: SBTI-CALC-015
    name: "Well-below 2C no longer accepted"
    input:
      target_request:
        ambition_level: "well_below_2C"
        submission_date: "2025-01-01"
    expected:
      validation:
        valid: false
        error: "Well-below 2C no longer accepted for new commitments after July 2022"
```

### 6.2 Progress Tracking Tests (12 tests)

```yaml
golden_tests_progress_tracking:
  - test_id: SBTI-PROG-001
    name: "On track to meet target"
    input:
      existing_targets:
        near_term_target:
          base_year: 2020
          target_year: 2030
          base_emissions_tco2e: 100000
          target_emissions_tco2e: 58000
      emissions_history:
        - year: 2020
          total_tco2e: 100000
        - year: 2021
          total_tco2e: 95000
        - year: 2022
          total_tco2e: 91000
        - year: 2023
          total_tco2e: 87000
        - year: 2024
          total_tco2e: 83000
    expected:
      progress_report:
        trajectory_status: "on_track"
        on_track: true

  - test_id: SBTI-PROG-002
    name: "Behind target - needs acceleration"
    input:
      existing_targets:
        near_term_target:
          base_year: 2020
          target_year: 2030
          base_emissions_tco2e: 100000
          target_emissions_tco2e: 58000
      emissions_history:
        - year: 2020
          total_tco2e: 100000
        - year: 2024
          total_tco2e: 92000  # Expected: ~84,000
    expected:
      progress_report:
        trajectory_status: "behind"
        gap_tco2e:
          greater_than: 5000
        recommendations:
          - action: "Accelerate emission reduction efforts"

  - test_id: SBTI-PROG-003
    name: "Significantly behind - major intervention needed"
    input:
      emissions_history:
        - year: 2020
          total_tco2e: 100000
        - year: 2024
          total_tco2e: 98000  # Only 2% reduction in 4 years
    expected:
      progress_report:
        trajectory_status: "significantly_behind"
        required_annual_reduction_remaining:
          greater_than: 8  # Need >8% annual to catch up

  - test_id: SBTI-PROG-004
    name: "Ahead of target"
    input:
      emissions_history:
        - year: 2020
          total_tco2e: 100000
        - year: 2024
          total_tco2e: 75000  # 25% reduction (expected ~16%)
    expected:
      progress_report:
        trajectory_status: "ahead"
        gap_percent:
          less_than: -5

  - test_id: SBTI-PROG-005
    name: "Emissions increased year-over-year"
    input:
      emissions_history:
        - year: 2023
          total_tco2e: 85000
        - year: 2024
          total_tco2e: 90000  # Increased
    expected:
      progress_report:
        yoy_change_percent:
          greater_than: 0
        trajectory_status: "behind"
        recommendations:
          - category: "emissions_increase"
            action: "Investigate cause of emissions increase"

  - test_id: SBTI-PROG-006
    name: "Scope 3 progress separate tracking"
    input:
      existing_targets:
        scope_3_target:
          base_emissions_tco2e: 500000
          target_emissions_tco2e: 335000
      emissions_history:
        - year: 2020
          scope_3_tco2e: 500000
        - year: 2024
          scope_3_tco2e: 450000
    expected:
      progress_report:
        scope_3:
          actual_reduction_percent: 10
          on_track: true

  - test_id: SBTI-PROG-007
    name: "FLAG progress tracking"
    input:
      existing_targets:
        flag_target:
          base_emissions_tco2e: 200000
          target_emissions_tco2e: 100000
          target_year: 2030
      emissions_history:
        - year: 2020
          flag_tco2e: 200000
        - year: 2024
          flag_tco2e: 160000
    expected:
      progress_report:
        flag:
          actual_reduction_percent: 20
          on_track: true

  - test_id: SBTI-PROG-008
    name: "Net-zero milestone tracking"
    input:
      existing_targets:
        long_term_target:
          net_zero_year: 2050
          base_year: 2020
          base_emissions_tco2e: 500000
      emissions_history:
        - year: 2020
          total_tco2e: 500000
        - year: 2030
          total_tco2e: 300000  # 40% reduction
    expected:
      progress_report:
        net_zero_progress:
          milestone_2030:
            expected_reduction: 35
            actual_reduction: 40
            status: "on_track"

  - test_id: SBTI-PROG-009
    name: "Recalculation trigger - M&A"
    input:
      structural_change:
        type: "acquisition"
        emissions_impact_percent: 15
    expected:
      recalculation_required: true
      reason: "Structural change >5% of base year emissions"

  - test_id: SBTI-PROG-010
    name: "Progress with intensity metric"
    input:
      existing_targets:
        near_term_target:
          target_type: "intensity"
          base_intensity: 100  # tCO2e/$M
          target_intensity: 65
      activity_data:
        base_year:
          revenue_usd: 1000000000
        current_year:
          revenue_usd: 1200000000
      emissions_history:
        - year: 2024
          total_tco2e: 96000  # 80 tCO2e/$M
    expected:
      progress_report:
        intensity_progress:
          current_intensity: 80
          target_intensity: 65
          on_track: true

  - test_id: SBTI-PROG-011
    name: "Multiple target progress summary"
    input:
      existing_targets:
        scope_1_2: true
        scope_3: true
        flag: true
        net_zero: true
    expected:
      progress_report:
        summary:
          total_targets: 4
          on_track: 3
          behind: 1

  - test_id: SBTI-PROG-012
    name: "COVID year adjustment"
    input:
      emissions_history:
        - year: 2019
          total_tco2e: 100000
        - year: 2020
          total_tco2e: 70000  # COVID impact
        - year: 2021
          total_tco2e: 90000  # Recovery
      base_year_adjustment_request: true
    expected:
      recommendation:
        adjust_base_year: true
        suggested_base_year: 2019
        reason: "2020 not representative due to COVID"
```

### 6.3 Criteria Validation Tests (13 tests)

```yaml
golden_tests_criteria_validation:
  - test_id: SBTI-CRIT-001
    name: "Full criteria compliance"
    input:
      targets:
        ambition_level: "1_5C"
        scope_1_2_coverage: 98
        scope_3_required: true
        scope_3_coverage: 70
        timeframe_years: 7
        base_year_age: 1
    expected:
      sbti_criteria_check:
        compliant: true
        checks:
          - criterion: "Ambition level"
            status: "PASS"
          - criterion: "Scope 1&2 coverage"
            status: "PASS"
          - criterion: "Scope 3 coverage"
            status: "PASS"
          - criterion: "Timeframe"
            status: "PASS"

  - test_id: SBTI-CRIT-002
    name: "Scope 1&2 coverage below 95%"
    input:
      targets:
        scope_1_2_coverage: 90
    expected:
      sbti_criteria_check:
        compliant: false
        checks:
          - criterion: "Scope 1&2 coverage"
            status: "FAIL"
            details: "90% coverage below 95% requirement"

  - test_id: SBTI-CRIT-003
    name: "Scope 3 coverage below 67%"
    input:
      targets:
        scope_3_required: true
        scope_3_coverage: 50
    expected:
      sbti_criteria_check:
        checks:
          - criterion: "Scope 3 coverage"
            status: "FAIL"
            details: "50% coverage below 67% requirement"

  - test_id: SBTI-CRIT-004
    name: "Target timeframe 4 years (invalid)"
    input:
      targets:
        timeframe_years: 4
    expected:
      sbti_criteria_check:
        checks:
          - criterion: "Timeframe"
            status: "FAIL"
            details: "4-year timeframe below 5-year minimum"

  - test_id: SBTI-CRIT-005
    name: "Target timeframe 12 years (invalid)"
    input:
      targets:
        timeframe_years: 12
    expected:
      sbti_criteria_check:
        checks:
          - criterion: "Timeframe"
            status: "FAIL"
            details: "12-year timeframe exceeds 10-year maximum"

  - test_id: SBTI-CRIT-006
    name: "Ambition below 1.5C threshold"
    input:
      targets:
        annual_reduction_rate: 3.0  # Below 4.2%
    expected:
      sbti_criteria_check:
        checks:
          - criterion: "Ambition level"
            status: "FAIL"
            details: "3.0% annual reduction below 1.5C requirement (4.2%)"

  - test_id: SBTI-CRIT-007
    name: "Net-zero residual above 10%"
    input:
      targets:
        long_term:
          residual_emissions_percent: 15
    expected:
      sbti_criteria_check:
        checks:
          - criterion: "Net-zero residual"
            status: "FAIL"
            details: "15% residual exceeds 10% maximum"

  - test_id: SBTI-CRIT-008
    name: "FLAG no-deforestation missing"
    input:
      company_profile:
        sector: "food_beverage"
        flag_applicable: true
      targets:
        flag_target:
          no_deforestation_commitment: false
    expected:
      sbti_criteria_check:
        checks:
          - criterion: "FLAG no-deforestation"
            status: "FAIL"
            details: "No-deforestation commitment required for FLAG sectors"

  - test_id: SBTI-CRIT-009
    name: "Intensity target with absolute increase"
    input:
      targets:
        target_type: "intensity"
        intensity_reduction: 30
      emissions_projection:
        absolute_increase: 20  # Absolute emissions projected to increase
    expected:
      sbti_criteria_check:
        checks:
          - criterion: "Intensity target absolute check"
            status: "WARNING"
            details: "Intensity target may allow absolute emissions increase"

  - test_id: SBTI-CRIT-010
    name: "Verification completeness check"
    input:
      submission:
        ghg_inventory: true
        methodology_doc: true
        verification_statement: false
    expected:
      validation_readiness:
        ready: false
        missing_items:
          - "Third-party verification statement"

  - test_id: SBTI-CRIT-011
    name: "Market-based vs location-based Scope 2"
    input:
      emissions_inventory:
        scope_2:
          location_based_tco2e: 50000
          market_based_tco2e: 30000
      targets:
        scope_2_method: "market_based"
    expected:
      sbti_criteria_check:
        checks:
          - criterion: "Scope 2 accounting"
            status: "PASS"
            notes: "Both methods reported, market-based used for target"

  - test_id: SBTI-CRIT-012
    name: "Submission package completeness"
    input:
      submission_package:
        target_submission_form: true
        ghg_inventory: true
        methodology_documentation: true
        board_approval: false
    expected:
      validation_readiness:
        ready: false
        missing_items:
          - "Board/executive approval documentation"

  - test_id: SBTI-CRIT-013
    name: "Criteria version compatibility"
    input:
      submission:
        criteria_version: "4.2"
        submission_date: "2025-01-01"
    expected:
      sbti_criteria_check:
        checks:
          - criterion: "Criteria version"
            status: "WARNING"
            details: "Criteria v4.2 outdated, v5.1 now in effect"
```

### 6.4 Sector Pathway Tests (10 tests)

```yaml
golden_tests_sector_pathways:
  - test_id: SBTI-PATH-001
    name: "Power generation 2030 pathway"
    input:
      sector: "power_generation"
      target_year: 2030
      current_intensity: 450  # gCO2/kWh
    expected:
      sector_pathway_comparison:
        pathway_target: 138
        current_vs_pathway: "above"
        gap: 312

  - test_id: SBTI-PATH-002
    name: "Steel industry 2030 pathway"
    input:
      sector: "steel"
      target_year: 2030
      current_intensity: 1.89  # tCO2/t
    expected:
      sector_pathway_comparison:
        pathway_target: 1.28
        required_reduction_percent: 32

  - test_id: SBTI-PATH-003
    name: "Cement industry pathway"
    input:
      sector: "cement"
      target_year: 2030
      current_intensity: 611
    expected:
      sector_pathway_comparison:
        pathway_target: 469

  - test_id: SBTI-PATH-004
    name: "Aluminum industry pathway"
    input:
      sector: "aluminum"
      target_year: 2030
      current_intensity: 11.5
    expected:
      sector_pathway_comparison:
        pathway_target: 3.3
        required_reduction_percent: 71

  - test_id: SBTI-PATH-005
    name: "Road transport pathway"
    input:
      sector: "road_transport"
      target_year: 2030
    expected:
      sector_pathway_comparison:
        metric: "gCO2/pkm"
        pathway_target: 70

  - test_id: SBTI-PATH-006
    name: "Aviation sector pathway"
    input:
      sector: "transport_aviation"
      target_year: 2030
    expected:
      sector_pathway_comparison:
        metric: "gCO2/RTK"
        pathway_source: "SBTi Aviation Guidance"

  - test_id: SBTI-PATH-007
    name: "Shipping sector pathway"
    input:
      sector: "transport_shipping"
      target_year: 2030
    expected:
      sector_pathway_comparison:
        metric: "gCO2/transport work"
        pathway_source: "IMO 2050 strategy"

  - test_id: SBTI-PATH-008
    name: "Buildings sector pathway"
    input:
      sector: "buildings"
      target_year: 2030
    expected:
      sector_pathway_comparison:
        metric: "kgCO2/m2"

  - test_id: SBTI-PATH-009
    name: "Chemical sector (no specific pathway)"
    input:
      sector: "chemicals"
    expected:
      methodology_recommendation: "ACA"
      reason: "No sector-specific pathway available, use Absolute Contraction"

  - test_id: SBTI-PATH-010
    name: "Multi-sector company pathway selection"
    input:
      company_profile:
        sectors:
          - sector: "power_generation"
            emissions_share: 60
          - sector: "manufacturing"
            emissions_share: 40
    expected:
      methodology_recommendation:
        power_generation: "SDA"
        manufacturing: "ACA"
```

---

## 7. Data Dependencies

```yaml
data_dependencies:
  sbti_resources:
    criteria:
      name: "SBTi Criteria v5.1"
      url: "https://sciencebasedtargets.org/resources/"
      update: "Annual"

    target_setting_tool:
      name: "SBTi Target Setting Tool"
      version: "2.0"
      format: "Excel"

    sda_tool:
      name: "SDA Tool"
      version: "1.0"
      sectors: ["Power", "Steel", "Cement", "Aluminum", "Transport"]

  sector_pathways:
    iea_nze:
      name: "IEA Net Zero by 2050"
      source: "International Energy Agency"

    ipcc_pathways:
      name: "IPCC 1.5C Pathways"
      source: "IPCC SR15"

  carbon_budgets:
    global_budget:
      source: "IPCC AR6"
      1_5C_budget_from_2020: "400 Gt CO2"
      well_below_2C_budget: "1150 Gt CO2"
```

---

## 8. Implementation Roadmap

```yaml
implementation_roadmap:
  phase_1_core:
    duration: "Weeks 1-4"
    deliverables:
      - "ACA target calculator"
      - "Scope coverage validator"
      - "Basic criteria checker"

  phase_2_sda:
    duration: "Weeks 5-8"
    deliverables:
      - "SDA pathway calculator"
      - "Sector pathway database"
      - "Intensity target support"

  phase_3_advanced:
    duration: "Weeks 9-12"
    deliverables:
      - "Net-zero calculator"
      - "FLAG targets"
      - "Progress tracker"

  phase_4_integration:
    duration: "Weeks 13-16"
    deliverables:
      - "Submission package generator"
      - "API integration"
      - "Dashboard"
```

---

## 9. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-ProductManager | Initial specification |

**Approvals:**

- Climate Science Lead: ___________________ Date: _______
- SBTi Methodology Expert: ___________________ Date: _______
- Engineering Lead: ___________________ Date: _______
- Product Manager: ___________________ Date: _______

---

**END OF SPECIFICATION**
