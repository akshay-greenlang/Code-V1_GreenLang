# GL-011 Climate Risk Agent Specification

**Agent ID:** gl-011-climate-risk-v1
**Version:** 1.0.0
**Date:** 2025-12-04
**Priority:** P4-STANDARD
**Deadline:** Ongoing (TCFD/ISSB reporting cycles)
**Status:** SPECIFICATION COMPLETE

---

## 1. Executive Summary

### 1.1 Framework Overview

**Task Force on Climate-related Financial Disclosures (TCFD) and ISSB**

The TCFD framework and ISSB IFRS S2 Climate-related Disclosures standard provide comprehensive guidance for companies to assess and disclose climate-related financial risks and opportunities. This agent automates climate risk assessment, scenario analysis, and financial impact quantification.

**Key Requirements:**
- Physical risk assessment (acute and chronic)
- Transition risk assessment (policy, technology, market, reputation)
- Scenario analysis using RCP/SSP pathways
- Financial impact quantification
- Resilience strategy assessment
- Opportunity identification

### 1.2 Agent Purpose

The Climate Risk Agent automates TCFD-aligned climate risk assessment and financial disclosure. It provides:

1. **Physical Risk Analysis** - Assess acute and chronic physical climate risks
2. **Transition Risk Analysis** - Evaluate policy, technology, market, and reputation risks
3. **Scenario Analysis** - Model impacts under RCP 2.6, RCP 4.5, RCP 8.5 / SSP pathways
4. **Financial Quantification** - Translate climate risks to financial impacts
5. **Resilience Assessment** - Evaluate organizational climate resilience
6. **Opportunity Identification** - Identify climate-related business opportunities

---

## 2. TCFD Framework Specification

### 2.1 TCFD Recommendations

```yaml
tcfd_framework:
  version: "2021 Final Recommendations"
  pillars:
    governance:
      description: "Organization's governance around climate-related risks and opportunities"
      disclosures:
        - "Board oversight of climate-related risks and opportunities"
        - "Management's role in assessing and managing climate risks"

    strategy:
      description: "Actual and potential impacts on business, strategy, and financial planning"
      disclosures:
        - "Climate-related risks and opportunities identified (short, medium, long-term)"
        - "Impact on business, strategy, and financial planning"
        - "Resilience of strategy under different climate scenarios"

    risk_management:
      description: "Processes for identifying, assessing, and managing climate-related risks"
      disclosures:
        - "Processes for identifying and assessing climate-related risks"
        - "Processes for managing climate-related risks"
        - "Integration into overall risk management"

    metrics_targets:
      description: "Metrics and targets used to assess and manage climate-related risks"
      disclosures:
        - "Metrics used to assess climate-related risks and opportunities"
        - "Scope 1, 2, and 3 GHG emissions"
        - "Targets and performance against targets"
```

### 2.2 ISSB IFRS S2 Alignment

```yaml
issb_ifrs_s2:
  standard: "IFRS S2 Climate-related Disclosures"
  effective_date: "2024-01-01"
  jurisdiction: "Global (adopted by various jurisdictions)"

  core_content:
    climate_related_risks_opportunities:
      - "Physical risks"
      - "Transition risks"
      - "Climate-related opportunities"

    time_horizons:
      short_term: "Up to 1 year"
      medium_term: "1 to 5 years"
      long_term: "Beyond 5 years"

    financial_statement_connection:
      - "Current period financial effects"
      - "Anticipated financial effects"
      - "Capital deployment toward climate response"

  scenario_analysis:
    requirement: "Mandatory"
    scenarios: "At least two scenarios including 2C or lower"
    alignment: "Compatible with Paris Agreement goals"
```

### 2.3 Risk Categories

```yaml
climate_risk_categories:
  physical_risks:
    acute:
      description: "Event-driven risks"
      hazards:
        - name: "Tropical cyclones"
          aliases: ["hurricanes", "typhoons"]
          metrics: ["wind_speed", "precipitation", "storm_surge"]

        - name: "Floods"
          types: ["fluvial", "pluvial", "coastal"]
          metrics: ["depth", "duration", "frequency"]

        - name: "Wildfires"
          metrics: ["fire_weather_index", "burned_area"]

        - name: "Extreme heat"
          metrics: ["temperature_anomaly", "heat_wave_duration"]

        - name: "Extreme cold"
          metrics: ["cold_wave_duration", "frost_days"]

        - name: "Drought"
          metrics: ["spi_index", "soil_moisture", "duration"]

        - name: "Severe storms"
          types: ["hail", "tornado", "thunderstorm"]

    chronic:
      description: "Longer-term shifts in climate patterns"
      hazards:
        - name: "Sea level rise"
          metrics: ["mean_sea_level_change_m", "coastal_erosion"]

        - name: "Temperature increase"
          metrics: ["mean_annual_temp_change_c", "cooling_degree_days"]

        - name: "Precipitation changes"
          metrics: ["annual_precip_change_percent", "seasonality_shift"]

        - name: "Water stress"
          metrics: ["water_stress_index", "aquifer_depletion"]

        - name: "Biodiversity loss"
          metrics: ["species_richness_change", "ecosystem_degradation"]

  transition_risks:
    policy_legal:
      description: "Risks from policy actions to address climate change"
      types:
        - "Carbon pricing (tax, ETS)"
        - "Emissions regulations"
        - "Energy efficiency mandates"
        - "Climate litigation"
        - "Disclosure requirements"

    technology:
      description: "Risks from technological changes for low-carbon transition"
      types:
        - "Stranded assets (fossil fuel)"
        - "Disruption from new technologies"
        - "Cost of transition technologies"
        - "Failed technology investments"

    market:
      description: "Risks from shifts in supply and demand"
      types:
        - "Changing customer preferences"
        - "Commodity price volatility"
        - "Stranded assets (demand shift)"
        - "Supply chain disruption"

    reputation:
      description: "Risks from changing stakeholder perceptions"
      types:
        - "Consumer sentiment shifts"
        - "Investor ESG screening"
        - "Talent attraction/retention"
        - "Greenwashing accusations"

  climate_opportunities:
    categories:
      resource_efficiency:
        - "Energy efficiency improvements"
        - "Water efficiency improvements"
        - "Circular economy adoption"

      energy_source:
        - "Renewable energy use"
        - "Clean energy products"
        - "Energy storage solutions"

      products_services:
        - "Low-carbon products"
        - "Climate adaptation services"
        - "R&D and innovation"

      markets:
        - "New market access"
        - "Public sector incentives"
        - "Green finance access"

      resilience:
        - "Supply chain diversification"
        - "Business continuity planning"
        - "Resource substitution"
```

---

## 3. Scenario Analysis Framework

### 3.1 Climate Scenarios

```yaml
climate_scenarios:
  rcp_pathways:
    description: "Representative Concentration Pathways (IPCC AR5)"

    rcp_2_6:
      name: "Paris-aligned / Low emissions"
      description: "Strong mitigation, 1.5-2C warming by 2100"
      peak_year: 2020
      2100_warming: "1.5-2.0 C"
      co2_concentration_2100: "490 ppm"
      physical_risk: "LOW"
      transition_risk: "HIGH"
      policy_assumption: "Aggressive climate policy from 2025"
      carbon_price_2030: "$100-200/tCO2"
      carbon_price_2050: "$200-500/tCO2"

    rcp_4_5:
      name: "Middle road / Moderate emissions"
      description: "Moderate mitigation, ~2.5C warming by 2100"
      peak_year: 2040
      2100_warming: "2.0-3.0 C"
      co2_concentration_2100: "650 ppm"
      physical_risk: "MEDIUM"
      transition_risk: "MEDIUM"
      policy_assumption: "Current policies + moderate strengthening"
      carbon_price_2030: "$30-60/tCO2"
      carbon_price_2050: "$60-100/tCO2"

    rcp_8_5:
      name: "Business as usual / High emissions"
      description: "Limited mitigation, 4-5C warming by 2100"
      peak_year: 2100+
      2100_warming: "4.0-5.5 C"
      co2_concentration_2100: "1370 ppm"
      physical_risk: "VERY HIGH"
      transition_risk: "LOW"
      policy_assumption: "Current policies only"
      carbon_price_2030: "$0-20/tCO2"
      carbon_price_2050: "$0-30/tCO2"

  ssp_pathways:
    description: "Shared Socioeconomic Pathways (IPCC AR6)"

    ssp1_1_9:
      name: "Sustainability - Taking the Green Road"
      warming: "1.4C by 2100"
      narrative: "Low challenges to mitigation and adaptation"

    ssp1_2_6:
      name: "Sustainability with 2C target"
      warming: "1.8C by 2100"

    ssp2_4_5:
      name: "Middle of the Road"
      warming: "2.7C by 2100"
      narrative: "Medium challenges to mitigation and adaptation"

    ssp3_7_0:
      name: "Regional Rivalry"
      warming: "3.6C by 2100"
      narrative: "High challenges to mitigation and adaptation"

    ssp5_8_5:
      name: "Fossil-fueled Development"
      warming: "4.4C by 2100"
      narrative: "High challenges to mitigation, low to adaptation"

  time_horizons:
    short_term:
      period: "2025-2030"
      physical_risk_relevance: "MEDIUM"
      transition_risk_relevance: "HIGH"

    medium_term:
      period: "2030-2040"
      physical_risk_relevance: "HIGH"
      transition_risk_relevance: "HIGH"

    long_term:
      period: "2040-2050+"
      physical_risk_relevance: "VERY HIGH"
      transition_risk_relevance: "MEDIUM"
```

### 3.2 Physical Risk Data Sources

```yaml
physical_risk_data:
  global_datasets:
    wri_aqueduct:
      name: "WRI Aqueduct"
      coverage: "Global"
      hazards: ["Water stress", "Flood risk", "Drought risk"]
      resolution: "Sub-basin"
      url: "https://www.wri.org/aqueduct"

    climate_central:
      name: "Climate Central Coastal Risk"
      coverage: "Coastal areas globally"
      hazards: ["Sea level rise", "Coastal flooding"]
      resolution: "Property-level"

    think_hazard:
      name: "ThinkHazard!"
      provider: "World Bank / GFDRR"
      coverage: "Global"
      hazards: ["Multi-hazard"]
      url: "https://thinkhazard.org"

    noaa_slr:
      name: "NOAA Sea Level Rise Viewer"
      coverage: "US coastal"
      resolution: "High resolution"

  climate_projections:
    cmip6:
      name: "CMIP6 Climate Projections"
      source: "IPCC AR6"
      variables: ["Temperature", "Precipitation", "Extreme events"]

    cordex:
      name: "CORDEX Regional Projections"
      resolution: "25-50 km"
      coverage: "Regional domains"

  proprietary_providers:
    - name: "Jupiter Intelligence"
      specialty: "Physical risk analytics"
    - name: "Four Twenty Seven (Moody's)"
      specialty: "Climate risk scores"
    - name: "XDI"
      specialty: "Asset-level physical risk"
    - name: "Cervest"
      specialty: "EarthScan climate intelligence"
```

---

## 4. Agent Architecture

### 4.1 Agent Specification (AgentSpec v2)

```yaml
agent_id: gl-011-climate-risk-v1
name: "Climate Risk Assessment Agent"
version: "1.0.0"
type: risk-assessment
priority: P4-STANDARD
deadline: "ongoing"

description: |
  TCFD/ISSB-aligned climate risk assessment agent. Analyzes physical and
  transition risks, performs scenario analysis, quantifies financial impacts,
  and generates disclosure-ready reports.

framework_context:
  frameworks:
    - name: "TCFD"
      version: "2021 Final Recommendations"
    - name: "ISSB IFRS S2"
      version: "2024"
  alignment: "Full TCFD/ISSB compliance"

inputs:
  company_profile:
    type: object
    required: true
    properties:
      company_id:
        type: string

      company_name:
        type: string

      sector:
        type: string
        enum: ["energy", "materials", "industrials", "consumer_discretionary", "consumer_staples", "healthcare", "financials", "information_technology", "communication_services", "utilities", "real_estate"]

      industry:
        type: string

      revenue_usd:
        type: number

      market_cap_usd:
        type: number

      geographic_operations:
        type: array
        items:
          type: string
          description: "ISO country codes"

  asset_inventory:
    type: array
    required: true
    description: "Physical assets and their locations"
    items:
      type: object
      properties:
        asset_id:
          type: string

        asset_name:
          type: string

        asset_type:
          type: string
          enum: ["facility", "equipment", "real_estate", "infrastructure", "inventory", "vehicle_fleet"]

        location:
          type: object
          properties:
            address:
              type: string
            latitude:
              type: number
              minimum: -90
              maximum: 90
            longitude:
              type: number
              minimum: -180
              maximum: 180
            country:
              type: string
            region:
              type: string

        asset_value_usd:
          type: number

        replacement_cost_usd:
          type: number

        useful_life_years:
          type: number

        business_criticality:
          type: string
          enum: ["critical", "important", "standard", "low"]

        climate_sensitivity:
          type: object
          properties:
            flood_sensitive:
              type: boolean
            heat_sensitive:
              type: boolean
            wind_sensitive:
              type: boolean
            water_dependent:
              type: boolean
            coastal:
              type: boolean

  operational_data:
    type: object
    properties:
      revenue_by_region:
        type: object
        description: "Revenue breakdown by region"
        additionalProperties:
          type: number

      supply_chain:
        type: array
        items:
          type: object
          properties:
            supplier_id:
              type: string
            supplier_name:
              type: string
            location:
              type: object
            spend_usd:
              type: number
            criticality:
              type: string

      energy_profile:
        type: object
        properties:
          total_energy_mwh:
            type: number
          fossil_fuel_percent:
            type: number
          renewable_percent:
            type: number
          electricity_spend_usd:
            type: number

      carbon_exposure:
        type: object
        properties:
          scope_1_tco2e:
            type: number
          scope_2_tco2e:
            type: number
          scope_3_tco2e:
            type: number
          carbon_intensity:
            type: number
            description: "tCO2e per $M revenue"

  financial_data:
    type: object
    properties:
      revenue_usd:
        type: number
      ebitda_usd:
        type: number
      total_assets_usd:
        type: number
      net_income_usd:
        type: number
      capex_usd:
        type: number
      r_and_d_spend_usd:
        type: number
      debt_usd:
        type: number
      cost_of_capital:
        type: number
        description: "WACC %"

  analysis_parameters:
    type: object
    properties:
      scenarios:
        type: array
        items:
          type: string
          enum: ["rcp_2_6", "rcp_4_5", "rcp_8_5", "ssp1_1_9", "ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
        default: ["rcp_2_6", "rcp_4_5", "rcp_8_5"]

      time_horizons:
        type: array
        items:
          type: string
          enum: ["2030", "2040", "2050"]
        default: ["2030", "2050"]

      risk_appetite:
        type: string
        enum: ["conservative", "moderate", "aggressive"]
        default: "moderate"

      confidence_level:
        type: number
        default: 0.95
        description: "For VaR calculations"

outputs:
  climate_risk_assessment:
    type: object
    properties:
      assessment_id:
        type: string
        format: uuid

      assessment_date:
        type: string
        format: date-time

      executive_summary:
        type: object
        properties:
          overall_risk_rating:
            type: string
            enum: ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
          physical_risk_rating:
            type: string
          transition_risk_rating:
            type: string
          total_value_at_risk_usd:
            type: number
          key_findings:
            type: array
            items:
              type: string
          priority_actions:
            type: array
            items:
              type: string

      physical_risk_analysis:
        type: object
        properties:
          acute_risks:
            type: array
            items:
              type: object
              properties:
                hazard:
                  type: string
                exposure_level:
                  type: string
                  enum: ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
                probability:
                  type: number
                  description: "Annual probability"
                assets_at_risk:
                  type: array
                  items:
                    type: string
                value_at_risk_usd:
                  type: number
                time_horizon:
                  type: string
                scenario:
                  type: string

          chronic_risks:
            type: array
            items:
              type: object
              properties:
                hazard:
                  type: string
                exposure_level:
                  type: string
                projected_change:
                  type: object
                  properties:
                    metric:
                      type: string
                    baseline:
                      type: number
                    projected_2030:
                      type: number
                    projected_2050:
                      type: number
                financial_impact_usd:
                  type: number
                time_horizon:
                  type: string

          geographic_hotspots:
            type: array
            items:
              type: object
              properties:
                location:
                  type: string
                risk_score:
                  type: number
                primary_hazards:
                  type: array
                  items:
                    type: string
                asset_exposure_usd:
                  type: number

      transition_risk_analysis:
        type: object
        properties:
          policy_legal_risks:
            type: array
            items:
              type: object
              properties:
                risk_name:
                  type: string
                description:
                  type: string
                likelihood:
                  type: string
                  enum: ["LOW", "MEDIUM", "HIGH"]
                financial_impact_usd:
                  type: number
                time_horizon:
                  type: string
                scenario:
                  type: string

          technology_risks:
            type: array

          market_risks:
            type: array

          reputation_risks:
            type: array

          carbon_pricing_impact:
            type: object
            properties:
              current_carbon_cost_usd:
                type: number
              projected_carbon_cost:
                type: object
                additionalProperties:
                  type: number
              ebitda_impact_percent:
                type: object
                additionalProperties:
                  type: number

      scenario_analysis_results:
        type: object
        properties:
          scenarios_analyzed:
            type: array
            items:
              type: string

          scenario_impacts:
            type: array
            items:
              type: object
              properties:
                scenario:
                  type: string
                time_horizon:
                  type: string
                physical_risk_impact_usd:
                  type: number
                transition_risk_impact_usd:
                  type: number
                opportunity_value_usd:
                  type: number
                net_impact_usd:
                  type: number
                revenue_impact_percent:
                  type: number
                ebitda_impact_percent:
                  type: number
                asset_value_impact_percent:
                  type: number

      climate_opportunities:
        type: array
        items:
          type: object
          properties:
            opportunity_name:
              type: string
            category:
              type: string
              enum: ["resource_efficiency", "energy_source", "products_services", "markets", "resilience"]
            description:
              type: string
            estimated_value_usd:
              type: number
            investment_required_usd:
              type: number
            payback_years:
              type: number
            implementation_timeline:
              type: string

      resilience_assessment:
        type: object
        properties:
          overall_resilience_score:
            type: number
            minimum: 0
            maximum: 100

          governance_score:
            type: number
          strategy_score:
            type: number
          risk_management_score:
            type: number
          adaptation_score:
            type: number

          gaps:
            type: array
            items:
              type: string

          recommendations:
            type: array
            items:
              type: object
              properties:
                recommendation:
                  type: string
                priority:
                  type: string
                estimated_cost_usd:
                  type: number
                risk_reduction_percent:
                  type: number

      financial_impact_summary:
        type: object
        properties:
          total_value_at_risk:
            type: object
            additionalProperties:
              type: number
            description: "VaR by scenario"

          stressed_revenue_impact:
            type: object
            additionalProperties:
              type: number

          stressed_ebitda_impact:
            type: object
            additionalProperties:
              type: number

          capital_expenditure_needs:
            type: number
            description: "Estimated CapEx for adaptation/mitigation"

          stranded_asset_risk_usd:
            type: number

      tcfd_disclosure_readiness:
        type: object
        properties:
          governance:
            type: object
            properties:
              score:
                type: number
              gaps:
                type: array
          strategy:
            type: object
          risk_management:
            type: object
          metrics_targets:
            type: object

      audit_trail:
        type: object
        properties:
          input_hash:
            type: string
          output_hash:
            type: string
          data_sources:
            type: array
            items:
              type: string
          methodology_version:
            type: string

  tcfd_disclosure_report:
    type: object
    description: "Disclosure-ready TCFD report"

  issb_disclosure_package:
    type: object
    description: "IFRS S2 compliant disclosure package"

tools:
  - name: physical_risk_analyzer
    type: analyzer
    description: "Analyze physical climate risks for assets"
    inputs: ["asset_inventory", "climate_projections"]
    outputs: ["physical_risk_assessment"]

  - name: hazard_exposure_calculator
    type: calculator
    description: "Calculate exposure to specific climate hazards"
    inputs: ["location", "hazard_type", "scenario"]
    outputs: ["exposure_score", "probability", "severity"]

  - name: transition_risk_analyzer
    type: analyzer
    description: "Analyze transition risks"
    inputs: ["company_profile", "carbon_exposure", "scenario"]
    outputs: ["transition_risk_assessment"]

  - name: carbon_pricing_modeler
    type: calculator
    description: "Model carbon pricing impacts"
    inputs: ["emissions_data", "carbon_price_scenarios"]
    outputs: ["carbon_cost_projections"]

  - name: scenario_modeler
    type: calculator
    description: "Run scenario analysis"
    inputs: ["company_data", "scenarios", "time_horizons"]
    outputs: ["scenario_results"]

  - name: var_calculator
    type: calculator
    description: "Calculate climate Value at Risk"
    inputs: ["risk_exposures", "probabilities", "confidence_level"]
    outputs: ["var_results"]

  - name: opportunity_identifier
    type: analyzer
    description: "Identify climate-related opportunities"
    inputs: ["company_profile", "sector_trends"]
    outputs: ["opportunities"]

  - name: resilience_assessor
    type: analyzer
    description: "Assess organizational climate resilience"
    inputs: ["company_data", "risk_assessment"]
    outputs: ["resilience_score", "recommendations"]

  - name: tcfd_report_generator
    type: generator
    description: "Generate TCFD disclosure report"
    inputs: ["climate_risk_assessment"]
    outputs: ["tcfd_report"]

  - name: financial_impact_calculator
    type: calculator
    description: "Calculate financial impacts of climate risks"
    inputs: ["risk_data", "financial_data"]
    outputs: ["financial_impacts"]

  - name: provenance_tracker
    type: utility
    description: "Track analysis provenance"
    inputs: ["inputs", "outputs"]
    outputs: ["provenance_hash"]

evaluation:
  golden_tests:
    total_count: 50
    categories:
      physical_risk: 15
      transition_risk: 12
      scenario_analysis: 13
      financial_impact: 10

  accuracy_thresholds:
    risk_rating: 0.85
    financial_impact: 0.20  # Within 20%
    scenario_consistency: 0.90

  benchmarks:
    latency_p95_seconds: 60
    cost_per_analysis_usd: 1.00

certification:
  required_approvals:
    - climate_science_team
    - risk_management_team
    - finance_team

  compliance_checks:
    - tcfd_alignment
    - issb_s2_compliance
    - scenario_methodology
```

---

## 5. Calculation Formulas

### 5.1 Physical Risk Scoring

```python
def calculate_physical_risk_score(
    asset: dict,
    hazard: str,
    scenario: str,
    time_horizon: str
) -> PhysicalRiskResult:
    """
    Calculate physical climate risk score for an asset.

    Formula:
    Risk_Score = Hazard_Probability * Vulnerability * Exposure_Value

    Where:
    - Hazard_Probability = Climate projection probability for hazard
    - Vulnerability = Asset sensitivity to hazard (0-1)
    - Exposure_Value = Asset value at risk

    Source: TCFD Physical Risk Framework, UNEP FI methodologies
    """

    # Get hazard probability from climate projections
    hazard_prob = get_hazard_probability(
        location=asset["location"],
        hazard=hazard,
        scenario=scenario,
        time_horizon=time_horizon
    )

    # Calculate vulnerability based on asset characteristics
    vulnerability = calculate_vulnerability(
        asset_type=asset["asset_type"],
        hazard=hazard,
        sensitivity=asset.get("climate_sensitivity", {})
    )

    # Exposure value
    exposure_value = asset.get("replacement_cost_usd", asset.get("asset_value_usd", 0))

    # Risk score (Expected Annual Loss approach)
    risk_score = hazard_prob * vulnerability * exposure_value

    # Categorize risk level
    if risk_score < exposure_value * 0.01:
        risk_level = "LOW"
    elif risk_score < exposure_value * 0.05:
        risk_level = "MEDIUM"
    elif risk_score < exposure_value * 0.10:
        risk_level = "HIGH"
    else:
        risk_level = "VERY_HIGH"

    return PhysicalRiskResult(
        asset_id=asset["asset_id"],
        hazard=hazard,
        scenario=scenario,
        time_horizon=time_horizon,
        hazard_probability=hazard_prob,
        vulnerability=vulnerability,
        exposure_value=exposure_value,
        expected_annual_loss=risk_score,
        risk_level=risk_level
    )

def calculate_vulnerability(
    asset_type: str,
    hazard: str,
    sensitivity: dict
) -> float:
    """
    Calculate asset vulnerability to specific hazard.

    Vulnerability factors based on asset type and hazard combination.
    """
    vulnerability_matrix = {
        "facility": {
            "flood": 0.4 if sensitivity.get("flood_sensitive") else 0.2,
            "tropical_cyclone": 0.3,
            "extreme_heat": 0.2 if sensitivity.get("heat_sensitive") else 0.1,
            "wildfire": 0.5,
            "sea_level_rise": 0.6 if sensitivity.get("coastal") else 0.1,
            "drought": 0.1 if sensitivity.get("water_dependent") else 0.05
        },
        "equipment": {
            "flood": 0.6,
            "tropical_cyclone": 0.4,
            "extreme_heat": 0.3,
            "wildfire": 0.7,
            "sea_level_rise": 0.5,
            "drought": 0.05
        },
        "real_estate": {
            "flood": 0.5,
            "tropical_cyclone": 0.4,
            "extreme_heat": 0.1,
            "wildfire": 0.8,
            "sea_level_rise": 0.7,
            "drought": 0.1
        }
    }

    default_vulnerability = 0.3
    asset_vulns = vulnerability_matrix.get(asset_type, {})
    return asset_vulns.get(hazard, default_vulnerability)
```

### 5.2 Transition Risk Assessment

```python
def calculate_transition_risk(
    company_profile: dict,
    carbon_exposure: dict,
    scenario: str,
    time_horizon: str
) -> TransitionRiskResult:
    """
    Calculate transition risk exposure.

    Components:
    1. Carbon Pricing Risk = Emissions * Carbon Price
    2. Technology Risk = Stranded asset exposure
    3. Market Risk = Revenue at risk from demand shifts
    4. Reputation Risk = Brand value impact

    Source: TCFD Transition Risk Framework
    """

    # 1. Carbon Pricing Risk
    carbon_price = get_carbon_price(scenario, time_horizon)
    total_emissions = (
        carbon_exposure.get("scope_1_tco2e", 0) +
        carbon_exposure.get("scope_2_tco2e", 0) +
        carbon_exposure.get("scope_3_tco2e", 0) * 0.1  # Partial Scope 3 exposure
    )
    carbon_cost = total_emissions * carbon_price

    # 2. Technology/Stranded Asset Risk
    fossil_exposure = calculate_fossil_asset_exposure(company_profile)
    stranded_asset_risk = fossil_exposure * get_stranding_probability(scenario, time_horizon)

    # 3. Market Risk
    market_risk = calculate_market_transition_risk(
        sector=company_profile["sector"],
        revenue=company_profile["revenue_usd"],
        scenario=scenario
    )

    # 4. Reputation Risk
    reputation_risk = calculate_reputation_risk(
        carbon_intensity=carbon_exposure.get("carbon_intensity", 0),
        sector_average=get_sector_carbon_intensity(company_profile["sector"]),
        revenue=company_profile["revenue_usd"]
    )

    total_transition_risk = (
        carbon_cost +
        stranded_asset_risk +
        market_risk +
        reputation_risk
    )

    return TransitionRiskResult(
        scenario=scenario,
        time_horizon=time_horizon,
        carbon_pricing_cost_usd=carbon_cost,
        stranded_asset_risk_usd=stranded_asset_risk,
        market_risk_usd=market_risk,
        reputation_risk_usd=reputation_risk,
        total_transition_risk_usd=total_transition_risk,
        ebitda_impact_percent=(total_transition_risk / company_profile.get("ebitda_usd", 1)) * 100
    )

def get_carbon_price(scenario: str, time_horizon: str) -> float:
    """Get projected carbon price for scenario and time horizon."""
    carbon_prices = {
        "rcp_2_6": {
            "2030": 150,
            "2040": 250,
            "2050": 400
        },
        "rcp_4_5": {
            "2030": 50,
            "2040": 80,
            "2050": 120
        },
        "rcp_8_5": {
            "2030": 15,
            "2040": 20,
            "2050": 30
        }
    }
    return carbon_prices.get(scenario, {}).get(time_horizon, 50)
```

### 5.3 Climate Value at Risk (CVaR)

```python
def calculate_climate_var(
    risk_exposures: list,
    confidence_level: float = 0.95,
    time_horizon: str = "2030"
) -> ClimateVaRResult:
    """
    Calculate Climate Value at Risk.

    Formula:
    CVaR = Total_Asset_Value * (1 - (1 - Loss_Probability)^Years) * Loss_Given_Event

    Aggregated across all risk exposures using Monte Carlo simulation.

    Source: UNEP FI Physical Risk Framework
    """
    import numpy as np

    # Monte Carlo parameters
    n_simulations = 10000
    years = int(time_horizon) - 2025

    total_losses = []

    for _ in range(n_simulations):
        simulation_loss = 0

        for exposure in risk_exposures:
            # Sample whether event occurs
            event_prob = exposure["annual_probability"]
            event_occurs = np.random.random() < (1 - (1 - event_prob) ** years)

            if event_occurs:
                # Sample loss severity (beta distribution)
                loss_severity = np.random.beta(2, 5)  # Skewed toward lower losses
                loss = exposure["exposure_value"] * loss_severity * exposure["vulnerability"]
                simulation_loss += loss

        total_losses.append(simulation_loss)

    total_losses = np.array(total_losses)

    # Calculate VaR at confidence level
    var = np.percentile(total_losses, confidence_level * 100)

    # Calculate Expected Shortfall (CVaR)
    cvar = total_losses[total_losses >= var].mean()

    return ClimateVaRResult(
        time_horizon=time_horizon,
        confidence_level=confidence_level,
        value_at_risk_usd=var,
        conditional_var_usd=cvar,
        expected_loss_usd=total_losses.mean(),
        worst_case_loss_usd=total_losses.max(),
        loss_distribution=total_losses
    )
```

### 5.4 Scenario Impact Analysis

```python
def run_scenario_analysis(
    company_data: dict,
    scenario: str,
    time_horizon: str
) -> ScenarioAnalysisResult:
    """
    Run comprehensive scenario analysis.

    Calculates net impact considering:
    - Physical risk costs
    - Transition risk costs
    - Climate opportunities

    Source: TCFD Scenario Analysis Guidance
    """

    # Physical risk impact
    physical_risks = calculate_total_physical_risk(
        assets=company_data["asset_inventory"],
        scenario=scenario,
        time_horizon=time_horizon
    )
    physical_impact = sum(r["expected_annual_loss"] for r in physical_risks)

    # Transition risk impact
    transition_risk = calculate_transition_risk(
        company_profile=company_data["company_profile"],
        carbon_exposure=company_data["carbon_exposure"],
        scenario=scenario,
        time_horizon=time_horizon
    )
    transition_impact = transition_risk.total_transition_risk_usd

    # Opportunity value
    opportunities = identify_opportunities(
        company_profile=company_data["company_profile"],
        sector=company_data["company_profile"]["sector"],
        scenario=scenario
    )
    opportunity_value = sum(opp["estimated_value_usd"] for opp in opportunities)

    # Net impact
    net_impact = physical_impact + transition_impact - opportunity_value

    # Calculate as percentage of key financial metrics
    revenue = company_data.get("financial_data", {}).get("revenue_usd", 1)
    ebitda = company_data.get("financial_data", {}).get("ebitda_usd", 1)
    total_assets = company_data.get("financial_data", {}).get("total_assets_usd", 1)

    return ScenarioAnalysisResult(
        scenario=scenario,
        time_horizon=time_horizon,
        physical_risk_impact_usd=physical_impact,
        transition_risk_impact_usd=transition_impact,
        opportunity_value_usd=opportunity_value,
        net_impact_usd=net_impact,
        revenue_impact_percent=(net_impact / revenue) * 100,
        ebitda_impact_percent=(net_impact / ebitda) * 100,
        asset_value_impact_percent=(net_impact / total_assets) * 100
    )
```

---

## 6. Golden Test Scenarios

### 6.1 Physical Risk Tests (15 tests)

```yaml
golden_tests_physical_risk:
  - test_id: CR-PHYS-001
    name: "Coastal facility flood risk RCP 8.5"
    input:
      asset_inventory:
        - asset_id: "FAC-001"
          asset_type: "facility"
          location:
            latitude: 25.7617
            longitude: -80.1918  # Miami
            country: "US"
          asset_value_usd: 50000000
          climate_sensitivity:
            flood_sensitive: true
            coastal: true
      analysis_parameters:
        scenarios: ["rcp_8_5"]
        time_horizons: ["2050"]
    expected:
      physical_risk_analysis:
        acute_risks:
          - hazard: "flood"
            exposure_level: "VERY_HIGH"
            value_at_risk_usd:
              range: [10000000, 30000000]

  - test_id: CR-PHYS-002
    name: "Inland facility low flood risk"
    input:
      asset_inventory:
        - asset_id: "FAC-002"
          location:
            latitude: 39.7392
            longitude: -104.9903  # Denver (high elevation)
          asset_value_usd: 50000000
          climate_sensitivity:
            coastal: false
    expected:
      physical_risk_analysis:
        acute_risks:
          - hazard: "flood"
            exposure_level: "LOW"

  - test_id: CR-PHYS-003
    name: "Wildfire risk California facility"
    input:
      asset_inventory:
        - asset_id: "FAC-003"
          location:
            latitude: 34.0522
            longitude: -118.2437  # Los Angeles
          asset_value_usd: 100000000
      analysis_parameters:
        scenarios: ["rcp_8_5"]
    expected:
      physical_risk_analysis:
        acute_risks:
          - hazard: "wildfire"
            exposure_level: "HIGH"

  - test_id: CR-PHYS-004
    name: "Heat stress manufacturing facility"
    input:
      asset_inventory:
        - asset_id: "FAC-004"
          asset_type: "facility"
          location:
            latitude: 25.2048
            longitude: 55.2708  # Dubai
          climate_sensitivity:
            heat_sensitive: true
    expected:
      physical_risk_analysis:
        chronic_risks:
          - hazard: "extreme_heat"
            exposure_level: "VERY_HIGH"
            projected_change:
              metric: "cooling_degree_days"

  - test_id: CR-PHYS-005
    name: "Water stress agricultural operation"
    input:
      asset_inventory:
        - asset_id: "FARM-001"
          asset_type: "facility"
          location:
            latitude: 36.7783
            longitude: -119.4179  # California Central Valley
          climate_sensitivity:
            water_dependent: true
    expected:
      physical_risk_analysis:
        chronic_risks:
          - hazard: "drought"
            exposure_level: "HIGH"
          - hazard: "water_stress"
            exposure_level: "HIGH"

  - test_id: CR-PHYS-006
    name: "Sea level rise coastal infrastructure"
    input:
      asset_inventory:
        - asset_id: "PORT-001"
          asset_type: "infrastructure"
          location:
            latitude: 51.9244
            longitude: 4.4777  # Rotterdam
          asset_value_usd: 500000000
          climate_sensitivity:
            coastal: true
      analysis_parameters:
        time_horizons: ["2050", "2100"]
    expected:
      physical_risk_analysis:
        chronic_risks:
          - hazard: "sea_level_rise"
            projected_change:
              metric: "mean_sea_level_change_m"
              projected_2050:
                range: [0.2, 0.4]
              projected_2100:
                range: [0.5, 1.0]

  - test_id: CR-PHYS-007
    name: "Tropical cyclone exposure Southeast Asia"
    input:
      asset_inventory:
        - asset_id: "FAC-005"
          location:
            latitude: 14.5995
            longitude: 120.9842  # Manila
          asset_value_usd: 80000000
    expected:
      physical_risk_analysis:
        acute_risks:
          - hazard: "tropical_cyclone"
            exposure_level: "VERY_HIGH"

  - test_id: CR-PHYS-008
    name: "Multiple hazard aggregation"
    input:
      asset_inventory:
        - asset_id: "FAC-006"
          location:
            latitude: 35.6762
            longitude: 139.6503  # Tokyo
          asset_value_usd: 200000000
    expected:
      physical_risk_analysis:
        acute_risks:
          # Tokyo exposed to multiple hazards
          - hazard: "tropical_cyclone"
          - hazard: "flood"
          - hazard: "extreme_heat"
        overall_physical_risk: "HIGH"

  - test_id: CR-PHYS-009
    name: "Geographic hotspot identification"
    input:
      asset_inventory:
        - asset_id: "FAC-A"
          location:
            country: "US"
            region: "Florida"
        - asset_id: "FAC-B"
          location:
            country: "US"
            region: "Colorado"
        - asset_id: "FAC-C"
          location:
            country: "NL"
            region: "Zuid-Holland"
    expected:
      geographic_hotspots:
        - location: "Florida"
          risk_score:
            greater_than: 70
        - location: "Zuid-Holland"
          risk_score:
            greater_than: 50

  - test_id: CR-PHYS-010
    name: "RCP scenario comparison"
    input:
      analysis_parameters:
        scenarios: ["rcp_2_6", "rcp_4_5", "rcp_8_5"]
        time_horizons: ["2050"]
    expected:
      scenario_comparison:
        rcp_2_6:
          physical_risk_rating: "LOW"
        rcp_4_5:
          physical_risk_rating: "MEDIUM"
        rcp_8_5:
          physical_risk_rating: "HIGH"

  - test_id: CR-PHYS-011
    name: "Supply chain physical risk"
    input:
      operational_data:
        supply_chain:
          - supplier_name: "Supplier A"
            location:
              country: "TH"  # Thailand - flood prone
            spend_usd: 10000000
            criticality: "critical"
    expected:
      physical_risk_analysis:
        supply_chain_risk:
          flood_exposed_suppliers: 1
          supply_chain_value_at_risk:
            greater_than: 0

  - test_id: CR-PHYS-012
    name: "Precipitation change impact"
    input:
      asset_inventory:
        - asset_id: "HYDRO-001"
          asset_type: "infrastructure"
          location:
            country: "NO"  # Norway
          notes: "Hydroelectric facility"
    expected:
      physical_risk_analysis:
        chronic_risks:
          - hazard: "precipitation_changes"
            projected_change:
              metric: "annual_precip_change_percent"

  - test_id: CR-PHYS-013
    name: "Permafrost thaw risk"
    input:
      asset_inventory:
        - asset_id: "PIPE-001"
          asset_type: "infrastructure"
          location:
            latitude: 68.9585
            longitude: 33.0827  # Murmansk
    expected:
      physical_risk_analysis:
        chronic_risks:
          - hazard: "permafrost_thaw"
            exposure_level: "HIGH"

  - test_id: CR-PHYS-014
    name: "Low physical risk location"
    input:
      asset_inventory:
        - asset_id: "FAC-LOW"
          location:
            latitude: 51.5074
            longitude: -0.1278  # London
          climate_sensitivity:
            coastal: false
            water_dependent: false
    expected:
      physical_risk_analysis:
        overall_physical_risk: "LOW"

  - test_id: CR-PHYS-015
    name: "Time horizon comparison"
    input:
      analysis_parameters:
        time_horizons: ["2030", "2050", "2100"]
        scenarios: ["rcp_4_5"]
    expected:
      physical_risk_analysis:
        time_horizon_comparison:
          2030:
            risk_rating: "MEDIUM"
          2050:
            risk_rating: "HIGH"
          2100:
            risk_rating: "VERY_HIGH"
```

### 6.2 Transition Risk Tests (12 tests)

```yaml
golden_tests_transition_risk:
  - test_id: CR-TRANS-001
    name: "High carbon intensity company - carbon pricing impact"
    input:
      company_profile:
        sector: "materials"
        revenue_usd: 5000000000
        ebitda_usd: 500000000
      carbon_exposure:
        scope_1_tco2e: 2000000
        scope_2_tco2e: 500000
        carbon_intensity: 500  # tCO2e/$M
      analysis_parameters:
        scenarios: ["rcp_2_6"]
        time_horizons: ["2030"]
    expected:
      transition_risk_analysis:
        carbon_pricing_impact:
          projected_carbon_cost:
            rcp_2_6_2030:
              range: [200000000, 400000000]
          ebitda_impact_percent:
            greater_than: 30

  - test_id: CR-TRANS-002
    name: "Low carbon company - minimal transition risk"
    input:
      company_profile:
        sector: "information_technology"
        revenue_usd: 2000000000
      carbon_exposure:
        scope_1_tco2e: 10000
        scope_2_tco2e: 50000
        carbon_intensity: 30
    expected:
      transition_risk_analysis:
        overall_transition_risk: "LOW"
        carbon_pricing_impact:
          ebitda_impact_percent:
            less_than: 5

  - test_id: CR-TRANS-003
    name: "Oil & gas stranded asset risk"
    input:
      company_profile:
        sector: "energy"
        industry: "oil_gas_exploration"
      asset_inventory:
        - asset_type: "fossil_fuel_reserves"
          asset_value_usd: 10000000000
    expected:
      transition_risk_analysis:
        technology_risks:
          - risk_name: "Stranded assets"
            stranded_asset_risk_usd:
              range: [2000000000, 5000000000]

  - test_id: CR-TRANS-004
    name: "Automotive sector technology disruption"
    input:
      company_profile:
        sector: "consumer_discretionary"
        industry: "automobiles"
        revenue_usd: 50000000000
      operational_data:
        ice_vehicle_revenue_percent: 80
    expected:
      transition_risk_analysis:
        technology_risks:
          - risk_name: "EV transition"
            likelihood: "HIGH"

  - test_id: CR-TRANS-005
    name: "Policy risk - emissions regulations"
    input:
      company_profile:
        sector: "utilities"
        geographic_operations: ["US", "EU"]
      carbon_exposure:
        scope_1_tco2e: 5000000
    expected:
      transition_risk_analysis:
        policy_legal_risks:
          - risk_name: "Emissions regulations"
            likelihood: "HIGH"

  - test_id: CR-TRANS-006
    name: "Market risk - consumer preference shift"
    input:
      company_profile:
        sector: "consumer_staples"
        industry: "food_products"
      operational_data:
        high_carbon_products_percent: 40
    expected:
      transition_risk_analysis:
        market_risks:
          - risk_name: "Consumer preference shift"
            description: "Shift toward low-carbon products"

  - test_id: CR-TRANS-007
    name: "Reputation risk - high emitter"
    input:
      company_profile:
        sector: "materials"
      carbon_exposure:
        carbon_intensity: 800  # Well above sector average
    expected:
      transition_risk_analysis:
        reputation_risks:
          - risk_name: "High carbon intensity"
            likelihood: "HIGH"

  - test_id: CR-TRANS-008
    name: "Scenario comparison - transition risks"
    input:
      analysis_parameters:
        scenarios: ["rcp_2_6", "rcp_4_5", "rcp_8_5"]
    expected:
      transition_risk_analysis:
        scenario_comparison:
          rcp_2_6:
            transition_risk_rating: "HIGH"
          rcp_4_5:
            transition_risk_rating: "MEDIUM"
          rcp_8_5:
            transition_risk_rating: "LOW"

  - test_id: CR-TRANS-009
    name: "Financial services - financed emissions"
    input:
      company_profile:
        sector: "financials"
        industry: "banks"
      operational_data:
        financed_emissions_tco2e: 50000000
        fossil_fuel_lending_usd: 10000000000
    expected:
      transition_risk_analysis:
        policy_legal_risks:
          - risk_name: "Climate-related financial regulation"
        market_risks:
          - risk_name: "Financed emissions exposure"

  - test_id: CR-TRANS-010
    name: "Real estate sector - building efficiency"
    input:
      company_profile:
        sector: "real_estate"
      asset_inventory:
        - asset_type: "real_estate"
          building_energy_rating: "D"
    expected:
      transition_risk_analysis:
        policy_legal_risks:
          - risk_name: "Building efficiency regulations"
        technology_risks:
          - risk_name: "Retrofit costs"

  - test_id: CR-TRANS-011
    name: "Carbon border adjustment impact"
    input:
      company_profile:
        sector: "materials"
        geographic_operations: ["CN"]
      operational_data:
        eu_exports_usd: 500000000
    expected:
      transition_risk_analysis:
        policy_legal_risks:
          - risk_name: "CBAM"
            financial_impact_usd:
              greater_than: 0

  - test_id: CR-TRANS-012
    name: "Litigation risk"
    input:
      company_profile:
        sector: "energy"
        carbon_exposure:
          scope_1_tco2e: 10000000
      historical_data:
        climate_litigation_target: true
    expected:
      transition_risk_analysis:
        policy_legal_risks:
          - risk_name: "Climate litigation"
            likelihood: "HIGH"
```

### 6.3 Scenario Analysis Tests (13 tests)

```yaml
golden_tests_scenario_analysis:
  - test_id: CR-SCEN-001
    name: "RCP 2.6 - orderly transition"
    input:
      analysis_parameters:
        scenarios: ["rcp_2_6"]
        time_horizons: ["2030", "2050"]
    expected:
      scenario_analysis_results:
        scenario_impacts:
          - scenario: "rcp_2_6"
            physical_risk_impact_usd:
              note: "Lower physical risks"
            transition_risk_impact_usd:
              note: "Higher transition costs"
            opportunity_value_usd:
              note: "Significant opportunities"

  - test_id: CR-SCEN-002
    name: "RCP 8.5 - hot house world"
    input:
      analysis_parameters:
        scenarios: ["rcp_8_5"]
        time_horizons: ["2050"]
    expected:
      scenario_analysis_results:
        scenario_impacts:
          - scenario: "rcp_8_5"
            physical_risk_impact_usd:
              note: "Severe physical risks"
            transition_risk_impact_usd:
              note: "Lower transition costs"

  - test_id: CR-SCEN-003
    name: "Net impact calculation"
    input:
      financial_data:
        revenue_usd: 10000000000
        ebitda_usd: 1000000000
    expected:
      scenario_analysis_results:
        scenario_impacts:
          - net_impact_usd:
              note: "Physical + Transition - Opportunities"
            revenue_impact_percent:
              note: "Calculated correctly"
            ebitda_impact_percent:
              note: "Calculated correctly"

  - test_id: CR-SCEN-004
    name: "SSP pathway analysis"
    input:
      analysis_parameters:
        scenarios: ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
    expected:
      scenario_analysis_results:
        scenarios_analyzed: 3

  - test_id: CR-SCEN-005
    name: "Time horizon sensitivity"
    input:
      analysis_parameters:
        scenarios: ["rcp_4_5"]
        time_horizons: ["2030", "2040", "2050"]
    expected:
      scenario_analysis_results:
        time_horizon_comparison:
          2050:
            net_impact_usd:
              greater_than_2030: true

  - test_id: CR-SCEN-006
    name: "Sector-specific scenario"
    input:
      company_profile:
        sector: "utilities"
      analysis_parameters:
        scenarios: ["rcp_2_6"]
    expected:
      scenario_analysis_results:
        sector_specific_insights:
          - "Power sector decarbonization pathway"

  - test_id: CR-SCEN-007
    name: "Opportunity identification in orderly scenario"
    input:
      company_profile:
        sector: "industrials"
      analysis_parameters:
        scenarios: ["rcp_2_6"]
    expected:
      climate_opportunities:
        - category: "resource_efficiency"
        - category: "products_services"

  - test_id: CR-SCEN-008
    name: "Disorderly transition scenario"
    input:
      analysis_parameters:
        scenarios: ["disorderly_transition"]
        description: "Delayed action then rapid change"
    expected:
      scenario_analysis_results:
        scenario_impacts:
          - scenario: "disorderly_transition"
            physical_risk_impact_usd:
              note: "Higher than orderly"
            transition_risk_impact_usd:
              note: "Higher than orderly"

  - test_id: CR-SCEN-009
    name: "1.5C scenario stress test"
    input:
      analysis_parameters:
        scenarios: ["ssp1_1_9"]
    expected:
      scenario_analysis_results:
        scenario_impacts:
          - scenario: "ssp1_1_9"
            carbon_price_assumption: "Very high ($200+/tCO2)"

  - test_id: CR-SCEN-010
    name: "Multi-scenario comparison"
    input:
      analysis_parameters:
        scenarios: ["rcp_2_6", "rcp_4_5", "rcp_8_5"]
    expected:
      scenario_analysis_results:
        comparison_summary:
          best_case_scenario: "rcp_2_6"
          worst_case_scenario: "Depends on company profile"

  - test_id: CR-SCEN-011
    name: "Regional scenario variations"
    input:
      company_profile:
        geographic_operations: ["US", "EU", "CN"]
      analysis_parameters:
        regional_breakdown: true
    expected:
      scenario_analysis_results:
        regional_impacts:
          EU:
            note: "Higher transition risk (stringent policy)"
          CN:
            note: "Higher physical risk"

  - test_id: CR-SCEN-012
    name: "Scenario stress on capital requirements"
    input:
      financial_data:
        capex_usd: 500000000
    expected:
      scenario_analysis_results:
        capital_requirements:
          adaptation_capex_usd:
            note: "Physical risk mitigation investment"
          transition_capex_usd:
            note: "Decarbonization investment"

  - test_id: CR-SCEN-013
    name: "Scenario narrative generation"
    input:
      analysis_parameters:
        scenarios: ["rcp_2_6"]
        generate_narrative: true
    expected:
      scenario_analysis_results:
        narratives:
          rcp_2_6:
            note: "Clear storyline describing scenario assumptions and impacts"
```

### 6.4 Financial Impact Tests (10 tests)

```yaml
golden_tests_financial_impact:
  - test_id: CR-FIN-001
    name: "Climate VaR calculation"
    input:
      risk_exposures:
        - exposure_value: 100000000
          annual_probability: 0.02
          vulnerability: 0.5
      analysis_parameters:
        confidence_level: 0.95
    expected:
      financial_impact_summary:
        value_at_risk_95:
          note: "Calculated using Monte Carlo"

  - test_id: CR-FIN-002
    name: "EBITDA at risk"
    input:
      financial_data:
        ebitda_usd: 500000000
      scenario_impacts:
        total_risk_usd: 100000000
    expected:
      financial_impact_summary:
        ebitda_at_risk_percent: 20

  - test_id: CR-FIN-003
    name: "Asset impairment risk"
    input:
      asset_inventory:
        - asset_value_usd: 1000000000
          physical_risk_exposure: "HIGH"
    expected:
      financial_impact_summary:
        potential_impairment_usd:
          greater_than: 0

  - test_id: CR-FIN-004
    name: "Insurance cost projections"
    input:
      operational_data:
        current_insurance_premium: 10000000
      physical_risk_analysis:
        overall_physical_risk: "HIGH"
    expected:
      financial_impact_summary:
        projected_insurance_increase:
          note: "Premium increase due to elevated risk"

  - test_id: CR-FIN-005
    name: "Capital expenditure needs"
    input:
      resilience_assessment:
        adaptation_gaps: ["flood_protection", "cooling_systems"]
    expected:
      financial_impact_summary:
        capital_expenditure_needs:
          greater_than: 0

  - test_id: CR-FIN-006
    name: "Revenue at risk from demand shift"
    input:
      company_profile:
        sector: "energy"
      operational_data:
        fossil_fuel_revenue_percent: 60
      financial_data:
        revenue_usd: 10000000000
    expected:
      financial_impact_summary:
        revenue_at_risk_usd:
          note: "Revenue from declining segments"

  - test_id: CR-FIN-007
    name: "Cost of capital impact"
    input:
      company_profile:
        current_wacc: 0.08
      climate_risk_assessment:
        overall_risk_rating: "HIGH"
    expected:
      financial_impact_summary:
        potential_wacc_increase:
          range: [0.005, 0.02]

  - test_id: CR-FIN-008
    name: "Working capital impact"
    input:
      physical_risk_analysis:
        supply_chain_risk: "HIGH"
    expected:
      financial_impact_summary:
        working_capital_impact:
          note: "Increased inventory requirements"

  - test_id: CR-FIN-009
    name: "Debt covenant risk"
    input:
      financial_data:
        debt_usd: 2000000000
        ebitda_usd: 500000000
      scenario_impacts:
        ebitda_impact_percent: -30
    expected:
      financial_impact_summary:
        covenant_risk:
          note: "Leverage ratio may breach covenants"

  - test_id: CR-FIN-010
    name: "Credit rating impact"
    input:
      climate_risk_assessment:
        overall_risk_rating: "VERY_HIGH"
        transition_risk_rating: "HIGH"
    expected:
      financial_impact_summary:
        potential_credit_downgrade:
          note: "1-2 notch downgrade risk"
```

---

## 7. Data Dependencies

```yaml
data_dependencies:
  climate_data:
    cmip6:
      name: "CMIP6 Climate Projections"
      source: "IPCC"
      access: "ESGF portal"

    wri_aqueduct:
      name: "WRI Aqueduct"
      source: "World Resources Institute"
      access: "API/Download"

  hazard_data:
    think_hazard:
      name: "ThinkHazard"
      source: "World Bank"
      hazards: ["Flood", "Cyclone", "Earthquake", "Wildfire"]

    noaa:
      name: "NOAA Climate Data"
      source: "NOAA"
      access: "API"

  regulatory_data:
    carbon_prices:
      name: "Carbon Price Projections"
      sources: ["IEA", "NGFS", "World Bank"]

    policy_tracker:
      name: "Climate Policy Tracker"
      source: "LSE Grantham Research Institute"

  financial_data:
    sector_benchmarks:
      name: "Sector Carbon Intensity Benchmarks"
      source: "CDP, SASB"

    credit_ratings:
      name: "ESG Rating Integration"
      sources: ["Moody's", "S&P", "Fitch"]
```

---

## 8. Implementation Roadmap

```yaml
implementation_roadmap:
  phase_1_physical:
    duration: "Weeks 1-6"
    deliverables:
      - "Physical risk analyzer"
      - "Hazard exposure calculator"
      - "Asset-level risk scoring"

  phase_2_transition:
    duration: "Weeks 7-12"
    deliverables:
      - "Transition risk analyzer"
      - "Carbon pricing modeler"
      - "Stranded asset calculator"

  phase_3_scenarios:
    duration: "Weeks 13-18"
    deliverables:
      - "Scenario modeler"
      - "Financial impact calculator"
      - "Climate VaR"

  phase_4_reporting:
    duration: "Weeks 19-24"
    deliverables:
      - "TCFD report generator"
      - "ISSB disclosure package"
      - "Dashboard and visualization"
```

---

## 9. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-ProductManager | Initial specification |

**Approvals:**

- Climate Science Lead: ___________________ Date: _______
- Risk Management Lead: ___________________ Date: _______
- Engineering Lead: ___________________ Date: _______
- Product Manager: ___________________ Date: _______

---

**END OF SPECIFICATION**
