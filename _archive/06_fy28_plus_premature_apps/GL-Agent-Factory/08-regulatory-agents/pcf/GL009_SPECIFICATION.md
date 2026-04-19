# GL-009 Product Carbon Footprint Agent Specification

**Agent ID:** gl-009-pcf-v1
**Version:** 1.0.0
**Date:** 2025-12-04
**Priority:** P3-MEDIUM
**Deadline:** February 2027 (Battery Regulation), Rolling (ESPR)
**Status:** SPECIFICATION COMPLETE

---

## 1. Executive Summary

### 1.1 Regulation Overview

**EU Ecodesign for Sustainable Products Regulation (ESPR) & Battery Regulation**

The EU is implementing comprehensive product carbon footprint (PCF) and digital product passport (DPP) requirements starting with batteries and expanding to other product categories. These regulations require manufacturers to calculate, verify, and disclose lifecycle carbon footprints.

**Key Regulations:**
- **Battery Regulation (EU) 2023/1542**: Mandatory battery carbon footprint by February 2027
- **ESPR (EU) 2024/xxx**: Digital Product Passports for multiple product categories
- **ISO 14067:2018**: Carbon footprint of products standard
- **PACT Pathfinder Framework**: Industry standard for PCF data exchange

**Key Requirements:**
- Cradle-to-gate and cradle-to-grave carbon footprint calculations
- 16 environmental impact categories (PEF methodology)
- Digital Product Passport with QR code access
- Third-party verification mandatory for batteries
- Catena-X compliant data exchange format

### 1.2 Agent Purpose

The Product Carbon Footprint Agent automates the calculation, verification, and exchange of product lifecycle carbon footprints. It provides:

1. **ISO 14067 Compliant PCF** - Cradle-to-gate and full lifecycle calculations
2. **PACT Pathfinder Integration** - Industry-standard data exchange
3. **16 Impact Categories** - Full PEF environmental assessment
4. **Battery Passport Support** - EU Battery Regulation compliance
5. **Catena-X Data Exchange** - Supply chain carbon data sharing

---

## 2. Regulatory Specification

### 2.1 Applicability Criteria

```yaml
applicability:
  battery_regulation:
    scope: "All batteries placed on EU market"
    categories:
      - "Electric vehicle batteries"
      - "Industrial batteries >2kWh"
      - "Light means of transport batteries"
    effective_dates:
      pcf_declaration: "2025-02-18"
      pcf_performance_class: "2026-08-18"
      pcf_maximum_threshold: "2027-08-18"

  espr_product_categories:
    priority_products:
      - "Textiles and footwear"
      - "Furniture"
      - "Tyres"
      - "Detergents"
      - "Paints"
      - "Lubricants"
      - "Iron, steel, and aluminium"
      - "Chemicals"
    implementation: "Rolling 2025-2030"

  pact_pathfinder:
    scope: "Voluntary industry standard"
    participants: "500+ companies globally"
    sectors:
      - "Automotive"
      - "Chemicals"
      - "Consumer goods"
      - "Electronics"
```

### 2.2 Timeline and Deadlines

```yaml
timeline:
  battery_regulation:
    2025-02-18: "Carbon footprint declaration required"
    2026-08-18: "Carbon footprint performance class labels"
    2027-02-18: "Maximum carbon footprint thresholds in force"
    2031-02-18: "Recycled content requirements enforced"

  espr_digital_product_passport:
    2026-H2: "First delegated acts expected"
    2027-2028: "DPP for priority products"
    2028-2030: "Expansion to all categories"

  pact_pathfinder:
    version_2_0: "2023-01"
    version_2_1: "2024-01"
    version_3_0: "2025 (expected)"
```

### 2.3 Penalty Structure

```yaml
penalties:
  battery_regulation:
    non_compliance:
      - "Market access denial in EU"
      - "Product recall orders"
      - "Administrative fines per member state"
    maximum_fine: "Varies by member state (up to EUR 10M in some)"

  espr:
    non_compliance:
      - "Prohibition of placing on market"
      - "Withdrawal from market"
      - "Administrative penalties"
```

---

## 3. Calculation Methodology

### 3.1 ISO 14067:2018 Framework

```yaml
iso_14067:
  standard: "ISO 14067:2018"
  title: "Carbon footprint of products"
  scope: "Quantification of GHG emissions and removals"

  system_boundaries:
    cradle_to_gate:
      stages:
        - "Raw material acquisition"
        - "Pre-processing"
        - "Production/manufacturing"
      note: "Most common for B2B PCF"

    cradle_to_grave:
      stages:
        - "Raw material acquisition"
        - "Pre-processing"
        - "Production/manufacturing"
        - "Distribution"
        - "Use phase"
        - "End of life"
      note: "Required for B2C claims"

    gate_to_gate:
      stages:
        - "Production/manufacturing only"
      note: "Supplier-specific contributions"

  functional_unit:
    definition: "Quantified performance of product system"
    examples:
      battery: "1 kWh of battery capacity"
      textile: "1 kg of finished fabric"
      chemical: "1 kg of chemical product"

  allocation_methods:
    hierarchy:
      1: "Avoid allocation (subdivision)"
      2: "Physical relationship (mass, energy)"
      3: "Economic allocation"
    circular_economy:
      method: "Circular Footprint Formula (CFF)"
      application: "Recycled content and recyclability"

  ghg_gases:
    mandatory:
      - { gas: "CO2", gwp_ar6: 1 }
      - { gas: "CH4", gwp_ar6: 27.9 }
      - { gas: "N2O", gwp_ar6: 273 }
    conditional:
      - { gas: "HFCs", gwp_ar6: "varies" }
      - { gas: "PFCs", gwp_ar6: "varies" }
      - { gas: "SF6", gwp_ar6: 25200 }
      - { gas: "NF3", gwp_ar6: 17400 }
```

### 3.2 PACT Pathfinder Framework

```yaml
pact_pathfinder:
  name: "Partnership for Carbon Transparency"
  version: "2.1"
  organization: "WBCSD"

  data_model:
    product_footprint:
      id: "UUID"
      spec_version: "2.1.0"
      version: "1"
      created: "ISO8601 timestamp"
      status: "Active|Deprecated"
      validity_period:
        start: "date"
        end: "date"

    pcf:
      declared_unit: "string"
      unitary_product_amount: "number"
      product_mass_per_declared_unit: "kg"
      exempted_emissions_percent: "0-5%"
      exempted_emissions_description: "string"
      packaging_emissions_included: "boolean"
      geographical_scope:
        global: "boolean"
        country_codes: ["ISO3166"]
      boundary_processes_description: "string"
      reference_period:
        start: "date"
        end: "date"
      characterization_factors: "AR6|AR5"
      cross_sectoral_standards_used:
        - "GHG Protocol Product Standard"
        - "ISO 14067:2018"
      product_or_sector_specific_rules:
        - operator: "string"
        - rule_names: ["string"]
      biogenic_accounting_methodology: "PEF|GHGP|ISO"

    emission_breakdown:
      pcf_excluding_biogenic: "number (kgCO2e)"
      pcf_including_biogenic: "number (kgCO2e)"
      fossil_ghg_emissions: "number"
      fossil_carbon_content: "number"
      biogenic_carbon_content: "number"
      dluc_ghg_emissions: "number"
      land_management_ghg_emissions: "number"
      other_biogenic_ghg_emissions: "number"
      iluc_ghg_emissions: "number"
      biogenic_carbon_withdrawal: "number"
      aircraft_ghg_emissions: "number"
      packaging_ghg_emissions: "number"
      allocation_rules_description: "string"
      uncertainty_assessment_description: "string"
      primary_data_share: "0-100%"
      dqi: "Data Quality Indicators object"
      assurance:
        coverage: "corporate|product_line|PCF"
        level: "limited|reasonable"
        provider_name: "string"
        completed: "date"
        standard: "ISO14064-3|ISAE3000"
```

### 3.3 PEF 16 Impact Categories

```yaml
pef_impact_categories:
  source: "EU Product Environmental Footprint Guide"
  version: "3.0"

  categories:
    1_climate_change:
      name: "Climate change, total"
      unit: "kg CO2 eq"
      method: "IPCC AR6 GWP100"
      weight: 0.2106
      sub_categories:
        - "Climate change - fossil"
        - "Climate change - biogenic"
        - "Climate change - land use and land use change"

    2_ozone_depletion:
      name: "Ozone depletion"
      unit: "kg CFC-11 eq"
      method: "WMO"
      weight: 0.0631

    3_human_toxicity_cancer:
      name: "Human toxicity, cancer"
      unit: "CTUh"
      method: "USEtox 2.1"
      weight: 0.0213

    4_human_toxicity_non_cancer:
      name: "Human toxicity, non-cancer"
      unit: "CTUh"
      method: "USEtox 2.1"
      weight: 0.0184

    5_particulate_matter:
      name: "Particulate matter"
      unit: "disease incidence"
      method: "UNEP"
      weight: 0.0896

    6_ionising_radiation:
      name: "Ionising radiation, human health"
      unit: "kBq U235 eq"
      method: "Frischknecht et al."
      weight: 0.0501

    7_photochemical_ozone:
      name: "Photochemical ozone formation, human health"
      unit: "kg NMVOC eq"
      method: "LOTOS-EUROS"
      weight: 0.0478

    8_acidification:
      name: "Acidification"
      unit: "mol H+ eq"
      method: "Accumulated Exceedance"
      weight: 0.0620

    9_eutrophication_terrestrial:
      name: "Eutrophication, terrestrial"
      unit: "mol N eq"
      method: "Accumulated Exceedance"
      weight: 0.0371

    10_eutrophication_freshwater:
      name: "Eutrophication, freshwater"
      unit: "kg P eq"
      method: "EUTREND"
      weight: 0.0280

    11_eutrophication_marine:
      name: "Eutrophication, marine"
      unit: "kg N eq"
      method: "EUTREND"
      weight: 0.0296

    12_ecotoxicity_freshwater:
      name: "Ecotoxicity, freshwater"
      unit: "CTUe"
      method: "USEtox 2.1"
      weight: 0.0192

    13_land_use:
      name: "Land use"
      unit: "Pt"
      method: "LANCA"
      weight: 0.0794

    14_water_use:
      name: "Water use"
      unit: "m3 world eq"
      method: "AWARE"
      weight: 0.0851

    15_resource_use_minerals:
      name: "Resource use, minerals and metals"
      unit: "kg Sb eq"
      method: "CML"
      weight: 0.0755

    16_resource_use_fossils:
      name: "Resource use, fossils"
      unit: "MJ"
      method: "CML"
      weight: 0.0832
```

### 3.4 Battery Carbon Footprint Requirements

```yaml
battery_pcf_requirements:
  regulation: "EU Battery Regulation 2023/1542"

  lifecycle_stages:
    1_raw_material_acquisition:
      scope: "Mining and processing of battery materials"
      materials:
        - "Lithium"
        - "Cobalt"
        - "Nickel"
        - "Manganese"
        - "Graphite"
        - "Copper"
        - "Aluminium"
      allocation: "Mass-based for co-products"

    2_manufacturing:
      scope: "Cell production to pack assembly"
      processes:
        - "Electrode manufacturing"
        - "Cell assembly"
        - "Module assembly"
        - "Pack integration"
        - "Battery management system"
      electricity_factor: "Grid or renewable PPA"

    3_distribution:
      scope: "Transport from factory to customer"
      included: true
      methods: ["Road", "Rail", "Sea", "Air"]

    4_recycling:
      scope: "End-of-life recycling benefit"
      method: "Circular Footprint Formula"
      parameters:
        A: 0.5  # Allocation factor
        R1: "Recycled input rate"
        R2: "Recycling output rate"
        Qs_Qp: "Quality ratio"

  functional_unit: "1 kWh total battery energy capacity"

  performance_classes:
    class_A: "Lowest carbon footprint"
    class_B: "Below average"
    class_C: "Average"
    class_D: "Above average"
    class_E: "Highest carbon footprint"

  maximum_threshold:
    effective: "2027-08-18"
    determination: "Based on market data collected 2024-2026"
```

---

## 4. Agent Architecture

### 4.1 Agent Specification (AgentSpec v2)

```yaml
agent_id: gl-009-pcf-v1
name: "Product Carbon Footprint Agent"
version: "1.0.0"
type: product-lifecycle
priority: P3-MEDIUM
deadline: "2027-02-01"

description: |
  Automated product carbon footprint calculation, verification, and data
  exchange agent. Supports ISO 14067 methodology, PACT Pathfinder framework,
  EU Battery Regulation, and Digital Product Passport requirements.

regulatory_context:
  regulations:
    - name: "EU Battery Regulation"
      reference: "(EU) 2023/1542"
      deadline: "2027-02-18"
    - name: "ESPR"
      reference: "(EU) 2024/xxx"
      deadline: "Rolling 2026-2030"
  standards:
    - "ISO 14067:2018"
    - "ISO 14044:2006"
    - "GHG Protocol Product Standard"
    - "PACT Pathfinder 2.1"

inputs:
  product_definition:
    type: object
    required: true
    properties:
      product_id:
        type: string
        description: "Unique product identifier"

      product_name:
        type: string
        description: "Product name"

      product_category:
        type: string
        enum: ["battery", "textile", "electronics", "chemical", "metal", "plastic", "other"]

      functional_unit:
        type: object
        properties:
          value:
            type: number
            minimum: 0
          unit:
            type: string
            example: "kWh, kg, piece, m2"
          description:
            type: string
            example: "1 kWh of battery capacity over 1000 cycles"

      reference_flow:
        type: object
        properties:
          value:
            type: number
          unit:
            type: string

      product_mass_kg:
        type: number
        minimum: 0

      product_lifetime_years:
        type: number
        minimum: 0

  bill_of_materials:
    type: array
    required: true
    description: "Complete BOM with quantities and origins"
    items:
      type: object
      properties:
        material_id:
          type: string
        material_name:
          type: string
        material_category:
          type: string
          enum: ["metal", "plastic", "chemical", "electronic", "composite", "other"]
        quantity:
          type: number
          minimum: 0
        unit:
          type: string
          enum: ["kg", "g", "pieces", "m", "m2", "m3", "L"]
        supplier_id:
          type: string
        origin_country:
          type: string
          pattern: "^[A-Z]{2}$"
        recycled_content_percent:
          type: number
          minimum: 0
          maximum: 100
        supplier_pcf:
          type: object
          description: "Supplier-specific PCF if available"
          properties:
            value_kg_co2e:
              type: number
            methodology:
              type: string
            verification_status:
              type: string

  manufacturing_data:
    type: object
    required: true
    properties:
      facility_id:
        type: string

      facility_location:
        type: object
        properties:
          country:
            type: string
          region:
            type: string
          grid_region:
            type: string

      electricity_consumption:
        type: object
        properties:
          total_kwh_per_unit:
            type: number
          source:
            type: string
            enum: ["grid", "renewable_ppa", "onsite_solar", "mixed"]
          renewable_percentage:
            type: number
            minimum: 0
            maximum: 100
          grid_emission_factor:
            type: number
            description: "kg CO2e/kWh if not auto-retrieved"

      natural_gas_consumption:
        type: object
        properties:
          total_m3_per_unit:
            type: number

      process_emissions:
        type: object
        description: "Direct process emissions (non-combustion)"
        properties:
          co2_kg_per_unit:
            type: number
          ch4_kg_per_unit:
            type: number
          n2o_kg_per_unit:
            type: number
          other_ghg:
            type: array
            items:
              type: object
              properties:
                gas:
                  type: string
                quantity_kg:
                  type: number

      waste_generated:
        type: array
        items:
          type: object
          properties:
            waste_type:
              type: string
            quantity_kg_per_unit:
              type: number
            treatment:
              type: string
              enum: ["landfill", "incineration", "recycling", "composting"]

  logistics_data:
    type: object
    properties:
      inbound_logistics:
        type: array
        description: "Supplier to manufacturing"
        items:
          type: object
          properties:
            from_country:
              type: string
            to_facility:
              type: string
            distance_km:
              type: number
            mode:
              type: string
              enum: ["road", "rail", "sea", "air"]
            weight_kg:
              type: number

      outbound_logistics:
        type: array
        description: "Manufacturing to customer"
        items:
          type: object
          properties:
            from_facility:
              type: string
            to_region:
              type: string
            distance_km:
              type: number
            mode:
              type: string
              enum: ["road", "rail", "sea", "air"]

  use_phase_data:
    type: object
    description: "Only for cradle-to-grave boundary"
    properties:
      energy_consumption_kwh:
        type: number
        description: "Total energy during product lifetime"

      energy_source:
        type: string
        enum: ["electricity", "fuel", "none"]

      consumables:
        type: array
        items:
          type: object
          properties:
            consumable_type:
              type: string
            quantity_per_lifetime:
              type: number
            unit:
              type: string

      maintenance_emissions_kg_co2e:
        type: number

  end_of_life_data:
    type: object
    properties:
      recyclability_rate:
        type: number
        minimum: 0
        maximum: 100

      actual_recycling_rate:
        type: number
        minimum: 0
        maximum: 100

      recycled_material_quality_factor:
        type: number
        minimum: 0
        maximum: 1
        description: "Qs/Qp ratio for CFF"

      collection_rate:
        type: number
        minimum: 0
        maximum: 100

      incineration_rate:
        type: number
        minimum: 0
        maximum: 100

      landfill_rate:
        type: number
        minimum: 0
        maximum: 100

  battery_specific_data:
    type: object
    description: "Required for battery products"
    properties:
      battery_type:
        type: string
        enum: ["LFP", "NMC", "NCA", "LTO", "Solid_State", "Other"]

      chemistry:
        type: object
        properties:
          cathode_material:
            type: string
          anode_material:
            type: string
          electrolyte_type:
            type: string

      capacity_kwh:
        type: number
        minimum: 0

      energy_density_wh_kg:
        type: number

      cycle_life:
        type: integer
        description: "Number of charge cycles"

      cell_manufacturing_location:
        type: string

      cobalt_content_percent:
        type: number
        minimum: 0
        maximum: 100

      lithium_source:
        type: string
        enum: ["brine", "hard_rock", "recycled"]

  calculation_parameters:
    type: object
    properties:
      system_boundary:
        type: string
        enum: ["cradle_to_gate", "cradle_to_grave", "gate_to_gate"]
        default: "cradle_to_gate"

      allocation_method:
        type: string
        enum: ["mass", "economic", "energy", "none"]
        default: "mass"

      gwp_timeframe:
        type: string
        enum: ["GWP100", "GWP20"]
        default: "GWP100"

      characterization_factors:
        type: string
        enum: ["AR6", "AR5"]
        default: "AR6"

      include_biogenic_carbon:
        type: boolean
        default: true

      data_quality_requirement:
        type: string
        enum: ["high", "medium", "low"]
        default: "medium"

outputs:
  product_carbon_footprint:
    type: object
    properties:
      pcf_id:
        type: string
        format: uuid

      product_id:
        type: string

      calculation_date:
        type: string
        format: date-time

      pcf_result:
        type: object
        properties:
          pcf_excluding_biogenic:
            type: number
            description: "kg CO2e per functional unit"

          pcf_including_biogenic:
            type: number
            description: "kg CO2e per functional unit"

          breakdown_by_lifecycle_stage:
            type: object
            properties:
              raw_materials_kg_co2e:
                type: number
              manufacturing_kg_co2e:
                type: number
              distribution_kg_co2e:
                type: number
              use_phase_kg_co2e:
                type: number
              end_of_life_kg_co2e:
                type: number

          breakdown_by_ghg:
            type: object
            properties:
              co2_fossil_kg:
                type: number
              co2_biogenic_kg:
                type: number
              ch4_kg_co2e:
                type: number
              n2o_kg_co2e:
                type: number
              other_ghg_kg_co2e:
                type: number

          primary_data_share:
            type: number
            minimum: 0
            maximum: 100

          data_quality_rating:
            type: string
            enum: ["very_good", "good", "fair", "poor"]

      pef_results:
        type: object
        description: "Full 16 impact category results"
        properties:
          climate_change:
            type: number
            unit: "kg CO2 eq"
          ozone_depletion:
            type: number
            unit: "kg CFC-11 eq"
          # ... all 16 categories

      battery_performance_class:
        type: string
        enum: ["A", "B", "C", "D", "E"]
        description: "Only for batteries"

      methodology:
        type: object
        properties:
          standard:
            type: string
          system_boundary:
            type: string
          functional_unit:
            type: string
          allocation_method:
            type: string
          characterization_factors:
            type: string
          reference_period:
            type: object

      verification:
        type: object
        properties:
          verification_status:
            type: string
            enum: ["unverified", "self_verified", "third_party_limited", "third_party_reasonable"]
          verifier_name:
            type: string
          verification_date:
            type: string
            format: date
          verification_standard:
            type: string

      audit_trail:
        type: object
        properties:
          input_hash:
            type: string
          output_hash:
            type: string
          emission_factor_versions:
            type: object
          calculation_timestamp:
            type: string
            format: date-time

  pact_pathfinder_export:
    type: object
    description: "PACT Pathfinder compliant data format"
    properties:
      spec_version:
        type: string
        example: "2.1.0"
      product_footprint:
        type: object
        # Full PACT schema

  digital_product_passport:
    type: object
    properties:
      passport_id:
        type: string
        format: uuid
      product_id:
        type: string
      qr_code_url:
        type: string
        format: uri
      carbon_footprint:
        type: number
      performance_class:
        type: string
      recycled_content:
        type: number
      recyclability:
        type: number
      supply_chain_transparency:
        type: number
      due_diligence_compliance:
        type: boolean

  catena_x_export:
    type: object
    description: "Catena-X PCF data exchange format"
    properties:
      asset_id:
        type: string
      pcf_exchange_data:
        type: object

tools:
  - name: bom_carbon_calculator
    type: calculator
    description: "Calculate carbon footprint from Bill of Materials"
    inputs: ["bill_of_materials", "emission_factors"]
    outputs: ["materials_pcf"]

  - name: manufacturing_calculator
    type: calculator
    description: "Calculate manufacturing stage emissions"
    inputs: ["manufacturing_data", "grid_factors"]
    outputs: ["manufacturing_pcf"]

  - name: logistics_calculator
    type: calculator
    description: "Calculate logistics/distribution emissions"
    inputs: ["logistics_data", "transport_factors"]
    outputs: ["logistics_pcf"]

  - name: use_phase_calculator
    type: calculator
    description: "Calculate use phase emissions"
    inputs: ["use_phase_data", "energy_factors"]
    outputs: ["use_phase_pcf"]

  - name: end_of_life_calculator
    type: calculator
    description: "Calculate end-of-life emissions with CFF"
    inputs: ["end_of_life_data", "recycling_factors"]
    outputs: ["eol_pcf", "recycling_credit"]

  - name: battery_pcf_calculator
    type: calculator
    description: "Battery-specific PCF with performance class"
    inputs: ["battery_data", "manufacturing_data"]
    outputs: ["battery_pcf", "performance_class"]

  - name: pef_impact_calculator
    type: calculator
    description: "Calculate all 16 PEF impact categories"
    inputs: ["lci_data", "characterization_factors"]
    outputs: ["pef_results"]

  - name: emission_factor_lookup
    type: lookup
    description: "Retrieve emission factors from databases"
    inputs: ["material_type", "process", "region"]
    outputs: ["emission_factor", "source", "uncertainty"]

  - name: pact_exporter
    type: exporter
    description: "Export to PACT Pathfinder format"
    inputs: ["pcf_result"]
    outputs: ["pact_json"]

  - name: dpp_generator
    type: generator
    description: "Generate Digital Product Passport"
    inputs: ["pcf_result", "product_data"]
    outputs: ["dpp", "qr_code"]

  - name: catena_x_exporter
    type: exporter
    description: "Export to Catena-X format"
    inputs: ["pcf_result"]
    outputs: ["catena_x_json"]

  - name: data_quality_assessor
    type: analyzer
    description: "Assess data quality per PACT DQI"
    inputs: ["input_data", "data_sources"]
    outputs: ["dqi_score", "quality_report"]

  - name: provenance_tracker
    type: utility
    description: "Track calculation provenance"
    inputs: ["inputs", "outputs"]
    outputs: ["provenance_hash"]

evaluation:
  golden_tests:
    total_count: 50
    categories:
      battery_pcf: 15
      general_pcf: 15
      pef_calculations: 10
      data_exchange: 10

  accuracy_thresholds:
    pcf_calculation: 0.02  # +/- 2%
    pef_categories: 0.05   # +/- 5%
    data_quality: 0.90     # 90% agreement with expert

  benchmarks:
    latency_p95_seconds: 30
    cost_per_calculation_usd: 0.50

certification:
  required_approvals:
    - climate_science_team
    - supply_chain_team
    - data_quality_team

  compliance_checks:
    - iso_14067_compliance
    - pact_pathfinder_compliance
    - battery_regulation_compliance
    - catena_x_compliance
```

---

## 5. Calculation Formulas

### 5.1 Product Carbon Footprint Calculation

```python
def calculate_pcf(
    bill_of_materials: list,
    manufacturing_data: dict,
    logistics_data: dict,
    use_phase_data: dict,
    end_of_life_data: dict,
    system_boundary: str
) -> PCFResult:
    """
    Calculate Product Carbon Footprint per ISO 14067.

    Formula:
    PCF = Raw Materials + Manufacturing + Distribution + Use Phase + End of Life

    Each stage:
    Stage_Emissions = SUM(Activity_Data_i * Emission_Factor_i)

    Source: ISO 14067:2018, GHG Protocol Product Standard
    """

    # 1. Raw Materials Stage
    raw_materials_emissions = 0.0
    for material in bill_of_materials:
        # Supplier-specific PCF if available
        if material.get("supplier_pcf"):
            material_emissions = material["supplier_pcf"]["value_kg_co2e"] * material["quantity"]
        else:
            # Use database emission factor
            ef = get_emission_factor(
                material_type=material["material_category"],
                material_name=material["material_name"],
                origin_country=material["origin_country"]
            )
            material_emissions = material["quantity"] * ef

        # Adjust for recycled content
        if material.get("recycled_content_percent", 0) > 0:
            virgin_fraction = 1 - (material["recycled_content_percent"] / 100)
            recycled_fraction = material["recycled_content_percent"] / 100

            ef_virgin = get_emission_factor(material["material_name"], "virgin")
            ef_recycled = get_emission_factor(material["material_name"], "recycled")

            material_emissions = material["quantity"] * (
                virgin_fraction * ef_virgin +
                recycled_fraction * ef_recycled
            )

        raw_materials_emissions += material_emissions

    # 2. Manufacturing Stage
    manufacturing_emissions = calculate_manufacturing_emissions(manufacturing_data)

    # 3. Distribution Stage
    distribution_emissions = calculate_logistics_emissions(logistics_data)

    # 4. Use Phase (if cradle-to-grave)
    use_phase_emissions = 0.0
    if system_boundary == "cradle_to_grave" and use_phase_data:
        use_phase_emissions = calculate_use_phase_emissions(use_phase_data)

    # 5. End of Life (if cradle-to-grave)
    eol_emissions = 0.0
    recycling_credit = 0.0
    if system_boundary == "cradle_to_grave" and end_of_life_data:
        eol_emissions, recycling_credit = calculate_eol_emissions(end_of_life_data)

    # Total PCF
    pcf_total = (
        raw_materials_emissions +
        manufacturing_emissions +
        distribution_emissions +
        use_phase_emissions +
        eol_emissions -
        recycling_credit  # Credit for recyclability
    )

    return PCFResult(
        pcf_total_kg_co2e=pcf_total,
        raw_materials_kg_co2e=raw_materials_emissions,
        manufacturing_kg_co2e=manufacturing_emissions,
        distribution_kg_co2e=distribution_emissions,
        use_phase_kg_co2e=use_phase_emissions,
        end_of_life_kg_co2e=eol_emissions,
        recycling_credit_kg_co2e=recycling_credit
    )
```

### 5.2 Circular Footprint Formula (CFF)

```python
def calculate_cff(
    material: dict,
    R1: float,  # Recycled content input rate
    R2: float,  # Recycling output rate
    Qs_Qp: float,  # Quality ratio (recycled/virgin)
    A: float = 0.5,  # Allocation factor (default 0.5)
    B: float = 0.0,  # Allocation for energy recovery
    EF_virgin: float = None,  # Virgin material emission factor
    EF_recycled: float = None,  # Recycled material emission factor
    EF_disposal: float = None  # Disposal emission factor
) -> dict:
    """
    Calculate emissions using Circular Footprint Formula.

    Formula (simplified):
    Material_Footprint = (1 - R1) * EF_virgin +
                         R1 * A * EF_recycled +
                         R1 * (1 - A) * EF_virgin * Qs_Qp +
                         (1 - A) * R2 * (EF_recycled - EF_virgin * Qs_Qp) +
                         (1 - R2) * EF_disposal

    Source: EU PEF Guide, Annex C
    """

    # Get emission factors if not provided
    if EF_virgin is None:
        EF_virgin = get_emission_factor(material["name"], "virgin")
    if EF_recycled is None:
        EF_recycled = get_emission_factor(material["name"], "recycled")
    if EF_disposal is None:
        EF_disposal = get_emission_factor(material["name"], "disposal")

    # CFF Calculation
    # Part 1: Virgin material input (not from recycled sources)
    virgin_input = (1 - R1) * EF_virgin

    # Part 2: Recycled input burden
    recycled_input_burden = R1 * A * EF_recycled

    # Part 3: Recycled input credit (quality adjusted)
    recycled_input_credit = R1 * (1 - A) * EF_virgin * Qs_Qp

    # Part 4: Recycling output credit
    recycling_output_credit = (1 - A) * R2 * (EF_recycled - EF_virgin * Qs_Qp)

    # Part 5: Disposal burden (non-recycled fraction)
    disposal_burden = (1 - R2) * EF_disposal

    # Total material footprint
    material_footprint = (
        virgin_input +
        recycled_input_burden +
        recycled_input_credit +
        recycling_output_credit +
        disposal_burden
    )

    return {
        "material_footprint_kg_co2e": material_footprint,
        "virgin_input_kg_co2e": virgin_input,
        "recycled_input_burden_kg_co2e": recycled_input_burden,
        "recycled_input_credit_kg_co2e": recycled_input_credit,
        "recycling_output_credit_kg_co2e": recycling_output_credit,
        "disposal_burden_kg_co2e": disposal_burden,
        "parameters": {
            "R1": R1,
            "R2": R2,
            "Qs_Qp": Qs_Qp,
            "A": A
        }
    }
```

### 5.3 Battery Carbon Footprint

```python
def calculate_battery_pcf(
    battery_data: dict,
    manufacturing_data: dict,
    logistics_data: dict
) -> BatteryPCFResult:
    """
    Calculate Battery Carbon Footprint per EU Battery Regulation.

    Formula:
    Battery_PCF = Materials + Cell_Manufacturing + Pack_Assembly + Distribution

    Materials include:
    - Cathode active materials (NCM, NCA, LFP)
    - Anode materials (graphite)
    - Electrolyte
    - Separator
    - Current collectors (Cu, Al)
    - Housing and BMS

    Source: EU Battery Regulation (EU) 2023/1542, Annex II
    """

    # 1. Cathode Active Materials
    cathode_emissions = calculate_cathode_emissions(
        chemistry=battery_data["chemistry"]["cathode_material"],
        capacity_kwh=battery_data["capacity_kwh"],
        cobalt_source=battery_data.get("cobalt_source", "average"),
        nickel_source=battery_data.get("nickel_source", "average"),
        lithium_source=battery_data.get("lithium_source", "brine")
    )

    # 2. Anode Materials
    anode_emissions = calculate_anode_emissions(
        material=battery_data["chemistry"]["anode_material"],
        capacity_kwh=battery_data["capacity_kwh"],
        graphite_source=battery_data.get("graphite_source", "natural")
    )

    # 3. Electrolyte and Separator
    electrolyte_emissions = calculate_electrolyte_emissions(
        electrolyte_type=battery_data["chemistry"]["electrolyte_type"],
        capacity_kwh=battery_data["capacity_kwh"]
    )

    # 4. Cell Manufacturing
    cell_manufacturing_emissions = calculate_cell_manufacturing(
        location=battery_data["cell_manufacturing_location"],
        capacity_kwh=battery_data["capacity_kwh"],
        electricity_consumption=manufacturing_data["electricity_consumption"],
        yield_rate=manufacturing_data.get("yield_rate", 0.95)
    )

    # 5. Pack Assembly
    pack_assembly_emissions = calculate_pack_assembly(
        manufacturing_data=manufacturing_data,
        battery_capacity=battery_data["capacity_kwh"]
    )

    # 6. Distribution
    distribution_emissions = calculate_logistics_emissions(logistics_data)

    # Total Battery PCF (per kWh)
    total_pcf = (
        cathode_emissions +
        anode_emissions +
        electrolyte_emissions +
        cell_manufacturing_emissions +
        pack_assembly_emissions +
        distribution_emissions
    ) / battery_data["capacity_kwh"]

    # Determine Performance Class
    performance_class = determine_battery_performance_class(total_pcf)

    return BatteryPCFResult(
        pcf_per_kwh=total_pcf,
        performance_class=performance_class,
        breakdown={
            "cathode_materials": cathode_emissions / battery_data["capacity_kwh"],
            "anode_materials": anode_emissions / battery_data["capacity_kwh"],
            "electrolyte_separator": electrolyte_emissions / battery_data["capacity_kwh"],
            "cell_manufacturing": cell_manufacturing_emissions / battery_data["capacity_kwh"],
            "pack_assembly": pack_assembly_emissions / battery_data["capacity_kwh"],
            "distribution": distribution_emissions / battery_data["capacity_kwh"]
        }
    )

def determine_battery_performance_class(pcf_per_kwh: float) -> str:
    """
    Determine battery performance class based on PCF.

    Classes (indicative thresholds - final to be set by EC):
    A: < 50 kg CO2e/kWh
    B: 50-70 kg CO2e/kWh
    C: 70-90 kg CO2e/kWh
    D: 90-110 kg CO2e/kWh
    E: > 110 kg CO2e/kWh
    """
    if pcf_per_kwh < 50:
        return "A"
    elif pcf_per_kwh < 70:
        return "B"
    elif pcf_per_kwh < 90:
        return "C"
    elif pcf_per_kwh < 110:
        return "D"
    else:
        return "E"
```

### 5.4 Data Quality Indicators (DQI)

```python
def calculate_data_quality_indicators(
    data_sources: list
) -> DataQualityResult:
    """
    Calculate Data Quality Indicators per PACT Pathfinder.

    Dimensions (each scored 1-5):
    1. Technological representativeness
    2. Geographical representativeness
    3. Temporal representativeness
    4. Completeness
    5. Reliability

    Source: PACT Pathfinder Framework 2.1, Section 4.2
    """

    dqi_scores = {
        "technological": [],
        "geographical": [],
        "temporal": [],
        "completeness": [],
        "reliability": []
    }

    for source in data_sources:
        # Technological representativeness
        if source["type"] == "primary_measured":
            tech_score = 1
        elif source["type"] == "primary_calculated":
            tech_score = 2
        elif source["type"] == "secondary_specific":
            tech_score = 3
        elif source["type"] == "secondary_average":
            tech_score = 4
        else:
            tech_score = 5
        dqi_scores["technological"].append(tech_score)

        # Geographical representativeness
        if source.get("geographic_match") == "exact":
            geo_score = 1
        elif source.get("geographic_match") == "region":
            geo_score = 2
        elif source.get("geographic_match") == "country":
            geo_score = 3
        elif source.get("geographic_match") == "continent":
            geo_score = 4
        else:
            geo_score = 5
        dqi_scores["geographical"].append(geo_score)

        # Temporal representativeness
        data_age_years = source.get("data_age_years", 5)
        if data_age_years <= 1:
            temp_score = 1
        elif data_age_years <= 2:
            temp_score = 2
        elif data_age_years <= 3:
            temp_score = 3
        elif data_age_years <= 5:
            temp_score = 4
        else:
            temp_score = 5
        dqi_scores["temporal"].append(temp_score)

        # Completeness
        coverage = source.get("coverage_percent", 50)
        if coverage >= 95:
            comp_score = 1
        elif coverage >= 80:
            comp_score = 2
        elif coverage >= 60:
            comp_score = 3
        elif coverage >= 40:
            comp_score = 4
        else:
            comp_score = 5
        dqi_scores["completeness"].append(comp_score)

        # Reliability
        if source.get("verification") == "third_party":
            rel_score = 1
        elif source.get("verification") == "internal_review":
            rel_score = 2
        elif source.get("verification") == "documented":
            rel_score = 3
        else:
            rel_score = 4
        dqi_scores["reliability"].append(rel_score)

    # Calculate weighted averages
    avg_dqi = {
        dimension: sum(scores) / len(scores) if scores else 5
        for dimension, scores in dqi_scores.items()
    }

    # Overall DQI (geometric mean)
    overall_dqi = (
        avg_dqi["technological"] *
        avg_dqi["geographical"] *
        avg_dqi["temporal"] *
        avg_dqi["completeness"] *
        avg_dqi["reliability"]
    ) ** (1/5)

    # Quality rating
    if overall_dqi <= 1.5:
        rating = "very_good"
    elif overall_dqi <= 2.5:
        rating = "good"
    elif overall_dqi <= 3.5:
        rating = "fair"
    else:
        rating = "poor"

    return DataQualityResult(
        overall_score=overall_dqi,
        rating=rating,
        dimension_scores=avg_dqi,
        primary_data_share=calculate_primary_data_share(data_sources)
    )
```

---

## 6. Data Exchange Formats

### 6.1 PACT Pathfinder Export

```yaml
pact_pathfinder_format:
  example:
    id: "3fa85f64-5717-4562-b3fc-2c963f66afa6"
    specVersion: "2.1.0"
    version: 1
    created: "2025-12-04T10:00:00Z"
    status: "Active"
    validityPeriod:
      start: "2025-01-01"
      end: "2025-12-31"
    companyName: "Example Battery Corp"
    companyIds:
      - "urn:lei:5493001KJTIIGC8Y1R17"
    productDescription: "EV Battery Pack 75kWh"
    productIds:
      - "urn:gtin:01234567890123"
    productCategoryCpc: "46410"
    productNameCompany: "PowerCell 75"
    comment: "Cradle-to-gate PCF for 75kWh EV battery"
    pcf:
      declaredUnit: "kWh"
      unitaryProductAmount: 75
      productMassPerDeclaredUnit: 6.67  # 500kg / 75kWh
      exemptedEmissionsPercent: 2.5
      exemptedEmissionsDescription: "Packaging materials <5%"
      packagingEmissionsIncluded: false
      geographyCountrySubdivision: ""
      geographyCountry: "DE"
      referencePeriodStart: "2025-01-01"
      referencePeriodEnd: "2025-12-31"
      characterizationFactors: "AR6"
      crossSectoralStandardsUsed:
        - "GHG Protocol Product Standard"
        - "ISO 14067:2018"
      productOrSectorSpecificRules:
        - operator: "PEF"
          ruleNames:
            - "PEFCR Rechargeable Batteries"
      biogenicAccountingMethodology: "PEF"
      pcfExcludingBiogenic: 65.0
      pcfIncludingBiogenic: 63.5
      fossilGhgEmissions: 62.0
      fossilCarbonContent: 0.5
      biogenicCarbonContent: 1.5
      dLucGhgEmissions: 0.0
      landManagementGhgEmissions: 0.0
      otherBiogenicGhgEmissions: 1.0
      iLucGhgEmissions: 0.0
      biogenicCarbonWithdrawal: -1.5
      aircraftGhgEmissions: 0.0
      packagingGhgEmissions: 0.0
      allocationRulesDescription: "Mass allocation for mining co-products"
      uncertaintyAssessmentDescription: "Monte Carlo simulation, 95% CI: +/- 8%"
      primaryDataShare: 75.0
      dqi:
        coveragePercent: 95
        technologicalDQR: 1.5
        geographicalDQR: 2.0
        temporalDQR: 1.0
      assurance:
        coverage: "product line"
        level: "limited"
        boundary: "cradle-to-gate"
        providerName: "TUV SUD"
        completedAt: "2025-03-15"
        standardName: "ISO 14064-3"
```

### 6.2 Catena-X PCF Exchange

```yaml
catena_x_format:
  namespace: "https://catenax.io/schema/pcf/1.0.0"
  example:
    assetId: "urn:uuid:3fa85f64-5717-4562-b3fc-2c963f66afa6"
    productId: "Battery-Pack-75kWh"
    specVersion: "1.0.0"
    companyId: "BPNL00000003AYRE"
    productDescription: "75kWh Electric Vehicle Battery"
    pcf:
      productCarbonFootprint: 65.0
      fossilEmissions: 62.0
      biogenicEmissions: 1.5
      biogenicCarbonWithdrawal: -1.5
      reportingPeriod: "2025"
      geographicScope: "DE"
      primaryDataShare: 0.75
      boundaryProcesses:
        - "raw_material_extraction"
        - "material_processing"
        - "manufacturing"
      pcfExclBiogenic: 63.5
      pcfInclBiogenic: 65.0
    dataQuality:
      technological: 1.5
      geographical: 2.0
      temporal: 1.0
      completeness: 1.5
      reliability: 1.5
    verification:
      verificationStandard: "ISO_14064_3"
      verifier: "TUV_SUD"
      verificationDate: "2025-03-15"
      assuranceLevel: "LIMITED"
```

---

## 7. Golden Test Scenarios

### 7.1 Battery PCF Tests (15 tests)

```yaml
golden_tests_battery:
  - test_id: PCF-BAT-001
    name: "NMC battery cradle-to-gate PCF"
    input:
      battery_data:
        battery_type: "NMC"
        chemistry:
          cathode_material: "NCM811"
          anode_material: "graphite"
          electrolyte_type: "liquid"
        capacity_kwh: 75
        cobalt_content_percent: 10
        lithium_source: "brine"
        cell_manufacturing_location: "DE"
      manufacturing_data:
        electricity_consumption:
          total_kwh_per_unit: 3750  # 50 kWh/kWh
          source: "grid"
          grid_emission_factor: 0.35
    expected:
      pcf_result:
        pcf_per_kwh:
          range: [60, 80]  # kg CO2e/kWh
        performance_class: "B"
      breakdown:
        cathode_materials:
          range: [25, 35]  # kg CO2e/kWh
        cell_manufacturing:
          range: [15, 25]  # kg CO2e/kWh

  - test_id: PCF-BAT-002
    name: "LFP battery with renewable energy"
    input:
      battery_data:
        battery_type: "LFP"
        chemistry:
          cathode_material: "LiFePO4"
          anode_material: "graphite"
        capacity_kwh: 60
        lithium_source: "hard_rock"
        cell_manufacturing_location: "CN"
      manufacturing_data:
        electricity_consumption:
          source: "renewable_ppa"
          renewable_percentage: 100
    expected:
      pcf_result:
        pcf_per_kwh:
          range: [45, 65]
        performance_class: "B"  # or "A"

  - test_id: PCF-BAT-003
    name: "High-cobalt NCA battery"
    input:
      battery_data:
        battery_type: "NCA"
        chemistry:
          cathode_material: "NCA"
        capacity_kwh: 100
        cobalt_content_percent: 15
        cobalt_source: "DRC_artisanal"  # High-emission source
    expected:
      pcf_result:
        pcf_per_kwh:
          range: [75, 95]
        performance_class: "C"

  - test_id: PCF-BAT-004
    name: "Battery with recycled materials"
    input:
      battery_data:
        battery_type: "NMC"
        capacity_kwh: 80
      bill_of_materials:
        - material_name: "cobalt"
          recycled_content_percent: 20
        - material_name: "lithium"
          recycled_content_percent: 10
        - material_name: "nickel"
          recycled_content_percent: 15
    expected:
      pcf_result:
        pcf_per_kwh:
          less_than: 70  # Lower due to recycled content
      recycled_content_benefit_kg_co2e:
        greater_than: 5

  - test_id: PCF-BAT-005
    name: "Solid-state battery (future technology)"
    input:
      battery_data:
        battery_type: "Solid_State"
        chemistry:
          cathode_material: "NCM"
          anode_material: "lithium_metal"
          electrolyte_type: "solid"
        capacity_kwh: 100
        energy_density_wh_kg: 400
    expected:
      pcf_result:
        pcf_per_kwh:
          range: [50, 70]
        performance_class: "A"

  - test_id: PCF-BAT-006
    name: "Battery manufacturing in high-carbon grid"
    input:
      battery_data:
        cell_manufacturing_location: "CN"
      manufacturing_data:
        electricity_consumption:
          source: "grid"
          grid_emission_factor: 0.58  # China average
    expected:
      pcf_result:
        pcf_per_kwh:
          range: [80, 110]
        breakdown:
          cell_manufacturing:
            greater_than: 25

  - test_id: PCF-BAT-007
    name: "Battery manufacturing in low-carbon grid"
    input:
      battery_data:
        cell_manufacturing_location: "SE"
      manufacturing_data:
        electricity_consumption:
          source: "grid"
          grid_emission_factor: 0.04  # Sweden
    expected:
      pcf_result:
        pcf_per_kwh:
          range: [40, 60]
        breakdown:
          cell_manufacturing:
            less_than: 10

  - test_id: PCF-BAT-008
    name: "Full lifecycle battery PCF (cradle-to-grave)"
    input:
      calculation_parameters:
        system_boundary: "cradle_to_grave"
      battery_data:
        capacity_kwh: 75
        cycle_life: 1500
      use_phase_data:
        energy_consumption_kwh: 15000  # Lifetime charging losses
        energy_source: "electricity"
      end_of_life_data:
        recyclability_rate: 95
        actual_recycling_rate: 70
    expected:
      pcf_result:
        breakdown:
          use_phase_kg_co2e:
            range: [3000, 6000]  # Depends on grid
          end_of_life_kg_co2e:
            range: [-500, 200]  # Can be credit

  - test_id: PCF-BAT-009
    name: "Battery with uncertain data quality"
    input:
      bill_of_materials:
        - material_name: "cathode_material"
          data_source: "industry_average"
          data_age_years: 4
      manufacturing_data:
        data_source: "estimate"
    expected:
      data_quality_rating: "fair"
      dqi:
        overall_score:
          greater_than: 3.0
      primary_data_share:
        less_than: 50

  - test_id: PCF-BAT-010
    name: "Battery with high primary data share"
    input:
      bill_of_materials:
        - material_name: "cathode_material"
          supplier_pcf:
            value_kg_co2e: 15
            verification_status: "third_party_verified"
      manufacturing_data:
        data_source: "measured"
        verification_status: "verified"
    expected:
      data_quality_rating: "very_good"
      primary_data_share:
        greater_than: 80

  - test_id: PCF-BAT-011
    name: "Battery performance class A threshold"
    input:
      battery_data:
        battery_type: "LFP"
        cell_manufacturing_location: "SE"
      manufacturing_data:
        electricity_consumption:
          source: "renewable_ppa"
    expected:
      pcf_result:
        pcf_per_kwh:
          less_than: 50
        performance_class: "A"

  - test_id: PCF-BAT-012
    name: "Battery performance class E threshold"
    input:
      battery_data:
        battery_type: "NCA"
        cobalt_source: "high_emission"
        cell_manufacturing_location: "high_carbon_grid"
      manufacturing_data:
        electricity_consumption:
          source: "coal_heavy_grid"
    expected:
      pcf_result:
        pcf_per_kwh:
          greater_than: 110
        performance_class: "E"

  - test_id: PCF-BAT-013
    name: "Industrial battery >2kWh"
    input:
      battery_data:
        battery_type: "LFP"
        capacity_kwh: 50
        application: "industrial_storage"
    expected:
      compliance:
        battery_regulation_scope: true

  - test_id: PCF-BAT-014
    name: "Light means of transport battery"
    input:
      battery_data:
        battery_type: "NMC"
        capacity_kwh: 0.5
        application: "e_bike"
    expected:
      compliance:
        battery_regulation_scope: true
        category: "LMT"

  - test_id: PCF-BAT-015
    name: "Battery with transportation emissions"
    input:
      logistics_data:
        inbound_logistics:
          - from_country: "AU"  # Lithium
            to_facility: "DE"
            distance_km: 14000
            mode: "sea"
          - from_country: "DRC"  # Cobalt
            to_facility: "DE"
            distance_km: 8000
            mode: "sea"
        outbound_logistics:
          - to_region: "EU"
            distance_km: 500
            mode: "road"
    expected:
      pcf_result:
        breakdown:
          distribution_kg_co2e:
            range: [2, 8]
```

### 7.2 General Product PCF Tests (15 tests)

```yaml
golden_tests_general_pcf:
  - test_id: PCF-GEN-001
    name: "Textile product cradle-to-gate"
    input:
      product_definition:
        product_category: "textile"
        functional_unit:
          value: 1
          unit: "kg"
          description: "1 kg of cotton fabric"
      bill_of_materials:
        - material_name: "cotton_fiber"
          quantity: 1.1
          unit: "kg"
          origin_country: "IN"
        - material_name: "dye_chemicals"
          quantity: 0.05
          unit: "kg"
    expected:
      pcf_result:
        pcf_excluding_biogenic:
          range: [8, 15]  # kg CO2e/kg

  - test_id: PCF-GEN-002
    name: "Recycled polyester textile"
    input:
      product_category: "textile"
      bill_of_materials:
        - material_name: "recycled_PET"
          quantity: 1.0
          unit: "kg"
          recycled_content_percent: 100
    expected:
      pcf_result:
        pcf_excluding_biogenic:
          range: [2, 5]  # Much lower than virgin

  - test_id: PCF-GEN-003
    name: "Chemical product with process emissions"
    input:
      product_category: "chemical"
      manufacturing_data:
        process_emissions:
          co2_kg_per_unit: 0.5
          ch4_kg_per_unit: 0.01
          n2o_kg_per_unit: 0.001
    expected:
      pcf_result:
        breakdown:
          process_emissions_kg_co2e:
            calculation: "0.5 + (0.01 * 27.9) + (0.001 * 273)"

  - test_id: PCF-GEN-004
    name: "Steel product with high recycled content"
    input:
      product_category: "metal"
      bill_of_materials:
        - material_name: "steel"
          quantity: 100
          unit: "kg"
          recycled_content_percent: 80
    expected:
      pcf_result:
        pcf_excluding_biogenic:
          range: [50, 100]  # kg CO2e per 100kg

  - test_id: PCF-GEN-005
    name: "Aluminium with primary vs recycled"
    input:
      bill_of_materials:
        - material_name: "aluminium_primary"
          quantity: 50
          unit: "kg"
        - material_name: "aluminium_recycled"
          quantity: 50
          unit: "kg"
    expected:
      pcf_result:
        comparison:
          primary_contribution: "~90% of materials emissions"
          recycled_contribution: "~10% of materials emissions"

  - test_id: PCF-GEN-006
    name: "Electronics with complex BOM"
    input:
      product_category: "electronics"
      bill_of_materials:
        - material_name: "PCB"
          quantity: 0.1
          unit: "kg"
        - material_name: "semiconductor_chips"
          quantity: 0.02
          unit: "kg"
        - material_name: "plastic_housing"
          quantity: 0.3
          unit: "kg"
        - material_name: "copper_wiring"
          quantity: 0.05
          unit: "kg"
    expected:
      pcf_result:
        breakdown:
          semiconductor_contribution:
            note: "High per-kg emission factor"

  - test_id: PCF-GEN-007
    name: "Product with use phase energy"
    input:
      calculation_parameters:
        system_boundary: "cradle_to_grave"
      use_phase_data:
        energy_consumption_kwh: 1000
        product_lifetime_years: 10
    expected:
      pcf_result:
        breakdown:
          use_phase_kg_co2e:
            note: "Depends on grid factor"

  - test_id: PCF-GEN-008
    name: "Furniture product"
    input:
      product_category: "furniture"
      bill_of_materials:
        - material_name: "wood_fsc"
          quantity: 20
          unit: "kg"
        - material_name: "metal_fittings"
          quantity: 2
          unit: "kg"
        - material_name: "foam_cushion"
          quantity: 5
          unit: "kg"
    expected:
      pcf_result:
        pcf_excluding_biogenic:
          range: [30, 60]  # kg CO2e

  - test_id: PCF-GEN-009
    name: "Product with biogenic carbon"
    input:
      bill_of_materials:
        - material_name: "wood"
          quantity: 10
          unit: "kg"
          biogenic_carbon_content: 5  # kg C
    expected:
      pcf_result:
        pcf_excluding_biogenic:
          note: "Does not include biogenic"
        pcf_including_biogenic:
          note: "Includes -18.3 kg CO2 (biogenic uptake)"

  - test_id: PCF-GEN-010
    name: "Paint/coating product"
    input:
      product_category: "chemical"
      functional_unit:
        value: 1
        unit: "L"
      bill_of_materials:
        - material_name: "resins"
          quantity: 0.4
          unit: "kg"
        - material_name: "pigments"
          quantity: 0.2
          unit: "kg"
        - material_name: "solvents"
          quantity: 0.3
          unit: "kg"
    expected:
      pcf_result:
        pcf_excluding_biogenic:
          range: [2, 5]  # kg CO2e/L

  - test_id: PCF-GEN-011
    name: "Tyre product"
    input:
      product_category: "rubber"
      bill_of_materials:
        - material_name: "natural_rubber"
          quantity: 2
          unit: "kg"
        - material_name: "synthetic_rubber"
          quantity: 3
          unit: "kg"
        - material_name: "carbon_black"
          quantity: 2
          unit: "kg"
        - material_name: "steel_cord"
          quantity: 1
          unit: "kg"
    expected:
      pcf_result:
        pcf_excluding_biogenic:
          range: [15, 30]  # kg CO2e per tyre

  - test_id: PCF-GEN-012
    name: "Detergent product"
    input:
      product_category: "chemical"
      functional_unit:
        value: 1
        unit: "wash"
      bill_of_materials:
        - material_name: "surfactants"
          quantity: 0.02
          unit: "kg"
        - material_name: "enzymes"
          quantity: 0.001
          unit: "kg"
    expected:
      pcf_result:
        pcf_excluding_biogenic:
          range: [0.02, 0.08]  # kg CO2e per wash

  - test_id: PCF-GEN-013
    name: "Lubricant product"
    input:
      product_category: "chemical"
      bill_of_materials:
        - material_name: "base_oil_mineral"
          quantity: 0.9
          unit: "kg"
        - material_name: "additives"
          quantity: 0.1
          unit: "kg"
    expected:
      pcf_result:
        pcf_excluding_biogenic:
          range: [1, 3]  # kg CO2e/kg

  - test_id: PCF-GEN-014
    name: "Product with air freight"
    input:
      logistics_data:
        inbound_logistics:
          - mode: "air"
            distance_km: 10000
            weight_kg: 100
    expected:
      pcf_result:
        breakdown:
          distribution_kg_co2e:
            note: "Air freight ~600 kg CO2e"

  - test_id: PCF-GEN-015
    name: "Product with complex allocation"
    input:
      calculation_parameters:
        allocation_method: "economic"
      manufacturing_data:
        co_products:
          - name: "main_product"
            value_eur: 100
          - name: "by_product"
            value_eur: 20
    expected:
      allocation:
        main_product_share: 0.833
        by_product_share: 0.167
```

### 7.3 PEF Calculation Tests (10 tests)

```yaml
golden_tests_pef:
  - test_id: PCF-PEF-001
    name: "All 16 impact categories calculated"
    input:
      product_category: "textile"
      bill_of_materials:
        - material_name: "cotton"
          quantity: 1
          unit: "kg"
    expected:
      pef_results:
        categories_calculated: 16
        climate_change:
          unit: "kg CO2 eq"
        ozone_depletion:
          unit: "kg CFC-11 eq"
        # ... all 16 categories present

  - test_id: PCF-PEF-002
    name: "Normalized and weighted PEF results"
    input:
      # Same as above
    expected:
      pef_results:
        normalized_scores:
          climate_change: "number"
        weighted_scores:
          climate_change: "number * 0.2106"
        single_score: "sum of weighted"

  - test_id: PCF-PEF-003
    name: "Water use in water-stressed region"
    input:
      manufacturing_data:
        water_consumption_m3: 100
        location: "water_stressed_region"
    expected:
      pef_results:
        water_use:
          note: "Higher impact due to water stress factor"

  - test_id: PCF-PEF-004
    name: "Land use impact for agricultural product"
    input:
      bill_of_materials:
        - material_name: "soy"
          origin_country: "BR"
          land_use_change: true
    expected:
      pef_results:
        land_use:
          note: "Includes DLUC impact"
        climate_change:
          includes: "Climate change - land use"

  - test_id: PCF-PEF-005
    name: "Toxicity impact for chemical product"
    input:
      bill_of_materials:
        - material_name: "heavy_metal_catalyst"
    expected:
      pef_results:
        human_toxicity_cancer:
          note: "Elevated due to heavy metals"
        ecotoxicity_freshwater:
          note: "Elevated due to heavy metals"

  - test_id: PCF-PEF-006
    name: "Acidification from combustion"
    input:
      manufacturing_data:
        fuel_consumption:
          type: "diesel"
          quantity_L: 100
    expected:
      pef_results:
        acidification:
          note: "SOx and NOx from combustion"

  - test_id: PCF-PEF-007
    name: "Eutrophication from fertilizer production"
    input:
      bill_of_materials:
        - material_name: "nitrogen_fertilizer"
    expected:
      pef_results:
        eutrophication_terrestrial:
          note: "N emissions from fertilizer"
        eutrophication_marine:
          note: "N emissions from fertilizer"

  - test_id: PCF-PEF-008
    name: "Resource depletion for rare materials"
    input:
      bill_of_materials:
        - material_name: "cobalt"
        - material_name: "rare_earth_elements"
    expected:
      pef_results:
        resource_use_minerals:
          note: "High due to rare materials"

  - test_id: PCF-PEF-009
    name: "Fossil resource use"
    input:
      bill_of_materials:
        - material_name: "plastic_PP"
          quantity: 1
          unit: "kg"
    expected:
      pef_results:
        resource_use_fossils:
          range: [60, 80]  # MJ/kg PP

  - test_id: PCF-PEF-010
    name: "Particulate matter from manufacturing"
    input:
      manufacturing_data:
        process_type: "metal_grinding"
    expected:
      pef_results:
        particulate_matter:
          note: "PM2.5 and PM10 from grinding"
```

### 7.4 Data Exchange Tests (10 tests)

```yaml
golden_tests_data_exchange:
  - test_id: PCF-EXP-001
    name: "Valid PACT Pathfinder export"
    input:
      pcf_result:
        pcf_excluding_biogenic: 65.0
    expected:
      pact_export:
        specVersion: "2.1.0"
        status: "Active"
        pcf:
          pcfExcludingBiogenic: 65.0
        valid_json: true

  - test_id: PCF-EXP-002
    name: "PACT export with all optional fields"
    input:
      full_pcf_data: true
    expected:
      pact_export:
        includes:
          - "dqi"
          - "assurance"
          - "aircraftGhgEmissions"
          - "packagingGhgEmissions"

  - test_id: PCF-EXP-003
    name: "Catena-X format export"
    input:
      pcf_result:
        pcf_excluding_biogenic: 65.0
    expected:
      catena_x_export:
        namespace: "https://catenax.io/schema/pcf/1.0.0"
        assetId: "uuid format"
        valid_schema: true

  - test_id: PCF-EXP-004
    name: "Digital Product Passport generation"
    input:
      product_id: "BAT-001"
      pcf_result:
        pcf_per_kwh: 65.0
        performance_class: "B"
    expected:
      digital_product_passport:
        passport_id: "uuid"
        qr_code_url: "valid URL"
        carbon_footprint: 65.0
        performance_class: "B"

  - test_id: PCF-EXP-005
    name: "QR code generation for DPP"
    input:
      dpp_data: true
    expected:
      qr_code:
        format: "PNG"
        size: "200x200"
        data_url: "https://..."

  - test_id: PCF-EXP-006
    name: "Export with verification data"
    input:
      verification:
        verifier_name: "TUV SUD"
        assurance_level: "limited"
    expected:
      pact_export:
        assurance:
          level: "limited"
          providerName: "TUV SUD"

  - test_id: PCF-EXP-007
    name: "Export with data quality indicators"
    input:
      dqi:
        technological: 1.5
        geographical: 2.0
    expected:
      pact_export:
        dqi:
          technologicalDQR: 1.5
          geographicalDQR: 2.0

  - test_id: PCF-EXP-008
    name: "Import PACT data from supplier"
    input:
      supplier_pact_json: "valid JSON"
    expected:
      import_result:
        parsed: true
        emission_factor_extracted: true

  - test_id: PCF-EXP-009
    name: "Bidirectional Catena-X exchange"
    input:
      direction: "send_and_receive"
    expected:
      catena_x:
        send_status: "success"
        receive_status: "success"

  - test_id: PCF-EXP-010
    name: "Export provenance and audit trail"
    input:
      pcf_calculation: true
    expected:
      audit_trail:
        input_hash: "SHA-256"
        output_hash: "SHA-256"
        calculation_timestamp: "ISO8601"
        emission_factor_versions: "object"
```

---

## 8. Data Dependencies

```yaml
data_dependencies:
  emission_factor_databases:
    ecoinvent:
      name: "ecoinvent 3.10"
      coverage: "18,000+ datasets"
      update: "Annual"

    gabi:
      name: "GaBi Databases"
      coverage: "Industry-specific"
      update: "Annual"

    elcd:
      name: "European Life Cycle Database"
      coverage: "EU reference data"
      access: "Free"

  characterization_factors:
    ef_3_1:
      name: "EF 3.1 characterization factors"
      source: "EU JRC"
      categories: 16

  grid_emission_factors:
    iea:
      name: "IEA Emission Factors"
      coverage: "Global by country"

    ecoinvent:
      name: "ecoinvent electricity markets"
      coverage: "Country/region specific"

  transport_factors:
    glec:
      name: "GLEC Framework"
      coverage: "All transport modes"
```

---

## 9. Implementation Roadmap

```yaml
implementation_roadmap:
  phase_1_core:
    duration: "Weeks 1-6"
    deliverables:
      - "BOM carbon calculator"
      - "Manufacturing calculator"
      - "Basic PCF output"

  phase_2_battery:
    duration: "Weeks 7-12"
    deliverables:
      - "Battery-specific calculator"
      - "Performance class determination"
      - "Battery Regulation compliance"

  phase_3_pef:
    duration: "Weeks 13-18"
    deliverables:
      - "All 16 PEF categories"
      - "Normalization and weighting"
      - "Full PEF output"

  phase_4_exchange:
    duration: "Weeks 19-24"
    deliverables:
      - "PACT Pathfinder export"
      - "Catena-X integration"
      - "DPP generation"
```

---

## 10. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-ProductManager | Initial specification |

**Approvals:**

- Climate Science Lead: ___________________ Date: _______
- Supply Chain Lead: ___________________ Date: _______
- Engineering Lead: ___________________ Date: _______
- Product Manager: ___________________ Date: _______

---

**END OF SPECIFICATION**
