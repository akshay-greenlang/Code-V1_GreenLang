# GL-012 Carbon Offset Verification Agent Specification

**Agent ID:** gl-012-offset-verification-v1
**Version:** 1.0.0
**Date:** 2025-12-04
**Priority:** P4-STANDARD
**Deadline:** Ongoing (Market-driven, increasing scrutiny)
**Status:** SPECIFICATION COMPLETE

---

## 1. Executive Summary

### 1.1 Market Overview

**Carbon Offset Market Context**

The voluntary carbon market has grown significantly but faces increasing scrutiny regarding offset quality, additionality, and permanence. Organizations using offsets for carbon neutrality or net-zero claims require robust verification to meet regulatory requirements (EU Green Claims Directive) and stakeholder expectations.

**Key Requirements:**
- Registry integration (Verra VCS, Gold Standard, ACR, CAR, Plan Vivo)
- Additionality assessment
- Permanence risk evaluation
- Double counting prevention
- Vintage and retirement tracking
- Article 6 compliance (Paris Agreement)

### 1.2 Agent Purpose

The Carbon Offset Verification Agent automates quality assessment, validation, and tracking of carbon offset credits. It provides:

1. **Project Validation** - Verify offset projects against registry standards
2. **Quality Scoring** - Multi-criteria quality assessment (additionality, permanence, co-benefits)
3. **Registry Integration** - Real-time data from major registries
4. **Double Counting Prevention** - Corresponding adjustment tracking
5. **Retirement Management** - Track retirements and claims matching
6. **Compliance Alignment** - EU Green Claims Directive and SBTi alignment

---

## 2. Offset Standards and Registries

### 2.1 Major Registry Standards

```yaml
offset_registries:
  verra_vcs:
    name: "Verified Carbon Standard (VCS)"
    organization: "Verra"
    url: "https://registry.verra.org"
    founded: 2007
    credits_issued: "1+ billion tCO2e"
    project_types:
      - "REDD+"
      - "Afforestation/Reforestation"
      - "Improved Forest Management"
      - "Renewable Energy"
      - "Energy Efficiency"
      - "Waste Management"
      - "Agriculture"
      - "Blue Carbon"
    methodology_count: "80+"
    quality_tier: "STANDARD"

    api:
      base_url: "https://registry.verra.org/api"
      endpoints:
        projects: "/projects"
        credits: "/credits"
        retirements: "/retirements"
      authentication: "API Key"

  gold_standard:
    name: "Gold Standard for the Global Goals"
    organization: "Gold Standard Foundation"
    url: "https://registry.goldstandard.org"
    founded: 2003
    credits_issued: "200+ million tCO2e"
    differentiator: "SDG co-benefits certification"
    project_types:
      - "Clean Cooking"
      - "Renewable Energy"
      - "Water Purification"
      - "Agroforestry"
      - "Biogas"
      - "Energy Efficiency"
    quality_tier: "PREMIUM"

    requirements:
      - "Stakeholder consultation"
      - "SDG impact certification"
      - "Third-party verification"
      - "Performance-based issuance"

  acr:
    name: "American Carbon Registry"
    organization: "Winrock International"
    url: "https://acr2.apx.com"
    founded: 1996
    focus: "North American projects"
    project_types:
      - "Forest Carbon"
      - "Avoided Conversion"
      - "Livestock Methane"
      - "Landfill Methane"
      - "N2O Destruction"
    quality_tier: "STANDARD"

  car:
    name: "Climate Action Reserve"
    organization: "Climate Action Reserve"
    url: "https://thereserve2.apx.com"
    founded: 2001
    focus: "North American compliance and voluntary"
    project_types:
      - "U.S. Forests"
      - "Livestock"
      - "Landfill"
      - "Ozone Depleting Substances"
      - "Rice Cultivation"
      - "Nitrogen Management"
    quality_tier: "STANDARD"

  plan_vivo:
    name: "Plan Vivo"
    organization: "Plan Vivo Foundation"
    url: "https://www.planvivo.org"
    founded: 1994
    focus: "Community-based land use"
    differentiator: "Smallholder and community focus"
    project_types:
      - "Community Forestry"
      - "Agroforestry"
      - "REDD+"
      - "Peatland Restoration"
    quality_tier: "PREMIUM"

  puro_earth:
    name: "Puro.earth"
    organization: "Puro.earth"
    url: "https://puro.earth"
    founded: 2019
    focus: "Engineered carbon removal (CDR)"
    project_types:
      - "Biochar"
      - "Bio-based Construction"
      - "Enhanced Weathering"
      - "Carbonated Materials"
    quality_tier: "PREMIUM_CDR"
    differentiator: "Long-term storage (100+ years)"
```

### 2.2 Project Type Categories

```yaml
project_categories:
  avoidance_reduction:
    description: "Projects that avoid or reduce emissions"
    types:
      renewable_energy:
        examples: ["Wind", "Solar", "Hydro"]
        additionality_risk: "HIGH"
        permanence: "N/A (no storage)"
        typical_quality: "MEDIUM"

      energy_efficiency:
        examples: ["Cookstoves", "Industrial efficiency"]
        additionality_risk: "MEDIUM"
        permanence: "N/A"
        typical_quality: "MEDIUM-HIGH"

      methane_capture:
        examples: ["Landfill gas", "Livestock manure"]
        additionality_risk: "LOW"
        permanence: "N/A"
        typical_quality: "HIGH"

      avoided_deforestation:
        examples: ["REDD+", "JNR"]
        additionality_risk: "MEDIUM-HIGH"
        permanence: "MEDIUM (reversal risk)"
        typical_quality: "VARIABLE"

  nature_based_removal:
    description: "Biological carbon removal and storage"
    types:
      afforestation_reforestation:
        examples: ["Tree planting", "Forest restoration"]
        additionality_risk: "MEDIUM"
        permanence: "MEDIUM (100-year contracts typical)"
        permanence_mechanism: "Buffer pool"
        typical_quality: "MEDIUM-HIGH"

      improved_forest_management:
        examples: ["Extended rotation", "Conservation"]
        additionality_risk: "MEDIUM"
        permanence: "MEDIUM"
        typical_quality: "MEDIUM"

      blue_carbon:
        examples: ["Mangroves", "Seagrass", "Salt marshes"]
        additionality_risk: "LOW"
        permanence: "MEDIUM-HIGH"
        typical_quality: "HIGH"

      soil_carbon:
        examples: ["Regenerative agriculture", "Biochar application"]
        additionality_risk: "MEDIUM"
        permanence: "LOW-MEDIUM (highly variable)"
        typical_quality: "VARIABLE"

  engineered_removal:
    description: "Technology-based carbon dioxide removal (CDR)"
    types:
      dac:
        full_name: "Direct Air Capture"
        examples: ["Climeworks", "Carbon Engineering"]
        additionality_risk: "VERY LOW"
        permanence: "VERY HIGH (geological storage)"
        permanence_duration: "1000+ years"
        typical_quality: "PREMIUM"
        current_cost: "$400-1000/tCO2"

      beccs:
        full_name: "Bioenergy with Carbon Capture"
        additionality_risk: "LOW"
        permanence: "VERY HIGH"
        typical_quality: "HIGH"

      biochar:
        additionality_risk: "LOW"
        permanence: "HIGH (100-1000 years)"
        typical_quality: "HIGH"

      enhanced_weathering:
        additionality_risk: "LOW"
        permanence: "VERY HIGH"
        typical_quality: "HIGH"

      ocean_alkalinity:
        additionality_risk: "LOW"
        permanence: "VERY HIGH"
        typical_quality: "EMERGING"
```

---

## 3. Quality Assessment Framework

### 3.1 Quality Scoring Methodology

```yaml
quality_scoring:
  dimensions:
    additionality:
      weight: 0.30
      description: "Would emissions reductions occur without carbon finance?"
      scoring:
        5_excellent: "Regulatory surplus, no other funding sources, high cost barrier"
        4_good: "Voluntary activity, carbon revenue essential"
        3_fair: "Carbon revenue accelerates timeline"
        2_poor: "Project viable without carbon finance"
        1_very_poor: "Mandatory activity or business-as-usual"

      tests:
        regulatory_test: "Is activity required by law?"
        barrier_test: "Are there investment, institutional, or technical barriers?"
        common_practice_test: "Is activity common in the region/sector?"
        financial_test: "Is carbon revenue necessary for financial viability?"

    permanence:
      weight: 0.25
      description: "How long will carbon remain stored?"
      scoring:
        5_excellent: "1000+ years (geological, durable CDR)"
        4_good: "100+ years with legal/contractual guarantees"
        3_fair: "40-100 years with buffer pool"
        2_poor: "20-40 years, high reversal risk"
        1_very_poor: "<20 years or no guarantees"

      risk_factors:
        natural: ["Fire", "Disease", "Climate change"]
        human: ["Deforestation", "Land use change", "Policy changes"]
        economic: ["Economic pressure to convert land"]

    measurement_verification:
      weight: 0.20
      description: "How accurately are emissions reductions measured?"
      scoring:
        5_excellent: "Direct measurement, third-party verified, conservative"
        4_good: "Robust methodology, verified, appropriate uncertainty"
        3_fair: "Standard methodology, some estimation"
        2_poor: "High uncertainty, limited verification"
        1_very_poor: "No reliable measurement"

    co_benefits:
      weight: 0.15
      description: "Additional environmental and social benefits"
      scoring:
        5_excellent: "Certified SDG impacts, multiple verified benefits"
        4_good: "Strong biodiversity/community benefits, documented"
        3_fair: "Some co-benefits claimed"
        2_poor: "Minimal co-benefits"
        1_very_poor: "No co-benefits or negative externalities"

      sdg_alignment:
        - "SDG 13: Climate Action (core)"
        - "SDG 15: Life on Land"
        - "SDG 14: Life Below Water"
        - "SDG 8: Decent Work"
        - "SDG 1: No Poverty"
        - "SDG 7: Clean Energy"

    governance_transparency:
      weight: 0.10
      description: "Project governance and stakeholder engagement"
      scoring:
        5_excellent: "Full FPIC, benefit sharing, transparent reporting"
        4_good: "Stakeholder consultation, clear governance"
        3_fair: "Basic governance, some transparency"
        2_poor: "Limited stakeholder engagement"
        1_very_poor: "No consultation, opaque governance"

  overall_score:
    formula: |
      Quality_Score =
        (Additionality * 0.30) +
        (Permanence * 0.25) +
        (MRV * 0.20) +
        (Co_Benefits * 0.15) +
        (Governance * 0.10)

    rating_scale:
      A_premium: "4.5 - 5.0"
      B_high: "3.5 - 4.4"
      C_standard: "2.5 - 3.4"
      D_low: "1.5 - 2.4"
      F_reject: "< 1.5"
```

### 3.2 Double Counting Prevention

```yaml
double_counting_prevention:
  types:
    double_issuance:
      description: "Same reduction registered multiple times"
      mitigation: "Unique project registration, exclusive registry"

    double_use:
      description: "Same credit used for multiple claims"
      mitigation: "Retirement tracking, unique serial numbers"

    double_claiming:
      description: "Credit claimed by both buyer and host country NDC"
      mitigation: "Corresponding adjustments (Article 6)"

  article_6_compliance:
    description: "Paris Agreement Article 6 requirements"
    corresponding_adjustment:
      definition: "Host country adjusts NDC to avoid double counting"
      requirement: "For international transfer of mitigation outcomes (ITMOs)"
      status:
        2024: "Voluntary for most projects"
        2025: "CORSIA requires CA"
        2026_plus: "Expected wider adoption"

    authorized_credits:
      icvcm_core_carbon_principles:
        - "Real emissions reductions"
        - "Additionality"
        - "Robust quantification"
        - "Permanence"
        - "No double counting"
        - "SDG safeguards"

  verification_checks:
    - "Unique serial number validation"
    - "Registry cross-check"
    - "Host country authorization status"
    - "Retirement status verification"
    - "Claim exclusivity confirmation"
```

---

## 4. Agent Architecture

### 4.1 Agent Specification (AgentSpec v2)

```yaml
agent_id: gl-012-offset-verification-v1
name: "Carbon Offset Verification Agent"
version: "1.0.0"
type: offset-verification
priority: P4-STANDARD
deadline: "ongoing"

description: |
  Automated carbon offset quality assessment, validation, and tracking agent.
  Integrates with major registries, assesses offset quality across multiple
  dimensions, prevents double counting, and generates substantiation reports
  for carbon neutrality claims.

market_context:
  vcm_size: "$2+ billion annually"
  quality_concerns: "High (media scrutiny, regulatory pressure)"
  key_drivers:
    - "EU Green Claims Directive"
    - "SBTi Beyond Value Chain Mitigation"
    - "ICVCM Core Carbon Principles"
    - "VCMI Claims Code of Practice"

inputs:
  offset_portfolio:
    type: array
    required: true
    description: "Portfolio of offset credits to verify"
    items:
      type: object
      properties:
        credit_id:
          type: string
          description: "Internal tracking ID"

        registry:
          type: string
          enum: ["Verra_VCS", "Gold_Standard", "ACR", "CAR", "Plan_Vivo", "Puro_Earth", "Other"]
          required: true

        project_id:
          type: string
          required: true
          description: "Registry project ID"
          examples: ["VCS-1234", "GS-5678"]

        serial_numbers:
          type: array
          items:
            type: string
          description: "Unique credit serial numbers"

        quantity_tco2e:
          type: number
          minimum: 0
          required: true

        vintage_year:
          type: integer
          minimum: 2000
          maximum: 2030
          required: true

        project_type:
          type: string
          enum: ["renewable_energy", "energy_efficiency", "methane_capture", "redd_plus", "afforestation", "improved_forest_management", "blue_carbon", "soil_carbon", "biochar", "dac", "beccs", "enhanced_weathering", "clean_cooking", "other"]

        project_country:
          type: string
          pattern: "^[A-Z]{2}$"

        retirement_status:
          type: string
          enum: ["active", "retired", "cancelled"]

        retirement_date:
          type: string
          format: date

        retirement_beneficiary:
          type: string

        purchase_price_usd:
          type: number

        purchase_date:
          type: string
          format: date

        broker_name:
          type: string

        verification_documents:
          type: array
          items:
            type: object
            properties:
              document_type:
                type: string
                enum: ["verification_report", "monitoring_report", "methodology", "project_design", "retirement_certificate"]
              document_url:
                type: string
                format: uri

  company_emissions:
    type: object
    description: "Company emissions for claims matching"
    properties:
      scope_1_tco2e:
        type: number
      scope_2_tco2e:
        type: number
      scope_3_tco2e:
        type: number
      total_tco2e:
        type: number
      reporting_year:
        type: integer

  claim_type:
    type: string
    enum: ["carbon_neutral", "net_zero", "climate_positive", "offset_contribution", "beyond_value_chain"]
    description: "Type of claim the offsets will support"

  verification_requirements:
    type: object
    properties:
      minimum_quality_score:
        type: number
        default: 3.0

      require_third_party_verification:
        type: boolean
        default: true

      require_corresponding_adjustment:
        type: boolean
        default: false
        description: "Require Article 6 CA for claims"

      vintage_cutoff_years:
        type: integer
        default: 5
        description: "Maximum vintage age in years"

      excluded_project_types:
        type: array
        items:
          type: string

      require_co_benefits:
        type: boolean
        default: false

outputs:
  portfolio_assessment:
    type: object
    properties:
      assessment_id:
        type: string
        format: uuid

      assessment_date:
        type: string
        format: date-time

      portfolio_summary:
        type: object
        properties:
          total_credits_tco2e:
            type: number
          verified_credits_tco2e:
            type: number
          qualified_credits_tco2e:
            type: number
          rejected_credits_tco2e:
            type: number
          average_quality_score:
            type: number
          quality_rating:
            type: string
            enum: ["A_premium", "B_high", "C_standard", "D_low", "F_reject"]
          total_value_usd:
            type: number
          average_price_per_tco2e:
            type: number

      credit_assessments:
        type: array
        items:
          type: object
          properties:
            credit_id:
              type: string
            project_id:
              type: string
            registry:
              type: string
            quantity_tco2e:
              type: number

            registry_verification:
              type: object
              properties:
                project_exists:
                  type: boolean
                project_active:
                  type: boolean
                credits_valid:
                  type: boolean
                retirement_confirmed:
                  type: boolean
                serial_numbers_verified:
                  type: boolean

            quality_assessment:
              type: object
              properties:
                overall_score:
                  type: number
                quality_rating:
                  type: string
                dimension_scores:
                  type: object
                  properties:
                    additionality:
                      type: number
                    permanence:
                      type: number
                    measurement_verification:
                      type: number
                    co_benefits:
                      type: number
                    governance:
                      type: number

            additionality_assessment:
              type: object
              properties:
                score:
                  type: number
                regulatory_test:
                  type: string
                  enum: ["PASS", "FAIL", "UNCERTAIN"]
                barrier_test:
                  type: string
                common_practice_test:
                  type: string
                financial_test:
                  type: string
                concerns:
                  type: array
                  items:
                    type: string

            permanence_assessment:
              type: object
              properties:
                score:
                  type: number
                storage_type:
                  type: string
                  enum: ["geological", "biological", "product", "none"]
                expected_duration_years:
                  type: number
                reversal_risk:
                  type: string
                  enum: ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
                buffer_pool_contribution:
                  type: number
                  description: "% of credits in buffer pool"
                insurance_mechanism:
                  type: string

            double_counting_check:
              type: object
              properties:
                double_issuance_risk:
                  type: string
                  enum: ["LOW", "MEDIUM", "HIGH"]
                double_use_risk:
                  type: string
                retirement_verified:
                  type: boolean
                corresponding_adjustment:
                  type: object
                  properties:
                    required:
                      type: boolean
                    obtained:
                      type: boolean
                    host_country_authorization:
                      type: boolean

            vintage_assessment:
              type: object
              properties:
                vintage_year:
                  type: integer
                age_years:
                  type: integer
                meets_cutoff:
                  type: boolean
                vintage_quality:
                  type: string
                  enum: ["CURRENT", "RECENT", "AGED", "EXPIRED"]

            co_benefits_assessment:
              type: object
              properties:
                score:
                  type: number
                sdg_certifications:
                  type: array
                  items:
                    type: object
                    properties:
                      sdg_number:
                        type: integer
                      certification_body:
                        type: string
                community_benefits:
                  type: boolean
                biodiversity_benefits:
                  type: boolean
                livelihood_benefits:
                  type: boolean

            compliance_check:
              type: object
              properties:
                meets_requirements:
                  type: boolean
                failures:
                  type: array
                  items:
                    type: string
                warnings:
                  type: array
                  items:
                    type: string

            recommendation:
              type: string
              enum: ["ACCEPT", "ACCEPT_WITH_CAVEATS", "REJECT"]

      claims_substantiation:
        type: object
        properties:
          claim_type:
            type: string
          emissions_to_offset:
            type: number
          qualified_offsets:
            type: number
          coverage_percent:
            type: number
          claim_supportable:
            type: boolean
          substantiation_statement:
            type: string
          disclosure_requirements:
            type: array
            items:
              type: string

      risk_summary:
        type: object
        properties:
          additionality_risk:
            type: string
            enum: ["LOW", "MEDIUM", "HIGH"]
          permanence_risk:
            type: string
          double_counting_risk:
            type: string
          reputational_risk:
            type: string
          key_concerns:
            type: array
            items:
              type: string

      recommendations:
        type: array
        items:
          type: object
          properties:
            category:
              type: string
            recommendation:
              type: string
            priority:
              type: string
              enum: ["HIGH", "MEDIUM", "LOW"]

      audit_trail:
        type: object
        properties:
          input_hash:
            type: string
          output_hash:
            type: string
          registry_queries:
            type: array
            items:
              type: object
          data_sources:
            type: array
            items:
              type: string

  substantiation_report:
    type: object
    description: "Report for Green Claims Directive compliance"

  retirement_tracking:
    type: object
    properties:
      retirements:
        type: array
        items:
          type: object
          properties:
            credit_id:
              type: string
            serial_numbers:
              type: array
            quantity_tco2e:
              type: number
            retirement_date:
              type: string
            beneficiary:
              type: string
            claim_type:
              type: string
            verification_url:
              type: string

tools:
  - name: registry_connector
    type: connector
    description: "Connect to carbon registry APIs"
    inputs: ["registry", "project_id", "serial_numbers"]
    outputs: ["project_data", "credit_status", "retirement_status"]

  - name: quality_assessor
    type: analyzer
    description: "Assess offset quality across dimensions"
    inputs: ["project_data", "verification_documents"]
    outputs: ["quality_scores", "quality_rating"]

  - name: additionality_evaluator
    type: analyzer
    description: "Evaluate project additionality"
    inputs: ["project_data", "methodology", "country_context"]
    outputs: ["additionality_score", "test_results"]

  - name: permanence_evaluator
    type: analyzer
    description: "Evaluate permanence and reversal risk"
    inputs: ["project_type", "storage_mechanism", "contract_terms"]
    outputs: ["permanence_score", "reversal_risk"]

  - name: double_counting_checker
    type: validator
    description: "Check for double counting risks"
    inputs: ["serial_numbers", "registry", "retirement_status"]
    outputs: ["double_counting_risk", "verification_status"]

  - name: vintage_validator
    type: validator
    description: "Validate credit vintage"
    inputs: ["vintage_year", "cutoff_requirements"]
    outputs: ["vintage_assessment"]

  - name: co_benefits_analyzer
    type: analyzer
    description: "Analyze SDG and other co-benefits"
    inputs: ["project_data", "certifications"]
    outputs: ["co_benefits_score", "sdg_alignment"]

  - name: claims_matcher
    type: calculator
    description: "Match offsets to emissions for claims"
    inputs: ["qualified_offsets", "company_emissions", "claim_type"]
    outputs: ["claims_substantiation"]

  - name: substantiation_generator
    type: generator
    description: "Generate substantiation report"
    inputs: ["portfolio_assessment", "claim_type"]
    outputs: ["substantiation_report"]

  - name: retirement_tracker
    type: tracker
    description: "Track credit retirements"
    inputs: ["retirement_requests"]
    outputs: ["retirement_confirmations"]

  - name: provenance_tracker
    type: utility
    description: "Track assessment provenance"
    inputs: ["inputs", "outputs"]
    outputs: ["provenance_hash"]

evaluation:
  golden_tests:
    total_count: 50
    categories:
      quality_assessment: 15
      additionality: 10
      permanence: 10
      double_counting: 10
      claims_matching: 5

  accuracy_thresholds:
    quality_score: 0.10  # Within 0.5 points
    registry_verification: 1.00  # 100% accuracy
    double_counting_detection: 0.99

  benchmarks:
    latency_p95_seconds: 30
    cost_per_verification_usd: 0.50

certification:
  required_approvals:
    - climate_science_team
    - carbon_markets_team
    - legal_team

  compliance_checks:
    - registry_data_accuracy
    - quality_methodology
    - green_claims_directive_alignment
    - icvcm_alignment
```

---

## 5. Calculation Formulas

### 5.1 Quality Score Calculation

```python
def calculate_quality_score(
    project_data: dict,
    verification_data: dict
) -> QualityResult:
    """
    Calculate multi-dimensional offset quality score.

    Formula:
    Quality_Score =
        (Additionality * 0.30) +
        (Permanence * 0.25) +
        (MRV * 0.20) +
        (Co_Benefits * 0.15) +
        (Governance * 0.10)

    Source: ICVCM Core Carbon Principles, GreenLang Quality Framework
    """

    # 1. Additionality Score (1-5)
    additionality = assess_additionality(
        project_type=project_data["project_type"],
        methodology=project_data["methodology"],
        country=project_data["country"],
        financial_data=project_data.get("financial_data", {})
    )

    # 2. Permanence Score (1-5)
    permanence = assess_permanence(
        project_type=project_data["project_type"],
        storage_mechanism=project_data.get("storage_mechanism"),
        contract_duration=project_data.get("contract_duration_years"),
        buffer_pool=project_data.get("buffer_pool_percent", 0)
    )

    # 3. Measurement/Verification Score (1-5)
    mrv = assess_mrv(
        methodology=project_data["methodology"],
        verification_reports=verification_data.get("reports", []),
        uncertainty=project_data.get("uncertainty_percent")
    )

    # 4. Co-Benefits Score (1-5)
    co_benefits = assess_co_benefits(
        sdg_certifications=project_data.get("sdg_certifications", []),
        community_benefits=project_data.get("community_benefits", False),
        biodiversity=project_data.get("biodiversity_benefits", False)
    )

    # 5. Governance Score (1-5)
    governance = assess_governance(
        stakeholder_consultation=project_data.get("fpic_obtained", False),
        benefit_sharing=project_data.get("benefit_sharing", False),
        transparency=project_data.get("public_monitoring", False)
    )

    # Calculate weighted score
    quality_score = (
        additionality.score * 0.30 +
        permanence.score * 0.25 +
        mrv.score * 0.20 +
        co_benefits.score * 0.15 +
        governance.score * 0.10
    )

    # Determine quality rating
    if quality_score >= 4.5:
        rating = "A_premium"
    elif quality_score >= 3.5:
        rating = "B_high"
    elif quality_score >= 2.5:
        rating = "C_standard"
    elif quality_score >= 1.5:
        rating = "D_low"
    else:
        rating = "F_reject"

    return QualityResult(
        overall_score=quality_score,
        quality_rating=rating,
        dimension_scores={
            "additionality": additionality.score,
            "permanence": permanence.score,
            "measurement_verification": mrv.score,
            "co_benefits": co_benefits.score,
            "governance": governance.score
        },
        additionality_assessment=additionality,
        permanence_assessment=permanence,
        mrv_assessment=mrv,
        co_benefits_assessment=co_benefits,
        governance_assessment=governance
    )
```

### 5.2 Additionality Assessment

```python
def assess_additionality(
    project_type: str,
    methodology: str,
    country: str,
    financial_data: dict
) -> AdditionalityResult:
    """
    Assess project additionality using multiple tests.

    Tests:
    1. Regulatory Test: Is the activity required by law?
    2. Barrier Test: Are there barriers overcome by carbon finance?
    3. Common Practice Test: Is this common in the region?
    4. Financial Test: Is carbon revenue necessary for viability?

    Source: CDM Additionality Tool, VCS Methodology Requirements
    """

    tests = {}
    concerns = []

    # 1. Regulatory Test
    regulatory_requirements = get_regulatory_requirements(country, project_type)
    if regulatory_requirements.get("mandatory"):
        tests["regulatory"] = "FAIL"
        concerns.append("Activity may be mandatory under local regulations")
    elif regulatory_requirements.get("incentivized"):
        tests["regulatory"] = "UNCERTAIN"
        concerns.append("Activity receives government incentives")
    else:
        tests["regulatory"] = "PASS"

    # 2. Barrier Test
    barriers = assess_barriers(project_type, country)
    if barriers["investment_barrier"] or barriers["institutional_barrier"] or barriers["technical_barrier"]:
        tests["barrier"] = "PASS"
    else:
        tests["barrier"] = "FAIL"
        concerns.append("No significant barriers identified")

    # 3. Common Practice Test
    common_practice_rate = get_common_practice_rate(project_type, country)
    if common_practice_rate < 0.05:  # Less than 5% adoption
        tests["common_practice"] = "PASS"
    elif common_practice_rate < 0.20:
        tests["common_practice"] = "UNCERTAIN"
        concerns.append(f"Activity has {common_practice_rate*100:.0f}% adoption rate")
    else:
        tests["common_practice"] = "FAIL"
        concerns.append(f"Activity is common practice ({common_practice_rate*100:.0f}% adoption)")

    # 4. Financial Test (if data available)
    if financial_data:
        irr_without_carbon = financial_data.get("irr_without_carbon", 0)
        irr_with_carbon = financial_data.get("irr_with_carbon", 0)
        benchmark_irr = financial_data.get("benchmark_irr", 0.10)

        if irr_without_carbon < benchmark_irr and irr_with_carbon >= benchmark_irr:
            tests["financial"] = "PASS"
        elif irr_without_carbon >= benchmark_irr:
            tests["financial"] = "FAIL"
            concerns.append("Project financially viable without carbon revenue")
        else:
            tests["financial"] = "UNCERTAIN"
    else:
        tests["financial"] = "UNCERTAIN"
        concerns.append("Financial data not available for assessment")

    # Calculate score
    test_scores = {"PASS": 5, "UNCERTAIN": 3, "FAIL": 1}
    test_values = [test_scores[v] for v in tests.values()]
    score = sum(test_values) / len(test_values)

    # Adjust for known problematic project types
    if project_type == "renewable_energy" and country in ["CN", "IN"]:
        score = min(score, 3.0)
        concerns.append("Renewable energy additionality concerns in this jurisdiction")

    return AdditionalityResult(
        score=score,
        tests=tests,
        concerns=concerns
    )
```

### 5.3 Permanence Assessment

```python
def assess_permanence(
    project_type: str,
    storage_mechanism: str,
    contract_duration: int,
    buffer_pool: float
) -> PermanenceResult:
    """
    Assess permanence and reversal risk.

    Factors:
    1. Storage type (geological, biological, product)
    2. Expected storage duration
    3. Reversal risk factors
    4. Insurance/buffer mechanisms

    Source: Oxford Offsetting Principles, SBTi Net-Zero Standard
    """

    reversal_risks = []

    # Base permanence by storage type
    storage_permanence = {
        "geological": {"base_score": 5.0, "duration_years": 10000, "reversal_risk": "LOW"},
        "enhanced_weathering": {"base_score": 4.8, "duration_years": 10000, "reversal_risk": "LOW"},
        "biochar": {"base_score": 4.5, "duration_years": 500, "reversal_risk": "LOW"},
        "product_wood": {"base_score": 3.5, "duration_years": 50, "reversal_risk": "MEDIUM"},
        "forest": {"base_score": 3.0, "duration_years": 100, "reversal_risk": "MEDIUM"},
        "soil": {"base_score": 2.5, "duration_years": 30, "reversal_risk": "HIGH"},
        "avoided_emissions": {"base_score": "N/A", "duration_years": "N/A", "reversal_risk": "N/A"}
    }

    # Get base metrics
    storage_info = storage_permanence.get(storage_mechanism, {
        "base_score": 3.0,
        "duration_years": 50,
        "reversal_risk": "MEDIUM"
    })

    base_score = storage_info["base_score"]
    if base_score == "N/A":
        # Avoidance projects don't have permanence risk
        return PermanenceResult(
            score=5.0,
            storage_type="none",
            expected_duration_years=None,
            reversal_risk="N/A",
            notes="Avoidance project - no storage involved"
        )

    # Adjust for contract duration
    if contract_duration:
        if contract_duration >= 100:
            duration_adjustment = 0.5
        elif contract_duration >= 40:
            duration_adjustment = 0
        elif contract_duration >= 20:
            duration_adjustment = -0.5
        else:
            duration_adjustment = -1.0
            reversal_risks.append(f"Short contract duration ({contract_duration} years)")
    else:
        duration_adjustment = -0.5
        reversal_risks.append("No contract duration specified")

    # Adjust for buffer pool
    if buffer_pool:
        if buffer_pool >= 20:
            buffer_adjustment = 0.5
        elif buffer_pool >= 10:
            buffer_adjustment = 0.25
        else:
            buffer_adjustment = 0
            reversal_risks.append(f"Small buffer pool ({buffer_pool}%)")
    else:
        buffer_adjustment = -0.5 if storage_mechanism in ["forest", "soil"] else 0
        if storage_mechanism in ["forest", "soil"]:
            reversal_risks.append("No buffer pool for nature-based project")

    # Additional risk factors for nature-based
    if storage_mechanism in ["forest", "soil"]:
        # Climate risk adjustment
        reversal_risks.append("Vulnerable to climate-related disturbances (fire, drought)")

    final_score = max(1.0, min(5.0, base_score + duration_adjustment + buffer_adjustment))

    return PermanenceResult(
        score=final_score,
        storage_type=storage_mechanism,
        expected_duration_years=storage_info["duration_years"],
        reversal_risk=storage_info["reversal_risk"],
        buffer_pool_percent=buffer_pool,
        contract_duration_years=contract_duration,
        reversal_risks=reversal_risks
    )
```

### 5.4 Claims Matching

```python
def match_offsets_to_claims(
    qualified_offsets: float,
    company_emissions: dict,
    claim_type: str
) -> ClaimsMatchingResult:
    """
    Match qualified offsets to emission claims.

    Claim Types:
    - carbon_neutral: Offsets >= Total emissions
    - net_zero: 90-95% reduction + offsets for residual
    - climate_positive: Offsets > Total emissions
    - offset_contribution: Partial offsetting
    - beyond_value_chain: SBTi BVCM approach

    Source: SBTi Net-Zero Standard, VCMI Claims Code, ISO 14068
    """

    total_emissions = company_emissions.get("total_tco2e", 0)
    scope_1 = company_emissions.get("scope_1_tco2e", 0)
    scope_2 = company_emissions.get("scope_2_tco2e", 0)
    scope_3 = company_emissions.get("scope_3_tco2e", 0)

    result = {
        "claim_type": claim_type,
        "emissions_to_offset": 0,
        "qualified_offsets": qualified_offsets,
        "coverage_percent": 0,
        "claim_supportable": False,
        "disclosure_requirements": []
    }

    if claim_type == "carbon_neutral":
        # Must offset 100% of emissions
        result["emissions_to_offset"] = total_emissions
        result["coverage_percent"] = (qualified_offsets / total_emissions * 100) if total_emissions > 0 else 0
        result["claim_supportable"] = qualified_offsets >= total_emissions

        result["disclosure_requirements"] = [
            "Total emissions (Scope 1, 2, 3) disclosed",
            "Offset project details disclosed",
            "Offset retirement certificates provided",
            "Residual emissions (if any) disclosed",
            "Emission reduction efforts described"
        ]

        if result["claim_supportable"]:
            result["substantiation_statement"] = (
                f"Carbon neutral claim substantiated with {qualified_offsets:.0f} tCO2e "
                f"qualified offsets against {total_emissions:.0f} tCO2e total emissions."
            )
        else:
            result["substantiation_statement"] = (
                f"Carbon neutral claim NOT substantiated. Only {result['coverage_percent']:.1f}% coverage."
            )

    elif claim_type == "net_zero":
        # Only residual emissions (5-10%) can be offset
        required_reduction = 0.93  # 93% reduction required
        residual_target = total_emissions * (1 - required_reduction)

        result["emissions_to_offset"] = residual_target
        result["coverage_percent"] = (qualified_offsets / residual_target * 100) if residual_target > 0 else 0

        # Check if actually at net-zero state
        result["disclosure_requirements"] = [
            "Baseline year emissions",
            "Current year emissions",
            "Reduction achieved (must be 90-95%)",
            "Residual emissions",
            "Neutralization credits (permanent removal only)"
        ]

        result["substantiation_statement"] = (
            f"Net-zero claims require 90-95% absolute reduction. "
            f"Offsets can only neutralize residual {residual_target:.0f} tCO2e."
        )

    elif claim_type == "climate_positive":
        # Offsets must exceed emissions
        result["emissions_to_offset"] = total_emissions
        result["coverage_percent"] = (qualified_offsets / total_emissions * 100) if total_emissions > 0 else 0
        result["claim_supportable"] = qualified_offsets > total_emissions

        if result["claim_supportable"]:
            excess = qualified_offsets - total_emissions
            result["substantiation_statement"] = (
                f"Climate positive claim substantiated. {excess:.0f} tCO2e net removal."
            )

    elif claim_type == "beyond_value_chain":
        # SBTi BVCM - separate from value chain targets
        result["disclosure_requirements"] = [
            "Separate from SBTi target claims",
            "Quality criteria met (SBTi BVCM requirements)",
            "Contribution, not compensation framing",
            "Annual progress reporting"
        ]

        result["claim_supportable"] = True  # Contribution claims more flexible
        result["substantiation_statement"] = (
            f"Beyond Value Chain Mitigation: {qualified_offsets:.0f} tCO2e contribution to climate action."
        )

    return ClaimsMatchingResult(**result)
```

---

## 6. Golden Test Scenarios

### 6.1 Quality Assessment Tests (15 tests)

```yaml
golden_tests_quality:
  - test_id: OFF-QUAL-001
    name: "Premium quality DAC project"
    input:
      offset_portfolio:
        - registry: "Puro_Earth"
          project_type: "dac"
          quantity_tco2e: 100
          vintage_year: 2024
    expected:
      credit_assessments:
        - quality_assessment:
            overall_score:
              range: [4.5, 5.0]
            quality_rating: "A_premium"

  - test_id: OFF-QUAL-002
    name: "Gold Standard cookstove project"
    input:
      offset_portfolio:
        - registry: "Gold_Standard"
          project_type: "clean_cooking"
          quantity_tco2e: 1000
          vintage_year: 2023
    expected:
      credit_assessments:
        - quality_assessment:
            overall_score:
              range: [3.5, 4.5]
            quality_rating: "B_high"
            dimension_scores:
              co_benefits:
                greater_than: 4.0

  - test_id: OFF-QUAL-003
    name: "Old vintage renewable energy"
    input:
      offset_portfolio:
        - registry: "Verra_VCS"
          project_type: "renewable_energy"
          quantity_tco2e: 5000
          vintage_year: 2018
          project_country: "CN"
    expected:
      credit_assessments:
        - quality_assessment:
            quality_rating: "C_standard"
            dimension_scores:
              additionality:
                less_than: 3.0
        - vintage_assessment:
            vintage_quality: "AGED"
            meets_cutoff: false

  - test_id: OFF-QUAL-004
    name: "REDD+ project with concerns"
    input:
      offset_portfolio:
        - registry: "Verra_VCS"
          project_type: "redd_plus"
          quantity_tco2e: 10000
          vintage_year: 2022
    expected:
      credit_assessments:
        - quality_assessment:
            quality_rating: "C_standard"
            dimension_scores:
              permanence:
                range: [2.5, 3.5]
        - permanence_assessment:
            reversal_risk: "MEDIUM"

  - test_id: OFF-QUAL-005
    name: "Biochar removal project"
    input:
      offset_portfolio:
        - registry: "Puro_Earth"
          project_type: "biochar"
          quantity_tco2e: 500
          vintage_year: 2024
    expected:
      credit_assessments:
        - quality_assessment:
            overall_score:
              range: [4.0, 4.8]
            quality_rating: "A_premium"
        - permanence_assessment:
            expected_duration_years:
              greater_than: 100

  - test_id: OFF-QUAL-006
    name: "Community forestry with SDGs"
    input:
      offset_portfolio:
        - registry: "Plan_Vivo"
          project_type: "afforestation"
          quantity_tco2e: 2000
          sdg_certifications: ["SDG 1", "SDG 8", "SDG 15"]
    expected:
      credit_assessments:
        - quality_assessment:
            dimension_scores:
              co_benefits:
                greater_than: 4.0
              governance:
                greater_than: 4.0

  - test_id: OFF-QUAL-007
    name: "Landfill methane capture"
    input:
      offset_portfolio:
        - registry: "ACR"
          project_type: "methane_capture"
          quantity_tco2e: 8000
          vintage_year: 2023
    expected:
      credit_assessments:
        - quality_assessment:
            dimension_scores:
              additionality:
                range: [3.5, 4.5]
              permanence: 5.0  # No reversal for avoidance

  - test_id: OFF-QUAL-008
    name: "Low quality renewable energy"
    input:
      offset_portfolio:
        - registry: "Verra_VCS"
          project_type: "renewable_energy"
          vintage_year: 2020
          project_country: "IN"
          common_practice_rate: 0.25
    expected:
      credit_assessments:
        - quality_assessment:
            quality_rating: "D_low"
        - additionality_assessment:
            tests:
              common_practice: "FAIL"

  - test_id: OFF-QUAL-009
    name: "Blue carbon mangrove project"
    input:
      offset_portfolio:
        - registry: "Verra_VCS"
          project_type: "blue_carbon"
          quantity_tco2e: 3000
    expected:
      credit_assessments:
        - quality_assessment:
            overall_score:
              range: [3.5, 4.5]
            dimension_scores:
              co_benefits:
                greater_than: 3.5

  - test_id: OFF-QUAL-010
    name: "Soil carbon with uncertainty"
    input:
      offset_portfolio:
        - registry: "Verra_VCS"
          project_type: "soil_carbon"
          uncertainty_percent: 30
    expected:
      credit_assessments:
        - quality_assessment:
            dimension_scores:
              measurement_verification:
                less_than: 3.0
              permanence:
                less_than: 3.0

  - test_id: OFF-QUAL-011
    name: "Portfolio with mixed quality"
    input:
      offset_portfolio:
        - registry: "Puro_Earth"
          project_type: "dac"
          quantity_tco2e: 100
        - registry: "Gold_Standard"
          project_type: "clean_cooking"
          quantity_tco2e: 500
        - registry: "Verra_VCS"
          project_type: "renewable_energy"
          quantity_tco2e: 2000
          vintage_year: 2019
    expected:
      portfolio_summary:
        average_quality_score:
          range: [2.5, 3.5]

  - test_id: OFF-QUAL-012
    name: "Enhanced weathering project"
    input:
      offset_portfolio:
        - registry: "Other"
          project_type: "enhanced_weathering"
          quantity_tco2e: 200
    expected:
      credit_assessments:
        - permanence_assessment:
            expected_duration_years: 10000
            reversal_risk: "LOW"

  - test_id: OFF-QUAL-013
    name: "IFM project extended rotation"
    input:
      offset_portfolio:
        - registry: "CAR"
          project_type: "improved_forest_management"
          quantity_tco2e: 5000
    expected:
      credit_assessments:
        - quality_assessment:
            dimension_scores:
              additionality:
                range: [2.5, 3.5]

  - test_id: OFF-QUAL-014
    name: "Project without verification"
    input:
      offset_portfolio:
        - verification_documents: []
    expected:
      credit_assessments:
        - quality_assessment:
            dimension_scores:
              measurement_verification:
                less_than: 2.0
        - recommendation: "REJECT"

  - test_id: OFF-QUAL-015
    name: "Unregistered project"
    input:
      offset_portfolio:
        - registry: "Other"
          project_id: "UNKNOWN"
    expected:
      credit_assessments:
        - registry_verification:
            project_exists: false
        - recommendation: "REJECT"
```

### 6.2 Additionality Tests (10 tests)

```yaml
golden_tests_additionality:
  - test_id: OFF-ADD-001
    name: "Clear additionality - high cost barrier"
    input:
      project_type: "dac"
      financial_data:
        irr_without_carbon: -0.05
        irr_with_carbon: 0.08
        benchmark_irr: 0.10
    expected:
      additionality_assessment:
        score:
          greater_than: 4.0
        tests:
          financial: "PASS"
          barrier: "PASS"

  - test_id: OFF-ADD-002
    name: "Mandatory activity - fail regulatory"
    input:
      project_type: "landfill_gas"
      country: "DE"
      regulatory_requirement: true
    expected:
      additionality_assessment:
        tests:
          regulatory: "FAIL"
        concerns:
          includes: "mandatory under local regulations"

  - test_id: OFF-ADD-003
    name: "Common practice concern"
    input:
      project_type: "solar"
      country: "CN"
      common_practice_rate: 0.35
    expected:
      additionality_assessment:
        tests:
          common_practice: "FAIL"
        score:
          less_than: 3.0

  - test_id: OFF-ADD-004
    name: "Financially viable without carbon"
    input:
      financial_data:
        irr_without_carbon: 0.15
        benchmark_irr: 0.10
    expected:
      additionality_assessment:
        tests:
          financial: "FAIL"

  - test_id: OFF-ADD-005
    name: "Cookstove with clear barriers"
    input:
      project_type: "clean_cooking"
      country: "KE"
      barriers:
        investment_barrier: true
        institutional_barrier: true
    expected:
      additionality_assessment:
        tests:
          barrier: "PASS"
        score:
          greater_than: 3.5

  - test_id: OFF-ADD-006
    name: "First-of-its-kind technology"
    input:
      project_type: "dac"
      common_practice_rate: 0.001
    expected:
      additionality_assessment:
        tests:
          common_practice: "PASS"

  - test_id: OFF-ADD-007
    name: "Grid-connected renewable in India"
    input:
      project_type: "renewable_energy"
      country: "IN"
      year: 2023
    expected:
      additionality_assessment:
        concerns:
          includes: "Renewable energy additionality concerns"
        score:
          less_than: 3.0

  - test_id: OFF-ADD-008
    name: "Forest conservation with baseline"
    input:
      project_type: "redd_plus"
      deforestation_baseline: "historical"
      reference_region: "matched"
    expected:
      additionality_assessment:
        score:
          range: [2.5, 3.5]

  - test_id: OFF-ADD-009
    name: "Afforestation on degraded land"
    input:
      project_type: "afforestation"
      land_type: "degraded"
      previous_use: "abandoned_agriculture"
    expected:
      additionality_assessment:
        score:
          range: [3.5, 4.5]

  - test_id: OFF-ADD-010
    name: "Carbon revenue accelerates timeline"
    input:
      financial_data:
        project_timeline_without_carbon: 2030
        project_timeline_with_carbon: 2025
    expected:
      additionality_assessment:
        tests:
          financial: "UNCERTAIN"
```

### 6.3 Permanence Tests (10 tests)

```yaml
golden_tests_permanence:
  - test_id: OFF-PERM-001
    name: "Geological storage - maximum permanence"
    input:
      storage_mechanism: "geological"
    expected:
      permanence_assessment:
        score: 5.0
        expected_duration_years: 10000
        reversal_risk: "LOW"

  - test_id: OFF-PERM-002
    name: "Forest with 100-year contract"
    input:
      project_type: "afforestation"
      storage_mechanism: "forest"
      contract_duration: 100
      buffer_pool: 20
    expected:
      permanence_assessment:
        score:
          range: [3.5, 4.5]
        reversal_risk: "MEDIUM"

  - test_id: OFF-PERM-003
    name: "Forest with short contract"
    input:
      storage_mechanism: "forest"
      contract_duration: 20
      buffer_pool: 5
    expected:
      permanence_assessment:
        score:
          less_than: 2.5
        reversal_risks:
          includes: "Short contract duration"

  - test_id: OFF-PERM-004
    name: "Soil carbon high uncertainty"
    input:
      storage_mechanism: "soil"
      contract_duration: 40
    expected:
      permanence_assessment:
        score:
          range: [2.0, 3.0]
        reversal_risk: "HIGH"

  - test_id: OFF-PERM-005
    name: "Biochar stable storage"
    input:
      storage_mechanism: "biochar"
    expected:
      permanence_assessment:
        expected_duration_years:
          range: [100, 1000]
        reversal_risk: "LOW"

  - test_id: OFF-PERM-006
    name: "No buffer pool for forest"
    input:
      storage_mechanism: "forest"
      buffer_pool: 0
    expected:
      permanence_assessment:
        reversal_risks:
          includes: "No buffer pool"

  - test_id: OFF-PERM-007
    name: "REDD+ with fire risk"
    input:
      project_type: "redd_plus"
      storage_mechanism: "forest"
      region: "tropical"
    expected:
      permanence_assessment:
        reversal_risks:
          includes: "climate-related disturbances"

  - test_id: OFF-PERM-008
    name: "Wood products in buildings"
    input:
      storage_mechanism: "product_wood"
      building_lifetime: 60
    expected:
      permanence_assessment:
        score:
          range: [3.0, 4.0]
        expected_duration_years: 50

  - test_id: OFF-PERM-009
    name: "Avoidance project - no permanence"
    input:
      project_type: "renewable_energy"
      storage_mechanism: "none"
    expected:
      permanence_assessment:
        score: 5.0
        reversal_risk: "N/A"

  - test_id: OFF-PERM-010
    name: "Enhanced weathering"
    input:
      storage_mechanism: "enhanced_weathering"
    expected:
      permanence_assessment:
        score:
          greater_than: 4.5
        expected_duration_years: 10000
```

### 6.4 Double Counting Tests (10 tests)

```yaml
golden_tests_double_counting:
  - test_id: OFF-DC-001
    name: "Properly retired credits"
    input:
      offset_portfolio:
        - serial_numbers: ["VCS-123-2024-001", "VCS-123-2024-002"]
          retirement_status: "retired"
          retirement_date: "2024-06-15"
    expected:
      credit_assessments:
        - double_counting_check:
            retirement_verified: true
            double_use_risk: "LOW"

  - test_id: OFF-DC-002
    name: "Active (not retired) credits"
    input:
      offset_portfolio:
        - retirement_status: "active"
    expected:
      credit_assessments:
        - double_counting_check:
            retirement_verified: false
            double_use_risk: "HIGH"
        - recommendation: "ACCEPT_WITH_CAVEATS"
        - compliance_check:
            warnings:
              includes: "Credits not yet retired"

  - test_id: OFF-DC-003
    name: "Host country without CA"
    input:
      offset_portfolio:
        - project_country: "BR"
          corresponding_adjustment: false
    expected:
      credit_assessments:
        - double_counting_check:
            corresponding_adjustment:
              obtained: false
            double_claiming_risk: "HIGH"

  - test_id: OFF-DC-004
    name: "Credits with corresponding adjustment"
    input:
      offset_portfolio:
        - project_country: "CL"
          corresponding_adjustment: true
          host_country_authorization: true
    expected:
      credit_assessments:
        - double_counting_check:
            corresponding_adjustment:
              obtained: true
            double_claiming_risk: "LOW"

  - test_id: OFF-DC-005
    name: "Cross-registry verification"
    input:
      offset_portfolio:
        - registry: "Verra_VCS"
          project_id: "VCS-1234"
          serial_numbers: ["VCS-1234-2024-001"]
    expected:
      credit_assessments:
        - registry_verification:
            credits_valid: true
            serial_numbers_verified: true

  - test_id: OFF-DC-006
    name: "Duplicate serial numbers detected"
    input:
      offset_portfolio:
        - serial_numbers: ["VCS-123-001", "VCS-123-001"]  # Duplicate
    expected:
      credit_assessments:
        - double_counting_check:
            double_issuance_risk: "HIGH"
            issues:
              includes: "Duplicate serial numbers"

  - test_id: OFF-DC-007
    name: "Previously retired to different beneficiary"
    input:
      offset_portfolio:
        - retirement_beneficiary: "Company A"
      claim_beneficiary: "Company B"
    expected:
      credit_assessments:
        - double_counting_check:
            double_use_risk: "HIGH"

  - test_id: OFF-DC-008
    name: "NDC-included credits"
    input:
      offset_portfolio:
        - project_country: "ID"
          host_country_ndc_inclusion: true
          corresponding_adjustment: false
    expected:
      credit_assessments:
        - double_counting_check:
            double_claiming_risk: "HIGH"

  - test_id: OFF-DC-009
    name: "CORSIA-eligible credits"
    input:
      offset_portfolio:
        - corsia_eligible: true
          corresponding_adjustment: true
    expected:
      credit_assessments:
        - double_counting_check:
            double_claiming_risk: "LOW"

  - test_id: OFF-DC-010
    name: "Article 6 authorized credits"
    input:
      offset_portfolio:
        - article_6_authorized: true
          host_country_letter_of_authorization: true
    expected:
      credit_assessments:
        - double_counting_check:
            corresponding_adjustment:
              obtained: true
```

### 6.5 Claims Matching Tests (5 tests)

```yaml
golden_tests_claims:
  - test_id: OFF-CLAIM-001
    name: "Carbon neutral claim - fully covered"
    input:
      company_emissions:
        total_tco2e: 10000
      qualified_offsets: 12000
      claim_type: "carbon_neutral"
    expected:
      claims_substantiation:
        claim_supportable: true
        coverage_percent: 120
        substantiation_statement:
          includes: "Carbon neutral claim substantiated"

  - test_id: OFF-CLAIM-002
    name: "Carbon neutral claim - insufficient offsets"
    input:
      company_emissions:
        total_tco2e: 10000
      qualified_offsets: 8000
      claim_type: "carbon_neutral"
    expected:
      claims_substantiation:
        claim_supportable: false
        coverage_percent: 80
        substantiation_statement:
          includes: "NOT substantiated"

  - test_id: OFF-CLAIM-003
    name: "Net-zero claim - residual only"
    input:
      company_emissions:
        total_tco2e: 100000
      qualified_offsets: 7000
      claim_type: "net_zero"
    expected:
      claims_substantiation:
        emissions_to_offset: 7000  # 7% residual
        disclosure_requirements:
          includes: "Reduction achieved (must be 90-95%)"

  - test_id: OFF-CLAIM-004
    name: "Climate positive claim"
    input:
      company_emissions:
        total_tco2e: 5000
      qualified_offsets: 6000
      claim_type: "climate_positive"
    expected:
      claims_substantiation:
        claim_supportable: true
        substantiation_statement:
          includes: "1000 tCO2e net removal"

  - test_id: OFF-CLAIM-005
    name: "Beyond value chain mitigation"
    input:
      qualified_offsets: 10000
      claim_type: "beyond_value_chain"
    expected:
      claims_substantiation:
        claim_supportable: true
        disclosure_requirements:
          includes: "Separate from SBTi target claims"
```

---

## 7. Data Dependencies

```yaml
data_dependencies:
  registry_apis:
    verra:
      name: "Verra Registry API"
      url: "https://registry.verra.org"
      authentication: "API Key"
      rate_limit: "100/minute"

    gold_standard:
      name: "Gold Standard Registry API"
      url: "https://registry.goldstandard.org"
      authentication: "API Key"

  quality_frameworks:
    icvcm:
      name: "ICVCM Core Carbon Principles"
      source: "Integrity Council for VCM"
      url: "https://icvcm.org"

    vcmi:
      name: "VCMI Claims Code"
      source: "Voluntary Carbon Markets Integrity"
      url: "https://vcmintegrity.org"

  pricing_data:
    s_and_p:
      name: "S&P Global Carbon Prices"
    ecosystem_marketplace:
      name: "Ecosystem Marketplace Data"

  regulatory_data:
    article_6:
      name: "Paris Agreement Article 6 Database"
      source: "UNFCCC"

    ndc_registry:
      name: "NDC Registry"
      source: "UNFCCC"
```

---

## 8. Implementation Roadmap

```yaml
implementation_roadmap:
  phase_1_registries:
    duration: "Weeks 1-4"
    deliverables:
      - "Verra VCS connector"
      - "Gold Standard connector"
      - "Basic quality scoring"

  phase_2_quality:
    duration: "Weeks 5-10"
    deliverables:
      - "Full quality assessment"
      - "Additionality evaluator"
      - "Permanence evaluator"

  phase_3_compliance:
    duration: "Weeks 11-16"
    deliverables:
      - "Double counting prevention"
      - "Claims matching"
      - "Article 6 tracking"

  phase_4_reporting:
    duration: "Weeks 17-20"
    deliverables:
      - "Substantiation report generator"
      - "Portfolio dashboard"
      - "API documentation"
```

---

## 9. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-ProductManager | Initial specification |

**Approvals:**

- Climate Science Lead: ___________________ Date: _______
- Carbon Markets Lead: ___________________ Date: _______
- Engineering Lead: ___________________ Date: _______
- Product Manager: ___________________ Date: _______

---

**END OF SPECIFICATION**
