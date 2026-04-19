# GL-008 Green Claims Agent Specification

**Agent ID:** gl-008-green-claims-v1
**Version:** 1.0.0
**Date:** 2025-12-04
**Priority:** P2-MEDIUM
**Deadline:** September 27, 2026 (EU Green Claims Directive transposition)
**Status:** SPECIFICATION COMPLETE

---

## 1. Executive Summary

### 1.1 Regulation Overview

**EU Green Claims Directive (Proposal COM/2023/166)**

The EU Green Claims Directive establishes requirements for businesses making environmental claims to consumers. It aims to combat greenwashing by requiring all environmental claims to be substantiated with robust scientific evidence before being communicated to consumers.

**Key Requirements:**
- All environmental claims must be substantiated with scientific evidence
- Comparative claims must use equivalent methodology and data
- Carbon neutrality claims require disclosure of offsets and residual emissions
- Third-party verification required before claim publication
- Claims must specify lifecycle stage and environmental aspects covered
- Penalty: Up to 4% annual EU turnover, exclusion from public procurement

### 1.2 Agent Purpose

The Green Claims Agent automates the verification and substantiation of environmental marketing claims for EU Green Claims Directive compliance. It provides:

1. **Claim Classification** - Categorize claims into 16 regulated types
2. **Substantiation Scoring** - Evaluate evidence quality against EU requirements
3. **Greenwashing Detection** - Identify vague, misleading, or unsubstantiated claims
4. **PEF/OEF Alignment** - Validate against Product Environmental Footprint methodology
5. **Offset Verification** - Validate carbon neutrality claims with offset quality checks

---

## 2. Regulatory Specification

### 2.1 Applicability Criteria

```yaml
applicability:
  scope: "All B2C environmental claims in EU market"
  jurisdiction: European Union (27 member states)

  covered_claims:
    - Environmental claims (explicit and implicit)
    - Comparative environmental claims
    - Environmental labeling schemes
    - Carbon neutrality/net-zero claims
    - Climate-related claims

  exclusions:
    - Claims regulated by other EU legislation
    - Mandatory labeling (EU Energy Label, etc.)
    - Financial products (separate SFDR regime)

  company_scope:
    - All companies making environmental claims to EU consumers
    - Online and offline marketing
    - Product packaging and advertising

  estimated_impact: "~200,000 companies making green claims in EU"
```

### 2.2 Timeline and Deadlines

```yaml
timeline:
  proposal_date: "2023-03-22"
  expected_adoption: "2024-Q4"
  transposition_deadline: "2026-09-27"
  application_date: "2026-09-27"

  implementation_phases:
    phase_1:
      date: "2026-09-27"
      scope: "All explicit environmental claims"

    phase_2:
      date: "2027-09-27"
      scope: "Environmental labeling schemes"

    phase_3:
      date: "2028-09-27"
      scope: "Full enforcement with penalties"
```

### 2.3 Penalty Structure

```yaml
penalties:
  administrative_fines:
    maximum: "4% of annual EU turnover"
    minimum: "EUR 50,000"

  other_penalties:
    - "Confiscation of revenues from non-compliant products"
    - "Exclusion from public procurement for up to 3 years"
    - "Prohibition of marketing claim for up to 12 months"
    - "Order to publish corrective statement"

  enforcement:
    - "National consumer protection authorities"
    - "Cross-border coordination via CPC Network"
    - "Consumer right of action for damages"
```

---

## 3. Claim Classification System

### 3.1 The 16 Regulated Claim Types

```yaml
claim_types:
  # Carbon/Climate Claims
  type_01_carbon_neutral:
    name: "Carbon Neutral"
    aliases: ["climate neutral", "CO2 neutral", "net zero carbon"]
    requirements:
      - "Full GHG inventory (Scope 1, 2, 3)"
      - "Verified offset credits"
      - "Offset quality assessment"
      - "Residual emissions disclosure"
    evidence_level: "HIGH"
    verification: "MANDATORY_THIRD_PARTY"

  type_02_climate_positive:
    name: "Climate Positive"
    aliases: ["carbon negative", "net negative"]
    requirements:
      - "Removal exceeds emissions"
      - "Permanent removal verification"
      - "Additionality proof"
    evidence_level: "HIGHEST"
    verification: "MANDATORY_THIRD_PARTY"

  type_03_net_zero:
    name: "Net Zero"
    aliases: ["zero emissions", "decarbonized"]
    requirements:
      - "SBTi-aligned pathway"
      - "90-95% absolute reduction"
      - "Neutralization of residual only"
      - "Long-term target (2050 or earlier)"
    evidence_level: "HIGHEST"
    verification: "MANDATORY_THIRD_PARTY"

  type_04_low_carbon:
    name: "Low Carbon"
    aliases: ["reduced carbon", "carbon light"]
    requirements:
      - "Quantified carbon footprint"
      - "Benchmark comparison"
      - "Lifecycle boundary specification"
    evidence_level: "HIGH"
    verification: "RECOMMENDED_THIRD_PARTY"

  # General Environmental Claims
  type_05_eco_friendly:
    name: "Eco-Friendly"
    aliases: ["environmentally friendly", "green", "ecological"]
    requirements:
      - "PEF/OEF study (all 16 impact categories)"
      - "No significant harm to any category"
      - "Lifecycle stage specification"
    evidence_level: "HIGHEST"
    verification: "MANDATORY_THIRD_PARTY"

  type_06_sustainable:
    name: "Sustainable"
    aliases: ["sustainability", "sustainably made"]
    requirements:
      - "Multi-dimensional assessment (E, S, G)"
      - "LCA for environmental dimension"
      - "Social impact assessment"
      - "Circular economy assessment"
    evidence_level: "HIGHEST"
    verification: "MANDATORY_THIRD_PARTY"

  type_07_natural:
    name: "Natural"
    aliases: ["100% natural", "all natural", "nature-based"]
    requirements:
      - "Ingredient origin documentation"
      - "Processing method disclosure"
      - "No synthetic additives (>X%)"
    evidence_level: "MEDIUM"
    verification: "DOCUMENTARY"

  type_08_organic:
    name: "Organic"
    aliases: ["bio", "biologically produced"]
    requirements:
      - "EU Organic certification"
      - "Certification body accreditation"
      - "Traceability to certified sources"
    evidence_level: "HIGH"
    verification: "CERTIFICATION_BASED"

  # Material/Resource Claims
  type_09_recycled:
    name: "Recycled Content"
    aliases: ["made from recycled", "contains recycled"]
    requirements:
      - "Recycled content percentage"
      - "Chain of custody documentation"
      - "Pre/post-consumer breakdown"
      - "ISO 14021 compliance"
    evidence_level: "MEDIUM"
    verification: "CHAIN_OF_CUSTODY"

  type_10_recyclable:
    name: "Recyclable"
    aliases: ["can be recycled", "recyclable packaging"]
    requirements:
      - "Material composition"
      - "Collection infrastructure availability"
      - "Actual recycling rate data"
      - "Geographic scope specification"
    evidence_level: "MEDIUM"
    verification: "INFRASTRUCTURE_EVIDENCE"

  type_11_biodegradable:
    name: "Biodegradable"
    aliases: ["compostable", "breaks down naturally"]
    requirements:
      - "Degradation conditions specification"
      - "Timeframe disclosure"
      - "Certification (EN 13432, ASTM D6400)"
      - "End-of-life pathway availability"
    evidence_level: "HIGH"
    verification: "CERTIFICATION_BASED"

  type_12_plastic_free:
    name: "Plastic-Free"
    aliases: ["no plastic", "zero plastic"]
    requirements:
      - "Complete material composition"
      - "Definition of plastic used"
      - "Packaging included in scope"
    evidence_level: "LOW"
    verification: "DOCUMENTARY"

  # Comparative Claims
  type_13_comparative_better:
    name: "Better Than"
    aliases: ["X% better", "improved vs", "outperforms"]
    requirements:
      - "Same methodology for both products"
      - "Same functional unit"
      - "Same system boundaries"
      - "Statistical significance"
      - "Comparator identification"
    evidence_level: "HIGH"
    verification: "MANDATORY_THIRD_PARTY"

  type_14_comparative_reduced:
    name: "Reduced Impact"
    aliases: ["X% less", "reduced by", "lower than"]
    requirements:
      - "Baseline specification"
      - "Reduction quantification"
      - "Time period specification"
      - "Measurement methodology"
    evidence_level: "MEDIUM"
    verification: "DOCUMENTARY"

  # Future/Commitment Claims
  type_15_commitment:
    name: "Future Commitment"
    aliases: ["committed to", "on track to", "working towards"]
    requirements:
      - "Specific, measurable target"
      - "Timeline with milestones"
      - "Progress tracking mechanism"
      - "Annual progress reporting"
    evidence_level: "MEDIUM"
    verification: "PROGRESS_TRACKING"

  type_16_offset_based:
    name: "Offset-Based Claim"
    aliases: ["offset", "compensated", "neutralized via credits"]
    requirements:
      - "Offset registry identification"
      - "Project type disclosure"
      - "Vintage year disclosure"
      - "Additionality assessment"
      - "Permanence risk disclosure"
      - "Residual emissions first approach"
    evidence_level: "HIGH"
    verification: "MANDATORY_THIRD_PARTY"
```

### 3.2 Claim Detection Patterns

```yaml
claim_detection:
  nlp_patterns:
    carbon_neutral:
      regex_patterns:
        - "carbon[\\s-]?neutral"
        - "climate[\\s-]?neutral"
        - "CO2[\\s-]?neutral"
        - "net[\\s-]?zero[\\s-]?carbon"
      keyword_list:
        - "carbon neutral"
        - "climate neutral"
        - "zero carbon"

    eco_friendly:
      regex_patterns:
        - "eco[\\s-]?friendly"
        - "environment(ally)?[\\s-]?friendly"
        - "green[\\s-]?(product|choice)"
      keyword_list:
        - "eco-friendly"
        - "environmentally friendly"
        - "green"

    sustainable:
      regex_patterns:
        - "sustain(able|ably|ability)"
        - "responsib(le|ly)[\\s-]?sourc"
      keyword_list:
        - "sustainable"
        - "sustainably made"
        - "responsibly sourced"

    comparative:
      regex_patterns:
        - "(\\d+)%?[\\s-]?(better|less|more|reduced|lower)"
        - "compared[\\s-]?to"
        - "vs[\\s.]"
      requires_context: true

  confidence_thresholds:
    high_confidence: 0.90
    medium_confidence: 0.70
    low_confidence: 0.50
    reject_below: 0.30
```

---

## 4. Substantiation Requirements

### 4.1 Evidence Quality Framework

```yaml
evidence_quality:
  dimensions:
    scientific_validity:
      weight: 0.30
      criteria:
        - "Peer-reviewed methodology"
        - "Recognized standards compliance"
        - "Reproducible results"

    data_quality:
      weight: 0.25
      criteria:
        - "Primary data preference"
        - "Temporal representativeness (<3 years)"
        - "Geographic representativeness"
        - "Technological representativeness"

    scope_completeness:
      weight: 0.20
      criteria:
        - "Full lifecycle coverage"
        - "All relevant impact categories"
        - "Materiality assessment"

    independence:
      weight: 0.15
      criteria:
        - "Third-party verification"
        - "Accredited verifier"
        - "No conflict of interest"

    transparency:
      weight: 0.10
      criteria:
        - "Methodology disclosure"
        - "Assumptions documented"
        - "Limitations acknowledged"

  scoring:
    formula: |
      substantiation_score =
        (scientific_validity * 0.30) +
        (data_quality * 0.25) +
        (scope_completeness * 0.20) +
        (independence * 0.15) +
        (transparency * 0.10)

    thresholds:
      approved: ">= 0.80"
      conditional: "0.60 - 0.79"
      rejected: "< 0.60"
```

### 4.2 PEF/OEF Methodology Requirements

```yaml
pef_oef_requirements:
  standard: "EU Product Environmental Footprint (PEF)"
  version: "3.0"

  impact_categories:
    required_16:
      1: { name: "Climate change", unit: "kg CO2 eq", weight: 0.2106 }
      2: { name: "Ozone depletion", unit: "kg CFC-11 eq", weight: 0.0631 }
      3: { name: "Human toxicity, cancer", unit: "CTUh", weight: 0.0213 }
      4: { name: "Human toxicity, non-cancer", unit: "CTUh", weight: 0.0184 }
      5: { name: "Particulate matter", unit: "Disease incidence", weight: 0.0896 }
      6: { name: "Ionising radiation", unit: "kBq U235 eq", weight: 0.0501 }
      7: { name: "Photochemical ozone formation", unit: "kg NMVOC eq", weight: 0.0478 }
      8: { name: "Acidification", unit: "mol H+ eq", weight: 0.0620 }
      9: { name: "Eutrophication, terrestrial", unit: "mol N eq", weight: 0.0371 }
      10: { name: "Eutrophication, freshwater", unit: "kg P eq", weight: 0.0280 }
      11: { name: "Eutrophication, marine", unit: "kg N eq", weight: 0.0296 }
      12: { name: "Ecotoxicity, freshwater", unit: "CTUe", weight: 0.0192 }
      13: { name: "Land use", unit: "Pt", weight: 0.0794 }
      14: { name: "Water use", unit: "m3 world eq", weight: 0.0851 }
      15: { name: "Resource use, minerals and metals", unit: "kg Sb eq", weight: 0.0755 }
      16: { name: "Resource use, fossils", unit: "MJ", weight: 0.0832 }

  normalization_factors:
    source: "JRC EU reference values"
    year: 2023

  weighting:
    method: "Equal weighting or PEF weighting factors"

  lifecycle_stages:
    - "Raw material acquisition"
    - "Manufacturing"
    - "Distribution"
    - "Use phase"
    - "End of life"
```

---

## 5. Greenwashing Detection Rules

### 5.1 Red Flag Patterns

```yaml
greenwashing_detection:
  red_flags:
    # Vagueness
    vague_claims:
      patterns:
        - "Environmentally friendly" without specification
        - "Green" without definition
        - "Eco" prefix without evidence
        - "Natural" without ingredient disclosure
      severity: "HIGH"
      action: "REQUIRE_SPECIFICITY"

    # Hidden Trade-offs
    hidden_tradeoffs:
      patterns:
        - Climate claim without other impact disclosure
        - Recyclable without collection availability
        - Bio-based without land use impact
      severity: "MEDIUM"
      action: "REQUIRE_FULL_ASSESSMENT"

    # No Proof
    no_proof:
      patterns:
        - Claim without supporting data
        - Reference to unavailable study
        - Self-certification without audit
      severity: "CRITICAL"
      action: "REJECT_CLAIM"

    # Irrelevance
    irrelevant_claims:
      patterns:
        - "CFC-free" (already banned)
        - Claiming absence of illegal substances
        - Highlighting common practice as special
      severity: "MEDIUM"
      action: "FLAG_MISLEADING"

    # Lesser of Two Evils
    lesser_evil:
      patterns:
        - "Eco-friendly cigarettes"
        - "Sustainable fast fashion"
        - "Green" for inherently harmful products
      severity: "HIGH"
      action: "FLAG_CONTEXT"

    # Fibbing
    fibbing:
      patterns:
        - Certification claimed without proof
        - Expired certification
        - Misrepresented scope
      severity: "CRITICAL"
      action: "REJECT_CLAIM"

    # Worshiping False Labels
    false_labels:
      patterns:
        - Self-created eco-labels
        - Unaccredited certification schemes
        - Misleading label design
      severity: "HIGH"
      action: "REQUIRE_ACCREDITATION"

  detection_rules:
    rule_001_specificity:
      trigger: "Generic environmental term without quantification"
      check: |
        IF claim_type IN [eco_friendly, sustainable, green] AND
           quantification IS NULL THEN
           FAIL with "Claim lacks specific, measurable criteria"

    rule_002_lifecycle_scope:
      trigger: "Environmental claim without lifecycle stage"
      check: |
        IF lifecycle_stage IS NULL THEN
           WARN with "Claim should specify lifecycle stage (production, use, disposal)"

    rule_003_comparison_baseline:
      trigger: "Comparative claim without baseline"
      check: |
        IF claim_type IN [comparative_better, comparative_reduced] AND
           baseline IS NULL THEN
           FAIL with "Comparative claim requires baseline specification"

    rule_004_offset_quality:
      trigger: "Carbon neutral claim with offsets"
      check: |
        IF claim_type == carbon_neutral AND
           uses_offsets == TRUE THEN
           REQUIRE offset_registry AND vintage_year AND additionality_proof

    rule_005_future_claims:
      trigger: "Future commitment without progress"
      check: |
        IF claim_type == commitment AND
           progress_data IS NULL THEN
           WARN with "Commitment claims require annual progress disclosure"
```

### 5.2 Greenwashing Score Calculation

```yaml
greenwashing_score:
  formula: |
    greenwashing_risk =
      (vagueness_score * 0.25) +
      (hidden_tradeoffs_score * 0.20) +
      (proof_quality_score * 0.25) +
      (relevance_score * 0.15) +
      (label_legitimacy_score * 0.15)

  interpretation:
    low_risk: "0.00 - 0.30"
    medium_risk: "0.31 - 0.60"
    high_risk: "0.61 - 0.80"
    critical_risk: "0.81 - 1.00"

  actions:
    low_risk: "APPROVE with monitoring"
    medium_risk: "REQUIRE additional evidence"
    high_risk: "REJECT without significant remediation"
    critical_risk: "IMMEDIATE rejection and regulatory referral"
```

---

## 6. Agent Architecture

### 6.1 Agent Specification (AgentSpec v2)

```yaml
agent_id: gl-008-green-claims-v1
name: "Green Claims Verification Agent"
version: "1.0.0"
type: claim-substantiation
priority: P2-MEDIUM
deadline: "2026-09-27"

description: |
  Automated environmental claim verification and substantiation agent
  for EU Green Claims Directive compliance. Analyzes marketing claims,
  assesses evidence quality, detects greenwashing, and generates
  verification reports for regulatory compliance.

regulatory_context:
  regulation: "EU Green Claims Directive (COM/2023/166)"
  jurisdiction: European Union
  expected_adoption: "2024-Q4"
  transposition_deadline: "2026-09-27"
  enforcement_agency: "National Consumer Protection Authorities"

inputs:
  claim_submission:
    type: object
    required: true
    properties:
      claim_id:
        type: string
        format: uuid
        description: "Unique identifier for the claim"

      claim_text:
        type: string
        maxLength: 2000
        description: "The environmental claim text to verify"
        example: "Our product is carbon neutral"

      claim_context:
        type: object
        properties:
          product_name:
            type: string
            description: "Name of product/service"
          product_category:
            type: string
            description: "Product category (e.g., apparel, electronics)"
          target_market:
            type: array
            items:
              type: string
              enum: ["EU", "DE", "FR", "IT", "ES", "NL", "BE", "AT", "PL", "SE", "DK", "FI", "IE", "PT", "GR", "CZ", "RO", "HU", "BG", "SK", "HR", "SI", "LT", "LV", "EE", "CY", "LU", "MT"]
          marketing_channel:
            type: string
            enum: ["product_packaging", "website", "advertising", "social_media", "press_release"]
          publication_date:
            type: string
            format: date

      company_profile:
        type: object
        required: true
        properties:
          company_name:
            type: string
          registration_country:
            type: string
          eu_vat_number:
            type: string
            pattern: "^[A-Z]{2}[0-9A-Z]{8,12}$"
          annual_revenue_eur:
            type: number
            minimum: 0

  supporting_evidence:
    type: object
    description: "Evidence supporting the environmental claim"
    properties:
      lca_study:
        type: object
        properties:
          study_id:
            type: string
          methodology:
            type: string
            enum: ["PEF", "ISO_14044", "GHG_Protocol", "Custom"]
          scope:
            type: string
            enum: ["cradle_to_gate", "cradle_to_grave", "gate_to_gate"]
          verifier:
            type: string
          verification_date:
            type: string
            format: date
          study_url:
            type: string
            format: uri

      carbon_footprint:
        type: object
        properties:
          total_kg_co2e:
            type: number
            minimum: 0
          scope_1_kg_co2e:
            type: number
          scope_2_kg_co2e:
            type: number
          scope_3_kg_co2e:
            type: number
          methodology:
            type: string
          boundary:
            type: string
          data_year:
            type: integer

      offset_credits:
        type: array
        items:
          type: object
          properties:
            registry:
              type: string
              enum: ["Verra_VCS", "Gold_Standard", "ACR", "CAR", "Plan_Vivo", "Puro_Earth"]
            project_id:
              type: string
            project_type:
              type: string
              enum: ["forestry", "renewable_energy", "methane_capture", "cookstoves", "DACCS", "BECCS", "biochar", "enhanced_weathering"]
            vintage_year:
              type: integer
              minimum: 2015
            quantity_tco2e:
              type: number
              minimum: 0
            serial_numbers:
              type: array
              items:
                type: string
            retirement_date:
              type: string
              format: date

      certifications:
        type: array
        items:
          type: object
          properties:
            scheme_name:
              type: string
              enum: ["EU_Ecolabel", "FSC", "PEFC", "Fairtrade", "Rainforest_Alliance", "Cradle_to_Cradle", "GOTS", "OEKO_TEX", "Bluesign", "B_Corp", "ISO_14001"]
            certificate_number:
              type: string
            issue_date:
              type: string
              format: date
            expiry_date:
              type: string
              format: date
            scope:
              type: string
            accreditation_body:
              type: string

      recycled_content:
        type: object
        properties:
          total_percentage:
            type: number
            minimum: 0
            maximum: 100
          pre_consumer_percentage:
            type: number
          post_consumer_percentage:
            type: number
          material_type:
            type: string
          chain_of_custody_cert:
            type: string

      third_party_verification:
        type: object
        properties:
          verifier_name:
            type: string
          verifier_accreditation:
            type: string
          verification_standard:
            type: string
            enum: ["ISO_14064", "ISAE_3000", "ISAE_3410", "AA1000AS"]
          verification_date:
            type: string
            format: date
          verification_statement_url:
            type: string
            format: uri

outputs:
  claim_assessment:
    type: object
    properties:
      assessment_id:
        type: string
        format: uuid

      claim_id:
        type: string

      assessment_date:
        type: string
        format: date-time

      claim_classification:
        type: object
        properties:
          claim_type:
            type: string
            enum: ["carbon_neutral", "climate_positive", "net_zero", "low_carbon", "eco_friendly", "sustainable", "natural", "organic", "recycled", "recyclable", "biodegradable", "plastic_free", "comparative_better", "comparative_reduced", "commitment", "offset_based"]
          confidence:
            type: number
            minimum: 0
            maximum: 1
          detected_keywords:
            type: array
            items:
              type: string

      substantiation_assessment:
        type: object
        properties:
          overall_score:
            type: number
            minimum: 0
            maximum: 1
          scientific_validity_score:
            type: number
          data_quality_score:
            type: number
          scope_completeness_score:
            type: number
          independence_score:
            type: number
          transparency_score:
            type: number
          verdict:
            type: string
            enum: ["APPROVED", "CONDITIONAL", "REJECTED"]
          evidence_gaps:
            type: array
            items:
              type: object
              properties:
                gap_type:
                  type: string
                severity:
                  type: string
                  enum: ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
                remediation:
                  type: string

      greenwashing_assessment:
        type: object
        properties:
          risk_score:
            type: number
            minimum: 0
            maximum: 1
          risk_level:
            type: string
            enum: ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
          red_flags:
            type: array
            items:
              type: object
              properties:
                flag_type:
                  type: string
                description:
                  type: string
                severity:
                  type: string
                location:
                  type: string
          recommendations:
            type: array
            items:
              type: string

      compliance_status:
        type: object
        properties:
          gcd_compliant:
            type: boolean
          compliance_gaps:
            type: array
            items:
              type: string
          required_actions:
            type: array
            items:
              type: string
          verification_requirement:
            type: string
            enum: ["MANDATORY_THIRD_PARTY", "RECOMMENDED_THIRD_PARTY", "DOCUMENTARY", "CERTIFICATION_BASED"]

      audit_trail:
        type: object
        properties:
          input_hash:
            type: string
          output_hash:
            type: string
          processing_timestamp:
            type: string
            format: date-time
          agent_version:
            type: string
          methodology_version:
            type: string

  verification_report:
    type: object
    properties:
      report_id:
        type: string
        format: uuid
      report_date:
        type: string
        format: date
      claim_summary:
        type: string
      evidence_summary:
        type: string
      substantiation_conclusion:
        type: string
      recommended_claim_text:
        type: string
      verifier_notes:
        type: string

tools:
  - name: claim_classifier
    type: nlp
    description: "Classify environmental claims into 16 regulated types"
    inputs: ["claim_text", "claim_context"]
    outputs: ["claim_type", "confidence", "keywords"]

  - name: evidence_validator
    type: validator
    description: "Validate supporting evidence quality and completeness"
    inputs: ["supporting_evidence", "claim_type"]
    outputs: ["evidence_score", "gaps", "recommendations"]

  - name: pef_calculator
    type: calculator
    description: "Calculate/validate Product Environmental Footprint"
    inputs: ["lca_data", "impact_categories"]
    outputs: ["pef_score", "normalized_results", "weighted_results"]

  - name: offset_verifier
    type: validator
    description: "Verify carbon offset quality and retirement"
    inputs: ["offset_credits"]
    outputs: ["offset_quality_score", "additionality_assessment", "retirement_status"]

  - name: greenwashing_detector
    type: analyzer
    description: "Detect greenwashing patterns and red flags"
    inputs: ["claim_text", "evidence", "context"]
    outputs: ["risk_score", "red_flags", "recommendations"]

  - name: comparison_validator
    type: validator
    description: "Validate comparative environmental claims"
    inputs: ["claim_data", "baseline_data", "comparator_data"]
    outputs: ["comparison_valid", "methodology_alignment", "statistical_significance"]

  - name: certification_checker
    type: lookup
    description: "Verify certification validity and scope"
    inputs: ["certifications"]
    outputs: ["certification_status", "scope_coverage", "accreditation_status"]

  - name: substantiation_scorer
    type: calculator
    description: "Calculate overall substantiation score"
    inputs: ["evidence_scores", "claim_type"]
    outputs: ["substantiation_score", "verdict", "gaps"]

  - name: report_generator
    type: generator
    description: "Generate verification report for regulators"
    inputs: ["claim_assessment"]
    outputs: ["verification_report", "compliance_certificate"]

  - name: provenance_tracker
    type: utility
    description: "Track SHA-256 provenance for audit trails"
    inputs: ["inputs", "outputs"]
    outputs: ["provenance_hash", "audit_record"]

evaluation:
  golden_tests:
    total_count: 50
    categories:
      carbon_claims: 15
      general_environmental: 10
      comparative_claims: 8
      material_claims: 8
      greenwashing_detection: 9

  accuracy_thresholds:
    claim_classification: 0.95
    substantiation_scoring: 0.90
    greenwashing_detection: 0.92

  benchmarks:
    latency_p95_seconds: 5
    cost_per_analysis_usd: 0.25

  domain_validation:
    validator: "GreenClaimsValidator"
    compliance_checks:
      - "claim_type_mapping"
      - "evidence_completeness"
      - "pef_methodology"
      - "offset_quality"
      - "greenwashing_rules"

certification:
  required_approvals:
    - climate_science_team
    - legal_team
    - marketing_compliance_team

  compliance_checks:
    - gcd_requirements_alignment
    - pef_methodology_compliance
    - offset_registry_integration
    - consumer_protection_rules

  deployment_gates:
    - golden_test_pass_rate: 100%
    - security_scan: "no_critical_issues"
    - performance_benchmark: "meets_targets"
    - documentation: "complete"
```

---

## 7. Calculation Formulas

### 7.1 Substantiation Score Calculation

```python
def calculate_substantiation_score(
    evidence: dict,
    claim_type: str
) -> SubstantiationResult:
    """
    Calculate substantiation score for environmental claim.

    Formula:
    substantiation_score =
        (scientific_validity * 0.30) +
        (data_quality * 0.25) +
        (scope_completeness * 0.20) +
        (independence * 0.15) +
        (transparency * 0.10)

    Source: EU Green Claims Directive requirements mapping
    """

    # Scientific Validity (0-1)
    scientific_validity = calculate_scientific_validity(
        methodology=evidence.get("methodology"),
        peer_review=evidence.get("peer_reviewed", False),
        standard_compliance=evidence.get("standard", None)
    )

    # Data Quality (0-1)
    data_quality = calculate_data_quality(
        primary_data_ratio=evidence.get("primary_data_ratio", 0),
        data_age_years=evidence.get("data_age_years", 5),
        geographic_match=evidence.get("geographic_match", False),
        technology_match=evidence.get("technology_match", False)
    )

    # Scope Completeness (0-1)
    scope_completeness = calculate_scope_completeness(
        lifecycle_stages=evidence.get("lifecycle_stages", []),
        impact_categories=evidence.get("impact_categories", []),
        claim_type=claim_type
    )

    # Independence (0-1)
    independence = calculate_independence(
        third_party_verified=evidence.get("third_party_verified", False),
        verifier_accredited=evidence.get("verifier_accredited", False),
        conflict_of_interest=evidence.get("conflict_of_interest", True)
    )

    # Transparency (0-1)
    transparency = calculate_transparency(
        methodology_disclosed=evidence.get("methodology_disclosed", False),
        assumptions_documented=evidence.get("assumptions_documented", False),
        limitations_acknowledged=evidence.get("limitations_acknowledged", False)
    )

    # Calculate weighted score
    substantiation_score = (
        scientific_validity * 0.30 +
        data_quality * 0.25 +
        scope_completeness * 0.20 +
        independence * 0.15 +
        transparency * 0.10
    )

    # Determine verdict
    if substantiation_score >= 0.80:
        verdict = "APPROVED"
    elif substantiation_score >= 0.60:
        verdict = "CONDITIONAL"
    else:
        verdict = "REJECTED"

    return SubstantiationResult(
        overall_score=substantiation_score,
        scientific_validity_score=scientific_validity,
        data_quality_score=data_quality,
        scope_completeness_score=scope_completeness,
        independence_score=independence,
        transparency_score=transparency,
        verdict=verdict
    )
```

### 7.2 Greenwashing Risk Score

```python
def calculate_greenwashing_risk(
    claim_text: str,
    evidence: dict,
    context: dict
) -> GreenwashingResult:
    """
    Calculate greenwashing risk score.

    Formula:
    greenwashing_risk =
        (vagueness_score * 0.25) +
        (hidden_tradeoffs_score * 0.20) +
        (proof_quality_inverse * 0.25) +
        (irrelevance_score * 0.15) +
        (label_legitimacy_inverse * 0.15)

    Source: TerraChoice Seven Sins of Greenwashing + GCD requirements
    """

    red_flags = []

    # Vagueness Score (0-1, higher = more vague)
    vagueness_score = detect_vagueness(claim_text)
    if vagueness_score > 0.5:
        red_flags.append(RedFlag(
            type="VAGUENESS",
            description="Claim uses vague terms without specificity",
            severity="HIGH"
        ))

    # Hidden Tradeoffs (0-1, higher = more hidden)
    hidden_tradeoffs_score = detect_hidden_tradeoffs(
        claim_focus=extract_claim_focus(claim_text),
        full_impact_data=evidence.get("full_impact_assessment", None)
    )
    if hidden_tradeoffs_score > 0.5:
        red_flags.append(RedFlag(
            type="HIDDEN_TRADEOFFS",
            description="Claim highlights one benefit while hiding other impacts",
            severity="MEDIUM"
        ))

    # Proof Quality (0-1, inverted: 0 = good proof, 1 = no proof)
    proof_quality_inverse = 1.0 - calculate_proof_quality(evidence)
    if proof_quality_inverse > 0.7:
        red_flags.append(RedFlag(
            type="NO_PROOF",
            description="Claim lacks sufficient supporting evidence",
            severity="CRITICAL"
        ))

    # Irrelevance (0-1)
    irrelevance_score = detect_irrelevance(
        claim_text=claim_text,
        product_category=context.get("product_category"),
        regulatory_requirements=get_mandatory_requirements(context)
    )
    if irrelevance_score > 0.5:
        red_flags.append(RedFlag(
            type="IRRELEVANCE",
            description="Claim highlights irrelevant or already mandatory features",
            severity="MEDIUM"
        ))

    # Label Legitimacy (0-1, inverted)
    label_legitimacy_inverse = 1.0 - verify_label_legitimacy(
        certifications=evidence.get("certifications", [])
    )
    if label_legitimacy_inverse > 0.5:
        red_flags.append(RedFlag(
            type="FALSE_LABELS",
            description="Certification or label not from accredited source",
            severity="HIGH"
        ))

    # Calculate overall risk
    greenwashing_risk = (
        vagueness_score * 0.25 +
        hidden_tradeoffs_score * 0.20 +
        proof_quality_inverse * 0.25 +
        irrelevance_score * 0.15 +
        label_legitimacy_inverse * 0.15
    )

    # Determine risk level
    if greenwashing_risk <= 0.30:
        risk_level = "LOW"
    elif greenwashing_risk <= 0.60:
        risk_level = "MEDIUM"
    elif greenwashing_risk <= 0.80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return GreenwashingResult(
        risk_score=greenwashing_risk,
        risk_level=risk_level,
        red_flags=red_flags
    )
```

### 7.3 Carbon Neutrality Claim Validation

```python
def validate_carbon_neutrality_claim(
    carbon_footprint: dict,
    offset_credits: list
) -> CarbonNeutralityResult:
    """
    Validate carbon neutrality claim per GCD requirements.

    Requirements:
    1. Full GHG inventory (Scope 1, 2, 3)
    2. Offsets from verified registries
    3. Offset quality assessment (additionality, permanence)
    4. Residual emissions disclosure
    5. Reduction-first approach demonstrated

    Source: EU Green Claims Directive Article 8
    """

    issues = []

    # 1. Verify full GHG inventory
    total_emissions = (
        carbon_footprint.get("scope_1_kg_co2e", 0) +
        carbon_footprint.get("scope_2_kg_co2e", 0) +
        carbon_footprint.get("scope_3_kg_co2e", 0)
    )

    if carbon_footprint.get("scope_3_kg_co2e") is None:
        issues.append("Scope 3 emissions not included in inventory")

    # 2. Calculate total offsets
    total_offsets = sum(
        credit.get("quantity_tco2e", 0)
        for credit in offset_credits
    )

    # 3. Verify offset registry
    verified_registries = ["Verra_VCS", "Gold_Standard", "ACR", "CAR", "Plan_Vivo", "Puro_Earth"]
    for credit in offset_credits:
        if credit.get("registry") not in verified_registries:
            issues.append(f"Offset registry {credit.get('registry')} not recognized")

    # 4. Check offset quality
    offset_quality_scores = []
    for credit in offset_credits:
        quality_score = assess_offset_quality(credit)
        offset_quality_scores.append(quality_score)
        if quality_score < 0.7:
            issues.append(f"Offset project {credit.get('project_id')} has low quality score")

    # 5. Verify neutrality
    is_neutral = total_offsets >= total_emissions

    # 6. Check reduction-first approach
    historical_emissions = carbon_footprint.get("historical_emissions", [])
    if historical_emissions:
        reduction_trend = calculate_reduction_trend(historical_emissions)
        if reduction_trend <= 0:
            issues.append("No emission reduction trend demonstrated (reduction-first approach)")
    else:
        issues.append("Historical emissions data not provided for reduction-first verification")

    # Calculate residual (what offsets cover)
    residual_emissions = max(0, total_emissions - total_offsets) if not is_neutral else 0

    return CarbonNeutralityResult(
        total_emissions_kg_co2e=total_emissions,
        total_offsets_kg_co2e=total_offsets,
        is_neutral=is_neutral,
        residual_emissions_kg_co2e=residual_emissions,
        offset_quality_average=sum(offset_quality_scores) / len(offset_quality_scores) if offset_quality_scores else 0,
        issues=issues,
        compliant=len([i for i in issues if "not" in i.lower() or "low" in i.lower()]) == 0
    )
```

---

## 8. Golden Test Scenarios

### 8.1 Carbon/Climate Claims Tests (15 tests)

```yaml
golden_tests_carbon_claims:
  # Test 1-5: Carbon Neutral Claims
  - test_id: GC-CARBON-001
    name: "Valid carbon neutral claim with full offsets"
    claim_text: "This product is carbon neutral, certified by Climate Partner"
    input:
      claim_type: "carbon_neutral"
      carbon_footprint:
        scope_1_kg_co2e: 100
        scope_2_kg_co2e: 50
        scope_3_kg_co2e: 850
        total_kg_co2e: 1000
      offset_credits:
        - registry: "Gold_Standard"
          project_id: "GS1234"
          quantity_tco2e: 1.0
          vintage_year: 2024
      third_party_verification:
        verifier_name: "Climate Partner"
        verification_standard: "ISO_14064"
    expected:
      claim_classification:
        claim_type: "carbon_neutral"
        confidence: 0.95
      substantiation_assessment:
        verdict: "APPROVED"
        overall_score: 0.85
      greenwashing_assessment:
        risk_level: "LOW"
      compliance_status:
        gcd_compliant: true

  - test_id: GC-CARBON-002
    name: "Carbon neutral claim without Scope 3"
    claim_text: "We are carbon neutral in our operations"
    input:
      claim_type: "carbon_neutral"
      carbon_footprint:
        scope_1_kg_co2e: 100
        scope_2_kg_co2e: 50
        scope_3_kg_co2e: null  # Missing
      offset_credits:
        - registry: "Verra_VCS"
          quantity_tco2e: 0.15
    expected:
      substantiation_assessment:
        verdict: "CONDITIONAL"
        evidence_gaps:
          - gap_type: "SCOPE_3_MISSING"
            severity: "HIGH"
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "HIDDEN_TRADEOFFS"
      compliance_status:
        gcd_compliant: false
        required_actions:
          - "Include Scope 3 emissions or limit claim scope explicitly"

  - test_id: GC-CARBON-003
    name: "Carbon neutral with unverified offsets"
    claim_text: "Carbon neutral product"
    input:
      carbon_footprint:
        total_kg_co2e: 500
      offset_credits:
        - registry: "Unknown_Registry"
          quantity_tco2e: 0.5
    expected:
      substantiation_assessment:
        verdict: "REJECTED"
      greenwashing_assessment:
        risk_level: "HIGH"
        red_flags:
          - flag_type: "FALSE_LABELS"
      compliance_status:
        gcd_compliant: false

  - test_id: GC-CARBON-004
    name: "Climate positive claim with valid removals"
    claim_text: "Our product is climate positive - we remove more CO2 than we emit"
    input:
      claim_type: "climate_positive"
      carbon_footprint:
        total_kg_co2e: 100
      offset_credits:
        - registry: "Puro_Earth"
          project_type: "biochar"
          quantity_tco2e: 0.15  # 150 kg removed vs 100 kg emitted
    expected:
      claim_classification:
        claim_type: "climate_positive"
      substantiation_assessment:
        verdict: "APPROVED"
      compliance_status:
        gcd_compliant: true

  - test_id: GC-CARBON-005
    name: "Net zero claim without reduction pathway"
    claim_text: "We are net zero"
    input:
      claim_type: "net_zero"
      carbon_footprint:
        total_kg_co2e: 1000
      offset_credits:
        - quantity_tco2e: 1.0
      sbti_target: null
    expected:
      substantiation_assessment:
        verdict: "REJECTED"
        evidence_gaps:
          - gap_type: "REDUCTION_PATHWAY_MISSING"
      greenwashing_assessment:
        risk_level: "HIGH"
      compliance_status:
        gcd_compliant: false
        required_actions:
          - "Net zero claims require SBTi-aligned reduction pathway"

  # Test 6-10: Low Carbon Claims
  - test_id: GC-CARBON-006
    name: "Low carbon claim with benchmark comparison"
    claim_text: "50% lower carbon footprint than industry average"
    input:
      claim_type: "comparative_reduced"
      carbon_footprint:
        total_kg_co2e: 500
      benchmark:
        industry_average_kg_co2e: 1000
        source: "EU PEF category rules"
    expected:
      claim_classification:
        claim_type: "comparative_reduced"
      substantiation_assessment:
        verdict: "APPROVED"
      comparison_valid: true

  - test_id: GC-CARBON-007
    name: "Low carbon claim without benchmark"
    claim_text: "Low carbon product"
    input:
      claim_type: "low_carbon"
      carbon_footprint:
        total_kg_co2e: 500
      benchmark: null
    expected:
      substantiation_assessment:
        verdict: "CONDITIONAL"
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "VAGUENESS"
            description: "Low carbon claim requires benchmark comparison"

  - test_id: GC-CARBON-008
    name: "Reduced emissions claim with valid comparison"
    claim_text: "We reduced our emissions by 30% compared to 2019"
    input:
      claim_type: "comparative_reduced"
      carbon_footprint:
        current_total_kg_co2e: 700
      baseline:
        year: 2019
        total_kg_co2e: 1000
        methodology: "GHG_Protocol"
    expected:
      substantiation_assessment:
        verdict: "APPROVED"
      actual_reduction: 0.30
      comparison_valid: true

  - test_id: GC-CARBON-009
    name: "Offset-based neutrality without emission reduction"
    claim_text: "Carbon neutral through offsetting"
    input:
      claim_type: "offset_based"
      carbon_footprint:
        total_kg_co2e: 1000
        historical_trend: "increasing"
      offset_credits:
        - quantity_tco2e: 1.0
    expected:
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "NO_REDUCTION_FIRST"
      compliance_status:
        required_actions:
          - "Demonstrate emission reduction efforts before offsetting"

  - test_id: GC-CARBON-010
    name: "Future commitment claim with measurable target"
    claim_text: "Committed to net zero by 2040, with 50% reduction by 2030"
    input:
      claim_type: "commitment"
      target:
        target_year: 2040
        interim_target_2030: "50% reduction"
        baseline_year: 2020
      progress:
        current_year: 2025
        reduction_achieved: "15%"
    expected:
      claim_classification:
        claim_type: "commitment"
      substantiation_assessment:
        verdict: "APPROVED"
      compliance_status:
        gcd_compliant: true

  # Test 11-15: Edge Cases
  - test_id: GC-CARBON-011
    name: "Carbon neutral claim for inherently high-emission product"
    claim_text: "Carbon neutral beef"
    input:
      product_category: "meat_beef"
      carbon_footprint:
        total_kg_co2e_per_kg: 25
      offset_credits:
        - quantity_tco2e: 25
    expected:
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "LESSER_EVIL"
            description: "Carbon neutrality for high-impact product may mislead consumers"
      compliance_status:
        required_actions:
          - "Disclose absolute emissions alongside neutrality claim"

  - test_id: GC-CARBON-012
    name: "Expired offset credits"
    claim_text: "Carbon neutral product"
    input:
      carbon_footprint:
        total_kg_co2e: 100
      offset_credits:
        - registry: "Verra_VCS"
          vintage_year: 2015  # Old vintage
          quantity_tco2e: 0.1
    expected:
      substantiation_assessment:
        verdict: "CONDITIONAL"
        evidence_gaps:
          - gap_type: "OLD_VINTAGE"
            severity: "MEDIUM"

  - test_id: GC-CARBON-013
    name: "Double-counted offset credits"
    claim_text: "Climate neutral operations"
    input:
      offset_credits:
        - registry: "Verra_VCS"
          project_id: "VCS123"
          corresponding_adjustment: false
          host_country_ndc: "included"
    expected:
      greenwashing_assessment:
        risk_level: "HIGH"
        red_flags:
          - flag_type: "DOUBLE_COUNTING_RISK"

  - test_id: GC-CARBON-014
    name: "Carbon neutral with only Scope 1"
    claim_text: "Our manufacturing is carbon neutral"
    input:
      scope: "manufacturing_only"
      carbon_footprint:
        scope_1_kg_co2e: 100
        scope_2_kg_co2e: null
        scope_3_kg_co2e: null
    expected:
      substantiation_assessment:
        verdict: "CONDITIONAL"
      greenwashing_assessment:
        risk_level: "MEDIUM"
      compliance_status:
        required_actions:
          - "Explicitly state scope limitation in claim"

  - test_id: GC-CARBON-015
    name: "Valid carbon footprint label"
    claim_text: "Carbon footprint: 2.5 kg CO2e per unit"
    input:
      claim_type: "carbon_footprint_label"
      carbon_footprint:
        total_kg_co2e: 2.5
        methodology: "PEF"
        boundary: "cradle_to_grave"
        verified: true
    expected:
      claim_classification:
        claim_type: "carbon_footprint_label"
      substantiation_assessment:
        verdict: "APPROVED"
      compliance_status:
        gcd_compliant: true
```

### 8.2 General Environmental Claims Tests (10 tests)

```yaml
golden_tests_environmental:
  - test_id: GC-ENV-001
    name: "Eco-friendly claim without LCA"
    claim_text: "Eco-friendly packaging"
    input:
      claim_type: "eco_friendly"
      supporting_evidence:
        lca_study: null
    expected:
      substantiation_assessment:
        verdict: "REJECTED"
      greenwashing_assessment:
        risk_level: "HIGH"
        red_flags:
          - flag_type: "VAGUENESS"
          - flag_type: "NO_PROOF"
      compliance_status:
        gcd_compliant: false
        required_actions:
          - "Eco-friendly claims require full PEF study"

  - test_id: GC-ENV-002
    name: "Sustainable claim with multi-dimensional assessment"
    claim_text: "Sustainably made clothing"
    input:
      claim_type: "sustainable"
      supporting_evidence:
        lca_study:
          methodology: "PEF"
          scope: "cradle_to_grave"
          all_16_categories: true
        social_assessment:
          conducted: true
          standard: "SA8000"
        circular_assessment:
          recyclability: 0.85
          recycled_content: 0.40
    expected:
      substantiation_assessment:
        verdict: "APPROVED"
        overall_score: 0.82
      compliance_status:
        gcd_compliant: true

  - test_id: GC-ENV-003
    name: "Natural claim without ingredient disclosure"
    claim_text: "100% Natural ingredients"
    input:
      claim_type: "natural"
      supporting_evidence:
        ingredient_list: null
        processing_disclosure: null
    expected:
      substantiation_assessment:
        verdict: "REJECTED"
      greenwashing_assessment:
        risk_level: "MEDIUM"

  - test_id: GC-ENV-004
    name: "Organic claim with valid EU certification"
    claim_text: "EU Organic certified"
    input:
      claim_type: "organic"
      certifications:
        - scheme_name: "EU_Organic"
          certificate_number: "DE-OKO-001-12345"
          expiry_date: "2026-12-31"
    expected:
      substantiation_assessment:
        verdict: "APPROVED"
      certification_status: "VALID"
      compliance_status:
        gcd_compliant: true

  - test_id: GC-ENV-005
    name: "Green claim without specification"
    claim_text: "Go green with our products"
    input:
      claim_type: "eco_friendly"
      supporting_evidence: {}
    expected:
      greenwashing_assessment:
        risk_level: "CRITICAL"
        red_flags:
          - flag_type: "VAGUENESS"
          - flag_type: "NO_PROOF"
      compliance_status:
        gcd_compliant: false

  - test_id: GC-ENV-006
    name: "Vegan claim (valid simple claim)"
    claim_text: "100% Vegan product"
    input:
      claim_type: "material_composition"
      supporting_evidence:
        ingredient_audit: true
        no_animal_derived: true
    expected:
      substantiation_assessment:
        verdict: "APPROVED"
      greenwashing_assessment:
        risk_level: "LOW"

  - test_id: GC-ENV-007
    name: "CFC-free claim (irrelevant)"
    claim_text: "CFC-free product"
    input:
      claim_type: "other"
      context:
        cfcs_banned_since: 1996
    expected:
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "IRRELEVANCE"
            description: "CFCs have been banned since 1996"

  - test_id: GC-ENV-008
    name: "Environmentally friendly with partial LCA"
    claim_text: "Environmentally friendly production"
    input:
      claim_type: "eco_friendly"
      lca_study:
        scope: "gate_to_gate"
        impact_categories: 3  # Only 3 of 16
    expected:
      substantiation_assessment:
        verdict: "CONDITIONAL"
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "HIDDEN_TRADEOFFS"

  - test_id: GC-ENV-009
    name: "Planet-friendly claim"
    claim_text: "Planet-friendly choice"
    input:
      claim_type: "eco_friendly"
      supporting_evidence: {}
    expected:
      greenwashing_assessment:
        risk_level: "HIGH"
        red_flags:
          - flag_type: "VAGUENESS"
      compliance_status:
        gcd_compliant: false

  - test_id: GC-ENV-010
    name: "Eco-friendly with hidden trade-off"
    claim_text: "Eco-friendly - low carbon footprint"
    input:
      claim_type: "eco_friendly"
      lca_study:
        climate_change_score: "excellent"
        water_use_score: "poor"
        toxicity_score: "poor"
    expected:
      greenwashing_assessment:
        risk_level: "HIGH"
        red_flags:
          - flag_type: "HIDDEN_TRADEOFFS"
            description: "Climate claim hides poor water and toxicity performance"
```

### 8.3 Comparative Claims Tests (8 tests)

```yaml
golden_tests_comparative:
  - test_id: GC-COMP-001
    name: "Valid comparative claim with same methodology"
    claim_text: "50% lower carbon footprint than Product X"
    input:
      claim_type: "comparative_better"
      own_product:
        carbon_footprint_kg: 500
        methodology: "PEF"
        functional_unit: "1 kg product"
      comparator:
        product_name: "Product X"
        carbon_footprint_kg: 1000
        methodology: "PEF"
        functional_unit: "1 kg product"
    expected:
      comparison_valid: true
      substantiation_assessment:
        verdict: "APPROVED"

  - test_id: GC-COMP-002
    name: "Comparative claim with different methodologies"
    claim_text: "Better for the environment than competitors"
    input:
      claim_type: "comparative_better"
      own_product:
        methodology: "PEF"
      comparator:
        methodology: "ISO_14044"
    expected:
      comparison_valid: false
      greenwashing_assessment:
        risk_level: "HIGH"
      compliance_status:
        required_actions:
          - "Comparative claims require same methodology"

  - test_id: GC-COMP-003
    name: "Comparative claim with different functional units"
    claim_text: "30% less emissions per wash"
    input:
      own_product:
        functional_unit: "1 wash cycle"
      comparator:
        functional_unit: "1 kg detergent"
    expected:
      comparison_valid: false
      greenwashing_assessment:
        risk_level: "HIGH"

  - test_id: GC-COMP-004
    name: "Comparative claim without statistical significance"
    claim_text: "5% better than average"
    input:
      own_product:
        value: 950
        uncertainty: 100
      comparator:
        value: 1000
        uncertainty: 100
    expected:
      statistical_significance: false
      substantiation_assessment:
        verdict: "CONDITIONAL"
        evidence_gaps:
          - gap_type: "STATISTICAL_SIGNIFICANCE"

  - test_id: GC-COMP-005
    name: "Improved version claim"
    claim_text: "New formula - 20% less plastic packaging"
    input:
      claim_type: "comparative_reduced"
      baseline:
        version: "previous"
        plastic_kg: 100
      current:
        plastic_kg: 80
    expected:
      comparison_valid: true
      substantiation_assessment:
        verdict: "APPROVED"

  - test_id: GC-COMP-006
    name: "Cherry-picked comparison"
    claim_text: "Best in class for carbon"
    input:
      own_product:
        climate_change: "best"
        water_use: "worst"
        toxicity: "average"
      comparison_scope: "climate_only"
    expected:
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "CHERRY_PICKING"

  - test_id: GC-COMP-007
    name: "Comparison to outdated baseline"
    claim_text: "60% better than 2010 products"
    input:
      baseline_year: 2010
      current_year: 2025
      industry_progress_since_baseline: "significant"
    expected:
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "OUTDATED_BASELINE"

  - test_id: GC-COMP-008
    name: "Valid industry benchmark comparison"
    claim_text: "30% below EU PEF benchmark"
    input:
      claim_type: "comparative_better"
      benchmark:
        source: "EU_PEF_Category_Rules"
        year: 2024
        value: 1000
      own_product:
        value: 700
    expected:
      comparison_valid: true
      substantiation_assessment:
        verdict: "APPROVED"
```

### 8.4 Material/Resource Claims Tests (8 tests)

```yaml
golden_tests_material:
  - test_id: GC-MAT-001
    name: "Valid recycled content claim"
    claim_text: "Made with 80% recycled plastic"
    input:
      claim_type: "recycled"
      recycled_content:
        total_percentage: 80
        post_consumer_percentage: 60
        pre_consumer_percentage: 20
        chain_of_custody_cert: "SCS-COC-12345"
    expected:
      substantiation_assessment:
        verdict: "APPROVED"
      compliance_status:
        gcd_compliant: true

  - test_id: GC-MAT-002
    name: "Recyclable claim without infrastructure"
    claim_text: "100% Recyclable packaging"
    input:
      claim_type: "recyclable"
      material: "PLA bioplastic"
      recycling_infrastructure:
        eu_average_collection_rate: 0.05
        actual_recycling_rate: 0.02
    expected:
      greenwashing_assessment:
        risk_level: "HIGH"
        red_flags:
          - flag_type: "INFRASTRUCTURE_GAP"
            description: "Material technically recyclable but infrastructure lacking"
      compliance_status:
        required_actions:
          - "Specify geographic availability of recycling"

  - test_id: GC-MAT-003
    name: "Biodegradable claim with conditions"
    claim_text: "Biodegradable packaging"
    input:
      claim_type: "biodegradable"
      degradation:
        conditions: "industrial_composting"
        timeframe_days: 180
        certification: "EN_13432"
    expected:
      substantiation_assessment:
        verdict: "CONDITIONAL"
      compliance_status:
        required_actions:
          - "Specify 'industrially compostable' instead of 'biodegradable'"
          - "Disclose that home composting not suitable"

  - test_id: GC-MAT-004
    name: "Plastic-free claim"
    claim_text: "Plastic-free product"
    input:
      claim_type: "plastic_free"
      material_composition:
        plastics: 0
        paper: 0.7
        metal: 0.3
    expected:
      substantiation_assessment:
        verdict: "APPROVED"
      greenwashing_assessment:
        risk_level: "LOW"

  - test_id: GC-MAT-005
    name: "Recycled claim without chain of custody"
    claim_text: "Contains recycled materials"
    input:
      claim_type: "recycled"
      recycled_content:
        total_percentage: 30
        chain_of_custody_cert: null
    expected:
      substantiation_assessment:
        verdict: "CONDITIONAL"
      compliance_status:
        required_actions:
          - "Provide chain of custody certification"

  - test_id: GC-MAT-006
    name: "Compostable claim - home vs industrial"
    claim_text: "Compostable at home"
    input:
      claim_type: "biodegradable"
      certification: "OK_Compost_HOME"
      degradation:
        conditions: "home_composting"
        timeframe_months: 6
    expected:
      substantiation_assessment:
        verdict: "APPROVED"
      certification_status: "VALID"

  - test_id: GC-MAT-007
    name: "Downcycled material claim"
    claim_text: "Made from recycled ocean plastic"
    input:
      claim_type: "recycled"
      material_source: "ocean_plastic"
      traceability: "verified"
      downcycling_factor: 0.3
    expected:
      substantiation_assessment:
        verdict: "APPROVED"
      greenwashing_assessment:
        risk_level: "LOW"

  - test_id: GC-MAT-008
    name: "Bio-based claim without land use disclosure"
    claim_text: "Made from plant-based materials"
    input:
      claim_type: "bio_based"
      bio_based_percentage: 100
      land_use_impact: null
    expected:
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "HIDDEN_TRADEOFFS"
            description: "Bio-based claim without land use impact disclosure"
```

### 8.5 Greenwashing Detection Tests (9 tests)

```yaml
golden_tests_greenwashing:
  - test_id: GC-GW-001
    name: "Classic vague green claim"
    claim_text: "Environmentally conscious choice"
    input:
      supporting_evidence: {}
    expected:
      greenwashing_assessment:
        risk_level: "CRITICAL"
        red_flags:
          - flag_type: "VAGUENESS"
          - flag_type: "NO_PROOF"
        recommendations:
          - "Replace with specific, measurable environmental benefit"
          - "Provide supporting evidence for any claim made"

  - test_id: GC-GW-002
    name: "Misleading imagery without claim"
    claim_text: "[Green leaf logo] Our Product"
    input:
      imagery_analysis:
        green_imagery: true
        nature_imagery: true
        explicit_claim: false
    expected:
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "MISLEADING_IMAGERY"
            description: "Green/nature imagery implies environmental benefit without substantiation"

  - test_id: GC-GW-003
    name: "Selective disclosure"
    claim_text: "Carbon neutral production process"
    input:
      scope: "production_only"
      full_lifecycle:
        production_pct: 0.20
        raw_materials_pct: 0.50
        use_phase_pct: 0.20
        end_of_life_pct: 0.10
    expected:
      greenwashing_assessment:
        risk_level: "HIGH"
        red_flags:
          - flag_type: "SELECTIVE_DISCLOSURE"
            description: "Claim covers only 20% of lifecycle impact"

  - test_id: GC-GW-004
    name: "Self-created eco-label"
    claim_text: "EcoSmart Certified [proprietary logo]"
    input:
      certification:
        scheme_name: "EcoSmart"
        self_created: true
        third_party_audit: false
        accreditation: null
    expected:
      greenwashing_assessment:
        risk_level: "CRITICAL"
        red_flags:
          - flag_type: "FALSE_LABELS"
            description: "Self-created certification without third-party accreditation"

  - test_id: GC-GW-005
    name: "Aspirational claim as fact"
    claim_text: "Sustainable fashion for a better future"
    input:
      current_state:
        sustainable_materials_pct: 0.10
        living_wage_suppliers_pct: 0.05
      targets:
        sustainable_materials_2030: 0.50
    expected:
      greenwashing_assessment:
        risk_level: "HIGH"
        red_flags:
          - flag_type: "ASPIRATIONAL_AS_FACT"
            description: "Current performance does not support sustainability claim"

  - test_id: GC-GW-006
    name: "Best available excuse"
    claim_text: "Most sustainable option in our category"
    input:
      category: "fast_fashion"
      own_impact: "high"
      category_average_impact: "very_high"
    expected:
      greenwashing_assessment:
        risk_level: "MEDIUM"
        red_flags:
          - flag_type: "LESSER_EVIL"
            description: "Being best in a harmful category may mislead consumers"

  - test_id: GC-GW-007
    name: "Expired certification"
    claim_text: "ISO 14001 Certified"
    input:
      certification:
        scheme_name: "ISO_14001"
        expiry_date: "2023-06-30"
        current_date: "2025-12-04"
    expected:
      greenwashing_assessment:
        risk_level: "CRITICAL"
        red_flags:
          - flag_type: "EXPIRED_CERTIFICATION"
      compliance_status:
        gcd_compliant: false

  - test_id: GC-GW-008
    name: "Mathematically impossible claim"
    claim_text: "110% recycled content"
    input:
      claim_type: "recycled"
      recycled_content:
        claimed_percentage: 110
    expected:
      greenwashing_assessment:
        risk_level: "CRITICAL"
        red_flags:
          - flag_type: "IMPOSSIBLE_CLAIM"
            description: "Recycled content cannot exceed 100%"

  - test_id: GC-GW-009
    name: "Misleading comparison scope"
    claim_text: "Twice as sustainable as our competitors"
    input:
      comparison:
        own_assessment: "partial_lca"
        competitor_assessment: "full_lca"
        methodology_match: false
    expected:
      greenwashing_assessment:
        risk_level: "HIGH"
        red_flags:
          - flag_type: "UNFAIR_COMPARISON"
          - flag_type: "VAGUENESS"
```

---

## 9. Data Dependencies

### 9.1 Required Data Sources

```yaml
data_dependencies:
  pef_database:
    name: "EU PEF Database"
    source: "European Commission JRC"
    url: "https://ec.europa.eu/jrc/en/publication/european-platform-life-cycle-assessment"
    content:
      - "PEF impact category characterization factors"
      - "PEF normalization factors"
      - "PEF weighting factors"
      - "PEFCR (Product Environmental Footprint Category Rules)"
    update_frequency: "Per EU publication"

  lci_databases:
    ecoinvent:
      name: "ecoinvent"
      version: "3.10"
      url: "https://ecoinvent.org/"
      content: "Life Cycle Inventory data"
      license: "Commercial"

    elcd:
      name: "European Reference Life Cycle Database"
      url: "https://eplca.jrc.ec.europa.eu/ELCD3/"
      content: "EU reference LCI data"
      license: "Free"

  offset_registries:
    verra:
      name: "Verra VCS Registry"
      url: "https://registry.verra.org/"
      api: "REST API"
      content: "VCS project and credit data"

    gold_standard:
      name: "Gold Standard Registry"
      url: "https://registry.goldstandard.org/"
      api: "REST API"
      content: "Gold Standard project and credit data"

  certification_databases:
    eu_ecolabel:
      name: "EU Ecolabel Product Catalogue"
      url: "https://ec.europa.eu/ecat/"
      content: "EU Ecolabel certified products"

    fsc:
      name: "FSC Certificate Database"
      url: "https://info.fsc.org/"
      content: "FSC certified entities"

  benchmark_data:
    pef_benchmarks:
      name: "PEF Category Benchmarks"
      source: "EU Commission"
      content: "Product category environmental benchmarks"

  regulatory_updates:
    eur_lex:
      name: "EUR-Lex"
      url: "https://eur-lex.europa.eu/"
      content: "EU regulatory updates"
```

---

## 10. Implementation Roadmap

### 10.1 Development Phases

```yaml
implementation_roadmap:
  phase_1_foundation:
    duration: "Weeks 1-4"
    deliverables:
      - "Claim classification NLP model (16 types)"
      - "Evidence validation framework"
      - "Basic greenwashing detection rules"
      - "Input/output schema validation"
    milestones:
      - "Claim classifier >90% accuracy"
      - "20 golden tests passing"

  phase_2_substantiation:
    duration: "Weeks 5-8"
    deliverables:
      - "PEF calculation engine"
      - "Offset quality assessment"
      - "Substantiation scoring algorithm"
      - "Certification verification"
    milestones:
      - "PEF calculator validated"
      - "35 golden tests passing"

  phase_3_greenwashing:
    duration: "Weeks 9-12"
    deliverables:
      - "Advanced greenwashing detection (all red flags)"
      - "Comparative claim validation"
      - "Risk scoring algorithm"
      - "Recommendations engine"
    milestones:
      - "Greenwashing detection >90% accuracy"
      - "45 golden tests passing"

  phase_4_reporting:
    duration: "Weeks 13-16"
    deliverables:
      - "Verification report generator"
      - "Compliance certificate generator"
      - "Audit trail implementation"
      - "API documentation"
    milestones:
      - "50 golden tests passing (100%)"
      - "API documentation complete"

  phase_5_deployment:
    duration: "Weeks 17-20"
    deliverables:
      - "Production deployment"
      - "Performance optimization"
      - "Beta customer onboarding"
      - "Monitoring and alerting"
    milestones:
      - "10 beta customers"
      - "99.9% uptime achieved"
```

### 10.2 Success Metrics

```yaml
success_metrics:
  technical:
    claim_classification_accuracy: ">95%"
    substantiation_scoring_accuracy: ">90%"
    greenwashing_detection_accuracy: ">92%"
    false_positive_rate: "<5%"
    false_negative_rate: "<3%"
    api_latency_p95: "<5 seconds"

  business:
    customer_adoption_year_1: 100
    claims_verified_monthly: 10000
    customer_satisfaction_nps: ">50"
    compliance_rate: ">95%"

  regulatory:
    gcd_requirement_coverage: "100%"
    audit_trail_completeness: "100%"
    verification_accuracy: ">99%"
```

---

## 11. Risk Analysis

```yaml
risks:
  regulatory_uncertainty:
    description: "GCD final text may differ from proposal"
    likelihood: "MEDIUM"
    impact: "HIGH"
    mitigation: "Monitor legislative process, modular design for updates"

  nlp_accuracy:
    description: "Claim classification may miss edge cases"
    likelihood: "MEDIUM"
    impact: "MEDIUM"
    mitigation: "Continuous model training, human review for edge cases"

  offset_registry_access:
    description: "Registry APIs may have rate limits or changes"
    likelihood: "MEDIUM"
    impact: "MEDIUM"
    mitigation: "Multi-registry support, caching, fallback mechanisms"

  pef_data_availability:
    description: "Not all products have PEF category rules"
    likelihood: "HIGH"
    impact: "MEDIUM"
    mitigation: "Support ISO 14044 as alternative, guidance for custom LCA"

  greenwashing_evasion:
    description: "Companies may craft claims to evade detection"
    likelihood: "HIGH"
    impact: "MEDIUM"
    mitigation: "Regular rule updates, adversarial testing, human review"
```

---

## 12. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-ProductManager | Initial specification |

**Approvals:**

- Climate Science Lead: ___________________ Date: _______
- Legal/Compliance Lead: ___________________ Date: _______
- Engineering Lead: ___________________ Date: _______
- Product Manager: ___________________ Date: _______

---

**END OF SPECIFICATION**
