# GL-CSRD Agent Specification Upgrade Guide
## Step-by-Step Instructions for AgentSpec V2.0 Compliance

**Version:** 1.0
**Date:** 2025-10-18
**Purpose:** Provide exact YAML sections to add to each CSRD agent spec for V2.0 compliance

---

## Quick Reference: What Each Agent Needs

| Agent | Critical Fixes | Priority |
|-------|---------------|----------|
| **IntakeAgent** | ✅ COMPLETE | - |
| **MaterialityAgent** | ai_integration (AI-powered), testing, deployment, compliance, metadata | CRITICAL |
| **CalculatorAgent** | ai_integration (deterministic), testing, deployment, documentation, compliance, metadata | HIGH |
| **AggregatorAgent** | ai_integration (deterministic), testing, deployment, documentation, compliance, metadata | HIGH |
| **AuditAgent** | ai_integration (deterministic), testing, deployment, documentation, compliance, metadata | HIGH |
| **ReportingAgent** | ai_integration (hybrid), testing, deployment, documentation, compliance, metadata | CRITICAL |

---

## 1. MaterialityAgent - Critical Upgrades

### File: `materiality_agent_spec.yaml`

**CRITICAL: This is an AI-POWERED agent - NOT deterministic!**

### Add After Line 13 (after metadata section):

```yaml
# ----------------------------------------------------------------------------
# SECTION 1: AGENT METADATA (Enhanced)
# ----------------------------------------------------------------------------
agent_metadata:
  agent_id: "csrd_materiality_agent"
  agent_name: "MaterialityAgent"
  display_name: "AI-Powered Materiality Assessment Agent"
  version: "1.0.0"
  domain: "CSRD/ESG Reporting"
  subdomain: "Double Materiality Assessment"
  agent_type: "ai-analyst"
  complexity: "high"
  priority: "critical"
  status: "production"
  deterministic: false  # NOT deterministic!
  llm_usage: true
  zero_hallucination: false  # Uses AI - requires human review
  requires_human_review: true
```

### Add After Line 42 (after overview section):

```yaml
# ----------------------------------------------------------------------------
# SECTION 2: DESCRIPTION (Enhanced)
# ----------------------------------------------------------------------------
description:
  purpose: |
    Conduct AI-powered double materiality assessment per ESRS 1 to identify
    material sustainability topics from both impact and financial perspectives.

  strategic_context:
    global_impact: "Supports CSRD compliance for 50,000+ EU companies requiring double materiality"
    opportunity: "AI-powered materiality cuts assessment time from 2 weeks to 2 hours (90% reduction)"
    market_size: "EU CSRD compliance market estimated at €5B annually"
    technology_maturity: "TRL 7 (System prototype in operational environment)"

  capabilities:
    - "Impact materiality scoring using AI (severity × scope × irremediability)"
    - "Financial materiality scoring using AI (magnitude × likelihood)"
    - "RAG-powered stakeholder consultation analysis"
    - "Automated material topic identification"
    - "Materiality matrix generation"
    - "Natural language rationale generation"
    - "Multi-stakeholder perspective synthesis"

  key_features:
    - "AI-powered analysis using GPT-4o / Claude 3.5 Sonnet"
    - "RAG-based stakeholder consultation synthesis (10,000+ documents)"
    - "<10 minutes processing time for 10 topics"
    - "80% AI automation, 20% mandatory human review"
    - "Complete audit trail and provenance tracking"

  dependencies:
    - agent_id: "intake_agent"
      relationship: "receives_data_from"
      data: "validated_esg_data"
    - agent_id: "calculator_agent"
      relationship: "provides_data_to"
      data: "materiality_matrix"
    - agent_id: "reporting_agent"
      relationship: "provides_data_to"
      data: "materiality_matrix"

  important_warnings:
    - "⚠️ MANDATORY HUMAN REVIEW: All AI assessments must be reviewed by qualified sustainability professionals"
    - "⚠️ NOT ZERO-HALLUCINATION: LLM-based scoring requires expert validation"
    - "⚠️ LEGAL RESPONSIBILITY: Final materiality determination is company's legal responsibility"
    - "⚠️ NOT DETERMINISTIC: Same input may produce slightly different AI outputs"
```

### Add Before Tools Section (around line 300):

```yaml
# ----------------------------------------------------------------------------
# SECTION 4: AI INTEGRATION
# ----------------------------------------------------------------------------
ai_integration:
  enabled: true
  model: "gpt-4o"  # Primary model
  alternative_models:
    - "claude-3-5-sonnet-20241022"
    - "gpt-4-turbo"
  temperature: 0.3  # NOT 0.0 - needs variability for nuanced analysis
  seed: null  # NOT deterministic - no fixed seed
  provenance_tracking: true
  tool_choice: "auto"
  max_iterations: 10
  budget_usd: 5.00  # Higher budget for complex multi-topic analysis
  requires_human_review: true
  hallucination_risk: "MODERATE"
  human_review_checklist:
    - "Impact scores reasonable given company context?"
    - "Financial scores aligned with business realities?"
    - "Stakeholder perspectives accurately reflected?"
    - "Materiality conclusions defensible?"
    - "Rationales clear and evidence-based?"

# ----------------------------------------------------------------------------
# SECTION 5: SUB-AGENTS
# ----------------------------------------------------------------------------
sub_agents:
  enabled: false
  coordination_pattern: "N/A"
  sub_agent_list: []
  rationale: "This is a leaf-level agent with no sub-agents"
```

### Upgrade Tools Section (Replace existing tools section):

```yaml
# ----------------------------------------------------------------------------
# SECTION 3: TOOLS
# ----------------------------------------------------------------------------
tools:
  tools_list:
    - tool_id: "impact_materiality_scorer"
      name: "impact-materiality-scorer"
      deterministic: false  # AI-powered
      category: "ai-analysis"
      description: "AI-powered impact materiality assessment (severity × scope × irremediability)"
      requires_review: true

      parameters:
        type: "object"
        properties:
          topic: {type: "string", description: "ESRS topic (E1, S1, etc.)"}
          company_context: {type: "object", description: "Company profile and context"}
          esg_data: {type: "object", description: "Relevant ESG data"}
          stakeholder_input: {type: "object", description: "Stakeholder perspectives"}
        required: ["topic", "company_context"]

      returns:
        type: "object"
        properties:
          severity_score: {type: "number", description: "Score 0-10"}
          scope_score: {type: "number", description: "Score 0-10"}
          irremediability_score: {type: "number", description: "Score 0-10"}
          impact_score: {type: "number", description: "Combined score"}
          rationale: {type: "string", description: "AI-generated explanation"}
          confidence: {type: "number", description: "AI confidence 0-1"}

      implementation:
        method: "LLM-based assessment with RAG context"
        model: "gpt-4o"
        calculation_method: "Impact = (Severity × Scope × Irremediability) / 100"
        data_source: "ESRS guidance, company context, stakeholder input"
        accuracy: "Requires expert validation - not deterministic"
        validation: "Human review mandatory"
        standards: ["ESRS 1", "EFRAG Implementation Guidance"]

    - tool_id: "financial_materiality_scorer"
      name: "financial-materiality-scorer"
      deterministic: false  # AI-powered
      category: "ai-analysis"
      description: "AI-powered financial materiality assessment (magnitude × likelihood)"
      requires_review: true

      parameters:
        type: "object"
        properties:
          topic: {type: "string", description: "ESRS topic"}
          financial_context: {type: "object", description: "Financial data"}
          risk_factors: {type: "array", description: "Identified risk factors"}
        required: ["topic", "financial_context"]

      returns:
        type: "object"
        properties:
          magnitude_score: {type: "number", description: "Score 0-10"}
          likelihood_score: {type: "number", description: "Score 0-10"}
          financial_score: {type: "number", description: "Combined score"}
          rationale: {type: "string", description: "AI-generated explanation"}
          confidence: {type: "number", description: "AI confidence 0-1"}

      implementation:
        method: "LLM-based financial risk assessment"
        model: "gpt-4o"
        calculation_method: "Financial = (Magnitude × Likelihood) / 10"
        data_source: "Financial statements, market analysis, ESRS guidance"
        accuracy: "Requires expert validation - not deterministic"
        validation: "Human review mandatory, CFO sign-off recommended"
        standards: ["ESRS 1", "TCFD", "IFRS S2"]

    - tool_id: "rag_stakeholder_analyzer"
      name: "rag-stakeholder-analyzer"
      deterministic: false  # AI-powered
      category: "ai-analysis"
      description: "RAG-based stakeholder consultation synthesis"
      requires_review: true

      parameters:
        type: "object"
        properties:
          stakeholder_data: {type: "object", description: "Surveys, interviews, workshops"}
          topic: {type: "string", description: "Topic to analyze"}
        required: ["stakeholder_data"]

      returns:
        type: "object"
        properties:
          key_perspectives: {type: "array", description: "Main stakeholder concerns"}
          consensus_areas: {type: "array", description: "Areas of agreement"}
          divergent_views: {type: "array", description: "Conflicting perspectives"}
          synthesis: {type: "string", description: "AI-generated summary"}

      implementation:
        method: "RAG with vector database retrieval + LLM synthesis"
        vector_db: "data/esrs_guidance_vectors/"
        embedding_model: "text-embedding-3-large"
        llm_model: "gpt-4o"
        calculation_method: "Semantic search + thematic analysis"
        accuracy: "Qualitative - requires human validation"
        validation: "Sustainability team review"
        standards: ["ESRS 1 (Stakeholder Engagement)"]

    - tool_id: "matrix_generator"
      name: "matrix-generator"
      deterministic: true  # This tool IS deterministic
      category: "visualization"
      description: "Generate 2D materiality matrix visualization data"

      parameters:
        type: "object"
        properties:
          topics: {type: "array", description: "Topics with scores"}
          thresholds: {type: "object", description: "Materiality thresholds"}
        required: ["topics"]

      returns:
        type: "object"
        properties:
          matrix_data: {type: "array", description: "Plotting coordinates"}
          quadrants: {type: "object", description: "Quadrant categorization"}
          material_topics: {type: "array", description: "Material topics list"}

      implementation:
        method: "Deterministic 2D plotting algorithm"
        calculation_method: "Cartesian coordinate mapping"
        data_source: "Impact and financial scores"
        accuracy: "100% deterministic"
        validation: "Unit tests with known coordinates"
        standards: ["ESRS 1 (Materiality Matrix)"]
```

### Add Testing Section:

```yaml
# ----------------------------------------------------------------------------
# SECTION 6: TESTING
# ----------------------------------------------------------------------------
testing:
  test_coverage_target: 0.80

  test_categories:
    - category: "unit_tests"
      description: "Test AI tool mocks and deterministic components"
      count: 12
      examples:
        - "test_matrix_generator_deterministic"
        - "test_score_calculation_formula"
        - "test_threshold_application"

    - category: "integration_tests"
      description: "Test full workflow with mocked AI responses"
      count: 6
      examples:
        - "test_full_materiality_assessment_workflow"
        - "test_rag_stakeholder_analysis_integration"
        - "test_multi_topic_assessment"

    - category: "determinism_tests"
      description: "Verify deterministic components only (matrix generation)"
      count: 3
      examples:
        - "test_matrix_generation_reproducible"
        - "test_score_aggregation_deterministic"
      note: "AI components are NOT tested for determinism"

    - category: "boundary_tests"
      description: "Test edge cases and error handling"
      count: 8
      examples:
        - "test_missing_stakeholder_input"
        - "test_low_confidence_scores"
        - "test_conflicting_ai_assessments"
        - "test_borderline_materiality_cases"

  performance_requirements:
    max_latency_ms: 600000  # 10 minutes for 10 topics
    max_cost_usd: 5.00
    accuracy_target: 0.85  # 85% agreement with expert assessments
    human_review_time_savings: 0.90  # 90% time reduction vs manual

  ai_validation:
    - "Compare AI scores vs expert scores (target: 85%+ agreement)"
    - "Track confidence scores (flag <0.7 for priority review)"
    - "Monitor hallucination rate"
    - "Collect human override data"
```

### Add Deployment, Documentation, Compliance, Metadata:

```yaml
# ----------------------------------------------------------------------------
# SECTION 7: DEPLOYMENT
# ----------------------------------------------------------------------------
deployment:
  pack_id: "csrd/materiality_agent"
  pack_version: "1.0.0"

  resource_requirements:
    memory_mb: 4096
    cpu_cores: 2
    gpu_required: false
    storage_mb: 5000  # For vector database

  dependencies:
    python_packages:
      - "openai>=1.0,<2.0"
      - "anthropic>=0.20,<1.0"
      - "chromadb>=0.4,<1.0"  # Vector database
      - "langchain>=0.1,<1.0"
      - "pydantic>=2.0,<3.0"
      - "pandas>=2.0,<3.0"

    greenlang_modules:
      - "greenlang.agents.base"
      - "greenlang.intelligence"
      - "greenlang.core.rag"

  api_endpoints:
    - endpoint: "/api/v1/csrd/materiality/execute"
      method: "POST"
      authentication: "required"
      rate_limit: "10 req/hour"  # Lower due to high cost

  environment_config:
    - name: "OPENAI_API_KEY"
      required: true
      description: "OpenAI API key for GPT-4"
    - name: "ESRS_VECTOR_DB_PATH"
      required: true
      description: "Path to ESRS guidance vector database"

# ----------------------------------------------------------------------------
# SECTION 8: DOCUMENTATION
# ----------------------------------------------------------------------------
documentation:
  readme_path: "docs/agents/materiality_agent/README.md"
  api_docs_path: "docs/agents/materiality_agent/API.md"

  example_use_cases:
    - title: "Manufacturing Company Double Materiality"
      description: "Full materiality assessment for industrial manufacturer"
      input_example:
        company_context:
          name: "ACME Manufacturing"
          sector: "Industrial Equipment"
          revenue: 500000000
      output_summary: "9/10 topics material, E1 climate critical priority"

    - title: "Financial Services Materiality"
      description: "Service sector with different material topics"
      output_summary: "Different profile: S1, G1 high priority"

    - title: "Stakeholder-Driven Assessment"
      description: "Assessment with extensive stakeholder input"
      output_summary: "RAG-enhanced analysis incorporating 500+ stakeholder responses"

  guides:
    - "Getting Started with MaterialityAgent"
    - "Understanding AI-Generated Scores"
    - "Human Review Best Practices"
    - "Interpreting Materiality Matrices"

# ----------------------------------------------------------------------------
# SECTION 9: COMPLIANCE
# ----------------------------------------------------------------------------
compliance:
  zero_secrets: true

  standards:
    - "ESRS 1 (General Requirements)"
    - "ESRS 1 AR 16 (Double Materiality)"
    - "CSRD Directive (EU 2022/2464)"
    - "EFRAG Implementation Guidance"
    - "GRI Universal Standards (Materiality)"

  security:
    secret_scanning: "GL-SecScan validated"
    vulnerability_scanning: "pip-audit clean"
    sbom_generated: true
    code_signing: false

  data_privacy:
    pii_handling: "Stakeholder data may contain PII - handle per GDPR"
    data_retention: "Delete after report completion"
    encryption: "Data in transit: TLS 1.3, At rest: AES-256"

  ai_governance:
    model_registry: "Approved models: GPT-4o, Claude 3.5"
    human_review_mandatory: true
    output_validation: "Sustainability expert review required"
    bias_monitoring: "Track sector-specific scoring patterns"
    explainability: "Rationales required for all scores"

# ----------------------------------------------------------------------------
# SECTION 10: METADATA (Version Control & Changelog)
# ----------------------------------------------------------------------------
metadata:
  created_date: "2025-10-18"
  last_modified: "2025-10-18"
  review_status: "Upgraded to AgentSpec V2.0"

  authors:
    - name: "CSRD Platform Team"
      role: "Development"

  reviewers:
    - name: "Head of AI & Climate Intelligence"
      role: "Technical Review"
      status: "Pending"
    - name: "Chief Sustainability Officer"
      role: "Domain Expert Review"
      status: "Pending"

  change_log:
    - version: "1.0.0"
      date: "2025-10-18"
      changes: "Initial production release with AgentSpec V2.0 compliance, AI-powered materiality assessment"
      author: "CSRD Platform Team"
      breaking_changes: false
```

---

## 2. CalculatorAgent - Critical Upgrades

### File: `calculator_agent_spec.yaml`

**CRITICAL: This is a DETERMINISTIC agent - temperature=0.0, seed=42!**

### Add AI Integration (Disabled):

```yaml
# ----------------------------------------------------------------------------
# SECTION 4: AI INTEGRATION
# ----------------------------------------------------------------------------
ai_integration:
  enabled: false
  model: "N/A - Deterministic calculator agent"
  temperature: 0.0  # MUST be exactly 0.0
  seed: 42  # MUST be exactly 42
  provenance_tracking: true
  tool_choice: "N/A"
  max_iterations: 0
  budget_usd: 0.0
  rationale: "This is a deterministic calculation agent with ZERO AI/LLM usage. All metrics calculated via exact formulas and database lookups only."
  zero_hallucination_guarantee: true
```

### Add Testing Section:

```yaml
# ----------------------------------------------------------------------------
# SECTION 6: TESTING
# ----------------------------------------------------------------------------
testing:
  test_coverage_target: 0.80

  test_categories:
    - category: "unit_tests"
      description: "Test individual calculation formulas"
      count: 50+
      examples:
        - "test_scope1_ghg_calculation"
        - "test_scope2_location_based"
        - "test_emission_factor_lookup"
        - "test_energy_intensity_formula"

    - category: "integration_tests"
      description: "Test full calculation workflows"
      count: 10
      examples:
        - "test_full_e1_climate_calculations"
        - "test_multi_standard_calculation"
        - "test_dependency_resolution"

    - category: "determinism_tests"
      description: "Verify 100% reproducibility"
      count: 10
      examples:
        - "test_same_input_same_output_1000_runs"
        - "test_calculation_reproducibility_across_environments"
        - "test_floating_point_consistency"
      requirement: "MUST pass 100% - bit-perfect reproducibility"

    - category: "boundary_tests"
      description: "Test edge cases and error handling"
      count: 15
      examples:
        - "test_zero_values"
        - "test_missing_emission_factors"
        - "test_division_by_zero_handling"
        - "test_large_values_no_overflow"

  performance_requirements:
    max_latency_ms: 3000  # 3 seconds for all 500+ metrics
    max_cost_usd: 0.0
    accuracy_target: 1.0  # 100% accuracy - deterministic
    throughput: "200+ metrics/sec"
```

### Add Other Missing Sections (Similar Pattern):

```yaml
# SUB-AGENTS
sub_agents:
  enabled: false
  coordination_pattern: "N/A"
  sub_agent_list: []

# DEPLOYMENT
deployment:
  pack_id: "csrd/calculator_agent"
  pack_version: "1.0.0"
  resource_requirements:
    memory_mb: 512
    cpu_cores: 2
    gpu_required: false
  dependencies:
    python_packages:
      - "pandas>=2.0,<3.0"
      - "numpy>=1.24,<2.0"
      - "pydantic>=2.0,<3.0"

# DOCUMENTATION
documentation:
  readme_path: "docs/agents/calculator_agent/README.md"
  api_docs_path: "docs/agents/calculator_agent/API.md"
  example_use_cases:
    - title: "GHG Emissions Calculation"
      description: "Calculate Scope 1, 2, 3 emissions"

# COMPLIANCE
compliance:
  zero_secrets: true
  standards:
    - "GHG Protocol Corporate Standard"
    - "ISO 14064-1:2018"
    - "ESRS E1 Climate Change"
  zero_hallucination_guarantee: true

# METADATA
metadata:
  created_date: "2025-10-18"
  last_modified: "2025-10-18"
  review_status: "Upgraded to AgentSpec V2.0"
  change_log:
    - version: "1.0.0"
      date: "2025-10-18"
      changes: "Initial release with V2.0 compliance"
```

---

## 3. Quick Templates for Remaining Agents

### AggregatorAgent, AuditAgent (Same Pattern as CalculatorAgent)

**Add these sections:**

1. **ai_integration:** enabled: false, temperature: 0.0, seed: 42
2. **sub_agents:** enabled: false
3. **testing:** 80% target, 4 categories, determinism tests critical
4. **deployment:** pack config, dependencies
5. **documentation:** README, API, use cases
6. **compliance:** zero_secrets: true, standards
7. **metadata:** version control, changelog

---

### ReportingAgent (Hybrid: Deterministic + AI)

**SPECIAL CASE: Mixed Determinism**

```yaml
ai_integration:
  enabled: true  # For narrative generation
  hybrid_mode: true
  deterministic_tools:
    - "xbrl_tagger"  # temperature=0.0
    - "esef_packager"  # temperature=0.0
  ai_powered_tools:
    - "narrative_generator"  # temperature=0.5, requires review
  model: "gpt-4o"
  temperature: 0.5  # For narrative generation only
  seed: null
  requires_human_review: true
  provenance_tracking: true
  budget_usd: 2.00
```

---

## Priority Order for Upgrades

1. **MaterialityAgent** (AI-powered, critical for platform)
2. **ReportingAgent** (Hybrid, complex configuration)
3. **CalculatorAgent** (Core calculation engine)
4. **AuditAgent** (Validation engine)
5. **AggregatorAgent** (Data integration)

---

## Validation Checklist

After updating each agent:

```bash
# 1. Check YAML syntax
python -c "import yaml; yaml.safe_load(open('specs/agent_spec.yaml'))"

# 2. Validate against V2.0 schema
python scripts/validate_agent_specs.py specs/agent_spec.yaml

# 3. Expected result
✅ 0 ERRORS
✅ 11/11 sections present
✅ All tools properly formatted
✅ AI config correct for agent type
```

---

## Common Mistakes to Avoid

1. ❌ **Setting temperature=0.0 for AI-powered agents** (Materiality, Reporting narratives)
2. ❌ **Setting deterministic: true for AI-powered tools**
3. ❌ **Forgetting requires_human_review for AI outputs**
4. ❌ **Missing test_coverage_target (must be 0.80)**
5. ❌ **Missing determinism_tests category**
6. ❌ **Not upgrading tools to full V2.0 format**

---

**END OF UPGRADE GUIDE**
