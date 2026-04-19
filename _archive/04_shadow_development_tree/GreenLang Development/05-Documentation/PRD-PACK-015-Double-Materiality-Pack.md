# PRD-PACK-015: Double Materiality Assessment Pack

**Pack ID:** PACK-015-double-materiality
**Category:** EU Compliance / CSRD
**Tier:** Standalone (Cross-Sector)
**Version:** 1.0.0
**Status:** Draft
**Author:** GreenLang Product Team
**Date:** 2026-03-16

---

## 1. Executive Summary

PACK-015 is a **standalone Double Materiality Assessment (DMA) Solution Pack** that provides deterministic, auditable, zero-hallucination double materiality assessment capability per ESRS 1 (European Sustainability Reporting Standards, Chapter 3). While GreenLang's existing GL-CSRD-APP includes a basic `materiality_agent.py` that uses LLM-based analysis for materiality scoring, PACK-015 replaces this with a fully deterministic engine-based approach where every score is traceable to a defined methodology, quantified inputs, and auditable calculation logic -- no LLM inference is used in any scoring or classification decision.

**Problem Statement:**

Double materiality assessment is the foundational step of CSRD compliance. Every company subject to CSRD must conduct a DMA before it can determine which ESRS topics are material and which disclosure requirements apply. The assessment must evaluate each sustainability matter from two perspectives simultaneously:

- **Impact materiality** (inside-out): How does the company's business model and operations impact people and the environment? Scored on severity (scale x scope x irremediability) and likelihood for potential impacts.
- **Financial materiality** (outside-in): How do sustainability matters create financial risks and opportunities for the company? Scored on financial magnitude, likelihood, and time horizon.

Current approaches suffer from three critical failures:

1. **Subjectivity**: Most DMA tools rely on qualitative scoring workshops where participants assign 1-5 ratings based on gut feel, producing inconsistent, non-reproducible results.
2. **Missing Provenance**: Auditors cannot trace how a topic scored 4.2 on impact severity -- there is no calculation chain from raw data to final score.
3. **Static Assessment**: DMAs are conducted once and filed, rather than maintained as living assessments that respond to regulatory changes, stakeholder input evolution, and business model shifts.

PACK-015 solves all three by implementing 8 deterministic engines that produce bit-perfect, SHA-256-hashed, fully auditable DMA outputs. Every materiality score is computed from quantified inputs through transparent formulas, and every decision (material/not material) is traceable to specific thresholds, stakeholder inputs, and IRO assessments.

**Key Differentiators vs GL-CSRD-APP materiality_agent.py:**

| Dimension | GL-CSRD-APP materiality_agent.py | PACK-015 Double Materiality Pack |
|-----------|----------------------------------|----------------------------------|
| Scoring method | LLM-based qualitative analysis | Deterministic formula-based scoring |
| Reproducibility | Non-deterministic (LLM varies) | Bit-perfect (same input = same output) |
| Auditability | Limited provenance | SHA-256 hashed calculation chain |
| Stakeholder management | Not included | Full stakeholder lifecycle |
| IRO identification | Basic topic listing | Structured IRO register per ESRS 1 ss28-39 |
| ESRS mapping | Simple topic-to-standard mapping | Full disclosure requirement mapping with gap analysis |
| Threshold management | Hardcoded thresholds | Configurable per industry, size, and methodology |
| Update capability | Full re-run required | Delta-based incremental update |
| Sector presets | None | 6 presets (large enterprise, mid-market, SME, financial services, manufacturing, multi-sector) |

**Target Users**: All companies subject to CSRD reporting under ESRS Set 1 (Omnibus I threshold: >1,000 employees AND >EUR 450M net turnover), sustainability consultants conducting DMAs for clients, and auditors validating DMA methodology and outcomes. PACK-015 is sector-agnostic and works across all NACE divisions as a standalone module or as the DMA foundation for sector-specific packs (PACK-012 through PACK-014).

---

## 2. Regulatory Scope

### 2.1 Primary Regulatory Requirements

| Regulation / Standard | Reference | Effective | DMA Relevance | Penalty |
|----------------------|-----------|-----------|---------------|---------|
| CSRD | Directive (EU) 2022/2464 | FY2025/2026+ | Mandates DMA as foundation for all ESRS reporting | Market access |
| ESRS 1 | Delegated Reg (EU) 2023/2772, Chapter 3 | With CSRD | Core DMA methodology: impact materiality (ss43-48), financial materiality (ss49-51), stakeholder engagement (ss22-23), IRO identification (ss28-39) | N/A (part of CSRD) |
| ESRS 2 | Delegated Reg (EU) 2023/2772 | With CSRD | IRO-1 (process description), IRO-2 (ESRS topics list), SBM-3 (material impacts, risks, opportunities) | N/A (part of CSRD) |
| Omnibus I | Directive (EU) 2026/470 | 2026 | Revised thresholds (>1,000 employees AND >EUR 450M turnover), 61% datapoint reduction applies after DMA determines material topics | N/A (threshold change) |
| EFRAG IG-1 | EFRAG Implementation Guidance | 2024 | Materiality Assessment Implementation Guidance: detailed methodology, worked examples, sector considerations | Non-binding guidance |
| EFRAG IG-2 | EFRAG Implementation Guidance | 2024 | Value Chain Implementation Guidance: value chain boundary for materiality assessment | Non-binding guidance |
| EFRAG IG-3 | EFRAG Implementation Guidance | 2024 | ESRS Datapoints: full list of ESRS datapoints for disclosure post-DMA | Non-binding guidance |

### 2.2 ESRS 1 Chapter 3 Detailed Requirements

#### 2.2.1 Impact Materiality (ESRS 1 ss43-48)

A sustainability matter is material from the impact perspective when it pertains to the undertaking's material actual or potential, positive or negative impacts on people or the environment over the short, medium, or long term.

**Actual Negative Impacts** -- scored on severity:
- **Scale**: How grave is the impact? (ESRS 1 ss44)
- **Scope**: How widespread is the impact? (ESRS 1 ss44)
- **Irremediability**: How hard is it to counteract or make good the resulting harm? (ESRS 1 ss44)

**Potential Negative Impacts** -- scored on severity AND likelihood:
- Severity (scale x scope x irremediability) as above
- **Likelihood**: How likely is the impact to occur? (ESRS 1 ss45)

**Positive Impacts** -- scored on:
- Scale and scope of the positive impact (ESRS 1 ss46)
- Likelihood for potential positive impacts

**Human Rights Impacts**: For human rights impacts, severity takes precedence over likelihood per ESRS 1 ss47. Even low-likelihood human rights impacts can be material if severity is high.

#### 2.2.2 Financial Materiality (ESRS 1 ss49-51)

A sustainability matter is material from the financial perspective if it triggers or may trigger material financial effects on the undertaking.

**Financial Effects** -- scored on:
- **Magnitude**: Size of financial effect (absolute or relative to revenue/assets/equity)
- **Likelihood**: Probability of occurrence
- **Time Horizon**: Short-term (<1 year), medium-term (1-5 years), long-term (>5 years)

Financial effects include effects on cash flows, development, financial position, financial performance, access to finance, or cost of capital.

#### 2.2.3 Stakeholder Engagement (ESRS 1 ss22-23)

The undertaking shall consider affected stakeholders in its materiality assessment process:
- Identify affected stakeholders for each sustainability matter
- Engage with affected stakeholders to understand their perspectives
- Consider the perspectives of users of sustainability statements
- Document the engagement process, outcomes, and how they influenced the DMA

#### 2.2.4 IRO Identification (ESRS 1 ss28-39)

The undertaking shall identify its impacts, risks, and opportunities (IROs):
- Map the value chain to identify where IROs occur (ss29)
- Consider the full list of sustainability matters in ESRS 1 Appendix A (ss30)
- Screen sustainability matters by relevance to the undertaking's business model and value chain (ss31-33)
- Classify each relevant sustainability matter as impact, risk, opportunity, or combination (ss34-36)
- Assess each IRO for materiality (ss37-39)

### 2.3 ESRS Topics Covered in DMA

| ESRS | Topic | Sub-Topics for DMA |
|------|-------|--------------------|
| E1 | Climate Change | Climate change mitigation, climate change adaptation, energy |
| E2 | Pollution | Pollution of air, water, soil; substances of concern; substances of very high concern |
| E3 | Water and Marine Resources | Water, marine resources |
| E4 | Biodiversity and Ecosystems | Direct impact drivers, impacts on state of species/ecosystems, impacts/dependencies on ecosystem services |
| E5 | Resource Use and Circular Economy | Resource inflows, resource outflows, waste |
| S1 | Own Workforce | Working conditions, equal treatment, other work-related rights |
| S2 | Workers in the Value Chain | Working conditions, equal treatment, other work-related rights |
| S3 | Affected Communities | Communities' economic/social/cultural rights, civil/political rights, rights of indigenous peoples |
| S4 | Consumers and End-Users | Information-related impacts, personal safety, social inclusion |
| G1 | Business Conduct | Corporate culture, whistleblowing, animal welfare, political engagement, supplier relationships, corruption/bribery |

---

## 3. Architecture

### 3.1 Pack Structure

```
PACK-015-double-materiality/
+-- __init__.py
+-- pack.yaml
+-- config/
|   +-- __init__.py
|   +-- pack_config.py
|   +-- presets/
|   |   +-- __init__.py
|   |   +-- large_enterprise.yaml        # Full DMA with all features enabled
|   |   +-- mid_market.yaml              # Standard DMA for mid-sized companies
|   |   +-- sme.yaml                     # Simplified DMA with reduced scope
|   |   +-- financial_services.yaml      # Financial sector specific (SFDR overlay)
|   |   +-- manufacturing.yaml           # Manufacturing sector specific (EU ETS/CBAM overlay)
|   |   +-- multi_sector.yaml            # Conglomerate with multiple NACE codes
|   +-- demo/
|       +-- __init__.py
|       +-- demo_config.yaml
+-- engines/
|   +-- __init__.py
|   +-- impact_materiality_engine.py       # Engine 1: Impact materiality scoring
|   +-- financial_materiality_engine.py    # Engine 2: Financial materiality scoring
|   +-- stakeholder_engagement_engine.py   # Engine 3: Stakeholder lifecycle management
|   +-- iro_identification_engine.py       # Engine 4: IRO identification and classification
|   +-- materiality_matrix_engine.py       # Engine 5: Double materiality matrix generation
|   +-- esrs_topic_mapping_engine.py       # Engine 6: ESRS disclosure requirement mapping
|   +-- threshold_scoring_engine.py        # Engine 7: Configurable thresholds and scoring
|   +-- dma_report_engine.py              # Engine 8: DMA report assembly
+-- workflows/
|   +-- __init__.py
|   +-- impact_assessment_workflow.py       # Workflow 1: Impact materiality assessment
|   +-- financial_assessment_workflow.py    # Workflow 2: Financial materiality assessment
|   +-- stakeholder_engagement_workflow.py  # Workflow 3: Stakeholder engagement lifecycle
|   +-- iro_identification_workflow.py      # Workflow 4: IRO identification process
|   +-- materiality_matrix_workflow.py      # Workflow 5: Matrix generation
|   +-- esrs_mapping_workflow.py            # Workflow 6: ESRS topic mapping
|   +-- full_dma_workflow.py               # Workflow 7: Full end-to-end DMA
|   +-- dma_update_workflow.py             # Workflow 8: Incremental DMA update
+-- templates/
|   +-- __init__.py
|   +-- impact_materiality_report.py        # Template 1: Impact materiality report
|   +-- financial_materiality_report.py     # Template 2: Financial materiality report
|   +-- stakeholder_engagement_report.py    # Template 3: Stakeholder engagement report
|   +-- materiality_matrix_report.py        # Template 4: Visual materiality matrix
|   +-- iro_register_report.py             # Template 5: IRO register
|   +-- esrs_disclosure_map.py             # Template 6: ESRS disclosure map
|   +-- dma_executive_summary.py           # Template 7: DMA executive summary
|   +-- dma_audit_report.py               # Template 8: DMA audit report
+-- integrations/
|   +-- __init__.py
|   +-- pack_orchestrator.py               # Master orchestrator (9-phase DMA pipeline)
|   +-- csrd_pack_bridge.py                # Bridge to PACK-001/002/003 (CSRD packs)
|   +-- mrv_materiality_bridge.py          # Bridge to MRV agents for emissions context
|   +-- data_materiality_bridge.py         # Bridge to DATA agents for external data
|   +-- sector_classification_bridge.py    # Bridge to NACE sector mapping
|   +-- regulatory_bridge.py              # Regulatory change monitoring
|   +-- health_check.py                    # 20-category system health verification
|   +-- setup_wizard.py                    # 7-step DMA configuration wizard
+-- tests/
    +-- __init__.py
    +-- conftest.py                         # Shared fixtures (DMA company profiles, IRO data)
    +-- test_manifest.py                    # Pack YAML validation
    +-- test_config.py                      # Config system tests
    +-- test_demo.py                        # Demo smoke tests
    +-- test_impact_materiality.py          # Engine 1 tests
    +-- test_financial_materiality.py       # Engine 2 tests
    +-- test_stakeholder_engagement.py      # Engine 3 tests
    +-- test_iro_identification.py          # Engine 4 tests
    +-- test_materiality_matrix.py          # Engine 5 tests
    +-- test_esrs_topic_mapping.py          # Engine 6 tests
    +-- test_threshold_scoring.py           # Engine 7 tests
    +-- test_dma_report.py                 # Engine 8 tests
    +-- test_workflows.py                   # All 8 workflows
    +-- test_templates.py                   # All 8 templates + registry
    +-- test_integrations.py               # All 8 integrations
    +-- test_e2e.py                         # End-to-end flows
    +-- test_agent_integration.py           # Agent wiring verification
```

### 3.2 Components Summary

| Category | Count | Description |
|----------|-------|-------------|
| Engines | 8 | Impact materiality, financial materiality, stakeholder engagement, IRO identification, materiality matrix, ESRS mapping, threshold/scoring, DMA reporting |
| Workflows | 8 | Impact assessment, financial assessment, stakeholder engagement, IRO identification, materiality matrix, ESRS mapping, full DMA, DMA update |
| Templates | 8 | Impact report, financial report, stakeholder report, materiality matrix, IRO register, ESRS disclosure map, executive summary, audit report |
| Integrations | 8 | Orchestrator, CSRD bridge, MRV bridge, DATA bridge, sector classification bridge, regulatory bridge, health check, setup wizard |
| Presets | 6 | Large enterprise, mid-market, SME, financial services, manufacturing, multi-sector |
| Tests | 18 | conftest + manifest + config + demo + 8 engines + workflows + templates + integrations + e2e + agent integration |

### 3.3 Key Architectural Decisions

1. **Deterministic scoring only**: All materiality scores are produced by formula-based engines operating on quantified inputs. No LLM inference is used for scoring, classification, or threshold decisions. This guarantees bit-perfect reproducibility: identical inputs always produce identical scores, rankings, and materiality determinations.

2. **Dual-axis independence**: Impact materiality and financial materiality are scored by independent engines (Engine 1 and Engine 2) that share no state. The materiality matrix engine (Engine 5) combines their outputs downstream. This ensures that impact and financial assessments are methodologically independent, as required by ESRS 1 ss43 and ss49.

3. **Configurable methodology**: The ThresholdScoringEngine (Engine 7) externalizes all scoring parameters -- scale definitions, weight factors, aggregation methods, threshold values -- into configuration. Different presets can use different scoring methodologies (e.g., multiplicative vs additive severity, linear vs logarithmic scaling) without engine code changes.

4. **Stakeholder-as-data**: Stakeholder engagement inputs (consultation results, survey responses, workshop outcomes) are treated as structured data inputs to the scoring engines, not as qualitative overrides. This ensures stakeholder perspectives are systematically integrated into scores rather than subjectively applied.

5. **IRO as first-class entity**: Every Impact, Risk, and Opportunity is a structured data object with a unique identifier, classification, scoring attributes, and lifecycle state. The IRO Register is the central data structure that flows through all engines.

6. **Delta-based updates**: The DMA Update workflow (Workflow 8) detects changes in inputs (new stakeholder data, regulatory changes, business model shifts) and re-scores only affected IROs, producing a delta report that documents what changed and why.

---

## 4. Engine Specifications

### 4.1 Engine 1: Impact Materiality Engine

**File**: `engines/impact_materiality_engine.py`
**Purpose**: Score impact materiality for each sustainability matter per ESRS 1 ss43-48, computing severity (scale x scope x irremediability) and likelihood for potential impacts.

**Key Features**:
- **Severity scoring (actual negative impacts)**: Three-component score:
  - **Scale** (1-5): Gravity of the impact, from negligible (1) to catastrophic (5), defined by quantified thresholds per topic (e.g., for E1 climate: 1 = <100 tCO2e, 5 = >1M tCO2e)
  - **Scope** (1-5): Breadth of impact, from isolated (1) to global (5), measured by affected population/area/ecosystem extent
  - **Irremediability** (1-5): Difficulty of remediation, from easily reversible (1) to permanent/irreversible (5)
  - **Severity = Scale x Scope x Irremediability** (range 1-125, normalized to 0-100)
- **Potential impact scoring**: Severity (as above) combined with Likelihood:
  - **Likelihood** (1-5): From very unlikely (<5%) to almost certain (>90%), with defined probability ranges per level
  - **Potential Impact Score = Severity x Likelihood weight** (configurable: multiplicative or matrix-based)
- **Positive impact scoring**: Scale x Scope x Likelihood (irremediability not applicable to positive impacts)
- **Human rights override**: Per ESRS 1 ss47, human rights impacts (S1-S4 categories flagged as human-rights-related) use severity-only scoring where severity alone can trigger materiality, regardless of likelihood
- **Time horizon tagging**: Each impact tagged as short-term (<1yr), medium-term (1-5yr), or long-term (>5yr)
- **Value chain stage mapping**: Each impact mapped to value chain stage (own operations, upstream, downstream) per ESRS 1 ss29

**Core Calculations**:
```
# Actual negative impact
Severity = Scale * Scope * Irremediability
Normalized_Severity = (Severity - 1) / (125 - 1) * 100

# Potential negative impact
Potential_Impact_Score = Normalized_Severity * Likelihood_Weight
  where Likelihood_Weight = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7, 5: 0.9}  [configurable]

# Human rights impact (severity-only mode)
HR_Impact_Score = Normalized_Severity  # Likelihood not used as per ESRS 1 ss47

# Positive impact
Positive_Impact_Score = (Scale * Scope / 25) * 100 * Likelihood_Weight

# Impact materiality determination
Is_Material = Impact_Score >= Impact_Materiality_Threshold  [default: 50/100]
```

**Regulatory References**:
- ESRS 1 ss43-48 (impact materiality definition and scoring)
- ESRS 1 Appendix A (list of sustainability matters)
- EFRAG IG-1 ss36-65 (impact materiality implementation guidance)

**Models**: `ImpactMaterialityConfig`, `SustainabilityMatter`, `ImpactAssessment`, `SeverityScore`, `LikelihoodScore`, `ImpactMaterialityResult`, `HumanRightsOverride`, `TimeHorizon`, `ValueChainStage`

**Edge Cases**:
- Human rights impact with low likelihood but high severity: Use severity-only mode per ESRS 1 ss47
- Impact spans multiple time horizons: Score for each horizon separately, report worst-case
- Positive and negative impacts on same topic: Score independently, present both on matrix
- Impact data unavailable: Reject scoring for that IRO with explicit data gap flag (do not estimate)

### 4.2 Engine 2: Financial Materiality Engine

**File**: `engines/financial_materiality_engine.py`
**Purpose**: Score financial materiality for each sustainability matter per ESRS 1 ss49-51, computing financial magnitude, likelihood, and time horizon of financial effects.

**Key Features**:
- **Financial magnitude scoring (absolute)**:
  - Revenue impact: Direct revenue at risk or opportunity (EUR)
  - Cost impact: Compliance costs, operational costs, remediation costs (EUR)
  - Asset impact: Stranded assets, impairment, revaluation (EUR)
  - Access to finance: Cost of capital change (bps), credit rating impact
  - Normalized to 0-100 scale relative to company financials (% of revenue, % of total assets, % of EBITDA)
- **Financial magnitude scoring (relative thresholds)**:
  - Level 1 (Negligible): <0.1% of revenue
  - Level 2 (Minor): 0.1-0.5% of revenue
  - Level 3 (Moderate): 0.5-2% of revenue
  - Level 4 (Significant): 2-5% of revenue
  - Level 5 (Critical): >5% of revenue
  - Thresholds configurable per company size and industry
- **Likelihood scoring**: Same 5-level scale as impact materiality (probability-based)
- **Time horizon scoring**: Financial effect proximity weighting:
  - Short-term (<1yr): Weight 1.0 (full impact)
  - Medium-term (1-5yr): Weight 0.7 (discounted)
  - Long-term (>5yr): Weight 0.4 (further discounted)
  - Weights configurable per methodology preference
- **Financial effect types per ESRS 1 ss50**:
  - Cash flow effects (operational and capital)
  - Development and performance effects (revenue growth/decline)
  - Financial position effects (assets, liabilities)
  - Cost of capital and access to finance
- **Risk vs opportunity classification**: Each financial assessment tagged as risk (downside), opportunity (upside), or both

**Core Calculations**:
```
# Financial magnitude (absolute to normalized)
Revenue_Impact_Score = min(100, abs(revenue_impact_eur) / annual_revenue * 100 / 5 * 100)
  # e.g., EUR 10M impact for EUR 1B company = 1% = Level 3 (Moderate) = ~40/100

# Financial materiality score
Financial_Score = Magnitude_Normalized * Likelihood_Weight * TimeHorizon_Weight

# Financial materiality determination
Is_Material = Financial_Score >= Financial_Materiality_Threshold  [default: 40/100]

# Combined risk-opportunity score (when both apply)
Risk_Score = Magnitude_Risk * Likelihood_Risk * TimeHorizon_Risk
Opportunity_Score = Magnitude_Opp * Likelihood_Opp * TimeHorizon_Opp
Net_Financial_Score = max(Risk_Score, Opportunity_Score)  # Report both, use max for materiality
```

**Regulatory References**:
- ESRS 1 ss49-51 (financial materiality definition)
- ESRS 1 ss50 (financial effects enumeration)
- EFRAG IG-1 ss66-95 (financial materiality implementation guidance)

**Models**: `FinancialMaterialityConfig`, `FinancialAssessment`, `FinancialMagnitude`, `FinancialEffect`, `TimeHorizonWeight`, `FinancialMaterialityResult`, `RiskOpportunityClassification`

**Edge Cases**:
- Financial effect is qualitative only (e.g., reputational risk with no quantified EUR value): Use proxy estimation from industry benchmarks, flag as estimated with lower DQ score
- Multiple financial effect types for same topic: Sum absolute values for magnitude, report breakdown
- Financial effect depends on regulatory outcome (e.g., pending legislation): Model as scenario (base, upside, downside)
- Zero financial impact but non-zero impact materiality: Topic can be material on impact axis alone

### 4.3 Engine 3: Stakeholder Engagement Engine

**File**: `engines/stakeholder_engagement_engine.py`
**Purpose**: Manage the full stakeholder engagement lifecycle -- identification, mapping, consultation, synthesis, and validation -- per ESRS 1 ss22-23.

**Key Features**:
- **Stakeholder identification**: Classify stakeholders per ESRS 1 ss22 categories:
  - Affected stakeholders: Employees, workers in the value chain, communities, consumers/end-users
  - Users of sustainability statements: Investors, lenders, creditors, rating agencies, regulators
  - Other: NGOs, industry associations, academic experts, civil society
- **Stakeholder mapping**: Assess each stakeholder group on:
  - Salience (how affected by the undertaking's activities): High/Medium/Low
  - Influence (how much power to affect the undertaking): High/Medium/Low
  - Proximity (directness of relationship): Direct/Indirect/Remote
  - Generates prioritization matrix: salience x influence grid
- **Consultation management**: Track consultation activities per stakeholder group:
  - Method: Survey, interview, focus group, workshop, public consultation, grievance mechanism, dialogue forum
  - Topics covered: Which ESRS sustainability matters were discussed
  - Participation: Number of participants, response rate
  - Key findings: Structured data extraction from consultation outputs
  - Dates and cadence of engagement
- **Input synthesis**: Convert qualitative stakeholder feedback into structured scoring inputs:
  - Topic importance ranking per stakeholder group (ordinal ranking)
  - Concern level per topic (1-5 Likert scale, aggregated)
  - Priority topics flagged by each stakeholder group
  - Consensus analysis: Agreement/disagreement across stakeholder groups
- **Stakeholder-weighted scoring**: Weight sustainability matters based on stakeholder salience:
  - Highly-affected stakeholders receive higher weight in impact materiality
  - Financial statement users receive higher weight in financial materiality
  - Configurable weight factors per stakeholder group and materiality axis
- **Engagement documentation**: Generate ESRS 2 IRO-1 compliant documentation of the engagement process

**Models**: `StakeholderEngagementConfig`, `StakeholderGroup`, `StakeholderMapping`, `ConsultationRecord`, `ConsultationFinding`, `StakeholderSynthesis`, `EngagementDocumentation`, `SalienceInfluenceMatrix`

**Regulatory References**:
- ESRS 1 ss22-23 (stakeholder engagement in materiality assessment)
- ESRS 2 IRO-1 (description of process to identify and assess material IROs)
- EFRAG IG-1 ss22-35 (stakeholder engagement guidance)

### 4.4 Engine 4: IRO Identification Engine

**File**: `engines/iro_identification_engine.py`
**Purpose**: Identify and classify Impacts, Risks, and Opportunities across all ESRS topics per ESRS 1 ss28-39, producing a structured IRO Register.

**Key Features**:
- **Value chain mapping**: Map the undertaking's value chain to identify where IROs occur:
  - Own operations: Production, facilities, offices
  - Upstream: Raw materials, suppliers, logistics
  - Downstream: Distribution, use, end-of-life
  - Mapping driven by NACE code, business description, and sector classification
- **ESRS topic screening**: Screen all sustainability matters from ESRS 1 Appendix A:
  - 10 ESRS topics (E1-E5, S1-S4, G1)
  - ~100 sub-topics across all ESRS standards
  - Pre-populated relevance indicators per NACE sector (from EFRAG sector guidance)
- **IRO discovery**: For each relevant sustainability matter, identify specific IROs:
  - **Impacts**: Actual (already occurring) or potential (may occur), positive or negative
  - **Risks**: Financial downside from sustainability matters (regulatory, physical, transition, litigation, reputational)
  - **Opportunities**: Financial upside (market differentiation, cost savings, new products, access to capital)
- **IRO classification**: Each IRO receives structured attributes:
  - Unique ID (e.g., `IRO-E1-001`)
  - ESRS topic and sub-topic
  - Type: Impact / Risk / Opportunity / Combined
  - Nature: Actual or Potential
  - Polarity: Positive or Negative (for impacts)
  - Value chain stage: Own operations / Upstream / Downstream
  - Time horizon: Short / Medium / Long
  - Related business activities
- **IRO prioritization**: Pre-score IROs using sector benchmarks and available data to guide assessment effort
- **IRO Register**: Central data structure holding all identified IROs with their attributes and assessment status

**Core IRO Register Schema**:
```
IRO Register Entry:
  iro_id: str                    # Unique identifier (e.g., IRO-E1-001)
  esrs_topic: ESRSTopic          # E1-E5, S1-S4, G1
  esrs_sub_topic: str            # Specific sub-topic
  iro_type: IROType              # IMPACT | RISK | OPPORTUNITY | COMBINED
  nature: IRONature              # ACTUAL | POTENTIAL
  polarity: Polarity             # POSITIVE | NEGATIVE (impacts only)
  value_chain_stage: VCStage     # OWN_OPERATIONS | UPSTREAM | DOWNSTREAM
  time_horizon: TimeHorizon      # SHORT | MEDIUM | LONG
  description: str               # Plain-language description
  business_activities: List[str] # Related NACE activities
  stakeholder_groups: List[str]  # Affected stakeholder groups
  assessment_status: Status      # IDENTIFIED | ASSESSED | MATERIAL | NOT_MATERIAL
  impact_score: Optional[float]  # From Engine 1 (0-100)
  financial_score: Optional[float] # From Engine 2 (0-100)
  provenance_hash: str           # SHA-256 hash of all inputs
```

**Models**: `IROIdentificationConfig`, `ValueChainMap`, `IROEntry`, `IRORegister`, `TopicScreeningResult`, `SectorRelevance`, `IRODiscoveryResult`

**Regulatory References**:
- ESRS 1 ss28-39 (IRO identification and assessment process)
- ESRS 1 Appendix A (list of sustainability matters)
- ESRS 2 SBM-3 (material impacts, risks, and opportunities)

### 4.5 Engine 5: Materiality Matrix Engine

**File**: `engines/materiality_matrix_engine.py`
**Purpose**: Generate the double materiality matrix by combining impact materiality scores (Engine 1) and financial materiality scores (Engine 2), applying thresholds to determine material topics.

**Key Features**:
- **Score aggregation**: For each ESRS topic, aggregate IRO-level scores to topic-level:
  - Impact topic score = max(IRO impact scores for that topic) -- worst-case drives materiality
  - Financial topic score = max(IRO financial scores for that topic) -- worst-case drives materiality
  - Configurable aggregation: max, weighted average, or percentile-based
- **Matrix generation**: Produce a 2D matrix:
  - X-axis: Financial materiality score (0-100)
  - Y-axis: Impact materiality score (0-100)
  - Each ESRS topic plotted as a point (or bubble, with size = combined score)
- **Threshold application**: Apply materiality thresholds to determine material/not-material:
  - **Material**: Impact score >= threshold OR Financial score >= threshold (ESRS 1 requires either axis)
  - **Double material**: Both impact AND financial scores above respective thresholds
  - **Impact-only material**: Impact >= threshold, Financial < threshold
  - **Financial-only material**: Financial >= threshold, Impact < threshold
  - **Not material**: Both below thresholds
- **Quadrant classification**: Classify each topic into one of 4 quadrants:
  - Q1 (top-right): High impact, high financial -- priority topics
  - Q2 (top-left): High impact, low financial -- stakeholder-driven topics
  - Q3 (bottom-right): Low impact, high financial -- business-driven topics
  - Q4 (bottom-left): Low impact, low financial -- non-material (monitored)
- **Sensitivity analysis**: Test threshold sensitivity by varying thresholds +/- 10% and identifying topics that change materiality status
- **Year-over-year comparison**: Compare current matrix with previous year's matrix, highlighting movements

**Core Calculations**:
```
# Topic-level aggregation (default: max)
Topic_Impact_Score(E1) = max(IRO_Impact_Scores where esrs_topic == E1)
Topic_Financial_Score(E1) = max(IRO_Financial_Scores where esrs_topic == E1)

# Materiality determination (OR logic per ESRS 1)
Is_Material(topic) = Topic_Impact_Score >= Impact_Threshold
                   OR Topic_Financial_Score >= Financial_Threshold

# Quadrant assignment
Quadrant(topic) = {
  Q1 if impact >= threshold AND financial >= threshold,
  Q2 if impact >= threshold AND financial < threshold,
  Q3 if impact < threshold AND financial >= threshold,
  Q4 if impact < threshold AND financial < threshold
}

# Sensitivity: topics within +/- 10% of threshold
Border_Topics = {t | abs(score(t) - threshold) <= threshold * 0.10}
```

**Models**: `MaterialityMatrixConfig`, `MatrixEntry`, `MaterialityMatrix`, `QuadrantClassification`, `SensitivityResult`, `YoYComparison`, `MaterialityDetermination`

### 4.6 Engine 6: ESRS Topic Mapping Engine

**File**: `engines/esrs_topic_mapping_engine.py`
**Purpose**: Map material topics identified by the materiality matrix to specific ESRS disclosure requirements, producing a disclosure obligation register and gap analysis.

**Key Features**:
- **ESRS disclosure requirement database**: Complete register of all ESRS disclosure requirements:
  - ESRS E1: 9 disclosure requirements (E1-1 through E1-9), ~60 datapoints
  - ESRS E2: 6 disclosure requirements (E2-1 through E2-6), ~40 datapoints
  - ESRS E3: 5 disclosure requirements (E3-1 through E3-5), ~35 datapoints
  - ESRS E4: 6 disclosure requirements (E4-1 through E4-6), ~50 datapoints
  - ESRS E5: 6 disclosure requirements (E5-1 through E5-6), ~40 datapoints
  - ESRS S1: 17 disclosure requirements (S1-1 through S1-17), ~80 datapoints
  - ESRS S2: 5 disclosure requirements (S2-1 through S2-5), ~30 datapoints
  - ESRS S3: 5 disclosure requirements (S3-1 through S3-5), ~30 datapoints
  - ESRS S4: 5 disclosure requirements (S4-1 through S4-5), ~30 datapoints
  - ESRS G1: 6 disclosure requirements (G1-1 through G1-6), ~35 datapoints
  - **Total**: ~70 disclosure requirements, ~430 datapoints (pre-Omnibus I)
- **Omnibus I reduction mapping**: Apply 61% datapoint reduction per Omnibus I:
  - ~170 datapoints remaining after Omnibus I
  - Voluntary vs mandatory classification per datapoint
  - Phased-in vs immediate classification
- **Material topic to disclosure mapping**: For each material ESRS topic, enumerate:
  - Mandatory disclosure requirements (always apply when topic is material)
  - Conditional disclosure requirements (apply if specific sub-topics are material)
  - Quantitative datapoints required
  - Narrative/qualitative disclosures required
  - Phased-in datapoints (e.g., Scope 3, biodiversity metrics)
- **Disclosure gap analysis**: Compare required disclosures against available data:
  - Green: Data available and validated
  - Amber: Data partially available or needs quality improvement
  - Red: Data not available, collection required
  - Generate data collection plan for red/amber gaps
- **Cross-reference to GreenLang agents**: Map each disclosure requirement to the GreenLang agent(s) capable of producing the required data:
  - E1 climate disclosures -> MRV agents (001-030)
  - S1 workforce disclosures -> HR data connectors
  - G1 governance disclosures -> Governance data
  - Identifies which agents need to be activated

**Models**: `ESRSTopicMappingConfig`, `DisclosureRequirement`, `DatapointDefinition`, `TopicDisclosureMap`, `DisclosureGapAnalysis`, `DataCollectionPlan`, `AgentCapabilityMap`, `OmnibusIReduction`

**Regulatory References**:
- ESRS Set 1 (all 12 standards: ESRS 1, ESRS 2, E1-E5, S1-S4, G1)
- Omnibus I Directive (EU) 2026/470 Annex (datapoint reduction)
- EFRAG IG-3 (ESRS datapoint catalogue)

### 4.7 Engine 7: Threshold Scoring Engine

**File**: `engines/threshold_scoring_engine.py`
**Purpose**: Provide configurable scoring methodologies and materiality thresholds that can be customized per industry, company size, and assessment methodology preference.

**Key Features**:
- **Scoring methodology selection**:
  - **Multiplicative** (default): Severity = Scale * Scope * Irremediability (ESRS 1 aligned)
  - **Additive weighted**: Severity = w1*Scale + w2*Scope + w3*Irremediability (alternative)
  - **Matrix-based**: Predefined severity matrix (5x5x5 lookup table)
  - **Logarithmic**: ln-scaled for high-dynamic-range inputs
- **Scale definitions**: Configurable per-level descriptions and quantified boundaries:
  - Scale 1-5 definitions for each ESRS topic (e.g., E1 climate scale tied to tCO2e bands)
  - Scope 1-5 definitions for each ESRS topic (e.g., geographic extent, population affected)
  - Irremediability 1-5 definitions per topic
  - Financial magnitude 1-5 definitions per company revenue band
  - Likelihood 1-5 with probability ranges
- **Materiality threshold configuration**:
  - Impact materiality threshold: default 50/100, configurable 30-70
  - Financial materiality threshold: default 40/100, configurable 20-60
  - Separate thresholds per ESRS pillar (E, S, G) if desired
  - Human rights floor: minimum threshold override for human rights topics
- **Industry-specific calibration**:
  - Pre-calibrated thresholds per NACE sector (extractives, manufacturing, financial, retail, services)
  - Sector-specific scale definitions (what constitutes "high severity" differs by industry)
  - Benchmark data from published DMAs by sector
- **Weight factor management**:
  - Stakeholder group weights for score aggregation
  - Time horizon discount factors
  - Value chain stage proximity weights (own operations > upstream > downstream)
  - Configurable and auditable (all weight changes tracked)
- **Scoring normalization**: Ensure all scores map to consistent 0-100 range regardless of methodology

**Models**: `ThresholdScoringConfig`, `ScoringMethodology`, `ScaleDefinition`, `ThresholdSet`, `IndustryCalibration`, `WeightFactors`, `NormalizationParams`, `MethodologyDocumentation`

### 4.8 Engine 8: DMA Report Engine

**File**: `engines/dma_report_engine.py`
**Purpose**: Assemble the complete DMA report document combining outputs from all engines, with methodology documentation and full audit trail for assurance provider review.

**Key Features**:
- **Report assembly**: Combine outputs from Engines 1-7 into a structured DMA report:
  - Executive summary with key findings (material topics, total IROs, stakeholder summary)
  - Methodology section (scoring approach, thresholds, stakeholder engagement process)
  - Impact materiality results (per-topic scores, ranking, detailed assessments)
  - Financial materiality results (per-topic scores, ranking, detailed assessments)
  - Double materiality matrix (visual representation with quadrant classification)
  - IRO register (full register with all assessed IROs)
  - ESRS disclosure mapping (material topics -> required disclosures)
  - Disclosure gap analysis (data availability status)
  - Stakeholder engagement summary (process, participants, findings)
  - Sensitivity analysis (threshold sensitivity, border topics)
  - Comparison with prior year (if applicable)
  - Appendices (data sources, calculation methodology, glossary)
- **Audit trail**: Every score traceable from final value back to:
  - Raw input data with source reference
  - Calculation formula applied
  - Threshold used
  - Stakeholder weight applied
  - SHA-256 provenance hash at each calculation step
- **ESRS 2 IRO-1 compliance**: Report structure directly maps to ESRS 2 IRO-1 disclosure requirements:
  - Process used to identify material IROs
  - How affected stakeholders were considered
  - Outcome of materiality assessment
  - Nature and scope of DMA
- **Export formats**: Structured JSON (for programmatic consumption), markdown (for human review), XBRL-ready tags (for iXBRL filing)
- **Version management**: DMA reports are versioned with change tracking between versions

**Models**: `DMAReportConfig`, `DMAReport`, `ReportSection`, `AuditTrailEntry`, `ProvenanceChain`, `ExportFormat`, `ReportVersion`, `ChangeLog`

---

## 5. Workflow Specifications

### 5.1 Workflow 1: Impact Assessment

**File**: `workflows/impact_assessment_workflow.py`
**Phases**: 4

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | DataCollection | Company profile, sector data, MRV emissions data, supply chain data, incident records | Collect and validate all data inputs relevant to impact assessment; normalize units and formats | Validated impact data package |
| 2 | TopicIdentification | Validated data, ESRS topic list, sector relevance indicators | Screen all ESRS topics for relevance using sector NACE mapping and value chain analysis | Shortlist of potentially relevant topics with preliminary IROs |
| 3 | SeverityScoring | Relevant topics, impact data, scale definitions, stakeholder input | Score each IRO on scale, scope, irremediability using Engine 1; apply human rights override where applicable | Scored IROs with severity and likelihood |
| 4 | ImpactRanking | Scored IROs, materiality threshold | Rank IROs by impact score; apply threshold to classify material vs not-material; generate impact materiality results | Impact materiality ranking with provenance hashes |

**Trigger**: At DMA initiation or when material changes to business model or value chain occur.
**Duration**: <2 minutes for 100 IROs.

### 5.2 Workflow 2: Financial Assessment

**File**: `workflows/financial_assessment_workflow.py`
**Phases**: 4

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | FinancialData | Company financials (revenue, assets, EBITDA), risk register, investment plans | Collect financial baseline data for magnitude normalization | Financial baseline package |
| 2 | RiskOpportunityMapping | IRO register, financial data, sector risk profiles, regulatory pipeline | Map each IRO to specific financial effects (revenue, cost, asset, access to capital); classify as risk/opportunity | IROs with financial effect mapping |
| 3 | FinancialScoring | Mapped IROs, financial magnitude definitions, likelihood scales, time horizon weights | Score each IRO on financial magnitude, likelihood, time horizon using Engine 2 | Scored IROs with financial materiality scores |
| 4 | FinancialRanking | Scored IROs, financial materiality threshold | Rank IROs by financial score; apply threshold; generate financial materiality results | Financial materiality ranking with provenance hashes |

**Trigger**: At DMA initiation or when material changes to company financials or risk profile occur.
**Duration**: <2 minutes for 100 IROs.

### 5.3 Workflow 3: Stakeholder Engagement

**File**: `workflows/stakeholder_engagement_workflow.py`
**Phases**: 5

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | StakeholderIdentification | Company profile, value chain map, ESRS stakeholder categories | Identify all relevant stakeholder groups per ESRS 1 ss22 categories | Stakeholder group register |
| 2 | StakeholderMapping | Stakeholder groups | Assess salience, influence, proximity for each group; generate prioritization matrix | Stakeholder prioritization matrix |
| 3 | ConsultationDesign | Prioritized stakeholders, ESRS topics | Design consultation plan: methods, topics, timeline, participation targets per group | Consultation plan |
| 4 | InputSynthesis | Consultation results (survey data, workshop outputs, interview transcripts) | Convert qualitative inputs to structured scoring data; aggregate responses; identify consensus and divergence | Structured stakeholder scoring inputs |
| 5 | Validation | Synthesized inputs, initial materiality results | Cross-validate stakeholder input against materiality scores; identify stakeholder-highlighted topics not flagged by data-driven assessment | Validated stakeholder inputs with cross-validation report |

**Trigger**: Before or concurrent with impact/financial assessment.
**Duration**: Data processing <5 minutes; actual stakeholder engagement is external to the system.

### 5.4 Workflow 4: IRO Identification

**File**: `workflows/iro_identification_workflow.py`
**Phases**: 4

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | ValueChainMapping | Company profile, NACE code, business description, supply chain data | Map value chain stages (own operations, upstream tiers, downstream stages) with activities per stage | Value chain map |
| 2 | IRODiscovery | Value chain map, ESRS topic list, sector relevance database | For each value chain stage x ESRS topic intersection, identify potential IROs using sector-specific IRO templates | Draft IRO register with all candidate IROs |
| 3 | IROClassification | Draft IROs | Classify each IRO: type (I/R/O), nature (actual/potential), polarity (positive/negative), time horizon, value chain stage | Classified IRO register |
| 4 | IROPrioritization | Classified IROs, available data indicators | Pre-prioritize IROs by sector relevance and data availability; flag IROs requiring primary data collection | Prioritized IRO register ready for assessment |

**Trigger**: At DMA initiation; repeated when business model or value chain changes.
**Duration**: <3 minutes for a multi-sector conglomerate.

### 5.5 Workflow 5: Materiality Matrix

**File**: `workflows/materiality_matrix_workflow.py`
**Phases**: 3

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | ScoreAggregation | Impact scores (Engine 1), financial scores (Engine 2), aggregation config | Aggregate IRO-level scores to ESRS topic-level using configured method (max/weighted avg/percentile) | Topic-level impact and financial scores |
| 2 | MatrixGeneration | Topic-level scores | Plot all ESRS topics on 2D matrix (impact x financial); assign quadrant classifications | Materiality matrix with quadrant assignments |
| 3 | ThresholdApplication | Matrix, thresholds, sensitivity config | Apply materiality thresholds; determine material/not-material per topic; run sensitivity analysis (+/- 10%); compare with prior year if available | Final materiality determination with sensitivity analysis |

**Trigger**: After both impact and financial assessments are complete.
**Duration**: <30 seconds.

### 5.6 Workflow 6: ESRS Mapping

**File**: `workflows/esrs_mapping_workflow.py`
**Phases**: 3

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | TopicSelection | Materiality determination (material topics list) | Extract list of material ESRS topics | Material topics list |
| 2 | ESRSMapping | Material topics, ESRS disclosure database, Omnibus I reduction rules | Map each material topic to required disclosure requirements and datapoints; apply Omnibus I reductions | Disclosure obligation register |
| 3 | GapAnalysis | Disclosure obligations, available data inventory, GreenLang agent capabilities | Assess data availability per disclosure; map to GreenLang agents; identify gaps requiring data collection | Disclosure gap analysis with data collection plan |

**Trigger**: After materiality matrix is finalized.
**Duration**: <1 minute.

### 5.7 Workflow 7: Full DMA

**File**: `workflows/full_dma_workflow.py`
**Phases**: 6

| Phase | Name | Sub-Workflows / Engines | Duration |
|-------|------|------------------------|----------|
| 1 | Initialization | Load config, validate preset, check agent availability, load sector data | <30s |
| 2 | StakeholderEngagement | Workflow 3 (stakeholder identification, mapping, synthesis) | <5min |
| 3 | IROIdentification | Workflow 4 (value chain mapping, IRO discovery, classification) | <3min |
| 4 | MaterialityAssessment | Workflow 1 (impact) + Workflow 2 (financial) -- run in parallel | <2min |
| 5 | MatrixAndMapping | Workflow 5 (matrix generation) + Workflow 6 (ESRS mapping) -- sequential | <1.5min |
| 6 | Reporting | Engine 8 (DMA report assembly with full audit trail) | <2min |

**Total Duration**: <15 minutes for full end-to-end DMA.
**Trigger**: Annual DMA cycle or new company onboarding.

### 5.8 Workflow 8: DMA Update

**File**: `workflows/dma_update_workflow.py`
**Phases**: 4

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | ChangeDetection | Previous DMA, new data inputs (regulatory changes, stakeholder updates, business changes, new emissions data) | Compare new inputs against previous DMA inputs; identify changed parameters | Change register with affected IROs |
| 2 | ReAssessment | Changed IROs, new data | Re-score only affected IROs through Engine 1 and/or Engine 2; retain unchanged scores | Updated IRO scores |
| 3 | DeltaAnalysis | Previous scores, updated scores, previous matrix | Determine if any topics changed materiality status; identify score movements; flag new/removed IROs | Delta report with materiality changes |
| 4 | UpdatePublication | Delta report, updated matrix | Generate updated DMA report with change annotations; update ESRS mapping if material topics changed; archive previous version | Updated DMA with change log |

**Trigger**: Quarterly monitoring cycle, regulatory change, or material business event.
**Duration**: <5 minutes (proportional to number of changes).

---

## 6. Template Specifications

### 6.1 Template 1: Impact Materiality Report

**File**: `templates/impact_materiality_report.py`
**Outputs**: Detailed impact materiality assessment results.

**Key Sections**:
- Executive summary: Number of IROs assessed, material impact topics, human rights flags
- Topic ranking table: All ESRS topics ranked by impact score with classification (material/not material)
- Per-topic detail: For each material topic -- specific IROs, severity breakdown (scale/scope/irremediability), likelihood, value chain stage, time horizon
- Human rights analysis: Topics where severity-only override was applied per ESRS 1 ss47
- Positive impacts: Positive IROs with scale/scope assessment
- Methodology documentation: Scoring methodology, scale definitions, threshold used
- Provenance hashes for all calculations

### 6.2 Template 2: Financial Materiality Report

**File**: `templates/financial_materiality_report.py`
**Outputs**: Detailed financial materiality assessment results.

**Key Sections**:
- Executive summary: Total financial exposure (risks vs opportunities), material financial topics
- Topic ranking table: All ESRS topics ranked by financial score
- Per-topic detail: Financial magnitude (EUR and normalized), likelihood, time horizon, financial effect type
- Risk register overlay: Top 10 financial risks from sustainability matters with quantified exposure
- Opportunity register: Top 10 financial opportunities with estimated upside
- Scenario sensitivity: Base/upside/downside for key financial assessments
- Methodology documentation: Magnitude thresholds, time horizon weights, normalization approach

### 6.3 Template 3: Stakeholder Engagement Report

**File**: `templates/stakeholder_engagement_report.py`
**Outputs**: ESRS 2 IRO-1 compliant stakeholder engagement documentation.

**Key Sections**:
- Stakeholder group inventory with salience/influence mapping
- Consultation plan and execution summary (methods, participation, coverage)
- Key findings per stakeholder group: topics flagged, concerns raised, priorities
- Consensus/divergence analysis across stakeholder groups
- Influence on materiality assessment: how stakeholder input modified scores
- Engagement gaps: stakeholder groups not yet consulted, planned next steps
- ESRS 2 IRO-1 compliance checklist

### 6.4 Template 4: Materiality Matrix Report

**File**: `templates/materiality_matrix_report.py`
**Outputs**: Visual double materiality matrix with supporting analysis.

**Key Sections**:
- Double materiality matrix visualization data (2D plot with all 10 ESRS topics positioned)
- Quadrant summary: Count and list of topics per quadrant (Q1: double material, Q2: impact-only, Q3: financial-only, Q4: not material)
- Threshold lines with values used
- Sensitivity analysis: border topics within 10% of threshold
- Year-over-year movement (if previous DMA exists): topics that changed quadrant or materiality status
- Material topics list for ESRS disclosure

### 6.5 Template 5: IRO Register Report

**File**: `templates/iro_register_report.py`
**Outputs**: Complete IRO register formatted for review and audit.

**Key Sections**:
- Summary statistics: Total IROs, by type (I/R/O), by ESRS topic, by value chain stage
- Full IRO register table: ID, topic, type, nature, polarity, value chain stage, time horizon, impact score, financial score, materiality status
- IRO heatmap: ESRS topic x value chain stage matrix showing IRO density and severity
- Top 20 IROs by combined score
- New IROs identified vs previous assessment (if applicable)
- IROs requiring further data collection

### 6.6 Template 6: ESRS Disclosure Map

**File**: `templates/esrs_disclosure_map.py`
**Outputs**: Complete mapping from material topics to required ESRS disclosures.

**Key Sections**:
- Material topics to disclosure requirements matrix
- Datapoint inventory: Total required datapoints, Omnibus I reduced count, voluntary vs mandatory
- Data availability dashboard: Green (available) / Amber (partial) / Red (missing) per disclosure
- GreenLang agent capability map: Which agents produce which disclosures
- Data collection plan: Timeline and ownership for red/amber gaps
- Phased-in disclosure timeline: Which datapoints are phased in for first reporting year

### 6.7 Template 7: DMA Executive Summary

**File**: `templates/dma_executive_summary.py`
**Outputs**: Board-ready executive summary (2-3 pages).

**Key Sections**:
- One-page summary: Number of material topics, top 5 by combined score, key changes from prior year
- Materiality matrix (simplified visual)
- Material topics with brief rationale (1-2 sentences each)
- ESRS disclosure scope: Number of standards applicable, number of datapoints required
- Strategic implications: Top 3 risks and top 3 opportunities from the DMA
- Next steps and recommended actions
- Assurance readiness indicator

### 6.8 Template 8: DMA Audit Report

**File**: `templates/dma_audit_report.py`
**Outputs**: Audit working paper package for limited or reasonable assurance engagement.

**Key Sections**:
- Methodology documentation: Complete description of scoring approach, formulas, parameters
- Input data inventory: All data sources with freshness dates, quality scores, and source references
- Calculation trace: For each material topic, full calculation chain from raw input to final score
- Provenance chain: SHA-256 hashes at every calculation step with verification checksums
- Threshold justification: Rationale for chosen thresholds with industry benchmarks
- Stakeholder engagement evidence: Consultation records, participant lists, finding summaries
- Sensitivity analysis: Impact of threshold changes on materiality outcomes
- Completeness assertion: Confirmation that all ESRS 1 Appendix A topics were screened
- Change log: All modifications since previous DMA with rationale

---

## 7. Integration Specifications

### 7.1 Pack Orchestrator

**File**: `integrations/pack_orchestrator.py`
**Purpose**: Manage the 9-phase DMA pipeline from initialization through report generation.

**Pipeline Phases**:
```
Phase 1:  INITIALIZATION          - Load config, validate preset, check dependencies
Phase 2:  DATA_INTAKE             - Ingest company profile, financials, sector data, emissions data
Phase 3:  QUALITY_ASSURANCE       - Validate input data quality, flag gaps, normalize formats
Phase 4:  STAKEHOLDER_ENGAGEMENT  - Process stakeholder inputs (Engine 3)
Phase 5:  IRO_IDENTIFICATION      - Identify and classify IROs (Engine 4)
Phase 6:  IMPACT_ASSESSMENT       - Score impact materiality (Engine 1)
Phase 7:  FINANCIAL_ASSESSMENT    - Score financial materiality (Engine 2) [parallel with Phase 6]
Phase 8:  MATRIX_AND_MAPPING      - Generate matrix (Engine 5) + ESRS mapping (Engine 6)
Phase 9:  REPORTING               - Assemble DMA report (Engine 8)
```

**Orchestrator Features**:
- Phase-level enable/disable (e.g., skip stakeholder engagement if already completed externally)
- Retry policy: Configurable max retries (default 3), exponential backoff (1.5x)
- Provenance tracking: SHA-256 hash at every phase boundary
- Dry-run mode: Validate pipeline configuration without executing engines
- Parallel execution: Phases 6 and 7 run in parallel (independent engines)
- Timeout management: Configurable per-phase timeout (default 120s)
- Progress callbacks: Real-time progress updates for UI integration
- Checkpoint/resume: Save state between phases for long-running assessments

**Models**: `OrchestratorConfig`, `RetryPolicy`, `PipelinePhase`, `PhaseStatus`, `PhaseResult`, `PhaseProvenance`, `DMAOrchestrationResult`

### 7.2 CSRD Pack Bridge

**File**: `integrations/csrd_pack_bridge.py`
**Purpose**: Bridge to PACK-001 (CSRD Starter), PACK-002 (CSRD Professional), PACK-003 (CSRD Enterprise) to feed DMA results into CSRD reporting.

**Bridge Functions**:
- Export material topics list to CSRD packs for disclosure scope
- Export ESRS disclosure map for datapoint collection
- Export materiality matrix for ESRS 2 IRO-1/IRO-2 disclosures
- Import governance data from CSRD packs for G1 assessment
- Import prior-year DMA results for comparison
- Version compatibility checking across pack versions
- Replace CSRD pack's built-in `materiality_agent.py` with PACK-015 deterministic engines

### 7.3 MRV Materiality Bridge

**File**: `integrations/mrv_materiality_bridge.py`
**Purpose**: Bridge to MRV agents (001-030) to obtain emissions data that informs impact and financial materiality scoring for environmental topics.

**Agent Routing**:
- **E1 Climate scoring context**: Total Scope 1 (MRV-001..008), Scope 2 (MRV-009..013), Scope 3 (MRV-014..030) emissions as input to E1 impact scale scoring
- **E2 Pollution scoring context**: Pollution-related data from process/fugitive agents (MRV-004, MRV-005)
- **E5 Circular Economy scoring context**: Waste emissions data from MRV-007
- **Financial magnitude context**: Carbon cost exposure (tCO2e x ETS price) for E1 financial scoring
- Emissions data quality score feeds into DMA data quality assessment

### 7.4 Data Materiality Bridge

**File**: `integrations/data_materiality_bridge.py`
**Purpose**: Bridge to DATA agents (001-020) for ingesting and validating external data that feeds into DMA scoring.

**Data Source Routing**:
- **Company profile data**: ERP Connector (DATA-003) for financials, employee counts, NACE classification
- **Supply chain data**: Supplier Questionnaire Processor (DATA-008) for value chain IRO identification
- **Sector benchmark data**: API Gateway (DATA-004) for industry benchmarks, peer DMA outcomes
- **Quality assurance**: Data Quality Profiler (DATA-010), Validation Rule Engine (DATA-019) for input validation
- **Data lineage**: Data Lineage Tracker (DATA-018) for provenance chain of DMA inputs

### 7.5 Sector Classification Bridge

**File**: `integrations/sector_classification_bridge.py`
**Purpose**: NACE sector classification for sector-specific DMA calibration and IRO template selection.

**Bridge Functions**:
- NACE code lookup and validation (4-digit level)
- Sector-to-topic relevance mapping (which ESRS topics are typically material per NACE sector)
- Sector-specific scale definitions and thresholds from ThresholdScoringEngine
- Multi-NACE support for conglomerates (separate DMA per division with group-level consolidation)
- EFRAG sector guidance integration (when available)
- Peer company identification for benchmarking

### 7.6 Regulatory Bridge

**File**: `integrations/regulatory_bridge.py`
**Purpose**: Monitor regulatory changes that may affect DMA outcomes and trigger reassessment.

**Bridge Functions**:
- Track ESRS amendment proposals (EFRAG, European Commission)
- Monitor Omnibus I/II implementation across Member States
- Flag regulatory changes affecting materiality thresholds (e.g., new CSRD scope, new penalty regimes)
- Trigger DMA Update workflow (Workflow 8) when material regulatory changes detected
- Maintain regulatory change log for DMA version management

### 7.7 Health Check

**File**: `integrations/health_check.py`
**Purpose**: 20-category system health verification covering all pack components.

**Health Check Categories**:
1. Pack manifest integrity
2. Configuration system and validation
3. Preset loading (all 6 presets)
4. Engine 1-8 availability and initialization
5. Workflow 1-8 readiness
6. Template 1-8 rendering capability
7. ESRS disclosure requirement database integrity (all datapoints loaded)
8. ESRS topic screening database integrity
9. MRV agent connectivity (30 agents via bridge)
10. DATA agent connectivity (20 agents via bridge)
11. FOUND agent connectivity (10 agents via bridge)
12. CSRD pack bridge status
13. Sector classification database availability
14. Regulatory change feed connectivity
15. Threshold scoring configuration validity
16. Scale definition completeness (all topics x all dimensions)
17. Database connectivity
18. Provenance hashing system operational
19. Export format generators operational
20. Overall system health score (0-100)

### 7.8 Setup Wizard

**File**: `integrations/setup_wizard.py`
**Purpose**: 7-step DMA-specific guided configuration wizard.

**Setup Steps**:
1. **Company Profile**: Legal entity name, NACE code(s), employee count, revenue, total assets, countries of operation, reporting year, parent/subsidiary structure
2. **Preset Selection**: Large enterprise / mid-market / SME / financial services / manufacturing / multi-sector; auto-loads sector-appropriate configuration
3. **Methodology Configuration**: Select scoring methodology (multiplicative/additive/matrix), set thresholds, configure time horizon weights, enable/disable human rights override
4. **Data Source Connection**: Connect to MRV agents (emissions data), DATA agents (company data), import prior-year DMA, connect financial systems
5. **Stakeholder Configuration**: Define stakeholder groups, set salience/influence, import consultation results
6. **Threshold Calibration**: Review and adjust materiality thresholds against sector benchmarks; run test assessment with demo data to validate configuration
7. **Go-Live Validation**: Run health check, validate demo data flows, confirm all engines operational, generate test DMA report

---

## 8. Configuration System

### 8.1 Pack Configuration

**File**: `config/pack_config.py`

**Configuration Hierarchy** (later overrides earlier):
1. Base `pack.yaml` manifest (defaults)
2. Preset YAML (large_enterprise / mid_market / sme / financial_services / manufacturing / multi_sector)
3. Environment overrides (DMA_PACK_* environment variables)
4. Explicit runtime overrides

**Top-Level Configuration Model** (`PackConfig`):
```python
class DMAMethodology(str, Enum):
    MULTIPLICATIVE = "multiplicative"  # Scale * Scope * Irremediability
    ADDITIVE = "additive"              # Weighted sum
    MATRIX = "matrix"                  # Lookup table
    LOGARITHMIC = "logarithmic"        # ln-scaled

class PackConfig(BaseModel):
    pack_id: str = "PACK-015-double-materiality"
    version: str = "1.0.0"
    reporting_year: int
    baseline_year: Optional[int] = None  # For YoY comparison

    # Company profile
    company_name: str
    nace_codes: List[str]            # Primary + secondary NACE codes
    employee_count: int
    annual_revenue_eur: float
    total_assets_eur: float
    countries_of_operation: List[str]

    # Methodology
    methodology: DMAMethodology = DMAMethodology.MULTIPLICATIVE
    impact_threshold: float = 50.0    # 0-100 scale
    financial_threshold: float = 40.0  # 0-100 scale
    human_rights_severity_only: bool = True  # ESRS 1 ss47

    # Engine configurations
    impact_materiality: ImpactMaterialityConfig
    financial_materiality: FinancialMaterialityConfig
    stakeholder_engagement: StakeholderEngagementConfig
    iro_identification: IROIdentificationConfig
    materiality_matrix: MaterialityMatrixConfig
    esrs_mapping: ESRSTopicMappingConfig
    threshold_scoring: ThresholdScoringConfig
    dma_report: DMAReportConfig

    # Feature flags
    omnibus_i_reduction: bool = True   # Apply 61% datapoint reduction
    stakeholder_weighting: bool = True # Weight scores by stakeholder salience
    sensitivity_analysis: bool = True  # Run threshold sensitivity
    yoy_comparison: bool = False       # Compare with prior year
    multi_sector_mode: bool = False    # Conglomerate DMA mode
```

### 8.2 Preset Specifications

#### 8.2.1 Large Enterprise Preset (`large_enterprise.yaml`)

**Target**: Companies with >5,000 employees and >EUR 1B revenue. Full ESRS scope.

**Configuration Overrides**:
- All engines enabled at full capability
- All 10 ESRS topics screened at sub-topic level (~100 sub-topics)
- Stakeholder weighting enabled with 4+ stakeholder group categories
- Sensitivity analysis enabled with 5-10-15% threshold variation
- YoY comparison enabled
- Full audit trail with SHA-256 provenance at every step
- Financial magnitude thresholds calibrated for large enterprise (e.g., Level 3 = 0.5-2% of revenue)

#### 8.2.2 Mid-Market Preset (`mid_market.yaml`)

**Target**: Companies with 1,000-5,000 employees and EUR 450M-1B revenue.

**Configuration Overrides**:
- All engines enabled at standard capability
- All 10 ESRS topics screened at topic level (10 topics, not 100 sub-topics)
- Stakeholder engagement: 2-3 stakeholder group categories
- Sensitivity analysis enabled at 10% variation only
- Financial magnitude thresholds calibrated for mid-market
- Simplified audit trail (provenance at phase boundaries, not every step)

#### 8.2.3 SME Preset (`sme.yaml`)

**Target**: Companies approaching Omnibus I threshold or conducting voluntary DMA.

**Configuration Overrides**:
- Simplified engines: Topic-level scoring only (no sub-topic granularity)
- IRO identification uses sector templates without custom discovery
- Stakeholder engagement: Single stakeholder group (management + key external)
- No sensitivity analysis
- No YoY comparison
- Simplified financial magnitude thresholds (3 levels instead of 5)
- Simplified audit trail (final scores with methodology reference)
- Guided data entry with industry-average defaults for missing inputs

#### 8.2.4 Financial Services Preset (`financial_services.yaml`)

**Target**: Banks, insurers, asset managers (NACE K64-K66) subject to CSRD + SFDR.

**Configuration Overrides**:
- S1, S2, G1 always material (regulatory expectation for financial sector)
- E1 financial materiality includes: financed emissions exposure, stranded asset risk, physical climate risk to loan/investment portfolio
- E4 biodiversity: TNFD LEAP approach overlay for nature-related financial risks
- Financial magnitude thresholds include: loan book exposure, AUM at risk, insurance claims exposure
- SFDR Principal Adverse Impact (PAI) indicators mapped to DMA topics
- EU Taxonomy alignment data feeds into E1/E2/E3/E4/E5 financial assessment
- Sector-specific IRO templates for: credit risk from climate transition, insurance underwriting risk, investment portfolio stranded assets

#### 8.2.5 Manufacturing Preset (`manufacturing.yaml`)

**Target**: Manufacturing companies (NACE C10-C33) with process emissions.

**Configuration Overrides**:
- E1 always material (process emissions, EU ETS exposure, CBAM)
- E2 often material (pollution, IED compliance)
- E5 often material (waste, circular economy, EPR)
- Financial magnitude includes: EU ETS compliance cost, CBAM certificate cost, BAT upgrade investment
- Process emissions data from MRV agents feeds directly into E1 impact scale scoring
- Sector-specific IRO templates for: carbon leakage risk, stranded industrial assets, technology transition (green hydrogen, CCS)

#### 8.2.6 Multi-Sector Preset (`multi_sector.yaml`)

**Target**: Conglomerates operating across multiple NACE divisions.

**Configuration Overrides**:
- `multi_sector_mode: true`
- Separate DMA per business division (NACE code), then consolidated group-level matrix
- Division-level thresholds may differ from group-level
- IRO identification per division with cross-division aggregation
- Group-level materiality determination: Topic is group-material if material in any division above significance threshold (e.g., >10% of group revenue)
- Consolidated matrix shows all divisions as separate markers per topic

---

## 9. Agent Dependencies

### 9.1 Summary

| Agent Layer | Count | Usage in PACK-015 |
|-------------|-------|-------------------|
| AGENT-MRV | 30 | Emissions data for E1/E2/E5 impact and financial scoring via MRV Materiality Bridge |
| AGENT-DATA | 20 | Data intake, quality, and lineage for all DMA inputs via Data Materiality Bridge |
| AGENT-FOUND | 10 | Orchestration, schema validation, units, citations, access control, audit trail |
| **Total** | **60** | **All agents accessed via integration bridges** |

### 9.2 MRV Agent Mapping (30 agents)

| MRV Agent | DMA Application | Priority |
|-----------|----------------|----------|
| MRV-001..008 Scope 1 agents | E1 impact scale (own operations emissions magnitude) | High |
| MRV-009..013 Scope 2 agents | E1 impact scale (purchased energy emissions) | High |
| MRV-014..030 Scope 3 agents | E1 impact scope (value chain emissions breadth) | High |
| MRV-004 Process Emissions | E2 pollution impact scoring context | Medium |
| MRV-005 Fugitive Emissions | E2 pollution impact scoring context | Medium |
| MRV-007 Waste Treatment | E5 circular economy impact scoring context | Medium |
| MRV-030 Audit Trail & Lineage | Provenance chain for DMA inputs from MRV | Critical |

### 9.3 DATA Agent Mapping (20 agents)

| DATA Agent | DMA Application |
|------------|----------------|
| DATA-001 PDF Extractor | Prior DMA documents, regulatory texts, stakeholder reports |
| DATA-002 Excel/CSV Normalizer | Stakeholder survey data, financial data imports |
| DATA-003 ERP Connector | Company financials (revenue, assets, EBITDA) for magnitude normalization |
| DATA-004 API Gateway | Sector benchmark data, regulatory feeds |
| DATA-008 Supplier Questionnaire | Value chain IRO identification inputs |
| DATA-009 Spend Categorizer | Financial exposure by sustainability topic |
| DATA-010 Data Quality Profiler | DMA input data quality scoring |
| DATA-018 Data Lineage Tracker | Full lineage from source to DMA score |
| DATA-019 Validation Rule Engine | DMA-specific validation rules (completeness, consistency) |

### 9.4 FOUND Agent Mapping (10 agents)

| FOUND Agent | DMA Application |
|-------------|----------------|
| FOUND-001 Orchestrator | DMA pipeline orchestration via Pack Orchestrator |
| FOUND-002 Schema Compiler | DMA data model validation (IRO schema, score schema) |
| FOUND-003 Unit Normalizer | Normalize financial units (EUR/USD/GBP), emissions units (tCO2e) |
| FOUND-004 Assumptions Registry | Track DMA assumptions (thresholds, weights, sector defaults) |
| FOUND-005 Citations Agent | Reference regulatory sources for each materiality decision |
| FOUND-006 Access Policy Guard | RBAC for DMA data (who can view/edit DMA, threshold sensitivity) |
| FOUND-007 Agent Registry | Service discovery for all agents used in DMA |
| FOUND-008 Reproducibility Agent | Verify DMA reproducibility (same inputs = same outputs) |
| FOUND-009 QA Test Harness | Automated DMA quality checks |
| FOUND-010 Observability Agent | DMA pipeline metrics and tracing |

---

## 10. Data Models

### 10.1 Core Data Models

```python
class ESRSTopic(str, Enum):
    E1 = "E1"   # Climate Change
    E2 = "E2"   # Pollution
    E3 = "E3"   # Water and Marine Resources
    E4 = "E4"   # Biodiversity and Ecosystems
    E5 = "E5"   # Resource Use and Circular Economy
    S1 = "S1"   # Own Workforce
    S2 = "S2"   # Workers in the Value Chain
    S3 = "S3"   # Affected Communities
    S4 = "S4"   # Consumers and End-Users
    G1 = "G1"   # Business Conduct

class IROType(str, Enum):
    IMPACT = "impact"
    RISK = "risk"
    OPPORTUNITY = "opportunity"
    COMBINED = "combined"

class IRONature(str, Enum):
    ACTUAL = "actual"
    POTENTIAL = "potential"

class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

class ValueChainStage(str, Enum):
    OWN_OPERATIONS = "own_operations"
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"

class TimeHorizon(str, Enum):
    SHORT = "short"      # <1 year
    MEDIUM = "medium"    # 1-5 years
    LONG = "long"        # >5 years

class MaterialityStatus(str, Enum):
    DOUBLE_MATERIAL = "double_material"       # Both axes above threshold
    IMPACT_MATERIAL = "impact_material"       # Impact only above threshold
    FINANCIAL_MATERIAL = "financial_material"  # Financial only above threshold
    NOT_MATERIAL = "not_material"             # Neither axis above threshold
    MONITORED = "monitored"                   # Below threshold but within sensitivity band

class SeverityScore(BaseModel):
    scale: int           # 1-5
    scope: int           # 1-5
    irremediability: int # 1-5
    severity_raw: float  # scale * scope * irremediability (1-125)
    severity_normalized: float  # Normalized to 0-100
    provenance_hash: str        # SHA-256

class IROEntry(BaseModel):
    iro_id: str
    esrs_topic: ESRSTopic
    esrs_sub_topic: str
    iro_type: IROType
    nature: IRONature
    polarity: Optional[Polarity]  # Only for impacts
    value_chain_stage: ValueChainStage
    time_horizon: TimeHorizon
    description: str
    business_activities: List[str]
    stakeholder_groups: List[str]
    is_human_rights: bool = False

    # Assessment scores (populated by engines)
    impact_score: Optional[float] = None       # 0-100
    financial_score: Optional[float] = None    # 0-100
    materiality_status: Optional[MaterialityStatus] = None

    # Audit
    provenance_hash: str
    assessed_date: Optional[str] = None
    assessor: Optional[str] = None

class MaterialityMatrixEntry(BaseModel):
    esrs_topic: ESRSTopic
    impact_score: float        # 0-100 (aggregated from IROs)
    financial_score: float     # 0-100 (aggregated from IROs)
    materiality_status: MaterialityStatus
    quadrant: int              # 1-4
    iro_count: int             # Number of IROs under this topic
    is_border_topic: bool      # Within sensitivity band of threshold
    provenance_hash: str
```

---

## 11. Testing Strategy

### 11.1 Test Files (18 files, 700+ tests target)

| Test File | Scope | Target Tests |
|-----------|-------|-------------|
| conftest.py | Shared fixtures: DMA company profiles, IRO data, stakeholder data, sector benchmarks | N/A (fixtures) |
| test_manifest.py | Pack YAML validation, version, structure | 60+ |
| test_config.py | Config system, preset loading, merge hierarchy, validation | 50+ |
| test_demo.py | Demo data smoke tests, demo config loading | 60+ |
| test_impact_materiality.py | Engine 1: severity scoring, likelihood, human rights override, time horizons | 55+ |
| test_financial_materiality.py | Engine 2: magnitude scoring, time horizon weights, risk/opportunity, scenarios | 55+ |
| test_stakeholder_engagement.py | Engine 3: identification, mapping, synthesis, weighting | 45+ |
| test_iro_identification.py | Engine 4: value chain mapping, IRO discovery, classification, register | 50+ |
| test_materiality_matrix.py | Engine 5: score aggregation, matrix generation, threshold application, sensitivity | 50+ |
| test_esrs_topic_mapping.py | Engine 6: disclosure mapping, Omnibus I reduction, gap analysis | 45+ |
| test_threshold_scoring.py | Engine 7: methodology variants, scale definitions, normalization | 45+ |
| test_dma_report.py | Engine 8: report assembly, audit trail, export formats | 40+ |
| test_workflows.py | All 8 workflows end-to-end with demo data | 35+ |
| test_templates.py | All 8 templates + template registry | 30+ |
| test_integrations.py | All 8 integrations | 25+ |
| test_e2e.py | End-to-end flows (5 scenarios) | 20+ |
| test_agent_integration.py | Agent wiring verification (MRV, DATA, FOUND bridges) | 15+ |
| **Total** | | **700+** |

### 11.2 Key Test Scenarios

**Scenario 1: Large Manufacturing Enterprise Full DMA**
- Company: GreenSteel Europa AG (synthetic), NACE C24.1, 8,000 employees, EUR 3B revenue
- Flow: Load large_enterprise preset with manufacturing overlay -> Import emissions data (500K tCO2e Scope 1) -> Map value chain (iron ore -> steelmaking -> distribution) -> Identify 45 IROs across E1, E2, E5, S1, S2 -> Score impact (E1 climate: scale 5, scope 4, irremediability 4 = high) -> Score financial (EU ETS exposure EUR 50M = Level 5) -> Generate matrix (E1, E2, S1 material) -> Map to ESRS disclosures (E1-1 through E1-9, E2-1 through E2-6, S1-1 through S1-17) -> Generate full DMA report with audit trail
- Expected: E1 double-material (Q1), E2 impact-material (Q2), S1 double-material (Q1), 7 ESRS standards requiring disclosure

**Scenario 2: Financial Services DMA with SFDR Overlay**
- Company: GreenBank Holding SE (synthetic), NACE K64.1, 3,000 employees, EUR 200B AUM
- Flow: Load financial_services preset -> Import portfolio emissions (financed emissions 2M tCO2e) -> Map value chain (deposits -> lending -> investment -> advisory) -> Identify 35 IROs with financial sector specifics -> Score E1 financial (stranded asset exposure EUR 5B, transition risk EUR 2B) -> Human rights S2 scoring for supply chain finance -> Generate matrix -> Map to ESRS + SFDR PAI overlay
- Expected: E1, S1, G1 double-material; E4 financial-material (nature-related portfolio risk); SFDR PAI indicators mapped

**Scenario 3: SME Simplified DMA**
- Company: BioTech Innovations KG (synthetic), NACE M72, 950 employees, EUR 400M revenue
- Flow: Load sme preset -> Simplified topic screening (10 topics only) -> Use sector default IRO templates -> Simplified scoring (3-level scale) -> Generate matrix -> Minimal ESRS mapping
- Expected: E1, S1, G1 material; simplified report; <5 minutes total

**Scenario 4: Multi-Sector Conglomerate DMA**
- Company: EuroGroup Holdings SE (synthetic), NACE C24 + G47 + K64, 25,000 employees, EUR 10B revenue
- Flow: Load multi_sector preset -> Separate DMA per division (manufacturing, retail, financial services) -> Division-level materiality per sector preset -> Consolidate to group-level matrix -> Group-material topics = material in any division >10% of group revenue
- Expected: Group has 7+ material topics; manufacturing drives E1/E2, retail drives E5/S4, financial drives G1; consolidated matrix shows division markers

**Scenario 5: DMA Update (Year 2)**
- Company: GreenSteel Europa AG (from Scenario 1), Year 2 update
- Flow: Load previous DMA -> Detect changes (new regulatory requirement, stakeholder feedback from Year 1 report, updated emissions data) -> Re-score only changed IROs (5 of 45) -> Delta analysis (E3 water moved from not-material to impact-material due to new water stress data) -> Generate update report with change log
- Expected: 1 topic changes materiality status; 5 IROs re-scored; full change log; <5 minutes

### 11.3 Test Infrastructure

- **Dynamic loading**: All tests use `importlib` dynamic loading (no package installation required)
- **Fixtures**: Shared conftest.py provides synthetic company profiles across sectors, pre-built IRO registers, stakeholder engagement data, financial baselines
- **Determinism verification**: Every test verifies SHA-256 provenance hashes -- identical test inputs must produce identical hashes across runs
- **Coverage target**: 85%+ line coverage across all engines, workflows, templates
- **CI integration**: Tests run in GitHub Actions via INFRA-007 CI/CD pipeline
- **Methodology matrix testing**: Each scoring methodology (multiplicative, additive, matrix, logarithmic) tested across all engines

---

## 12. Performance Requirements

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Single IRO impact scoring | <10ms | Time from input to scored output |
| Single IRO financial scoring | <10ms | Time from input to scored output |
| Full DMA (100 IROs, 10 topics) | <15 minutes | End-to-end Workflow 7 |
| Materiality matrix generation | <5 seconds | Engine 5 execution time |
| ESRS disclosure mapping | <10 seconds | Engine 6 execution time |
| DMA report generation | <30 seconds | Engine 8 execution time |
| DMA update (10% changed IROs) | <5 minutes | Workflow 8 execution time |
| Health check | <10 seconds | All 20 categories |
| Preset loading | <2 seconds | Config parse + validate |
| 100 concurrent DMA sessions | <30 minutes each | Load test scenario |

---

## 13. Security and Access Control

### 13.1 Data Sensitivity Classification

| Data Type | Classification | Access Control |
|-----------|---------------|----------------|
| DMA scores and materiality matrix | Confidential | Sustainability team + C-suite + auditors |
| IRO register | Confidential | Sustainability team + risk management |
| Stakeholder engagement data | Restricted | Sustainability team only |
| Financial magnitude data | Confidential | Sustainability team + finance |
| DMA methodology configuration | Internal | Sustainability team + admin |
| Published DMA report | Semi-public | Per CSRD disclosure rules |
| Audit working papers | Restricted | Auditors + sustainability team lead |

### 13.2 RBAC Integration

PACK-015 integrates with GreenLang SEC-002 RBAC system:
- `dma_admin`: Full configuration, threshold changes, methodology changes
- `dma_assessor`: Score IROs, run workflows, generate reports
- `dma_reviewer`: View scores, review reports, approve DMA (read + comment)
- `dma_auditor`: Full read access including provenance chain, no write
- `dma_stakeholder`: Submit stakeholder input, view own engagement data only

### 13.3 Audit Trail

- Every materiality score change logged with user, timestamp, old value, new value, rationale
- Threshold changes require dual approval (dma_admin + dma_reviewer)
- DMA report publication creates immutable snapshot (SHA-256 locked)
- All API calls to DMA engines logged via SEC-005 Centralized Audit Logging

---

## 14. Deployment and Installation

### 14.1 Pack Registration

PACK-015 registers in the GreenLang Solution Pack registry:
- Pack ID: `PACK-015-double-materiality`
- Category: `eu-compliance`
- Sector: `cross-sector`
- NACE: All (sector-agnostic)
- Tier: `standalone`
- Dependencies: None (standalone), Optional: `PACK-001/002/003` (CSRD bridge)
- Optional bridges: `PACK-012` (Financial Services), `PACK-013` (Manufacturing), `PACK-014` (Retail)

### 14.2 Database Migrations

No new database tables required for PACK-015 MVP. The pack uses existing tables from:
- FOUND agent tables (V021-V030) for orchestration, schema, audit
- DATA agent tables (V031-V050) for data quality and lineage
- Pack configuration stored in `pack_configurations` table (existing)

If DMA-specific reference data tables are needed (e.g., ESRS disclosure requirement database, sector relevance matrices, IRO templates), they will be added as V129+ migrations.

### 14.3 Infrastructure Requirements

| Resource | Requirement | Notes |
|----------|-------------|-------|
| Compute | 1 vCPU, 2 GB RAM (per pack instance) | DMA is less compute-intensive than MRV |
| Storage | 200 MB for ESRS reference data + sector databases | S3-backed |
| Database | Existing PostgreSQL + TimescaleDB | No additional DB needed |
| Cache | Existing Redis cluster | For sector benchmark caching |
| Network | Outbound HTTPS for regulatory feed | ESRS updates, EFRAG guidance |

### 14.4 Deployment Configuration

```yaml
# Kubernetes deployment snippet
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pack-015-double-materiality
  labels:
    app: greenlang
    component: solution-pack
    pack: double-materiality
spec:
  replicas: 2
  selector:
    matchLabels:
      pack: double-materiality
  template:
    spec:
      containers:
      - name: pack-015
        image: greenlang/pack-015-double-materiality:1.0.0
        resources:
          requests:
            cpu: "500m"
            memory: 1Gi
          limits:
            cpu: "1"
            memory: 2Gi
        env:
        - name: DMA_PACK_LOG_LEVEL
          value: "INFO"
        - name: DMA_PACK_PROVENANCE
          value: "true"
```

---

## 15. Success Metrics

### 15.1 Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test count | 700+ | pytest --count |
| Test pass rate | 100% | pytest exit code |
| Line coverage | 85%+ | coverage.py |
| Determinism | 100% | SHA-256 hash reproduction across 100 runs |
| Performance | All targets met | Benchmarks per Section 12 |
| Health check | 20/20 categories green | health_check.py output |

### 15.2 Product Metrics (Post-Launch)

| Metric | 30 Days | 60 Days | 90 Days |
|--------|---------|---------|---------|
| Active DMA assessments | 20 | 50 | 100 |
| Reports generated | 40 | 150 | 400 |
| Assurance engagements using PACK-015 audit report | 5 | 15 | 30 |
| Average time to complete full DMA | <15 min | <12 min (optimization) | <10 min |
| Data quality score (input data) | >80% | >85% | >90% |
| Customer satisfaction (NPS) | >40 | >50 | >60 |

### 15.3 Regulatory Compliance Metrics

| Metric | Target |
|--------|--------|
| ESRS 1 Chapter 3 compliance | 100% of requirements covered |
| ESRS 2 IRO-1/IRO-2/SBM-3 compliance | 100% of disclosure requirements producible |
| Omnibus I datapoint reduction correctly applied | 100% accuracy |
| EFRAG IG-1 guidance alignment | All recommended practices implemented |
| Auditor acceptance rate | >90% of DMA reports accepted without material findings |

---

## 16. Timeline

### Phase 1: Foundation (Weeks 1-4)

| Week | Deliverable | Owner |
|------|------------|-------|
| 1 | PRD finalization, architecture review, pack scaffold, ESRS reference data compilation | Product + Arch |
| 2 | pack_config.py, all 6 presets, demo_config.yaml, ThresholdScoringEngine (Engine 7) | Config + Engine engineer |
| 3 | IROIdentificationEngine (Engine 4), ImpactMaterialityEngine (Engine 1) | Engine engineer |
| 4 | FinancialMaterialityEngine (Engine 2), StakeholderEngagementEngine (Engine 3) | Engine engineer |

**Milestone**: Core scoring engines operational with demo data.

### Phase 2: Matrix, Mapping, and Reporting (Weeks 5-8)

| Week | Deliverable | Owner |
|------|------------|-------|
| 5 | MaterialityMatrixEngine (Engine 5), ESRSTopicMappingEngine (Engine 6) | Engine engineer |
| 6 | DMAReportEngine (Engine 8), all 8 templates | Engine + Template engineer |
| 7 | All 8 workflows implemented and tested with demo data | Workflow engineer |
| 8 | ESRS disclosure requirement database (all datapoints), Omnibus I reduction mapping | Data engineer |

**Milestone**: All engines, workflows, and templates complete. Full DMA producible.

### Phase 3: Integrations and Testing (Weeks 9-12)

| Week | Deliverable | Owner |
|------|------------|-------|
| 9 | Pack orchestrator, CSRD bridge, MRV bridge, DATA bridge | Integration engineer |
| 10 | Sector classification bridge, regulatory bridge, health check, setup wizard | Integration engineer |
| 11 | All unit tests (700+ target), integration tests, determinism verification | QA |
| 12 | E2E testing (5 scenarios), performance testing, security review, documentation | QA + DevOps |

**Milestone**: Pack launch-ready with 700+ tests, 100% pass rate.

### Phase 4: Beta and GA (Weeks 13-16)

| Week | Deliverable | Owner |
|------|------------|-------|
| 13 | Beta deployment to 5 pilot customers (1 manufacturing, 1 financial, 1 retail, 1 mid-market, 1 multi-sector) | Product + CS |
| 14 | Beta feedback integration, bug fixes, threshold calibration with real-world data | Engineering |
| 15 | GA readiness review, auditor walkthrough, final documentation | Product + QA |
| 16 | General Availability release | All teams |

**Milestone**: GA release.

---

## 17. Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Omnibus II further changes DMA requirements | Medium | High | Modular methodology engine (Engine 7) allows threshold/methodology changes via config; monitor EC legislative pipeline |
| EFRAG publishes sector-specific DMA guidance that conflicts with our approach | Medium | Medium | ThresholdScoringEngine supports per-sector calibration; sector presets updateable without code changes |
| Auditors reject deterministic scoring as insufficiently "judgment-based" | Low | High | Report includes full methodology documentation; supports manual score overrides with audit trail; present to Big 4 sustainability assurance teams during beta |
| Companies cannot quantify financial magnitude for all topics | High | Medium | Allow qualitative-to-quantitative proxy estimation with lower DQ score; provide industry benchmark proxies; flag as "estimated" in audit trail |
| Stakeholder engagement data quality varies wildly | High | Medium | Structured input templates; validation rules for survey data; weight adjustments based on engagement quality score |
| Multi-sector conglomerate DMA complexity exceeds single assessment | Medium | Medium | Multi-sector preset handles division-level DMA + group consolidation; tested with 5-division synthetic conglomerate |
| ESRS disclosure requirement database becomes stale (amendments, corrections) | Medium | Medium | Regulatory bridge monitors EFRAG/EC publications; versioned reference data with update alerts |
| Performance degrades for very large IRO registers (>500 IROs) | Low | Low | IRO scoring is O(n) with <10ms per IRO; 500 IROs = 5 seconds total; parallel execution for impact and financial |
| Competing DMA tools gain market share while we build | Medium | High | 70% platform reuse (agents already built); unique differentiator = deterministic scoring + full auditability; 16-week timeline is aggressive but achievable |
| Threshold selection influences materiality outcome (gaming risk) | Medium | High | Sensitivity analysis reveals border topics; mandatory disclosure of chosen thresholds; audit trail for threshold changes with dual approval |

---

## 18. Acceptance Criteria

1. All 8 engines implement deterministic calculations with SHA-256 provenance hashing
2. All Pydantic v2 models with field_validator/model_validator (NO `from __future__ import annotations`)
3. All 6 presets load and validate without errors
4. All 8 workflows complete end-to-end with demo data
5. All 8 templates generate valid output in all export formats
6. All 8 integrations pass health check (20/20 categories green)
7. 700+ unit tests, 100% pass rate
8. Cross-pack bridges verify connectivity to PACK-001/002/003 (CSRD packs)
9. ESRS 1 Chapter 3 compliance: All requirements mapped and implemented
10. ESRS 2 IRO-1/IRO-2/SBM-3 outputs producible from DMA results
11. Omnibus I 61% datapoint reduction correctly applied in ESRS mapping
12. Demo mode: GreenSteel Europa AG synthetic company produces complete DMA in <15 minutes
13. Sensitivity analysis correctly identifies border topics within +/- 10% of thresholds
14. Human rights severity-only override functions per ESRS 1 ss47
15. DMA Update workflow correctly detects changes and re-scores only affected IROs

---

## 19. Appendices

### Appendix A: ESRS 1 Appendix A -- Sustainability Matters

| ESRS | Topic | Sub-Topics |
|------|-------|------------|
| E1 | Climate Change | Climate change mitigation, Climate change adaptation, Energy |
| E2 | Pollution | Pollution of air, Pollution of water, Pollution of soil, Pollution of living organisms and food resources, Substances of concern, Substances of very high concern, Microplastics |
| E3 | Water and Marine Resources | Water (withdrawals, consumption, discharges), Marine resources (extraction, use) |
| E4 | Biodiversity and Ecosystems | Direct impact drivers (climate change, land use, exploitation, pollution, invasive species), Impacts on state of species, Impacts on ecosystems, Ecosystem services dependencies |
| E5 | Resource Use and Circular Economy | Resource inflows (material resources), Resource outflows (products, services, waste) |
| S1 | Own Workforce | Working conditions (secure employment, working time, adequate wages, social dialogue, work-life balance, health and safety), Equal treatment (gender equality, training, diversity, disability inclusion), Other work-related rights (child labour, forced labour, privacy, freedom of association) |
| S2 | Workers in the Value Chain | Same sub-topics as S1, applied to value chain workers |
| S3 | Affected Communities | Economic, social, cultural rights; Civil and political rights; Rights of indigenous peoples |
| S4 | Consumers and End-Users | Information-related impacts (privacy, freedom of expression, access to information), Personal safety (health and safety, security of person), Social inclusion (non-discrimination, access to products/services, responsible marketing) |
| G1 | Business Conduct | Corporate culture, Protection of whistle-blowers, Animal welfare, Political engagement and lobbying, Management of relationships with suppliers, Corruption and bribery |

### Appendix B: Glossary

| Term | Definition |
|------|-----------|
| CSRD | Corporate Sustainability Reporting Directive (EU) 2022/2464 |
| DMA | Double Materiality Assessment |
| EFRAG | European Financial Reporting Advisory Group |
| ESRS | European Sustainability Reporting Standards |
| IRO | Impact, Risk, Opportunity |
| NACE | Statistical Classification of Economic Activities in the European Community |
| Omnibus I | Directive (EU) 2026/470 revising CSRD thresholds and ESRS datapoints |
| PAI | Principal Adverse Impact (SFDR terminology) |
| SFDR | Sustainable Finance Disclosure Regulation |
| SHA-256 | Secure Hash Algorithm 256-bit (cryptographic hash for provenance) |
| TNFD | Taskforce on Nature-related Financial Disclosures |

### Appendix C: References

- ESRS 1 General Requirements (Delegated Regulation (EU) 2023/2772, Annex I)
- ESRS 2 General Disclosures (Delegated Regulation (EU) 2023/2772, Annex II)
- EFRAG Implementation Guidance 1: Materiality Assessment (EFRAG IG-1, 2024)
- EFRAG Implementation Guidance 2: Value Chain (EFRAG IG-2, 2024)
- EFRAG Implementation Guidance 3: ESRS Datapoints (EFRAG IG-3, 2024)
- Omnibus I Directive (EU) 2026/470
- GreenLang Platform Architecture Documentation

---

**Approval Signatures:**

- Product Manager: ___________________
- Engineering Lead: ___________________
- Regulatory Lead: ___________________
- CEO: ___________________
