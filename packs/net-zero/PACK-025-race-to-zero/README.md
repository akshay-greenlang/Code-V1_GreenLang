# PACK-025: Race to Zero Pack

**Comprehensive Race to Zero campaign lifecycle management powered by GreenLang AI agents**

## Overview

PACK-025 provides a complete, standalone Race to Zero campaign lifecycle
management solution covering the full journey from initial pledge commitment
through ongoing annual participation with credibility assurance. It implements
the requirements of the UNFCCC Race to Zero Campaign (2020, updated 2022),
the Race to Zero Interpretation Guide (June 2022), and the UN High-Level
Expert Group (HLEG) "Integrity Matters" report (November 2022).

The Race to Zero campaign is the UN-backed global initiative rallying non-state
actors -- corporations, financial institutions, cities, regions, universities,
and healthcare organizations -- to take rigorous, immediate action to halve
global emissions by 2030 and achieve net zero by 2050 at the latest. The
campaign mobilizes 11,000+ entities representing approximately 25% of global
CO2 emissions and 50% of GDP, operating through 40+ partner initiatives
including SBTi, CDP, C40, ICLEI, GFANZ, SME Climate Hub, and Under2 Coalition.

### Starting Line Criteria (4Ps)

Race to Zero participation requires meeting the four Starting Line Criteria,
known as the "4Ps":

1. **Pledge** -- Commit to reaching net-zero GHG emissions by 2050 at the
   latest, with an interim target covering all scopes.
2. **Plan** -- Publish a climate action plan with quantified actions and
   milestones within 12 months of pledging.
3. **Proceed** -- Take immediate action toward emission reductions consistent
   with the plan and interim target.
4. **Publish** -- Report progress annually through partner initiative
   reporting channels.

Each pillar has 5 sub-criteria (20 total) defined in the June 2022
Interpretation Guide.

| Metric | Value |
|--------|-------|
| Engines | 10 |
| Workflows | 8 |
| Templates | 10 |
| Integrations | 12 |
| Presets | 8 |
| Total Python files | 57 |
| YAML presets | 8 |
| Lines of code | ~50K |
| Tests | 797 |
| MRV agents | 30 (all Scope 1/2/3) |
| DATA agents | 20 (intake + quality) |
| Foundation agents | 10 (platform services) |
| Actor types | 8 (Corporate, Financial, City, Region, SME, Heavy Industry, Services, Manufacturing) |
| Partner initiatives | 40+ (7 primary: SBTi, CDP, C40, ICLEI, GFANZ, SME Climate Hub, Under2) |
| Sector pathways | 25+ (IEA, IPCC, TPI, MPP, ACT, CRREM) |
| HLEG recommendations | 10 (45+ sub-criteria) |

## Architecture

```
+-----------------------------------------------------------------------+
|                     PACK-025 Race to Zero Pack                        |
+-----------------------------------------------------------------------+
|                                                                       |
|  +-------------------+   +-------------------+   +------------------+ |
|  | Pledge            |   | Starting Line     |   | Interim Target   | |
|  | Commitment        |-->| Engine            |-->| Engine           | |
|  | Engine            |   | (4P, 20 criteria) |   | (1.5C pathway)   | |
|  +-------------------+   +-------------------+   +------------------+ |
|          |                        |                       |           |
|          v                        v                       v           |
|  +-------------------+   +-------------------+   +------------------+ |
|  | Action Plan       |   | Progress          |   | Sector Pathway   | |
|  | Engine            |-->| Tracking          |-->| Engine           | |
|  | (MACC, levers)    |   | Engine            |   | (25+ sectors)    | |
|  +-------------------+   +-------------------+   +------------------+ |
|          |                        |                       |           |
|          v                        v                       v           |
|  +-------------------+   +-------------------+   +------------------+ |
|  | Partnership       |   | Campaign          |   | Credibility      | |
|  | Scoring           |-->| Reporting         |-->| Assessment       | |
|  | Engine            |   | Engine            |   | Engine (HLEG)    | |
|  +-------------------+   +-------------------+   +------------------+ |
|                                                          |           |
|                                        +------------------+          |
|                                        | Race Readiness   |          |
|                                        | Engine (0-100)   |          |
|                                        +------------------+          |
|                                                                       |
+-----+---------+---------+---------+--------+--------+--------+-------+
      |         |         |         |        |        |        |
      v         v         v         v        v        v        v
  30 MRV    20 DATA   10 FOUND  GL-GHG   GL-SBTi  GL-CDP  GFANZ
  Agents    Agents    Agents    APP      APP      APP     Framework
```

### 10-Phase DAG Pipeline

The Pack Orchestrator executes a directed acyclic graph across 10 phases:

```
Phase 1: Health Check (22-category system verification)
    |
    v
Phase 2: Configuration (preset loading, actor-type setup)
    |
    v
Phase 3: Pledge Commitment <-- Eligibility validation (8 criteria)
    |
    v
Phase 4: Starting Line Assessment <-- 4P compliance (20 sub-criteria)
    |
    v
Phase 5: Interim Target Validation <-- 1.5C pathway (43% by 2030)
    |
    +----------+
    |          |
    v          v
Phase 6:    Phase 7:
Action      Sector           <-- IEA NZE, IPCC AR6, TPI, MPP, ACT, CRREM
Plan        Pathway
    |          |
    +----+-----+
         |
         v
Phase 8: Partnership Scoring <-- 40+ partner initiatives
    |
    v
Phase 9: Campaign Reporting + Credibility Assessment (HLEG 10 recs)
    |
    v
Phase 10: Race Readiness Scoring (8 dimensions, 0-100 composite)
    |
    v
  [COMPLETE] --> SHA-256 provenance hash
```

## Quick Start

```python
from packs.net_zero.PACK_025_race_to_zero import (
    PledgeCommitmentEngine,
    StartingLineEngine,
    RaceReadinessEngine,
    get_preset_path,
    AVAILABLE_PRESETS,
)

# 1. View available presets
print(f"Available presets: {sorted(AVAILABLE_PRESETS.keys())}")
# ['city_municipality', 'corporate_commitment', 'financial_institution',
#  'high_emitter', 'manufacturing_sector', 'region_state',
#  'service_sector', 'sme_business']

# 2. Get preset path for a specific actor type
preset_path = get_preset_path("corporate_commitment")
print(f"Preset: {preset_path}")

# 3. Initialize an engine
pledge_engine = PledgeCommitmentEngine()
result = pledge_engine.assess(
    organization_id="org-001",
    actor_type="CORPORATE",
    net_zero_target_year=2050,
    interim_target_year=2030,
    interim_reduction_pct=50.0,
    scope_coverage=["scope_1", "scope_2", "scope_3"],
    partner_initiative="SBTi",
)
print(f"Pledge quality: {result.quality_rating}")
print(f"Eligibility: {result.eligibility_status}")
```

## Installation

### Prerequisites

- Python >= 3.11
- PostgreSQL >= 16 with pgvector and TimescaleDB extensions
- Redis >= 7
- GreenLang Platform >= 2.0.0

### Install Steps

```bash
# 1. Verify Python version
python --version  # Requires 3.11+

# 2. Install Python dependencies
pip install pydantic>=2.0 pyyaml>=6.0 pandas>=2.0 numpy>=1.24 \
    httpx>=0.24 psycopg[binary]>=3.1 psycopg_pool>=3.1 redis>=5.0 \
    jinja2>=3.1 openpyxl>=3.1 cryptography>=41.0

# 3. Apply database migrations
#    Inherited: V001-V006, V007-V008, V009-V010, V019-V020,
#               V021-V030, V031-V050, V051-V081, V082-V088
#    New: V148 through V157 (10 migration files for PACK-025)
#
#    V148: Schema + organization profiles + pledge commitments + partner memberships
#    V149: Starting line assessments + starting line gaps
#    V150: Interim targets + action plans + abatement actions
#    V151: Annual reports (trajectory, verification, multi-channel)
#    V152: Sector pathways (IEA, IPCC, TPI, MPP, ACT, CRREM)
#    V153: Partnership collaborations (7 types, 5 engagement levels)
#    V154: Credibility assessments (HLEG 10 recommendations)
#    V155: Campaign submissions + readiness scores
#    V156: Audit trail + workflow executions
#    V157: Views (pledge summary, progress timeline, partner overview) + RLS

# 4. Seed reference data
#    - Race to Zero campaign criteria
#    - Starting Line 20 sub-criteria (June 2022 Interpretation Guide)
#    - HLEG 10 recommendations and 45+ sub-criteria
#    - Partner initiative registry (40+ initiatives)
#    - Sector pathway benchmarks (25+ sectors)
#    - Emission factor databases (DEFRA, EPA, ecoinvent)
#    - IPCC AR6 pathway data
#    - IEA NZE sector milestones
#    - Actor type configurations (8 types)
#    - Default presets

# 5. Run health check
python -c "
from packs.net_zero.PACK_025_race_to_zero.integrations import RaceToZeroHealthCheck
hc = RaceToZeroHealthCheck()
result = hc.run()
print(f'Health: {result.overall_status}')
print(f'Categories checked: {result.categories_checked}/22')
"
```

### Infrastructure Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU cores | 4 | 8 |
| Memory (GB) | 8 | 16 |
| Storage (GB) | 50 | 200 |
| DB connections | 20 | 60 |

## Component Overview

### Engines (10)

| # | Engine | Description | Standard/Framework |
|---|--------|-------------|--------------------|
| 1 | Pledge Commitment | Validates 8 eligibility criteria with 4 quality levels (Strong/Adequate/Weak/Ineligible) | Race to Zero Campaign |
| 2 | Starting Line | Assesses 4 pillars (Pledge/Plan/Proceed/Publish) with 20 sub-criteria | Interpretation Guide (June 2022) |
| 3 | Interim Target | Validates 2030 targets against 1.5C pathway (43% by 2030, 4.2%/yr) | IPCC AR6, SBTi Net-Zero |
| 4 | Action Plan | Generates quantified plans with MACC curves, milestones, and costs | HLEG, Interpretation Guide |
| 5 | Progress Tracking | Annual trajectory alignment (ON_TRACK/SLIGHTLY_OFF/SIGNIFICANTLY_OFF/REVERSED) | Race to Zero, GHG Protocol |
| 6 | Sector Pathway | Maps to 25+ sector pathways from IEA, IPCC, TPI, MPP, ACT, CRREM | IEA NZE, TPI, MPP |
| 7 | Partnership Scoring | Scores engagement across 40+ partner initiatives with 5 quality dimensions | Race to Zero Partnership |
| 8 | Campaign Reporting | Generates multi-partner reports (CDP, GFANZ, C40, ICLEI formats) | Partner reporting standards |
| 9 | Credibility Assessment | HLEG "Integrity Matters" 10 recommendations with 45+ sub-criteria (A-F rating) | HLEG Report (Nov 2022) |
| 10 | Race Readiness | 8-dimension composite score (0-100) with RAG status and readiness level | Composite framework |

### Workflows (8)

| # | Workflow | Phases | Schedule | Duration |
|---|----------|--------|----------|----------|
| 1 | Pledge Onboarding | 5 | On demand | 45 min |
| 2 | Starting Line Assessment | 4 | On demand | 60 min |
| 3 | Action Planning | 6 | On demand | 90 min |
| 4 | Annual Reporting | 7 | Annual | 60 min |
| 5 | Sector Pathway | 5 | On demand | 30 min |
| 6 | Partnership Engagement | 5 | On demand | 30 min |
| 7 | Credibility Review | 4 | Annual | 60 min |
| 8 | Full Race to Zero | 10 | Annual | 180 min |

### Templates (10)

| # | Template | Format | Description |
|---|----------|--------|-------------|
| 1 | Pledge Commitment Letter | PDF | Formal pledge for campaign secretariat submission |
| 2 | Starting Line Checklist | HTML | 20 sub-criteria compliance evidence mapping |
| 3 | Action Plan Document | PDF | MACC-based quantified plan with milestones |
| 4 | Annual Progress Report | HTML | Year-over-year trajectory and variance analysis |
| 5 | Sector Pathway Roadmap | HTML | Sector benchmark alignment visualization |
| 6 | Partnership Framework | HTML | Partner requirement crosswalk and synergy analysis |
| 7 | Credibility Assessment Report | HTML | HLEG 10-recommendation scoring matrix |
| 8 | Campaign Submission Package | PDF | Consolidated submission-ready package |
| 9 | Disclosure Dashboard | HTML | Real-time compliance status across all dimensions |
| 10 | Race to Zero Certificate | PDF | Composite readiness score with SHA-256 provenance |

### Integrations (12)

| # | Integration | Description |
|---|-------------|-------------|
| 1 | Pack Orchestrator | 10-phase DAG pipeline with retry and SHA-256 provenance |
| 2 | MRV Bridge | Routes to all 30 MRV agents (Scope 1/2/3) |
| 3 | GHG App Bridge | Connects to GL-GHG-APP v1.0 for inventory and base year |
| 4 | SBTi App Bridge | Connects to GL-SBTi-APP for target cross-validation |
| 5 | DECARB Bridge | Routes to 21 DECARB-X agents for reduction planning |
| 6 | EU Taxonomy Bridge | Connects to GL-Taxonomy-APP for CapEx alignment |
| 7 | Data Bridge | Connects to all 20 DATA agents for intake/quality |
| 8 | UNFCCC Bridge | Race to Zero campaign portal integration |
| 9 | CDP Bridge | CDP Climate Change questionnaire mapping |
| 10 | GFANZ Bridge | Financial institution transition plan framework |
| 11 | Setup Wizard | 8-step guided configuration wizard |
| 12 | Health Check | 22-category system verification |

### Presets (8)

| Preset | Actor Type | Primary Partner | Scope Focus | Key Features |
|--------|-----------|-----------------|-------------|--------------|
| `corporate_commitment` | CORPORATE | SBTi, CDP | 1+2+3 | Full HLEG, SDA/ACA, 25+ sector pathways |
| `financial_institution` | FINANCIAL_INSTITUTION | GFANZ | 1+2+3 (financed) | PCAF, portfolio temperature, GFANZ transition plan |
| `city_municipality` | CITY | C40, ICLEI | Community-wide | GPC methodology, city-scale benchmarks |
| `region_state` | REGION | Under2 Coalition | Sub-national | LULUCF, 15-year horizon, policy lever identification |
| `sme_business` | SME | SME Climate Hub | 1+2 (simplified) | 6-engine flow, spend-based Scope 3, streamlined |
| `high_emitter` | HEAVY_INDUSTRY | SBTi (SDA) | 1+2+3 (process) | MPP pathways, CCS/CCUS, fossil fuel phase-out |
| `service_sector` | SERVICES | SBTi (ACA) | 3 (Cat 1,6,7) | RE100, cloud emissions, remote work accounting |
| `manufacturing_sector` | MANUFACTURING | SBTi | 1+2+3 | Product carbon footprint, supplier engagement |

## Usage Examples

### Corporate Pledge Onboarding

```python
from packs.net_zero.PACK_025_race_to_zero.engines import (
    PledgeCommitmentEngine,
    StartingLineEngine,
    InterimTargetEngine,
)

# Step 1: Validate pledge eligibility
pledge_engine = PledgeCommitmentEngine()
pledge_result = pledge_engine.assess(
    organization_id="org-001",
    actor_type="CORPORATE",
    net_zero_target_year=2050,
    interim_target_year=2030,
    interim_reduction_pct=50.0,
    scope_coverage=["scope_1", "scope_2", "scope_3"],
    partner_initiative="SBTi",
    governance_endorsement=True,
    public_disclosure_commitment=True,
)
print(f"Eligibility: {pledge_result.eligibility_status}")
print(f"Quality: {pledge_result.quality_rating}")
# Eligibility: ELIGIBLE
# Quality: STRONG

# Step 2: Assess Starting Line Criteria compliance
starting_line = StartingLineEngine()
sl_result = starting_line.assess(
    organization_id="org-001",
    pledge_data=pledge_result,
    evidence_documents=evidence_list,
)
print(f"Pledge pillar: {sl_result.pledge_score}/100")
print(f"Plan pillar: {sl_result.plan_score}/100")
print(f"Proceed pillar: {sl_result.proceed_score}/100")
print(f"Publish pillar: {sl_result.publish_score}/100")
print(f"Overall compliance: {sl_result.overall_compliance}")

# Step 3: Validate interim target against 1.5C pathway
target_engine = InterimTargetEngine()
target_result = target_engine.validate(
    organization_id="org-001",
    base_year=2019,
    target_year=2030,
    reduction_pct=50.0,
    scope_coverage=["scope_1", "scope_2", "scope_3"],
    pathway_source="IPCC_AR6",
    methodology="SDA",
)
print(f"1.5C aligned: {target_result.is_aligned}")
print(f"Annual reduction rate: {target_result.annual_rate_pct:.1f}%")
```

### HLEG Credibility Assessment

```python
from packs.net_zero.PACK_025_race_to_zero.engines import (
    CredibilityAssessmentEngine,
)

credibility = CredibilityAssessmentEngine()
result = credibility.assess(
    organization_id="org-001",
    # R1: Net-zero pledge quality
    pledge_specificity="DETAILED",
    pledge_boundary="FULL_VALUE_CHAIN",
    # R2: Interim target ambition
    interim_reduction_pct=50.0,
    target_methodology="SDA",
    # R3: Voluntary credit use
    credits_used_for_abatement=True,
    offset_quality_score=85.0,
    # R4: Lobbying alignment
    trade_association_review=True,
    policy_advocacy_consistent=True,
    # R5: Just transition
    social_impact_assessed=True,
    stakeholder_engagement=True,
    # R6: Financial commitment
    green_capex_pct=35.0,
    # R7: Reporting transparency
    annual_disclosure=True,
    methodology_documented=True,
    # R8: Scope of pledge
    full_value_chain=True,
    # R9: Internal governance
    board_oversight=True,
    incentive_alignment=True,
    # R10: Fossil fuel phase-out
    no_new_fossil_capacity=True,
    divestment_plan=True,
)
print(f"Overall rating: {result.overall_rating}")       # A-F
print(f"Credibility score: {result.composite_score}")    # 0-100
for rec in result.recommendations:
    print(f"  R{rec.number}: {rec.title} = {rec.score}/100 ({rec.status})")
```

### Financial Institution (GFANZ Pathway)

```python
from packs.net_zero.PACK_025_race_to_zero.integrations import GFANZBridge

gfanz = GFANZBridge()

# Assess financed emissions portfolio
portfolio = gfanz.assess_portfolio(
    institution_id="fi-001",
    asset_classes=["corporate_bonds", "listed_equity", "project_finance"],
    pcaf_methodology=True,
    reporting_year=2025,
)
print(f"Financed emissions: {portfolio.total_financed_tco2e:,.0f} tCO2e")
print(f"Portfolio temperature: {portfolio.temperature_score:.1f}C")
print(f"GFANZ alignment tier: {portfolio.alignment_tier}")
```

### City/Municipality (C40 Pathway)

```python
from packs.net_zero.PACK_025_race_to_zero.workflows import (
    PledgeOnboardingWorkflow,
)

workflow = PledgeOnboardingWorkflow()
result = workflow.execute(
    organization_id="city-001",
    actor_type="CITY",
    partner_initiative="C40",
    inventory_methodology="GPC",
    sectors=["transport", "buildings", "waste", "energy"],
    population=500_000,
    net_zero_year=2050,
    interim_year=2030,
    interim_reduction_pct=45.0,
)
print(f"Onboarding phases completed: {result.phases_completed}/{result.phases_total}")
print(f"Pledge status: {result.pledge_status}")
```

### Running the Full Race to Zero Workflow

```python
from packs.net_zero.PACK_025_race_to_zero.integrations import (
    RaceToZeroOrchestrator,
)

orchestrator = RaceToZeroOrchestrator()
result = orchestrator.run_pipeline(
    organization_id="org-001",
    actor_type="CORPORATE",
    preset="corporate_commitment",
    reporting_year=2025,
)
print(f"Pipeline status: {result.status}")
print(f"Phases completed: {result.phases_completed}/{result.phases_total}")
print(f"Readiness score: {result.readiness_score}/100")
print(f"RAG status: {result.rag_status}")         # RED / AMBER / GREEN
print(f"Readiness level: {result.readiness_level}")  # Not Ready / Partially / Ready / Exemplary
```

### Annual Reporting Workflow

```python
from packs.net_zero.PACK_025_race_to_zero.workflows import (
    AnnualReportingWorkflow,
)

reporting = AnnualReportingWorkflow()
result = reporting.execute(
    organization_id="org-001",
    reporting_year=2025,
    submission_channels=["CDP", "SBTi"],
)
print(f"Trajectory status: {result.trajectory_status}")
print(f"YoY reduction: {result.yoy_reduction_pct:.1f}%")
print(f"Reports generated: {len(result.reports)}")
for report in result.reports:
    print(f"  {report.channel}: {report.format} ({report.status})")
```

### Sector Pathway Alignment

```python
from packs.net_zero.PACK_025_race_to_zero.engines import SectorPathwayEngine

sector = SectorPathwayEngine()
result = sector.assess(
    organization_id="org-001",
    sector="steel",
    current_intensity_tco2e_per_tonne=1.85,
    pathway_sources=["IEA_NZE", "MPP", "TPI"],
)
print(f"Gap to 2030 benchmark: {result.gap_2030_pct:.1f}%")
print(f"Alignment status: {result.alignment_status}")
for milestone in result.milestones:
    print(f"  {milestone.year}: {milestone.description}")
```

## Configuration Guide

### Configuration Hierarchy

Configuration is resolved in order of increasing priority:

1. **Base `pack.yaml`** -- Pack manifest with component definitions
2. **Preset YAML** -- Actor-type preset (e.g., `corporate_commitment.yaml`)
3. **Environment variables** -- Override with `RACE_TO_ZERO_*` prefix
4. **Runtime overrides** -- Explicit programmatic overrides

### Environment Variable Overrides

```bash
# Organization
export RACE_TO_ZERO_ORG_NAME="Acme Corp"
export RACE_TO_ZERO_ORG_SECTOR="steel"
export RACE_TO_ZERO_ACTOR_TYPE="CORPORATE"

# Targets
export RACE_TO_ZERO_BASE_YEAR=2019
export RACE_TO_ZERO_INTERIM_YEAR=2030
export RACE_TO_ZERO_NET_ZERO_YEAR=2050
export RACE_TO_ZERO_INTERIM_REDUCTION_PCT=50

# Partner
export RACE_TO_ZERO_PRIMARY_PARTNER="SBTi"

# Pathway
export RACE_TO_ZERO_PATHWAY_SOURCE="IPCC_AR6"
export RACE_TO_ZERO_PATHWAY_METHOD="SDA"

# Reporting
export RACE_TO_ZERO_REPORTING_CHANNELS="CDP,SBTi"
```

### Preset Reference

#### `corporate_commitment` (Default)

Large corporates (>1000 employees) joining through SBTi with CDP reporting.
All 10 engines enabled, full Scope 1+2+3 coverage, SDA or ACA pathway with
4.2%/yr minimum reduction rate, full HLEG 10-recommendation assessment,
25+ sector pathways available, 10-year planning horizon.

#### `financial_institution`

Banks, insurers, and asset managers joining through GFANZ. PCAF financed
emissions methodology, portfolio temperature scoring, Scope 3 Category 15
(investments) focus, GFANZ transition plan format output.

#### `city_municipality`

Cities and municipalities joining through C40 or ICLEI. Community-wide GPC
inventory methodology, transport/buildings/waste/energy sector focus, C40
Deadline 2020 pathway benchmarks.

#### `region_state`

Regions, states, and provinces joining through Under2 Coalition. Sub-national
inventory with LULUCF inclusion, energy and land use sector focus, 15-year
planning horizon, policy lever identification.

#### `sme_business`

SMEs (<250 employees) joining through SME Climate Hub. Simplified 6-engine
flow (pledge, starting_line, interim_target, action_plan, progress_tracking,
race_readiness), spend-based Scope 3 estimation, streamlined reporting,
abbreviated HLEG assessment.

#### `high_emitter`

Heavy industry, energy, and mining organizations. SDA mandatory pathway,
process emissions focus, fossil fuel phase-out assessment (HLEG R10 enhanced),
CCS/CCUS pathway support, MPP sector pathways (steel, cement, chemicals,
aluminium).

#### `service_sector`

Professional, financial, and technology services. Low Scope 1 / high Scope 3
(Categories 1, 6, 7) profile, ACA pathway, RE100 renewable procurement focus,
cloud computing emissions tracking, remote work accounting.

#### `manufacturing_sector`

General manufacturing organizations. Mixed scope profile with energy efficiency
and process optimization focus, SDA or ACA pathway, product carbon footprint
tracking, supplier engagement for Scope 3 Categories 1 and 4.

## Standards Compliance Matrix

### Primary Standards

| Standard | Version | Pack Coverage |
|----------|---------|---------------|
| Race to Zero Campaign | 2020, updated 2022 | Full lifecycle (Pledge through Annual Reporting) |
| Race to Zero Interpretation Guide | June 2022 | All 4 pillars, 20 sub-criteria |
| HLEG "Integrity Matters" | November 2022 | All 10 recommendations, 45+ sub-criteria |

### Secondary Standards

| Standard | Coverage |
|----------|----------|
| Paris Agreement (2015) | 1.5C pathway alignment for targets |
| IPCC AR6 WG3 (2022) | 43% by 2030 benchmark, GWP-100 values |
| SBTi Corporate Net-Zero Standard V1.3 | Near-term/long-term/net-zero target validation |
| CDP Climate Change (2024) | Questionnaire mapping for corporate reporting |
| C40 Deadline 2020 | City-scale pathway benchmarks |
| ICLEI GreenClimateCities | Local government climate action |
| GFANZ Guidance (2022) | Financial institution transition plans, PCAF |
| IEA Net Zero by 2050 (2021/2023) | 25+ sector milestones and benchmarks |
| TPI Global Climate Transition (2024) | Corporate transition pathway assessment |
| ACT Initiative (2023) | Low-carbon transition methodology |
| MPP Hard-to-Abate (2022) | Steel, cement, chemicals, aluminium pathways |
| CRREM (2023) | Real estate carbon risk benchmarks |
| GHG Protocol Corporate Standard | Scope 1/2/3 accounting |
| GHG Protocol Scope 2 Guidance | Location-based and market-based |
| GHG Protocol Scope 3 Standard | All 15 categories |
| ISO 14064-1:2018 | Organization-level GHG quantification |
| PCAF Global Standard (2022) | Financed emissions for financial institutions |
| ESRS E1 Climate Change | EU sustainability reporting |
| TCFD Recommendations | Climate-related financial disclosures |

## API Reference Summary

### Engines

```python
# All engines follow the same pattern
from packs.net_zero.PACK_025_race_to_zero.engines import (
    PledgeCommitmentEngine,       # pledge_engine.assess(...)
    StartingLineEngine,           # starting_line.assess(...)
    InterimTargetEngine,          # target_engine.validate(...)
    ActionPlanEngine,             # action_plan.generate(...)
    ProgressTrackingEngine,       # progress.track(...)
    SectorPathwayEngine,          # sector.assess(...)
    PartnershipScoringEngine,     # partnership.score(...)
    CampaignReportingEngine,      # reporting.generate(...)
    CredibilityAssessmentEngine,  # credibility.assess(...)
    RaceReadinessEngine,          # readiness.calculate(...)
)
```

### Workflows

```python
from packs.net_zero.PACK_025_race_to_zero.workflows import (
    PledgeOnboardingWorkflow,           # 5-phase onboarding
    StartingLineAssessmentWorkflow,     # 4-phase assessment
    ActionPlanningWorkflow,             # 6-phase planning
    AnnualReportingWorkflow,            # 7-phase reporting
    SectorPathwayWorkflow,             # 5-phase sector alignment
    PartnershipEngagementWorkflow,      # 5-phase partnership
    CredibilityReviewWorkflow,          # 4-phase HLEG review
    FullRaceToZeroWorkflow,             # 10-phase end-to-end
)
```

### Integrations

```python
from packs.net_zero.PACK_025_race_to_zero.integrations import (
    RaceToZeroOrchestrator,    # run_pipeline(...)
    MRVBridge,                 # 30 MRV agent routing
    GHGAppBridge,              # GL-GHG-APP inventory
    SBTiAppBridge,             # SBTi target validation
    DecarbBridge,              # DECARB reduction planning
    TaxonomyBridge,            # EU Taxonomy alignment
    DataBridge,                # 20 DATA agent routing
    UNFCCCBridge,              # UNFCCC portal integration
    CDPBridge,                 # CDP questionnaire mapping
    GFANZBridge,               # GFANZ transition plans
    RaceToZeroSetupWizard,     # 8-step configuration
    RaceToZeroHealthCheck,     # 22-category health check
)
```

### Configuration

```python
from packs.net_zero.PACK_025_race_to_zero.config.presets import (
    AVAILABLE_PRESETS,           # Dict[str, str] of preset paths
    ACTOR_TYPE_PRESET_MAP,      # Dict[str, str] actor -> preset
    DEFAULT_PRESET,             # "corporate_commitment"
    get_preset_path,            # get_preset_path("corporate_commitment")
    get_preset_for_actor_type,  # get_preset_for_actor_type("CORPORATE")
)
```

## Testing

### Run All Tests

```bash
# From pack root directory
cd packs/net-zero/PACK-025-race-to-zero

# Run all 797 tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing

# Run specific test suites
python -m pytest tests/test_engines.py -v          # Engine tests
python -m pytest tests/test_workflows.py -v        # Workflow tests
python -m pytest tests/test_templates.py -v        # Template tests
python -m pytest tests/test_integrations.py -v     # Integration tests
python -m pytest tests/test_config.py -v           # Config tests
python -m pytest tests/test_presets.py -v           # Preset tests
python -m pytest tests/test_e2e.py -v              # End-to-end tests
python -m pytest tests/test_init.py -v             # Package init tests
```

### Test Categories

| Category | File | Description |
|----------|------|-------------|
| Engine tests | `test_engines.py` | Unit tests for all 10 calculation engines |
| Workflow tests | `test_workflows.py` | Orchestration, phase sequencing, data flow |
| Template tests | `test_templates.py` | Report generation, formatting, output validation |
| Integration tests | `test_integrations.py` | Bridge connectivity, data routing, error handling |
| Config tests | `test_config.py` | Configuration loading, validation, defaults |
| Preset tests | `test_presets.py` | All 8 preset configurations, actor-type mapping |
| E2E tests | `test_e2e.py` | Full pipeline end-to-end scenarios |
| Init tests | `test_init.py` | Package imports, module resolution |

## Pack Structure

```
PACK-025-race-to-zero/
  __init__.py                          # Pack-level exports (242 lines)
  pack.yaml                           # Pack manifest (1711 lines)
  README.md                           # This file
  ARCHITECTURE.md                     # Technical architecture
  CHANGELOG.md                        # Version history
  VALIDATION_REPORT.md                # Validation results
  CONTRIBUTING.md                     # Development guidelines
  config/
    __init__.py                        # Config module exports
    presets/
      __init__.py                      # Preset loader functions
      corporate_commitment.yaml        # Large Corporate (SBTi/CDP)
      financial_institution.yaml       # Bank/Insurance (GFANZ/PCAF)
      city_municipality.yaml           # City (C40/ICLEI/GPC)
      region_state.yaml                # Region (Under2 Coalition)
      sme_business.yaml               # SME (SME Climate Hub)
      high_emitter.yaml               # Heavy Industry (SDA/MPP)
      service_sector.yaml             # Services (ACA/RE100)
      manufacturing_sector.yaml       # Manufacturing (SDA+ACA)
  engines/
    __init__.py                        # Engine exports
    pledge_commitment_engine.py        # Pledge eligibility & quality
    starting_line_engine.py            # 4P Starting Line assessment
    interim_target_engine.py           # 1.5C pathway target validation
    action_plan_engine.py              # MACC-based action plan
    progress_tracking_engine.py        # Annual trajectory alignment
    sector_pathway_engine.py           # 25+ sector pathway benchmarks
    partnership_scoring_engine.py      # 40+ partner initiative scoring
    campaign_reporting_engine.py       # Multi-partner disclosure
    credibility_assessment_engine.py   # HLEG 10-recommendation scoring
    race_readiness_engine.py           # 8-dimension composite score
  workflows/
    __init__.py                        # Workflow exports
    pledge_onboarding_workflow.py      # 5-phase onboarding
    starting_line_assessment_workflow.py # 4-phase assessment
    action_planning_workflow.py        # 6-phase action planning
    annual_reporting_workflow.py       # 7-phase annual reporting
    sector_pathway_workflow.py         # 5-phase sector alignment
    partnership_engagement_workflow.py # 5-phase partnership
    credibility_review_workflow.py     # 4-phase HLEG review
    full_race_to_zero_workflow.py      # 10-phase end-to-end
  templates/
    __init__.py                        # Template exports
    pledge_commitment_letter.py        # Formal pledge letter
    starting_line_checklist.py         # 20-criterion checklist
    action_plan_document.py            # Quantified action plan
    annual_progress_report.py          # YoY trajectory report
    sector_pathway_roadmap.py          # Sector benchmark roadmap
    partnership_framework.py           # Partner crosswalk
    credibility_assessment_report.py   # HLEG scoring matrix
    campaign_submission_package.py     # Submission-ready package
    disclosure_dashboard.py            # Real-time compliance
    race_to_zero_certificate.py        # Readiness certificate
  integrations/
    __init__.py                        # Integration exports (489 lines)
    pack_orchestrator.py               # 10-phase DAG pipeline
    mrv_bridge.py                      # 30 MRV agents
    ghg_app_bridge.py                  # GL-GHG-APP v1.0
    sbti_app_bridge.py                 # GL-SBTi-APP
    decarb_bridge.py                   # 21 DECARB-X agents
    taxonomy_bridge.py                 # GL-Taxonomy-APP
    data_bridge.py                     # 20 DATA agents
    unfccc_bridge.py                   # UNFCCC R2Z portal
    cdp_bridge.py                      # CDP disclosure platform
    gfanz_bridge.py                    # GFANZ framework
    setup_wizard.py                    # 8-step wizard
    health_check.py                    # 22-category health check
  tests/
    __init__.py
    conftest.py                        # Shared test infrastructure
    test_engines.py                    # Engine unit tests
    test_workflows.py                  # Workflow tests
    test_templates.py                  # Template tests
    test_integrations.py               # Integration tests
    test_config.py                     # Configuration tests
    test_presets.py                    # Preset validation tests
    test_e2e.py                       # End-to-end tests
    test_init.py                       # Package init tests
```

## Security

- **Authentication**: JWT (RS256)
- **Authorization**: RBAC with organization-level and pledge-level access control
- **Encryption at rest**: AES-256-GCM
- **Encryption in transit**: TLS 1.3
- **Audit logging**: All 10 engines produce audit events with SHA-256 provenance
- **PII redaction**: Automatic PII detection and redaction
- **Data classification**: CONFIDENTIAL, RESTRICTED, INTERNAL, PUBLIC

### Required Roles

| Role | Access Level |
|------|-------------|
| `race_to_zero_admin` | Full read/write on all pack resources |
| `sustainability_manager` | Manage pledges, assessments, and reports |
| `climate_analyst` | Read/write on calculations and pathway analysis |
| `pledge_coordinator` | Manage pledge onboarding and Starting Line assessment |
| `partnership_manager` | Manage partner initiative engagement |
| `external_auditor` | Read-only access to credibility and verification packages |
| `viewer` | Read-only access to reports and dashboards |

## Troubleshooting

### Common Issues

**Health check reports missing pathway data**

The sector pathway engine requires benchmark data from IEA, IPCC, TPI, MPP,
ACT, and CRREM. Run the seeding step during installation:

```bash
python -c "
from packs.net_zero.PACK_025_race_to_zero.integrations import RaceToZeroHealthCheck
hc = RaceToZeroHealthCheck()
result = hc.run()
for check in result.checks:
    if check.status != 'PASS':
        print(f'{check.category}: {check.message}')
        for suggestion in check.remediations:
            print(f'  -> {suggestion}')
"
```

**Starting Line assessment shows gaps for all criteria**

Ensure evidence documents are provided for each sub-criterion. The Starting
Line engine requires explicit evidence mapping:

```python
evidence = [
    {"criterion": "SL-P1", "document": "board_resolution.pdf", "date": "2025-01-15"},
    {"criterion": "SL-P2", "document": "interim_target_letter.pdf", "date": "2025-01-20"},
    # ... map all 20 sub-criteria
]
```

**GFANZ bridge returns empty portfolio**

Financial institution preset requires PCAF asset class configuration. Ensure
the `pcaf_methodology` flag is enabled and asset classes are specified.

**Pledge rated as INELIGIBLE despite net-zero commitment**

Check all 8 eligibility criteria. Common missing items:
- Governance endorsement (board-level commitment required)
- Public disclosure commitment (must agree to publish progress)
- Scope coverage (Scope 3 required for corporates if material)

**Sector pathway engine returns "SECTOR_NOT_FOUND"**

Use NACE/NAICS/GICS sector codes for classification. The engine supports
25+ sectors -- verify your sector maps to a supported pathway.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style
guidelines, testing requirements, and pull request process.

## License

Proprietary - GreenLang Platform Team

## Support

- **Documentation**: https://docs.greenlang.io/packs/race-to-zero
- **Changelog**: https://docs.greenlang.io/packs/race-to-zero/changelog
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Validation**: [VALIDATION_REPORT.md](VALIDATION_REPORT.md)
- **Support tier**: Enterprise

## External References

- [Race to Zero Campaign](https://unfccc.int/climate-action/race-to-zero-campaign)
- [Race to Zero Interpretation Guide (June 2022)](https://climatechampions.unfccc.int/wp-content/uploads/2022/06/Race-to-Zero-Interpretation-Guide.pdf)
- [HLEG Integrity Matters (November 2022)](https://www.un.org/sites/un2.un.org/files/high-level_expert_group_n7b.pdf)
- [Paris Agreement (2015)](https://unfccc.int/sites/default/files/english_paris_agreement.pdf)
- [IPCC AR6 WG3 (2022)](https://www.ipcc.ch/report/ar6/wg3/)
- [SBTi Corporate Net-Zero Standard](https://sciencebasedtargets.org/net-zero)
- [IEA Net Zero by 2050 Roadmap](https://www.iea.org/reports/net-zero-by-2050)
- [GFANZ Framework](https://www.gfanzero.com/)
- [CDP Climate Change](https://www.cdp.net/en)
- [C40 Cities](https://www.c40.org/)
- [ICLEI](https://iclei.org/)
- [GHG Protocol](https://ghgprotocol.org/)
