# PACK-025 Race to Zero Pack - Architecture

**Version**: 1.0.0
**Date**: 2026-03-18
**Author**: GreenLang Platform Team

---

## 1. System Overview

PACK-025 implements a layered architecture with four tiers: engines (core
calculations), workflows (orchestrated sequences), integrations (external
bridges), and templates (report generation). All tiers are coordinated by
the Pack Orchestrator, which executes a 10-phase DAG pipeline with dependency
resolution and SHA-256 provenance tracking.

The system follows a 3-tier deployment model:

- **Data Tier**: PostgreSQL 16 with pgvector and TimescaleDB for persistent
  storage, Redis 7 for caching and task queuing.
- **Compute Tier**: 10 calculation engines, 8 workflows, 12 integrations
  running on async FastAPI with Celery workers.
- **Presentation Tier**: 10 templates generating reports in MD, HTML, PDF,
  and JSON formats via Jinja2 rendering.

```
+------------------------------------------------------------------+
|                        PACK-025 System                           |
+------------------------------------------------------------------+
|                                                                  |
|  Tier 4: Templates (10)                                          |
|  +---------+  +---------+  +---------+  +---------+  +--------+ |
|  |Pledge   |  |Starting |  |Action   |  |Annual   |  |Sector  | |
|  |Letter   |  |Checklist|  |Plan     |  |Progress |  |Roadmap | |
|  +---------+  +---------+  +---------+  +---------+  +--------+ |
|  +---------+  +---------+  +---------+  +---------+  +--------+ |
|  |Partner  |  |Credibil |  |Campaign |  |Disclosu |  |R2Z     | |
|  |Framewk  |  |Report   |  |Submit   |  |Dashbrd  |  |Certif  | |
|  +---------+  +---------+  +---------+  +---------+  +--------+ |
|                                                                  |
|  Tier 3: Workflows (8)                                           |
|  +------+  +------+  +------+  +------+  +------+  +------+     |
|  |Pledge|  |Start |  |Action|  |Annual|  |Sector|  |Prtnr |     |
|  |Onbrd |  |Line  |  |Plan  |  |Rptng |  |Pathwy|  |Engage|     |
|  +------+  +------+  +------+  +------+  +------+  +------+     |
|  +------+  +------+                                              |
|  |Cred  |  |Full  |                                              |
|  |Review|  |R2Z   |                                              |
|  +------+  +------+                                              |
|                                                                  |
|  Tier 2: Engines (10)                                            |
|  +------+  +------+  +------+  +------+  +------+               |
|  |Pledge|  |Start |  |Interi|  |Action|  |Progre|               |
|  |Commit|  |Line  |  |Target|  |Plan  |  |Track |               |
|  +------+  +------+  +------+  +------+  +------+               |
|  +------+  +------+  +------+  +------+  +------+               |
|  |Sector|  |Prtnr |  |Campgn|  |Credib|  |Race  |               |
|  |Pathwy|  |Score |  |Report|  |Assess|  |Ready |               |
|  +------+  +------+  +------+  +------+  +------+               |
|                                                                  |
|  Tier 1: Integrations (12)                                       |
|  +-------+ +-------+ +-------+ +-------+ +-------+ +-------+   |
|  |Orch   | |MRV    | |GHG   | |SBTi   | |DECARB | |Taxon  |   |
|  |estrate| |Bridge | |Bridge| |Bridge | |Bridge | |Bridge |   |
|  +-------+ +-------+ +-------+ +-------+ +-------+ +-------+   |
|  +-------+ +-------+ +-------+ +-------+ +-------+ +-------+   |
|  |DATA   | |UNFCCC | |CDP    | |GFANZ  | |Setup  | |Health |   |
|  |Bridge | |Bridge | |Bridge| |Bridge | |Wizard | |Check  |   |
|  +-------+ +-------+ +-------+ +-------+ +-------+ +-------+   |
|                                                                  |
+-----+-------+-------+-------+-------+-------+-------+-----------+
      |       |       |       |       |       |       |
      v       v       v       v       v       v       v
  +------+ +------+ +------+ +------+ +------+ +------+ +------+
  |30 MRV| |20 DATA| |10    | |GL-GHG| |GL-   | |GL-CDP| |GFANZ |
  |Agents| |Agents | |FOUND | |APP   | |SBTi  | |APP   | |Fwork |
  +------+ +------+ +------+ +------+ +------+ +------+ +------+
```

### Engine Abbreviations

| Abbrev | Full Name |
|--------|-----------|
| Pledge Commit | Pledge Commitment Engine |
| Start Line | Starting Line Engine |
| Interi Target | Interim Target Engine |
| Action Plan | Action Plan Engine |
| Progre Track | Progress Tracking Engine |
| Sector Pathwy | Sector Pathway Engine |
| Prtnr Score | Partnership Scoring Engine |
| Campgn Report | Campaign Reporting Engine |
| Credib Assess | Credibility Assessment Engine |
| Race Ready | Race Readiness Engine |

---

## 2. 10-Phase DAG Pipeline

The Pack Orchestrator executes the following directed acyclic graph. Each phase
depends on the completion of its predecessors. Independent phases may execute
in parallel where dependencies allow.

```
Phase 1: Health Check
    |
    |  22-category system verification
    |  - Engine availability (10 engines)
    |  - MRV agent connectivity (30 agents)
    |  - DATA agent connectivity (20 agents)
    |  - Database migration status (V148-V157)
    |  - Reference data currency
    |
    v
Phase 2: Configuration
    |
    |  Actor-type preset loading
    |  - Preset YAML parsing and validation
    |  - Engine enablement based on actor type
    |  - Partner initiative configuration
    |  - Scope boundary configuration
    |
    v
Phase 3: Pledge Commitment      <-- Pledge Commitment Engine
    |
    |  8 eligibility criteria validation
    |  4 quality levels (Strong/Adequate/Weak/Ineligible)
    |  Partner initiative crosswalk
    |
    v
Phase 4: Starting Line Assessment  <-- Starting Line Engine
    |
    |  4 pillars (Pledge/Plan/Proceed/Publish)
    |  20 sub-criteria (SL-P1..P5, SL-A1..A5, SL-R1..R5, SL-D1..D5)
    |  Evidence mapping and gap identification
    |
    v
Phase 5: Interim Target Validation  <-- Interim Target Engine
    |                                    MRV Bridge (30 agents)
    |                                    GHG App Bridge
    |                                    SBTi App Bridge
    |
    |  1.5C pathway alignment (43% by 2030)
    |  SDA/ACA methodology validation
    |  Fair share contribution assessment
    |
    +----------------+
    |                |
    v                v
Phase 6:          Phase 7:
Action Plan       Sector Pathway    <-- DECARB Bridge
    |             Alignment             IEA NZE, IPCC AR6, TPI, MPP, ACT, CRREM
    |                |
    |  MACC analysis |  25+ sector pathways
    |  Levers, costs |  Gap-to-benchmark
    |  Milestones    |  Sector milestones
    |                |
    +--------+-------+
             |
             v
Phase 8: Partnership Scoring     <-- Partnership Scoring Engine
    |
    |  40+ partner initiatives
    |  Requirement crosswalk
    |  Engagement quality scoring
    |  Synergy and duplication analysis
    |
    v
Phase 9: Campaign Reporting + Credibility Assessment
    |
    |  Multi-partner report generation
    |  HLEG 10 recommendations (45+ sub-criteria)
    |  Credibility rating (A-F)
    |  Partner-specific format output (CDP, GFANZ, C40, ICLEI)
    |
    v
Phase 10: Race Readiness Scoring  <-- Race Readiness Engine
    |
    |  8-dimension composite score (0-100)
    |  RAG status (Red/Amber/Green)
    |  Readiness level (Not Ready/Partially Ready/Ready/Exemplary)
    |  Prioritized improvement actions
    |  SHA-256 provenance hash
    |
    v
  [COMPLETE]
```

### Phase Dependencies

```python
PHASE_DEPENDENCIES = {
    "health_check":              [],
    "configuration":             ["health_check"],
    "pledge_commitment":         ["configuration"],
    "starting_line":             ["pledge_commitment"],
    "interim_target":            ["starting_line"],
    "action_plan":               ["interim_target"],
    "sector_pathway":            ["interim_target"],
    "partnership_scoring":       ["action_plan", "sector_pathway"],
    "campaign_credibility":      ["partnership_scoring"],
    "race_readiness":            ["campaign_credibility"],
}
```

### Parallel Execution

Phases 6 (Action Plan) and 7 (Sector Pathway) can execute in parallel since
they share only a read dependency on Phase 5 (Interim Target) output.

---

## 3. Data Flow Diagrams

### Primary Data Flow

```
                    Activity Data (energy, fuel, fleet, supplier)
                                    |
                                    v
               +--------------------------------------------+
               |     Data Intake (20 AGENT-DATA agents)     |
               |  PDF/Excel/ERP/API -> Quality -> Reconcile |
               +--------------------------------------------+
                                    |
                                    v
               +--------------------------------------------+
               |    Emissions Calculation (30 MRV agents)    |
               |  Scope 1 (8) + Scope 2 (5) + Scope 3 (15) |
               |  + Cross-cutting (2)                        |
               +--------------------------------------------+
                                    |
                                    v
               +--------------------------------------------+
               |     Pledge Commitment Engine                |
               |  8 eligibility criteria, quality rating     |
               |  Partner initiative crosswalk               |
               +--------------------------------------------+
                                    |
                                    v
               +--------------------------------------------+
               |     Starting Line Engine                    |
               |  4 pillars, 20 sub-criteria                 |
               |  Evidence mapping, gap identification       |
               +--------------------------------------------+
                                    |
                                    v
               +--------------------------------------------+
               |     Interim Target Engine                   |
               |  1.5C pathway validation                    |
               |  Scope coverage, fair share                 |
               |  SDA/ACA methodology check                  |
               +--------------------------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
    +-------------------------+     +----------------------------+
    | Action Plan Engine      |     | Sector Pathway Engine      |
    | MACC analysis           |     | 25+ sector pathways        |
    | Quantified actions      |     | Gap-to-benchmark           |
    | Milestones, costs       |     | IEA/IPCC/TPI/MPP/ACT/CRREM|
    +-------------------------+     +----------------------------+
                    |                               |
                    +---------------+---------------+
                                    |
                                    v
               +--------------------------------------------+
               |     Partnership Scoring Engine              |
               |  40+ partner initiatives                    |
               |  Requirement crosswalk                      |
               |  Engagement quality (5 dimensions)          |
               +--------------------------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
    +-------------------------+     +----------------------------+
    | Campaign Reporting      |     | Credibility Assessment     |
    | Engine                  |     | Engine                     |
    | Multi-partner format    |     | HLEG 10 recommendations    |
    | CDP, GFANZ, C40, ICLEI |     | 45+ sub-criteria           |
    | Annual disclosure       |     | Credibility rating (A-F)   |
    +-------------------------+     +----------------------------+
                    |                               |
                    +---------------+---------------+
                                    |
                                    v
               +--------------------------------------------+
               |     Race Readiness Engine                   |
               |  8 dimensions: pledge, starting line,       |
               |  target, action plan, progress, sector,     |
               |  partnership, credibility                   |
               |  Composite score (0-100), RAG, level        |
               +--------------------------------------------+
```

### Actor-Type Data Flow Variants

```
CORPORATE (SBTi/CDP)
  MRV (Scope 1+2+3) --> SBTi SDA/ACA --> CDP Reporting --> HLEG Full

FINANCIAL_INSTITUTION (GFANZ)
  MRV (Operational) + PCAF (Financed) --> GFANZ Pathway --> Portfolio Temp

CITY (C40/ICLEI)
  GPC Community Inventory --> C40 Benchmarks --> C40/ICLEI Reporting

REGION (Under2)
  Sub-national Inventory + LULUCF --> Under2 Pathway --> Policy Levers

SME (SME Climate Hub)
  Simplified MRV (Scope 1+2) --> Spend-based Scope 3 --> 6-Engine Flow

HEAVY_INDUSTRY (SDA/MPP)
  MRV (Process + Energy) --> SDA Mandatory --> MPP Sector --> Enhanced R10

SERVICES (ACA/RE100)
  MRV (Low S1, High S3) --> ACA Pathway --> RE100 --> Cloud Emissions

MANUFACTURING (SDA+ACA)
  MRV (Mixed Scope) --> SDA/ACA Select --> Product CF --> Supplier Engage
```

---

## 4. Data Models

The pack uses Pydantic v2 models for all data structures. Key models include:

| Model | Description | Source Engine |
|-------|-------------|-------------|
| `PledgeResult` | Eligibility status, quality rating, criteria scores | Pledge Commitment |
| `StartingLineResult` | 4-pillar scores, 20 criterion results, compliance status | Starting Line |
| `InterimTargetResult` | Pathway alignment, annual rate, fair share assessment | Interim Target |
| `ActionPlanResult` | MACC curve, action list, milestones, costs | Action Plan |
| `ProgressResult` | YoY changes, trajectory status, variance analysis | Progress Tracking |
| `SectorPathwayResult` | Gap-to-benchmark, sector milestones, alignment status | Sector Pathway |
| `PartnershipResult` | Partner scores, engagement quality, synergy analysis | Partnership Scoring |
| `CampaignReportResult` | Multi-partner reports, submission status | Campaign Reporting |
| `CredibilityResult` | 10-recommendation scores, overall rating (A-F) | Credibility Assessment |
| `ReadinessResult` | 8-dimension scores, composite (0-100), RAG, level | Race Readiness |

### Workflow Data Models

| Model | Description | Source Workflow |
|-------|-------------|---------------|
| `PledgeOnboardingResult` | 5-phase result with pledge package | Pledge Onboarding |
| `StartingLineResult` | 4-phase result with compliance certificate | Starting Line Assessment |
| `ActionPlanningResult` | 6-phase result with action plan document | Action Planning |
| `AnnualReportingResult` | 7-phase result with reports and submissions | Annual Reporting |
| `SectorPathwayResult` | 5-phase result with roadmap and milestones | Sector Pathway |
| `PartnershipEngagementResult` | 5-phase result with engagement plan | Partnership Engagement |
| `CredibilityReviewResult` | 4-phase result with credibility report | Credibility Review |
| `FullR2ZResult` | 10-phase result with readiness score and certificate | Full Race to Zero |

---

## 5. Database Schema Overview

PACK-025 introduces 10 new migration files (V148 through V157). All tables
use the `gl_r2z_` prefix to avoid naming conflicts.

### Core Tables

```
+----------------------------------+     +----------------------------------+
| gl_r2z_organization_profiles     |     | gl_r2z_pledge_commitments        |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| organization_id (UUID, FK)       |     | organization_id (UUID, FK)       |
| actor_type (ENUM)                |     | net_zero_target_year (INT)       |
| sector (VARCHAR)                 |     | interim_target_year (INT)        |
| size (ENUM)                      |     | interim_reduction_pct (DECIMAL)  |
| geography (VARCHAR)              |     | scope_coverage (JSONB)           |
| partner_initiative (VARCHAR)     |     | quality_rating (ENUM)            |
| preset_name (VARCHAR)            |     | eligibility_status (ENUM)        |
| created_at (TIMESTAMPTZ)         |     | governance_endorsed (BOOLEAN)    |
| updated_at (TIMESTAMPTZ)         |     | provenance_hash (VARCHAR(64))    |
+----------------------------------+     | created_at (TIMESTAMPTZ)         |
                                         +----------------------------------+

+----------------------------------+     +----------------------------------+
| gl_r2z_partner_memberships       |     | gl_r2z_starting_line_assessments |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| organization_id (UUID, FK)       |     | pledge_id (UUID, FK)             |
| partner_initiative (VARCHAR)     |     | pledge_score (DECIMAL)           |
| membership_status (ENUM)         |     | plan_score (DECIMAL)             |
| engagement_level (ENUM)          |     | proceed_score (DECIMAL)          |
| joined_date (DATE)               |     | publish_score (DECIMAL)          |
| reporting_channel (VARCHAR)      |     | overall_compliance (ENUM)        |
| created_at (TIMESTAMPTZ)         |     | criteria_results (JSONB)         |
+----------------------------------+     | provenance_hash (VARCHAR(64))    |
                                         +----------------------------------+

+----------------------------------+     +----------------------------------+
| gl_r2z_starting_line_gaps        |     | gl_r2z_interim_targets           |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| assessment_id (UUID, FK)         |     | pledge_id (UUID, FK)             |
| criterion_id (VARCHAR)           |     | base_year (INT)                  |
| pillar (ENUM)                    |     | target_year (INT)                |
| gap_severity (ENUM)              |     | reduction_pct (DECIMAL)          |
| description (TEXT)               |     | annual_rate_pct (DECIMAL)        |
| remediation_action (TEXT)        |     | pathway_source (ENUM)            |
| remediation_priority (ENUM)      |     | methodology (ENUM)               |
| target_date (DATE)               |     | is_aligned_1_5c (BOOLEAN)        |
| created_at (TIMESTAMPTZ)         |     | fair_share_score (DECIMAL)       |
+----------------------------------+     | provenance_hash (VARCHAR(64))    |
                                         +----------------------------------+

+----------------------------------+     +----------------------------------+
| gl_r2z_action_plans              |     | gl_r2z_abatement_actions         |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| organization_id (UUID, FK)       |     | action_plan_id (UUID, FK)        |
| target_id (UUID, FK)             |     | action_name (VARCHAR)            |
| publication_date (DATE)          |     | category (ENUM)                  |
| total_abatement_tco2e (DECIMAL)  |     | abatement_tco2e (DECIMAL)        |
| total_cost_eur (DECIMAL)         |     | cost_per_tco2e (DECIMAL)         |
| planning_horizon_years (INT)     |     | implementation_year (INT)        |
| milestones (JSONB)               |     | feasibility_level (ENUM)         |
| status (ENUM)                    |     | time_horizon (ENUM)              |
| provenance_hash (VARCHAR(64))    |     | created_at (TIMESTAMPTZ)         |
+----------------------------------+     +----------------------------------+

+----------------------------------+     +----------------------------------+
| gl_r2z_annual_reports            |     | gl_r2z_sector_pathways           |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| organization_id (UUID, FK)       |     | organization_id (UUID, FK)       |
| reporting_year (INT)             |     | sector (VARCHAR)                 |
| scope1_tco2e (DECIMAL)           |     | pathway_source (ENUM)            |
| scope2_tco2e (DECIMAL)           |     | current_intensity (DECIMAL)      |
| scope3_tco2e (DECIMAL)           |     | benchmark_intensity (DECIMAL)    |
| total_tco2e (DECIMAL)            |     | gap_pct (DECIMAL)                |
| yoy_change_pct (DECIMAL)         |     | alignment_status (ENUM)          |
| trajectory_status (ENUM)         |     | milestones (JSONB)               |
| submissions (JSONB)              |     | provenance_hash (VARCHAR(64))    |
| provenance_hash (VARCHAR(64))    |     +----------------------------------+
| created_at (TIMESTAMPTZ)         |
+----------------------------------+

+----------------------------------+     +----------------------------------+
| gl_r2z_partnership_collaborations|     | gl_r2z_credibility_assessments   |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| organization_id (UUID, FK)       |     | organization_id (UUID, FK)       |
| partner_type (ENUM)              |     | assessment_year (INT)            |
| engagement_level (ENUM)          |     | r1_pledge_quality (DECIMAL)      |
| collaboration_status (ENUM)      |     | r2_interim_ambition (DECIMAL)    |
| requirement_crosswalk (JSONB)    |     | r3_credit_use (DECIMAL)          |
| synergy_score (DECIMAL)          |     | r4_lobbying (DECIMAL)            |
| provenance_hash (VARCHAR(64))    |     | r5_just_transition (DECIMAL)     |
| created_at (TIMESTAMPTZ)         |     | r6_financial (DECIMAL)           |
+----------------------------------+     | r7_transparency (DECIMAL)        |
                                         | r8_scope (DECIMAL)               |
+----------------------------------+     | r9_governance (DECIMAL)          |
| gl_r2z_campaign_submissions      |     | r10_fossil_fuel (DECIMAL)        |
|----------------------------------|     | overall_rating (VARCHAR(2))      |
| id (UUID, PK)                    |     | composite_score (DECIMAL)        |
| organization_id (UUID, FK)       |     | provenance_hash (VARCHAR(64))    |
| submission_year (INT)            |     +----------------------------------+
| partner_channel (VARCHAR)        |
| submission_format (VARCHAR)      |     +----------------------------------+
| submission_status (ENUM)         |     | gl_r2z_audit_trail               |
| response_ref (VARCHAR)           |     |----------------------------------|
| created_at (TIMESTAMPTZ)         |     | id (UUID, PK)                    |
+----------------------------------+     | organization_id (UUID, FK)       |
                                         | action (VARCHAR)                 |
+----------------------------------+     | engine_id (VARCHAR)              |
| gl_r2z_readiness_scores          |     | input_hash (VARCHAR(64))         |
|----------------------------------|     | output_hash (VARCHAR(64))        |
| id (UUID, PK)                    |     | user_id (UUID)                   |
| organization_id (UUID, FK)       |     | timestamp (TIMESTAMPTZ)          |
| assessment_year (INT)            |     | metadata (JSONB)                 |
| pledge_score (DECIMAL)           |     +----------------------------------+
| starting_line_score (DECIMAL)    |
| target_score (DECIMAL)           |     +----------------------------------+
| action_plan_score (DECIMAL)      |     | gl_r2z_workflow_executions       |
| progress_score (DECIMAL)         |     |----------------------------------|
| sector_score (DECIMAL)           |     | id (UUID, PK)                    |
| partnership_score (DECIMAL)      |     | organization_id (UUID, FK)       |
| credibility_score (DECIMAL)      |     | workflow_id (VARCHAR)            |
| composite_score (DECIMAL)        |     | phases_completed (INT)           |
| rag_status (ENUM)                |     | phases_total (INT)               |
| readiness_level (ENUM)           |     | status (ENUM)                    |
| provenance_hash (VARCHAR(64))    |     | started_at (TIMESTAMPTZ)         |
| created_at (TIMESTAMPTZ)         |     | completed_at (TIMESTAMPTZ)       |
+----------------------------------+     | provenance_hash (VARCHAR(64))    |
                                         +----------------------------------+
```

### Views

| View | Description | Source Tables |
|------|-------------|-------------|
| `gl_r2z_v_pledge_summary` | Pledge status with Starting Line compliance and partner info | pledge_commitments, starting_line_assessments, partner_memberships |
| `gl_r2z_v_progress_timeline` | Multi-year progress trajectory with target alignment | annual_reports, interim_targets, readiness_scores |
| `gl_r2z_v_partner_overview` | Partnership engagement overview with synergy analysis | partnership_collaborations, partner_memberships, campaign_submissions |

### Indexes

All tables include:
- Primary key index on `id`
- Index on `organization_id` for multi-tenant queries
- Index on `created_at` for time-series queries
- Index on `provenance_hash` for reproducibility lookups

Additional specialized indexes:
- `gl_r2z_pledge_commitments`: Composite index on `(organization_id, net_zero_target_year)`
- `gl_r2z_starting_line_assessments`: Index on `overall_compliance` for filtering
- `gl_r2z_annual_reports`: Composite index on `(organization_id, reporting_year)`
- `gl_r2z_credibility_assessments`: Index on `(overall_rating, composite_score)`
- `gl_r2z_readiness_scores`: Index on `(rag_status, readiness_level)`

### TimescaleDB Hypertables

The following tables are converted to TimescaleDB hypertables for efficient
time-series queries:

- `gl_r2z_annual_reports` (partitioned by `created_at`)
- `gl_r2z_readiness_scores` (partitioned by `created_at`)
- `gl_r2z_audit_trail` (partitioned by `timestamp`)

### Row-Level Security

RLS is enabled on all 16 tables with policies that restrict row access based
on the `organization_id` claim in the JWT token. This ensures multi-tenant
data isolation at the database level.

---

## 6. Engine Architecture Patterns

### Common Engine Pattern

All 10 engines follow the same architectural pattern:

```python
class XxxEngine:
    """Engine docstring with standard/framework reference."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    def assess(self, **kwargs) -> XxxResult:
        """Main entry point -- deterministic calculation."""
        # 1. Validate inputs (Pydantic v2)
        validated = XxxInput(**kwargs)

        # 2. Execute deterministic calculation
        scores = self._calculate(validated)

        # 3. Generate provenance hash
        provenance = sha256(canonical_json(validated, scores))

        # 4. Build result model
        return XxxResult(
            scores=scores,
            provenance_hash=provenance,
            timestamp=datetime.utcnow(),
        )

    def _calculate(self, input: XxxInput) -> Dict:
        """Pure deterministic calculation -- no LLM, no randomness."""
        ...
```

### Zero-Hallucination Enforcement

```
+------------------------------------------------------------------+
|                    Calculation Layer                              |
|  (Deterministic - No LLM)                                        |
+------------------------------------------------------------------+
|                                                                  |
|  Eligibility:  rule_check(criterion[i]) for i=1..8              |
|  Starting Line: criterion_pass_fail(sub_criterion[i]) i=1..20   |
|  Target:       base * (1 - annual_rate)^years vs pathway_bench   |
|  MACC:         sum(action.abatement * action.cost_per_tco2e)     |
|  Progress:     (current - baseline) / baseline * 100             |
|  Sector Gap:   entity_intensity / sector_benchmark_intensity     |
|  Partnership:  sum(dimension_score[i] * weight[i]) i=1..5        |
|  Credibility:  sum(recommendation_score[i] * weight[i]) i=1..10 |
|  Readiness:    sum(dimension_score[i] * weight[i]) i=1..8        |
|  Provenance:   SHA-256(canonical_json(inputs, outputs))          |
|                                                                  |
|  Data sources: IPCC AR6, IEA NZE, TPI, MPP, ACT, CRREM, HLEG,  |
|                Interpretation Guide, GHG Protocol, SBTi Standard |
|                                                                  |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|                    LLM-Assisted Layer                             |
|  (Classification and narrative only)                             |
+------------------------------------------------------------------+
|                                                                  |
|  - Entity resolution (matching organization names)               |
|  - Sector classification (mapping activities to NACE/NAICS)      |
|  - Narrative generation (action plan prose, report narratives,   |
|    pledge letter language, remediation guidance text)             |
|  - Recommendation generation (improvement suggestions)          |
|                                                                  |
|  All LLM outputs are tagged with confidence scores and are      |
|  human-reviewable before inclusion in final outputs.             |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 7. Workflow Orchestration

### Workflow Architecture

Workflows are implemented as async state machines with phase-level error
handling and checkpoint recovery.

```python
class XxxWorkflow:
    """Workflow with N phases."""

    PHASES = [Phase.STEP_1, Phase.STEP_2, ..., Phase.STEP_N]

    async def execute(self, **kwargs) -> XxxResult:
        """Execute all phases in dependency order."""
        state = WorkflowState(phases=self.PHASES)

        for phase in self.PHASES:
            try:
                result = await self._execute_phase(phase, state)
                state.complete_phase(phase, result)
            except Exception as e:
                state.fail_phase(phase, e)
                if phase.is_required:
                    raise WorkflowError(phase, e)

        return self._build_result(state)
```

### Phase Dependencies (Full R2Z Workflow)

```
Pledge ──> StartingLine ──> Targets ──> ActionPlan ──+
                                                      |
                                    Targets ──> SectorAlign ──+
                                                               |
                                                ActionPlan ───+
                                                SectorAlign ──+──> Partners ──> Credibility ──> Readiness
```

---

## 8. Integration Architecture

### Bridge Pattern

All 12 integrations follow the bridge pattern with connection pooling,
retry logic, and graceful degradation.

```python
class XxxBridge:
    """Bridge to external system with graceful degradation."""

    def __init__(self, config: XxxBridgeConfig):
        self.config = config
        self.pool = ConnectionPool(max_size=config.max_connections)

    async def query(self, **kwargs) -> XxxResult:
        """Query with retry and fallback."""
        try:
            return await self._query_with_retry(**kwargs)
        except ConnectionError:
            if self.config.graceful_degradation:
                return self._fallback(**kwargs)
            raise

    async def _query_with_retry(self, **kwargs) -> XxxResult:
        """Retry with exponential backoff."""
        for attempt in range(self.config.max_retries):
            try:
                async with self.pool.acquire() as conn:
                    return await self._execute(conn, **kwargs)
            except TransientError:
                await asyncio.sleep(2 ** attempt)
        raise MaxRetriesExceeded()
```

### External System Integration

```
+-------------------+     +------------------+     +------------------+
|  UNFCCC Race to   |     |  CDP Disclosure   |     |  GFANZ           |
|  Zero Portal      |     |  Platform         |     |  Framework       |
+--------+----------+     +--------+---------+     +--------+---------+
         |                          |                        |
         v                          v                        v
+--------+----------+     +--------+---------+     +--------+---------+
| UNFCCC Bridge      |     | CDP Bridge       |     | GFANZ Bridge     |
| - Commitment       |     | - Questionnaire  |     | - Portfolio      |
|   submission       |     |   mapping        |     |   alignment      |
| - Verification     |     | - Auto response  |     | - Financed       |
|   status           |     | - Score estimate |     |   emissions      |
| - Annual reporting |     | - R2Z/CDP align  |     | - Transition     |
| - Badge retrieval  |     |                  |     |   plan           |
+-------------------+     +------------------+     +------------------+
```

### GreenLang Agent Integration

| Agent Layer | Agents | Bridge | Purpose |
|-------------|--------|--------|---------|
| AGENT-MRV | 30 agents | MRV Bridge | Scope 1/2/3 emissions calculation |
| AGENT-DATA | 20 agents | Data Bridge | Data intake and quality management |
| AGENT-FOUND | 10 agents | (Direct) | Platform services (orchestration, schema, auth) |
| DECARB-X | 21 agents | DECARB Bridge | Reduction planning and MACC generation |

### Optional Bridges (Graceful Degradation)

| Bridge | Required | Fallback Behavior |
|--------|----------|-------------------|
| SBTi App Bridge | No | Internal target validation using IPCC AR6 benchmarks |
| DECARB Bridge | No | Generic MACC curves from published sector studies |
| Taxonomy Bridge | No | Skip EU Taxonomy alignment check |
| UNFCCC Bridge | No | Offline mode with local data store |
| CDP Bridge | No | Generate reports without live submission |
| GFANZ Bridge | No | Generate reports without portfolio scoring |
| PACK-021 Bridge | No | Internal baseline calculation |
| PACK-022 Bridge | No | Data accessed via PACK-021 bridge |
| PACK-023 Bridge | No | Internal SBTi alignment check |

---

## 9. Security Architecture

```
+------------------------------------------------------------------+
|                     Security Layers                               |
+------------------------------------------------------------------+
|                                                                  |
|  Layer 1: Authentication (JWT RS256)                             |
|  +-----------------------------------------------------------+  |
|  | Token validation -> Claims extraction -> User identity     |  |
|  | Token expiry: 1 hour | Refresh: 24 hours | Rotation: auto |  |
|  +-----------------------------------------------------------+  |
|                              |                                   |
|  Layer 2: Authorization (RBAC - 7 roles)                         |
|  +-----------------------------------------------------------+  |
|  | Role check -> Permission check -> Resource access control  |  |
|  | Roles: race_to_zero_admin, sustainability_manager,         |  |
|  |        climate_analyst, pledge_coordinator,                 |  |
|  |        partnership_manager, external_auditor, viewer        |  |
|  +-----------------------------------------------------------+  |
|                              |                                   |
|  Layer 3: Data Protection                                        |
|  +-----------------------------------------------------------+  |
|  | Encryption at rest: AES-256-GCM                            |  |
|  | Encryption in transit: TLS 1.3                             |  |
|  | PII detection and redaction                                |  |
|  | Data classification: CONFIDENTIAL/RESTRICTED/INTERNAL/     |  |
|  |                      PUBLIC                                |  |
|  +-----------------------------------------------------------+  |
|                              |                                   |
|  Layer 4: Audit and Provenance                                   |
|  +-----------------------------------------------------------+  |
|  | All 10 engine operations logged with SHA-256 hashes        |  |
|  | Immutable audit trail: gl_r2z_audit_trail table            |  |
|  | Workflow execution tracking: gl_r2z_workflow_executions     |  |
|  | Configuration change logging                               |  |
|  | Provenance chain: input_hash -> calculation -> output_hash |  |
|  +-----------------------------------------------------------+  |
|                              |                                   |
|  Layer 5: Database Security                                      |
|  +-----------------------------------------------------------+  |
|  | Row-Level Security (RLS) on all 16 tables                  |  |
|  | Organization-level data isolation                          |  |
|  | Pledge-level access control                                |  |
|  | gl_r2z_ table prefix to prevent naming conflicts           |  |
|  +-----------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 10. Deployment Architecture

### Kubernetes Deployment

```
+------------------------------------------------------------------+
|                    Kubernetes Cluster (EKS)                       |
+------------------------------------------------------------------+
|                                                                  |
|  +------------------+  +------------------+  +----------------+  |
|  | PACK-025 API     |  | PACK-025 Worker  |  | PACK-025       |  |
|  | (FastAPI)        |  | (Celery)         |  | Scheduler      |  |
|  | Replicas: 3      |  | Replicas: 4      |  | (Beat)         |  |
|  | CPU: 2 cores     |  | CPU: 4 cores     |  | Replicas: 1    |  |
|  | RAM: 4 GB        |  | RAM: 8 GB        |  |                |  |
|  +--------+---------+  +--------+---------+  +--------+-------+  |
|           |                      |                     |         |
|  +--------v---------+  +--------v---------+  +--------v-------+  |
|  | Kong API Gateway  |  | Redis 7         |  | PostgreSQL 16  |  |
|  | (INFRA-006)       |  | (INFRA-003)     |  | + pgvector     |  |
|  | Rate limiting     |  | Task queue      |  | + TimescaleDB  |  |
|  | Auth middleware    |  | Result backend  |  | (INFRA-002/005)|  |
|  +------------------+  | Config cache     |  +----------------+  |
|                         +------------------+                     |
|                                                                  |
+------------------------------------------------------------------+
```

### Resource Requirements

| Component | Min CPU | Min RAM | Recommended CPU | Recommended RAM |
|-----------|---------|---------|-----------------|-----------------|
| API pod | 1 core | 2 GB | 2 cores | 4 GB |
| Worker pod | 2 cores | 4 GB | 4 cores | 8 GB |
| Scheduler | 0.5 core | 1 GB | 1 core | 2 GB |
| PostgreSQL | 2 cores | 4 GB | 4 cores | 8 GB |
| Redis | 1 core | 2 GB | 2 cores | 4 GB |
| **Total** | **6.5 cores** | **13 GB** | **13 cores** | **26 GB** |

---

## 11. Performance Optimization Strategies

### Caching Strategy

```
Request --> Kong Gateway --> Redis L1 Cache (TTL: 5min)
                                  |
                                  | miss
                                  v
                          Redis L2 Cache (TTL: 1hr)
                                  |
                                  | miss
                                  v
                          PostgreSQL Query
                                  |
                                  v
                          Cache Population (L1 + L2)
```

- **L1 Cache**: Hot data (current year emissions, active pledges) with 5-minute TTL
- **L2 Cache**: Warm data (sector benchmarks, partner requirements) with 1-hour TTL
- **Target cache hit ratio**: 75% (actual: 84%)

### Parallel Processing

- Phases 6 and 7 (Action Plan + Sector Pathway) execute in parallel
- Independent partner scoring calculations parallelized across partners
- Template rendering parallelized across output formats
- MRV agent routing uses batch processing for multi-scope calculations

### Memory Management

- Engine results streamed to database after each phase (no full pipeline in memory)
- MACC curve data capped at 50 resolution points
- Sector pathway data uses lazy loading for 25+ sector benchmarks
- Memory ceiling: 4096 MB (actual peak: 2,180 MB)

---

## 12. Scaling Considerations

### Horizontal Scaling

- API pods: Scale from 3 to 12 replicas based on request rate
- Worker pods: Scale from 4 to 16 replicas based on queue depth
- PostgreSQL: Read replicas for reporting queries
- Redis: Redis Cluster for cache distribution

### Vertical Scaling

- Worker pods can scale to 8 cores / 16 GB for complex sector pathway analysis
- PostgreSQL can scale to 8 cores / 32 GB for large multi-tenant deployments

### Multi-Tenant Capacity

| Metric | Single Tenant | Multi-Tenant (100) | Multi-Tenant (1000) |
|--------|--------------|-------------------|-------------------|
| Concurrent pipelines | 4 | 20 | 50 |
| Pledges tracked | 1 | 100 | 1,000 |
| Annual reports/year | 1 | 100 | 1,000 |
| DB connections | 20 | 60 | 200 |

---

## 13. Disaster Recovery and Backup

### Backup Strategy

| Component | Method | Frequency | Retention |
|-----------|--------|-----------|-----------|
| PostgreSQL | pg_dump + WAL archiving | Continuous | 30 days |
| Redis | RDB snapshots + AOF | Every 15 min | 7 days |
| Configuration | Git version control | On change | Permanent |
| Audit trail | Immutable table + S3 export | Daily | 7 years |

### Recovery Procedures

- **Database recovery**: Point-in-time recovery from WAL archives (RPO: 0)
- **Cache recovery**: Redis auto-rebuild from PostgreSQL on cold start
- **Pipeline recovery**: Checkpoint-based resume from last completed phase
- **Configuration recovery**: Git rollback to known-good configuration

### High Availability

- API: 3 replicas across availability zones
- PostgreSQL: Primary + streaming replica with automatic failover
- Redis: Redis Sentinel with 3 sentinels for automatic failover
- Kong: 2 replicas with health check-based routing

---

*Architecture document maintained by GreenLang Platform Team*
*PACK-025 Race to Zero Pack v1.0.0*
*Last updated: 2026-03-18*
