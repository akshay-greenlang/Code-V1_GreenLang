# PACK-024 Carbon Neutral Pack - Architecture

**Version**: 1.0.0
**Date**: 2026-03-18
**Author**: GreenLang Platform Team

---

## 1. System Overview

PACK-024 implements a layered architecture with four tiers: engines (core
calculations), workflows (orchestrated sequences), integrations (external
bridges), and templates (report generation). All tiers are coordinated by
the Pack Orchestrator, which executes a 10-phase DAG pipeline with dependency
resolution and SHA-256 provenance tracking.

```
+------------------------------------------------------------------+
|                        PACK-024 System                           |
+------------------------------------------------------------------+
|                                                                  |
|  Tier 4: Templates (10)                                          |
|  +---------+  +---------+  +---------+  +---------+  +--------+ |
|  |Footprint|  |CMP      |  |Credit   |  |Retire   |  |Neutral | |
|  |Report   |  |Report   |  |Quality  |  |Cert     |  |Stmt    | |
|  +---------+  +---------+  +---------+  +---------+  +--------+ |
|  +---------+  +---------+  +---------+  +---------+  +--------+ |
|  |Claims   |  |Verif    |  |Annual   |  |Perm     |  |Public  | |
|  |Discl    |  |Package  |  |Progress |  |Risk     |  |Discl   | |
|  +---------+  +---------+  +---------+  +---------+  +--------+ |
|                                                                  |
|  Tier 3: Workflows (8)                                           |
|  +------+  +------+  +------+  +------+  +------+  +------+     |
|  |Full  |  |FP    |  |CMP   |  |Credit|  |Retire|  |Neutr |     |
|  |Cycle |  |Assess|  |Plan  |  |Proc  |  |ment  |  |alize |     |
|  +------+  +------+  +------+  +------+  +------+  +------+     |
|  +------+  +------+                                              |
|  |Claims|  |Verify|                                              |
|  |Valid  |  |      |                                              |
|  +------+  +------+                                              |
|                                                                  |
|  Tier 2: Engines (10)                                            |
|  +------+  +------+  +------+  +------+  +------+               |
|  |FP    |  |CMP   |  |CQ    |  |PO    |  |RR    |               |
|  |Quant |  |Engine|  |Engine|  |Engine|  |Engine|               |
|  +------+  +------+  +------+  +------+  +------+               |
|  +------+  +------+  +------+  +------+  +------+               |
|  |NB    |  |CS    |  |VP    |  |AC    |  |PR    |               |
|  |Engine|  |Engine|  |Engine|  |Engine|  |Engine|               |
|  +------+  +------+  +------+  +------+  +------+               |
|                                                                  |
|  Tier 1: Integrations (12)                                       |
|  +-------+ +-------+ +-------+ +-------+ +-------+ +-------+   |
|  |Orch   | |MRV    | |GHG   | |DECARB | |DATA   | |Reg    |   |
|  |estrate| |Bridge | |Bridge| |Bridge | |Bridge | |Bridge |   |
|  +-------+ +-------+ +-------+ +-------+ +-------+ +-------+   |
|  +-------+ +-------+ +-------+ +-------+ +-------+ +-------+   |
|  |Market | |Verif  | |P021  | |P023   | |Health | |Setup  |   |
|  |Bridge | |Body   | |Bridge| |Bridge | |Check  | |Wizard |   |
|  +-------+ +-------+ +-------+ +-------+ +-------+ +-------+   |
|                                                                  |
+-----+-------+-------+-------+-------+-------+-------+-----------+
      |       |       |       |       |       |       |
      v       v       v       v       v       v       v
  +------+ +------+ +------+ +------+ +------+ +------+
  |30 MRV| |20 DATA| |10    | |GL-GHG| |6 Reg | |PACK  |
  |Agents| |Agents | |FOUND | |APP   | |istrie| |021/23|
  +------+ +------+ +------+ +------+ +------+ +------+
```

### Engine Abbreviations

| Abbrev | Full Name |
|--------|-----------|
| FP Quant | Footprint Quantification Engine |
| CMP | Carbon Management Plan Engine |
| CQ | Credit Quality Engine |
| PO | Portfolio Optimization Engine |
| RR | Registry Retirement Engine |
| NB | Neutralization Balance Engine |
| CS | Claims Substantiation Engine |
| VP | Verification Package Engine |
| AC | Annual Cycle Engine |
| PR | Permanence Risk Engine |

---

## 2. Component Interaction Diagram

### 10-Phase DAG Pipeline

The Pack Orchestrator executes the following directed acyclic graph. Each phase
depends on the completion of its predecessors. Independent phases may execute
in parallel.

```
Phase 1: Health Check
    |
    v
Phase 2: Configuration
    |
    v
Phase 3: Footprint Quantification  <-- MRV Bridge (30 agents)
    |                                   DATA Bridge (20 agents)
    |                                   GHG App Bridge
    v
Phase 4: Carbon Management Plan    <-- DECARB Bridge
    |
    v
Phase 5: Credit Quality Assessment
    |
    +----------+
    |          |
    v          v
Phase 6:    Phase 7:
Portfolio   Registry           <-- Registry Bridge (6 registries)
Optimize    Retirement             Credit Marketplace Bridge
    |          |
    +----+-----+
         |
         v
Phase 8: Neutralization Balance
    |
    v
Phase 9: Claims Substantiation    <-- VCMI Claims Code validation
    |
    v
Phase 10: Verification Package    <-- Verification Body Bridge
    |                                  ISAE 3410 evidence assembly
    v
  [COMPLETE]
```

### Phase Dependencies

```python
PHASE_DEPENDENCIES = {
    "health_check":        [],
    "configuration":       ["health_check"],
    "footprint":           ["configuration"],
    "management_plan":     ["footprint"],
    "credit_quality":      ["management_plan"],
    "portfolio_optimize":  ["credit_quality"],
    "registry_retirement": ["credit_quality"],
    "neutralization":      ["portfolio_optimize", "registry_retirement"],
    "claims":              ["neutralization"],
    "verification":        ["claims"],
}
```

---

## 3. Data Flow

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
               |     Footprint Quantification Engine         |
               |  Total footprint (tCO2e) by scope/category |
               |  Data quality score, uncertainty range      |
               +--------------------------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
    +-------------------------+     +----------------------------+
    | Carbon Management Plan  |     | Credit Quality Engine      |
    | MACC analysis           |     | 12-dim ICVCM CCP scoring   |
    | Reduction trajectory    |     | Quality rating (A+ to F)   |
    | Residual emissions      |     +----------------------------+
    +-------------------------+                 |
                    |                +----------+----------+
                    |                |                     |
                    v                v                     v
    +-------------------------+  +------------------------+
    | Portfolio Optimization   |  | Registry Retirement    |
    | Avoidance/Removal mix   |  | 6-registry tracking    |
    | Nature/Tech allocation  |  | Serial number verify   |
    | Oxford Principles align |  | Double-counting check  |
    +-------------------------+  +------------------------+
                    |                     |
                    +----------+----------+
                               |
                               v
               +--------------------------------------------+
               |     Neutralization Balance Engine           |
               |  Footprint - Credits - Buffer = Balance     |
               |  Surplus/deficit, confidence interval       |
               +--------------------------------------------+
                               |
                               v
               +--------------------------------------------+
               |     Claims Substantiation Engine            |
               |  ISO 14068-1 / PAS 2060 / VCMI tier        |
               |  Qualifying statement generation            |
               +--------------------------------------------+
                               |
                               v
               +--------------------------------------------+
               |     Verification Package Engine             |
               |  ISAE 3410 evidence assembly                |
               |  SHA-256 content hashing                    |
               |  Evidence index and gap analysis            |
               +--------------------------------------------+
                               |
                               v
               +--------------------------------------------+
               |     Annual Cycle Engine                     |
               |  Milestone tracking                         |
               |  Year-over-year comparison                  |
               |  Renewal scheduling                         |
               +--------------------------------------------+
```

### Data Models

The pack uses Pydantic v2 models for all data structures. Key models include:

| Model | Description | Source Engine |
|-------|-------------|-------------|
| `FootprintResult` | Scope 1/2/3 emissions breakdown | Footprint Quantification |
| `ManagementPlanResult` | Reduction trajectory and MACC | Carbon Mgmt Plan |
| `CreditQualityResult` | 12-dimension scores and rating | Credit Quality |
| `PortfolioResult` | Optimized credit portfolio | Portfolio Optimization |
| `RetirementResult` | Registry retirement records | Registry Retirement |
| `NeutralizationResult` | Balance calculation | Neutralization Balance |
| `ClaimsResult` | Validated claims and statements | Claims Substantiation |
| `VerificationPackageResult` | Evidence package with SHA-256 | Verification Package |
| `AnnualCycleResult` | Cycle status and milestones | Annual Cycle |
| `PermanenceRiskResult` | Risk scores and buffer needs | Permanence Risk |

---

## 4. Integration Points

### External System Integration

```
+-------------------+     +------------------+     +------------------+
|  Carbon Credit     |     |  Registry APIs   |     |  Verification    |
|  Marketplaces      |     |  (6 registries)  |     |  Bodies          |
+--------+----------+     +--------+---------+     +--------+---------+
         |                          |                        |
         v                          v                        v
+--------+----------+     +--------+---------+     +--------+---------+
| Credit Marketplace |     | Registry Bridge  |     | Verification     |
| Bridge             |     | (Verra, GS,      |     | Body Bridge      |
| - Price discovery  |     |  ACR, CAR,       |     | - Engagement     |
| - Availability     |     |  Puro.earth,     |     | - Evidence pkg   |
| - Procurement      |     |  Isometric)      |     | - Opinion track  |
+-------------------+     +------------------+     +------------------+
```

### Registry API Specifications

| Registry | Protocol | Authentication | Rate Limit |
|----------|----------|----------------|------------|
| Verra VCS | REST API | API key | 100 req/min |
| Gold Standard | REST API | OAuth 2.0 | 60 req/min |
| ACR | REST API | API key | 50 req/min |
| CAR | REST API | API key | 50 req/min |
| Puro.earth | REST API | API key | 30 req/min |
| Isometric | GraphQL | Bearer token | 60 req/min |

### GreenLang Agent Integration

| Agent Layer | Agents | Bridge | Purpose |
|-------------|--------|--------|---------|
| AGENT-MRV | 30 agents | MRV Bridge | Scope 1/2/3 emissions calculation |
| AGENT-DATA | 20 agents | Data Bridge | Data intake and quality management |
| AGENT-FOUND | 10 agents | (Direct) | Platform services (orchestration, schema, auth) |

### Optional Pack Bridges

| Pack | Bridge | Graceful Degradation |
|------|--------|---------------------|
| PACK-021 | Pack021Bridge | Falls back to internal baseline calculation |
| PACK-022 | (Not used) | PACK-022 data accessed via PACK-021 bridge |
| PACK-023 | Pack023Bridge | Falls back to internal SBTi alignment check |

---

## 5. Database Schema Overview

PACK-024 introduces 10 new migration files (V084-PACK024-001 through
V084-PACK024-010). All tables use the `gl_cn_` prefix to avoid naming
conflicts.

### Core Tables

```
+----------------------------------+     +----------------------------------+
| gl_cn_footprint                  |     | gl_cn_management_plan            |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| organization_id (UUID, FK)       |     | organization_id (UUID, FK)       |
| reporting_year (INT)             |     | footprint_id (UUID, FK)          |
| base_year (INT)                  |     | planning_horizon_years (INT)     |
| scope1_tco2e (DECIMAL)           |     | reduction_target_pct (DECIMAL)   |
| scope2_location_tco2e (DECIMAL)  |     | residual_tco2e (DECIMAL)         |
| scope2_market_tco2e (DECIMAL)    |     | actions (JSONB)                  |
| scope3_tco2e (DECIMAL)           |     | macc_curve (JSONB)               |
| total_tco2e (DECIMAL)            |     | milestones (JSONB)               |
| consolidation_approach (ENUM)    |     | status (ENUM)                    |
| data_quality_score (DECIMAL)     |     | created_at (TIMESTAMPTZ)         |
| uncertainty_pct (DECIMAL)        |     | updated_at (TIMESTAMPTZ)         |
| provenance_hash (VARCHAR(64))    |     | provenance_hash (VARCHAR(64))    |
| created_at (TIMESTAMPTZ)         |     +----------------------------------+
| updated_at (TIMESTAMPTZ)         |
+----------------------------------+

+----------------------------------+     +----------------------------------+
| gl_cn_credit_quality             |     | gl_cn_portfolio                  |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| credit_id (VARCHAR)              |     | organization_id (UUID, FK)       |
| registry (ENUM)                  |     | reporting_year (INT)             |
| project_type (ENUM)              |     | total_credits_tco2e (DECIMAL)    |
| vintage_year (INT)               |     | avoidance_pct (DECIMAL)          |
| additionality_score (DECIMAL)    |     | removal_pct (DECIMAL)            |
| permanence_score (DECIMAL)       |     | nature_based_pct (DECIMAL)       |
| quantification_score (DECIMAL)   |     | tech_based_pct (DECIMAL)         |
| ... (9 more dimension scores)    |     | avg_quality_score (DECIMAL)      |
| overall_score (DECIMAL)          |     | total_cost (DECIMAL)             |
| quality_rating (VARCHAR(2))      |     | cost_per_tco2e (DECIMAL)         |
| created_at (TIMESTAMPTZ)         |     | oxford_alignment_score (DECIMAL) |
+----------------------------------+     | provenance_hash (VARCHAR(64))    |
                                         +----------------------------------+

+----------------------------------+     +----------------------------------+
| gl_cn_retirement                 |     | gl_cn_neutralization             |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| credit_id (VARCHAR)              |     | organization_id (UUID, FK)       |
| registry (ENUM)                  |     | reporting_year (INT)             |
| serial_number (VARCHAR)          |     | footprint_tco2e (DECIMAL)        |
| quantity_tco2e (DECIMAL)         |     | credits_retired_tco2e (DECIMAL)  |
| retirement_date (DATE)           |     | buffer_tco2e (DECIMAL)           |
| beneficiary_org (VARCHAR)        |     | balance_tco2e (DECIMAL)          |
| retirement_status (ENUM)         |     | balance_method (ENUM)            |
| confirmation_ref (VARCHAR)       |     | status (ENUM)  -- SURPLUS/DEFICIT|
| project_name (VARCHAR)           |     | confidence_interval_pct (DECIMAL)|
| vintage_year (INT)               |     | qualifying_statement (TEXT)      |
| provenance_hash (VARCHAR(64))    |     | provenance_hash (VARCHAR(64))    |
+----------------------------------+     +----------------------------------+

+----------------------------------+     +----------------------------------+
| gl_cn_claim                      |     | gl_cn_verification_package       |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| organization_id (UUID, FK)       |     | organization_id (UUID, FK)       |
| neutralization_id (UUID, FK)     |     | claim_id (UUID, FK)              |
| claim_type (ENUM)                |     | assurance_level (ENUM)           |
| standard_reference (VARCHAR)     |     | evidence_count (INT)             |
| vcmi_tier (ENUM, nullable)       |     | evidence_index (JSONB)           |
| qualifying_statement (TEXT)      |     | sha256_hashes (JSONB)            |
| verification_status (ENUM)       |     | gap_analysis (JSONB)             |
| valid_from (DATE)                |     | package_status (ENUM)            |
| valid_to (DATE)                  |     | delivered_at (TIMESTAMPTZ)       |
| provenance_hash (VARCHAR(64))    |     | provenance_hash (VARCHAR(64))    |
+----------------------------------+     +----------------------------------+

+----------------------------------+     +----------------------------------+
| gl_cn_annual_cycle               |     | gl_cn_permanence_risk            |
|----------------------------------|     |----------------------------------|
| id (UUID, PK)                    |     | id (UUID, PK)                    |
| organization_id (UUID, FK)       |     | credit_id (VARCHAR)              |
| cycle_year (INT)                 |     | portfolio_id (UUID, FK)          |
| cycle_status (ENUM)              |     | permanence_category (ENUM)       |
| milestones (JSONB)               |     | risk_score (DECIMAL)             |
| base_year_recalculated (BOOLEAN) |     | buffer_contribution_pct (DECIMAL)|
| renewal_status (ENUM)            |     | reversal_detected (BOOLEAN)      |
| improvement_actions (JSONB)      |     | climate_hazard_score (DECIMAL)   |
| yoy_comparison (JSONB)           |     | monitoring_status (ENUM)         |
| forward_projections (JSONB)      |     | last_assessed_at (TIMESTAMPTZ)   |
| provenance_hash (VARCHAR(64))    |     | provenance_hash (VARCHAR(64))    |
+----------------------------------+     +----------------------------------+
```

### Indexes

All tables include:
- Primary key index on `id`
- Index on `organization_id` for multi-tenant queries
- Index on `reporting_year` or `cycle_year` for temporal queries
- Index on `created_at` for time-series queries
- Index on `provenance_hash` for reproducibility lookups

Additional specialized indexes:
- `gl_cn_retirement`: Unique index on `(registry, serial_number)` for double-counting prevention
- `gl_cn_credit_quality`: Index on `(quality_rating, overall_score)` for portfolio filtering
- `gl_cn_footprint`: Composite index on `(organization_id, reporting_year)` for annual lookups

### TimescaleDB Hypertables

The following tables are converted to TimescaleDB hypertables for efficient
time-series queries:

- `gl_cn_footprint` (partitioned by `created_at`)
- `gl_cn_annual_cycle` (partitioned by `created_at`)
- `gl_cn_permanence_risk` (partitioned by `last_assessed_at`)

---

## 6. Configuration Architecture

### Configuration Hierarchy

```
+------------------------------------------+
| 4. Runtime Overrides (highest priority)  |
|    Explicit programmatic configuration    |
+------------------------------------------+
                    |
                    v
+------------------------------------------+
| 3. Environment Variables                 |
|    CARBON_NEUTRAL_* prefix               |
+------------------------------------------+
                    |
                    v
+------------------------------------------+
| 2. Neutrality-Type Preset               |
|    config/presets/{type}_neutrality.yaml  |
+------------------------------------------+
                    |
                    v
+------------------------------------------+
| 1. Base pack.yaml (lowest priority)     |
|    Default values and component registry  |
+------------------------------------------+
```

### Configuration Model Structure

```
PackConfig (root)
  |
  +-- CarbonNeutralConfig (pack)
        |
        +-- OrganizationConfig
        |     name, sector, size, neutrality_type
        |
        +-- BoundaryConfig
        |     consolidation_approach, scope_boundaries
        |
        +-- FootprintConfig
        |     base_year, reporting_year, gwp_source
        |
        +-- CarbonManagementPlanConfig
        |     reduction_target, planning_horizon, internal_carbon_price
        |
        +-- CreditQualityConfig
        |     icvcm_weights (12), min_quality_score, project_type_prefs
        |
        +-- PortfolioOptimizationConfig
        |     min_removal_pct, max_nature_pct, vintage_limit
        |
        +-- RegistryRetirementConfig
        |     preferred_registries, auto_retire
        |
        +-- NeutralizationBalanceConfig
        |     balance_method, buffer_pct
        |
        +-- ClaimsSubstantiationConfig
        |     claim_type, standard_reference, vcmi_tier_target
        |
        +-- VerificationPackageConfig
        |     assurance_level, evidence_hash_algorithm
        |
        +-- AnnualCycleConfig
        |     milestone_frequency, recalculation_threshold
        |
        +-- PermanenceRiskConfig
        |     risk_categories, buffer_range
        |
        +-- ReportingConfig
        |     frameworks, output_formats
        |
        +-- PerformanceConfig
        |     cache_ttl, memory_ceiling, parallelism
        |
        +-- AuditTrailConfig
              enabled, retention_days, hash_algorithm
```

---

## 7. Security Architecture

```
+------------------------------------------------------------------+
|                     Security Layers                               |
+------------------------------------------------------------------+
|                                                                  |
|  Layer 1: Authentication (JWT RS256)                             |
|  +-----------------------------------------------------------+  |
|  | Token validation -> Claims extraction -> User identity     |  |
|  +-----------------------------------------------------------+  |
|                              |                                   |
|  Layer 2: Authorization (RBAC)                                   |
|  +-----------------------------------------------------------+  |
|  | Role check -> Permission check -> Resource access control  |  |
|  | Roles: admin, manager, analyst, portfolio_mgr, auditor,    |  |
|  |        viewer                                              |  |
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
|  | All engine operations logged                               |  |
|  | SHA-256 provenance hashing at every pipeline phase         |  |
|  | Immutable audit trail with timestamp and user identity     |  |
|  +-----------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 8. Deployment Architecture

### Kubernetes Deployment

```
+------------------------------------------------------------------+
|                    Kubernetes Cluster (EKS)                       |
+------------------------------------------------------------------+
|                                                                  |
|  +------------------+  +------------------+  +----------------+  |
|  | PACK-024 API     |  | PACK-024 Worker  |  | PACK-024       |  |
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

## 9. Zero-Hallucination Architecture

A key architectural principle of PACK-024 is the strict separation between
deterministic calculation and LLM-assisted processing.

```
+------------------------------------------------------------------+
|                    Calculation Layer                              |
|  (Deterministic - No LLM)                                        |
+------------------------------------------------------------------+
|                                                                  |
|  Emissions:    activity_data * emission_factor * gwp             |
|  ICVCM Score:  sum(dimension_score[i] * weight[i])  for i=1..12 |
|  Portfolio:    constrained optimization with fixed parameters    |
|  Balance:      footprint - credits_retired - buffer              |
|  Risk:         weighted scoring rubric with published parameters |
|  Provenance:   SHA-256 hash of all inputs and outputs            |
|                                                                  |
|  Data sources: DEFRA, EPA, ecoinvent, IPCC AR6, ICVCM CCP,     |
|                VCMI Claims Code, registry API specs              |
|                                                                  |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|                    LLM-Assisted Layer                             |
|  (Classification and narrative only)                             |
+------------------------------------------------------------------+
|                                                                  |
|  - Entity resolution (matching organization names)               |
|  - Activity classification (mapping activities to emission       |
|    categories)                                                   |
|  - Narrative generation (management plan prose, qualifying       |
|    statement language, report narratives)                        |
|  - Recommendation generation (improvement suggestions)          |
|                                                                  |
|  All LLM outputs are tagged with confidence scores and are      |
|  human-reviewable before inclusion in final outputs.             |
|                                                                  |
+------------------------------------------------------------------+
```

---

*Architecture document maintained by GreenLang Platform Team*
*PACK-024 Carbon Neutral Pack v1.0.0*
*Last updated: 2026-03-18*
