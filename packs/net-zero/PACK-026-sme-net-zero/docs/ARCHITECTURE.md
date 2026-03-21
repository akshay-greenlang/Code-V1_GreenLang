# PACK-026 SME Net Zero Pack -- Architecture Document

**Pack ID:** PACK-026-sme-net-zero
**Version:** 1.0.0
**Date:** 2026-03-18
**Author:** GreenLang Platform Engineering

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Design Principles](#design-principles)
3. [6-Phase DAG Pipeline](#6-phase-dag-pipeline)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Three-Tier Data Architecture](#three-tier-data-architecture)
6. [Database Schema Overview](#database-schema-overview)
7. [Engine Architecture Patterns](#engine-architecture-patterns)
8. [Accounting Software Integration Architecture](#accounting-software-integration-architecture)
9. [Grant Database Architecture](#grant-database-architecture)
10. [Security Architecture](#security-architecture)
11. [Deployment Architecture](#deployment-architecture)
12. [Performance Optimization](#performance-optimization)
13. [Scaling Considerations](#scaling-considerations)
14. [Cost Optimization](#cost-optimization)

---

## System Overview

PACK-026 implements a 3-tier architecture optimized for SME constraints: minimal compute resources, limited data, and fast time-to-value.

```
+=========================================================================+
|                        PRESENTATION TIER                                 |
|  +------------------+  +------------------+  +------------------+       |
|  | Mobile Dashboard |  | Web Dashboard    |  | API Endpoints    |       |
|  | (HTML, < 3 sec)  |  | (HTML, Charts)   |  | (REST JSON)      |       |
|  +------------------+  +------------------+  +------------------+       |
+=========================================================================+
|                        APPLICATION TIER                                   |
|  +-------------------------------------------------------------------+  |
|  |                      Pack Orchestrator                             |  |
|  |  +----------+ +----------+ +----------+ +----------+ +----------+ | |
|  |  | Express  | | Standard | | Quick    | | Grant    | | Cert     | | |
|  |  | Onboard  | | Setup    | | Wins     | | Finder   | | Pathway  | | |
|  |  | Workflow | | Workflow | | Workflow | | Workflow | | Workflow | | |
|  |  +----------+ +----------+ +----------+ +----------+ +----------+ | |
|  +-------------------------------------------------------------------+  |
|  +-------------------------------------------------------------------+  |
|  |                         8 Engines                                  |  |
|  |  +----------+ +----------+ +----------+ +----------+             |  |
|  |  | Baseline | | Target   | | Quick    | | Action   |             |  |
|  |  | Engine   | | Engine   | | Wins     | | Priority |             |  |
|  |  +----------+ +----------+ +----------+ +----------+             |  |
|  |  +----------+ +----------+ +----------+ +----------+             |  |
|  |  | Progress | | Cost-    | | Grant    | | Cert     |             |  |
|  |  | Tracker  | | Benefit  | | Finder   | | Readines |             |  |
|  |  +----------+ +----------+ +----------+ +----------+             |  |
|  +-------------------------------------------------------------------+  |
+=========================================================================+
|                           DATA TIER                                      |
|  +------------------+  +------------------+  +------------------+       |
|  | PostgreSQL 16    |  | Redis 7          |  | Emission Factor  |       |
|  | 16 tables, RLS   |  | Cache, Sessions  |  | Database (const) |       |
|  +------------------+  +------------------+  +------------------+       |
|  +------------------+  +------------------+  +------------------+       |
|  | Accounting APIs  |  | Grant Database   |  | Cert Registry    |       |
|  | Xero/QB/Sage     |  | 50+ programs     |  | SCH/BCorp/ISO    |       |
|  +------------------+  +------------------+  +------------------+       |
+=========================================================================+
```

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **Lightweight deployment** (512MB-2GB RAM) | SMEs cannot afford dedicated infrastructure; must run on shared or low-cost hosting |
| **Synchronous calculation** (no queuing) | SME baselines complete in seconds; async complexity not justified |
| **Deterministic calculations** (zero-hallucination) | No LLM in numeric paths; all emission factors are constants |
| **Progressive data quality** (Bronze/Silver/Gold) | SMEs start with minimal data and improve over time |
| **Spend-based Scope 3** (not activity-based) | SMEs lack supply chain activity data; financial data is universally available |
| **Pre-populated industry averages** | Reduces data collection burden; industry benchmarks provide instant context |
| **Mobile-first dashboards** | 40%+ of SME sessions are on mobile devices |
| **OAuth2 for accounting** (never store passwords) | Security and trust; accounting data is the most sensitive |

---

## Design Principles

### 1. Time-Optimized

Every interaction is designed for SME time constraints:

```
Express Onboarding:     15 minutes (4 phases)
Standard Setup:         1-2 hours (6 phases)
Quarterly Review:       15-30 minutes (4 phases)
Grant Application:      45 minutes (4 phases)
```

No workflow takes longer than 2 hours. Most complete in under 30 minutes.

### 2. Zero-Hallucination Calculations

All numeric calculations use deterministic formulas with published emission factors:

```python
# Example: Scope 2 electricity calculation
# NO LLM, NO estimation, NO randomness

electricity_kwh = annual_spend_gbp / ENERGY_COST_PER_KWH["electricity"]  # 0.28 GBP/kWh
scope2_tco2e = (electricity_kwh * GRID_EF_KGCO2E_PER_KWH["UK"]) / 1000  # 0.2070 kgCO2e/kWh

# Source: DEFRA 2024 conversion factors
# Provenance: SHA-256 hash of all inputs and outputs
```

### 3. Progressive Data Quality

The system works with whatever data the SME has:

```
Bronze (5 min)  -->  Silver (1 hr)  -->  Gold (2-4 hrs)
     |                    |                    |
     v                    v                    v
Industry averages    Utility bills       Activity data
+/- 40% accuracy     +/- 15% accuracy   +/- 5% accuracy
```

### 4. Cost-First Framing

Every action is presented with financial impact first:

```
"Switch to LED lighting"
   Saves: GBP 960/year
   Costs: GBP 1,500 upfront
   Payback: 18 months
   Reduces: 2.0 tCO2e/year
```

Not "reduces 2.0 tCO2e" (which means nothing to most SME owners).

### 5. Auditable by Design

Every calculation produces a provenance hash:

```python
provenance = {
    "input_hash": sha256(input_data),
    "emission_factors": {
        "source": "DEFRA 2024",
        "electricity_ef": 0.2070,
        "gas_ef": 0.18293,
    },
    "calculation_version": "26.0.0",
    "output_hash": sha256(result),
    "timestamp": "2026-03-18T14:30:00Z",
}
```

---

## 6-Phase DAG Pipeline

PACK-026 implements a Directed Acyclic Graph (DAG) pipeline with 6 phases. Each phase can be executed independently or as part of a complete workflow.

```
              +-------------------------------------------+
              |          PACK-026 DAG PIPELINE             |
              +-------------------------------------------+
              |                                           |
              |    Phase 1          Phase 2                |
              |    ONBOARDING  ---> BASELINE               |
              |    [Profile]        [Bronze/Silver/Gold]    |
              |        |                |                  |
              |        |                v                  |
              |        |           Phase 3                 |
              |        +---------> TARGETS                 |
              |                    [SBTi SME / SCH]        |
              |                        |                  |
              |                        v                  |
              |                   Phase 4                  |
              |                   QUICK WINS               |
              |                   [500+ actions, ranked]   |
              |                     /     \               |
              |                    v       v              |
              |              Phase 5    Phase 6           |
              |              GRANTS &   REPORTING         |
              |              CERTS      [Dashboard,       |
              |              [Match,     Reports,         |
              |               Apply]     Progress]        |
              |                                           |
              +-------------------------------------------+

Dependencies:
  Phase 2 depends on Phase 1 (needs profile)
  Phase 3 depends on Phase 2 (needs baseline)
  Phase 4 depends on Phase 2 + Phase 3 (needs baseline + targets)
  Phase 5 depends on Phase 1 + Phase 4 (needs profile + actions)
  Phase 6 depends on Phase 2 + Phase 3 + Phase 4 (needs all calculation outputs)
```

### Phase Execution Modes

| Mode | Phases Executed | Duration | Use Case |
|------|----------------|----------|----------|
| Express | 1 -> 2 -> 3 -> 4 | 15 min | First-time onboarding |
| Standard | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1-2 hrs | Full setup with accounting data |
| Review | 2 -> 3 -> 6 | 15-30 min | Quarterly progress update |
| Grants | 5 (standalone) | 45 min | Grant discovery and application |
| Certification | 5 (cert only) | 1 hr | Certification readiness assessment |
| Report | 6 (standalone) | 5 min | Generate/refresh reports |

---

## Data Flow Diagrams

### Express Onboarding (15-minute path)

```
USER INPUT                    PROCESSING                      OUTPUT
+-----------+                 +-----------+                   +----------+
| Company   |                 | Profile   |                   | Baseline |
| Name      |---+             | Validator |                   | Report   |
| Sector    |   |             +-----------+                   | (1-2 pg) |
| Employees |   |                  |                          +----------+
| Country   |   |                  v                               ^
+-----------+   |   +---+    +-----------+    +-----------+       |
                +-->| 1 |--->| Baseline  |--->| Target    |---+   |
+-----------+   |   +---+    | Engine    |    | Engine    |   |   |
| Elec Spend|   |    ^       | (Bronze)  |    | (SBTi     |   |   |
| Gas Spend |---+    |       +-----------+    |  SME)     |   |   |
| Travel Sp |       Profile       |           +-----------+   |   |
+-----------+       validated     |                |          |   |
                                  v                v          |   |
                            +-----------+    +-----------+    |   |
                            | DEFRA/IEA |    | Quick Win |    |   |
                            | Emission  |    | Engine    |----+   |
                            | Factors   |    | (500+ DB) |    |   |
                            +-----------+    +-----------+    |   |
                                                  |           |   |
                                                  v           v   |
                                            +-----------+  +------+--+
                                            | Top 5     |  | Report  |
                                            | Actions   |  | Templat |
                                            | Ranked    |  | Engine  |
                                            +-----------+  +---------+
```

### Standard Setup (1-2 hour path)

```
USER                  ACCOUNTING SOFTWARE              PACK-026
+--------+            +-----------+                    +-----------+
| Auth   |--OAuth2--->| Xero/QB/  |--P&L data-------->| Classify  |
| Consent|            | Sage API  |                    | Spend     |
+--------+            +-----------+                    +-----+-----+
                                                             |
+--------+            +-----------+                    +-----v-----+
| Utility|--Upload--->| Data      |--kWh, m3---------->| Silver    |
| Bills  |            | Extractor |                    | Baseline  |
+--------+            +-----------+                    | Engine    |
                                                       +-----+-----+
+--------+            +-----------+                          |
| Fleet  |--Upload--->| Fleet     |--km, litres------->+    |
| Data   |            | Parser    |                    |    |
+--------+            +-----------+                    |    v
                                                  +----+----+-----+
+--------+            +-----------+               | Scope 1/2/3   |
| Travel |--Upload--->| Travel    |--journeys---->| Consolidation |
| Records|            | Parser    |               | (Gold Tier)   |
+--------+            +-----------+               +-------+-------+
                                                          |
                                                          v
                                               +----------+---------+
                                               |   Standard Output  |
                                               |  - Baseline Report |
                                               |  - Targets         |
                                               |  - Quick Wins      |
                                               |  - Grant Matches   |
                                               |  - Cert Readiness  |
                                               +--------------------+
```

### Quarterly Review Cycle

```
QUARTER START                REVIEW (15-30 min)              QUARTER END

+--------+                   +-----------+                   +--------+
| Collect|                   | Compare   |                   | Update |
| energy |--input data------>| vs Target |                   | status |
| bills  |                   | vs Prior  |                   | in DB  |
+--------+                   +-----+-----+                   +--------+
                                   |                              ^
+--------+                   +-----v-----+                        |
| Update |                   | RAG       |                        |
| action |--status---------->| Status    |--if off-track---+     |
| status |                   | Engine    |                 |     |
+--------+                   +-----+-----+                 v     |
                                   |              +---------+---+ |
                                   |              | Corrective  | |
                                   v              | Action      | |
                             +-----------+        | Recommender |-+
                             | Dashboard |        +-------------+
                             | Update    |
                             | (KPIs)    |
                             +-----------+
```

### Grant Application Workflow

```
SME PROFILE                  MATCHING                       APPLICATION
+-----------+                +-----------+                  +-----------+
| Country   |                | Grant DB  |                  | Pre-fill  |
| Region    |---profile----->| (50+      |---top matches--->| Template  |
| Sector    |                | Programs) |                  | Engine    |
| Size      |                +-----------+                  +-----+-----+
| Actions   |                     |                               |
+-----------+                     v                               v
                            +-----------+                  +-----------+
                            | Eligiblty |                  | Applicatn |
                            | Checker   |                  | Package   |
                            | (criteria |                  | (PDF, 3-5 |
                            |  matrix)  |                  |  pages)   |
                            +-----------+                  +-----------+
                                  |                               |
                                  v                               v
                            +-----------+                  +-----------+
                            | Score     |                  | Deadline  |
                            | 0-100     |                  | Tracker   |
                            | per match |                  | (remindrs)|
                            +-----------+                  +-----------+
```

---

## Three-Tier Data Architecture

### Tier Model

```
+------------------------------------------------------------------------+
|                      THREE-TIER DATA MODEL                              |
+------------------------------------------------------------------------+
|                                                                        |
|  BRONZE TIER                 SILVER TIER              GOLD TIER        |
|  (Industry Averages)         (Hybrid)                 (Activity-Based) |
|                                                                        |
|  Input:                      Input:                   Input:           |
|  - Revenue                   - Utility bills          - Meter data     |
|  - Headcount                 - Energy spend           - Fleet logs     |
|  - NACE sector               - Travel spend           - Travel records |
|  - Country                   - Procurement spend      - Supplier data  |
|                              - Waste spend            - Waste records  |
|                                                                        |
|  Method:                     Method:                  Method:          |
|  tCO2e/employee x count     kWh from bills x EF      Activity x EF    |
|  tCO2e/1000GBP x revenue    + spend-based S3         for all scopes   |
|                                                                        |
|  Accuracy: +/- 40%          Accuracy: +/- 15%        Accuracy: +/- 5% |
|  Time: 5 minutes            Time: 30-60 minutes      Time: 2-4 hours  |
|                                                                        |
|  Suitable for:              Suitable for:             Suitable for:    |
|  - First estimate           - SBTi SME submission    - ISO 14001      |
|  - SME Climate Hub          - B Corp application     - Carbon Trust   |
|  - Quick wins ID            - Annual reporting       - Verification   |
|                                                                        |
+------------------------------------------------------------------------+
```

### Data Quality Scoring Algorithm

Each data source receives a quality score (0-100) based on:

```python
QUALITY_WEIGHTS = {
    "data_source_type": 0.30,      # Primary (meter) > Secondary (bill) > Tertiary (average)
    "data_completeness": 0.25,      # 12 months > 6 months > 3 months > estimated
    "data_recency": 0.20,          # Current year > Prior year > 2+ years ago
    "emission_factor_specificity": 0.15,  # Supplier-specific > country > global
    "verification_status": 0.10,    # Third-party verified > self-declared > estimated
}

# Tier boundaries
# Gold:   score >= 80
# Silver: score >= 50
# Bronze: score < 50
```

### Upgrade Path Between Tiers

```
Bronze Baseline                Silver Baseline              Gold Baseline
(5 min, 5 questions)           (+30-60 min, utility bills)  (+2-4 hrs, activity data)
       |                              |                            |
       v                              v                            v
Total = EF x employees         Scope 1 = gas_kWh x EF       Scope 1 = metered_kWh x EF
       |                       Scope 2 = elec_kWh x EF      Scope 2 = metered_kWh x EF
       |                       Scope 3 = spend x EEIO       Scope 3 = activity x EF
       |                              |                            |
       v                              v                            v
Accuracy +/- 40%               Accuracy +/- 15%             Accuracy +/- 5%
```

---

## Database Schema Overview

### Entity Relationship Diagram

```
+---------------------------+         +---------------------------+
| gl_sme_organizations      |         | gl_sme_profiles           |
|---------------------------|         |---------------------------|
| PK  id (UUID)             |<-----+  | PK  id (UUID)             |
|     tenant_id (UUID)      |      |  | FK  organization_id       |
|     name (VARCHAR)        |      |  |     employee_count (INT)  |
|     sector (VARCHAR)      |      |  |     revenue (DECIMAL)     |
|     country (VARCHAR)     |      |  |     company_size (VARCHAR)|
|     created_at (TIMESTZ)  |      |  |     nace_code (VARCHAR)   |
+---------------------------+      |  |     profile_year (INT)    |
             |                     |  +---------------------------+
             |                     |
     +-------+-------+            |
     |               |            |
     v               v            |
+------------------+  +------------------+
| gl_sme_baselines |  | gl_sme_targets   |
|------------------|  |------------------|
| PK  id (UUID)    |  | PK  id (UUID)    |
| FK  org_id       |  | FK  org_id       |
|     tier (ENUM)  |  |     pathway      |
|     scope1_tco2e |  |     base_year    |
|     scope2_tco2e |  |     near_term_yr |
|     scope3_tco2e |  |     near_term_%  |
|     total_tco2e  |  |     long_term_yr |
|     per_employee |  |     long_term_%  |
|     base_year    |  |     annual_rate  |
|     dq_score     |  +------------------+
+------------------+          |
         |                    |
         v                    v
+------------------+  +------------------+
| gl_sme_baseline_ |  | gl_sme_milestones|
| details          |  |------------------|
|------------------|  | PK  id (UUID)    |
| PK  id (UUID)    |  | FK  target_id    |
| FK  baseline_id  |  |     year (INT)   |
|     scope (ENUM) |  |     target_tco2e |
|     category     |  |     reduction_%  |
|     tco2e        |  +------------------+
|     source_type  |
+------------------+

+------------------+  +------------------+  +------------------+
| gl_sme_quick_wins|  | gl_sme_progress  |  | gl_sme_grant_    |
|------------------|  |------------------|  | matches          |
| PK  id (UUID)    |  | PK  id (UUID)    |  |------------------|
| FK  org_id       |  | FK  org_id       |  | PK  id (UUID)    |
|     action_id    |  |     year (INT)   |  | FK  org_id       |
|     rank (INT)   |  |     total_tco2e  |  | FK  program_id   |
|     title        |  |     vs_baseline% |  |     match_score  |
|     savings_tco2e|  |     vs_prior%    |  |     eligible     |
|     savings_gbp  |  |     per_employee |  |     deadline     |
|     impl_cost    |  |     rag_status   |  |     award_range  |
|     payback_mos  |  +------------------+  +------------------+
|     difficulty   |          |
+------------------+          v
         |            +------------------+  +------------------+
         v            | gl_sme_kpi_      |  | gl_sme_grant_    |
+------------------+  | history          |  | programs         |
| gl_sme_action_   |  |------------------|  |------------------|
| status           |  | PK  id (UUID)    |  | PK  id (UUID)    |
|------------------|  | FK  progress_id  |  |     name         |
| PK  id (UUID)    |  |     kpi_name     |  |     provider     |
| FK  quick_win_id |  |     value        |  |     country      |
|     status       |  |     trend        |  |     sector       |
|     updated_at   |  +------------------+  |     min_award    |
+------------------+                        |     max_award    |
                                            |     deadline     |
+------------------+  +------------------+  +------------------+
| gl_sme_cost_     |  | gl_sme_          |
| benefit          |  | certifications   |
|------------------|  |------------------|
| PK  id (UUID)    |  | PK  id (UUID)    |
| FK  quick_win_id |  | FK  org_id       |
|     npv_5yr      |  |     cert_type    |
|     npv_10yr     |  |     readiness_%  |
|     irr          |  |     gaps         |
|     payback_disc |  |     est_cost     |
|     net_of_grant |  |     est_time     |
+------------------+  +------------------+
         |                    |
         v                    v
+------------------+  +------------------+
| gl_sme_financial_|  | gl_sme_cert_     |
| metrics          |  | readiness        |
|------------------|  |------------------|
| PK  id (UUID)    |  | PK  id (UUID)    |
| FK  cost_ben_id  |  | FK  cert_id      |
|     metric_name  |  |     criterion    |
|     value        |  |     status       |
|     horizon_yrs  |  |     gap_detail   |
+------------------+  +------------------+
```

### Views

```sql
-- vw_sme_dashboard: Mobile dashboard data (materialized, refreshed hourly)
CREATE MATERIALIZED VIEW vw_sme_dashboard AS
SELECT
    o.id AS org_id,
    o.name AS org_name,
    b.total_tco2e,
    b.per_employee_tco2e,
    b.tier AS data_tier,
    t.near_term_reduction_pct,
    t.near_term_year,
    p.rag_status,
    p.change_vs_baseline_pct,
    COUNT(qw.id) AS total_actions,
    COUNT(qw.id) FILTER (WHERE qs.status = 'completed') AS completed_actions
FROM gl_sme_organizations o
LEFT JOIN gl_sme_baselines b ON b.org_id = o.id AND b.is_latest = true
LEFT JOIN gl_sme_targets t ON t.org_id = o.id AND t.is_active = true
LEFT JOIN gl_sme_progress p ON p.org_id = o.id AND p.is_latest = true
LEFT JOIN gl_sme_quick_wins qw ON qw.org_id = o.id
LEFT JOIN gl_sme_action_status qs ON qs.quick_win_id = qw.id
GROUP BY o.id, o.name, b.total_tco2e, b.per_employee_tco2e,
         b.tier, t.near_term_reduction_pct, t.near_term_year,
         p.rag_status, p.change_vs_baseline_pct;

-- vw_sme_action_plan: Combined action plan with cost-benefit data
-- vw_sme_benchmark: Peer comparison data (anonymized, sector-level)
```

---

## Engine Architecture Patterns

### Common Engine Pattern

All 8 engines follow the same architectural pattern:

```python
class SMEEngine:
    """
    Standard PACK-026 engine pattern.

    1. Pydantic v2 input model with validation
    2. Deterministic calculation (no LLM)
    3. Pydantic v2 output model with provenance hash
    4. Logging at INFO level for audit trail
    5. Error handling with graceful degradation
    """

    ENGINE_ID: str = "engine_name"
    VERSION: str = "26.0.0"
    PACK_ID: str = "PACK-026"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"pack026.{self.ENGINE_ID}")

    async def calculate(self, input_data: InputModel) -> OutputModel:
        """
        Main calculation entry point.

        1. Validate input (Pydantic)
        2. Look up emission factors (deterministic constants)
        3. Perform calculations (arithmetic only)
        4. Compute provenance hash (SHA-256)
        5. Return validated output (Pydantic)
        """
        started = datetime.now(timezone.utc)

        # Validate
        validated_input = InputModel(**input_data.model_dump())

        # Calculate (deterministic)
        result = self._calculate_internal(validated_input)

        # Provenance
        result.provenance_hash = self._compute_hash(result)
        result.calculation_duration_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000

        # Log
        self.logger.info("Calculated %s: %.2f tCO2e in %.0fms",
                        self.ENGINE_ID, result.total_tco2e, result.calculation_duration_ms)

        return result
```

### Emission Factor Lookup Pattern

Emission factors are compile-time constants, never fetched at runtime:

```python
# All emission factors are module-level constants
# Source: DEFRA 2024, IEA 2024, EPA 2024
# Updated annually with pack version

GRID_EF_KGCO2E_PER_KWH = {
    "UK":     0.2070,   # DEFRA 2024
    "EU-AVG": 0.2556,   # IEA 2024
    "US-AVG": 0.3710,   # EPA eGRID 2024
    "GLOBAL": 0.4940,   # IEA World Average 2024
}

GAS_EF_KGCO2E_PER_KWH = 0.18293  # DEFRA 2024

# These are NEVER fetched from a database or API at calculation time.
# They are versioned with the pack and updated via pack upgrades.
```

### Calculation Isolation Pattern

Each engine is isolated -- no engine calls another engine directly:

```
+----------+     +----------+     +----------+
| Baseline |     | Target   |     | QuickWin |
| Engine   |     | Engine   |     | Engine   |
+----+-----+     +----+-----+     +----+-----+
     |                |                |
     v                v                v
+------------------------------------------------+
|              Pack Orchestrator                  |
|  Routes data between engines via workflow DAG   |
+------------------------------------------------+
```

Engines communicate only through the Pack Orchestrator, which passes output models from one engine as input to the next.

---

## Accounting Software Integration Architecture

### OAuth2 Flow

```
+--------+     +----------+     +------------------+     +--------+
|  User  |---->| GreenLng |---->| Xero/QB/Sage     |---->| User   |
| (click |     | (redirect|     | (OAuth2 consent  |     | (grant |
|  link) |     |  to auth)|     |  screen)         |     |  OK)   |
+--------+     +----------+     +------------------+     +---+----+
                                                             |
                +--------------------------------------------+
                | auth_code
                v
          +----------+     +------------------+
          | GreenLng |---->| Xero/QB/Sage     |
          | (exchange|     | (token endpoint) |
          |  code)   |     |                  |
          +----+-----+     +------+-----------+
               |                  |
               |    access_token, refresh_token
               |<-----------------+
               |
               v
          +----------+
          | Store in |
          | Vault    |
          | (encryp) |
          +----------+
```

### Data Sync Architecture

```
+------------------+
| Accounting API   |
| (Xero/QB/Sage)   |
+--------+---------+
         |
         | Raw P&L / Transactions
         v
+------------------+
| Data Normalizer  |
| - Date format    |
| - Currency       |
| - Account codes  |
+--------+---------+
         |
         | Normalized transactions
         v
+------------------+
| Spend Classifier |
| - GL code map    |
| - Regex patterns |
| - ML fallback    |
+--------+---------+
         |
         | Classified spend by Scope 3 category
         v
+------------------+
| Emission Calc    |
| - EEIO factors   |
| - Scope mapping   |
+--------+---------+
         |
         | tCO2e by category
         v
+------------------+
| Baseline Update  |
| - Silver/Gold    |
| - Provenance     |
+------------------+
```

### GL Code Mapping Architecture

The spend classifier maps General Ledger codes to emission categories using a three-level hierarchy:

```
Level 1: Exact GL code match (custom mapping per organization)
    |
    v (no match)
Level 2: GL code range match (default industry mapping)
    |
    v (no match)
Level 3: Transaction description keyword match (regex patterns)
    |
    v (no match)
Level 4: Default to "other purchased services" (Cat 1)
```

### Sync Modes

| Mode | Frequency | Latency | Use Case |
|------|-----------|---------|----------|
| Monthly Batch | 1st of month | 24 hours | Free tier, sufficient for annual reporting |
| Weekly Batch | Every Monday | 7 days | Silver tier, quarterly reporting |
| Real-Time | Transaction-level | < 1 hour | Gold tier, continuous monitoring |

---

## Grant Database Architecture

### Database Structure

```
+------------------+     +------------------+     +------------------+
| Grant Programs   |     | Eligibility      |     | Match Results    |
| (50+ records)    |     | Criteria         |     | (per-SME)        |
|------------------|     |------------------|     |------------------|
| program_id       |     | program_id       |     | match_id         |
| name             |<--->| country          |     | program_id       |
| provider         |     | region           |<--->| org_id           |
| description      |     | sector_codes[]   |     | match_score      |
| min_award        |     | max_employees    |     | eligible (bool)  |
| max_award        |     | max_revenue      |     | reasons[]        |
| application_url  |     | action_types[]   |     | created_at       |
| deadline         |     | exclusions[]     |     +------------------+
| last_verified    |     +------------------+
+------------------+
```

### Matching Algorithm

```python
def calculate_match_score(sme_profile, grant_program):
    score = 0
    max_score = 100

    # Country match (mandatory, 0 or 25 points)
    if sme_profile.country in grant_program.eligible_countries:
        score += 25
    else:
        return 0  # Ineligible

    # Sector match (20 points)
    if sme_profile.nace_code in grant_program.eligible_sectors:
        score += 20
    elif sme_profile.sector in grant_program.eligible_sector_groups:
        score += 10  # Partial match

    # Size match (15 points)
    if sme_profile.employee_count <= grant_program.max_employees:
        score += 15

    # Action match (25 points)
    matching_actions = set(sme_profile.planned_actions) & set(grant_program.supported_actions)
    if matching_actions:
        score += min(25, len(matching_actions) * 5)

    # Region match (10 points, bonus)
    if sme_profile.region in grant_program.eligible_regions:
        score += 10

    # Timing (5 points)
    if grant_program.deadline > today + 30_days:
        score += 5  # Enough time to apply

    return min(score, max_score)
```

### Sync Schedule

```
+------------------+     +------------------+     +------------------+
| Government APIs  |     | Grant Sync Job   |     | Local Database   |
| (gov.uk, ec.eu,  |     | (monthly cron)   |     | (PostgreSQL)     |
| grants.gov)      |---->| - Fetch new      |---->| - Upsert grants  |
|                  |     | - Update deadlins|     | - Mark expired   |
|                  |     | - Verify URLs    |     | - Re-match SMEs  |
+------------------+     +------------------+     +------------------+
                                |
                                v
                         +------------------+
                         | Notification     |
                         | - New grants     |
                         | - Deadline alerts|
                         | - Match updates  |
                         +------------------+
```

---

## Security Architecture

### Authentication Flow

```
Client              PACK-026 API             Auth Service           Database
  |                     |                        |                     |
  |-- POST /login ----->|                        |                     |
  |                     |-- validate creds ----->|                     |
  |                     |<-- JWT (RS256) --------|                     |
  |<-- 200 + JWT -------|                        |                     |
  |                     |                        |                     |
  |-- GET /baseline --->|                        |                     |
  | (Authorization:     |                        |                     |
  |  Bearer JWT)        |-- verify signature --->|                     |
  |                     |<-- claims (role,       |                     |
  |                     |    tenant_id) ---------|                     |
  |                     |                        |                     |
  |                     |-- SET current_tenant --|--------------------->|
  |                     |-- SELECT with RLS -----|--------------------->|
  |                     |<-- data (tenant-scoped)|<--------------------|
  |<-- 200 + data ------|                        |                     |
```

### RBAC Model (5 Roles)

```
+------------------------------------------------------------------+
|                        RBAC PERMISSION MATRIX                     |
+------------------------------------------------------------------+
|                    | Owner | Manager | Viewer | Advisor | Admin   |
|--------------------|:-----:|:-------:|:------:|:-------:|:-------:|
| View baselines     |  Yes  |   Yes   |  Yes   |   Yes   |  Yes    |
| Create baselines   |  Yes  |   Yes   |   No   |   No    |  Yes    |
| Update baselines   |  Yes  |   Yes   |   No   |   No    |  Yes    |
| Delete baselines   |  Yes  |   No    |   No   |   No    |  Yes    |
| View targets       |  Yes  |   Yes   |  Yes   |   Yes   |  Yes    |
| Set targets        |  Yes  |   Yes   |   No   |   No    |  Yes    |
| View quick wins    |  Yes  |   Yes   |  Yes   |   Yes   |  Yes    |
| Update action stat |  Yes  |   Yes   |   No   |   No    |  Yes    |
| View financials    |  Yes  |   Yes   |   No   |   No    |  Yes    |
| Export reports     |  Yes  |   Yes   |  Yes   |   Yes   |  Yes    |
| Manage accounting  |  Yes  |   No    |   No   |   No    |  Yes    |
| View audit log     |  Yes  |   Yes   |   No   |   Yes   |  Yes    |
| Manage users       |  Yes  |   No    |   No   |   No    |  Yes    |
| Change config      |  Yes  |   No    |   No   |   No    |  Yes    |
+------------------------------------------------------------------+
```

### Encryption Layers

```
+------------------------------------------------------------------+
|                      ENCRYPTION ARCHITECTURE                      |
+------------------------------------------------------------------+
|                                                                    |
|  Layer 1: TLS 1.3 (in transit)                                   |
|  +--------------------------------------------------------------+ |
|  | All API traffic encrypted end-to-end                          | |
|  | Certificate: Let's Encrypt or enterprise CA                   | |
|  | Minimum TLS 1.3, no fallback to TLS 1.2                     | |
|  +--------------------------------------------------------------+ |
|                                                                    |
|  Layer 2: AES-256-GCM (at rest)                                  |
|  +--------------------------------------------------------------+ |
|  | Financial data columns encrypted in PostgreSQL               | |
|  | Encrypted fields: revenue, spend amounts, cost data          | |
|  | Key management: HashiCorp Vault with auto-rotation           | |
|  +--------------------------------------------------------------+ |
|                                                                    |
|  Layer 3: OAuth2 Tokens (secrets)                                |
|  +--------------------------------------------------------------+ |
|  | Accounting API tokens stored in HashiCorp Vault              | |
|  | Refresh tokens encrypted with per-tenant key                 | |
|  | Token rotation enforced on every refresh                     | |
|  +--------------------------------------------------------------+ |
|                                                                    |
|  Layer 4: SHA-256 (provenance)                                   |
|  +--------------------------------------------------------------+ |
|  | All calculation outputs hashed for integrity                 | |
|  | Input hash + output hash + version = provenance chain        | |
|  | Tamper detection on any stored calculation result             | |
|  +--------------------------------------------------------------+ |
|                                                                    |
+------------------------------------------------------------------+
```

---

## Deployment Architecture

### Lightweight Deployment (Recommended for SME)

```
+------------------------------------------------------------------+
|                    LIGHTWEIGHT DEPLOYMENT                          |
|                    (Single Server / VPS)                          |
+------------------------------------------------------------------+
|                                                                    |
|  +------+  +--------+  +--------+  +-------+                     |
|  | Nginx|  | Python |  | Postgre|  | Redis |                     |
|  | Proxy|->| App    |->| SQL 16 |  | 7     |                     |
|  | :443 |  | :8000  |  | :5432  |  | :6379 |                     |
|  +------+  +--------+  +--------+  +-------+                     |
|                                                                    |
|  Resources:                                                       |
|  - CPU: 2 cores (minimum)                                        |
|  - RAM: 2 GB (minimum 512 MB for app, 512 MB for PG, 256 MB Redis)|
|  - Disk: 20 GB (10 GB data, 10 GB OS/app)                       |
|  - Network: 10 Mbps (accounting API sync)                        |
|                                                                    |
|  Cost: ~$20-50/month on DigitalOcean, Hetzner, or AWS Lightsail |
|                                                                    |
+------------------------------------------------------------------+
```

### Container Deployment

```yaml
# docker-compose.yml for PACK-026
version: "3.8"
services:
  app:
    image: greenlang/pack-026:1.0.0
    ports: ["8000:8000"]
    environment:
      DATABASE_URL: postgresql://pack026:secret@db:5432/greenlang
      REDIS_URL: redis://redis:6379/0
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 512M

  db:
    image: postgres:16
    volumes: ["pgdata:/var/lib/postgresql/data"]
    environment:
      POSTGRES_DB: greenlang
      POSTGRES_USER: pack026
      POSTGRES_PASSWORD: secret

  redis:
    image: redis:7-alpine
    deploy:
      resources:
        limits:
          memory: 256M

volumes:
  pgdata:
```

### Resource Requirements

| Component | Minimum | Recommended | Maximum (1,000 SMEs) |
|-----------|---------|-------------|---------------------|
| CPU | 2 cores | 4 cores | 8 cores |
| RAM | 512 MB | 2 GB | 8 GB |
| Disk | 20 GB | 50 GB | 200 GB |
| PostgreSQL connections | 5 | 20 | 50 |
| Redis memory | 64 MB | 256 MB | 1 GB |

---

## Performance Optimization

### Caching Strategy

```
+------------------------------------------------------------------+
|                      CACHING LAYERS                               |
+------------------------------------------------------------------+
|                                                                    |
|  Layer 1: Application Cache (Redis)                              |
|  +--------------------------------------------------------------+ |
|  | Emission factors: Permanent (versioned with pack)             | |
|  | Industry averages: 24-hour TTL                                | |
|  | Grant programs: 1-hour TTL (sync on demand)                   | |
|  | Baseline results: Until recalculation                         | |
|  | Dashboard views: 5-minute TTL                                 | |
|  +--------------------------------------------------------------+ |
|                                                                    |
|  Layer 2: Database Materialized Views                            |
|  +--------------------------------------------------------------+ |
|  | vw_sme_dashboard: Refreshed hourly                           | |
|  | vw_sme_action_plan: Refreshed on action status change        | |
|  | vw_sme_benchmark: Refreshed daily                            | |
|  +--------------------------------------------------------------+ |
|                                                                    |
|  Layer 3: Pre-Populated Data                                     |
|  +--------------------------------------------------------------+ |
|  | Sector averages: Pre-computed for all 11 sectors             | |
|  | Quick wins library: Pre-loaded at startup                    | |
|  | Grant programs: Pre-fetched at startup                       | |
|  | Certification criteria: Static, compiled into code           | |
|  +--------------------------------------------------------------+ |
|                                                                    |
+------------------------------------------------------------------+
```

### Query Optimization

Key database queries are optimized with:

1. **Composite indexes** on (org_id, reporting_year) for progress queries
2. **Partial indexes** on (is_latest = true) for current baseline lookups
3. **Covering indexes** for dashboard view to avoid table lookups
4. **Connection pooling** via psycopg_pool (min 5, max 20 connections)
5. **Prepared statements** for all repeated queries

### Computation Optimization

1. **Bronze baseline**: No database queries -- pure arithmetic on constants
2. **Quick wins**: Pre-filtered action library loaded at startup (O(n) scan, n=500)
3. **Grant matching**: Profile-indexed lookup (O(m) where m=50 programs)
4. **Progress tracking**: Single SQL query with window functions for trends

---

## Scaling Considerations

### Single-Instance Capacity

A single PACK-026 instance (4 cores, 4 GB RAM) can serve:

| Metric | Capacity |
|--------|----------|
| SME entities | 1,000 |
| Concurrent users | 50 |
| Baselines per day | 500 |
| Reports per day | 1,000 |
| API requests per minute | 300 |

### Horizontal Scaling (if needed)

For deployments serving 1,000+ SMEs:

```
+----------+     +-------------------+     +----------+
| Load     |     | App Instance 1    |     |          |
| Balancer |---->| App Instance 2    |---->| Shared   |
| (Nginx)  |     | App Instance 3    |     | PostgreSQL|
+----------+     +-------------------+     | + Redis  |
                                           +----------+
```

Scale triggers:
- CPU > 70% sustained for 15 minutes
- Response time p95 > 3 seconds
- Connection pool exhaustion > 3 times per hour

---

## Cost Optimization

### Infrastructure Costs

| Deployment | Monthly Cost | Capacity | Cost per SME |
|-----------|-------------|----------|-------------|
| Shared VPS (1 core, 1 GB) | $10-20 | 100 SMEs | $0.10-0.20 |
| Small VPS (2 cores, 2 GB) | $20-40 | 500 SMEs | $0.04-0.08 |
| Medium VPS (4 cores, 4 GB) | $40-80 | 1,000 SMEs | $0.04-0.08 |
| Kubernetes (auto-scale) | $100-300 | 5,000+ SMEs | $0.02-0.06 |

### Free Tier Cost Model

The free tier is designed to cost near-zero to serve:
- No external API calls (emission factors are constants)
- No GPU/ML inference (zero-hallucination design)
- Minimal database storage (~1 MB per SME entity)
- No real-time sync (monthly batch only)

### Premium Feature Cost Model

Premium features add incremental cost:
- Accounting API sync: ~$0.01 per sync (API call volume)
- Grant database sync: ~$0.50/month (government API access)
- Real-time dashboard: ~$0.02 per session (WebSocket overhead)
- Multi-entity: ~$0.01/month storage per additional entity

---

*Architecture Document -- PACK-026 SME Net Zero Pack v1.0.0*
*GreenLang Platform Engineering -- 2026-03-18*
