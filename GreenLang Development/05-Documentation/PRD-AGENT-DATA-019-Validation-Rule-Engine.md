# PRD: AGENT-DATA-019 — Validation Rule Engine (GL-DATA-X-022)

**Author:** GreenLang Platform Team
**Date:** February 2026
**Status:** Approved for Development
**Priority:** P1 — Data Quality Tier
**Agent ID:** GL-DATA-X-022
**PRD Number:** AGENT-DATA-019

---

## 1. Overview

The **Validation Rule Engine** is a centralized, persistent, version-controlled rule governance layer for validating data across all 47+ GreenLang agent pipelines. It provides composable validation rules with AND/OR/NOT logic, pre-built regulatory rule packs (GHG Protocol, CSRD/ESRS, EUDR, SOC 2), rule conflict detection, batch multi-dataset validation, and comprehensive audit-grade reporting.

This agent sits at **Layer 2** in the GreenLang architecture and builds on existing Layer 1 validation capabilities from AGENT-DATA-010 (Data Quality Profiler), the Schema Compiler (AGENT-FOUND-002), and the core validation framework.

### Key Differentiators from Existing Validation

| Existing (Layer 1) | New (AGENT-DATA-019 Layer 2) |
|---|---|
| In-memory rule storage per-instance | Persistent rule registry with DB backing |
| Single-condition rules | Composable AND/OR/NOT compound rules |
| No rule versioning | SemVer-based rule set versioning |
| Hardcoded severity thresholds | Dynamic per-rule-set SLA thresholds |
| No rule conflict detection | Contradiction & overlap detection |
| Per-agent validation logic | Centralized cross-agent validation |
| No regulatory rule packs | GHG/CSRD/EUDR/SOC2 pre-built packs |
| No batch pipeline | Multi-dataset batch validation pipeline |
| No rule inheritance | Template-based rule inheritance |

---

## 2. Problem Statement

### 2.1 Current Limitations

1. **Fragmented validation logic** — 8+ agents each implement their own validation (PDF Extractor, Excel Normalizer, Supplier Questionnaire, Data Quality Profiler, Schema Compiler, Assumptions Registry, Missing Value Imputer, ERP Connector), creating inconsistent behavior
2. **No persistent rule registry** — QualityRuleEngine (AGENT-DATA-010) stores rules in-memory only; no DB persistence means rules are lost on restart
3. **Single-condition rules only** — Cannot express "IF country = 'DE' AND sector = 'energy' THEN emission_factor BETWEEN 0.1 AND 2.5"
4. **No rule versioning** — When a regulatory threshold changes (e.g., CSRD ESRS E1 emission factor bounds), there is no audit trail of rule evolution
5. **No regulatory rule packs** — Each compliance framework (GHG Protocol, CSRD, EUDR) requires manual rule creation; no pre-built packs
6. **No cross-dataset validation** — Cannot validate that "total_scope1 in dataset A == sum(scope1_components) in dataset B"
7. **No rule conflict detection** — Contradictory rules (RANGE [0,100] and RANGE [50,200] on same field) go undetected
8. **No centralized validation reporting** — No unified view of validation results across all agent pipelines
9. **No rule inheritance** — Cannot create a "base emissions rules" template extended by sector-specific overrides
10. **No batch validation pipeline** — Cannot run 500 rules across 50 datasets in a single orchestrated run
11. **No SLA-based thresholds** — Cannot define "this dataset must pass 99% of critical rules and 95% of warning rules"
12. **No rule dependency tracking** — Cannot model "Rule B only applies if Rule A passes"

---

## 3. Existing Layer 1 Capabilities (Re-exports)

| Source | Class | What It Provides |
|--------|-------|-----------------|
| `data_quality_profiler.quality_rule_engine` | `QualityRuleEngine` | 6 rule types, 8 operators, quality gates, CRUD, import/export |
| `data_quality_profiler.validity_checker` | `ValidityChecker` | 20+ format validators, range/domain/cross-field checks |
| `data_quality_profiler.models` | `QualityDimension` | 6-dimension quality model |
| `data_quality_profiler.models` | `RuleType` | Rule type enumeration |
| `schema.validator.rules` | `RuleValidator` | JSON Schema rule expressions with logic operators |

---

## 4. Identified Gaps (AGENT-DATA-019 Must Build)

| # | Gap | Resolution |
|---|-----|-----------|
| G-01 | Persistent rule registry | RuleRegistryEngine with DB-backed CRUD |
| G-02 | Compound rule composition | RuleComposerEngine with AND/OR/NOT trees |
| G-03 | Rule set versioning | SemVer bump, version history, rollback |
| G-04 | Rule conflict detection | ConflictDetectorEngine with overlap/contradiction analysis |
| G-05 | Regulatory rule packs | RulePackEngine with GHG/CSRD/EUDR/SOC2 pre-built packs |
| G-06 | Multi-dataset batch validation | ValidationPipelineEngine with orchestration |
| G-07 | Cross-dataset rules | RuleEvaluatorEngine with multi-source joins |
| G-08 | Rule inheritance/templates | RuleComposerEngine with extends/overrides |
| G-09 | Centralized validation reporting | ValidationReporterEngine with compliance formats |
| G-10 | Dynamic SLA thresholds | Per-rule-set pass/warn/fail thresholds |
| G-11 | Rule dependency graphs | RuleComposerEngine with DAG evaluation order |
| G-12 | Audit-grade rule evolution tracking | Version-controlled with provenance chains |

---

## 5. Architecture

### 5.1 Seven-Engine Design

```
┌──────────────────────────────────────────────────────────────┐
│                  ValidationPipelineEngine (E7)                │
│  End-to-end orchestration: register → compose → evaluate →   │
│  detect conflicts → report → audit                           │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│ RuleRegistry │ RuleComposer │    Rule      │    Conflict     │
│  Engine (E1) │  Engine (E2) │ Evaluator    │   Detector      │
│              │              │  Engine (E3) │   Engine (E4)   │
│ CRUD, search │ AND/OR/NOT   │ Execute vs   │ Overlap &       │
│ version,     │ templates,   │ datasets,    │ contradiction   │
│ import/export│ inheritance  │ batch eval   │ analysis        │
├──────────────┼──────────────┼──────────────┼─────────────────┤
│  RulePack    │  Validation  │             SHARED              │
│  Engine (E5) │  Reporter    │  Config, Provenance, Metrics,  │
│              │  Engine (E6) │  Models                         │
│ GHG/CSRD/    │ Text/JSON/   │                                │
│ EUDR/SOC2    │ HTML/MD/CSV  │                                │
│ packs        │ compliance   │                                │
└──────────────┴──────────────┴────────────────────────────────┘
```

### 5.2 Engine Details

#### Engine 1: RuleRegistryEngine
- **Purpose:** Centralized CRUD for validation rules with persistence simulation
- **Capabilities:**
  - Register rules with unique IDs, names, descriptions, tags
  - 10 rule types: COMPLETENESS, RANGE, FORMAT, UNIQUENESS, CUSTOM, FRESHNESS, CROSS_FIELD, CONDITIONAL, STATISTICAL, REFERENTIAL
  - 12 operators: EQUALS, NOT_EQUALS, GREATER_THAN, LESS_THAN, GREATER_EQUAL, LESS_EQUAL, BETWEEN, MATCHES, CONTAINS, IN_SET, NOT_IN_SET, IS_NULL
  - 4 severity levels: CRITICAL, HIGH, MEDIUM, LOW
  - Version tracking with SemVer (auto-bump on update: breaking→major, additive→minor, cosmetic→patch)
  - Rule status lifecycle: draft → active → deprecated → archived
  - Search by type, severity, tags, column, status, name pattern
  - Bulk import/export (JSON)
  - Rule cloning with new ID
  - Thread-safe with SHA-256 provenance

#### Engine 2: RuleComposerEngine
- **Purpose:** Build compound rules from atomic rules using logical operators
- **Capabilities:**
  - AND/OR/NOT composition of rules into rule expressions
  - Rule sets (named collections of rules with shared metadata)
  - Rule set versioning with SemVer
  - Rule templates (base rule patterns for extension)
  - Rule inheritance (child rule set extends parent, overrides specific rules)
  - Rule dependency graph (DAG) with evaluation order
  - Cycle detection in dependencies
  - Rule set comparison (diff two versions)
  - Flatten compound rules for debugging
  - Rule set statistics

#### Engine 3: RuleEvaluatorEngine
- **Purpose:** Execute rules against datasets with detailed results
- **Capabilities:**
  - Evaluate single rule against a dataset (list of dicts)
  - Evaluate rule set against a dataset
  - Batch evaluate across multiple datasets
  - Cross-dataset evaluation (join datasets by key, then validate)
  - Compound rule evaluation (AND/OR/NOT tree traversal)
  - Conditional rule evaluation (IF predicate THEN rule)
  - Statistical rule evaluation (mean, median, std dev, percentile checks)
  - Referential integrity checks (FK existence in reference dataset)
  - Per-row results with pass/fail, actual value, expected value
  - Aggregated pass rate, fail count, severity distribution
  - Short-circuit evaluation for AND (fail-fast)
  - Performance tracking (evaluation duration per rule)
  - SLA threshold evaluation (pass/warn/fail per rule set)

#### Engine 4: ConflictDetectorEngine
- **Purpose:** Detect contradictory, overlapping, or redundant rules
- **Capabilities:**
  - Range overlap detection (two RANGE rules on same column with overlapping bounds)
  - Range contradiction detection (two RANGE rules with no valid intersection)
  - Format conflict detection (conflicting regex patterns on same column)
  - Severity inconsistency detection (same condition with different severities)
  - Redundancy detection (rule A is a subset of rule B)
  - Conditional conflict detection (IF conditions that can never be true)
  - Cross-rule-set conflict analysis
  - Conflict resolution suggestions
  - Conflict severity scoring
  - Thread-safe analysis

#### Engine 5: RulePackEngine
- **Purpose:** Pre-built regulatory rule packs for common compliance frameworks
- **Capabilities:**
  - **GHG Protocol Pack** (40+ rules): Scope 1/2/3 completeness, emission factor ranges by sector, activity data format, GWP version consistency, boundary completeness, calculation methodology
  - **CSRD/ESRS Pack** (35+ rules): ESRS E1 climate metrics, ESRS S1 social metrics, double materiality indicators, XBRL tagging validation, reporting boundary consistency
  - **EUDR Pack** (25+ rules): Plot geolocation validity (WGS84), commodity classification (CN/HS codes), chain of custody integrity, Dec 31 2020 cutoff date, due diligence completeness
  - **SOC 2 Pack** (20+ rules): Data classification validation, access control completeness, audit log integrity, encryption status, retention policy compliance
  - **Custom Pack Registration**: Create custom rule packs from existing rules
  - Pack versioning (tied to regulatory update cycles)
  - Pack composition (combine multiple packs)
  - Pack comparison (diff between versions)

#### Engine 6: ValidationReporterEngine
- **Purpose:** Generate validation reports in multiple formats
- **Capabilities:**
  - 5 report formats: text, JSON, HTML, Markdown, CSV
  - 5 report types: summary, detailed, compliance, trend, executive
  - Per-rule pass/fail breakdown
  - Per-dataset aggregation
  - Severity distribution charts (data for visualization)
  - Trend analysis (compare current vs. historical results)
  - Compliance mapping (rules → regulatory articles)
  - Executive summary (pass rate, critical failures, recommendations)
  - SHA-256 report hash for integrity
  - Report storage and retrieval

#### Engine 7: ValidationPipelineEngine
- **Purpose:** End-to-end orchestration of validation workflows
- **Capabilities:**
  - 7-stage pipeline: register → compose → evaluate → detect_conflicts → report → audit → notify
  - Batch validation across multiple datasets
  - Pipeline scheduling metadata
  - Run history with status tracking
  - Stage-level timing and error isolation
  - Configurable stage execution (skip stages)
  - Pipeline templates for common workflows
  - Health check and statistics
  - Change detection (new rule failures since last run)

### 5.3 Database Schema (V049)

**10 Tables:**
1. `validation_rules` — Rule definitions with versioning
2. `validation_rule_sets` — Named collections of rules
3. `validation_rule_set_members` — Many-to-many: rules ↔ rule sets
4. `validation_rule_versions` — Version history per rule
5. `validation_compound_rules` — AND/OR/NOT composition definitions
6. `validation_rule_packs` — Pre-built regulatory rule packs
7. `validation_evaluations` — Evaluation run results
8. `validation_evaluation_details` — Per-row evaluation details
9. `validation_reports` — Generated reports
10. `validation_audit_log` — Audit trail with provenance

**3 Hypertables (TimescaleDB):**
1. `validation_evaluation_events` — Time-series evaluation metrics
2. `validation_rule_change_events` — Time-series rule mutations
3. `validation_conflict_events` — Time-series conflict detections

**2 Continuous Aggregates:**
1. `validation_evaluations_hourly_stats` — Hourly aggregated evaluation metrics
2. `validation_rule_changes_hourly_stats` — Hourly aggregated rule mutation counts

### 5.4 Prometheus Metrics (12 metrics, `gl_vre_` prefix)

| # | Metric Name | Type | Labels |
|---|-------------|------|--------|
| 1 | `gl_vre_rules_registered_total` | Counter | `rule_type`, `severity` |
| 2 | `gl_vre_rule_sets_created_total` | Counter | `pack_type` |
| 3 | `gl_vre_evaluations_total` | Counter | `result`, `rule_type` |
| 4 | `gl_vre_evaluation_failures_total` | Counter | `severity` |
| 5 | `gl_vre_conflicts_detected_total` | Counter | `conflict_type` |
| 6 | `gl_vre_reports_generated_total` | Counter | `report_type`, `format` |
| 7 | `gl_vre_rules_per_set` | Histogram | buckets: (1,5,10,25,50,100,250,500) |
| 8 | `gl_vre_evaluation_duration_seconds` | Histogram | `operation`, buckets: (0.01,0.05,0.1,0.5,1,5,10,30) |
| 9 | `gl_vre_processing_duration_seconds` | Histogram | `operation`, buckets: (0.01,0.05,0.1,0.5,1,5,10) |
| 10 | `gl_vre_active_rules` | Gauge | — |
| 11 | `gl_vre_active_rule_sets` | Gauge | — |
| 12 | `gl_vre_pass_rate` | Gauge | — |

### 5.5 REST API Endpoints (20 endpoints at `/api/v1/validation-rules`)

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | `/rules` | Register a new validation rule |
| 2 | GET | `/rules` | List/search rules |
| 3 | GET | `/rules/{rule_id}` | Get rule details |
| 4 | PUT | `/rules/{rule_id}` | Update a rule |
| 5 | DELETE | `/rules/{rule_id}` | Archive a rule |
| 6 | POST | `/rule-sets` | Create a rule set |
| 7 | GET | `/rule-sets` | List rule sets |
| 8 | GET | `/rule-sets/{set_id}` | Get rule set details |
| 9 | PUT | `/rule-sets/{set_id}` | Update a rule set |
| 10 | DELETE | `/rule-sets/{set_id}` | Archive a rule set |
| 11 | POST | `/evaluate` | Evaluate rules against data |
| 12 | POST | `/evaluate/batch` | Batch evaluate across datasets |
| 13 | GET | `/evaluations/{eval_id}` | Get evaluation results |
| 14 | POST | `/conflicts/detect` | Detect rule conflicts |
| 15 | GET | `/conflicts` | List detected conflicts |
| 16 | POST | `/packs/{pack_name}/apply` | Apply a regulatory rule pack |
| 17 | GET | `/packs` | List available rule packs |
| 18 | POST | `/reports` | Generate validation report |
| 19 | POST | `/pipeline` | Run validation pipeline |
| 20 | GET | `/health` | Health check |

### 5.6 Configuration (22 settings, `GL_VRE_` prefix)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `GL_VRE_DATABASE_URL` | str | "" | PostgreSQL connection string |
| `GL_VRE_REDIS_URL` | str | "" | Redis connection string |
| `GL_VRE_LOG_LEVEL` | str | "INFO" | Logging level |
| `GL_VRE_MAX_RULES` | int | 100,000 | Max rules in registry |
| `GL_VRE_MAX_RULE_SETS` | int | 10,000 | Max rule sets |
| `GL_VRE_MAX_RULES_PER_SET` | int | 500 | Max rules per set |
| `GL_VRE_MAX_COMPOUND_DEPTH` | int | 10 | Max nesting depth for AND/OR/NOT |
| `GL_VRE_DEFAULT_PASS_THRESHOLD` | float | 0.95 | Default pass threshold |
| `GL_VRE_DEFAULT_WARN_THRESHOLD` | float | 0.80 | Default warning threshold |
| `GL_VRE_EVALUATION_TIMEOUT` | int | 300 | Max evaluation time (seconds) |
| `GL_VRE_BATCH_SIZE` | int | 1000 | Default batch size |
| `GL_VRE_MAX_BATCH_DATASETS` | int | 100 | Max datasets per batch |
| `GL_VRE_ENABLE_PROVENANCE` | bool | True | Enable SHA-256 provenance |
| `GL_VRE_GENESIS_HASH` | str | "greenlang-validation-rule-genesis" | Genesis hash |
| `GL_VRE_ENABLE_METRICS` | bool | True | Enable Prometheus metrics |
| `GL_VRE_POOL_SIZE` | int | 5 | Connection pool size |
| `GL_VRE_CACHE_TTL` | int | 300 | Cache TTL seconds |
| `GL_VRE_RATE_LIMIT` | int | 200 | Rate limit per minute |
| `GL_VRE_ENABLE_CONFLICT_DETECTION` | bool | True | Enable auto conflict detection |
| `GL_VRE_ENABLE_SHORT_CIRCUIT` | bool | True | Enable AND short-circuit eval |
| `GL_VRE_MAX_EVALUATION_ROWS` | int | 1,000,000 | Max rows per evaluation |
| `GL_VRE_REPORT_RETENTION_DAYS` | int | 90 | Report retention |

### 5.7 Layer 1 Re-exports

```python
# From AGENT-DATA-010: Data Quality Profiler
from greenlang.data_quality_profiler.quality_rule_engine import QualityRuleEngine
from greenlang.data_quality_profiler.validity_checker import ValidityChecker
from greenlang.data_quality_profiler.models import QualityDimension, RuleType

# From AGENT-FOUND-002: Schema Compiler
from greenlang.schema.validator.rules import RuleValidator
```

### 5.8 Auth Integration

Add to `route_protector.py` PERMISSION_MAP:
```python
# Validation Rule Engine routes (/api/v1/validation-rules) - AGENT-DATA-019
"POST:/api/v1/validation-rules/rules": "validation-rules:rules:create",
"GET:/api/v1/validation-rules/rules": "validation-rules:rules:read",
# ... (20 entries total)
```

Add to `auth_setup.py`:
```python
from greenlang.validation_rule_engine.setup import get_router as get_vre_router
```

---

## 6. Acceptance Criteria

1. **AC-01:** 7 engines fully implemented with all capabilities listed above
2. **AC-02:** 10 rule types supported (COMPLETENESS through REFERENTIAL)
3. **AC-03:** AND/OR/NOT compound rule composition with max 10 nesting depth
4. **AC-04:** 4 regulatory rule packs (GHG Protocol 40+ rules, CSRD/ESRS 35+ rules, EUDR 25+ rules, SOC 2 20+ rules)
5. **AC-05:** Rule conflict detection with overlap, contradiction, and redundancy analysis
6. **AC-06:** Multi-dataset batch validation with cross-dataset referential checks
7. **AC-07:** 5 report formats (text, JSON, HTML, Markdown, CSV)
8. **AC-08:** SemVer rule versioning with version history and rollback
9. **AC-09:** Rule inheritance with template extends/overrides
10. **AC-10:** 20 REST API endpoints with auth integration
11. **AC-11:** 12 Prometheus metrics with `gl_vre_` prefix
12. **AC-12:** V049 DB migration (10 tables + 3 hypertables + 2 continuous aggregates)
13. **AC-13:** SHA-256 provenance chains on all mutations
14. **AC-14:** 1000+ unit tests, 100+ integration tests
15. **AC-15:** K8s manifests, Dockerfile, CI/CD workflow, Grafana dashboard, alert rules

---

## 7. Dependencies

| Dependency | Agent | Why |
|-----------|-------|-----|
| Data Quality Profiler | AGENT-DATA-010 | Layer 1 QualityRuleEngine + ValidityChecker |
| Schema Compiler | AGENT-FOUND-002 | Layer 1 RuleValidator for expression evaluation |
| Data Lineage Tracker | AGENT-DATA-018 | Track validation in lineage graph |
| Observability Agent | AGENT-FOUND-010 | Metrics + tracing |
| Auth Service | SEC-001 | JWT + RBAC |
| PostgreSQL + TimescaleDB | INFRA-002 | Persistent storage |
| Redis | INFRA-003 | Caching |

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Rule evaluation performance with 100K+ rules | Medium | High | Index-accelerated rule lookup, short-circuit AND, batch chunking |
| Rule conflict detection exponential complexity | Medium | Medium | Limit conflict analysis to same-column rules, O(n²) bounded |
| Regulatory rule pack maintenance | High | Medium | Version-controlled packs, regulatory intelligence feed integration |
| Cross-dataset join memory pressure | Low | High | Streaming evaluation, configurable row limits |
