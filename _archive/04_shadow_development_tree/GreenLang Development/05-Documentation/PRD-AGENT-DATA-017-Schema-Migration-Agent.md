# PRD: AGENT-DATA-017 — GL-DATA-X-020 Schema Migration Agent

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-DATA-X-020 |
| Internal Label | AGENT-DATA-017 |
| Category | Layer 2 – Data Quality Agents |
| Purpose | Manage schema evolution across all GreenLang data pipelines — track schema versions, detect breaking vs non-breaking changes, check forward/backward compatibility, generate migration plans, execute schema transformations, support rollback, and maintain full provenance of all schema changes |
| Estimated Variants | 200 |
| Status | To Be Built |
| Author | GreenLang Platform Team |
| Date | February 2026 |
| DB Migration | V047 |

## 2. Problem Statement
GreenLang ingests data from dozens of heterogeneous sources — ERP systems (SAP, Oracle, NetSuite), IoT sensors, supplier questionnaires, government registries, satellite imagery feeds, and manual uploads. As these sources evolve their data formats, schemas change frequently and unpredictably. Organizations face critical challenges:

1. **No centralized schema registry** — cannot track which schema version each dataset or source uses, leading to silent data corruption when upstream formats change without notice
2. **No schema versioning** — no semantic versioning (major.minor.patch) with change history, making it impossible to reason about the evolution of data contracts over time
3. **No change detection** — cannot automatically detect what changed between schema versions (added/removed/renamed/retyped fields, constraint modifications, nested structure changes)
4. **No compatibility checking** — no forward/backward/full compatibility analysis per Avro/JSON Schema conventions, leading to runtime failures when producers and consumers operate on different schema versions
5. **No migration plan generation** — manual migration planning is prone to errors and omissions, especially when changes cascade across multiple dependent schemas
6. **No automated migration execution** — schema migrations require manual intervention across all affected pipelines, costing engineering hours and introducing human error
7. **No rollback support** — no ability to revert failed migrations safely, leaving data in inconsistent intermediate states after partial failures
8. **No impact analysis** — cannot assess which downstream agents, pipelines, calculations, and reports are affected by a schema change before it is applied
9. **No schema drift detection** — cannot detect when actual data arriving in pipelines deviates from the registered schema, resulting in silent data quality degradation
10. **No compliance documentation** — CSRD/ESRS/GHG Protocol require documented data governance processes including schema management, version control, and change audit trails
11. **No cross-source schema harmonization** — cannot map equivalent fields across different source schemas (e.g., SAP "MENGE" = Oracle "QUANTITY" = CSV "qty"), preventing unified analysis
12. **No provenance trail** — no audit chain for schema changes, migration decisions, rollback actions, and compatibility assessments, undermining regulatory auditability

## 3. Existing Layer 1 Capabilities
- `greenlang.agents.foundation.schema_compiler.SchemaCompilerEngine` — JSON Schema Draft 2020-12 validation, AST-to-IR compilation, 7-phase validation pipeline, ReDoS prevention, RFC 6902 fix suggestions
- `greenlang.agents.foundation.schema_compiler.SchemaRegistry` — Git-backed schema registry with versioning, Layer 1/Layer 2 delegation
- `greenlang.data_quality_profiler.validity_checker.ValidityChecker` — Schema validation against registered schemas, field-level validity scoring
- `greenlang.data_quality_profiler.consistency_analyzer.ConsistencyAnalyzer` — Cross-dataset consistency checking, referential integrity validation
- `greenlang.excel_normalizer.column_mapper.ColumnMapper` — 200+ canonical fields, 500+ synonym mappings, fuzzy column matching
- `greenlang.excel_normalizer.data_type_detector.DataTypeDetector` — 14 data type auto-detection, format inference

## 4. Identified Gaps (12)
| # | Gap | Layer 1 Provides | Layer 2 Needed |
|---|-----|------------------|----------------|
| 1 | Schema registry with full lifecycle | Git-backed registry with basic versioning | Register schemas with namespaces, ownership, tags, status lifecycle (draft/active/deprecated/archived), JSON Schema/Avro/Protobuf support, search, import/export, bulk operations |
| 2 | Semantic versioning with changelog | Basic version tracking | Automatic major.minor.patch bump classification (breaking=major, additive=minor, cosmetic=patch), structured changelogs, version comparison, deprecation management with sunset dates |
| 3 | Automatic change detection | None beyond schema validation | Structural diff engine: added/removed/renamed/retyped/reordered fields, constraint changes (required-to-optional, type widening/narrowing), nested object changes, array item type changes, enum value changes, default value changes |
| 4 | Compatibility matrix | None | Backward compatible (new schema can read old data), forward compatible (old schema can read new data), full compatible (both directions), breaking (neither direction), per Confluent Schema Registry conventions |
| 5 | Migration plan generation | None | Ordered transformation steps (field renames, type casts, default value injection, computed fields, field splits/merges), dependency-aware ordering across multiple schemas, dry-run estimation, effort scoring |
| 6 | Safe migration execution | None | Step-by-step execution with checkpoints after each step, intermediate validation, automatic rollback on failure, parallel execution for independent steps, progress tracking, retry logic with exponential backoff |
| 7 | Rollback with undo scripts | None | Generate and execute undo transformations, partial rollback to last checkpoint, full rollback to pre-migration state, rollback verification, cascading rollback across dependent schemas |
| 8 | Downstream impact analysis | None | Identify all agents, pipelines, calculations, and reports affected by a schema change; severity classification per consumer; blocking vs non-blocking impact; migration dependency graph |
| 9 | Schema drift monitoring | Basic validity checking | Continuous monitoring of incoming data against registered schema, drift type classification (missing fields, extra fields, type mismatches, constraint violations), drift severity scoring, trend analysis |
| 10 | Field mapping and harmonization | Column mapper with synonyms | Cross-source field equivalence mapping (SAP/Oracle/CSV/API), transformation rules (rename, cast, compute), bidirectional mapping, mapping confidence scoring, conflict resolution |
| 11 | Compliance documentation generation | None | Generate schema governance reports for CSRD/ESRS, GHG Protocol, SOC 2 audits; version history attestation; migration audit reports; drift incident reports |
| 12 | End-to-end provenance | Per-check SHA-256 in Layer 1 | Complete provenance chain: schema registration, version creation, change detection, compatibility assessment, migration planning, execution steps, rollback actions, drift events, all linked by SHA-256 hash chain |

## 5. Architecture

### 5.1 Seven Engines
| Engine | Class | Responsibility |
|--------|-------|----------------|
| 1 | SchemaRegistryEngine (`schema_registry.py`) | Register, catalog, and manage schemas with namespaces, tags, ownership, and status lifecycle (draft/active/deprecated/archived). Support JSON Schema Draft 2020-12, Avro, and Protobuf-like definitions. Provide schema search by name/namespace/tag/owner, import/export in standard formats, bulk registration, and schema grouping by domain |
| 2 | SchemaVersionerEngine (`schema_versioner.py`) | Semantic versioning (major.minor.patch) with automatic version bump classification: breaking changes trigger major bump, additive changes trigger minor bump, cosmetic/documentation changes trigger patch bump. Maintain version history with structured changelogs, version comparison (diff between any two versions), deprecation management with configurable sunset dates, and version pinning for downstream consumers |
| 3 | ChangeDetectorEngine (`change_detector.py`) | Detect structural differences between schema versions: added/removed/renamed/retyped/reordered fields, constraint changes (required-to-optional and vice versa, type widening such as int-to-float, type narrowing such as float-to-int), nested object changes at arbitrary depth, array item type changes, enum value additions/removals, default value changes, description/metadata changes. Classify each change by severity (breaking/non-breaking/cosmetic) |
| 4 | CompatibilityCheckerEngine (`compatibility_checker.py`) | Check compatibility between schema versions following Confluent Schema Registry conventions: backward compatible (consumers using new schema can read data produced with old schema), forward compatible (consumers using old schema can read data produced with new schema), full compatible (both backward and forward), breaking (neither direction). Generate detailed compatibility reports with per-field assessments and remediation suggestions |
| 5 | MigrationPlannerEngine (`migration_planner.py`) | Generate migration plans with ordered transformation steps: field renames, type casts with precision handling, default value injection for new required fields, computed field generation, field splits (one-to-many) and merges (many-to-one), null handling strategies. Dependency-aware ordering across multiple schemas when a change cascades. Dry-run estimation with record count and effort scoring. Plan validation before execution |
| 6 | MigrationExecutorEngine (`migration_executor.py`) | Execute migration plans step-by-step: apply transformations with checkpoints after each step, validate intermediate results against expected schema, automatic rollback on failure (configurable), parallel execution for independent transformation steps, progress tracking with percentage and ETA, retry logic with exponential backoff for transient failures, execution logging for audit trail |
| 7 | SchemaMigrationPipelineEngine (`schema_migration_pipeline.py`) | End-to-end orchestration: detect changes between schema versions, check compatibility, generate migration plan, validate plan (dry-run), execute migration, verify migrated data against target schema, update registry with new version. Support batch processing of multiple schema migrations, scheduled drift monitoring, pipeline pause/resume, and configurable pipeline stages (skip compatibility check, skip dry-run, etc.) |

### 5.2 Data Flow
```
Schema Registration → SchemaRegistryEngine (catalog + metadata + namespace)
                    → SchemaVersionerEngine (version bump + changelog)
                    → ChangeDetectorEngine (structural diff analysis)
                    → CompatibilityCheckerEngine (compatibility matrix)
                    → MigrationPlannerEngine (ordered migration plan)
                    → MigrationExecutorEngine (execute + checkpoint + rollback)
                    → SchemaMigrationPipelineEngine (orchestration + verification)
```

### 5.3 Database Schema (V047)
- `schema_registry` — registered schemas (id UUID PK, namespace VARCHAR, name VARCHAR, schema_type ENUM[json_schema/avro/protobuf], owner VARCHAR, tags JSONB, status ENUM[draft/active/deprecated/archived], description TEXT, metadata JSONB, created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ)
- `schema_versions` — schema version history (id UUID PK, schema_id UUID FK, version VARCHAR [semver], definition_json JSONB, changelog TEXT, is_deprecated BOOLEAN, deprecated_at TIMESTAMPTZ, sunset_date DATE, created_by VARCHAR, created_at TIMESTAMPTZ)
- `schema_changes` — detected changes between versions (id UUID PK, source_version_id UUID FK, target_version_id UUID FK, change_type ENUM[added/removed/renamed/retyped/reordered/constraint_changed/enum_changed/default_changed], field_path VARCHAR, old_value JSONB, new_value JSONB, severity ENUM[breaking/non_breaking/cosmetic], description TEXT, detected_at TIMESTAMPTZ)
- `schema_compatibility_checks` — compatibility results (id UUID PK, source_version_id UUID FK, target_version_id UUID FK, compatibility_level ENUM[backward/forward/full/breaking], issues_json JSONB, recommendations_json JSONB, checked_by VARCHAR, checked_at TIMESTAMPTZ)
- `schema_migration_plans` — migration plans (id UUID PK, source_schema_id UUID FK, target_schema_id UUID FK, source_version VARCHAR, target_version VARCHAR, steps_json JSONB, status ENUM[draft/validated/approved/executing/completed/failed/rolled_back], estimated_effort_minutes FLOAT, estimated_records INTEGER, dry_run_result JSONB, created_by VARCHAR, created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ)
- `schema_migration_executions` — execution records (id UUID PK, plan_id UUID FK, started_at TIMESTAMPTZ, completed_at TIMESTAMPTZ, status ENUM[running/completed/failed/rolled_back], current_step INTEGER, total_steps INTEGER, records_processed INTEGER, records_failed INTEGER, records_skipped INTEGER, checkpoint_data JSONB, error_details TEXT, execution_log JSONB)
- `schema_rollbacks` — rollback records (id UUID PK, execution_id UUID FK, reason TEXT, rollback_type ENUM[full/partial/checkpoint], rolled_back_to_step INTEGER, records_reverted INTEGER, started_at TIMESTAMPTZ, completed_at TIMESTAMPTZ, status ENUM[running/completed/failed], error_details TEXT)
- `schema_field_mappings` — cross-source field mappings (id UUID PK, source_schema_id UUID FK, target_schema_id UUID FK, source_field VARCHAR, target_field VARCHAR, transform_rule JSONB, confidence FLOAT, mapping_type ENUM[exact/alias/computed/manual], created_by VARCHAR, created_at TIMESTAMPTZ)
- `schema_drift_events` — schema drift detections (id UUID PK, schema_id UUID FK, version_id UUID FK, dataset_id VARCHAR, drift_type ENUM[missing_field/extra_field/type_mismatch/constraint_violation/enum_violation], field_path VARCHAR, expected_value JSONB, actual_value JSONB, severity ENUM[low/medium/high/critical], sample_count INTEGER, detected_at TIMESTAMPTZ)
- `schema_audit_log` — all actions with provenance (id UUID PK, action VARCHAR, entity_type VARCHAR, entity_id UUID, actor VARCHAR, details_json JSONB, previous_state JSONB, new_state JSONB, provenance_hash VARCHAR, parent_hash VARCHAR, created_at TIMESTAMPTZ)
- 3 hypertables (7-day chunks):
  - `schema_change_events` — time-series of schema changes (schema_id, version_from, version_to, change_count, breaking_count, ts TIMESTAMPTZ)
  - `migration_execution_events` — time-series of migration executions (execution_id, plan_id, step, records_processed, records_failed, duration_ms, ts TIMESTAMPTZ)
  - `drift_detection_events` — time-series of drift detections (schema_id, dataset_id, drift_count, severity_max, ts TIMESTAMPTZ)
- 2 continuous aggregates:
  - `schema_changes_hourly_stats` — hourly rollup of schema changes by type, severity, and namespace
  - `migration_executions_hourly_stats` — hourly rollup of migration executions by status, records processed, and duration

### 5.4 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| gl_sm_schemas_registered_total | Counter | Total schemas registered, labeled by schema_type and namespace |
| gl_sm_versions_created_total | Counter | Total schema versions created, labeled by bump_type (major/minor/patch) |
| gl_sm_changes_detected_total | Counter | Schema changes detected, labeled by change_type and severity |
| gl_sm_compatibility_checks_total | Counter | Compatibility checks performed, labeled by result (backward/forward/full/breaking) |
| gl_sm_migrations_planned_total | Counter | Migration plans generated, labeled by status |
| gl_sm_migrations_executed_total | Counter | Migrations executed, labeled by status (completed/failed/rolled_back) |
| gl_sm_rollbacks_total | Counter | Rollbacks performed, labeled by rollback_type and status |
| gl_sm_drift_detected_total | Counter | Schema drift events detected, labeled by drift_type and severity |
| gl_sm_migration_duration_seconds | Histogram | Migration execution duration in seconds, buckets [1, 5, 10, 30, 60, 300, 600, 1800, 3600] |
| gl_sm_records_migrated | Histogram | Records migrated per execution, buckets [10, 100, 1000, 5000, 10000, 50000, 100000] |
| gl_sm_processing_duration_seconds | Histogram | Processing duration by operation (register/version/detect/check/plan/execute/rollback) |
| gl_sm_active_migrations | Gauge | Number of currently running migration executions |

### 5.5 API Endpoints (20)
| # | Method | Path | Purpose |
|---|--------|------|---------|
| 1 | POST | /api/v1/schema-migration/schemas | Register a new schema with namespace, type, owner, and tags |
| 2 | GET | /api/v1/schema-migration/schemas | List registered schemas with filtering by namespace, type, owner, status, and tags |
| 3 | GET | /api/v1/schema-migration/schemas/{id} | Get full schema details including latest version and metadata |
| 4 | PUT | /api/v1/schema-migration/schemas/{id} | Update schema metadata (owner, tags, status, description) |
| 5 | DELETE | /api/v1/schema-migration/schemas/{id} | Deregister schema (soft delete, sets status to archived) |
| 6 | POST | /api/v1/schema-migration/versions | Create a new schema version with definition and auto-classified version bump |
| 7 | GET | /api/v1/schema-migration/versions | List schema versions with filtering by schema_id, version range, and deprecation status |
| 8 | GET | /api/v1/schema-migration/versions/{id} | Get version details including definition, changelog, and deprecation info |
| 9 | POST | /api/v1/schema-migration/changes/detect | Detect changes between two schema versions and return classified diff |
| 10 | GET | /api/v1/schema-migration/changes | List detected changes with filtering by schema, severity, and change_type |
| 11 | POST | /api/v1/schema-migration/compatibility/check | Check compatibility between two schema versions and return detailed assessment |
| 12 | GET | /api/v1/schema-migration/compatibility | List historical compatibility check results |
| 13 | POST | /api/v1/schema-migration/plans | Generate a migration plan for transforming data between schema versions |
| 14 | GET | /api/v1/schema-migration/plans/{id} | Get migration plan details including steps, effort estimate, and dry-run results |
| 15 | POST | /api/v1/schema-migration/execute | Execute a validated migration plan with optional dry-run flag |
| 16 | GET | /api/v1/schema-migration/executions/{id} | Get execution status including progress, current step, and error details |
| 17 | POST | /api/v1/schema-migration/rollback/{id} | Rollback a migration execution (full or to specific checkpoint) |
| 18 | POST | /api/v1/schema-migration/pipeline | Run the full migration pipeline: detect, check, plan, execute, verify |
| 19 | GET | /api/v1/schema-migration/health | Health check returning engine statuses, database connectivity, and active migration count |
| 20 | GET | /api/v1/schema-migration/stats | Service statistics: schemas registered, versions created, migrations executed, drift events, success rates |

### 5.6 Configuration (GL_SM_ prefix)
| Setting | Default | Description |
|---------|---------|-------------|
| GL_SM_DATABASE_URL | postgresql://localhost:5432/greenlang | PostgreSQL connection string |
| GL_SM_REDIS_URL | redis://localhost:6379/0 | Redis cache connection string |
| GL_SM_LOG_LEVEL | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |
| GL_SM_MAX_SCHEMAS | 50000 | Maximum number of registered schemas |
| GL_SM_MAX_VERSIONS_PER_SCHEMA | 1000 | Maximum versions retained per schema |
| GL_SM_MAX_MIGRATION_BATCH_SIZE | 10000 | Maximum records per migration batch |
| GL_SM_MIGRATION_TIMEOUT_SECONDS | 3600 | Maximum migration execution time before timeout |
| GL_SM_ENABLE_DRY_RUN | true | Enable dry-run validation before migration execution |
| GL_SM_ENABLE_AUTO_ROLLBACK | true | Automatically rollback on migration step failure |
| GL_SM_COMPATIBILITY_DEFAULT_LEVEL | backward | Default compatibility level for checks (backward/forward/full) |
| GL_SM_DRIFT_CHECK_INTERVAL_MINUTES | 60 | Interval for scheduled schema drift monitoring |
| GL_SM_DRIFT_SAMPLE_SIZE | 1000 | Number of records sampled per drift check |
| GL_SM_ENABLE_PROVENANCE | true | Enable SHA-256 provenance chain tracking |
| GL_SM_GENESIS_HASH | greenlang-schema-migration-genesis | Provenance chain genesis string |
| GL_SM_MAX_WORKERS | 4 | Concurrent worker threads for parallel operations |
| GL_SM_POOL_SIZE | 5 | Database connection pool size |
| GL_SM_CACHE_TTL | 300 | Cache TTL in seconds for schema lookups |
| GL_SM_RATE_LIMIT | 100 | Maximum API requests per minute |
| GL_SM_CHECKPOINT_INTERVAL | 100 | Records between checkpoints during migration execution |
| GL_SM_RETRY_MAX_ATTEMPTS | 3 | Maximum retry attempts for failed migration steps |
| GL_SM_RETRY_BACKOFF_BASE | 2.0 | Exponential backoff base for retries (seconds) |
| GL_SM_FIELD_MAPPING_MIN_CONFIDENCE | 0.8 | Minimum confidence score for automatic field mappings |
| GL_SM_DEPRECATION_WARNING_DAYS | 30 | Days before sunset date to begin deprecation warnings |
| GL_SM_MAX_CHANGE_DEPTH | 10 | Maximum nesting depth for change detection in nested schemas |
| GL_SM_ENABLE_IMPACT_ANALYSIS | true | Enable downstream impact analysis for schema changes |

### 5.7 Layer 1 Re-exports
- `greenlang.agents.foundation.schema_compiler.SchemaCompilerEngine`
- `greenlang.agents.foundation.schema_compiler.SchemaRegistry`
- `greenlang.data_quality_profiler.validity_checker.ValidityChecker`
- `greenlang.data_quality_profiler.consistency_analyzer.ConsistencyAnalyzer`
- `greenlang.data_quality_profiler.models.QualityDimension`
- `greenlang.data_quality_profiler.models.RuleType`
- `greenlang.excel_normalizer.column_mapper.ColumnMapper`
- `greenlang.excel_normalizer.data_type_detector.DataTypeDetector`

### 5.8 Provenance Design
- Genesis string: `"greenlang-schema-migration-genesis"`
- SHA-256 chain: each operation (register, version, detect, check, plan, execute, rollback, drift) appends to chain
- Per-schema version history: tracks all versions with definitions, changelogs, and provenance hashes
- Migration lifecycle: complete audit trail from plan creation through execution steps through completion or rollback
- Change detection provenance: every detected change linked to source version, target version, and detection timestamp
- Compatibility assessment provenance: every compatibility check result linked to the versions compared and the rules applied
- Rollback provenance: rollback actions linked to original execution, reason, reverted steps, and verification result
- Deterministic: same input schemas + same config = same provenance hash
- Cross-reference: provenance hashes link to AGENT-FOUND-002 (Schema Compiler) provenance chains for full traceability

## 6. Acceptance Criteria

### 6.1 SchemaRegistryEngine
- Register schemas with namespace, name, schema_type (JSON Schema/Avro/Protobuf), owner, tags, and description
- Enforce unique constraint on (namespace, name) combination
- Support schema status lifecycle: draft -> active -> deprecated -> archived
- Search schemas by namespace, name (substring), owner, tag, status, and schema_type
- Import/export schemas in JSON format with full metadata
- Bulk registration of up to 1000 schemas in a single operation
- Schema grouping by domain (e.g., "emissions", "supply-chain", "compliance")
- Validate schema definitions against their declared type (JSON Schema Draft 2020-12 validation for json_schema type)
- Return 400 with descriptive errors for invalid schema definitions

### 6.2 SchemaVersionerEngine
- Auto-classify version bumps: breaking changes increment major, additive changes increment minor, cosmetic changes increment patch
- Generate structured changelogs listing all changes per version with severity and description
- Compare any two versions of the same schema and return ordered list of changes
- Support deprecation with sunset_date; emit warnings when sunset_date is within GL_SM_DEPRECATION_WARNING_DAYS
- Prevent creation of versions for archived schemas (return 409 Conflict)
- Version pinning: allow downstream consumers to pin to a specific version range (e.g., ">=2.0.0 <3.0.0")
- Maintain version count per schema and enforce GL_SM_MAX_VERSIONS_PER_SCHEMA limit

### 6.3 ChangeDetectorEngine
- Detect added fields (present in target, absent in source)
- Detect removed fields (present in source, absent in target)
- Detect renamed fields using heuristics (same type + position, configurable confidence threshold)
- Detect retyped fields (field exists in both but type changed, e.g., string -> integer)
- Detect reordered fields (field exists in both but position changed)
- Detect constraint changes: required-to-optional, optional-to-required, type widening (int->float), type narrowing (float->int)
- Detect nested object changes at depth up to GL_SM_MAX_CHANGE_DEPTH
- Detect array item type changes (e.g., array of strings -> array of objects)
- Detect enum value additions and removals
- Detect default value changes
- Classify each change as breaking, non_breaking, or cosmetic
- Return changes as ordered list sorted by severity (breaking first) then field path

### 6.4 CompatibilityCheckerEngine
- Determine backward compatibility: new schema can deserialize data written with old schema
- Determine forward compatibility: old schema can deserialize data written with new schema
- Determine full compatibility: both backward and forward compatible
- Determine breaking: neither backward nor forward compatible
- Apply rules per Confluent Schema Registry conventions:
  - Adding optional field = backward compatible
  - Removing optional field = forward compatible
  - Adding required field without default = breaking
  - Removing required field = breaking
  - Widening type (int->float) = backward compatible
  - Narrowing type (float->int) = breaking
  - Adding enum value = backward compatible (new reader knows value)
  - Removing enum value = breaking (old data may contain removed value)
- Generate detailed compatibility report with per-field assessment
- Provide remediation suggestions for breaking changes (e.g., "add default value for new required field")

### 6.5 MigrationPlannerEngine
- Generate ordered list of transformation steps from source to target schema
- Support transformation types: rename_field, cast_type, set_default, compute_field, split_field, merge_fields, remove_field, add_field
- Each step includes: step_number, operation, source_field, target_field, parameters, reversible (boolean)
- Dependency-aware ordering: steps that depend on other steps are ordered after their dependencies
- Dry-run estimation: estimate records affected, execution time, and potential data loss
- Effort scoring: LOW (< 1 min), MEDIUM (1-10 min), HIGH (10-60 min), CRITICAL (> 60 min)
- Plan validation: verify all steps are internally consistent and cover all detected changes
- Multi-schema plans: when a schema change cascades to dependent schemas, generate a coordinated plan

### 6.6 MigrationExecutorEngine
- Execute migration plan step by step in dependency order
- Create checkpoint after each step with current state and rollback data
- Validate intermediate results against expected schema after each step
- Automatic rollback to last successful checkpoint on step failure (when GL_SM_ENABLE_AUTO_ROLLBACK is true)
- Parallel execution for independent steps (up to GL_SM_MAX_WORKERS concurrent)
- Progress tracking: records_processed, records_failed, records_skipped, current_step, total_steps, percentage, ETA
- Retry logic with exponential backoff (base GL_SM_RETRY_BACKOFF_BASE, max GL_SM_RETRY_MAX_ATTEMPTS attempts)
- Respect GL_SM_MIGRATION_TIMEOUT_SECONDS; abort and rollback if exceeded
- Respect GL_SM_MAX_MIGRATION_BATCH_SIZE; process records in batches
- Execution log: every step recorded with start_time, end_time, records_affected, status, error_details

### 6.7 SchemaMigrationPipelineEngine
- Full pipeline: detect changes -> check compatibility -> generate plan -> validate plan (dry-run) -> execute migration -> verify results -> update registry
- Configurable pipeline stages: allow skipping compatibility check or dry-run via flags
- Batch processing: run pipeline for multiple schema pairs in a single invocation
- Scheduled drift monitoring: check registered schemas against incoming data at GL_SM_DRIFT_CHECK_INTERVAL_MINUTES
- Pipeline pause/resume: save pipeline state and resume from last completed stage
- Pipeline status reporting: current stage, elapsed time, schemas processed, errors encountered
- Generate compliance reports: schema governance summary, version history, migration audit trail
- Emit Prometheus metrics for each pipeline stage

### 6.8 API and Integration
- All 20 REST API endpoints operational and returning correct HTTP status codes
- Request validation with Pydantic v2 models for all request/response bodies
- Pagination support (limit/offset) for all list endpoints
- Filtering support on all list endpoints as specified per endpoint
- Auth integration: all endpoints protected via PERMISSION_MAP entries
- Rate limiting at GL_SM_RATE_LIMIT requests per minute
- OpenAPI/Swagger documentation auto-generated from Pydantic models

### 6.9 Metrics and Observability
- All 12 Prometheus metrics collecting and exposed at /metrics
- Metric labels correctly applied (schema_type, namespace, change_type, severity, status, drift_type, operation)
- Histogram buckets configured as specified in Section 5.4
- Gauge gl_sm_active_migrations accurately reflects currently running migrations (incremented on start, decremented on completion/failure/rollback)

### 6.10 Provenance and Compliance
- SHA-256 provenance chains deterministic: same inputs + same config = same hash
- Every schema registration, version creation, change detection, compatibility check, migration plan, execution step, rollback action, and drift event recorded in schema_audit_log with provenance_hash and parent_hash
- Provenance chain is verifiable: given any entry, the chain can be traversed back to genesis
- Compliance report generation: produce schema governance reports in JSON and Markdown formats
- Audit log queryable by entity_type, entity_id, actor, action, and time range

### 6.11 Database and Infrastructure
- V047 database migration creates all 10 tables, 3 hypertables, and 2 continuous aggregates
- Migration is idempotent (can be re-run without error)
- All foreign key relationships correctly defined with ON DELETE CASCADE where appropriate
- Indexes on frequently queried columns (namespace, name, schema_id, status, created_at)
- TimescaleDB hypertables use 7-day chunk intervals
- K8s manifests (Deployment, Service, ConfigMap, HPA) ready for deployment
- Dockerfile with multi-stage build, non-root user, health check
- CI/CD pipeline with lint, test, build, push, deploy stages

### 6.12 Testing
- 1500+ unit tests passing across all 7 engines
- Integration tests covering multi-engine pipeline scenarios
- Edge case tests: empty schemas, deeply nested schemas (10+ levels), schemas with 500+ fields, circular references
- Compatibility test suite: verify all Confluent Schema Registry compatibility rules
- Migration test suite: verify rollback, checkpoint/resume, parallel execution, timeout handling
- Drift detection test suite: verify detection of all drift types
- Performance tests: schema registration (< 50ms), version creation (< 100ms), change detection (< 200ms for 500-field schemas), migration execution (< 1s per 1000 records)
- 95%+ code coverage across all engines

## 7. Non-Functional Requirements

### 7.1 Performance
- Schema registration: < 50ms per schema
- Version creation with auto-classification: < 100ms per version
- Change detection: < 200ms for schemas with up to 500 fields
- Compatibility check: < 100ms per version pair
- Migration plan generation: < 500ms per plan
- Migration execution: < 1 second per 1000 records (for simple transformations)
- Drift detection: < 5 seconds per schema (sampling GL_SM_DRIFT_SAMPLE_SIZE records)
- API response time: < 200ms for non-migration endpoints (p95)

### 7.2 Reliability
- All operations are idempotent where possible
- Migration execution uses checkpoints; no data loss on crash
- Automatic rollback on failure ensures data consistency
- Graceful degradation: if Redis is unavailable, bypass cache and operate directly against PostgreSQL
- If Layer 1 dependencies (SchemaCompiler, ValidityChecker) are unavailable, log warning and skip validation steps but continue core operations

### 7.3 Security
- All API endpoints protected by JWT authentication and RBAC authorization
- Schema definitions may contain sensitive field names; access controlled by namespace-level permissions
- Migration execution logs may contain sample data; redact PII fields per SEC-011 PII Detection
- Audit log entries are append-only; no deletion permitted
- Database connections use TLS 1.3 (SEC-004)

### 7.4 Code Quality
- Pure Python implementation; no external ML dependencies
- Zero-hallucination: all change detection and compatibility checks are deterministic rule-based logic, never LLM-generated
- Thread-safe engines: all engine instances safe for concurrent access via threading.Lock where necessary
- Type hints on all public methods (mypy strict)
- Docstrings on all public classes and methods (Google style)
- No global mutable state; all state in engine instances or database

### 7.5 Scalability
- Support up to GL_SM_MAX_SCHEMAS (50,000) registered schemas
- Support up to GL_SM_MAX_VERSIONS_PER_SCHEMA (1,000) versions per schema
- Migration batch processing up to GL_SM_MAX_MIGRATION_BATCH_SIZE (10,000) records per batch
- Horizontal scaling via K8s HPA based on CPU and active migration count
- Database connection pooling with GL_SM_POOL_SIZE connections

## 8. Dependencies

### 8.1 Layer 1 Dependencies
| Component | Import Path | Usage |
|-----------|-------------|-------|
| AGENT-FOUND-002 Schema Compiler | `greenlang.agents.foundation.schema_compiler.SchemaCompilerEngine` | JSON Schema validation, AST-to-IR compilation, 7-phase pipeline |
| AGENT-FOUND-002 Schema Registry | `greenlang.agents.foundation.schema_compiler.SchemaRegistry` | Git-backed schema storage, baseline versioning |
| AGENT-DATA-010 Data Quality Profiler | `greenlang.data_quality_profiler.validity_checker.ValidityChecker` | Schema validation for drift detection and migration verification |
| AGENT-DATA-010 Data Quality Profiler | `greenlang.data_quality_profiler.consistency_analyzer.ConsistencyAnalyzer` | Cross-dataset consistency checks post-migration |
| AGENT-DATA-002 Excel Normalizer | `greenlang.excel_normalizer.column_mapper.ColumnMapper` | 200+ canonical fields and 500+ synonym mappings for field harmonization |
| AGENT-DATA-002 Excel Normalizer | `greenlang.excel_normalizer.data_type_detector.DataTypeDetector` | 14 data type auto-detection for type inference during drift checks |

### 8.2 Infrastructure Dependencies
| Component | Version | Purpose |
|-----------|---------|---------|
| PostgreSQL | 15+ | Primary data store for all schema, version, migration, and audit data |
| TimescaleDB | 2.x | Hypertables for time-series events (changes, executions, drift) |
| Redis | 7.x | Caching schema lookups, migration state, and rate limiting |
| Prometheus | 2.x | Metrics collection and alerting |
| Kubernetes | 1.28+ | Container orchestration, HPA, health probes |

### 8.3 Python Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.109+ | REST API framework |
| pydantic | 2.x | Request/response models, settings management |
| psycopg | 3.x | Async PostgreSQL driver |
| psycopg_pool | 3.x | Connection pooling |
| redis | 5.x | Redis client for caching |
| prometheus_client | 0.20+ | Metrics exposition |
| jsonschema | 4.x | JSON Schema Draft 2020-12 validation |
| deepdiff | 7.x | Deep structural comparison for change detection |
| semver | 3.x | Semantic version parsing and comparison |

### 8.4 Downstream Dependents
The following agents and pipelines consume schemas managed by this agent and will be affected by schema changes:
- AGENT-DATA-001 through AGENT-DATA-010 (all Layer 1 data agents)
- AGENT-DATA-011 through AGENT-DATA-016 (all Layer 2 data quality agents)
- AGENT-FOUND-001 Orchestrator (DAG execution uses schema-validated inputs)
- AGENT-FOUND-009 QA Test Harness (golden file schemas)
- All emission calculation pipelines (schema-defined emission factor tables)

## 9. File Structure
```
greenlang/
  schema_migration/
    __init__.py
    config.py                          # GL_SM_ settings (Pydantic BaseSettings)
    models.py                          # Pydantic v2 models (Schema, Version, Change, Plan, Execution, etc.)
    engines/
      __init__.py
      schema_registry.py               # SchemaRegistryEngine
      schema_versioner.py              # SchemaVersionerEngine
      change_detector.py               # ChangeDetectorEngine
      compatibility_checker.py         # CompatibilityCheckerEngine
      migration_planner.py             # MigrationPlannerEngine
      migration_executor.py            # MigrationExecutorEngine
      schema_migration_pipeline.py     # SchemaMigrationPipelineEngine
    api/
      __init__.py
      routes.py                        # 20 FastAPI endpoints
      dependencies.py                  # Dependency injection
    provenance.py                      # SHA-256 provenance chain manager
    metrics.py                         # 12 Prometheus metrics definitions
    layer1_reexports.py                # Re-exports from Layer 1

deployment/
  database/migrations/
    V047__schema_migration_service.sql # 10 tables + 3 hypertables + 2 continuous aggregates

  kubernetes/
    schema-migration/
      deployment.yaml
      service.yaml
      configmap.yaml
      hpa.yaml

tests/
  unit/
    test_schema_migration/
      __init__.py
      test_schema_registry.py          # ~250 tests
      test_schema_versioner.py         # ~200 tests
      test_change_detector.py          # ~300 tests
      test_compatibility_checker.py    # ~250 tests
      test_migration_planner.py        # ~200 tests
      test_migration_executor.py       # ~200 tests
      test_pipeline.py                 # ~150 tests
      test_models.py                   # ~50 tests
      test_provenance.py               # ~50 tests
      test_api.py                      # ~50 tests
  integration/
    test_schema_migration_integration.py
```

## 10. Glossary
| Term | Definition |
|------|------------|
| Schema | A formal definition of data structure including field names, types, constraints, and relationships |
| Semantic Versioning (SemVer) | Version numbering as MAJOR.MINOR.PATCH where MAJOR = breaking, MINOR = additive, PATCH = cosmetic |
| Backward Compatible | New schema can read/deserialize data produced with old schema |
| Forward Compatible | Old schema can read/deserialize data produced with new schema |
| Full Compatible | Both backward and forward compatible |
| Breaking Change | A schema change that is neither backward nor forward compatible |
| Schema Drift | When actual data in a pipeline deviates from the registered/expected schema |
| Migration Plan | An ordered list of transformation steps to convert data from source schema to target schema |
| Checkpoint | A saved state during migration execution that allows rollback to that point |
| Provenance Chain | A linked sequence of SHA-256 hashes providing an auditable trail of all operations |
| Namespace | A logical grouping for schemas (e.g., "emissions", "supply-chain", "compliance") |
| Sunset Date | The date after which a deprecated schema version will no longer be supported |
| Field Harmonization | Mapping equivalent fields across different source schemas to a canonical representation |

## 11. References
- Confluent Schema Registry Compatibility Rules: https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html
- JSON Schema Draft 2020-12: https://json-schema.org/draft/2020-12/json-schema-core
- Apache Avro Schema Evolution: https://avro.apache.org/docs/current/specification/
- GreenLang AGENT-FOUND-002 Schema Compiler PRD
- GreenLang AGENT-DATA-010 Data Quality Profiler PRD
- EU CSRD/ESRS Data Governance Requirements (ESRS 2 GOV-1)
- GHG Protocol Corporate Standard, Chapter 7 (Data Management)
