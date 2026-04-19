# PRD: AGENT-DATA-018 — GL-DATA-X-021 Data Lineage Tracker

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-DATA-X-021 |
| Internal Label | AGENT-DATA-018 |
| Category | Layer 2 – Data Quality Agents |
| Purpose | Track end-to-end data lineage across all GreenLang agent pipelines — register data assets, capture transformation events, build lineage graphs, perform forward/backward impact analysis, validate lineage completeness, generate lineage visualizations and compliance reports, and maintain full provenance of every data movement and transformation |
| Estimated Variants | 200 |
| Status | To Be Built |
| Author | GreenLang Platform Team |
| Date | February 2026 |
| DB Migration | V048 |

## 2. Problem Statement
GreenLang processes environmental data through 47+ agent pipelines spanning data intake, quality, calculations, and reporting. Data flows through multiple transformations — from raw supplier invoices to GHG Protocol Scope 3 emission calculations to CSRD/ESRS compliance reports. Organizations face critical challenges:

1. **No data asset registry** — cannot catalog and discover datasets, fields, agents, and pipelines as first-class lineage nodes, making it impossible to understand what data exists and how it is organized
2. **No transformation tracking** — cannot capture when, where, and how data is transformed (filtered, aggregated, joined, calculated, imputed, deduplicated), leading to opaque data processing with no audit trail
3. **No lineage graph** — no directed acyclic graph (DAG) representation of data flow from source to destination, preventing visualization and traversal of data dependencies
4. **No backward lineage (provenance)** — cannot answer "where did this number come from?" — critical for GHG Protocol audit requirements and CSRD/ESRS data quality disclosures
5. **No forward lineage (impact analysis)** — cannot answer "what downstream reports and calculations are affected if this source data changes?" — essential for data quality incident response
6. **No lineage validation** — cannot verify that lineage is complete (no orphan nodes, no broken links, full source-to-report coverage), undermining confidence in data governance
7. **No column-level lineage** — cannot trace individual field/column transformations through the pipeline, only dataset-level relationships, missing critical granularity for compliance
8. **No cross-agent lineage** — data flowing between agents (e.g., PDF Extractor → Excel Normalizer → Spend Categorizer → Emission Calculator) has no unified lineage view
9. **No lineage-based data quality scoring** — cannot score data quality based on lineage characteristics (depth, freshness, transformation count, source credibility)
10. **No compliance lineage reports** — CSRD/ESRS Article 8 requires documented data flow from source to disclosure; GHG Protocol requires traceable calculation chains; SOC 2 requires data processing audit trails
11. **No lineage change detection** — cannot detect when lineage patterns change (new data sources added, transformation logic modified, pipeline topology altered)
12. **No lineage visualization** — no ability to render interactive lineage graphs for data stewards, auditors, and compliance officers

## 3. Existing Layer 1 Capabilities
- `greenlang.agents.foundation.orchestrator.OrchestratorEngine` — DAG execution engine with topological sort, provenance chain tracking, and execution logs
- `greenlang.agents.foundation.reproducibility.ReproducibilityEngine` — SHA-256 artifact hashing, input/output hash verification, drift detection
- `greenlang.agents.foundation.schema_compiler.SchemaCompilerEngine` — Schema validation and registry with version tracking
- `greenlang.agents.foundation.citations.CitationsEngine` — Source citation tracking with regulatory framework mapping
- `greenlang.data_quality_profiler.dataset_profiler.DatasetProfiler` — Dataset-level profiling with completeness, validity, and consistency scoring
- `greenlang.cross_source_reconciliation.source_registry.SourceRegistryEngine` — Source registration with credibility scoring

## 4. Identified Gaps (12)
| # | Gap | Layer 1 Provides | Layer 2 Needed |
|---|-----|------------------|----------------|
| 1 | Data asset registry | Orchestrator tracks agent nodes; source registry tracks data sources | Universal asset registry for datasets, fields, agents, pipelines, reports — each as a typed node with metadata, ownership, tags, classification level, and lifecycle status |
| 2 | Transformation event capture | Orchestrator logs execution events | Structured transformation events: input assets, output assets, transformation type (filter/aggregate/join/calculate/impute/deduplicate/enrich/merge/split), agent responsible, timestamp, record counts, transformation logic description, parameters |
| 3 | Lineage graph construction | Orchestrator has DAG for task execution | Data lineage DAG: nodes = data assets, edges = transformations. Support column-level lineage (field A in dataset X → field B in dataset Y via aggregation). Incremental graph building from transformation events |
| 4 | Backward lineage queries | Reproducibility tracks input/output hashes | Given any data asset or field, traverse upstream: all source data, transformations applied, agents involved, timestamps. Answer "where did this value come from?" with full chain |
| 5 | Forward lineage queries | None | Given any data asset or field, traverse downstream: all consumers, reports, calculations affected. Answer "what breaks if this source changes?" with impact severity scoring |
| 6 | Lineage validation | None | Completeness checks: no orphan nodes (every intermediate node has both upstream and downstream), no broken edges, source-to-report coverage percentage, cycle detection, freshness verification of lineage metadata |
| 7 | Column-level lineage | None | Fine-grained field-to-field lineage tracking: field renames, type casts, aggregations (field A → SUM(field A)), computed fields (field C = field A × field B), multi-field merges, conditional logic |
| 8 | Cross-agent lineage stitching | Agent registry catalogs agents | Stitch lineage across agent boundaries by matching output assets of one agent to input assets of the next. Build unified end-to-end lineage graph spanning all 47+ agents |
| 9 | Lineage-based quality scoring | Data quality profiler scores individual datasets | Score based on lineage characteristics: source credibility (authoritative vs estimated), transformation depth (fewer = better), data freshness through chain, number of manual interventions, completeness of lineage documentation |
| 10 | Compliance reporting | None | Generate CSRD/ESRS data lineage disclosures, GHG Protocol calculation chain documentation, SOC 2 data flow audit reports, custom compliance templates with lineage evidence |
| 11 | Lineage change detection | None | Monitor lineage topology for changes: new nodes, removed nodes, new edges, removed edges, transformation logic changes. Alert on unexpected lineage pattern shifts |
| 12 | Lineage visualization | None | Generate Mermaid diagrams, DOT/Graphviz notation, JSON graph format, D3-compatible adjacency lists. Support filtering by depth, agent, dataset, time range. Export for Grafana and web dashboards |

## 5. Architecture

### 5.1 Seven Engines
| Engine | Class | Responsibility |
|--------|-------|----------------|
| 1 | AssetRegistryEngine (`asset_registry.py`) | Register and manage data assets as lineage graph nodes. Asset types: dataset, field, agent, pipeline, report, metric, external_source. Each asset has a unique ID, qualified name, asset type, owner, tags, classification level (public/internal/confidential/restricted), status (active/deprecated/archived), schema reference, and metadata. Support search, filtering, grouping, and bulk operations |
| 2 | TransformationTrackerEngine (`transformation_tracker.py`) | Capture and store data transformation events as lineage graph edges. Each event records: source assets (inputs), target assets (outputs), transformation type (filter/aggregate/join/calculate/impute/deduplicate/enrich/merge/split/validate/normalize/classify), agent ID, pipeline ID, execution ID, timestamp, record counts (input/output/filtered/error), transformation logic description, parameters, duration. Support batch event ingestion and event replay |
| 3 | LineageGraphEngine (`lineage_graph.py`) | Build and maintain the lineage DAG from registered assets and transformation events. Adjacency list representation with node and edge metadata. Support incremental graph updates, subgraph extraction, topological ordering, connected component analysis, shortest path between assets, graph statistics (node count, edge count, depth, breadth). Both dataset-level and column-level lineage in a unified graph |
| 4 | ImpactAnalyzerEngine (`impact_analyzer.py`) | Perform forward and backward lineage traversal with impact scoring. Backward: trace any asset to all its upstream sources (data provenance). Forward: trace any asset to all its downstream consumers (impact analysis). Impact scoring: severity (critical/high/medium/low) based on consumer importance, data freshness sensitivity, and transformation complexity. Dependency matrix generation, blast radius calculation, root cause analysis |
| 5 | LineageValidatorEngine (`lineage_validator.py`) | Validate lineage graph completeness and consistency. Checks: orphan node detection (nodes with no edges), broken edge detection (edges referencing non-existent nodes), cycle detection (lineage should be acyclic), source coverage (percentage of report fields traceable to authoritative sources), freshness validation (lineage metadata up-to-date), completeness scoring per pipeline and per report, gap identification with remediation suggestions |
| 6 | LineageReporterEngine (`lineage_reporter.py`) | Generate lineage reports and visualizations. Output formats: Mermaid markdown, DOT/Graphviz, JSON graph, D3-compatible adjacency list, plain text summary. Compliance reports: CSRD/ESRS data flow documentation, GHG Protocol calculation chain, SOC 2 data processing audit, custom templates. Support filtering by depth, asset type, agent, time range, classification level. Lineage statistics and health dashboards |
| 7 | LineageTrackerPipelineEngine (`lineage_tracker_pipeline.py`) | End-to-end orchestration: register assets from pipeline metadata, capture transformation events from agent executions, build/update lineage graph, validate completeness, generate reports. Support scheduled lineage refresh, event-driven updates, batch processing, pipeline configuration (skip validation, report format, depth limit). Lineage change detection: compare current graph topology with previous snapshot, detect new/removed nodes and edges, alert on unexpected changes |

### 5.2 Data Flow
```
Asset Registration → AssetRegistryEngine (catalog datasets, fields, agents, pipelines)
                   → TransformationTrackerEngine (capture transformation events)
                   → LineageGraphEngine (build/update lineage DAG)
                   → ImpactAnalyzerEngine (forward/backward traversal + scoring)
                   → LineageValidatorEngine (completeness + consistency checks)
                   → LineageReporterEngine (visualizations + compliance reports)
                   → LineageTrackerPipelineEngine (orchestration + change detection)
```

### 5.3 Database Schema (V048)
- `lineage_assets` — registered data assets (id UUID PK, qualified_name VARCHAR UNIQUE, asset_type ENUM[dataset/field/agent/pipeline/report/metric/external_source], display_name VARCHAR, owner VARCHAR, tags JSONB, classification ENUM[public/internal/confidential/restricted], status ENUM[active/deprecated/archived], schema_ref VARCHAR, description TEXT, metadata JSONB, created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ)
- `lineage_transformations` — transformation events / edges (id UUID PK, transformation_type ENUM[filter/aggregate/join/calculate/impute/deduplicate/enrich/merge/split/validate/normalize/classify], agent_id VARCHAR, pipeline_id VARCHAR, execution_id VARCHAR, description TEXT, parameters JSONB, records_in INTEGER, records_out INTEGER, records_filtered INTEGER, records_error INTEGER, duration_ms INTEGER, started_at TIMESTAMPTZ, completed_at TIMESTAMPTZ, metadata JSONB)
- `lineage_edges` — source-to-target edges linking assets via transformations (id UUID PK, source_asset_id UUID FK, target_asset_id UUID FK, transformation_id UUID FK, edge_type ENUM[dataset_level/column_level], source_field VARCHAR NULL, target_field VARCHAR NULL, transformation_logic TEXT, confidence FLOAT, created_at TIMESTAMPTZ)
- `lineage_graph_snapshots` — periodic graph topology snapshots for change detection (id UUID PK, snapshot_name VARCHAR, node_count INTEGER, edge_count INTEGER, max_depth INTEGER, connected_components INTEGER, orphan_count INTEGER, coverage_score FLOAT, graph_hash VARCHAR, snapshot_data JSONB, created_at TIMESTAMPTZ)
- `lineage_impact_analyses` — stored impact analysis results (id UUID PK, root_asset_id UUID FK, direction ENUM[forward/backward], depth INTEGER, affected_assets_count INTEGER, critical_count INTEGER, high_count INTEGER, medium_count INTEGER, low_count INTEGER, blast_radius FLOAT, analysis_result JSONB, created_at TIMESTAMPTZ)
- `lineage_validations` — lineage validation results (id UUID PK, scope VARCHAR, orphan_nodes INTEGER, broken_edges INTEGER, cycles_detected INTEGER, source_coverage FLOAT, completeness_score FLOAT, freshness_score FLOAT, issues_json JSONB, recommendations_json JSONB, validated_at TIMESTAMPTZ)
- `lineage_reports` — generated compliance and visualization reports (id UUID PK, report_type ENUM[csrd_esrs/ghg_protocol/soc2/custom/visualization], format ENUM[mermaid/dot/json/d3/text/html/pdf], scope VARCHAR, parameters JSONB, content TEXT, report_hash VARCHAR, generated_by VARCHAR, generated_at TIMESTAMPTZ)
- `lineage_change_events` — detected lineage topology changes (id UUID PK, previous_snapshot_id UUID FK, current_snapshot_id UUID FK, change_type ENUM[node_added/node_removed/edge_added/edge_removed/topology_changed], entity_id UUID, entity_type VARCHAR, details JSONB, severity ENUM[low/medium/high/critical], detected_at TIMESTAMPTZ)
- `lineage_quality_scores` — lineage-based data quality scores (id UUID PK, asset_id UUID FK, source_credibility FLOAT, transformation_depth INTEGER, freshness_score FLOAT, documentation_score FLOAT, manual_intervention_count INTEGER, overall_score FLOAT, scoring_details JSONB, scored_at TIMESTAMPTZ)
- `lineage_audit_log` — all actions with provenance (id UUID PK, action VARCHAR, entity_type VARCHAR, entity_id UUID, actor VARCHAR, details_json JSONB, previous_state JSONB, new_state JSONB, provenance_hash VARCHAR, parent_hash VARCHAR, created_at TIMESTAMPTZ)
- 3 hypertables (7-day chunks):
  - `lineage_transformation_events` — time-series of transformation events (agent_id, pipeline_id, transformation_type, records_in, records_out, duration_ms, ts TIMESTAMPTZ)
  - `lineage_validation_events` — time-series of validation runs (scope, orphan_count, broken_count, coverage_score, completeness_score, ts TIMESTAMPTZ)
  - `lineage_impact_events` — time-series of impact analyses (root_asset_id, direction, affected_count, critical_count, blast_radius, ts TIMESTAMPTZ)
- 2 continuous aggregates:
  - `lineage_transformations_hourly_stats` — hourly rollup of transformation events by type, agent, and pipeline
  - `lineage_validations_hourly_stats` — hourly rollup of validation results by scope and completeness score

### 5.4 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| gl_dlt_assets_registered_total | Counter | Total data assets registered, labeled by asset_type and classification |
| gl_dlt_transformations_captured_total | Counter | Total transformation events captured, labeled by transformation_type and agent_id |
| gl_dlt_edges_created_total | Counter | Total lineage edges created, labeled by edge_type (dataset_level/column_level) |
| gl_dlt_impact_analyses_total | Counter | Impact analyses performed, labeled by direction (forward/backward) and severity |
| gl_dlt_validations_total | Counter | Lineage validations performed, labeled by result (pass/warn/fail) |
| gl_dlt_reports_generated_total | Counter | Reports generated, labeled by report_type and format |
| gl_dlt_change_events_total | Counter | Lineage change events detected, labeled by change_type and severity |
| gl_dlt_quality_scores_computed_total | Counter | Quality scores computed, labeled by score_tier (excellent/good/fair/poor) |
| gl_dlt_graph_traversal_duration_seconds | Histogram | Graph traversal duration, buckets [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30] |
| gl_dlt_processing_duration_seconds | Histogram | Engine operation processing duration, labeled by operation |
| gl_dlt_graph_node_count | Gauge | Current total nodes in the lineage graph |
| gl_dlt_graph_edge_count | Gauge | Current total edges in the lineage graph |

### 5.5 API Endpoints (20)
| # | Method | Path | Purpose |
|---|--------|------|---------|
| 1 | POST | /api/v1/data-lineage/assets | Register a new data asset (dataset, field, agent, pipeline, report) |
| 2 | GET | /api/v1/data-lineage/assets | List registered assets with filtering by type, owner, classification, status, tags |
| 3 | GET | /api/v1/data-lineage/assets/{id} | Get full asset details including lineage summary (upstream/downstream counts) |
| 4 | PUT | /api/v1/data-lineage/assets/{id} | Update asset metadata (owner, tags, status, description, classification) |
| 5 | DELETE | /api/v1/data-lineage/assets/{id} | Deregister asset (soft delete, sets status to archived) |
| 6 | POST | /api/v1/data-lineage/transformations | Record a transformation event with source/target assets |
| 7 | GET | /api/v1/data-lineage/transformations | List transformation events with filtering by type, agent, pipeline, time range |
| 8 | POST | /api/v1/data-lineage/edges | Create a lineage edge (dataset-level or column-level) between two assets |
| 9 | GET | /api/v1/data-lineage/edges | List edges with filtering by source, target, edge type, transformation |
| 10 | GET | /api/v1/data-lineage/graph | Get the full or filtered lineage graph as JSON adjacency list |
| 11 | GET | /api/v1/data-lineage/graph/subgraph/{asset_id} | Extract subgraph centered on a specific asset with configurable depth |
| 12 | GET | /api/v1/data-lineage/backward/{asset_id} | Backward lineage: trace upstream to all sources with full transformation chain |
| 13 | GET | /api/v1/data-lineage/forward/{asset_id} | Forward lineage: trace downstream to all consumers with impact scores |
| 14 | POST | /api/v1/data-lineage/impact | Run impact analysis for a specific asset (forward + backward + blast radius) |
| 15 | POST | /api/v1/data-lineage/validate | Validate lineage completeness and consistency for a scope/pipeline |
| 16 | GET | /api/v1/data-lineage/validate/{id} | Get validation result details including issues and recommendations |
| 17 | POST | /api/v1/data-lineage/reports | Generate a lineage report (compliance, visualization, or custom) |
| 18 | POST | /api/v1/data-lineage/pipeline | Run the full lineage tracking pipeline: register, capture, build, validate, report |
| 19 | GET | /api/v1/data-lineage/health | Health check returning engine statuses, graph stats, and validation summary |
| 20 | GET | /api/v1/data-lineage/stats | Service statistics: assets registered, transformations captured, graph size, coverage scores |

### 5.6 Configuration (GL_DLT_ prefix)
| Setting | Default | Description |
|---------|---------|-------------|
| GL_DLT_DATABASE_URL | postgresql://localhost:5432/greenlang | PostgreSQL connection string |
| GL_DLT_REDIS_URL | redis://localhost:6379/0 | Redis cache connection string |
| GL_DLT_LOG_LEVEL | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |
| GL_DLT_MAX_ASSETS | 100000 | Maximum number of registered data assets |
| GL_DLT_MAX_TRANSFORMATIONS | 500000 | Maximum transformation events stored |
| GL_DLT_MAX_EDGES | 1000000 | Maximum lineage edges in the graph |
| GL_DLT_MAX_GRAPH_DEPTH | 50 | Maximum traversal depth for lineage queries |
| GL_DLT_DEFAULT_TRAVERSAL_DEPTH | 10 | Default depth for forward/backward traversal |
| GL_DLT_SNAPSHOT_INTERVAL_MINUTES | 60 | Interval between automatic graph snapshots |
| GL_DLT_ENABLE_COLUMN_LINEAGE | true | Enable column-level lineage tracking |
| GL_DLT_ENABLE_CHANGE_DETECTION | true | Enable lineage topology change detection |
| GL_DLT_ENABLE_PROVENANCE | true | Enable SHA-256 provenance chain tracking |
| GL_DLT_GENESIS_HASH | greenlang-data-lineage-genesis | Genesis anchor for provenance chain |
| GL_DLT_POOL_SIZE | 5 | Database connection pool size |
| GL_DLT_CACHE_TTL | 300 | Cache time-to-live in seconds |
| GL_DLT_RATE_LIMIT | 200 | Maximum API requests per minute |
| GL_DLT_BATCH_SIZE | 1000 | Default batch size for bulk operations |
| GL_DLT_COVERAGE_WARN_THRESHOLD | 0.8 | Coverage score below which a warning is raised |
| GL_DLT_COVERAGE_FAIL_THRESHOLD | 0.5 | Coverage score below which validation fails |
| GL_DLT_FRESHNESS_MAX_AGE_HOURS | 24 | Maximum age (hours) before lineage metadata is stale |
| GL_DLT_QUALITY_SCORE_WEIGHTS | {"source_credibility":0.3,"transformation_depth":0.2,"freshness":0.2,"documentation":0.15,"manual_interventions":0.15} | Weights for lineage-based quality scoring |
| GL_DLT_ENABLE_METRICS | true | Enable Prometheus metrics collection |

### 5.7 Layer 1 Re-exports
| Source | Symbol | Purpose |
|--------|--------|---------|
| greenlang.agents.foundation.orchestrator | OrchestratorEngine | DAG execution for lineage pipeline |
| greenlang.agents.foundation.reproducibility | ReproducibilityEngine | SHA-256 provenance tracking |
| greenlang.cross_source_reconciliation.source_registry | SourceRegistryEngine | Source credibility scoring |
| greenlang.data_quality_profiler.models | QualityDimension | Quality dimension classification |

### 5.8 Auth Integration
- Router prefix: `/api/v1/data-lineage`
- Router variable: `dlt_router`
- PERMISSION_MAP entries: `lineage:read`, `lineage:write`, `lineage:admin`, `lineage:validate`, `lineage:report`
- Registration in `auth_setup.py`: import `get_router as get_dlt_router` from `greenlang.data_lineage_tracker.setup`

## 6. Acceptance Criteria
1. All 7 engines implemented with full docstrings and type hints
2. 20 REST API endpoints functional via FastAPI router
3. 12 Prometheus metrics with `gl_dlt_` prefix
4. V048 database migration (10 tables + 3 hypertables + 2 continuous aggregates)
5. SHA-256 provenance chain for all operations
6. Thread-safe configuration with `GL_DLT_` environment variable prefix
7. Unit tests: 800+ tests with >85% code coverage
8. Integration tests: 50+ end-to-end tests
9. Dockerfile for containerized deployment
10. CI/CD GitHub Actions workflow
11. Kubernetes manifests (10 files: deployment, service, configmap, secret, HPA, PDB, network policy, service monitor, alerts, Grafana dashboard)
12. Auth integration in `auth_setup.py`
13. Layer 1 re-exports with graceful fallback stubs
14. Mermaid, DOT, and JSON lineage visualization output
15. CSRD/ESRS and GHG Protocol compliance report generation

## 7. Dependencies
| Dependency | Agent | Purpose |
|------------|-------|---------|
| Orchestrator | AGENT-FOUND-001 | DAG execution patterns, provenance chain |
| Schema Compiler | AGENT-FOUND-002 | Schema validation for registered assets |
| Reproducibility | AGENT-FOUND-008 | SHA-256 hashing, drift detection patterns |
| Citations | AGENT-FOUND-005 | Source authority and regulatory framework mapping |
| Data Quality Profiler | AGENT-DATA-010 | Quality dimension re-exports |
| Cross-Source Reconciliation | AGENT-DATA-015 | Source registry and credibility scoring re-exports |
| Schema Migration | AGENT-DATA-017 | Schema version tracking patterns |

## 8. Risk Assessment
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Graph performance at scale (>100K nodes) | Medium | High | Adjacency list with indexes, BFS/DFS with depth limits, Redis caching for hot paths |
| Column-level lineage explosion | Medium | Medium | Configurable enable/disable, lazy loading, subgraph extraction |
| Incomplete lineage capture | High | High | Validation engine with coverage scoring, gap identification, remediation suggestions |
| Cross-agent stitching complexity | Medium | Medium | Standardized asset naming convention (agent.pipeline.dataset.field), fuzzy matching |
