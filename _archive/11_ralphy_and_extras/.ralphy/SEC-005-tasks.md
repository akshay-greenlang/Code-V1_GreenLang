# SEC-005: Centralized Audit Logging Service - Development Tasks

**Status:** COMPLETE
**Created:** 2026-02-06
**Completed:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** SEC-001, SEC-002, SEC-003, INFRA-002, INFRA-009
**Result:** 52 new files + 2 modified, ~18,000+ lines, 386 tests

---

## Phase 1: Core Audit Service (P0)

### 1.1 Package Init
- [x] Create `greenlang/infrastructure/audit_service/__init__.py`:
  - Public API exports: AuditService, UnifiedAuditEvent, AuditEventType, etc.
  - AuditServiceConfig dataclass with environment-aware defaults
  - Version constant

### 1.2 Event Types Consolidation
- [x] Create `greenlang/infrastructure/audit_service/event_types.py`:
  - `AuditEventCategory` enum (auth, rbac, encryption, data, agent, system, api, compliance)
  - `AuditSeverity` enum (debug, info, warning, error, critical)
  - `UnifiedAuditEventType` enum consolidating 70+ event types from auth/rbac/encryption
  - `AuditAction` enum (create, read, update, delete, execute, export)
  - `AuditResult` enum (success, failure, denied, error)

### 1.3 Event Data Model
- [x] Create `greenlang/infrastructure/audit_service/event_model.py`:
  - `UnifiedAuditEvent` dataclass with 25+ fields
  - `EventBuilder` for fluent event construction
  - `to_dict()` with PII redaction
  - `to_json()` for Loki emission
  - `from_auth_event()`, `from_rbac_event()`, `from_encryption_event()` converters

### 1.4 Event Collector
- [x] Create `greenlang/infrastructure/audit_service/event_collector.py`:
  - `AuditEventCollector` class with async queue
  - `collect(event)` - add to internal queue
  - `collect_batch(events)` - bulk collection
  - Backpressure handling when queue is full
  - Metrics for queue depth and collection rate

### 1.5 Event Enricher
- [x] Create `greenlang/infrastructure/audit_service/event_enricher.py`:
  - `AuditEventEnricher` class
  - `enrich(event)` - add geo-IP, user agent, context
  - GeoIP2 integration (MaxMind database)
  - User agent parsing (ua-parser)
  - Request context injection (tenant, user, session)

### 1.6 Event Router
- [x] Create `greenlang/infrastructure/audit_service/event_router.py`:
  - `AuditEventRouter` class
  - `route(event)` - write to DB, Loki, Redis concurrently
  - PostgreSQL batch writer with retry
  - Loki structured JSON emitter
  - Redis pub/sub publisher for streaming

### 1.7 Audit Service
- [x] Create `greenlang/infrastructure/audit_service/audit_service.py`:
  - `AuditService` class orchestrating collector, enricher, router
  - `log_event(event)` - main entry point
  - `log_auth_event(...)`, `log_rbac_event(...)`, etc. - convenience methods
  - Background worker for async processing
  - Graceful shutdown with flush

### 1.8 Audit Cache
- [x] Create `greenlang/infrastructure/audit_service/audit_cache.py`:
  - `AuditCache` for recent event deduplication
  - Redis-backed with 5-minute TTL
  - `check_duplicate(event_id)` - prevent reprocessing
  - `mark_processed(event_id)` - record completion

---

## Phase 2: Audit API (P0)

### 2.1 API Init
- [x] Create `greenlang/infrastructure/audit_service/api/__init__.py`:
  - Export audit_router combining all sub-routers

### 2.2 Events API
- [x] Create `greenlang/infrastructure/audit_service/api/events_routes.py`:
  - `GET /api/v1/audit/events` - List with filtering, pagination
  - `GET /api/v1/audit/events/{event_id}` - Single event by ID
  - Pydantic request/response models
  - Filter params: time_range, event_types, severity, tenant_id, user_id, etc.
  - Cursor-based and offset pagination

### 2.3 Search API
- [x] Create `greenlang/infrastructure/audit_service/api/search_routes.py`:
  - `POST /api/v1/audit/search` - Advanced search
  - LogQL-like query syntax parser
  - Full-text search on metadata
  - Aggregation support (count by type, group by user)

### 2.4 Stats API
- [x] Create `greenlang/infrastructure/audit_service/api/stats_routes.py`:
  - `GET /api/v1/audit/stats` - Event statistics
  - `GET /api/v1/audit/timeline` - Activity timeline
  - `GET /api/v1/audit/hotspots` - Most active users/resources
  - Time-series data for charts

### 2.5 Stream API
- [x] Create `greenlang/infrastructure/audit_service/api/stream_routes.py`:
  - WebSocket endpoint `/api/v1/audit/events/stream`
  - Redis pub/sub subscription
  - Filter by event type, tenant, severity
  - Heartbeat and reconnection handling

---

## Phase 3: Compliance Reporting (P0)

### 3.1 Report Service
- [x] Create `greenlang/infrastructure/audit_service/reporting/report_service.py`:
  - `ComplianceReportService` class
  - `generate_report(report_type, period, format)` - async report generation
  - Report queuing and background processing
  - Report storage and retrieval

### 3.2 SOC 2 Report Generator
- [x] Create `greenlang/infrastructure/audit_service/reporting/soc2_report.py`:
  - `SOC2ReportGenerator` class
  - Map events to CC6, CC7, CC8 controls
  - Executive summary generation
  - Control evidence collection
  - Gap analysis

### 3.3 ISO 27001 Report Generator
- [x] Create `greenlang/infrastructure/audit_service/reporting/iso27001_report.py`:
  - `ISO27001ReportGenerator` class
  - Map events to A.9, A.12, A.18 controls
  - Control compliance scoring
  - Audit trail documentation

### 3.4 GDPR Report Generator
- [x] Create `greenlang/infrastructure/audit_service/reporting/gdpr_report.py`:
  - `GDPRReportGenerator` class
  - Data subject access events
  - Data processing activities
  - Right to erasure compliance

### 3.5 Report Routes
- [x] Create `greenlang/infrastructure/audit_service/api/report_routes.py`:
  - `POST /api/v1/audit/reports/soc2` - Generate SOC 2 report
  - `POST /api/v1/audit/reports/iso27001` - Generate ISO 27001 report
  - `POST /api/v1/audit/reports/gdpr` - Generate GDPR report
  - `GET /api/v1/audit/reports/{job_id}` - Check report status
  - `GET /api/v1/audit/reports/{job_id}/download` - Download report

---

## Phase 4: Export Service (P0)

### 4.1 Export Service
- [x] Create `greenlang/infrastructure/audit_service/export/export_service.py`:
  - `AuditExportService` class
  - `export(filters, format)` - async export job creation
  - CSV, JSON, Parquet format support
  - Streaming export for large datasets
  - S3 upload for completed exports

### 4.2 Export Formats
- [x] Create `greenlang/infrastructure/audit_service/export/formats.py`:
  - `CSVExporter` - CSV format writer
  - `JSONExporter` - JSONL format writer
  - `ParquetExporter` - Parquet format with PyArrow
  - Compression support (gzip, zstd)

### 4.3 Export Routes
- [x] Create `greenlang/infrastructure/audit_service/api/export_routes.py`:
  - `POST /api/v1/audit/export` - Create export job
  - `GET /api/v1/audit/export/{job_id}` - Check export status
  - `GET /api/v1/audit/export/{job_id}/download` - Download export

---

## Phase 5: Audit Middleware (P1)

### 5.1 Request Audit Middleware
- [x] Create `greenlang/infrastructure/audit_service/middleware.py`:
  - `AuditMiddleware` for FastAPI
  - Capture request metadata (path, method, headers, timing)
  - Capture response metadata (status, timing)
  - Link to authenticated user context
  - Sensitivity classification by endpoint

### 5.2 Exclusion Rules
- [x] Create `greenlang/infrastructure/audit_service/exclusions.py`:
  - `AuditExclusionRules` class
  - Exclude health checks, metrics, static assets
  - Configurable exclusion patterns
  - Endpoint sensitivity mapping

---

## Phase 6: Metrics & Alerts (P1)

### 6.1 Audit Metrics
- [x] Create `greenlang/infrastructure/audit_service/audit_metrics.py`:
  - `gl_audit_events_total` Counter (event_type, severity, result)
  - `gl_audit_event_latency_seconds` Histogram (stage)
  - `gl_audit_events_queued` Gauge
  - `gl_audit_db_write_failures_total` Counter (error_type)
  - `gl_audit_stream_connections` Gauge
  - `gl_audit_export_jobs_total` Counter (status)
  - `gl_audit_report_generation_seconds` Histogram (report_type)
  - Lazy initialization pattern

### 6.2 Grafana Dashboard
- [x] Create `deployment/monitoring/dashboards/audit-service.json`:
  - Event rate over time (by type, severity) - 2 panels
  - Top users by activity - 1 panel
  - Top resources by access - 1 panel
  - Geographic distribution - 1 panel
  - Auth success/failure ratio - 1 panel
  - Authorization denial heatmap - 1 panel
  - Sensitive data access timeline - 1 panel
  - Anomaly alerts panel - 1 panel
  - Compliance score gauge - 1 panel
  - Active sessions - 1 panel
  - API latency by endpoint - 1 panel
  - Export volume - 1 panel
  - Failed operations breakdown - 1 panel
  - Event queue depth - 1 panel
  - Report status - 1 panel
  - System health summary - 1 panel

### 6.3 Prometheus Alerts
- [x] Create `deployment/monitoring/alerts/audit-service-alerts.yaml`:
  - HighAuditEventRate (>1000 events/sec)
  - AuditWriteFailures (any DB write failure)
  - AuditEventBacklog (queue >1000)
  - MissingAuditEvents (expected events missing)
  - SuspiciousActivityDetected (anomaly trigger)
  - AuditStreamDisconnected (no consumers)
  - ComplianceGapDetected (required audit missing)

---

## Phase 7: Retention & Archival (P1)

### 7.1 Retention Policy Service
- [x] Create `greenlang/infrastructure/audit_service/retention/retention_service.py`:
  - `RetentionPolicyService` class
  - Policy definitions (hot, warm, cold, archive)
  - Automatic tier migration
  - Compliance-aware retention periods

### 7.2 Archival Service
- [x] Create `greenlang/infrastructure/audit_service/retention/archival_service.py`:
  - `ArchivalService` class
  - S3 Parquet writer for cold tier
  - Glacier migration for archive tier
  - Restoration from archive

### 7.3 Retention CronJobs
- [x] Create `deployment/kubernetes/audit-service/retention-cronjob.yaml`:
  - Daily partition rotation
  - Weekly compression job
  - Monthly S3 archival
  - Yearly Glacier migration

---

## Phase 8: Integration (P1)

### 8.1 Auth Setup Integration
- [x] Modify `greenlang/infrastructure/auth_service/auth_setup.py`:
  - Import and include audit_router
  - Configure AuditMiddleware
  - Wire AuditService dependency

### 8.2 Route Protector Update
- [x] Update `greenlang/infrastructure/auth_service/route_protector.py`:
  - Add audit permission mappings to PERMISSION_MAP:
    - `GET:/api/v1/audit/events` -> `audit:read`
    - `GET:/api/v1/audit/events/{event_id}` -> `audit:read`
    - `POST:/api/v1/audit/search` -> `audit:search`
    - `GET:/api/v1/audit/stats` -> `audit:read`
    - `GET:/api/v1/audit/timeline` -> `audit:read`
    - `GET:/api/v1/audit/hotspots` -> `audit:read`
    - `POST:/api/v1/audit/export` -> `audit:export`
    - `GET:/api/v1/audit/export/{job_id}` -> `audit:export`
    - `GET:/api/v1/audit/export/{job_id}/download` -> `audit:export`
    - `POST:/api/v1/audit/reports/*` -> `audit:admin`
    - `GET:/api/v1/audit/reports/*` -> `audit:admin`

### 8.3 RBAC Permissions Migration
- [x] Create `deployment/database/migrations/sql/V013__audit_permissions.sql`:
  - Add audit permissions (audit:read, audit:search, audit:export, audit:admin)
  - Map permissions to roles (viewer, operator, admin)

---

## Phase 9: Testing (P2)

### 9.1 Unit Tests
- [x] Create `tests/unit/audit_service/__init__.py`
- [x] Create `tests/unit/audit_service/test_event_types.py` - 20+ tests
- [x] Create `tests/unit/audit_service/test_event_model.py` - 25+ tests
- [x] Create `tests/unit/audit_service/test_event_collector.py` - 20+ tests
- [x] Create `tests/unit/audit_service/test_event_enricher.py` - 15+ tests
- [x] Create `tests/unit/audit_service/test_event_router.py` - 20+ tests
- [x] Create `tests/unit/audit_service/test_audit_service.py` - 25+ tests
- [x] Create `tests/unit/audit_service/test_events_routes.py` - 20+ tests
- [x] Create `tests/unit/audit_service/test_search_routes.py` - 15+ tests
- [x] Create `tests/unit/audit_service/test_export_service.py` - 15+ tests
- [x] Create `tests/unit/audit_service/test_report_generators.py` - 20+ tests

### 9.2 Integration Tests
- [x] Create `tests/integration/audit_service/__init__.py`
- [x] Create `tests/integration/audit_service/test_audit_flow_e2e.py` - 15+ tests
- [x] Create `tests/integration/audit_service/test_websocket_stream.py` - 10+ tests
- [x] Create `tests/integration/audit_service/test_compliance_reports.py` - 10+ tests

### 9.3 Load Tests
- [x] Create `tests/load/audit_service/__init__.py`
- [x] Create `tests/load/audit_service/test_audit_throughput.py` - 10+ tests:
  - 10,000 events/sec ingestion
  - Query performance under load
  - Export large datasets

---

## Summary

| Phase | Tasks | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Core Audit Service | 8/8 | P0 | COMPLETE |
| Phase 2: Audit API | 5/5 | P0 | COMPLETE |
| Phase 3: Compliance Reporting | 5/5 | P0 | COMPLETE |
| Phase 4: Export Service | 3/3 | P0 | COMPLETE |
| Phase 5: Audit Middleware | 2/2 | P1 | COMPLETE |
| Phase 6: Metrics & Alerts | 3/3 | P1 | COMPLETE |
| Phase 7: Retention & Archival | 3/3 | P1 | COMPLETE |
| Phase 8: Integration | 3/3 | P1 | COMPLETE |
| Phase 9: Testing | 17/17 | P2 | COMPLETE |
| **TOTAL** | **49/49** | - | **COMPLETE** |
