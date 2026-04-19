# PRD-SEC-005: Centralized Audit Logging Service

**Status:** APPROVED
**Version:** 1.0
**Created:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** SEC-001 (JWT Auth), SEC-002 (RBAC), SEC-003 (Encryption), INFRA-002 (PostgreSQL), INFRA-009 (Loki)

---

## 1. Overview

### 1.1 Purpose
Implement a centralized Audit Logging Service that consolidates all security-relevant events across GreenLang Climate OS into a unified, queryable, and compliant system. This service provides the foundation for SOC 2, ISO 27001, and GDPR compliance by ensuring all actions are recorded, searchable, and auditable.

### 1.2 Scope
- **In Scope:**
  - Unified audit service consolidating auth, RBAC, encryption, and agent events
  - REST API for querying, searching, and exporting audit logs
  - Real-time audit event streaming via WebSocket
  - Compliance reporting endpoints (SOC 2, ISO 27001, GDPR)
  - Grafana dashboard for audit monitoring
  - Prometheus alerts for audit anomalies
  - Audit event correlation and enrichment
  - Automated retention and archival policies
  - Integration with Loki for log aggregation
- **Out of Scope:**
  - SIEM integration beyond JSON format (future enhancement)
  - Third-party audit compliance tools
  - Blockchain-based audit immutability

### 1.3 Success Criteria
- All audit events flow through unified service with <10ms overhead
- API query latency <100ms for last 24 hours, <500ms for 30 days
- Real-time streaming delivers events within 100ms
- Compliance reports generated within 5 seconds
- 100% audit log completeness (no dropped events)
- Retention policies enforced automatically

---

## 2. Technical Requirements

### TR-001: Unified Audit Service Core
**Priority:** P0
**Description:** Create a centralized audit service that provides a single interface for all audit operations.

**Requirements:**
1. `greenlang/infrastructure/audit_service/` module:
   - `AuditServiceConfig` dataclass with environment-aware defaults
   - `UnifiedAuditEventType` enum consolidating all 70+ event types
   - `UnifiedAuditEvent` dataclass with correlation, enrichment, and classification
   - `AuditService` class orchestrating all audit operations
2. Event consolidation:
   - Accept events from auth_audit, rbac_audit, encryption_audit, sandbox_audit
   - Normalize event schemas to unified format
   - Add correlation IDs, tenant context, and classification
3. Event routing:
   - Write to TimescaleDB `audit.audit_log` hypertable
   - Emit to Loki via structured JSON logging
   - Publish to Redis pub/sub for real-time streaming
4. Event enrichment:
   - Add geo-location from IP (MaxMind GeoIP2)
   - Add user agent parsing
   - Add resource context (agent, pack, tenant)

**Acceptance Criteria:**
- [ ] All event types consolidated into unified enum
- [ ] Events written to DB, Loki, and Redis in <10ms
- [ ] Correlation IDs link related events
- [ ] No audit events lost under 10,000 events/second load

### TR-002: Audit API Endpoints
**Priority:** P0
**Description:** REST API for querying, searching, filtering, and exporting audit logs.

**Requirements:**
1. Query endpoints:
   - `GET /api/v1/audit/events` - List events with filtering, pagination
   - `GET /api/v1/audit/events/{event_id}` - Get single event by ID
   - `POST /api/v1/audit/search` - Advanced search with LogQL-like syntax
   - `GET /api/v1/audit/events/stream` - WebSocket real-time event stream
2. Filter parameters:
   - Time range (start, end), event types, severity levels
   - Tenant ID, user ID, resource type, resource ID
   - Correlation ID, client IP, success/failure
3. Aggregation endpoints:
   - `GET /api/v1/audit/stats` - Event statistics over time
   - `GET /api/v1/audit/timeline` - Activity timeline for user/resource
   - `GET /api/v1/audit/hotspots` - Most active users/resources/IPs
4. Export endpoints:
   - `POST /api/v1/audit/export` - Export to CSV, JSON, or Parquet
   - `GET /api/v1/audit/export/{job_id}` - Check export status
   - `GET /api/v1/audit/export/{job_id}/download` - Download export file

**Acceptance Criteria:**
- [ ] All query endpoints return within 100ms for 24h window
- [ ] Pagination supports cursor-based and offset-based
- [ ] Export handles up to 10 million records
- [ ] WebSocket delivers events within 100ms

### TR-003: Compliance Reporting
**Priority:** P0
**Description:** Generate compliance reports for SOC 2, ISO 27001, and GDPR requirements.

**Requirements:**
1. Report types:
   - `POST /api/v1/audit/reports/soc2` - SOC 2 audit report
   - `POST /api/v1/audit/reports/iso27001` - ISO 27001 audit report
   - `POST /api/v1/audit/reports/gdpr` - GDPR data access report
   - `POST /api/v1/audit/reports/custom` - Custom report builder
2. Report content:
   - Executive summary with key metrics
   - Detailed event listing by control category
   - Anomaly detection highlights
   - Gap analysis (missing expected events)
3. Report formats:
   - PDF with charts and tables
   - CSV for data import
   - JSON for programmatic access
4. Scheduled reports:
   - Cron-based report generation
   - Email delivery to stakeholders
   - Report versioning and history

**Acceptance Criteria:**
- [ ] Reports generated within 5 seconds for 30-day period
- [ ] SOC 2 report maps to CC6/CC7/CC8 controls
- [ ] ISO 27001 report maps to A.9/A.12 controls
- [ ] GDPR report includes all data subject access events

### TR-004: Audit Metrics & Alerting
**Priority:** P1
**Description:** Prometheus metrics and alerts for audit service health and anomaly detection.

**Requirements:**
1. Prometheus metrics:
   - `gl_audit_events_total` Counter (event_type, severity, success)
   - `gl_audit_event_latency_seconds` Histogram (stage: receive, enrich, write)
   - `gl_audit_events_queued` Gauge (queue depth)
   - `gl_audit_db_write_failures` Counter (error_type)
   - `gl_audit_stream_connections` Gauge (active WebSocket connections)
   - `gl_audit_export_jobs_total` Counter (status)
   - `gl_audit_report_generation_seconds` Histogram (report_type)
2. Alert rules:
   - HighAuditEventRate - >1000 events/sec sustained
   - AuditWriteFailures - Any DB write failures
   - AuditEventBacklog - Queue depth >1000
   - MissingAuditEvents - Expected events not received
   - SuspiciousActivityDetected - Anomaly detection triggered
   - AuditStreamDisconnected - All stream consumers disconnected
   - ComplianceGapDetected - Required audit not captured

**Acceptance Criteria:**
- [ ] All 7 metrics exported to Prometheus
- [ ] All 7 alerts configured with appropriate thresholds
- [ ] PagerDuty integration for critical alerts

### TR-005: Audit Dashboard
**Priority:** P1
**Description:** Grafana dashboard for visualizing audit activity and trends.

**Requirements:**
1. Dashboard panels (16+):
   - Event rate over time (by type, severity)
   - Top users by activity
   - Top resources by access
   - Geographic distribution of events
   - Authentication success/failure ratio
   - Authorization denial heatmap
   - Sensitive data access timeline
   - Anomaly detection alerts
   - Compliance score gauge
   - Active session count
   - API request latency by endpoint
   - Data export volume
   - Failed operation breakdown
   - Event queue depth
   - Report generation status
   - System health summary
2. Filter variables:
   - Time range, tenant, user, event type, severity

**Acceptance Criteria:**
- [ ] Dashboard loads within 3 seconds
- [ ] All panels interactive with drill-down
- [ ] Template variables for multi-tenant filtering

### TR-006: Audit Middleware
**Priority:** P1
**Description:** Automatic audit capture for all API requests.

**Requirements:**
1. FastAPI middleware:
   - Capture request/response metadata (not body content)
   - Record timing, status code, headers
   - Link to authenticated user context
   - Classify by sensitivity (public, internal, sensitive, critical)
2. Selective body capture:
   - Opt-in for specific endpoints
   - Hash or mask sensitive fields
   - Size limits for captured content
3. Exclusions:
   - Health check endpoints
   - Metrics endpoints
   - Static asset requests

**Acceptance Criteria:**
- [ ] All API requests automatically audited
- [ ] <2ms overhead per request
- [ ] Sensitive data never logged in plaintext

### TR-007: Retention & Archival
**Priority:** P1
**Description:** Automated audit log retention and archival policies.

**Requirements:**
1. Retention tiers:
   - Hot: Last 30 days in PostgreSQL (fast query)
   - Warm: 30-90 days in compressed partitions
   - Cold: 90-365 days in S3 Parquet
   - Archive: 1-7 years in S3 Glacier
2. Compliance retention:
   - Financial data: 7 years
   - Security events: 3 years
   - User activity: 1 year (or per GDPR)
3. Automated processes:
   - Daily partition rotation
   - Weekly compression
   - Monthly archival to S3
   - Yearly glacier migration
4. Restoration:
   - API endpoint to restore archived data
   - Temporary table for restored queries

**Acceptance Criteria:**
- [ ] Retention policies enforced automatically
- [ ] Storage costs reduced 80% via tiering
- [ ] Archived data restorable within 4 hours

---

## 3. Architecture

### 3.1 Unified Audit Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GreenLang Application Layer                         │
└─────────────────────────────────────────────────────────────────────────────┘
                │                    │                    │
                ▼                    ▼                    ▼
    ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
    │   Auth Service    │  │   RBAC Service    │  │  Encryption Svc   │
    │  (auth_audit.py)  │  │ (rbac_audit.py)   │  │(encryption_audit) │
    └─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     ▼
          ┌──────────────────────────────────────────────────────────┐
          │                   Unified Audit Service                   │
          │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
          │  │   Event     │  │   Event     │  │    Event        │  │
          │  │  Collector  │→ │  Enricher   │→ │   Router        │  │
          │  └─────────────┘  └─────────────┘  └─────────────────┘  │
          └──────────────────────────┬───────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
  │  PostgreSQL  │          │    Redis     │          │    Loki      │
  │ (TimescaleDB)│          │   Pub/Sub    │          │  (Grafana)   │
  │  Hypertable  │          │  Streaming   │          │   Logging    │
  └──────────────┘          └──────────────┘          └──────────────┘
        │                            │
        ▼                            ▼
  ┌──────────────┐          ┌──────────────┐
  │  Audit API   │          │  WebSocket   │
  │  (REST)      │          │   Stream     │
  └──────────────┘          └──────────────┘
```

### 3.2 Event Schema

```python
@dataclass
class UnifiedAuditEvent:
    # Identity
    event_id: str              # UUID v7 (time-ordered)
    correlation_id: str        # Request correlation ID

    # Classification
    event_type: str            # e.g., "auth.login_success"
    event_category: str        # auth, rbac, encryption, data, system
    severity: str              # debug, info, warning, error, critical

    # Actor
    user_id: Optional[str]
    username: Optional[str]
    tenant_id: str
    session_id: Optional[str]
    client_ip: str
    user_agent: Optional[str]
    geo_location: Optional[dict]  # {country, city, lat, lon}

    # Resource
    resource_type: Optional[str]  # agent, pack, emission, report
    resource_id: Optional[str]
    resource_name: Optional[str]

    # Action
    action: str                # create, read, update, delete, execute
    result: str                # success, failure, denied
    error_message: Optional[str]

    # Context
    request_path: Optional[str]
    request_method: Optional[str]
    response_status: Optional[int]
    duration_ms: Optional[float]

    # Metadata
    metadata: Dict[str, Any]   # Event-specific details
    tags: List[str]            # Searchable tags

    # Timestamps
    occurred_at: datetime      # When event happened
    recorded_at: datetime      # When event was logged
```

---

## 4. Implementation Phases

### Phase 1: Core Audit Service (P0)
- Create `greenlang/infrastructure/audit_service/` module
- Implement `UnifiedAuditService` with event consolidation
- Database integration with existing `audit.audit_log`
- Redis pub/sub for real-time streaming
- Event enrichment (geo-IP, user agent)

### Phase 2: Audit API (P0)
- Query endpoints with filtering/pagination
- Search endpoint with LogQL-like syntax
- WebSocket real-time streaming
- Export endpoints (CSV, JSON, Parquet)

### Phase 3: Compliance Reporting (P0)
- SOC 2 report generator
- ISO 27001 report generator
- GDPR data access report
- Custom report builder

### Phase 4: Audit Middleware (P1)
- FastAPI middleware for automatic API auditing
- Request/response metadata capture
- Sensitivity classification
- Performance optimization

### Phase 5: Monitoring (P1)
- Grafana dashboard (16+ panels)
- Prometheus metrics (7 metrics)
- Alert rules (7 alerts)
- PagerDuty integration

### Phase 6: Retention & Archival (P1)
- Tiered storage implementation
- Automated archival CronJobs
- S3 Parquet writer
- Restoration API

### Phase 7: Testing (P2)
- Unit tests for all services
- Integration tests with real DB
- Load tests (10,000 events/sec)
- Compliance verification tests

---

## 5. Security Considerations

### 5.1 Data Protection
- PII redaction for all sensitive fields
- Encryption at rest via SEC-003
- TLS 1.3 for all transport via SEC-004
- No plaintext secrets in audit logs

### 5.2 Access Control
- RBAC permissions: `audit:read`, `audit:export`, `audit:admin`
- Tenant isolation via RLS
- Admin-only access to cross-tenant queries
- Rate limiting on export endpoints

### 5.3 Integrity
- Append-only audit tables
- No UPDATE/DELETE on audit records
- Cryptographic hash chain (future)
- Tamper detection alerts

---

## 6. Compliance Mapping

| Requirement | SOC 2 | ISO 27001 | GDPR | PCI DSS |
|-------------|-------|-----------|------|---------|
| Audit log capture | CC7.2 | A.12.4.1 | Art. 30 | 10.1 |
| Secure log storage | CC6.1 | A.12.4.2 | Art. 32 | 10.5 |
| Log retention | CC7.3 | A.12.4.1 | Art. 17 | 10.7 |
| Access monitoring | CC7.2 | A.12.4.3 | Art. 32 | 10.6 |
| Anomaly detection | CC7.4 | A.12.4.4 | N/A | 10.6.1 |

---

## 7. Deliverables Summary

| Component | Files | Priority |
|-----------|-------|----------|
| Core Audit Service | 8 | P0 |
| Audit API | 5 | P0 |
| Compliance Reporting | 4 | P0 |
| Audit Middleware | 2 | P1 |
| Monitoring (Dashboard + Alerts) | 2 | P1 |
| Retention & Archival | 3 | P1 |
| Testing | 8 | P2 |
| **TOTAL** | **~32** | - |

---

## 8. Appendix

### A. Event Type Categories

```
auth.*          - Authentication events (login, logout, token)
rbac.*          - Authorization events (role, permission, assignment)
encryption.*    - Encryption events (encrypt, decrypt, key rotation)
data.*          - Data access events (read, write, delete, export)
agent.*         - Agent execution events (start, complete, fail)
system.*        - System events (config change, service start/stop)
api.*           - API events (request, response, error)
compliance.*    - Compliance events (report, violation, remediation)
```

### B. Query Examples

```bash
# Get all failed auth events in last hour
GET /api/v1/audit/events?event_type=auth.*&result=failure&since=1h

# Search for specific user activity
POST /api/v1/audit/search
{"query": "user_id:u-123 AND action:delete AND resource_type:emission"}

# Export compliance report
POST /api/v1/audit/reports/soc2
{"period": "2026-Q1", "format": "pdf"}
```

### C. Performance Targets

| Operation | Target | Max |
|-----------|--------|-----|
| Event ingest | <5ms | 10ms |
| Query (24h) | <50ms | 100ms |
| Query (30d) | <200ms | 500ms |
| Export (1M rows) | <60s | 120s |
| Report generation | <3s | 5s |
| WebSocket latency | <50ms | 100ms |
