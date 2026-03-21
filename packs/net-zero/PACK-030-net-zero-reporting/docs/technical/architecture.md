# PACK-030: Architecture Overview

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Architecture](#component-architecture)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Integration Architecture](#integration-architecture)
5. [Security Architecture](#security-architecture)
6. [Deployment Architecture](#deployment-architecture)
7. [Caching Architecture](#caching-architecture)
8. [Error Handling Architecture](#error-handling-architecture)

---

## 1. System Architecture

### High-Level System Context

```
    External Systems                    PACK-030                        Outputs
    +------------------+     +----------------------------+     +-------------------+
    |                  |     |                            |     |                   |
    | Prerequisite     |     |  +--------------------+   |     | Framework Reports |
    | Packs            +---->+  | Data Aggregation   |   +---->+ (7 frameworks)    |
    | PACK-021/022/    |     |  | Layer              |   |     |                   |
    | 028/029          |     |  +--------+-----------+   |     | Multi-Format      |
    |                  |     |           |               |     | (PDF/HTML/Excel/  |
    +------------------+     |  +--------v-----------+   |     |  JSON/XBRL/iXBRL) |
    |                  |     |  | Processing         |   |     |                   |
    | GreenLang        +---->+  | Layer              |   +---->+ Assurance         |
    | Applications     |     |  | (Engines)          |   |     | Evidence Bundles  |
    | GL-SBTi/CDP/     |     |  +--------+-----------+   |     |                   |
    | TCFD/GHG         |     |           |               |     | Interactive       |
    |                  |     |  +--------v-----------+   +---->+ Dashboards        |
    +------------------+     |  | Presentation       |   |     |                   |
    |                  |     |  | Layer              |   |     | API Responses     |
    | External         +---->+  | (Templates/Render) |   +---->+ (JSON)            |
    | Services         |     |  +--------------------+   |     |                   |
    | (DeepL/XBRL/S3)  |     |                            |     +-------------------+
    +------------------+     +----------------------------+
```

### Architectural Principles

| Principle | Implementation |
|-----------|---------------|
| **Zero-Hallucination** | No LLM in calculation paths; deterministic Decimal arithmetic for all metrics; LLMs used only for narrative generation with human review required |
| **Single Source of Truth** | Data aggregated once and shared across all 7 framework workflows; no redundant data paths |
| **Cryptographic Provenance** | SHA-256 hashes on every calculated metric and report output; immutable audit trail |
| **Async-First** | All I/O operations use async/await; parallel data fetching and framework report generation |
| **Multi-Tenant Isolation** | Row-level security (RLS) policies on all 15 tables; organization_id-based isolation |
| **Graceful Degradation** | Circuit breaker pattern for upstream dependencies; reports generated with available data |
| **Schema-Driven Validation** | All outputs validated against official framework JSON schemas |
| **Separation of Concerns** | Clear boundaries between aggregation, processing, and presentation layers |

---

## 2. Component Architecture

### Layer Diagram

```
+------------------------------------------------------------------------+
|                          API Layer (FastAPI)                             |
|  /api/v1/reports  /api/v1/workflows  /api/v1/dashboards  /health       |
+--------+------------------+--------------------+-----------------------+
         |                  |                    |
+--------v------------------v--------------------v-----------------------+
|                       Workflow Layer                                     |
|  SBTi | CDP | TCFD | GRI | ISSB | SEC | CSRD | Multi-Framework        |
+--------+------------------+--------------------+-----------------------+
         |                  |                    |
+--------v------------------v--------------------v-----------------------+
|                       Engine Layer (10 Engines)                         |
|  DataAgg | Narrative | FrameworkMap | XBRL | Dashboard | Assurance     |
|  ReportCompile | Validation | Translation | FormatRendering            |
+--------+------------------+--------------------+-----------------------+
         |                  |                    |
+--------v------------------v--------------------v-----------------------+
|                       Template Layer (15 Templates)                     |
|  SBTi | CDP Gov | CDP Emit | TCFD (4) | GRI | ISSB | SEC | CSRD      |
|  Investor | Regulator | Customer | Assurance Evidence                  |
+--------+------------------+--------------------+-----------------------+
         |                  |                    |
+--------v------------------v--------------------v-----------------------+
|                       Integration Layer (12 Integrations)               |
|  PACK-021 | PACK-022 | PACK-028 | PACK-029                           |
|  GL-SBTi | GL-CDP | GL-TCFD | GL-GHG                                  |
|  XBRL Taxonomy | Translation | Orchestrator | Health Check             |
+--------+------------------+--------------------+-----------------------+
         |                  |                    |
+--------v------------------v--------------------v-----------------------+
|                       Data Layer                                        |
|  PostgreSQL 16+ (15 tables, 5 views, 350+ indexes, 30 RLS policies)   |
|  Redis 7+ (caching, sessions)                                          |
|  S3 (report archive)                                                    |
+------------------------------------------------------------------------+
```

### Engine Interaction Pattern

```
                    DataAggregationEngine
                           |
                    +------+------+
                    |             |
            NarrativeGen    FrameworkMapping
            Engine              Engine
                    |             |
                    +------+------+
                           |
                   ReportCompilationEngine
                           |
                    +------+------+------+
                    |      |      |      |
               Validation  XBRL  Translation  Dashboard
               Engine     Engine  Engine       Engine
                    |      |      |      |
                    +------+------+------+
                           |
                   FormatRenderingEngine
                           |
                   AssurancePackagingEngine
```

### Engine Responsibilities

| Engine | Responsibility | Dependencies |
|--------|---------------|--------------|
| Data Aggregation | Collect, reconcile, track lineage | Integrations (PACK + APP) |
| Narrative Generation | Draft narratives, manage citations | Data Aggregation output |
| Framework Mapping | Translate metrics between frameworks | Framework schemas |
| XBRL Tagging | Generate XBRL/iXBRL documents | XBRL Taxonomy Integration |
| Dashboard Generation | Create HTML5 interactive dashboards | Data Aggregation output |
| Assurance Packaging | Bundle evidence for auditors | All engine outputs |
| Report Compilation | Assemble sections into reports | Templates, Narratives |
| Validation | Validate against schemas | Framework schemas |
| Translation | Multi-language narrative support | Translation Integration |
| Format Rendering | Render PDF/HTML/Excel/JSON/XBRL | Report Compilation output |

---

## 3. Data Flow Architecture

### Multi-Framework Report Generation Flow

```
Step 1: Data Aggregation (parallel)
    PACK-021 ----+
    PACK-022 ----+----> DataAggregationEngine ----> Aggregated Dataset
    PACK-028 ----+                                      |
    PACK-029 ----+                                      |
    GL-SBTi  ----+                                      |
    GL-CDP   ----+                                      |
    GL-TCFD  ----+                                      |
    GL-GHG   ----+                                      |
                                                        v
Step 2: Shared Processing                     FrameworkMappingEngine
    Aggregated Dataset ----> NarrativeGenEngine      |
                                    |                |
                                    v                v
Step 3: Framework Workflows (parallel)
    +-> SBTi Workflow -----> SBTi Report
    +-> CDP Workflow  -----> CDP Questionnaire
    +-> TCFD Workflow -----> TCFD Report
    +-> GRI Workflow  -----> GRI 305 Disclosure
    +-> ISSB Workflow -----> IFRS S2 Report
    +-> SEC Workflow  -----> SEC 10-K Section
    +-> CSRD Workflow -----> ESRS E1 Disclosure

Step 4: Validation
    All Reports ----> ValidationEngine ----> Consistency Report

Step 5: Rendering (parallel per format)
    Reports ----> FormatRenderingEngine ----> PDF/HTML/Excel/JSON/XBRL/iXBRL

Step 6: Assurance
    All Outputs ----> AssurancePackagingEngine ----> Evidence Bundle (ZIP)
```

### Data Lineage Tracking

Every metric in every report maintains a complete lineage chain:

```
Source Transaction (GL-GHG-APP)
    -> Activity Data Record (fuel consumption)
        -> Emission Factor Application (DEFRA 2025)
            -> Scope 1 Calculation (stationary combustion)
                -> Total Scope 1 (aggregation)
                    -> SBTi Progress Table (framework mapping)
                        -> PDF Report (rendering)
                            -> SHA-256 Hash (provenance)
```

---

## 4. Integration Architecture

### Integration Patterns

| Pattern | Usage |
|---------|-------|
| **REST API Client** | Pack integrations (PACK-021/022/028/029) |
| **GraphQL Client** | Application integrations (GL-SBTi/CDP/TCFD/GHG) |
| **Database Direct** | Read-only views for co-located data |
| **Message Queue** | Async report generation notifications |
| **Webhook** | Deadline notifications, completion callbacks |
| **File Transfer** | S3 report archive, evidence bundle storage |

### Circuit Breaker Configuration

```
                    +-------------------+
                    |   Integration     |
                    |   Client          |
                    +--------+----------+
                             |
                    +--------v----------+
                    |  Circuit Breaker  |
                    |  State Machine    |
                    +--------+----------+
                             |
              +------+-------+-------+------+
              |      |               |      |
         +----v-+ +--v---+     +----v-+ +--v---+
         |CLOSED| |OPEN  |     |HALF  | |CLOSED|
         +------+ +------+     |OPEN  | +------+
                                +------+

   CLOSED: Normal operation, requests pass through
   OPEN: Fail fast, return cached/default data (after 5 consecutive failures)
   HALF-OPEN: Allow 1 test request after 60s cooldown
```

### Retry Strategy

```python
RetryConfig(
    max_retries=3,
    initial_delay_ms=1000,
    max_delay_ms=30000,
    backoff_multiplier=2.0,
    retry_on=[ConnectionError, TimeoutError],
    no_retry_on=[AuthenticationError, ValidationError],
)
```

---

## 5. Security Architecture

### Authentication Flow

```
Client                    API Gateway              PACK-030
  |                           |                       |
  |--- POST /auth/token ----->|                       |
  |                           |--- Validate creds --->|
  |                           |<-- JWT (RS256) -------|
  |<-- JWT Token ------------|                       |
  |                           |                       |
  |--- GET /api/v1/reports -->|                       |
  |   Authorization: Bearer   |--- Verify JWT ------->|
  |                           |--- Check RBAC ------->|
  |                           |--- Apply RLS -------->|
  |<-- Report Data -----------|<-- Filtered Data -----|
```

### Multi-Tenant Isolation

```sql
-- Every query is automatically filtered by organization_id
SET app.current_organization_id = 'org-uuid';

-- RLS policy ensures data isolation
CREATE POLICY nz_reports_isolation ON gl_nz_reports
    USING (organization_id = current_setting('app.current_organization_id')::UUID);
```

### Data Encryption

| Layer | Method | Key Management |
|-------|--------|---------------|
| At Rest | AES-256-GCM | Vault-managed keys |
| In Transit | TLS 1.3 | Certificate rotation |
| Report Files | AES-256 | Per-report keys |
| Provenance Hashes | SHA-256 | Immutable |

---

## 6. Deployment Architecture

### Kubernetes Deployment

```
                        Load Balancer (Kong)
                              |
                    +---------+---------+
                    |                   |
              +-----v------+    +------v-----+
              | PACK-030   |    | PACK-030   |
              | Pod 1      |    | Pod 2      |
              | (FastAPI)  |    | (FastAPI)  |
              +-----+------+    +------+-----+
                    |                   |
              +-----v-------------------v-----+
              |         Redis Cluster          |
              |   (3 nodes, caching)           |
              +-----+-------------------+-----+
                    |                   |
              +-----v-------------------v-----+
              |       PostgreSQL 16+           |
              |   (TimescaleDB, pgvector)      |
              |   (Primary + 2 Replicas)       |
              +-------------------------------+
```

### Container Specifications

```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"

replicas: 2

readinessProbe:
  httpGet:
    path: /health
    port: 8030
  initialDelaySeconds: 10
  periodSeconds: 5

livenessProbe:
  httpGet:
    path: /health
    port: 8030
  initialDelaySeconds: 30
  periodSeconds: 10
```

---

## 7. Caching Architecture

### Cache Layers

| Layer | Cache | TTL | Purpose |
|-------|-------|-----|---------|
| L1 | In-memory (LRU) | 5 min | Hot data, framework schemas |
| L2 | Redis | 1 hour | Report data, aggregation results |
| L3 | PostgreSQL | Persistent | Materialized views, reference data |
| L4 | S3 | Permanent | Generated reports, evidence bundles |

### Cache Key Strategy

```
pack030:{organization_id}:{framework}:{reporting_period}:{data_type}

Examples:
pack030:org-uuid:SBTi:2025:report_data
pack030:org-uuid:all:2025:aggregated_data
pack030:global:SEC:taxonomy:2024
```

### Cache Invalidation

- **Time-based**: TTL expiry (default 1 hour for report data)
- **Event-based**: Source data changes trigger cache invalidation
- **Manual**: API endpoint for forced cache refresh

---

## 8. Error Handling Architecture

### Error Classification

| Category | Behavior | Example |
|----------|----------|---------|
| **Recoverable** | Retry with backoff | Network timeout, database lock |
| **Degradable** | Continue with partial data | One pack unavailable |
| **Fatal** | Fail with detailed error | Database down, auth failure |
| **Validation** | Return error details | Invalid input, schema mismatch |

### Error Propagation

```
Integration Layer    Engine Layer    Workflow Layer    API Layer
     |                   |                |              |
     |-- Error --------->|                |              |
     |                   |-- Classify --->|              |
     |                   |                |-- Format --->|
     |                   |                |              |-- Response
     |                   |                |              |   (400/500)
     |                   |                |              |
     |                   |-- Log -------->|              |
     |                   |    (Loki)      |              |
     |                   |                |              |
     |                   |-- Trace ------>|              |
     |                   |    (Tempo)     |              |
```

### Observability Integration

| Signal | System | Usage |
|--------|--------|-------|
| Metrics | Prometheus/Thanos | Request rates, error rates, latency |
| Logs | Loki | Structured error logs, audit trail |
| Traces | OpenTelemetry/Tempo | Request tracing across services |
| Alerts | Alertmanager | Deadline alerts, error rate alerts |
| Dashboards | Grafana | Real-time monitoring |

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Language | Python | 3.11+ | Core implementation |
| API Framework | FastAPI | 0.110+ | REST API |
| Database | PostgreSQL | 16+ | Data persistence |
| Time-Series | TimescaleDB | 2.14+ | Temporal data |
| Caching | Redis | 7+ | Performance caching |
| Validation | Pydantic | 2.5+ | Type-safe models |
| Async | asyncio | stdlib | Non-blocking I/O |
| HTTP Client | httpx | 0.26+ | Async HTTP |
| PDF Rendering | WeasyPrint | 60+ | PDF generation |
| Excel | openpyxl | 3.1+ | Excel export |
| Charts | Plotly | 5.18+ | Interactive charts |
| Templates | Jinja2 | 3.1+ | Template engine |
| XML | lxml | 5.1+ | XBRL generation |
| Testing | pytest | 8.0+ | Test framework |
| Container | Docker | 25.0+ | Containerization |
| Orchestration | Kubernetes | 1.28+ | Deployment |

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
