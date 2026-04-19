# PRD: AGENT-FOUND-005 - GreenLang Citations & Evidence Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-FOUND-005 |
| **Agent ID** | GL-FOUND-X-005 |
| **Component** | Citations & Evidence Agent |
| **Category** | Foundations Agent |
| **Priority** | P1 - High (zero-hallucination compliance depends on this) |
| **Status** | Layer 1 Complete (~1,699 lines), Integration Gap-Fill Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS enforces a **zero-hallucination principle**: every calculation output,
emission factor, and compliance claim must be traceable to a verified, citable source.
The Citations & Evidence Agent provides the provenance backbone that makes this guarantee
auditable and legally defensible.

Without a production-grade citations service:
- Calculations cannot be traced to authoritative sources (DEFRA, EPA, IPCC, Ecoinvent)
- Regulatory compliance claims (CSRD, CBAM, EUDR, SB253) lack verifiable evidence
- Auditors cannot independently verify the provenance chain
- Annual emission factor updates cannot be versioned or supersession-tracked
- Evidence packages for audit submissions cannot be assembled programmatically

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/foundation/citations_agent.py` (1,699 lines)
- `CitationsEvidenceAgent` (BaseAgent subclass, AGENT_ID: GL-FOUND-X-005)
- `CitationType` enum: EMISSION_FACTOR, REGULATORY, METHODOLOGY, SCIENTIFIC, COMPANY_DATA, GUIDANCE, DATABASE
- `SourceAuthority` enum: DEFRA, EPA, ECOINVENT, IPCC, GHG_PROTOCOL, EXIOBASE, CLIMATIQ, EU_COMMISSION, SEC, EFRAG, CARB, ISO, GRI, SASB, CDP, INTERNAL, SUPPLIER, OTHER (18 values)
- `RegulatoryFramework` enum: CSRD, CBAM, EUDR, SB253, SB261, SEC_CLIMATE, TCFD, TNFD (8 values)
- `VerificationStatus` enum: VERIFIED, PENDING, EXPIRED, SUPERSEDED, UNVERIFIED, INVALID (6 values)
- `EvidenceType` enum: CALCULATION, DATA_POINT, METHODOLOGY, ASSUMPTION, VALIDATION, AUDIT_TRAIL (6 values)
- Pydantic models: CitationMetadata, Citation, EvidenceItem, EvidencePackage, MethodologyReference, RegulatoryRequirement, DataSourceAttribution, CitationsAgentInput, CitationsAgentOutput
- 13 action handlers: register_citation, lookup_citation, lookup_multiple, verify_citation, verify_sources, create_package, add_evidence, finalize_package, export_citations, query_citations, get_methodology, get_regulatory, check_validity
- SHA-256 content hashing for provenance
- BibTeX and JSON export
- Standard methodology initialization (GHG Protocol Corporate, Scope 3, ISO 14064-1)
- In-memory storage (no database persistence)

### 3.2 Layer 1 Tests
**File**: `tests/agents/foundation/test_citations_agent.py`
- Basic test coverage for Layer 1 foundation agent

### 3.3 Additional Files
- `greenlang/agents/citations.py` - Legacy citations module
- `tools/sdks/python/greenlang_sdk/core/citation.py` - SDK citation utilities
- `GreenLang Development/01-Core-Platform/agents/foundation/citations_agent.py` - Documentation copy

## 4. Identified Gaps

### Gap 1: No Integration Module in Main Codebase
No `greenlang/citations/` package providing a clean SDK for other services to
interact with citations using standard patterns (config, metrics, setup facade).

### Gap 2: No Prometheus Metrics (Standard Pattern)
No `greenlang/citations/metrics.py` following the standard GreenLang Prometheus pattern
used by other services (orchestrator, schema, normalizer, assumptions).

### Gap 3: No Service Setup Facade
No `configure_citations_service(app)` / `get_citations_service(app)` pattern matching
other GreenLang services.

### Gap 4: Foundation Agent Doesn't Delegate
Layer 1 agent has its own in-memory storage and doesn't delegate to a comprehensive
integration module that could provide persistent storage, caching, and shared state.

### Gap 5: No Standard REST API Router
No `greenlang/citations/api/router.py` with FastAPI endpoints following the standard
GreenLang API pattern used by all other services.

### Gap 6: No Standard Deployment Manifests
No K8s manifests in `deployment/kubernetes/citations-service/` following the standard
pattern.

### Gap 7: No Database Migration
No `V025__citations_service.sql` in the standard migration directory for persistent
citation storage, evidence packages, and verification audit trails.

### Gap 8: No Standard Monitoring
No dashboard/alerts in `deployment/monitoring/` following standard patterns.

### Gap 9: No CI/CD Pipeline
No `.github/workflows/citations-ci.yml` following the standard GreenLang CI pattern.

### Gap 10: No Operational Runbooks
No `docs/runbooks/` for citations service operations.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/citations/
  __init__.py           # Public API exports
  config.py             # CitationsConfig with GL_CITATIONS_ env prefix
  models.py             # Pydantic v2 models (re-export + enhance from foundation agent)
  registry.py           # CitationRegistry: create/get/update/delete/list citations
  evidence.py           # EvidenceManager: create/finalize/query evidence packages
  verification.py       # VerificationEngine: verify sources, DOI resolution, hash integrity
  provenance.py         # ProvenanceTracker: SHA-256 hash chain, audit trail
  export_import.py      # ExportImportManager: BibTeX, JSON, CSL export/import
  metrics.py            # 12 Prometheus metrics
  setup.py              # CitationsService facade, configure/get
  api/
    __init__.py
    router.py           # FastAPI router (20 endpoints)
```

### 5.2 Database Schema (V025)
```sql
CREATE SCHEMA citations_service;
-- citations (main registry with rich metadata)
-- citation_versions (hypertable - immutable version history)
-- evidence_packages (evidence package definitions)
-- evidence_items (individual evidence items)
-- citation_verifications (hypertable - verification audit trail)
-- regulatory_mappings (citation-to-regulation linkage)
-- methodology_references (standard methodology catalog)
-- data_source_attributions (data source tracking)
```

### 5.3 Prometheus Metrics (12)
| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_citations_operations_total` | Counter | Total operations by type, result |
| 2 | `gl_citations_operation_duration_seconds` | Histogram | Operation latency |
| 3 | `gl_citations_verifications_total` | Counter | Verifications by result |
| 4 | `gl_citations_verification_failures_total` | Counter | Verification failures by reason |
| 5 | `gl_citations_evidence_packages_total` | Counter | Evidence packages created |
| 6 | `gl_citations_evidence_items_total` | Counter | Evidence items added |
| 7 | `gl_citations_exports_total` | Counter | Exports by format |
| 8 | `gl_citations_total` | Gauge | Total registered citations |
| 9 | `gl_citations_packages_total` | Gauge | Total evidence packages |
| 10 | `gl_citations_cache_hits_total` | Counter | Citation cache hits |
| 11 | `gl_citations_cache_misses_total` | Counter | Citation cache misses |
| 12 | `gl_citations_provenance_chain_depth` | Histogram | Provenance chain depths |

### 5.4 API Endpoints (20)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/citations` | Register a new citation |
| GET | `/v1/citations` | List citations (with filters) |
| GET | `/v1/citations/{id}` | Get citation by ID |
| PUT | `/v1/citations/{id}` | Update citation |
| DELETE | `/v1/citations/{id}` | Delete citation |
| POST | `/v1/citations/{id}/verify` | Verify a citation |
| POST | `/v1/citations/verify-batch` | Batch verify citations |
| GET | `/v1/citations/{id}/versions` | Get version history |
| POST | `/v1/citations/export` | Export citations (BibTeX/JSON/CSL) |
| POST | `/v1/citations/import` | Import citations |
| POST | `/v1/evidence/packages` | Create evidence package |
| GET | `/v1/evidence/packages` | List evidence packages |
| GET | `/v1/evidence/packages/{id}` | Get evidence package |
| POST | `/v1/evidence/packages/{id}/items` | Add evidence item |
| POST | `/v1/evidence/packages/{id}/finalize` | Finalize package with hash |
| GET | `/v1/citations/methodologies` | List methodology references |
| GET | `/v1/citations/methodologies/{id}` | Get methodology by ID |
| GET | `/v1/citations/regulatory` | List regulatory requirements |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

### 5.5 Key Design Principles
1. **Zero-Hallucination**: Every data value must have a citation. No defaults without explicit source.
2. **Tamper-Evidence**: SHA-256 hash chains on all citations and evidence packages.
3. **Supersession Tracking**: When emission factors update (e.g., DEFRA 2025 -> 2026), old citations are marked SUPERSEDED with forward links.
4. **Multi-Format Export**: BibTeX for academic, JSON for API, CSL for reports.
5. **Regulatory Compliance**: Direct linkage between citations and regulatory framework requirements.
6. **Audit-Ready**: Evidence packages bundle all proof needed for CSRD/CBAM/EUDR audits.

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/citations/__init__.py` - Public API exports (50+ symbols)
2. Create `greenlang/citations/config.py` - CitationsConfig with GL_CITATIONS_ env prefix
3. Create `greenlang/citations/models.py` - Pydantic v2 models: Citation, EvidenceItem, EvidencePackage, VerificationRecord, etc.
4. Create `greenlang/citations/registry.py` - CitationRegistry wrapping foundation agent with persistent-ready interface
5. Create `greenlang/citations/evidence.py` - EvidenceManager for package creation, item management, finalization
6. Create `greenlang/citations/verification.py` - VerificationEngine for source verification, DOI resolution, hash integrity
7. Create `greenlang/citations/provenance.py` - ProvenanceTracker with SHA-256 hash chain
8. Create `greenlang/citations/export_import.py` - ExportImportManager for BibTeX, JSON, CSL
9. Create `greenlang/citations/metrics.py` - 12 Prometheus metrics
10. Create `greenlang/citations/api/router.py` - FastAPI router with 20 endpoints
11. Create `greenlang/citations/setup.py` - CitationsService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V025__citations_service.sql`
2. Create K8s manifests in `deployment/kubernetes/citations-service/` (7 files)
3. Create `deployment/monitoring/dashboards/citations-service.json`
4. Create `deployment/monitoring/alerts/citations-service-alerts.yaml`
5. Create `.github/workflows/citations-ci.yml`

### Phase 3: Tests (Test Engineer)
1-14. Create unit, integration, and load tests in `tests/*/citations_service/`

### Phase 4: Documentation (Tech Writer)
1-4. Create operational runbooks

## 7. Success Criteria
- Integration module provides clean SDK for all citation and evidence operations
- Foundation agent delegates to integration module when available
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V025 database migration for persistent citation storage
- 20 REST API endpoints operational
- 2+ operational runbooks
- 400+ new tests passing (unit + integration + load)
- Zero-hallucination guarantees maintained (every value traceable to a citation)
- Evidence packages are tamper-evident with SHA-256 hash chains
- BibTeX, JSON, and CSL export formats supported
- Regulatory framework mapping for CSRD, CBAM, EUDR, SB253

## 8. Integration Points

### 8.1 Upstream Dependencies
- **AGENT-FOUND-001 (Orchestrator)**: Citations are part of DAG execution provenance
- **AGENT-FOUND-003 (Unit Normalizer)**: Emission factors require citation backing
- **AGENT-FOUND-004 (Assumptions Registry)**: Assumptions reference citation IDs

### 8.2 Downstream Consumers
- **All calculation agents**: Must attach citation_ids to every output
- **Compliance reporting**: Evidence packages for audit submissions
- **Formula library**: Emission factors linked to source citations
- **API Gateway**: Citation endpoints exposed via Kong

### 8.3 Infrastructure Integration
- **PostgreSQL**: Persistent citation storage (V025 migration)
- **Redis**: Citation lookup caching
- **Prometheus**: 12 observability metrics
- **Grafana**: Citations service dashboard
- **Alertmanager**: 15 alert rules
- **K8s**: Standard deployment with HPA
