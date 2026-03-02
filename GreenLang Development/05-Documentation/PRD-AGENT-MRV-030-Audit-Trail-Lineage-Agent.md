# PRD: AGENT-MRV-030 — Audit Trail & Lineage Agent

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-MRV-030 |
| **Agent ID** | GL-MRV-X-042 |
| **Component** | AGENT-MRV-030 |
| **Category** | Cross-Cutting MRV Agent |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Author** | GreenLang Platform Team |
| **Date** | 2026-03-01 |
| **Ralphy Task ID** | AGENT-MRV-030 |

---

## 1. Overview

### 1.1 Purpose

The Audit Trail & Lineage Agent (GL-MRV-X-042) provides immutable, tamper-evident audit trails and end-to-end calculation lineage for all MRV (Measurement, Reporting, Verification) emissions calculations across Scope 1, 2, and 3. It serves as the cross-cutting backbone that ties every emission figure back to its source data, emission factors, calculation methodology, and regulatory framework — enabling third-party auditors, regulators, and internal assurance teams to verify any reported number.

### 1.2 Problem Statement

Climate disclosures under CSRD, SB 253, CBAM, and ISO 14064 require **verifiable audit trails** proving that every reported emission figure is:
- Traceable to authoritative source data and emission factors
- Calculated using documented, approved methodologies
- Free from unauthorized modification (tamper-evident)
- Reproducible by independent auditors
- Compliant with applicable regulatory frameworks

Without a unified MRV-specific audit layer, audit evidence is fragmented across 29+ individual calculation agents, making verification time-consuming and error-prone.

### 1.3 Cross-Cutting Role

| Aspect | Category-Specific Agents (001-029) | Audit Trail & Lineage Agent (030) |
|--------|-------------------------------------|-----------------------------------|
| Scope | Single scope/category calculation | All scopes and categories |
| Audit | Per-agent provenance hash | Unified audit trail with hash chains |
| Lineage | Input → Output for one agent | Source data → EF → Methodology → Emissions → Report |
| Evidence | Calculation-level metadata | Audit-ready evidence packages for verification |
| Compliance | Agent-specific checks | Cross-framework compliance traceability |
| Change Tracking | None | Full recalculation change detection and versioning |

### 1.4 Key Capabilities

1. **Immutable Audit Event Recording** — SHA-256 chain-hashed events with genesis anchoring
2. **MRV Calculation Lineage Graph** — DAG tracking from activity data through emission factors to reported figures
3. **Evidence Package Generation** — Bundled audit packages for third-party verification (CSRD, SB 253, ISO 14064)
4. **Compliance Traceability** — Map every data point to 9 regulatory framework requirements
5. **Change Detection & Versioning** — Track recalculations when EFs, methodologies, or source data change
6. **Cross-Scope Reconciliation Audit** — Verify Scope 1+2+3 totals match consolidated reporting
7. **Digital Signature Support** — Ed25519/RSA/ECDSA signing for audit packages
8. **Regulatory Framework Mapping** — Direct linkage to ESRS E1, GHG Protocol, ISO 14064 disclosure requirements
9. **Data Quality Lineage** — Track DQ scores through the calculation chain
10. **Batch Audit Trail Processing** — Bulk audit trail generation for historical calculations

### 1.5 Supported Audit Event Types

| Event Type | Description | Source Agents |
|------------|-------------|---------------|
| `calculation_initiated` | MRV calculation pipeline started | All MRV 001-029 |
| `activity_data_ingested` | Activity data loaded and validated | DATA 001-007 |
| `emission_factor_resolved` | EF selected from authoritative source | MRV 001-029 |
| `methodology_applied` | Calculation method selected and applied | MRV 001-029 |
| `uncertainty_quantified` | Uncertainty range computed | MRV 001-029 |
| `allocation_performed` | Emissions allocated to entity/product | MRV 014-028 |
| `double_counting_checked` | DC prevention rules applied | SCM 029 |
| `compliance_validated` | Framework compliance verified | All MRV |
| `aggregation_completed` | Cross-scope/category rollup | Consolidation |
| `recalculation_triggered` | Historical recalculation due to EF/data change | All MRV |
| `evidence_packaged` | Audit evidence bundle created | This agent |
| `audit_sealed` | Final seal with digital signature | This agent |

### 1.6 Supported Lineage Depths

| Lineage Level | Description | Example |
|---------------|-------------|---------|
| L1 — Source | Raw data origin | "SAP S/4HANA invoice #12345" |
| L2 — Ingestion | Data intake processing | "PDF Extractor → Excel Normalizer" |
| L3 — Validation | Quality checks applied | "Data Quality Profiler: DQ score 0.92" |
| L4 — Factor | Emission factor resolution | "DEFRA 2024 v1.2, natural gas, 0.18316 kgCO2e/kWh" |
| L5 — Calculation | Methodology application | "GHG Protocol Scope 1, Tier 2, mass balance" |
| L6 — Allocation | Attribution method | "Revenue-based allocation, 34.7% share" |
| L7 — Aggregation | Rollup and consolidation | "Operational control, entity ABC Corp" |
| L8 — Reporting | Disclosure formatting | "CSRD ESRS E1-6, Scope 1 total" |

---

## 2. Regulatory Requirements

### 2.1 GHG Protocol References

| Standard | Section | Requirement |
|----------|---------|-------------|
| Corporate Standard | Ch. 9 | "Companies shall document and archive sufficient information to enable verification" |
| Scope 3 Standard | Ch. 8 | "Complete record of data sources, assumptions, and calculation approaches" |
| Corporate Standard | Ch. 10 | "Accounting and reporting of GHG emissions shall follow: relevance, completeness, consistency, transparency, accuracy" |
| Scope 2 Guidance | Ch. 7 | "Companies should maintain records of instruments, certificates, and contracts" |

### 2.2 Regulatory Framework Requirements

| Framework | Article/Section | Audit Trail Requirement |
|-----------|----------------|------------------------|
| CSRD / ESRS E1 | ESRS E1-6 | Full data lineage for Scope 1, 2, 3 GHG emissions with methodology disclosure |
| CSRD / ESRS E1 | ESRS 2 BP-2 | "Policies, actions, targets, and metrics" with basis for preparation |
| ISO 14064-1:2018 | Clause 9 | "Documentation and records... sufficient to enable verification" |
| ISO 14064-3:2019 | Clause 6 | Verification requirements including evidence examination |
| California SB 253 | §38532 | Independent third-party assurance with limited/reasonable assurance levels |
| EU CBAM | Art. 35 | Verified emission data with methodology documentation |
| CDP | C6-C7 | Detailed methodology and data source disclosure |
| TCFD | Metrics & Targets | Scope 1, 2, 3 with methodologies and assumptions disclosed |
| SBTi | Criteria 13 | Base year emissions with recalculation policy documentation |

### 2.3 Compliance Frameworks Supported

| # | Framework | Version | Audit Trail Scope |
|---|-----------|---------|-------------------|
| 1 | GHG Protocol Corporate Standard | 2004 (rev 2015) | Calculation documentation, EF sources |
| 2 | GHG Protocol Scope 3 Standard | 2011 (rev 2013) | Category mapping, data quality, methods |
| 3 | ISO 14064-1:2018 | 2018 | Quantification records, verification support |
| 4 | ISO 14064-3:2019 | 2019 | Verification evidence, assertion validation |
| 5 | CSRD / ESRS E1 | 2023 | Data point traceability, basis for preparation |
| 6 | California SB 253 | 2023 | Third-party assurance evidence |
| 7 | EU CBAM | 2023 | Verified embedded emissions documentation |
| 8 | TCFD | 2017 (rev 2021) | Methodology and assumptions disclosure |
| 9 | PCAF Global Standard | 2022 (3rd ed.) | Financial emissions data quality scoring |

### 2.4 Assurance Level Requirements

| Assurance Level | Standard | Requirements for Audit Trail |
|----------------|----------|------------------------------|
| Limited Assurance | ISAE 3410 | Inquiry + analytical procedures; audit trail must support inquiry |
| Reasonable Assurance | ISAE 3000 | Detailed testing of controls + substantive procedures; full lineage required |
| Verification | ISO 14064-3 | Evidence examination, materiality assessment; complete provenance chain |

---

## 3. Architecture

### 3.1 Seven-Engine Design

| # | Engine | Class Name | Purpose |
|---|--------|------------|---------|
| 1 | Audit Event Engine | `AuditEventEngine` | Immutable event recording with SHA-256 hash chains |
| 2 | Lineage Graph Engine | `LineageGraphEngine` | MRV calculation lineage DAG construction and traversal |
| 3 | Evidence Packager Engine | `EvidencePackagerEngine` | Audit evidence bundling for third-party verification |
| 4 | Compliance Tracer Engine | `ComplianceTracerEngine` | Regulatory framework requirement traceability |
| 5 | Change Detector Engine | `ChangeDetectorEngine` | Recalculation change tracking and version comparison |
| 6 | Compliance Checker Engine | `ComplianceCheckerEngine` | Multi-framework audit trail compliance validation |
| 7 | Audit Trail Pipeline Engine | `AuditTrailPipelineEngine` | 10-stage orchestration pipeline |

### 3.2 Engine Details

#### Engine 1: Audit Event Engine (`audit_event_engine.py`)

Records immutable audit events with SHA-256 chain hashing. Every MRV calculation generates events that are appended to a tamper-evident chain.

**Capabilities:**
- Genesis-anchored hash chains (one chain per organization per reporting year)
- 12 event types covering the full MRV lifecycle
- Event payload validation with schema enforcement
- Chain integrity verification (forward and backward)
- Concurrent-safe event appending with RLock
- Event querying by type, time range, agent, scope, and category
- Chain export for external audit systems
- Batch event recording for historical data

**Hash Chain Algorithm:**
```
hash_input = f"{event_id}|{prev_hash}|{event_type}|{timestamp_iso}|{canonical_json(payload)}"
event_hash = SHA-256(hash_input.encode('utf-8')).hexdigest()
```

#### Engine 2: Lineage Graph Engine (`lineage_graph_engine.py`)

Constructs and traverses directed acyclic graphs (DAGs) representing the full lineage of MRV calculations — from raw source data through emission factors and methodologies to final reported figures.

**Capabilities:**
- 8-level lineage depth (L1-Source through L8-Reporting)
- Forward lineage: "What reports depend on this EF?"
- Backward lineage: "Where did this emission figure come from?"
- Cross-scope lineage: Track data flowing across Scope 1/2/3
- Lineage chain extraction for specific data points
- Lineage graph visualization (Mermaid, DOT, JSON)
- Cycle detection and prevention
- Orphan node identification
- Graph snapshot and comparison

#### Engine 3: Evidence Packager Engine (`evidence_packager_engine.py`)

Bundles audit evidence into verifiable packages suitable for third-party verification under ISAE 3410, ISO 14064-3, and CSRD assurance.

**Capabilities:**
- Evidence package creation per reporting period
- Multi-format export (JSON, PDF metadata, XBRL anchor)
- Digital signature attachment (Ed25519, RSA-PSS, ECDSA)
- Package integrity verification
- Evidence completeness scoring
- Regulatory framework-specific packaging (CSRD, SB 253, CBAM)
- Package version control and supersession
- Bulk packaging for multi-entity organizations

#### Engine 4: Compliance Tracer Engine (`compliance_tracer_engine.py`)

Maps every audit trail event and lineage node to specific regulatory framework requirements, enabling auditors to quickly verify coverage.

**Capabilities:**
- 9 framework requirement databases (200+ requirements total)
- Bidirectional mapping: requirement → evidence and evidence → requirements
- Coverage gap analysis per framework
- Data point traceability for CSRD ESRS E1
- Disclosure requirement fulfillment tracking
- Cross-framework requirement overlap detection
- Compliance heatmap generation
- Missing evidence identification

#### Engine 5: Change Detector Engine (`change_detector_engine.py`)

Tracks changes when emission factors, methodologies, organizational boundaries, or source data are updated, triggering and documenting recalculations.

**Capabilities:**
- Change event detection (EF updates, methodology changes, data corrections)
- Impact analysis: which calculations are affected by a change
- Version comparison: old vs. new calculation results
- Materiality assessment of changes
- Base year recalculation triggering
- Change approval workflow tracking
- Cascade analysis: downstream report impacts
- Historical change timeline

#### Engine 6: Compliance Checker Engine (`compliance_checker_engine.py`)

Validates that the audit trail itself meets the requirements of all 9 supported regulatory frameworks.

**Capabilities:**
- Audit trail completeness checking
- Hash chain integrity verification
- Evidence sufficiency assessment per framework
- Data quality threshold enforcement
- Temporal coverage validation
- Organizational boundary coverage
- Methodology documentation completeness
- Assurance readiness scoring (limited vs. reasonable)

#### Engine 7: Audit Trail Pipeline Engine (`audit_trail_pipeline_engine.py`)

Orchestrates the 10-stage audit trail pipeline from event capture through evidence sealing.

**10 Pipeline Stages:**

| Stage | Name | Purpose |
|-------|------|---------|
| 1 | VALIDATE | Validate audit event inputs and configuration |
| 2 | CLASSIFY | Classify events by scope, category, and framework |
| 3 | RECORD | Record events to immutable hash chain |
| 4 | LINK | Link events to lineage graph nodes |
| 5 | TRACE | Map events to regulatory requirements |
| 6 | DETECT | Detect changes requiring recalculation |
| 7 | VERIFY | Verify hash chain and lineage integrity |
| 8 | PACKAGE | Bundle evidence for verification |
| 9 | COMPLIANCE | Check audit trail completeness per framework |
| 10 | SEAL | Final seal with provenance hash and optional signature |

### 3.3 Package Layout

```
greenlang/audit_trail_lineage/
├── __init__.py                           # Package exports, metadata
├── models.py                             # Pydantic models (25 enums, 20+ models)
├── config.py                             # Thread-safe singleton config (GL_ATL_)
├── audit_event_engine.py                 # Engine 1: Immutable event recording
├── lineage_graph_engine.py               # Engine 2: MRV lineage DAG
├── evidence_packager_engine.py           # Engine 3: Audit evidence bundling
├── compliance_tracer_engine.py           # Engine 4: Regulatory traceability
├── change_detector_engine.py             # Engine 5: Change tracking
├── compliance_checker.py                 # Engine 6: Multi-framework compliance
├── audit_trail_pipeline.py               # Engine 7: 10-stage pipeline
├── provenance.py                         # SHA-256 chain provenance tracking
├── metrics.py                            # Prometheus metrics (gl_atl_)
├── setup.py                              # Service facade
└── api/
    ├── __init__.py
    └── router.py                         # FastAPI router (25+ endpoints)
```

### 3.4 Test Layout

```
tests/unit/mrv/test_audit_trail_lineage/
├── __init__.py
├── conftest.py                           # Shared fixtures
├── test_models.py                        # ~120 tests
├── test_config.py                        # ~40 tests
├── test_audit_event_engine.py            # ~100 tests
├── test_lineage_graph_engine.py          # ~100 tests
├── test_evidence_packager_engine.py      # ~80 tests
├── test_compliance_tracer_engine.py      # ~80 tests
├── test_change_detector_engine.py        # ~80 tests
├── test_compliance_checker.py            # ~60 tests
├── test_pipeline.py                      # ~60 tests
├── test_provenance.py                    # ~50 tests
├── test_setup.py                         # ~30 tests
└── test_api.py                           # ~100 tests
                                          # Total: ~900 tests
```

---

## 4. Data Models

### 4.1 Enumerations (25)

1. `AuditEventType` — 12 event types (calculation_initiated through audit_sealed)
2. `LineageLevel` — 8 levels (L1_SOURCE through L8_REPORTING)
3. `LineageNodeType` — 10 types (source_data, activity_data, emission_factor, methodology, calculation, allocation, aggregation, compliance_check, report_item, evidence)
4. `LineageEdgeType` — 8 types (data_flow, factor_application, method_selection, allocation_link, aggregation_link, compliance_link, report_link, evidence_link)
5. `EmissionScope` — 4 values (scope_1, scope_2_location, scope_2_market, scope_3)
6. `Scope3Category` — 15 categories (cat_1 through cat_15)
7. `EFSource` — 10 sources (DEFRA, EPA, IPCC, IEA, Ecoinvent, ADEME, BEIS, eGRID, GaBi, custom)
8. `CalculationMethodology` — 8 methodologies (emission_factor, mass_balance, stoichiometric, direct_measurement, eeio, average_data, supplier_specific, hybrid)
9. `ChangeType` — 8 types (ef_update, methodology_change, data_correction, boundary_change, base_year_recalc, allocation_change, scope_reclassification, structural_change)
10. `ChangeSeverity` — 4 levels (critical, high, medium, low)
11. `EvidencePackageStatus` — 5 statuses (draft, complete, signed, submitted, superseded)
12. `AssuranceLevel` — 3 levels (limited, reasonable, verification)
13. `ComplianceStatus` — 4 statuses (compliant, partial, non_compliant, not_applicable)
14. `DataQualityTier` — 5 tiers (tier_1 through tier_5, aligning with PCAF DQ scores)
15. `SignatureAlgorithm` — 4 algorithms (ed25519, rsa_pss_sha256, ecdsa_p256, hmac_sha256)
16. `AuditTrailStatus` — 4 statuses (active, sealed, archived, invalidated)
17. `VerificationResult` — 3 results (valid, invalid, inconclusive)
18. `TraversalDirection` — 2 directions (forward, backward)
19. `GraphFormat` — 4 formats (mermaid, dot, json, d3)
20. `ReportingPeriodType` — 3 types (annual, quarterly, monthly)
21. `RecalculationTrigger` — 6 triggers (ef_update, data_correction, boundary_change, methodology_change, error_correction, structural_change)
22. `FrameworkIdentifier` — 9 frameworks (ghg_protocol, iso_14064, csrd_esrs_e1, sb_253, cbam, cdp, tcfd, pcaf, sbti)
23. `MaterialityThreshold` — 3 thresholds (quantitative_5pct, quantitative_1pct, qualitative)
24. `ChainIntegrityStatus` — 3 statuses (intact, broken, unverified)
25. `PipelineStage` — 10 stages (VALIDATE through SEAL)

### 4.2 Core Input Models

- **AuditEventInput** — Event type, agent ID, scope, category, payload, metadata
- **LineageNodeInput** — Node type, level, agent ID, data reference, metadata
- **LineageEdgeInput** — Source node, target node, edge type, transformation details
- **EvidencePackageRequest** — Organization, reporting period, frameworks, scope filter
- **ComplianceTraceRequest** — Framework, data points, reporting period
- **ChangeDetectionInput** — Change type, affected entity, old/new values, trigger
- **BatchAuditRequest** — List of audit events for bulk processing
- **LineageQueryInput** — Start node, direction, max depth, filters
- **ChainVerificationRequest** — Organization, reporting year, chain segment
- **RecalculationRequest** — Change event ID, affected calculations, cascade flag

### 4.3 Core Output Models

- **AuditEventOutput** — Event ID, hash, chain position, verification status
- **LineageGraphOutput** — Nodes, edges, root nodes, leaf nodes, depth statistics
- **LineageChainOutput** — Ordered chain from source to target with all intermediate nodes
- **EvidencePackageOutput** — Package ID, contents, completeness score, signature, hash
- **ComplianceTraceOutput** — Requirements mapped, coverage percentage, gaps identified
- **ChangeDetectionOutput** — Changes detected, materiality assessment, affected calculations
- **ChainVerificationOutput** — Chain status, break points (if any), verification hash
- **AuditTrailSummary** — Event counts by type, chain integrity, coverage metrics
- **ComplianceReport** — Per-framework compliance status, evidence sufficiency, recommendations
- **AuditTrailCalculationResult** — Master result with all audit metadata, provenance, DC checks

---

## 5. Audit Event Schema

### 5.1 Event Payload Structure

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | UUID | Unique event identifier |
| `event_type` | AuditEventType | Type of audit event |
| `agent_id` | str | Source agent (e.g., GL-MRV-S1-001) |
| `scope` | EmissionScope | Emission scope |
| `category` | Optional[int] | Scope 3 category (1-15) |
| `organization_id` | UUID | Organization identifier |
| `reporting_year` | int | Reporting period year |
| `calculation_id` | UUID | Related calculation ID |
| `timestamp` | datetime | UTC timestamp |
| `payload` | dict | Event-specific data |
| `data_quality_score` | Decimal | DQ score (0.0-1.0) |
| `prev_event_hash` | str | Previous event hash in chain |
| `event_hash` | str | SHA-256 hash of this event |
| `metadata` | dict | Additional context |

### 5.2 Lineage Node Structure

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | UUID | Unique node identifier |
| `node_type` | LineageNodeType | Type of lineage node |
| `level` | LineageLevel | Lineage depth level (L1-L8) |
| `agent_id` | str | Agent that created this node |
| `qualified_name` | str | Fully qualified data reference |
| `value` | Decimal | Numeric value (if applicable) |
| `unit` | str | Unit of measurement |
| `data_quality_score` | Decimal | DQ score at this node |
| `provenance_hash` | str | SHA-256 provenance hash |
| `created_at` | datetime | Node creation timestamp |
| `metadata` | dict | Node-specific context |

---

## 6. API Endpoints

| # | Method | Path | Permission | Description |
|---|--------|------|------------|-------------|
| 1 | POST | `/api/v1/audit-trail-lineage/events` | `audit-trail-lineage:record` | Record a single audit event |
| 2 | POST | `/api/v1/audit-trail-lineage/events/batch` | `audit-trail-lineage:record` | Record batch of audit events |
| 3 | GET | `/api/v1/audit-trail-lineage/events/{event_id}` | `audit-trail-lineage:read` | Get single audit event |
| 4 | GET | `/api/v1/audit-trail-lineage/events` | `audit-trail-lineage:read` | List/query audit events |
| 5 | POST | `/api/v1/audit-trail-lineage/chain/verify` | `audit-trail-lineage:verify` | Verify hash chain integrity |
| 6 | GET | `/api/v1/audit-trail-lineage/chain/{org_id}/{year}` | `audit-trail-lineage:read` | Get chain for org/year |
| 7 | POST | `/api/v1/audit-trail-lineage/lineage/nodes` | `audit-trail-lineage:record` | Create lineage node |
| 8 | POST | `/api/v1/audit-trail-lineage/lineage/edges` | `audit-trail-lineage:record` | Create lineage edge |
| 9 | GET | `/api/v1/audit-trail-lineage/lineage/graph/{calc_id}` | `audit-trail-lineage:read` | Get lineage graph for calculation |
| 10 | POST | `/api/v1/audit-trail-lineage/lineage/trace` | `audit-trail-lineage:trace` | Trace lineage forward/backward |
| 11 | GET | `/api/v1/audit-trail-lineage/lineage/visualize/{calc_id}` | `audit-trail-lineage:read` | Get lineage visualization |
| 12 | POST | `/api/v1/audit-trail-lineage/evidence/package` | `audit-trail-lineage:package` | Create evidence package |
| 13 | GET | `/api/v1/audit-trail-lineage/evidence/{package_id}` | `audit-trail-lineage:read` | Get evidence package |
| 14 | POST | `/api/v1/audit-trail-lineage/evidence/{package_id}/sign` | `audit-trail-lineage:sign` | Sign evidence package |
| 15 | POST | `/api/v1/audit-trail-lineage/evidence/{package_id}/verify` | `audit-trail-lineage:verify` | Verify package signature |
| 16 | POST | `/api/v1/audit-trail-lineage/compliance/trace` | `audit-trail-lineage:compliance` | Trace compliance requirements |
| 17 | GET | `/api/v1/audit-trail-lineage/compliance/coverage/{org_id}` | `audit-trail-lineage:read` | Get compliance coverage |
| 18 | POST | `/api/v1/audit-trail-lineage/changes/detect` | `audit-trail-lineage:detect` | Detect changes |
| 19 | GET | `/api/v1/audit-trail-lineage/changes/{change_id}` | `audit-trail-lineage:read` | Get change details |
| 20 | GET | `/api/v1/audit-trail-lineage/changes/{change_id}/impact` | `audit-trail-lineage:read` | Get change impact analysis |
| 21 | POST | `/api/v1/audit-trail-lineage/pipeline/execute` | `audit-trail-lineage:execute` | Execute full audit trail pipeline |
| 22 | POST | `/api/v1/audit-trail-lineage/pipeline/execute/batch` | `audit-trail-lineage:execute` | Batch pipeline execution |
| 23 | GET | `/api/v1/audit-trail-lineage/summary/{org_id}/{year}` | `audit-trail-lineage:read` | Get audit trail summary |
| 24 | DELETE | `/api/v1/audit-trail-lineage/events/{event_id}` | `audit-trail-lineage:delete` | Soft-delete event (admin only) |
| 25 | GET | `/api/v1/audit-trail-lineage/health` | `audit-trail-lineage:read` | Health check |

---

## 7. Database Schema

### 7.1 Schema

```
Schema: audit_trail_lineage_service
Table prefix: gl_atl_
Migration: V081__audit_trail_lineage_service.sql
```

### 7.2 Tables (12)

| # | Table | Type | Description |
|---|-------|------|-------------|
| 1 | `gl_atl_event_types` | Reference | Audit event type definitions |
| 2 | `gl_atl_framework_requirements` | Reference | Regulatory framework requirements (200+ rows) |
| 3 | `gl_atl_lineage_node_types` | Reference | Lineage node type definitions |
| 4 | `gl_atl_change_type_definitions` | Reference | Change type severity mappings |
| 5 | `gl_atl_audit_events` | Operational (Hypertable) | Immutable audit events with hash chains |
| 6 | `gl_atl_lineage_nodes` | Operational | Lineage graph nodes |
| 7 | `gl_atl_lineage_edges` | Operational | Lineage graph edges |
| 8 | `gl_atl_evidence_packages` | Operational (Hypertable) | Audit evidence packages |
| 9 | `gl_atl_compliance_traces` | Operational (Hypertable) | Compliance requirement traces |
| 10 | `gl_atl_change_events` | Operational | Change detection events |
| 11 | `gl_atl_chain_verifications` | Supporting | Hash chain verification results |
| 12 | `gl_atl_audit_trail_summaries` | Supporting | Cached audit trail summaries |

### 7.3 Hypertables (3)

| Table | Chunk Interval | Retention | Compression |
|-------|---------------|-----------|-------------|
| `gl_atl_audit_events` | 7 days | 7 years | After 90 days |
| `gl_atl_evidence_packages` | 30 days | 10 years | After 180 days |
| `gl_atl_compliance_traces` | 30 days | 5 years | After 90 days |

### 7.4 Continuous Aggregates (2)

| View | Bucket | Source |
|------|--------|--------|
| `gl_atl_daily_event_stats` | 1 day | `gl_atl_audit_events` |
| `gl_atl_monthly_compliance_summary` | 1 month | `gl_atl_compliance_traces` |

---

## 8. Metrics

| # | Metric Name | Type | Labels | Description |
|---|------------|------|--------|-------------|
| 1 | `gl_atl_events_total` | Counter | event_type, scope, status | Total audit events recorded |
| 2 | `gl_atl_event_recording_duration_seconds` | Histogram | event_type | Event recording latency |
| 3 | `gl_atl_chain_length` | Gauge | org_id, year | Current chain length |
| 4 | `gl_atl_chain_verifications_total` | Counter | result | Chain verification outcomes |
| 5 | `gl_atl_lineage_nodes_total` | Counter | node_type, level | Lineage nodes created |
| 6 | `gl_atl_lineage_edges_total` | Counter | edge_type | Lineage edges created |
| 7 | `gl_atl_lineage_depth` | Histogram | — | Lineage graph depth distribution |
| 8 | `gl_atl_evidence_packages_total` | Counter | status, framework | Evidence packages created |
| 9 | `gl_atl_evidence_completeness_score` | Histogram | framework | Completeness score distribution |
| 10 | `gl_atl_compliance_coverage_pct` | Gauge | framework, org_id | Compliance coverage percentage |
| 11 | `gl_atl_changes_detected_total` | Counter | change_type, severity | Changes detected |
| 12 | `gl_atl_recalculations_triggered_total` | Counter | trigger_type | Recalculations triggered |
| 13 | `gl_atl_pipeline_duration_seconds` | Histogram | stage | Pipeline stage durations |
| 14 | `gl_atl_pipeline_executions_total` | Counter | status | Pipeline execution outcomes |

---

## 9. Provenance Chain

| # | Stage | Input Hash | Output Hash | Description |
|---|-------|------------|-------------|-------------|
| 1 | VALIDATE | Raw event data | Validated event | Input validation and schema enforcement |
| 2 | CLASSIFY | Validated event | Classified event | Scope/category/framework classification |
| 3 | RECORD | Classified event | Chained event | Append to immutable hash chain |
| 4 | LINK | Chained event | Linked event | Connect to lineage graph |
| 5 | TRACE | Linked event | Traced event | Map to regulatory requirements |
| 6 | DETECT | Traced event | Change analysis | Detect recalculation needs |
| 7 | VERIFY | Change analysis | Verification result | Verify chain and lineage integrity |
| 8 | PACKAGE | Verification result | Evidence bundle | Bundle audit evidence |
| 9 | COMPLIANCE | Evidence bundle | Compliance assessment | Check framework compliance |
| 10 | SEAL | Compliance assessment | Final sealed result | Seal with provenance hash and signature |

---

## 10. Zero-Hallucination Guarantees

1. **No LLM involvement** in audit event recording, hashing, or lineage construction
2. **Deterministic SHA-256 hashing** with canonical JSON serialization (sorted keys, consistent encoding)
3. **Immutable event chains** — events cannot be modified once recorded; only soft-delete with audit trail
4. **Cryptographic verification** — any tampering detectable via hash chain validation
5. **Authoritative regulatory databases** — framework requirements sourced from official regulation text
6. **Decimal arithmetic** — all emissions values use Python Decimal with ROUND_HALF_UP
7. **Reproducible lineage** — lineage graphs are deterministically constructed from the same inputs
8. **Version-controlled EF references** — every emission factor reference includes source, version, and effective date
9. **Thread-safe singleton engines** — no race conditions in concurrent audit event recording
10. **Genesis-anchored chains** — every hash chain starts from a known genesis hash for verification

---

## 11. Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| Event recording latency | < 10 ms (p99) | Single event append |
| Batch recording throughput | > 1,000 events/sec | Bulk historical processing |
| Chain verification | < 5 sec for 100K events | Full chain integrity check |
| Lineage traversal | < 100 ms for 8-level depth | Forward or backward |
| Evidence package generation | < 30 sec per package | Full reporting period |
| Compliance trace | < 500 ms per framework | Single framework check |
| Graph visualization | < 2 sec for 10K nodes | Mermaid/DOT export |

---

## 12. Integration Points

### 12.1 Upstream (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| MRV 001-013 | Scope 1 & 2 calculation events | Calculation ID, EF used, methodology, emissions result |
| MRV 014-028 | Scope 3 category calculation events | Category, method, allocation, supplier data references |
| MRV 029 | Scope 3 Category Mapper classifications | Category assignments, routing decisions |
| DATA 001-007 | Data intake events | Source references, data quality scores |
| DATA 008-020 | Data quality events | Validation results, corrections applied |
| FOUND 005 | Citations and evidence | EF citations, methodology references |
| FOUND 008 | Reproducibility artifacts | Environment fingerprints, version manifests |

### 12.2 Downstream (Consumers)

| Consumer | Integration | Data Provided |
|----------|-------------|---------------|
| CSRD Reporting | ESRS E1 data point lineage | Full traceability chain per disclosure |
| SB 253 Assurance | Third-party verification evidence | Evidence packages with signatures |
| CBAM Verification | Embedded emissions documentation | Calculation lineage and EF sources |
| Internal Audit | Audit trail queries | Event history, change logs |
| Consolidation Agent | Cross-scope reconciliation audit | Scope 1+2+3 aggregation lineage |
| Dashboards | Audit trail analytics | Event counts, coverage metrics |

---

## 13. Acceptance Criteria

- [ ] 7 engines implemented with thread-safe singleton pattern
- [ ] 25 enumerations and 20+ Pydantic models defined
- [ ] SHA-256 chain hashing with genesis anchoring
- [ ] 12 audit event types fully supported
- [ ] 8-level lineage depth (L1-L8) with DAG construction
- [ ] Forward and backward lineage traversal
- [ ] Evidence package generation for 9 frameworks
- [ ] Digital signature support (Ed25519, RSA-PSS, ECDSA)
- [ ] Compliance traceability for 200+ requirements
- [ ] Change detection with materiality assessment
- [ ] 10-stage pipeline with stage-level provenance
- [ ] 25+ REST API endpoints with OpenAPI documentation
- [ ] V081 database migration with 12 tables, 3 hypertables, 2 continuous aggregates
- [ ] 14 Prometheus metrics with `gl_atl_` prefix
- [ ] 900+ unit tests with > 85% code coverage
- [ ] Auth integration with RBAC permissions
- [ ] Zero LLM dependency in audit/lineage logic
- [ ] Event recording latency < 10ms p99

---

## 14. Risks and Mitigations

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | Hash chain corruption from concurrent writes | High | RLock on chain append operations |
| 2 | Storage growth from high-volume audit events | Medium | TimescaleDB compression + retention policies |
| 3 | Lineage graph cycles causing infinite traversal | Medium | Cycle detection with max depth limits |
| 4 | Evidence package generation timeout | Medium | Streaming generation with progress tracking |
| 5 | Framework requirement changes | Low | Versioned requirement database with effective dates |
| 6 | Cross-agent event ordering | Medium | UTC timestamps with microsecond precision + logical clocks |

---

## 15. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-01 | GreenLang Platform Team | Initial PRD |
