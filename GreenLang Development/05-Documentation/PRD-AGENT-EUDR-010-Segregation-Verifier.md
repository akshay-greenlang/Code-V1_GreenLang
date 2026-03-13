# PRD: AGENT-EUDR-010 -- Segregation Verifier Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-010 |
| **Agent ID** | GL-EUDR-SGV-010 |
| **Component** | Segregation Verifier Agent |
| **Category** | EUDR Regulatory Agent -- Physical Segregation & Cross-Contamination Prevention |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-08 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-08 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

EUDR Article 9 and the implementing guidance on chain of custody (CoC) require operators using the **Segregated** (SG) model to physically separate EUDR-compliant material from non-compliant material at every stage of the supply chain -- storage, transport, processing, and handling. Unlike the Mass Balance model (which allows physical mixing with administrative accounting), the Segregated model demands verifiable physical separation, making contamination a compliance-breaking event.

In practice, segregation failures are the single largest cause of non-compliance findings in commodity chain audits:

- **Storage co-mingling**: Compliant cocoa beans stored in adjacent silos without physical barriers, with shared conveyor belts that carry residual non-compliant material between loads.
- **Transport contamination**: A truck that previously carried non-compliant palm oil is loaded with compliant material without cleaning verification, destroying the segregation claim.
- **Processing line cross-over**: A processing facility runs compliant and non-compliant soya through the same extraction line on alternating shifts, with no flush/purge protocol between runs.
- **Labeling failures**: Compliant rubber bales are stored without segregation labels, making them indistinguishable from non-compliant stock and invalidating the SG claim.
- **Shared equipment**: Weighing scales, forklifts, and temporary storage bins used interchangeably for compliant and non-compliant material without cleaning logs.
- **Facility design gaps**: Warehouses lack physical partitioning, color-coded zones, or barrier systems to prevent accidental mixing.
- **Temporal overlap**: Compliant and non-compliant batches processed within the same shift window without adequate changeover time or cleaning verification.
- **Documentation gaps**: Segregation protocols exist on paper but facility audits reveal no cleaning logs, no separation checklists, and no photographic evidence of physical barriers.

Without a systematic segregation verification system, operators using the SG model cannot demonstrate to competent authorities that their products genuinely maintained physical separation throughout the supply chain. A single contamination event invalidates the segregation claim for the entire batch, potentially affecting thousands of tonnes of product and triggering penalties of up to 4% of annual EU turnover.

### 1.2 Solution Overview

Agent-EUDR-010: Segregation Verifier Agent provides comprehensive physical segregation verification across the entire supply chain for operators using the Segregated chain of custody model. It validates that compliant material is physically separated from non-compliant material at every storage, transport, processing, and handling point, detecting cross-contamination risks before they become compliance failures.

Core capabilities:

1. **Segregation point validation** -- Identifies and validates every point in the supply chain where segregation must be maintained (storage facilities, transport vehicles, processing lines, handling areas), scoring each point on physical separation adequacy.
2. **Storage segregation auditing** -- Verifies that storage facilities maintain physical separation between compliant and non-compliant material through dedicated zones, physical barriers, sealed containers, or separate buildings.
3. **Transport segregation tracking** -- Validates that transport vehicles, containers, and vessels used for compliant material are either dedicated or properly cleaned/flushed between compliant and non-compliant loads.
4. **Processing line verification** -- Verifies that processing facilities use either dedicated lines for compliant material or implement validated changeover/flush/purge protocols between compliant and non-compliant runs.
5. **Cross-contamination detection** -- Analyzes temporal, spatial, and equipment-sharing patterns to detect potential cross-contamination events that would invalidate segregation claims.
6. **Labeling and marking verification** -- Ensures all segregated material, storage areas, transport vehicles, and processing lines carry proper identification markings that distinguish compliant from non-compliant status.
7. **Facility segregation assessment** -- Evaluates overall facility segregation capability through layout analysis, protocol review, equipment inventory, and historical contamination event tracking.
8. **Segregation compliance reporting** -- Generates segregation audit reports, contamination risk assessments, facility readiness scores, and regulatory evidence packages for competent authority inspections.

### 1.3 Dependencies

| Dependency | Component | Integration |
|------------|-----------|-------------|
| AGENT-EUDR-001 | Supply Chain Mapping Master | Supply chain graph structure, facility nodes |
| AGENT-EUDR-002 | Geolocation Verification | Facility GPS verification |
| AGENT-EUDR-008 | Multi-Tier Supplier Tracker | Supplier profiles, facility capabilities |
| AGENT-EUDR-009 | Chain of Custody Agent | CoC model assignments, batch data, custody events |
| AGENT-DATA-005 | EUDR Traceability Connector | Raw segregation audit data intake |

---

## 2. Regulatory Context

### 2.1 EUDR Articles Addressed

| Article | Requirement | Agent Feature |
|---------|-------------|---------------|
| Art. 4 | Due diligence obligation on deforestation-free + legal production | Segregation ensures compliant-only material reaches EU market |
| Art. 9(1)(d) | Geolocation of all production plots | Segregation preserves plot-level traceability without mixing |
| Art. 9(1)(f) | Quantity/weight of product | Segregation prevents volume inflation from non-compliant mixing |
| Art. 10(2)(f) | Adequate and verifiable information on compliance | Segregation evidence package for competent authorities |
| Art. 12 | Simplified due diligence for low-risk countries | Segregation verification level based on origin risk |
| Art. 14 | 5-year record retention | Immutable segregation audit trail |
| Art. 16 | Due diligence assessment: risk mitigation | Segregation as primary risk mitigation for SG model |
| Art. 31 | Review and reporting | Segregation compliance analytics |

### 2.2 Chain of Custody Standards for Segregation

| Standard | Segregation Requirements | Key Rules |
|----------|--------------------------|-----------|
| ISO 22095:2020 | Physical separation at all chain stages | No physical mixing; administrative tracking of separation |
| FSC-STD-40-004 v3 | Dedicated storage, separate processing | Physical and temporal separation; cleaning protocols |
| RSPO SCC 2020 | Segregated supply chain | Separate tanks, silos, processing; no mixing at any point |
| ISCC 202 v4.0 | Physical segregation per batch | Mass balance alternative available; SG requires physical proof |
| UTZ/RA-CoC-002 | Separated handling and storage | Visual identification, physical barriers required |
| Fairtrade SOP | Physical separation throughout chain | Dedicated storage areas, labeled containers |

---

## 3. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Segregation point coverage | >= 95% of SG facilities verified | Verified points / total SG points |
| Cross-contamination detection rate | >= 98% of contamination risks identified | Detection vs manual audit findings |
| False positive rate | <= 5% of flagged contamination events | False alerts / total alerts |
| Facility assessment accuracy | >= 90% agreement with physical audits | Score correlation with on-site inspectors |
| Processing throughput | >= 500 segregation verifications/second | Batch verification benchmark |
| Verification latency | < 300ms per segregation point check | P95 latency |
| Test coverage | >= 500 unit tests | Pytest count |
| Labeling compliance rate | >= 99% detection of unlabeled material | Unlabeled detection accuracy |

---

## 4. Scope

### 4.1 In Scope
- All 7 EUDR commodities + derived products under Segregated CoC model
- Storage segregation verification (silos, warehouses, tanks, containers, yards)
- Transport segregation verification (trucks, containers, vessels, rail cars)
- Processing line segregation verification (dedicated vs shared lines with changeover)
- Cross-contamination risk detection and scoring
- Labeling and marking compliance verification
- Facility segregation capability assessment
- Cleaning and changeover protocol validation
- Temporal segregation analysis (same-line, different-shift scenarios)
- Equipment sharing risk assessment
- Segregation evidence compilation for regulatory audits
- 5-year segregation audit trail per Article 14

### 4.2 Out of Scope
- Physical IoT sensor data integration (future phase)
- Real-time GPS tracking of transport vehicles
- Automated image recognition of labels/markings
- Mass Balance model accounting (handled by EUDR-009)
- Certification audit management (handled by EUDR-008)
- Financial costs of segregation implementation

---

## 5. Zero-Hallucination Principles

1. All segregation verdicts are binary (PASS/FAIL/WARNING) based on verifiable physical evidence records -- no LLM inference.
2. Cross-contamination risk scores use deterministic scoring formulas with auditable weights.
3. Facility assessment scores are composite calculations from individually scored criteria -- no subjective ratings.
4. Cleaning/changeover protocol validation uses deterministic rule matching against published standards.
5. SHA-256 provenance hashing ensures tamper detection on all segregation records.
6. Every segregation failure links to a specific evidence record or is flagged as evidence-absent.

---

## 6. Feature Requirements

### 6.1 Feature 1: Segregation Point Validation (P0)

**Requirements**:
- F1.1: Identify all segregation control points (SCPs) in the supply chain: storage, transport, processing, handling, loading/unloading
- F1.2: SCP attributes: scp_id, facility_id, location (GPS), scp_type (storage/transport/processing/handling), commodity, capacity, segregation_method
- F1.3: 8 segregation methods: dedicated_facility, physical_barrier, sealed_container, temporal_separation, dedicated_line, color_coded_zone, locked_area, separate_building
- F1.4: SCP status tracking: verified, unverified, failed, expired, pending_inspection
- F1.5: SCP verification scheduling: initial verification + periodic re-verification (configurable interval, default 90 days)
- F1.6: Automatic SCP discovery from custody event chains (new facilities auto-generate SCP requirements)
- F1.7: SCP compliance scoring: 0-100 composite from physical evidence, documentation, historical performance
- F1.8: SCP risk classification: low (dedicated facility), medium (physical barrier), high (temporal separation only)
- F1.9: SCP history tracking with amendment trail
- F1.10: Bulk SCP import from facility audit reports

### 6.2 Feature 2: Storage Segregation Auditing (P0)

**Requirements**:
- F2.1: Storage zone mapping per facility: compliant zones, non-compliant zones, buffer zones, restricted areas
- F2.2: 12 storage types: silo, warehouse_bay, tank, container_yard, cold_room, dry_store, bonded_area, open_yard, covered_shed, sealed_unit, locked_cage, segregated_floor
- F2.3: Physical barrier validation: wall, fence, floor marking, sealed door, separate building, dedicated vehicle access
- F2.4: Capacity tracking per zone: maximum tonnage, current occupancy, overflow risk
- F2.5: Adjacent storage risk assessment: compliant and non-compliant zones sharing walls/access points
- F2.6: Storage event logging: material_in, material_out, zone_transfer, cleaning, inspection with timestamps
- F2.7: Inventory reconciliation: segregated zone quantities must match batch system quantities
- F2.8: Contamination incident recording: spill, overflow, misplacement, barrier_breach with impact assessment
- F2.9: Cleaning protocol verification: cleaning_date, method, verified_by, next_cleaning_due
- F2.10: Storage segregation score: composite of barrier_quality, zone_separation, cleaning_compliance, capacity_utilization

### 6.3 Feature 3: Transport Segregation Tracking (P0)

**Requirements**:
- F3.1: Transport vehicle/container registry: vehicle_id, type (truck/container/vessel/rail_car/barge), dedicated_status, last_cargo_type
- F3.2: 10 transport types: bulk_truck, container_truck, tanker, dry_bulk_vessel, container_vessel, tanker_vessel, rail_hopper, rail_container, barge, air_freight
- F3.3: Dedicated vehicle tracking: vehicles used exclusively for compliant material require no cleaning verification
- F3.4: Shared vehicle cleaning verification: cleaning_date, cleaning_method (wash/steam/fumigation/flush), cleaning_certificate, verified_by
- F3.5: Previous cargo tracking: last 5 cargoes per vehicle to assess contamination risk from residual material
- F3.6: Transit seal verification: seal applied at origin, seal number recorded, seal intact at destination
- F3.7: Multi-modal transport segregation: verify segregation maintained across mode changes (truck→vessel→truck)
- F3.8: Container condition assessment: structural integrity, seal quality, contamination history
- F3.9: Transport segregation score: composite of vehicle_dedication, cleaning_compliance, seal_integrity, cargo_history
- F3.10: Route segregation analysis: verify no stops at non-compliant facilities during transit

### 6.4 Feature 4: Processing Line Verification (P0)

**Requirements**:
- F4.1: Processing line registry: line_id, facility_id, line_type, commodity, capacity, dedicated_status
- F4.2: 15 processing line types: extraction, pressing, milling, refining, roasting, fermenting, drying, cutting, tanning, spinning, smelting, fractionation, blending, packaging, grading
- F4.3: Dedicated line verification: lines used exclusively for compliant material with no shared input/output equipment
- F4.4: Shared line changeover protocol: flush_volume, flush_duration, flush_product, purge_method, cleaning_verification
- F4.5: Changeover validation: verify minimum changeover time met, flush product disposed (not mixed with compliant output)
- F4.6: First-run-after-changeover tracking: first batch after changeover flagged for enhanced testing/segregation
- F4.7: Equipment sharing analysis: identify all shared equipment (scales, conveyors, hoppers, tanks) between compliant and non-compliant lines
- F4.8: Temporal segregation verification: compliant and non-compliant runs separated by minimum time gap
- F4.9: Processing line segregation score: composite of line_dedication, changeover_compliance, equipment_sharing, temporal_separation
- F4.10: Line contamination history: track all contamination incidents per line with root cause and corrective action

### 6.5 Feature 5: Cross-Contamination Detection (P0)

**Requirements**:
- F5.1: Cross-contamination risk assessment: evaluate all pathways where compliant and non-compliant material could mix
- F5.2: 10 contamination pathway types: shared_storage, shared_transport, shared_processing, shared_equipment, temporal_overlap, adjacent_storage, residual_material, handling_error, labeling_error, documentation_error
- F5.3: Temporal proximity analysis: detect when compliant and non-compliant material are handled within a configurable time window (default 4 hours)
- F5.4: Spatial proximity analysis: detect when compliant and non-compliant material are stored within configurable distance (default 5 meters)
- F5.5: Equipment sharing detection: flag when the same equipment is used for compliant and non-compliant material without cleaning verification
- F5.6: Contamination event recording: timestamp, location, pathway_type, affected_batches, affected_quantity_kg, severity (critical/major/minor)
- F5.7: Contamination impact propagation: trace all downstream batches affected by a contamination event
- F5.8: Automatic batch status downgrade: contaminated compliant batches automatically lose SG status (become MB or non-compliant)
- F5.9: Root cause analysis templates: predefined root cause categories with corrective action recommendations
- F5.10: Contamination risk heatmap data: aggregate risk scores per facility/location/commodity for dashboarding

### 6.6 Feature 6: Labeling and Marking Verification (P0)

**Requirements**:
- F6.1: Labeling requirements per segregation context: storage_label, transport_label, processing_label, batch_label, zone_marker
- F6.2: 8 label types: compliance_tag, zone_sign, vehicle_placard, container_seal_label, batch_sticker, pallet_marker, silo_sign, processing_line_marker
- F6.3: Label content validation: minimum required fields (compliance_status, batch_id, commodity, origin, date, operator_id)
- F6.4: Label placement verification: correct label on correct container/zone/vehicle (label_id linked to scp_id)
- F6.5: Label condition tracking: applied, readable, damaged, missing, expired
- F6.6: Color-code system validation: verify facility uses consistent color coding (e.g., green=compliant, red=non-compliant)
- F6.7: Labeling event logging: label_applied, label_verified, label_damaged, label_replaced, label_removed with timestamps
- F6.8: Missing label detection: flag any segregation point without a current valid label
- F6.9: Label audit trail: full history of labeling events per batch/zone/vehicle
- F6.10: Labeling compliance score: composite of coverage, readability, accuracy, timeliness

### 6.7 Feature 7: Facility Segregation Assessment (P0)

**Requirements**:
- F7.1: Facility segregation profile: facility_id, facility_type, commodities_handled, CoC_models_used, segregation_capability_level
- F7.2: 6 segregation capability levels: level_0 (no segregation), level_1 (temporal only), level_2 (marked zones), level_3 (physical barriers), level_4 (separate areas), level_5 (dedicated facility)
- F7.3: Layout assessment criteria: zone separation, barrier quality, access control, equipment dedication, cleaning infrastructure
- F7.4: Protocol assessment criteria: written SOPs, staff training records, cleaning schedules, inspection frequencies, incident response plans
- F7.5: Historical performance assessment: contamination incidents, near-misses, audit findings, corrective action completion
- F7.6: Facility readiness score: 0-100 composite from layout (30%), protocols (25%), history (20%), labeling (15%), documentation (10%)
- F7.7: Improvement recommendations: prioritized list of actions to improve segregation capability level
- F7.8: Peer comparison: facility score benchmarked against commodity-specific averages
- F7.9: Certification readiness: assessment against FSC/RSPO/ISCC segregation requirements
- F7.10: Re-assessment scheduling: automatic trigger when score drops below threshold or after incident

### 6.8 Feature 8: Segregation Compliance Reporting (P0)

**Requirements**:
- F8.1: Facility segregation audit report: comprehensive assessment with evidence compilation
- F8.2: Contamination incident report: event details, affected batches, root cause, corrective actions
- F8.3: Supply chain segregation summary: end-to-end segregation status from origin to EU entry
- F8.4: Labeling compliance report: coverage, accuracy, and gap analysis
- F8.5: Report formats: JSON, PDF, CSV, EUDR XML
- F8.6: Regulatory evidence package: compiled documentation for competent authority inspections
- F8.7: Trend analysis report: segregation performance over time with improvement trajectory
- F8.8: Risk assessment report: contamination risk heatmap with mitigation recommendations

---

## 7. Technical Requirements

### 7.1 Architecture

```
greenlang/agents/eudr/segregation_verifier/
    __init__.py                          # Package exports (80+ symbols)
    config.py                            # SegregationVerifierConfig singleton
    models.py                            # Pydantic v2 models, enums
    provenance.py                        # SHA-256 chain hashing
    metrics.py                           # Prometheus metrics (gl_eudr_sgv_ prefix)
    segregation_point_validator.py       # Engine 1: SCP identification & validation
    storage_segregation_auditor.py       # Engine 2: Storage zone verification
    transport_segregation_tracker.py     # Engine 3: Transport vehicle segregation
    processing_line_verifier.py          # Engine 4: Processing line verification
    cross_contamination_detector.py      # Engine 5: Contamination detection
    labeling_verification_engine.py      # Engine 6: Label/marking verification
    facility_assessment_engine.py        # Engine 7: Facility assessment
    compliance_reporter.py               # Engine 8: Reports & evidence
    setup.py                             # SegregationVerifierService facade
    reference_data/
        __init__.py
        segregation_standards.py         # FSC/RSPO/ISCC segregation rules
        cleaning_protocols.py            # Cleaning/changeover requirements
        labeling_requirements.py         # Label content/placement rules
    api/
        __init__.py
        router.py
        schemas.py
        dependencies.py
        scp_routes.py                    # Segregation control point routes
        storage_routes.py                # Storage segregation routes
        transport_routes.py              # Transport segregation routes
        processing_routes.py             # Processing line routes
        contamination_routes.py          # Contamination detection routes
        assessment_routes.py             # Facility assessment routes
        report_routes.py                 # Reporting routes
```

### 7.2 Database Schema (V098)

| Table | Type | Description |
|-------|------|-------------|
| `gl_eudr_sgv_segregation_points` | regular | Segregation control point master records |
| `gl_eudr_sgv_storage_zones` | regular | Storage zone definitions per facility |
| `gl_eudr_sgv_storage_events` | hypertable (monthly) | Storage material movement events |
| `gl_eudr_sgv_transport_vehicles` | regular | Transport vehicle/container registry |
| `gl_eudr_sgv_transport_verifications` | hypertable (monthly) | Transport segregation verification records |
| `gl_eudr_sgv_processing_lines` | regular | Processing line registry |
| `gl_eudr_sgv_changeover_records` | hypertable (monthly) | Line changeover/cleaning records |
| `gl_eudr_sgv_contamination_events` | hypertable (monthly) | Cross-contamination incident records |
| `gl_eudr_sgv_labels` | regular | Labeling and marking records |
| `gl_eudr_sgv_facility_assessments` | regular | Facility segregation assessment results |
| `gl_eudr_sgv_batch_jobs` | regular | Batch processing jobs |
| `gl_eudr_sgv_audit_log` | regular | Immutable audit trail |

### 7.3 Prometheus Metrics (18 metrics, `gl_eudr_sgv_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_sgv_scp_validations_total` | Counter | Total segregation point validations |
| `gl_eudr_sgv_scp_failures_total` | Counter | SCP validation failures |
| `gl_eudr_sgv_storage_audits_total` | Counter | Total storage segregation audits |
| `gl_eudr_sgv_transport_checks_total` | Counter | Total transport segregation checks |
| `gl_eudr_sgv_processing_checks_total` | Counter | Total processing line checks |
| `gl_eudr_sgv_contamination_events_total` | Counter | Total contamination events detected |
| `gl_eudr_sgv_contamination_critical_total` | Counter | Critical contamination events |
| `gl_eudr_sgv_labels_verified_total` | Counter | Total label verifications |
| `gl_eudr_sgv_label_failures_total` | Counter | Label verification failures |
| `gl_eudr_sgv_assessments_total` | Counter | Total facility assessments |
| `gl_eudr_sgv_reports_generated_total` | Counter | Total reports generated |
| `gl_eudr_sgv_batch_jobs_total` | Counter | Total batch processing jobs |
| `gl_eudr_sgv_api_errors_total` | Counter | Total API errors |
| `gl_eudr_sgv_scp_validation_duration_seconds` | Histogram | SCP validation latency |
| `gl_eudr_sgv_contamination_detection_duration_seconds` | Histogram | Contamination detection latency |
| `gl_eudr_sgv_assessment_duration_seconds` | Histogram | Facility assessment latency |
| `gl_eudr_sgv_active_segregation_points` | Gauge | Currently tracked segregation points |
| `gl_eudr_sgv_avg_facility_score` | Gauge | Average facility segregation score |

### 7.4 API Endpoints (~37 endpoints)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| SCPs | POST | `/api/v1/eudr-sgv/scp` | Register segregation control point |
| | GET | `/api/v1/eudr-sgv/scp/{scp_id}` | Get SCP details |
| | PUT | `/api/v1/eudr-sgv/scp/{scp_id}` | Update SCP |
| | POST | `/api/v1/eudr-sgv/scp/validate` | Validate SCP compliance |
| | POST | `/api/v1/eudr-sgv/scp/batch-import` | Bulk SCP import |
| | POST | `/api/v1/eudr-sgv/scp/search` | Search SCPs |
| Storage | POST | `/api/v1/eudr-sgv/storage/zones` | Register storage zone |
| | GET | `/api/v1/eudr-sgv/storage/zones/{facility_id}` | Get zones for facility |
| | POST | `/api/v1/eudr-sgv/storage/events` | Record storage event |
| | POST | `/api/v1/eudr-sgv/storage/audit` | Run storage segregation audit |
| | GET | `/api/v1/eudr-sgv/storage/score/{facility_id}` | Get storage segregation score |
| Transport | POST | `/api/v1/eudr-sgv/transport/vehicles` | Register transport vehicle |
| | GET | `/api/v1/eudr-sgv/transport/vehicles/{vehicle_id}` | Get vehicle details |
| | POST | `/api/v1/eudr-sgv/transport/verify` | Verify transport segregation |
| | POST | `/api/v1/eudr-sgv/transport/cleaning` | Record cleaning verification |
| | GET | `/api/v1/eudr-sgv/transport/history/{vehicle_id}` | Get vehicle cargo history |
| Processing | POST | `/api/v1/eudr-sgv/processing/lines` | Register processing line |
| | GET | `/api/v1/eudr-sgv/processing/lines/{line_id}` | Get processing line details |
| | POST | `/api/v1/eudr-sgv/processing/changeover` | Record line changeover |
| | POST | `/api/v1/eudr-sgv/processing/verify` | Verify processing segregation |
| | GET | `/api/v1/eudr-sgv/processing/score/{facility_id}` | Get processing segregation score |
| Contamination | POST | `/api/v1/eudr-sgv/contamination/detect` | Run contamination detection |
| | POST | `/api/v1/eudr-sgv/contamination/events` | Record contamination event |
| | GET | `/api/v1/eudr-sgv/contamination/events/{event_id}` | Get contamination event |
| | POST | `/api/v1/eudr-sgv/contamination/impact` | Assess contamination impact |
| | GET | `/api/v1/eudr-sgv/contamination/heatmap/{facility_id}` | Get risk heatmap data |
| Labels | POST | `/api/v1/eudr-sgv/labels` | Record label verification |
| | GET | `/api/v1/eudr-sgv/labels/{scp_id}` | Get labels for SCP |
| | POST | `/api/v1/eudr-sgv/labels/audit` | Run labeling compliance audit |
| Assessment | POST | `/api/v1/eudr-sgv/assessment` | Run facility assessment |
| | GET | `/api/v1/eudr-sgv/assessment/{facility_id}` | Get latest assessment |
| | GET | `/api/v1/eudr-sgv/assessment/history/{facility_id}` | Get assessment history |
| Reports | POST | `/api/v1/eudr-sgv/reports/audit` | Generate segregation audit report |
| | POST | `/api/v1/eudr-sgv/reports/contamination` | Generate contamination report |
| | POST | `/api/v1/eudr-sgv/reports/evidence` | Generate regulatory evidence package |
| | GET | `/api/v1/eudr-sgv/reports/{report_id}` | Get report |
| | GET | `/api/v1/eudr-sgv/reports/{report_id}/download` | Download report |
| Batch | POST | `/api/v1/eudr-sgv/batch` | Submit batch job |
| Health | GET | `/api/v1/eudr-sgv/health` | Health check |

---

## 8. Test Strategy

### 8.1 Unit Tests (500+)
- Segregation point registration and validation for all 8 methods
- Storage zone mapping with 12 storage types and barrier validation
- Transport vehicle registration with dedicated/shared status and cleaning verification
- Processing line verification with changeover protocol validation
- Cross-contamination detection for all 10 pathway types
- Contamination impact propagation through batch chains
- Labeling verification for all 8 label types
- Facility assessment scoring with all 5 criteria categories
- Report generation in all 4 formats
- SCP compliance scoring edge cases
- Temporal and spatial proximity analysis
- Equipment sharing risk assessment
- Batch status downgrade on contamination
- Risk classification and heatmap data generation

### 8.2 Performance Tests
- Bulk SCP validation for 10,000 segregation points
- Contamination detection across 1,000-facility supply chain
- Facility assessment batch processing for 500 facilities

---

## Appendices

### A. Segregation Methods by Risk Level

| Method | Risk Level | Physical Separation | Documentation Required |
|--------|------------|--------------------|-----------------------|
| Dedicated facility | Low | Complete -- separate building | Facility designation certificate |
| Separate building | Low | Complete -- separate structure | Building assignment record |
| Locked area | Low-Medium | Physical barrier with access control | Lock logs, key register |
| Physical barrier | Medium | Wall, fence, or permanent partition | Barrier inspection record |
| Sealed container | Medium | Sealed drums, bags, or containers | Seal number register |
| Color coded zone | Medium-High | Floor markings, signage, visual cues | Zone map, photo evidence |
| Dedicated line | Medium | Separate processing equipment | Line designation record |
| Temporal separation | High | Time-based separation only | Changeover logs, cleaning records |

### B. Cleaning Protocols by Transport Type

| Transport Type | Minimum Cleaning Method | Duration | Verification |
|---------------|------------------------|----------|-------------|
| Bulk truck | Power wash + inspection | 2 hours | Visual + swab test |
| Container truck | Sweep + wash | 1 hour | Visual inspection |
| Tanker | Steam clean + flush | 4 hours | Residue test |
| Dry bulk vessel | Hold cleaning + inspection | 8 hours | Surveyor certificate |
| Container vessel | Container cleaning | 2 hours | Seal verification |
| Tanker vessel | Tank washing + purge | 12 hours | Surveyor certificate |
| Rail hopper | Compressed air + wash | 3 hours | Visual + residue test |
| Rail container | Sweep + wash | 1 hour | Visual inspection |
| Barge | Hold cleaning + inspection | 6 hours | Surveyor certificate |
| Air freight | Container cleaning | 1 hour | Visual inspection |

### C. Facility Assessment Scoring Weights

| Category | Weight | Sub-criteria |
|----------|--------|-------------|
| Layout (physical) | 30% | Zone separation, barrier quality, access control, space adequacy |
| Protocols (SOPs) | 25% | Written procedures, staff training, cleaning schedules, inspection frequency |
| Historical performance | 20% | Contamination incidents, audit findings, corrective actions, near-misses |
| Labeling compliance | 15% | Label coverage, readability, accuracy, consistency |
| Documentation quality | 10% | Record completeness, timeliness, accessibility, retention |

### D. Contamination Severity Classification

| Severity | Definition | Impact | Required Action |
|----------|-----------|--------|----------------|
| Critical | Physical mixing of compliant and non-compliant material confirmed | Entire batch loses SG status | Immediate batch quarantine; root cause investigation; CA notification |
| Major | High probability of contamination based on evidence | Batch at risk; SG claim suspended | Batch testing; enhanced monitoring; corrective action plan |
| Minor | Low probability but gap in segregation controls identified | No immediate batch impact | Corrective action within 30 days; monitoring enhanced |
| Observation | Potential for contamination if controls not maintained | Preventive action recommended | Documented recommendation; tracked to closure |

### E. Labeling Color Code Standard

| Color | Meaning | Application |
|-------|---------|------------|
| Green | EUDR-compliant / deforestation-free | Compliant storage zones, containers, labels |
| Red | Non-compliant / unknown status | Non-compliant storage zones, containers |
| Yellow | Pending verification / in assessment | Material awaiting compliance determination |
| Blue | Buffer zone / transition area | Areas between compliant and non-compliant zones |
| White | Neutral / general | Non-commodity areas, offices, general storage |
