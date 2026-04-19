# PRD: AGENT-EUDR-009 -- Chain of Custody Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-009 |
| **Agent ID** | GL-EUDR-COC-009 |
| **Component** | Chain of Custody Agent |
| **Category** | EUDR Regulatory Agent -- Traceability & Custody Verification |
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

EUDR Article 9 requires every operator placing regulated commodities on the EU market to provide "the geolocation of all plots of land where the relevant commodities that the product contains, or was made using, were produced." Article 4 further mandates that operators verify every product is **deforestation-free** and **legally produced**. This requires an unbroken chain of custody from the production plot through every transfer, transformation, aggregation, and split until the final product enters the EU market.

In practice, commodity chains involve complex physical transformations where traceability is routinely lost:

- **Batch aggregation**: Cocoa beans from 200+ smallholder farms are mixed at a cooperative collection point, destroying individual plot traceability unless mass-balance or segregation protocols are enforced.
- **Processing transformations**: Raw palm fruit bunches become crude palm oil, then refined palm oil, then oleochemicals -- each step changes the physical form, yield ratios, and often the custody holder.
- **Batch splitting**: A 20-tonne container of coffee is split into 5 lots for different buyers, each requiring proportional allocation of origin plot data.
- **Multi-commodity blending**: A single chocolate bar contains cocoa butter (from one supply chain), cocoa liquor (from another), soy lecithin (from a third), and palm oil (from a fourth).
- **Temporal gaps**: Products stored in warehouses for months before shipment, creating gaps in the custody timeline.
- **Documentation gaps**: Bills of lading reference shipment IDs, not plot-level geolocation; certificates of origin cover country, not GPS coordinates.

Without a robust chain-of-custody system, operators cannot link their final products back to specific production plots as required by Article 9, resulting in non-compliant DDS submissions and penalties of up to 4% of annual EU turnover.

### 1.2 Solution Overview

Agent-EUDR-009: Chain of Custody Agent provides end-to-end custody chain tracking from production plot to EU market entry. It implements four internationally recognized chain-of-custody models (Identity Preserved, Segregated, Mass Balance, and Controlled Blending) and tracks every physical transformation, transfer of ownership, batch operation (split/merge/blend), and document handoff across the entire supply chain.

Core capabilities:

1. **Custody event tracking** -- Records every change of custody (transfer, receipt, storage, processing) with timestamps, locations, actors, and document references.
2. **Batch lifecycle management** -- Tracks batches from creation (harvest/collection) through aggregation, splitting, transformation, and final delivery with full genealogy.
3. **CoC model enforcement** -- Implements Identity Preserved, Segregated, Mass Balance, and Controlled Blending models per commodity with automatic validation of model-specific rules.
4. **Mass balance accounting** -- Maintains input/output mass balance ledgers per facility with configurable conversion factors, yield ratios, and loss tolerances.
5. **Transformation tracking** -- Records processing steps with input→output mappings, yield calculations, and waste/by-product tracking.
6. **Document chain verification** -- Links custody events to supporting documents (BL, packing list, phytosanitary cert, CoO, weight certificate) and validates completeness.
7. **Chain integrity verification** -- Validates that every custody chain from plot to EU market is unbroken, with no temporal gaps, missing handoffs, or unexplained volume changes.
8. **Compliance reporting** -- Generates Article 9 traceability reports linking final products to origin plots through verified custody chains.

### 1.3 Dependencies

| Dependency | Component | Integration |
|------------|-----------|-------------|
| AGENT-EUDR-001 | Supply Chain Mapping Master | Supply chain graph structure, node/edge data |
| AGENT-EUDR-002 | Geolocation Verification | Plot GPS verification status |
| AGENT-EUDR-007 | GPS Coordinate Validator | Coordinate parsing/validation |
| AGENT-EUDR-008 | Multi-Tier Supplier Tracker | Supplier profiles, relationships |
| AGENT-DATA-005 | EUDR Traceability Connector | Raw custody data intake |

---

## 2. Regulatory Context

### 2.1 EUDR Articles Addressed

| Article | Requirement | Agent Feature |
|---------|-------------|---------------|
| Art. 4 | Due diligence obligation on deforestation-free + legal production | Full custody chain verification |
| Art. 9(1)(d) | Geolocation of all production plots | Plot-to-product traceability |
| Art. 9(1)(e) | Date or time range of production | Temporal custody tracking |
| Art. 9(1)(f) | Quantity/weight of product | Mass balance accounting |
| Art. 10 | Trader obligations to identify operators | Custody transfer documentation |
| Art. 12 | Simplified due diligence for low-risk countries | CoC model selection per origin |
| Art. 14 | 5-year record retention | Immutable audit trail |
| Art. 31 | Review and reporting | Custody chain analytics |

### 2.2 Chain of Custody Standards

| Standard | Model | Description |
|----------|-------|-------------|
| ISO 22095 | All 4 | Global standard for chain of custody |
| FSC-STD-40-004 | IP, Segregated, Credit | FSC CoC certification |
| RSPO SCC | IP, Segregated, MB, B&C | RSPO supply chain certification |
| ISCC 202 | All | International Sustainability & Carbon Certification |
| UTZ/RA CoC | Segregated, MB | Rainforest Alliance chain of custody |

---

## 3. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Custody chain completeness | >= 95% of products traceable to origin | Unbroken chain ratio |
| Mass balance accuracy | <= 1% variance from expected | Input vs output accounting |
| Temporal gap detection | 100% of gaps > 24h flagged | Gap count per chain |
| Document linkage | >= 90% of events with supporting docs | Document coverage score |
| Processing throughput | >= 1,000 custody events/second | Batch processing benchmark |
| Chain verification time | < 500ms per complete chain | P95 latency |
| Test coverage | >= 500 unit tests | Pytest count |

---

## 4. Scope

### 4.1 In Scope
- All 7 EUDR commodities + derived products
- 4 CoC models (Identity Preserved, Segregated, Mass Balance, Controlled Blending)
- Custody event recording (transfer, receipt, storage, processing, export, import)
- Batch operations (create, split, merge, blend, transform)
- Mass balance ledger per facility/commodity
- Conversion factor and yield ratio management
- Document chain linking and verification
- Chain integrity validation (no gaps, no orphans, mass conservation)
- Article 9 traceability reporting
- 5-year retention per Article 14

### 4.2 Out of Scope
- Physical inspection or IoT sensor integration
- Customs clearance processing
- Financial transaction tracking
- Certification audit management (handled by EUDR-008)

---

## 5. Zero-Hallucination Principles

1. All mass balance calculations use deterministic arithmetic -- no LLM inference.
2. Conversion factors and yield ratios come from peer-reviewed/regulatory reference data.
3. Chain integrity verdicts are binary (complete/incomplete) based on verifiable event sequences.
4. Every custody event links to a document reference or is flagged as undocumented.
5. SHA-256 provenance hashing ensures tamper detection on all records.

---

## 6. Feature Requirements

### 6.1 Feature 1: Custody Event Tracking (P0)

**Requirements**:
- F1.1: Record custody events: transfer, receipt, storage_in, storage_out, processing_in, processing_out, export, import, inspection, sampling
- F1.2: Event attributes: timestamp, location (GPS + facility ID), actor (sender/receiver), batch_id, quantity, unit, document_refs
- F1.3: Event sequencing with automatic predecessor/successor linking
- F1.4: Temporal validation: events must be chronologically ordered per batch
- F1.5: Location validation: transfer sender location must match previous storage location
- F1.6: Actor validation: receiver of one event must be sender of next
- F1.7: Custody gap detection: flag temporal gaps exceeding configurable threshold (default 72h)
- F1.8: Event amendment with immutable audit trail (original preserved, amendment linked)
- F1.9: Bulk event import from EDI/XML/CSV sources
- F1.10: Real-time event streaming support (webhook/SSE notifications)

### 6.2 Feature 2: Batch Lifecycle Management (P0)

**Requirements**:
- F2.1: Batch creation from harvest/collection with origin plot linkage
- F2.2: Batch attributes: batch_id, commodity, quantity, unit, origin_plots[], production_date, quality_grade
- F2.3: Batch splitting: divide batch into N sub-batches with proportional origin allocation
- F2.4: Batch merging: combine M batches into one with combined origin tracking
- F2.5: Batch blending: combine batches of different commodities/origins with percentage tracking
- F2.6: Batch genealogy: full parent→child tree with unlimited depth
- F2.7: Batch status tracking: created, in_transit, at_facility, processing, processed, dispatched, delivered, consumed
- F2.8: Quantity tracking through every operation with loss/waste recording
- F2.9: Origin plot percentage allocation: track what % of a batch comes from each origin plot
- F2.10: Batch search by origin plot, commodity, date range, actor, location

### 6.3 Feature 3: Chain of Custody Model Enforcement (P0)

**Requirements**:
- F3.1: Identity Preserved (IP): physical separation from all other sources; 100% single-origin traceability
- F3.2: Segregated (SG): compliant material kept separate from non-compliant; batch mixing within compliant pool allowed
- F3.3: Mass Balance (MB): compliant and non-compliant mixed physically; accounting-based tracking ensures volume equivalence over credit period
- F3.4: Controlled Blending (CB): defined maximum blend ratio of compliant vs non-compliant material
- F3.5: Model assignment per commodity-facility combination
- F3.6: Model-specific validation rules (IP: no mixing; SG: only compliant sources; MB: credit period limits; CB: ratio caps)
- F3.7: Model transition tracking: upgrade from MB→SG→IP with timeline
- F3.8: Model compliance scoring per facility
- F3.9: Cross-model handoff validation (e.g., IP material entering MB facility becomes MB)
- F3.10: Certification standard linkage (FSC, RSPO, ISCC) per model

### 6.4 Feature 4: Mass Balance Accounting (P0)

**Requirements**:
- F4.1: Input/output ledger per facility per commodity per credit period
- F4.2: Credit period management: 3-month (RSPO), 12-month (FSC), or custom periods
- F4.3: Input recording: quantity, source batch, origin compliance status, date
- F4.4: Output recording: quantity, destination batch, allocated compliance status, date
- F4.5: Running balance calculation: inputs - outputs = current balance (must be >= 0)
- F4.6: Conversion factor application: raw→processed yield ratios per commodity per process type
- F4.7: Loss/waste accounting: acceptable loss percentages per commodity per process
- F4.8: Carry-forward rules: balance at period end carries to next period (with configurable expiry)
- F4.9: Overdraft detection: output exceeding available compliant input
- F4.10: Period-end reconciliation with variance reporting

### 6.5 Feature 5: Transformation Tracking (P0)

**Requirements**:
- F5.1: Processing step recording: input batches, output batches, process type, facility, timestamp
- F5.2: 25+ process types: drying, fermentation, roasting, milling, refining, pressing, extraction, fractionation, deodorization, hydrogenation, smelting, sawing, tanning, spinning, weaving, etc.
- F5.3: Input→output mapping with quantity allocation (one-to-many, many-to-one, many-to-many)
- F5.4: Yield ratio validation: actual yield vs expected yield per process type per commodity
- F5.5: By-product tracking: main product + by-products + waste from each process
- F5.6: Multi-step transformation chains: raw→intermediate→final product with full lineage
- F5.7: Derived product tracking: when commodity changes form (e.g., palm fruit → CPO → RBD palm oil → soap)
- F5.8: Co-product allocation: when one process produces multiple valuable outputs
- F5.9: Quality change tracking through transformations
- F5.10: Cross-commodity transformations (e.g., cattle → leather + beef)

### 6.6 Feature 6: Document Chain Verification (P0)

**Requirements**:
- F6.1: 15+ document types: bill_of_lading, packing_list, commercial_invoice, certificate_of_origin, phytosanitary_cert, weight_cert, quality_cert, customs_declaration, transport_waybill, warehouse_receipt, fumigation_cert, insurance_cert, dds_reference, delivery_note, purchase_order
- F6.2: Document linkage to custody events with many-to-many support
- F6.3: Document completeness scoring per custody chain
- F6.4: Required document validation per event type (e.g., export requires BL + phyto + CoO)
- F6.5: Document metadata: issuer, date, reference number, validity dates
- F6.6: Document gap detection: events without required supporting documents
- F6.7: Cross-reference validation: document quantities match event quantities
- F6.8: Document expiry monitoring with alerts
- F6.9: Document hash registration for tamper detection
- F6.10: DDS document package assembly for EU Information System submission

### 6.7 Feature 7: Chain Integrity Verification (P0)

**Requirements**:
- F7.1: End-to-end chain validation: every product traceable to origin plot(s)
- F7.2: Temporal continuity: no gaps exceeding threshold in custody timeline
- F7.3: Actor continuity: every transfer receiver matches next event sender
- F7.4: Location continuity: goods cannot teleport between facilities
- F7.5: Mass conservation: total output <= total input + tolerance per chain
- F7.6: Origin preservation: origin plot allocation percentages sum to 100%
- F7.7: Orphan detection: batches with no upstream origin or no downstream destination
- F7.8: Circular dependency detection in batch genealogy
- F7.9: Chain completeness score: 0-100 composite
- F7.10: Chain verification certificate with evidence compilation

### 6.8 Feature 8: Compliance Reporting (P0)

**Requirements**:
- F8.1: Article 9 traceability report: product → custody chain → origin plots with GPS
- F8.2: Mass balance period report per facility per commodity
- F8.3: Chain integrity report with gap analysis
- F8.4: Document completeness report
- F8.5: Report formats: JSON, PDF, CSV, EUDR XML
- F8.6: DDS submission data package (Article 9 fields)
- F8.7: Competent authority audit report (Article 14)
- F8.8: Batch genealogy report (full tree from any node)

---

## 7. Technical Requirements

### 7.1 Architecture

```
greenlang/agents/eudr/chain_of_custody/
    __init__.py                          # Package exports (80+ symbols)
    config.py                            # ChainOfCustodyConfig singleton
    models.py                            # Pydantic v2 models, enums
    provenance.py                        # SHA-256 chain hashing
    metrics.py                           # Prometheus metrics (gl_eudr_coc_ prefix)
    custody_event_tracker.py             # Engine 1: Event recording & sequencing
    batch_lifecycle_manager.py           # Engine 2: Batch CRUD & genealogy
    coc_model_enforcer.py                # Engine 3: IP/SG/MB/CB enforcement
    mass_balance_engine.py               # Engine 4: Ledger accounting
    transformation_tracker.py            # Engine 5: Processing steps
    document_chain_verifier.py           # Engine 6: Document linking
    chain_integrity_verifier.py          # Engine 7: End-to-end validation
    compliance_reporter.py               # Engine 8: Reports & DDS
    setup.py                             # ChainOfCustodyService facade
    reference_data/
        __init__.py
        conversion_factors.py            # Commodity conversion/yield ratios
        document_requirements.py         # Required docs per event type
        coc_model_rules.py               # CoC model validation rules
    api/
        __init__.py
        router.py
        schemas.py
        dependencies.py
        event_routes.py
        batch_routes.py
        model_routes.py
        balance_routes.py
        report_routes.py
        verification_routes.py
```

### 7.2 Database Schema (V097)

| Table | Type | Description |
|-------|------|-------------|
| `gl_eudr_coc_custody_events` | hypertable (monthly) | Custody event records |
| `gl_eudr_coc_batches` | regular | Batch master records |
| `gl_eudr_coc_batch_operations` | hypertable (monthly) | Split/merge/blend/transform records |
| `gl_eudr_coc_batch_origins` | regular | Origin plot allocations per batch |
| `gl_eudr_coc_mass_balance_ledger` | hypertable (monthly) | Input/output mass balance entries |
| `gl_eudr_coc_transformations` | hypertable (monthly) | Processing step records |
| `gl_eudr_coc_documents` | regular | Document metadata and linkage |
| `gl_eudr_coc_chain_verifications` | regular | Chain integrity verification results |
| `gl_eudr_coc_batch_jobs` | regular | Batch processing jobs |
| `gl_eudr_coc_audit_log` | regular | Immutable audit trail |

### 7.3 Prometheus Metrics (18 metrics, `gl_eudr_coc_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_coc_events_recorded_total` | Counter | Total custody events recorded |
| `gl_eudr_coc_batches_created_total` | Counter | Total batches created |
| `gl_eudr_coc_batch_operations_total` | Counter | Total batch split/merge/blend ops |
| `gl_eudr_coc_mass_balance_entries_total` | Counter | Total mass balance ledger entries |
| `gl_eudr_coc_transformations_total` | Counter | Total transformation records |
| `gl_eudr_coc_documents_linked_total` | Counter | Total documents linked |
| `gl_eudr_coc_verifications_total` | Counter | Total chain verifications |
| `gl_eudr_coc_verification_failures_total` | Counter | Chain verification failures |
| `gl_eudr_coc_reports_generated_total` | Counter | Total reports generated |
| `gl_eudr_coc_mass_balance_overdrafts_total` | Counter | Mass balance overdraft detections |
| `gl_eudr_coc_custody_gaps_total` | Counter | Custody temporal gaps detected |
| `gl_eudr_coc_batch_jobs_total` | Counter | Total batch processing jobs |
| `gl_eudr_coc_event_recording_duration_seconds` | Histogram | Event recording latency |
| `gl_eudr_coc_verification_duration_seconds` | Histogram | Chain verification latency |
| `gl_eudr_coc_mass_balance_duration_seconds` | Histogram | Mass balance calc latency |
| `gl_eudr_coc_active_batches` | Gauge | Currently active batches |
| `gl_eudr_coc_chain_completeness_avg` | Gauge | Average chain completeness score |
| `gl_eudr_coc_api_errors_total` | Counter | Total API errors |

### 7.4 API Endpoints (~34 endpoints)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| Events | POST | `/api/v1/eudr-coc/events` | Record custody event |
| | POST | `/api/v1/eudr-coc/events/batch` | Bulk event import |
| | GET | `/api/v1/eudr-coc/events/{event_id}` | Get event details |
| | GET | `/api/v1/eudr-coc/events/chain/{batch_id}` | Get event chain for batch |
| | POST | `/api/v1/eudr-coc/events/amend/{event_id}` | Amend event (immutable) |
| Batches | POST | `/api/v1/eudr-coc/batches` | Create batch |
| | GET | `/api/v1/eudr-coc/batches/{batch_id}` | Get batch details |
| | POST | `/api/v1/eudr-coc/batches/split` | Split batch |
| | POST | `/api/v1/eudr-coc/batches/merge` | Merge batches |
| | POST | `/api/v1/eudr-coc/batches/blend` | Blend batches |
| | GET | `/api/v1/eudr-coc/batches/{batch_id}/genealogy` | Get batch genealogy |
| | POST | `/api/v1/eudr-coc/batches/search` | Search batches |
| Models | POST | `/api/v1/eudr-coc/models/assign` | Assign CoC model to facility |
| | GET | `/api/v1/eudr-coc/models/{facility_id}` | Get facility CoC model |
| | POST | `/api/v1/eudr-coc/models/validate` | Validate against CoC model |
| | GET | `/api/v1/eudr-coc/models/compliance/{facility_id}` | Get model compliance score |
| Balance | POST | `/api/v1/eudr-coc/balance/input` | Record mass balance input |
| | POST | `/api/v1/eudr-coc/balance/output` | Record mass balance output |
| | GET | `/api/v1/eudr-coc/balance/{facility_id}` | Get current balance |
| | POST | `/api/v1/eudr-coc/balance/reconcile` | Reconcile period balance |
| | GET | `/api/v1/eudr-coc/balance/history/{facility_id}` | Get balance history |
| Transform | POST | `/api/v1/eudr-coc/transform` | Record transformation |
| | POST | `/api/v1/eudr-coc/transform/batch` | Batch transformation import |
| | GET | `/api/v1/eudr-coc/transform/{transform_id}` | Get transformation details |
| Documents | POST | `/api/v1/eudr-coc/documents` | Link document to event |
| | GET | `/api/v1/eudr-coc/documents/{batch_id}` | Get documents for batch |
| | POST | `/api/v1/eudr-coc/documents/validate` | Validate document chain |
| Verify | POST | `/api/v1/eudr-coc/verify/chain` | Verify complete custody chain |
| | POST | `/api/v1/eudr-coc/verify/batch` | Batch verification |
| | GET | `/api/v1/eudr-coc/verify/{verification_id}` | Get verification result |
| Reports | POST | `/api/v1/eudr-coc/reports/traceability` | Article 9 traceability report |
| | POST | `/api/v1/eudr-coc/reports/mass-balance` | Mass balance period report |
| | GET | `/api/v1/eudr-coc/reports/{report_id}` | Get report |
| | GET | `/api/v1/eudr-coc/reports/{report_id}/download` | Download report |
| Batch | POST | `/api/v1/eudr-coc/batch` | Submit batch job |
| | DELETE | `/api/v1/eudr-coc/batch/{batch_id}` | Cancel batch job |
| Health | GET | `/api/v1/eudr-coc/health` | Health check |

---

## 8. Test Strategy

### 8.1 Unit Tests (500+)
- Custody event recording with all 10 event types
- Batch lifecycle (create, split, merge, blend) with origin preservation
- CoC model enforcement for all 4 models with edge cases
- Mass balance accounting with conversion factors and loss tolerances
- Transformation tracking with yield ratio validation
- Document chain completeness scoring
- Chain integrity verification with gap/orphan/mass detection
- Report generation in all 4 formats
- Temporal, actor, and location continuity checks
- Batch genealogy tree construction for complex topologies

### 8.2 Performance Tests
- Batch import of 100,000 custody events
- Chain verification for 10,000-event chains
- Mass balance reconciliation across 1,000 facilities

---

## Appendices

### A. Conversion Factors by Commodity

| Commodity | Process | Input | Output | Yield Ratio | Source |
|-----------|---------|-------|--------|-------------|--------|
| Cocoa | Beans → Nibs | Beans | Nibs | 0.87 | ICCO |
| Cocoa | Nibs → Liquor | Nibs | Liquor | 0.80 | ICCO |
| Cocoa | Liquor → Butter | Liquor | Butter | 0.45 | ICCO |
| Cocoa | Liquor → Powder | Liquor | Powder | 0.55 | ICCO |
| Palm oil | FFB → CPO | FFB | CPO | 0.20-0.23 | MPOB |
| Palm oil | FFB → PKO | FFB | PKO | 0.03-0.04 | MPOB |
| Palm oil | CPO → RBD | CPO | RBD PO | 0.92 | PORAM |
| Coffee | Cherry → Green | Cherry | Green | 0.17-0.20 | ICO |
| Coffee | Green → Roasted | Green | Roasted | 0.80-0.85 | SCA |
| Soya | Beans → Oil | Beans | Oil | 0.18-0.20 | NOPA |
| Soya | Beans → Meal | Beans | Meal | 0.79-0.80 | NOPA |
| Rubber | Latex → Sheet | Latex | RSS | 0.30-0.35 | IRSG |
| Wood | Log → Sawn Timber | Log | Sawn | 0.45-0.55 | ITTO |
| Cattle | Live → Carcass | Live | Carcass | 0.52-0.58 | FAO |
| Cattle | Carcass → Leather | Live | Hide | 0.06-0.08 | ICHSLTA |

### B. Required Documents per Event Type

| Event Type | Required Documents | Optional Documents |
|------------|-------------------|--------------------|
| transfer | commercial_invoice, delivery_note | transport_waybill |
| export | bill_of_lading, phytosanitary_cert, certificate_of_origin, customs_declaration | fumigation_cert, insurance_cert |
| import | bill_of_lading, customs_declaration | commercial_invoice |
| processing_in | weight_cert, quality_cert | warehouse_receipt |
| processing_out | weight_cert | quality_cert |
| storage_in | warehouse_receipt, weight_cert | |
| storage_out | delivery_note, weight_cert | |
| inspection | quality_cert | sampling record |

### C. CoC Model Rules Summary

| Model | Mixing Allowed | Accounting | Credit Period | Certification |
|-------|---------------|------------|---------------|---------------|
| Identity Preserved | None -- 100% single origin | Physical tracking | N/A | FSC, RSPO |
| Segregated | Within compliant pool only | Physical tracking | N/A | FSC, RSPO, ISCC |
| Mass Balance | Physical mixing OK | Accounting-based | 3-12 months | RSPO, ISCC, UTZ |
| Controlled Blending | Up to defined ratio | Ratio-based | N/A | RSPO (B&C) |

### D. Risk Category Weights

| Category | Weight | Description |
|----------|--------|-------------|
| custody_gap | 0.25 | Temporal gaps in custody chain |
| document_gap | 0.20 | Missing supporting documents |
| mass_variance | 0.20 | Mass balance variance |
| origin_coverage | 0.20 | Percentage of product with origin traceability |
| actor_verification | 0.15 | Unverified actors in chain |
