# PRD: AGENT-EUDR-014 -- QR Code Generator Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-014 |
| **Agent ID** | GL-EUDR-QRG-014 |
| **Component** | QR Code Generator Agent |
| **Category** | EUDR Regulatory Agent -- Product Labeling & Physical-Digital Bridge |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-09 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-09 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation requires that every product placed on the EU market can be traced back to its source plot of land (Article 9) and that competent authorities, customs officials, and downstream buyers can verify a product's compliance status at any point in the supply chain. This creates a fundamental challenge: how does a physical commodity (a bag of coffee, a pallet of timber, a container of palm oil) carry its digital traceability data through the physical supply chain?

Current gaps in physical-digital traceability:

- **No standardized product-level identifier**: While DDS reference numbers exist at the declaration level, there is no standardized mechanism to link a physical product unit (bag, carton, pallet, container) to its digital traceability record in the EU Information System. A customs inspector examining a shipment of cocoa butter at a port has no efficient way to query the product's full deforestation-free provenance.
- **Manual record lookup is error-prone**: When a buyer receives a shipment, they must manually cross-reference paper documents (bill of lading, certificate of origin, phytosanitary certificate) against digital records. This process is slow, error-prone, and fails to scale for high-volume commodity flows.
- **Custody transfer verification gaps**: When a product changes hands (producer → collector → processor → trader → importer), each party must verify the provenance claims of the previous party. Without a machine-readable identifier on the product itself, this verification requires phone calls, email exchanges, and manual document retrieval.
- **Customs inspection bottleneck**: EUDR Article 16 requires competent authorities to perform checks on at least 9% of operators and 9% of products from high-risk countries. Without machine-readable identifiers, each inspection requires manual database lookups that can take 15-30 minutes per product, creating port congestion and trade delays.
- **Consumer transparency demands**: While not mandated by EUDR, the EU Empowering Consumers Directive (2024) and the Digital Product Passport regulation increasingly require that consumers can access product sustainability information. QR codes on packaging are the established mechanism for this.
- **Batch splitting and merging**: When a batch of cocoa beans is split across multiple processing lines or merged with other batches, the resulting products must maintain traceability to all source batches. Physical identifiers must encode this complex many-to-many relationship.
- **Multi-language and multi-format requirements**: EUDR supply chains span 50+ countries with different alphabets, label regulations, and scanning infrastructure. Physical identifiers must work universally regardless of language or local technology capabilities.
- **Counterfeiting risk**: Without cryptographically secured identifiers, fraudulent operators can print fake compliance labels on non-compliant products, undermining the entire EUDR framework.

Without a QR code generation system integrated into the traceability platform, operators face significant compliance risks, customs delays, and inability to provide the rapid product-level verification that competent authorities require.

### 1.2 Solution Overview

Agent-EUDR-014: QR Code Generator Agent provides a production-grade QR code generation and management system that creates machine-readable identifiers linking physical EUDR-regulated products to their digital traceability records. It generates standards-compliant QR codes encoding DDS references, batch identifiers, supply chain provenance, and blockchain anchor hashes for instant verification by competent authorities, customs officials, buyers, and consumers.

Core capabilities:

1. **QR code generation engine** -- Generates high-density QR codes (versions 1-40) encoding EUDR traceability data in structured formats (JSON, GS1 Digital Link, EUDR-XML compact). Supports error correction levels L/M/Q/H for industrial environments, configurable module sizes, and embedded logos for operator branding.
2. **Data payload composer** -- Composes structured data payloads from EUDR traceability records (DDS reference, batch ID, commodity type, country of origin, deforestation-free status, certification references, blockchain anchor hash) with schema validation and size optimization.
3. **Label template engine** -- Generates print-ready labels combining QR codes with human-readable text (product name, batch number, origin country, compliance status, operator name) in configurable layouts for various label sizes (30mm to 150mm) and print technologies (thermal, inkjet, laser).
4. **Batch code generator** -- Generates unique, sequential, and verifiable product codes for batches, sub-batches, and individual product units using hierarchical encoding (batch → sub-batch → unit) with check digits for error detection.
5. **Verification URL builder** -- Constructs verification URLs that link QR codes to the GreenLang verification portal, enabling anyone with a smartphone to scan and verify a product's EUDR compliance status in real-time.
6. **Anti-counterfeiting engine** -- Embeds cryptographic signatures (HMAC-SHA256) and rotating verification tokens in QR code payloads to prevent unauthorized duplication and forgery of compliance labels.
7. **Bulk generation pipeline** -- Generates QR codes and labels in bulk (up to 100,000 per batch) for high-volume production environments with parallel processing, progress tracking, and output format options (PNG, SVG, PDF, ZPL for Zebra printers).
8. **Code lifecycle manager** -- Manages the full lifecycle of generated QR codes: creation, activation, deactivation, revocation, and expiry. Tracks scan events, geographic scan locations, and verification outcomes for analytics.

### 1.3 Dependencies

| Dependency | Component | Integration |
|------------|-----------|-------------|
| AGENT-EUDR-001 | Supply Chain Mapping Master | Supply chain node/edge data for provenance encoding |
| AGENT-EUDR-009 | Chain of Custody Agent | CoC records for custody transfer QR codes |
| AGENT-EUDR-011 | Mass Balance Calculator | Batch/sub-batch identifiers for hierarchical coding |
| AGENT-EUDR-012 | Document Authentication | Document hashes for verification payload |
| AGENT-EUDR-013 | Blockchain Integration | On-chain anchor hashes embedded in QR payloads |
| AGENT-DATA-005 | EUDR Traceability Connector | DDS reference numbers for QR encoding |

---

## 2. Regulatory Context

### 2.1 EUDR Articles Addressed

| Article | Requirement | Agent Feature |
|---------|-------------|---------------|
| Art. 4(2) | Due diligence information collection | QR codes encode due diligence data for rapid access |
| Art. 9(1)(a-g) | DDS information requirements (geolocation, commodity, quantity, supplier) | Data payload composer encodes all Art. 9 required fields |
| Art. 10(2) | Supply chain complexity assessment | Hierarchical batch codes trace multi-tier supply chains |
| Art. 12 | DDS submission reference | QR codes embed DDS reference number for EU IS lookup |
| Art. 14 | 5-year record retention | Code lifecycle manager maintains 5-year QR code history |
| Art. 16(10) | Competent authority checks at 9% rate | QR scanning enables rapid inspection (< 30 seconds per product) |
| Art. 22 | Customs authorities verification | QR codes enable instant customs verification at ports |
| Art. 29 | Country benchmarking | QR payloads include country risk classification |

### 2.2 Supported QR Code Formats

| Format | Version Range | Capacity | Use Case |
|--------|--------------|----------|----------|
| QR Code (ISO 18004) | 1-40 | Up to 4,296 alphanumeric chars | Standard product labeling |
| Micro QR (ISO 18004) | M1-M4 | Up to 35 numeric chars | Small items, jewelry |
| GS1 Digital Link QR | Standard | URL + structured data | Retail and consumer-facing |
| Data Matrix (ISO 16022) | ECC 200 | Up to 2,335 alphanumeric chars | Industrial/logistics labels |

### 2.3 QR Code Content Types

| Content Type | Description | Target Audience | Data Size |
|-------------|-------------|-----------------|-----------|
| Full Traceability | Complete DDS reference, all origins, all certs | Competent authorities | 500-2000 bytes |
| Compact Verification | DDS ref + verification URL + HMAC | Customs inspectors | 100-300 bytes |
| Consumer Summary | Product name, origin, status, scan URL | End consumers | 50-150 bytes |
| Batch Identifier | Batch ID + sub-batch + unit code + check digit | Internal logistics | 30-80 bytes |
| Blockchain Anchor | Anchor hash + tx hash + Merkle proof ref | Auditors | 200-500 bytes |

---

## 3. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| QR generation speed | < 50ms per code | Single QR generation benchmark |
| Bulk generation throughput | >= 10,000 QR codes/minute | Bulk generation benchmark |
| QR scan success rate | >= 99.5% first-scan success | Test with 5 scanner types |
| Label generation speed | < 200ms per label | Label with QR + text |
| Anti-counterfeit detection | 100% detection of forged codes | HMAC verification test |
| Verification URL response | < 2 seconds to status | End-to-end scan-to-verify |
| Code lifecycle accuracy | 100% status tracking | Status query test |
| Supported output formats | >= 4 formats (PNG, SVG, PDF, ZPL) | Format coverage |
| GS1 Digital Link compliance | 100% valid GS1 URIs | GS1 validation test |
| Test coverage | >= 500 unit tests | Pytest count |

---

## 4. Scope

### 4.1 In Scope
- QR code generation (ISO 18004) with all error correction levels
- Data Matrix generation (ISO 16022) for industrial labels
- GS1 Digital Link URI construction
- Structured data payload composition from EUDR traceability records
- Print-ready label generation (PNG, SVG, PDF, ZPL)
- Bulk generation pipeline with parallel processing
- Anti-counterfeiting via HMAC-SHA256 signatures
- Batch code generation with hierarchical encoding
- Verification URL construction
- Code lifecycle management (create, activate, deactivate, revoke)
- Scan event tracking and analytics
- 5-year code history retention per EUDR Article 14

### 4.2 Out of Scope
- Physical label printing hardware integration (outputs print-ready files)
- Barcode scanning hardware SDKs
- Consumer-facing mobile application
- RFID/NFC tag generation
- 1D barcode generation (EAN-13, UPC-A, Code 128)
- Label design GUI/WYSIWYG editor
- Printer driver management
- Physical label adhesive/material selection

---

## 5. Zero-Hallucination Principles

1. QR code encoding uses deterministic ISO 18004 algorithms -- same input data produces identical QR matrix.
2. GS1 Digital Link URIs are constructed using deterministic string formatting rules per GS1 General Specifications.
3. HMAC-SHA256 anti-counterfeiting signatures use deterministic cryptographic functions -- no LLM inference.
4. Batch code check digits use Luhn or ISO 7064 Mod 11,10 algorithms -- deterministic computation.
5. Label layout uses deterministic coordinate-based positioning -- no dynamic reflowing.
6. Color encoding uses exact hex values -- no approximate color matching.
7. Error correction level selection is rule-based (environment mapping) -- no probabilistic estimation.
8. SHA-256 provenance hashing ensures tamper detection on all generated codes and payloads.

---

## 6. Feature Requirements

### 6.1 Feature 1: QR Code Generation Engine (P0)

**Requirements**:
- F1.1: Generate QR codes compliant with ISO 18004:2015 (versions 1-40)
- F1.2: Support 4 error correction levels: L (7%), M (15%), Q (25%), H (30%)
- F1.3: Output formats: PNG (raster), SVG (vector), PDF (print), ZPL (Zebra printers), EPS (professional print)
- F1.4: Configurable module size (1-20 pixels), quiet zone width, foreground/background colors
- F1.5: Embedded logo support: center logo with automatic error correction level upgrade
- F1.6: Data Matrix generation (ISO 16022 ECC 200) as alternative symbology
- F1.7: Micro QR code generation for small-form-factor labels
- F1.8: QR code versioning: automatic version selection based on data payload size
- F1.9: Quality grading: generate QR codes meeting ISO 15415 print quality grade A or B
- F1.10: DPI configuration: 72 (screen), 150 (draft), 300 (standard print), 600 (high-quality print)

### 6.2 Feature 2: Data Payload Composer (P0)

**Requirements**:
- F2.1: Compose structured payloads from EUDR traceability records
- F2.2: Payload schemas: full_traceability, compact_verification, consumer_summary, batch_identifier, blockchain_anchor
- F2.3: Field mapping: DDS reference number, batch ID, commodity type (7 EUDR commodities), country of origin (ISO 3166-1), HS code, deforestation-free status, certification references, geolocation hash, blockchain anchor hash
- F2.4: Payload size optimization: compress payloads using zlib when exceeding QR capacity thresholds
- F2.5: Schema validation: validate all payload fields against EUDR data requirements
- F2.6: GS1 Digital Link URI construction per GS1 General Specifications 22.0
- F2.7: Multi-language support: payload text in 24 EU official languages
- F2.8: Payload versioning: version header for forward-compatible parsing
- F2.9: Custom field injection: operator-defined additional fields within size constraints
- F2.10: Payload encryption: optional AES-256-GCM encryption for sensitive fields (accessible only with key)

### 6.3 Feature 3: Label Template Engine (P0)

**Requirements**:
- F3.1: Generate print-ready labels combining QR code with human-readable text
- F3.2: Pre-defined templates: product_label (50x30mm), shipping_label (100x150mm), pallet_label (148x210mm A5), container_label (297x210mm A4), consumer_label (30x20mm)
- F3.3: Template elements: QR code, product name, batch number, origin country (with flag icon), compliance status badge, operator name/logo, date, HS code
- F3.4: Custom template support: user-defined layouts with positioned elements
- F3.5: Multi-QR labels: labels with multiple QR codes (traceability + consumer + batch)
- F3.6: Label output formats: PNG, SVG, PDF (multi-page for bulk), ZPL (Zebra thermal printers)
- F3.7: Font configuration: embedded fonts for multi-language text rendering
- F3.8: Color schemes: EUDR green (compliant), EUDR amber (pending), EUDR red (non-compliant)
- F3.9: Serial number placement: configurable position for human-readable serial codes
- F3.10: Bleed and safe area: print production bleed margins and safe areas

### 6.4 Feature 4: Batch Code Generator (P0)

**Requirements**:
- F4.1: Generate unique batch codes with configurable format (alphanumeric, numeric, custom prefix)
- F4.2: Hierarchical encoding: batch → sub-batch → unit (e.g., BATCH-2026-001 / SUB-003 / UNIT-00042)
- F4.3: Check digit algorithms: Luhn, ISO 7064 Mod 11,10, CRC-8 for error detection
- F4.4: Sequential numbering with configurable start, increment, and padding (zero-padded)
- F4.5: Prefix templates: operator code + commodity code + year + sequence (e.g., OP01-COC-2026-00001)
- F4.6: Uniqueness guarantee: database-backed uniqueness check with retry on collision
- F4.7: Batch code reservation: reserve code ranges for future use without immediate generation
- F4.8: Code format validation: regex-based format enforcement
- F4.9: Bulk code generation: generate ranges of sequential codes (e.g., 1000 unit codes for a batch)
- F4.10: Code association: link batch codes to DDS references, supply chain nodes, and blockchain anchors

### 6.5 Feature 5: Verification URL Builder (P0)

**Requirements**:
- F5.1: Construct verification URLs pointing to the GreenLang verification portal
- F5.2: URL format: `https://verify.greenlang.eu/{operator_code}/{qr_code_id}?sig={hmac_signature}`
- F5.3: Short URL support: configurable short URL service integration for consumer-facing codes
- F5.4: URL parameters: include DDS reference, batch ID, scan timestamp, verification token
- F5.5: Deep link support: mobile app deep links for operator-specific verification apps
- F5.6: QR code ID encoding: base32-encoded unique identifier for URL-safe representation
- F5.7: Signature inclusion: HMAC-SHA256 signature truncated to 8 characters for URL inclusion
- F5.8: Expiry token: time-limited verification tokens with configurable TTL (default 5 years)
- F5.9: Language auto-detection: URL parameter for browser language preference
- F5.10: Offline verification: QR payload includes enough data for offline compliance check

### 6.6 Feature 6: Anti-Counterfeiting Engine (P0)

**Requirements**:
- F6.1: HMAC-SHA256 signature on QR code payload using operator-specific secret key
- F6.2: Rotating verification tokens: time-based tokens (TOTP-like) embedded in QR payloads
- F6.3: Digital watermarking: embed invisible patterns in QR code image for copy detection
- F6.4: Signature verification API: endpoint for verifying QR code authenticity
- F6.5: Counterfeit detection scoring: score each scan attempt for counterfeiting risk indicators
- F6.6: Key rotation: scheduled HMAC key rotation with backward-compatible verification
- F6.7: Revocation list: maintain list of revoked/invalidated QR codes for instant rejection
- F6.8: Geo-fencing: flag scans from unexpected geographic locations
- F6.9: Scan velocity detection: detect abnormally high scan rates indicating mass copying
- F6.10: Tamper-evident encoding: encode QR data such that partial modification invalidates the entire code

### 6.7 Feature 7: Bulk Generation Pipeline (P0)

**Requirements**:
- F7.1: Generate QR codes in bulk batches (up to 100,000 per job)
- F7.2: Parallel processing: configurable worker count for CPU-bound QR generation
- F7.3: Progress tracking: real-time progress percentage, ETA, items completed/failed
- F7.4: Output packaging: ZIP archive of individual QR images, or multi-page PDF
- F7.5: CSV/Excel manifest: generate accompanying manifest file mapping code IDs to images
- F7.6: Template application: apply label templates to all codes in a batch
- F7.7: Resumable jobs: resume interrupted bulk jobs from the last completed item
- F7.8: Job scheduling: queue jobs for off-peak execution
- F7.9: Memory-efficient streaming: stream output to disk without loading all images in memory
- F7.10: Output validation: verify generated QR codes are scannable before packaging

### 6.8 Feature 8: Code Lifecycle Manager (P0)

**Requirements**:
- F8.1: Track QR code status: created, active, deactivated, revoked, expired
- F8.2: Activation workflow: QR codes created in "created" status, activated when applied to product
- F8.3: Deactivation: temporarily disable QR code (e.g., product recalled)
- F8.4: Revocation: permanently invalidate QR code (e.g., counterfeiting detected)
- F8.5: Expiry: automatic expiry after configurable TTL (default 5 years per EUDR Article 14)
- F8.6: Scan event logging: record every scan with timestamp, GPS coordinates, scanner ID, outcome
- F8.7: Scan analytics: aggregate scan counts, geographic distribution, time-of-day patterns
- F8.8: Reprint tracking: track reprinted labels and link to original QR code
- F8.9: Replacement workflow: issue replacement QR codes with cross-reference to original
- F8.10: Audit trail: immutable log of all lifecycle events per QR code

---

## 7. Technical Requirements

### 7.1 Architecture

```
greenlang/agents/eudr/qr_code_generator/
    __init__.py                          # Package exports (80+ symbols)
    config.py                            # QRCodeGeneratorConfig singleton
    models.py                            # Pydantic v2 models, enums
    provenance.py                        # SHA-256 chain hashing
    metrics.py                           # Prometheus metrics (gl_eudr_qrg_ prefix)
    qr_encoder.py                       # Engine 1: QR code generation (ISO 18004)
    payload_composer.py                  # Engine 2: Data payload composition
    label_template_engine.py             # Engine 3: Label template rendering
    batch_code_generator.py              # Engine 4: Batch code generation
    verification_url_builder.py          # Engine 5: Verification URL construction
    anti_counterfeit_engine.py           # Engine 6: Anti-counterfeiting
    bulk_generation_pipeline.py          # Engine 7: Bulk generation
    code_lifecycle_manager.py            # Engine 8: Code lifecycle management
    setup.py                             # QRCodeGeneratorService facade
    reference_data/
        __init__.py
        label_templates.py               # Pre-defined label template definitions
        gs1_specifications.py            # GS1 Digital Link formatting rules
        commodity_codes.py               # EUDR commodity HS codes and identifiers
    api/
        __init__.py
        router.py
        schemas.py
        dependencies.py
        qr_routes.py                     # QR code generation endpoints
        payload_routes.py                # Payload composition endpoints
        label_routes.py                  # Label generation endpoints
        batch_code_routes.py             # Batch code endpoints
        verification_routes.py           # Verification URL endpoints
        counterfeit_routes.py            # Anti-counterfeiting endpoints
        bulk_routes.py                   # Bulk generation endpoints
        lifecycle_routes.py              # Code lifecycle endpoints
```

### 7.2 Database Schema (V102)

| Table | Type | Description |
|-------|------|-------------|
| `gl_eudr_qrg_codes` | hypertable (monthly) | Generated QR code records |
| `gl_eudr_qrg_payloads` | regular | Composed data payloads |
| `gl_eudr_qrg_labels` | regular | Generated label records |
| `gl_eudr_qrg_batch_codes` | regular | Batch/sub-batch/unit codes |
| `gl_eudr_qrg_verification_urls` | regular | Verification URL records |
| `gl_eudr_qrg_signatures` | regular | Anti-counterfeiting signatures and keys |
| `gl_eudr_qrg_scan_events` | hypertable (monthly) | Scan event records |
| `gl_eudr_qrg_bulk_jobs` | regular | Bulk generation job records |
| `gl_eudr_qrg_lifecycle_events` | hypertable (monthly) | Code lifecycle event log |
| `gl_eudr_qrg_templates` | regular | Label template definitions |
| `gl_eudr_qrg_code_associations` | regular | Code-to-DDS/batch/anchor associations |
| `gl_eudr_qrg_audit_log` | regular | Immutable audit trail |

### 7.3 Prometheus Metrics (18 metrics, `gl_eudr_qrg_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_qrg_codes_generated_total` | Counter | Total QR codes generated |
| `gl_eudr_qrg_labels_generated_total` | Counter | Total labels generated |
| `gl_eudr_qrg_payloads_composed_total` | Counter | Total payloads composed |
| `gl_eudr_qrg_batch_codes_total` | Counter | Total batch codes generated |
| `gl_eudr_qrg_verification_urls_total` | Counter | Total verification URLs created |
| `gl_eudr_qrg_scans_total` | Counter | Total scan events recorded |
| `gl_eudr_qrg_counterfeit_detections_total` | Counter | Counterfeit attempts detected |
| `gl_eudr_qrg_bulk_jobs_total` | Counter | Total bulk generation jobs |
| `gl_eudr_qrg_bulk_codes_total` | Counter | Total codes generated in bulk |
| `gl_eudr_qrg_revocations_total` | Counter | Total QR codes revoked |
| `gl_eudr_qrg_signature_verifications_total` | Counter | Total signature verifications |
| `gl_eudr_qrg_api_errors_total` | Counter | API errors |
| `gl_eudr_qrg_generation_duration_seconds` | Histogram | QR generation latency |
| `gl_eudr_qrg_label_duration_seconds` | Histogram | Label generation latency |
| `gl_eudr_qrg_bulk_duration_seconds` | Histogram | Bulk job duration |
| `gl_eudr_qrg_verification_duration_seconds` | Histogram | Signature verification latency |
| `gl_eudr_qrg_active_bulk_jobs` | Gauge | Active bulk generation jobs |
| `gl_eudr_qrg_active_codes` | Gauge | Currently active QR codes |

### 7.4 API Endpoints (~37 endpoints)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| QR | POST | `/api/v1/eudr-qrg/qr/generate` | Generate single QR code |
| | POST | `/api/v1/eudr-qrg/qr/generate/data-matrix` | Generate Data Matrix code |
| | GET | `/api/v1/eudr-qrg/qr/{code_id}` | Get QR code details |
| | GET | `/api/v1/eudr-qrg/qr/{code_id}/image` | Download QR code image |
| | GET | `/api/v1/eudr-qrg/qr/{code_id}/image/{format}` | Download in specific format |
| Payload | POST | `/api/v1/eudr-qrg/payloads/compose` | Compose data payload |
| | POST | `/api/v1/eudr-qrg/payloads/validate` | Validate payload schema |
| | GET | `/api/v1/eudr-qrg/payloads/{payload_id}` | Get payload details |
| | GET | `/api/v1/eudr-qrg/payloads/schemas` | List available schemas |
| Label | POST | `/api/v1/eudr-qrg/labels/generate` | Generate label |
| | POST | `/api/v1/eudr-qrg/labels/generate/batch` | Generate labels in batch |
| | GET | `/api/v1/eudr-qrg/labels/{label_id}` | Get label details |
| | GET | `/api/v1/eudr-qrg/labels/{label_id}/download` | Download label file |
| | GET | `/api/v1/eudr-qrg/labels/templates` | List label templates |
| BatchCode | POST | `/api/v1/eudr-qrg/batch-codes/generate` | Generate batch codes |
| | POST | `/api/v1/eudr-qrg/batch-codes/reserve` | Reserve code range |
| | GET | `/api/v1/eudr-qrg/batch-codes/{code}` | Lookup batch code |
| | GET | `/api/v1/eudr-qrg/batch-codes/{code}/hierarchy` | Get code hierarchy |
| Verify | POST | `/api/v1/eudr-qrg/verify/build-url` | Build verification URL |
| | POST | `/api/v1/eudr-qrg/verify/signature` | Verify QR code signature |
| | GET | `/api/v1/eudr-qrg/verify/{code_id}` | Get verification status |
| | POST | `/api/v1/eudr-qrg/verify/offline` | Offline verification check |
| Counterfeit | POST | `/api/v1/eudr-qrg/counterfeit/check` | Check for counterfeiting |
| | POST | `/api/v1/eudr-qrg/counterfeit/revoke/{code_id}` | Revoke counterfeit code |
| | GET | `/api/v1/eudr-qrg/counterfeit/revocation-list` | Get revocation list |
| | GET | `/api/v1/eudr-qrg/counterfeit/analytics` | Counterfeit analytics |
| Bulk | POST | `/api/v1/eudr-qrg/bulk/generate` | Submit bulk generation job |
| | GET | `/api/v1/eudr-qrg/bulk/{job_id}` | Get bulk job status |
| | GET | `/api/v1/eudr-qrg/bulk/{job_id}/download` | Download bulk output |
| | DELETE | `/api/v1/eudr-qrg/bulk/{job_id}` | Cancel bulk job |
| | GET | `/api/v1/eudr-qrg/bulk/{job_id}/manifest` | Download manifest CSV |
| Lifecycle | POST | `/api/v1/eudr-qrg/lifecycle/{code_id}/activate` | Activate QR code |
| | POST | `/api/v1/eudr-qrg/lifecycle/{code_id}/deactivate` | Deactivate QR code |
| | POST | `/api/v1/eudr-qrg/lifecycle/{code_id}/revoke` | Revoke QR code |
| | POST | `/api/v1/eudr-qrg/lifecycle/scan` | Record scan event |
| | GET | `/api/v1/eudr-qrg/lifecycle/{code_id}/history` | Get code lifecycle history |
| | GET | `/api/v1/eudr-qrg/lifecycle/{code_id}/scans` | Get scan events |
| | GET | `/api/v1/eudr-qrg/lifecycle/analytics` | Scan analytics dashboard |
| Batch | POST | `/api/v1/eudr-qrg/batch` | Submit batch processing job |
| | DELETE | `/api/v1/eudr-qrg/batch/{job_id}` | Cancel batch job |
| Health | GET | `/api/v1/eudr-qrg/health` | Health check |

---

## 8. Test Strategy

### 8.1 Unit Tests (500+)

- QR code generation for all versions (1, 5, 10, 20, 40) and error correction levels (L/M/Q/H)
- Data payload composition for all 5 content types with schema validation
- Label template rendering for all 5 pre-defined templates
- Batch code generation with all 3 check digit algorithms (Luhn, ISO 7064, CRC-8)
- Verification URL construction with HMAC signature and expiry tokens
- Anti-counterfeiting: HMAC generation/verification, key rotation, revocation list
- Bulk generation: 1, 100, 1000, 10000 codes, parallel processing, progress tracking
- Code lifecycle: all 5 status transitions (created→active→deactivated→revoked, created→expired)
- Scan event recording: timestamp, GPS, scanner ID, outcome
- GS1 Digital Link URI validation
- Edge cases: maximum payload size, empty payloads, invalid commodity codes, concurrent generation
- Deterministic output: same input produces identical QR code matrix
- Output format validation: PNG readable, SVG valid XML, PDF valid, ZPL parseable

### 8.2 Performance Tests

- QR code generation: 10,000 codes in under 60 seconds
- Label generation: 5,000 labels in under 100 seconds
- Bulk pipeline: 100,000 codes with < 1% failure rate

---

## Appendices

### A. QR Code Version-Capacity Reference

| Version | Modules | Numeric (L) | Alphanumeric (L) | Binary (L) | Numeric (H) | Alphanumeric (H) |
|---------|---------|------------|------------------|------------|------------|------------------|
| 1 | 21x21 | 41 | 25 | 17 | 17 | 10 |
| 5 | 37x37 | 154 | 93 | 64 | 60 | 36 |
| 10 | 57x57 | 395 | 240 | 164 | 174 | 106 |
| 20 | 97x97 | 1,249 | 758 | 520 | 536 | 325 |
| 40 | 177x177 | 7,089 | 4,296 | 2,953 | 3,057 | 1,852 |

### B. EUDR Commodity Codes

| Commodity | HS Code Range | EUDR Annex I Reference |
|-----------|--------------|----------------------|
| Cattle (beef/leather) | 0102, 0201-0202, 4101-4115 | Item 1 |
| Cocoa | 1801-1806 | Item 2 |
| Coffee | 0901 | Item 3 |
| Oil palm | 1511, 1513 | Item 4 |
| Rubber | 4001-4017 | Item 5 |
| Soya | 1201, 1507-1508, 2304 | Item 6 |
| Wood | 4401-4421, 4701-4813, 9401-9403 | Item 7 |

### C. Label Template Specifications

| Template | Width (mm) | Height (mm) | QR Size (mm) | Elements |
|----------|-----------|------------|-------------|----------|
| product_label | 50 | 30 | 20 | QR + product name + batch + origin |
| shipping_label | 100 | 150 | 40 | QR + recipient + batch + weight + origin + compliance badge |
| pallet_label | 148 | 210 | 60 | QR + batch details + all origins + full compliance status |
| container_label | 297 | 210 | 80 | Multi-QR + full traceability + all documents |
| consumer_label | 30 | 20 | 15 | QR + "Scan for origin" text |

### D. GS1 Digital Link URI Structure

```
https://id.gs1.org/01/{GTIN}/10/{BATCH_LOT}?3101={NET_WEIGHT}&7230={CERT_REF}&99={EUDR_DDS_REF}
```

| AI | Description | EUDR Mapping |
|----|-------------|-------------|
| 01 | Global Trade Item Number (GTIN) | Product identifier |
| 10 | Batch/Lot Number | EUDR batch code |
| 3101 | Net Weight (kg, 1 decimal) | Quantity declared |
| 7230 | Certification Reference | FSC/RSPO/ISCC certificate number |
| 99 | Custom: EUDR DDS Reference | DDS submission number |
