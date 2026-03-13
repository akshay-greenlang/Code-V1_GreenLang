# PRD: AGENT-EUDR-012 -- Document Authentication Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-012 |
| **Agent ID** | GL-EUDR-DAV-012 |
| **Component** | Document Authentication & Verification Agent |
| **Category** | EUDR Regulatory Agent -- Document Integrity & Fraud Detection |
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

The EU Deforestation Regulation requires operators and traders to collect, verify, and retain extensive documentary evidence for every due diligence statement (DDS) submitted to the EU Information System. These documents include certificates of origin, phytosanitary certificates, bills of lading, customs declarations, sustainability certificates (RSPO, FSC, ISCC, Fairtrade, UTZ/Rainforest Alliance), laboratory test reports, land title deeds, forest management plans, and operator self-declarations. Each DDS may reference dozens of supporting documents that collectively prove the commodity is deforestation-free and legally produced.

However, document fraud in commodity supply chains is pervasive and sophisticated:

- **Forged certificates of origin**: A trader fabricates a certificate of origin showing cocoa beans sourced from a low-risk country (Ghana) when the beans actually originated in a high-risk region (Ivory Coast border zone). The forged certificate uses a real template with altered issuing authority, serial number, and dates.
- **Tampered sustainability certificates**: An operator receives a legitimate RSPO certificate for 500 tonnes of Certified Sustainable Palm Oil (CSPO), then digitally alters the quantity field to 5,000 tonnes, creating 4,500 tonnes of phantom certified volume.
- **Expired certificate recycling**: A supplier submits FSC chain-of-custody certificates that expired 18 months ago but with digitally altered validity dates. Without automated expiry checking against the FSC certificate database, the expired certificates pass manual review.
- **Duplicate document reuse**: The same phytosanitary certificate is submitted against three different shipments to cover three times the volume actually inspected. Without cross-shipment document deduplication, the duplicate use goes undetected.
- **AI-generated fraudulent documents**: Advances in generative AI enable creation of realistic-looking certificates, stamps, and signatures that can pass visual inspection. Traditional manual document review cannot keep pace with AI-generated forgeries.
- **Metadata manipulation**: A PDF document is edited to change the creation date, author, and issuing authority, but the embedded metadata reveals the true creation date was days before submission -- inconsistent with the claimed issuance date months earlier.
- **Broken certificate chains**: A timber shipment is supported by an FSC certificate, but the certificate holder (Processor B) is not in the supply chain graph for that shipment. The certificate is real but applied to the wrong custody chain.
- **Cross-jurisdictional inconsistency**: A bill of lading from a Brazilian port claims export of 1,000 tonnes of soya, but the corresponding Brazilian customs declaration (Siscomex) shows only 600 tonnes. Without cross-document quantity reconciliation, the 400-tonne discrepancy goes unnoticed.

Without automated document authentication, EU operators face penalties of up to 4% of annual EU turnover for submitting DDS based on fraudulent or unverifiable documents, along with reputational damage, goods confiscation, and potential criminal liability under national implementing legislation.

### 1.2 Solution Overview

Agent-EUDR-012: Document Authentication Agent provides a production-grade, zero-hallucination document authentication and fraud detection engine for EUDR compliance. It verifies the authenticity, integrity, and validity of every document supporting a Due Diligence Statement through eight complementary verification engines.

Core capabilities:

1. **Document classification** -- Automatically classifies incoming documents into 20+ EUDR-relevant document types (certificates, invoices, bills of lading, customs declarations, sustainability certificates, laboratory reports, land titles) using deterministic rule-based classification with template matching.
2. **Digital signature verification** -- Validates PKI digital signatures (X.509, CAdES, PAdES, XAdES), PGP signatures, and qualified electronic signatures (QES) per eIDAS regulation against trusted certificate stores.
3. **Hash integrity validation** -- Computes and verifies SHA-256/SHA-512 document hashes for tamper detection, maintains hash registry for previously authenticated documents, and detects bit-level modifications.
4. **Certificate chain validation** -- Validates X.509 certificate chains from document signing certificates up to trusted root CAs, performs OCSP/CRL revocation checking, and validates certificate validity periods.
5. **Metadata extraction and validation** -- Extracts embedded metadata (creation date, modification date, author, producer application, GPS coordinates from EXIF) and cross-validates against claimed document attributes.
6. **Fraud pattern detection** -- Detects 15+ fraud patterns including duplicate document reuse, quantity tampering, date manipulation, template forgery, serial number anomalies, and cross-document inconsistencies using deterministic rule-based detection (no LLM).
7. **Cross-reference verification** -- Verifies documents against external registries (FSC certificate database, RSPO PalmTrace, ISCC certificate search, national customs databases) and cross-references quantities, dates, and parties across related documents.
8. **Compliance reporting** -- Generates document authentication reports for auditors and competent authorities, including verification status, fraud risk scores, and evidence packages.

### 1.3 Dependencies

| Dependency | Component | Integration |
|------------|-----------|-------------|
| AGENT-EUDR-001 | Supply Chain Mapping Master | Supply chain graph for cross-referencing document parties |
| AGENT-EUDR-009 | Chain of Custody Agent | CoC events and batch data for document-batch linkage |
| AGENT-EUDR-011 | Mass Balance Calculator | Quantity verification against mass balance ledger |
| AGENT-DATA-001 | PDF & Invoice Extractor | Raw document parsing and text extraction |
| AGENT-DATA-005 | EUDR Traceability Connector | DDS document requirements and submission |
| SEC-005 | Centralized Audit Logging | Immutable audit trail for all authentication events |

---

## 2. Regulatory Context

### 2.1 EUDR Articles Addressed

| Article | Requirement | Agent Feature |
|---------|-------------|---------------|
| Art. 4(2) | Due diligence -- collect and verify information | Document authentication validates collected evidence |
| Art. 9(1) | Information requirements for DDS | Metadata extraction ensures all required fields present |
| Art. 10(1) | Risk assessment | Fraud pattern detection feeds into risk assessment |
| Art. 10(2)(a) | Complexity of supply chain | Cross-reference verification validates document consistency |
| Art. 10(2)(b) | Presence of relevant legislation in country of production | Certificate chain validation verifies issuing authority legitimacy |
| Art. 10(2)(d) | Consultation of relevant sources of information | External registry verification against FSC/RSPO/ISCC databases |
| Art. 10(2)(f) | Risk of circumvention or mixing | Fraud pattern detection identifies document manipulation |
| Art. 14 | 5-year record retention | Hash registry maintains immutable document authentication records |
| Art. 16 | Risk mitigation measures | Authentication reports document verification steps taken |
| Art. 31 | Review and reporting | Compliance reports for competent authority inspections |

### 2.2 Document Types Covered

| Document Type | Abbreviation | Issuing Authority | Verification Method |
|---------------|-------------|-------------------|---------------------|
| Certificate of Origin | COO | Chamber of Commerce / Customs | Serial number + issuer + cross-reference |
| Phytosanitary Certificate | PC | National Plant Protection Org | IPPC database + serial number |
| Bill of Lading | BOL | Shipping Line | B/L number + carrier verification |
| Customs Declaration (Export) | CDE | Origin country customs | National customs database |
| Customs Declaration (Import) | CDI | EU member state customs | TARIC + national systems |
| RSPO Certificate | RSPO | RSPO Secretariat | PalmTrace database query |
| FSC Certificate | FSC | ASI-accredited certifier | FSC certificate database API |
| ISCC Certificate | ISCC | ISCC-approved auditor | ISCC certificate search |
| Fairtrade Certificate | FT | FLOCERT | Fairtrade certificate database |
| UTZ/RA Certificate | UTZ | Rainforest Alliance | RA certificate portal |
| Laboratory Test Report | LTR | Accredited laboratory | Lab accreditation + result cross-check |
| Land Title Deed | LTD | National land registry | Land registry query (where available) |
| Forest Management Plan | FMP | Forestry authority | Forestry authority cross-reference |
| Fumigation Certificate | FC | Pest control operator | Treatment record verification |
| Weight/Quality Certificate | WQC | Independent surveyor | Surveyor accreditation |
| Due Diligence Statement (Draft) | DDS | Operator (self-declared) | Completeness + consistency check |
| Supplier Self-Declaration | SSD | Supplier | Cross-reference against known data |
| Insurance Certificate | IC | Insurance company | Policy number verification |
| Transport Contract | TC | Logistics provider | Route + capacity validation |
| Warehouse Receipt | WR | Warehouse operator | Stock record cross-check |

### 2.3 Signature Standards Supported

| Standard | Format | Use Case |
|----------|--------|----------|
| CAdES | CMS Advanced Electronic Signatures | Detached/attached signatures on binary documents |
| PAdES | PDF Advanced Electronic Signatures | Embedded signatures within PDF documents |
| XAdES | XML Advanced Electronic Signatures | XML-based documents and invoices |
| JAdES | JSON Advanced Electronic Signatures | JSON-based API responses and data |
| QES (eIDAS) | Qualified Electronic Signature | EU-recognized legally binding signatures |
| PGP/GPG | OpenPGP | Email-attached documents and open-source tooling |
| PKCS#7/CMS | Cryptographic Message Syntax | Legacy signed documents |

---

## 3. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Document classification accuracy | >= 98% correct classification | Precision/recall on labeled test set |
| Signature verification rate | 100% of signed documents verified | Coverage score |
| Tamper detection rate | 100% of modified documents detected | Hash mismatch detection rate |
| Fraud pattern detection precision | >= 95% true positive rate | Precision on labeled fraud set |
| False positive rate | <= 2% for fraud alerts | FP rate on clean document set |
| Certificate chain validation | 100% of chains verified to root CA | Chain coverage score |
| External registry verification | >= 90% of certificates cross-referenced | Registry query success rate |
| Processing throughput | >= 500 documents/minute (batch mode) | Batch processing benchmark |
| Single document verification | < 2 seconds p95 latency | API response time |
| Test coverage | >= 500 unit tests | Pytest count |

---

## 4. Scope

### 4.1 In Scope
- All 20 EUDR-relevant document types listed in Section 2.2
- Digital signature verification for 7 signature standards (Section 2.3)
- SHA-256/SHA-512 hash-based tamper detection
- X.509 certificate chain validation with OCSP/CRL revocation checking
- PDF/image metadata extraction and cross-validation
- 15+ deterministic fraud pattern rules
- Cross-document quantity and date reconciliation
- External certificate registry integration (FSC, RSPO, ISCC, Fairtrade, UTZ/RA)
- Document authentication reports for auditors
- 5-year immutable authentication record retention per Article 14
- Batch document processing for bulk verification

### 4.2 Out of Scope
- OCR text extraction from scanned documents (handled by AGENT-DATA-001)
- Physical document verification (watermark, paper, ink analysis)
- LLM-based document understanding or classification
- Real-time video verification of physical inspections
- Document storage and archival (handled by S3/Object Storage INFRA-004)
- Translation of non-English documents (handled by separate NLP service)
- Financial fraud investigation or forensic accounting

---

## 5. Zero-Hallucination Principles

1. Document classification uses deterministic rule-based matching against known templates, field patterns, and document structure -- no LLM inference.
2. Signature verification uses standard cryptographic libraries (pyOpenSSL, cryptography) for PKI validation -- no probabilistic assessment.
3. Hash comparison is exact byte-level matching -- SHA-256(document_bytes) == stored_hash.
4. Fraud pattern detection uses deterministic rules (quantity_claimed > quantity_certified = fraud_flag) -- no machine learning scoring.
5. Certificate chain validation follows RFC 5280 X.509 path validation algorithm -- deterministic chain traversal.
6. Cross-reference verification queries external registries and compares exact field values -- no fuzzy matching on critical fields (certificate numbers, quantities).
7. SHA-256 provenance hashing ensures tamper detection on all authentication results.

---

## 6. Feature Requirements

### 6.1 Feature 1: Document Classifier Engine (P0)

**Requirements**:
- F1.1: Classify incoming documents into 20+ EUDR-relevant types (COO, PC, BOL, CDE, CDI, RSPO, FSC, ISCC, FT, UTZ, LTR, LTD, FMP, FC, WQC, DDS, SSD, IC, TC, WR)
- F1.2: Classification based on deterministic rules: file format, embedded metadata, header patterns, field structure, issuing authority keywords
- F1.3: Template matching against known document templates per issuing authority per country
- F1.4: Multi-page document handling: classify primary document and identify attachments/annexes
- F1.5: Confidence scoring: high (>95% match to known template), medium (70-95%), low (<70%)
- F1.6: Unknown document type flagging: documents that don't match any known type flagged for manual review
- F1.7: Document language detection (EN, FR, DE, ES, PT, ID, NL) for routing to appropriate template set
- F1.8: Batch classification: process 500+ documents per minute
- F1.9: Classification history: maintain log of all classification decisions with rationale
- F1.10: Template registry: add/update/retire document templates without code changes

### 6.2 Feature 2: Digital Signature Verifier (P0)

**Requirements**:
- F2.1: Verify CAdES, PAdES, XAdES, JAdES digital signatures
- F2.2: Verify PGP/GPG signatures on document files
- F2.3: Verify PKCS#7/CMS legacy signatures
- F2.4: Validate eIDAS qualified electronic signatures (QES) against EU trusted lists
- F2.5: Extract signer identity from certificate (CN, O, OU, C, email)
- F2.6: Verify signature timestamp (TSA) for non-repudiation
- F2.7: Detect unsigned documents that should be signed (per document type rules)
- F2.8: Detect stripped signatures (document modified after signing to remove signature)
- F2.9: Multi-signature validation: verify all signatures on multi-signed documents
- F2.10: Signature verification result: valid, invalid, expired, revoked, unknown_signer, no_signature

### 6.3 Feature 3: Hash Integrity Validator (P0)

**Requirements**:
- F3.1: Compute SHA-256 and SHA-512 hashes for all incoming documents
- F3.2: Store document hashes in immutable hash registry with first-seen timestamp
- F3.3: Detect previously seen documents via hash lookup (deduplication)
- F3.4: Detect modified documents: same logical document (same filename, similar metadata) but different hash
- F3.5: Support incremental hashing for large documents (>100 MB)
- F3.6: Hash chain anchoring: link document hashes to parent record (batch, shipment, DDS) hash
- F3.7: Merkle tree construction for document sets: single root hash for entire DDS evidence package
- F3.8: Hash verification API: external parties can verify document authenticity by submitting hash
- F3.9: Hash registry statistics: total documents, unique hashes, duplicate rate, modification rate
- F3.10: Support for HMAC-based integrity where documents include pre-computed HMAC tags

### 6.4 Feature 4: Certificate Chain Validator (P0)

**Requirements**:
- F4.1: Validate X.509 certificate chains from leaf to trusted root CA per RFC 5280
- F4.2: Maintain trusted CA store with EU eIDAS trusted service providers
- F4.3: OCSP (Online Certificate Status Protocol) real-time revocation checking
- F4.4: CRL (Certificate Revocation List) offline revocation checking with automatic CRL refresh
- F4.5: Certificate validity period checking: not-before and not-after date validation
- F4.6: Key usage validation: verify certificate key usage matches signing purpose
- F4.7: Certificate pinning for known issuers: FSC-ASI, RSPO, ISCC certification bodies
- F4.8: Self-signed certificate detection and flagging
- F4.9: Weak key detection: flag certificates with RSA < 2048 bits or ECDSA < 256 bits
- F4.10: Certificate transparency log checking (where available)

### 6.5 Feature 5: Metadata Extractor and Validator (P0)

**Requirements**:
- F5.1: Extract PDF metadata: Title, Author, Creator, Producer, CreationDate, ModDate, Keywords
- F5.2: Extract EXIF metadata from embedded images: GPS coordinates, camera model, capture date
- F5.3: Extract XMP metadata from PDF and image documents
- F5.4: Cross-validate creation date vs claimed issuance date (flag if creation date > issuance date)
- F5.5: Cross-validate author/producer vs claimed issuing authority
- F5.6: Detect metadata stripping: documents with suspiciously empty metadata
- F5.7: Detect metadata inconsistency: creation tool mismatch (e.g., "Microsoft Word" for official government certificate)
- F5.8: Extract and validate document serial numbers, reference numbers, and control numbers
- F5.9: Extract and validate date fields across document body and metadata
- F5.10: GPS coordinate extraction from scanned documents with EXIF data for photo-based evidence

### 6.6 Feature 6: Fraud Pattern Detector (P0)

**Requirements**:
- F6.1: **Duplicate document reuse**: Same document hash submitted against multiple shipments/batches
- F6.2: **Quantity tampering**: Quantity in document differs from quantity in linked batch record by >5%
- F6.3: **Date manipulation**: Document dates inconsistent with supply chain event dates (±30 day tolerance)
- F6.4: **Expired certificate submission**: Certificate validity end date before submission date
- F6.5: **Serial number anomaly**: Certificate serial numbers that don't match issuer numbering patterns
- F6.6: **Issuer authority mismatch**: Document claims issuance by authority not authorized for that document type in that country
- F6.7: **Template forgery detection**: Document structure deviates from known legitimate templates
- F6.8: **Cross-document quantity inconsistency**: Quantities across related documents (BOL, CDE, COO, WQC) differ by >5%
- F6.9: **Geographic impossibility**: Document claims origin in country A but supply chain graph shows origin in country B
- F6.10: **Velocity anomaly**: Same supplier submitting unusually high volume of certificates in short timeframe
- F6.11: **Modification timeline anomaly**: Document modification date after claimed issuance date
- F6.12: **Round number bias**: Suspiciously round quantities (1,000.00, 5,000.00) across multiple documents suggesting fabrication
- F6.13: **Copy-paste detection**: Multiple documents with identical text blocks but different reference numbers
- F6.14: **Missing required documents**: DDS submission lacking mandatory document types for the commodity
- F6.15: **Certification scope mismatch**: Certificate scope (product type) doesn't match the commodity being declared

**Fraud Severity Classification:**

| Level | Threshold | Action | Auto-Action |
|-------|-----------|--------|-------------|
| Low | Single minor inconsistency | Informational alert | Log only |
| Medium | Multiple minor or single major inconsistency | Review required | Flag for manual review |
| High | Strong fraud indicators | Investigation required | Block DDS submission |
| Critical | Confirmed document manipulation | Enforcement action | Block + notify compliance officer + audit trail |

### 6.7 Feature 7: Cross-Reference Verifier (P0)

**Requirements**:
- F7.1: Query FSC certificate database (info.fsc.org) to verify FSC certificate validity and scope
- F7.2: Query RSPO PalmTrace system to verify RSPO certificate validity and volume claims
- F7.3: Query ISCC certificate search to verify ISCC certificate validity
- F7.4: Query Fairtrade/FLOCERT database to verify Fairtrade certificate status
- F7.5: Query UTZ/Rainforest Alliance portal to verify certification status
- F7.6: Cross-reference document parties (exporter, importer, processor) against supply chain graph (AGENT-EUDR-001)
- F7.7: Cross-reference document quantities against mass balance ledger entries (AGENT-EUDR-011)
- F7.8: Cross-reference shipping dates against bill of lading and customs declaration dates
- F7.9: Verify accreditation of laboratories issuing test reports
- F7.10: Cache external registry responses with configurable TTL (default 24 hours) to reduce API calls

### 6.8 Feature 8: Compliance Reporter (P0)

**Requirements**:
- F8.1: Generate document authentication report per DDS with all verification results
- F8.2: Generate document evidence package: all verified documents with authentication stamps
- F8.3: Authentication certificate: per-document verification certificate with provenance hash
- F8.4: Fraud risk summary: aggregate fraud risk score per DDS, per supplier, per commodity
- F8.5: Document completeness report: identify missing required documents per commodity type
- F8.6: Report formats: JSON, PDF, CSV, EUDR XML
- F8.7: Historical authentication dashboard: authentication trends over time per operator
- F8.8: Regulatory evidence package: compiled documentation for competent authority inspections
- F8.9: Batch authentication summary: aggregate results for bulk document verification
- F8.10: Report retention: 5-year immutable storage per EUDR Article 14

---

## 7. Technical Requirements

### 7.1 Architecture

```
greenlang/agents/eudr/document_authentication/
    __init__.py                          # Package exports (80+ symbols)
    config.py                            # DocumentAuthenticationConfig singleton
    models.py                            # Pydantic v2 models, enums
    provenance.py                        # SHA-256 chain hashing
    metrics.py                           # Prometheus metrics (gl_eudr_dav_ prefix)
    document_classifier.py               # Engine 1: Document type classification
    signature_verifier.py                # Engine 2: Digital signature verification
    hash_integrity_validator.py          # Engine 3: Hash-based tamper detection
    certificate_chain_validator.py       # Engine 4: X.509 certificate chain validation
    metadata_extractor.py                # Engine 5: Metadata extraction and validation
    fraud_pattern_detector.py            # Engine 6: Deterministic fraud pattern detection
    cross_reference_verifier.py          # Engine 7: External registry verification
    compliance_reporter.py               # Engine 8: Authentication reporting
    setup.py                             # DocumentAuthenticationService facade
    reference_data/
        __init__.py
        document_templates.py            # Known templates per document type per country
        trusted_cas.py                   # Trusted CA certificates and pinned issuers
        fraud_rules.py                   # 15+ deterministic fraud detection rules
    api/
        __init__.py
        router.py
        schemas.py
        dependencies.py
        classify_routes.py               # Document classification endpoints
        signature_routes.py              # Signature verification endpoints
        hash_routes.py                   # Hash integrity endpoints
        certificate_routes.py            # Certificate chain endpoints
        metadata_routes.py               # Metadata extraction endpoints
        fraud_routes.py                  # Fraud detection endpoints
        crossref_routes.py               # Cross-reference verification endpoints
        report_routes.py                 # Compliance reporting endpoints
```

### 7.2 Database Schema (V100)

| Table | Type | Description |
|-------|------|-------------|
| `gl_eudr_dav_documents` | regular | Document master records (id, type, filename, size, hash) |
| `gl_eudr_dav_classifications` | hypertable (monthly) | Document classification results with confidence |
| `gl_eudr_dav_signatures` | regular | Digital signature verification records |
| `gl_eudr_dav_hash_registry` | regular | Immutable document hash registry (SHA-256/512) |
| `gl_eudr_dav_certificate_chains` | regular | X.509 certificate chain validation results |
| `gl_eudr_dav_metadata` | regular | Extracted document metadata (creation, author, GPS) |
| `gl_eudr_dav_fraud_alerts` | hypertable (monthly) | Fraud pattern detection alerts |
| `gl_eudr_dav_crossref_results` | hypertable (monthly) | External registry verification results |
| `gl_eudr_dav_trusted_cas` | regular | Trusted CA certificate store |
| `gl_eudr_dav_document_templates` | regular | Known document templates per type per country |
| `gl_eudr_dav_authentication_reports` | regular | Generated authentication reports |
| `gl_eudr_dav_audit_log` | regular | Immutable audit trail for all authentication events |

### 7.3 Prometheus Metrics (18 metrics, `gl_eudr_dav_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_dav_documents_processed_total` | Counter | Total documents processed |
| `gl_eudr_dav_classifications_total` | Counter | Document classifications performed |
| `gl_eudr_dav_signatures_verified_total` | Counter | Digital signatures verified |
| `gl_eudr_dav_signatures_invalid_total` | Counter | Invalid signatures detected |
| `gl_eudr_dav_hashes_computed_total` | Counter | Document hashes computed |
| `gl_eudr_dav_duplicates_detected_total` | Counter | Duplicate documents detected via hash |
| `gl_eudr_dav_tampering_detected_total` | Counter | Document tampering detected |
| `gl_eudr_dav_cert_chains_validated_total` | Counter | Certificate chains validated |
| `gl_eudr_dav_cert_revocations_total` | Counter | Revoked certificates detected |
| `gl_eudr_dav_fraud_alerts_total` | Counter | Fraud alerts generated |
| `gl_eudr_dav_fraud_critical_total` | Counter | Critical fraud alerts |
| `gl_eudr_dav_crossref_queries_total` | Counter | External registry queries |
| `gl_eudr_dav_reports_generated_total` | Counter | Authentication reports generated |
| `gl_eudr_dav_api_errors_total` | Counter | API errors |
| `gl_eudr_dav_classification_duration_seconds` | Histogram | Document classification latency |
| `gl_eudr_dav_verification_duration_seconds` | Histogram | Full verification pipeline latency |
| `gl_eudr_dav_crossref_duration_seconds` | Histogram | External registry query latency |
| `gl_eudr_dav_active_verifications` | Gauge | Currently active verification jobs |

### 7.4 API Endpoints (~37 endpoints)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| Classify | POST | `/api/v1/eudr-dav/classify` | Classify document type |
| | POST | `/api/v1/eudr-dav/classify/batch` | Batch classify documents |
| | GET | `/api/v1/eudr-dav/classify/{document_id}` | Get classification result |
| | GET | `/api/v1/eudr-dav/classify/templates` | List available templates |
| | POST | `/api/v1/eudr-dav/classify/templates` | Register new template |
| Signature | POST | `/api/v1/eudr-dav/signatures/verify` | Verify document signature |
| | POST | `/api/v1/eudr-dav/signatures/verify/batch` | Batch verify signatures |
| | GET | `/api/v1/eudr-dav/signatures/{verification_id}` | Get verification result |
| | GET | `/api/v1/eudr-dav/signatures/history/{document_id}` | Get signature history |
| Hash | POST | `/api/v1/eudr-dav/hashes/compute` | Compute document hash |
| | POST | `/api/v1/eudr-dav/hashes/verify` | Verify document against stored hash |
| | GET | `/api/v1/eudr-dav/hashes/registry/{hash}` | Look up hash in registry |
| | GET | `/api/v1/eudr-dav/hashes/merkle/{dds_id}` | Get Merkle root for DDS package |
| Certificate | POST | `/api/v1/eudr-dav/certificates/validate` | Validate certificate chain |
| | GET | `/api/v1/eudr-dav/certificates/{validation_id}` | Get validation result |
| | POST | `/api/v1/eudr-dav/certificates/trusted-cas` | Add trusted CA |
| | GET | `/api/v1/eudr-dav/certificates/trusted-cas` | List trusted CAs |
| Metadata | POST | `/api/v1/eudr-dav/metadata/extract` | Extract document metadata |
| | GET | `/api/v1/eudr-dav/metadata/{document_id}` | Get extracted metadata |
| | POST | `/api/v1/eudr-dav/metadata/validate` | Validate metadata consistency |
| Fraud | POST | `/api/v1/eudr-dav/fraud/detect` | Run fraud detection |
| | POST | `/api/v1/eudr-dav/fraud/detect/batch` | Batch fraud detection |
| | GET | `/api/v1/eudr-dav/fraud/alerts/{document_id}` | Get fraud alerts |
| | GET | `/api/v1/eudr-dav/fraud/alerts/summary/{operator_id}` | Get fraud summary |
| | GET | `/api/v1/eudr-dav/fraud/rules` | List active fraud rules |
| CrossRef | POST | `/api/v1/eudr-dav/crossref/verify` | Verify against external registry |
| | POST | `/api/v1/eudr-dav/crossref/verify/batch` | Batch cross-reference |
| | GET | `/api/v1/eudr-dav/crossref/{verification_id}` | Get verification result |
| | GET | `/api/v1/eudr-dav/crossref/cache/stats` | Get registry cache stats |
| Report | POST | `/api/v1/eudr-dav/reports/authentication` | Generate authentication report |
| | POST | `/api/v1/eudr-dav/reports/evidence-package` | Generate evidence package |
| | GET | `/api/v1/eudr-dav/reports/{report_id}` | Get report |
| | GET | `/api/v1/eudr-dav/reports/{report_id}/download` | Download report |
| | GET | `/api/v1/eudr-dav/reports/dashboard/{operator_id}` | Get authentication dashboard |
| Batch | POST | `/api/v1/eudr-dav/batch` | Submit batch verification job |
| | DELETE | `/api/v1/eudr-dav/batch/{job_id}` | Cancel batch job |
| Health | GET | `/api/v1/eudr-dav/health` | Health check |

---

## 8. Test Strategy

### 8.1 Unit Tests (500+)

- Document classification for all 20 document types with known templates
- Digital signature verification for all 7 signature standards (valid, invalid, expired, revoked)
- Hash computation and verification (SHA-256, SHA-512) with tamper detection
- Certificate chain validation (complete chain, broken chain, expired cert, revoked cert, self-signed)
- Metadata extraction from PDF, JPEG, PNG, TIFF documents
- All 15 fraud pattern rules with positive and negative test cases
- Cross-reference verification with mocked external registry responses
- Compliance report generation in all 4 formats (JSON, PDF, CSV, XML)
- Edge cases: empty documents, corrupted files, zero-byte files, oversized documents
- Batch processing with mixed document types and verification results
- Merkle tree construction and root hash verification
- Provenance hash chain integrity across all authentication operations

### 8.2 Performance Tests

- Batch classification of 10,000 documents
- Concurrent signature verification for 1,000 documents
- Hash registry lookup for 1,000,000 entries
- Cross-reference batch query with 500 certificates

---

## Appendices

### A. Document Classification Rules (Reference Data)

| Document Type | Key Indicators | Confidence Boost |
|---------------|----------------|------------------|
| COO | "Certificate of Origin", chamber of commerce logo, HS codes | Serial number matches country pattern |
| PC | "Phytosanitary Certificate", IPPC logo, pest inspection fields | IPPC member country issuer |
| BOL | "Bill of Lading", vessel/voyage fields, port of loading | Carrier registered with IMO |
| RSPO | "RSPO", "Certified Sustainable Palm Oil", RSPO logo | Certificate # matches RSPO format |
| FSC | "FSC", "Forest Stewardship Council", FSC logo, chain of custody # | Certificate # verified in FSC DB |
| ISCC | "ISCC", sustainability certificate fields | ISCC certificate number format |

### B. Fraud Detection Rules (Reference Data)

| Rule ID | Pattern | Severity | Detection Method |
|---------|---------|----------|------------------|
| FRD-001 | Duplicate hash across shipments | High | Hash registry exact match |
| FRD-002 | Quantity > 105% of certified amount | High | Exact arithmetic comparison |
| FRD-003 | Document date > 30 days from supply chain event | Medium | Date arithmetic |
| FRD-004 | Expired certificate submitted | High | Date comparison against validity period |
| FRD-005 | Serial number format mismatch | Medium | Regex pattern matching |
| FRD-006 | Unauthorized issuer for document type | High | Authority lookup table |
| FRD-007 | Template structure deviation | Medium | Field position/count comparison |
| FRD-008 | Cross-document quantity variance >5% | High | Cross-document arithmetic |
| FRD-009 | Geographic origin mismatch | Critical | Supply chain graph cross-reference |
| FRD-010 | Velocity anomaly (>10 certs/day) | Medium | Time-windowed count |
| FRD-011 | Modification date after issuance date | Medium | Metadata date comparison |
| FRD-012 | Round number bias (>80% round quantities) | Low | Statistical pattern |
| FRD-013 | Identical text blocks across documents | Medium | Text similarity hash |
| FRD-014 | Missing required document for commodity | High | Completeness check |
| FRD-015 | Certification scope mismatch | High | Scope field matching |

### C. Trusted CA Store (Reference)

| CA Category | Examples | Purpose |
|-------------|----------|---------|
| EU eIDAS TSPs | D-Trust, Swisscom, DigiCert EU | QES verification |
| Document signing CAs | GlobalSign, Entrust, Comodo | PAdES/CAdES signing |
| Government CAs | National CA roots per EU member state | Official government documents |
| Certification body CAs | ASI (for FSC), RSPO, ISCC issuing CAs | Sustainability certificates |

### D. External Registry Integration

| Registry | API Type | Rate Limit | Cache TTL |
|----------|----------|------------|-----------|
| FSC Certificate DB | REST API | 100 req/min | 24 hours |
| RSPO PalmTrace | SOAP/REST | 60 req/min | 24 hours |
| ISCC Certificate Search | Web scraping / API | 30 req/min | 24 hours |
| Fairtrade/FLOCERT | REST API | 60 req/min | 24 hours |
| UTZ/RA Portal | REST API | 60 req/min | 24 hours |
| IPPC ePhyto | REST API | 120 req/min | 12 hours |
