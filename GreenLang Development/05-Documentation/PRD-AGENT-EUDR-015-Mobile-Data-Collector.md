# PRD: AGENT-EUDR-015 -- Mobile Data Collector Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-015 |
| **Agent ID** | GL-EUDR-MDC-015 |
| **Component** | Mobile Data Collector Agent |
| **Category** | EUDR Regulatory Agent -- Field Data Collection & Offline Operations |
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

The EU Deforestation Regulation requires operators and traders to collect granular field-level data at the point of production: GPS coordinates of plots of land (Article 9(1)(d)), producer identification, commodity harvest records, and custody transfer documentation. This data must be collected at the very beginning of the supply chain -- in rural farms, smallholder cooperatives, forest concessions, and collection points that are overwhelmingly located in regions with limited or no internet connectivity.

Current gaps in field-level data collection for EUDR compliance:

- **No connectivity at the point of collection**: The majority of EUDR-regulated commodity production occurs in rural areas of tropical countries (Indonesia, Brazil, Cote d'Ivoire, Ghana, Colombia, DRC) where mobile network coverage is intermittent at best and absent at worst. Field agents collecting producer data, plot GPS coordinates, and harvest records cannot rely on cloud-connected applications. Current web-based compliance platforms are unusable in the field, forcing data collectors to use paper forms that are later manually digitized -- a process that is slow (weeks of lag), error-prone (15-25% transcription error rates), and loses critical metadata (GPS accuracy, timestamps, photo evidence).

- **GPS/geolocation capture is inconsistent and unverified**: EUDR Article 9(1)(d) requires geolocation coordinates for plots of land exceeding 4 hectares (full polygon boundaries) and point coordinates for smaller plots. Field agents currently use a mix of consumer GPS devices, smartphone apps, and hand-written coordinate transcriptions. There is no standardized protocol for accuracy verification (HDOP thresholds, satellite count minimums), no polygon boundary collection workflow, and no validation that captured coordinates fall within plausible geographic bounds for the declared commodity and country of origin.

- **Photo evidence lacks integrity guarantees**: Field verification increasingly requires photographic evidence (plot photos, commodity photos, facility photos, document photos). Photos taken on personal devices lack verifiable metadata: timestamps can be altered, GPS coordinates can be spoofed, and there is no chain of custody from capture to upload. Without integrity hashing at capture time, photo evidence cannot meet the evidentiary standards required for competent authority inspections under Article 16.

- **Data synchronization is fragile and lossy**: When field agents return to connectivity, they must upload collected data to the central platform. Current ad-hoc synchronization methods (email attachments, WhatsApp file sharing, USB drives) result in data loss (5-10% of records lost in transit), duplication (same record uploaded multiple times), and conflict (two agents update the same producer record). There is no conflict resolution protocol, no retry logic, and no guarantee of eventual consistency.

- **Form management is rigid and outdated**: EUDR data collection requires different forms for different contexts (producer registration, plot survey, harvest log, custody transfer, quality inspection, smallholder declaration). Paper forms cannot enforce validation rules, conditional logic, or required field completion. When regulatory requirements change or new data fields are needed, reprinting and redistributing paper forms takes weeks. Multi-language support is impossible to manage across 24 EU languages and dozens of local languages.

- **No digital signature capability**: Custody transfer acknowledgments and smallholder declarations require signatures from producers, collectors, and inspectors. Paper signatures are easily forged, cannot be timestamped with certainty, and are lost when documents are damaged in field conditions. There is no mechanism for tamper-evident digital signature capture with cryptographic verification.

- **Data packages lack provenance and integrity**: When field data (forms, GPS, photos, signatures) is assembled for upload, there is no self-describing package format that bundles all related evidence together with integrity verification. Individual files uploaded separately lose their contextual relationships, making it impossible to verify that a photo was taken at the same location and time as the GPS capture it claims to document.

- **No visibility into field operations**: Operations managers have no real-time visibility into field data collection activities. They cannot track which devices are active, which areas have been surveyed, how much data is pending synchronization, or whether devices have sufficient storage and battery for planned field activities. This makes field operations planning reactive rather than proactive.

Without a purpose-built mobile data collection platform with offline-first architecture, EUDR-regulated operators face systemic data quality failures at the very foundation of their compliance chain -- the field-level data that all downstream traceability, risk assessment, and due diligence reporting depends upon.

### 1.2 Solution Overview

Agent-EUDR-015: Mobile Data Collector Agent provides a production-grade, offline-first mobile data collection platform that enables field-level EUDR compliance data capture in environments with limited or no internet connectivity. It provides structured form collection, high-accuracy GPS/polygon capture, geotagged photo evidence with integrity hashing, CRDT-based offline synchronization, dynamic form templates, digital signature capture, tamper-evident data packaging, and device fleet management.

Core capabilities:

1. **Offline-first form engine** -- Structured EUDR data collection forms (producer registration, plot geolocation capture, commodity harvest records, custody transfer documentation) that work completely offline with local SQLite storage, Pydantic schema validation, and queue-based synchronization when connectivity is restored. Forms enforce required fields, conditional logic, and range validation entirely on-device without server connectivity.

2. **GPS/geolocation capture engine** -- High-accuracy GPS coordinate capture using device sensors, with support for plot boundary polygon collection (walk-around polygon tracing), altitude recording, accuracy metadata (HDOP, satellite count, fix type), and WGS84 coordinate validation per EUDR Article 9(1)(d). Enforces configurable accuracy thresholds (default < 3m CEP with WAAS/EGNOS) and rejects captures that do not meet minimum quality standards.

3. **Photo evidence collector** -- Geotagged photo capture for field verification (plot photos, commodity photos, document photos, operator facility photos) with EXIF metadata extraction, timestamp verification against device clock, image compression (JPEG quality optimization for bandwidth), and SHA-256 integrity hashing at capture time. Photos are immutably linked to form submissions and GPS captures.

4. **Offline data synchronization engine** -- Conflict-resolution sync protocol for offline-to-server data reconciliation using CRDT-like (Conflict-free Replicated Data Type) merge strategies, upload queue management with priority ordering, bandwidth-optimized delta compression, and retry logic with exponential backoff. Guarantees eventual consistency with deterministic conflict resolution.

5. **Form template manager** -- Dynamic form template system supporting EUDR-specific form types (producer registration, plot survey, harvest log, custody transfer, quality inspection, smallholder declaration) with conditional logic (show/hide fields based on responses), validation rules (required, regex, range, cross-field), and multi-language support (24 EU languages + local languages). Templates are versioned and distributed to devices during sync.

6. **Digital signature capture** -- Capture and verify digital signatures from producers, collectors, and inspectors for custody transfer acknowledgments and smallholder declarations. Uses ECDSA P-256 for cryptographic signing with timestamp binding, device attestation, and tamper detection. Signatures are bound to specific form submissions and cannot be reused or transferred.

7. **Data package builder** -- Assembles collected field data (forms, GPS coordinates, photos, signatures) into self-contained, signed, EUDR-compliant data packages for upload to the central platform. Packages include a SHA-256 Merkle root for integrity verification, a manifest listing all included artifacts, and a provenance chain linking every element to its capture context (device, user, timestamp, location).

8. **Device fleet manager** -- Tracks and manages a fleet of mobile data collection devices, monitoring sync status, storage capacity, battery levels, last-seen timestamps, and assigned collection areas. Provides operations dashboards for field managers to plan collection campaigns, identify offline devices, and ensure complete geographic coverage.

### 1.3 Dependencies

| Dependency | Component | Integration |
|------------|-----------|-------------|
| AGENT-EUDR-001 | Supply Chain Mapping Master | Supply chain node data for producer/collector registration forms |
| AGENT-EUDR-002 | Geolocation Verification | GPS capture validation and coordinate cross-checking |
| AGENT-EUDR-009 | Chain of Custody Agent | Custody transfer records from field collection |
| AGENT-EUDR-014 | QR Code Generator | QR code scanning for product identification in field |
| AGENT-DATA-005 | EUDR Traceability Connector | DDS data synchronization with EU Information System |

---

## 2. Regulatory Context

### 2.1 EUDR Articles Addressed

| Article | Requirement | Agent Feature |
|---------|-------------|---------------|
| Art. 4(2) | Due diligence information collection at source | Offline form engine collects all DDS fields at point of production |
| Art. 9(1)(a) | Description of product, trade name, commodity type | Form templates capture commodity classification and product details |
| Art. 9(1)(b) | Quantity (net mass in kg, volume, or number of items) | Harvest log forms capture quantity with unit validation |
| Art. 9(1)(c) | Country of production, all plots of land | GPS capture engine records country-linked plot coordinates |
| Art. 9(1)(d) | Geolocation of plots (polygon for > 4 ha, point for <= 4 ha) | GPS engine captures point coordinates and polygon boundary traces |
| Art. 9(1)(e) | Date or time range of production/harvest | Form timestamps with NTP-verified device clock |
| Art. 9(1)(f) | Name, postal address, email of supplier | Producer registration forms with offline storage |
| Art. 9(1)(g) | Name, postal address, email of buyer | Custody transfer forms capture buyer information |
| Art. 10(1) | Risk assessment based on collected information | Field inspection forms feed risk assessment data |
| Art. 14 | 5-year retention of due diligence information | Data packages retained for 5 years with integrity verification |
| Art. 16(10) | Competent authority checks at 9% rate | Device fleet manager supports field inspection campaigns |
| Art. 22 | Customs authorities verification | Data packages provide verifiable field-collected evidence |

### 2.2 Supported Form Types

| Form Type | EUDR Purpose | Key Fields | Typical Use |
|-----------|-------------|------------|-------------|
| Producer Registration | Art. 9(1)(f) supplier identification | Name, address, coordinates, ID documents, farm size, commodities | Initial producer onboarding |
| Plot Survey | Art. 9(1)(c-d) geolocation capture | GPS point/polygon, area measurement, land use, canopy cover | Plot boundary mapping |
| Harvest Log | Art. 9(1)(a-b,e) production data | Commodity type, quantity, harvest date, plot reference | Harvest event recording |
| Custody Transfer | Art. 9(1)(f-g) chain of custody | Sender, receiver, quantity, product, signatures, transport | Product handover documentation |
| Quality Inspection | Art. 10(1) risk assessment data | Visual inspection, moisture, defects, photos, grading | Quality verification at collection |
| Smallholder Declaration | Art. 4(2) due diligence | Producer self-declaration, no-deforestation commitment, land rights | Smallholder compliance |

### 2.3 Geolocation Requirements per EUDR

| Plot Size | Coordinate Requirement | Capture Method | Accuracy Target |
|-----------|----------------------|----------------|-----------------|
| <= 4 hectares | Single point coordinate (latitude, longitude) | GPS point capture | < 3m CEP (WAAS/EGNOS) |
| > 4 hectares | Polygon boundary (sufficient points to describe perimeter) | Walk-around polygon tracing | < 5m CEP per vertex |
| Any size | WGS84 datum, 6 decimal places minimum | Device GPS sensor | HDOP < 2.0 |
| Any size | Altitude optional but recommended | Barometric + GPS fusion | +/- 5m |

---

## 3. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Offline form submission latency | < 200ms local storage write | Benchmark on reference device (Android 11, 4GB RAM) |
| GPS capture accuracy | < 3m CEP (with WAAS/EGNOS augmentation) | Field test with surveyed reference points |
| Polygon capture accuracy | < 5m CEP per vertex | Field test against DGPS reference boundary |
| Sync completion for 100 forms | < 30 seconds on 3G connection | End-to-end sync benchmark |
| Photo compression ratio | > 10:1 with SSIM > 0.95 | Image quality benchmark against originals |
| Sync conflict detection | 100% detection rate | Simulated concurrent edit scenarios |
| Data package integrity | 5-year verification with SHA-256 Merkle root | Long-term integrity test |
| Device fleet scale | >= 1,000 active devices concurrently | Load test with simulated device fleet |
| Language support | 24 EU languages + 20 local languages | Form rendering test in all languages |
| Test coverage | >= 500 unit tests | Pytest count |

---

## 4. Scope

### 4.1 In Scope
- Offline-first form engine with local SQLite storage and schema validation
- GPS point coordinate capture with accuracy metadata (HDOP, satellite count, fix type)
- Plot boundary polygon capture via walk-around tracing
- Geotagged photo capture with EXIF extraction and SHA-256 integrity hashing
- CRDT-based offline data synchronization with conflict resolution
- Dynamic form templates with conditional logic and multi-language support
- ECDSA P-256 digital signature capture with timestamp binding
- Self-contained data package assembly with Merkle root integrity
- Device fleet management with sync status and telemetry monitoring
- Server-side APIs for sync, template distribution, fleet management, and package retrieval
- 5-year data retention per EUDR Article 14
- Support for 24 EU languages and 20+ local languages in form templates

### 4.2 Out of Scope
- Native mobile application development (agent provides server-side APIs; mobile app is a separate deliverable)
- Mobile device procurement and hardware provisioning
- Cellular network connectivity provisioning (SIM cards, data plans)
- Device MDM (Mobile Device Management) software
- Real-time video capture and streaming
- Voice recording and transcription
- Biometric authentication on device (fingerprint, face recognition)
- Satellite communication (Iridium, Starlink) for remote connectivity
- Drone-based data collection
- Offline map tile hosting (integrates with existing map providers)
- Physical device repair and maintenance

---

## 5. Zero-Hallucination Principles

1. GPS coordinates are captured from hardware sensors only -- no interpolation, extrapolation, or estimation of coordinates. If the GPS fix does not meet accuracy thresholds, the capture is rejected, not approximated.
2. Timestamps are sourced from the device clock with NTP verification at last sync -- no back-dating or forward-dating of form submissions. Timestamp drift is recorded and reported, not silently corrected.
3. Photo hashes (SHA-256) are computed at capture time from raw image bytes -- no post-hoc modification, re-hashing, or hash recalculation is permitted after initial capture.
4. Form validation uses deterministic Pydantic schema rules (required fields, regex patterns, numeric ranges, cross-field constraints) -- no LLM inference, fuzzy matching, or probabilistic validation.
5. Sync conflict resolution uses deterministic CRDT merge strategies (last-writer-wins with vector clocks for scalar fields, union for set fields) -- no probabilistic resolution, no AI-based merge, no human-in-the-loop for automated merges.
6. Digital signatures use ECDSA P-256 with deterministic k-value (RFC 6979) -- cryptographic verification is binary (valid/invalid) with no probabilistic scoring.
7. Data packages are sealed with a SHA-256 Merkle root computed over all included artifacts -- any modification to any artifact invalidates the entire package. Tamper detection is deterministic.
8. Device telemetry (battery level, storage capacity, GPS fix quality) is reported from hardware APIs -- no simulated, estimated, or interpolated device state data.

---

## 6. Feature Requirements

### 6.1 Feature 1: Offline Form Engine (P0)

**Requirements**:
- F1.1: Provide an offline-first form rendering and submission engine that stores all form data locally on-device in SQLite before any network transmission, guaranteeing zero data loss from connectivity interruptions
- F1.2: Support 6 EUDR-specific form types: producer_registration, plot_survey, harvest_log, custody_transfer, quality_inspection, smallholder_declaration -- each with pre-defined field schemas
- F1.3: Enforce required field validation locally on-device: all EUDR-mandatory fields (commodity type, quantity, geolocation, producer ID, date) must be completed before form submission is accepted
- F1.4: Support field-level validation rules: required, regex pattern, numeric min/max range, date range, enumeration (dropdown), cross-field dependency (e.g., if commodity = "wood" then require timber_species)
- F1.5: Store form submissions in a local queue with status tracking: draft (incomplete), pending (validated, awaiting sync), syncing (upload in progress), synced (confirmed by server), failed (sync error, retry scheduled)
- F1.6: Support form editing for draft and pending submissions -- synced submissions are immutable and cannot be modified (only corrective amendments via new submissions referencing the original)
- F1.7: Attach GPS captures, photos, and digital signatures to form submissions with referential integrity -- all attachments are linked by form_submission_id and cannot be orphaned
- F1.8: Support form submission in fully offline mode for a minimum of 30 days -- device must function without any server contact for at least 30 days without data loss or form template degradation
- F1.9: Achieve local storage write latency of < 200ms per form submission on reference device (Android 11, 4GB RAM, SQLite WAL mode)
- F1.10: Generate a local submission receipt (UUID, timestamp, SHA-256 hash of form data) for every form submission that serves as proof-of-collection for field agents

### 6.2 Feature 2: GPS/Geolocation Capture (P0)

**Requirements**:
- F2.1: Capture GPS point coordinates (latitude, longitude) from device hardware sensors with a minimum precision of 6 decimal places in WGS84 datum (approximately 0.11m resolution)
- F2.2: Record accuracy metadata for every GPS capture: HDOP (Horizontal Dilution of Precision), satellite count, fix type (GPS, GLONASS, Galileo, BeiDou, or combined), and estimated horizontal accuracy in meters
- F2.3: Enforce configurable accuracy thresholds -- reject GPS captures where estimated horizontal accuracy exceeds the configured maximum (default: 3m for point captures, 5m for polygon vertices); provide clear feedback to field agent to wait for better fix
- F2.4: Support plot boundary polygon capture via walk-around tracing: field agent walks the perimeter of a plot while the engine records GPS vertices at configurable intervals (distance-based: every N meters; time-based: every N seconds; or manual: tap to record vertex)
- F2.5: Validate polygon captures: minimum 3 vertices, polygon must close (first vertex = last vertex within tolerance), no self-intersecting edges, area calculation in hectares, area must be > 0.01 ha (minimum plausible plot size)
- F2.6: Calculate polygon area in hectares using the Shoelace formula with geodesic correction (Vincenty or Karney) for accurate area on the WGS84 ellipsoid
- F2.7: Record altitude (elevation above sea level in meters) using device barometric sensor when available, falling back to GPS-derived altitude; record altitude accuracy metadata
- F2.8: Validate that captured coordinates fall within plausible geographic bounds for the declared country of origin (coordinate bounding box check) and commodity type (e.g., coffee between latitudes 25N-25S)
- F2.9: Support WAAS (North America), EGNOS (Europe), and MSAS/GAGAN (Asia) augmentation systems for improved accuracy when available; record augmentation source in metadata
- F2.10: Store all GPS captures locally with full metadata (timestamp, accuracy, satellite count, HDOP, fix type, augmentation, device ID, operator ID) and link to associated form submissions

### 6.3 Feature 3: Photo Evidence Collection (P0)

**Requirements**:
- F3.1: Capture geotagged photos using device camera with automatic GPS coordinate embedding at capture time -- coordinates sourced from the GPS capture engine (F2) for consistency
- F3.2: Extract and preserve EXIF metadata from captured photos: timestamp (device clock), GPS coordinates, camera make/model, focal length, image dimensions, orientation
- F3.3: Compute SHA-256 integrity hash of the raw image bytes at capture time and store the hash alongside the photo record -- this hash serves as the immutable integrity anchor for the photo evidence
- F3.4: Support 6 photo categories: plot_photo (landscape view of production plot), commodity_photo (close-up of harvested commodity), document_photo (physical document capture), facility_photo (processing/storage facility), transport_photo (vehicle/container), identity_photo (producer/operator identification)
- F3.5: Apply configurable image compression for bandwidth optimization: JPEG quality levels (high: 90%, medium: 75%, low: 50%) with target compression ratio > 10:1 while maintaining SSIM (Structural Similarity Index) > 0.95 against original
- F3.6: Validate photo timestamp against device clock: flag photos where EXIF timestamp deviates from device system time by more than 60 seconds (potential clock manipulation indicator)
- F3.7: Support photo annotation: field agents can add text annotations (crop condition notes, defect descriptions) to photos without modifying the original image (annotations stored as separate metadata)
- F3.8: Enforce minimum photo quality requirements: minimum resolution 1280x960 pixels (1.2 MP), minimum file size 100 KB (prevents blank/corrupted captures), maximum file size 20 MB (prevents accidental video recording)
- F3.9: Link photos to form submissions and GPS captures with referential integrity -- each photo must be associated with exactly one form submission and optionally one GPS capture
- F3.10: Support batch photo capture: capture multiple photos in sequence for a single form submission (e.g., 4 cardinal direction photos of a plot) with sequential numbering and group linking

### 6.4 Feature 4: Offline Data Synchronization (P0)

**Requirements**:
- F4.1: Implement a queue-based synchronization engine that manages all pending uploads (forms, GPS captures, photos, signatures, data packages) in a prioritized upload queue with persistent storage
- F4.2: Detect connectivity state changes (offline, 2G, 3G, 4G, WiFi) and automatically initiate synchronization when connectivity is restored -- no manual trigger required (manual trigger also supported)
- F4.3: Implement CRDT-like (Conflict-free Replicated Data Type) merge strategies for conflict resolution: last-writer-wins (LWW) with vector clocks for scalar fields (name, address, quantity), grow-only set union for collection fields (photos, GPS captures), and deterministic merge for nested objects
- F4.4: Detect and record sync conflicts when the same record has been modified both locally and on the server since last sync -- conflicts are logged in gl_eudr_mdc_sync_conflicts with both versions preserved for manual resolution if needed
- F4.5: Implement bandwidth-optimized delta synchronization: only transmit changed fields (not entire records), compress upload payloads using gzip (minimum 60% compression ratio for JSON payloads), and batch multiple small uploads into single HTTP requests
- F4.6: Implement retry logic with exponential backoff: initial retry after 1 second, doubling up to maximum 5-minute interval, with configurable maximum retry count (default: 20 retries before marking as permanently_failed)
- F4.7: Support upload prioritization: forms (highest), GPS captures (high), signatures (high), photo metadata (medium), photo binary data (low -- largest payloads sent last to avoid blocking small critical data)
- F4.8: Guarantee exactly-once delivery using idempotency keys (UUID per upload item) -- server-side deduplication ensures that retried uploads do not create duplicate records
- F4.9: Provide sync progress reporting: items pending, items in progress, items completed, items failed, estimated time remaining, bytes uploaded, bytes remaining, current upload speed
- F4.10: Support configurable sync policies: sync_on_wifi_only (default for photos), sync_on_any_connection (default for forms), sync_manual_only (for large data packages), max_upload_size_per_sync (default 50 MB)

### 6.5 Feature 5: Form Template Management (P0)

**Requirements**:
- F5.1: Support dynamic form templates defined in JSON schema format that specify form structure (sections, fields, types, validation rules, conditional logic) independently of the rendering layer
- F5.2: Pre-define 6 EUDR form templates: producer_registration (25+ fields), plot_survey (20+ fields), harvest_log (15+ fields), custody_transfer (20+ fields), quality_inspection (15+ fields), smallholder_declaration (15+ fields)
- F5.3: Support field types: text, number (integer/decimal), date, datetime, dropdown (single/multi-select), radio, checkbox, gps_point, gps_polygon, photo, signature, barcode_scan (for QR codes), calculated (formula-based), section_header, instruction_text
- F5.4: Support conditional logic: show/hide fields based on other field values (e.g., if commodity_type = "wood" then show timber_species field), skip sections based on conditions, and conditional validation rules
- F5.5: Support multi-language form rendering: all field labels, placeholders, help text, validation messages, and dropdown options available in 24 EU official languages plus configurable local languages (minimum 20 local languages including Bahasa, Swahili, Portuguese-BR, French-WA, Twi, Fon, Yoruba, Hausa)
- F5.6: Implement template versioning: each template has a version number (semantic versioning), devices receive template updates during sync, backward-compatible field additions do not invalidate existing drafts
- F5.7: Support template inheritance: base templates can be extended with operator-specific custom fields without modifying the base template (e.g., operator adds "plantation_code" field to the standard producer_registration template)
- F5.8: Validate template definitions: JSON schema meta-validation ensures templates are well-formed before distribution (no circular dependencies, no undefined field references, all conditional targets exist)
- F5.9: Support template preview and testing: server-side template preview endpoint that renders a template with sample data for verification before deployment
- F5.10: Track template deployment status per device: which devices have which template versions, identify devices running outdated templates, and force-update mechanisms for critical template changes

### 6.6 Feature 6: Digital Signature Capture (P0)

**Requirements**:
- F6.1: Capture digital signatures from signatories (producers, collectors, inspectors, transport operators) as cryptographic signatures using ECDSA P-256 with deterministic k-value (RFC 6979)
- F6.2: Bind each signature to a specific form submission (form_submission_id) -- a signature cannot be captured without an associated form and cannot be reused across forms
- F6.3: Include timestamp binding in the signature payload: the signed data includes the form submission hash, the signer identity, and the ISO 8601 timestamp -- changing any element invalidates the signature
- F6.4: Support signatory identification: signer name, signer role (producer, collector, inspector, transport_operator, buyer), signer device ID, and optional photo of signer
- F6.5: Capture handwritten signature image (SVG touch path) in addition to cryptographic signature -- the visual signature is stored alongside the cryptographic signature for human-readable verification
- F6.6: Verify signatures locally on-device using the signer's public key without requiring server connectivity -- signature verification is fully offline-capable
- F6.7: Support multi-signature workflows: custody transfer forms require signatures from both sender and receiver; inspector forms require inspector signature plus producer acknowledgment
- F6.8: Detect signature tampering: any modification to the signed form data after signature capture invalidates the signature -- verification returns explicit tamper_detected status
- F6.9: Store signature records with full metadata: signer identity, signer role, form reference, timestamp, device ID, public key fingerprint, signature bytes (DER-encoded), and verification status
- F6.10: Support signature revocation: a signatory can revoke their signature within a configurable window (default 24 hours) with an explanatory reason -- revoked signatures are marked but preserved for audit trail

### 6.7 Feature 7: Data Package Builder (P0)

**Requirements**:
- F7.1: Assemble all collected field data for a given collection session (forms, GPS captures, photos, signatures) into a self-contained data package with a unique package_id (UUID v4)
- F7.2: Generate a package manifest (JSON) listing all included artifacts: artifact type, artifact ID, file name, file size, SHA-256 hash, capture timestamp, and capture GPS coordinates
- F7.3: Compute a SHA-256 Merkle root over all artifact hashes in the package -- the Merkle root serves as the single integrity anchor for the entire package (modifying any artifact changes the root)
- F7.4: Include a provenance chain in the package: device ID, operator ID, collection area, collection date range, GPS bounding box of all captures, template versions used, and agent version
- F7.5: Sign the package manifest with the device's ECDSA P-256 key -- the package signature proves the package was assembled on a specific registered device
- F7.6: Support package compression: gzip the entire package for bandwidth optimization with configurable compression level (default: 6, range: 1-9)
- F7.7: Support incremental package building: add artifacts to a package over multiple collection sessions before sealing -- once sealed (signed), no further artifacts can be added
- F7.8: Validate package completeness before sealing: verify all referenced form submissions have required GPS captures, all custody transfers have required signatures, all photos have integrity hashes
- F7.9: Generate package metadata for the server-side catalog: package size, artifact count (by type), collection date range, geographic extent, template versions, seal timestamp
- F7.10: Support package export in standard formats: ZIP (compressed archive), tar.gz (Linux-compatible archive), and JSON-LD (linked data format for machine-readable metadata)

### 6.8 Feature 8: Device Fleet Management (P0)

**Requirements**:
- F8.1: Register mobile data collection devices with the platform: device_id (UUID), device model, OS version, agent version, assigned operator, assigned collection area (GeoJSON polygon), and registration timestamp
- F8.2: Track device telemetry: battery level (%), storage capacity (total/used/free in bytes), GPS fix quality (last HDOP, satellite count), last sync timestamp, and last known GPS coordinates
- F8.3: Monitor device sync status: pending_forms (count), pending_photos (count), pending_gps (count), pending_signatures (count), last_sync_success (timestamp), last_sync_failure (timestamp and error), sync_queue_size (bytes)
- F8.4: Detect offline devices: flag devices that have not synced within a configurable threshold (default: 48 hours) as offline; generate alerts for devices offline > 7 days
- F8.5: Track device assignment: each device is assigned to an operator and a collection area -- assignment history is maintained for audit trail
- F8.6: Monitor agent version compliance: track which agent version each device is running, flag devices running outdated versions, and report version distribution across the fleet
- F8.7: Support device decommissioning: mark devices as decommissioned with reason (lost, damaged, retired, reassigned) -- decommissioned devices cannot sync new data but historical data is preserved
- F8.8: Provide fleet dashboard data: total devices, active (synced in last 48h), offline (not synced in 48h+), low battery (< 20%), low storage (< 500 MB free), outdated agent version, pending sync volume (total bytes)
- F8.9: Support collection campaign management: define collection campaigns (target area, target form count, target date range, assigned devices) and track progress against targets
- F8.10: Generate fleet utilization reports: forms collected per device per day, GPS captures per device, photos per device, sync frequency, data volume per device, geographic coverage map

---

## 7. Technical Requirements

### 7.1 Architecture

```
greenlang/agents/eudr/mobile_data_collector/
    __init__.py                          # Package exports (90+ symbols)
    config.py                            # MobileDataCollectorConfig singleton
    models.py                            # Pydantic v2 models, enums
    provenance.py                        # SHA-256 chain hashing, Merkle tree
    metrics.py                           # Prometheus metrics (gl_eudr_mdc_ prefix)
    offline_form_engine.py               # Engine 1: Offline form submission and local storage
    gps_capture_engine.py                # Engine 2: GPS point/polygon capture with accuracy validation
    photo_evidence_collector.py          # Engine 3: Geotagged photo capture with integrity hashing
    sync_engine.py                       # Engine 4: CRDT-based offline data synchronization
    form_template_manager.py             # Engine 5: Dynamic form template management
    digital_signature_engine.py          # Engine 6: ECDSA P-256 digital signature capture/verify
    data_package_builder.py              # Engine 7: Data package assembly with Merkle root
    device_fleet_manager.py              # Engine 8: Device fleet tracking and management
    setup.py                             # MobileDataCollectorService facade
    reference_data/
        __init__.py
        form_schemas.py                  # Pre-defined EUDR form template schemas
        coordinate_bounds.py             # Country/commodity geographic bounding boxes
        language_packs.py                # Multi-language label and message translations
    api/
        __init__.py
        router.py
        schemas.py
        dependencies.py
        form_routes.py                   # Form submission and retrieval endpoints
        gps_routes.py                    # GPS capture and polygon endpoints
        photo_routes.py                  # Photo upload and retrieval endpoints
        sync_routes.py                   # Sync trigger, status, and conflict endpoints
        template_routes.py               # Form template CRUD endpoints
        signature_routes.py              # Signature capture and verification endpoints
        package_routes.py                # Data package build, status, and download endpoints
        device_routes.py                 # Device registration, telemetry, and fleet endpoints
```

### 7.2 Database Schema (V103)

| Table | Type | Description |
|-------|------|-------------|
| `gl_eudr_mdc_forms` | hypertable (monthly) | Collected form submissions with status tracking (draft/pending/synced) |
| `gl_eudr_mdc_form_templates` | regular | Form template definitions with versioning and language packs |
| `gl_eudr_mdc_gps_captures` | hypertable (monthly) | GPS coordinate captures with accuracy metadata (HDOP, satellites, fix type) |
| `gl_eudr_mdc_polygon_traces` | regular | Plot boundary polygon traces with vertex arrays and area calculations |
| `gl_eudr_mdc_photos` | regular | Geotagged photo evidence records with SHA-256 integrity hashes |
| `gl_eudr_mdc_sync_queue` | regular | Offline sync queue with priority, retry count, and status |
| `gl_eudr_mdc_sync_conflicts` | regular | Sync conflict records with both local and server versions preserved |
| `gl_eudr_mdc_signatures` | regular | Digital signature records (ECDSA P-256) with form binding and verification status |
| `gl_eudr_mdc_data_packages` | regular | Assembled data packages with Merkle root and seal status |
| `gl_eudr_mdc_devices` | regular | Registered mobile devices with assignment and status |
| `gl_eudr_mdc_device_events` | hypertable (monthly) | Device telemetry events (battery, storage, GPS, sync) |
| `gl_eudr_mdc_audit_log` | regular | Immutable audit trail for all agent operations |

### 7.3 Prometheus Metrics (18 metrics, `gl_eudr_mdc_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_mdc_forms_submitted_total` | Counter | Total form submissions received from devices |
| `gl_eudr_mdc_gps_captures_total` | Counter | Total GPS captures received (point + polygon vertices) |
| `gl_eudr_mdc_photos_captured_total` | Counter | Total photos received from devices |
| `gl_eudr_mdc_syncs_completed_total` | Counter | Total successful sync sessions completed |
| `gl_eudr_mdc_sync_conflicts_total` | Counter | Total sync conflicts detected |
| `gl_eudr_mdc_signatures_captured_total` | Counter | Total digital signatures captured |
| `gl_eudr_mdc_packages_built_total` | Counter | Total data packages assembled and sealed |
| `gl_eudr_mdc_api_errors_total` | Counter | Total API errors across all endpoints |
| `gl_eudr_mdc_form_submission_duration_seconds` | Histogram | Form submission processing latency (server-side) |
| `gl_eudr_mdc_gps_capture_duration_seconds` | Histogram | GPS capture processing latency (server-side) |
| `gl_eudr_mdc_sync_duration_seconds` | Histogram | Full sync session duration |
| `gl_eudr_mdc_photo_upload_duration_seconds` | Histogram | Photo upload processing latency |
| `gl_eudr_mdc_package_build_duration_seconds` | Histogram | Data package build duration |
| `gl_eudr_mdc_pending_sync_items` | Gauge | Current count of items pending synchronization across all devices |
| `gl_eudr_mdc_active_devices` | Gauge | Devices that have synced within the last 48 hours |
| `gl_eudr_mdc_offline_devices` | Gauge | Devices that have not synced within 48 hours |
| `gl_eudr_mdc_storage_used_bytes` | Gauge | Total storage consumed by photos and data packages |
| `gl_eudr_mdc_pending_uploads` | Gauge | Total pending upload items across all device sync queues |

### 7.4 API Endpoints (~37 endpoints)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| Forms | POST | `/api/v1/eudr-mdc/forms/submit` | Submit a completed form from device |
| | POST | `/api/v1/eudr-mdc/forms/submit/batch` | Batch submit multiple forms |
| | GET | `/api/v1/eudr-mdc/forms/{form_id}` | Get form submission details |
| | GET | `/api/v1/eudr-mdc/forms` | List form submissions with filters |
| | POST | `/api/v1/eudr-mdc/forms/{form_id}/validate` | Validate form against template schema |
| GPS | POST | `/api/v1/eudr-mdc/gps/capture` | Submit GPS point capture |
| | POST | `/api/v1/eudr-mdc/gps/polygon` | Submit polygon boundary trace |
| | GET | `/api/v1/eudr-mdc/gps/{capture_id}` | Get GPS capture details |
| | GET | `/api/v1/eudr-mdc/gps/plot/{plot_id}/coordinates` | Get all coordinates for a plot |
| | POST | `/api/v1/eudr-mdc/gps/validate` | Validate coordinates against bounds |
| Photos | POST | `/api/v1/eudr-mdc/photos/upload` | Upload photo with metadata |
| | POST | `/api/v1/eudr-mdc/photos/upload/batch` | Batch upload multiple photos |
| | GET | `/api/v1/eudr-mdc/photos/{photo_id}` | Get photo metadata |
| | GET | `/api/v1/eudr-mdc/photos/{photo_id}/download` | Download photo binary |
| | POST | `/api/v1/eudr-mdc/photos/{photo_id}/verify` | Verify photo integrity (SHA-256) |
| Sync | POST | `/api/v1/eudr-mdc/sync/trigger` | Trigger sync for a device |
| | POST | `/api/v1/eudr-mdc/sync/upload` | Upload sync payload from device |
| | GET | `/api/v1/eudr-mdc/sync/{device_id}/status` | Get sync status for device |
| | GET | `/api/v1/eudr-mdc/sync/conflicts` | List unresolved sync conflicts |
| | POST | `/api/v1/eudr-mdc/sync/conflicts/{conflict_id}/resolve` | Resolve a sync conflict |
| Templates | POST | `/api/v1/eudr-mdc/templates` | Create form template |
| | GET | `/api/v1/eudr-mdc/templates` | List form templates |
| | GET | `/api/v1/eudr-mdc/templates/{template_id}` | Get template definition |
| | PUT | `/api/v1/eudr-mdc/templates/{template_id}` | Update form template |
| | GET | `/api/v1/eudr-mdc/templates/{template_id}/preview` | Preview template with sample data |
| | GET | `/api/v1/eudr-mdc/templates/device/{device_id}` | Get templates for device sync |
| Signatures | POST | `/api/v1/eudr-mdc/signatures/capture` | Submit captured signature |
| | POST | `/api/v1/eudr-mdc/signatures/{signature_id}/verify` | Verify signature authenticity |
| | POST | `/api/v1/eudr-mdc/signatures/{signature_id}/revoke` | Revoke a signature |
| | GET | `/api/v1/eudr-mdc/signatures/{signature_id}` | Get signature details |
| Packages | POST | `/api/v1/eudr-mdc/packages/build` | Build data package from collected data |
| | GET | `/api/v1/eudr-mdc/packages/{package_id}` | Get package metadata |
| | GET | `/api/v1/eudr-mdc/packages/{package_id}/download` | Download sealed data package |
| | POST | `/api/v1/eudr-mdc/packages/{package_id}/verify` | Verify package Merkle root integrity |
| Devices | POST | `/api/v1/eudr-mdc/devices/register` | Register a new device |
| | GET | `/api/v1/eudr-mdc/devices` | List all devices with fleet status |
| | GET | `/api/v1/eudr-mdc/devices/{device_id}` | Get device details and telemetry |
| | POST | `/api/v1/eudr-mdc/devices/{device_id}/telemetry` | Submit device telemetry event |
| | PUT | `/api/v1/eudr-mdc/devices/{device_id}/assign` | Assign device to operator/area |
| | POST | `/api/v1/eudr-mdc/devices/{device_id}/decommission` | Decommission a device |
| | GET | `/api/v1/eudr-mdc/devices/fleet/dashboard` | Fleet dashboard summary |
| Health | GET | `/api/v1/eudr-mdc/health` | Health check |

---

## 8. Test Strategy

### 8.1 Unit Tests (500+)

- Offline form engine: form creation, validation (all field types), status transitions (draft, pending, syncing, synced, failed), local storage write/read, attachment linking, 30-day offline operation
- GPS capture: point capture with accuracy validation, HDOP/satellite metadata, polygon vertex recording, polygon closure validation, self-intersection detection, area calculation (Shoelace with geodesic correction), coordinate bounds checking, WGS84 validation
- Photo evidence: EXIF extraction, SHA-256 hash computation, geotagging, timestamp validation, compression at 3 quality levels (SSIM verification), photo category assignment, minimum resolution enforcement, batch capture sequencing
- Sync engine: queue management (add, prioritize, retry, remove), CRDT merge (LWW scalar, set union, nested object), conflict detection and recording, delta compression (gzip ratio), exponential backoff timing, idempotency key deduplication, connectivity state detection, sync policy enforcement
- Form templates: JSON schema parsing, field type rendering (all 14 types), conditional logic (show/hide/skip), validation rule enforcement, multi-language rendering (sample of 5 languages), template versioning, template inheritance, meta-validation
- Digital signatures: ECDSA P-256 key generation, signature creation (RFC 6979), signature verification, timestamp binding, form binding, tamper detection, multi-signature workflows, signature revocation, offline verification
- Data packages: manifest generation, Merkle root computation (2, 10, 100, 1000 artifacts), package signing, completeness validation, compression, incremental building, package format export (ZIP, tar.gz, JSON-LD)
- Device fleet: device registration, telemetry recording, sync status tracking, offline detection, version compliance, device assignment, decommissioning, fleet dashboard aggregation, campaign tracking
- Edge cases: maximum form size (1000 fields), maximum polygon vertices (10,000), concurrent GPS captures, sync with 0 items, sync with 10,000 items, empty photo metadata, signature on empty form, package with single artifact, fleet with 1000 devices

### 8.2 Performance Tests

- Form submission: 10,000 forms processed in under 60 seconds (server-side)
- GPS capture ingestion: 50,000 point captures processed in under 120 seconds
- Photo upload: 1,000 photos (5 MB average) uploaded and hashed in under 300 seconds on 4G connection
- Sync pipeline: 100 forms + 100 GPS + 50 photos synced in under 30 seconds on 3G
- Data package: package with 500 artifacts built and sealed in under 10 seconds
- Fleet monitoring: dashboard query for 1,000 devices returns in under 2 seconds

---

## Appendices

### A. EUDR Article 9 Field-to-Form Mapping

| EUDR Art. 9 Requirement | Form Field | Form Type | Validation |
|------------------------|------------|-----------|------------|
| 9(1)(a) Product description | commodity_type, product_name, trade_name | All forms | Required, EUDR commodity enum |
| 9(1)(b) Quantity | quantity_kg, quantity_unit | Harvest log | Required, numeric > 0 |
| 9(1)(c) Country of production | country_code (ISO 3166-1 alpha-2) | Plot survey | Required, valid ISO code |
| 9(1)(d) Geolocation (point) | gps_latitude, gps_longitude | Plot survey | Required, WGS84, 6 decimals |
| 9(1)(d) Geolocation (polygon) | polygon_vertices[] | Plot survey (> 4 ha) | Min 3 vertices, closed, no self-intersect |
| 9(1)(e) Date of production | harvest_date, harvest_date_range | Harvest log | Required, ISO 8601, not future |
| 9(1)(f) Supplier info | producer_name, producer_address, producer_email | Producer registration | Required |
| 9(1)(g) Buyer info | buyer_name, buyer_address, buyer_email | Custody transfer | Required |

### B. GPS Accuracy Classification

| Accuracy Class | Horizontal Accuracy | HDOP Range | Satellite Count | Use Case |
|---------------|--------------------|-----------|--------------------|----------|
| Excellent | < 1m | < 1.0 | >= 12 | Reference surveys, boundary disputes |
| Good | 1-3m | 1.0-2.0 | 8-12 | Standard plot mapping, point captures |
| Acceptable | 3-5m | 2.0-3.0 | 6-8 | Polygon vertices in dense canopy |
| Poor | 5-10m | 3.0-5.0 | 4-6 | Emergency capture only (flagged for review) |
| Rejected | > 10m | > 5.0 | < 4 | Not accepted -- agent prompted to wait |

### C. Sync Conflict Resolution Rules

| Field Type | Conflict Strategy | Resolution Rule |
|-----------|------------------|-----------------|
| Scalar (name, address, quantity) | Last-Writer-Wins (LWW) | Compare vector clock timestamps; latest write prevails |
| Set (photos, GPS captures) | Grow-Only Set Union | Both versions merged; no deletions |
| Status (form status) | State Machine | Higher-precedence status wins (synced > pending > draft) |
| Signature | No Merge | Conflict flagged for manual resolution (signatures are immutable) |
| GPS coordinate | No Overwrite | Server version preserved; local version stored as "alternative_capture" |
| Nested object (template data) | Field-Level LWW | Each nested field resolved independently using LWW |

### D. Supported Languages

| Category | Languages | Count |
|----------|-----------|-------|
| EU Official | Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Irish, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish | 24 |
| Local (Africa) | Swahili, Twi, Fon, Yoruba, Hausa, Amharic, Lingala, Kinyarwanda, Malagasy, Wolof | 10 |
| Local (Americas) | Portuguese-BR, Spanish-LATAM, Quechua, Guarani, French-Guyane | 5 |
| Local (Asia) | Bahasa Indonesia, Bahasa Malay, Thai, Vietnamese, Khmer | 5 |
| **Total** | | **44** |

### E. Data Package Structure

```
package-{uuid}.zip
    manifest.json                    # Package manifest with artifact list
    manifest.sig                     # ECDSA P-256 signature of manifest
    provenance.json                  # Device, operator, area, date range metadata
    merkle_root.json                 # SHA-256 Merkle root and tree structure
    forms/
        form-{uuid-1}.json          # Form submission data
        form-{uuid-2}.json
    gps/
        capture-{uuid-1}.json       # GPS point capture with metadata
        capture-{uuid-2}.json
        polygon-{uuid-1}.json       # Polygon boundary trace
    photos/
        photo-{uuid-1}.jpg          # Compressed photo
        photo-{uuid-1}.meta.json    # Photo metadata + SHA-256 hash
        photo-{uuid-2}.jpg
        photo-{uuid-2}.meta.json
    signatures/
        sig-{uuid-1}.json           # Signature record with DER-encoded bytes
        sig-{uuid-2}.json
```

### F. Device Telemetry Event Schema

| Field | Type | Description |
|-------|------|-------------|
| device_id | UUID | Registered device identifier |
| event_type | enum | heartbeat, sync_start, sync_complete, sync_error, low_battery, low_storage, gps_fix_lost, gps_fix_acquired |
| battery_level | integer (0-100) | Current battery percentage |
| storage_total_bytes | bigint | Total device storage |
| storage_used_bytes | bigint | Used device storage |
| storage_free_bytes | bigint | Free device storage |
| gps_hdop | float | Current GPS HDOP |
| gps_satellites | integer | Current visible satellite count |
| gps_latitude | float | Last known latitude |
| gps_longitude | float | Last known longitude |
| pending_forms | integer | Forms awaiting sync |
| pending_photos | integer | Photos awaiting sync |
| pending_gps | integer | GPS captures awaiting sync |
| agent_version | string | Running agent version |
| os_version | string | Device OS version |
| connectivity_type | enum | none, 2g, 3g, 4g, 5g, wifi |
| timestamp | timestamptz | Event timestamp (device clock) |

### G. EUDR Commodity Geographic Bounds

| Commodity | Primary Production Latitudes | Primary Countries | Bounding Box Validation |
|-----------|-----------------------------|--------------------|------------------------|
| Cattle (beef/leather) | 35S - 25N | Brazil, Argentina, Paraguay, Uruguay | South America + Sub-Saharan Africa |
| Cocoa | 20N - 20S | Cote d'Ivoire, Ghana, Indonesia, Ecuador, Cameroon | Tropical belt only |
| Coffee | 25N - 25S | Brazil, Vietnam, Colombia, Indonesia, Ethiopia | Tropical/subtropical belt |
| Oil palm | 15N - 15S | Indonesia, Malaysia, Thailand, Colombia, Nigeria | Tropical belt only |
| Rubber | 25N - 15S | Thailand, Indonesia, Vietnam, Cote d'Ivoire, India | Tropical/subtropical Asia + W. Africa |
| Soya | 55N - 40S | Brazil, USA, Argentina, Paraguay, China | Americas + limited Asia |
| Wood | 70N - 50S | Brazil, Indonesia, Russia, Canada, DRC | Global (broadest bounds) |
