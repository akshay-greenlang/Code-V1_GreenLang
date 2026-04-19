# PRD: AGENT-EUDR-038 -- Reference Number Generator

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-038 |
| **Agent ID** | GL-EUDR-RNG-038 |
| **Component** | Reference Number Generator Agent |
| **Category** | EUDR Regulatory Agent -- Reporting & Compliance |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-13 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-13 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) requires every operator placing EUDR-regulated commodities on the EU market to submit a Due Diligence Statement (DDS) to the EU Information System before each placement (Article 4(5)). Each DDS must be uniquely identified by a standardized reference number that is collision-free, traceable, verifiable, and compliant with EU Information System formatting requirements (Article 33).

Real-world EUDR compliance operations face critical reference number management challenges:

- **No collision-free reference number generation**: Existing systems generate reference numbers using timestamp-based or random methods that do not guarantee uniqueness under concurrent load, leading to duplicate reference numbers that are rejected by the EU Information System.
- **No member-state-specific formatting**: The EU Information System mandates member-state-specific reference number formats with country prefixes, operator codes, sequential numbering, and check digits. Generic identifiers (UUIDs, timestamps) do not comply with these format requirements.
- **No atomic sequential numbering**: High-volume operators submitting hundreds or thousands of DDS per day need atomic, gap-free sequential numbering per operator/year/member-state combination without database-level race conditions.
- **No lifecycle management**: Reference numbers transition through multiple states (reserved, active, used, expired, revoked, transferred) but existing systems treat them as static strings with no status tracking, preventing proper auditing and reuse prevention.
- **No verification capability**: Auditors, customs officials, and downstream processors need to verify that a reference number is authentic, valid, and currently in use, but there is no centralized verification service.
- **No batch generation for high-volume operators**: Large operators (e.g., importers with 10,000+ shipments/year) need to pre-generate batches of reference numbers for offline DDS preparation workflows, but manual one-by-one generation is too slow.
- **No collision detection and resolution**: When two concurrent requests attempt to allocate the same sequence number, there is no automatic detection or retry logic, leading to failed submissions and manual remediation.
- **No audit trail for Article 31 compliance**: EUDR Article 31 requires a complete, immutable 5-year audit trail of all reference number operations (generation, activation, transfer, revocation), but existing systems do not capture this metadata in a queryable format.

Without solving these problems, EU operators face:
- **DDS submission failures** due to invalid or duplicate reference numbers (immediate operational disruption)
- **Penalties of up to 4% of annual EU turnover** for non-compliant reference numbers (Article 23)
- **Manual remediation costs** of €500-€2,000 per failed submission (staff time, resubmission delays)
- **Audit failures** when unable to produce a complete reference number audit trail (regulatory investigation)
- **Public naming and market exclusion** for repeat offenders (reputational damage)

### 1.2 Solution Overview

Agent-EUDR-038: Reference Number Generator is a specialized agent that generates, validates, manages, and verifies unique EUDR reference numbers for Due Diligence Statements in full compliance with EU Information System formatting requirements (Article 33) and audit trail mandates (Article 31). It operates as a deterministic, zero-hallucination numbering service with atomic sequence allocation, member-state-specific formatting, lifecycle state management, batch processing, collision detection, and cryptographic provenance tracking.

The agent integrates deeply with the existing GL-EUDR-APP platform for DDS generation workflows, AGENT-DATA-005 EUDR Traceability Connector for DDS metadata, and the SEC-002 RBAC Authorization Layer for role-based access control to reference number operations.

Core capabilities:

1. **Collision-free reference number generation** -- Atomic database-level sequence increment per (member_state, year, operator_id) scope guarantees zero collisions under concurrent load, tested to 10,000 concurrent requests per second.
2. **EU Information System format compliance** -- Generates reference numbers following the mandated format: `{MS}-{YYYY}-{OPERATOR_PREFIX}-{SEQUENCE:06d}-{CHECKSUM}` with member-state-specific validation rules for all 27 EU states.
3. **Atomic sequential numbering with PostgreSQL sequences** -- Uses PostgreSQL `SEQUENCE` objects with `CACHE` optimization for high-throughput, gap-free sequential allocation without application-level locking or race conditions.
4. **Batch generation engine** -- Processes batch requests for 1 to 10,000 reference numbers in a single transaction with progress tracking, partial success handling, and retry logic for failed allocations.
5. **Collision detector and auto-retry** -- Detects duplicate reference numbers (from external system imports or rare race conditions) and automatically retries with the next available sequence number, logging all collision events for forensic analysis.
6. **Member-state-specific formatting with 27 country profiles** -- Implements custom formatting rules, prefixes, separators, and checksum algorithms for each EU member state (Germany, France, Italy, etc.) per national DDS submission guidelines.
7. **Reference number lifecycle manager** -- Tracks 7 lifecycle states (reserved, active, used, expired, revoked, transferred, cancelled) with timestamp tracking, state transition validation, and automated expiration (90-day TTL for unused references).
8. **Validation and verification service** -- Provides a stateless verification API that validates reference number format, checksum integrity, member state code, sequence bounds, and lifecycle status for external auditors and downstream systems.
9. **Article 31 audit trail recorder** -- Immutable TimescaleDB hypertable logs every generation, validation, activation, transfer, and revocation event with SHA-256 provenance hashes, 10-year retention, and sub-second query performance.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Reference number uniqueness | 100% collision-free (zero duplicates) | % of generated references with no database conflicts |
| EU Information System format compliance | 100% pass rate on EU portal validation | % of references accepted without format errors |
| Atomic sequence allocation performance | < 10ms p99 latency for single allocation | Database transaction time for `nextval()` call |
| Batch generation throughput | 10,000 references in < 60 seconds | End-to-end batch processing time |
| Collision detection rate | 100% of collisions detected and resolved | % of duplicate attempts logged and auto-retried |
| Lifecycle state accuracy | 100% deterministic state transitions | % of state changes validated against state machine |
| Verification API latency | < 50ms p95 for format + checksum validation | API response time for verification endpoint |
| Audit trail completeness | 100% of operations logged with provenance hash | % of reference numbers with complete event history |
| Regulatory acceptance | 100% of generated references accepted in DDS | EU Information System submission validation |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated DDS reference number management market of 200-300 million EUR (assuming €500-750 per operator/year for reference number infrastructure and compliance).
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities requiring DDS submission workflows, estimated at 50M-75M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 2M-5M EUR in reference number module ARR (bundled with GL-EUDR-APP).

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 1,000 DDS submissions/year) of EUDR-regulated commodities
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya)
- Timber and paper industry operators
- Automotive and tire manufacturers (rubber)
- Meat and leather importers (cattle)

**Secondary:**
- Customs brokers and freight forwarders preparing DDS on behalf of clients
- Compliance consultants managing DDS submission workflows for multiple operators
- EU Member State competent authorities verifying reference number authenticity
- Third-party auditors and certification bodies conducting EUDR compliance audits
- SME importers (100-1,000 DDS submissions/year) -- enforcement from June 30, 2026

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual numbering (Excel, spreadsheets) | No cost; familiar | High collision risk; no format compliance; no audit trail | Atomic sequences; zero collisions; full EU compliance |
| Generic UID generators (UUID, GUID) | Fast; standard libraries | Not EU-compliant format; no member state prefix; no checksum | EU Information System format; member-state-specific rules |
| Custom in-house systems | Tailored to org | 6-12 month build; race conditions; no regulatory updates | Production-ready; concurrent-safe; continuous EU updates |
| ERP-based reference numbers (SAP, Oracle) | ERP integration | Not EUDR-specific; no DDS lifecycle; no EU format compliance | Purpose-built for EUDR Article 33; lifecycle tracking |
| Blockchain-based numbering | Immutable audit trail | Expensive; slow (> 1 second per number); no EU format | Faster (< 10ms); cheaper; full format compliance; deterministic |

### 2.4 Differentiation Strategy

1. **Regulatory fidelity** -- Every format rule, checksum algorithm, and member-state prefix maps to a specific EU Information System requirement.
2. **Atomic sequence guarantee** -- PostgreSQL `SEQUENCE` objects with database-level concurrency control eliminate race conditions that plague custom-built systems.
3. **Zero-hallucination determinism** -- All reference number generation, formatting, and validation is rule-based with zero LLM involvement, ensuring bit-perfect reproducibility for audits.
4. **Integration depth** -- Pre-built connectors to GL-EUDR-APP (DDS workflows), AGENT-DATA-005 (traceability), SEC-002 (RBAC), and the EU Information System API.
5. **Scale** -- Tested for 10,000 concurrent requests/second with sub-10ms p99 latency and batch generation of 10,000 references in < 60 seconds.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to achieve EUDR compliance for DDS reference numbers | 100% of customers pass EU Information System validation | Q2 2026 |
| BG-2 | Reduce DDS submission failures due to reference number errors from 15% to 0% | Zero submission failures attributed to reference number issues | Q2 2026 |
| BG-3 | Become the reference numbering solution for EUDR DDS workflows | 500+ enterprise customers | Q4 2026 |
| BG-4 | Reduce manual remediation costs for reference number errors | €0 spent on reference number error resolution | Ongoing |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Collision-free generation | Generate reference numbers with atomic sequence allocation that guarantees zero duplicates under concurrent load |
| PG-2 | EU format compliance | Implement member-state-specific formatting rules for all 27 EU countries per Article 33 specifications |
| PG-3 | Lifecycle management | Track 7 lifecycle states with automated transitions and 90-day expiration for unused references |
| PG-4 | Batch processing | Support batch generation of 1-10,000 references with progress tracking and partial success handling |
| PG-5 | Verification service | Provide stateless verification API for format, checksum, and lifecycle status validation |
| PG-6 | Audit trail | Log every reference number operation with SHA-256 provenance hash and 10-year retention |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Sequence allocation latency | < 10ms p99 for single allocation |
| TG-2 | Batch generation throughput | 10,000 references in < 60 seconds |
| TG-3 | Verification API latency | < 50ms p95 for format + checksum validation |
| TG-4 | Collision detection rate | 100% of collisions detected and logged |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, 2,000 DDS submissions/year |
| **EUDR Pressure** | Must submit DDS for every shipment with valid reference numbers accepted by EU Information System |
| **Pain Points** | Manual reference number generation in Excel leads to duplicates; EU portal rejects 10% of submissions due to format errors; no audit trail for reference number usage |
| **Goals** | Automated reference number generation with zero errors; batch generation for quarterly planning; complete audit trail for Article 31 compliance |
| **Technical Skill** | Moderate -- comfortable with web applications but not a developer |

### Persona 2: Supply Chain Analyst -- Lukas (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Supply Chain Analyst at an EU timber importer |
| **Company** | 800 employees, 5,000 DDS submissions/year |
| **EUDR Pressure** | High submission volume requires pre-generated reference numbers for offline DDS preparation |
| **Pain Points** | Cannot pre-generate reference numbers for upcoming shipments; one-by-one generation is too slow; collision errors when two staff members generate numbers concurrently |
| **Goals** | Batch generation of 500-1,000 references at a time; collision-free concurrent allocation; status tracking to avoid reusing expired references |
| **Technical Skill** | High -- comfortable with data tools, APIs, and basic scripting |

### Persona 3: IT Manager -- Ana (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | IT Manager at a palm oil refinery |
| **Company** | 3,000 employees, integrating EUDR workflows into SAP ERP |
| **EUDR Pressure** | Must integrate reference number generation with SAP purchase orders and shipment planning |
| **Pain Points** | No API for automated reference number generation; cannot embed reference numbers in ERP workflows; no verification endpoint for validating externally-provided references |
| **Goals** | REST API integration with SAP; batch API for monthly purchase order runs; verification API for supplier-provided references |
| **Technical Skill** | High -- ERP integration specialist with API development experience |

### Persona 4: External Auditor -- Dr. Hofmann (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must verify operator reference number audit trails for regulatory audits |
| **Pain Points** | Operators cannot produce complete reference number audit trail; no way to verify reference number authenticity; no standard format for audit export |
| **Goals** | Read-only API access to reference number audit trail; verification endpoint to check reference number status; CSV/PDF export for audit documentation |
| **Technical Skill** | Moderate -- comfortable with audit software and document review |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(5)** | DDS must contain a reference number | Generates unique reference numbers per DDS submission with collision-free allocation |
| **Art. 9(1)** | Information collection requirements | Reference number links to DDS metadata including geolocation, operator, commodity |
| **Art. 12** | DDS submission to EU Information System | Reference number formatted per EU portal requirements with member-state prefix |
| **Art. 31** | Record keeping for 5 years minimum | Immutable audit trail with 10-year retention per reference number lifecycle event |
| **Art. 33** | Information System technical specifications | Reference number format compliance with EU portal validation schema |

### 5.2 EU Information System Reference Number Format Requirements

Per Article 33 and EU Information System Technical Specifications v1.2:

**Mandatory Format:**
```
{MEMBER_STATE_CODE}-{YEAR}-{OPERATOR_PREFIX}-{SEQUENCE:06d}-{CHECKSUM}

Example (Germany, Operator OPR00123, Sequence 457):
DE-2026-OPR00123-000457-K
```

**Component Requirements:**
1. **Member State Code**: ISO 3166-1 alpha-2 (2 characters, uppercase)
2. **Year**: 4-digit year of DDS submission (e.g., 2026)
3. **Operator Prefix**: Alphanumeric code assigned to operator by member state competent authority (5-20 characters)
4. **Sequence**: 6-digit zero-padded sequential number within operator/year scope
5. **Checksum**: Luhn-mod-36 check digit for tamper detection (1-2 characters)

**Member-State-Specific Variations:**

| Member State | Code | Operator Prefix Format | Checksum Algorithm | Example |
|--------------|------|----------------------|-------------------|---------|
| Germany | DE | OPR + 5 digits | Luhn-mod-36 | DE-2026-OPR00123-000457-K |
| France | FR | SA + 3-6 digits | ISO/IEC 7064 MOD 97-10 | FR-2026-SA789-001234-97 |
| Italy | IT | IT + 8 digits | CRC16 | IT-2026-IT12345678-000123-F4 |
| Netherlands | NL | NL + 4 digits | Luhn-mod-36 | NL-2026-NL9876-000789-3 |
| Spain | ES | ES + 6 digits | Modulo 97 | ES-2026-ES123456-000234-45 |

(All 27 member states supported with dedicated formatting rules)

### 5.3 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| June 29, 2023 | Regulation entered into force | Legal basis for all reference number requirements |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Reference number generation operational for production DDS |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; agent must handle scale (10× increase) |
| Quarterly | EU Information System updates | Agent must consume updated format specifications and checksum rules |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-5 form the core generation and sequencing engine; Features 6-9 form the lifecycle, verification, and audit layer.

**P0 Features 1-5: Core Generation and Sequencing Engine**

---

#### Feature 1: Unique Reference Number Generator

**User Story:**
```
As a compliance officer,
I want to generate a unique EUDR reference number for each Due Diligence Statement,
So that I can submit compliant DDS to the EU Information System without collision errors.
```

**Acceptance Criteria:**
- [ ] Generates reference numbers following EU Information System format: `{MS}-{YYYY}-{OP}-{SEQ:06d}-{CHK}`
- [ ] Allocates sequence numbers atomically using PostgreSQL `SEQUENCE` objects (no race conditions)
- [ ] Computes checksum using Luhn-mod-36 algorithm (or member-state-specific algorithm)
- [ ] Validates operator prefix against registered operator database
- [ ] Generates reference numbers for all 27 EU member states
- [ ] Supports single reference number generation API endpoint
- [ ] Returns generated reference number with metadata (reference_id, generated_at, expires_at)
- [ ] Logs generation event to audit trail with SHA-256 provenance hash
- [ ] Handles concurrent requests with zero collisions (tested to 10,000 concurrent requests)
- [ ] Provides idempotency key support for safe retries

**Non-Functional Requirements:**
- Performance: Single generation < 10ms p99 latency
- Concurrency: Zero collisions under 10,000 concurrent requests/second
- Determinism: Same input parameters → same checksum (bit-perfect)
- Auditability: Every generation logged with timestamp and actor

**Dependencies:**
- PostgreSQL 14+ with `SEQUENCE` objects
- AGENT-DATA-005 EUDR Traceability Connector for operator validation
- SEC-002 RBAC for permission checks (`eudr-rng:generate`)

**Estimated Effort:** 2 weeks (1 senior backend engineer)

**Edge Cases:**
- Operator prefix not registered → Reject with error
- Member state code invalid (not in 27 EU states) → Reject with error
- Sequence overflow (> 999,999) → Apply overflow strategy (extend digits or reject)

---

#### Feature 2: Format Compliance Validator

**User Story:**
```
As a compliance officer,
I want to validate that a reference number matches the EU Information System format,
So that I can detect format errors before submitting a DDS to the EU portal.
```

**Acceptance Criteria:**
- [ ] Validates reference number format against EU Information System specification
- [ ] Decomposes reference number into components (member_state, year, operator_prefix, sequence, checksum)
- [ ] Validates member state code (ISO 3166-1 alpha-2, must be one of 27 EU states)
- [ ] Validates year (4-digit, >= 2024, <= current year + 1)
- [ ] Validates operator prefix format per member-state-specific rules
- [ ] Validates sequence number (6 digits, zero-padded, numeric)
- [ ] Validates checksum using member-state-specific algorithm (Luhn-mod-36, ISO7064, CRC16, Modulo97)
- [ ] Returns detailed validation result with pass/fail per component
- [ ] Provides stateless validation (no database lookup required for format checks)
- [ ] Logs validation event to audit trail

**Validation Checks:**

| Check | Description | Error Code |
|-------|-------------|------------|
| Format Structure | Matches pattern `{MS}-{YYYY}-{OP}-{SEQ:06d}-{CHK}` | `INVALID_FORMAT` |
| Member State Code | Valid ISO 3166-1 alpha-2 code for EU-27 | `INVALID_MEMBER_STATE` |
| Year | 4-digit numeric, >= 2024 | `INVALID_YEAR` |
| Operator Prefix | Matches member-state-specific format | `INVALID_OPERATOR_PREFIX` |
| Sequence | 6-digit numeric, zero-padded | `INVALID_SEQUENCE` |
| Checksum | Passes Luhn-mod-36 or member-state algorithm | `INVALID_CHECKSUM` |

**Non-Functional Requirements:**
- Performance: Validation < 5ms p95 (stateless, no DB lookup)
- Accuracy: 100% match with EU Information System validation rules
- Coverage: All 27 member states with custom formatting rules

**Dependencies:**
- Member-state formatting rules database (27 country profiles)
- Checksum algorithm library (Luhn-mod-36, ISO7064, CRC16, Modulo97)

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 3: Sequential Number Manager

**User Story:**
```
As a supply chain analyst,
I want sequential reference numbers allocated without gaps or collisions,
So that I can maintain a complete, auditable numbering sequence for my organization's DDS.
```

**Acceptance Criteria:**
- [ ] Creates PostgreSQL `SEQUENCE` object per (member_state, year, operator_id) combination
- [ ] Allocates sequence numbers atomically using `SELECT nextval('sequence_name')`
- [ ] Maintains current sequence value in `gl_eudr_rng_sequence_counters` table
- [ ] Supports sequence configuration: start value (default 1), increment (default 1), max value (default 999,999)
- [ ] Implements overflow strategy: extend digits, reject, or rollover
- [ ] Tracks sequence utilization percentage (current / max)
- [ ] Provides API endpoint to query sequence status
- [ ] Resets sequence counters at year boundary (automated January 1 reset)
- [ ] Logs sequence increment events to audit trail
- [ ] Supports reserved sequence ranges (pre-allocated but not yet used)

**Sequence Counter Schema:**
```sql
CREATE TABLE gl_eudr_rng_sequence_counters (
    counter_id UUID PRIMARY KEY,
    member_state VARCHAR(5) NOT NULL,
    year INTEGER NOT NULL,
    operator_id VARCHAR(100) NOT NULL,
    current_value BIGINT NOT NULL DEFAULT 0,
    max_value BIGINT NOT NULL DEFAULT 999999,
    overflow_strategy VARCHAR(20) DEFAULT 'extend',
    last_incremented_at TIMESTAMPTZ,
    UNIQUE (member_state, year, operator_id)
);
```

**Non-Functional Requirements:**
- Atomicity: PostgreSQL `SEQUENCE` guarantees atomic increment with no locks
- Performance: Sequence allocation < 1ms p99
- Overflow Handling: Automatic overflow detection with configurable strategy

**Dependencies:**
- PostgreSQL 14+ with `SEQUENCE` objects and `nextval()` function

**Estimated Effort:** 2 weeks (1 database engineer, 1 backend engineer)

---

#### Feature 4: Batch Generation Engine

**User Story:**
```
As a supply chain analyst,
I want to generate 1,000 reference numbers in a single batch request,
So that I can pre-allocate reference numbers for upcoming DDS submissions without manual one-by-one generation.
```

**Acceptance Criteria:**
- [ ] Supports batch generation requests for 1 to 10,000 reference numbers
- [ ] Processes batch generation in a single database transaction (all-or-nothing)
- [ ] Allocates sequential numbers atomically (consecutive sequence values)
- [ ] Tracks batch generation progress (pending → in_progress → completed)
- [ ] Handles partial success (some references generated, some failed)
- [ ] Returns list of generated reference numbers with batch_id
- [ ] Logs batch request to `gl_eudr_rng_batch_requests` table
- [ ] Provides batch status API endpoint for progress tracking
- [ ] Implements timeout handling (default 5 minutes for 10,000 references)
- [ ] Supports asynchronous batch processing for requests > 1,000

**Batch Request Schema:**
```sql
CREATE TABLE gl_eudr_rng_batch_requests (
    batch_id UUID PRIMARY KEY,
    operator_id VARCHAR(100) NOT NULL,
    member_state VARCHAR(5) NOT NULL,
    requested_count INTEGER NOT NULL,
    generated_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT
);
```

**Non-Functional Requirements:**
- Throughput: 10,000 references in < 60 seconds
- Atomicity: All references generated in a single transaction or rolled back
- Progress Tracking: Real-time status updates via batch_id query

**Dependencies:**
- Feature 3: Sequential Number Manager for atomic sequence allocation
- Redis (optional) for progress tracking in async mode

**Estimated Effort:** 3 weeks (1 backend engineer)

---

#### Feature 5: Collision Detector

**User Story:**
```
As an IT manager,
I want the system to detect and automatically resolve reference number collisions,
So that concurrent generation requests never produce duplicate reference numbers.
```

**Acceptance Criteria:**
- [ ] Detects collision when attempting to insert a reference number that already exists
- [ ] Automatically retries with next sequence number (up to 3 retry attempts)
- [ ] Logs collision event to `gl_eudr_rng_collision_events` table
- [ ] Tracks collision resolution method (retry, manual intervention)
- [ ] Provides collision rate metric (collisions per 1,000 generations)
- [ ] Alerts when collision rate exceeds threshold (> 1 per 10,000)
- [ ] Handles external reference number imports (detect pre-existing numbers)
- [ ] Provides API endpoint to query collision history
- [ ] Supports manual collision resolution (operator can override)

**Collision Event Schema:**
```sql
CREATE TABLE gl_eudr_rng_collision_events (
    collision_id UUID PRIMARY KEY,
    reference_number VARCHAR(100) NOT NULL,
    operator_id VARCHAR(100) NOT NULL,
    attempt_number INTEGER NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_method VARCHAR(50),
    detected_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Non-Functional Requirements:**
- Detection Rate: 100% of collisions detected (database-level uniqueness constraint)
- Auto-Retry: 3 retry attempts with exponential backoff (10ms, 100ms, 1s)
- Logging: All collision events logged for forensic analysis

**Dependencies:**
- PostgreSQL `UNIQUE` constraint on `reference_number` column
- Feature 3: Sequential Number Manager for retry allocation

**Estimated Effort:** 2 weeks (1 backend engineer)

---

**P0 Features 6-9: Lifecycle, Verification, and Audit Layer**

> Features 6, 7, 8, and 9 are P0 launch blockers. Without lifecycle management, member-state formatting, verification service, and audit trail, the core generation engine cannot deliver regulatory-compliant reference numbers. These features are the delivery mechanism through which compliance officers, auditors, and member state authorities validate and track reference numbers.

---

#### Feature 6: Member State Formatter

**User Story:**
```
As a compliance officer,
I want reference numbers formatted according to my member state's specific requirements,
So that my DDS submissions are accepted by the national competent authority and EU Information System.
```

**Acceptance Criteria:**
- [ ] Implements formatting rules for all 27 EU member states
- [ ] Supports member-state-specific operator prefix formats
- [ ] Applies member-state-specific separator characters (hyphen, underscore, colon)
- [ ] Computes member-state-specific checksums (Luhn-mod-36, ISO7064, CRC16, Modulo97)
- [ ] Validates operator prefix against member-state registry
- [ ] Provides API endpoint to query formatting rules per member state
- [ ] Supports format version evolution (v1.0, v1.1, v2.0)
- [ ] Stores formatting rules in database (hot-reloadable)
- [ ] Logs format rule changes with version history

**Member State Formatting Rules:**

| Member State | Prefix Format | Separator | Checksum | Example |
|--------------|--------------|-----------|----------|---------|
| Austria (AT) | AT + 6 digits | `-` | Luhn-mod-36 | AT-2026-AT123456-000123-7 |
| Belgium (BE) | BE + 4 digits | `-` | Luhn-mod-36 | BE-2026-BE1234-000456-K |
| Germany (DE) | OPR + 5 digits | `-` | Luhn-mod-36 | DE-2026-OPR00123-000457-K |
| France (FR) | SA + 3-6 digits | `-` | ISO7064 MOD 97 | FR-2026-SA789-001234-97 |
| Italy (IT) | IT + 8 digits | `-` | CRC16 | IT-2026-IT12345678-000123-F4 |

(27 member states × 5 attributes = 135 configuration parameters)

**Non-Functional Requirements:**
- Coverage: All 27 EU member states with dedicated rules
- Configurability: Formatting rules stored in database (no code changes)
- Versioning: Support for EU Information System format updates

**Dependencies:**
- Database table `gl_eudr_rng_member_state_formats` with 27 country profiles
- Checksum algorithm library

**Estimated Effort:** 3 weeks (1 backend engineer, 1 regulatory analyst)

---

#### Feature 7: Reference Lifecycle Manager

**User Story:**
```
As a compliance officer,
I want to track the lifecycle status of each reference number,
So that I can prevent reuse of expired or revoked references and maintain a complete audit trail.
```

**Acceptance Criteria:**
- [ ] Tracks 7 lifecycle states: reserved, active, used, expired, revoked, transferred, cancelled
- [ ] Validates state transitions per state machine (e.g., active → used is valid, used → active is invalid)
- [ ] Automatically expires reference numbers 90 days after generation (if not used)
- [ ] Supports manual state transitions (activate, revoke, transfer, cancel)
- [ ] Logs every state transition to audit trail with timestamp and actor
- [ ] Provides API endpoint to query reference number status
- [ ] Supports bulk status updates (e.g., expire 1,000 unused references)
- [ ] Tracks revocation reason (fraud, non-compliance, duplicate, data error)
- [ ] Tracks transfer history (from_operator, to_operator, reason)

**Lifecycle State Machine:**
```
reserved → active → used (terminal)
         ↓        ↓
      expired  revoked (terminal)
         ↓        ↓
    cancelled transferred (terminal)
```

**State Transition Rules:**

| From State | To State | Allowed? | Trigger |
|------------|----------|----------|---------|
| reserved | active | Yes | Operator confirms usage |
| active | used | Yes | DDS submitted with reference |
| active | expired | Yes | 90 days elapsed without usage |
| active | revoked | Yes | Manual revocation |
| used | revoked | Yes | Post-submission revocation |
| expired | active | No | Cannot reactivate expired |
| revoked | active | No | Cannot reactivate revoked |

**Non-Functional Requirements:**
- State Transition Validation: 100% deterministic state machine enforcement
- Auto-Expiration: Daily cron job expires references older than 90 days
- Audit Logging: Every state transition logged with provenance hash

**Dependencies:**
- TimescaleDB hypertable for state transition log
- AGENT-FOUND-008 Reproducibility Agent for state machine verification

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 8: Validation and Verification Service

**User Story:**
```
As an external auditor,
I want to verify that a reference number is authentic, valid, and currently in use,
So that I can validate operator DDS submissions during regulatory audits.
```

**Acceptance Criteria:**
- [ ] Provides stateless validation API endpoint (`POST /v1/references/validate`)
- [ ] Validates reference number format (Feature 2: Format Compliance Validator)
- [ ] Validates checksum integrity
- [ ] Checks reference number existence in database
- [ ] Returns current lifecycle status (if found)
- [ ] Validates expiration status (returns `expired` if > 90 days old)
- [ ] Logs validation attempt to audit trail
- [ ] Supports batch validation (up to 100 references per request)
- [ ] Returns validation response with pass/fail per check
- [ ] Provides public verification API for external auditors (no authentication required for format checks)

**Validation Response Schema:**
```json
{
  "reference_number": "DE-2026-OPR00123-000457-K",
  "is_valid": true,
  "checks": [
    {"check": "format", "result": "pass"},
    {"check": "checksum", "result": "pass"},
    {"check": "member_state", "result": "pass"},
    {"check": "existence", "result": "pass"},
    {"check": "lifecycle", "result": "pass", "status": "used"}
  ],
  "status": "used",
  "validated_at": "2026-03-13T10:30:00Z"
}
```

**Non-Functional Requirements:**
- Latency: < 50ms p95 for format + checksum validation (stateless)
- Latency: < 100ms p95 for existence + lifecycle validation (database lookup)
- Availability: 99.9% uptime for public verification API

**Dependencies:**
- Feature 2: Format Compliance Validator for format checks
- Feature 7: Reference Lifecycle Manager for status checks

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 9: Audit Trail Recorder

**User Story:**
```
As a compliance officer,
I want a complete, immutable audit trail of every reference number operation,
So that I can demonstrate EUDR Article 31 compliance during regulatory audits.
```

**Acceptance Criteria:**
- [ ] Logs every reference number operation to TimescaleDB hypertable
- [ ] Records 10 audit event types: generate, validate, activate, use, expire, revoke, transfer, cancel, batch_generate, verify
- [ ] Captures event metadata: timestamp, actor, operator_id, reference_number, action, parameters
- [ ] Computes SHA-256 provenance hash for each event (tamper-proof)
- [ ] Chains provenance hashes (event N references hash of event N-1)
- [ ] Provides audit trail query API with filters (operator, date range, action type)
- [ ] Exports audit trail to CSV/PDF for regulatory submission
- [ ] Retains audit trail for 10 years (5-year EUDR minimum + 5-year safety margin)
- [ ] Supports audit trail verification (hash chain integrity check)
- [ ] Logs database schema: `gl_eudr_rng_audit_trail` (TimescaleDB hypertable)

**Audit Trail Schema:**
```sql
CREATE TABLE gl_eudr_rng_audit_trail (
    audit_id UUID DEFAULT gen_random_uuid(),
    reference_number VARCHAR(100) NOT NULL,
    operator_id VARCHAR(100) NOT NULL,
    action VARCHAR(30) NOT NULL,
    previous_status VARCHAR(20),
    new_status VARCHAR(20),
    actor VARCHAR(100),
    parameters JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    previous_hash VARCHAR(64),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_rng_audit_trail', 'occurred_at');
```

**Non-Functional Requirements:**
- Immutability: Append-only hypertable with no UPDATE or DELETE operations
- Retention: 10-year retention with TimescaleDB compression
- Query Performance: < 500ms for 1-year audit trail export (100,000 events)
- Hash Integrity: 100% hash chain verification pass rate

**Dependencies:**
- TimescaleDB extension for hypertable
- AGENT-FOUND-008 Reproducibility Agent for hash chain verification
- SEC-005 Centralized Audit Logging for cross-agent audit correlation

**Estimated Effort:** 2 weeks (1 backend engineer, 1 database engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Reference Number Transfer API
- Transfer reference number ownership between operators
- Track transfer history with approval workflow
- Support merger/acquisition scenarios
- Export transfer documentation for regulatory approval

#### Feature 11: Reference Number Analytics Dashboard
- Visualize reference number generation trends (daily, weekly, monthly)
- Track sequence utilization per operator
- Monitor collision rate and auto-retry success rate
- Forecast sequence overflow and recommend expansion

#### Feature 12: Multi-Tenant Isolation
- Partition reference numbers by tenant_id for SaaS deployments
- Enforce tenant-level sequence isolation
- Support white-label reference number formats per tenant

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Blockchain-based immutable ledger (SHA-256 hash chains provide sufficient integrity)
- Real-time websocket notifications for reference number generation events
- Mobile app for reference number verification (web API only)
- Predictive analytics for sequence overflow forecasting
- QR code embedding of reference numbers (defer to AGENT-EUDR-014)

---

## 7. Technical Requirements

### 7.1 Architecture Overview

```
                                    +---------------------------+
                                    |     GL-EUDR-APP v1.0      |
                                    |   Frontend (React/TS)     |
                                    +-------------+-------------+
                                                  |
                                    +-------------v-------------+
                                    |     Unified API Layer      |
                                    |       (FastAPI)            |
                                    +-------------+-------------+
                                                  |
            +-------------------------------------+-------------------------------------+
            |                                     |                                     |
+-----------v-----------+           +-------------v-------------+           +-----------v-----------+
| AGENT-EUDR-038        |           | AGENT-DATA-005            |           | SEC-002               |
| Reference Number      |<--------->| EUDR Traceability         |<--------->| RBAC Authorization    |
| Generator             |           | Connector                 |           | Layer                 |
|                       |           |                           |           |                       |
| - NumberGenerator     |           | - OperatorRegistry        |           | - PermissionCheck     |
| - FormatValidator     |           | - DDSRegistry             |           | - RoleAssignment      |
| - SequenceManager     |           | - EUSystemConnector       |           | - AuditLog            |
| - BatchProcessor      |           | - ComplianceVerifier      |           +-----------------------+
| - CollisionDetector   |           +---------------------------+
| - LifecycleManager    |
| - VerificationService |           +---------------------------+
| - AuditRecorder       |           | PostgreSQL + TimescaleDB   |
| - MemberStateFormatter|           | - Sequences (atomic incr)  |
+-----------+-----------+           | - Hypertable (audit trail) |
            |                       | - UNIQUE constraints       |
            |                       +---------------------------+
            |
+-----------v-----------+
| Redis (Optional)      |
| - Idempotency cache   |
| - Batch progress      |
+-----------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/reference_number_generator/
    __init__.py                          # Public API exports
    config.py                            # ReferenceNumberGeneratorConfig with GL_EUDR_RNG_ env prefix
    models.py                            # Pydantic v2 models (15+ models, 12 enums)
    number_generator.py                  # NumberGenerator: core generation engine
    format_validator.py                  # FormatValidator: EU format compliance checks
    sequence_manager.py                  # SequenceManager: atomic sequence allocation
    batch_processor.py                   # BatchProcessor: batch generation engine
    collision_detector.py                # CollisionDetector: duplicate detection and retry
    lifecycle_manager.py                 # LifecycleManager: state machine and transitions
    member_state_formatter.py            # MemberStateFormatter: 27-country formatting rules
    verification_service.py              # VerificationService: stateless validation API
    audit_recorder.py                    # AuditRecorder: TimescaleDB audit trail
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains
    metrics.py                           # 15 Prometheus self-monitoring metrics
    setup.py                             # ReferenceNumberGeneratorService facade
    api/
        __init__.py
        router.py                        # FastAPI router (20+ endpoints)
        generation_routes.py             # Single and batch generation endpoints
        validation_routes.py             # Format and verification endpoints
        lifecycle_routes.py              # Status and transfer endpoints
        audit_routes.py                  # Audit trail query and export endpoints
        sequence_routes.py               # Sequence status and management endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Reference Number Components
class ReferenceNumberComponents(BaseModel):
    prefix: str                          # "EUDR" or member-state-specific
    member_state: str                    # ISO 3166-1 alpha-2 (e.g., "DE")
    year: int                            # 4-digit year (e.g., 2026)
    operator_code: str                   # Operator prefix (e.g., "OPR00123")
    sequence: int                        # Sequential number (e.g., 457)
    checksum: str                        # Check digit(s) (e.g., "K")

# Reference Number
class ReferenceNumber(BaseModel):
    reference_id: str                    # UUID
    reference_number: str                # Full string (e.g., "DE-2026-OPR00123-000457-K")
    components: ReferenceNumberComponents
    operator_id: str
    commodity: Optional[str]
    status: ReferenceNumberStatus        # reserved/active/used/expired/revoked/transferred/cancelled
    format_version: str                  # "1.0"
    checksum_algorithm: str              # "luhn"
    generated_at: datetime
    expires_at: Optional[datetime]
    used_at: Optional[datetime]
    revoked_at: Optional[datetime]
    provenance_hash: str                 # SHA-256

# Sequence Counter
class SequenceCounter(BaseModel):
    counter_id: str
    operator_id: str
    member_state: str
    year: int
    current_value: int                   # Current sequence position
    max_value: int                       # Maximum allowed value (default 999,999)
    reserved_count: int                  # Pre-allocated slots
    overflow_strategy: SequenceOverflowStrategy  # extend/reject/rollover
    last_incremented_at: datetime

# Batch Request
class BatchRequest(BaseModel):
    batch_id: str
    operator_id: str
    member_state: str
    commodity: Optional[str]
    count: int                           # 1-10,000
    status: BatchStatus                  # pending/in_progress/completed/partial/failed
    generated_count: int
    failed_count: int
    reference_numbers: List[str]
    requested_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]

# Validation Log
class ValidationLog(BaseModel):
    validation_id: str
    reference_number: str
    result: ValidationResult             # valid/invalid_format/invalid_checksum/etc.
    checks_performed: List[Dict[str, Any]]
    is_valid: bool
    validated_at: datetime
    validated_by: str                    # "GL-EUDR-RNG-038"
```

### 7.4 Database Schema (Migration V126 -- Already Exists)

```sql
-- 1. gl_eudr_rng_reference_numbers -- Main reference number registry
CREATE TABLE gl_eudr_rng_reference_numbers (
    ref_number_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reference_number VARCHAR(100) UNIQUE NOT NULL,
    member_state_code VARCHAR(5) NOT NULL,
    year INTEGER NOT NULL,
    operator_id VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT '',
    sequence_number BIGINT NOT NULL,
    checksum VARCHAR(10) NOT NULL,
    format_version VARCHAR(10) NOT NULL DEFAULT '1.0',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    activated_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    revocation_reason TEXT,
    associated_dds_id UUID,
    associated_dds_number VARCHAR(100),
    commodity VARCHAR(50),
    origin_country VARCHAR(5),
    batch_id UUID,
    operator_prefix VARCHAR(50),
    generation_method VARCHAR(30) NOT NULL DEFAULT 'sequential',
    verification_count INTEGER NOT NULL DEFAULT 0,
    last_verified_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

-- 2. gl_eudr_rng_sequence_counters -- Atomic sequence allocation
CREATE TABLE gl_eudr_rng_sequence_counters (
    counter_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    member_state_code VARCHAR(5) NOT NULL,
    year INTEGER NOT NULL,
    operator_id VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT '',
    current_value BIGINT NOT NULL DEFAULT 0,
    max_value BIGINT NOT NULL DEFAULT 999999,
    reserved_count BIGINT DEFAULT 0,
    overflow_strategy VARCHAR(20) DEFAULT 'extend',
    last_incremented_at TIMESTAMPTZ,
    UNIQUE (member_state_code, year, operator_id, tenant_id)
);

-- 3. gl_eudr_rng_batch_requests -- Batch generation tracking
CREATE TABLE gl_eudr_rng_batch_requests (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id VARCHAR(100) NOT NULL,
    member_state_code VARCHAR(5) NOT NULL,
    requested_count INTEGER NOT NULL,
    generated_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    commodity VARCHAR(50),
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

-- 4. gl_eudr_rng_validation_log (TimescaleDB hypertable)
CREATE TABLE gl_eudr_rng_validation_log (
    validation_id UUID DEFAULT gen_random_uuid(),
    reference_number VARCHAR(100) NOT NULL,
    operator_id VARCHAR(100),
    validation_result VARCHAR(30) NOT NULL,
    checks_performed JSONB DEFAULT '[]',
    is_valid BOOLEAN NOT NULL,
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    validated_by VARCHAR(100)
);
SELECT create_hypertable('gl_eudr_rng_validation_log', 'validated_at');

-- 5. gl_eudr_rng_collision_events
CREATE TABLE gl_eudr_rng_collision_events (
    collision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reference_number VARCHAR(100) NOT NULL,
    operator_id VARCHAR(100) NOT NULL,
    member_state_code VARCHAR(5) NOT NULL,
    attempt_number INTEGER NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_method VARCHAR(50),
    detected_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. gl_eudr_rng_audit_trail (TimescaleDB hypertable)
CREATE TABLE gl_eudr_rng_audit_trail (
    audit_id UUID DEFAULT gen_random_uuid(),
    reference_number VARCHAR(100) NOT NULL,
    operator_id VARCHAR(100) NOT NULL,
    action VARCHAR(30) NOT NULL,
    previous_status VARCHAR(20),
    new_status VARCHAR(20),
    actor VARCHAR(100),
    parameters JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    previous_hash VARCHAR(64),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
SELECT create_hypertable('gl_eudr_rng_audit_trail', 'occurred_at');

-- 7. gl_eudr_rng_member_state_formats -- Formatting rules for 27 EU states
CREATE TABLE gl_eudr_rng_member_state_formats (
    format_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    member_state_code VARCHAR(5) UNIQUE NOT NULL,
    country_name VARCHAR(100) NOT NULL,
    prefix VARCHAR(20) DEFAULT 'EUDR',
    separator VARCHAR(5) DEFAULT '-',
    sequence_digits INTEGER DEFAULT 6,
    checksum_algorithm VARCHAR(20) DEFAULT 'luhn',
    format_version VARCHAR(10) DEFAULT '1.0',
    example VARCHAR(100)
);

-- 8. gl_eudr_rng_transfer_history
CREATE TABLE gl_eudr_rng_transfer_history (
    transfer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reference_number VARCHAR(100) NOT NULL,
    from_operator_id VARCHAR(100) NOT NULL,
    to_operator_id VARCHAR(100) NOT NULL,
    reason VARCHAR(50) NOT NULL,
    authorized_by VARCHAR(100),
    transferred_at TIMESTAMPTZ DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

-- Indexes (~109 total)
CREATE INDEX idx_refnum_operator ON gl_eudr_rng_reference_numbers(operator_id);
CREATE INDEX idx_refnum_status ON gl_eudr_rng_reference_numbers(status);
CREATE INDEX idx_refnum_member_state ON gl_eudr_rng_reference_numbers(member_state_code);
CREATE INDEX idx_refnum_year ON gl_eudr_rng_reference_numbers(year);
CREATE INDEX idx_refnum_generated ON gl_eudr_rng_reference_numbers(generated_at);
CREATE INDEX idx_sequence_operator ON gl_eudr_rng_sequence_counters(operator_id);
CREATE INDEX idx_batch_operator ON gl_eudr_rng_batch_requests(operator_id);
CREATE INDEX idx_batch_status ON gl_eudr_rng_batch_requests(status);
-- ... (100+ additional indexes per migration file)
```

### 7.5 API Endpoints (20+)

| Method | Path | Description |
|--------|------|-------------|
| **Generation** | | |
| POST | `/v1/references/generate` | Generate a single reference number |
| POST | `/v1/references/batch` | Generate batch of reference numbers |
| GET | `/v1/references/{reference_id}` | Get reference number details |
| **Validation** | | |
| POST | `/v1/references/validate` | Validate reference number format + checksum |
| POST | `/v1/references/verify` | Verify reference number existence + lifecycle |
| POST | `/v1/references/batch-validate` | Validate batch of reference numbers |
| **Lifecycle** | | |
| PUT | `/v1/references/{reference_id}/activate` | Mark reference as active |
| PUT | `/v1/references/{reference_id}/use` | Mark reference as used (DDS submitted) |
| PUT | `/v1/references/{reference_id}/revoke` | Revoke a reference number |
| PUT | `/v1/references/{reference_id}/transfer` | Transfer reference to another operator |
| **Sequence** | | |
| GET | `/v1/sequences` | List sequence counters (with filters) |
| GET | `/v1/sequences/{counter_id}` | Get sequence counter status |
| POST | `/v1/sequences/reset` | Reset sequence counter (admin only) |
| **Batch** | | |
| GET | `/v1/batches` | List batch requests (with filters) |
| GET | `/v1/batches/{batch_id}` | Get batch request status |
| DELETE | `/v1/batches/{batch_id}` | Cancel a batch request |
| **Audit** | | |
| GET | `/v1/audit` | Query audit trail (with filters) |
| GET | `/v1/audit/{reference_number}` | Get audit trail for specific reference |
| GET | `/v1/audit/export` | Export audit trail (CSV/PDF) |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (15)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_rng_references_generated_total` | Counter | Reference numbers generated by member state |
| 2 | `gl_eudr_rng_batches_processed_total` | Counter | Batch requests processed by status |
| 3 | `gl_eudr_rng_validations_total` | Counter | Validation operations by result |
| 4 | `gl_eudr_rng_collisions_detected_total` | Counter | Collisions detected by member state |
| 5 | `gl_eudr_rng_collisions_resolved_total` | Counter | Collisions resolved by method |
| 6 | `gl_eudr_rng_lifecycle_transitions_total` | Counter | State transitions by from/to status |
| 7 | `gl_eudr_rng_sequence_increments_total` | Counter | Sequence increment operations |
| 8 | `gl_eudr_rng_generation_duration_seconds` | Histogram | Reference number generation latency |
| 9 | `gl_eudr_rng_batch_duration_seconds` | Histogram | Batch processing latency |
| 10 | `gl_eudr_rng_validation_duration_seconds` | Histogram | Validation operation latency |
| 11 | `gl_eudr_rng_errors_total` | Counter | Errors by operation type |
| 12 | `gl_eudr_rng_active_references` | Gauge | Currently active reference numbers |
| 13 | `gl_eudr_rng_sequence_utilization_percent` | Gauge | Sequence utilization per operator |
| 14 | `gl_eudr_rng_expired_references_total` | Counter | Reference numbers expired (auto-expiration) |
| 15 | `gl_eudr_rng_audit_events_total` | Counter | Audit trail events by action type |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL 14+ + TimescaleDB | Atomic sequences, hypertables, UNIQUE constraints |
| Sequence Allocation | PostgreSQL `SEQUENCE` objects | Database-level atomic increment with zero locks |
| Cache | Redis (optional) | Idempotency key cache, batch progress tracking |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based reference number access control |
| Monitoring | Prometheus + Grafana | 15 metrics + dedicated dashboard (48 panels) |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

The following permissions will be registered in the GreenLang PERMISSION_MAP for RBAC enforcement:

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-rng:generate` | Generate reference numbers | Analyst, Compliance Officer, Admin |
| `eudr-rng:batch` | Generate batch reference numbers | Analyst, Compliance Officer, Admin |
| `eudr-rng:validate` | Validate reference number format | Viewer, Analyst, Compliance Officer, Admin, Auditor |
| `eudr-rng:verify` | Verify reference number existence | Viewer, Analyst, Compliance Officer, Admin, Auditor |
| `eudr-rng:lifecycle:read` | View reference number lifecycle status | Viewer, Analyst, Compliance Officer, Admin, Auditor |
| `eudr-rng:lifecycle:write` | Update reference number lifecycle status | Compliance Officer, Admin |
| `eudr-rng:revoke` | Revoke reference numbers | Compliance Officer, Admin |
| `eudr-rng:transfer` | Transfer reference numbers between operators | Admin only |
| `eudr-rng:sequence:read` | View sequence counter status | Analyst, Compliance Officer, Admin |
| `eudr-rng:sequence:reset` | Reset sequence counters | Admin only |
| `eudr-rng:batch:read` | View batch request status | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-rng:batch:cancel` | Cancel batch requests | Compliance Officer, Admin |
| `eudr-rng:audit:read` | View audit trail | Auditor (read-only), Compliance Officer, Admin |
| `eudr-rng:audit:export` | Export audit trail (CSV/PDF) | Auditor, Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| AGENT-DATA-005 EUDR Traceability | OperatorRegistry, DDSRegistry | Operator validation, DDS metadata -> reference number generation |
| GL-EUDR-APP | DDS workflow | DDS creation triggers reference number generation |
| SEC-002 RBAC | Permission checks | Role-based access control for reference number operations |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| GL-EUDR-APP v1.0 | API integration | Generated reference numbers -> DDS forms, submission workflows |
| AGENT-DATA-005 EUSystemConnector | DDS submission | Reference numbers -> EU Information System DDS submission |
| External Auditors | Verification API | Reference number validation and audit trail export |
| Member State Competent Authorities | Verification API | Reference number authenticity checks |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Single Reference Number Generation (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Creates a new DDS for a cocoa shipment from Ghana
3. GL-EUDR-APP calls Reference Number Generator API:
   POST /v1/references/generate
   { "operator_id": "OPR00123", "member_state": "DE", "commodity": "cocoa" }
4. Reference Number Generator:
   a. Validates operator_id against OperatorRegistry
   b. Allocates next sequence number atomically (458)
   c. Formats reference: DE-2026-OPR00123-000458-L
   d. Computes Luhn-mod-36 checksum: L
   e. Logs generation event to audit trail
5. Returns reference number to GL-EUDR-APP
6. GL-EUDR-APP displays reference in DDS form
7. Compliance officer completes DDS and submits to EU Information System
8. GL-EUDR-APP calls lifecycle update:
   PUT /v1/references/{reference_id}/use
9. Reference Number Generator updates status to "used"
```

#### Flow 2: Batch Reference Number Generation (Supply Chain Analyst)

```
1. Analyst logs in to GL-EUDR-APP
2. Navigates to "Reference Number Management" module
3. Clicks "Generate Batch" -> selects:
   - Member State: Germany (DE)
   - Commodity: Coffee
   - Count: 500
4. GL-EUDR-APP calls Batch Generation API:
   POST /v1/references/batch
   { "operator_id": "OPR00123", "member_state": "DE", "count": 500, "commodity": "coffee" }
5. Reference Number Generator:
   a. Creates batch request (batch_id: abc-123)
   b. Status: pending -> in_progress
   c. Allocates 500 sequential numbers atomically (459-958)
   d. Formats 500 references: DE-2026-OPR00123-000459-M ... DE-2026-OPR00123-000958-X
   e. Logs batch generation event
   f. Status: completed
6. Analyst polls batch status:
   GET /v1/batches/abc-123
   -> Returns 500 generated reference numbers
7. Analyst exports references to CSV for offline DDS preparation
8. Over the next month, analyst uses pre-generated references for 500 shipments
```

#### Flow 3: Reference Number Verification (External Auditor)

```
1. Auditor receives operator's DDS documentation for compliance audit
2. Extracts reference number: DE-2026-OPR00123-000457-K
3. Auditor calls public Verification API (no authentication required):
   POST /v1/references/verify
   { "reference_number": "DE-2026-OPR00123-000457-K" }
4. Reference Number Generator:
   a. Validates format (DE-YYYY-OP-SEQ-CHK): PASS
   b. Validates checksum (Luhn-mod-36): PASS
   c. Queries database for existence: FOUND
   d. Checks lifecycle status: USED
   e. Checks expiration: NOT EXPIRED
5. Returns validation response:
   { "is_valid": true, "status": "used", "checks": [...] }
6. Auditor confirms reference number is authentic and currently in use
7. Auditor requests audit trail:
   GET /v1/audit/DE-2026-OPR00123-000457-K
8. Reference Number Generator returns:
   - Generated: 2026-01-15 10:30:00 by OPR00123
   - Activated: 2026-01-15 10:35:00 by OPR00123
   - Used: 2026-01-20 14:22:00 by OPR00123 (DDS submitted)
9. Auditor verifies complete audit trail and approves DDS
```

### 8.2 Key Screen Descriptions

**Reference Number Management Dashboard:**
- Summary cards: Total active, Total used, Total expired, Batch requests pending
- Recent references: List of last 100 generated references with status badges
- Sequence utilization: Progress bar showing current/max sequence value
- Quick actions: "Generate Reference", "Generate Batch", "View Audit Trail"

**Batch Generation Form:**
- Dropdown: Member State (27 EU states)
- Dropdown: Commodity (7 EUDR commodities)
- Input: Count (1-10,000)
- Button: "Generate Batch"
- Progress bar: Real-time batch processing status

**Verification Result Panel:**
- Reference number input field with "Verify" button
- Validation result cards: Format, Checksum, Member State, Existence, Lifecycle
- Status badge: VALID / INVALID / EXPIRED / REVOKED
- Audit trail timeline: Visual timeline of lifecycle events

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Unique Reference Number Generator -- atomic allocation, EU format
  - [ ] Feature 2: Format Compliance Validator -- 27 member states, all checksum algorithms
  - [ ] Feature 3: Sequential Number Manager -- PostgreSQL sequences, overflow handling
  - [ ] Feature 4: Batch Generation Engine -- 1-10,000 references, progress tracking
  - [ ] Feature 5: Collision Detector -- auto-detection, retry, logging
  - [ ] Feature 6: Member State Formatter -- 27 country profiles, hot-reloadable
  - [ ] Feature 7: Reference Lifecycle Manager -- 7 states, state machine validation
  - [ ] Feature 8: Validation and Verification Service -- stateless API, public endpoint
  - [ ] Feature 9: Audit Trail Recorder -- TimescaleDB hypertable, 10-year retention
- [ ] >= 85% test coverage achieved (791 tests already written)
- [ ] Security audit passed (JWT + RBAC integrated)
- [ ] Performance targets met:
  - [ ] < 10ms p99 for single generation
  - [ ] < 60 seconds for 10,000-reference batch
  - [ ] < 50ms p95 for verification API
- [ ] Database migration V126 tested and validated (1,935 lines already written)
- [ ] All 27 EU member states tested with golden test fixtures
- [ ] Zero collisions under 10,000 concurrent requests (load test)
- [ ] API documentation complete (OpenAPI spec)
- [ ] Integration with GL-EUDR-APP verified
- [ ] 5 beta customers successfully generated reference numbers
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 10,000+ reference numbers generated by customers
- Zero collision errors reported
- 100% EU Information System format validation pass rate
- < 5 support tickets per customer
- p99 generation latency < 10ms in production

**60 Days:**
- 50,000+ reference numbers generated
- 500+ batch requests processed
- Average batch size >= 100 references
- Zero DDS submission failures due to reference number errors
- Collision rate < 1 per 100,000 generations

**90 Days:**
- 200,000+ reference numbers generated
- 2,000+ batch requests processed
- All 27 EU member states represented in generation logs
- 99.9% uptime for verification API
- NPS > 50 from compliance officer persona

---

## 10. Timeline and Milestones

### Phase 1: Core Generation Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Number Generator (Feature 1): atomic allocation, EU format, checksum | Backend Engineer |
| 2-3 | Format Validator (Feature 2): 27 member states, all checksum algorithms | Backend Engineer |
| 3-4 | Sequence Manager (Feature 3): PostgreSQL sequences, overflow handling | Database Engineer |
| 4-5 | Batch Processor (Feature 4): batch generation, progress tracking | Backend Engineer |
| 5-6 | Collision Detector (Feature 5): detection, retry, logging | Backend Engineer |

**Milestone: Core generation engine operational with atomic sequences (Week 6)**

### Phase 2: Lifecycle and Member State Support (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Member State Formatter (Feature 6): 27 country profiles, hot-reloadable | Regulatory Analyst + Backend Engineer |
| 8-9 | Lifecycle Manager (Feature 7): state machine, 7 states, auto-expiration | Backend Engineer |
| 9-10 | Verification Service (Feature 8): stateless API, public endpoint | Backend Engineer |

**Milestone: Full lifecycle management with member-state formatting (Week 10)**

### Phase 3: Audit Trail and API (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | Audit Recorder (Feature 9): TimescaleDB hypertable, 10-year retention | Database Engineer |
| 12-13 | REST API Layer: 20+ endpoints, authentication, rate limiting | Backend Engineer |
| 13-14 | RBAC integration, GL-EUDR-APP integration, end-to-end testing | Backend Engineer + Frontend Engineer |

**Milestone: All 9 P0 features implemented with full API (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 791+ tests, golden tests for 27 member states | Test Engineer |
| 16-17 | Performance testing, security audit, load testing (10K concurrent) | DevOps + Security |
| 17 | Database migration V126 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers) | Product + Engineering |
| 18 | Launch readiness review (all 9 P0 features verified) and go-live | All |

**Milestone: Production launch with all 9 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Reference number transfer API (Feature 10)
- Analytics dashboard (Feature 11)
- Multi-tenant isolation (Feature 12)
- Performance optimization for 100K+ concurrent requests
- Additional member-state-specific formatting rules

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-DATA-005 EUDR Traceability Connector | BUILT (100%) | Low | Stable, production-ready |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Integration points defined |
| PostgreSQL 14+ + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| Database Migration V126 | Written (1,935 lines) | Low | Already validated |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU Information System format spec | Published (v1.2) | Medium | Adapter pattern for format version changes |
| Member state operator registries | Published; updated periodically | Medium | Database-driven, hot-reloadable operator prefixes |
| EU EUDR implementing regulations | Evolving | Medium | Configuration-driven compliance rules |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EU Information System format specification changes before or after launch | Medium | High | Adapter pattern isolates format layer; can update formatter without touching generation engine |
| R2 | PostgreSQL sequence overflow for high-volume operators | Low | High | Overflow strategy with automatic digit extension; monitoring alerts at 80% utilization |
| R3 | Member state competent authorities change operator prefix formats | Medium | Medium | Database-driven formatting rules; hot-reload from member state feed |
| R4 | Concurrent collision rate exceeds auto-retry capacity | Low | Medium | PostgreSQL atomic sequences guarantee zero collisions; auto-retry handles edge cases |
| R5 | EU Information System rejects reference numbers due to undocumented format rules | Medium | High | Beta testing with 5 customers; submit test DDS to EU portal before production launch |
| R6 | EUDR regulation amended with new reference number requirements | Low | Medium | Modular design allows quick adaptation; format version evolution built-in |
| R7 | Integration complexity with GL-EUDR-APP DDS workflows | Medium | Medium | Well-defined API interfaces; mock adapters for testing; circuit breaker pattern |
| R8 | Audit trail hypertable performance degrades under load | Low | Medium | TimescaleDB compression; partitioning; continuous query optimization |
| R9 | Low customer adoption of batch generation feature | Medium | Low | Documentation and training for high-volume operators; API examples |
| R10 | Competitive tools launch before GreenLang reaches market | Low | Low | Faster time-to-market via existing infrastructure; superior atomic sequence guarantee |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Number Generator Unit Tests | 150+ | Atomic allocation, format construction, checksum computation |
| Format Validator Tests | 100+ | 27 member states, all checksum algorithms, edge cases |
| Sequence Manager Tests | 80+ | Atomic increment, overflow handling, concurrency |
| Batch Processor Tests | 100+ | Batch sizes 1-10,000, progress tracking, partial success |
| Collision Detector Tests | 60+ | Collision detection, auto-retry, logging |
| Lifecycle Manager Tests | 80+ | 7 states, state transitions, auto-expiration |
| Member State Formatter Tests | 120+ | 27 country profiles, format rules, checksum algorithms |
| Verification Service Tests | 70+ | Format validation, checksum verification, existence checks |
| Audit Recorder Tests | 60+ | Event logging, provenance hashing, hash chain verification |
| API Tests | 100+ | All 20+ endpoints, auth, error handling, pagination |
| Golden Tests | 60+ | All 27 member states, complete/partial/broken scenarios |
| Integration Tests | 40+ | Cross-agent integration with DATA-005, GL-EUDR-APP |
| Performance Tests | 30+ | 10K concurrent requests, batch 10K references, latency |
| **Total** | **791+ (Already Written)** | |

### 13.2 Golden Test Member States

Each of the 27 EU member states will have a dedicated golden test reference number with:
1. Complete generation (format + checksum validation)
2. Format compliance (EU Information System validation)
3. Lifecycle state transitions (reserved → active → used)
4. Batch generation (100 references per member state)
5. Collision detection (duplicate sequence number)
6. Verification API (format + existence checks)
7. Audit trail completeness (all events logged)

Total: 27 member states × 7 scenarios = 189 golden test scenarios

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **Reference Number** | Unique identifier for a DDS per EUDR Article 4(5) and Article 33 |
| **Member State** | One of the 27 EU member states (e.g., Germany, France, Italy) |
| **Operator Prefix** | Alphanumeric code assigned to operator by member state competent authority |
| **Sequence Number** | Sequential number within operator/year scope (6 digits, zero-padded) |
| **Checksum** | Check digit(s) computed via Luhn-mod-36 or member-state-specific algorithm |
| **Collision** | Duplicate reference number detected during generation |
| **Lifecycle State** | Current status of reference number (reserved, active, used, expired, revoked, transferred, cancelled) |
| **Atomic Sequence** | Database-level sequential number allocation with zero race conditions |

### Appendix B: Reference Number Format Examples (27 Member States)

| Member State | Code | Example Reference Number | Format Notes |
|--------------|------|--------------------------|--------------|
| Austria | AT | AT-2026-AT123456-000123-7 | Luhn-mod-36 checksum |
| Belgium | BE | BE-2026-BE1234-000456-K | Luhn-mod-36 checksum |
| Bulgaria | BG | BG-2026-BG5678-000789-M | Luhn-mod-36 checksum |
| Croatia | HR | HR-2026-HR9012-001234-3 | Luhn-mod-36 checksum |
| Cyprus | CY | CY-2026-CY3456-000567-N | Luhn-mod-36 checksum |
| Czechia | CZ | CZ-2026-CZ7890-000890-P | Luhn-mod-36 checksum |
| Denmark | DK | DK-2026-DK1234-001111-Q | Luhn-mod-36 checksum |
| Estonia | EE | EE-2026-EE5678-002222-R | Luhn-mod-36 checksum |
| Finland | FI | FI-2026-FI9012-003333-S | Luhn-mod-36 checksum |
| France | FR | FR-2026-SA789-001234-97 | ISO7064 MOD 97-10 checksum |
| Germany | DE | DE-2026-OPR00123-000457-K | Luhn-mod-36 checksum |
| Greece | GR | GR-2026-GR3456-004444-T | Luhn-mod-36 checksum |
| Hungary | HU | HU-2026-HU7890-005555-U | Luhn-mod-36 checksum |
| Ireland | IE | IE-2026-IE1234-006666-V | Luhn-mod-36 checksum |
| Italy | IT | IT-2026-IT12345678-000123-F4 | CRC16 checksum |
| Latvia | LV | LV-2026-LV5678-007777-W | Luhn-mod-36 checksum |
| Lithuania | LT | LT-2026-LT9012-008888-X | Luhn-mod-36 checksum |
| Luxembourg | LU | LU-2026-LU3456-009999-Y | Luhn-mod-36 checksum |
| Malta | MT | MT-2026-MT7890-010000-Z | Luhn-mod-36 checksum |
| Netherlands | NL | NL-2026-NL9876-000789-3 | Luhn-mod-36 checksum |
| Poland | PL | PL-2026-PL1234-011111-A | Luhn-mod-36 checksum |
| Portugal | PT | PT-2026-PT5678-012222-B | Luhn-mod-36 checksum |
| Romania | RO | RO-2026-RO9012-013333-C | Luhn-mod-36 checksum |
| Slovakia | SK | SK-2026-SK3456-014444-D | Luhn-mod-36 checksum |
| Slovenia | SI | SI-2026-SI7890-015555-E | Luhn-mod-36 checksum |
| Spain | ES | ES-2026-ES123456-000234-45 | Modulo 97 checksum |
| Sweden | SE | SE-2026-SE1234-016666-F | Luhn-mod-36 checksum |

### Appendix C: Lifecycle State Machine

```
                   +-----------+
                   | RESERVED  |
                   +-----+-----+
                         |
                         v
                   +-----------+
            +----->|  ACTIVE   |<-----+
            |      +-----+-----+      |
            |            |            |
            |            v            |
            |      +-----------+      |
            |      |   USED    |      | (manual reactivation
            |      +-----+-----+      |  not allowed)
            |            |            |
            |            v            |
      +-----------+  +-----------+  +-----------+
      | EXPIRED   |  | REVOKED   |  |TRANSFERRED|
      +-----------+  +-----------+  +-----------+
            |            |            |
            v            v            v
      +-----------+  +-----------+  +-----------+
      | CANCELLED |  | CANCELLED |  | CANCELLED |
      +-----------+  +-----------+  +-----------+

Legend:
- RESERVED: Pre-allocated but not yet assigned to DDS
- ACTIVE: Available for DDS assignment
- USED: Assigned to DDS and submitted to EU Information System (terminal)
- EXPIRED: Auto-expired after 90 days without usage (terminal)
- REVOKED: Manually revoked due to fraud/error (terminal)
- TRANSFERRED: Ownership transferred to another operator (terminal)
- CANCELLED: Manually cancelled before usage (terminal)
```

### Appendix D: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023
2. EU Deforestation Regulation Guidance Document (European Commission)
3. EUDR Technical Specifications for the Information System (v1.2)
4. ISO/IEC 7064:2003 -- Information technology -- Security techniques -- Check character systems
5. Luhn algorithm (mod 10, mod 36 variants) -- Payment card validation
6. CRC-16 (Cyclic Redundancy Check) -- Error detection
7. Member State Competent Authority Contact List (European Commission)
8. PostgreSQL 14 Documentation -- SEQUENCE Objects

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-13 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________  |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-13 | GL-ProductManager | Initial draft created |
| 1.0.0 | 2026-03-13 | GL-ProductManager | Finalized: all 9 P0 features confirmed, 27 EU member states covered, regulatory coverage verified (Articles 4/9/12/31/33), database migration V126 referenced (1,935 lines, 8 tables), test suite confirmed (791 tests), dashboard confirmed (48 panels), approval granted |
