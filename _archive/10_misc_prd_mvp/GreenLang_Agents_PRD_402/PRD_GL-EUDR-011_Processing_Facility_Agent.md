# PRD: Processing Facility Agent (GL-EUDR-011)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Processing facilities, transformation tracking, manufacturing
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Processing Facility Agent (GL-EUDR-011)** manages the registration and tracking of processing facilities where commodities are transformed (mills, refineries, factories). It ensures all facilities handling EUDR commodities are documented and their operations are traceable.

---

## 2. Goals

1. **Facility registration**
   - Unique facility identifiers
   - Location and capacity
   - Certification status
   - Commodity types handled

2. **Operation tracking**
   - Input/output volumes
   - Processing records
   - Yield tracking

3. **Compliance monitoring**
   - Facility certifications
   - Operating licenses
   - Environmental permits

4. **Integration**
   - Link to supply chain
   - Link to transformations
   - Mass balance reconciliation

---

## 3. Data Model

```sql
-- Processing Facilities
CREATE TABLE processing_facilities (
    facility_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_code VARCHAR(50) UNIQUE NOT NULL,
    facility_name VARCHAR(500) NOT NULL,

    -- Type
    facility_type VARCHAR(100) NOT NULL,  -- MILL, REFINERY, FACTORY, etc.
    primary_commodity VARCHAR(50) NOT NULL,
    commodities_handled TEXT[],

    -- Location
    address JSONB NOT NULL,
    country_code CHAR(2) NOT NULL,
    coordinates GEOGRAPHY(POINT, 4326),

    -- Capacity
    processing_capacity DECIMAL(15,3),
    capacity_unit VARCHAR(20),
    capacity_period VARCHAR(20),  -- DAILY, MONTHLY, ANNUAL

    -- Certifications
    certifications JSONB DEFAULT '[]',

    -- Operating Details
    operating_license VARCHAR(100),
    license_expiry DATE,
    environmental_permit VARCHAR(100),

    -- Ownership
    operator_id UUID NOT NULL,
    ownership_type VARCHAR(50),

    -- Status
    status VARCHAR(50) DEFAULT 'ACTIVE',
    verification_status VARCHAR(50) DEFAULT 'UNVERIFIED',
    last_audit_date DATE,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Processing Records
CREATE TABLE processing_records (
    record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id UUID REFERENCES processing_facilities(facility_id),
    processing_date DATE NOT NULL,

    -- Inputs
    input_batches JSONB NOT NULL,  -- [{batch_id, quantity, unit}]
    input_commodity VARCHAR(100) NOT NULL,
    total_input_quantity DECIMAL(15,3) NOT NULL,

    -- Outputs
    output_batches JSONB NOT NULL,  -- [{batch_id, quantity, unit}]
    output_commodity VARCHAR(100) NOT NULL,
    total_output_quantity DECIMAL(15,3) NOT NULL,

    -- Yield
    yield_ratio DECIMAL(5,4) NOT NULL,
    expected_yield DECIMAL(5,4),
    yield_variance DECIMAL(5,4),

    -- Process
    process_type VARCHAR(100) NOT NULL,
    process_line VARCHAR(100),

    -- Compliance
    origin_plots_verified UUID[],
    compliance_status VARCHAR(50) DEFAULT 'PENDING',

    created_at TIMESTAMP DEFAULT NOW()
);

-- Facility Audits
CREATE TABLE facility_audits (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id UUID REFERENCES processing_facilities(facility_id),
    audit_date DATE NOT NULL,
    audit_type VARCHAR(50) NOT NULL,
    auditor VARCHAR(255),
    audit_body VARCHAR(255),
    result VARCHAR(50) NOT NULL,
    findings JSONB DEFAULT '[]',
    corrective_actions JSONB DEFAULT '[]',
    next_audit_due DATE,

    CONSTRAINT valid_result CHECK (
        result IN ('PASS', 'CONDITIONAL_PASS', 'FAIL', 'PENDING')
    )
);

-- Indexes
CREATE INDEX idx_facilities_type ON processing_facilities(facility_type);
CREATE INDEX idx_facilities_commodity ON processing_facilities(primary_commodity);
CREATE INDEX idx_facilities_country ON processing_facilities(country_code);
CREATE INDEX idx_facilities_operator ON processing_facilities(operator_id);
CREATE INDEX idx_processing_facility ON processing_records(facility_id);
CREATE INDEX idx_processing_date ON processing_records(processing_date);
```

---

## 4. Functional Requirements

### 4.1 Registration
- **FR-001 (P0):** Register processing facilities
- **FR-002 (P0):** Capture facility location and capacity
- **FR-003 (P0):** Track certifications
- **FR-004 (P0):** Link to operators/suppliers

### 4.2 Processing
- **FR-010 (P0):** Record processing events
- **FR-011 (P0):** Track input/output batches
- **FR-012 (P0):** Calculate and validate yields
- **FR-013 (P0):** Propagate origin data through processing

### 4.3 Compliance
- **FR-020 (P0):** Track facility audits
- **FR-021 (P0):** Monitor certification validity
- **FR-022 (P1):** Alert on compliance issues

---

## 5. Facility Types by Commodity

| Commodity | Facility Types |
|---|---|
| PALM_OIL | Palm Oil Mill, Refinery, Fractionation Plant |
| COCOA | Fermentation, Drying, Processing Plant |
| COFFEE | Wet Mill, Dry Mill, Roastery |
| RUBBER | Processing Factory, Tire Factory |
| SOY | Crushing Plant, Refinery, Feed Mill |
| WOOD | Sawmill, Plywood Factory, Pulp Mill |
| CATTLE | Slaughterhouse, Tannery, Meat Processing |

---

## 6. API Specification

```yaml
paths:
  /api/v1/facilities:
    post:
      summary: Register facility
    get:
      summary: List facilities

  /api/v1/facilities/{facility_id}:
    get:
      summary: Get facility details
    patch:
      summary: Update facility

  /api/v1/facilities/{facility_id}/processing:
    post:
      summary: Record processing event
    get:
      summary: Get processing history

  /api/v1/facilities/{facility_id}/audits:
    post:
      summary: Record audit
    get:
      summary: Get audit history
```

---

## 7. Success Metrics

- **Registration:** 100% of facilities registered
- **Processing Coverage:** 100% of processing events recorded
- **Yield Tracking:** <5% unexplained variance
- **Certification Status:** 100% current certifications tracked

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
