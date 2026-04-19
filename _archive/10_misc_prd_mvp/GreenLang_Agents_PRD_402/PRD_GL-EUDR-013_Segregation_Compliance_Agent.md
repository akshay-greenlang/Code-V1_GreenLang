# PRD: Segregation Compliance Agent (GL-EUDR-013)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Physical segregation, contamination prevention, chain integrity
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Segregation Compliance Agent (GL-EUDR-013)** ensures that EUDR-compliant commodities are physically segregated from non-compliant materials throughout the supply chain. It monitors and enforces segregation rules to prevent contamination.

---

## 2. Segregation Models

### 2.1 Identity Preserved (IP)
- Strictest model
- Complete physical separation
- Single origin traceability
- No mixing allowed

### 2.2 Segregated
- Physical separation maintained
- Compliant mixed only with compliant
- Origin grouping allowed
- No contamination with non-compliant

### 2.3 Mass Balance
- Accounting-based tracking
- Physical mixing allowed
- Volume-based compliance claims
- Requires robust accounting

---

## 3. Data Model

```sql
-- Segregation Rules
CREATE TABLE segregation_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(255) NOT NULL,
    commodity_category VARCHAR(50) NOT NULL,
    segregation_model VARCHAR(50) NOT NULL,
    description TEXT,

    -- Criteria
    allowed_mixing JSONB NOT NULL,  -- What can be mixed
    prohibited_mixing JSONB NOT NULL,  -- What cannot be mixed
    storage_requirements JSONB,
    transport_requirements JSONB,

    -- Scope
    applies_to_facilities UUID[],
    applies_to_commodities TEXT[],

    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_model CHECK (
        segregation_model IN ('IDENTITY_PRESERVED', 'SEGREGATED', 'MASS_BALANCE')
    )
);

-- Segregation Status
CREATE TABLE segregation_status (
    status_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    segregation_model VARCHAR(50) NOT NULL,
    segregation_status VARCHAR(50) NOT NULL,

    -- Classification
    compliance_class VARCHAR(50) NOT NULL,
    certification_type VARCHAR(100),
    origin_type VARCHAR(100),

    -- Integrity
    segregation_intact BOOLEAN DEFAULT TRUE,
    contamination_detected BOOLEAN DEFAULT FALSE,
    contamination_date TIMESTAMP,
    contamination_source VARCHAR(255),

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_status CHECK (
        segregation_status IN ('COMPLIANT', 'NON_COMPLIANT', 'UNKNOWN', 'CONTAMINATED')
    )
);

-- Contamination Events
CREATE TABLE contamination_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    facility_id UUID NOT NULL,
    event_date TIMESTAMP NOT NULL,

    -- Contamination details
    contamination_type VARCHAR(100) NOT NULL,
    contaminated_by_batch_id UUID,
    contamination_source VARCHAR(255),

    -- Impact
    quantity_affected DECIMAL(15,3) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL,

    -- Response
    detected_at TIMESTAMP,
    detected_by VARCHAR(100),
    response_action VARCHAR(255),
    quarantine_status VARCHAR(50),

    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolution_date TIMESTAMP,
    resolution_notes TEXT,

    created_at TIMESTAMP DEFAULT NOW()
);

-- Segregation Checks
CREATE TABLE segregation_checks (
    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id UUID NOT NULL,
    check_date TIMESTAMP NOT NULL,
    check_type VARCHAR(50) NOT NULL,

    -- Results
    batches_checked INTEGER,
    violations_found INTEGER,
    violation_details JSONB DEFAULT '[]',

    -- Status
    result VARCHAR(50) NOT NULL,
    checked_by VARCHAR(100),

    CONSTRAINT valid_result CHECK (
        result IN ('PASS', 'FAIL', 'WARNING')
    )
);

-- Indexes
CREATE INDEX idx_seg_status_batch ON segregation_status(batch_id);
CREATE INDEX idx_seg_status_class ON segregation_status(compliance_class);
CREATE INDEX idx_contamination_batch ON contamination_events(batch_id);
CREATE INDEX idx_contamination_facility ON contamination_events(facility_id);
CREATE INDEX idx_seg_checks_facility ON segregation_checks(facility_id);
```

---

## 4. Functional Requirements

### 4.1 Rule Management
- **FR-001 (P0):** Define segregation rules per commodity
- **FR-002 (P0):** Specify allowed/prohibited mixing
- **FR-003 (P0):** Configure by segregation model (IP, Segregated, MB)
- **FR-004 (P1):** Facility-specific rules

### 4.2 Enforcement
- **FR-010 (P0):** Validate operations against rules
- **FR-011 (P0):** Block prohibited mixing attempts
- **FR-012 (P0):** Alert on segregation violations
- **FR-013 (P0):** Track segregation status per batch

### 4.3 Contamination
- **FR-020 (P0):** Detect contamination events
- **FR-021 (P0):** Record contamination details
- **FR-022 (P0):** Initiate quarantine procedures
- **FR-023 (P0):** Track contamination resolution

### 4.4 Monitoring
- **FR-030 (P0):** Regular segregation audits
- **FR-031 (P0):** Segregation compliance reports
- **FR-032 (P1):** Facility segregation scores

---

## 5. Segregation Validation Engine

```python
class SegregationValidator:
    """
    Validates segregation rules are maintained.
    """

    def validate_mixing(
        self,
        batch_ids: List[UUID],
        facility_id: UUID
    ) -> ValidationResult:
        """
        Validate if batches can be mixed.
        """
        batches = [self.get_batch(bid) for bid in batch_ids]
        facility = self.get_facility(facility_id)
        rules = self.get_rules(facility.commodity_category)

        issues = []

        # Get compliance classes
        classes = set(b.segregation_status.compliance_class for b in batches)

        # Check if mixing allowed
        for rule in rules:
            if rule.segregation_model == "IDENTITY_PRESERVED":
                if len(set(b.origin_plots[0] for b in batches)) > 1:
                    issues.append(Issue(
                        type="IP_VIOLATION",
                        message="Identity preserved: cannot mix different origins"
                    ))

            elif rule.segregation_model == "SEGREGATED":
                if "COMPLIANT" in classes and "NON_COMPLIANT" in classes:
                    issues.append(Issue(
                        type="SEGREGATION_VIOLATION",
                        message="Cannot mix compliant with non-compliant"
                    ))

        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues
        )

    def check_contamination(
        self,
        batch_id: UUID,
        operation: str
    ) -> Optional[ContaminationEvent]:
        """
        Check if an operation would cause contamination.
        """
        batch = self.get_batch(batch_id)

        if batch.segregation_status.contamination_detected:
            return ContaminationEvent(
                batch_id=batch_id,
                contamination_type="ALREADY_CONTAMINATED",
                message="Batch is already marked as contaminated"
            )

        # Check storage location
        if operation == "STORAGE":
            location = self.get_storage_location(batch_id)
            other_batches = self.get_batches_in_location(location)

            for other in other_batches:
                if other.segregation_status.compliance_class != batch.segregation_status.compliance_class:
                    return ContaminationEvent(
                        batch_id=batch_id,
                        contamination_type="STORAGE_CONTAMINATION",
                        contaminated_by_batch_id=other.batch_id,
                        message=f"Storage location contains {other.segregation_status.compliance_class} material"
                    )

        return None
```

---

## 6. API Specification

```yaml
paths:
  /api/v1/segregation/rules:
    post:
      summary: Create segregation rule
    get:
      summary: List rules

  /api/v1/segregation/validate:
    post:
      summary: Validate mixing operation
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                batch_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                facility_id:
                  type: string
                  format: uuid

  /api/v1/segregation/batches/{batch_id}/status:
    get:
      summary: Get segregation status
    patch:
      summary: Update segregation status

  /api/v1/segregation/contamination:
    post:
      summary: Report contamination

  /api/v1/segregation/checks:
    post:
      summary: Record segregation check
    get:
      summary: Get check history
```

---

## 7. Success Metrics

- **Segregation Integrity:** 100% compliant batches uncontaminated
- **Violation Detection:** 100% violations detected before mixing
- **Contamination Rate:** <0.1% of batches contaminated
- **Check Coverage:** 100% of facilities checked monthly

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
