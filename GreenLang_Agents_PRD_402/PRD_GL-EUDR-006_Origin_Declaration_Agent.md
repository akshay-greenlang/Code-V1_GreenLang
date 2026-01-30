# PRD: Origin Declaration Agent (GL-EUDR-006)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Origin declarations, supplier statements, provenance claims
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Origin Declaration Agent (GL-EUDR-006)** manages the collection, validation, and storage of origin declarations from suppliers. These declarations are formal statements from suppliers about the geographic origin of their commodities, forming the basis for traceability claims in Due Diligence Statements.

Critical because:
- EUDR requires documented origin claims from all suppliers
- Declarations must include specific plot geolocation data
- Origin claims must be verifiable and auditable
- False declarations carry legal liability

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. **Declaration collection**
   - Structured declaration forms
   - Multi-language support
   - Digital signature support
   - Document attachments

2. **Validation**
   - Completeness checking
   - Cross-reference with known plots
   - Consistency with historical declarations
   - Conflict detection

3. **Management**
   - Version control
   - Expiry tracking
   - Renewal workflows
   - Audit trail

4. **Verification support**
   - Link to verification workflows
   - Evidence aggregation
   - Discrepancy flagging

### 2.2 Non-Goals

- Plot validation (GL-EUDR-002)
- Supplier verification (GL-EUDR-004)
- Risk assessment (GL-EUDR-020+)

---

## 3. Data Model

```sql
-- Origin Declarations
CREATE TABLE origin_declarations (
    declaration_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    declaration_number VARCHAR(100) UNIQUE NOT NULL,

    -- Declarant
    supplier_id UUID NOT NULL,
    declarant_name VARCHAR(500) NOT NULL,
    declarant_title VARCHAR(200),
    declarant_email VARCHAR(255),

    -- Declaration scope
    commodity_category VARCHAR(50) NOT NULL,
    commodity_types TEXT[],
    shipment_reference VARCHAR(100),
    quantity DECIMAL(15,3),
    quantity_unit VARCHAR(20),
    period_start DATE,
    period_end DATE,

    -- Origin claims
    origin_plots UUID[] NOT NULL,  -- Array of plot IDs
    origin_countries TEXT[] NOT NULL,
    origin_regions JSONB DEFAULT '[]',

    -- Statement
    declaration_text TEXT NOT NULL,
    declaration_date TIMESTAMP NOT NULL,
    signature_type VARCHAR(50),  -- DIGITAL, MANUAL, DELEGATED
    signature_data JSONB,
    witnessed_by VARCHAR(255),

    -- Attachments
    supporting_documents JSONB DEFAULT '[]',

    -- Status
    status VARCHAR(50) DEFAULT 'SUBMITTED',
    validation_status VARCHAR(50) DEFAULT 'PENDING',
    validation_issues JSONB DEFAULT '[]',

    -- Verification
    verified_at TIMESTAMP,
    verified_by VARCHAR(100),
    verification_notes TEXT,

    -- Validity
    valid_from DATE,
    valid_until DATE,
    supersedes_declaration_id UUID,

    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_status CHECK (
        status IN ('DRAFT', 'SUBMITTED', 'VALIDATED', 'REJECTED', 'SUPERSEDED', 'EXPIRED')
    )
);

-- Declaration Templates
CREATE TABLE declaration_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(255) NOT NULL,
    commodity_category VARCHAR(50) NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    template_version VARCHAR(20) DEFAULT '1.0',
    template_text TEXT NOT NULL,
    required_fields JSONB NOT NULL,
    optional_fields JSONB DEFAULT '[]',
    legal_clauses TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Declaration Validation Results
CREATE TABLE declaration_validations (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    declaration_id UUID REFERENCES origin_declarations(declaration_id),
    validation_date TIMESTAMP DEFAULT NOW(),
    overall_result VARCHAR(50) NOT NULL,
    checks_performed JSONB NOT NULL,
    issues_found JSONB DEFAULT '[]',
    validated_by VARCHAR(100),
    validation_method VARCHAR(50)
);

-- Indexes
CREATE INDEX idx_declarations_supplier ON origin_declarations(supplier_id);
CREATE INDEX idx_declarations_status ON origin_declarations(status);
CREATE INDEX idx_declarations_commodity ON origin_declarations(commodity_category);
CREATE INDEX idx_declarations_plots ON origin_declarations USING GIN(origin_plots);
CREATE INDEX idx_declarations_validity ON origin_declarations(valid_from, valid_until);
```

---

## 4. Functional Requirements

### 4.1 Collection
- **FR-001 (P0):** Provide declaration form templates per commodity
- **FR-002 (P0):** Collect plot geolocation data in declaration
- **FR-003 (P0):** Support digital signatures
- **FR-004 (P0):** Attach supporting documents
- **FR-005 (P1):** Multi-language form support

### 4.2 Validation
- **FR-010 (P0):** Validate all required fields present
- **FR-011 (P0):** Cross-reference plots with registry
- **FR-012 (P0):** Check for conflicting declarations
- **FR-013 (P0):** Verify declarant authority
- **FR-014 (P1):** Detect quantity inconsistencies

### 4.3 Management
- **FR-020 (P0):** Track declaration lifecycle
- **FR-021 (P0):** Monitor expiry dates
- **FR-022 (P0):** Support declaration renewal
- **FR-023 (P0):** Maintain version history
- **FR-024 (P1):** Send renewal reminders

### 4.4 Reporting
- **FR-030 (P0):** Export for DDS submission
- **FR-031 (P0):** Generate declaration summaries
- **FR-032 (P1):** Coverage analysis by supplier

---

## 5. Declaration Template

```markdown
## ORIGIN DECLARATION

### EUDR Compliance Origin Statement

**Declaration Number:** {declaration_number}
**Date:** {declaration_date}

I, {declarant_name}, acting in my capacity as {declarant_title} of {company_name},
hereby declare and confirm that:

1. The {commodity_type} products described below originate from the following
   verified production plots:

   | Plot ID | Location (Country/Region) | Area (ha) | Coordinates |
   |---------|---------------------------|-----------|-------------|
   {plot_table}

2. To the best of my knowledge and belief, based on our due diligence procedures:

   a) These production plots have NOT been subject to deforestation after
      December 31, 2020;

   b) The products were produced in accordance with all applicable laws and
      regulations of the country of production;

   c) The geolocation data provided is accurate to at least 6 decimal places
      in the WGS-84 coordinate system.

3. This declaration covers the following products:

   **Commodity Category:** {commodity_category}
   **Product Type(s):** {commodity_types}
   **Quantity:** {quantity} {quantity_unit}
   **Shipment Reference:** {shipment_reference}

4. I understand that false statements in this declaration may result in legal
   liability under EU Regulation 2023/1115 and applicable national laws.

5. I agree to provide additional documentation or evidence upon request to
   support the claims made in this declaration.

**Signature:** ________________________
**Name:** {declarant_name}
**Title:** {declarant_title}
**Date:** {signature_date}
```

---

## 6. API Specification

```yaml
paths:
  /api/v1/declarations:
    post:
      summary: Submit origin declaration
    get:
      summary: List declarations

  /api/v1/declarations/{declaration_id}:
    get:
      summary: Get declaration details
    patch:
      summary: Update draft declaration

  /api/v1/declarations/{declaration_id}/validate:
    post:
      summary: Validate declaration

  /api/v1/declarations/{declaration_id}/sign:
    post:
      summary: Add signature to declaration

  /api/v1/declarations/templates:
    get:
      summary: Get declaration templates

  /api/v1/declarations/expiring:
    get:
      summary: Get declarations expiring soon
```

---

## 7. Success Metrics

- **Collection Rate:** 100% of shipments have declarations
- **Validation Pass Rate:** >90% first-time validation
- **Signature Rate:** 100% declarations digitally signed
- **Renewal Rate:** 100% renewed before expiry
- **Conflict Rate:** <5% declarations with conflicts

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
