# PRD: Batch Tracking Agent (GL-EUDR-008)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Batch identification, batch lifecycle, lot tracking
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Batch Tracking Agent (GL-EUDR-008)** manages the unique identification and lifecycle tracking of commodity batches throughout the supply chain. It ensures every batch can be traced from origin to market with consistent identifiers.

---

## 2. Goals

1. **Batch identification**
   - Unique batch ID generation
   - Barcode/QR code support
   - Cross-reference with supplier IDs

2. **Lifecycle tracking**
   - Creation, movement, transformation
   - Splitting and merging
   - Consumption and disposal

3. **Relationship management**
   - Parent-child batch relationships
   - Origin plot linkage
   - Document association

4. **Inventory visibility**
   - Real-time batch locations
   - Quantity tracking
   - Age/freshness monitoring

---

## 3. Data Model

```sql
-- Batch Master
CREATE TABLE batches (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_number VARCHAR(100) UNIQUE NOT NULL,
    barcode VARCHAR(50),
    qr_code_data TEXT,

    -- Classification
    commodity_category VARCHAR(50) NOT NULL,
    commodity_type VARCHAR(100) NOT NULL,
    grade VARCHAR(50),
    quality_class VARCHAR(50),

    -- Quantity
    initial_quantity DECIMAL(15,3) NOT NULL,
    current_quantity DECIMAL(15,3) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL,

    -- Origins
    origin_plot_ids UUID[],
    origin_countries TEXT[],
    parent_batch_ids UUID[],
    child_batch_ids UUID[],

    -- Location
    current_facility_id UUID,
    current_location_type VARCHAR(50),

    -- Dates
    production_date DATE,
    harvest_date DATE,
    received_date TIMESTAMP,
    expiry_date DATE,

    -- Status
    status VARCHAR(50) DEFAULT 'ACTIVE',
    compliance_status VARCHAR(50) DEFAULT 'PENDING',

    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Batch Movements
CREATE TABLE batch_movements (
    movement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID REFERENCES batches(batch_id),
    movement_type VARCHAR(50) NOT NULL,
    movement_date TIMESTAMP NOT NULL,

    from_facility_id UUID,
    to_facility_id UUID,

    quantity DECIMAL(15,3) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL,

    document_reference VARCHAR(100),
    notes TEXT,

    CONSTRAINT valid_movement_type CHECK (
        movement_type IN (
            'RECEIPT', 'DISPATCH', 'TRANSFER', 'SPLIT',
            'MERGE', 'PROCESS', 'CONSUME', 'DISPOSE', 'ADJUST'
        )
    )
);

-- Batch Splits
CREATE TABLE batch_splits (
    split_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_batch_id UUID REFERENCES batches(batch_id),
    child_batch_ids UUID[] NOT NULL,
    split_date TIMESTAMP NOT NULL,
    split_reason VARCHAR(255),
    split_quantities JSONB NOT NULL
);

-- Batch Merges
CREATE TABLE batch_merges (
    merge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_batch_ids UUID[] NOT NULL,
    child_batch_id UUID REFERENCES batches(batch_id),
    merge_date TIMESTAMP NOT NULL,
    merge_reason VARCHAR(255),
    source_quantities JSONB NOT NULL
);

-- Indexes
CREATE INDEX idx_batches_number ON batches(batch_number);
CREATE INDEX idx_batches_commodity ON batches(commodity_category);
CREATE INDEX idx_batches_status ON batches(status);
CREATE INDEX idx_batches_facility ON batches(current_facility_id);
CREATE INDEX idx_movements_batch ON batch_movements(batch_id);
CREATE INDEX idx_movements_date ON batch_movements(movement_date);
```

---

## 4. Functional Requirements

### 4.1 Batch Creation
- **FR-001 (P0):** Generate unique batch identifiers
- **FR-002 (P0):** Link batches to origin plots
- **FR-003 (P0):** Support barcode/QR code generation
- **FR-004 (P0):** Set initial quantity and classification

### 4.2 Batch Operations
- **FR-010 (P0):** Record batch movements
- **FR-011 (P0):** Handle batch splitting
- **FR-012 (P0):** Handle batch merging
- **FR-013 (P0):** Track quantity changes
- **FR-014 (P1):** Support batch adjustments

### 4.3 Queries
- **FR-020 (P0):** Query by batch number
- **FR-021 (P0):** Query by origin
- **FR-022 (P0):** Query by location
- **FR-023 (P0):** Trace batch ancestry

---

## 5. Batch Number Format

```python
def generate_batch_number(
    facility_code: str,
    commodity: str,
    date: date,
    sequence: int
) -> str:
    """
    Generate unique batch number.
    Format: {FACILITY}-{COMMODITY}-{YYMMDD}-{SEQUENCE}
    Example: MYS-PLM-250130-00123
    """
    commodity_codes = {
        "PALM_OIL": "PLM",
        "COCOA": "COC",
        "COFFEE": "COF",
        "RUBBER": "RUB",
        "SOY": "SOY",
        "WOOD": "WOD",
        "CATTLE": "CAT"
    }

    date_str = date.strftime("%y%m%d")
    comm_code = commodity_codes.get(commodity, "XXX")

    return f"{facility_code}-{comm_code}-{date_str}-{sequence:05d}"
```

---

## 6. API Specification

```yaml
paths:
  /api/v1/batches:
    post:
      summary: Create batch
    get:
      summary: List batches

  /api/v1/batches/{batch_id}:
    get:
      summary: Get batch details
    patch:
      summary: Update batch

  /api/v1/batches/{batch_id}/movements:
    post:
      summary: Record movement
    get:
      summary: Get movement history

  /api/v1/batches/{batch_id}/split:
    post:
      summary: Split batch

  /api/v1/batches/merge:
    post:
      summary: Merge batches

  /api/v1/batches/{batch_id}/trace:
    get:
      summary: Trace batch origins
```

---

## 7. Success Metrics

- **Unique IDs:** 100% batches with unique identifiers
- **Origin Linkage:** 100% batches linked to plots
- **Movement Coverage:** 100% movements recorded
- **Quantity Accuracy:** <1% variance in quantity tracking

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
