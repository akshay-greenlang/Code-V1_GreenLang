# PRD: Warehouse Tracking Agent (GL-EUDR-010)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Warehouse tracking, storage monitoring, inventory location
**Priority:** P1 (high)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Warehouse Tracking Agent (GL-EUDR-010)** tracks commodities while in storage at warehouses, ports, and distribution centers. It maintains visibility of batch locations during storage phases of the supply chain.

---

## 2. Goals

1. **Storage tracking**
   - Warehouse entry/exit
   - Storage location (bin/zone)
   - Duration monitoring

2. **Inventory management**
   - Quantity in storage
   - Stock movements
   - FIFO/LIFO tracking

3. **Condition monitoring**
   - Storage conditions
   - Quality preservation
   - Expiry tracking

4. **Segregation**
   - Compliant/non-compliant separation
   - Certification-based segregation
   - Contamination prevention

---

## 3. Data Model

```sql
-- Warehouse Locations
CREATE TABLE warehouse_locations (
    location_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id UUID NOT NULL,
    location_code VARCHAR(50) NOT NULL,
    location_type VARCHAR(50) NOT NULL,  -- ZONE, AISLE, BIN
    parent_location_id UUID,
    capacity DECIMAL(15,3),
    capacity_unit VARCHAR(20),
    segregation_type VARCHAR(50),  -- EUDR_COMPLIANT, CERTIFIED, STANDARD
    is_active BOOLEAN DEFAULT TRUE,

    UNIQUE(facility_id, location_code)
);

-- Storage Records
CREATE TABLE storage_records (
    storage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    facility_id UUID NOT NULL,
    location_id UUID REFERENCES warehouse_locations(location_id),

    -- Quantity
    quantity DECIMAL(15,3) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL,

    -- Timeline
    received_at TIMESTAMP NOT NULL,
    dispatched_at TIMESTAMP,
    storage_duration_days INTEGER GENERATED ALWAYS AS (
        EXTRACT(DAY FROM COALESCE(dispatched_at, NOW()) - received_at)
    ) STORED,

    -- Status
    status VARCHAR(50) DEFAULT 'IN_STORAGE',
    condition VARCHAR(50) DEFAULT 'GOOD',

    created_at TIMESTAMP DEFAULT NOW()
);

-- Storage Movements
CREATE TABLE storage_movements (
    movement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    facility_id UUID NOT NULL,
    movement_type VARCHAR(50) NOT NULL,
    movement_date TIMESTAMP NOT NULL,

    from_location_id UUID,
    to_location_id UUID,

    quantity DECIMAL(15,3) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL,

    document_reference VARCHAR(100),

    CONSTRAINT valid_movement_type CHECK (
        movement_type IN ('RECEIPT', 'DISPATCH', 'TRANSFER', 'CYCLE_COUNT', 'ADJUSTMENT')
    )
);

-- Indexes
CREATE INDEX idx_storage_batch ON storage_records(batch_id);
CREATE INDEX idx_storage_facility ON storage_records(facility_id);
CREATE INDEX idx_storage_status ON storage_records(status);
CREATE INDEX idx_movements_batch ON storage_movements(batch_id);
```

---

## 4. Functional Requirements

### 4.1 Receipt/Dispatch
- **FR-001 (P0):** Record warehouse receipts
- **FR-002 (P0):** Record warehouse dispatches
- **FR-003 (P0):** Track storage location
- **FR-004 (P0):** Update chain of custody

### 4.2 Inventory
- **FR-010 (P0):** Track current stock by batch
- **FR-011 (P0):** Support internal transfers
- **FR-012 (P1):** FIFO tracking by batch
- **FR-013 (P1):** Cycle count reconciliation

### 4.3 Segregation
- **FR-020 (P0):** Enforce segregation rules
- **FR-021 (P0):** Track compliant vs non-compliant
- **FR-022 (P1):** Alert on contamination risk

---

## 5. API Specification

```yaml
paths:
  /api/v1/warehouses/{facility_id}/inventory:
    get:
      summary: Get current inventory

  /api/v1/warehouses/{facility_id}/receipt:
    post:
      summary: Record receipt

  /api/v1/warehouses/{facility_id}/dispatch:
    post:
      summary: Record dispatch

  /api/v1/storage/{batch_id}/history:
    get:
      summary: Get storage history for batch
```

---

## 6. Success Metrics

- **Visibility:** 100% batches with storage location
- **Accuracy:** <1% inventory variance
- **Segregation:** 100% compliance with rules
- **Duration Tracking:** 100% with timestamps

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
