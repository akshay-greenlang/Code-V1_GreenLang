# PRD: Transportation Tracking Agent (GL-EUDR-009)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Transportation tracking, shipment monitoring, logistics visibility
**Priority:** P1 (high)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Transportation Tracking Agent (GL-EUDR-009)** tracks the movement of commodities during transportation between supply chain nodes. It monitors shipments from origin countries to EU ports, ensuring continuous visibility and linking transport documents to traceability chains.

---

## 2. Goals

1. **Shipment tracking**
   - Container/vessel tracking
   - Truck/rail tracking
   - Multi-modal transport

2. **Document management**
   - Bill of lading linkage
   - Transport certificates
   - Customs documentation

3. **Route monitoring**
   - Origin-destination tracking
   - Intermediate stops
   - Deviation detection

4. **Timeline tracking**
   - Departure/arrival times
   - Transit duration
   - Delay detection

---

## 3. Data Model

```sql
-- Shipments
CREATE TABLE shipments (
    shipment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    shipment_number VARCHAR(100) UNIQUE NOT NULL,

    -- Cargo
    batch_ids UUID[] NOT NULL,
    commodity_category VARCHAR(50) NOT NULL,
    total_quantity DECIMAL(15,3) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL,

    -- Transport mode
    transport_mode VARCHAR(50) NOT NULL,  -- SEA, ROAD, RAIL, AIR
    container_number VARCHAR(50),
    vessel_name VARCHAR(255),
    voyage_number VARCHAR(100),
    vehicle_id VARCHAR(100),

    -- Route
    origin_port VARCHAR(100),
    origin_country CHAR(2),
    destination_port VARCHAR(100),
    destination_country CHAR(2),
    route_waypoints JSONB DEFAULT '[]',

    -- Timeline
    departure_date TIMESTAMP,
    estimated_arrival TIMESTAMP,
    actual_arrival TIMESTAMP,

    -- Status
    status VARCHAR(50) DEFAULT 'BOOKED',
    current_location GEOGRAPHY(POINT, 4326),
    last_update TIMESTAMP,

    -- Documents
    bill_of_lading VARCHAR(100),
    documents JSONB DEFAULT '[]',

    created_at TIMESTAMP DEFAULT NOW()
);

-- Shipment Events
CREATE TABLE shipment_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    shipment_id UUID REFERENCES shipments(shipment_id),
    event_type VARCHAR(50) NOT NULL,
    event_date TIMESTAMP NOT NULL,
    location GEOGRAPHY(POINT, 4326),
    location_name VARCHAR(255),
    description TEXT,

    CONSTRAINT valid_event_type CHECK (
        event_type IN (
            'BOOKED', 'LOADED', 'DEPARTED', 'IN_TRANSIT',
            'PORT_CALL', 'CUSTOMS_CLEARED', 'ARRIVED', 'DELIVERED'
        )
    )
);

-- Indexes
CREATE INDEX idx_shipments_number ON shipments(shipment_number);
CREATE INDEX idx_shipments_status ON shipments(status);
CREATE INDEX idx_shipments_batches ON shipments USING GIN(batch_ids);
CREATE INDEX idx_shipment_events_shipment ON shipment_events(shipment_id);
```

---

## 4. Functional Requirements

### 4.1 Shipment Management
- **FR-001 (P0):** Create shipments linked to batches
- **FR-002 (P0):** Track transport mode and carrier
- **FR-003 (P0):** Link bill of lading
- **FR-004 (P1):** Container tracking integration

### 4.2 Tracking
- **FR-010 (P0):** Record shipment events
- **FR-011 (P0):** Update current location
- **FR-012 (P1):** Track estimated arrival
- **FR-013 (P1):** Detect route deviations

### 4.3 Documents
- **FR-020 (P0):** Attach transport documents
- **FR-021 (P0):** Link to chain of custody
- **FR-022 (P1):** Validate document completeness

---

## 5. API Specification

```yaml
paths:
  /api/v1/shipments:
    post:
      summary: Create shipment
    get:
      summary: List shipments

  /api/v1/shipments/{shipment_id}:
    get:
      summary: Get shipment details

  /api/v1/shipments/{shipment_id}/events:
    post:
      summary: Add shipment event
    get:
      summary: Get shipment events

  /api/v1/shipments/{shipment_id}/track:
    get:
      summary: Get current location

  /api/v1/shipments/container/{container_number}:
    get:
      summary: Track by container number
```

---

## 6. Success Metrics

- **Tracking Coverage:** 100% of shipments tracked
- **Document Linkage:** 100% with bill of lading
- **Location Updates:** <24 hour update frequency
- **Arrival Accuracy:** <48 hour ETA variance

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
