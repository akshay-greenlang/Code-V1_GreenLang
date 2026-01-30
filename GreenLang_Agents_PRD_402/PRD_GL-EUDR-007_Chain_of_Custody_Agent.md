# PRD: Chain of Custody Agent (GL-EUDR-007)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Chain of custody, custody transfers, provenance tracking
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Chain of Custody Agent (GL-EUDR-007)** tracks the continuous physical control and transfer of commodities as they move through the supply chain. It ensures an unbroken chain of documented custody from origin plot to EU market entry, providing the foundation for EUDR compliance claims.

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. **Custody event tracking**
   - Receipt and dispatch events
   - Custody transfers between entities
   - Storage and processing events

2. **Chain integrity**
   - Continuous custody chain verification
   - Gap detection and alerting
   - Reconciliation of transfers

3. **Document linkage**
   - Link custody events to documents
   - Bill of lading, receipts, invoices
   - Transfer certificates

4. **Audit support**
   - Complete audit trail
   - Chain visualization
   - Evidence package generation

### 2.2 Non-Goals

- Logistics optimization
- Transportation management
- Customs clearance processing

---

## 3. Data Model

```sql
-- Custody Chains
CREATE TABLE custody_chains (
    chain_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    commodity_category VARCHAR(50) NOT NULL,

    -- Origin
    origin_plot_id UUID,
    origin_event_id UUID,
    origin_date TIMESTAMP,

    -- Destination
    destination_entity_id UUID,
    destination_event_id UUID,
    arrival_date TIMESTAMP,

    -- Chain status
    status VARCHAR(50) DEFAULT 'IN_PROGRESS',
    is_complete BOOLEAN DEFAULT FALSE,
    has_gaps BOOLEAN DEFAULT FALSE,
    gap_count INTEGER DEFAULT 0,

    -- Events
    event_count INTEGER DEFAULT 0,
    current_custodian_id UUID,
    current_location GEOGRAPHY(POINT, 4326),

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Custody Events
CREATE TABLE custody_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chain_id UUID REFERENCES custody_chains(chain_id),
    sequence_number INTEGER NOT NULL,

    -- Event details
    event_type VARCHAR(50) NOT NULL,
    event_date TIMESTAMP NOT NULL,
    event_location GEOGRAPHY(POINT, 4326),
    facility_id UUID,

    -- Parties
    from_entity_id UUID,
    from_entity_name VARCHAR(500),
    to_entity_id UUID,
    to_entity_name VARCHAR(500),

    -- Quantity
    quantity DECIMAL(15,3) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL,

    -- Documents
    document_references JSONB DEFAULT '[]',
    verification_status VARCHAR(50) DEFAULT 'PENDING',

    -- Audit
    recorded_at TIMESTAMP DEFAULT NOW(),
    recorded_by VARCHAR(100),
    evidence JSONB DEFAULT '{}',

    UNIQUE(chain_id, sequence_number),

    CONSTRAINT valid_event_type CHECK (
        event_type IN (
            'HARVEST', 'COLLECTION', 'RECEIPT', 'STORAGE',
            'PROCESSING', 'TRANSFER', 'DISPATCH', 'SHIPPING',
            'CUSTOMS', 'DELIVERY', 'EU_ENTRY'
        )
    )
);

-- Custody Gaps
CREATE TABLE custody_gaps (
    gap_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chain_id UUID REFERENCES custody_chains(chain_id),
    gap_type VARCHAR(50) NOT NULL,
    gap_start_event_id UUID,
    gap_end_event_id UUID,
    gap_duration_hours INTEGER,
    expected_event_type VARCHAR(50),
    severity VARCHAR(20) NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT,
    detected_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_custody_chains_batch ON custody_chains(batch_id);
CREATE INDEX idx_custody_chains_status ON custody_chains(status);
CREATE INDEX idx_custody_events_chain ON custody_events(chain_id);
CREATE INDEX idx_custody_events_date ON custody_events(event_date);
CREATE INDEX idx_custody_gaps_chain ON custody_gaps(chain_id);
```

---

## 4. Functional Requirements

### 4.1 Event Recording
- **FR-001 (P0):** Record all custody transfer events
- **FR-002 (P0):** Link events to source documents
- **FR-003 (P0):** Track quantity at each event
- **FR-004 (P0):** Capture location of each event
- **FR-005 (P1):** Support event corrections with audit trail

### 4.2 Chain Management
- **FR-010 (P0):** Create chain from origin to destination
- **FR-011 (P0):** Verify chain completeness
- **FR-012 (P0):** Detect and flag custody gaps
- **FR-013 (P0):** Reconcile transfers between parties
- **FR-014 (P1):** Support chain merging (aggregation)

### 4.3 Verification
- **FR-020 (P0):** Verify document authenticity
- **FR-021 (P0):** Cross-check with counterparty records
- **FR-022 (P0):** Validate quantity consistency
- **FR-023 (P1):** Detect suspicious patterns

### 4.4 Reporting
- **FR-030 (P0):** Generate chain visualization
- **FR-031 (P0):** Export chain for DDS
- **FR-032 (P0):** Generate custody certificates
- **FR-033 (P1):** Chain completeness reports

---

## 5. Chain Verification Algorithm

```python
def verify_chain(chain_id: UUID) -> ChainVerificationResult:
    """
    Verify completeness and integrity of custody chain.
    """
    chain = get_chain(chain_id)
    events = get_events(chain_id, order_by='sequence_number')

    issues = []
    gaps = []

    # Check for origin event
    if not events or events[0].event_type != 'HARVEST':
        issues.append(Issue(
            type="MISSING_ORIGIN",
            message="Chain does not start with harvest event"
        ))

    # Check event sequence
    expected_flow = [
        'HARVEST', 'COLLECTION', 'RECEIPT', 'PROCESSING',
        'DISPATCH', 'SHIPPING', 'CUSTOMS', 'EU_ENTRY'
    ]

    last_event = None
    for event in events:
        if last_event:
            # Check for custody gaps
            time_gap = event.event_date - last_event.event_date
            if time_gap > timedelta(hours=72):
                gaps.append(Gap(
                    start_event=last_event.event_id,
                    end_event=event.event_id,
                    duration_hours=time_gap.total_seconds() / 3600
                ))

            # Check quantity consistency
            if event.quantity > last_event.quantity * 1.01:  # 1% tolerance
                issues.append(Issue(
                    type="QUANTITY_INCREASE",
                    message=f"Quantity increased from {last_event.quantity} to {event.quantity}"
                ))

            # Check custody transfer
            if last_event.to_entity_id != event.from_entity_id:
                issues.append(Issue(
                    type="CUSTODY_BREAK",
                    message="Custody not transferred to next entity"
                ))

        last_event = event

    # Check for EU entry
    if events[-1].event_type != 'EU_ENTRY':
        issues.append(Issue(
            type="INCOMPLETE_CHAIN",
            message="Chain does not end with EU entry"
        ))

    return ChainVerificationResult(
        chain_id=chain_id,
        is_complete=len(issues) == 0 and len(gaps) == 0,
        has_gaps=len(gaps) > 0,
        gap_count=len(gaps),
        gaps=gaps,
        issues=issues,
        event_count=len(events)
    )
```

---

## 6. API Specification

```yaml
paths:
  /api/v1/custody-chains:
    post:
      summary: Create custody chain
    get:
      summary: List custody chains

  /api/v1/custody-chains/{chain_id}:
    get:
      summary: Get chain details with events

  /api/v1/custody-chains/{chain_id}/events:
    post:
      summary: Add custody event
    get:
      summary: Get chain events

  /api/v1/custody-chains/{chain_id}/verify:
    post:
      summary: Verify chain completeness

  /api/v1/custody-chains/{chain_id}/gaps:
    get:
      summary: Get chain gaps

  /api/v1/custody-chains/{chain_id}/visualize:
    get:
      summary: Get chain visualization data
```

---

## 7. Success Metrics

- **Chain Completeness:** >95% chains complete (no gaps)
- **Verification Rate:** 100% chains verified before DDS
- **Gap Resolution:** <24 hours average resolution time
- **Document Coverage:** 100% events with documents
- **Reconciliation Rate:** >99% transfers reconciled

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
