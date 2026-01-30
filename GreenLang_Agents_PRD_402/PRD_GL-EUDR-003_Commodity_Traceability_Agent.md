# PRD: Commodity Traceability Agent (GL-EUDR-003)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Commodity tracking, batch management, product transformation
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Commodity Traceability Agent (GL-EUDR-003)** tracks the physical flow of commodities from production plots through all processing, transformation, and trading stages to the final EU market entry point. This agent maintains the chain of custody linking finished products back to their geographic origins.

Critical because:
- EUDR requires tracing products to specific plots of origin
- Commodities undergo multiple transformations (e.g., cocoa beans → chocolate)
- Mass balance and segregation must be tracked
- Each batch must link to compliant origin plots

---

## 2. Problem Statement

Commodities covered by EUDR undergo complex transformations:
- **Cocoa:** Bean → Liquor → Butter/Powder → Chocolate
- **Coffee:** Cherry → Green → Roasted → Ground/Instant
- **Palm Oil:** FFB → CPO → Refined → Fractions
- **Wood:** Logs → Lumber → Veneer → Furniture/Paper
- **Soy:** Bean → Oil/Meal → Protein isolate
- **Rubber:** Latex → Sheets → Processed rubber
- **Cattle:** Live → Carcass → Cuts/Leather/Gelatin

Without dedicated traceability:
- Link between finished product and origin is lost
- Transformation yields cannot be validated
- Batch mixing obscures provenance
- Compliance verification becomes impossible

---

## 3. Goals and Non-Goals

### 3.1 Goals (must deliver)

1. **Batch tracking**
   - Unique batch identification
   - Batch lineage (parent → child relationships)
   - Batch splitting and merging

2. **Transformation tracking**
   - Input/output commodity mapping
   - Yield ratios and conversion factors
   - Processing facility linkage

3. **Chain of custody**
   - Unbroken traceability from origin to market
   - Document linkage at each stage
   - Timestamp tracking

4. **Mass balance**
   - Volume reconciliation through chain
   - Yield variance detection
   - Loss accounting

### 3.2 Non-Goals

- Physical inventory management
- Quality testing/grading
- Price/cost tracking
- Logistics optimization

---

## 4. Commodity Transformation Models

### 4.1 Transformation Definitions

```python
class CommodityTransformation:
    """Defines how commodities transform through processing"""

    TRANSFORMATIONS = {
        "COCOA": [
            Transformation(
                input="COCOA_BEANS",
                output=["COCOA_LIQUOR"],
                yield_ratio=0.80,
                process="grinding"
            ),
            Transformation(
                input="COCOA_LIQUOR",
                output=["COCOA_BUTTER", "COCOA_POWDER"],
                yield_ratio={"butter": 0.45, "powder": 0.55},
                process="pressing"
            ),
        ],
        "COFFEE": [
            Transformation(
                input="COFFEE_CHERRY",
                output=["GREEN_COFFEE"],
                yield_ratio=0.20,  # ~5kg cherry = 1kg green
                process="wet_processing"
            ),
            Transformation(
                input="GREEN_COFFEE",
                output=["ROASTED_COFFEE"],
                yield_ratio=0.85,  # Weight loss during roasting
                process="roasting"
            ),
        ],
        "PALM_OIL": [
            Transformation(
                input="FRESH_FRUIT_BUNCHES",
                output=["CRUDE_PALM_OIL", "PALM_KERNEL"],
                yield_ratio={"cpo": 0.22, "pk": 0.05},
                process="milling"
            ),
            Transformation(
                input="CRUDE_PALM_OIL",
                output=["REFINED_PALM_OIL"],
                yield_ratio=0.95,
                process="refining"
            ),
        ],
        "WOOD": [
            Transformation(
                input="ROUNDWOOD",
                output=["SAWNWOOD"],
                yield_ratio=0.45,  # Recovery rate varies
                process="sawmilling"
            ),
            Transformation(
                input="SAWNWOOD",
                output=["VENEER", "PLYWOOD"],
                yield_ratio=0.60,
                process="veneer_peeling"
            ),
        ],
        "SOY": [
            Transformation(
                input="SOYBEANS",
                output=["SOYBEAN_OIL", "SOYBEAN_MEAL"],
                yield_ratio={"oil": 0.18, "meal": 0.79},
                process="crushing"
            ),
        ],
        "RUBBER": [
            Transformation(
                input="FIELD_LATEX",
                output=["DRY_RUBBER"],
                yield_ratio=0.35,  # ~35% dry rubber content
                process="processing"
            ),
        ],
        "CATTLE": [
            Transformation(
                input="LIVE_CATTLE",
                output=["BEEF", "HIDES", "OFFAL"],
                yield_ratio={"beef": 0.45, "hides": 0.07, "offal": 0.15},
                process="slaughter"
            ),
        ],
    }
```

---

## 5. Data Model

### 5.1 Core Schema

```sql
-- Batches
CREATE TABLE commodity_batches (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_number VARCHAR(100) UNIQUE NOT NULL,
    commodity_category VARCHAR(50) NOT NULL,  -- EUDR category
    commodity_type VARCHAR(100) NOT NULL,     -- Specific type
    commodity_form VARCHAR(100),              -- Physical form

    -- Quantity
    quantity DECIMAL(15,3) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL,
    initial_quantity DECIMAL(15,3),

    -- Origin linkage
    origin_plots UUID[],  -- Array of plot IDs
    origin_countries TEXT[],
    supplier_id UUID,
    producer_ids UUID[],

    -- Processing stage
    processing_stage VARCHAR(100),
    facility_id UUID,

    -- Traceability
    parent_batch_ids UUID[],  -- Input batches (for transformed)
    child_batch_ids UUID[],   -- Output batches
    transformation_id UUID REFERENCES transformations(id),

    -- Chain of custody
    custody_chain JSONB DEFAULT '[]',
    documents JSONB DEFAULT '[]',

    -- Dates
    production_date DATE,
    harvest_date DATE,
    expiry_date DATE,
    received_date TIMESTAMP,

    -- Status
    status VARCHAR(50) DEFAULT 'ACTIVE',
    compliance_status VARCHAR(50) DEFAULT 'PENDING',

    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_commodity CHECK (
        commodity_category IN ('CATTLE', 'COCOA', 'COFFEE', 'PALM_OIL', 'RUBBER', 'SOY', 'WOOD')
    )
);

-- Transformations
CREATE TABLE transformations (
    transformation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id UUID REFERENCES facilities(id),

    -- Inputs
    input_batches JSONB NOT NULL,  -- [{batch_id, quantity}]
    input_commodity VARCHAR(100) NOT NULL,
    total_input_quantity DECIMAL(15,3) NOT NULL,

    -- Outputs
    output_batches JSONB NOT NULL,  -- [{batch_id, quantity}]
    output_commodity VARCHAR(100) NOT NULL,
    total_output_quantity DECIMAL(15,3) NOT NULL,

    -- Yield
    expected_yield_ratio DECIMAL(5,4),
    actual_yield_ratio DECIMAL(5,4),
    yield_variance DECIMAL(5,4),

    -- Process
    process_type VARCHAR(100) NOT NULL,
    process_date TIMESTAMP NOT NULL,

    -- Compliance
    compliance_verified BOOLEAN DEFAULT FALSE,
    origin_plots_verified UUID[],

    created_at TIMESTAMP DEFAULT NOW()
);

-- Chain of Custody Events
CREATE TABLE custody_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID REFERENCES commodity_batches(batch_id),

    event_type VARCHAR(50) NOT NULL,
    event_date TIMESTAMP NOT NULL,

    from_entity_id UUID,
    from_entity_type VARCHAR(50),
    to_entity_id UUID,
    to_entity_type VARCHAR(50),

    location GEOGRAPHY(POINT, 4326),
    facility_id UUID,

    quantity DECIMAL(15,3),
    quantity_unit VARCHAR(20),

    documents JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',

    CONSTRAINT valid_event_type CHECK (
        event_type IN (
            'HARVEST', 'COLLECTION', 'RECEIPT', 'PROCESSING',
            'STORAGE', 'SHIPMENT', 'CUSTOMS', 'DELIVERY', 'SALE'
        )
    )
);

-- Mass Balance Tracking
CREATE TABLE mass_balance_ledger (
    entry_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID REFERENCES commodity_batches(batch_id),
    facility_id UUID,

    entry_type VARCHAR(20) NOT NULL,  -- CREDIT or DEBIT
    quantity DECIMAL(15,3) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL,

    reference_type VARCHAR(50),  -- Receipt, Shipment, Processing, Loss
    reference_id UUID,

    running_balance DECIMAL(15,3),
    entry_date TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_entry_type CHECK (
        entry_type IN ('CREDIT', 'DEBIT')
    )
);

-- Indexes
CREATE INDEX idx_batches_commodity ON commodity_batches(commodity_category);
CREATE INDEX idx_batches_supplier ON commodity_batches(supplier_id);
CREATE INDEX idx_batches_plots ON commodity_batches USING GIN(origin_plots);
CREATE INDEX idx_batches_parent ON commodity_batches USING GIN(parent_batch_ids);
CREATE INDEX idx_custody_batch ON custody_events(batch_id);
CREATE INDEX idx_custody_date ON custody_events(event_date);
CREATE INDEX idx_balance_batch ON mass_balance_ledger(batch_id);
```

---

## 6. Functional Requirements

### 6.1 Batch Management
- **FR-001 (P0):** Create new batch with unique identifier
- **FR-002 (P0):** Link batch to origin plots
- **FR-003 (P0):** Track batch quantity and unit
- **FR-004 (P0):** Support batch splitting (one batch → multiple)
- **FR-005 (P0):** Support batch merging (multiple → one batch)

### 6.2 Transformation Tracking
- **FR-010 (P0):** Record commodity transformations
- **FR-011 (P0):** Link input batches to output batches
- **FR-012 (P0):** Calculate and validate yield ratios
- **FR-013 (P0):** Propagate origin plots through transformations
- **FR-014 (P1):** Alert on abnormal yield variance

### 6.3 Chain of Custody
- **FR-020 (P0):** Record custody transfer events
- **FR-021 (P0):** Track complete custody chain from origin
- **FR-022 (P0):** Link documents to custody events
- **FR-023 (P1):** Verify custody chain completeness
- **FR-024 (P1):** Identify custody gaps

### 6.4 Mass Balance
- **FR-030 (P0):** Maintain mass balance ledger per facility
- **FR-031 (P0):** Track credits (receipts) and debits (shipments)
- **FR-032 (P0):** Calculate running balance
- **FR-033 (P1):** Reconcile physical vs book inventory
- **FR-034 (P1):** Alert on balance discrepancies

### 6.5 Compliance Verification
- **FR-040 (P0):** Verify all origin plots are compliant
- **FR-041 (P0):** Calculate compliant vs non-compliant volume
- **FR-042 (P0):** Generate traceability report for DDS
- **FR-043 (P1):** Support segregated vs mass balance models
- **FR-044 (P1):** Identify contamination (non-compliant mixing)

---

## 7. Traceability Algorithms

### 7.1 Origin Propagation

```python
def propagate_origins(output_batch_id: UUID) -> List[UUID]:
    """
    Trace back through transformations to find all origin plots.
    """
    output_batch = get_batch(output_batch_id)

    # If batch has direct origin plots, return them
    if output_batch.origin_plots:
        return output_batch.origin_plots

    # Otherwise, trace through parent batches
    all_origins = []

    for parent_id in output_batch.parent_batch_ids:
        parent_origins = propagate_origins(parent_id)
        all_origins.extend(parent_origins)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(all_origins))


def calculate_origin_contribution(
    batch_id: UUID
) -> Dict[UUID, float]:
    """
    Calculate what percentage each origin plot contributes
    to the final batch.
    """
    batch = get_batch(batch_id)

    if not batch.parent_batch_ids:
        # This is an origin batch
        plot_contribution = 1.0 / len(batch.origin_plots)
        return {plot_id: plot_contribution for plot_id in batch.origin_plots}

    contributions = defaultdict(float)
    total_input = sum(
        get_batch(p).quantity for p in batch.parent_batch_ids
    )

    for parent_id in batch.parent_batch_ids:
        parent = get_batch(parent_id)
        parent_weight = parent.quantity / total_input

        parent_contributions = calculate_origin_contribution(parent_id)
        for plot_id, contrib in parent_contributions.items():
            contributions[plot_id] += contrib * parent_weight

    return dict(contributions)
```

### 7.2 Compliance Verification

```python
def verify_batch_compliance(batch_id: UUID) -> ComplianceResult:
    """
    Verify if batch is fully traceable to compliant origins.
    """
    origin_contributions = calculate_origin_contribution(batch_id)

    compliant_volume = 0.0
    non_compliant_volume = 0.0
    unknown_volume = 0.0

    issues = []

    for plot_id, contribution in origin_contributions.items():
        plot = get_plot(plot_id)

        if plot is None:
            unknown_volume += contribution
            issues.append(ComplianceIssue(
                type="UNKNOWN_ORIGIN",
                plot_id=plot_id,
                contribution=contribution
            ))
            continue

        if plot.deforestation_risk_score and plot.deforestation_risk_score > 0.7:
            non_compliant_volume += contribution
            issues.append(ComplianceIssue(
                type="HIGH_DEFORESTATION_RISK",
                plot_id=plot_id,
                risk_score=plot.deforestation_risk_score,
                contribution=contribution
            ))
        elif plot.validation_status != "VALID":
            unknown_volume += contribution
            issues.append(ComplianceIssue(
                type="UNVALIDATED_ORIGIN",
                plot_id=plot_id,
                contribution=contribution
            ))
        else:
            compliant_volume += contribution

    return ComplianceResult(
        batch_id=batch_id,
        is_compliant=non_compliant_volume == 0 and unknown_volume < 0.05,
        compliant_percentage=compliant_volume * 100,
        non_compliant_percentage=non_compliant_volume * 100,
        unknown_percentage=unknown_volume * 100,
        issues=issues
    )
```

---

## 8. API Specification

```yaml
paths:
  /api/v1/batches:
    post:
      summary: Create commodity batch
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateBatchRequest'
      responses:
        201:
          description: Batch created

  /api/v1/batches/{batch_id}/trace:
    get:
      summary: Trace batch to origins
      parameters:
        - name: batch_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        200:
          description: Traceability chain
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TraceabilityResult'

  /api/v1/batches/{batch_id}/compliance:
    get:
      summary: Verify batch compliance
      responses:
        200:
          description: Compliance verification result

  /api/v1/transformations:
    post:
      summary: Record commodity transformation
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TransformationRequest'

  /api/v1/custody-events:
    post:
      summary: Record custody transfer event

components:
  schemas:
    CreateBatchRequest:
      type: object
      required:
        - commodity_category
        - commodity_type
        - quantity
        - quantity_unit
      properties:
        batch_number:
          type: string
        commodity_category:
          type: string
          enum: [CATTLE, COCOA, COFFEE, PALM_OIL, RUBBER, SOY, WOOD]
        commodity_type:
          type: string
        quantity:
          type: number
        quantity_unit:
          type: string
        origin_plots:
          type: array
          items:
            type: string
            format: uuid
        supplier_id:
          type: string
          format: uuid
        production_date:
          type: string
          format: date

    TransformationRequest:
      type: object
      required:
        - input_batches
        - output_batches
        - process_type
        - facility_id
      properties:
        input_batches:
          type: array
          items:
            type: object
            properties:
              batch_id:
                type: string
                format: uuid
              quantity:
                type: number
        output_batches:
          type: array
          items:
            type: object
            properties:
              commodity_type:
                type: string
              quantity:
                type: number
              quantity_unit:
                type: string
        process_type:
          type: string
        facility_id:
          type: string
          format: uuid
        process_date:
          type: string
          format: date-time
```

---

## 9. Success Metrics

- **Traceability Coverage:** 100% of batches traced to origin
- **Compliance Accuracy:** 99% accuracy in compliance determination
- **Chain Completeness:** <5% batches with custody gaps
- **Mass Balance Accuracy:** <2% variance in reconciliation
- **Processing Speed:** <3 seconds for full trace query

---

## 10. Testing Strategy

### 10.1 Transformation Tests
- Valid transformation with correct yield
- Multi-output transformations
- Chain of transformations

### 10.2 Traceability Tests
- Single origin tracing
- Multi-origin (merged) tracing
- Complex transformation chains

### 10.3 Compliance Tests
- Fully compliant batch
- Partially compliant (contaminated)
- Unknown origins

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
