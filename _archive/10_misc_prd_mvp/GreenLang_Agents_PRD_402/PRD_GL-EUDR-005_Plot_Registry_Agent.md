# PRD: Plot Registry Agent (GL-EUDR-005)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Production plot management, registry maintenance
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Plot Registry Agent (GL-EUDR-005)** maintains the authoritative registry of all production plots (farms, forests, plantations) in the supply chain. It serves as the single source of truth for plot information, linking geographic data with ownership, commodities, and compliance status.

Critical because:
- EUDR requires a registry of all plots supplying commodities
- Plot data must be consistently maintained and versioned
- Risk assessments depend on accurate plot registry
- Due diligence statements reference specific plots

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. **Registry management**
   - Central plot database with unique identifiers
   - Plot versioning and change history
   - Ownership and operator tracking

2. **Plot lifecycle**
   - Registration and onboarding
   - Updates and amendments
   - Deactivation and archival

3. **Cross-reference**
   - Link plots to suppliers
   - Link plots to commodities
   - Link plots to risk assessments

4. **Search and query**
   - Spatial queries (proximity, overlap)
   - Attribute queries (commodity, country, status)
   - Full-text search

### 2.2 Non-Goals

- Geolocation validation (GL-EUDR-002)
- Deforestation assessment (GL-EUDR-020+)
- Supply chain mapping (GL-EUDR-001)

---

## 3. Data Model

```sql
-- Plot Registry (Master Table)
CREATE TABLE plot_registry (
    plot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_code VARCHAR(100) UNIQUE NOT NULL,  -- Human-readable code

    -- Ownership
    owner_name VARCHAR(500),
    owner_type VARCHAR(50),  -- INDIVIDUAL, COOPERATIVE, COMPANY
    operator_id UUID,  -- Who operates (may differ from owner)

    -- Location
    geometry GEOMETRY(GEOMETRY, 4326) NOT NULL,
    centroid GEOMETRY(POINT, 4326) GENERATED ALWAYS AS (ST_Centroid(geometry)) STORED,
    area_hectares DECIMAL(10,4) NOT NULL,
    country_code CHAR(2) NOT NULL,
    region VARCHAR(255),
    district VARCHAR(255),

    -- Commodity
    primary_commodity VARCHAR(50) NOT NULL,
    secondary_commodities TEXT[],
    production_type VARCHAR(100),  -- PLANTATION, SMALLHOLDER, FOREST

    -- Dates
    established_date DATE,
    first_registered DATE DEFAULT CURRENT_DATE,
    last_production_date DATE,

    -- Status
    status VARCHAR(50) DEFAULT 'ACTIVE',
    verification_status VARCHAR(50) DEFAULT 'UNVERIFIED',
    compliance_status VARCHAR(50) DEFAULT 'PENDING',
    last_risk_assessment DATE,
    risk_score DECIMAL(3,2),

    -- Documents
    registration_documents JSONB DEFAULT '[]',
    certifications JSONB DEFAULT '[]',

    -- Versioning
    version INTEGER DEFAULT 1,
    previous_version_id UUID,

    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    CONSTRAINT valid_status CHECK (
        status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED', 'ARCHIVED')
    )
);

-- Plot Change History
CREATE TABLE plot_history (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID REFERENCES plot_registry(plot_id),
    version INTEGER NOT NULL,
    change_type VARCHAR(50) NOT NULL,
    changed_fields JSONB NOT NULL,
    previous_values JSONB,
    new_values JSONB,
    change_reason TEXT,
    changed_at TIMESTAMP DEFAULT NOW(),
    changed_by VARCHAR(100),

    CONSTRAINT valid_change_type CHECK (
        change_type IN ('CREATE', 'UPDATE', 'DEACTIVATE', 'MERGE', 'SPLIT')
    )
);

-- Plot Supplier Links
CREATE TABLE plot_supplier_links (
    link_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID REFERENCES plot_registry(plot_id),
    supplier_id UUID NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,  -- OWNER, OPERATOR, BUYER
    commodity VARCHAR(50),
    volume_share DECIMAL(5,2),  -- Percentage if multiple buyers
    start_date DATE,
    end_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    documents JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Spatial indexes
CREATE INDEX idx_plot_registry_geometry ON plot_registry USING GIST (geometry);
CREATE INDEX idx_plot_registry_centroid ON plot_registry USING GIST (centroid);
CREATE INDEX idx_plot_registry_country ON plot_registry (country_code);
CREATE INDEX idx_plot_registry_commodity ON plot_registry (primary_commodity);
CREATE INDEX idx_plot_registry_status ON plot_registry (status);
CREATE INDEX idx_plot_history_plot ON plot_history (plot_id);
CREATE INDEX idx_plot_supplier_plot ON plot_supplier_links (plot_id);
CREATE INDEX idx_plot_supplier_supplier ON plot_supplier_links (supplier_id);
```

---

## 4. Functional Requirements

### 4.1 Registration
- **FR-001 (P0):** Register new plots with unique identifiers
- **FR-002 (P0):** Validate required fields before registration
- **FR-003 (P0):** Generate human-readable plot codes
- **FR-004 (P0):** Prevent duplicate registrations
- **FR-005 (P1):** Support bulk registration

### 4.2 Updates
- **FR-010 (P0):** Track all changes with version history
- **FR-011 (P0):** Preserve previous geometry versions
- **FR-012 (P0):** Require change reason for updates
- **FR-013 (P1):** Support plot splitting (one → multiple)
- **FR-014 (P1):** Support plot merging (multiple → one)

### 4.3 Queries
- **FR-020 (P0):** Query by plot ID or code
- **FR-021 (P0):** Spatial query (point-in-polygon, overlap)
- **FR-022 (P0):** Filter by commodity, country, status
- **FR-023 (P1):** Full-text search on owner/location
- **FR-024 (P1):** Export query results (GeoJSON, CSV)

### 4.4 Linking
- **FR-030 (P0):** Link plots to suppliers
- **FR-031 (P0):** Link plots to risk assessments
- **FR-032 (P0):** Track plot-to-batch relationships
- **FR-033 (P1):** Multi-supplier plots (cooperatives)

---

## 5. API Specification

```yaml
paths:
  /api/v1/plots:
    post:
      summary: Register new plot
    get:
      summary: List/search plots
      parameters:
        - name: commodity
        - name: country
        - name: status
        - name: bbox  # Bounding box for spatial filter

  /api/v1/plots/{plot_id}:
    get:
      summary: Get plot details
    patch:
      summary: Update plot
    delete:
      summary: Deactivate plot

  /api/v1/plots/{plot_id}/history:
    get:
      summary: Get plot change history

  /api/v1/plots/{plot_id}/suppliers:
    get:
      summary: Get linked suppliers
    post:
      summary: Link supplier to plot

  /api/v1/plots/search/spatial:
    post:
      summary: Spatial search
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                geometry:
                  type: object  # GeoJSON
                operation:
                  type: string
                  enum: [contains, intersects, within, overlaps]

  /api/v1/plots/bulk:
    post:
      summary: Bulk register plots
```

---

## 6. Plot Code Format

```python
def generate_plot_code(
    country_code: str,
    commodity: str,
    region: str,
    sequence: int
) -> str:
    """
    Generate human-readable plot code.
    Format: {COUNTRY}-{COMMODITY}-{REGION}-{SEQUENCE}
    Example: ID-PLM-SUM-000123 (Indonesia Palm Oil, Sumatra)
    """
    commodity_codes = {
        "CATTLE": "CAT",
        "COCOA": "COC",
        "COFFEE": "COF",
        "PALM_OIL": "PLM",
        "RUBBER": "RUB",
        "SOY": "SOY",
        "WOOD": "WOD"
    }

    region_code = region[:3].upper() if region else "XXX"
    comm_code = commodity_codes.get(commodity, "XXX")

    return f"{country_code}-{comm_code}-{region_code}-{sequence:06d}"
```

---

## 7. Success Metrics

- **Registry Coverage:** 100% of supplying plots registered
- **Data Quality:** <1% duplicate plots
- **Query Performance:** <500ms for spatial queries
- **Version Integrity:** 100% of changes tracked
- **Link Accuracy:** 100% of plots linked to suppliers

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
