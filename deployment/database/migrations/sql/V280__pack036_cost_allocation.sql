-- =============================================================================
-- V280: PACK-036 Utility Analysis Pack - Cost Allocation
-- =============================================================================
-- Pack:         PACK-036 (Utility Analysis Pack)
-- Migration:    005 of 010
-- Date:         March 2026
--
-- Tables for utility cost allocation across departments, tenants, cost
-- centres, and sub-metered zones. Supports multiple allocation methods
-- (direct metering, floor area, headcount, operating hours, production
-- units) and full reconciliation tracking.
--
-- Tables (4):
--   1. pack036_utility_analysis.gl_allocation_entities
--   2. pack036_utility_analysis.gl_allocation_rules
--   3. pack036_utility_analysis.gl_allocation_results
--   4. pack036_utility_analysis.gl_allocation_line_items
--
-- Previous: V279__pack036_demand_profiles.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack036_utility_analysis.gl_allocation_entities
-- =============================================================================
-- Entities (departments, tenants, zones, cost centres) to which utility
-- costs are allocated. Each entity has attributes that serve as allocation
-- bases (floor area, headcount, operating hours, production units).

CREATE TABLE pack036_utility_analysis.gl_allocation_entities (
    entity_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    entity_code             VARCHAR(50),
    entity_type             VARCHAR(30)     NOT NULL,
    parent_entity_id        UUID            REFERENCES pack036_utility_analysis.gl_allocation_entities(entity_id) ON DELETE SET NULL,
    floor_area_m2           NUMERIC(12,2),
    headcount               INTEGER,
    operating_hours         NUMERIC(8,2),
    production_units        NUMERIC(14,4),
    sub_meter_id            VARCHAR(100),
    allocation_weight       NUMERIC(8,4)    DEFAULT 1.0,
    cost_centre_code        VARCHAR(50),
    is_active               BOOLEAN         NOT NULL DEFAULT true,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_ae_entity_type CHECK (
        entity_type IN (
            'DEPARTMENT', 'TENANT', 'ZONE', 'FLOOR', 'BUILDING_WING',
            'COST_CENTRE', 'BUSINESS_UNIT', 'PRODUCTION_LINE', 'COMMON_AREA'
        )
    ),
    CONSTRAINT chk_p036_ae_area CHECK (
        floor_area_m2 IS NULL OR floor_area_m2 >= 0
    ),
    CONSTRAINT chk_p036_ae_headcount CHECK (
        headcount IS NULL OR headcount >= 0
    ),
    CONSTRAINT chk_p036_ae_hours CHECK (
        operating_hours IS NULL OR (operating_hours >= 0 AND operating_hours <= 8760)
    ),
    CONSTRAINT chk_p036_ae_production CHECK (
        production_units IS NULL OR production_units >= 0
    ),
    CONSTRAINT chk_p036_ae_weight CHECK (
        allocation_weight > 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_ae_tenant         ON pack036_utility_analysis.gl_allocation_entities(tenant_id);
CREATE INDEX idx_p036_ae_facility       ON pack036_utility_analysis.gl_allocation_entities(facility_id);
CREATE INDEX idx_p036_ae_type           ON pack036_utility_analysis.gl_allocation_entities(entity_type);
CREATE INDEX idx_p036_ae_parent         ON pack036_utility_analysis.gl_allocation_entities(parent_entity_id);
CREATE INDEX idx_p036_ae_sub_meter      ON pack036_utility_analysis.gl_allocation_entities(sub_meter_id);
CREATE INDEX idx_p036_ae_cost_centre    ON pack036_utility_analysis.gl_allocation_entities(cost_centre_code);
CREATE INDEX idx_p036_ae_active         ON pack036_utility_analysis.gl_allocation_entities(is_active);
CREATE INDEX idx_p036_ae_created        ON pack036_utility_analysis.gl_allocation_entities(created_at DESC);
CREATE INDEX idx_p036_ae_metadata       ON pack036_utility_analysis.gl_allocation_entities USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_ae_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_allocation_entities
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack036_utility_analysis.gl_allocation_rules
-- =============================================================================
-- Rules governing how specific cost components are allocated across
-- entities within a facility. Each rule specifies the allocation method,
-- parameters, and priority for conflict resolution.

CREATE TABLE pack036_utility_analysis.gl_allocation_rules (
    rule_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    rule_name               VARCHAR(255),
    cost_component          VARCHAR(50)     NOT NULL,
    method                  VARCHAR(30)     NOT NULL,
    parameters              JSONB           DEFAULT '{}',
    priority                INTEGER         NOT NULL DEFAULT 100,
    applies_to_commodity    VARCHAR(30),
    is_active               BOOLEAN         NOT NULL DEFAULT true,
    effective_date          DATE,
    expiry_date             DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_ar_cost_comp CHECK (
        cost_component IN (
            'ENERGY', 'DEMAND', 'FIXED', 'TRANSMISSION', 'DISTRIBUTION',
            'TAXES', 'SURCHARGES', 'TOTAL', 'COMMON_AREA', 'REACTIVE_POWER'
        )
    ),
    CONSTRAINT chk_p036_ar_method CHECK (
        method IN (
            'DIRECT_METER', 'FLOOR_AREA', 'HEADCOUNT', 'OPERATING_HOURS',
            'PRODUCTION_UNITS', 'EQUAL_SHARE', 'WEIGHTED', 'CUSTOM',
            'PEAK_DEMAND', 'CONSUMPTION_RATIO'
        )
    ),
    CONSTRAINT chk_p036_ar_priority CHECK (
        priority >= 1 AND priority <= 1000
    ),
    CONSTRAINT chk_p036_ar_commodity CHECK (
        applies_to_commodity IS NULL OR applies_to_commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'WATER', 'STEAM',
            'CHILLED_WATER', 'ALL'
        )
    ),
    CONSTRAINT chk_p036_ar_dates CHECK (
        effective_date IS NULL OR expiry_date IS NULL OR expiry_date > effective_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_ar_tenant         ON pack036_utility_analysis.gl_allocation_rules(tenant_id);
CREATE INDEX idx_p036_ar_facility       ON pack036_utility_analysis.gl_allocation_rules(facility_id);
CREATE INDEX idx_p036_ar_cost_comp      ON pack036_utility_analysis.gl_allocation_rules(cost_component);
CREATE INDEX idx_p036_ar_method         ON pack036_utility_analysis.gl_allocation_rules(method);
CREATE INDEX idx_p036_ar_priority       ON pack036_utility_analysis.gl_allocation_rules(priority);
CREATE INDEX idx_p036_ar_active         ON pack036_utility_analysis.gl_allocation_rules(is_active);
CREATE INDEX idx_p036_ar_created        ON pack036_utility_analysis.gl_allocation_rules(created_at DESC);

-- Composite: facility + cost component + active for rule lookup
CREATE INDEX idx_p036_ar_fac_comp       ON pack036_utility_analysis.gl_allocation_rules(facility_id, cost_component, priority)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_ar_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_allocation_rules
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack036_utility_analysis.gl_allocation_results
-- =============================================================================
-- Period-level allocation results summarizing total cost, allocated cost,
-- unallocated remainder, variance, and reconciliation status.

CREATE TABLE pack036_utility_analysis.gl_allocation_results (
    result_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    commodity               VARCHAR(30)     NOT NULL DEFAULT 'ELECTRICITY',
    total_cost_eur          NUMERIC(14,2)   NOT NULL,
    allocated_eur           NUMERIC(14,2)   NOT NULL DEFAULT 0,
    unallocated_eur         NUMERIC(14,2)   NOT NULL DEFAULT 0,
    variance_pct            NUMERIC(8,4),
    reconciliation_status   VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    entities_count          INTEGER         DEFAULT 0,
    bill_ids                UUID[],
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_ares_period CHECK (
        period_end >= period_start
    ),
    CONSTRAINT chk_p036_ares_total CHECK (
        total_cost_eur >= 0
    ),
    CONSTRAINT chk_p036_ares_allocated CHECK (
        allocated_eur >= 0
    ),
    CONSTRAINT chk_p036_ares_unallocated CHECK (
        unallocated_eur >= 0
    ),
    CONSTRAINT chk_p036_ares_recon CHECK (
        reconciliation_status IN (
            'PENDING', 'BALANCED', 'VARIANCE', 'APPROVED', 'DISPUTED'
        )
    ),
    CONSTRAINT chk_p036_ares_commodity CHECK (
        commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'WATER', 'STEAM',
            'CHILLED_WATER', 'ALL'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_ares_tenant       ON pack036_utility_analysis.gl_allocation_results(tenant_id);
CREATE INDEX idx_p036_ares_facility     ON pack036_utility_analysis.gl_allocation_results(facility_id);
CREATE INDEX idx_p036_ares_period       ON pack036_utility_analysis.gl_allocation_results(period_start DESC);
CREATE INDEX idx_p036_ares_commodity    ON pack036_utility_analysis.gl_allocation_results(commodity);
CREATE INDEX idx_p036_ares_recon        ON pack036_utility_analysis.gl_allocation_results(reconciliation_status);
CREATE INDEX idx_p036_ares_created      ON pack036_utility_analysis.gl_allocation_results(created_at DESC);

-- Composite: facility + period for time-series allocation lookup
CREATE INDEX idx_p036_ares_fac_period   ON pack036_utility_analysis.gl_allocation_results(facility_id, period_start DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_ares_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_allocation_results
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack036_utility_analysis.gl_allocation_line_items
-- =============================================================================
-- Individual allocation line items showing how much of each cost component
-- was allocated to each entity, using which method and basis value.

CREATE TABLE pack036_utility_analysis.gl_allocation_line_items (
    item_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    result_id               UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_allocation_results(result_id) ON DELETE CASCADE,
    entity_id               UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_allocation_entities(entity_id) ON DELETE CASCADE,
    cost_component          VARCHAR(50)     NOT NULL,
    allocated_amount_eur    NUMERIC(14,2)   NOT NULL,
    method                  VARCHAR(30)     NOT NULL,
    basis_value             NUMERIC(14,4),
    basis_unit              VARCHAR(30),
    share_pct               NUMERIC(8,4),
    rule_id                 UUID            REFERENCES pack036_utility_analysis.gl_allocation_rules(rule_id) ON DELETE SET NULL,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_ali_allocated CHECK (
        allocated_amount_eur >= 0
    ),
    CONSTRAINT chk_p036_ali_share CHECK (
        share_pct IS NULL OR (share_pct >= 0 AND share_pct <= 100)
    ),
    CONSTRAINT chk_p036_ali_cost_comp CHECK (
        cost_component IN (
            'ENERGY', 'DEMAND', 'FIXED', 'TRANSMISSION', 'DISTRIBUTION',
            'TAXES', 'SURCHARGES', 'TOTAL', 'COMMON_AREA', 'REACTIVE_POWER'
        )
    ),
    CONSTRAINT chk_p036_ali_method CHECK (
        method IN (
            'DIRECT_METER', 'FLOOR_AREA', 'HEADCOUNT', 'OPERATING_HOURS',
            'PRODUCTION_UNITS', 'EQUAL_SHARE', 'WEIGHTED', 'CUSTOM',
            'PEAK_DEMAND', 'CONSUMPTION_RATIO'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_ali_result        ON pack036_utility_analysis.gl_allocation_line_items(result_id);
CREATE INDEX idx_p036_ali_entity        ON pack036_utility_analysis.gl_allocation_line_items(entity_id);
CREATE INDEX idx_p036_ali_cost_comp     ON pack036_utility_analysis.gl_allocation_line_items(cost_component);
CREATE INDEX idx_p036_ali_method        ON pack036_utility_analysis.gl_allocation_line_items(method);
CREATE INDEX idx_p036_ali_rule          ON pack036_utility_analysis.gl_allocation_line_items(rule_id);
CREATE INDEX idx_p036_ali_created       ON pack036_utility_analysis.gl_allocation_line_items(created_at DESC);

-- Composite: entity + cost component for entity cost breakdown
CREATE INDEX idx_p036_ali_entity_comp   ON pack036_utility_analysis.gl_allocation_line_items(entity_id, cost_component);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack036_utility_analysis.gl_allocation_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_allocation_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_allocation_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_allocation_line_items ENABLE ROW LEVEL SECURITY;

CREATE POLICY p036_ae_tenant_isolation
    ON pack036_utility_analysis.gl_allocation_entities
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_ae_service_bypass
    ON pack036_utility_analysis.gl_allocation_entities
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_ar_tenant_isolation
    ON pack036_utility_analysis.gl_allocation_rules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_ar_service_bypass
    ON pack036_utility_analysis.gl_allocation_rules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_ares_tenant_isolation
    ON pack036_utility_analysis.gl_allocation_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_ares_service_bypass
    ON pack036_utility_analysis.gl_allocation_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_ali_tenant_isolation
    ON pack036_utility_analysis.gl_allocation_line_items
    USING (result_id IN (
        SELECT result_id FROM pack036_utility_analysis.gl_allocation_results
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p036_ali_service_bypass
    ON pack036_utility_analysis.gl_allocation_line_items
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_allocation_entities TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_allocation_rules TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_allocation_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_allocation_line_items TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack036_utility_analysis.gl_allocation_entities IS
    'Entities (departments, tenants, zones) to which utility costs are allocated with area, headcount, and production attributes.';

COMMENT ON TABLE pack036_utility_analysis.gl_allocation_rules IS
    'Rules governing how cost components are allocated across entities including method, parameters, and priority.';

COMMENT ON TABLE pack036_utility_analysis.gl_allocation_results IS
    'Period-level allocation results with total cost, allocated cost, unallocated remainder, and reconciliation status.';

COMMENT ON TABLE pack036_utility_analysis.gl_allocation_line_items IS
    'Individual allocation line items showing cost allocated to each entity by method and basis value.';

COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_entities.entity_id IS
    'Unique identifier for the allocation entity.';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_entities.entity_type IS
    'Entity type: DEPARTMENT, TENANT, ZONE, FLOOR, BUILDING_WING, COST_CENTRE, BUSINESS_UNIT, PRODUCTION_LINE, COMMON_AREA.';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_entities.floor_area_m2 IS
    'Floor area in square metres used as allocation basis for area-based methods.';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_entities.sub_meter_id IS
    'Sub-meter ID for direct metering allocation method.';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_entities.allocation_weight IS
    'Custom allocation weight for weighted allocation method (default 1.0).';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_rules.method IS
    'Allocation method: DIRECT_METER, FLOOR_AREA, HEADCOUNT, OPERATING_HOURS, PRODUCTION_UNITS, EQUAL_SHARE, WEIGHTED, CUSTOM.';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_rules.priority IS
    'Rule priority for conflict resolution (lower number = higher priority, range 1-1000).';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_results.variance_pct IS
    'Variance percentage between total cost and sum of allocated amounts.';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_results.reconciliation_status IS
    'Reconciliation status: PENDING, BALANCED, VARIANCE, APPROVED, DISPUTED.';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_results.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_line_items.share_pct IS
    'Share percentage of total cost allocated to this entity for this component.';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_line_items.basis_value IS
    'Numerical basis value used for allocation (e.g., 500 m2, 20 headcount).';
COMMENT ON COLUMN pack036_utility_analysis.gl_allocation_line_items.basis_unit IS
    'Unit of the basis value (e.g., m2, headcount, hours, units).';
