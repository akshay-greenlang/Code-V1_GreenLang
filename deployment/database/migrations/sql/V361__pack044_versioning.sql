-- =============================================================================
-- V361: PACK-044 GHG Inventory Management - Versioning Tables
-- =============================================================================
-- Pack:         PACK-044 (GHG Inventory Management)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Inventory versioning and snapshot tables. Maintains a complete version
-- history of the GHG inventory at each significant milestone (approval,
-- restatement, publication). Version snapshots capture the full state of
-- emissions data at a point in time. Field-level change tracking records
-- every individual data modification for complete audit trail.
--
-- Tables (3):
--   1. ghg_inventory.gl_inv_versions
--   2. ghg_inventory.gl_inv_version_snapshots
--   3. ghg_inventory.gl_inv_field_changes
--
-- Previous: V360__pack044_review_approval.sql
-- =============================================================================

SET search_path TO ghg_inventory, public;

-- =============================================================================
-- Table 1: ghg_inventory.gl_inv_versions
-- =============================================================================
-- Versioned releases of an inventory period. Each version represents a
-- complete, internally consistent state of the inventory. Versions are
-- created at approval milestones, restatements, or publication events.
-- The is_current flag identifies the active version.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_versions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    version_number              INTEGER         NOT NULL DEFAULT 1,
    version_label               VARCHAR(100),
    version_type                VARCHAR(30)     NOT NULL DEFAULT 'STANDARD',
    description                 TEXT,
    trigger_event               VARCHAR(50)     NOT NULL DEFAULT 'MANUAL',
    change_request_id           UUID,
    is_current                  BOOLEAN         NOT NULL DEFAULT true,
    is_published                BOOLEAN         NOT NULL DEFAULT false,
    total_scope1_tco2e          NUMERIC(14,3),
    total_scope2_location_tco2e NUMERIC(14,3),
    total_scope2_market_tco2e   NUMERIC(14,3),
    total_scope3_tco2e          NUMERIC(14,3),
    total_tco2e                 NUMERIC(14,3),
    delta_from_previous_tco2e   NUMERIC(14,3),
    delta_from_previous_pct     NUMERIC(8,3),
    completeness_pct            NUMERIC(5,2),
    data_quality_score          NUMERIC(5,2),
    created_by_user_id          UUID,
    created_by_name             VARCHAR(255),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_vr_version CHECK (
        version_number >= 1
    ),
    CONSTRAINT chk_p044_vr_type CHECK (
        version_type IN (
            'STANDARD', 'RESTATEMENT', 'CORRECTION', 'INTERIM', 'FINAL', 'PUBLISHED'
        )
    ),
    CONSTRAINT chk_p044_vr_trigger CHECK (
        trigger_event IN (
            'MANUAL', 'APPROVAL', 'RESTATEMENT', 'CHANGE_REQUEST',
            'PUBLICATION', 'BASE_YEAR_RECALC', 'SCHEDULED', 'SYSTEM'
        )
    ),
    CONSTRAINT chk_p044_vr_completeness CHECK (
        completeness_pct IS NULL OR (completeness_pct >= 0 AND completeness_pct <= 100)
    ),
    CONSTRAINT chk_p044_vr_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT uq_p044_vr_period_version UNIQUE (period_id, version_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_vr_tenant          ON ghg_inventory.gl_inv_versions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_vr_period          ON ghg_inventory.gl_inv_versions(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_vr_version         ON ghg_inventory.gl_inv_versions(version_number);
CREATE INDEX IF NOT EXISTS idx_p044_vr_type            ON ghg_inventory.gl_inv_versions(version_type);
CREATE INDEX IF NOT EXISTS idx_p044_vr_trigger         ON ghg_inventory.gl_inv_versions(trigger_event);
CREATE INDEX IF NOT EXISTS idx_p044_vr_current         ON ghg_inventory.gl_inv_versions(is_current) WHERE is_current = true;
CREATE INDEX IF NOT EXISTS idx_p044_vr_published       ON ghg_inventory.gl_inv_versions(is_published) WHERE is_published = true;
CREATE INDEX IF NOT EXISTS idx_p044_vr_created         ON ghg_inventory.gl_inv_versions(created_at DESC);

-- Composite: period + current version
CREATE INDEX IF NOT EXISTS idx_p044_vr_period_current  ON ghg_inventory.gl_inv_versions(period_id, version_number DESC)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_vr_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_versions
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_inventory.gl_inv_version_snapshots
-- =============================================================================
-- Point-in-time snapshots of emissions data for a specific version. Each
-- snapshot captures the emissions breakdown by scope, category, facility,
-- and gas. Enables comparison between versions and historical lookback.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_version_snapshots (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    version_id                  UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_versions(id) ON DELETE CASCADE,
    facility_id                 UUID,
    entity_id                   UUID,
    scope                       VARCHAR(10)     NOT NULL,
    category                    VARCHAR(60)     NOT NULL,
    sub_category                VARCHAR(60),
    gas                         VARCHAR(20)     DEFAULT 'CO2e',
    emissions_tco2e             NUMERIC(14,6)   NOT NULL DEFAULT 0,
    activity_value              NUMERIC(18,6),
    activity_unit               VARCHAR(50),
    emission_factor_value       NUMERIC(18,10),
    emission_factor_unit        VARCHAR(50),
    emission_factor_source      VARCHAR(100),
    methodology_tier            VARCHAR(20),
    data_quality_indicator      VARCHAR(30),
    is_estimated                BOOLEAN         NOT NULL DEFAULT false,
    snapshot_data               JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_vs_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3', 'BIOGENIC')
    ),
    CONSTRAINT chk_p044_vs_gas CHECK (
        gas IS NULL OR gas IN (
            'CO2', 'CH4', 'N2O', 'HFCs', 'PFCs', 'SF6', 'NF3', 'CO2e', 'BIOGENIC_CO2'
        )
    ),
    CONSTRAINT chk_p044_vs_emissions CHECK (
        emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p044_vs_tier CHECK (
        methodology_tier IS NULL OR methodology_tier IN (
            'TIER_1', 'TIER_2', 'TIER_3', 'DIRECT_MEASUREMENT', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p044_vs_quality CHECK (
        data_quality_indicator IS NULL OR data_quality_indicator IN (
            'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW', 'NOT_ASSESSED'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_vs_tenant          ON ghg_inventory.gl_inv_version_snapshots(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_vs_version         ON ghg_inventory.gl_inv_version_snapshots(version_id);
CREATE INDEX IF NOT EXISTS idx_p044_vs_facility        ON ghg_inventory.gl_inv_version_snapshots(facility_id);
CREATE INDEX IF NOT EXISTS idx_p044_vs_entity          ON ghg_inventory.gl_inv_version_snapshots(entity_id);
CREATE INDEX IF NOT EXISTS idx_p044_vs_scope           ON ghg_inventory.gl_inv_version_snapshots(scope);
CREATE INDEX IF NOT EXISTS idx_p044_vs_category        ON ghg_inventory.gl_inv_version_snapshots(category);
CREATE INDEX IF NOT EXISTS idx_p044_vs_gas             ON ghg_inventory.gl_inv_version_snapshots(gas);
CREATE INDEX IF NOT EXISTS idx_p044_vs_created         ON ghg_inventory.gl_inv_version_snapshots(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_p044_vs_snapshot        ON ghg_inventory.gl_inv_version_snapshots USING GIN(snapshot_data);

-- Composite: version + scope aggregation
CREATE INDEX IF NOT EXISTS idx_p044_vs_version_scope   ON ghg_inventory.gl_inv_version_snapshots(version_id, scope, category);

-- =============================================================================
-- Table 3: ghg_inventory.gl_inv_field_changes
-- =============================================================================
-- Granular field-level change log for inventory data. Records every
-- modification to a data value including the old value, new value, who
-- changed it, and why. Provides the most detailed audit trail layer.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_field_changes (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    version_id                  UUID            REFERENCES ghg_inventory.gl_inv_versions(id) ON DELETE SET NULL,
    entity_type                 VARCHAR(50)     NOT NULL,
    entity_id                   UUID            NOT NULL,
    field_name                  VARCHAR(100)    NOT NULL,
    old_value                   TEXT,
    new_value                   TEXT,
    old_numeric_value           NUMERIC(18,6),
    new_numeric_value           NUMERIC(18,6),
    change_reason               VARCHAR(50),
    change_description          TEXT,
    changed_by_user_id          UUID,
    changed_by_name             VARCHAR(255),
    change_request_id           UUID,
    is_automated                BOOLEAN         NOT NULL DEFAULT false,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_fc_entity_type CHECK (
        entity_type IN (
            'INVENTORY_PERIOD', 'SUBMISSION', 'EMISSION_RECORD', 'EMISSION_FACTOR',
            'BOUNDARY', 'FACILITY', 'SOURCE_CATEGORY', 'METHODOLOGY', 'ASSUMPTION'
        )
    ),
    CONSTRAINT chk_p044_fc_reason CHECK (
        change_reason IS NULL OR change_reason IN (
            'DATA_CORRECTION', 'METHODOLOGY_UPDATE', 'FACTOR_UPDATE',
            'BOUNDARY_CHANGE', 'REVIEW_FEEDBACK', 'QA_QC_FINDING',
            'RESTATEMENT', 'SYSTEM_RECALCULATION', 'USER_EDIT', 'IMPORT_UPDATE'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_fc_tenant          ON ghg_inventory.gl_inv_field_changes(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_fc_period          ON ghg_inventory.gl_inv_field_changes(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_fc_version         ON ghg_inventory.gl_inv_field_changes(version_id);
CREATE INDEX IF NOT EXISTS idx_p044_fc_entity_type     ON ghg_inventory.gl_inv_field_changes(entity_type);
CREATE INDEX IF NOT EXISTS idx_p044_fc_entity          ON ghg_inventory.gl_inv_field_changes(entity_id);
CREATE INDEX IF NOT EXISTS idx_p044_fc_field           ON ghg_inventory.gl_inv_field_changes(field_name);
CREATE INDEX IF NOT EXISTS idx_p044_fc_reason          ON ghg_inventory.gl_inv_field_changes(change_reason);
CREATE INDEX IF NOT EXISTS idx_p044_fc_changed_by      ON ghg_inventory.gl_inv_field_changes(changed_by_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_fc_change_req      ON ghg_inventory.gl_inv_field_changes(change_request_id);
CREATE INDEX IF NOT EXISTS idx_p044_fc_created         ON ghg_inventory.gl_inv_field_changes(created_at DESC);

-- Composite: entity + chronological changes
CREATE INDEX IF NOT EXISTS idx_p044_fc_entity_chrono   ON ghg_inventory.gl_inv_field_changes(entity_id, entity_type, created_at DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_inventory.gl_inv_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_version_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_field_changes ENABLE ROW LEVEL SECURITY;

CREATE POLICY p044_vr_tenant_isolation
    ON ghg_inventory.gl_inv_versions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_vr_service_bypass
    ON ghg_inventory.gl_inv_versions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_vs_tenant_isolation
    ON ghg_inventory.gl_inv_version_snapshots
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_vs_service_bypass
    ON ghg_inventory.gl_inv_version_snapshots
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_fc_tenant_isolation
    ON ghg_inventory.gl_inv_field_changes
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_fc_service_bypass
    ON ghg_inventory.gl_inv_field_changes
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_versions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_version_snapshots TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_field_changes TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_inventory.gl_inv_versions IS
    'Versioned releases of inventory periods capturing the complete state at approval, restatement, or publication milestones.';
COMMENT ON TABLE ghg_inventory.gl_inv_version_snapshots IS
    'Point-in-time emission snapshots by scope, category, facility, and gas for version comparison and historical lookback.';
COMMENT ON TABLE ghg_inventory.gl_inv_field_changes IS
    'Granular field-level change log recording every data modification with old/new values and audit context.';

COMMENT ON COLUMN ghg_inventory.gl_inv_versions.version_type IS 'Type of version: STANDARD, RESTATEMENT, CORRECTION, INTERIM, FINAL, PUBLISHED.';
COMMENT ON COLUMN ghg_inventory.gl_inv_versions.trigger_event IS 'What triggered version creation: MANUAL, APPROVAL, RESTATEMENT, CHANGE_REQUEST, PUBLICATION, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_versions.is_current IS 'Whether this is the current active version of the inventory period.';
COMMENT ON COLUMN ghg_inventory.gl_inv_versions.delta_from_previous_tco2e IS 'Absolute change in total emissions compared to the previous version.';
COMMENT ON COLUMN ghg_inventory.gl_inv_version_snapshots.scope IS 'Emission scope: SCOPE_1, SCOPE_2_LOCATION, SCOPE_2_MARKET, SCOPE_3, BIOGENIC.';
COMMENT ON COLUMN ghg_inventory.gl_inv_field_changes.entity_type IS 'Type of entity that was modified: INVENTORY_PERIOD, SUBMISSION, EMISSION_RECORD, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_field_changes.change_reason IS 'Reason for the change: DATA_CORRECTION, METHODOLOGY_UPDATE, REVIEW_FEEDBACK, etc.';
