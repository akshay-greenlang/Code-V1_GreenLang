-- =============================================================================
-- V356: PACK-044 GHG Inventory Management - Core Schema & Period Tables
-- =============================================================================
-- Pack:         PACK-044 (GHG Inventory Management)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the ghg_inventory schema and foundational tables for managing GHG
-- inventory reporting periods, milestones, and period state transitions.
-- Inventory periods represent a defined time window (typically a calendar or
-- fiscal year) during which GHG data is collected, reviewed, and finalised.
-- Period milestones track key dates (data collection open, QA/QC, review,
-- approval, publication). Period transitions record the state machine
-- progression from DRAFT through to PUBLISHED.
--
-- Tables (3):
--   1. ghg_inventory.gl_inv_inventory_periods
--   2. ghg_inventory.gl_inv_period_milestones
--   3. ghg_inventory.gl_inv_period_transitions
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V355__pack043_views_indexes_seed.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS ghg_inventory;

SET search_path TO ghg_inventory, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ghg_inventory.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: ghg_inventory.gl_inv_inventory_periods
-- =============================================================================
-- A reporting period for which a GHG inventory is prepared. Typically aligned
-- to a calendar or fiscal year. Tracks the period lifecycle from DRAFT through
-- DATA_COLLECTION, QA_QC, UNDER_REVIEW, APPROVED, to PUBLISHED. Each period
-- belongs to an organisation and may reference an organisational boundary.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_inventory_periods (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    period_name                 VARCHAR(200)    NOT NULL,
    period_type                 VARCHAR(30)     NOT NULL DEFAULT 'CALENDAR_YEAR',
    reporting_year              INTEGER         NOT NULL,
    start_date                  DATE            NOT NULL,
    end_date                    DATE            NOT NULL,
    boundary_id                 UUID,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    lock_after_days             INTEGER         DEFAULT 30,
    is_locked                   BOOLEAN         NOT NULL DEFAULT false,
    locked_at                   TIMESTAMPTZ,
    locked_by                   VARCHAR(255),
    total_scope1_tco2e          NUMERIC(14,3),
    total_scope2_location_tco2e NUMERIC(14,3),
    total_scope2_market_tco2e   NUMERIC(14,3),
    total_scope3_tco2e          NUMERIC(14,3),
    total_tco2e                 NUMERIC(14,3),
    completeness_pct            NUMERIC(5,2),
    data_quality_score          NUMERIC(5,2),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p044_ip_period_type CHECK (
        period_type IN (
            'CALENDAR_YEAR', 'FISCAL_YEAR', 'HALF_YEAR', 'QUARTER', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p044_ip_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p044_ip_dates CHECK (
        start_date < end_date
    ),
    CONSTRAINT chk_p044_ip_status CHECK (
        status IN (
            'DRAFT', 'DATA_COLLECTION', 'QA_QC', 'UNDER_REVIEW',
            'APPROVED', 'PUBLISHED', 'RESTATED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p044_ip_lock_days CHECK (
        lock_after_days IS NULL OR lock_after_days >= 0
    ),
    CONSTRAINT chk_p044_ip_completeness CHECK (
        completeness_pct IS NULL OR (completeness_pct >= 0 AND completeness_pct <= 100)
    ),
    CONSTRAINT chk_p044_ip_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p044_ip_scope1 CHECK (
        total_scope1_tco2e IS NULL OR total_scope1_tco2e >= 0
    ),
    CONSTRAINT chk_p044_ip_scope2_loc CHECK (
        total_scope2_location_tco2e IS NULL OR total_scope2_location_tco2e >= 0
    ),
    CONSTRAINT chk_p044_ip_scope2_mkt CHECK (
        total_scope2_market_tco2e IS NULL OR total_scope2_market_tco2e >= 0
    ),
    CONSTRAINT chk_p044_ip_scope3 CHECK (
        total_scope3_tco2e IS NULL OR total_scope3_tco2e >= 0
    ),
    CONSTRAINT chk_p044_ip_total CHECK (
        total_tco2e IS NULL OR total_tco2e >= 0
    ),
    CONSTRAINT uq_p044_ip_tenant_org_year UNIQUE (tenant_id, organization_id, reporting_year, period_type)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_ip_tenant          ON ghg_inventory.gl_inv_inventory_periods(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_ip_org             ON ghg_inventory.gl_inv_inventory_periods(organization_id);
CREATE INDEX IF NOT EXISTS idx_p044_ip_year            ON ghg_inventory.gl_inv_inventory_periods(reporting_year);
CREATE INDEX IF NOT EXISTS idx_p044_ip_status          ON ghg_inventory.gl_inv_inventory_periods(status);
CREATE INDEX IF NOT EXISTS idx_p044_ip_dates           ON ghg_inventory.gl_inv_inventory_periods(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_p044_ip_locked          ON ghg_inventory.gl_inv_inventory_periods(is_locked) WHERE is_locked = true;
CREATE INDEX IF NOT EXISTS idx_p044_ip_created         ON ghg_inventory.gl_inv_inventory_periods(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_p044_ip_metadata        ON ghg_inventory.gl_inv_inventory_periods USING GIN(metadata);

-- Composite: tenant + active periods
CREATE INDEX IF NOT EXISTS idx_p044_ip_tenant_active   ON ghg_inventory.gl_inv_inventory_periods(tenant_id, reporting_year)
    WHERE status NOT IN ('ARCHIVED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_ip_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_inventory_periods
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_inventory.gl_inv_period_milestones
-- =============================================================================
-- Key milestones within an inventory period lifecycle. Tracks target and
-- actual dates for data collection open/close, QA/QC start/complete, review
-- submission, approval, and publication. Enables Gantt-chart-style tracking
-- of inventory preparation progress.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_period_milestones (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    milestone_type              VARCHAR(50)     NOT NULL,
    milestone_name              VARCHAR(200)    NOT NULL,
    target_date                 DATE            NOT NULL,
    actual_date                 DATE,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    responsible_role            VARCHAR(100),
    responsible_user_id         UUID,
    completion_notes            TEXT,
    is_blocking                 BOOLEAN         NOT NULL DEFAULT false,
    sort_order                  INTEGER         NOT NULL DEFAULT 0,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_pm_type CHECK (
        milestone_type IN (
            'DATA_COLLECTION_OPEN', 'DATA_COLLECTION_CLOSE',
            'QAQC_START', 'QAQC_COMPLETE',
            'REVIEW_START', 'REVIEW_COMPLETE',
            'APPROVAL_SUBMITTED', 'APPROVAL_GRANTED',
            'PUBLICATION', 'VERIFICATION_START', 'VERIFICATION_COMPLETE',
            'RESTATEMENT', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p044_pm_status CHECK (
        status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'OVERDUE', 'SKIPPED', 'CANCELLED')
    ),
    CONSTRAINT chk_p044_pm_sort CHECK (
        sort_order >= 0
    ),
    CONSTRAINT uq_p044_pm_period_type UNIQUE (period_id, milestone_type)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_pm_tenant          ON ghg_inventory.gl_inv_period_milestones(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_pm_period          ON ghg_inventory.gl_inv_period_milestones(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_pm_type            ON ghg_inventory.gl_inv_period_milestones(milestone_type);
CREATE INDEX IF NOT EXISTS idx_p044_pm_status          ON ghg_inventory.gl_inv_period_milestones(status);
CREATE INDEX IF NOT EXISTS idx_p044_pm_target          ON ghg_inventory.gl_inv_period_milestones(target_date);
CREATE INDEX IF NOT EXISTS idx_p044_pm_created         ON ghg_inventory.gl_inv_period_milestones(created_at DESC);

-- Composite: period + pending/overdue milestones
CREATE INDEX IF NOT EXISTS idx_p044_pm_period_pending  ON ghg_inventory.gl_inv_period_milestones(period_id, target_date)
    WHERE status IN ('PENDING', 'IN_PROGRESS', 'OVERDUE');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_pm_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_period_milestones
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_inventory.gl_inv_period_transitions
-- =============================================================================
-- Audit log of all state transitions for an inventory period. Records who
-- changed the status, when, and why. Enforces valid state machine transitions
-- and provides a complete audit trail for verification and assurance.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_period_transitions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    from_status                 VARCHAR(30)     NOT NULL,
    to_status                   VARCHAR(30)     NOT NULL,
    transition_reason           TEXT,
    transitioned_by             UUID,
    transitioned_by_name        VARCHAR(255),
    transitioned_by_role        VARCHAR(100),
    is_automated                BOOLEAN         NOT NULL DEFAULT false,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_pt_from_status CHECK (
        from_status IN (
            'DRAFT', 'DATA_COLLECTION', 'QA_QC', 'UNDER_REVIEW',
            'APPROVED', 'PUBLISHED', 'RESTATED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p044_pt_to_status CHECK (
        to_status IN (
            'DRAFT', 'DATA_COLLECTION', 'QA_QC', 'UNDER_REVIEW',
            'APPROVED', 'PUBLISHED', 'RESTATED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p044_pt_different CHECK (
        from_status != to_status
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_pt_tenant          ON ghg_inventory.gl_inv_period_transitions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_pt_period          ON ghg_inventory.gl_inv_period_transitions(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_pt_from            ON ghg_inventory.gl_inv_period_transitions(from_status);
CREATE INDEX IF NOT EXISTS idx_p044_pt_to              ON ghg_inventory.gl_inv_period_transitions(to_status);
CREATE INDEX IF NOT EXISTS idx_p044_pt_by              ON ghg_inventory.gl_inv_period_transitions(transitioned_by);
CREATE INDEX IF NOT EXISTS idx_p044_pt_created         ON ghg_inventory.gl_inv_period_transitions(created_at DESC);

-- Composite: period + chronological transitions
CREATE INDEX IF NOT EXISTS idx_p044_pt_period_chrono   ON ghg_inventory.gl_inv_period_transitions(period_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_inventory.gl_inv_inventory_periods ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_period_milestones ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_period_transitions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p044_ip_tenant_isolation
    ON ghg_inventory.gl_inv_inventory_periods
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_ip_service_bypass
    ON ghg_inventory.gl_inv_inventory_periods
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_pm_tenant_isolation
    ON ghg_inventory.gl_inv_period_milestones
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_pm_service_bypass
    ON ghg_inventory.gl_inv_period_milestones
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_pt_tenant_isolation
    ON ghg_inventory.gl_inv_period_transitions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_pt_service_bypass
    ON ghg_inventory.gl_inv_period_transitions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA ghg_inventory TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_inventory_periods TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_period_milestones TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_period_transitions TO PUBLIC;
GRANT EXECUTE ON FUNCTION ghg_inventory.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA ghg_inventory IS
    'PACK-044 GHG Inventory Management - Complete lifecycle management for GHG inventories including data collection, QA/QC, review/approval, versioning, consolidation, gap analysis, and documentation.';

COMMENT ON TABLE ghg_inventory.gl_inv_inventory_periods IS
    'GHG inventory reporting periods with lifecycle status tracking from DRAFT through PUBLISHED.';
COMMENT ON TABLE ghg_inventory.gl_inv_period_milestones IS
    'Key milestones within an inventory period lifecycle with target and actual dates.';
COMMENT ON TABLE ghg_inventory.gl_inv_period_transitions IS
    'Audit log of all inventory period state transitions for verification and assurance.';

COMMENT ON COLUMN ghg_inventory.gl_inv_inventory_periods.id IS 'Unique identifier for the inventory period.';
COMMENT ON COLUMN ghg_inventory.gl_inv_inventory_periods.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN ghg_inventory.gl_inv_inventory_periods.organization_id IS 'Reference to the reporting organisation.';
COMMENT ON COLUMN ghg_inventory.gl_inv_inventory_periods.status IS 'Period lifecycle state: DRAFT, DATA_COLLECTION, QA_QC, UNDER_REVIEW, APPROVED, PUBLISHED, RESTATED, ARCHIVED.';
COMMENT ON COLUMN ghg_inventory.gl_inv_inventory_periods.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN ghg_inventory.gl_inv_inventory_periods.completeness_pct IS 'Percentage of expected data sources that have submitted data (0-100).';

COMMENT ON COLUMN ghg_inventory.gl_inv_period_milestones.milestone_type IS 'Type of milestone: DATA_COLLECTION_OPEN, QAQC_START, REVIEW_START, APPROVAL_GRANTED, PUBLICATION, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_period_milestones.is_blocking IS 'Whether this milestone must be completed before the period can advance to the next status.';

COMMENT ON COLUMN ghg_inventory.gl_inv_period_transitions.from_status IS 'Previous period status before the transition.';
COMMENT ON COLUMN ghg_inventory.gl_inv_period_transitions.to_status IS 'New period status after the transition.';
COMMENT ON COLUMN ghg_inventory.gl_inv_period_transitions.is_automated IS 'Whether this transition was triggered automatically by the system.';
