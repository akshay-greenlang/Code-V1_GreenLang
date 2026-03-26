-- =============================================================================
-- V362: PACK-044 GHG Inventory Management - Consolidation Tables
-- =============================================================================
-- Pack:         PACK-044 (GHG Inventory Management)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Multi-entity consolidation tables for group-level GHG inventory rollup.
-- Implements GHG Protocol consolidation approaches (equity share, operational
-- control, financial control). Entity hierarchy defines parent-child
-- relationships. Subsidiary submissions capture entity-level emissions.
-- Consolidation runs aggregate subsidiary data into group totals with
-- inter-company elimination and equity share adjustments.
--
-- Tables (3):
--   1. ghg_inventory.gl_inv_entity_hierarchy
--   2. ghg_inventory.gl_inv_subsidiary_submissions
--   3. ghg_inventory.gl_inv_consolidation_runs
--
-- Previous: V361__pack044_versioning.sql
-- =============================================================================

SET search_path TO ghg_inventory, public;

-- =============================================================================
-- Table 1: ghg_inventory.gl_inv_entity_hierarchy
-- =============================================================================
-- Defines the corporate entity hierarchy for consolidation. Maps parent-child
-- relationships between legal entities with ownership percentages. Supports
-- multi-level hierarchies (holding > subsidiary > sub-subsidiary). The
-- consolidation approach determines how emissions flow up the hierarchy.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_entity_hierarchy (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    parent_entity_id            UUID,
    child_entity_id             UUID            NOT NULL,
    child_entity_name           VARCHAR(500)    NOT NULL,
    entity_type                 VARCHAR(30)     NOT NULL DEFAULT 'SUBSIDIARY',
    ownership_pct               DECIMAL(5,2)    NOT NULL DEFAULT 100.00,
    has_operational_control     BOOLEAN         NOT NULL DEFAULT true,
    has_financial_control       BOOLEAN         NOT NULL DEFAULT true,
    consolidation_approach      VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    inclusion_pct               DECIMAL(5,2)    NOT NULL DEFAULT 100.00,
    country                     VARCHAR(3)      NOT NULL,
    sector                      VARCHAR(100),
    hierarchy_level             INTEGER         NOT NULL DEFAULT 1,
    sort_order                  INTEGER         NOT NULL DEFAULT 0,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_eh_entity_type CHECK (
        entity_type IN (
            'HOLDING', 'SUBSIDIARY', 'JOINT_VENTURE', 'ASSOCIATE',
            'PARTNERSHIP', 'FRANCHISE', 'BRANCH', 'DIVISION', 'SPV', 'OTHER'
        )
    ),
    CONSTRAINT chk_p044_eh_ownership CHECK (
        ownership_pct >= 0 AND ownership_pct <= 100
    ),
    CONSTRAINT chk_p044_eh_inclusion CHECK (
        inclusion_pct >= 0 AND inclusion_pct <= 100
    ),
    CONSTRAINT chk_p044_eh_consolidation CHECK (
        consolidation_approach IN (
            'EQUITY_SHARE', 'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL'
        )
    ),
    CONSTRAINT chk_p044_eh_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p044_eh_level CHECK (
        hierarchy_level >= 1 AND hierarchy_level <= 20
    ),
    CONSTRAINT chk_p044_eh_no_self_parent CHECK (
        parent_entity_id IS NULL OR parent_entity_id != child_entity_id
    ),
    CONSTRAINT uq_p044_eh_period_parent_child UNIQUE (period_id, parent_entity_id, child_entity_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_eh_tenant          ON ghg_inventory.gl_inv_entity_hierarchy(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_eh_period          ON ghg_inventory.gl_inv_entity_hierarchy(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_eh_parent          ON ghg_inventory.gl_inv_entity_hierarchy(parent_entity_id);
CREATE INDEX IF NOT EXISTS idx_p044_eh_child           ON ghg_inventory.gl_inv_entity_hierarchy(child_entity_id);
CREATE INDEX IF NOT EXISTS idx_p044_eh_type            ON ghg_inventory.gl_inv_entity_hierarchy(entity_type);
CREATE INDEX IF NOT EXISTS idx_p044_eh_consolidation   ON ghg_inventory.gl_inv_entity_hierarchy(consolidation_approach);
CREATE INDEX IF NOT EXISTS idx_p044_eh_country         ON ghg_inventory.gl_inv_entity_hierarchy(country);
CREATE INDEX IF NOT EXISTS idx_p044_eh_level           ON ghg_inventory.gl_inv_entity_hierarchy(hierarchy_level);
CREATE INDEX IF NOT EXISTS idx_p044_eh_active          ON ghg_inventory.gl_inv_entity_hierarchy(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_p044_eh_created         ON ghg_inventory.gl_inv_entity_hierarchy(created_at DESC);

-- Composite: period + active hierarchy
CREATE INDEX IF NOT EXISTS idx_p044_eh_period_active   ON ghg_inventory.gl_inv_entity_hierarchy(period_id, hierarchy_level)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_eh_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_entity_hierarchy
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_inventory.gl_inv_subsidiary_submissions
-- =============================================================================
-- Entity-level emission submissions within a consolidation period. Each
-- subsidiary submits its emissions by scope and category. Tracks submission
-- status, sign-off by the entity representative, and readiness for
-- consolidation into the group total.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_subsidiary_submissions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    entity_id                   UUID            NOT NULL,
    entity_name                 VARCHAR(500)    NOT NULL,
    hierarchy_node_id           UUID            REFERENCES ghg_inventory.gl_inv_entity_hierarchy(id) ON DELETE SET NULL,
    scope1_tco2e                NUMERIC(14,3)   DEFAULT 0,
    scope2_location_tco2e       NUMERIC(14,3)   DEFAULT 0,
    scope2_market_tco2e         NUMERIC(14,3)   DEFAULT 0,
    scope3_tco2e                NUMERIC(14,3)   DEFAULT 0,
    total_tco2e                 NUMERIC(14,3)   DEFAULT 0,
    equity_adjusted_tco2e       NUMERIC(14,3),
    inter_company_tco2e         NUMERIC(14,3)   DEFAULT 0,
    net_tco2e                   NUMERIC(14,3),
    submission_status           VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    completeness_pct            NUMERIC(5,2),
    data_quality_score          NUMERIC(5,2),
    submitted_at                TIMESTAMPTZ,
    submitted_by_name           VARCHAR(255),
    signed_off                  BOOLEAN         NOT NULL DEFAULT false,
    signed_off_at               TIMESTAMPTZ,
    signed_off_by               VARCHAR(255),
    currency                    VARCHAR(3)      DEFAULT 'USD',
    revenue                     NUMERIC(18,2),
    headcount                   INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_ss_status CHECK (
        submission_status IN (
            'PENDING', 'IN_PROGRESS', 'SUBMITTED', 'UNDER_REVIEW',
            'ACCEPTED', 'RETURNED', 'SIGNED_OFF'
        )
    ),
    CONSTRAINT chk_p044_ss_completeness CHECK (
        completeness_pct IS NULL OR (completeness_pct >= 0 AND completeness_pct <= 100)
    ),
    CONSTRAINT chk_p044_ss_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p044_ss_scope1 CHECK (
        scope1_tco2e IS NULL OR scope1_tco2e >= 0
    ),
    CONSTRAINT chk_p044_ss_scope2_loc CHECK (
        scope2_location_tco2e IS NULL OR scope2_location_tco2e >= 0
    ),
    CONSTRAINT chk_p044_ss_scope2_mkt CHECK (
        scope2_market_tco2e IS NULL OR scope2_market_tco2e >= 0
    ),
    CONSTRAINT chk_p044_ss_headcount CHECK (
        headcount IS NULL OR headcount >= 0
    ),
    CONSTRAINT uq_p044_ss_period_entity UNIQUE (period_id, entity_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_ss_tenant          ON ghg_inventory.gl_inv_subsidiary_submissions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_ss_period          ON ghg_inventory.gl_inv_subsidiary_submissions(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_ss_entity          ON ghg_inventory.gl_inv_subsidiary_submissions(entity_id);
CREATE INDEX IF NOT EXISTS idx_p044_ss_hierarchy       ON ghg_inventory.gl_inv_subsidiary_submissions(hierarchy_node_id);
CREATE INDEX IF NOT EXISTS idx_p044_ss_status          ON ghg_inventory.gl_inv_subsidiary_submissions(submission_status);
CREATE INDEX IF NOT EXISTS idx_p044_ss_signed_off      ON ghg_inventory.gl_inv_subsidiary_submissions(signed_off) WHERE signed_off = true;
CREATE INDEX IF NOT EXISTS idx_p044_ss_created         ON ghg_inventory.gl_inv_subsidiary_submissions(created_at DESC);

-- Composite: period + pending submissions
CREATE INDEX IF NOT EXISTS idx_p044_ss_period_pending  ON ghg_inventory.gl_inv_subsidiary_submissions(period_id, entity_name)
    WHERE submission_status IN ('PENDING', 'IN_PROGRESS', 'SUBMITTED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_ss_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_subsidiary_submissions
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_inventory.gl_inv_consolidation_runs
-- =============================================================================
-- Group-level consolidation runs that aggregate subsidiary submissions into
-- the consolidated group inventory. Applies equity share adjustments,
-- inter-company eliminations, and currency conversions. Produces the
-- official group total for each scope.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_consolidation_runs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    run_name                    VARCHAR(300)    NOT NULL,
    consolidation_approach      VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    started_at                  TIMESTAMPTZ,
    completed_at                TIMESTAMPTZ,
    total_entities              INTEGER         NOT NULL DEFAULT 0,
    entities_included           INTEGER         NOT NULL DEFAULT 0,
    entities_excluded           INTEGER         NOT NULL DEFAULT 0,
    raw_scope1_tco2e            NUMERIC(14,3)   DEFAULT 0,
    raw_scope2_location_tco2e   NUMERIC(14,3)   DEFAULT 0,
    raw_scope2_market_tco2e     NUMERIC(14,3)   DEFAULT 0,
    raw_scope3_tco2e            NUMERIC(14,3)   DEFAULT 0,
    equity_adjustment_tco2e     NUMERIC(14,3)   DEFAULT 0,
    inter_company_elim_tco2e    NUMERIC(14,3)   DEFAULT 0,
    consolidated_scope1_tco2e   NUMERIC(14,3),
    consolidated_scope2_loc_tco2e NUMERIC(14,3),
    consolidated_scope2_mkt_tco2e NUMERIC(14,3),
    consolidated_scope3_tco2e   NUMERIC(14,3),
    consolidated_total_tco2e    NUMERIC(14,3),
    run_by_user_id              UUID,
    run_by_name                 VARCHAR(255),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_cr2_consolidation CHECK (
        consolidation_approach IN (
            'EQUITY_SHARE', 'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL'
        )
    ),
    CONSTRAINT chk_p044_cr2_status CHECK (
        status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')
    ),
    CONSTRAINT chk_p044_cr2_entities CHECK (
        total_entities >= 0 AND entities_included >= 0 AND entities_excluded >= 0
    ),
    CONSTRAINT chk_p044_cr2_times CHECK (
        started_at IS NULL OR completed_at IS NULL OR started_at <= completed_at
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_cr2_tenant         ON ghg_inventory.gl_inv_consolidation_runs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_cr2_period         ON ghg_inventory.gl_inv_consolidation_runs(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_cr2_approach       ON ghg_inventory.gl_inv_consolidation_runs(consolidation_approach);
CREATE INDEX IF NOT EXISTS idx_p044_cr2_status         ON ghg_inventory.gl_inv_consolidation_runs(status);
CREATE INDEX IF NOT EXISTS idx_p044_cr2_created        ON ghg_inventory.gl_inv_consolidation_runs(created_at DESC);

-- Composite: period + completed runs
CREATE INDEX IF NOT EXISTS idx_p044_cr2_period_done    ON ghg_inventory.gl_inv_consolidation_runs(period_id, completed_at DESC)
    WHERE status = 'COMPLETED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_cr2_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_consolidation_runs
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_inventory.gl_inv_entity_hierarchy ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_subsidiary_submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_consolidation_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p044_eh_tenant_isolation
    ON ghg_inventory.gl_inv_entity_hierarchy
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_eh_service_bypass
    ON ghg_inventory.gl_inv_entity_hierarchy
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_ss_tenant_isolation
    ON ghg_inventory.gl_inv_subsidiary_submissions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_ss_service_bypass
    ON ghg_inventory.gl_inv_subsidiary_submissions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_cr2_tenant_isolation
    ON ghg_inventory.gl_inv_consolidation_runs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_cr2_service_bypass
    ON ghg_inventory.gl_inv_consolidation_runs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_entity_hierarchy TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_subsidiary_submissions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_consolidation_runs TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_inventory.gl_inv_entity_hierarchy IS
    'Corporate entity hierarchy for consolidation defining parent-child relationships with ownership and control per GHG Protocol.';
COMMENT ON TABLE ghg_inventory.gl_inv_subsidiary_submissions IS
    'Entity-level emission submissions within a consolidation period with sign-off tracking.';
COMMENT ON TABLE ghg_inventory.gl_inv_consolidation_runs IS
    'Group-level consolidation runs aggregating subsidiary data with equity adjustments and inter-company eliminations.';

COMMENT ON COLUMN ghg_inventory.gl_inv_entity_hierarchy.ownership_pct IS 'Equity ownership percentage (0-100). Used for equity share consolidation approach.';
COMMENT ON COLUMN ghg_inventory.gl_inv_entity_hierarchy.inclusion_pct IS 'Effective inclusion percentage after applying consolidation approach rules.';
COMMENT ON COLUMN ghg_inventory.gl_inv_entity_hierarchy.hierarchy_level IS 'Depth in the hierarchy tree (1=top-level subsidiary, 2=sub-subsidiary, etc.).';
COMMENT ON COLUMN ghg_inventory.gl_inv_subsidiary_submissions.equity_adjusted_tco2e IS 'Total emissions adjusted for equity share percentage.';
COMMENT ON COLUMN ghg_inventory.gl_inv_subsidiary_submissions.inter_company_tco2e IS 'Emissions to be eliminated as inter-company transactions.';
COMMENT ON COLUMN ghg_inventory.gl_inv_consolidation_runs.equity_adjustment_tco2e IS 'Total equity share adjustment applied during consolidation.';
COMMENT ON COLUMN ghg_inventory.gl_inv_consolidation_runs.inter_company_elim_tco2e IS 'Total inter-company emissions eliminated during consolidation.';
