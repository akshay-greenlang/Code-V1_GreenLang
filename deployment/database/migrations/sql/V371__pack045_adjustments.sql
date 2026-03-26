-- =============================================================================
-- V371: PACK-045 Base Year Management Pack - Adjustment Packages
-- =============================================================================
-- Pack:         PACK-045 (Base Year Management Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates adjustment package tables that bundle the specific emission line
-- item changes applied during a base year recalculation. An adjustment
-- package contains one or more adjustment lines, each modifying a specific
-- scope/category with original, adjustment, and adjusted tCO2e values.
-- Pro-rata factors handle partial-year acquisitions/divestitures.
--
-- Tables (2):
--   1. ghg_base_year.gl_by_adjustment_packages
--   2. ghg_base_year.gl_by_adjustment_lines
--
-- Also includes: indexes, RLS, comments.
-- Previous: V370__pack045_significance.sql
-- =============================================================================

SET search_path TO ghg_base_year, public;

-- =============================================================================
-- Table 1: ghg_base_year.gl_by_adjustment_packages
-- =============================================================================
-- A grouping of adjustment lines that together represent a single base year
-- recalculation event. The package tracks approval workflow, total net
-- adjustment, and links to the trigger(s) that caused the recalculation.

CREATE TABLE ghg_base_year.gl_by_adjustment_packages (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    base_year_id                UUID            NOT NULL REFERENCES ghg_base_year.gl_by_base_years(id) ON DELETE CASCADE,
    package_name                VARCHAR(255)    NOT NULL,
    package_type                VARCHAR(30)     NOT NULL DEFAULT 'RECALCULATION',
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    trigger_ids                 UUID[]          NOT NULL DEFAULT '{}',
    assessment_ids              UUID[]          DEFAULT '{}',
    original_total_tco2e        NUMERIC(14,3)   NOT NULL,
    total_adjustment_tco2e      NUMERIC(14,3)   NOT NULL DEFAULT 0,
    adjusted_total_tco2e        NUMERIC(14,3)   GENERATED ALWAYS AS (original_total_tco2e + total_adjustment_tco2e) STORED,
    adjustment_pct              NUMERIC(8,4),
    effective_date              DATE            NOT NULL DEFAULT CURRENT_DATE,
    description                 TEXT,
    justification               TEXT            NOT NULL,
    methodology_notes           TEXT,
    created_by                  UUID,
    created_by_name             VARCHAR(255),
    approved_by                 UUID,
    approved_by_name            VARCHAR(255),
    approved_date               TIMESTAMPTZ,
    rejection_reason            TEXT,
    verification_status         VARCHAR(30)     DEFAULT 'UNVERIFIED',
    verified_by                 VARCHAR(255),
    verified_date               DATE,
    evidence_refs               TEXT[],
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_ap_type CHECK (
        package_type IN (
            'RECALCULATION', 'ERROR_CORRECTION', 'METHODOLOGY_UPDATE',
            'STRUCTURAL_CHANGE', 'REGULATORY_MANDATE', 'INITIAL_SETUP'
        )
    ),
    CONSTRAINT chk_p045_ap_status CHECK (
        status IN (
            'DRAFT', 'PENDING_REVIEW', 'UNDER_REVIEW', 'APPROVED', 'APPLIED',
            'REJECTED', 'SUPERSEDED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p045_ap_original CHECK (
        original_total_tco2e >= 0
    ),
    CONSTRAINT chk_p045_ap_verification CHECK (
        verification_status IN ('UNVERIFIED', 'LIMITED_ASSURANCE', 'REASONABLE_ASSURANCE')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_ap_tenant          ON ghg_base_year.gl_by_adjustment_packages(tenant_id);
CREATE INDEX idx_p045_ap_base_year       ON ghg_base_year.gl_by_adjustment_packages(base_year_id);
CREATE INDEX idx_p045_ap_type            ON ghg_base_year.gl_by_adjustment_packages(package_type);
CREATE INDEX idx_p045_ap_status          ON ghg_base_year.gl_by_adjustment_packages(status);
CREATE INDEX idx_p045_ap_effective       ON ghg_base_year.gl_by_adjustment_packages(effective_date);
CREATE INDEX idx_p045_ap_created_by      ON ghg_base_year.gl_by_adjustment_packages(created_by);
CREATE INDEX idx_p045_ap_approved_by     ON ghg_base_year.gl_by_adjustment_packages(approved_by);
CREATE INDEX idx_p045_ap_verification    ON ghg_base_year.gl_by_adjustment_packages(verification_status);
CREATE INDEX idx_p045_ap_provenance      ON ghg_base_year.gl_by_adjustment_packages(provenance_hash);
CREATE INDEX idx_p045_ap_created         ON ghg_base_year.gl_by_adjustment_packages(created_at DESC);
CREATE INDEX idx_p045_ap_trigger_ids     ON ghg_base_year.gl_by_adjustment_packages USING GIN(trigger_ids);
CREATE INDEX idx_p045_ap_metadata        ON ghg_base_year.gl_by_adjustment_packages USING GIN(metadata);

-- Composite: base_year + applied packages
CREATE INDEX idx_p045_ap_by_applied      ON ghg_base_year.gl_by_adjustment_packages(base_year_id)
    WHERE status = 'APPLIED';

-- Composite: pending review for workflow dashboard
CREATE INDEX idx_p045_ap_pending         ON ghg_base_year.gl_by_adjustment_packages(status, created_at DESC)
    WHERE status IN ('PENDING_REVIEW', 'UNDER_REVIEW');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_ap_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_adjustment_packages
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_base_year.gl_by_adjustment_lines
-- =============================================================================
-- Individual line-item adjustments within a package. Each line modifies a
-- specific scope/category combination, recording original, adjustment, and
-- adjusted tCO2e values. Pro-rata factors handle partial-year events (e.g.,
-- acquisition completed mid-year applies a 6/12 factor).

CREATE TABLE ghg_base_year.gl_by_adjustment_lines (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    package_id                  UUID            NOT NULL REFERENCES ghg_base_year.gl_by_adjustment_packages(id) ON DELETE CASCADE,
    line_number                 INTEGER         NOT NULL,
    adjustment_type             VARCHAR(30)     NOT NULL,
    scope                       VARCHAR(10)     NOT NULL,
    category                    VARCHAR(60)     NOT NULL,
    sub_category                VARCHAR(60),
    facility_id                 UUID,
    facility_name               VARCHAR(500),
    description                 TEXT            NOT NULL,
    original_tco2e              NUMERIC(14,3)   NOT NULL,
    adjustment_tco2e            NUMERIC(14,3)   NOT NULL,
    adjusted_tco2e              NUMERIC(14,3)   GENERATED ALWAYS AS (original_tco2e + adjustment_tco2e) STORED,
    pro_rata_factor             NUMERIC(6,4)    DEFAULT 1.0000,
    pro_rata_months             INTEGER,
    original_activity_data      NUMERIC(18,6),
    adjusted_activity_data      NUMERIC(18,6),
    activity_data_unit          VARCHAR(50),
    original_emission_factor    NUMERIC(18,10),
    adjusted_emission_factor    NUMERIC(18,10),
    emission_factor_unit        VARCHAR(100),
    calculation_method          TEXT,
    evidence_ref                TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_al_type CHECK (
        adjustment_type IN (
            'ADD_SOURCE', 'REMOVE_SOURCE', 'MODIFY_ACTIVITY_DATA',
            'MODIFY_EMISSION_FACTOR', 'MODIFY_GWP', 'METHODOLOGY_CHANGE',
            'BOUNDARY_INCLUSION', 'BOUNDARY_EXCLUSION', 'PRO_RATA',
            'ERROR_CORRECTION', 'OTHER'
        )
    ),
    CONSTRAINT chk_p045_al_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3')
    ),
    CONSTRAINT chk_p045_al_original CHECK (
        original_tco2e >= 0
    ),
    CONSTRAINT chk_p045_al_pro_rata CHECK (
        pro_rata_factor > 0 AND pro_rata_factor <= 1.0
    ),
    CONSTRAINT chk_p045_al_pro_rata_months CHECK (
        pro_rata_months IS NULL OR (pro_rata_months >= 1 AND pro_rata_months <= 12)
    ),
    CONSTRAINT chk_p045_al_line_number CHECK (
        line_number >= 1
    ),
    CONSTRAINT uq_p045_al_package_line UNIQUE (package_id, line_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_al_tenant          ON ghg_base_year.gl_by_adjustment_lines(tenant_id);
CREATE INDEX idx_p045_al_package         ON ghg_base_year.gl_by_adjustment_lines(package_id);
CREATE INDEX idx_p045_al_type            ON ghg_base_year.gl_by_adjustment_lines(adjustment_type);
CREATE INDEX idx_p045_al_scope           ON ghg_base_year.gl_by_adjustment_lines(scope);
CREATE INDEX idx_p045_al_category        ON ghg_base_year.gl_by_adjustment_lines(category);
CREATE INDEX idx_p045_al_facility        ON ghg_base_year.gl_by_adjustment_lines(facility_id);
CREATE INDEX idx_p045_al_created         ON ghg_base_year.gl_by_adjustment_lines(created_at DESC);
CREATE INDEX idx_p045_al_metadata        ON ghg_base_year.gl_by_adjustment_lines USING GIN(metadata);

-- Composite: package + scope for scope-level rollup
CREATE INDEX idx_p045_al_package_scope   ON ghg_base_year.gl_by_adjustment_lines(package_id, scope);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_al_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_adjustment_lines
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_base_year.gl_by_adjustment_packages ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_base_year.gl_by_adjustment_lines ENABLE ROW LEVEL SECURITY;

CREATE POLICY p045_ap_tenant_isolation
    ON ghg_base_year.gl_by_adjustment_packages
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_ap_service_bypass
    ON ghg_base_year.gl_by_adjustment_packages
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p045_al_tenant_isolation
    ON ghg_base_year.gl_by_adjustment_lines
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_al_service_bypass
    ON ghg_base_year.gl_by_adjustment_lines
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_adjustment_packages TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_adjustment_lines TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_base_year.gl_by_adjustment_packages IS
    'Bundles of base year adjustment lines representing a single recalculation event with approval workflow and verification status.';
COMMENT ON TABLE ghg_base_year.gl_by_adjustment_lines IS
    'Individual emission line-item adjustments within a package, with original/adjusted values and pro-rata factors for partial-year events.';

COMMENT ON COLUMN ghg_base_year.gl_by_adjustment_packages.adjusted_total_tco2e IS 'Auto-calculated: original_total_tco2e + total_adjustment_tco2e.';
COMMENT ON COLUMN ghg_base_year.gl_by_adjustment_packages.trigger_ids IS 'Array of trigger IDs that caused this recalculation package.';
COMMENT ON COLUMN ghg_base_year.gl_by_adjustment_lines.adjusted_tco2e IS 'Auto-calculated: original_tco2e + adjustment_tco2e.';
COMMENT ON COLUMN ghg_base_year.gl_by_adjustment_lines.pro_rata_factor IS 'Fraction of year (0-1) for partial-year events, e.g., 0.5 for a mid-year acquisition.';
COMMENT ON COLUMN ghg_base_year.gl_by_adjustment_lines.pro_rata_months IS 'Number of months included when pro-rata factor is applied (1-12).';
