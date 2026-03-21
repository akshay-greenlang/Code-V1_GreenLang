-- =============================================================================
-- V252: PACK-033 Quick Wins Identifier - Utility Rebates
-- =============================================================================
-- Pack:         PACK-033 (Quick Wins Identifier Pack)
-- Migration:    007 of 010
-- Date:         March 2026
--
-- Creates utility rebate program tracking tables for matching quick-win
-- actions to available incentive programs, tracking application status,
-- and capturing approved rebate amounts.
--
-- Tables (2):
--   1. pack033_quick_wins.rebate_programs
--   2. pack033_quick_wins.rebate_applications
--
-- Previous: V251__pack033_quick_wins_006.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack033_quick_wins.rebate_programs
-- =============================================================================
-- Utility and government rebate/incentive programs with measure categories,
-- rebate amounts, stacking rules, and application deadlines.

CREATE TABLE pack033_quick_wins.rebate_programs (
    program_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    utility_name            VARCHAR(255)    NOT NULL,
    utility_region          VARCHAR(100)    NOT NULL,
    program_name            VARCHAR(500)    NOT NULL,
    program_type            VARCHAR(50)     NOT NULL,
    measure_category        VARCHAR(100)    NOT NULL,
    rebate_amount           NUMERIC(12,2)   NOT NULL,
    rebate_unit             VARCHAR(30)     NOT NULL,
    max_rebate              NUMERIC(14,2),
    requirements            TEXT,
    application_deadline    DATE,
    stacking_allowed        BOOLEAN         DEFAULT FALSE,
    active                  BOOLEAN         DEFAULT TRUE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_rp_program_type CHECK (
        program_type IN ('PRESCRIPTIVE', 'CUSTOM', 'PERFORMANCE_BASED', 'MIDSTREAM',
                          'UPSTREAM', 'DIRECT_INSTALL', 'TAX_CREDIT', 'GRANT', 'LOAN', 'OTHER')
    ),
    CONSTRAINT chk_p033_rp_measure_category CHECK (
        measure_category IN ('LIGHTING', 'HVAC', 'CONTROLS', 'ENVELOPE', 'PLUG_LOADS',
                              'MOTORS', 'COMPRESSED_AIR', 'STEAM', 'WATER_HEATING',
                              'BEHAVIORAL', 'PROCESS', 'RENEWABLE', 'WHOLE_BUILDING', 'OTHER')
    ),
    CONSTRAINT chk_p033_rp_rebate_amount CHECK (
        rebate_amount >= 0
    ),
    CONSTRAINT chk_p033_rp_rebate_unit CHECK (
        rebate_unit IN ('$/kWh', '$/kW', '$/unit', '$/project', '$/therm',
                          '$/ton_CO2e', 'EUR/kWh', 'EUR/unit', 'GBP/kWh', 'GBP/unit',
                          'PERCENT', 'FLAT', 'OTHER')
    ),
    CONSTRAINT chk_p033_rp_max_rebate CHECK (
        max_rebate IS NULL OR max_rebate >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_rp_utility       ON pack033_quick_wins.rebate_programs(utility_name);
CREATE INDEX idx_p033_rp_region        ON pack033_quick_wins.rebate_programs(utility_region);
CREATE INDEX idx_p033_rp_program_type  ON pack033_quick_wins.rebate_programs(program_type);
CREATE INDEX idx_p033_rp_measure       ON pack033_quick_wins.rebate_programs(measure_category);
CREATE INDEX idx_p033_rp_active        ON pack033_quick_wins.rebate_programs(active);
CREATE INDEX idx_p033_rp_deadline      ON pack033_quick_wins.rebate_programs(application_deadline);
CREATE INDEX idx_p033_rp_metadata      ON pack033_quick_wins.rebate_programs USING GIN(metadata);
CREATE INDEX idx_p033_rp_created       ON pack033_quick_wins.rebate_programs(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_rp_updated
    BEFORE UPDATE ON pack033_quick_wins.rebate_programs
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack033_quick_wins.rebate_applications
-- =============================================================================
-- Individual rebate applications linking scans/actions to rebate programs,
-- tracking application lifecycle from submission to approval/rejection.

CREATE TABLE pack033_quick_wins.rebate_applications (
    application_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    scan_id                 UUID            REFERENCES pack033_quick_wins.quick_wins_scans(scan_id) ON DELETE SET NULL,
    program_id              UUID            NOT NULL REFERENCES pack033_quick_wins.rebate_programs(program_id) ON DELETE CASCADE,
    action_id               UUID,
    applied_date            DATE            NOT NULL DEFAULT CURRENT_DATE,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    rebate_amount_requested NUMERIC(14,2)   NOT NULL,
    rebate_amount_approved  NUMERIC(14,2),
    approval_date           DATE,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_ra_status CHECK (
        status IN ('DRAFT', 'SUBMITTED', 'UNDER_REVIEW', 'APPROVED', 'REJECTED',
                    'PAID', 'EXPIRED', 'WITHDRAWN')
    ),
    CONSTRAINT chk_p033_ra_requested CHECK (
        rebate_amount_requested >= 0
    ),
    CONSTRAINT chk_p033_ra_approved CHECK (
        rebate_amount_approved IS NULL OR rebate_amount_approved >= 0
    ),
    CONSTRAINT chk_p033_ra_approval_date CHECK (
        approval_date IS NULL OR approval_date >= applied_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_ra_tenant        ON pack033_quick_wins.rebate_applications(tenant_id);
CREATE INDEX idx_p033_ra_scan          ON pack033_quick_wins.rebate_applications(scan_id);
CREATE INDEX idx_p033_ra_program       ON pack033_quick_wins.rebate_applications(program_id);
CREATE INDEX idx_p033_ra_action        ON pack033_quick_wins.rebate_applications(action_id);
CREATE INDEX idx_p033_ra_status        ON pack033_quick_wins.rebate_applications(status);
CREATE INDEX idx_p033_ra_applied       ON pack033_quick_wins.rebate_applications(applied_date DESC);
CREATE INDEX idx_p033_ra_approval      ON pack033_quick_wins.rebate_applications(approval_date DESC);
CREATE INDEX idx_p033_ra_created       ON pack033_quick_wins.rebate_applications(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_ra_updated
    BEFORE UPDATE ON pack033_quick_wins.rebate_applications
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack033_quick_wins.rebate_programs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack033_quick_wins.rebate_applications ENABLE ROW LEVEL SECURITY;

-- Rebate programs are shared reference data -- read access for all
CREATE POLICY p033_rp_read_all
    ON pack033_quick_wins.rebate_programs
    FOR SELECT
    USING (TRUE);
CREATE POLICY p033_rp_service_bypass
    ON pack033_quick_wins.rebate_programs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p033_ra_tenant_isolation
    ON pack033_quick_wins.rebate_applications
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p033_ra_service_bypass
    ON pack033_quick_wins.rebate_applications
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT ON pack033_quick_wins.rebate_programs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.rebate_programs TO greenlang_service;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.rebate_applications TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack033_quick_wins.rebate_programs IS
    'Utility and government rebate/incentive programs with measure categories, rebate amounts, and stacking rules.';

COMMENT ON TABLE pack033_quick_wins.rebate_applications IS
    'Individual rebate applications linking scans/actions to rebate programs with application lifecycle tracking.';

COMMENT ON COLUMN pack033_quick_wins.rebate_programs.program_type IS
    'Type of rebate program: PRESCRIPTIVE (fixed per unit), CUSTOM (calculated), PERFORMANCE_BASED (M&V), etc.';
COMMENT ON COLUMN pack033_quick_wins.rebate_programs.rebate_unit IS
    'Unit for the rebate amount (e.g., $/kWh saved, $/unit installed, PERCENT of cost).';
COMMENT ON COLUMN pack033_quick_wins.rebate_programs.stacking_allowed IS
    'Whether this rebate can be combined with other incentive programs.';
COMMENT ON COLUMN pack033_quick_wins.rebate_programs.max_rebate IS
    'Maximum rebate cap per application regardless of calculated amount.';
COMMENT ON COLUMN pack033_quick_wins.rebate_applications.status IS
    'Application lifecycle status: DRAFT > SUBMITTED > UNDER_REVIEW > APPROVED/REJECTED > PAID.';
COMMENT ON COLUMN pack033_quick_wins.rebate_applications.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
