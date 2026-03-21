-- =============================================================================
-- V253: PACK-033 Quick Wins Identifier - Progress Tracking
-- =============================================================================
-- Pack:         PACK-033 (Quick Wins Identifier Pack)
-- Migration:    008 of 010
-- Date:         March 2026
--
-- Creates progress tracking tables for monitoring implementation of quick-win
-- actions and verifying actual savings against estimates using measurement
-- and verification protocols.
--
-- Tables (2):
--   1. pack033_quick_wins.implementation_progress
--   2. pack033_quick_wins.savings_actuals
--
-- Previous: V252__pack033_quick_wins_007.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack033_quick_wins.implementation_progress
-- =============================================================================
-- Tracks implementation status of individual quick-win actions with planned
-- vs. actual dates, completion percentage, and cost tracking.

CREATE TABLE pack033_quick_wins.implementation_progress (
    progress_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    scan_id                 UUID            REFERENCES pack033_quick_wins.quick_wins_scans(scan_id) ON DELETE SET NULL,
    action_id               UUID            NOT NULL,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'NOT_STARTED',
    planned_start_date      DATE,
    planned_end_date        DATE,
    actual_start_date       DATE,
    actual_end_date         DATE,
    completion_pct          NUMERIC(5,2)    NOT NULL DEFAULT 0,
    actual_cost             NUMERIC(14,2),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_ip_status CHECK (
        status IN ('NOT_STARTED', 'PLANNING', 'PROCUREMENT', 'IN_PROGRESS',
                    'COMPLETED', 'VERIFIED', 'ON_HOLD', 'CANCELLED')
    ),
    CONSTRAINT chk_p033_ip_completion CHECK (
        completion_pct >= 0 AND completion_pct <= 100
    ),
    CONSTRAINT chk_p033_ip_actual_cost CHECK (
        actual_cost IS NULL OR actual_cost >= 0
    ),
    CONSTRAINT chk_p033_ip_planned_dates CHECK (
        planned_end_date IS NULL OR planned_start_date IS NULL
        OR planned_end_date >= planned_start_date
    ),
    CONSTRAINT chk_p033_ip_actual_dates CHECK (
        actual_end_date IS NULL OR actual_start_date IS NULL
        OR actual_end_date >= actual_start_date
    ),
    CONSTRAINT chk_p033_ip_status_completion CHECK (
        (status = 'COMPLETED' AND completion_pct = 100)
        OR (status = 'VERIFIED' AND completion_pct = 100)
        OR (status NOT IN ('COMPLETED', 'VERIFIED'))
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_ip_tenant        ON pack033_quick_wins.implementation_progress(tenant_id);
CREATE INDEX idx_p033_ip_scan          ON pack033_quick_wins.implementation_progress(scan_id);
CREATE INDEX idx_p033_ip_action        ON pack033_quick_wins.implementation_progress(action_id);
CREATE INDEX idx_p033_ip_status        ON pack033_quick_wins.implementation_progress(status);
CREATE INDEX idx_p033_ip_completion    ON pack033_quick_wins.implementation_progress(completion_pct);
CREATE INDEX idx_p033_ip_planned_start ON pack033_quick_wins.implementation_progress(planned_start_date);
CREATE INDEX idx_p033_ip_planned_end   ON pack033_quick_wins.implementation_progress(planned_end_date);
CREATE INDEX idx_p033_ip_actual_end    ON pack033_quick_wins.implementation_progress(actual_end_date DESC);
CREATE INDEX idx_p033_ip_created       ON pack033_quick_wins.implementation_progress(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_ip_updated
    BEFORE UPDATE ON pack033_quick_wins.implementation_progress
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack033_quick_wins.savings_actuals
-- =============================================================================
-- Verified actual savings measurements per action for M&V (Measurement and
-- Verification) comparison against estimated savings.

CREATE TABLE pack033_quick_wins.savings_actuals (
    actual_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    progress_id             UUID            NOT NULL REFERENCES pack033_quick_wins.implementation_progress(progress_id) ON DELETE CASCADE,
    measurement_period_start DATE           NOT NULL,
    measurement_period_end  DATE            NOT NULL,
    baseline_consumption    NUMERIC(16,2)   NOT NULL,
    actual_consumption      NUMERIC(16,2)   NOT NULL,
    verified_savings        NUMERIC(14,2)   NOT NULL,
    verification_method     VARCHAR(50)     NOT NULL,
    confidence_pct          NUMERIC(5,2),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_sa_period CHECK (
        measurement_period_end >= measurement_period_start
    ),
    CONSTRAINT chk_p033_sa_baseline CHECK (
        baseline_consumption > 0
    ),
    CONSTRAINT chk_p033_sa_actual CHECK (
        actual_consumption >= 0
    ),
    CONSTRAINT chk_p033_sa_verification_method CHECK (
        verification_method IN ('IPMVP_OPTION_A', 'IPMVP_OPTION_B', 'IPMVP_OPTION_C',
                                  'IPMVP_OPTION_D', 'BILLING_ANALYSIS', 'SUB_METERING',
                                  'ENGINEERING_CALCULATION', 'DEEMED', 'OTHER')
    ),
    CONSTRAINT chk_p033_sa_confidence CHECK (
        confidence_pct IS NULL OR (confidence_pct >= 0 AND confidence_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_sa_progress      ON pack033_quick_wins.savings_actuals(progress_id);
CREATE INDEX idx_p033_sa_period_start  ON pack033_quick_wins.savings_actuals(measurement_period_start);
CREATE INDEX idx_p033_sa_period_end    ON pack033_quick_wins.savings_actuals(measurement_period_end DESC);
CREATE INDEX idx_p033_sa_method        ON pack033_quick_wins.savings_actuals(verification_method);
CREATE INDEX idx_p033_sa_created       ON pack033_quick_wins.savings_actuals(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_sa_updated
    BEFORE UPDATE ON pack033_quick_wins.savings_actuals
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack033_quick_wins.implementation_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack033_quick_wins.savings_actuals ENABLE ROW LEVEL SECURITY;

CREATE POLICY p033_ip_tenant_isolation
    ON pack033_quick_wins.implementation_progress
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p033_ip_service_bypass
    ON pack033_quick_wins.implementation_progress
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p033_sa_tenant_isolation
    ON pack033_quick_wins.savings_actuals
    USING (progress_id IN (
        SELECT progress_id FROM pack033_quick_wins.implementation_progress
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_sa_service_bypass
    ON pack033_quick_wins.savings_actuals
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.implementation_progress TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.savings_actuals TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack033_quick_wins.implementation_progress IS
    'Tracks implementation status of individual quick-win actions with planned vs. actual dates and cost tracking.';

COMMENT ON TABLE pack033_quick_wins.savings_actuals IS
    'Verified actual savings measurements for M&V comparison against estimated savings using IPMVP protocols.';

COMMENT ON COLUMN pack033_quick_wins.implementation_progress.status IS
    'Implementation lifecycle: NOT_STARTED > PLANNING > PROCUREMENT > IN_PROGRESS > COMPLETED > VERIFIED.';
COMMENT ON COLUMN pack033_quick_wins.implementation_progress.completion_pct IS
    'Completion percentage (0-100); must be 100 when status is COMPLETED or VERIFIED.';
COMMENT ON COLUMN pack033_quick_wins.implementation_progress.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack033_quick_wins.savings_actuals.verification_method IS
    'IPMVP verification method used: Option A (Key Parameter), B (All Parameter), C (Utility Bill), D (Simulation).';
COMMENT ON COLUMN pack033_quick_wins.savings_actuals.verified_savings IS
    'Verified energy savings in the same unit as the original estimate (typically kWh).';
COMMENT ON COLUMN pack033_quick_wins.savings_actuals.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
