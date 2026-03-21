-- =============================================================================
-- V251: PACK-033 Quick Wins Identifier - Behavioral Actions
-- =============================================================================
-- Pack:         PACK-033 (Quick Wins Identifier Pack)
-- Migration:    006 of 010
-- Date:         March 2026
--
-- Creates behavioral action program tables for tracking zero-cost and low-cost
-- behavioral interventions, engagement campaigns, adoption rates, and savings
-- persistence over time.
--
-- Tables (2):
--   1. pack033_quick_wins.behavioral_programs
--   2. pack033_quick_wins.adoption_tracking
--
-- Previous: V250__pack033_quick_wins_005.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack033_quick_wins.behavioral_programs
-- =============================================================================
-- Behavioral energy efficiency programs targeting occupant behavior change
-- with engagement metrics, adoption rates, and persistence tracking.

CREATE TABLE pack033_quick_wins.behavioral_programs (
    program_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    program_name            VARCHAR(500)    NOT NULL,
    start_date              DATE            NOT NULL,
    end_date                DATE,
    target_audience         VARCHAR(100)    NOT NULL,
    actions                 JSONB           NOT NULL DEFAULT '[]',
    engagement_score        NUMERIC(6,2),
    adoption_rate           NUMERIC(6,2),
    savings_kwh             NUMERIC(14,2),
    savings_persistence_factor NUMERIC(6,4) DEFAULT 0.80,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_bp_target_audience CHECK (
        target_audience IN ('ALL_OCCUPANTS', 'OFFICE_WORKERS', 'FACILITY_MANAGERS',
                             'MAINTENANCE_STAFF', 'CLEANING_STAFF', 'EXECUTIVES',
                             'TENANTS', 'VISITORS', 'CONTRACTORS', 'OTHER')
    ),
    CONSTRAINT chk_p033_bp_engagement CHECK (
        engagement_score IS NULL OR (engagement_score >= 0 AND engagement_score <= 100)
    ),
    CONSTRAINT chk_p033_bp_adoption CHECK (
        adoption_rate IS NULL OR (adoption_rate >= 0 AND adoption_rate <= 100)
    ),
    CONSTRAINT chk_p033_bp_savings CHECK (
        savings_kwh IS NULL OR savings_kwh >= 0
    ),
    CONSTRAINT chk_p033_bp_persistence CHECK (
        savings_persistence_factor IS NULL OR (savings_persistence_factor >= 0 AND savings_persistence_factor <= 1)
    ),
    CONSTRAINT chk_p033_bp_dates CHECK (
        end_date IS NULL OR end_date >= start_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_bp_tenant        ON pack033_quick_wins.behavioral_programs(tenant_id);
CREATE INDEX idx_p033_bp_facility      ON pack033_quick_wins.behavioral_programs(facility_id);
CREATE INDEX idx_p033_bp_audience      ON pack033_quick_wins.behavioral_programs(target_audience);
CREATE INDEX idx_p033_bp_start_date    ON pack033_quick_wins.behavioral_programs(start_date DESC);
CREATE INDEX idx_p033_bp_end_date      ON pack033_quick_wins.behavioral_programs(end_date);
CREATE INDEX idx_p033_bp_actions       ON pack033_quick_wins.behavioral_programs USING GIN(actions);
CREATE INDEX idx_p033_bp_created       ON pack033_quick_wins.behavioral_programs(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_bp_updated
    BEFORE UPDATE ON pack033_quick_wins.behavioral_programs
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack033_quick_wins.adoption_tracking
-- =============================================================================
-- Granular adoption tracking per action within a behavioral program, with
-- adoption stage progression, decay rates, and persistence monitoring.

CREATE TABLE pack033_quick_wins.adoption_tracking (
    tracking_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    program_id              UUID            NOT NULL REFERENCES pack033_quick_wins.behavioral_programs(program_id) ON DELETE CASCADE,
    action_id               UUID            NOT NULL,
    adoption_stage          VARCHAR(30)     NOT NULL DEFAULT 'AWARENESS',
    adoption_pct            NUMERIC(6,2)    NOT NULL DEFAULT 0,
    measurement_date        DATE            NOT NULL DEFAULT CURRENT_DATE,
    decay_rate              NUMERIC(6,4),
    persistence_months      INTEGER,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_at_stage CHECK (
        adoption_stage IN ('AWARENESS', 'INTEREST', 'TRIAL', 'ADOPTION', 'SUSTAINED', 'DECLINED')
    ),
    CONSTRAINT chk_p033_at_adoption_pct CHECK (
        adoption_pct >= 0 AND adoption_pct <= 100
    ),
    CONSTRAINT chk_p033_at_decay_rate CHECK (
        decay_rate IS NULL OR (decay_rate >= 0 AND decay_rate <= 1)
    ),
    CONSTRAINT chk_p033_at_persistence CHECK (
        persistence_months IS NULL OR persistence_months >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_at_program       ON pack033_quick_wins.adoption_tracking(program_id);
CREATE INDEX idx_p033_at_action        ON pack033_quick_wins.adoption_tracking(action_id);
CREATE INDEX idx_p033_at_stage         ON pack033_quick_wins.adoption_tracking(adoption_stage);
CREATE INDEX idx_p033_at_measure_date  ON pack033_quick_wins.adoption_tracking(measurement_date DESC);
CREATE INDEX idx_p033_at_created       ON pack033_quick_wins.adoption_tracking(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_at_updated
    BEFORE UPDATE ON pack033_quick_wins.adoption_tracking
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack033_quick_wins.behavioral_programs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack033_quick_wins.adoption_tracking ENABLE ROW LEVEL SECURITY;

CREATE POLICY p033_bp_tenant_isolation
    ON pack033_quick_wins.behavioral_programs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p033_bp_service_bypass
    ON pack033_quick_wins.behavioral_programs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p033_at_tenant_isolation
    ON pack033_quick_wins.adoption_tracking
    USING (program_id IN (
        SELECT program_id FROM pack033_quick_wins.behavioral_programs
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_at_service_bypass
    ON pack033_quick_wins.adoption_tracking
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.behavioral_programs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.adoption_tracking TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack033_quick_wins.behavioral_programs IS
    'Behavioral energy efficiency programs targeting occupant behavior change with engagement and adoption metrics.';

COMMENT ON TABLE pack033_quick_wins.adoption_tracking IS
    'Granular adoption tracking per action within a behavioral program with decay rates and persistence monitoring.';

COMMENT ON COLUMN pack033_quick_wins.behavioral_programs.savings_persistence_factor IS
    'Long-term persistence factor for behavioral savings (0-1); typical range 0.5-0.9, default 0.8.';
COMMENT ON COLUMN pack033_quick_wins.behavioral_programs.engagement_score IS
    'Overall program engagement score (0-100) based on participation, feedback, and activity levels.';
COMMENT ON COLUMN pack033_quick_wins.behavioral_programs.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack033_quick_wins.adoption_tracking.adoption_stage IS
    'Adoption lifecycle stage: AWARENESS > INTEREST > TRIAL > ADOPTION > SUSTAINED or DECLINED.';
COMMENT ON COLUMN pack033_quick_wins.adoption_tracking.decay_rate IS
    'Monthly decay rate of adoption after program ends (0-1); used for persistence projections.';
COMMENT ON COLUMN pack033_quick_wins.adoption_tracking.persistence_months IS
    'Number of months the behavioral change has persisted after initial adoption.';
