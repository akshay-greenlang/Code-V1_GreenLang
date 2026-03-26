-- =============================================================================
-- V349: PACK-043 Scope 3 Complete Pack - SBTi Targets & Pathways
-- =============================================================================
-- Pack:         PACK-043 (Scope 3 Complete Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates SBTi (Science Based Targets initiative) target management and
-- pathway tracking tables. Supports target definition with base year and
-- target year emissions, annual reduction pathway with on-track status,
-- interim milestones, and target submission/validation workflow. Implements
-- SBTi Criteria V5.1 for Scope 3 targets including the requirement that
-- targets cover at least 67% of Scope 3 emissions and achieve a minimum
-- annual linear reduction rate.
--
-- Tables (4):
--   1. ghg_accounting_scope3_complete.sbti_targets
--   2. ghg_accounting_scope3_complete.sbti_pathways
--   3. ghg_accounting_scope3_complete.sbti_milestones
--   4. ghg_accounting_scope3_complete.sbti_submissions
--
-- Enums (1):
--   1. ghg_accounting_scope3_complete.sbti_target_type
--
-- Also includes: indexes, RLS, comments.
-- Previous: V348__pack043_scenario_macc.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3_complete, public;

-- ---------------------------------------------------------------------------
-- Enum: sbti_target_type
-- ---------------------------------------------------------------------------
-- SBTi target classification per SBTi Criteria V5.1.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'sbti_target_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.sbti_target_type AS ENUM (
            'NEAR_TERM',        -- 5-10 year target (required for validation)
            'LONG_TERM',        -- Net-zero by 2050 or sooner
            'SUPPLIER_ENGAGEMENT', -- 67% of suppliers by spend set SBTs
            'NET_ZERO'          -- SBTi Corporate Net-Zero Standard
        );
    END IF;
END;
$$;

-- =============================================================================
-- Table 1: ghg_accounting_scope3_complete.sbti_targets
-- =============================================================================
-- SBTi target definition. Each target specifies the type (near-term,
-- long-term, supplier engagement), base year, base year emissions, target
-- year, target emissions, annual reduction rate, coverage percentage, and
-- validation status. Implements SBTi Criteria for Scope 3 targets.

CREATE TABLE ghg_accounting_scope3_complete.sbti_targets (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    -- Target definition
    target_type                 ghg_accounting_scope3_complete.sbti_target_type NOT NULL,
    target_name                 VARCHAR(500)    NOT NULL,
    target_description          TEXT,
    -- Base year
    base_year                   INTEGER         NOT NULL,
    base_year_tco2e             DECIMAL(15,3)   NOT NULL,
    base_year_methodology       VARCHAR(200),
    -- Target year
    target_year                 INTEGER         NOT NULL,
    target_tco2e                DECIMAL(15,3)   NOT NULL,
    -- Reduction metrics
    annual_reduction_pct        DECIMAL(5,2)    NOT NULL,
    total_reduction_pct         DECIMAL(5,2)    GENERATED ALWAYS AS (
        CASE WHEN base_year_tco2e > 0
            THEN ROUND(((base_year_tco2e - target_tco2e) / base_year_tco2e * 100)::NUMERIC, 2)
            ELSE 0
        END
    ) STORED,
    -- Coverage
    coverage_pct                DECIMAL(5,2)    NOT NULL DEFAULT 67.00,
    covered_categories          ghg_accounting_scope3_complete.scope3_category_type[],
    excluded_categories         ghg_accounting_scope3_complete.scope3_category_type[],
    -- SBTi alignment
    ambition_level              VARCHAR(30)     DEFAULT '1.5C',
    method                      VARCHAR(50)     DEFAULT 'ABSOLUTE_CONTRACTION',
    sector_pathway              VARCHAR(100),
    flag_enabled                BOOLEAN         NOT NULL DEFAULT true,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    validated                   BOOLEAN         NOT NULL DEFAULT false,
    validated_at                TIMESTAMPTZ,
    validation_expiry           DATE,
    -- Progress
    latest_actual_tco2e         DECIMAL(15,3),
    latest_actual_year          INTEGER,
    on_track                    BOOLEAN,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p043_sbt_base_year CHECK (base_year >= 2015 AND base_year <= 2100),
    CONSTRAINT chk_p043_sbt_base_tco2e CHECK (base_year_tco2e >= 0),
    CONSTRAINT chk_p043_sbt_target_year CHECK (target_year > base_year AND target_year <= 2100),
    CONSTRAINT chk_p043_sbt_target_tco2e CHECK (target_tco2e >= 0),
    CONSTRAINT chk_p043_sbt_annual_pct CHECK (
        annual_reduction_pct >= 0 AND annual_reduction_pct <= 100
    ),
    CONSTRAINT chk_p043_sbt_coverage CHECK (
        coverage_pct >= 0 AND coverage_pct <= 100
    ),
    CONSTRAINT chk_p043_sbt_ambition CHECK (
        ambition_level IS NULL OR ambition_level IN ('1.5C', 'WELL_BELOW_2C', '2C')
    ),
    CONSTRAINT chk_p043_sbt_method CHECK (
        method IS NULL OR method IN (
            'ABSOLUTE_CONTRACTION', 'SDA', 'ECONOMIC_INTENSITY',
            'PHYSICAL_INTENSITY', 'SUPPLIER_ENGAGEMENT'
        )
    ),
    CONSTRAINT chk_p043_sbt_status CHECK (
        status IN ('DRAFT', 'SUBMITTED', 'UNDER_REVIEW', 'VALIDATED', 'APPROVED', 'EXPIRED', 'WITHDRAWN')
    ),
    CONSTRAINT chk_p043_sbt_latest_actual CHECK (
        latest_actual_tco2e IS NULL OR latest_actual_tco2e >= 0
    ),
    CONSTRAINT chk_p043_sbt_latest_year CHECK (
        latest_actual_year IS NULL OR (latest_actual_year >= base_year AND latest_actual_year <= 2100)
    ),
    CONSTRAINT uq_p043_sbt_inventory_type_year UNIQUE (inventory_id, target_type, target_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_sbt_tenant            ON ghg_accounting_scope3_complete.sbti_targets(tenant_id);
CREATE INDEX idx_p043_sbt_inventory         ON ghg_accounting_scope3_complete.sbti_targets(inventory_id);
CREATE INDEX idx_p043_sbt_type              ON ghg_accounting_scope3_complete.sbti_targets(target_type);
CREATE INDEX idx_p043_sbt_base_year         ON ghg_accounting_scope3_complete.sbti_targets(base_year);
CREATE INDEX idx_p043_sbt_target_year       ON ghg_accounting_scope3_complete.sbti_targets(target_year);
CREATE INDEX idx_p043_sbt_annual_pct        ON ghg_accounting_scope3_complete.sbti_targets(annual_reduction_pct DESC);
CREATE INDEX idx_p043_sbt_coverage          ON ghg_accounting_scope3_complete.sbti_targets(coverage_pct);
CREATE INDEX idx_p043_sbt_status            ON ghg_accounting_scope3_complete.sbti_targets(status);
CREATE INDEX idx_p043_sbt_validated         ON ghg_accounting_scope3_complete.sbti_targets(validated) WHERE validated = true;
CREATE INDEX idx_p043_sbt_on_track          ON ghg_accounting_scope3_complete.sbti_targets(on_track);
CREATE INDEX idx_p043_sbt_ambition          ON ghg_accounting_scope3_complete.sbti_targets(ambition_level);
CREATE INDEX idx_p043_sbt_created           ON ghg_accounting_scope3_complete.sbti_targets(created_at DESC);
CREATE INDEX idx_p043_sbt_covered           ON ghg_accounting_scope3_complete.sbti_targets USING GIN(covered_categories);

-- Composite: inventory + validated for active targets
CREATE INDEX idx_p043_sbt_inv_validated     ON ghg_accounting_scope3_complete.sbti_targets(inventory_id, target_type)
    WHERE validated = true AND status = 'VALIDATED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_sbt_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.sbti_targets
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3_complete.sbti_pathways
-- =============================================================================
-- Annual pathway tracking for each SBTi target. Each row represents a single
-- year with the required emission level (from linear or SDA interpolation)
-- and the actual reported emissions. Variance and on-track status are
-- calculated to enable early warning of pathway deviations.

CREATE TABLE ghg_accounting_scope3_complete.sbti_pathways (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    target_id                   UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.sbti_targets(id) ON DELETE CASCADE,
    -- Year
    year                        INTEGER         NOT NULL,
    -- Pathway values
    required_tco2e              DECIMAL(15,3)   NOT NULL,
    actual_tco2e                DECIMAL(15,3),
    -- Variance
    variance_tco2e              DECIMAL(15,3)   GENERATED ALWAYS AS (
        CASE WHEN actual_tco2e IS NOT NULL
            THEN actual_tco2e - required_tco2e
            ELSE NULL
        END
    ) STORED,
    variance_pct                DECIMAL(8,2)    GENERATED ALWAYS AS (
        CASE WHEN actual_tco2e IS NOT NULL AND required_tco2e > 0
            THEN ROUND(((actual_tco2e - required_tco2e) / required_tco2e * 100)::NUMERIC, 2)
            ELSE NULL
        END
    ) STORED,
    on_track                    BOOLEAN,
    -- Data quality
    data_quality_rating         DECIMAL(3,1),
    data_completeness_pct       DECIMAL(5,2),
    -- Methodology
    calculation_method          VARCHAR(100),
    is_recalculated             BOOLEAN         NOT NULL DEFAULT false,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_sbtp_year CHECK (year >= 2015 AND year <= 2100),
    CONSTRAINT chk_p043_sbtp_required CHECK (required_tco2e >= 0),
    CONSTRAINT chk_p043_sbtp_actual CHECK (actual_tco2e IS NULL OR actual_tco2e >= 0),
    CONSTRAINT chk_p043_sbtp_dqr CHECK (
        data_quality_rating IS NULL OR (data_quality_rating >= 1.0 AND data_quality_rating <= 5.0)
    ),
    CONSTRAINT chk_p043_sbtp_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT uq_p043_sbtp_target_year UNIQUE (target_id, year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_sbtp_tenant           ON ghg_accounting_scope3_complete.sbti_pathways(tenant_id);
CREATE INDEX idx_p043_sbtp_target           ON ghg_accounting_scope3_complete.sbti_pathways(target_id);
CREATE INDEX idx_p043_sbtp_year             ON ghg_accounting_scope3_complete.sbti_pathways(year);
CREATE INDEX idx_p043_sbtp_on_track         ON ghg_accounting_scope3_complete.sbti_pathways(on_track);
CREATE INDEX idx_p043_sbtp_created          ON ghg_accounting_scope3_complete.sbti_pathways(created_at DESC);

-- Composite: target + year for pathway chart
CREATE INDEX idx_p043_sbtp_target_year_asc  ON ghg_accounting_scope3_complete.sbti_pathways(target_id, year);

-- Composite: off-track years for alerts
CREATE INDEX idx_p043_sbtp_off_track        ON ghg_accounting_scope3_complete.sbti_pathways(target_id, year)
    WHERE on_track = false;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_sbtp_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.sbti_pathways
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3_complete.sbti_milestones
-- =============================================================================
-- Interim milestones for SBTi target tracking. Each milestone represents a
-- checkpoint year with a target emission level and achievement status. Used
-- for progress reporting and early warning of off-track trajectories.

CREATE TABLE ghg_accounting_scope3_complete.sbti_milestones (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    target_id                   UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.sbti_targets(id) ON DELETE CASCADE,
    -- Milestone
    milestone_name              VARCHAR(200),
    milestone_year              INTEGER         NOT NULL,
    milestone_tco2e             DECIMAL(15,3)   NOT NULL,
    milestone_reduction_pct     DECIMAL(5,2),
    -- Achievement
    achieved                    BOOLEAN         NOT NULL DEFAULT false,
    actual_tco2e                DECIMAL(15,3),
    achievement_date            DATE,
    achievement_variance_pct    DECIMAL(8,2),
    -- Actions
    key_actions                 JSONB           DEFAULT '[]',
    dependencies                JSONB           DEFAULT '[]',
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    risk_level                  VARCHAR(20)     DEFAULT 'LOW',
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_sbtm_year CHECK (milestone_year >= 2015 AND milestone_year <= 2100),
    CONSTRAINT chk_p043_sbtm_tco2e CHECK (milestone_tco2e >= 0),
    CONSTRAINT chk_p043_sbtm_reduction CHECK (
        milestone_reduction_pct IS NULL OR (milestone_reduction_pct >= 0 AND milestone_reduction_pct <= 100)
    ),
    CONSTRAINT chk_p043_sbtm_actual CHECK (actual_tco2e IS NULL OR actual_tco2e >= 0),
    CONSTRAINT chk_p043_sbtm_status CHECK (
        status IN ('PENDING', 'ON_TRACK', 'AT_RISK', 'ACHIEVED', 'MISSED', 'DEFERRED')
    ),
    CONSTRAINT chk_p043_sbtm_risk CHECK (
        risk_level IS NULL OR risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT uq_p043_sbtm_target_year UNIQUE (target_id, milestone_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_sbtm_tenant           ON ghg_accounting_scope3_complete.sbti_milestones(tenant_id);
CREATE INDEX idx_p043_sbtm_target           ON ghg_accounting_scope3_complete.sbti_milestones(target_id);
CREATE INDEX idx_p043_sbtm_year             ON ghg_accounting_scope3_complete.sbti_milestones(milestone_year);
CREATE INDEX idx_p043_sbtm_achieved         ON ghg_accounting_scope3_complete.sbti_milestones(achieved);
CREATE INDEX idx_p043_sbtm_status           ON ghg_accounting_scope3_complete.sbti_milestones(status);
CREATE INDEX idx_p043_sbtm_risk             ON ghg_accounting_scope3_complete.sbti_milestones(risk_level);
CREATE INDEX idx_p043_sbtm_created          ON ghg_accounting_scope3_complete.sbti_milestones(created_at DESC);

-- Composite: target + upcoming milestones
CREATE INDEX idx_p043_sbtm_upcoming         ON ghg_accounting_scope3_complete.sbti_milestones(target_id, milestone_year)
    WHERE achieved = false;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_sbtm_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.sbti_milestones
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3_complete.sbti_submissions
-- =============================================================================
-- SBTi target submission and validation workflow tracking. Records the
-- submission package, submission date, SBTi feedback, and validation status.

CREATE TABLE ghg_accounting_scope3_complete.sbti_submissions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    target_id                   UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.sbti_targets(id) ON DELETE CASCADE,
    -- Submission
    submission_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    submission_type             VARCHAR(50)     NOT NULL DEFAULT 'INITIAL',
    submitted_by                VARCHAR(255),
    -- Submission data
    submission_data             JSONB           NOT NULL DEFAULT '{}',
    target_summary              JSONB           DEFAULT '{}',
    supporting_documents        TEXT[],
    -- Validation
    validation_status           VARCHAR(30)     NOT NULL DEFAULT 'SUBMITTED',
    validation_date             TIMESTAMPTZ,
    validator_name              VARCHAR(255),
    -- Feedback
    feedback                    TEXT,
    feedback_items              JSONB           DEFAULT '[]',
    revision_required           BOOLEAN         NOT NULL DEFAULT false,
    revision_deadline           DATE,
    -- Outcome
    approved                    BOOLEAN,
    approval_date               DATE,
    certificate_ref             VARCHAR(100),
    expiry_date                 DATE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_sbts_type CHECK (
        submission_type IN ('INITIAL', 'REVISION', 'RESUBMISSION', 'ANNUAL_UPDATE', 'TARGET_UPDATE')
    ),
    CONSTRAINT chk_p043_sbts_validation CHECK (
        validation_status IN (
            'SUBMITTED', 'UNDER_REVIEW', 'ADDITIONAL_INFO_REQUESTED',
            'APPROVED', 'CONDITIONALLY_APPROVED', 'REJECTED', 'WITHDRAWN'
        )
    ),
    CONSTRAINT chk_p043_sbts_revision_deadline CHECK (
        revision_deadline IS NULL OR revision_deadline >= submission_date::DATE
    ),
    CONSTRAINT chk_p043_sbts_expiry CHECK (
        expiry_date IS NULL OR expiry_date > approval_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_sbts_tenant           ON ghg_accounting_scope3_complete.sbti_submissions(tenant_id);
CREATE INDEX idx_p043_sbts_target           ON ghg_accounting_scope3_complete.sbti_submissions(target_id);
CREATE INDEX idx_p043_sbts_date             ON ghg_accounting_scope3_complete.sbti_submissions(submission_date DESC);
CREATE INDEX idx_p043_sbts_type             ON ghg_accounting_scope3_complete.sbti_submissions(submission_type);
CREATE INDEX idx_p043_sbts_status           ON ghg_accounting_scope3_complete.sbti_submissions(validation_status);
CREATE INDEX idx_p043_sbts_approved         ON ghg_accounting_scope3_complete.sbti_submissions(approved) WHERE approved = true;
CREATE INDEX idx_p043_sbts_created          ON ghg_accounting_scope3_complete.sbti_submissions(created_at DESC);
CREATE INDEX idx_p043_sbts_submission_data  ON ghg_accounting_scope3_complete.sbti_submissions USING GIN(submission_data);
CREATE INDEX idx_p043_sbts_feedback         ON ghg_accounting_scope3_complete.sbti_submissions USING GIN(feedback_items);

-- Composite: target + latest submission
CREATE INDEX idx_p043_sbts_target_latest    ON ghg_accounting_scope3_complete.sbti_submissions(target_id, submission_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_sbts_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.sbti_submissions
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3_complete.sbti_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.sbti_pathways ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.sbti_milestones ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.sbti_submissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p043_sbt_tenant_isolation ON ghg_accounting_scope3_complete.sbti_targets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_sbt_service_bypass ON ghg_accounting_scope3_complete.sbti_targets
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_sbtp_tenant_isolation ON ghg_accounting_scope3_complete.sbti_pathways
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_sbtp_service_bypass ON ghg_accounting_scope3_complete.sbti_pathways
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_sbtm_tenant_isolation ON ghg_accounting_scope3_complete.sbti_milestones
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_sbtm_service_bypass ON ghg_accounting_scope3_complete.sbti_milestones
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_sbts_tenant_isolation ON ghg_accounting_scope3_complete.sbti_submissions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_sbts_service_bypass ON ghg_accounting_scope3_complete.sbti_submissions
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.sbti_targets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.sbti_pathways TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.sbti_milestones TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.sbti_submissions TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3_complete.sbti_targets IS
    'SBTi target definition per Criteria V5.1 with base year, target year, annual reduction rate, coverage percentage, and validation status.';
COMMENT ON TABLE ghg_accounting_scope3_complete.sbti_pathways IS
    'Annual pathway tracking per SBTi target with required vs actual emissions, variance, and on-track status.';
COMMENT ON TABLE ghg_accounting_scope3_complete.sbti_milestones IS
    'Interim milestones for SBTi target progress with achievement tracking, risk level, and key actions.';
COMMENT ON TABLE ghg_accounting_scope3_complete.sbti_submissions IS
    'SBTi target submission workflow with submission data, validation status, feedback, and approval tracking.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.sbti_targets.total_reduction_pct IS 'Generated column: ((base_year - target) / base_year) * 100.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.sbti_targets.coverage_pct IS 'Percentage of Scope 3 emissions covered by target boundary (SBTi minimum: 67%).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.sbti_targets.annual_reduction_pct IS 'Required annual linear reduction rate (SBTi near-term: ~2.5% for 1.5C alignment).';

COMMENT ON COLUMN ghg_accounting_scope3_complete.sbti_pathways.variance_tco2e IS 'Generated column: actual - required (negative = ahead of target).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.sbti_pathways.variance_pct IS 'Generated column: ((actual - required) / required) * 100.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.sbti_submissions.submission_data IS 'JSONB containing the full SBTi target submission form data.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.sbti_submissions.feedback_items IS 'JSONB array of individual feedback items from SBTi reviewers.';
