-- =============================================================================
-- V169: PACK-027 Enterprise Net Zero - SBTi Targets
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    004 of 015
-- Date:         March 2026
--
-- SBTi Corporate Standard target management with full near-term (C1-C28),
-- long-term, and net-zero (NZ-C1 to NZ-C14) target tracking. Supports ACA,
-- SDA (12 sectors), and FLAG pathways with criteria validation matrices,
-- annual milestones, and submission status tracking.
--
-- Tables (1):
--   1. pack027_enterprise_net_zero.gl_sbti_targets
--
-- Previous: V168__PACK027_comprehensive_baselines.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_sbti_targets
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_sbti_targets (
    target_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    baseline_id                 UUID            REFERENCES pack027_enterprise_net_zero.gl_enterprise_baselines(baseline_id) ON DELETE SET NULL,
    -- Target classification
    target_type                 VARCHAR(30)     NOT NULL,
    pathway_type                VARCHAR(30)     NOT NULL,
    ambition_level              VARCHAR(20)     NOT NULL DEFAULT '1.5C',
    -- Near-term target
    near_term_target            DECIMAL(6,2),
    near_term_year              INTEGER,
    near_term_base_year         INTEGER,
    near_term_annual_rate       DECIMAL(6,3),
    near_term_scope_coverage    JSONB           DEFAULT '{}',
    -- Long-term target
    long_term_target            DECIMAL(6,2),
    long_term_year              INTEGER,
    long_term_residual_pct      DECIMAL(6,2),
    -- FLAG target (if applicable)
    flag_target                 DECIMAL(6,2),
    flag_target_year            INTEGER,
    flag_applicable             BOOLEAN         DEFAULT FALSE,
    flag_emissions_pct          DECIMAL(6,2),
    flag_commodities            TEXT[]          DEFAULT '{}',
    -- SDA sector (if applicable)
    sda_sector                  VARCHAR(50),
    sda_intensity_metric        VARCHAR(50),
    sda_base_intensity          DECIMAL(18,6),
    sda_target_intensity        DECIMAL(18,6),
    -- Coverage
    scope1_coverage_pct         DECIMAL(6,2)    DEFAULT 95.00,
    scope2_coverage_pct         DECIMAL(6,2)    DEFAULT 95.00,
    scope3_coverage_pct         DECIMAL(6,2),
    scope3_categories_included  TEXT[]          DEFAULT '{}',
    -- Criteria validation (28 near-term + 14 net-zero)
    criteria_validation         JSONB           DEFAULT '{}',
    near_term_criteria_pass     INTEGER         DEFAULT 0,
    near_term_criteria_total    INTEGER         DEFAULT 28,
    netzero_criteria_pass       INTEGER         DEFAULT 0,
    netzero_criteria_total      INTEGER         DEFAULT 14,
    -- Milestones
    annual_milestones           JSONB           DEFAULT '{}',
    five_year_review_dates      DATE[]          DEFAULT '{}',
    -- Submission
    validation_status           VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    submission_date             DATE,
    validation_date             DATE,
    revalidation_due            DATE,
    sbti_target_id              VARCHAR(100),
    -- Neutralization plan (net-zero only)
    neutralization_approach     VARCHAR(50),
    neutralization_volume_tco2e DECIMAL(18,4),
    cdr_only                    BOOLEAN         DEFAULT TRUE,
    -- Governance
    board_approved              BOOLEAN         DEFAULT FALSE,
    board_approval_date         DATE,
    public_commitment           BOOLEAN         DEFAULT FALSE,
    public_commitment_url       TEXT,
    -- Metadata
    submission_readiness_score  DECIMAL(5,2),
    warnings                    TEXT[]          DEFAULT '{}',
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_st_target_type CHECK (
        target_type IN ('NEAR_TERM', 'LONG_TERM', 'NET_ZERO', 'FLAG', 'COMBINED')
    ),
    CONSTRAINT chk_p027_st_pathway CHECK (
        pathway_type IN ('ACA', 'SDA', 'FLAG', 'ACA_SDA_COMBINED', 'ACA_FLAG_COMBINED', 'CUSTOM')
    ),
    CONSTRAINT chk_p027_st_ambition CHECK (
        ambition_level IN ('1.5C', 'WELL_BELOW_2C', '2C', 'SECTOR_ALIGNED')
    ),
    CONSTRAINT chk_p027_st_near_term_target CHECK (
        near_term_target IS NULL OR (near_term_target >= 0 AND near_term_target <= 100)
    ),
    CONSTRAINT chk_p027_st_near_term_year CHECK (
        near_term_year IS NULL OR (near_term_year >= 2025 AND near_term_year <= 2040)
    ),
    CONSTRAINT chk_p027_st_long_term_target CHECK (
        long_term_target IS NULL OR (long_term_target >= 0 AND long_term_target <= 100)
    ),
    CONSTRAINT chk_p027_st_long_term_year CHECK (
        long_term_year IS NULL OR (long_term_year >= 2040 AND long_term_year <= 2060)
    ),
    CONSTRAINT chk_p027_st_residual CHECK (
        long_term_residual_pct IS NULL OR (long_term_residual_pct >= 0 AND long_term_residual_pct <= 10)
    ),
    CONSTRAINT chk_p027_st_scope1_coverage CHECK (
        scope1_coverage_pct IS NULL OR (scope1_coverage_pct >= 0 AND scope1_coverage_pct <= 100)
    ),
    CONSTRAINT chk_p027_st_scope3_coverage CHECK (
        scope3_coverage_pct IS NULL OR (scope3_coverage_pct >= 0 AND scope3_coverage_pct <= 100)
    ),
    CONSTRAINT chk_p027_st_validation CHECK (
        validation_status IN ('DRAFT', 'READY', 'SUBMITTED', 'UNDER_REVIEW', 'VALIDATED',
                              'REVISION_REQUESTED', 'REJECTED', 'EXPIRED', 'REVALIDATION_DUE')
    ),
    CONSTRAINT chk_p027_st_sda_sector CHECK (
        sda_sector IS NULL OR sda_sector IN (
            'POWER_GENERATION', 'CEMENT', 'IRON_STEEL', 'ALUMINIUM', 'PULP_PAPER',
            'CHEMICALS', 'AVIATION', 'MARITIME_SHIPPING', 'ROAD_TRANSPORT',
            'COMMERCIAL_BUILDINGS', 'RESIDENTIAL_BUILDINGS', 'FOOD_BEVERAGE'
        )
    ),
    CONSTRAINT chk_p027_st_neutralization CHECK (
        neutralization_approach IS NULL OR neutralization_approach IN (
            'PERMANENT_CDR', 'NATURE_BASED_CDR', 'HYBRID_CDR', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p027_st_readiness CHECK (
        submission_readiness_score IS NULL OR (submission_readiness_score >= 0 AND submission_readiness_score <= 100)
    ),
    CONSTRAINT chk_p027_st_near_term_rate CHECK (
        near_term_annual_rate IS NULL OR near_term_annual_rate >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_st_company            ON pack027_enterprise_net_zero.gl_sbti_targets(company_id);
CREATE INDEX idx_p027_st_tenant             ON pack027_enterprise_net_zero.gl_sbti_targets(tenant_id);
CREATE INDEX idx_p027_st_baseline           ON pack027_enterprise_net_zero.gl_sbti_targets(baseline_id);
CREATE INDEX idx_p027_st_target_type        ON pack027_enterprise_net_zero.gl_sbti_targets(target_type);
CREATE INDEX idx_p027_st_pathway            ON pack027_enterprise_net_zero.gl_sbti_targets(pathway_type);
CREATE INDEX idx_p027_st_ambition           ON pack027_enterprise_net_zero.gl_sbti_targets(ambition_level);
CREATE INDEX idx_p027_st_validation         ON pack027_enterprise_net_zero.gl_sbti_targets(validation_status);
CREATE INDEX idx_p027_st_near_term_year     ON pack027_enterprise_net_zero.gl_sbti_targets(near_term_year);
CREATE INDEX idx_p027_st_long_term_year     ON pack027_enterprise_net_zero.gl_sbti_targets(long_term_year);
CREATE INDEX idx_p027_st_submission         ON pack027_enterprise_net_zero.gl_sbti_targets(submission_date);
CREATE INDEX idx_p027_st_revalidation       ON pack027_enterprise_net_zero.gl_sbti_targets(revalidation_due);
CREATE INDEX idx_p027_st_flag               ON pack027_enterprise_net_zero.gl_sbti_targets(flag_applicable) WHERE flag_applicable = TRUE;
CREATE INDEX idx_p027_st_sda_sector         ON pack027_enterprise_net_zero.gl_sbti_targets(sda_sector);
CREATE INDEX idx_p027_st_board              ON pack027_enterprise_net_zero.gl_sbti_targets(board_approved);
CREATE INDEX idx_p027_st_criteria           ON pack027_enterprise_net_zero.gl_sbti_targets USING GIN(criteria_validation);
CREATE INDEX idx_p027_st_milestones         ON pack027_enterprise_net_zero.gl_sbti_targets USING GIN(annual_milestones);
CREATE INDEX idx_p027_st_created            ON pack027_enterprise_net_zero.gl_sbti_targets(created_at DESC);
CREATE INDEX idx_p027_st_metadata           ON pack027_enterprise_net_zero.gl_sbti_targets USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_sbti_targets_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_sbti_targets
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_sbti_targets ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_st_tenant_isolation
    ON pack027_enterprise_net_zero.gl_sbti_targets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p027_st_service_bypass
    ON pack027_enterprise_net_zero.gl_sbti_targets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_sbti_targets TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_sbti_targets IS
    'SBTi Corporate Standard targets with near-term (C1-C28), long-term, net-zero (NZ-C1 to NZ-C14), FLAG, and SDA pathway support including criteria validation and submission tracking.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_sbti_targets.target_id IS 'Unique target identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_sbti_targets.target_type IS 'Target classification: NEAR_TERM, LONG_TERM, NET_ZERO, FLAG, COMBINED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_sbti_targets.pathway_type IS 'Decarbonization pathway: ACA (4.2%/yr for 1.5C), SDA (sector-specific), FLAG (land use).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_sbti_targets.near_term_target IS 'Near-term absolute reduction target percentage from base year.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_sbti_targets.near_term_year IS 'Target year for near-term commitment (5-10 years from submission).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_sbti_targets.long_term_target IS 'Long-term absolute reduction target percentage (typically 90%+).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_sbti_targets.validation_status IS 'SBTi validation pipeline status: DRAFT through VALIDATED or REJECTED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_sbti_targets.criteria_validation IS 'JSONB matrix of 42 criteria (28 near-term + 14 net-zero) with pass/fail/warning per criterion.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_sbti_targets.submission_date IS 'Date SBTi target was submitted for validation.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_sbti_targets.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
