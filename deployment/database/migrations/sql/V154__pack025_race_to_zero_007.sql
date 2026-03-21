-- =============================================================================
-- V154: PACK-025 Race to Zero - Partnerships
-- =============================================================================
-- Pack:         PACK-025 (Race to Zero Pack)
-- Migration:    007 of 010
-- Date:         March 2026
--
-- Partnership collaboration agreements between organizations with joint
-- targets, governance structures, and year-over-year performance tracking.
--
-- Tables (2):
--   1. pack025_race_to_zero.partnerships
--   2. pack025_race_to_zero.partnership_performance
--
-- Previous: V153__pack025_race_to_zero_006.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack025_race_to_zero.partnerships
-- =============================================================================
-- Partnership collaboration agreements between Race to Zero participants
-- with joint emission reduction targets and governance structures.

CREATE TABLE pack025_race_to_zero.partnerships (
    partnership_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_org_id             UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    partner_org_id          UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    partnership_type        VARCHAR(50)     NOT NULL,
    partnership_name        VARCHAR(255),
    start_date              DATE            NOT NULL,
    end_date                DATE,
    joint_target_tco2e      DECIMAL(18,4),
    joint_target_year       INTEGER,
    governance_structure    TEXT,
    governance_details      JSONB           DEFAULT '{}',
    shared_resources        JSONB           DEFAULT '{}',
    reporting_cadence       VARCHAR(30)     DEFAULT 'ANNUAL',
    collaboration_areas     TEXT[]          DEFAULT '{}',
    partnership_status      VARCHAR(30)     NOT NULL DEFAULT 'proposed',
    alignment_score         DECIMAL(6,2),
    last_review_date        DATE,
    next_review_date        DATE,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_part_type CHECK (
        partnership_type IN ('SUPPLY_CHAIN', 'SECTOR_COALITION', 'GEOGRAPHIC',
                             'VALUE_CHAIN', 'CROSS_SECTOR', 'PUBLIC_PRIVATE',
                             'INDUSTRY_BODY', 'BILATERAL')
    ),
    CONSTRAINT chk_p025_part_status CHECK (
        partnership_status IN ('proposed', 'active', 'on_hold', 'completed', 'terminated')
    ),
    CONSTRAINT chk_p025_part_cadence CHECK (
        reporting_cadence IN ('MONTHLY', 'QUARTERLY', 'SEMI_ANNUAL', 'ANNUAL')
    ),
    CONSTRAINT chk_p025_part_target_non_neg CHECK (
        joint_target_tco2e IS NULL OR joint_target_tco2e >= 0
    ),
    CONSTRAINT chk_p025_part_alignment CHECK (
        alignment_score IS NULL OR (alignment_score >= 0 AND alignment_score <= 100)
    ),
    CONSTRAINT chk_p025_part_different_orgs CHECK (
        lead_org_id <> partner_org_id
    ),
    CONSTRAINT chk_p025_part_date_order CHECK (
        end_date IS NULL OR start_date <= end_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for partnerships
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_part_lead          ON pack025_race_to_zero.partnerships(lead_org_id);
CREATE INDEX idx_p025_part_partner       ON pack025_race_to_zero.partnerships(partner_org_id);
CREATE INDEX idx_p025_part_tenant        ON pack025_race_to_zero.partnerships(tenant_id);
CREATE INDEX idx_p025_part_type          ON pack025_race_to_zero.partnerships(partnership_type);
CREATE INDEX idx_p025_part_status        ON pack025_race_to_zero.partnerships(partnership_status);
CREATE INDEX idx_p025_part_start         ON pack025_race_to_zero.partnerships(start_date);
CREATE INDEX idx_p025_part_alignment     ON pack025_race_to_zero.partnerships(alignment_score);
CREATE INDEX idx_p025_part_created       ON pack025_race_to_zero.partnerships(created_at DESC);
CREATE INDEX idx_p025_part_governance    ON pack025_race_to_zero.partnerships USING GIN(governance_details);
CREATE INDEX idx_p025_part_resources     ON pack025_race_to_zero.partnerships USING GIN(shared_resources);
CREATE INDEX idx_p025_part_metadata      ON pack025_race_to_zero.partnerships USING GIN(metadata);

-- =============================================================================
-- Table 2: pack025_race_to_zero.partnership_performance
-- =============================================================================
-- Year-over-year performance tracking for partnership collaborations
-- with actual vs target reduction and quality scoring.

CREATE TABLE pack025_race_to_zero.partnership_performance (
    performance_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    partnership_id          UUID            NOT NULL REFERENCES pack025_race_to_zero.partnerships(partnership_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    year                    INTEGER         NOT NULL,
    actual_reduction_tco2e  DECIMAL(18,4),
    target_reduction_tco2e  DECIMAL(18,4),
    achievement_pct         DECIMAL(8,3),
    quality_score           DECIMAL(6,2),
    lead_contribution_tco2e DECIMAL(18,4),
    partner_contribution_tco2e DECIMAL(18,4),
    joint_initiatives       JSONB           DEFAULT '[]',
    challenges              TEXT[],
    milestones_achieved     JSONB           DEFAULT '[]',
    review_date             DATE,
    review_notes            TEXT,
    performance_status      VARCHAR(30)     DEFAULT 'PENDING',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_pp_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p025_pp_status CHECK (
        performance_status IN ('EXCEEDING', 'ON_TRACK', 'BEHIND', 'AT_RISK', 'PENDING')
    ),
    CONSTRAINT chk_p025_pp_quality CHECK (
        quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 100)
    ),
    CONSTRAINT chk_p025_pp_reduction_non_neg CHECK (
        (actual_reduction_tco2e IS NULL OR actual_reduction_tco2e >= 0) AND
        (target_reduction_tco2e IS NULL OR target_reduction_tco2e >= 0)
    ),
    CONSTRAINT uq_p025_pp_partnership_year UNIQUE (partnership_id, year)
);

-- ---------------------------------------------------------------------------
-- Indexes for partnership_performance
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_pp_partnership     ON pack025_race_to_zero.partnership_performance(partnership_id);
CREATE INDEX idx_p025_pp_tenant          ON pack025_race_to_zero.partnership_performance(tenant_id);
CREATE INDEX idx_p025_pp_year            ON pack025_race_to_zero.partnership_performance(year);
CREATE INDEX idx_p025_pp_achievement     ON pack025_race_to_zero.partnership_performance(achievement_pct);
CREATE INDEX idx_p025_pp_quality         ON pack025_race_to_zero.partnership_performance(quality_score);
CREATE INDEX idx_p025_pp_status          ON pack025_race_to_zero.partnership_performance(performance_status);
CREATE INDEX idx_p025_pp_created         ON pack025_race_to_zero.partnership_performance(created_at DESC);
CREATE INDEX idx_p025_pp_initiatives     ON pack025_race_to_zero.partnership_performance USING GIN(joint_initiatives);
CREATE INDEX idx_p025_pp_milestones      ON pack025_race_to_zero.partnership_performance USING GIN(milestones_achieved);
CREATE INDEX idx_p025_pp_metadata        ON pack025_race_to_zero.partnership_performance USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p025_partnerships_updated
    BEFORE UPDATE ON pack025_race_to_zero.partnerships
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

CREATE TRIGGER trg_p025_partnership_perf_updated
    BEFORE UPDATE ON pack025_race_to_zero.partnership_performance
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack025_race_to_zero.partnerships ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack025_race_to_zero.partnership_performance ENABLE ROW LEVEL SECURITY;

CREATE POLICY p025_part_tenant_isolation
    ON pack025_race_to_zero.partnerships
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_part_service_bypass
    ON pack025_race_to_zero.partnerships
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p025_pp_tenant_isolation
    ON pack025_race_to_zero.partnership_performance
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_pp_service_bypass
    ON pack025_race_to_zero.partnership_performance
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.partnerships TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.partnership_performance TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack025_race_to_zero.partnerships IS
    'Partnership collaboration agreements between Race to Zero participants with joint targets and governance structures.';
COMMENT ON TABLE pack025_race_to_zero.partnership_performance IS
    'Year-over-year performance tracking for partnership collaborations with actual vs target reduction.';

COMMENT ON COLUMN pack025_race_to_zero.partnerships.partnership_id IS 'Unique partnership identifier.';
COMMENT ON COLUMN pack025_race_to_zero.partnerships.lead_org_id IS 'Lead organization in the partnership.';
COMMENT ON COLUMN pack025_race_to_zero.partnerships.partner_org_id IS 'Partner organization in the collaboration.';
COMMENT ON COLUMN pack025_race_to_zero.partnerships.joint_target_tco2e IS 'Joint emission reduction target in tonnes CO2e.';
COMMENT ON COLUMN pack025_race_to_zero.partnerships.governance_structure IS 'Description of partnership governance structure.';
COMMENT ON COLUMN pack025_race_to_zero.partnership_performance.performance_id IS 'Unique performance record identifier.';
COMMENT ON COLUMN pack025_race_to_zero.partnership_performance.achievement_pct IS 'Percentage of target reduction achieved.';
COMMENT ON COLUMN pack025_race_to_zero.partnership_performance.quality_score IS 'Quality score for the partnership performance (0-100).';
