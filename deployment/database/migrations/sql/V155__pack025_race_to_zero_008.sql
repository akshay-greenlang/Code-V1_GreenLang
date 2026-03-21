-- =============================================================================
-- V155: PACK-025 Race to Zero - Credibility Assessments
-- =============================================================================
-- Pack:         PACK-025 (Race to Zero Pack)
-- Migration:    008 of 010
-- Date:         March 2026
--
-- HLEG "Integrity Matters" 10-recommendation credibility assessment with
-- individual recommendation scores, overall credibility tier, and
-- improvement recommendations. Separate lobbying assessment for climate
-- advocacy alignment and trade association scoring.
--
-- Tables (2):
--   1. pack025_race_to_zero.credibility_assessments
--   2. pack025_race_to_zero.lobbying_assessments
--
-- Previous: V154__pack025_race_to_zero_007.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack025_race_to_zero.credibility_assessments
-- =============================================================================
-- HLEG "Integrity Matters" credibility assessment with 10 individual
-- recommendation scores, overall credibility tier, and improvement plan.

CREATE TABLE pack025_race_to_zero.credibility_assessments (
    assessment_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    pledge_id               UUID            REFERENCES pack025_race_to_zero.pledges(pledge_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    assessment_date         DATE            NOT NULL,
    -- HLEG 10 Recommendation Scores (each 0-100)
    hleg_rec1_score         DECIMAL(6,2),   -- Ambition: net-zero pledge quality
    hleg_rec2_score         DECIMAL(6,2),   -- Integrity: targets based on science
    hleg_rec3_score         DECIMAL(6,2),   -- Credibility: no offsets for scope 1/2
    hleg_rec4_score         DECIMAL(6,2),   -- Accountability: external verification
    hleg_rec5_score         DECIMAL(6,2),   -- Just Transition: social equity
    hleg_rec6_score         DECIMAL(6,2),   -- Finance: climate finance alignment
    hleg_rec7_score         DECIMAL(6,2),   -- Transparency: public disclosure
    hleg_rec8_score         DECIMAL(6,2),   -- Scope: cover all material emissions
    hleg_rec9_score         DECIMAL(6,2),   -- Governance: board accountability
    hleg_rec10_score        DECIMAL(6,2),   -- Fossil Fuels: phase-out commitment
    -- Overall
    overall_credibility_score DECIMAL(6,2),
    credibility_tier        VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    sub_criteria_results    JSONB           DEFAULT '{}',
    recommendations         TEXT,
    improvement_priorities  JSONB           DEFAULT '[]',
    science_validation      BOOLEAN,
    governance_maturity     VARCHAR(30),
    fossil_fuel_exposure    JSONB           DEFAULT '{}',
    assessment_methodology  VARCHAR(100)    DEFAULT 'HLEG_INTEGRITY_MATTERS',
    assessor_id             UUID,
    assessor_notes          TEXT,
    warnings                TEXT[]          DEFAULT '{}',
    errors                  TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_ca_tier CHECK (
        credibility_tier IN ('STRONG', 'ADEQUATE', 'WEAK', 'INSUFFICIENT', 'PENDING')
    ),
    CONSTRAINT chk_p025_ca_governance CHECK (
        governance_maturity IS NULL OR governance_maturity IN ('ADVANCED', 'ESTABLISHED', 'DEVELOPING', 'INITIAL', 'NONE')
    ),
    CONSTRAINT chk_p025_ca_overall CHECK (
        overall_credibility_score IS NULL OR (overall_credibility_score >= 0 AND overall_credibility_score <= 100)
    ),
    CONSTRAINT chk_p025_ca_rec1 CHECK (hleg_rec1_score IS NULL OR (hleg_rec1_score >= 0 AND hleg_rec1_score <= 100)),
    CONSTRAINT chk_p025_ca_rec2 CHECK (hleg_rec2_score IS NULL OR (hleg_rec2_score >= 0 AND hleg_rec2_score <= 100)),
    CONSTRAINT chk_p025_ca_rec3 CHECK (hleg_rec3_score IS NULL OR (hleg_rec3_score >= 0 AND hleg_rec3_score <= 100)),
    CONSTRAINT chk_p025_ca_rec4 CHECK (hleg_rec4_score IS NULL OR (hleg_rec4_score >= 0 AND hleg_rec4_score <= 100)),
    CONSTRAINT chk_p025_ca_rec5 CHECK (hleg_rec5_score IS NULL OR (hleg_rec5_score >= 0 AND hleg_rec5_score <= 100)),
    CONSTRAINT chk_p025_ca_rec6 CHECK (hleg_rec6_score IS NULL OR (hleg_rec6_score >= 0 AND hleg_rec6_score <= 100)),
    CONSTRAINT chk_p025_ca_rec7 CHECK (hleg_rec7_score IS NULL OR (hleg_rec7_score >= 0 AND hleg_rec7_score <= 100)),
    CONSTRAINT chk_p025_ca_rec8 CHECK (hleg_rec8_score IS NULL OR (hleg_rec8_score >= 0 AND hleg_rec8_score <= 100)),
    CONSTRAINT chk_p025_ca_rec9 CHECK (hleg_rec9_score IS NULL OR (hleg_rec9_score >= 0 AND hleg_rec9_score <= 100)),
    CONSTRAINT chk_p025_ca_rec10 CHECK (hleg_rec10_score IS NULL OR (hleg_rec10_score >= 0 AND hleg_rec10_score <= 100))
);

-- ---------------------------------------------------------------------------
-- Indexes for credibility_assessments
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_ca_org             ON pack025_race_to_zero.credibility_assessments(org_id);
CREATE INDEX idx_p025_ca_pledge          ON pack025_race_to_zero.credibility_assessments(pledge_id);
CREATE INDEX idx_p025_ca_tenant          ON pack025_race_to_zero.credibility_assessments(tenant_id);
CREATE INDEX idx_p025_ca_date            ON pack025_race_to_zero.credibility_assessments(assessment_date);
CREATE INDEX idx_p025_ca_tier            ON pack025_race_to_zero.credibility_assessments(credibility_tier);
CREATE INDEX idx_p025_ca_overall         ON pack025_race_to_zero.credibility_assessments(overall_credibility_score);
CREATE INDEX idx_p025_ca_governance      ON pack025_race_to_zero.credibility_assessments(governance_maturity);
CREATE INDEX idx_p025_ca_science         ON pack025_race_to_zero.credibility_assessments(science_validation);
CREATE INDEX idx_p025_ca_created         ON pack025_race_to_zero.credibility_assessments(created_at DESC);
CREATE INDEX idx_p025_ca_sub_criteria    ON pack025_race_to_zero.credibility_assessments USING GIN(sub_criteria_results);
CREATE INDEX idx_p025_ca_priorities      ON pack025_race_to_zero.credibility_assessments USING GIN(improvement_priorities);
CREATE INDEX idx_p025_ca_fossil          ON pack025_race_to_zero.credibility_assessments USING GIN(fossil_fuel_exposure);
CREATE INDEX idx_p025_ca_metadata        ON pack025_race_to_zero.credibility_assessments USING GIN(metadata);

-- =============================================================================
-- Table 2: pack025_race_to_zero.lobbying_assessments
-- =============================================================================
-- Lobbying and climate advocacy alignment assessment for HLEG Rec 4
-- with trade association scoring and advocacy tracking.

CREATE TABLE pack025_race_to_zero.lobbying_assessments (
    assessment_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    credibility_id          UUID            REFERENCES pack025_race_to_zero.credibility_assessments(assessment_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    assessment_date         DATE            NOT NULL,
    lobbying_alignment      VARCHAR(30)     NOT NULL DEFAULT 'UNKNOWN',
    climate_advocacy_score  DECIMAL(6,2),
    trade_association_alignment DECIMAL(6,2),
    -- Detail fields
    direct_lobbying_review  JSONB           DEFAULT '{}',
    trade_associations      JSONB           DEFAULT '[]',
    industry_groups         JSONB           DEFAULT '[]',
    political_donations     JSONB           DEFAULT '{}',
    public_policy_positions JSONB           DEFAULT '[]',
    anti_climate_activities BOOLEAN         DEFAULT FALSE,
    anti_climate_details    TEXT,
    paris_alignment_statement BOOLEAN       DEFAULT FALSE,
    fossil_fuel_lobbying    BOOLEAN         DEFAULT FALSE,
    transparency_disclosure BOOLEAN         DEFAULT FALSE,
    disclosure_url          TEXT,
    overall_risk_level      VARCHAR(20)     DEFAULT 'MEDIUM',
    recommendations         TEXT[],
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_la_alignment CHECK (
        lobbying_alignment IN ('ALIGNED', 'PARTIALLY_ALIGNED', 'MISALIGNED', 'UNKNOWN')
    ),
    CONSTRAINT chk_p025_la_advocacy CHECK (
        climate_advocacy_score IS NULL OR (climate_advocacy_score >= 0 AND climate_advocacy_score <= 100)
    ),
    CONSTRAINT chk_p025_la_trade_assoc CHECK (
        trade_association_alignment IS NULL OR (trade_association_alignment >= 0 AND trade_association_alignment <= 100)
    ),
    CONSTRAINT chk_p025_la_risk CHECK (
        overall_risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for lobbying_assessments
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_la_org             ON pack025_race_to_zero.lobbying_assessments(org_id);
CREATE INDEX idx_p025_la_cred            ON pack025_race_to_zero.lobbying_assessments(credibility_id);
CREATE INDEX idx_p025_la_tenant          ON pack025_race_to_zero.lobbying_assessments(tenant_id);
CREATE INDEX idx_p025_la_date            ON pack025_race_to_zero.lobbying_assessments(assessment_date);
CREATE INDEX idx_p025_la_alignment       ON pack025_race_to_zero.lobbying_assessments(lobbying_alignment);
CREATE INDEX idx_p025_la_advocacy        ON pack025_race_to_zero.lobbying_assessments(climate_advocacy_score);
CREATE INDEX idx_p025_la_trade           ON pack025_race_to_zero.lobbying_assessments(trade_association_alignment);
CREATE INDEX idx_p025_la_risk            ON pack025_race_to_zero.lobbying_assessments(overall_risk_level);
CREATE INDEX idx_p025_la_fossil          ON pack025_race_to_zero.lobbying_assessments(fossil_fuel_lobbying);
CREATE INDEX idx_p025_la_created         ON pack025_race_to_zero.lobbying_assessments(created_at DESC);
CREATE INDEX idx_p025_la_trade_json      ON pack025_race_to_zero.lobbying_assessments USING GIN(trade_associations);
CREATE INDEX idx_p025_la_direct          ON pack025_race_to_zero.lobbying_assessments USING GIN(direct_lobbying_review);
CREATE INDEX idx_p025_la_positions       ON pack025_race_to_zero.lobbying_assessments USING GIN(public_policy_positions);
CREATE INDEX idx_p025_la_metadata        ON pack025_race_to_zero.lobbying_assessments USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p025_credibility_updated
    BEFORE UPDATE ON pack025_race_to_zero.credibility_assessments
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

CREATE TRIGGER trg_p025_lobbying_updated
    BEFORE UPDATE ON pack025_race_to_zero.lobbying_assessments
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack025_race_to_zero.credibility_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack025_race_to_zero.lobbying_assessments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p025_ca_tenant_isolation
    ON pack025_race_to_zero.credibility_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_ca_service_bypass
    ON pack025_race_to_zero.credibility_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p025_la_tenant_isolation
    ON pack025_race_to_zero.lobbying_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_la_service_bypass
    ON pack025_race_to_zero.lobbying_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.credibility_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.lobbying_assessments TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack025_race_to_zero.credibility_assessments IS
    'HLEG Integrity Matters 10-recommendation credibility assessment with individual scores, overall tier, and improvement plan.';
COMMENT ON TABLE pack025_race_to_zero.lobbying_assessments IS
    'Lobbying and climate advocacy alignment assessment with trade association scoring and fossil fuel lobbying tracking.';

COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.hleg_rec1_score IS 'Rec 1 - Ambition: net-zero pledge quality score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.hleg_rec2_score IS 'Rec 2 - Integrity: science-based target score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.hleg_rec3_score IS 'Rec 3 - Credibility: no offsets for scope 1/2 score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.hleg_rec4_score IS 'Rec 4 - Accountability: external verification score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.hleg_rec5_score IS 'Rec 5 - Just Transition: social equity score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.hleg_rec6_score IS 'Rec 6 - Finance: climate finance alignment score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.hleg_rec7_score IS 'Rec 7 - Transparency: public disclosure score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.hleg_rec8_score IS 'Rec 8 - Scope: material emissions coverage score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.hleg_rec9_score IS 'Rec 9 - Governance: board accountability score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.hleg_rec10_score IS 'Rec 10 - Fossil Fuels: phase-out commitment score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.credibility_assessments.credibility_tier IS 'Overall credibility tier: STRONG, ADEQUATE, WEAK, INSUFFICIENT, PENDING.';
COMMENT ON COLUMN pack025_race_to_zero.lobbying_assessments.lobbying_alignment IS 'Overall lobbying alignment: ALIGNED, PARTIALLY_ALIGNED, MISALIGNED, UNKNOWN.';
COMMENT ON COLUMN pack025_race_to_zero.lobbying_assessments.climate_advocacy_score IS 'Climate advocacy score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.lobbying_assessments.trade_association_alignment IS 'Trade association alignment score (0-100).';
