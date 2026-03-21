-- =============================================================================
-- V157: PACK-025 Race to Zero - Readiness, Audit, Workflows & Views
-- =============================================================================
-- Pack:         PACK-025 (Race to Zero Pack)
-- Migration:    010 of 010
-- Date:         March 2026
--
-- 8-dimension readiness scoring, SHA-256 provenance audit trail, and three
-- dashboard views: Race to Zero Dashboard, Credibility Leaderboard, and
-- Sector Benchmark aggregation.
--
-- Tables (2):
--   1. pack025_race_to_zero.readiness_scores
--   2. pack025_race_to_zero.audit_trail
--
-- Views (3):
--   1. pack025_race_to_zero.v_race_to_zero_dashboard
--   2. pack025_race_to_zero.v_credibility_leaderboard
--   3. pack025_race_to_zero.v_sector_benchmark
--
-- Previous: V156__pack025_race_to_zero_009.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack025_race_to_zero.readiness_scores
-- =============================================================================
-- 8-dimension readiness scoring covering pledge strength, starting line,
-- target ambition, action plan quality, progress trajectory, sector alignment,
-- partnership engagement, and HLEG credibility.

CREATE TABLE pack025_race_to_zero.readiness_scores (
    score_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    pledge_id               UUID            REFERENCES pack025_race_to_zero.pledges(pledge_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    assessment_date         DATE            NOT NULL,
    -- 8 readiness dimensions (each 0-100)
    dimension1_score        DECIMAL(6,2),   -- Pledge Strength
    dimension2_score        DECIMAL(6,2),   -- Starting Line Compliance
    dimension3_score        DECIMAL(6,2),   -- Target Ambition
    dimension4_score        DECIMAL(6,2),   -- Action Plan Quality
    dimension5_score        DECIMAL(6,2),   -- Progress Trajectory
    dimension6_score        DECIMAL(6,2),   -- Sector Alignment
    dimension7_score        DECIMAL(6,2),   -- Partnership Engagement
    dimension8_score        DECIMAL(6,2),   -- HLEG Credibility
    -- Composite
    composite_score         DECIMAL(6,2)    NOT NULL,
    readiness_level         VARCHAR(30)     NOT NULL DEFAULT 'NOT_READY',
    timeline_months         INTEGER,
    -- Detail
    rag_status              VARCHAR(10),
    dimension_weights       JSONB           DEFAULT '{}',
    phase_gates             JSONB           DEFAULT '{}',
    critical_gaps           TEXT[]          DEFAULT '{}',
    improvement_priorities  JSONB           DEFAULT '[]',
    next_milestone          TEXT,
    next_milestone_date     DATE,
    assessor_id             UUID,
    assessor_notes          TEXT,
    warnings                TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_rs_readiness CHECK (
        readiness_level IN ('CAMPAIGN_READY', 'MOSTLY_READY', 'PARTIALLY_READY',
                            'SIGNIFICANT_GAPS', 'NOT_READY')
    ),
    CONSTRAINT chk_p025_rs_rag CHECK (
        rag_status IS NULL OR rag_status IN ('GREEN', 'AMBER', 'RED')
    ),
    CONSTRAINT chk_p025_rs_composite CHECK (
        composite_score >= 0 AND composite_score <= 100
    ),
    CONSTRAINT chk_p025_rs_d1 CHECK (dimension1_score IS NULL OR (dimension1_score >= 0 AND dimension1_score <= 100)),
    CONSTRAINT chk_p025_rs_d2 CHECK (dimension2_score IS NULL OR (dimension2_score >= 0 AND dimension2_score <= 100)),
    CONSTRAINT chk_p025_rs_d3 CHECK (dimension3_score IS NULL OR (dimension3_score >= 0 AND dimension3_score <= 100)),
    CONSTRAINT chk_p025_rs_d4 CHECK (dimension4_score IS NULL OR (dimension4_score >= 0 AND dimension4_score <= 100)),
    CONSTRAINT chk_p025_rs_d5 CHECK (dimension5_score IS NULL OR (dimension5_score >= 0 AND dimension5_score <= 100)),
    CONSTRAINT chk_p025_rs_d6 CHECK (dimension6_score IS NULL OR (dimension6_score >= 0 AND dimension6_score <= 100)),
    CONSTRAINT chk_p025_rs_d7 CHECK (dimension7_score IS NULL OR (dimension7_score >= 0 AND dimension7_score <= 100)),
    CONSTRAINT chk_p025_rs_d8 CHECK (dimension8_score IS NULL OR (dimension8_score >= 0 AND dimension8_score <= 100)),
    CONSTRAINT chk_p025_rs_timeline CHECK (
        timeline_months IS NULL OR (timeline_months >= 0 AND timeline_months <= 120)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for readiness_scores
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_rs_org             ON pack025_race_to_zero.readiness_scores(org_id);
CREATE INDEX idx_p025_rs_pledge          ON pack025_race_to_zero.readiness_scores(pledge_id);
CREATE INDEX idx_p025_rs_tenant          ON pack025_race_to_zero.readiness_scores(tenant_id);
CREATE INDEX idx_p025_rs_date            ON pack025_race_to_zero.readiness_scores(assessment_date);
CREATE INDEX idx_p025_rs_level           ON pack025_race_to_zero.readiness_scores(readiness_level);
CREATE INDEX idx_p025_rs_composite       ON pack025_race_to_zero.readiness_scores(composite_score DESC);
CREATE INDEX idx_p025_rs_rag             ON pack025_race_to_zero.readiness_scores(rag_status);
CREATE INDEX idx_p025_rs_created         ON pack025_race_to_zero.readiness_scores(created_at DESC);
CREATE INDEX idx_p025_rs_weights         ON pack025_race_to_zero.readiness_scores USING GIN(dimension_weights);
CREATE INDEX idx_p025_rs_phases          ON pack025_race_to_zero.readiness_scores USING GIN(phase_gates);
CREATE INDEX idx_p025_rs_priorities      ON pack025_race_to_zero.readiness_scores USING GIN(improvement_priorities);
CREATE INDEX idx_p025_rs_metadata        ON pack025_race_to_zero.readiness_scores USING GIN(metadata);

-- =============================================================================
-- Table 2: pack025_race_to_zero.audit_trail
-- =============================================================================
-- SHA-256 provenance audit trail for all Race to Zero operations with
-- event tracking, user attribution, and change logging.

CREATE TABLE pack025_race_to_zero.audit_trail (
    audit_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    event_type              VARCHAR(50)     NOT NULL,
    event_date              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    user_id                 UUID,
    user_email              VARCHAR(255),
    entity_type             VARCHAR(50)     NOT NULL,
    entity_id               UUID            NOT NULL,
    action                  VARCHAR(50)     NOT NULL,
    changes_json            JSONB           NOT NULL DEFAULT '{}',
    previous_state          JSONB           DEFAULT '{}',
    new_state               JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    -- Context
    engine_name             VARCHAR(100),
    engine_version          VARCHAR(30),
    workflow_name           VARCHAR(100),
    workflow_phase          VARCHAR(50),
    execution_time_ms       DECIMAL(12,2),
    ip_address              VARCHAR(45),
    user_agent              TEXT,
    correlation_id          UUID,
    -- Status
    status                  VARCHAR(30)     DEFAULT 'success',
    error_message           TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_at_event_type CHECK (
        event_type IN ('PLEDGE', 'ASSESSMENT', 'TARGET', 'ACTION_PLAN', 'REPORT',
                        'SECTOR', 'PARTNERSHIP', 'CREDIBILITY', 'SUBMISSION',
                        'VERIFICATION', 'READINESS', 'SYSTEM')
    ),
    CONSTRAINT chk_p025_at_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'VALIDATE',
                   'SUBMIT', 'VERIFY', 'APPROVE', 'REJECT', 'EXPORT', 'IMPORT')
    ),
    CONSTRAINT chk_p025_at_status CHECK (
        status IN ('success', 'failure', 'warning', 'skipped')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for audit_trail
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_at_org             ON pack025_race_to_zero.audit_trail(org_id);
CREATE INDEX idx_p025_at_tenant          ON pack025_race_to_zero.audit_trail(tenant_id);
CREATE INDEX idx_p025_at_event_type      ON pack025_race_to_zero.audit_trail(event_type);
CREATE INDEX idx_p025_at_event_date      ON pack025_race_to_zero.audit_trail(event_date DESC);
CREATE INDEX idx_p025_at_user            ON pack025_race_to_zero.audit_trail(user_id);
CREATE INDEX idx_p025_at_entity          ON pack025_race_to_zero.audit_trail(entity_type, entity_id);
CREATE INDEX idx_p025_at_action          ON pack025_race_to_zero.audit_trail(action);
CREATE INDEX idx_p025_at_provenance      ON pack025_race_to_zero.audit_trail(provenance_hash);
CREATE INDEX idx_p025_at_correlation     ON pack025_race_to_zero.audit_trail(correlation_id);
CREATE INDEX idx_p025_at_engine          ON pack025_race_to_zero.audit_trail(engine_name);
CREATE INDEX idx_p025_at_workflow        ON pack025_race_to_zero.audit_trail(workflow_name);
CREATE INDEX idx_p025_at_status          ON pack025_race_to_zero.audit_trail(status);
CREATE INDEX idx_p025_at_created         ON pack025_race_to_zero.audit_trail(created_at DESC);
CREATE INDEX idx_p025_at_changes         ON pack025_race_to_zero.audit_trail USING GIN(changes_json);
CREATE INDEX idx_p025_at_metadata        ON pack025_race_to_zero.audit_trail USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p025_readiness_scores_updated
    BEFORE UPDATE ON pack025_race_to_zero.readiness_scores
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

-- Note: audit_trail is append-only, no update trigger needed

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack025_race_to_zero.readiness_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack025_race_to_zero.audit_trail ENABLE ROW LEVEL SECURITY;

CREATE POLICY p025_rs_tenant_isolation
    ON pack025_race_to_zero.readiness_scores
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_rs_service_bypass
    ON pack025_race_to_zero.readiness_scores
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p025_at_tenant_isolation
    ON pack025_race_to_zero.audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_at_service_bypass
    ON pack025_race_to_zero.audit_trail
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.readiness_scores TO PUBLIC;
GRANT SELECT, INSERT ON pack025_race_to_zero.audit_trail TO PUBLIC;

-- =============================================================================
-- View 1: v_race_to_zero_dashboard
-- =============================================================================
-- Organization summary view with latest pledge, assessment, and readiness
-- status for the Race to Zero dashboard.

CREATE OR REPLACE VIEW pack025_race_to_zero.v_race_to_zero_dashboard AS
SELECT
    op.org_id,
    op.tenant_id,
    op.name                     AS organization_name,
    op.sector_nace,
    op.actor_type,
    op.country,
    op.baseline_year,
    op.total_baseline_tco2e,
    op.profile_status,
    -- Latest pledge
    p.pledge_id,
    p.pledge_date,
    p.status                    AS pledge_status,
    p.quality_rating            AS pledge_quality,
    p.eligibility_score,
    p.target_year_interim,
    p.target_year_longterm,
    -- Latest starting line
    sl.overall_status           AS starting_line_status,
    sl.overall_compliance       AS starting_line_score,
    -- Latest target
    it.pathway_alignment        AS target_alignment,
    it.temperature_score,
    it.reduction_pct            AS target_reduction_pct,
    it.ipcc_compliance,
    -- Latest annual report
    ar.reporting_year           AS latest_report_year,
    ar.total_actual_tco2e       AS latest_emissions,
    ar.on_track_status,
    ar.verification_status,
    -- Latest credibility
    ca.credibility_tier,
    ca.overall_credibility_score AS credibility_score,
    -- Latest readiness
    rs.composite_score          AS readiness_score,
    rs.readiness_level,
    rs.rag_status
FROM pack025_race_to_zero.organization_profiles op
LEFT JOIN LATERAL (
    SELECT * FROM pack025_race_to_zero.pledges
    WHERE org_id = op.org_id
    ORDER BY created_at DESC LIMIT 1
) p ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM pack025_race_to_zero.starting_line_assessments
    WHERE org_id = op.org_id
    ORDER BY assessment_date DESC LIMIT 1
) sl ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM pack025_race_to_zero.interim_targets
    WHERE org_id = op.org_id
    ORDER BY created_at DESC LIMIT 1
) it ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM pack025_race_to_zero.annual_reports
    WHERE org_id = op.org_id
    ORDER BY reporting_year DESC LIMIT 1
) ar ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM pack025_race_to_zero.credibility_assessments
    WHERE org_id = op.org_id
    ORDER BY assessment_date DESC LIMIT 1
) ca ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM pack025_race_to_zero.readiness_scores
    WHERE org_id = op.org_id
    ORDER BY assessment_date DESC LIMIT 1
) rs ON TRUE;

-- =============================================================================
-- View 2: v_credibility_leaderboard
-- =============================================================================
-- Top organizations ranked by credibility score with HLEG recommendation
-- breakdown for benchmarking and comparison.

CREATE OR REPLACE VIEW pack025_race_to_zero.v_credibility_leaderboard AS
SELECT
    op.org_id,
    op.tenant_id,
    op.name                     AS organization_name,
    op.actor_type,
    op.sector_nace,
    op.country,
    ca.assessment_date,
    ca.overall_credibility_score,
    ca.credibility_tier,
    ca.hleg_rec1_score,
    ca.hleg_rec2_score,
    ca.hleg_rec3_score,
    ca.hleg_rec4_score,
    ca.hleg_rec5_score,
    ca.hleg_rec6_score,
    ca.hleg_rec7_score,
    ca.hleg_rec8_score,
    ca.hleg_rec9_score,
    ca.hleg_rec10_score,
    ca.governance_maturity,
    ca.science_validation,
    RANK() OVER (
        PARTITION BY op.tenant_id
        ORDER BY ca.overall_credibility_score DESC NULLS LAST
    ) AS credibility_rank
FROM pack025_race_to_zero.organization_profiles op
INNER JOIN LATERAL (
    SELECT * FROM pack025_race_to_zero.credibility_assessments
    WHERE org_id = op.org_id
    ORDER BY assessment_date DESC LIMIT 1
) ca ON TRUE
WHERE op.profile_status = 'active';

-- =============================================================================
-- View 3: v_sector_benchmark
-- =============================================================================
-- Sector-wide emission aggregates comparing average entity intensity
-- against pathway benchmarks for sector-level monitoring.

CREATE OR REPLACE VIEW pack025_race_to_zero.v_sector_benchmark AS
SELECT
    op.sector_nace,
    MAX(sp.sector_name)                             AS sector_name,
    op.tenant_id,
    COUNT(DISTINCT op.org_id)                       AS org_count,
    AVG(op.total_baseline_tco2e)                    AS avg_baseline_tco2e,
    SUM(op.total_baseline_tco2e)                    AS total_baseline_tco2e,
    AVG(sa.baseline_intensity)                      AS avg_entity_intensity,
    AVG(sa.benchmark_intensity)                     AS avg_benchmark_intensity,
    AVG(sa.gap_pct)                                 AS avg_gap_pct,
    COUNT(CASE WHEN sa.alignment_status = 'ALIGNED' THEN 1 END)    AS aligned_count,
    COUNT(CASE WHEN sa.alignment_status = 'MISALIGNED' THEN 1 END) AS misaligned_count,
    AVG(ca.overall_credibility_score)               AS avg_credibility_score,
    AVG(rs.composite_score)                         AS avg_readiness_score,
    MAX(sp.benchmark_emissions_intensity)            AS benchmark_2030_intensity,
    MAX(sp.technology_trl)                           AS max_technology_trl,
    MAX(sp.policy_requirements)                      AS policy_requirements
FROM pack025_race_to_zero.organization_profiles op
LEFT JOIN pack025_race_to_zero.sector_pathways sp
    ON op.sector_nace = sp.sector_nace AND sp.year = 2030 AND sp.pathway_source = 'IEA_NZE'
LEFT JOIN LATERAL (
    SELECT * FROM pack025_race_to_zero.sector_alignment
    WHERE org_id = op.org_id
    ORDER BY assessment_date DESC LIMIT 1
) sa ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM pack025_race_to_zero.credibility_assessments
    WHERE org_id = op.org_id
    ORDER BY assessment_date DESC LIMIT 1
) ca ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM pack025_race_to_zero.readiness_scores
    WHERE org_id = op.org_id
    ORDER BY assessment_date DESC LIMIT 1
) rs ON TRUE
WHERE op.profile_status = 'active'
GROUP BY op.sector_nace, op.tenant_id;

-- ---------------------------------------------------------------------------
-- Comments on tables and views
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack025_race_to_zero.readiness_scores IS
    '8-dimension readiness scoring for Race to Zero campaign participation across pledge, compliance, ambition, plan, progress, sector, partnership, and credibility.';
COMMENT ON TABLE pack025_race_to_zero.audit_trail IS
    'SHA-256 provenance audit trail for all Race to Zero operations with event tracking, user attribution, and change logging.';

COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.dimension1_score IS 'Dimension 1: Pledge Strength score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.dimension2_score IS 'Dimension 2: Starting Line Compliance score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.dimension3_score IS 'Dimension 3: Target Ambition score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.dimension4_score IS 'Dimension 4: Action Plan Quality score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.dimension5_score IS 'Dimension 5: Progress Trajectory score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.dimension6_score IS 'Dimension 6: Sector Alignment score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.dimension7_score IS 'Dimension 7: Partnership Engagement score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.dimension8_score IS 'Dimension 8: HLEG Credibility score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.composite_score IS 'Weighted composite readiness score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.readiness_level IS 'Readiness tier: CAMPAIGN_READY, MOSTLY_READY, PARTIALLY_READY, SIGNIFICANT_GAPS, NOT_READY.';
COMMENT ON COLUMN pack025_race_to_zero.readiness_scores.timeline_months IS 'Estimated months to reach CAMPAIGN_READY status (0-120).';
COMMENT ON COLUMN pack025_race_to_zero.audit_trail.audit_id IS 'Unique audit event identifier.';
COMMENT ON COLUMN pack025_race_to_zero.audit_trail.event_type IS 'Category of audited event.';
COMMENT ON COLUMN pack025_race_to_zero.audit_trail.changes_json IS 'JSONB diff of changes applied in this event.';
COMMENT ON COLUMN pack025_race_to_zero.audit_trail.provenance_hash IS 'SHA-256 hash chain for tamper-proof provenance.';

COMMENT ON VIEW pack025_race_to_zero.v_race_to_zero_dashboard IS
    'Organization summary view with latest pledge, assessment, target, report, credibility, and readiness status.';
COMMENT ON VIEW pack025_race_to_zero.v_credibility_leaderboard IS
    'Top organizations ranked by credibility score with HLEG recommendation breakdown.';
COMMENT ON VIEW pack025_race_to_zero.v_sector_benchmark IS
    'Sector-wide emission aggregates comparing average entity intensity against IEA NZE pathway benchmarks.';
