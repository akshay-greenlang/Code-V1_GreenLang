-- =============================================================================
-- V165: PACK-026 SME Net Zero - Audit Trails & Dashboard Views
-- =============================================================================
-- Pack:         PACK-026 (SME Net Zero Pack)
-- Migration:    008 of 008
-- Date:         March 2026
--
-- SHA-256 provenance audit trail for all SME Net Zero operations. Three
-- dashboard views: SME Dashboard (comprehensive summary), Grant Calendar
-- (upcoming deadlines), and Peer Leaderboard (top performers by sector/size).
--
-- Tables (1):
--   1. pack026_sme_net_zero.audit_trail
--
-- Views (3):
--   1. pack026_sme_net_zero.v_sme_dashboard
--   2. pack026_sme_net_zero.v_grant_calendar
--   3. pack026_sme_net_zero.v_peer_leaderboard
--
-- Previous: V164__PACK026_peer_benchmarking.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack026_sme_net_zero.audit_trail
-- =============================================================================
-- SHA-256 provenance audit trail for all SME Net Zero operations with event
-- tracking, user attribution, change logging, and tamper-proof hash chain.

CREATE TABLE pack026_sme_net_zero.audit_trail (
    audit_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    -- Event
    event_type              VARCHAR(50)     NOT NULL,
    event_date              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    user_id                 UUID,
    user_email              VARCHAR(255),
    -- Entity
    entity_type             VARCHAR(50)     NOT NULL,
    entity_id               UUID            NOT NULL,
    action                  VARCHAR(50)     NOT NULL,
    -- Changes
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
    CONSTRAINT chk_p026_at_event_type CHECK (
        event_type IN ('PROFILE', 'BASELINE', 'TARGET', 'QUICK_WIN', 'GRANT',
                       'CERTIFICATION', 'ACCOUNTING', 'SPEND', 'REVIEW',
                       'SNAPSHOT', 'BENCHMARK', 'RANKING', 'SYSTEM')
    ),
    CONSTRAINT chk_p026_at_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'VALIDATE',
                   'SUBMIT', 'VERIFY', 'APPROVE', 'REJECT', 'EXPORT',
                   'IMPORT', 'SYNC', 'MAP', 'RANK')
    ),
    CONSTRAINT chk_p026_at_entity_type CHECK (
        entity_type IN ('sme_profiles', 'sme_baselines', 'sme_targets',
                        'quick_wins_library', 'selected_actions', 'grant_programs',
                        'grant_applications', 'certifications', 'accounting_connections',
                        'spend_categories', 'spend_transactions', 'annual_reviews',
                        'quarterly_snapshots', 'peer_groups', 'peer_rankings')
    ),
    CONSTRAINT chk_p026_at_status CHECK (
        status IN ('success', 'failure', 'warning', 'skipped')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for audit_trail
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_at_sme              ON pack026_sme_net_zero.audit_trail(sme_id);
CREATE INDEX idx_p026_at_tenant           ON pack026_sme_net_zero.audit_trail(tenant_id);
CREATE INDEX idx_p026_at_event_type       ON pack026_sme_net_zero.audit_trail(event_type);
CREATE INDEX idx_p026_at_event_date       ON pack026_sme_net_zero.audit_trail(event_date DESC);
CREATE INDEX idx_p026_at_user             ON pack026_sme_net_zero.audit_trail(user_id);
CREATE INDEX idx_p026_at_entity           ON pack026_sme_net_zero.audit_trail(entity_type, entity_id);
CREATE INDEX idx_p026_at_action           ON pack026_sme_net_zero.audit_trail(action);
CREATE INDEX idx_p026_at_provenance       ON pack026_sme_net_zero.audit_trail(provenance_hash);
CREATE INDEX idx_p026_at_correlation      ON pack026_sme_net_zero.audit_trail(correlation_id);
CREATE INDEX idx_p026_at_engine           ON pack026_sme_net_zero.audit_trail(engine_name);
CREATE INDEX idx_p026_at_workflow         ON pack026_sme_net_zero.audit_trail(workflow_name);
CREATE INDEX idx_p026_at_status           ON pack026_sme_net_zero.audit_trail(status);
CREATE INDEX idx_p026_at_created          ON pack026_sme_net_zero.audit_trail(created_at DESC);
CREATE INDEX idx_p026_at_changes          ON pack026_sme_net_zero.audit_trail USING GIN(changes_json);
CREATE INDEX idx_p026_at_metadata         ON pack026_sme_net_zero.audit_trail USING GIN(metadata);

-- Note: audit_trail is append-only, no update trigger needed

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack026_sme_net_zero.audit_trail ENABLE ROW LEVEL SECURITY;

CREATE POLICY p026_at_tenant_isolation
    ON pack026_sme_net_zero.audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_at_service_bypass
    ON pack026_sme_net_zero.audit_trail
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT ON pack026_sme_net_zero.audit_trail TO PUBLIC;

-- =============================================================================
-- View 1: v_sme_dashboard
-- =============================================================================
-- Comprehensive SME summary view with latest baseline, targets, quick wins
-- progress, grant status, certification status, and peer ranking for the
-- main dashboard display.

CREATE OR REPLACE VIEW pack026_sme_net_zero.v_sme_dashboard AS
SELECT
    sp.sme_id,
    sp.tenant_id,
    sp.name                         AS sme_name,
    sp.industry_nace,
    sp.industry_description,
    sp.size_tier,
    sp.employee_count,
    sp.revenue_eur,
    sp.country,
    sp.region,
    sp.data_quality_tier,
    sp.accounting_software,
    sp.certification_pathway,
    sp.profile_status,
    -- Latest baseline
    bl.baseline_year,
    bl.scope1_tco2e                 AS baseline_scope1,
    bl.scope2_tco2e                 AS baseline_scope2,
    bl.scope3_tco2e                 AS baseline_scope3,
    bl.total_tco2e                  AS baseline_total,
    bl.intensity_per_employee       AS baseline_intensity,
    bl.industry_avg_comparison_percentile,
    bl.data_tier                    AS baseline_data_tier,
    -- Latest target
    tgt.target_year_interim,
    tgt.target_year_longterm,
    tgt.interim_reduction_pct,
    tgt.longterm_reduction_pct,
    tgt.pathway_type,
    tgt.compliance_status           AS target_compliance,
    tgt.sbti_aligned,
    tgt.climate_hub_committed,
    -- Quick wins summary
    qw.total_actions,
    qw.completed_actions,
    qw.in_progress_actions,
    qw.total_estimated_reduction,
    qw.total_actual_reduction,
    qw.total_actual_savings,
    -- Latest annual review
    ar.review_year                  AS latest_review_year,
    ar.actual_total                 AS latest_actual_tco2e,
    ar.on_track_status,
    ar.variance_pct,
    ar.reduction_from_baseline_pct,
    ar.grants_received_eur,
    ar.cost_savings_realized_eur,
    -- Certification count
    cert.active_certs,
    cert.latest_cert_type,
    cert.latest_cert_status,
    -- Grant applications
    ga.pending_applications,
    ga.total_funding_approved,
    -- Latest peer ranking
    pr.percentile                   AS peer_percentile,
    pr.peer_performance_tier,
    pr.trend                        AS peer_trend
FROM pack026_sme_net_zero.sme_profiles sp
-- Latest baseline
LEFT JOIN LATERAL (
    SELECT * FROM pack026_sme_net_zero.sme_baselines
    WHERE sme_id = sp.sme_id
    ORDER BY baseline_year DESC LIMIT 1
) bl ON TRUE
-- Latest target
LEFT JOIN LATERAL (
    SELECT * FROM pack026_sme_net_zero.sme_targets
    WHERE sme_id = sp.sme_id
    ORDER BY created_at DESC LIMIT 1
) tgt ON TRUE
-- Quick wins aggregate
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                    AS total_actions,
        COUNT(*) FILTER (WHERE implementation_status = 'COMPLETED') AS completed_actions,
        COUNT(*) FILTER (WHERE implementation_status = 'IN_PROGRESS') AS in_progress_actions,
        SUM(estimated_reduction_tco2e)                              AS total_estimated_reduction,
        SUM(actual_reduction_tco2e)                                 AS total_actual_reduction,
        SUM(actual_savings)                                         AS total_actual_savings
    FROM pack026_sme_net_zero.selected_actions
    WHERE sme_id = sp.sme_id
) qw ON TRUE
-- Latest annual review
LEFT JOIN LATERAL (
    SELECT * FROM pack026_sme_net_zero.annual_reviews
    WHERE sme_id = sp.sme_id
    ORDER BY review_year DESC LIMIT 1
) ar ON TRUE
-- Certification summary
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) FILTER (WHERE status IN ('CERTIFIED', 'VERIFIED')) AS active_certs,
        (SELECT certification_type FROM pack026_sme_net_zero.certifications
         WHERE sme_id = sp.sme_id ORDER BY created_at DESC LIMIT 1) AS latest_cert_type,
        (SELECT status FROM pack026_sme_net_zero.certifications
         WHERE sme_id = sp.sme_id ORDER BY created_at DESC LIMIT 1) AS latest_cert_status
    FROM pack026_sme_net_zero.certifications
    WHERE sme_id = sp.sme_id
) cert ON TRUE
-- Grant applications summary
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) FILTER (WHERE status IN ('DRAFT', 'SUBMITTED', 'UNDER_REVIEW')) AS pending_applications,
        SUM(funding_approved_eur) FILTER (WHERE status = 'APPROVED')             AS total_funding_approved
    FROM pack026_sme_net_zero.grant_applications
    WHERE sme_id = sp.sme_id
) ga ON TRUE
-- Latest peer ranking
LEFT JOIN LATERAL (
    SELECT * FROM pack026_sme_net_zero.peer_rankings
    WHERE sme_id = sp.sme_id
    ORDER BY ranking_date DESC LIMIT 1
) pr ON TRUE;

-- =============================================================================
-- View 2: v_grant_calendar
-- =============================================================================
-- Upcoming grant program deadlines sorted by deadline date for the grant
-- discovery and application planning calendar.

CREATE OR REPLACE VIEW pack026_sme_net_zero.v_grant_calendar AS
SELECT
    gp.grant_id,
    gp.name                         AS grant_name,
    gp.funding_body,
    gp.country,
    gp.region,
    gp.program_type,
    gp.min_funding_eur,
    gp.max_funding_eur,
    gp.co_funding_pct,
    gp.deadline,
    gp.opening_date,
    gp.recurring,
    gp.recurrence_note,
    gp.sector_codes,
    gp.size_tiers,
    gp.categories_funded,
    gp.contact_url,
    gp.application_url,
    gp.verified,
    gp.last_verified_date,
    -- Calculated fields
    CASE
        WHEN gp.deadline IS NULL THEN 'OPEN_ENDED'
        WHEN gp.deadline < CURRENT_DATE THEN 'EXPIRED'
        WHEN gp.deadline < CURRENT_DATE + INTERVAL '30 days' THEN 'CLOSING_SOON'
        WHEN gp.deadline < CURRENT_DATE + INTERVAL '90 days' THEN 'UPCOMING'
        ELSE 'FUTURE'
    END AS urgency_status,
    CASE
        WHEN gp.deadline IS NOT NULL THEN gp.deadline - CURRENT_DATE
        ELSE NULL
    END AS days_remaining,
    gp.description
FROM pack026_sme_net_zero.grant_programs gp
WHERE gp.active = TRUE
  AND (gp.deadline IS NULL OR gp.deadline >= CURRENT_DATE OR gp.recurring = TRUE)
ORDER BY
    CASE WHEN gp.deadline IS NULL THEN 1 ELSE 0 END,
    gp.deadline ASC;

-- =============================================================================
-- View 3: v_peer_leaderboard
-- =============================================================================
-- Top performing SMEs ranked by peer percentile within their sector and size
-- tier for competitive benchmarking and motivation.

CREATE OR REPLACE VIEW pack026_sme_net_zero.v_peer_leaderboard AS
SELECT
    sp.sme_id,
    sp.tenant_id,
    sp.name                         AS sme_name,
    sp.industry_nace,
    sp.industry_description,
    sp.size_tier,
    sp.country,
    sp.region,
    -- Peer ranking
    pr.percentile,
    pr.peer_performance_tier,
    pr.sme_intensity_per_employee,
    pr.intensity_vs_avg_pct,
    pr.trend,
    pr.data_year,
    -- Peer group context
    pg.avg_intensity_per_employee   AS peer_avg_intensity,
    pg.median_intensity_per_employee AS peer_median_intensity,
    pg.top_quartile_intensity       AS peer_top_quartile,
    pg.sample_size                  AS peer_group_size,
    pg.median_reduction_rate        AS peer_avg_reduction_rate,
    -- Achievement summary
    bl.total_tco2e                  AS baseline_total,
    ar.actual_total                 AS latest_actual_total,
    ar.on_track_status,
    ar.reduction_from_baseline_pct,
    -- Quick wins completed
    qw.completed_actions,
    qw.total_actual_reduction       AS reduction_from_actions,
    -- Certifications
    cert.active_certs,
    -- Rank within tenant
    RANK() OVER (
        PARTITION BY sp.tenant_id, sp.industry_nace, sp.size_tier
        ORDER BY pr.percentile DESC NULLS LAST
    ) AS leaderboard_rank,
    -- Overall rank within tenant
    RANK() OVER (
        PARTITION BY sp.tenant_id
        ORDER BY pr.percentile DESC NULLS LAST
    ) AS overall_rank
FROM pack026_sme_net_zero.sme_profiles sp
-- Latest peer ranking
INNER JOIN LATERAL (
    SELECT * FROM pack026_sme_net_zero.peer_rankings
    WHERE sme_id = sp.sme_id
    ORDER BY ranking_date DESC LIMIT 1
) pr ON TRUE
-- Peer group
LEFT JOIN pack026_sme_net_zero.peer_groups pg
    ON pr.group_id = pg.group_id
-- Latest baseline
LEFT JOIN LATERAL (
    SELECT * FROM pack026_sme_net_zero.sme_baselines
    WHERE sme_id = sp.sme_id
    ORDER BY baseline_year DESC LIMIT 1
) bl ON TRUE
-- Latest annual review
LEFT JOIN LATERAL (
    SELECT * FROM pack026_sme_net_zero.annual_reviews
    WHERE sme_id = sp.sme_id
    ORDER BY review_year DESC LIMIT 1
) ar ON TRUE
-- Quick wins aggregate
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) FILTER (WHERE implementation_status = 'COMPLETED') AS completed_actions,
        SUM(actual_reduction_tco2e)                                 AS total_actual_reduction
    FROM pack026_sme_net_zero.selected_actions
    WHERE sme_id = sp.sme_id
) qw ON TRUE
-- Certification count
LEFT JOIN LATERAL (
    SELECT COUNT(*) FILTER (WHERE status IN ('CERTIFIED', 'VERIFIED')) AS active_certs
    FROM pack026_sme_net_zero.certifications
    WHERE sme_id = sp.sme_id
) cert ON TRUE
WHERE sp.profile_status = 'active';

-- ---------------------------------------------------------------------------
-- Comments on tables and views
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack026_sme_net_zero.audit_trail IS
    'SHA-256 provenance audit trail for all SME Net Zero operations with event tracking, user attribution, and tamper-proof change logging.';

COMMENT ON COLUMN pack026_sme_net_zero.audit_trail.audit_id IS 'Unique audit event identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.audit_trail.sme_id IS 'SME that this audit event relates to.';
COMMENT ON COLUMN pack026_sme_net_zero.audit_trail.event_type IS 'Category of audited event: PROFILE, BASELINE, TARGET, QUICK_WIN, GRANT, CERTIFICATION, ACCOUNTING, SPEND, REVIEW, SNAPSHOT, BENCHMARK, RANKING, SYSTEM.';
COMMENT ON COLUMN pack026_sme_net_zero.audit_trail.entity_type IS 'Database table name of the affected entity.';
COMMENT ON COLUMN pack026_sme_net_zero.audit_trail.action IS 'Operation performed: CREATE, UPDATE, DELETE, CALCULATE, VALIDATE, SUBMIT, VERIFY, APPROVE, REJECT, EXPORT, IMPORT, SYNC, MAP, RANK.';
COMMENT ON COLUMN pack026_sme_net_zero.audit_trail.changes_json IS 'JSONB diff of changes applied in this event.';
COMMENT ON COLUMN pack026_sme_net_zero.audit_trail.provenance_hash IS 'SHA-256 hash chain for tamper-proof provenance.';
COMMENT ON COLUMN pack026_sme_net_zero.audit_trail.correlation_id IS 'UUID to correlate related audit events across a single workflow execution.';

COMMENT ON VIEW pack026_sme_net_zero.v_sme_dashboard IS
    'Comprehensive SME summary view with latest baseline, targets, quick wins progress, grants, certifications, and peer ranking for the main dashboard.';
COMMENT ON VIEW pack026_sme_net_zero.v_grant_calendar IS
    'Upcoming grant program deadlines sorted by date with urgency status (CLOSING_SOON, UPCOMING, FUTURE, OPEN_ENDED) for grant discovery planning.';
COMMENT ON VIEW pack026_sme_net_zero.v_peer_leaderboard IS
    'Top performing SMEs ranked by peer percentile within their sector and size tier for competitive benchmarking and motivation.';
