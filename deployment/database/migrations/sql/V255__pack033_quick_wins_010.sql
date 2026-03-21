-- =============================================================================
-- V255: PACK-033 Quick Wins Identifier - Views, Indexes, RLS & Functions
-- =============================================================================
-- Pack:         PACK-033 (Quick Wins Identifier Pack)
-- Migration:    010 of 010
-- Date:         March 2026
--
-- Creates audit trail table, analytical views, composite indexes, and utility
-- functions for the Quick Wins Identifier Pack.
--
-- Tables (1):
--   1. pack033_quick_wins.pack033_audit_trail
--
-- Views (4):
--   1. pack033_quick_wins.v_quick_wins_summary
--   2. pack033_quick_wins.v_action_rankings
--   3. pack033_quick_wins.v_savings_progress
--   4. pack033_quick_wins.v_rebate_status
--
-- Functions (2):
--   1. pack033_quick_wins.fn_aggregate_scan_savings(UUID)
--   2. pack033_quick_wins.fn_calculate_portfolio_savings(UUID)
--
-- Previous: V254__pack033_quick_wins_009.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack033_quick_wins.pack033_audit_trail
-- =============================================================================
-- Audit trail for all PACK-033 entity changes with old/new values for
-- regulatory compliance and data governance.

CREATE TABLE pack033_quick_wins.pack033_audit_trail (
    entry_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id                 UUID,
    tenant_id               UUID            NOT NULL,
    action                  VARCHAR(50)     NOT NULL,
    entity_type             VARCHAR(100)    NOT NULL,
    entity_id               UUID            NOT NULL,
    old_values              JSONB,
    new_values              JSONB,
    user_id                 UUID,
    timestamp               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_audit_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'ARCHIVE', 'RESTORE', 'APPROVE', 'REJECT')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_audit_scan       ON pack033_quick_wins.pack033_audit_trail(scan_id);
CREATE INDEX idx_p033_audit_tenant     ON pack033_quick_wins.pack033_audit_trail(tenant_id);
CREATE INDEX idx_p033_audit_entity     ON pack033_quick_wins.pack033_audit_trail(entity_type, entity_id);
CREATE INDEX idx_p033_audit_user       ON pack033_quick_wins.pack033_audit_trail(user_id);
CREATE INDEX idx_p033_audit_timestamp  ON pack033_quick_wins.pack033_audit_trail(timestamp DESC);
CREATE INDEX idx_p033_audit_action     ON pack033_quick_wins.pack033_audit_trail(action);
CREATE INDEX idx_p033_audit_old_vals   ON pack033_quick_wins.pack033_audit_trail USING GIN(old_values);
CREATE INDEX idx_p033_audit_new_vals   ON pack033_quick_wins.pack033_audit_trail USING GIN(new_values);

-- ---------------------------------------------------------------------------
-- Row-Level Security for audit trail
-- ---------------------------------------------------------------------------
ALTER TABLE pack033_quick_wins.pack033_audit_trail ENABLE ROW LEVEL SECURITY;

CREATE POLICY p033_audit_tenant_isolation
    ON pack033_quick_wins.pack033_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p033_audit_service_bypass
    ON pack033_quick_wins.pack033_audit_trail
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.pack033_audit_trail TO PUBLIC;

-- =============================================================================
-- View 1: v_quick_wins_summary
-- =============================================================================
-- Comprehensive scan-level summary joining scans with aggregated results,
-- financial totals, and carbon impact summaries.

CREATE OR REPLACE VIEW pack033_quick_wins.v_quick_wins_summary AS
SELECT
    qs.scan_id,
    qs.tenant_id,
    qs.facility_id,
    qs.scan_type,
    qs.building_type,
    qs.scan_date,
    qs.status,
    qs.total_actions_found,
    qs.total_savings_kwh,
    qs.total_savings_cost,
    qs.total_co2e_reduction,
    -- Aggregated scan results
    COALESCE(sr.result_count, 0)               AS identified_actions,
    COALESCE(sr.avg_priority, 0)               AS avg_priority_score,
    COALESCE(sr.avg_payback, 0)                AS avg_payback_months,
    COALESCE(sr.total_impl_cost, 0)            AS total_implementation_cost,
    -- Financial summary
    COALESCE(pa.total_npv, 0)                  AS total_npv,
    COALESCE(pa.avg_roi, 0)                    AS avg_roi_pct,
    COALESCE(pa.avg_irr, 0)                    AS avg_irr,
    -- Carbon impact
    COALESCE(ci.total_annual_co2e, 0)          AS total_annual_co2e_reduction,
    COALESCE(ci.sbti_aligned_count, 0)         AS sbti_aligned_action_count,
    -- Progress
    COALESCE(ip.completed_count, 0)            AS completed_actions,
    COALESCE(ip.in_progress_count, 0)          AS in_progress_actions,
    qs.created_at
FROM pack033_quick_wins.quick_wins_scans qs
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                    AS result_count,
        AVG(priority_score)         AS avg_priority,
        AVG(payback_months)         AS avg_payback,
        SUM(implementation_cost)    AS total_impl_cost
    FROM pack033_quick_wins.scan_results r
    WHERE r.scan_id = qs.scan_id
) sr ON TRUE
LEFT JOIN LATERAL (
    SELECT
        SUM(npv)                    AS total_npv,
        AVG(roi_pct)                AS avg_roi,
        AVG(irr)                    AS avg_irr
    FROM pack033_quick_wins.payback_analyses p
    WHERE p.scan_id = qs.scan_id
) pa ON TRUE
LEFT JOIN LATERAL (
    SELECT
        SUM(annual_co2e_reduction)  AS total_annual_co2e,
        SUM(CASE WHEN sbti_aligned THEN 1 ELSE 0 END) AS sbti_aligned_count
    FROM pack033_quick_wins.carbon_impacts c
    WHERE c.scan_id = qs.scan_id
) ci ON TRUE
LEFT JOIN LATERAL (
    SELECT
        SUM(CASE WHEN status IN ('COMPLETED', 'VERIFIED') THEN 1 ELSE 0 END) AS completed_count,
        SUM(CASE WHEN status = 'IN_PROGRESS' THEN 1 ELSE 0 END) AS in_progress_count
    FROM pack033_quick_wins.implementation_progress ip2
    WHERE ip2.scan_id = qs.scan_id
) ip ON TRUE;

-- =============================================================================
-- View 2: v_action_rankings
-- =============================================================================
-- Ranked action view combining priority scores, financial metrics, and
-- savings estimates for decision support.

CREATE OR REPLACE VIEW pack033_quick_wins.v_action_rankings AS
SELECT
    ps.scan_id,
    ps.action_id,
    qs.tenant_id,
    qs.facility_id,
    sr.category,
    sr.subcategory,
    sr.description,
    sr.confidence_level,
    ps.cost_score,
    ps.savings_score,
    ps.risk_score,
    ps.disruption_score,
    ps.complexity_score,
    ps.co_benefits_score,
    ps.weighted_total,
    ps.rank,
    ps.pareto_optimal,
    sr.estimated_savings_kwh,
    sr.estimated_savings_cost,
    sr.estimated_co2e,
    sr.payback_months,
    sr.implementation_cost,
    pa.npv,
    pa.irr,
    pa.roi_pct,
    pa.simple_payback_years,
    ci.scope                                    AS carbon_scope,
    ci.annual_co2e_reduction,
    ci.sbti_aligned
FROM pack033_quick_wins.priority_scores ps
INNER JOIN pack033_quick_wins.quick_wins_scans qs
    ON ps.scan_id = qs.scan_id
LEFT JOIN pack033_quick_wins.scan_results sr
    ON ps.scan_id = sr.scan_id AND ps.action_id = sr.action_id
LEFT JOIN LATERAL (
    SELECT npv, irr, roi_pct, simple_payback_years
    FROM pack033_quick_wins.payback_analyses p
    WHERE p.scan_id = ps.scan_id AND p.action_id = ps.action_id
    ORDER BY p.created_at DESC
    LIMIT 1
) pa ON TRUE
LEFT JOIN LATERAL (
    SELECT scope, annual_co2e_reduction, sbti_aligned
    FROM pack033_quick_wins.carbon_impacts c
    WHERE c.scan_id = ps.scan_id AND c.action_id = ps.action_id
    ORDER BY c.created_at DESC
    LIMIT 1
) ci ON TRUE;

-- =============================================================================
-- View 3: v_savings_progress
-- =============================================================================
-- Progress tracking view comparing estimated vs. actual savings for each
-- implemented action with variance calculations.

CREATE OR REPLACE VIEW pack033_quick_wins.v_savings_progress AS
SELECT
    ip.tenant_id,
    ip.scan_id,
    ip.action_id,
    ip.status,
    ip.completion_pct,
    ip.planned_start_date,
    ip.planned_end_date,
    ip.actual_start_date,
    ip.actual_end_date,
    ip.actual_cost,
    sr.estimated_savings_kwh,
    sr.estimated_savings_cost,
    sr.implementation_cost                       AS estimated_cost,
    COALESCE(sa.total_verified_savings, 0)       AS total_verified_savings,
    COALESCE(sa.measurement_count, 0)            AS measurement_count,
    COALESCE(sa.latest_period_end, NULL)         AS latest_measurement_date,
    -- Variance: positive = better than expected
    CASE WHEN sr.estimated_savings_kwh > 0
        THEN ROUND(
            (COALESCE(sa.total_verified_savings, 0) - sr.estimated_savings_kwh)
            / sr.estimated_savings_kwh * 100, 1
        )
        ELSE NULL
    END                                          AS savings_variance_pct,
    -- Cost variance: positive = under budget
    CASE WHEN sr.implementation_cost > 0 AND ip.actual_cost IS NOT NULL
        THEN ROUND(
            (sr.implementation_cost - ip.actual_cost)
            / sr.implementation_cost * 100, 1
        )
        ELSE NULL
    END                                          AS cost_variance_pct
FROM pack033_quick_wins.implementation_progress ip
LEFT JOIN pack033_quick_wins.scan_results sr
    ON ip.scan_id = sr.scan_id AND ip.action_id = sr.action_id
LEFT JOIN LATERAL (
    SELECT
        SUM(verified_savings)   AS total_verified_savings,
        COUNT(*)                AS measurement_count,
        MAX(measurement_period_end) AS latest_period_end
    FROM pack033_quick_wins.savings_actuals sa2
    WHERE sa2.progress_id = ip.progress_id
) sa ON TRUE;

-- =============================================================================
-- View 4: v_rebate_status
-- =============================================================================
-- Rebate application status overview by tenant with program details and
-- financial summaries.

CREATE OR REPLACE VIEW pack033_quick_wins.v_rebate_status AS
SELECT
    ra.tenant_id,
    ra.application_id,
    ra.scan_id,
    ra.action_id,
    rp.utility_name,
    rp.utility_region,
    rp.program_name,
    rp.program_type,
    rp.measure_category,
    ra.status,
    ra.applied_date,
    ra.rebate_amount_requested,
    ra.rebate_amount_approved,
    ra.approval_date,
    rp.stacking_allowed,
    rp.application_deadline,
    -- Days until deadline
    CASE WHEN rp.application_deadline IS NOT NULL
        THEN rp.application_deadline - CURRENT_DATE
        ELSE NULL
    END                                          AS days_until_deadline,
    -- Approval rate
    CASE WHEN ra.rebate_amount_requested > 0 AND ra.rebate_amount_approved IS NOT NULL
        THEN ROUND(ra.rebate_amount_approved / ra.rebate_amount_requested * 100, 1)
        ELSE NULL
    END                                          AS approval_rate_pct,
    ra.notes,
    ra.created_at
FROM pack033_quick_wins.rebate_applications ra
INNER JOIN pack033_quick_wins.rebate_programs rp
    ON ra.program_id = rp.program_id;

-- ---------------------------------------------------------------------------
-- Grants on views
-- ---------------------------------------------------------------------------
GRANT SELECT ON pack033_quick_wins.v_quick_wins_summary TO PUBLIC;
GRANT SELECT ON pack033_quick_wins.v_action_rankings TO PUBLIC;
GRANT SELECT ON pack033_quick_wins.v_savings_progress TO PUBLIC;
GRANT SELECT ON pack033_quick_wins.v_rebate_status TO PUBLIC;

-- =============================================================================
-- Function 1: fn_aggregate_scan_savings
-- =============================================================================
-- Aggregates total savings for a given scan across all results and returns
-- a summary JSONB object. Used by application layer for scan totals refresh.

CREATE OR REPLACE FUNCTION pack033_quick_wins.fn_aggregate_scan_savings(
    p_scan_id UUID
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'scan_id', p_scan_id,
        'total_actions', COUNT(*),
        'total_savings_kwh', COALESCE(SUM(estimated_savings_kwh), 0),
        'total_savings_cost', COALESCE(SUM(estimated_savings_cost), 0),
        'total_co2e', COALESCE(SUM(estimated_co2e), 0),
        'total_implementation_cost', COALESCE(SUM(implementation_cost), 0),
        'avg_payback_months', COALESCE(AVG(payback_months), 0),
        'avg_priority_score', COALESCE(AVG(priority_score), 0),
        'high_confidence_count', COUNT(*) FILTER (WHERE confidence_level IN ('HIGH', 'VERIFIED')),
        'category_breakdown', (
            SELECT jsonb_agg(jsonb_build_object(
                'category', category,
                'count', cnt,
                'savings_kwh', savings
            ))
            FROM (
                SELECT category, COUNT(*) AS cnt, COALESCE(SUM(estimated_savings_kwh), 0) AS savings
                FROM pack033_quick_wins.scan_results
                WHERE scan_id = p_scan_id
                GROUP BY category
                ORDER BY savings DESC
            ) sub
        )
    ) INTO v_result
    FROM pack033_quick_wins.scan_results
    WHERE scan_id = p_scan_id;

    RETURN COALESCE(v_result, '{}'::JSONB);
END;
$$;

-- =============================================================================
-- Function 2: fn_calculate_portfolio_savings
-- =============================================================================
-- Calculates portfolio-level savings across all scans for a tenant. Returns
-- a summary JSONB with totals, averages, and implementation progress.

CREATE OR REPLACE FUNCTION pack033_quick_wins.fn_calculate_portfolio_savings(
    p_tenant_id UUID
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'tenant_id', p_tenant_id,
        'total_scans', COUNT(DISTINCT qs.scan_id),
        'total_facilities', COUNT(DISTINCT qs.facility_id),
        'total_actions_identified', COALESCE(SUM(qs.total_actions_found), 0),
        'total_savings_kwh', COALESCE(SUM(qs.total_savings_kwh), 0),
        'total_savings_cost', COALESCE(SUM(qs.total_savings_cost), 0),
        'total_co2e_reduction', COALESCE(SUM(qs.total_co2e_reduction), 0),
        'completed_actions', (
            SELECT COUNT(*) FROM pack033_quick_wins.implementation_progress ip
            WHERE ip.tenant_id = p_tenant_id AND ip.status IN ('COMPLETED', 'VERIFIED')
        ),
        'in_progress_actions', (
            SELECT COUNT(*) FROM pack033_quick_wins.implementation_progress ip
            WHERE ip.tenant_id = p_tenant_id AND ip.status = 'IN_PROGRESS'
        ),
        'total_verified_savings', (
            SELECT COALESCE(SUM(sa.verified_savings), 0)
            FROM pack033_quick_wins.savings_actuals sa
            JOIN pack033_quick_wins.implementation_progress ip ON sa.progress_id = ip.progress_id
            WHERE ip.tenant_id = p_tenant_id
        ),
        'total_rebates_approved', (
            SELECT COALESCE(SUM(ra.rebate_amount_approved), 0)
            FROM pack033_quick_wins.rebate_applications ra
            WHERE ra.tenant_id = p_tenant_id AND ra.status = 'APPROVED'
        )
    ) INTO v_result
    FROM pack033_quick_wins.quick_wins_scans qs
    WHERE qs.tenant_id = p_tenant_id;

    RETURN COALESCE(v_result, '{}'::JSONB);
END;
$$;

-- ---------------------------------------------------------------------------
-- Grants on functions
-- ---------------------------------------------------------------------------
GRANT EXECUTE ON FUNCTION pack033_quick_wins.fn_aggregate_scan_savings(UUID) TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack033_quick_wins.fn_calculate_portfolio_savings(UUID) TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack033_quick_wins.pack033_audit_trail IS
    'Audit trail for all PACK-033 entity changes with old/new values for regulatory compliance and data governance.';

COMMENT ON VIEW pack033_quick_wins.v_quick_wins_summary IS
    'Comprehensive scan-level summary joining scans with aggregated results, financial totals, and carbon impact summaries.';

COMMENT ON VIEW pack033_quick_wins.v_action_rankings IS
    'Ranked action view combining priority scores, financial metrics, and savings estimates for decision support.';

COMMENT ON VIEW pack033_quick_wins.v_savings_progress IS
    'Progress tracking view comparing estimated vs. actual savings with variance calculations for each implemented action.';

COMMENT ON VIEW pack033_quick_wins.v_rebate_status IS
    'Rebate application status overview with program details, financial summaries, and deadline tracking.';

COMMENT ON FUNCTION pack033_quick_wins.fn_aggregate_scan_savings(UUID) IS
    'Aggregates total savings for a given scan across all results and returns a summary JSONB object.';

COMMENT ON FUNCTION pack033_quick_wins.fn_calculate_portfolio_savings(UUID) IS
    'Calculates portfolio-level savings across all scans for a tenant with implementation progress and rebate totals.';

COMMENT ON COLUMN pack033_quick_wins.pack033_audit_trail.entry_id IS
    'Unique identifier for the audit trail entry.';
COMMENT ON COLUMN pack033_quick_wins.pack033_audit_trail.action IS
    'Type of action performed (CREATE, UPDATE, DELETE, ARCHIVE, RESTORE, APPROVE, REJECT).';
COMMENT ON COLUMN pack033_quick_wins.pack033_audit_trail.old_values IS
    'JSON snapshot of entity state before the change.';
COMMENT ON COLUMN pack033_quick_wins.pack033_audit_trail.new_values IS
    'JSON snapshot of entity state after the change.';
