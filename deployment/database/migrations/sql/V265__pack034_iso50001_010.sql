-- =============================================================================
-- V265: PACK-034 ISO 50001 Energy Management System - Views, Indexes, RLS,
--       Functions, Triggers
-- =============================================================================
-- Pack:         PACK-034 (ISO 50001 Energy Management System Pack)
-- Migration:    010 of 010
-- Date:         March 2026
--
-- Creates audit trail table, analytical views, utility functions, and
-- status-change audit triggers for the ISO 50001 EnMS Pack.
--
-- Tables (1):
--   1. pack034_iso50001.pack034_audit_trail
--
-- Views (6):
--   1. pack034_iso50001.v_enms_overview
--   2. pack034_iso50001.v_seu_pareto
--   3. pack034_iso50001.v_enpi_performance
--   4. pack034_iso50001.v_cusum_status
--   5. pack034_iso50001.v_compliance_summary
--   6. pack034_iso50001.v_action_plan_progress
--
-- Functions (3):
--   1. pack034_iso50001.fn_calculate_enpi_improvement(UUID)
--   2. pack034_iso50001.fn_check_cusum_alert(UUID)
--   3. pack034_iso50001.fn_update_compliance_score(UUID)
--
-- Triggers (1):
--   1. pack034_iso50001.fn_audit_status_change() + triggers on key tables
--
-- Previous: V264__pack034_iso50001_009.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack034_iso50001.pack034_audit_trail
-- =============================================================================
-- Audit trail for all PACK-034 entity changes with old/new values for
-- regulatory compliance, certification evidence, and data governance.

CREATE TABLE pack034_iso50001.pack034_audit_trail (
    entry_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID,
    organization_id             UUID            NOT NULL,
    action                      VARCHAR(50)     NOT NULL,
    entity_type                 VARCHAR(100)    NOT NULL,
    entity_id                   UUID            NOT NULL,
    old_values                  JSONB,
    new_values                  JSONB,
    user_id                     UUID,
    timestamp                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_audit_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'ARCHIVE', 'RESTORE', 'APPROVE', 'REJECT', 'STATUS_CHANGE')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_audit_enms         ON pack034_iso50001.pack034_audit_trail(enms_id);
CREATE INDEX idx_p034_audit_org          ON pack034_iso50001.pack034_audit_trail(organization_id);
CREATE INDEX idx_p034_audit_entity       ON pack034_iso50001.pack034_audit_trail(entity_type, entity_id);
CREATE INDEX idx_p034_audit_user         ON pack034_iso50001.pack034_audit_trail(user_id);
CREATE INDEX idx_p034_audit_timestamp    ON pack034_iso50001.pack034_audit_trail(timestamp DESC);
CREATE INDEX idx_p034_audit_action       ON pack034_iso50001.pack034_audit_trail(action);
CREATE INDEX idx_p034_audit_old_vals     ON pack034_iso50001.pack034_audit_trail USING GIN(old_values);
CREATE INDEX idx_p034_audit_new_vals     ON pack034_iso50001.pack034_audit_trail USING GIN(new_values);

-- ---------------------------------------------------------------------------
-- RLS for audit trail
-- ---------------------------------------------------------------------------
ALTER TABLE pack034_iso50001.pack034_audit_trail ENABLE ROW LEVEL SECURITY;

CREATE POLICY p034_audit_tenant_isolation
    ON pack034_iso50001.pack034_audit_trail
    USING (organization_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p034_audit_service_bypass
    ON pack034_iso50001.pack034_audit_trail
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants for audit trail
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT ON pack034_iso50001.pack034_audit_trail TO PUBLIC;

-- =============================================================================
-- View 1: v_enms_overview
-- =============================================================================
-- Comprehensive EnMS overview with scope counts, SEU summary, baseline status,
-- EnPI performance, and certification readiness.

CREATE OR REPLACE VIEW pack034_iso50001.v_enms_overview AS
SELECT
    ems.id                                          AS enms_id,
    ems.organization_id,
    ems.name,
    ems.enms_status,
    ems.certification_body,
    ems.certification_date,
    ems.next_surveillance_date,
    ems.pdca_cycle_count,
    -- Scope summary
    COALESCE(sc.included_count, 0)                  AS scope_items_included,
    COALESCE(sc.excluded_count, 0)                  AS scope_items_excluded,
    -- SEU summary
    COALESCE(seu.total_seus, 0)                     AS total_seus,
    COALESCE(seu.significant_seus, 0)               AS significant_seus,
    COALESCE(seu.total_consumption, 0)              AS total_seu_consumption_kwh,
    -- Baseline summary
    COALESCE(bl.approved_baselines, 0)              AS approved_baselines,
    COALESCE(bl.avg_r_squared, 0)                   AS avg_baseline_r_squared,
    -- EnPI summary
    COALESCE(enpi.total_enpis, 0)                   AS total_enpis,
    COALESCE(enpi.avg_improvement, 0)               AS avg_enpi_improvement_pct,
    -- Compliance summary
    COALESCE(ca.latest_score, 0)                    AS latest_compliance_score,
    COALESCE(ca.open_ncs, 0)                        AS open_nonconformities,
    -- Days until next surveillance
    CASE WHEN ems.next_surveillance_date IS NOT NULL
        THEN ems.next_surveillance_date - CURRENT_DATE
        ELSE NULL
    END                                             AS days_until_surveillance,
    ems.created_at
FROM pack034_iso50001.energy_management_systems ems
LEFT JOIN LATERAL (
    SELECT
        SUM(CASE WHEN included THEN 1 ELSE 0 END) AS included_count,
        SUM(CASE WHEN NOT included THEN 1 ELSE 0 END) AS excluded_count
    FROM pack034_iso50001.enms_scope s
    WHERE s.enms_id = ems.id
) sc ON TRUE
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                    AS total_seus,
        SUM(CASE WHEN is_significant THEN 1 ELSE 0 END) AS significant_seus,
        COALESCE(SUM(annual_consumption_kwh), 0)    AS total_consumption
    FROM pack034_iso50001.significant_energy_uses su
    WHERE su.enms_id = ems.id
) seu ON TRUE
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                    AS approved_baselines,
        AVG(r_squared)                              AS avg_r_squared
    FROM pack034_iso50001.energy_baselines b
    WHERE b.enms_id = ems.id AND b.status = 'approved'
) bl ON TRUE
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                    AS total_enpis,
        AVG(improvement_pct)                        AS avg_improvement
    FROM pack034_iso50001.energy_performance_indicators ep
    WHERE ep.enms_id = ems.id
) enpi ON TRUE
LEFT JOIN LATERAL (
    SELECT
        ca2.overall_score                           AS latest_score,
        (SELECT COUNT(*) FROM pack034_iso50001.nonconformities nc
         JOIN pack034_iso50001.compliance_assessments ca3 ON nc.assessment_id = ca3.id
         WHERE ca3.enms_id = ems.id AND nc.status IN ('open', 'corrective_action')
        )                                           AS open_ncs
    FROM pack034_iso50001.compliance_assessments ca2
    WHERE ca2.enms_id = ems.id
    ORDER BY ca2.assessment_date DESC
    LIMIT 1
) ca ON TRUE;

-- =============================================================================
-- View 2: v_seu_pareto
-- =============================================================================
-- SEU Pareto analysis showing cumulative consumption percentage for
-- identifying the vital few energy uses (typically 80/20 rule).

CREATE OR REPLACE VIEW pack034_iso50001.v_seu_pareto AS
SELECT
    seu.id                                          AS seu_id,
    seu.enms_id,
    ems.organization_id,
    seu.seu_name,
    seu.seu_category,
    seu.annual_consumption_kwh,
    seu.percentage_of_total,
    seu.is_significant,
    seu.status,
    SUM(seu.percentage_of_total) OVER (
        PARTITION BY seu.enms_id
        ORDER BY seu.annual_consumption_kwh DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                               AS cumulative_pct,
    ROW_NUMBER() OVER (
        PARTITION BY seu.enms_id
        ORDER BY seu.annual_consumption_kwh DESC
    )                                               AS rank,
    COALESCE(eq.equipment_count, 0)                 AS equipment_count,
    COALESCE(dr.driver_count, 0)                    AS driver_count
FROM pack034_iso50001.significant_energy_uses seu
JOIN pack034_iso50001.energy_management_systems ems ON seu.enms_id = ems.id
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS equipment_count
    FROM pack034_iso50001.seu_equipment e
    WHERE e.seu_id = seu.id
) eq ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS driver_count
    FROM pack034_iso50001.energy_drivers d
    WHERE d.seu_id = seu.id
) dr ON TRUE;

-- =============================================================================
-- View 3: v_enpi_performance
-- =============================================================================
-- EnPI performance dashboard showing current vs. target vs. baseline
-- with improvement tracking and latest measurement details.

CREATE OR REPLACE VIEW pack034_iso50001.v_enpi_performance AS
SELECT
    enpi.id                                         AS enpi_id,
    enpi.enms_id,
    ems.organization_id,
    enpi.enpi_name,
    enpi.enpi_type,
    enpi.energy_type,
    enpi.baseline_value,
    enpi.current_value,
    enpi.target_value,
    enpi.improvement_pct,
    -- Target achievement
    CASE WHEN enpi.target_value IS NOT NULL AND enpi.baseline_value IS NOT NULL
              AND enpi.baseline_value != enpi.target_value
        THEN ROUND(
            (enpi.baseline_value - enpi.current_value)
            / (enpi.baseline_value - enpi.target_value) * 100, 2
        )
        ELSE NULL
    END                                             AS target_achievement_pct,
    -- Latest measurement
    lv.period_end                                   AS latest_period_end,
    lv.measured_value                               AS latest_measured,
    lv.normalized_value                             AS latest_normalized,
    lv.variance_pct                                 AS latest_variance_pct,
    lv.data_quality_score                           AS latest_quality,
    -- Target info
    et.target_year                                  AS next_target_year,
    et.target_value                                 AS next_target_value
FROM pack034_iso50001.energy_performance_indicators enpi
JOIN pack034_iso50001.energy_management_systems ems ON enpi.enms_id = ems.id
LEFT JOIN LATERAL (
    SELECT period_end, measured_value, normalized_value, variance_pct, data_quality_score
    FROM pack034_iso50001.enpi_values ev
    WHERE ev.enpi_id = enpi.id
    ORDER BY ev.period_end DESC
    LIMIT 1
) lv ON TRUE
LEFT JOIN LATERAL (
    SELECT target_year, target_value
    FROM pack034_iso50001.enpi_targets et2
    WHERE et2.enpi_id = enpi.id AND et2.target_year >= EXTRACT(YEAR FROM CURRENT_DATE)
    ORDER BY et2.target_year
    LIMIT 1
) et ON TRUE;

-- =============================================================================
-- View 4: v_cusum_status
-- =============================================================================
-- CUSUM monitoring status with latest data point, alert counts, and
-- current trend assessment for each active monitor.

CREATE OR REPLACE VIEW pack034_iso50001.v_cusum_status AS
SELECT
    cm.id                                           AS monitor_id,
    cm.enms_id,
    ems.organization_id,
    cm.monitor_name,
    cm.monitoring_interval,
    cm.alert_threshold,
    cm.status,
    -- Latest data point
    ldp.period_date                                 AS latest_date,
    ldp.actual_consumption                          AS latest_actual,
    ldp.expected_consumption                        AS latest_expected,
    ldp.cumulative_sum                              AS latest_cusum,
    -- Alert summary
    COALESCE(alerts.total_alerts, 0)                AS total_alerts,
    COALESCE(alerts.unack_alerts, 0)                AS unacknowledged_alerts,
    alerts.latest_alert_date,
    -- Trend assessment
    CASE
        WHEN ldp.cumulative_sum > cm.alert_threshold THEN 'OVER_CONSUMING'
        WHEN ldp.cumulative_sum < -cm.alert_threshold THEN 'UNDER_CONSUMING'
        ELSE 'WITHIN_LIMITS'
    END                                             AS cusum_assessment,
    -- Data point count
    COALESCE(dp_count.total_points, 0)              AS data_point_count
FROM pack034_iso50001.cusum_monitors cm
JOIN pack034_iso50001.energy_management_systems ems ON cm.enms_id = ems.id
LEFT JOIN LATERAL (
    SELECT period_date, actual_consumption, expected_consumption, cumulative_sum
    FROM pack034_iso50001.cusum_data_points cdp
    WHERE cdp.monitor_id = cm.id
    ORDER BY cdp.period_date DESC
    LIMIT 1
) ldp ON TRUE
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                    AS total_alerts,
        SUM(CASE WHEN NOT acknowledged THEN 1 ELSE 0 END) AS unack_alerts,
        MAX(alert_date)                             AS latest_alert_date
    FROM pack034_iso50001.cusum_alerts ca
    WHERE ca.monitor_id = cm.id
) alerts ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS total_points
    FROM pack034_iso50001.cusum_data_points cdp2
    WHERE cdp2.monitor_id = cm.id
) dp_count ON TRUE;

-- =============================================================================
-- View 5: v_compliance_summary
-- =============================================================================
-- Compliance assessment summary showing latest assessment results,
-- clause conformity breakdown, and NC resolution progress.

CREATE OR REPLACE VIEW pack034_iso50001.v_compliance_summary AS
SELECT
    ca.id                                           AS assessment_id,
    ca.enms_id,
    ems.organization_id,
    ca.assessment_date,
    ca.assessment_type,
    ca.assessor,
    ca.overall_score,
    ca.total_clauses,
    ca.conforming_clauses,
    ca.status,
    -- Conformity breakdown
    CASE WHEN ca.total_clauses > 0
        THEN ROUND(ca.conforming_clauses::DECIMAL / ca.total_clauses * 100, 1)
        ELSE 0
    END                                             AS conformity_pct,
    COALESCE(cs_agg.minor_ncs, 0)                   AS minor_ncs,
    COALESCE(cs_agg.major_ncs, 0)                   AS major_ncs,
    COALESCE(cs_agg.opportunities, 0)               AS opportunities,
    -- NC resolution
    COALESCE(nc_agg.total_ncs, 0)                   AS total_nonconformities,
    COALESCE(nc_agg.open_ncs, 0)                    AS open_ncs,
    COALESCE(nc_agg.closed_ncs, 0)                  AS closed_ncs,
    CASE WHEN nc_agg.total_ncs > 0
        THEN ROUND(nc_agg.closed_ncs::DECIMAL / nc_agg.total_ncs * 100, 1)
        ELSE 100
    END                                             AS nc_closure_rate_pct,
    ca.created_at
FROM pack034_iso50001.compliance_assessments ca
JOIN pack034_iso50001.energy_management_systems ems ON ca.enms_id = ems.id
LEFT JOIN LATERAL (
    SELECT
        SUM(CASE WHEN status = 'minor_nc' THEN 1 ELSE 0 END) AS minor_ncs,
        SUM(CASE WHEN status = 'major_nc' THEN 1 ELSE 0 END) AS major_ncs,
        SUM(CASE WHEN status = 'opportunity' THEN 1 ELSE 0 END) AS opportunities
    FROM pack034_iso50001.clause_scores cs
    WHERE cs.assessment_id = ca.id
) cs_agg ON TRUE
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                    AS total_ncs,
        SUM(CASE WHEN status IN ('open', 'corrective_action') THEN 1 ELSE 0 END) AS open_ncs,
        SUM(CASE WHEN status IN ('verified', 'closed') THEN 1 ELSE 0 END) AS closed_ncs
    FROM pack034_iso50001.nonconformities nc
    WHERE nc.assessment_id = ca.id
) nc_agg ON TRUE;

-- =============================================================================
-- View 6: v_action_plan_progress
-- =============================================================================
-- Action plan progress tracking with objective/target context, savings
-- estimates, and item-level completion rates.

CREATE OR REPLACE VIEW pack034_iso50001.v_action_plan_progress AS
SELECT
    ap.id                                           AS plan_id,
    obj.enms_id,
    ems.organization_id,
    obj.objective_text,
    obj.objective_type,
    tgt.target_description,
    tgt.target_value,
    tgt.target_unit,
    tgt.achievement_pct                             AS target_achievement_pct,
    ap.plan_name,
    ap.responsible_person,
    ap.department,
    ap.estimated_cost,
    ap.estimated_savings_kwh,
    ap.estimated_savings_cost,
    ap.start_date,
    ap.end_date,
    ap.status                                       AS plan_status,
    -- Action items progress
    COALESCE(ai_agg.total_items, 0)                 AS total_action_items,
    COALESCE(ai_agg.completed_items, 0)             AS completed_items,
    COALESCE(ai_agg.overdue_items, 0)               AS overdue_items,
    CASE WHEN ai_agg.total_items > 0
        THEN ROUND(ai_agg.completed_items::DECIMAL / ai_agg.total_items * 100, 1)
        ELSE 0
    END                                             AS item_completion_pct,
    -- Schedule status
    CASE
        WHEN ap.status = 'completed' THEN 'COMPLETED'
        WHEN ap.status = 'cancelled' THEN 'CANCELLED'
        WHEN ap.end_date < CURRENT_DATE AND ap.status != 'completed' THEN 'OVERDUE'
        WHEN ap.start_date > CURRENT_DATE THEN 'NOT_STARTED'
        ELSE 'ON_TRACK'
    END                                             AS schedule_status,
    ap.created_at
FROM pack034_iso50001.action_plans ap
JOIN pack034_iso50001.energy_targets tgt ON ap.target_id = tgt.id
JOIN pack034_iso50001.energy_objectives obj ON tgt.objective_id = obj.id
JOIN pack034_iso50001.energy_management_systems ems ON obj.enms_id = ems.id
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                    AS total_items,
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_items,
        SUM(CASE WHEN status = 'overdue' OR (due_date < CURRENT_DATE AND status NOT IN ('completed', 'cancelled')) THEN 1 ELSE 0 END) AS overdue_items
    FROM pack034_iso50001.action_items ai
    WHERE ai.plan_id = ap.id
) ai_agg ON TRUE;

-- ---------------------------------------------------------------------------
-- Grants on views
-- ---------------------------------------------------------------------------
GRANT SELECT ON pack034_iso50001.v_enms_overview TO PUBLIC;
GRANT SELECT ON pack034_iso50001.v_seu_pareto TO PUBLIC;
GRANT SELECT ON pack034_iso50001.v_enpi_performance TO PUBLIC;
GRANT SELECT ON pack034_iso50001.v_cusum_status TO PUBLIC;
GRANT SELECT ON pack034_iso50001.v_compliance_summary TO PUBLIC;
GRANT SELECT ON pack034_iso50001.v_action_plan_progress TO PUBLIC;

-- =============================================================================
-- Function 1: fn_calculate_enpi_improvement
-- =============================================================================
-- Calculates and returns EnPI improvement metrics for a given EnMS as JSONB.
-- Updates the improvement_pct on each EnPI based on latest values vs baseline.

CREATE OR REPLACE FUNCTION pack034_iso50001.fn_calculate_enpi_improvement(
    p_enms_id UUID
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_result JSONB;
    v_enpi RECORD;
    v_latest_value DECIMAL(18,6);
BEGIN
    -- Update improvement percentages for all EnPIs in this EnMS
    FOR v_enpi IN
        SELECT id, baseline_value
        FROM pack034_iso50001.energy_performance_indicators
        WHERE enms_id = p_enms_id AND baseline_value IS NOT NULL AND baseline_value != 0
    LOOP
        SELECT normalized_value INTO v_latest_value
        FROM pack034_iso50001.enpi_values
        WHERE enpi_id = v_enpi.id
        ORDER BY period_end DESC
        LIMIT 1;

        IF v_latest_value IS NOT NULL THEN
            UPDATE pack034_iso50001.energy_performance_indicators
            SET current_value = v_latest_value,
                improvement_pct = ROUND(
                    ((v_enpi.baseline_value - v_latest_value) / v_enpi.baseline_value) * 100, 4
                )
            WHERE id = v_enpi.id;
        END IF;
    END LOOP;

    -- Return summary
    SELECT jsonb_build_object(
        'enms_id', p_enms_id,
        'total_enpis', COUNT(*),
        'improving_count', COUNT(*) FILTER (WHERE improvement_pct > 0),
        'degrading_count', COUNT(*) FILTER (WHERE improvement_pct < 0),
        'avg_improvement_pct', ROUND(COALESCE(AVG(improvement_pct), 0), 2),
        'best_improvement_pct', COALESCE(MAX(improvement_pct), 0),
        'worst_improvement_pct', COALESCE(MIN(improvement_pct), 0),
        'calculated_at', NOW()
    ) INTO v_result
    FROM pack034_iso50001.energy_performance_indicators
    WHERE enms_id = p_enms_id;

    RETURN COALESCE(v_result, '{}'::JSONB);
END;
$$;

-- =============================================================================
-- Function 2: fn_check_cusum_alert
-- =============================================================================
-- Checks the latest CUSUM value for a monitor and generates an alert if
-- the cumulative sum exceeds the configured threshold.

CREATE OR REPLACE FUNCTION pack034_iso50001.fn_check_cusum_alert(
    p_monitor_id UUID
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_monitor RECORD;
    v_latest RECORD;
    v_alert_type VARCHAR(30);
    v_alert_id UUID;
    v_result JSONB;
BEGIN
    -- Get monitor configuration
    SELECT * INTO v_monitor
    FROM pack034_iso50001.cusum_monitors
    WHERE id = p_monitor_id AND status = 'active';

    IF NOT FOUND THEN
        RETURN jsonb_build_object('status', 'error', 'message', 'Monitor not found or not active');
    END IF;

    -- Get latest data point
    SELECT * INTO v_latest
    FROM pack034_iso50001.cusum_data_points
    WHERE monitor_id = p_monitor_id
    ORDER BY period_date DESC
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('status', 'no_data', 'message', 'No data points found');
    END IF;

    -- Check threshold
    IF v_latest.cumulative_sum > v_monitor.alert_threshold THEN
        v_alert_type := 'performance_degradation';
    ELSIF v_latest.cumulative_sum < -v_monitor.alert_threshold THEN
        v_alert_type := 'performance_improvement';
    ELSE
        RETURN jsonb_build_object(
            'status', 'ok',
            'cusum', v_latest.cumulative_sum,
            'threshold', v_monitor.alert_threshold,
            'message', 'Within control limits'
        );
    END IF;

    -- Insert alert
    INSERT INTO pack034_iso50001.cusum_alerts (
        monitor_id, alert_type, cumulative_value, threshold_exceeded
    ) VALUES (
        p_monitor_id, v_alert_type, v_latest.cumulative_sum, v_monitor.alert_threshold
    ) RETURNING id INTO v_alert_id;

    v_result := jsonb_build_object(
        'status', 'alert',
        'alert_id', v_alert_id,
        'alert_type', v_alert_type,
        'cusum', v_latest.cumulative_sum,
        'threshold', v_monitor.alert_threshold,
        'period_date', v_latest.period_date
    );

    RETURN v_result;
END;
$$;

-- =============================================================================
-- Function 3: fn_update_compliance_score
-- =============================================================================
-- Recalculates the overall compliance score for an assessment based on
-- individual clause scores and updates the assessment record.

CREATE OR REPLACE FUNCTION pack034_iso50001.fn_update_compliance_score(
    p_assessment_id UUID
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_total INTEGER;
    v_conforming INTEGER;
    v_score DECIMAL(6,2);
    v_result JSONB;
BEGIN
    -- Count clause statuses
    SELECT
        COUNT(*) FILTER (WHERE status != 'not_applicable' AND status != 'not_assessed'),
        COUNT(*) FILTER (WHERE status = 'conforming')
    INTO v_total, v_conforming
    FROM pack034_iso50001.clause_scores
    WHERE assessment_id = p_assessment_id;

    -- Calculate score
    IF v_total > 0 THEN
        v_score := ROUND((v_conforming::DECIMAL / v_total) * 100, 2);
    ELSE
        v_score := 0;
    END IF;

    -- Update assessment
    UPDATE pack034_iso50001.compliance_assessments
    SET overall_score = v_score,
        total_clauses = v_total,
        conforming_clauses = v_conforming
    WHERE id = p_assessment_id;

    v_result := jsonb_build_object(
        'assessment_id', p_assessment_id,
        'total_clauses', v_total,
        'conforming_clauses', v_conforming,
        'overall_score', v_score,
        'calculated_at', NOW()
    );

    RETURN v_result;
END;
$$;

-- ---------------------------------------------------------------------------
-- Grants on functions
-- ---------------------------------------------------------------------------
GRANT EXECUTE ON FUNCTION pack034_iso50001.fn_calculate_enpi_improvement(UUID) TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack034_iso50001.fn_check_cusum_alert(UUID) TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack034_iso50001.fn_update_compliance_score(UUID) TO PUBLIC;

-- =============================================================================
-- Audit trigger function: fn_audit_status_change
-- =============================================================================
-- Generic audit trigger that logs status changes on key tables to the
-- pack034_audit_trail for certification evidence and governance.

CREATE OR REPLACE FUNCTION pack034_iso50001.fn_audit_status_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_org_id UUID;
    v_enms_id UUID;
BEGIN
    -- Resolve organization_id through the table hierarchy
    IF TG_TABLE_NAME = 'energy_management_systems' THEN
        v_org_id := NEW.organization_id;
        v_enms_id := NEW.id;
    ELSIF TG_TABLE_NAME IN ('significant_energy_uses', 'energy_baselines',
                            'energy_performance_indicators', 'cusum_monitors',
                            'energy_objectives', 'compliance_assessments',
                            'metering_plans', 'performance_reports') THEN
        SELECT organization_id, id INTO v_org_id, v_enms_id
        FROM pack034_iso50001.energy_management_systems
        WHERE id = NEW.enms_id;
    ELSE
        v_org_id := '00000000-0000-0000-0000-000000000000'::UUID;
        v_enms_id := NULL;
    END IF;

    INSERT INTO pack034_iso50001.pack034_audit_trail (
        enms_id, organization_id, action, entity_type, entity_id, old_values, new_values
    ) VALUES (
        v_enms_id,
        v_org_id,
        'STATUS_CHANGE',
        TG_TABLE_NAME,
        NEW.id,
        jsonb_build_object('status', OLD.status),
        jsonb_build_object('status', NEW.status)
    );

    RETURN NEW;
END;
$$;

-- ---------------------------------------------------------------------------
-- Apply audit triggers on tables with status columns
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_ems_status_audit
    AFTER UPDATE OF enms_status ON pack034_iso50001.energy_management_systems
    FOR EACH ROW
    WHEN (OLD.enms_status IS DISTINCT FROM NEW.enms_status)
    EXECUTE FUNCTION pack034_iso50001.fn_audit_status_change();

CREATE TRIGGER trg_p034_seu_status_audit
    AFTER UPDATE OF status ON pack034_iso50001.significant_energy_uses
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION pack034_iso50001.fn_audit_status_change();

CREATE TRIGGER trg_p034_bl_status_audit
    AFTER UPDATE OF status ON pack034_iso50001.energy_baselines
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION pack034_iso50001.fn_audit_status_change();

CREATE TRIGGER trg_p034_nc_status_audit
    AFTER UPDATE OF status ON pack034_iso50001.nonconformities
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION pack034_iso50001.fn_audit_status_change();

CREATE TRIGGER trg_p034_ap_status_audit
    AFTER UPDATE OF status ON pack034_iso50001.action_plans
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION pack034_iso50001.fn_audit_status_change();

CREATE TRIGGER trg_p034_assess_status_audit
    AFTER UPDATE OF status ON pack034_iso50001.compliance_assessments
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION pack034_iso50001.fn_audit_status_change();

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack034_iso50001.pack034_audit_trail IS
    'Audit trail for all PACK-034 entity changes with old/new values for certification evidence and data governance.';

COMMENT ON VIEW pack034_iso50001.v_enms_overview IS
    'Comprehensive EnMS overview with scope counts, SEU summary, baseline status, EnPI performance, and certification readiness.';

COMMENT ON VIEW pack034_iso50001.v_seu_pareto IS
    'SEU Pareto analysis showing cumulative consumption percentage for identifying the vital few energy uses.';

COMMENT ON VIEW pack034_iso50001.v_enpi_performance IS
    'EnPI performance dashboard with current vs. target vs. baseline and improvement tracking.';

COMMENT ON VIEW pack034_iso50001.v_cusum_status IS
    'CUSUM monitoring status with latest data point, alert counts, and current trend assessment.';

COMMENT ON VIEW pack034_iso50001.v_compliance_summary IS
    'Compliance assessment summary with clause conformity breakdown and NC resolution progress.';

COMMENT ON VIEW pack034_iso50001.v_action_plan_progress IS
    'Action plan progress tracking with objective/target context, savings estimates, and item-level completion rates.';

COMMENT ON FUNCTION pack034_iso50001.fn_calculate_enpi_improvement(UUID) IS
    'Calculates EnPI improvement metrics for a given EnMS and updates improvement_pct on each EnPI.';

COMMENT ON FUNCTION pack034_iso50001.fn_check_cusum_alert(UUID) IS
    'Checks the latest CUSUM value for a monitor and generates an alert if threshold is exceeded.';

COMMENT ON FUNCTION pack034_iso50001.fn_update_compliance_score(UUID) IS
    'Recalculates overall compliance score for an assessment based on individual clause scores.';

COMMENT ON FUNCTION pack034_iso50001.fn_audit_status_change() IS
    'Generic audit trigger function that logs status changes on key EnMS tables.';

COMMENT ON COLUMN pack034_iso50001.pack034_audit_trail.entry_id IS
    'Unique identifier for the audit trail entry.';
COMMENT ON COLUMN pack034_iso50001.pack034_audit_trail.action IS
    'Type of action: CREATE, UPDATE, DELETE, ARCHIVE, RESTORE, APPROVE, REJECT, STATUS_CHANGE.';
COMMENT ON COLUMN pack034_iso50001.pack034_audit_trail.old_values IS
    'JSON snapshot of entity state before the change.';
COMMENT ON COLUMN pack034_iso50001.pack034_audit_trail.new_values IS
    'JSON snapshot of entity state after the change.';
