-- =============================================================================
-- V375: PACK-045 Base Year Management Pack - Views, Indexes & Seed Data
-- =============================================================================
-- Pack:         PACK-045 (Base Year Management Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates analytical views, materialized views, additional composite indexes
-- for common query patterns, and seed data. Views provide pre-joined
-- perspectives for dashboards and reporting. Materialized views support
-- high-performance summary and history queries. Seed data populates default
-- recalculation policies and significance thresholds.
--
-- Views (2):
--   1. ghg_base_year.v_base_year_summary
--   2. ghg_base_year.v_target_dashboard
--
-- Materialized Views (1):
--   3. ghg_base_year.mv_recalculation_history
--
-- Also includes: additional indexes, seed data, grants, comments.
-- Previous: V374__pack045_audit.sql
-- =============================================================================

SET search_path TO ghg_base_year, public;

-- =============================================================================
-- View 1: ghg_base_year.v_base_year_summary
-- =============================================================================
-- Dashboard view providing a per-organisation summary of the active base year
-- with scope breakdown, verification status, open trigger count, and total
-- adjustments applied. Single-query feed for the base year overview panel.

CREATE OR REPLACE VIEW ghg_base_year.v_base_year_summary AS
SELECT
    by_rec.id                       AS base_year_id,
    by_rec.tenant_id,
    by_rec.org_id,
    by_rec.base_year,
    by_rec.base_year_type,
    by_rec.status,
    by_rec.established_date,
    by_rec.total_tco2e,
    by_rec.scope1_tco2e,
    by_rec.scope2_location_tco2e,
    by_rec.scope2_market_tco2e,
    by_rec.scope3_tco2e,
    by_rec.gwp_version,
    by_rec.consolidation_approach,
    by_rec.verification_status,
    by_rec.verified_by,
    by_rec.verified_date,
    -- Open trigger count
    COALESCE(trg_agg.open_trigger_count, 0)     AS open_trigger_count,
    -- Total adjustments applied
    COALESCE(adj_agg.total_adjustments, 0)       AS total_adjustments_applied,
    COALESCE(adj_agg.net_adjustment_tco2e, 0)    AS net_adjustment_tco2e,
    -- Inventory line count
    COALESCE(inv_agg.inventory_line_count, 0)    AS inventory_line_count,
    COALESCE(inv_agg.avg_data_quality, 0)        AS avg_data_quality_score,
    by_rec.provenance_hash,
    by_rec.updated_at
FROM ghg_base_year.gl_by_base_years by_rec
LEFT JOIN (
    SELECT base_year_id, COUNT(*) AS open_trigger_count
    FROM ghg_base_year.gl_by_triggers
    WHERE status IN ('DETECTED', 'UNDER_ASSESSMENT', 'SIGNIFICANT', 'RECALCULATION_REQUIRED')
    GROUP BY base_year_id
) trg_agg ON by_rec.id = trg_agg.base_year_id
LEFT JOIN (
    SELECT base_year_id,
           COUNT(*) AS total_adjustments,
           SUM(total_adjustment_tco2e) AS net_adjustment_tco2e
    FROM ghg_base_year.gl_by_adjustment_packages
    WHERE status = 'APPLIED'
    GROUP BY base_year_id
) adj_agg ON by_rec.id = adj_agg.base_year_id
LEFT JOIN (
    SELECT base_year_id,
           COUNT(*) AS inventory_line_count,
           AVG(data_quality_score) AS avg_data_quality
    FROM ghg_base_year.gl_by_inventories
    GROUP BY base_year_id
) inv_agg ON by_rec.id = inv_agg.base_year_id
WHERE by_rec.status = 'ACTIVE';

-- =============================================================================
-- View 2: ghg_base_year.v_target_dashboard
-- =============================================================================
-- Target progress dashboard view showing each active target with its most
-- recent progress record, gap analysis, and status indicators.

CREATE OR REPLACE VIEW ghg_base_year.v_target_dashboard AS
SELECT
    tgt.id                          AS target_id,
    tgt.tenant_id,
    tgt.org_id,
    tgt.target_name,
    tgt.target_type,
    tgt.scope,
    tgt.base_year,
    tgt.base_year_tco2e,
    tgt.target_year,
    tgt.target_reduction_pct,
    tgt.target_tco2e,
    tgt.sbti_ambition,
    tgt.sbti_status,
    tgt.annual_linear_rate_pct,
    tgt.status                      AS target_status,
    -- Latest progress
    tp.year                         AS latest_year,
    tp.actual_tco2e                 AS latest_actual_tco2e,
    tp.expected_tco2e               AS latest_expected_tco2e,
    tp.gap_tco2e                    AS latest_gap_tco2e,
    tp.gap_pct                      AS latest_gap_pct,
    tp.cumulative_reduction_pct     AS latest_cumulative_reduction_pct,
    tp.status                       AS latest_progress_status,
    tp.carbon_budget_remaining,
    -- Derived: years remaining
    (tgt.target_year - COALESCE(tp.year, EXTRACT(YEAR FROM NOW())::INTEGER))
                                    AS years_remaining,
    -- Derived: overall progress percentage
    CASE WHEN tgt.target_reduction_pct > 0 AND tp.cumulative_reduction_pct IS NOT NULL
         THEN ROUND((tp.cumulative_reduction_pct / tgt.target_reduction_pct * 100)::NUMERIC, 1)
         ELSE 0
    END                             AS overall_progress_pct,
    tgt.provenance_hash,
    tgt.updated_at
FROM ghg_base_year.gl_by_targets tgt
LEFT JOIN LATERAL (
    SELECT *
    FROM ghg_base_year.gl_by_target_progress p
    WHERE p.target_id = tgt.id AND p.is_projected = false
    ORDER BY p.year DESC
    LIMIT 1
) tp ON true
WHERE tgt.status IN ('APPROVED', 'PUBLISHED');

-- =============================================================================
-- Materialized View: ghg_base_year.mv_recalculation_history
-- =============================================================================
-- Pre-computed view of all recalculation events with trigger details,
-- assessment outcomes, and adjustment summaries. Supports the recalculation
-- history timeline and audit reporting.

CREATE MATERIALIZED VIEW ghg_base_year.mv_recalculation_history AS
SELECT
    ap.id                           AS package_id,
    ap.tenant_id,
    ap.base_year_id,
    by_rec.org_id,
    by_rec.base_year,
    ap.package_name,
    ap.package_type,
    ap.status                       AS package_status,
    ap.effective_date,
    ap.original_total_tco2e,
    ap.total_adjustment_tco2e,
    ap.adjusted_total_tco2e,
    ap.adjustment_pct,
    ap.created_by_name,
    ap.approved_by_name,
    ap.approved_date,
    ap.verification_status,
    -- Trigger count and types
    COALESCE(array_length(ap.trigger_ids, 1), 0) AS trigger_count,
    -- Line item count
    COALESCE(line_agg.line_count, 0)              AS line_count,
    COALESCE(line_agg.scopes_affected, '{}')      AS scopes_affected,
    ap.provenance_hash,
    ap.created_at
FROM ghg_base_year.gl_by_adjustment_packages ap
INNER JOIN ghg_base_year.gl_by_base_years by_rec
    ON ap.base_year_id = by_rec.id
LEFT JOIN (
    SELECT package_id,
           COUNT(*) AS line_count,
           ARRAY_AGG(DISTINCT scope) AS scopes_affected
    FROM ghg_base_year.gl_by_adjustment_lines
    GROUP BY package_id
) line_agg ON ap.id = line_agg.package_id
WHERE ap.status IN ('APPLIED', 'APPROVED')
ORDER BY ap.effective_date DESC;

-- Index on materialized view
CREATE UNIQUE INDEX idx_p045_mv_recalc_pkg
    ON ghg_base_year.mv_recalculation_history(package_id);
CREATE INDEX idx_p045_mv_recalc_tenant
    ON ghg_base_year.mv_recalculation_history(tenant_id);
CREATE INDEX idx_p045_mv_recalc_org
    ON ghg_base_year.mv_recalculation_history(org_id);
CREATE INDEX idx_p045_mv_recalc_by
    ON ghg_base_year.mv_recalculation_history(base_year_id);
CREATE INDEX idx_p045_mv_recalc_date
    ON ghg_base_year.mv_recalculation_history(effective_date DESC);

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Cross-table: triggers with significance assessments
CREATE INDEX idx_p045_trg_org_date ON ghg_base_year.gl_by_triggers(org_id, detected_date DESC);

-- Cross-table: audit trail by org + date range
CREATE INDEX idx_p045_at_org_date ON ghg_base_year.gl_by_audit_trail(org_id, timestamp DESC);

-- Cross-table: approvals pending by org
CREATE INDEX idx_p045_appr_org_pending ON ghg_base_year.gl_by_approvals(org_id, due_date)
    WHERE status = 'PENDING';

-- Cross-table: time series trend query
CREATE INDEX idx_p045_ts_trend ON ghg_base_year.gl_by_time_series(org_id, scope, year)
    WHERE category IS NULL;

-- =============================================================================
-- Seed Data: Default Recalculation Policy Thresholds
-- =============================================================================
-- These are commonly-used threshold configurations per GHG Protocol guidance.
-- Organisations can override these by creating custom configurations.

-- Note: Seed data uses a well-known tenant_id (all-zeros UUID) as a template
-- that is copied when new organisations are onboarded.

INSERT INTO ghg_base_year.gl_by_configuration (
    tenant_id, org_id, config_version, preset_name,
    policy_json, config_json, thresholds_json,
    target_tracking_json, reporting_json,
    is_active, effective_date, notes, provenance_hash
) VALUES
-- Default: Corporate Office (5% threshold)
(
    '00000000-0000-0000-0000-000000000000'::UUID,
    '00000000-0000-0000-0000-000000000001'::UUID,
    1, 'corporate_office',
    '{"recalculation_triggers": ["structural_changes", "methodology_changes", "error_corrections", "outsourcing_insourcing", "acquisition_divestiture", "gwp_version_change"], "approval_workflow": {"require_approval": true, "minimum_approvers": 1, "auto_approve_below_pct": 2.0}}'::JSONB,
    '{"base_year_type": "FIXED", "selection_method": "REPRESENTATIVE_YEAR"}'::JSONB,
    '{"significance_threshold_pct": 5.0, "cumulative_threshold_pct": 10.0, "de_minimis_threshold_tco2e": 50.0}'::JSONB,
    '{"target_type": "ABSOLUTE", "ambition": "WELL_BELOW_2C", "annual_rate_pct": 4.2}'::JSONB,
    '{"frameworks": ["GHG_PROTOCOL", "ESRS_E1", "CDP"], "formats": ["HTML", "JSON"]}'::JSONB,
    false, '2026-01-01',
    'Template: Corporate office default configuration (5% threshold).',
    'seed_corporate_office_v1'
),
-- Default: Manufacturing (3% threshold)
(
    '00000000-0000-0000-0000-000000000000'::UUID,
    '00000000-0000-0000-0000-000000000002'::UUID,
    1, 'manufacturing',
    '{"recalculation_triggers": ["structural_changes", "methodology_changes", "error_corrections", "outsourcing_insourcing", "acquisition_divestiture", "gwp_version_change", "production_line_changes", "process_technology_upgrade"], "approval_workflow": {"require_approval": true, "minimum_approvers": 2, "auto_approve_below_pct": null}}'::JSONB,
    '{"base_year_type": "FIXED", "selection_method": "STABLE_PRODUCTION_YEAR"}'::JSONB,
    '{"significance_threshold_pct": 3.0, "cumulative_threshold_pct": 5.0, "de_minimis_threshold_tco2e": 500.0}'::JSONB,
    '{"target_type": "INTENSITY", "ambition": "WELL_BELOW_2C", "annual_rate_pct": 4.2, "intensity_metric": "tCO2e/tonne_output"}'::JSONB,
    '{"frameworks": ["GHG_PROTOCOL", "ESRS_E1", "CDP", "ISO_14064"], "formats": ["HTML", "JSON", "CSV"]}'::JSONB,
    false, '2026-01-01',
    'Template: Manufacturing default configuration (3% threshold).',
    'seed_manufacturing_v1'
),
-- Default: Energy Utility (2% threshold, regulatory-driven)
(
    '00000000-0000-0000-0000-000000000000'::UUID,
    '00000000-0000-0000-0000-000000000003'::UUID,
    1, 'energy_utility',
    '{"recalculation_triggers": ["structural_changes", "methodology_changes", "error_corrections", "outsourcing_insourcing", "acquisition_divestiture", "gwp_version_change", "fuel_mix_change", "capacity_addition_retirement", "regulatory_mandate"], "approval_workflow": {"require_approval": true, "minimum_approvers": 2, "require_board_above_pct": 15.0}}'::JSONB,
    '{"base_year_type": "REGULATORY", "selection_method": "REGULATORY_ALIGNED", "regulatory_reference": "EU_ETS_PHASE_4"}'::JSONB,
    '{"significance_threshold_pct": 2.0, "cumulative_threshold_pct": 3.0, "de_minimis_threshold_tco2e": 5000.0}'::JSONB,
    '{"target_type": "INTENSITY", "ambition": "1_5C", "annual_rate_pct": 7.0, "intensity_metric": "tCO2e/MWh_generated"}'::JSONB,
    '{"frameworks": ["GHG_PROTOCOL", "ESRS_E1", "CDP", "ISO_14064", "SBTI", "EU_ETS"], "formats": ["HTML", "JSON", "CSV"]}'::JSONB,
    false, '2026-01-01',
    'Template: Energy utility default configuration (2% threshold, regulatory base year).',
    'seed_energy_utility_v1'
),
-- Default: SME Simplified (10% threshold)
(
    '00000000-0000-0000-0000-000000000000'::UUID,
    '00000000-0000-0000-0000-000000000004'::UUID,
    1, 'sme_simplified',
    '{"recalculation_triggers": ["structural_changes", "methodology_changes", "error_corrections", "outsourcing_insourcing", "acquisition_divestiture"], "approval_workflow": {"require_approval": false, "auto_approve_all": true}}'::JSONB,
    '{"base_year_type": "FIXED", "selection_method": "MOST_RECENT_COMPLETE"}'::JSONB,
    '{"significance_threshold_pct": 10.0, "cumulative_threshold_pct": 15.0, "de_minimis_threshold_tco2e": 25.0}'::JSONB,
    '{"target_type": "ABSOLUTE", "ambition": "WELL_BELOW_2C", "annual_rate_pct": 4.2}'::JSONB,
    '{"frameworks": ["GHG_PROTOCOL"], "formats": ["HTML", "JSON"]}'::JSONB,
    false, '2026-01-01',
    'Template: SME simplified default configuration (10% threshold, no approval required).',
    'seed_sme_simplified_v1'
);

-- =============================================================================
-- Grants for Views
-- =============================================================================
GRANT SELECT ON ghg_base_year.v_base_year_summary TO PUBLIC;
GRANT SELECT ON ghg_base_year.v_target_dashboard TO PUBLIC;
GRANT SELECT ON ghg_base_year.mv_recalculation_history TO PUBLIC;

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON VIEW ghg_base_year.v_base_year_summary IS
    'Dashboard view: active base year per organisation with scope breakdown, open trigger count, and cumulative adjustments.';
COMMENT ON VIEW ghg_base_year.v_target_dashboard IS
    'Dashboard view: active targets with latest progress, gap analysis, years remaining, and overall completion percentage.';
COMMENT ON MATERIALIZED VIEW ghg_base_year.mv_recalculation_history IS
    'Pre-computed recalculation history timeline with trigger counts, adjustment summaries, and verification status. Refresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.';
