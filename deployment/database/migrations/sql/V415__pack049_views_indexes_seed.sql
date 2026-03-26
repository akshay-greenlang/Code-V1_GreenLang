-- =============================================================================
-- PACK-049 GHG Multi-Site Management Pack
-- Migration: V415 - Views, Additional Indexes & Seed Data
-- =============================================================================
-- Pack:         PACK-049 (GHG Multi-Site Management Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates materialized views for dashboard performance, additional
-- cross-table indexes, and seed data for reference lookups. This is
-- the final migration in the PACK-049 series.
--
-- Materialized Views (4):
--   1. mv_p049_site_portfolio_summary
--   2. mv_p049_consolidation_summary
--   3. mv_p049_quality_heatmap
--   4. mv_p049_completion_overview
--
-- Seed Data:
--   - 20 facility types
--   - 7 allocation methods
--   - 6 quality dimensions
--   - 8 KPI types
--   - 5 elimination types (+ 1 OTHER)
--   - 4 factor tiers
--   - Default reminder schedule
--
-- Previous: V414__pack049_completion.sql
-- =============================================================================

SET search_path TO ghg_multisite, public;

-- =============================================================================
-- Materialized View 1: Site Portfolio Summary
-- =============================================================================
-- Aggregated view of the entire site portfolio for dashboard rendering.
-- Provides per-site latest emission totals, quality scores, and completion.

CREATE MATERIALIZED VIEW ghg_multisite.mv_p049_site_portfolio_summary AS
SELECT
    s.tenant_id,
    s.config_id,
    s.id                        AS site_id,
    s.site_code,
    s.site_name,
    s.facility_type,
    s.country,
    s.region,
    s.lifecycle_status,
    s.is_active,
    sc.floor_area_m2,
    sc.headcount,
    sc.grid_region,
    sc.climate_zone,
    -- Latest submission totals (from most recent approved submission)
    COALESCE(sub.scope1_tco2e, 0)           AS scope1_tco2e,
    COALESCE(sub.scope2_location_tco2e, 0)  AS scope2_location_tco2e,
    COALESCE(sub.scope2_market_tco2e, 0)    AS scope2_market_tco2e,
    COALESCE(sub.scope3_tco2e, 0)           AS scope3_tco2e,
    COALESCE(sub.total_tco2e, 0)            AS total_tco2e,
    sub.data_quality_score                  AS submission_quality_score,
    -- Quality score
    qs.overall_score                        AS quality_score,
    qs.pcaf_equivalent_tier,
    -- Completion status
    cs.traffic_light,
    cs.completion_pct,
    cs.is_overdue,
    -- Timestamps
    s.created_at                            AS site_created_at,
    sub.submitted_at                        AS last_submission_at,
    qs.assessed_at                          AS last_quality_assessment_at,
    NOW()                                   AS refreshed_at
FROM ghg_multisite.gl_ms_sites s
LEFT JOIN ghg_multisite.gl_ms_site_characteristics sc
    ON sc.site_id = s.id
LEFT JOIN LATERAL (
    SELECT ss.*
    FROM ghg_multisite.gl_ms_site_submissions ss
    WHERE ss.site_id = s.id
      AND ss.status = 'APPROVED'
    ORDER BY ss.submitted_at DESC
    LIMIT 1
) sub ON true
LEFT JOIN LATERAL (
    SELECT qs2.*
    FROM ghg_multisite.gl_ms_quality_scores qs2
    WHERE qs2.site_id = s.id
    ORDER BY qs2.assessed_at DESC
    LIMIT 1
) qs ON true
LEFT JOIN LATERAL (
    SELECT cs2.*
    FROM ghg_multisite.gl_ms_completion_status cs2
    WHERE cs2.site_id = s.id
    ORDER BY cs2.updated_at DESC
    LIMIT 1
) cs ON true
WHERE s.is_active = true
WITH NO DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX uidx_p049_portfolio_site
    ON ghg_multisite.mv_p049_site_portfolio_summary(tenant_id, site_id);
CREATE INDEX idx_p049_portfolio_config
    ON ghg_multisite.mv_p049_site_portfolio_summary(config_id);
CREATE INDEX idx_p049_portfolio_country
    ON ghg_multisite.mv_p049_site_portfolio_summary(country);
CREATE INDEX idx_p049_portfolio_type
    ON ghg_multisite.mv_p049_site_portfolio_summary(facility_type);
CREATE INDEX idx_p049_portfolio_traffic
    ON ghg_multisite.mv_p049_site_portfolio_summary(traffic_light);

-- =============================================================================
-- Materialized View 2: Consolidation Summary
-- =============================================================================
-- Period-level consolidation summary for trending and reporting.

CREATE MATERIALIZED VIEW ghg_multisite.mv_p049_consolidation_summary AS
SELECT
    cr.tenant_id,
    cr.config_id,
    cr.period_id,
    rp.period_name,
    rp.period_start,
    rp.period_end,
    cr.id                                   AS run_id,
    cr.run_type,
    cr.consolidation_approach,
    cr.sites_included,
    cr.sites_excluded,
    cr.completeness_pct,
    cr.scope1_total_tco2e,
    cr.scope2_location_total_tco2e,
    cr.scope2_market_total_tco2e,
    cr.scope3_total_tco2e,
    cr.total_tco2e,
    cr.eliminations_tco2e,
    cr.net_total_tco2e,
    cr.status                               AS run_status,
    cr.completed_at,
    cr.approved_at,
    -- Scope 1+2 total (location-based)
    (cr.scope1_total_tco2e + cr.scope2_location_total_tco2e) AS scope1_2_location_tco2e,
    -- Scope 1+2 total (market-based)
    (cr.scope1_total_tco2e + cr.scope2_market_total_tco2e)   AS scope1_2_market_tco2e,
    NOW()                                   AS refreshed_at
FROM ghg_multisite.gl_ms_consolidation_runs cr
JOIN ghg_multisite.gl_ms_reporting_periods rp
    ON rp.id = cr.period_id
WHERE cr.status IN ('COMPLETED', 'APPROVED')
WITH NO DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX uidx_p049_consol_run
    ON ghg_multisite.mv_p049_consolidation_summary(tenant_id, run_id);
CREATE INDEX idx_p049_consol_period
    ON ghg_multisite.mv_p049_consolidation_summary(period_id);
CREATE INDEX idx_p049_consol_config
    ON ghg_multisite.mv_p049_consolidation_summary(config_id);
CREATE INDEX idx_p049_consol_type
    ON ghg_multisite.mv_p049_consolidation_summary(run_type);
CREATE INDEX idx_p049_consol_dates
    ON ghg_multisite.mv_p049_consolidation_summary(period_start, period_end);

-- =============================================================================
-- Materialized View 3: Quality Heatmap
-- =============================================================================
-- Site-by-dimension quality matrix for heatmap visualisation.

CREATE MATERIALIZED VIEW ghg_multisite.mv_p049_quality_heatmap AS
SELECT
    qs.tenant_id,
    qs.config_id,
    qs.period_id,
    qs.site_id,
    s.site_code,
    s.site_name,
    s.facility_type,
    s.country,
    qs.overall_score,
    qs.pcaf_equivalent_tier,
    qs.completeness_score,
    qs.accuracy_score,
    qs.consistency_score,
    qs.timeliness_score,
    qs.transparency_score,
    qs.reliability_score,
    qs.issues_found,
    qs.critical_issues,
    qs.improvement_potential_pct,
    qs.prior_period_score,
    qs.score_change,
    -- Heatmap classification
    CASE
        WHEN qs.overall_score >= 80 THEN 'HIGH'
        WHEN qs.overall_score >= 50 THEN 'MEDIUM'
        ELSE 'LOW'
    END                                     AS quality_band,
    NOW()                                   AS refreshed_at
FROM ghg_multisite.gl_ms_quality_scores qs
JOIN ghg_multisite.gl_ms_sites s
    ON s.id = qs.site_id
WITH NO DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX uidx_p049_heatmap_site_period
    ON ghg_multisite.mv_p049_quality_heatmap(tenant_id, site_id, period_id);
CREATE INDEX idx_p049_heatmap_config
    ON ghg_multisite.mv_p049_quality_heatmap(config_id);
CREATE INDEX idx_p049_heatmap_period
    ON ghg_multisite.mv_p049_quality_heatmap(period_id);
CREATE INDEX idx_p049_heatmap_band
    ON ghg_multisite.mv_p049_quality_heatmap(quality_band);
CREATE INDEX idx_p049_heatmap_overall
    ON ghg_multisite.mv_p049_quality_heatmap(overall_score);

-- =============================================================================
-- Materialized View 4: Completion Overview
-- =============================================================================
-- Period-level completion overview showing aggregate site status.

CREATE MATERIALIZED VIEW ghg_multisite.mv_p049_completion_overview AS
SELECT
    cs.tenant_id,
    cs.config_id,
    cs.period_id,
    rp.period_name,
    rp.period_start,
    rp.period_end,
    rp.deadline,
    COUNT(*)                                                    AS total_sites,
    COUNT(*) FILTER (WHERE cs.traffic_light = 'GREEN')          AS green_count,
    COUNT(*) FILTER (WHERE cs.traffic_light = 'AMBER')          AS amber_count,
    COUNT(*) FILTER (WHERE cs.traffic_light = 'RED')            AS red_count,
    COUNT(*) FILTER (WHERE cs.traffic_light = 'GREY')           AS grey_count,
    COUNT(*) FILTER (WHERE cs.is_overdue = true)                AS overdue_count,
    COUNT(*) FILTER (WHERE cs.validation_passed = true)         AS validated_count,
    COUNT(*) FILTER (WHERE cs.review_status = 'APPROVED')       AS approved_count,
    AVG(cs.completion_pct)                                      AS avg_completion_pct,
    MIN(cs.completion_pct)                                      AS min_completion_pct,
    MAX(cs.completion_pct)                                      AS max_completion_pct,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cs.completion_pct) AS median_completion_pct,
    SUM(cs.data_sources_expected)                               AS total_sources_expected,
    SUM(cs.data_sources_received)                               AS total_sources_received,
    CASE
        WHEN SUM(cs.data_sources_expected) > 0
        THEN ROUND(SUM(cs.data_sources_received)::numeric / SUM(cs.data_sources_expected) * 100, 2)
        ELSE 0
    END                                                         AS data_source_coverage_pct,
    NOW()                                                       AS refreshed_at
FROM ghg_multisite.gl_ms_completion_status cs
JOIN ghg_multisite.gl_ms_reporting_periods rp
    ON rp.id = cs.period_id
GROUP BY
    cs.tenant_id, cs.config_id, cs.period_id,
    rp.period_name, rp.period_start, rp.period_end, rp.deadline
WITH NO DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX uidx_p049_completion_period
    ON ghg_multisite.mv_p049_completion_overview(tenant_id, config_id, period_id);
CREATE INDEX idx_p049_completion_config
    ON ghg_multisite.mv_p049_completion_overview(config_id);
CREATE INDEX idx_p049_completion_dates
    ON ghg_multisite.mv_p049_completion_overview(period_start, period_end);

-- =============================================================================
-- Additional Cross-Table Indexes
-- =============================================================================

-- Site + period lookups (most common query pattern)
CREATE INDEX idx_p049_x_site_period_sub
    ON ghg_multisite.gl_ms_site_submissions(site_id, round_id);
CREATE INDEX idx_p049_x_site_period_kpi
    ON ghg_multisite.gl_ms_site_kpis(site_id, period_id, kpi_type);
CREATE INDEX idx_p049_x_site_period_qual
    ON ghg_multisite.gl_ms_quality_scores(site_id, period_id);
CREATE INDEX idx_p049_x_site_period_comp
    ON ghg_multisite.gl_ms_completion_status(site_id, period_id);

-- Consolidation run lookups
CREATE INDEX idx_p049_x_run_site_total
    ON ghg_multisite.gl_ms_site_totals(run_id, site_id);
CREATE INDEX idx_p049_x_run_elimination
    ON ghg_multisite.gl_ms_elimination_entries(run_id, elimination_type);

-- Factor lookups
CREATE INDEX idx_p049_x_factor_site_scope
    ON ghg_multisite.gl_ms_factor_assignments(site_id, factor_scope, factor_category);

-- Boundary lookups
CREATE INDEX idx_p049_x_boundary_site_incl
    ON ghg_multisite.gl_ms_boundary_inclusions(boundary_id, site_id, inclusion_status);

-- Allocation lookups
CREATE INDEX idx_p049_x_alloc_run_site
    ON ghg_multisite.gl_ms_allocation_results(run_id, site_id);

-- =============================================================================
-- Seed Data: Reference Lookups
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Seed: Facility Types (20)
-- ---------------------------------------------------------------------------
-- Note: Facility types are enforced via CHECK constraint on gl_ms_sites.
-- This comment documents the canonical list for application-level reference.
--
-- Canonical facility types:
--   1.  MANUFACTURING     - Production/assembly facilities
--   2.  OFFICE            - Administrative/corporate offices
--   3.  WAREHOUSE         - Storage and distribution centres
--   4.  RETAIL            - Customer-facing retail locations
--   5.  DATA_CENTER       - IT infrastructure/server facilities
--   6.  LABORATORY        - Research and testing laboratories
--   7.  HOSPITAL          - Healthcare facilities
--   8.  HOTEL             - Hospitality/accommodation
--   9.  SCHOOL            - Primary/secondary education
--  10.  UNIVERSITY        - Higher education/research campuses
--  11.  TRANSPORT_HUB     - Bus/rail/intermodal stations
--  12.  PORT              - Maritime/inland port facilities
--  13.  AIRPORT           - Aviation facilities
--  14.  MINE              - Extraction/mining operations
--  15.  REFINERY          - Oil/gas/chemical refining
--  16.  POWER_PLANT       - Electricity generation
--  17.  FARM              - Agricultural operations
--  18.  MIXED_USE         - Multi-purpose facilities
--  19.  CONSTRUCTION_SITE - Temporary construction operations
--  20.  OTHER             - Uncategorised facilities

-- ---------------------------------------------------------------------------
-- Seed: Allocation Methods Reference Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_multisite.gl_ms_ref_allocation_methods (
    id              SERIAL      PRIMARY KEY,
    method_code     VARCHAR(30) NOT NULL UNIQUE,
    method_name     VARCHAR(100) NOT NULL,
    description     TEXT,
    unit            VARCHAR(50),
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_multisite.gl_ms_ref_allocation_methods
    (method_code, method_name, description, unit, sort_order)
VALUES
    ('FLOOR_AREA',      'Floor Area',       'Allocate by occupied floor area (m2 or sqft)',                 'm2',       1),
    ('HEADCOUNT',       'Headcount',        'Allocate by full-time equivalent employee count',              'FTE',      2),
    ('REVENUE',         'Revenue',          'Allocate by site revenue as proportion of total',              'currency', 3),
    ('PRODUCTION',      'Production',       'Allocate by production output (units, tonnes, etc.)',          'units',    4),
    ('ENERGY_USE',      'Energy Use',       'Allocate by energy consumption (MWh)',                         'MWh',      5),
    ('OPERATING_HOURS', 'Operating Hours',  'Allocate by operational hours per period',                     'hours',    6),
    ('CUSTOM',          'Custom',           'Allocate by user-defined metric with custom denominator',      'custom',   7)
ON CONFLICT (method_code) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: Quality Dimensions Reference Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_multisite.gl_ms_ref_quality_dimensions (
    id              SERIAL      PRIMARY KEY,
    dimension_code  VARCHAR(30) NOT NULL UNIQUE,
    dimension_name  VARCHAR(100) NOT NULL,
    description     TEXT,
    default_weight  NUMERIC(10,4) NOT NULL DEFAULT 1.0000,
    default_threshold NUMERIC(10,4) NOT NULL DEFAULT 50.0000,
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_multisite.gl_ms_ref_quality_dimensions
    (dimension_code, dimension_name, description, default_weight, default_threshold, sort_order)
VALUES
    ('COMPLETENESS',  'Completeness',  'Coverage of all required data points, months, scopes, and categories',  1.0000, 50.0000, 1),
    ('ACCURACY',      'Accuracy',      'Correctness of emission factors, calculations, and unit conversions',   1.0000, 50.0000, 2),
    ('CONSISTENCY',   'Consistency',   'Internal consistency across scopes, periods, and related data points',  1.0000, 50.0000, 3),
    ('TIMELINESS',    'Timeliness',    'Data submitted within deadlines and reflects current reporting period',  0.8000, 40.0000, 4),
    ('TRANSPARENCY',  'Transparency',  'Documentation of methods, assumptions, emission factors, and sources',  0.8000, 40.0000, 5),
    ('RELIABILITY',   'Reliability',   'Data from verifiable sources with evidence and audit trail',            0.8000, 40.0000, 6)
ON CONFLICT (dimension_code) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: KPI Types Reference Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_multisite.gl_ms_ref_kpi_types (
    id              SERIAL      PRIMARY KEY,
    kpi_code        VARCHAR(50) NOT NULL UNIQUE,
    kpi_name        VARCHAR(100) NOT NULL,
    description     TEXT,
    numerator_unit  VARCHAR(50) NOT NULL DEFAULT 'tCO2e',
    denominator_unit VARCHAR(50),
    result_unit     VARCHAR(100) NOT NULL,
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_multisite.gl_ms_ref_kpi_types
    (kpi_code, kpi_name, description, numerator_unit, denominator_unit, result_unit, sort_order)
VALUES
    ('INTENSITY_PER_M2',      'Emission Intensity per m2',       'tCO2e per square metre of floor area',         'tCO2e', 'm2',       'tCO2e/m2',       1),
    ('INTENSITY_PER_FTE',     'Emission Intensity per FTE',      'tCO2e per full-time equivalent employee',      'tCO2e', 'FTE',      'tCO2e/FTE',      2),
    ('INTENSITY_PER_REVENUE', 'Emission Intensity per Revenue',  'tCO2e per million currency units of revenue',  'tCO2e', 'M_curr',   'tCO2e/M_curr',   3),
    ('INTENSITY_PER_UNIT',    'Emission Intensity per Unit',     'tCO2e per unit of production output',          'tCO2e', 'unit',     'tCO2e/unit',     4),
    ('INTENSITY_PER_MWH',     'Emission Intensity per MWh',      'tCO2e per megawatt-hour of energy consumed',   'tCO2e', 'MWh',      'tCO2e/MWh',      5),
    ('INTENSITY_PER_HOUR',    'Emission Intensity per Hour',     'tCO2e per operating hour',                     'tCO2e', 'hour',     'tCO2e/hour',     6),
    ('ABSOLUTE_EMISSIONS',    'Absolute Emissions',              'Total emissions in tCO2e (no denominator)',     'tCO2e', NULL,       'tCO2e',          7),
    ('ENERGY_CONSUMPTION',    'Energy Consumption',              'Total energy consumed in MWh',                 'MWh',   NULL,       'MWh',            8)
ON CONFLICT (kpi_code) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: Elimination Types Reference Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_multisite.gl_ms_ref_elimination_types (
    id              SERIAL      PRIMARY KEY,
    type_code       VARCHAR(30) NOT NULL UNIQUE,
    type_name       VARCHAR(100) NOT NULL,
    description     TEXT,
    typical_scope   VARCHAR(10),
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_multisite.gl_ms_ref_elimination_types
    (type_code, type_name, description, typical_scope, sort_order)
VALUES
    ('INTERNAL_ENERGY',     'Internal Energy Transfer',     'Electricity, heat, or steam transferred between sites within boundary',   'SCOPE_2', 1),
    ('INTERNAL_TRANSPORT',  'Internal Transport',           'Freight or logistics between sites within the same entity',               'SCOPE_3', 2),
    ('INTERNAL_WASTE',      'Internal Waste Processing',    'Waste processed at another site within the same boundary',                'SCOPE_3', 3),
    ('INTERCOMPANY_SALES',  'Intercompany Sales',           'Purchased goods/services between entities in the consolidated group',     'SCOPE_3', 4),
    ('SHARED_SERVICES',     'Shared Services',              'Shared corporate services allocated to multiple sites',                   'SCOPE_3', 5),
    ('OTHER',               'Other Elimination',            'Other intercompany or intra-group eliminations not classified above',     NULL,      6)
ON CONFLICT (type_code) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: Factor Tiers Reference Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_multisite.gl_ms_ref_factor_tiers (
    id              SERIAL      PRIMARY KEY,
    tier_number     INTEGER     NOT NULL UNIQUE,
    tier_name       VARCHAR(100) NOT NULL,
    description     TEXT,
    data_quality    VARCHAR(20) NOT NULL,
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_multisite.gl_ms_ref_factor_tiers
    (tier_number, tier_name, description, data_quality, sort_order)
VALUES
    (1, 'Tier 1 - Default',    'Country or global average emission factors from published databases (DEFRA, EPA, IEA)',   'LOW',    1),
    (2, 'Tier 2 - Regional',   'Region or grid-specific emission factors with location granularity',                      'MEDIUM', 2),
    (3, 'Tier 3 - Specific',   'Supplier-specific, product-specific, or technology-specific emission factors',            'HIGH',   3),
    (4, 'Tier 4 - Measured',   'Direct measurement or continuous emissions monitoring data (CEMS)',                        'HIGHEST',4)
ON CONFLICT (tier_number) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: Default Reminder Schedule
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_multisite.gl_ms_ref_reminder_schedule (
    id              SERIAL      PRIMARY KEY,
    reminder_code   VARCHAR(30) NOT NULL UNIQUE,
    reminder_name   VARCHAR(100) NOT NULL,
    description     TEXT,
    days_before_deadline INTEGER NOT NULL,
    template_id     VARCHAR(100),
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_multisite.gl_ms_ref_reminder_schedule
    (reminder_code, reminder_name, description, days_before_deadline, template_id, sort_order)
VALUES
    ('INITIAL',       'Initial Notification',     'Sent when collection round opens',                     30, 'tpl_initial_notify',     1),
    ('FIRST_REMINDER','First Reminder',           'Gentle reminder sent before deadline',                 14, 'tpl_first_reminder',     2),
    ('SECOND_REMINDER','Second Reminder',         'Follow-up reminder for non-responders',                7,  'tpl_second_reminder',    3),
    ('DEADLINE_WARNING','Deadline Warning',        'Urgent warning sent close to deadline',                3,  'tpl_deadline_warning',   4),
    ('DEADLINE_DAY',  'Deadline Day',             'Final notice on the day of the deadline',              0,  'tpl_deadline_day',       5),
    ('OVERDUE_1',     'First Overdue Notice',     'Sent 1 day after deadline for non-submitters',         -1, 'tpl_overdue_first',      6),
    ('OVERDUE_3',     'Second Overdue Notice',    'Sent 3 days after deadline with escalation warning',   -3, 'tpl_overdue_second',     7),
    ('ESCALATION',    'Escalation Notice',        'Escalated to site manager after extended non-response', -7, 'tpl_escalation',         8)
ON CONFLICT (reminder_code) DO NOTHING;

-- =============================================================================
-- Comments on Materialized Views
-- =============================================================================
COMMENT ON MATERIALIZED VIEW ghg_multisite.mv_p049_site_portfolio_summary IS
    'PACK-049: Site portfolio dashboard view with latest emissions, quality, and completion status.';
COMMENT ON MATERIALIZED VIEW ghg_multisite.mv_p049_consolidation_summary IS
    'PACK-049: Period-level consolidation results with scope totals, eliminations, and Scope 1+2 composites.';
COMMENT ON MATERIALIZED VIEW ghg_multisite.mv_p049_quality_heatmap IS
    'PACK-049: Site-by-dimension quality matrix for heatmap visualisation with HIGH/MEDIUM/LOW bands.';
COMMENT ON MATERIALIZED VIEW ghg_multisite.mv_p049_completion_overview IS
    'PACK-049: Period-level completion overview with traffic light counts, median completion, and source coverage.';

-- Comments on Reference Tables
COMMENT ON TABLE ghg_multisite.gl_ms_ref_allocation_methods IS
    'PACK-049 Seed: 7 canonical allocation methods (floor area, headcount, revenue, production, energy, hours, custom).';
COMMENT ON TABLE ghg_multisite.gl_ms_ref_quality_dimensions IS
    'PACK-049 Seed: 6 quality dimensions with default weights and thresholds.';
COMMENT ON TABLE ghg_multisite.gl_ms_ref_kpi_types IS
    'PACK-049 Seed: 8 KPI types (6 intensity + 2 absolute) with unit specifications.';
COMMENT ON TABLE ghg_multisite.gl_ms_ref_elimination_types IS
    'PACK-049 Seed: 6 elimination types for intercompany double-counting prevention.';
COMMENT ON TABLE ghg_multisite.gl_ms_ref_factor_tiers IS
    'PACK-049 Seed: 4 emission factor tiers (Default/Regional/Specific/Measured) per GHG Protocol guidance.';
COMMENT ON TABLE ghg_multisite.gl_ms_ref_reminder_schedule IS
    'PACK-049 Seed: 8-step default reminder schedule from initial notification to escalation.';

-- =============================================================================
-- Final Pack Comment
-- =============================================================================
COMMENT ON SCHEMA ghg_multisite IS
    'PACK-049: GHG Multi-Site Management - 10 migrations, 31+ tables, 4 materialized views, '
    '6 seed reference tables. Covers site registry, data collection, boundary management, '
    'regional factors, consolidation, allocation, comparison, quality, completion tracking, '
    'and portfolio reporting for multi-site emission management.';
