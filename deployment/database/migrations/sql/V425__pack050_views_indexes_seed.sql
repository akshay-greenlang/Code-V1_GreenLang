-- =============================================================================
-- PACK-050 GHG Consolidation Pack
-- Migration: V425 - Views, Additional Indexes & Seed Data
-- =============================================================================
-- Pack:         PACK-050 (GHG Consolidation Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates materialized views for dashboard performance, additional
-- cross-table indexes for common query patterns, and seed data for
-- default settings. This is the final migration in the PACK-050 series.
--
-- Views (4):
--   1. gl_cons_entity_summary_v
--   2. gl_cons_consolidated_totals_v
--   3. gl_cons_elimination_summary_v
--   4. gl_cons_completeness_v
--
-- Seed Data:
--   - Default consolidation approaches reference
--   - Control type reference
--   - Data quality tiers reference
--   - Report frameworks reference
--   - Default signoff levels reference
--   - Default settings template
--
-- Previous: V424__pack050_reporting.sql
-- =============================================================================

SET search_path TO ghg_consolidation, public;

-- =============================================================================
-- Materialized View 1: Entity Summary
-- =============================================================================
-- Aggregated view of all entities with their latest ownership, lifecycle
-- status, and most recent submission totals. Optimised for entity
-- portfolio dashboards and drill-down views.

CREATE MATERIALIZED VIEW ghg_consolidation.gl_cons_entity_summary_v AS
SELECT
    e.tenant_id,
    e.id                                AS entity_id,
    e.entity_code,
    e.legal_name,
    e.trading_name,
    e.entity_type::text                 AS entity_type,
    e.lifecycle_status::text            AS lifecycle_status,
    e.jurisdiction,
    e.sector_code,
    e.is_reporting_entity,
    e.is_active,
    -- Parent info
    e.parent_entity_id,
    p.legal_name                        AS parent_legal_name,
    p.entity_code                       AS parent_entity_code,
    -- Current ownership (most recent active)
    COALESCE(own.ownership_pct, 0)      AS current_ownership_pct,
    own.control_type::text              AS current_control_type,
    -- Effective ownership from equity chain
    COALESCE(ec.effective_pct, 0)       AS effective_ownership_pct,
    -- Hierarchy depth
    COALESCE(eh.depth, 0)               AS hierarchy_depth,
    -- Latest submission totals
    COALESCE(sub.total_scope1, 0)       AS latest_scope1_tco2e,
    COALESCE(sub.total_scope2l, 0)      AS latest_scope2_location_tco2e,
    COALESCE(sub.total_scope2m, 0)      AS latest_scope2_market_tco2e,
    COALESCE(sub.total_scope3, 0)       AS latest_scope3_tco2e,
    COALESCE(sub.total_all, 0)          AS latest_total_tco2e,
    sub.avg_dq_tier                     AS latest_avg_data_quality_tier,
    sub.submission_count                AS total_submissions,
    -- Timestamps
    e.created_at                        AS entity_created_at,
    e.updated_at                        AS entity_updated_at,
    NOW()                               AS refreshed_at
FROM ghg_consolidation.gl_cons_entities e
LEFT JOIN ghg_consolidation.gl_cons_entities p
    ON p.id = e.parent_entity_id
LEFT JOIN LATERAL (
    SELECT o.ownership_pct, o.control_type
    FROM ghg_consolidation.gl_cons_ownership o
    WHERE o.owned_entity_id = e.id
      AND o.effective_to IS NULL
    ORDER BY o.effective_from DESC
    LIMIT 1
) own ON true
LEFT JOIN LATERAL (
    SELECT ec2.effective_pct
    FROM ghg_consolidation.gl_cons_equity_chains ec2
    WHERE ec2.target_entity_id = e.id
      AND ec2.is_current = true
    ORDER BY ec2.resolved_at DESC
    LIMIT 1
) ec ON true
LEFT JOIN LATERAL (
    SELECT eh2.depth
    FROM ghg_consolidation.gl_cons_entity_hierarchy eh2
    WHERE eh2.descendant_id = e.id
      AND eh2.ancestor_id != eh2.descendant_id
    ORDER BY eh2.depth ASC
    LIMIT 1
) eh ON true
LEFT JOIN LATERAL (
    SELECT
        SUM(CASE WHEN es.scope = 'SCOPE_1' THEN es.emissions_tco2e ELSE 0 END)             AS total_scope1,
        SUM(CASE WHEN es.scope = 'SCOPE_2_LOCATION' THEN es.emissions_tco2e ELSE 0 END)    AS total_scope2l,
        SUM(CASE WHEN es.scope = 'SCOPE_2_MARKET' THEN es.emissions_tco2e ELSE 0 END)      AS total_scope2m,
        SUM(CASE WHEN es.scope = 'SCOPE_3' THEN es.emissions_tco2e ELSE 0 END)             AS total_scope3,
        SUM(es.emissions_tco2e)                                                              AS total_all,
        AVG(es.data_quality_tier)                                                            AS avg_dq_tier,
        COUNT(*)                                                                             AS submission_count
    FROM ghg_consolidation.gl_cons_entity_submissions es
    JOIN ghg_consolidation.gl_cons_data_requests dr ON dr.id = es.request_id
    WHERE es.entity_id = e.id
      AND es.validation_status IN ('VALID', 'WARNING', 'OVERRIDE')
      AND dr.status = 'APPROVED'
) sub ON true
WITH NO DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX uidx_p050_entity_summary
    ON ghg_consolidation.gl_cons_entity_summary_v(tenant_id, entity_id);
CREATE INDEX idx_p050_esv_type
    ON ghg_consolidation.gl_cons_entity_summary_v(entity_type);
CREATE INDEX idx_p050_esv_lifecycle
    ON ghg_consolidation.gl_cons_entity_summary_v(lifecycle_status);
CREATE INDEX idx_p050_esv_jurisdiction
    ON ghg_consolidation.gl_cons_entity_summary_v(jurisdiction);
CREATE INDEX idx_p050_esv_parent
    ON ghg_consolidation.gl_cons_entity_summary_v(parent_entity_id);
CREATE INDEX idx_p050_esv_reporting
    ON ghg_consolidation.gl_cons_entity_summary_v(is_reporting_entity)
    WHERE is_reporting_entity = true;

-- =============================================================================
-- Materialized View 2: Consolidated Totals
-- =============================================================================
-- Period-level consolidated emission totals from approved/completed runs.
-- Provides a single-row-per-period summary for trending and reporting.

CREATE MATERIALIZED VIEW ghg_consolidation.gl_cons_consolidated_totals_v AS
SELECT
    cr.tenant_id,
    cr.boundary_id,
    b.boundary_name,
    cr.id                                   AS run_id,
    cr.run_number,
    cr.run_type,
    cr.approach::text                       AS consolidation_approach,
    cr.status,
    cr.reporting_period_start,
    cr.reporting_period_end,
    cr.entities_included,
    cr.entities_excluded,
    cr.completeness_pct,
    -- Scope totals
    cr.scope1_total_tco2e,
    cr.scope2_location_total_tco2e,
    cr.scope2_market_total_tco2e,
    cr.scope3_total_tco2e,
    cr.total_tco2e,
    cr.eliminations_tco2e,
    cr.net_total_tco2e,
    cr.biogenic_total_tco2e,
    -- Computed composites
    (cr.scope1_total_tco2e + cr.scope2_location_total_tco2e)   AS scope1_2_location_tco2e,
    (cr.scope1_total_tco2e + cr.scope2_market_total_tco2e)     AS scope1_2_market_tco2e,
    -- Timing
    cr.started_at,
    cr.completed_at,
    cr.approved_at,
    cr.provenance_hash,
    NOW()                                   AS refreshed_at
FROM ghg_consolidation.gl_cons_consolidation_runs cr
JOIN ghg_consolidation.gl_cons_boundaries b
    ON b.id = cr.boundary_id
WHERE cr.status IN ('COMPLETED', 'APPROVED')
WITH NO DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX uidx_p050_consol_totals_run
    ON ghg_consolidation.gl_cons_consolidated_totals_v(tenant_id, run_id);
CREATE INDEX idx_p050_ctv_boundary
    ON ghg_consolidation.gl_cons_consolidated_totals_v(boundary_id);
CREATE INDEX idx_p050_ctv_type
    ON ghg_consolidation.gl_cons_consolidated_totals_v(run_type);
CREATE INDEX idx_p050_ctv_status
    ON ghg_consolidation.gl_cons_consolidated_totals_v(status);
CREATE INDEX idx_p050_ctv_period
    ON ghg_consolidation.gl_cons_consolidated_totals_v(reporting_period_start, reporting_period_end);
CREATE INDEX idx_p050_ctv_approved
    ON ghg_consolidation.gl_cons_consolidated_totals_v(approved_at)
    WHERE approved_at IS NOT NULL;

-- =============================================================================
-- Materialized View 3: Elimination Summary
-- =============================================================================
-- Summary of intercompany eliminations by type and scope within each
-- consolidation run. Provides aggregate elimination amounts for
-- reporting and reconciliation dashboards.

CREATE MATERIALIZED VIEW ghg_consolidation.gl_cons_elimination_summary_v AS
SELECT
    el.tenant_id,
    el.run_id,
    cr.reporting_period_start,
    cr.reporting_period_end,
    el.elimination_type,
    el.elimination_scope,
    COUNT(*)                                AS elimination_count,
    COUNT(DISTINCT el.transfer_id)          AS transfer_count,
    COUNT(DISTINCT el.seller_entity_id)     AS seller_count,
    COUNT(DISTINCT el.buyer_entity_id)      AS buyer_count,
    SUM(el.gross_amount_tco2e)              AS total_gross_tco2e,
    SUM(el.elimination_amount_tco2e)        AS total_elimination_tco2e,
    SUM(el.net_amount_tco2e)                AS total_net_tco2e,
    AVG(el.elimination_pct)                 AS avg_elimination_pct,
    COUNT(*) FILTER (WHERE el.is_automatic) AS automatic_count,
    COUNT(*) FILTER (WHERE el.is_partial)   AS partial_count,
    NOW()                                   AS refreshed_at
FROM ghg_consolidation.gl_cons_eliminations el
JOIN ghg_consolidation.gl_cons_consolidation_runs cr
    ON cr.id = el.run_id
WHERE cr.status IN ('COMPLETED', 'APPROVED')
GROUP BY
    el.tenant_id, el.run_id,
    cr.reporting_period_start, cr.reporting_period_end,
    el.elimination_type, el.elimination_scope
WITH NO DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX uidx_p050_elim_summary
    ON ghg_consolidation.gl_cons_elimination_summary_v(tenant_id, run_id, elimination_type, elimination_scope);
CREATE INDEX idx_p050_elsv_run
    ON ghg_consolidation.gl_cons_elimination_summary_v(run_id);
CREATE INDEX idx_p050_elsv_type
    ON ghg_consolidation.gl_cons_elimination_summary_v(elimination_type);
CREATE INDEX idx_p050_elsv_scope
    ON ghg_consolidation.gl_cons_elimination_summary_v(elimination_scope);
CREATE INDEX idx_p050_elsv_period
    ON ghg_consolidation.gl_cons_elimination_summary_v(reporting_period_start, reporting_period_end);

-- =============================================================================
-- Materialized View 4: Completeness
-- =============================================================================
-- Data collection completeness per boundary showing request/submission
-- status counts, average data quality, and estimated coverage gaps.

CREATE MATERIALIZED VIEW ghg_consolidation.gl_cons_completeness_v AS
SELECT
    dr.tenant_id,
    dr.boundary_id,
    b.boundary_name,
    b.reporting_period_start,
    b.reporting_period_end,
    b.approach::text                                            AS consolidation_approach,
    -- Request counts
    COUNT(*)                                                    AS total_requests,
    COUNT(*) FILTER (WHERE dr.status = 'APPROVED')              AS approved_count,
    COUNT(*) FILTER (WHERE dr.status = 'SUBMITTED')             AS submitted_count,
    COUNT(*) FILTER (WHERE dr.status IN ('PENDING', 'ASSIGNED', 'IN_PROGRESS')) AS pending_count,
    COUNT(*) FILTER (WHERE dr.status = 'OVERDUE')               AS overdue_count,
    COUNT(*) FILTER (WHERE dr.status = 'REJECTED')              AS rejected_count,
    COUNT(*) FILTER (WHERE dr.status = 'CANCELLED')             AS cancelled_count,
    -- Completion percentage
    CASE
        WHEN COUNT(*) FILTER (WHERE dr.status != 'CANCELLED') > 0
        THEN ROUND(
            COUNT(*) FILTER (WHERE dr.status = 'APPROVED')::numeric /
            COUNT(*) FILTER (WHERE dr.status != 'CANCELLED') * 100, 2
        )
        ELSE 0
    END                                                         AS completion_pct,
    -- Submission quality
    AVG(es_stats.avg_dq_tier)                                   AS avg_data_quality_tier,
    AVG(es_stats.avg_dq_score)                                  AS avg_data_quality_score,
    SUM(es_stats.submission_count)                               AS total_submissions,
    SUM(es_stats.estimated_count)                                AS estimated_submissions,
    -- Due date info
    MIN(dr.due_date)                                            AS earliest_due_date,
    MAX(dr.due_date)                                            AS latest_due_date,
    COUNT(*) FILTER (WHERE dr.due_date < CURRENT_DATE AND dr.status NOT IN ('APPROVED', 'CANCELLED')) AS past_due_count,
    NOW()                                                       AS refreshed_at
FROM ghg_consolidation.gl_cons_data_requests dr
JOIN ghg_consolidation.gl_cons_boundaries b
    ON b.id = dr.boundary_id
LEFT JOIN LATERAL (
    SELECT
        AVG(es.data_quality_tier)   AS avg_dq_tier,
        AVG(es.data_quality_score)  AS avg_dq_score,
        COUNT(*)                    AS submission_count,
        COUNT(*) FILTER (WHERE es.is_estimated = true) AS estimated_count
    FROM ghg_consolidation.gl_cons_entity_submissions es
    WHERE es.request_id = dr.id
) es_stats ON true
GROUP BY
    dr.tenant_id, dr.boundary_id,
    b.boundary_name, b.reporting_period_start, b.reporting_period_end, b.approach
WITH NO DATA;

-- Unique index for concurrent refresh
CREATE UNIQUE INDEX uidx_p050_completeness
    ON ghg_consolidation.gl_cons_completeness_v(tenant_id, boundary_id);
CREATE INDEX idx_p050_cv_boundary
    ON ghg_consolidation.gl_cons_completeness_v(boundary_id);
CREATE INDEX idx_p050_cv_period
    ON ghg_consolidation.gl_cons_completeness_v(reporting_period_start, reporting_period_end);
CREATE INDEX idx_p050_cv_completion
    ON ghg_consolidation.gl_cons_completeness_v(completion_pct);

-- =============================================================================
-- Additional Cross-Table Indexes
-- =============================================================================

-- Entity + ownership lookups (most common query pattern)
CREATE INDEX idx_p050_x_own_entity_active
    ON ghg_consolidation.gl_cons_ownership(owned_entity_id, effective_from)
    WHERE effective_to IS NULL;

-- Hierarchy root lookups
CREATE INDEX idx_p050_x_eh_root_depth0
    ON ghg_consolidation.gl_cons_entity_hierarchy(tenant_id, ancestor_id)
    WHERE depth = 0;

-- Boundary + inclusion + entity join path
CREATE INDEX idx_p050_x_ei_boundary_status
    ON ghg_consolidation.gl_cons_entity_inclusions(boundary_id, inclusion_status, entity_id);

-- Data request + entity + status join path
CREATE INDEX idx_p050_x_dr_entity_status
    ON ghg_consolidation.gl_cons_data_requests(entity_id, status);

-- Consolidation run + entity adjustment join path
CREATE INDEX idx_p050_x_ea_run_entity_scope
    ON ghg_consolidation.gl_cons_entity_adjustments(run_id, entity_id, scope);

-- Transfer register temporal lookups
CREATE INDEX idx_p050_x_tr_seller_period
    ON ghg_consolidation.gl_cons_transfer_register(seller_entity_id, period_start, period_end);
CREATE INDEX idx_p050_x_tr_buyer_period
    ON ghg_consolidation.gl_cons_transfer_register(buyer_entity_id, period_start, period_end);

-- Signoff workflow lookups
CREATE INDEX idx_p050_x_so_run_level_status
    ON ghg_consolidation.gl_cons_signoffs(run_id, level_order, status);

-- M&A event date range lookups
CREATE INDEX idx_p050_x_mna_tenant_date
    ON ghg_consolidation.gl_cons_mna_events(tenant_id, event_date, effective_date);

-- Audit log fast lookups by entity + time
CREATE INDEX idx_p050_x_al_entity_time
    ON ghg_consolidation.gl_cons_audit_log(entity_type, entity_id, created_at DESC);

-- =============================================================================
-- Seed Data: Reference Lookups
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Seed: Consolidation Approaches Reference Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_consolidation.gl_cons_ref_approaches (
    id              SERIAL      PRIMARY KEY,
    approach_code   VARCHAR(30) NOT NULL UNIQUE,
    approach_name   VARCHAR(100) NOT NULL,
    description     TEXT,
    ghg_protocol_ref VARCHAR(100),
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_consolidation.gl_cons_ref_approaches
    (approach_code, approach_name, description, ghg_protocol_ref, sort_order)
VALUES
    ('OPERATIONAL_CONTROL', 'Operational Control',  'Account for 100% of emissions from operations over which the company has operational control. Most common approach for corporate reporting.',  'GHG Protocol Ch.3', 1),
    ('FINANCIAL_CONTROL',   'Financial Control',    'Account for 100% of emissions from operations over which the company has financial control (ability to direct financial and operating policies).',  'GHG Protocol Ch.3', 2),
    ('EQUITY_SHARE',        'Equity Share',         'Account for emissions proportional to the equity interest in each operation. Aligns with financial accounting treatment.',  'GHG Protocol Ch.3', 3)
ON CONFLICT (approach_code) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: Control Types Reference Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_consolidation.gl_cons_ref_control_types (
    id              SERIAL      PRIMARY KEY,
    type_code       VARCHAR(30) NOT NULL UNIQUE,
    type_name       VARCHAR(100) NOT NULL,
    description     TEXT,
    default_inclusion_pct NUMERIC(10,4),
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_consolidation.gl_cons_ref_control_types
    (type_code, type_name, description, default_inclusion_pct, sort_order)
VALUES
    ('OPERATIONAL',             'Operational Control',          'Company has authority to introduce and implement operating policies',               100.0000, 1),
    ('FINANCIAL',               'Financial Control',            'Company has ability to direct financial and operating policies with a view to gain economic benefit', 100.0000, 2),
    ('JOINT_OPERATIONAL',       'Joint Operational Control',    'Shared operational control with one or more partners',                              50.0000, 3),
    ('JOINT_FINANCIAL',         'Joint Financial Control',      'Shared financial control with one or more partners',                                50.0000, 4),
    ('NO_CONTROL',              'No Control',                   'No operational or financial control (e.g., minority investment)',                    0.0000, 5),
    ('SIGNIFICANT_INFLUENCE',   'Significant Influence',        'Ability to participate in decisions but not control (typically 20-50% ownership)',   NULL, 6)
ON CONFLICT (type_code) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: Data Quality Tiers Reference Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_consolidation.gl_cons_ref_data_quality_tiers (
    id              SERIAL      PRIMARY KEY,
    tier_number     INTEGER     NOT NULL UNIQUE,
    tier_name       VARCHAR(100) NOT NULL,
    description     TEXT,
    data_quality    VARCHAR(20) NOT NULL,
    uncertainty_range VARCHAR(50),
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_consolidation.gl_cons_ref_data_quality_tiers
    (tier_number, tier_name, description, data_quality, uncertainty_range, sort_order)
VALUES
    (1, 'Tier 1 - Spend-Based',    'Spend-based estimates using environmentally extended input-output (EEIO) factors',             'LOW',     '+/- 50-100%', 1),
    (2, 'Tier 2 - Average Data',   'Estimated using average or proxy emission factors (e.g., industry averages, DEFRA/EPA)',        'MEDIUM',  '+/- 25-50%',  2),
    (3, 'Tier 3 - Supplier Data',  'Supplier-specific or product-level emission factors with documented methodology',              'HIGH',    '+/- 10-25%',  3),
    (4, 'Tier 4 - Primary Data',   'Primary data from direct measurement, metering, or continuous emissions monitoring',           'HIGHEST', '+/- 5-10%',   4),
    (5, 'Tier 5 - Verified Data',  'Primary data that has been externally verified or assured by an independent third party',       'VERIFIED','+/- 2-5%',    5)
ON CONFLICT (tier_number) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: Report Frameworks Reference Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_consolidation.gl_cons_ref_frameworks (
    id              SERIAL      PRIMARY KEY,
    framework_code  VARCHAR(30) NOT NULL UNIQUE,
    framework_name  VARCHAR(200) NOT NULL,
    description     TEXT,
    version         VARCHAR(20),
    website_url     VARCHAR(500),
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_consolidation.gl_cons_ref_frameworks
    (framework_code, framework_name, description, version, sort_order)
VALUES
    ('GHG_PROTOCOL', 'GHG Protocol Corporate Standard',             'World Resources Institute corporate accounting and reporting standard',      'Rev. 2015', 1),
    ('CDP',          'CDP Climate Change Questionnaire',             'Global disclosure system for environmental information',                     '2025',      2),
    ('CSRD',         'Corporate Sustainability Reporting Directive', 'EU directive requiring sustainability reporting per ESRS standards',         'ESRS 2024', 3),
    ('TCFD',         'Task Force on Climate-related Financial Disclosures', 'Recommendations for climate risk and opportunity disclosure',        '2021',      4),
    ('ISO_14064',    'ISO 14064-1',                                 'International standard for quantifying and reporting GHG emissions',         '2018',      5),
    ('SECR',         'Streamlined Energy and Carbon Reporting',      'UK mandatory energy and carbon reporting for qualifying companies',         '2019',      6),
    ('NGER',         'National Greenhouse and Energy Reporting',     'Australian mandatory reporting of GHG emissions and energy',                '2024',      7),
    ('EPA',          'EPA Greenhouse Gas Reporting Program',         'US mandatory reporting of GHG emissions from large sources',                '2024',      8),
    ('SBTi',         'Science Based Targets initiative',             'Framework for setting emission reduction targets aligned with climate science', 'v5.1',  9),
    ('CUSTOM',       'Custom Framework',                            'User-defined reporting framework',                                         NULL,        10)
ON CONFLICT (framework_code) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: Signoff Levels Reference Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ghg_consolidation.gl_cons_ref_signoff_levels (
    id              SERIAL      PRIMARY KEY,
    level_code      VARCHAR(30) NOT NULL UNIQUE,
    level_name      VARCHAR(100) NOT NULL,
    description     TEXT,
    typical_role    VARCHAR(100),
    is_mandatory    BOOLEAN     NOT NULL DEFAULT false,
    sort_order      INTEGER     NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT true
);

INSERT INTO ghg_consolidation.gl_cons_ref_signoff_levels
    (level_code, level_name, description, typical_role, is_mandatory, sort_order)
VALUES
    ('PREPARER',        'Preparer',         'Person who prepared the consolidated inventory and calculations',  'Sustainability Analyst',   true,  1),
    ('REVIEWER',        'Reviewer',         'Technical reviewer who checks calculations, methodology, and data quality', 'Senior Analyst',  true,  2),
    ('SENIOR_REVIEWER', 'Senior Reviewer',  'Senior technical review for complex consolidation issues',         'Sustainability Manager',   false, 3),
    ('APPROVER',        'Approver',         'Business approver who authorises the consolidated results',        'Head of Sustainability',   true,  4),
    ('EXECUTIVE',       'Executive',        'Executive sponsor who endorses the corporate GHG disclosure',      'CFO / CSO',                false, 5),
    ('BOARD',           'Board',            'Board-level review and sign-off for public disclosures',           'Board / Audit Committee',  false, 6),
    ('EXTERNAL',        'External Assurer', 'Independent external assurance provider',                          'Assurance Partner',        false, 7)
ON CONFLICT (level_code) DO NOTHING;

-- =============================================================================
-- Comments on Materialized Views
-- =============================================================================
COMMENT ON MATERIALIZED VIEW ghg_consolidation.gl_cons_entity_summary_v IS
    'PACK-050: Entity portfolio summary with ownership, hierarchy depth, and latest submission totals.';
COMMENT ON MATERIALIZED VIEW ghg_consolidation.gl_cons_consolidated_totals_v IS
    'PACK-050: Period-level consolidated totals from completed/approved runs with scope composites.';
COMMENT ON MATERIALIZED VIEW ghg_consolidation.gl_cons_elimination_summary_v IS
    'PACK-050: Elimination summary by type/scope with counts, totals, and automatic/partial breakdowns.';
COMMENT ON MATERIALIZED VIEW ghg_consolidation.gl_cons_completeness_v IS
    'PACK-050: Data collection completeness per boundary with request status counts and quality metrics.';

-- Comments on Reference Tables
COMMENT ON TABLE ghg_consolidation.gl_cons_ref_approaches IS
    'PACK-050 Seed: 3 GHG Protocol consolidation approaches with Chapter 3 references.';
COMMENT ON TABLE ghg_consolidation.gl_cons_ref_control_types IS
    'PACK-050 Seed: 6 control types with default inclusion percentages for boundary determination.';
COMMENT ON TABLE ghg_consolidation.gl_cons_ref_data_quality_tiers IS
    'PACK-050 Seed: 5 data quality tiers from spend-based (Tier 1) to externally verified (Tier 5).';
COMMENT ON TABLE ghg_consolidation.gl_cons_ref_frameworks IS
    'PACK-050 Seed: 10 reporting frameworks (GHG Protocol, CDP, CSRD, TCFD, ISO 14064, SECR, NGER, EPA, SBTi, Custom).';
COMMENT ON TABLE ghg_consolidation.gl_cons_ref_signoff_levels IS
    'PACK-050 Seed: 7 signoff levels from preparer to external assurer with mandatory flags.';

-- =============================================================================
-- Final Pack Comment
-- =============================================================================
COMMENT ON SCHEMA ghg_consolidation IS
    'PACK-050: GHG Consolidation Pack - 10 migrations, 19 core tables, 5 enums, '
    '4 materialized views, 5 seed reference tables. Covers entity registry, '
    'ownership/equity chains, organisational boundaries, data collection, '
    'consolidation runs, intercompany eliminations, M&A events, base year '
    'restatements, adjustments, reporting, signoffs, and assurance packages '
    'for multi-entity corporate GHG consolidation per GHG Protocol.';
