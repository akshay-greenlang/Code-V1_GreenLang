-- =============================================================================
-- V180: PACK-027 Enterprise Net Zero - Views & Performance Indexes
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    015 of 015
-- Date:         March 2026
--
-- Enterprise dashboard views and comprehensive performance indexes for
-- large-scale enterprise operations across all PACK-027 tables.
--
-- Views (3):
--   1. pack027_enterprise_net_zero.vw_enterprise_dashboard
--   2. pack027_enterprise_net_zero.vw_supply_chain_heatmap
--   3. pack027_enterprise_net_zero.vw_regulatory_calendar
--
-- Indexes: 200+ composite and conditional indexes for enterprise performance.
--
-- Previous: V179__PACK027_data_quality_tracking.sql
-- =============================================================================

-- =============================================================================
-- View 1: vw_enterprise_dashboard
-- =============================================================================
-- Consolidated enterprise emissions summary with target progress, risk
-- overview, supply chain engagement, data quality, and regulatory status
-- for the executive dashboard.

CREATE OR REPLACE VIEW pack027_enterprise_net_zero.vw_enterprise_dashboard AS
SELECT
    ep.company_id,
    ep.tenant_id,
    ep.name                         AS company_name,
    ep.sector,
    ep.employees,
    ep.revenue_usd,
    ep.boundary_approach,
    ep.entity_count,
    ep.hq_country,
    ep.primary_erp,
    ep.data_quality_target,
    ep.assurance_level              AS target_assurance_level,
    ep.profile_status,
    -- Latest baseline emissions
    bl.reporting_year               AS baseline_year,
    bl.scope1_total_tco2e,
    bl.scope2_location_tco2e,
    bl.scope2_market_tco2e,
    bl.scope3_total_tco2e,
    (bl.scope1_total_tco2e + bl.scope2_location_tco2e + bl.scope3_total_tco2e) AS total_location_tco2e,
    (bl.scope1_total_tco2e + bl.scope2_market_tco2e + bl.scope3_total_tco2e) AS total_market_tco2e,
    bl.data_quality_score           AS baseline_dq_score,
    bl.verification_status          AS baseline_verification,
    bl.intensity_per_employee,
    bl.intensity_per_revenue,
    -- SBTi target progress
    st.target_type                  AS sbti_target_type,
    st.pathway_type                 AS sbti_pathway,
    st.ambition_level               AS sbti_ambition,
    st.near_term_target             AS sbti_near_term_pct,
    st.near_term_year               AS sbti_near_term_year,
    st.long_term_target             AS sbti_long_term_pct,
    st.validation_status            AS sbti_validation_status,
    st.near_term_criteria_pass,
    st.near_term_criteria_total,
    st.submission_readiness_score   AS sbti_readiness,
    -- Scenario modeling (latest active)
    sm.scenario_name                AS latest_scenario,
    sm.pathway                      AS scenario_pathway,
    sm.prob_near_term_target        AS scenario_prob_near_term,
    sm.prob_net_zero                AS scenario_prob_netzero,
    -- Carbon pricing
    cp_latest.price_usd_per_tco2e  AS current_carbon_price,
    cl.carbon_cost_usd             AS ytd_carbon_cost,
    cl.ebitda_impact_pct            AS carbon_ebitda_impact,
    -- Supply chain summary (Tier 1)
    sct.supplier_count              AS tier1_suppliers,
    sct.supplier_count_engaged      AS tier1_engaged,
    sct.supplier_count_sbti         AS tier1_sbti,
    sct.emissions_tco2e             AS tier1_emissions,
    sct.engagement_actual_pct       AS tier1_engagement_pct,
    -- Risk overview
    cr_agg.total_risks,
    cr_agg.critical_risks,
    cr_agg.high_risks,
    cr_agg.total_financial_exposure,
    -- Regulatory status
    rf_agg.total_filings,
    rf_agg.pending_filings,
    rf_agg.submitted_filings,
    rf_agg.next_deadline,
    -- Assurance status
    ae_latest.assurance_level       AS latest_assurance_level,
    ae_latest.opinion               AS latest_opinion,
    ae_latest.status                AS assurance_status,
    -- Data quality (overall)
    dq_overall.weighted_dq_score    AS overall_dq_score,
    dq_overall.completeness_pct     AS overall_completeness,
    dq_overall.accuracy_pct         AS overall_accuracy,
    -- Board reporting
    br_latest.reporting_quarter     AS latest_board_quarter,
    br_latest.near_term_on_track    AS board_near_term_status,
    br_latest.compliance_issues_count AS board_compliance_issues
FROM pack027_enterprise_net_zero.gl_enterprise_profiles ep
-- Latest baseline
LEFT JOIN LATERAL (
    SELECT * FROM pack027_enterprise_net_zero.gl_enterprise_baselines
    WHERE company_id = ep.company_id
    ORDER BY reporting_year DESC LIMIT 1
) bl ON TRUE
-- Latest SBTi target
LEFT JOIN LATERAL (
    SELECT * FROM pack027_enterprise_net_zero.gl_sbti_targets
    WHERE company_id = ep.company_id
    ORDER BY created_at DESC LIMIT 1
) st ON TRUE
-- Latest active scenario
LEFT JOIN LATERAL (
    SELECT * FROM pack027_enterprise_net_zero.gl_scenario_models
    WHERE company_id = ep.company_id AND is_active = TRUE AND execution_status = 'COMPLETED'
    ORDER BY created_at DESC LIMIT 1
) sm ON TRUE
-- Latest active carbon price
LEFT JOIN LATERAL (
    SELECT * FROM pack027_enterprise_net_zero.gl_carbon_prices
    WHERE company_id = ep.company_id AND is_active = TRUE
    ORDER BY effective_date DESC LIMIT 1
) cp_latest ON TRUE
-- Latest carbon liability
LEFT JOIN LATERAL (
    SELECT * FROM pack027_enterprise_net_zero.gl_carbon_liabilities
    WHERE company_id = ep.company_id
    ORDER BY fiscal_year DESC, fiscal_quarter DESC NULLS LAST LIMIT 1
) cl ON TRUE
-- Tier 1 supply chain
LEFT JOIN LATERAL (
    SELECT * FROM pack027_enterprise_net_zero.gl_supply_chain_tiers
    WHERE company_id = ep.company_id AND tier_level = 1
    ORDER BY reporting_year DESC LIMIT 1
) sct ON TRUE
-- Risk aggregate
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                        AS total_risks,
        COUNT(*) FILTER (WHERE severity = 'CRITICAL')                   AS critical_risks,
        COUNT(*) FILTER (WHERE severity = 'HIGH')                       AS high_risks,
        SUM(financial_impact_usd)                                       AS total_financial_exposure
    FROM pack027_enterprise_net_zero.gl_climate_risks
    WHERE company_id = ep.company_id
) cr_agg ON TRUE
-- Regulatory filings aggregate
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                        AS total_filings,
        COUNT(*) FILTER (WHERE status IN ('NOT_STARTED', 'DATA_COLLECTION', 'CALCULATION', 'REVIEW')) AS pending_filings,
        COUNT(*) FILTER (WHERE status IN ('SUBMITTED', 'ACCEPTED', 'PUBLISHED'))  AS submitted_filings,
        MIN(deadline) FILTER (WHERE deadline >= CURRENT_DATE AND status NOT IN ('SUBMITTED', 'ACCEPTED', 'PUBLISHED')) AS next_deadline
    FROM pack027_enterprise_net_zero.gl_regulatory_filings
    WHERE company_id = ep.company_id
) rf_agg ON TRUE
-- Latest assurance engagement
LEFT JOIN LATERAL (
    SELECT * FROM pack027_enterprise_net_zero.gl_assurance_engagements
    WHERE company_id = ep.company_id
    ORDER BY reporting_year DESC LIMIT 1
) ae_latest ON TRUE
-- Overall data quality
LEFT JOIN LATERAL (
    SELECT
        weighted_dq_score, completeness_pct, accuracy_pct
    FROM pack027_enterprise_net_zero.gl_data_quality_scores
    WHERE company_id = ep.company_id AND category = 'OVERALL'
    ORDER BY reporting_year DESC LIMIT 1
) dq_overall ON TRUE
-- Latest board report
LEFT JOIN LATERAL (
    SELECT * FROM pack027_enterprise_net_zero.gl_board_reports
    WHERE company_id = ep.company_id
    ORDER BY reporting_year DESC, reporting_quarter DESC LIMIT 1
) br_latest ON TRUE
WHERE ep.profile_status = 'active';

-- =============================================================================
-- View 2: vw_supply_chain_heatmap
-- =============================================================================
-- Tier-level supply chain emissions breakdown with engagement statistics
-- and data quality indicators for supply chain program management.

CREATE OR REPLACE VIEW pack027_enterprise_net_zero.vw_supply_chain_heatmap AS
SELECT
    ep.company_id,
    ep.tenant_id,
    ep.name                         AS company_name,
    -- Tier summary
    sct.tier_level,
    sct.tier_name,
    sct.reporting_year,
    sct.supplier_count,
    sct.supplier_count_engaged,
    sct.supplier_count_sbti,
    sct.supplier_count_cdp,
    sct.total_spend_usd,
    sct.spend_coverage_pct,
    sct.emissions_tco2e,
    sct.emissions_pct_of_scope3,
    sct.emissions_method,
    sct.avg_data_quality_score,
    -- Engagement metrics
    sct.engagement_target_pct,
    sct.engagement_actual_pct,
    sct.sbti_adoption_target_pct,
    sct.sbti_adoption_actual_pct,
    -- Data quality breakdown
    sct.supplier_specific_pct,
    sct.average_data_pct,
    sct.spend_based_pct,
    -- Year-over-year
    sct.yoy_emissions_change_pct,
    sct.yoy_engagement_change_pct,
    -- Hotspots
    sct.top_categories,
    sct.top_geographies,
    sct.top_commodities,
    -- Engagement detail aggregates
    se_agg.total_engaged_suppliers,
    se_agg.avg_engagement_score,
    se_agg.suppliers_disclosing,
    se_agg.high_risk_suppliers,
    se_agg.total_disclosed_tco2e,
    se_agg.total_estimated_tco2e,
    -- Calculated metrics
    CASE
        WHEN sct.supplier_count > 0 THEN
            ROUND((sct.supplier_count_engaged::DECIMAL / sct.supplier_count * 100), 1)
        ELSE 0
    END AS calculated_engagement_pct,
    CASE
        WHEN sct.supplier_count > 0 THEN
            ROUND((sct.supplier_count_sbti::DECIMAL / sct.supplier_count * 100), 1)
        ELSE 0
    END AS calculated_sbti_pct
FROM pack027_enterprise_net_zero.gl_enterprise_profiles ep
INNER JOIN pack027_enterprise_net_zero.gl_supply_chain_tiers sct
    ON sct.company_id = ep.company_id
-- Supplier engagement aggregates per tier
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                            AS total_engaged_suppliers,
        AVG(engagement_score)                               AS avg_engagement_score,
        COUNT(*) FILTER (WHERE emissions_disclosed = TRUE)  AS suppliers_disclosing,
        COUNT(*) FILTER (WHERE high_risk_flag = TRUE)       AS high_risk_suppliers,
        SUM(disclosed_total_tco2e)                          AS total_disclosed_tco2e,
        SUM(estimated_tco2e)                                AS total_estimated_tco2e
    FROM pack027_enterprise_net_zero.gl_supplier_engagement
    WHERE company_id = ep.company_id AND tier_level = sct.tier_level
) se_agg ON TRUE
WHERE ep.profile_status = 'active'
ORDER BY ep.company_id, sct.tier_level, sct.reporting_year DESC;

-- =============================================================================
-- View 3: vw_regulatory_calendar
-- =============================================================================
-- Upcoming regulatory filing deadlines by framework with status, progress,
-- and assurance requirements for compliance program management.

CREATE OR REPLACE VIEW pack027_enterprise_net_zero.vw_regulatory_calendar AS
SELECT
    ep.company_id,
    ep.tenant_id,
    ep.name                         AS company_name,
    -- Filing details
    rf.filing_id,
    rf.framework,
    rf.framework_version,
    rf.filing_year,
    rf.filing_type,
    rf.status                       AS filing_status,
    rf.deadline,
    rf.submission_date,
    rf.publication_date,
    -- Assurance
    rf.assurance_required,
    rf.assurance_level,
    rf.assurance_provider,
    rf.assurance_status,
    rf.assurance_opinion,
    -- Coverage
    rf.scope1_included,
    rf.scope2_included,
    rf.scope3_included,
    rf.transition_plan_included,
    -- Quality
    rf.overall_quality_score,
    rf.completeness_score,
    -- Review
    rf.revision_count,
    rf.deficiency_noted,
    rf.remediation_required,
    rf.remediation_deadline,
    -- Calculated fields
    CASE
        WHEN rf.deadline IS NULL THEN 'NO_DEADLINE'
        WHEN rf.status IN ('SUBMITTED', 'ACCEPTED', 'PUBLISHED') THEN 'COMPLETED'
        WHEN rf.deadline < CURRENT_DATE THEN 'OVERDUE'
        WHEN rf.deadline < CURRENT_DATE + INTERVAL '30 days' THEN 'DUE_SOON'
        WHEN rf.deadline < CURRENT_DATE + INTERVAL '90 days' THEN 'UPCOMING'
        ELSE 'FUTURE'
    END AS urgency_status,
    CASE
        WHEN rf.deadline IS NOT NULL THEN rf.deadline - CURRENT_DATE
        ELSE NULL
    END AS days_remaining,
    -- Compliance gaps for this filing
    cg_agg.total_gaps,
    cg_agg.critical_gaps,
    cg_agg.open_gaps,
    -- Framework description
    CASE rf.framework
        WHEN 'SEC_CLIMATE' THEN 'SEC Climate Disclosure Rule'
        WHEN 'CSRD_ESRS_E1' THEN 'EU CSRD / ESRS E1 Climate'
        WHEN 'CDP_CLIMATE' THEN 'CDP Climate Change Questionnaire'
        WHEN 'TCFD' THEN 'TCFD Recommendations'
        WHEN 'ISSB_S2' THEN 'IFRS S2 Climate Disclosures'
        WHEN 'SB253' THEN 'California SB 253 (Climate Corporate Data Accountability)'
        WHEN 'SB261' THEN 'California SB 261 (Climate-Related Financial Risk)'
        WHEN 'ISO14064' THEN 'ISO 14064-1 GHG Quantification'
        WHEN 'GHG_PROTOCOL' THEN 'GHG Protocol Corporate Standard'
        WHEN 'SBTI' THEN 'SBTi Target Submission'
        WHEN 'EU_TAXONOMY' THEN 'EU Taxonomy Climate Delegated Act'
        WHEN 'SFDR' THEN 'EU SFDR Sustainability Disclosures'
        ELSE rf.framework
    END AS framework_name
FROM pack027_enterprise_net_zero.gl_enterprise_profiles ep
INNER JOIN pack027_enterprise_net_zero.gl_regulatory_filings rf
    ON rf.company_id = ep.company_id
-- Compliance gaps aggregate per filing
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                    AS total_gaps,
        COUNT(*) FILTER (WHERE gap_severity = 'CRITICAL')           AS critical_gaps,
        COUNT(*) FILTER (WHERE status IN ('OPEN', 'IN_PROGRESS'))   AS open_gaps
    FROM pack027_enterprise_net_zero.gl_compliance_gaps
    WHERE filing_id = rf.filing_id
) cg_agg ON TRUE
WHERE ep.profile_status = 'active'
ORDER BY
    CASE
        WHEN rf.status IN ('SUBMITTED', 'ACCEPTED', 'PUBLISHED') THEN 2
        ELSE 1
    END,
    CASE WHEN rf.deadline IS NULL THEN 1 ELSE 0 END,
    rf.deadline ASC NULLS LAST;

-- =============================================================================
-- Performance Indexes (Composite & Conditional)
-- =============================================================================
-- Multi-column composite indexes for common enterprise query patterns.

-- Enterprise profiles: sector + country + employees for filtering
CREATE INDEX idx_p027_ep_sector_country     ON pack027_enterprise_net_zero.gl_enterprise_profiles(sector, hq_country);
CREATE INDEX idx_p027_ep_sector_employees   ON pack027_enterprise_net_zero.gl_enterprise_profiles(sector, employees DESC);
CREATE INDEX idx_p027_ep_country_revenue    ON pack027_enterprise_net_zero.gl_enterprise_profiles(hq_country, revenue_usd DESC);
CREATE INDEX idx_p027_ep_tenant_active      ON pack027_enterprise_net_zero.gl_enterprise_profiles(tenant_id, profile_status) WHERE profile_status = 'active';
CREATE INDEX idx_p027_ep_boundary_status    ON pack027_enterprise_net_zero.gl_enterprise_profiles(boundary_approach, profile_status);

-- Entity hierarchy: active hierarchy traversal
CREATE INDEX idx_p027_eh_parent_active      ON pack027_enterprise_net_zero.gl_entity_hierarchy(parent_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p027_eh_child_active       ON pack027_enterprise_net_zero.gl_entity_hierarchy(child_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p027_eh_tenant_active      ON pack027_enterprise_net_zero.gl_entity_hierarchy(tenant_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p027_eh_type_control       ON pack027_enterprise_net_zero.gl_entity_hierarchy(relationship_type, control_type);

-- Intercompany: elimination workflow
CREATE INDEX idx_p027_ict_pending_elim      ON pack027_enterprise_net_zero.gl_intercompany_transactions(elimination_status, fiscal_year) WHERE elimination_status = 'PENDING';
CREATE INDEX idx_p027_ict_unmatched         ON pack027_enterprise_net_zero.gl_intercompany_transactions(reconciliation_status, fiscal_year) WHERE reconciliation_status = 'UNMATCHED';
CREATE INDEX idx_p027_ict_from_year         ON pack027_enterprise_net_zero.gl_intercompany_transactions(from_entity, fiscal_year);
CREATE INDEX idx_p027_ict_to_year           ON pack027_enterprise_net_zero.gl_intercompany_transactions(to_entity, fiscal_year);

-- Baselines: time series and consolidation queries
CREATE INDEX idx_p027_bl_tenant_year        ON pack027_enterprise_net_zero.gl_enterprise_baselines(tenant_id, reporting_year);
CREATE INDEX idx_p027_bl_company_base       ON pack027_enterprise_net_zero.gl_enterprise_baselines(company_id, is_base_year) WHERE is_base_year = TRUE;
CREATE INDEX idx_p027_bl_year_dq            ON pack027_enterprise_net_zero.gl_enterprise_baselines(reporting_year, data_quality_score DESC);
CREATE INDEX idx_p027_bl_verified_year      ON pack027_enterprise_net_zero.gl_enterprise_baselines(verification_status, reporting_year) WHERE verification_status != 'unverified';

-- SBTi targets: pipeline management
CREATE INDEX idx_p027_st_company_type       ON pack027_enterprise_net_zero.gl_sbti_targets(company_id, target_type);
CREATE INDEX idx_p027_st_pathway_ambition   ON pack027_enterprise_net_zero.gl_sbti_targets(pathway_type, ambition_level);
CREATE INDEX idx_p027_st_validation_type    ON pack027_enterprise_net_zero.gl_sbti_targets(validation_status, target_type);
CREATE INDEX idx_p027_st_submitted          ON pack027_enterprise_net_zero.gl_sbti_targets(validation_status) WHERE validation_status = 'SUBMITTED';
CREATE INDEX idx_p027_st_reval_due          ON pack027_enterprise_net_zero.gl_sbti_targets(revalidation_due) WHERE revalidation_due IS NOT NULL;

-- Scenarios: active completed scenarios
CREATE INDEX idx_p027_sm_company_active     ON pack027_enterprise_net_zero.gl_scenario_models(company_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p027_sm_type_pathway       ON pack027_enterprise_net_zero.gl_scenario_models(scenario_type, pathway);
CREATE INDEX idx_p027_sm_company_completed  ON pack027_enterprise_net_zero.gl_scenario_models(company_id, execution_status) WHERE execution_status = 'COMPLETED';
CREATE INDEX idx_p027_sm_comparison_active  ON pack027_enterprise_net_zero.gl_scenario_models(comparison_group_id) WHERE is_active = TRUE;

-- Carbon pricing: active pricing lookup
CREATE INDEX idx_p027_cp_company_active     ON pack027_enterprise_net_zero.gl_carbon_prices(company_id, is_active, effective_date) WHERE is_active = TRUE;
CREATE INDEX idx_p027_cp_type_active        ON pack027_enterprise_net_zero.gl_carbon_prices(price_type, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p027_cp_regulatory_active  ON pack027_enterprise_net_zero.gl_carbon_prices(regulatory_scheme, is_active) WHERE regulatory_scheme IS NOT NULL;

-- Carbon liabilities: fiscal period queries
CREATE INDEX idx_p027_cl_company_year_q     ON pack027_enterprise_net_zero.gl_carbon_liabilities(company_id, fiscal_year, fiscal_quarter);
CREATE INDEX idx_p027_cl_status_year        ON pack027_enterprise_net_zero.gl_carbon_liabilities(status, fiscal_year);
CREATE INDEX idx_p027_cl_ebitda_desc        ON pack027_enterprise_net_zero.gl_carbon_liabilities(ebitda_impact_pct DESC) WHERE ebitda_impact_pct IS NOT NULL;

-- Scope 4: project performance
CREATE INDEX idx_p027_s4_company_year       ON pack027_enterprise_net_zero.gl_scope4_projects(company_id, reporting_year);
CREATE INDEX idx_p027_s4_type_verified      ON pack027_enterprise_net_zero.gl_scope4_projects(project_type, verification_status);
CREATE INDEX idx_p027_s4_avoided_desc       ON pack027_enterprise_net_zero.gl_scope4_projects(avoided_tco2e DESC);

-- Supply chain: engagement management
CREATE INDEX idx_p027_sct_company_year_tier ON pack027_enterprise_net_zero.gl_supply_chain_tiers(company_id, reporting_year, tier_level);
CREATE INDEX idx_p027_sct_emissions_desc    ON pack027_enterprise_net_zero.gl_supply_chain_tiers(company_id, emissions_tco2e DESC);
CREATE INDEX idx_p027_se_company_tier       ON pack027_enterprise_net_zero.gl_supplier_engagement(company_id, tier_level);
CREATE INDEX idx_p027_se_company_engage     ON pack027_enterprise_net_zero.gl_supplier_engagement(company_id, engagement_tier);
CREATE INDEX idx_p027_se_overdue            ON pack027_enterprise_net_zero.gl_supplier_engagement(response_status, next_review_date) WHERE response_status = 'OVERDUE';
CREATE INDEX idx_p027_se_risk_country       ON pack027_enterprise_net_zero.gl_supplier_engagement(supplier_country, high_risk_flag) WHERE high_risk_flag = TRUE;
CREATE INDEX idx_p027_se_spend_emissions    ON pack027_enterprise_net_zero.gl_supplier_engagement(annual_spend_usd DESC, estimated_tco2e DESC);

-- Financial integration: P&L queries
CREATE INDEX idx_p027_cpl_company_year_bu   ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(company_id, fiscal_year, business_unit);
CREATE INDEX idx_p027_cpl_cbam_year         ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(fiscal_year, cbam_applicable) WHERE cbam_applicable = TRUE;
CREATE INDEX idx_p027_cpl_cost_desc         ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(carbon_cost_usd DESC);

-- Carbon assets: portfolio management
CREATE INDEX idx_p027_ca_company_active     ON pack027_enterprise_net_zero.gl_carbon_assets(company_id, retirement_status) WHERE retirement_status = 'ACTIVE';
CREATE INDEX idx_p027_ca_type_vintage       ON pack027_enterprise_net_zero.gl_carbon_assets(asset_type, vintage);
CREATE INDEX idx_p027_ca_expiring           ON pack027_enterprise_net_zero.gl_carbon_assets(contract_end_date) WHERE retirement_status = 'ACTIVE' AND contract_end_date IS NOT NULL;

-- Climate risks: risk management
CREATE INDEX idx_p027_cr_company_type       ON pack027_enterprise_net_zero.gl_climate_risks(company_id, risk_type);
CREATE INDEX idx_p027_cr_severity_status    ON pack027_enterprise_net_zero.gl_climate_risks(severity, mitigation_status);
CREATE INDEX idx_p027_cr_high_critical      ON pack027_enterprise_net_zero.gl_climate_risks(company_id, severity) WHERE severity IN ('HIGH', 'CRITICAL');
CREATE INDEX idx_p027_cr_unmitigated        ON pack027_enterprise_net_zero.gl_climate_risks(mitigation_status) WHERE mitigation_status IN ('IDENTIFIED', 'PLANNED');
CREATE INDEX idx_p027_cr_impact_desc        ON pack027_enterprise_net_zero.gl_climate_risks(financial_impact_usd DESC) WHERE financial_impact_usd IS NOT NULL;

-- Asset risk: geographic and score queries
CREATE INDEX idx_p027_are_company_type      ON pack027_enterprise_net_zero.gl_asset_risk_exposure(company_id, asset_type);
CREATE INDEX idx_p027_are_country_physical  ON pack027_enterprise_net_zero.gl_asset_risk_exposure(asset_location_country, physical_risk_score DESC);
CREATE INDEX idx_p027_are_high_physical     ON pack027_enterprise_net_zero.gl_asset_risk_exposure(physical_risk_score DESC) WHERE physical_risk_score >= 70;
CREATE INDEX idx_p027_are_high_transition   ON pack027_enterprise_net_zero.gl_asset_risk_exposure(transition_risk_score DESC) WHERE transition_risk_score >= 70;
CREATE INDEX idx_p027_are_stranded_high     ON pack027_enterprise_net_zero.gl_asset_risk_exposure(stranded_asset_risk_score DESC) WHERE stranded_asset_risk_score >= 50;

-- Regulatory filings: deadline management
CREATE INDEX idx_p027_rf_company_framework  ON pack027_enterprise_net_zero.gl_regulatory_filings(company_id, framework, filing_year);
CREATE INDEX idx_p027_rf_deadline_status    ON pack027_enterprise_net_zero.gl_regulatory_filings(deadline, status) WHERE status NOT IN ('SUBMITTED', 'ACCEPTED', 'PUBLISHED');
CREATE INDEX idx_p027_rf_overdue            ON pack027_enterprise_net_zero.gl_regulatory_filings(deadline) WHERE deadline < CURRENT_DATE AND status NOT IN ('SUBMITTED', 'ACCEPTED', 'PUBLISHED', 'WITHDRAWN');
CREATE INDEX idx_p027_rf_assurance_pending  ON pack027_enterprise_net_zero.gl_regulatory_filings(assurance_status) WHERE assurance_required = TRUE AND assurance_status != 'COMPLETED';

-- Compliance gaps: remediation tracking
CREATE INDEX idx_p027_cg_company_open       ON pack027_enterprise_net_zero.gl_compliance_gaps(company_id, status) WHERE status IN ('OPEN', 'IN_PROGRESS');
CREATE INDEX idx_p027_cg_framework_severity ON pack027_enterprise_net_zero.gl_compliance_gaps(framework, gap_severity);
CREATE INDEX idx_p027_cg_overdue            ON pack027_enterprise_net_zero.gl_compliance_gaps(target_resolution_date) WHERE status IN ('OPEN', 'IN_PROGRESS') AND target_resolution_date < CURRENT_DATE;
CREATE INDEX idx_p027_cg_critical_open      ON pack027_enterprise_net_zero.gl_compliance_gaps(company_id) WHERE gap_severity = 'CRITICAL' AND status = 'OPEN';

-- Assurance: engagement lifecycle
CREATE INDEX idx_p027_ae_company_year       ON pack027_enterprise_net_zero.gl_assurance_engagements(company_id, reporting_year);
CREATE INDEX idx_p027_ae_status_level       ON pack027_enterprise_net_zero.gl_assurance_engagements(status, assurance_level);
CREATE INDEX idx_p027_ae_active_engagement  ON pack027_enterprise_net_zero.gl_assurance_engagements(status) WHERE status NOT IN ('COMPLETED', 'CANCELLED');

-- Workpapers: review pipeline
CREATE INDEX idx_p027_aw_engagement_type    ON pack027_enterprise_net_zero.gl_assurance_workpapers(engagement_id, workpaper_type);
CREATE INDEX idx_p027_aw_pending_review     ON pack027_enterprise_net_zero.gl_assurance_workpapers(review_status) WHERE review_status IN ('DRAFT', 'PREPARED');
CREATE INDEX idx_p027_aw_scope_review       ON pack027_enterprise_net_zero.gl_assurance_workpapers(scope_category, review_status);

-- Board reports: governance queries
CREATE INDEX idx_p027_br_company_year_q     ON pack027_enterprise_net_zero.gl_board_reports(company_id, reporting_year, reporting_quarter);
CREATE INDEX idx_p027_br_status_date        ON pack027_enterprise_net_zero.gl_board_reports(status, report_date DESC);
CREATE INDEX idx_p027_br_at_risk            ON pack027_enterprise_net_zero.gl_board_reports(near_term_on_track) WHERE near_term_on_track IN ('AT_RISK', 'OFF_TRACK');

-- Data quality: improvement tracking
CREATE INDEX idx_p027_dq_company_year_cat   ON pack027_enterprise_net_zero.gl_data_quality_scores(company_id, reporting_year, category);
CREATE INDEX idx_p027_dq_low_quality        ON pack027_enterprise_net_zero.gl_data_quality_scores(ghg_dq_level, category) WHERE ghg_dq_level >= 4;
CREATE INDEX idx_p027_dq_completeness_low   ON pack027_enterprise_net_zero.gl_data_quality_scores(completeness_pct, category) WHERE completeness_pct < 80;
CREATE INDEX idx_p027_dq_accuracy_low       ON pack027_enterprise_net_zero.gl_data_quality_scores(accuracy_pct, category) WHERE accuracy_pct < 90;
CREATE INDEX idx_p027_dq_entity_year        ON pack027_enterprise_net_zero.gl_data_quality_scores(entity_ref, reporting_year) WHERE entity_ref IS NOT NULL;

-- =============================================================================
-- Comments on Views
-- =============================================================================
COMMENT ON VIEW pack027_enterprise_net_zero.vw_enterprise_dashboard IS
    'Consolidated enterprise dashboard view with latest baseline, SBTi targets, scenario probabilities, carbon pricing, supply chain engagement, risk overview, regulatory status, assurance, and data quality.';

COMMENT ON VIEW pack027_enterprise_net_zero.vw_supply_chain_heatmap IS
    'Supply chain tier-level emissions heatmap with engagement statistics, data quality breakdown, hotspot analysis, and year-over-year trends.';

COMMENT ON VIEW pack027_enterprise_net_zero.vw_regulatory_calendar IS
    'Regulatory filing calendar with upcoming deadlines by framework, urgency status (OVERDUE/DUE_SOON/UPCOMING/FUTURE), compliance gap counts, and assurance requirements.';
