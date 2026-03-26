-- =============================================================================
-- PACK-048 GHG Assurance Prep Pack
-- Migration: V405 - Views, Indexes & Seed Data
-- =============================================================================
-- Pack:         PACK-048 (GHG Assurance Prep Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates analytical materialized views, additional composite indexes
-- for common query patterns, and seed data. Views provide pre-joined
-- perspectives for dashboards. Seed data populates standard controls,
-- jurisdiction requirements, checklist templates, cost model parameters,
-- and materiality defaults.
--
-- Materialized Views (4):
--   1. ghg_assurance.gl_ap_v_readiness_dashboard
--   2. ghg_assurance.gl_ap_v_evidence_summary
--   3. ghg_assurance.gl_ap_v_control_summary
--   4. ghg_assurance.gl_ap_v_query_status
--
-- Also includes: additional indexes, seed data, grants, comments.
-- Previous: V404__pack048_cost_timeline.sql
-- =============================================================================

SET search_path TO ghg_assurance, public;

-- =============================================================================
-- Materialized View 1: ghg_assurance.gl_ap_v_readiness_dashboard
-- =============================================================================
-- Latest readiness assessment scores per configuration with gap and
-- remediation progress summary.

CREATE MATERIALIZED VIEW ghg_assurance.gl_ap_v_readiness_dashboard AS
SELECT
    ra.id                       AS assessment_id,
    ra.tenant_id,
    ra.config_id,
    cfg.config_name,
    cfg.assurance_standard,
    cfg.assurance_level,
    cfg.jurisdiction,
    ra.standard,
    ra.assessment_date,
    ra.overall_score,
    ra.readiness_level,
    ra.category_scores,
    ra.gap_count,
    ra.time_to_ready_days,
    COUNT(CASE WHEN g.status = 'OPEN' THEN 1 END)          AS open_gaps,
    COUNT(CASE WHEN g.status = 'IN_PROGRESS' THEN 1 END)   AS in_progress_gaps,
    COUNT(CASE WHEN g.status = 'REMEDIATED' THEN 1 END)    AS remediated_gaps,
    COUNT(CASE WHEN g.severity = 'CRITICAL' THEN 1 END)    AS critical_gaps,
    ra.provenance_hash,
    NOW()                       AS materialized_at
FROM ghg_assurance.gl_ap_readiness_assessments ra
JOIN ghg_assurance.gl_ap_configurations cfg ON ra.config_id = cfg.id
LEFT JOIN ghg_assurance.gl_ap_gaps g ON ra.id = g.assessment_id
WHERE ra.assessment_date = (
    SELECT MAX(ra2.assessment_date)
    FROM ghg_assurance.gl_ap_readiness_assessments ra2
    WHERE ra2.config_id = ra.config_id
      AND ra2.standard = ra.standard
)
GROUP BY ra.id, ra.tenant_id, ra.config_id, cfg.config_name,
         cfg.assurance_standard, cfg.assurance_level, cfg.jurisdiction,
         ra.standard, ra.assessment_date, ra.overall_score,
         ra.readiness_level, ra.category_scores, ra.gap_count,
         ra.time_to_ready_days, ra.provenance_hash;

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p048_vrd_pk
    ON ghg_assurance.gl_ap_v_readiness_dashboard(assessment_id);
CREATE INDEX idx_p048_vrd_tenant
    ON ghg_assurance.gl_ap_v_readiness_dashboard(tenant_id);
CREATE INDEX idx_p048_vrd_config
    ON ghg_assurance.gl_ap_v_readiness_dashboard(config_id);
CREATE INDEX idx_p048_vrd_level
    ON ghg_assurance.gl_ap_v_readiness_dashboard(readiness_level);
CREATE INDEX idx_p048_vrd_score
    ON ghg_assurance.gl_ap_v_readiness_dashboard(overall_score);

-- =============================================================================
-- Materialized View 2: ghg_assurance.gl_ap_v_evidence_summary
-- =============================================================================
-- Evidence completeness by scope and category across all packages for
-- a given configuration.

CREATE MATERIALIZED VIEW ghg_assurance.gl_ap_v_evidence_summary AS
SELECT
    ep.config_id,
    ep.tenant_id,
    ei.scope,
    ei.category,
    COUNT(*)                                                    AS item_count,
    COUNT(CASE WHEN ei.quality_grade = 'EXCELLENT' THEN 1 END) AS excellent_count,
    COUNT(CASE WHEN ei.quality_grade = 'GOOD' THEN 1 END)      AS good_count,
    COUNT(CASE WHEN ei.quality_grade = 'ADEQUATE' THEN 1 END)  AS adequate_count,
    COUNT(CASE WHEN ei.quality_grade = 'MARGINAL' THEN 1 END)  AS marginal_count,
    COUNT(CASE WHEN ei.quality_grade = 'INSUFFICIENT' THEN 1 END) AS insufficient_count,
    AVG(ei.data_quality_score)                                  AS avg_quality_score,
    MIN(ei.data_quality_score)                                  AS min_quality_score,
    MAX(ei.data_quality_score)                                  AS max_quality_score,
    NOW()                       AS materialized_at
FROM ghg_assurance.gl_ap_evidence_items ei
JOIN ghg_assurance.gl_ap_evidence_packages ep ON ei.package_id = ep.id
WHERE ep.package_version = 'FINAL'
   OR ep.package_version = (
       SELECT MAX(ep2.package_version)
       FROM ghg_assurance.gl_ap_evidence_packages ep2
       WHERE ep2.config_id = ep.config_id
   )
GROUP BY ep.config_id, ep.tenant_id, ei.scope, ei.category;

-- Indexes on materialized view
CREATE INDEX idx_p048_ves_config
    ON ghg_assurance.gl_ap_v_evidence_summary(config_id);
CREATE INDEX idx_p048_ves_tenant
    ON ghg_assurance.gl_ap_v_evidence_summary(tenant_id);
CREATE INDEX idx_p048_ves_scope
    ON ghg_assurance.gl_ap_v_evidence_summary(scope);
CREATE INDEX idx_p048_ves_category
    ON ghg_assurance.gl_ap_v_evidence_summary(category);
CREATE INDEX idx_p048_ves_scope_cat
    ON ghg_assurance.gl_ap_v_evidence_summary(scope, category);

-- =============================================================================
-- Materialized View 3: ghg_assurance.gl_ap_v_control_summary
-- =============================================================================
-- Control effectiveness summary with test results aggregated by category.

CREATE MATERIALIZED VIEW ghg_assurance.gl_ap_v_control_summary AS
SELECT
    c.config_id,
    c.tenant_id,
    c.category,
    COUNT(*)                                                        AS total_controls,
    COUNT(CASE WHEN c.is_key_control THEN 1 END)                   AS key_controls,
    COUNT(CASE WHEN ct.design_effective = 'EFFECTIVE' THEN 1 END)   AS design_effective,
    COUNT(CASE WHEN ct.operating_effective = 'EFFECTIVE' THEN 1 END) AS operating_effective,
    COUNT(CASE WHEN ct.design_effective = 'INEFFECTIVE' THEN 1 END)  AS design_ineffective,
    COUNT(CASE WHEN ct.operating_effective = 'INEFFECTIVE' THEN 1 END) AS operating_ineffective,
    COUNT(CASE WHEN ct.design_effective = 'NOT_TESTED' THEN 1 END)   AS not_tested,
    SUM(ct.exceptions_found)                                        AS total_exceptions,
    COUNT(DISTINCT d.id)                                            AS deficiency_count,
    COUNT(CASE WHEN d.deficiency_type = 'MATERIAL_WEAKNESS' THEN 1 END) AS material_weaknesses,
    AVG(CASE
        WHEN c.maturity_level = 'LEVEL_1' THEN 1
        WHEN c.maturity_level = 'LEVEL_2' THEN 2
        WHEN c.maturity_level = 'LEVEL_3' THEN 3
        WHEN c.maturity_level = 'LEVEL_4' THEN 4
        WHEN c.maturity_level = 'LEVEL_5' THEN 5
    END)                                                            AS avg_maturity,
    NOW()                       AS materialized_at
FROM ghg_assurance.gl_ap_controls c
LEFT JOIN ghg_assurance.gl_ap_control_tests ct ON c.id = ct.control_id
    AND ct.created_at = (
        SELECT MAX(ct2.created_at)
        FROM ghg_assurance.gl_ap_control_tests ct2
        WHERE ct2.control_id = c.id
    )
LEFT JOIN ghg_assurance.gl_ap_deficiencies d ON ct.id = d.control_test_id
WHERE c.is_active = true
GROUP BY c.config_id, c.tenant_id, c.category;

-- Indexes on materialized view
CREATE INDEX idx_p048_vcs_config
    ON ghg_assurance.gl_ap_v_control_summary(config_id);
CREATE INDEX idx_p048_vcs_tenant
    ON ghg_assurance.gl_ap_v_control_summary(tenant_id);
CREATE INDEX idx_p048_vcs_category
    ON ghg_assurance.gl_ap_v_control_summary(category);

-- =============================================================================
-- Materialized View 4: ghg_assurance.gl_ap_v_query_status
-- =============================================================================
-- Outstanding queries and findings by engagement with priority and overdue
-- tracking.

CREATE MATERIALIZED VIEW ghg_assurance.gl_ap_v_query_status AS
SELECT
    eng.id                      AS engagement_id,
    eng.tenant_id,
    eng.config_id,
    eng.engagement_name,
    eng.engagement_phase,
    eng.status                  AS engagement_status,
    -- Query metrics
    COUNT(DISTINCT q.id)                                            AS total_queries,
    COUNT(DISTINCT CASE WHEN q.status IN ('OPEN','IN_PROGRESS') THEN q.id END) AS open_queries,
    COUNT(DISTINCT CASE WHEN q.is_overdue THEN q.id END)            AS overdue_queries,
    COUNT(DISTINCT CASE WHEN q.priority = 'CRITICAL' THEN q.id END) AS critical_queries,
    -- Finding metrics
    COUNT(DISTINCT f.id)                                            AS total_findings,
    COUNT(DISTINCT CASE WHEN f.status IN ('OPEN','IN_PROGRESS') THEN f.id END) AS open_findings,
    COUNT(DISTINCT CASE WHEN f.severity = 'CRITICAL' THEN f.id END) AS critical_findings,
    COUNT(DISTINCT CASE WHEN f.severity = 'MAJOR' THEN f.id END)    AS major_findings,
    COUNT(DISTINCT CASE WHEN f.is_recurring THEN f.id END)           AS recurring_findings,
    -- Response metrics
    COUNT(DISTINCT r.id)                                            AS total_responses,
    COUNT(DISTINCT CASE WHEN r.is_accepted = true THEN r.id END)    AS accepted_responses,
    COUNT(DISTINCT CASE WHEN r.is_accepted = false THEN r.id END)   AS rejected_responses,
    NOW()                       AS materialized_at
FROM ghg_assurance.gl_ap_engagements eng
LEFT JOIN ghg_assurance.gl_ap_queries q ON eng.id = q.engagement_id
LEFT JOIN ghg_assurance.gl_ap_findings f ON eng.id = f.engagement_id
LEFT JOIN ghg_assurance.gl_ap_responses r ON (r.query_id = q.id OR r.finding_id = f.id)
WHERE eng.status IN ('ACTIVE', 'IN_PROGRESS')
GROUP BY eng.id, eng.tenant_id, eng.config_id, eng.engagement_name,
         eng.engagement_phase, eng.status;

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p048_vqs_pk
    ON ghg_assurance.gl_ap_v_query_status(engagement_id);
CREATE INDEX idx_p048_vqs_tenant
    ON ghg_assurance.gl_ap_v_query_status(tenant_id);
CREATE INDEX idx_p048_vqs_config
    ON ghg_assurance.gl_ap_v_query_status(config_id);
CREATE INDEX idx_p048_vqs_phase
    ON ghg_assurance.gl_ap_v_query_status(engagement_phase);
CREATE INDEX idx_p048_vqs_open_queries
    ON ghg_assurance.gl_ap_v_query_status(open_queries DESC);
CREATE INDEX idx_p048_vqs_open_findings
    ON ghg_assurance.gl_ap_v_query_status(open_findings DESC);

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Cross-table: evidence by config + scope + grade for dashboard
CREATE INDEX idx_p048_ei_cfg_scope_grade
    ON ghg_assurance.gl_ap_evidence_items(package_id, scope, quality_grade);

-- Cross-table: gaps by severity + status for priority queue
CREATE INDEX idx_p048_gap_sev_status_due
    ON ghg_assurance.gl_ap_gaps(severity, status, due_date);

-- Cross-table: controls by config + maturity for maturity matrix
CREATE INDEX idx_p048_ctrl_cfg_maturity
    ON ghg_assurance.gl_ap_controls(config_id, maturity_level);

-- Cross-table: queries by engagement + type + status for filtered views
CREATE INDEX idx_p048_qry_eng_type_status
    ON ghg_assurance.gl_ap_queries(engagement_id, query_type, status);

-- Cross-table: findings by engagement + type + severity for classification
CREATE INDEX idx_p048_fnd_eng_type_sev
    ON ghg_assurance.gl_ap_findings(engagement_id, finding_type, severity);

-- Cross-table: provenance chains by config + scope + completeness
CREATE INDEX idx_p048_pc_cfg_scope_comp
    ON ghg_assurance.gl_ap_provenance_chains(config_id, scope, completeness_pct);

-- Cross-table: requirements by config + jurisdiction + status for compliance
CREATE INDEX idx_p048_req_cfg_jur_status
    ON ghg_assurance.gl_ap_requirements(config_id, jurisdiction_id, compliance_status);

-- Cross-table: milestones by engagement + phase + status for Gantt
CREATE INDEX idx_p048_tm_eng_phase_status
    ON ghg_assurance.gl_ap_timeline_milestones(engagement_id, phase, status);

-- =============================================================================
-- Seed Data: Standard Controls (25 controls: DC-01 to IT-05)
-- =============================================================================
-- These seed controls provide a template for standard GHG reporting
-- internal controls. Organisations customise these to their specific
-- processes. Inserted without tenant_id/config_id as they serve as
-- reference templates.

CREATE TABLE IF NOT EXISTS ghg_assurance.gl_ap_control_templates (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    control_code                TEXT            NOT NULL UNIQUE,
    control_name                VARCHAR(255)    NOT NULL,
    control_description         TEXT            NOT NULL,
    category                    VARCHAR(30)     NOT NULL,
    control_type                VARCHAR(20)     NOT NULL,
    suggested_frequency         VARCHAR(50),
    is_key_control              BOOLEAN         NOT NULL DEFAULT false,
    sort_order                  INTEGER         NOT NULL DEFAULT 0,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

INSERT INTO ghg_assurance.gl_ap_control_templates (
    control_code, control_name, control_description, category, control_type,
    suggested_frequency, is_key_control, sort_order
) VALUES
-- Data Collection Controls (DC-01 to DC-07)
('DC-01', 'Source Data Validation', 'Validate all source data inputs against original documentation (invoices, meter readings, transport logs) before entry into the GHG system.', 'DATA_COLLECTION', 'PREVENTIVE', 'Per transaction', true, 1),
('DC-02', 'Completeness Check', 'Verify all emission sources within the organisational boundary are captured. Cross-reference against facility register and asset inventory.', 'DATA_COLLECTION', 'DETECTIVE', 'Monthly', true, 2),
('DC-03', 'Data Entry Reconciliation', 'Reconcile data entered in the GHG system against source systems (ERP, utility portals, fleet management). Investigate variances >5%.', 'DATA_COLLECTION', 'DETECTIVE', 'Monthly', true, 3),
('DC-04', 'Automated Data Import Validation', 'Validate automated data feeds from ERPs, IoT sensors, and utility providers. Check for missing periods, duplicates, and out-of-range values.', 'DATA_COLLECTION', 'PREVENTIVE', 'Per import', false, 4),
('DC-05', 'Unit Conversion Validation', 'Verify unit conversions (kWh to GJ, litres to kg, miles to km) using published conversion factors with source references.', 'DATA_COLLECTION', 'PREVENTIVE', 'Per calculation', false, 5),
('DC-06', 'Supplier Data Verification', 'Verify supplier-provided emission data and factors against third-party sources. Escalate discrepancies >10%.', 'DATA_COLLECTION', 'DETECTIVE', 'Quarterly', false, 6),
('DC-07', 'Boundary Completeness Review', 'Annual review of organisational and operational boundaries to ensure all material sources are included.', 'DATA_COLLECTION', 'DETECTIVE', 'Annually', true, 7),
-- Calculation Controls (CALC-01 to CALC-06)
('CALC-01', 'Emission Factor Source Verification', 'Verify all emission factors are from approved sources (DEFRA, EPA, IEA) and are the latest published version.', 'CALCULATION', 'PREVENTIVE', 'Annually', true, 8),
('CALC-02', 'Formula Validation', 'Validate all emission calculation formulas against GHG Protocol methodology. Independent recalculation of sample transactions.', 'CALCULATION', 'DETECTIVE', 'Quarterly', true, 9),
('CALC-03', 'GWP Version Consistency', 'Ensure consistent Global Warming Potential values (AR4/AR5/AR6) are applied across all calculations within a reporting period.', 'CALCULATION', 'PREVENTIVE', 'Per calculation', false, 10),
('CALC-04', 'Aggregation Reconciliation', 'Reconcile facility-level totals to group-level consolidation. Verify consolidation approach (equity share, operational control, financial control).', 'CALCULATION', 'DETECTIVE', 'Monthly', true, 11),
('CALC-05', 'Scope 3 Category Mapping', 'Verify Scope 3 spend and activity data are mapped to correct GHG Protocol categories (1-15).', 'CALCULATION', 'PREVENTIVE', 'Per mapping', false, 12),
('CALC-06', 'Uncertainty Quantification', 'Quantify calculation uncertainty using Monte Carlo simulation or error propagation. Document assumptions.', 'CALCULATION', 'DETECTIVE', 'Annually', false, 13),
-- Review Controls (REV-01 to REV-05)
('REV-01', 'Management Review', 'Senior management review and sign-off of quarterly emission results. Review includes trend analysis, year-on-year comparison, and explanations for significant variances.', 'REVIEW', 'DETECTIVE', 'Quarterly', true, 14),
('REV-02', 'Peer Review of Calculations', 'Independent peer review of all emission calculations by a qualified reviewer not involved in original preparation.', 'REVIEW', 'DETECTIVE', 'Per reporting period', true, 15),
('REV-03', 'Variance Analysis', 'Investigate and document explanations for emission variances >10% from prior period or budget. Obtain approval for accepted variances.', 'REVIEW', 'DETECTIVE', 'Monthly', false, 16),
('REV-04', 'Methodology Change Review', 'Formal review and approval process for any changes to calculation methodology, emission factors, or organisational boundary.', 'REVIEW', 'PREVENTIVE', 'Per change', true, 17),
('REV-05', 'Data Quality Assessment', 'Quarterly assessment of data quality across all scopes using the GHG Protocol data quality criteria.', 'REVIEW', 'DETECTIVE', 'Quarterly', false, 18),
-- Reporting Controls (RPT-01 to RPT-04)
('RPT-01', 'Disclosure Completeness', 'Verify all mandatory disclosure requirements are addressed per applicable framework (CSRD, SECR, SEC, CDP).', 'REPORTING', 'DETECTIVE', 'Per report', true, 19),
('RPT-02', 'Report Reconciliation', 'Reconcile reported figures in external disclosures against internal GHG system totals. Zero-tolerance for discrepancies.', 'REPORTING', 'DETECTIVE', 'Per report', true, 20),
('RPT-03', 'Prior Period Restatement Check', 'Verify whether any changes require restatement of prior period figures per GHG Protocol base year recalculation policy.', 'REPORTING', 'DETECTIVE', 'Annually', false, 21),
-- IT General Controls (IT-01 to IT-05)
('IT-01', 'System Access Control', 'Restrict access to GHG calculation systems to authorised personnel with role-based permissions. Quarterly access review.', 'IT_GENERAL', 'PREVENTIVE', 'Quarterly', true, 22),
('IT-02', 'Change Management', 'All changes to GHG calculation logic, emission factors, or system configuration require documented approval and testing.', 'IT_GENERAL', 'PREVENTIVE', 'Per change', true, 23),
('IT-03', 'Audit Trail Integrity', 'Maintain tamper-evident audit trail for all data entries, calculations, and approvals. SHA-256 hashing of calculation chains.', 'IT_GENERAL', 'PREVENTIVE', 'Continuous', true, 24),
('IT-04', 'Backup and Recovery', 'Daily backup of GHG data with tested recovery procedure. Recovery Point Objective (RPO) <24 hours.', 'IT_GENERAL', 'CORRECTIVE', 'Daily', false, 25),
('IT-05', 'Data Retention', 'Retain all GHG source data, calculations, and supporting documentation for minimum 7 years per regulatory requirements.', 'IT_GENERAL', 'PREVENTIVE', 'Continuous', false, 26);

-- =============================================================================
-- Seed Data: Jurisdiction Requirements (12 jurisdictions)
-- =============================================================================

INSERT INTO ghg_assurance.gl_ap_jurisdictions (
    jurisdiction_code, jurisdiction_name, region, assurance_standard,
    assurance_level_required, scopes_required, effective_date, transition_end,
    company_size_threshold, description
) VALUES
('EU_CSRD', 'European Union - CSRD', 'EUROPE',
 'ISAE 3000 / ISAE 3410', 'LIMITED',
 '["SCOPE_1","SCOPE_2","SCOPE_3"]'::jsonb,
 '2024-01-01', '2028-10-01',
 '>250 employees OR >EUR 40M turnover OR >EUR 20M total assets',
 'EU Corporate Sustainability Reporting Directive. Limited assurance from 2024, reasonable assurance from 2028 (transition). Mandatory for large undertakings and listed SMEs.'),

('UK_SECR', 'United Kingdom - SECR', 'EUROPE',
 'ISAE 3410', 'VOLUNTARY',
 '["SCOPE_1","SCOPE_2"]'::jsonb,
 '2019-04-01', NULL,
 'Quoted companies, large unquoted, LLPs >500 employees',
 'UK Streamlined Energy and Carbon Reporting. Mandatory reporting but assurance voluntary. FCA Listing Rules encourage assurance for premium-listed entities.'),

('US_SEC', 'United States - SEC Climate Disclosure', 'NORTH_AMERICA',
 'PCAOB / SSAE 18', 'LIMITED',
 '["SCOPE_1","SCOPE_2"]'::jsonb,
 '2026-01-01', '2033-01-01',
 'Large accelerated filers (>$700M public float)',
 'SEC climate-related disclosure rules. Limited assurance for Scope 1-2 phased in 2026-2028, reasonable assurance from 2033 for large accelerated filers.'),

('AU_NGER', 'Australia - NGER / ASRS', 'ASIA_PACIFIC',
 'ISAE 3410 / ASAE 3410', 'LIMITED',
 '["SCOPE_1","SCOPE_2","SCOPE_3"]'::jsonb,
 '2025-01-01', '2030-01-01',
 'Group 1: >$500M revenue; Group 2: >$200M; Group 3: >$50M',
 'Australian Sustainability Reporting Standards under ASRS. Phased from 2025. Limited assurance initially, progressing to reasonable.'),

('JP_ACT', 'Japan - Act on Promotion of GHG Reduction', 'ASIA_PACIFIC',
 'ISAE 3410 / JICPA', 'VOLUNTARY',
 '["SCOPE_1","SCOPE_2"]'::jsonb,
 '2006-04-01', NULL,
 'Large emitters (>3,000 tCO2e Scope 1)',
 'Japan mandatory GHG reporting. Assurance voluntary but encouraged. ISSB-aligned sustainability standards proposed from 2027.'),

('SG_MAS', 'Singapore - MAS / SGX', 'ASIA_PACIFIC',
 'ISAE 3410', 'LIMITED',
 '["SCOPE_1","SCOPE_2"]'::jsonb,
 '2025-01-01', '2027-01-01',
 'All SGX-listed companies from 2025',
 'Singapore Exchange mandatory climate reporting. Limited assurance for Scope 1-2 from 2025. Scope 3 reporting (not assured) from 2026.'),

('HK_HKEX', 'Hong Kong - HKEX', 'ASIA_PACIFIC',
 'ISAE 3410 / HKICPA', 'LIMITED',
 '["SCOPE_1","SCOPE_2"]'::jsonb,
 '2025-01-01', '2028-01-01',
 'All HKEX Main Board listed issuers',
 'Hong Kong Stock Exchange mandatory climate disclosure. Phased from 2025 for Hang Seng Index constituents, all Main Board by 2026.'),

('NZ_XRB', 'New Zealand - XRB / NZ CS', 'ASIA_PACIFIC',
 'ISAE 3410 / NZ SAE 1', 'LIMITED',
 '["SCOPE_1","SCOPE_2","SCOPE_3"]'::jsonb,
 '2023-01-01', '2026-01-01',
 '>$1B AUM (managers) OR >$1B assets (banks/insurers) OR >250 employees (listed)',
 'New Zealand Climate Standards. First jurisdiction to mandate climate-related disclosures. Limited assurance, moving to reasonable from 2026.'),

('CA_CSSB', 'Canada - CSSB', 'NORTH_AMERICA',
 'CSAE 3410', 'LIMITED',
 '["SCOPE_1","SCOPE_2"]'::jsonb,
 '2025-01-01', '2029-01-01',
 'Federally regulated financial institutions and publicly traded companies',
 'Canadian Sustainability Standards Board. ISSB-aligned. Limited assurance for Scope 1-2 from 2025. Scope 3 phased in later.'),

('CH_TCFD', 'Switzerland - TCFD / ORD / CO2 Act', 'EUROPE',
 'ISAE 3410', 'VOLUNTARY',
 '["SCOPE_1","SCOPE_2"]'::jsonb,
 '2024-01-01', NULL,
 '>500 employees AND >CHF 20M total assets OR >CHF 40M turnover',
 'Swiss Climate Ordinance (ORD) mandates climate reporting. Assurance voluntary but market practice is developing. CO2 Act requires emissions reporting.'),

('BR_CVM', 'Brazil - CVM / CBPS', 'SOUTH_AMERICA',
 'NBC TO / ISA 3410 (BR)', 'VOLUNTARY',
 '["SCOPE_1","SCOPE_2"]'::jsonb,
 '2026-01-01', NULL,
 'CVM-regulated listed companies',
 'Brazil CVM sustainability disclosure resolution. ISSB-aligned via CBPS. Assurance voluntary initially, expected mandatory from 2028.'),

('ZA_JSE', 'South Africa - JSE', 'AFRICA',
 'ISAE 3410', 'VOLUNTARY',
 '["SCOPE_1","SCOPE_2"]'::jsonb,
 '2023-01-01', NULL,
 'JSE-listed companies (main board and AltX)',
 'Johannesburg Stock Exchange sustainability disclosure. King IV recommends independent assurance. JSE Climate Disclosure Guidance expects Scope 1-2 with assurance.');

-- =============================================================================
-- Seed Data: ISAE 3410 Checklist Categories and Weights
-- =============================================================================

CREATE TABLE IF NOT EXISTS ghg_assurance.gl_ap_checklist_templates (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    standard                    VARCHAR(30)     NOT NULL,
    category                    VARCHAR(100)    NOT NULL,
    item_code                   VARCHAR(30)     NOT NULL,
    item_description            TEXT            NOT NULL,
    max_score                   INTEGER         NOT NULL DEFAULT 4,
    is_mandatory                BOOLEAN         NOT NULL DEFAULT false,
    is_gate                     BOOLEAN         NOT NULL DEFAULT false,
    sort_order                  INTEGER         NOT NULL DEFAULT 0,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p048_clt_std_code UNIQUE (standard, item_code)
);

INSERT INTO ghg_assurance.gl_ap_checklist_templates (
    standard, category, item_code, item_description, max_score, is_mandatory, is_gate, sort_order
) VALUES
-- ISAE 3410 Checklist
('ISAE_3410', 'GHG Quantification', 'ISAE-GQ-01', 'Documented quantification methodology aligned with GHG Protocol or ISO 14064-1', 4, true, true, 1),
('ISAE_3410', 'GHG Quantification', 'ISAE-GQ-02', 'Emission factor sources identified and version-controlled', 4, true, false, 2),
('ISAE_3410', 'GHG Quantification', 'ISAE-GQ-03', 'Calculation formulas documented with worked examples', 4, true, false, 3),
('ISAE_3410', 'GHG Quantification', 'ISAE-GQ-04', 'GWP values specified and consistently applied (AR4/AR5/AR6)', 4, true, false, 4),
('ISAE_3410', 'GHG Quantification', 'ISAE-GQ-05', 'Uncertainty assessment conducted and documented', 4, false, false, 5),
('ISAE_3410', 'Organisational Boundary', 'ISAE-OB-01', 'Consolidation approach documented (equity share, operational control, financial control)', 4, true, true, 6),
('ISAE_3410', 'Organisational Boundary', 'ISAE-OB-02', 'Complete list of entities included/excluded with justification', 4, true, false, 7),
('ISAE_3410', 'Organisational Boundary', 'ISAE-OB-03', 'Boundary changes from prior period identified and explained', 4, false, false, 8),
('ISAE_3410', 'Data Management', 'ISAE-DM-01', 'Source data retention for minimum 5 years', 4, true, true, 9),
('ISAE_3410', 'Data Management', 'ISAE-DM-02', 'Audit trail from source data to reported figures', 4, true, true, 10),
('ISAE_3410', 'Data Management', 'ISAE-DM-03', 'Data quality assessment procedures documented', 4, true, false, 11),
('ISAE_3410', 'Data Management', 'ISAE-DM-04', 'Change management process for data corrections', 4, false, false, 12),
('ISAE_3410', 'Internal Controls', 'ISAE-IC-01', 'Control environment over GHG reporting documented', 4, true, true, 13),
('ISAE_3410', 'Internal Controls', 'ISAE-IC-02', 'Key controls identified and tested', 4, true, false, 14),
('ISAE_3410', 'Internal Controls', 'ISAE-IC-03', 'Segregation of duties between preparation and review', 4, true, false, 15),
('ISAE_3410', 'Internal Controls', 'ISAE-IC-04', 'IT general controls over GHG systems assessed', 4, false, false, 16),
('ISAE_3410', 'Reporting', 'ISAE-RP-01', 'GHG statement presented per applicable framework requirements', 4, true, true, 17),
('ISAE_3410', 'Reporting', 'ISAE-RP-02', 'Comparative prior period information provided', 4, true, false, 18),
('ISAE_3410', 'Reporting', 'ISAE-RP-03', 'Base year recalculation policy documented and applied', 4, false, false, 19),
('ISAE_3410', 'Reporting', 'ISAE-RP-04', 'Assumptions and limitations disclosed', 4, true, false, 20),
-- ISO 14064-3 Checklist
('ISO_14064_3', 'Verification Planning', 'ISO-VP-01', 'Verification scope and objectives clearly defined', 4, true, true, 1),
('ISO_14064_3', 'Verification Planning', 'ISO-VP-02', 'Materiality threshold established (quantitative and qualitative)', 4, true, true, 2),
('ISO_14064_3', 'Verification Planning', 'ISO-VP-03', 'Risk assessment of material misstatement completed', 4, true, false, 3),
('ISO_14064_3', 'Evidence Collection', 'ISO-EC-01', 'Sufficient appropriate evidence collected for each material source', 4, true, true, 4),
('ISO_14064_3', 'Evidence Collection', 'ISO-EC-02', 'Analytical procedures applied for trend and consistency checks', 4, true, false, 5),
('ISO_14064_3', 'Evidence Collection', 'ISO-EC-03', 'Site visits conducted for significant sources (if applicable)', 4, false, false, 6),
('ISO_14064_3', 'Evaluation', 'ISO-EV-01', 'GHG assertion evaluated against established criteria', 4, true, true, 7),
('ISO_14064_3', 'Evaluation', 'ISO-EV-02', 'Aggregation of errors assessed against materiality', 4, true, false, 8),
('ISO_14064_3', 'Evaluation', 'ISO-EV-03', 'Conclusion on GHG statement fairness documented', 4, true, true, 9),
-- AA1000AS v3 Checklist
('AA1000AS_V3', 'Inclusivity', 'AA-IN-01', 'Stakeholder engagement process documented and evidence of participation', 4, true, true, 1),
('AA1000AS_V3', 'Inclusivity', 'AA-IN-02', 'Stakeholder concerns reflected in sustainability priorities', 4, true, false, 2),
('AA1000AS_V3', 'Materiality', 'AA-MA-01', 'Material sustainability topics identified using systematic process', 4, true, true, 3),
('AA1000AS_V3', 'Materiality', 'AA-MA-02', 'Materiality assessment reviewed and updated annually', 4, true, false, 4),
('AA1000AS_V3', 'Responsiveness', 'AA-RE-01', 'Policies and actions address material topics', 4, true, false, 5),
('AA1000AS_V3', 'Responsiveness', 'AA-RE-02', 'Targets set for material topics with progress tracking', 4, false, false, 6),
('AA1000AS_V3', 'Impact', 'AA-IM-01', 'Impact of activities on stakeholders assessed and monitored', 4, true, true, 7),
('AA1000AS_V3', 'Impact', 'AA-IM-02', 'Impact measurement methodology documented', 4, false, false, 8);

-- =============================================================================
-- Seed Data: Cost Model Parameters
-- =============================================================================

CREATE TABLE IF NOT EXISTS ghg_assurance.gl_ap_cost_model_parameters (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    parameter_name              VARCHAR(100)    NOT NULL,
    company_size                VARCHAR(30)     NOT NULL,
    base_cost_eur               NUMERIC(14,2)   NOT NULL,
    limited_multiplier          NUMERIC(4,2)    NOT NULL DEFAULT 1.00,
    reasonable_multiplier       NUMERIC(4,2)    NOT NULL DEFAULT 1.80,
    scope3_basic_multiplier     NUMERIC(4,2)    NOT NULL DEFAULT 1.20,
    scope3_full_multiplier      NUMERIC(4,2)    NOT NULL DEFAULT 1.60,
    first_time_multiplier       NUMERIC(4,2)    NOT NULL DEFAULT 1.40,
    internal_fte_hours          NUMERIC(8,2)    NOT NULL,
    notes                       TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p048_cmp_name_size UNIQUE (parameter_name, company_size)
);

INSERT INTO ghg_assurance.gl_ap_cost_model_parameters (
    parameter_name, company_size, base_cost_eur,
    limited_multiplier, reasonable_multiplier,
    scope3_basic_multiplier, scope3_full_multiplier,
    first_time_multiplier, internal_fte_hours, notes
) VALUES
('GHG_ASSURANCE', 'MICRO', 8000.00, 1.00, 1.80, 1.15, 1.40, 1.50, 40, '<10 employees, simple operations, 1-2 sites'),
('GHG_ASSURANCE', 'SMALL', 15000.00, 1.00, 1.80, 1.20, 1.50, 1.45, 80, '10-50 employees, moderate operations, 2-5 sites'),
('GHG_ASSURANCE', 'MEDIUM', 30000.00, 1.00, 1.75, 1.25, 1.55, 1.40, 160, '50-250 employees, complex operations, 5-20 sites'),
('GHG_ASSURANCE', 'LARGE', 60000.00, 1.00, 1.70, 1.30, 1.60, 1.35, 320, '250-1000 employees, multi-sector, 20-50 sites'),
('GHG_ASSURANCE', 'ENTERPRISE', 120000.00, 1.00, 1.65, 1.35, 1.65, 1.30, 640, '1000-5000 employees, multinational, 50-200 sites'),
('GHG_ASSURANCE', 'MEGA', 250000.00, 1.00, 1.60, 1.40, 1.70, 1.25, 1200, '>5000 employees, global operations, 200+ sites');

-- =============================================================================
-- Seed Data: Materiality Default Percentages
-- =============================================================================

CREATE TABLE IF NOT EXISTS ghg_assurance.gl_ap_materiality_defaults (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assurance_level             VARCHAR(20)     NOT NULL,
    materiality_pct             NUMERIC(5,2)    NOT NULL,
    performance_pct_of_overall  NUMERIC(5,2)    NOT NULL,
    trivial_pct_of_overall      NUMERIC(5,2)    NOT NULL,
    source_reference            TEXT,
    notes                       TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p048_md_level UNIQUE (assurance_level)
);

INSERT INTO ghg_assurance.gl_ap_materiality_defaults (
    assurance_level, materiality_pct, performance_pct_of_overall,
    trivial_pct_of_overall, source_reference, notes
) VALUES
('LIMITED', 5.00, 75.00, 5.00, 'ISAE 3410 para A50-A55; GHG Protocol Guidance', 'Limited assurance: 5% overall materiality, 75% performance (of overall), 5% clearly trivial (of overall). Professional judgement may adjust.'),
('REASONABLE', 3.00, 60.00, 3.00, 'ISAE 3410 para A50-A55; ISA 320 analogous guidance', 'Reasonable assurance: 3% overall materiality, 60% performance, 3% trivial. Lower thresholds reflect higher confidence requirement.');

-- =============================================================================
-- Grants for Views and Reference Tables
-- =============================================================================
GRANT SELECT ON ghg_assurance.gl_ap_v_readiness_dashboard TO PUBLIC;
GRANT SELECT ON ghg_assurance.gl_ap_v_evidence_summary TO PUBLIC;
GRANT SELECT ON ghg_assurance.gl_ap_v_control_summary TO PUBLIC;
GRANT SELECT ON ghg_assurance.gl_ap_v_query_status TO PUBLIC;
GRANT SELECT ON ghg_assurance.gl_ap_control_templates TO PUBLIC;
GRANT SELECT ON ghg_assurance.gl_ap_checklist_templates TO PUBLIC;
GRANT SELECT ON ghg_assurance.gl_ap_cost_model_parameters TO PUBLIC;
GRANT SELECT ON ghg_assurance.gl_ap_materiality_defaults TO PUBLIC;

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON MATERIALIZED VIEW ghg_assurance.gl_ap_v_readiness_dashboard IS
    'Latest readiness scores per configuration with gap remediation progress. Refresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.';
COMMENT ON MATERIALIZED VIEW ghg_assurance.gl_ap_v_evidence_summary IS
    'Evidence completeness by scope and category from FINAL packages. Refresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.';
COMMENT ON MATERIALIZED VIEW ghg_assurance.gl_ap_v_control_summary IS
    'Control effectiveness summary by category with test results and deficiency counts. Refresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.';
COMMENT ON MATERIALIZED VIEW ghg_assurance.gl_ap_v_query_status IS
    'Outstanding queries and findings per active engagement with priority and overdue tracking. Refresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.';

COMMENT ON TABLE ghg_assurance.gl_ap_control_templates IS
    'PACK-048: Reference template of 25 standard GHG reporting controls (DC-01 to IT-05) for organisation customisation.';
COMMENT ON TABLE ghg_assurance.gl_ap_checklist_templates IS
    'PACK-048: Standard-specific checklist templates for ISAE 3410, ISO 14064-3, and AA1000AS v3 readiness assessment.';
COMMENT ON TABLE ghg_assurance.gl_ap_cost_model_parameters IS
    'PACK-048: Cost model parameters by company size for assurance cost estimation with multipliers.';
COMMENT ON TABLE ghg_assurance.gl_ap_materiality_defaults IS
    'PACK-048: Default materiality percentages by assurance level (limited vs reasonable).';

-- Table-level comments for all PACK-048 tables
COMMENT ON TABLE ghg_assurance.gl_ap_configurations IS
    'PACK-048: Assurance configuration with standard, level, target scopes, and jurisdiction.';
COMMENT ON TABLE ghg_assurance.gl_ap_engagements IS
    'PACK-048: Assurance engagement lifecycle with verifier, phase, dates, cost, and status.';
COMMENT ON TABLE ghg_assurance.gl_ap_evidence_packages IS
    'PACK-048: Versioned evidence packages with completeness scoring and quality distribution.';
COMMENT ON TABLE ghg_assurance.gl_ap_evidence_items IS
    'PACK-048: Individual evidence items with scope, category, quality grading, and file hashing.';
COMMENT ON TABLE ghg_assurance.gl_ap_evidence_links IS
    'PACK-048: Cross-references between evidence items and platform entities.';
COMMENT ON TABLE ghg_assurance.gl_ap_readiness_assessments IS
    'PACK-048: Readiness assessments with scoring, level classification, and gap tracking.';
COMMENT ON TABLE ghg_assurance.gl_ap_checklist_items IS
    'PACK-048: Granular checklist items per assessment with scoring and mandatory/gate flags.';
COMMENT ON TABLE ghg_assurance.gl_ap_gaps IS
    'PACK-048: Readiness gaps with severity, remediation planning, and lifecycle tracking.';
COMMENT ON TABLE ghg_assurance.gl_ap_provenance_chains IS
    'PACK-048: Provenance chains with Merkle-like hash linking for tamper detection.';
COMMENT ON TABLE ghg_assurance.gl_ap_provenance_steps IS
    'PACK-048: Individual provenance steps with input/output, formulas, EF sources, and hash chain.';
COMMENT ON TABLE ghg_assurance.gl_ap_controls IS
    'PACK-048: Internal controls for GHG reporting with category, type, and maturity level.';
COMMENT ON TABLE ghg_assurance.gl_ap_control_tests IS
    'PACK-048: Control test records with design/operating effectiveness and sample testing.';
COMMENT ON TABLE ghg_assurance.gl_ap_deficiencies IS
    'PACK-048: Control deficiencies with COSO severity classification and remediation lifecycle.';
COMMENT ON TABLE ghg_assurance.gl_ap_queries IS
    'PACK-048: Verifier queries with type, priority, SLA tracking, and status lifecycle.';
COMMENT ON TABLE ghg_assurance.gl_ap_findings IS
    'PACK-048: Verifier findings with severity, scope attribution, and remediation tracking.';
COMMENT ON TABLE ghg_assurance.gl_ap_responses IS
    'PACK-048: Responses to queries and findings with evidence references and acceptance.';
COMMENT ON TABLE ghg_assurance.gl_ap_materiality_assessments IS
    'PACK-048: Materiality thresholds (overall, performance, trivial) with scope breakdowns.';
COMMENT ON TABLE ghg_assurance.gl_ap_sampling_plans IS
    'PACK-048: Statistical sampling plans with method, confidence level, and sample sizing.';
COMMENT ON TABLE ghg_assurance.gl_ap_sample_selections IS
    'PACK-048: Strata within sampling plans with testing progress and projected misstatement.';
COMMENT ON TABLE ghg_assurance.gl_ap_jurisdictions IS
    'PACK-048: Regulatory jurisdiction reference data with assurance requirements.';
COMMENT ON TABLE ghg_assurance.gl_ap_requirements IS
    'PACK-048: Specific regulatory requirements per configuration and jurisdiction.';
COMMENT ON TABLE ghg_assurance.gl_ap_compliance_status IS
    'PACK-048: Aggregate compliance status per configuration-jurisdiction pair.';
COMMENT ON TABLE ghg_assurance.gl_ap_cost_estimates IS
    'PACK-048: Assurance cost estimation with multipliers for level, jurisdiction, and complexity.';
COMMENT ON TABLE ghg_assurance.gl_ap_timeline_milestones IS
    'PACK-048: Engagement milestones with planned vs actual dates and dependencies.';
COMMENT ON TABLE ghg_assurance.gl_ap_resource_plans IS
    'PACK-048: Resource allocation by role and phase with FTE hours and cost tracking.';
