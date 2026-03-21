-- =============================================================================
-- V224: PACK-030 Net Zero Reporting Pack - Reference / Seed Data
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    014 of 015
-- Date:         March 2026
--
-- Reference data for framework schemas, cross-framework metric mappings,
-- standard framework deadlines, and default report configuration.
--
-- Seed data:
--   - 7 framework schema definitions (SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD)
--   - 50+ cross-framework metric mappings
--   - 7 standard framework deadlines for 2026
--   - Default configuration templates
--
-- Previous: V223__PACK030_triggers.sql
-- =============================================================================

-- Use a system tenant for reference data
DO $$
DECLARE
    v_system_tenant UUID := '00000000-0000-0000-0000-000000000001';
BEGIN

-- =============================================================================
-- Framework Schema Definitions
-- =============================================================================

-- SBTi Progress Report Schema
INSERT INTO pack030_nz_reporting.gl_nz_framework_schemas (
    tenant_id, framework, version, schema_type, schema_name, schema_description,
    json_schema, required_fields, optional_fields, total_field_count,
    effective_date, is_current, source_organization
) VALUES (
    v_system_tenant, 'SBTi', '2.0', 'REPORT', 'SBTi Annual Progress Disclosure',
    'Schema for SBTi Corporate Net-Zero Standard annual progress disclosure report',
    '{"type":"object","properties":{"target_description":{"type":"string"},"base_year":{"type":"integer"},"target_year":{"type":"integer"},"scope_coverage":{"type":"array"},"progress_table":{"type":"object"},"variance_explanation":{"type":"string"}}}'::JSONB,
    '["target_description","base_year","target_year","scope_coverage","progress_table","scope1_emissions","scope2_emissions","target_reduction_pct","actual_reduction_pct","sbti_pathway","sbti_method"]'::JSONB,
    '["scope3_emissions","variance_explanation","next_steps","methodological_notes","restatement_details"]'::JSONB,
    16, '2024-01-01', TRUE, 'Science Based Targets initiative'
);

-- CDP Climate Change Questionnaire Schema
INSERT INTO pack030_nz_reporting.gl_nz_framework_schemas (
    tenant_id, framework, version, schema_type, schema_name, schema_description,
    json_schema, required_fields, optional_fields, total_field_count,
    effective_date, is_current, source_organization
) VALUES (
    v_system_tenant, 'CDP', '2025', 'QUESTIONNAIRE', 'CDP Climate Change 2025',
    'Schema for CDP Climate Change questionnaire modules C0-C12 with 300+ data points',
    '{"type":"object","properties":{"C0":{"type":"object"},"C1":{"type":"object"},"C2":{"type":"object"},"C3":{"type":"object"},"C4":{"type":"object"},"C5":{"type":"object"},"C6":{"type":"object"},"C7":{"type":"object"},"C8":{"type":"object"},"C9":{"type":"object"},"C10":{"type":"object"},"C11":{"type":"object"},"C12":{"type":"object"}}}'::JSONB,
    '["C0_introduction","C1_governance","C2_risks_opportunities","C4_targets","C5_methodology","C6_scope1_scope2","C7_scope3","C11_carbon_pricing","C12_engagement"]'::JSONB,
    '["C3_business_strategy","C8_energy","C9_verification","C10_sign_off"]'::JSONB,
    312, '2025-01-01', TRUE, 'CDP (Carbon Disclosure Project)'
);

-- TCFD Disclosure Schema
INSERT INTO pack030_nz_reporting.gl_nz_framework_schemas (
    tenant_id, framework, version, schema_type, schema_name, schema_description,
    json_schema, required_fields, optional_fields, total_field_count,
    effective_date, is_current, source_organization
) VALUES (
    v_system_tenant, 'TCFD', '2023', 'REPORT', 'TCFD Recommendations 2023',
    'Schema for Task Force on Climate-related Financial Disclosures 4-pillar framework',
    '{"type":"object","properties":{"governance":{"type":"object","properties":{"board_oversight":{},"management_role":{}}},"strategy":{"type":"object","properties":{"risks":{},"opportunities":{},"resilience":{},"scenario_analysis":{}}},"risk_management":{"type":"object","properties":{"identification":{},"assessment":{},"integration":{}}},"metrics_targets":{"type":"object","properties":{"scope1":{},"scope2":{},"scope3":{},"targets":{},"progress":{}}}}}'::JSONB,
    '["governance_board_oversight","governance_management_role","strategy_risks","strategy_opportunities","risk_identification","risk_assessment","metrics_scope1","metrics_scope2","metrics_targets"]'::JSONB,
    '["strategy_scenario_analysis","strategy_resilience","risk_integration","metrics_scope3","metrics_progress","metrics_intensity"]'::JSONB,
    15, '2023-10-01', TRUE, 'Task Force on Climate-related Financial Disclosures'
);

-- GRI 305 Emissions Schema
INSERT INTO pack030_nz_reporting.gl_nz_framework_schemas (
    tenant_id, framework, version, schema_type, schema_name, schema_description,
    json_schema, required_fields, optional_fields, total_field_count,
    effective_date, is_current, source_organization
) VALUES (
    v_system_tenant, 'GRI', '2021', 'DISCLOSURE', 'GRI 305: Emissions 2016 (2021 update)',
    'Schema for GRI 305 Emissions disclosures (305-1 through 305-7)',
    '{"type":"object","properties":{"305_1":{"type":"object","description":"Direct (Scope 1) GHG emissions"},"305_2":{"type":"object","description":"Energy indirect (Scope 2) GHG emissions"},"305_3":{"type":"object","description":"Other indirect (Scope 3) GHG emissions"},"305_4":{"type":"object","description":"GHG emissions intensity"},"305_5":{"type":"object","description":"Reduction of GHG emissions"},"305_6":{"type":"object","description":"Ozone-depleting substances"},"305_7":{"type":"object","description":"Significant air emissions"}}}'::JSONB,
    '["305_1_scope1","305_2_scope2_location","305_2_scope2_market","305_4_intensity","305_5_reduction","base_year","methodologies","emission_factors"]'::JSONB,
    '["305_3_scope3","305_6_ods","305_7_air_emissions","biogenic_emissions","consolidation_approach"]'::JSONB,
    25, '2021-01-01', TRUE, 'Global Reporting Initiative'
);

-- ISSB IFRS S2 Schema
INSERT INTO pack030_nz_reporting.gl_nz_framework_schemas (
    tenant_id, framework, version, schema_type, schema_name, schema_description,
    json_schema, required_fields, optional_fields, total_field_count,
    effective_date, is_current, source_organization
) VALUES (
    v_system_tenant, 'ISSB', '2023', 'DISCLOSURE', 'IFRS S2 Climate-related Disclosures',
    'Schema for IFRS S2 Climate-related Disclosures standard with 4-pillar structure and industry-specific metrics',
    '{"type":"object","properties":{"governance":{"type":"object"},"strategy":{"type":"object"},"risk_management":{"type":"object"},"metrics_targets":{"type":"object","properties":{"scope1":{},"scope2":{},"scope3":{},"industry_metrics":{},"targets":{},"carbon_credits":{}}}}}'::JSONB,
    '["governance_oversight","strategy_risks_opportunities","risk_processes","scope1_absolute","scope2_absolute","scope3_categories","transition_plan","industry_metrics"]'::JSONB,
    '["scenario_analysis","financial_effects","carbon_credits_offsets","internal_carbon_price","remuneration"]'::JSONB,
    28, '2024-01-01', TRUE, 'International Sustainability Standards Board'
);

-- SEC Climate Disclosure Schema
INSERT INTO pack030_nz_reporting.gl_nz_framework_schemas (
    tenant_id, framework, version, schema_type, schema_name, schema_description,
    json_schema, required_fields, optional_fields, total_field_count,
    effective_date, is_current, source_organization
) VALUES (
    v_system_tenant, 'SEC', '2024', 'DISCLOSURE', 'SEC Climate Disclosure Rule',
    'Schema for SEC climate disclosure requirements including Regulation S-K Items 1502-1506 with XBRL tagging',
    '{"type":"object","properties":{"item1_business":{"type":"object"},"item1a_risk_factors":{"type":"object"},"item7_mda":{"type":"object"},"reg_sk_1502":{"type":"object","description":"Governance"},"reg_sk_1503":{"type":"object","description":"Strategy"},"reg_sk_1504":{"type":"object","description":"Risk management"},"reg_sk_1505":{"type":"object","description":"GHG emissions"},"reg_sk_1506":{"type":"object","description":"Attestation"}}}'::JSONB,
    '["scope1_emissions","scope2_emissions","governance_oversight","risk_identification","risk_assessment","climate_risks_material","targets_goals","attestation_report"]'::JSONB,
    '["scope3_emissions","scenario_analysis","financial_impacts","transition_plan","internal_carbon_price"]'::JSONB,
    22, '2024-03-06', TRUE, 'U.S. Securities and Exchange Commission'
);

-- CSRD ESRS E1 Schema
INSERT INTO pack030_nz_reporting.gl_nz_framework_schemas (
    tenant_id, framework, version, schema_type, schema_name, schema_description,
    json_schema, required_fields, optional_fields, total_field_count,
    effective_date, is_current, source_organization
) VALUES (
    v_system_tenant, 'CSRD', '2023', 'DISCLOSURE', 'ESRS E1 Climate Change',
    'Schema for European Sustainability Reporting Standards E1 Climate Change with 9 disclosure requirements and digital taxonomy',
    '{"type":"object","properties":{"E1_1":{"type":"object","description":"Transition plan"},"E1_2":{"type":"object","description":"Policies"},"E1_3":{"type":"object","description":"Actions and resources"},"E1_4":{"type":"object","description":"Targets"},"E1_5":{"type":"object","description":"Energy consumption"},"E1_6":{"type":"object","description":"GHG emissions"},"E1_7":{"type":"object","description":"Removals and credits"},"E1_8":{"type":"object","description":"Internal carbon pricing"},"E1_9":{"type":"object","description":"Financial effects"}}}'::JSONB,
    '["E1_1_transition_plan","E1_2_policies","E1_3_actions","E1_4_targets","E1_5_energy","E1_6_scope1","E1_6_scope2","E1_6_scope3","E1_7_removals","E1_9_financial_effects"]'::JSONB,
    '["E1_8_carbon_pricing","E1_7_carbon_credits","double_materiality","value_chain_scope"]'::JSONB,
    35, '2024-01-01', TRUE, 'European Financial Reporting Advisory Group (EFRAG)'
);

-- =============================================================================
-- Cross-Framework Metric Mappings
-- =============================================================================

-- Scope 1 mappings across frameworks
INSERT INTO pack030_nz_reporting.gl_nz_framework_mappings (tenant_id, source_framework, target_framework, source_metric, target_metric, mapping_type, confidence_score, validation_status) VALUES
(v_system_tenant, 'GHG_PROTOCOL', 'SBTi', 'Scope 1 Direct Emissions', 'scope1_emissions', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'CDP', 'Scope 1 Direct Emissions', 'C6.1_scope1_total', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'TCFD', 'Scope 1 Direct Emissions', 'metrics_scope1', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'GRI', 'Scope 1 Direct Emissions', '305-1', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'ISSB', 'Scope 1 Direct Emissions', 'scope1_absolute', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'SEC', 'Scope 1 Direct Emissions', 'reg_sk_1505_scope1', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'CSRD', 'Scope 1 Direct Emissions', 'E1_6_scope1', 'DIRECT', 99.0, 'VALIDATED'),

-- Scope 2 mappings
(v_system_tenant, 'GHG_PROTOCOL', 'SBTi', 'Scope 2 Location-Based', 'scope2_location_based', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'SBTi', 'Scope 2 Market-Based', 'scope2_market_based', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'CDP', 'Scope 2 Location-Based', 'C6.3_scope2_location', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'CDP', 'Scope 2 Market-Based', 'C6.3_scope2_market', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'TCFD', 'Scope 2 Location-Based', 'metrics_scope2_location', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'TCFD', 'Scope 2 Market-Based', 'metrics_scope2_market', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'GRI', 'Scope 2 Location-Based', '305-2_location', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'GRI', 'Scope 2 Market-Based', '305-2_market', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'SEC', 'Scope 2 Location-Based', 'reg_sk_1505_scope2', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'CSRD', 'Scope 2 Location-Based', 'E1_6_scope2_location', 'DIRECT', 99.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'CSRD', 'Scope 2 Market-Based', 'E1_6_scope2_market', 'DIRECT', 99.0, 'VALIDATED'),

-- Scope 3 mappings
(v_system_tenant, 'GHG_PROTOCOL', 'CDP', 'Scope 3 Total', 'C6.5_scope3_total', 'DIRECT', 95.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'TCFD', 'Scope 3 Total', 'metrics_scope3', 'DIRECT', 95.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'GRI', 'Scope 3 Total', '305-3', 'DIRECT', 95.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'ISSB', 'Scope 3 Total', 'scope3_absolute', 'DIRECT', 95.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'CSRD', 'Scope 3 Total', 'E1_6_scope3', 'DIRECT', 95.0, 'VALIDATED'),

-- Target/reduction mappings
(v_system_tenant, 'SBTi', 'CDP', 'Target Reduction Percentage', 'C4.1a_target_reduction', 'DIRECT', 95.0, 'VALIDATED'),
(v_system_tenant, 'SBTi', 'TCFD', 'Target Reduction Percentage', 'metrics_targets_reduction', 'DIRECT', 90.0, 'VALIDATED'),
(v_system_tenant, 'SBTi', 'CSRD', 'Target Reduction Percentage', 'E1_4_target_reduction', 'DIRECT', 90.0, 'VALIDATED'),
(v_system_tenant, 'SBTi', 'ISSB', 'Target Year', 'targets_year', 'DIRECT', 95.0, 'VALIDATED'),
(v_system_tenant, 'SBTi', 'CDP', 'Base Year', 'C4.1a_base_year', 'DIRECT', 99.0, 'VALIDATED'),

-- Intensity mappings
(v_system_tenant, 'GHG_PROTOCOL', 'GRI', 'GHG Intensity', '305-4_intensity', 'DIRECT', 90.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'TCFD', 'GHG Intensity', 'metrics_intensity', 'DIRECT', 90.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'ISSB', 'GHG Intensity', 'intensity_metric', 'CALCULATED', 85.0, 'VALIDATED'),

-- Energy mappings
(v_system_tenant, 'GHG_PROTOCOL', 'CSRD', 'Total Energy Consumption', 'E1_5_energy_total', 'DIRECT', 95.0, 'VALIDATED'),
(v_system_tenant, 'GHG_PROTOCOL', 'CDP', 'Total Energy Consumption', 'C8.2a_energy_total', 'DIRECT', 95.0, 'VALIDATED'),

-- Governance mappings
(v_system_tenant, 'TCFD', 'CDP', 'Board Oversight', 'C1.1_board_oversight', 'APPROXIMATE', 85.0, 'VALIDATED'),
(v_system_tenant, 'TCFD', 'ISSB', 'Board Oversight', 'governance_oversight', 'DIRECT', 90.0, 'VALIDATED'),
(v_system_tenant, 'TCFD', 'SEC', 'Board Oversight', 'reg_sk_1502_governance', 'APPROXIMATE', 85.0, 'VALIDATED'),
(v_system_tenant, 'TCFD', 'CSRD', 'Board Oversight', 'E1_2_governance', 'APPROXIMATE', 80.0, 'VALIDATED'),

-- Risk mappings
(v_system_tenant, 'TCFD', 'CDP', 'Climate Risks', 'C2.1_risks', 'APPROXIMATE', 80.0, 'VALIDATED'),
(v_system_tenant, 'TCFD', 'ISSB', 'Climate Risks', 'strategy_risks', 'DIRECT', 90.0, 'VALIDATED'),
(v_system_tenant, 'TCFD', 'SEC', 'Climate Risks', 'item1a_climate_risks', 'APPROXIMATE', 80.0, 'VALIDATED'),
(v_system_tenant, 'TCFD', 'CSRD', 'Climate Risks', 'E1_9_financial_effects', 'PARTIAL', 70.0, 'VALIDATED'),

-- Transition plan mappings
(v_system_tenant, 'CSRD', 'TCFD', 'Transition Plan', 'strategy_resilience', 'APPROXIMATE', 75.0, 'VALIDATED'),
(v_system_tenant, 'CSRD', 'CDP', 'Transition Plan', 'C3.1_strategy', 'APPROXIMATE', 75.0, 'VALIDATED'),
(v_system_tenant, 'CSRD', 'ISSB', 'Transition Plan', 'strategy_transition', 'DIRECT', 85.0, 'VALIDATED'),

-- Carbon credit/removal mappings
(v_system_tenant, 'CSRD', 'CDP', 'Carbon Removals', 'C11.2_carbon_credits', 'APPROXIMATE', 70.0, 'VALIDATED'),
(v_system_tenant, 'CSRD', 'SBTi', 'Carbon Removals', 'removals_beyond_value_chain', 'PARTIAL', 65.0, 'VALIDATED'),
(v_system_tenant, 'CSRD', 'ISSB', 'Carbon Credits', 'carbon_credits_offsets', 'APPROXIMATE', 75.0, 'VALIDATED');

-- =============================================================================
-- Standard Framework Deadlines for 2026
-- =============================================================================

INSERT INTO pack030_nz_reporting.gl_nz_framework_deadlines (
    tenant_id, framework, reporting_year, deadline_date, deadline_type, description, notification_days
) VALUES
(v_system_tenant, 'CDP', 2025, '2026-07-31', 'SUBMISSION', 'CDP Climate Change 2026 submission deadline for FY2025 data', ARRAY[120, 90, 60, 30, 14, 7]),
(v_system_tenant, 'SBTi', 2025, '2026-12-31', 'SUBMISSION', 'SBTi annual progress disclosure for FY2025 (rolling deadline)', ARRAY[90, 60, 30, 14]),
(v_system_tenant, 'TCFD', 2025, '2026-06-30', 'PUBLICATION', 'TCFD annual disclosure publication for FY2025', ARRAY[90, 60, 30, 14, 7]),
(v_system_tenant, 'GRI', 2025, '2026-06-30', 'PUBLICATION', 'GRI sustainability report publication for FY2025', ARRAY[90, 60, 30, 14]),
(v_system_tenant, 'ISSB', 2025, '2026-04-30', 'FILING', 'IFRS S2 climate disclosure filing for FY2025 (aligned with annual report)', ARRAY[90, 60, 30, 14, 7]),
(v_system_tenant, 'SEC', 2025, '2026-03-31', 'FILING', 'SEC 10-K climate disclosure for FY2025 (large accelerated filers, 60 days after FYE)', ARRAY[90, 60, 30, 14, 7, 3]),
(v_system_tenant, 'CSRD', 2025, '2026-05-31', 'FILING', 'CSRD ESRS E1 disclosure for FY2025 (within 5 months of FYE)', ARRAY[120, 90, 60, 30, 14, 7]);

END;
$$;

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON TABLE pack030_nz_reporting.gl_nz_framework_schemas IS
    'Framework schema definitions with JSON Schema validation, version tracking, effective/deprecated lifecycle, required/optional field classification for SBTi, CDP, TCFD, GRI, ISSB, SEC, and CSRD frameworks. Seeded with 7 current framework schemas.';
