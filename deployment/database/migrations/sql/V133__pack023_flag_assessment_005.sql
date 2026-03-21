-- =============================================================================
-- V133: PACK-023-sbti-alignment-005: FLAG Assessment and Commodity Records
-- =============================================================================
-- Pack:         PACK-023 (SBTi Alignment Pack)
-- Date:         March 2026
--
-- Pack-level tables for FLAG (Forest, Land, and Agriculture) emissions assessment
-- and commodity-specific targets. Covers 11 commodity categories with 20% trigger
-- evaluation, no-deforestation commitments, land use change emissions, and
-- commodity-specific reduction pathways.
--
-- EXTENDS:
--   V087: GL-SBTi-APP v1.0 (FLAG baseline)
--   V129: PACK-023 Target Definitions
--
-- 11 Commodities: cattle, soy, palm_oil, timber, cocoa, coffee, rubber,
--   rice, sugarcane, maize, wheat
--
-- These tables sit in the pack023_sbti_alignment schema and provide
-- the FLAG assessment layer for the pack.
-- =============================================================================
-- Tables (4):
--   1. pack023_flag_assessments              - Overall FLAG assessment records
--   2. pack023_flag_commodity_breakdown      - Per-commodity emissions breakdown
--   3. pack023_flag_deforestation_commitments - Commodity deforestation commitments
--   4. pack023_flag_supply_chain_assessment  - Supply chain and certification analysis
--
-- Also includes: 40+ indexes, update triggers, security grants, and comments.
-- Previous: V132__pack023_sda_pathways_004.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack023_sbti_alignment.pack023_flag_assessments
-- =============================================================================
-- Overall FLAG assessment records determining if organization has >20% FLAG
-- emissions requiring separate FLAG targets, with trigger evaluation and
-- overall FLAG strategy definition.

CREATE TABLE pack023_sbti_alignment.pack023_flag_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    reporting_year          INTEGER,
    assessment_type         VARCHAR(50),
    total_emissions_mt      DECIMAL(18,6),
    total_scope3_emissions  DECIMAL(18,6),
    flag_emissions_mt       DECIMAL(18,6),
    flag_percentage         DECIMAL(6,2),
    flag_threshold          DECIMAL(6,2)    DEFAULT 20.0,
    exceeds_threshold       BOOLEAN         DEFAULT FALSE,
    requires_flag_target    BOOLEAN         DEFAULT FALSE,
    flag_scope              VARCHAR(100),
    flag_target_boundary    VARCHAR(200),
    flag_reduction_rate     DECIMAL(6,4),
    flag_reduction_amount   DECIMAL(18,6),
    flag_pathway_method     VARCHAR(100),
    forest_risk_commodities INTEGER,
    high_risk_commodities   INTEGER,
    low_risk_commodities    INTEGER,
    geographical_focus      VARCHAR(500),
    implementation_strategy TEXT,
    transition_plan         TEXT,
    annual_action_plan      TEXT,
    supply_chain_assessment_status VARCHAR(30),
    certification_coverage  DECIMAL(6,2),
    traceability_status     VARCHAR(30),
    assessed_by             VARCHAR(255),
    approval_status         VARCHAR(30)     DEFAULT 'pending',
    approved_by             VARCHAR(255),
    approved_date           DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_flag_pct CHECK (
        flag_percentage >= 0 AND flag_percentage <= 100
    ),
    CONSTRAINT chk_pk_flag_threshold CHECK (
        flag_threshold > 0 AND flag_threshold <= 100
    ),
    CONSTRAINT chk_pk_flag_reduction CHECK (
        flag_reduction_rate IS NULL OR (flag_reduction_rate >= 0 AND flag_reduction_rate <= 10)
    )
);

-- Indexes
CREATE INDEX idx_pk_flag_org ON pack023_sbti_alignment.pack023_flag_assessments(org_id);
CREATE INDEX idx_pk_flag_tenant ON pack023_sbti_alignment.pack023_flag_assessments(tenant_id);
CREATE INDEX idx_pk_flag_date ON pack023_sbti_alignment.pack023_flag_assessments(assessment_date DESC);
CREATE INDEX idx_pk_flag_year ON pack023_sbti_alignment.pack023_flag_assessments(reporting_year);
CREATE INDEX idx_pk_flag_type ON pack023_sbti_alignment.pack023_flag_assessments(assessment_type);
CREATE INDEX idx_pk_flag_exceeds ON pack023_sbti_alignment.pack023_flag_assessments(exceeds_threshold);
CREATE INDEX idx_pk_flag_requires ON pack023_sbti_alignment.pack023_flag_assessments(requires_flag_target);
CREATE INDEX idx_pk_flag_approval ON pack023_sbti_alignment.pack023_flag_assessments(approval_status);
CREATE INDEX idx_pk_flag_metadata ON pack023_sbti_alignment.pack023_flag_assessments USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_pk_flag_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_flag_assessments
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 2: pack023_sbti_alignment.pack023_flag_commodity_breakdown
-- =============================================================================
-- Per-commodity emissions breakdown for all 11 FLAG commodities with
-- emissions allocation, pathway targets, and commodity-specific metrics.

CREATE TABLE pack023_sbti_alignment.pack023_flag_commodity_breakdown (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_assessment_id      UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_flag_assessments(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    commodity_code          VARCHAR(50)     NOT NULL,
    commodity_name          VARCHAR(255)    NOT NULL,
    commodity_category      VARCHAR(100),
    sourcing_regions        VARCHAR(100)[],
    scope3_category         VARCHAR(50),
    emissions_mt_co2e       DECIMAL(18,6),
    percentage_of_flag      DECIMAL(6,2),
    percentage_of_total     DECIMAL(6,2),
    land_use_change_emissions DECIMAL(18,6),
    deforestation_risk      VARCHAR(30),
    conservation_status     VARCHAR(30),
    production_volume       DECIMAL(18,6),
    production_unit         VARCHAR(100),
    consumption_data_source VARCHAR(100),
    commodity_intensity     DECIMAL(16,8),
    commodity_intensity_unit VARCHAR(100),
    is_material             BOOLEAN         DEFAULT FALSE,
    requires_target         BOOLEAN         DEFAULT FALSE,
    flag_target_pathway     VARCHAR(100),
    flag_annual_reduction   DECIMAL(6,4),
    target_2030_reduction   DECIMAL(8,4),
    target_2050_reduction   DECIMAL(8,4),
    data_quality_tier       VARCHAR(30),
    supplier_list_available BOOLEAN         DEFAULT FALSE,
    supplier_count          INTEGER,
    certified_percentage    DECIMAL(6,2),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_commodity_code CHECK (
        commodity_code IN ('cattle', 'soy', 'palm_oil', 'timber', 'cocoa',
                          'coffee', 'rubber', 'rice', 'sugarcane', 'maize', 'wheat')
    ),
    CONSTRAINT chk_pk_commodity_pct CHECK (
        percentage_of_flag IS NULL OR (percentage_of_flag >= 0 AND percentage_of_flag <= 100)
    ),
    CONSTRAINT chk_pk_commodity_risk CHECK (
        deforestation_risk IN ('VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW')
    )
);

-- Indexes
CREATE INDEX idx_pk_comm_flag_id ON pack023_sbti_alignment.pack023_flag_commodity_breakdown(flag_assessment_id);
CREATE INDEX idx_pk_comm_tenant ON pack023_sbti_alignment.pack023_flag_commodity_breakdown(tenant_id);
CREATE INDEX idx_pk_comm_org ON pack023_sbti_alignment.pack023_flag_commodity_breakdown(org_id);
CREATE INDEX idx_pk_comm_code ON pack023_sbti_alignment.pack023_flag_commodity_breakdown(commodity_code);
CREATE INDEX idx_pk_comm_name ON pack023_sbti_alignment.pack023_flag_commodity_breakdown(commodity_name);
CREATE INDEX idx_pk_comm_material ON pack023_sbti_alignment.pack023_flag_commodity_breakdown(is_material);
CREATE INDEX idx_pk_comm_requires_target ON pack023_sbti_alignment.pack023_flag_commodity_breakdown(requires_target);
CREATE INDEX idx_pk_comm_risk ON pack023_sbti_alignment.pack023_flag_commodity_breakdown(deforestation_risk);
CREATE INDEX idx_pk_comm_created_at ON pack023_sbti_alignment.pack023_flag_commodity_breakdown(created_at DESC);
CREATE INDEX idx_pk_comm_regions ON pack023_sbti_alignment.pack023_flag_commodity_breakdown USING GIN(sourcing_regions);

-- Updated_at trigger
CREATE TRIGGER trg_pk_comm_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_flag_commodity_breakdown
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 3: pack023_sbti_alignment.pack023_flag_deforestation_commitments
-- =============================================================================
-- Deforestation commitments for each commodity with specificity level,
-- scope coverage, third-party validation, and monitoring arrangements.

CREATE TABLE pack023_sbti_alignment.pack023_flag_deforestation_commitments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity_breakdown_id  UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_flag_commodity_breakdown(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    commodity_code          VARCHAR(50),
    commitment_type         VARCHAR(100),
    commitment_statement    TEXT,
    commitment_year         INTEGER,
    commitment_scope        VARCHAR(200),
    geographic_scope        VARCHAR(500)[],
    supply_chain_coverage   VARCHAR(30),
    supply_chain_percentage DECIMAL(6,2),
    is_zero_deforestation   BOOLEAN         DEFAULT FALSE,
    conversion_included     BOOLEAN         DEFAULT FALSE,
    habitat_protection      BOOLEAN         DEFAULT FALSE,
    peatland_protection     BOOLEAN         DEFAULT FALSE,
    high_conservation_value BOOLEAN         DEFAULT FALSE,
    certification_requirement VARCHAR(200),
    monitoring_mechanism    VARCHAR(200)[],
    monitoring_frequency    VARCHAR(100),
    third_party_verification BOOLEAN        DEFAULT FALSE,
    verification_body       VARCHAR(255),
    verification_date       DATE,
    public_commitment       BOOLEAN         DEFAULT FALSE,
    public_commitment_url   VARCHAR(500),
    supplier_notification   BOOLEAN         DEFAULT FALSE,
    supplier_scorecards     BOOLEAN         DEFAULT FALSE,
    grievance_mechanism     BOOLEAN         DEFAULT FALSE,
    target_completion_year  INTEGER,
    implementation_progress DECIMAL(6,2),
    barriers_to_implementation TEXT[],
    mitigation_actions      TEXT[],
    status                  VARCHAR(30)     DEFAULT 'draft',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_defor_type CHECK (
        commitment_type IN ('ZERO_DEFORESTATION', 'CONVERSION_FREE', 'HCV_PROTECTION',
                           'PEATLAND_PROTECTION', 'CUSTOM')
    ),
    CONSTRAINT chk_pk_defor_supply_chain CHECK (
        supply_chain_coverage IN ('TIER1', 'TIER2', 'FULL_SUPPLY_CHAIN', 'SIGNIFICANT_PERCENTAGE')
    ),
    CONSTRAINT chk_pk_defor_progress CHECK (
        implementation_progress >= 0 AND implementation_progress <= 100
    )
);

-- Indexes
CREATE INDEX idx_pk_defor_commodity_id ON pack023_sbti_alignment.pack023_flag_deforestation_commitments(commodity_breakdown_id);
CREATE INDEX idx_pk_defor_tenant ON pack023_sbti_alignment.pack023_flag_deforestation_commitments(tenant_id);
CREATE INDEX idx_pk_defor_org ON pack023_sbti_alignment.pack023_flag_deforestation_commitments(org_id);
CREATE INDEX idx_pk_defor_type ON pack023_sbti_alignment.pack023_flag_deforestation_commitments(commitment_type);
CREATE INDEX idx_pk_defor_verified ON pack023_sbti_alignment.pack023_flag_deforestation_commitments(third_party_verification);
CREATE INDEX idx_pk_defor_status ON pack023_sbti_alignment.pack023_flag_deforestation_commitments(status);
CREATE INDEX idx_pk_defor_created_at ON pack023_sbti_alignment.pack023_flag_deforestation_commitments(created_at DESC);
CREATE INDEX idx_pk_defor_geographic ON pack023_sbti_alignment.pack023_flag_deforestation_commitments USING GIN(geographic_scope);
CREATE INDEX idx_pk_defor_mechanisms ON pack023_sbti_alignment.pack023_flag_deforestation_commitments USING GIN(monitoring_mechanism);

-- Updated_at trigger
CREATE TRIGGER trg_pk_defor_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_flag_deforestation_commitments
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 4: pack023_sbti_alignment.pack023_flag_supply_chain_assessment
-- =============================================================================
-- Supply chain and certification analysis for FLAG commodities including
-- supplier mapping, certification coverage, and traceability assessment.

CREATE TABLE pack023_sbti_alignment.pack023_flag_supply_chain_assessment (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity_breakdown_id  UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_flag_commodity_breakdown(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    commodity_code          VARCHAR(50),
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    supply_chain_tier_1_count INTEGER,
    supply_chain_tier_2_count INTEGER,
    supply_chain_tier_3_plus_count INTEGER,
    primary_region          VARCHAR(200),
    secondary_regions       VARCHAR(200)[],
    top_supplier_percentage DECIMAL(6,2),
    supply_diversity_score  DECIMAL(5,2),
    certifications_tracked  VARCHAR(100)[],
    fsc_certified_pct       DECIMAL(6,2),
    rainforest_alliance_pct DECIMAL(6,2),
    organic_certified_pct   DECIMAL(6,2),
    other_certification_pct DECIMAL(6,2),
    certification_gap       DECIMAL(6,2),
    traceability_method     VARCHAR(200),
    traceability_coverage   DECIMAL(6,2),
    blockchain_enabled      BOOLEAN         DEFAULT FALSE,
    lot_segregation_capability BOOLEAN     DEFAULT FALSE,
    mass_balance_capability BOOLEAN         DEFAULT FALSE,
    gis_mapping_available   BOOLEAN         DEFAULT FALSE,
    geolocation_coverage    DECIMAL(6,2),
    deforestation_risk_mapping BOOLEAN     DEFAULT FALSE,
    grievance_mechanism_active BOOLEAN     DEFAULT FALSE,
    grievance_count_12m     INTEGER,
    grievance_resolution_rate DECIMAL(6,2),
    remediation_tracking    BOOLEAN         DEFAULT FALSE,
    supplier_training_program BOOLEAN      DEFAULT FALSE,
    training_coverage_pct   DECIMAL(6,2),
    audit_schedule          VARCHAR(100),
    audit_frequency         INTEGER,
    audit_findings_summary  TEXT,
    assessment_status       VARCHAR(30),
    gaps_identified         TEXT[],
    remediation_plan        TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_supp_chain_tier CHECK (
        (supply_chain_tier_1_count >= 0) AND
        (supply_chain_tier_2_count >= 0) AND
        (supply_chain_tier_3_plus_count >= 0)
    ),
    CONSTRAINT chk_pk_supp_cert CHECK (
        (fsc_certified_pct IS NULL OR (fsc_certified_pct >= 0 AND fsc_certified_pct <= 100)) AND
        (rainforest_alliance_pct IS NULL OR (rainforest_alliance_pct >= 0 AND rainforest_alliance_pct <= 100))
    )
);

-- Indexes
CREATE INDEX idx_pk_supp_commodity_id ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment(commodity_breakdown_id);
CREATE INDEX idx_pk_supp_tenant ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment(tenant_id);
CREATE INDEX idx_pk_supp_org ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment(org_id);
CREATE INDEX idx_pk_supp_date ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment(assessment_date DESC);
CREATE INDEX idx_pk_supp_status ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment(assessment_status);
CREATE INDEX idx_pk_supp_traceability ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment(traceability_coverage);
CREATE INDEX idx_pk_supp_certifications ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment USING GIN(certifications_tracked);
CREATE INDEX idx_pk_supp_regions ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment USING GIN(secondary_regions);
CREATE INDEX idx_pk_supp_gaps ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment USING GIN(gaps_identified);

-- Updated_at trigger
CREATE TRIGGER trg_pk_supp_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack023_sbti_alignment TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack023_sbti_alignment TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack023_sbti_alignment TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack023_sbti_alignment.pack023_flag_assessments IS
'Overall FLAG assessment records determining if organization requires separate FLAG targets (>20% FLAG emissions) with strategy definition.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_flag_commodity_breakdown IS
'Per-commodity emissions breakdown for all 11 FLAG commodities with risk assessment and commodity-specific reduction pathway targets.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_flag_deforestation_commitments IS
'Deforestation commitments for each commodity with specificity level, scope, third-party verification, and monitoring arrangements.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_flag_supply_chain_assessment IS
'Supply chain analysis for FLAG commodities including supplier mapping, certification coverage, traceability capability, and audit findings.';
