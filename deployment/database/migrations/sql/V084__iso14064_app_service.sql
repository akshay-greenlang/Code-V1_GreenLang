-- =============================================================================
-- V084: GL-ISO14064-APP v1.0 Application Service Schema
-- =============================================================================
-- Application: GL-ISO14064-APP (ISO 14064-1 GHG Compliance Platform)
-- Date:        March 2026
--
-- Application-level tables for ISO 14064-1:2018 compliant GHG quantification
-- and reporting at the organization level.  Covers organizational and
-- operational boundary definition, emission source and removal inventories,
-- significance assessments, base year management, third-party verification,
-- report generation, and management action tracking.
--
-- EXTENDS:
--   V051-V070: AGENT-MRV calculation agents (Scope 1-3)
--   V080: Scope 3 Category Mapper Service
--   V081: Audit Trail & Lineage Service
--
-- These tables sit in the iso14064_app schema and aggregate results from
-- the underlying MRV agent schemas.  They provide the user-facing
-- application layer for ISO 14064-1 compliant GHG inventories including
-- organizations, entities, inventories, boundaries, emission sources,
-- removal sources, significance assessments, base year records,
-- verification, reports, and management actions.
-- =============================================================================
-- Tables (14):
--   1.  gl_iso14064_organizations           - Organization profiles
--   2.  gl_iso14064_entities                - Reporting entities
--   3.  gl_iso14064_inventories             - GHG inventories
--   4.  gl_iso14064_organizational_boundaries - Org boundary configs
--   5.  gl_iso14064_operational_boundaries  - Operational boundary configs
--   6.  gl_iso14064_emission_sources        - Emission source entries
--   7.  gl_iso14064_removal_sources         - GHG removal/sink entries
--   8.  gl_iso14064_significance_assessments - Significance assessments
--   9.  gl_iso14064_base_year_records       - Base year records
--  10.  gl_iso14064_base_year_triggers      - Base year recalc triggers
--  11.  gl_iso14064_verification_records    - Verification engagements
--  12.  gl_iso14064_findings                - Verification findings
--  13.  gl_iso14064_reports                 - Generated reports
--  14.  gl_iso14064_management_actions      - Management plan actions
--
-- Hypertables (3):
--  15.  gl_iso14064_emission_timeseries     - Emission time-series
--  16.  gl_iso14064_verification_events     - Verification event log
--  17.  gl_iso14064_audit_log               - Entity audit trail
--
-- Continuous Aggregates (2):
--  18.  gl_iso14064_daily_emissions         - Daily emission aggregates
--  19.  gl_iso14064_monthly_emissions       - Monthly emission aggregates
--
-- Also includes: 50+ indexes (B-tree, GIN), update triggers, security
-- grants, retention policies, compression policies, seed data, and comments.
-- Previous: V083__ghg_app_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS iso14064_app;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION iso14064_app.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: iso14064_app.gl_iso14064_organizations
-- =============================================================================
-- Organization profiles for ISO 14064-1 compliance.  Each organization
-- is the top-level reporting entity that defines its GHG inventory scope,
-- boundaries, and verification programme.

CREATE TABLE iso14064_app.gl_iso14064_organizations (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(500)    NOT NULL,
    industry        VARCHAR(200),
    country         CHAR(3),
    description     TEXT,
    metadata        JSONB           DEFAULT '{}',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_org_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_org_country_length CHECK (
        country IS NULL OR LENGTH(TRIM(country)) = 3
    )
);

-- Indexes for gl_iso14064_organizations
CREATE INDEX idx_iso14064_org_name ON iso14064_app.gl_iso14064_organizations(name);
CREATE INDEX idx_iso14064_org_industry ON iso14064_app.gl_iso14064_organizations(industry);
CREATE INDEX idx_iso14064_org_country ON iso14064_app.gl_iso14064_organizations(country);
CREATE INDEX idx_iso14064_org_created_at ON iso14064_app.gl_iso14064_organizations(created_at DESC);
CREATE INDEX idx_iso14064_org_metadata ON iso14064_app.gl_iso14064_organizations USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_iso14064_org_updated_at
    BEFORE UPDATE ON iso14064_app.gl_iso14064_organizations
    FOR EACH ROW
    EXECUTE FUNCTION iso14064_app.set_updated_at();

-- =============================================================================
-- Table 2: iso14064_app.gl_iso14064_entities
-- =============================================================================
-- Reporting entities forming the organizational hierarchy.  Entities
-- represent subsidiaries, facilities, or operations within the
-- organizational boundary.  Supports self-referencing hierarchy via
-- parent_id and ownership percentage for equity share consolidation.

CREATE TABLE iso14064_app.gl_iso14064_entities (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_organizations(id) ON DELETE CASCADE,
    name            VARCHAR(500)    NOT NULL,
    entity_type     VARCHAR(50)     NOT NULL,
    parent_id       UUID            REFERENCES iso14064_app.gl_iso14064_entities(id) ON DELETE SET NULL,
    ownership_pct   DECIMAL(5,2)    DEFAULT 100.00,
    country         CHAR(3),
    employees       INTEGER         DEFAULT 0,
    revenue         DECIMAL(15,2)   DEFAULT 0,
    floor_area_m2   DECIMAL(12,2),
    active          BOOLEAN         DEFAULT TRUE,
    metadata        JSONB           DEFAULT '{}',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_entity_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_entity_type CHECK (
        entity_type IN ('SUBSIDIARY', 'FACILITY', 'OPERATION', 'DIVISION', 'JOINT_VENTURE')
    ),
    CONSTRAINT chk_entity_ownership_range CHECK (
        ownership_pct >= 0 AND ownership_pct <= 100
    ),
    CONSTRAINT chk_entity_employees_non_neg CHECK (
        employees >= 0
    ),
    CONSTRAINT chk_entity_revenue_non_neg CHECK (
        revenue >= 0
    ),
    CONSTRAINT chk_entity_floor_area_non_neg CHECK (
        floor_area_m2 IS NULL OR floor_area_m2 >= 0
    ),
    CONSTRAINT chk_entity_country_length CHECK (
        country IS NULL OR LENGTH(TRIM(country)) = 3
    )
);

-- Indexes for gl_iso14064_entities
CREATE INDEX idx_iso14064_entity_org ON iso14064_app.gl_iso14064_entities(org_id);
CREATE INDEX idx_iso14064_entity_parent ON iso14064_app.gl_iso14064_entities(parent_id);
CREATE INDEX idx_iso14064_entity_type ON iso14064_app.gl_iso14064_entities(entity_type);
CREATE INDEX idx_iso14064_entity_country ON iso14064_app.gl_iso14064_entities(country);
CREATE INDEX idx_iso14064_entity_active ON iso14064_app.gl_iso14064_entities(active);
CREATE INDEX idx_iso14064_entity_created_at ON iso14064_app.gl_iso14064_entities(created_at DESC);
CREATE INDEX idx_iso14064_entity_metadata ON iso14064_app.gl_iso14064_entities USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_iso14064_entity_updated_at
    BEFORE UPDATE ON iso14064_app.gl_iso14064_entities
    FOR EACH ROW
    EXECUTE FUNCTION iso14064_app.set_updated_at();

-- =============================================================================
-- Table 3: iso14064_app.gl_iso14064_inventories
-- =============================================================================
-- GHG inventories per ISO 14064-1:2018 Clause 5.  The central object
-- aggregating all emission and removal data for an organization-year.
-- Each inventory records the consolidation approach, GWP source, and
-- lifecycle status.  Provenance hash ensures audit integrity.

CREATE TABLE iso14064_app.gl_iso14064_inventories (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_organizations(id) ON DELETE CASCADE,
    reporting_year          INTEGER         NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    consolidation_approach  VARCHAR(50)     NOT NULL,
    gwp_source              VARCHAR(100)    NOT NULL DEFAULT 'IPCC AR6',
    status                  VARCHAR(50)     NOT NULL DEFAULT 'draft',
    -- Aggregated totals (populated by calculation pipeline)
    total_direct_tco2e      DECIMAL(15,3)   DEFAULT 0,
    total_indirect_energy_tco2e DECIMAL(15,3) DEFAULT 0,
    total_indirect_other_tco2e  DECIMAL(15,3) DEFAULT 0,
    total_removals_tco2e    DECIMAL(15,3)   DEFAULT 0,
    net_emissions_tco2e     DECIMAL(15,3)   DEFAULT 0,
    biogenic_co2_tco2e      DECIMAL(15,3)   DEFAULT 0,
    provenance_hash         CHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_inv_year_range CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_inv_period_order CHECK (
        period_end >= period_start
    ),
    CONSTRAINT chk_inv_consolidation CHECK (
        consolidation_approach IN (
            'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL', 'EQUITY_SHARE'
        )
    ),
    CONSTRAINT chk_inv_status CHECK (
        status IN ('draft', 'in_review', 'approved', 'verified', 'published')
    ),
    CONSTRAINT chk_inv_total_direct_non_neg CHECK (
        total_direct_tco2e >= 0
    ),
    CONSTRAINT chk_inv_total_indirect_energy_non_neg CHECK (
        total_indirect_energy_tco2e >= 0
    ),
    CONSTRAINT chk_inv_total_indirect_other_non_neg CHECK (
        total_indirect_other_tco2e >= 0
    ),
    CONSTRAINT chk_inv_total_removals_non_neg CHECK (
        total_removals_tco2e >= 0
    ),
    CONSTRAINT chk_inv_biogenic_non_neg CHECK (
        biogenic_co2_tco2e >= 0
    ),
    UNIQUE(org_id, reporting_year)
);

-- Indexes for gl_iso14064_inventories
CREATE INDEX idx_iso14064_inv_org ON iso14064_app.gl_iso14064_inventories(org_id);
CREATE INDEX idx_iso14064_inv_year ON iso14064_app.gl_iso14064_inventories(reporting_year);
CREATE INDEX idx_iso14064_inv_status ON iso14064_app.gl_iso14064_inventories(status);
CREATE INDEX idx_iso14064_inv_consolidation ON iso14064_app.gl_iso14064_inventories(consolidation_approach);
CREATE INDEX idx_iso14064_inv_created_at ON iso14064_app.gl_iso14064_inventories(created_at DESC);
CREATE INDEX idx_iso14064_inv_metadata ON iso14064_app.gl_iso14064_inventories USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_iso14064_inv_updated_at
    BEFORE UPDATE ON iso14064_app.gl_iso14064_inventories
    FOR EACH ROW
    EXECUTE FUNCTION iso14064_app.set_updated_at();

-- =============================================================================
-- Table 4: iso14064_app.gl_iso14064_organizational_boundaries
-- =============================================================================
-- Organizational boundary configuration per ISO 14064-1:2018 Clause 5.1.
-- Defines the consolidation approach and which entities are included in
-- the GHG inventory organizational boundary.

CREATE TABLE iso14064_app.gl_iso14064_organizational_boundaries (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_organizations(id) ON DELETE CASCADE,
    consolidation_approach  VARCHAR(50)     NOT NULL,
    entity_ids              JSONB           NOT NULL DEFAULT '[]',
    justification           TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ob_consolidation CHECK (
        consolidation_approach IN (
            'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL', 'EQUITY_SHARE'
        )
    )
);

-- Indexes for gl_iso14064_organizational_boundaries
CREATE INDEX idx_iso14064_ob_org ON iso14064_app.gl_iso14064_organizational_boundaries(org_id);
CREATE INDEX idx_iso14064_ob_approach ON iso14064_app.gl_iso14064_organizational_boundaries(consolidation_approach);
CREATE INDEX idx_iso14064_ob_entity_ids ON iso14064_app.gl_iso14064_organizational_boundaries USING GIN(entity_ids);
CREATE INDEX idx_iso14064_ob_created_at ON iso14064_app.gl_iso14064_organizational_boundaries(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_iso14064_ob_updated_at
    BEFORE UPDATE ON iso14064_app.gl_iso14064_organizational_boundaries
    FOR EACH ROW
    EXECUTE FUNCTION iso14064_app.set_updated_at();

-- =============================================================================
-- Table 5: iso14064_app.gl_iso14064_operational_boundaries
-- =============================================================================
-- Operational boundary configuration per ISO 14064-1:2018 Clause 5.2.
-- Defines which emission/removal categories are included, their
-- significance level, and justification for any exclusions.  The
-- categories JSONB array stores objects with keys: category, included,
-- significance, justification.

CREATE TABLE iso14064_app.gl_iso14064_operational_boundaries (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_organizations(id) ON DELETE CASCADE,
    categories      JSONB           NOT NULL DEFAULT '[]',
    justification   TEXT,
    metadata        JSONB           DEFAULT '{}',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Indexes for gl_iso14064_operational_boundaries
CREATE INDEX idx_iso14064_opb_org ON iso14064_app.gl_iso14064_operational_boundaries(org_id);
CREATE INDEX idx_iso14064_opb_categories ON iso14064_app.gl_iso14064_operational_boundaries USING GIN(categories);
CREATE INDEX idx_iso14064_opb_created_at ON iso14064_app.gl_iso14064_operational_boundaries(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_iso14064_opb_updated_at
    BEFORE UPDATE ON iso14064_app.gl_iso14064_operational_boundaries
    FOR EACH ROW
    EXECUTE FUNCTION iso14064_app.set_updated_at();

-- =============================================================================
-- Table 6: iso14064_app.gl_iso14064_emission_sources
-- =============================================================================
-- Individual emission source entries per ISO 14064-1:2018 Clause 5.2.
-- Records activity data, emission factors, GWP values, and calculated
-- tCO2e for each source within an inventory.  Supports categories:
-- direct (Scope 1), indirect_energy (Scope 2), and
-- indirect_other (Scope 3) per ISO 14064-1 classification.

CREATE TABLE iso14064_app.gl_iso14064_emission_sources (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    inventory_id        UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_inventories(id) ON DELETE CASCADE,
    category            VARCHAR(50)     NOT NULL,
    source_name         VARCHAR(500)    NOT NULL,
    facility_id         UUID,
    gas                 VARCHAR(20)     NOT NULL DEFAULT 'CO2',
    method              VARCHAR(100),
    activity_data       DECIMAL(15,6),
    activity_unit       VARCHAR(50),
    emission_factor     DECIMAL(15,8),
    ef_unit             VARCHAR(100),
    ef_source           VARCHAR(200),
    gwp                 DECIMAL(10,2),
    raw_emissions_tonnes DECIMAL(15,6),
    tco2e               DECIMAL(15,6),
    biogenic_co2        DECIMAL(15,6)   DEFAULT 0,
    data_quality_tier   VARCHAR(20)     DEFAULT 'TIER_2',
    provenance_hash     CHAR(64),
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_es_category CHECK (
        category IN (
            'direct',
            'indirect_energy',
            'indirect_transportation',
            'indirect_products',
            'indirect_other'
        )
    ),
    CONSTRAINT chk_es_source_name_not_empty CHECK (
        LENGTH(TRIM(source_name)) > 0
    ),
    CONSTRAINT chk_es_gas CHECK (
        gas IN ('CO2', 'CH4', 'N2O', 'HFCs', 'PFCs', 'SF6', 'NF3', 'OTHER')
    ),
    CONSTRAINT chk_es_tco2e_non_neg CHECK (
        tco2e IS NULL OR tco2e >= 0
    ),
    CONSTRAINT chk_es_biogenic_non_neg CHECK (
        biogenic_co2 >= 0
    ),
    CONSTRAINT chk_es_data_quality CHECK (
        data_quality_tier IN ('TIER_1', 'TIER_2', 'TIER_3')
    ),
    CONSTRAINT chk_es_gwp_positive CHECK (
        gwp IS NULL OR gwp >= 0
    )
);

-- Indexes for gl_iso14064_emission_sources
CREATE INDEX idx_iso14064_es_inventory ON iso14064_app.gl_iso14064_emission_sources(inventory_id);
CREATE INDEX idx_iso14064_es_category ON iso14064_app.gl_iso14064_emission_sources(category);
CREATE INDEX idx_iso14064_es_gas ON iso14064_app.gl_iso14064_emission_sources(gas);
CREATE INDEX idx_iso14064_es_facility ON iso14064_app.gl_iso14064_emission_sources(facility_id);
CREATE INDEX idx_iso14064_es_method ON iso14064_app.gl_iso14064_emission_sources(method);
CREATE INDEX idx_iso14064_es_data_quality ON iso14064_app.gl_iso14064_emission_sources(data_quality_tier);
CREATE INDEX idx_iso14064_es_created_at ON iso14064_app.gl_iso14064_emission_sources(created_at DESC);
CREATE INDEX idx_iso14064_es_metadata ON iso14064_app.gl_iso14064_emission_sources USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_iso14064_es_updated_at
    BEFORE UPDATE ON iso14064_app.gl_iso14064_emission_sources
    FOR EACH ROW
    EXECUTE FUNCTION iso14064_app.set_updated_at();

-- =============================================================================
-- Table 7: iso14064_app.gl_iso14064_removal_sources
-- =============================================================================
-- GHG removal and sink entries per ISO 14064-1:2018 Clause 5.2.
-- Records gross removals, permanence assessment, crediting discounts,
-- biogenic CO2 flows, and verification status.  Supports the ISO 14064-1
-- requirement to report removals separately from emissions.

CREATE TABLE iso14064_app.gl_iso14064_removal_sources (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    inventory_id                UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_inventories(id) ON DELETE CASCADE,
    facility_id                 UUID,
    removal_type                VARCHAR(100)    NOT NULL,
    source_name                 VARCHAR(500)    NOT NULL,
    gross_removals_tco2e        DECIMAL(15,6)   NOT NULL,
    permanence_level            VARCHAR(50),
    permanence_discount_factor  DECIMAL(5,4)    DEFAULT 1.0000,
    credited_removals_tco2e     DECIMAL(15,6),
    biogenic_co2_removals       DECIMAL(15,6)   DEFAULT 0,
    biogenic_co2_emissions      DECIMAL(15,6)   DEFAULT 0,
    verification_status         VARCHAR(50)     DEFAULT 'unverified',
    monitoring_plan             TEXT,
    data_quality_tier           VARCHAR(20)     DEFAULT 'TIER_2',
    provenance_hash             CHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_rs_source_name_not_empty CHECK (
        LENGTH(TRIM(source_name)) > 0
    ),
    CONSTRAINT chk_rs_removal_type CHECK (
        removal_type IN (
            'AFFORESTATION', 'REFORESTATION', 'SOIL_CARBON',
            'BIOENERGY_CCS', 'DIRECT_AIR_CAPTURE', 'ENHANCED_WEATHERING',
            'BIOCHAR', 'OCEAN_BASED', 'OTHER'
        )
    ),
    CONSTRAINT chk_rs_gross_non_neg CHECK (
        gross_removals_tco2e >= 0
    ),
    CONSTRAINT chk_rs_permanence_level CHECK (
        permanence_level IS NULL OR permanence_level IN (
            'PERMANENT', 'LONG_TERM', 'MEDIUM_TERM', 'SHORT_TERM'
        )
    ),
    CONSTRAINT chk_rs_discount_range CHECK (
        permanence_discount_factor >= 0 AND permanence_discount_factor <= 1
    ),
    CONSTRAINT chk_rs_credited_non_neg CHECK (
        credited_removals_tco2e IS NULL OR credited_removals_tco2e >= 0
    ),
    CONSTRAINT chk_rs_biogenic_removals_non_neg CHECK (
        biogenic_co2_removals >= 0
    ),
    CONSTRAINT chk_rs_biogenic_emissions_non_neg CHECK (
        biogenic_co2_emissions >= 0
    ),
    CONSTRAINT chk_rs_verification_status CHECK (
        verification_status IN ('unverified', 'pending', 'verified', 'rejected')
    ),
    CONSTRAINT chk_rs_data_quality CHECK (
        data_quality_tier IN ('TIER_1', 'TIER_2', 'TIER_3')
    )
);

-- Indexes for gl_iso14064_removal_sources
CREATE INDEX idx_iso14064_rs_inventory ON iso14064_app.gl_iso14064_removal_sources(inventory_id);
CREATE INDEX idx_iso14064_rs_removal_type ON iso14064_app.gl_iso14064_removal_sources(removal_type);
CREATE INDEX idx_iso14064_rs_facility ON iso14064_app.gl_iso14064_removal_sources(facility_id);
CREATE INDEX idx_iso14064_rs_verification ON iso14064_app.gl_iso14064_removal_sources(verification_status);
CREATE INDEX idx_iso14064_rs_data_quality ON iso14064_app.gl_iso14064_removal_sources(data_quality_tier);
CREATE INDEX idx_iso14064_rs_created_at ON iso14064_app.gl_iso14064_removal_sources(created_at DESC);
CREATE INDEX idx_iso14064_rs_metadata ON iso14064_app.gl_iso14064_removal_sources USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_iso14064_rs_updated_at
    BEFORE UPDATE ON iso14064_app.gl_iso14064_removal_sources
    FOR EACH ROW
    EXECUTE FUNCTION iso14064_app.set_updated_at();

-- =============================================================================
-- Table 8: iso14064_app.gl_iso14064_significance_assessments
-- =============================================================================
-- Significance assessments per ISO 14064-1:2018 Clause 5.3.  Determines
-- whether an emission/removal category is significant based on weighted
-- criteria scoring.  Used to justify inclusions and exclusions in the
-- operational boundary.

CREATE TABLE iso14064_app.gl_iso14064_significance_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    inventory_id                UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_inventories(id) ON DELETE CASCADE,
    category                    VARCHAR(50)     NOT NULL,
    criteria                    JSONB           NOT NULL DEFAULT '{}',
    total_weighted_score        DECIMAL(8,4),
    threshold                   DECIMAL(8,4)    DEFAULT 5.0000,
    result                      VARCHAR(20)     NOT NULL DEFAULT 'pending',
    estimated_magnitude_tco2e   DECIMAL(15,3),
    magnitude_pct_of_total      DECIMAL(8,4),
    assessed_by                 VARCHAR(200),
    assessed_at                 TIMESTAMPTZ,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sa_category CHECK (
        category IN (
            'direct',
            'indirect_energy',
            'indirect_transportation',
            'indirect_products',
            'indirect_other'
        )
    ),
    CONSTRAINT chk_sa_result CHECK (
        result IN ('significant', 'not_significant', 'pending')
    ),
    CONSTRAINT chk_sa_magnitude_non_neg CHECK (
        estimated_magnitude_tco2e IS NULL OR estimated_magnitude_tco2e >= 0
    ),
    CONSTRAINT chk_sa_pct_range CHECK (
        magnitude_pct_of_total IS NULL OR (
            magnitude_pct_of_total >= 0 AND magnitude_pct_of_total <= 100
        )
    )
);

-- Indexes for gl_iso14064_significance_assessments
CREATE INDEX idx_iso14064_sa_inventory ON iso14064_app.gl_iso14064_significance_assessments(inventory_id);
CREATE INDEX idx_iso14064_sa_category ON iso14064_app.gl_iso14064_significance_assessments(category);
CREATE INDEX idx_iso14064_sa_result ON iso14064_app.gl_iso14064_significance_assessments(result);
CREATE INDEX idx_iso14064_sa_assessed_at ON iso14064_app.gl_iso14064_significance_assessments(assessed_at DESC);
CREATE INDEX idx_iso14064_sa_criteria ON iso14064_app.gl_iso14064_significance_assessments USING GIN(criteria);

-- =============================================================================
-- Table 9: iso14064_app.gl_iso14064_base_year_records
-- =============================================================================
-- Base year records per ISO 14064-1:2018 Clause 5.4.  Stores the
-- original and recalculated emissions for the base year, along with
-- recalculation policy parameters and significance thresholds.

CREATE TABLE iso14064_app.gl_iso14064_base_year_records (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                          UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_organizations(id) ON DELETE CASCADE,
    base_year                       INTEGER         NOT NULL,
    original_emissions_tco2e        DECIMAL(15,3)   NOT NULL,
    recalculated_emissions_tco2e    DECIMAL(15,3),
    recalculation_reason            TEXT,
    recalculation_date              TIMESTAMPTZ,
    recalculation_policy            TEXT,
    significance_threshold_pct      DECIMAL(5,2)    DEFAULT 5.00,
    provenance_hash                 CHAR(64),
    metadata                        JSONB           DEFAULT '{}',
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_by_year_range CHECK (
        base_year >= 1990 AND base_year <= 2100
    ),
    CONSTRAINT chk_by_original_non_neg CHECK (
        original_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_by_recalc_non_neg CHECK (
        recalculated_emissions_tco2e IS NULL OR recalculated_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_by_threshold_range CHECK (
        significance_threshold_pct >= 0 AND significance_threshold_pct <= 100
    ),
    UNIQUE(org_id, base_year)
);

-- Indexes for gl_iso14064_base_year_records
CREATE INDEX idx_iso14064_by_org ON iso14064_app.gl_iso14064_base_year_records(org_id);
CREATE INDEX idx_iso14064_by_year ON iso14064_app.gl_iso14064_base_year_records(base_year);
CREATE INDEX idx_iso14064_by_created_at ON iso14064_app.gl_iso14064_base_year_records(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_iso14064_by_updated_at
    BEFORE UPDATE ON iso14064_app.gl_iso14064_base_year_records
    FOR EACH ROW
    EXECUTE FUNCTION iso14064_app.set_updated_at();

-- =============================================================================
-- Table 10: iso14064_app.gl_iso14064_base_year_triggers
-- =============================================================================
-- Base year recalculation triggers per ISO 14064-1:2018 Clause 5.4.
-- Records structural changes (mergers, acquisitions, methodology changes)
-- that may require base year recalculation when their impact exceeds the
-- significance threshold.

CREATE TABLE iso14064_app.gl_iso14064_base_year_triggers (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_organizations(id) ON DELETE CASCADE,
    trigger_type            VARCHAR(100)    NOT NULL,
    description             TEXT,
    impact_tco2e            DECIMAL(15,3),
    impact_pct              DECIMAL(8,4),
    requires_recalculation  BOOLEAN         DEFAULT FALSE,
    triggered_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                JSONB           DEFAULT '{}',
    CONSTRAINT chk_bt_trigger_type CHECK (
        trigger_type IN (
            'MERGER', 'ACQUISITION', 'DIVESTITURE',
            'METHODOLOGY_CHANGE', 'BOUNDARY_CHANGE',
            'EF_UPDATE', 'STRUCTURAL_CHANGE', 'ERROR_CORRECTION',
            'OTHER'
        )
    ),
    CONSTRAINT chk_bt_impact_pct_range CHECK (
        impact_pct IS NULL OR (impact_pct >= 0 AND impact_pct <= 100)
    )
);

-- Indexes for gl_iso14064_base_year_triggers
CREATE INDEX idx_iso14064_bt_org ON iso14064_app.gl_iso14064_base_year_triggers(org_id);
CREATE INDEX idx_iso14064_bt_type ON iso14064_app.gl_iso14064_base_year_triggers(trigger_type);
CREATE INDEX idx_iso14064_bt_recalc ON iso14064_app.gl_iso14064_base_year_triggers(requires_recalculation);
CREATE INDEX idx_iso14064_bt_triggered_at ON iso14064_app.gl_iso14064_base_year_triggers(triggered_at DESC);

-- =============================================================================
-- Table 11: iso14064_app.gl_iso14064_verification_records
-- =============================================================================
-- Verification engagements per ISO 14064-1:2018 Clause 9 and
-- ISO 14064-3:2019.  Tracks verifier details, accreditation,
-- verification level, scope, stage, and the assurance opinion.

CREATE TABLE iso14064_app.gl_iso14064_verification_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    inventory_id            UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_inventories(id) ON DELETE CASCADE,
    verifier_name           VARCHAR(500)    NOT NULL,
    verifier_accreditation  VARCHAR(200),
    verification_level      VARCHAR(50)     NOT NULL,
    scope_of_verification   TEXT,
    stage                   VARCHAR(50)     NOT NULL DEFAULT 'planning',
    opinion                 TEXT,
    opinion_date            DATE,
    findings_summary        JSONB           DEFAULT '[]',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_vr_verifier_not_empty CHECK (
        LENGTH(TRIM(verifier_name)) > 0
    ),
    CONSTRAINT chk_vr_verification_level CHECK (
        verification_level IN (
            'LIMITED_ASSURANCE', 'REASONABLE_ASSURANCE'
        )
    ),
    CONSTRAINT chk_vr_stage CHECK (
        stage IN (
            'planning', 'fieldwork', 'reporting',
            'opinion_issued', 'completed', 'cancelled'
        )
    )
);

-- Indexes for gl_iso14064_verification_records
CREATE INDEX idx_iso14064_vr_inventory ON iso14064_app.gl_iso14064_verification_records(inventory_id);
CREATE INDEX idx_iso14064_vr_level ON iso14064_app.gl_iso14064_verification_records(verification_level);
CREATE INDEX idx_iso14064_vr_stage ON iso14064_app.gl_iso14064_verification_records(stage);
CREATE INDEX idx_iso14064_vr_opinion_date ON iso14064_app.gl_iso14064_verification_records(opinion_date DESC);
CREATE INDEX idx_iso14064_vr_created_at ON iso14064_app.gl_iso14064_verification_records(created_at DESC);
CREATE INDEX idx_iso14064_vr_findings ON iso14064_app.gl_iso14064_verification_records USING GIN(findings_summary);

-- Updated_at trigger
CREATE TRIGGER trg_iso14064_vr_updated_at
    BEFORE UPDATE ON iso14064_app.gl_iso14064_verification_records
    FOR EACH ROW
    EXECUTE FUNCTION iso14064_app.set_updated_at();

-- =============================================================================
-- Table 12: iso14064_app.gl_iso14064_findings
-- =============================================================================
-- Verification findings (non-conformities, observations, opportunities
-- for improvement) from verification engagements.  Tracks severity,
-- affected category, emissions impact, and resolution status.

CREATE TABLE iso14064_app.gl_iso14064_findings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id         UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_verification_records(id) ON DELETE CASCADE,
    category                VARCHAR(50)     NOT NULL,
    severity                VARCHAR(20)     NOT NULL,
    description             TEXT            NOT NULL,
    affected_category       VARCHAR(50),
    emissions_impact_tco2e  DECIMAL(15,3),
    recommendation          TEXT,
    management_response     TEXT,
    status                  VARCHAR(50)     NOT NULL DEFAULT 'open',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    resolved_at             TIMESTAMPTZ,
    CONSTRAINT chk_f_category CHECK (
        category IN (
            'NON_CONFORMITY_MAJOR', 'NON_CONFORMITY_MINOR',
            'OBSERVATION', 'OPPORTUNITY_FOR_IMPROVEMENT'
        )
    ),
    CONSTRAINT chk_f_severity CHECK (
        severity IN ('CRITICAL', 'MAJOR', 'MINOR', 'INFO')
    ),
    CONSTRAINT chk_f_description_not_empty CHECK (
        LENGTH(TRIM(description)) > 0
    ),
    CONSTRAINT chk_f_status CHECK (
        status IN ('open', 'in_progress', 'resolved', 'accepted', 'rejected')
    ),
    CONSTRAINT chk_f_resolved_consistency CHECK (
        (status NOT IN ('resolved', 'accepted') AND resolved_at IS NULL)
        OR (status IN ('resolved', 'accepted') AND resolved_at IS NOT NULL)
        OR resolved_at IS NOT NULL
    )
);

-- Indexes for gl_iso14064_findings
CREATE INDEX idx_iso14064_f_verification ON iso14064_app.gl_iso14064_findings(verification_id);
CREATE INDEX idx_iso14064_f_category ON iso14064_app.gl_iso14064_findings(category);
CREATE INDEX idx_iso14064_f_severity ON iso14064_app.gl_iso14064_findings(severity);
CREATE INDEX idx_iso14064_f_status ON iso14064_app.gl_iso14064_findings(status);
CREATE INDEX idx_iso14064_f_created_at ON iso14064_app.gl_iso14064_findings(created_at DESC);

-- =============================================================================
-- Table 13: iso14064_app.gl_iso14064_reports
-- =============================================================================
-- Generated ISO 14064-1 GHG reports.  Each report covers a specific
-- inventory and must include the mandatory elements defined in
-- ISO 14064-1:2018 Clause 8 (reporting principles).  Tracks mandatory
-- element completeness and provenance.

CREATE TABLE iso14064_app.gl_iso14064_reports (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    inventory_id                UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_inventories(id) ON DELETE CASCADE,
    org_id                      UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_organizations(id) ON DELETE CASCADE,
    title                       VARCHAR(500)    NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    format                      VARCHAR(20)     NOT NULL,
    mandatory_elements          JSONB           DEFAULT '{}',
    mandatory_completeness_pct  DECIMAL(5,2)    DEFAULT 0,
    sections                    JSONB           DEFAULT '[]',
    file_path                   VARCHAR(1000),
    file_size_bytes             BIGINT,
    provenance_hash             CHAR(64),
    generated_by                VARCHAR(200),
    metadata                    JSONB           DEFAULT '{}',
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_r_title_not_empty CHECK (
        LENGTH(TRIM(title)) > 0
    ),
    CONSTRAINT chk_r_year_range CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_r_format CHECK (
        format IN ('json', 'csv', 'excel', 'pdf', 'xbrl')
    ),
    CONSTRAINT chk_r_completeness_range CHECK (
        mandatory_completeness_pct >= 0 AND mandatory_completeness_pct <= 100
    ),
    CONSTRAINT chk_r_file_size_non_neg CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    )
);

-- Indexes for gl_iso14064_reports
CREATE INDEX idx_iso14064_r_inventory ON iso14064_app.gl_iso14064_reports(inventory_id);
CREATE INDEX idx_iso14064_r_org ON iso14064_app.gl_iso14064_reports(org_id);
CREATE INDEX idx_iso14064_r_year ON iso14064_app.gl_iso14064_reports(reporting_year);
CREATE INDEX idx_iso14064_r_format ON iso14064_app.gl_iso14064_reports(format);
CREATE INDEX idx_iso14064_r_generated_at ON iso14064_app.gl_iso14064_reports(generated_at DESC);
CREATE INDEX idx_iso14064_r_mandatory ON iso14064_app.gl_iso14064_reports USING GIN(mandatory_elements);
CREATE INDEX idx_iso14064_r_sections ON iso14064_app.gl_iso14064_reports USING GIN(sections);

-- =============================================================================
-- Table 14: iso14064_app.gl_iso14064_management_actions
-- =============================================================================
-- Management plan actions for emission reduction and removal enhancement.
-- Tracks action category, target emission category, status, priority,
-- estimated reduction, cost, responsibility, and progress.

CREATE TABLE iso14064_app.gl_iso14064_management_actions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                      UUID            NOT NULL REFERENCES iso14064_app.gl_iso14064_organizations(id) ON DELETE CASCADE,
    title                       VARCHAR(500)    NOT NULL,
    description                 TEXT,
    action_category             VARCHAR(50)     NOT NULL,
    target_category             VARCHAR(50),
    status                      VARCHAR(50)     NOT NULL DEFAULT 'planned',
    priority                    VARCHAR(20)     NOT NULL DEFAULT 'medium',
    estimated_reduction_tco2e   DECIMAL(15,3),
    estimated_cost_usd          DECIMAL(15,2),
    responsible_person          VARCHAR(200),
    start_date                  DATE,
    target_date                 DATE,
    completion_date             DATE,
    progress_notes              JSONB           DEFAULT '[]',
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ma_title_not_empty CHECK (
        LENGTH(TRIM(title)) > 0
    ),
    CONSTRAINT chk_ma_action_category CHECK (
        action_category IN (
            'ENERGY_EFFICIENCY', 'RENEWABLE_ENERGY', 'PROCESS_CHANGE',
            'FUEL_SWITCHING', 'CARBON_CAPTURE', 'OFFSETTING',
            'SUPPLY_CHAIN', 'TRANSPORT', 'WASTE_REDUCTION',
            'REMOVAL_ENHANCEMENT', 'OTHER'
        )
    ),
    CONSTRAINT chk_ma_target_category CHECK (
        target_category IS NULL OR target_category IN (
            'direct',
            'indirect_energy',
            'indirect_transportation',
            'indirect_products',
            'indirect_other',
            'removals'
        )
    ),
    CONSTRAINT chk_ma_status CHECK (
        status IN ('planned', 'in_progress', 'completed', 'cancelled', 'on_hold')
    ),
    CONSTRAINT chk_ma_priority CHECK (
        priority IN ('critical', 'high', 'medium', 'low')
    ),
    CONSTRAINT chk_ma_reduction_non_neg CHECK (
        estimated_reduction_tco2e IS NULL OR estimated_reduction_tco2e >= 0
    ),
    CONSTRAINT chk_ma_cost_non_neg CHECK (
        estimated_cost_usd IS NULL OR estimated_cost_usd >= 0
    ),
    CONSTRAINT chk_ma_target_after_start CHECK (
        target_date IS NULL OR start_date IS NULL OR target_date >= start_date
    )
);

-- Indexes for gl_iso14064_management_actions
CREATE INDEX idx_iso14064_ma_org ON iso14064_app.gl_iso14064_management_actions(org_id);
CREATE INDEX idx_iso14064_ma_action_cat ON iso14064_app.gl_iso14064_management_actions(action_category);
CREATE INDEX idx_iso14064_ma_target_cat ON iso14064_app.gl_iso14064_management_actions(target_category);
CREATE INDEX idx_iso14064_ma_status ON iso14064_app.gl_iso14064_management_actions(status);
CREATE INDEX idx_iso14064_ma_priority ON iso14064_app.gl_iso14064_management_actions(priority);
CREATE INDEX idx_iso14064_ma_target_date ON iso14064_app.gl_iso14064_management_actions(target_date);
CREATE INDEX idx_iso14064_ma_created_at ON iso14064_app.gl_iso14064_management_actions(created_at DESC);
CREATE INDEX idx_iso14064_ma_progress ON iso14064_app.gl_iso14064_management_actions USING GIN(progress_notes);

-- Updated_at trigger
CREATE TRIGGER trg_iso14064_ma_updated_at
    BEFORE UPDATE ON iso14064_app.gl_iso14064_management_actions
    FOR EACH ROW
    EXECUTE FUNCTION iso14064_app.set_updated_at();

-- =============================================================================
-- Table 15: iso14064_app.gl_iso14064_emission_timeseries (hypertable)
-- =============================================================================
-- Time-series emission data for trend tracking and dashboard queries.
-- Partitioned by time for efficient time-range aggregation.  Fed by
-- the emission source calculation pipeline.

CREATE TABLE iso14064_app.gl_iso14064_emission_timeseries (
    time            TIMESTAMPTZ     NOT NULL,
    inventory_id    UUID            NOT NULL,
    category        VARCHAR(50)     NOT NULL,
    gas             VARCHAR(20)     NOT NULL DEFAULT 'CO2',
    tco2e           DECIMAL(15,6)   NOT NULL DEFAULT 0,
    biogenic_co2    DECIMAL(15,6)   DEFAULT 0,
    metadata        JSONB           DEFAULT '{}'
);

-- Convert to hypertable for time-series partitioning
SELECT create_hypertable('iso14064_app.gl_iso14064_emission_timeseries', 'time',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes for gl_iso14064_emission_timeseries (hypertable-aware)
CREATE INDEX idx_iso14064_ets_inventory ON iso14064_app.gl_iso14064_emission_timeseries(inventory_id, time DESC);
CREATE INDEX idx_iso14064_ets_category ON iso14064_app.gl_iso14064_emission_timeseries(category, time DESC);
CREATE INDEX idx_iso14064_ets_gas ON iso14064_app.gl_iso14064_emission_timeseries(gas, time DESC);

-- =============================================================================
-- Table 16: iso14064_app.gl_iso14064_verification_events (hypertable)
-- =============================================================================
-- Time-series verification event log for tracking the full verification
-- lifecycle.  Records stage transitions, document submissions, reviewer
-- actions, and findings at each verification milestone.

CREATE TABLE iso14064_app.gl_iso14064_verification_events (
    time            TIMESTAMPTZ     NOT NULL,
    verification_id UUID            NOT NULL,
    event_type      VARCHAR(50)     NOT NULL,
    stage           VARCHAR(50),
    actor           VARCHAR(200),
    details         JSONB           DEFAULT '{}'
);

-- Convert to hypertable
SELECT create_hypertable('iso14064_app.gl_iso14064_verification_events', 'time',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes for gl_iso14064_verification_events (hypertable-aware)
CREATE INDEX idx_iso14064_ve_verification ON iso14064_app.gl_iso14064_verification_events(verification_id, time DESC);
CREATE INDEX idx_iso14064_ve_event_type ON iso14064_app.gl_iso14064_verification_events(event_type, time DESC);
CREATE INDEX idx_iso14064_ve_stage ON iso14064_app.gl_iso14064_verification_events(stage, time DESC);
CREATE INDEX idx_iso14064_ve_actor ON iso14064_app.gl_iso14064_verification_events(actor, time DESC);
CREATE INDEX idx_iso14064_ve_details ON iso14064_app.gl_iso14064_verification_events USING GIN(details);

-- =============================================================================
-- Table 17: iso14064_app.gl_iso14064_audit_log (hypertable)
-- =============================================================================
-- Audit trail for all entity changes in the ISO 14064 application.
-- Records entity type, entity ID, action, actor identity, and change
-- details (old/new values).  Partitioned by time for efficient retention
-- and querying.  ISO 14064-1 requires maintaining complete audit trails.

CREATE TABLE iso14064_app.gl_iso14064_audit_log (
    time            TIMESTAMPTZ     NOT NULL,
    entity_type     VARCHAR(50)     NOT NULL,
    entity_id       UUID            NOT NULL,
    action          VARCHAR(50)     NOT NULL,
    actor           VARCHAR(200),
    old_value       JSONB           DEFAULT '{}',
    new_value       JSONB           DEFAULT '{}'
);

-- Convert to hypertable
SELECT create_hypertable('iso14064_app.gl_iso14064_audit_log', 'time',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes for gl_iso14064_audit_log (hypertable-aware)
CREATE INDEX idx_iso14064_al_entity ON iso14064_app.gl_iso14064_audit_log(entity_type, entity_id, time DESC);
CREATE INDEX idx_iso14064_al_action ON iso14064_app.gl_iso14064_audit_log(action, time DESC);
CREATE INDEX idx_iso14064_al_actor ON iso14064_app.gl_iso14064_audit_log(actor, time DESC);
CREATE INDEX idx_iso14064_al_entity_type ON iso14064_app.gl_iso14064_audit_log(entity_type, time DESC);
CREATE INDEX idx_iso14064_al_old_value ON iso14064_app.gl_iso14064_audit_log USING GIN(old_value);
CREATE INDEX idx_iso14064_al_new_value ON iso14064_app.gl_iso14064_audit_log USING GIN(new_value);

-- =============================================================================
-- Continuous Aggregate: iso14064_app.gl_iso14064_daily_emissions
-- =============================================================================
-- Precomputed daily emission aggregates by category for dashboard
-- time-series charts and trend analysis.

CREATE MATERIALIZED VIEW iso14064_app.gl_iso14064_daily_emissions
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time)      AS bucket,
    inventory_id,
    category,
    SUM(tco2e)                      AS total_tco2e,
    SUM(biogenic_co2)               AS total_biogenic_co2,
    COUNT(*)                        AS entry_count
FROM iso14064_app.gl_iso14064_emission_timeseries
GROUP BY bucket, inventory_id, category
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 3 hours
SELECT add_continuous_aggregate_policy('iso14064_app.gl_iso14064_daily_emissions',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Continuous Aggregate: iso14064_app.gl_iso14064_monthly_emissions
-- =============================================================================
-- Precomputed monthly emission aggregates by category for compliance
-- reporting and year-over-year trend analysis.

CREATE MATERIALIZED VIEW iso14064_app.gl_iso14064_monthly_emissions
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', time)    AS bucket,
    inventory_id,
    category,
    SUM(tco2e)                      AS total_tco2e,
    SUM(biogenic_co2)               AS total_biogenic_co2,
    COUNT(*)                        AS entry_count
FROM iso14064_app.gl_iso14064_emission_timeseries
GROUP BY bucket, inventory_id, category
WITH NO DATA;

-- Refresh policy: refresh every hour, covering the last 3 days
SELECT add_continuous_aggregate_policy('iso14064_app.gl_iso14064_monthly_emissions',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep emission timeseries for 3650 days (10 years, ISO 14064 best practice)
SELECT add_retention_policy('iso14064_app.gl_iso14064_emission_timeseries', INTERVAL '3650 days');

-- Keep verification events for 3650 days (10 years)
SELECT add_retention_policy('iso14064_app.gl_iso14064_verification_events', INTERVAL '3650 days');

-- Keep audit log for 3650 days (10 years, regulatory retention)
SELECT add_retention_policy('iso14064_app.gl_iso14064_audit_log', INTERVAL '3650 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on emission_timeseries after 90 days
ALTER TABLE iso14064_app.gl_iso14064_emission_timeseries SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('iso14064_app.gl_iso14064_emission_timeseries', INTERVAL '90 days');

-- Enable compression on verification_events after 90 days
ALTER TABLE iso14064_app.gl_iso14064_verification_events SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('iso14064_app.gl_iso14064_verification_events', INTERVAL '90 days');

-- Enable compression on audit_log after 90 days
ALTER TABLE iso14064_app.gl_iso14064_audit_log SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('iso14064_app.gl_iso14064_audit_log', INTERVAL '90 days');

-- =============================================================================
-- Seed: Default Application Settings
-- =============================================================================
-- Note: Settings are stored in the organization-level inventories and
-- boundaries.  This seed provides reference data for the significance
-- assessment thresholds and verification defaults.

-- No separate settings table needed; configuration is embedded in
-- inventory and boundary records per ISO 14064-1 structure.

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA iso14064_app TO greenlang_app;
GRANT ALL ON ALL TABLES IN SCHEMA iso14064_app TO greenlang_app;
GRANT ALL ON ALL SEQUENCES IN SCHEMA iso14064_app TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON iso14064_app.gl_iso14064_daily_emissions TO greenlang_app;
GRANT SELECT ON iso14064_app.gl_iso14064_monthly_emissions TO greenlang_app;

-- Read-only role
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA iso14064_app TO greenlang_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA iso14064_app TO greenlang_readonly;
        GRANT SELECT ON iso14064_app.gl_iso14064_daily_emissions TO greenlang_readonly;
        GRANT SELECT ON iso14064_app.gl_iso14064_monthly_emissions TO greenlang_readonly;
    END IF;
END
$$;

-- Add ISO 14064 app service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'iso14064_app:organizations:read', 'iso14064_app', 'organizations_read', 'View ISO 14064 organization profiles'),
    (gen_random_uuid(), 'iso14064_app:organizations:write', 'iso14064_app', 'organizations_write', 'Create and manage ISO 14064 organization profiles'),
    (gen_random_uuid(), 'iso14064_app:entities:read', 'iso14064_app', 'entities_read', 'View ISO 14064 reporting entities'),
    (gen_random_uuid(), 'iso14064_app:entities:write', 'iso14064_app', 'entities_write', 'Create and manage ISO 14064 reporting entities'),
    (gen_random_uuid(), 'iso14064_app:boundaries:read', 'iso14064_app', 'boundaries_read', 'View organizational and operational boundaries'),
    (gen_random_uuid(), 'iso14064_app:boundaries:write', 'iso14064_app', 'boundaries_write', 'Configure organizational and operational boundaries'),
    (gen_random_uuid(), 'iso14064_app:inventories:read', 'iso14064_app', 'inventories_read', 'View GHG inventories'),
    (gen_random_uuid(), 'iso14064_app:inventories:write', 'iso14064_app', 'inventories_write', 'Create and manage GHG inventories'),
    (gen_random_uuid(), 'iso14064_app:emissions:read', 'iso14064_app', 'emissions_read', 'View emission sources and calculations'),
    (gen_random_uuid(), 'iso14064_app:emissions:write', 'iso14064_app', 'emissions_write', 'Create and manage emission sources'),
    (gen_random_uuid(), 'iso14064_app:removals:read', 'iso14064_app', 'removals_read', 'View GHG removal sources'),
    (gen_random_uuid(), 'iso14064_app:removals:write', 'iso14064_app', 'removals_write', 'Create and manage GHG removal sources'),
    (gen_random_uuid(), 'iso14064_app:significance:read', 'iso14064_app', 'significance_read', 'View significance assessments'),
    (gen_random_uuid(), 'iso14064_app:significance:write', 'iso14064_app', 'significance_write', 'Create and manage significance assessments'),
    (gen_random_uuid(), 'iso14064_app:baseyear:read', 'iso14064_app', 'baseyear_read', 'View base year records and triggers'),
    (gen_random_uuid(), 'iso14064_app:baseyear:write', 'iso14064_app', 'baseyear_write', 'Create and manage base year records'),
    (gen_random_uuid(), 'iso14064_app:verification:read', 'iso14064_app', 'verification_read', 'View verification records and findings'),
    (gen_random_uuid(), 'iso14064_app:verification:write', 'iso14064_app', 'verification_write', 'Create and manage verification engagements'),
    (gen_random_uuid(), 'iso14064_app:reports:read', 'iso14064_app', 'reports_read', 'View generated ISO 14064 reports'),
    (gen_random_uuid(), 'iso14064_app:reports:write', 'iso14064_app', 'reports_write', 'Create and manage reports'),
    (gen_random_uuid(), 'iso14064_app:reports:generate', 'iso14064_app', 'reports_generate', 'Generate ISO 14064 compliance reports'),
    (gen_random_uuid(), 'iso14064_app:management:read', 'iso14064_app', 'management_read', 'View management actions and plans'),
    (gen_random_uuid(), 'iso14064_app:management:write', 'iso14064_app', 'management_write', 'Create and manage reduction actions'),
    (gen_random_uuid(), 'iso14064_app:crosswalk:read', 'iso14064_app', 'crosswalk_read', 'View framework crosswalk mappings'),
    (gen_random_uuid(), 'iso14064_app:crosswalk:generate', 'iso14064_app', 'crosswalk_generate', 'Generate framework crosswalk reports'),
    (gen_random_uuid(), 'iso14064_app:dashboard:read', 'iso14064_app', 'dashboard_read', 'View ISO 14064 dashboards and analytics'),
    (gen_random_uuid(), 'iso14064_app:audit:read', 'iso14064_app', 'audit_read', 'View ISO 14064 audit trail records'),
    (gen_random_uuid(), 'iso14064_app:admin', 'iso14064_app', 'admin', 'ISO 14064 application administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA iso14064_app IS 'GL-ISO14064-APP v1.0 Application Schema - ISO 14064-1:2018 GHG Compliance Platform with organization profiles, entity hierarchy, boundary management, emission/removal inventories, significance assessments, base year tracking, verification, reporting, and management actions';

COMMENT ON TABLE iso14064_app.gl_iso14064_organizations IS 'Organization profiles for ISO 14064-1 GHG compliance with company details, industry classification, and country';
COMMENT ON TABLE iso14064_app.gl_iso14064_entities IS 'Reporting entities (subsidiaries, facilities, operations) forming the organizational hierarchy with ownership percentages for equity share consolidation';
COMMENT ON TABLE iso14064_app.gl_iso14064_inventories IS 'Annual GHG inventories per ISO 14064-1 Clause 5 aggregating direct/indirect emissions, removals, biogenic CO2, and lifecycle status';
COMMENT ON TABLE iso14064_app.gl_iso14064_organizational_boundaries IS 'Organizational boundary configuration per ISO 14064-1 Clause 5.1 defining consolidation approach and included entities';
COMMENT ON TABLE iso14064_app.gl_iso14064_operational_boundaries IS 'Operational boundary configuration per ISO 14064-1 Clause 5.2 defining included/excluded emission categories with significance justification';
COMMENT ON TABLE iso14064_app.gl_iso14064_emission_sources IS 'Individual emission source entries with activity data, emission factors, GWP values, and calculated tCO2e per ISO 14064-1 Clause 5.2';
COMMENT ON TABLE iso14064_app.gl_iso14064_removal_sources IS 'GHG removal and sink entries with gross/credited removals, permanence assessment, and biogenic CO2 flows per ISO 14064-1 Clause 5.2';
COMMENT ON TABLE iso14064_app.gl_iso14064_significance_assessments IS 'Significance assessments per ISO 14064-1 Clause 5.3 determining category inclusion/exclusion based on weighted criteria scoring';
COMMENT ON TABLE iso14064_app.gl_iso14064_base_year_records IS 'Base year emissions records per ISO 14064-1 Clause 5.4 with original/recalculated values and recalculation policy';
COMMENT ON TABLE iso14064_app.gl_iso14064_base_year_triggers IS 'Base year recalculation triggers (mergers, acquisitions, methodology changes) with impact assessment per ISO 14064-1 Clause 5.4';
COMMENT ON TABLE iso14064_app.gl_iso14064_verification_records IS 'Verification engagements per ISO 14064-3 with verifier details, accreditation, assurance level, stage tracking, and opinion';
COMMENT ON TABLE iso14064_app.gl_iso14064_findings IS 'Verification findings (non-conformities, observations) with severity, emissions impact, management response, and resolution tracking';
COMMENT ON TABLE iso14064_app.gl_iso14064_reports IS 'Generated ISO 14064-1 reports with mandatory element completeness tracking per Clause 8 and provenance hash';
COMMENT ON TABLE iso14064_app.gl_iso14064_management_actions IS 'Management plan actions for emission reduction and removal enhancement with priority, cost, responsibility, and progress tracking';
COMMENT ON TABLE iso14064_app.gl_iso14064_emission_timeseries IS 'TimescaleDB hypertable: time-series emission data by inventory, category, and gas for trend tracking with 10-year retention';
COMMENT ON TABLE iso14064_app.gl_iso14064_verification_events IS 'TimescaleDB hypertable: verification event log tracking stage transitions, document submissions, and reviewer actions';
COMMENT ON TABLE iso14064_app.gl_iso14064_audit_log IS 'TimescaleDB hypertable: audit trail for all entity changes with old/new values and 10-year retention per ISO 14064 requirements';

COMMENT ON MATERIALIZED VIEW iso14064_app.gl_iso14064_daily_emissions IS 'Continuous aggregate: daily emission aggregates by inventory and category for dashboard time-series charts';
COMMENT ON MATERIALIZED VIEW iso14064_app.gl_iso14064_monthly_emissions IS 'Continuous aggregate: monthly emission aggregates by inventory and category for compliance reporting and year-over-year trends';

COMMENT ON COLUMN iso14064_app.gl_iso14064_inventories.consolidation_approach IS 'ISO 14064-1 consolidation: OPERATIONAL_CONTROL, FINANCIAL_CONTROL, EQUITY_SHARE';
COMMENT ON COLUMN iso14064_app.gl_iso14064_inventories.status IS 'Inventory lifecycle: draft, in_review, approved, verified, published';
COMMENT ON COLUMN iso14064_app.gl_iso14064_inventories.gwp_source IS 'Global Warming Potential source: IPCC AR5, IPCC AR6, etc.';
COMMENT ON COLUMN iso14064_app.gl_iso14064_emission_sources.category IS 'ISO 14064-1 emission category: direct, indirect_energy, indirect_transportation, indirect_products, indirect_other';
COMMENT ON COLUMN iso14064_app.gl_iso14064_emission_sources.gas IS 'Greenhouse gas: CO2, CH4, N2O, HFCs, PFCs, SF6, NF3, OTHER';
COMMENT ON COLUMN iso14064_app.gl_iso14064_emission_sources.data_quality_tier IS 'Data quality tier: TIER_1 (default EF), TIER_2 (country-specific), TIER_3 (facility-specific)';
COMMENT ON COLUMN iso14064_app.gl_iso14064_removal_sources.removal_type IS 'Removal type: AFFORESTATION, REFORESTATION, SOIL_CARBON, BIOENERGY_CCS, DIRECT_AIR_CAPTURE, ENHANCED_WEATHERING, BIOCHAR, OCEAN_BASED, OTHER';
COMMENT ON COLUMN iso14064_app.gl_iso14064_removal_sources.permanence_level IS 'Permanence level: PERMANENT, LONG_TERM, MEDIUM_TERM, SHORT_TERM';
COMMENT ON COLUMN iso14064_app.gl_iso14064_removal_sources.verification_status IS 'Removal verification status: unverified, pending, verified, rejected';
COMMENT ON COLUMN iso14064_app.gl_iso14064_significance_assessments.result IS 'Assessment result: significant, not_significant, pending';
COMMENT ON COLUMN iso14064_app.gl_iso14064_base_year_triggers.trigger_type IS 'Trigger type: MERGER, ACQUISITION, DIVESTITURE, METHODOLOGY_CHANGE, BOUNDARY_CHANGE, EF_UPDATE, STRUCTURAL_CHANGE, ERROR_CORRECTION, OTHER';
COMMENT ON COLUMN iso14064_app.gl_iso14064_verification_records.verification_level IS 'Assurance level per ISO 14064-3: LIMITED_ASSURANCE, REASONABLE_ASSURANCE';
COMMENT ON COLUMN iso14064_app.gl_iso14064_verification_records.stage IS 'Verification stage: planning, fieldwork, reporting, opinion_issued, completed, cancelled';
COMMENT ON COLUMN iso14064_app.gl_iso14064_findings.category IS 'Finding type: NON_CONFORMITY_MAJOR, NON_CONFORMITY_MINOR, OBSERVATION, OPPORTUNITY_FOR_IMPROVEMENT';
COMMENT ON COLUMN iso14064_app.gl_iso14064_findings.severity IS 'Finding severity: CRITICAL, MAJOR, MINOR, INFO';
COMMENT ON COLUMN iso14064_app.gl_iso14064_reports.format IS 'Report format: json, csv, excel, pdf, xbrl';
COMMENT ON COLUMN iso14064_app.gl_iso14064_management_actions.action_category IS 'Action type: ENERGY_EFFICIENCY, RENEWABLE_ENERGY, PROCESS_CHANGE, FUEL_SWITCHING, CARBON_CAPTURE, OFFSETTING, SUPPLY_CHAIN, TRANSPORT, WASTE_REDUCTION, REMOVAL_ENHANCEMENT, OTHER';
COMMENT ON COLUMN iso14064_app.gl_iso14064_management_actions.status IS 'Action status: planned, in_progress, completed, cancelled, on_hold';
COMMENT ON COLUMN iso14064_app.gl_iso14064_management_actions.priority IS 'Action priority: critical, high, medium, low';

-- =============================================================================
-- End of V084: GL-ISO14064-APP v1.0 Application Service Schema
-- =============================================================================
-- Summary:
--   14 tables created
--   3 hypertables (emission_timeseries, verification_events, audit_log)
--   2 continuous aggregates (daily_emissions, monthly_emissions)
--   10 update triggers
--   50+ B-tree indexes
--   12 GIN indexes on JSONB columns
--   3 retention policies (10-year retention)
--   3 compression policies (90-day threshold)
--   28 security permissions
--   Security grants for greenlang_app and greenlang_readonly
-- Previous: V083__ghg_app_service.sql
-- =============================================================================
