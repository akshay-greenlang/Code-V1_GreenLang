-- =============================================================================
-- V173: PACK-027 Enterprise Net Zero - Supply Chain Mapping
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    008 of 015
-- Date:         March 2026
--
-- Multi-tier supplier mapping (Tier 1-5) with 100,000+ supplier engagement
-- tracking. Supplier tiering by emissions contribution, engagement program
-- management, CDP/EcoVadis integration, and data quality improvement tracking.
--
-- Tables (2):
--   1. pack027_enterprise_net_zero.gl_supply_chain_tiers
--   2. pack027_enterprise_net_zero.gl_supplier_engagement
--
-- Previous: V172__PACK027_scope4_projects.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_supply_chain_tiers
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_supply_chain_tiers (
    tier_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Tier definition
    tier_level                  INTEGER         NOT NULL,
    tier_name                   VARCHAR(100),
    reporting_year              INTEGER         NOT NULL,
    -- Supplier statistics
    supplier_count              INTEGER         NOT NULL DEFAULT 0,
    supplier_count_engaged      INTEGER         DEFAULT 0,
    supplier_count_sbti         INTEGER         DEFAULT 0,
    supplier_count_cdp          INTEGER         DEFAULT 0,
    -- Spend
    total_spend_usd             DECIMAL(18,2)   DEFAULT 0,
    spend_with_disclosed        DECIMAL(18,2)   DEFAULT 0,
    spend_coverage_pct          DECIMAL(6,2),
    -- Emissions
    emissions_tco2e             DECIMAL(18,4)   NOT NULL DEFAULT 0,
    emissions_pct_of_scope3     DECIMAL(6,2),
    emissions_method            VARCHAR(30)     DEFAULT 'HYBRID',
    -- Data quality
    avg_data_quality_score      DECIMAL(3,1),
    supplier_specific_pct       DECIMAL(6,2),
    average_data_pct            DECIMAL(6,2),
    spend_based_pct             DECIMAL(6,2),
    -- Hotspot analysis
    top_categories              JSONB           DEFAULT '{}',
    top_geographies             JSONB           DEFAULT '{}',
    top_commodities             JSONB           DEFAULT '{}',
    -- Engagement targets
    engagement_target_pct       DECIMAL(6,2),
    engagement_actual_pct       DECIMAL(6,2),
    sbti_adoption_target_pct    DECIMAL(6,2),
    sbti_adoption_actual_pct    DECIMAL(6,2),
    -- Year-over-year
    yoy_emissions_change_pct    DECIMAL(8,2),
    yoy_engagement_change_pct   DECIMAL(8,2),
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_sct_tier_level CHECK (
        tier_level >= 1 AND tier_level <= 5
    ),
    CONSTRAINT chk_p027_sct_reporting_year CHECK (
        reporting_year >= 2015 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p027_sct_supplier_count CHECK (
        supplier_count >= 0
    ),
    CONSTRAINT chk_p027_sct_spend_non_neg CHECK (
        total_spend_usd >= 0
    ),
    CONSTRAINT chk_p027_sct_emissions_non_neg CHECK (
        emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p027_sct_emissions_method CHECK (
        emissions_method IN ('SUPPLIER_SPECIFIC', 'HYBRID', 'AVERAGE_DATA', 'SPEND_BASED', 'MIXED')
    ),
    CONSTRAINT chk_p027_sct_dq_score CHECK (
        avg_data_quality_score IS NULL OR (avg_data_quality_score >= 1.0 AND avg_data_quality_score <= 5.0)
    ),
    CONSTRAINT chk_p027_sct_pct_ranges CHECK (
        (spend_coverage_pct IS NULL OR (spend_coverage_pct >= 0 AND spend_coverage_pct <= 100)) AND
        (engagement_target_pct IS NULL OR (engagement_target_pct >= 0 AND engagement_target_pct <= 100)) AND
        (engagement_actual_pct IS NULL OR (engagement_actual_pct >= 0 AND engagement_actual_pct <= 100))
    ),
    CONSTRAINT uq_p027_sct_company_tier_year UNIQUE (company_id, tier_level, reporting_year)
);

-- =============================================================================
-- Table 2: pack027_enterprise_net_zero.gl_supplier_engagement
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_supplier_engagement (
    engagement_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    -- Supplier identification
    supplier_id                 UUID            NOT NULL,
    supplier_name               VARCHAR(500)    NOT NULL,
    supplier_country            VARCHAR(3),
    supplier_sector             VARCHAR(100),
    tier_level                  INTEGER         DEFAULT 1,
    -- Spend and ranking
    annual_spend_usd            DECIMAL(18,2),
    spend_rank                  INTEGER,
    emissions_rank              INTEGER,
    -- Engagement
    engagement_tier             VARCHAR(30)     NOT NULL DEFAULT 'INFORM',
    engagement_start_date       DATE,
    last_engagement_date        DATE,
    engagement_score            DECIMAL(5,2),
    -- Questionnaire
    questionnaire_sent          BOOLEAN         DEFAULT FALSE,
    questionnaire_sent_date     DATE,
    response_status             VARCHAR(30)     DEFAULT 'NOT_SENT',
    response_date               DATE,
    -- Emissions disclosure
    emissions_disclosed         BOOLEAN         DEFAULT FALSE,
    disclosed_scope1_tco2e      DECIMAL(18,4),
    disclosed_scope2_tco2e      DECIMAL(18,4),
    disclosed_total_tco2e       DECIMAL(18,4),
    estimated_tco2e             DECIMAL(18,4),
    estimation_method           VARCHAR(30),
    data_quality_level          INTEGER,
    -- SBTi and CDP status
    has_sbti_target             BOOLEAN         DEFAULT FALSE,
    sbti_status                 VARCHAR(30),
    cdp_score                   VARCHAR(5),
    cdp_response_year           INTEGER,
    ecovadis_score              INTEGER,
    -- Reduction tracking
    base_year_emissions_tco2e   DECIMAL(18,4),
    current_year_emissions_tco2e DECIMAL(18,4),
    reduction_from_base_pct     DECIMAL(8,2),
    reduction_target_pct        DECIMAL(6,2),
    -- Engagement actions
    actions_requested           JSONB           DEFAULT '{}',
    actions_completed           JSONB           DEFAULT '{}',
    improvement_plan            TEXT,
    next_review_date            DATE,
    -- Risk flags
    high_risk_flag              BOOLEAN         DEFAULT FALSE,
    risk_factors                TEXT[]          DEFAULT '{}',
    -- Metadata
    external_references         JSONB           DEFAULT '{}',
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_se_tier_level CHECK (
        tier_level >= 1 AND tier_level <= 5
    ),
    CONSTRAINT chk_p027_se_engagement_tier CHECK (
        engagement_tier IN ('INFORM', 'ENGAGE', 'REQUIRE', 'COLLABORATE', 'STRATEGIC')
    ),
    CONSTRAINT chk_p027_se_response_status CHECK (
        response_status IN ('NOT_SENT', 'SENT', 'OPENED', 'IN_PROGRESS', 'SUBMITTED',
                             'INCOMPLETE', 'OVERDUE', 'DECLINED', 'NOT_APPLICABLE')
    ),
    CONSTRAINT chk_p027_se_estimation_method CHECK (
        estimation_method IS NULL OR estimation_method IN (
            'SUPPLIER_SPECIFIC', 'HYBRID', 'AVERAGE_DATA', 'SPEND_BASED', 'PROXY'
        )
    ),
    CONSTRAINT chk_p027_se_dq_level CHECK (
        data_quality_level IS NULL OR (data_quality_level >= 1 AND data_quality_level <= 5)
    ),
    CONSTRAINT chk_p027_se_sbti_status CHECK (
        sbti_status IS NULL OR sbti_status IN (
            'NO_TARGET', 'COMMITTED', 'NEAR_TERM_VALIDATED', 'NET_ZERO_VALIDATED', 'TARGETS_SET'
        )
    ),
    CONSTRAINT chk_p027_se_ecovadis CHECK (
        ecovadis_score IS NULL OR (ecovadis_score >= 0 AND ecovadis_score <= 100)
    ),
    CONSTRAINT chk_p027_se_spend_non_neg CHECK (
        annual_spend_usd IS NULL OR annual_spend_usd >= 0
    ),
    CONSTRAINT chk_p027_se_engagement_score CHECK (
        engagement_score IS NULL OR (engagement_score >= 0 AND engagement_score <= 100)
    ),
    CONSTRAINT uq_p027_se_company_supplier UNIQUE (company_id, supplier_id)
);

-- ---------------------------------------------------------------------------
-- Indexes for gl_supply_chain_tiers
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_sct_company           ON pack027_enterprise_net_zero.gl_supply_chain_tiers(company_id);
CREATE INDEX idx_p027_sct_tenant            ON pack027_enterprise_net_zero.gl_supply_chain_tiers(tenant_id);
CREATE INDEX idx_p027_sct_tier              ON pack027_enterprise_net_zero.gl_supply_chain_tiers(tier_level);
CREATE INDEX idx_p027_sct_year              ON pack027_enterprise_net_zero.gl_supply_chain_tiers(reporting_year);
CREATE INDEX idx_p027_sct_company_tier      ON pack027_enterprise_net_zero.gl_supply_chain_tiers(company_id, tier_level);
CREATE INDEX idx_p027_sct_emissions         ON pack027_enterprise_net_zero.gl_supply_chain_tiers(emissions_tco2e DESC);
CREATE INDEX idx_p027_sct_method            ON pack027_enterprise_net_zero.gl_supply_chain_tiers(emissions_method);
CREATE INDEX idx_p027_sct_created           ON pack027_enterprise_net_zero.gl_supply_chain_tiers(created_at DESC);
CREATE INDEX idx_p027_sct_categories        ON pack027_enterprise_net_zero.gl_supply_chain_tiers USING GIN(top_categories);
CREATE INDEX idx_p027_sct_metadata          ON pack027_enterprise_net_zero.gl_supply_chain_tiers USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Indexes for gl_supplier_engagement
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_se_tenant             ON pack027_enterprise_net_zero.gl_supplier_engagement(tenant_id);
CREATE INDEX idx_p027_se_company            ON pack027_enterprise_net_zero.gl_supplier_engagement(company_id);
CREATE INDEX idx_p027_se_supplier           ON pack027_enterprise_net_zero.gl_supplier_engagement(supplier_id);
CREATE INDEX idx_p027_se_tier               ON pack027_enterprise_net_zero.gl_supplier_engagement(tier_level);
CREATE INDEX idx_p027_se_engagement_tier    ON pack027_enterprise_net_zero.gl_supplier_engagement(engagement_tier);
CREATE INDEX idx_p027_se_response           ON pack027_enterprise_net_zero.gl_supplier_engagement(response_status);
CREATE INDEX idx_p027_se_disclosed          ON pack027_enterprise_net_zero.gl_supplier_engagement(emissions_disclosed) WHERE emissions_disclosed = TRUE;
CREATE INDEX idx_p027_se_sbti               ON pack027_enterprise_net_zero.gl_supplier_engagement(has_sbti_target) WHERE has_sbti_target = TRUE;
CREATE INDEX idx_p027_se_sbti_status        ON pack027_enterprise_net_zero.gl_supplier_engagement(sbti_status);
CREATE INDEX idx_p027_se_cdp               ON pack027_enterprise_net_zero.gl_supplier_engagement(cdp_score);
CREATE INDEX idx_p027_se_spend_rank         ON pack027_enterprise_net_zero.gl_supplier_engagement(spend_rank);
CREATE INDEX idx_p027_se_emissions_rank     ON pack027_enterprise_net_zero.gl_supplier_engagement(emissions_rank);
CREATE INDEX idx_p027_se_high_risk          ON pack027_enterprise_net_zero.gl_supplier_engagement(high_risk_flag) WHERE high_risk_flag = TRUE;
CREATE INDEX idx_p027_se_country            ON pack027_enterprise_net_zero.gl_supplier_engagement(supplier_country);
CREATE INDEX idx_p027_se_sector             ON pack027_enterprise_net_zero.gl_supplier_engagement(supplier_sector);
CREATE INDEX idx_p027_se_next_review        ON pack027_enterprise_net_zero.gl_supplier_engagement(next_review_date);
CREATE INDEX idx_p027_se_created            ON pack027_enterprise_net_zero.gl_supplier_engagement(created_at DESC);
CREATE INDEX idx_p027_se_metadata           ON pack027_enterprise_net_zero.gl_supplier_engagement USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_supply_chain_tiers_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_supply_chain_tiers
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p027_supplier_engagement_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_supplier_engagement
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_supply_chain_tiers ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack027_enterprise_net_zero.gl_supplier_engagement ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_sct_tenant_isolation
    ON pack027_enterprise_net_zero.gl_supply_chain_tiers
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_sct_service_bypass
    ON pack027_enterprise_net_zero.gl_supply_chain_tiers
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p027_se_tenant_isolation
    ON pack027_enterprise_net_zero.gl_supplier_engagement
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_se_service_bypass
    ON pack027_enterprise_net_zero.gl_supplier_engagement
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_supply_chain_tiers TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_supplier_engagement TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_supply_chain_tiers IS
    'Multi-tier supplier mapping (Tier 1-5) with aggregated spend, emissions, engagement statistics, and data quality tracking per tier.';
COMMENT ON TABLE pack027_enterprise_net_zero.gl_supplier_engagement IS
    'Individual supplier engagement records with questionnaire tracking, emissions disclosure, SBTi/CDP/EcoVadis status, and reduction progress.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_supply_chain_tiers.tier_id IS 'Unique supply chain tier record identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_supply_chain_tiers.tier_level IS 'Supply chain tier (1=direct suppliers, 2-5=indirect suppliers).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_supply_chain_tiers.emissions_tco2e IS 'Total emissions in tCO2e for this tier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_supply_chain_tiers.avg_data_quality_score IS 'Average data quality score (1-5) across suppliers in this tier.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_supplier_engagement.engagement_id IS 'Unique supplier engagement record identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_supplier_engagement.engagement_tier IS 'Engagement level: INFORM, ENGAGE, REQUIRE, COLLABORATE, STRATEGIC.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_supplier_engagement.response_status IS 'Questionnaire response status: NOT_SENT through SUBMITTED or DECLINED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_supplier_engagement.emissions_disclosed IS 'Whether the supplier has disclosed their emissions data.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_supplier_engagement.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
