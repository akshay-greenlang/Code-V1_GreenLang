-- =============================================================================
-- V171: PACK-027 Enterprise Net Zero - Carbon Pricing
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    006 of 015
-- Date:         March 2026
--
-- Internal carbon pricing configuration with price trajectories and scope
-- coverage rules. Carbon liability tracking with fiscal year exposure,
-- cost allocation by department/BU, and regulatory carbon cost modeling
-- (ETS, CBAM, carbon tax).
--
-- Tables (2):
--   1. pack027_enterprise_net_zero.gl_carbon_prices
--   2. pack027_enterprise_net_zero.gl_carbon_liabilities
--
-- Previous: V170__PACK027_scenario_models.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_carbon_prices
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_carbon_prices (
    price_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Pricing configuration
    price_name                  VARCHAR(255)    NOT NULL,
    price_type                  VARCHAR(30)     NOT NULL DEFAULT 'SHADOW',
    price_usd_per_tco2e         DECIMAL(12,2)   NOT NULL,
    currency                    VARCHAR(3)      DEFAULT 'USD',
    effective_date              DATE            NOT NULL,
    expiry_date                 DATE,
    -- Escalation
    annual_escalation_pct       DECIMAL(6,2)    DEFAULT 0,
    escalation_method           VARCHAR(30)     DEFAULT 'LINEAR',
    price_floor                 DECIMAL(12,2),
    price_ceiling               DECIMAL(12,2),
    -- Scope coverage
    scope_coverage              TEXT[]          NOT NULL DEFAULT '{SCOPE_1,SCOPE_2}',
    scope3_categories_covered   TEXT[]          DEFAULT '{}',
    -- Application rules
    application_rules           JSONB           DEFAULT '{}',
    business_units_covered      TEXT[]          DEFAULT '{}',
    geographies_covered         TEXT[]          DEFAULT '{}',
    apply_to_capex              BOOLEAN         DEFAULT TRUE,
    apply_to_opex               BOOLEAN         DEFAULT TRUE,
    capex_threshold_usd         DECIMAL(18,2),
    -- Benchmarking
    benchmark_source            VARCHAR(100),
    benchmark_price             DECIMAL(12,2),
    benchmark_date              DATE,
    -- Regulatory reference
    regulatory_scheme           VARCHAR(50),
    eu_ets_linked               BOOLEAN         DEFAULT FALSE,
    cbam_linked                 BOOLEAN         DEFAULT FALSE,
    -- Governance
    approved_by                 VARCHAR(255),
    approval_date               DATE,
    review_frequency            VARCHAR(30)     DEFAULT 'ANNUAL',
    next_review_date            DATE,
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_cp_price_type CHECK (
        price_type IN ('SHADOW', 'INTERNAL_FEE', 'IMPLICIT', 'EXPLICIT',
                        'REGULATORY', 'VOLUNTARY', 'OFFSET_PRICE')
    ),
    CONSTRAINT chk_p027_cp_price_positive CHECK (
        price_usd_per_tco2e > 0
    ),
    CONSTRAINT chk_p027_cp_escalation_method CHECK (
        escalation_method IN ('LINEAR', 'EXPONENTIAL', 'STEPPED', 'MARKET_LINKED', 'FIXED')
    ),
    CONSTRAINT chk_p027_cp_floor_ceiling CHECK (
        price_floor IS NULL OR price_ceiling IS NULL OR price_floor <= price_ceiling
    ),
    CONSTRAINT chk_p027_cp_date_order CHECK (
        expiry_date IS NULL OR expiry_date > effective_date
    ),
    CONSTRAINT chk_p027_cp_review_frequency CHECK (
        review_frequency IN ('QUARTERLY', 'SEMI_ANNUAL', 'ANNUAL', 'BIENNIAL')
    ),
    CONSTRAINT chk_p027_cp_regulatory CHECK (
        regulatory_scheme IS NULL OR regulatory_scheme IN (
            'EU_ETS', 'UK_ETS', 'CBAM', 'CA_CAP_AND_TRADE', 'RGGI',
            'CHINA_ETS', 'KOREA_ETS', 'CARBON_TAX', 'OTHER'
        )
    )
);

-- =============================================================================
-- Table 2: pack027_enterprise_net_zero.gl_carbon_liabilities
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_carbon_liabilities (
    liability_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    price_id                    UUID            REFERENCES pack027_enterprise_net_zero.gl_carbon_prices(price_id) ON DELETE SET NULL,
    -- Period
    fiscal_year                 INTEGER         NOT NULL,
    fiscal_quarter              INTEGER,
    -- Emissions volume
    total_tco2e                 DECIMAL(18,4)   NOT NULL,
    scope1_tco2e                DECIMAL(18,4)   DEFAULT 0,
    scope2_tco2e                DECIMAL(18,4)   DEFAULT 0,
    scope3_tco2e                DECIMAL(18,4)   DEFAULT 0,
    -- Cost
    carbon_price_applied        DECIMAL(12,2)   NOT NULL,
    carbon_cost_usd             DECIMAL(18,2)   NOT NULL,
    carbon_cost_local           DECIMAL(18,2),
    local_currency              VARCHAR(3),
    exchange_rate               DECIMAL(12,6),
    -- Allocation
    allocation_by_dept          JSONB           DEFAULT '{}',
    allocation_by_bu            JSONB           DEFAULT '{}',
    allocation_by_product       JSONB           DEFAULT '{}',
    allocation_by_geography     JSONB           DEFAULT '{}',
    -- P&L impact
    cogs_carbon_cost_usd        DECIMAL(18,2),
    sga_carbon_cost_usd         DECIMAL(18,2),
    ebitda_impact_pct           DECIMAL(8,4),
    revenue_intensity_usd       DECIMAL(18,8),
    -- Regulatory costs
    ets_compliance_cost_usd     DECIMAL(18,2),
    cbam_exposure_usd           DECIMAL(18,2),
    carbon_tax_cost_usd         DECIMAL(18,2),
    -- Abatement comparison
    abatement_cost_usd          DECIMAL(18,2),
    abatement_vs_liability      VARCHAR(30),
    -- Status
    status                      VARCHAR(30)     DEFAULT 'ESTIMATED',
    approved                    BOOLEAN         DEFAULT FALSE,
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_cl_fiscal_year CHECK (
        fiscal_year >= 2020 AND fiscal_year <= 2100
    ),
    CONSTRAINT chk_p027_cl_fiscal_quarter CHECK (
        fiscal_quarter IS NULL OR (fiscal_quarter >= 1 AND fiscal_quarter <= 4)
    ),
    CONSTRAINT chk_p027_cl_total_non_neg CHECK (
        total_tco2e >= 0
    ),
    CONSTRAINT chk_p027_cl_cost_non_neg CHECK (
        carbon_cost_usd >= 0
    ),
    CONSTRAINT chk_p027_cl_status CHECK (
        status IN ('ESTIMATED', 'CALCULATED', 'REVIEWED', 'APPROVED', 'ACCRUED', 'PAID')
    ),
    CONSTRAINT chk_p027_cl_abatement_compare CHECK (
        abatement_vs_liability IS NULL OR abatement_vs_liability IN (
            'CHEAPER_TO_ABATE', 'CHEAPER_TO_PAY', 'BREAK_EVEN', 'NOT_CALCULATED'
        )
    ),
    CONSTRAINT uq_p027_cl_company_year_quarter UNIQUE (company_id, fiscal_year, fiscal_quarter)
);

-- ---------------------------------------------------------------------------
-- Indexes for gl_carbon_prices
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_cp_company            ON pack027_enterprise_net_zero.gl_carbon_prices(company_id);
CREATE INDEX idx_p027_cp_tenant             ON pack027_enterprise_net_zero.gl_carbon_prices(tenant_id);
CREATE INDEX idx_p027_cp_type               ON pack027_enterprise_net_zero.gl_carbon_prices(price_type);
CREATE INDEX idx_p027_cp_effective          ON pack027_enterprise_net_zero.gl_carbon_prices(effective_date);
CREATE INDEX idx_p027_cp_active             ON pack027_enterprise_net_zero.gl_carbon_prices(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p027_cp_regulatory         ON pack027_enterprise_net_zero.gl_carbon_prices(regulatory_scheme);
CREATE INDEX idx_p027_cp_ets               ON pack027_enterprise_net_zero.gl_carbon_prices(eu_ets_linked) WHERE eu_ets_linked = TRUE;
CREATE INDEX idx_p027_cp_cbam              ON pack027_enterprise_net_zero.gl_carbon_prices(cbam_linked) WHERE cbam_linked = TRUE;
CREATE INDEX idx_p027_cp_scope              ON pack027_enterprise_net_zero.gl_carbon_prices USING GIN(scope_coverage);
CREATE INDEX idx_p027_cp_rules              ON pack027_enterprise_net_zero.gl_carbon_prices USING GIN(application_rules);
CREATE INDEX idx_p027_cp_created            ON pack027_enterprise_net_zero.gl_carbon_prices(created_at DESC);
CREATE INDEX idx_p027_cp_metadata           ON pack027_enterprise_net_zero.gl_carbon_prices USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Indexes for gl_carbon_liabilities
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_cl_company            ON pack027_enterprise_net_zero.gl_carbon_liabilities(company_id);
CREATE INDEX idx_p027_cl_tenant             ON pack027_enterprise_net_zero.gl_carbon_liabilities(tenant_id);
CREATE INDEX idx_p027_cl_price              ON pack027_enterprise_net_zero.gl_carbon_liabilities(price_id);
CREATE INDEX idx_p027_cl_fiscal_year        ON pack027_enterprise_net_zero.gl_carbon_liabilities(fiscal_year);
CREATE INDEX idx_p027_cl_year_quarter       ON pack027_enterprise_net_zero.gl_carbon_liabilities(fiscal_year, fiscal_quarter);
CREATE INDEX idx_p027_cl_status             ON pack027_enterprise_net_zero.gl_carbon_liabilities(status);
CREATE INDEX idx_p027_cl_approved           ON pack027_enterprise_net_zero.gl_carbon_liabilities(approved);
CREATE INDEX idx_p027_cl_ebitda             ON pack027_enterprise_net_zero.gl_carbon_liabilities(ebitda_impact_pct);
CREATE INDEX idx_p027_cl_allocation_dept    ON pack027_enterprise_net_zero.gl_carbon_liabilities USING GIN(allocation_by_dept);
CREATE INDEX idx_p027_cl_allocation_bu      ON pack027_enterprise_net_zero.gl_carbon_liabilities USING GIN(allocation_by_bu);
CREATE INDEX idx_p027_cl_created            ON pack027_enterprise_net_zero.gl_carbon_liabilities(created_at DESC);
CREATE INDEX idx_p027_cl_metadata           ON pack027_enterprise_net_zero.gl_carbon_liabilities USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_carbon_prices_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_carbon_prices
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p027_carbon_liabilities_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_carbon_liabilities
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_carbon_prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack027_enterprise_net_zero.gl_carbon_liabilities ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_cp_tenant_isolation
    ON pack027_enterprise_net_zero.gl_carbon_prices
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_cp_service_bypass
    ON pack027_enterprise_net_zero.gl_carbon_prices
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p027_cl_tenant_isolation
    ON pack027_enterprise_net_zero.gl_carbon_liabilities
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_cl_service_bypass
    ON pack027_enterprise_net_zero.gl_carbon_liabilities
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_carbon_prices TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_carbon_liabilities TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_carbon_prices IS
    'Internal carbon pricing configuration with price trajectories, scope coverage rules, escalation mechanisms, and regulatory scheme linking.';
COMMENT ON TABLE pack027_enterprise_net_zero.gl_carbon_liabilities IS
    'Carbon liability tracking with fiscal year exposure, cost allocation by department/BU/product, P&L impact, and regulatory cost modeling.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_prices.price_id IS 'Unique carbon price configuration identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_prices.price_type IS 'Pricing mechanism: SHADOW (internal decision-making), INTERNAL_FEE (actual charge), REGULATORY (compliance), etc.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_prices.price_usd_per_tco2e IS 'Carbon price in USD per tonne CO2e.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_prices.scope_coverage IS 'Array of scopes this price applies to (SCOPE_1, SCOPE_2, SCOPE_3).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_prices.application_rules IS 'JSONB rules for when and how to apply the carbon price (CapEx thresholds, BU rules, etc.).';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_liabilities.liability_id IS 'Unique carbon liability record identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_liabilities.carbon_cost_usd IS 'Total carbon cost in USD for the period (emissions x price).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_liabilities.allocation_by_dept IS 'JSONB breakdown of carbon cost by department.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_liabilities.ebitda_impact_pct IS 'Carbon cost as percentage of EBITDA.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_liabilities.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
