-- =============================================================================
-- V174: PACK-027 Enterprise Net Zero - Financial Integration
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    009 of 015
-- Date:         March 2026
--
-- Carbon-adjusted financial data integration with P&L line item allocation,
-- EBITDA carbon intensity, carbon asset tracking (allowances, credits, PPAs),
-- and CBAM exposure modeling. Supports ESRS E1-8/E1-9 disclosures.
--
-- Tables (2):
--   1. pack027_enterprise_net_zero.gl_carbon_pl_allocation
--   2. pack027_enterprise_net_zero.gl_carbon_assets
--
-- Previous: V173__PACK027_supply_chain_mapping.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_carbon_pl_allocation
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_carbon_pl_allocation (
    allocation_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Period
    fiscal_year                 INTEGER         NOT NULL,
    fiscal_quarter              INTEGER,
    -- Product/BU allocation
    product_line                VARCHAR(255)    NOT NULL,
    business_unit               VARCHAR(255),
    cost_center                 VARCHAR(100),
    -- Revenue
    revenue_usd                 DECIMAL(18,2),
    units_sold                  DECIMAL(18,2),
    unit_type                   VARCHAR(50),
    -- Carbon cost allocation
    carbon_cost_usd             DECIMAL(18,2)   NOT NULL,
    carbon_cost_per_unit        DECIMAL(18,6),
    carbon_cost_per_revenue     DECIMAL(18,8),
    -- P&L impact
    cogs_allocation_usd         DECIMAL(18,2),
    sga_allocation_usd          DECIMAL(18,2),
    rnd_allocation_usd          DECIMAL(18,2),
    total_pl_impact_usd         DECIMAL(18,2),
    -- EBITDA impact
    ebitda_before_carbon        DECIMAL(18,2),
    ebitda_after_carbon         DECIMAL(18,2),
    ebitda_impact_pct           DECIMAL(8,4),
    -- Emissions
    allocated_tco2e             DECIMAL(18,4)   NOT NULL,
    scope1_allocated_tco2e      DECIMAL(18,4)   DEFAULT 0,
    scope2_allocated_tco2e      DECIMAL(18,4)   DEFAULT 0,
    scope3_allocated_tco2e      DECIMAL(18,4)   DEFAULT 0,
    -- Carbon intensity
    carbon_intensity_per_unit   DECIMAL(18,8),
    carbon_intensity_per_rev    DECIMAL(18,8),
    -- CBAM exposure
    cbam_applicable             BOOLEAN         DEFAULT FALSE,
    cbam_exposure_usd           DECIMAL(18,2),
    cbam_product_category       VARCHAR(100),
    cbam_embedded_emissions     DECIMAL(18,4),
    -- Year-over-year
    yoy_carbon_cost_change_pct  DECIMAL(8,2),
    yoy_intensity_change_pct    DECIMAL(8,2),
    -- Allocation method
    allocation_method           VARCHAR(30)     DEFAULT 'ACTIVITY_BASED',
    allocation_basis            TEXT,
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_cpl_fiscal_year CHECK (
        fiscal_year >= 2020 AND fiscal_year <= 2100
    ),
    CONSTRAINT chk_p027_cpl_fiscal_quarter CHECK (
        fiscal_quarter IS NULL OR (fiscal_quarter >= 1 AND fiscal_quarter <= 4)
    ),
    CONSTRAINT chk_p027_cpl_carbon_cost_non_neg CHECK (
        carbon_cost_usd >= 0
    ),
    CONSTRAINT chk_p027_cpl_emissions_non_neg CHECK (
        allocated_tco2e >= 0
    ),
    CONSTRAINT chk_p027_cpl_allocation_method CHECK (
        allocation_method IN ('ACTIVITY_BASED', 'REVENUE_BASED', 'PHYSICAL_BASED',
                               'HEADCOUNT_BASED', 'AREA_BASED', 'HYBRID', 'DIRECT')
    ),
    CONSTRAINT uq_p027_cpl_company_year_product UNIQUE (company_id, fiscal_year, fiscal_quarter, product_line)
);

-- =============================================================================
-- Table 2: pack027_enterprise_net_zero.gl_carbon_assets
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_carbon_assets (
    asset_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Asset identification
    asset_name                  VARCHAR(255)    NOT NULL,
    asset_type                  VARCHAR(50)     NOT NULL,
    asset_subtype               VARCHAR(50),
    -- Quantity and vintage
    quantity                    DECIMAL(18,4)   NOT NULL,
    unit                        VARCHAR(20)     NOT NULL DEFAULT 'tCO2e',
    vintage                     INTEGER,
    vintage_range               VARCHAR(20),
    -- Valuation
    acquisition_cost_usd        DECIMAL(18,2),
    current_value_usd           DECIMAL(18,2),
    fair_value_date             DATE,
    -- Counterparty
    issuer                      VARCHAR(255),
    registry                    VARCHAR(100),
    registry_serial             VARCHAR(255),
    project_id_external         VARCHAR(255),
    -- Status
    retirement_status           VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    retirement_date             DATE,
    retirement_purpose          VARCHAR(100),
    retirement_year_applied     INTEGER,
    -- Contract terms
    contract_start_date         DATE,
    contract_end_date           DATE,
    delivery_schedule           JSONB           DEFAULT '{}',
    -- Quality criteria
    methodology                 VARCHAR(100),
    additionality_verified      BOOLEAN         DEFAULT FALSE,
    permanence_years            INTEGER,
    co_benefits                 TEXT[]          DEFAULT '{}',
    sdg_alignment               TEXT[]          DEFAULT '{}',
    -- Accounting treatment
    accounting_classification   VARCHAR(50),
    balance_sheet_line          VARCHAR(100),
    amortization_method         VARCHAR(30),
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_ca_asset_type CHECK (
        asset_type IN ('ETS_ALLOWANCE', 'CARBON_CREDIT', 'REMOVAL_CREDIT', 'PPA',
                        'REC', 'GO', 'I_REC', 'VOLUNTARY_CREDIT', 'OFFSET',
                        'INSET', 'GREEN_TARIFF', 'OTHER')
    ),
    CONSTRAINT chk_p027_ca_quantity_positive CHECK (
        quantity > 0
    ),
    CONSTRAINT chk_p027_ca_vintage CHECK (
        vintage IS NULL OR (vintage >= 2010 AND vintage <= 2100)
    ),
    CONSTRAINT chk_p027_ca_retirement_status CHECK (
        retirement_status IN ('ACTIVE', 'RETIRED', 'EXPIRED', 'TRANSFERRED', 'CANCELLED', 'PENDING')
    ),
    CONSTRAINT chk_p027_ca_accounting CHECK (
        accounting_classification IS NULL OR accounting_classification IN (
            'INTANGIBLE_ASSET', 'FINANCIAL_ASSET', 'PREPAID_EXPENSE',
            'INVENTORY', 'RIGHT_OF_USE', 'OFF_BALANCE_SHEET'
        )
    ),
    CONSTRAINT chk_p027_ca_amortization CHECK (
        amortization_method IS NULL OR amortization_method IN (
            'STRAIGHT_LINE', 'UNITS_OF_PRODUCTION', 'DECLINING_BALANCE', 'NONE'
        )
    ),
    CONSTRAINT chk_p027_ca_contract_dates CHECK (
        contract_end_date IS NULL OR contract_start_date IS NULL OR contract_end_date >= contract_start_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for gl_carbon_pl_allocation
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_cpl_company           ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(company_id);
CREATE INDEX idx_p027_cpl_tenant            ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(tenant_id);
CREATE INDEX idx_p027_cpl_fiscal_year       ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(fiscal_year);
CREATE INDEX idx_p027_cpl_year_quarter      ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(fiscal_year, fiscal_quarter);
CREATE INDEX idx_p027_cpl_product           ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(product_line);
CREATE INDEX idx_p027_cpl_bu                ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(business_unit);
CREATE INDEX idx_p027_cpl_cost_center       ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(cost_center);
CREATE INDEX idx_p027_cpl_cbam              ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(cbam_applicable) WHERE cbam_applicable = TRUE;
CREATE INDEX idx_p027_cpl_allocation_method ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(allocation_method);
CREATE INDEX idx_p027_cpl_emissions         ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(allocated_tco2e DESC);
CREATE INDEX idx_p027_cpl_created           ON pack027_enterprise_net_zero.gl_carbon_pl_allocation(created_at DESC);
CREATE INDEX idx_p027_cpl_metadata          ON pack027_enterprise_net_zero.gl_carbon_pl_allocation USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Indexes for gl_carbon_assets
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_ca_company            ON pack027_enterprise_net_zero.gl_carbon_assets(company_id);
CREATE INDEX idx_p027_ca_tenant             ON pack027_enterprise_net_zero.gl_carbon_assets(tenant_id);
CREATE INDEX idx_p027_ca_type               ON pack027_enterprise_net_zero.gl_carbon_assets(asset_type);
CREATE INDEX idx_p027_ca_subtype            ON pack027_enterprise_net_zero.gl_carbon_assets(asset_subtype);
CREATE INDEX idx_p027_ca_vintage            ON pack027_enterprise_net_zero.gl_carbon_assets(vintage);
CREATE INDEX idx_p027_ca_retirement         ON pack027_enterprise_net_zero.gl_carbon_assets(retirement_status);
CREATE INDEX idx_p027_ca_active             ON pack027_enterprise_net_zero.gl_carbon_assets(retirement_status) WHERE retirement_status = 'ACTIVE';
CREATE INDEX idx_p027_ca_registry           ON pack027_enterprise_net_zero.gl_carbon_assets(registry);
CREATE INDEX idx_p027_ca_issuer             ON pack027_enterprise_net_zero.gl_carbon_assets(issuer);
CREATE INDEX idx_p027_ca_accounting         ON pack027_enterprise_net_zero.gl_carbon_assets(accounting_classification);
CREATE INDEX idx_p027_ca_contract_end       ON pack027_enterprise_net_zero.gl_carbon_assets(contract_end_date);
CREATE INDEX idx_p027_ca_sdg                ON pack027_enterprise_net_zero.gl_carbon_assets USING GIN(sdg_alignment);
CREATE INDEX idx_p027_ca_created            ON pack027_enterprise_net_zero.gl_carbon_assets(created_at DESC);
CREATE INDEX idx_p027_ca_metadata           ON pack027_enterprise_net_zero.gl_carbon_assets USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_carbon_pl_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_carbon_pl_allocation
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p027_carbon_assets_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_carbon_assets
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_carbon_pl_allocation ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack027_enterprise_net_zero.gl_carbon_assets ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_cpl_tenant_isolation
    ON pack027_enterprise_net_zero.gl_carbon_pl_allocation
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_cpl_service_bypass
    ON pack027_enterprise_net_zero.gl_carbon_pl_allocation
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p027_ca_tenant_isolation
    ON pack027_enterprise_net_zero.gl_carbon_assets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_ca_service_bypass
    ON pack027_enterprise_net_zero.gl_carbon_assets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_carbon_pl_allocation TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_carbon_assets TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_carbon_pl_allocation IS
    'Carbon cost allocation to P&L line items by product/BU with EBITDA impact, carbon intensity metrics, and CBAM exposure.';
COMMENT ON TABLE pack027_enterprise_net_zero.gl_carbon_assets IS
    'Carbon asset inventory tracking ETS allowances, carbon credits, removal credits, PPAs, RECs, and other carbon instruments.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_pl_allocation.allocation_id IS 'Unique P&L carbon allocation identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_pl_allocation.carbon_cost_usd IS 'Allocated carbon cost in USD for this product/BU.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_pl_allocation.ebitda_impact_pct IS 'Carbon cost as percentage impact on product/BU EBITDA.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_pl_allocation.cbam_exposure_usd IS 'CBAM border adjustment cost exposure in USD.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_assets.asset_id IS 'Unique carbon asset identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_assets.asset_type IS 'Carbon asset type: ETS_ALLOWANCE, CARBON_CREDIT, REMOVAL_CREDIT, PPA, REC, GO, etc.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_assets.quantity IS 'Quantity of carbon assets in tCO2e or appropriate unit.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_assets.retirement_status IS 'Asset lifecycle status: ACTIVE, RETIRED, EXPIRED, TRANSFERRED, CANCELLED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_carbon_assets.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
