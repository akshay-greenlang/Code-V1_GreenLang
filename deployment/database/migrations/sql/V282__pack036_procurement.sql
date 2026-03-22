-- =============================================================================
-- V282: PACK-036 Utility Analysis Pack - Procurement & Market Analysis
-- =============================================================================
-- Pack:         PACK-036 (Utility Analysis Pack)
-- Migration:    007 of 010
-- Date:         March 2026
--
-- Tables for energy procurement contract management, wholesale market
-- price tracking, procurement strategy analysis, and green energy
-- procurement (RECs, PPAs, GOs) tracking.
--
-- Tables (4):
--   1. pack036_utility_analysis.gl_procurement_contracts
--   2. pack036_utility_analysis.gl_market_prices
--   3. pack036_utility_analysis.gl_procurement_analyses
--   4. pack036_utility_analysis.gl_green_procurements
--
-- Previous: V281__pack036_budget_forecasts.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack036_utility_analysis.gl_procurement_contracts
-- =============================================================================
-- Energy supply contracts with fixed, indexed, and blended pricing.
-- Tracks contract terms, volume commitments, green energy percentage,
-- and contract status lifecycle.

CREATE TABLE pack036_utility_analysis.gl_procurement_contracts (
    contract_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID,
    contract_reference      VARCHAR(100),
    supplier                VARCHAR(255)    NOT NULL,
    contract_type           VARCHAR(30)     NOT NULL,
    commodity               VARCHAR(30)     NOT NULL DEFAULT 'ELECTRICITY',
    start_date              DATE            NOT NULL,
    end_date                DATE            NOT NULL,
    volume_mwh              NUMERIC(14,4),
    volume_tolerance_pct    NUMERIC(6,2),
    fixed_price_per_mwh     NUMERIC(12,6),
    index_reference         VARCHAR(100),
    adder_per_mwh           NUMERIC(12,6),
    cap_price_per_mwh       NUMERIC(12,6),
    floor_price_per_mwh     NUMERIC(12,6),
    green_percentage        NUMERIC(6,2)    DEFAULT 0,
    auto_renewal            BOOLEAN         DEFAULT false,
    notice_period_days      INTEGER,
    early_termination_fee   NUMERIC(14,2),
    status                  VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_pc_contract_type CHECK (
        contract_type IN (
            'FIXED', 'INDEX_PLUS', 'BLOCK_INDEX', 'FULLY_INDEXED',
            'MANAGED', 'SPOT', 'PPA', 'VPPA', 'TOLLING', 'SLEEVED'
        )
    ),
    CONSTRAINT chk_p036_pc_commodity CHECK (
        commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'LPG', 'FUEL_OIL',
            'BIOMASS', 'HYDROGEN', 'RENEWABLE'
        )
    ),
    CONSTRAINT chk_p036_pc_dates CHECK (
        end_date > start_date
    ),
    CONSTRAINT chk_p036_pc_volume CHECK (
        volume_mwh IS NULL OR volume_mwh >= 0
    ),
    CONSTRAINT chk_p036_pc_tolerance CHECK (
        volume_tolerance_pct IS NULL OR (volume_tolerance_pct >= 0 AND volume_tolerance_pct <= 100)
    ),
    CONSTRAINT chk_p036_pc_fixed_price CHECK (
        fixed_price_per_mwh IS NULL OR fixed_price_per_mwh >= 0
    ),
    CONSTRAINT chk_p036_pc_green CHECK (
        green_percentage >= 0 AND green_percentage <= 100
    ),
    CONSTRAINT chk_p036_pc_status CHECK (
        status IN (
            'DRAFT', 'NEGOTIATING', 'ACTIVE', 'EXPIRING', 'EXPIRED',
            'TERMINATED', 'RENEWED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p036_pc_notice CHECK (
        notice_period_days IS NULL OR notice_period_days >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_pc_tenant         ON pack036_utility_analysis.gl_procurement_contracts(tenant_id);
CREATE INDEX idx_p036_pc_facility       ON pack036_utility_analysis.gl_procurement_contracts(facility_id);
CREATE INDEX idx_p036_pc_supplier       ON pack036_utility_analysis.gl_procurement_contracts(supplier);
CREATE INDEX idx_p036_pc_type           ON pack036_utility_analysis.gl_procurement_contracts(contract_type);
CREATE INDEX idx_p036_pc_commodity      ON pack036_utility_analysis.gl_procurement_contracts(commodity);
CREATE INDEX idx_p036_pc_start_date     ON pack036_utility_analysis.gl_procurement_contracts(start_date);
CREATE INDEX idx_p036_pc_end_date       ON pack036_utility_analysis.gl_procurement_contracts(end_date);
CREATE INDEX idx_p036_pc_status         ON pack036_utility_analysis.gl_procurement_contracts(status);
CREATE INDEX idx_p036_pc_created        ON pack036_utility_analysis.gl_procurement_contracts(created_at DESC);
CREATE INDEX idx_p036_pc_metadata       ON pack036_utility_analysis.gl_procurement_contracts USING GIN(metadata);

-- Composite: active contracts expiring soon for renewal alerts
CREATE INDEX idx_p036_pc_active_expiry  ON pack036_utility_analysis.gl_procurement_contracts(end_date)
    WHERE status IN ('ACTIVE', 'EXPIRING');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_pc_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_procurement_contracts
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack036_utility_analysis.gl_market_prices
-- =============================================================================
-- Wholesale energy market price data for benchmarking contract prices
-- and monitoring market conditions. Supports day-ahead, intra-day,
-- forward, and balancing market indices.

CREATE TABLE pack036_utility_analysis.gl_market_prices (
    price_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID,
    market_index            VARCHAR(100)    NOT NULL,
    price_date              DATE            NOT NULL,
    delivery_period         VARCHAR(50),
    price_per_mwh           NUMERIC(12,6)   NOT NULL,
    price_low               NUMERIC(12,6),
    price_high              NUMERIC(12,6),
    volume_mwh              NUMERIC(14,4),
    currency                VARCHAR(3)      NOT NULL DEFAULT 'EUR',
    market_type             VARCHAR(30)     DEFAULT 'DAY_AHEAD',
    source                  VARCHAR(100),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_mp_price CHECK (
        price_per_mwh >= 0
    ),
    CONSTRAINT chk_p036_mp_low CHECK (
        price_low IS NULL OR price_low >= 0
    ),
    CONSTRAINT chk_p036_mp_high CHECK (
        price_high IS NULL OR price_high >= 0
    ),
    CONSTRAINT chk_p036_mp_market_type CHECK (
        market_type IN (
            'DAY_AHEAD', 'INTRA_DAY', 'FORWARD_MONTH', 'FORWARD_QUARTER',
            'FORWARD_YEAR', 'BALANCING', 'SPOT', 'FUTURES'
        )
    ),
    CONSTRAINT uq_p036_mp_index_date UNIQUE (market_index, price_date, market_type, delivery_period)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_mp_index          ON pack036_utility_analysis.gl_market_prices(market_index);
CREATE INDEX idx_p036_mp_date           ON pack036_utility_analysis.gl_market_prices(price_date DESC);
CREATE INDEX idx_p036_mp_market_type    ON pack036_utility_analysis.gl_market_prices(market_type);
CREATE INDEX idx_p036_mp_source         ON pack036_utility_analysis.gl_market_prices(source);
CREATE INDEX idx_p036_mp_created        ON pack036_utility_analysis.gl_market_prices(created_at DESC);

-- Composite: index + date for price curve lookup
CREATE INDEX idx_p036_mp_idx_date       ON pack036_utility_analysis.gl_market_prices(market_index, price_date DESC);

-- =============================================================================
-- Table 3: pack036_utility_analysis.gl_procurement_analyses
-- =============================================================================
-- Procurement strategy analysis results comparing current costs against
-- market conditions and recommending optimal procurement strategies
-- with risk quantification (Value at Risk).

CREATE TABLE pack036_utility_analysis.gl_procurement_analyses (
    analysis_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    analysis_date           DATE            NOT NULL DEFAULT CURRENT_DATE,
    analysis_period_start   DATE,
    analysis_period_end     DATE,
    commodity               VARCHAR(30)     NOT NULL DEFAULT 'ELECTRICITY',
    current_cost_eur        NUMERIC(14,2)   NOT NULL,
    current_price_per_mwh   NUMERIC(12,6),
    market_price_per_mwh    NUMERIC(12,6),
    optimal_cost_eur        NUMERIC(14,2),
    savings_eur             NUMERIC(14,2)   NOT NULL DEFAULT 0,
    savings_pct             NUMERIC(8,4),
    strategy                VARCHAR(50)     NOT NULL,
    risk_var95_eur          NUMERIC(14,2),
    risk_cvar95_eur         NUMERIC(14,2),
    hedge_ratio_pct         NUMERIC(6,2),
    contracts_evaluated     INTEGER         DEFAULT 0,
    recommendation          TEXT,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'COMPLETED',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_pa_commodity CHECK (
        commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'LPG', 'FUEL_OIL',
            'BIOMASS', 'ALL'
        )
    ),
    CONSTRAINT chk_p036_pa_strategy CHECK (
        strategy IN (
            'FIXED_FORWARD', 'LAYERED_HEDGE', 'INDEX_FOLLOW',
            'BLOCK_INDEX', 'MANAGED_PORTFOLIO', 'SPOT_EXPOSURE',
            'GREEN_PPA', 'HYBRID', 'NO_CHANGE'
        )
    ),
    CONSTRAINT chk_p036_pa_current CHECK (
        current_cost_eur >= 0
    ),
    CONSTRAINT chk_p036_pa_hedge CHECK (
        hedge_ratio_pct IS NULL OR (hedge_ratio_pct >= 0 AND hedge_ratio_pct <= 100)
    ),
    CONSTRAINT chk_p036_pa_status CHECK (
        status IN ('IN_PROGRESS', 'COMPLETED', 'APPROVED', 'IMPLEMENTED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p036_pa_period CHECK (
        analysis_period_start IS NULL OR analysis_period_end IS NULL
        OR analysis_period_end >= analysis_period_start
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_pa_tenant         ON pack036_utility_analysis.gl_procurement_analyses(tenant_id);
CREATE INDEX idx_p036_pa_facility       ON pack036_utility_analysis.gl_procurement_analyses(facility_id);
CREATE INDEX idx_p036_pa_date           ON pack036_utility_analysis.gl_procurement_analyses(analysis_date DESC);
CREATE INDEX idx_p036_pa_commodity      ON pack036_utility_analysis.gl_procurement_analyses(commodity);
CREATE INDEX idx_p036_pa_strategy       ON pack036_utility_analysis.gl_procurement_analyses(strategy);
CREATE INDEX idx_p036_pa_savings        ON pack036_utility_analysis.gl_procurement_analyses(savings_eur DESC);
CREATE INDEX idx_p036_pa_status         ON pack036_utility_analysis.gl_procurement_analyses(status);
CREATE INDEX idx_p036_pa_created        ON pack036_utility_analysis.gl_procurement_analyses(created_at DESC);

-- Composite: facility + date for historical analysis lookup
CREATE INDEX idx_p036_pa_fac_date       ON pack036_utility_analysis.gl_procurement_analyses(facility_id, analysis_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_pa_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_procurement_analyses
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack036_utility_analysis.gl_green_procurements
-- =============================================================================
-- Green energy procurement instruments (RECs, GOs, PPAs, VPPAs) linked
-- to procurement analyses. Tracks volume, price premium, and CO2
-- avoidance for scope 2 market-based emissions reporting.

CREATE TABLE pack036_utility_analysis.gl_green_procurements (
    green_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id             UUID            REFERENCES pack036_utility_analysis.gl_procurement_analyses(analysis_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    facility_id             UUID,
    product_type            VARCHAR(30)     NOT NULL,
    instrument_name         VARCHAR(255),
    supplier                VARCHAR(255),
    technology              VARCHAR(50),
    volume_mwh              NUMERIC(14,4)   NOT NULL,
    price_premium_per_mwh   NUMERIC(12,6)   DEFAULT 0,
    total_cost_eur          NUMERIC(14,2),
    co2_avoided_tonnes      NUMERIC(14,4)   NOT NULL DEFAULT 0,
    vintage_year            INTEGER,
    certification_body      VARCHAR(100),
    certificate_id          VARCHAR(100),
    country_of_origin       CHAR(2),
    additionality           BOOLEAN,
    delivery_period_start   DATE,
    delivery_period_end     DATE,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_gp_product_type CHECK (
        product_type IN (
            'REC', 'GO', 'I_REC', 'PPA', 'VPPA', 'GREEN_TARIFF',
            'CARBON_OFFSET', 'BUNDLED', 'UNBUNDLED'
        )
    ),
    CONSTRAINT chk_p036_gp_volume CHECK (
        volume_mwh >= 0
    ),
    CONSTRAINT chk_p036_gp_premium CHECK (
        price_premium_per_mwh >= 0
    ),
    CONSTRAINT chk_p036_gp_co2 CHECK (
        co2_avoided_tonnes >= 0
    ),
    CONSTRAINT chk_p036_gp_technology CHECK (
        technology IS NULL OR technology IN (
            'SOLAR', 'WIND_ONSHORE', 'WIND_OFFSHORE', 'HYDRO',
            'BIOMASS', 'GEOTHERMAL', 'MIXED_RENEWABLE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p036_gp_status CHECK (
        status IN ('DRAFT', 'ACTIVE', 'DELIVERED', 'RETIRED', 'CANCELLED')
    ),
    CONSTRAINT chk_p036_gp_vintage CHECK (
        vintage_year IS NULL OR (vintage_year >= 2000 AND vintage_year <= 2100)
    ),
    CONSTRAINT chk_p036_gp_country CHECK (
        country_of_origin IS NULL OR LENGTH(country_of_origin) = 2
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_gp_analysis       ON pack036_utility_analysis.gl_green_procurements(analysis_id);
CREATE INDEX idx_p036_gp_tenant         ON pack036_utility_analysis.gl_green_procurements(tenant_id);
CREATE INDEX idx_p036_gp_facility       ON pack036_utility_analysis.gl_green_procurements(facility_id);
CREATE INDEX idx_p036_gp_product        ON pack036_utility_analysis.gl_green_procurements(product_type);
CREATE INDEX idx_p036_gp_technology     ON pack036_utility_analysis.gl_green_procurements(technology);
CREATE INDEX idx_p036_gp_vintage        ON pack036_utility_analysis.gl_green_procurements(vintage_year);
CREATE INDEX idx_p036_gp_status         ON pack036_utility_analysis.gl_green_procurements(status);
CREATE INDEX idx_p036_gp_co2            ON pack036_utility_analysis.gl_green_procurements(co2_avoided_tonnes DESC);
CREATE INDEX idx_p036_gp_created        ON pack036_utility_analysis.gl_green_procurements(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_gp_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_green_procurements
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack036_utility_analysis.gl_procurement_contracts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_market_prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_procurement_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_green_procurements ENABLE ROW LEVEL SECURITY;

CREATE POLICY p036_pc_tenant_isolation
    ON pack036_utility_analysis.gl_procurement_contracts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_pc_service_bypass
    ON pack036_utility_analysis.gl_procurement_contracts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Market prices may not have tenant_id (shared reference data)
CREATE POLICY p036_mp_public_read
    ON pack036_utility_analysis.gl_market_prices
    USING (TRUE);
CREATE POLICY p036_mp_service_bypass
    ON pack036_utility_analysis.gl_market_prices
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_pa_tenant_isolation
    ON pack036_utility_analysis.gl_procurement_analyses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_pa_service_bypass
    ON pack036_utility_analysis.gl_procurement_analyses
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_gp_tenant_isolation
    ON pack036_utility_analysis.gl_green_procurements
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_gp_service_bypass
    ON pack036_utility_analysis.gl_green_procurements
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_procurement_contracts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_market_prices TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_procurement_analyses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_green_procurements TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack036_utility_analysis.gl_procurement_contracts IS
    'Energy supply contracts with fixed, indexed, and blended pricing including volume commitments and green percentage.';

COMMENT ON TABLE pack036_utility_analysis.gl_market_prices IS
    'Wholesale energy market price data for benchmarking contract prices across day-ahead, forward, and balancing markets.';

COMMENT ON TABLE pack036_utility_analysis.gl_procurement_analyses IS
    'Procurement strategy analysis results with current vs optimal cost, savings, and risk quantification (VaR).';

COMMENT ON TABLE pack036_utility_analysis.gl_green_procurements IS
    'Green energy procurement instruments (RECs, GOs, PPAs) with volume, price premium, and CO2 avoidance for scope 2 reporting.';

COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_contracts.contract_id IS
    'Unique identifier for the procurement contract.';
COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_contracts.contract_type IS
    'Contract type: FIXED, INDEX_PLUS, BLOCK_INDEX, FULLY_INDEXED, MANAGED, SPOT, PPA, VPPA, TOLLING, SLEEVED.';
COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_contracts.volume_mwh IS
    'Contracted annual volume in MWh.';
COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_contracts.index_reference IS
    'Market index reference for indexed contracts (e.g., EPEX DE Base, TTF Month-Ahead).';
COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_contracts.adder_per_mwh IS
    'Adder/spread per MWh above the index reference price.';
COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_contracts.green_percentage IS
    'Percentage of contracted volume from renewable/green sources.';
COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_contracts.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_market_prices.market_index IS
    'Market index identifier (e.g., EPEX_SPOT_DE, TTF, NBP, OMIE).';
COMMENT ON COLUMN pack036_utility_analysis.gl_market_prices.price_per_mwh IS
    'Settlement or closing price per MWh.';
COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_analyses.strategy IS
    'Recommended procurement strategy: FIXED_FORWARD, LAYERED_HEDGE, INDEX_FOLLOW, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_analyses.risk_var95_eur IS
    'Value at Risk at 95% confidence level in EUR - maximum expected loss.';
COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_analyses.hedge_ratio_pct IS
    'Recommended hedge ratio (fixed vs index exposure) as percentage.';
COMMENT ON COLUMN pack036_utility_analysis.gl_procurement_analyses.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_green_procurements.product_type IS
    'Green product type: REC, GO (Guarantee of Origin), I_REC, PPA, VPPA, GREEN_TARIFF, CARBON_OFFSET, BUNDLED, UNBUNDLED.';
COMMENT ON COLUMN pack036_utility_analysis.gl_green_procurements.co2_avoided_tonnes IS
    'CO2 emissions avoided in tonnes for scope 2 market-based reporting.';
COMMENT ON COLUMN pack036_utility_analysis.gl_green_procurements.additionality IS
    'Whether the green procurement meets additionality criteria (new renewable capacity).';
COMMENT ON COLUMN pack036_utility_analysis.gl_green_procurements.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
